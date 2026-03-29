//! JPEG decoder implementation.
//!
//! This module provides the main decoder interface for reading JPEG images.
//!
//! # Quick Start
//!
//! ```ignore
//! use zenjpeg::decode::Decoder;
//!
//! let result = Decoder::new().decode(&jpeg_data, enough::Unstoppable)?;
//! let pixels: &[u8] = result.pixels_u8().unwrap();
//! ```
//!
//! # ICC Profile Support
//!
//! The decoder can extract and apply embedded ICC profiles, including XYB profiles
//! used by jpegli. ICC profile support requires enabling `moxcms` feature.
//!
//! ```ignore
//! use zenjpeg::decode::Decoder;
//!
//! use zenjpeg::color::icc::TargetColorSpace;
//! let result = Decoder::new()
//!     .correct_color(Some(TargetColorSpace::Srgb))
//!     .decode(&jpeg_data, enough::Unstoppable)?;
//! ```

// IDCT modules (decoder-only)
#[doc(hidden)]
pub mod idct;
#[doc(hidden)]
pub mod idct_int;

mod config;
mod depth;
mod extras;
#[cfg(feature = "parallel")]
mod fused_parallel;
mod image;
mod parser;
mod pipeline;
mod pool;
mod request;
mod row_slice;
#[cfg(feature = "parallel")]
pub(crate) mod rst_scan;
mod scanline;
mod upsample;

#[cfg(feature = "ultrahdr")]
mod ultrahdr_reader;

// These types are public API for coefficient analysis and decode results
#[allow(unused_imports)]
pub use image::{
    CoefficientComparison, ComponentCoefficients, DecodedCoefficients, DecodedImage,
    DecodedImageF32, DecodedYCbCr,
};

// New unified types
#[allow(unused_imports)]
pub use config::{
    CropRegion, DecodeConfig, DecodeInfo, DecodeResult, DecodedPixels, GainMapHandling,
    GainMapResult, OutputTarget, OwnedDecodedPixels, ParallelStrategy,
};
/// Backward-compatible alias for [`DecodeConfig`].
pub type Decoder = DecodeConfig;
use parser::JpegParser;

pub use pool::DecodePool;
pub use request::DecodeRequest;
pub use row_slice::{RowSlice, RowSliceF32};
pub use scanline::{ScanlineInfo, ScanlineReader};

// UltraHDR streaming reader
#[cfg(feature = "ultrahdr")]
#[allow(unused_imports)] // Re-exports for public API
pub use ultrahdr_reader::{GainMapMemory, UltraHdrMode, UltraHdrReader, UltraHdrReaderConfig};

// Re-export extras types for public API
#[allow(unused_imports)]
pub use extras::{
    AdobeColorTransform, AdobeInfo, Confidence, DecodedExtras, DensityUnits, DqtTable,
    EncoderFamily, IccPreserve, JfifInfo, JpegProbe, MpfDirectory, MpfEntry, MpfImageType,
    MpfImageTypeExt, PreserveConfig, PreservedMpfImage, PreservedSegment, QualityEstimate,
    QualityScale, SegmentType, StandardProfile,
};

// Re-export depth map types for public API
#[allow(unused_imports)]
pub use depth::{
    DepthMapData, DepthSource, GDepthFormat, GDepthMeasureType, GDepthMetadata, GDepthUnits,
};

// Re-export types used in public struct fields so users can access them
#[allow(unused_imports)]
pub use crate::types::{ColorSpace, Dimensions, JpegMode, PixelFormat};
use crate::types::{Component, Subsampling};

// Re-export Stop trait for cancellation support
pub use enough::Stop;
use enough::Unstoppable;

use crate::error::{Error, Result};
use crate::foundation::consts::MAX_COMPONENTS;
use imgref::ImgRefMut;

/// Result of wave parallel eligibility check.
#[cfg(feature = "parallel")]
#[allow(clippy::large_enum_variant)]
enum WaveResult<'a> {
    /// Wave-only reader (4:2:0 + box filter): all output paths use wave decode.
    WaveOnly(ScanlineReader<'a>),
    /// Wave state only: caller creates sequential reader and attaches this for planar i16.
    WaveState(fused_parallel::WaveParallelState),
}

/// Compute subsampling mode from component sampling factors.
pub(crate) fn compute_subsampling(
    components: &[Component; MAX_COMPONENTS],
    num_components: u8,
) -> Subsampling {
    if num_components == 1 {
        return Subsampling::S444; // Grayscale
    }

    // Find max sampling factors
    let max_h = components[..num_components as usize]
        .iter()
        .map(|c| c.h_samp_factor)
        .max()
        .unwrap_or(1);
    let max_v = components[..num_components as usize]
        .iter()
        .map(|c| c.v_samp_factor)
        .max()
        .unwrap_or(1);

    subsampling_from_max(max_h, max_v, false)
}

/// Convert max sampling factors to Subsampling enum.
fn subsampling_from_max(max_h: u8, max_v: u8, is_grayscale: bool) -> Subsampling {
    if is_grayscale {
        return Subsampling::S444;
    }
    match (max_h, max_v) {
        (1, 1) => Subsampling::S444,
        (2, 1) => Subsampling::S422,
        (2, 2) => Subsampling::S420,
        (1, 2) => Subsampling::S440,
        // For other patterns, approximate as 4:2:0
        _ => Subsampling::S420,
    }
}

// Re-export config types (defined in config.rs, public API preserved)
pub use config::{ChromaUpsampling, DeblockMode, DecodeWarning, IdctMethod, JpegInfo, Strictness};

#[cfg(feature = "moxcms")]
use crate::color::icc::apply_icc_transform_f32;
#[cfg(feature = "moxcms")]
use crate::color::icc::{TargetColorSpace, apply_icc_transform};

impl DecodeConfig {
    /// Creates a new decoder configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a per-job decode request binding this config with JPEG data.
    ///
    /// The returned [`DecodeRequest`] can optionally attach a [`DecodePool`]
    /// for adaptive threading and a cancellation token before decoding.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zenjpeg::decode::{Decoder, DecodePool};
    ///
    /// let decoder = Decoder::new().chroma_upsampling(ChromaUpsampling::NearestNeighbor);
    /// let pool = DecodePool::new();
    ///
    /// // With pool (server)
    /// let result = decoder.request(&jpeg_data)
    ///     .pool(&pool)
    ///     .decode()?;
    ///
    /// // Without pool (standalone)
    /// let result = decoder.request(&jpeg_data).decode()?;
    /// ```
    #[must_use]
    pub fn request<'a>(&'a self, data: &'a [u8]) -> request::DecodeRequest<'a> {
        request::DecodeRequest::new(self, data)
    }

    /// Sets the output pixel format.
    #[must_use]
    pub fn output_format(mut self, format: PixelFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Sets the chroma upsampling method.
    ///
    /// Controls how subsampled chroma channels (4:2:0, 4:2:2, 4:4:0) are
    /// upsampled to match luma resolution.
    ///
    /// - [`ChromaUpsampling::Triangle`] (default): fused 2D filter, matches libjpeg-turbo/mozjpeg
    /// - [`ChromaUpsampling::NearestNeighbor`]: fastest, lowest quality
    #[must_use]
    pub fn chroma_upsampling(mut self, method: ChromaUpsampling) -> Self {
        self.chroma_upsampling = method;
        self
    }

    /// Enables or disables fancy (triangle filter) upsampling.
    ///
    /// Sets the integer IDCT algorithm.
    ///
    /// Controls which fixed-point IDCT is used during decoding. Different
    /// algorithms produce slightly different pixel values (max 2-3 levels).
    ///
    /// - [`IdctMethod::Jpegli`] (default): 12-bit fixed-point, matches jpegli
    /// - [`IdctMethod::Libjpeg`]: 13-bit Loeffler, matches libjpeg-turbo/mozjpeg
    ///
    /// The default IDCT is `Jpegli` regardless of upsampling mode.
    /// Set `Libjpeg` for pixel-exact mozjpeg matching (adds ~37% overhead).
    #[must_use]
    pub fn idct_method(mut self, method: IdctMethod) -> Self {
        self.idct_method = Some(method);
        self
    }

    /// Returns the effective IDCT method, considering both explicit setting
    /// and chroma upsampling mode.
    ///
    /// - Explicit `idct_method()` always wins
    /// - Default is `Jpegli` for all upsampling modes
    ///
    /// The default `Triangle` upsampling matches libjpeg-turbo/mozjpeg within
    /// max_diff ≤ 3. For pixel-exact matching (max_diff ≤ 2), also set
    /// `.idct_method(IdctMethod::Libjpeg)` — adds ~37% decode overhead.
    pub(crate) fn effective_idct_method(&self) -> IdctMethod {
        self.idct_method.unwrap_or(IdctMethod::Jpegli)
    }

    /// Enable post-decode deblocking to reduce JPEG block artifacts.
    ///
    /// Deblocking improves visual quality by smoothing 8x8 block boundaries.
    /// The effect is strongest at low quality levels (Q5-Q50) where blocking
    /// artifacts are most visible.
    ///
    /// # Supported paths
    ///
    /// | Mode | `decode()` | `scanline_reader()` |
    /// |------|-----------|-------------------|
    /// | `Off` | no-op | no-op (zero overhead) |
    /// | `Boundary4Tap` | f32 planes | i16 planes (streaming) |
    /// | `Knusperli` | DCT-domain | fallback to `decode()` internally |
    /// | `Auto` | knusperli at low Q, boundary otherwise | same (falls back when needed) |
    ///
    /// `Knusperli` and `Auto` (when it picks Knusperli) work in `scanline_reader()`
    /// by transparently falling back to coefficient-based decoding. Output is
    /// consistent regardless of which decode path you use; only memory behavior
    /// differs (fallback buffers the full image).
    ///
    /// # Performance
    ///
    /// Boundary 4-tap adds ~5-15% decode time. Knusperli adds ~20-40% due to extra
    /// IDCT work. When `Off` (default), zero overhead in both paths.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::{Decoder, DeblockMode};
    ///
    /// let result = Decoder::new()
    ///     .deblock(DeblockMode::Auto)
    ///     .decode(&jpeg_data, enough::Unstoppable)?;
    /// ```
    #[must_use]
    pub fn deblock(mut self, mode: DeblockMode) -> Self {
        self.deblock_mode = mode;
        self
    }

    /// Convert embedded ICC color profile to a target color space during decode.
    ///
    /// When set to `Some(target)`, the decoder applies the embedded ICC profile
    /// to convert pixel data to the specified color space. When `None` (default),
    /// no color conversion is performed — pixels are returned in the JPEG's
    /// native color space.
    ///
    /// Requires the `moxcms` feature. Without it, this method is not available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::color::icc::TargetColorSpace;
    ///
    /// let img = Decoder::new()
    ///     .correct_color(Some(TargetColorSpace::Srgb))
    ///     .decode(&jpeg, stop)?;
    /// ```
    #[cfg(feature = "moxcms")]
    #[must_use]
    pub fn correct_color(mut self, target: Option<TargetColorSpace>) -> Self {
        self.correct_color = target;
        self
    }

    /// Sets the maximum number of pixels allowed (for DoS protection).
    ///
    /// Default is 100 megapixels. Set to 0 for unlimited.
    #[must_use]
    pub fn max_pixels(mut self, pixels: u64) -> Self {
        self.max_pixels = pixels;
        self
    }

    /// Sets the maximum memory allowed for allocations during decoding.
    ///
    /// Default is 512 MB. Set to `usize::MAX` for unlimited.
    /// This prevents memory exhaustion attacks from malicious images.
    #[must_use]
    pub fn max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = bytes;
        self
    }

    /// Sets resource limits from a [`Limits`] struct.
    ///
    /// This applies `max_pixels` and `max_memory` from the `Limits` struct.
    /// `None` values in `Limits` are treated as unlimited (0).
    #[must_use]
    pub fn limits(mut self, limits: crate::types::Limits) -> Self {
        if let Some(pixels) = limits.max_pixels {
            self.max_pixels = pixels;
        }
        if let Some(memory) = limits.max_memory {
            self.max_memory = memory;
        }
        self
    }

    /// Configure what metadata and secondary images to preserve during decode.
    ///
    /// By default, most metadata (EXIF, XMP, ICC, IPTC) and gain maps are preserved.
    /// Thumbnails and other MPF images are dropped by default.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zenjpeg::decode::{Decoder, PreserveConfig};
    ///
    /// // Preserve nothing (minimal memory)
    /// let decoder = Decoder::new().preserve(PreserveConfig::none());
    ///
    /// // Preserve everything
    /// let decoder = Decoder::new().preserve(PreserveConfig::all());
    ///
    /// // Custom: keep gain maps under 500KB only
    /// let config = PreserveConfig::default()
    ///     .mpf_filter(|_idx, typ, size| {
    ///         typ.is_gainmap() && size < 500_000
    ///     });
    /// let decoder = Decoder::new().preserve(config);
    /// ```
    #[must_use]
    pub fn preserve(mut self, config: PreserveConfig) -> Self {
        self.preserve = config;
        self
    }

    /// Convenience: preserve no metadata (minimal memory, pixels only).
    #[must_use]
    pub fn preserve_no_metadata(self) -> Self {
        self.preserve(PreserveConfig::none())
    }

    /// Convenience: preserve all metadata (EXIF, XMP, ICC, IPTC, comments, MPF).
    #[must_use]
    pub fn preserve_all_metadata(self) -> Self {
        self.preserve(PreserveConfig::all())
    }

    /// Sets the strictness level for error handling.
    ///
    /// - [`Strictness::Strict`]: Fail on any spec violation or truncation
    /// - [`Strictness::Balanced`]: Reject violations, recover from truncation (default)
    /// - [`Strictness::Lenient`]: Recover from all errors when possible
    /// - [`Strictness::Permissive`]: Maximum libjpeg-turbo compatibility
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zenjpeg::decode::{Decoder, Strictness};
    ///
    /// // Strict mode for validation
    /// let decoder = Decoder::new().strictness(Strictness::Strict);
    ///
    /// // Balanced mode (default) for production
    /// let decoder = Decoder::new().strictness(Strictness::Balanced);
    ///
    /// // Lenient mode for corrupt file recovery
    /// let decoder = Decoder::new().strictness(Strictness::Lenient);
    ///
    /// // Permissive mode for maximum compatibility
    /// let decoder = Decoder::new().strictness(Strictness::Permissive);
    /// ```
    #[must_use]
    pub fn strictness(mut self, strictness: Strictness) -> Self {
        self.strictness = strictness;
        self
    }

    /// Convenience: use strict mode (fail on any recoverable error).
    #[must_use]
    pub fn strict(self) -> Self {
        self.strictness(Strictness::Strict)
    }

    /// Convenience: use lenient mode (recover from all errors).
    #[must_use]
    pub fn lenient(self) -> Self {
        self.strictness(Strictness::Lenient)
    }

    /// Convenience: use permissive mode (maximum libjpeg-turbo compatibility).
    #[must_use]
    pub fn permissive(self) -> Self {
        self.strictness(Strictness::Permissive)
    }

    /// Sets the output target controlling precision, transfer function, and IDCT variant.
    ///
    /// See [`OutputTarget`] for available options.
    #[must_use]
    pub fn output_target(mut self, target: OutputTarget) -> Self {
        self.output_target = target;
        self
    }

    /// Sets how UltraHDR gain maps are handled.
    ///
    /// See [`GainMapHandling`] for available options.
    #[must_use]
    pub fn gain_map(mut self, handling: GainMapHandling) -> Self {
        self.gain_map = handling;
        self
    }

    /// Controls automatic EXIF orientation correction during decode.
    ///
    /// **Enabled by default.** When enabled, the decoder reads the EXIF
    /// orientation tag and applies the corresponding lossless transform in
    /// DCT-coefficient space before IDCT. The output pixels will have the
    /// correct visual orientation.
    ///
    /// If the image has no EXIF data or orientation is 1 (normal), this has
    /// no effect on the decode path.
    ///
    /// Can be combined with [`transform()`](Self::transform) — the EXIF
    /// correction is applied first, then the explicit transform.
    ///
    /// Pass `false` to disable and get raw pixel orientation (e.g., for
    /// lossless re-encoding where you want to preserve the original
    /// orientation tag).
    #[must_use]
    pub fn auto_orient(mut self, enable: bool) -> Self {
        self.auto_orient = enable;
        self
    }

    /// Sets an explicit lossless transform to apply during decode.
    ///
    /// The transform is applied in DCT-coefficient space before IDCT,
    /// so there is no quality loss from the transform itself. Only one
    /// full entropy decode pass is performed.
    ///
    /// When combined with [`auto_orient(true)`](Self::auto_orient), the
    /// EXIF orientation correction is applied first, then this transform.
    #[must_use]
    pub fn transform(mut self, transform: crate::lossless::LosslessTransform) -> Self {
        self.decode_transform = Some(transform);
        self
    }

    /// Apply optimal Laplacian dequantization biases (Price & Rabbani 2000).
    ///
    /// Convenience method equivalent to
    /// `.output_target(OutputTarget::SrgbF32Precise)`.
    ///
    /// When enabled, the decoder computes per-coefficient biases from DCT
    /// coefficient statistics and applies them during dequantization. This
    /// reduces reconstruction error compared to the default midpoint
    /// reconstruction, producing measurably higher quality output.
    ///
    /// Tradeoff: bypasses the fast integer IDCT path, using f32 dequantization
    /// and IDCT instead. Expect ~1.3-2x slower decoding.
    #[must_use]
    pub fn dequant_bias(mut self, enable: bool) -> Self {
        if enable {
            self.output_target = OutputTarget::SrgbF32Precise;
        } else if self.output_target.is_precise() {
            self.output_target = OutputTarget::Srgb8;
        }
        self
    }

    /// Sets a crop region for decode.
    ///
    /// Only the specified sub-region of the image will be output.
    /// Entropy decoding still runs for all MCUs (required by the DC predictor
    /// chain), but IDCT and color conversion are skipped for MCU rows outside
    /// the crop, providing significant speedup for small crops of large images.
    ///
    /// The crop is applied in output space (after `auto_orient` / `transform`).
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::decode::{Decoder, CropRegion};
    ///
    /// // Crop a 200x200 region starting at (100, 100)
    /// let result = Decoder::new()
    ///     .crop(CropRegion::pixels(100, 100, 200, 200))
    ///     .decode(&jpeg_data, enough::Unstoppable)?;
    ///
    /// assert_eq!(result.width(), 200);
    /// assert_eq!(result.height(), 200);
    /// ```
    #[must_use]
    pub fn crop(mut self, region: config::CropRegion) -> Self {
        self.crop_region = Some(region);
        self
    }

    /// Sets the number of threads for parallel decode paths.
    ///
    /// - `0` (default): Auto — uses rayon's global thread pool. Fused parallel
    ///   decode activates when DRI is present with enough segments.
    /// - `1`: Force sequential — disables all parallel decode paths.
    ///
    /// Values > 1 are reserved for future use and currently behave like `0`.
    #[must_use]
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Sets the parallel decode strategy.
    ///
    /// Controls how restart segments are mapped to rayon tasks during parallel
    /// decode. Only affects baseline images with MCU-row-aligned DRI when
    /// compiled with `--features parallel`.
    ///
    /// See [`ParallelStrategy`](config::ParallelStrategy) for available options.
    #[must_use]
    pub fn parallel_strategy(mut self, strategy: config::ParallelStrategy) -> Self {
        self.parallel_strategy = strategy;
        self
    }

    /// Reads JPEG info without decoding.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        // Preserve metadata segments for extraction without full decode
        let preserve = PreserveConfig::none()
            .jfif(true)
            .exif(true)
            .xmp(true)
            .icc(IccPreserve::All);

        let mut parser =
            JpegParser::with_strictness(data, self.max_pixels, Some(&preserve), self.strictness)?;
        parser.read_header()?;
        Ok(parser.info())
    }

    /// Estimates peak memory usage for decoding an image of given dimensions.
    ///
    /// This is useful for checking if an image can be decoded within memory limits
    /// before attempting to decode it. The estimate includes:
    /// - Strip buffers for one MCU row (Y, Cb, Cr at 2 bytes per pixel)
    /// - Output RGB buffer (3 bytes per pixel)
    ///
    /// For streaming decode (baseline 4:4:4), no coefficient storage is needed.
    /// For progressive or subsampled images, add ~128 bytes per DCT block for coefficients.
    ///
    /// # Example
    /// ```
    /// use zenjpeg::decode::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let estimated = decoder.estimate_memory_usage(4096, 4096);
    /// println!("Estimated peak memory: {} MB", estimated / 1024 / 1024);
    /// ```
    #[must_use]
    pub fn estimate_memory_usage(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;

        // MCU width for strip buffers (padded to 8)
        let mcu_cols = (w + 7) / 8;
        let strip_width = mcu_cols * 8;
        let strip_height = 8; // One MCU row

        // Strip buffers: Y, Cb, Cr each at i16 (2 bytes per pixel)
        let strip_size = strip_width * strip_height;
        let strip_total = strip_size * 2 * 3; // 3 components, 2 bytes each

        // Output RGB buffer: 3 bytes per pixel
        let rgb_size = w * h * 3;

        // Total for streaming decode (baseline 4:4:4)
        let streaming_total = strip_total + rgb_size;

        // For non-streaming paths (progressive, subsampled), coefficient storage is needed
        // ~130 bytes per block (64 i16 coefficients + u8 coeff count + alignment)
        // 3 components, (w/8) * (h/8) blocks per component
        let blocks_per_component = mcu_cols * ((h + 7) / 8);
        let coeff_storage = blocks_per_component * 130 * 3;

        // Return worst case (non-streaming)
        streaming_total.max(coeff_storage + rgb_size)
    }

    /// Creates a pull-based scanline reader for streaming decode.
    ///
    /// This allows reading the image row by row without loading the entire
    /// image into memory. Only supports baseline JPEGs with 4:4:4 subsampling.
    ///
    /// # Example
    /// ```ignore
    /// use zenjpeg::{Decoder, ImgRefMut};
    ///
    /// let mut reader = Decoder::new().scanline_reader(&jpeg_data)?;
    /// let width = reader.width() as usize;
    /// let height = reader.height() as usize;
    ///
    /// let mut pixels = vec![0u8; width * height * 3];
    /// let mut rows_read = 0;
    /// while rows_read < height {
    ///     let remaining = height - rows_read;
    ///     let slice = &mut pixels[rows_read * width * 3..];
    ///     let output = ImgRefMut::new(slice, width * 3, remaining);
    ///     rows_read += reader.read_rows_rgb8(output)?;
    /// }
    /// ```
    pub fn scanline_reader<'a>(&self, data: &'a [u8]) -> Result<ScanlineReader<'a>> {
        // Check if we need a transform — if so, use coefficient-based path
        let effective_transform = self.compute_effective_transform_from_data(data);
        if effective_transform != crate::lossless::LosslessTransform::None {
            return self.scanline_reader_with_transform(data, effective_transform);
        }

        let mut parser = JpegParser::with_strictness(data, self.max_pixels, None, self.strictness)?;
        parser.read_header()?;

        // Knusperli needs full coefficient access. When requested (explicitly or
        // via Auto at low Q), fall back to decode() + buffered scanline reader
        // transparently — the caller gets correct deblocked output without
        // streaming memory savings.
        let needs_coefficient_deblock = match self.deblock_mode {
            DeblockMode::Knusperli => true,
            DeblockMode::Auto => {
                let dc_quant = parser
                    .quant_tables
                    .iter()
                    .find_map(|qt| qt.as_ref().map(|t| t[0]))
                    .unwrap_or(0);
                dc_quant >= 27
            }
            _ => false,
        };
        if needs_coefficient_deblock {
            return self.scanline_reader_deblock_fallback(data);
        }

        // DNL mode (height=0 in SOF) not supported - scanline reader needs dimensions upfront
        if parser.height == 0 {
            return Err(Error::unsupported_feature(
                "scanline reader does not support DNL mode (height=0 in SOF)",
            ));
        }

        // 12-bit precision (Extended Sequential) not yet fully supported
        // The level shift and output scaling differ from 8-bit
        if parser.precision != 8 {
            return Err(Error::unsupported_feature(
                "12-bit precision JPEG (Extended Sequential) is not yet supported. \
                 Only 8-bit precision is currently implemented.",
            ));
        }

        // Support grayscale (1), color (3), and CMYK/YCCK (4) images
        if parser.num_components != 1 && parser.num_components != 3 && parser.num_components != 4 {
            return Err(Error::unsupported_feature(
                "scanline reader requires 1, 3, or 4 component image",
            ));
        }

        let is_grayscale = parser.num_components == 1;
        let is_cmyk = parser.num_components == 4;
        let is_xyb = parser.info().is_xyb;

        // Compute MCU height for crop resolution
        let max_v_samp = parser.components[..parser.num_components as usize]
            .iter()
            .map(|c| c.v_samp_factor as usize)
            .max()
            .unwrap_or(1);
        let mcu_height = max_v_samp * 8;

        // Use buffered mode for:
        // - Progressive JPEGs (format requires all scans before final coefficients)
        // - Arithmetic-coded JPEGs (streaming entropy decoder is Huffman-only for now)
        // - CMYK (4-component streaming not implemented yet)
        let needs_buffered = matches!(
            parser.mode,
            JpegMode::Progressive
                | JpegMode::ArithmeticSequential
                | JpegMode::ArithmeticProgressive
        ) || is_cmyk;

        if needs_buffered {
            let width = parser.width;
            let height = parser.height;
            let num_components = parser.num_components;

            // Fully decode the image (scanline reader doesn't support cancellation)
            parser.chroma_upsampling = self.chroma_upsampling;
            parser.idct_method = self.effective_idct_method();
            parser.decode(&Unstoppable)?;

            // Compute subsampling from sampling factors
            let subsampling = compute_subsampling(&parser.components, num_components);

            // Convert to pixels (RGB for color, grayscale for 1-component)
            let output_format = if is_grayscale {
                PixelFormat::Gray
            } else {
                PixelFormat::Rgb
            };
            let pixels = parser.to_pixels(
                output_format,
                is_xyb,
                self.chroma_upsampling,
                OutputTarget::Srgb8,
                &Unstoppable,
            )?;

            let mut reader = ScanlineReader::new_buffered(
                data,
                width,
                height,
                num_components,
                subsampling,
                pixels,
                is_xyb,
            );
            self.apply_crop(&mut reader, width, height, mcu_height)?;
            return Ok(reader);
        }

        // Baseline: use streaming mode
        // Check for high sampling factors (>2x2) which need buffered mode
        let max_h = parser.components[..parser.num_components as usize]
            .iter()
            .map(|c| c.h_samp_factor)
            .max()
            .unwrap_or(1);

        // Detect exotic sampling that the strip processor cannot handle:
        // - Factors > 2 in any dimension
        // - Cb and Cr have different sampling factors (asymmetric chroma)
        // - Chroma factors exceed luma (inverted subsampling)
        // These are routed through the buffered decode path instead.
        let needs_buffered_sampling = if is_grayscale || parser.num_components < 3 {
            max_h > 2 || max_v_samp > 2
        } else {
            let comps = &parser.components[..parser.num_components as usize];
            let cb_h = comps[1].h_samp_factor;
            let cb_v = comps[1].v_samp_factor;
            let cr_h = comps[2].h_samp_factor;
            let cr_v = comps[2].v_samp_factor;
            max_h > 2
                || max_v_samp > 2
                || cb_h != cr_h
                || cb_v != cr_v
                || cb_h > comps[0].h_samp_factor
                || cb_v > comps[0].v_samp_factor
        };

        if needs_buffered_sampling {
            let width = parser.width;
            let height = parser.height;
            let num_components = parser.num_components;
            let subsampling = subsampling_from_max(max_h, max_v_samp as u8, is_grayscale);

            parser.chroma_upsampling = self.chroma_upsampling;
            parser.idct_method = self.effective_idct_method();
            parser.decode(&Unstoppable)?;

            let output_format = if is_grayscale {
                PixelFormat::Gray
            } else {
                PixelFormat::Rgb
            };
            let pixels = parser.to_pixels(
                output_format,
                is_xyb,
                self.chroma_upsampling,
                OutputTarget::Srgb8,
                &Unstoppable,
            )?;

            let mut reader = ScanlineReader::new_buffered(
                data,
                width,
                height,
                num_components,
                subsampling,
                pixels,
                is_xyb,
            );
            self.apply_crop(&mut reader, width, height, mcu_height)?;
            return Ok(reader);
        }

        // Extract scan data and construct scanline reader
        let scan_data = parser.into_scan_data(is_grayscale)?;
        let width = scan_data.width;
        let height = scan_data.height;

        // Try wave-parallel mode for images with DRI restart markers
        #[cfg(feature = "parallel")]
        if self.num_threads != 1
            && let Some(result) = self.try_wave_parallel(&scan_data, mcu_height)?
        {
            match result {
                WaveResult::WaveOnly(reader) => return Ok(reader),
                WaveResult::WaveState(wave_state) => {
                    // Sequential reader with wave state for planar i16 only
                    let mut reader = ScanlineReader::from_scan_data(
                        scan_data,
                        self.chroma_upsampling,
                        self.effective_idct_method(),
                        self.output_target,
                        self.deblock_mode,
                    )?;
                    reader.attach_wave_state(wave_state);
                    self.apply_crop(&mut reader, width, height, mcu_height)?;
                    return Ok(reader);
                }
            }
        }

        let mut reader = ScanlineReader::from_scan_data(
            scan_data,
            self.chroma_upsampling,
            self.effective_idct_method(),
            self.output_target,
            self.deblock_mode,
        )?;
        self.apply_crop(&mut reader, width, height, mcu_height)?;
        Ok(reader)
    }

    /// Like [`scanline_reader`](Self::scanline_reader), but accepts a `Cow` for
    /// the input data. When `Cow::Owned`, the returned reader owns the JPEG data
    /// and can be effectively `'static` (no external borrow).
    ///
    /// `Cow::Borrowed` delegates to [`scanline_reader`](Self::scanline_reader).
    pub fn scanline_reader_cow<'a>(
        &self,
        data: alloc::borrow::Cow<'a, [u8]>,
    ) -> Result<ScanlineReader<'a>> {
        use alloc::borrow::Cow;

        match data {
            Cow::Borrowed(slice) => self.scanline_reader(slice),
            Cow::Owned(vec) => self.scanline_reader_owned(vec),
        }
    }

    /// Internal: create a scanline reader that owns its JPEG data.
    ///
    /// The parser borrows `&vec` temporarily for header parsing, then the
    /// reader stores `Cow::Owned(vec)` for ongoing entropy decoding.
    fn scanline_reader_owned<'a>(&self, vec: alloc::vec::Vec<u8>) -> Result<ScanlineReader<'a>> {
        use crate::types::JpegMode;
        use alloc::borrow::Cow;

        // Check if we need a transform
        let effective_transform = self.compute_effective_transform_from_data(&vec);
        if effective_transform != crate::lossless::LosslessTransform::None {
            return self.scanline_reader_with_transform_owned(vec, effective_transform);
        }

        // Phase 1: parse the JPEG data, extract everything we need.
        // The parser borrows &vec; we must drop all parser-derived borrows
        // before moving vec into Cow::Owned.
        #[allow(clippy::large_enum_variant)]
        enum ParseResult {
            Buffered {
                width: u32,
                height: u32,
                num_components: u8,
                subsampling: Subsampling,
                pixels: alloc::vec::Vec<u8>,
                is_xyb: bool,
                mcu_height: usize,
            },
            Streaming {
                scan_data: parser::ParsedScanData<'static>,
                width: u32,
                height: u32,
                mcu_height: usize,
                #[cfg(feature = "parallel")]
                wave_info: Option<(fused_parallel::WaveParallelState, bool)>,
            },
        }

        // Check if deblock mode needs coefficient path (Knusperli or Auto at low Q)
        {
            let needs_coefficient_deblock = match self.deblock_mode {
                DeblockMode::Knusperli => true,
                DeblockMode::Auto => {
                    let mut peek =
                        JpegParser::with_strictness(&vec, self.max_pixels, None, self.strictness)?;
                    peek.read_header()?;
                    let dc_quant = peek
                        .quant_tables
                        .iter()
                        .find_map(|qt| qt.as_ref().map(|t| t[0]))
                        .unwrap_or(0);
                    dc_quant >= 27
                }
                _ => false,
            };
            if needs_coefficient_deblock {
                return self.scanline_reader_deblock_fallback_owned(vec);
            }
        }

        let parse_result = {
            let mut parser =
                JpegParser::with_strictness(&vec, self.max_pixels, None, self.strictness)?;
            parser.read_header()?;

            if parser.height == 0 {
                return Err(Error::unsupported_feature(
                    "scanline reader does not support DNL mode (height=0 in SOF)",
                ));
            }

            if parser.precision != 8 {
                return Err(Error::unsupported_feature(
                    "12-bit precision JPEG (Extended Sequential) is not yet supported. \
                     Only 8-bit precision is currently implemented.",
                ));
            }

            if parser.num_components != 1
                && parser.num_components != 3
                && parser.num_components != 4
            {
                return Err(Error::unsupported_feature(
                    "scanline reader requires 1, 3, or 4 component image",
                ));
            }

            let is_grayscale = parser.num_components == 1;
            let is_cmyk = parser.num_components == 4;
            let is_xyb = parser.info().is_xyb;

            let max_v_samp = parser.components[..parser.num_components as usize]
                .iter()
                .map(|c| c.v_samp_factor as usize)
                .max()
                .unwrap_or(1);
            let mcu_height = max_v_samp * 8;

            let needs_buffered = matches!(
                parser.mode,
                JpegMode::Progressive
                    | JpegMode::ArithmeticSequential
                    | JpegMode::ArithmeticProgressive
            ) || is_cmyk;

            if needs_buffered {
                let width = parser.width;
                let height = parser.height;
                let num_components = parser.num_components;

                parser.chroma_upsampling = self.chroma_upsampling;
                parser.idct_method = self.effective_idct_method();
                parser.decode(&Unstoppable)?;

                let subsampling = compute_subsampling(&parser.components, num_components);

                let output_format = if is_grayscale {
                    PixelFormat::Gray
                } else {
                    PixelFormat::Rgb
                };
                let pixels = parser.to_pixels(
                    output_format,
                    is_xyb,
                    self.chroma_upsampling,
                    OutputTarget::Srgb8,
                    &Unstoppable,
                )?;

                ParseResult::Buffered {
                    width,
                    height,
                    num_components,
                    subsampling,
                    pixels,
                    is_xyb,
                    mcu_height,
                }
            } else {
                // Check for exotic sampling
                let max_h = parser.components[..parser.num_components as usize]
                    .iter()
                    .map(|c| c.h_samp_factor)
                    .max()
                    .unwrap_or(1);

                let needs_buffered_sampling = if is_grayscale || parser.num_components < 3 {
                    max_h > 2 || max_v_samp > 2
                } else {
                    let comps = &parser.components[..parser.num_components as usize];
                    let cb_h = comps[1].h_samp_factor;
                    let cb_v = comps[1].v_samp_factor;
                    let cr_h = comps[2].h_samp_factor;
                    let cr_v = comps[2].v_samp_factor;
                    max_h > 2
                        || max_v_samp > 2
                        || cb_h != cr_h
                        || cb_v != cr_v
                        || cb_h > comps[0].h_samp_factor
                        || cb_v > comps[0].v_samp_factor
                };

                if needs_buffered_sampling {
                    let width = parser.width;
                    let height = parser.height;
                    let num_components = parser.num_components;
                    let subsampling = subsampling_from_max(max_h, max_v_samp as u8, is_grayscale);

                    parser.chroma_upsampling = self.chroma_upsampling;
                    parser.idct_method = self.effective_idct_method();
                    parser.decode(&Unstoppable)?;

                    let output_format = if is_grayscale {
                        PixelFormat::Gray
                    } else {
                        PixelFormat::Rgb
                    };
                    let pixels = parser.to_pixels(
                        output_format,
                        is_xyb,
                        self.chroma_upsampling,
                        OutputTarget::Srgb8,
                        &Unstoppable,
                    )?;

                    ParseResult::Buffered {
                        width,
                        height,
                        num_components,
                        subsampling,
                        pixels,
                        is_xyb,
                        mcu_height,
                    }
                } else {
                    // Baseline streaming mode
                    let scan_data = parser.into_scan_data(is_grayscale)?;
                    let width = scan_data.width;
                    let height = scan_data.height;

                    // Try wave-parallel — only extract the WaveParallelState,
                    // not a full ScanlineReader (which would borrow vec).
                    #[cfg(feature = "parallel")]
                    let wave_info = if self.num_threads != 1 {
                        self.compute_wave_state(&scan_data, mcu_height)?
                    } else {
                        None
                    };

                    // Release the borrow on vec by converting to owned scan data
                    let scan_data = scan_data.into_owned();

                    ParseResult::Streaming {
                        scan_data,
                        width,
                        height,
                        mcu_height,
                        #[cfg(feature = "parallel")]
                        wave_info,
                    }
                }
            }
        };
        // parser and all borrows of vec are now dropped.

        // Phase 2: construct the reader with owned data.
        match parse_result {
            ParseResult::Buffered {
                width,
                height,
                num_components,
                subsampling,
                pixels,
                is_xyb,
                mcu_height,
            } => {
                let mut reader = ScanlineReader::new_buffered_cow(
                    Cow::Owned(vec),
                    width,
                    height,
                    num_components,
                    subsampling,
                    pixels,
                    is_xyb,
                );
                self.apply_crop(&mut reader, width, height, mcu_height)?;
                Ok(reader)
            }
            ParseResult::Streaming {
                scan_data,
                width,
                height,
                mcu_height,
                #[cfg(feature = "parallel")]
                wave_info,
            } => {
                #[cfg(feature = "parallel")]
                if let Some((wave_state, is_wave_only)) = wave_info {
                    if is_wave_only {
                        // Wave-only: create reader from wave state with owned data
                        let mut reader =
                            ScanlineReader::new_wave_parallel_cow(Cow::Owned(vec), wave_state);
                        self.apply_crop(&mut reader, width, height, mcu_height)?;
                        return Ok(reader);
                    } else {
                        let mut reader = ScanlineReader::from_scan_data_cow(
                            scan_data,
                            Cow::Owned(vec),
                            self.chroma_upsampling,
                            self.effective_idct_method(),
                            self.output_target,
                            self.deblock_mode,
                        )?;
                        reader.attach_wave_state(wave_state);
                        self.apply_crop(&mut reader, width, height, mcu_height)?;
                        return Ok(reader);
                    }
                }

                let mut reader = ScanlineReader::from_scan_data_cow(
                    scan_data,
                    Cow::Owned(vec),
                    self.chroma_upsampling,
                    self.effective_idct_method(),
                    self.output_target,
                    self.deblock_mode,
                )?;
                self.apply_crop(&mut reader, width, height, mcu_height)?;
                Ok(reader)
            }
        }
    }

    /// Transform path for owned data: fully decodes and stores owned data.
    fn scanline_reader_with_transform_owned<'a>(
        &self,
        vec: alloc::vec::Vec<u8>,
        transform: crate::lossless::LosslessTransform,
    ) -> Result<ScanlineReader<'a>> {
        use crate::lossless::LosslessTransform;
        use crate::types::Subsampling;
        use alloc::borrow::Cow;

        enum TransformResult {
            Buffered {
                vis_w: u32,
                vis_h: u32,
                num_components: u8,
                pixels: alloc::vec::Vec<u8>,
                mcu_height: usize,
            },
            Coefficient {
                coefficients: DecodedCoefficients,
                width: u32,
                height: u32,
                mcu_height: usize,
            },
        }

        let result = {
            let mut parser =
                JpegParser::with_strictness(&vec, self.max_pixels, None, self.strictness)?;
            parser.decode_mode = parser::DecodeMode::Coefficient;
            parser.chroma_upsampling = self.chroma_upsampling;
            parser.idct_method = self.effective_idct_method();
            parser.decode(&Unstoppable)?;

            let max_v_samp = parser.components[..parser.num_components as usize]
                .iter()
                .map(|c| c.v_samp_factor as usize)
                .max()
                .unwrap_or(1);
            let mcu_height = max_v_samp * 8;

            let orig_w = parser.width as usize;
            let orig_h = parser.height as usize;
            let pad_x = ((orig_w + 7) / 8) * 8 - orig_w;
            let pad_y = ((orig_h + 7) / 8) * 8 - orig_h;

            let (crop_x, crop_y) = match transform {
                LosslessTransform::None => (0, 0),
                LosslessTransform::FlipHorizontal => (pad_x, 0),
                LosslessTransform::FlipVertical => (0, pad_y),
                LosslessTransform::Rotate180 => (pad_x, pad_y),
                LosslessTransform::Transpose => (0, 0),
                LosslessTransform::Rotate90 => (pad_y, 0),
                LosslessTransform::Rotate270 => (0, pad_x),
                LosslessTransform::Transverse => (pad_y, pad_x),
            };

            parser.apply_dct_transform(transform);

            let is_cmyk = parser.num_components == 4;

            if crop_x > 0 || crop_y > 0 || transform.swaps_dimensions() || is_cmyk {
                let mut config_no_crop = self.clone();
                config_no_crop.crop_region = None;
                let result = config_no_crop.decode(&vec, Unstoppable)?;
                let vis_w = result.width();
                let vis_h = result.height();
                let num_components = result.format().num_channels() as u8;
                let pixels = result
                    .into_pixels_u8()
                    .ok_or_else(|| Error::internal("expected u8 pixel data for scanline crop"))?;

                TransformResult::Buffered {
                    vis_w,
                    vis_h,
                    num_components,
                    pixels,
                    mcu_height,
                }
            } else {
                let coefficients = parser.extract_coefficients()?;
                let width = parser.width;
                let height = parser.height;

                TransformResult::Coefficient {
                    coefficients,
                    width,
                    height,
                    mcu_height,
                }
            }
        };
        // parser dropped, borrow on vec released.

        match result {
            TransformResult::Buffered {
                vis_w,
                vis_h,
                num_components,
                pixels,
                mcu_height,
            } => {
                let mut reader = ScanlineReader::new_buffered_cow(
                    Cow::Owned(vec),
                    vis_w,
                    vis_h,
                    num_components,
                    Subsampling::S444,
                    pixels,
                    false,
                );
                self.apply_crop(&mut reader, vis_w, vis_h, mcu_height)?;
                Ok(reader)
            }
            TransformResult::Coefficient {
                coefficients,
                width,
                height,
                mcu_height,
            } => {
                let mut reader = ScanlineReader::from_coefficients(
                    coefficients,
                    self.chroma_upsampling,
                    self.effective_idct_method(),
                    self.output_target,
                )?;
                reader.replace_data(Cow::Owned(vec));
                self.apply_crop(&mut reader, width, height, mcu_height)?;
                Ok(reader)
            }
        }
    }

    /// Compute wave-parallel state from scan data without creating a reader.
    ///
    /// Returns `Ok(Some((wave_state, is_wave_only)))` if wave parallel is
    /// eligible. `is_wave_only` is true when the wave decode can serve all
    /// output paths (box filter 4:2:0).
    ///
    /// This is the core computation shared by [`try_wave_parallel`] and the
    /// owned-data path.
    #[cfg(feature = "parallel")]
    fn compute_wave_state(
        &self,
        scan_data: &parser::ParsedScanData<'_>,
        mcu_height: usize,
    ) -> Result<Option<(fused_parallel::WaveParallelState, bool)>> {
        use fused_parallel::{WaveParallelState, build_huffman_tables_from_scan_data};
        use rst_scan::{compute_segments, scan_rst_markers};

        let num_comps = scan_data.num_components as usize;

        let ri = scan_data.restart_interval as usize;
        if ri == 0 {
            return Ok(None);
        }

        let max_h = scan_data.h_samp[..num_comps]
            .iter()
            .copied()
            .max()
            .unwrap_or(1) as usize;
        let max_v = scan_data.v_samp[..num_comps]
            .iter()
            .copied()
            .max()
            .unwrap_or(1) as usize;
        let mcu_width = max_h * 8;
        let mcu_cols = (scan_data.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (scan_data.height as usize + mcu_height - 1) / mcu_height;
        let total_mcus = mcu_cols * mcu_rows;

        // MCU-row alignment check
        if ri % mcu_cols != 0 {
            return Ok(None);
        }

        if total_mcus < 1024 {
            return Ok(None);
        }

        // Scan for RST markers
        let expected_markers = total_mcus / ri;
        let entropy_data = &scan_data.data[scan_data.scan_data_start..];
        let rst_result = scan_rst_markers(entropy_data, expected_markers);

        if rst_result.markers.is_empty() {
            return Ok(None);
        }

        let (seg_starts, seg_ends) = compute_segments(&rst_result.markers, rst_result.entropy_end);
        let num_segments = seg_starts.len();

        if num_segments < 4 {
            return Ok(None);
        }

        // Build scan_comps from table_mapping
        let scan_comps: Vec<(usize, u8, u8)> = (0..num_comps)
            .map(|i| {
                let (dc, ac) = scan_data.table_mapping[i];
                (i, dc as u8, ac as u8)
            })
            .collect();

        // Build Huffman tables
        let (dc_tables, ac_tables) = build_huffman_tables_from_scan_data(scan_data, &scan_comps);

        // Determine wave size: balance parallelism, load balancing, and memory
        let num_threads = rayon::current_num_threads();
        let width = scan_data.width as usize;
        let mcu_w = max_h * 8;
        let mcu_h = max_v * 8;
        let mcu_cols_wave = (width + mcu_w - 1) / mcu_w;
        let mcu_rows_per_ri = ri / mcu_cols_wave;
        let pixel_rows_per_seg = mcu_rows_per_ri * mcu_h;
        let seg_rgb_bytes = pixel_rows_per_seg * width * 3;

        // Cap wave_buf at ~6 MB to reduce peak memory vs full-buffer decode.
        const WAVE_BUF_TARGET_BYTES: usize = 6 * 1024 * 1024;
        let max_wave_by_mem = if seg_rgb_bytes > 0 {
            (WAVE_BUF_TARGET_BYTES / seg_rgb_bytes).max(1)
        } else {
            num_segments
        };

        // 2x oversubscription for load balancing, capped by memory budget
        let wave_size = (num_threads * 2)
            .max(4)
            .min(num_segments)
            .min(max_wave_by_mem);

        let wave_state = WaveParallelState::new(
            scan_data,
            seg_starts,
            seg_ends,
            scan_comps,
            dc_tables,
            ac_tables,
            self.strictness,
            self.effective_idct_method(),
            wave_size,
        );

        // Determine if we can use the wave-only RGB box path (original behavior)
        let is_box_filter = matches!(self.chroma_upsampling, ChromaUpsampling::NearestNeighbor);
        let is_subsampled_color = num_comps == 3
            && (scan_data.h_samp[1] != scan_data.h_samp[0]
                || scan_data.v_samp[1] != scan_data.v_samp[0]);
        let is_wave_only = is_box_filter && is_subsampled_color;

        Ok(Some((wave_state, is_wave_only)))
    }

    /// Try to create a wave-parallel scanline reader.
    ///
    /// Returns `Ok(Some(reader))` if wave parallel is eligible and activated,
    /// `Ok(None)` to fall through to the sequential streaming path.
    ///
    /// Eligibility: baseline 4:2:0, box filter, MCU-row-aligned DRI, enough segments.
    #[cfg(feature = "parallel")]
    fn try_wave_parallel<'a>(
        &self,
        scan_data: &parser::ParsedScanData<'a>,
        mcu_height: usize,
    ) -> Result<Option<WaveResult<'a>>> {
        let Some((wave_state, is_wave_only)) = self.compute_wave_state(scan_data, mcu_height)?
        else {
            return Ok(None);
        };

        let width = scan_data.width;
        let height = scan_data.height;

        if is_wave_only {
            // Wave-only reader: both RGB and planar served from wave decode
            let mut reader = ScanlineReader::new_wave_parallel(scan_data.data, wave_state);
            self.apply_crop(&mut reader, width, height, mcu_height)?;
            Ok(Some(WaveResult::WaveOnly(reader)))
        } else {
            // Return wave state for caller to attach to sequential reader
            Ok(Some(WaveResult::WaveState(wave_state)))
        }
    }

    /// Creates a scanline reader that applies a DCT-domain transform.
    ///
    /// Does a full coefficient decode + transform, then creates a reader
    /// that streams pixels from the transformed coefficients.
    /// Fallback for deblock modes that need coefficient access (Knusperli, Auto at low Q).
    /// Runs full `decode()` then wraps the result in a buffered `ScanlineReader`.
    fn scanline_reader_deblock_fallback<'a>(&self, data: &'a [u8]) -> Result<ScanlineReader<'a>> {
        let result = self.decode(data, Unstoppable)?;
        let (vis_w, vis_h, num_ch) = (
            result.width(),
            result.height(),
            result.format().num_channels() as u8,
        );
        let pixels = result
            .into_pixels_u8()
            .ok_or_else(|| Error::internal("expected u8 pixels from deblock fallback"))?;
        Ok(ScanlineReader::new_buffered(
            data,
            vis_w,
            vis_h,
            num_ch,
            Subsampling::S444,
            pixels,
            false,
        ))
    }

    /// Owned-data variant of deblock fallback.
    fn scanline_reader_deblock_fallback_owned<'a>(
        &self,
        data: alloc::vec::Vec<u8>,
    ) -> Result<ScanlineReader<'a>> {
        let result = self.decode(&data, Unstoppable)?;
        let (vis_w, vis_h, num_ch) = (
            result.width(),
            result.height(),
            result.format().num_channels() as u8,
        );
        let pixels = result
            .into_pixels_u8()
            .ok_or_else(|| Error::internal("expected u8 pixels from deblock fallback"))?;
        Ok(ScanlineReader::new_buffered_cow(
            alloc::borrow::Cow::Owned(data),
            vis_w,
            vis_h,
            num_ch,
            Subsampling::S444,
            pixels,
            false,
        ))
    }

    ///
    /// For non-MCU-aligned images where the transform moves padding to a visible
    /// edge, falls back to a buffered decode + crop approach (same as `decode()`).
    fn scanline_reader_with_transform<'a>(
        &self,
        data: &'a [u8],
        transform: crate::lossless::LosslessTransform,
    ) -> Result<ScanlineReader<'a>> {
        use crate::lossless::LosslessTransform;

        let mut parser = JpegParser::with_strictness(data, self.max_pixels, None, self.strictness)?;
        parser.decode_mode = parser::DecodeMode::Coefficient; // Need coefficient storage
        parser.chroma_upsampling = self.chroma_upsampling;
        parser.idct_method = self.effective_idct_method();
        parser.decode(&Unstoppable)?;

        // Compute MCU height for crop resolution
        let max_v_samp = parser.components[..parser.num_components as usize]
            .iter()
            .map(|c| c.v_samp_factor as usize)
            .max()
            .unwrap_or(1);
        let mcu_height = max_v_samp * 8;

        // Check if crop is needed for non-MCU-aligned images
        let orig_w = parser.width as usize;
        let orig_h = parser.height as usize;
        let pad_x = ((orig_w + 7) / 8) * 8 - orig_w;
        let pad_y = ((orig_h + 7) / 8) * 8 - orig_h;

        let (crop_x, crop_y) = match transform {
            LosslessTransform::None => (0, 0),
            LosslessTransform::FlipHorizontal => (pad_x, 0),
            LosslessTransform::FlipVertical => (0, pad_y),
            LosslessTransform::Rotate180 => (pad_x, pad_y),
            LosslessTransform::Transpose => (0, 0),
            LosslessTransform::Rotate90 => (pad_y, 0),
            LosslessTransform::Rotate270 => (0, pad_x),
            LosslessTransform::Transverse => (pad_y, pad_x),
        };

        parser.apply_dct_transform(transform);

        // CMYK (4-component) coefficient path not supported by StripProcessor
        // (h_samp/v_samp are [u8; 3]), so always use buffered decode for CMYK.
        let is_cmyk = parser.num_components == 4;

        if crop_x > 0 || crop_y > 0 || transform.swaps_dimensions() || is_cmyk {
            // Crop needed, dimension-swapping transform (which uses f32 IDCT),
            // or CMYK (4-component): fall back to full buffered decode + crop.
            // This ensures the scanline path matches the buffered decode() path exactly.
            // Use a config without crop_region for decode — we apply crop on the reader.
            let mut config_no_crop = self.clone();
            config_no_crop.crop_region = None;
            let result = config_no_crop.decode(data, Unstoppable)?;
            let vis_w = result.width();
            let vis_h = result.height();
            let num_components = result.format().num_channels() as u8;
            let is_xyb = false; // XYB is converted to RGB during decode
            let pixels = result
                .into_pixels_u8()
                .ok_or_else(|| Error::internal("expected u8 pixel data for scanline crop"))?;

            let mut reader = ScanlineReader::new_buffered(
                data,
                vis_w,
                vis_h,
                num_components,
                Subsampling::S444,
                pixels,
                is_xyb,
            );
            // Crop is in output space (post-transform), resolve against visible dims
            self.apply_crop(&mut reader, vis_w, vis_h, mcu_height)?;
            return Ok(reader);
        }

        let coefficients = parser.extract_coefficients()?;
        let width = parser.width;
        let height = parser.height;
        let mut reader = ScanlineReader::from_coefficients(
            coefficients,
            self.chroma_upsampling,
            self.effective_idct_method(),
            self.output_target,
        )?;
        self.apply_crop(&mut reader, width, height, mcu_height)?;
        Ok(reader)
    }

    /// Resolves and applies the user's crop region to a scanline reader.
    fn apply_crop(
        &self,
        reader: &mut ScanlineReader<'_>,
        img_w: u32,
        img_h: u32,
        mcu_height: usize,
    ) -> Result<()> {
        if let Some(crop_region) = self.crop_region {
            let resolved = crop_region.resolve(img_w, img_h, mcu_height)?;
            reader.set_crop(resolved);
        }
        Ok(())
    }

    /// Compute the effective transform from raw JPEG data.
    ///
    /// Scans raw bytes for EXIF orientation (lightweight, no full parse),
    /// composes with any user-specified transform.
    /// Returns `LosslessTransform::None` if no transform is needed.
    pub(crate) fn compute_effective_transform_from_data(
        &self,
        data: &[u8],
    ) -> crate::lossless::LosslessTransform {
        use crate::lossless::LosslessTransform;

        let exif_transform = if self.auto_orient {
            find_exif_orientation(data)
                .and_then(LosslessTransform::from_exif_orientation)
                .unwrap_or(LosslessTransform::None)
        } else {
            LosslessTransform::None
        };

        let user_transform = self.decode_transform.unwrap_or(LosslessTransform::None);

        if exif_transform == LosslessTransform::None {
            user_transform
        } else if user_transform == LosslessTransform::None {
            exif_transform
        } else {
            exif_transform.then(user_transform)
        }
    }

    /// Decodes a JPEG image.
    ///
    /// For large images or memory-constrained environments, consider using
    /// [`scanline_reader()`](Self::scanline_reader) to decode row-by-row
    /// into caller-provided buffers.
    pub fn decode(&self, data: &[u8], stop: impl Stop) -> Result<DecodeResult> {
        // Track whether we force-preserved EXIF just for auto_orient
        let forced_exif = self.auto_orient && !self.preserve.exif;
        let preserve = if self.auto_orient {
            // Ensure EXIF is preserved for orientation reading
            let mut p = self.preserve.clone();
            p.exif = true;
            p
        } else {
            self.preserve.clone()
        };

        // Pre-compute effective transform from raw data before full parse.
        // This avoids disabling streaming for the common case of orientation=1.
        let effective_transform = self.compute_effective_transform_from_data(data);

        let mut parser =
            JpegParser::with_strictness(data, self.max_pixels, Some(&preserve), self.strictness)?;

        // Streaming decode produces RGB u8 directly — disable it when the output
        // needs coefficients (f32, u16, precise, dequant_bias, transform, non-RGB formats,
        // or Knusperli deblocking which requires DCT coefficients).
        //
        // Boundary4Tap, Auto, and AutoStreamable do NOT need coefficient storage —
        // they operate in the pixel domain after IDCT and are applied as a post-decode
        // pass on the u8 output. Only Knusperli needs raw coefficients.
        {
            let output_format = self.output_format.unwrap_or(PixelFormat::Rgb);
            let needs_coefficients = self.output_target.is_f32()
                || self.output_target.is_precise()
                || self.output_target.uses_dequant_bias()
                || effective_transform != crate::lossless::LosslessTransform::None
                || self.deblock_mode == DeblockMode::Knusperli
                || !matches!(
                    output_format,
                    PixelFormat::Rgb
                        | PixelFormat::Bgr
                        | PixelFormat::Rgba
                        | PixelFormat::Bgra
                        | PixelFormat::Bgrx
                );
            if needs_coefficients {
                parser.decode_mode = parser::DecodeMode::Coefficient;
            }
        }
        parser.chroma_upsampling = self.chroma_upsampling;
        parser.idct_method = self.effective_idct_method();
        parser.num_threads = self.num_threads;
        #[cfg(feature = "parallel")]
        {
            parser.parallel_strategy = self.parallel_strategy;
        }
        // Propagate force_f32_idct from config (set by dimension-swapping transforms).
        // Must be set before decode() so can_use_streaming() sees it.
        if self.force_f32_idct {
            parser.force_f32_idct = true;
        }
        parser.decode(&stop)?;

        // Apply DCT transform if needed.
        //
        // For non-MCU-aligned images, padding moves to different edges after
        // the transform (e.g., FlipH moves right padding to the left).
        // We compute the crop offset, inflate parser dimensions so to_pixels()
        // renders the full padded region, then crop the result.
        let (crop_x, crop_y, visible_w, visible_h) =
            if effective_transform != crate::lossless::LosslessTransform::None {
                use crate::lossless::LosslessTransform;

                // Compute padding before transform
                let orig_w = parser.width as usize;
                let orig_h = parser.height as usize;
                let pad_x = ((orig_w + 7) / 8) * 8 - orig_w;
                let pad_y = ((orig_h + 7) / 8) * 8 - orig_h;

                parser.apply_dct_transform(effective_transform);

                let vis_w = parser.width;
                let vis_h = parser.height;

                // Crop offset: where the visible region starts after transform
                let (cx, cy) = match effective_transform {
                    LosslessTransform::None => (0, 0),
                    LosslessTransform::FlipHorizontal => (pad_x, 0),
                    LosslessTransform::FlipVertical => (0, pad_y),
                    LosslessTransform::Rotate180 => (pad_x, pad_y),
                    LosslessTransform::Transpose => (0, 0),
                    LosslessTransform::Rotate90 => (pad_y, 0),
                    LosslessTransform::Rotate270 => (0, pad_x),
                    LosslessTransform::Transverse => (pad_y, pad_x),
                };

                // Inflate dimensions so to_pixels() renders enough data
                if cx > 0 || cy > 0 {
                    parser.width = vis_w + cx as u32;
                    parser.height = vis_h + cy as u32;
                }

                (cx, cy, vis_w, vis_h)
            } else {
                (0, 0, parser.width, parser.height)
            };

        // Extract gain map before pixel conversion (needs parser state)
        #[cfg(feature = "ultrahdr")]
        let gain_map_result = if self.gain_map != GainMapHandling::Discard {
            self.extract_gain_map(&mut parser, data)?
        } else {
            None
        };
        #[cfg(not(feature = "ultrahdr"))]
        let gain_map_result: Option<GainMapResult> = None;

        let info = parser.info();
        let output_format = self.output_format.unwrap_or(PixelFormat::Rgb);

        let mut result = if self.output_target.is_f32() {
            // f32 output path
            #[allow(unused_mut)]
            let mut pixels = if self.deblock_mode != DeblockMode::Off {
                parser.to_pixels_f32_deblock(
                    output_format,
                    info.is_xyb,
                    self.chroma_upsampling,
                    self.deblock_mode,
                    &stop,
                )?
            } else {
                parser.to_pixels_f32(output_format, info.is_xyb, self.chroma_upsampling, &stop)?
            };

            // Crop to visible region if transform introduced a crop offset
            if crop_x > 0 || crop_y > 0 {
                let inflated_w = parser.width as usize;
                let vis_w = visible_w as usize;
                let vis_h = visible_h as usize;
                let channels = output_format.num_channels();
                let mut cropped = vec![0f32; vis_w * vis_h * channels];
                for y in 0..vis_h {
                    let src_off = ((crop_y + y) * inflated_w + crop_x) * channels;
                    let dst_off = y * vis_w * channels;
                    let row_elems = vis_w * channels;
                    cropped[dst_off..dst_off + row_elems]
                        .copy_from_slice(&pixels[src_off..src_off + row_elems]);
                }
                pixels = cropped;
            }

            // Apply ICC profile if enabled and present.
            // At this point pixels are always RGB (CMYK/YCCK already converted).
            // The ICC profile may describe any source color space (sRGB, Adobe RGB,
            // CMYK working space) — moxcms handles the transform to the target.
            #[cfg(feature = "moxcms")]
            if let Some(target) = self.correct_color
                && let Some(ref icc_profile) = parser.icc_profile
            {
                match apply_icc_transform_f32(
                    &pixels,
                    visible_w as usize,
                    visible_h as usize,
                    icc_profile,
                    target,
                ) {
                    Ok(transformed) => pixels = transformed,
                    Err(_e) => {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Warning: ICC f32 transform failed, using original colors: {_e:?}"
                        );
                    }
                }
            }

            // Apply sRGB→linear transfer if requested
            if self.output_target.is_linear() {
                crate::color::icc::srgb_to_linear_inplace(&mut pixels);
            }

            // Apply user crop region (pixel-level crop of decoded buffer)
            let (out_w, out_h) = if let Some(crop_region) = self.crop_region {
                let resolved = crop_region.resolve(visible_w, visible_h, 8)?;
                let cw = resolved.width as usize;
                let ch = resolved.height as usize;
                let cx = resolved.x as usize;
                let cy = resolved.y as usize;
                let src_w = visible_w as usize;
                let channels = output_format.num_channels();
                let mut cropped = vec![0f32; cw * ch * channels];
                for y in 0..ch {
                    let src_off = ((cy + y) * src_w + cx) * channels;
                    let dst_off = y * cw * channels;
                    let row_elems = cw * channels;
                    cropped[dst_off..dst_off + row_elems]
                        .copy_from_slice(&pixels[src_off..src_off + row_elems]);
                }
                pixels = cropped;
                (resolved.width, resolved.height)
            } else {
                (visible_w, visible_h)
            };

            let extras = finalize_extras(parser.take_extras(), data, forced_exif);
            let warnings = parser.take_warnings();
            DecodeResult::new_f32(
                out_w,
                out_h,
                output_format,
                self.output_target,
                pixels,
                extras,
                warnings,
            )
        } else if self.deblock_mode == DeblockMode::Knusperli {
            // u8 output with Knusperli deblocking: route through f32 deblock path.
            // Knusperli requires raw DCT coefficients, so it must use the coefficient path.
            let f32_pixels = parser.to_pixels_f32_deblock(
                output_format,
                info.is_xyb,
                self.chroma_upsampling,
                self.deblock_mode,
                &stop,
            )?;
            #[allow(unused_mut)]
            let mut pixels: Vec<u8> = f32_pixels
                .iter()
                .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
                .collect();

            // Crop to visible region if transform introduced a crop offset
            if crop_x > 0 || crop_y > 0 {
                let inflated_w = parser.width as usize;
                let vis_w = visible_w as usize;
                let vis_h = visible_h as usize;
                let bpp = output_format.bytes_per_pixel();
                let mut cropped = vec![0u8; vis_w * vis_h * bpp];
                for y in 0..vis_h {
                    let src_off = ((crop_y + y) * inflated_w + crop_x) * bpp;
                    let dst_off = y * vis_w * bpp;
                    let row_bytes = vis_w * bpp;
                    cropped[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&pixels[src_off..src_off + row_bytes]);
                }
                pixels = cropped;
            }

            let (out_w, out_h) = if let Some(crop_region) = self.crop_region {
                let resolved = crop_region.resolve(visible_w, visible_h, 8)?;
                let cw = resolved.width as usize;
                let ch = resolved.height as usize;
                let cx = resolved.x as usize;
                let cy = resolved.y as usize;
                let src_w = visible_w as usize;
                let bpp = output_format.bytes_per_pixel();
                let mut cropped = vec![0u8; cw * ch * bpp];
                for y in 0..ch {
                    let src_off = ((cy + y) * src_w + cx) * bpp;
                    let dst_off = y * cw * bpp;
                    let row_bytes = cw * bpp;
                    cropped[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&pixels[src_off..src_off + row_bytes]);
                }
                pixels = cropped;
                (resolved.width, resolved.height)
            } else {
                (visible_w, visible_h)
            };

            let extras = finalize_extras(parser.take_extras(), data, forced_exif);
            let warnings = parser.take_warnings();
            DecodeResult::new_u8(
                out_w,
                out_h,
                output_format,
                self.output_target,
                pixels,
                extras,
                warnings,
            )
        } else {
            // u8 output path (fast streaming decode).
            // Boundary4Tap/Auto/AutoStreamable deblocking is applied post-decode.
            #[allow(unused_mut)]
            let mut pixels = parser.to_pixels(
                output_format,
                info.is_xyb,
                self.chroma_upsampling,
                self.output_target,
                &stop,
            )?;

            // Apply boundary 4-tap deblock as post-processing on interleaved u8 data.
            // This handles Boundary4Tap, Auto (resolved to Boundary4Tap in streaming),
            // and AutoStreamable modes. The filter operates per-channel at 8-pixel block
            // boundaries in the RGB domain, using luma DC quant for filter strength.
            if matches!(
                self.deblock_mode,
                DeblockMode::Boundary4Tap | DeblockMode::Auto | DeblockMode::AutoStreamable
            ) {
                let width = visible_w as usize;
                let height = visible_h as usize;
                // bytes_per_pixel gives the interleaved channel count (3 for RGB, 4 for RGBA)
                let channels = output_format.bytes_per_pixel();
                // Use luma DC quant for strength — it's the dominant visual component.
                let dc_quant = parser.quant_tables[parser.components[0].quant_table_idx as usize]
                    .map(|qt| qt[0])
                    .unwrap_or(1);
                let strength = crate::deblock::BoundaryStrength::from_dc_quant(dc_quant);
                crate::deblock::filter_interleaved_u8_boundary_4tap(
                    &mut pixels,
                    width,
                    height,
                    channels,
                    strength,
                );
            }

            // Crop to visible region if transform introduced a crop offset
            if crop_x > 0 || crop_y > 0 {
                let inflated_w = parser.width as usize;
                let vis_w = visible_w as usize;
                let vis_h = visible_h as usize;
                let bpp = output_format.bytes_per_pixel();
                let mut cropped = vec![0u8; vis_w * vis_h * bpp];
                for y in 0..vis_h {
                    let src_off = ((crop_y + y) * inflated_w + crop_x) * bpp;
                    let dst_off = y * vis_w * bpp;
                    let row_bytes = vis_w * bpp;
                    cropped[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&pixels[src_off..src_off + row_bytes]);
                }
                pixels = cropped;
            }

            // Apply ICC profile if enabled and present.
            // Pixels are always RGB here (CMYK/YCCK converted earlier in output.rs).
            #[cfg(feature = "moxcms")]
            if let Some(target) = self.correct_color
                && let Some(ref icc_profile) = parser.icc_profile
            {
                match apply_icc_transform(
                    &pixels,
                    visible_w as usize,
                    visible_h as usize,
                    icc_profile,
                    target,
                ) {
                    Ok(transformed) => pixels = transformed,
                    Err(_e) => {
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Warning: ICC profile transform failed, using original colors: {_e:?}"
                        );
                    }
                }
            }

            // Apply user crop region (pixel-level crop of decoded buffer)
            let (out_w, out_h) = if let Some(crop_region) = self.crop_region {
                let resolved = crop_region.resolve(visible_w, visible_h, 8)?;
                let cw = resolved.width as usize;
                let ch = resolved.height as usize;
                let cx = resolved.x as usize;
                let cy = resolved.y as usize;
                let src_w = visible_w as usize;
                let bpp = output_format.bytes_per_pixel();
                let mut cropped = vec![0u8; cw * ch * bpp];
                for y in 0..ch {
                    let src_off = ((cy + y) * src_w + cx) * bpp;
                    let dst_off = y * cw * bpp;
                    let row_bytes = cw * bpp;
                    cropped[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&pixels[src_off..src_off + row_bytes]);
                }
                pixels = cropped;
                (resolved.width, resolved.height)
            } else {
                (visible_w, visible_h)
            };

            let extras = finalize_extras(parser.take_extras(), data, forced_exif);
            let warnings = parser.take_warnings();
            DecodeResult::new_u8(
                out_w,
                out_h,
                output_format,
                self.output_target,
                pixels,
                extras,
                warnings,
            )
        };

        result.set_gain_map(gain_map_result);
        Ok(result)
    }

    /// Push-based callback decode: calls `callback` once per row of decoded u8 pixels.
    ///
    /// This is the push counterpart to the pull-based [`scanline_reader()`](Self::scanline_reader).
    /// The callback receives a [`RowSlice`] borrowing the decoder's internal buffer (zero-copy).
    ///
    /// Supported formats: [`PixelFormat::Rgb`], [`Bgr`](PixelFormat::Bgr),
    /// [`Rgba`](PixelFormat::Rgba), [`Bgra`](PixelFormat::Bgra),
    /// [`Bgrx`](PixelFormat::Bgrx), [`Gray`](PixelFormat::Gray).
    ///
    /// For f32 output, use [`decode_rows_f32()`](Self::decode_rows_f32).
    ///
    /// # Early abort
    ///
    /// Return `Err` from the callback to stop decoding immediately.
    /// The error is propagated as the return value.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::{Decoder, PixelFormat};
    ///
    /// let mut hash = 0u64;
    /// let info = Decoder::new().decode_rows(
    ///     &jpeg_data,
    ///     PixelFormat::Rgb,
    ///     |row| {
    ///         for &b in row.as_bytes() {
    ///             hash = hash.wrapping_mul(31).wrapping_add(b as u64);
    ///         }
    ///         Ok(())
    ///     },
    ///     enough::Unstoppable,
    /// )?;
    /// println!("{}x{}, hash={}", info.dimensions.width, info.dimensions.height, hash);
    /// ```
    pub fn decode_rows<F>(
        &self,
        data: &[u8],
        format: PixelFormat,
        mut callback: F,
        stop: impl Stop,
    ) -> Result<ScanlineInfo>
    where
        F: FnMut(RowSlice<'_>) -> Result<()>,
    {
        // Validate format is u8-based
        match format {
            PixelFormat::Rgb
            | PixelFormat::Bgr
            | PixelFormat::Rgba
            | PixelFormat::Bgra
            | PixelFormat::Bgrx
            | PixelFormat::Gray => {}
            _ => {
                return Err(Error::unsupported_feature(
                    "decode_rows() only supports u8 formats (Rgb, Bgr, Rgba, Bgra, Bgrx, Gray). \
                     For f32 formats use decode_rows_f32().",
                ));
            }
        }

        let mut reader = self.scanline_reader(data)?;
        let info = reader.info();
        let width = reader.width() as usize;
        let bpp = format.bytes_per_pixel();
        let row_bytes = width * bpp;

        // Reusable single-row buffer
        let mut row_buf = vec![0u8; row_bytes];

        let mut row_index = 0usize;
        while !reader.is_finished() {
            stop.check()?;

            // Wrap as 1-row ImgRefMut (width = stride in bytes for u8)
            let img = ImgRefMut::new(&mut row_buf, row_bytes, 1);
            let read = match format {
                PixelFormat::Rgb => reader.read_rows_rgb8(img)?,
                PixelFormat::Bgr => reader.read_rows_bgr8(img)?,
                PixelFormat::Rgba => reader.read_rows_rgba8(img)?,
                PixelFormat::Bgra => reader.read_rows_bgra8(img)?,
                PixelFormat::Bgrx => reader.read_rows_bgrx8(img)?,
                PixelFormat::Gray => reader.read_rows_gray8(img)?,
                _ => unreachable!(),
            };

            if read > 0 {
                let slice = RowSlice::new(&row_buf[..row_bytes], row_index, width, format);
                callback(slice)?;
                row_index += 1;
            }
        }

        Ok(info)
    }

    /// Push-based callback decode for f32 pixel formats.
    ///
    /// The callback receives a [`RowSliceF32`] borrowing the decoder's internal buffer.
    ///
    /// Supported formats: [`PixelFormat::RgbaF32`], [`GrayF32`](PixelFormat::GrayF32).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::{Decoder, PixelFormat};
    ///
    /// Decoder::new().decode_rows_f32(
    ///     &jpeg_data,
    ///     PixelFormat::RgbaF32,
    ///     |row| {
    ///         let floats = row.as_slice();
    ///         // Process RGBA f32 data...
    ///         Ok(())
    ///     },
    ///     enough::Unstoppable,
    /// )?;
    /// ```
    pub fn decode_rows_f32<F>(
        &self,
        data: &[u8],
        format: PixelFormat,
        mut callback: F,
        stop: impl Stop,
    ) -> Result<ScanlineInfo>
    where
        F: FnMut(RowSliceF32<'_>) -> Result<()>,
    {
        // Validate format is f32-based
        let floats_per_pixel = match format {
            PixelFormat::RgbaF32 => 4,
            PixelFormat::GrayF32 => 1,
            _ => {
                return Err(Error::unsupported_feature(
                    "decode_rows_f32() only supports f32 formats (RgbaF32, GrayF32). \
                     For u8 formats use decode_rows().",
                ));
            }
        };

        let mut reader = self.scanline_reader(data)?;
        let info = reader.info();
        let width = reader.width() as usize;
        let row_floats = width * floats_per_pixel;

        // Reusable single-row buffer
        let mut row_buf = vec![0f32; row_floats];

        let mut row_index = 0usize;
        while !reader.is_finished() {
            stop.check()?;

            // Wrap as 1-row ImgRefMut (width = stride in f32 elements)
            let img = ImgRefMut::new(&mut row_buf, row_floats, 1);
            let read = match format {
                PixelFormat::RgbaF32 => reader.read_rows_rgba_f32(img)?,
                PixelFormat::GrayF32 => reader.read_rows_gray_f32(img)?,
                _ => unreachable!(),
            };

            if read > 0 {
                let slice = RowSliceF32::new(&row_buf[..row_floats], row_index, width, format);
                callback(slice)?;
                row_index += 1;
            }
        }

        Ok(info)
    }

    /// Decodes a JPEG and extracts raw quantized DCT coefficients.
    ///
    /// This provides access to the coefficients before IDCT and color conversion,
    /// useful for debugging, quality analysis, and encoder comparison.
    ///
    /// Coefficients are stored in zigzag order as they appear in the JPEG file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let coeffs = decoder.decode_coefficients(&jpeg_data)?;
    ///
    /// // Access Y component DC coefficient for first block
    /// let y_dc = coeffs.components[0].block(0)[0];
    /// println!("Y DC coefficient: {}", y_dc);
    ///
    /// // Compare with another JPEG's coefficients
    /// let other_coeffs = decoder.decode_coefficients(&other_jpeg_data)?;
    /// let comparison = coeffs.compare(&other_coeffs);
    /// println!("{}% of blocks differ", comparison.diff_block_pct());
    /// ```
    ///
    /// For analysis of large images, consider streaming APIs.
    pub fn decode_coefficients(&self, data: &[u8], stop: impl Stop) -> Result<DecodedCoefficients> {
        let mut parser = JpegParser::with_strictness(data, self.max_pixels, None, self.strictness)?;
        // Disable streaming - we need coefficients stored
        parser.decode_mode = parser::DecodeMode::Coefficient;
        parser.decode(&stop)?;

        // Extract coefficients from parser
        parser.extract_coefficients()
    }

    /// Decodes a JPEG to coefficients AND preserved metadata in a single parse pass.
    ///
    /// This avoids the overhead of decoding twice when you need both coefficients
    /// and metadata (e.g., for lossless transforms that preserve EXIF/ICC/XMP).
    pub fn decode_coefficients_with_extras(
        &self,
        data: &[u8],
        stop: impl Stop,
    ) -> Result<(DecodedCoefficients, Option<DecodedExtras>)> {
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            Some(&self.preserve),
            self.strictness,
        )?;
        parser.decode_mode = parser::DecodeMode::Coefficient;
        parser.decode(&stop)?;

        let extras = finalize_extras(parser.take_extras(), data, false);
        let coeffs = parser.extract_coefficients()?;
        Ok((coeffs, extras))
    }

    /// Decodes a JPEG image to planar YCbCr f32 data.
    ///
    /// This bypasses the YCbCr→RGB color conversion, providing direct access
    /// to the decoded YCbCr planes. This is significantly faster than decoding
    /// to RGB when you need YCbCr data (e.g., for re-encoding or video pipelines).
    ///
    /// # Value Range
    ///
    /// Values are in centered range [-128, 127] (raw DCT output).
    /// To convert to standard JPEG range [0, 255], add 128 to each value.
    ///
    /// # Chroma Planes
    ///
    /// Chroma planes (Cb, Cr) are always upsampled to full resolution,
    /// matching the Y plane dimensions. The upsampling method is controlled
    /// by the `chroma_upsampling` setting.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let ycbcr = decoder.decode_to_ycbcr_f32(&jpeg_data)?;
    ///
    /// // Access planes directly
    /// let y_plane = &ycbcr.y;   // [-128, 127] range
    /// let cb_plane = &ycbcr.cb;
    /// let cr_plane = &ycbcr.cr;
    ///
    /// // Or convert to JPEG range [0, 255]
    /// let y_jpeg = ycbcr.y_to_jpeg_range();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The image is grayscale (only 1 component)
    /// - The image uses XYB color space (not YCbCr)
    /// - Parsing or decoding fails
    ///
    /// For large images, consider using streaming APIs for memory-efficient decoding.
    pub fn decode_to_ycbcr_f32(&self, data: &[u8], stop: impl Stop) -> Result<DecodedYCbCr> {
        let mut parser = JpegParser::with_strictness(data, self.max_pixels, None, self.strictness)?;
        // Disable streaming - f32 YCbCr decode needs coefficients
        parser.decode_mode = parser::DecodeMode::Coefficient;
        parser.decode(&stop)?;

        let info = parser.info();

        // XYB images store data differently - not actual YCbCr
        if info.is_xyb {
            return Err(Error::unsupported_feature(
                "YCbCr output not available for XYB images",
            ));
        }

        // Grayscale images have only Y component
        if info.color_space == ColorSpace::Grayscale {
            return Err(Error::unsupported_feature(
                "YCbCr output requires 3-component image",
            ));
        }

        // Get the YCbCr planes directly
        let (y, cb, cr) = parser.to_ycbcr_planes_f32(self.chroma_upsampling)?;

        // Pass through ICC profile if present
        let icc_profile = parser.icc_profile.clone();

        Ok(DecodedYCbCr {
            y,
            cb,
            cr,
            width: info.dimensions.width,
            height: info.dimensions.height,
            icc_profile,
        })
    }

    /// Extract gain map from UltraHDR image if present.
    ///
    /// Called during `decode()` when `gain_map != Discard`.
    #[cfg(feature = "ultrahdr")]
    fn extract_gain_map(
        &self,
        parser: &mut JpegParser,
        data: &[u8],
    ) -> Result<Option<GainMapResult>> {
        let (gainmap_range, _metadata) = parser.extract_gainmap_early(data)?;

        let (start, end) = match gainmap_range {
            Some(range) => range,
            None => return Ok(None), // Not an UltraHDR image
        };

        let gainmap_jpeg = data[start..end].to_vec();

        let (pixels, width, height) = if self.gain_map == GainMapHandling::Decode {
            // Decode the gain map JPEG to pixels
            let gm_result = Decoder::new().decode(&gainmap_jpeg, enough::Unstoppable)?;
            let w = gm_result.width;
            let h = gm_result.height;
            (Some(gm_result.into_pixels_u8().unwrap()), w, h)
        } else {
            // PreserveRaw: just get dimensions from header without decoding pixels
            let gm_info = Decoder::new().read_info(&gainmap_jpeg)?;
            (None, gm_info.dimensions.width, gm_info.dimensions.height)
        };

        Ok(Some(GainMapResult {
            jpeg: gainmap_jpeg,
            pixels,
            width,
            height,
        }))
    }

    /// Creates a streaming reader for UltraHDR JPEGs.
    ///
    /// This allows decoding UltraHDR images row-by-row with configurable output modes:
    /// - **SDR-only**: Fastest decode, ignores gain map
    /// - **HDR**: Applies gain map to reconstruct HDR output
    /// - **SDR+HDR**: Dual output for preview + processing workflows
    /// - **SDR+GainMap**: For editing workflows that preserve gain maps
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::{Decoder, UltraHdrReaderConfig, UltraHdrMode};
    ///
    /// let config = UltraHdrReaderConfig::new()
    ///     .mode(UltraHdrMode::Hdr)
    ///     .display_boost(4.0);
    ///
    /// let mut reader = Decoder::new().ultrahdr_reader(&jpeg_data, config)?;
    ///
    /// while !reader.is_finished() {
    ///     let rows = reader.read_rows(16, None, Some(&mut hdr_buf), None)?;
    ///     // Process HDR rows...
    /// }
    /// ```
    ///
    /// # Memory Efficiency
    ///
    /// For a 4K image (3840x2160):
    /// - SdrOnly: ~500 KB peak
    /// - Hdr (Full): ~1 MB peak
    /// - Hdr (Streaming): ~515 KB peak
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The JPEG cannot be parsed
    /// - The image is not a baseline JPEG
    /// - The image is grayscale
    #[cfg(feature = "ultrahdr")]
    pub fn ultrahdr_reader<'a>(
        &self,
        data: &'a [u8],
        config: UltraHdrReaderConfig,
    ) -> Result<UltraHdrReader<'a>> {
        // Parse the JPEG header and get scanline reader
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            Some(&self.preserve),
            self.strictness,
        )?;
        parser.read_header()?;

        // Only baseline supported for scanline reading
        if parser.mode != JpegMode::Baseline {
            return Err(Error::unsupported_feature(
                "ultrahdr reader only supports baseline JPEG",
            ));
        }

        if parser.num_components != 3 {
            return Err(Error::unsupported_feature(
                "ultrahdr reader requires 3-component YCbCr image",
            ));
        }

        // Extract gain map byte range from MPF secondary images (if present)
        // Uses byte range instead of copying for zero-copy access
        let (gainmap_range, metadata) = parser.extract_gainmap_early(data)?;

        // Create base scanline reader
        let base_reader = self.scanline_reader(data)?;

        // Extract extras if preserving metadata
        let extras = if config.preserve_metadata {
            finalize_extras(parser.take_extras(), data, false)
        } else {
            None
        };

        UltraHdrReader::new(data, config, base_reader, extras, gainmap_range, metadata)
    }

    /// Gets the configured maximum pixels limit.
    #[must_use]
    pub fn get_max_pixels(&self) -> u64 {
        self.max_pixels
    }

    /// Gets the configured maximum memory limit.
    #[must_use]
    pub fn get_max_memory(&self) -> u64 {
        self.max_memory
    }
}

/// Scan raw JPEG data for EXIF orientation tag without full parsing.
///
/// Looks for APP1 segments containing EXIF data and reads the orientation tag.
/// Returns `Some(1..=8)` if found, `None` otherwise.
fn find_exif_orientation(data: &[u8]) -> Option<u8> {
    const EXIF_PREFIX: &[u8] = b"Exif\0\0";

    if data.len() < 4 {
        return None;
    }

    let mut pos = 2; // Skip SOI
    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];

        // Stop at SOS or EOI — no more metadata after scan data starts
        if marker == 0xDA || marker == 0xD9 {
            break;
        }

        // Skip markers without length
        if marker == 0xFF || marker == 0x00 || (0xD0..=0xD7).contains(&marker) {
            pos += 2;
            continue;
        }

        // Read segment length
        if pos + 4 > data.len() {
            break;
        }
        let length = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
        if length < 2 {
            break;
        }
        let seg_start = pos + 4;
        let seg_end = pos + 2 + length;
        if seg_end > data.len() {
            break;
        }

        // APP1 (0xE1) with EXIF prefix
        if marker == 0xE1 && seg_end - seg_start >= EXIF_PREFIX.len() {
            let seg_data = &data[seg_start..seg_end];
            if seg_data.starts_with(EXIF_PREFIX)
                && let Some(orientation) = crate::lossless::parse_exif_orientation(seg_data)
            {
                return Some(orientation);
            }
        }

        pos = seg_end;
    }
    None
}

/// Finalize extras: inject probe result and strip forced EXIF if needed.
///
/// Creates extras if none exist but probe data is available (e.g., for
/// programmatically-encoded JPEGs with no metadata segments).
fn finalize_extras(
    extras: Option<DecodedExtras>,
    data: &[u8],
    forced_exif: bool,
) -> Option<DecodedExtras> {
    let mut extras = extras.unwrap_or_else(DecodedExtras::new);

    // Inject probe result (quality estimation, encoder ID, DQT tables).
    // This is a header-only scan (<1us) — no entropy decoding.
    if let Ok(probe) = crate::detect::probe(data) {
        extras.probe = Some(probe);
    }

    if forced_exif {
        extras.remove_segments_by_type(SegmentType::Exif);
    }

    Some(extras)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
    use enough::Unstoppable;

    #[test]
    fn test_decoder_creation() {
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);

        assert_eq!(decoder.output_format, Some(PixelFormat::Rgb));
        assert_eq!(decoder.chroma_upsampling, ChromaUpsampling::Triangle);
    }

    #[test]
    fn test_encode_decode_roundtrip_gray() {
        // Create a simple 8x8 grayscale image
        let width = 8u32;
        let height = 8u32;
        let mut input = vec![0u8; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                input[y * width as usize + x] = ((x + y) * 16) as u8;
            }
        }

        // Encode using v2 API
        let config = EncoderConfig::grayscale(95.0);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Verify JPEG structure
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8); // SOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9); // EOI

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder
            .decode(&jpeg, Unstoppable)
            .expect("decoding should succeed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(
            decoded.pixels_u8().unwrap().len(),
            (width * height) as usize
        );

        // Check pixel values are reasonably close (JPEG is lossy)
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.pixels_u8().unwrap()[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        // At quality 95, differences should be small
        assert!(max_diff < 20, "max_diff {} too large", max_diff);
    }

    #[test]
    fn test_encode_decode_roundtrip_rgb() {
        // Create a simple 16x16 RGB image
        let width = 16u32;
        let height = 16u32;
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode using v2 API
        let config = EncoderConfig::ycbcr(95.0, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded = decoder
            .decode(&jpeg, Unstoppable)
            .expect("decoding should succeed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(
            decoded.pixels_u8().unwrap().len(),
            (width * height * 3) as usize
        );

        // Check pixel values are reasonably close
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.pixels_u8().unwrap()[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        // At quality 95, differences should be small
        assert!(max_diff < 30, "max_diff {} too large", max_diff);
    }

    #[test]
    fn test_decode_f32_roundtrip() {
        // Create a simple 16x16 RGB image
        let width = 16u32;
        let height = 16u32;
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode using v2 API
        let config = EncoderConfig::ycbcr(95.0, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Decode to f32
        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .output_target(OutputTarget::SrgbF32);
        let decoded_f32 = decoder
            .decode(&jpeg, Unstoppable)
            .expect("f32 decoding should succeed");

        assert_eq!(decoded_f32.width, width);
        assert_eq!(decoded_f32.height, height);
        let f32_pixels = decoded_f32.pixels_f32().unwrap();
        assert_eq!(f32_pixels.len(), (width * height * 3) as usize);

        // Verify values are approximately in 0.0-1.0 range.
        // YCbCr→RGB color matrix can produce values slightly outside [0, 1]
        // due to ringing — this is intentional to preserve full precision.
        for &v in f32_pixels {
            assert!(
                (-0.05..=1.05).contains(&v),
                "f32 value {} too far out of range",
                v
            );
        }

        // Compare with u8 decode
        let decoded_u8 = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg, Unstoppable)
            .expect("u8 decoding should succeed");
        let u8_pixels = decoded_u8.pixels_u8().unwrap();

        // Convert f32→u8 manually for comparison
        let converted: Vec<u8> = f32_pixels
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        // Values should be close - allow diff of 2 because u8 path uses integer IDCT
        // while f32 path uses f32 IDCT (standard JPEG precision difference)
        let mut max_diff = 0i32;
        for i in 0..u8_pixels.len() {
            let diff = (u8_pixels[i] as i32 - converted[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff <= 2,
            "f32→u8 conversion differs by {} from direct u8",
            max_diff
        );
    }

    #[test]
    fn test_decode_f32_precision() {
        // Create a gradient image to test precision
        let width = 64u32;
        let height = 64u32;
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                // Create a smooth gradient
                let val = ((x + y) * 2) as u8;
                input[idx] = val;
                input[idx + 1] = val;
                input[idx + 2] = val;
            }
        }

        // Encode at high quality using v2 API
        let config = EncoderConfig::ycbcr(98.0, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Decode to f32
        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .output_target(OutputTarget::SrgbF32);
        let decoded_f32 = decoder
            .decode(&jpeg, Unstoppable)
            .expect("f32 decoding should succeed");

        // Check that f32 values show more precision than just u8/255
        // by verifying we have non-quantized intermediate values
        let mut found_fractional = false;
        for &v in decoded_f32.pixels_f32().unwrap() {
            let scaled = v * 255.0;
            let frac = scaled - scaled.round();
            if frac.abs() > 0.001 && frac.abs() < 0.999 {
                found_fractional = true;
                break;
            }
        }
        // f32 should preserve sub-integer precision
        assert!(
            found_fractional,
            "f32 output should have fractional precision"
        );
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;

    #[test]
    fn test_read_info_includes_metadata_fields() {
        // Create minimal valid JPEG
        let jpeg = include_bytes!("../../tests/outputs/1_q85.jpg");

        let decoder = Decoder::new();
        let info = decoder.read_info(jpeg);

        // Should successfully parse
        assert!(info.is_ok(), "read_info should succeed on valid JPEG");

        let info = info.unwrap();

        // New fields should exist (even if None)
        // This tests that the struct has been extended with metadata fields
        let _ = &info.icc_profile;
        let _ = &info.exif;
        let _ = &info.xmp;

        println!("✅ JpegInfo includes icc_profile, exif, and xmp fields");
    }

    #[test]
    fn test_read_info_extracts_metadata() {
        // Test with UltraHDR sample which has XMP
        let jpeg = include_bytes!("../../tests/images/ultrahdr_sample.jpg");

        let decoder = Decoder::new();
        let info = decoder
            .read_info(jpeg)
            .expect("Should decode ultrahdr_sample.jpg");

        // Check that metadata extraction doesn't require full decode
        println!(
            "Image: {}x{}",
            info.dimensions.width, info.dimensions.height
        );

        if info.has_icc_profile {
            assert!(
                info.icc_profile.is_some(),
                "If has_icc_profile is true, icc_profile should be Some"
            );
            println!(
                "✅ ICC profile extracted: {} bytes",
                info.icc_profile.as_ref().unwrap().len()
            );
        }

        if let Some(ref xmp) = info.xmp {
            println!("✅ XMP extracted: {} chars", xmp.len());
        }

        if let Some(ref exif) = info.exif {
            println!("✅ EXIF extracted: {} bytes", exif.len());
        }
    }
}
#[cfg(test)]
mod limits_tests {
    use super::*;

    #[test]
    fn test_limits_builder_methods() {
        let config = Decoder::new()
            .max_pixels(50_000_000)
            .max_memory(256 * 1024 * 1024);

        assert_eq!(config.get_max_pixels(), 50_000_000);
        assert_eq!(config.get_max_memory(), 256 * 1024 * 1024);
    }

    #[test]
    fn test_limits_from_struct() {
        let limits = crate::types::Limits {
            max_pixels: Some(10_000_000),
            max_memory: Some(128 * 1024 * 1024),
            max_output: None,
        };

        let config = Decoder::new().limits(limits);

        assert_eq!(config.get_max_pixels(), 10_000_000);
        assert_eq!(config.get_max_memory(), 128 * 1024 * 1024);
    }

    #[test]
    fn test_max_memory_is_u64() {
        // Verify max_memory is u64 and doesn't require casting
        let large_limit: u64 = 5_000_000_000; // 5GB, would overflow on 32-bit if usize
        let config = Decoder::new().max_memory(large_limit);

        assert_eq!(config.get_max_memory(), large_limit);
    }

    #[test]
    fn test_fields_not_directly_accessible() {
        // This test verifies that fields are private and must use methods
        let config = Decoder::new();

        // These should compile (using getters)
        let _ = config.get_max_pixels();
        let _ = config.get_max_memory();

        // Direct field access should NOT compile (would fail if uncommented):
        // let _ = config.max_pixels;
        // let _ = config.max_memory;
    }

    #[test]
    fn test_max_scans_enforced() {
        use crate::encode::{EncoderConfig, PixelLayout};
        use crate::error::ErrorKind;
        use crate::foundation::alloc::MAX_SCANS;
        use enough::Unstoppable;

        // Encode a tiny 8x8 grayscale progressive JPEG
        let width = 8u32;
        let height = 8u32;
        let input = vec![128u8; (width * height) as usize];

        let config = EncoderConfig::grayscale(50.0).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Find the first SOS marker (0xFF 0xDA) and extract the scan segment
        // (SOS header + entropy-coded data up to the next marker).
        let mut first_sos = None;
        let mut first_scan_end = None;
        let mut i = 0;
        while i < jpeg.len() - 1 {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA && first_sos.is_none() {
                first_sos = Some(i);
                // Skip past the SOS marker and scan the entropy data
                // to find the next marker (0xFF followed by non-0x00).
                let mut j = i + 2;
                // Read the SOS segment length
                if j + 2 <= jpeg.len() {
                    let seg_len = ((jpeg[j] as usize) << 8) | (jpeg[j + 1] as usize);
                    j += seg_len; // Skip past the SOS segment header
                }
                // Now scan the entropy-coded data
                while j < jpeg.len() - 1 {
                    if jpeg[j] == 0xFF && jpeg[j + 1] != 0x00 {
                        first_scan_end = Some(j);
                        break;
                    }
                    j += 1;
                }
                break;
            }
            i += 1;
        }

        let sos_start = first_sos.expect("JPEG should contain an SOS marker");
        let scan_end = first_scan_end.expect("should find end of first scan");
        let scan_segment = &jpeg[sos_start..scan_end];

        // Build a malicious JPEG: header + (MAX_SCANS + 1) copies of the scan segment + EOI
        let header = &jpeg[..sos_start];
        let num_scans = MAX_SCANS + 1; // 257, exceeding the limit of 256

        let mut malicious = Vec::with_capacity(header.len() + scan_segment.len() * num_scans + 2);
        malicious.extend_from_slice(header);
        for _ in 0..num_scans {
            malicious.extend_from_slice(scan_segment);
        }
        malicious.push(0xFF);
        malicious.push(0xD9); // EOI

        // Attempt to decode — should fail with TooManyScans, not panic
        let decoder = Decoder::new();
        let result = decoder.decode(&malicious, Unstoppable);
        assert!(result.is_err(), "decoding should fail with too many scans");
        let err = result.unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::TooManyScans { .. }),
            "expected TooManyScans error, got: {err}",
        );
    }
}

#[cfg(test)]
mod quality_estimation_tests {
    use super::*;
    use crate::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
    use enough::Unstoppable;

    /// Helper: encode an RGB image at the given quality and return the JPEG bytes.
    fn encode_test_image(quality: f32, width: u32, height: u32) -> Vec<u8> {
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                input[idx] = ((x * 7 + y * 3) % 256) as u8;
                input[idx + 1] = ((x * 3 + y * 7 + 50) % 256) as u8;
                input[idx + 2] = ((x * 5 + y * 5 + 100) % 256) as u8;
            }
        }
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation");
        enc.push_packed(&input, Unstoppable).expect("push");
        enc.finish().expect("finish")
    }

    #[test]
    fn test_extras_quality_estimate_available() {
        let jpeg = encode_test_image(75.0, 64, 64);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras should be present");

        let estimate = extras
            .quality_estimate()
            .expect("quality estimate should be available");
        // zenjpeg uses jpegli tables, so the estimate should be on the butteraugli scale
        assert!(
            estimate.value > 0.0,
            "quality estimate value should be positive"
        );
    }

    #[test]
    fn test_extras_encoder_detected_as_jpegli() {
        let jpeg = encode_test_image(80.0, 64, 64);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");

        let encoder = extras.encoder().expect("encoder should be detected");
        // zenjpeg uses jpegli internally — should be detected as CjpegliYcbcr
        assert!(
            matches!(
                encoder,
                crate::detect::EncoderFamily::CjpegliYcbcr
                    | crate::detect::EncoderFamily::CjpegliXyb
            ),
            "expected jpegli encoder family, got: {encoder:?}"
        );
    }

    #[test]
    fn test_extras_dqt_tables_present() {
        let jpeg = encode_test_image(85.0, 64, 64);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");

        let tables = extras.dqt_tables().expect("DQT tables should be present");
        // Color images should have at least 2 DQT tables (jpegli uses 3)
        assert!(
            tables.len() >= 2,
            "expected at least 2 DQT tables, got {}",
            tables.len()
        );
    }

    #[test]
    fn test_extras_luminance_qt_present() {
        let jpeg = encode_test_image(90.0, 64, 64);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");

        let luma_qt = extras
            .luminance_qt()
            .expect("luminance QT should be present");
        // All values should be >= 1 (quantization table values are at least 1)
        for &v in luma_qt {
            assert!(v >= 1, "QT value should be >= 1, got {v}");
        }
        // At Q90, values should be relatively small
        assert!(
            luma_qt[0] < 20,
            "DC quant at Q90 should be small, got {}",
            luma_qt[0]
        );
    }

    #[test]
    fn test_extras_probe_full_access() {
        let jpeg = encode_test_image(75.0, 64, 64);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");

        let probe = extras.probe().expect("probe should be present");
        assert_eq!(probe.dimensions.width, 64);
        assert_eq!(probe.dimensions.height, 64);
        assert_eq!(probe.num_components, 3);
        assert!(probe.scan_count >= 1);
    }

    #[test]
    fn test_extras_quality_range_sweep() {
        // Encode at various qualities and verify estimates are in reasonable range
        for q in [10.0, 25.0, 50.0, 75.0, 90.0, 95.0] {
            let jpeg = encode_test_image(q, 32, 32);
            let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
            let extras = decoded.extras().expect("extras");
            let estimate = extras
                .quality_estimate()
                .expect("quality estimate should be available");

            // The estimate should be a finite positive number
            assert!(
                estimate.value.is_finite() && estimate.value > 0.0,
                "Q{q}: estimate should be finite positive, got {}",
                estimate.value
            );
        }
    }

    #[test]
    fn test_extras_higher_quality_means_smaller_qt_values() {
        // Higher quality should produce smaller quantization table values
        let jpeg_low = encode_test_image(50.0, 32, 32);
        let jpeg_high = encode_test_image(95.0, 32, 32);

        let low = Decoder::new()
            .decode(&jpeg_low, Unstoppable)
            .expect("decode");
        let high = Decoder::new()
            .decode(&jpeg_high, Unstoppable)
            .expect("decode");

        let low_qt = low.extras().unwrap().luminance_qt().unwrap();
        let high_qt = high.extras().unwrap().luminance_qt().unwrap();

        // Sum of QT values should be larger for lower quality
        let low_sum: u64 = low_qt.iter().map(|&v| v as u64).sum();
        let high_sum: u64 = high_qt.iter().map(|&v| v as u64).sum();
        assert!(
            low_sum > high_sum,
            "Q50 QT sum ({low_sum}) should be > Q95 QT sum ({high_sum})"
        );
    }

    #[test]
    fn test_extras_chrominance_qt_present_for_color() {
        let jpeg = encode_test_image(80.0, 32, 32);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");

        // Color image should have chrominance QT
        let chroma_qt = extras.chrominance_qt();
        assert!(
            chroma_qt.is_some(),
            "chrominance QT should be present for color image"
        );
    }

    #[test]
    fn test_extras_grayscale_has_luminance_qt() {
        // Encode a grayscale image
        let width = 32u32;
        let height = 32u32;
        let mut input = vec![0u8; (width * height) as usize];
        for i in 0..input.len() {
            input[i] = (i % 256) as u8;
        }
        let config = EncoderConfig::grayscale(80.0);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .expect("encoder");
        enc.push_packed(&input, Unstoppable).expect("push");
        let jpeg = enc.finish().expect("finish");

        let decoded = Decoder::new()
            .output_format(PixelFormat::Gray)
            .decode(&jpeg, Unstoppable)
            .expect("decode");
        let extras = decoded.extras().expect("extras");

        // Grayscale should have luminance QT
        assert!(extras.luminance_qt().is_some());

        // Probe should report 1 component
        let probe = extras.probe().expect("probe");
        assert_eq!(probe.num_components, 1, "grayscale should have 1 component");
    }

    #[test]
    fn test_extras_quality_estimate_confidence() {
        // Build a synthetic JPEG with known IJG Q75 tables and verify exact match
        let jpeg = build_ijg_synthetic_jpeg(75);
        let probe = crate::detect::probe(&jpeg).expect("probe should work");
        assert_eq!(probe.quality.confidence, crate::detect::Confidence::Exact,);
        assert_eq!(probe.quality.value, 75.0);
    }

    #[test]
    fn test_extras_ijg_quality_roundtrip() {
        // Build synthetic IJG JPEGs at various qualities and verify roundtrip
        for q in [10, 25, 50, 75, 85, 90, 95, 100] {
            let jpeg = build_ijg_synthetic_jpeg(q);
            let probe = crate::detect::probe(&jpeg).expect("probe");
            assert_eq!(
                probe.quality.value, q as f32,
                "IJG Q{q} should roundtrip exactly"
            );
            assert_eq!(probe.quality.confidence, crate::detect::Confidence::Exact,);
            assert_eq!(probe.quality.scale, crate::detect::QualityScale::IjgQuality,);
        }
    }

    /// Build a minimal synthetic JPEG with IJG tables at a given quality.
    /// Produces a valid header but minimal entropy data (not decodable to pixels).
    fn build_ijg_synthetic_jpeg(quality: u8) -> Vec<u8> {
        use crate::detect::generate_ijg_table;
        use crate::foundation::consts::{MARKER_DQT, MARKER_SOF0, MARKER_SOI, MARKER_SOS};

        let mut data = Vec::new();

        // SOI
        data.extend_from_slice(&[0xFF, MARKER_SOI]);

        // JFIF APP0
        data.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
        data.extend_from_slice(b"JFIF\0");
        data.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]);

        // DQT table 0 (luminance)
        let luma = generate_ijg_table(quality, false);
        data.extend_from_slice(&[0xFF, MARKER_DQT, 0x00, 0x43, 0x00]);
        for z in 0..64 {
            let ni = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(luma[ni] as u8);
        }

        // DQT table 1 (chrominance)
        let chroma = generate_ijg_table(quality, true);
        data.extend_from_slice(&[0xFF, MARKER_DQT, 0x00, 0x43, 0x01]);
        for z in 0..64 {
            let ni = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(chroma[ni] as u8);
        }

        // SOF0
        data.extend_from_slice(&[0xFF, MARKER_SOF0, 0x00, 0x11, 0x08]);
        data.extend_from_slice(&[0x00, 0x08, 0x00, 0x08]);
        data.push(0x03);
        data.extend_from_slice(&[0x01, 0x11, 0x00]);
        data.extend_from_slice(&[0x02, 0x11, 0x01]);
        data.extend_from_slice(&[0x03, 0x11, 0x01]);

        // Standard DHT tables (162 AC symbols each → libjpeg-turbo detection)
        // DC table 0
        data.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x00]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // AC table 0 (162 symbols)
        data.extend_from_slice(&[0xFF, 0xC4, 0x00, 0xB5, 0x10]);
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D,
        ]);
        for i in 0..162u8 {
            data.push(i);
        }

        // DC table 1
        data.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x01]);
        data.extend_from_slice(&[
            0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // AC table 1 (162 symbols)
        data.extend_from_slice(&[0xFF, 0xC4, 0x00, 0xB5, 0x11]);
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01,
            0x02, 0x77,
        ]);
        for i in 0..162u8 {
            data.push(i);
        }

        // SOS
        data.extend_from_slice(&[0xFF, MARKER_SOS, 0x00, 0x0C, 0x03]);
        data.extend_from_slice(&[0x01, 0x00, 0x02, 0x11, 0x03, 0x11]);
        data.extend_from_slice(&[0x00, 0x3F, 0x00]);
        data.push(0x00);

        // EOI
        data.extend_from_slice(&[0xFF, 0xD9]);

        data
    }

    #[test]
    fn test_extras_photoshop_detection() {
        // Build a synthetic JPEG with APP13 (Photoshop 3.0) + APP14 (Adobe) markers
        // and non-IJG tables to trigger Photoshop detection
        let jpeg = build_photoshop_jpeg();
        let probe = crate::detect::probe(&jpeg).expect("probe");
        assert_eq!(
            probe.encoder,
            crate::detect::EncoderFamily::Photoshop,
            "should detect as Photoshop"
        );
    }

    /// Build a synthetic JPEG that mimics Photoshop output:
    /// non-IJG DQT + APP13 Photoshop 3.0 + APP14 Adobe
    fn build_photoshop_jpeg() -> Vec<u8> {
        use crate::foundation::consts::{MARKER_DQT, MARKER_SOF0, MARKER_SOI, MARKER_SOS};

        let mut data = Vec::new();

        // SOI
        data.extend_from_slice(&[0xFF, MARKER_SOI]);

        // APP0 (JFIF)
        data.extend_from_slice(&[0xFF, 0xE0]);
        data.extend_from_slice(&[0x00, 0x10]);
        data.extend_from_slice(b"JFIF\0");
        data.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00]);

        // APP13 (Photoshop 3.0 IPTC)
        let ps_data = b"Photoshop 3.0\08BIM\x00";
        let ps_len = (ps_data.len() + 2) as u16;
        data.extend_from_slice(&[0xFF, 0xED]);
        data.push((ps_len >> 8) as u8);
        data.push((ps_len & 0xFF) as u8);
        data.extend_from_slice(ps_data);

        // APP14 (Adobe)
        data.extend_from_slice(&[0xFF, 0xEE]);
        data.extend_from_slice(&[0x00, 0x0E]); // Length 14
        data.extend_from_slice(b"Adobe");
        data.extend_from_slice(&[0x00, 0x64]); // Version 100
        data.extend_from_slice(&[0x00, 0x00]); // Flags0
        data.extend_from_slice(&[0x00, 0x00]); // Flags1
        data.push(0x01); // YCbCr color transform

        // DQT table 0 — non-IJG custom table (Photoshop quality 10)
        // These are NOT generated by the IJG formula
        let custom_luma: [u8; 64] = [
            2, 2, 2, 2, 3, 4, 5, 6, 2, 2, 2, 2, 3, 4, 5, 6, 2, 2, 2, 2, 4, 5, 7, 9, 2, 2, 2, 4, 5,
            7, 9, 12, 3, 3, 4, 5, 8, 10, 12, 12, 4, 4, 5, 7, 10, 12, 12, 12, 5, 5, 7, 9, 12, 12,
            12, 12, 6, 6, 9, 12, 12, 12, 12, 12,
        ];
        data.extend_from_slice(&[0xFF, MARKER_DQT]);
        data.extend_from_slice(&[0x00, 0x43]); // Length 67
        data.push(0x00); // Precision 0, table 0
        // Write in zigzag order
        for z in 0..64 {
            let natural_idx = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(custom_luma[natural_idx]);
        }

        // DQT table 1 — non-IJG custom chrominance
        let custom_chroma: [u8; 64] = [
            3, 3, 5, 9, 13, 15, 15, 15, 3, 4, 6, 11, 14, 12, 12, 12, 5, 6, 9, 14, 12, 12, 12, 12,
            9, 11, 14, 12, 12, 12, 12, 12, 13, 14, 12, 12, 12, 12, 12, 12, 15, 12, 12, 12, 12, 12,
            12, 12, 15, 12, 12, 12, 12, 12, 12, 12, 15, 12, 12, 12, 12, 12, 12, 12,
        ];
        data.extend_from_slice(&[0xFF, MARKER_DQT]);
        data.extend_from_slice(&[0x00, 0x43]);
        data.push(0x01);
        for z in 0..64 {
            let natural_idx = crate::foundation::consts::JPEG_NATURAL_ORDER[z] as usize;
            data.push(custom_chroma[natural_idx]);
        }

        // SOF0
        data.extend_from_slice(&[0xFF, MARKER_SOF0]);
        data.extend_from_slice(&[0x00, 0x11, 0x08]);
        data.extend_from_slice(&[0x00, 0x08, 0x00, 0x08]);
        data.push(0x03);
        data.extend_from_slice(&[0x01, 0x11, 0x00]); // Y
        data.extend_from_slice(&[0x02, 0x11, 0x01]); // Cb
        data.extend_from_slice(&[0x03, 0x11, 0x01]); // Cr

        // Minimal DHT + SOS + EOI
        // DHT DC table 0
        data.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x00]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // DHT AC table 0 (100 symbols — not 162 → triggers optimized Huffman)
        let ac_sym_count: u16 = 100;
        let ac_len = 2 + 1 + 16 + ac_sym_count;
        data.extend_from_slice(&[0xFF, 0xC4]);
        data.push((ac_len >> 8) as u8);
        data.push((ac_len & 0xFF) as u8);
        data.push(0x10);
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x3F,
        ]);
        for i in 0..100u8 {
            data.push(i);
        }

        // DHT DC table 1
        data.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x01]);
        data.extend_from_slice(&[
            0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ]);
        data.extend_from_slice(&[
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        ]);

        // DHT AC table 1
        data.extend_from_slice(&[0xFF, 0xC4]);
        data.push((ac_len >> 8) as u8);
        data.push((ac_len & 0xFF) as u8);
        data.push(0x11);
        data.extend_from_slice(&[
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x3F,
        ]);
        for i in 0..100u8 {
            data.push(i);
        }

        // SOS
        data.extend_from_slice(&[0xFF, MARKER_SOS]);
        data.extend_from_slice(&[0x00, 0x0C, 0x03]);
        data.extend_from_slice(&[0x01, 0x00, 0x02, 0x11, 0x03, 0x11]);
        data.extend_from_slice(&[0x00, 0x3F, 0x00]);
        data.push(0x00);

        // EOI
        data.extend_from_slice(&[0xFF, 0xD9]);

        data
    }

    #[test]
    fn test_extras_quality_q1_edge_case() {
        // Q1 = maximum compression, highest quantization values
        let jpeg = encode_test_image(1.0, 32, 32);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");
        let estimate = extras.quality_estimate().expect("quality estimate");
        assert!(
            estimate.value > 0.0,
            "Q1 quality estimate should be positive"
        );
    }

    #[test]
    fn test_extras_quality_q100_edge_case() {
        // Q100 = near-lossless, lowest quantization values
        let jpeg = encode_test_image(100.0, 32, 32);
        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras = decoded.extras().expect("extras");

        let luma_qt = extras.luminance_qt().expect("luma QT");
        // At Q100, most quantization values should be very small (1-2)
        let all_ones = luma_qt.iter().all(|&v| v <= 2);
        assert!(all_ones, "Q100 QT should have very small values");
    }

    #[test]
    fn test_extras_probe_matches_standalone_probe() {
        // Verify that the probe result stored in extras matches
        // a standalone probe on the same JPEG bytes
        let jpeg = encode_test_image(75.0, 64, 64);

        let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");
        let extras_probe = decoded.extras().unwrap().probe().expect("probe in extras");

        let standalone_probe = crate::detect::probe(&jpeg).expect("standalone probe");

        assert_eq!(extras_probe.encoder, standalone_probe.encoder);
        assert_eq!(extras_probe.quality.value, standalone_probe.quality.value);
        assert_eq!(extras_probe.quality.scale, standalone_probe.quality.scale);
        assert_eq!(
            extras_probe.quality.confidence,
            standalone_probe.quality.confidence
        );
        assert_eq!(
            extras_probe.dimensions.width,
            standalone_probe.dimensions.width
        );
        assert_eq!(
            extras_probe.dimensions.height,
            standalone_probe.dimensions.height
        );
        assert_eq!(
            extras_probe.dqt_tables.len(),
            standalone_probe.dqt_tables.len()
        );
    }
}
