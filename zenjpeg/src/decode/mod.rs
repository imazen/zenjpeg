//! JPEG decoder implementation.
//!
//! This module provides the main decoder interface for reading JPEG images.
//!
//! # Quick Start
//!
//! ```ignore
//! use zenjpeg::decode::DecodeConfig;
//!
//! let result = DecodeConfig::new().decode(&jpeg_data, enough::Unstoppable)?;
//! let pixels: &[u8] = result.pixels_u8().unwrap();
//! ```
//!
//! # ICC Profile Support
//!
//! The decoder can extract and apply embedded ICC profiles, including XYB profiles
//! used by jpegli. ICC profile support requires enabling `cms-lcms2` or `cms-moxcms` feature.
//!
//! ```ignore
//! use zenjpeg::decode::DecodeConfig;
//!
//! let result = DecodeConfig::new().apply_icc(true).decode(&jpeg_data, enough::Unstoppable)?;
//! ```

// IDCT modules (decoder-only)
#[doc(hidden)]
pub mod idct;
#[doc(hidden)]
pub mod idct_int;

mod config;
mod extras;
mod image;
mod parser;
mod pipeline;
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
pub use config::{DecodeConfig, DecodeInfo, DecodeResult, GainMapHandling, GainMapResult, OutputTarget};
use parser::JpegParser;

pub use scanline::{ScanlineInfo, ScanlineReader};

// UltraHDR streaming reader
#[cfg(feature = "ultrahdr")]
#[allow(unused_imports)] // Re-exports for public API
pub use ultrahdr_reader::{GainMapMemory, UltraHdrMode, UltraHdrReader, UltraHdrReaderConfig};

// Re-export extras types for public API
#[allow(unused_imports)]
pub use extras::{
    AdobeColorTransform, AdobeInfo, DecodedExtras, DensityUnits, IccPreserve, JfifInfo,
    MpfDirectory, MpfEntry, MpfImageType, PreserveConfig, PreservedMpfImage, PreservedSegment,
    SegmentType, StandardProfile,
};

// Re-export types used in public struct fields so users can access them
pub use crate::types::{ColorSpace, Dimensions, JpegMode, PixelFormat};
use crate::types::{Component, Subsampling};

// Re-export Stop trait for cancellation support
pub use enough::Stop;
use enough::Unstoppable;

use crate::error::{Error, Result};
use crate::foundation::consts::MAX_COMPONENTS;

/// Compute subsampling mode from component sampling factors.
fn compute_subsampling(
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
pub use config::{ChromaUpsampling, DecodeWarning, DecoderConfig, JpegInfo, Strictness};

/// Backward compatibility alias: `Decoder` is now [`DecodeConfig`].
pub type Decoder = DecodeConfig;

#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
use crate::color::icc::apply_icc_transform;

impl DecodeConfig {
    /// Creates a new decoder configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
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
    /// - [`ChromaUpsampling::Triangle`] (default): jpegli-style separable filter
    /// - [`ChromaUpsampling::LibjpegCompat`]: exact libjpeg-turbo/mozjpeg match
    /// - [`ChromaUpsampling::NearestNeighbor`]: fastest, lowest quality
    #[must_use]
    pub fn chroma_upsampling(mut self, method: ChromaUpsampling) -> Self {
        self.chroma_upsampling = method;
        self
    }

    /// Enables or disables fancy (triangle filter) upsampling.
    ///
    /// This is a convenience method for backwards compatibility.
    /// `true` maps to [`ChromaUpsampling::Triangle`],
    /// `false` maps to [`ChromaUpsampling::NearestNeighbor`].
    ///
    /// For more control, use [`chroma_upsampling()`](Self::chroma_upsampling).
    #[must_use]
    pub fn fancy_upsampling(mut self, enable: bool) -> Self {
        self.chroma_upsampling = if enable {
            ChromaUpsampling::Triangle
        } else {
            ChromaUpsampling::NearestNeighbor
        };
        self
    }

    /// Enables block smoothing.
    #[must_use]
    pub fn block_smoothing(mut self, enable: bool) -> Self {
        self.block_smoothing = enable;
        self
    }

    /// Enables ICC profile application.
    ///
    /// When enabled, embedded ICC profiles will be applied to convert
    /// the image to sRGB. This is required for correct display of
    /// XYB-encoded images.
    ///
    /// Note: Requires `cms-lcms2` or `cms-moxcms` feature to be enabled.
    /// Without a CMS feature, this setting has no effect.
    #[must_use]
    pub fn apply_icc(mut self, enable: bool) -> Self {
        self.apply_icc = enable;
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
    pub fn max_memory(mut self, bytes: usize) -> Self {
        self.max_memory = bytes;
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

    /// Convenience: preserve nothing extra (minimal memory).
    #[must_use]
    pub fn preserve_none(self) -> Self {
        self.preserve(PreserveConfig::none())
    }

    /// Convenience: preserve everything.
    #[must_use]
    pub fn preserve_all(self) -> Self {
        self.preserve(PreserveConfig::all())
    }

    /// Sets the strictness level for error handling.
    ///
    /// - [`Strictness::Strict`]: Fail on any spec violation or truncation
    /// - [`Strictness::Balanced`]: Reject violations, recover from truncation (default)
    /// - [`Strictness::Lenient`]: Recover from all errors when possible
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

    /// Convenience: use lenient mode (maximum compatibility).
    #[must_use]
    pub fn lenient(self) -> Self {
        self.strictness(Strictness::Lenient)
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

    /// Reads JPEG info without decoding.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            None,
            self.strictness,
        )?;
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
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            None,
            self.strictness,
        )?;
        parser.read_header()?;

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

            // Compute subsampling from sampling factors
            let subsampling = compute_subsampling(&parser.components, num_components);

            // Fully decode the image (scanline reader doesn't support cancellation)
            parser.decode(&Unstoppable)?;

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
                self.output_target.uses_dequant_bias(),
                &Unstoppable,
            )?;

            return Ok(ScanlineReader::new_buffered(
                data,
                width,
                height,
                num_components,
                subsampling,
                pixels,
                is_xyb,
            ));
        }

        // Baseline: use streaming mode
        // Check for high sampling factors (>2x2) which need buffered mode
        let max_h = parser.components[..parser.num_components as usize]
            .iter()
            .map(|c| c.h_samp_factor)
            .max()
            .unwrap_or(1);
        let max_v = parser.components[..parser.num_components as usize]
            .iter()
            .map(|c| c.v_samp_factor)
            .max()
            .unwrap_or(1);

        if max_h > 2 || max_v > 2 {
            let width = parser.width;
            let height = parser.height;
            let num_components = parser.num_components;
            let subsampling = subsampling_from_max(max_h, max_v, is_grayscale);

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
                self.output_target.uses_dequant_bias(),
                &Unstoppable,
            )?;

            return Ok(ScanlineReader::new_buffered(
                data,
                width,
                height,
                num_components,
                subsampling,
                pixels,
                is_xyb,
            ));
        }

        // Extract scan data and construct scanline reader
        let scan_data = parser.into_scan_data(is_grayscale)?;
        ScanlineReader::from_scan_data(scan_data, self.chroma_upsampling)
    }

    /// Decodes a JPEG image.
    ///
    /// For large images or memory-constrained environments, consider using
    /// [`scanline_reader()`](Self::scanline_reader) to decode row-by-row
    /// into caller-provided buffers.
    pub fn decode(&self, data: &[u8], stop: impl Stop) -> Result<DecodedImage> {
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            Some(&self.preserve),
            self.strictness,
        )?;
        parser.decode(&stop)?;

        let info = parser.info();
        let output_format = self.output_format.unwrap_or(PixelFormat::Rgb);

        // Convert to output format
        // For XYB images, use simple dequantization so ICC profile works correctly
        #[allow(unused_mut)] // pixels is mutated when cms features are enabled
        let mut pixels = parser.to_pixels(
            output_format,
            info.is_xyb,
            self.chroma_upsampling,
            self.output_target.uses_dequant_bias(),
            &stop,
        )?;

        // Apply ICC profile if enabled and present
        // Note: ICC transform failures are non-fatal - we fall back to un-color-managed pixels
        // rather than failing the decode, since the JPEG itself decoded successfully
        #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
        if self.apply_icc && output_format == PixelFormat::Rgb {
            if let Some(ref icc_profile) = parser.icc_profile {
                match apply_icc_transform(
                    &pixels,
                    info.dimensions.width as usize,
                    info.dimensions.height as usize,
                    icc_profile,
                ) {
                    Ok(transformed) => pixels = transformed,
                    Err(_e) => {
                        // ICC transform failed - continue with un-color-managed pixels
                        // This can happen with unusual profiles that CMS libraries don't support
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Warning: ICC profile transform failed, using original colors: {_e:?}"
                        );
                    }
                }
            }
        }

        // Extract preserved extras and warnings
        let extras = parser.take_extras();
        let warnings = parser.take_warnings();

        Ok(DecodedImage {
            width: info.dimensions.width,
            height: info.dimensions.height,
            format: output_format,
            data: pixels,
            extras,
            warnings,
        })
    }

    /// Decodes a JPEG image to 32-bit floating point pixels.
    ///
    /// This preserves the full 12-bit internal precision of jpegli's decoder
    /// without quantization to 8-bit. Values are normalized to range 0.0-1.0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use zenjpeg::decode::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let image = decoder.decode_f32(&jpeg_data)?;
    /// // image.data contains f32 values in range 0.0-1.0
    /// ```
    ///
    /// Note: ICC profile application is not supported for f32 output.
    /// If you need ICC profile transformation, decode to u8 first.
    ///
    /// For large images, consider using streaming APIs for memory-efficient decoding.
    pub fn decode_f32(&self, data: &[u8], stop: impl Stop) -> Result<DecodedImageF32> {
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            None,
            self.strictness,
        )?;
        // Disable streaming - f32 decode needs coefficients for precision
        parser.prefer_streaming = false;
        parser.decode(&stop)?;

        let info = parser.info();
        let output_format = self.output_format.unwrap_or(PixelFormat::Rgb);

        // Convert to output format as f32
        let pixels = parser.to_pixels_f32(
            output_format,
            info.is_xyb,
            self.chroma_upsampling,
            &stop,
        )?;

        let warnings = parser.take_warnings();

        Ok(DecodedImageF32 {
            width: info.dimensions.width,
            height: info.dimensions.height,
            format: output_format,
            data: pixels,
            warnings,
        })
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
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            None,
            self.strictness,
        )?;
        // Disable streaming - we need coefficients stored
        parser.prefer_streaming = false;
        parser.decode(&stop)?;

        // Extract coefficients from parser
        parser.extract_coefficients()
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
        let mut parser = JpegParser::with_strictness(
            data,
            self.max_pixels,
            None,
            self.strictness,
        )?;
        // Disable streaming - f32 YCbCr decode needs coefficients
        parser.prefer_streaming = false;
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
    /// For a 4K image (3840×2160):
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
            parser.take_extras()
        } else {
            None
        };

        UltraHdrReader::new(data, config, base_reader, extras, gainmap_range, metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
    use enough::Unstoppable;

    #[test]
    fn test_decoder_creation() {
        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .fancy_upsampling(true);

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
        assert_eq!(decoded.data.len(), (width * height) as usize);

        // Check pixel values are reasonably close (JPEG is lossy)
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.data[i] as i32).abs();
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
        assert_eq!(decoded.data.len(), (width * height * 3) as usize);

        // Check pixel values are reasonably close
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.data[i] as i32).abs();
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
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded_f32 = decoder
            .decode_f32(&jpeg, Unstoppable)
            .expect("f32 decoding should succeed");

        assert_eq!(decoded_f32.width, width);
        assert_eq!(decoded_f32.height, height);
        assert_eq!(decoded_f32.data.len(), (width * height * 3) as usize);

        // Verify values are approximately in 0.0-1.0 range.
        // YCbCr→RGB color matrix can produce values slightly outside [0, 1]
        // due to ringing — this is intentional to preserve full precision.
        for &v in &decoded_f32.data {
            assert!(
                (-0.05..=1.05).contains(&v),
                "f32 value {} too far out of range",
                v
            );
        }

        // Compare with u8 decode - converted f32 should match
        let decoded_u8 = decoder
            .decode(&jpeg, Unstoppable)
            .expect("u8 decoding should succeed");
        let converted_u8 = decoded_f32.to_u8();

        // Values should be close - allow diff of 2 because u8 path uses integer IDCT
        // while f32 path uses f32 IDCT (standard JPEG precision difference)
        let mut max_diff = 0i32;
        for i in 0..decoded_u8.data.len() {
            let diff = (decoded_u8.data[i] as i32 - converted_u8.data[i] as i32).abs();
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
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded_f32 = decoder
            .decode_f32(&jpeg, Unstoppable)
            .expect("f32 decoding should succeed");

        // Check that f32 values show more precision than just u8/255
        // by verifying we have non-quantized intermediate values
        let mut found_fractional = false;
        for &v in &decoded_f32.data {
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
