//! Decoder configuration types.
//!
//! This module contains the configuration enums and structs used to control
//! JPEG decoding behavior.

use crate::foundation::alloc::{DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS};
use crate::types::Dimensions;

use super::extras::{DecodedExtras, PreserveConfig};

/// Chroma upsampling method for subsampled JPEG images (4:2:0, 4:2:2, 4:4:0).
///
/// This controls how chroma (Cb/Cr) channels are upsampled to match luma (Y)
/// resolution during decoding. Different methods produce slightly different
/// pixel values, which matters for exact decoder matching.
///
/// # Compatibility
///
/// | Method | Matches |
/// |--------|---------|
/// | `Triangle` | jpegli (default zenjpeg behavior) |
/// | `LibjpegCompat` | libjpeg-turbo, mozjpeg, djpeg |
/// | `NearestNeighbor` | fastest, lowest quality |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ChromaUpsampling {
    /// Pixel replication (box filter). Fastest, lowest quality.
    ///
    /// Each chroma sample is duplicated to fill the corresponding output pixels.
    /// No interpolation is performed.
    NearestNeighbor,

    /// Separable triangle filter with uniform `+2` rounding bias (jpegli-style).
    ///
    /// Applies horizontal then vertical 3:1 interpolation. This is the default
    /// and matches jpegli's upsampling behavior.
    #[default]
    Triangle,

    /// Exact libjpeg-turbo/mozjpeg compatible triangle filter.
    ///
    /// Uses a fused 2D filter for h2v2 (not separable) with alternating rounding
    /// bias (`+1`/`+2` for 1D, `+7`/`+8` for 2D). This avoids both systematic
    /// rounding bias and intermediate rounding errors from separable passes.
    ///
    /// Use this mode when you need pixel-exact match with `djpeg` or mozjpeg output.
    LibjpegCompat,
}

/// Controls how the decoder handles non-fatal errors.
///
/// The default is [`Strictness::Balanced`], which matches mozjpeg/libjpeg-turbo
/// behavior: data errors (truncation, missing padding, DNL conflicts) recover
/// gracefully, and missing DHT falls back to standard tables (for MJPEG compat).
///
/// Use [`Strictness::Strict`] for validation/conformance testing,
/// [`Strictness::Lenient`] for maximum compatibility.
///
/// # Behavior matrix
///
/// | Situation | ITU-T T.81 spec | mozjpeg | Strict | Balanced | Lenient |
/// |---|---|---|---|---|---|
/// | Truncated scan data | Invalid | JWRN_HIT_MARKER (fill 0) | Error | Fill zeros | Fill zeros |
/// | Missing padding blocks | Invalid (MCUs required) | Implicit zero fill | Error | Speculative+zero | Speculative+zero |
/// | DNL conflicts with SOF | Invalid (B.2.5) | Ignored entirely | Error | Ignored | Ignored |
/// | Bad Huffman at end-of-scan | Invalid | JWRN_HUFF_BAD_CODE (use 0) | Error | EndOfScan | EndOfScan |
/// | Missing DHT before scan | Invalid (B.2.4.2) | std_huff_tables() fallback | Error | Std tables | Std tables |
/// | Progressive scan truncated | Invalid | JWRN_HIT_MARKER (fill 0) | Error | Fill zeros | Fill zeros |
/// | AC index overflow | Invalid | ERREXIT (fatal) | Error | Error | Treat as EOB |
/// | Invalid Huffman mid-scan | Invalid | ERREXIT (fatal) | Error | Error | Treat as EOB |
/// | Bad DQT/DHT structure | Invalid | ERREXIT (fatal) | Error | Error | Error |
/// | Bad component ID in SOS | Invalid (B.2.3) | ERREXIT (fatal) | Error | Error | Error |
///
/// "Speculative+zero" means: attempt to decode the block; if the data is
/// missing or invalid, restore decoder state and fill with zeros.
///
/// Note on missing DHT: mozjpeg calls `std_huff_tables()` in `jinit_huff_decoder()`
/// before decode begins, automatically filling any missing tables with ITU-T T.81
/// section K.3 standard tables. This is specifically for MJPEG/AVI1 compatibility.
/// Balanced matches this behavior. Only Strict rejects missing DHT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Strictness {
    /// Fail on any spec violation, truncation, or recoverable error.
    ///
    /// Stricter than mozjpeg. Errors on everything mozjpeg would warn about.
    ///
    /// Use for:
    /// - Validation/conformance testing
    /// - When partial results are worse than no result
    /// - Quality assurance pipelines
    Strict,

    /// Match mozjpeg/libjpeg-turbo error handling behavior (default).
    ///
    /// Recovers from data errors (like mozjpeg's WARNMS):
    /// - Truncated scan data (fills remaining with zeros)
    /// - Missing padding blocks (speculative decode, zero fill)
    /// - DNL marker conflicts with SOF height (ignored)
    /// - End-of-scan fill bits that don't form valid Huffman codes
    /// - Missing DHT markers (falls back to ITU-T T.81 K.3 standard tables)
    ///
    /// Errors on structural violations (like mozjpeg's ERREXIT):
    /// - Bad DQT/DHT table structure
    /// - Bad component ID in SOS
    ///
    /// Use for:
    /// - General image processing
    /// - Production pipelines expecting mozjpeg-compatible behavior
    /// - MJPEG streams (which often omit DHT markers)
    #[default]
    Balanced,

    /// Recover from all errors when possible.
    ///
    /// Goes beyond mozjpeg's error handling with additional recovery:
    /// - AC coefficient index overflow (treated as end-of-block)
    /// - Invalid Huffman codes mid-scan (treated as end-of-block)
    ///
    /// Use for:
    /// - Corrupt file recovery
    /// - Maximum compatibility
    /// - Forensic analysis of damaged files
    Lenient,
}

/// Issues discovered during JPEG decoding.
///
/// In [`Strictness::Strict`] mode, any issue triggers an immediate error
/// (the variant is embedded in the error message for programmatic matching).
///
/// In [`Strictness::Balanced`] and [`Strictness::Lenient`] modes, issues are
/// collected as warnings and accessible via [`DecodeResult::warnings()`].
///
/// This allows the same enum to serve as both warning data and error context.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decoder::{Decoder, DecodeWarning};
///
/// let result = Decoder::new().decode(&data, enough::Unstoppable)?;
/// for warning in result.warnings() {
///     match warning {
///         DecodeWarning::MissingHuffmanTables => eprintln!("MJPEG: used standard tables"),
///         DecodeWarning::TruncatedScan { .. } => eprintln!("Scan data was truncated"),
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeWarning {
    /// No DHT markers found; standard Huffman tables (ITU-T T.81 K.3) were used.
    ///
    /// Common in MJPEG/AVI1 frames which omit DHT to save space.
    /// mozjpeg handles this via `std_huff_tables()` in `jinit_huff_decoder()`.
    MissingHuffmanTables,

    /// Scan data was truncated; remaining blocks filled with zeros.
    ///
    /// The image may have partial content. `blocks_decoded` and `blocks_expected`
    /// indicate how much data was recovered.
    TruncatedScan {
        /// Number of MCU blocks successfully decoded before truncation.
        blocks_decoded: u32,
        /// Total number of MCU blocks expected for this scan.
        blocks_expected: u32,
    },

    /// Padding blocks beyond image boundary couldn't be decoded; filled with zeros.
    ///
    /// When the image dimensions aren't MCU-aligned, padding blocks are needed.
    /// If the entropy data doesn't contain valid padding, zeros are used.
    PaddingBlockError,

    /// DNL marker height conflicts with SOF header height; DNL value ignored.
    ///
    /// Per ITU-T T.81, DNL is only valid when SOF height is 0. mozjpeg ignores
    /// DNL entirely (skip_variable).
    DnlHeightConflict {
        /// Height from the SOF marker.
        sof_height: u32,
        /// Height from the DNL marker (ignored).
        dnl_height: u32,
    },

    /// Progressive scan data was truncated; remaining coefficients filled with zeros.
    TruncatedProgressiveScan,

    /// AC coefficient index exceeded block bounds; treated as end-of-block.
    ///
    /// Only recovered in Lenient mode. Indicates malformed run-length data
    /// where the run + position would exceed the 64-coefficient block.
    AcIndexOverflow,

    /// Invalid Huffman code encountered; treated as end-of-block.
    ///
    /// Only recovered in Lenient mode. Indicates corrupted entropy data
    /// where a bit sequence doesn't match any valid Huffman code.
    InvalidHuffmanCode,
}

impl core::fmt::Display for DecodeWarning {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingHuffmanTables => {
                write!(f, "missing DHT markers; standard Huffman tables used")
            }
            Self::TruncatedScan {
                blocks_decoded,
                blocks_expected,
            } => write!(
                f,
                "scan truncated at block {}/{}; remaining filled with zeros",
                blocks_decoded, blocks_expected
            ),
            Self::PaddingBlockError => {
                write!(f, "padding block decode failed; filled with zeros")
            }
            Self::DnlHeightConflict {
                sof_height,
                dnl_height,
            } => write!(
                f,
                "DNL height {} conflicts with SOF height {}; DNL ignored",
                dnl_height, sof_height
            ),
            Self::TruncatedProgressiveScan => {
                write!(
                    f,
                    "progressive scan truncated; remaining coefficients are zero"
                )
            }
            Self::AcIndexOverflow => {
                write!(f, "AC index overflow; treated as end-of-block")
            }
            Self::InvalidHuffmanCode => {
                write!(f, "invalid Huffman code; treated as end-of-block")
            }
        }
    }
}

// ============================================================================
// OutputTarget — pixel format + transfer function + precision
// ============================================================================

/// Controls the output pixel format, precision, and transfer function.
///
/// This determines the IDCT variant used, whether the output is u8 or f32,
/// and whether sRGB linearization is applied.
///
/// # Variants
///
/// | Variant | Type | Transfer | IDCT | Speed |
/// |---------|------|----------|------|-------|
/// | `Srgb8` | u8 | sRGB gamma | Clamped integer | Fastest |
/// | `SrgbF32` | f32 | sRGB gamma | Unclamped integer | ~same |
/// | `LinearF32` | f32 | Linear light | Unclamped integer + linearize | ~same |
/// | `SrgbF32Precise` | f32 | sRGB gamma | f32 + Laplacian biases | ~1.5-2x slower |
/// | `LinearF32Precise` | f32 | Linear light | f32 + Laplacian biases | ~1.5-2x slower |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OutputTarget {
    /// u8 sRGB output with clamped integer IDCT. Fastest path. Default.
    ///
    /// Values are clamped to [0, 255]. This is the standard decode path
    /// matching libjpeg-turbo / zune-jpeg behavior.
    #[default]
    Srgb8,

    /// f32 sRGB output (gamma-encoded, [0,1] nominal) with unclamped integer IDCT.
    ///
    /// Same speed as `Srgb8` but preserves ringing outside [0, 255] as values
    /// outside [0.0, 1.0]. Useful for high-quality resampling or compositing
    /// where clamping artifacts would be visible.
    SrgbF32,

    /// f32 linear light output ([0,1] nominal) with unclamped integer IDCT.
    ///
    /// Same as `SrgbF32` but applies sRGB→linear transfer function after
    /// color conversion. Use for physically-correct blending, compositing,
    /// or machine learning pipelines that expect linear input.
    LinearF32,

    /// f32 sRGB output with Laplacian dequantization biases (Price & Rabbani 2000).
    ///
    /// Uses f32 IDCT with per-coefficient biases computed from DCT statistics.
    /// Produces measurably higher quality reconstruction at the cost of
    /// ~1.5-2x slower decoding. Closely matches C++ jpegli decoder behavior.
    SrgbF32Precise,

    /// f32 linear light output with Laplacian dequantization biases.
    ///
    /// Combines the quality benefits of `SrgbF32Precise` with linear-light output.
    /// Best reconstruction quality available.
    LinearF32Precise,
}

impl OutputTarget {
    /// Returns `true` if output is f32 (any variant except `Srgb8`).
    #[inline]
    #[must_use]
    pub fn is_f32(self) -> bool {
        !matches!(self, Self::Srgb8)
    }

    /// Returns `true` if output is in linear light.
    #[inline]
    #[must_use]
    pub fn is_linear(self) -> bool {
        matches!(self, Self::LinearF32 | Self::LinearF32Precise)
    }

    /// Returns `true` if using Laplacian dequantization biases.
    #[inline]
    #[must_use]
    pub fn is_precise(self) -> bool {
        matches!(self, Self::SrgbF32Precise | Self::LinearF32Precise)
    }

    /// Returns `true` if the IDCT should skip [0, 255] clamping.
    #[inline]
    pub(crate) fn needs_unclamped_idct(self) -> bool {
        // f32 output benefits from unclamped IDCT to preserve ringing precision.
        // Precise variants use f32 IDCT entirely (not integer), so this is only
        // relevant for SrgbF32 and LinearF32.
        matches!(self, Self::SrgbF32 | Self::LinearF32)
    }

    /// Returns `true` if this target uses dequant biases (same as `is_precise`).
    #[inline]
    pub(crate) fn uses_dequant_bias(self) -> bool {
        self.is_precise()
    }
}

// ============================================================================
// GainMapHandling — UltraHDR gain map control
// ============================================================================

/// Controls how UltraHDR gain maps are handled during decoding.
///
/// Regular JPEGs without gain maps are unaffected by this setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GainMapHandling {
    /// Ignore any gain map data. Default.
    #[default]
    Discard,

    /// Preserve the raw gain map JPEG bytes and parsed XMP metadata.
    ///
    /// The gain map JPEG is extracted but not decoded to pixels.
    /// Use this when you need to re-embed or forward the gain map.
    PreserveRaw,

    /// Decode the gain map to pixel data in addition to preserving raw bytes.
    ///
    /// This is the most expensive option — it decodes both the base image
    /// and the gain map JPEG.
    Decode,
}

/// Decoded gain map from an UltraHDR image.
#[derive(Debug, Clone)]
pub struct GainMapResult {
    /// Raw gain map JPEG bytes.
    pub jpeg: Vec<u8>,
    /// Decoded gain map pixels (RGB u8). Only present if [`GainMapHandling::Decode`].
    pub pixels: Option<Vec<u8>>,
    /// Gain map width in pixels.
    pub width: u32,
    /// Gain map height in pixels.
    pub height: u32,
}

// ============================================================================
// Decoder — replaces the old Decoder struct
// ============================================================================

/// JPEG decode configuration.
///
/// This is the main entry point for decoding. Create a `Decoder`, configure
/// it with builder methods, then call [`decode()`](Decoder::decode) or
/// [`scanline_reader()`](Decoder::scanline_reader).
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decoder::Decoder;
///
/// // Default (u8 sRGB, fastest)
/// let result = Decoder::new().decode(&jpeg_data, enough::Unstoppable)?;
/// let pixels: &[u8] = result.pixels_u8().unwrap();
///
/// // f32 sRGB with unclamped IDCT
/// let result = Decoder::new()
///     .output_target(OutputTarget::SrgbF32)
///     .decode(&jpeg_data, enough::Unstoppable)?;
/// let pixels: &[f32] = result.pixels_f32().unwrap();
/// ```
#[derive(Clone)]
pub struct Decoder {
    /// Output pixel format (None = use source format)
    pub output_format: Option<crate::types::PixelFormat>,
    /// Output target controlling precision, transfer function, and IDCT variant.
    pub output_target: OutputTarget,
    /// How to handle UltraHDR gain maps.
    pub gain_map: GainMapHandling,
    /// Chroma upsampling method for subsampled images
    pub chroma_upsampling: ChromaUpsampling,
    /// Whether to apply block smoothing
    pub block_smoothing: bool,
    /// Whether to apply embedded ICC profile (requires cms feature)
    pub apply_icc: bool,
    /// Maximum pixels allowed (for DoS protection).
    /// Default is 100 megapixels. Set to 0 for unlimited.
    /// Use `max_pixels()` method to set.
    pub(crate) max_pixels: u64,
    /// Maximum total memory for allocations (for DoS protection).
    /// Default is 512 MB. Set to 0 for unlimited.
    /// Use `max_memory()` method to set.
    pub(crate) max_memory: u64,
    /// What metadata and secondary images to preserve during decode.
    pub preserve: PreserveConfig,
    /// How to handle recoverable errors (truncation, minor spec violations).
    /// Default is [`Strictness::Balanced`].
    pub strictness: Strictness,
}

impl core::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Decoder")
            .field("output_format", &self.output_format)
            .field("output_target", &self.output_target)
            .field("gain_map", &self.gain_map)
            .field("chroma_upsampling", &self.chroma_upsampling)
            .field("block_smoothing", &self.block_smoothing)
            .field("apply_icc", &self.apply_icc)
            .field("max_pixels", &self.max_pixels)
            .field("max_memory", &self.max_memory)
            .field("preserve", &self.preserve)
            .field("strictness", &self.strictness)
            .finish()
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            output_format: None,
            output_target: OutputTarget::default(),
            gain_map: GainMapHandling::default(),
            chroma_upsampling: ChromaUpsampling::default(),
            block_smoothing: false,
            // Apply ICC by default when CMS is available
            apply_icc: cfg!(any(feature = "cms-lcms2", feature = "cms-moxcms")),
            max_pixels: DEFAULT_MAX_PIXELS,
            max_memory: DEFAULT_MAX_MEMORY,
            preserve: PreserveConfig::default(),
            strictness: Strictness::default(),
        }
    }
}

// ============================================================================
// DecodeResult — unified output type
// ============================================================================

/// Unified decode result, replacing `DecodedImage` and `DecodedImageF32`.
///
/// Contains decoded pixel data in either u8 or f32 format depending on the
/// [`OutputTarget`] used. Access pixels via [`pixels_u8()`](Self::pixels_u8)
/// or [`pixels_f32()`](Self::pixels_f32).
#[derive(Clone)]
#[non_exhaustive]
pub struct DecodeResult {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel format of the decoded data.
    pub format: crate::types::PixelFormat,
    output_target: OutputTarget,
    pixels_u8: Option<Vec<u8>>,
    pixels_f32: Option<Vec<f32>>,
    /// Gain map from UltraHDR images. `None` for regular JPEGs or Discard mode.
    pub gain_map: Option<GainMapResult>,
    pub(crate) extras: Option<DecodedExtras>,
    pub(crate) warnings: Vec<DecodeWarning>,
}

impl core::fmt::Debug for DecodeResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodeResult")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("output_target", &self.output_target)
            .field("pixels_u8_len", &self.pixels_u8.as_ref().map(|v| v.len()))
            .field("pixels_f32_len", &self.pixels_f32.as_ref().map(|v| v.len()))
            .field("has_gain_map", &self.gain_map.is_some())
            .field("has_extras", &self.extras.is_some())
            .finish()
    }
}

impl DecodeResult {
    /// Create a new u8 result.
    pub(crate) fn new_u8(
        width: u32,
        height: u32,
        format: crate::types::PixelFormat,
        output_target: OutputTarget,
        pixels: Vec<u8>,
        extras: Option<DecodedExtras>,
        warnings: Vec<DecodeWarning>,
    ) -> Self {
        Self {
            width,
            height,
            format,
            output_target,
            pixels_u8: Some(pixels),
            pixels_f32: None,
            gain_map: None,
            extras,
            warnings,
        }
    }

    /// Create a new f32 result.
    pub(crate) fn new_f32(
        width: u32,
        height: u32,
        format: crate::types::PixelFormat,
        output_target: OutputTarget,
        pixels: Vec<f32>,
        extras: Option<DecodedExtras>,
        warnings: Vec<DecodeWarning>,
    ) -> Self {
        Self {
            width,
            height,
            format,
            output_target,
            pixels_u8: None,
            pixels_f32: Some(pixels),
            gain_map: None,
            extras,
            warnings,
        }
    }

    /// Set the gain map result.
    pub(crate) fn set_gain_map(&mut self, gain_map: Option<GainMapResult>) {
        self.gain_map = gain_map;
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Image dimensions as (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Pixel format of the decoded data.
    #[must_use]
    pub fn format(&self) -> crate::types::PixelFormat {
        self.format
    }

    /// The output target that was used for decoding.
    #[must_use]
    pub fn output_target(&self) -> OutputTarget {
        self.output_target
    }

    /// Returns u8 pixel data, or `None` if output target is f32.
    #[must_use]
    pub fn pixels_u8(&self) -> Option<&[u8]> {
        self.pixels_u8.as_deref()
    }

    /// Returns f32 pixel data, or `None` if output target is u8.
    #[must_use]
    pub fn pixels_f32(&self) -> Option<&[f32]> {
        self.pixels_f32.as_deref()
    }

    /// Takes ownership of u8 pixel data.
    #[must_use]
    pub fn into_pixels_u8(self) -> Option<Vec<u8>> {
        self.pixels_u8
    }

    /// Takes ownership of f32 pixel data.
    #[must_use]
    pub fn into_pixels_f32(self) -> Option<Vec<f32>> {
        self.pixels_f32
    }

    /// Number of bytes per pixel for this image's format (u8 path).
    #[must_use]
    pub fn bytes_per_pixel(&self) -> usize {
        self.format.bytes_per_pixel()
    }

    /// Stride (elements per row) of the image.
    ///
    /// For u8: bytes per row. For f32: floats per row.
    #[must_use]
    pub fn stride(&self) -> usize {
        if self.output_target.is_f32() {
            self.width as usize * self.format.num_channels()
        } else {
            self.width as usize * self.bytes_per_pixel()
        }
    }

    /// Access preserved extras (metadata and secondary images).
    #[must_use]
    pub fn extras(&self) -> Option<&DecodedExtras> {
        self.extras.as_ref()
    }

    /// Take ownership of preserved extras.
    #[must_use]
    pub fn take_extras(&mut self) -> Option<DecodedExtras> {
        self.extras.take()
    }

    /// Warnings collected during decode.
    #[must_use]
    pub fn warnings(&self) -> &[DecodeWarning] {
        &self.warnings
    }

    /// Returns true if any warnings were collected.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Converts f32 pixel data to 16-bit integer format.
    ///
    /// Values are scaled from 0.0-1.0 to 0-65535 and clamped.
    /// Returns `None` if the result doesn't contain f32 data.
    #[must_use]
    pub fn to_u16(&self) -> Option<Vec<u16>> {
        let data = self.pixels_f32.as_ref()?;
        let len = data.len();
        let mut result = vec![0u16; len];
        for i in 0..len {
            result[i] = (data[i] * 65535.0).round().clamp(0.0, 65535.0) as u16;
        }
        Some(result)
    }

    /// Decompose into parts: (pixels_u8, pixels_f32, width, height, format, extras).
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Option<Vec<u8>>,
        Option<Vec<f32>>,
        u32,
        u32,
        crate::types::PixelFormat,
        Option<DecodedExtras>,
    ) {
        (
            self.pixels_u8,
            self.pixels_f32,
            self.width,
            self.height,
            self.format,
            self.extras,
        )
    }
}

// ============================================================================
// DecodeInfo — returned by decode_into_*
// ============================================================================

/// Metadata returned by [`Decoder::decode_into_u8`] and
/// [`Decoder::decode_into_f32`].
///
/// Contains everything except pixel data (which was written to the caller's buffer).
#[derive(Debug, Clone)]
pub struct DecodeInfo {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel format of the decoded data.
    pub format: crate::types::PixelFormat,
    /// Number of bytes (u8) or floats (f32) written to the output buffer.
    pub bytes_written: usize,
    /// Gain map from UltraHDR images.
    pub gain_map: Option<GainMapResult>,
    pub(crate) extras: Option<DecodedExtras>,
    pub(crate) warnings: Vec<DecodeWarning>,
}

impl DecodeInfo {
    /// Access preserved extras.
    #[must_use]
    pub fn extras(&self) -> Option<&DecodedExtras> {
        self.extras.as_ref()
    }

    /// Take ownership of preserved extras.
    #[must_use]
    pub fn take_extras(&mut self) -> Option<DecodedExtras> {
        self.extras.take()
    }

    /// Warnings collected during decode.
    #[must_use]
    pub fn warnings(&self) -> &[DecodeWarning] {
        &self.warnings
    }
}

// ============================================================================
// JpegInfo
// ============================================================================

/// Information about a decoded JPEG.
#[derive(Debug, Clone)]
pub struct JpegInfo {
    /// Image dimensions
    pub dimensions: Dimensions,
    /// Color space
    pub color_space: crate::types::ColorSpace,
    /// Sample precision (8 or 12 bits)
    pub precision: u8,
    /// Number of components
    pub num_components: u8,
    /// Encoding mode
    pub mode: crate::types::JpegMode,
    /// Whether an ICC profile is embedded
    pub has_icc_profile: bool,
    /// Whether the ICC profile is an XYB profile
    pub is_xyb: bool,
    /// ICC color profile (if embedded). Extracted during header parsing.
    pub icc_profile: Option<Vec<u8>>,
    /// EXIF metadata (raw bytes for external parsing). Extracted during header parsing.
    pub exif: Option<Vec<u8>>,
    /// XMP metadata string. Extracted during header parsing.
    pub xmp: Option<String>,
}
