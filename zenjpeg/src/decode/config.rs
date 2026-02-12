//! Decoder configuration types.
//!
//! This module contains the configuration enums and structs used to control
//! JPEG decoding behavior.

use crate::foundation::alloc::{DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS};
use crate::lossless::LosslessTransform;
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
// DctScale — shrink-on-load DCT scaling
// ============================================================================

/// DCT scaling factor for shrink-on-load decoding.
///
/// Controls how many output pixels are produced per 8x8 DCT block.
/// Smaller scales skip high-frequency coefficients and produce fewer pixels,
/// giving a significant speedup over full decode + post-process downscale.
///
/// # Scaling math
///
/// Output dimensions use ceiling division: `(original * numerator + 7) / 8`.
/// This ensures the output is always large enough to cover the source.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decoder::{Decoder, ShrinkHint, DctScale};
///
/// // Decode a 4000x3000 image at 1/4 scale → 1000x750
/// let result = Decoder::new()
///     .shrink(ShrinkHint::ExactScale(DctScale::Quarter))
///     .decode(&jpeg_data, enough::Unstoppable)?;
/// assert_eq!(result.width(), 1000);
/// assert_eq!(result.height(), 750);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum DctScale {
    /// 1/16 scale: 1 pixel per 2×2 blocks (16×16 source pixels).
    ///
    /// Internally decodes at 1/8 (DC-only), then applies a 2×2 box filter
    /// to halve each dimension. Ideal for very large images where even 1/8
    /// produces too many pixels (e.g., 100MP → 640×640 instead of 1280×1280).
    Sixteenth,
    /// 1x1 output per block (DC only). 1/8 scale.
    Eighth,
    /// 2x2 output per block. 1/4 scale.
    Quarter,
    /// 4x4 output per block. 1/2 scale.
    Half,
    /// 8x8 output per block (full resolution, no scaling). Default.
    #[default]
    Full,
}

impl DctScale {
    /// Scale numerator (1, 2, 4, or 8).
    ///
    /// For `Sixteenth`, returns 1 (same as `Eighth`) since the internal IDCT
    /// operates at 1/8 scale. Use [`scaled_dimension()`](Self::scaled_dimension)
    /// for the actual output size which accounts for the post-filter.
    #[inline]
    #[must_use]
    pub const fn numerator(self) -> u32 {
        match self {
            Self::Sixteenth | Self::Eighth => 1,
            Self::Quarter => 2,
            Self::Half => 4,
            Self::Full => 8,
        }
    }

    /// Output pixels per block edge for the internal IDCT (1, 2, 4, or 8).
    ///
    /// For `Sixteenth`, returns 1 (same as `Eighth`) since the IDCT operates
    /// at 1/8 scale. The 2×2 post-filter is applied separately.
    #[inline]
    #[must_use]
    pub const fn block_output_size(self) -> usize {
        self.numerator() as usize
    }

    /// Compute scaled dimension from original using ceiling division.
    ///
    /// Formula: `(original * numerator + 7) / 8`, except for `Sixteenth`
    /// which uses `(original + 15) / 16` (one pixel per 2×2 blocks).
    #[inline]
    #[must_use]
    pub const fn scaled_dimension(self, original: u32) -> u32 {
        match self {
            Self::Sixteenth => (original as u64 + 15) as u32 / 16,
            _ => (original as u64 * self.numerator() as u64 + 7) as u32 / 8,
        }
    }

    /// Compute scaled dimensions from original dimensions.
    #[inline]
    #[must_use]
    pub const fn scaled_dimensions(self, dims: Dimensions) -> Dimensions {
        Dimensions::new(
            self.scaled_dimension(dims.width),
            self.scaled_dimension(dims.height),
        )
    }

    /// The internal DCT scale used for IDCT processing.
    ///
    /// For `Sixteenth`, returns `Eighth` (the IDCT runs at 1/8, then a 2×2
    /// box filter halves the result). All other scales return themselves.
    #[inline]
    #[must_use]
    pub const fn internal_scale(self) -> DctScale {
        match self {
            Self::Sixteenth => Self::Eighth,
            other => other,
        }
    }

    /// Returns `true` if this scale requires a post-filter after IDCT.
    #[inline]
    #[must_use]
    pub const fn needs_post_filter(self) -> bool {
        matches!(self, Self::Sixteenth)
    }

    /// All supported scales in order from smallest to largest output.
    pub const ALL: [DctScale; 5] = [
        DctScale::Sixteenth,
        DctScale::Eighth,
        DctScale::Quarter,
        DctScale::Half,
        DctScale::Full,
    ];
}

impl core::fmt::Display for DctScale {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Sixteenth => write!(f, "1/16"),
            Self::Eighth => write!(f, "1/8"),
            Self::Quarter => write!(f, "1/4"),
            Self::Half => write!(f, "1/2"),
            Self::Full => write!(f, "1/1"),
        }
    }
}

/// Hint for shrink-on-load decoding.
///
/// The decoder uses this to select the smallest DCT scale that meets
/// the requested output size. Shrink is a *hint*, not a guarantee —
/// the actual output dimensions depend on available DCT scales.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decoder::{Decoder, ShrinkHint};
///
/// // Request at least 800x600 — decoder picks smallest scale that fits
/// let result = Decoder::new()
///     .shrink(ShrinkHint::FitWithin { width: 800, height: 600 })
///     .decode(&jpeg_data, enough::Unstoppable)?;
/// assert!(result.width() >= 800);
/// assert!(result.height() >= 600);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShrinkHint {
    /// Decoder picks smallest scale where output >= both target dimensions.
    FitWithin {
        /// Minimum output width.
        width: u32,
        /// Minimum output height.
        height: u32,
    },
    /// Exact DCT scale factor.
    ExactScale(DctScale),
}

impl ShrinkHint {
    /// Select the best DCT scale for the given source dimensions.
    ///
    /// For `FitWithin`, picks the smallest scale where the scaled output
    /// meets or exceeds both target dimensions. For `ExactScale`, returns
    /// the requested scale directly.
    #[must_use]
    pub fn resolve(self, source: Dimensions) -> DctScale {
        match self {
            Self::ExactScale(scale) => scale,
            Self::FitWithin { width, height } => {
                for &scale in &DctScale::ALL {
                    let sw = scale.scaled_dimension(source.width);
                    let sh = scale.scaled_dimension(source.height);
                    if sw >= width && sh >= height {
                        return scale;
                    }
                }
                DctScale::Full
            }
        }
    }
}

/// Quality tier for shrink-on-load decoding.
///
/// Controls the tradeoff between decode speed and output quality at
/// reduced scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShrinkQuality {
    /// Reduced NxN IDCT per block. Fast, may show block boundary artifacts
    /// at reduced scales. Default for `Srgb8` output.
    #[default]
    Fast,
    /// Full 8x8 IDCT + cross-block spatial downscale with extended taps.
    /// Eliminates block boundary artifacts. Default for `Precise` output targets.
    Best,
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
pub struct DecodeConfig {
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
    /// Whether to automatically correct EXIF orientation during decode.
    ///
    /// When enabled, the decoder reads the EXIF orientation tag and applies
    /// the corresponding transform in DCT-coefficient space before IDCT.
    /// The output pixels will have correct visual orientation.
    ///
    /// Default: `false`.
    pub(crate) auto_orient: bool,
    /// Explicit lossless transform to apply during decode.
    ///
    /// Applied in DCT-coefficient space before IDCT, so there is no
    /// quality loss from the transform itself. When combined with
    /// `auto_orient`, the EXIF correction is applied first, then this
    /// transform.
    ///
    /// Default: `None`.
    pub(crate) decode_transform: Option<LosslessTransform>,
    /// Force f32 IDCT for symmetric rounding (used internally by dimension-swapping
    /// transforms, also available for testing).
    pub(crate) force_f32_idct: bool,
    /// Shrink-on-load hint. When set, the decoder produces reduced-resolution
    /// output by exploiting DCT structure (fewer IDCT operations, smaller buffers).
    pub(crate) shrink_hint: Option<ShrinkHint>,
    /// Quality tier for shrink-on-load. `None` means auto-select based on
    /// output target (Fast for Srgb8, Best for Precise).
    pub(crate) shrink_quality: Option<ShrinkQuality>,
}

impl core::fmt::Debug for DecodeConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodeConfig")
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
            .field("auto_orient", &self.auto_orient)
            .field("decode_transform", &self.decode_transform)
            .field("shrink_hint", &self.shrink_hint)
            .field("shrink_quality", &self.shrink_quality)
            .finish()
    }
}

impl Default for DecodeConfig {
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
            auto_orient: false,
            decode_transform: None,
            force_f32_idct: false,
            shrink_hint: None,
            shrink_quality: None,
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
    /// Available shrink-on-load scales and their output dimensions.
    ///
    /// Computed from SOF dimensions. Allows callers to inspect what
    /// DCT scale factors produce what output sizes before committing
    /// to decode.
    pub available_scales: [(DctScale, Dimensions); 5],
}

impl JpegInfo {
    /// Compute `available_scales` from source dimensions.
    pub(crate) fn compute_available_scales(dims: Dimensions) -> [(DctScale, Dimensions); 5] {
        [
            (
                DctScale::Sixteenth,
                DctScale::Sixteenth.scaled_dimensions(dims),
            ),
            (DctScale::Eighth, DctScale::Eighth.scaled_dimensions(dims)),
            (
                DctScale::Quarter,
                DctScale::Quarter.scaled_dimensions(dims),
            ),
            (DctScale::Half, DctScale::Half.scaled_dimensions(dims)),
            (DctScale::Full, DctScale::Full.scaled_dimensions(dims)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_scale_numerator() {
        assert_eq!(DctScale::Sixteenth.numerator(), 1);
        assert_eq!(DctScale::Eighth.numerator(), 1);
        assert_eq!(DctScale::Quarter.numerator(), 2);
        assert_eq!(DctScale::Half.numerator(), 4);
        assert_eq!(DctScale::Full.numerator(), 8);
    }

    #[test]
    fn dct_scale_block_output_size() {
        assert_eq!(DctScale::Sixteenth.block_output_size(), 1);
        assert_eq!(DctScale::Eighth.block_output_size(), 1);
        assert_eq!(DctScale::Quarter.block_output_size(), 2);
        assert_eq!(DctScale::Half.block_output_size(), 4);
        assert_eq!(DctScale::Full.block_output_size(), 8);
    }

    #[test]
    fn dct_scale_internal_scale() {
        assert_eq!(DctScale::Sixteenth.internal_scale(), DctScale::Eighth);
        assert_eq!(DctScale::Eighth.internal_scale(), DctScale::Eighth);
        assert_eq!(DctScale::Quarter.internal_scale(), DctScale::Quarter);
        assert_eq!(DctScale::Half.internal_scale(), DctScale::Half);
        assert_eq!(DctScale::Full.internal_scale(), DctScale::Full);
    }

    #[test]
    fn dct_scale_needs_post_filter() {
        assert!(DctScale::Sixteenth.needs_post_filter());
        assert!(!DctScale::Eighth.needs_post_filter());
        assert!(!DctScale::Quarter.needs_post_filter());
        assert!(!DctScale::Half.needs_post_filter());
        assert!(!DctScale::Full.needs_post_filter());
    }

    #[test]
    fn dct_scale_scaled_dimension_exact_multiples() {
        // 4000x3000, exact multiple of 8
        assert_eq!(DctScale::Full.scaled_dimension(4000), 4000);
        assert_eq!(DctScale::Half.scaled_dimension(4000), 2000);
        assert_eq!(DctScale::Quarter.scaled_dimension(4000), 1000);
        assert_eq!(DctScale::Eighth.scaled_dimension(4000), 500);
        assert_eq!(DctScale::Sixteenth.scaled_dimension(4000), 250);

        assert_eq!(DctScale::Full.scaled_dimension(3000), 3000);
        assert_eq!(DctScale::Half.scaled_dimension(3000), 1500);
        assert_eq!(DctScale::Quarter.scaled_dimension(3000), 750);
        assert_eq!(DctScale::Eighth.scaled_dimension(3000), 375);
        assert_eq!(DctScale::Sixteenth.scaled_dimension(3000), 188); // (3000+15)/16 = 3015/16 = 188
    }

    #[test]
    fn dct_scale_scaled_dimension_non_aligned() {
        // 1118x1105 (not 8-aligned)
        // (1118 * num + 7) / 8
        assert_eq!(DctScale::Full.scaled_dimension(1118), 1118);
        // (1118 * 4 + 7) / 8 = 4479 / 8 = 559 (integer division)
        assert_eq!(DctScale::Half.scaled_dimension(1118), 559);
        assert_eq!(DctScale::Quarter.scaled_dimension(1118), 280); // (1118*2+7)/8 = 2243/8 = 280
        assert_eq!(DctScale::Eighth.scaled_dimension(1118), 140); // (1118*1+7)/8 = 1125/8 = 140
        assert_eq!(DctScale::Sixteenth.scaled_dimension(1118), 70); // (1118+15)/16 = 1133/16 = 70

        assert_eq!(DctScale::Full.scaled_dimension(1105), 1105);
        assert_eq!(DctScale::Half.scaled_dimension(1105), 553); // (1105*4+7)/8 = 4427/8 = 553
        assert_eq!(DctScale::Quarter.scaled_dimension(1105), 277); // (1105*2+7)/8 = 2217/8 = 277
        assert_eq!(DctScale::Eighth.scaled_dimension(1105), 139); // (1105*1+7)/8 = 1112/8 = 139
        assert_eq!(DctScale::Sixteenth.scaled_dimension(1105), 70); // (1105+15)/16 = 1120/16 = 70
    }

    #[test]
    fn dct_scale_scaled_dimension_small() {
        // Minimum useful: 1x1
        assert_eq!(DctScale::Full.scaled_dimension(1), 1);
        assert_eq!(DctScale::Half.scaled_dimension(1), 1); // (1*4+7)/8 = 11/8 = 1
        assert_eq!(DctScale::Quarter.scaled_dimension(1), 1); // (1*2+7)/8 = 9/8 = 1
        assert_eq!(DctScale::Eighth.scaled_dimension(1), 1); // (1*1+7)/8 = 8/8 = 1
        assert_eq!(DctScale::Sixteenth.scaled_dimension(1), 1); // (1+15)/16 = 16/16 = 1

        // 8x8 (single block)
        assert_eq!(DctScale::Full.scaled_dimension(8), 8);
        assert_eq!(DctScale::Half.scaled_dimension(8), 4);
        assert_eq!(DctScale::Quarter.scaled_dimension(8), 2);
        assert_eq!(DctScale::Eighth.scaled_dimension(8), 1);
        assert_eq!(DctScale::Sixteenth.scaled_dimension(8), 1); // (8+15)/16 = 23/16 = 1

        // 16x16 (2x2 blocks, Sixteenth produces exactly 1 pixel)
        assert_eq!(DctScale::Sixteenth.scaled_dimension(16), 1); // (16+15)/16 = 31/16 = 1

        // 10240x10240 (100MP target use case)
        assert_eq!(DctScale::Eighth.scaled_dimension(10240), 1280);
        assert_eq!(DctScale::Sixteenth.scaled_dimension(10240), 640);
    }

    #[test]
    fn shrink_hint_fit_within() {
        let src = Dimensions::new(4000, 3000);

        // Want at most 250x188 → 1/16 gives (250, 188), fits
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 250,
                height: 188
            }
            .resolve(src),
            DctScale::Sixteenth,
        );

        // Want at least 251x189 → 1/16 gives (250, 188), too small → 1/8 gives (500, 375)
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 251,
                height: 189
            }
            .resolve(src),
            DctScale::Eighth,
        );

        // Want at least 500x375 → 1/8 works (500x375)
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 500,
                height: 375
            }
            .resolve(src),
            DctScale::Eighth,
        );

        // Want at least 501x376 → 1/8 gives 500x375, too small → 1/4 gives 1000x750
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 501,
                height: 376
            }
            .resolve(src),
            DctScale::Quarter,
        );

        // Want at least 1001x751 → 1/4 gives 1000x750, too small → 1/2 gives 2000x1500
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 1001,
                height: 751
            }
            .resolve(src),
            DctScale::Half,
        );

        // Want at least 2001x1501 → 1/2 gives 2000x1500, too small → full
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 2001,
                height: 1501
            }
            .resolve(src),
            DctScale::Full,
        );
    }

    #[test]
    fn shrink_hint_exact_scale() {
        let src = Dimensions::new(4000, 3000);
        assert_eq!(
            ShrinkHint::ExactScale(DctScale::Quarter).resolve(src),
            DctScale::Quarter,
        );
    }

    #[test]
    fn shrink_hint_height_constrained() {
        // 800x6000: portrait image
        let src = Dimensions::new(800, 6000);

        // Want 100x100 → 1/8 gives (100, 750) → both >= 100 ✓
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 100,
                height: 100
            }
            .resolve(src),
            DctScale::Eighth,
        );

        // Want 100x751 → 1/8 gives (100, 750) → height too small → 1/4 gives (200, 1500)
        assert_eq!(
            ShrinkHint::FitWithin {
                width: 100,
                height: 751
            }
            .resolve(src),
            DctScale::Quarter,
        );
    }

    #[test]
    fn available_scales_computation() {
        let dims = Dimensions::new(4000, 3000);
        let scales = JpegInfo::compute_available_scales(dims);

        assert_eq!(scales[0], (DctScale::Sixteenth, Dimensions::new(250, 188)));
        assert_eq!(scales[1], (DctScale::Eighth, Dimensions::new(500, 375)));
        assert_eq!(scales[2], (DctScale::Quarter, Dimensions::new(1000, 750)));
        assert_eq!(scales[3], (DctScale::Half, Dimensions::new(2000, 1500)));
        assert_eq!(scales[4], (DctScale::Full, Dimensions::new(4000, 3000)));
    }
}
