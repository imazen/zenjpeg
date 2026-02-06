//! Decoder configuration types.
//!
//! This module contains the configuration enums and structs used to control
//! JPEG decoding behavior.

use crate::foundation::alloc::{DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS};
use crate::types::Dimensions;

use super::extras::PreserveConfig;

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
/// collected as warnings and accessible via [`DecodedImage::warnings()`].
///
/// This allows the same enum to serve as both warning data and error context.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decoder::{Decoder, DecodeWarning};
///
/// let image = Decoder::new().decode(&data)?;
/// for warning in image.warnings() {
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

/// Decoder configuration.
#[derive(Clone)]
pub struct DecoderConfig {
    /// Output pixel format (None = use source format)
    pub output_format: Option<crate::types::PixelFormat>,
    /// Chroma upsampling method for subsampled images
    pub chroma_upsampling: ChromaUpsampling,
    /// Whether to apply block smoothing
    pub block_smoothing: bool,
    /// Whether to apply embedded ICC profile (requires cms feature)
    pub apply_icc: bool,
    /// Maximum pixels allowed (for DoS protection).
    /// Default is 100 megapixels. Set to 0 for unlimited.
    pub max_pixels: u64,
    /// Maximum total memory for allocations (for DoS protection).
    /// Default is 512 MB. Set to 0 for unlimited.
    pub max_memory: usize,
    /// What metadata and secondary images to preserve during decode.
    pub preserve: PreserveConfig,
    /// How to handle recoverable errors (truncation, minor spec violations).
    /// Default is [`Strictness::Balanced`].
    pub strictness: Strictness,
    /// Apply optimal Laplacian dequantization biases (Price & Rabbani 2000)
    /// for reduced reconstruction error.
    ///
    /// When enabled, the decoder uses f32 dequantization with per-coefficient
    /// biases computed from DCT coefficient statistics, instead of the default
    /// integer dequantization. This produces higher-quality output at the cost
    /// of slower decoding (falls through to the f32 IDCT path).
    ///
    /// Default: `false`.
    pub dequant_bias: bool,
}

impl core::fmt::Debug for DecoderConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecoderConfig")
            .field("output_format", &self.output_format)
            .field("chroma_upsampling", &self.chroma_upsampling)
            .field("block_smoothing", &self.block_smoothing)
            .field("apply_icc", &self.apply_icc)
            .field("max_pixels", &self.max_pixels)
            .field("max_memory", &self.max_memory)
            .field("preserve", &self.preserve)
            .field("strictness", &self.strictness)
            .field("dequant_bias", &self.dequant_bias)
            .finish()
    }
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            output_format: None,
            chroma_upsampling: ChromaUpsampling::default(),
            block_smoothing: false,
            // Apply ICC by default when CMS is available
            apply_icc: cfg!(any(feature = "cms-lcms2", feature = "cms-moxcms")),
            max_pixels: DEFAULT_MAX_PIXELS,
            max_memory: DEFAULT_MAX_MEMORY,
            preserve: PreserveConfig::default(),
            strictness: Strictness::default(),
            dequant_bias: false,
        }
    }
}

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
}
