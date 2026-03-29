//! Decoder configuration types.
//!
//! This module contains the configuration enums and structs used to control
//! JPEG decoding behavior.

use crate::color::icc::TargetColorSpace;
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
/// | `Triangle` | libjpeg-turbo, mozjpeg, djpeg (default) |
/// | `NearestNeighbor` | fastest, lowest quality |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ChromaUpsampling {
    /// Pixel replication (box filter). Fastest, lowest quality.
    ///
    /// Each chroma sample is duplicated to fill the corresponding output pixels.
    /// No interpolation is performed.
    NearestNeighbor,

    /// Fused 2D triangle filter with alternating rounding bias (default).
    ///
    /// Uses fused vertical+horizontal interpolation with alternating `+7`/`+8`
    /// rounding bias, avoiding systematic bias and intermediate rounding errors.
    /// Matches libjpeg-turbo/mozjpeg upsampling within max_diff ≤ 3.
    ///
    /// For pixel-exact matching (max_diff ≤ 2), also set
    /// `.idct_method(IdctMethod::Libjpeg)` — but note this adds ~37% decode
    /// overhead.
    #[default]
    Triangle,
}

/// Integer IDCT algorithm selection.
///
/// Controls which fixed-point IDCT implementation is used during decoding.
/// Different algorithms produce slightly different rounding, which matters
/// when comparing output against reference decoders.
///
/// | Method | Precision | Matches |
/// |--------|-----------|---------|
/// | `Jpegli` | 12-bit fixed-point | jpegli (Google JPEG XL project) |
/// | `Libjpeg` | 13-bit Loeffler | libjpeg-turbo, mozjpeg, djpeg |
///
/// The default is `Jpegli`. For pixel-exact mozjpeg matching, set
/// `.idct_method(IdctMethod::Libjpeg)` explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IdctMethod {
    /// 12-bit fixed-point IDCT (jpegli-derived).
    ///
    /// Uses AVX2 or portable wide SIMD with 12-bit precision.
    /// This is the default and matches jpegli's IDCT behavior.
    /// Max diff vs libjpeg-turbo: 2-3 levels.
    #[default]
    Jpegli,

    /// 13-bit Loeffler IDCT (libjpeg-turbo compatible).
    ///
    /// Uses the Loeffler, Ligtenberg, Moschytz algorithm with 13-bit
    /// fixed-point constants, matching libjpeg-turbo's `jpeg_idct_islow`.
    /// Use this when you need pixel-exact match with mozjpeg/djpeg output.
    Libjpeg,
}

/// Post-decode deblocking mode.
///
/// Controls whether and how deblocking filters are applied after JPEG decoding
/// to reduce 8x8 block boundary artifacts. Deblocking is most effective at low
/// quality levels (Q5-Q50) where blocking artifacts are most visible.
///
/// # Strategies
///
/// | Mode | Description | Best for |
/// |------|-------------|----------|
/// | `Off` | No deblocking (default) | Fastest, pixel-exact decode |
/// | `Auto` | Content-aware strategy selection | General use |
/// | `Boundary4Tap` | H.264-style pixel-domain filter | All quality levels |
/// | `Knusperli` | DCT-domain boundary correction | Low quality (Q5-Q30) |
///
/// `Auto` uses [`detect::content::recommend_deblock()`](crate::detect::content::recommend_deblock)
/// to pick the optimal strategy based on content type (photo vs screenshot),
/// encoder family, and quality level. Screenshots are skipped at Q10+ because
/// deblocking harms synthetic content.
///
/// # Performance
///
/// Boundary 4-tap adds ~5-15% decode time. Knusperli adds ~20-40% due to
/// extra IDCT work. Both modes force the coefficient decode path (no streaming),
/// since they need access to quantization tables and/or raw DCT coefficients.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DeblockMode {
    /// No deblocking (default). Fastest, pixel-exact decode output.
    #[default]
    Off,
    /// Auto-detect: pick the best deblocking strategy based on quality level.
    ///
    /// Uses Knusperli (DCT-domain correction) when DC quant ≥ 27 (roughly
    /// Q5–Q50), Boundary4Tap otherwise.
    ///
    /// When used with [`scanline_reader()`](super::DecodeConfig::scanline_reader)
    /// and the image needs Knusperli, the scanline reader transparently falls
    /// back to coefficient-based decoding (same as [`decode()`](super::DecodeConfig::decode))
    /// to produce correct deblocked output. This means `Auto` produces consistent
    /// output regardless of which decode path you use.
    Auto,
    /// Like [`Auto`](Self::Auto), but only picks strategies that work in streaming
    /// mode. Never falls back to coefficient-based decoding.
    ///
    /// Currently equivalent to [`Boundary4Tap`](Self::Boundary4Tap), but future
    /// streaming-compatible filters will be eligible for selection.
    AutoStreamable,
    /// Always apply H.264-style 4-tap boundary filter.
    ///
    /// Operates in the pixel domain at 8x8 block boundaries. Effective across
    /// all quality levels with moderate cost (~5-15% decode time).
    Boundary4Tap,
    /// Always apply Knusperli DCT-domain correction.
    ///
    /// Analytically computes boundary discontinuities and distributes corrections
    /// across low-frequency DCT coefficients. Best at low quality (Q5-Q30);
    /// may slightly hurt at high quality levels.
    ///
    /// Requires coefficient access. In [`scanline_reader()`](super::DecodeConfig::scanline_reader),
    /// transparently falls back to coefficient-based decoding (buffers full image).
    Knusperli,
}

/// Controls how restart segments are mapped to rayon tasks during parallel decode.
///
/// When DRI is MCU-row-aligned, the decoder can parallelize across restart
/// segments. This enum controls the grouping strategy — how many segments
/// each rayon task processes sequentially.
///
/// # Strategies
///
/// | Strategy | Tasks | Cache behavior |
/// |----------|-------|----------------|
/// | `PerSegment` | One per RST segment | High parallelism, scattered access |
/// | `Grouped` | `threads × groups_per_thread` | Contiguous strips per thread |
/// | `FixedStride(n)` | `ceil(segments / n)` | Explicit control for benchmarking |
/// | `Auto` | Adaptive | PerSegment for small, Grouped for large |
///
/// Only affects baseline images with MCU-row-aligned DRI and `--features parallel`.
/// Progressive images and sequential decode are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ParallelStrategy {
    /// One rayon task per restart segment. Current/legacy behavior.
    PerSegment,

    /// Group segments so each thread gets contiguous vertical strips.
    /// `groups_per_thread` controls oversubscription (2 = 2× tasks per thread).
    Grouped {
        /// Number of task groups per thread. Higher values give better load
        /// balancing at the cost of more task overhead. Typical: 1–4.
        groups_per_thread: usize,
    },

    /// Explicit segments per group (for benchmarking).
    FixedStride(usize),

    /// Auto-select: PerSegment for ≤16 segments, Grouped { groups_per_thread: 2 } otherwise.
    #[default]
    Auto,
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
/// | Situation | ITU-T T.81 spec | mozjpeg | Strict | Balanced | Lenient | Permissive |
/// |---|---|---|---|---|---|---|
/// | Truncated scan data | Invalid | JWRN_HIT_MARKER (fill 0) | Error | Fill zeros | Fill zeros | Fill zeros |
/// | Missing padding blocks | Invalid (MCUs required) | Implicit zero fill | Error | Speculative+zero | Speculative+zero | Speculative+zero |
/// | DNL conflicts with SOF | Invalid (B.2.5) | Ignored entirely | Error | Ignored | Ignored | Ignored |
/// | Bad Huffman at end-of-scan | Invalid | JWRN_HUFF_BAD_CODE (use 0) | Error | EndOfScan | EndOfScan | EndOfScan |
/// | Missing DHT before scan | Invalid (B.2.4.2) | std_huff_tables() fallback | Error | Std tables | Std tables | Std tables |
/// | Progressive scan truncated | Invalid | JWRN_HIT_MARKER (fill 0) | Error | Fill zeros | Fill zeros | Fill zeros |
/// | AC index overflow | Invalid | ERREXIT (fatal) | Error | Error | Treat as EOB | Treat as EOB |
/// | Invalid Huffman mid-scan | Invalid | ERREXIT (fatal) | Error | Error | Treat as EOB | Treat as EOB |
/// | Zero quant value in DQT | Invalid | ERREXIT (fatal) | Error | Error | Error | Clamp to 1 |
/// | Malformed segment length | Invalid | ERREXIT (fatal) | Error | Error | Error | Skip segment |
/// | RST marker mismatch | Invalid | jpeg_resync_to_restart | Error | Error | Error | Accept any RST |
/// | Bad DQT/DHT structure | Invalid | ERREXIT (fatal) | Error | Error | Error | Error |
/// | Bad component ID in SOS | Invalid (B.2.3) | ERREXIT (fatal) | Error | Error | Error | Error |
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
    /// - Forensic analysis of damaged files
    Lenient,

    /// Maximum compatibility: accept anything libjpeg-turbo accepts.
    ///
    /// Includes all Lenient recovery, plus:
    /// - Zero quantization values (clamped to 1)
    /// - Malformed segment lengths (skipped)
    /// - Restart marker sequence mismatches (resynced)
    ///
    /// Use for:
    /// - Processing images from unknown/untrusted sources
    /// - Web crawlers and image scrapers
    /// - Maximum libjpeg-turbo compatibility
    Permissive,
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

    /// Zero quantization value clamped to 1.
    ///
    /// Only recovered in Permissive mode. Zero values are invalid per spec
    /// (division by zero during dequantization).
    ZeroQuantValue {
        /// Which quantization table contained the zero value.
        table_idx: u8,
    },

    /// Malformed segment with invalid length was skipped.
    ///
    /// Only recovered in Permissive mode. Segments must have length >= 2.
    MalformedSegmentSkipped,

    /// Restart marker sequence mismatch was resynced.
    ///
    /// Only recovered in Permissive mode. Expected one RST marker number
    /// but found a different one; accepted the found marker and continued.
    RestartMarkerResync {
        /// Expected restart marker number (0-7).
        expected: u8,
        /// Actual restart marker number found (0-7).
        found: u8,
    },
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
            Self::ZeroQuantValue { table_idx } => {
                write!(
                    f,
                    "zero quantization value in table {}; clamped to 1",
                    table_idx
                )
            }
            Self::MalformedSegmentSkipped => {
                write!(f, "malformed segment with invalid length; skipped")
            }
            Self::RestartMarkerResync { expected, found } => {
                write!(
                    f,
                    "restart marker mismatch: expected RST{}, found RST{}; resynced",
                    expected, found
                )
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
// CropRegion — crop-on-decode
// ============================================================================

/// A region to crop during decoding.
///
/// When set on [`DecodeConfig`], the decoder will skip IDCT/upsampling for
/// MCU rows outside the crop region, significantly reducing decode cost for
/// small crops of large images.
///
/// Entropy decoding still runs for the full image (the DC predictor chain
/// requires it), but IDCT — the heavier operation — is skipped for rows
/// outside the crop.
///
/// Coordinates are in output space (after any `auto_orient` or `transform`).
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decode::{Decoder, CropRegion};
///
/// // Crop a 100x100 region starting at (50, 50)
/// let result = Decoder::new()
///     .crop(CropRegion::pixels(50, 50, 100, 100))
///     .decode(&jpeg_data, enough::Unstoppable)?;
/// assert_eq!(result.width(), 100);
/// assert_eq!(result.height(), 100);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CropRegion {
    /// Crop specified in pixel coordinates.
    Pixels {
        /// X offset of the crop region (left edge).
        x: u32,
        /// Y offset of the crop region (top edge).
        y: u32,
        /// Width of the crop region.
        width: u32,
        /// Height of the crop region.
        height: u32,
    },
    /// Crop specified as fractions of image dimensions (0.0–1.0).
    Percent {
        /// X offset as a fraction of image width.
        x: f32,
        /// Y offset as a fraction of image height.
        y: f32,
        /// Width as a fraction of image width.
        width: f32,
        /// Height as a fraction of image height.
        height: f32,
    },
}

impl CropRegion {
    /// Create a pixel-coordinate crop region.
    #[must_use]
    pub fn pixels(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self::Pixels {
            x,
            y,
            width,
            height,
        }
    }

    /// Create a percentage-based crop region (values in 0.0–1.0).
    #[must_use]
    pub fn percent(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self::Percent {
            x,
            y,
            width,
            height,
        }
    }

    /// Resolve to absolute pixel coordinates given image dimensions.
    pub(crate) fn resolve(
        self,
        img_w: u32,
        img_h: u32,
        mcu_height: usize,
    ) -> crate::error::Result<ResolvedCrop> {
        let (x, y, w, h) = match self {
            CropRegion::Pixels {
                x,
                y,
                width,
                height,
            } => (x, y, width, height),
            CropRegion::Percent {
                x,
                y,
                width,
                height,
            } => {
                if !(0.0..=1.0).contains(&x)
                    || !(0.0..=1.0).contains(&y)
                    || !(0.0..=1.0).contains(&width)
                    || !(0.0..=1.0).contains(&height)
                {
                    return Err(crate::error::Error::invalid_jpeg_data(
                        "crop percentages must be in 0.0..=1.0",
                    ));
                }
                let px = (x * img_w as f32).round() as u32;
                let py = (y * img_h as f32).round() as u32;
                let pw = (width * img_w as f32).round() as u32;
                let ph = (height * img_h as f32).round() as u32;
                (px, py, pw, ph)
            }
        };

        if w == 0 || h == 0 {
            return Err(crate::error::Error::invalid_jpeg_data(
                "crop region must have non-zero width and height",
            ));
        }
        if x.saturating_add(w) > img_w || y.saturating_add(h) > img_h {
            return Err(crate::error::Error::invalid_jpeg_data(
                "crop region extends beyond image bounds",
            ));
        }

        let crop_end_y = (y + h) as usize;
        let mcu_row_start = y as usize / mcu_height;
        let mcu_row_end = (crop_end_y + mcu_height - 1) / mcu_height;

        Ok(ResolvedCrop {
            x,
            y,
            width: w,
            height: h,
            mcu_row_start,
            mcu_row_end,
        })
    }
}

/// Resolved crop region with precomputed MCU row range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedCrop {
    /// X offset in pixels.
    pub x: u32,
    /// Y offset in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// First MCU row overlapping the crop (inclusive).
    pub mcu_row_start: usize,
    /// First MCU row past the crop (exclusive).
    pub mcu_row_end: usize,
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
    /// Convert embedded ICC color profile to a target color space.
    ///
    /// When `Some(target)`, the decoder applies the embedded ICC profile
    /// to convert pixel data to the specified color space. When `None`
    /// (default), no color conversion is performed.
    pub correct_color: Option<TargetColorSpace>,
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
    /// Crop region to decode (skip IDCT for MCU rows outside the crop).
    pub(crate) crop_region: Option<CropRegion>,
    /// Thread control for parallel decode paths.
    /// 0 = auto (default, uses rayon global pool), 1 = force sequential.
    pub(crate) num_threads: usize,
    /// How restart segments are mapped to rayon tasks during parallel decode.
    pub(crate) parallel_strategy: ParallelStrategy,
    /// Integer IDCT algorithm override.
    ///
    /// When `None` (default), `IdctMethod::Jpegli` is used for all upsampling
    /// modes. Set to `IdctMethod::Libjpeg` for pixel-exact mozjpeg matching.
    pub(crate) idct_method: Option<IdctMethod>,
    /// Post-decode deblocking mode.
    ///
    /// Default: [`DeblockMode::Off`]. When set to a non-Off mode, forces
    /// coefficient decode path (no streaming) since deblocking needs access
    /// to quantization tables and/or raw DCT coefficients.
    pub(crate) deblock_mode: DeblockMode,
}

impl core::fmt::Debug for DecodeConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodeConfig")
            .field("output_format", &self.output_format)
            .field("output_target", &self.output_target)
            .field("gain_map", &self.gain_map)
            .field("chroma_upsampling", &self.chroma_upsampling)
            .field("correct_color", &self.correct_color)
            .field("max_pixels", &self.max_pixels)
            .field("max_memory", &self.max_memory)
            .field("preserve", &self.preserve)
            .field("strictness", &self.strictness)
            .field("auto_orient", &self.auto_orient)
            .field("decode_transform", &self.decode_transform)
            .field("crop_region", &self.crop_region)
            .field("num_threads", &self.num_threads)
            .field("parallel_strategy", &self.parallel_strategy)
            .field("idct_method", &self.idct_method)
            .field("deblock_mode", &self.deblock_mode)
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
            correct_color: None,
            max_pixels: DEFAULT_MAX_PIXELS,
            max_memory: DEFAULT_MAX_MEMORY,
            preserve: PreserveConfig::default(),
            strictness: Strictness::default(),
            auto_orient: true,
            decode_transform: None,
            force_f32_idct: false,
            crop_region: None,
            num_threads: 0,
            parallel_strategy: ParallelStrategy::default(),
            idct_method: None,
            deblock_mode: DeblockMode::default(),
        }
    }
}

// ============================================================================
// DecodedPixels — type-safe pixel access
// ============================================================================

/// Borrowed pixel data from a [`DecodeResult`], with format encoded in the variant.
///
/// Returned by [`DecodeResult::pixels()`]. Eliminates the need to call
/// `pixels_u8()` / `pixels_f32()` and handle `Option` when you don't know
/// the output target at compile time.
///
/// ```rust,ignore
/// let result = decoder.decode(&jpeg_data, Unstoppable)?;
/// match result.pixels() {
///     DecodedPixels::U8(data) => process_u8(data),
///     DecodedPixels::F32(data) => process_f32(data),
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub enum DecodedPixels<'a> {
    /// 8-bit pixel data (from [`OutputTarget::Srgb8`]).
    U8(&'a [u8]),
    /// 32-bit float pixel data (from [`OutputTarget::SrgbF32`], [`OutputTarget::LinearF32`], etc.).
    F32(&'a [f32]),
}

/// Owned pixel data from a [`DecodeResult`], with format encoded in the variant.
///
/// Returned by [`DecodeResult::into_pixels()`].
#[derive(Debug, Clone)]
pub enum OwnedDecodedPixels {
    /// 8-bit pixel data.
    U8(Vec<u8>),
    /// 32-bit float pixel data.
    F32(Vec<f32>),
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

    /// Returns the decoded pixel data as a [`DecodedPixels`] enum.
    ///
    /// This is the preferred way to access pixels when you don't know the
    /// output target at compile time. The variant tells you whether the data
    /// is u8 or f32.
    ///
    /// # Panics
    ///
    /// Panics if the result contains no pixel data (should not happen for
    /// successful decodes).
    #[must_use]
    pub fn pixels(&self) -> DecodedPixels<'_> {
        if let Some(ref data) = self.pixels_u8 {
            DecodedPixels::U8(data)
        } else if let Some(ref data) = self.pixels_f32 {
            DecodedPixels::F32(data)
        } else {
            panic!("DecodeResult contains no pixel data")
        }
    }

    /// Takes ownership of the decoded pixel data as an [`OwnedDecodedPixels`] enum.
    ///
    /// # Panics
    ///
    /// Panics if the result contains no pixel data.
    #[must_use]
    pub fn into_pixels(self) -> OwnedDecodedPixels {
        if let Some(data) = self.pixels_u8 {
            OwnedDecodedPixels::U8(data)
        } else if let Some(data) = self.pixels_f32 {
            OwnedDecodedPixels::F32(data)
        } else {
            panic!("DecodeResult contains no pixel data")
        }
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
    /// Chroma subsampling mode
    pub subsampling: crate::types::Subsampling,
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
    /// JFIF density info (resolution/DPI). Extracted during header parsing.
    pub jfif: Option<crate::encode::extras::JfifInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_crop_basic() {
        let crop = CropRegion::pixels(10, 20, 100, 50);
        let resolved = crop.resolve(640, 480, 16).unwrap();
        assert_eq!(resolved.x, 10);
        assert_eq!(resolved.y, 20);
        assert_eq!(resolved.width, 100);
        assert_eq!(resolved.height, 50);
        assert_eq!(resolved.mcu_row_start, 1); // 20 / 16
        assert_eq!(resolved.mcu_row_end, 5); // ceil(70 / 16)
    }

    #[test]
    fn resolve_crop_percent() {
        let crop = CropRegion::percent(0.25, 0.25, 0.5, 0.5);
        let resolved = crop.resolve(640, 480, 16).unwrap();
        assert_eq!(resolved.x, 160);
        assert_eq!(resolved.y, 120);
        assert_eq!(resolved.width, 320);
        assert_eq!(resolved.height, 240);
    }

    #[test]
    fn resolve_crop_out_of_bounds() {
        let crop = CropRegion::pixels(600, 0, 100, 100);
        assert!(crop.resolve(640, 480, 16).is_err());
    }
}
