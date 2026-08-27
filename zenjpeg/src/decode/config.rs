//! Decoder configuration types.
//!
//! This module contains the configuration enums and structs used to control
//! JPEG decoding behavior.

use crate::color::icc::TargetColorSpace;
use crate::foundation::alloc::{DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS};
use crate::lossless::LosslessTransform;
use crate::types::Dimensions;
use zenpixels::Orientation;

use super::extras::{DecodedExtras, PreserveConfig};

/// How the decoder should handle image orientation.
///
/// Combines EXIF auto-orientation and explicit transforms into a single enum,
/// allowing the decoder to coalesce operations (the combined transform is
/// applied as a single lossless pixel permutation of the decoded image).
///
/// Matches [`zencodec::OrientationHint`] when the `zencodec` feature is enabled.
///
/// # Examples
///
/// ```
/// use zenjpeg::decode::{DecodeConfig, OrientationHint};
/// use zenpixels::Orientation;
///
/// // Auto-correct EXIF orientation (default)
/// let cfg = DecodeConfig::new();
///
/// // Preserve raw orientation (for lossless re-encoding)
/// let cfg = DecodeConfig::new().orientation(OrientationHint::Preserve);
///
/// // Auto-correct EXIF, then rotate 90° more
/// let cfg = DecodeConfig::new()
///     .orientation(OrientationHint::CorrectAndTransform(Orientation::Rotate90));
///
/// // Ignore EXIF, apply exact transform
/// let cfg = DecodeConfig::new()
///     .orientation(OrientationHint::ExactTransform(Orientation::FlipH));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OrientationHint {
    /// Don't touch orientation. Report intrinsic orientation in output metadata.
    Preserve,

    /// Resolve EXIF/container orientation to identity (default).
    ///
    /// The decoder applies the orientation as a lossless pixel permutation
    /// of the decoded image — identical to decoding upright and reorienting
    /// afterwards. The output will have correct visual orientation.
    #[default]
    Correct,

    /// Resolve EXIF orientation, then apply an additional transform.
    ///
    /// The decoder coalesces the combined operation when possible.
    /// For example, if EXIF says Rotate90 and the hint says Rotate180,
    /// the decoder applies Rotate270 in a single step.
    CorrectAndTransform(Orientation),

    /// Ignore EXIF orientation. Apply exactly this transform.
    ///
    /// The EXIF orientation is not consulted. The given transform is
    /// applied literally.
    ExactTransform(Orientation),
}

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
/// | `Triangle` | libjpeg-turbo/mozjpeg/djpeg within ±1 (default; see below) |
/// | `NearestNeighbor` | fastest, lowest quality |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ChromaUpsampling {
    /// Pixel replication (box filter). Fastest, lowest quality.
    ///
    /// Each chroma sample is duplicated to fill the corresponding output pixels.
    /// No interpolation is performed. Typically 5-10% faster than [`Triangle`](Self::Triangle)
    /// with minimal perceptual quality difference on photographic content.
    NearestNeighbor,

    /// Fused 2D triangle filter with alternating rounding bias (default).
    ///
    /// The same fused 9:3:3:1 filter as libjpeg-turbo's fancy upsampling.
    /// 4:2:2 (h2v1) and 4:4:0 (h1v2) are bit-identical to libjpeg-turbo;
    /// 4:2:0 (h2v2) row-alternates the `+8`/`+7` rounding bias where turbo
    /// keeps it fixed, so odd output rows differ from turbo by ±1 on the
    /// rounding-boundary cases (~3% of pixels; even rows bit-identical).
    /// Both schemes are equally accurate vs the exact filter (max err 0.5,
    /// ~zero bias).
    ///
    /// With the default [`IdctMethod::Libjpeg`] IDCT, all three stages are
    /// turbo's — the islow IDCT, the h2v2 fixed upsampling biases, AND the
    /// 16-bit YCbCr→RGB tables — so decoded RGB is BYTE-FOR-BYTE identical to
    /// mozjpeg/djpeg (max_diff == 0), verified over sizes × qualities × 4:2:0 by
    /// `test_idct_method_libjpeg_fancy_matches_mozjpeg_exact` (`__ffi-tests`).
    /// That is the out-of-the-box behavior.
    ///
    /// Opting into [`IdctMethod::Jpegli`] keeps this upsampler's alternating
    /// bias and the f32 YCbCr→RGB path, so decoded RGB then matches
    /// libjpeg-turbo/mozjpeg only within max_diff <= 3 (the IDCT precision
    /// differs), in exchange for ~3% faster decode
    /// (`benches/decode_zenbench.rs`).
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
/// | `Jpegli` | 12-bit fixed-point | stb / zune-jpeg (NOT jpegli — see below) |
/// | `Libjpeg` | 13-bit Loeffler | libjpeg-turbo, mozjpeg, djpeg (default) |
///
/// The default is `Libjpeg` (since 2026-07-15): it is both the more accurate
/// integer kernel and byte-exact with libjpeg-turbo/mozjpeg/djpeg. Set
/// `.idct_method(IdctMethod::Jpegli)` explicitly to opt into the faster 12-bit
/// kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IdctMethod {
    /// 12-bit fixed-point IDCT (stb/zune-derived).
    ///
    /// Uses AVX2 or portable wide SIMD with 12-bit precision. Roughly 3% faster
    /// to decode than [`Libjpeg`](Self::Libjpeg) (the islow kernel's two
    /// overflow guards), at the cost of accuracy and reference-exactness.
    /// Max diff vs libjpeg-turbo: 2-3 levels.
    ///
    /// **The name is a misnomer** kept for API stability: this is *not* what
    /// the C++ jpegli decoder uses. jpegli decodes with a float IDCT
    /// (`lib/jpegli/idct.cc`) — zenjpeg's f32 path, which is what XYB and
    /// `dequant_bias` route to. Both integer kernels here are the same Loeffler
    /// islow butterfly, differing only in fixed-point precision (12 vs 13 bit),
    /// rounding-bias placement, and intermediate width.
    ///
    /// **Less accurate than [`Libjpeg`](Self::Libjpeg)**: measured over 10k
    /// random blocks per config against an f64 reference, this kernel carries a
    /// systematic **+0.002..+0.004 bias** (from the extra `+512` in its pass-2
    /// `SCALE_BITS`; 512/2^17 = +0.0039, which dominates at small coefficients),
    /// where islow is unbiased (mean ≈ ±2e-4). See
    /// `test_idct_accuracy_stats_vs_reference` in `idct_int.rs`.
    ///
    /// Note 1-component (grayscale) sources always use
    /// [`Libjpeg`](Self::Libjpeg) regardless of this setting (#154) — the gray
    /// plane must be identical across decode paths, and with no chroma there is
    /// nothing for the 12-bit tuning to trade against.
    Jpegli,

    /// 13-bit Loeffler IDCT (libjpeg-turbo compatible). **Default.**
    ///
    /// Uses the Loeffler, Ligtenberg, Moschytz algorithm with 13-bit
    /// fixed-point constants, matching libjpeg-turbo's `jpeg_idct_islow`.
    /// Also selects libjpeg-turbo's fixed h2v2 fancy-upsampling rounding
    /// biases (see [`ChromaUpsampling::Triangle`]) and its 16-bit YCbCr→RGB
    /// tables, so all three decode stages are bit-exact with mozjpeg/djpeg:
    /// decoded RGB is byte-for-byte identical (max_diff == 0), verified
    /// against libjpeg-turbo by the `__ffi-tests` parity tests
    /// (`test_idct_method_libjpeg_fancy_matches_mozjpeg_exact`).
    ///
    /// Paired with the default [`ChromaUpsampling::Triangle`] this makes the
    /// out-of-the-box decode byte-exact with mozjpeg/djpeg. It is also the more
    /// accurate integer kernel (unbiased; see [`Jpegli`](Self::Jpegli)), which
    /// is why it is the default despite costing ~3% of decode wall time.
    ///
    /// The islow kernel is SIMD (guarded, bit-exact with its scalar kernel:
    /// inputs and pass-1 outputs must fit `[-32768, 32767]`, else it falls back
    /// to scalar — honest 8-bit imagery peaks around |4096| and never trips it).
    ///
    /// Honored identically (byte-for-byte) on every u8 decode path — streaming
    /// `decode()`, `scanline_reader()`, and the parallel path — across all
    /// subsampling modes and baseline/progressive (see
    /// `tests/libjpeg_idct_all_paths_parity.rs`).
    #[default]
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
/// # Performance and decode paths
///
/// Boundary 4-tap adds ~5-15% decode time and works in streaming mode.
/// Knusperli adds ~20-40% and forces the buffered coefficient decode path
/// (no streaming) since it needs raw DCT coefficients. `Auto` and
/// `AutoStreamable` stay streaming when they select Boundary4Tap; `Auto`
/// falls back to buffered only when it selects Knusperli (DC quant >= 27,
/// roughly Q5-Q50).
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
/// | Arith spectral/mag overflow | Invalid | JWRN_ARITH_BAD_CODE (EOS) | Error | EndOfScan | EndOfScan | EndOfScan |
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
    /// - Zero quantization values (clamped to 1, matching libjpeg-turbo)
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
    /// - Malformed segment lengths (skipped)
    /// - Restart marker sequence mismatches (resynced)
    ///
    /// Use for:
    /// - Processing images from unknown/untrusted sources
    /// - Web crawlers and image scrapers
    /// - Maximum libjpeg-turbo compatibility
    Permissive,
}

impl Strictness {
    /// Strict mode: every recoverable data error is fatal.
    #[inline]
    pub(crate) fn is_strict(self) -> bool {
        self == Self::Strict
    }

    /// Recover from data errors instead of failing — truncated scan data
    /// (zero fill), DNL/SOF height conflicts, missing DHT (K.3 fallback),
    /// zero quant values (clamp to 1). All levels except Strict.
    #[inline]
    pub(crate) fn recovers_data_errors(self) -> bool {
        self != Self::Strict
    }

    /// Lenient and Permissive: enable the entropy decoder's lenient
    /// recovery and warn (rather than error) on bad restart counts.
    #[inline]
    pub(crate) fn lenient_entropy_recovery(self) -> bool {
        matches!(self, Self::Lenient | Self::Permissive)
    }

    /// Permissive only: skip malformed header segments, resync
    /// out-of-sequence restart markers, clamp bad Huffman table indices.
    #[inline]
    pub(crate) fn is_permissive(self) -> bool {
        self == Self::Permissive
    }
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
// New variants may be added in minor releases as decoder hardening expands.
#[non_exhaustive]
pub enum DecodeWarning {
    /// No DHT markers found; standard Huffman tables (ITU-T T.81 K.3) were used.
    ///
    /// Common in MJPEG/AVI1 frames which omit DHT to save space.
    /// mozjpeg handles this via `std_huff_tables()` in `jinit_huff_decoder()`.
    MissingHuffmanTables,

    /// Scan data was truncated mid-scan; remaining blocks filled with zeros.
    ///
    /// The image may have partial content. `blocks_decoded` and `blocks_expected`
    /// indicate how much data was recovered. Both are always real counts — a
    /// truncation with no partial scan to describe reports
    /// [`TruncatedBetweenScans`](Self::TruncatedBetweenScans) instead.
    TruncatedScan {
        /// Number of MCU blocks successfully decoded before truncation.
        blocks_decoded: u32,
        /// Total number of MCU blocks expected for this scan.
        blocks_expected: u32,
    },

    /// The stream ended between scans — after a complete scan, before the
    /// next one's entropy data: at a marker boundary (a missing EOI, a file
    /// cut between scans) or inside a table / metadata segment (DHT, DQT,
    /// DRI, APPn, COM) that precedes the next scan.
    ///
    /// Every scan that was started also finished, so unlike
    /// [`TruncatedScan`](Self::TruncatedScan) there is no partially-decoded scan
    /// to report block counts for. For a baseline JPEG the image is complete; for
    /// a progressive one it is decoded to `scans_decoded` scans of refinement,
    /// which is a valid (if lower-quality) image.
    TruncatedBetweenScans {
        /// Number of scans fully decoded before the stream ended.
        scans_decoded: u32,
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
    /// Recovered in Balanced, Lenient, and Permissive modes. The decoder
    /// accepted a wrong RST number or forward-scanned to find the next
    /// valid RST marker and continued decoding.
    RestartMarkerResync {
        /// Number of RST marker resyncs during this scan.
        count: u32,
    },

    /// Extraneous bytes between markers were skipped.
    ///
    /// Non-0xFF bytes found where a marker was expected. These indicate
    /// file corruption or non-standard padding. In Strict mode this is
    /// an error; in Balanced/Lenient mode the bytes are skipped with a warning.
    ExtraneousBytesSkipped {
        /// Number of non-0xFF bytes skipped before a valid marker was found.
        count: u32,
    },

    /// Arithmetic coding overflow (bad code) recovered.
    ///
    /// Matches libjpeg-turbo's `JWRN_ARITH_BAD_CODE` warning. The arithmetic
    /// decoder encountered a spectral overflow (AC coefficient index exceeded
    /// spectral selection range) or magnitude overflow (decoded magnitude
    /// exceeded valid range). Remaining coefficients in the affected block
    /// are left as-is and subsequent blocks return immediately (end-of-scan).
    ///
    /// Recovered in Balanced, Lenient, and Permissive modes.
    ArithmeticBadCode,
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
            Self::TruncatedBetweenScans { scans_decoded } => write!(
                f,
                "stream ended after {} complete scan(s); no EOI",
                scans_decoded
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
            Self::RestartMarkerResync { count } => {
                write!(f, "{} restart marker resync(s) during scan", count)
            }
            Self::ExtraneousBytesSkipped { count } => {
                write!(f, "{} extraneous byte(s) skipped between markers", count)
            }
            Self::ArithmeticBadCode => {
                write!(
                    f,
                    "arithmetic coding overflow (bad code); treated as end-of-scan"
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
    /// Values are clamped to \[0, 255\]. This is the standard decode path
    /// matching libjpeg-turbo / zune-jpeg behavior.
    #[default]
    Srgb8,

    /// f32 sRGB output (gamma-encoded, \[0,1\] nominal) with unclamped integer IDCT.
    ///
    /// Same speed as `Srgb8` but preserves ringing outside \[0, 255\] as values
    /// outside \[0.0, 1.0\]. Useful for high-quality resampling or compositing
    /// where clamping artifacts would be visible.
    SrgbF32,

    /// f32 linear light output (\[0,1\] nominal) with unclamped integer IDCT.
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
/// This is the main entry point for decoding. Create a `DecodeConfig`, configure
/// it with builder methods, then call [`decode()`](DecodeConfig::decode) or
/// [`scanline_reader()`](DecodeConfig::scanline_reader).
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
    /// Default is 120 megapixels (admits common ~108 MP camera photos). Set to 0 for unlimited.
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

    /// Caller preference for allocation fallibility, applied per call site.
    ///
    /// Internal carrier (`pub(crate)`): the zencodec decode path sets it from
    /// [`ResourceLimits::prefer_fallible_allocations`](zencodec::ResourceLimits::prefer_fallible_allocations)
    /// in `codec::build_decode_config`; the direct [`DecodeConfig`] API leaves
    /// it [`CodecDefault`](zencodec::AllocPreference::CodecDefault), so each
    /// decode allocation site keeps its own default (big untrusted output /
    /// coefficient buffers fallible, small bounded MCU scratch infallible).
    /// See [`crate::foundation::alloc`].
    pub(crate) alloc_pref: zencodec::AllocPreference,
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
            .field("alloc_pref", &self.alloc_pref)
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
            alloc_pref: zencodec::AllocPreference::CodecDefault,
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

    /// Apply an orientation transform to the decoded pixels in place.
    ///
    /// Pure pixel permutation (no resampling, no precision loss) on whichever
    /// buffer is present (u8 or f32), swapping `width`/`height` for
    /// dimension-swapping transforms. Used by the pixel-domain orientation
    /// fallback for images where the DCT-domain transform is not exact
    /// (subsampled chroma with non-MCU-aligned dimensions, issue #149).
    pub(crate) fn apply_pixel_transform(&mut self, transform: LosslessTransform) {
        if transform == LosslessTransform::None {
            return;
        }
        let w = self.width as usize;
        let h = self.height as usize;
        if let Some(px) = self.pixels_u8.take() {
            // Interleaved u8: stride unit is bytes_per_pixel (4 for Bgrx).
            let ch = self.format.bytes_per_pixel();
            self.pixels_u8 = Some(transform_interleaved(&px, w, h, ch, transform));
        }
        if let Some(px) = self.pixels_f32.take() {
            // f32 buffers are packed with one element per channel.
            let ch = self.format.num_channels();
            self.pixels_f32 = Some(transform_interleaved(&px, w, h, ch, transform));
        }
        if transform.swaps_dimensions() {
            core::mem::swap(&mut self.width, &mut self.height);
        }
    }

    /// Crop the decoded pixels in place to the given rectangle.
    ///
    /// Coordinates are in the current (post-transform) output space and must
    /// be fully inside the image.
    pub(crate) fn crop_in_place(&mut self, x: u32, y: u32, new_w: u32, new_h: u32) {
        let w = self.width as usize;
        let (x, y) = (x as usize, y as usize);
        let (new_w, new_h) = (new_w as usize, new_h as usize);
        if let Some(px) = self.pixels_u8.take() {
            let ch = self.format.bytes_per_pixel();
            self.pixels_u8 = Some(crop_interleaved(&px, w, x, y, new_w, new_h, ch));
        }
        if let Some(px) = self.pixels_f32.take() {
            let ch = self.format.num_channels();
            self.pixels_f32 = Some(crop_interleaved(&px, w, x, y, new_w, new_h, ch));
        }
        self.width = new_w as u32;
        self.height = new_h as u32;
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

/// Metadata returned by `DecodeConfig::decode_into_u8` and
/// `DecodeConfig::decode_into_f32`.
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

/// Apply an orientation transform to an interleaved pixel buffer.
///
/// `src` is `h` rows of `w` pixels with `ch` elements per pixel (tightly
/// packed). Returns a new buffer in the transformed geometry (dimensions
/// swap for Transpose/Rotate90/Rotate270/Transverse). Pure permutation —
/// every output pixel is copied verbatim from exactly one input pixel.
pub(crate) fn transform_interleaved<T: Copy>(
    src: &[T],
    w: usize,
    h: usize,
    ch: usize,
    transform: LosslessTransform,
) -> Vec<T> {
    debug_assert_eq!(src.len(), w * h * ch);
    let (ow, oh) = if transform.swaps_dimensions() {
        (h, w)
    } else {
        (w, h)
    };
    let mut dst = Vec::with_capacity(src.len());
    for oy in 0..oh {
        for ox in 0..ow {
            // Source pixel for output (ox, oy). All transforms follow EXIF
            // display semantics (Rotate90 = 90° clockwise, EXIF 6).
            let (sx, sy) = match transform {
                LosslessTransform::None => (ox, oy),
                LosslessTransform::FlipHorizontal => (w - 1 - ox, oy),
                LosslessTransform::FlipVertical => (ox, h - 1 - oy),
                LosslessTransform::Rotate180 => (w - 1 - ox, h - 1 - oy),
                LosslessTransform::Transpose => (oy, ox),
                LosslessTransform::Rotate90 => (oy, h - 1 - ox),
                LosslessTransform::Rotate270 => (w - 1 - oy, ox),
                LosslessTransform::Transverse => (w - 1 - oy, h - 1 - ox),
            };
            let s = (sy * w + sx) * ch;
            dst.extend_from_slice(&src[s..s + ch]);
        }
    }
    dst
}

/// Copy a sub-rectangle out of an interleaved pixel buffer.
fn crop_interleaved<T: Copy>(
    src: &[T],
    src_w: usize,
    x: usize,
    y: usize,
    new_w: usize,
    new_h: usize,
    ch: usize,
) -> Vec<T> {
    let mut dst = Vec::with_capacity(new_w * new_h * ch);
    for row in 0..new_h {
        let start = ((y + row) * src_w + x) * ch;
        dst.extend_from_slice(&src[start..start + new_w * ch]);
    }
    dst
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
