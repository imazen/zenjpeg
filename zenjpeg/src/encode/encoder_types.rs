//! Core types for the v2 encoder API.

/// Quality/compression setting.
///
/// All variants map to internal quality through empirical lookup tables.
/// Results vary by image - these are rough approximations, not guarantees.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Quality {
    /// Approximate jpegli quality scale (this is a fork, not exact jpegli).
    /// Range: 0.0–100.0, where ~90 is visually lossless for most images.
    ApproxJpegli(f32),

    /// Approximate mozjpeg quality behavior.
    /// Range: 0–100. Maps to quality producing similar file sizes.
    ApproxMozjpeg(u8),

    /// Approximate SSIMULACRA2 score target.
    /// Range: 0–100 (higher = better). 90+ is roughly visually lossless.
    ApproxSsim2(f32),

    /// Approximate Butteraugli distance target.
    /// Range: 0.0+ (lower = better). <1.0 excellent, <3.0 good.
    ApproxButteraugli(f32),
}

impl Default for Quality {
    fn default() -> Self {
        Quality::ApproxJpegli(90.0)
    }
}

impl From<f32> for Quality {
    fn from(q: f32) -> Self {
        Quality::ApproxJpegli(q)
    }
}

impl From<u8> for Quality {
    fn from(q: u8) -> Self {
        Quality::ApproxJpegli(q as f32)
    }
}

impl From<i32> for Quality {
    fn from(q: i32) -> Self {
        Quality::ApproxJpegli(q as f32)
    }
}

impl Quality {
    /// Convert to internal quality value (0.0-100.0 scale).
    #[must_use]
    pub fn to_internal(&self) -> f32 {
        match self {
            Quality::ApproxJpegli(q) => *q,
            Quality::ApproxMozjpeg(q) => mozjpeg_to_internal(*q),
            Quality::ApproxSsim2(score) => ssim2_to_internal(*score),
            Quality::ApproxButteraugli(dist) => butteraugli_to_internal(*dist),
        }
    }

    /// Convert to butteraugli distance.
    ///
    /// Uses the exact same formula as C++ jpegli's `jpegli_quality_to_distance`.
    #[must_use]
    pub fn to_distance(&self) -> f32 {
        // If already butteraugli distance, return it directly
        if let Quality::ApproxButteraugli(d) = self {
            return *d;
        }
        // Exact C++ jpegli formula from lib/jpegli/encode.cc:jpegli_quality_to_distance
        let q = self.to_internal();
        if q >= 100.0 {
            0.01
        } else if q >= 30.0 {
            0.1 + (100.0 - q) * 0.09
        } else {
            // Quadratic for very low quality
            53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
        }
    }
}

// Calibrated mozjpeg→jpegli quality mapping (4:4:4, DSSIM metric)
// From corpus testing on CID22-512 and Kodak datasets
const MOZJPEG_TO_JPEGLI: [(u8, u8); 10] = [
    (30, 28),
    (40, 37),
    (50, 47),
    (60, 55),
    (70, 65),
    (75, 71),
    (80, 77),
    (85, 83),
    (90, 89),
    (95, 94),
];

fn mozjpeg_to_internal(q: u8) -> f32 {
    if q >= 100 {
        return 100.0;
    }
    if q <= 30 {
        // Extrapolate below table range
        return (q as f32 / 30.0) * 28.0;
    }

    // Find bracketing entries and interpolate
    let mut lower = (30u8, 28u8);
    let mut upper = (95u8, 94u8);

    for &(moz_q, jpegli_q) in &MOZJPEG_TO_JPEGLI {
        if moz_q <= q && moz_q > lower.0 {
            lower = (moz_q, jpegli_q);
        }
        if moz_q >= q && moz_q < upper.0 {
            upper = (moz_q, jpegli_q);
        }
    }

    if lower.0 == upper.0 {
        return lower.1 as f32;
    }

    // Linear interpolation
    let t = (q - lower.0) as f32 / (upper.0 - lower.0) as f32;
    lower.1 as f32 + t * (upper.1 as f32 - lower.1 as f32)
}

// Calibrated SSIMULACRA2→jpegli quality mapping (4:4:4)
// SSIM2 scores: higher is better, 100 = identical
const SSIM2_TO_JPEGLI: [(u8, u8); 8] = [
    (70, 55), // Low quality
    (75, 65),
    (80, 73),
    (85, 80),
    (88, 85),
    (90, 88),
    (93, 92),
    (95, 95),
];

fn ssim2_to_internal(score: f32) -> f32 {
    if score >= 100.0 {
        return 100.0;
    }
    if score <= 70.0 {
        return (score / 70.0) * 55.0;
    }

    let q = score as u8;
    let mut lower = (70u8, 55u8);
    let mut upper = (95u8, 95u8);

    for &(ssim_score, jpegli_q) in &SSIM2_TO_JPEGLI {
        if ssim_score <= q && ssim_score > lower.0 {
            lower = (ssim_score, jpegli_q);
        }
        if ssim_score >= q && ssim_score < upper.0 {
            upper = (ssim_score, jpegli_q);
        }
    }

    if lower.0 == upper.0 {
        return lower.1 as f32;
    }

    let t = (score - lower.0 as f32) / (upper.0 - lower.0) as f32;
    lower.1 as f32 + t * (upper.1 as f32 - lower.1 as f32)
}

// Calibrated butteraugli→jpegli quality mapping
// Butteraugli: lower is better, 0 = identical, <1 excellent, <3 good
const BUTTERAUGLI_TO_JPEGLI: [(f32, f32); 7] = [
    (0.3, 96.0),
    (0.5, 93.0),
    (1.0, 88.0),
    (1.5, 82.0),
    (2.0, 76.0),
    (3.0, 68.0),
    (5.0, 55.0),
];

fn butteraugli_to_internal(dist: f32) -> f32 {
    if dist <= 0.0 {
        return 100.0;
    }
    if dist <= 0.3 {
        return 96.0 + (0.3 - dist) / 0.3 * 4.0;
    }
    if dist >= 5.0 {
        return 55.0 - (dist - 5.0) * 3.0;
    }

    let mut lower = (0.3f32, 96.0f32);
    let mut upper = (5.0f32, 55.0f32);

    for &(ba_dist, jpegli_q) in &BUTTERAUGLI_TO_JPEGLI {
        if ba_dist <= dist && ba_dist > lower.0 {
            lower = (ba_dist, jpegli_q);
        }
        if ba_dist >= dist && ba_dist < upper.0 {
            upper = (ba_dist, jpegli_q);
        }
    }

    if (lower.0 - upper.0).abs() < 0.001 {
        return lower.1;
    }

    let t = (dist - lower.0) / (upper.0 - lower.0);
    lower.1 + t * (upper.1 - lower.1)
}

/// Output color space with bundled subsampling options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ColorMode {
    /// Standard YCbCr with configurable chroma subsampling.
    YCbCr { subsampling: ChromaSubsampling },

    /// XYB perceptual color space (jpegli-specific).
    /// Computed internally from linear RGB input.
    Xyb { subsampling: XybSubsampling },

    /// Single-channel grayscale.
    Grayscale,
}

impl Default for ColorMode {
    fn default() -> Self {
        ColorMode::YCbCr {
            subsampling: ChromaSubsampling::None, // 4:4:4 - no subsampling
        }
    }
}

/// YCbCr chroma subsampling (spatial resolution).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// 4:4:4 - No subsampling (full chroma resolution, best quality, largest files)
    None,
    /// 4:2:2 - Half horizontal resolution
    HalfHorizontal,
    /// 4:2:0 - Quarter resolution (half each direction, most common)
    Quarter,
    /// 4:4:0 - Half vertical resolution
    HalfVertical,
}

impl ChromaSubsampling {
    /// Horizontal subsampling factor (1 or 2).
    #[must_use]
    pub const fn h_factor(&self) -> u8 {
        match self {
            ChromaSubsampling::None | ChromaSubsampling::HalfVertical => 1,
            ChromaSubsampling::HalfHorizontal | ChromaSubsampling::Quarter => 2,
        }
    }

    /// Vertical subsampling factor (1 or 2).
    #[must_use]
    pub const fn v_factor(&self) -> u8 {
        match self {
            ChromaSubsampling::None | ChromaSubsampling::HalfHorizontal => 1,
            ChromaSubsampling::HalfVertical | ChromaSubsampling::Quarter => 2,
        }
    }

    /// Returns the horizontal sampling factor for luma.
    ///
    /// This is the luma block count in horizontal direction per MCU.
    /// Returns 1 for 4:4:4/4:4:0, returns 2 for 4:2:0/4:2:2.
    #[must_use]
    pub const fn h_samp_factor_luma(self) -> u8 {
        self.h_factor()
    }

    /// Returns the vertical sampling factor for luma.
    ///
    /// This is the luma block count in vertical direction per MCU.
    /// Returns 1 for 4:4:4/4:2:2, returns 2 for 4:2:0/4:4:0.
    #[must_use]
    pub const fn v_samp_factor_luma(self) -> u8 {
        self.v_factor()
    }

    /// Returns the MCU (Minimum Coded Unit) size for this subsampling mode.
    ///
    /// - 8 for 4:4:4 (no subsampling)
    /// - 16 for modes with 2x sampling (4:2:0, 4:2:2, 4:4:0)
    #[must_use]
    pub const fn mcu_size(self) -> usize {
        match self {
            ChromaSubsampling::None => 8,
            ChromaSubsampling::Quarter
            | ChromaSubsampling::HalfHorizontal
            | ChromaSubsampling::HalfVertical => 16,
        }
    }
}

/// XYB component subsampling.
///
/// Unlike YCbCr where only luma is full, XYB keeps X and Y full
/// even in subsampled mode - only B is reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum XybSubsampling {
    /// X, Y, B all at full resolution (1x1, 1x1, 1x1)
    Full,
    /// X, Y full, B at quarter resolution (1x1, 1x1, 2x2)
    #[default]
    BQuarter,
}

/// Chroma downsampling algorithm for RGB->YCbCr conversion.
///
/// **Only applies to RGB/RGBX input.** Ignored for grayscale, YCbCr, and planar input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum DownsamplingMethod {
    /// Simple box filter averaging (fast, matches C++ jpegli default)
    #[default]
    Box,
    /// Gamma-aware averaging (better color accuracy at edges)
    GammaAware,
    /// Iterative optimization (SharpYUV-style, best quality, ~3x slower)
    GammaAwareIterative,
}

impl DownsamplingMethod {
    /// Returns true if this method uses gamma-aware downsampling.
    #[must_use]
    pub const fn uses_gamma_aware(self) -> bool {
        matches!(self, Self::GammaAware | Self::GammaAwareIterative)
    }
}

/// Pixel data layout for raw byte input.
///
/// Describes channel order, bit depth, and color space interpretation.
/// Use with `encode_from_bytes()` when working with raw buffers.
///
/// For rgb crate types, use `encode_from_rgb()` which infers layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout {
    // === 8-bit sRGB (gamma-encoded) ===
    /// RGB, 3 bytes/pixel, sRGB gamma
    Rgb8Srgb,
    /// BGR, 3 bytes/pixel, sRGB gamma (Windows/GDI order)
    Bgr8Srgb,
    /// RGBX, 4 bytes/pixel, sRGB gamma (4th byte ignored)
    Rgbx8Srgb,
    /// BGRX, 4 bytes/pixel, sRGB gamma (4th byte ignored)
    Bgrx8Srgb,
    /// Grayscale, 1 byte/pixel, sRGB gamma
    Gray8Srgb,

    // === 16-bit linear ===
    /// RGB, 6 bytes/pixel, linear light (0-65535)
    Rgb16Linear,
    /// RGBX, 8 bytes/pixel, linear light (4th channel ignored)
    Rgbx16Linear,
    /// Grayscale, 2 bytes/pixel, linear light
    Gray16Linear,

    // === 32-bit float linear ===
    /// RGB, 12 bytes/pixel, linear light (0.0-1.0)
    RgbF32Linear,
    /// RGBX, 16 bytes/pixel, linear light (4th channel ignored)
    RgbxF32Linear,
    /// Grayscale, 4 bytes/pixel, linear light
    GrayF32Linear,

    // === Pre-converted YCbCr (skip RGB->YCbCr conversion) ===
    /// YCbCr interleaved, 3 bytes/pixel, u8
    YCbCr8,
    /// YCbCr interleaved, 12 bytes/pixel, f32
    YCbCrF32,
}

impl PixelLayout {
    /// Bytes per pixel for this layout.
    #[must_use]
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Gray8Srgb => 1,
            Self::Gray16Linear => 2,
            Self::Rgb8Srgb | Self::Bgr8Srgb | Self::YCbCr8 => 3,
            Self::Rgbx8Srgb | Self::Bgrx8Srgb | Self::GrayF32Linear => 4,
            Self::Rgb16Linear => 6,
            Self::Rgbx16Linear => 8,
            Self::RgbF32Linear | Self::YCbCrF32 => 12,
            Self::RgbxF32Linear => 16,
        }
    }

    /// Number of channels (including ignored channels).
    #[must_use]
    pub const fn channels(&self) -> usize {
        match self {
            Self::Gray8Srgb | Self::Gray16Linear | Self::GrayF32Linear => 1,
            Self::Rgb8Srgb
            | Self::Bgr8Srgb
            | Self::Rgb16Linear
            | Self::RgbF32Linear
            | Self::YCbCr8
            | Self::YCbCrF32 => 3,
            Self::Rgbx8Srgb | Self::Bgrx8Srgb | Self::Rgbx16Linear | Self::RgbxF32Linear => 4,
        }
    }

    /// Whether this is a grayscale format.
    #[must_use]
    pub const fn is_grayscale(&self) -> bool {
        matches!(
            self,
            Self::Gray8Srgb | Self::Gray16Linear | Self::GrayF32Linear
        )
    }

    /// Whether this is pre-converted YCbCr.
    #[must_use]
    pub const fn is_ycbcr(&self) -> bool {
        matches!(self, Self::YCbCr8 | Self::YCbCrF32)
    }

    /// Whether this uses BGR channel order.
    #[must_use]
    pub const fn is_bgr(&self) -> bool {
        matches!(self, Self::Bgr8Srgb | Self::Bgrx8Srgb)
    }

    /// Whether this is a float format (linear color space).
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::RgbF32Linear | Self::RgbxF32Linear | Self::GrayF32Linear | Self::YCbCrF32
        )
    }

    /// Whether this is a 16-bit format (linear color space).
    #[must_use]
    pub const fn is_16bit(&self) -> bool {
        matches!(
            self,
            Self::Rgb16Linear | Self::Rgbx16Linear | Self::Gray16Linear
        )
    }
}

/// Planar YCbCr data for a strip of rows.
///
/// Each plane has its own stride. All planes are f32.
#[derive(Clone, Copy, Debug)]
pub struct YCbCrPlanes<'a> {
    pub y: &'a [f32],
    pub y_stride: usize,
    pub cb: &'a [f32],
    pub cb_stride: usize,
    pub cr: &'a [f32],
    pub cr_stride: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_default() {
        let q = Quality::default();
        assert!(matches!(q, Quality::ApproxJpegli(90.0)));
    }

    #[test]
    fn test_quality_from() {
        let q: Quality = 85.0.into();
        assert!(matches!(q, Quality::ApproxJpegli(85.0)));

        let q: Quality = 75u8.into();
        assert!(matches!(q, Quality::ApproxJpegli(75.0)));
    }

    #[test]
    fn test_pixel_layout_bytes() {
        assert_eq!(PixelLayout::Rgb8Srgb.bytes_per_pixel(), 3);
        assert_eq!(PixelLayout::Rgbx8Srgb.bytes_per_pixel(), 4);
        assert_eq!(PixelLayout::RgbF32Linear.bytes_per_pixel(), 12);
        assert_eq!(PixelLayout::Gray8Srgb.bytes_per_pixel(), 1);
    }

    #[test]
    fn test_chroma_subsampling_factors() {
        assert_eq!(ChromaSubsampling::None.h_factor(), 1);
        assert_eq!(ChromaSubsampling::None.v_factor(), 1);
        assert_eq!(ChromaSubsampling::Quarter.h_factor(), 2);
        assert_eq!(ChromaSubsampling::Quarter.v_factor(), 2);
        assert_eq!(ChromaSubsampling::HalfHorizontal.h_factor(), 2);
        assert_eq!(ChromaSubsampling::HalfHorizontal.v_factor(), 1);
    }
}

// =============================================================================
// Parallel Encoding Configuration
// =============================================================================

/// Parallel encoding strategy.
///
/// Controls how the encoder uses multiple threads for improved throughput.
/// Parallel encoding uses JPEG restart markers to enable independent encoding
/// of image segments, which are then concatenated.
///
/// # Restart Marker Behavior
///
/// Parallel encoding requires restart markers between segments. When enabled:
/// - If `restart_interval` is 0 or too small, it will be **increased** to an
///   optimal value based on thread count and image size
/// - If `restart_interval` is already set to a reasonable value, it will be
///   preserved (parallel encoding respects user-specified intervals)
///
/// Restart markers add ~2 bytes per interval but enable:
/// - Parallel encoding/decoding
/// - Error recovery in corrupted streams
/// - Random access to image regions
///
/// # Performance
///
/// Parallel encoding is most beneficial for larger images (512x512+):
/// - 2 threads: ~1.2-1.6x speedup
/// - 4 threads: ~1.3-1.7x speedup
/// - Diminishing returns beyond 4 threads for typical images
///
/// Small images (<256x256) may see no benefit or slight overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg(feature = "parallel")]
pub enum ParallelEncoding {
    /// Automatically configure parallel encoding.
    ///
    /// Uses available CPU cores and selects an optimal restart interval
    /// based on image dimensions. The restart interval will be increased
    /// if needed, but never decreased below the user-specified value.
    #[default]
    Auto,
}

/// Huffman table strategy for entropy encoding.
///
/// Controls how Huffman tables are selected for the JPEG output:
/// - `Optimize`: Two-pass encoding collects symbol frequencies, then builds optimal tables.
///   Produces the smallest files. Required for progressive mode.
/// - `Fixed`: Single-pass encoding with general-purpose trained Huffman tables.
///   Fastest, with ~2.5% overhead vs per-image optimal.
/// - `Custom`: Single-pass encoding with caller-provided tables.
///   Use [`HuffmanTableSet::annex_k()`] for the original JPEG standard tables,
///   or provide your own pre-computed tables.
#[derive(Clone, Debug, Default)]
pub(crate) enum HuffmanStrategy {
    /// Two-pass: collect frequencies, build optimal tables.
    #[default]
    Optimize,
    /// Single-pass: use general-purpose trained tables.
    Fixed,
    /// Single-pass: use caller-provided tables.
    Custom(Box<crate::huffman::optimize::HuffmanTableSet>),
}

/// Progressive scan script strategy.
///
/// Controls how progressive JPEG scan scripts are generated. All progressive
/// modes use optimized Huffman tables (two-pass encoding).
///
/// # Baseline JPEG
///
/// For baseline (non-progressive) encoding, this setting has no effect.
/// Use `.progressive(false)` to disable progressive mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScanStrategy {
    /// Use encoder's default progressive script (jpegli-style).
    ///
    /// - Frequency split at AC coefficients 2/3
    /// - Successive approximation for all components (Al=2 → 1 → 0)
    /// - Separate DC scans per component
    ///
    /// This is the default when `.progressive(true)` is used.
    #[default]
    Default,

    /// Search 64 candidate scripts and pick the smallest (mozjpeg-style).
    ///
    /// Tests combinations of:
    /// - Frequency splits at [2, 5, 8, 12, 18]
    /// - Successive approximation levels 0-3 for luma, 0-2 for chroma
    /// - DC scan interleaving options
    ///
    /// Typically saves 1-3% vs fixed scripts, at the cost of ~2x encode time.
    Search,

    /// mozjpeg's default progressive script.
    ///
    /// - Frequency split at AC coefficients 8/9
    /// - No successive approximation for chroma (full precision in one pass)
    /// - Separate DC scans per component
    ///
    /// Use this for closest parity with C mozjpeg's default progressive output.
    Mozjpeg,
}
