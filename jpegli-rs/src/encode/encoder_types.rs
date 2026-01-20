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

    /// Create quality from a 0-100 scale value.
    ///
    /// This is equivalent to `Quality::ApproxJpegli(q)` or `Quality::from(q)`.
    #[deprecated(
        since = "0.5.0",
        note = "Use Quality::from(f32) or Quality::ApproxJpegli(f32) instead"
    )]
    #[must_use]
    pub fn from_quality(q: f32) -> Self {
        Quality::ApproxJpegli(q)
    }

    /// Create quality from butteraugli distance (backward compatibility).
    ///
    /// This is equivalent to `Quality::ApproxButteraugli(d)`.
    #[deprecated(since = "0.5.0", note = "Use Quality::ApproxButteraugli(f32) instead")]
    #[must_use]
    pub fn from_distance(d: f32) -> Self {
        Quality::ApproxButteraugli(d)
    }

    /// Backward-compatible constructor matching old `Quality::Traditional(f32)`.
    #[deprecated(since = "0.5.0", note = "Use Quality::ApproxJpegli(f32) instead")]
    #[must_use]
    #[allow(non_snake_case)] // Mimics old enum variant naming
    pub fn Traditional(q: f32) -> Self {
        Quality::ApproxJpegli(q)
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

/// Quantization table configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
#[derive(Default)]
#[allow(clippy::large_enum_variant)] // Custom matrices are rarely used
pub enum QuantTableConfig {
    /// Jpegli's perceptual tables, scaled by Quality. (default)
    #[default]
    Perceptual,

    /// Custom base matrices, scaled by Quality.
    /// Provide f32 matrices (typically 1.0–255.0 range).
    /// These are multiplied by the quality-derived scale factor.
    CustomBase {
        /// Luma (Y) base matrix - 64 coefficients in row-major order
        luma: [f32; 64],
        /// Blue chroma (Cb) base matrix - 64 coefficients in row-major order
        cb: [f32; 64],
        /// Red chroma (Cr) base matrix - 64 coefficients in row-major order
        cr: [f32; 64],
    },

    /// Exact quantization tables. **Quality is ignored.**
    /// Values should be in range 1-255 for baseline JPEG compatibility.
    Exact {
        /// Luma (Y) quantization table - 64 coefficients in row-major order
        luma: [u16; 64],
        /// Blue chroma (Cb) quantization table - 64 coefficients in row-major order
        cb: [u16; 64],
        /// Red chroma (Cr) quantization table - 64 coefficients in row-major order
        cr: [u16; 64],
    },
}

/// Zero-bias configuration for quantization.
///
/// Zero-bias controls how coefficients are rounded toward zero during quantization.
/// Higher multipliers mean more aggressive zeroing of small coefficients, which
/// improves compression but may reduce quality.
///
/// The jpegli defaults are perceptually optimized and vary by:
/// - Quality level (blends between HQ and LQ tables based on butteraugli distance)
/// - Component (Y, Cb, Cr have different optimal bias values)
/// - Coefficient position (different DCT frequencies have different biases)
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)] // Custom tables are rarely used
pub enum ZeroBiasConfig {
    /// Auto-select based on color mode (default).
    ///
    /// - **YCbCr mode**: Uses quality-adaptive perceptual tables that blend
    ///   between HQ and LQ tables based on the effective butteraugli distance.
    /// - **XYB mode**: Uses simple 0.5/0.5 mul/offset for all AC coefficients.
    ///
    /// This matches C++ jpegli behavior exactly.
    #[default]
    Default,

    /// Force YCbCr quality-adaptive perceptual tables.
    ///
    /// Blends between HQ and LQ tables based on the effective butteraugli
    /// distance. Use this to force YCbCr tables even in XYB mode.
    YCbCr,

    /// Force XYB zero-bias (0.5 multiplier, 0.5 offset for all AC coefficients).
    ///
    /// DC coefficients use 0.0/0.0. This matches what C++ jpegli uses
    /// internally for XYB mode. Use this to force XYB tables even in YCbCr mode.
    Xyb,

    /// Disable zero-bias (all multipliers = 0, all offsets = 0).
    /// This matches standard JPEG behavior without adaptive quantization.
    Disabled,

    /// Custom zero-bias tables.
    /// Each component has 64 multiplier values and 64 offset values.
    Custom {
        /// Luma (Y) zero-bias: (multipliers[64], offsets[64])
        luma: ([f32; 64], [f32; 64]),
        /// Blue chroma (Cb) zero-bias: (multipliers[64], offsets[64])
        cb: ([f32; 64], [f32; 64]),
        /// Red chroma (Cr) zero-bias: (multipliers[64], offsets[64])
        cr: ([f32; 64], [f32; 64]),
    },
}

impl QuantTableConfig {
    /// Convert to internal CustomQuantMatrices format.
    ///
    /// Returns `None` for `Perceptual` (use defaults), or `Some` with the
    /// appropriate internal representation.
    #[must_use]
    pub(crate) fn to_custom_matrices(&self) -> Option<crate::quant::CustomQuantMatrices> {
        use crate::quant::CustomQuantMatrices;

        match self {
            QuantTableConfig::Perceptual => None,
            QuantTableConfig::CustomBase { luma, cb, cr } => {
                // Pack into 192-element array: Y[64], Cb[64], Cr[64]
                let mut matrix = [0.0f32; 192];
                matrix[..64].copy_from_slice(luma);
                matrix[64..128].copy_from_slice(cb);
                matrix[128..192].copy_from_slice(cr);
                Some(CustomQuantMatrices::new().with_ycbcr(matrix))
            }
            QuantTableConfig::Exact { luma, cb, cr } => {
                // Pack into 192-element array: Y[64], Cb[64], Cr[64]
                let mut tables = [0u16; 192];
                tables[..64].copy_from_slice(luma);
                tables[64..128].copy_from_slice(cb);
                tables[128..192].copy_from_slice(cr);
                Some(CustomQuantMatrices::new().with_direct_tables(tables))
            }
        }
    }
}

impl ZeroBiasConfig {
    /// Convert to internal ZeroBiasParams for a specific component.
    ///
    /// # Arguments
    /// * `component` - 0 for Y, 1 for Cb, 2 for Cr
    /// * `distance` - Butteraugli distance (used for YCbCr perceptual tables)
    /// * `is_xyb` - Whether encoding in XYB mode (used for `Default` variant)
    #[must_use]
    pub(crate) fn to_zero_bias_params(
        &self,
        component: usize,
        distance: f32,
        is_xyb: bool,
    ) -> crate::quant::ZeroBiasParams {
        use crate::quant::ZeroBiasParams;

        match self {
            ZeroBiasConfig::Default => {
                // Auto-select based on color mode (matches C++ jpegli behavior)
                if is_xyb {
                    ZeroBiasParams::for_xyb()
                } else {
                    ZeroBiasParams::for_ycbcr(distance, component)
                }
            }
            ZeroBiasConfig::YCbCr => ZeroBiasParams::for_ycbcr(distance, component),
            ZeroBiasConfig::Xyb => ZeroBiasParams::for_xyb(),
            ZeroBiasConfig::Disabled => ZeroBiasParams::default(),
            ZeroBiasConfig::Custom { luma, cb, cr } => {
                let (mul, offset) = match component {
                    0 => luma,
                    1 => cb,
                    _ => cr,
                };
                ZeroBiasParams {
                    mul: *mul,
                    offset: *offset,
                }
            }
        }
    }
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
    /// Convert to the legacy Subsampling enum.
    #[must_use]
    pub fn to_legacy(&self) -> crate::types::Subsampling {
        match self {
            ChromaSubsampling::None => crate::types::Subsampling::S444,
            ChromaSubsampling::HalfHorizontal => crate::types::Subsampling::S422,
            ChromaSubsampling::Quarter => crate::types::Subsampling::S420,
            ChromaSubsampling::HalfVertical => crate::types::Subsampling::S440,
        }
    }

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

    /// Convert to legacy PixelFormat (best effort).
    #[must_use]
    pub fn to_legacy(&self) -> crate::types::PixelFormat {
        match self {
            Self::Rgb8Srgb => crate::types::PixelFormat::Rgb,
            Self::Bgr8Srgb => crate::types::PixelFormat::Bgr,
            Self::Rgbx8Srgb => crate::types::PixelFormat::Rgba,
            Self::Bgrx8Srgb => crate::types::PixelFormat::Bgrx,
            Self::Gray8Srgb => crate::types::PixelFormat::Gray,
            Self::Rgb16Linear => crate::types::PixelFormat::Rgb16,
            Self::Rgbx16Linear => crate::types::PixelFormat::Rgba16,
            Self::Gray16Linear => crate::types::PixelFormat::Gray16,
            Self::RgbF32Linear => crate::types::PixelFormat::RgbF32,
            Self::RgbxF32Linear => crate::types::PixelFormat::RgbaF32,
            Self::GrayF32Linear => crate::types::PixelFormat::GrayF32,
            // YCbCr layouts don't have direct legacy equivalents
            Self::YCbCr8 | Self::YCbCrF32 => crate::types::PixelFormat::Rgb,
        }
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
