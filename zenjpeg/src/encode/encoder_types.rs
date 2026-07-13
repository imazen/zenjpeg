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

    /// Target perceptual quality in `zq` units (issue #113).
    ///
    /// `zq` is calibrated against zensim score so that the encoder lands
    /// at roughly the same PERCEIVED quality regardless of content type.
    /// Unlike `ApproxJpegli` (which is a fixed algorithmic knob), `Zq`
    /// triggers a closed-loop encode: the encoder iterates per-block
    /// quantization until the achieved zensim score is at or above
    /// `target` (within the default budget).
    ///
    /// Range: ~30–100 useful, with 75–90 the typical web/CDN range.
    /// Uses [`zq::ZqTarget::default()`] for everything except the target
    /// value; for explicit budget/strictness control, use
    /// [`Quality::ZqExplicit`].
    Zq(f32),

    /// Target perceptual quality with explicit iteration policy
    /// (issue #113). See [`zq::ZqTarget`] for the per-knob semantics.
    ZqExplicit(crate::encode::zq::ZqTarget),
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
    ///
    /// For [`Quality::Zq`] / [`Quality::ZqExplicit`] this returns the
    /// STARTING jpegli quality for the iteration loop's first pass, NOT
    /// the eventual achieved quality. The closed-loop encoder adjusts
    /// per-block AQ from this baseline.
    #[must_use]
    pub fn to_internal(&self) -> f32 {
        match self {
            Quality::ApproxJpegli(q) => *q,
            Quality::ApproxMozjpeg(q) => mozjpeg_to_internal(*q),
            Quality::ApproxSsim2(score) => ssim2_to_internal(*score),
            Quality::ApproxButteraugli(dist) => butteraugli_to_internal(*dist),
            Quality::Zq(zq) => crate::encode::zq::zq_to_starting_jpegli_q(*zq),
            Quality::ZqExplicit(t) => crate::encode::zq::zq_to_starting_jpegli_q(t.target),
        }
    }

    /// Whether this quality variant triggers the closed-loop iteration
    /// path (i.e. requires the decoder + zensim measurement on each
    /// pass). Returns `false` for the existing one-shot variants.
    #[must_use]
    pub fn is_zq_target(&self) -> bool {
        matches!(self, Quality::Zq(_) | Quality::ZqExplicit(_))
    }

    /// Resolve to a [`crate::encode::zq::ZqTarget`] when this is a
    /// closed-loop variant. Returns `None` for the one-shot variants.
    #[must_use]
    pub fn zq_target(&self) -> Option<crate::encode::zq::ZqTarget> {
        match self {
            Quality::Zq(zq) => Some(crate::encode::zq::ZqTarget::new(*zq)),
            Quality::ZqExplicit(t) => Some(*t),
            _ => None,
        }
    }

    /// Quality value for mozjpeg's Robidoux table scaling formula.
    ///
    /// For `ApproxMozjpeg(q)`, returns `q` unchanged — the user specified a
    /// mozjpeg quality and expects tables matching `jpeg_set_quality(q)` with
    /// Robidoux base tables.
    ///
    /// For other variants, falls back to `to_internal()` as a best-effort
    /// mapping. This is imprecise (jpegli and mozjpeg quality scales differ
    /// substantially) but there's no lossless conversion between them.
    #[must_use]
    pub fn for_mozjpeg_tables(&self) -> u8 {
        match self {
            Quality::ApproxMozjpeg(q) => *q,
            _ => self.to_internal().round().clamp(1.0, 100.0) as u8,
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

    /// RGB stored without color transformation (issue #185).
    ///
    /// Channels pass through to JPEG components R, G, B verbatim (only
    /// DCT quantization is lossy) — no RGB→YCbCr matrix, no chroma
    /// subsampling (always 4:4:4). Signaled with component IDs
    /// 'R','G','B' and an Adobe APP14 marker with transform=0, matching
    /// libjpeg's `JCS_RGB`. For channel-packed data (e.g. microscopy
    /// stains) where cross-channel bleed from a color transform is
    /// unacceptable.
    Rgb,
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
    /// RGBA, 4 bytes/pixel, sRGB gamma (alpha ignored on encode)
    Rgba8Srgb,
    /// BGRX, 4 bytes/pixel, sRGB gamma (4th byte ignored)
    Bgrx8Srgb,
    /// BGRA, 4 bytes/pixel, sRGB gamma (alpha ignored on encode)
    Bgra8Srgb,
    /// Grayscale, 1 byte/pixel, sRGB gamma
    Gray8Srgb,

    // === 16-bit linear ===
    /// RGB, 6 bytes/pixel, linear light (0-65535)
    Rgb16Linear,
    /// RGBX, 8 bytes/pixel, linear light (4th channel ignored)
    Rgbx16Linear,
    /// RGBA, 8 bytes/pixel, linear light (alpha ignored on encode)
    Rgba16Linear,
    /// Grayscale, 2 bytes/pixel, linear light
    Gray16Linear,

    // === 32-bit float linear ===
    /// RGB, 12 bytes/pixel, linear light (0.0-1.0)
    RgbF32Linear,
    /// RGBX, 16 bytes/pixel, linear light (4th channel ignored)
    RgbxF32Linear,
    /// RGBA, 16 bytes/pixel, linear light (alpha ignored on encode)
    RgbaF32Linear,
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
            Self::Rgbx8Srgb
            | Self::Rgba8Srgb
            | Self::Bgrx8Srgb
            | Self::Bgra8Srgb
            | Self::GrayF32Linear => 4,
            Self::Rgb16Linear => 6,
            Self::Rgbx16Linear | Self::Rgba16Linear => 8,
            Self::RgbF32Linear | Self::YCbCrF32 => 12,
            Self::RgbxF32Linear | Self::RgbaF32Linear => 16,
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
            Self::Rgbx8Srgb
            | Self::Rgba8Srgb
            | Self::Bgrx8Srgb
            | Self::Bgra8Srgb
            | Self::Rgbx16Linear
            | Self::Rgba16Linear
            | Self::RgbxF32Linear
            | Self::RgbaF32Linear => 4,
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
        matches!(self, Self::Bgr8Srgb | Self::Bgrx8Srgb | Self::Bgra8Srgb)
    }

    /// Whether this is a float format (linear color space).
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(
            self,
            Self::RgbF32Linear
                | Self::RgbxF32Linear
                | Self::RgbaF32Linear
                | Self::GrayF32Linear
                | Self::YCbCrF32
        )
    }

    /// Whether this is a 16-bit format (linear color space).
    #[must_use]
    pub const fn is_16bit(&self) -> bool {
        matches!(
            self,
            Self::Rgb16Linear | Self::Rgbx16Linear | Self::Rgba16Linear | Self::Gray16Linear
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
    use crate::types::PixelFormat;

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

    #[test]
    fn test_optimization_preset_all_variants() {
        let all: Vec<_> = OptimizationPreset::all().collect();
        assert_eq!(all.len(), 8);
        // No duplicates
        let mut set = std::collections::HashSet::new();
        for p in &all {
            assert!(set.insert(p), "duplicate preset: {:?}", p);
        }
    }

    #[test]
    fn test_optimization_preset_progressive_jpegli() {
        assert!(!OptimizationPreset::JpegliBaseline.is_progressive());
        assert!(OptimizationPreset::JpegliProgressive.is_progressive());
    }
    #[test]
    fn test_optimization_preset_progressive_trellis() {
        assert!(!OptimizationPreset::MozjpegBaseline.is_progressive());
        assert!(OptimizationPreset::MozjpegProgressive.is_progressive());
        assert!(OptimizationPreset::MozjpegMaxCompression.is_progressive());
        assert!(!OptimizationPreset::HybridBaseline.is_progressive());
        assert!(OptimizationPreset::HybridProgressive.is_progressive());
        assert!(OptimizationPreset::HybridMaxCompression.is_progressive());
    }

    #[test]
    fn test_optimization_preset_trellis_jpegli() {
        assert!(!OptimizationPreset::JpegliBaseline.uses_trellis());
        assert!(!OptimizationPreset::JpegliProgressive.uses_trellis());
    }
    #[test]
    fn test_optimization_preset_trellis() {
        assert!(OptimizationPreset::MozjpegBaseline.uses_trellis());
        assert!(OptimizationPreset::MozjpegProgressive.uses_trellis());
        assert!(OptimizationPreset::MozjpegMaxCompression.uses_trellis());
        assert!(OptimizationPreset::HybridBaseline.uses_trellis());
        assert!(OptimizationPreset::HybridProgressive.uses_trellis());
        assert!(OptimizationPreset::HybridMaxCompression.uses_trellis());
    }

    #[test]
    fn test_optimization_preset_aq_jpegli() {
        assert!(OptimizationPreset::JpegliBaseline.uses_aq());
        assert!(OptimizationPreset::JpegliProgressive.uses_aq());
    }
    #[test]
    fn test_optimization_preset_aq_trellis() {
        assert!(!OptimizationPreset::MozjpegBaseline.uses_aq());
        assert!(!OptimizationPreset::MozjpegProgressive.uses_aq());
        assert!(!OptimizationPreset::MozjpegMaxCompression.uses_aq());
        assert!(OptimizationPreset::HybridBaseline.uses_aq());
        assert!(OptimizationPreset::HybridProgressive.uses_aq());
        assert!(OptimizationPreset::HybridMaxCompression.uses_aq());
    }

    #[test]
    fn test_optimization_preset_scan_strategy_jpegli() {
        assert_eq!(
            OptimizationPreset::JpegliBaseline.scan_strategy(),
            ScanStrategy::Default
        );
    }
    #[test]
    fn test_optimization_preset_scan_strategy_trellis() {
        assert_eq!(
            OptimizationPreset::MozjpegProgressive.scan_strategy(),
            ScanStrategy::Mozjpeg
        );
        assert_eq!(
            OptimizationPreset::MozjpegMaxCompression.scan_strategy(),
            ScanStrategy::Search
        );
        assert_eq!(
            OptimizationPreset::HybridMaxCompression.scan_strategy(),
            ScanStrategy::Search
        );
    }

    #[test]
    fn test_optimization_preset_quant_table_source_jpegli() {
        assert_eq!(
            OptimizationPreset::JpegliBaseline.quant_table_source(),
            QuantTableSource::Jpegli
        );
        assert_eq!(
            OptimizationPreset::JpegliProgressive.quant_table_source(),
            QuantTableSource::Jpegli
        );
    }
    #[test]
    fn test_optimization_preset_quant_table_source_trellis() {
        assert_eq!(
            OptimizationPreset::MozjpegBaseline.quant_table_source(),
            QuantTableSource::MozjpegDefault
        );
        assert_eq!(
            OptimizationPreset::MozjpegProgressive.quant_table_source(),
            QuantTableSource::MozjpegDefault
        );
        assert_eq!(
            OptimizationPreset::MozjpegMaxCompression.quant_table_source(),
            QuantTableSource::MozjpegDefault
        );
        assert_eq!(
            OptimizationPreset::HybridBaseline.quant_table_source(),
            QuantTableSource::Jpegli
        );
        assert_eq!(
            OptimizationPreset::HybridProgressive.quant_table_source(),
            QuantTableSource::Jpegli
        );
        assert_eq!(
            OptimizationPreset::HybridMaxCompression.quant_table_source(),
            QuantTableSource::Jpegli
        );
    }

    #[test]
    fn test_quant_table_source_default() {
        assert_eq!(QuantTableSource::default(), QuantTableSource::Jpegli);
    }

    #[test]
    fn test_optimization_preset_display() {
        assert_eq!(
            OptimizationPreset::JpegliBaseline.to_string(),
            "jpegli-baseline"
        );
    }
    #[test]
    fn test_optimization_preset_display_trellis() {
        assert_eq!(
            OptimizationPreset::HybridMaxCompression.to_string(),
            "hybrid-max"
        );
    }

    #[test]
    fn test_pixel_format_to_layout_conversion() {
        // 8-bit → sRGB gamma
        assert_eq!(PixelLayout::from(PixelFormat::Rgb), PixelLayout::Rgb8Srgb);
        assert_eq!(PixelLayout::from(PixelFormat::Rgba), PixelLayout::Rgba8Srgb);
        assert_eq!(PixelLayout::from(PixelFormat::Bgr), PixelLayout::Bgr8Srgb);
        assert_eq!(PixelLayout::from(PixelFormat::Bgra), PixelLayout::Bgra8Srgb);
        assert_eq!(PixelLayout::from(PixelFormat::Bgrx), PixelLayout::Bgrx8Srgb);
        assert_eq!(PixelLayout::from(PixelFormat::Gray), PixelLayout::Gray8Srgb);

        // 16-bit → linear light
        assert_eq!(
            PixelLayout::from(PixelFormat::Rgb16),
            PixelLayout::Rgb16Linear
        );
        assert_eq!(
            PixelLayout::from(PixelFormat::Rgba16),
            PixelLayout::Rgba16Linear
        );
        assert_eq!(
            PixelLayout::from(PixelFormat::Gray16),
            PixelLayout::Gray16Linear
        );

        // f32 → linear light
        assert_eq!(
            PixelLayout::from(PixelFormat::RgbF32),
            PixelLayout::RgbF32Linear
        );
        assert_eq!(
            PixelLayout::from(PixelFormat::RgbaF32),
            PixelLayout::RgbaF32Linear
        );
        assert_eq!(
            PixelLayout::from(PixelFormat::GrayF32),
            PixelLayout::GrayF32Linear
        );
    }

    #[test]
    fn test_pixel_layout_to_format_conversion() {
        // sRGB 8-bit → 8-bit format
        assert_eq!(PixelFormat::from(PixelLayout::Rgb8Srgb), PixelFormat::Rgb);
        assert_eq!(PixelFormat::from(PixelLayout::Bgr8Srgb), PixelFormat::Bgr);
        assert_eq!(PixelFormat::from(PixelLayout::Rgba8Srgb), PixelFormat::Rgba);
        assert_eq!(PixelFormat::from(PixelLayout::Rgbx8Srgb), PixelFormat::Rgba); // RGBX maps to RGBA

        // Linear 16-bit → 16-bit format
        assert_eq!(
            PixelFormat::from(PixelLayout::Rgb16Linear),
            PixelFormat::Rgb16
        );
        assert_eq!(
            PixelFormat::from(PixelLayout::Rgba16Linear),
            PixelFormat::Rgba16
        );

        // Linear f32 → f32 format
        assert_eq!(
            PixelFormat::from(PixelLayout::RgbF32Linear),
            PixelFormat::RgbF32
        );
        assert_eq!(
            PixelFormat::from(PixelLayout::RgbaF32Linear),
            PixelFormat::RgbaF32
        );
    }

    #[test]
    fn test_roundtrip_conversions() {
        // Test that common formats roundtrip correctly
        let formats = vec![
            PixelFormat::Rgb,
            PixelFormat::Bgr,
            PixelFormat::Gray,
            PixelFormat::Rgb16,
            PixelFormat::RgbF32,
        ];

        for pf in formats {
            let pl: PixelLayout = pf.into();
            let pf2: PixelFormat = pl.into();
            assert_eq!(
                pf, pf2,
                "Roundtrip failed for {:?} -> {:?} -> {:?}",
                pf, pl, pf2
            );
        }
    }
}

// =============================================================================
// Optimization Presets
// =============================================================================

/// Preset optimization modes for the encoder.
///
/// Each preset configures [`ProgressiveScanMode`], [`QuantTableConfig`], trellis,
/// deringing, and Huffman strategy to match a specific encoder profile.
/// The three lineages are:
///
/// - **Jpegli** — matches C jpegli/cjpegli behavior (AQ-driven, no trellis)
/// - **Mozjpeg** — matches C mozjpeg behavior (trellis-driven, no AQ)
/// - **Hybrid** — combines jpegli AQ + mozjpeg trellis (zenjpeg-only)
///
/// # Preset Matrix
///
/// | Preset | AQ | Trellis | Deringing | Tables | Scan |
/// |--------|-----|---------|-----------|--------|------|
/// | JpegliBaseline | ✓ | — | ✓ | Jpegli 3T | Baseline |
/// | JpegliProgressive | ✓ | — | ✓ | Jpegli 3T | Progressive |
/// | MozjpegBaseline | — | Thorough | — | Robidoux 2T | Baseline |
/// | MozjpegProgressive | — | Thorough | — | Robidoux 2T | ProgMozjpeg |
/// | MozjpegMaxCompression | — | Thorough | ✓ | Robidoux 2T | ProgSearch |
/// | HybridBaseline | ✓ | Adaptive | ✓ | Jpegli 3T | Baseline |
/// | HybridProgressive | ✓ | Adaptive | ✓ | Jpegli 3T | Progressive |
/// | HybridMaxCompression | ✓ | Thorough | ✓ | Jpegli 3T | ProgSearch |
///
/// # Exploration Gaps
///
/// Interesting combinations NOT covered by the 8 presets (build manually):
///
/// - **Robidoux + AQ**: `MozjpegRobidoux` tables with jpegli AQ. Tests whether
///   AQ can compensate for Robidoux's less precise frequency-dependent scaling.
/// - **Jpegli + trellis without AQ**: Inverse hybrid — tests trellis value
///   with jpegli's perceptual tables when AQ is disabled.
/// - **Universal deringing**: All presets with deringing=true. C mozjpeg only
///   enables it for JCP_MAX_COMPRESSION, but it may help mozjpeg baseline/prog too.
/// - **JpegliMaxCompression**: AQ + scan search + no trellis. Tests whether
///   scan search alone (without trellis overhead) is worthwhile for jpegli.
///
/// Use [`OptimizationPreset::all()`] to iterate all variants for search.
///
/// Individual settings can still be overridden after applying a preset
/// via [`EncoderConfig::optimization()`](super::encoder_config::EncoderConfig::optimization).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OptimizationPreset {
    // === Jpegli lineage (matches C cjpegli) ===
    /// Baseline JPEG, optimized Huffman, jpegli AQ. No trellis.
    /// Matches `cjpegli --jpeg_type baseline`.
    JpegliBaseline,

    /// Progressive JPEG with jpegli scan script (freq split at 2/3,
    /// SA for all). Optimized Huffman. Jpegli AQ. No trellis.
    /// Matches default `cjpegli` output. This is the encoder default.
    JpegliProgressive,

    // === Mozjpeg lineage (matches C cjpeg/mozjpeg) ===
    /// Baseline JPEG, optimized Huffman, trellis (full search). No jpegli AQ.
    /// 2 quant tables (shared chroma), Robidoux tables. No deringing.
    /// Matches C mozjpeg default profile with baseline output.
    MozjpegBaseline,

    /// Progressive with mozjpeg scan script (freq split at 8/9, no chroma SA).
    /// Trellis (full search). No AQ. No deringing. 2 quant tables, Robidoux.
    /// Matches C mozjpeg default profile with progressive output.
    MozjpegProgressive,

    /// Mozjpeg progressive + scan search (64 candidates). Trellis (full search).
    /// No AQ. Deringing enabled. 2 quant tables, Robidoux.
    /// Matches C mozjpeg `JCP_MAX_COMPRESSION` profile.
    MozjpegMaxCompression,

    // === Hybrid lineage (zenjpeg-unique) ===
    /// Baseline JPEG with jpegli AQ + trellis (Adaptive speed) + deringing.
    /// 3 quant tables (separate chroma), jpegli perceptual tables.
    HybridBaseline,

    /// Jpegli AQ + trellis (Adaptive) + deringing + jpegli scan script.
    /// 3 quant tables. Typically the best quality/size tradeoff.
    HybridProgressive,

    /// AQ + trellis (full search) + deringing + scan search (64 candidates).
    /// 3 quant tables. Slowest, smallest files.
    HybridMaxCompression,
}

/// Effort level for convenience constructors.
///
/// Maps to the corresponding [`OptimizationPreset`] for the chosen color mode.
/// Use this with [`EncoderConfig::ycbcr_effort()`](super::encoder_config::EncoderConfig::ycbcr_effort),
/// [`EncoderConfig::xyb_effort()`](super::encoder_config::EncoderConfig::xyb_effort),
/// or [`EncoderConfig::grayscale_effort()`](super::encoder_config::EncoderConfig::grayscale_effort).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Effort {
    /// Fast encoding: jpegli baseline, no trellis.
    /// Maps to [`OptimizationPreset::JpegliBaseline`].
    Fast,

    /// Balanced quality/speed: AQ + adaptive trellis + progressive.
    /// Maps to [`OptimizationPreset::HybridProgressive`].
    Balanced,

    /// Maximum compression: thorough trellis + scan optimization.
    /// Maps to [`OptimizationPreset::HybridMaxCompression`].
    Max,
}

impl Effort {
    /// Convert to the corresponding [`OptimizationPreset`].
    #[must_use]
    pub const fn to_preset(self) -> OptimizationPreset {
        match self {
            Self::Fast => OptimizationPreset::JpegliBaseline,
            Self::Balanced => OptimizationPreset::HybridProgressive,
            Self::Max => OptimizationPreset::HybridMaxCompression,
        }
    }
}

/// All preset variants in a fixed array for iteration.
const ALL_PRESETS: [OptimizationPreset; 8] = [
    OptimizationPreset::JpegliBaseline,
    OptimizationPreset::JpegliProgressive,
    OptimizationPreset::MozjpegBaseline,
    OptimizationPreset::MozjpegProgressive,
    OptimizationPreset::MozjpegMaxCompression,
    OptimizationPreset::HybridBaseline,
    OptimizationPreset::HybridProgressive,
    OptimizationPreset::HybridMaxCompression,
];

impl OptimizationPreset {
    /// Returns an iterator over all preset variants.
    /// Useful for searching the optimization space across lineages.
    pub fn all() -> impl Iterator<Item = Self> {
        ALL_PRESETS.iter().copied()
    }

    /// Returns true if this preset uses progressive encoding.
    #[must_use]
    pub const fn is_progressive(self) -> bool {
        match self {
            Self::JpegliBaseline => false,
            Self::JpegliProgressive => true,
            Self::MozjpegBaseline => false,
            Self::MozjpegProgressive => true,
            Self::MozjpegMaxCompression => true,
            Self::HybridBaseline => false,
            Self::HybridProgressive => true,
            Self::HybridMaxCompression => true,
        }
    }

    /// Returns true if this preset enables trellis quantization.
    #[must_use]
    pub const fn uses_trellis(self) -> bool {
        match self {
            Self::JpegliBaseline | Self::JpegliProgressive => false,
            Self::MozjpegBaseline
            | Self::MozjpegProgressive
            | Self::MozjpegMaxCompression
            | Self::HybridBaseline
            | Self::HybridProgressive
            | Self::HybridMaxCompression => true,
        }
    }

    /// Returns true if this preset uses jpegli AQ (adaptive quantization).
    #[must_use]
    pub const fn uses_aq(self) -> bool {
        match self {
            Self::JpegliBaseline | Self::JpegliProgressive => true,
            Self::MozjpegBaseline | Self::MozjpegProgressive | Self::MozjpegMaxCompression => false,
            Self::HybridBaseline | Self::HybridProgressive | Self::HybridMaxCompression => true,
        }
    }

    /// Returns the quantization table source for this preset.
    #[must_use]
    pub const fn quant_table_source(self) -> QuantTableSource {
        match self {
            Self::MozjpegBaseline | Self::MozjpegProgressive | Self::MozjpegMaxCompression => {
                QuantTableSource::MozjpegDefault
            }
            _ => QuantTableSource::Jpegli,
        }
    }

    /// Returns the scan strategy for this preset.
    #[must_use]
    pub const fn scan_strategy(self) -> ScanStrategy {
        match self {
            Self::JpegliBaseline => ScanStrategy::Default,
            Self::JpegliProgressive => ScanStrategy::Default,
            Self::MozjpegBaseline | Self::HybridBaseline => ScanStrategy::Default,
            Self::HybridProgressive => ScanStrategy::Default,
            Self::MozjpegProgressive => ScanStrategy::Mozjpeg,
            Self::MozjpegMaxCompression | Self::HybridMaxCompression => ScanStrategy::Search,
        }
    }

    /// Returns the short name for display/logging.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::JpegliBaseline => "jpegli-baseline",
            Self::JpegliProgressive => "jpegli-progressive",
            Self::MozjpegBaseline => "mozjpeg-baseline",
            Self::MozjpegProgressive => "mozjpeg-progressive",
            Self::MozjpegMaxCompression => "mozjpeg-max",
            Self::HybridBaseline => "hybrid-baseline",
            Self::HybridProgressive => "hybrid-progressive",
            Self::HybridMaxCompression => "hybrid-max",
        }
    }
}

impl core::fmt::Display for OptimizationPreset {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
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
/// Huffman table selection strategy.
///
/// - `Optimize`: Two-pass encoding collects symbol frequencies, then builds optimal tables.
///   Produces the smallest files. Required for progressive mode.
/// - `Fixed`: Single-pass encoding with general-purpose trained Huffman tables.
///   Fastest, with ~2.5% overhead vs per-image optimal.
/// - `FixedAnnexK`: Single-pass with JPEG Annex K standard tables.
///   Maximum compatibility, ~5-12% larger than optimal.
/// - `Custom`: Single-pass encoding with caller-provided tables.
///   Use `HuffmanTableSet::annex_k()` for the original JPEG standard tables,
///   or provide your own pre-computed tables.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum HuffmanStrategy {
    /// Two-pass: collect frequencies, build optimal tables.
    /// Smallest files, but cannot stream output (must buffer all blocks).
    #[default]
    Optimize,
    /// Single-pass: use general-purpose trained tables (~2.5% overhead).
    /// Enables streaming output for sequential mode.
    Fixed,
    /// Single-pass: use JPEG Annex K standard tables (~5-12% overhead).
    /// Maximum decoder compatibility. Enables streaming output.
    FixedAnnexK,
    /// Single-pass: use caller-provided tables.
    /// Use `HuffmanTableSet::annex_k()` for Annex K, or provide custom tables.
    Custom(Box<crate::huffman::optimize::HuffmanTableSet>),
}

impl From<bool> for HuffmanStrategy {
    /// `true` → `Optimize`, `false` → `Fixed`
    fn from(optimize: bool) -> Self {
        if optimize {
            HuffmanStrategy::Optimize
        } else {
            HuffmanStrategy::Fixed
        }
    }
}

impl From<crate::huffman::optimize::HuffmanTableSet> for HuffmanStrategy {
    fn from(tables: crate::huffman::optimize::HuffmanTableSet) -> Self {
        HuffmanStrategy::Custom(Box::new(tables))
    }
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

/// Controls tiny-file optimizations for small images.
///
/// For small images (thumbnails, icons, placeholders) JPEG header overhead can
/// exceed the compressed pixel data itself. This mode collapses a few
/// header-side wins into one switch:
///
/// - Shared Huffman tables: emits two DHT tables (one DC, one AC) derived from
///   combined luma+chroma symbol counts, with `Td=0, Ta=0` for every component
///   in SOS. Saves roughly 200 bytes versus the four-table default.
/// - Header marker pruning: already-conservative by default (zenjpeg does not
///   emit JFIF APP0; Adobe APP14 only for XYB; EXIF only on request). Future
///   extensions to this mode may prune further.
///
/// All optimizations remain spec-legal baseline JPEG and are accepted by
/// libjpeg-turbo, Chromium/Blink, and Apple's ImageIO.
///
/// # Variants
///
/// - `Auto` (default): enable tiny-file optimizations when the image size
///   heuristic (see [`should_activate_tiny_file_mode_for_subsampling`])
///   indicates a likely win.
/// - `Off`: always use the standard four-table Huffman layout.
/// - `Force`: always apply tiny-file optimizations regardless of image size.
///
/// `Auto` currently applies only to baseline-sequential YCbCr/grayscale
/// output. Progressive mode and XYB paths are unaffected — XYB already
/// uses a shared-pair Huffman layout, and progressive's per-scan AC slot
/// cycling would need separate work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum TinyFileMode {
    /// Apply tiny-file optimizations when the built-in heuristic matches the
    /// image size. This is the default.
    #[default]
    Auto,
    /// Never apply tiny-file optimizations. Emits standard four-table Huffman
    /// layout with per-channel DC/AC selectors.
    Off,
    /// Always apply tiny-file optimizations regardless of image size.
    Force,
}

/// Heuristic for [`TinyFileMode::Auto`]: returns `true` when tiny-file
/// optimizations are likely to reduce file size for the given image.
///
/// Thresholds are calibrated empirically against CID22-train at q=75 and
/// q=90. See the `tiny_file_mode` integration tests and the benchmark in
/// `examples/tiny_file_crossover.rs` for the underlying data.
///
/// `is_color` distinguishes color (RGB/YCbCr) from grayscale. For the
/// per-subsampling threshold used by the encoder internally, see
/// [`should_activate_tiny_file_mode_for_subsampling`].
#[inline]
#[must_use]
pub fn should_activate_tiny_file_mode(width: u32, height: u32, is_color: bool) -> bool {
    // Conservative subsampling-agnostic variant. The encoder internally uses
    // the subsampling-aware variant (see `should_activate_tiny_file_mode_for_subsampling`).
    // This variant assumes 4:4:4 for color (worst case), which matches the
    // tightest threshold. Grayscale always benefits.
    should_activate_tiny_file_mode_for_subsampling(
        width,
        height,
        is_color,
        crate::types::Subsampling::S444,
    )
}

/// Subsampling-aware variant of [`should_activate_tiny_file_mode`].
///
/// - **Grayscale**: always activates. The shared-Huffman path drops two
///   unused chroma table headers (~208 B fixed saving), never costs bits.
/// - **4:2:0 / 4:4:0 / 4:2:2 color**: activates below ~128×128. Chroma is
///   subsampled, so the merged DC/AC tables retain most of their luma
///   efficiency; the header saving beats the coding overhead up to
///   ~16-30k pixels.
/// - **4:4:4 color**: activates below ~64×64 only. Chroma planes are
///   full-resolution here, so merged tables lose efficiency much faster.
///
/// Thresholds come from CID22-train measurements at q=75 and q=90 (see
/// `examples/tiny_file_crossover.rs`). At the crossover points the average
/// delta is within ±0.5% of the four-table baseline in either direction.
#[inline]
#[must_use]
pub fn should_activate_tiny_file_mode_for_subsampling(
    width: u32,
    height: u32,
    is_color: bool,
    subsampling: crate::types::Subsampling,
) -> bool {
    let pixels = (width as u64) * (height as u64);
    if !is_color {
        // Grayscale: always a win (fixed ~208 B saving).
        return true;
    }
    use crate::types::Subsampling::*;
    match subsampling {
        // 4:4:4: full-resolution chroma. Shared-Huffman starts losing above
        // 64×64 (+0.16% at exactly 64² per CID22 q=75, -1.5% at 96²). Keep
        // the threshold at 64×64 inclusive.
        S444 => pixels <= 64 * 64,
        // Subsampled chroma (4:2:0 most common, also 4:2:2 and 4:4:0):
        // shared tables retain efficiency up to ~128×128 (+0.69% at 128² at
        // q=75, neutral at 192²). Threshold 128×128 inclusive.
        S420 | S422 | S440 => pixels <= 128 * 128,
    }
}

/// Source of quantization tables for encoding.
///
/// Controls which base quantization tables and scaling formula are used:
///
/// - **Jpegli** (default) — jpegli perceptual defaults with distance-based,
///   frequency-dependent scaling and adaptive zero-bias.
/// - **MozjpegDefault** — Nicolas Robidoux's psychovisual tables with
///   libjpeg's quality-scaling formula and neutral zero-bias (0.5 AC offset).
///
/// For other mozjpeg presets (MSSIM, Klein, etc.), use the `mozjpeg-tables`
/// feature and supply custom tables via
/// `EncoderConfig::quant_tables()`.
///
/// **Prefer [`QuantTableConfig`]** for new code — it bundles table source,
/// chroma table count, and custom tables into one type-safe enum that
/// prevents invalid combinations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum QuantTableSource {
    /// Jpegli perceptual defaults (distance-based, frequency-dependent scaling).
    #[default]
    Jpegli,
    /// Mozjpeg Robidoux tables (psychovisual, quality-scaled).
    MozjpegDefault,
}

/// Unified quantization table configuration.
///
/// Bundles the table family with the knobs that are live for that family —
/// each knob exists only on the variant where it has an effect:
///
/// | Variant | Tables | Live knobs | `Quality` drives tables? |
/// |---------|--------|------------|--------------------------|
/// | `Jpegli` | 3 (Y, Cb, Cr) | `chroma_distance_scales` | yes (distance scaling) |
/// | `JpegliSharedChroma` | 2 (Y, shared) | `chroma_distance_scales` | yes |
/// | `MozjpegRobidoux` | 2 (Y, shared) | `chroma_quality` | yes (libjpeg quality scaling) |
/// | `Custom` | user-defined | the tables themselves | only if `ScalingParams::Scaled` |
/// | `PiecewiseV4` | 3 (Y, Cb, Cr) | — | yes (SA anchor interpolation) |
/// | `GlassaLowBpp` | 2 (shared) | — | yes (anchor interpolation, q clamped 3–25) |
///
/// XYB color mode accepts only `Jpegli` and `Custom` — the 2-table
/// shared-chroma layouts are YCbCr concepts with no defined meaning for
/// X,Y,B channels and are rejected by `EncoderConfig::validate()`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum QuantTableConfig {
    /// Jpegli perceptual defaults with 3 separate quantization tables
    /// (Y, Cb, Cr). Matches C++ jpegli's `jpegli_set_distance()` behavior.
    Jpegli {
        /// Per-channel multipliers on the butteraugli distance for the two
        /// chroma-like channels, in ascending component order:
        /// `[Cb, Cr]` in YCbCr mode, `[X, B]` in XYB mode (Y, the
        /// luma-like channel, always keeps the base distance).
        /// `[1.0, 1.0]` = same treatment as luma. Each clamped to
        /// `[0.1, 5.0]` at resolution.
        ///
        /// `> 1.0` encodes that channel more aggressively (flat-chroma
        /// images); `< 1.0` more carefully (sharp-chroma images). The two
        /// entries are independent — optimizers can move Cb and Cr (or X
        /// and B) separately. When the scales are non-neutral, zero-bias
        /// follows each channel's own effective distance.
        chroma_distance_scales: [f32; 2],
    },

    /// Jpegli perceptual defaults with 2 quantization tables (Y, shared
    /// chroma using the Cr base matrix). Matches C++ jpegli's
    /// `jpeg_set_quality()` behavior. YCbCr only.
    JpegliSharedChroma {
        /// Same semantics as [`Jpegli`](Self::Jpegli)'s field. The shared
        /// chroma table is generated from the Cr base matrix using
        /// `scales[1]`; `scales[0]` still shapes the Cb distance input
        /// (relevant only to zero-bias when divergent).
        chroma_distance_scales: [f32; 2],
    },

    /// Mozjpeg Robidoux psychovisual tables scaled by libjpeg's quality
    /// formula. Always 2 tables (shared chroma), matching C mozjpeg.
    /// YCbCr only.
    MozjpegRobidoux {
        /// Optional independent chroma quality on the mozjpeg 1–100
        /// scale (clamped at resolution). `None` scales chroma with the
        /// luma quality — bit-identical to the historical behaviour.
        chroma_quality: Option<u8>,
    },

    /// User-provided custom encoding tables.
    ///
    /// The tables own the chroma policy via their per-component base
    /// matrices. `Quality` drives them only when
    /// `scaling == ScalingParams::Scaled`; `Exact` tables are used as-is
    /// (quality then affects only gates like `auto_optimize`, not bytes).
    Custom(Box<super::tuning::EncodingTables>),

    /// SA-piecewise v4 tables: 20 quality-anchored sets with linear
    /// interpolation, trained by coefficient's piecewise SA optimizer on
    /// CID22-512 (209 photos) with GPU butteraugli, jpegli encoder,
    /// 4:2:0 chroma. Beats jpegli's parametric tables on 99/100 quality
    /// levels — mean pareto +6.602 (training), +6.09 (41-image holdout).
    ///
    /// The anchor quality derives from the config's one `Quality` knob
    /// (internal jpegli q). YCbCr only (anchors are YCbCr-space; rejected
    /// for XYB in `validate()`). Trained on photographic content —
    /// screenshots/illustrations may regress; retrain via
    /// `coefficient`'s piecewise optimizer for other corpora.
    ///
    /// Open avenues (deliberately not closed by this wiring): per-anchor
    /// zero-bias SA (the v4 run held zero-bias at neutral), per-content
    /// anchor sets, 4:4:4-trained anchors, trellis-interaction sweeps
    /// before `adaptive()` adopts it per-cell.
    PiecewiseV4,

    /// Glassa low-BPP optimized tables for extreme compression.
    ///
    /// SA-optimized anchors achieving +20 to +33 pareto gains at
    /// 0.15–0.50 BPP (thumbnails, LQIP, progressive placeholders).
    /// The anchor quality derives from the config's `Quality`: internal
    /// jpegli q, rounded and clamped to the trained 3–25 range. At Q30+
    /// prefer [`Jpegli`](Self::Jpegli) or mozjpeg defaults.
    GlassaLowBpp,
}

impl Default for QuantTableConfig {
    /// Jpegli 3-table with neutral chroma scale — matches
    /// `jpegli_set_distance()`.
    fn default() -> Self {
        Self::Jpegli {
            chroma_distance_scales: [1.0, 1.0],
        }
    }
}

impl QuantTableConfig {
    /// Returns the internal `QuantTableSource` for this configuration.
    #[must_use]
    pub const fn quant_source(&self) -> QuantTableSource {
        match self {
            Self::Jpegli { .. }
            | Self::JpegliSharedChroma { .. }
            | Self::Custom(_)
            | Self::PiecewiseV4
            | Self::GlassaLowBpp => QuantTableSource::Jpegli,
            Self::MozjpegRobidoux { .. } => QuantTableSource::MozjpegDefault,
        }
    }

    /// Returns whether Cb and Cr use separate quantization tables.
    #[must_use]
    pub const fn separate_chroma_tables(&self) -> bool {
        match self {
            Self::Jpegli { .. } | Self::PiecewiseV4 => true,
            Self::JpegliSharedChroma { .. } | Self::MozjpegRobidoux { .. } | Self::GlassaLowBpp => {
                false
            }
            // Custom tables define their own layout; default to separate.
            Self::Custom(_) => true,
        }
    }

    /// Jpegli family with one uniform chroma scale applied to both
    /// chroma-like channels — the common case.
    #[must_use]
    pub const fn jpegli_chroma_scale(scale: f32) -> Self {
        Self::Jpegli {
            chroma_distance_scales: [scale, scale],
        }
    }

    /// The per-channel chroma distance scales, when this family has them
    /// (jpegli families only).
    #[must_use]
    pub fn chroma_distance_scales(&self) -> Option<[f32; 2]> {
        match self {
            Self::Jpegli {
                chroma_distance_scales,
            }
            | Self::JpegliSharedChroma {
                chroma_distance_scales,
            } => Some(*chroma_distance_scales),
            _ => None,
        }
    }

    /// The independent chroma quality, when this family has one
    /// (`MozjpegRobidoux` only).
    #[must_use]
    pub fn chroma_quality(&self) -> Option<u8> {
        match self {
            Self::MozjpegRobidoux { chroma_quality } => *chroma_quality,
            _ => None,
        }
    }
}

/// JPEG scan mode — baseline vs progressive, with script strategy.
///
/// Bundles the progressive flag and scan script strategy into a single type,
/// preventing invalid combinations such as:
///
/// - Baseline + Search scan strategy (search only applies to progressive)
/// - Baseline + Mozjpeg scan script (mozjpeg script is progressive-only)
///
/// All progressive modes automatically enable optimized Huffman tables.
///
/// # Variants
///
/// | Variant | Progressive | Script | Notes |
/// |---------|-------------|--------|-------|
/// | `Baseline` | No | N/A | Sequential JPEG (SOF0/SOF1) |
/// | `Progressive` | Yes | jpegli default | Freq split at 2/3, SA for all |
/// | `ProgressiveMozjpeg` | Yes | mozjpeg default | Freq split at 8/9, no chroma SA |
/// | `ProgressiveSearch` | Yes | search (64 candidates) | Best compression, ~2x slower |
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProgressiveScanMode {
    /// Baseline (sequential) JPEG.
    ///
    /// Single scan containing all components. No progressive refinement.
    /// Compatible with all JPEG decoders.
    #[default]
    Baseline,

    /// Progressive JPEG with jpegli's default scan script.
    ///
    /// - Frequency split at AC coefficients 2/3
    /// - Successive approximation for all components (Al=2 → 1 → 0)
    /// - Separate DC scans per component
    Progressive,

    /// Progressive JPEG with mozjpeg's default scan script.
    ///
    /// - Frequency split at AC coefficients 8/9
    /// - No successive approximation for chroma
    /// - Separate DC scans per component
    ///
    /// Closest parity with C mozjpeg's default progressive output.
    ProgressiveMozjpeg,

    /// Smallest-output selection across the entropy stage.
    ///
    /// Quantized coefficients are computed once and the progressive
    /// (jpegli script) candidate is serialized. If its output is at or
    /// below 16 KiB — the byte domain where every observed sequential
    /// win lives — the sequential candidates (+ tiny-file shared-table
    /// variant when eligible and [`TinyFileMode`] is not `Off`) are
    /// also serialized and the exact `min(bytes)` is emitted: a
    /// **lossless rate decision** (identical pixels). Small progressive
    /// output implies little entropy content, so the extra passes are
    /// bounded cheap regardless of image dimensions.
    ///
    /// Above the byte gate the progressive stream is emitted directly
    /// (one serialization): sweeps found zero sequential wins there.
    /// Empirical bound with provenance, documented at the constant.
    ///
    /// Restart markers are suppressed in every candidate (they are
    /// strictly additive bytes, and the progressive alternative cannot
    /// restart-parallel-decode either, so nothing is traded away);
    /// `force_restart_markers(true)` restores them — sequential
    /// candidates then honor `restart_mcu_rows` like a plain Baseline
    /// encode.
    Smallest,

    /// [`Smallest`](Self::Smallest) with the scan-script search as the
    /// progressive candidate: the 64-candidate search picks the best
    /// progressive structure (it always includes the default jpegli
    /// script, so its winner is ≤ `Smallest`'s progressive candidate),
    /// and the sequential (+ tiny) trials run under the same byte gate.
    /// Strictly ≤ both [`Smallest`](Self::Smallest) and
    /// [`ProgressiveSearch`](Self::ProgressiveSearch) at identical
    /// pixels, for the search's extra trial-encode cost. The search is
    /// skipped for XYB (existing emission rule); the sequential trials
    /// still apply there.
    SmallestSearch,

    /// Progressive JPEG with scan search optimization.
    ///
    /// Tests 64 candidate scan configurations (frequency splits, SA levels,
    /// DC interleaving) and picks the smallest. Typically saves 1-3% vs
    /// fixed scripts, at the cost of ~2x encode time.
    ProgressiveSearch,
}

impl ProgressiveScanMode {
    /// Returns true if this mode uses progressive encoding.
    #[must_use]
    pub const fn is_progressive(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    /// Returns the internal `ScanStrategy` for this mode.
    ///
    /// Baseline returns `Default` (ignored for non-progressive encoding).
    #[must_use]
    pub const fn scan_strategy(self) -> ScanStrategy {
        match self {
            Self::Baseline | Self::Progressive | Self::Smallest => ScanStrategy::Default,
            Self::ProgressiveMozjpeg => ScanStrategy::Mozjpeg,
            Self::ProgressiveSearch | Self::SmallestSearch => ScanStrategy::Search,
        }
    }
}

impl From<bool> for ProgressiveScanMode {
    /// `true` → `Progressive`, `false` → `Baseline`
    fn from(progressive: bool) -> Self {
        if progressive {
            ProgressiveScanMode::Progressive
        } else {
            ProgressiveScanMode::Baseline
        }
    }
}

/// Convert from legacy PixelFormat to explicit PixelLayout.
///
/// Assumes standard color space conventions:
/// - 8-bit formats → sRGB gamma encoding
/// - 16-bit formats → linear light
/// - f32 formats → linear light
impl From<crate::types::PixelFormat> for PixelLayout {
    fn from(format: crate::types::PixelFormat) -> Self {
        use crate::types::PixelFormat;
        match format {
            // 8-bit → sRGB gamma (standard assumption)
            PixelFormat::Gray => Self::Gray8Srgb,
            PixelFormat::Rgb => Self::Rgb8Srgb,
            PixelFormat::Rgba => Self::Rgba8Srgb,
            PixelFormat::Bgr => Self::Bgr8Srgb,
            PixelFormat::Bgra => Self::Bgra8Srgb,
            PixelFormat::Bgrx => Self::Bgrx8Srgb,

            // 16-bit → linear light
            PixelFormat::Gray16 => Self::Gray16Linear,
            PixelFormat::Rgb16 => Self::Rgb16Linear,
            PixelFormat::Rgba16 => Self::Rgba16Linear,

            // f32 → linear light
            PixelFormat::GrayF32 => Self::GrayF32Linear,
            PixelFormat::RgbF32 => Self::RgbF32Linear,
            PixelFormat::RgbaF32 => Self::RgbaF32Linear,

            // CMYK → treat as RGB (best effort, will be converted)
            PixelFormat::Cmyk => Self::Rgba8Srgb,
        }
    }
}
