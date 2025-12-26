//! Core types for zenjpeg

/// Color space for input pixels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// RGB color space (most common input)
    #[default]
    Rgb,
    /// RGBA with alpha channel (alpha is ignored for JPEG)
    Rgba,
    /// Grayscale
    Gray,
    /// YCbCr (native JPEG color space)
    YCbCr,
}

/// Pixel format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelFormat {
    /// 8-bit RGB, 3 bytes per pixel
    #[default]
    Rgb8,
    /// 8-bit RGBA, 4 bytes per pixel
    Rgba8,
    /// 8-bit grayscale, 1 byte per pixel
    Gray8,
    /// 16-bit RGB, 6 bytes per pixel (will be converted to 8-bit)
    Rgb16,
}

impl PixelFormat {
    /// Bytes per pixel for this format
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 => 4,
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb16 => 6,
        }
    }

    /// Number of color components
    #[must_use]
    pub const fn components(self) -> usize {
        match self {
            PixelFormat::Rgb8 | PixelFormat::Rgb16 => 3,
            PixelFormat::Rgba8 => 4,
            PixelFormat::Gray8 => 1,
        }
    }
}

/// Chroma subsampling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Subsampling {
    /// No subsampling (4:4:4) - highest quality
    #[default]
    S444,
    /// Horizontal subsampling only (4:2:2)
    S422,
    /// Both horizontal and vertical (4:2:0) - smallest files
    S420,
}

impl Subsampling {
    /// Horizontal sampling factor for chroma components
    #[must_use]
    pub const fn h_factor(self) -> u8 {
        match self {
            Subsampling::S444 => 1,
            Subsampling::S422 | Subsampling::S420 => 2,
        }
    }

    /// Vertical sampling factor for chroma components
    #[must_use]
    pub const fn v_factor(self) -> u8 {
        match self {
            Subsampling::S444 | Subsampling::S422 => 1,
            Subsampling::S420 => 2,
        }
    }
}

/// Quality specification for encoding
///
/// zenjpeg supports multiple quality modes to optimize for different use cases.
#[derive(Debug, Clone, Copy)]
pub enum Quality {
    /// Standard JPEG quality (1-100)
    /// Uses mozjpeg-style encoding for Q < 70, jpegli-style for Q >= 70
    Standard(u8),

    /// Low quality mode optimized for small file sizes (Q 1-50)
    /// Uses trellis quantization from mozjpeg
    Low(u8),

    /// High quality mode optimized for visual fidelity (Q 70-100)
    /// Uses adaptive quantization from jpegli
    High(u8),

    /// Target a specific perceptual quality (SSIMULACRA2 score 0-100)
    /// Automatically selects encoding strategy to achieve target
    Perceptual(f32),

    /// Target a specific file size in bytes
    /// Will binary search for optimal quality
    TargetSize(usize),

    /// Linear perceptual quality (0-100)
    ///
    /// Unlike Standard quality which has non-linear perceptual impact,
    /// Linear quality maps to a linearized DSSIM curve:
    /// - Linear(0) ≈ Standard(20) (low quality, DSSIM ~0.008)
    /// - Linear(50) produces half the DSSIM delta
    /// - Linear(100) ≈ Standard(95) (high quality, DSSIM ~0.0001)
    ///
    /// This makes quality steps feel perceptually uniform:
    /// the difference between Linear(40) and Linear(50) is the same
    /// perceptual delta as between Linear(80) and Linear(90).
    Linear(f32),
}

impl Default for Quality {
    fn default() -> Self {
        Quality::Standard(85)
    }
}

impl Quality {
    /// Get the raw quality value (1-100 scale)
    ///
    /// For Linear quality, this returns the mapped Standard quality value.
    #[must_use]
    pub fn value(&self) -> f32 {
        match *self {
            Quality::Standard(q) | Quality::Low(q) | Quality::High(q) => q as f32,
            Quality::Perceptual(target) => target,
            Quality::TargetSize(_) => 50.0, // placeholder
            Quality::Linear(linear) => linear_to_standard_quality(linear),
        }
    }

    /// Whether this quality setting prefers trellis quantization
    #[must_use]
    pub fn prefers_trellis(&self) -> bool {
        let q = self.value();
        match *self {
            Quality::Low(_) => true,
            Quality::High(_) => false,
            Quality::TargetSize(_) => true, // trellis is better for size optimization
            _ => q < 70.0,
        }
    }

    /// Whether this quality setting prefers adaptive quantization
    #[must_use]
    pub fn prefers_adaptive_quant(&self) -> bool {
        let q = self.value();
        match *self {
            Quality::High(_) => true,
            Quality::Low(_) => false,
            Quality::TargetSize(_) => false,
            _ => q >= 70.0,
        }
    }
}

/// Convert linear perceptual quality (0-100) to standard JPEG quality.
///
/// Linear quality produces uniform perceptual steps in DSSIM space.
/// Based on observed relationship: DSSIM ≈ 0.01 * exp(-Q/25)
///
/// | Linear | Standard | Approx DSSIM |
/// |--------|----------|--------------|
/// | 0      | 20       | 0.0077       |
/// | 25     | 40       | 0.0027       |
/// | 50     | 60       | 0.0022       |
/// | 75     | 80       | 0.0003       |
/// | 100    | 95       | 0.0001       |
///
/// The mapping uses logarithmic interpolation to linearize the
/// exponential relationship between quality and DSSIM.
fn linear_to_standard_quality(linear: f32) -> f32 {
    // Clamp input to valid range
    let linear = linear.clamp(0.0, 100.0);

    // Linear interpolation points based on observed DSSIM data
    // We want uniform DSSIM steps, which requires non-uniform Q steps
    //
    // Observed data:
    // Q20: DSSIM 0.0077  -> log(DSSIM) = -4.87
    // Q50: DSSIM 0.0025  -> log(DSSIM) = -6.00
    // Q80: DSSIM 0.00035 -> log(DSSIM) = -7.96
    // Q95: DSSIM 0.00007 -> log(DSSIM) = -9.57
    //
    // Linear mapping: interpolate Q based on linear position in log-DSSIM space

    // Reference points (Q, log_dssim)
    const Q_MIN: f32 = 20.0;
    const Q_MAX: f32 = 95.0;
    const LOG_DSSIM_AT_Q_MIN: f32 = -4.87; // ln(0.0077)
    const LOG_DSSIM_AT_Q_MAX: f32 = -9.57; // ln(0.00007)

    // Map linear (0-100) to log_dssim range
    let t = linear / 100.0;
    let target_log_dssim = LOG_DSSIM_AT_Q_MIN + t * (LOG_DSSIM_AT_Q_MAX - LOG_DSSIM_AT_Q_MIN);

    // Now we need Q such that log(dssim(Q)) ≈ target_log_dssim
    // Approximate relationship: log(dssim) ≈ -4.87 + (Q - 20) * slope
    // where slope ≈ (LOG_DSSIM_AT_Q_MAX - LOG_DSSIM_AT_Q_MIN) / (Q_MAX - Q_MIN)

    let log_dssim_range = LOG_DSSIM_AT_Q_MAX - LOG_DSSIM_AT_Q_MIN;
    let q_range = Q_MAX - Q_MIN;

    // Inverse mapping: Q = Q_MIN + (target_log_dssim - LOG_DSSIM_AT_Q_MIN) * q_range / log_dssim_range
    let q = Q_MIN + (target_log_dssim - LOG_DSSIM_AT_Q_MIN) * q_range / log_dssim_range;

    q.clamp(Q_MIN, Q_MAX)
}

/// Convert standard JPEG quality to linear perceptual quality (inverse mapping).
///
/// This is the inverse of `linear_to_standard_quality`.
#[allow(dead_code)]
fn standard_to_linear_quality(standard: f32) -> f32 {
    const Q_MIN: f32 = 20.0;
    const Q_MAX: f32 = 95.0;
    const LOG_DSSIM_AT_Q_MIN: f32 = -4.87;
    const LOG_DSSIM_AT_Q_MAX: f32 = -9.57;

    let standard = standard.clamp(Q_MIN, Q_MAX);

    let log_dssim_range = LOG_DSSIM_AT_Q_MAX - LOG_DSSIM_AT_Q_MIN;
    let q_range = Q_MAX - Q_MIN;

    // Forward mapping: log_dssim = LOG_DSSIM_AT_Q_MIN + (q - Q_MIN) * log_dssim_range / q_range
    let log_dssim = LOG_DSSIM_AT_Q_MIN + (standard - Q_MIN) * log_dssim_range / q_range;

    // Convert log_dssim to linear (0-100)
    let t = (log_dssim - LOG_DSSIM_AT_Q_MIN) / log_dssim_range;
    (t * 100.0).clamp(0.0, 100.0)
}

/// Encoding strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingStrategy {
    /// Use mozjpeg-style encoding (trellis, progressive)
    Mozjpeg,
    /// Use jpegli-style encoding (adaptive quant, perceptual)
    Jpegli,
    /// Hybrid approach (try both, pick best for target)
    Hybrid,
    /// Auto-select based on quality setting
    Auto,
}

/// Progressive scan script selection.
///
/// Different scan scripts optimize for different trade-offs:
/// - Minimal: Fastest encoding, reasonable progressive display
/// - Simple: Good progressive display with frequency band splitting
/// - Standard: Best compression via successive approximation
/// - Custom: User-defined scan script
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ScanScript {
    /// Minimal progressive: DC + full AC scans.
    /// Fast encoding, good for web preview.
    #[default]
    Minimal,

    /// Simple progressive: DC + AC bands (1-5, 6-63).
    /// Better progressive display with low/high frequency split.
    Simple,

    /// Standard progressive with successive approximation.
    /// Best compression, uses refinement scans.
    Standard,

    /// Custom scan script (advanced usage).
    Custom(Vec<crate::progressive::ScanInfo>),
}

impl Default for EncodingStrategy {
    fn default() -> Self {
        EncodingStrategy::Auto
    }
}

// Re-export configuration types
pub use crate::adaptive_quant::AdaptiveQuantConfig;
pub use crate::trellis::TrellisConfig;
