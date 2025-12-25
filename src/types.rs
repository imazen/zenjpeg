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
}

impl Default for Quality {
    fn default() -> Self {
        Quality::Standard(85)
    }
}

impl Quality {
    /// Get the raw quality value (1-100 scale)
    #[must_use]
    pub fn value(&self) -> f32 {
        match *self {
            Quality::Standard(q) | Quality::Low(q) | Quality::High(q) => q as f32,
            Quality::Perceptual(target) => target,
            Quality::TargetSize(_) => 50.0, // placeholder
        }
    }

    /// Whether this quality setting prefers trellis quantization
    #[must_use]
    pub fn prefers_trellis(&self) -> bool {
        match *self {
            Quality::Low(_) => true,
            Quality::Standard(q) => q < 70,
            Quality::High(_) => false,
            Quality::Perceptual(target) => target < 70.0,
            Quality::TargetSize(_) => true, // trellis is better for size optimization
        }
    }

    /// Whether this quality setting prefers adaptive quantization
    #[must_use]
    pub fn prefers_adaptive_quant(&self) -> bool {
        match *self {
            Quality::High(_) => true,
            Quality::Standard(q) => q >= 70,
            Quality::Low(_) => false,
            Quality::Perceptual(target) => target >= 70.0,
            Quality::TargetSize(_) => false,
        }
    }
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

impl Default for EncodingStrategy {
    fn default() -> Self {
        EncodingStrategy::Auto
    }
}

// TrellisConfig is defined in trellis.rs
// AdaptiveQuantConfig is defined in adaptive_quant.rs
pub use crate::trellis::TrellisConfig;
pub use crate::adaptive_quant::AdaptiveQuantConfig;
