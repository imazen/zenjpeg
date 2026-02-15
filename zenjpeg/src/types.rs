//! Core types for jpegli.

#![allow(dead_code)]

use crate::foundation::consts::DCT_BLOCK_SIZE;

/// Color space representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
#[repr(u8)]
pub enum ColorSpace {
    /// Unknown or unspecified color space
    #[default]
    Unknown = 0,
    /// Grayscale (single channel)
    Grayscale = 1,
    /// RGB color space
    Rgb = 2,
    /// YCbCr color space (typical JPEG)
    YCbCr = 3,
    /// CMYK color space
    Cmyk = 4,
    /// YCCK color space (CMYK encoded as YCbCr + K)
    Ycck = 5,
    /// XYB color space (jpegli's perceptual color space)
    Xyb = 6,
}

impl ColorSpace {
    /// Returns the number of components for this color space.
    #[must_use]
    pub const fn num_components(self) -> usize {
        match self {
            Self::Unknown => 0,
            Self::Grayscale => 1,
            Self::Rgb | Self::YCbCr | Self::Xyb => 3,
            Self::Cmyk | Self::Ycck => 4,
        }
    }

    /// Returns true if this color space uses chroma subsampling by default.
    #[must_use]
    pub const fn default_subsampling(self) -> bool {
        matches!(self, Self::YCbCr | Self::Ycck)
    }
}

/// Pixel format for input/output data.
///
/// # Fast Paths
///
/// The following formats have SIMD-optimized conversion to YCbCr:
/// - [`Rgb`](Self::Rgb) - Most common, best performance
/// - [`Bgr`](Self::Bgr) - Windows/OpenCV, optimized swap
/// - [`Bgrx`](Self::Bgrx) - Windows BGRX32, padding ignored (fast path)
/// - [`Rgb16`](Self::Rgb16) - High precision input
/// - [`RgbF32`](Self::RgbF32) - Linear HDR input
///
/// Other formats are converted to a fast-path format first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum PixelFormat {
    // === 8-bit formats ===
    /// Grayscale, 1 byte per pixel
    Gray,
    /// RGB, 3 bytes per pixel (fast path)
    #[default]
    Rgb,
    /// RGBA, 4 bytes per pixel (alpha ignored)
    Rgba,
    /// BGR, 3 bytes per pixel (fast path)
    Bgr,
    /// BGRA, 4 bytes per pixel (alpha ignored)
    Bgra,
    /// BGRX, 4 bytes per pixel (padding ignored, fast path)
    ///
    /// Common Windows/DirectX format. The X byte is padding (not alpha).
    /// This is a fast path because we can process 4 pixels at a time
    /// without alpha handling overhead.
    Bgrx,

    // === 16-bit formats ===
    /// Grayscale, 2 bytes per pixel (native endian)
    Gray16,
    /// RGB, 6 bytes per pixel (native endian, fast path)
    Rgb16,
    /// RGBA, 8 bytes per pixel (native endian, alpha ignored)
    Rgba16,

    // === 32-bit float formats (linear color space) ===
    /// Grayscale, 4 bytes per pixel (linear, 0.0-1.0)
    GrayF32,
    /// RGB, 12 bytes per pixel (linear, 0.0-1.0, fast path)
    ///
    /// Input is assumed to be in **linear** color space (not sRGB).
    /// Values outside 0.0-1.0 are clamped.
    RgbF32,
    /// RGBA, 16 bytes per pixel (linear, alpha ignored)
    RgbaF32,

    // === Special formats ===
    /// CMYK, 4 bytes per pixel
    Cmyk,
}

impl PixelFormat {
    /// Returns the number of bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Gray16 => 2,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra | Self::Bgrx | Self::Cmyk | Self::GrayF32 => 4,
            Self::Rgb16 => 6,
            Self::Rgba16 => 8,
            Self::RgbF32 => 12,
            Self::RgbaF32 => 16,
        }
    }

    /// Returns the number of color channels (excluding alpha/padding).
    #[must_use]
    pub const fn num_channels(self) -> usize {
        match self {
            Self::Gray | Self::Gray16 | Self::GrayF32 => 1,
            Self::Rgb
            | Self::Bgr
            | Self::Rgba
            | Self::Bgra
            | Self::Bgrx
            | Self::Rgb16
            | Self::Rgba16
            | Self::RgbF32
            | Self::RgbaF32 => 3,
            Self::Cmyk => 4,
        }
    }

    /// Returns the corresponding color space.
    #[must_use]
    pub const fn color_space(self) -> ColorSpace {
        match self {
            Self::Gray | Self::Gray16 | Self::GrayF32 => ColorSpace::Grayscale,
            Self::Rgb
            | Self::Rgba
            | Self::Bgr
            | Self::Bgra
            | Self::Bgrx
            | Self::Rgb16
            | Self::Rgba16
            | Self::RgbF32
            | Self::RgbaF32 => ColorSpace::Rgb,
            Self::Cmyk => ColorSpace::Cmyk,
        }
    }

    /// Returns true if this is a grayscale format (1 channel).
    ///
    /// Grayscale formats: Gray, Gray16, GrayF32
    #[must_use]
    pub const fn is_grayscale(self) -> bool {
        matches!(self, Self::Gray | Self::Gray16 | Self::GrayF32)
    }

    /// Returns true if this format has a SIMD-optimized fast path.
    ///
    /// Fast path formats are converted directly to YCbCr planes without
    /// intermediate conversion steps.
    #[must_use]
    pub const fn is_fast_path(self) -> bool {
        matches!(
            self,
            Self::Rgb
                | Self::Bgr
                | Self::Bgrx
                | Self::Rgb16
                | Self::RgbF32
                | Self::Gray
                | Self::Gray16
                | Self::GrayF32
        )
    }

    /// Returns the bit depth per channel.
    #[must_use]
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::Gray
            | Self::Rgb
            | Self::Rgba
            | Self::Bgr
            | Self::Bgra
            | Self::Bgrx
            | Self::Cmyk => 8,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 => 16,
            Self::GrayF32 | Self::RgbF32 | Self::RgbaF32 => 32,
        }
    }

    /// Returns true if this is a floating-point format.
    ///
    /// Float formats are assumed to be in linear color space (not sRGB).
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::GrayF32 | Self::RgbF32 | Self::RgbaF32)
    }

    /// Returns true if this format has an alpha or padding channel.
    #[must_use]
    pub const fn has_alpha_or_padding(self) -> bool {
        matches!(
            self,
            Self::Rgba | Self::Bgra | Self::Bgrx | Self::Rgba16 | Self::RgbaF32
        )
    }
}

/// Chroma subsampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Subsampling {
    /// 4:4:4 - No subsampling
    #[default]
    S444,
    /// 4:2:2 - Horizontal subsampling only
    S422,
    /// 4:2:0 - Both horizontal and vertical subsampling
    S420,
    /// 4:4:0 - Vertical subsampling only (rare)
    S440,
}

impl Subsampling {
    /// Returns the horizontal sampling factor for luma.
    #[must_use]
    pub const fn h_samp_factor_luma(self) -> u8 {
        match self {
            Self::S444 | Self::S440 => 1,
            Self::S422 | Self::S420 => 2,
        }
    }

    /// Returns the vertical sampling factor for luma.
    #[must_use]
    pub const fn v_samp_factor_luma(self) -> u8 {
        match self {
            Self::S444 | Self::S422 => 1,
            Self::S420 | Self::S440 => 2,
        }
    }

    /// Returns the MCU (Minimum Coded Unit) size for this subsampling mode.
    ///
    /// - 16 for modes with 2x sampling (4:2:0, 4:2:2, 4:4:0)
    /// - 8 for 4:4:4 (no subsampling)
    #[must_use]
    pub const fn mcu_size(self) -> usize {
        match self {
            Self::S444 => 8,
            Self::S420 | Self::S422 | Self::S440 => 16,
        }
    }
}

impl From<crate::encode::encoder_types::ChromaSubsampling> for Subsampling {
    fn from(cs: crate::encode::encoder_types::ChromaSubsampling) -> Self {
        use crate::encode::encoder_types::ChromaSubsampling;
        match cs {
            ChromaSubsampling::None => Self::S444,
            ChromaSubsampling::HalfHorizontal => Self::S422,
            ChromaSubsampling::Quarter => Self::S420,
            ChromaSubsampling::HalfVertical => Self::S440,
        }
    }
}

impl From<Subsampling> for crate::encode::encoder_types::ChromaSubsampling {
    fn from(s: Subsampling) -> Self {
        match s {
            Subsampling::S444 => Self::None,
            Subsampling::S422 => Self::HalfHorizontal,
            Subsampling::S420 => Self::Quarter,
            Subsampling::S440 => Self::HalfVertical,
        }
    }
}

impl From<crate::encode::encoder_types::PixelLayout> for PixelFormat {
    fn from(layout: crate::encode::encoder_types::PixelLayout) -> Self {
        use crate::encode::encoder_types::PixelLayout;
        match layout {
            PixelLayout::Rgb8Srgb => Self::Rgb,
            PixelLayout::Bgr8Srgb => Self::Bgr,
            PixelLayout::Rgbx8Srgb | PixelLayout::Rgba8Srgb => Self::Rgba,
            PixelLayout::Bgrx8Srgb | PixelLayout::Bgra8Srgb => Self::Bgrx,
            PixelLayout::Gray8Srgb => Self::Gray,
            PixelLayout::Rgb16Linear => Self::Rgb16,
            PixelLayout::Rgbx16Linear | PixelLayout::Rgba16Linear => Self::Rgba16,
            PixelLayout::Gray16Linear => Self::Gray16,
            PixelLayout::RgbF32Linear => Self::RgbF32,
            PixelLayout::RgbxF32Linear | PixelLayout::RgbaF32Linear => Self::RgbaF32,
            PixelLayout::GrayF32Linear => Self::GrayF32,
            // YCbCr layouts don't have direct legacy equivalents
            PixelLayout::YCbCr8 | PixelLayout::YCbCrF32 => Self::Rgb,
        }
    }
}

/// Strategy for padding partial MCU blocks at image edges.
///
/// When image dimensions are not multiples of the MCU size (8 or 16 pixels),
/// the encoder must pad the edge blocks. Different strategies produce different
/// compression characteristics and visual artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum EdgePadding {
    /// Replicate the edge pixel outward (C++ jpegli behavior).
    ///
    /// For pixels beyond the edge, use the last valid pixel value.
    /// This is the safest choice for chroma as it produces smooth
    /// upsampling results with no color fringing.
    #[default]
    Replicate,

    /// Mirror/reflect at the edge boundary.
    ///
    /// For pixels beyond the edge, reflect back into the image.
    /// Better preserves gradients but may cause slight color fringing
    /// on chroma channels due to upsampling artifacts.
    Mirror,

    /// Wrap around (tile the image).
    ///
    /// For pixels beyond the edge, wrap to the opposite side.
    /// Generally produces poor results due to discontinuities.
    /// Included for completeness/testing.
    Wrap,
}

/// Edge padding configuration with per-channel control.
///
/// Allows different padding strategies for luma (Y) and chroma (Cb/Cr) channels.
/// This is useful because:
/// - Luma: Mirror can better preserve gradients at edges
/// - Chroma: Replicate is safer due to upsampling artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgePaddingConfig {
    /// Padding strategy for luma (Y) channel.
    pub luma: EdgePadding,
    /// Padding strategy for chroma (Cb/Cr) channels.
    pub chroma: EdgePadding,
}

impl EdgePaddingConfig {
    /// Use the same strategy for all channels.
    #[must_use]
    pub const fn uniform(strategy: EdgePadding) -> Self {
        Self {
            luma: strategy,
            chroma: strategy,
        }
    }

    /// Recommended configuration: Mirror for luma, Replicate for chroma.
    ///
    /// - Luma uses Mirror to better preserve gradients at edges
    /// - Chroma uses Replicate for safe upsampling without color fringing
    #[must_use]
    pub const fn recommended() -> Self {
        Self {
            luma: EdgePadding::Mirror,
            chroma: EdgePadding::Replicate,
        }
    }

    /// C++ jpegli compatible configuration: Replicate for all channels.
    #[must_use]
    pub const fn cpp_compat() -> Self {
        Self::uniform(EdgePadding::Replicate)
    }
}

impl Default for EdgePaddingConfig {
    fn default() -> Self {
        Self::cpp_compat()
    }
}

/// JPEG encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum JpegMode {
    /// Baseline sequential DCT (most compatible)
    #[default]
    Baseline,
    /// Extended sequential DCT (12-bit precision)
    Extended,
    /// Progressive DCT (multiple scans)
    Progressive,
    /// Lossless (not implemented)
    Lossless,
    /// Arithmetic sequential DCT (SOF9)
    ArithmeticSequential,
    /// Arithmetic progressive DCT (SOF10)
    ArithmeticProgressive,
}

/// Huffman table optimization algorithm.
///
/// Controls which algorithm is used to build optimal Huffman tables from
/// symbol frequency counts when `optimize_huffman` is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum HuffmanMethod {
    /// jpegli C++ CreateHuffmanTree algorithm (default for jpegli compatibility).
    ///
    /// Uses the sorted two-pointer merge approach from jpegli's huffman.cc.
    /// This is the reference implementation that we're porting.
    #[default]
    JpegliCreateTree,

    /// mozjpeg/libjpeg classic algorithm (JPEG spec Section K.2).
    ///
    /// Uses the classic Huffman merge with chain tracking. Simpler implementation
    /// (~100 lines vs ~150) and follows the JPEG specification exactly.
    ///
    /// In testing: 100/122 cases match jpegli, 22 cases produce 1 bit less (better).
    /// Can be useful for maximum compatibility or when jpegli's algorithm has issues.
    MozjpegClassic,
}

/// A single component in a JPEG image.
#[derive(Debug, Clone)]
pub struct Component {
    /// Component ID (1-255, typically 1=Y, 2=Cb, 3=Cr)
    pub id: u8,
    /// Horizontal sampling factor (1-4)
    pub h_samp_factor: u8,
    /// Vertical sampling factor (1-4)
    pub v_samp_factor: u8,
    /// Quantization table index (0-3)
    pub quant_table_idx: u8,
    /// DC Huffman table index (0-1)
    pub dc_huffman_idx: u8,
    /// AC Huffman table index (0-1)
    pub ac_huffman_idx: u8,
}

impl Default for Component {
    fn default() -> Self {
        Self {
            id: 0,
            h_samp_factor: 1,
            v_samp_factor: 1,
            quant_table_idx: 0,
            dc_huffman_idx: 0,
            ac_huffman_idx: 0,
        }
    }
}

/// A quantization table.
#[derive(Debug, Clone)]
pub struct QuantTable {
    /// Quantization values in zigzag order (1-255 for baseline, 1-65535 for extended)
    pub values: [u16; DCT_BLOCK_SIZE],
    /// Precision: 0 = 8-bit, 1 = 16-bit
    pub precision: u8,
}

impl Default for QuantTable {
    fn default() -> Self {
        Self {
            values: [16; DCT_BLOCK_SIZE], // Default to flat table
            precision: 0,
        }
    }
}

impl QuantTable {
    /// Creates a new quantization table from values in natural (row-major) order.
    #[must_use]
    pub fn from_natural_order(values: &[u16; DCT_BLOCK_SIZE]) -> Self {
        let mut zigzag = [0u16; DCT_BLOCK_SIZE];
        for (i, &v) in values.iter().enumerate() {
            let zi = crate::foundation::consts::JPEG_ZIGZAG_ORDER[i] as usize;
            zigzag[zi] = v;
        }
        Self {
            values: zigzag,
            precision: if values.iter().any(|&v| v > 255) {
                1
            } else {
                0
            },
        }
    }

    /// Returns values in natural (row-major) order.
    #[must_use]
    pub fn to_natural_order(&self) -> [u16; DCT_BLOCK_SIZE] {
        let mut natural = [0u16; DCT_BLOCK_SIZE];
        for (i, &zi) in crate::foundation::consts::JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE]
            .iter()
            .enumerate()
        {
            natural[zi as usize] = self.values[i];
        }
        natural
    }

    /// Clamp values to baseline range (1-255) and set precision to 0.
    ///
    /// This ensures the table produces baseline-compatible JPEGs (SOF0).
    #[must_use]
    pub fn clamp_to_baseline(self) -> Self {
        let mut values = self.values;
        for v in &mut values {
            *v = (*v).clamp(1, 255);
        }
        Self {
            values,
            precision: 0,
        }
    }
}

/// A Huffman table.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// Number of codes of each length (1-16 bits)
    pub bits: [u8; 16],
    /// Symbol values (up to 256)
    pub values: Vec<u8>,
    /// True if this is a DC table, false for AC
    pub is_dc: bool,
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            bits: [0; 16],
            values: Vec::new(),
            is_dc: true,
        }
    }
}

/// DCT coefficient type (after quantization).
pub type Coeff = i16;

/// A single 8x8 block of DCT coefficients.
pub type CoeffBlock = [Coeff; DCT_BLOCK_SIZE];

/// Image dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Dimensions {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Dimensions {
    /// Creates new dimensions.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns the number of 8x8 blocks horizontally.
    #[must_use]
    pub const fn width_in_blocks(self) -> u32 {
        (self.width + 7) / 8
    }

    /// Returns the number of 8x8 blocks vertically.
    #[must_use]
    pub const fn height_in_blocks(self) -> u32 {
        (self.height + 7) / 8
    }

    /// Returns the total number of pixels.
    #[must_use]
    pub const fn num_pixels(self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

/// Resource limits for encoding and decoding operations.
///
/// All fields are `Option<u64>` — `None` means no limit (unlimited).
/// This struct provides consistent resource management across encoder and decoder.
///
/// # Defaults
///
/// By default, all limits are `None` (unlimited). Use the builder methods to set specific limits.
///
/// # Example
///
/// ```
/// use zenjpeg::encoder::Limits;
///
/// let limits = Limits::default()
///     .max_pixels(100_000_000)       // 100 megapixels
///     .max_memory(512 * 1024 * 1024) // 512 MB
///     .max_output(50 * 1024 * 1024); // 50 MB output
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Limits {
    /// Maximum number of pixels (width * height). Prevents DoS from decompression bombs.
    pub max_pixels: Option<u64>,
    /// Maximum total memory allocation in bytes.
    pub max_memory: Option<u64>,
    /// Maximum output size in bytes (for encoding).
    pub max_output: Option<u64>,
}

impl Limits {
    /// Sets the maximum number of pixels allowed.
    #[must_use]
    pub fn max_pixels(mut self, pixels: u64) -> Self {
        self.max_pixels = Some(pixels);
        self
    }

    /// Sets the maximum total memory allocation in bytes.
    #[must_use]
    pub fn max_memory(mut self, bytes: u64) -> Self {
        self.max_memory = Some(bytes);
        self
    }

    /// Sets the maximum output size in bytes.
    #[must_use]
    pub fn max_output(mut self, bytes: u64) -> Self {
        self.max_output = Some(bytes);
        self
    }

    /// Returns the pixel limit, or `u64::MAX` if unlimited.
    #[must_use]
    pub fn effective_max_pixels(&self) -> u64 {
        self.max_pixels.unwrap_or(u64::MAX)
    }

    /// Returns the memory limit, or `u64::MAX` if unlimited.
    #[must_use]
    pub fn effective_max_memory(&self) -> u64 {
        self.max_memory.unwrap_or(u64::MAX)
    }

    /// Returns the output size limit, or `u64::MAX` if unlimited.
    #[must_use]
    pub fn effective_max_output(&self) -> u64 {
        self.max_output.unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_components() {
        assert_eq!(ColorSpace::Grayscale.num_components(), 1);
        assert_eq!(ColorSpace::Rgb.num_components(), 3);
        assert_eq!(ColorSpace::YCbCr.num_components(), 3);
        assert_eq!(ColorSpace::Cmyk.num_components(), 4);
    }

    #[test]
    fn test_pixel_format_bytes() {
        assert_eq!(PixelFormat::Gray.bytes_per_pixel(), 1);
        assert_eq!(PixelFormat::Rgb.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_quant_table_order_conversion() {
        let natural = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];
        let table = QuantTable::from_natural_order(&natural);
        let recovered = table.to_natural_order();
        assert_eq!(natural, recovered);
    }
}
