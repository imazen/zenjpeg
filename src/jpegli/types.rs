//! Core types for jpegli.

use crate::jpegli::consts::DCT_BLOCK_SIZE;

/// Color space representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PixelFormat {
    /// Grayscale, 1 byte per pixel
    Gray,
    /// RGB, 3 bytes per pixel
    #[default]
    Rgb,
    /// RGBA, 4 bytes per pixel (alpha is ignored for encoding)
    Rgba,
    /// BGR, 3 bytes per pixel
    Bgr,
    /// BGRA, 4 bytes per pixel
    Bgra,
    /// CMYK, 4 bytes per pixel
    Cmyk,
}

impl PixelFormat {
    /// Returns the number of bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra | Self::Cmyk => 4,
        }
    }

    /// Returns the number of color channels (excluding alpha).
    #[must_use]
    pub const fn num_channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb | Self::Bgr | Self::Rgba | Self::Bgra => 3,
            Self::Cmyk => 4,
        }
    }

    /// Returns the corresponding color space.
    #[must_use]
    pub const fn color_space(self) -> ColorSpace {
        match self {
            Self::Gray => ColorSpace::Grayscale,
            Self::Rgb | Self::Rgba | Self::Bgr | Self::Bgra => ColorSpace::Rgb,
            Self::Cmyk => ColorSpace::Cmyk,
        }
    }
}

/// Sample bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SampleDepth {
    /// 8-bit samples (0-255)
    #[default]
    Bits8,
    /// 16-bit samples (0-65535)
    Bits16,
    /// 32-bit floating point samples (0.0-1.0)
    Float32,
}

impl SampleDepth {
    /// Returns the number of bytes per sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Bits8 => 1,
            Self::Bits16 => 2,
            Self::Float32 => 4,
        }
    }
}

/// Chroma subsampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
}

/// JPEG encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
            let zi = crate::jpegli::consts::JPEG_ZIGZAG_ORDER[i] as usize;
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
        for (i, &zi) in crate::jpegli::consts::JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE]
            .iter()
            .enumerate()
        {
            natural[zi as usize] = self.values[i];
        }
        natural
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

/// Scan parameters for progressive JPEG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanSpec {
    /// First component index in this scan
    pub comp_start: u8,
    /// Number of components in this scan
    pub num_comps: u8,
    /// Spectral selection start (0-63)
    pub ss: u8,
    /// Spectral selection end (0-63)
    pub se: u8,
    /// Successive approximation high bit
    pub ah: u8,
    /// Successive approximation low bit
    pub al: u8,
}

impl Default for ScanSpec {
    fn default() -> Self {
        Self {
            comp_start: 0,
            num_comps: 3,
            ss: 0,
            se: 63,
            ah: 0,
            al: 0,
        }
    }
}

/// Restart interval (number of MCUs between restart markers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RestartInterval(pub u16);

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
