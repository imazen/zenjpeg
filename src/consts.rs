//! Constants and tables for JPEG encoding
//!
//! This module contains standard JPEG tables as well as optimized tables
//! from mozjpeg and jpegli.

/// DCT block dimension
pub const DCTSIZE: usize = 8;

/// DCT block size (8x8 = 64)
pub const DCTSIZE2: usize = 64;

/// Zigzag scan order: maps zigzag position to natural (row-major) position.
/// Use this when iterating in zigzag order to access coefficients.
/// Example: natural_pos = JPEG_NATURAL_ORDER[zigzag_pos]
pub const JPEG_NATURAL_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Alias for backwards compatibility
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Standard JPEG Annex K luminance quantization table
pub const STD_LUMA_QUANT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG Annex K chrominance quantization table
pub const STD_CHROMA_QUANT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// mozjpeg ImageMagick-style luminance table (index 3)
/// Used by mozjpeg for better quality at same file size
pub const MOZJPEG_LUMA_QUANT: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
    135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74, 94,
    124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311, 418,
];

/// jpegli zero-bias thresholds for luminance (from C++ source)
pub const JPEGLI_ZERO_BIAS_LUMA: [f32; 64] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// JPEG markers
pub mod marker {
    pub const SOI: u8 = 0xD8; // Start of image
    pub const EOI: u8 = 0xD9; // End of image
    pub const SOF0: u8 = 0xC0; // Baseline DCT
    pub const SOF2: u8 = 0xC2; // Progressive DCT
    pub const DHT: u8 = 0xC4; // Define Huffman table
    pub const DQT: u8 = 0xDB; // Define quantization table
    pub const DRI: u8 = 0xDD; // Define restart interval
    pub const SOS: u8 = 0xDA; // Start of scan
    pub const APP0: u8 = 0xE0; // JFIF marker
    pub const APP1: u8 = 0xE1; // EXIF marker
    pub const APP2: u8 = 0xE2; // ICC profile marker
    pub const COM: u8 = 0xFE; // Comment
}

/// Quality crossover point where jpegli-style encoding becomes better
/// Below this, mozjpeg-style trellis is more effective
pub const QUALITY_CROSSOVER: u8 = 70;

/// Default quality for perceptual encoding
pub const DEFAULT_QUALITY: u8 = 85;

// Re-export Huffman tables and quantization tables from consts_moz
pub use crate::consts_moz::{
    AC_CHROMINANCE_BITS, AC_CHROMINANCE_VALUES, AC_LUMINANCE_BITS, AC_LUMINANCE_VALUES,
    DC_CHROMINANCE_BITS, DC_CHROMINANCE_VALUES, DC_LUMINANCE_BITS, DC_LUMINANCE_VALUES,
    STD_CHROMINANCE_QUANT_TBL, STD_LUMINANCE_QUANT_TBL,
};
