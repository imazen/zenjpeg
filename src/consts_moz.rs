//! JPEG constants, markers, and quantization tables.
//!
//! This module contains all static data required for JPEG encoding:
//! - JPEG marker codes
//! - DCT block dimensions
//! - Zigzag scan order
//! - 9 quantization table variants from mozjpeg

// =============================================================================
// JPEG Markers
// =============================================================================

/// Start of image marker
pub const JPEG_SOI: u8 = 0xD8;
/// End of image marker
pub const JPEG_EOI: u8 = 0xD9;
/// Start of frame (baseline DCT)
pub const JPEG_SOF0: u8 = 0xC0;
/// Start of frame (progressive DCT)
pub const JPEG_SOF2: u8 = 0xC2;
/// Define Huffman table
pub const JPEG_DHT: u8 = 0xC4;
/// Define quantization table
pub const JPEG_DQT: u8 = 0xDB;
/// Define restart interval
pub const JPEG_DRI: u8 = 0xDD;
/// Start of scan
pub const JPEG_SOS: u8 = 0xDA;
/// Restart markers (RST0-RST7)
pub const JPEG_RST0: u8 = 0xD0;
/// APP0 marker (JFIF)
pub const JPEG_APP0: u8 = 0xE0;
/// APP1 marker (EXIF)
pub const JPEG_APP1: u8 = 0xE1;
/// APP14 marker (Adobe)
pub const JPEG_APP14: u8 = 0xEE;
/// Comment marker
pub const JPEG_COM: u8 = 0xFE;

// =============================================================================
// DCT Constants
// =============================================================================

/// DCT block size (8x8)
pub const DCTSIZE: usize = 8;
/// DCT block size squared (64 coefficients)
pub const DCTSIZE2: usize = 64;
/// Number of quantization tables (0-3)
pub const NUM_QUANT_TBLS: usize = 4;
/// Number of Huffman tables (0-3)
pub const NUM_HUFF_TBLS: usize = 4;
/// Maximum components in one scan
pub const MAX_COMPS_IN_SCAN: usize = 4;
/// Maximum sampling factor
pub const MAX_SAMP_FACTOR: usize = 4;

// =============================================================================
// Zigzag Order
// =============================================================================

/// Natural order to zigzag order mapping.
/// Maps linear index [0..63] to zigzag position.
pub const JPEG_NATURAL_ORDER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Zigzag order to natural order mapping (inverse of JPEG_NATURAL_ORDER).
pub const JPEG_ZIGZAG_ORDER: [usize; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

// =============================================================================
// Quantization Tables
// =============================================================================
// mozjpeg provides 9 different quantization table sets, indexed 0-8.
// Each set has a luminance table and a chrominance table.

/// Number of quantization table variants
pub const NUM_QUANT_TABLE_VARIANTS: usize = 9;

/// Quantization table variant indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QuantTableIdx {
    /// JPEG Annex K (standard)
    JpegAnnexK = 0,
    /// Flat (uniform quantization)
    Flat = 1,
    /// MSSIM-tuned on Kodak image set
    MssimTuned = 2,
    /// ImageMagick default (mozjpeg default for MAX_COMPRESSION)
    ImageMagick = 3,
    /// PSNR-HVS-M tuned
    PsnrHvsM = 4,
    /// Klein, Silverstein, Carney (1992)
    Klein = 5,
    /// Watson, Taylor, Borthwick (DCTune, 1997)
    Watson = 6,
    /// Ahumada, Watson, Peterson (1993)
    Ahumada = 7,
    /// Peterson, Ahumada, Watson (1993)
    Peterson = 8,
}

impl QuantTableIdx {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::JpegAnnexK),
            1 => Some(Self::Flat),
            2 => Some(Self::MssimTuned),
            3 => Some(Self::ImageMagick),
            4 => Some(Self::PsnrHvsM),
            5 => Some(Self::Klein),
            6 => Some(Self::Watson),
            7 => Some(Self::Ahumada),
            8 => Some(Self::Peterson),
            _ => None,
        }
    }
}

/// Standard luminance quantization tables (9 variants).
/// Source: mozjpeg/jcparam.c std_luminance_quant_tbl
pub const STD_LUMINANCE_QUANT_TBL: [[u16; DCTSIZE2]; NUM_QUANT_TABLE_VARIANTS] = [
    // 0: JPEG Annex K
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ],
    // 1: Flat
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    // 2: MSSIM-tuned on Kodak
    [
        12, 17, 20, 21, 30, 34, 56, 63, 18, 20, 20, 26, 28, 51, 61, 55, 19, 20, 21, 26, 33, 58, 69,
        55, 26, 26, 26, 30, 46, 87, 86, 66, 31, 33, 36, 40, 46, 96, 100, 73, 40, 35, 46, 62, 81,
        100, 111, 91, 46, 66, 76, 86, 102, 121, 120, 101, 68, 90, 90, 96, 113, 102, 105, 103,
    ],
    // 3: ImageMagick (mozjpeg default for MAX_COMPRESSION)
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    // 4: PSNR-HVS-M tuned
    [
        9, 10, 12, 14, 27, 32, 51, 62, 11, 12, 14, 19, 27, 44, 59, 73, 12, 14, 18, 25, 42, 59, 79,
        78, 17, 18, 25, 42, 61, 92, 87, 92, 23, 28, 42, 75, 79, 112, 112, 99, 40, 42, 59, 84, 88,
        124, 132, 111, 42, 64, 78, 95, 105, 126, 125, 99, 70, 75, 100, 102, 116, 100, 107, 98,
    ],
    // 5: Klein, Silverstein, Carney (1992)
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    // 6: Watson, Taylor, Borthwick (DCTune, 1997)
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    // 7: Ahumada, Watson, Peterson (1993)
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    // 8: Peterson, Ahumada, Watson (1993)
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

/// Standard chrominance quantization tables (9 variants).
/// Source: mozjpeg/jcparam.c std_chrominance_quant_tbl
pub const STD_CHROMINANCE_QUANT_TBL: [[u16; DCTSIZE2]; NUM_QUANT_TABLE_VARIANTS] = [
    // 0: JPEG Annex K
    [
        17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99,
        99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
    // 1: Flat
    [
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    ],
    // 2: MSSIM-tuned on Kodak
    [
        8, 12, 15, 15, 86, 96, 96, 98, 13, 13, 15, 26, 90, 96, 99, 98, 12, 15, 18, 96, 99, 99, 99,
        99, 17, 16, 90, 96, 99, 99, 99, 99, 96, 96, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    ],
    // 3: ImageMagick (copied from luminance)
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    // 4: PSNR-HVS-M tuned
    [
        9, 10, 17, 19, 62, 89, 91, 97, 12, 13, 18, 29, 84, 91, 88, 98, 14, 19, 29, 93, 95, 95, 98,
        97, 20, 26, 84, 88, 95, 95, 98, 94, 26, 86, 91, 93, 97, 99, 98, 99, 99, 100, 98, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 97, 97, 99, 99, 99, 99, 97, 99,
    ],
    // 5: Klein (copied from luminance)
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    // 6: Watson (copied from luminance)
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 19, 31, 58, 127,
        255, 14, 15, 19, 27, 44, 83, 181, 255, 23, 25, 31, 44, 72, 136, 255, 255, 44, 47, 58, 83,
        136, 255, 255, 255, 95, 102, 127, 181, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    // 7: Ahumada (copied from luminance)
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    // 8: Peterson (copied from luminance)
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

// =============================================================================
// Trellis Quantization Constants
// =============================================================================

/// Default lambda log scale 1 for trellis quantization
pub const DEFAULT_LAMBDA_LOG_SCALE1: f32 = 14.75;
/// Default lambda log scale 2 for trellis quantization
pub const DEFAULT_LAMBDA_LOG_SCALE2: f32 = 16.5;
/// Default trellis frequency split point
pub const DEFAULT_TRELLIS_FREQ_SPLIT: i32 = 8;
/// Default number of trellis optimization loops
pub const DEFAULT_TRELLIS_NUM_LOOPS: i32 = 1;
/// Default DC delta weight for trellis
pub const DEFAULT_TRELLIS_DELTA_DC_WEIGHT: f32 = 0.0;

// =============================================================================
// Standard Huffman Tables
// =============================================================================

/// DC luminance Huffman table - number of codes of each length (bits)
pub const DC_LUMINANCE_BITS: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];

/// DC luminance Huffman table - symbol values
pub const DC_LUMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// DC chrominance Huffman table - number of codes of each length
pub const DC_CHROMINANCE_BITS: [u8; 17] = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

/// DC chrominance Huffman table - symbol values
pub const DC_CHROMINANCE_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// AC luminance Huffman table - number of codes of each length
pub const AC_LUMINANCE_BITS: [u8; 17] = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];

/// AC luminance Huffman table - symbol values (162 symbols)
pub const AC_LUMINANCE_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// AC chrominance Huffman table - number of codes of each length
pub const AC_CHROMINANCE_BITS: [u8; 17] = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];

/// AC chrominance Huffman table - symbol values (162 symbols)
pub const AC_CHROMINANCE_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_inverse() {
        // Verify that JPEG_ZIGZAG_ORDER is the inverse of JPEG_NATURAL_ORDER
        for i in 0..64 {
            let zigzag_pos = JPEG_NATURAL_ORDER[i];
            assert_eq!(
                JPEG_ZIGZAG_ORDER[zigzag_pos], i,
                "Zigzag inverse mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn test_quant_table_sizes() {
        assert_eq!(STD_LUMINANCE_QUANT_TBL.len(), NUM_QUANT_TABLE_VARIANTS);
        assert_eq!(STD_CHROMINANCE_QUANT_TBL.len(), NUM_QUANT_TABLE_VARIANTS);

        for i in 0..NUM_QUANT_TABLE_VARIANTS {
            assert_eq!(STD_LUMINANCE_QUANT_TBL[i].len(), DCTSIZE2);
            assert_eq!(STD_CHROMINANCE_QUANT_TBL[i].len(), DCTSIZE2);
        }
    }

    #[test]
    fn test_quant_table_nonzero() {
        // All quantization values should be > 0
        for variant in 0..NUM_QUANT_TABLE_VARIANTS {
            for i in 0..DCTSIZE2 {
                assert!(
                    STD_LUMINANCE_QUANT_TBL[variant][i] > 0,
                    "Zero quant value in luminance table {} at {}",
                    variant,
                    i
                );
                assert!(
                    STD_CHROMINANCE_QUANT_TBL[variant][i] > 0,
                    "Zero quant value in chrominance table {} at {}",
                    variant,
                    i
                );
            }
        }
    }

    #[test]
    fn test_huffman_table_consistency() {
        // Count total symbols from bits array
        let dc_lum_count: usize = DC_LUMINANCE_BITS[1..].iter().map(|&x| x as usize).sum();
        assert_eq!(dc_lum_count, DC_LUMINANCE_VALUES.len());

        let dc_chr_count: usize = DC_CHROMINANCE_BITS[1..].iter().map(|&x| x as usize).sum();
        assert_eq!(dc_chr_count, DC_CHROMINANCE_VALUES.len());

        let ac_lum_count: usize = AC_LUMINANCE_BITS[1..].iter().map(|&x| x as usize).sum();
        assert_eq!(ac_lum_count, AC_LUMINANCE_VALUES.len());

        let ac_chr_count: usize = AC_CHROMINANCE_BITS[1..].iter().map(|&x| x as usize).sum();
        assert_eq!(ac_chr_count, AC_CHROMINANCE_VALUES.len());
    }
}
