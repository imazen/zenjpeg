//! JPEG and jpegli constants.
//!
//! This module contains all the fundamental constants used in JPEG encoding/decoding,
//! including zigzag order tables, quantization matrices, and XYB color space parameters.

/// DCT block dimension (8x8)
pub const DCT_SIZE: usize = 8;

/// DCT block size (64 coefficients)
pub const DCT_BLOCK_SIZE: usize = 64;

/// Maximum number of color components
pub const MAX_COMPONENTS: usize = 4;

/// Maximum number of quantization tables
pub const MAX_QUANT_TABLES: usize = 4;

/// Maximum number of Huffman tables
pub const MAX_HUFFMAN_TABLES: usize = 4;

/// Maximum number of Huffman codes per table
pub const MAX_HUFFMAN_CODES: usize = 256;

/// JPEG precision (8 bits)
pub const JPEG_PRECISION: u32 = 8;

/// Maximum Huffman code bit length
pub const HUFFMAN_MAX_BIT_LENGTH: usize = 16;

/// Huffman alphabet size
pub const HUFFMAN_ALPHABET_SIZE: usize = 256;

/// DC coefficient alphabet size
pub const DC_ALPHABET_SIZE: usize = 12;

/// Maximum image dimension in pixels
pub const MAX_DIM_PIXELS: u32 = 65535;

// =============================================================================
// JPEG Markers
// =============================================================================

/// Start of Image marker
pub const MARKER_SOI: u8 = 0xD8;
/// End of Image marker
pub const MARKER_EOI: u8 = 0xD9;
/// Start of Frame (Baseline)
pub const MARKER_SOF0: u8 = 0xC0;
/// Start of Frame (Extended Sequential)
pub const MARKER_SOF1: u8 = 0xC1;
/// Start of Frame (Progressive)
pub const MARKER_SOF2: u8 = 0xC2;
/// Define Huffman Table
pub const MARKER_DHT: u8 = 0xC4;
/// Define Quantization Table
pub const MARKER_DQT: u8 = 0xDB;
/// Define Restart Interval
pub const MARKER_DRI: u8 = 0xDD;
/// Start of Scan
pub const MARKER_SOS: u8 = 0xDA;
/// Restart marker base (0-7)
pub const MARKER_RST0: u8 = 0xD0;
/// Application marker 0 (JFIF)
pub const MARKER_APP0: u8 = 0xE0;
/// Application marker 1 (EXIF)
pub const MARKER_APP1: u8 = 0xE1;
/// Application marker 2 (ICC Profile)
pub const MARKER_APP2: u8 = 0xE2;
/// Comment marker
pub const MARKER_COM: u8 = 0xFE;

/// ICC Profile marker tag
pub const ICC_PROFILE_TAG: &[u8; 12] = b"ICC_PROFILE\0";
/// EXIF marker tag
pub const EXIF_TAG: &[u8; 6] = b"Exif\0\0";

// =============================================================================
// Zigzag Order Tables
// =============================================================================

/// JPEG natural order (zigzag to linear).
/// Maps zigzag index to linear 8x8 block index.
/// Extra entries (64-79) are set to 63 for safety in decoder.
#[rustfmt::skip]
pub const JPEG_NATURAL_ORDER: [u8; 80] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    // Extra entries for safety in decoder
    63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63,
];

/// JPEG zigzag order (linear to zigzag).
/// Maps linear 8x8 block index to zigzag index.
#[rustfmt::skip]
pub const JPEG_ZIGZAG_ORDER: [u8; 64] = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
];

// =============================================================================
// Quantization - Global Scales
// =============================================================================

/// Global scale for XYB color space quantization.
/// Fitted to match butteraugli 3-norm with libjpeg at quality 90.
pub const GLOBAL_SCALE_XYB: f32 = 1.439_516_68;

/// Global scale for YCbCr color space quantization.
pub const GLOBAL_SCALE_YCBCR: f32 = 1.739_660_1;

/// Global scale for 4:2:0 chroma subsampling.
pub const GLOBAL_SCALE_420: f32 = 1.22;

/// Distance threshold where non-linear scaling kicks in.
pub const DIST_NONLINEAR_THRESHOLD: f32 = 1.5;

// =============================================================================
// Quantization - Base Matrices
// =============================================================================

/// Base quantization matrix for XYB color space.
/// Three 8x8 matrices (192 values total), one per channel (X, Y, B).
#[rustfmt::skip]
pub const BASE_QUANT_MATRIX_XYB: [f32; 192] = [
    // Channel 0 (X)
    7.562_993_5, 19.824_781, 22.572_495, 20.670_67, 22.686_459, 23.569_628, 25.812_908, 36.330_757,
    19.824_781, 21.550_318, 19.937_223, 20.542_421, 21.864_55, 23.904_139, 28.284_407, 32.660_976,
    22.572_495, 19.937_223, 21.901_726, 19.122_345, 21.751_581, 24.672_47, 25.424_965, 32.665_382,
    20.670_67, 20.542_421, 19.122_345, 20.161_022, 25.371_969, 25.966_89, 30.980_495, 31.340_601,
    22.686_459, 21.864_55, 21.751_581, 25.371_969, 26.243_185, 40.599_22, 43.262_463, 63.301_094,
    23.569_628, 23.904_139, 24.672_47, 25.966_89, 40.599_22, 48.302_677, 34.096_436, 61.985_214,
    25.812_908, 28.284_407, 25.424_965, 30.980_495, 43.262_463, 34.096_436, 34.493_744, 66.970_276,
    36.330_757, 32.660_976, 32.665_382, 31.340_601, 63.301_094, 61.985_214, 66.970_276, 39.965_27,
    // Channel 1 (Y)
    1.626_200_1, 3.219_924_2, 3.490_378, 3.914_835_9, 4.833_721_2, 4.910_884_4, 5.313_712, 6.167_679_3,
    3.219_924_2, 3.454_789_9, 3.603_683, 4.265_283_6, 4.836_838_7, 4.822_622_3, 5.612_051_5, 6.343_147_3,
    3.490_378, 3.603_683, 3.904_456, 4.337_439_5, 4.843_509_7, 5.405_798, 5.606_636, 6.107_513_4,
    3.914_835_9, 4.265_283_6, 4.337_439_5, 4.606_483_5, 5.175_147_5, 5.401_392_5, 6.039_981, 6.782_523,
    4.833_721_2, 4.836_838_7, 4.843_509_7, 5.175_147_5, 5.374_805, 6.141_083_7, 7.652_930_7, 7.523_521_4,
    4.910_884_4, 4.822_622_3, 5.405_798, 5.401_392_5, 6.141_083_7, 6.343_147_3, 7.108_305, 7.600_83,
    5.313_712, 5.612_051_5, 5.606_636, 6.039_981, 7.652_930_7, 7.108_305, 7.094_315_5, 7.047_836_3,
    6.167_679_3, 6.343_147_3, 6.107_513_4, 6.782_523, 7.523_521_4, 7.600_83, 7.047_836_3, 6.918_614_4,
    // Channel 2 (B)
    3.303_847_3, 10.068_926, 12.278_522, 14.604_117, 16.210_732, 19.231_453, 28.012_955, 55.668_29,
    10.068_926, 11.408_502, 11.387_135, 15.493_417, 16.536_493, 14.915_342, 26.374_872, 40.861_443,
    12.278_522, 11.387_135, 17.088_688, 13.950_035, 16.000_322, 28.566_063, 26.212_42, 30.126_013,
    14.604_117, 15.493_417, 13.950_035, 21.123_503, 26.157_978, 25.557_922, 40.685_936, 33.805_634,
    16.210_732, 16.536_493, 16.000_322, 26.157_978, 26.804_283, 26.158_772, 35.734_398, 43.685_703,
    19.231_453, 14.915_342, 28.566_063, 25.557_922, 26.158_772, 34.541_813, 41.319_794, 48.786_766,
    28.012_955, 26.374_872, 26.212_42, 40.685_936, 35.734_398, 41.319_794, 47.632_946, 55.349_846,
    55.668_29, 40.861_443, 30.126_013, 33.805_634, 43.685_703, 48.786_766, 55.349_846, 63.606_56,
];

/// Base quantization matrix for YCbCr color space.
/// Three 8x8 matrices (192 values total), one per channel (Y, Cb, Cr).
#[rustfmt::skip]
pub const BASE_QUANT_MATRIX_YCBCR: [f32; 192] = [
    // Channel 0 (Y - Luminance)
    1.239_740_9, 1.722_711_5, 2.921_216_7, 2.812_737_4, 3.339_819_7, 3.463_603_8, 3.840_915_2, 3.869_56,
    1.722_711_5, 2.092_889_4, 2.845_676, 2.704_506_8, 3.440_767_4, 3.166_232_4, 4.025_208_7, 4.035_324_5,
    2.921_216_7, 2.845_676, 2.958_740_4, 3.386_295, 3.619_523_8, 3.904_628, 3.757_835_8, 4.237_447_5,
    2.812_737_4, 2.704_506_8, 3.386_295, 3.380_058_8, 4.167_986_7, 4.805_510_6, 4.784_259, 4.605_934,
    3.339_819_7, 3.440_767_4, 3.619_523_8, 4.167_986_7, 4.579_851_3, 4.923_237, 5.574_107, 5.485_333_4,
    3.463_603_8, 3.166_232_4, 3.904_628, 4.805_510_6, 4.923_237, 5.439_36, 5.093_895_7, 6.087_225_4,
    3.840_915_2, 4.025_208_7, 3.757_835_8, 4.784_259, 5.574_107, 5.093_895_7, 5.438_461, 5.403_736,
    3.869_56, 4.035_324_5, 4.237_447_5, 4.605_934, 5.485_333_4, 6.087_225_4, 5.403_736, 4.377_871,
    // Channel 1 (Cb - Blue difference)
    2.823_619_8, 6.495_639_4, 9.310_489, 10.647_479, 11.074_191, 17.146_39, 18.463_982, 29.087_002,
    6.495_639_4, 8.890_104, 8.976_895_8, 13.666_27, 16.547_072, 16.638_714, 26.778_397, 21.330_343,
    9.310_489, 8.976_895_8, 11.087_377, 18.205_482, 19.752_482, 23.985_66, 102.645_74, 24.450_989,
    10.647_479, 13.666_27, 18.205_482, 18.628_012, 16.042_51, 25.049_183, 25.017_14, 35.797_89,
    11.074_191, 16.547_072, 19.752_482, 16.042_51, 19.373_483, 14.677_53, 19.946_96, 51.094_112,
    17.146_39, 16.638_714, 23.985_66, 25.049_183, 14.677_53, 31.320_412, 46.357_234, 67.481_11,
    18.463_982, 26.778_397, 102.645_74, 25.017_14, 19.946_96, 46.357_234, 61.315_765, 88.346_65,
    29.087_002, 21.330_343, 24.450_989, 35.797_89, 51.094_112, 67.481_11, 88.346_65, 112.160_99,
    // Channel 2 (Cr - Red difference)
    2.921_725_5, 4.497_681, 7.356_344_5, 6.583_891_5, 8.535_608_7, 8.799_434_4, 9.188_341_5, 9.482_7,
    4.497_681, 6.309_548_9, 7.024_609, 7.156_445_3, 8.049_059_2, 7.012_429, 6.711_923_2, 8.380_308,
    7.356_344_5, 7.024_609, 6.892_101_2, 6.882_82, 8.782_226, 6.877_475, 7.885_817_6, 8.679_09,
    6.583_891_5, 7.156_445_3, 6.882_82, 7.003_073, 7.722_346_5, 7.955_425_7, 7.473_411, 8.362_933,
    8.535_608_7, 8.049_059_2, 8.782_226, 7.722_346_5, 6.778_005_9, 9.484_922_7, 9.043_702_7, 8.053_178_2,
    8.799_434_4, 7.012_429, 6.877_475, 7.955_425_7, 9.484_922_7, 8.607_606_5, 9.922_697_4, 64.251_35,
    9.188_341_5, 6.711_923_2, 7.885_817_6, 7.473_411, 9.043_702_7, 9.922_697_4, 63.184_937, 83.352_94,
    9.482_7, 8.380_308, 8.679_09, 8.362_933, 8.053_178_2, 64.251_35, 83.352_94, 114.892_02,
];

/// Standard JPEG quantization matrix (from JPEG spec Annex K).
/// Two 8x8 matrices (128 values), one for luminance and one for chrominance.
#[rustfmt::skip]
pub const BASE_QUANT_MATRIX_STD: [f32; 128] = [
    // Luminance
    16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0,
    12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0,
    14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0,
    14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0,
    18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0,
    24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0, 92.0,
    49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0,
    72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0,
    // Chrominance
    17.0, 18.0, 24.0, 47.0, 99.0, 99.0, 99.0, 99.0,
    18.0, 21.0, 26.0, 66.0, 99.0, 99.0, 99.0, 99.0,
    24.0, 26.0, 56.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    47.0, 66.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
    99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0,
];

/// Rescaling factors for 4:2:0 chroma subsampling.
#[rustfmt::skip]
pub const RESCALE_420: [f32; 64] = [
    0.4093, 0.3209, 0.3477, 0.3333, 0.3144, 0.2823, 0.3214, 0.3354,
    0.3209, 0.3111, 0.3489, 0.2801, 0.3059, 0.3119, 0.4135, 0.3445,
    0.3477, 0.3489, 0.3586, 0.3257, 0.2727, 0.3754, 0.3369, 0.3484,
    0.3333, 0.2801, 0.3257, 0.3020, 0.3515, 0.3410, 0.3971, 0.3839,
    0.3144, 0.3059, 0.2727, 0.3515, 0.3105, 0.3397, 0.2716, 0.3836,
    0.2823, 0.3119, 0.3754, 0.3410, 0.3397, 0.3212, 0.3203, 0.0726,
    0.3214, 0.4135, 0.3369, 0.3971, 0.2716, 0.3203, 0.0798, 0.0553,
    0.3354, 0.3445, 0.3484, 0.3839, 0.3836, 0.0726, 0.0553, 0.3368,
];

// =============================================================================
// Quality Conversion
// =============================================================================

/// Convert a quality value (0-100) to butteraugli distance.
///
/// This matches jpegli's `jpegli_quality_to_distance` function.
/// Lower distance = higher quality. Distance 1.0 is approximately "visually lossless".
#[must_use]
pub fn quality_to_distance(quality: i32) -> f32 {
    if quality >= 100 {
        0.01
    } else if quality >= 30 {
        0.1 + (100 - quality) as f32 * 0.09
    } else {
        let q = quality as f32;
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}

/// Convert a linear quality scale factor to butteraugli distance.
///
/// This matches jpegli's `LinearQualityToDistance` function.
#[must_use]
pub fn linear_quality_to_distance(scale_factor: i32) -> f32 {
    let scale_factor = scale_factor.clamp(0, 5000);
    let quality = if scale_factor < 100 {
        100 - scale_factor / 2
    } else {
        5000 / scale_factor
    };
    quality_to_distance(quality)
}

// =============================================================================
// XYB Color Space Constants
// =============================================================================

/// XYB opsin absorbance matrix.
/// 3x3 matrix that converts linear RGB to opsin (cone response) space.
pub mod xyb {
    /// Matrix row 0 coefficients
    pub const M00: f32 = 0.30;
    pub const M01: f32 = 0.622; // 1.0 - M00 - M02
    pub const M02: f32 = 0.078;

    /// Matrix row 1 coefficients
    pub const M10: f32 = 0.23;
    pub const M11: f32 = 0.692; // 1.0 - M10 - M12
    pub const M12: f32 = 0.078;

    /// Matrix row 2 coefficients
    pub const M20: f32 = 0.243_422_69;
    pub const M21: f32 = 0.204_767_44;
    pub const M22: f32 = 0.551_809_87; // 1.0 - M20 - M21

    /// Opsin absorbance matrix (3x3, row-major)
    pub const OPSIN_ABSORBANCE_MATRIX: [[f32; 3]; 3] =
        [[M00, M01, M02], [M10, M11, M12], [M20, M21, M22]];

    /// Opsin absorbance bias (added before cube root)
    pub const OPSIN_ABSORBANCE_BIAS: f32 = 0.003_793_073_3;
    pub const OPSIN_ABSORBANCE_BIAS_VEC: [f32; 3] = [
        OPSIN_ABSORBANCE_BIAS,
        OPSIN_ABSORBANCE_BIAS,
        OPSIN_ABSORBANCE_BIAS,
    ];

    /// Inverse opsin absorbance matrix (for XYB to linear RGB)
    pub const INVERSE_OPSIN_ABSORBANCE_MATRIX: [[f32; 3]; 3] = [
        [11.031_567, -9.866_944, -0.164_623],
        [-3.254_147_4, 4.418_770_4, -0.164_623],
        [-3.658_851_3, 2.712_923, 1.945_928_2],
    ];

    /// Negative opsin absorbance bias (for cube root offset)
    pub const NEG_OPSIN_ABSORBANCE_BIAS_RGB: [f32; 4] = [
        -OPSIN_ABSORBANCE_BIAS,
        -OPSIN_ABSORBANCE_BIAS,
        -OPSIN_ABSORBANCE_BIAS,
        1.0,
    ];
}

// =============================================================================
// DCT Constants
// =============================================================================

/// DCT scaling constants for the AAN (Arai-Agui-Nakajima) algorithm.
pub mod dct {
    /// DCT-II normalization factor for DC coefficient (1/sqrt(8))
    pub const ALPHA_DC: f32 = 0.353_553_39; // 1/sqrt(8)

    /// DCT-II normalization factor for AC coefficients
    pub const ALPHA_AC: f32 = 0.5;

    /// Compute DCT-II normalization factor for coefficient k.
    #[must_use]
    pub const fn alpha(k: usize) -> f32 {
        if k == 0 {
            ALPHA_DC
        } else {
            ALPHA_AC
        }
    }

    // Precomputed cosine values for 8-point DCT
    // cos(k * pi / 16) for k = 1..7
    pub const C1: f32 = 0.980_785_28; // cos(1 * PI / 16)
    pub const C2: f32 = 0.923_879_53; // cos(2 * PI / 16)
    pub const C3: f32 = 0.831_469_61; // cos(3 * PI / 16)
    pub const C4: f32 = 0.707_106_78; // cos(4 * PI / 16) = 1/sqrt(2)
    pub const C5: f32 = 0.555_570_23; // cos(5 * PI / 16)
    pub const C6: f32 = 0.382_683_43; // cos(6 * PI / 16)
    pub const C7: f32 = 0.195_090_32; // cos(7 * PI / 16)

    // AAN algorithm constants
    pub const S0: f32 = 0.353_553_39; // 1 / (2 * sqrt(2))
    pub const S1: f32 = 0.254_897_69;
    pub const S2: f32 = 0.270_598_05;
    pub const S3: f32 = 0.300_672_44;
    pub const S4: f32 = 0.353_553_39;
    pub const S5: f32 = 0.449_988_11;
    pub const S6: f32 = 0.653_281_48;
    pub const S7: f32 = 1.281_457_7;
}

// =============================================================================
// Color Space Constants
// =============================================================================

/// BT.601 color space conversion constants (full range).
pub mod bt601 {
    /// RGB to YCbCr conversion matrix (row-major).
    /// Y  =  0.299 * R + 0.587 * G + 0.114 * B
    /// Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
    /// Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128
    pub const RGB_TO_YCBCR: [[f32; 3]; 3] = [
        [0.299, 0.587, 0.114],
        [-0.168_736, -0.331_264, 0.5],
        [0.5, -0.418_688, -0.081_312],
    ];

    /// YCbCr to RGB conversion matrix (row-major).
    /// R = Y + 1.402 * (Cr - 128)
    /// G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
    /// B = Y + 1.772 * (Cb - 128)
    pub const YCBCR_TO_RGB: [[f32; 3]; 3] = [
        [1.0, 0.0, 1.402],
        [1.0, -0.344_136, -0.714_136],
        [1.0, 1.772, 0.0],
    ];
}

// =============================================================================
// YCbCr Color Conversion (individual constants for direct use)
// =============================================================================

// RGB to YCbCr conversion factors (BT.601)
/// R contribution to Y
pub const YCBCR_R_TO_Y: f32 = 0.299;
/// G contribution to Y
pub const YCBCR_G_TO_Y: f32 = 0.587;
/// B contribution to Y
pub const YCBCR_B_TO_Y: f32 = 0.114;

/// R contribution to Cb
pub const YCBCR_R_TO_CB: f32 = -0.168_736;
/// G contribution to Cb
pub const YCBCR_G_TO_CB: f32 = -0.331_264;
/// B contribution to Cb
pub const YCBCR_B_TO_CB: f32 = 0.5;

/// R contribution to Cr
pub const YCBCR_R_TO_CR: f32 = 0.5;
/// G contribution to Cr
pub const YCBCR_G_TO_CR: f32 = -0.418_688;
/// B contribution to Cr
pub const YCBCR_B_TO_CR: f32 = -0.081_312;

// YCbCr to RGB conversion factors (BT.601)
/// Y contribution to R
pub const YCBCR_Y_TO_R: f32 = 1.0;
/// Cb contribution to R (after subtracting 128)
pub const YCBCR_CB_TO_R: f32 = 0.0;
/// Cr contribution to R (after subtracting 128)
pub const YCBCR_CR_TO_R: f32 = 1.402;

/// Y contribution to G
pub const YCBCR_Y_TO_G: f32 = 1.0;
/// Cb contribution to G (after subtracting 128)
pub const YCBCR_CB_TO_G: f32 = -0.344_136;
/// Cr contribution to G (after subtracting 128)
pub const YCBCR_CR_TO_G: f32 = -0.714_136;

/// Y contribution to B
pub const YCBCR_Y_TO_B: f32 = 1.0;
/// Cb contribution to B (after subtracting 128)
pub const YCBCR_CB_TO_B: f32 = 1.772;
/// Cr contribution to B (after subtracting 128)
pub const YCBCR_CR_TO_B: f32 = 0.0;

// =============================================================================
// XYB Color Space (flat arrays for direct use)
// =============================================================================

/// XYB opsin absorbance matrix as flat 9-element array (row-major 3x3).
pub const XYB_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    0.30,
    0.622,
    0.078, // Row 0
    0.23,
    0.692,
    0.078, // Row 1
    0.243_422_69,
    0.204_767_44,
    0.551_809_87, // Row 2
];

/// XYB opsin absorbance bias (3-element array).
pub const XYB_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [0.003_793_073_3, 0.003_793_073_3, 0.003_793_073_3];

/// Negative of opsin absorbance bias after cube root (for XYB to RGB conversion).
/// This is -cbrt(bias) for each channel.
pub const XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT: [f32; 3] = [
    -0.155_954_12, // -cbrt(0.003_793_073_3)
    -0.155_954_12,
    -0.155_954_12,
];

// =============================================================================
// ICC Profile
// =============================================================================

/// ICC profile signature for APP2 markers
pub const ICC_PROFILE_SIGNATURE: [u8; 12] = *b"ICC_PROFILE\0";

/// Maximum ICC profile bytes per APP2 marker segment
/// (65535 max segment size - 2 length bytes - 12 signature - 2 chunk info)
pub const MAX_ICC_BYTES_PER_MARKER: usize = 65519;

/// XYB ICC Profile generated from C++ jpegli.
///
/// This ICC profile contains the inverse XYB transform so that standard
/// JPEG decoders can convert XYB-encoded values back to sRGB.
/// The profile includes:
/// - A-to-B LUT with the XYB inverse transform
/// - B-to-A LUT (identity for rendering intent)
/// - XYB colorspace description
#[rustfmt::skip]
pub const XYB_ICC_PROFILE: [u8; 720] = [
    0x00, 0x00, 0x02, 0xd0, 0x6a, 0x78, 0x6c, 0x20, 0x04, 0x40, 0x00, 0x00, 0x73, 0x63, 0x6e, 0x72,
    0x52, 0x47, 0x42, 0x20, 0x58, 0x59, 0x5a, 0x20, 0x07, 0xe3, 0x00, 0x0c, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x61, 0x63, 0x73, 0x70, 0x41, 0x50, 0x50, 0x4c, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0xd6, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2d,
    0x6a, 0x78, 0x6c, 0x20, 0x55, 0x8e, 0xdd, 0x94, 0x0f, 0x32, 0x04, 0x06, 0x99, 0xc6, 0x8a, 0x17,
    0xb4, 0x0d, 0x3f, 0x7b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x06, 0x64, 0x65, 0x73, 0x63, 0x00, 0x00, 0x00, 0xcc, 0x00, 0x00, 0x00, 0x2c,
    0x63, 0x70, 0x72, 0x74, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x00, 0x00, 0x24, 0x77, 0x74, 0x70, 0x74,
    0x00, 0x00, 0x01, 0x1c, 0x00, 0x00, 0x00, 0x14, 0x63, 0x68, 0x61, 0x64, 0x00, 0x00, 0x01, 0x30,
    0x00, 0x00, 0x00, 0x2c, 0x41, 0x32, 0x42, 0x30, 0x00, 0x00, 0x01, 0x5c, 0x00, 0x00, 0x01, 0x24,
    0x42, 0x32, 0x41, 0x30, 0x00, 0x00, 0x02, 0x80, 0x00, 0x00, 0x00, 0x50, 0x6d, 0x6c, 0x75, 0x63,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x65, 0x6e, 0x55, 0x53,
    0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x58, 0x00, 0x59, 0x00, 0x42, 0x00, 0x5f,
    0x00, 0x50, 0x00, 0x65, 0x00, 0x72, 0x00, 0x00, 0x6d, 0x6c, 0x75, 0x63, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x65, 0x6e, 0x55, 0x53, 0x00, 0x00, 0x00, 0x06,
    0x00, 0x00, 0x00, 0x1c, 0x00, 0x43, 0x00, 0x43, 0x00, 0x30, 0x00, 0x00, 0x58, 0x59, 0x5a, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf6, 0xd6, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xd3, 0x2d,
    0x73, 0x66, 0x33, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x40, 0x00, 0x00, 0x05, 0xdd,
    0xff, 0xff, 0xf3, 0x29, 0x00, 0x00, 0x07, 0x92, 0x00, 0x00, 0xfd, 0x90, 0xff, 0xff, 0xfb, 0xa2,
    0xff, 0xff, 0xfd, 0xa2, 0x00, 0x00, 0x03, 0xdb, 0x00, 0x00, 0xc0, 0x81, 0x6d, 0x41, 0x42, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0xf4,
    0x00, 0x00, 0x00, 0x94, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x20, 0x70, 0x61, 0x72, 0x61,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x70, 0x61, 0x72, 0x61,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x70, 0x61, 0x72, 0x61,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x02, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x0c, 0x86, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x86, 0x70, 0xc9, 0xf3, 0x79, 0xff, 0xff,
    0x8f, 0x36, 0xf3, 0x79, 0xff, 0xff, 0xff, 0xff, 0x0c, 0x86, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x86,
    0x00, 0x00, 0x70, 0xc9, 0xff, 0xff, 0xf3, 0x79, 0x8f, 0x36, 0xff, 0xff, 0xf3, 0x79, 0xff, 0xff,
    0x70, 0x61, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x00, 0xe3, 0x88, 0x00, 0x00, 0x23, 0xfc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x70, 0x61, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x00, 0xe3, 0x88, 0x00, 0x00, 0x20, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x70, 0x61, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x01, 0x82, 0xd3, 0xff, 0xff, 0xe0, 0xd5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0xa1,
    0x00, 0x01, 0x84, 0x5b, 0xff, 0xfe, 0xe4, 0xbb, 0x00, 0x00, 0x12, 0x56, 0xff, 0xff, 0xf3, 0x32,
    0x00, 0x00, 0x91, 0x80, 0xff, 0xff, 0xfb, 0x4e, 0xff, 0xfe, 0x9c, 0xc1, 0x00, 0x01, 0x1d, 0x54,
    0x00, 0x00, 0xaf, 0x8c, 0xff, 0xff, 0xff, 0x88, 0xff, 0xff, 0xff, 0x84, 0xff, 0xff, 0xff, 0x99,
    0x6d, 0x42, 0x41, 0x20, 0x00, 0x00, 0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x70, 0x61, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x70, 0x61, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x70, 0x61, 0x72, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_inverse() {
        // Verify that natural order and zigzag order are inverses
        for i in 0..64 {
            let zigzag_idx = JPEG_ZIGZAG_ORDER[i] as usize;
            let natural_idx = JPEG_NATURAL_ORDER[zigzag_idx] as usize;
            assert_eq!(natural_idx, i, "Zigzag inverse failed at {i}");
        }
    }

    #[test]
    fn test_quality_to_distance() {
        // Quality 100 should give very low distance
        assert!(quality_to_distance(100) < 0.1);

        // Quality 90 should give distance around 1.0
        let d90 = quality_to_distance(90);
        assert!(d90 > 0.5 && d90 < 2.0);

        // Lower quality should give higher distance
        assert!(quality_to_distance(50) > quality_to_distance(90));
        assert!(quality_to_distance(10) > quality_to_distance(50));
    }

    #[test]
    fn test_xyb_matrix_rows_sum_to_one() {
        use xyb::*;

        let row0_sum = M00 + M01 + M02;
        let row1_sum = M10 + M11 + M12;
        let row2_sum = M20 + M21 + M22;

        assert!((row0_sum - 1.0).abs() < 1e-6);
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }
}
