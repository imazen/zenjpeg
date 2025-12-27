//! Constants for butteraugli and XYB color space.
//!
//! These values are from the libjxl C++ implementation.

// ============================================================================
// XYB Color Space Constants
// ============================================================================

/// Opsin absorbance matrix for converting linear RGB to opsin space.
/// This is an LMS-like transform (matches jpegli values).
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

/// Bias added to opsin values before cube root (matches jpegli values).
pub const XYB_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [0.003_793_073_3, 0.003_793_073_3, 0.003_793_073_3];

/// Negative cube root of opsin absorbance bias.
/// This is subtracted after the cube root operation.
pub const XYB_NEG_OPSIN_ABSORBANCE_BIAS_CBRT: [f32; 3] = [
    -0.155_954_12, // -cbrt(0.003_793_073_3)
    -0.155_954_12,
    -0.155_954_12,
];

// ============================================================================
// Butteraugli Constants
// ============================================================================

/// Malta filter weights for MF band.
pub const W_MF_MALTA: f64 = 37.0819870399;
/// Normalization for MF band.
pub const NORM1_MF: f64 = 130_262_059.556;
/// Malta filter weights for MF-X band.
pub const W_MF_MALTA_X: f64 = 8246.75321353;
/// Normalization for MF-X band.
pub const NORM1_MF_X: f64 = 1_009_002.70582;

/// Malta filter weights for HF band.
pub const W_HF_MALTA: f64 = 18.7237414387;
/// Normalization for HF band.
pub const NORM1_HF: f64 = 4_498_534.45232;
/// Malta filter weights for HF-X band.
pub const W_HF_MALTA_X: f64 = 6923.99476109;
/// Normalization for HF-X band.
pub const NORM1_HF_X: f64 = 8051.15833247;

/// Malta filter weights for UHF band.
pub const W_UHF_MALTA: f64 = 1.10039032555;
/// Normalization for UHF band.
pub const NORM1_UHF: f64 = 71.7800275169;
/// Malta filter weights for UHF-X band.
pub const W_UHF_MALTA_X: f64 = 173.5;
/// Normalization for UHF-X band.
pub const NORM1_UHF_X: f64 = 5.0;

/// Weighted multipliers for different frequency bands.
pub const WMUL: [f64; 9] = [
    400.0,
    1.50815703118,
    0.0,
    2150.0,
    10.6195433239,
    16.2176043152,
    29.2353797994,
    0.844626970982,
    0.703646627719,
];

// ============================================================================
// Blur Sigma Values
// ============================================================================

/// Sigma for LF (low frequency) blur.
pub const SIGMA_LF: f64 = 7.15593339443;
/// Sigma for HF (high frequency) blur.
pub const SIGMA_HF: f64 = 3.22489901262;
/// Sigma for UHF (ultra high frequency) blur.
pub const SIGMA_UHF: f64 = 1.56416327805;

// ============================================================================
// Masking Constants
// ============================================================================

/// Range removed around zero for MF band.
pub const REMOVE_MF_RANGE: f64 = 0.29;
/// Range added around zero for MF band.
pub const ADD_MF_RANGE: f64 = 0.1;
/// Range removed around zero for HF band.
pub const REMOVE_HF_RANGE: f64 = 1.5;
/// Range added around zero for HF band.
pub const ADD_HF_RANGE: f64 = 0.132;
/// Range removed around zero for UHF band.
pub const REMOVE_UHF_RANGE: f64 = 0.04;

/// Maximum clamp for HF band.
pub const MAXCLAMP_HF: f64 = 28.4691806922;
/// Maximum clamp for UHF band.
pub const MAXCLAMP_UHF: f64 = 5.19175294647;

/// Multiplier for Y channel in HF band.
pub const MUL_Y_HF: f64 = 2.155;
/// Multiplier for Y channel in UHF band.
pub const MUL_Y_UHF: f64 = 2.69313763794;

// ============================================================================
// LF to Vals Conversion Constants
// ============================================================================

/// X channel multiplier for LF-to-vals conversion.
pub const XMUL_LF_TO_VALS: f64 = 33.832837186260;
/// Y channel multiplier for LF-to-vals conversion.
pub const YMUL_LF_TO_VALS: f64 = 14.458268100570;
/// B channel multiplier for LF-to-vals conversion.
pub const BMUL_LF_TO_VALS: f64 = 49.87984651440;
/// Y-to-B mixing multiplier for LF-to-vals conversion.
pub const Y_TO_B_MUL_LF_TO_VALS: f64 = -0.362267051518;

// ============================================================================
// Suppression Constants
// ============================================================================

/// Suppression amount for X by Y.
pub const SUPPRESS_XY: f64 = 46.0;
/// Suppression scaling factor.
pub const SUPPRESS_S: f64 = 0.653020556257;

// ============================================================================
// Scoring Constants
// ============================================================================

/// Normalization factor for intensity target: ln(80) / ln(255).
pub const INTENSITY_TARGET_NORMALIZATION: f32 = 0.790_799_17;

/// Internal good quality threshold.
pub const INTERNAL_GOOD_QUALITY_THRESHOLD: f32 = 17.83 * INTENSITY_TARGET_NORMALIZATION;

/// Global scale factor.
pub const GLOBAL_SCALE: f32 = 1.0 / INTERNAL_GOOD_QUALITY_THRESHOLD;

// ============================================================================
// Mask Function Constants (from C++ butteraugli.cc lines 1220-1282)
// ============================================================================

/// Multiplier for DiffPrecompute.
pub const MASK_MUL: f32 = 6.19424080439;
/// Bias for DiffPrecompute.
pub const MASK_BIAS: f32 = 12.61050594197;
/// Blur radius for masking.
pub const MASK_RADIUS: f32 = 2.7;
/// Multiplier for mask-to-error in Mask function.
pub const MASK_TO_ERROR_MUL: f32 = 10.0;

/// MaskY offset constant.
pub const MASK_Y_OFFSET: f64 = 0.829591754942;
/// MaskY scaler constant.
pub const MASK_Y_SCALER: f64 = 0.451936922203;
/// MaskY mul constant.
pub const MASK_Y_MUL: f64 = 2.5485944793;

/// MaskDcY offset constant.
pub const MASK_DC_Y_OFFSET: f64 = 0.20025578522;
/// MaskDcY scaler constant.
pub const MASK_DC_Y_SCALER: f64 = 3.87449418804;
/// MaskDcY mul constant.
pub const MASK_DC_Y_MUL: f64 = 0.505054525019;

/// Multipliers for CombineChannelsForMasking.
pub const COMBINE_CHANNELS_MULS: [f32; 3] = [2.5, 0.4, 0.4];
