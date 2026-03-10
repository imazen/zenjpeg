//! Encoding table tuning for optimization experiments.
//!
//! This module provides fine-grained control over quantization and zero-bias
//! tables for researching better encoding parameters.
//!
//! # Overview
//!
//! JPEG encoding quality is controlled by:
//! - **Quantization tables**: How much to divide each DCT coefficient
//! - **Zero-bias tables**: Rounding behavior (dead zone around zero)
//! - **Scaling parameters**: How quality maps to table values
//!
//! # Table Structure
//!
//! Both YCbCr and XYB use 3 components × 64 coefficients:
//! - YCbCr: Y (luma), Cb (blue chroma), Cr (red chroma)
//! - XYB: X (red-green), Y (intensity), B (blue-yellow)
//!
//! # Quality Scaling
//!
//! The final quantization value for coefficient `k` is:
//! ```text
//! final_quant[k] = base[k] × distance_to_scale(distance, k) × global_scale
//! ```
//!
//! Where `distance_to_scale` applies per-frequency non-linear scaling
//! controlled by `frequency_exponents`.

use crate::foundation::consts::{
    BASE_QUANT_MATRIX_XYB, BASE_QUANT_MATRIX_YCBCR, GLOBAL_SCALE_XYB, GLOBAL_SCALE_YCBCR,
};
use crate::quant::{
    FREQUENCY_EXPONENT, ZERO_BIAS_MUL_XYB_LQ, ZERO_BIAS_MUL_YCBCR_HQ, ZERO_BIAS_MUL_YCBCR_LQ,
    ZERO_BIAS_OFFSET_XYB_AC, ZERO_BIAS_OFFSET_YCBCR_AC, ZERO_BIAS_OFFSET_YCBCR_DC,
};

/// Per-component data with type aliases for each color space.
#[derive(Clone, Debug, PartialEq)]
pub struct PerComponent<T> {
    /// Component 0: Y (YCbCr) or X (XYB)
    pub c0: T,
    /// Component 1: Cb (YCbCr) or Y (XYB)
    pub c1: T,
    /// Component 2: Cr (YCbCr) or B (XYB)
    pub c2: T,
}

/// YCbCr-specific component access.
pub type YCbCrComponents<T> = PerComponent<T>;

/// XYB-specific component access.
pub type XybComponents<T> = PerComponent<T>;

impl<T> PerComponent<T> {
    /// Create from three values.
    pub fn new(c0: T, c1: T, c2: T) -> Self {
        Self { c0, c1, c2 }
    }

    /// Map a function over all components.
    pub fn map<U, F: Fn(&T) -> U>(&self, f: F) -> PerComponent<U> {
        PerComponent {
            c0: f(&self.c0),
            c1: f(&self.c1),
            c2: f(&self.c2),
        }
    }

    /// Get component by index (0, 1, or 2).
    pub fn get(&self, index: usize) -> &T {
        match index {
            0 => &self.c0,
            1 => &self.c1,
            _ => &self.c2,
        }
    }

    /// Get mutable component by index.
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.c0,
            1 => &mut self.c1,
            _ => &mut self.c2,
        }
    }
}

// YCbCr-specific accessors
impl<T> YCbCrComponents<T> {
    /// Luma component (Y).
    pub fn y(&self) -> &T {
        &self.c0
    }
    /// Blue chroma component (Cb).
    pub fn cb(&self) -> &T {
        &self.c1
    }
    /// Red chroma component (Cr).
    pub fn cr(&self) -> &T {
        &self.c2
    }
    /// Mutable luma component.
    pub fn y_mut(&mut self) -> &mut T {
        &mut self.c0
    }
    /// Mutable blue chroma component.
    pub fn cb_mut(&mut self) -> &mut T {
        &mut self.c1
    }
    /// Mutable red chroma component.
    pub fn cr_mut(&mut self) -> &mut T {
        &mut self.c2
    }
}

// XYB-specific accessors
impl<T> XybComponents<T> {
    /// X component (red-green).
    pub fn x(&self) -> &T {
        &self.c0
    }
    /// Y component (intensity/luma).
    pub fn luma(&self) -> &T {
        &self.c1
    }
    /// B component (blue-yellow).
    pub fn b(&self) -> &T {
        &self.c2
    }
    /// Mutable X component.
    pub fn x_mut(&mut self) -> &mut T {
        &mut self.c0
    }
    /// Mutable Y component.
    pub fn luma_mut(&mut self) -> &mut T {
        &mut self.c1
    }
    /// Mutable B component.
    pub fn b_mut(&mut self) -> &mut T {
        &mut self.c2
    }
}

impl<T: Copy> PerComponent<[T; 64]> {
    /// Get a coefficient from a specific component.
    pub fn coeff(&self, component: usize, k: usize) -> T {
        self.get(component)[k]
    }

    /// Set a coefficient in a specific component.
    pub fn set_coeff(&mut self, component: usize, k: usize, value: T) {
        self.get_mut(component)[k] = value;
    }
}

impl PerComponent<[f32; 64]> {
    /// Scale a single coefficient.
    pub fn scale_coeff(&mut self, component: usize, k: usize, factor: f32) {
        self.get_mut(component)[k] *= factor;
    }

    /// Scale all coefficients in a component.
    pub fn scale_component(&mut self, component: usize, factor: f32) {
        for v in self.get_mut(component).iter_mut() {
            *v *= factor;
        }
    }

    /// Scale all coefficients in all components.
    pub fn scale_all(&mut self, factor: f32) {
        for c in 0..3 {
            self.scale_component(c, factor);
        }
    }

    /// Linear blend: self * (1-t) + other * t
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let blend_arr = |a: &[f32; 64], b: &[f32; 64]| {
            let mut result = [0.0f32; 64];
            for i in 0..64 {
                result[i] = a[i] * (1.0 - t) + b[i] * t;
            }
            result
        };
        Self {
            c0: blend_arr(&self.c0, &other.c0),
            c1: blend_arr(&self.c1, &other.c1),
            c2: blend_arr(&self.c2, &other.c2),
        }
    }
}

/// Quality scaling parameters.
///
/// Controls how the quality setting maps to final quantization values.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalingParams {
    /// No scaling - quant values are used directly (must be valid u16).
    ///
    /// Use this when you want exact control over final quantization tables.
    Exact,

    /// Scale by quality using these parameters.
    ///
    /// Final value = `base[k] × distance_to_scale(distance, k) × global_scale`
    Scaled {
        /// Global multiplier applied to all coefficients.
        /// Default: 1.74 for YCbCr, 1.44 for XYB
        global_scale: f32,

        /// Per-frequency exponents for non-linear scaling.
        ///
        /// - `1.0` = linear scaling (quality maps directly)
        /// - `< 1.0` = compressive (quality drops slower at low quality)
        /// - `> 1.0` = expansive (quality drops faster at low quality)
        ///
        /// Indexed by coefficient position (0-63, row-major order).
        frequency_exponents: Box<[f32; 64]>,
    },
}

impl ScalingParams {
    /// Default scaling for YCbCr mode.
    #[must_use]
    pub fn default_ycbcr() -> Self {
        Self::Scaled {
            global_scale: GLOBAL_SCALE_YCBCR,
            frequency_exponents: Box::new(FREQUENCY_EXPONENT),
        }
    }

    /// Default scaling for XYB mode.
    #[must_use]
    pub fn default_xyb() -> Self {
        Self::Scaled {
            global_scale: GLOBAL_SCALE_XYB,
            frequency_exponents: Box::new(FREQUENCY_EXPONENT),
        }
    }

    /// No scaling - use exact values.
    #[must_use]
    pub fn exact() -> Self {
        Self::Exact
    }
}

/// Encoding tables with full per-coefficient control.
///
/// Use this for experimenting with quantization and zero-bias parameters.
/// Works for both YCbCr and XYB modes (same structure, different semantics).
#[derive(Clone, Debug, PartialEq)]
pub struct EncodingTables {
    /// Base quantization matrix (64 coefficients per component, row-major).
    ///
    /// These are the "base" values that get scaled by quality. If using
    /// `ScalingParams::Exact`, these should be the final u16-range values.
    pub quant: PerComponent<[f32; 64]>,

    /// Zero-bias multiplier (64 coefficients per component).
    ///
    /// Controls the frequency-specific dead zone scaling.
    /// Higher values = more aggressive zeroing of small coefficients.
    pub zero_bias_mul: PerComponent<[f32; 64]>,

    /// Zero-bias offset for DC coefficient (one per component).
    pub zero_bias_offset_dc: [f32; 3],

    /// Zero-bias offset for AC coefficients (one per component).
    ///
    /// Shared across all 63 AC coefficients within each component.
    pub zero_bias_offset_ac: [f32; 3],

    /// Quality scaling parameters.
    pub scaling: ScalingParams,
}

impl EncodingTables {
    /// Default tables for YCbCr encoding.
    ///
    /// Uses jpegli's perceptual quantization tables with quality-adaptive
    /// zero-bias that blends between HQ and LQ tables.
    #[must_use]
    pub fn default_ycbcr() -> Self {
        // Extract per-component quant tables
        let quant = PerComponent {
            c0: std::array::from_fn(|i| BASE_QUANT_MATRIX_YCBCR[i]),
            c1: std::array::from_fn(|i| BASE_QUANT_MATRIX_YCBCR[64 + i]),
            c2: std::array::from_fn(|i| BASE_QUANT_MATRIX_YCBCR[128 + i]),
        };

        // Use LQ tables as default (HQ tables can be blended at runtime)
        let zero_bias_mul = PerComponent {
            c0: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_LQ[i]),
            c1: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_LQ[64 + i]),
            c2: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_LQ[128 + i]),
        };

        Self {
            quant,
            zero_bias_mul,
            zero_bias_offset_dc: ZERO_BIAS_OFFSET_YCBCR_DC,
            zero_bias_offset_ac: ZERO_BIAS_OFFSET_YCBCR_AC,
            scaling: ScalingParams::default_ycbcr(),
        }
    }

    /// Default tables for XYB encoding.
    ///
    /// Uses v3 frequency-dependent, per-component zero-bias tables with LQ
    /// values as the default (HQ tables blended at runtime via quality-adaptive
    /// `ZeroBiasParams::for_xyb(distance, component)`). This matches the YCbCr
    /// pattern of storing LQ tables as the default.
    ///
    /// +0.76 SSIM2 at Q75, +0.68 at Q85 vs the old flat 0.5 baseline.
    #[must_use]
    pub fn default_xyb() -> Self {
        // Extract per-component quant tables
        let quant = PerComponent {
            c0: std::array::from_fn(|i| BASE_QUANT_MATRIX_XYB[i]),
            c1: std::array::from_fn(|i| BASE_QUANT_MATRIX_XYB[64 + i]),
            c2: std::array::from_fn(|i| BASE_QUANT_MATRIX_XYB[128 + i]),
        };

        // v3 frequency-dependent tables (LQ as default, matches YCbCr pattern)
        let zero_bias_mul = PerComponent {
            c0: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_LQ[i]),
            c1: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_LQ[64 + i]),
            c2: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_LQ[128 + i]),
        };

        Self {
            quant,
            zero_bias_mul,
            zero_bias_offset_dc: [0.0, 0.0, 0.0],
            zero_bias_offset_ac: ZERO_BIAS_OFFSET_XYB_AC,
            scaling: ScalingParams::default_xyb(),
        }
    }

    /// Get the HQ zero-bias multiplier tables for YCbCr.
    ///
    /// Use this with `blend_zero_bias_mul` for quality-adaptive blending.
    #[must_use]
    pub fn ycbcr_hq_zero_bias_mul() -> PerComponent<[f32; 64]> {
        PerComponent {
            c0: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_HQ[i]),
            c1: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_HQ[64 + i]),
            c2: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_HQ[128 + i]),
        }
    }

    /// Get the LQ zero-bias multiplier tables for YCbCr.
    #[must_use]
    pub fn ycbcr_lq_zero_bias_mul() -> PerComponent<[f32; 64]> {
        PerComponent {
            c0: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_LQ[i]),
            c1: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_LQ[64 + i]),
            c2: std::array::from_fn(|i| ZERO_BIAS_MUL_YCBCR_LQ[128 + i]),
        }
    }

    /// Get the HQ zero-bias multiplier tables for XYB (v3).
    #[must_use]
    pub fn xyb_hq_zero_bias_mul() -> PerComponent<[f32; 64]> {
        use crate::quant::ZERO_BIAS_MUL_XYB_HQ;
        PerComponent {
            c0: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_HQ[i]),
            c1: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_HQ[64 + i]),
            c2: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_HQ[128 + i]),
        }
    }

    /// Get the LQ zero-bias multiplier tables for XYB (v3).
    #[must_use]
    pub fn xyb_lq_zero_bias_mul() -> PerComponent<[f32; 64]> {
        PerComponent {
            c0: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_LQ[i]),
            c1: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_LQ[64 + i]),
            c2: std::array::from_fn(|i| ZERO_BIAS_MUL_XYB_LQ[128 + i]),
        }
    }

    /// Blend zero-bias mul tables based on quality.
    ///
    /// `t = 0.0` → LQ tables (lower quality, more aggressive zeroing)
    /// `t = 1.0` → HQ tables (higher quality, preserve more detail)
    pub fn blend_zero_bias_mul(&mut self, hq: &PerComponent<[f32; 64]>, t: f32) {
        let lq = Self::ycbcr_lq_zero_bias_mul();
        self.zero_bias_mul = lq.blend(hq, t);
    }

    // === Manipulation helpers ===

    /// Scale a single quant coefficient.
    pub fn scale_quant(&mut self, component: usize, k: usize, factor: f32) {
        self.quant.scale_coeff(component, k, factor);
    }

    /// Scale a single zero-bias mul coefficient.
    pub fn scale_mul(&mut self, component: usize, k: usize, factor: f32) {
        self.zero_bias_mul.scale_coeff(component, k, factor);
    }

    /// Perturb a quant coefficient by adding delta.
    pub fn perturb_quant(&mut self, component: usize, k: usize, delta: f32) {
        self.quant.get_mut(component)[k] += delta;
    }

    /// Perturb a zero-bias mul coefficient by adding delta.
    pub fn perturb_mul(&mut self, component: usize, k: usize, delta: f32) {
        self.zero_bias_mul.get_mut(component)[k] += delta;
    }

    /// Linear blend between two table sets.
    #[must_use]
    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let blend_3 = |a: &[f32; 3], b: &[f32; 3]| {
            [
                a[0] * (1.0 - t) + b[0] * t,
                a[1] * (1.0 - t) + b[1] * t,
                a[2] * (1.0 - t) + b[2] * t,
            ]
        };

        Self {
            quant: self.quant.blend(&other.quant, t),
            zero_bias_mul: self.zero_bias_mul.blend(&other.zero_bias_mul, t),
            zero_bias_offset_dc: blend_3(&self.zero_bias_offset_dc, &other.zero_bias_offset_dc),
            zero_bias_offset_ac: blend_3(&self.zero_bias_offset_ac, &other.zero_bias_offset_ac),
            // Use self's scaling (caller should handle if different)
            scaling: self.scaling.clone(),
        }
    }

    // === Quant table and zero-bias generation ===

    /// Generate a quantization table for a single component.
    ///
    /// # Arguments
    /// * `component` - 0 for Y/X, 1 for Cb/Y, 2 for Cr/B
    /// * `distance` - Butteraugli distance (quality parameter)
    /// * `is_420` - Whether 4:2:0 chroma subsampling is used (applies quality compensation)
    #[must_use]
    pub fn generate_quant_table(
        &self,
        component: usize,
        distance: f32,
        is_420: bool,
    ) -> crate::quant::QuantTable {
        use crate::quant::create_quant_table;

        let c = component.min(2);
        let base = self.quant.get(c);

        match &self.scaling {
            ScalingParams::Exact => {
                // Use values directly without scaling
                let mut values = [0u16; 64];
                for i in 0..64 {
                    values[i] = base[i].round().clamp(1.0, 65535.0) as u16;
                }
                create_quant_table(values, true)
            }
            ScalingParams::Scaled {
                global_scale,
                frequency_exponents,
            } => Self::compute_scaled_quant(
                base,
                *global_scale,
                frequency_exponents,
                distance,
                is_420,
                component,
            ),
        }
    }

    /// Scaling logic for the Scaled variant.
    fn compute_scaled_quant(
        base: &[f32; 64],
        global_scale: f32,
        frequency_exponents: &[f32; 64],
        distance: f32,
        is_420: bool,
        component: usize,
    ) -> crate::quant::QuantTable {
        use crate::foundation::consts::{GLOBAL_SCALE_420, K420_RESCALE};
        use crate::quant::{DIST_THRESHOLD, create_quant_table};

        let mut values = [0u16; 64];
        let mut scale_factor = global_scale;

        if is_420 {
            scale_factor *= GLOBAL_SCALE_420;
        }

        let is_chroma_420 = is_420 && component > 0;

        for i in 0..64 {
            let freq_scale = if distance < DIST_THRESHOLD {
                distance
            } else {
                let exp = frequency_exponents[i];
                let mul = DIST_THRESHOLD.powf(1.0 - exp);
                (0.5 * distance).max(mul * distance.powf(exp))
            };

            let mut scale = freq_scale * scale_factor;

            if is_chroma_420 {
                scale *= K420_RESCALE[i];
            }

            let q = (base[i] * scale).round();
            values[i] = q as u16;
        }

        create_quant_table(values, true)
    }

    /// Generate quantization tables for all three components.
    ///
    /// Returns (Y/X, Cb/Y, Cr/B) quantization tables.
    #[must_use]
    pub fn generate_quant_tables(
        &self,
        distance: f32,
        is_420: bool,
    ) -> (
        crate::quant::QuantTable,
        crate::quant::QuantTable,
        crate::quant::QuantTable,
    ) {
        (
            self.generate_quant_table(0, distance, is_420),
            self.generate_quant_table(1, distance, is_420),
            self.generate_quant_table(2, distance, is_420),
        )
    }

    /// Generate zero-bias parameters for a single component.
    ///
    /// # Arguments
    /// * `component` - 0 for Y/X, 1 for Cb/Y, 2 for Cr/B
    #[must_use]
    pub fn generate_zero_bias_params(&self, component: usize) -> crate::quant::ZeroBiasParams {
        let c = component.min(2);
        let mul = *self.zero_bias_mul.get(c);
        let mut offset = [0.0f32; 64];
        offset[0] = self.zero_bias_offset_dc[c];
        for i in 1..64 {
            offset[i] = self.zero_bias_offset_ac[c];
        }

        crate::quant::ZeroBiasParams { mul, offset }
    }

    /// Generate zero-bias parameters for all three components.
    ///
    /// Returns (Y/X, Cb/Y, Cr/B) zero-bias parameters.
    #[must_use]
    pub fn generate_zero_bias_all(
        &self,
    ) -> (
        crate::quant::ZeroBiasParams,
        crate::quant::ZeroBiasParams,
        crate::quant::ZeroBiasParams,
    ) {
        (
            self.generate_zero_bias_params(0),
            self.generate_zero_bias_params(1),
            self.generate_zero_bias_params(2),
        )
    }

    /// Check if this uses exact (unscaled) tables.
    #[must_use]
    pub fn is_exact(&self) -> bool {
        matches!(self.scaling, ScalingParams::Exact)
    }
}

/// DCT coefficient utilities.
pub mod dct {
    /// Frequency "distance" from DC (Manhattan distance in 8x8 grid).
    ///
    /// DC (top-left) = 0, bottom-right = 14.
    #[inline]
    #[must_use]
    pub const fn freq_distance(k: usize) -> usize {
        let row = k / 8;
        let col = k % 8;
        row + col
    }

    /// Row and column in 8x8 block.
    #[inline]
    #[must_use]
    pub const fn row_col(k: usize) -> (usize, usize) {
        (k / 8, k % 8)
    }

    /// Convert row-major index to zigzag order.
    #[must_use]
    pub const fn to_zigzag(k: usize) -> usize {
        ZIGZAG_ORDER[k]
    }

    /// Convert zigzag order to row-major index.
    #[must_use]
    pub const fn from_zigzag(z: usize) -> usize {
        INVERSE_ZIGZAG[z]
    }

    /// Coefficients sorted by approximate perceptual importance.
    ///
    /// DC first, then low frequencies (diagonals), then high frequencies.
    /// Useful for prioritizing search over the most impactful coefficients.
    pub const IMPORTANCE_ORDER: [usize; 64] = [
        0, // DC - most important
        1, 8, // First diagonal (AC01, AC10)
        16, 9, 2, // Second diagonal
        3, 10, 17, 24, // Third diagonal
        32, 25, 18, 11, 4, // Fourth diagonal
        5, 12, 19, 26, 33, 40, // Fifth diagonal
        48, 41, 34, 27, 20, 13, 6, // Sixth diagonal
        7, 14, 21, 28, 35, 42, 49, 56, // Seventh diagonal
        57, 50, 43, 36, 29, 22, 15, // Eighth diagonal
        23, 30, 37, 44, 51, 58, // Ninth diagonal
        59, 52, 45, 38, 31, // Tenth diagonal
        39, 46, 53, 60, // Eleventh diagonal
        61, 54, 47, // Twelfth diagonal
        55, 62, // Thirteenth diagonal
        63, // Last - least important
    ];

    /// Zigzag scan order (row-major index → zigzag position).
    const ZIGZAG_ORDER: [usize; 64] = [
        0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9,
        11, 18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
    ];

    /// Inverse zigzag (zigzag position → row-major index).
    const INVERSE_ZIGZAG: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_freq_distance() {
            assert_eq!(freq_distance(0), 0); // DC
            assert_eq!(freq_distance(1), 1); // First row, second col
            assert_eq!(freq_distance(8), 1); // Second row, first col
            assert_eq!(freq_distance(63), 14); // Bottom-right
        }

        #[test]
        fn test_zigzag_roundtrip() {
            for k in 0..64 {
                let z = to_zigzag(k);
                let back = from_zigzag(z);
                assert_eq!(back, k, "Roundtrip failed for k={}", k);
            }
        }

        #[test]
        fn test_importance_order_complete() {
            let mut seen = [false; 64];
            for &k in &IMPORTANCE_ORDER {
                assert!(!seen[k], "Duplicate index {} in IMPORTANCE_ORDER", k);
                seen[k] = true;
            }
            assert!(
                seen.iter().all(|&s| s),
                "Missing indices in IMPORTANCE_ORDER"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_ycbcr_tables() {
        let tables = EncodingTables::default_ycbcr();

        // Check quant tables are populated
        assert!(tables.quant.c0[0] > 0.0, "DC quant should be positive");
        assert!(tables.quant.c1[0] > 0.0);
        assert!(tables.quant.c2[0] > 0.0);

        // Check zero-bias DC is 0 for YCbCr
        assert_eq!(tables.zero_bias_offset_dc, [0.0, 0.0, 0.0]);

        // Check zero-bias AC is non-zero
        assert!(tables.zero_bias_offset_ac[0] > 0.0);
    }

    #[test]
    fn test_default_xyb_tables() {
        let tables = EncodingTables::default_xyb();

        // XYB v3 uses per-component AC offsets [X=0.50, Y=0.48, B=0.55]
        assert_eq!(tables.zero_bias_offset_ac, [0.5, 0.48, 0.55]);

        // DC offset is 0
        assert_eq!(tables.zero_bias_offset_dc, [0.0, 0.0, 0.0]);

        // Zero-bias mul DC should be 0
        assert_eq!(tables.zero_bias_mul.c0[0], 0.0);
        assert_eq!(tables.zero_bias_mul.c1[0], 0.0);
        assert_eq!(tables.zero_bias_mul.c2[0], 0.0);

        // v3: Zero-bias mul AC is frequency-dependent (LQ defaults), not flat 0.5
        assert!(tables.zero_bias_mul.c0[1] > 0.0);
        assert!(tables.zero_bias_mul.c0[1] < 1.0);
    }

    #[test]
    fn test_blend() {
        let a = EncodingTables::default_ycbcr();
        let b = EncodingTables::default_xyb();

        let mid = a.blend(&b, 0.5);

        // Mid should be between a and b
        let a_dc = a.quant.c0[0];
        let b_dc = b.quant.c0[0];
        let mid_dc = mid.quant.c0[0];

        assert!(
            (mid_dc - (a_dc + b_dc) / 2.0).abs() < 0.01,
            "Blend should be midpoint"
        );
    }

    #[test]
    fn test_scale_quant() {
        let mut tables = EncodingTables::default_ycbcr();
        let original = tables.quant.c0[5];

        tables.scale_quant(0, 5, 2.0);

        assert!((tables.quant.c0[5] - original * 2.0).abs() < 0.001);
    }

    #[test]
    fn test_ycbcr_accessors() {
        let tables = EncodingTables::default_ycbcr();

        // Type alias accessors should work
        let y = tables.quant.y();
        let cb = tables.quant.cb();
        let cr = tables.quant.cr();

        assert_eq!(y, &tables.quant.c0);
        assert_eq!(cb, &tables.quant.c1);
        assert_eq!(cr, &tables.quant.c2);
    }

    #[test]
    fn test_xyb_accessors() {
        let tables = EncodingTables::default_xyb();

        // XYB accessors
        let x = tables.quant.x();
        let luma = tables.quant.luma();
        let b = tables.quant.b();

        assert_eq!(x, &tables.quant.c0);
        assert_eq!(luma, &tables.quant.c1);
        assert_eq!(b, &tables.quant.c2);
    }

    #[test]
    fn test_optimized_scaling_params() {
        use crate::quant::{OPTIMIZED_FREQUENCY_EXPONENT, OPTIMIZED_GLOBAL_SCALE};

        let optimized = ScalingParams::Scaled {
            global_scale: OPTIMIZED_GLOBAL_SCALE,
            frequency_exponents: Box::new(OPTIMIZED_FREQUENCY_EXPONENT),
        };
        let default = ScalingParams::default_ycbcr();

        match (&optimized, &default) {
            (
                ScalingParams::Scaled {
                    global_scale: opt_scale,
                    ..
                },
                ScalingParams::Scaled {
                    global_scale: def_scale,
                    ..
                },
            ) => {
                assert!(
                    *opt_scale > *def_scale * 2.5,
                    "Optimized scale ({}) should be > 2.5x default ({})",
                    opt_scale,
                    def_scale
                );
            }
            _ => panic!("Expected Scaled variants"),
        }
    }

    #[test]
    fn test_optimized_generates_valid_quant_tables() {
        use crate::quant::{OPTIMIZED_FREQUENCY_EXPONENT, OPTIMIZED_GLOBAL_SCALE};

        let mut tables = EncodingTables::default_ycbcr();
        tables.scaling = ScalingParams::Scaled {
            global_scale: OPTIMIZED_GLOBAL_SCALE,
            frequency_exponents: Box::new(OPTIMIZED_FREQUENCY_EXPONENT),
        };

        for q in [50, 75, 90] {
            let distance = (100.0 - q as f32) / 10.0;
            let (y, cb, cr) = tables.generate_quant_tables(distance, false);

            assert!(y.values.iter().all(|&v| v > 0), "Y quant at q{q} has zeros");
            assert!(
                cb.values.iter().all(|&v| v > 0),
                "Cb quant at q{q} has zeros"
            );
            assert!(
                cr.values.iter().all(|&v| v > 0),
                "Cr quant at q{q} has zeros"
            );
            assert!(
                y.values[0] < 1000,
                "Y DC quant at q{q} too high: {}",
                y.values[0]
            );
        }
    }
}
