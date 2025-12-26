//! Tone mapping for HDR/WCG content.
//!
//! Implements Rec.2408 tone mapping, HLG OOTF, and gamut mapping.

use crate::jpegli::transfer_functions::PQ;

/// RGB color triplet
pub type Color = [f32; 3];

/// Primary luminance weights (Y coefficients)
pub type PrimariesLuminances = [f32; 3];

/// Source/target luminance range [min, max]
pub type Range = [f32; 2];

// ============================================================================
// Rec.2408 Tone Mapper
// ============================================================================

/// Rec.2408 HDR to SDR tone mapper.
///
/// Maps HDR content to SDR by applying a parametric curve that preserves
/// highlight detail while compressing the dynamic range.
#[derive(Debug, Clone)]
pub struct Rec2408ToneMapper {
    source_range: Range,
    target_range: Range,
    red_y: f32,
    green_y: f32,
    blue_y: f32,
    pq_mastering_min: f32,
    pq_mastering_range: f32,
    inv_pq_mastering_range: f32,
    min_lum: f32,
    max_lum: f32,
    ks: f32,
    inv_one_minus_ks: f32,
    normalizer: f32,
    inv_target_peak: f32,
}

impl Rec2408ToneMapper {
    /// Create a new Rec.2408 tone mapper.
    ///
    /// # Arguments
    /// * `source_range` - [min, max] luminance of source content in nits
    /// * `target_range` - [min, max] luminance of target display in nits
    /// * `luminances` - Primary luminance weights [R, G, B]
    pub fn new(source_range: Range, target_range: Range, luminances: PrimariesLuminances) -> Self {
        let pq = PQ::new(1.0); // Use 1.0 intensity target for internal calculations

        let pq_mastering_min = pq.encoded_from_display(source_range[0] as f64) as f32;
        let pq_mastering_max = pq.encoded_from_display(source_range[1] as f64) as f32;
        let pq_mastering_range = pq_mastering_max - pq_mastering_min;
        let inv_pq_mastering_range = 1.0 / pq_mastering_range;

        let min_lum = (pq.encoded_from_display(target_range[0] as f64) as f32 - pq_mastering_min)
            * inv_pq_mastering_range;
        let max_lum = (pq.encoded_from_display(target_range[1] as f64) as f32 - pq_mastering_min)
            * inv_pq_mastering_range;

        let ks = 1.5 * max_lum - 0.5;
        let inv_one_minus_ks = 1.0 / (1.0 - ks).max(1e-6);

        Self {
            source_range,
            target_range,
            red_y: luminances[0],
            green_y: luminances[1],
            blue_y: luminances[2],
            pq_mastering_min,
            pq_mastering_range,
            inv_pq_mastering_range,
            min_lum,
            max_lum,
            ks,
            inv_one_minus_ks,
            normalizer: source_range[1] / target_range[1],
            inv_target_peak: 1.0 / target_range[1],
        }
    }

    /// Apply tone mapping to an RGB color.
    pub fn tone_map(&self, rgb: &mut Color) {
        let pq = PQ::new(1.0);

        // Compute luminance in source range
        let luminance = self.source_range[1]
            * (self.red_y * rgb[0] + self.green_y * rgb[1] + self.blue_y * rgb[2]);

        // Convert to PQ and normalize
        let inv_eotf = pq.encoded_from_display(luminance as f64) as f32;
        let normalized_pq =
            ((inv_eotf - self.pq_mastering_min) * self.inv_pq_mastering_range).min(1.0);

        // Apply knee curve
        let e2 = if normalized_pq < self.ks {
            normalized_pq
        } else {
            self.p(normalized_pq)
        };

        // Apply S-curve for smooth black level
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let one_minus_e2_4 = one_minus_e2_2 * one_minus_e2_2;
        let e3 = self.min_lum * one_minus_e2_4 + e2;

        // Convert back from PQ
        let e4 = e3 * self.pq_mastering_range + self.pq_mastering_min;
        let d4 = pq.display_from_encoded(e4 as f64) as f32;
        let new_luminance = d4.clamp(0.0, self.target_range[1]);

        // Compute scaling ratio
        let min_luminance = 1e-6;
        let use_cap = luminance <= min_luminance;
        let ratio = new_luminance / luminance.max(min_luminance);
        let cap = new_luminance * self.inv_target_peak;
        let multiplier = ratio * self.normalizer;

        // Apply to RGB
        for c in rgb.iter_mut() {
            *c = if use_cap { cap } else { *c * multiplier };
        }
    }

    #[inline]
    fn t(&self, a: f32) -> f32 {
        (a - self.ks) * self.inv_one_minus_ks
    }

    #[inline]
    fn p(&self, b: f32) -> f32 {
        let t_b = self.t(b);
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        (2.0 * t_b_3 - 3.0 * t_b_2 + 1.0) * self.ks
            + (t_b_3 - 2.0 * t_b_2 + t_b) * (1.0 - self.ks)
            + (-2.0 * t_b_3 + 3.0 * t_b_2) * self.max_lum
    }
}

// ============================================================================
// HLG OOTF
// ============================================================================

/// HLG Opto-Optical Transfer Function.
///
/// Applies system gamma correction for HLG content displayed at different
/// luminance levels than the reference.
#[derive(Debug, Clone)]
pub struct HlgOotf {
    exponent: f32,
    apply_ootf: bool,
    red_y: f32,
    green_y: f32,
    blue_y: f32,
}

impl HlgOotf {
    /// Create an HLG OOTF for mapping between source and target luminances.
    ///
    /// # Arguments
    /// * `source_luminance` - Source peak luminance in nits
    /// * `target_luminance` - Target peak luminance in nits
    /// * `luminances` - Primary luminance weights [R, G, B]
    pub fn new(
        source_luminance: f32,
        target_luminance: f32,
        luminances: PrimariesLuminances,
    ) -> Self {
        let gamma = 1.111_f32.powf((target_luminance / source_luminance).log2());
        let exponent = gamma - 1.0;

        Self {
            exponent,
            apply_ootf: exponent < -0.01 || exponent > 0.01,
            red_y: luminances[0],
            green_y: luminances[1],
            blue_y: luminances[2],
        }
    }

    /// Create an HLG OOTF with explicit gamma value.
    pub fn with_gamma(gamma: f32, luminances: PrimariesLuminances) -> Self {
        let exponent = gamma - 1.0;
        Self {
            exponent,
            apply_ootf: exponent < -0.01 || exponent > 0.01,
            red_y: luminances[0],
            green_y: luminances[1],
            blue_y: luminances[2],
        }
    }

    /// Apply the OOTF to an RGB color.
    pub fn apply(&self, rgb: &mut Color) {
        if !self.apply_ootf {
            return;
        }

        let luminance = self.red_y * rgb[0] + self.green_y * rgb[1] + self.blue_y * rgb[2];
        let ratio = luminance.powf(self.exponent).min(1e9);

        rgb[0] *= ratio;
        rgb[1] *= ratio;
        rgb[2] *= ratio;
    }
}

// ============================================================================
// Gamut Mapping
// ============================================================================

/// Map out-of-gamut colors to the target gamut.
///
/// This function desaturates out-of-gamut pixels by mixing with gray of
/// the target luminance. It balances saturation preservation with luminance
/// preservation based on the `preserve_saturation` parameter.
///
/// # Arguments
/// * `rgb` - RGB color to modify (in-place)
/// * `luminances` - Primary luminance weights [R, G, B]
/// * `preserve_saturation` - Balance between saturation (0) and luminance (1) preservation
pub fn gamut_map(rgb: &mut Color, luminances: PrimariesLuminances, preserve_saturation: f32) {
    let luminance = luminances[0] * rgb[0] + luminances[1] * rgb[1] + luminances[2] * rgb[2];

    // Calculate mixing amounts for saturation and luminance preservation
    let mut gray_mix_saturation = 0.0_f32;
    let mut gray_mix_luminance = 0.0_f32;

    for &val in rgb.iter() {
        let val_minus_gray = val - luminance;
        let inv_val_minus_gray = if val_minus_gray == 0.0 {
            1.0
        } else {
            1.0 / val_minus_gray
        };
        let val_over_val_minus_gray = val * inv_val_minus_gray;

        gray_mix_saturation = if val_minus_gray >= 0.0 {
            gray_mix_saturation
        } else {
            gray_mix_saturation.max(val_over_val_minus_gray)
        };

        gray_mix_luminance = gray_mix_luminance.max(if val_minus_gray <= 0.0 {
            gray_mix_saturation
        } else {
            val_over_val_minus_gray - inv_val_minus_gray
        });
    }

    // Blend between saturation and luminance preservation
    let gray_mix = (preserve_saturation * (gray_mix_saturation - gray_mix_luminance)
        + gray_mix_luminance)
        .clamp(0.0, 1.0);

    // Apply gray mix
    for c in rgb.iter_mut() {
        *c = gray_mix * (luminance - *c) + *c;
    }

    // Normalize to [0, 1] range
    let max_clr = 1.0_f32.max(rgb[0].max(rgb[1].max(rgb[2])));
    let normalizer = 1.0 / max_clr;
    for c in rgb.iter_mut() {
        *c *= normalizer;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rec2408_tone_mapper() {
        let luminances: PrimariesLuminances = [0.3, 0.3, 0.3];
        let mapper = Rec2408ToneMapper::new([0.0, 1000.0], [0.0, 100.0], luminances);

        // Test that tone mapping produces finite values
        // Note: Output may exceed [0,1] due to normalizer scaling
        let mut rgb: Color = [0.5, 0.5, 0.5];
        mapper.tone_map(&mut rgb);
        assert!(rgb[0].is_finite());
        assert!(rgb[1].is_finite());
        assert!(rgb[2].is_finite());
    }

    #[test]
    fn test_hlg_ootf() {
        let luminances: PrimariesLuminances = [0.3, 0.3, 0.3];
        let ootf = HlgOotf::new(300.0, 100.0, luminances);

        let mut rgb: Color = [0.5, 0.5, 0.5];
        ootf.apply(&mut rgb);

        // OOTF should produce finite values
        assert!(rgb[0].is_finite());
        assert!(rgb[1].is_finite());
        assert!(rgb[2].is_finite());
    }

    #[test]
    fn test_hlg_ootf_identity() {
        let luminances: PrimariesLuminances = [0.3, 0.3, 0.3];
        // Same source and target should give gamma = 1, exponent = 0
        let ootf = HlgOotf::new(300.0, 300.0, luminances);

        let mut rgb: Color = [0.5, 0.5, 0.5];
        let original = rgb;
        ootf.apply(&mut rgb);

        // Identity transform
        assert!((rgb[0] - original[0]).abs() < 1e-6);
    }

    #[test]
    fn test_gamut_map_in_gamut() {
        let luminances: PrimariesLuminances = [0.2126, 0.7152, 0.0722];
        let mut rgb: Color = [0.5, 0.5, 0.5];
        let original = rgb;

        gamut_map(&mut rgb, luminances, 0.1);

        // In-gamut colors should not change much
        assert!((rgb[0] - original[0]).abs() < 1e-6);
        assert!((rgb[1] - original[1]).abs() < 1e-6);
        assert!((rgb[2] - original[2]).abs() < 1e-6);
    }

    #[test]
    fn test_gamut_map_out_of_gamut() {
        let luminances: PrimariesLuminances = [0.2126, 0.7152, 0.0722];
        let mut rgb: Color = [1.5, 0.5, -0.2];

        gamut_map(&mut rgb, luminances, 0.1);

        // Result should be in [0, 1] range
        assert!(rgb[0] >= 0.0 && rgb[0] <= 1.0);
        assert!(rgb[1] >= 0.0 && rgb[1] <= 1.0);
        assert!(rgb[2] >= 0.0 && rgb[2] <= 1.0);
    }
}
