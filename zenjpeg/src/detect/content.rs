//! Content-aware deblocking classification.
//!
//! Classifies JPEG content as photo, screenshot, or mixed, then recommends
//! a quality-adaptive deblocking strategy. Based on experiments across 75 images
//! (66 photos, 9 screenshots), 3 encoders, and 17 quality levels.
//!
//! Two classification tiers:
//! - **Header-only** (`classify_from_probe`): subsampling as soft signal
//! - **Coefficient-level** (`classify_from_luma_coefficients`): zero-AC-block fraction
//!
//! Three-tier quality-adaptive dispatch:
//! - **Low quality** (DC quant >= 27): Knusperli DCT-domain smoothing (+8.5 SS2 at Q5)
//! - **Mid-high quality** (DC quant < 27): Boundary 4-tap filter (+0.9 SS2)
//! - **Screenshots at Q10+**: Skip (deblocking causes up to -36.7 SS2 harm)

use super::{EncoderFamily, JpegProbe};
use crate::types::Subsampling;

/// Content type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Natural photograph — deblocking helps.
    Photo,
    /// Screenshot / synthetic UI — deblocking hurts at Q10+.
    Screenshot,
    /// Ambiguous or mixed content.
    Mixed,
}

/// Recommended deblocking action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeblockAction {
    /// Do not apply any deblocking filter.
    Skip,
    /// Apply H.264-style 4-tap boundary filter (good at all quality levels).
    Boundary4Tap,
    /// Apply Knusperli DCT-domain boundary correction (best at low quality).
    Knusperli,
}

/// Deblocking recommendation based on content analysis.
#[derive(Debug, Clone, Copy)]
pub struct DeblockRecommendation {
    /// What filter action to take.
    pub action: DeblockAction,
    /// Content classification that led to this recommendation.
    pub content_type: ContentType,
    /// Zero-AC-block fraction (0.0-1.0), NaN if not computed.
    pub zero_ac_frac: f32,
}

/// Fraction of luma blocks with all-zero AC coefficients above which
/// content is classified as screenshot.
///
/// Calibration (66 photos, 9 screenshots, turbo/mozjpeg/cjpegli Q5-Q97):
/// - Photos at Q10+: typically <5% zero-AC blocks
/// - Screenshots at Q10+: typically >15% zero-AC blocks
/// - Threshold biased toward false negatives: better to deblock a screenshot
///   (mild harm at Q10-Q20) than skip a photo (lost +9 SS2 at Q5).
const SCREENSHOT_ZERO_AC_THRESHOLD: f32 = 0.10;

/// DC quant threshold for Knusperli vs Boundary4Tap on turbo/mozjpeg photos.
///
/// Calibration: knusperli beats boundary_4tap when DC quant >= 27 (roughly Q30-).
/// At Q30 (DC quant=27), knusperli advantage is +0.3 SS2 (turbo) / +0.05 SS2 (mozjpeg).
/// At Q40 (DC quant=20), boundary_4tap wins by -0.3 to -0.6 SS2.
const KNUSPERLI_DC_QUANT_THRESHOLD_IJG: u16 = 27;

/// DC quant threshold for Knusperli vs Boundary4Tap on cjpegli photos.
///
/// cjpegli crosses over earlier: knusperli wins at Q15 (DC quant=53) but loses
/// at Q20 (DC quant=40). Use 40 as the threshold — at the crossover point
/// knusperli's advantage is only -0.13 SS2, so boundary_4tap is fine there.
const KNUSPERLI_DC_QUANT_THRESHOLD_CJPEGLI: u16 = 40;

/// Classify content from header-only probe data.
///
/// Weak signal: 4:4:4 subsampling in non-cjpegli encoders hints at screenshots
/// (photo encoders default to 4:2:0). Returns `Mixed` when uncertain.
pub fn classify_from_probe(probe: &JpegProbe) -> ContentType {
    // cjpegli uses 4:4:4 for both photos and screenshots — not informative
    if matches!(
        probe.encoder,
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb
    ) {
        return ContentType::Mixed;
    }

    // Non-cjpegli 4:4:4 is a screenshot hint
    if probe.subsampling == Subsampling::S444 {
        return ContentType::Screenshot;
    }

    ContentType::Mixed
}

/// Classify content from luma DCT coefficients.
///
/// Counts blocks where all 63 AC coefficients are zero (pure DC blocks).
/// Screenshots have many flat UI regions producing all-DC blocks; photos don't.
///
/// # Arguments
/// - `luma_coeffs`: Luma coefficients in zigzag order, 64 per block
/// - `num_blocks`: Total number of luma blocks
///
/// # Returns
/// `(ContentType, zero_ac_fraction)`
pub fn classify_from_luma_coefficients(
    luma_coeffs: &[i16],
    num_blocks: usize,
) -> (ContentType, f32) {
    if num_blocks == 0 {
        return (ContentType::Mixed, 0.0);
    }

    let mut zero_ac_count = 0u32;

    for bi in 0..num_blocks {
        let block = &luma_coeffs[bi * 64..(bi + 1) * 64];
        // Zigzag positions 1..64 are AC coefficients
        if block[1..64].iter().all(|&c| c == 0) {
            zero_ac_count += 1;
        }
    }

    let frac = zero_ac_count as f32 / num_blocks as f32;
    let content_type = if frac >= SCREENSHOT_ZERO_AC_THRESHOLD {
        ContentType::Screenshot
    } else {
        ContentType::Photo
    };

    (content_type, frac)
}

/// Recommend deblocking strategy based on content type, encoder, and quality.
///
/// Three-tier quality-adaptive dispatch (from 66-photo, 9-screenshot experiments):
///
/// | Encoder | Content | Quality | Action |
/// |---------|---------|---------|--------|
/// | cjpegli | photo/mixed | DC quant >= 40 | knusperli |
/// | cjpegli | photo/mixed | DC quant < 40 | boundary_4tap |
/// | cjpegli | screenshot | any | boundary_4tap (always safe) |
/// | turbo/moz | photo/mixed | DC quant >= 27 | knusperli |
/// | turbo/moz | photo/mixed | DC quant < 27 | boundary_4tap |
/// | turbo/moz | screenshot | Q5 (DC >= 25) | boundary_4tap |
/// | turbo/moz | screenshot | Q10+ (DC < 25) | skip |
pub fn recommend_deblock(
    probe: &JpegProbe,
    content: ContentType,
    zero_ac_frac: f32,
) -> DeblockRecommendation {
    let is_cjpegli = matches!(
        probe.encoder,
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb
    );

    let dc_quant = luma_dc_quant(probe);

    // --- Screenshots ---
    if content == ContentType::Screenshot {
        // cjpegli screenshots: every spatial strategy is positive, use boundary_4tap
        if is_cjpegli {
            return DeblockRecommendation {
                action: DeblockAction::Boundary4Tap,
                content_type: content,
                zero_ac_frac,
            };
        }

        // turbo/mozjpeg screenshots: only safe at very low quality
        if dc_quant >= 25 {
            // Q5 territory — marginal benefit from deblocking
            return DeblockRecommendation {
                action: DeblockAction::Boundary4Tap,
                content_type: content,
                zero_ac_frac,
            };
        }

        // Q10+ screenshot — skip (severe harm up to -36.7 SS2)
        return DeblockRecommendation {
            action: DeblockAction::Skip,
            content_type: content,
            zero_ac_frac,
        };
    }

    // --- Photos and mixed content ---
    let knusperli_threshold = if is_cjpegli {
        KNUSPERLI_DC_QUANT_THRESHOLD_CJPEGLI
    } else {
        KNUSPERLI_DC_QUANT_THRESHOLD_IJG
    };

    let action = if dc_quant >= knusperli_threshold {
        // Low quality: knusperli's aggressive DCT-domain smoothing dominates
        // turbo Q5: +14.5 vs +9.3 for boundary_4tap (+5.2 SS2 advantage)
        DeblockAction::Knusperli
    } else {
        // Mid-high quality: boundary_4tap preserves detail better
        DeblockAction::Boundary4Tap
    };

    DeblockRecommendation {
        action,
        content_type: content,
        zero_ac_frac,
    }
}

/// Estimate blocking severity from DC quantization step.
///
/// Returns a value roughly proportional to expected blocking artifact strength.
/// Higher values mean more severe blocking (lower quality).
pub fn estimate_blocking_severity(probe: &JpegProbe) -> f32 {
    let dc_quant = luma_dc_quant(probe) as f32;
    // Normalize: DC quant of 1 (Q100) → ~0, DC quant of 40 (Q5) → ~1.0
    (dc_quant / 40.0).min(1.0)
}

/// Extract luma DC quantization value from probe data.
fn luma_dc_quant(probe: &JpegProbe) -> u16 {
    probe.dqt_tables.first().map(|t| t.values[0]).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_ac_classification_photo() {
        let num_blocks = 100;
        let mut coeffs = vec![0i16; num_blocks * 64];
        for bi in 0..95 {
            coeffs[bi * 64 + 1] = 5;
        }
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Photo);
        assert!((frac - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_zero_ac_classification_screenshot() {
        let num_blocks = 100;
        let mut coeffs = vec![0i16; num_blocks * 64];
        for bi in 0..10 {
            coeffs[bi * 64 + 1] = 5;
        }
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Screenshot);
        assert!((frac - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_zero_ac_threshold_boundary() {
        let num_blocks = 100;

        // 9% zero-AC (below threshold) → Photo
        let mut coeffs = vec![0i16; num_blocks * 64];
        for bi in 0..91 {
            coeffs[bi * 64 + 1] = 1;
        }
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Photo);
        assert!((frac - 0.09).abs() < 0.001);

        // 10% zero-AC (at threshold) → Screenshot
        coeffs = vec![0i16; num_blocks * 64];
        for bi in 0..90 {
            coeffs[bi * 64 + 1] = 1;
        }
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Screenshot);
        assert!((frac - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_empty_blocks() {
        let (ct, frac) = classify_from_luma_coefficients(&[], 0);
        assert_eq!(ct, ContentType::Mixed);
        assert_eq!(frac, 0.0);
    }
}
