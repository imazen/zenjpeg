//! Content-aware deblocking classification.
//!
//! Classifies JPEG content as photo, screenshot, or mixed to decide whether
//! deblocking filters should be applied. Screenshots with flat UI regions
//! are harmed by deblocking at Q10+, while photos always benefit.
//!
//! Two classification tiers:
//! - **Header-only** (`classify_from_probe`): uses subsampling as a soft signal
//! - **Coefficient-level** (`classify_from_luma_coefficients`): zero-AC-block fraction

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
    /// Apply H.264-style 4-tap boundary filter.
    Boundary4Tap,
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
/// Calibration notes:
/// - Photos at Q10+: typically <5% zero-AC blocks
/// - Screenshots at Q10+: typically >15% zero-AC blocks
/// - Threshold set conservatively to avoid false positives on photos.
///   Better to accidentally deblock a screenshot (mild harm) than skip
///   a photo (lost +9 SS2 at Q5).
const SCREENSHOT_ZERO_AC_THRESHOLD: f32 = 0.10;

/// Classify content from header-only probe data.
///
/// This is a weak signal: 4:4:4 subsampling in non-cjpegli encoders is a
/// screenshot hint (photo encoders default to 4:2:0). Returns `Mixed` when
/// uncertain.
pub fn classify_from_probe(probe: &JpegProbe) -> ContentType {
    // cjpegli can use 4:4:4 for both photos and screenshots — not informative
    if matches!(
        probe.encoder,
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb
    ) {
        return ContentType::Mixed;
    }

    // Non-cjpegli 4:4:4 is a screenshot hint (turbo/mozjpeg default to 4:2:0 for photos)
    if probe.subsampling == Subsampling::S444 {
        return ContentType::Screenshot;
    }

    ContentType::Mixed
}

/// Classify content from luma DCT coefficients.
///
/// Computes the fraction of luma blocks where all 63 AC coefficients are zero
/// (pure DC blocks). Screenshots have many flat UI regions producing all-DC
/// blocks; photos almost never do.
///
/// # Arguments
/// - `luma_coeffs`: Luma component coefficients in zigzag order, 64 per block
/// - `num_blocks`: Total number of luma blocks (blocks_wide * blocks_high)
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
        let all_ac_zero = block[1..64].iter().all(|&c| c == 0);
        if all_ac_zero {
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

/// Recommend deblocking action based on probe data and content classification.
///
/// Decision rules (from experimental data on 75 images):
///
/// | Encoder | Content | Quality | Action |
/// |---------|---------|---------|--------|
/// | cjpegli | any | any | boundary_4tap (always helps) |
/// | turbo/mozjpeg | photo | any | boundary_4tap |
/// | turbo/mozjpeg | screenshot | Q5 | boundary_4tap (marginal) |
/// | turbo/mozjpeg | screenshot | Q10+ | skip (severe harm) |
/// | turbo/mozjpeg | mixed | any | boundary_4tap |
pub fn recommend_deblock(
    probe: &JpegProbe,
    content: ContentType,
    zero_ac_frac: f32,
) -> DeblockRecommendation {
    let is_cjpegli = matches!(
        probe.encoder,
        EncoderFamily::CjpegliYcbcr | EncoderFamily::CjpegliXyb
    );

    // cjpegli input always benefits from deblocking
    if is_cjpegli {
        return DeblockRecommendation {
            action: DeblockAction::Boundary4Tap,
            content_type: content,
            zero_ac_frac,
        };
    }

    // For turbo/mozjpeg: skip deblocking on screenshots at Q10+
    if content == ContentType::Screenshot {
        // Estimate quality from DC quant value
        let dc_quant = probe.dqt_tables.first().map(|t| t.values[0]).unwrap_or(1);

        // DC quant <= 20 corresponds roughly to Q10+ (lower quant = higher quality)
        // At Q5, DC quant is typically 32-40 for IJG encoders
        let is_low_quality = dc_quant >= 25;

        if is_low_quality {
            // Q5 equivalent — marginal benefit, allow deblocking
            return DeblockRecommendation {
                action: DeblockAction::Boundary4Tap,
                content_type: content,
                zero_ac_frac,
            };
        }

        // Q10+ screenshot — skip deblocking (severe harm up to -36.7 SS2)
        return DeblockRecommendation {
            action: DeblockAction::Skip,
            content_type: content,
            zero_ac_frac,
        };
    }

    // Photos and mixed content: always deblock
    DeblockRecommendation {
        action: DeblockAction::Boundary4Tap,
        content_type: content,
        zero_ac_frac,
    }
}

/// Estimate blocking severity from DC quantization step.
///
/// Returns a value roughly proportional to expected blocking artifact strength.
/// Higher values mean more severe blocking (lower quality).
pub fn estimate_blocking_severity(probe: &JpegProbe) -> f32 {
    let dc_quant = probe
        .dqt_tables
        .first()
        .map(|t| t.values[0] as f32)
        .unwrap_or(1.0);

    // Normalize: DC quant of 1 (Q100) → ~0, DC quant of 40 (Q5) → ~1.0
    (dc_quant / 40.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_ac_classification_photo() {
        // Simulate photo: few zero-AC blocks
        let num_blocks = 100;
        let mut coeffs = vec![0i16; num_blocks * 64];
        // Fill most blocks with some AC content
        for bi in 0..95 {
            coeffs[bi * 64 + 1] = 5; // At least one nonzero AC
        }
        // 5 blocks all-zero AC
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Photo);
        assert!((frac - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_zero_ac_classification_screenshot() {
        // Simulate screenshot: many zero-AC blocks
        let num_blocks = 100;
        let mut coeffs = vec![0i16; num_blocks * 64];
        // Only 10 blocks have AC content
        for bi in 0..10 {
            coeffs[bi * 64 + 1] = 5;
        }
        // 90 blocks all-zero AC
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Screenshot);
        assert!((frac - 0.90).abs() < 0.001);
    }

    #[test]
    fn test_zero_ac_threshold_boundary() {
        let num_blocks = 100;
        let mut coeffs = vec![0i16; num_blocks * 64];
        // 9 blocks zero-AC (below threshold)
        for bi in 0..91 {
            coeffs[bi * 64 + 1] = 1;
        }
        let (ct, frac) = classify_from_luma_coefficients(&coeffs, num_blocks);
        assert_eq!(ct, ContentType::Photo);
        assert!((frac - 0.09).abs() < 0.001);

        // 10 blocks zero-AC (at threshold)
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
