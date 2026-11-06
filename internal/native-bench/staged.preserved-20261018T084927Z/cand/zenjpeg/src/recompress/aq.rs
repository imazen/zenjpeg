//! Activity-quantization block classifier.
//!
//! Per-block decision: which AC coefficients to force to zero?
//!
//! # The decision matrix
//!
//! For each 8×8 luma block, we measure the energy in three zigzag
//! bands:
//!
//! - **Low** (AC indices 1..16): low-frequency content (gradients,
//!   smooth tonal shifts, edges in the dominant orientation).
//! - **Mid** (AC indices 16..32): mid-frequency content (texture,
//!   fine edges, oriented details).
//! - **High** (AC indices 32..64): high-frequency content (textures,
//!   noise, sharp edges).
//!
//! Each block is classified into one of four tiers based on the ratio
//! of high-band to low-band energy:
//!
//! | Tier | Ratio (high/low) | AC zero-bias begins at |
//! |---|---|---|
//! | `VeryFlat` | ≤ 5 % | index 16 — zero **half** the AC |
//! | `Flat` | ≤ 15 % | index 32 — zero the upper quarter |
//! | `MidDetail` | ≤ 40 % | index 48 — zero just the corner |
//! | `Detailed` | > 40 % | nothing — **block is untouched** |
//!
//! The thresholds are tuned for natural photographic content. Very
//! flat blocks (sky, walls, skin gradients) tolerate aggressive zeroing
//! without visible quality loss; detailed blocks (edges, text, fine
//! texture) keep all their AC coefficients.
//!
//! Returns a per-luma-block `u64` mask; bit `i` is set iff AC index `i`
//! should be forced to zero in the emitted JPEG.
//!
//! # When does this help vs hurt?
//!
//! The 10-reference ablation
//! (`benchmarks/aq_ablation_10refs_2026-05-28.tsv`) measured AQ as a
//! **consistent size-for-quality trade**: ~3-5 % smaller output for
//! ~0.5-1.7 zensim-A lower quality, across every (source-q, target)
//! cell. It is *not* a free lunch — zeroing high-AC always removes
//! some signal.
//!
//! Therefore AQ is **only beneficial when there is quality headroom**:
//! when calibration projects the recompressed output will land *above*
//! the target, the slack can be spent on size. When the output is
//! projected to land near the target, AQ pushes it under. The
//! [`crate::recompress::strategies::preserve`] strategy gates AQ on
//! `projected_zensim_a - target_zensim_a >= 2.0` for exactly this
//! reason — measured to cut under-target delivery from 34.6 % to
//! 8.4 % on the CID22 4:2:0 sweep.
//!
//! Per-block, within an AQ-enabled image:
//! - **Flat blocks** (high/low AC energy ratio ≤ 8 %) shed their
//!   high-frequency tail — that energy was rounding noise.
//! - **Detailed blocks** (ratio > 25 %) are left fully intact.

use crate::decode::DecodedCoefficients;

use crate::recompress::strategies::preserve_emit::AqMask;

/// Activity tier for a single 8×8 luma block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityTier {
    /// AC zero-bias starts at index 16 — aggressive.
    VeryFlat,
    /// AC zero-bias starts at index 32 — moderate.
    Flat,
    /// AC zero-bias starts at index 48 — conservative.
    MidDetail,
    /// No AC zero-bias — block is left untouched.
    Detailed,
}

impl ActivityTier {
    /// First AC index to zero. `64` = no zeroing.
    ///
    /// Conservative tiers (tuned 2026-05-28): even the flattest blocks
    /// keep AC through index 32 — they only shed the highest-frequency
    /// quarter (32..64), which on a genuinely flat block is rounding
    /// noise. `MidDetail` and `Detailed` are left fully intact. This
    /// makes AQ a strictly-small, near-always-safe refinement rather
    /// than an aggressive size lever; the head-to-head A/B
    /// (benchmarks/aq_on_vs_off_*.tsv) confirmed the earlier
    /// "zero from 16" tiers cost 3-6 zensim-A for ~7% size.
    #[inline]
    pub fn zero_from(self) -> usize {
        match self {
            ActivityTier::VeryFlat => 32,
            ActivityTier::Flat => 48,
            ActivityTier::MidDetail => 64,
            ActivityTier::Detailed => 64,
        }
    }
}

// Thresholds tuned 2026-05-28 against the 4:2:0 CID22 sweep.
// Initial pass at 0.05/0.15/0.40 was too aggressive (avg_measured
// dropped 6 zensim-A, bad rate to 7.6%). Tightening so only the
// genuinely-noise-only blocks tip into VeryFlat:
const VERY_FLAT_RATIO: f32 = 0.02;
const FLAT_RATIO: f32 = 0.08;
const MID_DETAIL_RATIO: f32 = 0.25;

/// Classify a single 8×8 block (in zigzag order) into an activity
/// tier by the ratio of high-band to low-band AC energy.
pub fn classify_block(block: &[i16; 64]) -> ActivityTier {
    let low_energy: u64 = block[1..16]
        .iter()
        .map(|c| (c.unsigned_abs() as u64) * (c.unsigned_abs() as u64))
        .sum();
    let high_energy: u64 = block[32..64]
        .iter()
        .map(|c| (c.unsigned_abs() as u64) * (c.unsigned_abs() as u64))
        .sum();
    // +1 in denominator avoids div-by-zero on pure-DC blocks (those
    // get classified as VeryFlat which is the right call).
    let ratio = (high_energy as f32) / (low_energy as f32 + 1.0);
    if ratio <= VERY_FLAT_RATIO {
        ActivityTier::VeryFlat
    } else if ratio <= FLAT_RATIO {
        ActivityTier::Flat
    } else if ratio <= MID_DETAIL_RATIO {
        ActivityTier::MidDetail
    } else {
        ActivityTier::Detailed
    }
}

/// Build a per-luma-block AQ zero-bias mask. Returns `None` if there
/// is no luma component (shouldn't happen on a valid JPEG).
pub fn build_aq_mask(coeffs: &DecodedCoefficients) -> Option<AqMask> {
    let luma = coeffs.components.first()?;
    let n_blocks = luma.num_blocks();
    let mut mask: AqMask = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let bytes = luma.block(b);
        let block: &[i16; 64] = bytes
            .try_into()
            .expect("block accessor returns exactly 64 coefficients");
        let tier = classify_block(block);
        let zero_from = tier.zero_from();
        let mut m: u64 = 0;
        if zero_from < 64 {
            for i in zero_from..64 {
                m |= 1u64 << i;
            }
        }
        mask.push(m);
    }
    Some(mask)
}

/// **Masking-aligned** AQ mask: zero the high-frequency tail in
/// *busy* blocks, where surrounding texture masks the loss (the
/// classical HVS contrast-masking principle that jpegli/mozjpeg AQ
/// exploit). This is the OPPOSITE block-selection from
/// [`build_aq_mask`], which targets flat blocks.
///
/// Rationale for offering both: zeroing high-AC in a *flat* block
/// saves few bytes (a flat block has little high-freq energy to begin
/// with) and risks visible banding. Zeroing high-AC in a *busy* block
/// saves more bytes (lots of high-freq energy present) and the
/// remaining texture masks the distortion. Perceptual-coding theory
/// favours the busy-targeting variant; whether the zensim metric
/// agrees is an empirical question settled by the ablation
/// (`benchmarks/aq_direction_*.tsv`), not by assertion.
///
/// `tail_from` is the first AC zigzag index to zero in a busy block
/// (e.g. 48 = zero only the top quarter).
///
/// This and the two diagnostics below are reachable only through the
/// `recompress-expert` re-export; under plain `recompress` nothing in-crate
/// calls them, so dead-code analysis is silenced for that configuration
/// (#143). Kept unconditionally compiled so the module has one shape.
#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
pub fn build_aq_mask_busy(coeffs: &DecodedCoefficients, tail_from: usize) -> Option<AqMask> {
    let luma = coeffs.components.first()?;
    let n_blocks = luma.num_blocks();
    let mut mask: AqMask = Vec::with_capacity(n_blocks);
    let tail_from = tail_from.clamp(1, 64);
    for b in 0..n_blocks {
        let bytes = luma.block(b);
        let block: &[i16; 64] = bytes
            .try_into()
            .expect("block accessor returns exactly 64 coefficients");
        // Busy = Detailed tier (high/low AC energy ratio > MID_DETAIL_RATIO).
        let is_busy = matches!(classify_block(block), ActivityTier::Detailed);
        let mut m: u64 = 0;
        if is_busy {
            for i in tail_from..64 {
                m |= 1u64 << i;
            }
        }
        mask.push(m);
    }
    Some(mask)
}

/// Fraction of blocks the mask flags as low-activity (any non-zero
/// mask bit). Useful for sanity checks.
#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
pub fn mask_low_activity_fraction(mask: &AqMask) -> f32 {
    if mask.is_empty() {
        return 0.0;
    }
    let n_active = mask.iter().filter(|&&m| m != 0).count();
    n_active as f32 / mask.len() as f32
}

/// Histogram of activity tiers across all luma blocks. Returns counts
/// in order `[VeryFlat, Flat, MidDetail, Detailed]`. Useful for
/// diagnostics and benchmark reporting.
#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]
pub fn tier_histogram(coeffs: &DecodedCoefficients) -> [u32; 4] {
    let mut hist = [0u32; 4];
    let Some(luma) = coeffs.components.first() else {
        return hist;
    };
    for b in 0..luma.num_blocks() {
        let bytes = luma.block(b);
        let block: &[i16; 64] = bytes
            .try_into()
            .expect("block accessor returns exactly 64 coefficients");
        let idx = match classify_block(block) {
            ActivityTier::VeryFlat => 0,
            ActivityTier::Flat => 1,
            ActivityTier::MidDetail => 2,
            ActivityTier::Detailed => 3,
        };
        hist[idx] += 1;
    }
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_coeffs(blocks: Vec<[i16; 64]>) -> DecodedCoefficients {
        use crate::decode::ComponentCoefficients;
        let n = blocks.len();
        let mut flat = Vec::with_capacity(n * 64);
        for b in &blocks {
            flat.extend_from_slice(b);
        }
        let comp = ComponentCoefficients {
            id: 1,
            coeffs: flat,
            blocks_wide: n,
            blocks_high: 1,
            h_samp: 1,
            v_samp: 1,
            quant_table_idx: 0,
        };
        DecodedCoefficients {
            width: (n * 8) as u32,
            height: 8,
            components: vec![comp],
            quant_tables: vec![Some([1u16; 64])],
            huffman_tables: None,
        }
    }

    #[test]
    fn pure_dc_block_is_very_flat() {
        let mut block = [0i16; 64];
        block[0] = 100;
        assert_eq!(classify_block(&block), ActivityTier::VeryFlat);
    }

    #[test]
    fn low_band_only_is_flat_or_very_flat() {
        let mut block = [0i16; 64];
        block[0] = 100;
        for i in 1..16 {
            block[i] = 20;
        }
        // Low band carries energy, high band empty → ratio = 0/15 → VeryFlat.
        assert_eq!(classify_block(&block), ActivityTier::VeryFlat);
    }

    #[test]
    fn balanced_high_and_low_is_detailed() {
        let mut block = [0i16; 64];
        block[0] = 100;
        for i in 1..16 {
            block[i] = 20;
        }
        for i in 32..64 {
            block[i] = 30;
        }
        // high/low = 32*900 / 15*400 = 28800/6000 = 4.8 → Detailed
        assert_eq!(classify_block(&block), ActivityTier::Detailed);
    }

    #[test]
    fn mid_band_only_is_detailed() {
        let mut block = [0i16; 64];
        block[0] = 100;
        for i in 16..32 {
            block[i] = 25;
        }
        // High-band still empty → low ratio → VeryFlat (per our defn,
        // low-band excludes mid-band).
        // This documents the boundary: only ABOVE index 32 counts as
        // "high" in our classifier. Mid-band energy doesn't trigger
        // the detailed tier.
        assert_eq!(classify_block(&block), ActivityTier::VeryFlat);
    }

    #[test]
    fn high_freq_block_is_detailed() {
        let mut block = [0i16; 64];
        block[0] = 100;
        for i in 32..50 {
            block[i] = 40;
        }
        assert_eq!(classify_block(&block), ActivityTier::Detailed);
    }

    #[test]
    fn tier_histogram_counts_correctly() {
        let mut flat_block = [0i16; 64];
        flat_block[0] = 100;
        let mut detailed_block = [0i16; 64];
        detailed_block[0] = 100;
        for i in 32..50 {
            detailed_block[i] = 50;
        }
        let coeffs = fake_coeffs(vec![flat_block, flat_block, detailed_block]);
        let hist = tier_histogram(&coeffs);
        assert_eq!(hist[0], 2, "two very-flat blocks");
        assert_eq!(hist[3], 1, "one detailed block");
    }

    #[test]
    fn mask_bits_match_tier() {
        let mut very_flat = [0i16; 64];
        very_flat[0] = 100;
        let mut detailed = [0i16; 64];
        detailed[0] = 100;
        for i in 32..50 {
            detailed[i] = 50;
        }
        let coeffs = fake_coeffs(vec![very_flat, detailed]);
        let mask = build_aq_mask(&coeffs).unwrap();
        // VeryFlat: zero from 32 (conservative tier).
        for i in 32..64 {
            assert!(mask[0] & (1u64 << i) != 0, "very-flat must zero AC {i}");
        }
        for i in 0..32 {
            assert!(mask[0] & (1u64 << i) == 0, "very-flat must keep AC {i}");
        }
        // Detailed: no zeroing
        assert_eq!(mask[1], 0, "detailed block must have no AC zeroed");
    }
}
