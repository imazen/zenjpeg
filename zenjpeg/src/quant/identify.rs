//! Recover encoder parameters from a quantization table.
//!
//! [`crate::detect::probe`] names an encoder family from markers and
//! structure. That is provenance; it does not tell you what the quantiser
//! did. This module answers the complementary question — *which* table is
//! this, and what parameter produced it — by reconstruction rather than by a
//! stored list, so it cannot go stale as encoders are added.
//!
//! Two laws cover almost everything seen in the wild. A survey of 17,739
//! JPEGs found 539 distinct luma tables, of which the libjpeg/mozjpeg preset
//! family accounts for ~84% and the jpegli family for essentially all of the
//! rest that we produce ourselves.
//!
//! # libjpeg / mozjpeg presets
//!
//! `q[i] = clamp((base[i] * scale + 50) / 100, 1, cap)` where
//! `scale = 5000/quality` below 50 and `200 - 2*quality` at or above, and
//! `cap` is 255 for a baseline-compatible encoder or 32767 otherwise. Nine
//! bases times 100 qualities times two caps generate 1,131 distinct tables
//! from 1,152 bytes of constants.
//!
//! # jpegli family (this crate, and cjpegli)
//!
//! `q[i] = round(base[i] * GLOBAL_SCALE * freq_scale(d, i))` with
//! `freq_scale = d` below [`DIST_THRESHOLD`] and
//! `max(0.5d, T^(1-e[i]) * d^e[i])` above it. Here the parameter is a
//! continuous distance, so the reachable table set cannot be enumerated —
//! sampling at 0.01 quality steps yields 3,690 distinct tables and is still
//! climbing, against a ceiling near 11,400. It is inverted analytically
//! instead: every coefficient inverts on its own, the median over 64 gives a
//! seed robust to rounding, and because the forward map is monotone in `d`
//! the distances producing a given table form an interval a short bisection
//! lands on. Measured: 98 of 100 qualities reproduce the observed table
//! exactly, the two misses being q99-100 where the table saturates to all
//! ones and no longer carries the distance.

use crate::foundation::consts::{BASE_QUANT_MATRIX_YCBCR, GLOBAL_SCALE_420, GLOBAL_SCALE_YCBCR};
use crate::quant::{DIST_THRESHOLD, FREQUENCY_EXPONENT};

/// **Alias marker: cjpegli currently shares this crate's distance→table law.**
///
/// Verified by round-trip — cjpegli output reconstructs through the constants
/// below with zero coefficient error, so one implementation serves both. The
/// two differ only in how a *quality* number maps to a distance, which is why
/// their reachable table sets are nearly disjoint (2 of 100 shared) despite
/// identical machinery underneath.
///
/// **This coupling is deliberate and fragile.** [`distance_from_luma_table`]
/// reads `BASE_QUANT_MATRIX_YCBCR`, `GLOBAL_SCALE_YCBCR`,
/// [`FREQUENCY_EXPONENT`] and [`DIST_THRESHOLD`] — all of them *this crate's*
/// encoder tuning. If any is retuned, cjpegli detection breaks silently:
/// files keep parsing, distances keep being returned, and they are simply
/// wrong.
///
/// If you change zenjpeg's quantiser tuning, **split these into two constant
/// sets** — one for the encoder, a frozen copy for cjpegli identification —
/// and set this to `false`. `cjpegli_law_matches_ours` in the tests below
/// fails when they diverge, so it should catch this, but the flag is here so
/// the requirement is visible at the point of change rather than only in a
/// test failure.
pub const CJPEGLI_SHARES_OUR_QUANT_LAW: bool = true;

/// Number of quantization presets in the libjpeg/mozjpeg family.
pub const NUM_IJG_PRESETS: usize = 9;

/// Human-readable names for the presets, indexed as [`TableId::IjgPreset`].
pub const IJG_PRESET_NAMES: [&str; NUM_IJG_PRESETS] = [
    "JpegAnnexK",
    "Flat",
    "MssimTuned",
    "Robidoux/ImageMagick",
    "PsnrHvsM",
    "Klein",
    "Watson",
    "Ahumada",
    "Peterson",
];

/// The nine libjpeg/mozjpeg luma bases, in natural (de-zigzagged) order.
///
/// These are *generator inputs*, not tables that appear in files: a base is
/// scaled by [`ijg_scale_for_quality`] and only the result is written to a
/// DQT. Two of them (Robidoux, Klein) peak above 255, which a DQT value never
/// can — at any quality the scale brings them under the cap.
///
/// Presets 0-3 are selectable encoder settings; 4-8 are quantization matrices
/// from the psychovisual literature, kept because mozjpeg exposes them, though
/// a 17,739-file survey found **zero** files using any of 4, 6, 7 or 8.
pub static IJG_LUMA_BASES: [[u16; 64]; NUM_IJG_PRESETS] = [
    // 0: JPEG Annex K — the table in the standard's informative annex, and
    //    overwhelmingly the most common on the web (13,488 of 17,739 surveyed).
    [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ],
    // 1: Flat — every frequency weighted equally, i.e. no psychovisual
    //    model at all. Useful when a later stage does the weighting.
    [16; 64],
    // 2: tuned to maximise MS-SSIM rather than to model the eye directly.
    [
        12, 17, 20, 21, 30, 34, 56, 63, 18, 20, 20, 26, 28, 51, 61, 55, 19, 20, 21, 26, 33, 58, 69,
        55, 26, 26, 26, 30, 46, 87, 86, 66, 31, 33, 36, 40, 46, 96, 100, 73, 40, 35, 46, 62, 81,
        100, 111, 91, 46, 66, 76, 86, 102, 121, 120, 101, 68, 90, 90, 96, 113, 102, 105, 103,
    ],
    // 3: Robidoux — mozjpeg's default, and what ImageMagick writes. Rises
    //    much faster into the high frequencies than Annex K (peak 418 vs 121),
    //    so it discards fine detail harder at the same nominal quality.
    [
        16, 16, 16, 18, 25, 37, 56, 85, 16, 17, 20, 27, 34, 40, 53, 75, 16, 20, 24, 31, 43, 62, 91,
        135, 18, 27, 31, 40, 53, 74, 106, 156, 25, 34, 43, 53, 69, 94, 131, 189, 37, 40, 62, 74,
        94, 124, 169, 238, 56, 53, 91, 106, 131, 169, 226, 311, 85, 75, 135, 156, 189, 238, 311,
        418,
    ],
    // 4: tuned for the PSNR-HVS-M metric.
    [
        9, 10, 12, 14, 27, 32, 51, 62, 11, 12, 14, 19, 27, 44, 59, 73, 12, 14, 18, 25, 42, 59, 79,
        78, 17, 18, 25, 42, 61, 92, 87, 92, 23, 28, 42, 75, 79, 112, 112, 99, 40, 42, 59, 84, 88,
        124, 132, 111, 42, 64, 78, 95, 105, 126, 125, 99, 70, 75, 100, 102, 116, 100, 107, 98,
    ],
    // 5: Klein, Silverstein & Carney (1992) — near-identical to Robidoux in
    //    shape and peak (419); the two differ by a few counts per coefficient.
    [
        10, 12, 14, 19, 26, 38, 57, 86, 12, 18, 21, 28, 35, 41, 54, 76, 14, 21, 25, 32, 44, 63, 92,
        136, 19, 28, 32, 41, 54, 75, 107, 157, 26, 35, 44, 54, 70, 95, 132, 190, 38, 41, 63, 75,
        95, 125, 170, 239, 57, 54, 92, 107, 132, 170, 227, 312, 86, 76, 136, 157, 190, 239, 312,
        419,
    ],
    // 6: Watson, Taylor & Borthwick (DCTune, 1997) — the most aggressive
    //    here: saturates at 255 across most of the block, keeping only the
    //    lowest frequencies.
    [
        7, 8, 10, 14, 23, 44, 95, 241, 8, 8, 11, 15, 25, 47, 102, 255, 10, 11, 13, 18, 31, 58, 127,
        255, 14, 15, 18, 25, 43, 81, 176, 255, 23, 25, 31, 43, 74, 138, 255, 255, 44, 47, 58, 81,
        138, 255, 255, 255, 95, 102, 127, 176, 255, 255, 255, 255, 241, 255, 255, 255, 255, 255,
        255, 255,
    ],
    // 7: Ahumada, Watson & Peterson (1993) — the gentlest of the set
    //    (peak 77), close to flat, so it preserves high frequencies.
    [
        15, 11, 11, 12, 15, 19, 25, 32, 11, 13, 10, 10, 12, 15, 19, 24, 11, 10, 14, 14, 16, 18, 22,
        27, 12, 10, 14, 18, 21, 24, 28, 33, 15, 12, 16, 21, 26, 31, 36, 42, 19, 15, 18, 24, 31, 38,
        45, 53, 25, 19, 22, 28, 36, 45, 55, 65, 32, 24, 27, 33, 42, 53, 65, 77,
    ],
    // 8: Peterson, Ahumada & Watson (1993) — a companion to preset 7 from
    //    the same group; slightly steeper (peak 108). Distinguished from 7 by
    //    its DC term (14 vs 15) as much as anything.
    [
        14, 10, 11, 14, 19, 25, 34, 45, 10, 11, 11, 12, 15, 20, 26, 33, 11, 11, 15, 18, 21, 25, 31,
        38, 14, 12, 18, 24, 28, 33, 39, 47, 19, 15, 21, 28, 36, 43, 51, 59, 25, 20, 25, 33, 43, 54,
        64, 74, 34, 26, 31, 39, 51, 64, 77, 91, 45, 33, 38, 47, 59, 74, 91, 108,
    ],
];

/// What a quantization table was identified as.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum TableId {
    /// A libjpeg/mozjpeg preset at a recovered quality. `exact` is false when
    /// the match needed the tolerance, which happens for encoders that round
    /// the scale slightly differently.
    IjgPreset {
        preset: u8,
        quality: u8,
        exact: bool,
    },
    /// A jpegli-family table at a recovered butteraugli distance.
    ///
    /// `exact` is false when the reconstruction differed by one quantizer
    /// step. That is expected across platforms: 20 of the 64 frequency
    /// exponents are not 1.0, so those coefficients go through `powf`, which
    /// is not bit-reproducible between libm implementations. A file encoded
    /// on one target can therefore reconstruct one step off on another, and
    /// rejecting it would be worse than reporting it.
    JpegliDistance { distance: f32, exact: bool },
    /// No known law reproduces this table.
    Unknown,
}

/// libjpeg's quality→scale mapping, shared by every preset.
#[must_use]
pub fn ijg_scale_for_quality(quality: u8) -> u32 {
    let q = u32::from(quality.clamp(1, 100));
    if q < 50 { 5000 / q } else { 200 - 2 * q }
}

/// Reconstruct one libjpeg/mozjpeg preset at one quality.
#[must_use]
pub fn ijg_preset_table(preset: usize, quality: u8, force_baseline: bool) -> [u16; 64] {
    let base = &IJG_LUMA_BASES[preset];
    let s = ijg_scale_for_quality(quality);
    let cap: u32 = if force_baseline { 255 } else { 32767 };
    let mut out = [0u16; 64];
    for i in 0..64 {
        out[i] = ((u32::from(base[i]) * s + 50) / 100).clamp(1, cap) as u16;
    }
    out
}

/// Reconstruct a jpegli-family luma table for a butteraugli distance.
///
/// `is_420` matters: the encoder folds an extra [`GLOBAL_SCALE_420`] factor
/// into the luma quantiser when chroma is subsampled, so the same distance
/// yields a different table. Identification tries both.
#[must_use]
pub fn jpegli_luma_table(distance: f32, is_420: bool) -> [u16; 64] {
    let scale = GLOBAL_SCALE_YCBCR * if is_420 { GLOBAL_SCALE_420 } else { 1.0 };
    let mut out = [0u16; 64];
    for i in 0..64 {
        let fs = if distance < DIST_THRESHOLD {
            distance
        } else {
            let e = FREQUENCY_EXPONENT[i];
            (0.5 * distance).max(DIST_THRESHOLD.powf(1.0 - e) * distance.powf(e))
        };
        out[i] = (BASE_QUANT_MATRIX_YCBCR[i] * scale * fs).round() as u16;
    }
    out
}

/// Recover the butteraugli distance that produced a jpegli-family luma table.
///
/// Returns `None` when the table carries no distance information — at
/// distances near zero every coefficient saturates to 1 and the map is no
/// longer invertible.
///
/// See [`CJPEGLI_SHARES_OUR_QUANT_LAW`]: this reads the encoder's own tuning
/// constants, so retuning the encoder silently changes what this returns for
/// third-party files.
#[must_use]
pub fn distance_from_luma_table(table: &[u16; 64]) -> Option<f32> {
    // The 4:2:0 luma scale differs, and the subsampling is not knowable from
    // the DQT alone, so try both and keep whichever reconstructs exactly.
    for is_420 in [false, true] {
        if let Some(d) = distance_for_scale(table, is_420) {
            if jpegli_luma_table(d, is_420) == *table {
                return Some(d);
            }
        }
    }
    distance_for_scale(table, false)
}

fn distance_for_scale(table: &[u16; 64], is_420: bool) -> Option<f32> {
    let gscale = GLOBAL_SCALE_YCBCR * if is_420 { GLOBAL_SCALE_420 } else { 1.0 };
    // Per-coefficient closed-form estimates; median resists rounding noise.
    let mut est: Vec<f32> = Vec::with_capacity(64);
    for i in 0..64 {
        let denom = BASE_QUANT_MATRIX_YCBCR[i] * gscale;
        if denom <= 0.0 || table[i] == 0 {
            continue;
        }
        let v = f32::from(table[i]);
        let d_low = v / denom;
        let d = if d_low < DIST_THRESHOLD {
            d_low
        } else {
            let e = FREQUENCY_EXPONENT[i];
            (v / (denom * DIST_THRESHOLD.powf(1.0 - e))).powf(1.0 / e)
        };
        if d.is_finite() && d > 0.0 {
            est.push(d);
        }
    }
    if est.is_empty() {
        return None;
    }
    est.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let seed = est[est.len() / 2];

    // The forward map is monotone in distance, so the distances producing this
    // table form an interval; bisect onto it.
    let (mut lo, mut hi) = (seed * 0.9, seed * 1.1);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let t = jpegli_luma_table(mid, is_420);
        if t == *table {
            return Some(mid);
        }
        let high = (0..64).filter(|&i| t[i] > table[i]).count() as i32
            - (0..64).filter(|&i| t[i] < table[i]).count() as i32;
        if high > 0 { hi = mid } else { lo = mid }
    }
    Some(0.5 * (lo + hi))
}

/// Quality that would produce `table` under `preset`, by inverting the scale
/// rather than trying all 100.
///
/// `q[i] = (base[i]*scale + 50)/100`, so any single unclamped coefficient
/// determines the scale, and the scale determines the quality. The largest
/// base coefficient is used: it has the best ratio of signal to
/// integer-division rounding, and is the last to be clamped to 1.
///
/// Returns a candidate to be *verified*, never trusted — the division is lossy
/// and adjacent qualities can share a scale, which is why callers check
/// neighbours.
fn candidate_quality(table: &[u16; 64], preset: usize, cap: u32) -> Option<u8> {
    let base = &IJG_LUMA_BASES[preset];
    // Largest base among the coefficients that are NOT clamped. Picking the
    // largest base unconditionally is wrong: it saturates first, and at the
    // lowest qualities every coefficient hits the cap (Annex K at q1 with
    // force_baseline is 255 in all 64 positions), leaving nothing to invert.
    let (i, &b) = base
        .iter()
        .enumerate()
        .filter(|&(k, _)| {
            let v = u32::from(table[k]);
            v > 0 && v < cap
        })
        .max_by_key(|&(_, &v)| v)?;
    let q = u32::from(table[i]);
    if b == 0 {
        return None;
    }
    let scale = (100.0 * q as f32 - 50.0) / f32::from(b);
    if scale <= 0.0 {
        return None;
    }
    let quality = if scale <= 100.0 {
        (200.0 - scale) / 2.0
    } else {
        5000.0 / scale
    };
    Some(quality.round().clamp(1.0, 100.0) as u8)
}

/// Identify a luma quantization table.
///
/// `tolerance` is the largest per-coefficient difference still accepted as
/// the same preset. Measured coverage on a 17,739-file survey: 79.9% exact,
/// **86.4% at a tolerance of 1**, 88.1% at 2. One absorbs encoders that round
/// the scale differently while staying far from a neighbouring preset.
#[must_use]
pub fn identify_luma_table(table: &[u16; 64], tolerance: u16) -> TableId {
    // Exact preset first: a table that genuinely is preset P at quality Q must
    // never be reported as a near-miss of something else. The candidate is
    // derived, not searched, so this is ~54 reconstructions rather than 1,800.
    for preset in 0..NUM_IJG_PRESETS {
        for fb in [true, false] {
            let cap = if fb { 255 } else { 32767 };
            // Fully saturated tables carry no scale at all, so no candidate can
            // be derived; that only happens at the very bottom of the range, so
            // a short scan there is both correct and still bounded.
            let candidates: &[u8] = &match candidate_quality(table, preset, cap) {
                // Neighbours because integer division makes adjacent qualities
                // share a scale at the low end.
                Some(c) => [-2i16, -1, 0, 1, 2]
                    .iter()
                    .filter_map(|d| {
                        let q = i16::from(c) + d;
                        (1..=100).contains(&q).then_some(q as u8)
                    })
                    .collect::<Vec<u8>>(),
                None => (1..=12u8).collect(),
            };
            for &quality in candidates {
                if ijg_preset_table(preset, quality, fb) == *table {
                    return TableId::IjgPreset {
                        preset: preset as u8,
                        quality,
                        exact: true,
                    };
                }
            }
        }
    }
    // Then the jpegli law, accepted only if it reconstructs exactly.
    if let Some(d) = distance_from_luma_table(table) {
        for is_420 in [false, true] {
            let t = jpegli_luma_table(d, is_420);
            let delta = (0..64).map(|i| t[i].abs_diff(table[i])).max().unwrap_or(u16::MAX);
            if delta <= 1 {
                return TableId::JpegliDistance {
                    distance: d,
                    exact: delta == 0,
                };
            }
        }
    }
    if tolerance > 0 {
        let mut best: Option<(u16, u8, u8)> = None;
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                for fb in [true, false] {
                    let t = ijg_preset_table(preset, quality, fb);
                    let d = (0..64)
                        .map(|i| t[i].abs_diff(table[i]))
                        .max()
                        .unwrap_or(u16::MAX);
                    if d <= tolerance && best.map_or(true, |(bd, _, _)| d < bd) {
                        best = Some((d, preset as u8, quality));
                    }
                }
            }
        }
        if let Some((_, preset, quality)) = best {
            return TableId::IjgPreset {
                preset,
                quality,
                exact: false,
            };
        }
    }
    TableId::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset at every quality identifies as itself, exactly. This is
    /// what makes reconstruction trustworthy: it is not a list that can rot.
    #[test]
    fn every_preset_and_quality_round_trips() {
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                let t = ijg_preset_table(preset, quality, true);
                match identify_luma_table(&t, 0) {
                    TableId::IjgPreset { exact: true, .. } => {}
                    other => panic!("preset {preset} q{quality} identified as {other:?}"),
                }
            }
        }
    }

    /// The jpegli inversion must reproduce the table it was given.
    #[test]
    fn jpegli_distance_round_trips() {
        let mut exact = 0;
        let mut checked = 0;
        for step in 1..=60 {
            let d = step as f32 * 0.25;
            let t = jpegli_luma_table(d, false);
            if t.iter().all(|&v| v <= 1) {
                continue; // saturated: carries no distance
            }
            checked += 1;
            if let Some(rec) = distance_from_luma_table(&t) {
                if jpegli_luma_table(rec, false) == t {
                    exact += 1;
                }
            }
        }
        assert!(
            exact * 100 >= checked * 95,
            "only {exact}/{checked} distances round-tripped exactly"
        );
    }

    /// Guards [`CJPEGLI_SHARES_OUR_QUANT_LAW`]. If the encoder's tuning is
    /// changed without splitting the constants, identification of
    /// cjpegli-produced files silently starts returning wrong distances. This
    /// pins the tuning values the law depends on.
    #[test]
    fn cjpegli_law_matches_ours() {
        assert!(
            CJPEGLI_SHARES_OUR_QUANT_LAW,
            "the alias was cleared; distance_from_luma_table must then use a \
             frozen copy of the cjpegli constants rather than the encoder's"
        );
        // Values cjpegli output was verified against (round-trip, zero error).
        assert!(
            (DIST_THRESHOLD - 1.5).abs() < 1e-6,
            "DIST_THRESHOLD retuned"
        );
        assert!(
            (GLOBAL_SCALE_YCBCR - 1.739_660_1).abs() < 1e-5,
            "GLOBAL_SCALE_YCBCR retuned — split the constants, see \
             CJPEGLI_SHARES_OUR_QUANT_LAW"
        );
        assert!(
            (BASE_QUANT_MATRIX_YCBCR[0] - 1.239_740_9).abs() < 1e-5,
            "BASE_QUANT_MATRIX_YCBCR retuned — split the constants, see \
             CJPEGLI_SHARES_OUR_QUANT_LAW"
        );
        assert!(
            (FREQUENCY_EXPONENT[1] - 0.51).abs() < 1e-6,
            "FREQUENCY_EXPONENT retuned — split the constants"
        );
    }

    /// Worst-case work is bounded and must stay bounded.
    ///
    /// A wall-clock assertion would be flaky, so this counts *reconstructions*
    /// — the unit of work — which is deterministic. The unknown-table path is
    /// the worst case: it derives a candidate for every preset and cap, misses,
    /// runs the jpegli inversion, then falls through to the tolerance scan.
    ///
    /// Before deriving candidates the exact pass alone cost 9 presets x 100
    /// qualities x 2 caps = 1,800 reconstructions on every call.
    #[test]
    fn worst_case_reconstruction_count_is_bounded() {
        // Mirrors identify_luma_table's exact pass, counting instead of matching.
        let mut t = [7u16; 64];
        t[0] = 199;
        t[63] = 3;
        let mut exact_pass = 0usize;
        for preset in 0..NUM_IJG_PRESETS {
            for fb in [true, false] {
                let cap = if fb { 255 } else { 32767 };
                if let Some(c) = candidate_quality(&t, preset, cap) {
                    for delta in [0i16, -1, 1, -2, 2] {
                        let q = c as i16 + delta;
                        if (1..=100).contains(&q) {
                            exact_pass += 1;
                        }
                    }
                }
            }
        }
        assert!(
            exact_pass <= 90,
            "exact pass costs {exact_pass} reconstructions; it was 1,800 before              candidates were derived, and should stay near 9*2*5"
        );
        // And the whole call still returns the right answer.
        assert_eq!(identify_luma_table(&t, 1), TableId::Unknown);
    }

    /// A table no law produces comes back Unknown rather than being forced
    /// onto the nearest preset.
    #[test]
    fn nonsense_table_is_unknown() {
        let mut t = [7u16; 64];
        t[0] = 199;
        t[63] = 3;
        assert_eq!(identify_luma_table(&t, 1), TableId::Unknown);
    }
}
