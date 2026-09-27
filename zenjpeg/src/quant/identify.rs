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
//! `q[i] = clamp(round(base[i] * GLOBAL_SCALE * freq_scale(d, i)), 1, 32767)`
//! with `freq_scale = d` below [`DIST_THRESHOLD`] and
//! `max(0.5d, T^(1-e[i]) * d^e[i])` above it. The clamp is not decoration: the
//! raw law rounds to zero below distance 0.23, and a zero quantizer is a
//! division by zero in the decoder, so the encoder never emits one. Here the
//! parameter is a
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
///
/// This must stay arithmetically identical to the encoder's own generator
/// ([`crate::quant::generate_quant_table_with_distance`] for the luma
/// component with `allow_16bit`), or tables this crate emits stop being
/// recognisable. Two details carry that:
///
/// * The `clamp(1, 32767)` is the encoder's, applied in
///   [`crate::quant::create_quant_table`]. Without it the raw law rounds to
///   **0** below distance 0.23 — a quantizer value no encoder may emit, since
///   a decoder divides by it.
/// * The multiplication associates as `base * (freq_scale * global_scale)`,
///   matching the encoder. `f32` multiplication is not associative, so
///   grouping it as `(base * global_scale) * freq_scale` instead moves a
///   product across a rounding boundary for roughly one distance in ten
///   thousand.
///
/// `jpegli_helper_matches_the_encoders_own_table_generator` in the tests below
/// holds the two in agreement.
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
        out[i] = ((BASE_QUANT_MATRIX_YCBCR[i] * (fs * scale)).round() as u16).clamp(1, 32767);
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
        if let Some(d) = distance_for_scale(table, is_420)
            && jpegli_luma_table(d, is_420) == *table
        {
            return Some(d);
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
                // Neighbours because the derivation is lossy: `(base*scale+50)/100`
                // truncates, so recovering the scale from one coefficient lands
                // up to one quality off (measured maximum: exactly 1, over all
                // 9x100x2 cells — see `derived_candidate_is_never_more_than_one_
                // quality_off`). The window is +/-2 to keep a step of margin.
                // Note no two qualities share a scale; what adjacent qualities
                // do share, at the *high* end, is a resulting table.
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
    // Then the jpegli law. Accepted within one quantizer step, not only when
    // exact — see [`TableId::JpegliDistance`] for why one step has to be
    // allowed. This runs ahead of the tolerance scan below, so a table that
    // sits one step from an IJG preset can be reported as a jpegli distance
    // instead; both are one step out, so neither is the better answer.
    if let Some(d) = distance_from_luma_table(table) {
        for is_420 in [false, true] {
            let t = jpegli_luma_table(d, is_420);
            let delta = (0..64)
                .map(|i| t[i].abs_diff(table[i]))
                .max()
                .unwrap_or(u16::MAX);
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
                    if d <= tolerance && best.is_none_or(|(bd, _, _)| d < bd) {
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
            if let Some(rec) = distance_from_luma_table(&t)
                && jpegli_luma_table(rec, false) == t
            {
                exact += 1;
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
    // The constant assertion is the point: it is a tripwire that has to start
    // failing the moment someone clears the flag, so it must not be folded away.
    #[allow(clippy::assertions_on_constants)]
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

    // ---------------------------------------------------------------------
    // Helpers shared by the tests below.
    // ---------------------------------------------------------------------

    /// Does the answer actually reproduce the table it was given?
    ///
    /// Every assertion about correctness routes through this rather than
    /// comparing to an expected `(preset, quality)`. Several inputs are
    /// genuinely ambiguous — the same bytes come from more than one parameter
    /// — so demanding one particular answer would encode an arbitrary choice
    /// as a requirement.
    fn reconstructs(id: TableId, table: &[u16; 64]) -> bool {
        match id {
            TableId::IjgPreset {
                preset, quality, ..
            } => {
                ijg_preset_table(preset as usize, quality, true) == *table
                    || ijg_preset_table(preset as usize, quality, false) == *table
            }
            TableId::JpegliDistance { distance, .. } => {
                jpegli_luma_table(distance, false) == *table
                    || jpegli_luma_table(distance, true) == *table
            }
            TableId::Unknown => false,
        }
    }

    /// Largest per-coefficient distance between the answer and the table, or
    /// `None` for `Unknown`. Used to compare answers across tolerances.
    fn answer_delta(id: TableId, table: &[u16; 64]) -> Option<u16> {
        let worst = |t: [u16; 64]| {
            (0..64)
                .map(|i| t[i].abs_diff(table[i]))
                .max()
                .unwrap_or(u16::MAX)
        };
        match id {
            TableId::IjgPreset {
                preset, quality, ..
            } => Some(
                worst(ijg_preset_table(preset as usize, quality, true))
                    .min(worst(ijg_preset_table(preset as usize, quality, false))),
            ),
            TableId::JpegliDistance { distance, .. } => Some(
                worst(jpegli_luma_table(distance, false))
                    .min(worst(jpegli_luma_table(distance, true))),
            ),
            TableId::Unknown => None,
        }
    }

    /// A fixed-seed LCG, so a property failure reproduces exactly. Hand-rolled
    /// on purpose — a random table generator is not worth a dependency.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    // ---------------------------------------------------------------------
    // libjpeg/mozjpeg preset family.
    // ---------------------------------------------------------------------

    /// The sweep the existing round-trip test does, extended over
    /// `force_baseline` as well.
    ///
    /// `force_baseline` chooses between a 255 and a 32767 ceiling, and the two
    /// give different tables for 297 of the 900 (preset, quality) cells — every
    /// one of those is a table the 255-only sweep never sees. The clamp is also
    /// where information is destroyed, so it is exactly where an inversion that
    /// derives rather than searches can come up empty.
    #[test]
    fn every_preset_quality_and_cap_round_trips() {
        let mut cells_where_the_cap_matters = 0;
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                if ijg_preset_table(preset, quality, true)
                    != ijg_preset_table(preset, quality, false)
                {
                    cells_where_the_cap_matters += 1;
                }
                for fb in [true, false] {
                    let t = ijg_preset_table(preset, quality, fb);
                    let id = identify_luma_table(&t, 0);
                    assert!(
                        matches!(id, TableId::IjgPreset { exact: true, .. }),
                        "preset {preset} q{quality} force_baseline={fb} identified as {id:?}"
                    );
                    assert!(
                        reconstructs(id, &t),
                        "preset {preset} q{quality} force_baseline={fb} -> {id:?}, \
                         which does not reproduce the table"
                    );
                }
            }
        }
        assert_eq!(
            cells_where_the_cap_matters, 297,
            "the number of cells where the 255 ceiling bites changed; the \
             force_baseline arm of this sweep is what covers them"
        );
    }

    /// The 32767 ceiling is unreachable, and that is a property of the bases.
    ///
    /// The largest product any base can make is Klein's 419 at quality 1
    /// (scale 5000), which lands at 20,950 — well under 32767. So the second
    /// ceiling never actually clamps anything, and `force_baseline` only ever
    /// selects between "clamp at 255" and "do not clamp". Adding a base with a
    /// peak above 655 would make the 32767 clamp live for the first time, and
    /// this test is the notice that the branch needs its own coverage.
    #[test]
    fn the_32767_ceiling_is_never_reached() {
        let mut peak = 0u16;
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                let t = ijg_preset_table(preset, quality, false);
                peak = peak.max(*t.iter().max().unwrap());
                assert!(
                    t.iter().all(|&v| v < 32767),
                    "preset {preset} q{quality} reaches the 32767 ceiling"
                );
            }
        }
        assert_eq!(
            peak, 20950,
            "peak reachable quantizer changed; if it now exceeds 32767 the \
             extended clamp is live and needs coverage of its own"
        );
    }

    /// Quality derivation gives up only where the table genuinely carries no
    /// scale, and every such case is inside the fallback scan.
    ///
    /// This is the shape of the bug that motivated the whole exercise: at
    /// quality 1 every Annex K coefficient clamps to 255, so no coefficient is
    /// left to invert. The guarantee needed is not "the fallback exists" but
    /// "the fallback is wide enough" — it scans 1..=12, and the deepest
    /// saturation is at quality 3.
    #[test]
    fn candidate_derivation_fails_only_on_fully_saturated_tables() {
        let mut deepest = 0u8;
        let mut cases = 0;
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                for fb in [true, false] {
                    let cap = if fb { 255 } else { 32767 };
                    let t = ijg_preset_table(preset, quality, fb);
                    let unclamped = t.iter().filter(|&&v| u32::from(v) < cap).count();
                    match candidate_quality(&t, preset, cap) {
                        Some(_) => assert!(
                            unclamped > 0,
                            "preset {preset} q{quality} fb={fb} derived a quality \
                             from a table with nothing unclamped"
                        ),
                        None => {
                            assert_eq!(
                                unclamped, 0,
                                "preset {preset} q{quality} fb={fb} gave up with \
                                 {unclamped} coefficients still carrying scale"
                            );
                            cases += 1;
                            deepest = deepest.max(quality);
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 14, "the set of scale-free tables changed");
        assert!(
            deepest <= 12,
            "a fully saturated table appears at quality {deepest}, past the \
             1..=12 fallback scan — such a table can no longer be identified"
        );
        assert_eq!(deepest, 3, "deepest saturation moved; check the scan bound");
    }

    /// One surviving coefficient is enough.
    ///
    /// Annex K at quality 2 with `force_baseline` puts 63 of its 64
    /// coefficients on the 255 ceiling. The entire scale has to be recovered
    /// from the single remaining one, which is the tightest the derivation ever
    /// gets while still having anything to work with — one step further (
    /// quality 1) is the saturated case that needs the fallback scan.
    #[test]
    fn a_single_unclamped_coefficient_is_enough() {
        let t = ijg_preset_table(0, 2, true);
        let clamped = t.iter().filter(|&&v| v == 255).count();
        assert_eq!(
            clamped, 63,
            "Annex K q2 no longer has exactly one coefficient below the ceiling"
        );
        assert_eq!(
            identify_luma_table(&t, 0),
            TableId::IjgPreset {
                preset: 0,
                quality: 2,
                exact: true
            }
        );
        // And the worst partial saturation anywhere on the grid is this cell.
        let mut worst = 0;
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                let t = ijg_preset_table(preset, quality, true);
                let n = t.iter().filter(|&&v| v == 255).count();
                if n < 64 {
                    worst = worst.max(n);
                }
            }
        }
        assert_eq!(worst, 63);
    }

    /// The derived quality is never more than one step from a correct one.
    ///
    /// `identify_luma_table` reconstructs at the candidate plus or minus two.
    /// That window is only justified if the derivation error stays inside it,
    /// and the margin is invisible from the code — the division that loses the
    /// information is two steps removed from the quality it produces. Measured
    /// error is at most 1, so the window has a full step spare; if this starts
    /// reporting 2 the window is exactly tight and the next rounding change
    /// silently breaks identification.
    #[test]
    fn derived_candidate_is_never_more_than_one_quality_off() {
        let mut worst = 0i32;
        for preset in 0..NUM_IJG_PRESETS {
            for quality in 1..=100u8 {
                for fb in [true, false] {
                    let cap = if fb { 255 } else { 32767 };
                    let t = ijg_preset_table(preset, quality, fb);
                    let Some(c) = candidate_quality(&t, preset, cap) else {
                        continue;
                    };
                    // Distance to the nearest quality that reproduces this table,
                    // not to `quality` — colliding qualities are equally correct.
                    let err = (1..=100u8)
                        .filter(|&q| ijg_preset_table(preset, q, fb) == t)
                        .map(|q| (i32::from(c) - i32::from(q)).abs())
                        .min()
                        .expect("the generating quality always reproduces its own table");
                    assert!(
                        err <= 2,
                        "preset {preset} q{quality} fb={fb}: derived candidate {c} is \
                         {err} qualities from any correct answer, outside the +/-2 window"
                    );
                    worst = worst.max(err);
                }
            }
        }
        assert_eq!(
            worst, 1,
            "derivation error changed; at 2 the +/-2 window is exactly tight"
        );
    }

    /// Qualities that produce the same bytes are answered with one of
    /// themselves, not with a fixed favourite.
    ///
    /// `(base*scale + 50)/100` truncates, so distinct scales collapse onto one
    /// table — the flat base does it 24 times, and quality 96 through 100 all
    /// give all-ones. Such an input has no single right answer, so the only
    /// thing worth asserting is that the answer is *a* right answer.
    #[test]
    fn colliding_qualities_return_a_member_of_their_group() {
        let mut groups = 0;
        for preset in 0..NUM_IJG_PRESETS {
            for fb in [true, false] {
                let mut quality = 1u8;
                while quality <= 100 {
                    let t = ijg_preset_table(preset, quality, fb);
                    let group: Vec<u8> = (quality..=100)
                        .take_while(|&q| ijg_preset_table(preset, q, fb) == t)
                        .collect();
                    if group.len() > 1 {
                        groups += 1;
                        let id = identify_luma_table(&t, 0);
                        assert!(
                            reconstructs(id, &t),
                            "preset {preset} qualities {group:?} all give the same table, \
                             and {id:?} does not reproduce it"
                        );
                    }
                    quality += group.len() as u8;
                }
            }
        }
        assert!(
            groups >= 40,
            "only {groups} collision groups found; this test stops covering the \
             ambiguous case if the truncation stops collapsing qualities"
        );
    }

    /// No two qualities share a scale, so collisions come from the table
    /// rounding rather than from the quality mapping.
    ///
    /// Worth pinning separately because the two causes call for different
    /// neighbourhoods: a shared scale would make the *derivation* ambiguous,
    /// while shared bytes only make the *answer* ambiguous. Below 50 the
    /// mapping is `5000/quality`, which does look like it should collide, and
    /// does not — the smallest gap is `5000/48 - 5000/49` = 2.
    #[test]
    fn no_two_qualities_share_a_scale() {
        let mut seen = [false; 5001];
        for quality in 1..=100u8 {
            let s = ijg_scale_for_quality(quality) as usize;
            assert!(!seen[s], "quality {quality} reuses scale {s}");
            seen[s] = true;
        }
    }

    /// Qualities outside 1..=100 alias the ends rather than producing a
    /// degenerate table, and identification never answers with quality 0.
    #[test]
    fn out_of_range_qualities_alias_the_ends() {
        for preset in 0..NUM_IJG_PRESETS {
            for fb in [true, false] {
                assert_eq!(
                    ijg_preset_table(preset, 0, fb),
                    ijg_preset_table(preset, 1, fb),
                    "quality 0 should clamp up to 1"
                );
                assert_eq!(
                    ijg_preset_table(preset, u8::MAX, fb),
                    ijg_preset_table(preset, 100, fb),
                    "quality above 100 should clamp down to 100"
                );
            }
        }
        assert_eq!(ijg_scale_for_quality(0), ijg_scale_for_quality(1));
        assert_eq!(ijg_scale_for_quality(u8::MAX), ijg_scale_for_quality(100));
    }

    /// Two presets produce the same table only where the table has collapsed
    /// to a single repeated value.
    ///
    /// The bases differ everywhere, so any shared table means the scale has
    /// destroyed the difference: all-255 (every preset at the bottom, under
    /// `force_baseline`) and all-ones (every preset at quality 100). Those two
    /// are unavoidable and unattributable. A *third* shared table would mean
    /// the preset really cannot be recovered from the bytes for some ordinary
    /// quality, which is worth knowing about immediately.
    #[test]
    fn presets_share_a_table_only_when_it_is_degenerate() {
        let mut shared: Vec<[u16; 64]> = Vec::new();
        for a in 0..NUM_IJG_PRESETS {
            for b in (a + 1)..NUM_IJG_PRESETS {
                for qa in 1..=100u8 {
                    for fb in [true, false] {
                        let ta = ijg_preset_table(a, qa, fb);
                        for qb in 1..=100u8 {
                            if ijg_preset_table(b, qb, fb) != ta {
                                continue;
                            }
                            let flat = ta.iter().all(|&v| v == ta[0]);
                            assert!(
                                flat,
                                "preset {a} q{qa} and preset {b} q{qb} (fb={fb}) produce the \
                                 same non-degenerate table, so the preset cannot be recovered"
                            );
                            if !shared.contains(&ta) {
                                shared.push(ta);
                            }
                        }
                    }
                }
            }
        }
        shared.sort_by_key(|t| t[0]);
        let values: Vec<u16> = shared.iter().map(|t| t[0]).collect();
        assert_eq!(
            values,
            vec![1, 255],
            "the set of tables shared between presets changed"
        );
    }

    /// Robidoux and Klein stay apart at every quality.
    ///
    /// Their bases peak at 418 and 419 and track each other within a few counts
    /// the whole way, so they are the pair most likely to become
    /// indistinguishable once the scale divides the difference away. They do
    /// get within a single count of each other — at quality 86, where 15
    /// coefficients differ by exactly 1 — which means any tolerance at all
    /// makes them interchangeable, and only the exact pass keeps them apart.
    #[test]
    fn robidoux_and_klein_stay_distinguishable() {
        let mut closest = u16::MAX;
        for q in 1..=100u8 {
            for fb in [true, false] {
                let r = ijg_preset_table(3, q, fb);
                let k = ijg_preset_table(5, q, fb);
                if r.iter().all(|&v| v == r[0]) {
                    continue; // the degenerate tables every preset shares
                }
                let delta = (0..64).map(|i| r[i].abs_diff(k[i])).max().unwrap();
                assert!(
                    delta > 0,
                    "Robidoux and Klein coincide at quality {q} (fb={fb})"
                );
                closest = closest.min(delta);
            }
        }
        assert_eq!(closest, 1, "the two bases moved relative to each other");
        // Exactness is the only thing separating them there.
        assert_eq!(
            identify_luma_table(&ijg_preset_table(5, 86, true), 1),
            TableId::IjgPreset {
                preset: 5,
                quality: 86,
                exact: true
            },
            "Klein at q86 must not be absorbed by Robidoux, which is one count away"
        );
    }

    // ---------------------------------------------------------------------
    // jpegli family.
    // ---------------------------------------------------------------------

    /// The reconstruction helper must agree with the encoder that produced the
    /// files it is meant to recognise.
    ///
    /// Every other jpegli test here compares the helper against itself, which
    /// cannot catch the helper drifting away from the encoder — and it had.
    /// The encoder's generator is the oracle: it applies
    /// [`crate::quant::create_quant_table`]'s `clamp(1, 32767)`, and it
    /// multiplies `base * (freq_scale * global_scale)`. Reproducing the law
    /// without the clamp emits quantizer zeros below distance 0.23; grouping
    /// the multiplication differently moves the occasional product across a
    /// rounding boundary. Either makes a table this crate wrote unrecognisable.
    #[test]
    fn jpegli_helper_matches_the_encoders_own_table_generator() {
        use crate::quant::generate_quant_table_with_distance;
        use crate::types::ColorSpace;

        for step in 1..=3000u32 {
            let distance = step as f32 * 0.01; // 0.01 .. 30
            for is_420 in [false, true] {
                let ours = jpegli_luma_table(distance, is_420);
                let encoder = generate_quant_table_with_distance(
                    distance,
                    0,
                    ColorSpace::YCbCr,
                    false,
                    is_420,
                    true,
                )
                .values;
                assert_eq!(
                    ours, encoder,
                    "distance {distance} is_420={is_420}: the identification law and \
                     the encoder disagree, so a table zenjpeg emits here is not \
                     recognisable"
                );
            }
        }
    }

    /// No reachable distance produces a quantizer of zero.
    ///
    /// A DQT entry of zero is a division by zero in every decoder, so an
    /// encoder may not emit one — the clamp in
    /// [`crate::quant::create_quant_table`] is what prevents it. Without the
    /// same clamp here, identification would accept tables containing zeros as
    /// exact jpegli matches while rejecting the real, clamped tables the
    /// encoder writes at those distances.
    #[test]
    fn no_distance_produces_a_zero_quantizer() {
        for step in 0..=4000u32 {
            let distance = step as f32 * 0.005;
            for is_420 in [false, true] {
                let t = jpegli_luma_table(distance, is_420);
                assert!(
                    t.iter().all(|&v| v >= 1),
                    "distance {distance} is_420={is_420} yields a zero quantizer"
                );
                assert!(t.iter().all(|&v| v <= 32767));
            }
        }
        // Degenerate inputs land on the floor rather than wrapping or panicking.
        assert!(jpegli_luma_table(0.0, false).iter().all(|&v| v == 1));
        assert!(jpegli_luma_table(f32::NAN, false).iter().all(|&v| v == 1));
        assert!(jpegli_luma_table(-1.0, false).iter().all(|&v| v == 1));
        assert!(
            jpegli_luma_table(f32::MAX, false)
                .iter()
                .all(|&v| v == 32767)
        );
        assert!(
            jpegli_luma_table(f32::INFINITY, false)
                .iter()
                .all(|&v| v == 32767)
        );
    }

    /// Distances round-trip in both subsampling modes, densely across the
    /// piecewise seam.
    ///
    /// The frequency scale switches branch at [`DIST_THRESHOLD`], and 20 of the
    /// 64 exponents are not 1.0, so the inversion changes shape there. A seam
    /// in a piecewise function is where a closed-form inverse is most likely to
    /// pick the wrong branch, so the grid is deliberately dense on both sides
    /// of 1.5 as well as broad.
    #[test]
    fn jpegli_distances_round_trip_in_both_subsampling_modes() {
        let mut distances: Vec<f32> = (1..=1200).map(|s| s as f32 * 0.025).collect();
        distances.extend((-60..=60).map(|k| DIST_THRESHOLD + k as f32 * 0.002));
        for &distance in &distances {
            for is_420 in [false, true] {
                let t = jpegli_luma_table(distance, is_420);
                if t.iter().all(|&v| v == 1) {
                    continue; // on the floor: the table no longer carries a distance
                }
                let id = identify_luma_table(&t, 0);
                assert!(
                    reconstructs(id, &t),
                    "distance {distance} is_420={is_420} -> {id:?}, which does not \
                     reproduce the table"
                );
            }
        }
    }

    /// Below the threshold a 4:2:0 table *is* a 4:4:4 table at 1.22x the
    /// distance, and identification reports the 4:4:4 reading.
    ///
    /// The extra [`GLOBAL_SCALE_420`] is a plain multiplier while the frequency
    /// scale is still linear in the distance, so the two parameterisations
    /// produce byte-identical tables and no amount of analysis of the DQT can
    /// separate them. The recovered distance is therefore 22% high for a
    /// subsampled file, and flagged `exact` — correctly, since it does
    /// reconstruct. Recording it here so the caveat is visible: the subsampling
    /// is knowable from the frame header, not from the table, and a caller who
    /// has it should not take this distance at face value.
    #[test]
    fn a_420_table_below_the_threshold_reads_as_a_444_distance() {
        let crossover = DIST_THRESHOLD / GLOBAL_SCALE_420; // ~1.2295
        for step in 15..=240u32 {
            let distance = step as f32 * 0.005; // 0.075 .. 1.2
            if distance >= crossover {
                break;
            }
            let t = jpegli_luma_table(distance, true);
            if t.iter().all(|&v| v == 1) {
                continue;
            }
            assert_eq!(
                t,
                jpegli_luma_table(distance * GLOBAL_SCALE_420, false),
                "distance {distance}: 4:2:0 and 4:4:4-at-1.22x stopped coinciding"
            );
            match identify_luma_table(&t, 0) {
                TableId::JpegliDistance { distance: got, .. } => {
                    let ratio = got / distance;
                    assert!(
                        (1.15..=1.30).contains(&ratio),
                        "distance {distance} recovered as {got} (ratio {ratio}); the \
                         4:4:4 alias is documented behaviour, a native reading is not"
                    );
                }
                other => panic!("distance {distance} is_420=true identified as {other:?}"),
            }
        }
        // Far enough above the seam the two readings separate and the native
        // 4:2:0 distance comes back.
        for step in 4..=40u32 {
            let distance = step as f32 * 0.5; // 2.0 .. 20.0
            let t = jpegli_luma_table(distance, true);
            match identify_luma_table(&t, 0) {
                TableId::JpegliDistance { distance: got, .. } => assert!(
                    (got - distance).abs() < 0.05 * distance,
                    "distance {distance} is_420=true recovered as {got}, not the \
                     native reading"
                ),
                other => panic!("distance {distance} is_420=true identified as {other:?}"),
            }
        }
    }

    /// Near-lossless jpegli tables lose their distance, and say so.
    ///
    /// Below distance 0.15 the clamp pins most of the table to 1 and the
    /// remaining coefficients cannot pin down a distance; the table becomes
    /// all-ones, which is also Annex K at quality 100. Identification answers
    /// with the preset, and the answer does reconstruct — it is simply not
    /// recoverable which encoder wrote it. Directly above that floor there is a
    /// narrow band where enough coefficients are pinned to mislead the
    /// bisection; those come back non-exact rather than wrong.
    #[test]
    fn near_lossless_jpegli_tables_lose_their_distance() {
        assert!(jpegli_luma_table(0.05, false).iter().all(|&v| v == 1));
        let id = identify_luma_table(&[1u16; 64], 0);
        assert!(reconstructs(id, &[1u16; 64]));

        // The band above the floor: honest about not being exact, never wrong
        // about being exact.
        let mut inexact = 0;
        for step in 1..=200u32 {
            let distance = step as f32 * 0.0025; // 0.0025 .. 0.5
            for is_420 in [false, true] {
                let t = jpegli_luma_table(distance, is_420);
                let id = identify_luma_table(&t, 0);
                match id {
                    TableId::JpegliDistance { exact: true, .. }
                    | TableId::IjgPreset { exact: true, .. } => assert!(
                        reconstructs(id, &t),
                        "distance {distance} is_420={is_420} claims an exact match \
                         that does not reproduce the table"
                    ),
                    TableId::JpegliDistance { exact: false, .. } => inexact += 1,
                    other => panic!("distance {distance} is_420={is_420} -> {other:?}"),
                }
            }
        }
        assert!(
            inexact <= 20,
            "{inexact} distances below 0.5 no longer invert exactly; the \
             non-invertible band has widened"
        );
    }

    // ---------------------------------------------------------------------
    // Tolerance, adversarial input, and the property test.
    // ---------------------------------------------------------------------

    /// Raising the tolerance never makes the answer worse.
    ///
    /// The tolerant scan keeps the smallest per-coefficient distance it finds,
    /// so a larger tolerance searches a superset and must return a match at
    /// least as close. Two ways that could break: an exact answer getting
    /// replaced by a tolerant one (the exact passes run first, so this would
    /// mean the ordering was changed), or the scan keeping the first match
    /// within tolerance rather than the closest.
    #[test]
    fn raising_the_tolerance_never_degrades_the_answer() {
        let mut rng = Lcg(0xC0FF_EE00_1234_5678);
        for _ in 0..150 {
            let preset = rng.below(NUM_IJG_PRESETS as u64) as usize;
            let quality = rng.below(100) as u8 + 1;
            let fb = rng.below(2) == 0;
            let mut t = ijg_preset_table(preset, quality, fb);
            for _ in 0..rng.below(6) {
                let i = rng.below(64) as usize;
                let nudge = rng.below(9) as i32 - 4;
                t[i] = (i32::from(t[i]) + nudge).clamp(1, 65535) as u16;
            }

            let exact_answer = match identify_luma_table(&t, 0) {
                id @ (TableId::IjgPreset { exact: true, .. }
                | TableId::JpegliDistance { exact: true, .. }) => Some(id),
                _ => None,
            };
            let mut previous: Option<u16> = None;
            for tolerance in [0u16, 1, 2, 4, 8, 16] {
                let id = identify_luma_table(&t, tolerance);
                if let Some(want) = exact_answer {
                    assert_eq!(
                        id, want,
                        "tolerance {tolerance} replaced the exact answer {want:?} for {t:?}"
                    );
                }
                let delta = answer_delta(id, &t);
                if let (Some(before), Some(now)) = (previous, delta) {
                    assert!(
                        now <= before,
                        "tolerance {tolerance} answered {now} away when {before} was \
                         already reachable, for {t:?}"
                    );
                }
                assert!(
                    !(previous.is_some() && delta.is_none()),
                    "tolerance {tolerance} lost a match it had at a lower tolerance, \
                     for {t:?}"
                );
                if delta.is_some() {
                    previous = delta;
                }
            }
        }
    }

    /// Malformed and degenerate tables come back with a reconstructing answer
    /// or `Unknown`, and never panic.
    ///
    /// A DQT is attacker-controlled: zeros (illegal, but parseable), values
    /// above 255 in a table that is otherwise baseline, and every coefficient
    /// pinned to a ceiling all have to be survivable. The interesting failure
    /// is not a panic but a confident wrong answer — `exact: true` on something
    /// no encoder could have produced.
    #[test]
    fn adversarial_tables_never_panic_or_claim_a_false_exact() {
        let mut cases: Vec<(String, [u16; 64])> = vec![
            ("all zeros".into(), [0u16; 64]),
            ("all ones".into(), [1u16; 64]),
            ("all 255".into(), [255u16; 64]),
            ("all 256".into(), [256u16; 64]),
            ("all 32767".into(), [32767u16; 64]),
            ("all 65535".into(), [u16::MAX; 64]),
        ];
        // One coefficient out of place in an otherwise real table.
        for (label, mut t) in [
            ("annex k q75", ijg_preset_table(0, 75, true)),
            ("robidoux q40", ijg_preset_table(3, 40, false)),
            ("jpegli d2.0", jpegli_luma_table(2.0, false)),
        ] {
            let base = t;
            for (what, value) in [("zero", 0u16), ("over 255", 999), ("max", u16::MAX)] {
                t = base;
                t[40] = value;
                cases.push((format!("{label} with {what} at 40"), t));
                t = base;
                t[0] = value;
                cases.push((format!("{label} with {what} at DC"), t));
            }
            // Half the table zeroed.
            t = base;
            for v in t.iter_mut().take(32) {
                *v = 0;
            }
            cases.push((format!("{label} half zeroed"), t));
        }
        // A baseline table with a single 16-bit value smuggled in.
        let mut smuggled = ijg_preset_table(0, 50, true);
        smuggled[63] = 300;
        cases.push(("annex k q50 with a 16-bit tail".into(), smuggled));

        for (label, table) in cases {
            for tolerance in [0u16, 1, 4, u16::MAX] {
                let id = identify_luma_table(&table, tolerance);
                if matches!(
                    id,
                    TableId::IjgPreset { exact: true, .. }
                        | TableId::JpegliDistance { exact: true, .. }
                ) {
                    assert!(
                        reconstructs(id, &table),
                        "{label} at tolerance {tolerance} claimed exact {id:?}, which \
                         does not reproduce the table"
                    );
                }
            }
        }

        // Specific answers worth pinning: a table of ceilings belongs to no law.
        assert_eq!(identify_luma_table(&[32767u16; 64], 4), TableId::Unknown);
        assert_eq!(identify_luma_table(&[u16::MAX; 64], 4), TableId::Unknown);
        // Zeros are illegal in a DQT; nothing may claim to have produced them.
        assert_eq!(identify_luma_table(&[0u16; 64], 0), TableId::Unknown);
    }

    /// Over a broad population of tables, an `exact` claim always reproduces
    /// the input and nothing panics.
    ///
    /// The population deliberately mixes tables from both laws, tables nudged
    /// off them, and noise, because uniform noise alone never reaches the exact
    /// arms at all — it is answered `Unknown` essentially always, and would
    /// test nothing. The seed is fixed so a counterexample is reproducible.
    #[test]
    fn random_tables_never_panic_and_exact_claims_reconstruct() {
        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let mut reached_ijg = 0;
        let mut reached_jpegli = 0;
        let mut reached_unknown = 0;

        for n in 0..3000 {
            let mut t = match rng.below(5) {
                0 => ijg_preset_table(
                    rng.below(NUM_IJG_PRESETS as u64) as usize,
                    rng.below(100) as u8 + 1,
                    rng.below(2) == 0,
                ),
                1 => jpegli_luma_table(rng.below(4000) as f32 * 0.005 + 0.01, rng.below(2) == 0),
                2 => {
                    // A real table knocked slightly off its law.
                    let mut t = ijg_preset_table(
                        rng.below(NUM_IJG_PRESETS as u64) as usize,
                        rng.below(100) as u8 + 1,
                        rng.below(2) == 0,
                    );
                    for _ in 0..=rng.below(4) {
                        let i = rng.below(64) as usize;
                        t[i] = (i32::from(t[i]) + rng.below(5) as i32 - 2).clamp(0, 65535) as u16;
                    }
                    t
                }
                3 => {
                    let mut t = [0u16; 64];
                    for v in t.iter_mut() {
                        *v = rng.below(300) as u16;
                    }
                    t
                }
                _ => {
                    let mut t = [0u16; 64];
                    for v in t.iter_mut() {
                        *v = rng.below(65536) as u16;
                    }
                    t
                }
            };
            // Sprinkle in the illegal-but-parseable zero.
            if n % 7 == 0 {
                t[rng.below(64) as usize] = 0;
            }

            let tolerance = if n % 10 == 0 { 2 } else { 0 };
            let id = identify_luma_table(&t, tolerance);
            match id {
                TableId::IjgPreset { exact, .. } => {
                    reached_ijg += 1;
                    assert!(
                        !exact || reconstructs(id, &t),
                        "exact {id:?} does not reproduce {t:?}"
                    );
                }
                TableId::JpegliDistance { distance, exact } => {
                    reached_jpegli += 1;
                    assert!(
                        distance.is_finite() && distance > 0.0,
                        "bad distance {distance}"
                    );
                    assert!(
                        !exact || reconstructs(id, &t),
                        "exact {id:?} does not reproduce {t:?}"
                    );
                }
                TableId::Unknown => reached_unknown += 1,
            }
        }

        // The population has to actually reach every arm, or the property above
        // is vacuous.
        assert!(reached_ijg > 300, "only {reached_ijg} preset answers");
        assert!(reached_jpegli > 300, "only {reached_jpegli} jpegli answers");
        assert!(
            reached_unknown > 300,
            "only {reached_unknown} unknown answers"
        );
    }
}
