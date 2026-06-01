//! Source-feature quality picker — warm-starts the `Quality::Zq` closed
//! loop with the RD-optimal categorical config for the target.
//!
//! At encode time, given the source pixels and a target zensim Profile-A
//! score, this:
//!   1. extracts the 108 zenanalyze `SUPPORTED` source features (the
//!      `experimental` + `hdr` set — the exact features and ORDER the picker
//!      trained on; see `picker_data/feature_order.txt`),
//!   2. appends `zq_norm = target / 100`,
//!   3. runs the distilled ZNPR-v3 MLP (`picker_data/picker_zenjpeg_a_v3_f16.bin`,
//!      held-out byte-overhead 3.35 %, bytes-SROCC 0.906), and
//!   4. argmins the predicted per-cell `bytes_log` to the cheapest
//!      categorical cell (subsampling × progressive × sharp_yuv × effort)
//!      that reaches the target.
//!
//! That cell seeds the closed loop's config so the first pass already uses
//! the right subsampling/effort/etc. instead of a content-blind default.
//! The picker only predicts the *config*; the loop still refines the codec
//! `q` to land the achieved zensim:A score on target.
//!
//! Gated behind `__picker-research` (pulls the unpublished `zenpredict` v3
//! runtime + the `zenanalyze/hdr` features); off by default.

use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, AnalysisResults, FeatureSet};
use zenpredict::{
    AllowedMask, Model, Predictor, ScoreTransform, argmin_masked_in_range,
    first_out_of_distribution,
};

use crate::encode::{ChromaSubsampling, Effort};

/// Shipped FIXED picker bake (f16, distilled [64,64]; sha256 `5b807ce2…`).
const PICKER_BAKE: &[u8] = include_bytes!("picker_data/picker_zenjpeg_a_v3_f16.bin");

/// zenanalyze `SUPPORTED` source features (experimental + hdr).
const N_FEATURES: usize = 108;
/// Inputs = 108 features + `zq_norm`.
const N_INPUTS: usize = 109;
/// Categorical cells = subsampling{420,422,444} × progressive{f,t} ×
/// sharp_yuv{f,t} × effort{0,1,2}, in the bake's `cell_labels` order.
const N_CELLS: usize = 36;
/// Cell index range `(start, end)` for `argmin_masked_in_range`.
const RANGE_BYTES_LOG: (usize, usize) = (0, N_CELLS);

/// The config the picker chose for the target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PickedConfig {
    pub subsampling: ChromaSubsampling,
    pub progressive: bool,
    pub sharp_yuv: bool,
    pub effort: Effort,
}

/// One feature's value as f32, matching the training extractor: the NaN
/// sentinel ("too few samples for this percentile") and any missing value
/// map to 0.0 (training blank cells were imputed to 0.0).
fn feat_f32(a: &AnalysisResults, f: AnalysisFeature) -> f32 {
    match a.get_f32(f) {
        Some(v) if !v.is_nan() => v,
        _ => 0.0,
    }
}

/// The picker's feature contract: `feat_<i>\t<zenanalyze name>` per line, in
/// TRAINING ORDER. Embedded so the runtime resolves each input BY NAME against
/// whatever features this zenanalyze build exposes — never by `SUPPORTED`
/// position.
///
/// **Forward-compatible by construction.** zenanalyze adds features in chunks
/// (e.g. the `hdr` tier) with new stable ids; `SUPPORTED` grows and its
/// iteration order shifts. Because we resolve OUR features by name, those
/// additions do not touch this bake's inputs. The picker only breaks if a
/// feature it NEEDS is REMOVED — genuinely breaking (semver-major) — at which
/// point [`resolve_features`] returns `None` and the encoder keeps its
/// heuristic rather than feed the MLP a misaligned vector.
const FEATURE_ORDER: &str = include_str!("picker_data/feature_order.txt");

/// Resolve the embedded feature names to `AnalysisFeature`s in training order.
/// `None` if any name is no longer known to this zenanalyze build (a needed
/// feature was removed). This is the forward-compatibility seam: it depends
/// only on the NAMES the bake declares, not on `SUPPORTED`'s size or order.
fn resolve_features() -> Option<Vec<AnalysisFeature>> {
    // name -> feature, over every feature THIS build knows. SUPPORTED (with
    // experimental+hdr) is a superset of the picker's set; extra features are
    // simply ignored, which is exactly what makes additions non-breaking.
    let mut by_name = std::collections::HashMap::with_capacity(256);
    for f in FeatureSet::SUPPORTED.iter() {
        by_name.insert(f.name(), f);
    }
    let mut out = Vec::with_capacity(N_FEATURES);
    for line in FEATURE_ORDER.lines() {
        let Some(col) = line.split('\t').nth(1).map(str::trim) else {
            continue;
        };
        if col.is_empty() {
            continue;
        }
        // The extractor writes columns as `feat_<name>`; `AnalysisFeature::name()`
        // returns the bare field name (`stringify!($field)`). Strip the prefix so
        // the two conventions meet. Fall back to the raw column if a future
        // exporter drops the prefix.
        let name = col.strip_prefix("feat_").unwrap_or(col);
        out.push(*by_name.get(name)?);
    }
    (out.len() == N_FEATURES).then_some(out)
}

/// Build the 109-input vector from an already-computed analysis result + the
/// resolved `features` (in training order) + `zq_norm`. The analysis-free core
/// shared by the fresh-analysis path ([`build_inputs`]) and the packed path
/// ([`pick_config_from_packed`]).
fn build_inputs_from_results(
    a: &AnalysisResults,
    target_zensim_a: f32,
    features: &[AnalysisFeature],
) -> [f32; N_INPUTS] {
    let mut x = [0.0f32; N_INPUTS];
    for (i, &f) in features.iter().enumerate().take(N_FEATURES) {
        // Degenerate-image guard: a non-finite feature (NaN/inf from an
        // all-flat or pathological source) collapses to 0.0 — the same value
        // the trainer used for a missing/empty cell — rather than poisoning
        // the whole forward pass with a propagating NaN.
        let v = feat_f32(a, f);
        x[i] = if v.is_finite() { v } else { 0.0 };
    }
    // OOB target guard: the Profile-A dial is 0..100. A finite target beyond
    // it saturates to the nearest edge (there is no quality below 0 or above
    // 100), so e.g. -2000 → 0.0 and 100/200 → 1.0. A non-finite target is
    // left as NaN here and rejected downstream (caller bug, not a quality
    // request).
    x[N_FEATURES] = if target_zensim_a.is_finite() {
        (target_zensim_a / 100.0).clamp(0.0, 1.0)
    } else {
        f32::NAN
    };
    x
}

/// Build the 109-input vector by analyzing `rgb` fresh.
fn build_inputs(
    rgb: &[u8],
    w: u32,
    h: u32,
    target_zensim_a: f32,
    features: &[AnalysisFeature],
) -> [f32; N_INPUTS] {
    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let a = zenanalyze::analyze_features_rgb8(rgb, w, h, &query);
    build_inputs_from_results(&a, target_zensim_a, features)
}

/// Map a cell index (argmin over the 36 `bytes_log` outputs) to its config.
/// Order MUST match the bake's `cell_labels`: outer→inner =
/// subsampling{420,422,444} × progressive{false,true} × sharp_yuv{false,true}
/// × effort{0,1,2}.
fn cell_to_config(cell: usize) -> PickedConfig {
    let effort_i = cell % 3;
    let sharp = (cell / 3) % 2 == 1;
    let prog = (cell / 6) % 2 == 1;
    let sub_i = (cell / 12) % 3;
    PickedConfig {
        subsampling: match sub_i {
            0 => ChromaSubsampling::Quarter,        // 4:2:0
            1 => ChromaSubsampling::HalfHorizontal, // 4:2:2
            _ => ChromaSubsampling::None,           // 4:4:4
        },
        progressive: prog,
        sharp_yuv: sharp,
        effort: match effort_i {
            0 => Effort::Fast,
            1 => Effort::Balanced,
            _ => Effort::Max,
        },
    }
}

/// Run the bake on a built input vector and argmin to a config. `None` if the
/// bake can't load/predict, the target is non-finite, or an input is
/// out-of-distribution — in every case the caller keeps its heuristic config.
fn run_model(x: &[f32; N_INPUTS]) -> Option<PickedConfig> {
    // A non-finite target (NaN/±inf) is a caller bug, not a quality request —
    // saturating it would silently pick a wrong config, so keep the encoder's
    // heuristic instead. Finite OOB targets were already saturated to the
    // [0,100] dial in `build_inputs*`, so they fall through here.
    if !x[N_FEATURES].is_finite() {
        return None;
    }
    let model = Model::from_bytes(PICKER_BAKE).ok()?;
    let mut predictor = Predictor::new(&model);
    // OOD rescue seam: if the bake declares per-input bounds, an input outside
    // them means the forward pass would extrapolate — `first_out_of_distribution`
    // also catches any NaN/inf that slipped through. With no known-good
    // fallback table in this bake, fall through to the caller's heuristic (the
    // "known-good rescue strategy on a hit" `zenpredict::bounds` documents).
    // Dormant while the bake's `feature_bounds` are empty; auto-activates if a
    // future re-bake includes them — no code change needed.
    let bounds = model.feature_bounds();
    if !bounds.is_empty() && first_out_of_distribution(x, bounds).is_some() {
        return None;
    }
    let out = if model.has_nontrivial_feature_transforms() {
        predictor.predict_transformed(x).ok()?
    } else {
        predictor.predict(x).ok()?
    };
    // All cells allowed; pick the argmin of predicted log-bytes (Exp because
    // the outputs are in log space — matches the trainer's argmin metric).
    let allow = [true; N_CELLS];
    let mask = AllowedMask::new(&allow);
    let cell = argmin_masked_in_range(out, RANGE_BYTES_LOG, &mask, ScoreTransform::Exp, None)?;
    Some(cell_to_config(cell))
}

/// Predict the RD-optimal config for `target_zensim_a` (a Profile-A score,
/// 0..100) from the source RGB8. `None` if the bake can't load/predict — the
/// caller then keeps its bucket-heuristic config.
pub(crate) fn pick_config(
    rgb: &[u8],
    w: u32,
    h: u32,
    target_zensim_a: f32,
) -> Option<PickedConfig> {
    // Forward-compat seam: resolve the bake's features BY NAME. Returns None
    // (→ caller's heuristic) only if a feature the picker NEEDS was REMOVED
    // from zenanalyze; features ADDED since training are tolerated.
    let features = resolve_features()?;
    let x = build_inputs(rgb, w, h, target_zensim_a, &features);
    run_model(&x)
}

/// Why a packed-feature pick could not be made from caller-supplied data.
///
/// A bake/predict failure is deliberately NOT one of these — that returns
/// `Ok(None)` so the encoder keeps its heuristic, exactly like [`pick_config`].
/// These variants are reserved for the caller having supplied *bad or
/// incomplete* feature data.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum PackedPickError {
    /// The packed pairs were malformed (duplicate id or non-finite value).
    Unpack(zenanalyze::feature::PackError),
    /// The pack omitted one or more features the picker needs; carries their
    /// stable ids.
    Missing(zenanalyze::feature::MissingFeatures),
}

/// Predict the RD-optimal config from **pre-computed** zenanalyze features —
/// `packed` is a [`zenanalyze::feature::AnalysisResults::pack`] blob — so the
/// encoder skips re-analyzing the source. This is the cross-version path: the
/// caller may have produced `packed` with a different zenanalyze version, and
/// it travels as plain `(u16, f32)` data rather than a versioned type.
///
/// Returns `Err` if the caller's data is bad: [`PackedPickError::Unpack`] for a
/// malformed pack, or [`PackedPickError::Missing`] (with the absent ids) if it
/// doesn't carry every feature the picker needs — a clear error beats silently
/// feeding the model zeroed inputs. Returns `Ok(None)` (→ heuristic) on a
/// bake/predict failure or a non-finite target, matching [`pick_config`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn pick_config_from_packed(
    packed: &[(u16, f32)],
    target_zensim_a: f32,
) -> Result<Option<PickedConfig>, PackedPickError> {
    // If a feature the picker NEEDS was removed from THIS zenanalyze build, the
    // picker can't run here at all — not the caller's fault, so Ok(None)
    // (heuristic), same as pick_config's `resolve_features()?`.
    let Some(features) = resolve_features() else {
        return Ok(None);
    };
    let results =
        AnalysisResults::from_packed(packed).map_err(PackedPickError::Unpack)?;
    // Demand the pack carries every feature the picker needs — otherwise the
    // degenerate-guard would silently zero the absent ones and mis-pick.
    let mut needed = FeatureSet::new();
    for &f in &features {
        needed = needed.with(f);
    }
    results.require(needed).map_err(PackedPickError::Missing)?;
    let x = build_inputs_from_results(&results, target_zensim_a, &features);
    Ok(run_model(&x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_to_config_round_trips_the_taxonomy() {
        // Re-encode each config back to its index and confirm it equals the
        // original cell — i.e. cell_to_config is a bijection over the 36 cells.
        // (ChromaSubsampling/Effort aren't Hash, so map to discriminants.)
        let enc = |c: PickedConfig| -> usize {
            let sub = match c.subsampling {
                ChromaSubsampling::Quarter => 0,
                ChromaSubsampling::HalfHorizontal => 1,
                ChromaSubsampling::None => 2,
                _ => 99,
            };
            let eff = match c.effort {
                Effort::Fast => 0,
                Effort::Balanced => 1,
                Effort::Max => 2,
            };
            sub * 12 + (c.progressive as usize) * 6 + (c.sharp_yuv as usize) * 3 + eff
        };
        for cell in 0..N_CELLS {
            assert_eq!(enc(cell_to_config(cell)), cell, "cell {cell} round-trip");
        }
        // Spot-check the first + last labels against the toml ordering.
        let c0 = cell_to_config(0); // 420 | prog=false | sharp=false | effort=0
        assert_eq!(c0.subsampling, ChromaSubsampling::Quarter);
        assert!(!c0.progressive && !c0.sharp_yuv);
        assert_eq!(c0.effort, Effort::Fast);
        let c35 = cell_to_config(35); // 444 | prog=true | sharp=true | effort=2
        assert_eq!(c35.subsampling, ChromaSubsampling::None);
        assert!(c35.progressive && c35.sharp_yuv);
        assert_eq!(c35.effort, Effort::Max);
    }

    #[test]
    fn picker_features_resolve_by_name_forward_compatibly() {
        // The picker's 108 features must all resolve BY NAME against this
        // zenanalyze build — and the resolution must NOT require SUPPORTED to
        // have exactly 108 entries. That's the forward-compat contract: a
        // future zenanalyze can ADD features (SUPPORTED grows) without breaking
        // this bake, because we look up our features by name, not by position.
        let feats = resolve_features()
            .expect("all 108 picker features must resolve by name in this zenanalyze build");
        assert_eq!(feats.len(), N_FEATURES);
        // Resolution tolerates SUPPORTED being a strict superset (the whole
        // point): assert it works whenever SUPPORTED ⊇ the picker's set.
        assert!(
            FeatureSet::SUPPORTED.iter().count() >= N_FEATURES,
            "SUPPORTED must at least cover the picker's feature set"
        );
        // Every resolved feature is distinct (no name collision / duplicate).
        let uniq: std::collections::HashSet<_> = feats.iter().map(|f| f.id()).collect();
        assert_eq!(uniq.len(), N_FEATURES, "resolved features must be distinct");
    }

    #[test]
    fn bake_loads_and_predicts_a_valid_cell() {
        // A tiny synthetic RGB image must produce a valid config without
        // panicking — exercises feature extraction + the v3 forward + argmin.
        let (w, h) = (32u32, 32u32);
        let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 255) as u8).collect();
        let picked = pick_config(&rgb, w, h, 70.0);
        assert!(picked.is_some(), "picker should return a config for a 32x32 image");
    }

    #[test]
    fn out_of_bounds_target_zq_is_handled_without_panic_or_garbage() {
        let (w, h) = (32u32, 32u32);
        let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i % 255) as u8).collect();

        // Finite targets — including ones well outside the 0..100 dial —
        // saturate to the nearest edge and still yield a valid config. The MLP
        // never sees a wild zq_norm, so no NaN-driven mis-pick.
        for t in [-2000.0f32, -1.0, 0.0, 50.0, 100.0, 200.0, 1.0e9] {
            let picked = pick_config(&rgb, w, h, t);
            assert!(
                picked.is_some(),
                "finite OOB target {t} must saturate to the dial and yield a config",
            );
        }

        // The normalized target is always within [0,1] for finite inputs —
        // confirm the saturation directly so the contract is pinned, not just
        // the downstream pick.
        let feats = resolve_features().expect("features resolve");
        for (t, want) in [(-2000.0f32, 0.0f32), (100.0, 1.0), (200.0, 1.0), (50.0, 0.5)] {
            let x = build_inputs(&rgb, w, h, t, &feats);
            assert!(
                (x[N_FEATURES] - want).abs() < 1e-6,
                "target {t} → zq_norm {} (want {want})",
                x[N_FEATURES],
            );
        }

        // Non-finite targets are caller bugs: the picker returns None so the
        // encoder keeps its heuristic instead of silently picking a config off
        // a NaN forward pass.
        for t in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                pick_config(&rgb, w, h, t).is_none(),
                "non-finite target {t} must yield None (heuristic fallback)",
            );
        }
    }

    #[test]
    fn packed_path_picks_from_complete_pack_and_flags_bad_input() {
        // A COMPLETE pack — every needed feature id present — is accepted and
        // yields a config. Synthetic values exercise the plumbing without
        // depending on which features a given image populates.
        let feats = resolve_features().expect("features resolve");
        let complete: Vec<(u16, f32)> =
            feats.iter().enumerate().map(|(i, f)| (f.id(), i as f32 * 0.01)).collect();
        let picked = pick_config_from_packed(&complete, 70.0).expect("complete pack accepted");
        assert!(picked.is_some(), "a complete pack yields a config");

        // A pack missing a needed feature → Err(Missing) naming the absent id,
        // NOT a silent zero-fill.
        let missing_id = feats[0].id();
        let pruned: Vec<(u16, f32)> =
            complete.iter().copied().filter(|(id, _)| *id != missing_id).collect();
        match pick_config_from_packed(&pruned, 70.0) {
            Err(PackedPickError::Missing(m)) => {
                assert!(m.missing.contains(&missing_id), "must name the absent id")
            }
            other => panic!("expected Missing({missing_id}), got {other:?}"),
        }

        // A malformed pack (duplicate id) → Err(Unpack).
        let mut dup = complete.clone();
        dup.push(complete[0]);
        assert!(matches!(
            pick_config_from_packed(&dup, 70.0),
            Err(PackedPickError::Unpack(_))
        ));
    }

    #[test]
    fn packed_path_matches_fresh_analysis_round_trip() {
        // When a real analysis populates the full needed set, the packed pick is
        // identical to the fresh-analysis pick — proving pack→from_packed is a
        // faithful round-trip of the inputs the model sees. 256x256 textured
        // noise populates every picker feature (hard-asserted via `require`, so
        // a future feature that DOESN'T populate fails loudly rather than
        // silently skipping the comparison).
        let (w, h) = (256u32, 256);
        let rgb: Vec<u8> = (0..(w * h * 3)).map(|i| (i.wrapping_mul(7) % 251) as u8).collect();
        let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
        let results = zenanalyze::analyze_features_rgb8(&rgb, w, h, &query);

        let packed = results.pack();
        let from_packed = pick_config_from_packed(&packed, 70.0)
            .expect("256x256 analysis must populate the full picker feature set");
        let fresh = pick_config(&rgb, w, h, 70.0);
        assert_eq!(from_packed, fresh, "packed pick must match fresh-analysis pick");
    }
}
