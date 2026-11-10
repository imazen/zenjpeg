//! Crate-internal sanity checks — model parser, metadata parser,
//! argmin math. Round-trip tests that exercise bake → load → forward
//! live in `tests/roundtrip.rs`.

use crate::*;

#[test]
fn argmin_identity_no_offsets() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true, true, true, true, true];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(1));
}

#[test]
fn argmin_respects_mask() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true, false, true, true, true];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(3));
}

#[test]
fn argmin_empty_mask_returns_none() {
    let pred = [3.0f32, 1.0, 4.0];
    let mask = [false; 3];
    let m = AllowedMask::new(&mask);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, None);
}

#[test]
fn argmin_with_per_output_offsets_shifts_pick() {
    // Without offsets the lowest score is index 1 (=1.0).
    // With per_output adding +5 to index 1, the new lowest is 3 (=1.5).
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let offsets = ArgminOffsets {
        uniform: 0.0,
        per_output: Some(&[0.0, 5.0, 0.0, 0.0, 0.0]),
    };
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, Some(&offsets));
    assert_eq!(pick, Some(3));
}

#[test]
#[cfg(feature = "advanced")]
fn argmin_top_k_returns_sorted_indices() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(top, [Some(1), Some(3), Some(0)]);
}

#[test]
#[cfg(feature = "advanced")]
fn argmin_top_k_skips_nan_scores() {
    // Same documented NaN contract as `argmin_masked`: a NaN-scoring cell must never occupy a
    // top-K slot, even while the array still has room (count < K) — that's exactly the case
    // the prior bug missed, since the shift-loop's `top[i - 1].0 > score` never fires to
    // displace a NaN sitting in an empty slot.
    let pred = [f32::NAN, 3.0f32, 1.0, f32::NAN, 1.5];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(top, [Some(2), Some(4), Some(1)]); // 1.0, 1.5, 3.0 — NaNs never appear

    // All-NaN allowed set: every slot stays None (matches argmin_all_nan_returns_none).
    let all_nan = [f32::NAN; 4];
    let mask4 = [true; 4];
    let m4 = AllowedMask::new(&mask4);
    let top_all_nan =
        argmin::argmin_masked_top_k::<3>(&all_nan, &m4, ScoreTransform::Identity, None);
    assert_eq!(top_all_nan, [None, None, None]);
}

#[test]
#[cfg(feature = "advanced")]
fn pick_with_confidence_reports_gap() {
    let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
    let mask = [true; 5];
    let m = AllowedMask::new(&mask);
    let pick = argmin::pick_with_confidence(&pred, &m, ScoreTransform::Identity, None);
    let (idx, gap) = pick.unwrap();
    assert_eq!(idx, 1);
    assert!((gap - 0.5).abs() < 1e-6);
}

#[test]
#[cfg(feature = "advanced")]
fn pick_with_confidence_inf_when_only_one_allowed() {
    let pred = [3.0f32, 1.0, 4.0];
    let mask = [false, true, false];
    let m = AllowedMask::new(&mask);
    let (idx, gap) =
        argmin::pick_with_confidence(&pred, &m, ScoreTransform::Identity, None).unwrap();
    assert_eq!(idx, 1);
    assert!(gap.is_infinite());
}

#[test]
fn mask_at_least_finite_gate() {
    // INFINITY is non-finite → treated like NaN, gated out. The mask
    // is intentionally finite-only: a non-finite attribute (NaN or
    // INFINITY) conventionally means "missing data," never "always
    // passes."
    let rates = [0.99, 0.5, f32::NAN, 0.95, f32::INFINITY];
    let mut out = [false; 5];
    argmin::mask_at_least(&rates, 0.95, &mut out);
    assert_eq!(out, [true, false, false, true, false]);
}

#[test]
#[cfg(feature = "advanced")]
fn first_out_of_distribution_finds_first() {
    let bounds = [
        FeatureBound::new(0.0, 1.0),
        FeatureBound::new(-1.0, 1.0),
        FeatureBound::new(0.0, 100.0),
    ];
    assert_eq!(first_out_of_distribution(&[0.5, 0.0, 50.0], &bounds), None);
    assert_eq!(
        first_out_of_distribution(&[2.0, 0.0, 50.0], &bounds),
        Some(0)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, 0.0, f32::NAN], &bounds),
        Some(2)
    );
    assert_eq!(
        first_out_of_distribution(&[0.5, f32::INFINITY, 50.0], &bounds),
        Some(1)
    );
}

#[test]
fn metadata_empty_blob_yields_empty() {
    let m = Metadata::parse(&[]).unwrap();
    assert!(m.is_empty());
}

// --- Default-surface selection kit: top-K query + constraint masks ---
//
// Top-K and the runtime constraint masks (`mask_at_least` /
// `mask_at_most`) live on the default surface so the masking /
// score-transform / NaN / tie-break contract is defined once, not
// re-derived in each codec consumer.
mod topk_surface {
    use super::*;

    #[test]
    fn stable_top_k_sorted_indices() {
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let mask = [true; 5];
        let m = AllowedMask::new(&mask);
        // Free function.
        let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(1), Some(3), Some(0)]);
        // Re-exported at crate root under the same gate.
        let top2 = crate::argmin_masked_top_k::<2>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top2, [Some(1), Some(3)]);
    }

    #[test]
    fn stable_top_k_respects_mask_and_fills_none() {
        // Only 2 cells allowed but K=3 ⇒ third slot is None.
        let pred = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let mask = [false, true, false, true, false];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k::<3>(&pred, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(1), Some(3), None]);
    }

    #[test]
    fn stable_top_k_in_range_indices_are_local() {
        // Sub-range [2..5] of the output; returned indices are within
        // the sub-range (0..3), not absolute.
        let pred = [9.0f32, 9.0, 4.0, 1.5, 3.0];
        let mask = [true; 3];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k_in_range::<2>(
            &pred,
            (2, 5),
            &m,
            ScoreTransform::Identity,
            None,
        );
        // sub-slice = [4.0, 1.5, 3.0] ⇒ argmin order 1 (1.5), 2 (3.0).
        assert_eq!(top, [Some(1), Some(2)]);
    }

    #[test]
    fn stable_top_k_in_range_out_of_bounds_all_none() {
        let pred = [1.0f32, 2.0, 3.0];
        let mask = [true; 2];
        let m = AllowedMask::new(&mask);
        let top = argmin::argmin_masked_top_k_in_range::<2>(
            &pred,
            (2, 9), // end > len
            &m,
            ScoreTransform::Identity,
            None,
        );
        assert_eq!(top, [None, None]);
    }

    // --- constraint masks: mask_at_least (quality floor) +
    //     mask_at_most (perf/cost ceiling) ---

    #[test]
    fn mask_at_least_admits_floor_inclusive_nan_fails() {
        // Boundary: value == floor is ADMITTED; NaN never admitted.
        let quality = [0.99f32, 0.5, f32::NAN, 0.95];
        let mut out = [false; 4];
        crate::mask_at_least(&quality, 0.95, &mut out);
        assert_eq!(out, [true, false, false, true]);
    }

    #[test]
    fn mask_at_most_admits_ceiling_inclusive_nan_fails() {
        // Boundary: value == limit is ADMITTED; NaN (unknown cost) is
        // never admitted under a limit.
        let cost = [1.0f32, 8.0, f32::NAN, 3.0];
        let mut out = [false; 4];
        crate::mask_at_most(&cost, 3.0, &mut out);
        assert_eq!(out, [true, false, false, true]);
    }

    #[test]
    #[should_panic(expected = "mask_at_least")]
    fn mask_at_least_length_mismatch_panics() {
        let v = [1.0f32, 2.0, 3.0];
        let mut out = [false; 2]; // len 2 != 3
        crate::mask_at_least(&v, 1.0, &mut out);
    }

    #[test]
    #[should_panic(expected = "mask_at_most")]
    fn mask_at_most_length_mismatch_panics() {
        let v = [1.0f32, 2.0, 3.0];
        let mut out = [false; 2]; // len 2 != 3
        crate::mask_at_most(&v, 1.0, &mut out);
    }

    #[test]
    fn perf_limit_and_target_quality_compose_then_top_k() {
        // The intended runtime shape: rank by predicted cost (bytes),
        // but first constrain by a perf ceiling AND a quality floor —
        // both runtime inputs — by ANDing their masks. Cell 1 is the
        // cheapest by score but too slow (perf 8 > limit 3); cell 4 is
        // cheap + fast but below the quality target. So the admissible
        // cheapest is cell 3.
        let scores = [3.0f32, 1.0, 4.0, 1.5, 0.5]; // predicted cost (bytes)
        let perf = [1.0f32, 8.0, 2.0, 3.0, 1.0]; // encode cost
        let quality = [0.97f32, 0.99, 0.96, 0.98, 0.80]; // predicted quality

        let mut gate_perf = [false; 5];
        let mut gate_qual = [false; 5];
        crate::mask_at_most(&perf, 3.0, &mut gate_perf); // perf <= 3
        crate::mask_at_least(&quality, 0.95, &mut gate_qual); // quality >= 0.95

        // AND the two runtime-constraint masks together.
        let mut gate = [false; 5];
        for ((g, &p), &q) in gate.iter_mut().zip(gate_perf.iter()).zip(gate_qual.iter()) {
            *g = p && q;
        }
        // perf<=3: [T,F,T,T,T]; qual>=.95: [T,T,T,T,F] ⇒ AND [T,F,T,T,F]
        assert_eq!(gate, [true, false, true, true, false]);

        let m = AllowedMask::new(&gate);
        let pick = argmin::argmin_masked(&scores, &m, ScoreTransform::Identity, None);
        assert_eq!(pick, Some(3)); // cheapest admissible (score 1.5)

        // Top-K over the same admissible set, for an encode-verify pass.
        let top = argmin::argmin_masked_top_k::<2>(&scores, &m, ScoreTransform::Identity, None);
        assert_eq!(top, [Some(3), Some(0)]); // 1.5 then 3.0
    }
}

#[test]
fn argmin_nan_score_is_silently_skipped() {
    // NaN never compares less than a finite value, so a cell with
    // NaN score is never picked. Documented contract.
    let pred = [3.0_f32, f32::NAN, 1.0];
    let m = AllowedMask::new(&[true, true, true]);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, Some(2));
}

#[test]
fn argmin_all_nan_returns_none() {
    // If every allowed cell scores NaN, returns None — same as no
    // allowed cells. Callers needing the distinction must
    // pre-validate.
    let pred = [f32::NAN, f32::NAN, f32::NAN];
    let m = AllowedMask::new(&[true, true, true]);
    let pick = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
    assert_eq!(pick, None);
}

#[test]
fn argmin_ties_prefer_lowest_index() {
    // Documented tie-break: lowest index wins (uses `<`, not `<=`).
    let pred = [5.0_f32, 5.0, 5.0];
    let m = AllowedMask::new(&[true, true, true]);
    assert_eq!(
        argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None),
        Some(0)
    );
}

#[test]
#[should_panic(expected = "mask.len()")]
fn argmin_short_mask_panics() {
    // mask.len() < predictions.len() panics in both debug and
    // release. Used to silently deny high-index cells, which masked
    // bugs.
    let pred = [3.0_f32, 1.0, 4.0];
    let m = AllowedMask::new(&[true, true]); // len 2 < 3
    let _ = argmin::argmin_masked(&pred, &m, ScoreTransform::Identity, None);
}

#[test]
#[should_panic(expected = "mask.len()")]
#[cfg(feature = "advanced")]
fn argmin_with_scorer_short_mask_panics() {
    let m = AllowedMask::new(&[true, true]); // len 2 < n=3
    let _ = argmin::argmin_masked_with_scorer(3, &m, |i| i as f32);
}

#[test]
#[cfg(feature = "advanced")]
fn rescue_default_threshold_three_pp() {
    let policy = RescuePolicy::default();
    assert!((policy.rescue_threshold - 3.0).abs() < f32::EPSILON);
}
