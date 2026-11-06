//! Directed multi-shot search over ranked trial candidates.
//!
//! The codec's trial loop shouldn't blindly encode the picker's top-K and take the max.
//! It should navigate toward the caller's RD target using the over/undershoot of the
//! trials it has already run:
//! - [`next_trial`] picks the next candidate to encode (or stops) by bracketing the
//!   target on the candidates' *predicted* quality — probe leaner after a reach, higher
//!   after an undershoot.
//! - [`best_trial`] picks the winner among the *measured* trials, preference-aware
//!   (reach the quality then minimize bytes, or fit the bytes then maximize quality).
//!
//! Pure + alloc-free: the codec owns the loop and the `done` slice; this only decides.
//! `predicted_zq[c]` is the picker's predicted quality for ranked candidate `c` (read
//! from the predictor's scores at the `argmin_masked_top_k` indices).

use core::cmp::Ordering;

/// The caller's RD target + preference for the multi-shot search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QualityTarget {
    /// Reach at least `target_zq`; among reaching trials, fewest bytes wins.
    Quality { target_zq: f32 },
    /// Stay within `max_bytes`; among fitting trials, highest quality wins.
    Bytes { max_bytes: u64 },
}

/// A completed trial encode: which ranked candidate, and what it achieved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trial {
    /// Index into the ranked candidate list.
    pub candidate: usize,
    pub achieved_zq: f32,
    pub bytes: u64,
}

#[inline]
fn by_zq(zq: &[f32]) -> impl Fn(&usize, &usize) -> Ordering + '_ {
    move |&a, &b| zq[a].total_cmp(&zq[b])
}

/// The best completed trial per `target` (over/undershoot-aware). Returns the index
/// into `done`, or `None` if `done` is empty.
pub fn best_trial(done: &[Trial], target: QualityTarget) -> Option<usize> {
    if done.is_empty() {
        return None;
    }
    let idx = match target {
        QualityTarget::Quality { target_zq } => (0..done.len())
            .filter(|&i| done[i].achieved_zq >= target_zq)
            .min_by_key(|&i| done[i].bytes) // reaching -> fewest bytes
            .unwrap_or_else(|| {
                // none reach -> the highest achieved quality (best effort)
                (0..done.len())
                    .max_by(|&a, &b| done[a].achieved_zq.total_cmp(&done[b].achieved_zq))
                    .unwrap()
            }),
        QualityTarget::Bytes { max_bytes } => (0..done.len())
            .filter(|&i| done[i].bytes <= max_bytes)
            .max_by(|&a, &b| done[a].achieved_zq.total_cmp(&done[b].achieved_zq)) // fitting -> max quality
            .unwrap_or_else(|| {
                // none fit -> the smallest output (best effort)
                (0..done.len()).min_by_key(|&i| done[i].bytes).unwrap()
            }),
    };
    Some(idx)
}

/// The next candidate to trial, directed by the over/undershoot of `done` against
/// `target`. Returns an untried candidate index, or `None` to stop (the target is
/// bracketed or candidates are exhausted). Alloc-free.
pub fn next_trial(predicted_zq: &[f32], done: &[Trial], target: QualityTarget) -> Option<usize> {
    let n = predicted_zq.len();
    if n == 0 {
        return None;
    }
    let tried = |c: usize| done.iter().any(|t| t.candidate == c);

    if done.is_empty() {
        // First probe: the candidate whose predicted quality best brackets the target.
        return match target {
            QualityTarget::Quality { target_zq } => (0..n)
                .filter(|&c| predicted_zq[c] >= target_zq)
                .min_by(by_zq(predicted_zq)) // leanest predicted to reach
                .or_else(|| (0..n).max_by(by_zq(predicted_zq))), // else the best shot
            QualityTarget::Bytes { .. } => (0..n).max_by(by_zq(predicted_zq)), // start high, walk leaner
        };
    }

    // `done` is caller-supplied; a `Trial::candidate` past `predicted_zq`'s bounds (a stale
    // index, a mismatched candidate list between calls, etc.) must not panic here — `.get()`
    // treats an out-of-range trial as carrying no predicted-quality signal, simply excluded
    // from the fold, rather than indexing straight into `predicted_zq`.
    match target {
        QualityTarget::Quality { target_zq } => {
            let reaching_pred = done
                .iter()
                .filter(|t| t.achieved_zq >= target_zq)
                .filter_map(|t| predicted_zq.get(t.candidate).copied())
                .fold(f32::INFINITY, f32::min);
            if reaching_pred.is_finite() {
                // have a reach -> probe a leaner untried (predicted below it): may also reach, fewer bytes
                (0..n)
                    .filter(|&c| !tried(c) && predicted_zq[c] < reaching_pred)
                    .max_by(by_zq(predicted_zq))
            } else {
                // no reach yet -> probe a higher untried, closest above the highest tried
                let max_tried = done
                    .iter()
                    .filter_map(|t| predicted_zq.get(t.candidate).copied())
                    .fold(f32::NEG_INFINITY, f32::max);
                (0..n)
                    .filter(|&c| !tried(c) && predicted_zq[c] > max_tried)
                    .min_by(by_zq(predicted_zq))
            }
        }
        QualityTarget::Bytes { max_bytes } => {
            if done.iter().any(|t| t.bytes <= max_bytes) {
                // already fitting -> probe a higher-quality untried that might still fit
                let best_fit_pred = done
                    .iter()
                    .filter(|t| t.bytes <= max_bytes)
                    .filter_map(|t| predicted_zq.get(t.candidate).copied())
                    .fold(f32::NEG_INFINITY, f32::max);
                (0..n)
                    .filter(|&c| !tried(c) && predicted_zq[c] > best_fit_pred)
                    .min_by(by_zq(predicted_zq))
            } else {
                // nothing fits -> probe a leaner untried (lower predicted quality -> fewer bytes)
                let min_tried = done
                    .iter()
                    .filter_map(|t| predicted_zq.get(t.candidate).copied())
                    .fold(f32::INFINITY, f32::min);
                (0..n)
                    .filter(|&c| !tried(c) && predicted_zq[c] < min_tried)
                    .max_by(by_zq(predicted_zq))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn t(candidate: usize, zq: f32, bytes: u64) -> Trial {
        Trial {
            candidate,
            achieved_zq: zq,
            bytes,
        }
    }

    #[test]
    fn best_quality_prefers_reaching_min_bytes() {
        let done = [t(0, 72.0, 5000), t(1, 71.0, 4000), t(2, 68.0, 3000)];
        // target 70: trials 0 & 1 reach; fewest bytes among them = trial 1
        assert_eq!(
            best_trial(&done, QualityTarget::Quality { target_zq: 70.0 }),
            Some(1)
        );
    }
    #[test]
    fn best_quality_none_reach_returns_highest() {
        let done = [t(0, 60.0, 5000), t(1, 65.0, 4000)];
        assert_eq!(
            best_trial(&done, QualityTarget::Quality { target_zq: 70.0 }),
            Some(1)
        );
    }
    #[test]
    fn best_bytes_prefers_fitting_max_quality() {
        let done = [t(0, 72.0, 5000), t(1, 70.0, 3000), t(2, 71.0, 3500)];
        // budget 4000: trials 1 & 2 fit; max quality = trial 2
        assert_eq!(
            best_trial(&done, QualityTarget::Bytes { max_bytes: 4000 }),
            Some(2)
        );
    }
    #[test]
    fn first_probe_is_leanest_predicted_to_reach() {
        let pred = [75.0, 71.0, 68.0];
        assert_eq!(
            next_trial(&pred, &[], QualityTarget::Quality { target_zq: 70.0 }),
            Some(1)
        );
    }
    #[test]
    fn directs_leaner_after_a_reach() {
        let pred = [75.0, 71.0, 68.0];
        let done = [t(1, 72.0, 4000)]; // reached -> probe leaner untried (pred < 71) = cand 2
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Quality { target_zq: 70.0 }),
            Some(2)
        );
    }
    #[test]
    fn directs_higher_after_undershoot() {
        let pred = [75.0, 71.0, 68.0];
        let done = [t(2, 67.0, 3000)]; // undershot -> probe higher untried, closest above 68 = cand 1
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Quality { target_zq: 70.0 }),
            Some(1)
        );
    }
    #[test]
    fn stops_when_bracketed() {
        let pred = [71.0];
        let done = [t(0, 72.0, 4000)];
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Quality { target_zq: 70.0 }),
            None
        );
    }

    // Prior bug: `done` (caller-supplied) with a `Trial::candidate` past
    // `predicted_zq`'s bounds indexed straight into `predicted_zq[t.candidate]` and
    // panicked. An out-of-range trial must instead be treated as carrying no
    // predicted-quality signal (excluded from the fold), not crash the search.
    #[test]
    fn out_of_bounds_candidate_in_reach_fold_does_not_panic() {
        let pred = [75.0, 71.0, 68.0];
        // candidate 99 is out of bounds for `pred` (len 3) but reaches the target;
        // candidate 0 also reaches, so reaching_pred must come from candidate 0 alone.
        let done = [t(0, 72.0, 4000), t(99, 80.0, 1000)];
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Quality { target_zq: 70.0 }),
            Some(1) // leaner untried below 75.0 -> candidate 1 (71.0)
        );
    }

    #[test]
    fn out_of_bounds_candidate_in_no_reach_fold_does_not_panic() {
        let pred = [75.0, 71.0, 68.0];
        // neither trial reaches target_zq=70; candidate 99 is out of bounds.
        let done = [t(1, 60.0, 4000), t(99, 50.0, 999)];
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Quality { target_zq: 70.0 }),
            Some(0) // probe higher than the highest valid tried (71.0) -> candidate 0 (75.0)
        );
    }

    #[test]
    fn out_of_bounds_candidate_in_bytes_fit_fold_does_not_panic() {
        let pred = [75.0, 71.0, 68.0];
        // candidate 1 fits the byte budget; candidate 99 (out of bounds) also "fits".
        let done = [t(1, 71.0, 3000), t(99, 999.0, 1)];
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Bytes { max_bytes: 4000 }),
            Some(0) // probe higher-quality untried above the valid best-fit (71.0) -> candidate 0
        );
    }

    #[test]
    fn out_of_bounds_candidate_in_bytes_no_fit_fold_does_not_panic() {
        let pred = [75.0, 71.0, 68.0];
        // neither trial fits the tiny byte budget; candidate 99 is out of bounds.
        let done = [t(0, 72.0, 9000), t(99, 1.0, 10000)];
        assert_eq!(
            next_trial(&pred, &done, QualityTarget::Bytes { max_bytes: 100 }),
            Some(1) // probe leaner than the lowest valid tried (75.0) -> candidate 1 (71.0)
        );
    }
}
