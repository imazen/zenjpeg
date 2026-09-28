//! Experimental source-aware table selection. No encoder defaults depend on it.

use std::collections::BTreeMap;

pub type Histogram = BTreeMap<i16, u64>;

/// Requantize in integer arithmetic, with signed midpoint ties away from zero.
pub fn requantize(k: i16, old: u16, new: u16) -> i64 {
    assert!(old > 0 && new > 0);
    let x = i64::from(k) * i64::from(old);
    let magnitude = (x.abs() + i64::from(new) / 2) / i64::from(new);
    magnitude * x.signum()
}

pub fn distortion(hist: &Histogram, old: u16, new: u16) -> f64 {
    let count: u64 = hist.values().sum();
    if count == 0 {
        return 0.0;
    }
    hist.iter()
        .map(|(&k, &n)| {
            let delta = requantize(k, old, new) * i64::from(new) - i64::from(k) * i64::from(old);
            n as f64 * (delta as f64).powi(2)
        })
        .sum::<f64>()
        / count as f64
}

/// Table-only prior; deliberately independent of the image being evaluated.
/// This is a heuristic distribution of stored levels, not a reconstruction
/// of the unknown original coefficient distribution.
pub fn prior() -> Histogram {
    (-16i16..=16)
        .map(|k| (k, 17 - u64::from(k.unsigned_abs())))
        .collect()
}

/// Search legal steps within a fractional window of the destination step.
/// Fixed target units prevent candidates from redefining their error scale.
/// The log-step penalty retains the destination's spectral weighting; this
/// is candidate generation, NOT an estimate of actual JPEG rate.
pub fn select(old: u16, target: u16, hist: &Histogram, window: f64, strength: f64) -> u16 {
    assert!(old > 0 && (1..=255).contains(&target));
    assert!(window.is_finite() && (0.0..=0.5).contains(&window));
    assert!(strength.is_finite() && strength >= 0.0);
    let lo = (f64::from(target) * (1.0 - window)).ceil().max(1.0) as u16;
    let hi = (f64::from(target) * (1.0 + window)).floor().min(255.0) as u16;
    let cost = |b| {
        distortion(hist, old, b) / f64::from(target).powi(2)
            + strength * (f64::from(b) / f64::from(target)).ln().powi(2)
    };
    let mut best = target;
    let mut best_cost = cost(target);
    for b in lo..=hi {
        let candidate = cost(b);
        if candidate < best_cost {
            best = b;
            best_cost = candidate;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisors_preserve_dequantized_values_including_extremes() {
        for old in [1u16, 12, 255, 256, 32767, 65535] {
            for new in (1..=old).filter(|b| old % b == 0) {
                for k in [i16::MIN, -1024, -3, -1, 0, 1, 3, 1023, i16::MAX] {
                    assert_eq!(
                        requantize(k, old, new) * i64::from(new),
                        i64::from(k) * i64::from(old)
                    );
                }
            }
        }
    }

    #[test]
    fn midpoint_ties_are_symmetric_and_away_from_zero() {
        assert_eq!(requantize(1, 12, 24), 1);
        assert_eq!(requantize(-1, 12, 24), -1);
        assert_eq!(requantize(3, 12, 24), 2);
        assert_eq!(requantize(-3, 12, 24), -2);
    }

    #[test]
    fn zero_frequencies_keep_target_and_window_is_respected() {
        let zeros = [(0, 100)].into_iter().collect();
        for target in 1..=255 {
            assert_eq!(select(12, target, &zeros, 0.2, 1.0), target);
            let selected = select(12, target, &prior(), 0.2, 1.0);
            assert!((1..=255).contains(&selected));
            assert!(f64::from(selected) >= (f64::from(target) * 0.8).ceil());
            assert!(f64::from(selected) <= (f64::from(target) * 1.2).floor());
        }
    }

    #[test]
    fn finer_candidate_can_choose_an_exact_divisor() {
        assert_eq!(select(12, 5, &prior(), 0.2, 0.0), 4);
        assert_eq!(distortion(&prior(), 12, 6), 0.0);
        assert!(distortion(&prior(), 12, 5) > 0.0);
    }
}
