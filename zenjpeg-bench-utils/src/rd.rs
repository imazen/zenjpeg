//! Rate-distortion curve construction and comparison utilities.
//!
//! This module provides primitives for comparing two encoder configurations
//! on a rate-distortion basis:
//!
//! - [`RdPoint`] — a single (rate, distortion, quality) sample.
//! - [`RdCurve`] — a collection of points, with Pareto-hull extraction.
//! - [`bd_rate`] — Bjøntegaard delta rate: the average relative rate
//!   saving (%) the candidate curve gives over the baseline curve across
//!   the overlap region of log-rate in distortion space. Negative means the
//!   candidate is better.
//! - [`closest_point_distance`] — signed perpendicular distance of a single
//!   candidate point to the baseline curve in `(log_rate, distortion)`
//!   space. Positive = candidate dominates (lower distortion than the
//!   curve would predict at that rate).
//! - [`RdComparison`] — aggregate of both metrics for a single
//!   image / metric / config pair.
//!
//! # Distortion convention
//!
//! **SMALLER DISTORTION IS BETTER**, always. Adapt your metric to this
//! convention before constructing curves:
//!
//! | Metric | Better direction | Distortion |
//! |--------|------------------|-----------|
//! | SSIMULACRA2 | higher (100 = identical) | `100.0 - score` |
//! | BBS (this crate) | lower (0 = identical) | `total` directly |
//! | Butteraugli | lower (0 = identical) | score directly |
//! | DSSIM | lower (0 = identical) | score directly |
//! | PSNR | higher | `-score` (but don't use PSNR) |
//!
//! # BD-rate
//!
//! The implementation uses piecewise-cubic interpolation (natural cubic
//! spline) in `(log10(rate_bpp), distortion)` space, following the
//! original Bjøntegaard 2001 formulation (VCEG-M33). When the spline is
//! ill-conditioned (too few points, or distortion non-monotonic after
//! Pareto hull), the function falls back to piecewise-linear integration,
//! which is still a valid BD-rate — just less smooth.
//!
//! Both implementations integrate over the overlapping distortion interval
//! `[max(dist_min_baseline, dist_min_candidate),
//!   min(dist_max_baseline, dist_max_candidate)]`.
//! If the overlap is empty (or spans fewer than two points on either
//! curve), [`bd_rate`] returns `None`.

// (alloc types are re-exported through std; no explicit imports needed)

/// A single rate-distortion sample.
///
/// `rate_bpp` is bits-per-pixel. `distortion` is whichever scalar is being
/// measured, in "smaller is better" orientation — see the
/// [module docs][self] for the convention by metric. `quality` is the
/// encoder-side input parameter that produced this point (usually JPEG
/// quality 0–100), retained for traceability back to the sweep config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdPoint {
    /// Bits per pixel of the encoded output.
    pub rate_bpp: f64,
    /// Distortion, "smaller is better" orientation.
    pub distortion: f64,
    /// Encoder quality parameter that produced this sample (e.g. JPEG Q).
    pub quality: u8,
}

/// Ordered set of rate-distortion points.
///
/// Points are sorted by `rate_bpp` ascending on construction. Duplicate
/// rates are retained (pareto_hull will collapse them). Points that are
/// non-finite or have non-positive rate are dropped with a warning-level
/// silent filter — construction never panics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RdCurve {
    /// Points in rate-ascending order.
    pub points: Vec<RdPoint>,
}

impl RdCurve {
    /// Build a curve from an arbitrary iterator of points. Filters out
    /// non-finite and non-positive-rate points, then sorts by rate.
    pub fn from_points<I: IntoIterator<Item = RdPoint>>(iter: I) -> Self {
        let mut points: Vec<RdPoint> = iter
            .into_iter()
            .filter(|p| {
                p.rate_bpp.is_finite()
                    && p.distortion.is_finite()
                    && p.rate_bpp > 0.0
                    && p.distortion >= 0.0
            })
            .collect();
        points.sort_by(|a, b| a.rate_bpp.partial_cmp(&b.rate_bpp).unwrap());
        Self { points }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if the curve has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return the lower-convex-hull subset of points in `(rate, distortion)`
    /// space — i.e. the Pareto frontier when minimising both rate and
    /// distortion.
    ///
    /// Points that are dominated by some other point (there exists another
    /// point with both lower-or-equal rate and lower-or-equal distortion,
    /// and strictly smaller in at least one) are dropped. Among points
    /// with identical rate, only the one with the smallest distortion is
    /// retained.
    ///
    /// The returned curve's points are in rate-ascending, distortion-
    /// descending order (monotone-decreasing distortion as rate grows).
    /// This is the correct shape for BD-rate interpolation.
    pub fn pareto_hull(&self) -> RdCurve {
        if self.points.is_empty() {
            return RdCurve::default();
        }
        // Collapse duplicates-by-rate, keeping the smallest distortion.
        let mut dedup: Vec<RdPoint> = Vec::with_capacity(self.points.len());
        for p in &self.points {
            if let Some(last) = dedup.last_mut() {
                if (last.rate_bpp - p.rate_bpp).abs() < f64::EPSILON {
                    if p.distortion < last.distortion {
                        *last = *p;
                    }
                    continue;
                }
            }
            dedup.push(*p);
        }
        // Walk rate-ascending: retain only points where distortion is
        // strictly less than every retained point so far. This is the
        // lower-envelope trace.
        let mut hull: Vec<RdPoint> = Vec::with_capacity(dedup.len());
        let mut best_distortion = f64::INFINITY;
        for p in &dedup {
            if p.distortion < best_distortion {
                hull.push(*p);
                best_distortion = p.distortion;
            }
        }
        RdCurve { points: hull }
    }
}

/// Aggregated comparison of a candidate curve against a baseline curve.
#[derive(Debug, Clone, PartialEq)]
pub struct RdComparison {
    /// Bjøntegaard delta rate, percentage. Negative = candidate is better
    /// (uses less rate for equal quality). `None` when curves don't
    /// overlap enough for integration.
    pub bd_rate: Option<f64>,
    /// Mean of the per-point signed perpendicular distances (positive =
    /// candidate below the baseline curve in distortion, i.e. better).
    /// Computed from [`per_point_distances`](Self::per_point_distances).
    pub mean_distance: f64,
    /// Fraction of candidate points that strictly dominate the baseline
    /// curve (positive distance > 0). In `[0, 1]`.
    pub win_rate: f64,
    /// Signed distances of each candidate point to the baseline curve,
    /// in the same order as the input candidate curve.
    pub per_point_distances: Vec<f64>,
}

/// Compute the signed perpendicular distance from `candidate_pt` to the
/// baseline curve in `(log10(rate), distortion)` space.
///
/// `baseline` should have already been reduced to its Pareto hull.
/// Positive = candidate has lower distortion than the baseline curve
/// interpolates at the candidate's rate.
///
/// When the candidate's rate falls outside the baseline's rate range,
/// the baseline is linearly extrapolated from its nearest two points in
/// log-rate space. Returns `0.0` if the baseline has fewer than 2 points.
pub fn closest_point_distance(baseline: &RdCurve, candidate_pt: &RdPoint) -> f64 {
    if baseline.points.len() < 2 {
        return 0.0;
    }
    let log_r = candidate_pt.rate_bpp.log10();
    let pts = &baseline.points;
    // Find the baseline distortion at candidate's rate via linear
    // interpolation in log-rate space.
    let baseline_distortion = {
        let first = &pts[0];
        let last = &pts[pts.len() - 1];
        if log_r <= first.rate_bpp.log10() {
            let p0 = first;
            let p1 = &pts[1];
            let x0 = p0.rate_bpp.log10();
            let x1 = p1.rate_bpp.log10();
            let t = (log_r - x0) / (x1 - x0);
            p0.distortion + t * (p1.distortion - p0.distortion)
        } else if log_r >= last.rate_bpp.log10() {
            let p0 = &pts[pts.len() - 2];
            let p1 = last;
            let x0 = p0.rate_bpp.log10();
            let x1 = p1.rate_bpp.log10();
            let t = (log_r - x0) / (x1 - x0);
            p0.distortion + t * (p1.distortion - p0.distortion)
        } else {
            // Binary search for the bracketing interval.
            let mut lo = 0usize;
            let mut hi = pts.len() - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if pts[mid].rate_bpp.log10() <= log_r {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let p0 = &pts[lo];
            let p1 = &pts[hi];
            let x0 = p0.rate_bpp.log10();
            let x1 = p1.rate_bpp.log10();
            let t = (log_r - x0) / (x1 - x0);
            p0.distortion + t * (p1.distortion - p0.distortion)
        }
    };
    // Signed: positive when candidate has lower distortion than the curve.
    baseline_distortion - candidate_pt.distortion
}

/// Natural cubic spline through a strictly-monotone-X set of points.
/// Returns `None` if `xs` is not strictly increasing or has < 3 points.
fn natural_cubic_spline_coeffs(xs: &[f64], ys: &[f64]) -> Option<Vec<(f64, f64, f64, f64)>> {
    let n = xs.len();
    if n < 3 || xs.len() != ys.len() {
        return None;
    }
    for w in xs.windows(2) {
        if w[1] <= w[0] {
            return None;
        }
    }
    let mut h = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = xs[i + 1] - xs[i];
    }
    // Solve tridiagonal system for second derivatives (natural BC: m[0]=m[n-1]=0).
    let mut alpha = vec![0.0; n];
    for i in 1..n - 1 {
        alpha[i] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
    }
    let mut l = vec![0.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];
    l[0] = 1.0;
    for i in 1..n - 1 {
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
        if l[i].abs() < 1e-14 {
            return None;
        }
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n - 1] = 1.0;
    let mut c = vec![0.0; n];
    let mut b = vec![0.0; n - 1];
    let mut d = vec![0.0; n - 1];
    for j in (0..n - 1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
    }
    // Per-segment coefficients (a, b, c, d): s(x) = a + b*(x-xj) + c*(x-xj)^2 + d*(x-xj)^3.
    let mut out = Vec::with_capacity(n - 1);
    for j in 0..n - 1 {
        out.push((ys[j], b[j], c[j], d[j]));
    }
    Some(out)
}

/// Integrate a cubic-spline segment `s(x) = a + b*(x-xj) + c*(x-xj)^2 + d*(x-xj)^3`
/// over `[x0, x1]` relative to the segment anchor `xj`.
fn integrate_cubic_segment(coef: (f64, f64, f64, f64), xj: f64, x0: f64, x1: f64) -> f64 {
    let (a, b, c, d) = coef;
    let t1 = x1 - xj;
    let t0 = x0 - xj;
    let eval = |t: f64| a * t + b * t * t / 2.0 + c * t * t * t / 3.0 + d * t * t * t * t / 4.0;
    eval(t1) - eval(t0)
}

/// Integrate a cubic spline (as produced by [`natural_cubic_spline_coeffs`])
/// over `[x_lo, x_hi]`. Returns `None` if the range lies outside the knot
/// span.
fn integrate_spline(
    xs: &[f64],
    coeffs: &[(f64, f64, f64, f64)],
    x_lo: f64,
    x_hi: f64,
) -> Option<f64> {
    if x_lo >= x_hi {
        return None;
    }
    let xmin = *xs.first()?;
    let xmax = *xs.last()?;
    if x_lo < xmin - 1e-12 || x_hi > xmax + 1e-12 {
        return None;
    }
    let x_lo = x_lo.max(xmin);
    let x_hi = x_hi.min(xmax);
    let mut acc = 0.0;
    for j in 0..coeffs.len() {
        let seg_lo = xs[j];
        let seg_hi = xs[j + 1];
        if seg_hi <= x_lo {
            continue;
        }
        if seg_lo >= x_hi {
            break;
        }
        let a = seg_lo.max(x_lo);
        let b = seg_hi.min(x_hi);
        acc += integrate_cubic_segment(coeffs[j], seg_lo, a, b);
    }
    Some(acc)
}

/// Piecewise-linear integral fallback — same interface as
/// [`integrate_spline`] but uses linear interpolation between knots.
fn integrate_piecewise_linear(xs: &[f64], ys: &[f64], x_lo: f64, x_hi: f64) -> Option<f64> {
    if xs.len() < 2 || xs.len() != ys.len() || x_lo >= x_hi {
        return None;
    }
    let xmin = *xs.first()?;
    let xmax = *xs.last()?;
    if x_lo < xmin - 1e-12 || x_hi > xmax + 1e-12 {
        return None;
    }
    let x_lo = x_lo.max(xmin);
    let x_hi = x_hi.min(xmax);
    let eval_at = |x: f64| -> f64 {
        // Interpolate y at x via binary search over xs.
        let mut lo = 0;
        let mut hi = xs.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if xs[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
        ys[lo] + t * (ys[hi] - ys[lo])
    };
    let mut acc = 0.0;
    // Integrate over segments, clipping to [x_lo, x_hi].
    for j in 0..xs.len() - 1 {
        let seg_lo = xs[j];
        let seg_hi = xs[j + 1];
        if seg_hi <= x_lo {
            continue;
        }
        if seg_lo >= x_hi {
            break;
        }
        let a = seg_lo.max(x_lo);
        let b = seg_hi.min(x_hi);
        let ya = eval_at(a);
        let yb = eval_at(b);
        acc += 0.5 * (ya + yb) * (b - a);
    }
    Some(acc)
}

/// Compute Bjøntegaard delta rate between `baseline` and `candidate`.
///
/// Both curves are internally reduced to their Pareto hulls before
/// integration. We integrate log10(rate) as a function of distortion
/// (i.e. `x = distortion`, `y = log10(rate)`), which is the orientation
/// BD-rate uses. The value returned is:
///
/// ```text
/// BD-rate = (10^avg_log_rate_diff - 1) * 100.0
/// ```
///
/// in percent. Negative means the candidate curve needs less rate at
/// equal distortion, averaged across the overlap interval.
///
/// Returns `None` when:
/// - Either curve has fewer than 2 Pareto-hull points.
/// - Distortion ranges don't overlap (no common x-interval).
/// - Cubic-spline fits fail AND piecewise-linear fallback fails.
pub fn bd_rate(baseline: &RdCurve, candidate: &RdCurve) -> Option<f64> {
    let base = baseline.pareto_hull();
    let cand = candidate.pareto_hull();
    if base.points.len() < 2 || cand.points.len() < 2 {
        return None;
    }
    // Use distortion as x, log10(rate) as y, and require x strictly
    // increasing. Pareto-hull already gives us distortion strictly
    // decreasing with increasing rate; flip to get distortion ascending.
    let to_xy = |c: &RdCurve| -> (Vec<f64>, Vec<f64>) {
        let mut xs: Vec<f64> = c.points.iter().rev().map(|p| p.distortion).collect();
        let mut ys: Vec<f64> = c.points.iter().rev().map(|p| p.rate_bpp.log10()).collect();
        // Remove any runs of equal-x (shouldn't happen post-hull with unique
        // rates, but guard anyway) by nudging.
        for i in 1..xs.len() {
            if xs[i] <= xs[i - 1] {
                xs[i] = xs[i - 1] + 1e-9;
            }
        }
        // Also drop any trailing non-finite.
        let mut keep = xs.len();
        while keep > 0 && !(ys[keep - 1].is_finite() && xs[keep - 1].is_finite()) {
            keep -= 1;
        }
        xs.truncate(keep);
        ys.truncate(keep);
        (xs, ys)
    };
    let (bx, by) = to_xy(&base);
    let (cx, cy) = to_xy(&cand);
    if bx.len() < 2 || cx.len() < 2 {
        return None;
    }
    let x_lo = bx[0].max(cx[0]);
    let x_hi = bx[bx.len() - 1].min(cx[cx.len() - 1]);
    if x_hi <= x_lo + 1e-12 {
        return None;
    }
    let span = x_hi - x_lo;
    // Try cubic first on both curves; fall back to linear on either side.
    let base_int = natural_cubic_spline_coeffs(&bx, &by)
        .and_then(|c| integrate_spline(&bx, &c, x_lo, x_hi))
        .or_else(|| integrate_piecewise_linear(&bx, &by, x_lo, x_hi))?;
    let cand_int = natural_cubic_spline_coeffs(&cx, &cy)
        .and_then(|c| integrate_spline(&cx, &c, x_lo, x_hi))
        .or_else(|| integrate_piecewise_linear(&cx, &cy, x_lo, x_hi))?;
    let avg_log_rate_diff = (cand_int - base_int) / span;
    let bd = (10.0_f64.powf(avg_log_rate_diff) - 1.0) * 100.0;
    if bd.is_finite() { Some(bd) } else { None }
}

/// Aggregate the candidate vs baseline comparison into per-point
/// distances, mean distance, win rate, and BD-rate.
pub fn compare(baseline: &RdCurve, candidate: &RdCurve) -> RdComparison {
    let base_hull = baseline.pareto_hull();
    let per_point_distances: Vec<f64> = candidate
        .points
        .iter()
        .map(|p| closest_point_distance(&base_hull, p))
        .collect();
    let (sum, wins) = per_point_distances.iter().fold((0.0, 0), |(s, w), d| {
        (s + d, w + if *d > 0.0 { 1 } else { 0 })
    });
    let mean_distance = if per_point_distances.is_empty() {
        0.0
    } else {
        sum / per_point_distances.len() as f64
    };
    let win_rate = if per_point_distances.is_empty() {
        0.0
    } else {
        wins as f64 / per_point_distances.len() as f64
    };
    RdComparison {
        bd_rate: bd_rate(baseline, candidate),
        mean_distance,
        win_rate,
        per_point_distances,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(r: f64, d: f64, q: u8) -> RdPoint {
        RdPoint {
            rate_bpp: r,
            distortion: d,
            quality: q,
        }
    }

    #[test]
    fn rd_curve_filters_and_sorts() {
        let curve = RdCurve::from_points([
            pt(2.0, 1.0, 90),
            pt(1.0, 5.0, 50),
            pt(f64::NAN, 1.0, 70),
            pt(3.0, 0.5, 95),
            pt(-1.0, 1.0, 10),
        ]);
        assert_eq!(curve.points.len(), 3);
        assert_eq!(curve.points[0].rate_bpp, 1.0);
        assert_eq!(curve.points[2].rate_bpp, 3.0);
    }

    #[test]
    fn pareto_hull_monotone() {
        // A clean monotone-decreasing curve stays intact.
        let curve = RdCurve::from_points([
            pt(0.5, 20.0, 40),
            pt(1.0, 10.0, 60),
            pt(2.0, 5.0, 80),
            pt(4.0, 1.0, 95),
        ]);
        let hull = curve.pareto_hull();
        assert_eq!(hull.points.len(), 4);
    }

    #[test]
    fn pareto_hull_collapses_non_monotonic() {
        // A curve where a mid-quality point has HIGHER distortion than a
        // lower-quality point at smaller rate — the bad point gets dropped.
        let curve = RdCurve::from_points([
            pt(1.0, 5.0, 60),
            pt(2.0, 7.0, 70), // dominated: higher rate AND higher distortion
            pt(3.0, 3.0, 80),
            pt(4.0, 1.0, 95),
        ]);
        let hull = curve.pareto_hull();
        assert_eq!(hull.points.len(), 3);
        assert_eq!(hull.points[0].quality, 60);
        assert_eq!(hull.points[1].quality, 80);
        assert_eq!(hull.points[2].quality, 95);
    }

    #[test]
    fn pareto_hull_collapses_dup_rate() {
        let curve = RdCurve::from_points([
            pt(1.0, 5.0, 50),
            pt(1.0, 3.0, 55), // same rate, lower distortion wins
            pt(2.0, 1.0, 90),
        ]);
        let hull = curve.pareto_hull();
        assert_eq!(hull.points.len(), 2);
        assert_eq!(hull.points[0].quality, 55);
    }

    #[test]
    fn identical_curves_bd_rate_zero() {
        let points: Vec<_> = (0..5)
            .map(|i| {
                let q = 50 + i * 10;
                pt(0.5 * (i + 1) as f64, 10.0 / (i + 1) as f64, q as u8)
            })
            .collect();
        let a = RdCurve::from_points(points.clone());
        let b = RdCurve::from_points(points);
        let bd = bd_rate(&a, &b).expect("identical curves should yield a BD");
        assert!(
            bd.abs() < 1e-6,
            "identical curves should yield BD-rate ≈ 0, got {}",
            bd
        );
        let cmp = compare(&a, &b);
        assert!(
            cmp.mean_distance.abs() < 1e-9,
            "identical curves mean_distance=0, got {}",
            cmp.mean_distance
        );
    }

    #[test]
    fn better_candidate_negative_bd_rate() {
        // Baseline and candidate span the same distortion range, but the
        // candidate is strictly lower rate at every distortion.
        let baseline = RdCurve::from_points([
            pt(0.5, 20.0, 40),
            pt(1.0, 10.0, 60),
            pt(2.0, 5.0, 80),
            pt(4.0, 1.0, 95),
        ]);
        let candidate = RdCurve::from_points([
            pt(0.4, 20.0, 40),
            pt(0.8, 10.0, 60),
            pt(1.6, 5.0, 80),
            pt(3.2, 1.0, 95),
        ]);
        let bd = bd_rate(&baseline, &candidate).expect("should produce BD-rate");
        assert!(
            bd < 0.0,
            "better candidate should have negative BD-rate, got {}",
            bd
        );
        // 20% rate reduction at every point → BD-rate ≈ -20%.
        assert!((bd - (-20.0)).abs() < 1.0, "expected ≈ -20%, got {}", bd);

        let cmp = compare(&baseline, &candidate);
        assert!(
            cmp.mean_distance > 0.0,
            "candidate at lower rate, same distortion → positive distance"
        );
    }

    #[test]
    fn worse_candidate_positive_bd_rate() {
        let baseline = RdCurve::from_points([
            pt(0.5, 20.0, 40),
            pt(1.0, 10.0, 60),
            pt(2.0, 5.0, 80),
            pt(4.0, 1.0, 95),
        ]);
        // Candidate uses 25% more rate at every distortion.
        let candidate = RdCurve::from_points([
            pt(0.625, 20.0, 40),
            pt(1.25, 10.0, 60),
            pt(2.5, 5.0, 80),
            pt(5.0, 1.0, 95),
        ]);
        let bd = bd_rate(&baseline, &candidate).expect("should produce BD-rate");
        assert!(
            bd > 0.0,
            "worse candidate should have positive BD-rate, got {}",
            bd
        );
        assert!((bd - 25.0).abs() < 1.5, "expected ≈ +25%, got {}", bd);

        let cmp = compare(&baseline, &candidate);
        assert!(
            cmp.mean_distance < 0.0,
            "candidate at higher rate, same distortion → negative distance"
        );
    }

    #[test]
    fn disjoint_distortion_ranges_return_none() {
        let a = RdCurve::from_points([pt(0.1, 100.0, 10), pt(0.2, 80.0, 20)]);
        let b = RdCurve::from_points([pt(0.5, 5.0, 90), pt(1.0, 1.0, 99)]);
        assert!(
            bd_rate(&a, &b).is_none(),
            "disjoint distortion ranges → None"
        );
    }

    #[test]
    fn too_few_points_returns_none() {
        let a = RdCurve::from_points([pt(1.0, 5.0, 75)]);
        let b = RdCurve::from_points([pt(0.5, 10.0, 50), pt(1.0, 5.0, 75), pt(2.0, 1.0, 90)]);
        assert!(bd_rate(&a, &b).is_none(), "single-point baseline → None");
        assert!(bd_rate(&b, &a).is_none(), "single-point candidate → None");
    }

    #[test]
    fn pareto_hull_fixes_non_monotonic_for_bd_rate() {
        // Non-monotonic curve (like the SA-optimized-tables Bug #2 reference).
        // With interior dominated points removed, BD-rate against a reasonable
        // baseline should still come out finite.
        let baseline = RdCurve::from_points([
            pt(0.5, 20.0, 50),
            pt(1.0, 10.0, 70),
            pt(2.0, 5.0, 85),
            pt(4.0, 1.0, 95),
        ]);
        let non_monotonic = RdCurve::from_points([
            pt(0.5, 19.0, 50),
            pt(1.0, 25.0, 70), // dominated by the Q=50 point ABOVE it
            pt(2.0, 5.0, 85),
            pt(4.0, 1.0, 95),
        ]);
        let bd = bd_rate(&baseline, &non_monotonic).expect("pareto hull should salvage it");
        // After hull, the candidate is: (0.5, 19) (2.0, 5) (4.0, 1) — just as
        // good as baseline at extreme points, slightly better at low rate.
        assert!(bd.is_finite());
    }

    #[test]
    fn closest_point_distance_sign_and_magnitude() {
        let baseline = RdCurve::from_points([pt(1.0, 10.0, 60), pt(2.0, 5.0, 80)]).pareto_hull();
        // At rate = sqrt(2) ≈ 1.414 (log10-midpoint of 1 and 2),
        // linear interp in log-rate gives distortion = 7.5.
        let pt_below = pt(2.0_f64.sqrt(), 6.0, 70);
        let d = closest_point_distance(&baseline, &pt_below);
        assert!(d > 0.0);
        assert!((d - 1.5).abs() < 0.2);

        let pt_above = pt(2.0_f64.sqrt(), 9.0, 70);
        let d = closest_point_distance(&baseline, &pt_above);
        assert!(d < 0.0);
    }
}
