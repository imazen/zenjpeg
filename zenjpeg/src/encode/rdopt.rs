//! RD-OPT: Rate-distortion optimal quantization table refinement.
//!
//! Implements content-adaptive quantization table optimization based on
//! Ratnakar & Livny's RD-OPT algorithm (US 5,724,453, expired 2015).
//!
//! The algorithm works in two phases:
//!
//! 1. **Histogram collection**: During the normal DCT pass, accumulate
//!    per-frequency-position coefficient histograms from the raw f32 DCT output.
//!
//! 2. **Lagrangian optimization**: For each of the 64 frequency positions,
//!    search candidate quantizer values (and optionally zeroing thresholds)
//!    near the base table to minimize `λ·R(q) + w·D(q)`, where R is the
//!    entropy estimate and D is the weighted MSE.
//!
//! The refined table is then used to re-quantize the already-buffered i16 blocks.
//!
//! # References
//!
//! - V. Ratnakar and M. Livny, "RD-OPT: An Efficient Algorithm for Optimizing
//!   DCT Quantization Tables," Proc. Data Compression Conference, 1995.
//! - V. Ratnakar and M. Livny, "Extending RD-OPT with Global Thresholding
//!   for JPEG Optimization," Proc. Data Compression Conference, 1996.

use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_ZIGZAG_ORDER};
use crate::quant::QuantTable;

// ---------------------------------------------------------------------------
// Histogram collection
// ---------------------------------------------------------------------------

/// Half-integer bucket scale: coefficient c maps to bucket `floor(c * BUCKET_SCALE)`.
///
/// Using 2.0 gives half-integer precision (0.5 steps). This matches the
/// original RD-OPT paper's bucketing while keeping memory bounded.
const BUCKET_SCALE: f32 = 2.0;

/// Maximum absolute bucket index we track. Coefficients outside this range
/// are clamped. At BUCKET_SCALE=2, this covers DCT values up to ±512.
const MAX_BUCKET: i32 = 1024;

/// Total number of buckets: [-MAX_BUCKET, MAX_BUCKET] inclusive.
const NUM_BUCKETS: usize = (2 * MAX_BUCKET + 1) as usize;

/// Maps a bucket index to the array offset.
#[inline]
fn bucket_offset(bucket: i32) -> usize {
    (bucket + MAX_BUCKET) as usize
}

/// Per-frequency-position histogram of DCT coefficient values.
///
/// Buckets are indexed by `floor(coeff * BUCKET_SCALE)`, giving half-integer
/// precision. This is sufficient for accurate entropy and MSE estimation.
struct CoeffHistogram {
    /// Counts per bucket. Index 0 = bucket -MAX_BUCKET, index NUM_BUCKETS-1 = +MAX_BUCKET.
    counts: Box<[u32; NUM_BUCKETS]>,
    /// Total number of samples (blocks) accumulated.
    total: u64,
    /// Maximum absolute bucket index seen (for bounding the search).
    max_abs_bucket: i32,
}

impl CoeffHistogram {
    fn new() -> Self {
        Self {
            counts: Box::new([0u32; NUM_BUCKETS]),
            total: 0,
            max_abs_bucket: 0,
        }
    }

    /// Record a single DCT coefficient value.
    #[inline]
    fn record(&mut self, value: f32) {
        let bucket = (value * BUCKET_SCALE).floor() as i32;
        let clamped = bucket.clamp(-MAX_BUCKET, MAX_BUCKET);
        self.counts[bucket_offset(clamped)] += 1;
        self.total += 1;
        let abs = clamped.unsigned_abs() as i32;
        if abs > self.max_abs_bucket {
            self.max_abs_bucket = abs;
        }
    }
}

/// Histograms for all 64 DCT frequency positions of one component.
pub(crate) struct ComponentHistograms {
    /// Per-frequency histograms in natural (row-major) order.
    histograms: Box<[CoeffHistogram; DCT_BLOCK_SIZE]>,
}

impl ComponentHistograms {
    pub(crate) fn new() -> Self {
        Self {
            histograms: Box::new(std::array::from_fn(|_| CoeffHistogram::new())),
        }
    }

    /// Accumulate one 8x8 DCT block (natural order f32 coefficients).
    #[inline]
    pub(crate) fn accumulate_block(&mut self, block: &crate::foundation::simd_types::Block8x8f) {
        for row in 0..8 {
            for col in 0..8 {
                let n = row * 8 + col;
                self.histograms[n].record(block.rows[row][col]);
            }
        }
    }

    /// Total blocks accumulated (from position 0's count).
    fn total_blocks(&self) -> u64 {
        self.histograms[0].total
    }
}

// ---------------------------------------------------------------------------
// RD-OPT context: holds histograms + optimization parameters
// ---------------------------------------------------------------------------

/// Context for RD-OPT quantization table refinement.
///
/// Collects DCT coefficient histograms during the encode pass and then
/// uses Lagrangian optimization to produce refined quantization tables
/// and optional global zeroing thresholds.
pub(crate) struct RdOptContext {
    // Manual Debug impl below (histogram arrays are too large for derive).
    /// Per-component histograms: [Y, Cb, Cr].
    pub(crate) histograms: [ComponentHistograms; 3],
    /// Lagrangian multiplier λ controlling the rate-distortion tradeoff.
    /// Higher λ = more compression (larger quantization steps).
    /// Automatically derived from the quality level if not set explicitly.
    pub(crate) lambda: f32,
    /// Enable global thresholding (Phase 2).
    pub(crate) enable_thresholds: bool,
}

impl RdOptContext {
    /// Create a new RD-OPT context.
    ///
    /// `lambda` controls the rate-distortion tradeoff. A good default is
    /// derived from the butteraugli distance: `lambda ≈ distance²`.
    pub(crate) fn new(lambda: f32, enable_thresholds: bool) -> Self {
        Self {
            histograms: std::array::from_fn(|_| ComponentHistograms::new()),
            lambda,
            enable_thresholds,
        }
    }

    /// Accumulate a Y-channel DCT block.
    #[inline]
    pub(crate) fn accumulate_y(&mut self, block: &crate::foundation::simd_types::Block8x8f) {
        self.histograms[0].accumulate_block(block);
    }

    /// Accumulate a Cb-channel DCT block.
    #[inline]
    pub(crate) fn accumulate_cb(&mut self, block: &crate::foundation::simd_types::Block8x8f) {
        self.histograms[1].accumulate_block(block);
    }

    /// Accumulate a Cr-channel DCT block.
    #[inline]
    pub(crate) fn accumulate_cr(&mut self, block: &crate::foundation::simd_types::Block8x8f) {
        self.histograms[2].accumulate_block(block);
    }

    /// Optimize a single component's quantization table using Lagrangian search.
    ///
    /// For each frequency position n, finds the quantizer q (and optionally
    /// threshold t) that minimizes `λ·R_n(q,t) + D_n(q,t)`.
    ///
    /// Returns `(refined_quant_values_zigzag, thresholds_natural)`.
    pub(crate) fn optimize_component(
        &self,
        component: usize,
        base_quant: &QuantTable,
        perceptual_weights: &[f32; DCT_BLOCK_SIZE],
    ) -> OptimizedTable {
        let hists = &self.histograms[component];
        let total_blocks = hists.total_blocks();

        if total_blocks == 0 {
            // No data collected — return base table unchanged
            return OptimizedTable {
                quant_values_zigzag: base_quant.values,
                thresholds_natural: [0.0; DCT_BLOCK_SIZE],
            };
        }

        let mut quant_natural = [0u16; DCT_BLOCK_SIZE];
        let mut thresholds = [0.0f32; DCT_BLOCK_SIZE];

        let total_pixels = total_blocks * 64;

        // Phase 1: Compute base table's per-position R and D to calibrate lambda.
        // We want the lambda that makes the base table approximately optimal.
        // At the optimal point: λ = -dD/dR (the slope of the R-D curve).
        // We estimate this per position by evaluating R,D at base_q and base_q+1.
        let mut sum_delta_d = 0.0f64;
        let mut sum_delta_r = 0.0f64;
        for n in 0..DCT_BLOCK_SIZE {
            let hist = &hists.histograms[n];
            let zigzag_idx = JPEG_ZIGZAG_ORDER[n] as usize;
            let base_q = base_quant.values[zigzag_idx].max(1) as u32;

            let (r0, d0) = estimate_rd(hist, total_blocks, total_pixels, base_q, None);
            let (r1, d1) = estimate_rd(hist, total_blocks, total_pixels, base_q + 1, None);

            let dr = r0 - r1; // rate decrease from coarser q (positive)
            let dd = d1 - d0; // distortion increase from coarser q (positive)
            if dr > 1e-12 {
                sum_delta_d += perceptual_weights[n] as f64 * dd;
                sum_delta_r += dr;
            }
        }

        // Calibrated lambda: at this lambda, the base table is approximately
        // at a critical point (marginal R-D tradeoff is balanced).
        // We scale by the user's lambda factor to allow pushing toward
        // more compression (lambda > 1) or more quality (lambda < 1).
        let calibrated_lambda = if sum_delta_r > 1e-12 {
            sum_delta_d / sum_delta_r
        } else {
            1.0
        };
        // Apply user's lambda as a multiplicative factor on the calibrated value
        let lambda = calibrated_lambda * self.lambda as f64;

        // Phase 2: Optimize each position with the calibrated lambda.
        for n in 0..DCT_BLOCK_SIZE {
            let hist = &hists.histograms[n];
            let zigzag_idx = JPEG_ZIGZAG_ORDER[n] as usize;
            let base_q = base_quant.values[zigzag_idx].max(1);

            // Conservative search range: ±30% of base
            let q_min = ((base_q as f32 * 0.7).floor() as u32).max(1);
            let max_coeff = (hist.max_abs_bucket as f32 / BUCKET_SCALE).ceil() as u32;
            let q_max = if max_coeff == 0 {
                base_q as u32
            } else {
                let upper = (base_q as f32 * 1.3).ceil() as u32;
                upper.min(2 * max_coeff + 1).min(65535)
            };
            let q_max = q_max.max(q_min);

            let w = perceptual_weights[n] as f64;

            // Evaluate base cost
            let (base_rate, base_dist) =
                estimate_rd(hist, total_blocks, total_pixels, base_q as u32, None);
            let base_cost = lambda * base_rate + w * base_dist;

            let mut best_cost = base_cost;
            let mut best_q = base_q as u32;
            let mut best_t = 0.0f32;

            for q in q_min..=q_max {
                if q == base_q as u32 {
                    continue; // already evaluated as base_cost
                }

                if self.enable_thresholds {
                    let t_min = q as f32 * 0.5;
                    let t_max = (q as f32 * 1.5).min(max_coeff as f32 + 1.0);
                    let mut t = t_min;
                    while t <= t_max {
                        let (rate, dist) =
                            estimate_rd(hist, total_blocks, total_pixels, q, Some(t));
                        let cost = lambda * rate + w * dist;
                        if cost < best_cost {
                            best_cost = cost;
                            best_q = q;
                            best_t = t;
                        }
                        t += 0.5;
                    }
                } else {
                    let (rate, dist) = estimate_rd(hist, total_blocks, total_pixels, q, None);
                    let cost = lambda * rate + w * dist;
                    if cost < best_cost {
                        best_cost = cost;
                        best_q = q;
                    }
                }
            }

            quant_natural[n] = (best_q as u16).max(1);
            thresholds[n] = best_t;
        }

        // Convert to zigzag order for QuantTable
        let mut quant_zigzag = [0u16; DCT_BLOCK_SIZE];
        for n in 0..DCT_BLOCK_SIZE {
            let zi = JPEG_ZIGZAG_ORDER[n] as usize;
            quant_zigzag[zi] = quant_natural[n];
        }

        OptimizedTable {
            quant_values_zigzag: quant_zigzag,
            thresholds_natural: thresholds,
        }
    }
}

impl std::fmt::Debug for RdOptContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RdOptContext")
            .field("lambda", &self.lambda)
            .field("enable_thresholds", &self.enable_thresholds)
            .field(
                "total_blocks",
                &[
                    self.histograms[0].total_blocks(),
                    self.histograms[1].total_blocks(),
                    self.histograms[2].total_blocks(),
                ],
            )
            .finish()
    }
}

/// Result of RD-OPT optimization for one component.
pub(crate) struct OptimizedTable {
    /// Refined quantization values in zigzag order (ready for `QuantTable`).
    pub(crate) quant_values_zigzag: [u16; DCT_BLOCK_SIZE],
    /// Per-frequency zeroing thresholds in natural order.
    /// Value of 0.0 means "use standard JPEG threshold (q/2)".
    pub(crate) thresholds_natural: [f32; DCT_BLOCK_SIZE],
}

// ---------------------------------------------------------------------------
// Rate and distortion estimation
// ---------------------------------------------------------------------------

/// Estimate rate R_n(q, t) and distortion D_n(q, t) for one frequency position.
///
/// - Rate: Shannon entropy of the quantized distribution, divided by 64
///   (contribution of one coefficient to bits-per-pixel).
/// - Distortion: Mean squared error between original and dequantized values.
///
/// If `threshold` is Some, coefficients with |value| < threshold are forced to 0.
fn estimate_rd(
    hist: &CoeffHistogram,
    total_blocks: u64,
    total_pixels: u64,
    q: u32,
    threshold: Option<f32>,
) -> (f64, f64) {
    // We'll accumulate counts per quantized value using a small hashmap.
    // Most quantized values cluster near 0, so a bounded array is efficient.
    // Range of quantized values: roughly [-MAX_BUCKET / (BUCKET_SCALE * q), +same]
    let max_qval = (MAX_BUCKET as f64 / (BUCKET_SCALE as f64 * q as f64)).ceil() as i32 + 1;
    let max_qval = max_qval.min(512); // reasonable bound

    // Use a Vec-backed map: index = qval + max_qval
    let map_size = (2 * max_qval + 1) as usize;
    let mut freq = vec![0u64; map_size];

    let mut mse_accum = 0.0f64;
    let t = threshold.unwrap_or(q as f32 * 0.5);

    let bucket_lo = bucket_offset(-hist.max_abs_bucket);
    let bucket_hi = bucket_offset(hist.max_abs_bucket);

    for bi in bucket_lo..=bucket_hi {
        let count = hist.counts[bi] as u64;
        if count == 0 {
            continue;
        }

        let bucket = bi as i32 - MAX_BUCKET;
        let original = bucket as f64 / BUCKET_SCALE as f64;

        // Apply thresholding + quantization
        let qval = if original.abs() < t as f64 {
            0i32
        } else {
            // Standard JPEG rounding: round(original / q)
            let v = original / q as f64;
            if v >= 0.0 {
                (v + 0.5) as i32
            } else {
                (v - 0.5) as i32
            }
        };

        // Dequantized (reconstructed) value
        let reconstructed = qval as f64 * q as f64;
        let error = original - reconstructed;
        mse_accum += count as f64 * error * error;

        // Accumulate frequency count for entropy
        let map_idx = (qval + max_qval) as usize;
        if map_idx < map_size {
            freq[map_idx] += count;
        }
    }

    // Compute Shannon entropy
    let total = total_blocks as f64;
    let mut entropy = 0.0f64;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    // R_n(q) = entropy / 64 (contribution of one coefficient to bpp)
    let rate = entropy / 64.0;

    // D_n(q) = MSE / total_pixels
    let distortion = mse_accum / total_pixels as f64;

    (rate, distortion)
}

// ---------------------------------------------------------------------------
// Global thresholding of buffered i16 blocks
// ---------------------------------------------------------------------------

/// Apply global thresholding to already-quantized i16 blocks.
///
/// For each coefficient at natural position n:
///   - Approximate the original DCT value: `approx = coeff * quant_step`
///   - If `|approx| < threshold[n]`, zero the coefficient
///
/// This is always safe: zeroing can only reduce file size. The quality
/// impact is bounded by the quant step (the coefficient was already
/// quantized to at most ±1 of its optimal value).
///
/// Unlike `requantize_blocks`, this does NOT change the quant table —
/// it only zeros coefficients, avoiding all re-quantization error.
pub(crate) fn apply_global_thresholds(
    blocks: &mut [[i16; DCT_BLOCK_SIZE]],
    quant: &QuantTable,
    thresholds_natural: &[f32; DCT_BLOCK_SIZE],
) {
    // Precompute threshold in zigzag order
    let mut thresh_zigzag = [0.0f32; DCT_BLOCK_SIZE];
    let mut any_threshold = false;

    for n in 0..DCT_BLOCK_SIZE {
        let zi = JPEG_ZIGZAG_ORDER[n] as usize;
        let t = thresholds_natural[n];
        // Only apply threshold if it's ABOVE the default q/2
        // (below q/2, the standard quantization already zeroed it)
        let default_thresh = quant.values[zi] as f32 * 0.5;
        if t > default_thresh {
            thresh_zigzag[zi] = t;
            any_threshold = true;
        }
    }

    if !any_threshold {
        return; // No thresholds to apply
    }

    for block in blocks.iter_mut() {
        for zi in 0..DCT_BLOCK_SIZE {
            if thresh_zigzag[zi] == 0.0 {
                continue; // no threshold for this position
            }
            let coeff = block[zi];
            if coeff == 0 {
                continue; // already zero
            }
            // Approximate original DCT value
            let approx = coeff as f32 * quant.values[zi].max(1) as f32;
            if approx.abs() < thresh_zigzag[zi] {
                block[zi] = 0;
            }
        }
    }
}

/// Re-quantize already-quantized i16 blocks from old quant table to new quant table.
///
/// CAUTION: This introduces rounding error from the i16→f32→i16 conversion.
/// Use `apply_global_thresholds` when only zeroing is needed.
#[allow(dead_code)] // Available for future table refinement when encoding from f32
pub(crate) fn requantize_blocks(
    blocks: &mut [[i16; DCT_BLOCK_SIZE]],
    old_quant: &QuantTable,
    new_quant: &QuantTable,
    thresholds_natural: &[f32; DCT_BLOCK_SIZE],
) {
    let mut ratio = [0.0f32; DCT_BLOCK_SIZE];
    let mut thresh_zigzag = [0.0f32; DCT_BLOCK_SIZE];

    for n in 0..DCT_BLOCK_SIZE {
        let zi = JPEG_ZIGZAG_ORDER[n] as usize;
        let old_q = old_quant.values[zi].max(1) as f32;
        let new_q = new_quant.values[zi].max(1) as f32;
        ratio[zi] = old_q / new_q;
        thresh_zigzag[zi] = thresholds_natural[n];
    }

    for block in blocks.iter_mut() {
        for zi in 0..DCT_BLOCK_SIZE {
            let old_coeff = block[zi] as f32;
            if old_coeff == 0.0 {
                continue;
            }
            let approx_original = old_coeff * old_quant.values[zi].max(1) as f32;
            if thresh_zigzag[zi] > 0.0 && approx_original.abs() < thresh_zigzag[zi] {
                block[zi] = 0;
                continue;
            }
            let new_val = old_coeff * ratio[zi];
            block[zi] = if new_val >= 0.0 {
                (new_val + 0.5) as i16
            } else {
                (new_val - 0.5) as i16
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Perceptual weights
// ---------------------------------------------------------------------------

/// Default perceptual weights for the 64 DCT frequency positions (natural order).
///
/// These weights modulate the distortion term in the Lagrangian so the
/// optimizer preserves visually important frequencies.
///
/// DC (position 0) gets weight 64 because a DC quantization error of ±1
/// shifts all 64 pixels in the block. Low-frequency AC positions get
/// moderate weights (4-16), and high frequencies get low weights (1-2).
pub(crate) fn default_perceptual_weights() -> [f32; DCT_BLOCK_SIZE] {
    let mut weights = [0.0f32; DCT_BLOCK_SIZE];
    for row in 0..8u32 {
        for col in 0..8u32 {
            let n = (row * 8 + col) as usize;
            if row == 0 && col == 0 {
                // DC: error affects all 64 pixels equally
                weights[n] = 64.0;
            } else {
                // AC: weight decreases with frequency. Low AC positions
                // affect large-scale structure; high positions affect detail.
                // Use 1/(1 + 0.5*(u+v)) which gives:
                //   (0,1)→2.0, (1,0)→2.0, (1,1)→1.0, (7,7)→0.13
                let freq = (row + col) as f32;
                weights[n] = 2.0 / (1.0 + 0.5 * freq);
            }
        }
    }
    weights
}

/// Derive a Lagrangian multiplier from butteraugli distance.
///
/// The lambda is automatically calibrated from the actual histogram data
/// rather than being set from distance alone. This function provides an
/// initial estimate that `optimize_component` will refine.
///
/// The value is deliberately conservative (low lambda = more weight on
/// quality) to avoid the re-quantization from i16 degrading quality.
pub(crate) fn lambda_from_distance(distance: f32) -> f32 {
    // Conservative: lambda grows slowly with distance.
    // At d=0.5: λ=0.25 (barely any adjustment)
    // At d=1.0: λ=1.0 (modest)
    // At d=2.0: λ=4.0 (moderate)
    // At d=3.0: λ=9.0 (allows meaningful compression improvement)
    distance * distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_offset_roundtrip() {
        assert_eq!(bucket_offset(0), MAX_BUCKET as usize);
        assert_eq!(bucket_offset(-MAX_BUCKET), 0);
        assert_eq!(bucket_offset(MAX_BUCKET), NUM_BUCKETS - 1);
    }

    #[test]
    fn test_histogram_basic() {
        let mut hist = CoeffHistogram::new();
        hist.record(0.0);
        hist.record(1.5);
        hist.record(-1.5);
        assert_eq!(hist.total, 3);
        // 0.0 → bucket 0
        assert_eq!(hist.counts[bucket_offset(0)], 1);
        // 1.5 → bucket floor(1.5*2) = 3
        assert_eq!(hist.counts[bucket_offset(3)], 1);
        // -1.5 → bucket floor(-1.5*2) = -3
        assert_eq!(hist.counts[bucket_offset(-3)], 1);
        assert_eq!(hist.max_abs_bucket, 3);
    }

    #[test]
    fn test_estimate_rd_all_zero() {
        let mut hist = CoeffHistogram::new();
        for _ in 0..100 {
            hist.record(0.0);
        }
        let (rate, dist) = estimate_rd(&hist, 100, 6400, 16, None);
        // All zeros → entropy = 0, distortion = 0
        assert!(rate.abs() < 1e-10);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_estimate_rd_uniform() {
        let mut hist = CoeffHistogram::new();
        // Symmetric distribution: values at ±8, ±16, ±24
        for _ in 0..100 {
            hist.record(8.0);
            hist.record(-8.0);
            hist.record(16.0);
            hist.record(-16.0);
        }
        // With q=16, values ±8 → quantized to ±1, ±16 → ±1
        let (rate, dist) = estimate_rd(&hist, 400, 400 * 64, 16, None);
        assert!(rate > 0.0, "should have non-zero entropy");
        assert!(dist >= 0.0, "distortion should be non-negative");
    }

    #[test]
    fn test_requantize_identity() {
        let qt = QuantTable {
            values: [16; DCT_BLOCK_SIZE],
            precision: 0,
        };
        let mut blocks = vec![[0i16; DCT_BLOCK_SIZE]; 2];
        blocks[0][0] = 100;
        blocks[0][1] = -50;
        blocks[1][0] = 200;

        let thresholds = [0.0f32; DCT_BLOCK_SIZE];
        // Same old/new table → no change
        requantize_blocks(&mut blocks, &qt, &qt, &thresholds);
        assert_eq!(blocks[0][0], 100);
        assert_eq!(blocks[0][1], -50);
        assert_eq!(blocks[1][0], 200);
    }

    #[test]
    fn test_requantize_doubles_q() {
        let old_qt = QuantTable {
            values: [16; DCT_BLOCK_SIZE],
            precision: 0,
        };
        let new_qt = QuantTable {
            values: [32; DCT_BLOCK_SIZE],
            precision: 0,
        };
        let mut blocks = vec![[0i16; DCT_BLOCK_SIZE]; 1];
        blocks[0][0] = 10; // original ~= 10*16 = 160, new = round(160/32) = 5
        blocks[0][1] = 3; // original ~= 3*16 = 48, new = round(48/32) = 2 (1.5 rounds to 2)

        let thresholds = [0.0f32; DCT_BLOCK_SIZE];
        requantize_blocks(&mut blocks, &old_qt, &new_qt, &thresholds);
        assert_eq!(blocks[0][0], 5);
        assert_eq!(blocks[0][1], 2);
    }

    #[test]
    fn test_perceptual_weights_dc_highest() {
        let w = default_perceptual_weights();
        // DC (0,0) should have the highest weight (64)
        assert!((w[0] - 64.0).abs() < 1e-6);
        // Higher frequencies should have lower weight
        assert!(w[63] < w[0]);
        assert!(w[7] < w[0]); // (0,7)
        assert!(w[56] < w[0]); // (7,0)
        // AC weights should be moderate
        assert!(w[1] > 1.0); // (0,1)
        assert!(w[1] < 3.0);
    }

    #[test]
    fn test_lambda_from_distance() {
        assert!((lambda_from_distance(1.0) - 1.0).abs() < 1e-6);
        assert!((lambda_from_distance(2.0) - 4.0).abs() < 1e-6);
        assert!((lambda_from_distance(3.0) - 9.0).abs() < 1e-6);
    }
}
