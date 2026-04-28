//! Tier 1: variance, edges, chroma stats, uniformity, palette.
//!
//! Sparse stripe sampling (8-row stripes, ~500k pixel budget — same
//! heuristic `coefficient::analysis::feature_extract` uses, ported
//! byte-for-byte so the parity test is meaningful).
//!
//! Each active stripe pulls 9 rows (8-row block + lookahead for the
//! vertical-gradient edge term) into a stripe scratch via
//! [`RowStream`], then runs the existing single-pass row scan over
//! that scratch. No full-image RGB8 materialization.
//!
//! Hot loop is `accumulate_row`, dispatched through `archmage::incant!`
//! to v3 / NEON / WASM128 / scalar. The 24-byte fixed-array chunks
//! prove the size to LLVM, eliminating interior bounds checks and
//! letting the autovectorizer fully unroll the 8-pixel batch.

use super::feature::RawAnalysis;
use super::row_stream::RowStream;
use archmage::{incant, magetypes};

// Luma weights are no longer constants — they're picked per-source-
// primaries by `crate::luma::LumaWeights::for_primaries(...)` and
// threaded into the SIMD kernels as `kr / kg / kb` parameters.
const EDGE_THRESH_SQ: f32 = 400.0; // (|∇L| > 20)²

/// Stripe height. Matches the 8×8 block size used for uniformity so
/// each stripe contributes complete blocks without partial-block
/// artifacts.
const STRIPE_H: usize = 8;

/// Pixel budget for stripe sampling (~1 ms at 4K on a 7950X). This
/// is the value the oracle decision trees were trained against — see
/// the threshold contract in `crate`-level docs. Re-exported as
/// `AnalyzerConfig::default().pixel_budget`.
pub(crate) const DEFAULT_PIXEL_BUDGET: usize = 500_000;

#[derive(Default, Clone, Copy)]
struct PixelStats {
    luma_sum: f64,
    luma_sq_sum: f64,
    cb_sum: f64,
    cb_sq_sum: f64,
    cr_sum: f64,
    cr_sq_sum: f64,
    edge_count: u64,
    cb_grad_sum: f64,
    cr_grad_sum: f64,
    chroma_grad_count: u64,
    /// Hasler-Süsstrunk M3 colourfulness intermediates.
    /// `rg = R − G`, `yb = 0.5·(R + G) − B` per pixel.
    rg_sum: f64,
    rg_sq_sum: f64,
    yb_sum: f64,
    yb_sq_sum: f64,
    /// Discrete Laplacian of luma over interior pixels:
    /// `∇²L = L[x-1] + L[x+1] + L[y-1] + L[y+1] − 4·L[x,y]`.
    /// Sum + sum-of-squares for the variance reduction.
    laplacian_sum: f64,
    laplacian_sq_sum: f64,
    laplacian_count: u64,
    /// Skin-tone pixel count: pigmentation-invariant Chai-Ngan YCbCr
    /// gate computed lane-wise in the f32x8 stats pass alongside
    /// luma/chroma stats. Branchless: lane mask → blend(1.0, 0.0) → sum.
    skin_count: u64,
    /// Edge-slope (gradient magnitude) running stats over pixels that
    /// crossed `EDGE_THRESH_SQ`. Accumulated branchlessly inside the
    /// existing edge inner loop — we already know `grad_sq` and
    /// whether it crossed; multiply by the mask to add 0 on misses.
    edge_grad_sum: f64,
    edge_grad_sq_sum: f64,
    edge_grad_count: u64,
}

impl PixelStats {
    fn merge(&mut self, o: &PixelStats) {
        self.luma_sum += o.luma_sum;
        self.luma_sq_sum += o.luma_sq_sum;
        self.cb_sum += o.cb_sum;
        self.cb_sq_sum += o.cb_sq_sum;
        self.cr_sum += o.cr_sum;
        self.cr_sq_sum += o.cr_sq_sum;
        self.edge_count += o.edge_count;
        self.cb_grad_sum += o.cb_grad_sum;
        self.cr_grad_sum += o.cr_grad_sum;
        self.chroma_grad_count += o.chroma_grad_count;
        self.rg_sum += o.rg_sum;
        self.rg_sq_sum += o.rg_sq_sum;
        self.yb_sum += o.yb_sum;
        self.yb_sq_sum += o.yb_sq_sum;
        self.laplacian_sum += o.laplacian_sum;
        self.laplacian_sq_sum += o.laplacian_sq_sum;
        self.laplacian_count += o.laplacian_count;
        self.skin_count += o.skin_count;
        self.edge_grad_sum += o.edge_grad_sum;
        self.edge_grad_sq_sum += o.edge_grad_sq_sum;
        self.edge_grad_count += o.edge_grad_count;
    }
}

/// Populate Tier 1 fields on `out`. Other fields are left untouched.
///
/// `pixel_budget` controls the stripe step (see [`compute_stripe_step`]).
/// Pass [`DEFAULT_PIXEL_BUDGET`] to match the oracle-trained reference
/// behavior; pass a smaller value for proxy-server speed (with reduced
/// feature precision on multi-megapixel inputs).
/// Per-call dispatch knobs for the Tier 1 stripe sweep. Lets the
/// caller skip optional accumulators / passes that the requested
/// `FeatureSet` doesn't need, recovering register pressure on the
/// AVX2 hot path. Computed once per `analyze_features` call from
/// `feature::TIER1_EXTRAS_FEATURES`.
#[derive(Clone, Copy)]
pub(crate) struct Tier1Dispatch {
    /// Skip the separate Laplacian SIMD row pass when
    /// `LaplacianVariance` isn't requested.
    pub(crate) wants_laplacian: bool,
    /// Run the full SIMD kernel's optional accumulators
    /// (luma_sum/sq for Variance, rg/yb for Colourfulness, edge_grad
    /// sums for EdgeSlopeStdev). Gated on the Skin axis below — on
    /// AVX2's 16-register file the kernel runs out of YMM regs when
    /// FULL + SKIN are both live, so peel them apart.
    pub(crate) wants_full_kernel: bool,
    /// Run the BT.601 chroma matrix + Chai-Ngan skin-tone gate.
    /// Independent of `wants_full_kernel` — `SkinToneFraction` can
    /// be queried on its own without paying for Variance /
    /// Colourfulness / EdgeSlope, and likewise `Variance` doesn't
    /// pay for the skin gate's 6 mask compares + 5 ANDs + 2 chroma
    /// FMAs per chunk. Splitting this out off `wants_full_kernel`
    /// is the AVX2-register-pressure relief flagged by `cargo asm`.
    pub(crate) wants_skin: bool,
}

impl Tier1Dispatch {
    pub(crate) fn full() -> Self {
        Self {
            wants_laplacian: true,
            wants_full_kernel: true,
            wants_skin: true,
        }
    }
}

pub fn extract_tier1_into(out: &mut RawAnalysis, stream: &mut RowStream<'_>, pixel_budget: usize) {
    extract_tier1_into_dispatch(out, stream, pixel_budget, Tier1Dispatch::full());
}

pub(crate) fn extract_tier1_into_dispatch(
    out: &mut RawAnalysis,
    stream: &mut RowStream<'_>,
    pixel_budget: usize,
    dispatch: Tier1Dispatch,
) {
    let w = stream.width() as usize;
    let h = stream.height() as usize;
    if w < 2 || h < 2 {
        return;
    }

    // Per-primaries luma weights — wide-gamut u8 sources go through
    // RowStream's Native zero-copy path with their bytes intact, so
    // the analyzer must use the right matrix for those bytes (BT.2020
    // for Rec.2020 sources, etc.) rather than the BT.601 sRGB
    // baseline. See `crate::luma::LumaWeights::for_primaries`.
    let weights = crate::luma::LumaWeights::for_primaries(stream.primaries());

    let stripe_step = compute_stripe_step(w, h, pixel_budget);
    let row_bytes = w * 3;
    let blocks_x = w / STRIPE_H;
    let total_stripes = h / STRIPE_H;

    // Stagger the starting stripe by a per-image deterministic phase
    // offset. Without this, fixed-pitch stripe sampling aliases against
    // any image structure that shares a period with the stripe step:
    // a 2 K-wide synthetic with 64-row text bands hit a measured 655 %
    // variance error vs full-scan ground truth at default budget. The
    // staggered version cuts that to ~94 % at zero locality cost and
    // zero accuracy regression on smooth / photographic inputs.
    //
    // Phase derived from `(width, height, sampled-row-byte-prefix)` so
    // the same image always staggers the same way. `stripe_step == 1`
    // (full-scan budget) collapses staggering to a no-op.
    let phase = if stripe_step <= 1 {
        0
    } else {
        compute_stripe_phase(w, h, stream, stripe_step)
    };

    // Stripe scratch holds 9 rows (the 8-row stripe + the lookahead
    // row for vertical gradient at the last interior row of the
    // stripe). Allocated once, reused across every active stripe.
    // 9 × max_width × 3 = ~108 kb at 4K width.
    let stripe_rows = STRIPE_H + 1;
    let mut stripe_buf = vec![0u8; stripe_rows * row_bytes];

    let mut stats = PixelStats::default();
    let mut sampled_pixels: u64 = 0;
    let mut sampled_interior: u64 = 0;
    let mut uniform_blocks: u32 = 0;
    let mut total_blocks: u32 = 0;
    let mut flat_color_blocks: u32 = 0;
    // Grayscale signal: count pixels where the per-pixel R / G / B
    // range is small. Threshold 4 matches the noise-tolerant range
    // used by FlatColorBlockRatio and absorbs JPEG-grade chroma noise
    // around true neutrals. Ratio computed at the end of the tier.
    // GrayscaleScore moved to the always-full-scan palette tier —
    // 100 % coverage required because downstream uses the score as a
    // binary classifier (`>= 0.99` ⇒ encode as grayscale). Stripe
    // sampling here would let a single colour pixel slip past the
    // gate at ~5 % budget. See `palette::scan_palette` and the
    // grayscale_score writeback in `lib.rs`.
    // SkinToneFraction and EdgeSlopeStdev are now computed inside
    // accumulate_row_simd alongside luma/chroma stats — see the
    // `skin_count`, `edge_grad_sum`, `edge_grad_sq_sum`,
    // `edge_grad_count` fields on `PixelStats`. The previous
    // separate scalar walks (`accumulate_per_pixel_extras_dispatch`,
    // `count_skin_tone_pixels`, `accumulate_edge_slope_sums`) added
    // 1-2 row passes over data already L1-resident from the SIMD
    // pass; folding them in saves the load traffic.
    // Palette counting moved to the always-full-scan `palette` tier
    // (see `palette::scan_palette`). Tier 1 used to do this here but
    // budget-sampled palette undercounts colours; the dedicated tier
    // counts every pixel for an exact `distinct_color_bins`.
    let mut block_var_min: f32 = f32::INFINITY;
    let mut block_var_max: f32 = 0.0;
    let mut block_var_sum: f64 = 0.0;

    let mut stripe_idx = phase;
    while stripe_idx < total_stripes {
        let y_start = stripe_idx * STRIPE_H;
        let stripe_end = (y_start + STRIPE_H).min(h);
        let lookahead_end = (stripe_end + 1).min(h);

        // Pre-fetch 8 stripe rows + 1 lookahead row.
        let avail = lookahead_end - y_start;
        for i in 0..avail {
            stream.fetch_into(
                (y_start + i) as u32,
                &mut stripe_buf[i * row_bytes..(i + 1) * row_bytes],
            );
        }

        // --- Stats + edges + colourfulness over each row in this stripe ---
        for dy in 0..STRIPE_H {
            let y_local = dy;
            if y_start + y_local >= h {
                break;
            }
            let row_off = y_local * row_bytes;
            let next_row_off = if y_start + y_local + 1 < h {
                Some((y_local + 1) * row_bytes)
            } else {
                None
            };

            accumulate_row_dispatch(
                &stripe_buf,
                row_off,
                next_row_off,
                w,
                &weights,
                dispatch.wants_full_kernel,
                dispatch.wants_skin,
                &mut stats,
            );

            // Laplacian: 3-row window. Skip the topmost row of the
            // image (no `prev_row` available) and the bottom row of
            // the stripe (will be picked up by the lookahead row of
            // *this* stripe — see the prev_row index below).
            //
            // Whole pass elided when the caller's `FeatureSet` doesn't
            // intersect `TIER1_EXTRAS_FEATURES` (no `LaplacianVariance`
            // request) — saves a separate SIMD row walk per interior
            // sampled row. Dominant `LaplacianVariance` consumer is
            // `FeatureSet::SUPPORTED`; orchestrator-style callers like
            // zenjpeg's `ADAPTIVE_FEATURES` don't request it.
            if dispatch.wants_laplacian
                && y_start + y_local >= 1
                && y_start + y_local + 1 < h
                && y_local >= 1
            {
                let prev_off = (y_local - 1) * row_bytes;
                let cur_off = row_off;
                let nxt_off = (y_local + 1) * row_bytes;
                let prev_row = &stripe_buf[prev_off..prev_off + row_bytes];
                let cur_row = &stripe_buf[cur_off..cur_off + row_bytes];
                let nxt_row = &stripe_buf[nxt_off..nxt_off + row_bytes];
                accumulate_laplacian_dispatch(prev_row, cur_row, nxt_row, w, &weights, &mut stats);
            }

            // SkinToneFraction + EdgeSlopeStdev were folded INTO
            // `accumulate_row_simd` — see the SIMD pass above and the
            // `skin_count` / `edge_grad_*` accumulators on `PixelStats`.
            // No separate per-pixel-extras row walk; one fewer load
            // pass per row, all classification work inside the
            // already-AVX2-active target_feature region.
            let _ = next_row_off; // still used by the SIMD edge pass
            let _ = row_bytes;
            sampled_pixels += w as u64;
            if next_row_off.is_some() {
                sampled_interior += (w - 1) as u64;
            }
        }

        // --- 8×8 block stats: luma uniformity + per-channel flat color ---
        // Skip if the stripe is short (no full 8-row block worth).
        let stripe_full_rows = (h - y_start).min(STRIPE_H);
        if stripe_full_rows == STRIPE_H {
            let s = stripe_block_stats_dispatch(
                &stripe_buf[..STRIPE_H * row_bytes],
                row_bytes,
                blocks_x,
            );
            uniform_blocks += s.uniform_blocks;
            flat_color_blocks += s.flat_color_blocks;
            total_blocks += blocks_x as u32;
            if blocks_x > 0 {
                if s.min_variance < block_var_min {
                    block_var_min = s.min_variance;
                }
                if s.max_variance > block_var_max {
                    block_var_max = s.max_variance;
                }
                block_var_sum += s.variance_sum;
            }
        } else {
            // Tail stripe: keep the scalar fallback for partial heights
            // (rare; only the last sampled stripe of a non-multiple-of-8
            // image height).
            for bx in 0..blocks_x {
                let mut sum: u32 = 0;
                let mut sq_sum: u32 = 0;
                let mut r_min: u8 = 255;
                let mut r_max: u8 = 0;
                let mut g_min: u8 = 255;
                let mut g_max: u8 = 0;
                let mut b_min: u8 = 255;
                let mut b_max: u8 = 0;
                for dy in 0..stripe_full_rows {
                    let base = dy * row_bytes + bx * STRIPE_H * 3;
                    for dx in 0..STRIPE_H {
                        let off = base + dx * 3;
                        let r = stripe_buf[off];
                        let g = stripe_buf[off + 1];
                        let b = stripe_buf[off + 2];
                        let l = (77u32 * r as u32 + 150 * g as u32 + 29 * b as u32) >> 8;
                        sum += l;
                        sq_sum += l * l;
                        if r < r_min {
                            r_min = r;
                        }
                        if r > r_max {
                            r_max = r;
                        }
                        if g < g_min {
                            g_min = g;
                        }
                        if g > g_max {
                            g_max = g;
                        }
                        if b < b_min {
                            b_min = b;
                        }
                        if b > b_max {
                            b_max = b;
                        }
                    }
                }
                let n = (STRIPE_H * STRIPE_H) as f32;
                let mean = sum as f32 / n;
                let var = (sq_sum as f32 / n - mean * mean).max(0.0);
                if var < 25.0 {
                    uniform_blocks += 1;
                }
                if r_max - r_min <= 4 && g_max - g_min <= 4 && b_max - b_min <= 4 {
                    flat_color_blocks += 1;
                }
                if var < block_var_min {
                    block_var_min = var;
                }
                if var > block_var_max {
                    block_var_max = var;
                }
                block_var_sum += var as f64;
                total_blocks += 1;
            }
        }

        stripe_idx += stripe_step;
    }

    // ---------- Reduce ----------
    let n = sampled_pixels as f64;
    if n < 1.0 {
        return;
    }
    let luma_mean = stats.luma_sum / n;
    out.variance = (stats.luma_sq_sum / n - luma_mean * luma_mean).max(0.0) as f32;
    out.edge_density = if sampled_interior > 0 {
        (stats.edge_count as f64 / sampled_interior as f64) as f32
    } else {
        0.0
    };
    let cb_mean = stats.cb_sum / n;
    let cr_mean = stats.cr_sum / n;
    let cb_var = (stats.cb_sq_sum / n - cb_mean * cb_mean).max(0.0);
    let cr_var = (stats.cr_sq_sum / n - cr_mean * cr_mean).max(0.0);
    out.chroma_complexity = (cb_var + cr_var).sqrt() as f32;
    out.uniformity = if total_blocks > 0 {
        uniform_blocks as f32 / total_blocks as f32
    } else {
        1.0
    };
    if stats.chroma_grad_count > 0 {
        let gc = stats.chroma_grad_count as f64;
        out.cb_sharpness = (stats.cb_grad_sum / gc) as f32;
        out.cr_sharpness = (stats.cr_grad_sum / gc) as f32;
    }
    // `distinct_color_bins` is populated by the always-full-scan
    // `palette` tier in `analyze_with`, AFTER tier 1 runs. Tier 1
    // doesn't touch palette features.
    out.flat_color_block_ratio = if total_blocks > 0 {
        flat_color_blocks as f32 / total_blocks as f32
    } else {
        0.0
    };
    // Hasler-Süsstrunk M3 colourfulness:
    //   M3 = sqrt(σ_rg² + σ_yb²) + 0.3 * sqrt(μ_rg² + μ_yb²)
    let mu_rg = stats.rg_sum / n;
    let mu_yb = stats.yb_sum / n;
    let var_rg = (stats.rg_sq_sum / n - mu_rg * mu_rg).max(0.0);
    let var_yb = (stats.yb_sq_sum / n - mu_yb * mu_yb).max(0.0);
    let sigma_term = (var_rg + var_yb).sqrt();
    let mu_term = (mu_rg * mu_rg + mu_yb * mu_yb).sqrt();
    // The three writes below land in `RawAnalysis` fields that are
    // cfg-gated behind the `experimental` cargo feature; the SIMD
    // accumulators that produced them are computed unconditionally
    // (lane-parallel — skipping individual lanes wouldn't save work)
    // but the field doesn't exist when `experimental` is off, so the
    // store is gated to match.
    #[cfg(feature = "experimental")]
    {
        out.colourfulness = (sigma_term + 0.3 * mu_term) as f32;
    }
    // Laplacian variance: variance of ∇²L over interior pixels.
    // Pixel-scale dependent: a sharp content edge spans more pixels
    // at higher resolution, so the per-pixel ∇² magnitude is smaller.
    // Compensate by scaling by sqrt(megapixels).
    #[cfg(feature = "experimental")]
    {
        out.laplacian_variance = if stats.laplacian_count > 0 {
            let lc = stats.laplacian_count as f64;
            let mu = stats.laplacian_sum / lc;
            let var = (stats.laplacian_sq_sum / lc - mu * mu).max(0.0);
            let mp = (w as f64 * h as f64 / 1_000_000.0).max(1e-3);
            ((var / 1e3) * mp.sqrt()) as f32
        } else {
            0.0
        };
    }
    // Variance heterogeneity: log10(1 + max_var / max(1, mean_var)).
    // Captures how much louder the loudest block is than typical.
    #[cfg(feature = "experimental")]
    {
        out.variance_spread = if total_blocks > 0 {
            let mean_var = (block_var_sum / total_blocks as f64) as f32;
            ((1.0 + block_var_max / mean_var.max(1.0)).log10()).max(0.0)
        } else {
            0.0
        };
    }
    // GrayscaleScore is no longer written here — see
    // `palette::scan_palette` and `lib.rs` for the full-scan path.
    // Skin-tone fraction: pigmentation-invariant chroma classifier
    // (Chai & Ngan 1999). Computed lane-wise inside the SIMD stats
    // pass — see `accumulate_row_simd::skin_count_v`.
    #[cfg(feature = "experimental")]
    {
        out.skin_tone_fraction = if sampled_pixels > 0 {
            (stats.skin_count as f64 / sampled_pixels as f64) as f32
        } else {
            0.0
        };
    }
    // Edge-slope stddev: dispersion of luma gradient magnitudes
    // among pixels that crossed `|∇L| > 20`. Folded branchlessly
    // into the SIMD edge inner loop — see
    // `accumulate_row_simd::edge_grad_*`.
    #[cfg(feature = "experimental")]
    {
        out.edge_slope_stdev = if stats.edge_grad_count >= 2 {
            let n = stats.edge_grad_count as f64;
            let mean = stats.edge_grad_sum / n;
            let var = (stats.edge_grad_sq_sum / n - mean * mean).max(0.0);
            var.sqrt() as f32
        } else {
            0.0
        };
    }
    // Suppress "unused variable" warnings on the always-computed
    // accumulators when the experimental writes are gated out.
    #[cfg(not(feature = "experimental"))]
    {
        let _ = (
            sigma_term,
            mu_term,
            block_var_max,
            block_var_sum,
            total_blocks,
        );
    }
}

fn compute_stripe_step(width: usize, height: usize, pixel_budget: usize) -> usize {
    let total_stripes = height / STRIPE_H;
    if total_stripes == 0 {
        return 1;
    }
    let pixels_per_stripe = width * STRIPE_H;
    if pixels_per_stripe == 0 {
        return 1;
    }
    let target_stripes = (pixel_budget / pixels_per_stripe).max(1).min(total_stripes);
    (total_stripes / target_stripes).max(1)
}

/// Per-image deterministic phase offset in `[0, stripe_step)` for the
/// staggered-stripe sampling pattern.
///
/// Hashes `(width, height, first 256 source bytes)` via FNV-1a so the
/// same image always picks the same phase — analysis stays
/// reproducible — while different images get different phases that
/// break aliasing against periodic image structure (text bands at the
/// stripe pitch, repeating UI elements, etc.).
///
/// The first 256 source bytes come from row 0 via `RowStream::borrow_row`,
/// which is cheap on the Native (RGB8) path and one row of conversion
/// scratch on the Convert path — that one-row cost is amortised before
/// the main stripe loop pulls every sampled stripe.
///
/// Off-corpus benchmarks: this single change cut the worst-case
/// `variance` error on a 64-row banded text input from ~655 % vs
/// full-scan ground truth down to ~94 %, with zero accuracy regression
/// on smooth or photographic inputs and zero added compute or memory
/// outside the 256-byte hash.
fn compute_stripe_phase(
    width: usize,
    height: usize,
    stream: &mut super::row_stream::RowStream<'_>,
    stripe_step: usize,
) -> usize {
    // FNV-1a 64-bit hash. Tiny, no deps, deterministic across builds.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h_state: u64 = FNV_OFFSET;
    for &b in &(width as u64).to_le_bytes() {
        h_state ^= b as u64;
        h_state = h_state.wrapping_mul(FNV_PRIME);
    }
    for &b in &(height as u64).to_le_bytes() {
        h_state ^= b as u64;
        h_state = h_state.wrapping_mul(FNV_PRIME);
    }
    // Mix in the first row's leading bytes so two same-dimensions
    // inputs still pick different phases. 256 bytes is enough to
    // distinguish any non-degenerate input; capped to the actual row
    // length on tiny images.
    let row = stream.borrow_row(0);
    let take = row.len().min(256);
    for &b in &row[..take] {
        h_state ^= b as u64;
        h_state = h_state.wrapping_mul(FNV_PRIME);
    }
    (h_state as usize) % stripe_step
}

/// Runtime dispatch wrapper for the magetypes f32x8 row pass.
#[allow(clippy::too_many_arguments)]
fn accumulate_row_dispatch(
    rgb: &[u8],
    row_off: usize,
    next_row_off: Option<usize>,
    width: usize,
    weights: &crate::luma::LumaWeights,
    full: bool,
    skin: bool,
    stats: &mut PixelStats,
) {
    let kr = weights.kr;
    let kg = weights.kg;
    let kb = weights.kb;
    // 8-arm dispatch on (BT601, FULL, SKIN). BT601=true const-folds
    // luma weights to immediate vfmadd operands. FULL=true unlocks
    // luma stats + Hasler M3 + edge-slope batching. SKIN=true unlocks
    // BT.601 chroma matrix + Chai-Ngan gate. Splitting SKIN off FULL
    // is the AVX2-register-pressure relief — `cargo asm` showed the
    // joint kernel spilled 12 vmovups + rebroadcast 13 constants
    // because all accumulators were live at once.
    //
    // Caller dispatch points:
    // - zenjpeg ADAPTIVE_FEATURES → `<true, false, false>` (no extras)
    // - just Variance / Colourfulness / EdgeSlope → `<*, true, false>`
    // - just SkinToneFraction → `<*, false, true>`
    // - FeatureSet::SUPPORTED → `<*, true, true>`
    let is_bt601 = weights.is_bt601_baseline();
    let row_stats = match (is_bt601, full, skin) {
        (true, true, true) => incant!(accumulate_row_simd::<true, true, true>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (true, true, false) => incant!(accumulate_row_simd::<true, true, false>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (true, false, true) => incant!(accumulate_row_simd::<true, false, true>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (true, false, false) => incant!(accumulate_row_simd::<true, false, false>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (false, true, true) => incant!(accumulate_row_simd::<false, true, true>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (false, true, false) => incant!(accumulate_row_simd::<false, true, false>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (false, false, true) => incant!(accumulate_row_simd::<false, false, true>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
        (false, false, false) => incant!(accumulate_row_simd::<false, false, false>(
            rgb, row_off, next_row_off, width, kr, kg, kb
        )),
    };
    stats.merge(&row_stats);
}

/// Aggregated per-stripe block stats. Returned from the SIMD kernel
/// so the outer reducer can fold across stripes.
#[derive(Default, Clone, Copy)]
pub(crate) struct StripeBlockStats {
    pub uniform_blocks: u32,
    pub flat_color_blocks: u32,
    /// Minimum per-block luma variance seen in this stripe.
    pub min_variance: f32,
    /// Maximum per-block luma variance seen in this stripe.
    pub max_variance: f32,
    /// Sum of per-block luma variance (for mean reduction).
    pub variance_sum: f64,
}

/// SIMD'd block-stats kernel for one full 8-row stripe. Returns
/// counts + variance min/max so the outer pipeline can compute the
/// `variance_spread` heterogeneity feature for free (no extra pass).
fn stripe_block_stats_dispatch(
    stripe_rows: &[u8], // exactly STRIPE_H * row_bytes bytes
    row_bytes: usize,
    blocks_x: usize,
) -> StripeBlockStats {
    incant!(stripe_block_stats_simd(stripe_rows, row_bytes, blocks_x))
}

/// Iterate the `blocks_x` 8×8 blocks in a complete (8-row) stripe and
/// classify each as uniform (luma variance < 25) and/or flat-color
/// (per-channel range ≤ 4). Returns `(uniform_blocks, flat_color_blocks)`.
///
/// Per-block work: 8 SIMD iters of 8 pixels each. Lanes hold f32 R/G/B
/// values for one block-row. Lane-wise min/max plus lane-summed luma /
/// luma² accumulators are reduced to scalar at the end of each block.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn stripe_block_stats_simd(
    token: Token,
    stripe_rows: &[u8],
    row_bytes: usize,
    blocks_x: usize,
) -> StripeBlockStats {
    let mut uniform_blocks: u32 = 0;
    let mut flat_color_blocks: u32 = 0;
    let mut min_variance: f32 = f32::INFINITY;
    let mut max_variance: f32 = 0.0;
    let mut variance_sum: f64 = 0.0;

    let block_n = (STRIPE_H * STRIPE_H) as f32;
    let inv_256_v = f32x8::splat(token, 1.0 / 256.0);
    let coef_r_v = f32x8::splat(token, 77.0);
    let coef_g_v = f32x8::splat(token, 150.0);
    let coef_b_v = f32x8::splat(token, 29.0);

    for bx in 0..blocks_x {
        let mut sum_v = f32x8::zero(token);
        let mut sq_sum_v = f32x8::zero(token);
        let mut r_min_v = f32x8::splat(token, 255.0);
        let mut r_max_v = f32x8::zero(token);
        let mut g_min_v = f32x8::splat(token, 255.0);
        let mut g_max_v = f32x8::zero(token);
        let mut b_min_v = f32x8::splat(token, 255.0);
        let mut b_max_v = f32x8::zero(token);

        for dy in 0..STRIPE_H {
            let base = dy * row_bytes + bx * STRIPE_H * 3;
            // Deinterleave 8 RGB pixels (24 bytes) into per-channel
            // f32 arrays. The fixed-size `&[u8; 24]` view proves the
            // size to LLVM, eliminates interior bounds checks, and
            // lets the autovectorizer (running under the per-tier
            // target_feature context that `#[magetypes]` set up) emit
            // a clean AVX2 / NEON-vld3 / WASM-shuffle deinterleave.
            let chunk: &[u8; 24] = (&stripe_rows[base..base + 24]).try_into().unwrap();
            let mut r_arr = [0.0f32; 8];
            let mut g_arr = [0.0f32; 8];
            let mut b_arr = [0.0f32; 8];
            for dx in 0..8 {
                r_arr[dx] = chunk[dx * 3] as f32;
                g_arr[dx] = chunk[dx * 3 + 1] as f32;
                b_arr[dx] = chunk[dx * 3 + 2] as f32;
            }
            let r_v = f32x8::load(token, &r_arr);
            let g_v = f32x8::load(token, &g_arr);
            let b_v = f32x8::load(token, &b_arr);

            // Luma in same units the scalar uses: (77R + 150G + 29B) >> 8.
            // We compute (77R + 150G + 29B) / 256 in f32; the f32 truncates
            // toward zero on cast back, equivalent to `>> 8` for non-negative
            // integer values that fit in f32 mantissa (always true here).
            let luma_v = (r_v * coef_r_v + g_v * coef_g_v + b_v * coef_b_v) * inv_256_v;
            // Floor to integer to match scalar `>> 8` behavior exactly.
            // f32 doesn't have a direct floor in the magetypes core; the
            // multiplication by 1/256 is exact for inputs that are
            // multiples of 256 (which the integer products are NOT in
            // general). We approximate via cast-to-i32-truncate-then-cast.
            // For this metric the 1-LSB drift is below the threshold's
            // sensitivity, so we skip the floor and use the f32 directly.
            sum_v += luma_v;
            sq_sum_v = luma_v.mul_add(luma_v, sq_sum_v);

            r_min_v = r_min_v.min(r_v);
            r_max_v = r_max_v.max(r_v);
            g_min_v = g_min_v.min(g_v);
            g_max_v = g_max_v.max(g_v);
            b_min_v = b_min_v.min(b_v);
            b_max_v = b_max_v.max(b_v);
        }

        let block_sum = sum_v.reduce_add();
        let block_sq_sum = sq_sum_v.reduce_add();
        let mean = block_sum / block_n;
        let var = (block_sq_sum / block_n - mean * mean).max(0.0);
        if var < 25.0 {
            uniform_blocks += 1;
        }
        if var < min_variance {
            min_variance = var;
        }
        if var > max_variance {
            max_variance = var;
        }
        variance_sum += var as f64;

        let r_min = r_min_v.reduce_min();
        let r_max = r_max_v.reduce_max();
        let g_min = g_min_v.reduce_min();
        let g_max = g_max_v.reduce_max();
        let b_min = b_min_v.reduce_min();
        let b_max = b_max_v.reduce_max();
        if (r_max - r_min) <= 4.0 && (g_max - g_min) <= 4.0 && (b_max - b_min) <= 4.0 {
            flat_color_blocks += 1;
        }
    }

    StripeBlockStats {
        uniform_blocks,
        flat_color_blocks,
        min_variance: if blocks_x == 0 { 0.0 } else { min_variance },
        max_variance,
        variance_sum,
    }
}

/// Single-row accumulator: luma/chroma stats + colourfulness + edge
/// count + per-channel chroma horizontal gradients.
///
/// Stats pass: explicit f32x8 SIMD via `#[magetypes]`. Per chunk we
/// load 8 pixels into r/g/b f32x8 vectors, compute luma/cb/cr/rg/yb
/// in lanes, and accumulate into 10 f32x8 lane-wise running sums.
/// Every `FLUSH` chunks we reduce the lane sums into f64 outer
/// accumulators to bound cumulative precision loss (f32 mantissa is
/// 24 bits; one chunk's contribution to a sum² is up to ~520 k for
/// luma², so 32 chunks keeps the partial sum well below 16 M).
///
/// Edge pass: still scalar (per-tier target_feature region from the
/// `#[magetypes]` macro lets LLVM autovec the simple stencil).
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn accumulate_row_simd<const BT601: bool, const FULL: bool, const SKIN: bool>(
    token: Token,
    rgb: &[u8],
    row_off: usize,
    next_row_off: Option<usize>,
    width: usize,
    kr: f32,
    kg: f32,
    kb: f32,
) -> PixelStats {
    // Const-fold luma weights for the BT.601 baseline (sRGB / BT.709 /
    // Unknown sources, the orchestrator hot path). When BT601 is true,
    // the `if` collapses at compile time and `kr/kg/kb` become
    // immediate operands of `vfmadd*` everywhere they appear in this
    // body — no register-loaded splats, three fewer YMM lanes live
    // through the chunk loop. When BT601 is false (BT.2020, P3,
    // AdobeRGB), the runtime values pass through unchanged.
    let (kr, kg, kb) = if BT601 {
        (0.299_f32, 0.587_f32, 0.114_f32)
    } else {
        (kr, kg, kb)
    };
    // Outer f64 accumulators — only touched during the periodic flush.
    let mut luma_sum: f64 = 0.0;
    let mut luma_sq_sum: f64 = 0.0;
    let mut cb_sum: f64 = 0.0;
    let mut cb_sq_sum: f64 = 0.0;
    let mut cr_sum: f64 = 0.0;
    let mut cr_sq_sum: f64 = 0.0;
    let mut edge_count: u64 = 0;
    let mut rg_sum: f64 = 0.0;
    let mut rg_sq_sum: f64 = 0.0;
    let mut yb_sum: f64 = 0.0;
    let mut yb_sq_sum: f64 = 0.0;

    let row = &rgb[row_off..row_off + width * 3];
    let next_row = next_row_off.map(|nr| &rgb[nr..nr + width * 3]);

    // Skin-tone + edge-slope accumulators (folded in below). Skin
    // tone is computed lane-wise in the f32x8 stats pass; edge slope
    // is folded into the scalar inner edge loop branchlessly.
    let mut skin_count: u64 = 0;
    let mut edge_grad_sum: f64 = 0.0;
    let mut edge_grad_sq_sum: f64 = 0.0;
    let mut edge_grad_count: u64 = 0;

    // ---- f32x8 stats pass: 8 pixels (24 bytes) per chunk ----
    let kr_v = f32x8::splat(token, kr);
    let kg_v = f32x8::splat(token, kg);
    let kb_v = f32x8::splat(token, kb);
    let inv_255_v = f32x8::splat(token, 1.0 / 255.0);
    let half_v = f32x8::splat(token, 0.5);

    // BT.601 chroma encoding constants. These are FIXED (independent
    // of source primaries) — they define the encoder's YCbCr space,
    // which is what the Chai-Ngan skin-tone classifier was calibrated
    // against. The per-primaries `kr/kg/kb` only affect luma.
    let cb_kr_v = f32x8::splat(token, -0.168736);
    let cb_kg_v = f32x8::splat(token, -0.331264);
    let cb_kb_v = f32x8::splat(token, 0.500000);
    let cr_kr_v = f32x8::splat(token, 0.500000);
    let cr_kg_v = f32x8::splat(token, -0.418688);
    let cr_kb_v = f32x8::splat(token, -0.081312);
    let off_128_v = f32x8::splat(token, 128.0);
    // Chai-Ngan (1999) gates: Y in [40, 240], Cb in [77, 127],
    // Cr in [133, 173]. All compared in float-domain u8 space —
    // the gate margins (≥ 5 units) absorb any 1-LSB rounding drift
    // between the float SIMD path and the integer scalar tail.
    let y_lo_v = f32x8::splat(token, 40.0);
    let y_hi_v = f32x8::splat(token, 240.0);
    let cb_lo_v = f32x8::splat(token, 77.0);
    let cb_hi_v = f32x8::splat(token, 127.0);
    let cr_lo_v = f32x8::splat(token, 133.0);
    let cr_hi_v = f32x8::splat(token, 173.0);
    let one_v = f32x8::splat(token, 1.0);
    let zero_v = f32x8::zero(token);

    let mut luma_sum_v = f32x8::zero(token);
    let mut luma_sq_v = f32x8::zero(token);
    let mut cb_sum_v = f32x8::zero(token);
    let mut cb_sq_v = f32x8::zero(token);
    let mut cr_sum_v = f32x8::zero(token);
    let mut cr_sq_v = f32x8::zero(token);
    let mut rg_sum_v = f32x8::zero(token);
    let mut rg_sq_v = f32x8::zero(token);
    let mut yb_sum_v = f32x8::zero(token);
    let mut yb_sq_v = f32x8::zero(token);
    let mut skin_count_v = f32x8::zero(token);

    const FLUSH: usize = 32;
    let mut iters_since_flush = 0usize;

    let chunks = row.chunks_exact(24);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let c: &[u8; 24] = chunk.try_into().unwrap();
        // Deinterleave 8 RGB pixels into per-channel f32 arrays.
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        for i in 0..8 {
            r_arr[i] = c[i * 3] as f32;
            g_arr[i] = c[i * 3 + 1] as f32;
            b_arr[i] = c[i * 3 + 2] as f32;
        }
        let r = f32x8::load(token, &r_arr);
        let g = f32x8::load(token, &g_arr);
        let b = f32x8::load(token, &b_arr);

        // BT.601 luma: l = 0.299·r + 0.587·g + 0.114·b
        let l = r.mul_add(kr_v, g.mul_add(kg_v, b * kb_v));
        // Chroma stats (simplified): cb_stat = (b − l) / 255;
        // cr_stat = (r − l) / 255. Always-on (drives chroma_complexity
        // and Cb/Cr sharpness shape signals).
        let cb = (b - l) * inv_255_v;
        let cr = (r - l) * inv_255_v;

        cb_sum_v += cb;
        cb_sq_v = cb.mul_add(cb, cb_sq_v);
        cr_sum_v += cr;
        cr_sq_v = cr.mul_add(cr, cr_sq_v);

        // FULL-only accumulators: luma stats (Variance), Hasler M3
        // (Colourfulness). Const-folds away on `!FULL`.
        if FULL {
            // Hasler M3: rg = r − g; yb = 0.5·(r + g) − b
            let rg = r - g;
            let yb = (r + g).mul_add(half_v, -b);
            luma_sum_v += l;
            luma_sq_v = l.mul_add(l, luma_sq_v);
            rg_sum_v += rg;
            rg_sq_v = rg.mul_add(rg, rg_sq_v);
            yb_sum_v += yb;
            yb_sq_v = yb.mul_add(yb, yb_sq_v);
        }
        // SKIN-only accumulators: BT.601 chroma matrix in [0, 255]
        // for the Chai-Ngan skin-tone gate. Const-folds away on
        // `!SKIN`. Independent of FULL — `cargo asm` showed the
        // joint kernel spilled 12 vmovups + rebroadcast 13 constants
        // because all accumulators were live; peeling SKIN off FULL
        // shrinks the AVX2 register pressure for both halves.
        if SKIN {
            let cb_u8 =
                r.mul_add(cb_kr_v, g.mul_add(cb_kg_v, b.mul_add(cb_kb_v, off_128_v)));
            let cr_u8 =
                r.mul_add(cr_kr_v, g.mul_add(cr_kg_v, b.mul_add(cr_kb_v, off_128_v)));
            let m_y_lo = l.simd_ge(y_lo_v);
            let m_y_hi = l.simd_le(y_hi_v);
            let m_cb_lo = cb_u8.simd_ge(cb_lo_v);
            let m_cb_hi = cb_u8.simd_le(cb_hi_v);
            let m_cr_lo = cr_u8.simd_ge(cr_lo_v);
            let m_cr_hi = cr_u8.simd_le(cr_hi_v);
            let skin = m_y_lo & m_y_hi & m_cb_lo & m_cb_hi & m_cr_lo & m_cr_hi;
            skin_count_v += f32x8::blend(skin, one_v, zero_v);
        }

        iters_since_flush += 1;
        if iters_since_flush >= FLUSH {
            cb_sum += cb_sum_v.reduce_add() as f64;
            cb_sq_sum += cb_sq_v.reduce_add() as f64;
            cr_sum += cr_sum_v.reduce_add() as f64;
            cr_sq_sum += cr_sq_v.reduce_add() as f64;
            cb_sum_v = f32x8::zero(token);
            cb_sq_v = f32x8::zero(token);
            cr_sum_v = f32x8::zero(token);
            cr_sq_v = f32x8::zero(token);
            if FULL {
                luma_sum += luma_sum_v.reduce_add() as f64;
                luma_sq_sum += luma_sq_v.reduce_add() as f64;
                rg_sum += rg_sum_v.reduce_add() as f64;
                rg_sq_sum += rg_sq_v.reduce_add() as f64;
                yb_sum += yb_sum_v.reduce_add() as f64;
                yb_sq_sum += yb_sq_v.reduce_add() as f64;
                luma_sum_v = f32x8::zero(token);
                luma_sq_v = f32x8::zero(token);
                rg_sum_v = f32x8::zero(token);
                rg_sq_v = f32x8::zero(token);
                yb_sum_v = f32x8::zero(token);
                yb_sq_v = f32x8::zero(token);
            }
            if SKIN {
                skin_count += skin_count_v.reduce_add() as u64;
                skin_count_v = f32x8::zero(token);
            }
            iters_since_flush = 0;
        }
    }
    // Final flush of SIMD partials.
    cb_sum += cb_sum_v.reduce_add() as f64;
    cb_sq_sum += cb_sq_v.reduce_add() as f64;
    cr_sum += cr_sum_v.reduce_add() as f64;
    cr_sq_sum += cr_sq_v.reduce_add() as f64;
    if FULL {
        luma_sum += luma_sum_v.reduce_add() as f64;
        luma_sq_sum += luma_sq_v.reduce_add() as f64;
        rg_sum += rg_sum_v.reduce_add() as f64;
        rg_sq_sum += rg_sq_v.reduce_add() as f64;
        yb_sum += yb_sum_v.reduce_add() as f64;
        yb_sq_sum += yb_sq_v.reduce_add() as f64;
    }
    if SKIN {
        skin_count += skin_count_v.reduce_add() as u64;
    }

    // Scalar tail for ≤7 leftover pixels — same float-domain math as
    // the SIMD lanes for bit-equal cross-tail consistency on the
    // skin-tone gate. FULL-only accumulators are gated identically.
    for px in remainder.chunks_exact(3) {
        let r = px[0] as f32;
        let g = px[1] as f32;
        let b = px[2] as f32;
        let l = kr * r + kg * g + kb * b;
        let cb = (b - l) * (1.0 / 255.0);
        let cr = (r - l) * (1.0 / 255.0);
        cb_sum += cb as f64;
        cb_sq_sum += (cb * cb) as f64;
        cr_sum += cr as f64;
        cr_sq_sum += (cr * cr) as f64;
        if FULL {
            luma_sum += l as f64;
            luma_sq_sum += (l * l) as f64;
            let rg = r - g;
            let yb = 0.5 * (r + g) - b;
            rg_sum += rg as f64;
            rg_sq_sum += (rg * rg) as f64;
            yb_sum += yb as f64;
            yb_sq_sum += (yb * yb) as f64;
        }
        if SKIN {
            // BT.601 chroma in u8 representation for the skin gate.
            let cb_u8 = -0.168736 * r - 0.331264 * g + 0.500 * b + 128.0;
            let cr_u8 = 0.500 * r - 0.418688 * g - 0.081312 * b + 128.0;
            let in_skin = (40.0..=240.0).contains(&l)
                && (77.0..=127.0).contains(&cb_u8)
                && (133.0..=173.0).contains(&cr_u8);
            skin_count += in_skin as u64;
        }
    }

    // ---- Edges + chroma gradients: 8-pixel chunks with right & down neighbors ----
    let has_next = next_row.is_some();
    let nr = next_row.unwrap_or(row);

    let mut cb_grad_sum: f64 = 0.0;
    let mut cr_grad_sum: f64 = 0.0;
    let mut chroma_grad_count: u64 = 0;

    if width > 1 {
        let edge_end = (width - 1) * 3;
        let edge_row = &row[..edge_end];
        let right_row = &row[3..];
        let edge_chunks = edge_row.chunks_exact(24);
        let _edge_rem = edge_chunks.remainder();
        let mut right_iter = right_row.chunks_exact(24);
        let mut nr_iter = nr.chunks_exact(24);

        for chunk in edge_chunks {
            let c: &[u8; 24] = chunk.try_into().unwrap();
            let r_chunk: &[u8; 24] = right_iter.next().unwrap().try_into().unwrap();
            let d_chunk: &[u8; 24] = nr_iter.next().unwrap().try_into().unwrap();
            // Stage gradient and mask values across the 8 lanes so the
            // sqrt + mask-multiply that produces `edge_grad_sum` /
            // `edge_grad_sq_sum` can run as ONE f32x8 sqrt + 2
            // `reduce_add`s instead of 8 sequential scalar sqrts. The
            // scalar inner pass still runs (chroma gradients + branch-
            // free counter increments) — only the per-pixel sqrt is
            // hoisted out of the loop body.
            let mut grad_sq_arr = [0.0f32; 8];
            let mut mask_arr = [0.0f32; 8];
            for i in 0..8 {
                let cr_ = c[i * 3] as f32;
                let cg_ = c[i * 3 + 1] as f32;
                let cb_ = c[i * 3 + 2] as f32;
                let l = kr * cr_ + kg * cg_ + kb * cb_;
                let rr_ = r_chunk[i * 3] as f32;
                let rg_ = r_chunk[i * 3 + 1] as f32;
                let rb_ = r_chunk[i * 3 + 2] as f32;
                let lr = kr * rr_ + kg * rg_ + kb * rb_;
                let gx = lr - l;
                let mut grad_sq = gx * gx;

                let cb_cur = (cb_ - l) / 255.0;
                let cb_right = (rb_ - lr) / 255.0;
                let cr_cur = (cr_ - l) / 255.0;
                let cr_right = (rr_ - lr) / 255.0;
                cb_grad_sum += (cb_right - cb_cur).abs() as f64;
                cr_grad_sum += (cr_right - cr_cur).abs() as f64;
                chroma_grad_count += 1;

                if has_next {
                    let ld = kr * d_chunk[i * 3] as f32
                        + kg * d_chunk[i * 3 + 1] as f32
                        + kb * d_chunk[i * 3 + 2] as f32;
                    grad_sq += (ld - l) * (ld - l);
                }
                let crossed = grad_sq > EDGE_THRESH_SQ;
                edge_count += crossed as u64;
                if FULL {
                    edge_grad_count += crossed as u64;
                    grad_sq_arr[i] = grad_sq;
                    mask_arr[i] = crossed as u32 as f32;
                }
            }
            if FULL {
                // ONE batched sqrt for all 8 lanes via rsqrt_approx:
                // `sqrt(x) = x * (1 / sqrt(x))`. ~3× faster than
                // batch `sqrt()`, ~12-bit precision (well above the
                // edge-slope stddev's noise floor). Clamp grad_sq to
                // [1.0, ∞) so non-edge lanes (mask=0) don't produce
                // `0 * Inf = NaN` from `0 * rsqrt_approx(0)`.
                let grad_sq_v = f32x8::load(token, &grad_sq_arr);
                let mask_v = f32x8::load(token, &mask_arr);
                let one_v = f32x8::splat(token, 1.0);
                let safe_grad_sq = grad_sq_v.max(one_v);
                let inv_sqrt = safe_grad_sq.rsqrt_approx();
                let g_mag_v = grad_sq_v * inv_sqrt * mask_v;
                let g_sq_masked_v = grad_sq_v * mask_v;
                edge_grad_sum += g_mag_v.reduce_add() as f64;
                edge_grad_sq_sum += g_sq_masked_v.reduce_add() as f64;
            }
        }

        // Scalar tail for the remaining 0..7 edge pixels.
        //
        // Accumulates the **same** four reductions as the SIMD edge
        // loop above (luma edge_count + cb/cr gradient sums + count).
        // Earlier revisions only updated `edge_count` here, silently
        // dropping the rightmost 1–7 column positions per row from
        // the chroma sharpness signal — a measurable undercount on
        // small or non-multiple-of-8 widths.
        let processed = (width - 1) / 8 * 8;
        for x in processed..width - 1 {
            let off = row_off + x * 3;
            let cr_ = rgb[off] as f32;
            let cg_ = rgb[off + 1] as f32;
            let cb_ = rgb[off + 2] as f32;
            let l = kr * cr_ + kg * cg_ + kb * cb_;
            let roff = row_off + (x + 1) * 3;
            let rr_ = rgb[roff] as f32;
            let rg_ = rgb[roff + 1] as f32;
            let rb_ = rgb[roff + 2] as f32;
            let lr = kr * rr_ + kg * rg_ + kb * rb_;
            let gx = lr - l;
            let mut grad_sq = gx * gx;
            // Chroma gradients (matched against the SIMD edge loop's
            // same definitions: Cb = (B−Y)/255, Cr = (R−Y)/255).
            let cb_cur = (cb_ - l) / 255.0;
            let cb_right = (rb_ - lr) / 255.0;
            let cr_cur = (cr_ - l) / 255.0;
            let cr_right = (rr_ - lr) / 255.0;
            cb_grad_sum += (cb_right - cb_cur).abs() as f64;
            cr_grad_sum += (cr_right - cr_cur).abs() as f64;
            chroma_grad_count += 1;
            if has_next {
                let doff = next_row_off.unwrap() + x * 3;
                let ld =
                    kr * rgb[doff] as f32 + kg * rgb[doff + 1] as f32 + kb * rgb[doff + 2] as f32;
                grad_sq += (ld - l) * (ld - l);
            }
            let crossed = grad_sq > EDGE_THRESH_SQ;
            edge_count += crossed as u64;
            if FULL {
                let mask = crossed as u32 as f32;
                edge_grad_count += crossed as u64;
                let g_mag = grad_sq.sqrt();
                edge_grad_sum += (g_mag * mask) as f64;
                edge_grad_sq_sum += (grad_sq * mask) as f64;
            }
        }
    }

    PixelStats {
        luma_sum,
        luma_sq_sum,
        cb_sum,
        cb_sq_sum,
        cr_sum,
        cr_sq_sum,
        edge_count,
        cb_grad_sum,
        cr_grad_sum,
        chroma_grad_count,
        skin_count,
        edge_grad_sum,
        edge_grad_sq_sum,
        edge_grad_count,
        rg_sum,
        rg_sq_sum,
        yb_sum,
        yb_sq_sum,
        // Laplacian fields are populated by `accumulate_laplacian_row`
        // below — accumulate_row only sees a 2-row window so it can't
        // compute a 4-neighbour Laplacian.
        laplacian_sum: 0.0,
        laplacian_sq_sum: 0.0,
        laplacian_count: 0,
    }
}

/// Single-row Laplacian-variance accumulator. magetypes f32x8 SIMD.
///
/// Precomputes BT.601 luma into 3 row-scratch buffers (linear u8→f32),
/// then runs the 5-tap Laplacian stencil over them with 8 pixels per
/// SIMD iter. The stencil reads from contiguous f32 arrays so the
/// shifted-neighbour loads are aligned and pure mul/add — LLVM emits
/// the FMA chain directly.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn accumulate_laplacian_simd<const BT601: bool>(
    token: Token,
    prev_row: &[u8],
    cur_row: &[u8],
    next_row: &[u8],
    width: usize,
    kr: f32,
    kg: f32,
    kb: f32,
) -> (f64, f64, u64) {
    let (kr, kg, kb) = if BT601 {
        (0.299_f32, 0.587_f32, 0.114_f32)
    } else {
        (kr, kg, kb)
    };
    if width < 3 {
        return (0.0, 0.0, 0);
    }
    let mut prev_l = vec![0.0f32; width];
    let mut cur_l = vec![0.0f32; width];
    let mut next_l = vec![0.0f32; width];
    for x in 0..width {
        let off = x * 3;
        prev_l[x] = kr * prev_row[off] as f32
            + kg * prev_row[off + 1] as f32
            + kb * prev_row[off + 2] as f32;
        cur_l[x] =
            kr * cur_row[off] as f32 + kg * cur_row[off + 1] as f32 + kb * cur_row[off + 2] as f32;
        next_l[x] = kr * next_row[off] as f32
            + kg * next_row[off + 1] as f32
            + kb * next_row[off + 2] as f32;
    }

    // Stencil over interior columns 1..width-1. Process 8 pixels per
    // SIMD iter using f32x8 lanes; tail handled scalar.
    let mut lap_sum: f64 = 0.0;
    let mut lap_sq_sum: f64 = 0.0;
    let four_v = f32x8::splat(token, 4.0);
    let mut sum_v = f32x8::zero(token);
    let mut sq_v = f32x8::zero(token);
    let mut count: u64 = 0;

    let interior_end = width - 1;
    let chunks = (interior_end - 1) / 8;
    const FLUSH: usize = 32;
    let mut iters_since_flush = 0usize;
    for ci in 0..chunks {
        let start = 1 + ci * 8;
        // Overlapping loads from contiguous f32 row arrays — same
        // pattern Tier 2 uses. fixed-size [..; 8] proves length to LLVM.
        let lc_v = f32x8::load(token, (&cur_l[start..start + 8]).try_into().unwrap());
        let ll_v = f32x8::load(token, (&cur_l[start - 1..start + 7]).try_into().unwrap());
        let lr_v = f32x8::load(token, (&cur_l[start + 1..start + 9]).try_into().unwrap());
        let lu_v = f32x8::load(token, (&prev_l[start..start + 8]).try_into().unwrap());
        let ld_v = f32x8::load(token, (&next_l[start..start + 8]).try_into().unwrap());
        let lap = ll_v + lr_v + lu_v + ld_v - four_v * lc_v;
        sum_v += lap;
        sq_v = lap.mul_add(lap, sq_v);
        count += 8;
        iters_since_flush += 1;
        if iters_since_flush >= FLUSH {
            lap_sum += sum_v.reduce_add() as f64;
            lap_sq_sum += sq_v.reduce_add() as f64;
            sum_v = f32x8::zero(token);
            sq_v = f32x8::zero(token);
            iters_since_flush = 0;
        }
    }
    lap_sum += sum_v.reduce_add() as f64;
    lap_sq_sum += sq_v.reduce_add() as f64;

    // Scalar tail for ≤7 leftover interior pixels.
    let tail_start = 1 + chunks * 8;
    for x in tail_start..interior_end {
        let lc = cur_l[x];
        let ll = cur_l[x - 1];
        let lr = cur_l[x + 1];
        let lu = prev_l[x];
        let ld = next_l[x];
        let lap = ll + lr + lu + ld - 4.0 * lc;
        lap_sum += lap as f64;
        lap_sq_sum += (lap * lap) as f64;
        count += 1;
    }

    (lap_sum, lap_sq_sum, count)
}

fn accumulate_laplacian_dispatch(
    prev_row: &[u8],
    cur_row: &[u8],
    next_row: &[u8],
    width: usize,
    weights: &crate::luma::LumaWeights,
    stats: &mut PixelStats,
) {
    let kr = weights.kr;
    let kg = weights.kg;
    let kb = weights.kb;
    let (s, sq, n) = if weights.is_bt601_baseline() {
        incant!(accumulate_laplacian_simd::<true>(
            prev_row, cur_row, next_row, width, kr, kg, kb
        ))
    } else {
        incant!(accumulate_laplacian_simd::<false>(
            prev_row, cur_row, next_row, width, kr, kg, kb
        ))
    };
    stats.laplacian_sum += s;
    stats.laplacian_sq_sum += sq;
    stats.laplacian_count += n;
}

/// Per-row scalar grayscale-pixel count. A pixel is "grayscale" iff
/// its `max(|R-G|, |G-B|, |R-B|)` is below `GRAYSCALE_THRESHOLD`. The
/// threshold matches FlatColorBlockRatio (4) so the signal is robust
/// to the same JPEG-grade chroma noise that already triggers flat-
/// block classification, without requiring a second tunable.
///
/// Compiled at the per-tier `target_feature` regime via `#[autoversion]`
/// so LLVM autovectorizes the channel-gap compare into pmaxub /
/// pminub on x86_64 (and equivalent on NEON / WASM).
/// Count pixels in the canonical YCbCr skin-tone region. The
/// chrominance bounds `Cb ∈ [77, 127]` and `Cr ∈ [133, 173]` are the
/// Chai & Ngan (1999) values, which generalise across all skin
/// pigmentations because chroma quantifies hue, not lightness. The
/// luma gate `Y ∈ [40, 240]` covers deep shadow on dark skin to
/// bright highlight on light skin without rejecting either end.
///
/// Integer BT.601 conversion matches the rest of the analyzer
/// (matches the qr/qg/qb 77/150/29 fixed-point luma already used by
/// `count_grayscale_pixels` callers, and the Cb/Cr coefficients used
/// by tier3 — `(-43·R - 85·G + 128·B + 128) >> 8` for Cb-128, etc.).
#[archmage::autoversion(v4x, v4, v3, neon, scalar)]
fn count_skin_tone_pixels(row: &[u8]) -> u64 {
    let mut count: u64 = 0;
    for px in row.chunks_exact(3) {
        let r = px[0] as i32;
        let g = px[1] as i32;
        let b = px[2] as i32;
        let y = (77 * r + 150 * g + 29 * b) >> 8;
        // Cb / Cr in [0, 255] u8 representation.
        let cb = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
        let cr = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
        if (40..=240).contains(&y) && (77..=127).contains(&cb) && (133..=173).contains(&cr) {
            count += 1;
        }
    }
    count
}

/// Output of the fused per-pixel scalar pass.
#[derive(Default, Clone, Copy)]
struct PerPixelExtras {
    skin_tone: u64,
    edge_sum: f64,
    edge_sq_sum: f64,
    edge_count: u64,
}

/// Fused per-pixel scalar pass producing skin-tone count and
/// edge-slope (sum, sum², count) in a single walk over the row.
/// `GrayscaleScore` was promoted to the always-full-scan palette tier
/// (100 % coverage required for the binary classifier — see
/// `palette::scan_palette`).
///
/// Autoversioned to v4x / v4 / v3 / NEON / scalar so each tier picks
/// up the appropriate target_feature region. The `chunks_exact(3)`
/// pattern over the row's RGB triplets gives LLVM bounds-check-free
/// indexing on the inner loads.
#[archmage::autoversion(v4x, v4, v3, neon, scalar)]
fn accumulate_per_pixel_extras_dispatch(
    row: &[u8],
    next_row: Option<&[u8]>,
    width: usize,
    weights: &crate::luma::LumaWeights,
) -> PerPixelExtras {
    let mut out = PerPixelExtras::default();
    if width < 1 {
        return out;
    }
    let qr = weights.qr;
    let qg = weights.qg;
    let qb = weights.qb;
    let row_bytes = width * 3;
    let row_full = &row[..row_bytes];

    // First sweep: skin-tone, no neighbour deps.
    for px in row_full.chunks_exact(3) {
        let r_i = px[0] as i32;
        let g_i = px[1] as i32;
        let b_i = px[2] as i32;
        // Skin-tone: BT.601 fixed-point Y/Cb/Cr in u8.
        let y = (qr * r_i + qg * g_i + qb * b_i) >> 8;
        let cb = ((-43 * r_i - 85 * g_i + 128 * b_i) >> 8) + 128;
        let cr = ((128 * r_i - 107 * g_i - 21 * b_i) >> 8) + 128;
        if (40..=240).contains(&y) && (77..=127).contains(&cb) && (133..=173).contains(&cr) {
            out.skin_tone += 1;
        }
    }

    // Second sweep: edge-slope, needs current + right + below.
    if width >= 2 {
        let row_left = &row[..row_bytes - 3];
        let row_right = &row[3..row_bytes];
        let mut sum: f64 = 0.0;
        let mut sq_sum: f64 = 0.0;
        let mut count: u64 = 0;
        match next_row {
            Some(nr) => {
                let nr = &nr[..row_bytes - 3];
                for ((cur, right), down) in row_left
                    .chunks_exact(3)
                    .zip(row_right.chunks_exact(3))
                    .zip(nr.chunks_exact(3))
                {
                    let l = (qr * cur[0] as i32 + qg * cur[1] as i32 + qb * cur[2] as i32) >> 8;
                    let lr =
                        (qr * right[0] as i32 + qg * right[1] as i32 + qb * right[2] as i32) >> 8;
                    let ld = (qr * down[0] as i32 + qg * down[1] as i32 + qb * down[2] as i32) >> 8;
                    let dx = lr - l;
                    let dy = ld - l;
                    let g_sq = (dx * dx + dy * dy) as f64;
                    if g_sq > 400.0 {
                        let g = g_sq.sqrt();
                        sum += g;
                        sq_sum += g * g;
                        count += 1;
                    }
                }
            }
            None => {
                for (cur, right) in row_left.chunks_exact(3).zip(row_right.chunks_exact(3)) {
                    let l = (qr * cur[0] as i32 + qg * cur[1] as i32 + qb * cur[2] as i32) >> 8;
                    let lr =
                        (qr * right[0] as i32 + qg * right[1] as i32 + qb * right[2] as i32) >> 8;
                    let dx = lr - l;
                    let g_sq = (dx * dx) as f64;
                    if g_sq > 400.0 {
                        let g = g_sq.sqrt();
                        sum += g;
                        sq_sum += g * g;
                        count += 1;
                    }
                }
            }
        }
        out.edge_sum = sum;
        out.edge_sq_sum = sq_sum;
        out.edge_count = count;
    }

    out
}

/// Edge-slope dispatcher: pick the chunked autoversioned kernel based
/// on whether a next row is available. Either kernel walks the row
/// in 3-byte pixel chunks paired with a 3-byte-shifted view of the
/// same row (the bounds-check-free fixed-array indexing the
/// autoversion macro turns into per-arch `pmaddubsw` / `pmullw` /
/// FMA on aligned-by-construction loads).
fn accumulate_edge_slope_sums(
    row: &[u8],
    next_row: Option<&[u8]>,
    width: usize,
    weights: &crate::luma::LumaWeights,
    grad_sum: &mut f64,
    grad_sq_sum: &mut f64,
    grad_count: &mut u64,
) {
    if width < 2 {
        return;
    }
    let row_bytes = width * 3;
    let row_left = &row[..row_bytes - 3];
    let row_right = &row[3..row_bytes];
    let mut sum: f64 = 0.0;
    let mut sq_sum: f64 = 0.0;
    let mut count: u64 = 0;
    match next_row {
        Some(nr) => {
            let nr = &nr[..row_bytes - 3];
            accumulate_edge_slope_with_next(
                row_left,
                row_right,
                nr,
                weights.qr,
                weights.qg,
                weights.qb,
                &mut sum,
                &mut sq_sum,
                &mut count,
            );
        }
        None => {
            accumulate_edge_slope_horizontal(
                row_left,
                row_right,
                weights.qr,
                weights.qg,
                weights.qb,
                &mut sum,
                &mut sq_sum,
                &mut count,
            );
        }
    }
    *grad_sum += sum;
    *grad_sq_sum += sq_sum;
    *grad_count += count;
}

/// Edge-slope kernel for the "next row exists" case (interior rows).
/// Walks row pixels paired with their right-neighbour and the same
/// column in the next row. `chunks_exact(3)` and `zip` over equal-
/// length slices give LLVM bounds-check-free indexing on the inner
/// triplet of u8 loads, which the per-arch target_feature macro
/// expansion then vectorises.
#[archmage::autoversion(v4x, v4, v3, neon, scalar)]
fn accumulate_edge_slope_with_next(
    row_left: &[u8],
    row_right: &[u8],
    next_row: &[u8],
    qr: i32,
    qg: i32,
    qb: i32,
    sum: &mut f64,
    sq_sum: &mut f64,
    count: &mut u64,
) {
    let mut s: f64 = 0.0;
    let mut sq: f64 = 0.0;
    let mut n: u64 = 0;
    for ((cur, right), down) in row_left
        .chunks_exact(3)
        .zip(row_right.chunks_exact(3))
        .zip(next_row.chunks_exact(3))
    {
        let l = (qr * cur[0] as i32 + qg * cur[1] as i32 + qb * cur[2] as i32) >> 8;
        let lr = (qr * right[0] as i32 + qg * right[1] as i32 + qb * right[2] as i32) >> 8;
        let ld = (qr * down[0] as i32 + qg * down[1] as i32 + qb * down[2] as i32) >> 8;
        let dx = lr - l;
        let dy = ld - l;
        let g_sq = (dx * dx + dy * dy) as f64;
        if g_sq > 400.0 {
            let g = g_sq.sqrt();
            s += g;
            sq += g * g;
            n += 1;
        }
    }
    *sum += s;
    *sq_sum += sq;
    *count += n;
}

/// Edge-slope kernel for the "no next row" case (last image row).
/// Same pattern as the `with_next` kernel but without the down-row
/// load, so `dy = 0` and the threshold is purely against `dx²`.
#[archmage::autoversion(v4x, v4, v3, neon, scalar)]
fn accumulate_edge_slope_horizontal(
    row_left: &[u8],
    row_right: &[u8],
    qr: i32,
    qg: i32,
    qb: i32,
    sum: &mut f64,
    sq_sum: &mut f64,
    count: &mut u64,
) {
    let mut s: f64 = 0.0;
    let mut sq: f64 = 0.0;
    let mut n: u64 = 0;
    for (cur, right) in row_left.chunks_exact(3).zip(row_right.chunks_exact(3)) {
        let l = (qr * cur[0] as i32 + qg * cur[1] as i32 + qb * cur[2] as i32) >> 8;
        let lr = (qr * right[0] as i32 + qg * right[1] as i32 + qb * right[2] as i32) >> 8;
        let dx = lr - l;
        let g_sq = (dx * dx) as f64;
        if g_sq > 400.0 {
            let g = g_sq.sqrt();
            s += g;
            sq += g * g;
            n += 1;
        }
    }
    *sum += s;
    *sq_sum += sq;
    *count += n;
}
