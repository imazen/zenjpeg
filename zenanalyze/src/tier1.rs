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

const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;
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
    }
}

/// Populate Tier 1 fields on `out`. Other fields are left untouched.
///
/// `pixel_budget` controls the stripe step (see [`compute_stripe_step`]).
/// Pass [`DEFAULT_PIXEL_BUDGET`] to match the oracle-trained reference
/// behavior; pass a smaller value for proxy-server speed (with reduced
/// feature precision on multi-megapixel inputs).
pub fn extract_tier1_into(out: &mut RawAnalysis, stream: &mut RowStream<'_>, pixel_budget: usize) {
    let w = stream.width() as usize;
    let h = stream.height() as usize;
    if w < 2 || h < 2 {
        return;
    }

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
    // 9 × max_width × 3 = ~108 KB at 4K width.
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
    let mut grayscale_pixels: u64 = 0;
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

            accumulate_row_dispatch(&stripe_buf, row_off, next_row_off, w, &mut stats);

            // Laplacian: 3-row window. Skip the topmost row of the
            // image (no `prev_row` available) and the bottom row of
            // the stripe (will be picked up by the lookahead row of
            // *this* stripe — see the prev_row index below).
            if y_start + y_local >= 1 && y_start + y_local + 1 < h {
                // `prev_row` is one row above the current row in the
                // image. Inside the stripe buffer that's `y_local - 1`
                // when `y_local >= 1`; when `y_local == 0` and we have
                // an `y_start >= 1`, we'd need the row from the prior
                // stripe — skip those rows for simplicity (cost: drop
                // ~1 row per stripe, which is <1.5% of sampled pixels).
                if y_local >= 1 {
                    let prev_off = (y_local - 1) * row_bytes;
                    let cur_off = row_off;
                    let nxt_off = (y_local + 1) * row_bytes;
                    let prev_row = &stripe_buf[prev_off..prev_off + row_bytes];
                    let cur_row = &stripe_buf[cur_off..cur_off + row_bytes];
                    let nxt_row = &stripe_buf[nxt_off..nxt_off + row_bytes];
                    accumulate_laplacian_dispatch(prev_row, cur_row, nxt_row, w, &mut stats);
                }
            }

            // (Palette counting was here — moved to `palette` tier.)
            // Grayscale walk over the row: per-pixel max channel gap
            // < 4 ⇒ pixel is effectively neutral. The tight scalar
            // loop autovectorizes well and the row data is hot in L1
            // (just walked by the SIMD stats pass).
            let row = &stripe_buf[row_off..row_off + row_bytes];
            grayscale_pixels += count_grayscale_pixels(row);
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
    // Grayscale fraction: drives ColorMode::Grayscale-style decisions
    // in zenjpeg, indexed-gray paths in png/avif/jxl. Counted across
    // every sampled row in the stripe walk above.
    #[cfg(feature = "experimental")]
    {
        out.grayscale_score = if sampled_pixels > 0 {
            (grayscale_pixels as f64 / sampled_pixels as f64) as f32
        } else {
            0.0
        };
    }
    // Suppress unused warning in default builds.
    let _ = grayscale_pixels;
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
fn accumulate_row_dispatch(
    rgb: &[u8],
    row_off: usize,
    next_row_off: Option<usize>,
    width: usize,
    stats: &mut PixelStats,
) {
    let row_stats = incant!(accumulate_row_simd(rgb, row_off, next_row_off, width));
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
fn accumulate_row_simd(
    token: Token,
    rgb: &[u8],
    row_off: usize,
    next_row_off: Option<usize>,
    width: usize,
) -> PixelStats {
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

    // ---- f32x8 stats pass: 8 pixels (24 bytes) per chunk ----
    let kr_v = f32x8::splat(token, KR);
    let kg_v = f32x8::splat(token, KG);
    let kb_v = f32x8::splat(token, KB);
    let inv_255_v = f32x8::splat(token, 1.0 / 255.0);
    let half_v = f32x8::splat(token, 0.5);

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
        // Chroma: cb = (b − l) / 255; cr = (r − l) / 255
        let cb = (b - l) * inv_255_v;
        let cr = (r - l) * inv_255_v;
        // Hasler M3: rg = r − g; yb = 0.5·(r + g) − b
        let rg = r - g;
        let yb = (r + g).mul_add(half_v, -b);

        luma_sum_v += l;
        luma_sq_v = l.mul_add(l, luma_sq_v);
        cb_sum_v += cb;
        cb_sq_v = cb.mul_add(cb, cb_sq_v);
        cr_sum_v += cr;
        cr_sq_v = cr.mul_add(cr, cr_sq_v);
        rg_sum_v += rg;
        rg_sq_v = rg.mul_add(rg, rg_sq_v);
        yb_sum_v += yb;
        yb_sq_v = yb.mul_add(yb, yb_sq_v);

        iters_since_flush += 1;
        if iters_since_flush >= FLUSH {
            luma_sum += luma_sum_v.reduce_add() as f64;
            luma_sq_sum += luma_sq_v.reduce_add() as f64;
            cb_sum += cb_sum_v.reduce_add() as f64;
            cb_sq_sum += cb_sq_v.reduce_add() as f64;
            cr_sum += cr_sum_v.reduce_add() as f64;
            cr_sq_sum += cr_sq_v.reduce_add() as f64;
            rg_sum += rg_sum_v.reduce_add() as f64;
            rg_sq_sum += rg_sq_v.reduce_add() as f64;
            yb_sum += yb_sum_v.reduce_add() as f64;
            yb_sq_sum += yb_sq_v.reduce_add() as f64;
            luma_sum_v = f32x8::zero(token);
            luma_sq_v = f32x8::zero(token);
            cb_sum_v = f32x8::zero(token);
            cb_sq_v = f32x8::zero(token);
            cr_sum_v = f32x8::zero(token);
            cr_sq_v = f32x8::zero(token);
            rg_sum_v = f32x8::zero(token);
            rg_sq_v = f32x8::zero(token);
            yb_sum_v = f32x8::zero(token);
            yb_sq_v = f32x8::zero(token);
            iters_since_flush = 0;
        }
    }
    // Final flush of SIMD partials.
    luma_sum += luma_sum_v.reduce_add() as f64;
    luma_sq_sum += luma_sq_v.reduce_add() as f64;
    cb_sum += cb_sum_v.reduce_add() as f64;
    cb_sq_sum += cb_sq_v.reduce_add() as f64;
    cr_sum += cr_sum_v.reduce_add() as f64;
    cr_sq_sum += cr_sq_v.reduce_add() as f64;
    rg_sum += rg_sum_v.reduce_add() as f64;
    rg_sq_sum += rg_sq_v.reduce_add() as f64;
    yb_sum += yb_sum_v.reduce_add() as f64;
    yb_sq_sum += yb_sq_v.reduce_add() as f64;

    // Scalar tail for ≤7 leftover pixels.
    for px in remainder.chunks_exact(3) {
        let r = px[0] as f32;
        let g = px[1] as f32;
        let b = px[2] as f32;
        let l = KR * r + KG * g + KB * b;
        luma_sum += l as f64;
        luma_sq_sum += (l * l) as f64;
        let cb = (b - l) * (1.0 / 255.0);
        let cr = (r - l) * (1.0 / 255.0);
        cb_sum += cb as f64;
        cb_sq_sum += (cb * cb) as f64;
        cr_sum += cr as f64;
        cr_sq_sum += (cr * cr) as f64;
        let rg = r - g;
        let yb = 0.5 * (r + g) - b;
        rg_sum += rg as f64;
        rg_sq_sum += (rg * rg) as f64;
        yb_sum += yb as f64;
        yb_sq_sum += (yb * yb) as f64;
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
            for i in 0..8 {
                let cr_ = c[i * 3] as f32;
                let cg_ = c[i * 3 + 1] as f32;
                let cb_ = c[i * 3 + 2] as f32;
                let l = KR * cr_ + KG * cg_ + KB * cb_;
                let rr_ = r_chunk[i * 3] as f32;
                let rg_ = r_chunk[i * 3 + 1] as f32;
                let rb_ = r_chunk[i * 3 + 2] as f32;
                let lr = KR * rr_ + KG * rg_ + KB * rb_;
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
                    let ld = KR * d_chunk[i * 3] as f32
                        + KG * d_chunk[i * 3 + 1] as f32
                        + KB * d_chunk[i * 3 + 2] as f32;
                    grad_sq += (ld - l) * (ld - l);
                }
                if grad_sq > EDGE_THRESH_SQ {
                    edge_count += 1;
                }
            }
        }

        // Scalar tail for the remaining 0..7 edge pixels.
        let processed = (width - 1) / 8 * 8;
        for x in processed..width - 1 {
            let off = row_off + x * 3;
            let l = KR * rgb[off] as f32 + KG * rgb[off + 1] as f32 + KB * rgb[off + 2] as f32;
            let roff = row_off + (x + 1) * 3;
            let lr = KR * rgb[roff] as f32 + KG * rgb[roff + 1] as f32 + KB * rgb[roff + 2] as f32;
            let gx = lr - l;
            let mut grad_sq = gx * gx;
            if has_next {
                let doff = next_row_off.unwrap() + x * 3;
                let ld =
                    KR * rgb[doff] as f32 + KG * rgb[doff + 1] as f32 + KB * rgb[doff + 2] as f32;
                grad_sq += (ld - l) * (ld - l);
            }
            if grad_sq > EDGE_THRESH_SQ {
                edge_count += 1;
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
fn accumulate_laplacian_simd(
    token: Token,
    prev_row: &[u8],
    cur_row: &[u8],
    next_row: &[u8],
    width: usize,
) -> (f64, f64, u64) {
    if width < 3 {
        return (0.0, 0.0, 0);
    }
    let mut prev_l = vec![0.0f32; width];
    let mut cur_l = vec![0.0f32; width];
    let mut next_l = vec![0.0f32; width];
    for x in 0..width {
        let off = x * 3;
        prev_l[x] = KR * prev_row[off] as f32
            + KG * prev_row[off + 1] as f32
            + KB * prev_row[off + 2] as f32;
        cur_l[x] =
            KR * cur_row[off] as f32 + KG * cur_row[off + 1] as f32 + KB * cur_row[off + 2] as f32;
        next_l[x] = KR * next_row[off] as f32
            + KG * next_row[off + 1] as f32
            + KB * next_row[off + 2] as f32;
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
    stats: &mut PixelStats,
) {
    let (s, sq, n) = incant!(accumulate_laplacian_simd(
        prev_row, cur_row, next_row, width
    ));
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
#[archmage::autoversion(v4x, v4, v3, neon, scalar)]
fn count_grayscale_pixels(row: &[u8]) -> u64 {
    const GRAYSCALE_THRESHOLD: u8 = 4;
    let mut count: u64 = 0;
    for px in row.chunks_exact(3) {
        let r = px[0];
        let g = px[1];
        let b = px[2];
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        if mx - mn <= GRAYSCALE_THRESHOLD {
            count += 1;
        }
    }
    count
}
