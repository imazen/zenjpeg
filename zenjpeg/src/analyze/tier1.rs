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

use super::AnalyzerOutput;
use super::row_stream::RowStream;

const KR: f32 = 0.299;
const KG: f32 = 0.587;
const KB: f32 = 0.114;
const EDGE_THRESH_SQ: f32 = 400.0; // (|∇L| > 20)²

/// Stripe height. Matches the 8×8 block size used for uniformity so
/// each stripe contributes complete blocks without partial-block
/// artifacts.
const STRIPE_H: usize = 8;

/// Pixel budget for stripe sampling (~1 ms at 4K on a 7950X).
const DEFAULT_PIXEL_BUDGET: usize = 500_000;

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
    }
}

/// Populate Tier 1 fields on `out`. Other fields are left untouched.
pub fn extract_tier1_into(out: &mut AnalyzerOutput, stream: &mut RowStream<'_>) {
    let w = stream.width() as usize;
    let h = stream.height() as usize;
    if w < 2 || h < 2 {
        return;
    }

    let stripe_step = compute_stripe_step(w, h, DEFAULT_PIXEL_BUDGET);
    let row_bytes = w * 3;
    let blocks_x = w / STRIPE_H;
    let total_stripes = h / STRIPE_H;

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
    let mut color_bins = [0u64; 512];

    let mut stripe_idx = 0;
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

        // --- Stats + edges over each row in this stripe ---
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

            // Color bins piggyback on the same scan.
            let row = &stripe_buf[row_off..row_off + row_bytes];
            for px in row.chunks_exact(3) {
                let idx = (((px[0] >> 3) as usize) << 10)
                    | (((px[1] >> 3) as usize) << 5)
                    | ((px[2] >> 3) as usize);
                color_bins[idx >> 6] |= 1u64 << (idx & 63);
            }
            sampled_pixels += w as u64;
            if next_row_off.is_some() {
                sampled_interior += (w - 1) as u64;
            }
        }

        // --- 8×8 block stats: luma uniformity + per-channel flat color ---
        for bx in 0..blocks_x {
            let mut sum: u32 = 0;
            let mut sq_sum: u32 = 0;
            let mut r_min: u8 = 255;
            let mut r_max: u8 = 0;
            let mut g_min: u8 = 255;
            let mut g_max: u8 = 0;
            let mut b_min: u8 = 255;
            let mut b_max: u8 = 0;
            for dy in 0..STRIPE_H {
                if y_start + dy >= h {
                    break;
                }
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
            let var = sq_sum as f32 / n - mean * mean;
            if var < 25.0 {
                uniform_blocks += 1;
            }
            if r_max - r_min <= 4 && g_max - g_min <= 4 && b_max - b_min <= 4 {
                flat_color_blocks += 1;
            }
            total_blocks += 1;
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
    out.distinct_color_bins = color_bins.iter().map(|w| w.count_ones()).sum();
    out.flat_color_block_ratio = if total_blocks > 0 {
        flat_color_blocks as f32 / total_blocks as f32
    } else {
        0.0
    };
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

/// Runtime dispatch wrapper — same five-tier ladder zenjpeg uses elsewhere.
fn accumulate_row_dispatch(
    rgb: &[u8],
    row_off: usize,
    next_row_off: Option<usize>,
    width: usize,
    stats: &mut PixelStats,
) {
    let row_stats = archmage::incant!(
        accumulate_row(rgb, row_off, next_row_off, width),
        [v3, neon, wasm128, scalar]
    );
    stats.merge(&row_stats);
}

/// Single-row accumulator: luma/chroma stats + edge count + per-channel
/// chroma horizontal gradients in one fused pass.
///
/// Local-stats return (no `&mut` chain) keeps the seven f64
/// accumulators in registers and lets LLVM unroll the 8-pixel batch
/// fully. The fixed `&[u8; 24]` chunk view eliminates interior bounds
/// checks.
#[archmage::autoversion]
fn accumulate_row(
    rgb: &[u8],
    row_off: usize,
    next_row_off: Option<usize>,
    width: usize,
) -> PixelStats {
    let mut luma_sum: f64 = 0.0;
    let mut luma_sq_sum: f64 = 0.0;
    let mut cb_sum: f64 = 0.0;
    let mut cb_sq_sum: f64 = 0.0;
    let mut cr_sum: f64 = 0.0;
    let mut cr_sq_sum: f64 = 0.0;
    let mut edge_count: u64 = 0;

    let row = &rgb[row_off..row_off + width * 3];
    let next_row = next_row_off.map(|nr| &rgb[nr..nr + width * 3]);

    // ---- Luma + chroma: 8 pixels (24 bytes) per chunk ----
    let chunks = row.chunks_exact(24);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let c: &[u8; 24] = chunk.try_into().unwrap();
        for i in 0..8 {
            let r = c[i * 3] as f32;
            let g = c[i * 3 + 1] as f32;
            let b = c[i * 3 + 2] as f32;
            let l = KR * r + KG * g + KB * b;
            luma_sum += l as f64;
            luma_sq_sum += (l * l) as f64;
            let cb = (b - l) * (1.0 / 255.0);
            let cr = (r - l) * (1.0 / 255.0);
            cb_sum += cb as f64;
            cb_sq_sum += (cb * cb) as f64;
            cr_sum += cr as f64;
            cr_sq_sum += (cr * cr) as f64;
        }
    }
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
    }
}
