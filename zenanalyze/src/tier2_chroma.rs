//! Tier 2: per-channel per-axis chroma sharpness.
//!
//! Forked from `evalchroma 1.0.3` (`image_sharpness` in `lib.rs` 111–320)
//! so zenjpeg doesn't need to drag the `evalchroma` crate (and the
//! `imgref` re-export) into its dependency tree just for one O(WH)
//! function. Math is preserved verbatim; the data layout was changed
//! to walk a 3-row sliding window pulled on demand from a
//! [`RowStream`] instead of `ImgRef<RGB8>`.
//!
//! Output is normalized into `AnalyzerOutput` using the same scales
//! `coefficient::analysis::evalchroma_ext::populate_tier23` uses
//! (horiz/vert ÷ 1e5, peak already on 0..100). Don't change those
//! scales — every fitted decision tree was trained on this exact
//! normalization.

use super::feature::RawAnalysis;
use super::row_stream::RowStream;
use archmage::{incant, magetypes};

/// Quantized integer YCbCr approximation, originally from
/// evalchroma 1.0.3.
///
/// **This is not strict BT.601 / BT.709** — the coefficients are
/// integer scales (`Y = 3R + 5G + B`, ratios ~ BT.601 ÷ 0.114, plus
/// `Cb = 3B − 2G − R`, `Cr = 6R − 5G − B`) chosen for cheap
/// branch-free arithmetic. Magnitudes are roughly 9× BT.601 8-bit
/// scale, which the per-axis normalization (`÷ 1e5` for horiz/vert,
/// `÷ peak_div` for peak) absorbs.
///
/// **Symmetry repaired (2026-04-27).** Previous versions halved Cb/Cr
/// when `cr < 0`, biasing the metric toward warm-edge content
/// (positive R−G dominance). That asymmetry has been removed —
/// gradients are now treated identically regardless of chroma sign.
/// Any downstream consumer that calibrated against the old
/// asymmetric values needs re-validation; see the CHANGELOG entry
/// under [Unreleased] for the breaking change notice.
#[inline(always)]
fn rgb_to_ycbcr_q(r: u8, g: u8, b: u8) -> (i32, i32, i32) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let y = 3 * r + 5 * g + b;
    let cb = 3 * b - 2 * g - r;
    let cr = 6 * r - 5 * g - b;
    // Offset into a non-negative range so downstream gradient/peak
    // computations don't have to special-case sign.
    (y, cb + 3 * 255, cr + 6 * 255)
}

#[inline(always)]
fn pixel_at(row: &[u8], x: usize) -> (i32, i32, i32) {
    let off = x * 3;
    rgb_to_ycbcr_q(row[off], row[off + 1], row[off + 2])
}

#[inline(always)]
fn gradient_diff_ycbcr(
    a0: (i32, i32, i32),
    a1: (i32, i32, i32),
    a2: (i32, i32, i32),
) -> (u32, u32) {
    let cb_d = (a0.1 + a2.1) - 2 * a1.1;
    let cr_d = (a0.2 + a2.2) - 2 * a1.2;
    let y_max: i32 = 9 * 255;
    let contrast_boost = y_max - (y_max / 2 - a1.0).abs();
    let edge = (a0.0 - a2.0).abs();
    let no_edge_boost = y_max * 2 - edge;
    let boost = ((no_edge_boost + contrast_boost).max(0) as u32) / 32;
    // u64 intermediates: post-symmetry-repair `cr_d` reaches ±6120
    // (Cr = 6R − 5G − B + 6×255 ⇒ second-difference range is ±6120).
    // `cr_d² × boost_max = 37 454 400 × 215 = 8 052 696 000 > u32::MAX`,
    // so the previous `(cr_d.pow(2) as u32).saturating_mul(boost)` was
    // silently clamping the saturated-chroma case to u32::MAX / 128 =
    // 33 554 431 instead of the real 62 911 687 — a 1.87× undercount
    // of the per-group `max_diff_cr`. The SIMD path uses f32
    // throughout and was unaffected; this matched it. Cb's range is
    // narrower (max product ≈ 2.01 B, fits in u32), but using u64
    // for both keeps the two paths symmetric and rules out future
    // symmetry-repair surprises.
    let cb_diff = ((cb_d.pow(2) as u64 * boost as u64) / 128) as u32;
    let cr_diff = ((cr_d.pow(2) as u64 * boost as u64) / 128) as u32;
    (cb_diff, cr_diff)
}

#[derive(Default, Clone, Copy)]
struct ChannelSharpness {
    horiz: u32,
    vert: u32,
    peak: u32,
}

#[derive(Default, Clone, Copy)]
struct ChromaSharpnessBreakdown {
    cb: ChannelSharpness,
    cr: ChannelSharpness,
}

/// Walk three rows at a time (`a0` / `a1` / `a2`) and accumulate
/// horizontal + vertical 2nd-difference Cb/Cr energies plus
/// per-channel peaks.
///
/// `pixel_budget` controls how aggressively to subsample triplets.
/// At default (500_000) the analyzer caps work at ~1 ms regardless
/// of image size. Pass `usize::MAX` for a full-image scan.
fn image_sharpness_breakdown(
    stream: &mut RowStream<'_>,
    width: usize,
    height: usize,
    pixel_budget: usize,
) -> ChromaSharpnessBreakdown {
    if width < 3 || height < 3 {
        let dud = ChannelSharpness {
            horiz: 0,
            vert: 0,
            peak: 100,
        };
        return ChromaSharpnessBreakdown { cb: dud, cr: dud };
    }
    let row_bytes = width * 3;
    // Triplet stride: how many triplet-iterations to skip between
    // sampled triplets, chosen to stay within `pixel_budget`. Each
    // sampled triplet covers 2 rows of pixel area (windows overlap by
    // one row between adjacent triplets), so total budgeted
    // sampled-triplets = budget / (2 * width). Mirrors the same
    // pattern Tier 1 uses (see tier1::compute_stripe_step).
    let total_triplets = (height - 2).div_ceil(2);
    let pixels_per_triplet = 2 * width;
    let target_triplets = (pixel_budget / pixels_per_triplet.max(1))
        .max(1)
        .min(total_triplets.max(1));
    let triplet_stride = (total_triplets / target_triplets).max(1);

    // Three rolling rows. With stride = 1 we rotate (row2 → row0)
    // between iters; with stride > 1 we always pull three fresh rows.
    let mut row0 = vec![0u8; row_bytes];
    let mut row1 = vec![0u8; row_bytes];
    let mut row2 = vec![0u8; row_bytes];

    stream.fetch_into(0, &mut row0);
    stream.fetch_into(1, &mut row1);
    stream.fetch_into(2, &mut row2);

    // u64 (not usize) so 32-bit builds don't overflow:
    // gradient_diff_ycbcr can return values up to ~437K per pixel.
    let mut sumh: (u64, u64) = (0, 0);
    let mut sumv: (u64, u64) = (0, 0);
    let mut total_triplets: u64 = 0;
    let mut max_diff: (u32, u32) = (0, 0);
    // Per-group peaks: aggregated for an approximate p99 over
    // sampled groups so a single hot pixel doesn't skew the metric
    // at higher resolutions.
    let mut peak_samples_cb: Vec<u32> = Vec::with_capacity(4096);
    let mut peak_samples_cr: Vec<u32> = Vec::with_capacity(4096);

    // Previous versions used a per-fragment "max-of-fragment-averages"
    // normalization. That design coupled the output scale to image
    // height (fragment_max_height = h/6 at h>128, scaled by
    // triplet_stride) and produced a 0.5+ scale-CV between
    // resolutions of the same content. Replaced with a single global
    // mean + p99 peak — both are scale-invariant.
    let mut y0: usize = 0;
    loop {
        let group = process_row_group_dispatch(&row0, &row1, &row2, width);
        sumh.0 += group.sumh_cb;
        sumh.1 += group.sumh_cr;
        sumv.0 += group.sumv_cb;
        sumv.1 += group.sumv_cr;
        if group.max_diff_cb > max_diff.0 {
            max_diff.0 = group.max_diff_cb;
        }
        if group.max_diff_cr > max_diff.1 {
            max_diff.1 = group.max_diff_cr;
        }
        if peak_samples_cb.len() < 4096 {
            peak_samples_cb.push(group.max_diff_cb);
            peak_samples_cr.push(group.max_diff_cr);
        }
        total_triplets += (width.saturating_sub(2)) as u64;

        // Advance by 2 * triplet_stride rows to the next sampled
        // triplet. With stride = 1 we can rotate (row2 → row0) and
        // only fetch two new rows; with stride > 1 the next triplet
        // is non-adjacent so all three rows are fresh.
        y0 += 2 * triplet_stride;
        let need_y2 = y0 + 2;
        if need_y2 >= height {
            break;
        }
        if triplet_stride == 1 {
            core::mem::swap(&mut row0, &mut row2);
            stream.fetch_into((y0 + 1) as u32, &mut row1);
            stream.fetch_into(need_y2 as u32, &mut row2);
        } else {
            stream.fetch_into(y0 as u32, &mut row0);
            stream.fetch_into((y0 + 1) as u32, &mut row1);
            stream.fetch_into(need_y2 as u32, &mut row2);
        }
    }
    // Global mean over all sampled triplets — scale-invariant (same
    // per-pixel rate at any image size). The denominator is the count
    // of column triplets sampled across the row groups.
    let denom = total_triplets.max(1);
    let mean_h_cb = (sumh.0 / denom) as u32;
    let mean_h_cr = (sumh.1 / denom) as u32;
    let mean_v_cb = (sumv.0 / denom) as u32;
    let mean_v_cr = (sumv.1 / denom) as u32;

    // p99 peak: 99th percentile of per-group max-pixel-diffs. With
    // a single hot pixel, the absolute max grows linearly with image
    // size; p99 is more stable. Falls back to the running global max
    // when the sample is too small (≤ 4 entries) for percentile to
    // mean anything.
    //
    // **Accuracy note:** this is genuinely the 99th percentile only
    // for `N ≥ 100` samples. At smaller N the index `floor(0.99 * (N
    // − 1))` lands at a coarser percentile:
    //
    //   N = 5  → 80th percentile
    //   N = 10 → 90th
    //   N = 50 → 98th
    //   N ≥ 100 → ~99th
    //
    // Tier 2 hits N ≥ 100 row-groups on any image taller than ~600
    // px, which is the dominant case. Smaller images get a coarser-
    // percentile peak that still serves the "robust against single
    // hot pixel" goal — accepted trade-off, not a bug.
    fn percentile_99(samples: &mut [u32], fallback: u32) -> u32 {
        if samples.len() <= 4 {
            return fallback;
        }
        samples.sort_unstable();
        samples[((samples.len() as f32 - 1.0) * 0.99) as usize]
    }
    let peak_cb = percentile_99(&mut peak_samples_cb, max_diff.0);
    let peak_cr = percentile_99(&mut peak_samples_cr, max_diff.1);

    // Peak normalization: `(6 * 256 * 2)² / 100` reference scale,
    // carried over from the pre-symmetry-repair code where Cr's
    // effective range was narrower.
    //
    // **Output range:** for natural photographic content the peak
    // field typically lands < 100 (the original calibration target).
    // After the symmetry repair (CHANGELOG note above) the SIMD
    // path's true `max_diff_cr` is ≈ 62 911 687, which divides to
    // a peak of ≈ 666 — saturated synthetic content (e.g. alternating
    // pure-red / pure-green columns) reaches that ceiling. Code that
    // calibrates against `cb_peak_sharpness` / `cr_peak_sharpness`
    // must NOT clamp to [0, 100]; production photos do, but
    // synthetic / chart / extreme-chroma inputs don't.
    // Renormalising the peak fields onto a stable [0, 100] scale is a
    // follow-up calibration task tracked in `infer_bucket`'s
    // `CALIBRATION-PENDING` note.
    let max_diff_max = (6 * 256 * 2u32).pow(2);
    let peak_div = (max_diff_max / 100).max(1);

    ChromaSharpnessBreakdown {
        cb: ChannelSharpness {
            horiz: mean_h_cb,
            vert: mean_v_cb,
            peak: peak_cb / peak_div,
        },
        cr: ChannelSharpness {
            horiz: mean_h_cr,
            vert: mean_v_cr,
            peak: peak_cr / peak_div,
        },
    }
}

/// Aggregated stats from one 3-row group's worth of triplets. Returned
/// from [`process_row_group_dispatch`] so the outer loop in
/// [`image_sharpness_breakdown`] can fold them into fragment counters.
struct RowGroupStats {
    /// `(cb, cr)` sums of horizontal-2nd-diff energy × boost / 128.
    sumh_cb: u64,
    sumh_cr: u64,
    /// `(cb, cr)` sums of vertical-2nd-diff energy × boost / 128.
    sumv_cb: u64,
    sumv_cr: u64,
    /// `(cb, cr)` running max of any single-pixel diff in this group.
    max_diff_cb: u32,
    max_diff_cr: u32,
}

/// Runtime-dispatched entry to the SIMD'd row-group kernel.
fn process_row_group_dispatch(
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    width: usize,
) -> RowGroupStats {
    incant!(process_row_group_simd(row0, row1, row2, width))
}

/// Process one 3-row group: compute every horizontal triplet
/// `(row0[x], row0[x+1], row0[x+2])` for `x ∈ 0..=width-3`, every
/// vertical triplet `(row0[x], row1[x], row2[x])`, accumulate
/// boost-weighted Cb/Cr 2nd-difference energies and per-channel peaks.
///
/// f32x8 lanes process 8 column triplets per iter; tail handled by an
/// inline scalar fallback. Generated for v4 / v3 / NEON / WASM128 /
/// scalar tiers via `#[magetypes]`; runtime-selected through
/// `process_row_group_dispatch`.
#[magetypes(define(f32x8), v4, v3, neon, wasm128, scalar)]
fn process_row_group_simd(
    token: Token,
    row0: &[u8],
    row1: &[u8],
    row2: &[u8],
    width: usize,
) -> RowGroupStats {
    // Constants for the boost factor.
    let y_max_v = f32x8::splat(token, 9.0 * 255.0);
    let y_max_x2_v = f32x8::splat(token, 2.0 * 9.0 * 255.0);
    let y_half_v = f32x8::splat(token, 9.0 * 255.0 / 2.0);
    let inv_32_v = f32x8::splat(token, 1.0 / 32.0);
    let inv_128_v = f32x8::splat(token, 1.0 / 128.0);
    let zero_v = f32x8::zero(token);

    let mut sum_cb_h_v = zero_v;
    let mut sum_cr_h_v = zero_v;
    let mut sum_cb_v_v = zero_v;
    let mut sum_cr_v_v = zero_v;
    let mut max_cb_v = zero_v;
    let mut max_cr_v = zero_v;

    // Periodic-flush bookkeeping: reduce SIMD-lane f32 partials into
    // f64 outer accumulators every FLUSH_EVERY iters to bound
    // cumulative precision loss. Each iter contributes up to 8 × 62 M
    // ≈ 500 M to the f32 accumulator; flushing every 32 iters keeps
    // the partial sum below ~16 G — well within f32's gradual-rounding
    // range (mantissa 24 bits exact up to 16 M, ~24 bits relative
    // beyond), so cumulative ULP error stays under 0.001 % of any
    // single-iter contribution.
    const FLUSH_EVERY: usize = 32;
    let mut iters_since_flush = 0usize;
    let mut sumh_cb_acc: u64 = 0;
    let mut sumh_cr_acc: u64 = 0;
    let mut sumv_cb_acc: u64 = 0;
    let mut sumv_cr_acc: u64 = 0;
    let mut max_cb_scalar: u32 = 0;
    let mut max_cr_scalar: u32 = 0;

    if width < 3 {
        return RowGroupStats {
            sumh_cb: 0,
            sumh_cr: 0,
            sumv_cb: 0,
            sumv_cr: 0,
            max_diff_cb: 0,
            max_diff_cr: 0,
        };
    }
    let span = width - 2; // number of valid triplet starts
    let chunks = span / 8;

    // Per-row YCbCr scratch. Deinterleave each row ONCE, then take
    // overlapping slices (cols 0..8, 1..9, 2..10) for the a/b/c
    // triplet positions on row0. Single-pass over row0 gives 10
    // pixels' worth of work; old per-pixel approach did 24 (3 of each
    // for a/b/c).
    //
    // Sized for the maximum chunk: chunks_x_8 + 2 trailing pixels for
    // the c position. Reused across iters; no per-iter allocation.
    let row0_len = chunks * 8 + 2;
    let row12_len = chunks * 8;
    let mut y0 = vec![0.0f32; row0_len];
    let mut cb0 = vec![0.0f32; row0_len];
    let mut cr0 = vec![0.0f32; row0_len];
    let mut y1 = vec![0.0f32; row12_len];
    let mut cb1 = vec![0.0f32; row12_len];
    let mut cr1 = vec![0.0f32; row12_len];
    let mut y2 = vec![0.0f32; row12_len];
    let mut cb2 = vec![0.0f32; row12_len];
    let mut cr2 = vec![0.0f32; row12_len];

    // Bulk-deinterleave row0 (chunks*8 + 2 pixels). Simple per-pixel
    // scalar — LLVM autovectorizes well under the per-tier
    // target_feature context that `#[magetypes]` sets up, and the
    // straight loop avoids the iterator-chain overhead that
    // chunks_exact + enumerate + try_into adds in release builds
    // (measured 3× regression at 4 MP).
    for i in 0..row0_len {
        let off = i * 3;
        let r = row0[off] as f32;
        let g = row0[off + 1] as f32;
        let b = row0[off + 2] as f32;
        y0[i] = 3.0 * r + 5.0 * g + b;
        cb0[i] = 3.0 * b - 2.0 * g - r + 3.0 * 255.0;
        cr0[i] = 6.0 * r - 5.0 * g - b + 6.0 * 255.0;
    }
    for i in 0..row12_len {
        let off = i * 3;
        let r = row1[off] as f32;
        let g = row1[off + 1] as f32;
        let b = row1[off + 2] as f32;
        y1[i] = 3.0 * r + 5.0 * g + b;
        cb1[i] = 3.0 * b - 2.0 * g - r + 3.0 * 255.0;
        cr1[i] = 6.0 * r - 5.0 * g - b + 6.0 * 255.0;
        let r = row2[off] as f32;
        let g = row2[off + 1] as f32;
        let b = row2[off + 2] as f32;
        y2[i] = 3.0 * r + 5.0 * g + b;
        cb2[i] = 3.0 * b - 2.0 * g - r + 3.0 * 255.0;
        cr2[i] = 6.0 * r - 5.0 * g - b + 6.0 * 255.0;
    }

    for ci in 0..chunks {
        let s = ci * 8;
        // Overlapping f32x8 loads from the deinterleaved scratch.
        // a/b/c on row0: cols [s..s+8], [s+1..s+9], [s+2..s+10].
        // a1/a2 on row1/row2: cols [s..s+8].
        let a_y_v = f32x8::load(token, (&y0[s..s + 8]).try_into().unwrap());
        let b_y_v = f32x8::load(token, (&y0[s + 1..s + 9]).try_into().unwrap());
        let c_y_v = f32x8::load(token, (&y0[s + 2..s + 10]).try_into().unwrap());
        let a_cb_v = f32x8::load(token, (&cb0[s..s + 8]).try_into().unwrap());
        let b_cb_v = f32x8::load(token, (&cb0[s + 1..s + 9]).try_into().unwrap());
        let c_cb_v = f32x8::load(token, (&cb0[s + 2..s + 10]).try_into().unwrap());
        let a_cr_v = f32x8::load(token, (&cr0[s..s + 8]).try_into().unwrap());
        let b_cr_v = f32x8::load(token, (&cr0[s + 1..s + 9]).try_into().unwrap());
        let c_cr_v = f32x8::load(token, (&cr0[s + 2..s + 10]).try_into().unwrap());
        let a1_y_v = f32x8::load(token, (&y1[s..s + 8]).try_into().unwrap());
        let a1_cb_v = f32x8::load(token, (&cb1[s..s + 8]).try_into().unwrap());
        let a1_cr_v = f32x8::load(token, (&cr1[s..s + 8]).try_into().unwrap());
        let a2_y_v = f32x8::load(token, (&y2[s..s + 8]).try_into().unwrap());
        let a2_cb_v = f32x8::load(token, (&cb2[s..s + 8]).try_into().unwrap());
        let a2_cr_v = f32x8::load(token, (&cr2[s..s + 8]).try_into().unwrap());

        // Horizontal: (a, b, c) — center b. 2nd diff = a + c − 2b.
        let cb_h = a_cb_v + c_cb_v - b_cb_v - b_cb_v;
        let cr_h = a_cr_v + c_cr_v - b_cr_v - b_cr_v;
        let cb_h_sq = cb_h * cb_h;
        let cr_h_sq = cr_h * cr_h;
        let edge_h = (a_y_v - c_y_v).abs();
        let contrast_h = y_max_v - (y_half_v - b_y_v).abs();
        let no_edge_h = y_max_x2_v - edge_h;
        let boost_h = (no_edge_h + contrast_h).max(zero_v) * inv_32_v;
        let cb_diff_h = cb_h_sq * boost_h * inv_128_v;
        let cr_diff_h = cr_h_sq * boost_h * inv_128_v;

        // Vertical: (a, a1, a2) — center a1. 2nd diff = a + a2 − 2 a1.
        let cb_v_d = a_cb_v + a2_cb_v - a1_cb_v - a1_cb_v;
        let cr_v_d = a_cr_v + a2_cr_v - a1_cr_v - a1_cr_v;
        let cb_v_sq = cb_v_d * cb_v_d;
        let cr_v_sq = cr_v_d * cr_v_d;
        let edge_v = (a_y_v - a2_y_v).abs();
        let contrast_v = y_max_v - (y_half_v - a1_y_v).abs();
        let no_edge_v = y_max_x2_v - edge_v;
        let boost_v = (no_edge_v + contrast_v).max(zero_v) * inv_32_v;
        let cb_diff_v = cb_v_sq * boost_v * inv_128_v;
        let cr_diff_v = cr_v_sq * boost_v * inv_128_v;

        sum_cb_h_v += cb_diff_h;
        sum_cr_h_v += cr_diff_h;
        sum_cb_v_v += cb_diff_v;
        sum_cr_v_v += cr_diff_v;
        max_cb_v = max_cb_v.max(cb_diff_h).max(cb_diff_v);
        max_cr_v = max_cr_v.max(cr_diff_h).max(cr_diff_v);

        iters_since_flush += 1;
        if iters_since_flush >= FLUSH_EVERY {
            sumh_cb_acc += sum_cb_h_v.reduce_add() as u64;
            sumh_cr_acc += sum_cr_h_v.reduce_add() as u64;
            sumv_cb_acc += sum_cb_v_v.reduce_add() as u64;
            sumv_cr_acc += sum_cr_v_v.reduce_add() as u64;
            max_cb_scalar = max_cb_scalar.max(max_cb_v.reduce_max() as u32);
            max_cr_scalar = max_cr_scalar.max(max_cr_v.reduce_max() as u32);
            sum_cb_h_v = zero_v;
            sum_cr_h_v = zero_v;
            sum_cb_v_v = zero_v;
            sum_cr_v_v = zero_v;
            max_cb_v = zero_v;
            max_cr_v = zero_v;
            iters_since_flush = 0;
        }
    }

    // Final flush of SIMD partial sums into outer scalar accumulators.
    sumh_cb_acc += sum_cb_h_v.reduce_add() as u64;
    sumh_cr_acc += sum_cr_h_v.reduce_add() as u64;
    sumv_cb_acc += sum_cb_v_v.reduce_add() as u64;
    sumv_cr_acc += sum_cr_v_v.reduce_add() as u64;
    max_cb_scalar = max_cb_scalar.max(max_cb_v.reduce_max() as u32);
    max_cr_scalar = max_cr_scalar.max(max_cr_v.reduce_max() as u32);

    // Scalar tail for any column triplets past the last SIMD chunk.
    for x in (chunks * 8)..span {
        let off_a = x * 3;
        let off_b = (x + 1) * 3;
        let off_c = (x + 2) * 3;
        let a0 = rgb_to_ycbcr_q(row0[off_a], row0[off_a + 1], row0[off_a + 2]);
        let b0 = rgb_to_ycbcr_q(row0[off_b], row0[off_b + 1], row0[off_b + 2]);
        let c0 = rgb_to_ycbcr_q(row0[off_c], row0[off_c + 1], row0[off_c + 2]);
        let a1 = rgb_to_ycbcr_q(row1[off_a], row1[off_a + 1], row1[off_a + 2]);
        let a2 = rgb_to_ycbcr_q(row2[off_a], row2[off_a + 1], row2[off_a + 2]);
        let h = gradient_diff_ycbcr(a0, b0, c0);
        let v = gradient_diff_ycbcr(a0, a1, a2);
        sumh_cb_acc += h.0 as u64;
        sumh_cr_acc += h.1 as u64;
        sumv_cb_acc += v.0 as u64;
        sumv_cr_acc += v.1 as u64;
        max_cb_scalar = max_cb_scalar.max(h.0).max(v.0);
        max_cr_scalar = max_cr_scalar.max(h.1).max(v.1);
    }

    RowGroupStats {
        sumh_cb: sumh_cb_acc,
        sumh_cr: sumh_cr_acc,
        sumv_cb: sumv_cb_acc,
        sumv_cr: sumv_cr_acc,
        max_diff_cb: max_cb_scalar,
        max_diff_cr: max_cr_scalar,
    }
}

/// Populate `cb_*_sharpness` and `cr_*_sharpness` on `out`.
///
/// `pixel_budget` controls Tier 2 stride sampling — same budget unit
/// as Tier 1. Default (500_000) caps work at ~1 ms regardless of
/// image size; `usize::MAX` runs the full triplet sweep.
pub fn populate_tier2(out: &mut RawAnalysis, stream: &mut RowStream<'_>, pixel_budget: usize) {
    let w = stream.width() as usize;
    let h = stream.height() as usize;
    let bd = image_sharpness_breakdown(stream, w, h, pixel_budget);

    const NORM: f32 = 1e5;
    out.cb_horiz_sharpness = bd.cb.horiz as f32 / NORM;
    out.cb_vert_sharpness = bd.cb.vert as f32 / NORM;
    out.cb_peak_sharpness = bd.cb.peak as f32;
    out.cr_horiz_sharpness = bd.cr.horiz as f32 / NORM;
    out.cr_vert_sharpness = bd.cr.vert as f32 / NORM;
    out.cr_peak_sharpness = bd.cr.peak as f32;
}
