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

use super::AnalyzerOutput;
use super::row_stream::RowStream;

#[inline(always)]
fn rgb_to_ycbcr_q(r: u8, g: u8, b: u8) -> (i32, i32, i32) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let y = 3 * r + 5 * g + b;
    let mut cb = 3 * b - 2 * g - r;
    let mut cr = 6 * r - 5 * g - b;
    if cr < 0 {
        cb /= 2;
        cr /= 2;
    }
    cb += 3 * 255;
    cr += 6 * 255;
    (y, cb, cr)
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
    let cb_diff = (cb_d.pow(2) as u32).saturating_mul(boost) / 128;
    let cr_diff = (cr_d.pow(2) as u32).saturating_mul(boost) / 128;
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
/// per-channel peaks. Three-row sliding window pulled on demand from
/// `stream`.
fn image_sharpness_breakdown(
    stream: &mut RowStream<'_>,
    width: usize,
    height: usize,
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
    // Three rolling rows. After processing (y0, y0+1, y0+2) we
    // advance y0 by 2 and rotate the buffers.
    let mut row0 = vec![0u8; row_bytes];
    let mut row1 = vec![0u8; row_bytes];
    let mut row2 = vec![0u8; row_bytes];

    stream.fetch_into(0, &mut row0);
    stream.fetch_into(1, &mut row1);
    stream.fetch_into(2, &mut row2);

    let mut sumh: (usize, usize) = (0, 0);
    let mut sumv: (usize, usize) = (0, 0);
    let mut max_sumh: (u32, u32) = (0, 0);
    let mut max_sumv: (u32, u32) = (0, 0);
    let mut max_diff: (u32, u32) = (0, 0);

    let fragment_max_height = if height > 128 {
        (height + 5) / 6
    } else {
        ((height + 3) / 4).max(16)
    };
    let mut fragment_height = 0usize;

    let mut y0: usize = 0;
    loop {
        // (a0, b0, c0) walk by 2 along row0; a1 = row1[x]; a2 = row2[x].
        let span = (width - 2) / 2;
        let mut c0 = pixel_at(&row0, 0);
        for i in 0..span {
            let x = i * 2;
            let a0 = c0;
            let b0 = pixel_at(&row0, x + 1);
            c0 = pixel_at(&row0, x + 2);
            let a1 = pixel_at(&row1, x);
            let a2 = pixel_at(&row2, x);

            let h = gradient_diff_ycbcr(a0, b0, c0);
            let v = gradient_diff_ycbcr(a0, a1, a2);

            if v.0 > max_diff.0 {
                max_diff.0 = v.0;
            }
            if v.1 > max_diff.1 {
                max_diff.1 = v.1;
            }
            if h.0 > max_diff.0 {
                max_diff.0 = h.0;
            }
            if h.1 > max_diff.1 {
                max_diff.1 = h.1;
            }

            sumh.0 += h.0 as usize;
            sumh.1 += h.1 as usize;
            sumv.0 += v.0 as usize;
            sumv.1 += v.1 as usize;
        }

        fragment_height += 1;
        if fragment_height >= fragment_max_height {
            let denom = fragment_height * width;
            max_sumh.0 = max_sumh.0.max((sumh.0 / denom) as u32);
            max_sumh.1 = max_sumh.1.max((sumh.1 / denom) as u32);
            max_sumv.0 = max_sumv.0.max((sumv.0 / denom) as u32);
            max_sumv.1 = max_sumv.1.max((sumv.1 / denom) as u32);
            sumh = (0, 0);
            sumv = (0, 0);
            fragment_height = 0;
        }

        // Advance: y0 ← y0 + 2; rotate (row2 → row0), pull two new rows.
        y0 += 2;
        let need_y1 = y0 + 1;
        let need_y2 = y0 + 2;
        if need_y2 >= height {
            break;
        }
        core::mem::swap(&mut row0, &mut row2);
        stream.fetch_into(need_y1 as u32, &mut row1);
        stream.fetch_into(need_y2 as u32, &mut row2);
    }
    if fragment_height > 16 {
        let denom = fragment_height * width;
        max_sumh.0 = max_sumh.0.max((sumh.0 / denom) as u32);
        max_sumh.1 = max_sumh.1.max((sumh.1 / denom) as u32);
        max_sumv.0 = max_sumv.0.max((sumv.0 / denom) as u32);
        max_sumv.1 = max_sumv.1.max((sumv.1 / denom) as u32);
    }

    let max_diff_max = (6 * 256 * 2u32).pow(2);
    let peak_scale = max_diff_max / 100;
    let peak_div = peak_scale.max(1);
    ChromaSharpnessBreakdown {
        cb: ChannelSharpness {
            horiz: max_sumh.0,
            vert: max_sumv.0,
            peak: max_diff.0 / peak_div,
        },
        cr: ChannelSharpness {
            horiz: max_sumh.1,
            vert: max_sumv.1,
            peak: max_diff.1 / peak_div,
        },
    }
}

/// Populate `cb_*_sharpness` and `cr_*_sharpness` on `out`.
pub fn populate_tier2(out: &mut AnalyzerOutput, stream: &mut RowStream<'_>) {
    let w = stream.width() as usize;
    let h = stream.height() as usize;
    let bd = image_sharpness_breakdown(stream, w, h);

    const NORM: f32 = 1e5;
    out.cb_horiz_sharpness = bd.cb.horiz as f32 / NORM;
    out.cb_vert_sharpness = bd.cb.vert as f32 / NORM;
    out.cb_peak_sharpness = bd.cb.peak as f32;
    out.cr_horiz_sharpness = bd.cr.horiz as f32 / NORM;
    out.cr_vert_sharpness = bd.cr.vert as f32 / NORM;
    out.cr_peak_sharpness = bd.cr.peak as f32;
}
