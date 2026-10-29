//! Per-image "color-space loss" research tool for the zenjpeg codec picker.
//!
//! zenjpeg encodes in two color modes: standard **YCbCr (BT.601)** and
//! **XYB** (perceptual, JPEG-XL-style). A per-image picker chooses the mode.
//! This tool measures, for EACH path, how much COLOR the path's *color
//! transform + spatial chroma/B subsampling* loses — independent of DCT
//! quantization. The hypothesis: images whose colors survive one path much
//! better than the other are exactly the ones where that mode wins.
//!
//! For each path we run the COLOR-ONLY lossy round-trip (NO DCT):
//!
//!   1. **YCbCr path**: RGB8 → BT.601 YCbCr (Y=0.299R+0.587G+0.114B) →
//!      4:2:0 chroma subsample (2×2 box average on Cb,Cr) → bilinear
//!      upsample → YCbCr→RGB8, clip to [0,255].
//!   2. **XYB path**: RGB8 → linear (sRGB EOTF) → scaled-XYB (zenjpeg's
//!      exact opsin matrix + cube root + scale_xyb, REUSED from
//!      `zenjpeg::color::xyb`) → B-channel quarter subsample (XYB BQuarter:
//!      X & Y full-res, scaled-B 2×2 box average) → bilinear upsample →
//!      unscale → XYB→linear→sRGB→RGB8, clip to [0,255].
//!
//! The XYB pipeline reuses `srgb_to_scaled_xyb` / `scaled_xyb_to_srgb`
//! verbatim from the encoder so the conversion math and the subsampling
//! *space* (scaled-B, where `scaled_b = (b - y + offset) * scale`) match
//! `convert_strip_to_xyb`'s BQuarter mode exactly (R:2×2 G:2×2 B:1×1).
//!
//! Per-image, per-path metrics: MAE (mean |ΔRGB| over R,G,B,all pixels),
//! max ΔRGB (max per-channel abs diff), and % pixels with max-channel
//! ΔRGB > {2,5,10}. We also bin the high-error pixels by hue/saturation to
//! characterise WHICH part of the color space each path loses.
//!
//! Predictive signal per image: `delta = ycbcr_loss − xyb_loss`
//! (positive ⇒ XYB preserves color better, by MAE).
//!
//! ## 444/Full pure-color-conversion mode
//!
//! The 4:2:0 vs BQuarter comparison above conflates two opposing forces:
//! XYB drops ONE chroma plane (scaled-B) while YCbCr drops TWO (Cb,Cr), so
//! XYB always "loses less" — but that is a SUBSAMPLING artifact. To isolate
//! the COLOR CONVERSION itself, we also run both paths at full chroma/B
//! resolution (YCbCr 4:4:4, XYB Full):
//!
//!   - `ycbcr_444_roundtrip`: RGB8 → BT.601 → RGB8 (no subsample). Near-
//!     lossless; the only loss is the f32→u8 requant.
//!   - `xyb_full_roundtrip`: RGB8 → scaled-XYB (all 3 channels full res) →
//!     RGB8. Isolates the opsin → cbrt → unscale → XYB→linear → sRGB
//!     transfer → 8-bit requant loss, which sheds gamut detail (notably
//!     YELLOW and some RED hues) that BT.601 keeps.
//!
//! With subsampling removed, the conversion alone FAVORS YCbCr. Per image
//! we emit `ycbcr444_mae`, `xyb444_mae`, their max errors, plus:
//!
//!   - `xyb_conv_loss_frac` = fraction of pixels where (err_xyb444 > 8) AND
//!     (err_ycbcr444 <= 3): colors ONLY XYB's conversion sheds — the
//!     "favor-YCbCr" picker signal.
//!   - `ycbcr_conv_loss_frac` = fraction where (err_ycbcr444 > 8) AND
//!     (err_xyb444 <= 3).
//!   - a hue histogram (R,Y,G,C,B,M, sat>0.15) of the XYB-uniquely-lost
//!     pixels, to confirm the yellow/red concentration.
//!
//! Together the two forces — subsampling (favors XYB, the `delta_mae`
//! column) and color conversion (favors YCbCr, `xyb_conv_loss_frac`) —
//! give the picker BOTH signals it needs.
//!
//! This harness lives in `zenjpeg/dev/` (a research tool, not part of the
//! public examples dir or library API). It is gated behind `__test-utils`
//! (which makes `zenjpeg::color` public).
//!
//! Usage:
//!   cargo run --release -p zenjpeg --features __test-utils \
//!     --example color_loss -- \
//!     --corpus /mnt/v/zen/xyb-combinatorial-pilot-2026-06-01/sources \
//!     --out-tsv /mnt/v/zen/color-loss-research/color_loss_metrics.tsv

#![cfg(feature = "__test-utils")]
#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use zenjpeg::color::xyb::{scaled_xyb_to_srgb, srgb_to_scaled_xyb};
use zenjpeg::color::ycbcr::{rgb_to_ycbcr_f32, ycbcr_to_rgb_f32};

/// A single-channel f32 plane with explicit dimensions.
struct Plane {
    w: usize,
    h: usize,
    data: Vec<f32>,
}

impl Plane {
    fn new(w: usize, h: usize) -> Self {
        Plane {
            w,
            h,
            data: vec![0.0; w * h],
        }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.w + x]
    }
}

/// 2×2 box-average downsample. Output is ceil(w/2) × ceil(h/2). Edge cells
/// average only the in-bounds samples (clamped), matching how a 2×2 box
/// filter behaves on odd dimensions.
fn downsample_2x2(p: &Plane) -> Plane {
    let dw = p.w.div_ceil(2);
    let dh = p.h.div_ceil(2);
    let mut out = Plane::new(dw, dh);
    for dy in 0..dh {
        for dx in 0..dw {
            let x0 = dx * 2;
            let y0 = dy * 2;
            let x1 = (x0 + 1).min(p.w - 1);
            let y1 = (y0 + 1).min(p.h - 1);
            let s = p.at(x0, y0) + p.at(x1, y0) + p.at(x0, y1) + p.at(x1, y1);
            out.data[dy * dw + dx] = s * 0.25;
        }
    }
    out
}

/// Bilinear upsample of a 2×-downsampled plane back to (out_w, out_h).
///
/// Uses the standard JPEG "centered" chroma sample geometry: downsampled
/// sample k covers full-res pixels (2k, 2k+1), so its center sits at
/// full-res coordinate 2k + 0.5. The full-res pixel at coordinate i maps to
/// source coordinate src = (i - 0.5) / 2 = i/2 - 0.25, then bilinearly
/// interpolated with edge clamping.
fn upsample_bilinear(p: &Plane, out_w: usize, out_h: usize) -> Plane {
    let mut out = Plane::new(out_w, out_h);
    let sw = p.w as isize;
    let sh = p.h as isize;
    for i in 0..out_h {
        // map full-res row i to source row coordinate
        let sy = (i as f32) * 0.5 - 0.25;
        let fy = sy.floor();
        let wy = sy - fy;
        let y0 = (fy as isize).clamp(0, sh - 1) as usize;
        let y1 = ((fy as isize) + 1).clamp(0, sh - 1) as usize;
        for j in 0..out_w {
            let sx = (j as f32) * 0.5 - 0.25;
            let fx = sx.floor();
            let wx = sx - fx;
            let x0 = (fx as isize).clamp(0, sw - 1) as usize;
            let x1 = ((fx as isize) + 1).clamp(0, sw - 1) as usize;
            let top = p.at(x0, y0) * (1.0 - wx) + p.at(x1, y0) * wx;
            let bot = p.at(x0, y1) * (1.0 - wx) + p.at(x1, y1) * wx;
            out.data[i * out_w + j] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

#[inline]
fn clip_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// YCbCr 4:2:0 color-only round-trip. Returns the reconstructed RGB8 buffer.
fn ycbcr_420_roundtrip(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut y = Plane::new(w, h);
    let mut cb = Plane::new(w, h);
    let mut cr = Plane::new(w, h);
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let (yy, cbb, crr) = rgb_to_ycbcr_f32(r, g, b);
        y.data[i] = yy;
        cb.data[i] = cbb;
        cr.data[i] = crr;
    }
    // 4:2:0 = 2×2 subsample of chroma, luma stays full res.
    let cb_d = downsample_2x2(&cb);
    let cr_d = downsample_2x2(&cr);
    let cb_u = upsample_bilinear(&cb_d, w, h);
    let cr_u = upsample_bilinear(&cr_d, w, h);

    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let (r, g, b) = ycbcr_to_rgb_f32(y.data[i], cb_u.data[i], cr_u.data[i]);
        out[i * 3] = clip_u8(r);
        out[i * 3 + 1] = clip_u8(g);
        out[i * 3 + 2] = clip_u8(b);
    }
    out
}

/// XYB BQuarter color-only round-trip. X & Y full-res, scaled-B 2×2
/// subsampled (the channel the encoder downsamples in BQuarter mode).
/// Returns the reconstructed RGB8 buffer.
fn xyb_bquarter_roundtrip(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut sx = Plane::new(w, h);
    let mut sy = Plane::new(w, h);
    let mut sb = Plane::new(w, h);
    for i in 0..n {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        // scaled-XYB: exact encoder conversion (opsin + cbrt + scale_xyb).
        let (x, yv, bv) = srgb_to_scaled_xyb(r, g, b);
        sx.data[i] = x;
        sy.data[i] = yv;
        sb.data[i] = bv;
    }
    // BQuarter: only the (scaled) B channel is 2×2 subsampled.
    let sb_d = downsample_2x2(&sb);
    let sb_u = upsample_bilinear(&sb_d, w, h);

    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let (r, g, b) = scaled_xyb_to_srgb(sx.data[i], sy.data[i], sb_u.data[i]);
        out[i * 3] = r;
        out[i * 3 + 1] = g;
        out[i * 3 + 2] = b;
    }
    out
}

/// YCbCr 4:4:4 (Full) color-only round-trip — NO chroma subsampling.
/// Isolates the pure color-conversion loss of the BT.601 matrix + 8-bit
/// JPEG-sample requant: RGB8 → BT.601 YCbCr (full res) → **8-bit sample
/// requant on Y,Cb,Cr** → YCbCr→RGB8, clip.
///
/// `rgb_to_ycbcr_f32` already outputs Y,Cb,Cr in the 0-255 JPEG sample
/// range (Cb,Cr centered at 128). The encoder stores `round(Y)` etc. as
/// 8-bit DCT samples; at q100 the round trip is round-to-nearest. We model
/// that with `round()` on each channel so this path goes through the same
/// 8-bit sample grid as `xyb_full_roundtrip` — making the conversion-loss
/// comparison apples-to-apples (both at 444/Full, both 8-bit samples).
/// BT.601 is well-conditioned, so this is the YCbCr "conversion floor".
fn ycbcr_444_roundtrip(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let (yy, cbb, crr) = rgb_to_ycbcr_f32(r, g, b);
        // 8-bit JPEG-sample requant (Y,Cb,Cr stored as integer samples).
        let yq = yy.round();
        let cbq = cbb.round();
        let crq = crr.round();
        let (r2, g2, b2) = ycbcr_to_rgb_f32(yq, cbq, crq);
        out[i * 3] = clip_u8(r2);
        out[i * 3 + 1] = clip_u8(g2);
        out[i * 3 + 2] = clip_u8(b2);
    }
    out
}

/// XYB Full color-only round-trip — NO B subsampling, all 3 XYB channels
/// full res. Isolates the pure color-conversion loss of XYB's
/// opsin → cube-root → scale → **8-bit JPEG-sample requant** → unscale →
/// XYB→linear → sRGB transfer → 8-bit RGB requant.
///
/// The dominant loss is the 8-bit JPEG-sample requant on the scaled-XYB
/// channels. The encoder maps each scaled-XYB value to a JPEG sample via
/// `scaled_xyb * 255.0` (`convert_strip_to_xyb`, strip/convert.rs:642),
/// the sample is stored as an 8-bit DCT coefficient (level-shifted), and
/// the decoder reverses it as `sample / 255.0` before `unscale_xyb`. At
/// q100 the DCT quant step is 1, so the spatial round trip is exactly
/// `round(scaled_xyb * 255) / 255` per channel — the 8-bit sample grid is
/// the conversion floor. XYB's cube-root packs the (large) yellow/red
/// gamut into that grid more coarsely than BT.601 packs Cb/Cr, so XYB
/// sheds those hues here. Returns the reconstructed RGB8 buffer.
///
/// We do NOT clamp the sample to [0,255]: XYB JPEGs use SOF1 / extended
/// range and the encoder does not clamp the scaled f32 at this stage, so
/// the only quantization modeled is round-to-nearest-integer sample.
fn xyb_full_roundtrip(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        // forward: sRGB8 -> scaled XYB (encoder's exact conversion).
        let (x, yv, bv) = srgb_to_scaled_xyb(r, g, b);
        // 8-bit JPEG-sample requant on each scaled-XYB channel: the
        // encoder stores `scaled * 255` as an integer DCT sample; at q100
        // the round trip is round-to-nearest. Reverse with `/ 255`.
        let xq = (x * 255.0).round() / 255.0;
        let yq = (yv * 255.0).round() / 255.0;
        let bq = (bv * 255.0).round() / 255.0;
        // inverse: scaled XYB -> sRGB8 (encoder's exact inverse).
        let (r2, g2, b2) = scaled_xyb_to_srgb(xq, yq, bq);
        out[i * 3] = r2;
        out[i * 3 + 1] = g2;
        out[i * 3 + 2] = b2;
    }
    out
}

/// Per-pixel max-channel |ΔRGB| between two RGB8 buffers.
#[inline]
fn pixel_err(orig: &[u8], recon: &[u8], i: usize) -> u32 {
    let dr = (orig[i * 3] as i32 - recon[i * 3] as i32).unsigned_abs();
    let dg = (orig[i * 3 + 1] as i32 - recon[i * 3 + 1] as i32).unsigned_abs();
    let db = (orig[i * 3 + 2] as i32 - recon[i * 3 + 2] as i32).unsigned_abs();
    dr.max(dg).max(db)
}

/// Pure-color-conversion (444/Full) metrics comparing the YCbCr-4:4:4 and
/// XYB-Full round trips per pixel. Both paths run at full chroma/B
/// resolution, so any difference is COLOR CONVERSION, not subsampling.
struct ConvMetrics {
    ycbcr444_mae: f64,
    xyb444_mae: f64,
    ycbcr444_maxerr: u32,
    xyb444_maxerr: u32,
    /// fraction of pixels where (err_xyb444 > 8) AND (err_ycbcr444 <= 3):
    /// colors ONLY XYB's conversion sheds (the "favor-YCbCr" signal).
    xyb_conv_loss_frac: f64,
    /// fraction where (err_ycbcr444 > 8) AND (err_xyb444 <= 3).
    ycbcr_conv_loss_frac: f64,
    /// hue histogram (R,Y,G,C,B,M) of the XYB-uniquely-lost pixels,
    /// counting only those with saturation > 0.15.
    xyb_lost_hue: [u64; 6],
    /// count of XYB-uniquely-lost pixels with sat > 0.15 (hist denominator).
    xyb_lost_sat_pixels: u64,
}

fn compute_conv_metrics(
    orig: &[u8],
    yc444: &[u8],
    xyb444: &[u8],
    w: usize,
    h: usize,
) -> ConvMetrics {
    let n = w * h;
    let mut yc_sum: u64 = 0;
    let mut xy_sum: u64 = 0;
    let mut yc_max: u32 = 0;
    let mut xy_max: u32 = 0;
    let mut xyb_only_lost: u64 = 0;
    let mut ycbcr_only_lost: u64 = 0;
    let mut xyb_lost_hue = [0u64; 6];
    let mut xyb_lost_sat_pixels: u64 = 0;
    for i in 0..n {
        let dr_y = (orig[i * 3] as i32 - yc444[i * 3] as i32).unsigned_abs();
        let dg_y = (orig[i * 3 + 1] as i32 - yc444[i * 3 + 1] as i32).unsigned_abs();
        let db_y = (orig[i * 3 + 2] as i32 - yc444[i * 3 + 2] as i32).unsigned_abs();
        yc_sum += (dr_y + dg_y + db_y) as u64;
        let e_yc = dr_y.max(dg_y).max(db_y);
        if e_yc > yc_max {
            yc_max = e_yc;
        }
        let e_xy = pixel_err(orig, xyb444, i);
        xy_sum += {
            let dr = (orig[i * 3] as i32 - xyb444[i * 3] as i32).unsigned_abs();
            let dg = (orig[i * 3 + 1] as i32 - xyb444[i * 3 + 1] as i32).unsigned_abs();
            let db = (orig[i * 3 + 2] as i32 - xyb444[i * 3 + 2] as i32).unsigned_abs();
            (dr + dg + db) as u64
        };
        if e_xy > xy_max {
            xy_max = e_xy;
        }
        // colors ONLY XYB sheds (favor-YCbCr).
        if e_xy > 8 && e_yc <= 3 {
            xyb_only_lost += 1;
            let (sector, s) = hue_sat(orig[i * 3], orig[i * 3 + 1], orig[i * 3 + 2]);
            if s > 0.15 {
                xyb_lost_hue[sector] += 1;
                xyb_lost_sat_pixels += 1;
            }
        }
        // colors ONLY YCbCr sheds (favor-XYB on conversion alone).
        if e_yc > 8 && e_xy <= 3 {
            ycbcr_only_lost += 1;
        }
    }
    let np = n as f64;
    ConvMetrics {
        ycbcr444_mae: yc_sum as f64 / (np * 3.0),
        xyb444_mae: xy_sum as f64 / (np * 3.0),
        ycbcr444_maxerr: yc_max,
        xyb444_maxerr: xy_max,
        xyb_conv_loss_frac: xyb_only_lost as f64 / np,
        ycbcr_conv_loss_frac: ycbcr_only_lost as f64 / np,
        xyb_lost_hue,
        xyb_lost_sat_pixels,
    }
}

/// Per-image, per-path color-loss metrics.
struct PathMetrics {
    mae: f64,      // mean |ΔRGB| over all channels & pixels
    max: u32,      // max per-channel abs diff
    pct_gt2: f64,  // % pixels whose max-channel |Δ| > 2
    pct_gt5: f64,  // % pixels whose max-channel |Δ| > 5
    pct_gt10: f64, // % pixels whose max-channel |Δ| > 10
    // hue/sat characterisation of HIGH-error pixels (max-channel |Δ| > 5):
    // counts of high-error pixels falling in coarse hue × saturation bins.
    hue_hist: [u64; 6], // R, Y, G, C, B, M (60° hue sectors)
    sat_lo: u64,        // high-err pixels with low saturation (<0.25)
    sat_mid: u64,       // 0.25..0.6
    sat_hi: u64,        // >=0.6
    n_high: u64,        // total high-error pixels (max-channel |Δ| > 5)
}

/// Map an sRGB8 pixel to a coarse hue sector index (0..6) and a saturation
/// in [0,1] (HSV S). Used only to characterise where errors concentrate.
fn hue_sat(r: u8, g: u8, b: u8) -> (usize, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let d = max - min;
    let s = if max <= 0.0 { 0.0 } else { d / max };
    if d <= 1e-6 {
        return (0, 0.0); // achromatic; sector irrelevant (s≈0)
    }
    let mut hue = if max == rf {
        ((gf - bf) / d) % 6.0
    } else if max == gf {
        (bf - rf) / d + 2.0
    } else {
        (rf - gf) / d + 4.0
    };
    if hue < 0.0 {
        hue += 6.0;
    }
    // hue is in [0,6): 0=R,1=Y,2=G,3=C,4=B,5=M sectors.
    let sector = (hue.floor() as usize) % 6;
    (sector, s)
}

fn compute_metrics(orig: &[u8], recon: &[u8], w: usize, h: usize) -> PathMetrics {
    let n = w * h;
    let mut sum_abs: u64 = 0;
    let mut max: u32 = 0;
    let (mut g2, mut g5, mut g10) = (0u64, 0u64, 0u64);
    let mut hue_hist = [0u64; 6];
    let (mut sat_lo, mut sat_mid, mut sat_hi) = (0u64, 0u64, 0u64);
    let mut n_high = 0u64;
    for i in 0..n {
        let dr = (orig[i * 3] as i32 - recon[i * 3] as i32).unsigned_abs();
        let dg = (orig[i * 3 + 1] as i32 - recon[i * 3 + 1] as i32).unsigned_abs();
        let db = (orig[i * 3 + 2] as i32 - recon[i * 3 + 2] as i32).unsigned_abs();
        sum_abs += (dr + dg + db) as u64;
        let pmax = dr.max(dg).max(db);
        if pmax > max {
            max = pmax;
        }
        if pmax > 2 {
            g2 += 1;
        }
        if pmax > 5 {
            g5 += 1;
            n_high += 1;
            // characterise the ORIGINAL color at this high-error pixel.
            let (sector, s) = hue_sat(orig[i * 3], orig[i * 3 + 1], orig[i * 3 + 2]);
            hue_hist[sector] += 1;
            if s < 0.25 {
                sat_lo += 1;
            } else if s < 0.6 {
                sat_mid += 1;
            } else {
                sat_hi += 1;
            }
        }
        if pmax > 10 {
            g10 += 1;
        }
    }
    let np = n as f64;
    PathMetrics {
        mae: sum_abs as f64 / (np * 3.0),
        max,
        pct_gt2: 100.0 * g2 as f64 / np,
        pct_gt5: 100.0 * g5 as f64 / np,
        pct_gt10: 100.0 * g10 as f64 / np,
        hue_hist,
        sat_lo,
        sat_mid,
        sat_hi,
        n_high,
    }
}

fn load_rgb8(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let img = zenjpeg_bench_utils::load_png(path).ok()?;
    let (buf, w, h) = img.into_contiguous_buf();
    let bytes: Vec<u8> = buf.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Some((bytes, w, h))
}

fn main() {
    let mut corpus = PathBuf::from("/mnt/v/zen/xyb-combinatorial-pilot-2026-06-01/sources");
    let mut out_tsv = PathBuf::from("/mnt/v/zen/color-loss-research/color_loss_metrics.tsv");
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                corpus = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out-tsv" => {
                out_tsv = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                i += 1;
            }
        }
    }

    if let Some(parent) = out_tsv.parent() {
        fs::create_dir_all(parent).expect("create out dir");
    }

    let mut entries: Vec<PathBuf> = fs::read_dir(&corpus)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", corpus.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();

    let mut f = fs::File::create(&out_tsv).expect("create tsv");
    writeln!(
        f,
        "image\tclass\twidth\theight\t\
ycbcr_mae\tycbcr_max\tycbcr_pct_gt2\tycbcr_pct_gt5\tycbcr_pct_gt10\t\
xyb_mae\txyb_max\txyb_pct_gt2\txyb_pct_gt5\txyb_pct_gt10\t\
delta_mae\tdelta_pct_gt5\t\
ycbcr_high_R\tycbcr_high_Y\tycbcr_high_G\tycbcr_high_C\tycbcr_high_B\tycbcr_high_M\t\
ycbcr_high_satlo\tycbcr_high_satmid\tycbcr_high_sathi\t\
xyb_high_R\txyb_high_Y\txyb_high_G\txyb_high_C\txyb_high_B\txyb_high_M\t\
xyb_high_satlo\txyb_high_satmid\txyb_high_sathi\t\
ycbcr444_mae\txyb444_mae\tycbcr444_maxerr\txyb444_maxerr\t\
xyb_conv_loss_frac\tycbcr_conv_loss_frac\t\
xyblost_R\txyblost_Y\txyblost_G\txyblost_C\txyblost_B\txyblost_M\txyblost_satpx"
    )
    .unwrap();

    println!(
        "{:<40} {:>8} {:>8} {:>8}  {:>10} {:>10}",
        "image", "yc_mae", "xyb_mae", "delta", "xyb444_mae", "xybconvlos"
    );
    println!("{}", "-".repeat(96));

    for path in &entries {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let class = if name.starts_with("gb_") {
            "screen"
        } else {
            "photo"
        };
        let Some((rgb, w, h)) = load_rgb8(path) else {
            eprintln!("skip (load failed): {name}");
            continue;
        };

        let yc = ycbcr_420_roundtrip(&rgb, w, h);
        let xy = xyb_bquarter_roundtrip(&rgb, w, h);
        let mc = compute_metrics(&rgb, &yc, w, h);
        let mx = compute_metrics(&rgb, &xy, w, h);
        let delta_mae = mc.mae - mx.mae;
        let delta_pct5 = mc.pct_gt5 - mx.pct_gt5;

        // 444/Full pure-color-conversion round trips (NO subsampling).
        let yc444 = ycbcr_444_roundtrip(&rgb, w, h);
        let xy444 = xyb_full_roundtrip(&rgb, w, h);
        let cm = compute_conv_metrics(&rgb, &yc444, &xy444, w, h);

        writeln!(
            f,
            "{name}\t{class}\t{w}\t{h}\t\
{:.4}\t{}\t{:.3}\t{:.3}\t{:.3}\t\
{:.4}\t{}\t{:.3}\t{:.3}\t{:.3}\t\
{:.4}\t{:.3}\t\
{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
{:.4}\t{:.4}\t{}\t{}\t\
{:.6}\t{:.6}\t\
{}\t{}\t{}\t{}\t{}\t{}\t{}",
            mc.mae,
            mc.max,
            mc.pct_gt2,
            mc.pct_gt5,
            mc.pct_gt10,
            mx.mae,
            mx.max,
            mx.pct_gt2,
            mx.pct_gt5,
            mx.pct_gt10,
            delta_mae,
            delta_pct5,
            mc.hue_hist[0],
            mc.hue_hist[1],
            mc.hue_hist[2],
            mc.hue_hist[3],
            mc.hue_hist[4],
            mc.hue_hist[5],
            mc.sat_lo,
            mc.sat_mid,
            mc.sat_hi,
            mx.hue_hist[0],
            mx.hue_hist[1],
            mx.hue_hist[2],
            mx.hue_hist[3],
            mx.hue_hist[4],
            mx.hue_hist[5],
            mx.sat_lo,
            mx.sat_mid,
            mx.sat_hi,
            cm.ycbcr444_mae,
            cm.xyb444_mae,
            cm.ycbcr444_maxerr,
            cm.xyb444_maxerr,
            cm.xyb_conv_loss_frac,
            cm.ycbcr_conv_loss_frac,
            cm.xyb_lost_hue[0],
            cm.xyb_lost_hue[1],
            cm.xyb_lost_hue[2],
            cm.xyb_lost_hue[3],
            cm.xyb_lost_hue[4],
            cm.xyb_lost_hue[5],
            cm.xyb_lost_sat_pixels,
        )
        .unwrap();

        let _ = mc.n_high;
        let _ = mx.n_high;
        println!(
            "{:<40} {:>8.4} {:>8.4} {:>+8.4}  {:>10.4} {:>10.5}",
            name, mc.mae, mx.mae, delta_mae, cm.xyb444_mae, cm.xyb_conv_loss_frac
        );
    }

    println!("\nWrote {}", out_tsv.display());
}
