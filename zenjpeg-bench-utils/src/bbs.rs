//! Block Boundary Score (BBS) — perceptual blocking metric for JPEG.
//!
//! BBS measures how much the reconstructed image's cross-seam gradients
//! deviate from the original's, at every 8-pixel grid-aligned JPEG block
//! boundary. SSIM2 and Butteraugli underweight blocking; this metric is
//! direct about what it's measuring.
//!
//! See GitHub issue #91 for the motivation and encoder-side uses.
//!
//! # Algorithm
//!
//! For each 8-pixel grid-aligned seam in the reconstructed image:
//!
//! ```text
//! grad_across_seam(img, p) = img[p + 1] - img[p]
//! BBS = Σ_over_all_seams (grad_across_seam(rec) - grad_across_seam(orig))²
//! ```
//!
//! Seams are the lines *between* block rows/columns, i.e.
//! - **Horizontal seams** (between block rows `B` and `B+1`): pairs of pixels
//!   `(img[8B+7, x], img[8B+8, x])` for all `x` in `[0, width)`.
//! - **Vertical seams** (between block cols `B` and `B+1`): pairs of pixels
//!   `(img[y, 8B+7], img[y, 8B+8])` for all `y` in `[0, height)`.
//!
//! The sum of squared gradient deltas is divided by the number of seam pixels
//! contributing, giving a per-seam-pixel MSE. This keeps values comparable
//! across image sizes.
//!
//! An "interior gradient score" is computed identically but on non-seam
//! gradients (pixels `p` and `p+1` that are *not* straddling a block
//! boundary). The ratio `BBS / interior` is a normalized blocking indicator:
//! values close to 1 mean the seam gradients behave like interior gradients
//! (no visible blocking); values significantly greater than 1 mean the seams
//! carry extra gradient error vs. the interior, i.e. visible blocking.
//!
//! # Color space notes
//!
//! BBS is computed per-channel. Interpretation depends on the input
//! interpretation:
//!
//! - `bbs_rgb8` operates on RGB u8 pixels. It reports Y/Cb/Cr via BT.601
//!   conversion (standard JPEG matrix) and a sum-of-per-channel total.
//!   Blocking in Y is by far the most perceptually important; Cb/Cr
//!   values are typically smaller and dominated by 4:2:0 upsampling.
//! - `bbs_planar_u8` operates directly on planar single-channel data,
//!   leaving color space interpretation to the caller.
//!
//! No gamma correction is applied. This matches the gamma-domain arithmetic
//! that JPEG's decoder outputs, and what a user would see on screen.

use imgref::ImgRef;
use rgb::RGB;

/// BBS block-size constant. JPEG blocks are always 8×8.
pub const BLOCK: usize = 8;

/// Result of a BBS computation across one or more channels.
///
/// All scores are mean-squared gradient differences. Lower is better (0 =
/// reconstructed gradients match original exactly on those seams).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BbsResult {
    /// Per-channel mean-squared seam gradient difference. For RGB inputs this
    /// is `[Y, Cb, Cr]` in BT.601 space. For planar inputs, `[C0, 0, 0]`.
    pub per_channel: [f64; 3],
    /// Number of channels populated in `per_channel` (1 for planar, 3 for RGB).
    pub channel_count: u8,
    /// Sum of `per_channel[0..channel_count]` — a single scalar "total blocking".
    pub total: f64,
    /// Mean-squared *interior* (non-seam) gradient difference, on the same
    /// channel set. Same semantics as `total`.
    pub interior_total: f64,
    /// Horizontal-seam contribution to `total` (between block rows).
    pub horizontal_total: f64,
    /// Vertical-seam contribution to `total` (between block columns).
    pub vertical_total: f64,
    /// Seam-pixel count that contributed to `total`. 0 if no seams existed
    /// (image smaller than 9 pixels in both dimensions).
    pub seam_pixels: u64,
    /// Interior-pixel count that contributed to `interior_total`.
    pub interior_pixels: u64,
}

impl BbsResult {
    /// Ratio of mean seam-gradient error to mean interior-gradient error.
    ///
    /// Returns `None` if there were no interior gradient samples (image too
    /// small or degenerate). A value significantly greater than 1 means the
    /// seams carry more gradient error than the interior — the classic
    /// blocking signal. A value near 1 means seams are indistinguishable
    /// from the interior.
    pub fn interior_ratio(&self) -> Option<f64> {
        if self.interior_total > 0.0 {
            Some(self.total / self.interior_total)
        } else {
            None
        }
    }
}

/// BT.601 RGB→Y coefficients. Matches JPEG's standard color conversion.
const Y_R: f32 = 0.299;
const Y_G: f32 = 0.587;
const Y_B: f32 = 0.114;

#[inline]
fn rgb_to_y(rgb: RGB<u8>) -> f32 {
    Y_R * rgb.r as f32 + Y_G * rgb.g as f32 + Y_B * rgb.b as f32
}

#[inline]
fn rgb_to_cb(rgb: RGB<u8>) -> f32 {
    // BT.601: Cb = 128 + (-0.168736*R - 0.331264*G + 0.5*B)
    128.0 + (-0.168_736 * rgb.r as f32 - 0.331_264 * rgb.g as f32 + 0.5 * rgb.b as f32)
}

#[inline]
fn rgb_to_cr(rgb: RGB<u8>) -> f32 {
    // BT.601: Cr = 128 + (0.5*R - 0.418688*G - 0.081312*B)
    128.0 + (0.5 * rgb.r as f32 - 0.418_688 * rgb.g as f32 - 0.081_312 * rgb.b as f32)
}

/// Compute BBS on an RGB u8 image pair, reporting per-YCbCr-channel results.
///
/// `reconstructed` and `original` must be the same dimensions. They are
/// interpreted as BT.601 primaries (standard sRGB-like) for the YCbCr split;
/// only gamma-domain gradient differences are taken, not perceptual ones.
///
/// # Panics
///
/// Panics if the two images disagree on `width()` or `height()`.
pub fn bbs_rgb8(reconstructed: ImgRef<'_, RGB<u8>>, original: ImgRef<'_, RGB<u8>>) -> BbsResult {
    assert_eq!(
        reconstructed.width(),
        original.width(),
        "BBS: width mismatch"
    );
    assert_eq!(
        reconstructed.height(),
        original.height(),
        "BBS: height mismatch"
    );

    // Compute three channels in one pass. We build f32 scratch rows on the
    // fly rather than materializing three full planes, keeping peak memory
    // to O(width) per channel regardless of image size.
    //
    // Strategy: walk row-by-row. For vertical seams we need a single row of
    // each channel at a time. For horizontal seams we need the "just above"
    // row and the "current" row. Keep a pair of row buffers per channel.
    let w = reconstructed.width();
    let h = reconstructed.height();

    if w < 2 && h < 2 {
        // Nothing to compare.
        return BbsResult::default();
    }

    let mut ch_accum = [ChannelAccum::default(); 3];

    let rec = reconstructed;
    let orig = original;

    // Row buffers: prev (y-1) and cur (y). Two buffers per channel.
    // Storage layout: 3 channels × 2 rows × w floats, interleaved by channel
    // for locality on each channel's inner loop.
    let mut rec_prev = vec![0.0f32; w * 3];
    let mut rec_cur = vec![0.0f32; w * 3];
    let mut orig_prev = vec![0.0f32; w * 3];
    let mut orig_cur = vec![0.0f32; w * 3];

    for y in 0..h {
        // Load row y into *_cur.
        let rec_row = &rec.buf()[y * rec.stride()..y * rec.stride() + w];
        let orig_row = &orig.buf()[y * orig.stride()..y * orig.stride() + w];
        for x in 0..w {
            let rp = rec_row[x];
            let op = orig_row[x];
            rec_cur[x * 3] = rgb_to_y(rp);
            rec_cur[x * 3 + 1] = rgb_to_cb(rp);
            rec_cur[x * 3 + 2] = rgb_to_cr(rp);
            orig_cur[x * 3] = rgb_to_y(op);
            orig_cur[x * 3 + 1] = rgb_to_cb(op);
            orig_cur[x * 3 + 2] = rgb_to_cr(op);
        }

        // Vertical seams & vertical interior: pairs (x, x+1) within this row.
        for x in 0..w - 1 {
            // is this a block boundary? seam between x=7,8 / 15,16 / ...
            // i.e. (x % 8) == 7
            let is_seam = x % BLOCK == BLOCK - 1;
            for c in 0..3 {
                let gr = rec_cur[(x + 1) * 3 + c] - rec_cur[x * 3 + c];
                let go = orig_cur[(x + 1) * 3 + c] - orig_cur[x * 3 + c];
                let d = (gr - go) as f64;
                let d2 = d * d;
                if is_seam {
                    ch_accum[c].vertical_sum += d2;
                    ch_accum[c].vertical_count += 1;
                } else {
                    ch_accum[c].interior_sum += d2;
                    ch_accum[c].interior_count += 1;
                }
            }
        }

        // Horizontal seams & horizontal interior: pairs (y-1, y) for this row
        // vs. prev. Only if y > 0.
        if y > 0 {
            let is_seam = (y - 1) % BLOCK == BLOCK - 1;
            for x in 0..w {
                for c in 0..3 {
                    let gr = rec_cur[x * 3 + c] - rec_prev[x * 3 + c];
                    let go = orig_cur[x * 3 + c] - orig_prev[x * 3 + c];
                    let d = (gr - go) as f64;
                    let d2 = d * d;
                    if is_seam {
                        ch_accum[c].horizontal_sum += d2;
                        ch_accum[c].horizontal_count += 1;
                    } else {
                        ch_accum[c].interior_sum += d2;
                        ch_accum[c].interior_count += 1;
                    }
                }
            }
        }

        // Advance: cur becomes prev.
        core::mem::swap(&mut rec_prev, &mut rec_cur);
        core::mem::swap(&mut orig_prev, &mut orig_cur);
    }

    build_result(&ch_accum, 3)
}

/// Compute BBS on a single-channel planar u8 image.
///
/// Color space is whatever the caller says it is (Y, Cb, Cr, L*, etc.).
/// Only one channel is reported; `per_channel[1]` and `per_channel[2]`
/// are zero.
pub fn bbs_planar_u8(reconstructed: ImgRef<'_, u8>, original: ImgRef<'_, u8>) -> BbsResult {
    assert_eq!(
        reconstructed.width(),
        original.width(),
        "BBS: width mismatch"
    );
    assert_eq!(
        reconstructed.height(),
        original.height(),
        "BBS: height mismatch"
    );

    let w = reconstructed.width();
    let h = reconstructed.height();
    if w < 2 && h < 2 {
        return BbsResult::default();
    }

    let mut accum = ChannelAccum::default();
    let rec = reconstructed;
    let orig = original;

    // Vertical seams + interior (within each row).
    for y in 0..h {
        let rec_row = &rec.buf()[y * rec.stride()..y * rec.stride() + w];
        let orig_row = &orig.buf()[y * orig.stride()..y * orig.stride() + w];
        for x in 0..w.saturating_sub(1) {
            let is_seam = x % BLOCK == BLOCK - 1;
            let gr = rec_row[x + 1] as i32 - rec_row[x] as i32;
            let go = orig_row[x + 1] as i32 - orig_row[x] as i32;
            let d = (gr - go) as f64;
            let d2 = d * d;
            if is_seam {
                accum.vertical_sum += d2;
                accum.vertical_count += 1;
            } else {
                accum.interior_sum += d2;
                accum.interior_count += 1;
            }
        }
    }

    // Horizontal seams + interior (between rows).
    for y in 1..h {
        let is_seam = (y - 1) % BLOCK == BLOCK - 1;
        let prev_row = &rec.buf()[(y - 1) * rec.stride()..(y - 1) * rec.stride() + w];
        let cur_row = &rec.buf()[y * rec.stride()..y * rec.stride() + w];
        let prev_orig = &orig.buf()[(y - 1) * orig.stride()..(y - 1) * orig.stride() + w];
        let cur_orig = &orig.buf()[y * orig.stride()..y * orig.stride() + w];
        for x in 0..w {
            let gr = cur_row[x] as i32 - prev_row[x] as i32;
            let go = cur_orig[x] as i32 - prev_orig[x] as i32;
            let d = (gr - go) as f64;
            let d2 = d * d;
            if is_seam {
                accum.horizontal_sum += d2;
                accum.horizontal_count += 1;
            } else {
                accum.interior_sum += d2;
                accum.interior_count += 1;
            }
        }
    }

    build_result(
        &[accum, ChannelAccum::default(), ChannelAccum::default()],
        1,
    )
}

#[derive(Debug, Clone, Copy, Default)]
struct ChannelAccum {
    horizontal_sum: f64,
    horizontal_count: u64,
    vertical_sum: f64,
    vertical_count: u64,
    interior_sum: f64,
    interior_count: u64,
}

impl ChannelAccum {
    fn seam_mean(&self) -> f64 {
        let n = self.horizontal_count + self.vertical_count;
        if n == 0 {
            0.0
        } else {
            (self.horizontal_sum + self.vertical_sum) / n as f64
        }
    }
    fn interior_mean(&self) -> f64 {
        if self.interior_count == 0 {
            0.0
        } else {
            self.interior_sum / self.interior_count as f64
        }
    }
}

fn build_result(ch: &[ChannelAccum], channel_count: u8) -> BbsResult {
    let mut per_channel = [0.0; 3];
    let mut total = 0.0;
    let mut interior_total = 0.0;
    let mut horizontal_total = 0.0;
    let mut vertical_total = 0.0;
    let mut seam_pixels = 0u64;
    let mut interior_pixels = 0u64;

    for (i, a) in ch.iter().enumerate().take(channel_count as usize) {
        per_channel[i] = a.seam_mean();
        total += a.seam_mean();
        interior_total += a.interior_mean();
        // Horizontal/vertical breakdown aggregates raw sums divided by the
        // corresponding count for that orientation, summed across channels.
        // Report per-seam-pixel MSE so horizontal_total + vertical_total ≈ total
        // when horizontal and vertical seam counts are similar.
        if a.horizontal_count > 0 {
            horizontal_total += a.horizontal_sum / a.horizontal_count as f64;
        }
        if a.vertical_count > 0 {
            vertical_total += a.vertical_sum / a.vertical_count as f64;
        }
        // Counts aggregate once (they're the same across channels in this
        // implementation; we just add channel 0's).
        if i == 0 {
            seam_pixels = a.horizontal_count + a.vertical_count;
            interior_pixels = a.interior_count;
        }
    }

    BbsResult {
        per_channel,
        channel_count,
        total,
        interior_total,
        horizontal_total,
        vertical_total,
        seam_pixels,
        interior_pixels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imgref::Img;

    fn rgb(r: u8, g: u8, b: u8) -> RGB<u8> {
        RGB { r, g, b }
    }

    #[test]
    fn bbs_identical_images_is_zero() {
        // Random-ish but deterministic pixels, 32x32.
        let w = 32;
        let h = 32;
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 7 + y * 13) & 0xff) as u8;
                let g = ((x * 5 + y * 11) & 0xff) as u8;
                let b = ((x * 17 + y * 3) & 0xff) as u8;
                pixels.push(rgb(r, g, b));
            }
        }
        let img = Img::new(&pixels[..], w, h);
        let bbs = bbs_rgb8(img, img);
        assert_eq!(bbs.total, 0.0, "identical images must score 0");
        assert_eq!(bbs.interior_total, 0.0);
        assert_eq!(bbs.horizontal_total, 0.0);
        assert_eq!(bbs.vertical_total, 0.0);
        assert!(
            bbs.seam_pixels > 0,
            "32x32 has seams at x=7,15,23 y=7,15,23"
        );
    }

    #[test]
    fn bbs_planar_identical_is_zero() {
        let w = 16;
        let h = 16;
        let pixels: Vec<u8> = (0..(w * h)).map(|i| (i * 3) as u8).collect();
        let img = Img::new(&pixels[..], w, h);
        let bbs = bbs_planar_u8(img, img);
        assert_eq!(bbs.total, 0.0);
        assert_eq!(bbs.interior_total, 0.0);
        assert_eq!(bbs.channel_count, 1);
    }

    #[test]
    fn bbs_synthetic_seam_detects_blocking() {
        // Original: uniform gradient (no seams).
        // Reconstructed: same gradient but with +30 offset jump at every
        // block boundary column. This is what textbook blocking looks like.
        let w = 16;
        let h = 16;
        let mut orig = Vec::with_capacity(w * h);
        let mut rec = Vec::with_capacity(w * h);
        for _y in 0..h {
            for x in 0..w {
                let base = (x * 4) as u8;
                orig.push(rgb(base, base, base));
                // Reconstructed: +30 added to x >= 8, i.e. one seam at x=7->8.
                let offset = if x >= 8 { 30 } else { 0 };
                let v = base.saturating_add(offset);
                rec.push(rgb(v, v, v));
            }
        }
        let orig_img = Img::new(&orig[..], w, h);
        let rec_img = Img::new(&rec[..], w, h);
        let bbs = bbs_rgb8(rec_img, orig_img);
        assert!(
            bbs.vertical_total > 100.0,
            "synthetic vertical seam must show large vertical_total, got {}",
            bbs.vertical_total
        );
        // Interior gradient is exactly 4 on both images (uniform 4/pixel
        // step), so interior_total should be tiny (zero up to rounding).
        assert!(
            bbs.interior_total < 0.01,
            "interior gradient unchanged, got {}",
            bbs.interior_total
        );
        assert!(
            bbs.interior_ratio().is_none() || bbs.interior_ratio().unwrap() > 100.0,
            "blocking must overwhelm interior, got ratio {:?}",
            bbs.interior_ratio()
        );
    }

    #[test]
    fn bbs_small_image_no_panic() {
        // 1-pixel image has no seams, no interior. Must not panic.
        let one = vec![rgb(0, 0, 0)];
        let img = Img::new(&one[..], 1, 1);
        let bbs = bbs_rgb8(img, img);
        assert_eq!(bbs.total, 0.0);
        assert_eq!(bbs.seam_pixels, 0);
    }

    #[test]
    fn bbs_horizontal_seam_detected() {
        // A horizontal-only seam: rows y<8 are 0, rows y>=8 are 100. Seam
        // between y=7 and y=8.
        let w = 24;
        let h = 24;
        let mut orig = Vec::with_capacity(w * h);
        let mut rec = Vec::with_capacity(w * h);
        for y in 0..h {
            for _x in 0..w {
                orig.push(rgb(50, 50, 50));
                let v = if y >= 8 { 100u8 } else { 50u8 };
                rec.push(rgb(v, v, v));
            }
        }
        let orig_img = Img::new(&orig[..], w, h);
        let rec_img = Img::new(&rec[..], w, h);
        let bbs = bbs_rgb8(rec_img, orig_img);
        assert!(
            bbs.horizontal_total > bbs.vertical_total,
            "horizontal seam must dominate: horiz={}, vert={}",
            bbs.horizontal_total,
            bbs.vertical_total
        );
    }
}
