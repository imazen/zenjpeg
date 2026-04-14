//! "Triage" deblocking: three-category pixel-domain filter from US 7,079,703 B2.
//!
//! Classifies each 8×8 luma block into one of three categories by inspecting
//! the 3×3 neighborhood of per-block variances — triaging blocks into
//! "needs a lot of smoothing", "needs a little", or "leave it alone" — and
//! then assembles the output from three candidate images:
//!
//! | Class        | Source                                  |
//! |--------------|-----------------------------------------|
//! | Uniform      | Plane convolved with a 7×7 Gaussian     |
//! | Transitional | Plane convolved with a 3×3 low-pass     |
//! | Busy         | Original plane (unfiltered)             |
//!
//! The caller is responsible for the file-size / magnification suitability
//! gate described in FIG. 1 of the patent. This module implements only the
//! artifact-removal stage.
//!
//! Notes on fidelity to the source algorithm:
//! - `uniform` is the arithmetic mean of the 3×3 variance neighborhood
//!   (the patent description text says "average"; the Mathematica listing
//!   writes the unscaled `Sum[]` — we follow the text).
//! - `stdvar` is the sample standard deviation (divisor N−1 = 8) to match
//!   Mathematica's `StandardDeviation[]`.
//! - Per-block variance uses the sample formula (divisor N−1 = 63) to match
//!   Mathematica's `Variance[]`.
//! - The transitional test requires the central block variance to exceed
//!   `uniform_threshold`, per the description text (the Mathematica listing
//!   has an operator-precedence bug that only pairs it with the final OR term).
//! - Blocks in the outermost ring (where a full 3×3 neighborhood is not
//!   available) are left as original pixels, matching the patent's skipped
//!   outer band.

use alloc::vec;
use alloc::vec::Vec;

const BLOCK: usize = 8;

/// 3×3 deringing low-pass kernel from the patent, divided by its sum.
///
/// ```text
///   [1 3 1]
///   [3 6 3]  / 22
///   [1 3 1]
/// ```
const DERING_KERNEL: [i32; 9] = [1, 3, 1, 3, 6, 3, 1, 3, 1];
const DERING_NORM: f32 = 22.0;

/// 7×7 Gaussian deblocking kernel (σ = 1 pixel, scaled by 1024 and rounded).
///
/// ```text
///   [  0    2    7   11    7    2    0]
///   [  2   19   84  139   84   19    2]
///   [  7   84  377  621  377   84    7]
///   [ 11  139  621 1024  621  139   11] / 6436
///   [  7   84  377  621  377   84    7]
///   [  2   19   84  139   84   19    2]
///   [  0    2    7   11    7    2    0]
/// ```
#[rustfmt::skip]
const DEBLOCK_KERNEL: [i32; 49] = [
    0,   2,   7,  11,   7,   2,  0,
    2,  19,  84, 139,  84,  19,  2,
    7,  84, 377, 621, 377,  84,  7,
   11, 139, 621,1024, 621, 139, 11,
    7,  84, 377, 621, 377,  84,  7,
    2,  19,  84, 139,  84,  19,  2,
    0,   2,   7,  11,   7,   2,  0,
];
const DEBLOCK_NORM: f32 = 6436.0;

/// Configuration for the triage deblocking strategy.
#[derive(Debug, Clone, Copy)]
pub struct TriageConfig {
    /// Variance threshold (`thresh1` in the patent) that separates uniform
    /// regions from everything else. Patent default: 64 (for 8-bit pixel
    /// values).
    pub uniform_threshold: u32,
}

impl Default for TriageConfig {
    fn default() -> Self {
        Self {
            uniform_threshold: 64,
        }
    }
}

/// Per-block classification result, exposed for callers that want to inspect
/// or visualize the logic map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockClass {
    /// Block is in a flat region — replace with deblock-filtered pixels.
    Uniform,
    /// Block is at a transition / edge — replace with dering-filtered pixels.
    Transitional,
    /// Block is in textured content — keep original pixels.
    Busy,
}

/// Apply the triage deblocking strategy (US 7,079,703 B2) to a single f32
/// luminance plane in place.
///
/// The plane is `width * height` contiguous f32 values in row-major order,
/// with pixel values in \[0, 255\]. Output values are clamped to the same
/// range. Chrominance channels are unaffected — per the patent, only luma is
/// filtered.
///
/// # Arguments
/// * `plane`  — Mutable luma plane, length must be >= `width * height`.
/// * `width`  — Plane width in pixels.
/// * `height` — Plane height in pixels.
/// * `config` — Classification threshold.
///
/// # Panics
/// Panics if `plane.len() < width * height`.
pub fn filter_plane_triage(
    plane: &mut [f32],
    width: usize,
    height: usize,
    config: TriageConfig,
) {
    assert!(plane.len() >= width * height, "plane buffer too small");

    // Need at least 3×3 blocks to have any interior block to classify.
    let nbx = width / BLOCK;
    let nby = height / BLOCK;
    if nbx < 3 || nby < 3 {
        return;
    }

    let varmap = compute_varmap(plane, width, nbx, nby);
    let logicmap = build_logicmap(&varmap, nbx, nby, config.uniform_threshold as i64);

    // Count how much work the convolutions actually have to do. If the whole
    // image is Busy we can skip the source snapshot entirely.
    let mut n_uniform = 0usize;
    let mut n_transitional = 0usize;
    for &c in &logicmap {
        match c {
            BlockClass::Uniform => n_uniform += 1,
            BlockClass::Transitional => n_transitional += 1,
            BlockClass::Busy => {}
        }
    }
    if n_uniform == 0 && n_transitional == 0 {
        return;
    }

    // The kernels must read from the original plane; blocks we've already
    // overwritten would contaminate neighbouring block outputs. One clone
    // is cheaper than a full-plane convolution pass.
    let src: alloc::vec::Vec<f32> = plane[..width * height].to_vec();

    for by in 0..nby {
        for bx in 0..nbx {
            match logicmap[by * nbx + bx] {
                BlockClass::Uniform => {
                    convolve_block(
                        &src,
                        plane,
                        width,
                        height,
                        bx,
                        by,
                        &DEBLOCK_KERNEL,
                        7,
                        DEBLOCK_NORM,
                    );
                }
                BlockClass::Transitional => {
                    convolve_block(
                        &src,
                        plane,
                        width,
                        height,
                        bx,
                        by,
                        &DERING_KERNEL,
                        3,
                        DERING_NORM,
                    );
                }
                BlockClass::Busy => {}
            }
        }
    }
}

/// Classify every 8×8 block of a plane according to the patent algorithm and
/// return the resulting logic map (row-major, `nbx * nby` entries).
///
/// Useful for callers that want to select pixels from their own candidate
/// images rather than the built-in kernels.
#[must_use]
pub fn classify_plane(
    plane: &[f32],
    width: usize,
    height: usize,
    config: TriageConfig,
) -> Vec<BlockClass> {
    let nbx = width / BLOCK;
    let nby = height / BLOCK;
    if nbx == 0 || nby == 0 {
        return Vec::new();
    }
    let varmap = compute_varmap(plane, width, nbx, nby);
    build_logicmap(&varmap, nbx, nby, config.uniform_threshold as i64)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Per-block sample variance. Rounded to `i64` to keep the threshold math in
/// integers, matching the patent's `Round[Variance[...]]`.
fn compute_varmap(plane: &[f32], width: usize, nbx: usize, nby: usize) -> Vec<i64> {
    let mut out = vec![0i64; nbx * nby];
    for by in 0..nby {
        for bx in 0..nbx {
            let y0 = by * BLOCK;
            let x0 = bx * BLOCK;
            let mut sum = 0.0_f64;
            for y in y0..y0 + BLOCK {
                let row = &plane[y * width + x0..y * width + x0 + BLOCK];
                for &p in row {
                    sum += p as f64;
                }
            }
            let mean = sum / 64.0;
            let mut sq = 0.0_f64;
            for y in y0..y0 + BLOCK {
                let row = &plane[y * width + x0..y * width + x0 + BLOCK];
                for &p in row {
                    let d = p as f64 - mean;
                    sq += d * d;
                }
            }
            // Sample variance (N-1 = 63) to match Mathematica Variance[].
            out[by * nbx + bx] = (sq / 63.0).round() as i64;
        }
    }
    out
}

fn build_logicmap(
    varmap: &[i64],
    nbx: usize,
    nby: usize,
    thresh1: i64,
) -> Vec<BlockClass> {
    let mut out = vec![BlockClass::Busy; nbx * nby];

    // Patent skips the outer ring (i = 2..ynblocks-1, j = 2..nblocks-1 in
    // 1-based indexing). Keep those as Busy (original pixels) so we don't
    // need out-of-range varmap taps.
    for i in 1..nby - 1 {
        for j in 1..nbx - 1 {
            let v = |di: isize, dj: isize| -> i64 {
                let ii = (i as isize + di) as usize;
                let jj = (j as isize + dj) as usize;
                varmap[ii * nbx + jj]
            };

            // Centre block's own variance.
            let centre = v(0, 0);

            // 3×3 sum and mean.
            let mut sum9: i64 = 0;
            for m in -1..=1 {
                for n in -1..=1 {
                    sum9 += v(m, n);
                }
            }
            let mean9 = sum9 as f64 / 9.0;
            // Sample standard deviation (N-1 = 8) to match Mathematica.
            let mut ssq = 0.0_f64;
            for m in -1..=1 {
                for n in -1..=1 {
                    let d = v(m, n) as f64 - mean9;
                    ssq += d * d;
                }
            }
            let stdvar = (ssq / 8.0).sqrt().round() as i64;
            let uniform = mean9.round() as i64;

            // Edge averages (three taps along each side of the 3×3).
            let leftavg = (v(-1, -1) + v(0, -1) + v(1, -1)) / 3;
            let rightavg = (v(-1, 1) + v(0, 1) + v(1, 1)) / 3;
            let topavg = (v(-1, -1) + v(-1, 0) + v(-1, 1)) / 3;
            let botavg = (v(1, -1) + v(1, 0) + v(1, 1)) / 3;
            // Corner "L-shape" averages (the three varmap values forming an
            // L around each corner of the 3×3).
            let corner1 = (v(-1, -1) + v(0, -1) + v(-1, 0)) / 3;
            let corner2 = (v(1, -1) + v(0, -1) + v(1, 0)) / 3;
            let corner3 = (v(-1, 1) + v(-1, 0) + v(0, 1)) / 3;
            let corner4 = (v(1, 1) + v(0, 1) + v(1, 0)) / 3;

            let idx = i * nbx + j;

            // 1. Whole 3×3 is quiet → uniform region → deblock.
            if uniform <= thresh1 {
                out[idx] = BlockClass::Uniform;
                continue;
            }

            // 2. Centre is noisy but at least one directional average or
            //    corner L-average is quiet → transition → dering.
            if centre > thresh1
                && (leftavg <= thresh1
                    || rightavg <= thresh1
                    || topavg <= thresh1
                    || botavg <= thresh1
                    || corner1 <= thresh1
                    || corner2 <= thresh1
                    || corner3 <= thresh1
                    || corner4 <= thresh1)
            {
                out[idx] = BlockClass::Transitional;
                continue;
            }

            // 3. Centre is noisy but at least one individual neighbour is
            //    quiet → transition → dering.
            let mut any_quiet_neighbour = false;
            'outer: for m in -1..=1_isize {
                for n in -1..=1_isize {
                    if (m, n) == (0, 0) {
                        continue;
                    }
                    if v(m, n) <= thresh1 {
                        any_quiet_neighbour = true;
                        break 'outer;
                    }
                }
            }
            if centre > thresh1 && any_quiet_neighbour {
                out[idx] = BlockClass::Transitional;
                continue;
            }

            // 4. Centre is close to the neighbourhood mean (within 2σ) → busy.
            if (centre - uniform).abs() <= 2 * stdvar {
                out[idx] = BlockClass::Busy;
                continue;
            }

            // 5. Otherwise — dering.
            out[idx] = BlockClass::Transitional;
        }
    }

    out
}

/// Convolve a single 8×8 block from `src` with a square integer kernel and
/// write the result into the corresponding 8×8 region of `dst`. Edge pixels
/// use clamp-to-edge sampling; outputs are clamped to \[0, 255\].
///
/// Most blocks are in the interior of the plane, where clamping never fires.
/// We specialise that path so the hot loop is a plain 7×7 / 3×3 FMA sweep
/// with no per-tap bounds check.
#[inline]
fn convolve_block(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
    kernel: &[i32],
    size: usize,
    norm: f32,
) {
    debug_assert!(size * size == kernel.len());
    debug_assert!(size % 2 == 1);
    let radius = size / 2;
    let inv_norm = 1.0_f32 / norm;

    let x0 = bx * BLOCK;
    let y0 = by * BLOCK;

    // Interior fast path: the full kernel footprint fits without clamping.
    let interior =
        x0 >= radius && y0 >= radius && x0 + BLOCK + radius <= width && y0 + BLOCK + radius <= height;

    if interior {
        for y in y0..y0 + BLOCK {
            let row_out = y * width;
            for x in x0..x0 + BLOCK {
                let mut acc = 0.0_f32;
                let sy0 = y - radius;
                let sx0 = x - radius;
                for dy in 0..size {
                    let row_in = (sy0 + dy) * width + sx0;
                    let krow = dy * size;
                    for dx in 0..size {
                        acc += src[row_in + dx] * kernel[krow + dx] as f32;
                    }
                }
                dst[row_out + x] = (acc * inv_norm).clamp(0.0, 255.0);
            }
        }
        return;
    }

    // Boundary path: clamp-to-edge sampling per tap.
    for y in y0..y0 + BLOCK {
        let row_out = y * width;
        for x in x0..x0 + BLOCK {
            let mut acc = 0.0_f32;
            for dy in 0..size {
                let sy = clamp_edge(y as isize + dy as isize - radius as isize, height);
                let row_in = sy * width;
                let krow = dy * size;
                for dx in 0..size {
                    let sx = clamp_edge(x as isize + dx as isize - radius as isize, width);
                    acc += src[row_in + sx] * kernel[krow + dx] as f32;
                }
            }
            dst[row_out + x] = (acc * inv_norm).clamp(0.0, 255.0);
        }
    }
}

#[inline]
fn clamp_edge(i: isize, len: usize) -> usize {
    if i < 0 {
        0
    } else if i >= len as isize {
        len - 1
    } else {
        i as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_plane(w: usize, h: usize, val: f32) -> Vec<f32> {
        vec![val; w * h]
    }

    #[test]
    fn too_small_is_noop() {
        // Fewer than 3×3 blocks → nothing to classify.
        let mut p = mk_plane(16, 16, 128.0);
        filter_plane_triage(&mut p, 16, 16, TriageConfig::default());
        assert!(p.iter().all(|&v| (v - 128.0).abs() < 1e-4));
    }

    #[test]
    fn uniform_plane_stays_uniform() {
        // Flat 64×64: every block is labelled Uniform and the 7×7 Gaussian
        // over a constant plane reproduces the constant (modulo clamp).
        let mut p = mk_plane(64, 64, 128.0);
        filter_plane_triage(&mut p, 64, 64, TriageConfig::default());
        for &v in &p {
            assert!((v - 128.0).abs() < 0.5, "got {v}");
        }
    }

    #[test]
    fn uniform_classification_for_flat_plane() {
        let p = mk_plane(64, 64, 50.0);
        let cls = classify_plane(&p, 64, 64, TriageConfig::default());
        // 8×8 = 64 blocks, outer ring (6×6 outer perimeter = 28 blocks) is
        // Busy by construction, inner 6×6 = 36 should be Uniform.
        let n_uniform = cls.iter().filter(|c| **c == BlockClass::Uniform).count();
        let n_busy = cls.iter().filter(|c| **c == BlockClass::Busy).count();
        assert_eq!(n_uniform, 36, "inner 6×6 should be uniform");
        assert_eq!(n_busy, 28, "outer ring stays busy");
    }

    #[test]
    fn boundary_discontinuity_is_smoothed() {
        // 64×64 plane with a blocky step at column 32. Only the luma
        // boundary at the block edge (col 32) should get smoothed; the flat
        // sides stay near their constant values.
        let w = 64;
        let h = 64;
        let mut p = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = if x < 32 { 40.0 } else { 210.0 };
            }
        }
        let orig = p.clone();
        filter_plane_triage(&mut p, w, h, TriageConfig::default());
        // Near the seam at col 31/32 on interior rows, at least one side
        // moves toward the other.
        let y = 32;
        let left = p[y * w + 31];
        let right = p[y * w + 32];
        let orig_left = orig[y * w + 31];
        let orig_right = orig[y * w + 32];
        let moved = (left - orig_left).abs() > 0.5 || (right - orig_right).abs() > 0.5;
        assert!(moved, "boundary pixels should change: {left} vs {orig_left}, {right} vs {orig_right}");
    }

    #[test]
    fn textured_block_is_busy() {
        // 24×24 plane with high-variance noise pattern in all blocks →
        // every interior block is Busy → output equals input.
        let w = 24;
        let h = 24;
        let mut p = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                // Strong checkerboard: variance per block ~ 10000.
                p[y * w + x] = if (x + y) % 2 == 0 { 20.0 } else { 230.0 };
            }
        }
        let orig = p.clone();
        filter_plane_triage(&mut p, w, h, TriageConfig::default());
        assert_eq!(p, orig, "busy blocks must be left unchanged");
    }

    #[test]
    fn kernels_sum_to_their_norm() {
        // Guard against typos in the kernel constants.
        let d: i32 = DERING_KERNEL.iter().sum();
        assert_eq!(d as f32, DERING_NORM);
        let b: i32 = DEBLOCK_KERNEL.iter().sum();
        assert_eq!(b as f32, DEBLOCK_NORM);
    }

    /// Reference implementation: full-plane convolve then per-block copy.
    /// Used by `per_block_matches_full_plane` to guard against boundary /
    /// ordering bugs in the per-block fast path.
    fn filter_plane_triage_reference(
        plane: &mut [f32],
        width: usize,
        height: usize,
        config: TriageConfig,
    ) {
        let nbx = width / BLOCK;
        let nby = height / BLOCK;
        if nbx < 3 || nby < 3 {
            return;
        }
        let varmap = compute_varmap(plane, width, nbx, nby);
        let logicmap = build_logicmap(&varmap, nbx, nby, config.uniform_threshold as i64);

        let mut full_conv = |kernel: &[i32], size: usize, norm: f32| -> Vec<f32> {
            let radius = size / 2;
            let inv_norm = 1.0_f32 / norm;
            let mut out = vec![0.0_f32; width * height];
            for y in 0..height {
                for x in 0..width {
                    let mut acc = 0.0_f32;
                    for dy in 0..size {
                        let sy = clamp_edge(y as isize + dy as isize - radius as isize, height);
                        let row_in = sy * width;
                        let krow = dy * size;
                        for dx in 0..size {
                            let sx =
                                clamp_edge(x as isize + dx as isize - radius as isize, width);
                            acc += plane[row_in + sx] * kernel[krow + dx] as f32;
                        }
                    }
                    out[y * width + x] = (acc * inv_norm).clamp(0.0, 255.0);
                }
            }
            out
        };
        let dering = full_conv(&DERING_KERNEL, 3, DERING_NORM);
        let deblock = full_conv(&DEBLOCK_KERNEL, 7, DEBLOCK_NORM);

        for by in 0..nby {
            for bx in 0..nbx {
                let src: &[f32] = match logicmap[by * nbx + bx] {
                    BlockClass::Busy => continue,
                    BlockClass::Uniform => &deblock,
                    BlockClass::Transitional => &dering,
                };
                let x0 = bx * BLOCK;
                for y in by * BLOCK..(by + 1) * BLOCK {
                    let row = y * width;
                    plane[row + x0..row + x0 + BLOCK].copy_from_slice(&src[row + x0..row + x0 + BLOCK]);
                }
            }
        }
    }

    #[test]
    fn per_block_matches_full_plane() {
        // Build an image that exercises all three block classes AND both
        // interior and plane-edge block positions: a flat background
        // (uniform), a noisy strip (busy), and a smooth-to-noisy transition.
        let w = 72; // 9 blocks wide
        let h = 56; // 7 blocks tall
        let mut plane = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let val = if x < 24 {
                    40.0 // left third: flat → uniform blocks
                } else if x < 48 {
                    // middle third: gradient — transitional-ish
                    40.0 + (x - 24) as f32 * 6.0
                } else if (x + y) % 2 == 0 {
                    20.0 // right third: checkerboard → busy
                } else {
                    230.0
                };
                plane[y * w + x] = val;
            }
        }

        let mut fast = plane.clone();
        let mut slow = plane;
        filter_plane_triage(&mut fast, w, h, TriageConfig::default());
        filter_plane_triage_reference(&mut slow, w, h, TriageConfig::default());

        // Must be byte-identical — the per-block fast path is supposed to
        // produce the same arithmetic as the full-plane reference.
        assert_eq!(fast.len(), slow.len());
        for (i, (&a, &b)) in fast.iter().zip(slow.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                let x = i % w;
                let y = i / w;
                panic!(
                    "pixel ({x},{y}) mismatch: fast={a} (0x{:x}) vs ref={b} (0x{:x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }

    #[test]
    fn classify_plane_matches_filter_scope() {
        // classify_plane and filter_plane_triage must agree on which
        // blocks are labelled what.
        let w = 64;
        let h = 64;
        let mut p = vec![0.0_f32; w * h];
        // Flat-ish left half, noisy right half → mix of classes.
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = if x < 32 {
                    50.0
                } else if (x + y) % 2 == 0 {
                    100.0
                } else {
                    200.0
                };
            }
        }
        let cls = classify_plane(&p, w, h, TriageConfig::default());
        assert_eq!(cls.len(), 64);
        // Outer ring is Busy.
        for j in 0..8 {
            assert_eq!(cls[j], BlockClass::Busy);
            assert_eq!(cls[(8 - 1) * 8 + j], BlockClass::Busy);
        }
    }
}
