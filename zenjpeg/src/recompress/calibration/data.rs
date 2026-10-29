//! Corpus-fitted calibration constants.
//!
//! Generated 2026-05-28 by `scripts/fit_calibration.py` from
//! `benchmarks/cid22_15img_postfit_v2_420_2026-05-28.tsv` (1080-cell
//! sweep, 15 CID22 references, 4:2:0 chroma, zenjpeg encoder with
//! auto_optimize). The Tuned strategy's measured zensim-A vs reference
//! and size ratio are tabulated per `(source_ba_bucket, target)` cell.
//!
//! Indexing: `CELL_MEDIAN_ZENSIM_A_420[src_idx][tgt_idx]` where
//! `src_idx` is the position in [`SOURCE_BA_BUCKETS_420`] and
//! `tgt_idx` is the position in [`TARGET_GRID_420`]. Cells with no
//! sweep coverage carry `f32::NAN`.
//!
//! When a NaN cell is encountered, the lookup falls back to nearest-
//! covered neighbour (see `super::lookup`). New chroma modes need their
//! own fitted table — see DESIGN.md for the v0.2 plan.

/// 4:4:4 buckets carry one less because the high-BA tail compresses
/// less under no-chroma-subsampling — the highest detected BA at
/// `--source-qs 20` for 4:4:4 sits at 6.7 not 8.0.
pub const SOURCE_BA_BUCKETS_444: &[f32] = &[0.5, 1.3, 1.8, 2.5, 3.5, 4.6, 5.7, 6.7];
pub const TARGET_GRID_444: &[f32] = &[30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0];

#[rustfmt::skip]
pub const CELL_MEDIAN_ZENSIM_A_444: &[[f32; 7]; 8] = &[
    [51.638, 58.870, 63.786, 67.843, 74.628, 84.175, 87.793],
    [51.295, 58.564, 63.529, 67.510, 73.159, 82.906, f32::NAN],
    [50.615, 57.180, 60.017, 64.104, 70.299, f32::NAN, f32::NAN],
    [50.359, 55.721, 58.486, 64.018, 70.849, f32::NAN, f32::NAN],
    [47.718, 54.260, 56.658, 64.082, f32::NAN, f32::NAN, f32::NAN],
    [48.420, 53.850, 58.432, 64.595, f32::NAN, f32::NAN, f32::NAN],
    [46.528, 54.132, 54.429, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
    [46.556, 50.519, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
];

#[rustfmt::skip]
pub const CELL_MEDIAN_SIZE_RATIO_444: &[[f32; 7]; 8] = &[
    [0.2227, 0.2555, 0.2797, 0.3396, 0.4014, 0.6373, 0.9543],
    [0.3085, 0.3551, 0.3893, 0.4865, 0.5750, 0.9009, f32::NAN],
    [0.4550, 0.5177, 0.5644, 0.6815, 0.8166, f32::NAN, f32::NAN],
    [0.5551, 0.6340, 0.7066, 0.8457, 0.9864, f32::NAN, f32::NAN],
    [0.6630, 0.7512, 0.8360, 0.9729, f32::NAN, f32::NAN, f32::NAN],
    [0.7493, 0.8742, 0.9490, 0.9899, f32::NAN, f32::NAN, f32::NAN],
    [0.8651, 0.9693, 0.9898, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
    [0.9317, 0.9898, f32::NAN, f32::NAN, f32::NAN, f32::NAN, f32::NAN],
];

/// Lookup for 4:4:4 (`Subsampling::S444`). Mirrors [`lookup_420`].
pub fn lookup_444(source_ba: f32, target: f32) -> Option<(f32, f32)> {
    let s_idx = nearest_index(source_ba, SOURCE_BA_BUCKETS_444)?;
    let t_idx = nearest_index(target, TARGET_GRID_444)?;
    let zensim_a = nearest_covered(CELL_MEDIAN_ZENSIM_A_444, s_idx, t_idx)?;
    let size_ratio = nearest_covered(CELL_MEDIAN_SIZE_RATIO_444, s_idx, t_idx)?;
    Some((zensim_a, size_ratio))
}

pub const SOURCE_BA_BUCKETS_420: &[f32] = &[0.5, 1.3, 2.5, 3.5, 4.6, 5.7, 6.7, 8.0];

pub const TARGET_GRID_420: &[f32] = &[30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 90.0];

// n=50 forced-Tuned cumulative sweep (2026-05-29): every cell is the
// MEDIAN over 50 CID22-512 refs × 9 synthesized jpegli source-q at 4:2:0,
// recompressed with the Tuned strategy (`cumulative-sweep --force-tuned`,
// fit MAE 6.05, n=3600). Fully populated — no NaN, and adds the target=90
// column — unlike the prior 15-image smart-router fit which left the
// lower-right uncovered (the router routed those cells away from Tuned).
// Raw sweep in benchmarks/calibration-n50-2026-05-29.pointer.md → /mnt/v.
#[rustfmt::skip]
pub const CELL_MEDIAN_ZENSIM_A_420: &[[f32; 8]; 8] = &[
    [51.178, 55.854, 59.030, 63.950, 68.220, 79.302, 82.158, 82.743],
    [51.109, 55.578, 58.512, 63.912, 66.895, 77.327, 78.660, 79.853],
    [50.760, 54.651, 56.883, 63.734, 64.381, 68.876, 71.127, 71.510],
    [48.879, 52.968, 54.566, 61.492, 64.111, 64.071, 64.128, 64.174],
    [47.235, 51.961, 54.158, 63.646, 60.110, 63.722, 63.751, 63.782],
    [45.190, 51.669, 53.772, 54.820, 55.736, 58.334, 59.130, 59.532],
    [44.892, 51.613, 53.116, 51.130, 53.771, 54.422, 54.891, 55.335],
    [41.632, 42.001, 42.069, 44.686, 46.328, 47.127, 47.751, 48.084],
];

#[rustfmt::skip]
pub const CELL_MEDIAN_SIZE_RATIO_420: &[[f32; 8]; 8] = &[
    [0.2214, 0.2486, 0.2730, 0.3187, 0.3831, 0.6081, 0.9021, 1.4397],
    [0.3174, 0.3626, 0.3918, 0.4622, 0.5585, 0.8774, 1.0934, 1.7132],
    [0.4591, 0.5161, 0.5666, 0.6663, 0.8048, 1.0600, 1.2514, 2.0250],
    [0.5665, 0.6378, 0.6977, 0.8269, 0.9655, 1.1299, 1.3506, 2.2290],
    [0.6647, 0.7539, 0.8342, 0.9539, 1.0187, 1.1886, 1.4042, 2.4073],
    [0.7631, 0.8722, 0.9332, 0.9927, 1.0585, 1.2228, 1.4374, 2.5674],
    [0.8577, 0.9412, 0.9738, 1.0280, 1.0903, 1.2647, 1.4943, 2.7433],
    [0.9506, 0.9900, 1.0148, 1.0733, 1.1427, 1.3266, 1.5673, 2.9518],
];

/// Look up the table by `(source_estimated_zensim_a, target)`. Source
/// estimate is converted to BA-distance via the inverse of the
/// `target.rs` BA↔zensim-A anchor; nearest-neighbour bucket then
/// indexes into the table. Falls back to NaN→nearest-covered neighbour.
pub fn lookup_420(source_ba: f32, target: f32) -> Option<(f32, f32)> {
    let s_idx = nearest_index(source_ba, SOURCE_BA_BUCKETS_420)?;
    let t_idx = nearest_index(target, TARGET_GRID_420)?;
    let zensim_a = nearest_covered(CELL_MEDIAN_ZENSIM_A_420, s_idx, t_idx)?;
    let size_ratio = nearest_covered(CELL_MEDIAN_SIZE_RATIO_420, s_idx, t_idx)?;
    Some((zensim_a, size_ratio))
}

fn nearest_index(value: f32, axis: &[f32]) -> Option<usize> {
    if axis.is_empty() {
        return None;
    }
    let mut best = 0usize;
    let mut best_d = (value - axis[0]).abs();
    for (i, &a) in axis.iter().enumerate().skip(1) {
        let d = (value - a).abs();
        if d < best_d {
            best = i;
            best_d = d;
        }
    }
    Some(best)
}

/// Read `table[s_idx][t_idx]`; if NaN, spiral out to find the nearest
/// covered cell. This keeps the lookup defined even when a sweep cell
/// has no coverage (the NoOp band filters out infeasible cells before
/// this point, so spiralling typically lands on an adjacent
/// definitely-covered neighbour).
fn nearest_covered<const N: usize>(
    table: &[[f32; N]; 8],
    s_idx: usize,
    t_idx: usize,
) -> Option<f32> {
    if !table[s_idx][t_idx].is_nan() {
        return Some(table[s_idx][t_idx]);
    }
    let rows = table.len();
    let cols = N;
    // BFS in (ds, dt) space.
    let mut radius = 1usize;
    let max_radius = rows.max(cols);
    while radius <= max_radius {
        for ds in -(radius as isize)..=(radius as isize) {
            for dt in -(radius as isize)..=(radius as isize) {
                if ds.abs() != radius as isize && dt.abs() != radius as isize {
                    continue;
                }
                let nr = (s_idx as isize) + ds;
                let nc = (t_idx as isize) + dt;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let v = table[nr as usize][nc as usize];
                if !v.is_nan() {
                    return Some(v);
                }
            }
        }
        radius += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_420_returns_known_cell() {
        // src_ba 0.5, target 70 → row 0, col 4 (n=50 forced-Tuned table):
        // z = 68.220, ratio = 0.3831.
        let (z, r) = lookup_420(0.5, 70.0).expect("known cell");
        assert!((z - 68.220).abs() < 0.001);
        assert!((r - 0.3831).abs() < 0.001);
    }

    #[test]
    fn lookup_420_covered_cell_is_finite() {
        // The n=50 forced-Tuned table is fully populated (no NaN); every
        // covered cell returns a finite, in-range value. (Was the
        // NaN-spiral fallback test against the prior sparse 15-image fit.)
        let (z, _r) = lookup_420(8.0, 60.0).expect("covered cell");
        assert!(z.is_finite());
        assert!(z > 30.0 && z < 90.0);
    }

    #[test]
    fn lookup_444_nan_cell_spirals_to_finite_neighbour() {
        // The 4:4:4 table is still NaN-heavy in the lower-right; `lookup_444`
        // relies on `nearest_covered`'s spiral for those cells. src_ba 6.7
        // (idx 7) × target 85 (idx 6) is `f32::NAN` in the raw table — the
        // lookup must spiral out to a finite, in-range neighbour. Guards the
        // NaN-fallback path the renamed 420 test no longer covers.
        assert!(CELL_MEDIAN_ZENSIM_A_444[7][6].is_nan());
        let (z, r) = lookup_444(6.7, 85.0).expect("nan cell spirals to neighbour");
        assert!(z.is_finite() && z > 30.0 && z < 95.0);
        assert!(r.is_finite() && r > 0.0);
    }

    #[test]
    fn lookup_444_returns_known_cell() {
        // src_ba 0.5 (idx 0) × target 50 (idx 2) = 63.786 / 0.2797.
        let (z, r) = lookup_444(0.5, 50.0).expect("known cell");
        assert!((z - 63.786).abs() < 0.001);
        assert!((r - 0.2797).abs() < 0.001);
    }
}
