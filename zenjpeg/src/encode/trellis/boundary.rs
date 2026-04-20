//! Boundary-continuity D term for the trellis path (Phase 3 of issue #91).
//!
//! This module provides the left-neighbor-only boundary-continuity
//! distortion used to augment the trellis rate-distortion search. The key
//! insight from issue #91 is that evaluating the boundary term per
//! trellis candidate does **not** require a full IDCT: with a normalized
//! 2-D DCT-II, the reconstructed left-edge column of a block is a fixed
//! **linear** function of its natural-order DCT coefficients. The
//! coefficient → left-edge mapping is a constant 8×64 matrix E
//! (precomputed once at process start); per candidate the boundary edge
//! is just a dot product.
//!
//! # Architecture
//!
//! - [`LeftEdgeMatrix`] — the 8×64 precomputed matrix. Validated against
//!   [`crate::decode::idct::inverse_dct_8x8`] by unit test so its column
//!   matches the full IDCT to f32 precision.
//! - [`BoundaryContext`] — per-block boundary inputs passed into the
//!   trellis candidate evaluation: original edge columns (reference,
//!   committed-left-neighbor).
//! - [`boundary_distortion_from_coeffs`] — the D_boundary(C) formula
//!   from #91 evaluated via the dot-product trick.
//!
//! # Color space
//!
//! D_boundary is computed on the luma (Y) channel only. YCbCr for JPEG
//! uses BT.601 always; Y is a scalar so there is no colorspace
//! ambiguity. Chroma is left for future phases.

use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_ZIGZAG_ORDER};

/// 8×64 matrix E such that `edge[r] = sum_k E[r*64 + k] * C_natural[k]`
/// where `edge[r]` is the reconstructed left-column pixel `(row=r, col=0)`
/// of an 8×8 block whose natural-order dequantized DCT coefficients are
/// `C_natural[0..64]`.
///
/// The matrix is the partial evaluation of the 2-D IDCT-II at `x = 0`:
///
/// ```text
/// f(0, y) = sum_{u,v} N(u) N(v) F(u,v) cos((2y+1)vπ/16) cos(uπ/16)
/// ```
///
/// where `N(0) = 1/sqrt(2)` and `N(k) = 1` for `k > 0`. The inverse-DCT
/// scale convention used here matches
/// [`crate::decode::idct::inverse_dct_8x8`] (a 1/4 outer factor), verified
/// by [`tests::left_edge_matrix_matches_full_idct`].
#[derive(Debug, Clone)]
pub(crate) struct LeftEdgeMatrix {
    /// Row-major `[row(0..8)][coef(0..64)]`. Stored as a flat array so it
    /// fits in one allocation; length = 512.
    e: [f32; 8 * DCT_BLOCK_SIZE],
}

impl LeftEdgeMatrix {
    /// Build the matrix by measuring the actual IDCT response of a unit
    /// impulse at each coefficient position.
    ///
    /// This is algorithmically equivalent to evaluating the IDCT basis
    /// functions analytically, but by using the project's
    /// [`crate::decode::idct::inverse_dct_8x8`] directly we inherit its
    /// exact scale convention — important because the AAN / jpegli
    /// scaling and normalization differ from the textbook orthonormal
    /// IDCT by constant factors that would otherwise produce off-by-scale
    /// bugs.
    ///
    /// The construction cost is 64 IDCT calls (one per coefficient
    /// position) executed once per process; amortized to zero.
    fn new() -> Self {
        use crate::decode::idct::inverse_dct_8x8;
        let mut e = [0.0f32; 8 * DCT_BLOCK_SIZE];
        for k in 0..DCT_BLOCK_SIZE {
            let mut impulse = [0.0f32; DCT_BLOCK_SIZE];
            impulse[k] = 1.0;
            let spatial = inverse_dct_8x8(&impulse);
            for r in 0..8 {
                e[r * DCT_BLOCK_SIZE + k] = spatial[r * 8];
            }
        }
        Self { e }
    }

    /// Shared singleton — the matrix is constant across all encoder
    /// instances, so we build it once per process.
    fn shared() -> &'static Self {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<LeftEdgeMatrix> = OnceLock::new();
        INSTANCE.get_or_init(LeftEdgeMatrix::new)
    }

    /// Row `r` (0..8) as a natural-indexed basis slice of length 64.
    #[inline]
    fn row(&self, r: usize) -> &[f32; DCT_BLOCK_SIZE] {
        debug_assert!(r < 8);
        // SAFETY-equivalent via array_ref: split the flat [f32; 512] into
        // 8 chunks of 64, index by r. Written without unsafe.
        let start = r * DCT_BLOCK_SIZE;
        let end = start + DCT_BLOCK_SIZE;
        let slice = &self.e[start..end];
        <&[f32; DCT_BLOCK_SIZE]>::try_from(slice)
            .expect("row slice is exactly 64 elements by construction")
    }
}

/// Per-block boundary-continuity inputs.
///
/// Populated by the strip processor before invoking the trellis with
/// the boundary-RD flag on. When `left_neighbor_committed_right` is
/// `None`, D_boundary is 0 — the current block has no left neighbor
/// (first block in a row).
#[derive(Debug, Clone)]
pub(crate) struct BoundaryContext {
    /// Original block's left-edge column (from IDCT of unquantized DCT).
    /// Values live in the DCT-input pixel domain (`[-128, 127]`
    /// level-shifted).
    pub orig_curr_left: [f32; 8],
    /// Original left neighbor's right-edge column. `None` when there is
    /// no left neighbor.
    pub orig_left_right: Option<[f32; 8]>,
    /// Left neighbor's **committed** (quantized-and-reconstructed)
    /// right-edge column. `None` when there is no left neighbor or when
    /// boundary-RD was skipped for the left block.
    pub left_neighbor_committed_right: Option<[f32; 8]>,
    /// Seam-jump weight α (from the D_b formula in #91). Not β —
    /// β scales the entire D_boundary contribution in the trellis
    /// rate-distortion cost and is applied by the caller.
    pub alpha: f32,
}

/// Reconstruct the left-edge column of a block from its **natural-order**,
/// dequantized f32 DCT coefficients via the precomputed [`LeftEdgeMatrix`].
///
/// This is the dot-product trick that makes per-candidate boundary
/// evaluation cheap inside the trellis loop: O(64) multiplies per edge
/// row, O(512) per candidate, no IDCT transpose, no register pressure.
#[inline]
pub(crate) fn reconstruct_left_edge_from_natural(
    coeffs_natural: &[f32; DCT_BLOCK_SIZE],
) -> [f32; 8] {
    let m = LeftEdgeMatrix::shared();
    let mut edge = [0.0f32; 8];
    for r in 0..8 {
        let row = m.row(r);
        let mut s = 0.0f32;
        for k in 0..DCT_BLOCK_SIZE {
            s += row[k] * coeffs_natural[k];
        }
        edge[r] = s;
    }
    edge
}

/// Compute D_boundary(C) from a block's quantized zigzag-ordered i16
/// coefficients plus its natural-order quant table and a
/// [`BoundaryContext`].
///
/// ```text
/// D_b = ||rec_curr.left - orig_curr.left||²
///     + ||rec_left.right - orig_left.right||²
///     + α · ||seam_jump_rec - seam_jump_orig||²
/// ```
///
/// The `rec_curr.left` is built from the candidate's coefficients via the
/// dot-product matrix. `rec_left.right` is **not** recomputed per
/// candidate — it was committed when the left block finished and only
/// depends on the current block's left edge for the seam-jump term.
///
/// When there is no left neighbor, D_boundary is 0.
#[must_use]
pub(crate) fn boundary_distortion_from_quantized(
    zigzag_coeffs: &[i16; DCT_BLOCK_SIZE],
    quant_natural: &[u16; DCT_BLOCK_SIZE],
    ctx: &BoundaryContext,
) -> f32 {
    let Some(rec_lr) = ctx.left_neighbor_committed_right.as_ref() else {
        return 0.0;
    };
    let Some(orig_lr) = ctx.orig_left_right.as_ref() else {
        return 0.0;
    };
    // Dequantize to natural-order f32.
    let mut natural = [0.0f32; DCT_BLOCK_SIZE];
    for n in 0..DCT_BLOCK_SIZE {
        let z = JPEG_ZIGZAG_ORDER[n] as usize;
        natural[n] = zigzag_coeffs[z] as f32 * quant_natural[n] as f32;
    }
    let rec_curr_left = reconstruct_left_edge_from_natural(&natural);
    boundary_distortion_from_edges(
        &rec_curr_left,
        Some(rec_lr),
        &ctx.orig_curr_left,
        Some(orig_lr),
        ctx.alpha,
    )
}

/// Primitive: D_b formula given all four edges.
///
/// Factored out so unit tests can cover it without constructing a full
/// trellis block. Matches
/// [`crate::encode::boundary_rd::boundary_distortion`] in Phase 2 but
/// lives in this module so the trellis path does not take a dependency on
/// Phase 2's module (Phase 2 and Phase 3 are parallel stack branches).
#[inline]
#[must_use]
pub(crate) fn boundary_distortion_from_edges(
    rec_curr_left: &[f32; 8],
    rec_left_right: Option<&[f32; 8]>,
    orig_curr_left: &[f32; 8],
    orig_left_right: Option<&[f32; 8]>,
    alpha: f32,
) -> f32 {
    let Some(rec_lr) = rec_left_right else {
        return 0.0;
    };
    let Some(orig_lr) = orig_left_right else {
        return 0.0;
    };
    let mut edge_current = 0.0f32;
    let mut edge_left = 0.0f32;
    let mut seam = 0.0f32;
    for r in 0..8 {
        let d_cur = rec_curr_left[r] - orig_curr_left[r];
        edge_current += d_cur * d_cur;
        let d_left = rec_lr[r] - orig_lr[r];
        edge_left += d_left * d_left;
        let jump_rec = rec_curr_left[r] - rec_lr[r];
        let jump_orig = orig_curr_left[r] - orig_lr[r];
        let d_jump = jump_rec - jump_orig;
        seam += d_jump * d_jump;
    }
    edge_current + edge_left + alpha * seam
}

/// Reconstruct left-edge column from an unquantized f32 natural-order
/// block. Used by the strip to cache `orig_*_left` / `orig_*_right`
/// without running the full IDCT.
#[inline]
#[must_use]
pub(crate) fn reconstruct_left_edge_from_natural_slice(
    coeffs_natural: &[f32; DCT_BLOCK_SIZE],
) -> [f32; 8] {
    reconstruct_left_edge_from_natural(coeffs_natural)
}

/// 8×64 matrix R analogous to [`LeftEdgeMatrix`] but for `x = 7`.
/// Built by IDCT impulse response just like the left matrix.
#[derive(Debug, Clone)]
struct RightEdgeMatrix {
    e: [f32; 8 * DCT_BLOCK_SIZE],
}

impl RightEdgeMatrix {
    fn new() -> Self {
        use crate::decode::idct::inverse_dct_8x8;
        let mut e = [0.0f32; 8 * DCT_BLOCK_SIZE];
        for k in 0..DCT_BLOCK_SIZE {
            let mut impulse = [0.0f32; DCT_BLOCK_SIZE];
            impulse[k] = 1.0;
            let spatial = inverse_dct_8x8(&impulse);
            for r in 0..8 {
                e[r * DCT_BLOCK_SIZE + k] = spatial[r * 8 + 7];
            }
        }
        Self { e }
    }

    fn shared() -> &'static Self {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<RightEdgeMatrix> = OnceLock::new();
        INSTANCE.get_or_init(RightEdgeMatrix::new)
    }

    #[inline]
    fn row(&self, r: usize) -> &[f32; DCT_BLOCK_SIZE] {
        let start = r * DCT_BLOCK_SIZE;
        let slice = &self.e[start..start + DCT_BLOCK_SIZE];
        <&[f32; DCT_BLOCK_SIZE]>::try_from(slice)
            .expect("row slice is exactly 64 elements by construction")
    }
}

/// Reconstruct right-edge column (`x = 7`) from natural-order coefficients.
///
/// Symmetric companion to [`reconstruct_left_edge_from_natural`] but for
/// `x = 7`. Used to cache a block's committed right edge after it finishes
/// trellis quantization, so the next block in the row can read it as its
/// `left_neighbor_committed_right`.
#[must_use]
pub(crate) fn reconstruct_right_edge_from_natural(
    coeffs_natural: &[f32; DCT_BLOCK_SIZE],
) -> [f32; 8] {
    let m = RightEdgeMatrix::shared();
    let mut edge = [0.0f32; 8];
    for r in 0..8 {
        let row = m.row(r);
        let mut s = 0.0f32;
        for k in 0..DCT_BLOCK_SIZE {
            s += row[k] * coeffs_natural[k];
        }
        edge[r] = s;
    }
    edge
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::idct::inverse_dct_8x8;
    use crate::encode::dct::forward_dct_8x8;

    /// Build a deterministic but non-trivial 8×8 spatial block.
    fn make_spatial(f: impl Fn(usize, usize) -> f32) -> [f32; DCT_BLOCK_SIZE] {
        let mut out = [0.0f32; DCT_BLOCK_SIZE];
        for r in 0..8 {
            for c in 0..8 {
                out[r * 8 + c] = f(r, c);
            }
        }
        out
    }

    /// The E matrix applied to a DCT of a known block must recover the
    /// same left column as the full IDCT of the same DCT.
    #[test]
    fn left_edge_matrix_matches_full_idct() {
        // Use a spatial block in the level-shifted [-128, 127] domain.
        let spatial = make_spatial(|r, c| ((r * 13 + c * 7) as i32 - 64) as f32);
        // encode's forward_dct_8x8 uses a 1/64 outer scale (matching C++
        // jpegli); the decoder's inverse_dct_8x8 takes a 1/8-scaled input
        // (the ×8 bridge lives in the dequant multiplier). So to round-trip
        // via inverse_dct_8x8 we must rescale forward_dct output by ×8.
        let dct = forward_dct_8x8(&spatial);
        let mut dct_scaled = [0.0f32; DCT_BLOCK_SIZE];
        for i in 0..DCT_BLOCK_SIZE {
            dct_scaled[i] = dct[i] * 8.0;
        }
        let full_idct = inverse_dct_8x8(&dct_scaled);
        let via_matrix = reconstruct_left_edge_from_natural(&dct_scaled);
        for r in 0..8 {
            let want = full_idct[r * 8];
            let got = via_matrix[r];
            let err = (want - got).abs();
            assert!(
                err < 1e-2,
                "row {r}: want {want}, got {got}, err {err}"
            );
        }
        // And the reconstructed left column should approximately equal the
        // original spatial left column (modulo f32 DCT-IDCT roundtrip
        // precision).
        for r in 0..8 {
            let want = spatial[r * 8];
            let got = via_matrix[r];
            assert!(
                (want - got).abs() < 1e-2,
                "roundtrip row {r}: want {want}, got {got}"
            );
        }
    }

    /// Right-edge basis likewise matches the full IDCT.
    #[test]
    fn right_edge_matches_full_idct() {
        let spatial = make_spatial(|r, c| ((r * 5 + c * 11) as i32 - 30) as f32);
        let dct = forward_dct_8x8(&spatial);
        let mut dct_scaled = [0.0f32; DCT_BLOCK_SIZE];
        for i in 0..DCT_BLOCK_SIZE {
            dct_scaled[i] = dct[i] * 8.0;
        }
        let full_idct = inverse_dct_8x8(&dct_scaled);
        let via_matrix = reconstruct_right_edge_from_natural(&dct_scaled);
        for r in 0..8 {
            let want = full_idct[r * 8 + 7];
            let got = via_matrix[r];
            assert!(
                (want - got).abs() < 1e-2,
                "row {r}: want {want}, got {got}"
            );
        }
    }

    #[test]
    fn boundary_distortion_zero_when_all_equal() {
        let left_col = [0.5f32; 8];
        let right_col = [-0.3f32; 8];
        let d = boundary_distortion_from_edges(
            &left_col,
            Some(&right_col),
            &left_col,
            Some(&right_col),
            1.0,
        );
        assert_eq!(d, 0.0);
    }

    #[test]
    fn boundary_distortion_zero_without_left_neighbor() {
        let rec = [10.0f32; 8];
        let orig = [0.0f32; 8];
        let d = boundary_distortion_from_edges(&rec, None, &orig, None, 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn boundary_distortion_scales_with_alpha() {
        let rec_curr = [0.0f32; 8];
        let orig_curr = [4.0f32; 8];
        let rec_lr = [0.0f32; 8];
        let orig_lr = [0.0f32; 8];
        let d1 = boundary_distortion_from_edges(
            &rec_curr,
            Some(&rec_lr),
            &orig_curr,
            Some(&orig_lr),
            1.0,
        );
        let d2 = boundary_distortion_from_edges(
            &rec_curr,
            Some(&rec_lr),
            &orig_curr,
            Some(&orig_lr),
            2.0,
        );
        // With α=1 vs α=2, seam term is 8 × (4-0)² = 128; Δ = 128.
        assert!(
            (d2 - d1 - 128.0).abs() < 1e-3,
            "d1={d1} d2={d2}, expected 128 delta"
        );
        // And D_b must strictly increase with α when seam jump ≠ 0.
        assert!(d2 > d1);
    }

    #[test]
    fn dot_product_matches_idct_on_quantized_block() {
        // Full cycle: spatial → forward DCT → quantize → dequant → IDCT,
        // vs spatial → forward DCT → quantize → dot-product.
        // The two must agree on the left column.
        let spatial = make_spatial(|r, c| ((r * 7 + c * 13) as i32 - 50) as f32);
        let dct = forward_dct_8x8(&spatial);
        // Fabricate a quant table with varying values.
        let quant: [u16; DCT_BLOCK_SIZE] = core::array::from_fn(|i| (i as u16 + 2));
        // Quantize (simple rounding, ×8 bridge).
        let mut zigzag = [0i16; DCT_BLOCK_SIZE];
        for i in 0..DCT_BLOCK_SIZE {
            let zz = JPEG_ZIGZAG_ORDER[i] as usize;
            let q = (dct[i] * 8.0 / quant[i] as f32).round() as i32;
            zigzag[zz] = q.clamp(-32768, 32767) as i16;
        }
        // Dequant -> natural f32 (for both paths).
        let mut natural = [0.0f32; DCT_BLOCK_SIZE];
        for n in 0..DCT_BLOCK_SIZE {
            let z = JPEG_ZIGZAG_ORDER[n] as usize;
            natural[n] = zigzag[z] as f32 * quant[n] as f32;
        }
        let full_idct = inverse_dct_8x8(&natural);
        let via_matrix = reconstruct_left_edge_from_natural(&natural);
        for r in 0..8 {
            let want = full_idct[r * 8];
            let got = via_matrix[r];
            assert!(
                (want - got).abs() < 1e-2,
                "row {r}: want {want}, got {got}"
            );
        }
    }
}
