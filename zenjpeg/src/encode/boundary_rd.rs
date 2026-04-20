//! Boundary-continuity refinement (Phase 2 of issue #91).
//!
//! This is the non-trellis, left-neighbor-only refinement: after a luma
//! block is quantized via the fast SIMD path, we IDCT it, compare its
//! reconstructed left-edge column against the committed right-edge column
//! of its left neighbor (and against the block's own pre-quantization
//! reconstruction), and — if the boundary distortion exceeds a fraction of
//! the block's AC DCT energy — redo the quantize with a shrunken AQ
//! strength. The candidate with the lowest boundary distortion wins.
//!
//! This runs inside [`super::strip::StripProcessor::quantize_prev_pending_imcu`]
//! under the opt-in `EncoderConfig::boundary_rd` flag. It is a no-op when
//! the flag is off, and is also skipped when trellis quantization is
//! active (trellis has its own boundary-D-augment — Phase 3 of #91).
//!
//! # Domain conventions
//!
//! All pixel buffers in this module are in the **DCT-input domain**
//! (level-shifted to `[-128, 127]`). Both [`decode::idct::inverse_dct_8x8`]
//! output and the forward-DCT input produced by
//! [`super::strip::extract_block_from_strip_wide`] use this convention, so
//! "original" edge columns can be read directly from either the raw f32
//! DCT block (via [`idct_reference_block`]) or the IDCT of the quantized
//! coefficients (via [`idct_quantized_block`]) without additional shift.
//!
//! # Color space note
//!
//! D_b is computed on the luma (Y) channel only. JPEG's YCbCr uses the
//! BT.601 matrix; Y is a single scalar so there is no colorspace
//! ambiguity here. Chroma blocks are not refined in Phase 2 — block-seam
//! artifacts at chroma subsampling granularity (16-pixel MCU boundaries on
//! 4:2:0) would need a separate treatment.

use crate::decode::idct::inverse_dct_8x8;
use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER};
use crate::foundation::simd_types::Block8x8f;

/// Reconstruct the 8×8 spatial block from an unquantized f32 DCT block.
///
/// Used as the "original" reference in D_b: DCT is invertible, so this
/// is equivalent (up to f32 precision) to the source pixels that the
/// forward DCT was run on. Output is in the level-shifted
/// `[-128, 127]` domain.
///
/// The encoder's forward DCT writes coefficients in a 1/64-scaled
/// convention, while the decoder's [`inverse_dct_8x8`] expects 1/8-scaled
/// input (the quantize × 8/q → dequant × q path bridges the gap for the
/// quantized path). We apply the ×8 factor here so the reference block
/// comes out in the same pixel domain as
/// [`idct_quantized_block`].
#[must_use]
pub(crate) fn idct_reference_block(dct_f32: &Block8x8f) -> [f32; DCT_BLOCK_SIZE] {
    let mut natural = [0.0f32; DCT_BLOCK_SIZE];
    for r in 0..8 {
        for c in 0..8 {
            natural[r * 8 + c] = dct_f32.rows[r][c] * 8.0;
        }
    }
    inverse_dct_8x8(&natural)
}

/// Dequantize a zigzag-ordered i16 coefficient block to a natural-order
/// f32 block, then IDCT to produce the reconstructed 8×8 spatial block.
///
/// `quant_values_natural` is the natural-row-major quant table (as stored
/// in [`crate::types::QuantTable::values`]); the `u16` values are treated
/// as positive divisors. Output is in the level-shifted `[-128, 127]`
/// domain.
#[must_use]
pub(crate) fn idct_quantized_block(
    zigzag_coeffs: &[i16; DCT_BLOCK_SIZE],
    quant_values_natural: &[u16; DCT_BLOCK_SIZE],
) -> [f32; DCT_BLOCK_SIZE] {
    let mut natural = [0.0f32; DCT_BLOCK_SIZE];
    for n in 0..DCT_BLOCK_SIZE {
        // `values` is natural-indexed; zigzag output position for natural
        // index n is JPEG_ZIGZAG_ORDER[n], but equivalently
        // natural[n] = zigzag[JPEG_ZIGZAG_ORDER[n]]. We walk zigzag
        // positions and scatter into natural via JPEG_NATURAL_ORDER:
        // natural[JPEG_NATURAL_ORDER[z]] = zigzag[z] * quant_natural[JPEG_NATURAL_ORDER[z]].
        // Use the simpler form: iterate natural, pick zigzag slot.
        let zigzag_idx = crate::foundation::consts::JPEG_ZIGZAG_ORDER[n] as usize;
        natural[n] = zigzag_coeffs[zigzag_idx] as f32 * quant_values_natural[n] as f32;
    }
    // Silence unused-import lint if the natural-order LUT becomes redundant.
    let _ = JPEG_NATURAL_ORDER;
    inverse_dct_8x8(&natural)
}

/// Extract the left-edge column (x = 0) of an 8×8 spatial block.
#[inline]
#[must_use]
pub(crate) fn left_edge_col(block: &[f32; DCT_BLOCK_SIZE]) -> [f32; 8] {
    let mut col = [0.0f32; 8];
    for r in 0..8 {
        col[r] = block[r * 8];
    }
    col
}

/// Extract the right-edge column (x = 7) of an 8×8 spatial block.
#[inline]
#[must_use]
pub(crate) fn right_edge_col(block: &[f32; DCT_BLOCK_SIZE]) -> [f32; 8] {
    let mut col = [0.0f32; 8];
    for r in 0..8 {
        col[r] = block[r * 8 + 7];
    }
    col
}

/// Sum of squared differences between two 8-element edge columns.
#[inline]
#[must_use]
fn ssd_col(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut s = 0.0f32;
    for r in 0..8 {
        let d = a[r] - b[r];
        s += d * d;
    }
    s
}

/// Sum of squared differences between two "seam jumps" across an 8-pixel
/// vertical seam, where `seam_jump[r] = right_col[r] - left_col[r]`.
///
/// This is the perceptual-target term from #91: if the reconstruction
/// preserves the original cross-seam gradient, the seam is invisible;
/// if it inflates or deflates the jump, the seam is perceptually visible.
#[inline]
#[must_use]
fn ssd_seam_jump(
    rec_left_right: &[f32; 8],
    rec_curr_left: &[f32; 8],
    orig_left_right: &[f32; 8],
    orig_curr_left: &[f32; 8],
) -> f32 {
    let mut s = 0.0f32;
    for r in 0..8 {
        let jump_rec = rec_curr_left[r] - rec_left_right[r];
        let jump_orig = orig_curr_left[r] - orig_left_right[r];
        let d = jump_rec - jump_orig;
        s += d * d;
    }
    s
}

/// Compute the boundary-continuity distortion D_b (issue #91 formula).
///
/// ```text
/// D_b = ||rec_current.left - orig_current.left||²
///     + ||rec_left.right   - orig_left.right  ||²
///     + α · ||seam_jump_rec - seam_jump_orig  ||²
/// ```
///
/// All four columns are expected in the same (level-shifted) pixel domain.
/// If `rec_left_right_committed` is `None` (no left neighbor; first block
/// of a row, or boundary_rd disabled for the left neighbor), D_b is
/// returned as 0 — the refinement has nothing to consult.
#[inline]
#[must_use]
pub(crate) fn boundary_distortion(
    rec_curr_left: &[f32; 8],
    rec_left_right_committed: Option<&[f32; 8]>,
    orig_curr_left: &[f32; 8],
    orig_left_right: Option<&[f32; 8]>,
    alpha: f32,
) -> f32 {
    let Some(rec_lr) = rec_left_right_committed else {
        return 0.0;
    };
    let Some(orig_lr) = orig_left_right else {
        return 0.0;
    };
    let edge_current = ssd_col(rec_curr_left, orig_curr_left);
    let edge_left = ssd_col(rec_lr, orig_lr);
    let seam = ssd_seam_jump(rec_lr, rec_curr_left, orig_lr, orig_curr_left);
    edge_current + edge_left + alpha * seam
}

/// Per-block AC DCT energy. Sum of squared AC coefficients (DC excluded).
///
/// Used as the denominator of the D_b threshold check: refinement fires
/// when `D_b > threshold × ac_energy`. Blocks with very small AC energy
/// (flat regions) will never trigger refinement; blocks with large AC
/// energy (texture) will tolerate proportionally higher D_b before
/// refinement is deemed worthwhile.
#[inline]
#[must_use]
pub(crate) fn ac_dct_energy(dct_f32: &Block8x8f) -> f32 {
    let mut e = 0.0f32;
    for r in 0..8 {
        for c in 0..8 {
            if r == 0 && c == 0 {
                continue;
            }
            let v = dct_f32.rows[r][c];
            e += v * v;
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::dct::forward_dct_8x8;

    fn make_block(f: impl Fn(usize, usize) -> f32) -> Block8x8f {
        let mut rows = [[0.0f32; 8]; 8];
        for r in 0..8 {
            for c in 0..8 {
                rows[r][c] = f(r, c);
            }
        }
        Block8x8f { rows }
    }

    #[test]
    fn idct_reference_roundtrips_forward_dct() {
        // A DCT-IDCT roundtrip of a known spatial block should recover it
        // to f32 precision. We use reasonably bounded values to stay in
        // the level-shifted domain.
        let spatial_in = make_block(|r, c| ((r * 13 + c * 7) as i32 - 64) as f32);
        // Forward DCT -> Block8x8f ; then IDCT via our helper must recover.
        let mut coeff_arr = [0.0f32; DCT_BLOCK_SIZE];
        for r in 0..8 {
            for c in 0..8 {
                coeff_arr[r * 8 + c] = spatial_in.rows[r][c];
            }
        }
        let dct = forward_dct_8x8(&coeff_arr);
        // forward_dct_8x8 returns natural-order f32 coefficients; wrap in Block8x8f.
        let mut dct_block = Block8x8f::default();
        for r in 0..8 {
            for c in 0..8 {
                dct_block.rows[r][c] = dct[r * 8 + c];
            }
        }
        let spatial_out = idct_reference_block(&dct_block);
        let mut max_err = 0.0f32;
        for r in 0..8 {
            for c in 0..8 {
                let want = spatial_in.rows[r][c];
                let got = spatial_out[r * 8 + c];
                max_err = max_err.max((want - got).abs());
            }
        }
        // f32 DCT+IDCT roundtrip typically stays within ~1e-3 for
        // bounded inputs.
        assert!(max_err < 1e-2, "roundtrip error too large: {max_err}");
    }

    #[test]
    fn db_zero_when_reconstruction_matches_original() {
        let left_col = [0.5f32; 8];
        let right_col = [-0.3f32; 8];
        // Both "rec" and "orig" use the same values — D_b must be zero.
        let d = boundary_distortion(&left_col, Some(&right_col), &left_col, Some(&right_col), 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn db_zero_without_left_neighbor() {
        // First block of a row has no committed left neighbor — D_b = 0
        // regardless of how far the current edge is from the original.
        let rec = [10.0f32; 8];
        let orig = [0.0f32; 8];
        let d = boundary_distortion(&rec, None, &orig, None, 1.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn db_positive_for_synthetic_seam() {
        // Construct a case where the reconstructed current-block left-col
        // deviates from the original: D_b must be > 0.
        let rec_curr_left = [0.0f32; 8];
        let orig_curr_left = [4.0f32; 8];
        let rec_left_right = [0.0f32; 8];
        let orig_left_right = [0.0f32; 8];
        let d = boundary_distortion(
            &rec_curr_left,
            Some(&rec_left_right),
            &orig_curr_left,
            Some(&orig_left_right),
            1.0,
        );
        // Edge-current term: 8 × (0 - 4)² = 128.
        // Edge-left term: 0.
        // Seam term: seam_orig = 4 - 0 = 4, seam_rec = 0 - 0 = 0; 8 × 16 = 128.
        // Total with α=1: 128 + 0 + 128 = 256.
        assert!((d - 256.0).abs() < 1e-3, "expected 256, got {d}");
    }

    #[test]
    fn seam_term_scales_with_alpha() {
        let rec_curr_left = [0.0f32; 8];
        let orig_curr_left = [4.0f32; 8];
        let rec_left_right = [0.0f32; 8];
        let orig_left_right = [0.0f32; 8];
        let d1 = boundary_distortion(
            &rec_curr_left,
            Some(&rec_left_right),
            &orig_curr_left,
            Some(&orig_left_right),
            1.0,
        );
        let d2 = boundary_distortion(
            &rec_curr_left,
            Some(&rec_left_right),
            &orig_curr_left,
            Some(&orig_left_right),
            2.0,
        );
        // D_b(α=2) - D_b(α=1) = one seam-term contribution (128).
        assert!((d2 - d1 - 128.0).abs() < 1e-3, "d1={d1} d2={d2}");
    }

    #[test]
    fn ac_energy_excludes_dc() {
        let b = make_block(|r, c| if r == 0 && c == 0 { 1000.0 } else { 0.0 });
        assert_eq!(ac_dct_energy(&b), 0.0);

        let b = make_block(|r, c| if r == 0 && c == 1 { 3.0 } else { 0.0 });
        assert!((ac_dct_energy(&b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn edge_column_extraction_matches_raster() {
        let mut arr = [0.0f32; DCT_BLOCK_SIZE];
        for r in 0..8 {
            for c in 0..8 {
                arr[r * 8 + c] = (r * 10 + c) as f32;
            }
        }
        let left = left_edge_col(&arr);
        let right = right_edge_col(&arr);
        for r in 0..8 {
            assert_eq!(left[r], (r * 10) as f32);
            assert_eq!(right[r], (r * 10 + 7) as f32);
        }
    }
}
