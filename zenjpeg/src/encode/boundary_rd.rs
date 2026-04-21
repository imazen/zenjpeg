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
#[cfg(target_arch = "x86_64")]
use crate::decode::idct::inverse_dct_8x8_into_with_token;
use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_NATURAL_ORDER};
use crate::foundation::simd_types::Block8x8f;

use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;

/// Pre-summoned SIMD tokens passed through the boundary-RD hot path so
/// that per-kernel `incant!` atomic loads aren't repeated. Callers get
/// this once at the top of `quantize_y_with_boundary_rd_impl` and pass
/// by copy.
///
/// On non-x86_64 builds this is a zero-sized struct — the kernels
/// dispatch via `incant!` which does the right thing.
#[derive(Copy, Clone, Default)]
pub(crate) struct BoundaryRdTokens {
    #[cfg(target_arch = "x86_64")]
    pub x64v3: Option<archmage::X64V3Token>,
}

impl BoundaryRdTokens {
    #[inline]
    pub(crate) fn summon() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            x64v3: archmage::X64V3Token::summon(),
        }
    }
}

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
///
/// Both the ×8 scaling and the IDCT use SIMD dispatch via archmage.
#[must_use]
pub(crate) fn idct_reference_block(dct_f32: &Block8x8f) -> [f32; DCT_BLOCK_SIZE] {
    let natural = incant!(mage_scale_block_x8(dct_f32));
    inverse_dct_8x8(&natural)
}

/// Cached-token variant of [`idct_reference_block`]. Also skips the
/// DC-only fast-path check in `inverse_dct_8x8` — reference blocks are
/// the IDCT of unquantized DCT coefficients from forward-DCT'd photos
/// or lineart, which almost never have all-zero AC energy in practice.
#[inline]
#[must_use]
pub(crate) fn idct_reference_block_fast(
    tokens: BoundaryRdTokens,
    dct_f32: &Block8x8f,
) -> [f32; DCT_BLOCK_SIZE] {
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = tokens.x64v3 {
        let natural = mage_scale_block_x8_v3(t, dct_f32);
        let mut out = [0.0f32; DCT_BLOCK_SIZE];
        inverse_dct_8x8_into_with_token(t, &natural, &mut out);
        return out;
    }
    let _ = tokens;
    let natural = incant!(mage_scale_block_x8(dct_f32));
    inverse_dct_8x8(&natural)
}

/// SIMD ×8 scaler: packs `Block8x8f` into a flat `[f32; 64]` scaled by 8.
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn mage_scale_block_x8(token: Token, dct_f32: &Block8x8f) -> [f32; DCT_BLOCK_SIZE] {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let scale = f32x8::splat(token, 8.0);
    let mut out = [0.0f32; DCT_BLOCK_SIZE];

    // Loop through rows; LLVM unrolls this at #[magetypes] expansion time.
    for row in 0..8 {
        let v = f32x8::from_array(token, dct_f32.rows[row]);
        let scaled = v * scale;
        let arr = scaled.to_array();
        let base = row * 8;
        out[base] = arr[0];
        out[base + 1] = arr[1];
        out[base + 2] = arr[2];
        out[base + 3] = arr[3];
        out[base + 4] = arr[4];
        out[base + 5] = arr[5];
        out[base + 6] = arr[6];
        out[base + 7] = arr[7];
    }
    out
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

/// Cached-token variant of [`idct_quantized_block`].
///
/// At quality levels typical for boundary-RD use (Q ≥ 50) the default
/// quantize virtually never produces a DC-only block — each iteration
/// of the refinement retry only tightens AQ, making AC zeros even more
/// likely. The DC-only check in the generic `inverse_dct_8x8` wrapper
/// is thus pure overhead in this hot path; this helper bypasses it.
#[inline]
#[must_use]
pub(crate) fn idct_quantized_block_fast(
    tokens: BoundaryRdTokens,
    zigzag_coeffs: &[i16; DCT_BLOCK_SIZE],
    quant_values_natural: &[u16; DCT_BLOCK_SIZE],
) -> [f32; DCT_BLOCK_SIZE] {
    let mut natural = [0.0f32; DCT_BLOCK_SIZE];
    for n in 0..DCT_BLOCK_SIZE {
        let zigzag_idx = crate::foundation::consts::JPEG_ZIGZAG_ORDER[n] as usize;
        natural[n] = zigzag_coeffs[zigzag_idx] as f32 * quant_values_natural[n] as f32;
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(t) = tokens.x64v3 {
        let mut out = [0.0f32; DCT_BLOCK_SIZE];
        inverse_dct_8x8_into_with_token(t, &natural, &mut out);
        return out;
    }
    let _ = tokens;
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

/// Extract the top-edge row (y = 0) of an 8×8 spatial block (Phase 4 of #91).
///
/// The top-edge row is the cross-seam pixel set between the current block
/// and the block directly above it. Used by the optional above-neighbor
/// boundary-continuity term.
#[inline]
#[must_use]
pub(crate) fn top_edge_row(block: &[f32; DCT_BLOCK_SIZE]) -> [f32; 8] {
    let mut row = [0.0f32; 8];
    row.copy_from_slice(&block[0..8]);
    row
}

/// Extract the bottom-edge row (y = 7) of an 8×8 spatial block (Phase 4 of #91).
///
/// Used to populate the cross-iMCU "above neighbor" buffer after each block
/// commits: the bottom-edge row of block (bx, by) becomes the "row above"
/// seen by block (bx, by+1) in a later iMCU.
#[inline]
#[must_use]
pub(crate) fn bottom_edge_row(block: &[f32; DCT_BLOCK_SIZE]) -> [f32; 8] {
    let mut row = [0.0f32; 8];
    row.copy_from_slice(&block[56..64]);
    row
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
///
/// Implementation: dispatches to a SIMD FMA chain (f32x8 load × 4, three
/// fused SSDs in parallel) via magetypes multi-tier dispatch on x86/NEON/WASM.
#[cfg(test)]
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
    incant!(mage_boundary_distortion(
        rec_curr_left,
        rec_lr,
        orig_curr_left,
        orig_lr,
        alpha,
    ))
}

/// Like [`boundary_distortion`] but without the Option-wrapping — the
/// caller asserts both neighbor edges exist. Used by the hot
/// refinement loop, which gates on `has_left_neighbor` / `has_above_neighbor`
/// and dispatches only when the edge pair is valid, eliminating the two
/// `Option` tag branches per call (~4 per block in the hot path).
#[inline]
#[must_use]
pub(crate) fn boundary_distortion_raw(
    rec_curr_left: &[f32; 8],
    rec_lr: &[f32; 8],
    orig_curr_left: &[f32; 8],
    orig_lr: &[f32; 8],
    alpha: f32,
) -> f32 {
    incant!(mage_boundary_distortion(
        rec_curr_left,
        rec_lr,
        orig_curr_left,
        orig_lr,
        alpha,
    ))
}

/// SIMD core: one f32x8 pass computes edge_current + edge_left + α·seam.
///
/// Layout of the computation (all ops are 8-wide):
///   d_cur  = rec_curr_left  - orig_curr_left      // len 8
///   d_left = rec_lr         - orig_lr             // len 8
///   jump_rec  = rec_curr_left  - rec_lr
///   jump_orig = orig_curr_left - orig_lr
///   d_seam = jump_rec - jump_orig
///          = (rec_curr_left - rec_lr) - (orig_curr_left - orig_lr)
///          = d_cur - d_left
///   D_b = Σ d_cur² + Σ d_left² + α · Σ d_seam²
#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn mage_boundary_distortion(
    token: Token,
    rec_curr_left: &[f32; 8],
    rec_lr: &[f32; 8],
    orig_curr_left: &[f32; 8],
    orig_lr: &[f32; 8],
    alpha: f32,
) -> f32 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let rec_cur = f32x8::from_array(token, *rec_curr_left);
    let rec_l = f32x8::from_array(token, *rec_lr);
    let orig_cur = f32x8::from_array(token, *orig_curr_left);
    let orig_l = f32x8::from_array(token, *orig_lr);

    let d_cur = rec_cur - orig_cur;
    let d_left = rec_l - orig_l;
    let d_seam = d_cur - d_left;

    // Fused square+accumulate: sum = d_cur² + d_left² + α·d_seam² (element-wise),
    // then horizontal sum.
    let alpha_v = f32x8::splat(token, alpha);
    let sq_cur = d_cur * d_cur;
    let sq_left = d_left * d_left;
    let sq_seam_a = d_seam.mul_add(alpha_v * d_seam, sq_cur + sq_left);
    sq_seam_a.reduce_add()
}

/// Per-block AC DCT energy. Sum of squared AC coefficients (DC excluded).
///
/// Used as the denominator of the D_b threshold check: refinement fires
/// when `D_b > threshold × ac_energy`. Blocks with very small AC energy
/// (flat regions) will never trigger refinement; blocks with large AC
/// energy (texture) will tolerate proportionally higher D_b before
/// refinement is deemed worthwhile.
///
/// Dispatches to a SIMD FMA sum-of-squares via magetypes.
#[inline]
#[must_use]
pub(crate) fn ac_dct_energy(dct_f32: &Block8x8f) -> f32 {
    incant!(mage_ac_dct_energy(dct_f32))
}

#[magetypes(v3, neon, wasm128, scalar)]
#[inline(always)]
fn mage_ac_dct_energy(token: Token, dct_f32: &Block8x8f) -> f32 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    // Row 0: zero out the DC (position [0][0]) then sum-of-squares.
    let mut row0 = dct_f32.rows[0];
    row0[0] = 0.0;
    let v0 = f32x8::from_array(token, row0);
    let mut acc = v0 * v0;

    // Rows 1..8: full f32x8 square-and-accumulate.
    let v1 = f32x8::from_array(token, dct_f32.rows[1]);
    acc = v1.mul_add(v1, acc);
    let v2 = f32x8::from_array(token, dct_f32.rows[2]);
    acc = v2.mul_add(v2, acc);
    let v3 = f32x8::from_array(token, dct_f32.rows[3]);
    acc = v3.mul_add(v3, acc);
    let v4 = f32x8::from_array(token, dct_f32.rows[4]);
    acc = v4.mul_add(v4, acc);
    let v5 = f32x8::from_array(token, dct_f32.rows[5]);
    acc = v5.mul_add(v5, acc);
    let v6 = f32x8::from_array(token, dct_f32.rows[6]);
    acc = v6.mul_add(v6, acc);
    let v7 = f32x8::from_array(token, dct_f32.rows[7]);
    acc = v7.mul_add(v7, acc);

    acc.reduce_add()
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
        let d = boundary_distortion(
            &left_col,
            Some(&right_col),
            &left_col,
            Some(&right_col),
            1.0,
        );
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

    #[test]
    fn edge_row_extraction_matches_raster() {
        let mut arr = [0.0f32; DCT_BLOCK_SIZE];
        for r in 0..8 {
            for c in 0..8 {
                arr[r * 8 + c] = (r * 10 + c) as f32;
            }
        }
        let top = top_edge_row(&arr);
        let bottom = bottom_edge_row(&arr);
        for c in 0..8 {
            assert_eq!(top[c], c as f32);
            assert_eq!(bottom[c], (70 + c) as f32);
        }
    }

    // --- SIMD parity checks ----------------------------------------------
    //
    // Scalar references of the three SIMD kernels, used only in tests to
    // guard against divergence under any of the magetypes dispatch targets
    // (x86-v3, NEON, wasm128, scalar).

    fn scalar_boundary_distortion(
        rec_curr_left: &[f32; 8],
        rec_lr: &[f32; 8],
        orig_curr_left: &[f32; 8],
        orig_lr: &[f32; 8],
        alpha: f32,
    ) -> f32 {
        let mut sum = 0.0f32;
        for r in 0..8 {
            let d_cur = rec_curr_left[r] - orig_curr_left[r];
            let d_left = rec_lr[r] - orig_lr[r];
            let d_seam = d_cur - d_left;
            sum += d_cur * d_cur + d_left * d_left + alpha * d_seam * d_seam;
        }
        sum
    }

    fn scalar_ac_dct_energy(dct_f32: &Block8x8f) -> f32 {
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

    #[test]
    fn simd_boundary_distortion_matches_scalar() {
        // A non-trivial pattern that exercises the full f32x8 load, three
        // subtractions, three squared terms, and the FMA accumulate.
        let rec_cur: [f32; 8] = [1.0, -3.0, 10.5, -7.5, 42.0, 0.0, -15.25, 8.0];
        let rec_lr: [f32; 8] = [0.5, -2.5, 11.0, -8.0, 40.0, 1.0, -16.0, 7.0];
        let orig_cur: [f32; 8] = [2.0, -4.0, 9.0, -6.5, 45.5, -1.0, -14.0, 6.5];
        let orig_lr: [f32; 8] = [1.5, -3.0, 10.0, -9.0, 38.75, 2.5, -17.5, 9.125];

        for &alpha in &[0.0f32, 0.5, 1.0, 2.0, 4.0] {
            let expected =
                scalar_boundary_distortion(&rec_cur, &rec_lr, &orig_cur, &orig_lr, alpha);
            let got =
                boundary_distortion(&rec_cur, Some(&rec_lr), &orig_cur, Some(&orig_lr), alpha);
            // f32 FMA reorders additions vs scalar `+=`, so accept a small
            // relative tolerance. Scale tolerance with |expected| for large
            // magnitudes.
            let tol = 1e-5f32 * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() <= tol,
                "alpha={alpha}: expected {expected}, got {got}, diff {}",
                (got - expected).abs()
            );
        }
    }

    #[test]
    fn simd_ac_dct_energy_matches_scalar() {
        // Fill the block with a deterministic non-trivial pattern.
        let b = make_block(|r, c| ((r * 13 + c * 7) as i32 - 32) as f32 * 0.5);
        let expected = scalar_ac_dct_energy(&b);
        let got = ac_dct_energy(&b);
        let tol = 1e-4f32 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tol,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn simd_ac_dct_energy_dc_only_is_zero() {
        // DC coefficient huge, AC all zero — SIMD path must zero out DC
        // explicitly and return 0.
        let b = make_block(|r, c| if r == 0 && c == 0 { 12345.0 } else { 0.0 });
        assert_eq!(ac_dct_energy(&b), 0.0);
    }
}
