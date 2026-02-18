//! Shared IDCT helpers for sequential and parallel output paths.
//!
//! These functions are used by both `output.rs` (sequential) and
//! `output_parallel.rs` (parallel) to avoid duplicating IDCT+dequant logic.

use crate::decode::idct_int::idct_int_dc_only;
use crate::foundation::consts::{DCT_BLOCK_SIZE, DCT_SIZE};
use crate::quant::dequantize_unzigzag_i32_into_partial;

use super::CompInfo;

/// IDCT one block into a strip buffer at the given offset.
///
/// Handles DC-only fast path (coeff_count <= 1) and general tiered IDCT.
/// The `idct_fn` parameter selects between standard and libjpeg-compat IDCT.
#[inline]
pub(super) fn idct_block_into(
    coeffs: &[i16; DCT_BLOCK_SIZE],
    coeff_count: u8,
    quant: &[u16; DCT_BLOCK_SIZE],
    strip: &mut [i16],
    dst_offset: usize,
    strip_width: usize,
    idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
    dequant_buf: &mut [i32; DCT_BLOCK_SIZE],
) {
    if coeff_count <= 1 {
        let dc = coeffs[0] as i32 * quant[0] as i32;
        idct_int_dc_only(dc, &mut strip[dst_offset..], strip_width);
    } else {
        dequantize_unzigzag_i32_into_partial(coeffs, quant, dequant_buf, coeff_count);
        idct_fn(
            dequant_buf,
            &mut strip[dst_offset..],
            strip_width,
            coeff_count,
        );
    }
}

/// IDCT all blocks for one component in one MCU row into a strip buffer.
///
/// Iterates over all block rows (v_samp) and columns (comp_blocks_h) in the
/// given MCU row, calling `idct_block_into` for each block.
pub(super) fn idct_comp_mcu_row(
    coeffs: &[[i16; DCT_BLOCK_SIZE]],
    coeff_counts: &[u8],
    info: &CompInfo,
    quant: &[u16; DCT_BLOCK_SIZE],
    imcu_row: usize,
    strip: &mut [i16],
    strip_width: usize,
    idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
    dequant_buf: &mut [i32; DCT_BLOCK_SIZE],
) {
    for iy in 0..info.v_samp {
        let by = imcu_row * info.v_samp + iy;
        if by >= info.comp_blocks_v {
            continue;
        }
        let strip_row = iy * DCT_SIZE;

        for bx in 0..info.comp_blocks_h {
            let block_idx = by * info.comp_blocks_h + bx;
            if block_idx >= coeffs.len() {
                continue;
            }
            let base_px = bx * DCT_SIZE;
            let dst_offset = strip_row * strip_width + base_px;

            idct_block_into(
                &coeffs[block_idx],
                coeff_counts[block_idx],
                quant,
                strip,
                dst_offset,
                strip_width,
                idct_fn,
                dequant_buf,
            );
        }
    }
}

/// IDCT chroma blocks for one MCU row into extended buffer rows 1..c_strip_height+1.
/// Then replicate the last valid row to fill any padding (partial MCU at image bottom).
///
/// The extended buffer layout is:
/// - Row 0: above context (filled by caller)
/// - Rows 1..c_strip_height: IDCT output data
/// - Row c_strip_height+1: below context (filled by caller)
pub(super) fn idct_chroma_into_ext(
    ext: &mut [i16],
    coeffs: &[[i16; DCT_BLOCK_SIZE]],
    coeff_counts: &[u8],
    info: &CompInfo,
    quant: &[u16; DCT_BLOCK_SIZE],
    imcu_row: usize,
    c_strip_width: usize,
    c_strip_height: usize,
    chroma_height_total: usize,
    idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
    dequant_buf: &mut [i32; DCT_BLOCK_SIZE],
) {
    let data_offset = c_strip_width; // skip context row 0

    for iy in 0..info.v_samp {
        let by = imcu_row * info.v_samp + iy;
        if by >= info.comp_blocks_v {
            continue;
        }
        let strip_row = iy * DCT_SIZE;

        for bx in 0..info.comp_blocks_h {
            let block_idx = by * info.comp_blocks_h + bx;
            if block_idx >= coeffs.len() {
                continue;
            }
            let base_px = bx * DCT_SIZE;
            let dst_offset = data_offset + strip_row * c_strip_width + base_px;

            idct_block_into(
                &coeffs[block_idx],
                coeff_counts[block_idx],
                quant,
                ext,
                dst_offset,
                c_strip_width,
                idct_fn,
                dequant_buf,
            );
        }
    }

    // Replicate last valid chroma row to fill padding rows
    let c_row_start = imcu_row * c_strip_height;
    let c_valid = chroma_height_total
        .saturating_sub(c_row_start)
        .min(c_strip_height);
    if c_valid > 0 && c_valid < c_strip_height {
        let last_valid_start = data_offset + (c_valid - 1) * c_strip_width;
        for pad_row in c_valid..c_strip_height {
            let pad_start = data_offset + pad_row * c_strip_width;
            ext.copy_within(
                last_valid_start..last_valid_start + c_strip_width,
                pad_start,
            );
        }
    }
}
