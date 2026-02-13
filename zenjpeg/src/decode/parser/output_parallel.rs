//! Parallel output pass for decoded JPEG data.
//!
//! Parallelizes IDCT + color conversion across MCU rows using rayon.
//! Behind `#[cfg(feature = "parallel")]`.
//!
//! ## Paths
//!
//! - `to_pixels_fast_i16_parallel`: 4:4:4 non-XYB images
//! - `to_pixels_fast_i16_subsampled_parallel`: 4:2:0/4:2:2/4:4:0 non-XYB images

use crate::color::ycbcr::{fused_h2v2_box_ycbcr_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8};
use crate::decode::idct_int::{idct_int_dc_only, idct_int_tiered, idct_int_tiered_libjpeg};
use crate::decode::parser::CompInfo;
use crate::decode::upsample::{
    upsample_h1v2_i16_fancy, upsample_h1v2_i16_libjpeg, upsample_h1v2_i16_nearest,
    upsample_h2v1_i16_fancy, upsample_h2v1_i16_libjpeg, upsample_h2v1_i16_nearest,
    upsample_h2v2_i16_fancy, upsample_h2v2_i16_libjpeg, upsample_h2v2_i16_nearest,
};
use crate::decode::ChromaUpsampling;
use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit};
use crate::foundation::consts::{DCT_BLOCK_SIZE, DCT_SIZE};
use crate::quant::dequantize_unzigzag_i32_partial;
use rayon::prelude::*;

use crate::decode::parser::JpegParser;

/// Minimum MCU rows to justify parallel overhead.
const MIN_MCU_ROWS_PARALLEL: usize = 8;

/// Minimum pixel count (width × height) to justify parallel output.
/// Below this, the overhead of thread-local allocations and rayon scheduling
/// exceeds the parallelism benefit. Empirically determined: 2K (4.2M px) is
/// slower with parallel output, 4K (8.8M px) breaks even.
const MIN_PIXELS_PARALLEL: usize = 8_000_000;

/// IDCT one block into a strip buffer at the given offset.
#[inline]
fn idct_block_into(
    coeffs: &[i16; DCT_BLOCK_SIZE],
    coeff_count: u8,
    quant: &[u16; DCT_BLOCK_SIZE],
    strip: &mut [i16],
    dst_offset: usize,
    strip_width: usize,
    idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
) {
    if coeff_count <= 1 {
        let dc = coeffs[0] as i32 * quant[0] as i32;
        idct_int_dc_only(dc, &mut strip[dst_offset..], strip_width);
    } else {
        let mut dequant_i32 = dequantize_unzigzag_i32_partial(coeffs, quant, coeff_count);
        idct_fn(
            &mut dequant_i32,
            &mut strip[dst_offset..],
            strip_width,
            coeff_count,
        );
    }
}

/// IDCT all blocks for one component in one MCU row into a strip buffer.
fn idct_comp_mcu_row(
    coeffs: &[[i16; DCT_BLOCK_SIZE]],
    coeff_counts: &[u8],
    info: &CompInfo,
    quant: &[u16; DCT_BLOCK_SIZE],
    imcu_row: usize,
    strip: &mut [i16],
    strip_width: usize,
    idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
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
            );
        }
    }
}

/// Parallel output methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Parallel 4:4:4 decode: IDCT + color convert, split by MCU row.
    ///
    /// Returns `None` if the image is too small for parallelism to help,
    /// falling through to the sequential path.
    pub(super) fn to_pixels_fast_i16_parallel(
        &self,
        chroma_upsampling: ChromaUpsampling,
    ) -> Result<Option<Vec<u8>>> {
        let width = self.width as usize;
        let height = self.height as usize;

        let max_h_samp = self.components[0].h_samp_factor as usize;
        let max_v_samp = self.components[0].v_samp_factor as usize;

        let mcu_height = max_v_samp * 8;
        let mcu_cols = (width + max_h_samp * 8 - 1) / (max_h_samp * 8);
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        if mcu_rows < MIN_MCU_ROWS_PARALLEL || width * height < MIN_PIXELS_PARALLEL {
            return Ok(None);
        }

        let comp_infos = self.build_comp_infos(mcu_cols, mcu_rows, max_h_samp, max_v_samp, 3)?;

        let strip_height = mcu_height;
        let strip_width = comp_infos[0].comp_width;

        // Pre-fetch quant tables
        let quant_tables: [&[u16; DCT_BLOCK_SIZE]; 3] = [
            self.quant_tables[comp_infos[0].quant_idx]
                .as_ref()
                .ok_or_else(|| Error::internal("missing Y quant table"))?,
            self.quant_tables[comp_infos[1].quant_idx]
                .as_ref()
                .ok_or_else(|| Error::internal("missing Cb quant table"))?,
            self.quant_tables[comp_infos[2].quant_idx]
                .as_ref()
                .ok_or_else(|| Error::internal("missing Cr quant table"))?,
        ];

        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        let is_rgb = self.is_rgb_jpeg();
        let rgb_row_stride = width * 3;
        let mcu_row_rgb_bytes = mcu_height * rgb_row_stride;

        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => idct_int_tiered_libjpeg,
            _ => idct_int_tiered,
        };

        // Extract references to coefficient data (Sync-safe) before parallel section.
        // JpegParser contains OnceCell (not Sync), so we can't capture &self in rayon closures.
        let coeffs = &self.coeffs;
        let coeff_counts = &self.coeff_counts;

        rgb.par_chunks_mut(mcu_row_rgb_bytes)
            .enumerate()
            .for_each(|(imcu_row, rgb_chunk)| {
                // Thread-local strip buffers
                let strip_size = strip_width * strip_height;
                let mut y_strip = vec![0i16; strip_size];
                let mut cb_strip = vec![0i16; strip_size];
                let mut cr_strip = vec![0i16; strip_size];

                // IDCT all blocks in this MCU row for all 3 components
                let strips: [&mut Vec<i16>; 3] = [&mut y_strip, &mut cb_strip, &mut cr_strip];
                for (comp_idx, strip) in strips.into_iter().enumerate() {
                    idct_comp_mcu_row(
                        &coeffs[comp_idx],
                        &coeff_counts[comp_idx],
                        &comp_infos[comp_idx],
                        quant_tables[comp_idx],
                        imcu_row,
                        strip,
                        strip_width,
                        idct_fn,
                    );
                }

                // Color convert this MCU row
                let y_start = imcu_row * mcu_height;
                let rows_this_mcu = mcu_height.min(height.saturating_sub(y_start));
                let cols_this_mcu = width.min(strip_width);

                for row in 0..rows_this_mcu {
                    let strip_offset = row * strip_width;
                    let rgb_offset = row * rgb_row_stride;

                    if is_rgb {
                        for px in 0..cols_this_mcu {
                            let i = strip_offset + px;
                            let o = rgb_offset + px * 3;
                            rgb_chunk[o] = y_strip[i].clamp(0, 255) as u8;
                            rgb_chunk[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                            rgb_chunk[o + 2] = cr_strip[i].clamp(0, 255) as u8;
                        }
                    } else {
                        ycbcr_planes_i16_to_rgb_u8(
                            &y_strip[strip_offset..strip_offset + cols_this_mcu],
                            &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                            &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                            &mut rgb_chunk[rgb_offset..rgb_offset + cols_this_mcu * 3],
                        );
                    }
                }
            });

        Ok(Some(rgb))
    }

    /// Parallel 4:2:0/4:2:2/4:4:0 decode: two-phase approach.
    ///
    /// Phase 1: Pre-IDCT all chroma into full plane buffers (parallel over block-rows).
    /// Phase 2: Parallel MCU rows for Y IDCT + upsample + color convert.
    ///
    /// Returns `None` if the image is too small for parallelism to help.
    pub(super) fn to_pixels_fast_i16_subsampled_parallel(
        &self,
        chroma_upsampling: ChromaUpsampling,
    ) -> Result<Option<Vec<u8>>> {
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => idct_int_tiered_libjpeg,
            _ => idct_int_tiered,
        };

        let width = self.width as usize;
        let height = self.height as usize;

        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let c_h = self.components[1].h_samp_factor as usize;
        let c_v = self.components[1].v_samp_factor as usize;

        let h_ratio = y_h / c_h;
        let v_ratio = y_v / c_v;

        let mcu_width = y_h * 8;
        let mcu_height = y_v * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        if mcu_rows < MIN_MCU_ROWS_PARALLEL || width * height < MIN_PIXELS_PARALLEL {
            return Ok(None);
        }

        let comp_infos = self.build_comp_infos(mcu_cols, mcu_rows, y_h, y_v, 3)?;

        let y_strip_height = y_v * 8;
        let y_strip_width = comp_infos[0].comp_width;

        let c_strip_height = c_v * 8;
        let c_strip_width = comp_infos[1].comp_width;

        // Pre-fetch quant tables
        let quant_y = self.quant_tables[comp_infos[0].quant_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Y quant table"))?;
        let quant_cb = self.quant_tables[comp_infos[1].quant_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Cb quant table"))?;
        let quant_cr = self.quant_tables[comp_infos[2].quant_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Cr quant table"))?;

        let y_cols_this_image = width.min(y_strip_width);

        // Select upsampling function
        type UpsampleFn = fn(&[i16], usize, usize, &mut [i16], usize, usize);
        let needs_full_upsample = !matches!(chroma_upsampling, ChromaUpsampling::NearestNeighbor)
            || h_ratio != 2
            || v_ratio != 2;

        let upsample_fn: UpsampleFn = if needs_full_upsample {
            let (upsample_h2v2, upsample_h2v1, upsample_h1v2): (
                UpsampleFn,
                UpsampleFn,
                UpsampleFn,
            ) = match chroma_upsampling {
                ChromaUpsampling::Triangle => (
                    upsample_h2v2_i16_fancy,
                    upsample_h2v1_i16_fancy,
                    upsample_h1v2_i16_fancy,
                ),
                ChromaUpsampling::LibjpegCompat => (
                    upsample_h2v2_i16_libjpeg,
                    upsample_h2v1_i16_libjpeg,
                    upsample_h1v2_i16_libjpeg,
                ),
                ChromaUpsampling::NearestNeighbor => (
                    upsample_h2v2_i16_nearest,
                    upsample_h2v1_i16_nearest,
                    upsample_h1v2_i16_nearest,
                ),
            };

            match (h_ratio, v_ratio) {
                (2, 2) => upsample_h2v2,
                (2, 1) => upsample_h2v1,
                (1, 2) => upsample_h1v2,
                _ => unreachable!(
                    "unsupported ratio should be filtered by can_use_fast_i16_subsampled"
                ),
            }
        } else {
            upsample_h2v2_i16_nearest // placeholder for fused path
        };

        // ===================================================================
        // Phase 1: Pre-IDCT all chroma into full plane buffers (parallel)
        //
        // This trades memory (~32MB for 8K 4:2:0) for parallelizability.
        // The sequential path uses ~200KB of double-buffered strips.
        // ===================================================================
        let chroma_height = comp_infos[1].comp_height;
        let cb_plane_size = checked_size_2d(c_strip_width, chroma_height)?;
        let cr_plane_size = cb_plane_size;

        let mut cb_plane: Vec<i16> = try_alloc_maybeuninit(cb_plane_size, "Cb plane")?;
        let mut cr_plane: Vec<i16> = try_alloc_maybeuninit(cr_plane_size, "Cr plane")?;

        let chroma_block_rows = comp_infos[1].comp_blocks_v;
        let chroma_blocks_h = comp_infos[1].comp_blocks_h;

        // Extract coefficient references (Sync-safe) before parallel section
        let coeffs = &self.coeffs;
        let coeff_counts = &self.coeff_counts;

        // IDCT Cb plane — parallel over block rows
        cb_plane
            .par_chunks_mut(DCT_SIZE * c_strip_width)
            .enumerate()
            .take(chroma_block_rows)
            .for_each(|(by, row_chunk)| {
                for bx in 0..chroma_blocks_h {
                    let block_idx = by * chroma_blocks_h + bx;
                    if block_idx >= coeffs[1].len() {
                        continue;
                    }
                    let base_px = bx * DCT_SIZE;
                    idct_block_into(
                        &coeffs[1][block_idx],
                        coeff_counts[1][block_idx],
                        quant_cb,
                        row_chunk,
                        base_px,
                        c_strip_width,
                        idct_fn,
                    );
                }
            });

        // IDCT Cr plane — parallel over block rows
        cr_plane
            .par_chunks_mut(DCT_SIZE * c_strip_width)
            .enumerate()
            .take(chroma_block_rows)
            .for_each(|(by, row_chunk)| {
                for bx in 0..chroma_blocks_h {
                    let block_idx = by * chroma_blocks_h + bx;
                    if block_idx >= coeffs[2].len() {
                        continue;
                    }
                    let base_px = bx * DCT_SIZE;
                    idct_block_into(
                        &coeffs[2][block_idx],
                        coeff_counts[2][block_idx],
                        quant_cr,
                        row_chunk,
                        base_px,
                        c_strip_width,
                        idct_fn,
                    );
                }
            });

        // ===================================================================
        // Phase 2: Parallel MCU rows — Y IDCT + upsample + color convert
        //
        // Each thread reads from the shared chroma planes (immutable after phase 1)
        // and writes to a disjoint region of the RGB output buffer.
        // ===================================================================
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        let rgb_row_stride = width * 3;
        let mcu_row_rgb_bytes = mcu_height * rgb_row_stride;

        let chroma_height_total = (height + v_ratio - 1) / v_ratio;

        // Borrow chroma planes as shared slices for the parallel section
        let cb_plane = &cb_plane[..];
        let cr_plane = &cr_plane[..];
        let info_y = &comp_infos[0];

        rgb.par_chunks_mut(mcu_row_rgb_bytes)
            .enumerate()
            .for_each(|(imcu_row, rgb_chunk)| {
                // Thread-local Y strip buffer
                let y_strip_size = y_strip_width * y_strip_height;
                let mut y_strip = vec![0i16; y_strip_size];

                // IDCT Y blocks for this MCU row
                idct_comp_mcu_row(
                    &coeffs[0],
                    &coeff_counts[0],
                    info_y,
                    quant_y,
                    imcu_row,
                    &mut y_strip,
                    y_strip_width,
                    idct_fn,
                );

                let y_rows_this_mcu =
                    y_strip_height.min(height.saturating_sub(imcu_row * mcu_height));

                if !needs_full_upsample {
                    // NearestNeighbor 4:2:0: fused box-filter path
                    let c_cols = (y_cols_this_image + 1) / 2;
                    let c_plane_row_start = imcu_row * c_strip_height;

                    for row in 0..y_rows_this_mcu {
                        let y_offset = row * y_strip_width;
                        let c_row =
                            (c_plane_row_start + row / 2).min(chroma_height_total.saturating_sub(1));
                        let c_offset = c_row * c_strip_width;
                        let rgb_offset = row * rgb_row_stride;

                        fused_h2v2_box_ycbcr_to_rgb_u8(
                            &y_strip[y_offset..y_offset + y_cols_this_image],
                            &cb_plane[c_offset..c_offset + c_cols],
                            &cr_plane[c_offset..c_offset + c_cols],
                            &mut rgb_chunk[rgb_offset..rgb_offset + y_cols_this_image * 3],
                            y_cols_this_image,
                        );
                    }
                } else {
                    // Build extended chroma buffer with ±1 row context for the upsampler
                    let ext_height = c_strip_height + 2;
                    let ext_size = ext_height * c_strip_width;

                    let mut ext_cb = vec![0i16; ext_size];
                    let mut ext_cr = vec![0i16; ext_size];

                    let c_row_start = imcu_row * c_strip_height;
                    let c_valid = chroma_height_total
                        .saturating_sub(c_row_start)
                        .min(c_strip_height);

                    // Copy data rows (rows 1..c_strip_height+1 in ext buffer)
                    for iy in 0..c_valid {
                        let src_row = c_row_start + iy;
                        if src_row < chroma_height_total {
                            let src_start = src_row * c_strip_width;
                            let src_end = src_start + c_strip_width;
                            let dst_start = (iy + 1) * c_strip_width;
                            ext_cb[dst_start..dst_start + c_strip_width]
                                .copy_from_slice(&cb_plane[src_start..src_end]);
                            ext_cr[dst_start..dst_start + c_strip_width]
                                .copy_from_slice(&cr_plane[src_start..src_end]);
                        }
                    }

                    // Replicate last valid row to fill padding
                    if c_valid > 0 && c_valid < c_strip_height {
                        let last_valid_start = c_valid * c_strip_width;
                        for pad_row in c_valid..c_strip_height {
                            let pad_start = (pad_row + 1) * c_strip_width;
                            ext_cb.copy_within(
                                last_valid_start..last_valid_start + c_strip_width,
                                pad_start,
                            );
                            ext_cr.copy_within(
                                last_valid_start..last_valid_start + c_strip_width,
                                pad_start,
                            );
                        }
                    }

                    // Above context row (row 0)
                    if c_row_start > 0 {
                        let src_row = c_row_start - 1;
                        let src_start = src_row * c_strip_width;
                        ext_cb[..c_strip_width]
                            .copy_from_slice(&cb_plane[src_start..src_start + c_strip_width]);
                        ext_cr[..c_strip_width]
                            .copy_from_slice(&cr_plane[src_start..src_start + c_strip_width]);
                    } else {
                        ext_cb.copy_within(c_strip_width..2 * c_strip_width, 0);
                        ext_cr.copy_within(c_strip_width..2 * c_strip_width, 0);
                    }

                    // Below context row (row c_strip_height+1)
                    let below_ctx_start = (c_strip_height + 1) * c_strip_width;
                    let next_row = c_row_start + c_strip_height;
                    if next_row < chroma_height_total {
                        let src_start = next_row * c_strip_width;
                        ext_cb[below_ctx_start..below_ctx_start + c_strip_width]
                            .copy_from_slice(&cb_plane[src_start..src_start + c_strip_width]);
                        ext_cr[below_ctx_start..below_ctx_start + c_strip_width]
                            .copy_from_slice(&cr_plane[src_start..src_start + c_strip_width]);
                    } else {
                        let last_data_start = c_strip_height * c_strip_width;
                        ext_cb.copy_within(
                            last_data_start..last_data_start + c_strip_width,
                            below_ctx_start,
                        );
                        ext_cr.copy_within(
                            last_data_start..last_data_start + c_strip_width,
                            below_ctx_start,
                        );
                    }

                    // Upsample extended strip
                    let upsample_out_height = ext_height * v_ratio;
                    let upsample_out_size = upsample_out_height * y_strip_width;
                    let mut cb_up = vec![0i16; upsample_out_size];
                    let mut cr_up = vec![0i16; upsample_out_size];

                    upsample_fn(
                        &ext_cb,
                        c_strip_width,
                        ext_height,
                        &mut cb_up,
                        y_strip_width,
                        upsample_out_height,
                    );
                    upsample_fn(
                        &ext_cr,
                        c_strip_width,
                        ext_height,
                        &mut cr_up,
                        y_strip_width,
                        upsample_out_height,
                    );

                    // Color convert: use upsampled rows starting at offset v_ratio
                    for row in 0..y_rows_this_mcu {
                        let strip_offset = row * y_strip_width;
                        let up_row = v_ratio + row;
                        let chroma_offset = up_row * y_strip_width;
                        let rgb_offset = row * rgb_row_stride;

                        ycbcr_planes_i16_to_rgb_u8(
                            &y_strip[strip_offset..strip_offset + y_cols_this_image],
                            &cb_up[chroma_offset..chroma_offset + y_cols_this_image],
                            &cr_up[chroma_offset..chroma_offset + y_cols_this_image],
                            &mut rgb_chunk[rgb_offset..rgb_offset + y_cols_this_image * 3],
                        );
                    }
                }
            });

        Ok(Some(rgb))
    }
}
