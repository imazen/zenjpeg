//! Parallel output pass for decoded JPEG data.
//!
//! Parallelizes IDCT + color conversion across MCU rows using rayon.
//! Behind `#[cfg(feature = "parallel")]`.
//!
//! Each thread processes a contiguous batch of MCU rows with reused strip
//! buffers, matching the sequential path's allocation pattern. No full-image
//! chroma plane allocations — the 4:2:0 path uses double-buffered extended
//! strips per thread, identical to the sequential approach.
//!
//! ## Paths
//!
//! - `to_pixels_fast_i16_parallel`: 4:4:4 non-XYB images
//! - `to_pixels_fast_i16_subsampled_parallel`: 4:2:0/4:2:2/4:4:0 non-XYB images

use crate::color::ycbcr::{fused_h2v2_box_ycbcr_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8};
use crate::decode::idct_int::{idct_int_tiered, idct_int_tiered_libjpeg};
use crate::decode::upsample::{
    upsample_h1v2_i16_libjpeg, upsample_h1v2_i16_nearest, upsample_h2v1_i16_libjpeg,
    upsample_h2v1_i16_nearest, upsample_h2v2_i16_libjpeg, upsample_h2v2_i16_nearest,
};
use crate::decode::{ChromaUpsampling, IdctMethod};
use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit};
use crate::foundation::consts::DCT_BLOCK_SIZE;
use rayon::prelude::*;

use super::output_helpers::{idct_chroma_into_ext, idct_comp_mcu_row};
use crate::decode::parser::JpegParser;

/// Minimum MCU rows to justify parallel overhead.
const MIN_MCU_ROWS_PARALLEL: usize = 8;

/// Minimum pixel count (width × height) to justify parallel output.
/// 2M disables parallelism at 1024×1024 (1M) where rayon overhead exceeds
/// the benefit, while keeping it for 2048×2048 (4.2M) and above.
const MIN_PIXELS_PARALLEL: usize = 2_000_000;

/// Parallel output methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Parallel 4:4:4 decode: IDCT + color convert, batched by thread.
    ///
    /// Each thread processes a contiguous range of MCU rows with reused strip
    /// buffers (3 allocs per thread, not per MCU row).
    ///
    /// Returns `None` if the image is too small for parallelism to help.
    pub(super) fn to_pixels_fast_i16_parallel(
        &self,
        _chroma_upsampling: ChromaUpsampling,
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
        let strip_size = strip_width * strip_height;

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

        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match self.idct_method {
            IdctMethod::Libjpeg => idct_int_tiered_libjpeg,
            IdctMethod::Jpegli => idct_int_tiered,
        };

        // Extract Sync-safe references before parallel section
        let coeffs = &self.coeffs;
        let coeff_counts = &self.coeff_counts;

        // Batch MCU rows into ~num_threads chunks so each thread allocates once
        let num_threads = rayon::current_num_threads();
        let batch_size = (mcu_rows + num_threads - 1) / num_threads;
        let batch_rgb_bytes = batch_size * mcu_row_rgb_bytes;

        rgb.par_chunks_mut(batch_rgb_bytes)
            .enumerate()
            .for_each(|(batch_idx, rgb_batch)| {
                // Allocate strip buffers ONCE per thread — reused across MCU rows
                let mut y_strip = vec![0i16; strip_size];
                let mut cb_strip = vec![0i16; strip_size];
                let mut cr_strip = vec![0i16; strip_size];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];

                let start_row = batch_idx * batch_size;
                let end_row = (start_row + batch_size).min(mcu_rows);

                for imcu_row in start_row..end_row {
                    // IDCT all blocks in this MCU row for all 3 components
                    for (comp_idx, strip) in [&mut y_strip, &mut cb_strip, &mut cr_strip]
                        .into_iter()
                        .enumerate()
                    {
                        idct_comp_mcu_row(
                            &coeffs[comp_idx],
                            &coeff_counts[comp_idx],
                            &comp_infos[comp_idx],
                            quant_tables[comp_idx],
                            imcu_row,
                            strip,
                            strip_width,
                            idct_fn,
                            &mut dequant_buf,
                        );
                    }

                    // Color convert this MCU row
                    let y_start = imcu_row * mcu_height;
                    let rows_this_mcu = mcu_height.min(height.saturating_sub(y_start));
                    let cols_this_mcu = width.min(strip_width);
                    let local_offset = (imcu_row - start_row) * mcu_row_rgb_bytes;

                    for row in 0..rows_this_mcu {
                        let strip_offset = row * strip_width;
                        let rgb_offset = local_offset + row * rgb_row_stride;

                        if is_rgb {
                            for px in 0..cols_this_mcu {
                                let i = strip_offset + px;
                                let o = rgb_offset + px * 3;
                                rgb_batch[o] = y_strip[i].clamp(0, 255) as u8;
                                rgb_batch[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                                rgb_batch[o + 2] = cr_strip[i].clamp(0, 255) as u8;
                            }
                        } else {
                            ycbcr_planes_i16_to_rgb_u8(
                                &y_strip[strip_offset..strip_offset + cols_this_mcu],
                                &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                                &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                                &mut rgb_batch[rgb_offset..rgb_offset + cols_this_mcu * 3],
                            );
                        }
                    }
                }
            });

        Ok(Some(rgb))
    }

    /// Parallel 4:2:0/4:2:2/4:4:0 decode: batched by thread.
    ///
    /// Each thread processes a contiguous range of MCU rows using double-buffered
    /// extended chroma strips, identical to the sequential approach. No full-image
    /// chroma plane allocation — each thread uses ~200KB of strip buffers.
    ///
    /// Returns `None` if the image is too small for parallelism to help.
    pub(super) fn to_pixels_fast_i16_subsampled_parallel(
        &self,
        chroma_upsampling: ChromaUpsampling,
    ) -> Result<Option<Vec<u8>>> {
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match self.idct_method {
            IdctMethod::Libjpeg => idct_int_tiered_libjpeg,
            IdctMethod::Jpegli => idct_int_tiered,
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
        let y_strip_size = y_strip_width * y_strip_height;

        let c_strip_height = c_v * 8;
        let c_strip_width = comp_infos[1].comp_width;

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

        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        let rgb_row_stride = width * 3;
        let mcu_row_rgb_bytes = mcu_height * rgb_row_stride;

        let chroma_height_total = (height + v_ratio - 1) / v_ratio;

        // Extract Sync-safe references before parallel section
        let coeffs = &self.coeffs;
        let coeff_counts = &self.coeff_counts;
        let info_y = &comp_infos[0];
        let info_cb = &comp_infos[1];
        let info_cr = &comp_infos[2];

        // Batch MCU rows into ~num_threads chunks
        let num_threads = rayon::current_num_threads();
        let batch_size = (mcu_rows + num_threads - 1) / num_threads;
        let batch_rgb_bytes = batch_size * mcu_row_rgb_bytes;

        rgb.par_chunks_mut(batch_rgb_bytes)
            .enumerate()
            .for_each(|(batch_idx, rgb_batch)| {
                let start_row = batch_idx * batch_size;
                let end_row = (start_row + batch_size).min(mcu_rows);
                let batch_mcu_rows = end_row - start_row;

                // Allocate per-thread buffers ONCE — reused across MCU rows
                let ext_height = c_strip_height + 2;
                let ext_size = ext_height * c_strip_width;

                let mut y_strip = vec![0i16; y_strip_size];
                let mut ext_cb_a = vec![0i16; ext_size];
                let mut ext_cb_b = vec![0i16; ext_size];
                let mut ext_cr_a = vec![0i16; ext_size];
                let mut ext_cr_b = vec![0i16; ext_size];

                let (mut cb_up, mut cr_up) = if needs_full_upsample {
                    let upsample_out_height = ext_height * v_ratio;
                    let upsample_out_size = upsample_out_height * y_strip_width;
                    (vec![0i16; upsample_out_size], vec![0i16; upsample_out_size])
                } else {
                    (Vec::new(), Vec::new())
                };
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];

                // IDCT first chroma strip into ext_a
                idct_chroma_into_ext(
                    &mut ext_cb_a,
                    &coeffs[1],
                    &coeff_counts[1],
                    info_cb,
                    quant_cb,
                    start_row,
                    c_strip_width,
                    c_strip_height,
                    chroma_height_total,
                    idct_fn,
                    &mut dequant_buf,
                );
                idct_chroma_into_ext(
                    &mut ext_cr_a,
                    &coeffs[2],
                    &coeff_counts[2],
                    info_cr,
                    quant_cr,
                    start_row,
                    c_strip_width,
                    c_strip_height,
                    chroma_height_total,
                    idct_fn,
                    &mut dequant_buf,
                );

                // Set above context for first strip in this batch.
                // Do this BEFORE populating ext_b so we can use ext_b as scratch.
                if start_row == 0 {
                    // First MCU row: edge replication
                    ext_cb_a.copy_within(c_strip_width..2 * c_strip_width, 0);
                    ext_cr_a.copy_within(c_strip_width..2 * c_strip_width, 0);
                } else {
                    // IDCT the MCU row before our range into ext_b (scratch),
                    // copy its last data row into ext_a's above-context row.
                    idct_chroma_into_ext(
                        &mut ext_cb_b,
                        &coeffs[1],
                        &coeff_counts[1],
                        info_cb,
                        quant_cb,
                        start_row - 1,
                        c_strip_width,
                        c_strip_height,
                        chroma_height_total,
                        idct_fn,
                        &mut dequant_buf,
                    );
                    idct_chroma_into_ext(
                        &mut ext_cr_b,
                        &coeffs[2],
                        &coeff_counts[2],
                        info_cr,
                        quant_cr,
                        start_row - 1,
                        c_strip_width,
                        c_strip_height,
                        chroma_height_total,
                        idct_fn,
                        &mut dequant_buf,
                    );
                    let last_data = c_strip_height * c_strip_width;
                    ext_cb_a[..c_strip_width]
                        .copy_from_slice(&ext_cb_b[last_data..last_data + c_strip_width]);
                    ext_cr_a[..c_strip_width]
                        .copy_from_slice(&ext_cr_b[last_data..last_data + c_strip_width]);
                }

                // IDCT second chroma strip into ext_b (if batch has >1 row)
                if batch_mcu_rows > 1 && start_row + 1 < mcu_rows {
                    idct_chroma_into_ext(
                        &mut ext_cb_b,
                        &coeffs[1],
                        &coeff_counts[1],
                        info_cb,
                        quant_cb,
                        start_row + 1,
                        c_strip_width,
                        c_strip_height,
                        chroma_height_total,
                        idct_fn,
                        &mut dequant_buf,
                    );
                    idct_chroma_into_ext(
                        &mut ext_cr_b,
                        &coeffs[2],
                        &coeff_counts[2],
                        info_cr,
                        quant_cr,
                        start_row + 1,
                        c_strip_width,
                        c_strip_height,
                        chroma_height_total,
                        idct_fn,
                        &mut dequant_buf,
                    );
                }

                // Process each MCU row in this batch
                for imcu_row in start_row..end_row {
                    let local_idx = imcu_row - start_row;

                    // Set below context for current strip (ext_a)
                    let last_data_row_start = c_strip_height * c_strip_width;
                    let below_ctx_start = (c_strip_height + 1) * c_strip_width;
                    if imcu_row < mcu_rows - 1 {
                        // Below context = first data row of next strip (ext_b row 1)
                        let src_start = c_strip_width;
                        ext_cb_a[below_ctx_start..below_ctx_start + c_strip_width]
                            .copy_from_slice(&ext_cb_b[src_start..src_start + c_strip_width]);
                        ext_cr_a[below_ctx_start..below_ctx_start + c_strip_width]
                            .copy_from_slice(&ext_cr_b[src_start..src_start + c_strip_width]);
                    } else {
                        // Last MCU row: edge replication
                        ext_cb_a.copy_within(
                            last_data_row_start..last_data_row_start + c_strip_width,
                            below_ctx_start,
                        );
                        ext_cr_a.copy_within(
                            last_data_row_start..last_data_row_start + c_strip_width,
                            below_ctx_start,
                        );
                    }

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
                        &mut dequant_buf,
                    );

                    let y_rows_this_mcu =
                        y_strip_height.min(height.saturating_sub(imcu_row * mcu_height));
                    let local_rgb_offset = local_idx * mcu_row_rgb_bytes;

                    if !needs_full_upsample {
                        // NearestNeighbor 4:2:0: fused box-filter path
                        let c_rows_this_mcu = c_strip_height.min(
                            (height.saturating_sub(imcu_row * mcu_height) + v_ratio - 1) / v_ratio,
                        );
                        let c_cols = (y_cols_this_image + 1) / 2;

                        for row in 0..y_rows_this_mcu {
                            let y_offset = row * y_strip_width;
                            let c_row = (row / 2).min(c_rows_this_mcu.saturating_sub(1));
                            let c_offset = (1 + c_row) * c_strip_width;
                            let rgb_offset = local_rgb_offset + row * rgb_row_stride;

                            fused_h2v2_box_ycbcr_to_rgb_u8(
                                &y_strip[y_offset..y_offset + y_cols_this_image],
                                &ext_cb_a[c_offset..c_offset + c_cols],
                                &ext_cr_a[c_offset..c_offset + c_cols],
                                &mut rgb_batch[rgb_offset..rgb_offset + y_cols_this_image * 3],
                                y_cols_this_image,
                            );
                        }
                    } else {
                        // Upsample extended strip → reusable upsample buffers
                        let upsample_out_height = ext_height * v_ratio;
                        upsample_fn(
                            &ext_cb_a,
                            c_strip_width,
                            ext_height,
                            &mut cb_up,
                            y_strip_width,
                            upsample_out_height,
                        );
                        upsample_fn(
                            &ext_cr_a,
                            c_strip_width,
                            ext_height,
                            &mut cr_up,
                            y_strip_width,
                            upsample_out_height,
                        );

                        for row in 0..y_rows_this_mcu {
                            let strip_offset = row * y_strip_width;
                            let up_row = v_ratio + row;
                            let chroma_offset = up_row * y_strip_width;
                            let rgb_offset = local_rgb_offset + row * rgb_row_stride;

                            ycbcr_planes_i16_to_rgb_u8(
                                &y_strip[strip_offset..strip_offset + y_cols_this_image],
                                &cb_up[chroma_offset..chroma_offset + y_cols_this_image],
                                &cr_up[chroma_offset..chroma_offset + y_cols_this_image],
                                &mut rgb_batch[rgb_offset..rgb_offset + y_cols_this_image * 3],
                            );
                        }
                    }

                    // Swap buffers for next MCU row
                    if imcu_row + 1 < end_row {
                        // ext_b's above context = last data row of ext_a
                        ext_cb_b[..c_strip_width].copy_from_slice(
                            &ext_cb_a[last_data_row_start..last_data_row_start + c_strip_width],
                        );
                        ext_cr_b[..c_strip_width].copy_from_slice(
                            &ext_cr_a[last_data_row_start..last_data_row_start + c_strip_width],
                        );

                        core::mem::swap(&mut ext_cb_a, &mut ext_cb_b);
                        core::mem::swap(&mut ext_cr_a, &mut ext_cr_b);

                        // IDCT the strip after next into the now-free ext_b
                        if imcu_row + 2 < end_row {
                            idct_chroma_into_ext(
                                &mut ext_cb_b,
                                &coeffs[1],
                                &coeff_counts[1],
                                info_cb,
                                quant_cb,
                                imcu_row + 2,
                                c_strip_width,
                                c_strip_height,
                                chroma_height_total,
                                idct_fn,
                                &mut dequant_buf,
                            );
                            idct_chroma_into_ext(
                                &mut ext_cr_b,
                                &coeffs[2],
                                &coeff_counts[2],
                                info_cr,
                                quant_cr,
                                imcu_row + 2,
                                c_strip_width,
                                c_strip_height,
                                chroma_height_total,
                                idct_fn,
                                &mut dequant_buf,
                            );
                        }
                    }
                }
            });

        Ok(Some(rgb))
    }
}
