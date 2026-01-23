//! Pixel output conversion for decoded JPEG data.
//!
//! This module handles the final conversion from decoded DCT coefficients
//! to pixel output in various formats (RGB u8, RGB f32, YCbCr f32).
//!
//! ## Fast Paths
//!
//! - `to_pixels_fast_i16`: For 4:4:4 non-XYB images, uses integer IDCT throughout
//! - `to_pixels_fast_i16_subsampled`: For 4:2:0/4:2:2/4:4:0 non-XYB images
//!
//! ## Generic Paths
//!
//! - `to_pixels`: General f32 path with bias computation, handles XYB
//! - `to_pixels_f32`: f32 output normalized to [0.0, 1.0]
//! - `to_ycbcr_planes_f32`: Raw YCbCr planes for custom processing

use super::super::idct::inverse_dct_8x8;
use super::super::idct_int::{idct_int_auto, idct_int_tiered};
use super::super::upsample::upsample_fancy;
use crate::color::{
    gray_f32_to_gray_f32, gray_f32_to_gray_u8, gray_f32_to_rgb_f32, gray_f32_to_rgb_u8,
    ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8,
};
use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit};
use crate::foundation::consts::{DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER};
use crate::quant::{
    dequantize_block, dequantize_block_i32, dequantize_block_with_bias, dequantize_unzigzag_i32,
    DequantBiasStats,
};
use crate::types::PixelFormat;

use super::JpegParser;

/// Pixel output conversion methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Check if we can use the fast i16 path for 4:4:4 images.
    ///
    /// Fast path requirements:
    /// - Non-XYB (standard JPEG)
    /// - 4:4:4 subsampling (no chroma downsampling to avoid f32 upsampling)
    /// - RGB output format
    fn can_use_fast_i16_path(&self, format: PixelFormat, is_xyb: bool) -> bool {
        if is_xyb {
            return false;
        }
        if format != PixelFormat::Rgb {
            return false;
        }
        if self.num_components != 3 {
            return false;
        }

        // Check for 4:4:4 (all components have same sampling factors)
        let h_samp_0 = self.components[0].h_samp_factor;
        let v_samp_0 = self.components[0].v_samp_factor;
        for i in 1..3 {
            if self.components[i].h_samp_factor != h_samp_0
                || self.components[i].v_samp_factor != v_samp_0
            {
                return false;
            }
        }

        true
    }

    /// Check if we can use the fast i16 path for subsampled images (4:2:0, 4:2:2, 4:4:0).
    ///
    /// Fast path requirements:
    /// - Non-XYB (standard JPEG)
    /// - RGB output format
    /// - 3 components (YCbCr)
    /// - Standard subsampling (Y full-res, Cb/Cr subsampled)
    fn can_use_fast_i16_subsampled(&self, format: PixelFormat, is_xyb: bool) -> bool {
        if is_xyb {
            return false;
        }
        if format != PixelFormat::Rgb {
            return false;
        }
        if self.num_components != 3 {
            return false;
        }

        // Y component should have the highest sampling factors
        let y_h = self.components[0].h_samp_factor;
        let y_v = self.components[0].v_samp_factor;

        // Cb and Cr should have <= Y sampling
        let cb_h = self.components[1].h_samp_factor;
        let cb_v = self.components[1].v_samp_factor;
        let cr_h = self.components[2].h_samp_factor;
        let cr_v = self.components[2].v_samp_factor;

        // Cb and Cr must match each other
        if cb_h != cr_h || cb_v != cr_v {
            return false;
        }

        // Chroma must be subsampled (not 4:4:4, that uses the other path)
        if cb_h == y_h && cb_v == y_v {
            return false;
        }

        // Only support standard ratios: 2x1 (4:2:2), 1x2 (4:4:0), 2x2 (4:2:0)
        let h_ratio = y_h / cb_h;
        let v_ratio = y_v / cb_v;

        matches!((h_ratio, v_ratio), (2, 1) | (1, 2) | (2, 2))
    }

    /// Fast decode path using integer arithmetic throughout.
    ///
    /// This path avoids f32 entirely by using:
    /// - Integer IDCT (outputs i16 [0, 255])
    /// - Integer color conversion (i16 YCbCr → u8 RGB)
    ///
    /// Streams MCU row by row to keep data in L2 cache.
    /// Only works for non-XYB 4:4:4 RGB output.
    fn to_pixels_fast_i16(&self, _fancy_upsampling: bool) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors (should all be the same for 4:4:4)
        let max_h_samp = self.components[0].h_samp_factor as usize;
        let max_v_samp = self.components[0].v_samp_factor as usize;

        // MCU dimensions
        let mcu_height = max_v_samp * 8;
        let mcu_cols = (width + max_h_samp * 8 - 1) / (max_h_samp * 8);
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Component info
        let comp_infos = self.build_comp_infos(mcu_cols, mcu_rows, max_h_samp, max_v_samp, 3)?;

        // Allocate strip buffers for one MCU row (reused each iteration)
        // Strip height = max_v_samp * 8 pixels
        let strip_height = mcu_height;
        let strip_width = comp_infos[0].comp_width;
        let strip_size = strip_width * strip_height;

        // Allocate strip buffers - values will be fully overwritten by IDCT
        // Note: Strips are fully written by IDCT before color conversion reads them
        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Y strip buffer")?;
        let mut cb_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cb strip buffer")?;
        let mut cr_strip: Vec<i16> = try_alloc_maybeuninit(strip_size, "Cr strip buffer")?;

        // Allocate output RGB buffer
        // Note: All pixels are written by color conversion before the buffer is returned
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        // Process MCU row by row
        for imcu_row in 0..mcu_rows {
            // No need to clear strips - we write all pixels we'll read

            // IDCT all blocks in this MCU row for all 3 components
            for comp_idx in 0..3 {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                let strip = match comp_idx {
                    0 => &mut y_strip,
                    1 => &mut cb_strip,
                    _ => &mut cr_strip,
                };

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    let strip_row = iy * DCT_SIZE; // Row within the strip

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];
                        let coeff_count = self.coeff_counts[comp_idx][block_idx];

                        // Fused dequantize + unzigzag (single pass)
                        let mut dequant_i32 = dequantize_unzigzag_i32(coeffs, quant);

                        // IDCT writes directly to strip buffer (no intermediate copy)
                        // Use tiered IDCT based on coefficient count for speed
                        let base_px = bx * DCT_SIZE;
                        let dst_offset = strip_row * strip_width + base_px;
                        idct_int_tiered(
                            &mut dequant_i32,
                            &mut strip[dst_offset..],
                            strip_width,
                            coeff_count,
                        );
                    }
                }
            }

            // Color convert this MCU row's strips directly to RGB output
            let y_start = imcu_row * mcu_height;
            let rows_this_mcu = mcu_height.min(height.saturating_sub(y_start));
            let cols_this_mcu = width.min(strip_width);

            for row in 0..rows_this_mcu {
                let strip_offset = row * strip_width;
                let rgb_offset = (y_start + row) * width * 3;

                // Convert one row at a time for cache efficiency
                ycbcr_planes_i16_to_rgb_u8(
                    &y_strip[strip_offset..strip_offset + cols_this_mcu],
                    &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                    &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                    &mut rgb[rgb_offset..rgb_offset + cols_this_mcu * 3],
                );
            }
        }

        Ok(rgb)
    }

    /// Fast decode path for subsampled images (4:2:0, 4:2:2, 4:4:0) using i16 throughout.
    ///
    /// This path avoids f32 entirely by using:
    /// - Integer IDCT (outputs i16 [0, 255])
    /// - Integer upsampling (i16 → i16)
    /// - Integer color conversion (i16 YCbCr → u8 RGB)
    fn to_pixels_fast_i16_subsampled(&self, fancy_upsampling: bool) -> Result<Vec<u8>> {
        use crate::decode::upsample::{
            upsample_h1v2_i16_fancy, upsample_h2v1_i16_fancy, upsample_h2v2_i16_fancy,
        };

        let width = self.width as usize;
        let height = self.height as usize;

        // Get sampling factors
        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let c_h = self.components[1].h_samp_factor as usize;
        let c_v = self.components[1].v_samp_factor as usize;

        let h_ratio = y_h / c_h;
        let v_ratio = y_v / c_v;

        // MCU dimensions
        let mcu_width = y_h * 8;
        let mcu_height = y_v * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Component info
        let comp_infos = self.build_comp_infos(mcu_cols, mcu_rows, y_h, y_v, 3)?;

        // Allocate strip buffers for one MCU row
        let y_strip_height = y_v * 8;
        let y_strip_width = comp_infos[0].comp_width;
        let y_strip_size = y_strip_width * y_strip_height;

        let c_strip_height = c_v * 8;
        let c_strip_width = comp_infos[1].comp_width;
        let c_strip_size = c_strip_width * c_strip_height;

        // Y strip at full resolution
        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(y_strip_size, "Y strip buffer")?;
        // Chroma strips at subsampled resolution
        let mut cb_strip_sub: Vec<i16> = try_alloc_maybeuninit(c_strip_size, "Cb strip buffer")?;
        let mut cr_strip_sub: Vec<i16> = try_alloc_maybeuninit(c_strip_size, "Cr strip buffer")?;
        // Upsampled chroma strips
        let mut cb_strip: Vec<i16> = try_alloc_maybeuninit(y_strip_size, "Cb upsampled buffer")?;
        let mut cr_strip: Vec<i16> = try_alloc_maybeuninit(y_strip_size, "Cr upsampled buffer")?;

        // Allocate output RGB buffer
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        // Pre-fetch quant tables outside the loop (avoids Error allocation per MCU row)
        let quant_y = self.quant_tables[comp_infos[0].quant_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Y quant table"))?;
        let quant_cb = self.quant_tables[comp_infos[1].quant_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Cb quant table"))?;
        let quant_cr = self.quant_tables[comp_infos[2].quant_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("missing Cr quant table"))?;

        // Process MCU row by row
        for imcu_row in 0..mcu_rows {
            // IDCT Y blocks (full resolution)
            {
                let info = &comp_infos[0];
                let quant = quant_y;

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }
                    let strip_row = iy * DCT_SIZE;

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[0].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[0][block_idx];
                        let coeff_count = self.coeff_counts[0][block_idx];

                        let mut dequant_i32 = dequantize_unzigzag_i32(coeffs, quant);
                        let base_px = bx * DCT_SIZE;
                        let dst_offset = strip_row * y_strip_width + base_px;
                        idct_int_tiered(
                            &mut dequant_i32,
                            &mut y_strip[dst_offset..],
                            y_strip_width,
                            coeff_count,
                        );
                    }
                }
            }

            // IDCT Cb blocks (subsampled)
            {
                let info = &comp_infos[1];
                let quant = quant_cb;

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }
                    let strip_row = iy * DCT_SIZE;

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[1].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[1][block_idx];
                        let coeff_count = self.coeff_counts[1][block_idx];

                        let mut dequant_i32 = dequantize_unzigzag_i32(coeffs, quant);
                        let base_px = bx * DCT_SIZE;
                        let dst_offset = strip_row * c_strip_width + base_px;
                        idct_int_tiered(
                            &mut dequant_i32,
                            &mut cb_strip_sub[dst_offset..],
                            c_strip_width,
                            coeff_count,
                        );
                    }
                }
            }

            // IDCT Cr blocks (subsampled)
            {
                let info = &comp_infos[2];
                let quant = quant_cr;

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }
                    let strip_row = iy * DCT_SIZE;

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[2].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[2][block_idx];
                        let coeff_count = self.coeff_counts[2][block_idx];

                        let mut dequant_i32 = dequantize_unzigzag_i32(coeffs, quant);
                        let base_px = bx * DCT_SIZE;
                        let dst_offset = strip_row * c_strip_width + base_px;
                        idct_int_tiered(
                            &mut dequant_i32,
                            &mut cr_strip_sub[dst_offset..],
                            c_strip_width,
                            coeff_count,
                        );
                    }
                }
            }

            // Upsample chroma to match Y resolution
            let y_rows_this_mcu = y_strip_height.min(height.saturating_sub(imcu_row * mcu_height));
            let y_cols_this_mcu = width.min(y_strip_width);
            let c_rows_this_mcu = c_strip_height.min(
                (height.saturating_sub(imcu_row * mcu_height) + v_ratio - 1) / v_ratio,
            );
            let _c_cols_this_mcu = (y_cols_this_mcu + h_ratio - 1) / h_ratio;

            if fancy_upsampling {
                match (h_ratio, v_ratio) {
                    (2, 2) => {
                        // Uses multiversion for automatic SIMD dispatch
                        upsample_h2v2_i16_fancy(
                            &cb_strip_sub,
                            c_strip_width,
                            c_rows_this_mcu,
                            &mut cb_strip,
                            y_strip_width,
                            y_rows_this_mcu,
                        );
                        upsample_h2v2_i16_fancy(
                            &cr_strip_sub,
                            c_strip_width,
                            c_rows_this_mcu,
                            &mut cr_strip,
                            y_strip_width,
                            y_rows_this_mcu,
                        );
                    }
                    (2, 1) => {
                        upsample_h2v1_i16_fancy(
                            &cb_strip_sub,
                            c_strip_width,
                            c_rows_this_mcu,
                            &mut cb_strip,
                            y_strip_width,
                            y_rows_this_mcu,
                        );
                        upsample_h2v1_i16_fancy(
                            &cr_strip_sub,
                            c_strip_width,
                            c_rows_this_mcu,
                            &mut cr_strip,
                            y_strip_width,
                            y_rows_this_mcu,
                        );
                    }
                    (1, 2) => {
                        upsample_h1v2_i16_fancy(
                            &cb_strip_sub,
                            c_strip_width,
                            c_rows_this_mcu,
                            &mut cb_strip,
                            y_strip_width,
                            y_rows_this_mcu,
                        );
                        upsample_h1v2_i16_fancy(
                            &cr_strip_sub,
                            c_strip_width,
                            c_rows_this_mcu,
                            &mut cr_strip,
                            y_strip_width,
                            y_rows_this_mcu,
                        );
                    }
                    _ => unreachable!("unsupported ratio should be filtered by can_use_fast_i16_subsampled"),
                }
            } else {
                // Box filter (simple pixel duplication)
                use crate::decode::upsample::upsample_h2v2_i16_box;
                // Only 4:2:0 has a box filter implementation; others use fancy even without flag
                if h_ratio == 2 && v_ratio == 2 {
                    upsample_h2v2_i16_box(
                        &cb_strip_sub,
                        c_strip_width,
                        c_rows_this_mcu,
                        &mut cb_strip,
                        y_strip_width,
                        y_rows_this_mcu,
                    );
                    upsample_h2v2_i16_box(
                        &cr_strip_sub,
                        c_strip_width,
                        c_rows_this_mcu,
                        &mut cr_strip,
                        y_strip_width,
                        y_rows_this_mcu,
                    );
                } else {
                    // Fall back to fancy for 4:2:2 and 4:4:0
                    match (h_ratio, v_ratio) {
                        (2, 1) => {
                            upsample_h2v1_i16_fancy(
                                &cb_strip_sub,
                                c_strip_width,
                                c_rows_this_mcu,
                                &mut cb_strip,
                                y_strip_width,
                                y_rows_this_mcu,
                            );
                            upsample_h2v1_i16_fancy(
                                &cr_strip_sub,
                                c_strip_width,
                                c_rows_this_mcu,
                                &mut cr_strip,
                                y_strip_width,
                                y_rows_this_mcu,
                            );
                        }
                        (1, 2) => {
                            upsample_h1v2_i16_fancy(
                                &cb_strip_sub,
                                c_strip_width,
                                c_rows_this_mcu,
                                &mut cb_strip,
                                y_strip_width,
                                y_rows_this_mcu,
                            );
                            upsample_h1v2_i16_fancy(
                                &cr_strip_sub,
                                c_strip_width,
                                c_rows_this_mcu,
                                &mut cr_strip,
                                y_strip_width,
                                y_rows_this_mcu,
                            );
                        }
                        _ => unreachable!(),
                    }
                }
            }

            // Color convert and write to output
            let y_start = imcu_row * mcu_height;
            for row in 0..y_rows_this_mcu {
                let strip_offset = row * y_strip_width;
                let rgb_offset = (y_start + row) * width * 3;

                ycbcr_planes_i16_to_rgb_u8(
                    &y_strip[strip_offset..strip_offset + y_cols_this_mcu],
                    &cb_strip[strip_offset..strip_offset + y_cols_this_mcu],
                    &cr_strip[strip_offset..strip_offset + y_cols_this_mcu],
                    &mut rgb[rgb_offset..rgb_offset + y_cols_this_mcu * 3],
                );
            }
        }

        Ok(rgb)
    }

    /// Convert decoded coefficients to pixels in the requested format.
    ///
    /// This is the main entry point for pixel output. It automatically selects
    /// the fastest path based on the image characteristics.
    #[allow(clippy::wrong_self_convention)] // Takes &mut self to take() internal buffer
    pub(in crate::decode) fn to_pixels(
        &mut self,
        format: PixelFormat,
        is_xyb: bool,
        fancy_upsampling: bool,
    ) -> Result<Vec<u8>> {
        // If streaming decode was used, return its result directly (zero-copy)
        if format == PixelFormat::Rgb && !is_xyb {
            if let Some(rgb) = self.streaming_rgb.take() {
                return Ok(rgb);
            }
        }

        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        // Try fast integer path for non-XYB 4:4:4 RGB images
        if self.can_use_fast_i16_path(format, is_xyb) {
            return self.to_pixels_fast_i16(fancy_upsampling);
        }

        // Try fast integer path for subsampled images (4:2:0, 4:2:2, 4:4:0)
        if self.can_use_fast_i16_subsampled(format, is_xyb) {
            return self.to_pixels_fast_i16_subsampled(fancy_upsampling);
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Pre-compute component info for efficiency
        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            self.num_components as usize,
        )?;

        // Initialize bias stats and biases (C++ initializes to 0 via memset)
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32 (C++ jpegli keeps f32 until final output)
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row (matching C++ incremental bias recomputation)
        for imcu_row in 0..mcu_rows {
            // For each component in this MCU row
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                // Phase 1: Gather stats for full-res components
                if info.is_full_res {
                    for iy in 0..info.v_samp {
                        let by = imcu_row * info.v_samp + iy;
                        if by >= info.comp_blocks_v {
                            continue;
                        }
                        for bx in 0..info.comp_blocks_h {
                            let block_idx = by * info.comp_blocks_h + bx;
                            if block_idx >= self.coeffs[comp_idx].len() {
                                continue;
                            }
                            let coeffs = &self.coeffs[comp_idx][block_idx];
                            let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Phase 2: Recompute biases every 4 MCU rows (matching C++ behavior)
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // Phase 3: IDCT for this component in this MCU row
                // Store as f32 (C++ jpegli keeps f32 until final output for precision)
                let _biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    // Pre-compute base y position and check row bounds once
                    let base_py = by * DCT_SIZE;
                    let rows_to_copy = DCT_SIZE.min(info.comp_height.saturating_sub(base_py));

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        // Zigzag reorder
                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Store pixels - use row-based copy for efficiency
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        if is_xyb {
                            // XYB mode: use f32 IDCT for extended gamut precision
                            let dequant = dequantize_block(&natural_coeffs, quant);
                            let pixels = inverse_dct_8x8(&dequant);

                            if cols_to_copy == DCT_SIZE {
                                for y in 0..rows_to_copy {
                                    let dst_offset = (base_py + y) * info.comp_width + base_px;
                                    let src_offset = y * DCT_SIZE;
                                    comp_plane_f32[dst_offset..dst_offset + DCT_SIZE]
                                        .copy_from_slice(
                                            &pixels[src_offset..src_offset + DCT_SIZE],
                                        );
                                }
                            } else {
                                for y in 0..rows_to_copy {
                                    for x in 0..cols_to_copy {
                                        comp_plane_f32
                                            [(base_py + y) * info.comp_width + base_px + x] =
                                            pixels[y * DCT_SIZE + x];
                                    }
                                }
                            }
                        } else {
                            // Standard JPEG: use fast integer IDCT
                            let mut dequant_i32 = dequantize_block_i32(&natural_coeffs, quant);
                            let mut pixels_i16 = [0i16; DCT_BLOCK_SIZE];
                            idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8);

                            // Convert i16 [0,255] to f32 centered [-128,127]
                            if cols_to_copy == DCT_SIZE {
                                for y in 0..rows_to_copy {
                                    let dst_offset = (base_py + y) * info.comp_width + base_px;
                                    let src_offset = y * DCT_SIZE;
                                    for x in 0..DCT_SIZE {
                                        comp_plane_f32[dst_offset + x] =
                                            pixels_i16[src_offset + x] as f32 - 128.0;
                                    }
                                }
                            } else {
                                for y in 0..rows_to_copy {
                                    for x in 0..cols_to_copy {
                                        comp_plane_f32
                                            [(base_py + y) * info.comp_width + base_px + x] =
                                            pixels_i16[y * DCT_SIZE + x] as f32 - 128.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample if needed - keep as f32 for precision
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    // Triangle filter (3:1 weights) - separable implementation
                    // First upsample horizontally, then vertically
                    upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        // Convert to output format using batch conversion functions
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and convert to u8
                let mut output = vec![0u8; output_size];
                gray_f32_to_gray_u8(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];
                gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb);
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];

                if is_xyb {
                    // XYB mode: Output raw level-shifted values, NO YCbCr→RGB conversion.
                    // The XYB values are stored in YCbCr positions but are NOT YCbCr.
                    // The ICC profile transforms these directly to sRGB.
                    crate::color::xyb::xyb_planes_to_rgb_u8_simd(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                } else {
                    // YCbCr to RGB conversion using batch function
                    ycbcr_planes_f32_to_rgb_u8(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                }
                Ok(rgb)
            }
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// Convert decoded coefficients to f32 pixels.
    /// Values are normalized to range 0.0-1.0.
    pub(in crate::decode) fn to_pixels_f32(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        fancy_upsampling: bool,
    ) -> Result<Vec<f32>> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Pre-compute component info
        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            self.num_components as usize,
        )?;

        // Initialize bias stats and biases
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                // Gather stats for full-res components
                if info.is_full_res {
                    for iy in 0..info.v_samp {
                        let by = imcu_row * info.v_samp + iy;
                        if by >= info.comp_blocks_v {
                            continue;
                        }
                        for bx in 0..info.comp_blocks_h {
                            let block_idx = by * info.comp_blocks_h + bx;
                            if block_idx >= self.coeffs[comp_idx].len() {
                                continue;
                            }
                            let coeffs = &self.coeffs[comp_idx][block_idx];
                            let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Recompute biases every 4 MCU rows
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // IDCT for this component
                let biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Always use f32 IDCT for f32 output - preserves fractional precision
                        let dequant = if is_xyb {
                            dequantize_block(&natural_coeffs, quant)
                        } else {
                            dequantize_block_with_bias(&natural_coeffs, quant, biases)
                        };
                        let pixels = inverse_dct_8x8(&dequant);

                        for y in 0..DCT_SIZE {
                            for x in 0..DCT_SIZE {
                                let px = bx * DCT_SIZE + x;
                                let py = by * DCT_SIZE + y;
                                if px < info.comp_width && py < info.comp_height {
                                    comp_plane_f32[py * info.comp_width + px] =
                                        pixels[y * DCT_SIZE + x];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample if needed
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    // Triangle filter (3:1 weights) - separable implementation
                    upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        // Convert to output format as f32 (values normalized to 0.0-1.0)
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and normalize to 0.0-1.0
                let mut output = vec![0.0f32; output_size];
                gray_f32_to_gray_f32(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0.0f32; rgb_size];
                gray_f32_to_rgb_f32(&planes_f32[0], &mut rgb);
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0.0f32; rgb_size];

                if is_xyb {
                    // XYB mode: Output raw level-shifted values, normalized to 0.0-1.0
                    crate::color::xyb::xyb_planes_to_rgb_f32_simd(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                } else {
                    // YCbCr to RGB conversion using batch function
                    ycbcr_planes_f32_to_rgb_f32(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                }
                Ok(rgb)
            }
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// Convert decoded coefficients to YCbCr f32 planes.
    ///
    /// Returns (Y, Cb, Cr) planes, each width×height in size.
    /// Values are in centered range [-128, 127] (raw DCT output).
    /// Chroma planes are upsampled to full resolution.
    pub(in crate::decode) fn to_ycbcr_planes_f32(
        &self,
        fancy_upsampling: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        if self.num_components != 3 {
            return Err(Error::unsupported_feature(
                "YCbCr planes require 3-component image",
            ));
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Pre-compute component info
        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            self.num_components as usize,
        )?;

        // Initialize bias stats and biases
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                // Phase 1: Gather stats for full-res components
                if info.is_full_res {
                    for iy in 0..info.v_samp {
                        let by = imcu_row * info.v_samp + iy;
                        if by >= info.comp_blocks_v {
                            continue;
                        }
                        for bx in 0..info.comp_blocks_h {
                            let block_idx = by * info.comp_blocks_h + bx;
                            if block_idx >= self.coeffs[comp_idx].len() {
                                continue;
                            }
                            let coeffs = &self.coeffs[comp_idx][block_idx];
                            let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Recompute biases every 4 MCU rows
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // Phase 2: IDCT
                let _biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    let base_py = by * DCT_SIZE;
                    let rows_to_copy = DCT_SIZE.min(info.comp_height.saturating_sub(base_py));

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        // Zigzag reorder
                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Use fast integer IDCT (always non-XYB for YCbCr output)
                        let mut dequant_i32 = dequantize_block_i32(&natural_coeffs, quant);
                        let mut pixels_i16 = [0i16; DCT_BLOCK_SIZE];
                        idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8);

                        // Store pixels
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        // Convert i16 [0,255] to f32 centered [-128,127]
                        if cols_to_copy == DCT_SIZE {
                            for y in 0..rows_to_copy {
                                let dst_offset = (base_py + y) * info.comp_width + base_px;
                                let src_offset = y * DCT_SIZE;
                                for x in 0..DCT_SIZE {
                                    comp_plane_f32[dst_offset + x] =
                                        pixels_i16[src_offset + x] as f32 - 128.0;
                                }
                            }
                        } else {
                            for y in 0..rows_to_copy {
                                for x in 0..cols_to_copy {
                                    comp_plane_f32[(base_py + y) * info.comp_width + base_px + x] =
                                        pixels_i16[y * DCT_SIZE + x] as f32 - 128.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample chroma and clip to image dimensions
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::with_capacity(3);

        for comp_idx in 0..3 {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        Ok((
            core::mem::take(&mut planes_f32[0]),
            core::mem::take(&mut planes_f32[1]),
            core::mem::take(&mut planes_f32[2]),
        ))
    }
}
