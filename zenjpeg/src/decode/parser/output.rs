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

#[path = "output_helpers.rs"]
mod output_helpers;

#[cfg(feature = "parallel")]
#[path = "output_parallel.rs"]
mod output_parallel;

use super::super::idct::inverse_dct_8x8;
use super::super::idct_int::{
    idct_int_auto, idct_int_libjpeg, idct_int_tiered, idct_int_tiered_libjpeg,
};
use super::super::upsample::{upsample_libjpeg_f32, upsample_nearest_f32};
use crate::color::{
    cmyk_planes_to_rgb_u8, gray_f32_to_gray_f32, gray_f32_to_gray_u8, gray_f32_to_rgb_f32,
    gray_f32_to_rgb_u8,
    ycbcr::{
        fused_h2v2_box_ycbcr_to_rgb_u8, rgb_u8_swap_rb_inplace, rgb_u8_to_bgra_u8,
        rgb_u8_to_rgba_u8,
    },
    ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8,
    ycck_planes_to_rgb_u8,
};
use crate::decode::extras::AdobeColorTransform;
use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit};
use crate::foundation::consts::{DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER};
use crate::quant::{
    DequantBiasStats, dequantize_block, dequantize_block_i32, dequantize_block_with_bias,
};
use crate::types::PixelFormat;
use enough::Stop;

use super::{CompInfo, JpegParser};
use output_helpers::{idct_chroma_into_ext, idct_comp_mcu_row};

/// Returns true for formats that decode via the RGB u8 fast paths
/// (i16 IDCT → direct u8 output), then optionally reformat.
fn is_rgb_family_u8(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb
            | PixelFormat::Bgr
            | PixelFormat::Rgba
            | PixelFormat::Bgra
            | PixelFormat::Bgrx
    )
}

/// Reformats an RGB u8 buffer into the target `PixelFormat`.
///
/// - `Rgb` → passthrough (returns `rgb` unchanged)
/// - `Bgr` → in-place R/B swap (returns same buffer)
/// - `Rgba` → allocate 4-bpp buffer, copy with alpha = 255
/// - `Bgra` / `Bgrx` → allocate 4-bpp buffer, swap R/B + alpha = 255
fn reformat_rgb_output(
    mut rgb: Vec<u8>,
    format: PixelFormat,
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    match format {
        PixelFormat::Rgb => Ok(rgb),
        PixelFormat::Bgr => {
            rgb_u8_swap_rb_inplace(&mut rgb);
            Ok(rgb)
        }
        PixelFormat::Rgba => {
            let dst_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 4))?;
            let mut dst = vec![0u8; dst_size];
            rgb_u8_to_rgba_u8(&rgb, &mut dst);
            Ok(dst)
        }
        PixelFormat::Bgra | PixelFormat::Bgrx => {
            let dst_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 4))?;
            let mut dst = vec![0u8; dst_size];
            rgb_u8_to_bgra_u8(&rgb, &mut dst);
            Ok(dst)
        }
        _ => Err(Error::unsupported_feature("unsupported color conversion")),
    }
}

/// Pixel output conversion methods for JpegParser.
impl<'a> JpegParser<'a> {
    /// Check if this JPEG stores raw RGB (not YCbCr).
    ///
    /// Adobe APP14 with transform=0 and 3 components means the data is stored
    /// as raw RGB. Also detect RGB component IDs (R=82, G=71, B=66) without
    /// Adobe marker. In these cases we must NOT apply YCbCr→RGB conversion.
    pub(crate) fn is_rgb_jpeg(&self) -> bool {
        if self.num_components != 3 {
            return false;
        }
        // Adobe APP14 transform byte determines encoding:
        // 0 = Unknown/RGB (no transform), 1 = YCbCr, 2 = YCCK
        match self.adobe_transform {
            Some(AdobeColorTransform::YCbCr) => return false, // Explicitly YCbCr
            Some(AdobeColorTransform::Unknown) => return true, // Explicitly RGB (transform=0)
            Some(AdobeColorTransform::Ycck) => return false,  // YCCK (shouldn't be 3-component)
            None => {}                                        // No Adobe marker, check IDs
        }
        // No Adobe APP14: check component IDs for RGB ('R'=82, 'G'=71, 'B'=66)
        let ids: [u8; 3] = [
            self.components[0].id,
            self.components[1].id,
            self.components[2].id,
        ];
        ids == [b'R', b'G', b'B']
    }

    /// Check if we can use the fast i16 path for 4:4:4 images.
    ///
    /// Fast path requirements:
    /// - Non-XYB (standard JPEG)
    /// - 4:4:4 subsampling (no chroma downsampling to avoid f32 upsampling)
    /// - RGB-family output format (Rgb, Bgr, Rgba, Bgra, Bgrx)
    fn can_use_fast_i16_path(&self, format: PixelFormat, is_xyb: bool) -> bool {
        if is_xyb || self.force_f32_idct {
            return false;
        }
        if !is_rgb_family_u8(format) {
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
    /// - RGB-family output format (Rgb, Bgr, Rgba, Bgra, Bgrx)
    /// - 3 components (YCbCr)
    /// - Standard subsampling (Y full-res, Cb/Cr subsampled)
    fn can_use_fast_i16_subsampled(&self, format: PixelFormat, is_xyb: bool) -> bool {
        if is_xyb || self.force_f32_idct {
            return false;
        }
        if !is_rgb_family_u8(format) {
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
    fn to_pixels_fast_i16(
        &self,
        _chroma_upsampling: super::super::ChromaUpsampling,
    ) -> Result<Vec<u8>> {
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
        // Reusable dequant buffer — avoids per-block [0i32; 64] zeroing
        let mut dequant_i32 = [0i32; DCT_BLOCK_SIZE];

        for imcu_row in 0..mcu_rows {
            // No need to clear strips - we write all pixels we'll read

            // IDCT all blocks in this MCU row for all 3 components
            let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match self.idct_method {
                super::super::IdctMethod::Libjpeg => idct_int_tiered_libjpeg,
                super::super::IdctMethod::Jpegli => idct_int_tiered,
            };

            for (comp_idx, strip) in [&mut y_strip, &mut cb_strip, &mut cr_strip]
                .into_iter()
                .enumerate()
            {
                let quant = self.quant_tables[comp_infos[comp_idx].quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                idct_comp_mcu_row(
                    &self.coeffs[comp_idx],
                    &self.coeff_counts[comp_idx],
                    &comp_infos[comp_idx],
                    quant,
                    imcu_row,
                    strip,
                    strip_width,
                    idct_fn,
                    &mut dequant_i32,
                );
            }

            // Color convert this MCU row's strips directly to RGB output
            let y_start = imcu_row * mcu_height;
            let rows_this_mcu = mcu_height.min(height.saturating_sub(y_start));
            let cols_this_mcu = width.min(strip_width);
            let is_rgb = self.is_rgb_jpeg();

            for row in 0..rows_this_mcu {
                let strip_offset = row * strip_width;
                let rgb_offset = (y_start + row) * width * 3;

                if is_rgb {
                    // RGB JPEG: just interleave the 3 planes (no YCbCr→RGB matrix)
                    // IDCT already output level-shifted values [0, 255]
                    for px in 0..cols_this_mcu {
                        let i = strip_offset + px;
                        let o = rgb_offset + px * 3;
                        rgb[o] = y_strip[i].clamp(0, 255) as u8;
                        rgb[o + 1] = cb_strip[i].clamp(0, 255) as u8;
                        rgb[o + 2] = cr_strip[i].clamp(0, 255) as u8;
                    }
                } else {
                    // Convert one row at a time for cache efficiency
                    ycbcr_planes_i16_to_rgb_u8(
                        &y_strip[strip_offset..strip_offset + cols_this_mcu],
                        &cb_strip[strip_offset..strip_offset + cols_this_mcu],
                        &cr_strip[strip_offset..strip_offset + cols_this_mcu],
                        &mut rgb[rgb_offset..rgb_offset + cols_this_mcu * 3],
                    );
                }
            }
        }

        Ok(rgb)
    }

    /// Fast decode path for subsampled images (4:2:0, 4:2:2, 4:4:0) using i16 throughout.
    ///
    /// Uses double-buffered extended chroma strips instead of full-plane allocation.
    /// Each extended strip has `c_strip_height + 2` rows: one boundary context row
    /// above, `c_strip_height` data rows (IDCT output), and one boundary context row
    /// below. The upsampler's edge-replication logic (`saturating_sub(1)` and
    /// `.min(in_height-1)`) naturally reads the context rows, producing correct
    /// cross-MCU-boundary interpolation without any changes to the upsample functions.
    ///
    /// Memory: ~200KB for 2048x2048 vs ~20MB for full-plane approach.
    fn to_pixels_fast_i16_subsampled(
        &self,
        chroma_upsampling: super::super::ChromaUpsampling,
    ) -> Result<Vec<u8>> {
        use super::super::ChromaUpsampling;
        use crate::decode::upsample::{
            upsample_h1v2_i16_libjpeg, upsample_h1v2_i16_nearest, upsample_h2v1_i16_libjpeg,
            upsample_h2v1_i16_nearest, upsample_h2v2_i16_libjpeg, upsample_h2v2_i16_nearest,
        };

        // Select IDCT function based on configured method
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match self.idct_method {
            super::super::IdctMethod::Libjpeg => idct_int_tiered_libjpeg,
            super::super::IdctMethod::Jpegli => idct_int_tiered,
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

        // Y strip dimensions (one MCU row)
        let y_strip_height = y_v * 8;
        let y_strip_width = comp_infos[0].comp_width;
        let y_strip_size = y_strip_width * y_strip_height;

        // Chroma dimensions (per strip, subsampled)
        let c_strip_height = c_v * 8;
        let c_strip_width = comp_infos[1].comp_width;

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

        let y_cols_this_image = width.min(y_strip_width);

        // Select upsampling function based on method
        type UpsampleFn = fn(&[i16], usize, usize, &mut [i16], usize, usize);
        let needs_full_upsample = !matches!(chroma_upsampling, ChromaUpsampling::NearestNeighbor)
            || h_ratio != 2
            || v_ratio != 2;

        let (upsample_fn, _): (UpsampleFn, ()) = if needs_full_upsample {
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

            let f = match (h_ratio, v_ratio) {
                (2, 2) => upsample_h2v2,
                (2, 1) => upsample_h2v1,
                (1, 2) => upsample_h1v2,
                _ => unreachable!(
                    "unsupported ratio should be filtered by can_use_fast_i16_subsampled"
                ),
            };
            (f, ())
        } else {
            // Placeholder — NearestNeighbor 4:2:0 uses the fused path
            (upsample_h2v2_i16_nearest, ())
        };

        // ===================================================================
        // Extended-buffer chroma strips (double-buffered).
        //
        // Each extended buffer has c_strip_height + 2 rows:
        //   [row 0]                     = above context (last row from previous strip)
        //   [rows 1..c_strip_height+1]  = IDCT output (current strip data)
        //   [row c_strip_height+1]      = below context (first row from next strip)
        //
        // The upsampler sees in_height = c_strip_height + 2 and its edge
        // replication reads the context rows, producing correct cross-MCU
        // boundary interpolation without any changes to upsample functions.
        // ===================================================================
        let ext_height = c_strip_height + 2;
        let ext_size = ext_height * c_strip_width;

        // Double buffers: ext_a is the current strip, ext_b holds the next
        let mut ext_cb_a: Vec<i16> = try_alloc_maybeuninit(ext_size, "Cb ext_a buffer")?;
        let mut ext_cb_b: Vec<i16> = try_alloc_maybeuninit(ext_size, "Cb ext_b buffer")?;
        let mut ext_cr_a: Vec<i16> = try_alloc_maybeuninit(ext_size, "Cr ext_a buffer")?;
        let mut ext_cr_b: Vec<i16> = try_alloc_maybeuninit(ext_size, "Cr ext_b buffer")?;

        // Upsampled output for one extended strip (used per MCU row)
        let upsample_out_height = ext_height * v_ratio;
        let upsample_out_size = upsample_out_height * y_strip_width;
        let mut cb_up: Vec<i16> = if needs_full_upsample {
            try_alloc_maybeuninit(upsample_out_size, "Cb upsample buffer")?
        } else {
            Vec::new()
        };
        let mut cr_up: Vec<i16> = if needs_full_upsample {
            try_alloc_maybeuninit(upsample_out_size, "Cr upsample buffer")?
        } else {
            Vec::new()
        };

        let mut y_strip: Vec<i16> = try_alloc_maybeuninit(y_strip_size, "Y strip buffer")?;
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "RGB output buffer")?;

        // Total valid chroma rows for the whole image
        let chroma_height_total = (height + v_ratio - 1) / v_ratio;

        // Reusable dequant buffer — avoids per-block [0i32; 64] zeroing
        let mut dequant_i32 = [0i32; DCT_BLOCK_SIZE];

        // IDCT strip 0 into ext_a
        idct_chroma_into_ext(
            &mut ext_cb_a,
            &self.coeffs[1],
            &self.coeff_counts[1],
            &comp_infos[1],
            quant_cb,
            0,
            c_strip_width,
            c_strip_height,
            chroma_height_total,
            idct_fn,
            &mut dequant_i32,
        );
        idct_chroma_into_ext(
            &mut ext_cr_a,
            &self.coeffs[2],
            &self.coeff_counts[2],
            &comp_infos[2],
            quant_cr,
            0,
            c_strip_width,
            c_strip_height,
            chroma_height_total,
            idct_fn,
            &mut dequant_i32,
        );

        // IDCT strip 1 into ext_b (if exists)
        if mcu_rows > 1 {
            idct_chroma_into_ext(
                &mut ext_cb_b,
                &self.coeffs[1],
                &self.coeff_counts[1],
                &comp_infos[1],
                quant_cb,
                1,
                c_strip_width,
                c_strip_height,
                chroma_height_total,
                idct_fn,
                &mut dequant_i32,
            );
            idct_chroma_into_ext(
                &mut ext_cr_b,
                &self.coeffs[2],
                &self.coeff_counts[2],
                &comp_infos[2],
                quant_cr,
                1,
                c_strip_width,
                c_strip_height,
                chroma_height_total,
                idct_fn,
                &mut dequant_i32,
            );
        }

        // Set above context for first strip: edge replication (copy first data row)
        ext_cb_a.copy_within(c_strip_width..2 * c_strip_width, 0);
        ext_cr_a.copy_within(c_strip_width..2 * c_strip_width, 0);

        // Pre-allocate scratch buffer for upsample — reused across all MCU rows
        // to avoid re-zeroing [0i16; 4096] on every upsample call (saves ~3M instr/decode).
        // Only used for h2v2 triangle-filter path; the scratch is written by the vertical
        // pass before the horizontal pass reads it, so it doesn't need re-zeroing.
        // Note: scratch upsample was only used for removed SeparableBiased path

        // Horizontal chroma padding fixup (same as scan.rs streaming path)
        let downsampled_w = (width + h_ratio - 1) / h_ratio;
        let has_h_padding = downsampled_w < c_strip_width;
        let fixup_h_padding = |buf: &mut [i16]| {
            if !has_h_padding {
                return;
            }
            let total_rows = ext_height;
            for row in 0..total_rows {
                let row_off = row * c_strip_width;
                let last_val = buf[row_off + downsampled_w - 1];
                for col in downsampled_w..c_strip_width {
                    buf[row_off + col] = last_val;
                }
            }
        };

        for imcu_row in 0..mcu_rows {
            // Set below context for current strip (ext_a)
            let last_data_row_start = c_strip_height * c_strip_width; // row c_strip_height in ext
            let below_ctx_start = (c_strip_height + 1) * c_strip_width; // row c_strip_height+1
            if imcu_row < mcu_rows - 1 {
                // Below context = first data row of next strip (ext_b row 1)
                let src_start = c_strip_width; // row 1 of ext_b
                ext_cb_a[below_ctx_start..below_ctx_start + c_strip_width]
                    .copy_from_slice(&ext_cb_b[src_start..src_start + c_strip_width]);
                ext_cr_a[below_ctx_start..below_ctx_start + c_strip_width]
                    .copy_from_slice(&ext_cr_b[src_start..src_start + c_strip_width]);
            } else {
                // Last strip: edge-replicate from the last REAL chroma row,
                // not the last padding row. Padding rows have IDCT rounding
                // differences vs the last real row (matching libjpeg-turbo's
                // set_bottom_pointers behavior).
                let downsampled_h = (height + v_ratio - 1) / v_ratio;
                let real_rows =
                    c_strip_height.min(downsampled_h.saturating_sub(imcu_row * c_strip_height));
                // Overwrite padding data rows with last real row
                if real_rows < c_strip_height {
                    // Data rows start at offset c_strip_width (row 1 in extended buffer)
                    let last_real_start = real_rows * c_strip_width; // in ext: row (real_rows)
                    for pad_row in real_rows..c_strip_height {
                        let dst = (1 + pad_row) * c_strip_width;
                        ext_cb_a.copy_within(last_real_start..last_real_start + c_strip_width, dst);
                        ext_cr_a.copy_within(last_real_start..last_real_start + c_strip_width, dst);
                    }
                }
                // Below context = last real data row (now also at last_data_row_start after fixup)
                ext_cb_a.copy_within(
                    last_data_row_start..last_data_row_start + c_strip_width,
                    below_ctx_start,
                );
                ext_cr_a.copy_within(
                    last_data_row_start..last_data_row_start + c_strip_width,
                    below_ctx_start,
                );
            }

            // IDCT Y blocks (full resolution)
            idct_comp_mcu_row(
                &self.coeffs[0],
                &self.coeff_counts[0],
                &comp_infos[0],
                quant_y,
                imcu_row,
                &mut y_strip,
                y_strip_width,
                idct_fn,
                &mut dequant_i32,
            );

            let y_rows_this_mcu = y_strip_height.min(height.saturating_sub(imcu_row * mcu_height));
            let y_start = imcu_row * mcu_height;

            if !needs_full_upsample {
                // NearestNeighbor 4:2:0: fused box-filter path
                // Read chroma from ext_a data region (rows 1..c_strip_height+1)
                let c_rows_this_mcu = c_strip_height
                    .min((height.saturating_sub(imcu_row * mcu_height) + v_ratio - 1) / v_ratio);
                let c_cols = (y_cols_this_image + 1) / 2;

                for row in 0..y_rows_this_mcu {
                    let y_offset = row * y_strip_width;
                    let c_row = (row / 2).min(c_rows_this_mcu.saturating_sub(1));
                    // +1 to skip the above-context row in the extended buffer
                    let c_offset = (1 + c_row) * c_strip_width;
                    let rgb_offset = (y_start + row) * width * 3;

                    fused_h2v2_box_ycbcr_to_rgb_u8(
                        &y_strip[y_offset..y_offset + y_cols_this_image],
                        &ext_cb_a[c_offset..c_offset + c_cols],
                        &ext_cr_a[c_offset..c_offset + c_cols],
                        &mut rgb[rgb_offset..rgb_offset + y_cols_this_image * 3],
                        y_cols_this_image,
                    );
                }
            } else {
                // Upsample extended strip → upsampled output buffer
                fixup_h_padding(&mut ext_cb_a);
                fixup_h_padding(&mut ext_cr_a);
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

                // Use upsampled rows starting at offset v_ratio (skip context rows)
                for row in 0..y_rows_this_mcu {
                    let strip_offset = row * y_strip_width;
                    let up_row = v_ratio + row; // skip the v_ratio context output rows
                    let chroma_offset = up_row * y_strip_width;
                    let rgb_offset = (y_start + row) * width * 3;

                    ycbcr_planes_i16_to_rgb_u8(
                        &y_strip[strip_offset..strip_offset + y_cols_this_image],
                        &cb_up[chroma_offset..chroma_offset + y_cols_this_image],
                        &cr_up[chroma_offset..chroma_offset + y_cols_this_image],
                        &mut rgb[rgb_offset..rgb_offset + y_cols_this_image * 3],
                    );
                }
            }

            // Prepare for next iteration: swap buffers
            if imcu_row + 1 < mcu_rows {
                // ext_b's above context = last data row of ext_a
                let last_data_start = c_strip_height * c_strip_width;
                ext_cb_b[..c_strip_width]
                    .copy_from_slice(&ext_cb_a[last_data_start..last_data_start + c_strip_width]);
                ext_cr_b[..c_strip_width]
                    .copy_from_slice(&ext_cr_a[last_data_start..last_data_start + c_strip_width]);

                // Swap: ext_b becomes ext_a (current), ext_a becomes ext_b (free for next IDCT)
                core::mem::swap(&mut ext_cb_a, &mut ext_cb_b);
                core::mem::swap(&mut ext_cr_a, &mut ext_cr_b);

                // IDCT the strip after next into the now-free ext_b
                if imcu_row + 2 < mcu_rows {
                    idct_chroma_into_ext(
                        &mut ext_cb_b,
                        &self.coeffs[1],
                        &self.coeff_counts[1],
                        &comp_infos[1],
                        quant_cb,
                        imcu_row + 2,
                        c_strip_width,
                        c_strip_height,
                        chroma_height_total,
                        idct_fn,
                        &mut dequant_i32,
                    );
                    idct_chroma_into_ext(
                        &mut ext_cr_b,
                        &self.coeffs[2],
                        &self.coeff_counts[2],
                        &comp_infos[2],
                        quant_cr,
                        imcu_row + 2,
                        c_strip_width,
                        c_strip_height,
                        chroma_height_total,
                        idct_fn,
                        &mut dequant_i32,
                    );
                }
            }
        }

        Ok(rgb)
    }

    // =========================================================================
    // Shared helpers for f32 decode paths
    // =========================================================================

    /// Set up MCU grid, component info, bias tracking, and f32 planes
    /// for all three f32 output paths (to_pixels, to_pixels_f32, to_ycbcr).
    ///
    /// Returns (comp_infos, bias_stats, component_biases, comp_planes_f32,
    /// mcu_rows, max_h_samp, max_v_samp).
    fn setup_f32_decode(
        &self,
        num_components: usize,
    ) -> Result<(
        Vec<CompInfo>,
        DequantBiasStats,
        Vec<[f32; DCT_BLOCK_SIZE]>,
        Vec<Vec<f32>>,
        usize,
        u8,
        u8,
    )> {
        let width = self.width as usize;
        let height = self.height as usize;

        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..num_components {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            num_components,
        )?;

        let bias_stats = DequantBiasStats::new(num_components);
        let component_biases = vec![[0.0f32; DCT_BLOCK_SIZE]; num_components];

        let mut comp_planes_f32 = Vec::with_capacity(num_components);
        for info in &comp_infos {
            let size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; size]);
        }

        Ok((
            comp_infos,
            bias_stats,
            component_biases,
            comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ))
    }

    /// Gather bias statistics for full-res components in a single MCU row.
    fn gather_bias_stats(
        &self,
        imcu_row: usize,
        comp_idx: usize,
        info: &CompInfo,
        bias_stats: &mut DequantBiasStats,
    ) {
        if !info.is_full_res {
            return;
        }
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
                bias_stats.gather_block(comp_idx, &natural_coeffs);
            }
        }
    }

    /// Apply deblocking filters to component planes in-place.
    ///
    /// Planes are centered around 0 (IDCT output: [-128, 127]). Boundary 4-tap
    /// expects [0, 255], so we level-shift before filtering and shift back after.
    /// Knusperli replaces the plane entirely (it does its own IDCT from coefficients).
    fn apply_deblock_to_planes(
        &self,
        comp_planes: &mut [Vec<f32>],
        comp_infos: &[CompInfo],
        mode: super::super::DeblockMode,
    ) -> Result<()> {
        use super::super::DeblockMode;

        if mode == DeblockMode::Off {
            return Ok(());
        }

        for (comp_idx, info) in comp_infos.iter().enumerate() {
            let quant = self.quant_tables[info.quant_idx]
                .as_ref()
                .ok_or(Error::internal("missing quant table for deblock"))?;
            let dc_quant = quant[0];

            // Decide strategy per-component
            let use_knusperli = match mode {
                DeblockMode::Off => unreachable!(),
                DeblockMode::Knusperli => true,
                DeblockMode::Boundary4Tap | DeblockMode::AutoStreamable => false,
                DeblockMode::Auto => {
                    // Simple heuristic: knusperli at low Q (high DC quant), boundary otherwise
                    dc_quant >= 27
                }
            };

            if use_knusperli && comp_idx < self.coeffs.len() {
                // Knusperli: replace plane with its own IDCT + boundary correction.
                // Output is [0, 255] range; shift to [-128, 127] to match pipeline.
                // Coefficients are stored as Vec<[i16; 64]>; flatten to &[i16] for knusperli.
                let flat_coeffs: &[i16] = bytemuck::cast_slice(&self.coeffs[comp_idx]);
                let mut plane = crate::deblock::knusperli::process_component(
                    flat_coeffs,
                    info.comp_blocks_h,
                    info.comp_blocks_v,
                    quant,
                );
                // Level-shift from [0, 255] to [-128, 127] to match the rest of the pipeline
                for v in &mut plane {
                    *v -= 128.0;
                }
                comp_planes[comp_idx] = plane;
            } else {
                // Boundary 4-tap: level-shift to [0, 255], filter in-place, shift back.
                let plane = &mut comp_planes[comp_idx];
                let (w, h) = (info.comp_width, info.comp_height);

                // Shift to [0, 255]
                for v in plane.iter_mut() {
                    *v += 128.0;
                }

                let strength = crate::deblock::BoundaryStrength::from_dc_quant(dc_quant);
                crate::deblock::filter_plane_boundary_4tap(plane, w, h, strength);

                // Shift back to [-128, 127]
                for v in plane.iter_mut() {
                    *v -= 128.0;
                }
            }
        }

        Ok(())
    }

    /// Upsample component f32 planes to full image resolution.
    ///
    /// Handles all chroma upsampling modes (Triangle, Jpegli, NearestNeighbor).
    /// Full-res components are clipped to image dimensions without interpolation.
    fn upsample_planes_f32(
        &self,
        comp_planes_f32: &[Vec<f32>],
        comp_infos: &[CompInfo],
        max_h_samp: u8,
        max_v_samp: u8,
        chroma_upsampling: super::super::ChromaUpsampling,
    ) -> Result<Vec<Vec<f32>>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32 = Vec::with_capacity(comp_infos.len());

        for (comp_idx, info) in comp_infos.iter().enumerate() {
            let comp_plane = &comp_planes_f32[comp_idx];

            let plane = if info.h_samp < max_h_samp as usize || info.v_samp < max_v_samp as usize {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                match chroma_upsampling {
                    super::super::ChromaUpsampling::Triangle => upsample_libjpeg_f32(
                        comp_plane,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    ),
                    super::super::ChromaUpsampling::NearestNeighbor => {
                        let mut upsampled = vec![0.0f32; output_size];
                        upsample_nearest_f32(
                            comp_plane,
                            info.comp_width,
                            info.comp_height,
                            &mut upsampled,
                            width,
                            height,
                            scale_x,
                            scale_y,
                        );
                        upsampled
                    }
                }
            } else {
                // Full resolution — clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    let src = &comp_plane[py * info.comp_width..py * info.comp_width + width];
                    let dst = &mut plane[py * width..py * width + width];
                    dst.copy_from_slice(src);
                }
                plane
            };

            planes_f32.push(plane);
        }

        Ok(planes_f32)
    }

    // =========================================================================
    // Public output methods
    // =========================================================================

    /// Convert decoded coefficients to pixels in the requested format.
    ///
    /// This is the main entry point for pixel output. It selects the fastest
    /// available path based on the image characteristics:
    ///
    /// 1. **Streaming result** (`streaming_rgb`): Already decoded during `parse_scan()`
    ///    in a single entropy→IDCT→color pass. Zero-copy for RGB, reformat for others.
    /// 2. **Fused parallel result** (`fused_result`): Entropy+IDCT+color per restart
    ///    segment via rayon. Activated by `DecodeMode::Auto` + DRI + `parallel` feature.
    /// 3. **i16 fast path** (`to_pixels_fast_i16`/`_subsampled`): Integer IDCT from
    ///    buffered coefficients. Used for non-XYB, non-bias RGB output.
    /// 4. **f32 generic path**: Full f32 pipeline with dequant bias, XYB support.
    ///
    /// Paths 1-2 produce results during entropy decode (no separate output pass).
    /// Wave-parallel scanline decode is handled by `ScanlineReader`, not here.
    ///
    /// The `stop` parameter allows cancellation of long-running operations.
    #[allow(clippy::wrong_self_convention)] // Takes &mut self to take() internal buffer
    pub(in crate::decode) fn to_pixels(
        &mut self,
        format: PixelFormat,
        is_xyb: bool,
        chroma_upsampling: super::super::ChromaUpsampling,
        output_target: super::super::OutputTarget,
        _stop: &impl Stop,
    ) -> Result<Vec<u8>> {
        let dequant_bias = output_target.uses_dequant_bias();
        let width = self.width as usize;
        let height = self.height as usize;

        // If streaming decode was used, return its result directly (zero-copy for Rgb,
        // reformat for Bgr/Rgba/Bgra/Bgrx)
        if is_rgb_family_u8(format)
            && !is_xyb
            && let Some(rgb) = self.streaming_rgb.take()
        {
            return reformat_rgb_output(rgb, format, width, height);
        }

        // If fused parallel decode was used, return its result
        #[cfg(feature = "parallel")]
        if is_rgb_family_u8(format)
            && !is_xyb
            && !dequant_bias
            && let Some(fused) = self.fused_result.take()
        {
            use super::super::fused_parallel::FusedResult;
            let FusedResult(rgb) = fused;
            return reformat_rgb_output(rgb, format, width, height);
        }

        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        // Try parallel fast integer paths first (fall through to sequential if image too small)
        #[cfg(feature = "parallel")]
        if self.num_threads != 1
            && !dequant_bias
            && self.can_use_fast_i16_path(format, is_xyb)
            && let Some(rgb) = self.to_pixels_fast_i16_parallel(chroma_upsampling)?
        {
            return reformat_rgb_output(rgb, format, width, height);
        }
        #[cfg(feature = "parallel")]
        if self.num_threads != 1
            && !dequant_bias
            && self.can_use_fast_i16_subsampled(format, is_xyb)
            && let Some(rgb) = self.to_pixels_fast_i16_subsampled_parallel(chroma_upsampling)?
        {
            return reformat_rgb_output(rgb, format, width, height);
        }

        // Try fast integer path for non-XYB 4:4:4 images
        // (dequant_bias requires f32 path for fractional bias application)
        if !dequant_bias && self.can_use_fast_i16_path(format, is_xyb) {
            let rgb = self.to_pixels_fast_i16(chroma_upsampling)?;
            return reformat_rgb_output(rgb, format, width, height);
        }

        // Try fast integer path for subsampled images (4:2:0, 4:2:2, 4:4:0)
        if !dequant_bias && self.can_use_fast_i16_subsampled(format, is_xyb) {
            let rgb = self.to_pixels_fast_i16_subsampled(chroma_upsampling)?;
            return reformat_rgb_output(rgb, format, width, height);
        }

        let num_components = self.num_components as usize;
        let (
            comp_infos,
            mut bias_stats,
            mut component_biases,
            mut comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ) = self.setup_f32_decode(num_components)?;

        // Process MCU row by MCU row (matching C++ incremental bias recomputation)
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..num_components {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                self.gather_bias_stats(imcu_row, comp_idx, info, &mut bias_stats);
                if info.is_full_res && imcu_row % 4 == 3 {
                    component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                }

                // IDCT for this component in this MCU row
                let biases = &component_biases[comp_idx];
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

                        if is_xyb || dequant_bias || self.force_f32_idct {
                            // f32 IDCT path: XYB needs extended gamut precision,
                            // dequant_bias needs fractional bias application,
                            // dimension-swapping transforms need symmetric IDCT
                            let dequant = if dequant_bias && !is_xyb {
                                dequantize_block_with_bias(&natural_coeffs, quant, biases)
                            } else {
                                dequantize_block(&natural_coeffs, quant)
                            };
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
                            match chroma_upsampling {
                                super::super::ChromaUpsampling::Triangle => {
                                    idct_int_libjpeg(&mut dequant_i32, &mut pixels_i16, 8);
                                }
                                _ => {
                                    idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8);
                                }
                            }

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

        // Upsample and convert to output format
        let planes_f32 = self.upsample_planes_f32(
            &comp_planes_f32,
            &comp_infos,
            max_h_samp,
            max_v_samp,
            chroma_upsampling,
        )?;

        let output_size = checked_size_2d(width, height)?;

        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and convert to u8
                let mut output = vec![0u8; output_size];
                gray_f32_to_gray_u8(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, f) if is_rgb_family_u8(f) => {
                // Grayscale → RGB-family: produce RGB, then reformat
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];
                gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb);
                reformat_rgb_output(rgb, f, width, height)
            }
            (3, f) if is_rgb_family_u8(f) => {
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
                } else if self.is_rgb_jpeg() {
                    // RGB JPEG (Adobe APP14 transform=0 or RGB component IDs):
                    // Level-shift from [-128..127] to [0..255] and interleave.
                    for i in 0..output_size {
                        let r = (planes_f32[0][i] + 128.0).round().clamp(0.0, 255.0) as u8;
                        let g = (planes_f32[1][i] + 128.0).round().clamp(0.0, 255.0) as u8;
                        let b = (planes_f32[2][i] + 128.0).round().clamp(0.0, 255.0) as u8;
                        rgb[i * 3] = r;
                        rgb[i * 3 + 1] = g;
                        rgb[i * 3 + 2] = b;
                    }
                } else {
                    // YCbCr to RGB conversion using batch function
                    ycbcr_planes_f32_to_rgb_u8(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                }
                reformat_rgb_output(rgb, f, width, height)
            }
            (4, f) if is_rgb_family_u8(f) => {
                // CMYK or YCCK → RGB, then reformat
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];

                // Check Adobe transform to determine conversion type
                // YCCK (transform=2) uses YCbCr→CMY then applies K
                // CMYK (transform=0 or absent) uses raw CMYK with Adobe inversion
                match self.adobe_transform {
                    Some(AdobeColorTransform::Ycck) => {
                        // YCCK: YCbCr channels + K
                        ycck_planes_to_rgb_u8(
                            &planes_f32[0],
                            &planes_f32[1],
                            &planes_f32[2],
                            &planes_f32[3],
                            &mut rgb,
                        );
                    }
                    _ => {
                        // CMYK (Adobe inverted format)
                        cmyk_planes_to_rgb_u8(
                            &planes_f32[0],
                            &planes_f32[1],
                            &planes_f32[2],
                            &planes_f32[3],
                            &mut rgb,
                        );
                    }
                }
                reformat_rgb_output(rgb, f, width, height)
            }
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// Convert decoded coefficients to f32 pixels.
    /// Values are normalized to range 0.0-1.0.
    ///
    /// The `stop` parameter allows cancellation of long-running operations.
    pub(in crate::decode) fn to_pixels_f32(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        chroma_upsampling: super::super::ChromaUpsampling,
        _stop: &impl Stop,
    ) -> Result<Vec<f32>> {
        self.to_pixels_f32_inner(
            format,
            is_xyb,
            chroma_upsampling,
            super::super::DeblockMode::Off,
            _stop,
        )
    }

    /// Internal f32 pixel conversion with optional deblocking.
    ///
    /// When `deblock_mode` is not `Off`, deblocking is applied to component
    /// planes after IDCT but before chroma upsampling and color conversion.
    pub(in crate::decode) fn to_pixels_f32_deblock(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        chroma_upsampling: super::super::ChromaUpsampling,
        deblock_mode: super::super::DeblockMode,
        _stop: &impl Stop,
    ) -> Result<Vec<f32>> {
        self.to_pixels_f32_inner(format, is_xyb, chroma_upsampling, deblock_mode, _stop)
    }

    fn to_pixels_f32_inner(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        chroma_upsampling: super::super::ChromaUpsampling,
        deblock_mode: super::super::DeblockMode,
        _stop: &impl Stop,
    ) -> Result<Vec<f32>> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        let num_components = self.num_components as usize;
        let (
            comp_infos,
            mut bias_stats,
            mut component_biases,
            mut comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ) = self.setup_f32_decode(num_components)?;

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..num_components {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                self.gather_bias_stats(imcu_row, comp_idx, info, &mut bias_stats);
                if info.is_full_res && imcu_row % 4 == 3 {
                    component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                }

                // IDCT for this component — always f32 for f32 output
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

        // Apply deblocking to component planes (between IDCT and upsampling).
        // Deblocking operates in [0, 255] pixel domain, so we level-shift
        // the centered IDCT output (+128), filter, then shift back (-128).
        self.apply_deblock_to_planes(&mut comp_planes_f32, &comp_infos, deblock_mode)?;

        // Upsample and convert to output format
        let planes_f32 = self.upsample_planes_f32(
            &comp_planes_f32,
            &comp_infos,
            max_h_samp,
            max_v_samp,
            chroma_upsampling,
        )?;

        let width = self.width as usize;
        let height = self.height as usize;
        let output_size = checked_size_2d(width, height)?;

        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
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
                } else if self.is_rgb_jpeg() {
                    // RGB JPEG: level-shift and normalize to [0.0, 1.0]
                    for i in 0..output_size {
                        rgb[i * 3] = (planes_f32[0][i] + 128.0) / 255.0;
                        rgb[i * 3 + 1] = (planes_f32[1][i] + 128.0) / 255.0;
                        rgb[i * 3 + 2] = (planes_f32[2][i] + 128.0) / 255.0;
                    }
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
        chroma_upsampling: super::super::ChromaUpsampling,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        if self.num_components != 3 {
            return Err(Error::unsupported_feature(
                "YCbCr planes require 3-component image",
            ));
        }

        let num_components = self.num_components as usize;
        let (
            comp_infos,
            mut bias_stats,
            mut component_biases,
            mut comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ) = self.setup_f32_decode(num_components)?;

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..num_components {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                self.gather_bias_stats(imcu_row, comp_idx, info, &mut bias_stats);
                if info.is_full_res && imcu_row % 4 == 3 {
                    component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                }

                // IDCT — integer path for YCbCr output
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
                        match chroma_upsampling {
                            super::super::ChromaUpsampling::Triangle => {
                                idct_int_libjpeg(&mut dequant_i32, &mut pixels_i16, 8);
                            }
                            _ => {
                                idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8);
                            }
                        }

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

        // Upsample chroma and return planes
        let mut planes_f32 = self.upsample_planes_f32(
            &comp_planes_f32,
            &comp_infos,
            max_h_samp,
            max_v_samp,
            chroma_upsampling,
        )?;

        Ok((
            core::mem::take(&mut planes_f32[0]),
            core::mem::take(&mut planes_f32[1]),
            core::mem::take(&mut planes_f32[2]),
        ))
    }
}
