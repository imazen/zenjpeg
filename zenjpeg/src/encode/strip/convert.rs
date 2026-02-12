//! Color conversion and strip processing methods for StripProcessor.
//!
//! This module contains methods for:
//! - RGB to YCbCr conversion
//! - RGB to XYB conversion
//! - YCbCr input handling (f32 direct input)
//! - Chroma downsampling
//! - Strip padding (horizontal and vertical)

#![allow(dead_code)]

use crate::encode::encoder_types::DownsamplingMethod;
use crate::error::Result;
use crate::types::{PixelFormat, Subsampling};

use super::StripProcessor;

impl StripProcessor {
    /// Copies YCbCr f32 data to strip buffers with level shift.
    ///
    /// Converts from centered [-128, 127] to JPEG range [0, 255].
    pub(super) fn copy_ycbcr_to_strips(
        &mut self,
        y_row: &[f32],
        cb_row: &[f32],
        cr_row: &[f32],
        strip_height: usize,
    ) -> Result<()> {
        let width = self.layout.width;
        let padded_width = self.layout.padded_width;

        // Validate input sizes
        let expected_y_size = strip_height * width;
        if y_row.len() < expected_y_size {
            return Err(crate::error::Error::internal("Y plane too small for strip"));
        }

        if !self.pixel_format.is_grayscale()
            && (cb_row.len() < expected_y_size || cr_row.len() < expected_y_size)
        {
            return Err(crate::error::Error::internal(
                "Cb/Cr planes too small for strip",
            ));
        }

        // Copy Y with level shift and padded stride
        for row in 0..strip_height {
            let src_start = row * width;
            let dst_start = row * padded_width;

            // Copy and level-shift Y values
            for x in 0..width {
                self.y_strip[dst_start + x] = y_row[src_start + x] + 128.0;
            }

            // Edge-pad Y row
            if width < padded_width {
                let edge_val = self.y_strip[dst_start + width - 1];
                for x in width..padded_width {
                    self.y_strip[dst_start + x] = edge_val;
                }
            }
        }

        // Copy Cb/Cr with level shift (no padding, full resolution)
        if !self.pixel_format.is_grayscale() {
            let num_pixels = strip_height * width;
            for i in 0..num_pixels {
                self.cb_strip[i] = cb_row[i] + 128.0;
                self.cr_strip[i] = cr_row[i] + 128.0;
            }
        }

        Ok(())
    }

    /// Copies pre-downsampled YCbCr f32 data to strip buffers.
    ///
    /// Y goes to y_strip with level shift.
    /// Cb/Cr go directly to cb_down/cr_down (already downsampled).
    pub(super) fn copy_ycbcr_subsampled_to_strips(
        &mut self,
        y_row: &[f32],
        cb_row: &[f32],
        cr_row: &[f32],
        strip_height: usize,
    ) -> Result<()> {
        let width = self.layout.width;
        let padded_width = self.layout.padded_width;

        let chroma_width = self.layout.c_width;
        let chroma_height = self.layout.c_strip_height_for(strip_height);

        // Validate input sizes
        let expected_y_size = strip_height * width;
        if y_row.len() < expected_y_size {
            return Err(crate::error::Error::internal("Y plane too small for strip"));
        }

        let expected_chroma_size = chroma_width * chroma_height;
        if !self.pixel_format.is_grayscale()
            && (cb_row.len() < expected_chroma_size || cr_row.len() < expected_chroma_size)
        {
            return Err(crate::error::Error::internal(
                "Cb/Cr planes too small for subsampled strip",
            ));
        }

        // Copy Y with level shift and padded stride
        for row in 0..strip_height {
            let src_start = row * width;
            let dst_start = row * padded_width;

            for x in 0..width {
                self.y_strip[dst_start + x] = y_row[src_start + x] + 128.0;
            }

            if width < padded_width {
                let edge_val = self.y_strip[dst_start + width - 1];
                for x in width..padded_width {
                    self.y_strip[dst_start + x] = edge_val;
                }
            }
        }

        // Copy Cb/Cr directly to downsampled buffers with level shift
        if !self.pixel_format.is_grayscale() {
            for i in 0..expected_chroma_size {
                self.cb_down[i] = cb_row[i] + 128.0;
                self.cr_down[i] = cr_row[i] + 128.0;
            }
        }

        Ok(())
    }

    /// Pads chroma downsampled buffers vertically for partial bottom strips.
    pub(super) fn pad_chroma_down_vertically(&mut self, actual_height: usize) -> Result<()> {
        let chroma_width = self.layout.c_width;
        let target_height = self.layout.c_strip_height;
        let actual_chroma_height = self.layout.c_strip_height_for(actual_height);

        if actual_chroma_height >= target_height {
            return Ok(());
        }

        // Replicate last row - copy to temp first to avoid borrow conflict
        let last_row_start = (actual_chroma_height - 1) * chroma_width;
        let cb_last_row: Vec<f32> =
            self.cb_down[last_row_start..last_row_start + chroma_width].to_vec();
        let cr_last_row: Vec<f32> =
            self.cr_down[last_row_start..last_row_start + chroma_width].to_vec();
        for row in actual_chroma_height..target_height {
            let dst_start = row * chroma_width;
            self.cb_down[dst_start..dst_start + chroma_width].copy_from_slice(&cb_last_row);
            self.cr_down[dst_start..dst_start + chroma_width].copy_from_slice(&cr_last_row);
        }

        Ok(())
    }

    /// Converts RGB strip data to YCbCr in the strip buffers.
    ///
    /// Uses strided SIMD conversion that writes Y directly with padded stride,
    /// eliminating the need for a separate rearrange pass.
    pub(super) fn convert_strip_to_ycbcr(
        &mut self,
        rgb_strip: &[u8],
        strip_height: usize,
    ) -> Result<()> {
        let width = self.layout.width;
        let padded_width = self.layout.padded_width;
        let num_pixels = strip_height * width;
        let y_size = strip_height * padded_width;

        match self.pixel_format {
            PixelFormat::Rgb | PixelFormat::Rgba => {
                let bpp = self.pixel_format.bytes_per_pixel();
                // Use reuse version with pre-allocated buffers when yuv feature is enabled
                #[cfg(feature = "yuv")]
                {
                    crate::encode_simd::rgb_to_ycbcr_strided_reuse(
                        rgb_strip,
                        &mut self.y_strip[..y_size],
                        &mut self.cb_strip[..num_pixels],
                        &mut self.cr_strip[..num_pixels],
                        &mut self.yuv_temp_y[..num_pixels],
                        &mut self.yuv_temp_cb[..num_pixels],
                        &mut self.yuv_temp_cr[..num_pixels],
                        width,
                        strip_height,
                        padded_width,
                        bpp,
                    );
                }
                #[cfg(not(feature = "yuv"))]
                {
                    crate::encode_simd::rgb_to_ycbcr_strided_inplace(
                        rgb_strip,
                        &mut self.y_strip[..y_size],
                        &mut self.cb_strip[..num_pixels],
                        &mut self.cr_strip[..num_pixels],
                        width,
                        strip_height,
                        padded_width,
                        bpp,
                    );
                }
            }
            PixelFormat::Bgr => {
                let bpp = self.pixel_format.bytes_per_pixel();
                // Use reuse version with pre-allocated buffers when yuv feature is enabled
                #[cfg(feature = "yuv")]
                {
                    crate::encode_simd::bgr_to_ycbcr_strided_reuse(
                        rgb_strip,
                        &mut self.y_strip[..y_size],
                        &mut self.cb_strip[..num_pixels],
                        &mut self.cr_strip[..num_pixels],
                        &mut self.yuv_temp_y[..num_pixels],
                        &mut self.yuv_temp_cb[..num_pixels],
                        &mut self.yuv_temp_cr[..num_pixels],
                        width,
                        strip_height,
                        padded_width,
                        bpp,
                    );
                }
                #[cfg(not(feature = "yuv"))]
                {
                    crate::encode_simd::bgr_to_ycbcr_strided_inplace(
                        rgb_strip,
                        &mut self.y_strip[..y_size],
                        &mut self.cb_strip[..num_pixels],
                        &mut self.cr_strip[..num_pixels],
                        width,
                        strip_height,
                        padded_width,
                        bpp,
                    );
                }
            }
            PixelFormat::Gray => {
                // Grayscale: write Y with strided layout directly
                for row in 0..strip_height {
                    let src_start = row * width;
                    let dst_start = row * padded_width;
                    for x in 0..width {
                        self.y_strip[dst_start + x] = rgb_strip[src_start + x] as f32;
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[dst_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[dst_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::Cmyk => {
                // CMYK: scalar conversion with strided Y output
                use crate::foundation::consts::{
                    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR,
                    YCBCR_G_TO_Y, YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
                };
                let bpp = self.pixel_format.bytes_per_pixel();
                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    for x in 0..width {
                        let idx = (row * width + x) * bpp;

                        let c = rgb_strip[idx] as f32 / 255.0;
                        let m = rgb_strip[idx + 1] as f32 / 255.0;
                        let y_val = rgb_strip[idx + 2] as f32 / 255.0;
                        let k = rgb_strip[idx + 3] as f32 / 255.0;

                        let r = 255.0 * (1.0 - c) * (1.0 - k);
                        let g = 255.0 * (1.0 - m) * (1.0 - k);
                        let b = 255.0 * (1.0 - y_val) * (1.0 - k);

                        // Use FMA for accuracy (single rounding)
                        self.y_strip[y_row_start + x] =
                            YCBCR_R_TO_Y.mul_add(r, YCBCR_G_TO_Y.mul_add(g, YCBCR_B_TO_Y * b));
                        self.cb_strip[cbcr_row_start + x] = YCBCR_R_TO_CB
                            .mul_add(r, YCBCR_G_TO_CB.mul_add(g, YCBCR_B_TO_CB.mul_add(b, 128.0)));
                        self.cr_strip[cbcr_row_start + x] = YCBCR_R_TO_CR
                            .mul_add(r, YCBCR_G_TO_CR.mul_add(g, YCBCR_B_TO_CR.mul_add(b, 128.0)));
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[y_row_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[y_row_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::Bgra | PixelFormat::Bgrx => {
                // BGRA/BGRX: fast path - 4 bytes per pixel, alpha/padding ignored
                crate::encode_simd::bgr_to_ycbcr_strided_inplace(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    padded_width,
                    4, // BGRA/BGRX is 4 bytes per pixel
                );
            }
            // 16-bit and float formats: linear RGB input
            // Uses optimized LUT conversion: linear -> sRGB -> YCbCr
            // For HDR (values > 1.0), applies Reinhard tone mapping
            PixelFormat::Gray16 => {
                use super::super::linear_lut::{linear_u16_to_srgb_255, linear_u16_to_srgb_255_x8};

                // Gray16: 2 bytes per pixel, native endian, linear
                for row in 0..strip_height {
                    let src_start = row * width * 2;
                    let dst_start = row * padded_width;

                    // Process 8 pixels at a time with SIMD
                    let simd_width = width / 8 * 8;
                    for x in (0..simd_width).step_by(8) {
                        let idx = src_start + x * 2;
                        // Single unaligned load of 8 u16 values (16 bytes)
                        let values: [u16; 8] =
                            bytemuck::pod_read_unaligned(&rgb_strip[idx..idx + 16]);
                        let srgb = linear_u16_to_srgb_255_x8(values);
                        self.y_strip[dst_start + x..dst_start + x + 8]
                            .copy_from_slice(&srgb.to_array());
                    }

                    // Handle remainder with scalar
                    for x in simd_width..width {
                        let idx = src_start + x * 2;
                        let value = u16::from_ne_bytes([rgb_strip[idx], rgb_strip[idx + 1]]);
                        self.y_strip[dst_start + x] = linear_u16_to_srgb_255(value);
                    }

                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[dst_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[dst_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                use super::super::linear_lut::{linear_rgb16_to_ycbcr, linear_rgb16_to_ycbcr_x8};

                // RGB16/RGBA16: 6/8 bytes per pixel, native endian, linear
                // Uses SIMD for fast linear -> YCbCr conversion
                let bpp = self.pixel_format.bytes_per_pixel();

                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    let row_base = row * width * bpp;

                    // Process 8 pixels at a time with SIMD
                    let simd_width = width / 8 * 8;
                    for x in (0..simd_width).step_by(8) {
                        // Deinterleave 8 RGB pixels into separate R, G, B arrays
                        let mut r_arr = [0u16; 8];
                        let mut g_arr = [0u16; 8];
                        let mut b_arr = [0u16; 8];

                        for i in 0..8 {
                            let base = row_base + (x + i) * bpp;
                            let rgb: [u16; 3] =
                                bytemuck::pod_read_unaligned(&rgb_strip[base..base + 6]);
                            r_arr[i] = rgb[0];
                            g_arr[i] = rgb[1];
                            b_arr[i] = rgb[2];
                        }

                        let (y, cb, cr) = linear_rgb16_to_ycbcr_x8(r_arr, g_arr, b_arr);

                        self.y_strip[y_row_start + x..y_row_start + x + 8]
                            .copy_from_slice(&y.to_array());
                        self.cb_strip[cbcr_row_start + x..cbcr_row_start + x + 8]
                            .copy_from_slice(&cb.to_array());
                        self.cr_strip[cbcr_row_start + x..cbcr_row_start + x + 8]
                            .copy_from_slice(&cr.to_array());
                    }

                    // Handle remainder with scalar
                    for x in simd_width..width {
                        let base = row_base + x * bpp;
                        let rgb: [u16; 3] =
                            bytemuck::pod_read_unaligned(&rgb_strip[base..base + 6]);

                        let (y, cb, cr) = linear_rgb16_to_ycbcr(rgb[0], rgb[1], rgb[2]);
                        self.y_strip[y_row_start + x] = y;
                        self.cb_strip[cbcr_row_start + x] = cb;
                        self.cr_strip[cbcr_row_start + x] = cr;
                    }

                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[y_row_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[y_row_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::GrayF32 => {
                use super::super::linear_lut::{
                    linear_f32_to_srgb_255_fast, linear_to_srgb_255_x8,
                };
                use wide::f32x8;

                // GrayF32: 4 bytes per pixel, linear
                for row in 0..strip_height {
                    let src_start = row * width * 4;
                    let dst_start = row * padded_width;

                    // Process 8 pixels at a time with SIMD
                    let simd_width = width / 8 * 8;
                    for x in (0..simd_width).step_by(8) {
                        let idx = src_start + x * 4;
                        // Single unaligned load of 8 f32 values (32 bytes)
                        let values: [f32; 8] =
                            bytemuck::pod_read_unaligned(&rgb_strip[idx..idx + 32]);
                        let srgb = linear_to_srgb_255_x8(f32x8::new(values));
                        self.y_strip[dst_start + x..dst_start + x + 8]
                            .copy_from_slice(&srgb.to_array());
                    }

                    // Handle remainder with scalar
                    for x in simd_width..width {
                        let idx = src_start + x * 4;
                        let linear: f32 = bytemuck::pod_read_unaligned(&rgb_strip[idx..idx + 4]);
                        self.y_strip[dst_start + x] = linear_f32_to_srgb_255_fast(linear);
                    }

                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[dst_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[dst_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::RgbF32 | PixelFormat::RgbaF32 => {
                use super::super::linear_lut::{
                    linear_rgbf32_to_ycbcr_fast, linear_rgbf32_to_ycbcr_x8,
                };
                use wide::f32x8;

                // RgbF32/RgbaF32: 12/16 bytes per pixel, linear
                // Uses SIMD for fast linear -> YCbCr conversion
                let bpp = self.pixel_format.bytes_per_pixel();

                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    let row_base = row * width * bpp;

                    // Process 8 pixels at a time with SIMD
                    let simd_width = width / 8 * 8;
                    for x in (0..simd_width).step_by(8) {
                        // Deinterleave 8 RGB pixels into separate R, G, B vectors
                        let mut r_arr = [0.0f32; 8];
                        let mut g_arr = [0.0f32; 8];
                        let mut b_arr = [0.0f32; 8];

                        for i in 0..8 {
                            let base = row_base + (x + i) * bpp;
                            let rgb: [f32; 3] =
                                bytemuck::pod_read_unaligned(&rgb_strip[base..base + 12]);
                            r_arr[i] = rgb[0];
                            g_arr[i] = rgb[1];
                            b_arr[i] = rgb[2];
                        }

                        let (y, cb, cr) = linear_rgbf32_to_ycbcr_x8(
                            f32x8::new(r_arr),
                            f32x8::new(g_arr),
                            f32x8::new(b_arr),
                        );

                        self.y_strip[y_row_start + x..y_row_start + x + 8]
                            .copy_from_slice(&y.to_array());
                        self.cb_strip[cbcr_row_start + x..cbcr_row_start + x + 8]
                            .copy_from_slice(&cb.to_array());
                        self.cr_strip[cbcr_row_start + x..cbcr_row_start + x + 8]
                            .copy_from_slice(&cr.to_array());
                    }

                    // Handle remainder with scalar
                    for x in simd_width..width {
                        let base = row_base + x * bpp;
                        let rgb: [f32; 3] =
                            bytemuck::pod_read_unaligned(&rgb_strip[base..base + 12]);

                        let (y, cb, cr) = linear_rgbf32_to_ycbcr_fast(rgb[0], rgb[1], rgb[2]);
                        self.y_strip[y_row_start + x] = y;
                        self.cb_strip[cbcr_row_start + x] = cb;
                        self.cr_strip[cbcr_row_start + x] = cr;
                    }

                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[y_row_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[y_row_start + x] = edge_val;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Converts RGB strip to YCbCr 4:2:0 in a single fused pass.
    ///
    /// This is significantly faster than 444→downsample because:
    /// - Single pass through RGB data (better cache locality)
    /// - SIMD downsampling integrated into color conversion
    /// - No intermediate full-resolution Cb/Cr buffers
    ///
    /// Only available when `yuv` feature is enabled and for RGB/RGBA/BGR/BGRA formats.
    /// Other formats fall back to the standard 444+downsample path.
    ///
    /// Returns true if the fused path was used, false if caller should use standard path.
    #[cfg(feature = "yuv")]
    pub(super) fn convert_strip_to_ycbcr_420(
        &mut self,
        rgb_strip: &[u8],
        strip_height: usize,
    ) -> Result<bool> {
        let width = self.layout.width;
        let padded_width = self.layout.padded_width;
        let num_pixels = strip_height * width;
        let y_size = strip_height * padded_width;
        let c_width = self.layout.c_width;
        let c_height = self.layout.c_strip_height_for(strip_height);
        let c_size = c_width * c_height;

        match self.pixel_format {
            PixelFormat::Rgb | PixelFormat::Rgba => {
                let bpp = self.pixel_format.bytes_per_pixel();
                crate::color::fast_yuv::rgb_to_ycbcr_420_reuse(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    &mut self.yuv_temp_y[..num_pixels],
                    &mut self.yuv_temp_cb[..c_size],
                    &mut self.yuv_temp_cr[..c_size],
                    width,
                    strip_height,
                    padded_width,
                    bpp,
                );

                // Pad chroma down buffers horizontally
                self.pad_chroma_down_strip(c_height, c_width);
                Ok(true)
            }
            PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx => {
                let bpp = self.pixel_format.bytes_per_pixel();
                crate::color::fast_yuv::bgr_to_ycbcr_420_reuse(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    &mut self.yuv_temp_y[..num_pixels],
                    &mut self.yuv_temp_cb[..c_size],
                    &mut self.yuv_temp_cr[..c_size],
                    width,
                    strip_height,
                    padded_width,
                    bpp,
                );

                // Pad chroma down buffers horizontally
                self.pad_chroma_down_strip(c_height, c_width);
                Ok(true)
            }
            // Other formats: fall back to standard path
            _ => Ok(false),
        }
    }

    /// Converts RGB strip to scaled XYB color space.
    ///
    /// XYB layout in strip buffers:
    /// - y_strip: scaled X component (full res, padded stride)
    /// - cb_strip: scaled Y component (full res)
    /// - cr_strip: scaled B component (full res, before downsampling)
    /// - cb_down: scaled Y component (copied, full res but in downsampled buffer layout)
    /// - cr_down: scaled B component (2x2 downsampled)
    ///
    /// Note: XYB always uses fixed subsampling: X=1x1, Y=1x1, B=2x2
    pub(super) fn convert_strip_to_xyb(
        &mut self,
        rgb_strip: &[u8],
        strip_height: usize,
    ) -> Result<()> {
        use crate::color::xyb::srgb_to_scaled_xyb;

        let width = self.layout.width;
        let padded_width = self.layout.padded_width;
        let bpp = self.pixel_format.bytes_per_pixel();

        // XYB supports RGB formats (8-bit sRGB or linear float/16-bit)
        // Grayscale and CMYK are not supported
        match self.pixel_format {
            PixelFormat::Rgb
            | PixelFormat::Rgba
            | PixelFormat::Bgr
            | PixelFormat::Bgra
            | PixelFormat::Bgrx
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
            | PixelFormat::RgbF32
            | PixelFormat::RgbaF32 => {}
            PixelFormat::Gray | PixelFormat::Gray16 | PixelFormat::GrayF32 | PixelFormat::Cmyk => {
                return Err(crate::error::Error::unsupported_feature(
                    "XYB mode only supports RGB/RGBA pixel formats",
                ));
            }
        }

        // Convert RGB to scaled XYB
        // XYB values are stored as:
        // - X in y_strip with padded stride
        // - Y in cb_strip with packed stride
        // - B in cr_strip with packed stride
        for row in 0..strip_height {
            let y_row_start = row * padded_width;
            let cbcr_row_start = row * width;

            for x in 0..width {
                let src_idx = (row * width + x) * bpp;

                // Get linear RGB values based on pixel format
                let (r_linear, g_linear, b_linear): (f32, f32, f32) = match self.pixel_format {
                    // 8-bit sRGB: convert to linear first
                    PixelFormat::Rgb | PixelFormat::Rgba => {
                        let r = rgb_strip[src_idx];
                        let g = rgb_strip[src_idx + 1];
                        let b = rgb_strip[src_idx + 2];
                        // Use existing sRGB path (converts internally)
                        let (sx, sy, sb) = srgb_to_scaled_xyb(r, g, b);
                        // Store directly and continue
                        self.y_strip[y_row_start + x] = sx * 255.0;
                        self.cb_strip[cbcr_row_start + x] = sy * 255.0;
                        self.cr_strip[cbcr_row_start + x] = sb * 255.0;
                        continue;
                    }
                    PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx => {
                        let r = rgb_strip[src_idx + 2];
                        let g = rgb_strip[src_idx + 1];
                        let b = rgb_strip[src_idx];
                        let (sx, sy, sb) = srgb_to_scaled_xyb(r, g, b);
                        self.y_strip[y_row_start + x] = sx * 255.0;
                        self.cb_strip[cbcr_row_start + x] = sy * 255.0;
                        self.cr_strip[cbcr_row_start + x] = sb * 255.0;
                        continue;
                    }
                    // 16-bit linear: read and normalize to 0-1
                    PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                        let r = u16::from_ne_bytes([rgb_strip[src_idx], rgb_strip[src_idx + 1]])
                            as f32
                            / 65535.0;
                        let g = u16::from_ne_bytes([rgb_strip[src_idx + 2], rgb_strip[src_idx + 3]])
                            as f32
                            / 65535.0;
                        let b = u16::from_ne_bytes([rgb_strip[src_idx + 4], rgb_strip[src_idx + 5]])
                            as f32
                            / 65535.0;
                        (r, g, b)
                    }
                    // Float linear: read directly
                    PixelFormat::RgbF32 | PixelFormat::RgbaF32 => {
                        let r = f32::from_ne_bytes([
                            rgb_strip[src_idx],
                            rgb_strip[src_idx + 1],
                            rgb_strip[src_idx + 2],
                            rgb_strip[src_idx + 3],
                        ]);
                        let g = f32::from_ne_bytes([
                            rgb_strip[src_idx + 4],
                            rgb_strip[src_idx + 5],
                            rgb_strip[src_idx + 6],
                            rgb_strip[src_idx + 7],
                        ]);
                        let b = f32::from_ne_bytes([
                            rgb_strip[src_idx + 8],
                            rgb_strip[src_idx + 9],
                            rgb_strip[src_idx + 10],
                            rgb_strip[src_idx + 11],
                        ]);
                        (r, g, b)
                    }
                    _ => unreachable!(),
                };

                // Convert linear RGB to XYB directly (XYB is defined in linear space)
                // Scale to match C++ jpegli's expected range (0-255 linear input)
                let (scaled_x, scaled_y, scaled_b) = crate::color::xyb::linear_rgb_to_xyb_255(
                    r_linear * 255.0,
                    g_linear * 255.0,
                    b_linear * 255.0,
                );

                // Store: X→y_strip, Y→cb_strip, B→cr_strip
                // Scale to JPEG sample range for level shift consistency
                self.y_strip[y_row_start + x] = scaled_x * 255.0;
                self.cb_strip[cbcr_row_start + x] = scaled_y * 255.0;
                self.cr_strip[cbcr_row_start + x] = scaled_b * 255.0;
            }

            // Edge-pad X (y_strip) row
            if width < padded_width {
                let edge_val = self.y_strip[y_row_start + width - 1];
                for x in width..padded_width {
                    self.y_strip[y_row_start + x] = edge_val;
                }
            }
        }

        // For XYB mode, we handle the components differently:
        // - X is in y_strip (full res, already padded)
        // - Y is in cb_strip (full res, needs to stay there for DCT)
        // - B needs 2x2 downsampling (cr_strip → cr_down)
        //
        // Note: The DCT step will need to handle XYB's component structure specially
        // since Y (cb_strip) is full resolution unlike standard chroma.

        // Downsample B channel (cr_strip → cr_down) using 2x2 box filter
        let b_width = self.layout.b_width;
        let b_height = self.layout.b_strip_height_for(strip_height);
        crate::encode_simd::downsample_2x2_simd_inplace(
            &self.cr_strip[..strip_height * width],
            width,
            strip_height,
            &mut self.cr_down[..b_width * b_height],
        );

        // Rearrange and pad cr_down (B channel) using XYB-specific function
        self.pad_b_down_strip(b_height, b_width);

        // For Y component (cb_strip): rearrange to padded layout directly
        // We'll use cb_strip as the source for DCT in XYB mode
        if padded_width > width {
            for row in (0..strip_height).rev() {
                let src_start = row * width;
                let dst_start = row * padded_width;
                for x in (0..width).rev() {
                    self.cb_strip[dst_start + x] = self.cb_strip[src_start + x];
                }
                let edge_val = self.cb_strip[dst_start + width - 1];
                for x in width..padded_width {
                    self.cb_strip[dst_start + x] = edge_val;
                }
            }
        }

        Ok(())
    }

    /// Converts RGB strip to YCbCr using gamma-aware chroma downsampling.
    ///
    /// This computes Y at full resolution and Cb/Cr directly at the downsampled
    /// resolution using gamma-aware averaging in linear RGB space.
    pub(super) fn convert_strip_gamma_aware(
        &mut self,
        rgb_strip: &[u8],
        strip_y: usize,
        strip_height: usize,
    ) -> Result<()> {
        let width = self.layout.width;
        let bpp = self.pixel_format.bytes_per_pixel();
        let use_iterative = self.chroma_downsampling == DownsamplingMethod::GammaAwareIterative;

        if self.layout.subsampling == Subsampling::S444 {
            // No downsampling needed for 4:4:4, use standard path
            return self.convert_strip_to_ycbcr(rgb_strip, strip_height);
        }

        let c_width = self.layout.c_width;
        let c_strip_height = self.layout.c_strip_height_for(strip_height);
        let num_pixels = strip_height * width;
        let c_size = c_width * c_strip_height;

        match self.layout.subsampling {
            Subsampling::S420 => {
                crate::encode::chroma::gamma_aware_strip_420(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    strip_y,
                    self.layout.height,
                    bpp,
                    use_iterative,
                );
            }
            Subsampling::S422 => {
                crate::encode::chroma::gamma_aware_strip_422(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                    use_iterative,
                );
            }
            Subsampling::S440 => {
                crate::encode::chroma::gamma_aware_strip_440(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                    use_iterative,
                );
            }
            Subsampling::S444 => unreachable!(), // early return above
        }

        // Rearrange Y strip from packed to padded layout
        self.rearrange_y_strip_only(strip_height);

        // Pad chroma strips (cb_down, cr_down are already at downsampled resolution)
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
    }

    /// Converts RGB strip using fast fused Box downsampling.
    ///
    /// This computes Y at full resolution and Cb/Cr directly at the downsampled
    /// resolution using simple box averaging (no gamma correction).
    /// Faster than separate convert + downsample steps.
    #[allow(dead_code)]
    pub(super) fn convert_strip_box_fused(
        &mut self,
        rgb_strip: &[u8],
        strip_height: usize,
    ) -> Result<()> {
        let width = self.layout.width;
        let bpp = self.pixel_format.bytes_per_pixel();

        if self.layout.subsampling == Subsampling::S444 {
            // No downsampling needed for 4:4:4, use standard path
            return self.convert_strip_to_ycbcr(rgb_strip, strip_height);
        }

        let num_pixels = strip_height * width;
        let c_width = self.layout.c_width;
        let c_strip_height = self.layout.c_strip_height_for(strip_height);
        let c_size = c_width * c_strip_height;

        match self.layout.subsampling {
            Subsampling::S420 => {
                crate::encode::chroma::box_fused_strip_420(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                );
            }
            Subsampling::S422 => {
                crate::encode::chroma::box_fused_strip_422(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                );
            }
            Subsampling::S440 => {
                crate::encode::chroma::box_fused_strip_440(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                );
            }
            Subsampling::S444 => unreachable!(), // early return above
        }

        // Rearrange Y strip from packed to padded layout
        self.rearrange_y_strip_only(strip_height);

        // Pad chroma strips (cb_down, cr_down are already at downsampled resolution)
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
    }

    /// Rearranges only the Y strip from packed to padded layout.
    /// Used by gamma-aware conversion where Cb/Cr go directly to cb_down/cr_down.
    pub(super) fn rearrange_y_strip_only(&mut self, strip_height: usize) {
        let width = self.layout.width;
        let padded_width = self.layout.padded_width;

        if padded_width == width {
            return;
        }

        for row in (0..strip_height).rev() {
            let src_start = row * width;
            let dst_start = row * padded_width;

            for x in (0..width).rev() {
                self.y_strip[dst_start + x] = self.y_strip[src_start + x];
            }

            let edge_val = self.y_strip[dst_start + width - 1];
            for x in width..padded_width {
                self.y_strip[dst_start + x] = edge_val;
            }
        }
    }

    /// Pads strips vertically by replicating the last valid row.
    ///
    /// This is needed for the bottom strip when it has fewer rows than strip_height.
    /// Called after color conversion and horizontal padding.
    pub(super) fn pad_strips_vertically(&mut self, actual_height: usize, target_height: usize) {
        if actual_height >= target_height {
            return;
        }

        let padded_width = self.layout.padded_width;
        let is_color = !self.pixel_format.is_grayscale();

        // Get last valid row index
        let last_row = actual_height - 1;
        let src_start = last_row * padded_width;

        // Replicate to all remaining rows
        for row in actual_height..target_height {
            let dst_start = row * padded_width;
            self.y_strip
                .copy_within(src_start..src_start + padded_width, dst_start);
        }

        if is_color {
            // For cb_strip/cr_strip (if they're in padded layout)
            // Note: these are still in packed layout at this point
            let width = self.layout.width;
            let last_src = last_row * width;
            for row in actual_height..target_height {
                let dst = row * width;
                self.cb_strip.copy_within(last_src..last_src + width, dst);
                self.cr_strip.copy_within(last_src..last_src + width, dst);
            }
        }
    }

    /// Pads chroma down strips (cb_down, cr_down) horizontally.
    pub(super) fn pad_chroma_down_strip(&mut self, c_strip_height: usize, c_width: usize) {
        let padded_c_width = self.layout.padded_c_width;

        if padded_c_width == c_width {
            return;
        }

        // Rearrange and pad cb_down
        for row in (0..c_strip_height).rev() {
            let src_start = row * c_width;
            let dst_start = row * padded_c_width;

            for x in (0..c_width).rev() {
                self.cb_down[dst_start + x] = self.cb_down[src_start + x];
                self.cr_down[dst_start + x] = self.cr_down[src_start + x];
            }

            let cb_edge = self.cb_down[dst_start + c_width - 1];
            let cr_edge = self.cr_down[dst_start + c_width - 1];
            for x in c_width..padded_c_width {
                self.cb_down[dst_start + x] = cb_edge;
                self.cr_down[dst_start + x] = cr_edge;
            }
        }
    }

    /// Rearranges B channel (cr_down) from packed to padded layout for XYB mode.
    ///
    /// Unlike `pad_chroma_down_strip`, this only handles cr_down and uses
    /// `padded_b_width` which is correct for the 2x2 downsampled B channel.
    pub(super) fn pad_b_down_strip(&mut self, b_strip_height: usize, b_width: usize) {
        let padded_b_width = self.layout.padded_b_width;

        if padded_b_width == b_width {
            return;
        }

        // Rearrange and pad cr_down (B channel) only
        for row in (0..b_strip_height).rev() {
            let src_start = row * b_width;
            let dst_start = row * padded_b_width;

            for x in (0..b_width).rev() {
                self.cr_down[dst_start + x] = self.cr_down[src_start + x];
            }

            let edge = self.cr_down[dst_start + b_width - 1];
            for x in b_width..padded_b_width {
                self.cr_down[dst_start + x] = edge;
            }
        }
    }

    /// Downsamples chroma strips according to subsampling mode.
    ///
    /// Uses SIMD downsampling for floating-point parity with full-plane encoder.
    /// Input cb_strip/cr_strip are in packed layout (width pixels per row).
    /// Output cb_down/cr_down are rearranged to padded layout.
    pub(super) fn downsample_chroma_strip(&mut self, strip_height: usize) -> Result<()> {
        let width = self.layout.width;
        let num_pixels = strip_height * width;
        let c_width = self.layout.c_width;
        let c_strip_height = self.layout.c_strip_height;
        let c_size = c_width * c_strip_height;

        match self.layout.subsampling {
            Subsampling::S420 => {
                crate::encode_simd::downsample_2x2_simd_inplace(
                    &self.cb_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cb_down[..c_size],
                );
                crate::encode_simd::downsample_2x2_simd_inplace(
                    &self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cr_down[..c_size],
                );
            }
            Subsampling::S422 => {
                crate::encode_simd::downsample_2x1_simd_inplace(
                    &self.cb_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cb_down[..c_size],
                );
                crate::encode_simd::downsample_2x1_simd_inplace(
                    &self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cr_down[..c_size],
                );
            }
            Subsampling::S440 => {
                crate::encode_simd::downsample_1x2_simd_inplace(
                    &self.cb_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cb_down[..c_size],
                );
                crate::encode_simd::downsample_1x2_simd_inplace(
                    &self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cr_down[..c_size],
                );
            }
            Subsampling::S444 => {
                // No downsampling - copy directly
                self.cb_down[..c_size].copy_from_slice(&self.cb_strip[..c_size]);
                self.cr_down[..c_size].copy_from_slice(&self.cr_strip[..c_size]);
            }
        }

        // Rearrange cb_down/cr_down to padded layout for DCT block extraction
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
    }
}
