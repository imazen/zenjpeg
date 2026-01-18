//! Color conversion and strip processing methods for StripProcessor.
//!
//! This module contains methods for:
//! - RGB to YCbCr conversion
//! - RGB to XYB conversion
//! - YCbCr input handling (f32 direct input)
//! - Chroma downsampling
//! - Strip padding (horizontal and vertical)

#![allow(dead_code)]

use crate::error::Result;
use crate::types::{ChromaDownsampling, PixelFormat, Subsampling};

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
        let width = self.width;
        let padded_width = self.padded_width;

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
        let width = self.width;
        let padded_width = self.padded_width;

        // Calculate expected chroma dimensions based on subsampling
        let (chroma_width, chroma_height) = match self.subsampling {
            Subsampling::S444 => (width, strip_height),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
            Subsampling::S440 => (width, (strip_height + 1) / 2),
        };

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
        let (chroma_width, target_height) = match self.subsampling {
            Subsampling::S444 => (self.width, self.strip_height),
            Subsampling::S422 => ((self.width + 1) / 2, self.strip_height),
            Subsampling::S420 => ((self.width + 1) / 2, (self.strip_height + 1) / 2),
            Subsampling::S440 => (self.width, (self.strip_height + 1) / 2),
        };

        let actual_chroma_height = match self.subsampling {
            Subsampling::S444 | Subsampling::S422 => actual_height,
            Subsampling::S420 | Subsampling::S440 => (actual_height + 1) / 2,
        };

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
        let width = self.width;
        let padded_width = self.padded_width;
        let num_pixels = strip_height * width;
        let y_size = strip_height * padded_width;

        match self.pixel_format {
            PixelFormat::Rgb | PixelFormat::Rgba => {
                let bpp = self.pixel_format.bytes_per_pixel();
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
            PixelFormat::Bgr | PixelFormat::Bgra => {
                let bpp = self.pixel_format.bytes_per_pixel();
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
            PixelFormat::Bgrx => {
                // BGRX: fast path - 4 bytes per pixel, padding ignored
                crate::encode_simd::bgr_to_ycbcr_strided_inplace(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    padded_width,
                    4, // BGRX is 4 bytes per pixel
                );
            }
            // 16-bit and float formats: linear RGB input
            // Uses optimized LUT conversion: linear -> sRGB -> YCbCr
            // For HDR (values > 1.0), applies Reinhard tone mapping
            PixelFormat::Gray16 => {
                use super::super::linear_lut::linear_u16_to_srgb_255;
                // Gray16: 2 bytes per pixel, native endian, linear
                for row in 0..strip_height {
                    let src_start = row * width * 2;
                    let dst_start = row * padded_width;
                    for x in 0..width {
                        let idx = src_start + x * 2;
                        let value = u16::from_ne_bytes([rgb_strip[idx], rgb_strip[idx + 1]]);
                        // Direct LUT lookup: linear u16 -> sRGB [0-255]
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
                use super::super::linear_lut::linear_rgb16_to_ycbcr;
                // RGB16/RGBA16: 6/8 bytes per pixel, native endian, linear
                // Uses optimized LUT: direct u16 -> YCbCr conversion
                let bpp = self.pixel_format.bytes_per_pixel();

                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    for x in 0..width {
                        let base = (row * width + x) * bpp;

                        // Read 16-bit values directly
                        let r = u16::from_ne_bytes([rgb_strip[base], rgb_strip[base + 1]]);
                        let g = u16::from_ne_bytes([rgb_strip[base + 2], rgb_strip[base + 3]]);
                        let b = u16::from_ne_bytes([rgb_strip[base + 4], rgb_strip[base + 5]]);

                        // LUT-optimized: linear RGB16 -> YCbCr in one step
                        let (y, cb, cr) = linear_rgb16_to_ycbcr(r, g, b);
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
                        // Load 8 f32 values (32 bytes)
                        let v = f32x8::new([
                            f32::from_ne_bytes([
                                rgb_strip[idx],
                                rgb_strip[idx + 1],
                                rgb_strip[idx + 2],
                                rgb_strip[idx + 3],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 4],
                                rgb_strip[idx + 5],
                                rgb_strip[idx + 6],
                                rgb_strip[idx + 7],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 8],
                                rgb_strip[idx + 9],
                                rgb_strip[idx + 10],
                                rgb_strip[idx + 11],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 12],
                                rgb_strip[idx + 13],
                                rgb_strip[idx + 14],
                                rgb_strip[idx + 15],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 16],
                                rgb_strip[idx + 17],
                                rgb_strip[idx + 18],
                                rgb_strip[idx + 19],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 20],
                                rgb_strip[idx + 21],
                                rgb_strip[idx + 22],
                                rgb_strip[idx + 23],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 24],
                                rgb_strip[idx + 25],
                                rgb_strip[idx + 26],
                                rgb_strip[idx + 27],
                            ]),
                            f32::from_ne_bytes([
                                rgb_strip[idx + 28],
                                rgb_strip[idx + 29],
                                rgb_strip[idx + 30],
                                rgb_strip[idx + 31],
                            ]),
                        ]);
                        let srgb = linear_to_srgb_255_x8(v);
                        let arr = srgb.to_array();
                        self.y_strip[dst_start + x..dst_start + x + 8].copy_from_slice(&arr);
                    }

                    // Handle remainder with scalar
                    for x in simd_width..width {
                        let idx = src_start + x * 4;
                        let linear = f32::from_ne_bytes([
                            rgb_strip[idx],
                            rgb_strip[idx + 1],
                            rgb_strip[idx + 2],
                            rgb_strip[idx + 3],
                        ]);
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
                            r_arr[i] = f32::from_ne_bytes([
                                rgb_strip[base],
                                rgb_strip[base + 1],
                                rgb_strip[base + 2],
                                rgb_strip[base + 3],
                            ]);
                            g_arr[i] = f32::from_ne_bytes([
                                rgb_strip[base + 4],
                                rgb_strip[base + 5],
                                rgb_strip[base + 6],
                                rgb_strip[base + 7],
                            ]);
                            b_arr[i] = f32::from_ne_bytes([
                                rgb_strip[base + 8],
                                rgb_strip[base + 9],
                                rgb_strip[base + 10],
                                rgb_strip[base + 11],
                            ]);
                        }

                        let r = f32x8::new(r_arr);
                        let g = f32x8::new(g_arr);
                        let b = f32x8::new(b_arr);

                        let (y, cb, cr) = linear_rgbf32_to_ycbcr_x8(r, g, b);

                        let y_arr = y.to_array();
                        let cb_arr = cb.to_array();
                        let cr_arr = cr.to_array();

                        self.y_strip[y_row_start + x..y_row_start + x + 8].copy_from_slice(&y_arr);
                        self.cb_strip[cbcr_row_start + x..cbcr_row_start + x + 8]
                            .copy_from_slice(&cb_arr);
                        self.cr_strip[cbcr_row_start + x..cbcr_row_start + x + 8]
                            .copy_from_slice(&cr_arr);
                    }

                    // Handle remainder with scalar
                    for x in simd_width..width {
                        let base = row_base + x * bpp;

                        let r_linear = f32::from_ne_bytes([
                            rgb_strip[base],
                            rgb_strip[base + 1],
                            rgb_strip[base + 2],
                            rgb_strip[base + 3],
                        ]);
                        let g_linear = f32::from_ne_bytes([
                            rgb_strip[base + 4],
                            rgb_strip[base + 5],
                            rgb_strip[base + 6],
                            rgb_strip[base + 7],
                        ]);
                        let b_linear = f32::from_ne_bytes([
                            rgb_strip[base + 8],
                            rgb_strip[base + 9],
                            rgb_strip[base + 10],
                            rgb_strip[base + 11],
                        ]);

                        let (y, cb, cr) = linear_rgbf32_to_ycbcr_fast(r_linear, g_linear, b_linear);
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

        let width = self.width;
        let padded_width = self.padded_width;
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
        let b_width = (width + 1) / 2;
        let b_height = (strip_height + 1) / 2;
        crate::encode_simd::downsample_2x2_simd_inplace(
            &self.cr_strip[..strip_height * width],
            width,
            strip_height,
            &mut self.cr_down[..b_width * b_height],
        );

        // Rearrange and pad cr_down (B channel)
        self.pad_chroma_down_strip(b_height, b_width);

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
        let width = self.width;
        let bpp = self.pixel_format.bytes_per_pixel();
        let use_iterative = self.chroma_downsampling == ChromaDownsampling::GammaAwareIterative;

        // Determine chroma strip dimensions
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, (strip_height + 1) / 2),
            Subsampling::S444 => {
                // No downsampling needed for 4:4:4, use standard path
                return self.convert_strip_to_ycbcr(rgb_strip, strip_height);
            }
        };

        let num_pixels = strip_height * width;
        let c_size = c_width * c_strip_height;

        match self.subsampling {
            Subsampling::S420 => {
                crate::encode::chroma::gamma_aware_strip_420(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    strip_y,
                    self.height,
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
            Subsampling::S444 => unreachable!(), // Handled above
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
        let width = self.width;
        let bpp = self.pixel_format.bytes_per_pixel();
        let num_pixels = strip_height * width;

        // Determine chroma strip dimensions
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, (strip_height + 1) / 2),
            Subsampling::S444 => {
                // No downsampling needed for 4:4:4, use standard path
                return self.convert_strip_to_ycbcr(rgb_strip, strip_height);
            }
        };

        let c_size = c_width * c_strip_height;

        match self.subsampling {
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
            Subsampling::S444 => unreachable!(), // Handled above
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
        let width = self.width;
        let padded_width = self.padded_width;

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

        let padded_width = self.padded_width;
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
            let width = self.width;
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
        let padded_c_width = self.padded_c_width;

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

    /// Downsamples chroma strips according to subsampling mode.
    ///
    /// Uses SIMD downsampling for floating-point parity with full-plane encoder.
    /// Input cb_strip/cr_strip are in packed layout (width pixels per row).
    /// Output cb_down/cr_down are rearranged to padded layout.
    pub(super) fn downsample_chroma_strip(&mut self, strip_height: usize) -> Result<()> {
        let width = self.width;
        let num_pixels = strip_height * width;

        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => {
                // 2x2 box filter using SIMD
                let c_width = (width + 1) / 2;
                let c_height = (strip_height + 1) / 2;
                let c_size = c_width * c_height;

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
                (c_width, c_height)
            }
            Subsampling::S422 => {
                // 2x1 horizontal filter using SIMD
                let c_width = (width + 1) / 2;
                let c_size = c_width * strip_height;

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
                (c_width, strip_height)
            }
            Subsampling::S440 => {
                // 1x2 vertical filter using SIMD
                let c_height = (strip_height + 1) / 2;
                let c_size = width * c_height;

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
                (width, c_height)
            }
            Subsampling::S444 => {
                // No downsampling - copy directly
                self.cb_down[..num_pixels].copy_from_slice(&self.cb_strip[..num_pixels]);
                self.cr_down[..num_pixels].copy_from_slice(&self.cr_strip[..num_pixels]);
                (width, strip_height)
            }
        };

        // Rearrange cb_down/cr_down to padded layout for DCT block extraction
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
    }
}
