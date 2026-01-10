//! Color conversion methods for the encoder.
//!
//! This module contains all color space conversion and chroma subsampling
//! methods used by the encoder, including:
//! - XYB color space conversion
//! - YCbCr conversion (f32 and legacy u8)
//! - Chroma downsampling (2x2, 2x1, 1x2)
//! - Gamma-aware chroma downsampling (internal implementation)

use super::Encoder;
use crate::alloc::{checked_size_2d, try_alloc_filled, try_with_capacity};
use crate::color;
use crate::error::{Error, Result};
use crate::types::{PixelFormat, Subsampling};

impl Encoder {
    /// Converts input data to scaled XYB planes.
    ///
    /// Performs the full conversion: sRGB u8 → linear RGB → XYB → scaled XYB
    /// Output values are in [0, 1] range, ready to be scaled to [0, 255] for JPEG.
    pub(super) fn convert_to_scaled_xyb(
        &self,
        data: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        match self.config.pixel_format {
            PixelFormat::Rgb => {
                // Use SIMD-optimized version (allocates internally)
                Ok(crate::xyb::srgb_to_scaled_xyb_planes_simd(data, num_pixels))
            }
            PixelFormat::Rgba => {
                // Use RGBA-native SIMD (avoids intermediate allocation)
                Ok(crate::xyb::srgb_to_scaled_xyb_planes_simd_rgba(
                    data, num_pixels,
                ))
            }
            PixelFormat::Gray => {
                // Grayscale: expand to RGB then use SIMD
                let rgb: Vec<u8> = data.iter().flat_map(|&v| [v, v, v]).collect();
                Ok(crate::xyb::srgb_to_scaled_xyb_planes_simd(&rgb, num_pixels))
            }
            PixelFormat::Bgr => {
                // Swap B and R, then use SIMD
                let rgb: Vec<u8> = data
                    .chunks(3)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                Ok(crate::xyb::srgb_to_scaled_xyb_planes_simd(&rgb, num_pixels))
            }
            PixelFormat::Bgra => {
                // Use BGRA-native SIMD (avoids intermediate allocation)
                Ok(crate::xyb::srgb_to_scaled_xyb_planes_simd_bgra(
                    data, num_pixels,
                ))
            }
            PixelFormat::Cmyk => Err(Error::UnsupportedFeature {
                feature: "CMYK with XYB mode",
            }),
        }
    }

    /// Downsamples a float plane by 2x2 (box filter averaging).
    pub(super) fn downsample_2x2_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        // Use SIMD-optimized version
        crate::encode_simd::downsample_2x2_simd(plane, width, height)
    }

    /// Downsamples a float plane by 2x1 (horizontal only, box filter averaging).
    pub(super) fn downsample_2x1_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        crate::encode_simd::downsample_2x1_simd(plane, width, height)
    }

    /// Downsamples a float plane by 1x2 (vertical only, box filter averaging).
    pub(super) fn downsample_1x2_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        crate::encode_simd::downsample_1x2_simd(plane, width, height)
    }

    /// Converts RGB to YCbCr using gamma-aware downsampling for 4:2:0 subsampling.
    ///
    /// If `use_sharp` is true, uses iterative gamma-aware algorithm (Sharp YUV-like).
    /// If `use_sharp` is false, uses simple gamma-aware averaging (better than box filter).
    ///
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    pub(super) fn convert_gamma_aware_420(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;
        let c_height = (height + 1) / 2;

        // Handle special pixel formats
        match self.config.pixel_format {
            PixelFormat::Gray => {
                // Grayscale: Y = pixel value, Cb/Cr = 128
                let num_pixels = checked_size_2d(width, height)?;
                let c_size = checked_size_2d(c_width, c_height)?;
                let cb_plane = try_alloc_filled(c_size, 128.0f32, "Cb plane")?;
                let cr_plane = try_alloc_filled(c_size, 128.0f32, "Cr plane")?;
                // Convert u8 to f32 using SIMD
                let y_plane = crate::encode_simd::u8_slice_to_f32_simd(&data[..num_pixels])?;
                return Ok((y_plane, cb_plane, cr_plane, c_width, c_height));
            }
            PixelFormat::Cmyk => {
                return Err(Error::InvalidColorFormat {
                    reason: "gamma-aware conversion does not support CMYK input",
                });
            }
            _ => {}
        }

        // Get RGB data (convert if needed)
        let rgb_data: Vec<u8>;
        let rgb_ref: &[u8] = match self.config.pixel_format {
            PixelFormat::Rgb => data,
            PixelFormat::Rgba => {
                rgb_data = try_with_capacity(width * height * 3, "RGBA to RGB")?;
                let mut rgb = rgb_data;
                for i in 0..(width * height) {
                    rgb.push(data[i * 4]);
                    rgb.push(data[i * 4 + 1]);
                    rgb.push(data[i * 4 + 2]);
                }
                return self.convert_gamma_aware_420_rgb(&rgb, width, height, use_sharp);
            }
            PixelFormat::Bgr => {
                rgb_data = try_with_capacity(width * height * 3, "BGR to RGB")?;
                let mut rgb = rgb_data;
                for i in 0..(width * height) {
                    rgb.push(data[i * 3 + 2]); // R
                    rgb.push(data[i * 3 + 1]); // G
                    rgb.push(data[i * 3]); // B
                }
                return self.convert_gamma_aware_420_rgb(&rgb, width, height, use_sharp);
            }
            PixelFormat::Bgra => {
                rgb_data = try_with_capacity(width * height * 3, "BGRA to RGB")?;
                let mut rgb = rgb_data;
                for i in 0..(width * height) {
                    rgb.push(data[i * 4 + 2]); // R
                    rgb.push(data[i * 4 + 1]); // G
                    rgb.push(data[i * 4]); // B
                }
                return self.convert_gamma_aware_420_rgb(&rgb, width, height, use_sharp);
            }
            _ => unreachable!(),
        };

        // Use internal gamma-aware implementation (data is already RGB)
        if use_sharp {
            crate::chroma::convert_gamma_aware_iterative_420(
                rgb_ref,
                width,
                height,
                PixelFormat::Rgb,
            )
        } else {
            crate::chroma::convert_gamma_aware_420(rgb_ref, width, height, PixelFormat::Rgb)
        }
    }

    /// Helper for gamma-aware 4:2:0 with pre-converted RGB data.
    fn convert_gamma_aware_420_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        if use_sharp {
            crate::chroma::convert_gamma_aware_iterative_420(rgb, width, height, PixelFormat::Rgb)
        } else {
            crate::chroma::convert_gamma_aware_420(rgb, width, height, PixelFormat::Rgb)
        }
    }

    /// Converts RGB to YCbCr using gamma-aware downsampling for 4:2:2 subsampling.
    ///
    /// If `use_sharp` is true, uses iterative gamma-aware algorithm (Sharp YUV-like).
    /// If `use_sharp` is false, uses simple gamma-aware averaging.
    pub(super) fn convert_gamma_aware_422(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;

        // Handle special pixel formats
        match self.config.pixel_format {
            PixelFormat::Gray => {
                // Grayscale: Y = pixel value, Cb/Cr = 128
                let num_pixels = checked_size_2d(width, height)?;
                let c_size = checked_size_2d(c_width, height)?;
                let cb_plane = try_alloc_filled(c_size, 128.0f32, "Cb plane")?;
                let cr_plane = try_alloc_filled(c_size, 128.0f32, "Cr plane")?;
                let y_plane = crate::encode_simd::u8_slice_to_f32_simd(&data[..num_pixels])?;
                return Ok((y_plane, cb_plane, cr_plane, c_width, height));
            }
            PixelFormat::Cmyk => {
                return Err(Error::InvalidColorFormat {
                    reason: "gamma-aware conversion does not support CMYK input",
                });
            }
            _ => {}
        }

        // Get RGB data (convert if needed)
        let rgb_data: Vec<u8>;
        let rgb_ref: &[u8] = match self.config.pixel_format {
            PixelFormat::Rgb => data,
            PixelFormat::Rgba => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 4], data[i * 4 + 1], data[i * 4 + 2]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Bgr => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 3 + 2], data[i * 3 + 1], data[i * 3]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Bgra => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 4 + 2], data[i * 4 + 1], data[i * 4]])
                    .collect();
                &rgb_data
            }
            _ => unreachable!(),
        };

        // Use internal gamma-aware implementation (data is already RGB)
        if use_sharp {
            crate::chroma::convert_gamma_aware_iterative_422(
                rgb_ref,
                width,
                height,
                PixelFormat::Rgb,
            )
        } else {
            crate::chroma::convert_gamma_aware_422(rgb_ref, width, height, PixelFormat::Rgb)
        }
    }

    /// Converts RGB to YCbCr using gamma-aware downsampling for 4:4:0 subsampling.
    ///
    /// If `use_sharp` is true, uses iterative gamma-aware algorithm (Sharp YUV-like).
    /// If `use_sharp` is false, uses simple gamma-aware averaging.
    pub(super) fn convert_gamma_aware_440(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_height = (height + 1) / 2;

        // Handle special pixel formats
        match self.config.pixel_format {
            PixelFormat::Gray => {
                // Grayscale: Y = pixel value, Cb/Cr = 128
                let num_pixels = checked_size_2d(width, height)?;
                let c_size = checked_size_2d(width, c_height)?;
                let cb_plane = try_alloc_filled(c_size, 128.0f32, "Cb plane")?;
                let cr_plane = try_alloc_filled(c_size, 128.0f32, "Cr plane")?;
                let y_plane = crate::encode_simd::u8_slice_to_f32_simd(&data[..num_pixels])?;
                return Ok((y_plane, cb_plane, cr_plane, width, c_height));
            }
            PixelFormat::Cmyk => {
                return Err(Error::InvalidColorFormat {
                    reason: "gamma-aware conversion does not support CMYK input",
                });
            }
            _ => {}
        }

        // Get RGB data (convert if needed)
        let rgb_data: Vec<u8>;
        let rgb_ref: &[u8] = match self.config.pixel_format {
            PixelFormat::Rgb => data,
            PixelFormat::Rgba => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 4], data[i * 4 + 1], data[i * 4 + 2]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Bgr => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 3 + 2], data[i * 3 + 1], data[i * 3]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Bgra => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 4 + 2], data[i * 4 + 1], data[i * 4]])
                    .collect();
                &rgb_data
            }
            _ => unreachable!(),
        };

        // Use internal gamma-aware implementation (data is already RGB)
        if use_sharp {
            crate::chroma::convert_gamma_aware_iterative_440(
                rgb_ref,
                width,
                height,
                PixelFormat::Rgb,
            )
        } else {
            crate::chroma::convert_gamma_aware_440(rgb_ref, width, height, PixelFormat::Rgb)
        }
    }

    /// Converts to YCbCr using f32 Intrinsic path and applies chroma subsampling.
    ///
    /// This is the default path that matches C++ jpegli behavior.
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    pub(super) fn convert_intrinsic_with_subsampling(
        &self,
        data: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert to YCbCr using f32 precision throughout (matches C++ jpegli)
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Handle chroma subsampling
        let (cb_final, cr_final, c_width, c_height) = match self.config.subsampling {
            Subsampling::S420 => {
                // 4:2:0: Downsample both Cb and Cr by 2x2
                let cb_down = self.downsample_2x2_f32(&cb_plane, width, height)?;
                let cr_down = self.downsample_2x2_f32(&cr_plane, width, height)?;
                let c_w = (width + 1) / 2;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, c_w, c_h)
            }
            Subsampling::S422 => {
                // 4:2:2: Downsample horizontally only
                let cb_down = self.downsample_2x1_f32(&cb_plane, width, height)?;
                let cr_down = self.downsample_2x1_f32(&cr_plane, width, height)?;
                let c_w = (width + 1) / 2;
                (cb_down, cr_down, c_w, height)
            }
            Subsampling::S440 => {
                // 4:4:0: Downsample vertically only
                let cb_down = self.downsample_1x2_f32(&cb_plane, width, height)?;
                let cr_down = self.downsample_1x2_f32(&cr_plane, width, height)?;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, width, c_h)
            }
            Subsampling::S444 => {
                // 4:4:4: No subsampling
                (cb_plane, cr_plane, width, height)
            }
        };

        Ok((y_plane, cb_final, cr_final, c_width, c_height))
    }

    /// Converts input data to YCbCr planes (u8 version - legacy).
    #[allow(dead_code)]
    pub(super) fn convert_to_ycbcr(&self, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                let y = data.to_vec();
                let cb = try_alloc_filled(num_pixels, 128u8, "YCbCr Cb plane")?;
                let cr = try_alloc_filled(num_pixels, 128u8, "YCbCr Cr plane")?;
                Ok((y, cb, cr))
            }
            PixelFormat::Rgb => color::rgb_to_ycbcr_planes(data, width, height),
            PixelFormat::Rgba => {
                // Strip alpha and convert
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Bgr => {
                let rgb: Vec<u8> = data
                    .chunks(3)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Bgra => {
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Cmyk => Err(Error::UnsupportedFeature {
                feature: "CMYK encoding",
            }),
        }
    }

    /// Converts input data to YCbCr planes using full f32 precision.
    /// This matches C++ jpegli which uses float throughout the pipeline.
    /// Output values are in [0, 255] range (not level-shifted).
    pub(super) fn convert_to_ycbcr_f32(
        &self,
        data: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                // Use SIMD-optimized version (allocates internally)
                crate::encode_simd::gray_to_ycbcr_planes_simd(data, num_pixels)
            }
            PixelFormat::Rgb => {
                // Use SIMD-optimized version (allocates internally)
                crate::encode_simd::rgb_to_ycbcr_planes_simd(data, num_pixels)
            }
            PixelFormat::Rgba => {
                // Use SIMD-optimized version (allocates internally)
                crate::encode_simd::rgba_to_ycbcr_planes_simd(data, num_pixels)
            }
            PixelFormat::Bgr => {
                // Use SIMD-optimized version (allocates internally)
                crate::encode_simd::bgr_to_ycbcr_planes_simd(data, num_pixels)
            }
            PixelFormat::Bgra => {
                // Use SIMD-optimized version (allocates internally)
                crate::encode_simd::bgra_to_ycbcr_planes_simd(data, num_pixels)
            }
            PixelFormat::Cmyk => Err(Error::UnsupportedFeature {
                feature: "CMYK encoding",
            }),
        }
    }

}
