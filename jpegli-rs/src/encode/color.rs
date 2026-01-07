//! Color conversion methods for the encoder.
//!
//! This module contains all color space conversion and chroma subsampling
//! methods used by the encoder, including:
//! - XYB color space conversion
//! - YCbCr conversion (f32 and legacy u8)
//! - Chroma downsampling (2x2, 2x1, 1x2)
//! - YUV crate integration for Sharp YUV

use super::Encoder;
use crate::alloc::{checked_size_2d, try_alloc_filled, try_with_capacity};
use crate::color;
use crate::error::{Error, Result};
use crate::types::{PixelFormat, Subsampling};

use yuv::{
    rgb_to_sharp_yuv420, rgb_to_sharp_yuv422, rgb_to_yuv420, rgb_to_yuv422, SharpYuvGammaTransfer,
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

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

    /// Applies input smoothing to a plane before downsampling.
    ///
    /// This is a 3x3 weighted blur matching libjpeg/jpegli's smoothing_factor:
    /// - Center pixel weight: 1.0 - 8 * (factor / 1024)
    /// - Neighbor pixel weight: factor / 1024
    ///
    /// Only applied when smoothing_factor > 0 and plane will be downsampled.
    pub(super) fn apply_input_smoothing(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        crate::encode_simd::apply_smoothing_simd(plane, width, height, self.config.smoothing_factor)
    }

    /// Converts RGB to YCbCr using yuv crate for 4:2:0 subsampling.
    ///
    /// If `use_sharp` is true, uses Sharp YUV (gamma-aware, better edges).
    /// If `use_sharp` is false, uses standard conversion (fast, simple box filter).
    ///
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    pub(super) fn convert_yuv_crate_420(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;
        let c_height = (height + 1) / 2;

        // Allocate YUV planar image
        let mut yuv_image =
            YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

        // Get RGB data in the right format
        let (rgb_data, rgb_stride) = match self.config.pixel_format {
            PixelFormat::Rgb => (data, width as u32 * 3),
            PixelFormat::Rgba => {
                // Convert RGBA to RGB
                let mut rgb = try_with_capacity(width * height * 3, "RGBA to RGB")?;
                for i in 0..(width * height) {
                    rgb.push(data[i * 4]);
                    rgb.push(data[i * 4 + 1]);
                    rgb.push(data[i * 4 + 2]);
                }
                return self
                    .convert_yuv_crate_420_rgb(&rgb, width, height, c_width, c_height, use_sharp);
            }
            PixelFormat::Bgr => {
                // Convert BGR to RGB
                let mut rgb = try_with_capacity(width * height * 3, "BGR to RGB")?;
                for i in 0..(width * height) {
                    rgb.push(data[i * 3 + 2]); // R
                    rgb.push(data[i * 3 + 1]); // G
                    rgb.push(data[i * 3]); // B
                }
                return self
                    .convert_yuv_crate_420_rgb(&rgb, width, height, c_width, c_height, use_sharp);
            }
            PixelFormat::Bgra => {
                // Convert BGRA to RGB
                let mut rgb = try_with_capacity(width * height * 3, "BGRA to RGB")?;
                for i in 0..(width * height) {
                    rgb.push(data[i * 4 + 2]); // R
                    rgb.push(data[i * 4 + 1]); // G
                    rgb.push(data[i * 4]); // B
                }
                return self
                    .convert_yuv_crate_420_rgb(&rgb, width, height, c_width, c_height, use_sharp);
            }
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
                    reason: "yuv crate does not support CMYK input",
                });
            }
        };

        // Perform YUV conversion (sharp or standard)
        if use_sharp {
            rgb_to_sharp_yuv420(
                &mut yuv_image,
                rgb_data,
                rgb_stride,
                YuvRange::Full,              // JPEG uses full range (0-255)
                YuvStandardMatrix::Bt601,    // Standard JPEG matrix
                SharpYuvGammaTransfer::Srgb, // sRGB input
            )
            .map_err(|e| Error::IoError {
                reason: format!("Sharp YUV conversion failed: {:?}", e),
            })?;
        } else {
            rgb_to_yuv420(
                &mut yuv_image,
                rgb_data,
                rgb_stride,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Balanced, // Fast path uses balanced mode
            )
            .map_err(|e| Error::IoError {
                reason: format!("YUV conversion failed: {:?}", e),
            })?;
        }

        // Convert u8 planes to f32
        let num_pixels = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, c_height)?;

        // Convert u8 planes to f32 using SIMD
        let y_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.y_plane.borrow()[..num_pixels])?;
        let cb_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.u_plane.borrow()[..c_size])?;
        let cr_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.v_plane.borrow()[..c_size])?;

        Ok((y_plane_f32, cb_plane_f32, cr_plane_f32, c_width, c_height))
    }

    /// Helper for yuv crate with pre-converted RGB data.
    fn convert_yuv_crate_420_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        c_width: usize,
        c_height: usize,
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let mut yuv_image =
            YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

        if use_sharp {
            rgb_to_sharp_yuv420(
                &mut yuv_image,
                rgb,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                SharpYuvGammaTransfer::Srgb,
            )
            .map_err(|e| Error::IoError {
                reason: format!("Sharp YUV conversion failed: {:?}", e),
            })?;
        } else {
            rgb_to_yuv420(
                &mut yuv_image,
                rgb,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Balanced,
            )
            .map_err(|e| Error::IoError {
                reason: format!("YUV conversion failed: {:?}", e),
            })?;
        }

        let num_pixels = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, c_height)?;

        // Convert u8 planes to f32 using SIMD
        let y_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.y_plane.borrow()[..num_pixels])?;
        let cb_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.u_plane.borrow()[..c_size])?;
        let cr_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.v_plane.borrow()[..c_size])?;

        Ok((y_plane_f32, cb_plane_f32, cr_plane_f32, c_width, c_height))
    }

    /// Converts RGB to YCbCr using yuv crate for 4:2:2 subsampling.
    ///
    /// If `use_sharp` is true, uses Sharp YUV (gamma-aware, better edges).
    /// If `use_sharp` is false, uses standard conversion (fast, simple box filter).
    pub(super) fn convert_yuv_crate_422(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;

        // For formats other than RGB, convert first
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
            PixelFormat::Gray => {
                // Grayscale doesn't benefit from yuv crate conversion
                let num_pixels = checked_size_2d(width, height)?;
                let c_size = checked_size_2d(c_width, height)?;
                let cb_plane = try_alloc_filled(c_size, 128.0f32, "Cb plane")?;
                let cr_plane = try_alloc_filled(c_size, 128.0f32, "Cr plane")?;
                // Convert u8 to f32 using SIMD
                let y_plane = crate::encode_simd::u8_slice_to_f32_simd(&data[..num_pixels])?;
                return Ok((y_plane, cb_plane, cr_plane, c_width, height));
            }
            PixelFormat::Cmyk => {
                return Err(Error::InvalidColorFormat {
                    reason: "yuv crate does not support CMYK input",
                });
            }
        };

        let mut yuv_image =
            YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv422);

        if use_sharp {
            rgb_to_sharp_yuv422(
                &mut yuv_image,
                rgb_ref,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                SharpYuvGammaTransfer::Srgb,
            )
            .map_err(|e| Error::IoError {
                reason: format!("Sharp YUV 422 conversion failed: {:?}", e),
            })?;
        } else {
            rgb_to_yuv422(
                &mut yuv_image,
                rgb_ref,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Balanced,
            )
            .map_err(|e| Error::IoError {
                reason: format!("YUV 422 conversion failed: {:?}", e),
            })?;
        }

        let num_pixels = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, height)?;

        // Convert u8 planes to f32 using SIMD
        let y_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.y_plane.borrow()[..num_pixels])?;
        let cb_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.u_plane.borrow()[..c_size])?;
        let cr_plane_f32 =
            crate::encode_simd::u8_slice_to_f32_simd(&yuv_image.v_plane.borrow()[..c_size])?;

        Ok((y_plane_f32, cb_plane_f32, cr_plane_f32, c_width, height))
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

        // Handle chroma subsampling (with optional input smoothing)
        // Only apply smoothing if smoothing_factor > 0 to avoid unnecessary copies
        let use_smoothing = self.config.smoothing_factor > 0;
        let (cb_final, cr_final, c_width, c_height) = match self.config.subsampling {
            Subsampling::S420 => {
                // 4:2:0: Apply smoothing (if enabled) then downsample both Cb and Cr by 2x2
                let (cb_down, cr_down) = if use_smoothing {
                    let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                    let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                    (
                        self.downsample_2x2_f32(&cb_smooth, width, height)?,
                        self.downsample_2x2_f32(&cr_smooth, width, height)?,
                    )
                } else {
                    (
                        self.downsample_2x2_f32(&cb_plane, width, height)?,
                        self.downsample_2x2_f32(&cr_plane, width, height)?,
                    )
                };
                let c_w = (width + 1) / 2;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, c_w, c_h)
            }
            Subsampling::S422 => {
                // 4:2:2: Apply smoothing (if enabled) then downsample horizontally only
                let (cb_down, cr_down) = if use_smoothing {
                    let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                    let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                    (
                        self.downsample_2x1_f32(&cb_smooth, width, height)?,
                        self.downsample_2x1_f32(&cr_smooth, width, height)?,
                    )
                } else {
                    (
                        self.downsample_2x1_f32(&cb_plane, width, height)?,
                        self.downsample_2x1_f32(&cr_plane, width, height)?,
                    )
                };
                let c_w = (width + 1) / 2;
                (cb_down, cr_down, c_w, height)
            }
            Subsampling::S440 => {
                // 4:4:0: Apply smoothing (if enabled) then downsample vertically only
                let (cb_down, cr_down) = if use_smoothing {
                    let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                    let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                    (
                        self.downsample_1x2_f32(&cb_smooth, width, height)?,
                        self.downsample_1x2_f32(&cr_smooth, width, height)?,
                    )
                } else {
                    (
                        self.downsample_1x2_f32(&cb_plane, width, height)?,
                        self.downsample_1x2_f32(&cr_plane, width, height)?,
                    )
                };
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, width, c_h)
            }
            Subsampling::S444 => {
                // 4:4:4: No subsampling, no smoothing needed
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

    /// Converts to YCbCr using workspace buffers and applies chroma subsampling.
    ///
    /// This is the zero-allocation hot path for batch encoding. Uses workspace
    /// buffers for all intermediate data.
    ///
    /// Returns: (y_vec, cb_vec, cr_vec, chroma_width, chroma_height)
    /// Copies data out of workspace to avoid lifetime issues with downstream code.
    pub(super) fn convert_intrinsic_with_subsampling_workspace(
        &self,
        data: &[u8],
        workspace: &mut super::EncoderWorkspace,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = crate::alloc::checked_size_2d(width, height)?;

        // Convert to YCbCr in-place to workspace buffers
        {
            let (y_plane, cb_plane, cr_plane) = workspace.planes_mut(num_pixels);
            self.convert_to_ycbcr_f32_inplace(data, y_plane, cb_plane, cr_plane)?;
        }

        // Handle chroma subsampling
        // For smoothing, fall back to allocating path (rare case)
        if self.config.smoothing_factor > 0 {
            // Slow path: smoothing - use the original allocating implementation
            return self.convert_intrinsic_with_subsampling(data);
        }

        // Fast path: no smoothing, downsample directly using workspace
        match self.config.subsampling {
            Subsampling::S420 => {
                let c_w = (width + 1) / 2;
                let c_h = (height + 1) / 2;
                let c_size = c_w * c_h;
                // Downsample cb/cr planes to temp buffers
                crate::encode_simd::downsample_2x2_simd_inplace(
                    &workspace.cb_plane[..num_pixels],
                    width,
                    height,
                    &mut workspace.temp_cb[..c_size],
                );
                crate::encode_simd::downsample_2x2_simd_inplace(
                    &workspace.cr_plane[..num_pixels],
                    width,
                    height,
                    &mut workspace.temp_cr[..c_size],
                );
                // Return owned copies from workspace
                Ok((
                    workspace.y_plane[..num_pixels].to_vec(),
                    workspace.temp_cb[..c_size].to_vec(),
                    workspace.temp_cr[..c_size].to_vec(),
                    c_w,
                    c_h,
                ))
            }
            Subsampling::S422 => {
                let c_w = (width + 1) / 2;
                let c_size = c_w * height;
                crate::encode_simd::downsample_2x1_simd_inplace(
                    &workspace.cb_plane[..num_pixels],
                    width,
                    height,
                    &mut workspace.temp_cb[..c_size],
                );
                crate::encode_simd::downsample_2x1_simd_inplace(
                    &workspace.cr_plane[..num_pixels],
                    width,
                    height,
                    &mut workspace.temp_cr[..c_size],
                );
                Ok((
                    workspace.y_plane[..num_pixels].to_vec(),
                    workspace.temp_cb[..c_size].to_vec(),
                    workspace.temp_cr[..c_size].to_vec(),
                    c_w,
                    height,
                ))
            }
            Subsampling::S440 => {
                let c_h = (height + 1) / 2;
                let c_size = width * c_h;
                crate::encode_simd::downsample_1x2_simd_inplace(
                    &workspace.cb_plane[..num_pixels],
                    width,
                    height,
                    &mut workspace.temp_cb[..c_size],
                );
                crate::encode_simd::downsample_1x2_simd_inplace(
                    &workspace.cr_plane[..num_pixels],
                    width,
                    height,
                    &mut workspace.temp_cr[..c_size],
                );
                Ok((
                    workspace.y_plane[..num_pixels].to_vec(),
                    workspace.temp_cb[..c_size].to_vec(),
                    workspace.temp_cr[..c_size].to_vec(),
                    width,
                    c_h,
                ))
            }
            Subsampling::S444 => {
                // No downsampling, return planes directly
                Ok((
                    workspace.y_plane[..num_pixels].to_vec(),
                    workspace.cb_plane[..num_pixels].to_vec(),
                    workspace.cr_plane[..num_pixels].to_vec(),
                    width,
                    height,
                ))
            }
        }
    }

    /// Converts input data to YCbCr planes using workspace buffers (zero-allocation).
    ///
    /// This is the hot-path version for batch encoding. Writes directly to
    /// pre-allocated workspace buffers to avoid allocation overhead.
    pub(super) fn convert_to_ycbcr_f32_inplace(
        &self,
        data: &[u8],
        y_plane: &mut [f32],
        cb_plane: &mut [f32],
        cr_plane: &mut [f32],
    ) -> Result<()> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                crate::encode_simd::gray_to_ycbcr_planes_simd_inplace(
                    data, y_plane, cb_plane, cr_plane, num_pixels,
                );
                Ok(())
            }
            PixelFormat::Rgb => {
                crate::encode_simd::rgb_to_ycbcr_planes_simd_inplace(
                    data, y_plane, cb_plane, cr_plane, num_pixels,
                );
                Ok(())
            }
            PixelFormat::Rgba => {
                crate::encode_simd::rgba_to_ycbcr_planes_simd_inplace(
                    data, y_plane, cb_plane, cr_plane, num_pixels,
                );
                Ok(())
            }
            PixelFormat::Bgr => {
                crate::encode_simd::bgr_to_ycbcr_planes_simd_inplace(
                    data, y_plane, cb_plane, cr_plane, num_pixels,
                );
                Ok(())
            }
            PixelFormat::Bgra => {
                crate::encode_simd::bgra_to_ycbcr_planes_simd_inplace(
                    data, y_plane, cb_plane, cr_plane, num_pixels,
                );
                Ok(())
            }
            PixelFormat::Cmyk => Err(Error::UnsupportedFeature {
                feature: "CMYK encoding",
            }),
        }
    }
}
