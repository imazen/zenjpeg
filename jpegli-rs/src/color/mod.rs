//! Color space conversions for JPEG encoding and decoding.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (standard JPEG)
//! - RGB and XYB (jpegli perceptual color space)
//! - Grayscale handling
//! - CMYK support

pub mod xyb;
pub mod ycbcr;

// Re-export commonly used items from ycbcr
pub use ycbcr::{
    bgr_to_rgb, bgra_to_rgba, cmyk_to_rgb, convert_rgb_to_ycbcr_buffer,
    convert_ycbcr_to_rgb_buffer, extract_channel, gray_f32_to_gray_f32, gray_f32_to_gray_u8,
    gray_f32_to_rgb_f32, gray_f32_to_rgb_u8, rgb_to_cmyk, rgb_to_ycbcr, rgb_to_ycbcr_f32,
    rgb_to_ycbcr_planes, ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8,
    ycbcr_planes_i16_to_rgb_u8, ycbcr_planes_to_rgb, ycbcr_to_rgb, ycbcr_to_rgb_f32,
    ycbcr_to_rgb_i16_x16,
};

// Re-export commonly used items from xyb
pub use xyb::{
    linear_rgb_to_xyb, linear_rgb_to_xyb_255, linear_to_srgb, linear_to_srgb_fast, srgb_to_linear,
    srgb_to_scaled_xyb, srgb_u8_to_linear, xyb_planes_to_rgb_f32_simd, xyb_planes_to_rgb_u8_simd,
    xyb_to_linear_rgb,
};
