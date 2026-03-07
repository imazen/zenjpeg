//! Color space conversions for JPEG encoding and decoding.
//!
//! This module provides conversions between:
//! - RGB and YCbCr (standard JPEG)
//! - RGB and XYB (jpegli perceptual color space)
//! - Grayscale handling
//! - CMYK support

pub mod icc;
pub mod xyb;
pub mod ycbcr;

// Fast SIMD RGB→YCbCr conversion using the `yuv` crate (10-150× faster)
#[cfg(feature = "yuv")]
pub mod fast_yuv;

#[cfg(test)]
mod xyb_tests;

// Re-export commonly used items from ycbcr
pub use ycbcr::{rgb_to_ycbcr_f32, ycbcr_to_rgb_f32};

// Decoder-only YCbCr->RGB conversions
#[cfg(feature = "decoder")]
#[allow(unused_imports)]
pub use ycbcr::{
    cmyk_adobe_to_rgb, cmyk_planes_to_rgb_u8, gray_f32_to_gray_f32, gray_f32_to_gray_u8,
    gray_f32_to_rgb_f32, gray_f32_to_rgb_u8, rgb_u8_swap_rb_inplace, rgb_u8_to_bgra_u8,
    rgb_u8_to_bgrx_u8, rgb_u8_to_rgba_u8, ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8,
    ycbcr_planes_i16_to_rgb_u8, ycbcr_to_rgb, ycck_planes_to_rgb_u8, ycck_to_rgb,
};

// Re-export commonly used items from xyb
