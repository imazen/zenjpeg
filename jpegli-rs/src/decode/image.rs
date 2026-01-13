//! Decoded image types for JPEG decoding.
//!
//! This module contains the output types returned by the decoder.

use crate::types::PixelFormat;

#[cfg(feature = "simd")]
use wide::f32x8;

/// A decoded image with dimensions and pixel data.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedImage {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Pixel format of the data
    pub format: PixelFormat,
    /// Raw pixel data in the specified format
    pub data: Vec<u8>,
}

impl DecodedImage {
    /// Returns the image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the pixel data.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.data
    }

    /// Returns the number of bytes per pixel for this image's format.
    #[must_use]
    pub fn bytes_per_pixel(&self) -> usize {
        self.format.bytes_per_pixel()
    }

    /// Returns the stride (bytes per row) of the image.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.width as usize * self.bytes_per_pixel()
    }
}

/// A decoded image with 32-bit floating point pixel data.
///
/// This preserves the full 12-bit internal precision of jpegli's decoder
/// without quantization to 8-bit. Values are in the range 0.0-1.0.
///
/// Use this format when you need:
/// - Maximum precision for further image processing
/// - HDR workflows
/// - Scientific/medical imaging applications
/// - Input to machine learning models
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedImageF32 {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Pixel format of the data
    pub format: PixelFormat,
    /// Float pixel data in range 0.0-1.0
    pub data: Vec<f32>,
}

impl DecodedImageF32 {
    /// Returns the image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the pixel data.
    #[must_use]
    pub fn pixels(&self) -> &[f32] {
        &self.data
    }

    /// Returns the number of channels for this image's format.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.format.num_channels()
    }

    /// Returns the stride (floats per row) of the image.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.width as usize * self.channels()
    }

    /// Converts to 8-bit integer format.
    ///
    /// Values are scaled from 0.0-1.0 to 0-255 and clamped.
    #[must_use]
    pub fn to_u8(&self) -> DecodedImage {
        #[cfg(feature = "simd")]
        let data = {
            let len = self.data.len();
            let mut result = vec![0u8; len];

            let scale = f32x8::splat(255.0);
            let zero = f32x8::splat(0.0);
            let max_val = f32x8::splat(255.0);

            let chunks = len / 8;
            for chunk in 0..chunks {
                let k = chunk * 8;
                let v = f32x8::from([
                    self.data[k],
                    self.data[k + 1],
                    self.data[k + 2],
                    self.data[k + 3],
                    self.data[k + 4],
                    self.data[k + 5],
                    self.data[k + 6],
                    self.data[k + 7],
                ]);
                let scaled = (v * scale).round().max(zero).min(max_val);
                let arr: [f32; 8] = scaled.into();
                for j in 0..8 {
                    result[k + j] = arr[j] as u8;
                }
            }
            // Remainder
            for i in (chunks * 8)..len {
                result[i] = (self.data[i] * 255.0).round().clamp(0.0, 255.0) as u8;
            }
            result
        };

        #[cfg(not(feature = "simd"))]
        let data: Vec<u8> = self
            .data
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        DecodedImage {
            width: self.width,
            height: self.height,
            format: self.format,
            data,
        }
    }

    /// Converts to 16-bit integer format.
    ///
    /// Values are scaled from 0.0-1.0 to 0-65535 and clamped.
    #[must_use]
    pub fn to_u16(&self) -> Vec<u16> {
        #[cfg(feature = "simd")]
        {
            let len = self.data.len();
            let mut result = vec![0u16; len];

            let scale = f32x8::splat(65535.0);
            let zero = f32x8::splat(0.0);
            let max_val = f32x8::splat(65535.0);

            let chunks = len / 8;
            for chunk in 0..chunks {
                let k = chunk * 8;
                let v = f32x8::from([
                    self.data[k],
                    self.data[k + 1],
                    self.data[k + 2],
                    self.data[k + 3],
                    self.data[k + 4],
                    self.data[k + 5],
                    self.data[k + 6],
                    self.data[k + 7],
                ]);
                let scaled = (v * scale).round().max(zero).min(max_val);
                let arr: [f32; 8] = scaled.into();
                for j in 0..8 {
                    result[k + j] = arr[j] as u16;
                }
            }
            // Remainder
            for i in (chunks * 8)..len {
                result[i] = (self.data[i] * 65535.0).round().clamp(0.0, 65535.0) as u16;
            }
            result
        }

        #[cfg(not(feature = "simd"))]
        {
            self.data
                .iter()
                .map(|&v| (v * 65535.0).round().clamp(0.0, 65535.0) as u16)
                .collect()
        }
    }
}

/// Decoded YCbCr planes as 32-bit floats.
///
/// This provides direct access to the YCbCr color space data without
/// conversion to RGB, bypassing the expensive color conversion step.
///
/// Values are in centered range [-128, 127] (raw DCT output after level shift).
/// To convert to standard JPEG range [0, 255], add 128 to each value.
///
/// # Use Cases
///
/// - Video pipelines that work in YCbCr space
/// - Re-encoding without color space round-trip
/// - Custom color space transformations
/// - Maximum performance when RGB is not needed
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedYCbCr {
    /// Luma plane (width × height), range [-128, 127]
    pub y: Vec<f32>,
    /// Chroma-blue plane (width × height, upsampled), range [-128, 127]
    pub cb: Vec<f32>,
    /// Chroma-red plane (width × height, upsampled), range [-128, 127]
    pub cr: Vec<f32>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Embedded ICC profile, if present
    pub icc_profile: Option<Vec<u8>>,
}

impl DecodedYCbCr {
    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the number of pixels in each plane.
    #[must_use]
    pub fn plane_size(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Converts Y plane to standard JPEG range [0, 255].
    ///
    /// Returns a new vector with values shifted by +128.
    #[must_use]
    pub fn y_to_jpeg_range(&self) -> Vec<f32> {
        self.y.iter().map(|&v| v + 128.0).collect()
    }

    /// Converts Cb plane to standard JPEG range [0, 255].
    ///
    /// Returns a new vector with values shifted by +128.
    #[must_use]
    pub fn cb_to_jpeg_range(&self) -> Vec<f32> {
        self.cb.iter().map(|&v| v + 128.0).collect()
    }

    /// Converts Cr plane to standard JPEG range [0, 255].
    ///
    /// Returns a new vector with values shifted by +128.
    #[must_use]
    pub fn cr_to_jpeg_range(&self) -> Vec<f32> {
        self.cr.iter().map(|&v| v + 128.0).collect()
    }
}
