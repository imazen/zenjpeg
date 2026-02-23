//! Decoded image types for JPEG decoding.
//!
//! This module contains the output types returned by the decoder.
//!
//! For memory-efficient decoding of large images, prefer streaming APIs like
//! `Decoder::scanline_reader()`.

use crate::types::PixelFormat;

use super::DecodeWarning;
use super::extras::DecodedExtras;
use wide::f32x8;

/// A decoded image with dimensions and pixel data.
///
/// For large images, consider using `Decoder::scanline_reader()` to decode
/// row-by-row into caller-provided buffers.
#[derive(Clone)]
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
    /// Preserved metadata and secondary images (if preservation was enabled)
    pub(crate) extras: Option<DecodedExtras>,
    /// Warnings collected during decode (empty in Strict mode, which errors instead).
    pub(crate) warnings: Vec<DecodeWarning>,
}

impl core::fmt::Debug for DecodedImage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodedImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("data_len", &self.data.len())
            .field("has_extras", &self.extras.is_some())
            .finish()
    }
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

    /// Access preserved extras (metadata and secondary images).
    ///
    /// Returns `None` if preservation wasn't configured or if there were
    /// no segments to preserve.
    #[must_use]
    pub fn extras(&self) -> Option<&DecodedExtras> {
        self.extras.as_ref()
    }

    /// Take ownership of preserved extras.
    #[must_use]
    pub fn take_extras(&mut self) -> Option<DecodedExtras> {
        self.extras.take()
    }

    /// Returns warnings collected during decode.
    ///
    /// In [`Strictness::Strict`] mode, this is always empty because warnings
    /// become errors. In [`Strictness::Balanced`] and [`Strictness::Lenient`]
    /// modes, issues like truncation or missing DHT are collected here.
    #[must_use]
    pub fn warnings(&self) -> &[DecodeWarning] {
        &self.warnings
    }

    /// Returns true if any warnings were collected during decode.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Decompose the image into its parts.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, u32, u32, PixelFormat, Option<DecodedExtras>) {
        (self.data, self.width, self.height, self.format, self.extras)
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
///
/// For large images, consider using streaming APIs to decode row-by-row.
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
    /// Warnings collected during decode (empty in Strict mode).
    pub(crate) warnings: Vec<DecodeWarning>,
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
        let len = self.data.len();
        let mut data = vec![0u8; len];

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
                data[k + j] = arr[j] as u8;
            }
        }
        // Remainder
        for i in (chunks * 8)..len {
            data[i] = (self.data[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }

        DecodedImage {
            width: self.width,
            height: self.height,
            format: self.format,
            data,
            extras: None,
            warnings: self.warnings.clone(),
        }
    }

    /// Converts to 16-bit integer format.
    ///
    /// Values are scaled from 0.0-1.0 to 0-65535 and clamped.
    #[must_use]
    pub fn to_u16(&self) -> Vec<u16> {
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

    /// Returns warnings collected during decode.
    ///
    /// In [`Strictness::Strict`] mode, this is always empty because warnings
    /// become errors. In [`Strictness::Balanced`] and [`Strictness::Lenient`]
    /// modes, issues like truncation or missing DHT are collected here.
    #[must_use]
    pub fn warnings(&self) -> &[DecodeWarning] {
        &self.warnings
    }

    /// Returns true if any warnings were collected during decode.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
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
///
/// For large images, consider using streaming APIs to decode row-by-row.
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

    /// Shifts all planes in-place from centered range \[-128, 127\] to
    /// standard JPEG range \[0, 255\] by adding 128 to every sample.
    ///
    /// After calling this, `y` is in \[0, 255\] and `cb`/`cr` are in \[0, 255\].
    /// This avoids the three separate allocations of the per-plane methods.
    pub fn shift_to_jpeg_range(&mut self) {
        for v in &mut self.y {
            *v += 128.0;
        }
        for v in &mut self.cb {
            *v += 128.0;
        }
        for v in &mut self.cr {
            *v += 128.0;
        }
    }

    /// Converts Y plane to standard JPEG range \[0, 255\].
    ///
    /// Returns a new vector with values shifted by +128.
    /// Prefer [`shift_to_jpeg_range()`](Self::shift_to_jpeg_range) to avoid
    /// allocating three new vectors.
    #[must_use]
    pub fn y_to_jpeg_range(&self) -> Vec<f32> {
        self.y.iter().map(|&v| v + 128.0).collect()
    }

    /// Converts Cb plane to standard JPEG range \[0, 255\].
    ///
    /// Returns a new vector with values shifted by +128.
    /// Prefer [`shift_to_jpeg_range()`](Self::shift_to_jpeg_range) to avoid
    /// allocating three new vectors.
    #[must_use]
    pub fn cb_to_jpeg_range(&self) -> Vec<f32> {
        self.cb.iter().map(|&v| v + 128.0).collect()
    }

    /// Converts Cr plane to standard JPEG range \[0, 255\].
    ///
    /// Returns a new vector with values shifted by +128.
    /// Prefer [`shift_to_jpeg_range()`](Self::shift_to_jpeg_range) to avoid
    /// allocating three new vectors.
    #[must_use]
    pub fn cr_to_jpeg_range(&self) -> Vec<f32> {
        self.cr.iter().map(|&v| v + 128.0).collect()
    }
}

/// DCT coefficients for a single component.
///
/// Coefficients are stored in zigzag order as they appear in the JPEG file.
/// Each block contains 64 i16 values.
#[derive(Debug, Clone)]
pub struct ComponentCoefficients {
    /// Component ID (typically 1=Y, 2=Cb, 3=Cr for YCbCr)
    pub id: u8,
    /// Coefficients in block-row-major order, zigzag within each block.
    /// Length = blocks_wide * blocks_high * 64
    pub coeffs: Vec<i16>,
    /// Number of horizontal blocks (component width / 8)
    pub blocks_wide: usize,
    /// Number of vertical blocks (component height / 8)
    pub blocks_high: usize,
    /// Horizontal sampling factor
    pub h_samp: u8,
    /// Vertical sampling factor
    pub v_samp: u8,
    /// Quantization table index (which quant table this component uses)
    pub quant_table_idx: u8,
}

impl ComponentCoefficients {
    /// Returns a block's coefficients by block index.
    ///
    /// Block index is `by * blocks_wide + bx` where (bx, by) is block position.
    #[must_use]
    pub fn block(&self, block_idx: usize) -> &[i16] {
        let start = block_idx * 64;
        &self.coeffs[start..start + 64]
    }

    /// Returns a block's coefficients by position.
    #[must_use]
    pub fn block_at(&self, bx: usize, by: usize) -> &[i16] {
        self.block(by * self.blocks_wide + bx)
    }

    /// Returns the total number of blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks_wide * self.blocks_high
    }
}

/// Decoded DCT coefficients for analysis and comparison.
///
/// This provides access to the raw quantized DCT coefficients before IDCT,
/// useful for debugging, quality analysis, and encoder comparison.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decode::Decoder;
///
/// let decoder = Decoder::new();
/// let coeffs = decoder.decode_coefficients(&jpeg_data)?;
///
/// // Access Y component DC coefficient for first block
/// let y_dc = coeffs.components[0].block(0)[0];
/// println!("Y DC: {}", y_dc);
/// ```
///
/// For analysis of large images, consider streaming APIs.
#[derive(Debug, Clone)]
pub struct DecodedCoefficients {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Per-component coefficient data
    pub components: Vec<ComponentCoefficients>,
    /// Quantization tables (one per table slot used)
    /// Index matches component's quant_table_idx
    pub quant_tables: Vec<Option<[u16; 64]>>,
}

impl DecodedCoefficients {
    /// Returns the number of components.
    #[must_use]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Compares coefficients with another decode result, returning statistics.
    ///
    /// Returns (total_blocks, differing_blocks, max_diff, total_diff_coeffs)
    #[must_use]
    pub fn compare(&self, other: &DecodedCoefficients) -> CoefficientComparison {
        let mut total_blocks = 0usize;
        let mut differing_blocks = 0usize;
        let mut max_diff = 0i16;
        let mut total_diff_coeffs = 0usize;
        let mut diff_by_position = [0u64; 64];

        for (comp_idx, (c1, c2)) in self.components.iter().zip(&other.components).enumerate() {
            let num_blocks = c1.num_blocks().min(c2.num_blocks());
            for block_idx in 0..num_blocks {
                total_blocks += 1;
                let b1 = c1.block(block_idx);
                let b2 = c2.block(block_idx);
                let mut has_diff = false;
                for coeff_idx in 0..64 {
                    let diff = (b1[coeff_idx] as i32 - b2[coeff_idx] as i32).abs() as i16;
                    if diff != 0 {
                        has_diff = true;
                        total_diff_coeffs += 1;
                        diff_by_position[coeff_idx] += 1;
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
                if has_diff {
                    differing_blocks += 1;
                }
            }
            // Warn if block counts differ
            if c1.num_blocks() != c2.num_blocks() {
                eprintln!(
                    "Warning: component {} block count mismatch: {} vs {}",
                    comp_idx,
                    c1.num_blocks(),
                    c2.num_blocks()
                );
            }
        }

        CoefficientComparison {
            total_blocks,
            differing_blocks,
            max_diff,
            total_diff_coeffs,
            diff_by_position,
        }
    }
}

/// Statistics from comparing two coefficient sets.
#[derive(Debug, Clone)]
pub struct CoefficientComparison {
    /// Total number of blocks compared
    pub total_blocks: usize,
    /// Number of blocks with at least one differing coefficient
    pub differing_blocks: usize,
    /// Maximum absolute difference found
    pub max_diff: i16,
    /// Total count of differing coefficients
    pub total_diff_coeffs: usize,
    /// Difference counts by zigzag position (0=DC, 1-63=AC)
    pub diff_by_position: [u64; 64],
}

impl CoefficientComparison {
    /// Returns the percentage of blocks with differences.
    #[must_use]
    pub fn diff_block_pct(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            100.0 * self.differing_blocks as f64 / self.total_blocks as f64
        }
    }

    /// Returns the percentage of DC coefficients that differ.
    #[must_use]
    pub fn dc_diff_pct(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            100.0 * self.diff_by_position[0] as f64 / self.total_blocks as f64
        }
    }
}
