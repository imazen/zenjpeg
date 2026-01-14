//! JPEG decoder implementation.
//!
//! This module provides the main decoder interface for reading JPEG images.
//!
//! # ICC Profile Support
//!
//! The decoder can extract and apply embedded ICC profiles, including XYB profiles
//! used by jpegli. ICC profile support requires enabling `cms-lcms2` or `cms-moxcms` feature.
//!
//! ```ignore
//! use jpegli::decode::Decoder;
//!
//! let decoder = Decoder::new().apply_icc(true);
//! let decoded = decoder.decode(&jpeg_data)?;
//! ```

// IDCT modules (decoder-only)
#[doc(hidden)]
pub mod idct;
#[doc(hidden)]
pub mod idct_int;

mod image;
mod parser;
mod scanline;
mod upsample;

pub use image::{DecodedImage, DecodedImageF32, DecodedYCbCr};
use parser::JpegParser;

pub use scanline::{ScanlineInfo, ScanlineReader};

// Re-export types used in public struct fields so users can access them
pub use crate::types::{ColorSpace, Dimensions, JpegMode, PixelFormat};

use crate::error::{Error, Result};

#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
use crate::color::icc::apply_icc_transform;
use crate::foundation::alloc::{DEFAULT_MAX_MEMORY, DEFAULT_MAX_PIXELS};

/// Decoder configuration.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Output pixel format (None = use source format)
    pub output_format: Option<PixelFormat>,
    /// Whether to apply fancy upsampling
    pub fancy_upsampling: bool,
    /// Whether to apply block smoothing
    pub block_smoothing: bool,
    /// Whether to apply embedded ICC profile (requires cms feature)
    pub apply_icc: bool,
    /// Maximum pixels allowed (for DoS protection).
    /// Default is 100 megapixels. Set to 0 for unlimited.
    pub max_pixels: u64,
    /// Maximum total memory for allocations (for DoS protection).
    /// Default is 512 MB. Set to 0 for unlimited.
    pub max_memory: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            output_format: None,
            fancy_upsampling: true,
            block_smoothing: false,
            // Apply ICC by default when CMS is available
            apply_icc: cfg!(any(feature = "cms-lcms2", feature = "cms-moxcms")),
            max_pixels: DEFAULT_MAX_PIXELS,
            max_memory: DEFAULT_MAX_MEMORY,
        }
    }
}

/// Information about a decoded JPEG.
#[derive(Debug, Clone)]
pub struct JpegInfo {
    /// Image dimensions
    pub dimensions: Dimensions,
    /// Color space
    pub color_space: ColorSpace,
    /// Sample precision (8 or 12 bits)
    pub precision: u8,
    /// Number of components
    pub num_components: u8,
    /// Encoding mode
    pub mode: JpegMode,
    /// Whether an ICC profile is embedded
    pub has_icc_profile: bool,
    /// Whether the ICC profile is an XYB profile
    pub is_xyb: bool,
}

/// JPEG decoder.
pub struct Decoder {
    config: DecoderConfig,
}

impl Decoder {
    /// Creates a new decoder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: DecoderConfig::default(),
        }
    }

    /// Creates a decoder from configuration.
    #[must_use]
    pub fn from_config(config: DecoderConfig) -> Self {
        Self { config }
    }

    /// Sets the output pixel format.
    #[must_use]
    pub fn output_format(mut self, format: PixelFormat) -> Self {
        self.config.output_format = Some(format);
        self
    }

    /// Enables fancy upsampling.
    #[must_use]
    pub fn fancy_upsampling(mut self, enable: bool) -> Self {
        self.config.fancy_upsampling = enable;
        self
    }

    /// Enables block smoothing.
    #[must_use]
    pub fn block_smoothing(mut self, enable: bool) -> Self {
        self.config.block_smoothing = enable;
        self
    }

    /// Enables ICC profile application.
    ///
    /// When enabled, embedded ICC profiles will be applied to convert
    /// the image to sRGB. This is required for correct display of
    /// XYB-encoded images.
    ///
    /// Note: Requires `cms-lcms2` or `cms-moxcms` feature to be enabled.
    /// Without a CMS feature, this setting has no effect.
    #[must_use]
    pub fn apply_icc(mut self, enable: bool) -> Self {
        self.config.apply_icc = enable;
        self
    }

    /// Sets the maximum number of pixels allowed (for DoS protection).
    ///
    /// Default is 100 megapixels. Set to 0 for unlimited.
    #[must_use]
    pub fn max_pixels(mut self, pixels: u64) -> Self {
        self.config.max_pixels = pixels;
        self
    }

    /// Sets the maximum memory allowed for allocations during decoding.
    ///
    /// Default is 512 MB. Set to `usize::MAX` for unlimited.
    /// This prevents memory exhaustion attacks from malicious images.
    #[must_use]
    pub fn max_memory(mut self, bytes: usize) -> Self {
        self.config.max_memory = bytes;
        self
    }

    /// Reads JPEG info without decoding.
    pub fn read_info(&self, data: &[u8]) -> Result<JpegInfo> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
        parser.read_header()?;
        Ok(parser.info())
    }

    /// Estimates peak memory usage for decoding an image of given dimensions.
    ///
    /// This is useful for checking if an image can be decoded within memory limits
    /// before attempting to decode it. The estimate includes:
    /// - Strip buffers for one MCU row (Y, Cb, Cr at 2 bytes per pixel)
    /// - Output RGB buffer (3 bytes per pixel)
    ///
    /// For streaming decode (baseline 4:4:4), no coefficient storage is needed.
    /// For progressive or subsampled images, add ~128 bytes per DCT block for coefficients.
    ///
    /// # Example
    /// ```
    /// use jpegli::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let estimated = decoder.estimate_memory_usage(4096, 4096);
    /// println!("Estimated peak memory: {} MB", estimated / 1024 / 1024);
    /// ```
    #[must_use]
    pub fn estimate_memory_usage(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;

        // MCU width for strip buffers (padded to 8)
        let mcu_cols = (w + 7) / 8;
        let strip_width = mcu_cols * 8;
        let strip_height = 8; // One MCU row

        // Strip buffers: Y, Cb, Cr each at i16 (2 bytes per pixel)
        let strip_size = strip_width * strip_height;
        let strip_total = strip_size * 2 * 3; // 3 components, 2 bytes each

        // Output RGB buffer: 3 bytes per pixel
        let rgb_size = w * h * 3;

        // Total for streaming decode (baseline 4:4:4)
        let streaming_total = strip_total + rgb_size;

        // For non-streaming paths (progressive, subsampled), coefficient storage is needed
        // ~130 bytes per block (64 i16 coefficients + u8 coeff count + alignment)
        // 3 components, (w/8) * (h/8) blocks per component
        let blocks_per_component = mcu_cols * ((h + 7) / 8);
        let coeff_storage = blocks_per_component * 130 * 3;

        // Return worst case (non-streaming)
        streaming_total.max(coeff_storage + rgb_size)
    }

    /// Creates a pull-based scanline reader for streaming decode.
    ///
    /// This allows reading the image row by row without loading the entire
    /// image into memory. Only supports baseline JPEGs with 4:4:4 subsampling.
    ///
    /// # Example
    /// ```ignore
    /// use jpegli::{Decoder, ImgRefMut};
    ///
    /// let mut reader = Decoder::new().scanline_reader(&jpeg_data)?;
    /// let width = reader.width() as usize;
    /// let height = reader.height() as usize;
    ///
    /// let mut pixels = vec![0u8; width * height * 3];
    /// let mut rows_read = 0;
    /// while rows_read < height {
    ///     let remaining = height - rows_read;
    ///     let slice = &mut pixels[rows_read * width * 3..];
    ///     let output = ImgRefMut::new(slice, width * 3, remaining);
    ///     rows_read += reader.read_rows_rgb8(output)?;
    /// }
    /// ```
    pub fn scanline_reader<'a>(&self, data: &'a [u8]) -> Result<ScanlineReader<'a>> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
        parser.read_header()?;

        // Only baseline supported for scanline reading
        if parser.mode != JpegMode::Baseline {
            return Err(Error::UnsupportedFeature {
                feature: "scanline reader only supports baseline JPEG",
            });
        }

        if parser.num_components != 3 {
            return Err(Error::UnsupportedFeature {
                feature: "scanline reader requires 3-component YCbCr image",
            });
        }

        // Extract sampling factors
        let h_samp = [
            parser.components[0].h_samp_factor,
            parser.components[1].h_samp_factor,
            parser.components[2].h_samp_factor,
        ];
        let v_samp = [
            parser.components[0].v_samp_factor,
            parser.components[1].v_samp_factor,
            parser.components[2].v_samp_factor,
        ];

        // Validate sampling factors - support 4:4:4, 4:2:2, and 4:2:0
        let max_h = h_samp.iter().copied().max().unwrap_or(1);
        let max_v = v_samp.iter().copied().max().unwrap_or(1);
        if max_h > 2 || max_v > 2 {
            return Err(Error::UnsupportedFeature {
                feature: "scanline reader only supports sampling factors up to 2x2",
            });
        }

        // Extract quant table indices
        let quant_indices = [
            parser.components[0].quant_table_idx as usize,
            parser.components[1].quant_table_idx as usize,
            parser.components[2].quant_table_idx as usize,
        ];

        // Find SOS marker to get table mapping and scan data position
        let scan_info = parser.find_scan_info()?;

        // Get info before moving tables
        let is_xyb = parser.info().is_xyb;

        ScanlineReader::new(
            data,
            parser.width,
            parser.height,
            parser.num_components,
            h_samp,
            v_samp,
            parser.quant_tables,
            quant_indices,
            parser.dc_tables,
            parser.ac_tables,
            scan_info.table_mapping,
            scan_info.data_start,
            parser.restart_interval,
            is_xyb,
        )
    }

    /// Decodes a JPEG image.
    pub fn decode(&self, data: &[u8]) -> Result<DecodedImage> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
        parser.decode()?;

        let info = parser.info();
        let output_format = self.config.output_format.unwrap_or(PixelFormat::Rgb);

        // Convert to output format
        // For XYB images, use simple dequantization so ICC profile works correctly
        let mut pixels =
            parser.to_pixels(output_format, info.is_xyb, self.config.fancy_upsampling)?;

        // Apply ICC profile if enabled and present
        // Note: ICC transform failures are non-fatal - we fall back to un-color-managed pixels
        // rather than failing the decode, since the JPEG itself decoded successfully
        #[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
        if self.config.apply_icc && output_format == PixelFormat::Rgb {
            if let Some(ref icc_profile) = parser.icc_profile {
                match apply_icc_transform(
                    &pixels,
                    info.dimensions.width as usize,
                    info.dimensions.height as usize,
                    icc_profile,
                ) {
                    Ok(transformed) => pixels = transformed,
                    Err(_e) => {
                        // ICC transform failed - continue with un-color-managed pixels
                        // This can happen with unusual profiles that CMS libraries don't support
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "Warning: ICC profile transform failed, using original colors: {_e:?}"
                        );
                    }
                }
            }
        }

        Ok(DecodedImage {
            width: info.dimensions.width,
            height: info.dimensions.height,
            format: output_format,
            data: pixels,
        })
    }

    /// Decodes a JPEG image to 32-bit floating point pixels.
    ///
    /// This preserves the full 12-bit internal precision of jpegli's decoder
    /// without quantization to 8-bit. Values are normalized to range 0.0-1.0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::decode::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let image = decoder.decode_f32(&jpeg_data)?;
    /// // image.data contains f32 values in range 0.0-1.0
    /// ```
    ///
    /// Note: ICC profile application is not supported for f32 output.
    /// If you need ICC profile transformation, decode to u8 first.
    pub fn decode_f32(&self, data: &[u8]) -> Result<DecodedImageF32> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
        // Disable streaming - f32 decode needs coefficients for precision
        parser.prefer_streaming = false;
        parser.decode()?;

        let info = parser.info();
        let output_format = self.config.output_format.unwrap_or(PixelFormat::Rgb);

        // Convert to output format as f32
        let pixels =
            parser.to_pixels_f32(output_format, info.is_xyb, self.config.fancy_upsampling)?;

        Ok(DecodedImageF32 {
            width: info.dimensions.width,
            height: info.dimensions.height,
            format: output_format,
            data: pixels,
        })
    }

    /// Decodes a JPEG image to planar YCbCr f32 data.
    ///
    /// This bypasses the YCbCr→RGB color conversion, providing direct access
    /// to the decoded YCbCr planes. This is significantly faster than decoding
    /// to RGB when you need YCbCr data (e.g., for re-encoding or video pipelines).
    ///
    /// # Value Range
    ///
    /// Values are in centered range [-128, 127] (raw DCT output).
    /// To convert to standard JPEG range [0, 255], add 128 to each value.
    ///
    /// # Chroma Planes
    ///
    /// Chroma planes (Cb, Cr) are always upsampled to full resolution,
    /// matching the Y plane dimensions. The upsampling method is controlled
    /// by the `fancy_upsampling` setting.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use jpegli::decode::Decoder;
    ///
    /// let decoder = Decoder::new();
    /// let ycbcr = decoder.decode_to_ycbcr_f32(&jpeg_data)?;
    ///
    /// // Access planes directly
    /// let y_plane = &ycbcr.y;   // [-128, 127] range
    /// let cb_plane = &ycbcr.cb;
    /// let cr_plane = &ycbcr.cr;
    ///
    /// // Or convert to JPEG range [0, 255]
    /// let y_jpeg = ycbcr.y_to_jpeg_range();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The image is grayscale (only 1 component)
    /// - The image uses XYB color space (not YCbCr)
    /// - Parsing or decoding fails
    pub fn decode_to_ycbcr_f32(&self, data: &[u8]) -> Result<DecodedYCbCr> {
        let mut parser = JpegParser::new(data, self.config.max_pixels)?;
        // Disable streaming - f32 YCbCr decode needs coefficients
        parser.prefer_streaming = false;
        parser.decode()?;

        let info = parser.info();

        // XYB images store data differently - not actual YCbCr
        if info.is_xyb {
            return Err(Error::UnsupportedFeature {
                feature: "YCbCr output not available for XYB images",
            });
        }

        // Grayscale images have only Y component
        if info.color_space == ColorSpace::Grayscale {
            return Err(Error::UnsupportedFeature {
                feature: "YCbCr output requires 3-component image",
            });
        }

        // Get the YCbCr planes directly
        let (y, cb, cr) = parser.to_ycbcr_planes_f32(self.config.fancy_upsampling)?;

        // Pass through ICC profile if present
        let icc_profile = parser.icc_profile.clone();

        Ok(DecodedYCbCr {
            y,
            cb,
            cr,
            width: info.dimensions.width,
            height: info.dimensions.height,
            icc_profile,
        })
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a scan needed for scanline reading.
pub(crate) struct ScanInfo {
    /// Huffman table mapping: (dc_table_idx, ac_table_idx) per component
    pub table_mapping: [(usize, usize); 3],
    /// Position in data where entropy-coded scan data begins
    pub data_start: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{EncoderConfig, PixelLayout};
    use enough::Unstoppable;

    #[test]
    fn test_decoder_creation() {
        let decoder = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .fancy_upsampling(true);

        assert_eq!(decoder.config.output_format, Some(PixelFormat::Rgb));
        assert!(decoder.config.fancy_upsampling);
    }

    #[test]
    fn test_encode_decode_roundtrip_gray() {
        // Create a simple 8x8 grayscale image
        let width = 8u32;
        let height = 8u32;
        let mut input = vec![0u8; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                input[y * width as usize + x] = ((x + y) * 16) as u8;
            }
        }

        // Encode using v2 API
        let config = EncoderConfig::new().quality(95.0).grayscale();
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Verify JPEG structure
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8); // SOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9); // EOI

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder.decode(&jpeg).expect("decoding should succeed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.data.len(), (width * height) as usize);

        // Check pixel values are reasonably close (JPEG is lossy)
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.data[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        // At quality 95, differences should be small
        assert!(max_diff < 20, "max_diff {} too large", max_diff);
    }

    #[test]
    fn test_encode_decode_roundtrip_rgb() {
        // Create a simple 16x16 RGB image
        let width = 16u32;
        let height = 16u32;
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode using v2 API
        let config = EncoderConfig::new().quality(95.0);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded = decoder.decode(&jpeg).expect("decoding should succeed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.data.len(), (width * height * 3) as usize);

        // Check pixel values are reasonably close
        let mut max_diff = 0i32;
        for i in 0..input.len() {
            let diff = (input[i] as i32 - decoded.data[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        // At quality 95, differences should be small
        assert!(max_diff < 30, "max_diff {} too large", max_diff);
    }

    #[test]
    fn test_decode_f32_roundtrip() {
        // Create a simple 16x16 RGB image
        let width = 16u32;
        let height = 16u32;
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode using v2 API
        let config = EncoderConfig::new().quality(95.0);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Decode to f32
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded_f32 = decoder
            .decode_f32(&jpeg)
            .expect("f32 decoding should succeed");

        assert_eq!(decoded_f32.width, width);
        assert_eq!(decoded_f32.height, height);
        assert_eq!(decoded_f32.data.len(), (width * height * 3) as usize);

        // Verify values are in 0.0-1.0 range
        for &v in &decoded_f32.data {
            assert!(v >= 0.0 && v <= 1.0, "f32 value {} out of range", v);
        }

        // Compare with u8 decode - converted f32 should match
        let decoded_u8 = decoder.decode(&jpeg).expect("u8 decoding should succeed");
        let converted_u8 = decoded_f32.to_u8();

        // Values should be close - allow diff of 2 because u8 path uses integer IDCT
        // while f32 path uses f32 IDCT (standard JPEG precision difference)
        let mut max_diff = 0i32;
        for i in 0..decoded_u8.data.len() {
            let diff = (decoded_u8.data[i] as i32 - converted_u8.data[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff <= 2,
            "f32→u8 conversion differs by {} from direct u8",
            max_diff
        );
    }

    #[test]
    fn test_decode_f32_precision() {
        // Create a gradient image to test precision
        let width = 64u32;
        let height = 64u32;
        let mut input = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                // Create a smooth gradient
                let val = ((x + y) * 2) as u8;
                input[idx] = val;
                input[idx + 1] = val;
                input[idx + 2] = val;
            }
        }

        // Encode at high quality using v2 API
        let config = EncoderConfig::new().quality(98.0);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("encoder creation should succeed");
        enc.push_packed(&input, Unstoppable)
            .expect("push should succeed");
        let jpeg = enc.finish().expect("encoding should succeed");

        // Decode to f32
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded_f32 = decoder
            .decode_f32(&jpeg)
            .expect("f32 decoding should succeed");

        // Check that f32 values show more precision than just u8/255
        // by verifying we have non-quantized intermediate values
        let mut found_fractional = false;
        for &v in &decoded_f32.data {
            let scaled = v * 255.0;
            let frac = scaled - scaled.round();
            if frac.abs() > 0.001 && frac.abs() < 0.999 {
                found_fractional = true;
                break;
            }
        }
        // f32 should preserve sub-integer precision
        assert!(
            found_fractional,
            "f32 output should have fractional precision"
        );
    }
}
