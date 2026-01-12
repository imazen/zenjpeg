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

use crate::alloc::{
    checked_size_2d, try_alloc_dct_blocks, validate_dimensions, DEFAULT_MAX_MEMORY,
    DEFAULT_MAX_PIXELS,
};
use crate::color::{
    gray_f32_to_gray_f32, gray_f32_to_gray_u8, gray_f32_to_rgb_f32, gray_f32_to_rgb_u8,
    ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8,
};
use crate::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_COM, MARKER_DHT, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
    MAX_COMPONENTS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result};
use crate::huffman::HuffmanDecodeTable;
#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
use crate::icc::apply_icc_transform;
use crate::icc::{extract_icc_profile, is_xyb_profile};
use crate::idct::inverse_dct_8x8;
use crate::quant::{dequantize_block, dequantize_block_with_bias, DequantBiasStats};
use crate::types::{ColorSpace, Component, Dimensions, JpegMode, PixelFormat};

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
    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
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
    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
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
            use wide::f32x8;
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
            use wide::f32x8;
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

/// Internal JPEG parser state.
struct JpegParser<'a> {
    data: &'a [u8],
    position: usize,

    // Frame info
    width: u32,
    height: u32,
    precision: u8,
    num_components: u8,
    mode: JpegMode,

    // Component info
    components: [Component; MAX_COMPONENTS],

    // Tables
    quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; MAX_QUANT_TABLES],
    dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],

    // Restart
    restart_interval: u16,

    // Decoded coefficient data
    coeffs: Vec<Vec<[i16; DCT_BLOCK_SIZE]>>, // Per component

    // ICC profile (extracted from raw data, not during parsing)
    icc_profile: Option<Vec<u8>>,

    // Security limits
    max_pixels: u64,
}

impl<'a> JpegParser<'a> {
    fn new(data: &'a [u8], max_pixels: u64) -> Result<Self> {
        // Check for SOI
        if data.len() < 2 || data[0] != 0xFF || data[1] != MARKER_SOI {
            return Err(Error::InvalidJpegData {
                reason: "missing SOI marker",
            });
        }

        // Extract ICC profile from raw data upfront
        let icc_profile = extract_icc_profile(data);

        Ok(Self {
            data,
            position: 2,
            width: 0,
            height: 0,
            precision: 8,
            num_components: 0,
            mode: JpegMode::Baseline,
            components: std::array::from_fn(|_| Component::default()),
            quant_tables: [None, None, None, None],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            restart_interval: 0,
            coeffs: Vec::new(),
            icc_profile,
            max_pixels,
        })
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::TruncatedData {
                context: "reading marker data",
            });
        }
        let byte = self.data[self.position];
        self.position += 1;
        Ok(byte)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let high = self.read_u8()? as u16;
        let low = self.read_u8()? as u16;
        Ok((high << 8) | low)
    }

    fn read_marker(&mut self) -> Result<u8> {
        loop {
            // Skip until we find 0xFF
            let byte = self.read_u8()?;
            if byte != 0xFF {
                continue;
            }

            // Skip fill bytes (consecutive 0xFF)
            loop {
                let marker = self.read_u8()?;
                if marker == 0xFF {
                    // Fill byte, keep looking
                    continue;
                }
                if marker == 0x00 {
                    // Byte stuffing (0xFF 0x00 = literal 0xFF in data)
                    // This shouldn't happen in marker parsing, but skip it
                    break;
                }
                // Found a real marker
                return Ok(marker);
            }
        }
    }

    fn read_header(&mut self) -> Result<()> {
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOF0 | MARKER_SOF1 => {
                    self.mode = JpegMode::Baseline;
                    self.parse_frame_header()?;
                    return Ok(());
                }
                MARKER_SOF2 => {
                    self.mode = JpegMode::Progressive;
                    self.parse_frame_header()?;
                    return Ok(());
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                MARKER_EOI => {
                    return Err(Error::InvalidJpegData {
                        reason: "unexpected EOI before frame header",
                    });
                }
                _ => self.skip_segment()?,
            }
        }
    }

    fn parse_frame_header(&mut self) -> Result<()> {
        let length = self.read_u16()?;
        if length < 8 {
            return Err(Error::InvalidJpegData {
                reason: "frame header too short",
            });
        }

        self.precision = self.read_u8()?;
        // Validate precision: must be 8 for baseline JPEG, 8 or 12 for extended
        if self.precision != 8 && self.precision != 12 {
            return Err(Error::InvalidJpegData {
                reason: "invalid data precision (must be 8 or 12)",
            });
        }

        self.height = self.read_u16()? as u32;
        self.width = self.read_u16()? as u32;

        // Validate dimensions against security limits
        // max_pixels == 0 means unlimited
        let effective_max = if self.max_pixels == 0 {
            u64::MAX
        } else {
            self.max_pixels
        };
        validate_dimensions(self.width, self.height, effective_max)?;

        self.num_components = self.read_u8()?;

        // Validate num_components
        if self.num_components == 0 {
            return Err(Error::InvalidJpegData {
                reason: "number of components is zero",
            });
        }
        if self.num_components > MAX_COMPONENTS as u8 {
            return Err(Error::UnsupportedFeature {
                feature: "more than 4 components",
            });
        }

        // Validate marker length matches expected size
        let expected_length = 8 + 3 * self.num_components as u16;
        if length != expected_length {
            return Err(Error::InvalidJpegData {
                reason: "SOF marker length mismatch",
            });
        }

        for i in 0..self.num_components as usize {
            self.components[i].id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let h_samp = sampling >> 4;
            let v_samp = sampling & 0x0F;

            // Validate sampling factors are non-zero and <= 4
            if h_samp == 0 || v_samp == 0 {
                return Err(Error::InvalidJpegData {
                    reason: "sampling factor is zero",
                });
            }
            if h_samp > 4 || v_samp > 4 {
                return Err(Error::InvalidJpegData {
                    reason: "sampling factor exceeds maximum (4)",
                });
            }

            self.components[i].h_samp_factor = h_samp;
            self.components[i].v_samp_factor = v_samp;

            let quant_idx = self.read_u8()?;
            // Validate quant table index
            if quant_idx as usize >= MAX_QUANT_TABLES {
                return Err(Error::InvalidJpegData {
                    reason: "quantization table index out of range",
                });
            }
            self.components[i].quant_table_idx = quant_idx;
        }

        Ok(())
    }

    fn parse_quant_table(&mut self) -> Result<()> {
        let mut length = self.read_u16()? as i32 - 2;

        while length > 0 {
            let info = self.read_u8()?;
            let precision = info >> 4;
            let table_idx = (info & 0x0F) as usize;

            // Validate precision (0 = 8-bit, 1 = 16-bit)
            if precision > 1 {
                return Err(Error::InvalidQuantTable {
                    table_idx: table_idx as u8,
                    reason: "invalid precision (must be 0 or 1)",
                });
            }

            if table_idx >= MAX_QUANT_TABLES {
                return Err(Error::InvalidQuantTable {
                    table_idx: table_idx as u8,
                    reason: "table index out of range",
                });
            }

            // Read values in zigzag order (as stored in JPEG)
            let mut zigzag_values = [0u16; DCT_BLOCK_SIZE];

            if precision == 0 {
                // 8-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = self.read_u8()? as u16;
                    if val == 0 {
                        return Err(Error::InvalidQuantTable {
                            table_idx: table_idx as u8,
                            reason: "quantization value is zero",
                        });
                    }
                    zigzag_values[i] = val;
                }
                length -= 65;
            } else {
                // 16-bit values
                for i in 0..DCT_BLOCK_SIZE {
                    let val = self.read_u16()?;
                    if val == 0 {
                        return Err(Error::InvalidQuantTable {
                            table_idx: table_idx as u8,
                            reason: "quantization value is zero",
                        });
                    }
                    zigzag_values[i] = val;
                }
                length -= 129;
            }

            // Validate DQT marker length consistency
            if length < 0 {
                return Err(Error::InvalidJpegData {
                    reason: "DQT marker length mismatch",
                });
            }

            // Convert from zigzag order to natural order for dequantization
            let mut natural_values = [0u16; DCT_BLOCK_SIZE];
            for i in 0..DCT_BLOCK_SIZE {
                natural_values[JPEG_NATURAL_ORDER[i] as usize] = zigzag_values[i];
            }

            self.quant_tables[table_idx] = Some(natural_values);
        }

        Ok(())
    }

    fn parse_huffman_table(&mut self) -> Result<()> {
        let mut length = self.read_u16()? as i32 - 2;

        while length > 0 {
            let info = self.read_u8()?;
            let table_class = info >> 4; // 0 = DC, 1 = AC
            let table_idx = (info & 0x0F) as usize;

            // Validate table class (must be 0 for DC or 1 for AC)
            if table_class > 1 {
                return Err(Error::InvalidHuffmanTable {
                    table_idx: table_idx as u8,
                    reason: "invalid table class (must be 0 or 1)",
                });
            }

            if table_idx >= MAX_HUFFMAN_TABLES {
                return Err(Error::InvalidHuffmanTable {
                    table_idx: table_idx as u8,
                    reason: "table index out of range",
                });
            }

            let mut bits = [0u8; 16];
            for i in 0..16 {
                bits[i] = self.read_u8()?;
            }

            let num_values: usize = bits.iter().map(|&b| b as usize).sum();
            let mut values = vec![0u8; num_values];
            for i in 0..num_values {
                values[i] = self.read_u8()?;
            }

            length -= 17 + num_values as i32;

            // Validate that we didn't read past the marker length
            if length < 0 {
                return Err(Error::InvalidJpegData {
                    reason: "DHT marker length mismatch",
                });
            }

            let table = HuffmanDecodeTable::from_bits_values(&bits, &values)?;

            if table_class == 0 {
                self.dc_tables[table_idx] = Some(table);
            } else {
                self.ac_tables[table_idx] = Some(table);
            }
        }

        Ok(())
    }

    fn parse_restart_interval(&mut self) -> Result<()> {
        let _length = self.read_u16()?;
        self.restart_interval = self.read_u16()?;
        Ok(())
    }

    fn skip_segment(&mut self) -> Result<()> {
        let length = self.read_u16()? as usize;
        if length < 2 {
            return Err(Error::InvalidJpegData {
                reason: "segment length too short",
            });
        }
        self.position += length - 2;
        Ok(())
    }

    fn decode(&mut self) -> Result<()> {
        // First read header
        self.position = 2; // Skip SOI
        self.read_header()?;

        // Continue parsing until we hit EOI
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOS => {
                    self.parse_scan()?;
                    // After scan, look for more markers
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_EOI => break,
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                _ => self.skip_segment()?,
            }
        }

        Ok(())
    }

    fn parse_scan(&mut self) -> Result<()> {
        let _length = self.read_u16()?;
        let num_components = self.read_u8()?;

        // Validate num_components in scan
        if num_components == 0 {
            return Err(Error::InvalidJpegData {
                reason: "SOS num_components is zero",
            });
        }
        if num_components > self.num_components {
            return Err(Error::InvalidJpegData {
                reason: "SOS num_components exceeds frame components",
            });
        }
        if num_components > MAX_COMPONENTS as u8 {
            return Err(Error::InvalidJpegData {
                reason: "SOS num_components too large",
            });
        }

        let mut scan_components = Vec::with_capacity(num_components as usize);

        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            let dc_table = tables >> 4;
            let ac_table = tables & 0x0F;

            // Validate Huffman table indexes
            if dc_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::InvalidJpegData {
                    reason: "SOS DC Huffman table index out of range",
                });
            }
            if ac_table as usize >= MAX_HUFFMAN_TABLES {
                return Err(Error::InvalidJpegData {
                    reason: "SOS AC Huffman table index out of range",
                });
            }

            // Find component index
            let comp_idx = self.components[..self.num_components as usize]
                .iter()
                .position(|c| c.id == component_id)
                .ok_or(Error::InvalidJpegData {
                    reason: "unknown component in scan",
                })?;

            scan_components.push((comp_idx, dc_table, ac_table));
        }

        let ss = self.read_u8()?; // Spectral selection start
        let se = self.read_u8()?; // Spectral selection end
        let ah_al = self.read_u8()?;
        let ah = ah_al >> 4;
        let al = ah_al & 0x0F;

        // Validate spectral selection (must be 0-63)
        if ss > 63 {
            return Err(Error::InvalidJpegData {
                reason: "SOS Ss (spectral start) out of range",
            });
        }
        if se > 63 {
            return Err(Error::InvalidJpegData {
                reason: "SOS Se (spectral end) out of range",
            });
        }

        // Decode entropy-coded segment based on mode
        if self.mode == JpegMode::Progressive {
            self.decode_progressive_scan(&scan_components, ss, se, ah, al)?;
        } else {
            self.decode_scan(&scan_components)?;
        }

        Ok(())
    }

    fn decode_scan(&mut self, scan_components: &[(usize, u8, u8)]) -> Result<()> {
        // Calculate max sampling factors to determine MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions in pixels
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;

        // Number of MCUs
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage - size depends on component's sampling factor
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            // Use explicit table if provided, otherwise use standard JPEG tables.
            // MJPEG files often omit DHT markers and expect standard tables.
            // Tables are borrowed, not cloned (~1.5KB savings per table).
            let dc_table_ref: &HuffmanDecodeTable = match &self.dc_tables[dc_idx] {
                Some(table) => table,
                None => {
                    if dc_idx == 0 {
                        HuffmanDecodeTable::std_dc_luminance()
                    } else {
                        HuffmanDecodeTable::std_dc_chrominance()
                    }
                }
            };
            decoder.set_dc_table(dc_idx, dc_table_ref);

            let ac_table_ref: &HuffmanDecodeTable = match &self.ac_tables[ac_idx] {
                Some(table) => table,
                None => {
                    if ac_idx == 0 {
                        HuffmanDecodeTable::std_ac_luminance()
                    } else {
                        HuffmanDecodeTable::std_ac_chrominance()
                    }
                }
            };
            decoder.set_ac_table(ac_idx, ac_table_ref);
        }

        // Decode MCUs with proper interleaving
        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Align to byte boundary (discard padding bits)
                    decoder.align_to_byte();
                    // Read and verify restart marker
                    decoder.read_restart_marker(next_restart_num)?;
                    // Update expected marker number (cycles 0-7)
                    next_restart_num = (next_restart_num + 1) & 7;
                    // Reset DC predictors
                    decoder.reset_dc();
                }

                // For each component in the scan
                for (comp_idx, dc_table, ac_table) in scan_components {
                    let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                    let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                    let comp_blocks_h = mcu_cols * h_samp;

                    // Calculate actual content dimensions for this component
                    // Some encoders omit padding blocks beyond the image bounds
                    let comp_width = (self.width as usize * h_samp + max_h_samp as usize - 1)
                        / max_h_samp as usize;
                    let comp_height = (self.height as usize * v_samp + max_v_samp as usize - 1)
                        / max_v_samp as usize;
                    let actual_blocks_h = (comp_width + 7) / 8;
                    let actual_blocks_v = (comp_height + 7) / 8;

                    // For single-component images with unusual sampling (grayscale with h/v > 1),
                    // some encoders omit padding blocks entirely. Detect this case.
                    let is_single_component_oversample =
                        scan_components.len() == 1 && (h_samp > 1 || v_samp > 1);

                    // Decode all blocks for this component in this MCU
                    for v in 0..v_samp {
                        for h in 0..h_samp {
                            let block_x = mcu_x * h_samp + h;
                            let block_y = mcu_y * v_samp + v;
                            let block_idx = block_y * comp_blocks_h + block_x;

                            // Check if this block is beyond actual image bounds (padding)
                            let is_padding =
                                block_x >= actual_blocks_h || block_y >= actual_blocks_v;

                            if is_padding && is_single_component_oversample {
                                // Single-component with oversampling: skip padding blocks
                                // These encoders typically omit them
                                self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                continue;
                            }

                            if is_padding {
                                // For padding blocks in multi-component images, use speculative decoding
                                // Most encoders include them, but some might not
                                let saved_state = decoder.save_state();
                                match decoder.decode_block(
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(coeffs) => {
                                        // Encoder included padding block
                                        self.coeffs[*comp_idx][block_idx] = coeffs;
                                    }
                                    Err(Error::EndOfScanData) => {
                                        // Encoder omitted padding block - restore state and fill zeros
                                        decoder.restore_state(saved_state);
                                        self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                    }
                                    Err(_e) => {
                                        // Other error - also restore and skip
                                        decoder.restore_state(saved_state);
                                        self.coeffs[*comp_idx][block_idx] = [0i16; 64];
                                        // Log but don't fail on padding block errors
                                        #[cfg(debug_assertions)]
                                        eprintln!(
                                            "DEBUG: Padding block ({},{}) error: {:?}",
                                            block_x, block_y, _e
                                        );
                                    }
                                }
                            } else {
                                let coeffs = decoder.decode_block(
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                )?;
                                self.coeffs[*comp_idx][block_idx] = coeffs;
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    fn decode_progressive_scan(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
    ) -> Result<()> {
        // Calculate max sampling factors to determine MCU structure
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions in pixels
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;

        // Number of MCUs
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        // Initialize coefficient storage if not already done
        if self.coeffs.is_empty() {
            for i in 0..self.num_components as usize {
                let h_samp = self.components[i].h_samp_factor as usize;
                let v_samp = self.components[i].v_samp_factor as usize;
                let comp_blocks_h = checked_size_2d(mcu_cols, h_samp)?;
                let comp_blocks_v = checked_size_2d(mcu_rows, v_samp)?;
                let num_blocks = checked_size_2d(comp_blocks_h, comp_blocks_v)?;
                self.coeffs.push(try_alloc_dct_blocks(
                    num_blocks,
                    "allocating DCT coefficients",
                )?);
            }
        }

        // Set up entropy decoder
        let scan_data = &self.data[self.position..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for (_comp_idx, dc_table, ac_table) in scan_components {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);

            // Use explicit table if provided, otherwise use standard JPEG tables.
            // MJPEG files often omit DHT markers and expect standard tables.
            // Tables are borrowed, not cloned (~1.5KB savings per table).
            let dc_table_ref: &HuffmanDecodeTable = match &self.dc_tables[dc_idx] {
                Some(table) => table,
                None => {
                    if dc_idx == 0 {
                        HuffmanDecodeTable::std_dc_luminance()
                    } else {
                        HuffmanDecodeTable::std_dc_chrominance()
                    }
                }
            };
            decoder.set_dc_table(dc_idx, dc_table_ref);

            let ac_table_ref: &HuffmanDecodeTable = match &self.ac_tables[ac_idx] {
                Some(table) => table,
                None => {
                    if ac_idx == 0 {
                        HuffmanDecodeTable::std_ac_luminance()
                    } else {
                        HuffmanDecodeTable::std_ac_chrominance()
                    }
                }
            };
            decoder.set_ac_table(ac_idx, ac_table_ref);
        }

        // Determine scan type
        let is_dc_scan = ss == 0 && se == 0;
        let is_first_scan = ah == 0;

        // EOB run tracking for AC scans
        let mut eob_run = 0u16;

        // Restart marker handling
        let mut mcu_count = 0u32;
        let restart_interval = self.restart_interval as u32;
        let mut next_restart_num = 0u8;

        if is_dc_scan {
            // DC scan - can be interleaved (multiple components) or non-interleaved (single component)
            // For non-interleaved scans, blocks are in raster order (like AC scans)
            // For interleaved scans, blocks follow MCU order

            if scan_components.len() == 1 {
                // Non-interleaved DC scan: blocks in raster order (like AC scans)
                let (comp_idx, dc_table, _ac_table) = scan_components[0];
                let h_samp = self.components[comp_idx].h_samp_factor as usize;
                let v_samp = self.components[comp_idx].v_samp_factor as usize;
                let comp_blocks_h = mcu_cols * h_samp;
                let comp_blocks_v = mcu_rows * v_samp;
                let total_blocks = comp_blocks_h * comp_blocks_v;

                for block_idx in 0..total_blocks {
                    // Check for restart marker
                    if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                        decoder.align_to_byte();
                        decoder.read_restart_marker(next_restart_num)?;
                        next_restart_num = (next_restart_num + 1) & 7;
                        decoder.reset_dc();
                    }

                    if is_first_scan {
                        match decoder.decode_dc_first(comp_idx, dc_table as usize, al) {
                            Ok(dc) => self.coeffs[comp_idx][block_idx][0] = dc,
                            Err(Error::EndOfScanData) => {
                                // End of scan data - remaining blocks have DC=0
                                break;
                            }
                            Err(e) => return Err(e),
                        }
                    } else {
                        match decoder.decode_dc_refine(al) {
                            Ok(bit) => self.coeffs[comp_idx][block_idx][0] |= bit,
                            Err(Error::EndOfScanData) => {
                                // End of scan data - remaining blocks unchanged
                                break;
                            }
                            Err(e) => return Err(e),
                        }
                    }

                    mcu_count += 1;
                }
            } else {
                // Interleaved DC scan: blocks in MCU order
                'dc_scan: for mcu_y in 0..mcu_rows {
                    for mcu_x in 0..mcu_cols {
                        // Check for restart marker
                        if restart_interval > 0
                            && mcu_count > 0
                            && mcu_count % restart_interval == 0
                        {
                            // Align to byte boundary (discard padding bits)
                            decoder.align_to_byte();
                            // Read and verify restart marker
                            decoder.read_restart_marker(next_restart_num)?;
                            // Update expected marker number (cycles 0-7)
                            next_restart_num = (next_restart_num + 1) & 7;
                            // Reset DC predictors
                            decoder.reset_dc();
                        }

                        for (comp_idx, dc_table, _ac_table) in scan_components {
                            let h_samp = self.components[*comp_idx].h_samp_factor as usize;
                            let v_samp = self.components[*comp_idx].v_samp_factor as usize;
                            let comp_blocks_h = mcu_cols * h_samp;

                            for v in 0..v_samp {
                                for h in 0..h_samp {
                                    let block_x = mcu_x * h_samp + h;
                                    let block_y = mcu_y * v_samp + v;
                                    let block_idx = block_y * comp_blocks_h + block_x;

                                    if is_first_scan {
                                        // DC first scan
                                        match decoder.decode_dc_first(
                                            *comp_idx,
                                            *dc_table as usize,
                                            al,
                                        ) {
                                            Ok(dc) => {
                                                self.coeffs[*comp_idx][block_idx][0] = dc;
                                            }
                                            Err(Error::EndOfScanData) => {
                                                // End of scan data - remaining blocks have DC=0
                                                break 'dc_scan;
                                            }
                                            Err(e) => return Err(e),
                                        }
                                    } else {
                                        // DC refinement scan
                                        match decoder.decode_dc_refine(al) {
                                            Ok(bit) => {
                                                self.coeffs[*comp_idx][block_idx][0] |= bit;
                                            }
                                            Err(Error::EndOfScanData) => {
                                                // End of scan data - remaining blocks unchanged
                                                break 'dc_scan;
                                            }
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                            }
                        }

                        mcu_count += 1;
                    }
                }
            }
        } else {
            // AC scan (single component only for progressive)
            // Progressive AC scans can only have one component
            if scan_components.len() != 1 {
                return Err(Error::InvalidJpegData {
                    reason: "progressive AC scan must have single component",
                });
            }

            let (comp_idx, _dc_table, ac_table) = scan_components[0];
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;

            // For non-interleaved AC scans, blocks are encoded in raster order
            // NOT in interleaved MCU order. Each MCU contains exactly 1 block.
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let total_blocks = comp_blocks_h * comp_blocks_v;

            // Reset MCU count and restart number for AC scan (each scan has its own restart sequence)
            mcu_count = 0;
            next_restart_num = 0;

            for block_idx in 0..total_blocks {
                // Check for restart marker
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Align to byte boundary (discard padding bits)
                    decoder.align_to_byte();
                    // Read and verify restart marker
                    decoder.read_restart_marker(next_restart_num)?;
                    // Update expected marker number (cycles 0-7)
                    next_restart_num = (next_restart_num + 1) & 7;
                    // Reset DC predictors and EOB run
                    decoder.reset_dc();
                    eob_run = 0;
                }

                if is_first_scan {
                    // AC first scan
                    match decoder.decode_ac_first(
                        &mut self.coeffs[comp_idx][block_idx],
                        ac_table as usize,
                        ss,
                        se,
                        al,
                        &mut eob_run,
                    ) {
                        Ok(()) => {}
                        Err(Error::EndOfScanData) => {
                            // End of scan data - remaining blocks have zeros (implicit EOB)
                            // This is normal in progressive JPEG when encoder uses
                            // implicit EOB at end of scan
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    // AC refinement scan
                    match decoder.decode_ac_refine(
                        &mut self.coeffs[comp_idx][block_idx],
                        ac_table as usize,
                        ss,
                        se,
                        al,
                        &mut eob_run,
                    ) {
                        Ok(()) => {}
                        Err(Error::EndOfScanData) => {
                            // End of scan data - remaining blocks unchanged
                            break;
                        }
                        Err(e) => return Err(e),
                    }
                }

                mcu_count += 1;
            }
        }

        self.position += decoder.position();
        Ok(())
    }

    fn info(&self) -> JpegInfo {
        let has_icc = self.icc_profile.is_some();
        let is_xyb = self.icc_profile.as_ref().is_some_and(|p| is_xyb_profile(p));

        // Determine color space, considering XYB profile
        let color_space = if is_xyb {
            ColorSpace::Xyb
        } else {
            match self.num_components {
                1 => ColorSpace::Grayscale,
                3 => ColorSpace::YCbCr,
                4 => ColorSpace::Cmyk,
                _ => ColorSpace::Unknown,
            }
        };

        JpegInfo {
            dimensions: Dimensions::new(self.width, self.height),
            color_space,
            precision: self.precision,
            num_components: self.num_components,
            mode: self.mode,
            has_icc_profile: has_icc,
            is_xyb,
        }
    }

    // Fancy upsampling with triangle filter (3:1 weights)
    // Applies separable 3:1 interpolation: (3 * near + far) / 4
    fn upsample_fancy(
        input: &[f32],
        in_width: usize,
        in_height: usize,
        out_width: usize,
        out_height: usize,
        scale_x: usize,
        scale_y: usize,
    ) -> Vec<f32> {
        // Dispatch to specialized implementation based on scale factors
        match (scale_x, scale_y) {
            (1, 1) => {
                // No upsampling needed, but still need to crop to output dimensions
                // Input may be block-aligned (e.g., 320x304) while output is image-sized (300x300)
                let mut output = vec![0.0f32; out_width * out_height];
                for y in 0..out_height {
                    let in_y = y.min(in_height.saturating_sub(1));
                    for x in 0..out_width {
                        let in_x = x.min(in_width.saturating_sub(1));
                        output[y * out_width + x] = input[in_y * in_width + in_x];
                    }
                }
                output
            }
            (2, 1) => Self::upsample_h2v1(input, in_width, in_height, out_width, out_height),
            (1, 2) => Self::upsample_h1v2(input, in_width, in_height, out_width, out_height),
            (2, 2) => Self::upsample_h2v2(input, in_width, in_height, out_width, out_height),
            _ => {
                // Fall back to box filter for unusual scale factors (e.g., 4x2)
                let mut output = vec![0.0f32; out_width * out_height];
                for y in 0..out_height {
                    let in_y = (y / scale_y).min(in_height.saturating_sub(1));
                    for x in 0..out_width {
                        let in_x = (x / scale_x).min(in_width.saturating_sub(1));
                        output[y * out_width + x] = input[in_y * in_width + in_x];
                    }
                }
                output
            }
        }
    }

    // Horizontal 2x upsampling (4:2:2)
    #[inline]
    fn upsample_h2v1(
        input: &[f32],
        in_width: usize,
        in_height: usize,
        out_width: usize,
        out_height: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; out_width * out_height];

        for y in 0..out_height {
            let in_y = y.min(in_height.saturating_sub(1));
            for out_x in 0..out_width {
                let in_x = out_x / 2;
                let curr = input[in_y * in_width + in_x];

                if out_x % 2 == 0 {
                    let left = if in_x > 0 {
                        input[in_y * in_width + in_x - 1]
                    } else {
                        curr
                    };
                    output[y * out_width + out_x] = (3.0 * curr + left) * 0.25;
                } else {
                    let right = if in_x + 1 < in_width {
                        input[in_y * in_width + in_x + 1]
                    } else {
                        curr
                    };
                    output[y * out_width + out_x] = (3.0 * curr + right) * 0.25;
                }
            }
        }

        output
    }

    // Vertical 2x upsampling (4:4:0)
    #[inline]
    fn upsample_h1v2(
        input: &[f32],
        in_width: usize,
        in_height: usize,
        out_width: usize,
        out_height: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; out_width * out_height];

        #[cfg(feature = "simd")]
        {
            use wide::f32x8;
            let three = f32x8::splat(3.0);
            let quarter = f32x8::splat(0.25);

            for out_y in 0..out_height {
                let in_y = out_y / 2;
                let is_top = out_y % 2 == 0;
                let out_row_start = out_y * out_width;

                // Get neighbor row (above for top, below for bottom)
                let neighbor_y = if is_top {
                    in_y.saturating_sub(1)
                } else {
                    (in_y + 1).min(in_height - 1)
                };

                let curr_row_start = in_y * in_width;
                let neighbor_row_start = neighbor_y * in_width;

                // SIMD path: process 8 pixels at a time in interior
                let simd_width = in_width.min(out_width);
                let chunks = simd_width / 8;

                for chunk in 0..chunks {
                    let x = chunk * 8;

                    let curr = f32x8::from([
                        input[curr_row_start + x],
                        input[curr_row_start + x + 1],
                        input[curr_row_start + x + 2],
                        input[curr_row_start + x + 3],
                        input[curr_row_start + x + 4],
                        input[curr_row_start + x + 5],
                        input[curr_row_start + x + 6],
                        input[curr_row_start + x + 7],
                    ]);
                    let neighbor = f32x8::from([
                        input[neighbor_row_start + x],
                        input[neighbor_row_start + x + 1],
                        input[neighbor_row_start + x + 2],
                        input[neighbor_row_start + x + 3],
                        input[neighbor_row_start + x + 4],
                        input[neighbor_row_start + x + 5],
                        input[neighbor_row_start + x + 6],
                        input[neighbor_row_start + x + 7],
                    ]);

                    let blended = (three * curr + neighbor) * quarter;
                    let arr: [f32; 8] = blended.into();
                    output[out_row_start + x..out_row_start + x + 8].copy_from_slice(&arr);
                }

                // Scalar remainder
                for x in (chunks * 8)..out_width {
                    let in_x = x.min(in_width.saturating_sub(1));
                    let curr = input[curr_row_start + in_x];
                    let neighbor = input[neighbor_row_start + in_x];
                    output[out_row_start + x] = (3.0 * curr + neighbor) * 0.25;
                }
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            for out_y in 0..out_height {
                for x in 0..out_width {
                    let in_x = x.min(in_width.saturating_sub(1));
                    let in_y = out_y / 2;
                    let curr = input[in_y * in_width + in_x];

                    if out_y % 2 == 0 {
                        let above = if in_y > 0 {
                            input[(in_y - 1) * in_width + in_x]
                        } else {
                            curr
                        };
                        output[out_y * out_width + x] = (3.0 * curr + above) * 0.25;
                    } else {
                        let below = if in_y + 1 < in_height {
                            input[(in_y + 1) * in_width + in_x]
                        } else {
                            curr
                        };
                        output[out_y * out_width + x] = (3.0 * curr + below) * 0.25;
                    }
                }
            }
        }

        output
    }

    // Both horizontal and vertical 2x upsampling (4:2:0)
    // Apply separably: horizontal first, then vertical
    #[inline]
    fn upsample_h2v2(
        input: &[f32],
        in_width: usize,
        in_height: usize,
        out_width: usize,
        out_height: usize,
    ) -> Vec<f32> {
        // First upsample horizontally
        let h_upsampled = Self::upsample_h2v1(input, in_width, in_height, out_width, in_height);
        // Then upsample vertically
        Self::upsample_h1v2(&h_upsampled, out_width, in_height, out_width, out_height)
    }

    fn to_pixels(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        fancy_upsampling: bool,
    ) -> Result<Vec<u8>> {
        if self.coeffs.is_empty() {
            return Err(Error::InternalError {
                reason: "no decoded data",
            });
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Pre-compute component info for efficiency
        struct CompInfo {
            quant_idx: usize,
            h_samp: usize,
            v_samp: usize,
            comp_blocks_h: usize,
            comp_blocks_v: usize,
            comp_width: usize,
            comp_height: usize,
            is_full_res: bool,
        }

        let mut comp_infos: Vec<CompInfo> = Vec::new();
        for comp_idx in 0..self.num_components as usize {
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let comp_width = checked_size_2d(comp_blocks_h, 8)?;
            let comp_height = checked_size_2d(comp_blocks_v, 8)?;
            comp_infos.push(CompInfo {
                quant_idx: self.components[comp_idx].quant_table_idx as usize,
                h_samp,
                v_samp,
                comp_blocks_h,
                comp_blocks_v,
                comp_width,
                comp_height,
                is_full_res: h_samp == max_h_samp as usize && v_samp == max_v_samp as usize,
            });
        }

        // Initialize bias stats and biases (C++ initializes to 0 via memset)
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32 (C++ jpegli keeps f32 until final output)
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row (matching C++ incremental bias recomputation)
        for imcu_row in 0..mcu_rows {
            // For each component in this MCU row
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant =
                    self.quant_tables[info.quant_idx]
                        .as_ref()
                        .ok_or(Error::InternalError {
                            reason: "missing quantization table",
                        })?;

                // Phase 1: Gather stats for full-res components
                if info.is_full_res {
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
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Phase 2: Recompute biases every 4 MCU rows (matching C++ behavior)
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // Phase 3: IDCT for this component in this MCU row
                // Store as f32 (C++ jpegli keeps f32 until final output for precision)
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

                        // Dequantize and IDCT
                        let dequant = if is_xyb {
                            dequantize_block(&natural_coeffs, quant)
                        } else {
                            dequantize_block_with_bias(&natural_coeffs, quant, biases)
                        };
                        let pixels = inverse_dct_8x8(&dequant);

                        // Store pixels - use row-based copy for efficiency
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        if cols_to_copy == DCT_SIZE {
                            // Fast path: full 8-pixel row copy
                            for y in 0..rows_to_copy {
                                let dst_offset = (base_py + y) * info.comp_width + base_px;
                                let src_offset = y * DCT_SIZE;
                                comp_plane_f32[dst_offset..dst_offset + DCT_SIZE]
                                    .copy_from_slice(&pixels[src_offset..src_offset + DCT_SIZE]);
                            }
                        } else {
                            // Slow path: partial row copy (edge blocks)
                            for y in 0..rows_to_copy {
                                for x in 0..cols_to_copy {
                                    comp_plane_f32[(base_py + y) * info.comp_width + base_px + x] =
                                        pixels[y * DCT_SIZE + x];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample if needed - keep as f32 for precision
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    // Triangle filter (3:1 weights) - separable implementation
                    // First upsample horizontally, then vertically
                    Self::upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        // Convert to output format using batch conversion functions
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and convert to u8
                let mut output = vec![0u8; output_size];
                gray_f32_to_gray_u8(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];
                gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb);
                Ok(rgb)
            }
            (3, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = vec![0u8; rgb_size];

                if is_xyb {
                    // XYB mode: Output raw level-shifted values, NO YCbCr→RGB conversion.
                    // The XYB values are stored in YCbCr positions but are NOT YCbCr.
                    // The ICC profile transforms these directly to sRGB.
                    crate::encode_simd::xyb_planes_to_rgb_u8_simd(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                } else {
                    // YCbCr to RGB conversion using batch function
                    ycbcr_planes_f32_to_rgb_u8(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
                }
                Ok(rgb)
            }
            _ => Err(Error::UnsupportedFeature {
                feature: "unsupported color conversion",
            }),
        }
    }

    /// Convert decoded coefficients to f32 pixels.
    /// Values are normalized to range 0.0-1.0.
    fn to_pixels_f32(
        &self,
        format: PixelFormat,
        is_xyb: bool,
        fancy_upsampling: bool,
    ) -> Result<Vec<f32>> {
        if self.coeffs.is_empty() {
            return Err(Error::InternalError {
                reason: "no decoded data",
            });
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Pre-compute component info
        struct CompInfo {
            quant_idx: usize,
            h_samp: usize,
            v_samp: usize,
            comp_blocks_h: usize,
            comp_blocks_v: usize,
            comp_width: usize,
            comp_height: usize,
            is_full_res: bool,
        }

        let mut comp_infos: Vec<CompInfo> = Vec::new();
        for comp_idx in 0..self.num_components as usize {
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let comp_width = checked_size_2d(comp_blocks_h, 8)?;
            let comp_height = checked_size_2d(comp_blocks_v, 8)?;
            comp_infos.push(CompInfo {
                quant_idx: self.components[comp_idx].quant_table_idx as usize,
                h_samp,
                v_samp,
                comp_blocks_h,
                comp_blocks_v,
                comp_width,
                comp_height,
                is_full_res: h_samp == max_h_samp as usize && v_samp == max_v_samp as usize,
            });
        }

        // Initialize bias stats and biases
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant =
                    self.quant_tables[info.quant_idx]
                        .as_ref()
                        .ok_or(Error::InternalError {
                            reason: "missing quantization table",
                        })?;

                // Gather stats for full-res components
                if info.is_full_res {
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
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Recompute biases every 4 MCU rows
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // IDCT for this component
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

        // Upsample if needed
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::new();

        for comp_idx in 0..self.num_components as usize {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    // Triangle filter (3:1 weights) - separable implementation
                    Self::upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        // Convert to output format as f32 (values normalized to 0.0-1.0)
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                // Grayscale: level shift and normalize to 0.0-1.0
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
                    crate::encode_simd::xyb_planes_to_rgb_f32_simd(
                        &planes_f32[0],
                        &planes_f32[1],
                        &planes_f32[2],
                        &mut rgb,
                    );
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
            _ => Err(Error::UnsupportedFeature {
                feature: "unsupported color conversion",
            }),
        }
    }

    /// Convert decoded coefficients to YCbCr f32 planes.
    ///
    /// Returns (Y, Cb, Cr) planes, each width×height in size.
    /// Values are in centered range [-128, 127] (raw DCT output).
    /// Chroma planes are upsampled to full resolution.
    fn to_ycbcr_planes_f32(&self, fancy_upsampling: bool) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.coeffs.is_empty() {
            return Err(Error::InternalError {
                reason: "no decoded data",
            });
        }

        if self.num_components != 3 {
            return Err(Error::UnsupportedFeature {
                feature: "YCbCr planes require 3-component image",
            });
        }

        let width = self.width as usize;
        let height = self.height as usize;

        // Calculate max sampling factors
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        // MCU dimensions
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        // Pre-compute component info
        struct CompInfo {
            quant_idx: usize,
            h_samp: usize,
            v_samp: usize,
            comp_blocks_h: usize,
            comp_blocks_v: usize,
            comp_width: usize,
            comp_height: usize,
            is_full_res: bool,
        }

        let mut comp_infos: Vec<CompInfo> = Vec::new();
        for comp_idx in 0..self.num_components as usize {
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let comp_width = checked_size_2d(comp_blocks_h, 8)?;
            let comp_height = checked_size_2d(comp_blocks_v, 8)?;
            comp_infos.push(CompInfo {
                quant_idx: self.components[comp_idx].quant_table_idx as usize,
                h_samp,
                v_samp,
                comp_blocks_h,
                comp_blocks_v,
                comp_width,
                comp_height,
                is_full_res: h_samp == max_h_samp as usize && v_samp == max_v_samp as usize,
            });
        }

        // Initialize bias stats and biases
        let mut bias_stats = DequantBiasStats::new(self.num_components as usize);
        let mut component_biases: Vec<[f32; DCT_BLOCK_SIZE]> =
            vec![[0.0f32; DCT_BLOCK_SIZE]; self.num_components as usize];

        // Allocate component planes as f32
        let mut comp_planes_f32: Vec<Vec<f32>> = Vec::new();
        for info in &comp_infos {
            let comp_plane_size = checked_size_2d(info.comp_width, info.comp_height)?;
            comp_planes_f32.push(vec![0.0f32; comp_plane_size]);
        }

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..self.num_components as usize {
                let info = &comp_infos[comp_idx];
                let quant =
                    self.quant_tables[info.quant_idx]
                        .as_ref()
                        .ok_or(Error::InternalError {
                            reason: "missing quantization table",
                        })?;

                // Phase 1: Gather stats for full-res components
                if info.is_full_res {
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
                            for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate()
                            {
                                natural_coeffs[zi as usize] = coeffs[i];
                            }
                            bias_stats.gather_block(comp_idx, &natural_coeffs);
                        }
                    }

                    // Recompute biases every 4 MCU rows
                    if imcu_row % 4 == 3 {
                        component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                    }
                }

                // Phase 2: IDCT
                let biases = &component_biases[comp_idx];
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

                        // Dequantize and IDCT (always use bias for non-XYB)
                        let dequant = dequantize_block_with_bias(&natural_coeffs, quant, biases);
                        let pixels = inverse_dct_8x8(&dequant);

                        // Store pixels
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        if cols_to_copy == DCT_SIZE {
                            for y in 0..rows_to_copy {
                                let dst_offset = (base_py + y) * info.comp_width + base_px;
                                let src_offset = y * DCT_SIZE;
                                comp_plane_f32[dst_offset..dst_offset + DCT_SIZE]
                                    .copy_from_slice(&pixels[src_offset..src_offset + DCT_SIZE]);
                            }
                        } else {
                            for y in 0..rows_to_copy {
                                for x in 0..cols_to_copy {
                                    comp_plane_f32[(base_py + y) * info.comp_width + base_px + x] =
                                        pixels[y * DCT_SIZE + x];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Upsample chroma and clip to image dimensions
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32: Vec<Vec<f32>> = Vec::with_capacity(3);

        for comp_idx in 0..3 {
            let info = &comp_infos[comp_idx];
            let comp_plane_f32 = &comp_planes_f32[comp_idx];

            let plane_f32 = if info.h_samp < max_h_samp as usize
                || info.v_samp < max_v_samp as usize
            {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                if fancy_upsampling {
                    Self::upsample_fancy(
                        comp_plane_f32,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    )
                } else {
                    // Box filter (nearest neighbor)
                    let mut upsampled = vec![0.0f32; output_size];
                    for py in 0..height {
                        for px in 0..width {
                            let sx = (px / scale_x).min(info.comp_width - 1);
                            let sy = (py / scale_y).min(info.comp_height - 1);
                            upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
                        }
                    }
                    upsampled
                }
            } else {
                // Full resolution - just clip to image dimensions
                let mut plane = vec![0.0f32; output_size];
                for py in 0..height {
                    for px in 0..width {
                        plane[py * width + px] = comp_plane_f32[py * info.comp_width + px];
                    }
                }
                plane
            };

            planes_f32.push(plane_f32);
        }

        Ok((
            std::mem::take(&mut planes_f32[0]),
            std::mem::take(&mut planes_f32[1]),
            std::mem::take(&mut planes_f32[2]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::Encoder;
    use crate::quant::Quality;

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
        let width = 8;
        let height = 8;
        let mut input = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                input[y * width + x] = ((x + y) * 16) as u8;
            }
        }

        // Encode
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Gray)
            .jpegli_quality(Quality::from_quality(95.0));

        let jpeg = encoder.encode(&input).expect("encoding should succeed");

        // Verify JPEG structure
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8); // SOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9); // EOI

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder.decode(&jpeg).expect("decoding should succeed");

        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.data.len(), width * height);

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
        let width = 16;
        let height = 16;
        let mut input = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(95.0));

        let jpeg = encoder.encode(&input).expect("encoding should succeed");

        // Decode
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded = decoder.decode(&jpeg).expect("decoding should succeed");

        assert_eq!(decoded.width, width as u32);
        assert_eq!(decoded.height, height as u32);
        assert_eq!(decoded.data.len(), width * height * 3);

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
        let width = 16;
        let height = 16;
        let mut input = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                input[idx] = (x * 16) as u8; // R
                input[idx + 1] = (y * 16) as u8; // G
                input[idx + 2] = 128; // B
            }
        }

        // Encode
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(95.0));

        let jpeg = encoder.encode(&input).expect("encoding should succeed");

        // Decode to f32
        let decoder = Decoder::new().output_format(PixelFormat::Rgb);
        let decoded_f32 = decoder
            .decode_f32(&jpeg)
            .expect("f32 decoding should succeed");

        assert_eq!(decoded_f32.width, width as u32);
        assert_eq!(decoded_f32.height, height as u32);
        assert_eq!(decoded_f32.data.len(), width * height * 3);

        // Verify values are in 0.0-1.0 range
        for &v in &decoded_f32.data {
            assert!(v >= 0.0 && v <= 1.0, "f32 value {} out of range", v);
        }

        // Compare with u8 decode - converted f32 should match
        let decoded_u8 = decoder.decode(&jpeg).expect("u8 decoding should succeed");
        let converted_u8 = decoded_f32.to_u8();

        // Values should be very close (within 1 due to rounding)
        let mut max_diff = 0i32;
        for i in 0..decoded_u8.data.len() {
            let diff = (decoded_u8.data[i] as i32 - converted_u8.data[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff <= 1,
            "f32→u8 conversion differs by {} from direct u8",
            max_diff
        );
    }

    #[test]
    fn test_decode_f32_precision() {
        // Create a gradient image to test precision
        let width = 64;
        let height = 64;
        let mut input = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                // Create a smooth gradient
                let val = ((x + y) * 2) as u8;
                input[idx] = val;
                input[idx + 1] = val;
                input[idx + 2] = val;
            }
        }

        // Encode at high quality
        let encoder = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(98.0));

        let jpeg = encoder.encode(&input).expect("encoding should succeed");

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
