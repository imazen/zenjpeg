//! # jpegli
//!
//! Rust port of jpegli - an improved JPEG encoder and decoder.
//!
//! jpegli provides enhanced compression quality compared to standard JPEG
//! through advanced quantization, optional XYB color space, and other
//! perceptual optimizations.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! // Encode RGB image to JPEG (quality 85, 1-100 scale)
//! let jpeg_data = jpegli::encode_rgb(640, 480, &rgb_pixels, 85)?;
//!
//! // Decode JPEG to RGB
//! let image = jpegli::decode(&jpeg_data)?;
//! let pixels: &[u8] = image.pixels();
//! ```
//!
//! ## Advanced Usage
//!
//! For more control, use the builder API:
//!
//! ```rust,ignore
//! use jpegli::{JpegEncoder, Subsampling};
//!
//! let jpeg = JpegEncoder::new(640, 480)
//!     .quality(85)                    // 1-100 scale
//!     .subsampling(Subsampling::S420) // 4:2:0 chroma subsampling
//!     .progressive(true)              // Progressive JPEG
//!     .encode(&rgb_pixels)?;
//! ```
//!
//! ## Features
//!
//! - **Baseline JPEG**: Standard 8-bit JPEG encoding/decoding
//! - **Progressive JPEG**: Multi-scan progressive encoding
//! - **XYB Color Space**: Perceptually optimized color space for better quality
//! - **Adaptive Quantization**: Content-aware quantization for improved detail
//! - **16-bit Support**: High bit-depth input/output
//! - **Parallel Encoding**: Multi-threaded encoding (with `parallel` feature)

// Lint configuration is in workspace Cargo.toml [workspace.lints.clippy]
#![allow(missing_docs)]
#![allow(clippy::module_name_repetitions)]

// ============================================================================
// Module structure
// ============================================================================

// ============================================================================
// Public Modules
// ============================================================================

pub mod decode;
pub mod encode;
pub mod error;

// Internal modules - hidden but accessible for tests/examples
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod foundation;
#[doc(hidden)]
pub mod quant;
#[doc(hidden)]
pub mod types;
#[doc(hidden)]
pub mod encode_simd;
#[doc(hidden)]
pub mod entropy;
#[doc(hidden)]
pub mod huffman;
#[doc(hidden)]
pub mod test_utils;

// Hybrid quantization (jpegli AQ + mozjpeg trellis)
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid;

// Internal re-exports for backward compatibility with internal code paths
// These allow `crate::chroma` to work as an alias for `crate::encode::chroma`
#[doc(hidden)]
pub use encode::chroma;
#[doc(hidden)]
pub use encode::dct;
#[doc(hidden)]
pub use encode::scan_script;

// ============================================================================
// Primary Encoder API (from encode::v2)
// ============================================================================

pub use encode::v2::{
    BytesEncoder, ChromaSubsampling, ColorMode, DownsamplingMethod,
    EdgePadding, EdgePaddingConfig, EncoderConfig, PixelLayout, Quality,
    QuantTableConfig, RgbEncoder, Stop, XybSubsampling, YCbCrPlanes,
    YCbCrPlanarEncoder,
};

// Internal encoder (used in doc examples)
#[allow(deprecated)]
pub use encode::Encoder;

// Backward-compatible alias for legacy code (deprecated, will be removed)
#[doc(hidden)]
#[deprecated(since = "0.5.0", note = "Use EncoderConfig instead")]
pub use encode::streaming::StreamingEncoder as JpegEncoder;
#[doc(hidden)]
#[deprecated(since = "0.5.0", note = "Use EncoderConfig instead")]
pub use encode::streaming::StreamingEncoderBuilder as JpegEncoderBuilder;

// ============================================================================
// Decoder API
// ============================================================================

pub use decode::{DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig};
pub use decode::Decoder as JpegDecoder;

// ============================================================================
// Error Types
// ============================================================================

pub use error::{Error, Result};

// ============================================================================
// Legacy/Internal Types (hidden from docs)
// ============================================================================

// Legacy types from types module
#[doc(hidden)]
pub use types::{
    ChromaDownsampling as LegacyChromaDownsampling, ColorSpace,
    HuffmanMethod, JpegMode, PixelFormat, Subsampling,
};
// Backward-compatible alias for old type name
#[doc(hidden)]
#[deprecated(since = "0.5.0", note = "Use DownsamplingMethod instead")]
pub use types::ChromaDownsampling;

// Legacy quality types
#[doc(hidden)]
pub use quant::Quality as LegacyQuality;
#[doc(hidden)]
pub use quant::quality_conversion::{QualityComparisonMetric, QualityConversion};
#[doc(hidden)]
pub use quant::{CustomQuantMatrices, QuantTable};

// Foundation utilities
#[doc(hidden)]
pub use foundation::AllocationStats;
#[doc(hidden)]
pub use foundation::{aligned_alloc, alloc, bitstream, consts, simd_types};

// Huffman internals
#[doc(hidden)]
pub use huffman::classic as huffman_classic;
#[doc(hidden)]
pub use huffman::types as huffman_types;

// Decode internals
#[doc(hidden)]
pub use decode::idct;
#[doc(hidden)]
pub use decode::idct_int;

// Color internals
#[doc(hidden)]
pub use color::icc;
#[doc(hidden)]
pub use color::xyb;

// imgref re-export
#[doc(hidden)]
pub use imgref::{Img, ImgRef, ImgRefMut, ImgVec};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Encodes an RGB image to JPEG at the specified quality.
///
/// This is the simplest way to encode a JPEG. For more control over encoding
/// options, use [`JpegEncoder`].
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `rgb_data` - RGB pixel data (3 bytes per pixel, row-major order)
/// * `quality` - JPEG quality (1-100, where 100 is best quality)
///
/// # Example
///
/// ```rust,ignore
/// let rgb_pixels: Vec<u8> = vec![128; 640 * 480 * 3];
/// let jpeg = jpegli::encode_rgb(640, 480, &rgb_pixels, 85)?;
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Dimensions are zero or exceed maximum
/// - Data size doesn't match width × height × 3
/// - Encoding fails
pub fn encode_rgb(width: u32, height: u32, rgb_data: &[u8], quality: u8) -> Result<Vec<u8>> {
    use enough::Never;
    let config = EncoderConfig::new().quality(quality.clamp(1, 100));
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(rgb_data, Never)?;
    enc.finish()
}

/// Encodes an RGBA image to JPEG at the specified quality.
///
/// The alpha channel is ignored during encoding.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `rgba_data` - RGBA pixel data (4 bytes per pixel, row-major order)
/// * `quality` - JPEG quality (1-100, where 100 is best quality)
///
/// # Example
///
/// ```rust,ignore
/// let rgba_pixels: Vec<u8> = vec![128; 640 * 480 * 4];
/// let jpeg = jpegli::encode_rgba(640, 480, &rgba_pixels, 85)?;
/// ```
pub fn encode_rgba(width: u32, height: u32, rgba_data: &[u8], quality: u8) -> Result<Vec<u8>> {
    use enough::Never;
    let config = EncoderConfig::new().quality(quality.clamp(1, 100));
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgbx8Srgb)?;
    enc.push_packed(rgba_data, Never)?;
    enc.finish()
}

/// Encodes a grayscale image to JPEG at the specified quality.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `gray_data` - Grayscale pixel data (1 byte per pixel, row-major order)
/// * `quality` - JPEG quality (1-100, where 100 is best quality)
///
/// # Example
///
/// ```rust,ignore
/// let gray_pixels: Vec<u8> = vec![128; 640 * 480];
/// let jpeg = jpegli::encode_gray(640, 480, &gray_pixels, 85)?;
/// ```
pub fn encode_gray(width: u32, height: u32, gray_data: &[u8], quality: u8) -> Result<Vec<u8>> {
    use enough::Never;
    let config = EncoderConfig::new().quality(quality.clamp(1, 100));
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(gray_data, Never)?;
    enc.finish()
}

/// Decodes a JPEG image to RGB.
///
/// Returns a [`DecodedImage`] containing the pixel data and image information.
/// By default, images are decoded to RGB format with ICC profile applied
/// (if a CMS feature is enabled).
///
/// # Example
///
/// ```rust,ignore
/// let jpeg_data = std::fs::read("photo.jpg")?;
/// let image = jpegli::decode(&jpeg_data)?;
///
/// println!("{}x{}", image.width(), image.height());
/// let rgb_pixels: &[u8] = image.pixels();
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - Data is not a valid JPEG
/// - Image dimensions exceed limits
/// - Memory allocation fails
pub fn decode(jpeg_data: &[u8]) -> Result<DecodedImage> {
    Decoder::new().decode(jpeg_data)
}

/// Decodes a JPEG image to 32-bit floating point RGB.
///
/// This preserves the full 12-bit internal precision of jpegli's decoder.
/// Values are normalized to range 0.0-1.0.
///
/// # Example
///
/// ```rust,ignore
/// let jpeg_data = std::fs::read("photo.jpg")?;
/// let image = jpegli::decode_f32(&jpeg_data)?;
///
/// println!("{}x{}", image.width(), image.height());
/// let rgb_pixels: &[f32] = image.pixels();
/// ```
pub fn decode_f32(jpeg_data: &[u8]) -> Result<DecodedImageF32> {
    Decoder::new().decode_f32(jpeg_data)
}

/// Decodes a JPEG image to the specified pixel format.
///
/// # Example
///
/// ```rust,ignore
/// use jpegli::PixelFormat;
///
/// let jpeg_data = std::fs::read("photo.jpg")?;
/// let image = jpegli::decode_to_format(&jpeg_data, PixelFormat::Rgba)?;
///
/// let rgba_pixels: &[u8] = image.pixels(); // 4 bytes per pixel
/// ```
pub fn decode_to_format(jpeg_data: &[u8], format: PixelFormat) -> Result<DecodedImage> {
    Decoder::new().output_format(format).decode(jpeg_data)
}
