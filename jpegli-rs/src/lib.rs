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

// Public modules (stable API)
pub mod aligned_alloc;
pub mod decode;
pub mod encode;
pub mod error;
pub mod pixel;
pub mod quality_conversion;
pub mod quant;
pub mod simd_types;
pub mod types;

// Internal modules - NOT part of the stable public API.
// These are hidden from documentation and may change without notice.
// Use at your own risk.
#[doc(hidden)]
pub mod adaptive_quant;
#[doc(hidden)]
pub mod chroma;
#[doc(hidden)]
pub mod color;
#[doc(hidden)]
pub mod dct;

// Foundation module (low-level utilities)
#[doc(hidden)]
pub mod foundation;

// Backward-compatible re-exports from foundation
#[doc(hidden)]
pub use foundation::alloc;
#[doc(hidden)]
pub use foundation::bitstream;
#[doc(hidden)]
pub use foundation::consts;
#[doc(hidden)]
pub mod encode_simd;
#[doc(hidden)]
pub mod entropy;

// Huffman module (encoding, tables, optimization)
#[doc(hidden)]
pub mod huffman;
#[doc(hidden)]
pub mod huffman_opt;

// Backward-compatible re-exports from huffman module
#[doc(hidden)]
pub use huffman::classic as huffman_classic;
#[doc(hidden)]
pub use huffman::types as huffman_types;
#[doc(hidden)]
pub mod icc;
#[doc(hidden)]
pub mod idct;
#[doc(hidden)]
pub mod idct_int;
#[doc(hidden)]
pub mod scan_script;
#[doc(hidden)]
pub mod simplified_quant;
#[doc(hidden)]
pub mod tone_mapping;
#[doc(hidden)]
pub mod transfer_functions;
#[doc(hidden)]
pub mod xyb;

// Hybrid quantization (jpegli AQ + mozjpeg trellis)
#[cfg(feature = "experimental-hybrid-trellis")]
pub mod hybrid;

// Test utilities (available for tests and examples)
// Hidden from docs but always available for integration tests
#[doc(hidden)]
pub mod test_utils;

// ============================================================================
// Re-exports for public API
// ============================================================================

pub use error::{Error, Result};
pub use types::{
    ChromaDownsampling, ColorSpace, EdgePadding, EdgePaddingConfig, HuffmanMethod, JpegMode,
    OutputDataType, PixelFormat, SampleDepth, Subsampling,
};

// Backward compatibility alias
#[deprecated(since = "0.4.0", note = "Use ChromaDownsampling instead")]
pub use types::ChromaDownsampling as ChromaConversion;

// Encoder API - new names
pub use encode::streaming::StreamingEncoder as JpegEncoder;
pub use encode::streaming::StreamingEncoderBuilder as JpegEncoderBuilder;

// Encoder API - old names (deprecated aliases)
#[deprecated(since = "0.5.0", note = "Use JpegEncoder instead")]
pub use encode::streaming::StreamingEncoder;
#[deprecated(since = "0.5.0", note = "Use JpegEncoderBuilder instead")]
pub use encode::streaming::StreamingEncoderBuilder;

// Legacy encoder (deprecated)
#[doc(hidden)]
#[allow(deprecated)]
pub use encode::Encoder;
pub use encode::EncoderConfig;

// Decoder API - new name for consistency
pub use decode::Decoder as JpegDecoder;

// Decoder API - original names (kept for compatibility)
pub use decode::{DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig};

// Quality settings
pub use quality_conversion::{QualityComparisonMetric, QualityConversion};
pub use quant::{CustomQuantMatrices, Quality, QuantTable};

// Allocation tracking
pub use foundation::AllocationStats;

// Pixel types for typed encode/decode API
pub use pixel::{Gray16, Gray8, Pixel, RGB16, RGB8, RGBA16, RGBA8};

// Re-export imgref for convenient image buffer handling
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
    JpegEncoder::new(width, height)
        .quality(Quality::from_quality(f32::from(quality.clamp(1, 100))))
        .pixel_format(PixelFormat::Rgb)
        .encode_all(rgb_data)
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
    JpegEncoder::new(width, height)
        .quality(Quality::from_quality(f32::from(quality.clamp(1, 100))))
        .pixel_format(PixelFormat::Rgba)
        .encode_all(rgba_data)
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
    JpegEncoder::new(width, height)
        .quality(Quality::from_quality(f32::from(quality.clamp(1, 100))))
        .pixel_format(PixelFormat::Gray)
        .encode_all(gray_data)
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
