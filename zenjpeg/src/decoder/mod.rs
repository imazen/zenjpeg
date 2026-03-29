//! JPEG Decoder - Public API.
//!
//! This module provides everything needed for JPEG decoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use zenjpeg::decoder::{Decoder, DecodeResult, Result};
//!
//! fn decode_jpeg(data: &[u8]) -> Result<DecodeResult> {
//!     Decoder::new().decode(data, enough::Unstoppable)
//! }
//! ```
//!
//! # Decode to specific format
//!
//! ```rust,ignore
//! use zenjpeg::decoder::{Decoder, OutputTarget, PixelFormat, Result};
//!
//! fn decode_f32(data: &[u8]) -> Result<Vec<f32>> {
//!     let result = Decoder::new()
//!         .output_target(OutputTarget::SrgbF32)
//!         .decode(data, enough::Unstoppable)?;
//!     Ok(result.into_pixels_f32().unwrap())
//! }
//! ```
//!
//! # Resource Limits and Cancellation
//!
//! Protect against malicious images and support cooperative cancellation:
//!
//! ```rust,ignore
//! use zenjpeg::decoder::Decoder;
//! use zenjpeg::types::Limits;
//! use enough::Unstoppable;
//!
//! // Set resource limits (DoS protection)
//! let decoder = Decoder::new()
//!     .max_pixels(100_000_000)      // 100 megapixels max
//!     .max_memory(512_000_000);     // 512 MB max allocation
//!
//! // Or use Limits struct
//! let limits = Limits {
//!     max_pixels: Some(100_000_000),
//!     max_memory: Some(512_000_000),
//!     max_output: None,
//! };
//! let decoder = Decoder::new().limits(limits);
//!
//! // Custom stop token for cancellation
//! let result = decoder.decode(data, &my_cancel_token)?;
//! ```

// Note: Currently re-exporting internal error types since the decoder
// types we re-export from crate::decode use them internally.
// === Error types ===
/// Errors that can occur during JPEG decoding.
pub type DecodeError = crate::error::Error;
// Keep legacy aliases for backward compatibility
pub use crate::error::{Error, Result};

// === Main decoder types ===
pub use crate::decode::{
    ChromaUpsampling, CropRegion, DecodeConfig, DecodeInfo, DecodePool, DecodeRequest,
    DecodeResult, DecodeWarning, DecodedCoefficients, DecodedImage, DecodedImageF32, DecodedYCbCr,
    Decoder, GainMapHandling, GainMapResult, JpegInfo, OutputTarget, RowSlice, RowSliceF32,
    ScanlineInfo, ScanlineReader, Strictness,
};

// === Metadata preservation types ===
pub use crate::decode::{
    AdobeColorTransform, AdobeInfo, DecodedExtras, DensityUnits, IccPreserve, JfifInfo,
    MpfDirectory, MpfEntry, MpfImageType, MpfImageTypeExt, PreserveConfig, PreservedMpfImage,
    PreservedSegment, SegmentType, StandardProfile,
};

// === Depth map extraction types ===
pub use crate::decode::{
    DepthMapData, DepthSource, GDepthFormat, GDepthMeasureType, GDepthMetadata, GDepthUnits,
};

// === Types used in public structs ===
pub use crate::types::{ColorSpace, Dimensions, JpegMode, Limits, PixelFormat, Subsampling};
// Also re-export PixelLayout from encoder for easy conversion
pub use crate::encode::encoder_types::PixelLayout;

// === ICC profile support ===
pub use crate::color::icc::TargetColorSpace;
#[cfg(feature = "moxcms")]
pub use crate::color::icc::extract_icc_profile;
