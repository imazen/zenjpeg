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

// Note: Currently re-exporting internal error types since the decoder
// types we re-export from crate::decode use them internally.
// === Error types ===
/// Errors that can occur during JPEG decoding.
pub type DecodeError = crate::error::Error;
// Keep legacy aliases for backward compatibility
pub use crate::error::{Error, Result};

// === Main decoder types ===
pub use crate::decode::{
    ChromaUpsampling, DecodeInfo, DecodeResult, DecodeWarning, Decoder, DecodedCoefficients,
    DecodedImage, DecodedImageF32, DecodedYCbCr, GainMapHandling, GainMapResult, JpegInfo,
    OutputTarget, ScanlineInfo, ScanlineReader, Strictness,
};

// === Metadata preservation types ===
pub use crate::decode::{
    AdobeColorTransform, AdobeInfo, DecodedExtras, DensityUnits, IccPreserve, JfifInfo,
    MpfDirectory, MpfEntry, MpfImageType, PreserveConfig, PreservedMpfImage, PreservedSegment,
    SegmentType, StandardProfile,
};

// === Types used in public structs ===
pub use crate::types::{ColorSpace, Dimensions, JpegMode, Limits, PixelFormat, Subsampling};
// Also re-export PixelLayout from encoder for easy conversion
pub use crate::encode::encoder_types::PixelLayout;

// === ICC profile support ===
#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
pub use crate::color::icc::{decode_jpeg_with_icc, extract_icc_profile};
