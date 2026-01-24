//! JPEG Decoder - Public API.
//!
//! This module provides everything needed for JPEG decoding.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use zenjpeg::decoder::{Decoder, DecodedImage, Result};
//!
//! fn decode_jpeg(data: &[u8]) -> Result<DecodedImage> {
//!     Decoder::new().decode(data)
//! }
//! ```
//!
//! # Decode to specific format
//!
//! ```rust,ignore
//! use zenjpeg::decoder::{Decoder, PixelFormat, Result};
//!
//! fn decode_rgba(data: &[u8]) -> Result<Vec<u8>> {
//!     let image = Decoder::new()
//!         .output_format(PixelFormat::Rgba)
//!         .decode(data)?;
//!     Ok(image.into_pixels())
//! }
//! ```

// Note: Currently re-exporting internal error types since the decoder
// types we re-export from crate::decode use them internally.
// TODO: Create wrapper types or unified error type in the future.

// === Error types ===
// Re-export internal error types since the decoder types use them
pub use crate::error::{Error, Result};

// === Main decoder types ===
pub use crate::decode::{
    DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig, JpegInfo, ScanlineInfo,
    ScanlineReader,
};

// === Metadata preservation types ===
pub use crate::decode::{
    AdobeColorTransform, AdobeInfo, DecodedExtras, DensityUnits, IccPreserve, JfifInfo,
    MpfDirectory, MpfEntry, MpfImageType, PreserveConfig, PreservedMpfImage, PreservedSegment,
    SegmentType, StandardProfile,
};

// === Types used in public structs ===
pub use crate::types::{ColorSpace, Dimensions, JpegMode, PixelFormat, Subsampling};

// === ICC profile support ===
#[cfg(any(feature = "cms-lcms2", feature = "cms-moxcms"))]
pub use crate::color::icc::{decode_jpeg_with_icc, extract_icc_profile};
