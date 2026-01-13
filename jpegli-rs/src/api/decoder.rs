//! Decoder API types.
//!
//! The primary decoder entry point is [`Decoder`], which provides
//! methods for decoding JPEG images to various output formats.

// Re-export decoder types from decode module
pub use crate::decode::{
    DecodedImage, DecodedImageF32, DecodedYCbCr, Decoder, DecoderConfig, JpegInfo,
};
