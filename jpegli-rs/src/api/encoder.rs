//! Encoder API types.
//!
//! The primary encoder entry point is [`JpegEncoder`], which provides
//! a builder-style API for encoding JPEG images.

// Re-export the streaming encoder as the main encoder
pub use crate::encode::streaming::StreamingEncoder as JpegEncoder;

// Re-export v2 config as the primary config type
// (v2 is dimension-independent and better designed)
pub use crate::encode::v2::EncoderConfig;
