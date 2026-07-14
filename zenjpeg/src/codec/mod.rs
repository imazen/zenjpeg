//! zencodec trait implementations for zenjpeg.
//!
//! Provides [`JpegEncoderConfig`] and [`JpegDecoderConfig`] types that implement
//! the encode/decode trait hierarchy from zencodec, wrapping the native
//! zenjpeg API.
//!
//! The native API remains untouched — this is a thin adapter layer.
//!
//! # Trait mapping
//!
//! | zencodec | zenjpeg adapter |
//! |----------------|-----------------|
//! | `EncoderConfig` | [`JpegEncoderConfig`] |
//! | `EncodeJob<'a>` | [`JpegEncodeJob`] |
//! | `Encoder` | [`JpegEncoder`] |
//! | `AnimationFrameEncoder` | `()` (JPEG has no animation) |
//! | `DecoderConfig` | [`JpegDecoderConfig`] |
//! | `DecodeJob<'a>` | [`JpegDecodeJob`] |
//! | `Decode` | [`JpegDecoder`] |
//! | `StreamingDecode` | [`JpegStreamingDecoder`] |
//! | `AnimationFrameDecode` | `Unsupported<At<CodecError>>` (JPEG has no animation) |
//!
//! # Error envelope (Pattern B)
//!
//! Every trait impl below uses `type Error = At<zencodec::CodecError>` — the
//! shared envelope — so a generic consumer recovers the
//! [`ErrorCategory`](zencodec::ErrorCategory) and codec name even after the
//! concrete error is erased to `BoxedError` through `Dyn*` dispatch. The native
//! [`Error`]/[`ErrorKind`] remain the detail + category source, bridged into the
//! envelope by `From<Error>`/`From<ErrorKind> for At<CodecError>` (see
//! `crate::error`); `?` and `.into()` carry codec internals across the boundary
//! unchanged.

mod decode;
mod encode;
mod info;
mod streaming;

#[cfg(test)]
mod tests;

pub use decode::{CmykHandling, JpegDecodeJob, JpegDecoder, JpegDecoderConfig};
pub use encode::{JpegEncodeJob, JpegEncoder, JpegEncoderConfig};
pub use streaming::JpegStreamingDecoder;

/// Alias for backwards compatibility within the `zencodec` feature gate.
pub type JpegEncoding = JpegEncoderConfig;
/// Alias for backwards compatibility within the `zencodec` feature gate.
pub type JpegDecoding = JpegDecoderConfig;
