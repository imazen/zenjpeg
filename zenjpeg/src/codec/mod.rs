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
//! # EXIF at the zencodec boundary
//!
//! [`zencodec::ImageInfo::exif`] and [`zencodec::Metadata::exif`] hold the
//! EXIF TIFF structure, starting with its byte-order mark (`II` or `MM`),
//! without the JPEG APP1 `Exif\0\0` header: decoding strips it and encoding
//! adds it. That is the form PNG's `eXIf` chunk holds (PNG Third Edition,
//! 11.3.4.5: "the JPEG APP1 marker, length, and the 'Exif ID code' ... are not
//! included"), so a JPEG's EXIF can go to another format as is. Encoding also
//! accepts a blob that still carries the header and writes it once. The native
//! API is unchanged: [`crate::decode::DecodedExtras::exif`] returns the APP1
//! payload, header included.
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
#[cfg(feature = "inventory")]
mod inventory;
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
