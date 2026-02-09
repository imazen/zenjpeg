//! Intermediate request layer between config and encoder.
//!
//! [`EncodeRequest`] binds per-image metadata (ICC, EXIF, XMP) and controls
//! (stop token, limits) to a reusable [`EncoderConfig`] for a single encode
//! operation.
//!
//! # Usage
//!
//! ```rust,ignore
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, Exif, Orientation};
//!
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
//!     .progressive(true);
//!
//! // One-shot with per-image metadata
//! let jpeg = config.request()
//!     .icc_profile(&srgb_icc)
//!     .exif(Exif::build().orientation(Orientation::Rotate90))
//!     .encode(&pixels, 1920, 1080)?;
//!
//! // Streaming with per-image metadata
//! let mut enc = config.request()
//!     .icc_profile(&srgb_icc)
//!     .encode_from_rgb::<rgb::RGB<u8>>(1920, 1080)?;
//! enc.push_packed(&pixels, Unstoppable)?;
//! let jpeg = enc.finish()?;
//! ```

extern crate alloc;
use alloc::borrow::Cow;

use super::byte_encoders::{BytesEncoder, Pixel, RgbEncoder, YCbCrPlanarEncoder};
use super::encoder_config::EncoderConfig;
use super::encoder_types::PixelLayout;
use super::exif::Exif;
use super::extras::EncoderSegments;
use crate::error::Result;
use crate::types::Limits;
use enough::Stop;

/// Per-image encode request: binds metadata and controls to a reusable config.
///
/// Created via [`EncoderConfig::request()`]. Metadata is consumed (cloned into
/// the encoder) at build time — no lifetime accumulation on streaming encoders.
///
/// # Examples
///
/// ```rust,ignore
/// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
///
/// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
///
/// // Reuse config with different metadata per image
/// let jpeg1 = config.request()
///     .icc_profile(&srgb_bytes)
///     .encode(&pixels1, 1920, 1080)?;
///
/// let jpeg2 = config.request()
///     .icc_profile(&p3_bytes)
///     .encode(&pixels2, 3840, 2160)?;
/// ```
pub struct EncodeRequest<'a> {
    config: &'a EncoderConfig,
    icc_profile: Option<Cow<'a, [u8]>>,
    exif: Option<Exif>,
    xmp: Option<Cow<'a, [u8]>>,
    segments: Option<EncoderSegments>,
    stop: Option<&'a dyn Stop>,
    limits: Option<Limits>,
}

impl<'a> EncodeRequest<'a> {
    /// Create a new request from a config reference.
    pub(crate) fn new(config: &'a EncoderConfig) -> Self {
        Self {
            config,
            icc_profile: None,
            exif: None,
            xmp: None,
            segments: None,
            stop: None,
            limits: None,
        }
    }

    // === Metadata ===

    /// Attach an ICC color profile (borrowed).
    #[must_use]
    pub fn icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(Cow::Borrowed(data));
        self
    }

    /// Attach an ICC color profile (owned).
    #[must_use]
    pub fn icc_profile_owned(mut self, data: alloc::vec::Vec<u8>) -> Self {
        self.icc_profile = Some(Cow::Owned(data));
        self
    }

    /// Attach EXIF metadata.
    ///
    /// Accepts [`Exif::raw()`][Exif::raw] for raw bytes or
    /// [`Exif::build()`][Exif::build] for field-based construction.
    #[must_use]
    pub fn exif(mut self, exif: impl Into<Exif>) -> Self {
        self.exif = Some(exif.into());
        self
    }

    /// Attach XMP metadata (borrowed).
    #[must_use]
    pub fn xmp(mut self, data: &'a [u8]) -> Self {
        self.xmp = Some(Cow::Borrowed(data));
        self
    }

    /// Attach XMP metadata (owned).
    #[must_use]
    pub fn xmp_owned(mut self, data: alloc::vec::Vec<u8>) -> Self {
        self.xmp = Some(Cow::Owned(data));
        self
    }

    /// Attach encoder segments (for metadata round-tripping from decoded images).
    #[must_use]
    pub fn segments(mut self, segments: EncoderSegments) -> Self {
        self.segments = Some(segments);
        self
    }

    // === Controls ===

    /// Set a cooperative cancellation token for one-shot encode methods.
    ///
    /// For streaming encoders, pass the stop token to each `push*()` call instead.
    #[must_use]
    pub fn stop(mut self, stop: &'a dyn Stop) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set resource limits for encoding.
    #[must_use]
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = Some(limits);
        self
    }

    // === Build Streaming Encoders ===

    /// Create a streaming encoder from raw byte input with explicit pixel layout.
    ///
    /// Metadata from the request is passed to the encoder.
    pub fn encode_from_bytes(
        self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<BytesEncoder> {
        let (config, icc, exif, xmp, _segments) = self.extract_metadata();
        BytesEncoder::new(config, width, height, layout, icc, exif, xmp)
    }

    /// Create a streaming encoder from `rgb` crate pixel types.
    ///
    /// Metadata from the request is passed to the encoder.
    pub fn encode_from_rgb<P: Pixel>(self, width: u32, height: u32) -> Result<RgbEncoder<P>> {
        let (config, icc, exif, xmp, _segments) = self.extract_metadata();
        RgbEncoder::new(config, width, height, icc, exif, xmp)
    }

    /// Create a streaming encoder from planar YCbCr data.
    ///
    /// Metadata from the request is passed to the encoder.
    pub fn encode_from_ycbcr_planar(self, width: u32, height: u32) -> Result<YCbCrPlanarEncoder> {
        let (config, icc, exif, xmp, _segments) = self.extract_metadata();
        YCbCrPlanarEncoder::new(config, width, height, icc, exif, xmp)
    }

    // === One-Shot Convenience ===

    /// Encode a complete image from `rgb` crate pixel types in one call.
    ///
    /// Uses the request's stop token (if set) for cooperative cancellation.
    /// Metadata (EXIF, ICC, XMP) from the request is included in the output.
    pub fn encode<P: Pixel>(
        self,
        pixels: &[P],
        width: u32,
        height: u32,
    ) -> Result<alloc::vec::Vec<u8>> {
        let stop = self.stop.unwrap_or(&enough::Unstoppable);
        let mut enc = self.encode_from_rgb::<P>(width, height)?;
        enc.push_packed(pixels, stop)?;
        enc.finish()
    }

    /// Encode a complete image into a caller-provided buffer.
    ///
    /// Uses the request's stop token (if set) for cooperative cancellation.
    /// Metadata (EXIF, ICC, XMP) from the request is included in the output.
    pub fn encode_into<P: Pixel>(
        self,
        pixels: &[P],
        width: u32,
        height: u32,
        output: &mut alloc::vec::Vec<u8>,
    ) -> Result<()> {
        let stop = self.stop.unwrap_or(&enough::Unstoppable);
        let mut enc = self.encode_from_rgb::<P>(width, height)?;
        enc.push_packed(pixels, stop)?;
        enc.finish_into(output)
    }

    /// Encode a complete image from raw byte data in one call.
    ///
    /// Uses the request's stop token (if set) for cooperative cancellation.
    /// Metadata (EXIF, ICC, XMP) from the request is included in the output.
    pub fn encode_bytes(
        self,
        data: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<alloc::vec::Vec<u8>> {
        let stop = self.stop.unwrap_or(&enough::Unstoppable);
        let mut enc = self.encode_from_bytes(width, height, layout)?;
        enc.push_packed(data, stop)?;
        enc.finish()
    }

    /// Encode a complete image from raw byte data into a caller-provided buffer.
    ///
    /// Uses the request's stop token (if set) for cooperative cancellation.
    /// Metadata (EXIF, ICC, XMP) from the request is included in the output.
    pub fn encode_bytes_into(
        self,
        data: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        output: &mut alloc::vec::Vec<u8>,
    ) -> Result<()> {
        let stop = self.stop.unwrap_or(&enough::Unstoppable);
        let mut enc = self.encode_from_bytes(width, height, layout)?;
        enc.push_packed(data, stop)?;
        enc.finish_into(output)
    }

    // === Internal ===

    /// Extract metadata for encoder construction.
    fn extract_metadata(
        self,
    ) -> (
        EncoderConfig,
        Option<alloc::vec::Vec<u8>>,
        Option<Exif>,
        Option<alloc::vec::Vec<u8>>,
        Option<EncoderSegments>,
    ) {
        let config = self.config.clone();
        let icc = self.icc_profile.map(|c| c.into_owned());
        let exif = self.exif;
        let xmp = self.xmp.map(|c| c.into_owned());
        let segments = self.segments;
        (config, icc, exif, xmp, segments)
    }
}
