//! ISO 21496-1 JPEG APP2 envelope.
//!
//! Wraps the ISO 21496-1 binary gain-map metadata payload (defined and
//! parsed by [`zencodec::gainmap`]) in the JPEG-specific envelope:
//!
//! - `FF E2` APP2 marker
//! - 2-byte big-endian length field
//! - [`ISO_21496_1_URN`] namespace string (28 bytes including trailing NUL)
//! - payload (produced or consumed by `zencodec::gainmap`)
//!
//! This module owns *only* the JPEG envelope. The inner payload (flags,
//! fractions, channel count) is codec-agnostic and lives in
//! [`zencodec::gainmap`]. AVIF's `tmap` item and JXL's `jhgm` box use the
//! same payload with different framings — see [`zencodec::Iso21496Format`].
//!
//! # Stable entry points
//!
//! - [`parse_iso_app2`]: scan a full JPEG buffer via
//!   [`super::marker::iter`] for the ISO APP2 segment and parse it.
//! - [`parse_iso21496`] / [`serialize_iso21496`]: direct wrappers over
//!   `zencodec::gainmap::{parse_iso21496_fmt, serialize_iso21496_fmt}`
//!   for callers who already have the payload.
//! - [`create_iso_app2_marker`]: wrap an arbitrary payload into an APP2
//!   segment with the correct URN.
//! - [`create_version_only_iso_app2`]: the 4-byte version marker that
//!   lives in the primary JPEG of a canonical Ultra HDR file.
//! - [`create_jpeg_iso_markers`]: convenience — both markers at once.

use alloc::string::ToString;
use alloc::vec::Vec;
use thiserror::Error;

use super::marker::{MarkerKind, iter};

pub use zencodec::Iso21496Format;

/// The ISO 21496-1 namespace string that prefixes every gain-map APP2
/// marker payload. 28 bytes including the trailing NUL.
///
/// Defined by ISO/IEC 21496-1. libultrahdr writes the same byte sequence
/// (`libultrahdr/lib/src/jpegr.cpp:69` via `kIsoNameSpace`).
pub const ISO_21496_1_URN: &[u8; 28] = b"urn:iso:std:iso:ts:21496:-1\0";

/// Errors from parsing an ISO 21496-1 JPEG envelope.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IsoJpegError {
    /// No APP2 segment with the ISO 21496-1 URN was found.
    #[error("ISO 21496-1 APP2 segment not found")]
    NotFound,

    /// The APP2 segment was found but its payload is shorter than the URN.
    #[error("APP2 payload too short for ISO 21496-1 URN")]
    Truncated,

    /// The inner payload failed to parse.
    #[error("inner ISO 21496-1 payload parse failed: {0}")]
    Payload(alloc::string::String),
}

/// The two ISO 21496-1 APP2 markers needed for a canonical Ultra HDR JPEG.
///
/// Returned by [`create_jpeg_iso_markers`].
#[derive(Debug, Clone)]
pub struct JpegIsoMarkers {
    /// APP2 marker for the **primary** JPEG codestream.
    ///
    /// Signals ISO 21496-1 awareness. Payload is 4 bytes of zeros
    /// (`min_version=0, writer_version=0`); does NOT carry gain map
    /// parameters — those live in the secondary JPEG's APP2.
    pub primary: Vec<u8>,

    /// APP2 marker for the **gain map** (secondary) JPEG codestream.
    ///
    /// Carries the full serialized gain map metadata.
    pub gain_map: Vec<u8>,
}

/// Scan a full JPEG byte buffer for its ISO 21496-1 APP2 segment and
/// parse the gain-map metadata payload.
///
/// Returns `NotFound` if no APP2 segment carries the [`ISO_21496_1_URN`]
/// prefix. `Truncated` if the segment is found but its payload is shorter
/// than the URN. `Payload(..)` if the inner payload fails to parse.
pub fn parse_iso_app2(
    data: &[u8],
    format: Iso21496Format,
) -> Result<zencodec::GainMapParams, IsoJpegError> {
    for span in iter(data) {
        if !matches!(span.kind, MarkerKind::App(2)) {
            continue;
        }
        if !span.payload.starts_with(ISO_21496_1_URN) {
            continue;
        }
        let inner = &span.payload[ISO_21496_1_URN.len()..];
        return zencodec::gainmap::parse_iso21496_fmt(inner, format)
            .map_err(|e| IsoJpegError::Payload(e.to_string()));
    }
    Err(IsoJpegError::NotFound)
}

/// Create both ISO 21496-1 APP2 markers for a canonical Ultra HDR JPEG
/// (primary version-only + gain map full metadata).
#[must_use]
pub fn create_jpeg_iso_markers(metadata: &zencodec::GainMapParams) -> JpegIsoMarkers {
    let iso_payload = zencodec::gainmap::serialize_iso21496_fmt(metadata, Iso21496Format::JpegApp2);
    JpegIsoMarkers {
        primary: create_version_only_iso_app2(),
        gain_map: create_iso_app2_marker(&iso_payload),
    }
}

/// Create the version-only ISO 21496-1 APP2 marker for the primary JPEG.
///
/// Signals ISO 21496-1 awareness without carrying gain-map parameters.
#[must_use]
pub fn create_version_only_iso_app2() -> Vec<u8> {
    create_iso_app2_marker(&[0x00, 0x00, 0x00, 0x00])
}

/// Wrap an arbitrary payload in a `FF E2` APP2 marker with the ISO
/// 21496-1 URN prefix.
///
/// The payload is not validated — pass whatever
/// [`zencodec::gainmap::serialize_iso21496_fmt`] produces.
#[must_use]
pub fn create_iso_app2_marker(iso_data: &[u8]) -> Vec<u8> {
    let total_length = 2 + ISO_21496_1_URN.len() + iso_data.len();
    let mut marker = Vec::with_capacity(2 + total_length);
    marker.push(0xFF);
    marker.push(0xE2);
    marker.push(((total_length >> 8) & 0xFF) as u8);
    marker.push((total_length & 0xFF) as u8);
    marker.extend_from_slice(ISO_21496_1_URN);
    marker.extend_from_slice(iso_data);
    marker
}

/// Parse the bare ISO 21496-1 payload (no APP2 envelope, no URN).
///
/// Thin wrapper over [`zencodec::gainmap::parse_iso21496_fmt`] with an
/// error type that matches the rest of this module.
pub fn parse_iso21496(
    data: &[u8],
    format: Iso21496Format,
) -> Result<zencodec::GainMapParams, IsoJpegError> {
    zencodec::gainmap::parse_iso21496_fmt(data, format)
        .map_err(|e| IsoJpegError::Payload(e.to_string()))
}

/// Serialize gain-map metadata to the bare ISO 21496-1 payload (no APP2
/// envelope, no URN).
///
/// Thin wrapper over [`zencodec::gainmap::serialize_iso21496_fmt`].
#[must_use]
pub fn serialize_iso21496(metadata: &zencodec::GainMapParams, format: Iso21496Format) -> Vec<u8> {
    zencodec::gainmap::serialize_iso21496_fmt(metadata, format)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> zencodec::GainMapParams {
        // Construct a minimal gain-map params struct with non-default
        // values so roundtrip tests detect regressions.
        let mut m = zencodec::GainMapParams::default();
        for i in 0..3 {
            m.channels[i].min = 0.0;
            m.channels[i].max = 2.0;
            m.channels[i].gamma = 1.0;
            m.channels[i].base_offset = 1.0 / 64.0;
            m.channels[i].alternate_offset = 1.0 / 64.0;
        }
        m.base_hdr_headroom = 0.0;
        m.alternate_hdr_headroom = 2.0;
        m
    }

    #[test]
    fn urn_is_28_bytes_with_null_terminator() {
        assert_eq!(ISO_21496_1_URN.len(), 28);
        assert_eq!(ISO_21496_1_URN[27], 0);
        assert_eq!(&ISO_21496_1_URN[..27], b"urn:iso:std:iso:ts:21496:-1");
    }

    /// ISO 21496-1 serializes fractions as `(numerator i32, denominator
    /// u32)` pairs. Roundtrip error is bounded by the continued-
    /// fraction approximation zencodec uses; empirically ≤ 1e-6 for
    /// the inputs we care about. A slack of 1e-5 catches truncation
    /// regressions while tolerating rounding noise.
    const ISO_FLOAT_TOL: f64 = 1e-5;

    #[test]
    fn roundtrip_jpeg_framing() {
        let original = test_metadata();
        let bytes = serialize_iso21496(&original, Iso21496Format::JpegApp2);
        let parsed = parse_iso21496(&bytes, Iso21496Format::JpegApp2).unwrap();
        // Verify every field roundtrips, not just .max.
        for i in 0..3 {
            assert!((parsed.channels[i].max - original.channels[i].max).abs() < ISO_FLOAT_TOL);
            assert!((parsed.channels[i].min - original.channels[i].min).abs() < ISO_FLOAT_TOL);
            assert!((parsed.channels[i].gamma - original.channels[i].gamma).abs() < ISO_FLOAT_TOL);
            assert!(
                (parsed.channels[i].base_offset - original.channels[i].base_offset).abs()
                    < ISO_FLOAT_TOL
            );
        }
        assert!(
            (parsed.alternate_hdr_headroom - original.alternate_hdr_headroom).abs() < ISO_FLOAT_TOL
        );
    }

    #[test]
    fn roundtrip_avif_framing() {
        let original = test_metadata();
        let bytes = serialize_iso21496(&original, Iso21496Format::AvifTmap);
        let parsed = parse_iso21496(&bytes, Iso21496Format::AvifTmap).unwrap();
        assert!((parsed.channels[0].max - original.channels[0].max).abs() < ISO_FLOAT_TOL);
        assert!(
            (parsed.alternate_hdr_headroom - original.alternate_hdr_headroom).abs() < ISO_FLOAT_TOL
        );
    }

    #[test]
    fn create_jpeg_iso_markers_both_emitted_with_app2_header() {
        let markers = create_jpeg_iso_markers(&test_metadata());
        assert!(markers.primary.len() > 4);
        assert!(markers.gain_map.len() > 4);
        assert_eq!(markers.primary[0], 0xFF);
        assert_eq!(markers.primary[1], 0xE2);
        assert_eq!(markers.gain_map[0], 0xFF);
        assert_eq!(markers.gain_map[1], 0xE2);
    }

    #[test]
    fn version_only_app2_has_expected_layout() {
        let marker = create_version_only_iso_app2();
        assert_eq!(marker[0], 0xFF);
        assert_eq!(marker[1], 0xE2);
        // 2 (marker) + 2 (length) + 28 (URN) + 4 (version payload) = 36.
        assert_eq!(marker.len(), 36);
        // Length field counts itself + URN + payload: 2 + 28 + 4 = 34.
        assert_eq!(u16::from_be_bytes([marker[2], marker[3]]), 34);
        assert_eq!(&marker[4..4 + ISO_21496_1_URN.len()], ISO_21496_1_URN);
        assert_eq!(&marker[32..36], &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn parse_iso_app2_in_full_jpeg() {
        let metadata = test_metadata();
        let markers = create_jpeg_iso_markers(&metadata);

        // Build a minimal JPEG carrying the gain-map APP2.
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
        jpeg.extend_from_slice(&markers.gain_map);
        // Tiny SOF0 + SOS + EOI so marker iter has structure.
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
        jpeg.extend_from_slice(&[0x08, 0x00, 0x08, 0x00, 0x08, 0x03]);
        jpeg.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let parsed = parse_iso_app2(&jpeg, Iso21496Format::JpegApp2).unwrap();
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < ISO_FLOAT_TOL);
    }

    #[test]
    fn parse_iso_app2_not_found_on_plain_jpeg() {
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
        let err = parse_iso_app2(&jpeg, Iso21496Format::JpegApp2).unwrap_err();
        assert!(matches!(err, IsoJpegError::NotFound));
    }

    #[test]
    fn parse_iso_app2_not_found_when_app2_has_other_urn() {
        // APP2 with ICC_PROFILE identifier instead of ISO URN.
        let mut jpeg = vec![0xFF, 0xD8];
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let payload = b"ICC_PROFILE\0\x00\x01";
        let len = 2 + payload.len();
        jpeg.extend_from_slice(&(len as u16).to_be_bytes());
        jpeg.extend_from_slice(payload);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        assert!(matches!(
            parse_iso_app2(&jpeg, Iso21496Format::JpegApp2),
            Err(IsoJpegError::NotFound)
        ));
    }

    #[test]
    fn parse_iso21496_empty_bytes_errors() {
        let err = parse_iso21496(&[], Iso21496Format::JpegApp2).unwrap_err();
        assert!(matches!(err, IsoJpegError::Payload(_)));
    }

    #[test]
    fn parse_iso_app2_no_panic_on_garbage() {
        let _ = parse_iso_app2(&[], Iso21496Format::JpegApp2);
        let garbage: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let _ = parse_iso_app2(&garbage, Iso21496Format::JpegApp2);
    }
}
