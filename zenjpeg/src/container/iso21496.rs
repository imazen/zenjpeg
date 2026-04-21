//! ISO 21496-1 JPEG APP2 envelope (internal).
//!
//! Wraps the ISO 21496-1 binary gain-map metadata payload (defined and
//! parsed by [`zencodec::gainmap`]) in the JPEG-specific envelope:
//!
//! - `FF E2` APP2 marker
//! - 2-byte big-endian length field
//! - [`ISO_21496_1_URN`] namespace string (28 bytes including trailing NUL)
//! - payload (produced or consumed by `zencodec::gainmap`)
//!
//! # Scope
//!
//! This module owns *only* the JPEG envelope. The inner payload (flags,
//! fractions, channel count) is codec-agnostic and lives in
//! [`zencodec::gainmap`]. AVIF's `tmap` item and JXL's `jhgm` box use the
//! same payload with different framings.
//!
//! # Provisional: `pub(crate)`
//!
//! The module is internal. Downstream Ultra HDR writers (e.g. ultrahdr-rs)
//! consume a higher-level zenjpeg encode API rather than these primitives
//! directly. This ceiling lets the envelope helpers migrate into
//! `zencodec` wholesale later without breaking zenjpeg's public surface.
//!
//! # Allocation profile
//!
//! - [`PRIMARY_APP2`] is a compile-time constant — the primary JPEG's
//!   version-only marker has no variable inputs.
//! - [`append_gainmap_app2`] performs one zencodec payload alloc (owned
//!   by `zencodec::gainmap`), then appends envelope bytes in place.
//! - [`append_app2_marker`] is a pure in-place writer.
//!
//! Nothing in this module allocates an intermediate envelope buffer.

// The module is a provisional staging ground: helpers land here as
// zenjpeg wires them into higher-level encode/decode paths, with the
// eventual destination being `zencodec::gainmap::envelope` or similar.
// Until those call sites land, the public-to-crate helpers read as
// dead code to clippy — silence it at module scope with rationale.
#![allow(dead_code)]

use alloc::vec::Vec;
use thiserror::Error;

use super::marker::{MarkerKind, iter};

pub(crate) use zencodec::Iso21496Format;

/// The ISO 21496-1 namespace string that prefixes every gain-map APP2
/// marker payload. 28 bytes including the trailing NUL.
///
/// Defined by ISO/IEC 21496-1. libultrahdr writes the same byte sequence
/// (`libultrahdr/lib/src/jpegr.cpp:69` via `kIsoNameSpace`).
pub(crate) const ISO_21496_1_URN: &[u8; 28] = b"urn:iso:std:iso:ts:21496:-1\0";

/// The canonical primary-JPEG ISO 21496-1 APP2 marker.
///
/// 36 bytes, fully fixed: `FF E2 00 22 <URN> 00 00 00 00`. Written into
/// the primary (SDR) JPEG of a canonical Ultra HDR file to signal
/// ISO 21496-1 awareness without carrying gain-map parameters.
///
/// The gain-map (secondary) JPEG's APP2 is variable-length — use
/// [`append_gainmap_app2`] for that.
pub(crate) const PRIMARY_APP2: &[u8; 36] = &[
    0xFF, 0xE2, // APP2 marker
    0x00, 0x22, // length field = 34 (= 2 + 28 URN + 4 version)
    // URN: "urn:iso:std:iso:ts:21496:-1\0"
    b'u', b'r', b'n', b':', b'i', b's', b'o', b':', b's', b't', b'd', b':', b'i', b's', b'o', b':',
    b't', b's', b':', b'2', b'1', b'4', b'9', b'6', b':', b'-', b'1', 0x00,
    // 4-byte version payload: min_version=0, writer_version=0
    0x00, 0x00, 0x00, 0x00,
];

// Sanity: the constant IS exactly what `append_app2_marker` would produce
// for a 4-byte zero payload. Enforced by a compile-time check on length;
// runtime equivalence is covered by the `primary_app2_equals_appended`
// test below.
const _: () = assert!(PRIMARY_APP2.len() == 36);

/// Errors from parsing an ISO 21496-1 JPEG envelope.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(crate) enum Error {
    /// No APP2 segment with the ISO 21496-1 URN was found.
    #[error("ISO 21496-1 APP2 segment not found")]
    NotFound,

    /// The inner payload failed to parse.
    #[error("inner ISO 21496-1 payload parse failed: {0}")]
    Payload(#[source] zencodec::gainmap::GainMapParseError),
}

/// Scan a full JPEG byte buffer for its ISO 21496-1 APP2 segment and
/// parse the gain-map metadata payload.
///
/// Returns `NotFound` if no APP2 segment carries the [`ISO_21496_1_URN`]
/// prefix. `Payload(..)` if the inner payload fails to parse.
pub(crate) fn parse_app2(
    data: &[u8],
    format: Iso21496Format,
) -> Result<zencodec::GainMapParams, Error> {
    for span in iter(data) {
        if !matches!(span.kind, MarkerKind::App(2)) {
            continue;
        }
        if !span.payload.starts_with(ISO_21496_1_URN) {
            continue;
        }
        let inner = &span.payload[ISO_21496_1_URN.len()..];
        return zencodec::gainmap::parse_iso21496_fmt(inner, format).map_err(Error::Payload);
    }
    Err(Error::NotFound)
}

/// Append the ISO 21496-1 APP2 marker carrying gain-map metadata to `dst`.
///
/// Serializes `metadata` via [`zencodec::gainmap::serialize_iso21496_fmt`]
/// then wraps with the ISO URN + APP2 framing. Used in the gain-map
/// (secondary) JPEG of a canonical Ultra HDR file.
pub(crate) fn append_gainmap_app2(dst: &mut Vec<u8>, metadata: &zencodec::GainMapParams) {
    let payload = zencodec::gainmap::serialize_iso21496_fmt(metadata, Iso21496Format::JpegApp2);
    append_app2_marker(dst, &payload);
}

/// Append a raw `FF E2 <len> <URN> <payload>` APP2 marker to `dst`.
///
/// Low-level primitive: `payload` is written verbatim after the URN with
/// no validation. Prefer [`PRIMARY_APP2`] or [`append_gainmap_app2`] for
/// the canonical Ultra HDR markers.
pub(crate) fn append_app2_marker(dst: &mut Vec<u8>, payload: &[u8]) {
    let total_length = 2 + ISO_21496_1_URN.len() + payload.len();
    dst.reserve(2 + total_length);
    dst.push(0xFF);
    dst.push(0xE2);
    dst.extend_from_slice(&(total_length as u16).to_be_bytes());
    dst.extend_from_slice(ISO_21496_1_URN);
    dst.extend_from_slice(payload);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> zencodec::GainMapParams {
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

    const ISO_FLOAT_TOL: f64 = 1e-5;

    #[test]
    fn urn_is_28_bytes_with_null_terminator() {
        assert_eq!(ISO_21496_1_URN.len(), 28);
        assert_eq!(ISO_21496_1_URN[27], 0);
        assert_eq!(&ISO_21496_1_URN[..27], b"urn:iso:std:iso:ts:21496:-1");
    }

    /// Lock in that the precomputed [`PRIMARY_APP2`] constant byte-exactly
    /// matches what `append_app2_marker` would produce for the zero
    /// version payload. If the URN or framing ever changes, both the
    /// constant and the appender update in lockstep and this catches
    /// any drift.
    #[test]
    fn primary_app2_equals_appended() {
        let mut v = Vec::new();
        append_app2_marker(&mut v, &[0x00, 0x00, 0x00, 0x00]);
        assert_eq!(v.as_slice(), PRIMARY_APP2.as_slice());
    }

    #[test]
    fn primary_app2_layout() {
        assert_eq!(PRIMARY_APP2[0], 0xFF);
        assert_eq!(PRIMARY_APP2[1], 0xE2);
        assert_eq!(u16::from_be_bytes([PRIMARY_APP2[2], PRIMARY_APP2[3]]), 34);
        assert_eq!(
            &PRIMARY_APP2[4..4 + ISO_21496_1_URN.len()],
            ISO_21496_1_URN.as_slice()
        );
        assert_eq!(&PRIMARY_APP2[32..36], &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn append_gainmap_app2_roundtrip_via_parse_app2() {
        let metadata = test_metadata();
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8]); // SOI
        append_gainmap_app2(&mut jpeg, &metadata);
        // Minimal SOF + SOS + EOI so marker iter has structure.
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
        jpeg.extend_from_slice(&[0x08, 0x00, 0x08, 0x00, 0x08, 0x03]);
        jpeg.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        jpeg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let parsed = parse_app2(&jpeg, Iso21496Format::JpegApp2).unwrap();
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < ISO_FLOAT_TOL);
    }

    #[test]
    fn append_app2_marker_layout_matches_spec() {
        let payload = b"hello";
        let mut v = Vec::new();
        append_app2_marker(&mut v, payload);
        assert_eq!(v[0], 0xFF);
        assert_eq!(v[1], 0xE2);
        let len = u16::from_be_bytes([v[2], v[3]]);
        assert_eq!(len as usize, 2 + ISO_21496_1_URN.len() + payload.len());
        assert_eq!(&v[4..4 + ISO_21496_1_URN.len()], ISO_21496_1_URN.as_slice());
        assert_eq!(&v[4 + ISO_21496_1_URN.len()..], payload);
    }

    #[test]
    fn parse_app2_not_found_on_plain_jpeg() {
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
        let err = parse_app2(&jpeg, Iso21496Format::JpegApp2).unwrap_err();
        assert!(matches!(err, Error::NotFound));
    }

    #[test]
    fn parse_app2_not_found_when_app2_has_other_urn() {
        let mut jpeg = vec![0xFF, 0xD8];
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let payload = b"ICC_PROFILE\0\x00\x01";
        let len = 2 + payload.len();
        jpeg.extend_from_slice(&(len as u16).to_be_bytes());
        jpeg.extend_from_slice(payload);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        assert!(matches!(
            parse_app2(&jpeg, Iso21496Format::JpegApp2),
            Err(Error::NotFound)
        ));
    }

    #[test]
    fn parse_app2_no_panic_on_garbage() {
        let _ = parse_app2(&[], Iso21496Format::JpegApp2);
        let garbage: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let _ = parse_app2(&garbage, Iso21496Format::JpegApp2);
    }

    /// Regression: a crafted APP2 that claims to carry ISO 21496-1 but
    /// truncates mid-URN must not panic or false-match.
    #[test]
    fn parse_app2_rejects_truncated_urn() {
        let mut jpeg = vec![0xFF, 0xD8];
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let partial = &ISO_21496_1_URN[..10];
        let len = 2 + partial.len();
        jpeg.extend_from_slice(&(len as u16).to_be_bytes());
        jpeg.extend_from_slice(partial);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        assert!(matches!(
            parse_app2(&jpeg, Iso21496Format::JpegApp2),
            Err(Error::NotFound)
        ));
    }
}
