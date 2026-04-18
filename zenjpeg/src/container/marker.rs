//! Zero-copy iterator over top-level JPEG marker segments.
//!
//! # Why this module exists
//!
//! Prior to this module, at least four independent implementations of
//! "walk a JPEG marker stream" existed across `zenjpeg`, `ultrahdr-core`,
//! and `ultrahdr-rs` — with subtly different bugs (see the regression
//! tests in this file). This is now the **single** canonical
//! implementation. Everything that needs to locate JPEG images, pull
//! APP segments, or skip past embedded thumbnails goes through
//! [`MarkerIter`].
//!
//! # Design
//!
//! - Zero allocation. Iterator yields [`MarkerSpan`] values that borrow
//!   from the input buffer.
//! - Entropy-aware. SOS segments are yielded covering the SOS header
//!   AND the entropy-coded scan up to (but not including) the next
//!   real marker. FF00 byte-stuffing and FFD0..FFD7 restart markers
//!   inside an entropy stream are correctly skipped.
//! - Length-aware for APPn, DQT, DHT, SOFn, DRI, DNL, COM and any
//!   other marker that carries a big-endian length word. This prevents
//!   the scanner from looking inside APP1 EXIF payloads for "EOI"
//!   bytes (the historical "embedded thumbnail EOI" bug).
//! - memchr-backed. Scanning for the next `0xFF` byte both in an
//!   entropy-coded scan AND in the gap between concatenated images
//!   uses [`memchr::memchr`] (AVX2/NEON SIMD, ~15 GB/s on modern x86).
//! - Never panics. Malformed input (length overflow, missing EOI,
//!   truncated mid-segment) terminates iteration cleanly.
//!
//! # Example
//!
//! ```no_run
//! use zenjpeg::container::marker;
//!
//! let jpeg: &[u8] = &[]; // your bytes here
//! for span in marker::iter(jpeg) {
//!     println!("{:?} at offset {} (len {})", span.kind, span.offset, span.length);
//! }
//!
//! if let Some(range) = marker::primary_bounds(jpeg) {
//!     let primary = &jpeg[range];
//!     // feed `primary` to a JPEG decoder
//! }
//! ```

use alloc::vec::Vec;
use core::ops::Range;

/// Classified JPEG marker.
///
/// JPEG markers are two bytes: `0xFF` followed by an identifier byte. This
/// enum groups them by structural role. The inner `u8` for [`Sof`],
/// [`App`], [`Restart`], and [`Other`] is the low byte of the marker so
/// the original value can be recovered.
///
/// [`Sof`]: MarkerKind::Sof
/// [`App`]: MarkerKind::App
/// [`Restart`]: MarkerKind::Restart
/// [`Other`]: MarkerKind::Other
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MarkerKind {
    /// Start of Image (0xFFD8). Standalone marker; empty payload.
    Soi,
    /// End of Image (0xFFD9). Standalone marker; empty payload.
    Eoi,
    /// Start of Frame. Low nibble of the marker byte (SOF0 = 0, SOF2 = 2, …).
    /// Baseline = 0, Extended sequential = 1, Progressive = 2, Lossless = 3,
    /// Differential variants = 5..7, arithmetic variants = 9..11, 13..15.
    Sof(u8),
    /// Define Quantization Table (0xFFDB).
    Dqt,
    /// Define Huffman Table (0xFFC4).
    Dht,
    /// Start of Scan (0xFFDA). The [`MarkerSpan`] covers the SOS header
    /// plus the entropy-coded scan up to the next real marker.
    Sos,
    /// Define Restart Interval (0xFFDD).
    Dri,
    /// Define Number of Lines (0xFFDC).
    Dnl,
    /// Comment (0xFFFE).
    Com,
    /// APPn marker (0xFFE0..0xFFEF). `u8` is the application index (0..15).
    App(u8),
    /// Restart marker RSTn (0xFFD0..0xFFD7). `u8` is the index (0..7).
    /// Should normally appear only inside an entropy-coded scan; if
    /// surfaced at top level it indicates a malformed stream.
    Restart(u8),
    /// Any other marker the scanner doesn't classify specifically. `u8`
    /// is the raw marker identifier (the byte after 0xFF).
    Other(u8),
}

/// A single top-level JPEG marker segment.
///
/// Borrows from the input buffer with lifetime `'a`. For length-bearing
/// segments, [`payload`] is the bytes AFTER the 2-byte length field. For
/// standalone markers (SOI, EOI, restarts), [`payload`] is empty.
///
/// For the Sos variant, [`payload`] is the SOS header body (component
/// selectors and scan parameters) — NOT the entropy-coded stream.
/// [`length`] includes the entropy-coded stream, however, so the next
/// marker begins at `offset + length`.
///
/// [`payload`]: MarkerSpan::payload
/// [`length`]: MarkerSpan::length
#[derive(Debug, Copy, Clone)]
pub struct MarkerSpan<'a> {
    /// Classified marker type.
    pub kind: MarkerKind,
    /// Byte offset within the iterated buffer of the leading `0xFF`.
    pub offset: usize,
    /// Total length of this segment in bytes (marker bytes + optional
    /// length word + payload, plus entropy-coded stream for SOS).
    pub length: usize,
    /// Zero-copy slice of the segment's payload, following the length
    /// word. Empty for standalone markers.
    pub payload: &'a [u8],
}

/// Begin iteration over top-level JPEG markers in `data`.
///
/// See the [module docs](self) for semantics.
#[inline]
#[must_use]
pub fn iter(data: &'_ [u8]) -> MarkerIter<'_> {
    MarkerIter::new(data)
}

/// Iterator over top-level JPEG marker segments.
///
/// Construct with [`iter`] or [`MarkerIter::new`]. Yields exactly one
/// [`MarkerSpan`] per top-level segment and stops after the first EOI is
/// emitted, when the input is exhausted, or on malformed structure.
pub struct MarkerIter<'a> {
    data: &'a [u8],
    pos: usize,
    /// `true` after we yield an EOI span. Subsequent `.next()` returns `None`.
    ended: bool,
}

impl<'a> MarkerIter<'a> {
    /// Construct a new iterator over `data`.
    #[inline]
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, ended: false }
    }

    /// Current byte offset within the input buffer.
    ///
    /// After iteration ends, this is the offset of the first unconsumed
    /// byte (either past the EOI, past the malformed segment, or at
    /// `data.len()`).
    #[inline]
    #[must_use]
    pub fn position(&self) -> usize {
        self.pos
    }
}

impl<'a> Iterator for MarkerIter<'a> {
    type Item = MarkerSpan<'a>;

    fn next(&mut self) -> Option<MarkerSpan<'a>> {
        if self.ended {
            return None;
        }
        // We need at least 2 bytes for a marker.
        if self.pos + 1 >= self.data.len() {
            return None;
        }
        // Any leading non-FF byte at the top level is malformed.
        if self.data[self.pos] != 0xFF {
            return None;
        }
        // Skip pre-marker fill bytes: JPEG permits any number of 0xFF
        // padding bytes before the real marker identifier.
        while self.pos + 1 < self.data.len() && self.data[self.pos + 1] == 0xFF {
            self.pos += 1;
        }
        if self.pos + 1 >= self.data.len() {
            return None;
        }

        let marker_offset = self.pos;
        let m = self.data[self.pos + 1];
        let kind = classify(m);

        // 1) Standalone markers (no length, no payload).
        if is_standalone(m) {
            self.pos += 2;
            if m == MARKER_EOI {
                self.ended = true;
            }
            return Some(MarkerSpan {
                kind,
                offset: marker_offset,
                length: 2,
                payload: &[],
            });
        }

        // 2) SOS — length word, then header, then entropy-coded stream
        //    terminated by the next real marker.
        if m == MARKER_SOS {
            let (payload, header_end) = match read_length_and_payload(self.data, self.pos) {
                Some(x) => x,
                None => {
                    self.pos = self.data.len();
                    return None;
                }
            };
            let scan_end = skip_entropy_scan(self.data, header_end);
            let length = scan_end - marker_offset;
            self.pos = scan_end;
            return Some(MarkerSpan {
                kind,
                offset: marker_offset,
                length,
                payload,
            });
        }

        // 3) Any other length-bearing marker.
        let (payload, seg_end) = match read_length_and_payload(self.data, self.pos) {
            Some(x) => x,
            None => {
                self.pos = self.data.len();
                return None;
            }
        };
        let length = seg_end - marker_offset;
        self.pos = seg_end;
        Some(MarkerSpan {
            kind,
            offset: marker_offset,
            length,
            payload,
        })
    }
}

// ───────────────────────────────────────────────────────────────────────
// Convenience helpers
// ───────────────────────────────────────────────────────────────────────

/// Locate the first top-level JPEG image in `data`.
///
/// Returns a byte range `start..end` where `end` is 2 past the first EOI
/// marker, or `None` if `data` does not start with SOI, or no matching
/// EOI is reachable, or the structure is malformed.
///
/// For multi-image files (Ultra HDR, depth maps, JPEG-MPF) this returns
/// just the primary. Use [`find_jpeg_boundaries`] for all images.
#[must_use]
pub fn primary_bounds(data: &[u8]) -> Option<Range<usize>> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    for span in iter(data) {
        if matches!(span.kind, MarkerKind::Eoi) {
            return Some(0..span.offset + span.length);
        }
    }
    None
}

/// Locate every top-level JPEG image in `data`.
///
/// Each returned range is `[SOI_offset .. EOI_end]`. Ranges are in order
/// of appearance and never overlap. Bytes that are neither inside an
/// image nor between images are skipped via memchr, so long inter-image
/// gaps cost only a SIMD byte search.
///
/// # Example: Ultra HDR file
///
/// An Ultra HDR JPEG contains a primary JPEG followed by a gain map JPEG.
/// This function returns two ranges — the primary at the start, and the
/// gain map after MPF padding.
#[must_use]
pub fn find_jpeg_boundaries(data: &[u8]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        // memchr to the next 0xFF byte.
        let Some(rel) = memchr::memchr(0xFF, &data[pos..]) else {
            break;
        };
        let soi = pos + rel;
        if soi + 1 >= data.len() {
            break;
        }
        if data[soi + 1] != 0xD8 {
            // Not an SOI — keep searching.
            pos = soi + 1;
            continue;
        }
        // Found SOI; walk markers from here.
        let mut it = MarkerIter::new(&data[soi..]);
        let mut eoi_rel = None;
        for span in it.by_ref() {
            if matches!(span.kind, MarkerKind::Eoi) {
                eoi_rel = Some(span.offset + span.length);
                break;
            }
        }
        match eoi_rel {
            Some(end) => {
                out.push(soi..soi + end);
                pos = soi + end;
            }
            None => break,
        }
    }
    out
}

// ───────────────────────────────────────────────────────────────────────
// Internal helpers
// ───────────────────────────────────────────────────────────────────────

const MARKER_SOI: u8 = 0xD8;
const MARKER_EOI: u8 = 0xD9;
const MARKER_SOS: u8 = 0xDA;
const MARKER_TEM: u8 = 0x01;

#[inline]
fn classify(m: u8) -> MarkerKind {
    match m {
        0xD8 => MarkerKind::Soi,
        0xD9 => MarkerKind::Eoi,
        0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC5 | 0xC6 | 0xC7 | 0xC9 | 0xCA | 0xCB | 0xCD | 0xCE
        | 0xCF => MarkerKind::Sof(m & 0x0F),
        0xC4 => MarkerKind::Dht,
        0xDB => MarkerKind::Dqt,
        0xDA => MarkerKind::Sos,
        0xDC => MarkerKind::Dnl,
        0xDD => MarkerKind::Dri,
        0xFE => MarkerKind::Com,
        0xE0..=0xEF => MarkerKind::App(m - 0xE0),
        0xD0..=0xD7 => MarkerKind::Restart(m - 0xD0),
        other => MarkerKind::Other(other),
    }
}

/// Markers that consist of just the 2-byte marker with no length word
/// and no payload: SOI, EOI, TEM, and restart markers RST0..RST7.
#[inline]
fn is_standalone(m: u8) -> bool {
    matches!(m, MARKER_SOI | MARKER_EOI | MARKER_TEM | 0xD0..=0xD7)
}

/// Read a length word at `pos+2..pos+4` and return `(payload, end_offset)`
/// where `payload` is the bytes after the length field and `end_offset`
/// is the absolute offset one past the segment. Returns `None` on
/// truncated or malformed (length < 2, length beyond buffer) input.
#[inline]
fn read_length_and_payload(data: &[u8], pos: usize) -> Option<(&[u8], usize)> {
    if pos + 4 > data.len() {
        return None;
    }
    let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
    // Length field INCLUDES its own 2 bytes, so minimum is 2 (no payload)
    // but that's malformed for anything we parse here; accept ≥ 2 and let
    // the callers validate stricter.
    if seg_len < 2 {
        return None;
    }
    let seg_end = pos.checked_add(2)?.checked_add(seg_len)?;
    if seg_end > data.len() {
        return None;
    }
    let payload = &data[pos + 4..seg_end];
    Some((payload, seg_end))
}

/// Walk the entropy-coded byte stream starting at `start` and return the
/// offset of the next real marker's `0xFF` (or `data.len()` if none).
///
/// FF00 byte-stuffing and FFD0..FFD7 restart markers are not real
/// markers; they are skipped. Any other FFxx is a real marker.
///
/// Uses `memchr::memchr` for the FF-search so the bulk of entropy data
/// is scanned at memory-bandwidth-like throughput rather than scalar
/// byte-compare.
#[inline]
fn skip_entropy_scan(data: &[u8], start: usize) -> usize {
    let mut pos = start;
    while pos < data.len() {
        let Some(rel) = memchr::memchr(0xFF, &data[pos..]) else {
            return data.len();
        };
        let ff = pos + rel;
        if ff + 1 >= data.len() {
            return data.len();
        }
        let nm = data[ff + 1];
        if nm == 0x00 {
            // Stuffed literal FF — advance past the pair.
            pos = ff + 2;
            continue;
        }
        if (0xD0..=0xD7).contains(&nm) {
            // Restart marker — advance past.
            pos = ff + 2;
            continue;
        }
        // Real marker — that's where the scan ends.
        return ff;
    }
    data.len()
}

// ───────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimum well-formed JPEG: SOI + SOF0 + DQT + DHT + SOS + (entropy) + EOI.
    fn tiny_jpeg() -> Vec<u8> {
        let mut v = Vec::new();
        // SOI
        v.extend_from_slice(&[0xFF, 0xD8]);
        // SOF0 length=17 (3 components), minimal
        v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
        v.extend_from_slice(&[0x08, 0x00, 0x10, 0x00, 0x10, 0x03]); // precision, h, w, nc
        v.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        // DQT (minimal, length=5)
        v.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x05, 0x00, 0x00, 0x00]);
        // DHT (minimal, length=5)
        v.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x05, 0x00, 0x00, 0x00]);
        // SOS (length=8, 1 component scan)
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        // Entropy: arbitrary bytes, plus a stuffed FF00 and a restart RST0.
        v.extend_from_slice(&[0xAA, 0xBB, 0xFF, 0x00, 0xCC, 0xFF, 0xD0, 0xDD, 0xEE]);
        // EOI
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    #[test]
    fn iter_yields_expected_kinds() {
        let data = tiny_jpeg();
        let kinds: Vec<MarkerKind> = iter(&data).map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                MarkerKind::Soi,
                MarkerKind::Sof(0),
                MarkerKind::Dqt,
                MarkerKind::Dht,
                MarkerKind::Sos,
                MarkerKind::Eoi,
            ]
        );
    }

    #[test]
    fn iter_sos_span_covers_entropy_and_stuffing() {
        let data = tiny_jpeg();
        let sos = iter(&data)
            .find(|s| matches!(s.kind, MarkerKind::Sos))
            .expect("sos");
        // Span must reach up to but not including the next marker (EOI).
        let next = sos.offset + sos.length;
        assert_eq!(data[next], 0xFF);
        assert_eq!(data[next + 1], 0xD9);
        // Payload (header body) is 6 bytes: scan component header.
        assert_eq!(sos.payload.len(), 6);
    }

    #[test]
    fn iter_stops_after_eoi() {
        // Append trailing junk after EOI; iterator must not emit anything past it.
        let mut data = tiny_jpeg();
        data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let count = iter(&data).count();
        assert_eq!(count, 6);
    }

    #[test]
    fn iter_handles_ff_padding_before_marker() {
        // Insert FFFF padding before SOF0.
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF]); // fill bytes
        data.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]); // SOF0 marker + length
        data.extend_from_slice(&[0x08, 0x00, 0x10, 0x00, 0x10, 0x03]);
        data.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        data.extend_from_slice(&[0xFF, 0xD9]); // EOI
        let kinds: Vec<_> = iter(&data).map(|s| s.kind).collect();
        assert_eq!(kinds, vec![MarkerKind::Soi, MarkerKind::Sof(0), MarkerKind::Eoi]);
    }

    #[test]
    fn iter_stops_on_truncated_segment() {
        // SOI + APP0 with length=1000 but data ends at byte 10.
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x03, 0xE8, 0x00, 0x00, 0x00, 0x00];
        let spans: Vec<_> = iter(&data).collect();
        // Must yield SOI but then stop cleanly.
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].kind, MarkerKind::Soi);
    }

    #[test]
    fn iter_rejects_zero_length_field() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x00, 0xFF, 0xD9];
        let spans: Vec<_> = iter(&data).collect();
        // SOI then stops on bad length.
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn iter_rejects_length_one() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01, 0xFF, 0xD9];
        let spans: Vec<_> = iter(&data).collect();
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn iter_length_exactly_equals_buffer_end() {
        // APP0 with length=4 (just the length word + 2 bytes), then EOI.
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0xAA, 0xBB, 0xFF, 0xD9];
        let kinds: Vec<_> = iter(&data).map(|s| s.kind).collect();
        assert_eq!(kinds, vec![MarkerKind::Soi, MarkerKind::App(0), MarkerKind::Eoi]);
    }

    #[test]
    fn iter_empty_input() {
        assert_eq!(iter(&[]).count(), 0);
        assert_eq!(iter(&[0xFF]).count(), 0);
    }

    #[test]
    fn iter_non_jpeg_input() {
        let data = b"hello world";
        assert_eq!(iter(data).count(), 0);
    }

    #[test]
    fn iter_all_app_markers() {
        let mut data = vec![0xFF, 0xD8];
        for n in 0..=15u8 {
            data.extend_from_slice(&[0xFF, 0xE0 + n, 0x00, 0x02]); // zero-payload APPn
        }
        data.extend_from_slice(&[0xFF, 0xD9]);
        let kinds: Vec<_> = iter(&data).map(|s| s.kind).collect();
        let mut expected = vec![MarkerKind::Soi];
        expected.extend((0..=15u8).map(MarkerKind::App));
        expected.push(MarkerKind::Eoi);
        // Note: payload=2 (length includes self) with 0 content bytes is
        // technically malformed (length must be >=2 minimum), our parser
        // accepts len==2 as "empty payload" — verify.
        assert_eq!(kinds, expected);
    }

    #[test]
    fn iter_restart_marker_at_top_level_surfaces_as_restart() {
        // Unusual but parseable: standalone RST1 between segments.
        let data = [
            0xFF, 0xD8, // SOI
            0xFF, 0xD1, // RST1 (standalone)
            0xFF, 0xD9, // EOI
        ];
        let kinds: Vec<_> = iter(&data).map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![MarkerKind::Soi, MarkerKind::Restart(1), MarkerKind::Eoi]
        );
    }

    #[test]
    fn skip_entropy_honors_stuffing() {
        // FF00 FF00 FF00 FFD0 FFAA ← the FFAA is the first "real" marker.
        let data = [0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xD0, 0xFF, 0xAA];
        assert_eq!(skip_entropy_scan(&data, 0), 6);
    }

    #[test]
    fn primary_bounds_basic() {
        let data = tiny_jpeg();
        let r = primary_bounds(&data).expect("primary");
        assert_eq!(r.start, 0);
        assert_eq!(r.end, data.len());
    }

    #[test]
    fn primary_bounds_none_on_non_jpeg() {
        assert_eq!(primary_bounds(b"hello world"), None);
        assert_eq!(primary_bounds(&[]), None);
    }

    #[test]
    fn primary_bounds_none_on_missing_eoi() {
        let mut data = tiny_jpeg();
        // Strip the final EOI.
        assert_eq!(&data[data.len() - 2..], &[0xFF, 0xD9]);
        data.truncate(data.len() - 2);
        assert_eq!(primary_bounds(&data), None);
    }

    /// Regression: a primary JPEG with an APP1 EXIF segment that contains
    /// an embedded thumbnail JPEG must be parsed as ONE image ending at
    /// the outer EOI — NOT split at the thumbnail's EOI.
    #[test]
    fn primary_bounds_skips_embedded_thumbnail_in_app1() {
        let mut data = vec![0xFF, 0xD8];
        // Embedded thumbnail "JPEG" inside APP1 EXIF.
        let mut thumb = Vec::new();
        thumb.extend_from_slice(&[0xFF, 0xD8]);
        thumb.extend_from_slice(&[0x00; 30]);
        thumb.extend_from_slice(&[0xFF, 0xD9]);
        let exif_header = b"Exif\0\0";
        let app1_payload_len = exif_header.len() + thumb.len();
        let app1_seg_len = 2 + app1_payload_len;
        data.push(0xFF);
        data.push(0xE1);
        data.extend_from_slice(&(app1_seg_len as u16).to_be_bytes());
        data.extend_from_slice(exif_header);
        data.extend_from_slice(&thumb);
        // Minimal SOF0
        data.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
        data.extend_from_slice(&[0x08, 0x00, 0x08, 0x00, 0x08, 0x03]);
        data.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        // SOS
        data.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08]);
        data.extend_from_slice(&[0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        data.extend_from_slice(&[0xAB, 0xCD]);
        let outer_eoi = data.len();
        data.extend_from_slice(&[0xFF, 0xD9]);

        let r = primary_bounds(&data).expect("primary");
        assert_eq!(r.end, outer_eoi + 2);
    }

    #[test]
    fn find_jpeg_boundaries_two_images() {
        let j1 = tiny_jpeg();
        let j2 = tiny_jpeg();
        let mut data = j1.clone();
        data.extend_from_slice(&[0x00; 32]); // inter-image gap
        data.extend_from_slice(&j2);

        let rs = find_jpeg_boundaries(&data);
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0], 0..j1.len());
        assert_eq!(rs[1], j1.len() + 32..data.len());
    }

    #[test]
    fn find_jpeg_boundaries_empty() {
        assert!(find_jpeg_boundaries(&[]).is_empty());
        assert!(find_jpeg_boundaries(b"not a jpeg").is_empty());
    }

    #[test]
    fn find_jpeg_boundaries_ranges_are_disjoint_and_ordered() {
        let j = tiny_jpeg();
        let mut data = Vec::new();
        data.extend_from_slice(&j);
        data.extend_from_slice(&j);
        data.extend_from_slice(&j);

        let rs = find_jpeg_boundaries(&data);
        assert_eq!(rs.len(), 3);
        for pair in rs.windows(2) {
            assert!(pair[0].end <= pair[1].start);
        }
    }

    #[test]
    fn position_tracks_progress() {
        let data = tiny_jpeg();
        let mut it = MarkerIter::new(&data);
        let _soi = it.next().unwrap();
        assert_eq!(it.position(), 2);
    }
}
