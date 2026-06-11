//! CIPA DC-007 Multi-Picture Format (MPF) APP2 segment parse + emit.
//!
//! MPF is a TIFF-structured directory embedded in a JPEG APP2 segment
//! with identifier `b"MPF\0"`. It lists the primary image and zero or
//! more secondary images with their byte offsets and sizes. Used by
//! Ultra HDR gain maps, iPhone depth maps, Pixel panoramas, and other
//! multi-image use cases.
//!
//! This module provides:
//!
//! - [`parse_mpf`]: scans a full JPEG buffer for the APP2 MPF segment
//!   via [`super::marker::iter`] and parses its directory.
//! - [`parse_mpf_segment`]: parses the TIFF data directly, given the
//!   APP2 payload (after the `MPF\0` identifier) and the absolute file
//!   offset of the TIFF header.
//! - [`create_mpf_header`] / [`create_mpf_header_typed`]: emitters for
//!   writing an APP2 MPF segment during encode.

use alloc::string::String;
use alloc::vec::Vec;
use thiserror::Error;

use super::marker::{MarkerKind, iter};
use super::types::{MpImageType, MpfEntry};

/// `"MPF\0"` — the 4-byte identifier at the start of an MPF APP2 segment.
pub const MPF_IDENTIFIER: &[u8; 4] = b"MPF\0";

/// Hard cap on declared MPF image count honored by [`parse_mpf_segment`].
///
/// Defensive ceiling against crafted inputs; real files hold single-digit
/// entries. Upstream `ultrahdr-core` didn't have an explicit cap — it
/// relied on per-iteration bounds checks to terminate early — so inputs
/// declaring >1000 entries will now silently truncate at 1000 here where
/// the old parser processed up to what fit in the payload. Documented
/// deliberate divergence.
pub const MAX_MPF_ENTRIES: usize = 1000;

/// `"0100"` — the MPF version string written into the Version tag value.
pub const MPF_VERSION: &[u8; 4] = b"0100";

// MPF tag IDs (from CIPA DC-007-2009 Annex C Table C.1).
const TAG_VERSION: u16 = 0xB000;
const TAG_NUMBER_OF_IMAGES: u16 = 0xB001;
const TAG_MP_ENTRY: u16 = 0xB002;

// TIFF type constants.
const TYPE_UNDEFINED: u16 = 7;
const TYPE_LONG: u16 = 4;

/// Errors that can occur during MPF parsing.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MpfError {
    /// No APP2 segment with `MPF\0` identifier was found.
    #[error("MPF segment not found in JPEG data")]
    NotFound,

    /// The MPF payload is too short to contain a TIFF header.
    #[error("MPF data too short ({got} bytes, need at least 8)")]
    TooShort {
        /// Number of bytes available.
        got: usize,
    },

    /// The TIFF endianness magic (`"MM"` or `"II"`) is missing or invalid.
    #[error("invalid MPF endianness marker: {0:?}")]
    BadEndian([u8; 2]),

    /// The IFD offset points past the end of the MPF payload.
    #[error("invalid MPF IFD offset: {offset} > {payload_len}")]
    BadIfdOffset {
        /// Declared IFD offset.
        offset: usize,
        /// Payload length.
        payload_len: usize,
    },

    /// The MP Entry array offset points past the end of the MPF payload.
    #[error("MP Entry array overflows payload ({offset} + {needed} > {payload_len})")]
    BadEntryOffset {
        /// Declared array start offset.
        offset: usize,
        /// Required bytes past that offset.
        needed: usize,
        /// Payload length.
        payload_len: usize,
    },

    /// The MPF declares zero images (degenerate).
    #[error("MPF declares zero images")]
    NoImages,

    /// Structural parse failure with a human-readable detail.
    #[error("MPF parse error: {0}")]
    Other(String),
}

// ===========================================================================
// Parse
// ===========================================================================

/// Parse MPF entries from a full JPEG byte buffer.
///
/// Scans for an APP2 segment beginning with `MPF\0` using the public
/// [`super::marker::iter`] (length-aware, entropy-aware) and then
/// parses its TIFF-structured directory.
///
/// The returned entries are in the order declared by the MPF. Entry 0
/// is conventionally the primary image at absolute offset `0`; secondary
/// entries carry absolute offsets computed from `tiff_header_pos +
/// data_offset`.
///
/// # Examples
///
/// ```
/// use zenjpeg::container::mpf::{self, MpfError};
///
/// // A plain JPEG (no MPF segment) returns NotFound, not a panic.
/// # let jpeg: &[u8] = &[0xFF, 0xD8, 0xFF, 0xD9];
/// assert!(matches!(mpf::parse_mpf(jpeg), Err(MpfError::NotFound)));
///
/// // Build a synthetic MPF with one secondary and roundtrip it.
/// let header = mpf::create_mpf_header(100_000, 5_000, None);
/// let mut doc = vec![0xFF, 0xD8];
/// doc.extend_from_slice(&header);
/// doc.push(0xFF);
/// doc.push(0xD9);
/// let entries = mpf::parse_mpf(&doc).unwrap();
/// assert_eq!(entries.len(), 2);
/// assert_eq!(entries[0].size, 100_000);
/// assert_eq!(entries[1].size, 5_000);
/// ```
pub fn parse_mpf(data: &[u8]) -> Result<Vec<MpfEntry>, MpfError> {
    for span in iter(data) {
        if !matches!(span.kind, MarkerKind::App(2)) {
            continue;
        }
        if !span.payload.starts_with(MPF_IDENTIFIER) {
            continue;
        }
        // APP2 span layout: [FF E2][len:2][MPF\0][tiff...]
        // payload == everything after the length word, so payload[0..4] == MPF\0
        // and the TIFF data begins at payload[4..].
        let tiff_header_pos = span.offset + 2 /*FFEn*/ + 2 /*length*/ + MPF_IDENTIFIER.len();
        return parse_mpf_segment(&span.payload[MPF_IDENTIFIER.len()..], tiff_header_pos);
    }
    Err(MpfError::NotFound)
}

/// Parse an MPF TIFF-structured directory given the payload (bytes
/// AFTER the `MPF\0` identifier) and the absolute file offset of the
/// TIFF header.
///
/// `tiff_header_pos` is used to compute absolute secondary-image
/// offsets per CIPA DC-007 §5.2.3.2 (data offsets are relative to the
/// TIFF header).
pub fn parse_mpf_segment(
    mpf_data: &[u8],
    tiff_header_pos: usize,
) -> Result<Vec<MpfEntry>, MpfError> {
    if mpf_data.len() < 8 {
        return Err(MpfError::TooShort {
            got: mpf_data.len(),
        });
    }

    let big_endian = match &mpf_data[0..2] {
        b"MM" => true,
        b"II" => false,
        bytes => {
            let mut arr = [0u8; 2];
            arr.copy_from_slice(bytes);
            return Err(MpfError::BadEndian(arr));
        }
    };

    let r16 = |off: usize| -> Option<u16> {
        let b = mpf_data.get(off..off + 2)?;
        Some(if big_endian {
            u16::from_be_bytes([b[0], b[1]])
        } else {
            u16::from_le_bytes([b[0], b[1]])
        })
    };
    let r32 = |off: usize| -> Option<u32> {
        let b = mpf_data.get(off..off + 4)?;
        Some(if big_endian {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    };

    // IFD offset comes from an attacker-controlled u32. On 32-bit targets
    // `usize == u32`, so any addition below must use checked_add to avoid
    // wrap. Each check surfaces `BadIfdOffset` / `BadEntryOffset` rather
    // than treating an overflow as "in bounds".
    let ifd_offset = r32(4).ok_or_else(|| MpfError::Other("missing IFD pointer".into()))? as usize;
    let ifd_end = ifd_offset.checked_add(2).ok_or(MpfError::BadIfdOffset {
        offset: ifd_offset,
        payload_len: mpf_data.len(),
    })?;
    if ifd_end > mpf_data.len() {
        return Err(MpfError::BadIfdOffset {
            offset: ifd_offset,
            payload_len: mpf_data.len(),
        });
    }

    let num_entries =
        r16(ifd_offset).ok_or_else(|| MpfError::Other("missing IFD count".into()))? as usize;

    let mut mp_entry_offset = 0usize;
    let mut mp_entry_count = 0u32;
    // Cursor walk (not `ifd_end + i*12` indexing) so the out-of-line
    // version quirk below can resync the stream for subsequent entries.
    let mut cursor = ifd_end;
    for _ in 0..num_entries {
        // `cursor + 12` must be <= mpf_data.len(); checked so an
        // attacker-controlled chain of advances can't wrap usize on 32-bit.
        let Some(off_end) = cursor.checked_add(12) else {
            break;
        };
        if off_end > mpf_data.len() {
            break;
        }
        let off = cursor;
        cursor = off_end;
        // Bounds proved by the `cursor + 12 > len` guard above: `r16(off)`
        // and `r32(off + 8)` both read within [off, off+12). Propagate
        // defensively via `?` so a future refactor of the guard can
        // surface as a parse error rather than a panic.
        let tag = r16(off).ok_or(MpfError::BadIfdOffset {
            offset: off,
            payload_len: mpf_data.len(),
        })?;
        let value_offset = r32(off + 8).ok_or(MpfError::BadIfdOffset {
            offset: off + 8,
            payload_len: mpf_data.len(),
        })?;
        match tag {
            TAG_NUMBER_OF_IMAGES => mp_entry_count = value_offset,
            TAG_MP_ENTRY => mp_entry_offset = value_offset as usize,
            // In-the-wild quirk (#148): some writers emit the MPFVersion
            // UNDEFINED×4 value OUT OF LINE — the entry's value field is
            // zeroed and the four ASCII version bytes (`"0100"`) are
            // spliced directly after the 12-byte entry. That shifts every
            // following entry by 4, so a strict stride-12 walk reads
            // garbage tags, never sees B001/B002, and reports "MPF
            // declares zero images" for a file whose index is otherwise
            // fully self-consistent. Detect exactly that shape and consume
            // the stray bytes; conformant files (version inline in the
            // value field, non-zero) take the `_` arm untouched.
            TAG_VERSION
                if value_offset == 0
                    && r16(off + 2) == Some(TYPE_UNDEFINED)
                    && r32(off + 4) == Some(4)
                    && mpf_data.get(cursor..cursor + 4) == Some(MPF_VERSION.as_slice()) =>
            {
                cursor += 4;
            }
            _ => {}
        }
    }

    if mp_entry_count == 0 {
        // Matches upstream `ultrahdr-core::metadata::mpf` behavior: a zero
        // declared image count surfaces as `NoImages`.
        return Err(MpfError::NoImages);
    }

    // Resolve where the MP entries actually live. Conformant files put a
    // TIFF-relative offset in B002; the same writer family as the version
    // quirk above records it IFD-relative instead (observed: declared 0x2E
    // for entries that physically sit at TIFF+0x36 — landing the declared
    // offset *inside the IFD*). Sanity-check the declared offset by the
    // first entry's image size (a real entry is never zero-sized); when it
    // fails, fall back to the structural position — MP entries directly
    // follow the IFD's next-IFD pointer (`cursor + 4`).
    let entry_looks_sane = |off: usize| -> bool {
        off != 0
            && r32(off + 4).is_some_and(|size| size > 0)
            && (mp_entry_count < 2 || r32(off + 16 + 4).is_some_and(|size| size > 0))
    };
    if !entry_looks_sane(mp_entry_offset) {
        let post_ifd = cursor.saturating_add(4);
        if entry_looks_sane(post_ifd) {
            mp_entry_offset = post_ifd;
        } else {
            return Err(MpfError::NoImages);
        }
    }

    // Cap entry count against actual payload size to prevent OOM on
    // crafted inputs. Each MP entry is 16 bytes; also cap at
    // MAX_MPF_ENTRIES (1000) as defense-in-depth. Real files hold
    // ≤ ~5 entries; truncation at 1000 is documented and surfaced via
    // a future overflow flag if we add one — for now it's silent, same
    // as the implicit per-iteration break the upstream version does.
    let max_possible = if mp_entry_offset < mpf_data.len() {
        (mpf_data.len() - mp_entry_offset) / 16
    } else {
        0
    };
    let mp_entry_count = (mp_entry_count as usize)
        .min(max_possible)
        .min(MAX_MPF_ENTRIES);

    // Capped above by `max_possible = (len - offset) / 16` and
    // `MAX_MPF_ENTRIES`, so this multiply + add is bounded by `len`.
    // checked_* anyway to stay wrap-safe on 32-bit.
    let mp_entry_bytes = mp_entry_count
        .checked_mul(16)
        .ok_or(MpfError::BadEntryOffset {
            offset: mp_entry_offset,
            needed: usize::MAX,
            payload_len: mpf_data.len(),
        })?;
    let mp_entry_end =
        mp_entry_offset
            .checked_add(mp_entry_bytes)
            .ok_or(MpfError::BadEntryOffset {
                offset: mp_entry_offset,
                needed: mp_entry_bytes,
                payload_len: mpf_data.len(),
            })?;
    if mp_entry_end > mpf_data.len() {
        return Err(MpfError::BadEntryOffset {
            offset: mp_entry_offset,
            needed: mp_entry_bytes,
            payload_len: mpf_data.len(),
        });
    }

    let mut entries = Vec::with_capacity(mp_entry_count);
    for i in 0..mp_entry_count {
        // checked because on 32-bit, attacker-controlled mp_entry_offset
        // (from a u32 in the file) can approach usize::MAX.
        let pos = i
            .checked_mul(16)
            .and_then(|delta| mp_entry_offset.checked_add(delta))
            .ok_or(MpfError::BadEntryOffset {
                offset: mp_entry_offset,
                needed: mp_entry_bytes,
                payload_len: mpf_data.len(),
            })?;
        // Bounds proved by the `mp_entry_offset + mp_entry_count * 16 > len`
        // guard above: each read stays within [pos, pos+16). Propagate
        // defensively via `?` to surface refactor breakage as a parse
        // error rather than a panic.
        let attribute = r32(pos).ok_or(MpfError::BadEntryOffset {
            offset: pos,
            needed: 4,
            payload_len: mpf_data.len(),
        })?;
        let size = r32(pos + 4).ok_or(MpfError::BadEntryOffset {
            offset: pos + 4,
            needed: 4,
            payload_len: mpf_data.len(),
        })? as usize;
        let data_offset = r32(pos + 8).ok_or(MpfError::BadEntryOffset {
            offset: pos + 8,
            needed: 4,
            payload_len: mpf_data.len(),
        })? as usize;
        let abs_offset = if i == 0 {
            0
        } else {
            tiff_header_pos + data_offset
        };
        entries.push(MpfEntry {
            image_type: MpImageType::from_type_code(attribute),
            offset: abs_offset,
            size,
        });
    }

    if entries.is_empty() {
        return Err(MpfError::NoImages);
    }
    Ok(entries)
}

// ===========================================================================
// Emit
// ===========================================================================

/// Build an MPF APP2 segment (including the `FF E2` marker + length
/// word + `MPF\0` identifier + TIFF directory) for a primary image plus
/// N typed secondary images.
///
/// - `primary_length`: total length of the primary JPEG in bytes.
/// - `secondary_images`: `(image_type, byte_length)` for each secondary.
/// - `mpf_insert_offset`: absolute file offset at which this APP2
///   segment will be inserted. Used to compute secondary data offsets
///   relative to the TIFF header (per CIPA DC-007). Pass `None` for the
///   legacy "MPF at file start" behavior.
#[must_use]
pub fn create_mpf_header_typed(
    primary_length: usize,
    secondary_images: &[(MpImageType, usize)],
    mpf_insert_offset: Option<usize>,
) -> Vec<u8> {
    let num_images = 1 + secondary_images.len();
    let mut mpf = Vec::with_capacity(128);

    let tiff_header_pos = mpf_insert_offset.unwrap_or(0) + 4 + MPF_IDENTIFIER.len();

    // TIFF header (big-endian).
    mpf.extend_from_slice(b"MM");
    mpf.push(0x00);
    mpf.push(0x2A);
    mpf.extend_from_slice(&8u32.to_be_bytes()); // IFD at offset 8.

    // IFD: 3 entries.
    mpf.extend_from_slice(&3u16.to_be_bytes());

    // Entry 1: Version.
    let version_value = u32::from_be_bytes(*MPF_VERSION);
    write_ifd_entry(&mut mpf, TAG_VERSION, TYPE_UNDEFINED, 4, version_value);

    // Entry 2: Number of images.
    write_ifd_entry(
        &mut mpf,
        TAG_NUMBER_OF_IMAGES,
        TYPE_LONG,
        1,
        num_images as u32,
    );

    // Entry 3: MP Entry array.
    let mp_entry_size = (num_images * 16) as u32;
    let mp_entry_offset = mpf.len() as u32 + 12 + 4; // this entry + next IFD ptr.
    write_ifd_entry(
        &mut mpf,
        TAG_MP_ENTRY,
        TYPE_UNDEFINED,
        mp_entry_size,
        mp_entry_offset,
    );

    // Next IFD offset (0 = none).
    mpf.extend_from_slice(&0u32.to_be_bytes());

    // MP Entry: primary.
    write_mp_entry(
        &mut mpf,
        MpImageType::BaselinePrimary.type_code(),
        primary_length as u32,
        0,
    );

    // MP Entry: secondaries (offsets accumulate after primary).
    let mut offset_from_tiff = primary_length.saturating_sub(tiff_header_pos) as u32;
    for (image_type, length) in secondary_images {
        write_mp_entry(
            &mut mpf,
            image_type.type_code(),
            *length as u32,
            offset_from_tiff,
        );
        offset_from_tiff += *length as u32;
    }

    super::marker::create_app_segment(2, MPF_IDENTIFIER, &mpf)
}

/// Build an MPF APP2 segment for primary + a single gain map secondary.
///
/// Convenience wrapper over [`create_mpf_header_typed`] for the Ultra HDR
/// case. The secondary uses [`MpImageType::Undefined`] per current Ultra
/// HDR practice.
#[must_use]
pub fn create_mpf_header(
    primary_length: usize,
    gainmap_length: usize,
    mpf_insert_offset: Option<usize>,
) -> Vec<u8> {
    create_mpf_header_typed(
        primary_length,
        &[(MpImageType::Undefined, gainmap_length)],
        mpf_insert_offset,
    )
}

fn write_ifd_entry(buf: &mut Vec<u8>, tag: u16, type_id: u16, count: u32, value_or_offset: u32) {
    buf.extend_from_slice(&tag.to_be_bytes());
    buf.extend_from_slice(&type_id.to_be_bytes());
    buf.extend_from_slice(&count.to_be_bytes());
    buf.extend_from_slice(&value_or_offset.to_be_bytes());
}

fn write_mp_entry(buf: &mut Vec<u8>, type_code: u32, size: u32, offset: u32) {
    buf.extend_from_slice(&type_code.to_be_bytes());
    buf.extend_from_slice(&size.to_be_bytes());
    buf.extend_from_slice(&offset.to_be_bytes());
    // Two unused u16 fields at the end.
    buf.extend_from_slice(&0u16.to_be_bytes());
    buf.extend_from_slice(&0u16.to_be_bytes());
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Wrap an MPF APP2 payload into a minimal JPEG:
    /// SOI + APP2(MPF + payload) + (minimal SOF0) + (minimal SOS) + EOI.
    fn jpeg_with_mpf_app2(mpf_payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0xFF, 0xD8]); // SOI
        // APP2 length = 2 (length word) + 4 (MPF\0) + mpf_payload.len().
        let app2_len = 2 + 4 + mpf_payload.len();
        v.push(0xFF);
        v.push(0xE2);
        v.extend_from_slice(&(app2_len as u16).to_be_bytes());
        v.extend_from_slice(MPF_IDENTIFIER);
        v.extend_from_slice(mpf_payload);
        // Minimal SOF0, SOS, EOI so the iterator has something structured.
        v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11]);
        v.extend_from_slice(&[0x08, 0x00, 0x08, 0x00, 0x08, 0x03]);
        v.extend_from_slice(&[0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        v.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);
        v.extend_from_slice(&[0xAA, 0xBB]);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    /// Build an MPF payload (after `MPF\0`) with N images of given
    /// (type_code, size, offset_from_tiff) values.
    fn build_mpf_payload(entries: &[(u32, u32, u32)]) -> Vec<u8> {
        let num_images = entries.len() as u32;
        let mut v = Vec::new();
        v.extend_from_slice(b"MM");
        v.push(0x00);
        v.push(0x2A);
        v.extend_from_slice(&8u32.to_be_bytes()); // IFD at offset 8.
        v.extend_from_slice(&3u16.to_be_bytes()); // 3 IFD entries.

        // Entry: Version.
        v.extend_from_slice(&TAG_VERSION.to_be_bytes());
        v.extend_from_slice(&TYPE_UNDEFINED.to_be_bytes());
        v.extend_from_slice(&4u32.to_be_bytes());
        v.extend_from_slice(MPF_VERSION);

        // Entry: Number of images.
        v.extend_from_slice(&TAG_NUMBER_OF_IMAGES.to_be_bytes());
        v.extend_from_slice(&TYPE_LONG.to_be_bytes());
        v.extend_from_slice(&1u32.to_be_bytes());
        v.extend_from_slice(&num_images.to_be_bytes());

        // Entry: MP Entry array. It lives right after the IFD + next-IFD
        // offset, which is 8 (TIFF header) + 2 (count) + 3*12 (entries)
        // + 4 (next IFD ptr) = 50.
        let mp_entry_offset = 50u32;
        v.extend_from_slice(&TAG_MP_ENTRY.to_be_bytes());
        v.extend_from_slice(&TYPE_UNDEFINED.to_be_bytes());
        v.extend_from_slice(&(num_images * 16).to_be_bytes());
        v.extend_from_slice(&mp_entry_offset.to_be_bytes());

        // Next IFD offset = 0.
        v.extend_from_slice(&0u32.to_be_bytes());

        // MP Entries.
        for (type_code, size, offset) in entries {
            v.extend_from_slice(&type_code.to_be_bytes());
            v.extend_from_slice(&size.to_be_bytes());
            v.extend_from_slice(&offset.to_be_bytes());
            v.extend_from_slice(&0u16.to_be_bytes());
            v.extend_from_slice(&0u16.to_be_bytes());
        }

        v
    }

    #[test]
    fn parse_mpf_finds_segment_in_jpeg() {
        let payload = build_mpf_payload(&[
            (MpImageType::BaselinePrimary.type_code(), 100_000, 0),
            (MpImageType::Undefined.type_code(), 5_000, 100_000),
        ]);
        let jpeg = jpeg_with_mpf_app2(&payload);
        let entries = parse_mpf(&jpeg).expect("parse");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].image_type, MpImageType::BaselinePrimary);
        assert_eq!(entries[0].size, 100_000);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[1].image_type, MpImageType::Undefined);
        assert_eq!(entries[1].size, 5_000);
        // Secondary absolute offset = tiff_header_pos + data_offset.
        // tiff_header_pos is the file offset of the TIFF `MM` header,
        // which in this test fixture is 2 (SOI) + 4 (APP2 marker +
        // length) + 4 (MPF\0) = 10.
        assert_eq!(entries[1].offset, 10 + 100_000);
    }

    #[test]
    fn parse_mpf_not_found_on_jpeg_without_app2() {
        // Minimal JPEG without an APP2 MPF segment.
        let jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
        let err = parse_mpf(&jpeg).unwrap_err();
        assert!(matches!(err, MpfError::NotFound));
    }

    #[test]
    fn parse_mpf_not_found_when_app2_has_other_identifier() {
        // APP2 with ICC_PROFILE identifier.
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8]);
        jpeg.push(0xFF);
        jpeg.push(0xE2);
        let app2_payload = b"ICC_PROFILE\0\x00\x01";
        let len = 2 + app2_payload.len();
        jpeg.extend_from_slice(&(len as u16).to_be_bytes());
        jpeg.extend_from_slice(app2_payload);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        let err = parse_mpf(&jpeg).unwrap_err();
        assert!(matches!(err, MpfError::NotFound));
    }

    #[test]
    fn parse_mpf_segment_be() {
        let payload = build_mpf_payload(&[
            (MpImageType::BaselinePrimary.type_code(), 1_000, 0),
            (MpImageType::Undefined.type_code(), 200, 1_000),
        ]);
        let entries = parse_mpf_segment(&payload, 100).expect("parse");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].offset, 0);
        // Secondary offset = tiff_header_pos + data_offset = 100 + 1000.
        assert_eq!(entries[1].offset, 1_100);
    }

    #[test]
    fn parse_mpf_segment_le() {
        let mut v = Vec::new();
        v.extend_from_slice(b"II");
        v.push(0x2A);
        v.push(0x00);
        v.extend_from_slice(&8u32.to_le_bytes());
        v.extend_from_slice(&3u16.to_le_bytes());

        v.extend_from_slice(&TAG_VERSION.to_le_bytes());
        v.extend_from_slice(&TYPE_UNDEFINED.to_le_bytes());
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(MPF_VERSION);

        v.extend_from_slice(&TAG_NUMBER_OF_IMAGES.to_le_bytes());
        v.extend_from_slice(&TYPE_LONG.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());

        v.extend_from_slice(&TAG_MP_ENTRY.to_le_bytes());
        v.extend_from_slice(&TYPE_UNDEFINED.to_le_bytes());
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&50u32.to_le_bytes());

        v.extend_from_slice(&0u32.to_le_bytes()); // next IFD.

        // Primary entry.
        v.extend_from_slice(&MpImageType::BaselinePrimary.type_code().to_le_bytes());
        v.extend_from_slice(&100u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());

        let entries = parse_mpf_segment(&v, 0).expect("parse LE");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].image_type, MpImageType::BaselinePrimary);
        assert_eq!(entries[0].size, 100);
    }

    #[test]
    fn parse_mpf_segment_rejects_bad_endian() {
        let data = [b'X', b'Y', 0, 0, 0, 0, 0, 0];
        assert!(matches!(
            parse_mpf_segment(&data, 0),
            Err(MpfError::BadEndian([b'X', b'Y']))
        ));
    }

    #[test]
    fn parse_mpf_segment_rejects_too_short() {
        assert!(matches!(
            parse_mpf_segment(&[], 0),
            Err(MpfError::TooShort { .. })
        ));
        assert!(matches!(
            parse_mpf_segment(&[0; 7], 0),
            Err(MpfError::TooShort { .. })
        ));
    }

    #[test]
    fn parse_mpf_segment_rejects_bad_ifd_offset() {
        let mut v = Vec::new();
        v.extend_from_slice(b"MM");
        v.push(0x00);
        v.push(0x2A);
        // IFD offset points past end of payload.
        v.extend_from_slice(&0xFF_FF_FF_FFu32.to_be_bytes());
        let err = parse_mpf_segment(&v, 0).unwrap_err();
        assert!(matches!(err, MpfError::BadIfdOffset { .. }));
    }

    #[test]
    fn parse_mpf_segment_caps_entry_count() {
        // Craft: declare 1_000_000 images; payload can only fit a few.
        // Parser must cap without OOM and produce at most `max_possible`.
        let mut v = build_mpf_payload(&[
            (MpImageType::BaselinePrimary.type_code(), 10, 0),
            (MpImageType::Undefined.type_code(), 20, 10),
        ]);
        // Overwrite the "number of images" value (offset 8 + 2 + 12 + 8
        // = 30 within this layout, hit it by scanning for the TAG_*
        // bytes).
        let needle = [0xB0, 0x01];
        if let Some(pos) = v.windows(2).position(|w| w == needle) {
            // Tag at pos; value at pos + 8 (skip tag, type, count).
            let value_start = pos + 8;
            v[value_start..value_start + 4].copy_from_slice(&1_000_000u32.to_be_bytes());
        }
        // Must still parse cleanly (count capped to what actually fits).
        let entries = parse_mpf_segment(&v, 0).expect("parse with capped count");
        assert!(
            entries.len() <= 1000,
            "entry count exceeded hard cap: {}",
            entries.len()
        );
    }

    #[test]
    fn create_mpf_header_roundtrips() {
        let built = create_mpf_header(100_000, 5_000, None);
        // Strip APP2 marker + length word + MPF\0 to isolate TIFF data.
        assert_eq!(built[0], 0xFF);
        assert_eq!(built[1], 0xE2);
        let tiff_start = 4 + MPF_IDENTIFIER.len();
        let entries = parse_mpf_segment(&built[tiff_start..], tiff_start).expect("parse");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].image_type, MpImageType::BaselinePrimary);
        assert_eq!(entries[0].size, 100_000);
        assert_eq!(entries[1].image_type, MpImageType::Undefined);
        assert_eq!(entries[1].size, 5_000);
    }

    #[test]
    fn create_mpf_header_typed_roundtrips_multi() {
        let built = create_mpf_header_typed(
            100_000,
            &[
                (MpImageType::Disparity, 8_000),
                (MpImageType::Undefined, 4_000),
            ],
            None,
        );
        let tiff_start = 4 + MPF_IDENTIFIER.len();
        let entries = parse_mpf_segment(&built[tiff_start..], tiff_start).expect("parse");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].image_type, MpImageType::BaselinePrimary);
        assert_eq!(entries[1].image_type, MpImageType::Disparity);
        assert_eq!(entries[1].size, 8_000);
        assert_eq!(entries[2].image_type, MpImageType::Undefined);
        assert_eq!(entries[2].size, 4_000);
    }

    #[test]
    fn create_mpf_header_respects_insert_offset() {
        let built = create_mpf_header(100_000, 5_000, Some(20));
        // tiff_header_pos from the ENCODER's perspective = 20 + 4 + 4 = 28.
        let tiff_start = 4 + MPF_IDENTIFIER.len();
        let entries = parse_mpf_segment(&built[tiff_start..], 28).expect("parse");
        // Secondary should sit at primary_length - tiff_header_pos
        // (encoder's compute) + tiff_header_pos (parser's compute) =
        // primary_length.
        assert_eq!(entries[1].offset, 100_000);
    }

    /// Parser must not panic on arbitrary garbage — and must surface
    /// an `Err` for every garbage case (never return `Ok` with bogus
    /// data).
    #[test]
    fn parse_mpf_segment_no_panic_on_garbage() {
        assert!(parse_mpf_segment(&[0xAB; 4], 0).is_err());
        let mut v = vec![b'M', b'M', 0, 0x2A];
        v.extend_from_slice(&0x12345678u32.to_be_bytes());
        v.extend_from_slice(&[0; 30]);
        assert!(parse_mpf_segment(&v, 0).is_err());
        let v: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(17)).collect();
        assert!(parse_mpf_segment(&v, 0).is_err());
    }

    /// Regression for the "mp_entry_offset == 0" branch. An MPF IFD
    /// that declares a positive image count but forgot to populate the
    /// MP Entry offset must yield `NoImages`, matching upstream
    /// `ultrahdr-core` semantics. If the `|| mp_entry_offset == 0`
    /// half of the check is deleted, this test catches it.
    #[test]
    fn parse_mpf_segment_rejects_zero_entry_offset() {
        // Build a valid-looking TIFF header + IFD but set MP_ENTRY
        // offset to 0.
        let mut v = Vec::new();
        v.extend_from_slice(b"MM");
        v.push(0x00);
        v.push(0x2A);
        v.extend_from_slice(&8u32.to_be_bytes()); // IFD at offset 8
        v.extend_from_slice(&3u16.to_be_bytes()); // 3 IFD entries

        // Version tag.
        v.extend_from_slice(&TAG_VERSION.to_be_bytes());
        v.extend_from_slice(&TYPE_UNDEFINED.to_be_bytes());
        v.extend_from_slice(&4u32.to_be_bytes());
        v.extend_from_slice(MPF_VERSION);

        // Number of images = 1 (positive!).
        v.extend_from_slice(&TAG_NUMBER_OF_IMAGES.to_be_bytes());
        v.extend_from_slice(&TYPE_LONG.to_be_bytes());
        v.extend_from_slice(&1u32.to_be_bytes());
        v.extend_from_slice(&1u32.to_be_bytes());

        // MP Entry tag with value_offset = 0 (the bug trigger).
        v.extend_from_slice(&TAG_MP_ENTRY.to_be_bytes());
        v.extend_from_slice(&TYPE_UNDEFINED.to_be_bytes());
        v.extend_from_slice(&16u32.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes()); // <-- offset = 0

        v.extend_from_slice(&0u32.to_be_bytes()); // next IFD = 0
        // Payload is intentionally shorter than any sane parser would
        // use; the point is that offset==0 should reject BEFORE any
        // read.

        let err = parse_mpf_segment(&v, 0).unwrap_err();
        assert!(
            matches!(err, MpfError::NoImages),
            "mp_entry_offset == 0 with declared count > 0 must yield NoImages, got {err:?}"
        );
    }

    /// Verify `create_mpf_header` rejects oversize primary / gainmap
    /// lengths rather than silently truncating to u32 and producing
    /// corrupt output.
    ///
    /// Current implementation DOES truncate via `as u32`. This test
    /// documents the behavior: on a 5 GiB primary the emitted
    /// `primary_size` field will be 5 GiB mod 2^32 ≈ 705 MB. Until
    /// the emitter gains a fallible variant, caller responsibility
    /// is to keep lengths below `u32::MAX`.
    #[test]
    fn create_mpf_header_truncates_oversize_lengths() {
        let primary_5_gib = 5u64 * 1024 * 1024 * 1024;
        let gainmap_small = 1024usize;
        let bytes = create_mpf_header(primary_5_gib as usize, gainmap_small, None);
        let tiff_start = 4 + MPF_IDENTIFIER.len();
        let entries = parse_mpf_segment(&bytes[tiff_start..], tiff_start).expect("parse");
        // Documented behavior: primary size wraps via `as u32`.
        let expected_wrapped = (primary_5_gib & 0xFFFF_FFFF) as usize;
        assert_eq!(
            entries[0].size, expected_wrapped,
            "emitter truncates oversize primary via `as u32` — this test pins that \
             behavior so a future fallible-emitter migration is a conscious change"
        );
    }

    // -----------------------------------------------------------------------
    // Property tests: parse(serialize(x)) == canonicalize(x)
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    fn arb_mpf_image_type() -> impl Strategy<Value = MpImageType> {
        prop_oneof![
            Just(MpImageType::Undefined),
            Just(MpImageType::LargeThumbnailVga),
            Just(MpImageType::LargeThumbnailFullHd),
            Just(MpImageType::Panorama),
            Just(MpImageType::Disparity),
            Just(MpImageType::MultiAngle),
            // BaselinePrimary is implicitly prepended by create_mpf_header_typed,
            // so we don't pass it as a secondary.
            (0u32..=0x00FF_FFFFu32).prop_map(MpImageType::Other),
        ]
    }

    proptest! {
        /// Any list of typed secondary images, with any primary size and
        /// secondary sizes in the u32 range, must survive a parse →
        /// serialize → parse roundtrip byte-for-byte at the entry level.
        #[test]
        fn mpf_roundtrip_typed_secondaries(
            primary_len in 1usize..10_000_000,
            secondaries in proptest::collection::vec(
                (arb_mpf_image_type(), 1usize..10_000_000),
                0..5,
            ),
        ) {
            let built = create_mpf_header_typed(primary_len, &secondaries, None);
            let tiff_start = 4 + MPF_IDENTIFIER.len();
            let entries = parse_mpf_segment(&built[tiff_start..], tiff_start)
                .expect("parse should succeed on self-generated data");

            // 1 primary + N secondaries.
            prop_assert_eq!(entries.len(), 1 + secondaries.len());
            prop_assert_eq!(entries[0].image_type, MpImageType::BaselinePrimary);
            prop_assert_eq!(entries[0].size, primary_len);

            for (i, (typ, size)) in secondaries.iter().enumerate() {
                prop_assert_eq!(entries[i + 1].image_type, *typ);
                prop_assert_eq!(entries[i + 1].size, *size);
            }
        }

        /// Untyped create_mpf_header (single-secondary convenience) also
        /// roundtrips — same contract, narrower input.
        #[test]
        fn mpf_roundtrip_untyped_single(
            primary_len in 1usize..10_000_000,
            secondary_len in 1usize..10_000_000,
            insert_offset in proptest::option::of(0usize..1024),
        ) {
            let built = create_mpf_header(primary_len, secondary_len, insert_offset);
            let tiff_start = 4 + MPF_IDENTIFIER.len();
            let reader_tiff_pos = insert_offset.unwrap_or(0) + tiff_start;
            let entries = parse_mpf_segment(&built[tiff_start..], reader_tiff_pos)
                .expect("parse should succeed");
            prop_assert_eq!(entries.len(), 2);
            prop_assert_eq!(entries[0].size, primary_len);
            prop_assert_eq!(entries[1].size, secondary_len);
        }
    }

    /// Issue #148 regression: a real-world MP Index (the 7.6 KB
    /// `ultrahdr_sample.jpg` fixture committed in zencodecs and
    /// ultrahdr-rs) has two quirks from the same writer family:
    /// the MPFVersion UNDEFINED×4 value is written OUT OF LINE (value
    /// field zeroed, ASCII `"0100"` spliced after the 12-byte entry,
    /// shifting later entries by 4), and the B002 MPEntry offset is
    /// IFD-relative (0x2E) instead of TIFF-relative (entries physically
    /// at TIFF+0x36). The strict walk used to report "MPF declares zero
    /// images" for an index whose sizes sum exactly to the file length.
    #[test]
    fn out_of_line_version_and_ifd_relative_entries_resync() {
        // Mirror the fixture's TIFF payload structure byte-for-byte:
        // primary size 0x1A9D at offset 0, gain map size 0x321 at 0x1A9D.
        let mut p = Vec::new();
        p.extend_from_slice(b"MM");
        p.extend_from_slice(&[0x00, 0x2A]);
        p.extend_from_slice(&8u32.to_be_bytes()); // IFD at TIFF+8
        p.extend_from_slice(&3u16.to_be_bytes()); // 3 IFD entries
        // B000 Version, UNDEFINED×4 — value field ZEROED…
        p.extend_from_slice(&TAG_VERSION.to_be_bytes());
        p.extend_from_slice(&TYPE_UNDEFINED.to_be_bytes());
        p.extend_from_slice(&4u32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        // …with the version bytes spliced out-of-line after the entry.
        p.extend_from_slice(MPF_VERSION);
        // B001 NumberOfImages, LONG×1 = 2.
        p.extend_from_slice(&TAG_NUMBER_OF_IMAGES.to_be_bytes());
        p.extend_from_slice(&TYPE_LONG.to_be_bytes());
        p.extend_from_slice(&1u32.to_be_bytes());
        p.extend_from_slice(&2u32.to_be_bytes());
        // B002 MPEntry, UNDEFINED×32 — declared offset 0x2E is
        // IFD-relative (lands inside the IFD when read TIFF-relative).
        p.extend_from_slice(&TAG_MP_ENTRY.to_be_bytes());
        p.extend_from_slice(&TYPE_UNDEFINED.to_be_bytes());
        p.extend_from_slice(&32u32.to_be_bytes());
        p.extend_from_slice(&0x2Eu32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes()); // next-IFD pointer
        assert_eq!(p.len(), 0x36, "MP entries physically start at TIFF+0x36");
        // MP entry 1: primary, attr 0x00030000, size 0x1A9D, offset 0.
        p.extend_from_slice(&0x0003_0000u32.to_be_bytes());
        p.extend_from_slice(&0x1A9Du32.to_be_bytes());
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);
        // MP entry 2: gain map, attr 0, size 0x321, offset 0x1A9D.
        p.extend_from_slice(&0u32.to_be_bytes());
        p.extend_from_slice(&0x321u32.to_be_bytes());
        p.extend_from_slice(&0x1A9Du32.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);

        let jpeg = jpeg_with_mpf_app2(&p);
        let entries = parse_mpf(&jpeg).expect("resync must recover the index");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].size, 0x1A9D);
        assert_eq!(entries[1].size, 0x321);
        // Secondary offsets are absolute: tiff_header_pos (SOI 2 + APP2
        // marker+len 4 + "MPF\0" 4 = 10) + the entry's data offset.
        assert_eq!(entries[1].offset, 10 + 0x1A9D);
    }
}
