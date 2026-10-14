//! Payloads that span several segments, reassembled with validation.
//!
//! Each function reads the header items it is given, borrows when one
//! segment holds the whole payload, and otherwise allocates the returned
//! buffer, fallibly, once every check has passed. Either way the payload's
//! size must be within
//! [`crate::container::metadata::MetadataLimits::max_reassembled_bytes`].
//! Extended XMP split over several chunks also keeps a list of them (24 bytes
//! per chunk on 64-bit targets, reserved once, fallibly).

use alloc::borrow::Cow;
use alloc::vec::Vec;

use super::identify::PHOTOSHOP_SIGNATURE;
use super::{Item, ItemKind, MetadataError, MetadataLimits, Unparsed};
use crate::foundation::md5::Md5;

/// ICC.1:2010 Annex B.4 reassembly with the validation of libjpeg-turbo's
/// `jpeg_read_icc_profile` (iccread.c): every chunk agrees on the count,
/// sequence numbers lie in `1..=count`, none repeats and none is missing.
/// Chunk order in the file does not matter.
pub(super) fn icc_profile<'a>(
    items: &[Item<'a>],
    limits: &MetadataLimits,
) -> Result<Option<Cow<'a, [u8]>>, MetadataError> {
    crate::container::chunks::icc_profile(
        items.iter().filter_map(|item| match item.kind {
            ItemKind::IccProfile(chunk) => Some((chunk.sequence, chunk.count, chunk.data)),
            _ => None,
        }),
        limits.max_reassembled_bytes,
    )
    .map_err(|error| {
        use crate::container::chunks::Error;
        match error {
            Error::IccZeroChunkCount => MetadataError::IccZeroChunkCount,
            Error::IccChunkCountMismatch { expected, found } => {
                MetadataError::IccChunkCountMismatch { expected, found }
            }
            Error::IccSequenceOutOfRange { sequence, count } => {
                MetadataError::IccSequenceOutOfRange { sequence, count }
            }
            Error::IccDuplicateChunk { sequence } => MetadataError::IccDuplicateChunk { sequence },
            Error::IccMissingChunk { sequence } => MetadataError::IccMissingChunk { sequence },
            Error::IccEmptyProfile => MetadataError::IccEmptyProfile,
            Error::TooLarge { size, limit } => MetadataError::TooLarge { size, limit },
            Error::OutOfMemory => MetadataError::OutOfMemory,
        }
    })
}

/// XMP Specification Part 3, 1.1.3.1, over the header items: the standard
/// packet names the extended packet with `xmpNote:HasExtendedXMP="<GUID>"`
/// (the last such value, as in ExifTool); the chunks carrying that GUID
/// (compared without regard to case) are placed at their offsets and must
/// cover the full length exactly; the GUID is the MD5 digest of the result.
/// The digest is computed over the chunks in place, so a packet that fails
/// any check is never copied, and one chunk holding the whole packet is
/// borrowed.
///
/// Stricter than ExifTool 12.40, which only warns about chunks that
/// disagree on the full length, and reads a packet with a chunk repeated at
/// the same offset; both are errors here.
pub(super) fn extended_xmp<'a>(
    header: &[Item<'a>],
    limits: &MetadataLimits,
) -> Result<Option<Cow<'a, [u8]>>, MetadataError> {
    let guid = header.iter().rev().find_map(|item| match item.kind {
        ItemKind::Xmp(packet) => has_extended_xmp(packet),
        _ => None,
    });
    let Some(guid) = guid else {
        return Ok(None);
    };
    let matching = || {
        header.iter().filter_map(move |item| match &item.kind {
            ItemKind::ExtendedXmp(chunk) if chunk.guid.eq_ignore_ascii_case(guid) => Some(*chunk),
            _ => None,
        })
    };
    let mut full_length = None;
    let mut count = 0usize;
    for chunk in matching() {
        let expected = *full_length.get_or_insert(chunk.full_length);
        if chunk.full_length != expected {
            return Err(MetadataError::ExtendedXmpLengthMismatch {
                expected,
                found: chunk.full_length,
            });
        }
        count += 1;
    }
    let Some(full_length) = full_length else {
        return Err(MetadataError::ExtendedXmpMissing);
    };
    let full = usize::try_from(full_length).map_err(|_| MetadataError::TooLarge {
        size: usize::MAX,
        limit: limits.max_reassembled_bytes,
    })?;
    check_size(full, limits)?;
    if count == 1 {
        // One chunk: verify it in place and borrow it.
        if let Some(chunk) = matching().next() {
            verify_extended_xmp(&[(chunk.offset, chunk.data)], full_length, guid)?;
            return Ok(Some(Cow::Borrowed(chunk.data)));
        }
    }
    let mut chunks: Vec<(u32, &[u8])> = Vec::new();
    chunks
        .try_reserve_exact(count)
        .map_err(|_| MetadataError::OutOfMemory)?;
    chunks.extend(matching().map(|c| (c.offset, c.data)));
    // Writers emit chunks in offset order; sort only if one did not.
    if !chunks.is_sorted_by_key(|c| c.0) {
        chunks.sort_unstable_by_key(|c| c.0);
    }
    verify_extended_xmp(&chunks, full_length, guid)?;
    let mut packet = try_vec(full)?;
    for (_, data) in &chunks {
        packet.extend_from_slice(data);
    }
    Ok(Some(Cow::Owned(packet)))
}

/// Offset-sorted chunks must tile `0..full_length` exactly, and their MD5
/// must equal `guid`.
fn verify_extended_xmp(
    chunks: &[(u32, &[u8])],
    full_length: u32,
    guid: &[u8],
) -> Result<(), MetadataError> {
    let mut covered = 0u32;
    let mut digest = Md5::new();
    for &(offset, data) in chunks {
        let end = u32::try_from(data.len())
            .ok()
            .and_then(|len| offset.checked_add(len));
        if offset != covered {
            return Err(MetadataError::ExtendedXmpDiscontinuous {
                expected: covered,
                offset,
            });
        }
        match end {
            Some(end) if end <= full_length => covered = end,
            _ => {
                return Err(MetadataError::ExtendedXmpChunkPastEnd {
                    offset,
                    length: data.len(),
                    full_length,
                });
            }
        }
        digest.update(data);
    }
    if covered != full_length {
        return Err(MetadataError::ExtendedXmpIncomplete {
            got: covered,
            full_length,
        });
    }
    if !guid_matches(guid, &digest.finish()) {
        return Err(MetadataError::ExtendedXmpDigestMismatch);
    }
    Ok(())
}

/// The value of `HasExtendedXMP` in a standard XMP packet, as an attribute
/// (`xmpNote:HasExtendedXMP="..."` or with single quotes) or an element
/// (`<xmpNote:HasExtendedXMP>...</...>`, the opening tag), trimmed of
/// whitespace: the last occurrence of the name that has either form
/// (ExifTool keeps the last value it reads). The name must follow a `:`, so
/// `y:FakeHasExtendedXMP` is a different name; the namespace prefix itself
/// is not resolved.
///
/// Divergences from ExifTool 12.40, which parses the packet as RDF: a
/// prefixed name in element text or in an XML comment
/// (`<!-- xmpNote:HasExtendedXMP="..." -->`) counts here when `=` and a
/// quote or `>` follow it, and ExifTool ignores it; since the prefix is not
/// resolved, the name under another namespace (`y:HasExtendedXMP`, or even
/// an `xmlns:HasExtendedXMP` declaration) after the real property wins
/// here, while ExifTool reads the packet the `xmpNote` property names; a
/// GUID padded with whitespace (attribute or element) is trimmed and used
/// here, while ExifTool ignores the extended packet ("Ignored non-standard
/// extended XMP").
///
/// Linear time: candidates are tried from the end, and a candidate whose
/// closing `"`, `'` or `<` would have to lie past the last one in the packet
/// fails without a scan, so only the candidate that succeeds is scanned to
/// its closing byte.
fn has_extended_xmp(packet: &[u8]) -> Option<&[u8]> {
    const NAME: &[u8] = b"HasExtendedXMP";
    let last = |b: u8| memchr::memrchr(b, packet);
    let (last_lt, last_dq, last_sq) = (last(b'<'), last(b'"'), last(b'\''));
    memchr::memmem::rfind_iter(packet, NAME).find_map(|at| {
        if at == 0 || packet[at - 1] != b':' {
            return None;
        }
        let rest = packet[at + NAME.len()..].trim_ascii_start();
        let (&sep, rest) = rest.split_first()?;
        let (value, closer, last_closer) = match sep {
            b'=' => {
                let rest = rest.trim_ascii_start();
                let (&quote, value) = rest.split_first()?;
                let last_quote = match quote {
                    b'"' => last_dq,
                    b'\'' => last_sq,
                    _ => return None,
                };
                (value, quote, last_quote)
            }
            b'>' => {
                // Element form: an opening tag `<prefix:HasExtendedXMP>`, not
                // the closing `</prefix:HasExtendedXMP>`. The walk back covers
                // only the tag's own name, so it stays linear.
                let name_start = packet[..at]
                    .iter()
                    .rposition(|&b| {
                        !(b.is_ascii_alphanumeric() || matches!(b, b':' | b'_' | b'-' | b'.'))
                    })
                    .map_or(0, |i| i + 1);
                if name_start == 0 || packet[name_start - 1] != b'<' {
                    return None;
                }
                (rest, b'<', last_lt)
            }
            _ => return None,
        };
        // `value` starts at this offset in `packet`.
        let start = packet.len() - value.len();
        if last_closer? < start {
            return None;
        }
        Some(value[..memchr::memchr(closer, value)?].trim_ascii())
    })
}

/// `true` when `guid` is `digest` as 32 hex digits (either case).
fn guid_matches(guid: &[u8], digest: &[u8; 16]) -> bool {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    if guid.len() != 32 {
        return false;
    }
    digest.iter().enumerate().all(|(i, byte)| {
        guid[2 * i].eq_ignore_ascii_case(&HEX[usize::from(byte >> 4)])
            && guid[2 * i + 1].eq_ignore_ascii_case(&HEX[usize::from(byte & 0x0F)])
    })
}

/// The first run of consecutive APP13 Photoshop items, joined the way
/// ExifTool's `ProcessJPEG` joins them: a run continues while the next
/// segment is APP13 `Photoshop 3.0`. ExifTool finds the next marker by
/// reading up to the first `0xFF` and skipping `0xFF` fill, so fill and stray
/// bytes without an `0xFF` do not end a run; stray bytes holding one (an
/// `FF 00` or an `RSTn`) do, since ExifTool reads that as the next marker. An
/// `Adobe_Photoshop2.5:` segment can only start a run.
///
/// Deliberate divergence: ExifTool applies the header length of the run's
/// *last* segment to the joined buffer (`DirStart = $isOld ? 27 : 14`), so a
/// run that starts with an `Adobe_Photoshop2.5:` segment and continues with
/// `Photoshop 3.0` is read from byte 14, inside the old 27-byte header, and
/// yields no resources. Here each segment's own header is stripped, so that
/// run yields the joined resources.
pub(super) fn photoshop<'a>(
    items: &[Item<'a>],
    limits: &MetadataLimits,
) -> Result<Option<PhotoshopIrb<'a>>, MetadataError> {
    let photoshop_data = |item: &Item<'a>| match item.kind {
        ItemKind::Photoshop(data) => Some(data),
        _ => None,
    };
    let Some(first) = items.iter().position(|item| photoshop_data(item).is_some()) else {
        // A segment may lie past an item-limit cut.
        let cut = items
            .last()
            .is_some_and(|item| matches!(item.kind, ItemKind::Unparsed(Unparsed::LimitExceeded)));
        return if cut {
            Err(MetadataError::ItemLimitReached)
        } else {
            Ok(None)
        };
    };
    // `end`: one past the run's last segment; `next`: the item examined.
    let mut total = photoshop_data(&items[first]).map_or(0, <[u8]>::len);
    let mut segments = 1usize;
    let mut end = first + 1;
    let mut next = end;
    loop {
        while next < items.len()
            && match items[next].kind {
                ItemKind::Padding => true,
                ItemKind::Unparsed(Unparsed::ExtraneousBytes) => {
                    !items[next].payload.contains(&0xFF)
                }
                _ => false,
            }
        {
            next += 1;
        }
        let Some(item) = items.get(next) else { break };
        if matches!(item.kind, ItemKind::Unparsed(Unparsed::LimitExceeded)) {
            return Err(MetadataError::ItemLimitReached);
        }
        match photoshop_data(item) {
            Some(data) if item.payload.starts_with(PHOTOSHOP_SIGNATURE) => {
                total += data.len();
                segments += 1;
                next += 1;
                end = next;
            }
            _ => break,
        }
    }
    check_size(total, limits)?;
    let run = items[first..end].iter().filter_map(photoshop_data);
    if segments == 1 {
        return Ok(run.map(|data| PhotoshopIrb(Cow::Borrowed(data))).next());
    }
    let mut joined = try_vec(total)?;
    for data in run {
        joined.extend_from_slice(data);
    }
    Ok(Some(PhotoshopIrb(Cow::Owned(joined))))
}

/// Photoshop image resource blocks (Adobe Photoshop File Formats
/// Specification, "Image Resource Blocks"), from APP13 `Photoshop 3.0` (or
/// the older `Adobe_Photoshop2.5:`) segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhotoshopIrb<'a>(pub(super) Cow<'a, [u8]>);

impl<'a> PhotoshopIrb<'a> {
    /// The resource block bytes (after each segment's identifier).
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.0
    }

    /// The resource block bytes, keeping the input's lifetime when they are
    /// borrowed (a single segment).
    #[must_use]
    pub fn into_data(self) -> Cow<'a, [u8]> {
        self.0
    }

    /// The resources in order. Iteration stops at the first block whose
    /// signature is not one ExifTool accepts (`8BIM`, `PHUT`, `DCSR`,
    /// `AgHg`, `MeSa`) or that does not fit.
    #[must_use]
    pub fn resources(&self) -> PhotoshopResources<'_> {
        PhotoshopResources(&self.0)
    }

    /// The IPTC-IIM record: the data of the first `8BIM` resource `0x0404`.
    #[must_use]
    pub fn iptc(&self) -> Option<&[u8]> {
        self.resources()
            .find(|r| r.id == 0x0404 && &r.signature == b"8BIM")
            .map(|r| r.data)
    }
}

/// One Photoshop image resource block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PhotoshopResource<'b> {
    /// Block signature, normally `8BIM`.
    pub signature: [u8; 4],
    /// Resource ID (e.g. `0x0404` IPTC-NAA, `0x040F` ICC profile,
    /// `0x0424` XMP).
    pub id: u16,
    /// Resource name (the Pascal string's characters, often empty).
    pub name: &'b [u8],
    /// Resource data.
    pub data: &'b [u8],
}

/// Iterator over the [`crate::container::metadata::PhotoshopResource`]s of a [`crate::container::metadata::PhotoshopIrb`].
#[derive(Debug, Clone)]
pub struct PhotoshopResources<'b>(&'b [u8]);

impl<'b> Iterator for PhotoshopResources<'b> {
    type Item = PhotoshopResource<'b>;

    fn next(&mut self) -> Option<PhotoshopResource<'b>> {
        let parsed = Self::parse_one(self.0);
        match parsed {
            Some((resource, rest)) => {
                self.0 = rest;
                Some(resource)
            }
            None => {
                self.0 = &[];
                None
            }
        }
    }
}

impl<'b> PhotoshopResources<'b> {
    /// Signature, ID u16, Pascal name padded to an even length, size u32,
    /// data padded to an even length; all big-endian (ExifTool
    /// `Photoshop::ProcessPhotoshop`).
    fn parse_one(d: &'b [u8]) -> Option<(PhotoshopResource<'b>, &'b [u8])> {
        let signature: [u8; 4] = d.get(..4)?.try_into().ok()?;
        if !matches!(&signature, b"8BIM" | b"PHUT" | b"DCSR" | b"AgHg" | b"MeSa") {
            return None;
        }
        let id = u16::from_be_bytes([*d.get(4)?, *d.get(5)?]);
        let name_len = usize::from(*d.get(6)?);
        let name = d.get(7..7 + name_len)?;
        // The length byte plus the name are padded to an even count.
        let size_at = 7 + name_len + usize::from(name_len % 2 == 0);
        let size = d.get(size_at..size_at + 4)?;
        let size =
            usize::try_from(u32::from_be_bytes([size[0], size[1], size[2], size[3]])).ok()?;
        let data_at = size_at + 4;
        let data = d.get(data_at..data_at.checked_add(size)?)?;
        let next = (data_at + size + size % 2).min(d.len());
        Some((
            PhotoshopResource {
                signature,
                id,
                name,
                data,
            },
            &d[next..],
        ))
    }
}

fn check_size(size: usize, limits: &MetadataLimits) -> Result<(), MetadataError> {
    if size > limits.max_reassembled_bytes {
        return Err(MetadataError::TooLarge {
            size,
            limit: limits.max_reassembled_bytes,
        });
    }
    Ok(())
}

fn try_vec(capacity: usize) -> Result<Vec<u8>, MetadataError> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity)
        .map_err(|_| MetadataError::OutOfMemory)?;
    Ok(v)
}
