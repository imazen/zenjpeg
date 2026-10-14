//! Everything after the primary image's EOI.
//!
//! Vendor trailers end with a footer that gives their own length, so they
//! are found the way ExifTool's `IdentifyTrailer`/`ProcessTrailers` find
//! them: look at the last bytes of the file, identify the trailer by its
//! footer signature, compute where it starts, and repeat on the bytes before
//! it. Footer checks run in ExifTool's order (AFCP, FotoStation,
//! PhotoMechanic, CanonVRD, MIE, Samsung); the length rules come from the
//! matching ExifTool reader, cited per function. What precedes the vendor
//! trailers is scanned forward for appended JPEG streams (MPF images,
//! preview images); the gaps between them are padding or unknown data.
//!
//! ExifTool 12.40's `IdentifyTrailer` also names two footers this module
//! does not size: Insta360 (`8db42d694ccc418790edff439fe026bf`) and
//! NikonApp (six NULs + `/NIKON APP`). Their readers are not implemented, so
//! such a trailer, and any vendor trailer before it, is reported as
//! [`crate::container::metadata::ItemKind::UnknownTrailer`]
//! (or as appended JPEG streams where those walk cleanly).
//!
//! Allocation: the vendor trailers found are listed (last first) in a vector
//! that grows fallibly, one entry per trailer or FotoStation record, never
//! past the remaining item budget.

use alloc::vec::Vec;
use core::ops::{ControlFlow, Range};

use super::walk::{Sink, simple, walk_segments};
use super::{ItemKind, MetadataError};

/// A trailer found from its footer: its range and its kind.
type Found = (Range<usize>, ItemKind<'static>);

/// Push every trailer item of `data[start..]` into `sink`, in file order.
pub(super) fn collect<'a>(
    data: &'a [u8],
    start: usize,
    sink: &mut Sink<'a, '_>,
) -> Result<(), MetadataError> {
    // Vendor trailers, last first.
    let mut chain: Vec<Found> = Vec::new();
    let mut end = data.len();
    let budget = sink.remaining();
    while chain.len() < budget {
        let before = chain.len();
        if footer_trailer(data, start, end, budget, &mut chain).is_err() {
            return Err(MetadataError::OutOfMemory);
        }
        let Some((range, _)) = chain.last().filter(|_| chain.len() > before) else {
            break;
        };
        end = range.start;
    }

    if forward_region(data, start, end, sink).is_break() {
        return Ok(());
    }
    for (range, kind) in chain.into_iter().rev() {
        if sink
            .push(simple(data, range.start, range.end, kind))
            .is_break()
        {
            break;
        }
    }
    Ok(())
}

/// Append the vendor trailer(s) whose footer ends at `end` (bounded below by
/// `start`) to `chain`, last first, never growing `chain` past `budget`
/// entries. FotoStation contributes one entry per record; every other type
/// one entry. Appends nothing when the bytes before `end` end in no
/// recognized footer.
fn footer_trailer(
    data: &[u8],
    start: usize,
    end: usize,
    budget: usize,
    chain: &mut Vec<Found>,
) -> Result<(), alloc::collections::TryReserveError> {
    let region = &data[start..end];
    let n = region.len();
    // As in IdentifyTrailer, the footer signature alone picks the type; a
    // footer whose length does not check out ends the chain.
    let found = if n >= 12
        && region[n - 12..].starts_with(b"AXS")
        && matches!(region[n - 9], b'!' | b'*')
    {
        afcp(data, start, end).map(|s| (s..end, ItemKind::Afcp))
    } else if region.ends_with(&FOTOSTATION_SIGNATURE) {
        return fotostation(data, start, end, budget, chain);
    } else if region.ends_with(b"cbipcbbl") {
        photo_mechanic(region).map(|s| (start + s..end, ItemKind::PhotoMechanic))
    } else if n >= 0x40 && region[n - 0x40..].starts_with(CANON_VRD_SIGNATURE) {
        canon_vrd(region).map(|s| (start + s..end, ItemKind::CanonVrd))
    } else if let Some(s) = mie(region) {
        Some((start + s..end, ItemKind::Mie))
    } else {
        samsung(region).map(|s| (start + s..end, ItemKind::Samsung))
    };
    if let Some(found) = found {
        chain.try_reserve(1)?;
        chain.push(found);
    }
    Ok(())
}

/// AFCP (ExifTool `AFCP::ProcessAFCP`): a 12-byte footer `AXS!` (big-endian)
/// or `AXS*` (little-endian) + the absolute file offset of the AFCP header +
/// 4 bytes; the header there begins with the same 4 bytes.
fn afcp(data: &[u8], start: usize, end: usize) -> Option<usize> {
    let footer = data.get(end.checked_sub(12)?..end)?;
    if end - 12 < start || !footer.starts_with(b"AXS") || !matches!(footer[3], b'!' | b'*') {
        return None;
    }
    let offset = [footer[4], footer[5], footer[6], footer[7]];
    let header = usize::try_from(if footer[3] == b'!' {
        u32::from_be_bytes(offset)
    } else {
        u32::from_le_bytes(offset)
    })
    .ok()?;
    let header_end = header.checked_add(12)?;
    (header >= start && header_end <= end - 12 && data[header..header + 4] == footer[..4])
        .then_some(header)
}

const FOTOSTATION_SIGNATURE: [u8; 4] = [0xA1, 0xB2, 0xC3, 0xD4];

/// FotoStation (ExifTool `FotoStation::ProcessFotoStation`): records chained
/// backward from the end, each ending in a 10-byte footer `tag u16, size u32
/// (record size including the footer), A1 B2 C3 D4`, big-endian.
fn fotostation(
    data: &[u8],
    start: usize,
    end: usize,
    budget: usize,
    chain: &mut Vec<Found>,
) -> Result<(), alloc::collections::TryReserveError> {
    let mut cur = end;
    while cur - start >= 10 && chain.len() < budget {
        let footer = &data[cur - 10..cur];
        if footer[6..] != FOTOSTATION_SIGNATURE {
            break;
        }
        let tag = u16::from_be_bytes([footer[0], footer[1]]);
        let size = u32::from_be_bytes([footer[2], footer[3], footer[4], footer[5]]);
        let Ok(size) = usize::try_from(size) else {
            break;
        };
        if size < 10 || size > cur - start {
            break;
        }
        chain.try_reserve(1)?;
        chain.push((cur - size..cur, ItemKind::FotoStation(tag)));
        cur -= size;
    }
    Ok(())
}

/// Photo Mechanic (ExifTool `PhotoMechanic::ProcessPhotoMechanic`): a
/// 12-byte footer `size u32 (big-endian), "cbipcbbl"` after `size` bytes of
/// data. Returns the start offset within `region`.
fn photo_mechanic(region: &[u8]) -> Option<usize> {
    let footer = region.get(region.len().checked_sub(12)?..)?;
    if &footer[4..] != b"cbipcbbl" {
        return None;
    }
    let size = u32::from_be_bytes([footer[0], footer[1], footer[2], footer[3]]);
    if size & 0x8000_0000 != 0 {
        return None;
    }
    let total = usize::try_from(size).ok()?.checked_add(12)?;
    region.len().checked_sub(total)
}

const CANON_VRD_SIGNATURE: &[u8] = b"CANON OPTIONAL DATA\0";

/// Canon VRD (ExifTool `CanonVRD::ProcessCanonVRD`): a 0x1C-byte header and a
/// 0x40-byte footer, both starting with `CANON OPTIONAL DATA\0`; the footer
/// holds the data size (big-endian u32 at 0x14), so the trailer is
/// `size + 0x5C` bytes. Returns the start offset within `region`.
fn canon_vrd(region: &[u8]) -> Option<usize> {
    let footer = region.get(region.len().checked_sub(0x40)?..)?;
    if !footer.starts_with(CANON_VRD_SIGNATURE) {
        return None;
    }
    let size = u32::from_be_bytes([footer[0x14], footer[0x15], footer[0x16], footer[0x17]]);
    let total = usize::try_from(size).ok()?.checked_add(0x5C)?;
    if total >= 0x8000_0000 {
        return None;
    }
    let s = region.len().checked_sub(total)?;
    region[s..].starts_with(CANON_VRD_SIGNATURE).then_some(s)
}

/// MIE (ExifTool `IdentifyTrailer` + `MIE::ProcessMIE`): the last bytes are
/// the `zmie` trailer signature element followed by the group terminator,
/// whose 4- or 8-byte data length is the group length (byte order `0x10`
/// big-endian, `0x18` little-endian). Further MIE groups may precede it,
/// each ending in a terminator. The first group must open with the `0MIE`
/// element (`~[\x10\x18]\x04.0MIE`, `ProcessMIE`), or ExifTool does not
/// read the trailer. Returns the start offset within `region`.
fn mie(region: &[u8]) -> Option<usize> {
    const ZMIE: &[u8] = b"~\0\x04\0zmie~\0\0";
    let n = region.len();
    // `~\0\x04\0zmie~\0\0` + 0x06 + length[4] + order + 0x04, or
    // `~\0\x04\0zmie~\0\0` + 0x0A + length[8] + order + 0x08.
    let short = n >= 18
        && &region[n - 18..n - 7] == ZMIE
        && region[n - 7] == 0x06
        && matches!(region[n - 2], 0x10 | 0x18)
        && region[n - 1] == 0x04;
    let long = n >= 22
        && &region[n - 22..n - 11] == ZMIE
        && region[n - 11] == 0x0A
        && matches!(region[n - 2], 0x10 | 0x18)
        && region[n - 1] == 0x08;
    if !short && !long {
        return None;
    }
    let mut first = None;
    let mut cur = n;
    while cur >= 10 {
        let t = &region[cur - 10..cur];
        let big = match t[8] {
            0x10 => true,
            0x18 => false,
            _ => break,
        };
        let len = match t[9] {
            0x04 if t[..4] == *b"~\0\0\x06" => {
                let b = [t[4], t[5], t[6], t[7]];
                u64::from(if big {
                    u32::from_be_bytes(b)
                } else {
                    u32::from_le_bytes(b)
                })
            }
            0x08 if cur >= 14 && region[cur - 14..cur - 10] == *b"~\0\0\x0a" => {
                let b = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]];
                if big {
                    u64::from_be_bytes(b)
                } else {
                    u64::from_le_bytes(b)
                }
            }
            _ => break,
        };
        let Ok(len) = usize::try_from(len) else {
            break;
        };
        if len < 12 || len > cur || (t[9] == 0x08 && len < 14) {
            break;
        }
        cur -= len;
        first = Some(cur);
    }
    let first = first?;
    match region.get(first..first + 8)? {
        [b'~', 0x10 | 0x18, 0x04, _, b'0', b'M', b'I', b'E'] => Some(first),
        _ => None,
    }
}

/// Samsung (ExifTool `Samsung::ProcessSamsung`): blocks chained backward from
/// a `\0\0SEFT` (or `\0\0QDIOBS`, as `IdentifyTrailer` requires) ending,
/// each `data, length u32 LE, type[4]`.
/// The `SEFT` block holds the `SEFH` directory: `SEFH, version u32, count
/// u32`, then 12-byte entries whose second u32 is the entry's distance back
/// from the directory; the trailer starts at the farthest one. Returns the
/// start offset within `region`.
fn samsung(region: &[u8]) -> Option<usize> {
    let n = region.len();
    let tail = region.get(n.checked_sub(6)?..)?;
    let mut block_end = match tail {
        b"\0\0SEFT" => n,
        b"QDIOBS" if region.get(n.checked_sub(8)?..n - 6)? == b"\0\0" => n - 2,
        _ => return None,
    };
    loop {
        let footer = region.get(block_end.checked_sub(8)?..block_end)?;
        let len = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
        let kind = &footer[4..];
        if !kind.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_') {
            return None;
        }
        let len = usize::try_from(len).ok()?;
        if !(4..0x10000).contains(&len) || len + 8 >= block_end {
            return None;
        }
        let block = &region[block_end - 8 - len..block_end - 8];
        block_end -= len + 8;
        if kind != b"SEFT" {
            continue;
        }
        if len < 12 || !block.starts_with(b"SEFH") {
            return None;
        }
        let dir = block_end;
        let count = usize::try_from(u32::from_le_bytes([
            block[8], block[9], block[10], block[11],
        ]))
        .ok()?;
        if count.checked_mul(12)?.checked_add(12)? > len {
            return None;
        }
        let farthest = (0..count)
            .map(|i| {
                let e = 12 + 12 * i + 4;
                u32::from_le_bytes([block[e], block[e + 1], block[e + 2], block[e + 3]])
            })
            .max()
            .unwrap_or(0);
        return dir.checked_sub(usize::try_from(farthest).ok()?);
    }
}

/// Items for `data[start..end]`: appended JPEG streams, and the bytes around
/// them as [`ItemKind::Padding`] (all `0x00` or all `0xFF`) or
/// [`ItemKind::UnknownTrailer`].
///
/// An appended stream is an `FF D8` from which [`walk_segments`] reaches an
/// EOI within the region with nothing unparsed on the way: the same walk
/// that delimits the primary image, so an appended image walked on its own
/// ends exactly at its item's end. A candidate whose walk hits unparsed
/// bytes is not an image; the search resumes where that walk stopped, and
/// the first candidate that reaches the end of the region without an EOI
/// ends the search. Candidate walks never overlap: past where the search
/// resumes, a failed walk has read only the stray-byte run it stopped on,
/// and that run holds no `FF D8` (resynchronization stops at any marker but
/// `RSTn`). Each byte is read by at most one walk and the search, so the
/// scan is linear in the region length (about two passes at worst).
fn forward_region<'a>(
    data: &'a [u8],
    start: usize,
    end: usize,
    sink: &mut Sink<'a, '_>,
) -> ControlFlow<()> {
    let region = &data[start..end];
    let mut pos = 0; // relative to `start`: end of the last item pushed
    let mut from = 0;
    while let Some(rel) = memchr::memchr(0xFF, &region[from..]) {
        let soi = from + rel;
        if region.get(soi + 1) != Some(&0xD8) {
            from = soi + 1;
            continue;
        }
        let mut unparsed_at = None;
        let walked = walk_segments(&region[soi..], &mut |item| {
            if matches!(item.kind, ItemKind::Unparsed(_)) {
                unparsed_at = Some(item.offset);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        let len = match walked {
            ControlFlow::Continue(Some(len)) => len,
            ControlFlow::Continue(None) => break,
            ControlFlow::Break(()) => {
                // Not a clean stream; skip what the walk consumed.
                from = soi + unparsed_at.unwrap_or(0).max(2);
                continue;
            }
        };
        if soi > pos {
            sink.push(gap(data, start + pos, start + soi))?;
        }
        sink.push(simple(
            data,
            start + soi,
            start + soi + len,
            ItemKind::EmbeddedJpeg,
        ))?;
        pos = soi + len;
        from = pos;
    }
    if region.len() > pos {
        sink.push(gap(data, start + pos, end))?;
    }
    ControlFlow::Continue(())
}

fn gap(data: &[u8], start: usize, end: usize) -> super::Item<'_> {
    let bytes = &data[start..end];
    let padding = bytes.iter().all(|&b| b == 0) || bytes.iter().all(|&b| b == 0xFF);
    let kind = if padding {
        ItemKind::Padding
    } else {
        ItemKind::UnknownTrailer
    };
    simple(data, start, end, kind)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fotostation_chain_never_grows_past_the_budget() {
        // Five chained records; a budget of 2 keeps the last two (the chain
        // is built backward from the end) and reads no further.
        let mut data = Vec::new();
        for tag in 1..=5u16 {
            let body = [tag as u8; 3];
            data.extend_from_slice(&body);
            data.extend_from_slice(&tag.to_be_bytes());
            data.extend_from_slice(&13u32.to_be_bytes());
            data.extend_from_slice(&FOTOSTATION_SIGNATURE);
        }
        let mut chain = Vec::new();
        fotostation(&data, 0, data.len(), 2, &mut chain).unwrap();
        let tags: Vec<_> = chain.iter().map(|(_, k)| k.clone()).collect();
        assert_eq!(tags, [ItemKind::FotoStation(5), ItemKind::FotoStation(4)]);
        let mut all = Vec::new();
        fotostation(&data, 0, data.len(), 16, &mut all).unwrap();
        assert_eq!(all.len(), 5);
    }
}
