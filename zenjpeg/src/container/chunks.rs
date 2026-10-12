//! JPEG ICC chunk validation shared by native decoding and metadata inspection.
use alloc::{borrow::Cow, vec::Vec};

#[derive(Debug)]
#[cfg_attr(not(feature = "extended-metadata"), allow(dead_code))]
pub(crate) enum Error {
    IccZeroChunkCount,
    IccChunkCountMismatch { expected: u8, found: u8 },
    IccSequenceOutOfRange { sequence: u8, count: u8 },
    IccDuplicateChunk { sequence: u8 },
    IccMissingChunk { sequence: u8 },
    IccEmptyProfile,
    TooLarge { size: usize, limit: usize },
    OutOfMemory,
}

/// Chunks are `(sequence, count, payload)`, excluding APP2 framing.
pub(crate) fn icc_profile<'a>(
    mut chunks: impl Iterator<Item = (u8, u8, &'a [u8])>,
    limit: usize,
) -> Result<Option<Cow<'a, [u8]>>, Error> {
    let Some(first) = chunks.next() else {
        return Ok(None);
    };
    let count = first.1;
    if count == 0 {
        return Err(Error::IccZeroChunkCount);
    }
    // Indexed by sequence number; slot 0 is unused.
    let mut by_seq: [Option<&'a [u8]>; 256] = [None; 256];
    let mut total = 0usize;
    for chunk in core::iter::once(first).chain(chunks) {
        if chunk.1 != count {
            return Err(Error::IccChunkCountMismatch {
                expected: count,
                found: chunk.1,
            });
        }
        if chunk.0 == 0 || chunk.0 > count {
            return Err(Error::IccSequenceOutOfRange {
                sequence: chunk.0,
                count,
            });
        }
        let slot = &mut by_seq[usize::from(chunk.0)];
        if slot.is_some() {
            return Err(Error::IccDuplicateChunk { sequence: chunk.0 });
        }
        *slot = Some(chunk.2);
        total = total.checked_add(chunk.2.len()).ok_or(Error::TooLarge {
            size: usize::MAX,
            limit,
        })?;
    }
    if let Some(missing) = (1..=count).find(|&s| by_seq[usize::from(s)].is_none()) {
        return Err(Error::IccMissingChunk { sequence: missing });
    }
    // jpeg_read_icc_profile rejects a total length of 0 (JWRN_BOGUS_ICC).
    if total == 0 {
        return Err(Error::IccEmptyProfile);
    }
    if total > limit {
        return Err(Error::TooLarge { size: total, limit });
    }
    if count == 1 {
        return Ok(by_seq[1].map(Cow::Borrowed));
    }
    let mut profile = Vec::new();
    profile
        .try_reserve_exact(total)
        .map_err(|_| Error::OutOfMemory)?;
    for data in by_seq[1..=usize::from(count)].iter().flatten() {
        profile.extend_from_slice(data);
    }
    Ok(Some(Cow::Owned(profile)))
}
