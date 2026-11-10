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

/// Native compatibility for a complete ICC carried as a single chunk 0 of 1.
/// Structural inspection retains the canonical 1-based contract above.
pub(crate) fn icc_profile_for_decode<'a>(
    chunks: impl Iterator<Item = (u8, u8, &'a [u8])>,
    limit: usize,
) -> Result<Option<Cow<'a, [u8]>>, Error> {
    let mut chunks = chunks.peekable();
    if chunks.peek().is_some_and(|&(sequence, count, data)| {
        sequence == 0
            && count == 1
            && data.len() >= 132
            && data.get(36..40) == Some(b"acsp")
            && u32::from_be_bytes(data[..4].try_into().unwrap()) as usize == data.len()
    }) {
        let first = chunks.next().unwrap();
        if chunks.peek().is_none() {
            return icc_profile(core::iter::once((1, 1, first.2)), limit);
        }
        return icc_profile(core::iter::once(first).chain(chunks), limit);
    }
    icc_profile(chunks, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_numbering_never_weakens_chunk_or_size_limits() {
        let mut profile = [0u8; 132];
        profile[..4].copy_from_slice(&132u32.to_be_bytes());
        profile[36..40].copy_from_slice(b"acsp");
        assert_eq!(
            icc_profile_for_decode([(0, 1, profile.as_slice())].into_iter(), 132)
                .unwrap()
                .unwrap()
                .as_ref(),
            profile
        );
        assert!(matches!(
            icc_profile_for_decode([(0, 1, profile.as_slice())].into_iter(), 131),
            Err(Error::TooLarge { .. })
        ));
        for second in [0, 1] {
            assert!(
                icc_profile_for_decode(
                    [(0, 1, profile.as_slice()), (second, 1, profile.as_slice())].into_iter(),
                    264
                )
                .is_err()
            );
        }
        profile[3] = 131;
        assert!(icc_profile_for_decode([(0, 1, profile.as_slice())].into_iter(), 132).is_err());
        assert!(icc_profile([(0, 1, profile.as_slice())].into_iter(), 132).is_err());
    }
}
