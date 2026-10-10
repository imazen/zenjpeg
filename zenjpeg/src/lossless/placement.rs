//! Which side of SOF each preserved APPn/COM segment sat on in its source.
//!
//! The decoder decides the colour model at SOF (jpegli's `ProcessSOF`): a JFIF
//! APP0 or Adobe APP14 placed between SOF and the first SOS does not count. A
//! coefficient-domain rewrite (lossless transform, restructure, recompress
//! preserve) that wrote every preserved segment right after SOI would move
//! such a marker in front of SOF and change how the output decodes:
//! `decode(transform(x))` would no longer equal `transform(decode(x))`. The
//! writers here keep each segment on its side of SOF, so the output decodes
//! like the source under both decision rules (at SOF, and libjpeg-turbo's at
//! the first SOS).

use crate::container::marker::{self, MarkerKind};
use crate::decode::PreservedSegment;

/// Preserved segments with the side of SOF each sat on in the source.
#[derive(Clone, Copy)]
pub(crate) struct Placed<'a> {
    segments: &'a [PreservedSegment],
    /// One flag per segment, `true` for between SOF and the first SOS. Empty
    /// means every segment goes before SOF (no source positions known).
    after_sof: &'a [bool],
}

impl<'a> Placed<'a> {
    /// Segments with their source positions (`after_sof` from [`after_sof_flags`]).
    pub(crate) fn new(segments: &'a [PreservedSegment], after_sof: &'a [bool]) -> Self {
        debug_assert!(after_sof.is_empty() || after_sof.len() == segments.len());
        Self {
            segments,
            after_sof,
        }
    }

    /// The segments on one side of SOF, in source order.
    pub(crate) fn side(self, after_sof: bool) -> impl Iterator<Item = &'a PreservedSegment> {
        self.segments
            .iter()
            .enumerate()
            .filter(move |(i, _)| self.after_sof.get(*i).copied().unwrap_or(false) == after_sof)
            .map(|(_, s)| s)
    }
}

/// For each preserved segment, whether `source` holds it between SOF and the
/// first SOS. Segments are matched to the source's APPn/COM segments by marker
/// and payload, in order; a segment the source does not hold verbatim (an
/// edited EXIF, say) is placed before SOF.
pub(crate) fn after_sof_flags(
    source: &[u8],
    segments: &[PreservedSegment],
) -> alloc::vec::Vec<bool> {
    let mut found = alloc::vec::Vec::new();
    let mut seen_sof = false;
    for span in marker::iter(source) {
        match span.kind {
            MarkerKind::Sof(_) => seen_sof = true,
            MarkerKind::Sos | MarkerKind::Eoi => break,
            MarkerKind::App(_) | MarkerKind::Com => {
                let byte = source.get(span.offset + 1).copied().unwrap_or(0);
                found.push((byte, span.payload, seen_sof, false));
            }
            _ => {}
        }
    }
    segments
        .iter()
        .map(|seg| {
            match found
                .iter_mut()
                .find(|(m, p, _, used)| !*used && *m == seg.marker && *p == seg.data.as_slice())
            {
                Some(entry) => {
                    entry.3 = true;
                    entry.2
                }
                None => false,
            }
        })
        .collect()
}
