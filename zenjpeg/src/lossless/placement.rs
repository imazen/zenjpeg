//! Where each preserved APPn/COM segment sat in its source, relative to SOF and
//! the first SOS.
//!
//! The decoder decides the colour model at SOF (jpegli's `ProcessSOF`): a JFIF
//! APP0 or Adobe APP14 placed between SOF and the first SOS does not count, and
//! neither does one after the first SOS (libjpeg-turbo, which decides at the
//! first SOS, ignores that one too). A coefficient-domain rewrite (lossless
//! transform, restructure, recompress preserve) that wrote every preserved
//! segment right after SOI would move such a marker in front of SOF and change
//! how the output decodes: `decode(transform(x))` would no longer equal
//! `transform(decode(x))`. The writers here keep each segment where it had the
//! same effect:
//!
//! - before SOF: right after SOI;
//! - between SOF and the first SOS: right after the output's frame header;
//! - after the first SOS: in a progressive output, right before its second scan
//!   header; in a single-scan output, after the frame header, except the
//!   colour signals (JFIF, Adobe, ICC), which had no effect on the source and
//!   would start to count there for decoders that decide at the first SOS.
//!   Preservation writers refuse that single-scan conversion; a multi-scan
//!   output can retain those signals without activating them.
//!
//! The source is walked the way the decoder reads it
//! ([`MarkerIter::for_decoder_metadata`]): stray bytes between segments are
//! skipped, so a stray byte before SOF does not hide which side a segment sat
//! on.

use crate::container::marker::{MarkerIter, MarkerKind};
use crate::decode::{PreservedSegment, SegmentType};

/// Where a preserved segment sat in its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Position {
    /// Before the frame header (or not found in the source verbatim).
    BeforeSof,
    /// Between the frame header and the first SOS.
    AfterSof,
    /// After the first SOS (between scans).
    AfterFirstScan,
}

/// Preserved segments with the position each sat at in the source.
#[derive(Clone, Copy)]
pub(crate) struct Placed<'a> {
    segments: &'a [PreservedSegment],
    /// One entry per segment. Empty means every segment goes before SOF (no
    /// source positions known).
    positions: &'a [Position],
}

impl<'a> Placed<'a> {
    /// Segments with their source positions (from [`source_positions`]).
    pub(crate) fn new(segments: &'a [PreservedSegment], positions: &'a [Position]) -> Self {
        debug_assert!(positions.is_empty() || positions.len() == segments.len());
        Self {
            segments,
            positions,
        }
    }

    fn at(self, position: Position) -> impl Iterator<Item = &'a PreservedSegment> {
        self.segments
            .iter()
            .enumerate()
            .filter(move |(i, _)| {
                self.positions
                    .get(*i)
                    .copied()
                    .unwrap_or(Position::BeforeSof)
                    == position
            })
            .map(|(_, s)| s)
    }

    /// The segments written right after SOI, in source order.
    pub(crate) fn before_sof(self) -> impl Iterator<Item = &'a PreservedSegment> {
        self.at(Position::BeforeSof)
    }

    /// A single-scan output cannot retain these segments in an inert position.
    /// Preservation writers must reject that output structure or choose a
    /// multi-scan structure, rather than silently filtering the segments out.
    pub(crate) fn requires_multiple_scans(self) -> bool {
        self.at(Position::AfterFirstScan).any(is_colour_signal)
    }

    /// The segments written right after the frame header, in source order.
    ///
    /// With `single_scan`, the output has no second scan header to hold the
    /// source's after-first-SOS segments, so they follow here too, minus the
    /// colour signals ([`is_colour_signal`]): those had no effect on the
    /// source's decode, and a decoder that decides the colour model at the
    /// first SOS (libjpeg-turbo) would act on them here. T.81 allows no
    /// segment after the last scan, so this is the only place left for the
    /// other metadata (EXIF, XMP, COM, ...), which decoders do not use to
    /// interpret pixels.
    pub(crate) fn after_sof(self, single_scan: bool) -> impl Iterator<Item = &'a PreservedSegment> {
        self.at(Position::AfterSof).chain(
            self.at(Position::AfterFirstScan)
                .filter(move |s| single_scan && !is_colour_signal(s)),
        )
    }

    /// The segments a multi-scan output writes right before its second scan
    /// header, in source order.
    pub(crate) fn after_first_scan(self) -> impl Iterator<Item = &'a PreservedSegment> {
        self.at(Position::AfterFirstScan)
    }
}

/// A segment that can decide how the pixels are interpreted: JFIF APP0, Adobe
/// APP14 and ICC APP2 (which can select XYB).
pub(crate) fn is_colour_signal(seg: &PreservedSegment) -> bool {
    matches!(
        seg.segment_type,
        SegmentType::Jfif | SegmentType::Adobe | SegmentType::Icc
    )
}

/// For each preserved segment, where `source` holds it. Segments are matched
/// to the source's APPn/COM segments by marker and payload, in order; a
/// segment the source does not hold verbatim (an edited EXIF, say) is placed
/// before SOF.
pub(crate) fn source_positions(
    source: &[u8],
    segments: &[PreservedSegment],
) -> alloc::vec::Vec<Position> {
    let mut found = alloc::vec::Vec::new();
    let mut at = Position::BeforeSof;
    for span in MarkerIter::for_decoder_metadata(source) {
        match span.kind {
            MarkerKind::Sof(_) if at == Position::BeforeSof => at = Position::AfterSof,
            MarkerKind::Sos => at = Position::AfterFirstScan,
            MarkerKind::Eoi => break,
            MarkerKind::App(_) | MarkerKind::Com => {
                let byte = source.get(span.offset + 1).copied().unwrap_or(0);
                found.push((byte, span.payload, at, false));
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
                None => Position::BeforeSof,
            }
        })
        .collect()
}
