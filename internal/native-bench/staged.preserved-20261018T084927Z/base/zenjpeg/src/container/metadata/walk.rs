//! The segment walk: [`MarkerIter`] plus the bytes it cannot structure.
//!
//! [`walk_segments`] turns the marker iterator's spans into [`Item`]s and
//! covers everything between them: fill bytes become [`ItemKind::Padding`],
//! and where [`MarkerIter`] stops on malformed structure the walk reports the
//! bad bytes and resynchronizes at the next marker the way libjpeg-turbo's
//! `next_marker` (jdmarker.c) does, so a stray byte run does not hide the
//! rest of the file. [`collect`] adds the trailer after EOI and the item
//! limit.

use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::identify::identify;
use super::{Item, ItemKind, MetadataError, Unparsed};
use crate::container::marker::{IterStop, MarkerIter};

const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOS: u8 = 0xDA;
/// Item-list capacity reserved before the walk.
const INITIAL_ITEMS: usize = 32;

/// Walk `data` from SOI through the primary image's EOI (or the end of the
/// input), calling `f` once per item in file order.
///
/// Items tile `data[..end]`, where `end` is the returned EOI end or, without
/// an EOI, `data.len()`. Returns `Break` if `f` did, else
/// `Continue(Some(offset just past EOI))` or `Continue(None)` when there is
/// no EOI. Allocation-free.
pub(super) fn walk_segments<'a, F>(data: &'a [u8], f: &mut F) -> ControlFlow<(), Option<usize>>
where
    F: FnMut(Item<'a>) -> ControlFlow<()>,
{
    if data.is_empty() {
        return ControlFlow::Continue(None);
    }
    // libjpeg's first_marker: the input must open with exactly FF D8.
    if data.len() < 2 || data[0] != 0xFF || data[1] != SOI {
        f(unparsed(data, 0, data.len(), None, Unparsed::NotJpeg))?;
        return ControlFlow::Continue(None);
    }

    let mut base = 0usize; // offset of the iterator's slice within `data`
    let mut iter = MarkerIter::new(data);
    let mut pos = 0usize; // end of the last item handed to `f`
    loop {
        // `FF 00` where a marker should be is a stuffed zero, not a marker:
        // libjpeg's next_marker discards it and keeps scanning, and so does
        // the walk (MarkerIter would read a length word after it).
        let here = base + iter.position();
        if data.get(here) == Some(&0xFF) {
            let last = last_ff_of_run(data, here);
            if data.get(last + 1) == Some(&0x00) {
                let next = next_marker(data, last + 2);
                f(unparsed(data, here, next, None, Unparsed::ExtraneousBytes))?;
                if next >= data.len() {
                    return ControlFlow::Continue(None);
                }
                base = next;
                pos = next;
                iter = MarkerIter::new(&data[next..]);
                continue;
            }
        }
        if let Some(span) = iter.next() {
            let at = base + span.offset;
            if at > pos {
                // MarkerIter skipped fill bytes (T.81 B.1.1.2) before this marker.
                f(simple(data, pos, at, ItemKind::Padding))?;
            }
            let end = at + span.length;
            let marker = data[at + 1];
            let kind = match marker {
                SOI => ItemKind::Soi,
                EOI => ItemKind::Eoi,
                0xD0..=0xD7 => ItemKind::Restart,
                _ if span.length == 2 => ItemKind::Unknown, // TEM
                SOS => {
                    let entropy = &data[at + 4 + span.payload.len()..end];
                    identify(marker, span.payload, entropy)
                }
                _ => identify(marker, span.payload, &[]),
            };
            f(Item {
                offset: at,
                length: span.length,
                marker: Some(marker),
                payload: span.payload,
                kind,
            })?;
            pos = end;
            if marker == EOI {
                return ControlFlow::Continue(Some(end));
            }
            continue;
        }

        // The iterator stopped: cleanly at the end of the input, or on
        // malformed structure that the stop reason locates.
        let Some(stop) = iter.stop() else {
            debug_assert!(iter.saw_eoi() || base + iter.position() >= data.len());
            return ControlFlow::Continue(None);
        };
        let resume = match stop {
            IterStop::NotAMarker { offset } => {
                let at = base + offset;
                let next = next_marker(data, at);
                f(unparsed(data, at, next, None, Unparsed::ExtraneousBytes))?;
                next
            }
            IterStop::BadLength { offset } => {
                let fill = base + offset;
                let at = last_ff_of_run(data, fill);
                if at > fill {
                    f(simple(data, fill, at, ItemKind::Padding))?;
                }
                // Marker + length word; MarkerIter only reports BadLength
                // when all four bytes are present.
                let end = at + 4;
                let mut item = unparsed(data, at, end, Some(data[at + 1]), Unparsed::BadLength);
                item.payload = &[]; // nothing follows the length word
                f(item)?;
                // Resume right after the length word: a marker there (an
                // RSTn included) is walked as usual, and stray bytes become
                // an ExtraneousBytes item on the next pass.
                end
            }
            IterStop::Truncated { offset } => {
                let fill = base + offset;
                let at = last_ff_of_run(data, fill);
                let marker = data.get(at + 1).copied();
                let start = if marker.is_some() && at > fill {
                    f(simple(data, fill, at, ItemKind::Padding))?;
                    at
                } else {
                    fill
                };
                let mut item =
                    unparsed(data, start, data.len(), marker, Unparsed::TruncatedSegment);
                if marker.is_some() {
                    // Whatever follows the (possibly cut) length word.
                    item.payload = data.get(start + 4..).unwrap_or(&[]);
                }
                f(item)?;
                data.len()
            }
        };
        if resume >= data.len() {
            return ControlFlow::Continue(None);
        }
        base = resume;
        pos = resume;
        iter = MarkerIter::new(&data[resume..]);
    }
}

/// Collect every item of `data` into `out`, trailers included, honoring
/// `max_items` (at least 1). Returns the offset just past the primary EOI.
pub(super) fn collect<'a>(
    data: &'a [u8],
    max_items: usize,
    out: &mut Vec<Item<'a>>,
) -> Result<Option<usize>, MetadataError> {
    // Real files carry tens of items: reserve once, then grow by doubling.
    out.try_reserve(max_items.min(INITIAL_ITEMS))
        .map_err(|_| MetadataError::OutOfMemory)?;
    let mut sink = Sink {
        out,
        max_items,
        data,
        error: None,
    };
    let walked = walk_segments(data, &mut |item| sink.push(item));
    let eoi_end = match walked {
        ControlFlow::Continue(eoi_end) => {
            if let Some(start) = eoi_end
                && start < data.len()
            {
                super::trailer::collect(data, start, &mut sink)?;
            }
            eoi_end
        }
        ControlFlow::Break(()) => {
            if let Some(e) = sink.error {
                return Err(e);
            }
            // The limit item stopped the walk before the primary EOI: an EOI
            // that ends the input is pushed as itself and the walk returns
            // `Continue`, and any other EOI would have been replaced by the
            // limit item covering it.
            None
        }
    };
    match sink.error {
        Some(e) => Err(e),
        None => Ok(eoi_end),
    }
}

/// Appends items to the output, enforcing the item limit: the last slot
/// holds an [`Unparsed::LimitExceeded`] item covering the rest of the input.
pub(super) struct Sink<'a, 'v> {
    out: &'v mut Vec<Item<'a>>,
    max_items: usize,
    data: &'a [u8],
    pub(super) error: Option<MetadataError>,
}

impl<'a> Sink<'a, '_> {
    /// Items that can still be pushed before the limit item.
    pub(super) fn remaining(&self) -> usize {
        self.max_items.saturating_sub(self.out.len())
    }

    pub(super) fn push(&mut self, item: Item<'a>) -> ControlFlow<()> {
        let item = if self.out.len() + 1 >= self.max_items
            && item.offset + item.length < self.data.len()
        {
            unparsed(
                self.data,
                item.offset,
                self.data.len(),
                None,
                Unparsed::LimitExceeded,
            )
        } else {
            item
        };
        if self.out.try_reserve(1).is_err() {
            self.error = Some(MetadataError::OutOfMemory);
            return ControlFlow::Break(());
        }
        let limit_hit = matches!(item.kind, ItemKind::Unparsed(Unparsed::LimitExceeded));
        self.out.push(item);
        if limit_hit {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// An item with no marker whose payload is all of its bytes.
pub(super) fn simple<'a>(data: &'a [u8], start: usize, end: usize, kind: ItemKind<'a>) -> Item<'a> {
    Item {
        offset: start,
        length: end - start,
        marker: None,
        payload: &data[start..end],
        kind,
    }
}

fn unparsed(data: &[u8], start: usize, end: usize, marker: Option<u8>, why: Unparsed) -> Item<'_> {
    Item {
        marker,
        ..simple(data, start, end, ItemKind::Unparsed(why))
    }
}

/// Given the first `0xFF` of a fill run at `start`, the offset of the run's
/// last `0xFF` (the one directly before the marker identifier).
fn last_ff_of_run(data: &[u8], start: usize) -> usize {
    let mut at = start;
    while at + 1 < data.len() && data[at + 1] == 0xFF {
        at += 1;
    }
    at
}

/// Where walking resumes after stray bytes: the first `0xFF` (of a fill
/// run) at or after `start` that begins a marker other than a restart
/// marker. Stuffed `FF 00` pairs are skipped as libjpeg-turbo's
/// `next_marker` (jdmarker.c) skips them, and `RSTn` markers as its
/// `read_markers` skips a parameterless marker between segments, so a run of
/// entropy-coded data that lost its scan header stays one item. A run of
/// `0xFF` that reaches the end of the input counts as a (cut) marker.
/// `data.len()` when there is none.
fn next_marker(data: &[u8], start: usize) -> usize {
    let mut from = start;
    while from < data.len() {
        let Some(rel) = memchr::memchr(0xFF, &data[from..]) else {
            break;
        };
        let ff = from + rel;
        let last = last_ff_of_run(data, ff);
        match data.get(last + 1) {
            Some(0x00 | 0xD0..=0xD7) => from = last + 2,
            _ => return ff,
        }
    }
    data.len()
}
