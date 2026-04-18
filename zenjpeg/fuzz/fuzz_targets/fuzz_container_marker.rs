//! Fuzz target for `zenjpeg::container::marker`.
//!
//! Exercises all three public entry points with adversarial bytes:
//!   1. `iter(data)` exhausted to completion
//!   2. `primary_bounds(data)`
//!   3. `find_jpeg_boundaries(data)`
//!
//! Invariants checked on every fuzz input:
//!
//! - No panic, no unhandled arithmetic overflow, no OOM.
//! - `iter()` terminates in O(data.len()) iterations.
//! - Every yielded `MarkerSpan` has `offset + length <= data.len()`.
//! - Every yielded `MarkerSpan::payload` is a slice of `data` of length
//!   `<= length` (zero-copy, inside the buffer).
//! - Yielded spans are non-overlapping and strictly forward-progressing.
//! - `primary_bounds` agrees with `find_jpeg_boundaries` on the first
//!   element when both produce a result.
//! - Every `find_jpeg_boundaries` range is `[start, end]` with
//!   `start < end <= data.len()` and the ranges are strictly ordered.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::container::{find_jpeg_boundaries, iter, primary_bounds};

fuzz_target!(|data: &[u8]| {
    // Cap input length to keep the fuzzer from exploring pathological
    // GB-scale inputs. 4 MiB is enough to hit every structural edge case
    // in the scanner.
    let data = if data.len() > 4 * 1024 * 1024 {
        &data[..4 * 1024 * 1024]
    } else {
        data
    };

    // 1. Iterator must terminate and respect slice boundaries.
    let mut last_end = 0usize;
    let mut iterations = 0usize;
    for span in iter(data) {
        iterations += 1;
        // Hard cap on iterations — one per byte is generous.
        assert!(
            iterations <= data.len() + 2,
            "MarkerIter yielded more than data.len()+2 spans — non-terminating?"
        );
        // In-bounds offset + length.
        assert!(
            span.offset + span.length <= data.len(),
            "span beyond buffer: offset={}, length={}, data.len()={}",
            span.offset,
            span.length,
            data.len()
        );
        // Forward progress.
        assert!(
            span.offset >= last_end,
            "span overlaps previous: offset={}, last_end={}",
            span.offset,
            last_end
        );
        last_end = span.offset + span.length;
        // Payload is inside [offset, offset + length).
        let payload_start_offset =
            (span.payload.as_ptr() as usize).wrapping_sub(data.as_ptr() as usize);
        if !span.payload.is_empty() {
            assert!(
                payload_start_offset >= span.offset && payload_start_offset < span.offset + span.length,
                "payload pointer not inside span"
            );
            assert!(
                payload_start_offset + span.payload.len() <= span.offset + span.length,
                "payload extends past span"
            );
        }
    }

    // 2. primary_bounds.
    let primary = primary_bounds(data);
    if let Some(ref r) = primary {
        assert!(r.start < r.end, "primary_bounds returned empty range");
        assert!(r.end <= data.len(), "primary_bounds exceeds buffer");
        // Must start at 0 (primary is always the first image).
        assert_eq!(r.start, 0, "primary_bounds must start at 0");
    }

    // 3. find_jpeg_boundaries.
    let all = find_jpeg_boundaries(data);
    for r in &all {
        assert!(r.start < r.end, "find_jpeg_boundaries returned empty range");
        assert!(r.end <= data.len(), "find_jpeg_boundaries exceeds buffer");
    }
    for pair in all.windows(2) {
        assert!(
            pair[0].end <= pair[1].start,
            "find_jpeg_boundaries ranges overlap: {:?} vs {:?}",
            pair[0],
            pair[1]
        );
    }

    // 4. Agreement: if primary is found, it matches the first
    //    find_jpeg_boundaries range.
    if let (Some(p), Some(first)) = (primary, all.first()) {
        assert_eq!(p, *first, "primary_bounds disagrees with find_jpeg_boundaries first");
    }
});
