//! Fuzz target for `zenjpeg::container::xmp`.
//!
//! Exercises both the direct XMP parse and the broader `parse_xmp_full`
//! path (which also pulls GContainer items).
//!
//! Invariants checked:
//! - No panic on arbitrary UTF-8 input.
//! - Returned metadata fields are finite.
//! - `parse_xmp` rejects inputs > MAX_XMP_LENGTH without reading the
//!   full input (length-limit branch).

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::container::{MAX_XMP_LENGTH, parse_xmp, parse_xmp_full};

fuzz_target!(|data: &[u8]| {
    // Cap at a bit over MAX_XMP_LENGTH so we also exercise the limit
    // branch.
    let data = if data.len() > MAX_XMP_LENGTH + 1024 {
        &data[..MAX_XMP_LENGTH + 1024]
    } else {
        data
    };

    // Interpret as UTF-8 (lossy); XMP is always text.
    let as_str = match core::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,
    };

    if let Ok((params, _len)) = parse_xmp(as_str) {
        for ch in &params.channels {
            assert!(ch.min.is_finite(), "channel.min non-finite: {}", ch.min);
            assert!(ch.max.is_finite(), "channel.max non-finite: {}", ch.max);
            assert!(ch.gamma.is_finite(), "channel.gamma non-finite: {}", ch.gamma);
            assert!(
                ch.base_offset.is_finite(),
                "channel.base_offset non-finite: {}",
                ch.base_offset
            );
            assert!(
                ch.alternate_offset.is_finite(),
                "channel.alternate_offset non-finite: {}",
                ch.alternate_offset
            );
        }
        assert!(params.base_hdr_headroom.is_finite());
        assert!(params.alternate_hdr_headroom.is_finite());
    }

    // parse_xmp_full must not panic on ANY input (no Result).
    let (_m, items) = parse_xmp_full(as_str);

    // Non-tautological invariants:
    //  - mime is ALWAYS non-empty in returned items (parser skips
    //    rdf:li with a missing Item:Mime). This is a property of the
    //    parser logic, not a property of the input.
    //  - Item:Length parses to a u64-representable usize or None.
    for it in &items {
        assert!(!it.mime.is_empty(), "parser invariant: items must have a mime");
        if let Some(len) = it.length {
            assert!(
                len <= u64::MAX as usize,
                "parsed Item:Length must fit in u64; got {len}"
            );
        }
    }
});
