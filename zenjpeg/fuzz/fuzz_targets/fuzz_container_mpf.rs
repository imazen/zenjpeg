//! Fuzz target for `zenjpeg::container::mpf`.
//!
//! Exercises both entry points:
//!   1. `parse_mpf(data)` — full-JPEG scan + parse.
//!   2. `parse_mpf_segment(payload, tiff_pos)` — direct parse of a
//!      TIFF-structured payload.
//!
//! Invariants checked on every fuzz input:
//!
//! - No panic, no unhandled arithmetic overflow, no OOM.
//! - Entry count never exceeds the hard cap (1000).
//! - Every returned [`MpfEntry`] has `offset + size <= usize::MAX`.
//! - `parse_mpf` never returns entries whose absolute offsets overflow.
//! - Emission via `create_mpf_header(p, g, off)` followed by
//!   `parse_mpf_segment` of the result always yields exactly 2 entries
//!   with the types and sizes we passed in (roundtrip invariant).

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::container::{
    MpfEntry, create_mpf_header, parse_mpf, parse_mpf_segment,
};

fuzz_target!(|data: &[u8]| {
    // Cap input size to avoid pathological fuzzer paths.
    let data = if data.len() > 2 * 1024 * 1024 {
        &data[..2 * 1024 * 1024]
    } else {
        data
    };

    // 1. parse_mpf on arbitrary JPEG-shape bytes.
    if let Ok(entries) = parse_mpf(data) {
        validate_entries(&entries);
    }

    // 2. parse_mpf_segment on arbitrary bytes at an arbitrary tiff pos.
    // Use a bounded tiff_pos so we don't cause huge absolute offsets.
    let tiff_pos = data.len().saturating_mul(3);
    if let Ok(entries) = parse_mpf_segment(data, tiff_pos) {
        validate_entries(&entries);
    }

    // 3. Roundtrip invariant: build a header from any two u32 sizes
    // derivable from the input bytes, parse it back, assert recovery.
    if data.len() >= 8 {
        let primary = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let gainmap = u32::from_be_bytes([data[4], data[5], data[6], data[7]]) as usize;
        // Cap sizes so u32 arithmetic stays sane.
        let primary = primary.min(1_000_000);
        let gainmap = gainmap.min(1_000_000);
        let built = create_mpf_header(primary, gainmap, None);
        // Strip APP2 marker + length + MPF\0 (4 + 4 = 8 bytes).
        let tiff_start = 8;
        if built.len() > tiff_start {
            let parsed =
                parse_mpf_segment(&built[tiff_start..], tiff_start).expect("roundtrip parse");
            assert_eq!(parsed.len(), 2);
            assert_eq!(parsed[0].size, primary);
            assert_eq!(parsed[1].size, gainmap);
        }
    }
});

fn validate_entries(entries: &[MpfEntry]) {
    assert!(
        entries.len() <= 1000,
        "parse returned more than the hard cap: {}",
        entries.len()
    );
    for e in entries {
        // offset + size must not overflow usize.
        let _ = e.offset.checked_add(e.size).expect("offset+size overflow");
    }
}
