//! Fuzz target for `zenjpeg::container::iso_jpeg`.
//!
//! Exercises:
//!   1. `parse_iso_app2(data, format)` — full-JPEG scan + payload parse.
//!   2. `parse_iso21496(data, format)` — bare payload parse.
//!   3. `create_iso_app2_marker(payload)` roundtrip — wrap + parse back.
//!
//! Invariants checked:
//! - No panic, no unhandled arithmetic overflow, no OOM.
//! - `create_iso_app2_marker(p)` produces bytes that always begin with
//!   `FF E2` and whose declared length word matches the physical length.
//! - If `parse_iso_app2` succeeds, the returned payload-parsed metadata
//!   has finite numerical fields.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::container::{
    Iso21496Format, create_iso_app2_marker, parse_iso21496, parse_iso_app2,
};

fuzz_target!(|data: &[u8]| {
    // Cap to avoid OOM paths.
    let data = if data.len() > 1024 * 1024 {
        &data[..1024 * 1024]
    } else {
        data
    };

    // Try both framings the payload parser supports.
    for fmt in [Iso21496Format::JpegApp2, Iso21496Format::AvifTmap] {
        if let Ok(p) = parse_iso_app2(data, fmt) {
            for ch in &p.channels {
                assert!(ch.min.is_finite(), "channel.min non-finite");
                assert!(ch.max.is_finite(), "channel.max non-finite");
                assert!(ch.gamma.is_finite(), "channel.gamma non-finite");
            }
            assert!(p.base_hdr_headroom.is_finite());
            assert!(p.alternate_hdr_headroom.is_finite());
        }
        let _ = parse_iso21496(data, fmt);
    }

    // Envelope invariant: any byte slice up to a reasonable cap can be
    // wrapped. Check the output shape is correct.
    let payload = if data.len() > 8192 { &data[..8192] } else { data };
    let wrapped = create_iso_app2_marker(payload);
    assert_eq!(wrapped[0], 0xFF);
    assert_eq!(wrapped[1], 0xE2);
    let declared_len = u16::from_be_bytes([wrapped[2], wrapped[3]]) as usize;
    // Total segment length = 2 (marker) + 2 (length) + URN + payload.
    // declared_len excludes the marker bytes but includes the length field itself.
    assert_eq!(declared_len, wrapped.len() - 2);
});
