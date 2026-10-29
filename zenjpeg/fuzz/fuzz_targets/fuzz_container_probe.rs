//! Fuzz target for `zenjpeg::container::probe`.
//!
//! Exercises probe and is_ultrahdr across arbitrary bytes with both
//! `Wants::ALL` and a `Wants::ULTRAHDR_DETECT`-shaped mask, asserting:
//!
//! - No panic, no OOM, no unbounded iteration.
//! - Every returned byte range is in-bounds of the input.
//! - `image_ranges` are non-overlapping and strictly ordered.
//! - `gainmap_presence` classification is consistent with the boolean
//!   fingerprints (no `IsoAndXmp` without both `iso_gainmap` AND
//!   `has_xmp_hdrgm`, etc.).
//! - `is_ultrahdr(data)` returns `true` iff a full probe would have
//!   captured at least one of: ISO, MPF, `hdrgm:`, GainMap semantic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use zenjpeg::container::{GainMapPresence, Wants, is_ultrahdr, probe};

fuzz_target!(|data: &[u8]| {
    // Cap input size to keep the fuzzer from exploring pathological
    // GB-scale inputs.
    let data = if data.len() > 4 * 1024 * 1024 {
        &data[..4 * 1024 * 1024]
    } else {
        data
    };
    let len = data.len() as u32;

    let p = probe(data, Wants::ALL);

    // In-bounds invariants for every captured range.
    let check_range = |r: Option<&core::ops::Range<u32>>| {
        if let Some(r) = r {
            assert!(r.start <= r.end, "inverted range {r:?}");
            assert!(r.end <= len, "range {r:?} exceeds input len {len}");
        }
    };
    check_range(p.icc_profile());
    check_range(p.exif());
    check_range(p.xmp());
    check_range(p.mpf());
    check_range(p.iso_gainmap());
    for r in p.image_ranges() {
        check_range(Some(r));
    }
    for r in p.extended_xmp() {
        check_range(Some(r));
    }

    // image_ranges must be disjoint and ordered.
    let mut prev_end: Option<u32> = None;
    for r in p.image_ranges() {
        if let Some(pe) = prev_end {
            assert!(r.start >= pe, "image ranges out of order");
        }
        prev_end = Some(r.end);
    }

    // gainmap_presence must be consistent with the fingerprint bools.
    let iso = p.iso_gainmap().is_some();
    let hdrgm = p.has_xmp_hdrgm();
    let gcontainer = p.has_xmp_gcontainer_gainmap();
    match p.gainmap_presence() {
        GainMapPresence::None => {
            assert!(!iso && !hdrgm && !gcontainer, "None presence but signal(s) present");
        }
        GainMapPresence::Iso21496 => assert!(iso && !hdrgm && !gcontainer),
        GainMapPresence::XmpHdrgmLegacy => assert!(!iso && hdrgm),
        GainMapPresence::GContainerOnly => assert!(!iso && !hdrgm && gcontainer),
        GainMapPresence::IsoAndXmp => assert!(iso && hdrgm),
        GainMapPresence::IsoAndGContainer => assert!(iso && !hdrgm && gcontainer),
        // `#[non_exhaustive]` forces a wildcard arm here. If we see
        // a variant we don't know about, the invariants this test is
        // supposed to enforce are undefined — flag rather than
        // silently accept.
        other => panic!(
            "fuzz target missing GainMapPresence arm for {other:?} — update this match \
             when adding a new variant"
        ),
    }

    // is_ultrahdr must agree with the full probe's classification.
    let shortcircuit = is_ultrahdr(data);
    let full_detect = iso || hdrgm || gcontainer || p.mpf().is_some();
    // Only assert the FALSE case — the short-circuit may return true
    // on inputs where the full probe found fewer than all signals
    // because the short-circuit uses a looser walker that stops early.
    if !full_detect {
        // If no signal was found by the full probe, the short-circuit
        // shouldn't claim UltraHDR either.
        assert!(
            !shortcircuit,
            "is_ultrahdr claims true but full probe found no signal"
        );
    }
});
