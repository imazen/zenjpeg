//! The `container::metadata` invariants, shared by the
//! `fuzz_container_metadata` target and `tests/fuzz_regression.rs` (both
//! include this file with `#[path]`), so a committed seed replays exactly
//! what the fuzzer asserts.
//!
//! [`check_all`] walks the input with the default limits and with tight ones
//! (an item limit of 1..=16 taken from the input length, so every small limit
//! is reached as inputs grow and shrink, and a 4 KiB reassembly cap), and
//! asserts:
//!
//! - No panic, no OOM, no unbounded work: the item count stays within
//!   `max_items` and every reassembled payload within `max_reassembled_bytes`.
//! - Items tile the input exactly; each payload lies inside its item; an
//!   item with a marker starts with `FF <marker>`; a `LimitExceeded` item can
//!   only be last.
//! - `primary_image_end()` is the end of an EOI item; everything after it is
//!   a trailer kind, and nothing before it is.
//! - Every `EmbeddedJpeg` payload walks on its own, without unparsed bytes,
//!   to an EOI at its end.
//! - Field accessors and the component/record/resource iterators terminate
//!   and stay within their segment; a reassembled ICC profile is never empty.
//! - No accessor reports `ItemLimitReached` unless the walk ended in a
//!   `LimitExceeded` item, and a borrowed result lies inside the input.
//! - Under the tight limits each accessor returns what it returns under the
//!   default ones, `ItemLimitReached` (only after a `LimitExceeded` item), or
//!   `TooLarge`: a cut never turns a result into a silent miss.

use std::borrow::Cow;

use zenjpeg::container::metadata::{
    ItemKind, JpegMetadata, MetadataError, MetadataLimits, Unparsed,
};

/// Every check, with the default limits and with tight ones.
pub fn check_all(data: &[u8]) {
    check(data, MetadataLimits::default(), true);
    let tight = MetadataLimits::default()
        .with_max_items(data.len() % 16 + 1)
        .with_max_reassembled_bytes(4096);
    check(data, tight, false);
    differential(data, tight);
}

/// The tight-limit result of each accessor against the default-limit one.
fn differential(data: &[u8], tight: MetadataLimits) {
    let full = JpegMetadata::parse(data).expect("parse");
    let cut = JpegMetadata::parse_with_limits(data, tight).expect("parse");
    let limited = cut
        .items()
        .last()
        .is_some_and(|i| matches!(i.kind, ItemKind::Unparsed(Unparsed::LimitExceeded)));
    fn same_or_cut<T: PartialEq + std::fmt::Debug>(
        what: &str,
        full: Result<T, MetadataError>,
        cut: Result<T, MetadataError>,
        limited: bool,
    ) {
        match &cut {
            r if *r == full => {}
            Err(MetadataError::ItemLimitReached) => {
                assert!(
                    limited,
                    "{what}: ItemLimitReached without a LimitExceeded item"
                );
            }
            Err(MetadataError::TooLarge { .. }) => {}
            other => panic!("{what}: tight limits gave {other:?}, default {full:?}"),
        }
    }
    let owned = |r: Result<Option<&[u8]>, MetadataError>| r.map(|o| o.map(<[u8]>::to_vec));
    same_or_cut("exif", owned(full.exif()), owned(cut.exif()), limited);
    same_or_cut("xmp", owned(full.xmp()), owned(cut.xmp()), limited);
    let cow = |r: Result<Option<Cow<'_, [u8]>>, MetadataError>| r.map(|o| o.map(Cow::into_owned));
    same_or_cut(
        "extended_xmp",
        cow(full.extended_xmp()),
        cow(cut.extended_xmp()),
        limited,
    );
    same_or_cut(
        "icc_profile",
        cow(full.icc_profile()),
        cow(cut.icc_profile()),
        limited,
    );
    let irb = |m: &JpegMetadata<'_>| m.photoshop().map(|o| o.map(|p| p.data().to_vec()));
    same_or_cut("photoshop", irb(&full), irb(&cut), limited);
    same_or_cut("mpf", full.mpf(), cut.mpf(), limited);
}

fn is_trailer(kind: &ItemKind<'_>) -> bool {
    matches!(
        kind,
        ItemKind::EmbeddedJpeg
            | ItemKind::Afcp
            | ItemKind::CanonVrd
            | ItemKind::FotoStation(_)
            | ItemKind::PhotoMechanic
            | ItemKind::Mie
            | ItemKind::Samsung
            | ItemKind::UnknownTrailer
    )
}

fn check(data: &[u8], limits: MetadataLimits, recurse: bool) {
    let meta = JpegMetadata::parse_with_limits(data, limits)
        .expect("parsing fails only when allocation fails");
    let items = meta.items();
    assert!(
        items.len() <= limits.max_items().max(1),
        "item limit exceeded"
    );
    let base = data.as_ptr() as usize;
    let mut pos = 0;
    for (i, item) in items.iter().enumerate() {
        assert_eq!(item.offset, pos, "items must tile the input");
        assert!(item.length > 0, "empty item");
        let end = item.offset + item.length;
        assert!(end <= data.len(), "item past the input");
        if !item.payload.is_empty() {
            let p = item.payload.as_ptr() as usize;
            assert!(p >= base + item.offset && p + item.payload.len() <= base + end);
        }
        if let Some(m) = item.marker {
            assert_eq!((data[item.offset], data[item.offset + 1]), (0xFF, m));
        }
        if matches!(item.kind, ItemKind::Unparsed(Unparsed::LimitExceeded)) {
            assert_eq!(i, items.len() - 1, "LimitExceeded must be last");
        }
        exercise_fields(&item.kind);
        if recurse && matches!(item.kind, ItemKind::EmbeddedJpeg) {
            let inner = JpegMetadata::parse(item.payload).expect("parse embedded");
            assert_eq!(inner.primary_image_end(), Some(item.payload.len()));
            assert!(
                !inner
                    .items()
                    .iter()
                    .any(|i| matches!(i.kind, ItemKind::Unparsed(_))),
                "an embedded image walks without unparsed bytes"
            );
            check(item.payload, limits, false);
        }
        pos = end;
    }
    assert_eq!(pos, data.len(), "items must cover the input");

    let primary_end = meta.primary_image_end();
    if let Some(e) = primary_end {
        assert!(
            items
                .iter()
                .any(|i| i.kind == ItemKind::Eoi && i.offset + i.length == e)
        );
    }
    for item in items {
        let after_eoi = primary_end.is_some_and(|e| item.offset >= e);
        if after_eoi {
            assert!(
                is_trailer(&item.kind)
                    || matches!(
                        item.kind,
                        ItemKind::Padding | ItemKind::Unparsed(Unparsed::LimitExceeded)
                    ),
                "{:?} after EOI",
                item.kind
            );
        } else {
            assert!(!is_trailer(&item.kind), "{:?} before EOI", item.kind);
        }
    }

    let cap = limits.max_reassembled_bytes();
    let limited = items
        .last()
        .is_some_and(|i| matches!(i.kind, ItemKind::Unparsed(Unparsed::LimitExceeded)));
    let check_err = |e: &MetadataError| {
        assert!(
            limited || *e != MetadataError::ItemLimitReached,
            "ItemLimitReached without a LimitExceeded item"
        );
    };
    // An empty slice may be a static `&[]`; only non-empty ones are checked.
    let inside = |bytes: &[u8]| {
        if bytes.is_empty() {
            return;
        }
        let p = bytes.as_ptr() as usize;
        assert!(p >= base && p + bytes.len() <= base + data.len());
    };
    match meta.icc_profile() {
        Ok(Some(icc)) => {
            assert!(!icc.is_empty() && icc.len() <= cap);
            if let Cow::Borrowed(b) = icc {
                inside(b);
            }
        }
        Ok(None) => {}
        Err(e) => check_err(&e),
    }
    match meta.extended_xmp() {
        Ok(Some(xmp)) => {
            assert!(xmp.len() <= cap);
            if let Cow::Borrowed(b) = xmp {
                inside(b);
            }
        }
        Ok(None) => {}
        Err(e) => check_err(&e),
    }
    match meta.photoshop() {
        Ok(Some(irb)) => {
            assert!(irb.data().len() <= cap);
            for r in irb.resources() {
                assert!(r.data.len() + r.name.len() <= irb.data().len());
            }
            let _ = irb.iptc();
            if let Cow::Borrowed(b) = irb.into_data() {
                inside(b);
            }
        }
        Ok(None) => {}
        Err(e) => check_err(&e),
    }
    match meta.mpf() {
        Ok(Some(entries)) => assert!(!entries.is_empty()),
        Ok(None) => {}
        Err(e) => check_err(&e),
    }
    for single in [meta.exif(), meta.xmp()] {
        match single {
            Ok(Some(b)) => inside(b),
            Ok(None) => {}
            Err(e) => check_err(&e),
        }
    }
}

fn exercise_fields(kind: &ItemKind<'_>) {
    match kind {
        ItemKind::Sof(s) => {
            assert!(s.components().count() <= usize::from(s.num_components));
            let _ = s.ycbcr_subsampling();
        }
        ItemKind::Sos(s) => {
            let _ = s.components().count();
        }
        ItemKind::Jfif(j) => {
            let declared = 3 * usize::from(j.thumbnail_width) * usize::from(j.thumbnail_height);
            assert!(j.thumbnail.len() <= declared);
        }
        ItemKind::Ducky(d) => {
            let _ = d.records().count();
            let _ = (d.quality(), d.comment().map(|c| c.chars().count()));
            let _ = d.copyright().map(|c| c.chars().count());
        }
        ItemKind::PhotoStudioComment(t) => {
            assert!(t.chars().count() <= t.bytes().len() / 2);
        }
        ItemKind::Eppim(e) => {
            if let Some(version) = e.print_im_version() {
                assert_eq!(version.len(), 4);
            }
        }
        _ => {}
    }
}
