//! Unit tests for the segment model. Fixtures are built byte by byte so each
//! test states exactly which bytes it feeds the walker.

use super::*;
use alloc::vec;
use alloc::vec::Vec;

// ── Fixture builders ────────────────────────────────────────────────────

/// A marker segment: `FF m`, length word, payload.
fn seg(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, marker];
    v.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    v.extend_from_slice(payload);
    v
}

fn cat(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

const SOI: &[u8] = &[0xFF, 0xD8];
const EOI: &[u8] = &[0xFF, 0xD9];

/// SOF0 for a 16x8 three-component 4:2:0 frame.
fn sof0() -> Vec<u8> {
    seg(
        0xC0,
        &[8, 0, 8, 0, 16, 3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1],
    )
}

/// SOS for components 1..=3 followed by entropy data containing a stuffed
/// `FF 00` and a restart marker, which must stay inside the scan item.
fn sos_with_entropy() -> Vec<u8> {
    let mut v = seg(0xDA, &[3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);
    v.extend_from_slice(&[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD3, 0x56]);
    v
}

/// A minimal walkable JPEG with the given metadata segments after SOI.
fn jpeg_with(segments: &[Vec<u8>]) -> Vec<u8> {
    let mut v = SOI.to_vec();
    for s in segments {
        v.extend_from_slice(s);
    }
    v.extend_from_slice(&sof0());
    v.extend_from_slice(&sos_with_entropy());
    v.extend_from_slice(EOI);
    v
}

fn parse(data: &[u8]) -> JpegMetadata<'_> {
    let meta = JpegMetadata::parse(data).expect("parse");
    assert_tiles(&meta, data);
    meta
}

/// The invariant every walk must keep: items are contiguous, start at 0,
/// end at `data.len()`, and every payload lies inside its item.
fn assert_tiles(meta: &JpegMetadata<'_>, data: &[u8]) {
    let base = data.as_ptr() as usize;
    let mut pos = 0;
    for item in meta.items() {
        assert_eq!(item.offset, pos, "gap or overlap before {item:?}");
        assert!(item.length > 0, "empty item {item:?}");
        let p = item.payload.as_ptr() as usize;
        if !item.payload.is_empty() {
            assert!(
                p >= base + item.offset
                    && p + item.payload.len() <= base + item.offset + item.length,
                "payload outside its item: {item:?}"
            );
        }
        pos += item.length;
    }
    assert_eq!(pos, data.len(), "items do not cover the input");
}

fn kinds<'a>(meta: &JpegMetadata<'a>) -> Vec<ItemKind<'a>> {
    meta.items().iter().map(|i| i.kind.clone()).collect()
}

fn names(meta: &JpegMetadata<'_>) -> Vec<&'static str> {
    meta.items().iter().map(|i| i.kind.name()).collect()
}

/// The single item of the given name; panics if there is not exactly one.
fn only<'a>(meta: &JpegMetadata<'a>, name: &str) -> Item<'a> {
    let found: Vec<_> = meta
        .items()
        .iter()
        .filter(|i| i.kind.name() == name)
        .collect();
    assert_eq!(found.len(), 1, "expected one {name} in {:?}", names(meta));
    found[0].clone()
}

/// Identify one APP/COM segment in isolation.
fn identify_one(marker: u8, payload: &[u8]) -> ItemKind<'_> {
    identify::identify(marker, payload, &[])
}

// ── Walk structure ──────────────────────────────────────────────────────

#[test]
fn minimal_jpeg_walks_in_file_order() {
    let jfif = seg(0xE0, b"JFIF\0\x01\x02\x01\x00\x48\x00\x48\x00\x00");
    let dqt = seg(0xDB, &[0; 65]);
    let dht = seg(0xC4, &[0; 17 + 1]);
    let data = cat(&[SOI, &jfif, &dqt, &sof0(), &dht, &sos_with_entropy(), EOI]);
    let meta = parse(&data);
    assert_eq!(
        names(&meta),
        ["SOI", "JFIF", "DQT", "SOF", "DHT", "SOS", "EOI"]
    );
    assert_eq!(meta.primary_image_end(), Some(data.len()));
    let sos = only(&meta, "SOS");
    let ItemKind::Sos(scan) = sos.kind else {
        unreachable!()
    };
    // The stuffed byte and the restart marker stay inside the scan.
    assert_eq!(scan.entropy, &[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD3, 0x56]);
    assert_eq!(sos.payload.len(), 10);
    assert_eq!(sos.length, 2 + 2 + 10 + 7);
    assert_eq!(sos.marker, Some(0xDA));
}

#[test]
fn empty_input_has_no_items() {
    let meta = parse(&[]);
    assert!(meta.items().is_empty());
    assert_eq!(meta.primary_image_end(), None);
}

#[test]
fn input_without_soi_is_one_unparsed_item() {
    for data in [
        &b"GIF89a....."[..],
        &[0xFF][..],
        &[0xFF, 0xD9, 0xFF, 0xD8][..],
    ] {
        let meta = parse(data);
        assert_eq!(kinds(&meta), [ItemKind::Unparsed(Unparsed::NotJpeg)]);
        assert_eq!(meta.items()[0].payload, data);
        assert_eq!(meta.items()[0].marker, None);
    }
}

#[test]
fn fill_bytes_before_a_marker_are_padding() {
    let com = seg(0xFE, b"hi");
    let data = cat(&[SOI, &[0xFF, 0xFF, 0xFF], &com, EOI]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "Padding", "Comment", "EOI"]);
    let padding = &meta.items()[1];
    assert_eq!((padding.offset, padding.length), (2, 3));
    assert_eq!(meta.items()[2].offset, 5);
}

#[test]
fn extraneous_bytes_are_reported_and_the_walk_resyncs() {
    // Garbage with a stuffed FF 00 pair (libjpeg's next_marker discards it)
    // between two segments.
    let garbage: &[u8] = &[0x01, 0x02, 0xFF, 0x00, 0x03];
    let com = seg(0xFE, b"after");
    let data = cat(&[SOI, &seg(0xFE, b"before"), garbage, &com, EOI]);
    let meta = parse(&data);
    assert_eq!(
        names(&meta),
        ["SOI", "Comment", "Unparsed", "Comment", "EOI"]
    );
    assert_eq!(
        meta.items()[2].kind,
        ItemKind::Unparsed(Unparsed::ExtraneousBytes)
    );
    assert_eq!(meta.items()[2].payload, garbage);
    assert_eq!(meta.items()[3].payload, b"after");
    assert_eq!(meta.primary_image_end(), Some(data.len()));
}

#[test]
fn stuffed_zero_where_a_marker_belongs_is_extraneous() {
    // `FF 00` is never a marker; libjpeg's next_marker discards it.
    let data = cat(&[SOI, &[0xFF, 0x00, 0x00, 0x07], &seg(0xFE, b"ok"), EOI]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "Unparsed", "Comment", "EOI"]);
    assert_eq!(meta.items()[1].payload, &[0xFF, 0x00, 0x00, 0x07]);
    // With fill bytes before it, and at the very end.
    let data = cat(&[SOI, &[0xFF, 0xFF, 0x00]]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "Unparsed"]);
    assert_eq!(
        meta.items()[1].kind,
        ItemKind::Unparsed(Unparsed::ExtraneousBytes)
    );
}

#[test]
fn stray_entropy_data_with_restart_markers_is_one_item() {
    // Entropy-coded data with restart markers but no scan header: libjpeg's
    // read_markers skips the parameterless RSTn markers while it searches.
    let orphan: &[u8] = &[
        0x11, 0x22, 0xFF, 0xD0, 0x33, 0xFF, 0x00, 0x44, 0xFF, 0xD1, 0x55,
    ];
    let data = cat(&[SOI, &seg(0xFE, b"c"), orphan, &seg(0xFE, b"d"), EOI]);
    let meta = parse(&data);
    assert_eq!(
        names(&meta),
        ["SOI", "Comment", "Unparsed", "Comment", "EOI"]
    );
    assert_eq!(meta.items()[2].payload, orphan);
    // A restart marker right after a segment is still its own item.
    let data = cat(&[SOI, &[0xFF, 0xD3], &seg(0xFE, b"d"), EOI]);
    assert_eq!(names(&parse(&data)), ["SOI", "RST", "Comment", "EOI"]);
}

#[test]
fn extraneous_bytes_to_the_end_are_one_item() {
    let data = cat(&[SOI, &seg(0xFE, b"x"), &[1, 2, 3, 0xFF, 0x00]]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "Comment", "Unparsed"]);
    assert_eq!(meta.primary_image_end(), None);
}

#[test]
fn bad_length_is_reported_and_the_walk_resyncs() {
    // APP4 with length word 1, then three stray bytes, then a comment.
    let data = cat(&[
        SOI,
        &[0xFF, 0xE4, 0x00, 0x01],
        &[9, 9, 9],
        &seg(0xFE, b"ok"),
        EOI,
    ]);
    let meta = parse(&data);
    assert_eq!(
        kinds(&meta)[1..4],
        [
            ItemKind::Unparsed(Unparsed::BadLength),
            ItemKind::Unparsed(Unparsed::ExtraneousBytes),
            ItemKind::Comment,
        ]
    );
    let bad = &meta.items()[1];
    assert_eq!((bad.offset, bad.length, bad.marker), (2, 4, Some(0xE4)));
    assert!(bad.payload.is_empty());
}

#[test]
fn bad_length_then_restart_marker_walks_the_marker() {
    // The walk resumes right after the bad length word, so an RSTn there is
    // an item of its own rather than part of a stray-byte run.
    let data = cat(&[
        SOI,
        &[0xFF, 0xE4, 0x00, 0x01],
        &[0xFF, 0xD3],
        &seg(0xFE, b"ok"),
        EOI,
    ]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "Unparsed", "RST", "Comment", "EOI"]);
}

#[test]
fn item_limit_before_the_eoi_has_no_primary_end() {
    let data = jpeg_with(&[seg(0xFE, b"a")]);
    for max in 1..parse(&data).items().len() {
        let limits = MetadataLimits::default().with_max_items(max);
        let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
        assert_eq!(meta.primary_image_end(), None, "max_items {max}");
    }
    let limits = MetadataLimits::default().with_max_items(0);
    assert_eq!(limits.max_items(), 0);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(names(&meta), ["Unparsed"]);
}

#[test]
fn bad_length_after_fill_bytes_keeps_the_fill_as_padding() {
    let data = cat(&[SOI, &[0xFF, 0xFF, 0xE4, 0x00, 0x00], &seg(0xFE, b"ok"), EOI]);
    let meta = parse(&data);
    assert_eq!(
        names(&meta),
        ["SOI", "Padding", "Unparsed", "Comment", "EOI"]
    );
    assert_eq!(meta.items()[2].marker, Some(0xE4));
}

#[test]
fn truncated_segment_keeps_its_marker_and_partial_payload() {
    // APP1 declares 100 bytes, 6 are present.
    let data = cat(&[SOI, &[0xFF, 0xE1, 0x00, 100], b"Exif\0\0"]);
    let meta = parse(&data);
    let last = meta.items().last().unwrap();
    assert_eq!(last.kind, ItemKind::Unparsed(Unparsed::TruncatedSegment));
    assert_eq!(last.marker, Some(0xE1));
    assert_eq!(last.payload, b"Exif\0\0");
    assert_eq!(meta.primary_image_end(), None);
}

#[test]
fn cut_marker_and_cut_length_word_are_truncated() {
    for tail in [
        &[0xFF][..],
        &[0xFF, 0xFF][..],
        &[0xFF, 0xE1][..],
        &[0xFF, 0xE1, 0x00][..],
    ] {
        let data = cat(&[SOI, &seg(0xFE, b"c"), tail]);
        let meta = parse(&data);
        let last = meta.items().last().unwrap();
        assert_eq!(
            last.kind,
            ItemKind::Unparsed(Unparsed::TruncatedSegment),
            "tail {tail:x?}"
        );
    }
    // A single stray non-FF byte at the end is extraneous, not truncation.
    let data = cat(&[SOI, &seg(0xFE, b"c"), &[0x42]]);
    let meta = parse(&data);
    assert_eq!(
        meta.items().last().unwrap().kind,
        ItemKind::Unparsed(Unparsed::ExtraneousBytes)
    );
}

#[test]
fn scan_running_to_the_end_without_eoi() {
    let mut data = cat(&[SOI, &sof0(), &sos_with_entropy()]);
    data.extend_from_slice(&[1, 2, 3]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "SOF", "SOS"]);
    assert_eq!(meta.primary_image_end(), None);
}

#[test]
fn standalone_markers_outside_a_scan() {
    let data = cat(&[SOI, &[0xFF, 0xD5], &[0xFF, 0x01], EOI]);
    let meta = parse(&data);
    assert_eq!(names(&meta), ["SOI", "RST", "Unknown", "EOI"]);
    assert_eq!(meta.items()[1].marker, Some(0xD5));
    assert_eq!(meta.items()[2].marker, Some(0x01));
    assert_eq!(meta.items()[2].length, 2);
}

/// Pins the behavior the `Restart` and `Sos::entropy` docs describe: a
/// restart marker preceded by fill bytes ends the scan's entropy-coded data
/// (a decoder reads it as part of the scan). The module docs do not promise
/// this extent (an unusual layout): when the items are built from the decoder's walker
/// (the planned follow-up, `docs/METADATA.md` "Follow-up: one walker for
/// metadata and decoding"), this test is rewritten to the decoder's extent.
/// That rewrite is the planned change, not a relaxed test.
#[test]
fn restart_marker_after_fill_bytes_ends_the_entropy_data() {
    let mut scan = sos_with_entropy();
    let entropy_end = scan.len();
    scan.extend_from_slice(&[0xFF, 0xFF, 0xD0, 0x77]);
    let data = cat(&[SOI, &sof0(), &scan, EOI]);
    let meta = parse(&data);
    assert_tiles(&meta, &data);
    assert_eq!(
        names(&meta),
        ["SOI", "SOF", "SOS", "Padding", "RST", "Unparsed", "EOI"]
    );
    let ItemKind::Sos(sos) = meta.items()[2].kind else {
        panic!("not SOS: {:?}", meta.items()[2].kind)
    };
    // Marker, length word and the 10-byte scan header precede the data.
    assert_eq!(sos.entropy, &scan[14..entropy_end]);
}

#[test]
fn item_limit_puts_the_rest_in_one_item() {
    let data = jpeg_with(&[seg(0xFE, b"a"), seg(0xFE, b"b"), seg(0xFE, b"c")]);
    let all = parse(&data).items().len(); // SOI, 3 COM, SOF, SOS, EOI
    assert_eq!(all, 7);
    for max in 1..all {
        let limits = MetadataLimits {
            max_items: max,
            ..MetadataLimits::default()
        };
        let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
        assert_tiles(&meta, &data);
        assert_eq!(meta.items().len(), max, "max_items {max}");
        assert_eq!(
            meta.items().last().unwrap().kind,
            ItemKind::Unparsed(Unparsed::LimitExceeded)
        );
    }
    // Exactly enough room: no limit item.
    let mut limits = MetadataLimits {
        max_items: all,
        ..MetadataLimits::default()
    };
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(meta.items().len(), all);
    assert_eq!(meta.items().last().unwrap().kind, ItemKind::Eoi);
    // Zero is treated as one.
    limits.max_items = 0;
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(meta.items().len(), 1);
}

#[test]
fn item_limit_applies_to_trailers() {
    let mut data = jpeg_with(&[]);
    let primary_end = data.len();
    data.extend_from_slice(&jpeg_with(&[]));
    data.extend_from_slice(&[0x55; 8]);
    let meta = parse(&data);
    assert_eq!(names(&meta)[4..], ["EmbeddedJPEG", "UnknownTrailer"]);
    let limits = MetadataLimits {
        max_items: 5,
        ..MetadataLimits::default()
    };
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_tiles(&meta, &data);
    let last = &meta.items()[4];
    assert_eq!(last.kind, ItemKind::Unparsed(Unparsed::LimitExceeded));
    assert_eq!(last.offset, primary_end);
    assert_eq!(meta.primary_image_end(), Some(primary_end));
}

// ── Frame structure fields ──────────────────────────────────────────────

#[test]
fn sof_fields_and_subsampling() {
    let sof0 = sof0();
    let ItemKind::Sof(sof) = identify_one(0xC2, &sof0[4..]) else {
        panic!("not SOF")
    };
    assert_eq!(
        (
            sof.process,
            sof.precision,
            sof.height,
            sof.width,
            sof.num_components
        ),
        (2, 8, 8, 16, 3)
    );
    let comps: Vec<_> = sof
        .components()
        .map(|c| (c.id, c.h, c.v, c.quant_table))
        .collect();
    assert_eq!(comps, [(1, 2, 2, 0), (2, 1, 1, 1), (3, 1, 1, 1)]);
    assert_eq!(sof.ycbcr_subsampling(), Some((2, 2)));

    // 4:2:2 and 4:4:0.
    let s422 = [8, 0, 8, 0, 8, 3, 1, 0x21, 0, 2, 0x11, 1, 3, 0x11, 1];
    let ItemKind::Sof(sof) = identify_one(0xC0, &s422) else {
        panic!()
    };
    assert_eq!(sof.ycbcr_subsampling(), Some((2, 1)));
    // A fractional ratio (3 vs 2) has no whole-number subsampling.
    let s32 = [8, 0, 8, 0, 8, 3, 1, 0x31, 0, 2, 0x21, 1, 3, 0x21, 1];
    let ItemKind::Sof(sof) = identify_one(0xC0, &s32) else {
        panic!()
    };
    assert_eq!(sof.ycbcr_subsampling(), None);
    // Not three components.
    let gray = [8, 0, 8, 0, 8, 1, 1, 0x11, 0];
    let ItemKind::Sof(sof) = identify_one(0xC1, &gray) else {
        panic!()
    };
    assert_eq!((sof.process, sof.ycbcr_subsampling()), (1, None));
}

#[test]
fn sof_with_cut_component_table_keeps_whole_records() {
    // Declares 3 components, carries one and a half.
    let ItemKind::Sof(sof) = identify_one(0xC0, &[8, 0, 8, 0, 8, 3, 1, 0x22, 0, 2, 0x11]) else {
        panic!()
    };
    assert_eq!(sof.num_components, 3);
    assert_eq!(sof.components().count(), 1);
    assert_eq!(sof.ycbcr_subsampling(), None);
    // Shorter than the six fixed bytes: not a usable SOF.
    assert_eq!(identify_one(0xC0, &[8, 0, 8, 0, 8]), ItemKind::Unknown);
}

#[test]
fn sos_fields() {
    let header = [2, 1, 0x00, 2, 0x11, 1, 5, 0x21];
    let entropy = [0xAB, 0xCD];
    let ItemKind::Sos(sos) = identify::identify(0xDA, &header, &entropy) else {
        panic!()
    };
    assert_eq!(
        (
            sos.spectral_start,
            sos.spectral_end,
            sos.approx_high,
            sos.approx_low
        ),
        (1, 5, 2, 1)
    );
    let comps: Vec<_> = sos
        .components()
        .map(|c| (c.selector, c.dc_table, c.ac_table))
        .collect();
    assert_eq!(comps, [(1, 0, 0), (2, 1, 1)]);
    assert_eq!(sos.entropy, &entropy);
    // A header shorter than its component count requires.
    assert_eq!(identify::identify(0xDA, &[3, 1, 0], &[]), ItemKind::Unknown);
}

#[test]
fn dri_dnl_dht_dqt_dac() {
    assert_eq!(identify_one(0xDD, &[0x01, 0x40]), ItemKind::Dri(320));
    assert_eq!(identify_one(0xDC, &[0x02, 0x00]), ItemKind::Dnl(512));
    assert_eq!(identify_one(0xDD, &[0x01]), ItemKind::Unknown);
    assert_eq!(identify_one(0xC4, &[0]), ItemKind::Dht);
    assert_eq!(identify_one(0xDB, &[0]), ItemKind::Dqt);
    assert_eq!(identify_one(0xCC, &[0]), ItemKind::Dac);
    // JPG (0xC8), JPGn and EXP are not classified.
    assert_eq!(identify_one(0xC8, &[0]), ItemKind::Unknown);
    assert_eq!(identify_one(0xF3, &[0]), ItemKind::Unknown);
    assert_eq!(identify_one(0xDF, &[0]), ItemKind::Unknown);
}

// ── APP segment identification ──────────────────────────────────────────

#[test]
fn jfif_fields_and_thumbnail() {
    let payload = b"JFIF\0\x01\x02\x02\x00\x96\x00\x64\x01\x01\xAA\xBB\xCC";
    let ItemKind::Jfif(j) = identify_one(0xE0, payload) else {
        panic!()
    };
    assert_eq!((j.version_major, j.version_minor, j.units), (1, 2, 2));
    assert_eq!((j.x_density, j.y_density), (150, 100));
    assert_eq!((j.thumbnail_width, j.thumbnail_height), (1, 1));
    assert_eq!(j.thumbnail, &[0xAA, 0xBB, 0xCC]);
    // The thumbnail is the declared 3*w*h bytes; extra bytes are not part
    // of it, and a short segment keeps what it has (ExifTool ThumbnailTIFF).
    let extra = cat(&[payload, b"\xDD\xEE"]);
    let ItemKind::Jfif(j) = identify_one(0xE0, &extra) else {
        panic!()
    };
    assert_eq!(j.thumbnail, &[0xAA, 0xBB, 0xCC]);
    let ItemKind::Jfif(j) = identify_one(0xE0, &payload[..15]) else {
        panic!()
    };
    assert_eq!(j.thumbnail, &[0xAA]);
    let mut none = payload.to_vec();
    none[12] = 0;
    let ItemKind::Jfif(j) = identify_one(0xE0, &none) else {
        panic!()
    };
    assert!(j.thumbnail.is_empty());
    // One byte short of the fixed fields.
    assert_eq!(identify_one(0xE0, &payload[..13]), ItemKind::Unknown);
}

#[test]
fn app0_jfxx_ciff_avi1() {
    let ItemKind::Jfxx(x) = identify_one(0xE0, b"JFXX\0\x10\xFF\xD8") else {
        panic!()
    };
    assert_eq!((x.extension_code, x.data), (0x10, &[0xFF, 0xD8][..]));
    // ExifTool accepts only codes 0x10, 0x11, 0x13.
    assert_eq!(identify_one(0xE0, b"JFXX\0\x12abc"), ItemKind::Unknown);
    assert_eq!(
        identify_one(0xE0, b"II\x1a\0\0\0HEAPJPGMxx"),
        ItemKind::Ciff
    );
    assert_eq!(identify_one(0xE0, b"MM\0\0\0\x1aHEAPJPGM"), ItemKind::Ciff);
    assert_eq!(
        identify_one(0xE0, b"MM\0\0\0\x1aHEAPJPG"),
        ItemKind::Unknown
    );
    assert_eq!(
        identify_one(0xE0, b"AVI1\x01\0\0\0"),
        ItemKind::Avi1(Avi1 {
            interleaved_field: Some(1)
        })
    );
    assert_eq!(
        identify_one(0xE0, b"AVI1"),
        ItemKind::Avi1(Avi1 {
            interleaved_field: None
        })
    );
    assert_eq!(identify_one(0xE0, b"Ocad$Rev: 14797 $"), ItemKind::Unknown);
}

#[test]
fn app1_exif_and_xmp() {
    assert_eq!(
        identify_one(0xE1, b"Exif\0\0II*\0"),
        ItemKind::Exif(b"II*\0")
    );
    // The TIFF starts at byte 6 whatever byte 5 holds (ExifTool DirStart 6).
    assert_eq!(
        identify_one(0xE1, b"Exif\0\xFFMM\0*"),
        ItemKind::Exif(b"MM\0*")
    );
    assert_eq!(identify_one(0xE1, b"Exif\0\0"), ItemKind::Exif(b""));
    assert_eq!(identify_one(0xE1, b"Exif\0"), ItemKind::Exif(b""));
    // A writer that drops byte 5: the TIFF header starts at 5, not at 6.
    assert_eq!(identify_one(0xE1, b"Exif\0MM\0*"), ItemKind::Exif(b"MM\0*"));
    assert_eq!(identify_one(0xE1, b"Exif\0II*\0"), ItemKind::Exif(b"II*\0"));
    assert_eq!(
        identify_one(0xE1, b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta/>"),
        ItemKind::Xmp(b"<x:xmpmeta/>")
    );
    let mut ext = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    ext.extend_from_slice(&[b'A'; 32]);
    ext.extend_from_slice(&1000u32.to_be_bytes());
    ext.extend_from_slice(&40u32.to_be_bytes());
    ext.extend_from_slice(b"data");
    let ItemKind::ExtendedXmp(c) = identify_one(0xE1, &ext) else {
        panic!()
    };
    assert_eq!(
        (c.guid, c.full_length, c.offset, c.data),
        (&[b'A'; 32][..], 1000, 40, &b"data"[..])
    );
    // The 40-byte chunk header is mandatory.
    assert_eq!(identify_one(0xE1, &ext[..35 + 39]), ItemKind::Unknown);
    assert_eq!(identify_one(0xE1, b"QVCI\0..."), ItemKind::Unknown);
}

#[test]
fn app2_icc_fpxr_mpf_iso() {
    assert_eq!(
        identify_one(0xE2, b"ICC_PROFILE\0\x02\x03abc"),
        ItemKind::IccProfile(IccChunk {
            sequence: 2,
            count: 3,
            data: b"abc"
        })
    );
    assert_eq!(identify_one(0xE2, b"ICC_PROFILE\0\x01"), ItemKind::Unknown);
    assert_eq!(identify_one(0xE2, b"FPXR\0\x00\x01"), ItemKind::Fpxr);
    assert_eq!(identify_one(0xE2, b"MPF\0MM\0*"), ItemKind::Mpf(b"MM\0*"));
    let mut iso = zencodec::ISO_21496_1_URN.to_vec();
    iso.extend_from_slice(&[0, 0]);
    assert_eq!(
        identify_one(0xE2, &iso),
        ItemKind::GainMapIso21496(GainMapIso21496 { data: &[0, 0] })
    );
}

#[test]
fn app3_app5_kodak_and_ricoh() {
    for sig in [&b"Meta\0\0"[..], b"META\0\0", b"Exif\0\0"] {
        let payload = cat(&[sig, b"II*\0"]);
        assert_eq!(identify_one(0xE3, &payload), ItemKind::KodakMeta(b"II*\0"));
    }
    assert_eq!(identify_one(0xE3, b"Stim\0..."), ItemKind::Unknown);
    assert_eq!(
        identify_one(0xE5, b"RMETA\0II"),
        ItemKind::RicohRmeta(b"II")
    );
}

/// A TIFF structure whose IFD0 holds one entry: `tag`, UNDEFINED, `value`.
fn tiff_with(little_endian: bool, tag: u16, value: &[u8]) -> Vec<u8> {
    let w16 = |v: u16| {
        if little_endian {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        }
    };
    let w32 = |v: u32| {
        if little_endian {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        }
    };
    let mut t = if little_endian {
        b"II*\0".to_vec()
    } else {
        b"MM\0*".to_vec()
    };
    t.extend_from_slice(&w32(8));
    t.extend_from_slice(&w16(1));
    t.extend_from_slice(&w16(tag));
    t.extend_from_slice(&w16(7));
    t.extend_from_slice(&w32(value.len() as u32));
    if value.len() <= 4 {
        let mut inline = [0u8; 4];
        inline[..value.len()].copy_from_slice(value);
        t.extend_from_slice(&inline);
        t.extend_from_slice(&w32(0));
    } else {
        t.extend_from_slice(&w32(8 + 2 + 12 + 4));
        t.extend_from_slice(&w32(0)); // next IFD
        t.extend_from_slice(value);
    }
    t
}

#[test]
fn app6_eppim_print_im() {
    let print_im = b"PrintIM\x000300\0\0\0\0";
    for le in [true, false] {
        let payload = cat(&[b"EPPIM\0", &tiff_with(le, 0xC4A5, print_im)]);
        let ItemKind::Eppim(e) = identify_one(0xE6, &payload) else {
            panic!()
        };
        assert_eq!(e.print_im, Some(&print_im[..]));
        assert_eq!(e.print_im_version(), Some(&b"0300"[..]));
    }
    // Inline (<= 4 byte) values are read from the entry itself.
    let payload = cat(&[b"EPPIM\0", &tiff_with(true, 0xC4A5, b"abc")]);
    let ItemKind::Eppim(e) = identify_one(0xE6, &payload) else {
        panic!()
    };
    assert_eq!(
        (e.print_im, e.print_im_version()),
        (Some(&b"abc"[..]), None)
    );
    // Another tag, a bad TIFF header, and a value offset past the end.
    let other = cat(&[b"EPPIM\0", &tiff_with(true, 0x0100, print_im)]);
    let ItemKind::Eppim(e) = identify_one(0xE6, &other) else {
        panic!()
    };
    assert_eq!(e.print_im, None);
    let ItemKind::Eppim(e) = identify_one(0xE6, b"EPPIM\0XX*\0") else {
        panic!()
    };
    assert_eq!(e.print_im, None);
    // ExifTool does not check the TIFF magic number, but needs IFD0 at >= 8.
    let mut magic = tiff_with(true, 0xC4A5, print_im);
    magic[2] = 0x2B;
    let payload = cat(&[b"EPPIM\0", &magic]);
    let ItemKind::Eppim(e) = identify_one(0xE6, &payload) else {
        panic!()
    };
    assert_eq!(e.print_im, Some(&print_im[..]));
    // IFD0 at offset 3 overlaps the header. Read anyway, its count
    // (bytes 3..5, big-endian 0x0100) and second entry (at 17) would find
    // the tag; ExifTool's `DoProcessTIFF` rejects an IFD0 offset below 8.
    let mut low = b"MM\x00\x01\x00\x00\x00\x03".to_vec();
    low.resize(17, 0);
    low.extend_from_slice(&0xC4A5u16.to_be_bytes());
    low.extend_from_slice(&7u16.to_be_bytes());
    low.extend_from_slice(&(print_im.len() as u32).to_be_bytes());
    low.extend_from_slice(&29u32.to_be_bytes());
    low.extend_from_slice(print_im);
    let payload = cat(&[b"EPPIM\0", &low]);
    let ItemKind::Eppim(e) = identify_one(0xE6, &payload) else {
        panic!()
    };
    assert_eq!(e.print_im, None);
    let mut cut = cat(&[b"EPPIM\0", &tiff_with(false, 0xC4A5, print_im)]);
    cut.truncate(cut.len() - 1);
    let ItemKind::Eppim(e) = identify_one(0xE6, &cut) else {
        panic!()
    };
    assert_eq!(e.print_im, None);
}

#[test]
fn print_im_version_follows_exiftool_checks() {
    let version_of = |block: &[u8]| {
        let payload = cat(&[b"EPPIM\0", &tiff_with(true, 0xC4A5, block)]);
        let ItemKind::Eppim(e) = identify_one(0xE6, &payload) else {
            panic!()
        };
        e.print_im_version().map(<[u8]>::to_vec)
    };
    // ExifTool compares 7 bytes: the eighth need not be NUL.
    assert_eq!(version_of(b"PrintIM!0250\0\0\0\0"), Some(b"0250".to_vec()));
    // One record of 6 bytes, counted big-endian or little-endian.
    let one_be = b"PrintIM\x000300\0\0\0\x01abcdef";
    let one_le = b"PrintIM\x000300\0\0\x01\0abcdef";
    assert_eq!(version_of(one_be), Some(b"0300".to_vec()));
    assert_eq!(version_of(one_le), Some(b"0300".to_vec()));
    // Records past the block in both byte orders, too short, wrong magic.
    assert_eq!(version_of(b"PrintIM\x000300\0\0\x01\x01abcdef"), None);
    assert_eq!(version_of(b"PrintIM\x000300\0\0\0"), None);
    assert_eq!(version_of(b"PrintIX\x000300\0\0\0\0"), None);
}

#[test]
fn app8_spiff_both_layouts() {
    // T.84: version 1.0, profile 1, 3 components, 480 x 640, YCbCr (3),
    // 8 bits, JPEG (5), dpi (1), 72 x 96.
    let mut t84 = b"SPIFF\0\x01\x00\x01\x03".to_vec();
    t84.extend_from_slice(&480u32.to_be_bytes());
    t84.extend_from_slice(&640u32.to_be_bytes());
    t84.extend_from_slice(&[3, 8, 5, 1]);
    t84.extend_from_slice(&72u32.to_be_bytes());
    t84.extend_from_slice(&96u32.to_be_bytes());
    let expected = Spiff {
        version_major: 1,
        version_minor: 0,
        profile_id: 1,
        num_components: 3,
        height: 480,
        width: 640,
        color_space: 3,
        bits_per_sample: 8,
        compression: 5,
        resolution_units: 1,
        vertical_resolution: 72,
        horizontal_resolution: 96,
    };
    assert_eq!(identify_one(0xE8, &t84), ItemKind::Spiff(expected));
    // ExifTool's layout: two extra bytes after the component count.
    let mut exiftool = t84[..10].to_vec();
    exiftool.extend_from_slice(&[0xEE, 0xEE]);
    exiftool.extend_from_slice(&t84[10..]);
    assert_eq!(identify_one(0xE8, &exiftool), ItemKind::Spiff(expected));
    assert_eq!(identify_one(0xE8, &t84[..29]), ItemKind::Unknown);
}

#[test]
fn app10_photostudio_comment() {
    let ItemKind::PhotoStudioComment(text) = identify_one(0xEA, b"UNICODE\0\0H\0i\xD8\x3D\xDE\x00")
    else {
        panic!()
    };
    assert_eq!(text.chars().collect::<String>(), "Hi\u{1F600}");
    // Odd trailing byte and an unpaired surrogate.
    let ItemKind::PhotoStudioComment(text) = identify_one(0xEA, b"UNICODE\0\xD8\x00\0A\0") else {
        panic!()
    };
    assert_eq!(text.chars().collect::<String>(), "\u{FFFD}A");
    assert_eq!(
        identify_one(0xEA, b"AROT\0\0\0\0\0\0\0\0"),
        ItemKind::Unknown
    );
}

#[test]
fn app11_jumbf() {
    let mut p = b"JP".to_vec();
    p.extend_from_slice(&7u16.to_be_bytes());
    p.extend_from_slice(&1u32.to_be_bytes());
    p.extend_from_slice(&100u32.to_be_bytes());
    p.extend_from_slice(b"jumb");
    p.extend_from_slice(b"content");
    let ItemKind::Jumbf(j) = identify_one(0xEB, &p) else {
        panic!()
    };
    assert_eq!((j.instance, j.sequence, &j.box_type), (7, 1, b"jumb"));
    assert_eq!(&j.data[..8], &p[8..16]);
    assert_eq!(identify_one(0xEB, &p[..15]), ItemKind::Unknown);
    assert_eq!(identify_one(0xEB, b"HDR_RI ver=11;..."), ItemKind::Unknown);
}

fn ducky(records: &[(u16, &[u8])], terminate: bool) -> Vec<u8> {
    let mut p = b"Ducky".to_vec();
    for (tag, data) in records {
        p.extend_from_slice(&tag.to_be_bytes());
        p.extend_from_slice(&(data.len() as u16).to_be_bytes());
        p.extend_from_slice(data);
    }
    if terminate {
        p.extend_from_slice(&[0, 0]);
    }
    p
}

#[test]
fn app12_ducky_records() {
    let comment = [&3u32.to_be_bytes()[..], b"\0a\0b\0c"].concat();
    let copyright = [&1u32.to_be_bytes()[..], b"\0\xA9"].concat();
    let p = ducky(
        &[(1, &75u32.to_be_bytes()), (2, &comment), (3, &copyright)],
        true,
    );
    let ItemKind::Ducky(d) = identify_one(0xEC, &p) else {
        panic!()
    };
    assert_eq!(d.quality(), Some(75));
    assert_eq!(d.comment().unwrap().chars().collect::<String>(), "abc");
    assert_eq!(d.copyright().unwrap().chars().collect::<String>(), "\u{A9}");
    assert_eq!(d.records().map(|r| r.tag).collect::<Vec<_>>(), [1, 2, 3]);
    // A record that runs past the segment ends the records.
    let mut cut = ducky(&[(1, &50u32.to_be_bytes())], false);
    cut.extend_from_slice(&[0, 2, 0, 9, b'x']);
    let ItemKind::Ducky(d) = identify_one(0xEC, &cut) else {
        panic!()
    };
    assert_eq!(d.records().count(), 1);
    assert_eq!((d.quality(), d.comment()), (Some(50), None));
    // ExifTool reads past a zero tag; only fewer than 4 bytes ends the run.
    let p = ducky(&[(1, &75u32.to_be_bytes()), (0, &[]), (2, &comment)], true);
    let ItemKind::Ducky(d) = identify_one(0xEC, &p) else {
        panic!()
    };
    assert_eq!(d.records().map(|r| r.tag).collect::<Vec<_>>(), [1, 0, 2]);
    assert_eq!(d.comment().unwrap().chars().collect::<String>(), "abc");
    // Repeated tags: ExifTool gives all three Ducky tags priority 0, so the
    // first value of each stays.
    let later = [&1u32.to_be_bytes()[..], b"\0z"].concat();
    let p = ducky(
        &[
            (1, &75u32.to_be_bytes()),
            (2, &comment),
            (3, &copyright),
            (1, &10u32.to_be_bytes()),
            (2, &later),
            (3, &later),
        ],
        true,
    );
    let ItemKind::Ducky(d) = identify_one(0xEC, &p) else {
        panic!()
    };
    assert_eq!(d.quality(), Some(75));
    assert_eq!(d.comment().unwrap().chars().collect::<String>(), "abc");
    assert_eq!(d.copyright().unwrap().chars().collect::<String>(), "\u{A9}");
}

#[test]
fn app12_picture_info() {
    assert_eq!(
        identify_one(0xEC, b"[picture info]\r\nTimeDate=966665853\r\n"),
        ItemKind::PictureInfo
    );
    assert_eq!(identify_one(0xEC, b"Type=2\0"), ItemKind::PictureInfo);
    assert_eq!(
        identify_one(0xEC, b"\x00\x01\x02 = \x03"),
        ItemKind::Unknown
    );
    // A bracket pair split across lines, and a tag with no printable value.
    assert_eq!(identify_one(0xEC, b"[a\nb]"), ItemKind::Unknown);
    assert_eq!(identify_one(0xEC, b"tag=\x01"), ItemKind::Unknown);
    // Expected values from Perl running ExifTool's regex on each input.
    let cases: [(&[u8], bool); 14] = [
        (b"a=b\x01", false),
        (b"[x=y]\x80", true),
        (b"a=[\x80]", true),
        (b"a=[]\x80", true),
        (b"a=b[\x80", true),
        (b"a=bc=d\x80", true),
        (b"a=b\tc=\x80", true),
        (b"a=b\x0b\r\x80", true),
        (b"a=b\x80", false),
        (b"a=b\x80c", false),
        (b"a=b c\x80", false),
        (b"x==\x80", false),
        (b"a=b=\x80", false),
        (b"a=b \x0b\x80", false),
    ];
    for (payload, expected) in cases {
        let kind = identify_one(0xEC, payload);
        assert_eq!(kind == ItemKind::PictureInfo, expected, "{payload:?}");
    }
}

#[test]
fn app13_app14_app15() {
    assert_eq!(
        identify_one(0xED, b"Photoshop 3.0\08BIM"),
        ItemKind::Photoshop(b"8BIM")
    );
    assert_eq!(
        identify_one(0xED, b"Adobe_CM\x00\x02"),
        ItemKind::AdobeCm(2)
    );
    assert_eq!(identify_one(0xED, b"Adobe_CM\x00"), ItemKind::Unknown);
    assert_eq!(
        identify_one(0xEE, b"Adobe\x00\x64\x80\x00\x00\x01\x02"),
        ItemKind::Adobe(Adobe {
            dct_encode_version: 100,
            flags0: 0x8000,
            flags1: 1,
            color_transform: 2,
        })
    );
    assert_eq!(
        identify_one(0xEE, b"Adobe\x00\x64\x80\x00\x00\x01"),
        ItemKind::Unknown
    );
    assert_eq!(identify_one(0xEF, b"Q 85"), ItemKind::GraphicConverter(85));
    assert_eq!(identify_one(0xEF, b"Q100"), ItemKind::GraphicConverter(100));
    assert_eq!(identify_one(0xEF, b"Q9 x"), ItemKind::GraphicConverter(9));
    assert_eq!(identify_one(0xEF, b"Qabc"), ItemKind::Unknown);
    assert_eq!(identify_one(0xEF, b"Q  85"), ItemKind::Unknown);
    assert_eq!(identify_one(0xFE, b"\xFFanything"), ItemKind::Comment);
    assert_eq!(identify_one(0xE4, b"SCALADO\0"), ItemKind::Unknown);
}

#[test]
fn every_kind_has_a_distinct_name() {
    let all = [
        ItemKind::Soi,
        ItemKind::Eoi,
        ItemKind::Restart,
        ItemKind::Dht,
        ItemKind::Dqt,
        ItemKind::Dac,
        ItemKind::Dri(0),
        ItemKind::Dnl(0),
        ItemKind::Ciff,
        ItemKind::Avi1(Avi1 {
            interleaved_field: None,
        }),
        ItemKind::Exif(&[]),
        ItemKind::Xmp(&[]),
        ItemKind::Fpxr,
        ItemKind::Mpf(&[]),
        ItemKind::GainMapIso21496(GainMapIso21496 { data: &[] }),
        ItemKind::KodakMeta(&[]),
        ItemKind::RicohRmeta(&[]),
        ItemKind::PictureInfo,
        ItemKind::Photoshop(&[]),
        ItemKind::AdobeCm(0),
        ItemKind::GraphicConverter(0),
        ItemKind::Comment,
        ItemKind::Unknown,
        ItemKind::EmbeddedJpeg,
        ItemKind::Afcp,
        ItemKind::CanonVrd,
        ItemKind::FotoStation(0),
        ItemKind::PhotoMechanic,
        ItemKind::Mie,
        ItemKind::Samsung,
        ItemKind::Padding,
        ItemKind::UnknownTrailer,
        ItemKind::Unparsed(Unparsed::NotJpeg),
    ];
    let ext_xmp = cat(&[
        b"http://ns.adobe.com/xmp/extension/\0",
        &[b'A'; 32],
        &[0; 8],
    ]);
    let sos = [1, 1, 0, 0, 63, 0];
    let parsed = [
        identify_one(0xC0, &[8, 0, 1, 0, 1, 1, 1, 0x11, 0]),
        identify::identify(0xDA, &sos, &[]),
        identify_one(0xE0, b"JFIF\0\x01\x02\0\0\x01\0\x01\0\0"),
        identify_one(0xE0, b"JFXX\0\x10"),
        identify_one(0xE1, &ext_xmp),
        identify_one(0xE2, b"ICC_PROFILE\0\x01\x01"),
        identify_one(0xE6, b"EPPIM\0"),
        identify_one(
            0xE8,
            b"SPIFF\0\x01\0\x01\x03\0\0\0\x08\0\0\0\x10\x03\x08\x05\x01\0\0\0\x48\0\0\0\x48",
        ),
        identify_one(0xEA, b"UNICODE\0"),
        identify_one(0xEB, b"JP\0\x01\0\0\0\x01\0\0\0\x10jumb"),
        identify_one(0xEC, b"Ducky"),
        identify_one(0xEE, b"Adobe\0\x64\0\0\0\0\x01"),
    ];
    let all: Vec<&ItemKind<'_>> = all.iter().chain(parsed.iter()).collect();
    // `variant_index` matches without a wildcard, so a new variant fails to
    // compile until it is listed there, and the list must reach all of them.
    let mut covered: Vec<usize> = all.iter().map(|k| variant_index(k)).collect();
    covered.sort_unstable();
    covered.dedup();
    assert_eq!(covered, (0..VARIANTS).collect::<Vec<_>>());
    let mut seen: Vec<&str> = all.iter().map(|k| k.name()).collect();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!((seen.len(), before), (VARIANTS, VARIANTS));
}

const VARIANTS: usize = 45;

fn variant_index(kind: &ItemKind<'_>) -> usize {
    match kind {
        ItemKind::Soi => 0,
        ItemKind::Eoi => 1,
        ItemKind::Restart => 2,
        ItemKind::Sof(_) => 3,
        ItemKind::Dht => 4,
        ItemKind::Dqt => 5,
        ItemKind::Dac => 6,
        ItemKind::Sos(_) => 7,
        ItemKind::Dri(_) => 8,
        ItemKind::Dnl(_) => 9,
        ItemKind::Jfif(_) => 10,
        ItemKind::Jfxx(_) => 11,
        ItemKind::Ciff => 12,
        ItemKind::Avi1(_) => 13,
        ItemKind::Exif(_) => 14,
        ItemKind::Xmp(_) => 15,
        ItemKind::ExtendedXmp(_) => 16,
        ItemKind::IccProfile(_) => 17,
        ItemKind::Fpxr => 18,
        ItemKind::Mpf(_) => 19,
        ItemKind::GainMapIso21496(_) => 20,
        ItemKind::KodakMeta(_) => 21,
        ItemKind::RicohRmeta(_) => 22,
        ItemKind::Eppim(_) => 23,
        ItemKind::Spiff(_) => 24,
        ItemKind::PhotoStudioComment(_) => 25,
        ItemKind::Jumbf(_) => 26,
        ItemKind::Ducky(_) => 27,
        ItemKind::PictureInfo => 28,
        ItemKind::Photoshop(_) => 29,
        ItemKind::AdobeCm(_) => 30,
        ItemKind::Adobe(_) => 31,
        ItemKind::GraphicConverter(_) => 32,
        ItemKind::Comment => 33,
        ItemKind::Unknown => 34,
        ItemKind::EmbeddedJpeg => 35,
        ItemKind::Afcp => 36,
        ItemKind::CanonVrd => 37,
        ItemKind::FotoStation(_) => 38,
        ItemKind::PhotoMechanic => 39,
        ItemKind::Mie => 40,
        ItemKind::Samsung => 41,
        ItemKind::Padding => 42,
        ItemKind::UnknownTrailer => 43,
        ItemKind::Unparsed(_) => 44,
    }
}

/// The committed metadata seeds (`fuzz/make_metadata_seeds.py`) together
/// produce every item kind, so the fuzz-regression replay walks every
/// identification, field parser and trailer reader. `variant_index` has no
/// wildcard: a new kind fails to compile until it is listed there, then fails
/// here until a seed produces it.
#[test]
fn metadata_seeds_reach_every_item_kind() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/metadata");
    let mut covered = [false; VARIANTS];
    let mut reasons = Vec::new();
    let mut seeds = 0;
    for entry in std::fs::read_dir(&dir).expect("fuzz/seeds/metadata") {
        let data = std::fs::read(entry.expect("dir entry").path()).expect("read seed");
        let meta = parse(&data);
        assert_tiles(&meta, &data);
        for item in meta.items() {
            covered[variant_index(&item.kind)] = true;
            if let ItemKind::Unparsed(why) = item.kind {
                reasons.push(why);
            }
        }
        seeds += 1;
    }
    assert_eq!(seeds, 8, "fuzz/make_metadata_seeds.py writes 8 seeds");
    let missing: Vec<usize> = (0..VARIANTS).filter(|&i| !covered[i]).collect();
    assert!(
        missing.is_empty(),
        "no seed produces these `variant_index` kinds: {missing:?}"
    );
    for why in [
        Unparsed::ExtraneousBytes,
        Unparsed::BadLength,
        Unparsed::TruncatedSegment,
    ] {
        assert!(reasons.contains(&why), "no seed produces Unparsed({why:?})");
    }
}

// ── ICC reassembly ──────────────────────────────────────────────────────

fn icc_chunk(seq: u8, count: u8, data: &[u8]) -> Vec<u8> {
    seg(0xE2, &cat(&[b"ICC_PROFILE\0", &[seq, count], data]))
}

#[test]
fn icc_single_chunk_is_borrowed() {
    let data = jpeg_with(&[icc_chunk(1, 1, b"profile")]);
    let meta = parse(&data);
    let icc = meta.icc_profile().unwrap().unwrap();
    assert!(matches!(icc, Cow::Borrowed(_)));
    assert_eq!(&*icc, b"profile");
}

#[test]
fn icc_chunks_join_in_sequence_order() {
    let data = jpeg_with(&[
        icc_chunk(3, 3, b"CC"),
        icc_chunk(1, 3, b"A"),
        seg(0xFE, b"between"),
        icc_chunk(2, 3, b"BBB"),
    ]);
    let meta = parse(&data);
    assert_eq!(&*meta.icc_profile().unwrap().unwrap(), b"ABBBCC");
}

#[test]
fn icc_invalid_chunk_sets_are_errors() {
    let cases: [(Vec<Vec<u8>>, MetadataError); 6] = [
        (
            vec![icc_chunk(1, 0, b"x")],
            MetadataError::IccZeroChunkCount,
        ),
        (
            vec![icc_chunk(1, 2, b"x"), icc_chunk(2, 3, b"y")],
            MetadataError::IccChunkCountMismatch {
                expected: 2,
                found: 3,
            },
        ),
        (
            vec![icc_chunk(0, 2, b"x"), icc_chunk(1, 2, b"y")],
            MetadataError::IccSequenceOutOfRange {
                sequence: 0,
                count: 2,
            },
        ),
        (
            vec![icc_chunk(3, 2, b"x")],
            MetadataError::IccSequenceOutOfRange {
                sequence: 3,
                count: 2,
            },
        ),
        (
            vec![icc_chunk(1, 2, b"x"), icc_chunk(1, 2, b"y")],
            MetadataError::IccDuplicateChunk { sequence: 1 },
        ),
        (
            vec![icc_chunk(1, 3, b"x"), icc_chunk(3, 3, b"z")],
            MetadataError::IccMissingChunk { sequence: 2 },
        ),
    ];
    for (segments, expected) in cases {
        let data = jpeg_with(&segments);
        assert_eq!(parse(&data).icc_profile(), Err(expected));
    }
    assert_eq!(parse(&jpeg_with(&[])).icc_profile(), Ok(None));
}

#[test]
fn icc_chunks_without_data_are_an_error() {
    // libjpeg-turbo's jpeg_read_icc_profile rejects a total length of 0.
    for segments in [
        vec![icc_chunk(1, 1, b"")],
        vec![icc_chunk(2, 2, b""), icc_chunk(1, 2, b"")],
    ] {
        let data = jpeg_with(&segments);
        assert_eq!(
            parse(&data).icc_profile(),
            Err(MetadataError::IccEmptyProfile)
        );
    }
    // One empty chunk among non-empty ones is fine.
    let data = jpeg_with(&[icc_chunk(1, 2, b""), icc_chunk(2, 2, b"x")]);
    assert_eq!(&*parse(&data).icc_profile().unwrap().unwrap(), b"x");
}

#[test]
fn icc_size_limit_is_checked_before_allocating() {
    let data = jpeg_with(&[icc_chunk(1, 2, &[1; 10]), icc_chunk(2, 2, &[2; 10])]);
    let limits = MetadataLimits::default().with_max_reassembled_bytes(19);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(
        meta.icc_profile(),
        Err(MetadataError::TooLarge {
            size: 20,
            limit: 19
        })
    );
    let limits = limits.with_max_reassembled_bytes(20);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(meta.icc_profile().unwrap().unwrap().len(), 20);
}

// ── Extended XMP reassembly ─────────────────────────────────────────────

fn hex_md5(data: &[u8]) -> Vec<u8> {
    crate::foundation::md5::md5(data)
        .iter()
        .flat_map(|b| alloc::format!("{b:02X}").into_bytes())
        .collect()
}

fn ext_chunk(guid: &[u8], full: u32, offset: u32, data: &[u8]) -> Vec<u8> {
    let mut p = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    p.extend_from_slice(guid);
    p.extend_from_slice(&full.to_be_bytes());
    p.extend_from_slice(&offset.to_be_bytes());
    p.extend_from_slice(data);
    seg(0xE1, &p)
}

fn standard_xmp(packet: &[u8]) -> Vec<u8> {
    seg(0xE1, &cat(&[b"http://ns.adobe.com/xap/1.0/\0", packet]))
}

const EXTENDED: &[u8] =
    b"<rdf:Description xmlns:GDepth='http://ns.google.com/photos/1.0/depthmap/'/>";

fn has_extended(guid: &[u8]) -> Vec<u8> {
    cat(&[b"<rdf:Description xmpNote:HasExtendedXMP=\"", guid, b"\"/>"])
}

#[test]
fn extended_xmp_reassembles_out_of_order_chunks() {
    let guid = hex_md5(EXTENDED);
    let full = EXTENDED.len() as u32;
    let data = jpeg_with(&[
        standard_xmp(&has_extended(&guid)),
        ext_chunk(&guid, full, 30, &EXTENDED[30..]),
        // A chunk of another packet is ignored.
        ext_chunk(&[b'F'; 32], 4, 0, b"junk"),
        ext_chunk(&guid, full, 0, &EXTENDED[..30]),
    ]);
    let meta = parse(&data);
    assert_eq!(meta.xmp(), Ok(Some(&has_extended(&guid)[..])));
    assert_eq!(meta.extended_xmp().unwrap().as_deref(), Some(EXTENDED));
}

#[test]
fn extended_xmp_guid_forms() {
    let guid = hex_md5(EXTENDED);
    let full = EXTENDED.len() as u32;
    let lower: Vec<u8> = guid.to_ascii_lowercase();
    // The padded element value is trimmed here; ExifTool 12.40 ignores the
    // extended packet for a padded GUID (documented divergence).
    for standard in [
        cat(&[b"<x xmpNote:HasExtendedXMP = '", &guid, b"'/>"]),
        cat(&[
            b"<xmpNote:HasExtendedXMP>\n ",
            &guid,
            b" </xmpNote:HasExtendedXMP>",
        ]),
    ] {
        let data = jpeg_with(&[standard_xmp(&standard), ext_chunk(&guid, full, 0, EXTENDED)]);
        assert_eq!(
            parse(&data).extended_xmp().unwrap().as_deref(),
            Some(EXTENDED)
        );
    }
    // The digest check accepts a lowercase GUID, and chunks are selected
    // without regard to hex digit case.
    for (declared, chunk_guid) in [(&lower, &lower), (&guid, &lower), (&lower, &guid)] {
        let data = jpeg_with(&[
            standard_xmp(&has_extended(declared)),
            ext_chunk(chunk_guid, full, 0, EXTENDED),
        ]);
        assert_eq!(
            parse(&data).extended_xmp().unwrap().as_deref(),
            Some(EXTENDED)
        );
    }
    // A mention of the name that is neither an attribute nor an element
    // does not hide the declaration after it.
    let standard = cat(&[b"<!-- HasExtendedXMP below -->", &has_extended(&guid)]);
    let data = jpeg_with(&[standard_xmp(&standard), ext_chunk(&guid, full, 0, EXTENDED)]);
    assert_eq!(
        parse(&data).extended_xmp().unwrap().as_deref(),
        Some(EXTENDED)
    );
}

#[test]
fn extended_xmp_errors() {
    let guid = hex_md5(EXTENDED);
    let full = EXTENDED.len() as u32;
    let std = || standard_xmp(&has_extended(&guid));
    let case = |chunks: Vec<Vec<u8>>| {
        let mut segments = vec![std()];
        segments.extend(chunks);
        parse(&jpeg_with(&segments))
            .extended_xmp()
            .map(|o| o.map(Cow::into_owned))
    };
    assert_eq!(case(vec![]), Err(MetadataError::ExtendedXmpMissing));
    assert_eq!(
        case(vec![
            ext_chunk(&guid, full, 0, &EXTENDED[..10]),
            ext_chunk(&guid, full + 1, 10, &EXTENDED[10..]),
        ]),
        Err(MetadataError::ExtendedXmpLengthMismatch {
            expected: full,
            found: full + 1
        })
    );
    // Gap.
    assert_eq!(
        case(vec![
            ext_chunk(&guid, full, 0, &EXTENDED[..10]),
            ext_chunk(&guid, full, 12, &EXTENDED[12..]),
        ]),
        Err(MetadataError::ExtendedXmpDiscontinuous {
            expected: 10,
            offset: 12
        })
    );
    // Overlap.
    assert_eq!(
        case(vec![
            ext_chunk(&guid, full, 0, &EXTENDED[..10]),
            ext_chunk(&guid, full, 8, &EXTENDED[8..]),
        ]),
        Err(MetadataError::ExtendedXmpDiscontinuous {
            expected: 10,
            offset: 8
        })
    );
    // The whole packet twice at offset 0: an overlap too. ExifTool 12.40
    // reads it (documented divergence), as it only warns about chunks that
    // disagree on the full length (the mismatch case above).
    assert_eq!(
        case(vec![
            ext_chunk(&guid, full, 0, EXTENDED),
            ext_chunk(&guid, full, 0, EXTENDED),
        ]),
        Err(MetadataError::ExtendedXmpDiscontinuous {
            expected: full,
            offset: 0
        })
    );
    // Past the full length.
    assert_eq!(
        case(vec![ext_chunk(&guid, 5, 0, &EXTENDED[..10])]),
        Err(MetadataError::ExtendedXmpChunkPastEnd {
            offset: 0,
            length: 10,
            full_length: 5
        })
    );
    assert_eq!(
        case(vec![ext_chunk(&guid, full, 0, &EXTENDED[..10])]),
        Err(MetadataError::ExtendedXmpIncomplete {
            got: 10,
            full_length: full
        })
    );
    // Right GUID, altered content.
    let mut altered = EXTENDED.to_vec();
    altered[0] = b'(';
    assert_eq!(
        case(vec![ext_chunk(&guid, full, 0, &altered)]),
        Err(MetadataError::ExtendedXmpDigestMismatch)
    );
}

#[test]
fn extended_xmp_absent_or_undeclared_is_none() {
    let guid = hex_md5(EXTENDED);
    // No standard XMP at all, and standard XMP without the declaration: the
    // chunks stay visible as items but name no extended packet.
    for segments in [
        vec![ext_chunk(&guid, EXTENDED.len() as u32, 0, EXTENDED)],
        vec![
            standard_xmp(b"<x:xmpmeta/>"),
            ext_chunk(&guid, EXTENDED.len() as u32, 0, EXTENDED),
        ],
    ] {
        let data = jpeg_with(&segments);
        let meta = parse(&data);
        assert_eq!(meta.extended_xmp(), Ok(None));
        assert_eq!(only(&meta, "ExtendedXMP").marker, Some(0xE1));
    }
}

#[test]
fn extended_xmp_size_limit() {
    let guid = hex_md5(EXTENDED);
    let data = jpeg_with(&[
        standard_xmp(&has_extended(&guid)),
        ext_chunk(&guid, EXTENDED.len() as u32, 0, EXTENDED),
    ]);
    let limits = MetadataLimits::default().with_max_reassembled_bytes(EXTENDED.len() - 1);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(
        meta.extended_xmp(),
        Err(MetadataError::TooLarge {
            size: EXTENDED.len(),
            limit: EXTENDED.len() - 1
        })
    );
}

// ── Photoshop ───────────────────────────────────────────────────────────

fn resource(sig: &[u8; 4], id: u16, name: &[u8], data: &[u8]) -> Vec<u8> {
    let mut r = sig.to_vec();
    r.extend_from_slice(&id.to_be_bytes());
    r.push(name.len() as u8);
    r.extend_from_slice(name);
    if name.len() % 2 == 0 {
        r.push(0);
    }
    r.extend_from_slice(&(data.len() as u32).to_be_bytes());
    r.extend_from_slice(data);
    if data.len() % 2 == 1 {
        r.push(0);
    }
    r
}

fn photoshop_seg(blocks: &[u8]) -> Vec<u8> {
    seg(0xED, &cat(&[b"Photoshop 3.0\0", blocks]))
}

#[test]
fn photoshop_resources_walk_with_padding() {
    let blocks = cat(&[
        &resource(b"8BIM", 0x0404, b"", b"\x1c\x02\x00\x00\x02\x00\x04"),
        &resource(b"8BIM", 0x03ED, b"abc", b"0123456789abcdef"),
        &resource(b"PHUT", 0x0001, b"x", b"odd"),
    ]);
    let data = jpeg_with(&[photoshop_seg(&blocks)]);
    let meta = parse(&data);
    let irb = meta.photoshop().unwrap().unwrap();
    assert_eq!(irb.data(), &blocks[..]);
    let rs: Vec<_> = irb
        .resources()
        .map(|r| (r.signature, r.id, r.name, r.data))
        .collect();
    assert_eq!(
        rs,
        [
            (
                *b"8BIM",
                0x0404,
                &b""[..],
                &b"\x1c\x02\x00\x00\x02\x00\x04"[..]
            ),
            (*b"8BIM", 0x03ED, &b"abc"[..], &b"0123456789abcdef"[..]),
            (*b"PHUT", 0x0001, &b"x"[..], &b"odd"[..]),
        ]
    );
    assert_eq!(irb.iptc(), Some(&b"\x1c\x02\x00\x00\x02\x00\x04"[..]));
}

#[test]
fn photoshop_resources_stop_at_a_bad_block() {
    let blocks = cat(&[
        &resource(b"8BIM", 0x0404, b"", b"ok"),
        b"XXXX\x04\x04\0\0\0\0\0\x02hi",
    ]);
    let irb = PhotoshopIrb(Cow::Borrowed(&blocks));
    assert_eq!(irb.resources().count(), 1);
    // Declared size past the end.
    let mut cut = resource(b"8BIM", 0x0404, b"", b"data");
    cut.truncate(cut.len() - 1);
    assert_eq!(PhotoshopIrb(Cow::Borrowed(&cut)).resources().count(), 0);
}

#[test]
fn photoshop_consecutive_segments_join_and_later_runs_are_separate() {
    let block = resource(b"8BIM", 0x0404, b"", &[7; 20]);
    let (a, b) = block.split_at(9);
    let data = jpeg_with(&[
        photoshop_seg(a),
        photoshop_seg(b),
        seg(0xFE, b"split"),
        photoshop_seg(&resource(b"8BIM", 0x0405, b"", b"later")),
    ]);
    let meta = parse(&data);
    let irb = meta.photoshop().unwrap().unwrap();
    assert!(matches!(irb.0, Cow::Owned(_)));
    assert_eq!(irb.data(), &block[..]);
    assert_eq!(irb.iptc(), Some(&[7u8; 20][..]));
    // Single segment: borrowed.
    let data = jpeg_with(&[photoshop_seg(&block)]);
    let meta = parse(&data);
    assert!(matches!(
        meta.photoshop().unwrap().unwrap().0,
        Cow::Borrowed(_)
    ));
    assert_eq!(parse(&jpeg_with(&[])).photoshop(), Ok(None));
}

#[test]
fn photoshop_run_continues_past_fill_and_stray_bytes() {
    // ExifTool reads the next marker with ReadLine up to the first 0xFF and
    // skips 0xFF fill, so fill and stray bytes without an 0xFF keep the run.
    let block = resource(b"8BIM", 0x0404, b"", &[7; 20]);
    let (a, b) = block.split_at(9);
    for between in [&[0xFF, 0xFF][..], &[0x01, 0x02], &[0x01, 0xFF]] {
        let data = jpeg_with(&[photoshop_seg(a), between.to_vec(), photoshop_seg(b)]);
        let meta = parse(&data);
        let irb = meta.photoshop().unwrap().unwrap();
        assert_eq!(irb.data(), &block[..], "{between:?}");
    }
    // An FF 00 or an RSTn among stray bytes is a marker to ExifTool: the
    // first run ends there.
    for between in [&[0xFF, 0x00][..], &[0x01, 0xFF, 0xD0]] {
        let data = jpeg_with(&[photoshop_seg(a), between.to_vec(), photoshop_seg(b)]);
        let meta = parse(&data);
        assert!(names(&meta).contains(&"Unparsed"), "{between:?}");
        let irb = meta.photoshop().unwrap().unwrap();
        assert_eq!(irb.data(), a, "{between:?}");
    }
}

#[test]
fn photoshop_old_header_starts_but_never_continues_a_run() {
    let old = |blocks: &[u8]| seg(0xED, &cat(&[b"Adobe_Photoshop2.5:", &[0; 8], blocks]));
    let block = resource(b"8BIM", 0x0404, b"", &[7; 20]);
    let (a, b) = block.split_at(9);
    // Old header first, 3.0 continuation: joined. Deliberate divergence
    // from ExifTool (docs/METADATA.md, "Deliberate divergences"): ExifTool
    // applies the run's last segment's header length to the joined buffer
    // (`DirStart = $isOld ? 27 : 14`), reads this run from byte 14, inside
    // the old 27-byte header, and finds no resources. Here each segment's own
    // header is stripped, so the block comes back whole.
    let data = jpeg_with(&[old(a), photoshop_seg(b)]);
    assert_eq!(
        parse(&data).photoshop().unwrap().unwrap().data(),
        &block[..]
    );
    // 3.0 first, old header next: the old segment starts a later run.
    let data = jpeg_with(&[photoshop_seg(&block), old(b"tail")]);
    let meta = parse(&data);
    assert_eq!(meta.photoshop().unwrap().unwrap().data(), &block[..]);
    assert_eq!(
        kinds(&meta)
            .iter()
            .filter(|k| matches!(k, ItemKind::Photoshop(_)))
            .count(),
        2
    );
    // Shorter than the 27-byte old header: not identified.
    let short = seg(0xED, b"Adobe_Photoshop2.5:\0\0");
    assert_eq!(names(&parse(&jpeg_with(&[short])))[1], "Unknown");
}

#[test]
fn photoshop_run_cut_by_the_item_limit_is_an_error() {
    let block = resource(b"8BIM", 0x0404, b"", &[7; 20]);
    let (a, b) = block.split_at(9);
    let data = jpeg_with(&[photoshop_seg(a), photoshop_seg(b)]);
    // SOI, the first Photoshop segment, then the limit item.
    let limits = MetadataLimits::default().with_max_items(3);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(names(&meta), ["SOI", "Photoshop", "Unparsed"]);
    assert_eq!(meta.photoshop(), Err(MetadataError::ItemLimitReached));
}

#[test]
fn photoshop_size_limit_is_checked_before_allocating() {
    let block = resource(b"8BIM", 0x0404, b"", &[7; 20]);
    let (a, b) = block.split_at(9);
    let data = jpeg_with(&[photoshop_seg(a), photoshop_seg(b)]);
    let limits = MetadataLimits::default().with_max_reassembled_bytes(block.len() - 1);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(
        meta.photoshop(),
        Err(MetadataError::TooLarge {
            size: block.len(),
            limit: block.len() - 1
        })
    );
    // A single segment over the limit is refused too, though it would borrow.
    let data = jpeg_with(&[photoshop_seg(&block)]);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert!(matches!(
        meta.photoshop(),
        Err(MetadataError::TooLarge { .. })
    ));
}

// ── MPF and trailers ────────────────────────────────────────────────────

#[test]
fn mpf_directory_and_appended_images() {
    let primary_body = jpeg_with(&[]);
    let gainmap = jpeg_with(&[seg(0xFE, b"gain map")]);
    // Insert the MPF APP2 right after SOI; its size is fixed, so compute it
    // once to learn the primary's length.
    let probe = crate::container::mpf::create_mpf_header(0, 0, Some(2));
    let primary_len = primary_body.len() + probe.len();
    let header = crate::container::mpf::create_mpf_header(primary_len, gainmap.len(), Some(2));
    assert_eq!(header.len(), probe.len());
    let mut data = SOI.to_vec();
    data.extend_from_slice(&header);
    data.extend_from_slice(&primary_body[2..]);
    data.extend_from_slice(&gainmap);
    let meta = parse(&data);
    assert_eq!(meta.primary_image_end(), Some(primary_len));
    let embedded = only(&meta, "EmbeddedJPEG");
    assert_eq!(
        (embedded.offset, embedded.payload),
        (primary_len, &gainmap[..])
    );
    let entries = meta.mpf().unwrap().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(
        (entries[1].offset, entries[1].size),
        (primary_len, gainmap.len())
    );
    // The embedded stream walks on its own.
    let inner = parse(embedded.payload);
    assert_eq!(only(&inner, "Comment").payload, b"gain map");
}

#[test]
fn trailer_gaps_are_padding_or_unknown() {
    let primary = jpeg_with(&[]);
    let second = jpeg_with(&[]);
    let data = cat(&[&primary, &[0; 16], &second, &[0xFF; 3], b"vendor"]);
    let meta = parse(&data);
    assert_eq!(
        names(&meta)[4..],
        ["Padding", "EmbeddedJPEG", "UnknownTrailer"]
    );
    let only_padding = cat(&[&primary, &[0xFF; 4]]);
    assert_eq!(names(&parse(&only_padding))[4..], ["Padding"]);
}

#[test]
fn appended_jpeg_needs_a_clean_walk() {
    let primary = jpeg_with(&[]);
    // A vendor blob with a stray FF D8, garbage, and a later FF D9: walking
    // it would need resynchronization, so it is not an image.
    let blob: &[u8] = &[0x01, 0xFF, 0xD8, 0x55, 0x66, 0x77, 0xFF, 0xD9, 0x02];
    let data = cat(&[&primary, blob]);
    assert_eq!(names(&parse(&data))[4..], ["UnknownTrailer"]);
    // The same blob followed by a real image: the search resumes after the
    // failed candidate and finds it.
    let second = jpeg_with(&[seg(0xFE, b"second")]);
    let data = cat(&[&primary, blob, &second]);
    let meta = parse(&data);
    assert_eq!(names(&meta)[4..], ["UnknownTrailer", "EmbeddedJPEG"]);
    let embedded = only(&meta, "EmbeddedJPEG");
    assert_eq!(embedded.payload, &second[..]);
    assert_eq!(
        parse(embedded.payload).primary_image_end(),
        Some(second.len())
    );
}

fn afcp_trailer(abs_start: u32, big_endian: bool) -> Vec<u8> {
    let sig: &[u8] = if big_endian { b"AXS!" } else { b"AXS*" };
    let pos = if big_endian {
        abs_start.to_be_bytes()
    } else {
        abs_start.to_le_bytes()
    };
    cat(&[
        sig,
        b"\x00\x01\x00\x00\x00\x00\x00\x00",
        b"IPTC entries",
        sig,
        &pos,
        b"\0\0\0\0",
    ])
}

#[test]
fn afcp_trailer_both_byte_orders() {
    for big in [true, false] {
        let primary = jpeg_with(&[]);
        let data = cat(&[&primary, &afcp_trailer(primary.len() as u32, big)]);
        let meta = parse(&data);
        let afcp = only(&meta, "AFCP");
        assert_eq!(
            (afcp.offset, afcp.length),
            (primary.len(), data.len() - primary.len())
        );
    }
    // A header offset that does not point at the header.
    let primary = jpeg_with(&[]);
    let data = cat(&[&primary, &afcp_trailer(primary.len() as u32 + 1, true)]);
    assert_eq!(names(&parse(&data))[4..], ["UnknownTrailer"]);
}

fn fotostation_record(tag: u16, data: &[u8]) -> Vec<u8> {
    let size = (data.len() + 10) as u32;
    cat(&[
        data,
        &tag.to_be_bytes(),
        &size.to_be_bytes(),
        &[0xA1, 0xB2, 0xC3, 0xD4],
    ])
}

fn photo_mechanic(data: &[u8]) -> Vec<u8> {
    cat(&[data, &(data.len() as u32).to_be_bytes(), b"cbipcbbl"])
}

#[test]
fn chained_vendor_trailers_in_file_order() {
    let primary = jpeg_with(&[]);
    let preview = jpeg_with(&[]);
    let data = cat(&[
        &primary,
        &preview,
        &photo_mechanic(b"\x1c\x02\x00\x00\x02\x00\x04"),
        &fotostation_record(1, b"iptc"),
        &fotostation_record(4, b"preview"),
    ]);
    let meta = parse(&data);
    assert_eq!(
        kinds(&meta)[4..],
        [
            ItemKind::EmbeddedJpeg,
            ItemKind::PhotoMechanic,
            ItemKind::FotoStation(1),
            ItemKind::FotoStation(4),
        ]
    );
    // PhotoMechanic with the size's high bit set is not a trailer.
    let data = cat(&[
        &primary,
        b"data",
        &0x8000_0004u32.to_be_bytes(),
        b"cbipcbbl",
    ]);
    assert_eq!(names(&parse(&data))[4..], ["UnknownTrailer"]);
}

#[test]
fn long_record_chain_respects_the_item_limit() {
    // 1000 FotoStation records: the chain stops at the item budget instead
    // of materializing every record.
    let primary = jpeg_with(&[]);
    let mut data = primary.clone();
    for i in 0..1000u16 {
        data.extend_from_slice(&fotostation_record(i, b"x"));
    }
    let meta = parse(&data);
    assert_eq!(meta.items().len(), 4 + 1000);
    let limits = MetadataLimits {
        max_items: 10,
        ..MetadataLimits::default()
    };
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_tiles(&meta, &data);
    assert_eq!(meta.items().len(), 10);
    assert_eq!(
        meta.items()[9].kind,
        ItemKind::Unparsed(Unparsed::LimitExceeded)
    );
}

#[test]
fn canon_vrd_trailer() {
    let primary = jpeg_with(&[]);
    let body = b"VRD data!";
    let mut header = b"CANON OPTIONAL DATA\0".to_vec();
    header.resize(0x1C, 0);
    let mut footer = b"CANON OPTIONAL DATA\0".to_vec();
    footer.resize(0x40, 0);
    footer[0x14..0x18].copy_from_slice(&(body.len() as u32).to_be_bytes());
    let data = cat(&[&primary, &header, body, &footer]);
    let meta = parse(&data);
    let vrd = only(&meta, "CanonVRD");
    assert_eq!(vrd.length, 0x1C + body.len() + 0x40);
    // A size that does not land on the header.
    footer[0x17] += 1;
    let data = cat(&[&primary, &header, body, &footer]);
    assert_eq!(names(&parse(&data))[4..], ["UnknownTrailer"]);
}

/// A MIE group ending in the `zmie` signature and a terminator with a
/// 4-byte (`long == false`) or 8-byte length.
fn mie_group(content: &[u8], big_endian: bool, long: bool) -> Vec<u8> {
    let mut g = content.to_vec();
    g.extend_from_slice(b"~\0\x04\0zmie");
    let order = if big_endian { 0x10 } else { 0x18 };
    let len = g.len() + if long { 14 } else { 10 };
    if long {
        g.extend_from_slice(b"~\0\0\x0a");
        let l = len as u64;
        g.extend_from_slice(&if big_endian {
            l.to_be_bytes()
        } else {
            l.to_le_bytes()
        });
        g.extend_from_slice(&[order, 0x08]);
    } else {
        g.extend_from_slice(b"~\0\0\x06");
        let l = len as u32;
        g.extend_from_slice(&if big_endian {
            l.to_be_bytes()
        } else {
            l.to_le_bytes()
        });
        g.extend_from_slice(&[order, 0x04]);
    }
    g
}

#[test]
fn mie_trailer_all_terminator_forms() {
    let primary = jpeg_with(&[]);
    for (big, long) in [(true, false), (false, false), (true, true), (false, true)] {
        let group = mie_group(b"~\x10\x04\x000MIE....", big, long);
        let data = cat(&[&primary, b"xx", &group]);
        let meta = parse(&data);
        let mie = only(&meta, "MIE");
        assert_eq!(mie.length, group.len(), "big {big} long {long}");
        assert_eq!(names(&meta)[4], "UnknownTrailer");
    }
}

#[test]
fn mie_trailer_chains_preceding_groups() {
    let primary = jpeg_with(&[]);
    // An earlier group (opening with the 0MIE element) ends in a plain
    // 4-byte terminator.
    let mut earlier = b"~\x10\x04\x000MIEgroup one".to_vec();
    let len = (earlier.len() + 10) as u32;
    earlier.extend_from_slice(b"~\0\0\x06");
    earlier.extend_from_slice(&len.to_be_bytes());
    earlier.extend_from_slice(&[0x10, 0x04]);
    let last = mie_group(b"group two", true, false);
    let data = cat(&[&primary, &earlier, &last]);
    let mie = only(&parse(&data), "MIE");
    assert_eq!(mie.length, earlier.len() + last.len());
}

#[test]
fn mie_trailer_needs_the_0mie_element_first() {
    // ExifTool's ProcessMIE rejects a file-level group that does not open
    // with `~[\x10\x18]\x04?0MIE`; the bytes are then an unknown trailer.
    let primary = jpeg_with(&[]);
    for content in [
        &b"group one"[..],
        b"~\x10\x04\x000MIX....",
        b"~\x20\x04\x000MIE....",
    ] {
        let group = mie_group(content, true, false);
        let data = cat(&[&primary, &group]);
        assert_eq!(names(&parse(&data))[4..], ["UnknownTrailer"], "{content:?}");
    }
    let group = mie_group(b"~\x18\x04\xff0MIE....", false, false);
    let data = cat(&[&primary, &group]);
    assert_eq!(only(&parse(&data), "MIE").length, group.len());
}

/// A Samsung trailer: data blocks, then the SEFH directory as the `SEFT`
/// block, optionally followed by a `QDIO` block and `BS`.
fn samsung_trailer(qdio: bool) -> Vec<u8> {
    let block = b"\0\0\x01\x0aImage_UTC_Data1234";
    let mut t = block.to_vec();
    let dir_start = t.len();
    let mut dir = b"SEFH".to_vec();
    dir.extend_from_slice(&101u32.to_le_bytes());
    dir.extend_from_slice(&1u32.to_le_bytes());
    dir.extend_from_slice(&[0, 0, 0x01, 0x0a]);
    dir.extend_from_slice(&(dir_start as u32).to_le_bytes()); // distance back to the block
    dir.extend_from_slice(&(block.len() as u32).to_le_bytes());
    t.extend_from_slice(&dir);
    t.extend_from_slice(&(dir.len() as u32).to_le_bytes());
    t.extend_from_slice(b"SEFT");
    if qdio {
        let q = b"QDIO\x65\0\0\0\x01\0\0\0\0\0\0\0\0\0\0\0";
        t.extend_from_slice(q);
        t.extend_from_slice(&(q.len() as u32).to_le_bytes());
        t.extend_from_slice(b"QDIOBS");
    }
    t
}

#[test]
fn samsung_trailer_seft_and_qdio() {
    for qdio in [false, true] {
        let primary = jpeg_with(&[]);
        let trailer = samsung_trailer(qdio);
        let data = cat(&[&primary, &trailer]);
        let meta = parse(&data);
        let s = only(&meta, "Samsung");
        assert_eq!(
            (s.offset, s.length),
            (primary.len(), trailer.len()),
            "qdio {qdio}"
        );
    }
}

// ── MarkerIter stop reasons ─────────────────────────────────────────────

#[test]
fn marker_iter_records_why_it_stopped() {
    use crate::container::marker::{IterStop, MarkerIter};
    let stop = |data: &[u8]| {
        let mut it = MarkerIter::new(data);
        while it.next().is_some() {}
        it.stop()
    };
    assert_eq!(stop(&cat(&[SOI, EOI, b"junk"])), None);
    assert_eq!(stop(SOI), None);
    assert_eq!(
        stop(&cat(&[SOI, b"x"])),
        Some(IterStop::NotAMarker { offset: 2 })
    );
    assert_eq!(
        stop(&cat(&[SOI, &[0xFF]])),
        Some(IterStop::Truncated { offset: 2 })
    );
    assert_eq!(
        stop(&cat(&[SOI, &[0xFF, 0xFF, 0xE0, 0x00, 0x01]])),
        Some(IterStop::BadLength { offset: 2 })
    );
    assert_eq!(
        stop(&cat(&[SOI, &[0xFF, 0xE0, 0x00, 0x10, 1]])),
        Some(IterStop::Truncated { offset: 2 })
    );
    assert_eq!(
        stop(&cat(&[SOI, &[0xFF, 0xDA, 0x00]])),
        Some(IterStop::Truncated { offset: 2 })
    );
    assert_eq!(
        stop(&cat(&[SOI, &[0xFF, 0xDA, 0x00, 0x00]])),
        Some(IterStop::BadLength { offset: 2 })
    );
}

// ── Header scope and the item limit ─────────────────────────────────────

/// One of each segment an accessor reads.
fn accessor_segments() -> Vec<Vec<u8>> {
    let mpf = crate::container::mpf::create_mpf_header(100, 10, None);
    vec![
        seg(0xE1, b"Exif\0\0II*\0\x08\0\0\0"),
        standard_xmp(b"<x:xmpmeta/>"),
        icc_chunk(1, 1, b"profile"),
        photoshop_seg(&resource(b"8BIM", 0x0404, b"", b"ok")),
        mpf,
    ]
}

#[test]
fn accessors_read_the_header_only() {
    // The same segments after the first SOS are items, not metadata.
    let mut data = cat(&[SOI, &sof0(), &sos_with_entropy()]);
    for s in accessor_segments() {
        data.extend_from_slice(&s);
    }
    data.extend_from_slice(EOI);
    let meta = parse(&data);
    assert!(names(&meta).contains(&"ICC_Profile") && names(&meta).contains(&"MPF"));
    assert_eq!(meta.exif(), Ok(None));
    assert_eq!(meta.xmp(), Ok(None));
    assert_eq!(meta.icc_profile(), Ok(None));
    assert_eq!(meta.photoshop(), Ok(None));
    assert_eq!(meta.mpf(), Ok(None));
    // Before the SOS they are read.
    let data = jpeg_with(&accessor_segments());
    let meta = parse(&data);
    assert!(meta.exif().unwrap().is_some());
    assert!(meta.xmp().unwrap().is_some());
    assert_eq!(&*meta.icc_profile().unwrap().unwrap(), b"profile");
    assert!(meta.photoshop().unwrap().is_some());
    assert_eq!(meta.mpf().unwrap().unwrap().len(), 2);
}

#[test]
fn icc_chunk_after_the_first_sos_does_not_break_the_header_profile() {
    // A stray chunk between progressive scans would otherwise duplicate
    // chunk 1; libjpeg-turbo reads only the header's markers.
    let data = cat(&[
        SOI,
        &icc_chunk(1, 1, b"header"),
        &sof0(),
        &sos_with_entropy(),
        &icc_chunk(1, 1, b"later"),
        &sos_with_entropy(),
        EOI,
    ]);
    let meta = parse(&data);
    assert_eq!(&*meta.icc_profile().unwrap().unwrap(), b"header");
}

#[test]
fn a_header_cut_by_the_item_limit_is_reported_by_every_accessor() {
    // SOI, three comments, then the limit item: nothing past the cut is
    // known, so a miss is an error, not `None`.
    let mut segments = vec![seg(0xFE, b"a"), seg(0xFE, b"b"), seg(0xFE, b"c")];
    segments.extend(accessor_segments());
    let data = jpeg_with(&segments);
    let limits = MetadataLimits::default().with_max_items(5);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(names(&meta).last(), Some(&"Unparsed"));
    let cut = |e: MetadataError| e == MetadataError::ItemLimitReached;
    assert!(meta.exif().is_err_and(cut));
    assert!(meta.xmp().is_err_and(cut));
    assert!(meta.extended_xmp().is_err_and(cut));
    assert!(meta.icc_profile().is_err_and(cut));
    assert!(meta.photoshop().is_err_and(cut));
    assert!(meta.mpf().is_err_and(cut));
    // A single-segment payload found before the cut is complete.
    let data = jpeg_with(&accessor_segments());
    let limits = MetadataLimits::default().with_max_items(3);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert!(meta.exif().unwrap().is_some());
    assert!(meta.icc_profile().is_err_and(cut));
    // A header that ends at its SOS before the limit is not cut.
    // (SOI, SOF, SOS, then the limit item over the EOI and a trailer.)
    let data = cat(&[&jpeg_with(&[]), b"trailer"]);
    let limits = MetadataLimits::default().with_max_items(4);
    let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
    assert_eq!(names(&meta), ["SOI", "SOF", "SOS", "Unparsed"]);
    assert_eq!(meta.icc_profile(), Ok(None));
    assert_eq!(meta.exif(), Ok(None));
    // The limit item may replace the SOS itself (after fill bytes or not):
    // the header still ended there.
    for data in [
        jpeg_with(&[]),
        cat(&[SOI, &sof0(), &[0xFF, 0xFF], &sos_with_entropy(), EOI]),
    ] {
        let limits = MetadataLimits::default().with_max_items(3);
        let meta = JpegMetadata::parse_with_limits(&data, limits).unwrap();
        assert_eq!(names(&meta), ["SOI", "SOF", "Unparsed"]);
        assert_eq!(meta.icc_profile(), Ok(None));
        assert_eq!(meta.extended_xmp(), Ok(None));
        assert_eq!(meta.photoshop(), Ok(None));
        assert_eq!(meta.xmp(), Ok(None));
    }
    // With no SOS the header ends at the EOI, which the limit item may
    // replace too; any other marker there leaves the header cut.
    let no_scan = cat(&[SOI, &seg(0xFE, b"a"), EOI, b"trailer"]);
    let limits = MetadataLimits::default().with_max_items(2);
    let meta = JpegMetadata::parse_with_limits(&no_scan, limits).unwrap();
    assert_eq!(names(&meta), ["SOI", "Unparsed"]);
    assert!(meta.icc_profile().is_err_and(cut));
    let limits = MetadataLimits::default().with_max_items(3);
    let meta = JpegMetadata::parse_with_limits(&no_scan, limits).unwrap();
    assert_eq!(names(&meta), ["SOI", "Comment", "Unparsed"]);
    assert_eq!(meta.icc_profile(), Ok(None));
    assert_eq!(meta.exif(), Ok(None));
}

#[test]
fn extended_xmp_in_one_chunk_is_borrowed() {
    let guid = hex_md5(EXTENDED);
    let full = EXTENDED.len() as u32;
    let data = jpeg_with(&[
        standard_xmp(&has_extended(&guid)),
        ext_chunk(&guid, full, 0, EXTENDED),
    ]);
    let meta = parse(&data);
    assert!(matches!(
        meta.extended_xmp(),
        Ok(Some(Cow::Borrowed(EXTENDED)))
    ));
    let data = jpeg_with(&[
        standard_xmp(&has_extended(&guid)),
        ext_chunk(&guid, full, 0, &EXTENDED[..9]),
        ext_chunk(&guid, full, 9, &EXTENDED[9..]),
    ]);
    let meta = parse(&data);
    let packet = meta.extended_xmp().unwrap().unwrap();
    assert!(matches!(packet, Cow::Owned(_)));
    assert_eq!(&*packet, EXTENDED);
}

#[test]
fn the_last_has_extended_xmp_value_wins() {
    // ExifTool keeps the last HasExtendedXMP value it reads.
    let guid = hex_md5(EXTENDED);
    let bogus = [b'0'; 32];
    let full = EXTENDED.len() as u32;
    let chunk = ext_chunk(&guid, full, 0, EXTENDED);
    let two =
        |first: &[u8], second: &[u8]| cat(&[&has_extended(first), b"\n", &has_extended(second)]);
    let data = jpeg_with(&[standard_xmp(&two(&bogus, &guid)), chunk.clone()]);
    assert_eq!(
        parse(&data).extended_xmp().unwrap().as_deref(),
        Some(EXTENDED)
    );
    let data = jpeg_with(&[standard_xmp(&two(&guid, &bogus)), chunk.clone()]);
    assert_eq!(
        parse(&data).extended_xmp(),
        Err(MetadataError::ExtendedXmpMissing)
    );
    // Across standard packets too.
    let data = jpeg_with(&[
        standard_xmp(&has_extended(&guid)),
        standard_xmp(&has_extended(&bogus)),
        chunk,
    ]);
    assert_eq!(
        parse(&data).extended_xmp(),
        Err(MetadataError::ExtendedXmpMissing)
    );
}

#[test]
fn unclosed_has_extended_xmp_candidates_are_skipped() {
    // Candidates after the last `<` (or quote) cannot be closed; the valid
    // element before them is the last one that parses.
    let guid = hex_md5(EXTENDED);
    let mut packet = cat(&[
        b"<xmpNote:HasExtendedXMP>",
        &guid,
        b"</xmpNote:HasExtendedXMP>",
    ]);
    for _ in 0..1000 {
        packet.extend_from_slice(b" x:HasExtendedXMP>");
    }
    packet.extend_from_slice(b" x:HasExtendedXMP=\" x:HasExtendedXMP='");
    let data = jpeg_with(&[
        standard_xmp(&packet),
        ext_chunk(&guid, EXTENDED.len() as u32, 0, EXTENDED),
    ]);
    assert_eq!(
        parse(&data).extended_xmp().unwrap().as_deref(),
        Some(EXTENDED)
    );
}

#[test]
fn a_longer_name_ending_in_has_extended_xmp_is_not_the_property() {
    // `y:FakeHasExtendedXMP` after the real property, as an attribute or an
    // element: ExifTool reads the extended packet the real one names, so the
    // name must follow a `:`.
    let guid = hex_md5(EXTENDED);
    let bogus = [b'0'; 32];
    let chunk = ext_chunk(&guid, EXTENDED.len() as u32, 0, EXTENDED);
    let attribute = cat(&[
        &has_extended(&guid),
        b"<rdf:Description y:FakeHasExtendedXMP=\"",
        &bogus,
        b"\"/>",
    ]);
    let element = cat(&[
        &has_extended(&guid),
        b"<rdf:Description><y:FakeHasExtendedXMP>",
        &bogus,
        b"</y:FakeHasExtendedXMP></rdf:Description>",
    ]);
    let unprefixed = cat(&[
        &has_extended(&guid),
        b"<x HasExtendedXMP=\"",
        &bogus,
        b"\"/>",
    ]);
    let text = cat(&[
        &has_extended(&guid),
        b"<dc:title>HasExtendedXMP=\"",
        &bogus,
        b"\"</dc:title>",
    ]);
    for packet in [attribute, element, unprefixed, text] {
        let data = jpeg_with(&[standard_xmp(&packet), chunk.clone()]);
        assert_eq!(
            parse(&data).extended_xmp().unwrap().as_deref(),
            Some(EXTENDED),
            "{}",
            String::from_utf8_lossy(&packet)
        );
    }
    // A name at the very start of the packet has no prefix either.
    let data = jpeg_with(&[
        standard_xmp(&cat(&[b"HasExtendedXMP=\"", &guid, b"\""])),
        chunk.clone(),
    ]);
    assert_eq!(parse(&data).extended_xmp(), Ok(None));
    // Documented divergence: a prefixed name in a comment (or in element
    // text) still counts here; ExifTool 12.40 ignores it and reads the
    // packet the real property names.
    let comment = cat(&[
        &has_extended(&guid),
        b"<!-- xmpNote:HasExtendedXMP=\"",
        &bogus,
        b"\" -->",
    ]);
    let data = jpeg_with(&[standard_xmp(&comment), chunk.clone()]);
    assert_eq!(
        parse(&data).extended_xmp(),
        Err(MetadataError::ExtendedXmpMissing)
    );
    // Documented divergence: the prefix is not resolved, so the name under
    // another namespace, or as a namespace declaration, after the real
    // property wins here; ExifTool 12.40 reads the packet `xmpNote` names.
    for name in [&b"y:HasExtendedXMP"[..], b"xmlns:HasExtendedXMP"] {
        let other = cat(&[
            &has_extended(&guid),
            b"<rdf:Description ",
            name,
            b"='",
            &bogus,
            b"'/>",
        ]);
        let data = jpeg_with(&[standard_xmp(&other), chunk.clone()]);
        assert_eq!(
            parse(&data).extended_xmp(),
            Err(MetadataError::ExtendedXmpMissing)
        );
    }
}

#[test]
fn a_closing_tag_is_not_a_has_extended_xmp_value() {
    // XML after the element: its closing tag must not win as the last
    // occurrence (its "value" would run to the next `<`).
    let guid = hex_md5(EXTENDED);
    let packet = cat(&[
        b"<rdf:Description><xmpNote:HasExtendedXMP>",
        &guid,
        b"</xmpNote:HasExtendedXMP> </rdf:Description>",
    ]);
    let data = jpeg_with(&[
        standard_xmp(&packet),
        ext_chunk(&guid, EXTENDED.len() as u32, 0, EXTENDED),
    ]);
    assert_eq!(
        parse(&data).extended_xmp().unwrap().as_deref(),
        Some(EXTENDED)
    );
}

#[test]
fn an_mpf_parse_error_is_a_metadata_error() {
    use crate::container::mpf::MpfError;
    let data = jpeg_with(&[seg(0xE2, b"MPF\0XX\0*\0\0\0\x08")]);
    assert_eq!(
        parse(&data).mpf(),
        Err(MetadataError::Mpf(MpfError::BadEndian(*b"XX")))
    );
    // One out-of-memory variant for callers to check.
    assert_eq!(
        MetadataError::from(MpfError::OutOfMemory),
        MetadataError::OutOfMemory
    );
    assert_eq!(
        MetadataError::from(MpfError::NoImages),
        MetadataError::Mpf(MpfError::NoImages)
    );
}
