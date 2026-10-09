//! Inventory tests on hand-built JPEGs, so every range is known exactly.

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
use zencodec::inventory::{Disposition as D, Inventory, MetadataKind as M, PartKind as K, PartTag};

use super::{Options, exif_orientation_segment, inventory};
use crate::JpegDecoderConfig;

fn segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, marker];
    v.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    v.extend_from_slice(payload);
    v
}

/// One 8-bit quantisation table of ones.
fn dqt() -> Vec<u8> {
    let mut p = vec![0x00];
    p.extend([1u8; 64]);
    segment(0xDB, &p)
}

/// DC and AC tables 0, each with a single one-bit code: DC category 0 and EOB.
fn dht() -> Vec<u8> {
    let mut bits = [0u8; 16];
    bits[0] = 1;
    let mut p = vec![0x00];
    p.extend(bits);
    p.push(0x00);
    p.push(0x10);
    p.extend(bits);
    p.push(0x00);
    segment(0xC4, &p)
}

fn sof(marker: u8, w: u16, h: u16) -> Vec<u8> {
    let mut p = vec![8];
    p.extend(h.to_be_bytes());
    p.extend(w.to_be_bytes());
    p.extend([1, 1, 0x11, 0]);
    segment(marker, &p)
}

fn sos() -> Vec<u8> {
    segment(0xDA, &[1, 1, 0x00, 0, 63, 0])
}

/// EXIF with one IFD0 entry: Orientation.
fn exif(orientation: u16) -> Vec<u8> {
    let mut p = b"Exif\0\0MM\0\x2a\0\0\0\x08\0\x01".to_vec();
    p.extend([0x01, 0x12, 0x00, 0x03, 0, 0, 0, 1]);
    p.extend(orientation.to_be_bytes());
    p.extend([0, 0, 0, 0, 0, 0]);
    segment(0xE1, &p)
}

/// The primary/secondary MPF index: entry 1 at `rel` (relative to the TIFF
/// header after `MPF\0`).
fn mpf(primary_len: u32, gm_len: u32, rel: u32) -> Vec<u8> {
    let mut p = b"MPF\0MM\0\x2a\0\0\0\x08".to_vec();
    p.extend([0, 3]);
    p.extend([0xB0, 0x00, 0, 7, 0, 0, 0, 4]);
    p.extend(b"0100");
    p.extend([0xB0, 0x01, 0, 4, 0, 0, 0, 1, 0, 0, 0, 2]);
    // 8 + 2 + 3*12 + 4 = 50: the entries follow the IFD.
    p.extend([0xB0, 0x02, 0, 7, 0, 0, 0, 32, 0, 0, 0, 50]);
    p.extend([0, 0, 0, 0]);
    p.extend([0x00, 0x03, 0x00, 0x00]);
    p.extend(primary_len.to_be_bytes());
    p.extend([0, 0, 0, 0, 0, 0, 0, 0]);
    p.extend([0, 0, 0, 0]);
    p.extend(gm_len.to_be_bytes());
    p.extend(rel.to_be_bytes());
    p.extend([0, 0, 0, 0]);
    segment(0xE2, &p)
}

/// A minimal 8x8 greyscale JPEG, as (marker, bytes) units in file order;
/// marker 0 is the scan data.
fn tiny_units() -> Vec<(u8, Vec<u8>)> {
    vec![
        (0xD8, vec![0xFF, 0xD8]),
        (
            0xE1,
            segment(0xE1, b"http://ns.adobe.com/xap/1.0/\0<x:xmpmeta/>"),
        ),
        (0xDB, dqt()),
        (0xC4, dht()),
        (0xC0, sof(0xC0, 8, 8)),
        (0xDA, sos()),
        (0, vec![0x3F]),
        (0xD9, vec![0xFF, 0xD9]),
    ]
}

fn tiny_jpeg() -> Vec<u8> {
    tiny_units().into_iter().flat_map(|(_, b)| b).collect()
}

/// Byte ranges recorded while building a fixture.
struct Fixture {
    bytes: Vec<u8>,
    marks: Vec<(&'static str, Range<usize>)>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            marks: Vec::new(),
        }
    }
    fn put(&mut self, name: &'static str, b: &[u8]) {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(b);
        self.marks.push((name, start..self.bytes.len()));
    }
    fn at(&self, name: &str) -> Range<u64> {
        let r = &self
            .marks
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no mark {name}"))
            .1;
        r.start as u64..r.end as u64
    }
}

const XMP: &str = concat!(
    "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF><rdf:Description>",
    "<Container:Directory><rdf:Seq>",
    "<rdf:li rdf:parseType=\"Resource\"><Container:Item Item:Semantic=\"Primary\" ",
    "Item:Mime=\"image/jpeg\"/></rdf:li>",
    "<rdf:li rdf:parseType=\"Resource\"><Container:Item Item:Semantic=\"GainMap\" ",
    "Item:Mime=\"image/jpeg\" Item:Length=\"GMLEN\"/></rdf:li>",
    "<rdf:li rdf:parseType=\"Resource\"><Container:Item Item:Semantic=\"MotionPhoto\" ",
    "Item:Mime=\"video/mp4\" Item:Length=\"0000000016\"/></rdf:li>",
    "</rdf:Seq></Container:Directory></rdf:Description></rdf:RDF></x:xmpmeta>",
);

/// A file with one of every unit type: every APPn kind the decoder
/// classifies plus private ones, COM, TEM, a skipped marker, stray and fill
/// bytes, DQT, DHT, DAC, DRI, SOF0, a scan with a restart marker, DNL, a
/// second frame header, EOI, then an MPF gain map, a GContainer video item,
/// trailing junk and a Samsung SEF trailer.
fn everything() -> Fixture {
    let gm = tiny_jpeg();
    let xmp = XMP.replace("GMLEN", &alloc::format!("{:010}", gm.len()));
    let mut f = Fixture::new();
    f.put("soi", &[0xFF, 0xD8]);
    // JFIF 1.02, 72 dpi, a 1x1 thumbnail.
    f.put(
        "jfif",
        &segment(0xE0, b"JFIF\0\x01\x02\x01\0\x48\0\x48\x01\x01\xAA\xBB\xCC"),
    );
    f.put("jfxx", &segment(0xE0, b"JFXX\0\x10\xFF\xD8\xFF\xD9"));
    f.put("exif", &exif(6));
    f.put("exif2", &exif(3));
    let mut x = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
    x.extend(xmp.as_bytes());
    f.put("xmp", &segment(0xE1, &x));
    let mut ext = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    ext.extend(b"0123456789ABCDEF0123456789ABCDEF");
    ext.extend(12u32.to_be_bytes());
    ext.extend(0u32.to_be_bytes());
    ext.extend(b"<!--ext-->  ");
    f.put("xmp_ext", &segment(0xE1, &ext));
    f.put("icc1", &segment(0xE2, b"ICC_PROFILE\0\x01\x02abcd"));
    f.put("icc2", &segment(0xE2, b"ICC_PROFILE\0\x02\x02efgh"));
    f.put("mpf", &mpf(0, 0, 0)); // patched below
    f.put(
        "iso",
        &segment(0xE2, b"urn:iso:std:iso:ts:21496:-1\0\0\0\0\0"),
    );
    f.put("jumbf", &segment(0xEB, b"JP\0\x01\0\0\0\x01jumb"));
    f.put("ducky", &segment(0xEC, b"Ducky\x01\0\x04\0\0\0\x50"));
    f.put(
        "ps",
        &segment(0xED, b"Photoshop 3.0\08BIM\x04\x04\0\0\0\0\0\0"),
    );
    f.put("adobe", &segment(0xEE, b"Adobe\0\x64\0\0\0\0\x01"));
    f.put("app15", &segment(0xEF, b"zzPRIVATE\0secret"));
    f.put("com", &segment(0xFE, b"a comment"));
    f.put("tem", &[0xFF, 0x01]);
    f.put("jpg0", &segment(0xF0, b"jpg0"));
    f.put("dhp", &segment(0xDE, &[8, 0, 8, 0, 16, 1, 1, 0x11, 0]));
    f.put("exp", &segment(0xDF, &[0x11]));
    f.put("sof5", &sof(0xC5, 16, 8));
    f.put("jpg", &segment(0xC8, b"x"));
    f.put("res", &segment(0x02, b"reserved"));
    f.put("stray", &[0x00, 0x11]);
    f.put("fill", &[0xFF, 0xFF]);
    f.put("dqt", &dqt());
    f.put("dht", &dht());
    f.put("dac", &segment(0xCC, &[0x10, 0x01]));
    f.put("dri", &segment(0xDD, &[0x00, 0x01]));
    f.put("sof0", &sof(0xC0, 16, 8));
    f.put("sos", &sos());
    f.put("scan", &[0x3F, 0xFF, 0xD0, 0x3F]);
    f.put("dnl", &segment(0xDC, &[0x00, 0x08]));
    f.put("sof1", &sof(0xC1, 16, 8));
    f.put("com2", &segment(0xFE, b"late"));
    f.put("eoi", &[0xFF, 0xD9]);
    let primary_len = f.bytes.len();
    f.put("gainmap", &gm);
    f.put("video", b"\0\0\0\x10ftypmp42\0\0\0\0");
    f.put("junk", b"junk!");
    // Samsung SEF trailer: one block, the SEFH directory, the footer.
    let mut block = vec![0, 0, 0x01, 0x0A];
    block.extend(14u32.to_le_bytes());
    block.extend(b"Image_UTC_Data1566395145421");
    let mut dir = b"SEFH".to_vec();
    dir.extend(107u32.to_le_bytes());
    dir.extend(1u32.to_le_bytes());
    dir.extend([0, 0, 0x01, 0x0A]);
    dir.extend((block.len() as u32).to_le_bytes());
    dir.extend((block.len() as u32).to_le_bytes());
    let dir_len = dir.len() as u32;
    f.put("sef_block", &block);
    f.put("sefh", &dir);
    let mut foot = dir_len.to_le_bytes().to_vec();
    foot.extend(b"SEFT");
    f.put("seft_foot", &foot);

    // Patch the MPF index now that the layout is known.
    let mpf_at = f.at("mpf").start as usize;
    let tiff = mpf_at + 4 + 4;
    let gm_at = f.at("gainmap").start as usize;
    let patched = mpf(primary_len as u32, gm.len() as u32, (gm_at - tiff) as u32);
    f.bytes[mpf_at..mpf_at + patched.len()].copy_from_slice(&patched);
    f
}

fn opts() -> Options {
    Options {
        auto_orient: false,
        gain_map_decoded: false,
        max_pixels: 0,
    }
}

/// `(kind, tag, range, disposition, label)` of every part, children after
/// their parent, in file order.
fn flat(inv: &Inventory) -> Vec<(K, PartTag, Range<u64>, D, Option<alloc::string::String>)> {
    fn walk(
        inv: &Inventory,
        parent: Option<zencodec::inventory::PartId>,
        out: &mut Vec<(K, PartTag, Range<u64>, D, Option<alloc::string::String>)>,
    ) {
        for id in inv.children(parent) {
            let p = inv.get(id).unwrap();
            out.push((
                p.kind,
                p.tag.clone(),
                p.range.clone(),
                p.disposition,
                p.label.as_ref().map(|l| l.to_string()),
            ));
            walk(inv, Some(id), out);
        }
    }
    let mut out = Vec::new();
    walk(inv, None, &mut out);
    out
}

#[test]
fn everything_fixture_part_list_is_pinned() {
    let f = everything();
    let inv = inventory(&f.bytes, opts()).unwrap();
    inv.validate().unwrap();
    let mk = |m: u8| PartTag::Marker(m);
    let l = |s: &str| Some(s.to_string());
    let r = |a: u64, b: u64| a..b;
    // The gain-map JPEG's own units, every consumed one Skipped (BaseOnly).
    let mut gain_map_parts = Vec::new();
    let mut at = f.at("gainmap").start;
    for (m, b) in tiny_units() {
        let range = at..at + b.len() as u64;
        at = range.end;
        let label = (m == 0xE1).then(|| "http://ns.adobe.com/xap/1.0/".to_string());
        let (kind, tag) = if m == 0 {
            (K::ScanData, PartTag::None)
        } else {
            (K::Segment, PartTag::Marker(m))
        };
        gain_map_parts.push((kind, tag, range, D::Skipped, label));
    }
    let mut expected = vec![
        (K::Segment, mk(0xD8), f.at("soi"), D::Structure, None),
        (
            K::Segment,
            mk(0xE0),
            f.at("jfif"),
            D::Metadata(M::Resolution),
            l("JFIF"),
        ),
        // The 1x1 RGB thumbnail at the end of the JFIF segment.
        (
            K::EmbeddedImage,
            PartTag::None,
            r(f.at("jfif").end - 3, f.at("jfif").end),
            D::Skipped,
            None,
        ),
        (K::Segment, mk(0xE0), f.at("jfxx"), D::Unknown, l("JFXX")),
        (
            K::Segment,
            mk(0xE1),
            f.at("exif"),
            D::Metadata(M::Exif),
            l("Exif"),
        ),
        (K::Segment, mk(0xE1), f.at("exif2"), D::Skipped, l("Exif")),
        (
            K::Segment,
            mk(0xE1),
            f.at("xmp"),
            D::Metadata(M::Xmp),
            l("http://ns.adobe.com/xap/1.0/"),
        ),
        (
            K::Segment,
            mk(0xE1),
            f.at("xmp_ext"),
            D::Metadata(M::Xmp),
            l("http://ns.adobe.com/xmp/extension/"),
        ),
        (
            K::Segment,
            mk(0xE2),
            f.at("icc1"),
            D::Metadata(M::Icc),
            l("ICC_PROFILE"),
        ),
        (
            K::Segment,
            mk(0xE2),
            f.at("icc2"),
            D::Metadata(M::Icc),
            l("ICC_PROFILE"),
        ),
        (K::Segment, mk(0xE2), f.at("mpf"), D::Structure, l("MPF")),
        (
            K::Segment,
            mk(0xE2),
            f.at("iso"),
            D::Unknown,
            l("urn:iso:std:iso:ts:21496:-1"),
        ),
        (K::Segment, mk(0xEB), f.at("jumbf"), D::Unknown, l("JP")),
        (
            K::Segment,
            mk(0xEC),
            f.at("ducky"),
            D::Unknown,
            l("Ducky\\x01"),
        ),
        (
            K::Segment,
            mk(0xED),
            f.at("ps"),
            D::Skipped,
            l("Photoshop 3.0"),
        ),
        (K::Segment, mk(0xEE), f.at("adobe"), D::Dropped, l("Adobe")),
        (
            K::Segment,
            mk(0xEF),
            f.at("app15"),
            D::Unknown,
            l("zzPRIVATE"),
        ),
        (
            K::Segment,
            mk(0xFE),
            f.at("com"),
            D::Skipped,
            l("a comment"),
        ),
        (K::Segment, mk(0x01), f.at("tem"), D::Skipped, None),
        (K::Segment, mk(0xF0), f.at("jpg0"), D::Skipped, None),
        (K::Segment, mk(0xDE), f.at("dhp"), D::Skipped, None),
        (K::Segment, mk(0xDF), f.at("exp"), D::Skipped, None),
        (K::Segment, mk(0xC5), f.at("sof5"), D::Skipped, None),
        (K::Segment, mk(0xC8), f.at("jpg"), D::Skipped, None),
        (K::Segment, mk(0x02), f.at("res"), D::Skipped, None),
        (K::Gap, PartTag::None, f.at("stray"), D::Malformed, None),
        // Both FFs before DQT's own marker prefix are fill.
        (K::Gap, PartTag::None, f.at("fill"), D::Padding, None),
        (K::Segment, mk(0xDB), f.at("dqt"), D::Structure, None),
        (K::Segment, mk(0xC4), f.at("dht"), D::Structure, None),
        (K::Segment, mk(0xCC), f.at("dac"), D::Structure, None),
        (K::Segment, mk(0xDD), f.at("dri"), D::Structure, None),
        (K::Segment, mk(0xC0), f.at("sof0"), D::Structure, None),
        (K::Segment, mk(0xDA), f.at("sos"), D::Structure, None),
        (K::ScanData, PartTag::None, f.at("scan"), D::ImageData, None),
        (K::Segment, mk(0xDC), f.at("dnl"), D::Structure, None),
        (K::Segment, mk(0xC1), f.at("sof1"), D::Skipped, None),
        (K::Segment, mk(0xFE), f.at("com2"), D::Skipped, l("late")),
        (K::Segment, mk(0xD9), f.at("eoi"), D::Structure, None),
        // The MPF gain map, walked: not decoded by a BaseOnly job.
        (
            K::EmbeddedImage,
            PartTag::Code(1),
            f.at("gainmap"),
            D::Skipped,
            l("MPF"),
        ),
    ];
    expected.extend(gain_map_parts);
    expected.extend([
        (
            K::EmbeddedImage,
            PartTag::None,
            f.at("video"),
            D::Skipped,
            l("MotionPhoto"),
        ),
        (K::Gap, PartTag::None, f.at("junk"), D::Trailing, None),
        (
            K::Trailer,
            PartTag::None,
            r(f.at("sef_block").start, f.at("seft_foot").end),
            D::Skipped,
            l("SEFT"),
        ),
        (
            K::Chunk,
            PartTag::Code(0x0A01),
            f.at("sef_block"),
            D::Skipped,
            l("Image_UTC_Data"),
        ),
        (
            K::Chunk,
            PartTag::FourCc(*b"SEFH"),
            r(f.at("sefh").start, f.at("seft_foot").end),
            D::Skipped,
            l("SEFH"),
        ),
    ]);
    let got = flat(&inv);
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
        assert_eq!(e, g, "part {i} differs\n{inv}");
    }
    assert_eq!(expected.len(), got.len(), "{inv}");
}

/// What the zencodec decode actually reports agrees with the dispositions.
#[test]
fn everything_fixture_dispositions_match_the_decoder() {
    let f = everything();
    let out = JpegDecoderConfig::new()
        .job()
        .decoder(alloc::borrow::Cow::Borrowed(&f.bytes[..]), &[])
        .unwrap()
        .decode()
        .unwrap();
    let info = out.info();
    let payload = |name: &str| {
        let r = f.at(name);
        f.bytes[r.start as usize + 4..r.end as usize].to_vec()
    };
    // The first EXIF segment, not the second.
    assert_eq!(
        info.embedded_metadata.exif.as_deref(),
        Some(&payload("exif")[..])
    );
    // Both ICC chunks, reassembled.
    assert_eq!(
        info.source_color.icc_profile.as_deref(),
        Some(&b"abcdefgh"[..])
    );
    // Standard XMP plus the extended chunk.
    let xmp = info.embedded_metadata.xmp.as_deref().unwrap();
    assert!(core::str::from_utf8(xmp).unwrap().ends_with("<!--ext-->  "));
    // JFIF density.
    let res = info.resolution.unwrap();
    assert_eq!((res.x, res.y), (72.0, 72.0));
    // EXIF orientation 6 from the first segment.
    assert_eq!(
        info.orientation,
        zencodec::Orientation::from_exif(6).unwrap()
    );
    assert_eq!((out.pixels().width(), out.pixels().rows()), (16, 8));
}

#[test]
fn everything_fixture_passes_check_inventory() {
    let f = everything();
    zencodec_testkit::check_inventory(JpegDecoderConfig::new(), &f.bytes).unwrap();
    zencodec_testkit::check_inventory(JpegDecoderConfig::new(), &tiny_jpeg()).unwrap();
}

/// With `GainMapRender::Components` the gain map is decoded, so its image is
/// gain-map metadata; its own XMP may be read for hdrgm parameters.
#[cfg(feature = "ultrahdr")]
#[test]
fn decoded_gain_map_is_gain_map_metadata() {
    let f = everything();
    let job = JpegDecoderConfig::new()
        .job()
        .with_gain_map_render(zencodec::GainMapRender::Components);
    let inv = job.inventory(&f.bytes).unwrap().unwrap();
    inv.validate().unwrap();
    let gm = f.at("gainmap");
    for p in inv.parts() {
        if p.range.start >= gm.start && p.range.end <= gm.end {
            assert_eq!(p.disposition, D::Metadata(M::GainMap), "{inv}");
        }
    }
}

/// The orientation walker is a copy of `find_exif_orientation`'s; the two
/// must agree on which segment carries the orientation.
#[test]
fn orientation_segment_agrees() {
    let f = everything();
    let mut cases: Vec<Vec<u8>> = vec![f.bytes.clone(), tiny_jpeg()];
    // First EXIF without an orientation tag, the second with one.
    let mut no_orient = vec![0xFF, 0xD8];
    no_orient.extend(segment(0xE1, b"Exif\0\0MM\0\x2a\0\0\0\x08\0\0\0\0\0\0"));
    no_orient.extend(exif(8));
    no_orient.extend(&tiny_jpeg()[2..]);
    cases.push(no_orient);
    // A single fill byte before the EXIF segment desynchronises the walk.
    let mut fill = vec![0xFF, 0xD8, 0xFF];
    fill.extend(exif(6));
    fill.extend(&tiny_jpeg()[2..]);
    cases.push(fill);
    for data in &cases {
        let at = exif_orientation_segment(data);
        let ours = at.and_then(|at| {
            let n = u16::from_be_bytes([data[at + 2], data[at + 3]]) as usize;
            crate::lossless::parse_exif_orientation(&data[at + 4..at + 2 + n])
        });
        assert_eq!(ours, crate::decode::find_exif_orientation(data));
    }
}

/// With auto-orient, an EXIF segment other than the first can drive the
/// pixel orientation: it is consumed for orientation only.
#[test]
fn auto_orient_reads_the_first_exif_with_an_orientation() {
    let mut data = vec![0xFF, 0xD8];
    let first = segment(0xE1, b"Exif\0\0MM\0\x2a\0\0\0\x08\0\0\0\0\0\0");
    data.extend(&first);
    data.extend(exif(8));
    data.extend(&tiny_jpeg()[2..]);
    let o = Options {
        auto_orient: true,
        ..opts()
    };
    let inv = inventory(&data, o).unwrap();
    inv.validate().unwrap();
    let at = |start: usize| {
        inv.parts()
            .iter()
            .find(|p| p.range.start == start as u64)
            .unwrap()
            .disposition
    };
    assert_eq!(at(2), D::Metadata(M::Exif));
    assert_eq!(at(2 + first.len()), D::Metadata(M::Orientation));
    let inv = inventory(&data, opts()).unwrap();
    assert_eq!(
        inv.parts()
            .iter()
            .find(|p| p.range.start == (2 + first.len()) as u64)
            .unwrap()
            .disposition,
        D::Skipped
    );
}

/// After the frame header the decoder reads TEM as if it carried a length
/// (decode/parser/mod.rs: TEM falls through to `skip_segment`); the
/// inventory follows the decoder, not the standard.
#[test]
fn tem_after_the_frame_header_is_read_as_a_length() {
    let mut data = vec![0xFF, 0xD8];
    data.extend(dqt());
    data.extend(dht());
    data.extend(sof(0xC0, 8, 8));
    let tem_at = data.len();
    data.extend([0xFF, 0x01, 0x00, 0x06, 0xAA, 0xBB, 0xCC, 0xDD]);
    data.extend(sos());
    data.push(0x3F);
    data.extend([0xFF, 0xD9]);
    let inv = inventory(&data, opts()).unwrap();
    inv.validate().unwrap();
    let tem = inv
        .parts()
        .iter()
        .find(|p| p.tag == PartTag::Marker(0x01))
        .unwrap();
    assert_eq!(tem.range, tem_at as u64..tem_at as u64 + 8);
    assert_eq!(tem.disposition, D::Skipped);
    // The decoder agrees: the image still decodes after skipping 6 bytes.
    JpegDecoderConfig::new()
        .job()
        .decoder(alloc::borrow::Cow::Borrowed(&data[..]), &[])
        .unwrap()
        .decode()
        .unwrap();
}

#[test]
fn junk_before_soi_and_no_soi() {
    let mut data = b"junk".to_vec();
    data.extend(tiny_jpeg());
    let inv = inventory(&data, opts()).unwrap();
    inv.validate().unwrap();
    let parts = flat(&inv);
    assert_eq!(
        (parts[0].0, parts[0].2.clone(), parts[0].3),
        (K::Gap, 0..4, D::Malformed)
    );
    // The decoder rejects the file, so nothing is consumed.
    assert!(
        inv.parts().iter().all(|p| !p.disposition.is_consumed()),
        "{inv}"
    );
    assert!(
        JpegDecoderConfig::new()
            .job()
            .decoder(alloc::borrow::Cow::Borrowed(&data[..]), &[])
            .unwrap()
            .decode()
            .is_err()
    );

    let inv = inventory(b"not a jpeg", opts()).unwrap();
    inv.validate().unwrap();
    assert_eq!(inv.parts().len(), 1);
    assert_eq!(inv.parts()[0].disposition, D::Malformed);
    inventory(&[], opts()).unwrap().validate().unwrap();
}

/// 12-bit frames: `probe()` reads the header, `decode()` refuses the
/// precision, so the scan is not image data.
#[test]
fn twelve_bit_frame_keeps_probe_metadata_only() {
    let mut data = vec![0xFF, 0xD8];
    data.extend(exif(1));
    data.extend(dqt());
    data.extend(dht());
    let mut s = sof(0xC1, 8, 8);
    s[4] = 12;
    data.extend(s);
    data.extend(sos());
    data.push(0x3F);
    data.extend([0xFF, 0xD9]);
    let inv = inventory(&data, opts()).unwrap();
    inv.validate().unwrap();
    let find = |t: u8| {
        inv.parts()
            .iter()
            .find(|p| p.tag == PartTag::Marker(t))
            .unwrap()
    };
    assert_eq!(find(0xE1).disposition, D::Metadata(M::Exif));
    assert_eq!(find(0xC1).disposition, D::Structure);
    assert_eq!(find(0xDA).disposition, D::Skipped);
    assert!(inv.parts().iter().all(|p| p.disposition != D::ImageData));
}

#[test]
fn lossless_frame_is_skipped() {
    let mut data = vec![0xFF, 0xD8];
    data.extend(dht());
    data.extend(sof(0xC3, 8, 8));
    data.extend(sos());
    data.push(0x3F);
    data.extend([0xFF, 0xD9]);
    let inv = inventory(&data, opts()).unwrap();
    inv.validate().unwrap();
    let sof3 = inv
        .parts()
        .iter()
        .find(|p| p.tag == PartTag::Marker(0xC3))
        .unwrap();
    assert_eq!(sof3.disposition, D::Skipped);
    assert!(sof3.detail.as_deref().unwrap().contains("SOF3"));
    assert!(
        inv.parts().iter().all(|p| !p.disposition.is_consumed()),
        "{inv}"
    );
}

/// Every prefix of the fixture still yields a valid inventory.
#[test]
fn every_prefix_validates() {
    let f = everything();
    for n in 0..=f.bytes.len() {
        let inv = inventory(&f.bytes[..n], opts()).unwrap();
        inv.validate()
            .unwrap_or_else(|e| panic!("prefix {n}: {e}\n{inv}"));
    }
}
