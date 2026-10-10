//! Review round 1 of the structural inventory (`DecodeJob::inventory`):
//! each test pins one finding of the review against a real decode of the
//! same bytes. Inputs are codec-corpus conformance files modified in memory.

#![cfg(feature = "zencodec")]

use std::borrow::Cow;

use zencodec::decode::{Decode as _, DecodeJob as _, DecodeOutput, DecoderConfig as _};
use zencodec::inventory::{Disposition as D, Inventory, Part};
use zenjpeg::{JpegDecodeJob, JpegDecoderConfig};

const TESTORIG: (&str, &str) = ("jpeg-conformance", "valid/testorig.jpg");
const ARI: (&str, &str) = ("jpeg-conformance", "valid/testimgari.jpg");
const UHDR: (&str, &str) = (
    "ultrahdr-conformance",
    "valid/jpeg/awesome-gain-maps/rgba_uhdr.jpg",
);
const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

fn load((set, rel): (&str, &str)) -> Vec<u8> {
    let dir = codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)")
        .get(set)
        .unwrap_or_else(|e| panic!("corpus.get({set}): {e}"));
    std::fs::read(dir.join(rel)).unwrap_or_else(|e| panic!("{set}/{rel}: {e}"))
}

fn seg(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, marker];
    v.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    v.extend_from_slice(payload);
    v
}

fn insert(data: &[u8], at: usize, ins: &[u8]) -> Vec<u8> {
    let mut v = data[..at].to_vec();
    v.extend_from_slice(ins);
    v.extend_from_slice(&data[at..]);
    v
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn be32(d: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

fn put32(d: &mut [u8], at: usize, v: u32) {
    d[at..at + 4].copy_from_slice(&v.to_be_bytes());
}

/// `(payload_start, payload_end)` of the first APP2 MPF segment.
fn mpf_payload(d: &[u8]) -> (usize, usize) {
    let p = find(d, b"MPF\0").expect("MPF segment");
    let l = (d[p - 2] as usize) << 8 | d[p - 1] as usize;
    (p, p - 2 + l)
}

fn job() -> JpegDecodeJob {
    JpegDecoderConfig::new().job()
}

fn inv(job: &JpegDecodeJob, data: &[u8]) -> Inventory {
    let inv = job.inventory(data).expect("inventory").expect("Some");
    inv.validate().expect("validate");
    inv
}

fn decode(job: JpegDecodeJob, data: &[u8]) -> Result<DecodeOutput, String> {
    job.decoder(Cow::Owned(data.to_vec()), &[])
        .map_err(|e| e.to_string())?
        .decode()
        .map_err(|e| e.to_string())
}

fn pixels(out: &DecodeOutput) -> Vec<u8> {
    let ps = out.pixels();
    (0..ps.rows()).flat_map(|y| ps.row(y).to_vec()).collect()
}

/// The innermost part covering `off`.
fn leaf_at(inv: &Inventory, off: usize) -> &Part {
    let parts = inv.parts();
    let depth = |p: &Part| {
        let mut d = 0;
        let mut q = p.parent;
        while let Some(id) = q {
            d += 1;
            q = parts[id.index()].parent;
        }
        d
    };
    parts
        .iter()
        .filter(|p| p.range.start <= off as u64 && (off as u64) < p.range.end)
        .max_by_key(|p| depth(p))
        .unwrap_or_else(|| panic!("no part covers {off}"))
}

fn show(p: &Part) -> String {
    format!(
        "{} {} {}..{} {} label={:?} detail={:?}",
        p.kind.name(),
        p.tag,
        p.range.start,
        p.range.end,
        p.disposition,
        p.label,
        p.detail
    )
}

/// Finding 2: a DAC entry is one definition per (class, table). Entries no
/// scan selects, and entries a later DAC replaces before the scan, are
/// `Dropped`; a segment that mixes both gets a child per run.
#[test]
fn dac_entries_are_definitions_per_table() {
    let orig = load(ARI);
    // testimgari.jpg: SOF9, its own DAC at 177..189 sets DC0, AC0, DC1, AC1,
    // and the one scan selects tables 0 and 1 for DC and AC.
    assert_eq!(&orig[177..179], &[0xFF, 0xCC]);
    let base = pixels(&decode(job(), &orig).unwrap());

    // Before the file's DAC: 37 entries for AC table 2 (no scan selects it)
    // and 11 for AC table 0 (the file's DAC redefines it before the scan).
    let mut body = Vec::new();
    for &b in b"SECRET-DAC-PAYLOAD-SECRET-DAC-PAYLOAD" {
        body.extend([0x12, b & 0x3F]);
    }
    for &b in b"OVERWRITTEN" {
        body.extend([0x10, b & 0x3F]);
    }
    let d = insert(&orig, 177, &seg(0xCC, &body));
    assert_eq!(pixels(&decode(job(), &d).unwrap()), base);
    let i = inv(&job(), &d);
    let p = leaf_at(&i, 177 + 6);
    assert_eq!(p.disposition, D::Dropped, "{}", show(p));
    assert_eq!(p.range, 177..177 + 4 + body.len() as u64, "{}", show(p));

    // After the file's DAC: five unused AC-table-2 entries, then AC table 0
    // with the file's own Kx (pixels unchanged), which replaces the file's
    // AC0 entry before the scan.
    let mut body = Vec::new();
    for _ in 0..5 {
        body.extend([0x12, 0x07]);
    }
    body.extend([0x10, 0x05]);
    let d = insert(&orig, 189, &seg(0xCC, &body));
    assert_eq!(pixels(&decode(job(), &d).unwrap()), base);
    let i = inv(&job(), &d);
    let planted = leaf_at(&i, 189 + 4);
    assert_eq!(planted.disposition, D::Dropped, "{}", show(planted));
    assert_eq!(planted.range, 193..203, "{}", show(planted));
    let used = leaf_at(&i, 203);
    assert_eq!(used.disposition, D::Structure, "{}", show(used));
    assert_eq!(used.range, 203..205, "{}", show(used));
    // The file's AC0 entry (bytes 183..185) is replaced before the scan.
    let replaced = leaf_at(&i, 183);
    assert_eq!(replaced.disposition, D::Dropped, "{}", show(replaced));
    assert_eq!(replaced.range, 183..185, "{}", show(replaced));
    assert_eq!(leaf_at(&i, 181).disposition, D::Structure);
}

/// Finding 11: a flood of repeated gaps or standalone markers carries one
/// detail on the first part, not one string per part.
#[test]
fn repeated_gaps_carry_one_detail() {
    let orig = load(TESTORIG);
    let mut d = vec![0xFF, 0xD8];
    for _ in 0..1000 {
        d.extend([0xFF, 0x01]);
    }
    d.extend(&orig[2..]);
    let i = inv(&job(), &d);
    let tem: Vec<&Part> = i
        .parts()
        .iter()
        .filter(|p| p.range.end - p.range.start == 2 && p.range.start < 2002 && p.range.start >= 2)
        .collect();
    assert_eq!(tem.len(), 1000);
    let with_detail: Vec<&&Part> = tem.iter().filter(|p| p.detail.is_some()).collect();
    assert_eq!(
        with_detail.len(),
        1,
        "{:?}",
        with_detail.iter().map(|p| show(p)).collect::<Vec<_>>()
    );
    let note = with_detail[0].detail.as_deref().unwrap_or("");
    assert!(note.contains("999 more"), "{note}");
}

/// Finding 3: bytes `parse_mpf_directory` never reads are `Unreferenced`
/// children of the MPF segment, wherever they sit: here 124 bytes between
/// the MP IFD and the MP entry array, which the decoder still follows.
#[test]
fn mpf_index_holes_are_unreferenced() {
    let orig = load(UHDR);
    let (p0, _) = mpf_payload(&orig);
    let tiff = p0 + 4;
    let ifd = tiff + be32(&orig, tiff + 4) as usize;
    let n = u16::from_be_bytes([orig[ifd], orig[ifd + 1]]) as usize;
    let b002 = (0..n)
        .map(|i| ifd + 2 + 12 * i)
        .find(|&e| orig[e] == 0xB0 && orig[e + 1] == 0x02)
        .expect("B002 entry");
    let entries_rel = be32(&orig, b002 + 8) as usize;
    let entries = tiff + entries_rel;
    let gap: Vec<u8> = b"SECRET-MPF-GAP-0123456789abcdef".repeat(4);
    let k = gap.len();
    let mut d = insert(&orig, entries, &gap);
    // The segment length, the B002 value offset, the primary's size and the
    // gain map's offset all move by `k`.
    let seglen = u16::from_be_bytes([d[p0 - 2], d[p0 - 1]]) as usize + k;
    d[p0 - 2..p0].copy_from_slice(&(seglen as u16).to_be_bytes());
    put32(&mut d, b002 + 8, (entries_rel + k) as u32);
    let e0 = entries + k;
    let s0 = be32(&d, e0 + 4);
    put32(&mut d, e0 + 4, s0 + k as u32);
    let o1 = be32(&d, e0 + 16 + 8);
    put32(&mut d, e0 + 16 + 8, o1 + k as u32);

    let out = decode(job(), &d).unwrap();
    let ex = out
        .extras::<zenjpeg::decode::DecodedExtras>()
        .expect("native extras");
    assert_eq!(
        ex.secondary_images().len(),
        1,
        "the decoder follows the index"
    );

    let i = inv(&job(), &d);
    for at in [entries, entries + 3, entries + k - 1] {
        let p = leaf_at(&i, at);
        assert_eq!(p.disposition, D::Unreferenced, "{}", show(p));
    }
    // The TIFF magic after the byte-order mark is never read either.
    let p = leaf_at(&i, tiff + 2);
    assert_eq!(p.disposition, D::Unreferenced, "{}", show(p));
    // The IFD offset and the MP entries are read.
    assert_eq!(leaf_at(&i, tiff + 4).disposition, D::Structure);
    assert_eq!(leaf_at(&i, e0).disposition, D::Structure);
}

/// Finding 12: an MPF IFD offset near `u32::MAX` would overflow a 32-bit
/// `usize` inside `parse_mpf_directory`; the walker rejects it first.
#[test]
fn mpf_ifd_offset_near_u32_max_is_rejected_before_parsing() {
    let mut d = load(UHDR);
    let (p0, _) = mpf_payload(&d);
    put32(&mut d, p0 + 8, u32::MAX);
    let i = inv(&job(), &d);
    let p = leaf_at(&i, p0);
    assert_eq!(p.disposition, D::Dropped, "{}", show(p));
}

/// Finding 8: an extended-XMP chunk's GUID and full length are never read
/// (`reassemble_xmp` reads the offset and the data only).
#[test]
fn extended_xmp_guid_and_length_are_dropped() {
    let orig = load(TESTORIG);
    let mut x = XMP_NS.to_vec();
    x.extend(b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>");
    let mut ext = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    let guid_at = ext.len();
    ext.extend(b"SECRET-GUID-0123456789ABCDEFGHIJ");
    ext.extend(9999u32.to_be_bytes());
    ext.extend(0u32.to_be_bytes());
    ext.extend(b"<!--x-->");
    let mut ins = seg(0xE1, &x);
    let ext_payload = 20 + ins.len() + 4;
    ins.extend(seg(0xE1, &ext));
    let d = insert(&orig, 20, &ins);

    let out = decode(job(), &d).unwrap();
    let xmp = out.info().embedded_metadata.xmp.clone().expect("xmp");
    let xmp = String::from_utf8(xmp.to_vec()).unwrap();
    assert!(
        xmp.ends_with("<!--x-->") && !xmp.contains("SECRET"),
        "{xmp}"
    );

    let i = inv(&job(), &d);
    for at in [ext_payload + guid_at, ext_payload + guid_at + 35] {
        let p = leaf_at(&i, at);
        assert_eq!(p.disposition, D::Dropped, "{}", show(p));
    }
    // The offset field and the data are read.
    for at in [ext_payload + guid_at + 36, ext_payload + guid_at + 40] {
        let p = leaf_at(&i, at);
        assert_eq!(
            p.disposition,
            D::Metadata(zencodec::inventory::MetadataKind::Xmp),
            "{}",
            show(p)
        );
    }
}
