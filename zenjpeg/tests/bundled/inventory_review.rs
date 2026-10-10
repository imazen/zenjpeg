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
const PROG: (&str, &str) = ("jpeg-conformance", "valid/progressive3.jpg");
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

/// Finding 4: one fill byte desynchronises `find_exif_orientation` into a
/// COM payload holding an APP1 EXIF lookalike. The COM stays `Skipped`; only
/// the 12-byte orientation entry the decoder applies is consumed.
#[test]
fn orientation_lookalike_inside_another_part_is_a_field() {
    use zencodec::OrientationHint;
    let orig = load(TESTORIG);
    let mut exif = b"Exif\0\0MM\0\x2a\0\0\0\x08\0\x01".to_vec();
    exif.extend([0x01, 0x12, 0x00, 0x03, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0]);
    let mut com = b"SECRET-COMMENT-TEXT ".to_vec();
    com.extend(seg(0xE1, &exif));
    com.extend(b" MORE-SECRET-TEXT");
    let mut ins = vec![0xFF]; // one fill byte
    ins.extend(seg(0xFE, &com));
    let d = insert(&orig, 2, &ins);
    let correct = || job().with_orientation(OrientationHint::Correct);
    let out = decode(correct(), &d).unwrap();
    assert_eq!(
        (out.width(), out.height()),
        (149, 227),
        "orientation 6 applied"
    );

    let i = inv(&correct(), &d);
    // COM at 3: marker, length, then the text; the lookalike APP1 at 27.
    let text = leaf_at(&i, 3 + 4 + 2);
    assert_eq!(text.disposition, D::Skipped, "{}", show(text));
    let lookalike = 3 + 4 + 20;
    let entry = lookalike + 4 + 6 + 8 + 2;
    let p = leaf_at(&i, entry);
    assert_eq!(
        p.disposition,
        D::Metadata(zencodec::inventory::MetadataKind::Orientation),
        "{}",
        show(p)
    );
    assert_eq!(p.range, entry as u64..entry as u64 + 12, "{}", show(p));
    assert_eq!(leaf_at(&i, entry + 12).disposition, D::Skipped);
}

/// The scan-data part's disposition (every test file here has one scan
/// part or more; the first is enough).
fn scan_disposition(i: &Inventory) -> D {
    i.parts()
        .iter()
        .find(|p| p.kind == zencodec::inventory::PartKind::ScanData)
        .expect("scan data")
        .disposition
}

/// Finding 7: the job's policy, inner strictness and dimension limits
/// decide whether the decode fails, and the inventory follows them.
#[test]
fn job_policy_strictness_and_limits_are_followed() {
    use zencodec::decode::DecodePolicy;
    let orig = load(TESTORIG);

    // One stray byte between APP0 and DQT: a warning, an error when Strict.
    let d = insert(&orig, 20, &[0x00]);
    let strict = || job().with_policy(DecodePolicy::strict());
    assert!(decode(job(), &d).is_ok());
    assert!(decode(strict(), &d).is_err());
    assert_eq!(scan_disposition(&inv(&job(), &d)), D::ImageData);
    assert_eq!(scan_disposition(&inv(&strict(), &d)), D::Skipped);
    // `allow_truncated: false` alone also makes the decode Strict.
    let mut no_trunc = DecodePolicy::none();
    no_trunc.allow_truncated = Some(false);
    assert!(decode(job().with_policy(no_trunc), &d).is_err());
    assert_eq!(
        scan_disposition(&inv(&job().with_policy(no_trunc), &d)),
        D::Skipped
    );

    // APP15 with length 1 before the frame header: Permissive skips it.
    let d = insert(&orig, 20, &[0xFF, 0xEF, 0x00, 0x01]);
    let mut cfg = JpegDecoderConfig::new();
    let permissive = cfg.inner().clone().permissive();
    *cfg.inner_mut() = permissive;
    assert!(decode(job(), &d).is_err());
    assert!(decode(cfg.clone().job(), &d).is_ok());
    assert_eq!(scan_disposition(&inv(&job(), &d)), D::Skipped);
    assert_eq!(scan_disposition(&inv(&cfg.job(), &d)), D::ImageData);

    // A zero quantization value: clamped, an error when Strict.
    let mut d = orig.clone();
    assert_eq!(&d[20..22], &[0xFF, 0xDB]);
    d[20 + 5 + 10] = 0;
    assert!(decode(job(), &d).is_ok());
    assert!(decode(strict(), &d).is_err());
    assert_eq!(scan_disposition(&inv(&job(), &d)), D::ImageData);
    assert_eq!(scan_disposition(&inv(&strict(), &d)), D::Skipped);

    // allow_progressive = false refuses a progressive frame in decode();
    // probe() still reads the header.
    let p = load(PROG);
    let mut pol = DecodePolicy::none();
    pol.allow_progressive = Some(false);
    assert!(decode(job().with_policy(pol), &p).is_err());
    let i = inv(&job().with_policy(pol), &p);
    assert_eq!(scan_disposition(&i), D::Skipped);
    assert!(
        i.parts()[0].disposition.is_consumed(),
        "SOI: probe reads it"
    );

    // max_width below the frame width.
    let width = job().probe(&p).unwrap().width;
    let lim = zencodec::ResourceLimits::none().with_max_width(width - 1);
    assert!(decode(job().with_limits(lim.clone()), &p).is_err());
    assert_eq!(
        scan_disposition(&inv(&job().with_limits(lim), &p)),
        D::Skipped
    );
}

/// Finding 6: a malformed segment after the only scan of a sequential
/// frame without a restart interval certainly stops the decode.
#[test]
fn failure_after_a_sequential_scan_is_certain() {
    let orig = load(TESTORIG);
    let eoi = orig.len() - 2;
    for ins in [vec![0xFF, 0xEF, 0x00, 0x01], seg(0xC4, &[0x20; 17])] {
        let d = insert(&orig, eoi, &ins);
        assert!(decode(job(), &d).is_err());
        let i = inv(&job(), &d);
        assert_eq!(scan_disposition(&i), D::Skipped);
        // Only what probe() reads before the frame header stays consumed.
        assert_eq!(&orig[158..160], &[0xFF, 0xC0]);
        let sof_end = 160 + u16::from_be_bytes([orig[160], orig[161]]) as u64;
        for p in i.parts().iter().filter(|p| p.range.end > sof_end) {
            assert!(!p.disposition.is_consumed(), "{}", show(p));
        }
    }
}

fn decode_f32(data: &[u8]) -> Vec<u8> {
    let out = job()
        .decoder(
            Cow::Owned(data.to_vec()),
            &[zenpixels::PixelDescriptor::RGBF32_LINEAR],
        )
        .unwrap()
        .decode()
        .unwrap();
    pixels(&out)
}

fn adobe(transform: u8) -> Vec<u8> {
    seg(
        0xEE,
        &[b'A', b'd', b'o', b'b', b'e', 0, 0x64, 0, 0, 0, 0, transform],
    )
}

/// Finding 5: a baseline 4:2:0 frame decoded to the default u8 RGB output
/// dequantises and converts colour during its scan; an f32 output does
/// both at the end. Post-scan DQT and APP14 matter only to the second, and
/// the parts say so.
#[test]
fn post_scan_tables_depend_on_the_output_path() {
    let orig = load(TESTORIG);
    let eoi = orig.len() - 2;
    let base = pixels(&decode(job(), &orig).unwrap());

    // A DQT after the scan redefining table 0.
    let mut q = vec![0x00];
    q.extend([1u8; 64]);
    let d = insert(&orig, eoi, &seg(0xDB, &q));
    assert_eq!(pixels(&decode(job(), &d).unwrap()), base);
    assert_ne!(decode_f32(&d), decode_f32(&orig));
    let i = inv(&job(), &d);
    let late = leaf_at(&i, eoi + 5);
    assert_eq!(late.disposition, D::Dropped, "{}", show(late));
    assert!(
        late.detail
            .as_deref()
            .unwrap_or("")
            .contains("coefficients"),
        "{}",
        show(late)
    );
    // testorig's own table 0 (DQT at 20) is what the default output uses.
    let own = leaf_at(&i, 20 + 5);
    assert_eq!(own.disposition, D::Structure, "{}", show(own));
    assert!(
        own.detail
            .as_deref()
            .unwrap_or("")
            .contains("default u8 RGB"),
        "{}",
        show(own)
    );

    // An APP14 after the scan.
    let d = insert(&orig, eoi, &adobe(0));
    assert_eq!(pixels(&decode(job(), &d).unwrap()), base);
    assert_ne!(decode_f32(&d), decode_f32(&orig));
    let i = inv(&job(), &d);
    let p = leaf_at(&i, eoi + 4);
    assert_eq!(p.disposition, D::Dropped, "{}", show(p));

    // APP14 t=0 after SOI, t=1 before EOI: the early one is applied.
    let early = insert(&orig, 2, &adobe(0));
    let early_px = pixels(&decode(job(), &early).unwrap());
    let both = insert(&early, early.len() - 2, &adobe(1));
    assert_eq!(pixels(&decode(job(), &both).unwrap()), early_px);
    let i = inv(&job(), &both);
    let first = leaf_at(&i, 4);
    assert_eq!(
        first.disposition,
        D::Metadata(zencodec::inventory::MetadataKind::Colour),
        "{}",
        show(first)
    );
    assert!(!first.detail.as_deref().unwrap_or("").contains("superseded"));
    let second = leaf_at(&i, both.len() - 2 - 16 + 4);
    assert_eq!(second.disposition, D::Dropped, "{}", show(second));
}

/// Finding 5, control: a progressive frame always converts at the end, so
/// a DQT after its scans is the table in effect.
#[test]
fn post_scan_dqt_of_a_progressive_frame_is_used() {
    let orig = load(PROG);
    let eoi = orig.len() - 2;
    let mut q = vec![0x00];
    q.extend([1u8; 64]);
    let d = insert(&orig, eoi, &seg(0xDB, &q));
    assert_ne!(
        pixels(&decode(job(), &d).unwrap()),
        pixels(&decode(job(), &orig).unwrap())
    );
    let i = inv(&job(), &d);
    let p = leaf_at(&i, eoi + 5);
    assert_eq!(p.disposition, D::Structure, "{}", show(p));
}

fn render_job(r: zencodec::GainMapRender) -> JpegDecodeJob {
    job().with_gain_map_render(r)
}

const RECONSTRUCT: zencodec::GainMapRender = zencodec::GainMapRender::ReconstructHdr {
    target_headroom: None,
};

/// Parts the inventory reports as the decoded gain map.
fn gain_map_parts(i: &Inventory) -> Vec<&Part> {
    i.parts()
        .iter()
        .filter(|p| p.disposition == D::Metadata(zencodec::inventory::MetadataKind::GainMap))
        .collect()
}

fn decoded_gain_map(out: &DecodeOutput) -> Option<(u32, u32)> {
    out.extras::<zencodec::decode::DecodedGainMap>()
        .map(|g| (g.width(), g.height()))
}

/// End of the primary image: a plain marker walk, enough for these files.
fn primary_eoi_end(d: &[u8]) -> usize {
    let mut i = 2;
    while i + 1 < d.len() {
        if d[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = d[i + 1];
        if m == 0xFF {
            i += 1;
            continue;
        }
        if m == 0x00 || (0xD0..=0xD7).contains(&m) {
            i += 2;
            continue;
        }
        if m == 0xD9 {
            return i + 2;
        }
        let l = (d[i + 2] as usize) << 8 | d[i + 3] as usize;
        i += 2 + l;
        if m == 0xDA {
            while i + 1 < d.len()
                && !(d[i] == 0xFF && d[i + 1] != 0 && !(0xD0..=0xD7).contains(&d[i + 1]))
            {
                i += 1;
            }
        }
    }
    panic!("no EOI")
}

/// Finding 1, the base case: rgba_uhdr.jpg keeps its hdrgm parameters in
/// the gain-map image's XMP. Components and ReconstructHdr decode the gain
/// map, and read that XMP packet; BaseOnly does neither.
#[cfg(feature = "ultrahdr")]
#[test]
fn gain_map_and_its_xmp_follow_the_decode() {
    let d = load(UHDR);
    let eoi = primary_eoi_end(&d);
    let gm_xmp = eoi + 2; // the gain map's APP1 right after its SOI
    assert_eq!(&d[gm_xmp..gm_xmp + 2], &[0xFF, 0xE1]);
    for r in [zencodec::GainMapRender::Components, RECONSTRUCT] {
        let out = decode(render_job(r), &d).unwrap();
        if r == zencodec::GainMapRender::Components {
            assert!(decoded_gain_map(&out).is_some());
        }
        let i = inv(&render_job(r), &d);
        let p = leaf_at(&i, eoi);
        assert_eq!(
            p.disposition,
            D::Metadata(zencodec::inventory::MetadataKind::GainMap),
            "{}",
            show(p)
        );
        let x = leaf_at(&i, gm_xmp + 40);
        assert_eq!(
            x.disposition,
            D::Metadata(zencodec::inventory::MetadataKind::GainMap),
            "{}",
            show(x)
        );
    }
    let i = inv(&job(), &d);
    assert!(gain_map_parts(&i).is_empty());
}

/// Finding 1 (review r02): with both XMP namespaces corrupted there are no
/// parameters, so Components decodes no gain map and ReconstructHdr
/// decodes the base only; the inventory claims no gain map.
#[cfg(feature = "ultrahdr")]
#[test]
fn no_hdrgm_parameters_no_gain_map() {
    let mut d = load(UHDR);
    let mut at = 0;
    while let Some(j) = find(&d[at..], XMP_NS) {
        let j = at + j;
        d[j + 21] = b'q'; // "xap" -> "xaq"
        at = j + 1;
    }
    assert!(find(&d, XMP_NS).is_none());
    for r in [zencodec::GainMapRender::Components, RECONSTRUCT] {
        let out = decode(render_job(r), &d).unwrap();
        assert!(decoded_gain_map(&out).is_none());
        let i = inv(&render_job(r), &d);
        assert!(gain_map_parts(&i).is_empty(), "{r:?}: {i}");
    }
}

/// Finding 1 (review r03): when the primary XMP carries non-default hdrgm
/// values, the gain map's own XMP is never read (changing it leaves the
/// decoded metadata unchanged) and is Dropped.
#[cfg(feature = "ultrahdr")]
#[test]
fn gain_map_xmp_unread_when_the_primary_has_parameters() {
    let orig = load(UHDR);
    let anchor = b"hdrgm:Version=\"1.0\"";
    let at = find(&orig, anchor).unwrap() + anchor.len();
    let add = b" hdrgm:GainMapMax=\"2.5\" hdrgm:HDRCapacityMax=\"2.5\"";
    let mut a = insert(&orig, at, add);
    // The primary XMP APP1 starts at 2; the MPF offsets after it move too,
    // but they are relative to the MPF header, which moves with them.
    let l = u16::from_be_bytes([a[4], a[5]]) as usize + add.len();
    a[4..6].copy_from_slice(&(l as u16).to_be_bytes());
    let gm = primary_eoi_end(&a);
    let mut b = a.clone();
    let rel = find(&b[gm..], b"GainMapMax=\"5.62238\"").unwrap();
    b[gm + rel + 12..gm + rel + 19].copy_from_slice(b"1.11111");
    let c = zencodec::GainMapRender::Components;
    let ma = format!(
        "{:?}",
        decode(render_job(c), &a)
            .unwrap()
            .extras::<zencodec::decode::DecodedGainMap>()
            .unwrap()
            .metadata
    );
    let mb = format!(
        "{:?}",
        decode(render_job(c), &b)
            .unwrap()
            .extras::<zencodec::decode::DecodedGainMap>()
            .unwrap()
            .metadata
    );
    assert_eq!(ma, mb, "the gain-map XMP is not read");
    let i = inv(&render_job(c), &b);
    let x = leaf_at(&i, gm + rel);
    assert_eq!(x.disposition, D::Dropped, "{}", show(x));
    assert!(!gain_map_parts(&i).is_empty());
}

/// Finding 9 (review r04): a crafted SEF footer whose one entry points back
/// to the primary EOI no longer swallows the decoded gain map: MPF images
/// are placed first, SEF blocks only in space still free.
#[cfg(feature = "ultrahdr")]
#[test]
fn sef_trailer_does_not_swallow_the_gain_map() {
    let mut d = load(UHDR);
    let eoi = primary_eoi_end(&d);
    let dir_pos = d.len();
    let noff = (dir_pos - eoi) as u32;
    let mut dir = b"SEFH".to_vec();
    dir.extend(107u32.to_le_bytes());
    dir.extend(1u32.to_le_bytes());
    dir.extend([0, 0, 0x01, 0x0A]);
    dir.extend(noff.to_le_bytes());
    dir.extend(8u32.to_le_bytes());
    let dl = dir.len() as u32;
    d.extend(dir);
    d.extend(dl.to_le_bytes());
    d.extend(b"SEFT");
    let c = zencodec::GainMapRender::Components;
    assert!(decoded_gain_map(&decode(render_job(c), &d).unwrap()).is_some());
    let i = inv(&render_job(c), &d);
    let p = leaf_at(&i, eoi + 100);
    assert_eq!(
        p.disposition,
        D::Metadata(zencodec::inventory::MetadataKind::GainMap),
        "{}",
        show(p)
    );
}

/// Finding 1 (review r05): the decoder takes the first extracted Undefined
/// MPF image wherever it lies. Here entry 1 points into an APP15 inside the
/// primary (holding the real gain map) and entry 2 after EOI (another
/// JPEG): entry 1 is the gain map, nested in the APP15; entry 2 is not.
#[cfg(feature = "ultrahdr")]
#[test]
fn the_first_extracted_undefined_entry_is_the_gain_map() {
    let orig = load(UHDR);
    let eoi = primary_eoi_end(&orig);
    let gm = orig[eoi..].to_vec();
    let other = load(TESTORIG);
    let (p0, pend) = mpf_payload(&orig);
    let mpf_seg_start = p0 - 4;
    let tiff = p0 + 4;
    let build = |rel1: u32, rel2: u32, primary_len: u32| {
        let mut p = b"MPF\0MM\0\x2a\0\0\0\x08".to_vec();
        p.extend([0, 3]);
        p.extend([0xB0, 0x00, 0, 7, 0, 0, 0, 4]);
        p.extend(b"0100");
        p.extend([0xB0, 0x01, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3]);
        p.extend([0xB0, 0x02, 0, 7, 0, 0, 0, 48, 0, 0, 0, 50]);
        p.extend([0, 0, 0, 0]);
        p.extend([0x20, 0x03, 0x00, 0x00]);
        p.extend(primary_len.to_be_bytes());
        p.extend([0, 0, 0, 0, 0, 0, 0, 0]);
        p.extend([0, 0, 0, 0]);
        p.extend((gm.len() as u32).to_be_bytes());
        p.extend(rel1.to_be_bytes());
        p.extend([0, 0, 0, 0]);
        p.extend([0, 0, 0, 0]);
        p.extend((other.len() as u32).to_be_bytes());
        p.extend(rel2.to_be_bytes());
        p.extend([0, 0, 0, 0]);
        seg(0xE2, &p)
    };
    let app15 = seg(0xEF, &gm);
    let mpf_len = build(0, 0, 0).len();
    let app15_payload = mpf_seg_start + mpf_len + 4;
    let primary_len = (eoi - (pend - mpf_seg_start) + mpf_len + app15.len()) as u32;
    let after = primary_len as usize;
    let mut d = orig[..mpf_seg_start].to_vec();
    d.extend(build(
        (app15_payload - tiff) as u32,
        (after - tiff) as u32,
        primary_len,
    ));
    d.extend(&app15);
    d.extend(&orig[pend..eoi]);
    assert_eq!(d.len(), after);
    d.extend(&other);

    let c = zencodec::GainMapRender::Components;
    let dims = decoded_gain_map(&decode(render_job(c), &d).unwrap());
    assert!(dims.is_some() && dims != Some((227, 149)), "{dims:?}");
    let i = inv(&render_job(c), &d);
    let inside = leaf_at(&i, app15_payload + 100);
    assert_eq!(
        inside.disposition,
        D::Metadata(zencodec::inventory::MetadataKind::GainMap),
        "{}",
        show(inside)
    );
    let outside = leaf_at(&i, after + 100);
    assert_ne!(
        outside.disposition,
        D::Metadata(zencodec::inventory::MetadataKind::GainMap),
        "{}",
        show(outside)
    );
}

/// Finding 1 (review r20): an extended-XMP chunk inside the gain map is
/// never read (`extract_xmp_from_jpeg` reads the standard packet only).
#[cfg(feature = "ultrahdr")]
#[test]
fn gain_map_extended_xmp_is_dropped() {
    let orig = load(UHDR);
    let eoi = primary_eoi_end(&orig);
    let (p0, _) = mpf_payload(&orig);
    let e1 = p0 + 4 + 50 + 16;
    let mut ext = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    ext.extend(b"0123456789ABCDEF0123456789ABCDEF");
    ext.extend(20u32.to_be_bytes());
    ext.extend(0u32.to_be_bytes());
    let mk = |body: &[u8]| {
        let mut e = ext.clone();
        e.extend(body);
        let s = seg(0xE1, &e);
        let gx = eoi + 2;
        let gl = u16::from_be_bytes([orig[gx + 2], orig[gx + 3]]) as usize;
        let at = gx + 2 + gl;
        let mut d = insert(&orig, at, &s);
        let s1 = be32(&d, e1 + 4);
        put32(&mut d, e1 + 4, s1 + s.len() as u32);
        (d, at)
    };
    let (a, at) = mk(b"<!--SECRET-ONE-----");
    let (b, _) = mk(b"<!--SECRET-TWO-----");
    let c = zencodec::GainMapRender::Components;
    let oa = decode(render_job(c), &a).unwrap();
    let ob = decode(render_job(c), &b).unwrap();
    let meta = |o: &DecodeOutput| {
        format!(
            "{:?}",
            o.extras::<zencodec::decode::DecodedGainMap>()
                .map(|g| &g.metadata)
        )
    };
    assert_eq!(meta(&oa), meta(&ob));
    let i = inv(&render_job(c), &a);
    let p = leaf_at(&i, at + 4 + 40);
    assert_eq!(p.disposition, D::Dropped, "{}", show(p));
}

/// Decoder issue (filed separately): Components fails the whole decode on
/// a plain JPEG whose XMP has no hdrgm. Whatever the decoder does, the
/// inventory must agree on whether image data reaches the caller.
#[cfg(feature = "ultrahdr")]
#[test]
fn components_on_plain_xmp_agrees_with_the_decoder() {
    let orig = load(TESTORIG);
    let mut x = XMP_NS.to_vec();
    x.extend(b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>");
    let d = insert(&orig, 20, &seg(0xE1, &x));
    let c = zencodec::GainMapRender::Components;
    let ok = decode(render_job(c), &d).is_ok();
    let i = inv(&render_job(c), &d);
    assert_eq!(scan_disposition(&i) == D::ImageData, ok);
}

/// ISO 21496-1: no zencodec decode or probe path reads the APP2 payload.
/// Two payloads give identical BaseOnly, Components and ReconstructHdr
/// output, and the segment stays Unknown.
#[cfg(feature = "ultrahdr")]
#[test]
fn iso_21496_payload_is_not_read() {
    let orig = load(UHDR);
    let iso = |fill: u8| {
        let mut p = b"urn:iso:std:iso:ts:21496:-1\0".to_vec();
        p.extend([0, 0, 0, 0]);
        p.extend([fill; 40]);
        insert(&orig, 2, &seg(0xE2, &p))
    };
    let (a, b) = (iso(b'A'), iso(b'B'));
    for r in [
        zencodec::GainMapRender::BaseOnly,
        zencodec::GainMapRender::Components,
        RECONSTRUCT,
    ] {
        let (oa, ob) = (
            decode(render_job(r), &a).unwrap(),
            decode(render_job(r), &b).unwrap(),
        );
        assert_eq!(pixels(&oa), pixels(&ob), "{r:?}");
        assert_eq!(
            format!("{:?}", oa.info()),
            format!("{:?}", ob.info()),
            "{r:?}"
        );
        let gm = |o: &DecodeOutput| {
            o.extras::<zencodec::decode::DecodedGainMap>()
                .map(|g| (format!("{:?}", g.metadata), pixels_of(g)))
        };
        assert_eq!(gm(&oa), gm(&ob), "{r:?}");
        let i = inv(&render_job(r), &a);
        let p = leaf_at(&i, 2 + 10);
        assert_eq!(p.disposition, D::Unknown, "{}", show(p));
    }
}

#[cfg(feature = "ultrahdr")]
fn pixels_of(g: &zencodec::decode::DecodedGainMap) -> Vec<u8> {
    let ps = g.pixels.as_slice();
    (0..ps.rows()).flat_map(|y| ps.row(y).to_vec()).collect()
}

/// The container probe (`zenjpeg::container::probe`) and the inventory
/// agree on where the XMP, MPF and ISO segments and the embedded images are,
/// on every Ultra HDR conformance file.
#[test]
fn container_probe_agrees_with_the_inventory() {
    use zenjpeg::container::{Wants, probe};
    let dir = codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)")
        .get("ultrahdr-conformance")
        .expect("ultrahdr-conformance");
    let mut files = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("jpg")) {
                files.push(p);
            }
        }
    }
    assert!(files.len() > 20, "{}", files.len());
    let wants =
        Wants::IMAGE_RANGES | Wants::XMP_LOCATION | Wants::MPF_LOCATION | Wants::ISO_GAINMAP;
    let mut checked = 0;
    for f in &files {
        let d = std::fs::read(f).unwrap();
        let pr = probe(&d, wants);
        let i = inv(&job(), &d);
        let top_segment = |r: &std::ops::Range<u32>| {
            i.parts().iter().find(|p| {
                p.parent.is_none()
                    && p.kind == zencodec::inventory::PartKind::Segment
                    && p.range.start <= r.start as u64
                    && r.end as u64 <= p.range.end
            })
        };
        for (what, r) in [
            ("xmp", pr.xmp()),
            ("mpf", pr.mpf()),
            ("iso", pr.iso_gainmap()),
        ] {
            if let Some(r) = r {
                let p = top_segment(r).unwrap_or_else(|| {
                    panic!("{}: probe {what} {r:?} has no segment", f.display())
                });
                // Where the decode succeeds, the probe's XMP is reported.
                if what == "xmp" && f.to_string_lossy().contains("/valid/") {
                    assert_eq!(
                        p.disposition,
                        D::Metadata(zencodec::inventory::MetadataKind::Xmp),
                        "{}",
                        f.display()
                    );
                }
                checked += 1;
            }
        }
        // Every image after the primary that the probe finds is a part.
        for r in pr.image_ranges().iter().skip(1) {
            let found = i.parts().iter().any(|p| {
                p.kind == zencodec::inventory::PartKind::EmbeddedImage
                    && p.range.start == r.start as u64
                    && p.range.end == r.end as u64
            });
            assert!(
                found,
                "{}: probe image {r:?} has no embedded-image part\n{i}",
                f.display()
            );
            checked += 1;
        }
    }
    assert!(checked > 40, "{checked}");
}

/// Scan-data parts the count-only entropy pass counted to the last MCU.
fn counted_scans(i: &Inventory) -> Vec<std::ops::Range<usize>> {
    i.parts()
        .iter()
        .filter(|p| {
            p.kind == zencodec::inventory::PartKind::ScanData
                && p.detail
                    .as_deref()
                    .is_some_and(|d| d.contains("counted to the last MCU"))
        })
        .map(|p| p.range.start as usize..p.range.end as usize)
        .collect()
}

fn unreferenced_in_scans(i: &Inventory) -> Vec<String> {
    i.parts()
        .iter()
        .filter(|p| {
            p.disposition == D::Unreferenced
                && p.parent.is_some_and(|q| {
                    i.parts()[q.index()].kind == zencodec::inventory::PartKind::ScanData
                })
        })
        .map(show)
        .collect()
}

/// Count-only entropy pass: bytes after a baseline scan's last MCU are an
/// `Unreferenced` child of the scan data, starting exactly where the
/// original data ends; a one-byte shift either way is caught.
#[test]
fn scan_tail_after_the_last_mcu_is_unreferenced() {
    let orig = load(TESTORIG);
    let eoi = orig.len() - 2; // testorig: scan data 623..5768, then EOI
    assert_eq!(&orig[eoi..], &[0xFF, 0xD9]);
    let base = pixels(&decode(job(), &orig).unwrap());
    let i0 = inv(&job(), &orig);
    assert_eq!(counted_scans(&i0), vec![623..eoi]);
    assert!(unreferenced_in_scans(&i0).is_empty());

    for tail in [&b"SECRET-AFTER-LAST-MCU"[..], &b"X"[..]] {
        let d = insert(&orig, eoi, tail);
        assert_eq!(pixels(&decode(job(), &d).unwrap()), base);
        let i = inv(&job(), &d);
        let p = leaf_at(&i, eoi);
        assert_eq!(p.disposition, D::Unreferenced, "{}", show(p));
        assert_eq!(
            p.range,
            eoi as u64..(eoi + tail.len()) as u64,
            "{}",
            show(p)
        );
        assert_eq!(leaf_at(&i, eoi - 1).disposition, D::ImageData);
    }

    // One data byte short: the bits run out (the decoder pads with zero
    // bits), so there is no tail and the scan no longer counts to its end.
    let mut d = orig.clone();
    d.remove(eoi - 1);
    let i = inv(&job(), &d);
    assert!(
        unreferenced_in_scans(&i).is_empty(),
        "{:?}",
        unreferenced_in_scans(&i)
    );
    assert!(counted_scans(&i).is_empty());
}

/// Junk planted at the end of every counted scan of a progressive file and
/// of restart-interval files changes no pixel and is found, exactly.
#[test]
fn planted_scan_tails_change_nothing_and_are_found() {
    let files = [
        PROG,
        ("jpeg-conformance", "valid/restarts.jpg"),
        ("jpeg-conformance", "valid/rst_1block.jpg"),
        ("jpeg-conformance", "valid/progressive_rst_420.jpg"),
        ("jpeg-conformance", "valid/non-interleaved-mcu.jpg"),
    ];
    let junk = b"JUNK!JUNK";
    let mut planted = 0;
    for file in files {
        let orig = load(file);
        let base = pixels(&decode(job(), &orig).unwrap());
        let i0 = inv(&job(), &orig);
        assert!(
            unreferenced_in_scans(&i0).is_empty(),
            "{file:?}: {:?}",
            unreferenced_in_scans(&i0)
        );
        for scan in counted_scans(&i0) {
            let d = insert(&orig, scan.end, junk);
            assert_eq!(
                pixels(&decode(job(), &d).unwrap()),
                base,
                "{file:?} at {}",
                scan.end
            );
            let i = inv(&job(), &d);
            let p = leaf_at(&i, scan.end);
            assert_eq!(p.disposition, D::Unreferenced, "{file:?}: {}", show(p));
            assert_eq!(p.range.start, scan.end as u64, "{file:?}: {}", show(p));
            planted += 1;
        }
    }
    assert!(planted >= 8, "{planted}");
}

/// Junk between a restart interval's data and its RSTn: the decoder drains
/// it; the inventory makes it an `Unreferenced` child.
#[test]
fn junk_before_an_rst_marker_is_unreferenced() {
    let orig = load(("jpeg-conformance", "valid/restarts.jpg"));
    let base = pixels(&decode(job(), &orig).unwrap());
    let i0 = inv(&job(), &orig);
    let scan = counted_scans(&i0)[0].clone();
    let rst = scan.start
        + orig[scan.clone()]
            .windows(2)
            .position(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
            .expect("an RSTn inside the scan");
    let d = insert(&orig, rst, b"JUNK");
    assert_eq!(pixels(&decode(job(), &d).unwrap()), base);
    let i = inv(&job(), &d);
    let p = leaf_at(&i, rst);
    assert_eq!(p.disposition, D::Unreferenced, "{}", show(p));
    assert_eq!(p.range, rst as u64..rst as u64 + 4, "{}", show(p));
}
