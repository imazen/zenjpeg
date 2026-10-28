//! Crafted Multi-Picture (MPF) directories: repeated, overlapping, nested and
//! misplaced entries. The secondary images an MPF index names are copied out
//! of the file on decode and carried through the lossless pipeline; a small
//! crafted index must not multiply that work or the output.
//!
//! Fixtures are built here byte by byte: the MPF APP2 segment is written by
//! [`mpf_app2`] so entries can repeat or overlap, which no encoder produces.

use enough::Unstoppable;
use zenjpeg::decode::{DecodeConfig, PreserveConfig};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// MP Entry type codes (CIPA DC-007): baseline primary, undefined (gain map).
const PRIMARY: u32 = 0x03_0000;
const UNDEFINED: u32 = 0x00_0000;
#[cfg(feature = "metadata")]
const LARGE_THUMBNAIL_VGA: u32 = 0x01_0001;
/// File offset of the MPF TIFF header when the APP2 directly follows SOI:
/// SOI (2) + marker (2) + length (2) + `MPF\0` (4).
const TIFF_POS: usize = 10;

fn gray_jpeg(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let px: Vec<u8> = (0..w * h)
        .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 24) as u8)
        .collect();
    let mut enc = EncoderConfig::grayscale(80)
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn color_jpeg(w: u32, h: u32) -> Vec<u8> {
    let px: Vec<u8> = (0..w * h * 3)
        .map(|i| (i.wrapping_mul(2_246_822_519) >> 24) as u8)
        .collect();
    let mut enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// Length in bytes of an MPF APP2 segment with `n` entries.
fn mpf_app2_len(n: usize) -> usize {
    // marker + length + "MPF\0" + TIFF header (8) + IFD count (2) +
    // 3 IFD entries (36) + next-IFD (4) + n MP entries (16 each)
    2 + 2 + 4 + 8 + 2 + 36 + 4 + 16 * n
}

/// An MPF APP2 segment listing `entries` as (type code, size, offset from
/// the TIFF header), big-endian.
fn mpf_app2(entries: &[(u32, u32, u32)]) -> Vec<u8> {
    let n = entries.len() as u32;
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"MM\x00\x2a");
    tiff.extend_from_slice(&8u32.to_be_bytes());
    tiff.extend_from_slice(&3u16.to_be_bytes());
    let ifd = |t: &mut Vec<u8>, tag: u16, ty: u16, count: u32, value: u32| {
        t.extend_from_slice(&tag.to_be_bytes());
        t.extend_from_slice(&ty.to_be_bytes());
        t.extend_from_slice(&count.to_be_bytes());
        t.extend_from_slice(&value.to_be_bytes());
    };
    ifd(&mut tiff, 0xB000, 7, 4, u32::from_be_bytes(*b"0100"));
    ifd(&mut tiff, 0xB001, 4, 1, n);
    ifd(&mut tiff, 0xB002, 7, n * 16, 8 + 2 + 36 + 4);
    tiff.extend_from_slice(&0u32.to_be_bytes());
    for &(ty, size, off) in entries {
        let attr = if ty == PRIMARY { 0x2000_0000 | ty } else { ty };
        tiff.extend_from_slice(&attr.to_be_bytes());
        tiff.extend_from_slice(&size.to_be_bytes());
        tiff.extend_from_slice(&off.to_be_bytes());
        tiff.extend_from_slice(&[0, 0, 0, 0]);
    }
    let mut seg = vec![0xFF, 0xE2];
    seg.extend_from_slice(&u16::try_from(2 + 4 + tiff.len()).unwrap().to_be_bytes());
    seg.extend_from_slice(b"MPF\0");
    seg.extend_from_slice(&tiff);
    assert_eq!(seg.len(), mpf_app2_len(entries.len()));
    seg
}

/// `primary` with an MPF APP2 inserted right after SOI, then `tail`.
/// `secondaries` receives the final primary length and returns the
/// non-primary entries as (type code, size, absolute file offset).
fn with_mpf(
    primary: &[u8],
    n_secondaries: usize,
    secondaries: impl Fn(usize) -> Vec<(u32, u32, usize)>,
    tail: &[u8],
) -> Vec<u8> {
    let primary_len = primary.len() + mpf_app2_len(1 + n_secondaries);
    let mut entries = vec![(PRIMARY, primary_len as u32, 0)];
    let rest = secondaries(primary_len);
    assert_eq!(rest.len(), n_secondaries);
    entries.extend(
        rest.into_iter()
            .map(|(t, s, abs)| (t, s, (abs - TIFF_POS) as u32)),
    );
    let mut out = primary[..2].to_vec();
    out.extend_from_slice(&mpf_app2(&entries));
    out.extend_from_slice(&primary[2..]);
    assert_eq!(out.len(), primary_len);
    out.extend_from_slice(tail);
    out
}

fn extracted(jpeg: &[u8]) -> Vec<(usize, usize)> {
    let decoded = DecodeConfig::new()
        .preserve(PreserveConfig::all())
        .decode(jpeg, Unstoppable)
        .expect("decode");
    decoded
        .extras()
        .map(|e| {
            e.secondary_images()
                .iter()
                .map(|s| (s.mpf_index, s.data.len()))
                .collect()
        })
        .unwrap_or_default()
}

/// Native MPF parsing has an independent compatibility fallback. Both its
/// normal IFD walk and its fallback must reject offsets that wrap on i686.
#[test]
fn native_mpf_offsets_do_not_wrap() {
    let primary = color_jpeg(32, 32);
    let map = gray_jpeg(16, 16, 8);
    let jpeg = with_mpf(
        &primary,
        1,
        |p| vec![(UNDEFINED, map.len() as u32, p)],
        &map,
    );
    for (field, fallback) in [(14, false), (52, false), (52, true)] {
        let mut input = jpeg.clone();
        input[field..field + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        if fallback {
            input[18..20].copy_from_slice(&0u16.to_be_bytes());
        }
        let result = DecodeConfig::new()
            .preserve(PreserveConfig::all())
            .decode(&input, Unstoppable)
            .unwrap();
        assert!(result.extras().unwrap().mpf().is_none());
    }
}

/// A directory naming the same secondary 255 times extracts it once (it
/// copied it 255 times).
#[test]
fn repeated_mpf_entries_are_extracted_once() {
    let gm = gray_jpeg(16, 16, 7);
    let jpeg = with_mpf(
        &color_jpeg(32, 32),
        255,
        |p| vec![(UNDEFINED, gm.len() as u32, p); 255],
        &gm,
    );
    assert_eq!(extracted(&jpeg), vec![(1, gm.len())]);
}

/// An entry overlapping one already taken is skipped.
#[test]
fn overlapping_mpf_entries_are_skipped() {
    let a = gray_jpeg(16, 16, 1);
    let b = gray_jpeg(16, 16, 2);
    let tail = [a.clone(), b.clone()].concat();
    let jpeg = with_mpf(
        &color_jpeg(32, 32),
        3,
        |p| {
            vec![
                (UNDEFINED, a.len() as u32, p),
                // Starts at `a`'s SOI (so the SOI check passes) and runs over
                // `b`.
                (UNDEFINED, (a.len() + b.len()) as u32, p),
                (UNDEFINED, b.len() as u32, p + a.len()),
            ]
        },
        &tail,
    );
    assert_eq!(extracted(&jpeg), vec![(1, a.len()), (3, b.len())]);
}

/// An entry pointing inside the primary (here at a JPEG embedded in an APP15
/// segment) remains readable by default, but strict decode and rewriting reject it.
#[test]
fn embedded_mpf_entries_are_readable_except_in_strict_mode() {
    let inner = gray_jpeg(16, 16, 3);
    let base = color_jpeg(32, 32);
    // APP15 right after SOI carrying `inner`.
    let mut app15 = vec![0xFF, 0xEF];
    app15.extend_from_slice(&u16::try_from(2 + inner.len()).unwrap().to_be_bytes());
    app15.extend_from_slice(&inner);
    let primary = [&base[..2], &app15[..], &base[2..]].concat();
    let app2 = mpf_app2_len(2);
    // `inner` sits after SOI + MPF APP2 + the APP15 marker and length.
    let jpeg = with_mpf(
        &primary,
        1,
        |_| vec![(UNDEFINED, inner.len() as u32, 2 + app2 + 4)],
        &[],
    );
    assert_eq!(extracted(&jpeg), vec![(1, inner.len())]);
    let strict = DecodeConfig::new()
        .strict()
        .preserve(PreserveConfig::all())
        .decode(&jpeg, Unstoppable);
    assert!(
        strict
            .unwrap_err()
            .to_string()
            .contains("overlaps the primary")
    );
    for result in run_both(&jpeg) {
        assert!(result.unwrap_err().to_string().contains("cannot relocate"));
    }
}

// ---------------------------------------------------------------------------
// The lossless pipeline on crafted and partly signalled containers.
// ---------------------------------------------------------------------------

#[cfg(feature = "metadata")]
use zenjpeg::container::{
    ContainerItem, ItemSemantic, generate_primary_xmp, generate_primary_xmp_with_items,
    parse_xmp_full,
};
use zenjpeg::container::{MpImageType, find_jpeg_boundaries, parse_mpf};
use zenjpeg::decoder::ErrorKind;
use zenjpeg::encode::EncoderSegments;
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, RestructureConfig, TransformConfig, restructure, transform,
};

fn rotate180() -> TransformConfig {
    TransformConfig {
        transform: LosslessTransform::Rotate180,
        edge_handling: EdgeHandling::RejectPartialBlocks,
    }
}

/// `transform(Rotate180)` and `restructure` (no transform, the verbatim
/// carry) on `jpeg`.
fn run_both(jpeg: &[u8]) -> [Result<Vec<u8>, zenjpeg::decoder::DecodeError>; 2] {
    [
        transform(jpeg, &rotate180(), Unstoppable),
        restructure(jpeg, &RestructureConfig::default(), Unstoppable),
    ]
}

fn kind_of(r: &Result<Vec<u8>, zenjpeg::decoder::DecodeError>) -> Option<String> {
    r.as_ref().err().map(|e| format!("{:?}", e.kind()))
}

/// 255 entries naming one secondary: refused (the index names images the
/// file does not hold separately), not carried 255 times.
#[test]
fn lossless_refuses_repeated_mpf_entries() {
    let gm = gray_jpeg(16, 16, 7);
    let jpeg = with_mpf(
        &color_jpeg(32, 32),
        255,
        |p| vec![(UNDEFINED, gm.len() as u32, p); 255],
        &gm,
    );
    for r in run_both(&jpeg) {
        assert!(
            matches!(
                r.as_ref().map_err(|e| e.kind()),
                Err(ErrorKind::InvalidJpegData { .. })
            ),
            "{:?}",
            kind_of(&r)
        );
    }
}

/// A secondary that is itself an MPF container: transforming it is refused
/// (secondaries are carried one level deep); the verbatim carry keeps its
/// bytes, nested image included, once.
#[test]
fn lossless_carries_secondaries_one_level_deep() {
    let inner = gray_jpeg(16, 16, 9);
    let outer = with_mpf(
        &gray_jpeg(16, 16, 8),
        1,
        |p| vec![(UNDEFINED, inner.len() as u32, p)],
        &inner,
    );
    let jpeg = with_mpf(
        &color_jpeg(32, 32),
        1,
        |p| vec![(UNDEFINED, outer.len() as u32, p)],
        &outer,
    );
    let [rotated, restructured] = run_both(&jpeg);
    assert!(
        matches!(
            rotated.as_ref().map_err(|e| e.kind()),
            Err(ErrorKind::UnsupportedFeature { .. })
        ),
        "{:?}",
        kind_of(&rotated)
    );
    let out = restructured.expect("verbatim carry");
    assert!(
        out.ends_with(&outer),
        "the nested secondary is carried verbatim"
    );
    assert!(out.len() < 2 * jpeg.len());
}

/// 500 nested levels: one level is examined, the rest refused, with no
/// recursion (the old carry-through recursed once per level).
#[test]
fn lossless_does_not_recurse_into_deep_nesting() {
    let mut cur = gray_jpeg(8, 8, 0);
    for k in 1..500u32 {
        let next = cur;
        cur = with_mpf(
            &gray_jpeg(8, 8, k),
            1,
            |p| vec![(UNDEFINED, next.len() as u32, p)],
            &next,
        );
    }
    let jpeg = with_mpf(
        &color_jpeg(32, 32),
        1,
        |p| vec![(UNDEFINED, cur.len() as u32, p)],
        &cur,
    );
    let [rotated, restructured] = run_both(&jpeg);
    assert!(rotated.is_err());
    assert!(restructured.expect("verbatim carry").len() < 2 * jpeg.len());
}

/// A gain map signalled only by `hdrgm` XMP, with no MPF index (Android
/// writes some files this way): carried, rotated with the primary, and
/// given an MPF index. It used to be dropped while the XMP stayed.
#[test]
#[cfg(feature = "metadata")]
fn lossless_carries_a_gain_map_without_an_mpf_index() {
    let xmp = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0" hdrgm:GainMapMax="2.0"/></rdf:RDF></x:xmpmeta>"#;
    let px: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 7 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .with_segments(EncoderSegments::new().set_xmp(xmp))
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    let mut jpeg = enc.finish().unwrap();
    let gm = gray_jpeg(16, 8, 5);
    jpeg.extend_from_slice(&gm);

    let out = transform(&jpeg, &rotate180(), Unstoppable).expect("rotate");
    let ranges = find_jpeg_boundaries(&out);
    assert_eq!(ranges.len(), 2, "gain map carried");
    let index = parse_mpf(&out).expect("an MPF index was added");
    assert_eq!(index.len(), 2);
    assert_eq!(index[1].image_type, MpImageType::Undefined);
    assert_eq!(index[1].offset, ranges[1].start);
    let gm_out = DecodeConfig::new()
        .decode(&out[ranges[1].clone()], Unstoppable)
        .unwrap();
    assert_eq!((gm_out.width(), gm_out.height()), (16, 8));
}

/// A GContainer item that is not a JPEG (a Motion Photo video after the gain
/// map) is carried verbatim after the gain map, with the directory lengths
/// rewritten: the gain map's to its new size, the video's as it was. It was
/// refused; the carry policy decided on 2026-10-11 refuses only real loss.
#[test]
#[cfg(feature = "metadata")]
fn lossless_carries_a_motion_photo_video_verbatim() {
    let gm = gray_jpeg(16, 8, 4);
    let video = mp4(&[]);
    let xmp = generate_primary_xmp_with_items(&[
        ContainerItem::primary("image/jpeg"),
        ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", gm.len()),
        ContainerItem::secondary(
            ItemSemantic::Other("MotionPhoto".into()),
            "video/mp4",
            video.len(),
        ),
    ]);
    let jpeg = primary_with(EncoderSegments::new().set_xmp(&xmp).add_gainmap(gm), &video);
    for (name, r, gm_dims) in [
        (
            "transform",
            transform(&jpeg, &rotate90(), Unstoppable),
            (8, 16),
        ),
        (
            "restructure",
            restructure(&jpeg, &RestructureConfig::default(), Unstoppable),
            (16, 8),
        ),
    ] {
        let out = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(out.ends_with(&video), "{name}: the video is carried");
        let index = parse_mpf(&out).expect("MPF index");
        assert_eq!(index.len(), 2, "{name}");
        let gm_end = index[1].offset + index[1].size;
        assert_eq!(
            gm_end + video.len(),
            out.len(),
            "{name}: the video follows the gain map"
        );
        assert_eq!(dims(&out[index[1].offset..gm_end]), gm_dims, "{name}");
        let lengths: Vec<usize> = parse_xmp_full(&xmp_of(&out))
            .1
            .iter()
            .filter_map(|i| i.length)
            .collect();
        assert_eq!(lengths, vec![index[1].size, video.len()], "{name}");
    }
}

/// The GContainer directory's `Item:Length` follows its item's semantic, not
/// MPF order: here the MPF index lists a VGA thumbnail before the gain map,
/// and the directory names only the gain map. (Before, the first carried
/// secondary's length went to the first `Item:Length`: the thumbnail's.)
#[test]
#[cfg(feature = "metadata")]
fn lossless_matches_directory_lengths_by_semantic() {
    let thumb = gray_jpeg(32, 16, 11);
    let gm = gray_jpeg(16, 16, 12);
    let px: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 3 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .with_segments(EncoderSegments::new().set_xmp(&generate_primary_xmp(gm.len())))
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    let primary = enc.finish().unwrap();
    let jpeg = with_mpf(
        &primary,
        2,
        |p| {
            vec![
                (LARGE_THUMBNAIL_VGA, thumb.len() as u32, p),
                (UNDEFINED, gm.len() as u32, p + thumb.len()),
            ]
        },
        &[thumb.clone(), gm.clone()].concat(),
    );
    let out = transform(&jpeg, &rotate180(), Unstoppable).expect("rotate");
    let index = parse_mpf(&out).expect("MPF index");
    assert_eq!(index.len(), 3, "{index:?}");
    let entry = |t: MpImageType| {
        index
            .iter()
            .find(|e| e.image_type == t)
            .unwrap_or_else(|| panic!("{t:?} entry in {index:?}"))
            .size
    };
    let (gm_size, thumb_size) = (
        entry(MpImageType::Undefined),
        entry(MpImageType::LargeThumbnailVga),
    );
    assert_ne!(gm_size, thumb_size, "the fixture must tell them apart");
    let xmp = DecodeConfig::new()
        .preserve(PreserveConfig::all())
        .decode(&out, Unstoppable)
        .unwrap()
        .extras()
        .and_then(|e| e.xmp().map(str::to_owned))
        .expect("xmp");
    let lengths: Vec<usize> = parse_xmp_full(&xmp)
        .1
        .iter()
        .filter_map(|i| i.length)
        .collect();
    assert_eq!(lengths, vec![gm_size]);
}

/// A secondary's failure keeps its kind: an edge rejection of the gain map
/// stays InvalidConfig (it was re-wrapped as DecodeError). The other kinds
/// (cancellation) are covered by `lossless::pipeline` unit tests.
#[test]
#[cfg(feature = "metadata")]
fn secondary_failures_keep_their_kind() {
    // 32x32 primary (aligned), 16x8 4:2:0 gain map: a 180-degree rotation
    // would trim its 8 rows.
    let px: Vec<u8> = (0..16 * 8 * 3).map(|i| (i * 5 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(80, ChromaSubsampling::Quarter)
        .encode_from_bytes(16, 8, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    let gm = enc.finish().unwrap();
    let px: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 7 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .with_segments(
            EncoderSegments::new()
                .set_xmp(&generate_primary_xmp(gm.len()))
                .add_gainmap(gm),
        )
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();
    let err = transform(&jpeg, &rotate180(), Unstoppable).expect_err("gain map rejected");
    assert!(matches!(err.kind(), ErrorKind::InvalidConfig(_)), "{err:?}");
    assert!(err.to_string().contains("MPF secondary image #1"), "{err}");
}

// ---------------------------------------------------------------------------
// Carry policy: every byte after the primary is carried, or the call fails.

const HDRGM_XMP: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0" hdrgm:GainMapMax="2.0"/></rdf:RDF></x:xmpmeta>"#;

fn rotate90() -> TransformConfig {
    TransformConfig {
        transform: LosslessTransform::Rotate90,
        edge_handling: EdgeHandling::RejectPartialBlocks,
    }
}

/// An EXIF TIFF structure holding only an orientation tag.
#[cfg(feature = "metadata")]
fn exif_orientation(o: u16) -> Vec<u8> {
    let mut t = b"MM\x00\x2a\x00\x00\x00\x08\x00\x01".to_vec();
    t.extend_from_slice(&0x0112u16.to_be_bytes());
    t.extend_from_slice(&3u16.to_be_bytes());
    t.extend_from_slice(&1u32.to_be_bytes());
    t.extend_from_slice(&o.to_be_bytes());
    t.extend_from_slice(&[0; 6]);
    t
}

/// A 32x32 4:2:0 primary written with `segments`, then `tail`.
fn primary_with(segments: EncoderSegments, tail: &[u8]) -> Vec<u8> {
    let px: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 11 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .with_segments(segments)
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    let mut out = enc.finish().unwrap();
    out.extend_from_slice(tail);
    out
}

/// A minimal MP4: `ftyp`, then an `mdat` holding `inner` between filler.
fn mp4(inner: &[u8]) -> Vec<u8> {
    let mut out = 24u32.to_be_bytes().to_vec();
    out.extend_from_slice(b"ftypmp42\x00\x00\x00\x00mp42isom");
    let body = [&[0x11; 200][..], inner, &[0x22; 200]].concat();
    out.extend_from_slice(&(8 + body.len() as u32).to_be_bytes());
    out.extend_from_slice(b"mdat");
    out.extend_from_slice(&body);
    out
}

#[cfg(feature = "metadata")]
fn xmp_of(jpeg: &[u8]) -> String {
    DecodeConfig::new()
        .preserve(PreserveConfig::all())
        .decode(jpeg, Unstoppable)
        .unwrap()
        .extras()
        .and_then(|e| e.xmp().map(str::to_owned))
        .expect("xmp")
}

#[cfg(feature = "metadata")]
fn dims(jpeg: &[u8]) -> (u32, u32) {
    let d = DecodeConfig::new().read_info(jpeg).unwrap().dimensions;
    (d.width, d.height)
}

/// An Ultra HDR file whose gain map was stripped (`jpegtran -copy all` keeps
/// the MPF index and the XMP and drops what follows the primary): every
/// entry point passes it through. The XMP keeps its signalling as it was;
/// the index, which names bytes the output does not hold, is not written.
/// All three refused it.
#[test]
#[cfg(feature = "metadata")]
fn lossless_passes_a_stripped_mpf_trailer_through() {
    let gm = gray_jpeg(16, 8, 3);
    let full = primary_with(
        EncoderSegments::new()
            .set_exif(exif_orientation(6))
            .set_xmp(&generate_primary_xmp(gm.len()))
            .add_gainmap(gm),
        &[],
    );
    let stripped = &full[..find_jpeg_boundaries(&full)[0].end];
    assert!(
        parse_mpf(stripped).is_ok_and(|i| i.len() == 2),
        "the index still names the gain map"
    );
    let xmp_in = xmp_of(stripped);
    for (name, r) in [
        ("transform", transform(stripped, &rotate90(), Unstoppable)),
        (
            "restructure",
            restructure(stripped, &RestructureConfig::default(), Unstoppable),
        ),
        (
            "apply_exif_orientation",
            zenjpeg::lossless::apply_exif_orientation(stripped, Unstoppable),
        ),
    ] {
        let out = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(find_jpeg_boundaries(&out).len(), 1, "{name}");
        assert_eq!(out.len(), find_jpeg_boundaries(&out)[0].end, "{name}");
        assert!(parse_mpf(&out).is_err(), "{name}: no index");
        assert_eq!(xmp_of(&out), xmp_in, "{name}: the XMP is unchanged");
    }
}

/// A legacy `GCamera:MicroVideo` file (the video follows the primary, its
/// offset counted from the end of the file, no index): the video is carried
/// verbatim at the end, so the XMP offset stays right. It was dropped.
#[test]
#[cfg(feature = "metadata")]
fn lossless_carries_a_microvideo_trailer() {
    let video = mp4(&[]);
    let xmp = format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:GCamera="http://ns.google.com/photos/1.0/camera/" GCamera:MicroVideo="1" GCamera:MicroVideoVersion="1" GCamera:MicroVideoOffset="{}"/></rdf:RDF></x:xmpmeta>"#,
        video.len()
    );
    let jpeg = primary_with(
        EncoderSegments::new()
            .set_exif(exif_orientation(8))
            .set_xmp(&xmp),
        &video,
    );
    for (name, r) in [
        ("transform", transform(&jpeg, &rotate90(), Unstoppable)),
        (
            "restructure",
            restructure(&jpeg, &RestructureConfig::default(), Unstoppable),
        ),
        (
            "apply_exif_orientation",
            zenjpeg::lossless::apply_exif_orientation(&jpeg, Unstoppable),
        ),
    ] {
        let out = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(out.ends_with(&video), "{name}: the video is carried");
        assert_eq!(
            find_jpeg_boundaries(&out)[0].end + video.len(),
            out.len(),
            "{name}"
        );
        assert_eq!(xmp_of(&out), xmp, "{name}");
    }
}

/// An `hdrgm` gain map that the file's MPF index does not list: carried,
/// transformed with the primary, and listed in the rebuilt index. It was
/// dropped while the XMP still signalled it.
#[test]
#[cfg(feature = "metadata")]
fn lossless_carries_a_gain_map_the_index_does_not_list() {
    let gm = gray_jpeg(16, 8, 6);
    let jpeg = with_mpf(
        &primary_with(EncoderSegments::new().set_xmp(HDRGM_XMP), &[]),
        0,
        |_| vec![],
        &gm,
    );
    for (name, r, gm_dims) in [
        (
            "transform",
            transform(&jpeg, &rotate90(), Unstoppable),
            (8, 16),
        ),
        (
            "restructure",
            restructure(&jpeg, &RestructureConfig::default(), Unstoppable),
            (16, 8),
        ),
    ] {
        let out = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        let index = parse_mpf(&out).expect("MPF index");
        assert_eq!(index.len(), 2, "{name}");
        assert_eq!(index[1].image_type, MpImageType::Undefined, "{name}");
        assert_eq!(index[1].offset + index[1].size, out.len(), "{name}");
        assert_eq!(dims(&out[index[1].offset..]), gm_dims, "{name}");
    }
}

/// A gain map without its EOI cannot be transformed: refused, not dropped
/// (it was dropped). Without a transform it is carried as it is.
#[test]
#[cfg(feature = "metadata")]
fn lossless_refuses_a_gain_map_it_cannot_read() {
    let gm = gray_jpeg(16, 8, 7);
    let cut = &gm[..gm.len() - 2];
    let jpeg = primary_with(EncoderSegments::new().set_xmp(HDRGM_XMP), cut);
    let err = transform(&jpeg, &rotate90(), Unstoppable).expect_err("refused");
    assert!(
        matches!(err.kind(), ErrorKind::InvalidJpegData { .. }),
        "{err:?}"
    );
    let out = restructure(&jpeg, &RestructureConfig::default(), Unstoppable).expect("restructure");
    assert!(out.ends_with(cut));
}

/// `hdrgm` XMP followed by an MP4 that holds a JPEG: the JPEG inside the
/// video is not taken as the gain map (it was, and the rest of the video
/// was dropped). The video is carried whole and no index is added.
#[test]
#[cfg(feature = "metadata")]
fn lossless_does_not_take_a_jpeg_inside_a_video_as_the_gain_map() {
    let video = mp4(&gray_jpeg(64, 8, 5));
    let jpeg = primary_with(EncoderSegments::new().set_xmp(HDRGM_XMP), &video);
    for (name, r) in [
        ("transform", transform(&jpeg, &rotate90(), Unstoppable)),
        (
            "restructure",
            restructure(&jpeg, &RestructureConfig::default(), Unstoppable),
        ),
    ] {
        let out = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(out.ends_with(&video), "{name}");
        assert_eq!(
            find_jpeg_boundaries(&out)[0].end + video.len(),
            out.len(),
            "{name}"
        );
        assert!(parse_mpf(&out).is_err(), "{name}: no index added");
    }
}

/// The GContainer directory is read from the standard XMP packet. With
/// Adobe-style extended XMP present, the reassembled XMP is two packets and
/// no directory parses, so every lossless call failed with "invalid XMP
/// directory". The extended chunk is carried unchanged.
#[test]
#[cfg(feature = "metadata")]
fn lossless_reads_the_directory_from_the_standard_xmp_packet() {
    let gm = gray_jpeg(16, 8, 8);
    let jpeg = primary_with(
        EncoderSegments::new()
            .set_xmp(&generate_primary_xmp(gm.len()))
            .add_gainmap(gm),
        &[],
    );
    let ext_packet = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" photoshop:History="edited"/></rdf:RDF></x:xmpmeta>"#;
    let mut payload = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    payload.extend_from_slice(b"0123456789ABCDEF0123456789ABCDEF");
    payload.extend_from_slice(&(ext_packet.len() as u32).to_be_bytes());
    payload.extend_from_slice(&0u32.to_be_bytes());
    payload.extend_from_slice(ext_packet.as_bytes());
    let mut segment = vec![0xFF, 0xE1];
    segment.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    segment.extend_from_slice(&payload);
    // Right after SOI: before the MPF index, so its relative offsets hold.
    let jpeg = [&jpeg[..2], &segment, &jpeg[2..]].concat();
    for (name, r) in [
        ("transform", transform(&jpeg, &rotate90(), Unstoppable)),
        (
            "restructure",
            restructure(&jpeg, &RestructureConfig::default(), Unstoppable),
        ),
    ] {
        let out = r.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            out.windows(segment.len()).any(|w| w == segment),
            "{name}: the extended chunk is carried"
        );
        let index = parse_mpf(&out).expect("MPF index");
        let decoded = DecodeConfig::new()
            .preserve(PreserveConfig::all())
            .decode(&out, Unstoppable)
            .unwrap();
        let standard = decoded
            .extras()
            .unwrap()
            .segments()
            .iter()
            .find(|s| s.segment_type == zenjpeg::decode::SegmentType::Xmp)
            .map(|s| String::from_utf8(s.data[29..].to_vec()).unwrap())
            .unwrap();
        let lengths: Vec<usize> = parse_xmp_full(&standard)
            .1
            .iter()
            .filter_map(|i| i.length)
            .collect();
        assert_eq!(lengths, vec![index[1].size], "{name}");
    }
}

/// Each transformed secondary is decoded at its declared size, so the carry
/// is bounded by the primary: one secondary may not exceed the primary's
/// pixels, all of them together not four times them. A 16x16 scan whose
/// frame header declares 4096x4096 is refused before it is decoded (16 such
/// secondaries in a 2 KB file took 3.75 s). Untransformed, it is copied.
#[test]
fn lossless_bounds_the_work_secondaries_declare() {
    let mut big = gray_jpeg(16, 16, 1);
    let sof = big
        .windows(2)
        .position(|w| w[0] == 0xFF && matches!(w[1], 0xC0..=0xC2))
        .unwrap();
    big[sof + 5..sof + 9].copy_from_slice(&[0x10, 0x00, 0x10, 0x00]);
    let jpeg = with_mpf(
        &color_jpeg(32, 32),
        1,
        |p| vec![(UNDEFINED, big.len() as u32, p)],
        &big,
    );
    let err = transform(&jpeg, &rotate90(), Unstoppable).expect_err("refused");
    assert!(
        matches!(err.kind(), ErrorKind::ImageTooLarge { .. }),
        "{err:?}"
    );
    let out = restructure(&jpeg, &RestructureConfig::default(), Unstoppable).expect("copied");
    assert!(out.ends_with(&big));

    // Primary-sized secondaries: four fit the budget, five do not.
    for (n, fits) in [(4usize, true), (5, false)] {
        let secondaries: Vec<Vec<u8>> = (0..n).map(|k| gray_jpeg(32, 32, 40 + k as u32)).collect();
        let jpeg = with_mpf(
            &color_jpeg(32, 32),
            n,
            |p| {
                let mut at = p;
                secondaries
                    .iter()
                    .map(|s| {
                        let e = (UNDEFINED, s.len() as u32, at);
                        at += s.len();
                        e
                    })
                    .collect()
            },
            &secondaries.concat(),
        );
        let r = transform(&jpeg, &rotate90(), Unstoppable);
        if fits {
            assert_eq!(parse_mpf(&r.expect("four fit")).unwrap().len(), 5);
        } else {
            assert!(
                matches!(
                    r.as_ref().map_err(|e| e.kind()),
                    Err(ErrorKind::ImageTooLarge { .. })
                ),
                "{:?}",
                kind_of(&r)
            );
        }
    }
}

/// Without `metadata` the XMP directory cannot be classified, so a transform
/// whose bytes after the primary the XMP could name refuses before writing
/// anything, rather than dropping or guessing what those bytes are (the
/// `metadata` feature's contract). The carry tests above need the XML parser
/// to classify the same files.
#[test]
#[cfg(not(feature = "metadata"))]
fn without_metadata_a_transform_refuses_xmp_containers_it_cannot_classify() {
    let gm = gray_jpeg(16, 8, 6);
    let cases = [
        (
            "gain map without an index",
            primary_with(EncoderSegments::new().set_xmp(HDRGM_XMP), &gm),
        ),
        (
            "gain map the index does not list",
            with_mpf(
                &primary_with(EncoderSegments::new().set_xmp(HDRGM_XMP), &[]),
                0,
                |_| vec![],
                &gm,
            ),
        ),
        (
            "unreadable gain map",
            primary_with(
                EncoderSegments::new().set_xmp(HDRGM_XMP),
                &gm[..gm.len() - 2],
            ),
        ),
        (
            "JPEG inside a video",
            primary_with(
                EncoderSegments::new().set_xmp(HDRGM_XMP),
                &mp4(&gray_jpeg(64, 8, 5)),
            ),
        ),
    ];
    for (name, jpeg) in &cases {
        for config in [rotate90(), rotate180()] {
            let err = transform(jpeg, &config, Unstoppable).expect_err(name);
            assert!(
                matches!(err.kind(), ErrorKind::UnsupportedFeature { .. }),
                "{name}: {err:?}"
            );
        }
    }
}
