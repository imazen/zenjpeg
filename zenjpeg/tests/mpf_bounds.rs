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
/// segment) is not a secondary image.
#[test]
fn mpf_entries_before_the_primary_eoi_are_skipped() {
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
    assert_eq!(extracted(&jpeg), vec![]);
}

// ---------------------------------------------------------------------------
// The lossless pipeline on crafted and partly signalled containers.
// ---------------------------------------------------------------------------

use zenjpeg::container::{
    ContainerItem, ItemSemantic, MpImageType, find_jpeg_boundaries, generate_primary_xmp,
    generate_primary_xmp_with_items, parse_mpf, parse_xmp_full,
};
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

/// A GContainer item that is not a carried JPEG (a Motion Photo video after
/// the gain map) is refused, not dropped under a directory still listing it.
#[test]
fn lossless_refuses_directory_items_it_cannot_carry() {
    let gm = gray_jpeg(16, 16, 4);
    let video = b"\x00\x00\x00\x18ftypmp42not-a-jpeg-video".to_vec();
    let xmp = generate_primary_xmp_with_items(&[
        ContainerItem::primary("image/jpeg"),
        ContainerItem::secondary(ItemSemantic::GainMap, "image/jpeg", gm.len()),
        ContainerItem::secondary(
            ItemSemantic::Other("MotionPhoto".into()),
            "video/mp4",
            video.len(),
        ),
    ]);
    let px: Vec<u8> = (0..32 * 32 * 3).map(|i| (i * 13 % 251) as u8).collect();
    let mut enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .with_segments(EncoderSegments::new().set_xmp(&xmp).add_gainmap(gm))
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    let mut jpeg = enc.finish().unwrap();
    jpeg.extend_from_slice(&video);
    for r in run_both(&jpeg) {
        assert!(
            matches!(
                r.as_ref().map_err(|e| e.kind()),
                Err(ErrorKind::UnsupportedFeature { .. })
            ),
            "{:?}",
            kind_of(&r)
        );
    }
}

/// The GContainer directory's `Item:Length` follows its item's semantic, not
/// MPF order: here the MPF index lists a VGA thumbnail before the gain map,
/// and the directory names only the gain map. (Before, the first carried
/// secondary's length went to the first `Item:Length`: the thumbnail's.)
#[test]
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
