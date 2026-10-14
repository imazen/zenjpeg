//! `container::metadata` over the codec-corpus JPEGs: every file walks
//! without panicking and tiles exactly, reassembly holds up, and specific
//! files carry the segments and trailers they are known to carry (values
//! cross-checked with ExifTool by `tests/extended_metadata_exiftool.rs`).
#![cfg(feature = "extended-metadata")]

#[path = "support/metadata_corpus.rs"]
mod metadata_corpus;

use std::collections::BTreeMap;
use std::path::PathBuf;

use zenjpeg::container::metadata::{
    ItemKind, JpegMetadata, MetadataError, MetadataLimits, Unparsed,
};

fn corpus_file(path: &str) -> Vec<u8> {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus init");
    let (dataset, rest) = path.split_once('/').expect("dataset/path");
    let dir: PathBuf = corpus
        .get(dataset)
        .unwrap_or_else(|e| panic!("codec-corpus dataset {dataset}: {e}"));
    std::fs::read(dir.join(rest)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// The declared size in an ICC profile header (ICC.1:2010 7.2.2).
fn icc_declared_size(profile: &[u8]) -> usize {
    u32::from_be_bytes(profile[..4].try_into().unwrap()) as usize
}

/// A real ICC profile carries `acsp` at bytes 36..40 (ICC.1:2010 7.2.9);
/// the chunking fixtures carry synthetic bytes instead.
fn is_real_icc(profile: &[u8]) -> bool {
    profile.get(36..40) == Some(b"acsp")
}

#[test]
fn every_corpus_file_walks_and_reassembles() {
    let mut kinds: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut icc_ok: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut extended_xmp_ok: BTreeMap<&'static str, usize> = BTreeMap::new();
    // Separate limits, so the byte limit sees every segment of the file.
    let few_items = MetadataLimits::default().with_max_items(3);
    let few_bytes = MetadataLimits::default().with_max_reassembled_bytes(16);
    let mut refused = 0;
    for (dataset, path) in metadata_corpus::files_by_dataset() {
        let data = std::fs::read(&path).unwrap();
        let name = path.display().to_string();
        let meta = JpegMetadata::parse(&data).expect("parse");
        metadata_corpus::assert_tiles(&meta, &data, &name);
        for item in meta.items() {
            *kinds.entry(item.kind.name()).or_default() += 1;
            // Appended images are JPEGs in their own right.
            if matches!(item.kind, ItemKind::EmbeddedJpeg) {
                let inner = JpegMetadata::parse(item.payload).unwrap();
                metadata_corpus::assert_tiles(&inner, item.payload, &name);
                assert_eq!(
                    inner.primary_image_end(),
                    Some(item.payload.len()),
                    "{name}"
                );
            }
            if let ItemKind::Photoshop(_) = item.kind {
                // Resource iteration terminates on any input.
                assert!(meta.photoshop().is_ok());
            }
        }
        if let Ok(Some(icc)) = meta.icc_profile()
            && is_real_icc(&icc)
        {
            // A valid chunk set reassembles to the size its header declares.
            assert_eq!(icc_declared_size(&icc), icc.len(), "{name}: ICC size");
            *icc_ok.entry(dataset).or_default() += 1;
        }
        if let Ok(Some(ext)) = meta.extended_xmp() {
            assert!(!ext.is_empty(), "{name}");
            *extended_xmp_ok.entry(dataset).or_default() += 1;
        }
        if let Ok(Some(irb)) = meta.photoshop() {
            let _ = irb.resources().count();
        }
        let _ = meta.mpf();
        // Tight limits: still a tiling, never more items than allowed, and
        // reassembly refuses rather than allocating past the limit.
        let small = JpegMetadata::parse_with_limits(&data, few_items).unwrap();
        metadata_corpus::assert_tiles(&small, &data, &name);
        assert!(small.items().len() <= 3, "{name}");
        let capped = JpegMetadata::parse_with_limits(&data, few_bytes).unwrap();
        assert_eq!(capped.items(), meta.items(), "{name}");
        for result in [
            capped.icc_profile().map(|o| o.map(|c| c.len())),
            capped.extended_xmp().map(|o| o.map(|v| v.len())),
            capped.photoshop().map(|o| o.map(|p| p.data().len())),
        ] {
            match result {
                Ok(Some(len)) => assert!(len <= 16, "{name}: {len} bytes past the limit"),
                Err(MetadataError::TooLarge { .. }) => refused += 1,
                _ => {}
            }
        }
    }
    println!("payloads refused by the 16-byte limit: {refused}");
    println!("item kinds over the corpus: {kinds:?}");
    println!("valid ICC profiles: {icc_ok:?}, extended XMP packets: {extended_xmp_ok:?}");
    // The corpus exercises the main formats; a walk that stopped
    // identifying them would show up here.
    for kind in [
        "JFIF",
        "EXIF",
        "XMP",
        "ExtendedXMP",
        "ICC_Profile",
        "MPF",
        "Photoshop",
        "Adobe",
        "Ducky",
        "AVI1",
        "Comment",
        "EmbeddedJPEG",
        "Samsung",
        "SOF",
        "SOS",
        "DHT",
        "DQT",
        "DRI",
        "Unparsed",
        "UnknownTrailer",
    ] {
        assert!(
            kinds.contains_key(kind),
            "no {kind} item anywhere in the corpus"
        );
    }
    metadata_corpus::assert_pinned("ICC profiles reassembled", &icc_ok, ICC_REASSEMBLED);
    metadata_corpus::assert_pinned(
        "extended XMP reassembled",
        &extended_xmp_ok,
        EXTENDED_XMP_REASSEMBLED,
    );
    // Every real profile and packet is over 16 bytes, so each is refused.
    let reassembled: usize = icc_ok.values().chain(extended_xmp_ok.values()).sum();
    assert!(refused >= reassembled, "{refused} < {reassembled}");
}

#[test]
fn icc_chunk_quirk_files_are_rejected_precisely() {
    // Each fixture holds 1-byte chunks; the expected error names the first
    // violation in file order (chunk lists read from the files' bytes).
    let dir = "jpeg-conformance/non-conformant/metadata-quirks";
    let cases = [
        // Every chunk declares 250, but sequence numbers run to 254.
        (
            "icc_chunk_count_mismatch.jpeg",
            "IccSequenceOutOfRange { sequence: 254, count: 250 }",
        ),
        // Sequence 100 appears twice (and 101 never).
        (
            "icc_chunk_double_seq_no.jpeg",
            "IccDuplicateChunk { sequence: 100 }",
        ),
        (
            "icc_chunk_seq_no_0.jpeg",
            "IccSequenceOutOfRange { sequence: 0, count: 254 }",
        ),
        // 253 of 254 chunks; 100 is absent.
        (
            "icc_missing_chunk.jpeg",
            "IccMissingChunk { sequence: 100 }",
        ),
    ];
    for (file, expected) in cases {
        let data = corpus_file(&format!("{dir}/{file}"));
        let meta = JpegMetadata::parse(&data).unwrap();
        // The struct-like variants are `#[non_exhaustive]`; compare their
        // Debug form.
        let err = meta.icc_profile().expect_err(file);
        assert_eq!(format!("{err:?}"), expected, "{file}");
    }
    // Out-of-order chunks are valid and reassemble by sequence number:
    // chunk n of the fixture holds the byte n.
    let data = corpus_file(&format!("{dir}/icc_chunk_order.jpeg"));
    let meta = JpegMetadata::parse(&data).unwrap();
    let icc = meta.icc_profile().unwrap().expect("profile");
    assert_eq!(icc.len(), 254);
    let chunk_bytes: Vec<(u8, u8)> = meta
        .items()
        .iter()
        .filter_map(|i| match i.kind {
            ItemKind::IccProfile(c) => Some((c.sequence, c.data[0])),
            _ => None,
        })
        .collect();
    for (seq, byte) in chunk_bytes {
        assert_eq!(icc[usize::from(seq) - 1], byte);
    }
}

#[test]
fn multi_chunk_icc_matches_the_existing_extractor() {
    // Independent implementation: `extract_icc_profile` (color/icc.rs) sorts
    // chunks and joins them without validation; on valid files both agree.
    for (file, chunks) in [
        ("jpeg-conformance/valid/ycck.jpg", 18),
        (
            "zune/test-images/jpeg/Kiara_limited_progressive_four_components.jpg",
            9,
        ),
    ] {
        let data = corpus_file(file);
        let meta = JpegMetadata::parse(&data).unwrap();
        let n = meta
            .items()
            .iter()
            .filter(|i| matches!(i.kind, ItemKind::IccProfile(_)))
            .count();
        assert_eq!(n, chunks, "{file}");
        let ours = meta.icc_profile().unwrap().unwrap();
        let theirs = zenjpeg::decoder::extract_icc_profile(&data).unwrap();
        assert_eq!(&*ours, &theirs[..], "{file}");
        assert_eq!(icc_declared_size(&ours), ours.len(), "{file}");
    }
}

#[test]
fn ultrahdr_mpf_images_are_the_appended_jpegs() {
    let corpus = codec_corpus::Corpus::new().unwrap();
    let dir = corpus.get("ultrahdr-conformance").unwrap();
    let mut checked = 0;
    for path in metadata_corpus::files() {
        if !path.starts_with(&dir) {
            continue;
        }
        let data = std::fs::read(&path).unwrap();
        let meta = JpegMetadata::parse(&data).unwrap();
        if !meta
            .items()
            .iter()
            .any(|i| matches!(i.kind, ItemKind::Mpf(_)))
        {
            continue;
        }
        // Every MPF directory in this dataset parses.
        let entries = meta
            .mpf()
            .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()))
            .expect("an MPF item is present");
        for e in entries.iter().skip(1) {
            // Every MPF secondary that lies in the file and is a complete
            // JPEG stream is exactly one EmbeddedJPEG item.
            let Some(bytes) = data.get(e.offset..e.offset + e.size) else {
                continue;
            };
            let complete = JpegMetadata::parse(bytes)
                .unwrap()
                .primary_image_end()
                .is_some_and(|end| end == e.size);
            let item = meta.items().iter().find(|i| i.offset == e.offset);
            if complete {
                let item =
                    item.unwrap_or_else(|| panic!("{}: no item at {}", path.display(), e.offset));
                assert_eq!(item.kind, ItemKind::EmbeddedJpeg, "{}", path.display());
                assert_eq!(item.length, e.size, "{}", path.display());
                checked += 1;
            }
        }
    }
    println!("MPF images checked: {checked}");
    assert_eq!(checked, MPF_IMAGES_EMBEDDED);
}

#[test]
fn samsung_trailers_match_exiftool() {
    // Offsets and sizes as `exiftool -v3` reports them (12.40).
    for (file, offset, length) in [
        (
            "jpeg-conformance/crash-repro/jpeg-decoder/jd_095_exif_zero.jpg",
            0x25f47d,
            98,
        ),
        (
            "jpeg-conformance/crash-repro/jpeg-decoder/jd_130_progressive_jpeg.jpg",
            0x16d407,
            160,
        ),
    ] {
        let data = corpus_file(file);
        let meta = JpegMetadata::parse(&data).unwrap();
        let s = meta
            .items()
            .iter()
            .find(|i| i.kind == ItemKind::Samsung)
            .unwrap_or_else(|| panic!("{file}: no Samsung trailer"));
        assert_eq!((s.offset, s.length), (offset, length), "{file}");
        assert_eq!(s.offset + s.length, data.len(), "{file}");
    }
}

#[test]
fn extended_xmp_digest_holds_on_real_files() {
    let mut found: BTreeMap<&'static str, usize> = BTreeMap::new();
    for (dataset, path) in metadata_corpus::files_by_dataset() {
        let data = std::fs::read(&path).unwrap();
        let meta = JpegMetadata::parse(&data).unwrap();
        let declared = meta
            .xmp()
            .unwrap()
            .is_some_and(|x| x.windows(14).any(|w| w == b"HasExtendedXMP"));
        if !declared {
            continue;
        }
        // Every real file that declares extended XMP reassembles with a
        // matching MD5 GUID.
        let ext = meta.extended_xmp();
        assert!(matches!(ext, Ok(Some(_))), "{}: {ext:?}", path.display());
        *found.entry(dataset).or_default() += 1;
    }
    metadata_corpus::assert_pinned(
        "files declaring extended XMP",
        &found,
        EXTENDED_XMP_REASSEMBLED,
    );
}

#[test]
fn known_segments_in_known_files() {
    let check = |file: &str, f: &dyn Fn(&JpegMetadata<'_>)| {
        let data = corpus_file(file);
        let meta = JpegMetadata::parse(&data).unwrap();
        f(&meta);
    };
    // Motion JPEG frame: AVI1 APP0 with field 0 (ExifTool: InterleavedField 0).
    check("jpeg-conformance/valid/mjpeg.jpg", &|m| {
        assert!(m.items().iter().any(|i| matches!(
            i.kind,
            ItemKind::Avi1(a) if a.interleaved_field == Some(0)
        )));
    });
    // GIMP comment (ExifTool: "comment in GIMP 2.4.5").
    check("jpeg-conformance/valid/Nikon_D70.jpg", &|m| {
        let com = m
            .items()
            .iter()
            .find(|i| i.kind == ItemKind::Comment)
            .unwrap();
        assert_eq!(com.payload, b"comment in GIMP 2.4.5");
    });
    // YCCK with Adobe transform 2, EXIF, Photoshop resources with IPTC, XMP.
    check("jpeg-conformance/valid/ycck.jpg", &|m| {
        let adobe = m
            .items()
            .iter()
            .find_map(|i| match i.kind {
                ItemKind::Adobe(a) => Some(a),
                _ => None,
            })
            .unwrap();
        assert_eq!((adobe.dct_encode_version, adobe.color_transform), (100, 2));
        assert!(m.exif().unwrap().is_some_and(|t| t.starts_with(b"II*\0")));
        assert!(m.xmp().unwrap().is_some());
        let irb = m.photoshop().unwrap().unwrap();
        assert!(irb.resources().count() > 1);
        assert!(irb.resources().all(|r| &r.signature == b"8BIM"));
    });
}

#[test]
fn not_jpeg_inputs_are_one_item() {
    for data in [&b"\x89PNG\r\n\x1a\n"[..], b"RIFF....WEBP", &[0u8; 64][..]] {
        let meta = JpegMetadata::parse(data).unwrap();
        assert_eq!(meta.items().len(), 1);
        assert_eq!(meta.items()[0].kind, ItemKind::Unparsed(Unparsed::NotJpeg));
    }
}

/// The decoder's MPF directory (`DecodedExtras::mpf`, a parser separate from
/// `container::mpf`) for a file that decodes, with entry offsets as written
/// (relative to the MPF TIFF header).
fn decoder_mpf(data: &[u8]) -> Option<Vec<(u32, u32)>> {
    let decoded = zenjpeg::decoder::Decoder::new()
        .preserve(zenjpeg::decode::PreserveConfig::all())
        .decode(data, enough::Unstoppable)
        .ok()?;
    let dir = decoded.extras()?.mpf()?;
    Some(dir.images.iter().map(|e| (e.offset, e.size)).collect())
}

#[test]
fn icc_profiles_agree_with_the_decoder_extractor() {
    // `decoder::extract_icc_profile` (color/icc.rs) is a separate
    // implementation: it collects header chunks up to SOS, sorts them and
    // joins them without validation. Wherever it yields a real profile whose
    // header declares exactly its length, ours must be that profile, except
    // where the chunk numbering breaks ICC.1 Annex B.4 (a chunk numbered 0),
    // which libjpeg-turbo's jpeg_read_icc_profile rejects as we do.
    //
    // If `extract_icc_profile` is ever routed through this module, this
    // stops being an independent oracle: freeze its current output for the
    // corpus (or use libjpeg-turbo / ExifTool) before that change.
    let mut agreed: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut settled: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for (dataset, path) in metadata_corpus::files_by_dataset() {
        let data = std::fs::read(&path).unwrap();
        let Some(theirs) = zenjpeg::decoder::extract_icc_profile(&data) else {
            continue;
        };
        if !is_real_icc(&theirs) || icc_declared_size(&theirs) != theirs.len() {
            continue;
        }
        let meta = JpegMetadata::parse(&data).unwrap();
        match meta.icc_profile() {
            Ok(Some(ours)) if *ours == theirs[..] => *agreed.entry(dataset).or_default() += 1,
            Err(MetadataError::IccSequenceOutOfRange { sequence: 0, .. }) => {
                println!("settled (chunk numbered 0): {}", path.display());
                *settled.entry(dataset).or_default() += 1;
            }
            other => failures.push(format!(
                "{}: {:?}",
                path.display(),
                other.map(|o| o.map(|p| p.len()))
            )),
        }
    }
    println!("ICC profiles agreeing with extract_icc_profile: {agreed:?}");
    assert!(failures.is_empty(), "{failures:#?}");
    metadata_corpus::assert_pinned("ICC agreeing with extract_icc_profile", &agreed, ICC_AGREED);
    metadata_corpus::assert_pinned(
        "ICC chunk numbered 0 (rejected per ICC.1 B.4)",
        &settled,
        ICC_SEQUENCE_ZERO,
    );
}

#[test]
fn mpf_directories_agree_with_the_decoder_parser() {
    // Every file that decodes and whose decoder MPF directory is non-empty:
    // ours must parse, with the same sizes and the same offsets made
    // absolute (`container::mpf` adds the TIFF header position to non-zero
    // offsets).
    let mut agreed: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut failures = Vec::new();
    for (dataset, path) in metadata_corpus::files_by_dataset() {
        let data = std::fs::read(&path).unwrap();
        let meta = JpegMetadata::parse(&data).unwrap();
        let Some(tiff_at) = meta
            .items()
            .iter()
            .find(|i| matches!(i.kind, ItemKind::Mpf(_)))
            .map(|i| i.offset + 8)
        else {
            continue;
        };
        let Some(theirs) = decoder_mpf(&data).filter(|d| !d.is_empty()) else {
            continue;
        };
        let expected: Vec<(usize, usize)> = theirs
            .iter()
            .map(|&(offset, size)| {
                let abs = if offset == 0 {
                    0
                } else {
                    offset as usize + tiff_at
                };
                (abs, size as usize)
            })
            .collect();
        let ours = meta
            .mpf()
            .map(|o| o.map(|v| v.iter().map(|e| (e.offset, e.size)).collect::<Vec<_>>()));
        match ours {
            Ok(Some(ours)) if ours == expected => *agreed.entry(dataset).or_default() += 1,
            other => failures.push(format!(
                "{}: ours {other:?} decoder {expected:?}",
                path.display()
            )),
        }
    }
    println!("MPF directories agreeing with the decoder: {agreed:?}");
    assert!(failures.is_empty(), "{failures:#?}");
    metadata_corpus::assert_pinned("MPF agreeing with the decoder", &agreed, MPF_AGREED);
}

// Counts measured 2026-10-10 over the metadata corpus
// (`tests/support/metadata_corpus.rs`, codec-corpus 1.1.0), per dataset.
// Datasets in `metadata_corpus::MOVING` are left out (their upstream moves);
// a dataset not listed is pinned at 0. A change means the corpus or a parser
// changed: find out which.

/// Files where the two ICC extractors return the same profile.
const ICC_AGREED: &[(&str, usize)] = &[
    ("imageflow/test_inputs", 21),
    ("jpeg-conformance", 17),
    ("ultrahdr-conformance", 39),
    ("zune/test-images/jpeg", 18),
];
/// Files whose only ICC chunk is numbered 0 (`amazed2k1424ktv.jpg`).
const ICC_SEQUENCE_ZERO: &[(&str, usize)] = &[("jpeg-conformance", 1)];
/// Files where the two MPF parsers return the same directory.
const MPF_AGREED: &[(&str, usize)] = &[("jpeg-conformance", 3), ("ultrahdr-conformance", 38)];
/// MPF secondary images in ultrahdr-conformance that are complete JPEG
/// streams, each one `EmbeddedJpeg` item.
const MPF_IMAGES_EMBEDDED: usize = 37;
/// Real ICC profiles and extended XMP packets that reassemble.
const ICC_REASSEMBLED: &[(&str, usize)] = ICC_AGREED;
const EXTENDED_XMP_REASSEMBLED: &[(&str, usize)] = &[("ultrahdr-conformance", 4)];
