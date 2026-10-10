//! The JPEG corpus the extended-metadata tests walk, and the invariants every
//! walk must keep. Shared by `tests/extended_metadata.rs` and
//! `tests/extended_metadata_exiftool.rs`.

use std::fs;
use std::path::{Path, PathBuf};

use zenjpeg::container::metadata::{Item, JpegMetadata};

/// codec-corpus datasets that hold JPEGs. Every file in them is walked.
pub const DATASETS: &[&str] = &[
    "jpeg-conformance",
    "ultrahdr-conformance",
    "imageflow/test_inputs",
    "mozjpeg",
    "zune/test-images/jpeg",
    "zune/fuzz-corpus/jpeg",
    "image-rs/tests/images/jpg",
];

/// Datasets that track a moving upstream branch (codec-corpus fetches
/// image-rs from its `main` and refreshes it after 7 days), so exact counts
/// over them would change without a code change. Tests still check every
/// file in them; only their counts stay out of the exact pins.
pub const MOVING: &[&str] = &["image-rs/tests/images/jpg"];

/// JPEG-looking files per dataset outside [`MOVING`], measured 2026-10-10.
/// Every count pinned by the extended-metadata tests assumes codec-corpus
/// 1.1.0 content (`Cargo.lock`); the crate fetches the imazen datasets at
/// that tag, or from the default branch when the tag is missing, so an
/// upstream change shows up here first.
const FILE_COUNTS: &[(&str, usize)] = &[
    ("jpeg-conformance", 254),
    ("ultrahdr-conformance", 49),
    ("imageflow/test_inputs", 30),
    ("mozjpeg", 4),
    ("zune/test-images/jpeg", 30),
    ("zune/fuzz-corpus/jpeg", 1834),
];

/// The fewest files a [`MOVING`] dataset may hold: its count is not pinned,
/// but an empty fetch must fail.
const MOVING_FLOOR: usize = 1;

/// Every JPEG-looking file of [`DATASETS`]: a `.jpg`/`.jpeg`/`.jfif`
/// extension, or no extension and an `FF D8` start (the zune fuzz corpus
/// names files by hash). Sorted, so runs are reproducible.
#[allow(dead_code)] // used by one of the two test binaries that include this file
pub fn files() -> Vec<PathBuf> {
    files_by_dataset().into_iter().map(|(_, p)| p).collect()
}

/// [`files`] with the dataset each one came from. Each dataset outside
/// [`MOVING`] must hold its pinned file count, and each moving one at least
/// [`MOVING_FLOOR`] files.
///
/// `METADATA_CORPUS_EXCLUDE` (comma-separated) leaves datasets out. The
/// caller sets it (the i686 cross job, whose container cannot fetch
/// image-rs); only [`MOVING`] datasets may be excluded, so no pinned count
/// changes.
pub fn files_by_dataset() -> Vec<(&'static str, PathBuf)> {
    let corpus = codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)");
    let exclude = std::env::var("METADATA_CORPUS_EXCLUDE").unwrap_or_default();
    let exclude: Vec<&str> = exclude.split(',').filter(|d| !d.is_empty()).collect();
    for d in &exclude {
        assert!(
            MOVING.contains(d),
            "METADATA_CORPUS_EXCLUDE may only name a moving dataset, not {d}"
        );
        println!("METADATA_CORPUS_EXCLUDE: leaving out {d}");
    }
    let mut out = Vec::new();
    for dataset in DATASETS.iter().filter(|d| !exclude.contains(d)) {
        let dir = corpus
            .get(dataset)
            .unwrap_or_else(|e| panic!("codec-corpus dataset {dataset}: {e}"));
        let mut paths = Vec::new();
        collect(&dir, &mut paths);
        if MOVING.contains(dataset) {
            assert!(
                paths.len() >= MOVING_FLOOR,
                "{dataset}: {} files; a moving dataset must hold at least {MOVING_FLOOR}",
                paths.len()
            );
        } else {
            let pinned = FILE_COUNTS
                .iter()
                .find(|(d, _)| d == dataset)
                .map(|&(_, n)| n);
            assert_eq!(Some(paths.len()), pinned, "{dataset}: file count");
        }
        out.extend(paths.into_iter().map(|p| (*dataset, p)));
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out.dedup_by(|a, b| a.1 == b.1);
    assert!(out.len() > 300, "corpus too small: {} files", out.len());
    out
}

/// Per-dataset counts against pins: every dataset outside [`MOVING`] must
/// match `expected` exactly (absent means 0); moving datasets are printed.
#[allow(dead_code)] // used by one of the two test binaries that include this file
pub fn assert_pinned(
    what: &str,
    counts: &std::collections::BTreeMap<&'static str, usize>,
    expected: &[(&str, usize)],
) {
    println!("{what}: {counts:?}");
    for dataset in DATASETS {
        let got = counts.get(dataset).copied().unwrap_or(0);
        if MOVING.contains(dataset) {
            println!("{what}: {dataset}: {got} (moving upstream, not pinned)");
            continue;
        }
        let want = expected
            .iter()
            .find(|(d, _)| d == dataset)
            .map_or(0, |&(_, n)| n);
        assert_eq!(got, want, "{what}: {dataset}");
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect(&path, out);
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let jpeg = match ext.as_deref() {
            Some("jpg" | "jpeg" | "jfif") => true,
            None => fs::read(&path).is_ok_and(|d| d.starts_with(&[0xFF, 0xD8])),
            _ => false,
        };
        if jpeg {
            out.push(path);
        }
    }
}

/// Items tile the input, every payload lies inside its item, and markers are
/// present exactly where the item came from a marker.
pub fn assert_tiles(meta: &JpegMetadata<'_>, data: &[u8], name: &str) {
    let base = data.as_ptr() as usize;
    let mut pos = 0;
    for item in meta.items() {
        let Item {
            offset,
            length,
            payload,
            ..
        } = *item;
        assert_eq!(offset, pos, "{name}: gap or overlap at {offset}");
        assert!(length > 0, "{name}: empty item at {offset}");
        if !payload.is_empty() {
            let p = payload.as_ptr() as usize;
            assert!(
                p >= base + offset && p + payload.len() <= base + offset + length,
                "{name}: payload outside item at {offset}"
            );
        }
        if let Some(m) = item.marker {
            assert_eq!(
                data[offset], 0xFF,
                "{name}: marker item at {offset} without FF"
            );
            assert_eq!(
                data[offset + 1],
                m,
                "{name}: marker byte mismatch at {offset}"
            );
        }
        pos += length;
    }
    assert_eq!(pos, data.len(), "{name}: items do not cover the input");
}
