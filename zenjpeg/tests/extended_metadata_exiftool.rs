//! Cross-check of `container::metadata` against ExifTool, the reference
//! JPEG metadata reader, over the codec-corpus JPEGs.
//!
//! Opt-in: needs `exiftool` on PATH and runs only with the
//! `__oracle-exiftool` feature (`just metadata-oracle`). A missing exiftool
//! fails the test. ExifTool is a test oracle here only; it never writes a
//! fixture.
//!
//! For each file, `exiftool -v3 -a -u` reports, in order, every marker it
//! reads (with the payload size), an "Unknown APPn segment" warning for each
//! APP segment it does not recognize, and every trailer after EOI (with
//! offset and size). The test derives the same three views from our items
//! and compares them:
//!
//! 1. Marker sequence: identical, including restart markers inside scans
//!    (ExifTool lists them; our SOS item holds them in its entropy data).
//! 2. APP identification: an APP segment is identified by both or by
//!    neither, except for signatures outside our table (ExifTool 12.40
//!    additions after 6.99), listed in [`EXIFTOOL_ONLY`].
//! 3. Trailers: every vendor trailer ExifTool reports is one of our items
//!    with the same kind, offset and length; every MPF/preview image it
//!    reports starts one of our `EmbeddedJpeg` items; and each of our vendor
//!    trailers appears in ExifTool's list.
//!
//! Known, settled disagreements are classified by rule, not by file name,
//! and every one is printed. See `zenjpeg/docs/METADATA.md` for each rule's
//! settlement against the spec. The rules and [`EXIFTOOL_ONLY`] describe
//! ExifTool 12.40, so the test requires that version.
//!
//! The file list is the corpus plus the synthetic coverage seeds in
//! `fuzz/seeds/metadata/`, which reach the formats the corpus lacks.
//!
//! A second test runs ExifTool's APP12 Picture Info token regex in Perl over
//! generated inputs and compares its verdict with our identification.
#![cfg(all(feature = "extended-metadata", feature = "__oracle-exiftool"))]

#[path = "support/metadata_corpus.rs"]
mod metadata_corpus;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use zenjpeg::container::metadata::{Item, ItemKind, JpegMetadata, Unparsed};

/// APP signatures ExifTool 12.40 recognizes that are outside our table
/// (they postdate the 6.99 table this model follows). An APP segment that
/// starts with one of these may be identified by ExifTool and `Unknown` here.
const EXIFTOOL_ONLY: &[(u8, &[u8])] = &[
    (0xE0, b"Ocad"),
    (0xE1, b"QVCI\0"),
    (0xE1, b"FLIR\0"),
    (0xE1, b"PARROT\0"),
    // Preview images stored in APP2/APP3 (Samsung, GE, GoPro, BenQ, Rollei).
    (0xE2, b"\xff\xd8\xff"),
    (0xE2, b"QVGA\0\xff\xd8\xff"),
    (0xE2, b"BGTH\xff\xd8\xff"),
    (0xE3, b"\xff\xd8\xff\xdb"),
    (0xE4, b"FPXR\0"),
    (0xE3, b"Stim\0"),
    (0xE3, b"_JPSJPS_"),
    (0xE4, b"SCALADO\0"),
    (0xE5, b"ssuniqueid\0"),
    (0xE6, b"NITF\0"),
    (0xE6, b"TDHD\x01\0\0\0"),
    (0xE6, b"GoPro\0"),
    (0xE7, b"PENTAX \0"),
    (0xE7, b"HUAWEI\0\0"),
    (0xE7, b"\x1aQualcomm Camera Attributes"),
    (0xE9, b"Media Jukebox\0"),
    (0xEA, b"AROT\0"),
    (0xEB, b"HDR_RI "),
];

/// The ExifTool version the rules below were settled against.
const EXIFTOOL_VERSION: &str = "12.40";

fn assert_exiftool_version() {
    let out = Command::new("exiftool")
        .arg("-ver")
        .output()
        .expect("exiftool not found on PATH: install exiftool to run this oracle");
    let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert_eq!(
        version, EXIFTOOL_VERSION,
        "the settlement rules are for ExifTool {EXIFTOOL_VERSION}; re-settle them for {version}"
    );
}

/// The corpus files plus the committed coverage seeds.
fn oracle_files() -> Vec<(&'static str, PathBuf)> {
    let mut files = metadata_corpus::files_by_dataset();
    let seeds = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/metadata");
    let before = files.len();
    for entry in std::fs::read_dir(&seeds).expect("fuzz/seeds/metadata") {
        files.push(("fuzz/seeds/metadata", entry.expect("dir entry").path()));
    }
    assert_eq!(files.len(), before + 8, "coverage seeds missing");
    files
}

/// Files settled under each rule, outside `metadata_corpus::MOVING`: the
/// number of files and the FNV-1a hash of the sorted `name<TAB>detail`
/// lines. A regression that lands inside a rule, or changes how an already
/// settled file is settled, changes its list. When a change is intended,
/// paste the table the test prints.
const SETTLED: &[(&str, usize, u64)] = &[
    ("ExifTool needs FF D8 FF", 79, 0xd17c54b25400c658),
    (
        "ExifTool reads XMP without its signature",
        1,
        0x9b1b47088d75d588,
    ),
    (
        "HasExtendedXMP outside rdf:Description",
        1,
        0xf48608acd5887f16,
    ),
    ("ISO 21496-1 gain map metadata", 1, 0x9ffe776aab906857),
    (
        "JPEG 2000 markers are standalone to ExifTool",
        28,
        0x2c0664fca43c36c2,
    ),
    (
        "MPF image that is not a complete JPEG stream",
        1,
        0xc6f023ce9ed44065,
    ),
    (
        "SOS header ExifTool reads differently (a marker in it)",
        17,
        0xbca14f037c125fb5,
    ),
    ("SPIFF with the T.84 header length", 1, 0x66fa68544c323b5b),
    (
        "appended JPEG that exiftool reports inside an Unknown trailer",
        1,
        0x58fb1f041104fbe9,
    ),
    (
        "exiftool drops the last marker before EOF",
        116,
        0xa216f0c4104204cd,
    ),
    (
        "exiftool stops at a bad or truncated segment",
        170,
        0x233c9e1dc7124329,
    ),
    ("not a JPEG for either", 3, 0x72cae34007183f05),
    ("trailers after a different EOI", 2, 0x65836bd7db5af0c7),
];
/// Files that agree exactly, outside `metadata_corpus::MOVING`.
const AGREE: usize = 1791;
/// Segments whose identification was checked against ExifTool's, and field
/// values compared, outside `metadata_corpus::MOVING`. A parser regression
/// that returns nothing would skip its checks without changing any other
/// pin.
const IDENTITY_CHECKED: usize = 3491;
const FIELDS_CHECKED: usize = 16426;

fn fnv1a(lines: &[String]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for line in lines {
        for &b in line.as_bytes().iter().chain(b"\n") {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// One file's view as `exiftool -v3` prints it.
#[derive(Debug, Default)]
struct ExifToolView {
    file_type_jpeg: bool,
    /// Markers in order (`item` unused).
    markers: Vec<Event>,
    /// `(trailer name, offset, length)` in order.
    trailers: Vec<(String, usize, usize)>,
    /// Size of the extended XMP packet ExifTool parsed after the walk (its
    /// `+ [XMP directory, N bytes]` line outside an APP1 segment).
    extended_xmp: Option<usize>,
}

fn run_exiftool(files: &[PathBuf]) -> BTreeMap<PathBuf, ExifToolView> {
    let mut views = BTreeMap::new();
    for batch in files.chunks(200) {
        let out = Command::new("exiftool")
            .args(["-v3", "-a", "-u", "-charset", "filename=utf8"])
            .args(batch)
            .output()
            .expect("exiftool not found on PATH: install exiftool to run this oracle");
        let text = String::from_utf8_lossy(&out.stdout);
        let mut current: Option<(PathBuf, ExifToolView)> = None;
        // With one file there is no "======== " header.
        if batch.len() == 1 {
            current = Some((batch[0].clone(), ExifToolView::default()));
        }
        for line in text.lines() {
            if let Some(path) = line.strip_prefix("======== ") {
                if let Some((p, v)) = current.take() {
                    views.insert(p, v);
                }
                current = Some((PathBuf::from(path), ExifToolView::default()));
                continue;
            }
            let Some((_, view)) = current.as_mut() else {
                continue;
            };
            let in_app1 = view.markers.last().is_some_and(|m| m.name == "APP1");
            if line.trim_start().starts_with("FileType = JPEG") {
                view.file_type_jpeg = true;
            } else if let Some(size) = line.strip_prefix("  + [XMP directory, ")
                && !in_app1
            {
                view.extended_xmp = size.strip_suffix(" bytes]").and_then(|n| n.parse().ok());
            } else if let Some(rest) = line.strip_prefix("JPEG ") {
                let rest = rest.trim_end_matches(':');
                let (name, bytes) = match rest.rsplit_once(" (") {
                    Some((name, b)) if b.ends_with(" bytes)") => {
                        (name.to_string(), b.trim_end_matches(" bytes)").parse().ok())
                    }
                    _ => (rest.to_string(), None),
                };
                view.markers.push(Event {
                    name,
                    bytes,
                    unknown_app: false,
                    item: 0,
                    hint: None,
                });
            } else if line.contains("Warning = [minor] Unknown APP")
                && let Some(last) = view.markers.last_mut()
            {
                last.unknown_app = true;
            } else if let Some(content) = line.strip_prefix("  ")
                && !content.starts_with(' ')
                && !content.starts_with('|')
                && let Some(last) = view.markers.last_mut()
                && last.hint.is_none()
            {
                last.hint = Some(content.to_string());
            } else if let Some((name, rest)) = line.split_once(" trailer (")
                && !line.starts_with(' ')
            {
                // "<Name> trailer (<n> bytes at offset 0x<hex>):"
                let mut parts = rest.split_whitespace();
                let len = parts.next().and_then(|n| n.parse().ok());
                let off = rest
                    .split("offset 0x")
                    .nth(1)
                    .and_then(|h| usize::from_str_radix(h.trim_end_matches("):"), 16).ok());
                if let (Some(len), Some(off)) = (len, off) {
                    view.trailers.push((name.to_string(), off, len));
                }
            }
        }
        if let Some((p, v)) = current.take() {
            views.insert(p, v);
        }
    }
    views
}

/// ExifTool's `JpegMarkerName`.
fn marker_name(m: u8) -> String {
    match m {
        0x01 => "TEM".into(),
        0xC4 => "DHT".into(),
        0xC8 => "JPGA".into(),
        0xCC => "DAC".into(),
        0xC0..=0xCF => format!("SOF{}", m & 0x0F),
        0xD0..=0xD7 => format!("RST{}", m & 0x0F),
        0xD8 => "SOI".into(),
        0xD9 => "EOI".into(),
        0xDA => "SOS".into(),
        0xDB => "DQT".into(),
        0xDC => "DNL".into(),
        0xDD => "DRI".into(),
        0xDE => "DHP".into(),
        0xDF => "EXP".into(),
        0xE0..=0xEF => format!("APP{}", m & 0x0F),
        0xF0..=0xFD => format!("JPG{}", m & 0x0F),
        0xFE => "COM".into(),
        _ => format!("marker 0x{m:02x}"),
    }
}

/// One marker as ExifTool prints it: name, payload bytes, and whether the
/// APP segment is unrecognized. `item` is the index of our item it came from.
#[derive(Debug, Clone)]
struct Event {
    name: String,
    bytes: Option<usize>,
    unknown_app: bool,
    item: usize,
    /// ExifTool's first line about the segment's content (theirs only).
    hint: Option<String>,
}

/// Our view in ExifTool's terms.
fn our_markers(meta: &JpegMetadata<'_>, data: &[u8]) -> Vec<Event> {
    let mut out = Vec::new();
    let end = meta.primary_image_end().unwrap_or(usize::MAX);
    for (i, item) in meta.items().iter().enumerate() {
        if item.offset >= end {
            break;
        }
        let Some(m) = item.marker else {
            // ExifTool prints the restart markers it meets while it skips
            // stray bytes; ours stay inside the stray-bytes item.
            if let ItemKind::Unparsed(Unparsed::ExtraneousBytes) = item.kind {
                push_restarts(&mut out, item.payload, i);
            }
            continue;
        };
        if i == 0 && matches!(item.kind, ItemKind::Soi) {
            continue; // ExifTool reads the leading SOI as the file header
        }
        if matches!(item.kind, ItemKind::Unparsed(_)) {
            // ExifTool's table makes SOS standalone, so it prints a bad-length
            // or truncated SOS; for any other marker it stops instead.
            if m == 0xDA {
                out.push(Event {
                    name: "SOS".into(),
                    bytes: None,
                    unknown_app: false,
                    item: i,
                    hint: None,
                });
                // ExifTool scans the bytes after a truncated SOS too. (A
                // bad-length SOS item is only its marker and length word;
                // the stray-bytes item after it lists what follows.)
                if let ItemKind::Unparsed(Unparsed::TruncatedSegment) = item.kind {
                    push_restarts(
                        &mut out,
                        &data[item.offset + 2..item.offset + item.length],
                        i,
                    );
                }
            }
            continue;
        }
        let bytes = (item.length > 2 && m != 0xDA).then(|| item.length - 4);
        let unknown_app = (0xE0..=0xEF).contains(&m) && matches!(item.kind, ItemKind::Unknown);
        out.push(Event {
            name: marker_name(m),
            bytes,
            unknown_app,
            item: i,
            hint: None,
        });
        if m == 0xDA {
            // Restart markers inside the scan, as ExifTool lists them.
            // ExifTool never reads `Ls` (its marker table gives SOS no
            // length): it scans every byte after the marker, the scan header
            // included, whether or not the header parses (`Unknown`).
            push_restarts(
                &mut out,
                &data[item.offset + 2..item.offset + item.length],
                i,
            );
        }
    }
    out
}

/// Restart markers in `bytes` (after a fill run, not stuffed), as events.
fn push_restarts(out: &mut Vec<Event>, bytes: &[u8], item: usize) {
    let mut w = 0;
    while w + 1 < bytes.len() {
        if bytes[w] == 0xFF && bytes[w + 1] != 0xFF {
            if (0xD0..=0xD7).contains(&bytes[w + 1]) {
                out.push(Event {
                    name: marker_name(bytes[w + 1]),
                    bytes: None,
                    unknown_app: false,
                    item,
                    hint: None,
                });
            }
            w += 2;
        } else {
            w += 1;
        }
    }
}

/// Our kind names that may match the format ExifTool's first content line
/// names for a segment, or `None` when the line does not name one.
fn expected_kinds(marker: &str, hint: Option<&str>) -> Option<&'static [&'static str]> {
    let hint = hint?;
    Some(match (marker, hint) {
        ("APP1", h) if h.starts_with("ExifByteOrder") => &["EXIF"],
        ("APP3", h) if h.starts_with("ExifByteOrder") => &["Meta"],
        // The last extended XMP chunk triggers the parse of the whole packet.
        ("APP1", h) if h.starts_with("+ [XMP directory") => &["XMP", "ExtendedXMP"],
        ("APP2", h) if h.starts_with("ICC_Profile chunk") => &["ICC_Profile"],
        ("APP2", h) if h.starts_with("+ [MPF0 directory") => &["MPF"],
        ("APP13", h) if h.starts_with("+ [Photoshop directory") => &["Photoshop"],
        ("APP14", h) if h.starts_with("+ [BinaryData directory") => &["Adobe"],
        ("APP13", h) if h.starts_with("+ [BinaryData directory") => &["Adobe_CM"],
        ("APP8", h) if h.starts_with("+ [BinaryData directory") => &["SPIFF"],
        ("APP0", h) if h.starts_with("+ [BinaryData directory") => &["JFIF", "AVI1"],
        ("COM", h) if h.starts_with("Comment =") => &["Comment"],
        _ => return None,
    })
}

fn lenient_xmp(payload: &[u8]) -> bool {
    let has = |needle: &[u8]| payload.windows(needle.len()).any(|w| w == needle);
    payload.starts_with(b"http")
        || payload.starts_with(b"XMP\0")
        || has(b"<exif:")
        || has(b"<?xpacket")
}

fn exiftool_only_signature(marker: u8, payload: &[u8]) -> bool {
    EXIFTOOL_ONLY
        .iter()
        .any(|&(m, sig)| m == marker && payload.starts_with(sig))
}

#[derive(Default)]
struct Tally {
    files: usize,
    agree: usize,
    /// Exactly agreeing files outside the moving datasets.
    agree_pinned: usize,
    identity_checked: usize,
    fields_checked: usize,
    /// `identity_checked` and `fields_checked` outside the moving datasets.
    identity_checked_pinned: usize,
    fields_checked_pinned: usize,
    /// Files whose fields were compared over the items before a settled
    /// divergence only.
    prefix_compared: usize,
    /// Whether the file being compared is in a moving dataset.
    moving: bool,
    /// `(file, detail, moving)` per rule.
    rules: BTreeMap<&'static str, Vec<(String, String, bool)>>,
    failures: Vec<String>,
}

impl Tally {
    fn settled(&mut self, rule: &'static str, file: &str, detail: String) {
        let moving = self.moving;
        self.rules
            .entry(rule)
            .or_default()
            .push((file.to_string(), detail, moving));
    }
}

#[test]
fn exiftool_agrees_on_every_segment_and_trailer() {
    assert_exiftool_version();
    let files = oracle_files();
    let paths: Vec<PathBuf> = files.iter().map(|(_, p)| p.clone()).collect();
    let views = run_exiftool(&paths);
    let tags = run_exiftool_json(&paths);
    let mut tally = Tally::default();
    for (dataset, path) in &files {
        let data = std::fs::read(path).expect("read corpus file");
        let name = short(path);
        tally.moving = metadata_corpus::MOVING.contains(dataset);
        let meta = JpegMetadata::parse(&data).expect("parse");
        metadata_corpus::assert_tiles(&meta, &data, &name);
        let view = views
            .get(path)
            .unwrap_or_else(|| panic!("{name}: no exiftool output"));
        // `ORACLE_DUMP=<substring>` prints both views of matching files.
        if let Ok(pat) = std::env::var("ORACLE_DUMP")
            && name.contains(&pat)
        {
            println!(
                "== {name}\nexiftool: {:?}\n          {:?}",
                view.markers, view.trailers
            );
            for item in meta.items() {
                let kind = match &item.kind {
                    ItemKind::Unparsed(why) => format!("Unparsed({why:?})"),
                    k => k.name().to_string(),
                };
                println!(
                    "  {:>8} {:>8} {:?} {kind}",
                    item.offset, item.length, item.marker
                );
            }
        }
        tally.files += 1;
        let agree_before = tally.agree;
        let checks_before = (tally.identity_checked, tally.fields_checked);
        let field_items = compare(&name, &data, &meta, view, &mut tally);
        if tally.agree > agree_before && !tally.moving {
            tally.agree_pinned += 1;
        }
        if field_items > 0 {
            let tags = tags
                .get(path)
                .unwrap_or_else(|| panic!("{name}: no exiftool JSON"));
            if field_items < meta.items().len() {
                tally.prefix_compared += 1;
            }
            compare_fields(&name, &meta, tags, field_items, &mut tally);
        }
        if !tally.moving {
            tally.identity_checked_pinned += tally.identity_checked - checks_before.0;
            tally.fields_checked_pinned += tally.fields_checked - checks_before.1;
        }
    }

    let mut report = String::new();
    writeln!(
        report,
        "exiftool oracle: {} files, {} agree exactly; {} segments identity-checked; {} field values compared ({} files over the prefix before a settled divergence)",
        tally.files,
        tally.agree,
        tally.identity_checked,
        tally.fields_checked,
        tally.prefix_compared
    )
    .unwrap();
    let mut settled_files = std::collections::BTreeSet::new();
    let mut entries = 0;
    let mut table = String::new();
    let mut pins: Vec<(&str, usize, u64)> = Vec::new();
    for (rule, cases) in &tally.rules {
        let mut pinned: Vec<String> = cases
            .iter()
            .filter(|c| !c.2)
            .map(|c| format!("{}\t{}", c.0, c.1))
            .collect();
        pinned.sort_unstable();
        pinned.dedup();
        let mut pinned_files: Vec<&str> = cases
            .iter()
            .filter(|c| !c.2)
            .map(|c| c.0.as_str())
            .collect();
        pinned_files.sort_unstable();
        pinned_files.dedup();
        writeln!(report, "settled [{rule}]: {} files", cases.len()).unwrap();
        for (file, detail, moving) in cases {
            let tag = if *moving { " (moving dataset)" } else { "" };
            writeln!(report, "    {file}: {detail}{tag}").unwrap();
            settled_files.insert(file.as_str());
            entries += 1;
        }
        writeln!(
            table,
            "    ({rule:?}, {}, 0x{:016x}),",
            pinned_files.len(),
            fnv1a(&pinned)
        )
        .unwrap();
        pins.push((rule, pinned_files.len(), fnv1a(&pinned)));
    }
    writeln!(
        report,
        "{entries} settled entries over {} distinct files; pinned table:\n{table}\
         agree outside moving datasets: {}; identity checks {}, field values {}",
        settled_files.len(),
        tally.agree_pinned,
        tally.identity_checked_pinned,
        tally.fields_checked_pinned
    )
    .unwrap();
    println!("{report}");
    assert!(
        tally.failures.is_empty(),
        "{} unsettled disagreements:\n{}",
        tally.failures.len(),
        tally.failures.join("\n")
    );
    pins.retain(|p| p.1 > 0);
    assert_eq!(
        pins, SETTLED,
        "settled-rule lists changed; see the table above"
    );
    assert_eq!(tally.agree_pinned, AGREE, "exactly agreeing files changed");
    assert_eq!(
        (tally.identity_checked_pinned, tally.fields_checked_pinned),
        (IDENTITY_CHECKED, FIELDS_CHECKED),
        "identity checks or compared field values changed"
    );
}

fn short(path: &Path) -> String {
    let s = path.to_string_lossy();
    s.split("codec-corpus/v1/")
        .nth(1)
        .or_else(|| s.split("zenjpeg/fuzz/").nth(1))
        .unwrap_or(&s)
        .to_string()
}

/// Compare one file. Returns how many leading items have field values
/// ExifTool read the same way: all of them when the marker sequences agree
/// exactly; the items before the divergence when a settled rule explains it
/// (ExifTool printed, so processed, every segment before it), unless the rule
/// is that ExifTool does not take the file as a JPEG; otherwise none.
fn compare(
    name: &str,
    data: &[u8],
    meta: &JpegMetadata<'_>,
    view: &ExifToolView,
    tally: &mut Tally,
) -> usize {
    let items = meta.items();
    if matches!(
        items.first().map(|i| &i.kind),
        Some(ItemKind::Unparsed(Unparsed::NotJpeg))
    ) {
        if view.file_type_jpeg {
            tally
                .failures
                .push(format!("{name}: we say not a JPEG, exiftool reads JPEG"));
        } else {
            tally.settled("not a JPEG for either", name, String::new());
        }
        return 0;
    }
    let ours = our_markers(meta, data);
    let theirs = &view.markers;
    let mut clean = true;

    // 1: marker sequence.
    let same = |a: &Event, b: &Event| a.name == b.name && a.bytes == b.bytes;
    let common = ours.len().min(theirs.len());
    let k = (0..common)
        .find(|&k| !same(&ours[k], &theirs[k]))
        .unwrap_or(common);
    let mut field_items = items.len();
    if k < ours.len() || k < theirs.len() {
        clean = false;
        field_items = 0;
        if let Some((rule, detail)) = settle_markers(data, meta, &ours, theirs, k) {
            if rule != NEEDS_FF_D8_FF {
                field_items = ours.get(k).map_or(items.len(), |e| e.item);
            }
            tally.settled(rule, name, detail);
        } else {
            let show = |v: &[Event], k: usize| {
                v.get(k)
                    .map_or("<end>".to_string(), |e| format!("{} {:?}", e.name, e.bytes))
            };
            let window = |v: &[Event]| {
                let lo = k.saturating_sub(3);
                let hi = (k + 4).min(v.len());
                v[lo.min(hi)..hi]
                    .iter()
                    .map(|e| format!("{} {:?}", e.name, e.bytes))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            tally.failures.push(format!(
                "{name}: marker #{k} exiftool {} / ours {}\n      exiftool: {}\n      ours:     {}",
                show(theirs, k),
                show(&ours, k),
                window(theirs),
                window(&ours),
            ));
        }
    }

    // Field values are comparable over the agreed markers, whatever the
    // identification step below settles.

    // 2: APP identification, over the markers both sequences agree on.
    for (ev, theirs) in ours[..k].iter().zip(&theirs[..k]) {
        // An `Unknown` of ours is judged by the identified/unknown rule below.
        if let Some(allowed) = expected_kinds(&theirs.name, theirs.hint.as_deref())
            && !matches!(items[ev.item].kind, ItemKind::Unknown)
        {
            let ours_kind = items[ev.item].kind.name();
            tally.identity_checked += 1;
            if !allowed.contains(&ours_kind) {
                clean = false;
                tally.failures.push(format!(
                    "{name}: {} at {}: exiftool reads {allowed:?}, we read {ours_kind}",
                    ev.name, items[ev.item].offset
                ));
            }
        }
        if !ev.name.starts_with("APP") || ev.unknown_app == theirs.unknown_app {
            continue;
        }
        clean = false;
        let item = &items[ev.item];
        let m = item.marker.unwrap_or(0);
        let sig = String::from_utf8_lossy(&item.payload[..item.payload.len().min(16)]).to_string();
        if ev.unknown_app && exiftool_only_signature(m, item.payload) {
            tally.settled(
                "ExifTool-only signature",
                name,
                format!("{} {sig:?}", ev.name),
            );
        } else if !ev.unknown_app
            && matches!(item.kind, ItemKind::Spiff(_))
            && item.payload.len() == 30
        {
            // T.84 F.2.1's 24-byte header; ExifTool identifies only the
            // 26-byte layout of its one sample.
            tally.settled(
                "SPIFF with the T.84 header length",
                name,
                format!("at {}", item.offset),
            );
        } else if !ev.unknown_app && matches!(item.kind, ItemKind::GainMapIso21496(_)) {
            // ISO 21496-1 (2025) postdates ExifTool 12.40.
            tally.settled(
                "ISO 21496-1 gain map metadata",
                name,
                format!("at {}", item.offset),
            );
        } else if ev.unknown_app && m == 0xE1 && lenient_xmp(item.payload) {
            // ExifTool reads any APP1 starting "http" or "XMP\0", or holding
            // "<exif:" or "<?xpacket", as XMP (with a warning); XMP Part 3
            // 1.1.3 requires the exact namespace signature, which we do.
            tally.settled(
                "ExifTool reads XMP without its signature",
                name,
                format!("{sig:?}"),
            );
        } else {
            tally.failures.push(format!(
                "{name}: {} at {} identified by {} only ({}, {sig:?})",
                ev.name,
                item.offset,
                if ev.unknown_app { "exiftool" } else { "us" },
                item.kind.name()
            ));
        }
    }

    // 2b: extended XMP. After the walk ExifTool parses the complete
    // extended packet whose GUID the standard packet's HasExtendedXMP names
    // (without checking the MD5); compare that with our reassembly.
    let ext_chunks = items
        .iter()
        .filter(|i| matches!(i.kind, ItemKind::ExtendedXmp(_)))
        .count();
    if ext_chunks > 0 || view.extended_xmp.is_some() {
        let theirs_parsed = view.extended_xmp;
        let ours_parsed = meta.extended_xmp().ok().flatten().map(|v| v.len());
        if theirs_parsed.is_some() && ours_parsed.is_some() {
            tally.identity_checked += 1;
        }
        if theirs_parsed != ours_parsed {
            clean = false;
            let rdf = meta
                .xmp()
                .ok()
                .flatten()
                .is_some_and(|x| x.windows(15).any(|w| w == b"rdf:Description"));
            if ours_parsed.is_some() && theirs_parsed.is_none() && !rdf {
                // XMP Part 3 names the property, not where it sits; ExifTool
                // reads a HasExtendedXMP outside rdf:Description as a
                // different tag (e.g. `XHasExtendedXMP`) and never looks for
                // the extended packet.
                tally.settled(
                    "HasExtendedXMP outside rdf:Description",
                    name,
                    format!("{ext_chunks} chunks"),
                );
            } else {
                tally.failures.push(format!(
                    "{name}: extended XMP exiftool {theirs_parsed:?} bytes, ours {:?}",
                    meta.extended_xmp().map(|o| o.map(|v| v.len()))
                ));
            }
        }
    }

    // 3: trailers. ExifTool's first trailer starts at its EOI; when a
    // settled marker rule moved that EOI, the trailers are not comparable.
    let markers_clean = clean;
    let their_eoi = view.trailers.iter().map(|t| t.1).min();
    if !markers_clean && their_eoi.is_some() && their_eoi != meta.primary_image_end() {
        tally.settled(
            "trailers after a different EOI",
            name,
            format!(
                "exiftool {their_eoi:?}, ours {:?}",
                meta.primary_image_end()
            ),
        );
        return field_items;
    }
    let trailer_items: Vec<&Item<'_>> = match meta.primary_image_end() {
        Some(end) => items.iter().filter(|i| i.offset >= end).collect(),
        None => Vec::new(),
    };
    let mpf = meta.mpf().ok().flatten().unwrap_or_default();
    for (tname, off, len) in &view.trailers {
        let at = trailer_items.iter().find(|i| i.offset == *off);
        let ok = match tname.as_str() {
            "Unknown" => trailer_items
                .iter()
                .any(|i| i.offset <= *off && *off < i.offset + i.length),
            n if n.starts_with("MPImage") || n == "PreviewImage" => {
                if at.is_some_and(|i| matches!(i.kind, ItemKind::EmbeddedJpeg) && i.length == *len)
                {
                    true
                } else if at
                    .is_some_and(|i| matches!(i.kind, ItemKind::UnknownTrailer) && i.length == *len)
                    && mpf.iter().any(|e| e.offset == *off && e.size == *len)
                {
                    // The MPF directory names these bytes, but they are not a
                    // complete JPEG stream, so we keep them opaque.
                    clean = false;
                    tally.settled(
                        "MPF image that is not a complete JPEG stream",
                        name,
                        format!("{tname} ({len} bytes at {off})"),
                    );
                    true
                } else {
                    false
                }
            }
            other => at.is_some_and(|i| i.kind.name() == vendor_name(other) && i.length == *len),
        };
        if !ok {
            clean = false;
            tally.failures.push(format!(
                "{name}: exiftool trailer {tname} ({len} bytes at {off}) not matched; ours: {:?}",
                trailer_items
                    .iter()
                    .map(|i| (i.kind.name(), i.offset, i.length))
                    .collect::<Vec<_>>()
            ));
        }
    }
    for item in &trailer_items {
        let vendor = !matches!(
            item.kind,
            ItemKind::EmbeddedJpeg
                | ItemKind::Padding
                | ItemKind::UnknownTrailer
                | ItemKind::Unparsed(_)
        );
        if vendor
            && !view
                .trailers
                .iter()
                .any(|(n, off, _)| *off == item.offset && vendor_name(n) == item.kind.name())
        {
            clean = false;
            tally.failures.push(format!(
                "{name}: our {} trailer at {} is not in exiftool's list",
                item.kind.name(),
                item.offset
            ));
        }
        if matches!(item.kind, ItemKind::EmbeddedJpeg)
            && !view.trailers.iter().any(|(n, off, _)| {
                *off == item.offset && (n.starts_with("MPImage") || n == "PreviewImage")
            })
        {
            clean = false;
            let end = item.offset + item.length;
            let covered = view
                .trailers
                .iter()
                .any(|(n, off, len)| n == "Unknown" && *off <= item.offset && end <= off + len);
            if covered {
                tally.settled(
                    "appended JPEG that exiftool reports inside an Unknown trailer",
                    name,
                    format!("at {} ({} bytes)", item.offset, item.length),
                );
            } else {
                tally.failures.push(format!(
                    "{name}: our appended JPEG at {} ({} bytes) is in no exiftool trailer",
                    item.offset, item.length
                ));
            }
        }
    }
    if clean {
        tally.agree += 1;
    }
    field_items
}

/// `exiftool -j -n -G1 -a -u`: each file's tags as `group:name -> value`.
/// A key that occurs more than once holds the value ExifTool's tag priority
/// keeps: the first for priority-0 tags (COM, the Ducky tags), otherwise the
/// last (measured with ExifTool 12.40).
fn run_exiftool_json(
    files: &[PathBuf],
) -> BTreeMap<PathBuf, serde_json::Map<String, serde_json::Value>> {
    let mut out = BTreeMap::new();
    for batch in files.chunks(200) {
        let o = Command::new("exiftool")
            .args(["-j", "-n", "-G1", "-a", "-u", "-charset", "filename=utf8"])
            .args(batch)
            .output()
            .expect("exiftool not found on PATH: install exiftool to run this oracle");
        let parsed: Vec<serde_json::Map<String, serde_json::Value>> =
            serde_json::from_slice(&o.stdout).expect("exiftool -j output");
        for obj in parsed {
            let src = obj["SourceFile"].as_str().expect("SourceFile").to_string();
            out.insert(PathBuf::from(src), obj);
        }
    }
    out
}

/// Compare the fields we parse with ExifTool's values for the same tags:
/// SOF, JFIF, Adobe, Adobe_CM, AVI1, Ducky, and an ASCII COM. Only a kind
/// that occurs once in the primary image is compared: with duplicates,
/// which value ExifTool's JSON keeps depends on its tag priorities. Only the
/// first `field_items` items are compared (see `compare`).
fn compare_fields(
    name: &str,
    meta: &JpegMetadata<'_>,
    tags: &serde_json::Map<String, serde_json::Value>,
    field_items: usize,
    tally: &mut Tally,
) {
    let end = meta.primary_image_end().unwrap_or(usize::MAX);
    // ExifTool extracts APP metadata only before the first SOS (it still
    // reads frame headers after it).
    let first_sos = meta
        .items()
        .iter()
        .find(|i| i.marker == Some(0xDA))
        .map_or(end, |i| i.offset);
    let all: Vec<(usize, ItemKind<'_>)> = meta
        .items()
        .iter()
        .enumerate()
        .take_while(|(_, i)| i.offset < end)
        .filter(|(_, i)| i.offset < first_sos || matches!(i.kind, ItemKind::Sof(_)))
        .map(|(n, i)| (n, i.kind.clone()))
        .collect();
    // Keep each kind only when it occurs once in the whole primary image,
    // then only the items ExifTool read the same way.
    let once = |k: &ItemKind<'_>| {
        all.iter()
            .filter(|o| core::mem::discriminant(&o.1) == core::mem::discriminant(k))
            .count()
            == 1
    };
    let primary: Vec<ItemKind<'_>> = all
        .iter()
        .filter(|(n, k)| *n < field_items && once(k))
        .map(|(_, k)| k.clone())
        .collect();
    let last_sof = primary.iter().rev().find_map(|k| match k {
        ItemKind::Sof(s) => Some(*s),
        _ => None,
    });
    if let Some(sof) = last_sof
        && sof.num_components == 3
        && sof.components().count() == 3
    {
        // ExifTool prints the ratios as numbers; ours are whole or None.
        let theirs = tags
            .get("File:YCbCrSubSampling")
            .and_then(|v| v.as_str())
            .and_then(|v| {
                let r: Vec<f64> = v.split(' ').filter_map(|x| x.parse().ok()).collect();
                (r.len() == 2 && r.iter().all(|x| x.fract() == 0.0))
                    .then(|| (r[0] as u8, r[1] as u8))
            });
        tally.fields_checked += 1;
        if theirs != sof.ycbcr_subsampling() {
            tally.failures.push(format!(
                "{name}: YCbCrSubSampling exiftool {:?}, ours {:?}",
                tags.get("File:YCbCrSubSampling"),
                sof.ycbcr_subsampling()
            ));
        }
    }
    let mut check = |key: &str, ours: serde_json::Value| {
        let Some(theirs) = tags.get(key) else {
            tally
                .failures
                .push(format!("{name}: exiftool has no {key}, ours {ours}"));
            return;
        };
        tally.fields_checked += 1;
        if *theirs != ours {
            tally
                .failures
                .push(format!("{name}: {key} exiftool {theirs}, ours {ours}"));
        }
    };
    use serde_json::json;
    if let Some(sof) = last_sof {
        check("File:ImageWidth", json!(sof.width));
        check("File:ImageHeight", json!(sof.height));
        check("File:EncodingProcess", json!(sof.process));
        check("File:BitsPerSample", json!(sof.precision));
        check("File:ColorComponents", json!(sof.num_components));
    }
    if let Some(j) = primary.iter().rev().find_map(|k| match k {
        ItemKind::Jfif(j) => Some(*j),
        _ => None,
    }) {
        check(
            "JFIF:JFIFVersion",
            json!(format!("{} {}", j.version_major, j.version_minor)),
        );
        check("JFIF:ResolutionUnit", json!(j.units));
        check("JFIF:XResolution", json!(j.x_density));
        check("JFIF:YResolution", json!(j.y_density));
    }
    if let Some(a) = primary.iter().rev().find_map(|k| match k {
        ItemKind::Adobe(a) => Some(*a),
        _ => None,
    }) {
        check("Adobe:DCTEncodeVersion", json!(a.dct_encode_version));
        check("Adobe:APP14Flags0", json!(a.flags0));
        check("Adobe:APP14Flags1", json!(a.flags1));
        check("Adobe:ColorTransform", json!(a.color_transform));
    }
    if let Some(cm) = primary.iter().rev().find_map(|k| match k {
        ItemKind::AdobeCm(t) => Some(*t),
        _ => None,
    }) {
        check("AdobeCM:AdobeCMType", json!(cm));
    }
    if let Some(Some(field)) = primary.iter().rev().find_map(|k| match k {
        ItemKind::Avi1(a) => Some(a.interleaved_field),
        _ => None,
    }) {
        check("AVI1:InterleavedField", json!(field));
    }
    if let Some(d) = primary.iter().rev().find_map(|k| match k {
        ItemKind::Ducky(d) => Some(*d),
        _ => None,
    }) {
        if let Some(q) = d.quality() {
            check("Ducky:Quality", json!(q));
        }
        if let Some(c) = d.comment() {
            check("Ducky:Comment", json!(c.chars().collect::<String>()));
        }
        if let Some(c) = d.copyright() {
            check("Ducky:Copyright", json!(c.chars().collect::<String>()));
        }
    }
    let coms: Vec<(usize, &Item<'_>)> = meta
        .items()
        .iter()
        .enumerate()
        .take_while(|(_, i)| i.offset < first_sos)
        .filter(|(_, i)| matches!(i.kind, ItemKind::Comment))
        .collect();
    let photostudio = all
        .iter()
        .any(|(_, k)| matches!(k, ItemKind::PhotoStudioComment(_)));
    if let [(n, com)] = coms[..]
        && n < field_items
        && !photostudio
    {
        // ExifTool strips trailing NULs from a comment.
        let text = com.payload.strip_suffix(b"\0").map_or(com.payload, |t| {
            let mut t = t;
            while let Some(s) = t.strip_suffix(b"\0") {
                t = s;
            }
            t
        });
        if !text.is_empty() && text.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            check("File:Comment", json!(String::from_utf8_lossy(text)));
        }
    }
}

/// The rule under which ExifTool does not read the file as a JPEG at all.
const NEEDS_FF_D8_FF: &str = "ExifTool needs FF D8 FF";

/// Classify a marker-sequence disagreement that starts at index `k` by one
/// of the settled rules (see `zenjpeg/docs/METADATA.md`), or `None`.
fn settle_markers(
    data: &[u8],
    meta: &JpegMetadata<'_>,
    ours: &[Event],
    theirs: &[Event],
    k: usize,
) -> Option<(&'static str, String)> {
    let items = meta.items();
    // ExifTool takes a file as JPEG only when it starts FF D8 FF; otherwise it
    // rejects it or scans for an embedded image. We follow libjpeg: SOI, then
    // resynchronize at the next marker.
    if data.len() > 2 && data[2] != 0xFF {
        return Some((NEEDS_FF_D8_FF, format!("byte 2 is {:#04x}", data[2])));
    }
    // ExifTool's marker table gives the JPEG 2000 codestream markers
    // (0x30..=0x3F, 0x4F) no length; T.81 Table B.1 makes only SOI, EOI,
    // RSTn and TEM standalone, so we read their length word.
    let j2c = |e: &Event| {
        e.bytes.is_none()
            && e.name.strip_prefix("marker 0x").is_some_and(|h| {
                u8::from_str_radix(h, 16).is_ok_and(|m| (0x30..=0x3F).contains(&m) || m == 0x4F)
            })
    };
    if theirs.get(k).is_some_and(j2c) {
        return Some((
            "JPEG 2000 markers are standalone to ExifTool",
            theirs[k].name.clone(),
        ));
    }
    // An SOS whose header ExifTool reads differently. We skip `Ls` bytes as
    // T.81 B.2.3 defines them; ExifTool never reads `Ls` (its marker table
    // gives SOS no length) and scans every byte after the marker for the next
    // marker, skipping `0xFF` fill and stuffed `FF 00`. So a header 0xFF
    // whose fill run ends, within the header or at the first byte after it,
    // in a byte other than 0x00 is a marker to ExifTool and not to us, unless
    // it is a restart marker: ExifTool lists those as standalone markers, and
    // so do our restart events, which cover every byte after the marker
    // (`our_markers`). A run that reaches further is read by our walker too,
    // from the header's end. A wrong `Ls` alone changes nothing for ExifTool, so it
    // is not odd, and neither is a bad-length SOS (the stray-bytes item after
    // it lists its restarts). A truncated SOS is odd only when its bytes hold
    // a marker other than a restart marker.
    // Only an SOS at the divergence qualifies: the event at it, the event
    // before it, or one in the run of events identical (name and size) to
    // the divergent one that ends there (identical events cannot place the
    // divergence any closer).
    let odd_sos = |i: &Item<'_>| {
        if i.marker != Some(0xDA) {
            return false;
        }
        let bytes = match i.kind {
            ItemKind::Unparsed(Unparsed::TruncatedSegment) => i.offset + 2..i.offset + i.length,
            // Bad length (marker and length word only): nothing ExifTool
            // reads differently. No other `Unparsed` item carries a marker.
            ItemKind::Unparsed(_) => return false,
            _ => i.offset + 2..i.offset + 4 + i.payload.len(),
        };
        // A run of 0xFF in `bytes` is a marker to ExifTool and not to us when
        // the byte that ends it lies in `bytes` or is the first byte after
        // them, and is neither 0x00 nor a restart marker. A run that reaches
        // further continues into bytes our walker reads itself, from where
        // `bytes` end: the same marker, stuffing or restart marker.
        let end = bytes.end;
        let marker_at = |j: usize| {
            data[j] == 0xFF
                && data[j..]
                    .iter()
                    .position(|&b| b != 0xFF)
                    .is_some_and(|run| {
                        let at = j + run;
                        at <= end && data[at] != 0x00 && !(0xD0..=0xD7).contains(&data[at])
                    })
        };
        bytes.into_iter().any(marker_at)
    };
    let divergent = theirs.get(k).or(ours.get(k));
    let identical = |e: &Event| divergent.is_some_and(|d| d.name == e.name && d.bytes == e.bytes);
    let mut run_start = k.min(ours.len());
    while run_start > 0 && identical(&ours[run_start - 1]) {
        run_start -= 1;
    }
    let at_divergence = ours
        .get(run_start.saturating_sub(1)..(k + 1).min(ours.len()))
        .unwrap_or_default()
        .iter()
        .map(|e| e.item)
        .find(|&i| odd_sos(&items[i]));
    if let Some(sos) = at_divergence {
        return Some((
            "SOS header ExifTool reads differently (a marker in it)",
            format!("SOS at {}", items[sos].offset),
        ));
    }
    if k == theirs.len() {
        // ExifTool reads one segment ahead and stops on a bad length word or
        // a truncated segment without printing the segment before it; we
        // report the bad bytes and keep walking. So ExifTool's list ends one
        // event before the first bad segment (or right at it).
        if let Some(bad) = items.iter().position(|i| {
            matches!(
                i.kind,
                ItemKind::Unparsed(Unparsed::BadLength | Unparsed::TruncatedSegment)
            )
        }) {
            let before = ours.iter().take_while(|e| e.item < bad).count();
            if theirs.len() + 1 == before || theirs.len() == before {
                return Some((
                    "exiftool stops at a bad or truncated segment",
                    format!(
                        "exiftool {} markers, ours {} before it",
                        theirs.len(),
                        before
                    ),
                ));
            }
        }
        // The same read-ahead never prints the last marker of a file that
        // ends without EOI.
        if meta.primary_image_end().is_none() && ours.len() == theirs.len() + 1 {
            return Some((
                "exiftool drops the last marker before EOF",
                ours[k].name.clone(),
            ));
        }
    }
    None
}

/// Our item name for an ExifTool trailer name.
fn vendor_name(exiftool: &str) -> &str {
    match exiftool {
        n if n.starts_with("FotoStation") => "FotoStation",
        n => n,
    }
}

/// ExifTool's `ProcessAPP12` token regex, whose first match makes an APP12
/// segment Picture Info. A copy of the line in APP12.pm; the test checks it
/// against the installed ExifTool before using it.
const PICTURE_INFO_REGEX: &str = r"(\[.*?\]|[\w#-]+=[\x20-\x7e]+?(?=\s*([\n\r\0]|[\w#-]+=|\[|$)))";

#[test]
fn picture_info_identification_matches_exiftool_regex() {
    // ExifTool's own line, read from the installed module.
    let module = Command::new("perl")
        .args([
            "-MImage::ExifTool::APP12",
            "-e",
            "print $INC{'Image/ExifTool/APP12.pm'}",
        ])
        .output()
        .expect("perl not found on PATH (it ships with exiftool)");
    let module = String::from_utf8(module.stdout).expect("module path");
    let source = std::fs::read_to_string(&module).unwrap_or_else(|e| panic!("{module}: {e}"));
    let installed: Vec<&str> = source
        .lines()
        .filter_map(|l| {
            l.split_once("while ($$dataPt =~ /")?
                .1
                .strip_suffix("/g) {")
        })
        .collect();
    assert_eq!(installed, [PICTURE_INFO_REGEX], "APP12.pm's regex changed");

    // Every string of up to 5 bytes over a small structural alphabet, every
    // string of up to 4 bytes over an alphabet with one byte of each class
    // the regex or our matcher distinguishes (`[`, `]`, `=`, letter, digit,
    // `_`, `#`, `-`, other printable, space, each other whitespace byte,
    // NUL, other control, DEL, high byte), and pseudo-random longer strings
    // over the wide alphabet. Perl runs ExifTool's regex; we identify the
    // same bytes as an APP12 segment.
    const SMALL: &[u8] = b"a=[] \n\r\0\x80\t#";
    const WIDE: &[u8] = b"a1_#-![]= \t\n\r\x0b\x0c\0\x1f\x7f\x80";
    let mut inputs: Vec<Vec<u8>> = vec![Vec::new()];
    for (alphabet, max_len) in [(SMALL, 5), (WIDE, 4)] {
        let mut frontier: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in &frontier {
                for &b in alphabet {
                    let mut s = prefix.clone();
                    s.push(b);
                    next.push(s);
                }
            }
            inputs.extend(next.iter().cloned());
            frontier = next;
        }
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_random = || {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for _ in 0..100_000 {
        let len = 5 + (next_random() % 36) as usize;
        let s = (0..len)
            .map(|_| WIDE[(next_random() % WIDE.len() as u64) as usize])
            .collect();
        inputs.push(s);
    }
    let mut hex = String::new();
    for s in &inputs {
        for b in s {
            write!(hex, "{b:02x}").unwrap();
        }
        hex.push('\n');
    }
    let script = format!(
        "while (<STDIN>) {{ chomp; my $s = pack('H*', $_); print(($s =~ /{PICTURE_INFO_REGEX}/) ? '1' : '0'); }}"
    );
    let mut child = Command::new("perl")
        .args(["-e", &script])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("perl not found on PATH (it ships with exiftool)");
    // Feed stdin from a thread: perl's verdicts fill the stdout pipe while
    // the input is still being written.
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        use std::io::Write as _;
        stdin.write_all(hex.as_bytes()).unwrap();
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout.len(), inputs.len(), "one verdict per input");
    let mut mismatches = Vec::new();
    let mut matched = 0;
    for (s, verdict) in inputs.iter().zip(&out.stdout) {
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xEC];
        jpeg.extend_from_slice(&u16::try_from(s.len() + 2).unwrap().to_be_bytes());
        jpeg.extend_from_slice(s);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        let meta = JpegMetadata::parse(&jpeg).unwrap();
        let ours = meta.items()[1].kind == ItemKind::PictureInfo;
        let theirs = *verdict == b'1';
        matched += usize::from(theirs);
        if ours != theirs {
            mismatches.push(format!("{s:?}: perl {theirs}, ours {ours}"));
        }
    }
    println!(
        "picture info: {} inputs, {matched} matched by the regex, {} mismatches",
        inputs.len(),
        mismatches.len()
    );
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

fn seg(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![0xFF, marker];
    v.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    v.extend_from_slice(payload);
    v
}

/// The index of the first event where two marker lists differ.
fn divergence(ours: &[Event], theirs: &[Event]) -> usize {
    let same = |a: &Event, b: &Event| a.name == b.name && a.bytes == b.bytes;
    (0..ours.len().min(theirs.len()))
        .find(|&k| !same(&ours[k], &theirs[k]))
        .unwrap_or(ours.len().min(theirs.len()))
}

#[test]
fn sos_rule_does_not_settle_a_lost_scan_with_component_255() {
    // Three scans of component 255 (T.81 B.2.2 allows it): each SOS header
    // holds `FF 00`, which ExifTool does not read as a marker. A walker
    // that lost the third scan must not be excused by the SOS rule.
    let sos = seg(0xDA, &[1, 0xFF, 0x00, 0, 63, 0]);
    let mut data = vec![0xFF, 0xD8];
    data.extend(seg(0xC0, &[8, 0, 8, 0, 8, 1, 0xFF, 0x11, 0]));
    for _ in 0..3 {
        data.extend_from_slice(&sos);
        data.extend_from_slice(&[0x12, 0x34]);
    }
    data.extend_from_slice(&[0xFF, 0xD9]);
    let meta = JpegMetadata::parse(&data).unwrap();
    let theirs = our_markers(&meta, &data);
    assert_eq!(theirs.iter().filter(|e| e.name == "SOS").count(), 3);
    let mut ours = theirs.clone();
    let third = ours.iter().rposition(|e| e.name == "SOS").unwrap();
    ours.remove(third);
    let k = divergence(&ours, &theirs);
    assert_eq!(settle_markers(&data, &meta, &ours, &theirs, k), None);
}

#[test]
fn sos_rule_does_not_settle_a_later_length_disagreement() {
    // An SOS whose length disagrees with its component count, then two DHT
    // segments; the second DHT's size differs. The divergence is at that
    // DHT, not in a run of identical events reaching back to the SOS.
    let mut data = vec![0xFF, 0xD8];
    data.extend(seg(0xDA, &[1, 1, 0x00, 0, 63, 0, 0, 0]));
    data.extend(seg(0xC4, &[0; 18]));
    data.extend(seg(0xC4, &[0; 31]));
    data.extend_from_slice(&[0xFF, 0xD9]);
    let meta = JpegMetadata::parse(&data).unwrap();
    let ours = our_markers(&meta, &data);
    let mut theirs = ours.clone();
    let last_dht = theirs.iter().rposition(|e| e.name == "DHT").unwrap();
    theirs[last_dht].bytes = Some(30);
    let k = divergence(&ours, &theirs);
    assert_eq!(k, last_dht);
    assert_eq!(settle_markers(&data, &meta, &ours, &theirs, k), None);
}

/// `SOI SOF0`, then each scan header with two entropy bytes, then `EOI`.
fn scans(headers: &[&[u8]]) -> Vec<u8> {
    let mut data = vec![0xFF, 0xD8];
    data.extend(seg(0xC0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]));
    for header in headers {
        data.extend(seg(0xDA, header));
        data.extend_from_slice(&[0x12, 0x34]);
    }
    data.extend_from_slice(&[0xFF, 0xD9]);
    data
}

/// Our markers for `data` stand in for ExifTool's; the same list with the
/// `nth` SOS event removed is a walker that lost that scan. Returns the
/// rule's verdict on the loss.
fn settle_lost_scan(data: &[u8], nth: usize) -> Option<(&'static str, String)> {
    let meta = JpegMetadata::parse(data).unwrap();
    let theirs = our_markers(&meta, data);
    let mut ours = theirs.clone();
    let lost = ours
        .iter()
        .enumerate()
        .filter(|(_, e)| e.name == "SOS")
        .nth(nth)
        .map(|(i, _)| i)
        .unwrap();
    ours.remove(lost);
    let k = divergence(&ours, &theirs);
    settle_markers(data, &meta, &ours, &theirs, k)
}

/// [`settle_lost_scan`] for the last SOS event.
fn settle_lost_last_scan(data: &[u8]) -> Option<(&'static str, String)> {
    let meta = JpegMetadata::parse(data).unwrap();
    let scans = our_markers(&meta, data)
        .iter()
        .filter(|e| e.name == "SOS")
        .count();
    settle_lost_scan(data, scans - 1)
}

const LEGAL_SOS: &[u8] = &[1, 1, 0x00, 0, 63, 0];

#[test]
fn sos_rule_does_not_settle_a_lost_scan_after_a_length_mismatch() {
    // Ls = 10 for Ns = 1 (two extra 0x00 bytes, no 0xFF): ExifTool never
    // reads Ls, so it lists the same markers as for a legal header.
    let data = scans(&[&[1, 1, 0x00, 0, 63, 0, 0, 0], LEGAL_SOS, LEGAL_SOS]);
    assert_eq!(settle_lost_last_scan(&data), None);
}

#[test]
fn sos_rule_does_not_settle_a_lost_scan_after_fill_and_a_stuffed_zero() {
    // Cs = 0xFF, Td/Ta = 0xFF, Ss = 0: the header holds `FF FF 00`, which
    // ExifTool reads as a fill byte and a stuffed zero, not a marker.
    let data = scans(&[&[1, 0xFF, 0xFF, 0, 63, 0], LEGAL_SOS, LEGAL_SOS]);
    assert_eq!(settle_lost_last_scan(&data), None);
}

#[test]
fn sos_rule_does_not_settle_a_segment_size_right_after_a_length_mismatch() {
    // The event before the divergence is an SOS whose Ls disagrees with Ns;
    // that alone must not excuse a DHT whose size differs.
    let mut data = vec![0xFF, 0xD8];
    data.extend(seg(0xDA, &[1, 1, 0x00, 0, 63, 0, 0, 0]));
    data.extend_from_slice(&[0x12, 0x34]);
    data.extend(seg(0xC4, &[0; 18]));
    data.extend_from_slice(&[0xFF, 0xD9]);
    let meta = JpegMetadata::parse(&data).unwrap();
    let ours = our_markers(&meta, &data);
    let mut theirs = ours.clone();
    let dht = theirs.iter().position(|e| e.name == "DHT").unwrap();
    assert_eq!(theirs[dht - 1].name, "SOS");
    theirs[dht].bytes = Some(17);
    let k = divergence(&ours, &theirs);
    assert_eq!(k, dht);
    assert_eq!(settle_markers(&data, &meta, &ours, &theirs, k), None);
}

#[test]
fn sos_rule_settles_a_marker_inside_the_scan_header() {
    // Cs = 0xFF, Td/Ta = 0xC4: ExifTool reads `FF C4` in the header as a
    // DHT marker; we skip the header by its length. The rule applies, also
    // after `0xFF` fill (`FF FF C4`).
    for header in [
        &[1, 0xFF, 0xC4, 0, 63, 0][..],
        &[1, 0xFF, 0xFF, 0xC4, 0, 63, 0, 0],
    ] {
        let data = scans(&[header, LEGAL_SOS]);
        assert!(
            settle_lost_last_scan(&data).is_some(),
            "{header:02x?} not settled"
        );
    }
    // An `0xFF` as the header's last byte reads as a marker when the first
    // non-fill byte after it is not 0x00.
    let mut data = scans(&[&[1, 1, 0x00, 0, 63, 0xFF], LEGAL_SOS]);
    let after = data.iter().position(|&b| b == 0x12).unwrap();
    assert!(settle_lost_last_scan(&data).is_some());
    data[after] = 0x00;
    assert_eq!(settle_lost_last_scan(&data), None);
}

#[test]
fn restarts_in_an_unparsable_scan_header_are_listed() {
    // Ns = 3 with one component's bytes does not parse as a scan header
    // (`Unknown`), but ExifTool still scans its bytes and the data after it
    // and lists the restart markers.
    let mut data = vec![0xFF, 0xD8];
    data.extend(seg(0xDA, &[3, 1, 0x00, 0, 63, 0]));
    data.extend_from_slice(&[0x12, 0xFF, 0xD0, 0x34, 0xFF, 0xD1, 0x56]);
    data.extend_from_slice(&[0xFF, 0xD9]);
    let meta = JpegMetadata::parse(&data).unwrap();
    assert!(matches!(meta.items()[1].kind, ItemKind::Unknown));
    let names: Vec<_> = our_markers(&meta, &data)
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(names, ["SOS", "RST0", "RST1", "EOI"]);
}

/// `SOI SOF0`, then `scans` legal scans with two entropy bytes each.
fn legal_scans(scans: usize) -> Vec<u8> {
    let mut data = vec![0xFF, 0xD8];
    data.extend(seg(0xC0, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]));
    for _ in 0..scans {
        data.extend(seg(0xDA, LEGAL_SOS));
        data.extend_from_slice(&[0x12, 0x34]);
    }
    data
}

#[test]
fn sos_rule_does_not_settle_a_lost_scan_after_a_restart_marker_in_a_header() {
    // Cs = 0xFF, Td/Ta = 0xD0: ExifTool lists the `FF D0` in the header as
    // RST0, and so do our restart events; nothing is read differently.
    let data = scans(&[&[1, 0xFF, 0xD0, 0, 63, 0], LEGAL_SOS, LEGAL_SOS]);
    let meta = JpegMetadata::parse(&data).unwrap();
    let names: Vec<_> = our_markers(&meta, &data)
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(names, ["SOF0", "SOS", "RST0", "SOS", "SOS", "EOI"]);
    assert_eq!(settle_lost_last_scan(&data), None);
    // The same across the header's end: its last byte 0xFF, the data
    // starting with 0xD3.
    let mut data = legal_scans(0);
    data.extend(seg(0xDA, &[1, 1, 0x00, 0, 63, 0xFF]));
    data.extend_from_slice(&[0xD3, 0x34]);
    for _ in 0..2 {
        data.extend(seg(0xDA, LEGAL_SOS));
        data.extend_from_slice(&[0x12, 0x34]);
    }
    data.extend_from_slice(&[0xFF, 0xD9]);
    assert_eq!(settle_lost_last_scan(&data), None);
}

#[test]
fn sos_rule_does_not_settle_a_lost_scan_after_a_bad_length_sos() {
    // `FF DA 00 01`: ExifTool never reads the length word and lists the same
    // markers as we do (the stray-bytes item after it lists its restarts).
    for stray in [&[0x12, 0x34][..], &[0x12, 0xFF, 0xD0, 0x34]] {
        let mut data = legal_scans(0);
        data.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x01]);
        data.extend_from_slice(stray);
        for _ in 0..2 {
            data.extend(seg(0xDA, LEGAL_SOS));
            data.extend_from_slice(&[0x12, 0x34]);
        }
        data.extend_from_slice(&[0xFF, 0xD9]);
        let meta = JpegMetadata::parse(&data).unwrap();
        assert!(matches!(
            meta.items()[2].kind,
            ItemKind::Unparsed(Unparsed::BadLength)
        ));
        assert_eq!(settle_lost_last_scan(&data), None, "{stray:02x?}");
    }
}

#[test]
fn sos_rule_does_not_settle_a_lost_scan_before_a_truncated_sos() {
    // Two legal scans, then an SOS cut off by the end of the input, without
    // a marker in its bytes: ExifTool lists three SOS, as we do.
    let mut data = legal_scans(2);
    data.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x20, 0x01, 0x01, 0x00]);
    let meta = JpegMetadata::parse(&data).unwrap();
    assert!(matches!(
        meta.items().last().unwrap().kind,
        ItemKind::Unparsed(Unparsed::TruncatedSegment)
    ));
    assert_eq!(settle_lost_scan(&data, 1), None);
    // Its restart markers are listed, as ExifTool lists them.
    let mut data = legal_scans(2);
    data.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x20, 0x01, 0xFF, 0xD5, 0x00]);
    let meta = JpegMetadata::parse(&data).unwrap();
    let names: Vec<_> = our_markers(&meta, &data)
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(names, ["SOF0", "SOS", "SOS", "SOS", "RST5"]);
    assert_eq!(settle_lost_scan(&data, 1), None);
    // A marker other than a restart marker in its bytes is read by ExifTool
    // and not by us: that truncated SOS is odd.
    let mut data = legal_scans(2);
    data.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x20, 0x01, 0xFF, 0xC4, 0x00]);
    assert!(settle_lost_scan(&data, 1).is_some());
}

#[test]
fn sos_rule_does_not_settle_a_lost_scan_when_the_next_marker_follows_a_header_ff() {
    // The first scan header ends in 0xFF and the next SOS marker follows it
    // directly: ExifTool reads `FF FF DA` as fill and SOS, and our walker
    // reads the same `FF DA` as the next item, so nothing is read
    // differently.
    let mut data = legal_scans(0);
    data.extend(seg(0xDA, &[1, 1, 0x00, 0, 63, 0xFF]));
    for _ in 0..2 {
        data.extend(seg(0xDA, LEGAL_SOS));
        data.extend_from_slice(&[0x12, 0x34]);
    }
    data.extend_from_slice(&[0xFF, 0xD9]);
    assert_eq!(settle_lost_last_scan(&data), None);
    assert_eq!(settle_lost_scan(&data, 1), None);
}

#[test]
fn sos_rule_settles_a_lost_scan_before_an_eoi_in_the_last_scan_header() {
    // Known gap, pinned so it stays visible: the last scan header holds
    // `FF D9`. ExifTool takes it as the EOI and lists the same markers we
    // do (our EOI is the real one after it), so the rule excuses a lost
    // earlier scan here. It cannot hide among agreeing files: ExifTool then
    // reports the bytes after its EOI as an Unknown trailer, which fails the
    // trailer comparison when nothing is lost.
    let data = scans(&[LEGAL_SOS, LEGAL_SOS, &[1, 0xFF, 0xD9, 0, 63, 0]]);
    let meta = JpegMetadata::parse(&data).unwrap();
    let names: Vec<_> = our_markers(&meta, &data)
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert_eq!(names, ["SOF0", "SOS", "SOS", "SOS", "EOI"]);
    assert!(settle_lost_scan(&data, 0).is_some());
}
