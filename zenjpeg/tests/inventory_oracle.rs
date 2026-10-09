//! Cross-check the structural inventory against ExifTool's segment map
//! (`exiftool -htmlDump0`, absolute offsets) on real files from
//! codec-corpus.
//!
//! Runs only when `INVENTORY_ORACLE_EXIFTOOL` names the exiftool binary; the
//! `just inventory-oracle` recipe sets it. The caller decides: the test never
//! probes for the tool itself. `INVENTORY_ORACLE_EXTRA_DIR` adds every JPEG
//! under a local directory to the set.
//!
//! Every top-level block ExifTool lists must line up with the inventory:
//!
//! - SOI, APPn, COM, DQT, DHT, DAC, DRI, SOFn and EOI: a top-level part with
//!   the same marker, offset and length.
//! - `[JPEG Image Data]`: ExifTool merges every scan (and anything between
//!   scans) from the first SOS to EOI into one block; the inventory splits
//!   it into SOS headers, scan data and the segments between. The block must
//!   start at an SOS part and end on a part boundary.
//! - Trailers (`GainMap trailer`, `Samsung trailer`, unknown data): their
//!   start and end must be part boundaries.
//!
//! One kind of difference is explained rather than matched, and only when
//! the inventory shows why: ExifTool starts `[JPEG Image Data]` at a
//! corrupted SOS marker (`FE DA`) that the decoder skips as stray bytes, so
//! the inventory has a `Malformed` gap there and the real SOS further on.

#![cfg(feature = "zencodec")]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
use zencodec::inventory::{Disposition, Inventory, PartKind, PartTag};
use zenjpeg::JpegDecoderConfig;

/// One block of ExifTool's HTML dump.
#[derive(Debug)]
struct Block {
    title: String,
    start: u64,
    len: u64,
}

/// Parse `exiftool -htmlDump0` output into its blocks: the hex column gives
/// each block's start (anchor `tN` at a byte position), the tooltip divs
/// give titles and sizes.
fn parse_html_dump(html: &str) -> Vec<Block> {
    let mut pres = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find("<pre") {
        let after = &rest[i..];
        let Some(open_end) = after.find('>') else {
            break;
        };
        let Some(close) = after.find("</pre>") else {
            break;
        };
        pres.push(&after[open_end + 1..close]);
        rest = &after[close + 6..];
    }
    assert!(pres.len() >= 2, "no address/hex columns in the dump");
    let mut starts: Vec<Option<u64>> = Vec::new();
    for (addr, hex) in pres[0].lines().zip(pres[1].lines()) {
        let Ok(base) = u64::from_str_radix(addr.trim(), 16) else {
            continue;
        };
        let b = hex.as_bytes();
        let (mut i, mut pos) = (0usize, 0u64);
        while i < b.len() {
            if hex[i..].starts_with("<a name=t") {
                let digits: String = hex[i + 9..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                let n: usize = digits.parse().unwrap();
                if starts.len() <= n {
                    starts.resize(n + 1, None);
                }
                starts[n].get_or_insert(base + pos);
                i += hex[i..].find('>').unwrap() + 1;
            } else if b[i] == b'<' {
                i += hex[i..].find('>').unwrap() + 1;
            } else if i + 1 < b.len()
                && b[i].is_ascii_hexdigit()
                && b[i + 1].is_ascii_hexdigit()
                && b.get(i + 2).is_none_or(|c| !c.is_ascii_hexdigit())
            {
                pos += 1;
                i += 2;
            } else {
                i += 1;
            }
        }
    }
    let mut blocks = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find("<div id=p") {
        let after = &rest[i + 9..];
        let n: usize = after
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .unwrap();
        let body_start = after.find('>').unwrap() + 1;
        let body_end = after.find("</div>").unwrap();
        let body = &after[body_start..body_end];
        rest = &after[body_end..];
        let text = body.replace("<br>", "|");
        let mut plain = String::new();
        let mut in_tag = false;
        for c in text.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => plain.push(c),
                _ => {}
            }
        }
        let title = plain.split('|').next().unwrap_or("").trim().to_string();
        let number_after = |key: &str| {
            plain.find(key).and_then(|k| {
                plain[k + key.len()..]
                    .trim_start()
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse::<u64>()
                    .ok()
            })
        };
        let len = if title.ends_with(" header") && title.starts_with("APP") {
            // A segment ExifTool splits into sub-blocks: marker + length
            // here, the payload size in the tooltip.
            number_after("Data size:").map(|n| n + 4)
        } else {
            plain
                .rfind('(')
                .and_then(|k| {
                    plain[k + 1..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse::<u64>()
                        .ok()
                })
                .or_else(|| number_after("Size:"))
        };
        if let (Some(Some(start)), Some(len)) = (starts.get(n), len) {
            blocks.push(Block {
                title,
                start: *start,
                len,
            });
        }
    }
    blocks.sort_by_key(|b| b.start);
    // Keep the top level: drop blocks inside an earlier block.
    let mut top: Vec<Block> = Vec::new();
    for b in blocks {
        if top.last().is_none_or(|t| b.start >= t.start + t.len) && b.len > 0 {
            top.push(b);
        }
    }
    top
}

/// The marker a block title names, for the unit types matched exactly.
fn marker_of(title: &str) -> Option<u8> {
    if title == "JPEG header" {
        return Some(0xD8);
    }
    if title == "JPEG EOI" {
        return Some(0xD9);
    }
    if let Some(rest) = title.strip_prefix("APP") {
        let n: u8 = rest
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
            .ok()?;
        return Some(0xE0 + n);
    }
    if title.starts_with("COM ") {
        return Some(0xFE);
    }
    let inner = title.strip_prefix("[JPEG ")?.strip_suffix(']')?;
    Some(match inner {
        "DQT" => 0xDB,
        "DHT" => 0xC4,
        "DRI" => 0xDD,
        "DAC" => 0xCC,
        "DNL" => 0xDC,
        _ => 0xC0 + inner.strip_prefix("SOF")?.parse::<u8>().ok()?,
    })
}

enum Verdict {
    Exact,
    Aligned,
    Explained(String),
    Mismatch(String),
}

fn judge(inv: &Inventory, b: &Block) -> Verdict {
    let top: Vec<_> = inv.parts().iter().filter(|p| p.parent.is_none()).collect();
    let (a, e) = (b.start, b.start + b.len);
    let starts_at = |x: u64| x == inv.input_len() || top.iter().any(|p| p.range.start == x);
    let ends_at = |x: u64| x == 0 || top.iter().any(|p| p.range.end == x);
    if b.title == "[JPEG Image Data]" {
        let sos = top
            .iter()
            .any(|p| p.range.start == a && p.tag == PartTag::Marker(0xDA));
        let stray_then_sos = top.iter().any(|p| {
            p.range.start == a && p.kind == PartKind::Gap && p.disposition == Disposition::Malformed
        }) && top
            .iter()
            .any(|p| p.range.start > a && p.range.start < e && p.tag == PartTag::Marker(0xDA));
        return if sos && ends_at(e) {
            Verdict::Aligned
        } else if stray_then_sos && ends_at(e) {
            Verdict::Explained(format!(
                "{} at {a}..{e}: ExifTool starts at bytes the decoder skips as stray (a corrupted \
                 SOS marker); the inventory has a Malformed gap there and the SOS inside the block",
                b.title
            ))
        } else {
            Verdict::Mismatch(format!(
                "{}: no SOS at {a} or no part boundary at {e}",
                b.title
            ))
        };
    }
    if let Some(m) = marker_of(&b.title) {
        return if top
            .iter()
            .any(|p| p.range == (a..e) && p.tag == PartTag::Marker(m))
        {
            Verdict::Exact
        } else {
            Verdict::Mismatch(format!(
                "{} at {a}..{e}: no FF{m:02X} part with that range",
                b.title
            ))
        };
    }
    if top.iter().any(|p| p.range == (a..e)) {
        Verdict::Exact
    } else if starts_at(a) && ends_at(e) {
        Verdict::Aligned
    } else {
        Verdict::Mismatch(format!("{} at {a}..{e}: not on part boundaries", b.title))
    }
}

fn oracle_files() -> Vec<PathBuf> {
    let corpus = codec_corpus::Corpus::new().expect("codec-corpus init failed");
    let mut files = Vec::new();
    let mut add = |dir: PathBuf| {
        let mut stack = vec![dir];
        while let Some(d) = stack.pop() {
            for e in fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| {
                    x.eq_ignore_ascii_case("jpg") || x.eq_ignore_ascii_case("jpeg")
                }) {
                    files.push(p);
                }
            }
        }
    };
    add(corpus.get("jpeg-conformance/valid").unwrap());
    add(corpus.get("ultrahdr-conformance/valid").unwrap());
    add(corpus.get("mozjpeg").unwrap());
    // Extra local files (for example real camera output that cannot be
    // committed), named by the caller.
    if let Ok(dir) = std::env::var("INVENTORY_ORACLE_EXTRA_DIR") {
        add(PathBuf::from(dir));
    }
    // The two Samsung SEF trailers in the corpus.
    let crash = corpus.get("jpeg-conformance/crash-repro").unwrap();
    for name in ["jd_130_progressive_jpeg.jpg", "jd_095_exif_zero.jpg"] {
        files.push(crash.join("jpeg-decoder").join(name));
    }
    files.sort();
    files
}

#[test]
fn inventory_matches_exiftool() {
    let Ok(exiftool) = std::env::var("INVENTORY_ORACLE_EXIFTOOL") else {
        eprintln!("INVENTORY_ORACLE_EXIFTOOL is unset; `just inventory-oracle` runs this check");
        return;
    };
    let files = oracle_files();
    assert!(
        files.len() >= 20,
        "need at least 20 oracle files, found {}",
        files.len()
    );
    let mut rows = Vec::new();
    let mut mismatches = Vec::new();
    let mut explained = Vec::new();
    for path in &files {
        let data = fs::read(path).unwrap();
        let out = Command::new(&exiftool)
            .arg("-htmlDump0")
            .arg(path)
            .output()
            .unwrap_or_else(|e| panic!("running {exiftool}: {e}"));
        assert!(
            out.status.success(),
            "exiftool failed on {}",
            path.display()
        );
        let blocks = parse_html_dump(&String::from_utf8_lossy(&out.stdout));
        assert!(
            !blocks.is_empty(),
            "no blocks parsed for {}",
            path.display()
        );
        let inv = JpegDecoderConfig::new()
            .job()
            .inventory(&data)
            .unwrap()
            .unwrap();
        inv.validate().unwrap();
        let (mut exact, mut aligned, mut expl) = (0, 0, 0);
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        for b in &blocks {
            match judge(&inv, b) {
                Verdict::Exact => exact += 1,
                Verdict::Aligned => aligned += 1,
                Verdict::Explained(why) => {
                    expl += 1;
                    explained.push(format!("{name}: {why}"));
                }
                Verdict::Mismatch(why) => mismatches.push(format!("{}: {why}", path.display())),
            }
        }
        rows.push(format!(
            "| {name} | {} | {} | {exact} | {aligned} | {expl} |",
            data.len(),
            blocks.len()
        ));
    }
    println!("| file | bytes | exiftool blocks | exact | aligned | explained |");
    println!("|---|---:|---:|---:|---:|---:|");
    for r in &rows {
        println!("{r}");
    }
    for e in &explained {
        println!("explained: {e}");
    }
    println!(
        "{} files, {} explained differences, {} mismatches",
        files.len(),
        explained.len(),
        mismatches.len()
    );
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
