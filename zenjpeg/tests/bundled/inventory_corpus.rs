//! Structural inventory (`DecodeJob::inventory`) over every JPEG in the
//! codec-corpus conformance sets: `jpeg-conformance` (valid, invalid,
//! non-conformant, crash-repro), `mozjpeg` and `ultrahdr-conformance`.
//!
//! - Where the zencodec decode succeeds and the stream ends with EOI,
//!   `zencodec_testkit::check_inventory` must pass (full coverage, appended
//!   junk never consumed, every truncation valid, at least one image-data
//!   part).
//! - Where the decode fails, the same coverage checks run without the
//!   image-data requirement: the walker may know the decode fails (and then
//!   claims no image data), or the failure may lie in entropy-coded data it
//!   does not decode.
//! - Where the stream has no EOI (a truncated file), bytes appended to it
//!   continue the cut-off scan or segment, and the decoder reads them as
//!   such; the appended-junk assertion is skipped for those files only.
//! - The walker never reports a container-level failure for a file the
//!   decoder accepts: a decodable file always has image data.

#![cfg(feature = "zencodec")]

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};

use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
use zencodec::inventory::{Disposition, Inventory};
use zenjpeg::JpegDecoderConfig;

fn corpus() -> codec_corpus::Corpus {
    codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)")
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            collect(&p, out);
        } else {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "jpg" || ext == "jpeg" {
                out.push(p);
            }
        }
    }
}

/// Every JPEG in the conformance sets this crate's decoder is tested on.
pub(crate) fn corpus_jpegs() -> Vec<PathBuf> {
    let corpus = corpus();
    let mut files = Vec::new();
    for set in ["jpeg-conformance", "mozjpeg", "ultrahdr-conformance"] {
        let dir = corpus
            .get(set)
            .unwrap_or_else(|e| panic!("corpus.get({set}): {e}"));
        collect(&dir, &mut files);
    }
    files
}

/// The prefix lengths `check_inventory` truncates to.
fn truncation_lengths(len: usize) -> Vec<usize> {
    let mut lens: Vec<usize> = vec![0, 1, 2, 3, 4, 8, 16];
    for (num, den) in [(1, 8), (1, 4), (3, 8), (1, 2), (5, 8), (3, 4), (7, 8)] {
        lens.push(len * num / den);
    }
    lens.push(len.saturating_sub(1));
    lens.retain(|&n| n < len);
    lens.sort_unstable();
    lens.dedup();
    lens
}

fn inventory_of(data: &[u8]) -> Result<Inventory, String> {
    let inv = JpegDecoderConfig::new()
        .job()
        .inventory(data)
        .map_err(|e| format!("inventory failed: {e}"))?
        .ok_or("inventory returned None")?;
    if inv.input_len() != data.len() as u64 {
        return Err("inventory length differs from the input".into());
    }
    inv.validate().map_err(|e| format!("invalid: {e}\n{inv}"))?;
    Ok(inv)
}

/// `check_inventory` without the image-data requirement; `junk_unconsumed`
/// is false for streams without EOI.
fn check_coverage(valid: &[u8], junk_unconsumed: bool) -> Result<(), String> {
    inventory_of(valid)?;
    let mut junked = valid.to_vec();
    junked.extend((0..37u8).map(|i| i.wrapping_mul(97) ^ 0x5A));
    let inv = inventory_of(&junked)?;
    let tail = valid.len() as u64;
    let mut has_child = vec![false; inv.parts().len()];
    for p in inv.parts() {
        if let Some(parent) = p.parent {
            has_child[parent.index()] = true;
        }
    }
    for (i, p) in inv.parts().iter().enumerate() {
        if junk_unconsumed && p.range.end > tail && !has_child[i] && p.disposition.is_consumed() {
            return Err(format!(
                "appended junk reported as {} at {:?}",
                p.disposition, p.range
            ));
        }
    }
    for n in truncation_lengths(valid.len()) {
        inventory_of(&valid[..n]).map_err(|e| format!("truncated to {n}: {e}"))?;
    }
    Ok(())
}

/// The zencodec decode's verdict: `Ok` or the error text.
fn decode_verdict(data: &[u8]) -> Result<(), String> {
    std::panic::catch_unwind(|| {
        JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(data), &[])
            .and_then(|d| d.decode())
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|_| Err("panic".into()))
}

/// Decode errors raised inside entropy-coded data, which the walker does
/// not decode.
const ENTROPY_LEVEL: &[&str] = &[
    "invalid Huffman table 0: invalid code",
    "DC Huffman category out of range",
    "could not resync to restart marker",
    "restart marker",
    "AC coefficient index",
    "arithmetic",
];

#[test]
fn inventory_covers_every_corpus_jpeg() {
    let files = corpus_jpegs();
    assert!(
        files.len() > 300,
        "corpus unexpectedly small: {}",
        files.len()
    );
    let (mut decoded, mut rejected, mut rejected_known) = (0usize, 0usize, 0usize);
    let mut unterminated = 0usize;
    let mut failures = Vec::new();
    for path in &files {
        let data = fs::read(path).unwrap();
        let name = path.display().to_string();
        let inv = match inventory_of(&data) {
            Ok(inv) => inv,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        let has_eoi = inv
            .parts()
            .iter()
            .any(|p| p.parent.is_none() && p.tag == zencodec::inventory::PartTag::Marker(0xD9));
        let has_image_data = inv
            .parts()
            .iter()
            .any(|p| p.disposition == Disposition::ImageData);
        if !has_eoi {
            unterminated += 1;
            if data.windows(2).any(|w| w == [0xFF, 0xD9]) {
                println!("no EOI reached, though FF D9 occurs: {name}");
                if name.contains("/valid/") {
                    failures.push(format!("{name}: a valid file without an EOI part"));
                }
            }
        }
        let verdict = decode_verdict(&data);
        if verdict.is_ok() {
            decoded += 1;
            let checked = if has_eoi {
                zencodec_testkit::check_inventory(JpegDecoderConfig::new(), &data)
                    .map_err(|e| e.to_string())
            } else if !has_image_data {
                Err("decodes, but the inventory has no image data".to_string())
            } else {
                check_coverage(&data, false)
            };
            if let Err(e) = checked {
                failures.push(format!("{name}: {e}"));
            }
        } else {
            rejected += 1;
            if let Err(e) = check_coverage(&data, has_eoi) {
                failures.push(format!("{name}: {e}"));
            }
            // Did the walker see the failure coming? Then nothing past the
            // header probe() reads is consumed (no image data at all).
            // Otherwise the failure must be one the walker cannot see: in
            // entropy-coded data, or past a scan coded with a restart
            // interval (where the decoder may resync over later parts).
            if !has_image_data {
                rejected_known += 1;
            } else {
                let why = verdict.err().unwrap_or_default();
                let uncertain = inv.parts().iter().any(|p| {
                    p.detail
                        .as_deref()
                        .is_some_and(|d| d.contains("if the decoder reaches it"))
                });
                let entropy = ENTROPY_LEVEL.iter().any(|e| why.contains(e));
                println!(
                    "rejected, not flagged ({}): {name}: {why}",
                    if entropy {
                        "entropy"
                    } else if uncertain {
                        "after a restart interval"
                    } else {
                        "UNCLASSIFIED"
                    }
                );
                if !entropy && !uncertain {
                    failures.push(format!(
                        "{name}: rejected ({why}) but the inventory has image data"
                    ));
                }
            }
        }
    }
    println!(
        "inventory corpus: {} files; {decoded} decode (check_inventory), {rejected} rejected \
         ({rejected_known} of them flagged by the walker without entropy decoding); \
         {unterminated} have no EOI (appended-junk check skipped)",
        files.len()
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
