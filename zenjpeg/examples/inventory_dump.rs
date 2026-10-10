//! Print the structural inventory (`DecodeJob::inventory`) of JPEG files.
//!
//! ```text
//! cargo run --release --features zencodec --example inventory_dump -- [--count] FILE...
//! ```
//!
//! `--sizes` prints the in-memory size of a part. `--count` prints only the part count and the bytes per disposition, which
//! is what a memory measurement under `/usr/bin/time -v` wants.

use zencodec::decode::{DecodeJob, DecoderConfig};
use zenjpeg::JpegDecoderConfig;

fn main() {
    let mut count_only = false;
    let mut files = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--count" {
            count_only = true;
        } else if arg == "--sizes" {
            println!(
                "size_of Part = {}, PartId = {}",
                core::mem::size_of::<zencodec::inventory::Part>(),
                core::mem::size_of::<zencodec::inventory::PartId>()
            );
        } else {
            files.push(arg);
        }
    }
    if files.is_empty() {
        eprintln!("usage: inventory_dump [--count] FILE...");
        std::process::exit(2);
    }
    let mut failed = false;
    for file in files {
        let data = match std::fs::read(&file) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{file}: {e}");
                failed = true;
                continue;
            }
        };
        let start = std::time::Instant::now();
        let result = JpegDecoderConfig::new().job().inventory(&data);
        let elapsed = start.elapsed().as_secs_f64();
        match result {
            Ok(Some(inv)) => {
                let valid = inv.validate().is_ok();
                println!(
                    "{file}: {} bytes, {} parts, validate {valid}, {elapsed:.3}s",
                    data.len(),
                    inv.parts().len()
                );
                if count_only {
                    for (name, bytes) in inv.bytes_by_disposition() {
                        println!("  {name}: {bytes}");
                    }
                } else {
                    print_level(&inv, None, 1);
                }
                failed |= !valid;
            }
            Ok(None) => println!("{file}: no inventory"),
            Err(e) => {
                println!("{file}: {} bytes, error {e}, {elapsed:.3}s", data.len());
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

fn print_level(
    inv: &zencodec::inventory::Inventory,
    parent: Option<zencodec::inventory::PartId>,
    depth: usize,
) {
    for id in inv.children(parent) {
        let Some(p) = inv.get(id) else { continue };
        let label = p.label.as_deref().unwrap_or("");
        let detail = p.detail.as_deref().unwrap_or("");
        println!(
            "{:indent$}{} {:?} {}..{} {} {label:?} {detail}",
            "",
            p.kind.name(),
            p.tag,
            p.range.start,
            p.range.end,
            p.disposition,
            indent = 2 * depth
        );
        print_level(inv, Some(id), depth + 1);
    }
}
