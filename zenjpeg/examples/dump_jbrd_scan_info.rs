//! Diagnostic: dump per-scan JBRD signals (reset_points + extra_zero_runs).
//!
//! Counts spurious reset_points caused by the bug in
//! decode_coefficients_with_jbrd_metadata. Used to validate the fix in
//! task #12 (jxl-encoder/zenjpeg) — agent commit memo lives in jxl-encoder
//! sibling `benchmarks/jpeg_reset_points_fix_2026-05-28.{tsv,meta}`.

use std::env;
use std::fs;
use std::process;

use zenjpeg::decoder::DecodeConfig;
use zenjpeg::encoder::Unstoppable;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: dump_jbrd_scan_info <jpg> [<jpg> ...]");
        process::exit(2);
    }
    println!("file\tnum_scans\ttotal_reset_points\ttotal_extra_zero_runs\trst_marker_count");
    for path in &args[1..] {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error reading {path}: {e}");
                continue;
            }
        };
        let rst_marker_count = count_rst_markers(&data);
        let cfg = DecodeConfig::new();
        match cfg.decode_coefficients_with_jbrd_metadata(&data, Unstoppable) {
            Ok((_coeffs, jbrd)) => {
                let total_rp: usize = jbrd.scans.iter().map(|s| s.reset_points.len()).sum();
                let total_ezr: usize = jbrd.scans.iter().map(|s| s.extra_zero_runs.len()).sum();
                println!(
                    "{path}\t{}\t{}\t{}\t{}",
                    jbrd.scans.len(),
                    total_rp,
                    total_ezr,
                    rst_marker_count,
                );
                for (i, s) in jbrd.scans.iter().enumerate() {
                    eprintln!(
                        "  scan[{i}] ss={} se={} ah={} al={} reset_points={} extra_zero_runs={}",
                        s.ss,
                        s.se,
                        s.ah,
                        s.al,
                        s.reset_points.len(),
                        s.extra_zero_runs.len(),
                    );
                }
            }
            Err(e) => {
                eprintln!("error decoding {path}: {e}");
            }
        }
    }
}

/// Count RST markers (0xFFD0..=0xFFD7) in the raw JPEG bytestream.
/// Coarse — counts any 0xFF Dn occurrence; close enough since JPEG escapes
/// other 0xFF Dn collisions to 0xFF 0x00 inside entropy data.
fn count_rst_markers(data: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF {
            let m = data[i + 1];
            if (0xD0..=0xD7).contains(&m) {
                count += 1;
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    count
}
