//! Phase 0 (jxl-encoder lossy-JPEG R&D): recompress each input JPEG to a
//! target zensim-A and write the resulting JPEG, so an external step can
//! transcode it losslessly to JXL and compare sizes at *matched pixels*
//! (i.e. matched quality). This isolates the JXL-entropy-coder advantage
//! over zenjpeg-recompress's JPEG output.
//!
//! Usage:
//!   jxl_phase0_recompress <out_dir> <target_zensim_a> <input.jpg>...
//!
//! Emits a TSV (one row per input) to stdout:
//!   path  target  strategy  projected_q  in_bytes  out_bytes  ratio

use std::fs;
use std::path::Path;

use zenjpeg::recompress::{Confidence, RecompressOptions, RecompressResult, recompress};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <out_dir> <target_zensim_a> <input.jpg>...",
            args[0]
        );
        std::process::exit(2);
    }
    let out_dir = &args[1];
    let target: f32 = args[2].parse().expect("target must be a float 0..100");
    fs::create_dir_all(out_dir).expect("create out_dir");

    println!("path\ttarget\tstrategy\tprojected_q\tin_bytes\tout_bytes\tratio");
    for input in &args[3..] {
        let bytes = match fs::read(input) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {input}: read error {e}");
                continue;
            }
        };
        let in_len = bytes.len();
        // P50 = no internal shift; OneShot = the default product path.
        let opts = RecompressOptions::new(target).with_confidence(Confidence::P50);
        let (out_bytes, strategy, projq) = match recompress(&bytes, &opts) {
            Ok(RecompressResult::Recompressed {
                bytes,
                strategy,
                projected_zensim_a,
                ..
            }) => (bytes, format!("{strategy:?}"), projected_zensim_a),
            Ok(RecompressResult::LosslessOnly { bytes, reason }) => {
                (bytes, format!("LosslessOnly({reason:?})"), f32::NAN)
            }
            Ok(RecompressResult::NoOp { reason }) => {
                // No work — the source itself is the "output".
                (bytes.clone(), format!("NoOp({reason:?})"), f32::NAN)
            }
            Ok(other) => {
                eprintln!("skip {input}: unexpected result variant {other:?}");
                continue;
            }
            Err(e) => {
                eprintln!("skip {input}: recompress error {e:?}");
                continue;
            }
        };
        let stem = Path::new(input)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("out");
        let out_path = format!("{out_dir}/{stem}.jpg");
        if let Err(e) = fs::write(&out_path, &out_bytes) {
            eprintln!("skip {input}: write error {e}");
            continue;
        }
        let ratio = out_bytes.len() as f64 / in_len as f64;
        println!(
            "{}\t{}\t{}\t{:.2}\t{}\t{}\t{:.4}",
            out_path,
            target,
            strategy,
            projq,
            in_len,
            out_bytes.len(),
            ratio
        );
    }
}
