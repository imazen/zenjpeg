//! Encode peak-memory probe — one JPEG encode, report measured peak RSS (VmHWM).
//!
//! The ENCODE counterpart to `zentiff/examples/mem_probe.rs` (decode side). Used
//! by the heaptrack / VmHWM sweep to calibrate the encode peak-memory model
//! (`heuristics::estimate_encode`, surfaced as `estimate_encode_resources`)
//! against measured reality, *per effort level* (baseline / progressive /
//! auto-optimize), instead of the current structural guess
//! (`estimate_memory × MULT × trellis_factor`).
//!
//!   cargo build -p zenjpeg --release --example mem_probe_encode
//!   GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072 \
//!     ./target/release/examples/mem_probe_encode <rgb8.bin> <w> <h> <444|422|420|gray> <effort 0|1|2> <quality>
//!   heaptrack ./target/release/examples/mem_probe_encode ...   # allocator peak heap
//!
//! One encode per process — peak RSS is a per-process high-water mark, so the
//! input must come from a cheap file read (raw RGB8 bin), never an in-process
//! decode (whose own peak would pollute VmHWM above the encode peak).
//!
//! TSV row:
//!   w  h  pixels  subsamp  effort  quality  progressive
//!   out_bytes  pre_rss_kb  vmhwm_kb  marginal_kb

use enough::Unstoppable;
use std::hint::black_box;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// A `/proc/self/status` field in KiB (e.g. `VmRSS:`, `VmHWM:`).
fn status_kb(field: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(field))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: mem_probe_encode <rgb8.bin> <w> <h> <444|422|420|gray> <effort 0|1|2> <quality>");
        std::process::exit(2);
    }
    let path = &a[1];
    let w: u32 = a[2].parse().expect("w");
    let h: u32 = a[3].parse().expect("h");
    let subsamp = match a[4].as_str() {
        "444" | "422" | "420" | "gray" => a[4].clone(),
        other => panic!("subsamp must be 444|422|420|gray, got {other}"),
    };
    let effort: u8 = a[5].parse().expect("effort");
    let quality: f32 = a[6].parse().expect("quality");

    let data = std::fs::read(path).expect("read rgb8.bin");
    assert_eq!(
        data.len(),
        (w as usize) * (h as usize) * 3,
        "bin size {} != w*h*3 {}",
        data.len(),
        (w as usize) * (h as usize) * 3
    );

    // effort → config knobs. These are the codec-layer effort levels:
    //   0 = baseline (no progressive, no RD search)
    //   1 = progressive + optimized Huffman
    //   2 = auto_optimize (hybrid trellis λ=14.5 + progressive — max compression)
    let mut cfg = match subsamp.as_str() {
        "444" => EncoderConfig::ycbcr(quality, ChromaSubsampling::None),
        "422" => EncoderConfig::ycbcr(quality, ChromaSubsampling::HalfHorizontal),
        "420" => EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter),
        _ => EncoderConfig::grayscale(quality), // "gray"
    };
    let progressive;
    match effort {
        0 => {
            progressive = false;
        }
        1 => {
            cfg = cfg.progressive(true).optimize_huffman(true);
            progressive = true;
        }
        _ => {
            cfg = cfg.auto_optimize(true);
            progressive = true;
        }
    }

    // Estimate-only mode (`est` as a 7th arg): print what the CURRENT model
    // predicts for this cell (native base + ceiling), no encode — so we can
    // compare model vs measured without an encode polluting anything.
    if a.get(7).map(String::as_str) == Some("est") {
        let base = cfg.estimate_memory(w, h);
        let ceil = cfg.estimate_memory_ceiling(w, h);
        let pixels = (w as u64) * (h as u64);
        println!(
            "{w}\t{h}\t{pixels}\t{subsamp}\t{effort}\t{quality}\tEST\tbase_kb={}\tceil_kb={}\tbase_bpp={:.2}\tceil_bpp={:.2}",
            base / 1024,
            ceil / 1024,
            base as f64 / pixels as f64,
            ceil as f64 / pixels as f64
        );
        return;
    }

    // Baseline RSS: process + libs + the input `data` we hold. Marginal =
    // VmHWM − pre isolates the encode's own working set (what the model predicts).
    let pre = status_kb("VmRSS:");

    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&data, Unstoppable).expect("push");
    let out = enc.finish().expect("encode");

    // High-water mark immediately after finish — VmHWM is monotonic, so it
    // reflects the peak *during* the encode.
    let peak = status_kb("VmHWM:");

    let pixels = (w as u64) * (h as u64);
    println!(
        "{w}\t{h}\t{pixels}\t{subsamp}\t{effort}\t{quality}\t{}\t{}\t{pre}\t{peak}\t{}",
        u8::from(progressive),
        out.len(),
        peak.saturating_sub(pre)
    );
    black_box(&out);
}
