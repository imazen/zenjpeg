//! Decode peak-memory probe — one decode per process, report measured peak RSS (VmHWM).
//!
//! The DECODE counterpart to `mem_probe_encode.rs`. Built to answer a specific
//! question from the #187 coefficient-centric unification: the strip path
//! retains `DecodedCoefficients` (`Vec<i16>`, ~2 bytes/sample) where the older
//! buffered path retained packed RGB (`Vec<u8>`, 3 bytes/px). For 4:4:4 that is
//! ~6 bytes/px vs 3 — so the unification could plausibly *raise* peak memory.
//! This measures it instead of reasoning about it.
//!
//!   cargo build -p zenjpeg --release --example mem_probe_decode
//!   ./target/release/examples/mem_probe_decode gen /tmp/x.jpg 4096 4096 444 prog
//!   ./target/release/examples/mem_probe_decode /tmp/x.jpg scanline
//!   heaptrack ./target/release/examples/mem_probe_decode /tmp/x.jpg scanline
//!
//! One decode per process — VmHWM is a per-process high-water mark, so the JPEG
//! must come from a cheap file read, never an in-process encode (whose own peak
//! would pollute VmHWM above the decode peak). That is why `gen` is a separate
//! mode you invoke as its own process.
//!
//! `scanline` consumes rows into a small reusable batch buffer and discards
//! them, so the probe's own footprint stays ~one batch and the reported marginal
//! is dominated by what the DECODER retains internally. `full` uses `decode()`,
//! which materializes the whole image, so its marginal includes that output.
//!
//! TSV row:
//!   mode  file_bytes  w  h  pixels  pre_rss_kb  vmhwm_kb  marginal_kb  bytes_per_px  checksum

use enough::Unstoppable;
use std::hint::black_box;
use zenjpeg::decode::Decoder;
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

/// Block-banded RGB with a high-frequency ramp — realistic coefficient spread,
/// not a gradient (gradients quantize to degenerate 0/±1 coefficients).
fn test_image(w: usize, h: usize) -> Vec<u8> {
    let mut d = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            d[i] = if (y / 8) % 2 == 0 { 255 } else { 0 };
            d[i + 1] = ((x * 3 + y * 7) % 200) as u8;
            d[i + 2] = ((x * 5 + y * 11) % 240) as u8;
        }
    }
    d
}

fn gen_jpeg(args: &[String]) {
    if args.len() < 7 {
        eprintln!("usage: mem_probe_decode gen <out.jpg> <w> <h> <444|422|420|440> <prog|base>");
        std::process::exit(2);
    }
    let out_path = &args[2];
    let w: u32 = args[3].parse().expect("w");
    let h: u32 = args[4].parse().expect("h");
    let sub = match args[5].as_str() {
        "444" => ChromaSubsampling::None,
        "422" => ChromaSubsampling::HalfHorizontal,
        "420" => ChromaSubsampling::Quarter,
        "440" => ChromaSubsampling::HalfVertical,
        other => panic!("subsamp must be 444|422|420|440, got {other}"),
    };
    let progressive = match args[6].as_str() {
        "prog" => true,
        "base" => false,
        other => panic!("mode must be prog|base, got {other}"),
    };

    let px = test_image(w as usize, h as usize);
    let mut enc = EncoderConfig::ycbcr(85.0, sub)
        .progressive(progressive)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(&px, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("encode");
    std::fs::write(out_path, &jpeg).expect("write jpeg");
    println!("wrote {out_path} ({} bytes) {w}x{h}", jpeg.len());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: mem_probe_decode <in.jpg> <scanline|full>");
        eprintln!("       mem_probe_decode gen <out.jpg> <w> <h> <444|422|420|440> <prog|base>");
        std::process::exit(2);
    }
    if a[1] == "gen" {
        gen_jpeg(&a);
        return;
    }

    let path = &a[1];
    let mode = a.get(2).map(String::as_str).unwrap_or("scanline");
    let data = std::fs::read(path).expect("read jpeg");
    let file_bytes = data.len();

    // Baseline RSS: process + libs + the compressed `data` we hold. Marginal =
    // VmHWM − pre isolates the decode's own working set.
    let pre = status_kb("VmRSS:");

    let (w, h, checksum) = match mode {
        "scanline" => {
            let mut r = Decoder::new()
                .auto_orient(false)
                .num_threads(1)
                .scanline_reader(&data)
                .expect("scanline_reader");
            let (w, h) = (r.width() as usize, r.height() as usize);
            let stride = w * 3;
            // Small reusable batch — the probe's own footprint stays ~16 rows,
            // so `marginal` reflects what the DECODER retains, not our output.
            const BATCH: usize = 16;
            let mut buf = vec![0u8; stride * BATCH];
            let mut sum = 0u64;
            let mut total = 0usize;
            while !r.is_finished() {
                let rows = BATCH.min(h - total);
                let out = imgref::ImgRefMut::new(&mut buf[..stride * rows], stride, rows);
                let got = r.read_rows_rgb8(out).expect("read_rows_rgb8");
                if got == 0 {
                    break;
                }
                for &b in &buf[..stride * got] {
                    sum = sum.wrapping_add(b as u64);
                }
                total += got;
            }
            assert_eq!(total, h, "scanline didn't read all rows");
            (w, h, sum)
        }
        "full" => {
            let img = Decoder::new()
                .auto_orient(false)
                .num_threads(1)
                .decode(&data, Unstoppable)
                .expect("decode");
            let (w, h) = (img.width() as usize, img.height() as usize);
            let px = img.into_pixels_u8().expect("u8");
            let sum = px.iter().fold(0u64, |s, &b| s.wrapping_add(b as u64));
            black_box(&px);
            (w, h, sum)
        }
        other => panic!("mode must be scanline|full, got {other}"),
    };

    // VmHWM is monotonic, so reading it after the decode reflects the peak
    // *during* the decode.
    let peak = status_kb("VmHWM:");
    let marginal = peak.saturating_sub(pre);
    let pixels = (w as u64) * (h as u64);
    println!(
        "{mode}\t{file_bytes}\t{w}\t{h}\t{pixels}\t{pre}\t{peak}\t{marginal}\t{:.2}\t{checksum}",
        (marginal * 1024) as f64 / pixels as f64
    );
}
