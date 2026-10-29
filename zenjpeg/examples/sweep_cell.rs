//! Encode ONE sweep-grammar cell for cross-codec RD harnesses.
//!
//! Takes a PNG, a sweep cell id (the canonical `<fam>_<coeff>_<scan>_<color>`
//! grammar from [`zenjpeg::encode::sweep::config_from_cell_id`] — e.g.
//! `jp3_t0_small_420`, `moz_tr14.75+dc_small_420`), and a quality; encodes,
//! then decodes the result with zenjpeg's OWN decoder (dogfooding the full
//! roundtrip, same discipline as zenavif's `save_png`) and writes the decoded
//! pixels as PNG for external scorers (`fast-ssim2-cli` / `butteraugli`).
//!
//! Prints one machine-readable line on stdout:
//! `bytes=<n> enc_ms=<f> dec_ms=<f>` — the timings are INTERNAL
//! (encode_bytes / decode call only, no PNG-load or PPM-write cost), so
//! cross-codec encode-ms ratios aren't poisoned by harness I/O.
//!
//! Usage:
//! ```bash
//! cargo run --release --example sweep_cell --features __expert -- \
//!   in.png out.jpg jp3_t0_small_420 85 [decoded.png]
//! ```
//!
//! Exit codes: 2 = bad args / bad cell id, 1 = encode or decode failure.

use std::path::PathBuf;
use std::time::Instant;

use rgb::{ComponentBytes, RGB8};
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::PixelLayout;
use zenjpeg::encode::sweep::config_from_cell_id;
use zenjpeg_bench_utils::{load_png, save_png};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("usage: sweep_cell <in.png> <out.jpg> <cell_id> <q> [decoded.png]");
        std::process::exit(2);
    }
    let in_png = PathBuf::from(&args[0]);
    let out_jpg = PathBuf::from(&args[1]);
    let cell_id = &args[2];
    let q: f32 = match args[3].parse() {
        Ok(v) => v,
        Err(_) => {
            eprintln!("bad quality {:?}", args[3]);
            std::process::exit(2);
        }
    };
    let dec_out = args.get(4).map(PathBuf::from);

    let img = match load_png(&in_png) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("PNG load failed for {}: {e:?}", in_png.display());
            std::process::exit(1);
        }
    };
    let cfg = match config_from_cell_id(cell_id, q) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("bad cell id {cell_id:?}: {e}");
            std::process::exit(2);
        }
    };

    let (w, h) = (img.width(), img.height());
    // Flatten to tightly-packed RGB bytes, stride-correct (load_png builds
    // contiguous buffers today, but don't assume it).
    let flat: Vec<u8> = if img.stride() == w {
        img.buf().as_bytes().to_vec()
    } else {
        let mut v = Vec::with_capacity(w * h * 3);
        for row in img.rows() {
            v.extend_from_slice(row.as_bytes());
        }
        v
    };

    let t0 = Instant::now();
    let jpeg = match cfg.encode_bytes(&flat, w as u32, h as u32, PixelLayout::Rgb8Srgb) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("encode failed ({cell_id} q{q}): {e}");
            std::process::exit(1);
        }
    };
    let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if let Err(e) = std::fs::write(&out_jpg, &jpeg) {
        eprintln!("write failed {}: {e}", out_jpg.display());
        std::process::exit(1);
    }

    // Roundtrip through OUR decoder — a stream we can't decode is a hard
    // failure at any quality, never a silent skip.
    let t1 = Instant::now();
    let decoded = match Decoder::new().decode(&jpeg, enough::Unstoppable) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("DECODE FAILED on own output ({cell_id} q{q}): {e}");
            std::process::exit(1);
        }
    };
    let dec_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let Some(pixels) = decoded.pixels_u8() else {
        eprintln!("decoder returned no u8 pixels ({cell_id} q{q})");
        std::process::exit(1);
    };
    if pixels.len() != w * h * 3 {
        eprintln!(
            "decoded size mismatch ({cell_id} q{q}): got {} bytes, want {}",
            pixels.len(),
            w * h * 3
        );
        std::process::exit(1);
    }
    if let Some(out) = dec_out {
        let px: Vec<RGB8> = pixels
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        let r = imgref::Img::new(px, w, h);
        if let Err(e) = save_png(&out, r.as_ref()) {
            eprintln!("PNG write failed {}: {e}", out.display());
            std::process::exit(1);
        }
    }

    println!("bytes={} enc_ms={enc_ms:.2} dec_ms={dec_ms:.2}", jpeg.len());
}
