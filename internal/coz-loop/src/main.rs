//! Coz causal-profiler loop harness for zenjpeg.
//!
//! One `coz::progress!()` fires per completed encode or decode of a full image,
//! so coz throughput deltas are attributable to whole codec operations.
//!
//! Usage:
//!   coz-loop <encode|decode> <noise|flat|smooth> <444|422|420|440>
//!            <base|prog> <quality> <WxH> <iters>
//!
//! Example: coz-loop decode noise 420 base 85 512x512 200

use enough::Unstoppable;
use std::env;
use std::time::Instant;
use zenbench::prelude::*;
use zenjpeg::{
    decoder::{Decoder, PixelFormat},
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

/// Noise+patches generator (same shape as zenjpeg/examples/decode_profile.rs):
/// realistic AC distribution, non-flat chroma. Seed-mixed by `seed`.
fn noise_image(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    let mut rng: u32 = 0xDEADBEEF ^ seed;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng >> 16) & 0xFF) as u8;
            let patch_x = (x / 64) & 3;
            let patch_y = (y / 64) & 3;
            let base = ((patch_x * 64 + patch_y * 32) & 255) as u8;
            data[idx] = base.wrapping_add(noise >> 2);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            data[idx + 1] = base.wrapping_add(((rng >> 16) & 0x3F) as u8);
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            data[idx + 2] = (255 - base).wrapping_add(((rng >> 16) & 0x1F) as u8);
        }
    }
    data
}

/// Flat-chroma generator: chroma constant inside each 8x8 block, varying
/// across blocks. Covers DC-only decode paths (CLAUDE.md: required coverage).
fn flat_chroma_image(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    let mut rng: u32 = 0x9E3779B9 ^ seed;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            // Luma carries texture; chroma is flat per 8x8 via shared base.
            let blk = ((x / 8) + (y / 8) * 7) & 255;
            let luma = ((rng >> 16) & 0xFF) as u8;
            data[idx] = luma;
            data[idx + 1] = (blk * 3) as u8;
            data[idx + 2] = (255u16 - blk as u16) as u8;
        }
    }
    data
}

/// Smooth/photographic-ish: low-frequency patches plus mild noise — mid-range
/// AC activity between `noise` and `flat`.
fn smooth_image(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    let mut rng: u32 = 0x13579BDF ^ seed;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let n = ((rng >> 16) & 0x0F) as u8; // mild noise only
            let cx = (x / 32) & 7;
            let cy = (y / 32) & 7;
            let base = ((cx * 31 + cy * 17) & 255) as u8;
            data[idx] = base.wrapping_add(n);
            data[idx + 1] = base.wrapping_add(n >> 1).wrapping_add(40);
            data[idx + 2] = 255u8.wrapping_sub(base).wrapping_add(n >> 2);
        }
    }
    data
}

fn fold(bytes: &[u8]) -> u64 {
    // FNV-1a over the whole buffer; prints a stable content fingerprint.
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn parse_size(s: &str) -> (u32, u32) {
    let (w, h) = s.split_once('x').expect("size must be WxH");
    (w.parse().unwrap(), h.parse().unwrap())
}

fn parse_subs(s: &str) -> ChromaSubsampling {
    match s {
        "444" => ChromaSubsampling::None,
        "422" => ChromaSubsampling::HalfHorizontal,
        "420" => ChromaSubsampling::Quarter,
        "440" => ChromaSubsampling::HalfVertical,
        other => panic!("unknown subsampling {other}"),
    }
}

fn encode_once(
    data: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    subs: ChromaSubsampling,
    progressive: bool,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subs).progressive(progressive);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

fn gen_pixels(input: &str, width: u32, height: u32) -> Vec<u8> {
    match input {
        "noise" => noise_image(width, height, 1),
        "flat" => flat_chroma_image(width, height, 1),
        "smooth" => smooth_image(width, height, 1),
        other => panic!("unknown input {other}"),
    }
}

/// `coz-loop bench <decode|encode> <input> <subs> <base|prog> <quality> <W1xH1,W2xH2,...>`
/// zenbench suite over the listed sizes; one lane per size so old/new binary
/// runs produce directly comparable paired stats.
fn run_bench(mode: &str, input: &str, subs: ChromaSubsampling, prog: bool, quality: f32, sizes: &str) {
    let dims: Vec<(u32, u32)> = sizes.split(',').map(parse_size).collect();
    zenbench::run(|suite| {
        for &(w, h) in &dims {
            let pixels = gen_pixels(input, w, h);
            let group = format!("{mode}_{input}_{}_{}x{}", if prog { "prog" } else { "base" }, w, h);
            suite.group(group, |g| {
                g.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
                g.config()
                    .max_time(std::time::Duration::from_secs(4))
                    .warmup_time(std::time::Duration::from_millis(150))
                    .min_rounds(8)
                    .max_rounds(40);
                if mode == "decode" {
                    let jpeg = encode_once(&pixels, w, h, quality, subs, prog);
                    g.bench("zenjpeg", move |b| {
                        let dec = Decoder::new().output_format(PixelFormat::Rgb);
                        b.iter(|| dec.decode(&jpeg, Unstoppable).unwrap())
                    });
                } else {
                    g.bench("zenjpeg", move |b| {
                        b.iter(|| encode_once(&pixels, w, h, quality, subs, prog))
                    });
                }
            });
        }
    });
}

/// Resolve a case spec into JPEG bytes.
/// `gen:<input>:<subs>:<prog>:<q>:<WxH>` builds synthetic input and encodes it;
/// `file:<path>` uses a real JPEG as-is.
fn spec_jpeg(spec: &str) -> Vec<u8> {
    if let Some(rest) = spec.strip_prefix("gen:") {
        let p: Vec<&str> = rest.split(':').collect();
        let (w, h) = parse_size(p[4]);
        let pixels = gen_pixels(p[0], w, h);
        encode_once(
            &pixels,
            w,
            h,
            p[3].parse().unwrap(),
            parse_subs(p[1]),
            p[2] == "prog",
        )
    } else if let Some(path) = spec.strip_prefix("file:") {
        std::fs::read(path).unwrap()
    } else {
        panic!("spec must be gen:... or file:...");
    }
}

fn decode_rgb(jpeg: &[u8]) -> Option<Vec<u8>> {
    let dec = Decoder::new().output_format(PixelFormat::Rgb);
    match dec.decode(jpeg, Unstoppable) {
        Ok(img) => img.pixels_u8().map(|p| p.to_vec()),
        Err(e) => {
            eprintln!("decode error: {e:?}");
            None
        }
    }
}

/// `coz-loop file <jpeg> <iters>` — decode a real JPEG; print size+fold+time.
/// `coz-loop trunc <jpeg> <numbytes> <iters>` — same on a truncated prefix.
fn run_file(path: &str, iters: usize, trunc: Option<usize>) {
    let data = std::fs::read(path).unwrap();
    let jpeg = &data[..trunc.unwrap_or(data.len()).min(data.len())];
    let t0 = Instant::now();
    let mut out_hash = 0u64;
    let mut out_len = 0usize;
    for _ in 0..iters {
        match decode_rgb(jpeg) {
            Some(px) => {
                out_len = px.len();
                out_hash = fold(&px);
            }
            None => {
                out_len = 0;
                out_hash = 0;
            }
        }
        coz::progress!();
    }
    eprintln!(
        "file {} trunc={:?} x{}: {} bytes, fold={:016x}, {:.2?} total",
        path,
        trunc,
        iters,
        out_len,
        out_hash,
        t0.elapsed()
    );
}

/// `coz-loop time <case.txt> <out.bin>` — zenbench-pair worker.
/// case.txt: line 1 = spec (`gen:...` or `file:...`), line 2 = iterations.
/// Prints one `elapsed_ns=<n>` row per iteration on stdout; writes the last
/// decoded RGB buffer to out.bin for the files check.
fn run_timed(case_path: &str, out_path: &str) {
    let text = std::fs::read_to_string(case_path).unwrap();
    let mut lines = text.lines();
    let spec = lines.next().expect("case line1 = spec").trim();
    let iters: usize = lines
        .next()
        .expect("case line2 = iters")
        .trim()
        .parse()
        .unwrap();
    let jpeg = spec_jpeg(spec);
    let dec = Decoder::new().output_format(PixelFormat::Rgb);
    let mut last: Vec<u8> = Vec::new();
    // Emit exactly ROWS metric lines regardless of `iters` (the pair metric's
    // expected row count is fixed per plan, while per-case iteration budgets
    // vary). Each row is the mean ns/decode over `chunk` iterations.
    const ROWS: usize = 200;
    let chunk = iters.div_ceil(ROWS);
    for _ in 0..ROWS {
        let t = Instant::now();
        for _ in 0..chunk {
            let img = dec.decode(&jpeg, Unstoppable).unwrap();
            last = img.pixels_u8().expect("u8 output").to_vec();
        }
        let row = t.elapsed().as_nanos() as u64 / chunk as u64;
        println!("elapsed_ns={row}");
        coz::progress!();
    }
    std::fs::write(out_path, &last).unwrap();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() == 8 && args[1] == "bench" {
        let subs = parse_subs(&args[4]);
        let prog = match args[5].as_str() {
            "base" => false,
            "prog" => true,
            other => panic!("unknown mode {other}"),
        };
        let quality: f32 = args[6].parse().unwrap();
        run_bench(&args[2], &args[3], subs, prog, quality, &args[7]);
        return;
    }
    if args.len() == 4 && args[1] == "time" {
        run_timed(&args[2], &args[3]);
        return;
    }
    if args.len() == 4 && args[1] == "file" {
        run_file(&args[2], args[3].parse().unwrap(), None);
        return;
    }
    if args.len() == 5 && args[1] == "trunc" {
        run_file(&args[2], args[4].parse().unwrap(), Some(args[3].parse().unwrap()));
        return;
    }
    if args.len() != 8 {
        eprintln!(
            "usage: {} <encode|decode|bench> <noise|flat|smooth|op-mode> <444|422|420|440> <base|prog> <quality> <WxH[,...]> <iters>",
            args[0]
        );
        std::process::exit(2);
    }
    let mode = &args[1];
    let input = &args[2];
    let subs = parse_subs(&args[3]);
    let progressive = match args[4].as_str() {
        "base" => false,
        "prog" => true,
        other => panic!("unknown mode {other}"),
    };
    let quality: f32 = args[5].parse().unwrap();
    let (width, height) = parse_size(&args[6]);
    let iters: usize = args[7].parse().unwrap();

    coz::thread_init();

    let pixels = match input.as_str() {
        "noise" => noise_image(width, height, 1),
        "flat" => flat_chroma_image(width, height, 1),
        "smooth" => smooth_image(width, height, 1),
        other => panic!("unknown input {other}"),
    };

    let t0 = Instant::now();
    match mode.as_str() {
        "encode" => {
            let mut jpeg_len = 0usize;
            let mut jpeg_hash = 0u64;
            for i in 0..iters {
                let jpeg = encode_once(&pixels, width, height, quality, subs, progressive);
                jpeg_len = jpeg.len();
                if i % 16 == 0 || i == iters - 1 {
                    jpeg_hash = fold(&jpeg);
                }
                coz::progress!();
            }
            eprintln!(
                "encode {} {} {} q{} {}x{} x{}: {} bytes, fold={:016x}, {:.2?} total",
                input,
                args[3],
                args[4],
                quality,
                width,
                height,
                iters,
                jpeg_len,
                jpeg_hash,
                t0.elapsed()
            );
        }
        "decode" => {
            let jpeg = encode_once(&pixels, width, height, quality, subs, progressive);
            eprintln!(
                "encoded input: {} bytes, fold={:016x}",
                jpeg.len(),
                fold(&jpeg)
            );
            let decoder = Decoder::new().output_format(PixelFormat::Rgb);
            let mut out_len = 0usize;
            let mut out_hash = 0u64;
            for i in 0..iters {
                let img = decoder.decode(&jpeg, Unstoppable).unwrap();
                let px = img.pixels_u8().expect("u8 output");
                out_len = px.len();
                // Hash a subset of iterations: keeps output determinism
                // observable without dwarfing the decoder in profile samples.
                if i % 16 == 0 || i == iters - 1 {
                    out_hash = fold(px);
                }
                coz::progress!();
            }
            eprintln!(
                "decode {} {} {} q{} {}x{} x{}: {} bytes, fold={:016x}, {:.2?} total",
                input,
                args[3],
                args[4],
                quality,
                width,
                height,
                iters,
                out_len,
                out_hash,
                t0.elapsed()
            );
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}
