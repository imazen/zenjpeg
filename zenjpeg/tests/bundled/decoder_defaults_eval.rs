//! Evaluate all decoder configurations: quality (zensim vs original) AND speed.
//!
//! Tests every combination of IDCT method × chroma upsampling × dequant_bias
//! on mozjpeg-encoded and zenjpeg-encoded JPEGs. Shows which decoder config
//! best reconstructs the original, and how much each option costs in runtime.
//!
//! Decoder configurations tested:
//!   1. zen-default:  Jpegli IDCT (12-bit) + Triangle upsampling
//!   2. zen-compat:   Libjpeg IDCT (13-bit) + LibjpegCompat upsampling
//!   3. zen-bias:     Jpegli IDCT + Triangle + dequant_bias (f32 IDCT + Laplacian bias)
//!   4. zen-bias-cmp: Libjpeg IDCT + LibjpegCompat + dequant_bias
//!   5. mozjpeg-sys:  libjpeg-turbo FFI (reference)
//!
//! Encoders tested:
//!   - mozjpeg-rs (ProgressiveSmallest, 4:2:0)
//!   - zenjpeg auto_optimize (native Q, hybrid trellis, AQ, 4:2:0)
//!
//! Quality range: Q50, Q75, Q85, Q95
//!
//! Run:
//! ```bash
//! cargo test --release -p zenjpeg --test decoder_defaults_eval \
//!     --features "trellis decoder" -- --nocapture --ignored
//! ```

use enough::Unstoppable;
use std::path::{Path, PathBuf};
use std::time::Instant;
use zensim::{RgbSlice, Zensim, ZensimProfile};

use zenjpeg::decode::ChromaUpsampling;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

// ── Image loading ───────────────────────────────────────────────────────────

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for chunk in src.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, w, h))
}

fn collect_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    files.sort();
    files
}

// ── Decoder configs ─────────────────────────────────────────────────────────

struct DecoderMode {
    label: &'static str,
    desc: &'static str,
    decode: fn(&[u8]) -> Vec<u8>,
}

fn dec_default(jpeg: &[u8]) -> Vec<u8> {
    Decoder::new()
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .unwrap()
}

fn dec_compat(jpeg: &[u8]) -> Vec<u8> {
    Decoder::new()
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .unwrap()
}

fn dec_bias(jpeg: &[u8]) -> Vec<u8> {
    let img = Decoder::new()
        .dequant_bias(true)
        .decode(jpeg, Unstoppable)
        .expect("decode");
    // dequant_bias uses f32 output — convert to u8 for comparison
    f32_pixels_to_u8(img.into_pixels_f32().expect("f32 pixels from bias decode"))
}

fn dec_bias_compat(jpeg: &[u8]) -> Vec<u8> {
    let img = Decoder::new()
        .dequant_bias(true)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(jpeg, Unstoppable)
        .expect("decode");
    f32_pixels_to_u8(
        img.into_pixels_f32()
            .expect("f32 pixels from bias+compat decode"),
    )
}

fn f32_pixels_to_u8(f32_pixels: Vec<f32>) -> Vec<u8> {
    f32_pixels
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect()
}

fn dec_mozjpeg_sys(jpeg: &[u8]) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::mem;
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);
        jpeg_mem_src(&mut cinfo, jpeg.as_ptr(), jpeg.len() as _);
        jpeg_read_header(&mut cinfo, true as boolean);
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);
        let (w, h) = (cinfo.output_width, cinfo.output_height);
        let stride = w as usize * cinfo.output_components as usize;
        let mut out = vec![0u8; h as usize * stride];
        while cinfo.output_scanline < h {
            let off = cinfo.output_scanline as usize * stride;
            let mut p = out[off..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut p, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        out
    }
}

const DECODERS: [DecoderMode; 5] = [
    DecoderMode {
        label: "zen-default",
        desc: "Jpegli IDCT (12-bit) + Triangle",
        decode: dec_default,
    },
    DecoderMode {
        label: "zen-compat",
        desc: "Libjpeg IDCT (13-bit) + Triangle",
        decode: dec_compat,
    },
    DecoderMode {
        label: "zen-bias",
        desc: "dequant_bias + Jpegli IDCT (f32) + Triangle",
        decode: dec_bias,
    },
    DecoderMode {
        label: "zen-bias-cmp",
        desc: "dequant_bias + Libjpeg IDCT (f32) + Triangle",
        decode: dec_bias_compat,
    },
    DecoderMode {
        label: "mozjpeg-sys",
        desc: "libjpeg-turbo FFI | islow IDCT (13-bit) | fancy",
        decode: dec_mozjpeg_sys,
    },
];

// ── Encoder helpers ─────────────────────────────────────────────────────────

fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(q)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg encode")
}

fn encode_zen_auto(pixels: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).auto_optimize(true);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("enc");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn as_rgb(rgb: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(rgb)
}

fn zensim_vs_orig(z: &Zensim, orig: &[u8], decoded: &[u8], w: usize, h: usize) -> f64 {
    let a = RgbSlice::new(as_rgb(orig), w, h);
    let b = RgbSlice::new(as_rgb(decoded), w, h);
    z.compute(&a, &b).map(|r| r.score()).unwrap_or(-1.0)
}

/// Time N iterations of a decode function, return mean microseconds.
fn bench_decode(jpeg: &[u8], decode_fn: fn(&[u8]) -> Vec<u8>, iters: u32) -> f64 {
    // Warmup
    let _ = decode_fn(jpeg);
    let _ = decode_fn(jpeg);

    let start = Instant::now();
    for _ in 0..iters {
        let decoded = decode_fn(jpeg);
        std::hint::black_box(&decoded);
    }
    start.elapsed().as_micros() as f64 / iters as f64
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires CID22 corpus and decoder/trellis features"]
fn decoder_defaults_quality_and_speed() {
    let corpus = match codec_corpus::Corpus::new() {
        Ok(c) => c,
        Err(e) => {
            println!("corpus init failed: {e}");
            return;
        }
    };
    let dir = match corpus.get("CID22/CID22-512/training") {
        Ok(d) => d,
        Err(_) => {
            println!("CID22 not available");
            return;
        }
    };

    let paths = collect_pngs(&dir);
    let images: Vec<(Vec<u8>, u32, u32)> = paths
        .iter()
        .take(25)
        .filter_map(|p| load_png_rgb(p))
        .collect();
    let n = images.len();
    println!("Loaded {n} images (512x512)\n");

    let qualities: [u8; 4] = [50, 75, 85, 95];
    let encoders: [(&str, fn(&[u8], u32, u32, u8) -> Vec<u8>); 2] =
        [("mozjpeg", encode_mozjpeg), ("zen-auto", encode_zen_auto)];

    println!("  Decoders:");
    for d in &DECODERS {
        println!("    {:<14} {}", d.label, d.desc);
    }
    println!();

    let zensim = Zensim::new(ZensimProfile::codec_target());

    // Decode timing iterations per image
    let timing_iters = 10u32;

    for (enc_label, enc_fn) in &encoders {
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║  Encoder: {enc_label:<52}║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        for &q in &qualities {
            // Encode all images at this quality
            let jpegs: Vec<Vec<u8>> = images
                .iter()
                .map(|(px, w, h)| enc_fn(px, *w, *h, q))
                .collect();

            println!("\n  Q{q} (mean across {n} images, {timing_iters} decode iterations each):");
            println!(
                "  {:<14} {:>8} {:>6} {:>8} {:>6}",
                "decoder", "zensim", "Δ", "µs", "speed"
            );
            println!("  {}", "-".repeat(50));

            let mut base_score = 0.0f64;
            let mut base_us = 0.0f64;

            for (di, dec) in DECODERS.iter().enumerate() {
                let mut total_score = 0.0f64;
                let mut total_us = 0.0f64;

                for (i, jpeg) in jpegs.iter().enumerate() {
                    let (ref orig, w, h) = images[i];
                    let decoded = (dec.decode)(jpeg);
                    let score = zensim_vs_orig(&zensim, orig, &decoded, w as usize, h as usize);
                    total_score += score;

                    let us = bench_decode(jpeg, dec.decode, timing_iters);
                    total_us += us;
                }

                let mean_score = total_score / n as f64;
                let mean_us = total_us / n as f64;

                if di == 0 {
                    base_score = mean_score;
                    base_us = mean_us;
                    println!(
                        "  {:<14} {:>8.2} {:>6} {:>7.0}µs {:>6}",
                        dec.label, mean_score, "base", mean_us, "1.00x"
                    );
                } else {
                    let delta = mean_score - base_score;
                    let speed = mean_us / base_us;
                    println!(
                        "  {:<14} {:>8.2} {:>+5.2} {:>7.0}µs {:>5.2}x",
                        dec.label, mean_score, delta, mean_us, speed
                    );
                }
            }
        }
        println!();
    }

    // Summary: which decoder is best across all (encoder, quality) combos?
    println!("=== Best decoder per (encoder, quality) — zensim vs original ===\n");
    println!(
        "  {:<10} {:>4} {:>14} {:>8} {:>14} {:>8}",
        "encoder", "Q", "best_decoder", "zensim", "worst_decoder", "zensim"
    );
    println!("  {}", "-".repeat(62));

    for (enc_label, enc_fn) in &encoders {
        for &q in &qualities {
            let jpegs: Vec<Vec<u8>> = images
                .iter()
                .map(|(px, w, h)| enc_fn(px, *w, *h, q))
                .collect();

            let mut scores: Vec<(&str, f64)> = Vec::new();
            for dec in &DECODERS {
                let total: f64 = jpegs
                    .iter()
                    .enumerate()
                    .map(|(i, jpeg)| {
                        let (ref orig, w, h) = images[i];
                        let decoded = (dec.decode)(jpeg);
                        zensim_vs_orig(&zensim, orig, &decoded, w as usize, h as usize)
                    })
                    .sum();
                scores.push((dec.label, total / n as f64));
            }
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let best = scores[0];
            let worst = scores[scores.len() - 1];
            println!(
                "  {:<10} Q{q:<2} {:>14} {:>7.2} {:>14} {:>7.2}",
                enc_label, best.0, best.1, worst.0, worst.1
            );
        }
    }

    println!("\nDone.");
}
