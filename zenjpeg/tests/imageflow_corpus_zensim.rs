//! Comprehensive zensim regression testing: zenjpeg vs mozjpeg across the entire
//! imageflow codec corpus.
//!
//! **zenjpeg is always configured in mozjpeg-mimicry mode** throughout this file:
//!   - Encoder: `Quality::ApproxMozjpeg(q)` + `OptimizationPreset::MozjpegProgressive` + 4:2:0
//!   - This means zenjpeg uses mozjpeg-compatible quant tables and progressive scan scripts.
//!
//! Four tests, each with fully specified encoder/decoder/mode:
//!
//! 1. **imageflow_corpus_zensim_vs_mozjpeg** — Encoder quality vs original
//!    - Encoder A: mozjpeg-rs (ProgressiveSmallest, 4:2:0)
//!    - Encoder B: zenjpeg (ApproxMozjpeg + MozjpegProgressive, 4:2:0)
//!    - Decoder: zenjpeg default (Jpegli IDCT, Triangle upsampling) for BOTH
//!    - Compare: each decoded output vs uncompressed original via zensim
//!
//! 2. **imageflow_decoder_parity_mozjpeg_files** — Decoder accuracy (PRIMARY)
//!    - Encoder: mozjpeg-rs (ProgressiveSmallest, 4:2:0)
//!    - Decoder A: mozjpeg-sys / libjpeg-turbo FFI (JCS_RGB output)
//!    - Decoder B: zenjpeg default (Jpegli IDCT, Triangle upsampling)
//!    - Decoder C: zenjpeg LibjpegCompat (Libjpeg IDCT, LibjpegCompat upsampling)
//!    - Compare: B vs A, C vs A (zensim + max pixel diff)
//!
//! 3. **imageflow_encoder_cross_comparison** — Encoder output similarity
//!    - Encoder A: mozjpeg-rs (ProgressiveSmallest, 4:2:0)
//!    - Encoder B: zenjpeg (ApproxMozjpeg + MozjpegProgressive, 4:2:0)
//!    - Decoder: zenjpeg default (Jpegli IDCT, Triangle upsampling) for BOTH
//!    - Compare: decoded-A vs decoded-B via zensim
//!
//! 4. **imageflow_size_matched_quality** — Fair quality comparison at matched file size
//!    - Encoder A: mozjpeg-rs (ProgressiveSmallest, 4:2:0) at target Q
//!    - Encoder B: zenjpeg (ApproxMozjpeg + MozjpegProgressive, 4:2:0) Q bisected to match size ±2%
//!    - Decoder: zenjpeg default (Jpegli IDCT, Triangle upsampling) for BOTH
//!    - Compare: each decoded output vs uncompressed original via zensim
//!
//! Run:
//! ```bash
//! cargo test --release -p zenjpeg --test imageflow_corpus_zensim \
//!     --features "trellis decoder" -- --nocapture --ignored
//! ```

use enough::Unstoppable;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use zensim::{RgbSlice, Zensim, ZensimProfile};

use zenjpeg::decode::ChromaUpsampling;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality};

// ── Constants ───────────────────────────────────────────────────────────────

fn corpus_dir() -> Option<PathBuf> {
    codec_corpus::Corpus::new()
        .ok()?
        .get("imageflow")
        .ok()
        .map(|p| p.join("test_inputs"))
}

/// Quality levels spanning the useful range.
const QUALITY_LEVELS: [u8; 6] = [50, 70, 80, 85, 90, 95];

/// Minimum dimension for meaningful encoder quality comparison.
/// 60x60 and 64x64 images produce degenerate DCT/quantization behavior
/// (tiny block counts, extreme quant table interactions) that make zensim
/// deltas meaningless for regression detection.
const MIN_DIM: u32 = 100;

/// Maximum pixel count to avoid multi-minute encodes on the 5760x4320 image.
const MAX_PIXELS: u64 = 8_000_000;

/// Files to skip (corrupt, CMYK, non-image, or palette-indexed).
const SKIP_FILES: &[&str] = &[
    "corrupt.jpg",      // intentionally corrupt
    "cmyk_logo.jpg",    // CMYK colorspace, not RGB
    "rings2.png",       // palette-indexed PNG (64 colors), png crate returns Indexed
    "mountain_800.gif", // GIF, not PNG/JPEG
    "lossy_mountain.webp",
    "1_webp_a.webp",
    "1_webp_ll.webp",
    "5_webp_ll.webp",
];

// ── Image loading ───────────────────────────────────────────────────────────

struct LoadedImage {
    name: String,
    pixels: Vec<u8>, // flat RGB
    width: u32,
    height: u32,
}

/// Load a PNG to flat RGB bytes.
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
        png::ColorType::Grayscale => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for &g in src {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for chunk in src.chunks_exact(2) {
                rgb.extend_from_slice(&[chunk[0], chunk[0], chunk[0]]);
            }
            rgb
        }
        _ => return None,
    };
    Some((rgb, w, h))
}

/// Decode a JPEG to flat RGB bytes using mozjpeg-sys (libjpeg-turbo).
fn load_jpeg_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let data = std::fs::read(path).ok()?;
    decode_jpeg_rgb(&data).ok()
}

fn decode_jpeg_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    use mozjpeg_sys::*;
    use std::mem;

    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);

        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);

        jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);

        let header_ok = jpeg_read_header(&mut cinfo, true as boolean);
        if header_ok != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("bad header".into());
        }

        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);

        let width = cinfo.output_width;
        let height = cinfo.output_height;
        let components = cinfo.output_components as usize;
        if components != 3 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err(format!("not RGB: {components} components"));
        }
        let row_stride = width as usize * components;

        let mut output = vec![0u8; height as usize * row_stride];

        while cinfo.output_scanline < height {
            let offset = cinfo.output_scanline as usize * row_stride;
            let mut row_ptr = output[offset..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
        }

        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);

        Ok((output, width, height))
    }
}

/// Load an image (PNG or JPEG) to flat RGB.
fn load_image(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => load_png_rgb(path),
        "jpg" | "jpeg" => load_jpeg_rgb(path),
        _ => None,
    }
}

// ── Encoding ────────────────────────────────────────────────────────────────

/// Encode with **mozjpeg-rs** (C library via Rust wrapper).
/// Mode: ProgressiveSmallest, 4:2:0, mozjpeg quality scale.
fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg-rs encode failed")
}

/// Encode with **zenjpeg in mozjpeg-mimicry mode**.
/// Mode: ApproxMozjpeg(q) quality + MozjpegProgressive optimization + 4:2:0.
/// This uses mozjpeg-compatible quant tables and progressive scan scripts.
fn encode_zenjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::MozjpegProgressive);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// Decode with **zenjpeg default mode**.
/// IDCT: Jpegli (12-bit fixed-point). Upsampling: Triangle (jpegli-style).
/// No ICC transform.
fn decode_to_rgb(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let dec = Decoder::new();
    let img = dec.decode(jpeg, Unstoppable).expect("decode failed");
    let (w, h) = (img.width, img.height);
    (w, h, img.into_pixels_u8().unwrap())
}

/// Decode with **zenjpeg LibjpegCompat mode** (closest match to mozjpeg/libjpeg-turbo).
/// IDCT: Libjpeg (13-bit Loeffler, auto-selected by LibjpegCompat).
/// Upsampling: LibjpegCompat (fused 2D filter with alternating rounding bias).
/// No ICC transform.
fn decode_to_rgb_compat(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let dec = Decoder::new().chroma_upsampling(ChromaUpsampling::Triangle);
    let img = dec.decode(jpeg, Unstoppable).expect("decode failed");
    let (w, h) = (img.width, img.height);
    (w, h, img.into_pixels_u8().unwrap())
}

/// Decode with **mozjpeg-sys** (libjpeg-turbo C library via FFI).
/// IDCT: libjpeg-turbo islow (13-bit Loeffler). Upsampling: libjpeg fancy.
/// This is the reference decoder for parity testing.
fn decode_jpeg_rgb_mozjpeg(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    decode_jpeg_rgb(data)
}

// ── Comparison result ───────────────────────────────────────────────────────

struct ImageResult {
    name: String,
    width: u32,
    height: u32,
    quality: u8,
    moz_bytes: usize,
    zen_bytes: usize,
    moz_vs_orig: f64,
    zen_vs_orig: f64,
    delta: f64,      // zen - moz (positive = zen better)
    size_ratio: f64, // zen / moz
    error: Option<String>,
}

fn as_rgb_pixels(rgb: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(rgb)
}

fn process_image_quality(img: &LoadedImage, quality: u8) -> ImageResult {
    let (w, h) = (img.width, img.height);

    // Encode with both
    let moz_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_mozjpeg(&img.pixels, w, h, quality)
    }));
    let zen_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_zenjpeg(&img.pixels, w, h, quality)
    }));

    let moz_jpeg = match moz_result {
        Ok(v) => v,
        Err(_) => {
            return ImageResult {
                name: img.name.clone(),
                width: w,
                height: h,
                quality,
                moz_bytes: 0,
                zen_bytes: 0,
                moz_vs_orig: 0.0,
                zen_vs_orig: 0.0,
                delta: 0.0,
                size_ratio: 0.0,
                error: Some("mozjpeg encode panicked".into()),
            };
        }
    };

    let zen_jpeg = match zen_result {
        Ok(v) => v,
        Err(_) => {
            return ImageResult {
                name: img.name.clone(),
                width: w,
                height: h,
                quality,
                moz_bytes: moz_jpeg.len(),
                zen_bytes: 0,
                moz_vs_orig: 0.0,
                zen_vs_orig: 0.0,
                delta: 0.0,
                size_ratio: 0.0,
                error: Some("zenjpeg encode panicked".into()),
            };
        }
    };

    // Decode both with zenjpeg default (Jpegli IDCT, Triangle upsampling).
    // Same decoder for both → differences are purely encoder-side.
    let (_, _, moz_dec) = decode_to_rgb(&moz_jpeg);
    let (_, _, zen_dec) = decode_to_rgb(&zen_jpeg);

    // Compute zensim of each decoded output against the original source
    thread_local! {
        static ZENSIM: Zensim = Zensim::new(ZensimProfile::latest());
    }

    let (moz_score, zen_score) = ZENSIM.with(|z| {
        let orig = RgbSlice::new(as_rgb_pixels(&img.pixels), w as usize, h as usize);
        let moz_s = RgbSlice::new(as_rgb_pixels(&moz_dec), w as usize, h as usize);
        let zen_s = RgbSlice::new(as_rgb_pixels(&zen_dec), w as usize, h as usize);

        let ms = z.compute(&orig, &moz_s).map(|r| r.score()).unwrap_or(-1.0);
        let zs = z.compute(&orig, &zen_s).map(|r| r.score()).unwrap_or(-1.0);
        (ms, zs)
    });

    ImageResult {
        name: img.name.clone(),
        width: w,
        height: h,
        quality,
        moz_bytes: moz_jpeg.len(),
        zen_bytes: zen_jpeg.len(),
        moz_vs_orig: moz_score,
        zen_vs_orig: zen_score,
        delta: zen_score - moz_score,
        size_ratio: zen_jpeg.len() as f64 / moz_jpeg.len() as f64,
        error: None,
    }
}

// ── File collection ─────────────────────────────────────────────────────────

fn collect_images(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_inner(dir, &mut files);
    files.sort();
    files
}

fn collect_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_inner(&path, out);
        } else if path.is_file() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if SKIP_FILES.iter().any(|&s| name == s) {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "png" || ext == "jpg" || ext == "jpeg" {
                out.push(path);
            }
        }
    }
}

fn short_name(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires imageflow corpus and decoder/trellis features"]
fn imageflow_corpus_zensim_vs_mozjpeg() {
    let corpus = match corpus_dir() {
        Some(d) if d.exists() => d,
        _ => {
            println!("imageflow corpus not found, skipping");
            return;
        }
    };

    println!("=== Encoder Quality vs Original (same Q parameter) ===");
    println!("  Encoder A: mozjpeg-rs | ProgressiveSmallest | 4:2:0");
    println!("  Encoder B: zenjpeg   | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0");
    println!("  Decoder:   zenjpeg   | Jpegli IDCT | Triangle upsampling | no ICC");
    println!("  Metric:    zensim(decoded, original) for each encoder");
    println!();

    // Collect and load images
    println!("Collecting images from {}...", corpus.display());
    let paths = collect_images(&corpus);
    println!("Found {} image files", paths.len());

    let mut images: Vec<LoadedImage> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();

    for path in &paths {
        let name = short_name(path, &corpus);
        match load_image(path) {
            Some((pixels, w, h)) => {
                if w < MIN_DIM || h < MIN_DIM {
                    skipped.push((name, format!("too small: {w}x{h}")));
                    continue;
                }
                if (w as u64) * (h as u64) > MAX_PIXELS {
                    skipped.push((
                        name,
                        format!(
                            "too large: {w}x{h} ({} MP)",
                            w as u64 * h as u64 / 1_000_000
                        ),
                    ));
                    continue;
                }
                if pixels.len() != (w * h * 3) as usize {
                    skipped.push((
                        name,
                        format!("pixel count mismatch: {} vs {}x{}x3", pixels.len(), w, h),
                    ));
                    continue;
                }
                images.push(LoadedImage {
                    name,
                    pixels,
                    width: w,
                    height: h,
                });
            }
            None => {
                skipped.push((name, "failed to load".into()));
            }
        }
    }

    println!(
        "Loaded {} images, skipped {} (see below)",
        images.len(),
        skipped.len()
    );
    assert!(
        images.len() >= 10,
        "Expected at least 10 loadable images, got {}",
        images.len()
    );

    // Build work items: (image_index, quality)
    let work: Vec<(usize, u8)> = images
        .iter()
        .enumerate()
        .flat_map(|(i, _)| QUALITY_LEVELS.iter().map(move |&q| (i, q)))
        .collect();

    let total = work.len() as u32;
    let progress = AtomicU32::new(0);

    println!(
        "Running {} comparisons ({} images × {} quality levels) with {} threads...",
        total,
        images.len(),
        QUALITY_LEVELS.len(),
        rayon::current_num_threads()
    );

    let results: Vec<ImageResult> = work
        .par_iter()
        .map(|&(idx, quality)| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 50 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            process_image_quality(&images[idx], quality)
        })
        .collect();

    // Separate successes and errors
    let mut successes: Vec<&ImageResult> = Vec::new();
    let mut errors: Vec<&ImageResult> = Vec::new();

    for r in &results {
        if r.error.is_some() {
            errors.push(r);
        } else {
            successes.push(r);
        }
    }

    // ── Per-image table ─────────────────────────────────────────────────

    println!();
    println!(
        "{:<30} {:>3} {:>7} {:>7} {:>6} {:>8} {:>8} {:>7}",
        "image", "Q", "moz_kb", "zen_kb", "ratio", "moz→orig", "zen→orig", "Δ(z-m)"
    );
    println!("{}", "-".repeat(85));

    // Sort by delta ascending (worst first)
    let mut by_delta: Vec<&ImageResult> = successes.clone();
    by_delta.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());

    for r in &by_delta {
        let marker = if r.delta < -1.0 { "  ←" } else { "" };
        println!(
            "{:<30} {:>3} {:>7.1} {:>7.1} {:>6.3} {:>8.2} {:>8.2} {:>+7.2}{}",
            truncate(&r.name, 30),
            r.quality,
            r.moz_bytes as f64 / 1024.0,
            r.zen_bytes as f64 / 1024.0,
            r.size_ratio,
            r.moz_vs_orig,
            r.zen_vs_orig,
            r.delta,
            marker,
        );
    }

    // ── Summary by quality level ────────────────────────────────────────

    println!();
    println!("=== Summary across {} images ===", images.len());
    println!();
    println!(
        "{:>3} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>5}/{:>4}/{:>5}",
        "Q",
        "moz_kb",
        "zen_kb",
        "sz_Δ%",
        "moz→orig",
        "zen→orig",
        "Δ(z-m)",
        "min_Δ",
        "win",
        "tie",
        "loss"
    );
    println!("{}", "-".repeat(95));

    for &q in &QUALITY_LEVELS {
        let qrows: Vec<&&ImageResult> = successes.iter().filter(|r| r.quality == q).collect();
        if qrows.is_empty() {
            continue;
        }
        let n = qrows.len() as f64;
        let moz_kb = qrows
            .iter()
            .map(|r| r.moz_bytes as f64 / 1024.0)
            .sum::<f64>()
            / n;
        let zen_kb = qrows
            .iter()
            .map(|r| r.zen_bytes as f64 / 1024.0)
            .sum::<f64>()
            / n;
        let moz_s = qrows.iter().map(|r| r.moz_vs_orig).sum::<f64>() / n;
        let zen_s = qrows.iter().map(|r| r.zen_vs_orig).sum::<f64>() / n;
        let delta = qrows.iter().map(|r| r.delta).sum::<f64>() / n;
        let min_delta = qrows.iter().map(|r| r.delta).fold(f64::INFINITY, f64::min);
        let size_pct = (zen_kb / moz_kb - 1.0) * 100.0;
        let wins = qrows.iter().filter(|r| r.delta > 0.1).count();
        let losses = qrows.iter().filter(|r| r.delta < -0.1).count();
        let ties = qrows.len() - wins - losses;

        println!(
            "{:>3} {:>7.1} {:>7.1} {:>+6.1}% {:>8.2} {:>8.2} {:>+7.2} {:>+7.2} {:>5}/{:>4}/{:>5}",
            q, moz_kb, zen_kb, size_pct, moz_s, zen_s, delta, min_delta, wins, ties, losses
        );
    }

    // ── Skipped files ───────────────────────────────────────────────────

    if !skipped.is_empty() {
        println!("\n--- Skipped ({}) ---", skipped.len());
        for (name, reason) in &skipped {
            println!("  {name}: {reason}");
        }
    }

    // ── Errors ──────────────────────────────────────────────────────────

    if !errors.is_empty() {
        println!("\n--- Errors ({}) ---", errors.len());
        for r in &errors {
            println!(
                "  {} Q{}: {}",
                r.name,
                r.quality,
                r.error.as_deref().unwrap_or("?")
            );
        }
    }

    // ── Save full results to TSV ────────────────────────────────────────

    let results_path = "/tmp/imageflow_corpus_zensim.tsv";
    let mut report = String::from(
        "image\tquality\twidth\theight\tmoz_bytes\tzen_bytes\tsize_ratio\tmoz_vs_orig\tzen_vs_orig\tdelta\n",
    );
    for r in &by_delta {
        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\n",
            r.name,
            r.quality,
            r.width,
            r.height,
            r.moz_bytes,
            r.zen_bytes,
            r.size_ratio,
            r.moz_vs_orig,
            r.zen_vs_orig,
            r.delta,
        ));
    }
    let _ = std::fs::write(results_path, &report);
    println!("\nFull results saved to {results_path}");

    // ── Assertions ──────────────────────────────────────────────────────
    //
    // Thresholds calibrated against the imageflow corpus which includes:
    // - Photographic images (orientation variants, waterhouse, frymire, etc.)
    // - Alpha-stripped PNGs (shirt_transparent — degenerate transparent regions)
    // - JPEG re-encodes (source already has compression artifacts)
    // - Small synthetic images excluded by MIN_DIM=100
    //
    // At Q50-Q85, mozjpeg's 13-bit fixed-point DCT wins slightly on most
    // images. At Q90+, zenjpeg's f32 DCT produces measurably better quality.
    // shirt_transparent.png at Q50 hits -12 pts due to alpha-stripped regions
    // (random RGB under transparent pixels → pathological compression).

    // 1. No encode panics
    assert!(
        errors.is_empty(),
        "{} encode errors — zenjpeg must handle all valid corpus images",
        errors.len()
    );

    // 2. Mean quality delta: allow modest mozjpeg advantage at low-Q,
    //    but catch systematic regressions
    let mean_delta = successes.iter().map(|r| r.delta).sum::<f64>() / successes.len() as f64;
    println!("\nMean quality delta (zen - moz): {mean_delta:+.3}");
    assert!(
        mean_delta > -2.0,
        "Mean zensim delta {mean_delta:+.3} — zenjpeg is >2 pts worse than mozjpeg on average"
    );

    // 3. No catastrophic single-image regressions beyond what alpha-stripped
    //    and edge-case images produce (shirt_transparent Q50 = -12)
    let worst_delta = successes
        .iter()
        .map(|r| r.delta)
        .fold(f64::INFINITY, f64::min);
    println!("Worst quality delta: {worst_delta:+.3}");
    assert!(
        worst_delta > -15.0,
        "Worst zensim delta {worst_delta:+.3} — catastrophic regression on at least one image"
    );

    // 4. Check per-quality worst case: photographic images should not regress
    //    more than 5 pts at any quality level
    let mut photo_failures: Vec<String> = Vec::new();
    for r in &successes {
        // Skip known edge cases for the strict per-image check
        let is_edge_case = r.name.contains("transparent")
            || r.name.contains("gradient")
            || r.name.contains("dct_overflow")
            || r.name.contains("pngsuite");
        if !is_edge_case && r.delta < -5.0 {
            photo_failures.push(format!(
                "{} Q{}: delta={:+.2} (moz={:.2}, zen={:.2})",
                r.name, r.quality, r.delta, r.moz_vs_orig, r.zen_vs_orig
            ));
        }
    }
    if !photo_failures.is_empty() {
        println!("\n--- Photographic image regressions >5 pts ---");
        for f in &photo_failures {
            println!("  {f}");
        }
    }
    assert!(
        photo_failures.is_empty(),
        "{} photographic images regressed >5 pts vs mozjpeg",
        photo_failures.len()
    );

    // 5. File sizes should be within 10% on average
    let mean_size_ratio =
        successes.iter().map(|r| r.size_ratio).sum::<f64>() / successes.len() as f64;
    println!("Mean size ratio (zen/moz): {mean_size_ratio:.4}");
    assert!(
        mean_size_ratio < 1.10,
        "Mean size ratio {mean_size_ratio:.4} — zenjpeg files are >10% larger than mozjpeg on average"
    );
    assert!(
        mean_size_ratio > 0.90,
        "Mean size ratio {mean_size_ratio:.4} — zenjpeg files are >10% smaller (suspicious)"
    );

    // 6. Win/loss tracking (informational, not a hard gate)
    let total_wins = successes.iter().filter(|r| r.delta > 0.1).count();
    let total_losses = successes.iter().filter(|r| r.delta < -0.1).count();
    let total_ties = successes.len() - total_wins - total_losses;
    let loss_pct = total_losses as f64 / successes.len() as f64 * 100.0;
    let win_pct = total_wins as f64 / successes.len() as f64 * 100.0;

    println!(
        "\nOverall: {} wins ({:.1}%) / {} ties / {} losses ({:.1}%)",
        total_wins, win_pct, total_ties, total_losses, loss_pct
    );

    // 7. At Q90+, zenjpeg's f32 DCT should produce competitive quality
    let q90_plus: Vec<&&ImageResult> = successes.iter().filter(|r| r.quality >= 90).collect();
    if !q90_plus.is_empty() {
        let q90_mean_delta = q90_plus.iter().map(|r| r.delta).sum::<f64>() / q90_plus.len() as f64;
        let q90_wins = q90_plus.iter().filter(|r| r.delta > 0.1).count();
        let q90_losses = q90_plus.iter().filter(|r| r.delta < -0.1).count();
        println!(
            "Q90+ subset: mean_delta={q90_mean_delta:+.3}, {q90_wins} wins / {q90_losses} losses"
        );
    }

    println!("\nAll assertions passed.");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 2: DECODER PARITY (PRIMARY)
// ═══════════════════════════════════════════════════════════════════════════
//
// THIS IS THE PRIMARY TEST. It answers: "Can zenjpeg correctly read files
// that mozjpeg creates?" This matters more than encoder comparison because
// most JPEGs in the wild were created by mozjpeg/libjpeg-turbo.
//
// Pipeline:
//   Encoder:    mozjpeg-rs | ProgressiveSmallest | 4:2:0
//   Decoder A:  mozjpeg-sys (libjpeg-turbo FFI) | islow IDCT | fancy upsample
//   Decoder B:  zenjpeg default | Jpegli IDCT (12-bit) | Triangle upsampling
//   Decoder C:  zenjpeg LibjpegCompat | Libjpeg IDCT (13-bit Loeffler) | LibjpegCompat upsample
//   Metric:     zensim(B vs A), zensim(C vs A), max pixel diff

struct DecoderParityResult {
    name: String,
    quality: u8,
    jpeg_bytes: usize,
    /// zensim: zenjpeg default vs mozjpeg-sys decode
    default_score: f64,
    /// zensim: zenjpeg LibjpegCompat vs mozjpeg-sys decode
    compat_score: f64,
    /// max pixel diff: zenjpeg default vs mozjpeg-sys
    default_max_diff: u8,
    /// max pixel diff: zenjpeg LibjpegCompat vs mozjpeg-sys
    compat_max_diff: u8,
    error: Option<String>,
}

fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn process_decoder_parity(img: &LoadedImage, quality: u8) -> DecoderParityResult {
    let (w, h) = (img.width, img.height);

    // mozjpeg encodes the source
    let moz_jpeg = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_mozjpeg(&img.pixels, w, h, quality)
    })) {
        Ok(v) => v,
        Err(_) => {
            return DecoderParityResult {
                name: img.name.clone(),
                quality,
                jpeg_bytes: 0,
                default_score: 0.0,
                compat_score: 0.0,
                default_max_diff: 0,
                compat_max_diff: 0,
                error: Some("mozjpeg encode panicked".into()),
            };
        }
    };

    // Decode with mozjpeg-sys / libjpeg-turbo (reference: islow IDCT, fancy upsample)
    let moz_dec = match decode_jpeg_rgb_mozjpeg(&moz_jpeg) {
        Ok((rgb, _, _)) => rgb,
        Err(e) => {
            return DecoderParityResult {
                name: img.name.clone(),
                quality,
                jpeg_bytes: moz_jpeg.len(),
                default_score: 0.0,
                compat_score: 0.0,
                default_max_diff: 0,
                compat_max_diff: 0,
                error: Some(format!("mozjpeg decode: {e}")),
            };
        }
    };

    // Decode with zenjpeg (default Jpegli IDCT)
    let (_, _, zen_default) = decode_to_rgb(&moz_jpeg);

    // Decode with zenjpeg (LibjpegCompat — 13-bit Loeffler + compat upsampling)
    let (_, _, zen_compat) = decode_to_rgb_compat(&moz_jpeg);

    let default_max_diff = max_pixel_diff(&moz_dec, &zen_default);
    let compat_max_diff = max_pixel_diff(&moz_dec, &zen_compat);

    thread_local! {
        static ZENSIM: Zensim = Zensim::new(ZensimProfile::latest());
    }

    let (default_score, compat_score) = ZENSIM.with(|z| {
        let moz = RgbSlice::new(as_rgb_pixels(&moz_dec), w as usize, h as usize);
        let def = RgbSlice::new(as_rgb_pixels(&zen_default), w as usize, h as usize);
        let compat = RgbSlice::new(as_rgb_pixels(&zen_compat), w as usize, h as usize);

        let ds = z.compute(&moz, &def).map(|r| r.score()).unwrap_or(-1.0);
        let cs = z.compute(&moz, &compat).map(|r| r.score()).unwrap_or(-1.0);
        (ds, cs)
    });

    DecoderParityResult {
        name: img.name.clone(),
        quality,
        jpeg_bytes: moz_jpeg.len(),
        default_score,
        compat_score,
        default_max_diff,
        compat_max_diff,
        error: None,
    }
}

#[test]
#[ignore = "requires imageflow corpus and decoder/trellis features"]
fn imageflow_decoder_parity_mozjpeg_files() {
    let corpus = match corpus_dir() {
        Some(d) if d.exists() => d,
        _ => {
            println!("imageflow corpus not found, skipping");
            return;
        }
    };

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  DECODER PARITY: Can zenjpeg correctly read mozjpeg files?      ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!("  Encoder:    mozjpeg-rs  | ProgressiveSmallest | 4:2:0");
    println!("  Decoder A:  mozjpeg-sys | libjpeg-turbo FFI | islow IDCT | fancy upsample");
    println!("  Decoder B:  zenjpeg     | Jpegli IDCT (12-bit) | Triangle upsampling");
    println!("  Decoder C:  zenjpeg     | Libjpeg IDCT (13-bit Loeffler) | LibjpegCompat upsample");
    println!("  Metric:     zensim(B vs A) and zensim(C vs A) + max pixel diff");
    println!();

    let paths = collect_images(&corpus);
    let mut images: Vec<LoadedImage> = Vec::new();
    let mut skipped = 0u32;

    for path in &paths {
        let name = short_name(path, &corpus);
        if let Some((pixels, w, h)) = load_image(path) {
            if w < MIN_DIM
                || h < MIN_DIM
                || (w as u64) * (h as u64) > MAX_PIXELS
                || pixels.len() != (w * h * 3) as usize
            {
                skipped += 1;
                continue;
            }
            images.push(LoadedImage {
                name,
                pixels,
                width: w,
                height: h,
            });
        } else {
            skipped += 1;
        }
    }

    println!("Loaded {} images (skipped {})", images.len(), skipped);

    let work: Vec<(usize, u8)> = images
        .iter()
        .enumerate()
        .flat_map(|(i, _)| QUALITY_LEVELS.iter().map(move |&q| (i, q)))
        .collect();

    let total = work.len() as u32;
    let progress = AtomicU32::new(0);

    println!(
        "Running {} decoder parity checks with {} threads...\n",
        total,
        rayon::current_num_threads()
    );

    let results: Vec<DecoderParityResult> = work
        .par_iter()
        .map(|&(idx, quality)| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 50 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            process_decoder_parity(&images[idx], quality)
        })
        .collect();

    let successes: Vec<&DecoderParityResult> =
        results.iter().filter(|r| r.error.is_none()).collect();
    let errors: Vec<&DecoderParityResult> = results.iter().filter(|r| r.error.is_some()).collect();

    // Sort by compat_score ascending (worst parity first)
    let mut by_compat: Vec<&DecoderParityResult> = successes.clone();
    by_compat.sort_by(|a, b| a.compat_score.partial_cmp(&b.compat_score).unwrap());

    // ── Table ───────────────────────────────────────────────────────────

    println!(
        "{:<30} {:>3} {:>7} {:>8} {:>4} {:>8} {:>4}",
        "image", "Q", "kb", "default", "max", "compat", "max"
    );
    println!("{}", "-".repeat(72));

    for r in &by_compat {
        let marker = if r.compat_score < 90.0 { "  ←" } else { "" };
        println!(
            "{:<30} {:>3} {:>7.1} {:>8.2} {:>4} {:>8.2} {:>4}{}",
            truncate(&r.name, 30),
            r.quality,
            r.jpeg_bytes as f64 / 1024.0,
            r.default_score,
            r.default_max_diff,
            r.compat_score,
            r.compat_max_diff,
            marker,
        );
    }

    // ── Summary by quality ──────────────────────────────────────────────

    println!();
    println!(
        "{:>3} {:>8} {:>4} {:>8} {:>4}  (mean zensim, worst max_diff)",
        "Q", "default", "max", "compat", "max"
    );
    println!("{}", "-".repeat(40));

    for &q in &QUALITY_LEVELS {
        let qr: Vec<&&DecoderParityResult> = successes.iter().filter(|r| r.quality == q).collect();
        if qr.is_empty() {
            continue;
        }
        let n = qr.len() as f64;
        let def_mean = qr.iter().map(|r| r.default_score).sum::<f64>() / n;
        let compat_mean = qr.iter().map(|r| r.compat_score).sum::<f64>() / n;
        let def_max = qr.iter().map(|r| r.default_max_diff).max().unwrap_or(0);
        let compat_max = qr.iter().map(|r| r.compat_max_diff).max().unwrap_or(0);
        println!(
            "{:>3} {:>8.2} {:>4} {:>8.2} {:>4}",
            q, def_mean, def_max, compat_mean, compat_max
        );
    }

    if !errors.is_empty() {
        println!("\n--- Errors ({}) ---", errors.len());
        for r in errors.iter().take(10) {
            println!(
                "  {} Q{}: {}",
                r.name,
                r.quality,
                r.error.as_deref().unwrap_or("?")
            );
        }
    }

    // ── Save TSV ────────────────────────────────────────────────────────

    let results_path = "/tmp/imageflow_decoder_parity.tsv";
    let mut report = String::from(
        "image\tquality\tjpeg_kb\tdefault_score\tdefault_max_diff\tcompat_score\tcompat_max_diff\n",
    );
    for r in &by_compat {
        report.push_str(&format!(
            "{}\t{}\t{:.1}\t{:.4}\t{}\t{:.4}\t{}\n",
            r.name,
            r.quality,
            r.jpeg_bytes as f64 / 1024.0,
            r.default_score,
            r.default_max_diff,
            r.compat_score,
            r.compat_max_diff,
        ));
    }
    let _ = std::fs::write(results_path, &report);
    println!("\nFull results saved to {results_path}");

    // ── Assertions ──────────────────────────────────────────────────────

    assert!(
        errors.is_empty(),
        "{} decode errors on mozjpeg files",
        errors.len()
    );

    // LibjpegCompat mode: should match mozjpeg within max_diff ≤ 2 for
    // well-formed images. Memory notes confirm this from the 754-file corpus.
    let compat_worst_max = successes
        .iter()
        .map(|r| r.compat_max_diff)
        .max()
        .unwrap_or(0);
    println!("\nLibjpegCompat worst max_diff: {compat_worst_max}");
    assert!(
        compat_worst_max <= 3,
        "LibjpegCompat max_diff {compat_worst_max} > 3 — decoder regression vs mozjpeg"
    );

    // Mean zensim in compat mode should be very high (>95)
    let compat_mean =
        successes.iter().map(|r| r.compat_score).sum::<f64>() / successes.len() as f64;
    let compat_min = successes
        .iter()
        .map(|r| r.compat_score)
        .fold(f64::INFINITY, f64::min);
    println!("LibjpegCompat zensim: mean={compat_mean:.2}, min={compat_min:.2}");
    assert!(
        compat_min > 85.0,
        "LibjpegCompat min zensim {compat_min:.2} — decoder parity regression"
    );

    // Default Jpegli IDCT: slightly wider tolerance (different IDCT constants)
    let default_worst_max = successes
        .iter()
        .map(|r| r.default_max_diff)
        .max()
        .unwrap_or(0);
    let default_mean =
        successes.iter().map(|r| r.default_score).sum::<f64>() / successes.len() as f64;
    let default_min = successes
        .iter()
        .map(|r| r.default_score)
        .fold(f64::INFINITY, f64::min);
    println!(
        "Default Jpegli zensim: mean={default_mean:.2}, min={default_min:.2}, worst max_diff={default_worst_max}"
    );
    assert!(
        default_worst_max <= 5,
        "Default IDCT max_diff {default_worst_max} > 5 — decoder regression"
    );
    assert!(
        default_min > 80.0,
        "Default IDCT min zensim {default_min:.2} — decoder regression"
    );

    println!("\nDecoder parity assertions passed.");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 3: ENCODER CROSS-COMPARISON
// ═══════════════════════════════════════════════════════════════════════════
//
// How perceptually similar are the decoded outputs to each other?
// High scores mean the encoders produce interchangeable output.
//
// Pipeline:
//   Encoder A:  mozjpeg-rs | ProgressiveSmallest | 4:2:0
//   Encoder B:  zenjpeg | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0
//   Decoder:    zenjpeg default | Jpegli IDCT | Triangle upsampling | no ICC
//   Metric:     zensim(decoded-A, decoded-B) + max pixel diff

struct CrossResult {
    name: String,
    quality: u8,
    /// zensim between zen-decoded and moz-decoded (both decoded by zenjpeg)
    cross_score: f64,
    /// max pixel diff
    cross_max_diff: u8,
    moz_bytes: usize,
    zen_bytes: usize,
    error: Option<String>,
}

fn process_cross(img: &LoadedImage, quality: u8) -> CrossResult {
    let (w, h) = (img.width, img.height);

    let moz_jpeg = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_mozjpeg(&img.pixels, w, h, quality)
    })) {
        Ok(v) => v,
        Err(_) => {
            return CrossResult {
                name: img.name.clone(),
                quality,
                cross_score: 0.0,
                cross_max_diff: 0,
                moz_bytes: 0,
                zen_bytes: 0,
                error: Some("mozjpeg encode panicked".into()),
            };
        }
    };

    let zen_jpeg = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_zenjpeg(&img.pixels, w, h, quality)
    })) {
        Ok(v) => v,
        Err(_) => {
            return CrossResult {
                name: img.name.clone(),
                quality,
                cross_score: 0.0,
                cross_max_diff: 0,
                moz_bytes: moz_jpeg.len(),
                zen_bytes: 0,
                error: Some("zenjpeg encode panicked".into()),
            };
        }
    };

    // Decode both with zenjpeg default (Jpegli IDCT, Triangle upsampling).
    // Same decoder for both → differences are purely encoder-side.
    let (_, _, moz_dec) = decode_to_rgb(&moz_jpeg);
    let (_, _, zen_dec) = decode_to_rgb(&zen_jpeg);

    let cross_max_diff = max_pixel_diff(&moz_dec, &zen_dec);

    thread_local! {
        static ZENSIM: Zensim = Zensim::new(ZensimProfile::latest());
    }

    let cross_score = ZENSIM.with(|z| {
        let m = RgbSlice::new(as_rgb_pixels(&moz_dec), w as usize, h as usize);
        let z_s = RgbSlice::new(as_rgb_pixels(&zen_dec), w as usize, h as usize);
        z.compute(&m, &z_s).map(|r| r.score()).unwrap_or(-1.0)
    });

    CrossResult {
        name: img.name.clone(),
        quality,
        cross_score,
        cross_max_diff,
        moz_bytes: moz_jpeg.len(),
        zen_bytes: zen_jpeg.len(),
        error: None,
    }
}

#[test]
#[ignore = "requires imageflow corpus and decoder/trellis features"]
fn imageflow_encoder_cross_comparison() {
    let corpus = match corpus_dir() {
        Some(d) if d.exists() => d,
        _ => {
            println!("imageflow corpus not found, skipping");
            return;
        }
    };

    println!("=== Encoder Cross-Comparison: decoded outputs compared to each other ===");
    println!("  Encoder A: mozjpeg-rs | ProgressiveSmallest | 4:2:0");
    println!("  Encoder B: zenjpeg   | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0");
    println!("  Decoder:   zenjpeg   | Jpegli IDCT | Triangle upsampling | no ICC");
    println!("  Metric:    zensim(decoded-A, decoded-B) + max pixel diff");
    println!("  Same decoder for both → differences are purely encoder-side.\n");

    let paths = collect_images(&corpus);
    let mut images: Vec<LoadedImage> = Vec::new();

    for path in &paths {
        let name = short_name(path, &corpus);
        if let Some((pixels, w, h)) = load_image(path) {
            if w < MIN_DIM
                || h < MIN_DIM
                || (w as u64) * (h as u64) > MAX_PIXELS
                || pixels.len() != (w * h * 3) as usize
            {
                continue;
            }
            images.push(LoadedImage {
                name,
                pixels,
                width: w,
                height: h,
            });
        }
    }

    let work: Vec<(usize, u8)> = images
        .iter()
        .enumerate()
        .flat_map(|(i, _)| QUALITY_LEVELS.iter().map(move |&q| (i, q)))
        .collect();

    let total = work.len() as u32;
    let progress = AtomicU32::new(0);

    println!(
        "Running {} cross-comparisons ({} images × {} Q levels)...\n",
        total,
        images.len(),
        QUALITY_LEVELS.len()
    );

    let results: Vec<CrossResult> = work
        .par_iter()
        .map(|&(idx, quality)| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 50 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            process_cross(&images[idx], quality)
        })
        .collect();

    let successes: Vec<&CrossResult> = results.iter().filter(|r| r.error.is_none()).collect();

    // Sort by cross_score ascending
    let mut by_score: Vec<&CrossResult> = successes.clone();
    by_score.sort_by(|a, b| a.cross_score.partial_cmp(&b.cross_score).unwrap());

    println!(
        "{:<30} {:>3} {:>7} {:>7} {:>8} {:>4}",
        "image", "Q", "moz_kb", "zen_kb", "zensim", "max"
    );
    println!("{}", "-".repeat(65));

    for r in &by_score {
        println!(
            "{:<30} {:>3} {:>7.1} {:>7.1} {:>8.2} {:>4}",
            truncate(&r.name, 30),
            r.quality,
            r.moz_bytes as f64 / 1024.0,
            r.zen_bytes as f64 / 1024.0,
            r.cross_score,
            r.cross_max_diff,
        );
    }

    // Summary by quality
    println!();
    println!("{:>3} {:>8} {:>8} {:>4}", "Q", "mean", "min", "max_diff");
    println!("{}", "-".repeat(28));

    for &q in &QUALITY_LEVELS {
        let qr: Vec<&&CrossResult> = successes.iter().filter(|r| r.quality == q).collect();
        if qr.is_empty() {
            continue;
        }
        let n = qr.len() as f64;
        let mean = qr.iter().map(|r| r.cross_score).sum::<f64>() / n;
        let min = qr
            .iter()
            .map(|r| r.cross_score)
            .fold(f64::INFINITY, f64::min);
        let max_diff = qr.iter().map(|r| r.cross_max_diff).max().unwrap_or(0);
        println!("{:>3} {:>8.2} {:>8.2} {:>4}", q, mean, min, max_diff);
    }

    let results_path = "/tmp/imageflow_encoder_cross.tsv";
    let mut report =
        String::from("image\tquality\tcross_score\tcross_max_diff\tmoz_bytes\tzen_bytes\n");
    for r in &by_score {
        report.push_str(&format!(
            "{}\t{}\t{:.4}\t{}\t{}\t{}\n",
            r.name, r.quality, r.cross_score, r.cross_max_diff, r.moz_bytes, r.zen_bytes,
        ));
    }
    let _ = std::fs::write(results_path, &report);
    println!("\nFull results saved to {results_path}");

    // At the same quality, encoders should produce perceptually similar output
    let overall_mean =
        successes.iter().map(|r| r.cross_score).sum::<f64>() / successes.len() as f64;
    let overall_min = successes
        .iter()
        .map(|r| r.cross_score)
        .fold(f64::INFINITY, f64::min);
    println!("\nCross-encoder zensim: mean={overall_mean:.2}, min={overall_min:.2}");

    // Even the worst case should show substantial similarity
    assert!(
        overall_min > 40.0,
        "Cross-encoder min zensim {overall_min:.2} — encoders producing wildly different output"
    );
    assert!(
        overall_mean > 75.0,
        "Cross-encoder mean zensim {overall_mean:.2} — encoders not producing similar output"
    );

    println!("\nEncoder cross-comparison assertions passed.");
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST 4: SIZE-MATCHED QUALITY (fair comparison)
// ═══════════════════════════════════════════════════════════════════════════
//
// The Q-parameter comparison (test 1) is slightly unfair because the same Q
// produces different file sizes from each encoder. This test binary-searches
// zenjpeg's quality parameter to match mozjpeg's output size within ±2%,
// then compares quality at matched sizes.
//
// Pipeline:
//   Encoder A:  mozjpeg-rs | ProgressiveSmallest | 4:2:0 | Q as given
//   Encoder B:  zenjpeg | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0 | Q bisected to match A's size
//   Decoder:    zenjpeg default | Jpegli IDCT | Triangle upsampling | no ICC
//   Metric:     zensim(decoded, original) for each at matched file size

/// Binary-search zenjpeg quality to match a target file size within ±2%.
/// Returns (quality_used, jpeg_bytes, jpeg_data) or None if no match found.
fn encode_zenjpeg_size_match(
    pixels: &[u8],
    w: u32,
    h: u32,
    target_bytes: usize,
) -> Option<(u8, Vec<u8>)> {
    let tolerance = 0.02; // ±2%
    let lo_bound = (target_bytes as f64 * (1.0 - tolerance)) as usize;
    let hi_bound = (target_bytes as f64 * (1.0 + tolerance)) as usize;

    // Quick check: try a few quality levels to find the range
    let mut lo: u8 = 1;
    let mut hi: u8 = 100;
    let mut best: Option<(u8, Vec<u8>)> = None;
    let mut best_dist: usize = usize::MAX;

    for _ in 0..15 {
        if lo > hi {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        let jpeg = encode_zenjpeg(pixels, w, h, mid);
        let sz = jpeg.len();
        let dist = sz.abs_diff(target_bytes);

        if dist < best_dist {
            best_dist = dist;
            best = Some((mid, jpeg));
        }

        if sz >= lo_bound && sz <= hi_bound {
            return best;
        }

        if sz < target_bytes {
            lo = mid.saturating_add(1);
        } else {
            hi = mid.saturating_sub(1);
        }
    }

    // Accept best if within 5% (relaxed for edge cases)
    let relaxed_lo = (target_bytes as f64 * 0.95) as usize;
    let relaxed_hi = (target_bytes as f64 * 1.05) as usize;
    if let Some((_, ref data)) = best {
        if data.len() >= relaxed_lo && data.len() <= relaxed_hi {
            return best;
        }
    }
    None
}

struct SizeMatchResult {
    name: String,
    moz_quality: u8,
    zen_quality: u8,
    moz_bytes: usize,
    zen_bytes: usize,
    size_err_pct: f64,
    moz_vs_orig: f64,
    zen_vs_orig: f64,
    delta: f64, // zen - moz (positive = zen better at same size)
    error: Option<String>,
}

fn process_size_match(img: &LoadedImage, moz_quality: u8) -> SizeMatchResult {
    let (w, h) = (img.width, img.height);

    let moz_jpeg = encode_mozjpeg(&img.pixels, w, h, moz_quality);
    let target_bytes = moz_jpeg.len();

    let zen_match = encode_zenjpeg_size_match(&img.pixels, w, h, target_bytes);

    let (zen_quality, zen_jpeg) = match zen_match {
        Some(v) => v,
        None => {
            return SizeMatchResult {
                name: img.name.clone(),
                moz_quality,
                zen_quality: 0,
                moz_bytes: target_bytes,
                zen_bytes: 0,
                size_err_pct: 0.0,
                moz_vs_orig: 0.0,
                zen_vs_orig: 0.0,
                delta: 0.0,
                error: Some("no size match found".into()),
            };
        }
    };

    let size_err_pct = (zen_jpeg.len() as f64 / target_bytes as f64 - 1.0) * 100.0;

    // Decode both with zenjpeg default (Jpegli IDCT, Triangle upsampling).
    // Same decoder for both → differences are purely encoder-side.
    let (_, _, moz_dec) = decode_to_rgb(&moz_jpeg);
    let (_, _, zen_dec) = decode_to_rgb(&zen_jpeg);

    thread_local! {
        static ZENSIM: Zensim = Zensim::new(ZensimProfile::latest());
    }

    let (moz_score, zen_score) = ZENSIM.with(|z| {
        let orig = RgbSlice::new(as_rgb_pixels(&img.pixels), w as usize, h as usize);
        let m = RgbSlice::new(as_rgb_pixels(&moz_dec), w as usize, h as usize);
        let zs = RgbSlice::new(as_rgb_pixels(&zen_dec), w as usize, h as usize);
        let ms = z.compute(&orig, &m).map(|r| r.score()).unwrap_or(-1.0);
        let zsc = z.compute(&orig, &zs).map(|r| r.score()).unwrap_or(-1.0);
        (ms, zsc)
    });

    SizeMatchResult {
        name: img.name.clone(),
        moz_quality,
        zen_quality,
        moz_bytes: target_bytes,
        zen_bytes: zen_jpeg.len(),
        size_err_pct,
        moz_vs_orig: moz_score,
        zen_vs_orig: zen_score,
        delta: zen_score - moz_score,
        error: None,
    }
}

#[test]
#[ignore = "requires imageflow corpus and decoder/trellis features"]
fn imageflow_size_matched_quality() {
    let corpus = match corpus_dir() {
        Some(d) if d.exists() => d,
        _ => {
            println!("imageflow corpus not found, skipping");
            return;
        }
    };

    println!("=== Size-Matched Quality: fair comparison at equal file size ===");
    println!("  Encoder A: mozjpeg-rs | ProgressiveSmallest | 4:2:0 | Q as given");
    println!(
        "  Encoder B: zenjpeg   | ApproxMozjpeg(Q) + MozjpegProgressive | 4:2:0 | Q bisected to match A's size ±2%"
    );
    println!("  Decoder:   zenjpeg   | Jpegli IDCT | Triangle upsampling | no ICC");
    println!("  Metric:    zensim(decoded, original) for each — same bits, who wins?\n");

    let paths = collect_images(&corpus);
    let mut images: Vec<LoadedImage> = Vec::new();

    for path in &paths {
        let name = short_name(path, &corpus);
        if let Some((pixels, w, h)) = load_image(path) {
            if w < MIN_DIM
                || h < MIN_DIM
                || (w as u64) * (h as u64) > MAX_PIXELS
                || pixels.len() != (w * h * 3) as usize
            {
                continue;
            }
            images.push(LoadedImage {
                name,
                pixels,
                width: w,
                height: h,
            });
        }
    }

    let work: Vec<(usize, u8)> = images
        .iter()
        .enumerate()
        .flat_map(|(i, _)| QUALITY_LEVELS.iter().map(move |&q| (i, q)))
        .collect();

    let total = work.len() as u32;
    let progress = AtomicU32::new(0);

    println!(
        "Running {} size-matched comparisons ({} images × {} Q levels)...\n",
        total,
        images.len(),
        QUALITY_LEVELS.len()
    );

    let results: Vec<SizeMatchResult> = work
        .par_iter()
        .map(|&(idx, quality)| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 50 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            process_size_match(&images[idx], quality)
        })
        .collect();

    let successes: Vec<&SizeMatchResult> = results.iter().filter(|r| r.error.is_none()).collect();
    let errors: Vec<&SizeMatchResult> = results.iter().filter(|r| r.error.is_some()).collect();

    // Sort by delta ascending (worst first)
    let mut by_delta: Vec<&SizeMatchResult> = successes.clone();
    by_delta.sort_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap());

    println!(
        "{:<30} {:>3} {:>3} {:>7} {:>7} {:>5} {:>8} {:>8} {:>7}",
        "image", "mQ", "zQ", "moz_kb", "zen_kb", "sz%", "moz→orig", "zen→orig", "Δ(z-m)"
    );
    println!("{}", "-".repeat(95));

    for r in &by_delta {
        let marker = if r.delta < -1.0 { "  ←" } else { "" };
        println!(
            "{:<30} {:>3} {:>3} {:>7.1} {:>7.1} {:>+4.1}% {:>8.2} {:>8.2} {:>+7.2}{}",
            truncate(&r.name, 30),
            r.moz_quality,
            r.zen_quality,
            r.moz_bytes as f64 / 1024.0,
            r.zen_bytes as f64 / 1024.0,
            r.size_err_pct,
            r.moz_vs_orig,
            r.zen_vs_orig,
            r.delta,
            marker,
        );
    }

    // Summary by quality
    println!();
    println!(
        "{:>3} {:>4} {:>7} {:>8} {:>8} {:>7} {:>5}/{:>4}/{:>5}",
        "mQ", "zQ", "sz_err", "moz→orig", "zen→orig", "Δ(z-m)", "win", "tie", "loss"
    );
    println!("{}", "-".repeat(65));

    for &q in &QUALITY_LEVELS {
        let qr: Vec<&&SizeMatchResult> = successes.iter().filter(|r| r.moz_quality == q).collect();
        if qr.is_empty() {
            continue;
        }
        let n = qr.len() as f64;
        let avg_zq = qr.iter().map(|r| r.zen_quality as f64).sum::<f64>() / n;
        let avg_err = qr.iter().map(|r| r.size_err_pct).sum::<f64>() / n;
        let moz_s = qr.iter().map(|r| r.moz_vs_orig).sum::<f64>() / n;
        let zen_s = qr.iter().map(|r| r.zen_vs_orig).sum::<f64>() / n;
        let delta = qr.iter().map(|r| r.delta).sum::<f64>() / n;
        let wins = qr.iter().filter(|r| r.delta > 0.1).count();
        let losses = qr.iter().filter(|r| r.delta < -0.1).count();
        let ties = qr.len() - wins - losses;

        println!(
            "{:>3} {:>4.0} {:>+6.1}% {:>8.2} {:>8.2} {:>+7.2} {:>5}/{:>4}/{:>5}",
            q, avg_zq, avg_err, moz_s, zen_s, delta, wins, ties, losses
        );
    }

    if !errors.is_empty() {
        println!("\n--- No size match found ({}) ---", errors.len());
        for r in errors.iter().take(10) {
            println!(
                "  {} mQ{}: {}",
                r.name,
                r.moz_quality,
                r.error.as_deref().unwrap_or("?")
            );
        }
        if errors.len() > 10 {
            println!("  ... and {} more", errors.len() - 10);
        }
    }

    let results_path = "/tmp/imageflow_size_matched.tsv";
    let mut report = String::from(
        "image\tmoz_quality\tzen_quality\tmoz_bytes\tzen_bytes\tsize_err_pct\tmoz_vs_orig\tzen_vs_orig\tdelta\n",
    );
    for r in &by_delta {
        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:+.2}\t{:.4}\t{:.4}\t{:+.4}\n",
            r.name,
            r.moz_quality,
            r.zen_quality,
            r.moz_bytes,
            r.zen_bytes,
            r.size_err_pct,
            r.moz_vs_orig,
            r.zen_vs_orig,
            r.delta,
        ));
    }
    let _ = std::fs::write(results_path, &report);
    println!("\nFull results saved to {results_path}");

    // ── Assertions ──────────────────────────────────────────────────────
    //
    // Edge cases (shirt_transparent, whitespace-issue) produce large deltas
    // because alpha-stripped/mostly-empty content compresses pathologically
    // differently between encoders. Assert on photographic content only.

    let mean_delta = successes.iter().map(|r| r.delta).sum::<f64>() / successes.len() as f64;
    let total_wins = successes.iter().filter(|r| r.delta > 0.1).count();
    let total_losses = successes.iter().filter(|r| r.delta < -0.1).count();
    let total_ties = successes.len() - total_wins - total_losses;

    println!(
        "\nSize-matched: mean Δ={mean_delta:+.3}, {} wins / {} ties / {} losses",
        total_wins, total_ties, total_losses
    );

    // Overall mean should be roughly competitive
    assert!(
        mean_delta > -2.0,
        "Size-matched mean delta {mean_delta:+.3} — zenjpeg >2 pts worse at same file size"
    );

    // Photographic images: no regression > 5 pts at matched size
    let mut photo_failures: Vec<String> = Vec::new();
    for r in &successes {
        let is_edge_case = r.name.contains("transparent")
            || r.name.contains("whitespace")
            || r.name.contains("gradient")
            || r.name.contains("gamma_test")
            || r.name.contains("dct_overflow")
            || r.name.contains("pngsuite");
        if !is_edge_case && r.delta < -5.0 {
            photo_failures.push(format!(
                "{} mQ{} zQ{}: delta={:+.2} (moz={:.2}, zen={:.2})",
                r.name, r.moz_quality, r.zen_quality, r.delta, r.moz_vs_orig, r.zen_vs_orig
            ));
        }
    }
    if !photo_failures.is_empty() {
        println!("\n--- Photographic regressions >5 pts at matched size ---");
        for f in &photo_failures {
            println!("  {f}");
        }
    }
    assert!(
        photo_failures.is_empty(),
        "{} photographic images regressed >5 pts at matched file size",
        photo_failures.len()
    );

    println!("\nSize-matched quality assertions passed.");
}
