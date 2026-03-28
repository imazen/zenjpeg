//! Comprehensive zensim regression testing: zenjpeg vs mozjpeg across the entire
//! imageflow codec corpus.
//!
//! For each loadable image in the imageflow test_inputs corpus, encodes with both
//! zenjpeg (MozjpegProgressive preset) and mozjpeg-rs, decodes both, and computes
//! zensim similarity of each decoded output against the uncompressed original.
//!
//! This catches:
//! - Quality regressions where zenjpeg produces worse output than mozjpeg
//! - Catastrophic encoding failures on unusual image dimensions/content
//! - Size ratio regressions (zenjpeg producing significantly larger files)
//!
//! Images are loaded from PNG (direct) and JPEG (decoded to RGB first via mozjpeg-sys).
//! CMYK, corrupt, and tiny images are skipped with warnings.
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

use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, Quality};

// ── Constants ───────────────────────────────────────────────────────────────

const CORPUS_DIR: &str = "/home/lilith/work/codec-eval/codec-corpus/imageflow/test_inputs";

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
    "corrupt.jpg",       // intentionally corrupt
    "cmyk_logo.jpg",     // CMYK colorspace, not RGB
    "rings2.png",        // palette-indexed PNG (64 colors), png crate returns Indexed
    "mountain_800.gif",  // GIF, not PNG/JPEG
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

fn encode_mozjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::ProgressiveSmallest)
        .quality(quality)
        .subsampling(mozjpeg_rs::Subsampling::S420)
        .encode_rgb(pixels, w, h)
        .expect("mozjpeg-rs encode failed")
}

fn encode_zenjpeg(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(quality), ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::MozjpegProgressive);
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(pixels, Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

fn decode_to_rgb(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let dec = Decoder::new().apply_icc(false);
    let img = dec.decode(jpeg, Unstoppable).expect("decode failed");
    let (w, h) = (img.width, img.height);
    (w, h, img.into_pixels_u8().unwrap())
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
    delta: f64,     // zen - moz (positive = zen better)
    size_ratio: f64, // zen / moz
    error: Option<String>,
}

fn as_rgb_pixels(rgb: &[u8]) -> &[[u8; 3]] {
    bytemuck::cast_slice(rgb)
}

fn process_image_quality(
    img: &LoadedImage,
    quality: u8,
) -> ImageResult {
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

    // Decode both with zenjpeg (constant decoder isolates encoder differences)
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
    let corpus = PathBuf::from(CORPUS_DIR);
    if !corpus.exists() {
        println!("Corpus not found at {CORPUS_DIR}, skipping");
        return;
    }

    // Collect and load images
    println!("Collecting images from {CORPUS_DIR}...");
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
                    skipped.push((name, format!("too large: {w}x{h} ({} MP)", w as u64 * h as u64 / 1_000_000)));
                    continue;
                }
                if pixels.len() != (w * h * 3) as usize {
                    skipped.push((name, format!("pixel count mismatch: {} vs {}x{}x3", pixels.len(), w, h)));
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
    println!(
        "=== Summary across {} images ===",
        images.len()
    );
    println!();
    println!(
        "{:>3} {:>7} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>5}/{:>4}/{:>5}",
        "Q", "moz_kb", "zen_kb", "sz_Δ%", "moz→orig", "zen→orig", "Δ(z-m)", "min_Δ", "win", "tie", "loss"
    );
    println!("{}", "-".repeat(95));

    for &q in &QUALITY_LEVELS {
        let qrows: Vec<&&ImageResult> = successes.iter().filter(|r| r.quality == q).collect();
        if qrows.is_empty() {
            continue;
        }
        let n = qrows.len() as f64;
        let moz_kb = qrows.iter().map(|r| r.moz_bytes as f64 / 1024.0).sum::<f64>() / n;
        let zen_kb = qrows.iter().map(|r| r.zen_bytes as f64 / 1024.0).sum::<f64>() / n;
        let moz_s = qrows.iter().map(|r| r.moz_vs_orig).sum::<f64>() / n;
        let zen_s = qrows.iter().map(|r| r.zen_vs_orig).sum::<f64>() / n;
        let delta = qrows.iter().map(|r| r.delta).sum::<f64>() / n;
        let min_delta = qrows
            .iter()
            .map(|r| r.delta)
            .fold(f64::INFINITY, f64::min);
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
            r.name, r.quality, r.width, r.height, r.moz_bytes, r.zen_bytes, r.size_ratio,
            r.moz_vs_orig, r.zen_vs_orig, r.delta,
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
        let q90_mean_delta =
            q90_plus.iter().map(|r| r.delta).sum::<f64>() / q90_plus.len() as f64;
        let q90_wins = q90_plus.iter().filter(|r| r.delta > 0.1).count();
        let q90_losses = q90_plus.iter().filter(|r| r.delta < -0.1).count();
        println!(
            "Q90+ subset: mean_delta={q90_mean_delta:+.3}, {q90_wins} wins / {q90_losses} losses"
        );
    }

    println!("\nAll assertions passed.");
}
