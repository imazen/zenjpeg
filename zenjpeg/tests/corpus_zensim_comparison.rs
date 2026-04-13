//! Compare zenjpeg decoder output against mozjpeg using zensim (perceptual quality).
//!
//! For every JPEG in the corpus-builder directory, decode with both decoders,
//! normalize to sRGB via ICC profile transform, and compute zensim similarity.
//!
//! This catches perceptual regressions that pixel-level max_diff misses
//! (e.g., systematic color shifts in wide-gamut images).
//!
//! Run: cargo test --release -p zenjpeg --test corpus_zensim_comparison --features "decoder,cms" -- --nocapture --ignored

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use zenjpeg::color::icc::{TargetColorSpace, apply_icc_transform, extract_icc_profile};
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn corpus_dir() -> PathBuf {
    zenjpeg_bench_utils::corpus_builder_dir()
}

const SKIP_DIRS: &[&str] = &["repro-images", "cc-index"];
const MAX_FILE_SIZE: u64 = 50 * 1024 * 1024;
/// Minimum image dimension for zensim (requires 8x8).
const MIN_DIM: u32 = 8;

// ── Decoders ─────────────────────────────────────────────────────────────

/// Decode with zenjpeg (no EXIF rotation, no internal ICC transform).
/// ICC transform is handled separately by normalize_to_srgb() to match mozjpeg path.
fn decode_zenjpeg(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;

    let decoder = Decoder::new().auto_orient(false);
    let decoded = decoder
        .decode(data, Unstoppable)
        .map_err(|e| format!("{e}"))?;
    let pixels = decoded.pixels_u8().ok_or("no pixel data")?.to_vec();
    Ok((decoded.width, decoded.height, pixels))
}

/// Decode with mozjpeg-sys (libjpeg-turbo). Returns (width, height, rgb_pixels).
fn decode_mozjpeg(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
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
            return Err("mozjpeg: bad header".into());
        }

        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        jpeg_start_decompress(&mut cinfo);

        let width = cinfo.output_width;
        let height = cinfo.output_height;
        let components = cinfo.output_components as usize;
        let row_stride = width as usize * components;

        let mut output = vec![0u8; height as usize * row_stride];

        while cinfo.output_scanline < height {
            let offset = cinfo.output_scanline as usize * row_stride;
            let mut row_ptr = output[offset..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
        }

        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);

        Ok((width, height, output))
    }
}

// ── sRGB normalization ───────────────────────────────────────────────────

/// Apply ICC profile to convert pixels to sRGB. Returns pixels unchanged if no ICC.
fn normalize_to_srgb(pixels: &[u8], width: u32, height: u32, icc: Option<&[u8]>) -> Vec<u8> {
    match icc {
        Some(profile) => apply_icc_transform(
            pixels,
            width as usize,
            height as usize,
            profile,
            TargetColorSpace::Srgb,
        )
        .unwrap_or_else(|_| pixels.to_vec()),
        None => pixels.to_vec(),
    }
}

// ── File collection ──────────────────────────────────────────────────────

fn collect_files(dir: &Path) -> Vec<PathBuf> {
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
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if SKIP_DIRS.iter().any(|&s| name == s) {
                continue;
            }
            collect_inner(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn short_path(path: &Path, base: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn is_jpeg_by_magic(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8
}

// ── Per-file result ──────────────────────────────────────────────────────

struct CompareResult {
    path: String,
    size: usize,
    width: u32,
    height: u32,
    has_icc: bool,
    score: f64,
    raw_max_diff: u8,
    srgb_max_diff: u8,
    error: Option<String>,
}

fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn process_file(path: &Path, base: &Path) -> Option<CompareResult> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() > MAX_FILE_SIZE {
        return None;
    }

    let data = std::fs::read(path).ok()?;
    if !is_jpeg_by_magic(&data) {
        return None;
    }

    let sp = short_path(path, base);
    let size = data.len();

    // Extract ICC profile before decoding
    let icc_profile = extract_icc_profile(&data);
    let has_icc = icc_profile.is_some();

    let err_result = |error: String| CompareResult {
        path: sp.clone(),
        size,
        width: 0,
        height: 0,
        has_icc,
        score: 0.0,
        raw_max_diff: 0,
        srgb_max_diff: 0,
        error: Some(error),
    };

    // Decode with both decoders
    let zen = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_zenjpeg(&data)))
    {
        Ok(r) => r,
        Err(_) => return Some(err_result("zenjpeg panicked".into())),
    };

    let moz = match std::panic::catch_unwind(|| decode_mozjpeg(&data)) {
        Ok(r) => r,
        Err(_) => return Some(err_result("mozjpeg panicked".into())),
    };

    let (zw, zh, zen_pixels) = match zen {
        Ok(v) => v,
        Err(e) => return Some(err_result(format!("zenjpeg: {e}"))),
    };

    let (mw, mh, moz_pixels) = match moz {
        Ok(v) => v,
        Err(e) => return Some(err_result(format!("mozjpeg: {e}"))),
    };

    // Dimensions must match
    if zw != mw || zh != mh {
        return Some(err_result(format!(
            "dimension mismatch: zen {zw}x{zh} vs moz {mw}x{mh}"
        )));
    }

    // zensim requires minimum 8x8
    if zw < MIN_DIM || zh < MIN_DIM {
        return None;
    }

    // Raw pixel diff (before ICC)
    let raw_max_diff = max_pixel_diff(&zen_pixels, &moz_pixels);

    // Normalize both to sRGB via ICC transform
    let zen_srgb = normalize_to_srgb(&zen_pixels, zw, zh, icc_profile.as_deref());
    let moz_srgb = normalize_to_srgb(&moz_pixels, mw, mh, icc_profile.as_deref());

    // sRGB pixel diff (after ICC)
    let srgb_max_diff = max_pixel_diff(&zen_srgb, &moz_srgb);

    // Cast to &[[u8; 3]] for zensim
    let zen_rgb: &[[u8; 3]] = bytemuck::cast_slice(&zen_srgb);
    let moz_rgb: &[[u8; 3]] = bytemuck::cast_slice(&moz_srgb);

    // Compute zensim score (thread-local instance to avoid contention)
    thread_local! {
        static ZENSIM: Zensim = Zensim::new(ZensimProfile::latest());
    }

    let score = ZENSIM.with(|z| {
        let src = RgbSlice::new(moz_rgb, zw as usize, zh as usize);
        let dst = RgbSlice::new(zen_rgb, zw as usize, zh as usize);
        z.compute(&src, &dst).map(|r| r.score()).unwrap_or(-1.0)
    });

    Some(CompareResult {
        path: sp,
        size,
        width: zw,
        height: zh,
        has_icc,
        score,
        raw_max_diff,
        srgb_max_diff,
        error: None,
    })
}

#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn zensim_compare_decoders() {
    let corpus = corpus_dir();
    if !corpus.exists() {
        println!("Corpus not found at {}, skipping", corpus.display());
        return;
    }

    println!("Collecting files...");
    let files = collect_files(&corpus);
    println!(
        "Found {} files, comparing with {} threads...",
        files.len(),
        rayon::current_num_threads()
    );

    let progress = AtomicU32::new(0);
    let total = files.len() as u32;

    let results: Vec<CompareResult> = files
        .par_iter()
        .filter_map(|path| {
            let done = progress.fetch_add(1, Ordering::Relaxed);
            if done > 0 && done % 5000 == 0 {
                eprintln!("  ... {done}/{total}");
            }
            process_file(path, &corpus)
        })
        .collect();

    // Separate successes and errors
    let mut successes: Vec<&CompareResult> = Vec::new();
    let mut errors: Vec<&CompareResult> = Vec::new();
    let mut icc_scores: Vec<f64> = Vec::new();
    let mut no_icc_scores: Vec<f64> = Vec::new();

    for r in &results {
        if r.error.is_some() {
            errors.push(r);
        } else {
            successes.push(r);
            if r.has_icc {
                icc_scores.push(r.score);
            } else {
                no_icc_scores.push(r.score);
            }
        }
    }

    // Sort by score ascending (worst first)
    let mut by_score: Vec<&CompareResult> = successes.clone();
    by_score.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    // Statistics
    let all_scores: Vec<f64> = successes.iter().map(|r| r.score).collect();
    let (mean, min, max, p5) = compute_stats(&all_scores);
    let (icc_mean, icc_min, _, _) = compute_stats(&icc_scores);
    let (no_icc_mean, no_icc_min, _, _) = compute_stats(&no_icc_scores);

    // Print results
    println!("\n=== zensim Decoder Comparison (mozjpeg = reference) ===\n");
    println!("Total JPEGs compared:  {}", successes.len());
    println!("  With ICC profile:    {} images", icc_scores.len());
    println!("  Without ICC profile: {} images", no_icc_scores.len());
    println!("  Errors/skipped:      {} images", errors.len());

    println!("\nzensim scores (100 = identical):");
    println!("  Overall:  mean={mean:.2}  min={min:.2}  max={max:.2}  p5={p5:.2}");
    if !icc_scores.is_empty() {
        println!("  ICC:      mean={icc_mean:.2}  min={icc_min:.2}");
    }
    if !no_icc_scores.is_empty() {
        println!("  No ICC:   mean={no_icc_mean:.2}  min={no_icc_min:.2}");
    }

    // Score distribution
    let buckets = [
        (100.0, 100.0, "perfect (100)"),
        (99.0, 100.0, "99-100"),
        (95.0, 99.0, "95-99"),
        (90.0, 95.0, "90-95"),
        (80.0, 90.0, "80-90"),
        (0.0, 80.0, "<80"),
    ];
    println!("\nScore distribution:");
    for (lo, hi, label) in &buckets {
        let count = all_scores
            .iter()
            .filter(|&&s| {
                if *lo == *hi {
                    (s - lo).abs() < 0.01
                } else {
                    s >= *lo && s < *hi
                }
            })
            .count();
        if count > 0 {
            println!(
                "  {label:>15}: {count:>5} ({:.1}%)",
                count as f64 / all_scores.len() as f64 * 100.0
            );
        }
    }

    // Worst 20 files
    println!("\n--- Worst 20 files (score  raw_diff  srgb_diff  dims  profile  path) ---");
    for r in by_score.iter().take(20) {
        println!(
            "  {:.2}  raw={:<3}  srgb={:<3}  {}x{}  {}  {}",
            r.score,
            r.raw_max_diff,
            r.srgb_max_diff,
            r.width,
            r.height,
            if r.has_icc { "ICC" } else { "sRGB" },
            r.path
        );
    }

    // Errors
    if !errors.is_empty() {
        println!("\n--- Decode errors ({}) ---", errors.len());
        for r in errors.iter().take(10) {
            println!(
                "  {} ({} bytes): {}",
                r.path,
                r.size,
                r.error.as_deref().unwrap_or("?")
            );
        }
        if errors.len() > 10 {
            println!("  ... and {} more", errors.len() - 10);
        }
    }

    // Save full results to TSV
    let results_path = "/tmp/corpus_zensim_comparison.tsv";
    let mut report =
        String::from("score\traw_max_diff\tsrgb_max_diff\twidth\theight\thas_icc\tsize\tpath\n");
    for r in &by_score {
        report.push_str(&format!(
            "{:.4}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.score, r.raw_max_diff, r.srgb_max_diff, r.width, r.height, r.has_icc, r.size, r.path
        ));
    }
    let _ = std::fs::write(results_path, &report);
    println!("\nFull results saved to {results_path}");

    // Assertions: enforce on non-ICC images
    assert!(
        no_icc_min > 90.0,
        "Worst non-ICC zensim score {no_icc_min:.2} is below 90 — decoder regression on sRGB images"
    );
    assert!(
        no_icc_mean > 97.0,
        "Mean non-ICC zensim score {no_icc_mean:.2} is below 97 — widespread sRGB decoder regression"
    );

    // ICC images: report but don't assert (ICC transform differences)
    if icc_min < 70.0 {
        let bad_count = icc_scores.iter().filter(|&&s| s < 70.0).count();
        println!("\nWARNING: {bad_count} ICC images have zensim score <70");
    }
}

fn compute_stats(scores: &[f64]) -> (f64, f64, f64, f64) {
    if scores.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p5_idx = (scores.len() as f64 * 0.05) as usize;
    let p5 = sorted[p5_idx.min(sorted.len() - 1)];
    (mean, min, max, p5)
}
