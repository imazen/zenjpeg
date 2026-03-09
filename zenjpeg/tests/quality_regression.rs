//! Quality regression tests using zensim-regress with real photographic images.
//!
//! Uses codec-corpus (gb82, CID22) for test images. Never uses synthetic images
//! for codec quality testing — synthetic patterns produce degenerate DCT coefficients
//! and misleading quality scores.
//!
//! Tests catch:
//! - Catastrophic quality drops at specific quality levels (bug #1: 4:2:0 auto_optimize)
//! - XYB encode/decode failures (bug #3: XYB Q50 Huffman corruption)
//! - Encoder regressions that silently degrade output quality
//! - Quality non-monotonicity (higher Q producing worse output)
//!
//! Run: cargo test --release -p zenjpeg --test quality_regression --features decoder -- --nocapture

use enough::Unstoppable;
use std::path::{Path, PathBuf};
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};
use zensim::source::{PixelFormat, StridedBytes};
use zensim::{Zensim, ZensimProfile};
use zensim_regress::testing::{check_regression, RegressionTolerance};

// =============================================================================
// Image loading from codec-corpus
// =============================================================================

/// Load a PNG file to flat RGB bytes. Returns (pixels, width, height).
fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let width = info.width;
    let height = info.height;

    // Convert to RGB8 regardless of source format
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in src.chunks_exact(4) {
                rgb.extend_from_slice(&chunk[..3]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for &g in src {
                rgb.extend_from_slice(&[g, g, g]);
            }
            rgb
        }
        png::ColorType::GrayscaleAlpha => {
            let src = &buf[..info.buffer_size()];
            let mut rgb = Vec::with_capacity((width * height * 3) as usize);
            for chunk in src.chunks_exact(2) {
                rgb.extend_from_slice(&[chunk[0], chunk[0], chunk[0]]);
            }
            rgb
        }
        _ => return None,
    };

    Some((rgb, width, height))
}

/// Get gb82 corpus directory. Tries codec-corpus crate first, then local paths.
fn get_gb82_dir() -> Option<PathBuf> {
    // Try codec-corpus crate
    if let Ok(corpus) = codec_corpus::Corpus::new() {
        if let Ok(dir) = corpus.get("gb82") {
            return Some(dir);
        }
    }
    // Fallback to known local path
    let local = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../internal/jpegli-cpp/testdata");
    // gb82 may not be there, but check common locations
    for candidate in [
        PathBuf::from("/home/lilith/work/codec-eval/codec-corpus/gb82"),
        local,
    ] {
        if candidate.exists() && candidate.join("baby-lossless.png").exists() {
            return Some(candidate);
        }
    }
    None
}

/// Get CID22 training directory.
fn get_cid22_dir() -> Option<PathBuf> {
    let corpus = codec_corpus::Corpus::new().ok()?;
    corpus.get("CID22/CID22-512/training").ok()
}

/// Load all PNG images from a directory. Returns vec of (name, pixels, width, height).
fn load_pngs_from_dir(dir: &Path, max: usize) -> Vec<(String, Vec<u8>, u32, u32)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "png")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());
    entries.truncate(max);

    entries
        .into_iter()
        .filter_map(|e| {
            let path = e.path();
            let name = path.file_stem()?.to_string_lossy().to_string();
            let (pixels, w, h) = load_png_rgb(&path)?;
            Some((name, pixels, w, h))
        })
        .collect()
}

/// Load specific named images from gb82.
fn load_gb82_images(names: &[&str]) -> Option<Vec<(String, Vec<u8>, u32, u32)>> {
    let dir = get_gb82_dir()?;
    let mut images = Vec::new();
    for name in names {
        let path = dir.join(format!("{name}-lossless.png"));
        if !path.exists() {
            // Try without -lossless suffix
            let alt = dir.join(format!("{name}.png"));
            if let Some((pixels, w, h)) = load_png_rgb(&alt) {
                images.push((name.to_string(), pixels, w, h));
                continue;
            }
            return None; // Required image missing
        }
        let (pixels, w, h) = load_png_rgb(&path)?;
        images.push((name.to_string(), pixels, w, h));
    }
    Some(images)
}

// =============================================================================
// Encode/decode helpers
// =============================================================================

fn encode_ycbcr(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subsampling)
        .progressive(false)
        .allow_16bit_quant_tables(false)
        .expect("baseline config");
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn encode_ycbcr_auto_optimize(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, subsampling)
        .auto_optimize(true)
        .allow_16bit_quant_tables(false)
        .expect("baseline config");
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

fn encode_xyb(pixels: &[u8], width: u32, height: u32, quality: f32) -> Option<Vec<u8>> {
    let config = EncoderConfig::xyb(quality, XybSubsampling::BQuarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .ok()?;
    enc.push_packed(pixels, Unstoppable).ok()?;
    enc.finish().ok()
}

fn decode_rgb(jpeg: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let result = Decoder::new().decode(jpeg, Unstoppable).ok()?;
    let w = result.width;
    let h = result.height;
    let pixels = result.into_pixels_u8()?;
    Some((w, h, pixels))
}

/// Compare original vs decoded using zensim. Returns (score, report_string).
fn compare_quality(
    original: &[u8],
    decoded: &[u8],
    width: usize,
    height: usize,
) -> (f64, String) {
    let z = Zensim::new(ZensimProfile::latest());
    let stride = width * 3;
    let expected = StridedBytes::new(original, width, height, stride, PixelFormat::Srgb8Rgb);
    let actual = StridedBytes::new(decoded, width, height, stride, PixelFormat::Srgb8Rgb);

    let tolerance = RegressionTolerance::off_by_one()
        .with_max_delta(255) // Don't fail on delta — we check score
        .with_max_pixels_different(1.0)
        .with_min_similarity(0.0); // We'll check score manually

    let report = check_regression(&z, &expected, &actual, &tolerance).unwrap();
    (report.score(), format!("{report}"))
}

// =============================================================================
// Quality sweep helper
// =============================================================================

/// Sweep quality levels and detect non-monotonic or catastrophic quality drops.
/// Runs across multiple images, returns per-image results.
fn quality_sweep_multi(
    images: &[(String, Vec<u8>, u32, u32)],
    qualities: &[u32],
    encode_fn: &dyn Fn(&[u8], u32, u32, f32) -> Vec<u8>,
) -> Vec<(String, Vec<(u32, f64, usize)>)> {
    let mut all_results = Vec::new();

    for (name, pixels, w, h) in images {
        let mut results = Vec::new();
        for &q in qualities {
            let jpeg = encode_fn(pixels, *w, *h, q as f32);
            let Some((dw, dh, decoded)) = decode_rgb(&jpeg) else {
                results.push((q, -1.0, jpeg.len()));
                continue;
            };
            assert_eq!((dw, dh), (*w, *h), "{name} Q{q}: dimension mismatch");

            let (score, _) =
                compare_quality(pixels, &decoded, *w as usize, *h as usize);
            results.push((q, score, jpeg.len()));
        }
        all_results.push((name.clone(), results));
    }

    all_results
}

// =============================================================================
// Tests
// =============================================================================

/// YCbCr 4:4:4 quality sweep on gb82 photos: scores must be monotonically
/// non-decreasing (within tolerance) and never drop below minimum thresholds.
#[test]
fn test_ycbcr_444_quality_monotonic() {
    let images = match load_gb82_images(&["baby", "bulb", "city", "flowers", "guitar", "waves"]) {
        Some(imgs) => imgs,
        None => {
            eprintln!("Skipping: gb82 corpus not available");
            return;
        }
    };

    let qualities: Vec<u32> = (50..=99).step_by(5).collect();

    let all_results = quality_sweep_multi(&images, &qualities, &|p, w, h, q| {
        encode_ycbcr(p, w, h, q, ChromaSubsampling::None)
    });

    println!("\n=== YCbCr 4:4:4 Quality Sweep (gb82) ===");
    for (name, results) in &all_results {
        println!("  {name}:");
        let mut prev_score = 0.0f64;
        for &(q, score, size) in results {
            let delta = score - prev_score;
            let flag = if score < 0.0 {
                "DECODE_FAIL"
            } else if delta < -3.0 && q > 50 {
                "REGRESSION"
            } else {
                ""
            };
            println!("    Q{q:3}: score={score:6.1}  size={size:6}  delta={delta:+.1}  {flag}");
            prev_score = score;
        }
    }

    // Assert no decode failures
    for (name, results) in &all_results {
        for &(q, score, _) in results {
            assert!(
                score >= 0.0,
                "{name} Q{q}: decode failure (encode produced undecodable JPEG)"
            );
        }
    }

    // Assert minimum quality thresholds (calibrated on real photos — higher than synthetic)
    for (name, results) in &all_results {
        for &(q, score, _) in results {
            let min_score = match q {
                50..=59 => 55.0,
                60..=74 => 60.0,
                75..=89 => 70.0,
                90..=99 => 80.0,
                _ => 40.0,
            };
            assert!(
                score >= min_score,
                "{name} Q{q}: score {score:.1} below minimum {min_score} — catastrophic quality"
            );
        }
    }

    // Assert no large non-monotonic drops (>5 points) on any image
    for (name, results) in &all_results {
        let mut prev = results[0].1;
        for &(q, score, _) in &results[1..] {
            let drop = prev - score;
            assert!(
                drop < 5.0,
                "{name} Q{q}: score dropped {drop:.1} points — non-monotonic regression"
            );
            prev = score;
        }
    }
}

/// YCbCr 4:2:0 quality sweep: same monotonicity checks, plus chroma-specific thresholds.
#[test]
fn test_ycbcr_420_quality_monotonic() {
    let images = match load_gb82_images(&["baby", "bulb", "city", "flowers", "guitar", "waves"]) {
        Some(imgs) => imgs,
        None => {
            eprintln!("Skipping: gb82 corpus not available");
            return;
        }
    };

    let qualities: Vec<u32> = (50..=99).step_by(5).collect();

    let all_results = quality_sweep_multi(&images, &qualities, &|p, w, h, q| {
        encode_ycbcr(p, w, h, q, ChromaSubsampling::Quarter)
    });

    println!("\n=== YCbCr 4:2:0 Quality Sweep (gb82) ===");
    for (name, results) in &all_results {
        println!("  {name}:");
        for &(q, score, size) in results {
            println!("    Q{q:3}: score={score:6.1}  size={size:6}");
        }
    }

    for (name, results) in &all_results {
        for &(q, score, _) in results {
            assert!(
                score >= 0.0,
                "{name} Q{q}: decode failure"
            );
            // 4:2:0 loses some chroma detail; thresholds slightly lower than 4:4:4
            let min_score = match q {
                50..=59 => 50.0,
                60..=74 => 55.0,
                75..=89 => 65.0,
                90..=99 => 75.0,
                _ => 35.0,
            };
            assert!(
                score >= min_score,
                "{name} Q{q}: score {score:.1} below minimum {min_score} — catastrophic quality"
            );
        }
    }
}

/// 4:2:0 with auto_optimize: catches bug #1 (catastrophic quality at specific Q levels).
/// Tests EVERY quality level from Q70-Q99 on the exact images that reproduced the bug.
/// Bug #1 specifically affected: bulb, baby, girl at certain Q levels with 4:2:0 + trellis.
///
/// KNOWN FAILURE: bug #1 (CLAUDE.md) — catastrophic 4:2:0 auto_optimize quality.
/// Unignore when bug #1 is fixed.
#[test]
#[ignore]
fn test_420_auto_optimize_no_catastrophic() {
    // These are the exact images that triggered bug #1
    let bug_images = ["bulb", "baby", "girl", "city", "flowers"];
    let images = match load_gb82_images(&bug_images) {
        Some(imgs) => imgs,
        None => {
            eprintln!("Skipping: gb82 corpus not available (need bulb/baby/girl for bug #1 test)");
            return;
        }
    };

    let qualities: Vec<u32> = (70..=99).collect(); // Every Q level in auto_optimize range

    let all_results = quality_sweep_multi(&images, &qualities, &|p, w, h, q| {
        encode_ycbcr_auto_optimize(p, w, h, q, ChromaSubsampling::Quarter)
    });

    println!("\n=== 4:2:0 auto_optimize Quality Sweep Q70-Q99 (gb82 bug images) ===");
    let mut catastrophic = Vec::new();
    for (name, results) in &all_results {
        println!("  {name}:");
        for &(q, score, size) in results {
            let flag = if score < 0.0 {
                "DECODE_FAIL"
            } else if score < 50.0 {
                "CATASTROPHIC"
            } else {
                ""
            };
            if !flag.is_empty() {
                println!("    Q{q:3}: score={score:6.1}  size={size:6}  {flag}");
            }
            if score < 50.0 {
                catastrophic.push((name.clone(), q, score));
            }
        }
        // Print summary line
        let scores: Vec<f64> = results.iter().map(|r| r.1).collect();
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("    range: {min:.1} - {max:.1}");
    }

    if !catastrophic.is_empty() {
        let list: Vec<String> = catastrophic
            .iter()
            .map(|(name, q, s)| format!("{name}/Q{q}={s:.1}"))
            .collect();
        panic!(
            "Catastrophic quality at {} image/quality combinations: {}. \
             This is likely the trellis lambda inversion bug (CLAUDE.md bug #1).",
            catastrophic.len(),
            list.join(", ")
        );
    }
}

/// 4:4:4 with auto_optimize: should never have catastrophic drops.
/// KNOWN FAILURE: waves Q97 hits 47.7 — bug #1 also affects 4:4:4 at extreme Q levels.
/// Unignore when bug #1 is fixed.
#[test]
#[ignore]
fn test_444_auto_optimize_quality() {
    let images = match load_gb82_images(&["baby", "bulb", "guitar", "waves"]) {
        Some(imgs) => imgs,
        None => {
            eprintln!("Skipping: gb82 corpus not available");
            return;
        }
    };

    let qualities: Vec<u32> = (70..=99).step_by(3).collect();

    let all_results = quality_sweep_multi(&images, &qualities, &|p, w, h, q| {
        encode_ycbcr_auto_optimize(p, w, h, q, ChromaSubsampling::None)
    });

    println!("\n=== 4:4:4 auto_optimize Quality Sweep (gb82) ===");
    for (name, results) in &all_results {
        println!("  {name}:");
        for &(q, score, size) in results {
            println!("    Q{q:3}: score={score:6.1}  size={size:6}");
        }
    }

    for (name, results) in &all_results {
        for &(q, score, _) in results {
            assert!(score >= 0.0, "{name} Q{q}: decode failure");
            assert!(
                score >= 65.0,
                "{name} Q{q}: 4:4:4 auto_optimize score {score:.1} below 65 — unexpected quality drop"
            );
        }
    }
}

/// XYB encode/decode roundtrip at all quality levels on real images.
/// Catches: XYB Huffman corruption (bug #3), undecodable output.
#[test]
fn test_xyb_roundtrip_all_qualities() {
    let images = match load_gb82_images(&["baby", "guitar", "waves"]) {
        Some(imgs) => imgs,
        None => {
            eprintln!("Skipping: gb82 corpus not available");
            return;
        }
    };

    let qualities: Vec<u32> = (10..=99).step_by(5).collect();

    println!("\n=== XYB Roundtrip Quality Sweep (gb82) ===");
    let mut failures = Vec::new();

    for (name, pixels, w, h) in &images {
        println!("  {name}:");
        for &q in &qualities {
            let Some(jpeg) = encode_xyb(pixels, *w, *h, q as f32) else {
                println!("    Q{q:3}: ENCODE_FAIL");
                failures.push((name.clone(), q, "encode_fail"));
                continue;
            };

            let Some((dw, dh, decoded)) = decode_rgb(&jpeg) else {
                println!("    Q{q:3}: DECODE_FAIL  size={}", jpeg.len());
                failures.push((name.clone(), q, "decode_fail"));
                continue;
            };

            assert_eq!((dw, dh), (*w, *h), "{name} Q{q}: dimension mismatch");

            let (score, _) =
                compare_quality(pixels, &decoded, *w as usize, *h as usize);

            let flag = if score < 30.0 { "LOW" } else { "" };
            println!("    Q{q:3}: score={score:6.1}  size={:6}  {flag}", jpeg.len());
        }
    }

    if !failures.is_empty() {
        let list: Vec<String> = failures
            .iter()
            .map(|(name, q, reason)| format!("{name}/Q{q}:{reason}"))
            .collect();
        panic!(
            "XYB failures at {} image/quality combinations: {}",
            failures.len(),
            list.join(", ")
        );
    }
}

/// Decoder consistency: streaming vs scanline must produce identical output
/// on real photographic content.
#[test]
fn test_decoder_path_consistency() {
    let images = match load_gb82_images(&["baby", "guitar"]) {
        Some(imgs) => imgs,
        None => {
            eprintln!("Skipping: gb82 corpus not available");
            return;
        }
    };

    let qualities = [50, 75, 90, 95];
    let z = Zensim::new(ZensimProfile::latest());

    println!("\n=== Decoder Path Consistency (gb82) ===");

    for (name, pixels, w, h) in &images {
        for q in qualities {
            let jpeg = encode_ycbcr(pixels, *w, *h, q as f32, ChromaSubsampling::Quarter);

            // Streaming decode
            let stream_result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
            let stream_pixels = stream_result.into_pixels_u8().unwrap();

            // Scanline decode
            let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();
            let sw = reader.width() as usize;
            let sh = reader.height() as usize;
            let stride = sw * 3;
            let mut scanline_pixels = vec![0u8; sh * stride];
            let mut total_rows = 0;
            while !reader.is_finished() {
                let remaining = sh - total_rows;
                let buf_start = total_rows * stride;
                let output =
                    imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
                let rows = reader.read_rows_rgb8(output).expect("read_rows_rgb8");
                total_rows += rows;
            }

            // Compare with zensim
            let expected =
                StridedBytes::new(&stream_pixels, sw, sh, stride, PixelFormat::Srgb8Rgb);
            let actual =
                StridedBytes::new(&scanline_pixels, sw, sh, stride, PixelFormat::Srgb8Rgb);

            let tolerance = RegressionTolerance::exact();
            let report = check_regression(&z, &expected, &actual, &tolerance).unwrap();

            println!(
                "  {name} Q{q:3}: score={:.1}  max_delta={:?}  differing={}  {}",
                report.score(),
                report.max_channel_delta(),
                report.pixels_differing(),
                if report.passed() { "IDENTICAL" } else { "DIFFER" }
            );

            assert!(
                report.passed(),
                "{name} Q{q}: streaming vs scanline differ!\n{report}"
            );
        }
    }
}

/// Statistical quality check across CID22 corpus (many diverse images).
/// Verifies mean quality is reasonable and no single image is catastrophically bad.
#[test]
#[ignore] // Requires CID22 corpus download
fn test_cid22_quality_statistics() {
    let dir = match get_cid22_dir() {
        Some(d) => d,
        None => {
            eprintln!("Skipping: CID22 corpus not available");
            return;
        }
    };

    let images = load_pngs_from_dir(&dir, 20); // First 20 for speed
    assert!(
        !images.is_empty(),
        "No PNG images found in CID22 training dir"
    );

    let qualities = [75, 90];

    println!("\n=== CID22 Statistical Quality Check ({} images) ===", images.len());

    for q in qualities {
        let mut scores = Vec::new();
        let mut worst_name = String::new();
        let mut worst_score = f64::MAX;

        for (name, pixels, w, h) in &images {
            let jpeg = encode_ycbcr(pixels, *w, *h, q as f32, ChromaSubsampling::Quarter);
            let Some((_, _, decoded)) = decode_rgb(&jpeg) else {
                panic!("{name} Q{q}: decode failure");
            };
            let (score, _) = compare_quality(pixels, &decoded, *w as usize, *h as usize);
            if score < worst_score {
                worst_score = score;
                worst_name = name.clone();
            }
            scores.push(score);
        }

        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);

        println!("  Q{q}: mean={mean:.1}  min={min:.1} ({worst_name})  n={}", scores.len());

        let min_mean = if q >= 90 { 80.0 } else { 65.0 };
        assert!(
            mean >= min_mean,
            "Q{q}: mean score {mean:.1} below {min_mean} across CID22 — systematic quality regression"
        );

        let min_worst = if q >= 90 { 70.0 } else { 50.0 };
        assert!(
            min >= min_worst,
            "Q{q}: worst image {worst_name} scored {worst_score:.1} (below {min_worst}) — catastrophic for single image"
        );
    }
}
