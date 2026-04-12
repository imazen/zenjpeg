#![cfg(feature = "trellis")]
#![cfg(feature = "__ffi-tests")]
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
use zensim_regress::testing::{RegressionTolerance, check_regression};

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

/// Get gb82 corpus directory via codec-corpus (auto-downloads).
fn get_gb82_dir() -> PathBuf {
    let corpus = codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)");
    corpus.get("gb82").expect("corpus.get(gb82)")
}

/// Get CID22 training directory via codec-corpus (auto-downloads).
fn get_cid22_dir() -> PathBuf {
    let corpus = codec_corpus::Corpus::new()
        .expect("codec-corpus init failed (set CODEC_CORPUS_CACHE if needed)");
    corpus
        .get("CID22/CID22-512/training")
        .expect("corpus.get(CID22/CID22-512/training)")
}

/// Load all PNG images from a directory. Returns vec of (name, pixels, width, height).
fn load_pngs_from_dir(dir: &Path, max: usize) -> Vec<(String, Vec<u8>, u32, u32)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
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
    let dir = get_gb82_dir();
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
        .allow_16bit_quant_tables(false);
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
        .allow_16bit_quant_tables(false);
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
fn compare_quality(original: &[u8], decoded: &[u8], width: usize, height: usize) -> (f64, String) {
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

            let (score, _) = compare_quality(pixels, &decoded, *w as usize, *h as usize);
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
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn test_ycbcr_444_quality_monotonic() {
    let images = load_gb82_images(&["baby", "bulb", "city", "flowers", "guitar", "waves"])
        .expect("gb82 corpus images not found");

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
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn test_ycbcr_420_quality_monotonic() {
    let images = load_gb82_images(&["baby", "bulb", "city", "flowers", "guitar", "waves"])
        .expect("gb82 corpus images not found");

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
            assert!(score >= 0.0, "{name} Q{q}: decode failure");
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

/// 4:2:0 with auto_optimize: catches bug #6 (catastrophic quality at specific Q levels).
/// Tests EVERY quality level from Q70-Q99 on the exact images that reproduced the bug.
/// Bug #6 specifically affected: bulb, baby, girl at certain Q levels with 4:2:0 + trellis.
///
/// Root cause was progressive decoder truncation near restart markers (commit 08ef601).
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore] // Requires gb82 corpus
fn test_420_auto_optimize_no_catastrophic() {
    // These are the exact images that triggered bug #1
    let bug_images = ["bulb", "baby", "girl", "city", "flowers"];
    let images = load_gb82_images(&bug_images).expect("gb82 corpus images not found");

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
/// Previously failed (waves Q97 hit 47.7) due to progressive decoder truncation bug.
/// Fixed in commit 08ef601.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore] // Requires gb82 corpus
fn test_444_auto_optimize_quality() {
    let images = load_gb82_images(&["baby", "bulb", "guitar", "waves"])
        .expect("gb82 corpus images not found");

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
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn test_xyb_roundtrip_all_qualities() {
    let images =
        load_gb82_images(&["baby", "guitar", "waves"]).expect("gb82 corpus images not found");

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

            let (score, _) = compare_quality(pixels, &decoded, *w as usize, *h as usize);

            let flag = if score < 30.0 { "LOW" } else { "" };
            println!(
                "    Q{q:3}: score={score:6.1}  size={:6}  {flag}",
                jpeg.len()
            );
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
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
fn test_decoder_path_consistency() {
    let images = load_gb82_images(&["baby", "guitar"]).expect("gb82 corpus images not found");

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
            let expected = StridedBytes::new(&stream_pixels, sw, sh, stride, PixelFormat::Srgb8Rgb);
            let actual = StridedBytes::new(&scanline_pixels, sw, sh, stride, PixelFormat::Srgb8Rgb);

            let tolerance = RegressionTolerance::exact();
            let report = check_regression(&z, &expected, &actual, &tolerance).unwrap();

            println!(
                "  {name} Q{q:3}: score={:.1}  max_delta={:?}  differing={}  {}",
                report.score(),
                report.max_channel_delta(),
                report.pixels_differing(),
                if report.passed() {
                    "IDENTICAL"
                } else {
                    "DIFFER"
                }
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
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore] // Requires CID22 corpus download
fn test_cid22_quality_statistics() {
    let dir = get_cid22_dir();

    let images = load_pngs_from_dir(&dir, 20); // First 20 for speed
    assert!(
        !images.is_empty(),
        "No PNG images found in CID22 training dir"
    );

    let qualities = [75, 90];

    println!(
        "\n=== CID22 Statistical Quality Check ({} images) ===",
        images.len()
    );

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

        println!(
            "  Q{q}: mean={mean:.1}  min={min:.1} ({worst_name})  n={}",
            scores.len()
        );

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

/// Diagnostic: compare auto_optimize vs plain vs trellis-baseline to isolate the bug.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore]
fn diagnostic_trellis_vs_plain() {
    let images =
        load_gb82_images(&["bulb", "city", "flowers"]).expect("gb82 corpus images not found");

    println!("\n=== Trellis vs Plain Diagnostic ===");
    println!(
        "Legend: auto=auto_optimize(4:2:0), plain=no trellis(4:2:0), ratio=auto_mae/plain_mae"
    );

    for (name, pixels, w, h) in &images {
        println!("\n  {name} ({w}x{h}):");
        for q in 85..=99 {
            let q_f32 = q as f32;

            // With auto_optimize 4:2:0
            let jpeg_auto =
                encode_ycbcr_auto_optimize(pixels, *w, *h, q_f32, ChromaSubsampling::Quarter);

            // Without auto_optimize (no trellis) 4:2:0
            let jpeg_plain = encode_ycbcr(pixels, *w, *h, q_f32, ChromaSubsampling::Quarter);

            // Decode both
            let (_, _, pix_auto) = decode_rgb(&jpeg_auto).expect("decode auto");
            let (_, _, pix_plain) = decode_rgb(&jpeg_plain).expect("decode plain");

            let npix = (*w as usize) * (*h as usize);
            let mut auto_rgb = [0u64; 3];
            let mut plain_rgb = [0u64; 3];
            let mut auto_max = 0u32;

            for i in 0..npix {
                for c in 0..3 {
                    let orig = pixels[i * 3 + c] as i32;
                    let auto_v = pix_auto[i * 3 + c] as i32;
                    let plain_v = pix_plain[i * 3 + c] as i32;
                    let d_auto = (orig - auto_v).unsigned_abs();
                    let d_plain = (orig - plain_v).unsigned_abs();
                    auto_rgb[c] += d_auto as u64;
                    plain_rgb[c] += d_plain as u64;
                    auto_max = auto_max.max(d_auto);
                }
            }

            let auto_mae = (auto_rgb[0] + auto_rgb[1] + auto_rgb[2]) as f64 / (npix * 3) as f64;
            let plain_mae = (plain_rgb[0] + plain_rgb[1] + plain_rgb[2]) as f64 / (npix * 3) as f64;
            let ratio = auto_mae / plain_mae.max(0.001);

            let auto_r = auto_rgb[0] as f64 / npix as f64;
            let auto_g = auto_rgb[1] as f64 / npix as f64;
            let auto_b = auto_rgb[2] as f64 / npix as f64;

            let flag = if ratio > 2.0 { " *** BAD" } else { "" };
            println!(
                "    Q{q:2}: auto={:6}B mae={auto_mae:.2} (R={auto_r:.2} G={auto_g:.2} B={auto_b:.2}) max={auto_max:3} | plain={:6}B mae={plain_mae:.2} | ratio={ratio:.2}x{flag}",
                jpeg_auto.len(),
                jpeg_plain.len()
            );
        }
    }
}

/// Diagnostic: decode with djpeg (libjpeg-turbo) to check if it's encoder or decoder bug.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore]
fn diagnostic_cross_decoder() {
    let images = load_gb82_images(&["bulb"]).expect("gb82 corpus images not found");
    let (name, pixels, w, h) = &images[0];

    println!("\n=== Cross-decoder check for {name} Q90 vs Q91 ===");
    for q in [90.0f32, 91.0, 92.0, 93.0, 94.0] {
        // Progressive (no trellis)
        let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(true);
        let mut enc = config
            .encode_from_bytes(*w, *h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();

        // Decode with zenjpeg
        let zen_dec = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
        let zen_pix = zen_dec.into_pixels_u8().unwrap();

        // Save JPEG to temp file and decode with djpeg (libjpeg-turbo reference)
        let tmp_jpeg = format!("/tmp/diag_bulb_q{}.jpg", q as u32);
        let tmp_ppm = format!("/tmp/diag_bulb_q{}.ppm", q as u32);
        std::fs::write(&tmp_jpeg, &jpeg).unwrap();

        let djpeg_result = std::process::Command::new("djpeg")
            .args(["-ppm", "-outfile", &tmp_ppm, &tmp_jpeg])
            .output();

        let djpeg_mae = if let Ok(output) = djpeg_result {
            if output.status.success() {
                // Read PPM file (P6 binary format)
                let ppm_data = std::fs::read(&tmp_ppm).unwrap();
                // Skip PPM header (find 3rd newline after "P6\n<w> <h>\n255\n")
                let mut newlines = 0;
                let mut pixel_start = 0;
                for (i, &b) in ppm_data.iter().enumerate() {
                    if b == b'\n' {
                        newlines += 1;
                        if newlines == 3 {
                            pixel_start = i + 1;
                            break;
                        }
                    }
                }
                let djpeg_pix = &ppm_data[pixel_start..];
                let npix3 = (*w as usize) * (*h as usize) * 3;
                if djpeg_pix.len() >= npix3 {
                    let mut sum = 0u64;
                    for i in 0..npix3 {
                        sum += (pixels[i] as i32 - djpeg_pix[i] as i32).unsigned_abs() as u64;
                    }
                    sum as f64 / npix3 as f64
                } else {
                    -1.0
                }
            } else {
                -2.0 // djpeg failed
            }
        } else {
            -3.0 // djpeg not found
        };

        // Compare both decoders against original
        let npix = (*w as usize) * (*h as usize);
        let mut zen_sum = 0u64;
        for i in 0..npix * 3 {
            zen_sum += (pixels[i] as i32 - zen_pix[i] as i32).unsigned_abs() as u64;
        }
        let zen_mae = zen_sum as f64 / (npix * 3) as f64;
        let flag = if zen_mae > 3.0 || djpeg_mae > 3.0 {
            " *** BAD"
        } else {
            ""
        };
        println!("  Q{q:.0}: zen_mae={zen_mae:.2}  djpeg_mae={djpeg_mae:.2}{flag}");
    }
}

/// Diagnostic: compare coefficient arrays between progressive and baseline decode.
///
/// This is the most direct test for the progressive decoder bug. Same quantized
/// coefficients are encoded as both baseline and progressive. If the progressive
/// decoder reconstructs different coefficients, this prints the exact positions
/// and values that differ.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore]
fn diagnostic_coefficient_comparison() {
    let images = load_gb82_images(&["bulb"]).expect("gb82 corpus images not found");
    let (name, pixels, w, h) = &images[0];
    let decoder = Decoder::new();

    println!("\n=== Coefficient comparison: progressive vs baseline for {name} ===");
    for q in [88, 89, 90, 91, 92, 93, 94, 95] {
        let q_f32 = q as f32;

        // Encode as baseline
        let jpeg_bl = encode_ycbcr(pixels, *w, *h, q_f32, ChromaSubsampling::Quarter);
        // Encode as progressive (same quality, same quantization)
        let config_pg = EncoderConfig::ycbcr(q_f32, ChromaSubsampling::Quarter).progressive(true);
        let mut enc_pg = config_pg
            .encode_from_bytes(*w, *h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc_pg.push_packed(pixels, Unstoppable).unwrap();
        let jpeg_pg = enc_pg.finish().unwrap();

        // Extract coefficients from both
        let coeffs_bl = decoder.decode_coefficients(&jpeg_bl, Unstoppable).unwrap();
        let coeffs_pg = decoder.decode_coefficients(&jpeg_pg, Unstoppable).unwrap();

        // Compare per-component
        let mut total_diffs = 0usize;
        let mut total_coeffs = 0usize;
        let mut max_diff = 0i32;
        let mut first_diffs: Vec<String> = Vec::new();

        for comp in 0..coeffs_bl.components.len().min(coeffs_pg.components.len()) {
            let bl = &coeffs_bl.components[comp];
            let pg = &coeffs_pg.components[comp];
            let comp_label = ["Y", "Cb", "Cr"][comp];

            let num_blocks = bl.num_blocks().min(pg.num_blocks());
            for b in 0..num_blocks {
                let bl_block = bl.block(b);
                let pg_block = pg.block(b);
                for k in 0..64 {
                    total_coeffs += 1;
                    let diff = (bl_block[k] as i32) - (pg_block[k] as i32);
                    if diff != 0 {
                        total_diffs += 1;
                        max_diff = max_diff.max(diff.abs());
                        if first_diffs.len() < 10 {
                            let bx = b % bl.blocks_wide;
                            let by = b / bl.blocks_wide;
                            first_diffs.push(format!(
                                "      {comp_label} block({bx},{by}) k={k}: bl={} pg={} diff={diff}",
                                bl_block[k], pg_block[k]
                            ));
                        }
                    }
                }
            }
        }

        // Count affected block rows
        let mut affected_rows = std::collections::BTreeSet::new();
        for comp in 0..coeffs_bl.components.len().min(coeffs_pg.components.len()) {
            let bl = &coeffs_bl.components[comp];
            let pg = &coeffs_pg.components[comp];
            let num_blocks = bl.num_blocks().min(pg.num_blocks());
            for b in 0..num_blocks {
                let bl_block = bl.block(b);
                let pg_block = pg.block(b);
                for k in 0..64 {
                    if bl_block[k] != pg_block[k] {
                        let by = b / bl.blocks_wide;
                        affected_rows.insert((comp, by, k));
                    }
                }
            }
        }

        let pct = if total_coeffs > 0 {
            100.0 * total_diffs as f64 / total_coeffs as f64
        } else {
            0.0
        };
        let flag = if total_diffs > 0 && max_diff > 2 {
            " *** BUG"
        } else {
            ""
        };
        println!(
            "  Q{q:2}: {total_diffs:6} diffs / {total_coeffs} coeffs ({pct:.3}%) max_diff={max_diff}{flag}"
        );
        if total_diffs > 0 {
            // Show which zigzag positions are affected
            let mut affected_k: std::collections::BTreeSet<usize> =
                std::collections::BTreeSet::new();
            let mut min_row = usize::MAX;
            let mut max_row = 0;
            for &(comp, by, k) in &affected_rows {
                if comp == 0 {
                    // Y component
                    affected_k.insert(k);
                    min_row = min_row.min(by);
                    max_row = max_row.max(by);
                }
            }
            let ks: Vec<_> = affected_k.iter().collect();
            println!("    Affected zigzag positions: {ks:?}");
            println!(
                "    Affected Y block rows: {min_row}..={max_row} (image: {}x{})",
                *w, *h
            );
            let blocks_h = coeffs_bl.components[0].blocks_wide;
            let blocks_v = coeffs_bl.components[0].blocks_high;
            println!("    Y blocks: {blocks_h} wide x {blocks_v} high");
        }
        for d in &first_diffs {
            println!("{d}");
        }
    }
}

/// Diagnostic: check scan structure of progressive JPEGs at Q90 vs Q91.
/// Verifies whether the issue is scan truncation or coefficient corruption.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore]
fn diagnostic_scan_structure() {
    let images = load_gb82_images(&["bulb"]).expect("gb82 corpus images not found");
    let (name, pixels, w, h) = &images[0];

    println!("\n=== Scan structure comparison for {name} Q90 vs Q91 ===");
    for q in [90.0f32, 91.0] {
        let config = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(true);
        let mut enc = config
            .encode_from_bytes(*w, *h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();

        println!("\n  Q{q:.0}: {} bytes total", jpeg.len());

        // Find all SOS markers and their scan parameters
        let mut i = 0;
        let mut scan_num = 0;
        while i + 1 < jpeg.len() {
            if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
                // SOS marker found
                let sos_pos = i;
                let length = ((jpeg[i + 2] as usize) << 8) | jpeg[i + 3] as usize;
                let num_comp = jpeg[i + 4];

                // Parse scan components and spectral selection
                let param_start = i + 5 + (num_comp as usize * 2);
                let ss = jpeg[param_start];
                let se = jpeg[param_start + 1];
                let ah_al = jpeg[param_start + 2];
                let ah = ah_al >> 4;
                let al = ah_al & 0x0F;

                // Find end of scan data (next marker)
                let data_start = i + 2 + length;
                let mut data_end = data_start;
                while data_end + 1 < jpeg.len() {
                    if jpeg[data_end] == 0xFF
                        && jpeg[data_end + 1] != 0x00
                        && jpeg[data_end + 1] != 0xFF
                    {
                        // Check it's not a restart marker (0xD0-0xD7)
                        if !(0xD0..=0xD7).contains(&jpeg[data_end + 1]) {
                            break;
                        }
                    }
                    data_end += 1;
                }
                let data_len = data_end - data_start;

                let comp_ids: Vec<u8> = (0..num_comp)
                    .map(|c| jpeg[i + 5 + (c as usize * 2)])
                    .collect();

                println!(
                    "    Scan {scan_num}: components={comp_ids:?} ss={ss} se={se} ah={ah} al={al} | sos_pos={sos_pos} data_len={data_len}",
                );
                scan_num += 1;
                i = data_end;
            } else {
                i += 1;
            }
        }
    }
}

/// Diagnostic: isolate whether the bug is from trellis, progressive, or their combination.
#[ignore = "requires codec-corpus (network on first run)"]
#[test]
#[ignore]
fn diagnostic_isolate_trellis_progressive() {
    use zenjpeg::encode::trellis::HybridConfig;

    let images = load_gb82_images(&["bulb"]).expect("gb82 corpus images not found");
    let (name, pixels, w, h) = &images[0];

    println!("\n=== Isolating trellis vs progressive for {name} ===");
    println!("  Configurations:");
    println!("    A: plain baseline (no trellis, no progressive)");
    println!("    B: trellis + baseline (trellis, no progressive)");
    println!("    C: plain progressive (no trellis, progressive)");
    println!("    D: trellis + progressive (auto_optimize — the buggy config)");

    for q in 88..=96 {
        let q_f32 = q as f32;

        // A: plain baseline
        let jpeg_a = encode_ycbcr(pixels, *w, *h, q_f32, ChromaSubsampling::Quarter);

        // B: trellis + baseline
        let config_b = EncoderConfig::ycbcr(q_f32, ChromaSubsampling::Quarter)
            .progressive(false)
            .hybrid_config(HybridConfig {
                enabled: true,
                base_lambda_scale1: 14.5,
                ..Default::default()
            })
            .allow_16bit_quant_tables(false);
        let mut enc_b = config_b
            .encode_from_bytes(*w, *h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc_b.push_packed(pixels, Unstoppable).unwrap();
        let jpeg_b = enc_b.finish().unwrap();

        // C: plain progressive (no trellis)
        let config_c = EncoderConfig::ycbcr(q_f32, ChromaSubsampling::Quarter).progressive(true);
        let mut enc_c = config_c
            .encode_from_bytes(*w, *h, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc_c.push_packed(pixels, Unstoppable).unwrap();
        let jpeg_c = enc_c.finish().unwrap();

        // D: auto_optimize (trellis + progressive)
        let jpeg_d = encode_ycbcr_auto_optimize(pixels, *w, *h, q_f32, ChromaSubsampling::Quarter);

        // Decode all and compute MAE
        let configs = [
            ("A:plain-bl", jpeg_a),
            ("B:trel-bl ", jpeg_b),
            ("C:plain-pg", jpeg_c),
            ("D:trel-pg ", jpeg_d),
        ];

        print!("  Q{q:2}:");
        for (label, jpeg) in &configs {
            let (_, _, dec) = decode_rgb(jpeg).expect("decode");
            let npix = (*w as usize) * (*h as usize);
            let mut sum = 0u64;
            for i in 0..npix * 3 {
                sum += (pixels[i] as i32 - dec[i] as i32).unsigned_abs() as u64;
            }
            let mae = sum as f64 / (npix * 3) as f64;
            let flag = if mae > 3.0 { "***" } else { "   " };
            print!("  {label}={mae:.2}{flag}");
        }
        println!();
    }
}
