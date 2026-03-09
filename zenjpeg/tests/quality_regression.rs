//! Quality regression tests using zensim-regress.
//!
//! These tests encode images at various quality levels and decoder configurations,
//! then verify that decoded output meets quality thresholds. Catches:
//! - Catastrophic quality drops at specific quality levels (bug #1: 4:2:0 auto_optimize)
//! - XYB encode/decode failures (bug #3: XYB Q50 Huffman corruption)
//! - Encoder regressions that silently degrade output quality
//! - Quality non-monotonicity (higher Q producing worse output)
//!
//! Run: cargo test --release -p zenjpeg --test quality_regression --features decoder -- --nocapture

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};
use zensim::source::{PixelFormat, StridedBytes};
use zensim::{Zensim, ZensimProfile};
use zensim_regress::testing::{check_regression, RegressionTolerance};

// =============================================================================
// Test image generators
// =============================================================================

/// Noise+patches: realistic DCT coefficient distribution.
/// NEVER use smooth gradients — they produce degenerate DCT coefficients.
fn make_noise_patches(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Base: smooth color gradient
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 128) / (width + height).max(1)) as u8;
            // Pseudo-noise from hash mixing
            let hash = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(2246822519)) as u32;
            let noise_r = ((hash >> 0) & 0x1F) as u8;
            let noise_g = ((hash >> 5) & 0x1F) as u8;
            let noise_b = ((hash >> 10) & 0x1F) as u8;
            data[idx] = r.saturating_add(noise_r);
            data[idx + 1] = g.saturating_add(noise_g);
            data[idx + 2] = b.saturating_add(noise_b);
        }
    }
    data
}

/// High-contrast blocks: alternating red/blue with green variation.
/// Stresses chroma subsampling and edge artifacts.
fn make_stress_blocks(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let block_y = y / 8;
            if block_y % 2 == 0 {
                data[idx] = 230;
                data[idx + 1] = 30;
                data[idx + 2] = 30;
            } else {
                data[idx] = 30;
                data[idx + 1] = 30;
                data[idx + 2] = 230;
            }
            // Green variation
            if x % 4 < 2 {
                data[idx + 1] = ((x * 3 + y * 7) % 200) as u8;
            }
        }
    }
    data
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
// Quality sweep: detect catastrophic drops
// =============================================================================

/// Sweep quality levels and detect non-monotonic or catastrophic quality drops.
/// Returns Vec<(quality, score, size)> sorted by quality.
fn quality_sweep(
    pixels: &[u8],
    width: u32,
    height: u32,
    qualities: &[u32],
    encode_fn: impl Fn(&[u8], u32, u32, f32) -> Vec<u8>,
) -> Vec<(u32, f64, usize)> {
    let mut results = Vec::new();

    for &q in qualities {
        let jpeg = encode_fn(pixels, width, height, q as f32);
        let Some((w, h, decoded)) = decode_rgb(&jpeg) else {
            // Decode failure is itself a catastrophic result
            results.push((q, -1.0, jpeg.len()));
            continue;
        };
        assert_eq!((w, h), (width, height), "Q{q}: dimension mismatch");

        let (score, _report) = compare_quality(pixels, &decoded, width as usize, height as usize);
        results.push((q, score, jpeg.len()));
    }

    results
}

// =============================================================================
// Tests
// =============================================================================

/// YCbCr 4:4:4 quality sweep: scores must be monotonically non-decreasing
/// (within tolerance) and never drop below minimum thresholds.
#[test]
fn test_ycbcr_444_quality_monotonic() {
    let (w, h) = (256, 256);
    let pixels = make_noise_patches(w, h);
    let qualities: Vec<u32> = (50..=99).step_by(5).collect();

    let results = quality_sweep(&pixels, w as u32, h as u32, &qualities, |p, w, h, q| {
        encode_ycbcr(p, w, h, q, ChromaSubsampling::None)
    });

    println!("\n=== YCbCr 4:4:4 Quality Sweep ===");
    let mut prev_score = 0.0f64;
    for &(q, score, size) in &results {
        let delta = score - prev_score;
        let flag = if score < 0.0 {
            "DECODE_FAIL"
        } else if delta < -3.0 && q > 50 {
            "REGRESSION"
        } else {
            ""
        };
        println!("  Q{q:3}: score={score:6.1}  size={size:6}  delta={delta:+.1}  {flag}");
        prev_score = score;
    }

    // Assert no decode failures
    for &(q, score, _) in &results {
        assert!(
            score >= 0.0,
            "Q{q}: decode failure (encode produced undecodable JPEG)"
        );
    }

    // Assert minimum quality thresholds (calibrated on 256x256 noise+patches)
    // zensim scores are 0-100 perceptual similarity; synthetic images score lower
    // than photos because they have high-frequency noise that JPEG attenuates
    for &(q, score, _) in &results {
        let min_score = match q {
            50..=59 => 45.0,
            60..=74 => 50.0,
            75..=89 => 55.0,
            90..=99 => 70.0,
            _ => 30.0,
        };
        assert!(
            score >= min_score,
            "Q{q}: score {score:.1} below minimum {min_score} — catastrophic quality"
        );
    }

    // Assert no large non-monotonic drops (>5 points)
    let mut prev = results[0].1;
    for &(q, score, _) in &results[1..] {
        let drop = prev - score;
        assert!(
            drop < 5.0,
            "Q{q}: score dropped {drop:.1} points from previous level — non-monotonic regression"
        );
        prev = score;
    }
}

/// YCbCr 4:2:0 quality sweep: same monotonicity checks, plus chroma-specific thresholds.
#[test]
fn test_ycbcr_420_quality_monotonic() {
    let (w, h) = (256, 256);
    let pixels = make_stress_blocks(w, h);
    let qualities: Vec<u32> = (50..=99).step_by(5).collect();

    let results = quality_sweep(&pixels, w as u32, h as u32, &qualities, |p, w, h, q| {
        encode_ycbcr(p, w, h, q, ChromaSubsampling::Quarter)
    });

    println!("\n=== YCbCr 4:2:0 Quality Sweep ===");
    for &(q, score, size) in &results {
        println!("  Q{q:3}: score={score:6.1}  size={size:6}");
    }

    for &(q, score, _) in &results {
        assert!(
            score >= 0.0,
            "Q{q}: decode failure"
        );
        // 4:2:0 has lower quality floor due to chroma subsampling
        // Stress blocks with high-frequency chroma alternation score low
        let min_score = match q {
            50..=59 => 40.0,
            60..=74 => 45.0,
            75..=89 => 50.0,
            90..=99 => 55.0,
            _ => 30.0,
        };
        assert!(
            score >= min_score,
            "Q{q}: score {score:.1} below minimum {min_score} — catastrophic quality"
        );
    }
}

/// 4:2:0 with auto_optimize: catches bug #1 (catastrophic quality at specific Q levels).
/// Tests EVERY quality level from Q70-Q99 (the auto_optimize range).
#[test]
fn test_420_auto_optimize_no_catastrophic() {
    let (w, h) = (256, 256);
    let pixels = make_stress_blocks(w, h);
    let qualities: Vec<u32> = (70..=99).collect(); // Every Q level in auto_optimize range

    let results = quality_sweep(&pixels, w as u32, h as u32, &qualities, |p, w, h, q| {
        encode_ycbcr_auto_optimize(p, w, h, q, ChromaSubsampling::Quarter)
    });

    println!("\n=== 4:2:0 auto_optimize Quality Sweep (Q70-Q99) ===");
    let mut catastrophic = Vec::new();
    for &(q, score, size) in &results {
        let flag = if score < 0.0 {
            "DECODE_FAIL"
        } else if score < 40.0 {
            "CATASTROPHIC"
        } else {
            ""
        };
        println!("  Q{q:3}: score={score:6.1}  size={size:6}  {flag}");
        if score < 40.0 {
            catastrophic.push((q, score));
        }
    }

    if !catastrophic.is_empty() {
        let list: Vec<String> = catastrophic
            .iter()
            .map(|(q, s)| format!("Q{q}={s:.1}"))
            .collect();
        panic!(
            "Catastrophic quality at {} quality level(s): {}. \
             This is likely the trellis lambda inversion bug (CLAUDE.md bug #1).",
            catastrophic.len(),
            list.join(", ")
        );
    }
}

/// 4:4:4 with auto_optimize as control: should never have catastrophic drops.
#[test]
fn test_444_auto_optimize_quality() {
    let (w, h) = (256, 256);
    let pixels = make_stress_blocks(w, h);
    let qualities: Vec<u32> = (70..=99).step_by(3).collect();

    let results = quality_sweep(&pixels, w as u32, h as u32, &qualities, |p, w, h, q| {
        encode_ycbcr_auto_optimize(p, w, h, q, ChromaSubsampling::None)
    });

    println!("\n=== 4:4:4 auto_optimize Quality Sweep ===");
    for &(q, score, size) in &results {
        println!("  Q{q:3}: score={score:6.1}  size={size:6}");
    }

    for &(q, score, _) in &results {
        assert!(score >= 0.0, "Q{q}: decode failure");
        assert!(
            score >= 55.0,
            "Q{q}: 4:4:4 auto_optimize score {score:.1} below 55 — unexpected quality drop"
        );
    }
}

/// XYB encode/decode roundtrip at all quality levels.
/// Catches: XYB Huffman corruption (bug #3), undecodable output.
#[test]
fn test_xyb_roundtrip_all_qualities() {
    let (w, h) = (128, 128);
    let pixels = make_noise_patches(w, h);
    let qualities: Vec<u32> = (10..=99).step_by(5).collect();

    println!("\n=== XYB Roundtrip Quality Sweep ===");
    let mut failures = Vec::new();
    let z = Zensim::new(ZensimProfile::latest());

    for &q in &qualities {
        let Some(jpeg) = encode_xyb(&pixels, w as u32, h as u32, q as f32) else {
            println!("  Q{q:3}: ENCODE_FAIL");
            failures.push((q, "encode_fail"));
            continue;
        };

        let Some((dw, dh, decoded)) = decode_rgb(&jpeg) else {
            println!("  Q{q:3}: DECODE_FAIL  size={}", jpeg.len());
            failures.push((q, "decode_fail"));
            continue;
        };

        assert_eq!((dw, dh), (w as u32, h as u32), "Q{q}: dimension mismatch");

        let stride = w * 3;
        let expected =
            StridedBytes::new(&pixels, w, h, stride, PixelFormat::Srgb8Rgb);
        let actual =
            StridedBytes::new(&decoded, w as usize, h as usize, stride, PixelFormat::Srgb8Rgb);

        let tolerance = RegressionTolerance::off_by_one()
            .with_max_delta(255)
            .with_max_pixels_different(1.0)
            .with_min_similarity(0.0);

        let report = check_regression(&z, &expected, &actual, &tolerance).unwrap();
        let score = report.score();

        let flag = if score < 30.0 { "LOW" } else { "" };
        println!("  Q{q:3}: score={score:6.1}  size={:6}  {flag}", jpeg.len());
    }

    if !failures.is_empty() {
        let list: Vec<String> = failures
            .iter()
            .map(|(q, reason)| format!("Q{q}:{reason}"))
            .collect();
        panic!(
            "XYB failures at {} quality level(s): {}",
            failures.len(),
            list.join(", ")
        );
    }
}

/// Decoder consistency: streaming vs scanline must produce identical output.
#[test]
fn test_decoder_path_consistency() {
    let (w, h) = (256, 256);
    let pixels = make_noise_patches(w, h);
    let qualities = [50, 75, 90, 95];
    let z = Zensim::new(ZensimProfile::latest());

    println!("\n=== Decoder Path Consistency ===");

    for q in qualities {
        let jpeg = encode_ycbcr(
            &pixels,
            w as u32,
            h as u32,
            q as f32,
            ChromaSubsampling::Quarter,
        );

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
            let output = imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgb8(output).expect("read_rows_rgb8");
            total_rows += rows;
        }

        // Compare with zensim
        let expected = StridedBytes::new(&stream_pixels, w, h, stride, PixelFormat::Srgb8Rgb);
        let actual = StridedBytes::new(&scanline_pixels, w, h, stride, PixelFormat::Srgb8Rgb);

        let tolerance = RegressionTolerance::exact();
        let report = check_regression(&z, &expected, &actual, &tolerance).unwrap();

        println!(
            "  Q{q:3}: score={:.1}  max_delta={:?}  differing={}  {}",
            report.score(),
            report.max_channel_delta(),
            report.pixels_differing(),
            if report.passed() { "IDENTICAL" } else { "DIFFER" }
        );

        assert!(
            report.passed(),
            "Q{q}: streaming vs scanline differ!\n{report}"
        );
    }
}

/// Regression guard: ensure zensim scoring itself is consistent.
/// A known image pair should produce a known score range.
#[test]
fn test_zensim_scoring_sanity() {
    let (w, h) = (64, 64);
    let original = make_noise_patches(w, h);

    // Identical images must score 100
    let (score_identical, _) = compare_quality(&original, &original, w, h);
    assert!(
        (score_identical - 100.0).abs() < 0.1,
        "Identical images scored {score_identical:.1}, expected 100.0"
    );

    // Slightly modified: change 1% of bytes by 1 level.
    // zensim is perceptually weighted — even sparse changes score lower than
    // you'd expect on small images because each pixel represents more visual area.
    let mut modified = original.clone();
    for i in (0..modified.len()).step_by(100) {
        modified[i] = modified[i].wrapping_add(1);
    }
    let (score_slight, _) = compare_quality(&original, &modified, w, h);
    assert!(
        score_slight > 50.0 && score_slight < 100.0,
        "Slightly modified scored {score_slight:.1}, expected 50-100"
    );

    // Heavily modified must score low
    let mut heavy = original.clone();
    for i in 0..heavy.len() {
        heavy[i] = 255 - heavy[i];
    }
    let (score_inverted, _) = compare_quality(&original, &heavy, w, h);
    assert!(
        score_inverted < 60.0,
        "Inverted image scored {score_inverted:.1}, expected <60"
    );

    println!("zensim sanity: identical={score_identical:.1}, slight={score_slight:.1}, inverted={score_inverted:.1}");
}
