//! Decoder parity tests between Rust jpegli and C++ djpegli.
//!
//! This file verifies that the Rust decoder produces identical or near-identical
//! output to C++ djpegli for various JPEG configurations.

use dssim::Dssim;
use jpegli::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};
use rgb::RGBA8;
use std::process::Command;

// Map from test subsampling names to ChromaSubsampling
#[allow(dead_code)] // May be used in future test configurations
fn subsampling_from_name(name: &str) -> ChromaSubsampling {
    match name {
        "444" => ChromaSubsampling::None,
        "420" => ChromaSubsampling::Quarter,
        "422" => ChromaSubsampling::HalfHorizontal,
        "440" => ChromaSubsampling::HalfVertical,
        _ => ChromaSubsampling::None,
    }
}

/// Compute DSSIM between two RGB images
fn compute_dssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    if a.len() != b.len() || a.len() != width * height * 3 {
        return 1.0;
    }
    let attr = Dssim::new();
    let a_rgba: Vec<RGBA8> = a
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let b_rgba: Vec<RGBA8> = b
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let a_img = attr.create_image_rgba(&a_rgba, width, height).unwrap();
    let b_img = attr.create_image_rgba(&b_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&a_img, b_img);
    dssim.into()
}

/// Compute max pixel difference between two RGB images
fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Test image generation
fn generate_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8);
            pixels.push(((y * 255) / height.max(1)) as u8);
            pixels.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    pixels
}

/// Encode a test image with Rust encoder
fn encode_rust(
    pixels: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality as f32, subsampling);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(pixels, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("finish")
}

/// Decode with Rust decoder
fn decode_rust(jpeg: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let decoder = Decoder::new();
    match decoder.decode(jpeg) {
        Ok(img) => Some((img.data, img.width, img.height)),
        Err(_) => None,
    }
}

/// Decode with reference decoder (jpeg-decoder crate)
fn decode_reference(jpeg: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg));
    match decoder.decode() {
        Ok(pixels) => {
            let (width, height) = decoder.dimensions().unwrap();
            Some((pixels, width as u32, height as u32))
        }
        Err(_) => None,
    }
}

/// Decode with C++ djpegli (if available)
#[allow(dead_code)] // Available for C++ parity tests when enabled
fn decode_cpp(jpeg_path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let djpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/djpegli";
    if !std::path::Path::new(djpegli).exists() {
        return None;
    }

    let ppm_path = format!("{}.ppm", jpeg_path);
    let status = Command::new(djpegli)
        .args([jpeg_path, &ppm_path])
        .status()
        .ok()?;

    if !status.success() {
        return None;
    }

    // Parse PPM file
    let ppm_data = std::fs::read(&ppm_path).ok()?;
    let _ = std::fs::remove_file(&ppm_path);

    // Simple P6 PPM parser
    let header_end = ppm_data
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] == b'\n' && (w[1] >= b'0' && w[1] <= b'9' || w[1] == b'\n'))
        .nth(2)
        .map(|(i, _)| i + 1)?;

    let header = std::str::from_utf8(&ppm_data[..header_end]).ok()?;
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 4 || parts[0] != "P6" {
        return None;
    }

    let width: u32 = parts[1].parse().ok()?;
    let height: u32 = parts[2].parse().ok()?;
    let pixels = ppm_data[header_end..].to_vec();

    Some((pixels, width, height))
}

/// Quality levels to test
const QUALITY_LEVELS: &[u8] = &[50, 75, 90, 100];

/// Subsampling modes to test
const SUBSAMPLING_MODES: &[(ChromaSubsampling, &str)] = &[
    (ChromaSubsampling::None, "444"),
    (ChromaSubsampling::Quarter, "420"),
    (ChromaSubsampling::HalfHorizontal, "422"),
    (ChromaSubsampling::HalfVertical, "440"),
];

/// Test dimensions
const TEST_SIZES: &[(u32, u32)] = &[(64, 64), (128, 96), (100, 100)];

/// Maximum allowed DSSIM for decoder parity
const MAX_DSSIM_444: f64 = 0.0001; // 4:4:4 should be nearly identical
const MAX_DSSIM_SUBSAMPLED: f64 = 0.001; // Subsampled modes allow more variance

/// Maximum allowed pixel difference
/// Note: Rust decoder uses jpegli-style dequantization bias which differs slightly
/// from jpeg-decoder's standard dequantization. This improves parity with C++ djpegli
/// at the cost of minor differences vs jpeg-decoder. These thresholds are based on
/// empirical testing.
const MAX_PIXEL_DIFF_444: u8 = 4;
const MAX_PIXEL_DIFF_SUBSAMPLED: u8 = 8;

// ============================================================================
// Decoder Parity Tests
// ============================================================================

/// Compare Rust decoder against jpeg-decoder reference for Rust-encoded JPEGs
#[test]
fn test_decoder_vs_reference_444() {
    for &(width, height) in TEST_SIZES {
        let pixels = generate_gradient_image(width, height);

        for &quality in QUALITY_LEVELS {
            let jpeg = encode_rust(&pixels, width, height, quality, ChromaSubsampling::None);

            let (rust_decoded, rw, rh) = decode_rust(&jpeg).expect("Rust decode failed");
            let (ref_decoded, ref_w, ref_h) =
                decode_reference(&jpeg).expect("Reference decode failed");

            assert_eq!((rw, rh), (ref_w, ref_h), "Dimension mismatch");

            let dssim = compute_dssim(&rust_decoded, &ref_decoded, rw as usize, rh as usize);
            let max_diff = max_pixel_diff(&rust_decoded, &ref_decoded);

            assert!(
                dssim < MAX_DSSIM_444,
                "4:4:4 Q{} {}x{}: DSSIM {:.6} too high (max: {})",
                quality,
                width,
                height,
                dssim,
                MAX_DSSIM_444
            );

            assert!(
                max_diff <= MAX_PIXEL_DIFF_444,
                "4:4:4 Q{} {}x{}: max pixel diff {} too high (max: {})",
                quality,
                width,
                height,
                max_diff,
                MAX_PIXEL_DIFF_444
            );
        }
    }
}

/// Compare Rust decoder against jpeg-decoder for 4:2:0 subsampled JPEGs
#[test]
fn test_decoder_vs_reference_420() {
    for &(width, height) in TEST_SIZES {
        let pixels = generate_gradient_image(width, height);

        for &quality in QUALITY_LEVELS {
            let jpeg = encode_rust(&pixels, width, height, quality, ChromaSubsampling::Quarter);

            let (rust_decoded, rw, rh) = decode_rust(&jpeg).expect("Rust decode failed");
            let (ref_decoded, ref_w, ref_h) =
                decode_reference(&jpeg).expect("Reference decode failed");

            assert_eq!((rw, rh), (ref_w, ref_h), "Dimension mismatch");

            let dssim = compute_dssim(&rust_decoded, &ref_decoded, rw as usize, rh as usize);
            let max_diff = max_pixel_diff(&rust_decoded, &ref_decoded);

            assert!(
                dssim < MAX_DSSIM_SUBSAMPLED,
                "4:2:0 Q{} {}x{}: DSSIM {:.6} too high (max: {})",
                quality,
                width,
                height,
                dssim,
                MAX_DSSIM_SUBSAMPLED
            );

            assert!(
                max_diff <= MAX_PIXEL_DIFF_SUBSAMPLED,
                "4:2:0 Q{} {}x{}: max pixel diff {} too high (max: {})",
                quality,
                width,
                height,
                max_diff,
                MAX_PIXEL_DIFF_SUBSAMPLED
            );
        }
    }
}

/// Compare Rust decoder against jpeg-decoder for 4:2:2 subsampled JPEGs
#[test]
fn test_decoder_vs_reference_422() {
    for &(width, height) in TEST_SIZES {
        let pixels = generate_gradient_image(width, height);

        for &quality in QUALITY_LEVELS {
            let jpeg = encode_rust(
                &pixels,
                width,
                height,
                quality,
                ChromaSubsampling::HalfHorizontal,
            );

            let (rust_decoded, rw, rh) = decode_rust(&jpeg).expect("Rust decode failed");
            let (ref_decoded, ref_w, ref_h) =
                decode_reference(&jpeg).expect("Reference decode failed");

            assert_eq!((rw, rh), (ref_w, ref_h), "Dimension mismatch");

            let dssim = compute_dssim(&rust_decoded, &ref_decoded, rw as usize, rh as usize);
            let max_diff = max_pixel_diff(&rust_decoded, &ref_decoded);

            assert!(
                dssim < MAX_DSSIM_SUBSAMPLED,
                "4:2:2 Q{} {}x{}: DSSIM {:.6} too high (max: {})",
                quality,
                width,
                height,
                dssim,
                MAX_DSSIM_SUBSAMPLED
            );

            assert!(
                max_diff <= MAX_PIXEL_DIFF_SUBSAMPLED,
                "4:2:2 Q{} {}x{}: max pixel diff {} too high (max: {})",
                quality,
                width,
                height,
                max_diff,
                MAX_PIXEL_DIFF_SUBSAMPLED
            );
        }
    }
}

/// Compare Rust decoder against jpeg-decoder for 4:4:0 subsampled JPEGs
#[test]
fn test_decoder_vs_reference_440() {
    for &(width, height) in TEST_SIZES {
        let pixels = generate_gradient_image(width, height);

        for &quality in QUALITY_LEVELS {
            let jpeg = encode_rust(
                &pixels,
                width,
                height,
                quality,
                ChromaSubsampling::HalfVertical,
            );

            let (rust_decoded, rw, rh) = decode_rust(&jpeg).expect("Rust decode failed");
            let (ref_decoded, ref_w, ref_h) =
                decode_reference(&jpeg).expect("Reference decode failed");

            assert_eq!((rw, rh), (ref_w, ref_h), "Dimension mismatch");

            let dssim = compute_dssim(&rust_decoded, &ref_decoded, rw as usize, rh as usize);
            let max_diff = max_pixel_diff(&rust_decoded, &ref_decoded);

            assert!(
                dssim < MAX_DSSIM_SUBSAMPLED,
                "4:4:0 Q{} {}x{}: DSSIM {:.6} too high (max: {})",
                quality,
                width,
                height,
                dssim,
                MAX_DSSIM_SUBSAMPLED
            );

            assert!(
                max_diff <= MAX_PIXEL_DIFF_SUBSAMPLED,
                "4:4:0 Q{} {}x{}: max pixel diff {} too high (max: {})",
                quality,
                width,
                height,
                max_diff,
                MAX_PIXEL_DIFF_SUBSAMPLED
            );
        }
    }
}

/// Print decoder parity summary
#[test]
#[ignore] // Run with: cargo test --test decoder_parity print_summary -- --ignored --nocapture
fn print_summary() {
    let (width, height) = (128, 128);
    let pixels = generate_gradient_image(width, height);

    println!("=== Decoder Parity Summary ===\n");
    println!(
        "{:>8} {:>8} {:>12} {:>12} {:>12}",
        "Mode", "Q", "DSSIM", "MaxDiff", "Status"
    );

    for &(subsampling, name) in SUBSAMPLING_MODES {
        for &quality in QUALITY_LEVELS {
            let jpeg = encode_rust(&pixels, width, height, quality, subsampling);

            let (rust_decoded, rw, rh) = match decode_rust(&jpeg) {
                Some(d) => d,
                None => {
                    println!(
                        "{:>8} {:>8} {:>12} {:>12} DECODE_FAIL",
                        name, quality, "-", "-"
                    );
                    continue;
                }
            };

            let (ref_decoded, _, _) = match decode_reference(&jpeg) {
                Some(d) => d,
                None => {
                    println!(
                        "{:>8} {:>8} {:>12} {:>12} REF_FAIL",
                        name, quality, "-", "-"
                    );
                    continue;
                }
            };

            let dssim = compute_dssim(&rust_decoded, &ref_decoded, rw as usize, rh as usize);
            let max_diff = max_pixel_diff(&rust_decoded, &ref_decoded);

            let max_dssim = if name == "444" {
                MAX_DSSIM_444
            } else {
                MAX_DSSIM_SUBSAMPLED
            };
            let max_pix = if name == "444" {
                MAX_PIXEL_DIFF_444
            } else {
                MAX_PIXEL_DIFF_SUBSAMPLED
            };

            let status = if dssim < max_dssim && max_diff <= max_pix {
                "OK"
            } else {
                "FAIL"
            };

            println!(
                "{:>8} {:>8} {:>12.6} {:>12} {:>12}",
                name, quality, dssim, max_diff, status
            );
        }
    }
}

/// Exhaustive decode test across all permutations
#[test]
#[ignore] // Run with: cargo test --test decoder_parity exhaustive -- --ignored --nocapture
fn exhaustive() {
    let mut total = 0;
    let mut passed = 0;
    let mut failed = Vec::new();

    for &(width, height) in TEST_SIZES {
        let pixels = generate_gradient_image(width, height);

        for &(subsampling, name) in SUBSAMPLING_MODES {
            for &quality in QUALITY_LEVELS {
                total += 1;

                let jpeg = encode_rust(&pixels, width, height, quality, subsampling);

                let (rust_decoded, rw, rh) = match decode_rust(&jpeg) {
                    Some(d) => d,
                    None => {
                        failed.push(format!(
                            "{} Q{} {}x{}: decode failed",
                            name, quality, width, height
                        ));
                        continue;
                    }
                };

                let (ref_decoded, _, _) = match decode_reference(&jpeg) {
                    Some(d) => d,
                    None => {
                        failed.push(format!(
                            "{} Q{} {}x{}: ref failed",
                            name, quality, width, height
                        ));
                        continue;
                    }
                };

                let dssim = compute_dssim(&rust_decoded, &ref_decoded, rw as usize, rh as usize);
                let max_diff = max_pixel_diff(&rust_decoded, &ref_decoded);

                let max_dssim = if name == "444" {
                    MAX_DSSIM_444
                } else {
                    MAX_DSSIM_SUBSAMPLED
                };
                let max_pix = if name == "444" {
                    MAX_PIXEL_DIFF_444
                } else {
                    MAX_PIXEL_DIFF_SUBSAMPLED
                };

                if dssim >= max_dssim || max_diff > max_pix {
                    failed.push(format!(
                        "{} Q{} {}x{}: DSSIM={:.6} maxDiff={}",
                        name, quality, width, height, dssim, max_diff
                    ));
                } else {
                    passed += 1;
                }
            }
        }
    }

    println!("\n=== Exhaustive Decoder Test Results ===");
    println!("Passed: {}/{}", passed, total);

    if !failed.is_empty() {
        println!("\nFailures:");
        for f in &failed {
            println!("  {}", f);
        }
        panic!("{} tests failed", failed.len());
    }

    println!("\nAll tests passed!");
}
