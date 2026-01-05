//! Hard mode synthetic test image for progressive + subsampling
//!
//! Adversarial test patterns to expose encoder bugs

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;

/// Generate adversarial test image with:
/// - Odd dimensions (not multiples of 8 or 16)
/// - High-frequency checkerboard regions
/// - Sharp color boundaries
/// - Random noise patches
/// - Single-pixel features
fn generate_hard_image(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    let mut rng = seed;

    let next_rand = |r: &mut u64| -> u8 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*r >> 33) as u8
    };

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;

            // Divide image into regions with different patterns
            let region_x = x * 4 / width;
            let region_y = y * 4 / height;

            let (r, g, b) = match (region_x, region_y) {
                // Top-left: high-frequency checkerboard (1px)
                (0, 0) => {
                    if (x + y) % 2 == 0 {
                        (255, 0, 0)
                    } else {
                        (0, 255, 0)
                    }
                }
                // Top-right: 2x2 checkerboard with blue/yellow
                (1, 0) => {
                    if ((x / 2) + (y / 2)) % 2 == 0 {
                        (0, 0, 255)
                    } else {
                        (255, 255, 0)
                    }
                }
                // Second row left: random noise
                (0, 1) => (next_rand(&mut rng), next_rand(&mut rng), next_rand(&mut rng)),
                // Second row right: horizontal stripes (1px)
                (1, 1) => {
                    if y % 2 == 0 {
                        (255, 128, 0)
                    } else {
                        (0, 128, 255)
                    }
                }
                // Third row: vertical stripes with varying width
                (0, 2) => {
                    let stripe = x % 7;
                    if stripe < 3 {
                        (255, 0, 255)
                    } else {
                        (0, 255, 255)
                    }
                }
                (1, 2) => {
                    // Diagonal pattern
                    if (x + y) % 3 == 0 {
                        (200, 50, 50)
                    } else if (x + y) % 3 == 1 {
                        (50, 200, 50)
                    } else {
                        (50, 50, 200)
                    }
                }
                // Bottom row: gradients with sharp transitions
                (0, 3) => {
                    let v = ((x * 255 / width.max(1)) as u8).wrapping_add(
                        if x % 8 == 0 { 128 } else { 0 }
                    );
                    (v, 255 - v, 128)
                }
                (1, 3) => {
                    // Sparse single-pixel features on solid background
                    if x % 13 == 7 && y % 11 == 5 {
                        (255, 255, 255)
                    } else if x % 17 == 3 && y % 19 == 7 {
                        (0, 0, 0)
                    } else {
                        (128, 128, 128)
                    }
                }
                // Additional regions for larger images
                (2, _) => {
                    // Color ramps with discontinuities
                    let phase = (x * 3 + y * 5) % 256;
                    if phase < 85 {
                        (phase as u8 * 3, 0, 0)
                    } else if phase < 170 {
                        (0, (phase - 85) as u8 * 3, 0)
                    } else {
                        (0, 0, (phase - 170) as u8 * 3)
                    }
                }
                (3, _) => {
                    // High-contrast edges at odd intervals
                    if x % 5 == 0 || y % 7 == 0 {
                        (0, 0, 0)
                    } else {
                        (255, 255, 255)
                    }
                }
                _ => {
                    // Fallback: more noise
                    (next_rand(&mut rng), next_rand(&mut rng), next_rand(&mut rng))
                }
            };

            data[idx] = r;
            data[idx + 1] = g;
            data[idx + 2] = b;
        }
    }

    data
}

fn decode_mozjpeg(data: &[u8]) -> Vec<u8> {
    let d = mozjpeg::Decompress::new_mem(data).unwrap();
    let mut dec = d.rgb().unwrap();
    let mut buf = vec![0u8; dec.width() * dec.height() * 3];
    #[allow(deprecated)]
    let _ = dec.read_scanlines_into::<u8>(&mut buf);
    buf
}

fn check_quality(name: &str, original: &[u8], decoded: &[u8]) -> (u8, f64) {
    if decoded.len() != original.len() {
        println!("{}: SIZE MISMATCH {} vs {}", name, decoded.len(), original.len());
        return (255, 255.0);
    }

    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    for (&o, &d) in original.iter().zip(decoded.iter()) {
        let diff = (o as i16 - d as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
        sum_diff += diff as u64;
    }
    let avg_diff = sum_diff as f64 / original.len() as f64;
    (max_diff, avg_diff)
}

fn main() {
    // Test multiple odd dimensions
    let test_sizes = [
        (63, 63),   // Just under 64 (8x8 MCU boundary)
        (65, 65),   // Just over 64
        (47, 53),   // Prime-ish dimensions
        (100, 75),  // Non-square, not power of 2
        (127, 127), // Just under 128
        (33, 17),   // Small odd dimensions
    ];

    let subsamplings = [
        ("444", Subsampling::S444),
        ("422", Subsampling::S422),
        ("420", Subsampling::S420),
        ("440", Subsampling::S440),
    ];

    // First test baseline to confirm it works
    println!("=== BASELINE (Sequential) Sanity Check ===\n");
    {
        let (width, height) = (63, 63);
        let original = generate_hard_image(width, height, 12345);

        for (sub_name, sub) in &subsamplings {
            let rust_jpeg = Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .pixel_format(PixelFormat::Rgb)
                .mode(JpegMode::Baseline)  // Sequential!
                .subsampling(*sub)
                .optimize_huffman(true)
                .jpegli_quality(Quality::from_quality(85.0))
                .encode(&original)
                .unwrap();

            let decoded = decode_mozjpeg(&rust_jpeg);
            let (max_diff, avg_diff) = check_quality("baseline", &original, &decoded);
            println!("Baseline {}: size={}, max_diff={}, avg_diff={:.2}",
                     sub_name, rust_jpeg.len(), max_diff, avg_diff);
        }
    }
    println!();

    println!("=== HARD MODE: Progressive + Subsampling Test ===\n");
    println!("{:<12} {:<6} {:>8} {:>8} {:>8} {:>8}",
             "Size", "Sub", "RustSz", "CppSz", "MaxDiff", "AvgDiff");
    println!("{:-<60}", "");

    let cjpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";
    let mut failures = Vec::new();

    for (width, height) in test_sizes {
        let original = generate_hard_image(width, height, 12345);

        // Save PNG for C++ encoder
        let png_path = format!("/tmp/hard_{}x{}.png", width, height);
        {
            let file = fs::File::create(&png_path).unwrap();
            let mut encoder = png::Encoder::new(file, width as u32, height as u32);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&original).unwrap();
        }

        for (sub_name, sub) in &subsamplings {
            // Rust encode
            let rust_result = Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .pixel_format(PixelFormat::Rgb)
                .mode(JpegMode::Progressive)
                .subsampling(*sub)
                .optimize_huffman(true)
                .jpegli_quality(Quality::from_quality(85.0))
                .encode(&original);

            let rust_jpeg = match rust_result {
                Ok(j) => j,
                Err(e) => {
                    println!("{:<12} {:<6} ENCODE ERROR: {}",
                             format!("{}x{}", width, height), sub_name, e);
                    continue;
                }
            };

            // C++ encode
            let cpp_path = format!("/tmp/hard_{}x{}_{}_cpp.jpg", width, height, sub_name);
            let cpp_sub = format!("--chroma_subsampling={}", sub_name);
            let _ = std::process::Command::new(cjpegli)
                .args([&png_path, &cpp_path, "-q", "85", &cpp_sub, "--progressive_level=2"])
                .output();

            let cpp_size = fs::read(&cpp_path).map(|d| d.len()).unwrap_or(0);

            // Decode and check quality
            let rust_decoded = decode_mozjpeg(&rust_jpeg);
            let (max_diff, avg_diff) = check_quality("rust", &original, &rust_decoded);

            let status = if max_diff > 50 { "FAIL" } else { "ok" };
            println!("{:<12} {:<6} {:>8} {:>8} {:>8} {:>8.2}  {}",
                     format!("{}x{}", width, height),
                     sub_name,
                     rust_jpeg.len(),
                     cpp_size,
                     max_diff,
                     avg_diff,
                     status);

            if max_diff > 50 {
                failures.push(format!("{}x{} {}", width, height, sub_name));
                // Save for inspection
                let rust_path = format!("/tmp/hard_{}x{}_{}_rust.jpg", width, height, sub_name);
                fs::write(&rust_path, &rust_jpeg).unwrap();
            }
        }
    }

    println!("\n{:-<60}", "");
    if failures.is_empty() {
        println!("All tests passed!");
    } else {
        println!("FAILURES ({}):", failures.len());
        for f in &failures {
            println!("  - {}", f);
        }
    }
}
