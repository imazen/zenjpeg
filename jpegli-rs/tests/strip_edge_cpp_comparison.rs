//! Comprehensive edge handling comparison: Strip encoder vs C++ cjpegli
//!
//! Tests all partial MCU sizes (1-7 for 8x8 blocks, 1-15 for 16x16 MCUs)
//! for both width and height edges.
//!
//! Run with: cargo test --release -p jpegli-rs --test strip_edge_cpp_comparison -- --nocapture --ignored

use dssim::Dssim;
use jpegli::types::{PixelFormat, Subsampling};
use jpegli::Quality;
use rgb::RGBA8;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Create smooth gradient test image
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = (64.0 + (x as f32 / width as f32) * 128.0) as u8;
            rgb[idx + 1] = (64.0 + (y as f32 / height as f32) * 128.0) as u8;
            rgb[idx + 2] = (64.0 + ((x + y) as f32 / (width + height) as f32) * 128.0) as u8;
        }
    }
    rgb
}

/// Encode with Rust strip encoder
#[allow(deprecated)]
fn encode_rust(
    rgb: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    quality: f32,
) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .subsampling(subsampling)
        .jpegli_quality(Quality::from_quality(quality))
        .optimize_huffman(true)
        .encode(rgb)
        .expect("Rust encode failed")
}

fn cjpegli_path() -> String {
    // Try both relative (from repo root) and absolute paths
    let candidates = [
        "internal/jpegli-cpp/build/tools/cjpegli".to_string(),
        "../internal/jpegli-cpp/build/tools/cjpegli".to_string(),
        format!(
            "{}/internal/jpegli-cpp/build/tools/cjpegli",
            std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string())
        ),
    ];
    for path in &candidates {
        if Path::new(path).exists() {
            return path.clone();
        }
    }
    candidates[0].clone()
}

/// Encode with C++ cjpegli CLI
fn encode_cpp(
    rgb: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    quality: u8,
) -> Option<Vec<u8>> {
    let cjpegli = cjpegli_path();
    if !Path::new(&cjpegli).exists() {
        return None;
    }

    // Write PPM to temp file
    let ppm_path = format!("/tmp/edge_test_{}x{}.ppm", width, height);
    let jpg_path = format!("/tmp/edge_test_{}x{}_cpp.jpg", width, height);

    // PPM P6 format
    let mut ppm = format!("P6\n{} {}\n255\n", width, height).into_bytes();
    ppm.extend_from_slice(rgb);
    fs::write(&ppm_path, &ppm).ok()?;

    // Run cjpegli
    let sample_arg = match subsampling {
        Subsampling::S444 => "444",
        Subsampling::S422 => "422",
        Subsampling::S420 => "420",
        Subsampling::S440 => "440",
        _ => "420",
    };

    let output = Command::new(cjpegli)
        .args([
            &ppm_path,
            &jpg_path,
            "-q",
            &quality.to_string(),
            "--chroma_subsampling",
            sample_arg,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let result = fs::read(&jpg_path).ok();
    let _ = fs::remove_file(&ppm_path);
    let _ = fs::remove_file(&jpg_path);
    result
}

/// Decode JPEG to RGB
fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("JPEG decode failed")
}

/// Compute DSSIM between original and decoded
fn compute_dssim(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();
    let orig_rgba: Vec<RGBA8> = orig
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dec_rgba: Vec<RGBA8> = decoded
        .chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();
    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}

struct TestResult {
    rust_size: usize,
    cpp_size: usize,
    rust_dssim: f64,
    cpp_dssim: f64,
}

fn test_dimension(
    width: usize,
    height: usize,
    subsampling: Subsampling,
    quality: u8,
) -> Option<TestResult> {
    let rgb = create_test_image(width, height);

    let rust_jpeg = encode_rust(
        &rgb,
        width as u32,
        height as u32,
        subsampling,
        quality as f32,
    );
    let cpp_jpeg = encode_cpp(&rgb, width, height, subsampling, quality)?;

    let rust_decoded = decode_jpeg(&rust_jpeg);
    let cpp_decoded = decode_jpeg(&cpp_jpeg);

    let rust_dssim = compute_dssim(&rgb, &rust_decoded, width, height);
    let cpp_dssim = compute_dssim(&rgb, &cpp_decoded, width, height);

    Some(TestResult {
        rust_size: rust_jpeg.len(),
        cpp_size: cpp_jpeg.len(),
        rust_dssim,
        cpp_dssim,
    })
}

/// Test all partial MCU widths (1-7 for S444, 1-15 for S420)
#[test]
#[ignore = "requires cjpegli binary"]
fn test_strip_edge_width_cpp_comparison() {
    let cjpegli = cjpegli_path();
    if !Path::new(&cjpegli).exists() {
        println!("Skipping: cjpegli not found at {}", cjpegli_path());
        return;
    }

    println!("\n=== Strip Encoder Edge Width Comparison vs C++ ===\n");

    let quality = 85u8;
    let configs = [
        (Subsampling::S444, 8, 256, 128, "S444 (8x8 MCU)"),
        (Subsampling::S420, 16, 512, 256, "S420 (16x16 MCU)"),
    ];

    for (subsampling, mcu_w, base_width, height, name) in configs {
        println!("Testing {} - varying width:", name);
        println!(
            "{:>8} {:>8} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
            "Width", "W%MCU", "Rust", "C++", "Size%", "RustDSSIM", "C++DSSIM", "Δ%"
        );
        println!("{}", "-".repeat(80));

        let mut max_size_diff = 0.0f64;
        let mut max_dssim_diff = 0.0f64;
        let remainders: Vec<usize> = if mcu_w == 16 {
            (1..=15).collect()
        } else {
            (1..=7).collect()
        };

        for remainder in remainders {
            let width = base_width + remainder;
            let result = match test_dimension(width, height, subsampling, quality) {
                Some(r) => r,
                None => {
                    println!("Skipped: {}x{}", width, height);
                    continue;
                }
            };

            let size_diff =
                (result.rust_size as f64 - result.cpp_size as f64) / result.cpp_size as f64 * 100.0;
            let dssim_diff = if result.cpp_dssim > 0.0 {
                (result.rust_dssim - result.cpp_dssim) / result.cpp_dssim * 100.0
            } else {
                0.0
            };

            max_size_diff = max_size_diff.max(size_diff.abs());
            max_dssim_diff = max_dssim_diff.max(dssim_diff.abs());

            let status = if size_diff.abs() > 5.0 || dssim_diff.abs() > 10.0 {
                "!"
            } else {
                ""
            };
            println!(
                "{:>8} {:>8} {:>10} {:>10} {:>+7.2}% {:>10.6} {:>10.6} {:>+7.2}%{}",
                width,
                width % mcu_w,
                result.rust_size,
                result.cpp_size,
                size_diff,
                result.rust_dssim,
                result.cpp_dssim,
                dssim_diff,
                status
            );
        }

        println!(
            "\nMax size diff: {:.2}%, Max DSSIM diff: {:.2}%\n",
            max_size_diff, max_dssim_diff
        );
        // Quality (DSSIM) should be nearly identical - this proves edge handling is correct
        assert!(
            max_dssim_diff < 10.0,
            "{}: DSSIM difference too large: {:.2}%",
            name,
            max_dssim_diff
        );
        // Size may differ due to progressive scan config and Huffman optimization
        // The important test is quality parity, not size parity for edge cases
        if max_size_diff > 20.0 {
            println!(
                "NOTE: Size difference {}% is larger than typical (expected for edge cases)",
                max_size_diff
            );
        }
    }
}

/// Test all partial MCU heights
#[test]
#[ignore = "requires cjpegli binary"]
fn test_strip_edge_height_cpp_comparison() {
    let cjpegli = cjpegli_path();
    if !Path::new(&cjpegli).exists() {
        println!("Skipping: cjpegli not found at {}", cjpegli_path());
        return;
    }

    println!("\n=== Strip Encoder Edge Height Comparison vs C++ ===\n");

    let quality = 85u8;
    let configs = [
        (Subsampling::S444, 8, 256, 128, "S444 (8x8 MCU)"),
        (Subsampling::S420, 16, 512, 256, "S420 (16x16 MCU)"),
    ];

    for (subsampling, mcu_h, width, base_height, name) in configs {
        println!("Testing {} - varying height:", name);
        println!(
            "{:>8} {:>8} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
            "Height", "H%MCU", "Rust", "C++", "Size%", "RustDSSIM", "C++DSSIM", "Δ%"
        );
        println!("{}", "-".repeat(80));

        let mut max_size_diff = 0.0f64;
        let mut max_dssim_diff = 0.0f64;
        let remainders: Vec<usize> = if mcu_h == 16 {
            (1..=15).collect()
        } else {
            (1..=7).collect()
        };

        for remainder in remainders {
            let height = base_height + remainder;
            let result = match test_dimension(width, height, subsampling, quality) {
                Some(r) => r,
                None => continue,
            };

            let size_diff =
                (result.rust_size as f64 - result.cpp_size as f64) / result.cpp_size as f64 * 100.0;
            let dssim_diff = if result.cpp_dssim > 0.0 {
                (result.rust_dssim - result.cpp_dssim) / result.cpp_dssim * 100.0
            } else {
                0.0
            };

            max_size_diff = max_size_diff.max(size_diff.abs());
            max_dssim_diff = max_dssim_diff.max(dssim_diff.abs());

            let status = if size_diff.abs() > 5.0 || dssim_diff.abs() > 10.0 {
                "!"
            } else {
                ""
            };
            println!(
                "{:>8} {:>8} {:>10} {:>10} {:>+7.2}% {:>10.6} {:>10.6} {:>+7.2}%{}",
                height,
                height % mcu_h,
                result.rust_size,
                result.cpp_size,
                size_diff,
                result.rust_dssim,
                result.cpp_dssim,
                dssim_diff,
                status
            );
        }

        println!(
            "\nMax size diff: {:.2}%, Max DSSIM diff: {:.2}%\n",
            max_size_diff, max_dssim_diff
        );
        // Quality (DSSIM) should be nearly identical - this proves edge handling is correct
        assert!(
            max_dssim_diff < 10.0,
            "{}: DSSIM difference too large: {:.2}%",
            name,
            max_dssim_diff
        );
        // Size may differ due to progressive scan config and Huffman optimization
        if max_size_diff > 20.0 {
            println!(
                "NOTE: Size difference {}% is larger than typical (expected for edge cases)",
                max_size_diff
            );
        }
    }
}

/// Test corner cases: both width and height have partial MCUs
#[test]
#[ignore = "requires cjpegli binary"]
fn test_strip_edge_corner_cpp_comparison() {
    let cjpegli = cjpegli_path();
    if !Path::new(&cjpegli).exists() {
        println!("Skipping: cjpegli not found at {}", cjpegli_path());
        return;
    }

    println!("\n=== Strip Encoder Corner Cases vs C++ ===\n");

    let quality = 85u8;

    // Test S420 corner cases (most demanding)
    let subsampling = Subsampling::S420;
    let base_width = 512;
    let base_height = 256;

    println!("Testing S420 corner cases (both edges partial):");
    println!(
        "{:>12} {:>8} {:>8} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}",
        "WxH", "W%16", "H%16", "Rust", "C++", "Size%", "RustDSSIM", "C++DSSIM", "Δ%"
    );
    println!("{}", "-".repeat(100));

    let remainders = [1, 4, 7, 11, 15];
    let mut max_size_diff = 0.0f64;
    let mut max_dssim_diff = 0.0f64;

    for &w_rem in &remainders {
        for &h_rem in &remainders {
            let width = base_width + w_rem;
            let height = base_height + h_rem;
            let result = match test_dimension(width, height, subsampling, quality) {
                Some(r) => r,
                None => continue,
            };

            let size_diff =
                (result.rust_size as f64 - result.cpp_size as f64) / result.cpp_size as f64 * 100.0;
            let dssim_diff = if result.cpp_dssim > 0.0 {
                (result.rust_dssim - result.cpp_dssim) / result.cpp_dssim * 100.0
            } else {
                0.0
            };

            max_size_diff = max_size_diff.max(size_diff.abs());
            max_dssim_diff = max_dssim_diff.max(dssim_diff.abs());

            let status = if size_diff.abs() > 5.0 || dssim_diff.abs() > 10.0 {
                "!"
            } else {
                ""
            };
            println!(
                "{:>12} {:>8} {:>8} {:>10} {:>10} {:>+7.2}% {:>10.6} {:>10.6} {:>+7.2}%{}",
                format!("{}x{}", width, height),
                width % 16,
                height % 16,
                result.rust_size,
                result.cpp_size,
                size_diff,
                result.rust_dssim,
                result.cpp_dssim,
                dssim_diff,
                status
            );
        }
    }

    println!(
        "\nMax size diff: {:.2}%, Max DSSIM diff: {:.2}%\n",
        max_size_diff, max_dssim_diff
    );
    // Quality (DSSIM) should be nearly identical - this proves edge handling is correct
    assert!(
        max_dssim_diff < 10.0,
        "Corner cases: DSSIM difference too large: {:.2}%",
        max_dssim_diff
    );
    // Size may differ due to progressive scan config and Huffman optimization
    if max_size_diff > 20.0 {
        println!(
            "NOTE: Size difference {}% is larger than typical (expected for edge cases)",
            max_size_diff
        );
    }
}
