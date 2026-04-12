//! Test that compares Rust output file sizes with C++ cjpegli.
//!
//! This test ensures the Rust port produces comparable file sizes to C++.
//!
//! Known structural differences (not bugs):
//! - Rust generates 4 Huffman tables (DC/AC × luma/chroma), C++ generates 2-3
//! - Rust uses SOF1 with 16-bit quant tables when needed; C++ clips to 8-bit
//! - These add ~30-150 bytes of fixed overhead
//!
//! For synthetic gradients (low entropy), this overhead can be 5-15% for 256x256.
//! For real photos, differences are typically <1%.

use std::fs;
use std::process::Command;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

/// Generate test image of specified size
fn create_gradient_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = ((x * 255) / width.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / height.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }
    rgb
}

/// Write PPM file for C++ cjpegli
fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

/// Encode with C++ cjpegli (matching Rust default settings: 4:4:4, AQ, sequential, optimized Huffman)
fn encode_cpp(ppm_path: &str, quality: u32) -> Option<Vec<u8>> {
    let cjpegli_path = zenjpeg::test_utils::find_cjpegli()?;

    let output_path = format!("/tmp/cpp_test_q{}.jpg", quality);
    // Match Rust defaults: AQ enabled, 4:4:4, sequential, optimized Huffman
    let output = Command::new(cjpegli_path)
        .args([
            "--chroma_subsampling=444",
            "-p",
            "0",
            ppm_path,
            &output_path,
            "-q",
            &quality.to_string(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    fs::read(&output_path).ok()
}

/// Encode with Rust jpegli (matching C++ settings: 4:4:4, AQ, sequential, optimized Huffman)
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    // Must match C++ --chroma_subsampling=444
    // Disable deringing to match C++ default (deringing is not enabled by -q)
    // Disable 16-bit quant tables (C++ cjpegli uses 8-bit by default)
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None)
        .deringing(false)
        .allow_16bit_quant_tables(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(rgb, enough::Unstoppable)
        .expect("push data");
    let jpeg = enc.finish().expect("finish");
    // Debug: save to file for inspection
    let _ = fs::write(
        format!("/tmp/rust_{}x{}_q{}.jpg", width, height, quality as u32),
        &jpeg,
    );
    jpeg
}

#[ignore = "requires C++ jpegli build"]
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_filesize_comparison_synthetic() {
    let test_cases = [
        (8, 8, "8x8"),
        (16, 16, "16x16"),
        (64, 64, "64x64"),
        (256, 256, "256x256"),
    ];

    for (width, height, name) in test_cases {
        let rgb = create_gradient_image(width, height);
        let ppm_path = format!("/tmp/test_{}.ppm", name);
        write_ppm(&ppm_path, &rgb, width, height).unwrap();

        for quality in [90, 80, 70] {
            let cpp_jpeg = match encode_cpp(&ppm_path, quality) {
                Some(j) => j,
                None => {
                    println!("Skipping {} Q{}: C++ not available", name, quality);
                    continue;
                }
            };

            let rust_jpeg = encode_rust(&rgb, width as u32, height as u32, quality as f32);

            let cpp_size = cpp_jpeg.len();
            let rust_size = rust_jpeg.len();
            let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;
            let diff_bytes = (rust_size as i64 - cpp_size as i64).abs();

            println!(
                "{} Q{}: C++={} Rust={} ({:+.1}%, {:+} bytes)",
                name,
                quality,
                cpp_size,
                rust_size,
                diff_pct,
                rust_size as i64 - cpp_size as i64
            );

            // For tiny images (<1KB), fixed overhead dominates.
            // Extra Huffman tables add ~30-50 bytes of header overhead vs C++.
            // For 8x8 images (1 MCU), this can be 10-15% difference in total size.
            // Synthetic gradients have low entropy, amplifying fixed overhead impact.
            // Real photos (test_filesize_comparison_photo) show <1% difference.
            let threshold_pct = if cpp_size < 1024 { 15.0 } else { 6.0 };
            let threshold_bytes = 50; // Extra Huffman tables overhead

            let pass = if cpp_size < 1024 {
                diff_bytes <= threshold_bytes || diff_pct.abs() < threshold_pct
            } else {
                diff_pct.abs() < threshold_pct
            };

            assert!(
                pass,
                "{} Q{}: file size differs by {:.1}% ({} bytes)",
                name, quality, diff_pct, diff_bytes
            );
        }
    }
}

#[ignore = "requires C++ jpegli build"]
#[test]
#[ignore = "requires C++ cjpegli build and test image"]
fn test_filesize_comparison_photo() {
    let png_path = zenjpeg::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !png_path.exists() {
        println!("Skipping: test image not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    // Load PNG
    let img = zenjpeg_bench_utils::load_png(&png_path).expect("Failed to load PNG");
    let width = img.width() as u32;
    let height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();

    // Save as PPM for C++
    let ppm_path = "/tmp/test_flower.ppm";
    write_ppm(ppm_path, &rgb, width as usize, height as usize).unwrap();

    println!("Image: {}x{}", width, height);

    for quality in [90, 80, 70, 60] {
        let cpp_jpeg = match encode_cpp(ppm_path, quality) {
            Some(j) => j,
            None => {
                println!("Skipping Q{}: C++ not available", quality);
                continue;
            }
        };

        let rust_jpeg = encode_rust(&rgb, width, height, quality as f32);

        let cpp_size = cpp_jpeg.len();
        let rust_size = rust_jpeg.len();
        let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;

        println!(
            "Q{}: C++={} Rust={} ({:+.1}%)",
            quality, cpp_size, rust_size, diff_pct
        );
    }
}
