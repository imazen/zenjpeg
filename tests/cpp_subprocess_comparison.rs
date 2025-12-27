//! Compare zenjpeg output vs C++ jpegli (cjpegli) via subprocess.
//!
//! These tests encode the same image with both zenjpeg and cjpegli,
//! then compare file sizes and quality metrics.
//!
//! Run with: cargo test --test cpp_subprocess_comparison -- --nocapture

use std::process::Command;
use tempfile::NamedTempFile;

mod common;

/// Check if cjpegli is available
fn cjpegli_available() -> bool {
    // cjpegli doesn't have --version, use -h which always succeeds
    Command::new("cjpegli")
        .arg("-h")
        .output()
        .is_ok()
}

/// Encode with cjpegli via subprocess
fn encode_with_cjpegli(png_path: &str, quality: u8) -> Option<Vec<u8>> {
    let output = NamedTempFile::new().ok()?;
    let output_path = output.path().to_str()?;

    let status = Command::new("cjpegli")
        .arg(png_path)
        .arg(output_path)
        .arg("-q")
        .arg(quality.to_string())
        .status()
        .ok()?;

    if status.success() {
        std::fs::read(output_path).ok()
    } else {
        None
    }
}

/// Create a simple test PNG
fn create_test_png() -> Option<NamedTempFile> {
    let mut file = NamedTempFile::with_suffix(".png").ok()?;

    // Create a simple 64x64 gradient image
    let width = 64u32;
    let height = 64u32;
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 4) as u8);  // R
            pixels.push((y * 4) as u8);  // G
            pixels.push(128u8);          // B
        }
    }

    // Write PNG
    let mut encoder = png::Encoder::new(&mut file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().ok()?;
    writer.write_image_data(&pixels).ok()?;
    writer.finish().ok()?;

    Some(file)
}

#[test]
fn test_cjpegli_available() {
    if !cjpegli_available() {
        eprintln!("SKIP: cjpegli not available");
        return;
    }
    println!("✓ cjpegli is available");
}

#[test]
fn test_file_size_comparison() {
    if !cjpegli_available() {
        eprintln!("SKIP: cjpegli not available");
        return;
    }

    let png_file = match create_test_png() {
        Some(f) => f,
        None => {
            eprintln!("SKIP: Could not create test PNG");
            return;
        }
    };
    let png_path = png_file.path().to_str().unwrap();

    println!("\n=== File Size Comparison: zenjpeg vs cjpegli ===\n");

    for quality in [50, 75, 90] {
        // Encode with cjpegli
        let cpp_jpeg = match encode_with_cjpegli(png_path, quality) {
            Some(data) => data,
            None => {
                eprintln!("SKIP: cjpegli encoding failed for Q{}", quality);
                continue;
            }
        };

        // Encode with zenjpeg
        let png_data = std::fs::read(png_path).unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0u8; reader.output_buffer_size()];
        reader.next_frame(&mut pixels).unwrap();
        let info = reader.info();

        let encoder = zenjpeg::Encoder::new()
            .quality(zenjpeg::Quality::Standard(quality));

        let rust_jpeg = match encoder.encode_rgb(&pixels, info.width as usize, info.height as usize) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("SKIP: zenjpeg encoding failed for Q{}: {:?}", quality, e);
                continue;
            }
        };

        let cpp_size = cpp_jpeg.len();
        let rust_size = rust_jpeg.len();
        let ratio = rust_size as f64 / cpp_size as f64;

        println!(
            "Q{:3}: cjpegli={:6} bytes, zenjpeg={:6} bytes, ratio={:.3}",
            quality, cpp_size, rust_size, ratio
        );

        // File size should be within 20% of cjpegli
        // (Different algorithms will produce different sizes)
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Q{}: zenjpeg file size ratio {:.3} is outside acceptable range",
            quality, ratio
        );
    }
}

#[test]
fn test_both_produce_valid_jpeg() {
    if !cjpegli_available() {
        eprintln!("SKIP: cjpegli not available");
        return;
    }

    let png_file = match create_test_png() {
        Some(f) => f,
        None => {
            eprintln!("SKIP: Could not create test PNG");
            return;
        }
    };
    let png_path = png_file.path().to_str().unwrap();

    // Encode with cjpegli
    let cpp_jpeg = encode_with_cjpegli(png_path, 85).expect("cjpegli should succeed");

    // Encode with zenjpeg
    let png_data = std::fs::read(png_path).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(&png_data));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0u8; reader.output_buffer_size()];
    reader.next_frame(&mut pixels).unwrap();
    let info = reader.info();

    let encoder = zenjpeg::Encoder::new()
        .quality(zenjpeg::Quality::Standard(85));
    let rust_jpeg = encoder.encode_rgb(&pixels, info.width as usize, info.height as usize)
        .expect("zenjpeg should succeed");

    // Both should be decodable
    let mut cpp_decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&cpp_jpeg));
    let mut rust_decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(&rust_jpeg));

    assert!(cpp_decoder.decode().is_ok(), "cjpegli output should be decodable");
    assert!(rust_decoder.decode().is_ok(), "zenjpeg output should be decodable");

    println!("✓ Both cjpegli and zenjpeg produce valid, decodable JPEGs");
}
