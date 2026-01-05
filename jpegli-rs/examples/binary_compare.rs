//! Binary comparison of Rust vs C++ progressive JPEG output.
//!
//! Finds exact byte differences to diagnose djpeg compatibility issues.

use jpegli::encode::Encoder;
use jpegli::quant::Quality;
use jpegli::types::{JpegMode, PixelFormat};
use std::fs;
use std::process::Command;

fn main() {
    // Create a small deterministic test image
    let width: u32 = 64;
    let height: u32 = 64;
    let mut pixels = vec![0u8; width as usize * height as usize * 3];

    // Fill with noise pattern
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            pixels[idx] = ((x * 17 + y * 23) % 256) as u8;
            pixels[idx + 1] = ((x * 29 + y * 31) % 256) as u8;
            pixels[idx + 2] = ((x * 37 + y * 41) % 256) as u8;
        }
    }

    // Save as PNG for C++ encoding
    let png_path = "/tmp/binary_compare.png";
    save_png(png_path, &pixels, width as usize, height as usize);

    // Encode with Rust
    let rust_jpg = encode_rust(&pixels, width, height);
    let rust_path = "/tmp/binary_compare_rust.jpg";
    fs::write(rust_path, &rust_jpg).unwrap();
    println!("Rust output: {} bytes", rust_jpg.len());

    // Encode with C++
    let cpp_path = "/tmp/binary_compare_cpp.jpg";
    let status =
        Command::new("/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli")
            .args([png_path, cpp_path, "-q", "50", "--progressive_level=2"])
            .status()
            .expect("Failed to run cjpegli");

    if !status.success() {
        eprintln!("C++ encoding failed");
        return;
    }

    let cpp_jpg = fs::read(cpp_path).unwrap();
    println!("C++ output: {} bytes", cpp_jpg.len());

    // Find markers in both files
    println!("\n=== Rust markers ===");
    let rust_markers = find_markers(&rust_jpg);
    for (pos, marker, len) in &rust_markers {
        println!("  0x{:04X}: {} (len {})", pos, marker_name(*marker), len);
    }

    println!("\n=== C++ markers ===");
    let cpp_markers = find_markers(&cpp_jpg);
    for (pos, marker, len) in &cpp_markers {
        println!("  0x{:04X}: {} (len {})", pos, marker_name(*marker), len);
    }

    // Compare marker sequences
    println!("\n=== Marker sequence comparison ===");
    let rust_seq: Vec<u8> = rust_markers.iter().map(|(_, m, _)| *m).collect();
    let cpp_seq: Vec<u8> = cpp_markers.iter().map(|(_, m, _)| *m).collect();

    if rust_seq == cpp_seq {
        println!("Marker sequences match!");
    } else {
        println!("Marker sequences DIFFER:");
        println!("  Rust: {:02X?}", rust_seq);
        println!("  C++:  {:02X?}", cpp_seq);
    }

    // Find first difference
    println!("\n=== First byte difference ===");
    let min_len = rust_jpg.len().min(cpp_jpg.len());
    for i in 0..min_len {
        if rust_jpg[i] != cpp_jpg[i] {
            println!("First difference at byte 0x{:04X}:", i);
            println!("  Rust: 0x{:02X}", rust_jpg[i]);
            println!("  C++:  0x{:02X}", cpp_jpg[i]);

            // Show context
            let start = i.saturating_sub(16);
            let end = (i + 16).min(min_len);
            println!("\n  Context (Rust):");
            print!("    ");
            for j in start..end {
                if j == i {
                    print!("[{:02X}]", rust_jpg[j]);
                } else {
                    print!(" {:02X} ", rust_jpg[j]);
                }
            }
            println!();

            println!("  Context (C++):");
            print!("    ");
            for j in start..end {
                if j == i {
                    print!("[{:02X}]", cpp_jpg[j]);
                } else {
                    print!(" {:02X} ", cpp_jpg[j]);
                }
            }
            println!();
            break;
        }
    }

    // Run djpeg on both
    println!("\n=== djpeg compatibility ===");

    println!("Rust output:");
    let output = Command::new("djpeg")
        .args(["-outfile", "/tmp/binary_compare_rust.ppm", rust_path])
        .output()
        .expect("Failed to run djpeg");

    if output.status.success() {
        println!("  SUCCESS");
    } else {
        println!("  FAILED: {}", String::from_utf8_lossy(&output.stderr));
    }

    println!("\nC++ output:");
    let output = Command::new("djpeg")
        .args(["-outfile", "/tmp/binary_compare_cpp.ppm", cpp_path])
        .output()
        .expect("Failed to run djpeg");

    if output.status.success() {
        println!("  SUCCESS");
    } else {
        println!("  FAILED: {}", String::from_utf8_lossy(&output.stderr));
    }
}

fn encode_rust(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(50.0))
        .mode(JpegMode::Progressive);

    encoder.encode(pixels).unwrap()
}

fn save_png(path: &str, pixels: &[u8], width: usize, height: usize) {
    let file = std::fs::File::create(path).unwrap();
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixels).unwrap();
}

fn find_markers(data: &[u8]) -> Vec<(usize, u8, usize)> {
    let mut markers = Vec::new();
    let mut i = 0;

    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            let marker = data[i + 1];

            // Skip RST markers (D0-D7) and markers without length
            if marker == 0xD8 || marker == 0xD9 || (marker >= 0xD0 && marker <= 0xD7) {
                markers.push((i, marker, 0));
                i += 2;
            } else if i + 3 < data.len() {
                let len = ((data[i + 2] as usize) << 8) | (data[i + 3] as usize);
                markers.push((i, marker, len));
                i += 2 + len;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    markers
}

fn marker_name(marker: u8) -> &'static str {
    match marker {
        0xD8 => "SOI",
        0xD9 => "EOI",
        0xE0 => "APP0",
        0xE1 => "APP1",
        0xE2 => "APP2",
        0xDB => "DQT",
        0xC0 => "SOF0 (baseline)",
        0xC2 => "SOF2 (progressive)",
        0xC4 => "DHT",
        0xDA => "SOS",
        0xDD => "DRI",
        0xFE => "COM",
        m if m >= 0xD0 && m <= 0xD7 => "RST",
        _ => "???",
    }
}
