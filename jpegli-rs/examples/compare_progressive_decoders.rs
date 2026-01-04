//! Compare progressive JPEG decoders: jpegli-rs vs zune-jpeg vs jpeg-decoder
//!
//! This helps identify bugs in our progressive decoder by comparing against
//! known-working implementations.

use jpegli::{Decoder, PixelFormat};
use std::fs;

fn main() {
    // First, create a C++ progressive JPEG to test with
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        use std::io::Write;
        use std::process::Command;

        let cpp_prog_path = "/tmp/cpp_progressive_ycbcr.jpg";

        // Load PNG for C++ encoding
        let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let rgb = &buf[..info.buffer_size()];
        let width = info.width as u32;
        let height = info.height as u32;

        // Write PPM for C++ cjpegli
        let ppm_path = "/tmp/test_progressive.ppm";
        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        // Encode with C++ cjpegli (progressive level 2 by default)
        println!("Encoding with C++ cjpegli (progressive level 2)...");
        let output = Command::new(&cjpegli)
            .args([ppm_path, cpp_prog_path, "-q", "90"])
            .output()
            .unwrap();

        if !output.status.success() {
            eprintln!("C++ encoding failed: {}", String::from_utf8_lossy(&output.stderr));
            return;
        }

        let cpp_jpeg = fs::read(cpp_prog_path).unwrap();
        println!("C++ progressive JPEG: {} bytes\n", cpp_jpeg.len());

        // Test 1: jpegli-rs decoder
        println!("=== Test 1: jpegli-rs decoder ===");
        match Decoder::new().decode(&cpp_jpeg) {
            Ok(decoded) => {
                println!("✓ Success: {}x{}, {} bytes",
                    decoded.width, decoded.height, decoded.data.len());
                println!("  Format: {:?}\n", decoded.format);
            }
            Err(e) => {
                println!("✗ Failed: {:?}\n", e);
            }
        }

        // Test 2: zune-jpeg decoder
        println!("=== Test 2: zune-jpeg decoder ===");
        let mut zune_decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_jpeg));
        match zune_decoder.decode() {
            Ok(pixels) => {
                let info = zune_decoder.info().unwrap();
                println!("✓ Success: {}x{}, {} bytes",
                    info.width, info.height, pixels.len());
                println!("  SOF: {:?}\n", info.sof);
            }
            Err(e) => {
                println!("✗ Failed: {:?}\n", e);
            }
        }

        // Test 3: jpeg-decoder (libjpeg-turbo)
        println!("=== Test 3: jpeg-decoder (libjpeg-turbo) ===");
        let mut libjpeg_decoder = jpeg_decoder::Decoder::new(&cpp_jpeg[..]);
        match libjpeg_decoder.decode() {
            Ok(pixels) => {
                let info = libjpeg_decoder.info().unwrap();
                println!("✓ Success: {}x{}, {} bytes",
                    info.width, info.height, pixels.len());
                println!("  Pixel format: {:?}\n", info.pixel_format);
            }
            Err(e) => {
                println!("✗ Failed: {:?}\n", e);
            }
        }

        // If both zune-jpeg and jpegli-rs succeeded, compare outputs
        let mut zune_decoder2 = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_jpeg));
        let jpegli_result = Decoder::new().decode(&cpp_jpeg);
        let zune_result = zune_decoder2.decode();

        if let (Ok(jpegli_img), Ok(zune_pixels)) = (jpegli_result, zune_result) {
            println!("=== Comparing pixel outputs ===");

            if jpegli_img.data.len() == zune_pixels.len() {
                let mut max_diff = 0i16;
                let mut diff_count = 0;

                for (j, z) in jpegli_img.data.iter().zip(zune_pixels.iter()) {
                    let diff = (*j as i16 - *z as i16).abs();
                    if diff > 0 {
                        diff_count += 1;
                    }
                    max_diff = max_diff.max(diff);
                }

                println!("Max pixel difference: {}", max_diff);
                println!("Pixels different: {} / {} ({:.2}%)",
                    diff_count, jpegli_img.data.len(),
                    (diff_count as f64 / jpegli_img.data.len() as f64) * 100.0);

                if max_diff == 0 {
                    println!("\n✓ PERFECT MATCH: jpegli-rs and zune-jpeg produce identical output!");
                } else if max_diff <= 2 {
                    println!("\n✓ EXCELLENT: Minor differences only (rounding)");
                } else {
                    println!("\n⚠ DIFFERENCES: Outputs differ by up to {} (investigate)", max_diff);
                }
            } else {
                println!("⚠ Size mismatch: jpegli-rs={} bytes, zune-jpeg={} bytes",
                    jpegli_img.data.len(), zune_pixels.len());
            }
        }

        // Now test with a Rust progressive JPEG
        println!("\n\n=== Testing Rust Progressive Encoding ===");
        let rust_prog = jpegli::Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
            .mode(jpegli::JpegMode::Progressive)
            .encode(rgb)
            .unwrap();

        println!("Rust progressive JPEG: {} bytes\n", rust_prog.len());

        // Test Rust progressive with all decoders
        println!("Decoding Rust progressive with jpegli-rs:");
        match Decoder::new().decode(&rust_prog) {
            Ok(decoded) => {
                println!("✓ Success: {}x{}", decoded.width, decoded.height);
            }
            Err(e) => {
                println!("✗ Failed: {:?}", e);
            }
        }

        println!("\nDecoding Rust progressive with zune-jpeg:");
        let mut zune_decoder3 = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&rust_prog));
        match zune_decoder3.decode() {
            Ok(_) => {
                let info = zune_decoder3.info().unwrap();
                println!("✓ Success: {}x{}", info.width, info.height);
            }
            Err(e) => {
                println!("✗ Failed: {:?}", e);
            }
        }

        println!("\nDecoding Rust progressive with jpeg-decoder:");
        let mut libjpeg_decoder2 = jpeg_decoder::Decoder::new(&rust_prog[..]);
        match libjpeg_decoder2.decode() {
            Ok(_) => {
                let info = libjpeg_decoder2.info().unwrap();
                println!("✓ Success: {}x{}", info.width, info.height);
            }
            Err(e) => {
                println!("✗ Failed: {:?}", e);
            }
        }
    } else {
        println!("Error: cjpegli not found. Build C++ jpegli first.");
    }
}
