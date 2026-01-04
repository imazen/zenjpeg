use jpegli::{Decoder, Encoder, PixelFormat};
use std::process::Command;

fn main() {
    println!("=== XYB Encoder/Decoder Matrix Test ===\n");

    // Test image - simple gradient
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;     // R gradient
            data[idx + 1] = ((y * 255) / height) as u8; // G gradient
            data[idx + 2] = 128;                        // B constant
        }
    }

    println!("Test image: {}x{} gradient\n", width, height);

    // Test modes
    let modes = [
        ("Baseline", false),
        ("Progressive", true),
    ];

    for (mode_name, progressive) in &modes {
        println!("=== {} Mode ===", mode_name);

        // 1. Rust XYB encoder
        println!("\n1. Encoding with Rust (XYB)...");
        let rust_xyb = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
            .mode(if *progressive {
                jpegli::types::JpegMode::Progressive
            } else {
                jpegli::types::JpegMode::Baseline
            })
            .use_xyb(true)
            .encode(&data)
            .unwrap();

        println!("   Rust XYB encoded: {} bytes", rust_xyb.len());
        let rust_path = format!("/tmp/xyb_matrix_rust_{}.jpg", if *progressive { "prog" } else { "base" });
        std::fs::write(&rust_path, &rust_xyb).unwrap();

        // 2. Rust YCbCr encoder (baseline)
        println!("\n2. Encoding with Rust (YCbCr)...");
        let rust_ycbcr = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
            .mode(if *progressive {
                jpegli::types::JpegMode::Progressive
            } else {
                jpegli::types::JpegMode::Baseline
            })
            .use_xyb(false)
            .encode(&data)
            .unwrap();

        println!("   Rust YCbCr encoded: {} bytes", rust_ycbcr.len());

        // 3. C++ XYB encoder (if available)
        println!("\n3. Encoding with C++ jpegli (XYB)...");
        std::fs::write("/tmp/test_input.ppm", generate_ppm(&data, width, height)).unwrap();

        let cpp_result = Command::new("internal/jpegli-cpp/build/tools/cjpegli")
            .args(&[
                "/tmp/test_input.ppm",
                "/tmp/xyb_matrix_cpp.jpg",
                "--xyb",
                "-p", if *progressive { "2" } else { "0" },
                "-q", "90",
            ])
            .output();

        let cpp_xyb = if cpp_result.is_ok() && cpp_result.as_ref().unwrap().status.success() {
            match std::fs::read("/tmp/xyb_matrix_cpp.jpg") {
                Ok(data) => {
                    println!("   C++ XYB encoded: {} bytes", data.len());
                    Some(data)
                }
                Err(e) => {
                    println!("   C++ XYB encode failed: {:?}", e);
                    None
                }
            }
        } else {
            println!("   C++ encoder not available");
            None
        };

        // Now test all decoder combinations
        println!("\n=== Decoding Matrix ===");
        println!("Format: [Encoder] → [Decoder]\n");

        // Test Rust XYB encoding
        test_decode("Rust XYB", &rust_xyb);

        // Test Rust YCbCr encoding (baseline)
        test_decode("Rust YCbCr", &rust_ycbcr);

        // Test C++ XYB encoding
        if let Some(ref cpp_data) = cpp_xyb {
            test_decode("C++ XYB", cpp_data);
        }

        println!("\n{}\n", "=".repeat(60));
    }
}

fn test_decode(encoder_name: &str, jpeg_data: &[u8]) {
    println!("{} →", encoder_name);

    // Test with jpegli-rs decoder
    print!("  jpegli-rs:    ");
    match Decoder::new().decode(jpeg_data) {
        Ok(decoded) => println!("✓ {}x{}", decoded.width, decoded.height),
        Err(e) => println!("✗ {:?}", e),
    }

    // Test with zune-jpeg decoder
    print!("  zune-jpeg:    ");
    use std::io::Cursor;
    let mut zune_decoder = zune_jpeg::JpegDecoder::new(Cursor::new(jpeg_data));
    match zune_decoder.decode() {
        Ok(_) => println!("✓"),
        Err(e) => println!("✗ {:?}", e),
    }

    // Test with mozjpeg decoder
    print!("  mozjpeg:      ");
    match mozjpeg::Decompress::new_mem(jpeg_data) {
        Ok(decoder) => match decoder.rgb() {
            Ok(_) => println!("✓"),
            Err(e) => println!("✗ {:?}", e),
        },
        Err(e) => println!("✗ {:?}", e),
    }

    // Test with jpeg-decoder
    print!("  jpeg-decoder: ");
    let mut jpeg_dec = jpeg_decoder::Decoder::new(&jpeg_data[..]);
    match jpeg_dec.decode() {
        Ok(_) => println!("✓"),
        Err(e) => println!("✗ {:?}", e),
    }

    println!();
}

fn generate_ppm(data: &[u8], width: usize, height: usize) -> Vec<u8> {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut ppm = header.into_bytes();
    ppm.extend_from_slice(data);
    ppm
}
