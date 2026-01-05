//! Compare decoded output: Rust vs C++ for XYB Baseline and Progressive

use jpegli::{Encoder, PixelFormat};
use std::fs;
use std::process::Command;

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

fn decode_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    match decode_zune(data) {
        Ok(pixels) => Some(pixels),
        Err(e) => {
            eprintln!("Decode error: {:?}", e);
            None
        }
    }
}

fn main() {
    let width = 64;
    let height = 64;
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    println!("Comparing XYB decoded output (64x64 gradient at Q70)\n");

    // Rust XYB Baseline
    let rust_xyb_base = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Baseline)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    // Rust XYB Progressive
    let rust_xyb_prog = Encoder::new()
        .width(64)
        .height(64)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(70.0))
        .mode(jpegli::types::JpegMode::Progressive)
        .use_xyb(true)
        .optimize_huffman(true)
        .encode(&data)
        .unwrap();

    // Save generated JPEGs for inspection
    std::fs::write("/tmp/rust_xyb_base.jpg", &rust_xyb_base).unwrap();
    std::fs::write("/tmp/rust_xyb_prog.jpg", &rust_xyb_prog).unwrap();

    let rust_base_decoded = decode_jpeg(&rust_xyb_base).unwrap();
    let rust_prog_decoded = match decode_jpeg(&rust_xyb_prog) {
        Some(pixels) => pixels,
        None => {
            eprintln!("ERROR: Failed to decode progressive JPEG!");
            eprintln!("Saved to /tmp/rust_xyb_prog.jpg for inspection");
            return;
        }
    };

    // C++ XYB Baseline and Progressive
    let ppm_path = "/tmp/test_xyb.ppm";
    write_ppm(ppm_path, &data, width, height).ok();

    let cjpegli = match jpegli::test_utils::find_cjpegli() {
        Some(p) => p,
        None => {
            println!("ERROR: C++ cjpegli not found");
            return;
        }
    };

    // C++ XYB Baseline
    Command::new(&cjpegli)
        .args(&[
            ppm_path,
            "/tmp/cpp_xyb_base.jpg",
            "--xyb",
            "-p",
            "0",
            "-q",
            "70",
        ])
        .output()
        .ok();

    // C++ XYB Progressive
    Command::new(&cjpegli)
        .args(&[
            ppm_path,
            "/tmp/cpp_xyb_prog.jpg",
            "--xyb",
            "-p",
            "2",
            "-q",
            "70",
        ])
        .output()
        .ok();

    let cpp_xyb_base = fs::read("/tmp/cpp_xyb_base.jpg").ok();
    let cpp_xyb_prog = fs::read("/tmp/cpp_xyb_prog.jpg").ok();

    let cpp_base_decoded = cpp_xyb_base.and_then(|d| decode_jpeg(&d));
    let cpp_prog_decoded = cpp_xyb_prog.and_then(|d| decode_jpeg(&d));

    println!("First 4 pixels decoded (showing R G B values):\n");

    // Compare Rust Baseline vs Progressive
    println!("RUST XYB Baseline:");
    for i in 0..4 {
        let idx = i * 3;
        println!(
            "  Pixel {}: R={:3} G={:3} B={:3}",
            i,
            rust_base_decoded[idx],
            rust_base_decoded[idx + 1],
            rust_base_decoded[idx + 2]
        );
    }

    println!("\nRUST XYB Progressive:");
    for i in 0..4 {
        let idx = i * 3;
        println!(
            "  Pixel {}: R={:3} G={:3} B={:3}",
            i,
            rust_prog_decoded[idx],
            rust_prog_decoded[idx + 1],
            rust_prog_decoded[idx + 2]
        );
    }

    if let Some(ref cpp_base) = cpp_base_decoded {
        println!("\nC++ XYB Baseline:");
        for i in 0..4 {
            let idx = i * 3;
            println!(
                "  Pixel {}: R={:3} G={:3} B={:3}",
                i,
                cpp_base[idx],
                cpp_base[idx + 1],
                cpp_base[idx + 2]
            );
        }
    }

    if let Some(ref cpp_prog) = cpp_prog_decoded {
        println!("\nC++ XYB Progressive:");
        for i in 0..4 {
            let idx = i * 3;
            println!(
                "  Pixel {}: R={:3} G={:3} B={:3}",
                i,
                cpp_prog[idx],
                cpp_prog[idx + 1],
                cpp_prog[idx + 2]
            );
        }
    }

    // Compare Rust baseline vs C++ baseline
    if let Some(cpp_base) = &cpp_base_decoded {
        let mut diffs = 0;
        for i in 0..rust_base_decoded.len() {
            if rust_base_decoded[i] != cpp_base[i] {
                diffs += 1;
            }
        }
        println!(
            "\nRust Baseline vs C++ Baseline: {} pixels differ (out of {})",
            diffs,
            rust_base_decoded.len()
        );
    }

    // Compare Rust progressive vs C++ progressive
    if let Some(cpp_prog) = &cpp_prog_decoded {
        let mut diffs = 0;
        for i in 0..rust_prog_decoded.len() {
            if rust_prog_decoded[i] != cpp_prog[i] {
                diffs += 1;
            }
        }
        println!(
            "Rust Progressive vs C++ Progressive: {} pixels differ (out of {})",
            diffs,
            rust_prog_decoded.len()
        );
    }

    // Key question: Do Rust baseline and progressive produce the same output?
    let mut rust_diff = 0;
    for i in 0..rust_base_decoded.len() {
        if rust_base_decoded[i] != rust_prog_decoded[i] {
            rust_diff += 1;
        }
    }
    println!(
        "\n⚠️  RUST Baseline vs Progressive: {} pixels differ (out of {})",
        rust_diff,
        rust_base_decoded.len()
    );

    if rust_diff == 0 {
        println!("✅ Rust baseline and progressive produce IDENTICAL decoded output");
    } else {
        println!("❌ Rust baseline and progressive produce DIFFERENT decoded output - BUG!");
    }

    // Same check for C++
    if let (Some(cpp_base), Some(cpp_prog)) = (&cpp_base_decoded, &cpp_prog_decoded) {
        let mut cpp_diff = 0;
        for i in 0..cpp_base.len() {
            if cpp_base[i] != cpp_prog[i] {
                cpp_diff += 1;
            }
        }
        println!(
            "C++ Baseline vs Progressive: {} pixels differ (out of {})",
            cpp_diff,
            cpp_base.len()
        );

        if cpp_diff == 0 {
            println!("✅ C++ baseline and progressive produce IDENTICAL decoded output");
        } else {
            println!("❌ C++ baseline and progressive also differ - not a Rust bug");
        }
    }
}

fn decode_zune(data: &[u8]) -> Result<Vec<u8>, zune_jpeg::errors::DecodeErrors> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode()
}
