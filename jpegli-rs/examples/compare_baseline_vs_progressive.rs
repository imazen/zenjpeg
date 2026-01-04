//! Compare decoder accuracy: baseline vs progressive

use jpegli::{Decoder, Encoder, PixelFormat};
use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    let png_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png"
    );

    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        // Load PNG
        let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let rgb = &buf[..info.buffer_size()];
        let width = info.width as u32;
        let height = info.height as u32;

        // Write PPM
        let ppm_path = "/tmp/compare_modes.ppm";
        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        // Test 1: C++ baseline JPEG
        println!("=== C++ Baseline JPEG ===");
        let cpp_baseline_path = "/tmp/cpp_baseline.jpg";
        Command::new(&cjpegli)
            .args([ppm_path, cpp_baseline_path, "-q", "90", "-p", "0"]) // -p 0 = baseline
            .output()
            .unwrap();

        let cpp_baseline = fs::read(cpp_baseline_path).unwrap();
        println!("Size: {} bytes", cpp_baseline.len());

        let jpegli_baseline = Decoder::new().decode(&cpp_baseline).unwrap();
        let mut zune_baseline = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_baseline));
        let zune_baseline_pixels = zune_baseline.decode().unwrap();

        let baseline_diff = compare_pixels(&jpegli_baseline.data, &zune_baseline_pixels, "Baseline");

        // Test 2: C++ progressive JPEG
        println!("\n=== C++ Progressive JPEG ===");
        let cpp_progressive_path = "/tmp/cpp_progressive.jpg";
        Command::new(&cjpegli)
            .args([ppm_path, cpp_progressive_path, "-q", "90"]) // default = progressive level 2
            .output()
            .unwrap();

        let cpp_progressive = fs::read(cpp_progressive_path).unwrap();
        println!("Size: {} bytes", cpp_progressive.len());

        let jpegli_progressive = Decoder::new().decode(&cpp_progressive).unwrap();
        let mut zune_progressive = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&cpp_progressive));
        let zune_progressive_pixels = zune_progressive.decode().unwrap();

        let progressive_diff = compare_pixels(&jpegli_progressive.data, &zune_progressive_pixels, "Progressive");

        // Test 3: Rust baseline
        println!("\n=== Rust Baseline JPEG ===");
        let rust_baseline = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
            .mode(jpegli::JpegMode::Baseline)
            .encode(rgb)
            .unwrap();
        println!("Size: {} bytes", rust_baseline.len());

        let jpegli_rust_baseline = Decoder::new().decode(&rust_baseline).unwrap();
        let mut zune_rust_baseline = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(&rust_baseline));
        let zune_rust_baseline_pixels = zune_rust_baseline.decode().unwrap();

        let rust_baseline_diff = compare_pixels(&jpegli_rust_baseline.data, &zune_rust_baseline_pixels, "Rust Baseline");

        // Summary
        println!("\n=== Summary ===");
        println!("C++ Baseline:    max_diff={}, differ={:.1}%", baseline_diff.0, baseline_diff.1);
        println!("C++ Progressive: max_diff={}, differ={:.1}%", progressive_diff.0, progressive_diff.1);
        println!("Rust Baseline:   max_diff={}, differ={:.1}%", rust_baseline_diff.0, rust_baseline_diff.1);

        if baseline_diff.0 == 0 {
            println!("\n✓ Baseline decoding is pixel-perfect!");
        } else {
            println!("\n⚠ Baseline decoding also has rounding differences");
        }
    }
}

fn compare_pixels(jpegli: &[u8], zune: &[u8], label: &str) -> (i16, f64) {
    let mut max_diff = 0i16;
    let mut diff_count = 0;

    for (j, z) in jpegli.iter().zip(zune.iter()) {
        let diff = (*j as i16 - *z as i16).abs();
        max_diff = max_diff.max(diff);
        if diff > 0 {
            diff_count += 1;
        }
    }

    let pct = (diff_count as f64 / jpegli.len() as f64) * 100.0;
    println!("{}: max_diff={}, pixels_differ={}/{} ({:.1}%)",
        label, max_diff, diff_count, jpegli.len(), pct);

    (max_diff, pct)
}
