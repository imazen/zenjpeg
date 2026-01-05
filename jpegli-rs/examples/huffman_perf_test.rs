//! Test Huffman algorithm performance and XYB mode file sizes.

use jpegli::{Decoder, Encoder, PixelFormat};
use std::fs;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

fn main() {
    let png_path = "../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];
    let width = info.0 as u32;
    let height = info.1 as u32;

    println!("Image: {}x{}\n", width, height);

    // Test YCbCr mode
    println!("=== YCbCr Mode (4:4:4) ===\n");

    // Rust with jpegli Huffman (default)
    let start = Instant::now();
    let rust_ycbcr_jpegli = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(rgb)
        .unwrap();
    let jpegli_time = start.elapsed();

    println!(
        "Rust jpegli Huffman: {} bytes, {:?}",
        rust_ycbcr_jpegli.len(),
        jpegli_time
    );

    // C++ reference
    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        let ppm_path = "/tmp/flower.ppm";
        let cpp_path = "/tmp/cpp_flower.jpg";

        // Write PPM
        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        // C++ YCbCr
        Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90"])
            .output()
            .unwrap();
        let cpp_ycbcr = fs::read(cpp_path).unwrap();
        println!("C++  jpegli Huffman: {} bytes", cpp_ycbcr.len());

        let diff = rust_ycbcr_jpegli.len() as i64 - cpp_ycbcr.len() as i64;
        let pct = (diff as f64 / cpp_ycbcr.len() as f64) * 100.0;
        println!("Difference: {:+} bytes ({:+.2}%)\n", diff, pct);

        // Test XYB mode
        println!("=== XYB Mode ===\n");

        // Rust XYB
        let start = Instant::now();
        let rust_xyb = Encoder::new()
            .width(width)
            .height(height)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
            .use_xyb(true)
            .encode(rgb)
            .unwrap();
        let xyb_time = start.elapsed();

        println!(
            "Rust XYB (jpegli Huffman): {} bytes, {:?}",
            rust_xyb.len(),
            xyb_time
        );

        // C++ XYB
        Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90", "--xyb"])
            .output()
            .unwrap();
        let cpp_xyb = fs::read(cpp_path).unwrap();
        println!("C++  XYB (jpegli Huffman): {} bytes", cpp_xyb.len());

        let diff_xyb = rust_xyb.len() as i64 - cpp_xyb.len() as i64;
        let pct_xyb = (diff_xyb as f64 / cpp_xyb.len() as f64) * 100.0;
        println!("Difference: {:+} bytes ({:+.2}%)\n", diff_xyb, pct_xyb);

        // Decode and check pixel differences
        println!("=== Pixel Differences ===\n");

        let rust_ycbcr_decoded = Decoder::new().decode(&rust_ycbcr_jpegli).unwrap();
        let cpp_ycbcr_decoded = Decoder::new().decode(&cpp_ycbcr).unwrap();

        let mut max_diff = 0i16;
        let mut diff_count = 0;
        for (r, c) in rust_ycbcr_decoded
            .data
            .iter()
            .zip(cpp_ycbcr_decoded.data.iter())
        {
            let d = (*r as i16 - *c as i16).abs();
            if d > 0 {
                diff_count += 1;
            }
            max_diff = max_diff.max(d);
        }

        println!(
            "YCbCr: max diff = {}, pixels different = {} / {} ({:.2}%)",
            max_diff,
            diff_count,
            rust_ycbcr_decoded.data.len(),
            (diff_count as f64 / rust_ycbcr_decoded.data.len() as f64) * 100.0
        );

        // XYB decoding with ICC
        let rust_xyb_decoded = Decoder::new().apply_icc(true).decode(&rust_xyb).unwrap();
        let cpp_xyb_decoded = Decoder::new().apply_icc(true).decode(&cpp_xyb).unwrap();

        max_diff = 0;
        diff_count = 0;
        for (r, c) in rust_xyb_decoded
            .data
            .iter()
            .zip(cpp_xyb_decoded.data.iter())
        {
            let d = (*r as i16 - *c as i16).abs();
            if d > 0 {
                diff_count += 1;
            }
            max_diff = max_diff.max(d);
        }

        println!(
            "XYB:   max diff = {}, pixels different = {} / {} ({:.2}%)",
            max_diff,
            diff_count,
            rust_xyb_decoded.data.len(),
            (diff_count as f64 / rust_xyb_decoded.data.len() as f64) * 100.0
        );

        println!("\n=== Performance ===\n");
        println!("YCbCr encode time: {:?}", jpegli_time);
        println!("XYB encode time:   {:?}", xyb_time);
        println!(
            "XYB overhead: {:?} ({:.1}x)",
            xyb_time - jpegli_time,
            xyb_time.as_secs_f64() / jpegli_time.as_secs_f64()
        );
    }
}
