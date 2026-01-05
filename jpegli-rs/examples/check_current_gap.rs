//! Quick check of current file size gap vs C++ cjpegli.

use jpegli::{Encoder, PixelFormat};
use std::fs;
use std::io::Write;
use std::process::Command;

fn main() {
    let png_path = "../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];
    let width = info.width as u32;
    let height = info.height as u32;

    println!("Image: {}x{}", width, height);
    println!();

    // Encode with Rust (4:4:4, Q90)
    let rust_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(rgb)
        .unwrap();

    println!("Rust 4:4:4 Q90: {} bytes", rust_jpeg.len());
    fs::write("/tmp/rust_flower.jpg", &rust_jpeg).unwrap();

    // Encode with C++ cjpegli if available
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

        // Run cjpegli (4:4:4 is default, Q90)
        let output = Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90"])
            .output()
            .unwrap();

        if !output.status.success() {
            eprintln!(
                "cjpegli failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let cpp_jpeg = fs::read(cpp_path).unwrap();
        println!("C++  4:4:4 Q90: {} bytes", cpp_jpeg.len());
        println!();

        let diff_bytes = rust_jpeg.len() as i64 - cpp_jpeg.len() as i64;
        let diff_pct =
            ((rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64) * 100.0;
        println!("Difference: {:+} bytes ({:+.2}%)", diff_bytes, diff_pct);
    } else {
        println!("C++ cjpegli not found - cannot compare");
    }
}
