//! Check if DCT coefficients differ between Rust and C++ encoders.

use jpegli::{Decoder, Encoder, PixelFormat};
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

    // Encode with Rust
    let rust_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .encode(rgb)
        .unwrap();

    // Encode with C++
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

        Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90"])
            .status()
            .unwrap();

        let cpp_jpeg = fs::read(cpp_path).unwrap();

        // Now decode both and check if pixels match
        let rust_decoded = Decoder::new().decode(&rust_jpeg).unwrap();
        let cpp_decoded = Decoder::new().decode(&cpp_jpeg).unwrap();

        // Check pixel differences
        let mut max_diff = 0i16;
        let mut diff_count = 0;
        for (r, c) in rust_decoded.data.iter().zip(cpp_decoded.data.iter()) {
            let diff = (*r as i16 - *c as i16).abs();
            if diff > 0 {
                diff_count += 1;
            }
            max_diff = max_diff.max(diff);
        }

        println!("Decoded pixel differences:");
        println!("  Max diff: {}", max_diff);
        println!(
            "  Pixels different: {} / {}",
            diff_count,
            rust_decoded.data.len()
        );
        println!(
            "  Percentage: {:.2}%",
            (diff_count as f64 / rust_decoded.data.len() as f64) * 100.0
        );

        if max_diff == 0 {
            println!(
                "\nDecoded pixels are IDENTICAL - difference must be in coefficient encoding!"
            );
        } else {
            println!(
                "\nDecoded pixels differ - the encoders produce different quality/coefficients"
            );
        }
    }
}
