//! Simple test of XYB encode/decode

use jpegli::{Decoder, Encoder, PixelFormat};

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

    println!("Encoding {}x{} image in XYB mode...", width, height);

    // Encode with Rust XYB (progressive to match C++ default)
    let xyb_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(90.0))
        .use_xyb(true)
        .mode(jpegli::JpegMode::Progressive)
        .encode(rgb)
        .unwrap();

    println!("Encoded: {} bytes", xyb_jpeg.len());

    // Try to decode without ICC
    println!("\nDecoding WITHOUT ICC transform...");
    match Decoder::new().apply_icc(false).decode(&xyb_jpeg) {
        Ok(decoded) => {
            println!(
                "✓ Decoded successfully: {}x{}, {} bytes",
                decoded.width,
                decoded.height,
                decoded.data.len()
            );
        }
        Err(e) => {
            println!("✗ Decode failed: {:?}", e);
        }
    }

    // Try to decode with ICC
    println!("\nDecoding WITH ICC transform...");
    match Decoder::new().apply_icc(true).decode(&xyb_jpeg) {
        Ok(decoded) => {
            println!(
                "✓ Decoded successfully: {}x{}, {} bytes",
                decoded.width,
                decoded.height,
                decoded.data.len()
            );
        }
        Err(e) => {
            println!("✗ Decode failed: {:?}", e);
        }
    }

    // Try C++ XYB
    if let Some(cjpegli) = jpegli::test_utils::find_cjpegli() {
        use std::fs;
        use std::io::Write;
        use std::process::Command;

        let ppm_path = "/tmp/test_xyb.ppm";
        let cpp_path = "/tmp/test_xyb_cpp.jpg";

        let mut ppm = fs::File::create(ppm_path).unwrap();
        writeln!(ppm, "P6").unwrap();
        writeln!(ppm, "{} {}", width, height).unwrap();
        writeln!(ppm, "255").unwrap();
        ppm.write_all(rgb).unwrap();
        drop(ppm);

        Command::new(&cjpegli)
            .args([ppm_path, cpp_path, "-q", "90", "--xyb"])
            .output()
            .unwrap();

        let cpp_xyb = fs::read(cpp_path).unwrap();
        println!("\nC++ XYB JPEG: {} bytes", cpp_xyb.len());

        println!("\nDecoding C++ XYB WITHOUT ICC...");
        match Decoder::new().apply_icc(false).decode(&cpp_xyb) {
            Ok(decoded) => {
                println!(
                    "✓ Decoded successfully: {}x{}, {} bytes",
                    decoded.width,
                    decoded.height,
                    decoded.data.len()
                );
            }
            Err(e) => {
                println!("✗ Decode failed: {:?}", e);
            }
        }

        println!("\nDecoding C++ XYB WITH ICC...");
        match Decoder::new().apply_icc(true).decode(&cpp_xyb) {
            Ok(decoded) => {
                println!(
                    "✓ Decoded successfully: {}x{}, {} bytes",
                    decoded.width,
                    decoded.height,
                    decoded.data.len()
                );
            }
            Err(e) => {
                println!("✗ Decode failed: {:?}", e);
            }
        }
    }
}
