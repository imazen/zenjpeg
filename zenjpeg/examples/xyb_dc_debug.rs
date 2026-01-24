//! Debug XYB DC coefficient differences between Rust and C++
//!
//! This test encodes a solid color image and compares the DC coefficients.

use enough::Unstoppable;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};
use std::process::Command;

fn encode_rust_xyb_dump(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::xyb(quality, XybSubsampling::Full);
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("push failed");
    encoder.finish().expect("finish failed")
}

fn encode_cpp_xyb(src_path: &str, quality: u32) -> Vec<u8> {
    let out_path = "/tmp/cpp_dc_debug.jpg";
    Command::new("cjpegli")
        .args([src_path, out_path, "-q", &quality.to_string(), "--xyb"])
        .output()
        .expect("cjpegli failed");
    std::fs::read(out_path).expect("read cpp output")
}

fn decode_jpeg_to_rgb(jpeg: &[u8], label: &str) -> (Vec<u8>, u32, u32) {
    let tmp_jpg = format!("/tmp/dc_debug_{}.jpg", label);
    let tmp_png = format!("/tmp/dc_debug_{}.png", label);
    std::fs::write(&tmp_jpg, jpeg).expect("write temp jpg");
    Command::new("djpegli")
        .args([&tmp_jpg, &tmp_png])
        .output()
        .expect("djpegli decode failed");
    let file = std::fs::File::open(&tmp_png).expect("open decoded png");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");
    let pixels = buf[..info.buffer_size()].to_vec();
    (pixels, info.width, info.height)
}

fn main() {
    // Create a small solid color test image
    let width = 8u32;
    let height = 8u32;

    // Test with various solid colors
    let test_colors = [
        (128, 128, 128, "mid-gray"),
        (255, 255, 255, "white"),
        (0, 0, 0, "black"),
        (255, 0, 0, "red"),
        (0, 255, 0, "green"),
        (0, 0, 255, "blue"),
    ];

    println!("XYB DC Coefficient Debug");
    println!("========================");
    println!("Encoding 8x8 solid color blocks at quality 90\n");

    for (r, g, b, name) in test_colors {
        // Create solid color image
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for _ in 0..(width * height) {
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }

        // Save as PNG for C++
        let png_path = "/tmp/solid_color_test.png";
        {
            let file = std::fs::File::create(png_path).expect("create png");
            let mut encoder = png::Encoder::new(file, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("write header");
            writer.write_image_data(&pixels).expect("write data");
        }

        // Encode with both
        let rust_jpeg = encode_rust_xyb_dump(&pixels, width, height, 90.0);
        let cpp_jpeg = encode_cpp_xyb(png_path, 90);

        // Decode and compare
        let (rust_rgb, _, _) = decode_jpeg_to_rgb(&rust_jpeg, "rust");
        let (cpp_rgb, _, _) = decode_jpeg_to_rgb(&cpp_jpeg, "cpp");

        // Calculate average decoded values
        let rust_avg: Vec<f64> = (0..3)
            .map(|c| {
                rust_rgb
                    .iter()
                    .skip(c)
                    .step_by(3)
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / (width * height) as f64
            })
            .collect();
        let cpp_avg: Vec<f64> = (0..3)
            .map(|c| {
                cpp_rgb
                    .iter()
                    .skip(c)
                    .step_by(3)
                    .map(|&v| v as f64)
                    .sum::<f64>()
                    / (width * height) as f64
            })
            .collect();

        println!("{} (input: {},{},{}):", name, r, g, b);
        println!(
            "  Rust decoded RGB: {:.1}, {:.1}, {:.1}",
            rust_avg[0], rust_avg[1], rust_avg[2]
        );
        println!(
            "  C++  decoded RGB: {:.1}, {:.1}, {:.1}",
            cpp_avg[0], cpp_avg[1], cpp_avg[2]
        );
        println!(
            "  Difference (R-C): {:.1}, {:.1}, {:.1}",
            rust_avg[0] - cpp_avg[0],
            rust_avg[1] - cpp_avg[1],
            rust_avg[2] - cpp_avg[2]
        );
        println!();
    }
}
