//! Check XYB quality metrics: Rust vs C++

use butteraugli::{compute_butteraugli, ButteraugliParams};
use enough::Unstoppable;
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use std::fs;
use std::io::Write;
use std::process::Command;
use zenjpeg::{
    decoder::Decoder,
    encoder::{EncoderConfig, PixelLayout, XybSubsampling},
};

fn main() {
    let png_path = "../internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png";

    // Load PNG
    let decoder = png::Decoder::new(std::fs::File::open(png_path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let rgb = &buf[..info.buffer_size()];
    let width = info.width as usize;
    let height = info.height as usize;

    println!("Image: {}x{}\n", width, height);

    // Encode with Rust XYB
    let config = EncoderConfig::xyb(90.0, XybSubsampling::Full);
    let mut enc = config
        .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(rgb, Unstoppable).unwrap();
    let rust_xyb = enc.finish().unwrap();

    println!("Rust XYB: {} bytes", rust_xyb.len());

    // Encode with C++ XYB
    if let Some(cjpegli) = zenjpeg::test_utils::find_cjpegli() {
        let ppm_path = "/tmp/flower.ppm";
        let cpp_path = "/tmp/cpp_flower_xyb.jpg";

        // Write PPM
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
        println!("C++  XYB: {} bytes", cpp_xyb.len());
        println!();

        // Decode both WITHOUT ICC (raw XYB data)
        let rust_decoded_raw = Decoder::new()
            .apply_icc(false)
            .decode(&rust_xyb, Unstoppable)
            .unwrap();
        let cpp_decoded_raw = Decoder::new()
            .apply_icc(false)
            .decode(&cpp_xyb, Unstoppable)
            .unwrap();

        // Check if raw XYB data matches
        let mut max_diff = 0i16;
        let mut diff_count = 0;
        for (r, c) in rust_decoded_raw
            .pixels_u8()
            .unwrap()
            .iter()
            .zip(cpp_decoded_raw.pixels_u8().unwrap().iter())
        {
            let d = (*r as i16 - *c as i16).abs();
            if d > 0 {
                diff_count += 1;
            }
            max_diff = max_diff.max(d);
        }

        println!(
            "Raw XYB data (no ICC): max diff = {}, pixels different = {} / {} ({:.2}%)",
            max_diff,
            diff_count,
            rust_decoded_raw.pixels_u8().unwrap().len(),
            (diff_count as f64 / rust_decoded_raw.pixels_u8().unwrap().len() as f64) * 100.0
        );

        // Try decoding with libjpeg-turbo (doesn't understand XYB ICC, treats as regular JPEG)
        let rust_decoded = Decoder::new()
            .apply_icc(false)
            .decode(&rust_xyb, Unstoppable)
            .unwrap();
        let cpp_decoded = Decoder::new()
            .apply_icc(false)
            .decode(&cpp_xyb, Unstoppable)
            .unwrap();

        // Compute quality metrics on the incorrectly-decoded XYB
        // (This is what SSIM2 sees since it doesn't understand XYB)
        let orig_rgb = Rgb::new(
            rgb.chunks(3)
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let rust_rgb = Rgb::new(
            rust_decoded
                .pixels_u8()
                .unwrap()
                .chunks(3)
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let cpp_rgb = Rgb::new(
            cpp_decoded
                .pixels_u8()
                .unwrap()
                .chunks(3)
                .map(|c| {
                    [
                        c[0] as f32 / 255.0,
                        c[1] as f32 / 255.0,
                        c[2] as f32 / 255.0,
                    ]
                })
                .collect(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let rust_ssim2 = compute_frame_ssimulacra2(orig_rgb.clone(), rust_rgb).unwrap();
        let cpp_ssim2 = compute_frame_ssimulacra2(orig_rgb, cpp_rgb).unwrap();

        println!("\nSSIMULACRA2 (raw XYB treated as sRGB - NOT correct metric for XYB!):");
        println!("  Rust XYB: {:.2}", rust_ssim2);
        println!("  C++  XYB: {:.2}", cpp_ssim2);
        println!("  Difference: {:.2}", rust_ssim2 - cpp_ssim2);

        // Butteraugli (should be used for XYB but we're comparing raw data)
        let params = ButteraugliParams::default();
        let rust_bfly = compute_butteraugli(
            rgb,
            &rust_decoded.pixels_u8().unwrap(),
            width,
            height,
            &params,
        )
        .unwrap()
        .score;
        let cpp_bfly = compute_butteraugli(
            rgb,
            &cpp_decoded.pixels_u8().unwrap(),
            width,
            height,
            &params,
        )
        .unwrap()
        .score;

        println!("\nButteraugli (raw XYB treated as sRGB - NOT correct!):");
        println!("  Rust XYB: {:.3}", rust_bfly);
        println!("  C++  XYB: {:.3}", cpp_bfly);
        println!("  Difference: {:+.3}", rust_bfly - cpp_bfly);

        println!("\nNote: Metrics computed on raw XYB data without ICC transform.");
        println!("For proper XYB quality assessment, ICC transform must be applied first.");
    }
}
