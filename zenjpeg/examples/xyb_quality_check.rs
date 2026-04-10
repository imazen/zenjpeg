//! Check XYB quality metrics: Rust vs C++

use butteraugli::ButteraugliParams;
use enough::Unstoppable;
use imgref::{Img, ImgVec};
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
    let loaded =
        zenjpeg_bench_utils::load_png(std::path::Path::new(png_path)).expect("Failed to load PNG");
    let rgb_vec: Vec<u8> = loaded.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let rgb = &rgb_vec[..];
    let width = loaded.width();
    let height = loaded.height();

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
        let rust_decoded_raw = Decoder::new().decode(&rust_xyb, Unstoppable).unwrap();
        let cpp_decoded_raw = Decoder::new().decode(&cpp_xyb, Unstoppable).unwrap();

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
        let rust_decoded = Decoder::new().decode(&rust_xyb, Unstoppable).unwrap();
        let cpp_decoded = Decoder::new().decode(&cpp_xyb, Unstoppable).unwrap();

        // Compute quality metrics on the incorrectly-decoded XYB
        // (This is what SSIM2 sees since it doesn't understand XYB)
        let to_pixels =
            |data: &[u8]| -> Vec<[u8; 3]> { data.chunks(3).map(|c| [c[0], c[1], c[2]]).collect() };

        let orig_img: ImgVec<[u8; 3]> = Img::new(to_pixels(rgb), width, height);
        let rust_img: ImgVec<[u8; 3]> =
            Img::new(to_pixels(rust_decoded.pixels_u8().unwrap()), width, height);
        let cpp_img: ImgVec<[u8; 3]> =
            Img::new(to_pixels(cpp_decoded.pixels_u8().unwrap()), width, height);

        let rust_ssim2 =
            fast_ssim2::compute_ssimulacra2(orig_img.as_ref(), rust_img.as_ref()).unwrap();
        let cpp_ssim2 =
            fast_ssim2::compute_ssimulacra2(orig_img.as_ref(), cpp_img.as_ref()).unwrap();

        println!("\nSSIMULACRA2 (raw XYB treated as sRGB - NOT correct metric for XYB!):");
        println!("  Rust XYB: {:.2}", rust_ssim2);
        println!("  C++  XYB: {:.2}", cpp_ssim2);
        println!("  Difference: {:.2}", rust_ssim2 - cpp_ssim2);

        // Butteraugli (should be used for XYB but we're comparing raw data)
        let params = ButteraugliParams::default();
        let orig_pixels: Vec<rgb::RGB8> = rgb
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        let orig_img = imgref::Img::new(&orig_pixels[..], width, height);

        let rust_pixels: Vec<rgb::RGB8> = rust_decoded
            .pixels_u8()
            .unwrap()
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        let rust_img = imgref::Img::new(&rust_pixels[..], width, height);
        let rust_bfly = butteraugli::butteraugli(orig_img, rust_img, &params)
            .unwrap()
            .score;

        let orig_pixels2: Vec<rgb::RGB8> = rgb
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        let orig_img2 = imgref::Img::new(&orig_pixels2[..], width, height);
        let cpp_pixels: Vec<rgb::RGB8> = cpp_decoded
            .pixels_u8()
            .unwrap()
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        let cpp_img = imgref::Img::new(&cpp_pixels[..], width, height);
        let cpp_bfly = butteraugli::butteraugli(orig_img2, cpp_img, &params)
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
