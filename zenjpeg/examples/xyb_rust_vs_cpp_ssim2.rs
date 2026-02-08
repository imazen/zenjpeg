//! Compare Rust vs C++ XYB outputs using SSIMULACRA2
//!
//! This compares the decoded outputs of both encoders directly to see
//! how similar they are to each other.

use enough::Unstoppable;
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;
use std::process::Command;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

fn load_test_image(path: &str) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).expect("open");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");
    let pixels = buf[..info.buffer_size()].to_vec();
    (pixels, info.width, info.height)
}

fn encode_rust_xyb(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::xyb(quality, XybSubsampling::Full).progressive(true); // Match C++ cjpegli's default progressive mode
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("push failed");
    encoder.finish().expect("finish failed")
}

fn encode_cpp_xyb(src_path: &str, quality: u32) -> Vec<u8> {
    let out_path = format!("/tmp/cpp_xyb_ssim2_{}.jpg", quality);
    Command::new("cjpegli")
        .args([src_path, &out_path, "-q", &quality.to_string(), "--xyb"])
        .output()
        .expect("cjpegli failed");
    std::fs::read(&out_path).expect("read cpp output")
}

fn decode_jpeg_to_rgb(jpeg: &[u8], label: &str) -> (Vec<u8>, u32, u32) {
    // Save JPEG to temp file
    let tmp_jpg = format!("/tmp/xyb_decode_test_{}.jpg", label);
    let tmp_png = format!("/tmp/xyb_decode_test_{}.png", label);
    std::fs::write(&tmp_jpg, jpeg).expect("write temp jpg");

    // Use djpegli to decode
    Command::new("djpegli")
        .args([&tmp_jpg, &tmp_png])
        .output()
        .expect("djpegli decode failed");

    // Load the decoded PNG
    let file = std::fs::File::open(&tmp_png).expect("open decoded png");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");
    let pixels = buf[..info.buffer_size()].to_vec();
    (pixels, info.width, info.height)
}

fn ssim2_from_rgb(img1: &[u8], img2: &[u8], width: usize, height: usize) -> f64 {
    let v1 = ImgVec::new(img1.to_vec(), width * 3, height);
    let v2 = ImgVec::new(img2.to_vec(), width * 3, height);
    compute_ssimulacra2(v1.as_ref(), v2.as_ref()).unwrap_or(0.0)
}

fn compute_mean_diff(img1: &[u8], img2: &[u8]) -> (f64, f64, f64) {
    assert_eq!(img1.len(), img2.len());
    let mut sum_diff = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut max_diff = 0.0f64;
    for (a, b) in img1.iter().zip(img2.iter()) {
        let diff = *a as f64 - *b as f64;
        sum_diff += diff;
        sum_abs_diff += diff.abs();
        max_diff = max_diff.max(diff.abs());
    }
    let n = img1.len() as f64;
    (sum_diff / n, sum_abs_diff / n, max_diff)
}

fn compute_channel_diff(img1: &[u8], img2: &[u8]) -> [(f64, f64, f64); 3] {
    assert_eq!(img1.len(), img2.len());
    let mut result = [(0.0f64, 0.0f64, 0.0f64); 3];
    let npixels = img1.len() / 3;

    for c in 0..3 {
        let mut sum_diff = 0.0f64;
        let mut sum_abs_diff = 0.0f64;
        let mut max_diff = 0.0f64;
        for i in 0..npixels {
            let a = img1[i * 3 + c] as f64;
            let b = img2[i * 3 + c] as f64;
            let diff = a - b;
            sum_diff += diff;
            sum_abs_diff += diff.abs();
            max_diff = max_diff.max(diff.abs());
        }
        result[c] = (
            sum_diff / npixels as f64,
            sum_abs_diff / npixels as f64,
            max_diff,
        );
    }
    result
}

fn main() {
    let src_path_string = codec_corpus::Corpus::new().ok()
        .and_then(|c| c.get("kodak").ok())
        .map(|p| p.join("1.png").to_string_lossy().to_string())
        .expect("codec-corpus unavailable; need kodak/1.png");
    let src_path: &str = &src_path_string;
    let (pixels, width, height) = load_test_image(src_path);

    println!("XYB Rust vs C++ Direct Comparison (SSIMULACRA2)");
    println!("Image: kodak/1.png ({}x{})", width, height);
    println!("=========================================================");
    println!();
    println!(
        "{:7}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}  {:>10}",
        "Quality", "Rust bytes", "C++ bytes", "Size Δ%", "Mean Diff", "Abs Diff", "Max Diff"
    );
    println!(
        "------------------------------------------------------------------------------------"
    );

    for q in [90] {
        // Just test one quality for detailed output
        let rust_jpeg = encode_rust_xyb(&pixels, width, height, q as f32);
        let cpp_jpeg = encode_cpp_xyb(src_path, q);

        let (rust_rgb, rw, rh) = decode_jpeg_to_rgb(&rust_jpeg, &format!("rust_{}", q));
        let (cpp_rgb, cw, ch) = decode_jpeg_to_rgb(&cpp_jpeg, &format!("cpp_{}", q));

        assert_eq!((rw, rh), (cw, ch), "dimension mismatch");

        let (mean_diff, mean_abs_diff, max_diff) = compute_mean_diff(&rust_rgb, &cpp_rgb);
        let channel_diff = compute_channel_diff(&rust_rgb, &cpp_rgb);
        let size_diff = (rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0) * 100.0;

        println!(
            "{:7}  {:>10}  {:>10}  {:>+7.1}%  {:>10.2}  {:>10.2}  {:>10.1}",
            q,
            rust_jpeg.len(),
            cpp_jpeg.len(),
            size_diff,
            mean_diff,
            mean_abs_diff,
            max_diff
        );

        // Per-channel breakdown
        println!();
        println!("Per-channel differences (Rust - C++):");
        for (i, name) in ["R (X)", "G (Y)", "B (B)"].iter().enumerate() {
            let (ch_mean, ch_abs, ch_max) = channel_diff[i];
            println!(
                "  {}: mean={:+.2}, abs={:.2}, max={:.1}",
                name, ch_mean, ch_abs, ch_max
            );
        }
    }

    println!();
    println!("SSIM2 interpretation (Rust output vs C++ output):");
    println!("  90+ = Nearly identical outputs");
    println!("  80-90 = Minor differences");
    println!("  70-80 = Noticeable differences");
    println!("  <70 = Significant differences");

    // Also show how each compares to original
    println!();
    println!("Quality vs Original (for reference):");
    println!(
        "{:7}  {:>16}  {:>16}",
        "Quality", "Rust vs Orig", "C++ vs Orig"
    );
    println!("---------------------------------------------------------");

    for q in [70, 80, 90] {
        let rust_jpeg = encode_rust_xyb(&pixels, width, height, q as f32);
        let cpp_jpeg = encode_cpp_xyb(src_path, q);

        let (rust_rgb, _, _) = decode_jpeg_to_rgb(&rust_jpeg, &format!("rust_orig_{}", q));
        let (cpp_rgb, _, _) = decode_jpeg_to_rgb(&cpp_jpeg, &format!("cpp_orig_{}", q));

        let ssim2_rust_vs_orig =
            ssim2_from_rgb(&pixels, &rust_rgb, width as usize, height as usize);
        let ssim2_cpp_vs_orig = ssim2_from_rgb(&pixels, &cpp_rgb, width as usize, height as usize);

        println!(
            "{:7}  {:>16.2}  {:>16.2}",
            q, ssim2_rust_vs_orig, ssim2_cpp_vs_orig
        );
    }
}
