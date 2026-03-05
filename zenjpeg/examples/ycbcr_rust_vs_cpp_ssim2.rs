//! Compare Rust vs C++ YCbCr outputs using SSIMULACRA2

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn load_test_image(path: &str) -> (Vec<u8>, u32, u32) {
    let file = std::fs::File::open(path).expect("open");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");
    let pixels = buf[..info.buffer_size()].to_vec();
    (pixels, info.width, info.height)
}

fn encode_rust_ycbcr(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("push failed");
    encoder.finish().expect("finish failed")
}

fn encode_cpp_ycbcr(src_path: &str, quality: u32) -> Vec<u8> {
    let out_path = format!("/tmp/cpp_ycbcr_ssim2_{}.jpg", quality);
    Command::new(zenjpeg_bench_utils::cjpegli_path())
        .args([src_path, &out_path, "-q", &quality.to_string()])
        .output()
        .expect("cjpegli failed");
    std::fs::read(&out_path).expect("read cpp output")
}

fn decode_jpeg_to_rgb(jpeg: &[u8], label: &str) -> (Vec<u8>, u32, u32) {
    let tmp_jpg = format!("/tmp/ycbcr_decode_test_{}.jpg", label);
    let tmp_png = format!("/tmp/ycbcr_decode_test_{}.png", label);
    std::fs::write(&tmp_jpg, jpeg).expect("write temp jpg");
    Command::new(zenjpeg_bench_utils::djpegli_path())
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

fn main() {
    let src_path_string = codec_corpus::Corpus::new()
        .ok()
        .and_then(|c| c.get("kodak").ok())
        .map(|p| p.join("1.png").to_string_lossy().to_string())
        .expect("codec-corpus unavailable; need kodak/1.png");
    let src_path: &str = &src_path_string;
    let (pixels, width, height) = load_test_image(src_path);

    println!("YCbCr Rust vs C++ Direct Comparison");
    println!("Image: kodak/1.png ({}x{})", width, height);
    println!("=========================================================");
    println!();
    println!(
        "{:7}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "Quality", "Rust bytes", "C++ bytes", "Size Δ%", "Mean Diff", "Max Diff"
    );
    println!("---------------------------------------------------------");

    for q in [70, 75, 80, 85, 90, 95] {
        let rust_jpeg = encode_rust_ycbcr(&pixels, width, height, q as f32);
        let cpp_jpeg = encode_cpp_ycbcr(src_path, q);

        let (rust_rgb, rw, rh) = decode_jpeg_to_rgb(&rust_jpeg, &format!("rust_{}", q));
        let (cpp_rgb, cw, ch) = decode_jpeg_to_rgb(&cpp_jpeg, &format!("cpp_{}", q));

        assert_eq!((rw, rh), (cw, ch), "dimension mismatch");

        let (_mean_diff, mean_abs_diff, max_diff) = compute_mean_diff(&rust_rgb, &cpp_rgb);
        let size_diff = (rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0) * 100.0;

        println!(
            "{:7}  {:>10}  {:>10}  {:>+7.1}%  {:>10.2}  {:>10.1}",
            q,
            rust_jpeg.len(),
            cpp_jpeg.len(),
            size_diff,
            mean_abs_diff,
            max_diff
        );
    }
}
