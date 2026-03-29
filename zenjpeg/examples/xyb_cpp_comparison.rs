//! Compare XYB + hybrid output against C jpegli.

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::color::icc::TargetColorSpace;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

fn main() {
    let image_path = std::env::args()
        .nth(1)
        .or_else(|| {
            codec_corpus::Corpus::new()
                .ok()
                .and_then(|c| c.get("kodak").ok())
                .map(|p| p.join("1.png").to_string_lossy().to_string())
        })
        .expect("Usage: xyb_cpp_comparison <image.png> or set up codec-corpus");

    // Load image
    let loaded = zenjpeg_bench_utils::load_png(std::path::Path::new(&image_path))
        .expect("Failed to load PNG");
    let pixel_bytes: Vec<u8> = loaded.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    let pixels = &pixel_bytes[..];
    let width = loaded.width() as u32;
    let height = loaded.height() as u32;

    println!("Comparing XYB encoding: Rust vs C jpegli");
    println!("Image: {} ({}x{})", image_path, width, height);
    println!();

    let qualities = [70, 80, 90];

    println!(
        "{:<8} {:<12} {:<12} {:<10} {:<12} {:<12} {:<10}",
        "Quality",
        "C bytes",
        "Rust bytes",
        "Size diff",
        "C butteraugli",
        "Rust butteraugli",
        "Δ quality"
    );
    println!("{}", "-".repeat(80));

    for &q in &qualities {
        // 1. Encode with C jpegli (XYB mode)
        let cpp_path = format!("/tmp/cpp_xyb_q{}.jpg", q);
        let cpp_status = Command::new("cjpegli")
            .args([&image_path, &cpp_path, "-q", &q.to_string(), "--xyb"])
            .output()
            .expect("Failed to run cjpegli");

        if !cpp_status.status.success() {
            eprintln!(
                "cjpegli failed: {}",
                String::from_utf8_lossy(&cpp_status.stderr)
            );
            continue;
        }

        let cpp_bytes = std::fs::metadata(&cpp_path).map(|m| m.len()).unwrap_or(0);

        // 2. Encode with Rust (XYB + hybrid)
        let rust_jpeg = {
            use zenjpeg::encode::trellis::HybridConfig;

            let config = EncoderConfig::xyb(q as f32, XybSubsampling::BQuarter)
                .hybrid_config(HybridConfig::default());
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .expect("encoder setup");
            enc.push_packed(pixels, Unstoppable).expect("push");
            enc.finish().expect("Rust encode")
        };

        let rust_path = format!("/tmp/rust_xyb_q{}.jpg", q);
        std::fs::write(&rust_path, &rust_jpeg).expect("Failed to write Rust JPEG");
        let rust_bytes = rust_jpeg.len() as u64;

        // 3. Compute butteraugli for both
        let cpp_butteraugli = compute_butteraugli(&image_path, &cpp_path);
        let rust_butteraugli = compute_butteraugli(&image_path, &rust_path);

        let size_diff = 100.0 * (rust_bytes as f64 - cpp_bytes as f64) / cpp_bytes as f64;
        let quality_diff = rust_butteraugli - cpp_butteraugli;

        println!(
            "{:<8} {:<12} {:<12} {:+.1}%{:<5} {:<12.4} {:<12.4} {:+.4}",
            q,
            cpp_bytes,
            rust_bytes,
            size_diff,
            "",
            cpp_butteraugli,
            rust_butteraugli,
            quality_diff
        );
    }
}

fn compute_butteraugli(original: &str, compressed: &str) -> f64 {
    // Use butteraugli_main if available, otherwise fall back to our Rust impl
    let output = Command::new("butteraugli")
        .args([original, compressed])
        .output();

    if let Ok(output) = output
        && output.status.success()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Parse butteraugli output (format: "distance")
        if let Some(line) = stdout.lines().next()
            && let Ok(val) = line.trim().parse::<f64>()
        {
            return val;
        }
    }

    // Fall back to Rust butteraugli
    compute_butteraugli_rust(original, compressed)
}

fn compute_butteraugli_rust(original_path: &str, compressed_path: &str) -> f64 {
    use butteraugli::ButteraugliParams;

    // Load original
    let loaded = zenjpeg_bench_utils::load_png(std::path::Path::new(original_path))
        .expect("Failed to load original PNG");
    let width = loaded.width();
    let height = loaded.height();
    let orig_pixels: Vec<rgb::RGB8> = loaded
        .buf()
        .iter()
        .map(|p| rgb::RGB8::new(p.r, p.g, p.b))
        .collect();
    let orig_img = imgref::Img::new(&orig_pixels[..], width, height);

    // Load compressed (JPEG) with ICC support for XYB
    let jpeg_data = std::fs::read(compressed_path).expect("read jpeg");
    let decoded = Decoder::new()
        .correct_color(Some(TargetColorSpace::Srgb))
        .decode(&jpeg_data, Unstoppable)
        .expect("decode jpeg");

    let dec_pixels: Vec<rgb::RGB8> = decoded
        .pixels_u8()
        .unwrap()
        .chunks_exact(3)
        .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
        .collect();
    let dec_img = imgref::Img::new(&dec_pixels[..], width, height);

    let params = ButteraugliParams::default();
    match butteraugli::butteraugli(orig_img, dec_img, &params) {
        Ok(result) => result.score,
        Err(_) => 999.0,
    }
}
