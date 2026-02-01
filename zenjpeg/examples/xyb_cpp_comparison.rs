//! Compare XYB + hybrid output against C jpegli.
//!
//! Run with:
//! ```
//! cargo run --release --example xyb_cpp_comparison --features experimental-hybrid-trellis
//! ```

use enough::Unstoppable;
use std::process::Command;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

fn main() {
    let image_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png".to_string());

    // Load image
    let file = std::fs::File::open(&image_path).expect("Failed to open image");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    if info.color_type != png::ColorType::Rgb {
        eprintln!("Image must be RGB");
        return;
    }

    let pixels = &buf[..info.buffer_size()];
    let width = info.width as u32;
    let height = info.height as u32;

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
        #[cfg(feature = "experimental-hybrid-trellis")]
        let rust_jpeg = {
            use zenjpeg::hybrid::HybridConfig;

            let config = EncoderConfig::xyb(q as f32, XybSubsampling::BQuarter)
                .hybrid_config(HybridConfig::default());
            let mut enc = config
                .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
                .expect("encoder setup");
            enc.push_packed(pixels, Unstoppable).expect("push");
            enc.finish().expect("Rust encode")
        };

        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let rust_jpeg = {
            let config = EncoderConfig::xyb(q as f32, XybSubsampling::BQuarter);
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

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse butteraugli output (format: "distance")
            if let Some(line) = stdout.lines().next() {
                if let Ok(val) = line.trim().parse::<f64>() {
                    return val;
                }
            }
        }
    }

    // Fall back to Rust butteraugli
    compute_butteraugli_rust(original, compressed)
}

fn compute_butteraugli_rust(original_path: &str, compressed_path: &str) -> f64 {
    use butteraugli::{compute_butteraugli, ButteraugliParams};

    // Load original
    let file = std::fs::File::open(original_path).expect("open original");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read info");
    let mut orig_buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut orig_buf).expect("decode");
    let width = info.width as usize;
    let height = info.height as usize;
    let orig_pixels = &orig_buf[..info.buffer_size()];

    // Load compressed (JPEG) with ICC support for XYB
    let jpeg_data = std::fs::read(compressed_path).expect("read jpeg");
    let decoded = Decoder::new()
        .apply_icc(true)
        .decode(&jpeg_data)
        .expect("decode jpeg");

    let params = ButteraugliParams::default();
    match compute_butteraugli(orig_pixels, &decoded.data, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => 999.0,
    }
}
