//! Q100-specific comparison test for fast iteration during debugging.
//!
//! Usage: cargo test --release -p jpegli --test q100_comparison -- --nocapture --ignored

use jpegli::types::{JpegMode, PixelFormat};
use jpegli::{Encoder, Quality};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn find_cjpegli() -> Option<PathBuf> {
    jpegli::test_utils::find_cjpegli()
}

fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let decoder = jpeg_decoder::Decoder::new(data);
    let mut decoder = decoder;
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    Some((pixels, info.width as u32, info.height as u32))
}

fn compute_dssim(orig: &[u8], w1: u32, h1: u32, comp: &[u8], w2: u32, h2: u32) -> f64 {
    use rgb::RGBA;
    if w1 != w2 || h1 != h2 {
        return f64::MAX;
    }
    let attr = dssim::Dssim::new();
    let orig_rgba: Vec<RGBA<u8>> = orig
        .chunks(3)
        .map(|c| RGBA::new(c[0], c[1], c[2], 255))
        .collect();
    let comp_rgba: Vec<RGBA<u8>> = comp
        .chunks(3)
        .map(|c| RGBA::new(c[0], c[1], c[2], 255))
        .collect();
    let orig_img = attr
        .create_image_rgba(&orig_rgba, w1 as usize, h1 as usize)
        .unwrap();
    let comp_img = attr
        .create_image_rgba(&comp_rgba, w2 as usize, h2 as usize)
        .unwrap();
    attr.compare(&orig_img, comp_img).0.into()
}

#[test]
#[ignore]
fn test_q100_rust_vs_cpp() {
    let cjpegli_path = match find_cjpegli() {
        Some(p) => p,
        None => {
            println!("Skipping: cjpegli not found");
            return;
        }
    };

    // Use a small test image for speed
    let test_img = jpegli::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !test_img.exists() {
        println!("Skipping: test image not found. Set JPEGLI_TESTDATA env var.");
        return;
    }
    let test_img = test_img.to_str().unwrap();

    // Load PNG
    let png_data = std::fs::read(test_img).unwrap();
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let width = info.width;
    let height = info.height;

    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    println!("\n=== Q100 Rust vs C++ Comparison ===");
    println!("Image: {}x{}", width, height);

    // Rust Q100 encoding - BASELINE (sequential) mode
    let rust_start = Instant::now();
    let rust_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(100.0))
        .optimize_huffman(true)
        .mode(JpegMode::Baseline)
        .encode(&rgb)
        .expect("encode");
    let rust_time = rust_start.elapsed();

    // C++ Q100 encoding - BASELINE (sequential) mode
    let tmp_png = "/tmp/q100_test_input.png";
    std::fs::write(tmp_png, &png_data).unwrap();
    let cpp_out = "/tmp/q100_cpp_output.jpg";
    let cpp_start = Instant::now();
    let status = Command::new(&cjpegli_path)
        .args([tmp_png, cpp_out, "-q", "100", "--progressive_level=0"])
        .output()
        .unwrap();
    let cpp_time = cpp_start.elapsed();

    if !status.status.success() {
        println!(
            "C++ encoding failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );
        return;
    }
    let cpp_jpeg = std::fs::read(cpp_out).unwrap();

    // Decode and compute quality
    let (rust_decoded, rw, rh) = decode_jpeg(&rust_jpeg).unwrap();
    let (cpp_decoded, cw, ch) = decode_jpeg(&cpp_jpeg).unwrap();

    let rust_dssim = compute_dssim(&rgb, width, height, &rust_decoded, rw, rh);
    let cpp_dssim = compute_dssim(&rgb, width, height, &cpp_decoded, cw, ch);

    // Results
    let size_diff = (rust_jpeg.len() as f64 / cpp_jpeg.len() as f64 - 1.0) * 100.0;
    let dssim_diff = (rust_dssim / cpp_dssim - 1.0) * 100.0;

    println!(
        "\n  Rust: {:6} bytes, DSSIM {:.6}, {:3.1}ms",
        rust_jpeg.len(),
        rust_dssim,
        rust_time.as_secs_f64() * 1000.0
    );
    println!(
        "  C++:  {:6} bytes, DSSIM {:.6}, {:3.1}ms",
        cpp_jpeg.len(),
        cpp_dssim,
        cpp_time.as_secs_f64() * 1000.0
    );
    println!(
        "\n  Size diff:  {:+.1}% (Rust {} C++)",
        size_diff,
        if size_diff > 0.0 {
            "larger than"
        } else {
            "smaller than"
        }
    );
    println!(
        "  DSSIM diff: {:+.1}% (Rust {} C++)",
        dssim_diff,
        if dssim_diff > 0.0 {
            "worse than"
        } else {
            "better than"
        }
    );

    // Save for manual inspection
    std::fs::write("/tmp/q100_rust_output.jpg", &rust_jpeg).unwrap();
    println!("\n  Saved: /tmp/q100_rust_output.jpg, /tmp/q100_cpp_output.jpg");
}
