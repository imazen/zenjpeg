//! Compare Rust vs C++ FFI encoding sizes.

use jpegli_bench_utils::{
    ChromaSubsampling, ColorMode, EncoderConfig, EncoderImpl, ImageData, ScanMode,
};
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let png_path = args.get(1).expect("need image path");
    let quality: u8 = args
        .get(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(75);

    // Load PNG
    let file = fs::File::open(png_path).expect("open png");
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().expect("read png info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("read frame");

    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        _ => panic!("unsupported color type"),
    };

    let img = ImageData {
        name: "test".to_string(),
        width: info.width as usize,
        height: info.height as usize,
        pixels: rgb,
    };

    println!("Image: {}x{}", info.width, info.height);
    println!("Quality: {}\n", quality);

    // Test baseline mode
    println!("=== Baseline Mode ===");
    let cpp_baseline = EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Baseline)
        .subsampling(ChromaSubsampling::S420)
        .quality(quality)
        .encode(&img)
        .expect("C++ encode");

    let rust_baseline = EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Baseline)
        .subsampling(ChromaSubsampling::S420)
        .quality(quality)
        .encode(&img)
        .expect("Rust encode");

    println!("C++ FFI:  {} bytes", cpp_baseline.len());
    println!("Rust:     {} bytes", rust_baseline.len());
    println!(
        "Diff:     {:+.2}%\n",
        (rust_baseline.len() as f64 / cpp_baseline.len() as f64 - 1.0) * 100.0
    );

    // Test progressive mode
    println!("=== Progressive Mode ===");
    let cpp_prog = EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(ChromaSubsampling::S420)
        .quality(quality)
        .encode(&img)
        .expect("C++ encode");

    let rust_prog = EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(ChromaSubsampling::S420)
        .quality(quality)
        .encode(&img)
        .expect("Rust encode");

    println!("C++ FFI:  {} bytes", cpp_prog.len());
    println!("Rust:     {} bytes", rust_prog.len());
    println!(
        "Diff:     {:+.2}%",
        (rust_prog.len() as f64 / cpp_prog.len() as f64 - 1.0) * 100.0
    );
}
