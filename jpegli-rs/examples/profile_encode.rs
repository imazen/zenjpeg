//! Simple profiling target for flamegraph
//!
//! Usage:
//! ```bash
//! cargo flamegraph --release --example profile_encode -o flamegraph.svg
//! ```

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use jpegli_bench_utils::{ImageData, SyntheticPattern};

fn create_test_image(width: u32, height: u32) -> ImageData {
    let pattern = SyntheticPattern::Complex;
    let img = pattern.generate(width, height);
    ImageData {
        name: format!("complex_{}x{}", width, height),
        pixels: img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect(),
        width: width as usize,
        height: height as usize,
    }
}

fn encode_rust(image: &ImageData, quality: u8) -> Vec<u8> {
    Encoder::new()
        .width(image.width as u32)
        .height(image.height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .mode(JpegMode::Progressive)
        .optimize_huffman(true)
        .subsampling(Subsampling::S420)
        .use_xyb(false)
        .encode(&image.pixels)
        .expect("Rust encode failed")
}

fn main() {
    // 2K image for meaningful profiling
    let image = create_test_image(2048, 2048);

    // Encode multiple times for better sampling
    let iterations = 50;
    let mut total_bytes = 0usize;

    for _ in 0..iterations {
        let jpeg = encode_rust(&image, 90);
        total_bytes += jpeg.len();
    }

    println!(
        "Encoded {} iterations, total {} bytes",
        iterations, total_bytes
    );
}
