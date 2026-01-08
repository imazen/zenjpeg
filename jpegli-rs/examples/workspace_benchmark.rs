//! Benchmark comparing encode with and without workspace reuse
//!
//! Note: Currently the workspace API validates correct usage but falls back
//! to the regular path. Full workspace integration requires refactoring the
//! encode pipeline to use slices instead of owned Vecs to avoid copy overhead.
//!
//! Usage:
//! ```bash
//! cargo run --release --example workspace_benchmark
//! ```

use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, EncoderWorkspace, Quality};
use jpegli_bench_utils::{ImageData, SyntheticPattern};
use std::time::Instant;

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

fn encode_without_workspace(image: &ImageData, quality: u8) -> Vec<u8> {
    Encoder::new()
        .width(image.width as u32)
        .height(image.height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .mode(JpegMode::Baseline) // Baseline uses workspace path
        .optimize_huffman(true)
        .subsampling(Subsampling::S420)
        .use_xyb(false)
        .encode(&image.pixels)
        .expect("encode failed")
}

fn encode_with_workspace(
    image: &ImageData,
    quality: u8,
    workspace: &mut EncoderWorkspace,
) -> Vec<u8> {
    Encoder::new()
        .width(image.width as u32)
        .height(image.height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .mode(JpegMode::Baseline) // Baseline uses workspace path
        .optimize_huffman(true)
        .subsampling(Subsampling::S420)
        .use_xyb(false)
        .encode_with_workspace(&image.pixels, workspace)
        .expect("encode failed")
}

fn main() {
    println!("=== Workspace Reuse Benchmark ===\n");

    // Test with 2K image
    let image = create_test_image(2048, 2048);
    let iterations = 20;
    let quality = 90;

    println!("Image size: {}x{}", image.width, image.height);
    println!("Iterations: {}", iterations);
    println!();

    // Create workspace FIRST (to avoid affecting "without" timing)
    let ws_start = Instant::now();
    let mut workspace = EncoderWorkspace::new(2048, 2048).expect("failed to create workspace");
    let ws_creation = ws_start.elapsed();
    println!(
        "Workspace creation time: {:.2}ms",
        ws_creation.as_secs_f64() * 1000.0
    );

    // Warmup both paths
    let _ = encode_without_workspace(&image, quality);
    let _ = encode_with_workspace(&image, quality, &mut workspace);

    // Benchmark with workspace FIRST (while workspace is in cache)
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_with_workspace(&image, quality, &mut workspace);
    }
    let time_with = start.elapsed();
    let avg_with = time_with.as_millis() as f64 / iterations as f64;

    // Benchmark without workspace
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = encode_without_workspace(&image, quality);
    }
    let time_without = start.elapsed();
    let avg_without = time_without.as_millis() as f64 / iterations as f64;

    println!();
    println!("Without workspace: {:.2}ms/encode", avg_without);
    println!("With workspace:    {:.2}ms/encode", avg_with);

    if avg_without > avg_with {
        let speedup = (avg_without - avg_with) / avg_without * 100.0;
        println!("Speedup:           {:.1}%", speedup);
    } else {
        println!("(No speedup detected)");
    }

    // Verify outputs are identical
    let output_without = encode_without_workspace(&image, quality);
    let output_with = encode_with_workspace(&image, quality, &mut workspace);
    if output_without == output_with {
        println!("\n✓ Output verification passed (identical JPEGs)");
    } else {
        println!(
            "\n✗ Output mismatch! Sizes: {} vs {}",
            output_without.len(),
            output_with.len()
        );
    }
}
