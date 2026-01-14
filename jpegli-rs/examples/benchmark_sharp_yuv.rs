//! Benchmark comparing Sharp YUV vs standard chroma downsampling.
//!
//! Run with:
//!   cargo run --release --example benchmark_sharp_yuv
//!
//! This measures encode time and file size for both methods to quantify the
//! performance/quality tradeoff of Sharp YUV.

use enough::Unstoppable;
use jpegli::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::time::Instant;

fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            // Create a colorful gradient with some high-frequency edges
            let r = ((x * 255 / width) as u8).wrapping_add((y % 32) as u8 * 8);
            let g = ((y * 255 / height) as u8).wrapping_add((x % 32) as u8 * 8);
            let b = (((x + y) * 127 / (width + height)) as u8).wrapping_add(64);
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

fn benchmark_encode_std(
    name: &str,
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    data: &[u8],
    iterations: u32,
) -> (f64, usize) {
    let config = EncoderConfig::new()
        .quality(90.0)
        .ycbcr(subsampling);

    // Warmup
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(data, Unstoppable).unwrap();
    let _ = enc.finish().unwrap();

    let start = Instant::now();
    let mut output_size = 0;
    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(data, Unstoppable).unwrap();
        let output = enc.finish().unwrap();
        output_size = output.len();
    }
    let elapsed = start.elapsed();
    let ms_per_encode = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "{}: {:.2} ms/encode, {} bytes",
        name, ms_per_encode, output_size
    );
    (ms_per_encode, output_size)
}

fn benchmark_encode_sharp(
    name: &str,
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    data: &[u8],
    iterations: u32,
) -> (f64, usize) {
    let config = EncoderConfig::new()
        .quality(90.0)
        .ycbcr(subsampling)
        .sharp_yuv(true);

    // Warmup
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(data, Unstoppable).unwrap();
    let _ = enc.finish().unwrap();

    let start = Instant::now();
    let mut output_size = 0;
    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(data, Unstoppable).unwrap();
        let output = enc.finish().unwrap();
        output_size = output.len();
    }
    let elapsed = start.elapsed();
    let ms_per_encode = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

    println!(
        "{}: {:.2} ms/encode, {} bytes",
        name, ms_per_encode, output_size
    );
    (ms_per_encode, output_size)
}

fn main() {
    let sizes = [
        (512, 512, 100),  // Small, many iterations
        (1920, 1080, 20), // HD
        (4096, 2160, 5),  // 4K
    ];

    println!("Sharp YUV vs Standard Downsampling Benchmark");
    println!("=============================================\n");

    for (width, height, iterations) in sizes {
        let data = generate_test_image(width, height);
        let megapixels = (width * height) as f64 / 1_000_000.0;

        println!(
            "{}x{} ({:.2} MP), {} iterations:",
            width, height, megapixels, iterations
        );
        println!("-------------------------------------------------");

        // 4:2:0 comparison
        let (std_420_ms, std_420_size) = benchmark_encode_std(
            "Standard 4:2:0",
            width as u32,
            height as u32,
            ChromaSubsampling::Quarter,
            &data,
            iterations,
        );

        let (sharp_420_ms, sharp_420_size) = benchmark_encode_sharp(
            "Sharp    4:2:0",
            width as u32,
            height as u32,
            ChromaSubsampling::Quarter,
            &data,
            iterations,
        );

        let slowdown_420 = ((sharp_420_ms / std_420_ms) - 1.0) * 100.0;
        let size_diff_420 = ((sharp_420_size as f64 / std_420_size as f64) - 1.0) * 100.0;
        println!(
            "  -> Sharp YUV is {:.1}% slower, {:.1}% size diff\n",
            slowdown_420, size_diff_420
        );

        // 4:2:2 comparison
        let (std_422_ms, std_422_size) = benchmark_encode_std(
            "Standard 4:2:2",
            width as u32,
            height as u32,
            ChromaSubsampling::HalfHorizontal,
            &data,
            iterations,
        );

        let (sharp_422_ms, sharp_422_size) = benchmark_encode_sharp(
            "Sharp    4:2:2",
            width as u32,
            height as u32,
            ChromaSubsampling::HalfHorizontal,
            &data,
            iterations,
        );

        let slowdown_422 = ((sharp_422_ms / std_422_ms) - 1.0) * 100.0;
        let size_diff_422 = ((sharp_422_size as f64 / std_422_size as f64) - 1.0) * 100.0;
        println!(
            "  -> Sharp YUV is {:.1}% slower, {:.1}% size diff\n",
            slowdown_422, size_diff_422
        );

        println!();
    }

    // Summary with throughput
    println!("Summary (4K 4:2:0):");
    println!("-------------------");
    let data = generate_test_image(4096, 2160);
    let megapixels = 4096.0 * 2160.0 / 1_000_000.0;

    let (std_ms, _) = benchmark_encode_std("Standard", 4096, 2160, ChromaSubsampling::Quarter, &data, 10);
    let (sharp_ms, _) =
        benchmark_encode_sharp("Sharp   ", 4096, 2160, ChromaSubsampling::Quarter, &data, 10);

    let std_mpps = megapixels / (std_ms / 1000.0);
    println!("\nStandard throughput: {:.1} MP/s", std_mpps);

    let sharp_mpps = megapixels / (sharp_ms / 1000.0);
    let slowdown = ((sharp_ms / std_ms) - 1.0) * 100.0;
    println!("Sharp YUV throughput: {:.1} MP/s", sharp_mpps);
    println!("\nSharp YUV overhead: {:.1}% slower encoding", slowdown);
}
