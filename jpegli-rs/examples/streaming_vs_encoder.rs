//! Benchmark comparing JpegEncoder.encode() vs row-by-row performance.
//!
//! Run with: cargo run --release --example streaming_vs_encoder

use jpegli::{JpegEncoder, Quality, Subsampling};
use std::time::{Duration, Instant};

fn format_throughput(bytes: usize, duration: Duration) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    let secs = duration.as_secs_f64();
    format!("{:.1} MB/s", mb / secs)
}

fn benchmark<F>(name: &str, iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..2 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let total = start.elapsed();
    let avg = total / iterations as u32;

    println!("  {}: {:?} avg ({} iterations)", name, avg, iterations);
    avg
}

fn run_comparison(width: u32, height: u32, subsampling: Subsampling, iterations: usize) {
    let pixels: Vec<u8> = (0..(width * height * 3) as usize)
        .map(|i| ((i * 17 + i / 256) % 256) as u8)
        .collect();

    let input_size = pixels.len();
    let quality = Quality::from_quality(85.0);

    println!(
        "\n{}x{} {:?} ({:.1} MB input)",
        width,
        height,
        subsampling,
        input_size as f64 / 1024.0 / 1024.0
    );
    println!("{}", "-".repeat(60));

    // Encoder (deprecated, uses strip backend internally)
    let enc_time = benchmark("Encoder", iterations, || {
        #[allow(deprecated)]
        let _jpeg = JpegEncoder::new(width, height)
            .quality(quality)
            .subsampling(subsampling)
            .encode(&pixels)
            .unwrap();
    });

    // JpegEncoder.encode()
    let stream_all_time = benchmark("JpegEncoder.encode()", iterations, || {
        let _jpeg = JpegEncoder::new(width, height)
            .quality(quality)
            .subsampling(subsampling)
            .encode(&pixels)
            .unwrap();
    });

    // JpegEncoder row-by-row
    let stream_rows_time = benchmark("JpegEncoder (row-by-row)", iterations, || {
        let mut encoder = JpegEncoder::new(width, height)
            .quality(quality)
            .subsampling(subsampling)
            .start()
            .unwrap();

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            encoder.push_row(&pixels[start..start + row_size]).unwrap();
        }
        let _jpeg = encoder.finish().unwrap();
    });

    // Summary
    println!();
    println!("  Throughput:");
    println!(
        "    Encoder:             {}",
        format_throughput(input_size, enc_time)
    );
    println!(
        "    JpegEncoder:    {}",
        format_throughput(input_size, stream_all_time)
    );
    println!(
        "    Streaming (rows):    {}",
        format_throughput(input_size, stream_rows_time)
    );

    // Relative performance
    let baseline = enc_time.as_nanos() as f64;
    println!();
    println!("  Relative to Encoder:");
    println!(
        "    JpegEncoder:    {:.1}%",
        (stream_all_time.as_nanos() as f64 / baseline) * 100.0
    );
    println!(
        "    Streaming (rows):    {:.1}%",
        (stream_rows_time.as_nanos() as f64 / baseline) * 100.0
    );

    // Verify outputs match
    #[allow(deprecated)]
    let enc_jpeg = JpegEncoder::new(width, height)
        .quality(quality)
        .subsampling(subsampling)
        .encode(&pixels)
        .unwrap();

    let stream_jpeg = JpegEncoder::new(width, height)
        .quality(quality)
        .subsampling(subsampling)
        .encode(&pixels)
        .unwrap();

    if enc_jpeg == stream_jpeg {
        println!("\n  ✓ Outputs match ({} bytes)", enc_jpeg.len());
    } else {
        println!(
            "\n  ✗ Outputs differ! Encoder: {} bytes, Streaming: {} bytes",
            enc_jpeg.len(),
            stream_jpeg.len()
        );
    }
}

fn main() {
    println!("Encoder vs JpegEncoder Performance Comparison");
    println!("==================================================");

    // Test various sizes
    let configs = [
        (640, 480, Subsampling::S420, 50),   // VGA
        (1920, 1080, Subsampling::S420, 20), // 1080p
        (1920, 1080, Subsampling::S444, 20), // 1080p 4:4:4
        (3840, 2160, Subsampling::S420, 5),  // 4K
        (256, 256, Subsampling::S420, 100),  // Small
        (123, 87, Subsampling::S420, 100),   // Non-aligned
    ];

    for (width, height, subsampling, iterations) in configs {
        run_comparison(width, height, subsampling, iterations);
    }

    // Memory estimate comparison
    println!("\n\nMemory Estimates (JpegEncoder)");
    println!("====================================");
    for (name, w, h, sub) in [
        ("VGA", 640, 480, Subsampling::S420),
        ("1080p", 1920, 1080, Subsampling::S420),
        ("4K", 3840, 2160, Subsampling::S420),
        ("8K", 7680, 4320, Subsampling::S420),
    ] {
        let estimate = JpegEncoder::new(w, h)
            .subsampling(sub)
            .estimate_memory_usage();
        let input_size = w as usize * h as usize * 3;
        println!(
            "  {}: {:.1} MB estimated (vs {:.1} MB input)",
            name,
            estimate as f64 / 1024.0 / 1024.0,
            input_size as f64 / 1024.0 / 1024.0
        );
    }
}
