//! Benchmark showing auto-dispatch behavior for large images
use jpegli::{Encoder, PixelFormat, Quality};
use std::time::Instant;

fn bench(width: usize, height: usize, use_strip: bool) -> (f64, usize) {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    // Warmup
    for _ in 0..3 {
        let enc = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0));
        if use_strip {
            let _ = enc.encode_strip_based(&data);
        } else {
            let _ = enc.encode(&data);
        }
    }

    let mut times = Vec::new();
    let mut size = 0;
    for _ in 0..5 {
        let start = Instant::now();
        let enc = Encoder::new()
            .width(width as u32)
            .height(height as u32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0));
        let result = if use_strip {
            enc.encode_strip_based(&data).unwrap()
        } else {
            enc.encode(&data).unwrap()
        };
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        size = result.len();
    }

    (times.iter().sum::<f64>() / 5.0, size)
}

fn main() {
    println!("Auto-dispatch threshold: 2,000,000 pixels (2MP)");
    println!("Large images (>2MP) automatically use strip-based encoding\n");

    // 1MP image (below threshold) - encode() uses full-plane
    println!("=== 1MP (1024x1024 = 1,048,576 pixels) - BELOW THRESHOLD ===");
    let (full_ms, full_size) = bench(1024, 1024, false);
    let (strip_ms, strip_size) = bench(1024, 1024, true);
    let pixels = 1024.0 * 1024.0;
    println!(
        "encode() (full-plane): {:.1} ms ({:.1} MP/s), {} KB",
        full_ms,
        pixels / (full_ms / 1000.0) / 1_000_000.0,
        full_size / 1024
    );
    println!(
        "encode_strip_based():  {:.1} ms ({:.1} MP/s), {} KB",
        strip_ms,
        pixels / (strip_ms / 1000.0) / 1_000_000.0,
        strip_size / 1024
    );

    // 4MP image (above threshold) - encode() auto-dispatches to strip
    println!("\n=== 4MP (2048x2048 = 4,194,304 pixels) - ABOVE THRESHOLD ===");
    let (full_ms, full_size) = bench(2048, 2048, false);
    let (strip_ms, strip_size) = bench(2048, 2048, true);
    let pixels = 2048.0 * 2048.0;
    println!(
        "encode() (auto-strip): {:.1} ms ({:.1} MP/s), {} KB",
        full_ms,
        pixels / (full_ms / 1000.0) / 1_000_000.0,
        full_size / 1024
    );
    println!(
        "encode_strip_based():  {:.1} ms ({:.1} MP/s), {} KB",
        strip_ms,
        pixels / (strip_ms / 1000.0) / 1_000_000.0,
        strip_size / 1024
    );

    // 8MP image (well above threshold)
    println!("\n=== 8MP (3840x2160 = 8,294,400 pixels) - WELL ABOVE THRESHOLD ===");
    let (full_ms, full_size) = bench(3840, 2160, false);
    let (strip_ms, strip_size) = bench(3840, 2160, true);
    let pixels = 3840.0 * 2160.0;
    println!(
        "encode() (auto-strip): {:.1} ms ({:.1} MP/s), {} KB",
        full_ms,
        pixels / (full_ms / 1000.0) / 1_000_000.0,
        full_size / 1024
    );
    println!(
        "encode_strip_based():  {:.1} ms ({:.1} MP/s), {} KB",
        strip_ms,
        pixels / (strip_ms / 1000.0) / 1_000_000.0,
        strip_size / 1024
    );
}
