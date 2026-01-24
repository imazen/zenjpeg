//! Benchmark showing encoding performance at different image sizes
use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::time::Instant;

fn bench(width: usize, height: usize) -> (f64, usize) {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = 128;
        }
    }

    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);

    // Warmup
    for _ in 0..3 {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&data, Unstoppable).unwrap();
        let _ = enc.finish();
    }

    let mut times = Vec::new();
    let mut size = 0;
    for _ in 0..5 {
        let start = Instant::now();
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&data, Unstoppable).unwrap();
        let result = enc.finish().unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        size = result.len();
    }

    (times.iter().sum::<f64>() / 5.0, size)
}

fn main() {
    println!("JPEG Encoding Performance Benchmark\n");

    // 1MP image
    println!("=== 1MP (1024x1024 = 1,048,576 pixels) ===");
    let (ms, size) = bench(1024, 1024);
    let pixels = 1024.0 * 1024.0;
    println!(
        "encode(): {:.1} ms ({:.1} MP/s), {} KB\n",
        ms,
        pixels / (ms / 1000.0) / 1_000_000.0,
        size / 1024
    );

    // 4MP image
    println!("=== 4MP (2048x2048 = 4,194,304 pixels) ===");
    let (ms, size) = bench(2048, 2048);
    let pixels = 2048.0 * 2048.0;
    println!(
        "encode(): {:.1} ms ({:.1} MP/s), {} KB\n",
        ms,
        pixels / (ms / 1000.0) / 1_000_000.0,
        size / 1024
    );

    // 8MP image
    println!("=== 8MP (3840x2160 = 8,294,400 pixels) ===");
    let (ms, size) = bench(3840, 2160);
    let pixels = 3840.0 * 2160.0;
    println!(
        "encode(): {:.1} ms ({:.1} MP/s), {} KB",
        ms,
        pixels / (ms / 1000.0) / 1_000_000.0,
        size / 1024
    );
}
