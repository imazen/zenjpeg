//! Quick 4K benchmark
use enough::Unstoppable;
use jpegli::encoder::{EncoderConfig, PixelLayout};
use std::time::Instant;

fn main() {
    let width = 3840;
    let height = 2160;

    // Create synthetic 4K image with realistic patterns
    println!("Creating {}x{} test image...", width, height);
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create gradient + noise pattern
            let noise = ((x * 7 + y * 13) % 256) as u8;
            data[idx] = ((x * 255) / width) as u8;
            data[idx + 1] = ((y * 255) / height) as u8;
            data[idx + 2] = noise;
        }
    }

    println!("Warming up...");
    for _ in 0..2 {
        let config = EncoderConfig::new().quality(90.0);
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&data, Unstoppable).unwrap();
        let _ = enc.finish().unwrap();
    }

    println!("\nBenchmarking 4K encoding ({}x{})...\n", width, height);

    for (name, quality) in [("q75", 75.0), ("q90", 90.0), ("q95", 95.0)] {
        let mut times = Vec::new();
        let iterations = 5;
        let config = EncoderConfig::new().quality(quality);

        for _ in 0..iterations {
            let start = Instant::now();
            let mut enc = config
                .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(&data, Unstoppable).unwrap();
            let result = enc.finish().unwrap();
            let elapsed = start.elapsed();
            times.push(elapsed.as_millis());

            if times.len() == 1 {
                let mpixels = (width * height) as f64 / 1_000_000.0;
                let mpps = mpixels / elapsed.as_secs_f64();
                println!(
                    "{}: {:.1} ms ({:.1} MP/s), size: {:.1} KB",
                    name,
                    elapsed.as_millis(),
                    mpps,
                    result.len() as f64 / 1024.0
                );
            }
        }

        let avg: u128 = times.iter().sum::<u128>() / iterations;
        let min = times.iter().min().unwrap();
        let max = times.iter().max().unwrap();
        println!("  avg: {} ms, min: {} ms, max: {} ms", avg, min, max);
    }
}
