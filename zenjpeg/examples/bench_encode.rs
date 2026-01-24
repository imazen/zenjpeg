use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::time::Instant;

fn benchmark(
    width: usize,
    height: usize,
    restart_interval: u16,
    iterations: usize,
) -> (f64, usize) {
    // Create test image - moderate complexity
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create a gradient with some noise for realistic entropy
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 32) as u8) % 20;
            pixels[idx] = ((x * 255 / width) as u8).saturating_add(noise);
            pixels[idx + 1] = ((y * 255 / height) as u8).saturating_add(noise);
            pixels[idx + 2] = (((x + y) * 127 / (width + height)) as u8).saturating_add(noise);
        }
    }

    let config =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).restart_interval(restart_interval);

    // Warmup
    for _ in 0..3 {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        let _ = enc.finish().unwrap();
    }

    // Timed runs
    let start = Instant::now();
    let mut result_size = 0;
    for _ in 0..iterations {
        let mut enc = config
            .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&pixels, Unstoppable).unwrap();
        let result = enc.finish().unwrap();
        result_size = result.len();
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();
    let avg = elapsed.as_millis() as f64 / iterations as f64;
    (avg, result_size)
}

fn main() {
    println!("Full Encoder Benchmark - Sequential vs Parallel\n");
    println!("Note: parallel feature requires restart_interval > 0\n");

    let sizes = [(1024, 1024), (2048, 2048), (4096, 4096)];
    let iterations = 10;

    for (width, height) in sizes {
        let megapixels = (width * height) as f64 / 1_000_000.0;
        println!("=== {}x{} ({:.1} MP) ===", width, height, megapixels);

        // Sequential (no restart interval)
        let (seq_time, seq_size) = benchmark(width, height, 0, iterations);
        println!(
            "Sequential (restart=0):   {:7.2}ms, {:7} bytes, {:5.1} MP/s",
            seq_time,
            seq_size,
            megapixels / seq_time * 1000.0
        );

        // With restart interval (enables parallel when feature is on)
        for &restart in &[64, 256, 1024] {
            let (par_time, par_size) = benchmark(width, height, restart, iterations);
            let speedup = seq_time / par_time;
            let size_diff = (par_size as f64 - seq_size as f64) / seq_size as f64 * 100.0;
            println!(
                "Restart interval {:4}:    {:7.2}ms, {:7} bytes ({:+.2}%), {:5.1} MP/s, {:.2}x",
                restart,
                par_time,
                par_size,
                size_diff,
                megapixels / par_time * 1000.0,
                speedup
            );
        }
        println!();
    }
}
