//! Benchmark Huffman table strategies runtime.
//!
//! Compares:
//! 1. Optimal: Two-pass, count all frequencies then build tables
//! 2. Pretrained: Use fixed trained tables (no frequency counting overhead)
//! 3. Partial@30%: Count frequencies from 30% of rows, build tables
//!
//! Run with: cargo run --release -p zenjpeg --features test-utils --example bench_huffman_strategies

use std::fs::{self, File};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use zenjpeg::encode::{Quality, StreamingEncoder};
use zenjpeg::huffman::trained::trained_tables_q85;
use zenjpeg::types::Subsampling;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const QUALITY: u8 = 85;
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

fn main() -> Result<()> {
    println!("=== Huffman Strategy Runtime Benchmark ===\n");
    println!("Quality: {}, Iterations: {} (after {} warmup)\n", QUALITY, BENCH_ITERS, WARMUP_ITERS);

    // Test CLIC 2025 images
    if let Ok(clic_dir) = find_clic_images() {
        println!("=== CLIC 2025 Images ===\n");
        let images: Vec<_> = fs::read_dir(&clic_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
            .take(10)
            .collect();

        if !images.is_empty() {
            let mut total_optimal = Duration::ZERO;
            let mut total_pretrained = Duration::ZERO;
            let mut total_partial = Duration::ZERO;
            let mut count = 0;

            println!("{:>12} {:>9}  {:>9} {:>9} {:>9}  {:>7} {:>7} {:>7}",
                "Image", "Size", "Optimal", "Pretrain", "Partial", "OptKB", "PreKB", "ParKB");
            println!("{}", "-".repeat(95));

            for entry in &images {
                let path = entry.path();
                let (w, h, pixels) = load_png(&path)?;
                let name: String = path.file_stem().unwrap().to_string_lossy().chars().take(12).collect();

                let r = bench_image(&pixels, w, h)?;

                println!(
                    "{:>12} {:>4}x{:<4}  {:>8.1}ms {:>8.1}ms {:>8.1}ms  {:>6} {:>6} {:>6}",
                    name, w, h,
                    r.optimal.as_secs_f64() * 1000.0,
                    r.pretrained.as_secs_f64() * 1000.0,
                    r.partial.as_secs_f64() * 1000.0,
                    r.optimal_size / 1024,
                    r.pretrained_size / 1024,
                    r.partial_size / 1024,
                );

                total_optimal += r.optimal;
                total_pretrained += r.pretrained;
                total_partial += r.partial;
                count += 1;
            }

            println!("\n{:>12} {:>9}  {:>8.1}ms {:>8.1}ms {:>8.1}ms",
                "AVERAGE", "",
                total_optimal.as_secs_f64() * 1000.0 / count as f64,
                total_pretrained.as_secs_f64() * 1000.0 / count as f64,
                total_partial.as_secs_f64() * 1000.0 / count as f64,
            );
        }
    }

    // Test synthetic 8K image
    println!("\n=== Synthetic 8K Image (7680x4320) ===\n");
    let (w, h) = (7680, 4320);
    let pixels = generate_synthetic_image(w, h);

    let r = bench_image(&pixels, w, h)?;
    println!(
        "8K synthetic  optimal:{:>6.1}ms  pretrained:{:>6.1}ms  partial:{:>6.1}ms",
        r.optimal.as_secs_f64() * 1000.0,
        r.pretrained.as_secs_f64() * 1000.0,
        r.partial.as_secs_f64() * 1000.0,
    );
    println!(
        "              sizes:   {:>6}KB             {:>6}KB            {:>6}KB",
        r.optimal_size / 1024,
        r.pretrained_size / 1024,
        r.partial_size / 1024,
    );

    // Also test 4K
    println!("\n=== Synthetic 4K Image (3840x2160) ===\n");
    let (w, h) = (3840, 2160);
    let pixels = generate_synthetic_image(w, h);

    let r = bench_image(&pixels, w, h)?;
    println!(
        "4K synthetic  optimal:{:>6.1}ms  pretrained:{:>6.1}ms  partial:{:>6.1}ms",
        r.optimal.as_secs_f64() * 1000.0,
        r.pretrained.as_secs_f64() * 1000.0,
        r.partial.as_secs_f64() * 1000.0,
    );
    println!(
        "              sizes:   {:>6}KB             {:>6}KB            {:>6}KB",
        r.optimal_size / 1024,
        r.pretrained_size / 1024,
        r.partial_size / 1024,
    );

    println!("\n=== Summary ===\n");
    println!("Pretrained vs Optimal: measures overhead of frequency counting + table building");
    println!("Partial vs Optimal: measures benefit of early transition (30% coverage)");

    Ok(())
}

struct BenchResult {
    optimal: Duration,
    pretrained: Duration,
    partial: Duration,
    optimal_size: usize,
    pretrained_size: usize,
    partial_size: usize,
}

fn bench_image(pixels: &[u8], w: u32, h: u32) -> Result<BenchResult> {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = encode_optimal(pixels, w, h)?;
        let _ = encode_pretrained(pixels, w, h)?;
        let _ = encode_partial(pixels, w, h, 0.30)?;
    }

    // Benchmark optimal (two-pass with optimize_huffman=true)
    let mut optimal_times = Vec::with_capacity(BENCH_ITERS);
    let mut optimal_size = 0;
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let out = encode_optimal(pixels, w, h)?;
        optimal_times.push(start.elapsed());
        optimal_size = out.len();
    }

    // Benchmark pretrained (fixed tables, no frequency counting)
    let mut pretrained_times = Vec::with_capacity(BENCH_ITERS);
    let mut pretrained_size = 0;
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let out = encode_pretrained(pixels, w, h)?;
        pretrained_times.push(start.elapsed());
        pretrained_size = out.len();
    }

    // Benchmark partial (30% coverage transition)
    let mut partial_times = Vec::with_capacity(BENCH_ITERS);
    let mut partial_size = 0;
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let out = encode_partial(pixels, w, h, 0.30)?;
        partial_times.push(start.elapsed());
        partial_size = out.len();
    }

    // Return median times
    optimal_times.sort();
    pretrained_times.sort();
    partial_times.sort();

    Ok(BenchResult {
        optimal: optimal_times[BENCH_ITERS / 2],
        pretrained: pretrained_times[BENCH_ITERS / 2],
        partial: partial_times[BENCH_ITERS / 2],
        optimal_size,
        pretrained_size,
        partial_size,
    })
}

fn encode_optimal(pixels: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let mut encoder = StreamingEncoder::new(w, h)
        .quality(Quality::ApproxJpegli(QUALITY as f32))
        .subsampling(Subsampling::S420)
        .optimize_huffman(true)
        .start()?;

    let stride = w as usize * 3;
    for y in 0..h {
        let row_start = y as usize * stride;
        let row_end = row_start + stride;
        encoder.push_rows(&pixels[row_start..row_end], 1)?;
    }

    Ok(encoder.finish()?)
}

fn encode_pretrained(pixels: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    // Use our corpus-trained tables instead of JPEG Annex K
    let tables = trained_tables_q85();

    let mut encoder = StreamingEncoder::new(w, h)
        .quality(Quality::ApproxJpegli(QUALITY as f32))
        .subsampling(Subsampling::S420)
        .custom_huffman_tables(tables)
        .start()?;

    let stride = w as usize * 3;
    for y in 0..h {
        let row_start = y as usize * stride;
        let row_end = row_start + stride;
        encoder.push_rows(&pixels[row_start..row_end], 1)?;
    }

    Ok(encoder.finish()?)
}

fn encode_partial(pixels: &[u8], w: u32, h: u32, coverage: f64) -> Result<Vec<u8>> {
    // Simulate bounded-memory streaming by setting a memory limit
    // that triggers transition at approximately the target coverage
    let bytes_per_row = w as usize * 3;
    let target_rows = (h as f64 * coverage) as usize;
    // Rough estimate: ~2 bytes per coefficient, 64 coeffs per 8x8 block
    let blocks_per_row = ((w + 7) / 8) as usize;
    let estimated_block_bytes = 128; // Conservative estimate
    let memory_limit = target_rows / 8 * blocks_per_row * estimated_block_bytes;

    let mut encoder = StreamingEncoder::new(w, h)
        .quality(Quality::ApproxJpegli(QUALITY as f32))
        .subsampling(Subsampling::S420)
        .memory_limit(memory_limit.max(1))
        .start()?;

    let stride = w as usize * 3;
    for y in 0..h {
        let row_start = y as usize * stride;
        let row_end = row_start + stride;
        encoder.push_rows(&pixels[row_start..row_end], 1)?;
    }

    Ok(encoder.finish()?)
}

fn generate_synthetic_image(w: u32, h: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (w * h * 3) as usize];

    // Generate a pattern with varying content (gradients + noise)
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;

            // Horizontal gradient
            let r = ((x as f32 / w as f32) * 255.0) as u8;
            // Vertical gradient
            let g = ((y as f32 / h as f32) * 255.0) as u8;
            // Diagonal pattern with some variation
            let b = (((x + y) % 256) as u8).wrapping_add(((x * y) % 64) as u8);

            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
        }
    }

    pixels
}

fn load_png(path: &PathBuf) -> Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(File::open(path)?);
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    let (w, h) = (info.width, info.height);
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return Err(format!("Unsupported: {:?}", info.color_type).into()),
    };

    Ok((w, h, pixels))
}

fn find_clic_images() -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".to_string());
    let candidates = [
        PathBuf::from(&home).join("work/codec-corpus/clic2025/final-test"),
        PathBuf::from(&home).join("work/codec-eval/codec-corpus/clic2025/final-test"),
        PathBuf::from(&home).join("work/codec-corpus/clic2025/validation"),
    ];

    for path in &candidates {
        if path.exists() {
            return Ok(path.clone());
        }
    }

    Err("No CLIC 2025 directory found".into())
}
