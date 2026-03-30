//! Latency distribution benchmark for proxy server scenarios.
//!
//! Measures per-decode latency percentiles (p50, p95, p99) under varying
//! concurrent load. Uses the `DecodePool` API for adaptive threading.
//!
//! Strategies tested:
//! - `seq`: num_threads(1), concurrent OS threads
//! - `par-global`: decode() with default rayon global pool, concurrent callers
//! - `pool-N`: DecodePool with parallel_threshold=N
//!
//! Run:
//! ```sh
//! cargo run --release --features decoder,parallel -p zenjpeg --example latency_bench
//! ```

use enough::Unstoppable;
use std::sync::{Arc, Barrier};
use std::time::Instant;
use zenjpeg::decode::{ChromaUpsampling, DecodePool, Decoder};
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn create_test_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let block_hash = bx
                .wrapping_mul(2654435761)
                .wrapping_add(by.wrapping_mul(40503));
            let block_type = block_hash % 4;
            let px = x as u32;
            let py = y as u32;
            let mut h = px
                .wrapping_mul(374761393)
                .wrapping_add(py.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            let noise = (h >> 24) as u8;
            match block_type {
                0 => {
                    let bias = ((bx.wrapping_mul(17) ^ by.wrapping_mul(31)) & 0xFF) as u8;
                    data[idx] = bias.wrapping_add(noise >> 2);
                    data[idx + 1] = bias.wrapping_add(noise >> 1);
                    data[idx + 2] = bias.wrapping_add(noise >> 3);
                }
                1 => {
                    data[idx] = ((x * 255) / width as usize) as u8;
                    data[idx + 1] = ((y * 255) / height as usize) as u8;
                    data[idx + 2] = noise >> 2;
                }
                2 => {
                    let edge = if (x % 8 < 4) ^ (y % 8 < 4) { 200 } else { 55 };
                    data[idx] = edge;
                    data[idx + 1] = edge.wrapping_add(noise >> 4);
                    data[idx + 2] = 255u8.wrapping_sub(edge);
                }
                _ => {
                    data[idx] = noise;
                    data[idx + 1] = noise.wrapping_mul(3);
                    data[idx + 2] = noise.wrapping_mul(7);
                }
            }
        }
    }

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&data, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Per-decode latency in microseconds
type LatencyUs = u64;

struct LatencyStats {
    #[allow(dead_code)]
    count: usize,
    p50: f64,
    p95: f64,
    p99: f64,
    mean: f64,
}

fn compute_stats(mut latencies: Vec<LatencyUs>) -> LatencyStats {
    latencies.sort_unstable();
    let n = latencies.len();
    if n == 0 {
        return LatencyStats {
            count: 0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            mean: 0.0,
        };
    }
    let p = |pct: f64| -> f64 {
        let idx = ((pct / 100.0) * (n - 1) as f64) as usize;
        latencies[idx.min(n - 1)] as f64 / 1000.0 // us -> ms
    };
    let mean = latencies.iter().sum::<u64>() as f64 / n as f64 / 1000.0;
    LatencyStats {
        count: n,
        p50: p(50.0),
        p95: p(95.0),
        p99: p(99.0),
        mean,
    }
}

/// Decode strategy
#[derive(Clone)]
enum Strategy {
    /// Sequential (num_threads=1), no pool
    Sequential,
    /// Parallel (rayon global pool), no pool
    ParGlobal,
    /// DecodePool with given threshold
    Pool(Arc<DecodePool>),
}

fn decode_one(data: &[u8], decoder: &Decoder, strategy: &Strategy) -> LatencyUs {
    let start = Instant::now();

    match strategy {
        Strategy::Sequential => {
            let _ = decoder
                .clone()
                .num_threads(1)
                .decode(data, Unstoppable)
                .expect("decode");
        }
        Strategy::ParGlobal => {
            let _ = decoder.decode(data, Unstoppable).expect("decode");
        }
        Strategy::Pool(pool) => {
            let _ = decoder.request(data).pool(pool).decode().expect("decode");
        }
    }

    start.elapsed().as_micros() as LatencyUs
}

/// Run a strategy at a given concurrency, collecting per-decode latencies.
fn run_bench(
    concurrency: usize,
    target_secs: f64,
    data: &[u8],
    decoder: &Decoder,
    strategy: &Strategy,
) -> Vec<LatencyUs> {
    if concurrency == 0 {
        return vec![];
    }

    // Warmup + estimate iterations
    let _ = decode_one(data, decoder, strategy);
    let t0 = Instant::now();
    let _ = decode_one(data, decoder, strategy);
    let single_us = t0.elapsed().as_micros() as f64;
    let per_thread = ((target_secs * 1_000_000.0 / single_us) as usize / concurrency).max(3);

    let data = Arc::new(data.to_vec());
    let decoder = decoder.clone();
    let barrier = Arc::new(Barrier::new(concurrency + 1));

    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let data = Arc::clone(&data);
            let decoder = decoder.clone();
            let strategy = strategy.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                // Warmup
                let _ = decode_one(&data, &decoder, &strategy);

                barrier.wait(); // sync start

                let mut latencies = Vec::with_capacity(per_thread);
                for _ in 0..per_thread {
                    latencies.push(decode_one(&data, &decoder, &strategy));
                }

                barrier.wait(); // sync end
                latencies
            })
        })
        .collect();

    barrier.wait(); // start
    barrier.wait(); // end

    let mut all_latencies = Vec::new();
    for h in handles {
        all_latencies.extend(h.join().unwrap());
    }
    all_latencies
}

fn main() {
    let num_cpus = rayon::current_num_threads();
    let physical = num_cpus / 2;
    println!("System: {} logical, {} physical cores", num_cpus, physical);
    println!("Rayon global pool: {} threads\n", num_cpus);

    let target_secs = 3.0;

    // Shared decoder configs — reused across all strategies
    let decoder_box = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .chroma_upsampling(ChromaUpsampling::NearestNeighbor);

    let decoder_fancy = Decoder::new().output_format(PixelFormat::Rgb);

    // Shared pools (created once, reused)
    let pool_2 = Arc::new(DecodePool::new().parallel_threshold(2));
    let pool_4 = Arc::new(DecodePool::new().parallel_threshold(4));
    let pool_phy = Arc::new(DecodePool::new().parallel_threshold(physical));

    for &(width, height) in &[(1024u32, 1024), (2048, 2048)] {
        let mpix = (width as f64 * height as f64) / 1e6;
        let jpeg = create_test_jpeg(width, height);
        println!(
            "=== {}x{} ({:.1} MP, JPEG {} KB) ===\n",
            width,
            height,
            mpix,
            jpeg.len() / 1024
        );

        let strategies: Vec<(&str, &Decoder, Strategy)> = vec![
            ("seq", &decoder_box, Strategy::Sequential),
            ("seq-fancy", &decoder_fancy, Strategy::Sequential),
            ("par-global", &decoder_box, Strategy::ParGlobal),
            ("par-fancy", &decoder_fancy, Strategy::ParGlobal),
            ("pool-2", &decoder_box, Strategy::Pool(Arc::clone(&pool_2))),
            ("pool-4", &decoder_box, Strategy::Pool(Arc::clone(&pool_4))),
            (
                "pool-phy",
                &decoder_box,
                Strategy::Pool(Arc::clone(&pool_phy)),
            ),
            (
                "pool-4-fancy",
                &decoder_fancy,
                Strategy::Pool(Arc::clone(&pool_4)),
            ),
        ];

        let conc_levels = [1, 2, 4, 8, 16, 24, 32]
            .into_iter()
            .filter(|&c| c <= num_cpus)
            .collect::<Vec<_>>();

        // Header
        println!(
            "{:<14} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Strategy", "Conc", "MP/s", "img/s", "mean", "p50", "p95", "p99"
        );
        println!(
            "{:<14} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "", "", "", "", "(ms)", "(ms)", "(ms)", "(ms)"
        );
        println!("{}", "-".repeat(78));

        for &(name, decoder, ref strategy) in &strategies {
            for &conc in &conc_levels {
                let latencies = run_bench(conc, target_secs, &jpeg, decoder, strategy);
                let stats = compute_stats(latencies);

                // Compute throughput from mean latency and concurrency
                let actual_ips = conc as f64 / (stats.mean / 1000.0);
                let actual_mps = actual_ips * mpix;

                println!(
                    "{:<14} {:>5} {:>8.0} {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                    name, conc, actual_mps, actual_ips, stats.mean, stats.p50, stats.p95, stats.p99,
                );
            }
            println!();
        }
    }
}
