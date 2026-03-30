//! System throughput benchmark for proxy image server scenarios.
//!
//! Measures total decode throughput (images/second) under saturated load
//! with different threading strategies. Sweeps concurrency from 1 to N_cores
//! to find peak throughput for each strategy.
//!
//! Strategies tested:
//! - `seq`: num_threads(1), N concurrent OS threads (no rayon)
//! - `par`: decode() with rayon, N concurrent callers sharing global pool
//! - `wave`: wave scanline_reader() with rayon, N concurrent callers
//! - Custom pool sizes: rayon pool of P threads, N concurrent callers
//!
//! Run with:
//! ```sh
//! cargo run --release --features decoder,parallel -p zenjpeg --example threading_throughput
//! ```

use enough::Unstoppable;
use std::sync::{Arc, Barrier};
use std::time::Instant;
use zenjpeg::decode::{ChromaUpsampling, Decoder};
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

/// Decode: full-frame parallel (rayon global pool)
fn decode_parallel(data: &[u8]) -> Vec<u8> {
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
    decoder
        .decode(data, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("pixels")
}

/// Decode: sequential (no rayon)
fn decode_sequential(data: &[u8]) -> Vec<u8> {
    use zenjpeg::decoder::PixelFormat;
    let decoder = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .num_threads(1)
        .chroma_upsampling(ChromaUpsampling::NearestNeighbor);
    decoder
        .decode(data, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("pixels")
}

/// Decode: wave-parallel scanline reader (rayon global pool)
fn decode_wave(data: &[u8]) -> Vec<u8> {
    let decoder = Decoder::new().chroma_upsampling(ChromaUpsampling::NearestNeighbor);
    let mut reader = decoder.scanline_reader(data).expect("scanline_reader");
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut pixels = vec![0u8; w * h * 3];
    let mut rows_read = 0;
    while rows_read < h {
        let remaining = h - rows_read;
        let output = imgref::ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
        rows_read += reader.read_rows_rgb8(output).expect("read");
    }
    pixels
}

/// Measure throughput: N concurrent callers each running decode_fn in a loop.
/// Returns (images/sec, avg_latency_ms).
fn measure_throughput(
    concurrency: usize,
    data: &[u8],
    decode_fn: fn(&[u8]) -> Vec<u8>,
    target_secs: f64,
) -> (f64, f64) {
    if concurrency == 0 {
        return (0.0, 0.0);
    }

    // Estimate iterations from a single warmup decode
    let t0 = Instant::now();
    let _ = decode_fn(data);
    let single_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Each thread runs enough iterations to fill target_secs
    let per_thread = ((target_secs / (single_ms / 1000.0)) as usize / concurrency).max(2);

    let data = Arc::new(data.to_vec());
    let barrier = Arc::new(Barrier::new(concurrency + 1));

    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let data = Arc::clone(&data);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                // Warmup
                let _ = decode_fn(&data);
                barrier.wait();
                for _ in 0..per_thread {
                    let _ = decode_fn(&data);
                }
                barrier.wait();
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    barrier.wait();
    let elapsed = start.elapsed();

    for h in handles {
        h.join().unwrap();
    }

    let total_images = concurrency * per_thread;
    let ips = total_images as f64 / elapsed.as_secs_f64();
    let avg_lat = elapsed.as_secs_f64() * 1000.0 / (total_images as f64 / concurrency as f64);
    (ips, avg_lat)
}

/// Same as measure_throughput but uses a custom rayon ThreadPool.
fn measure_throughput_custom_pool(
    concurrency: usize,
    pool_threads: usize,
    data: &[u8],
    target_secs: f64,
) -> (f64, f64) {
    if concurrency == 0 || pool_threads == 0 {
        return (0.0, 0.0);
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .build()
        .expect("build pool");

    // Warmup + estimate
    let t0 = Instant::now();
    pool.install(|| {
        let _ = decode_parallel(data);
    });
    let single_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let per_thread = ((target_secs / (single_ms / 1000.0)) as usize / concurrency).max(2);

    let data = Arc::new(data.to_vec());
    let pool = Arc::new(pool);
    let barrier = Arc::new(Barrier::new(concurrency + 1));

    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let data = Arc::clone(&data);
            let pool = Arc::clone(&pool);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                // Warmup
                pool.install(|| {
                    let _ = decode_parallel(&data);
                });
                barrier.wait();
                for _ in 0..per_thread {
                    pool.install(|| {
                        let _ = decode_parallel(&data);
                    });
                }
                barrier.wait();
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    barrier.wait();
    let elapsed = start.elapsed();

    for h in handles {
        h.join().unwrap();
    }

    let total_images = concurrency * per_thread;
    let ips = total_images as f64 / elapsed.as_secs_f64();
    let avg_lat = elapsed.as_secs_f64() * 1000.0 / (total_images as f64 / concurrency as f64);
    (ips, avg_lat)
}

struct Row {
    strategy: String,
    concurrency: usize,
    pool_threads: usize,
    ips: f64,
    latency_ms: f64,
    peak_mem_mb: f64,
    mpixels: f64, // megapixels per image
}

impl Row {
    fn mp_per_sec(&self) -> f64 {
        self.ips * self.mpixels
    }
}

fn main() {
    let num_cpus = rayon::current_num_threads();
    let physical_cores = num_cpus / 2; // Assume SMT
    println!(
        "System: {} logical cores, {} physical (assumed)",
        num_cpus, physical_cores
    );
    println!("Rayon global pool: {} threads", num_cpus);
    println!();

    let target_secs = 3.0; // Each measurement runs for ~3 seconds

    for &(width, height) in &[(512u32, 512), (1024, 1024), (2048, 2048), (4096, 4096)] {
        let mpix = (width as f64 * height as f64) / 1e6;
        let rgb_mb = (width as f64 * height as f64 * 3.0) / (1024.0 * 1024.0);

        println!(
            "=== {}x{} ({:.1} MP, {:.1} MB RGB) ===",
            width, height, mpix, rgb_mb
        );

        let jpeg = create_test_jpeg(width, height);
        println!("JPEG: {} KB\n", jpeg.len() / 1024);

        let mut rows: Vec<Row> = Vec::new();

        // Concurrency levels to test
        let conc_levels: Vec<usize> = [1, 2, 4, 8, 12, 16, 24, 32]
            .iter()
            .copied()
            .filter(|&c| c <= num_cpus)
            .collect();

        // === Strategy: seq (OS threads, no rayon) ===
        for &conc in &conc_levels {
            let (ips, lat) = measure_throughput(conc, &jpeg, decode_sequential, target_secs);
            rows.push(Row {
                strategy: "seq".into(),
                concurrency: conc,
                pool_threads: 0,
                ips,
                latency_ms: lat,
                peak_mem_mb: conc as f64 * rgb_mb,
                mpixels: mpix,
            });
        }

        // === Strategy: par (full-buf, shared global rayon pool) ===
        for &conc in &conc_levels {
            let (ips, lat) = measure_throughput(conc, &jpeg, decode_parallel, target_secs);
            rows.push(Row {
                strategy: "par".into(),
                concurrency: conc,
                pool_threads: num_cpus,
                ips,
                latency_ms: lat,
                peak_mem_mb: conc as f64 * rgb_mb * 2.0,
                mpixels: mpix,
            });
        }

        // === Strategy: wave (scanline, shared global rayon pool) ===
        for &conc in &conc_levels {
            let (ips, lat) = measure_throughput(conc, &jpeg, decode_wave, target_secs);
            let wave_buf_mb = 6.0_f64.min(rgb_mb);
            rows.push(Row {
                strategy: "wave".into(),
                concurrency: conc,
                pool_threads: num_cpus,
                ips,
                latency_ms: lat,
                peak_mem_mb: conc as f64 * (wave_buf_mb + rgb_mb),
                mpixels: mpix,
            });
        }

        // === Strategy: par with custom pool sizes ===
        for &pool_sz in &[4, 8, 16] {
            if pool_sz >= num_cpus {
                continue;
            }
            for &conc in &[1, 2, 4, 8, 16] {
                if conc > num_cpus {
                    continue;
                }
                let (ips, lat) = measure_throughput_custom_pool(conc, pool_sz, &jpeg, target_secs);
                rows.push(Row {
                    strategy: format!("par-{}t", pool_sz),
                    concurrency: conc,
                    pool_threads: pool_sz,
                    ips,
                    latency_ms: lat,
                    peak_mem_mb: conc as f64 * rgb_mb * 2.0,
                    mpixels: mpix,
                });
            }
        }

        // Find baseline (1x seq) and peak for each strategy
        let baseline_ips = rows
            .iter()
            .find(|r| r.strategy == "seq" && r.concurrency == 1)
            .map(|r| r.ips)
            .unwrap_or(1.0);

        // Sort by strategy then concurrency for display
        let strategies = ["seq", "par", "wave", "par-4t", "par-8t", "par-16t"];
        println!(
            "{:<10} {:>5} {:>6} {:>8} {:>9} {:>8} {:>8} {:>8}",
            "Strategy", "Conc", "Pool", "MP/s", "img/s", "lat(ms)", "mem(MB)", "vs seq1"
        );
        println!("{}", "-".repeat(68));

        for strat in &strategies {
            let mut strat_rows: Vec<&Row> = rows.iter().filter(|r| r.strategy == *strat).collect();
            if strat_rows.is_empty() {
                continue;
            }
            strat_rows.sort_by_key(|r| r.concurrency);

            let peak = strat_rows
                .iter()
                .max_by(|a, b| a.mp_per_sec().partial_cmp(&b.mp_per_sec()).unwrap())
                .unwrap();

            for r in &strat_rows {
                let marker =
                    if (r.mp_per_sec() - peak.mp_per_sec()).abs() / peak.mp_per_sec() < 0.03 {
                        " *"
                    } else {
                        ""
                    };
                let pool_str = if r.pool_threads == 0 {
                    "-".to_string()
                } else {
                    r.pool_threads.to_string()
                };
                println!(
                    "{:<10} {:>5} {:>6} {:>8.0} {:>9.1} {:>8.1} {:>8.1} {:>7.1}x{}",
                    r.strategy,
                    r.concurrency,
                    pool_str,
                    r.mp_per_sec(),
                    r.ips,
                    r.latency_ms,
                    r.peak_mem_mb,
                    r.ips / baseline_ips,
                    marker,
                );
            }
            println!();
        }

        // Summary: best throughput per strategy
        println!("Peak system throughput:");
        for strat in &strategies {
            if let Some(peak) = rows
                .iter()
                .filter(|r| r.strategy == *strat)
                .max_by(|a, b| a.mp_per_sec().partial_cmp(&b.mp_per_sec()).unwrap())
            {
                println!(
                    "  {:<10} {:>6.0} MP/s  ({:>7.0} img/s @ {}x conc, {:.0} MB, {:.1}ms lat)",
                    peak.strategy,
                    peak.mp_per_sec(),
                    peak.ips,
                    peak.concurrency,
                    peak.peak_mem_mb,
                    peak.latency_ms,
                );
            }
        }
        println!();
    }
}
