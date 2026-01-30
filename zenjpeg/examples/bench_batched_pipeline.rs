//! Benchmark: fused vs batched pipeline processing.
//!
//! Tests whether processing strips in batched stage order (all color, then all DCT,
//! then all quant) is faster than the fused per-strip order (color→AQ→DCT→quant).
//!
//! Usage:
//!   cargo run --release -p zenjpeg --features test-utils --example bench_batched_pipeline
//!   cargo run --release -p zenjpeg --features test-utils --example bench_batched_pipeline -- --ppm /tmp/test.ppm
//!   cargo run --release -p zenjpeg --features test-utils --example bench_batched_pipeline -- --width 2048 --height 2048
//!
//! For cachegrind:
//!   cargo build --release -p zenjpeg --features test-utils --example bench_batched_pipeline
//!   valgrind --tool=cachegrind ./target/release/examples/bench_batched_pipeline -- --cachegrind --batch 8

use std::env;
use std::time::Instant;

use zenjpeg::encode::strip::{BatchReuse, StripProcessor};
use zenjpeg::encode::encoder_types::Quality;
use zenjpeg::quant::{generate_quant_table_ex, ZeroBiasParams};
use zenjpeg::types::{ColorSpace, PixelFormat, Subsampling};

struct Config {
    width: usize,
    height: usize,
    ppm_path: Option<String>,
    cachegrind: bool,
    batch_only: Option<usize>,
    iterations: usize,
    warmup: usize,
}

fn parse_args() -> Config {
    let args: Vec<String> = env::args().collect();
    let mut cfg = Config {
        width: 4096,
        height: 2160,
        ppm_path: None,
        cachegrind: false,
        batch_only: None,
        iterations: 10,
        warmup: 3,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ppm" => {
                i += 1;
                cfg.ppm_path = Some(args[i].clone());
            }
            "--width" => {
                i += 1;
                cfg.width = args[i].parse().unwrap();
            }
            "--height" => {
                i += 1;
                cfg.height = args[i].parse().unwrap();
            }
            "--cachegrind" => {
                cfg.cachegrind = true;
                cfg.iterations = 1;
                cfg.warmup = 0;
            }
            "--batch" => {
                i += 1;
                cfg.batch_only = Some(args[i].parse().unwrap());
            }
            "--iterations" | "-n" => {
                i += 1;
                cfg.iterations = args[i].parse().unwrap();
            }
            "--" => {} // ignore separator
            _ => eprintln!("Unknown arg: {}", args[i]),
        }
        i += 1;
    }
    cfg
}

fn create_synthetic_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let noise = ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(13))) % 256) as u8;
            data[idx] = ((x * 255) / width.max(1)) as u8;
            data[idx + 1] = ((y * 255) / height.max(1)) as u8;
            data[idx + 2] = noise;
        }
    }
    data
}

fn load_ppm(path: &str) -> (usize, usize, Vec<u8>) {
    let data = std::fs::read(path).expect("Failed to read PPM");
    assert!(data[0] == b'P' && data[1] == b'6', "Not a P6 PPM");
    let mut pos = 2;

    let skip_ws = |data: &[u8], mut p: usize| -> usize {
        loop {
            while p < data.len() && data[p].is_ascii_whitespace() {
                p += 1;
            }
            if p < data.len() && data[p] == b'#' {
                while p < data.len() && data[p] != b'\n' {
                    p += 1;
                }
            } else {
                break;
            }
        }
        p
    };

    let read_int = |data: &[u8], mut p: usize| -> (usize, usize) {
        let start = p;
        while p < data.len() && data[p].is_ascii_digit() {
            p += 1;
        }
        let val: usize = std::str::from_utf8(&data[start..p]).unwrap().parse().unwrap();
        (val, p)
    };

    pos = skip_ws(&data, pos);
    let (width, p) = read_int(&data, pos);
    pos = skip_ws(&data, p);
    let (height, p) = read_int(&data, pos);
    pos = skip_ws(&data, p);
    let (_maxval, p) = read_int(&data, pos);
    pos = p + 1; // skip one whitespace

    let pixels = data[pos..pos + width * height * 3].to_vec();
    (width, height, pixels)
}

fn setup_processor(width: usize, height: usize) -> StripProcessor {
    let quality = Quality::ApproxJpegli(85.0);
    let subsampling = Subsampling::S420;
    let mut processor =
        StripProcessor::with_options(width, height, subsampling, PixelFormat::Rgb, Default::default(), 0)
            .expect("Failed to create processor");

    let is_420 = true;
    let y_quant = generate_quant_table_ex(quality, 0, ColorSpace::YCbCr, false, is_420, true);
    let cb_quant = generate_quant_table_ex(quality, 1, ColorSpace::YCbCr, false, is_420, true);
    let cr_quant = generate_quant_table_ex(quality, 2, ColorSpace::YCbCr, false, is_420, true);
    let distance = quality.to_distance();
    processor
        .set_quant_tables(
            y_quant,
            cb_quant,
            cr_quant,
            ZeroBiasParams::for_ycbcr(distance, 0),
            ZeroBiasParams::for_ycbcr(distance, 1),
            ZeroBiasParams::for_ycbcr(distance, 2),
        )
        .unwrap();

    processor
}

fn create_batch_reuse(
    max_batch: usize,
    width: usize,
    subsampling: Subsampling,
    strip_height: usize,
) -> BatchReuse {
    let mcu_size = subsampling.mcu_size();
    let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;
    let c_width = (width + 1) / 2;
    let padded_c_width = (c_width + 7) / 8 * 8;
    let c_strip_height = match subsampling {
        Subsampling::S420 | Subsampling::S440 => strip_height / 2,
        _ => strip_height,
    };

    let v_samp = match subsampling {
        Subsampling::S420 | Subsampling::S440 => 2,
        _ => 1,
    };
    let padded_y_blocks_h = padded_width / 8;
    let blocks_per_strip = padded_y_blocks_h * v_samp;
    let c_blocks_w = (c_width + 7) / 8;
    let c_blocks_per_strip = c_blocks_w; // 1 chroma block row per strip for 4:2:0

    BatchReuse::new(
        max_batch,
        padded_width,
        strip_height,
        padded_c_width,
        c_strip_height,
        blocks_per_strip,
        c_blocks_per_strip,
    )
}

fn bench_fused(
    width: usize,
    height: usize,
    pixels: &[u8],
    iterations: usize,
    warmup: usize,
) -> Vec<f64> {
    let strip_height = 16; // 4:2:0
    let row_bytes = width * 3;
    let mut times = Vec::with_capacity(iterations);

    for i in 0..(warmup + iterations) {
        let mut processor = setup_processor(width, height);

        let start = Instant::now();
        for y in (0..height).step_by(strip_height) {
            let strip_rows = strip_height.min(height - y);
            let strip_start = y * row_bytes;
            let strip_end = (y + strip_rows) * row_bytes;
            processor
                .process_strip(&pixels[strip_start..strip_end], y)
                .unwrap();
        }
        let _output = processor.finalize().unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        if i >= warmup {
            times.push(elapsed);
        }
    }
    times
}

fn bench_batched(
    width: usize,
    height: usize,
    pixels: &[u8],
    batch_size: usize,
    iterations: usize,
    warmup: usize,
) -> Vec<f64> {
    let strip_height = 16; // 4:2:0
    let row_bytes = width * 3;
    let mut times = Vec::with_capacity(iterations);

    // Pre-allocate reuse buffers
    let mut reuse = create_batch_reuse(batch_size, width, Subsampling::S420, strip_height);

    for i in 0..(warmup + iterations) {
        let mut processor = setup_processor(width, height);

        let start = Instant::now();

        // Collect strips into batches
        let mut batch: Vec<(&[u8], usize)> = Vec::with_capacity(batch_size);

        for y in (0..height).step_by(strip_height) {
            let strip_rows = strip_height.min(height - y);
            let strip_start = y * row_bytes;
            let strip_end = (y + strip_rows) * row_bytes;
            batch.push((&pixels[strip_start..strip_end], y));

            if batch.len() == batch_size {
                processor
                    .process_strips_batched(&batch, &mut reuse)
                    .unwrap();
                batch.clear();
            }
        }

        // Process remaining strips
        if !batch.is_empty() {
            processor
                .process_strips_batched(&batch, &mut reuse)
                .unwrap();
        }

        let _output = processor.finalize().unwrap();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        if i >= warmup {
            times.push(elapsed);
        }
    }
    times
}

fn median(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = times.len() / 2;
    if times.len() % 2 == 0 {
        (times[mid - 1] + times[mid]) / 2.0
    } else {
        times[mid]
    }
}

fn main() {
    let cfg = parse_args();

    let (width, height, pixels) = if let Some(ref path) = cfg.ppm_path {
        let (w, h, p) = load_ppm(path);
        eprintln!("Loaded PPM: {}x{}", w, h);
        (w, h, p)
    } else {
        eprintln!("Creating synthetic {}x{} image...", cfg.width, cfg.height);
        (cfg.width, cfg.height, create_synthetic_image(cfg.width, cfg.height))
    };

    let megapixels = width * height;
    let strip_height = 16;
    let total_strips = (height + strip_height - 1) / strip_height;

    eprintln!(
        "Image: {}x{} ({:.1} MP), {} strips of {} rows",
        width,
        height,
        megapixels as f64 / 1_000_000.0,
        total_strips,
        strip_height
    );
    eprintln!(
        "Mode: Q85, 4:2:0, optimize_huffman=false (strip processor only)",
    );
    eprintln!(
        "Iterations: {} (warmup: {})",
        cfg.iterations, cfg.warmup
    );

    if cfg.cachegrind {
        let batch_size = cfg.batch_only.unwrap_or(8);
        eprintln!("Cachegrind mode: single iteration, batch_size={}", batch_size);

        // Run fused
        eprintln!("Running fused...");
        let _ = bench_fused(width, height, &pixels, 1, 0);

        // Run batched
        eprintln!("Running batched (batch={})...", batch_size);
        let _ = bench_batched(width, height, &pixels, batch_size, 1, 0);

        eprintln!("Done. Use cg_annotate to compare.");
        return;
    }

    // Determine batch sizes to test
    let batch_sizes: Vec<usize> = if let Some(bs) = cfg.batch_only {
        vec![bs]
    } else {
        vec![1, 2, 4, 8, 16, 32]
            .into_iter()
            .filter(|&bs| bs <= total_strips)
            .collect()
    };

    // Run fused baseline
    eprintln!("\nRunning fused baseline...");
    let mut fused_times = bench_fused(width, height, &pixels, cfg.iterations, cfg.warmup);
    let fused_median = median(&mut fused_times);

    // Print header
    println!();
    println!(
        "{:<12} {:>10} {:>10} {:>10}",
        "Mode", "Median ms", "vs Fused", "MP/s"
    );
    println!("{}", "-".repeat(46));

    let mp = megapixels as f64 / 1_000_000.0;
    println!(
        "{:<12} {:>10.2} {:>10} {:>10.1}",
        "fused",
        fused_median,
        "baseline",
        mp / (fused_median / 1000.0)
    );

    // Run batched with each batch size
    for &bs in &batch_sizes {
        let label = format!("batch={}", bs);
        eprintln!("Running {}...", label);
        let mut times = bench_batched(width, height, &pixels, bs, cfg.iterations, cfg.warmup);
        let med = median(&mut times);
        let ratio = med / fused_median;
        let sign = if ratio >= 1.0 { "+" } else { "" };
        println!(
            "{:<12} {:>10.2} {:>9}{:.1}% {:>10.1}",
            label,
            med,
            sign,
            (ratio - 1.0) * 100.0,
            mp / (med / 1000.0)
        );
    }

    println!();
    eprintln!("Done.");
}
