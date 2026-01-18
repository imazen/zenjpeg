//! Profile encoder stages to identify parallelization candidates.
//!
//! Measures time spent in each major encoding stage.

use jpegli::dct::forward_dct_8x8;
use jpegli::entropy::encoder::EntropyEncoder;
use jpegli::huffman::HuffmanEncodeTable;
use jpegli::quant::aq::compute_aq_strength_map;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let sizes = [
        (256, 256),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1920, 1080),
        (2048, 2048),
        (3840, 2160),
        (4096, 4096),
    ];
    let thread_counts = [1, 2, 3, 4];

    println!("=== Parallel Scaling Analysis ===\n");
    println!("{:>12} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Size", "Threads", "Seq(ms)", "Par(ms)", "Speedup", "Eff%", "Blocks");
    println!("{}", "-".repeat(78));

    for &(width, height) in &sizes {
        let blocks = ((width + 7) / 8) * ((height + 7) / 8);

        // First measure sequential baseline (1 thread)
        let seq_time = {
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap()
                .install(|| benchmark_encode_time(width, height))
        };

        for &threads in &thread_counts {
            let par_time = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| benchmark_parallel_time(width, height));

            let speedup = seq_time / par_time;
            let efficiency = (speedup / threads as f64) * 100.0;

            println!("{:>5}x{:<5} {:>8} {:>10.1} {:>10.1} {:>10.2}x {:>9.0}% {:>8}",
                width, height, threads, seq_time, par_time, speedup, efficiency, blocks);
        }
        println!();
    }

    // Generate heuristic
    println!("\n=== Recommended Heuristic ===\n");
    generate_heuristic(&sizes, &thread_counts);
}

fn benchmark_encode_time(width: usize, height: usize) -> f64 {
    // Use same standalone pipeline as parallel version for fair comparison
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    // Warmup - full sequential pipeline
    for _ in 0..2 {
        let (y, cb, cr) = color_convert(&pixels, width, height);
        let y_dct = compute_dct(&y, width, height);
        let cb_dct = compute_dct(&cb, width, height);
        let cr_dct = compute_dct(&cr, width, height);
        let aq_map: Vec<f32> = vec![1.0; y_dct.len()];
        let y_quant = quantize_blocks(&y_dct, &aq_map);
        let cb_quant = quantize_blocks(&cb_dct, &aq_map);
        let cr_quant = quantize_blocks(&cr_dct, &aq_map);
        let _ = entropy_encode(&y_quant, &cb_quant, &cr_quant);
    }

    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        // Full sequential pipeline
        let (y, cb, cr) = color_convert(&pixels, width, height);
        let y_dct = compute_dct(&y, width, height);
        let cb_dct = compute_dct(&cb, width, height);
        let cr_dct = compute_dct(&cr, width, height);
        let aq_map: Vec<f32> = vec![1.0; y_dct.len()];
        let y_quant = quantize_blocks(&y_dct, &aq_map);
        let cb_quant = quantize_blocks(&cb_dct, &aq_map);
        let cr_quant = quantize_blocks(&cr_dct, &aq_map);
        let _ = entropy_encode(&y_quant, &cb_quant, &cr_quant);
    }
    start.elapsed().as_millis() as f64 / iters as f64
}

fn benchmark_parallel_time(width: usize, height: usize) -> f64 {
    use jpegli::encode::parallel::{parallel_entropy_encode_444, ParallelEntropyConfig};

    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    let restart_interval = 64u16;
    let config = ParallelEntropyConfig {
        dc_luma: HuffmanEncodeTable::std_dc_luminance().clone(),
        ac_luma: HuffmanEncodeTable::std_ac_luminance().clone(),
        dc_chroma: HuffmanEncodeTable::std_dc_chrominance().clone(),
        ac_chroma: HuffmanEncodeTable::std_ac_chrominance().clone(),
    };

    // Warmup - full pipeline
    for _ in 0..2 {
        let (y, cb, cr) = color_convert_parallel(&pixels, width, height);
        let y_dct = compute_dct(&y, width, height);
        let cb_dct = compute_dct(&cb, width, height);
        let cr_dct = compute_dct(&cr, width, height);
        let aq_map: Vec<f32> = vec![1.0; y_dct.len()];
        let y_quant = quantize_blocks(&y_dct, &aq_map);
        let cb_quant = quantize_blocks(&cb_dct, &aq_map);
        let cr_quant = quantize_blocks(&cr_dct, &aq_map);
        let _ = parallel_entropy_encode_444(&y_quant, &cb_quant, &cr_quant, true, restart_interval, &config);
    }

    let iters = 5;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        // Full pipeline with parallel color + parallel entropy
        let (y, cb, cr) = color_convert_parallel(&pixels, width, height);
        let y_dct = compute_dct(&y, width, height);
        let cb_dct = compute_dct(&cb, width, height);
        let cr_dct = compute_dct(&cr, width, height);
        let aq_map: Vec<f32> = vec![1.0; y_dct.len()];
        let y_quant = quantize_blocks(&y_dct, &aq_map);
        let cb_quant = quantize_blocks(&cb_dct, &aq_map);
        let cr_quant = quantize_blocks(&cr_dct, &aq_map);
        let _ = parallel_entropy_encode_444(&y_quant, &cb_quant, &cr_quant, true, restart_interval, &config);
    }
    start.elapsed().as_millis() as f64 / iters as f64
}

fn measure_color_time(width: usize, height: usize) -> f64 {
    let pixels = vec![128u8; width * height * 3];
    let start = std::time::Instant::now();
    for _ in 0..3 {
        let _ = color_convert(&pixels, width, height);
    }
    start.elapsed().as_millis() as f64 / 3.0
}

fn measure_entropy_time(width: usize, height: usize) -> f64 {
    let blocks = ((width + 7) / 8) * ((height + 7) / 8);
    let dummy_blocks: Vec<[i16; 64]> = vec![[0i16; 64]; blocks];

    let start = std::time::Instant::now();
    for _ in 0..3 {
        let _ = entropy_encode(&dummy_blocks, &dummy_blocks, &dummy_blocks);
    }
    start.elapsed().as_millis() as f64 / 3.0
}

fn generate_heuristic(sizes: &[(usize, usize)], thread_counts: &[usize]) {
    // Collect data points
    let mut data: Vec<(usize, usize, f64)> = Vec::new(); // (blocks, threads, speedup)

    for &(width, height) in sizes {
        let blocks = ((width + 7) / 8) * ((height + 7) / 8);

        let seq_time = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| benchmark_encode_time(width, height));

        for &threads in thread_counts {
            if threads == 1 { continue; }

            let par_time = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| benchmark_parallel_time(width, height));

            let speedup = seq_time / par_time;
            data.push((blocks, threads, speedup));
        }
    }

    // Find minimum block count where parallelism helps
    let min_blocks_for_speedup: usize = data.iter()
        .filter(|(_, threads, speedup)| *threads == 2 && *speedup > 1.05)
        .map(|(blocks, _, _)| *blocks)
        .min()
        .unwrap_or(4096);

    println!("Minimum blocks for 2-thread benefit: {} (~{}x{} image)",
        min_blocks_for_speedup,
        (min_blocks_for_speedup as f64).sqrt() as usize * 8,
        (min_blocks_for_speedup as f64).sqrt() as usize * 8);

    println!("\nRecommended parallel encoding heuristic:");
    println!("```rust");
    println!("fn should_use_parallel(width: u32, height: u32, available_threads: usize) -> bool {{");
    println!("    let blocks = ((width + 7) / 8) * ((height + 7) / 8);");
    println!("    let min_blocks = {};  // ~{}x{}", min_blocks_for_speedup,
        (min_blocks_for_speedup as f64).sqrt() as usize * 8,
        (min_blocks_for_speedup as f64).sqrt() as usize * 8);
    println!("    blocks >= min_blocks && available_threads >= 2");
    println!("}}");
    println!("```");

    // Show optimal thread count per size
    println!("\nOptimal thread count by image size:");
    for &(width, height) in sizes {
        let blocks = ((width + 7) / 8) * ((height + 7) / 8);

        let best = data.iter()
            .filter(|(b, _, _)| *b == blocks)
            .max_by(|(_, _, s1), (_, _, s2)| s1.partial_cmp(s2).unwrap());

        if let Some((_, threads, speedup)) = best {
            println!("  {}x{}: {} threads ({:.2}x speedup)", width, height, threads, speedup);
        }
    }
}

fn profile_encode(width: usize, height: usize) {
    // Create test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;
    let total_blocks = blocks_w * blocks_h;

    // Warmup
    let _ = color_convert(&pixels, width, height);

    let iters = 5;

    // 1. Color conversion
    let start = Instant::now();
    for _ in 0..iters {
        let _ = color_convert(&pixels, width, height);
    }
    let color_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    // Get YCbCr data for next stages
    let (y_plane, cb_plane, cr_plane) = color_convert(&pixels, width, height);

    // Convert Y plane to f32 for AQ (real jpegli uses f32)
    let y_plane_f32: Vec<f32> = y_plane.iter().map(|&v| v as f32).collect();

    // 2. Adaptive quantization (HF modulation) - REAL jpegli AQ
    let start = Instant::now();
    for _ in 0..iters {
        let _ = compute_aq_strength_map(&y_plane_f32, width, height, 8);
    }
    let aq_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let aq_map_real = compute_aq_strength_map(&y_plane_f32, width, height, 8).unwrap();
    // Convert to simple Vec<f32> for quantize_blocks
    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;
    let mut aq_map = Vec::with_capacity(blocks_w * blocks_h);
    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            aq_map.push(aq_map_real.get(bx, by));
        }
    }

    // 3. DCT
    let start = Instant::now();
    for _ in 0..iters {
        let _ = compute_dct(&y_plane, width, height);
    }
    let dct_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let y_dct = compute_dct(&y_plane, width, height);
    let cb_dct = compute_dct(&cb_plane, width, height);
    let cr_dct = compute_dct(&cr_plane, width, height);

    // 4. Quantization
    let start = Instant::now();
    for _ in 0..iters {
        let _ = quantize_blocks(&y_dct, &aq_map);
    }
    let quant_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let y_quant = quantize_blocks(&y_dct, &aq_map);
    let cb_quant = quantize_blocks(&cb_dct, &aq_map);
    let cr_quant = quantize_blocks(&cr_dct, &aq_map);

    // 5. Entropy encoding
    let start = Instant::now();
    for _ in 0..iters {
        let _ = entropy_encode(&y_quant, &cb_quant, &cr_quant);
    }
    let entropy_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let total = color_time + aq_time + dct_time + quant_time + entropy_time;

    println!(
        "Color conversion:    {:6.1}ms ({:4.1}%)",
        color_time,
        100.0 * color_time / total
    );
    println!(
        "Adaptive quant (AQ): {:6.1}ms ({:4.1}%)",
        aq_time,
        100.0 * aq_time / total
    );
    println!(
        "DCT:                 {:6.1}ms ({:4.1}%)",
        dct_time,
        100.0 * dct_time / total
    );
    println!(
        "Quantization:        {:6.1}ms ({:4.1}%)",
        quant_time,
        100.0 * quant_time / total
    );
    println!(
        "Entropy encoding:    {:6.1}ms ({:4.1}%)",
        entropy_time,
        100.0 * entropy_time / total
    );
    println!("─────────────────────────────────");
    println!("Total (stages):      {:6.1}ms", total);
    println!("\nBlocks: {}, per-block times:", total_blocks);
    println!(
        "  Color: {:.2}µs, AQ: {:.2}µs, DCT: {:.2}µs, Quant: {:.2}µs, Entropy: {:.2}µs",
        color_time * 1000.0 / total_blocks as f64,
        aq_time * 1000.0 / total_blocks as f64,
        dct_time * 1000.0 / total_blocks as f64,
        quant_time * 1000.0 / total_blocks as f64,
        entropy_time * 1000.0 / total_blocks as f64,
    );
}

fn color_convert(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    let mut cb = vec![0u8; width * height];
    let mut cr = vec![0u8; width * height];

    // Simple scalar conversion
    for i in 0..(width * height) {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;

        y[i] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
        cb[i] = (128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
        cr[i] = (128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b)
            .round()
            .clamp(0.0, 255.0) as u8;
    }

    (y, cb, cr)
}


fn compute_dct(plane: &[u8], width: usize, height: usize) -> Vec<[i16; 64]> {
    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;
    let mut blocks = Vec::with_capacity(blocks_w * blocks_h);

    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let mut input = [0.0f32; 64];

            for dy in 0..8 {
                for dx in 0..8 {
                    let x = (bx * 8 + dx).min(width - 1);
                    let y = (by * 8 + dy).min(height - 1);
                    input[dy * 8 + dx] = plane[y * width + x] as f32 - 128.0;
                }
            }

            let output_f32 = forward_dct_8x8(&input);
            let mut output = [0i16; 64];
            for i in 0..64 {
                output[i] = output_f32[i].round() as i16;
            }
            blocks.push(output);
        }
    }

    blocks
}

// Standard JPEG quantization table for quality ~85
const QUANT_TABLE: [i16; 64] = [
    4, 3, 3, 4, 6, 10, 13, 16, 3, 3, 4, 5, 7, 15, 16, 14, 3, 4, 4, 6, 10, 15, 18, 14, 3, 4, 5, 7,
    13, 22, 21, 17, 4, 5, 8, 14, 18, 28, 26, 20, 6, 8, 13, 16, 21, 27, 29, 24, 12, 16, 20, 22, 26,
    31, 31, 26, 18, 24, 24, 26, 29, 26, 27, 26,
];

fn quantize_blocks(dct_blocks: &[[i16; 64]], aq_map: &[f32]) -> Vec<[i16; 64]> {
    dct_blocks
        .iter()
        .enumerate()
        .map(|(i, block)| {
            let mut out = [0i16; 64];
            let aq_scale = aq_map.get(i).copied().unwrap_or(1.0);
            for j in 0..64 {
                let q = (QUANT_TABLE[j] as f32 * aq_scale).max(1.0) as i16;
                out[j] = if block[j] >= 0 {
                    (block[j] + q / 2) / q.max(1)
                } else {
                    (block[j] - q / 2) / q.max(1)
                };
            }
            out
        })
        .collect()
}

fn entropy_encode(y_blocks: &[[i16; 64]], cb_blocks: &[[i16; 64]], cr_blocks: &[[i16; 64]]) -> Vec<u8> {
    let mut encoder = EntropyEncoder::with_capacity(y_blocks.len() * 100);

    encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
    encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
    encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
    encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());

    for i in 0..y_blocks.len() {
        let _ = encoder.encode_block(&y_blocks[i], 0, 0, 0);
        if i < cb_blocks.len() {
            let _ = encoder.encode_block(&cb_blocks[i], 1, 1, 1);
        }
        if i < cr_blocks.len() {
            let _ = encoder.encode_block(&cr_blocks[i], 2, 1, 1);
        }
    }

    encoder.finish()
}

fn benchmark_parallel(width: usize, height: usize) {
    // Create test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    let iters = 10;

    // Warmup
    let _ = color_convert(&pixels, width, height);
    let _ = color_convert_parallel(&pixels, width, height);

    // 1. Color conversion
    let start = Instant::now();
    for _ in 0..iters {
        let _ = color_convert(&pixels, width, height);
    }
    let seq_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = color_convert_parallel(&pixels, width, height);
    }
    let par_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;
    println!("Color conversion:  seq={:.1}ms  par={:.1}ms  speedup={:.2}x",
        seq_time, par_time, seq_time / par_time);

    let (y_plane, cb_plane, cr_plane) = color_convert(&pixels, width, height);
    let y_plane_f32: Vec<f32> = y_plane.iter().map(|&v| v as f32).collect();

    // 2. DCT
    let start = Instant::now();
    for _ in 0..iters {
        let _ = compute_dct(&y_plane, width, height);
    }
    let seq_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = compute_dct_parallel(&y_plane, width, height);
    }
    let par_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;
    println!("DCT:               seq={:.1}ms  par={:.1}ms  speedup={:.2}x",
        seq_time, par_time, seq_time / par_time);

    let y_dct = compute_dct(&y_plane, width, height);

    // 3. Quantization
    let aq_map: Vec<f32> = vec![1.0; y_dct.len()]; // simplified for benchmark

    let start = Instant::now();
    for _ in 0..iters {
        let _ = quantize_blocks(&y_dct, &aq_map);
    }
    let seq_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = quantize_blocks_parallel(&y_dct, &aq_map);
    }
    let par_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;
    println!("Quantization:      seq={:.1}ms  par={:.1}ms  speedup={:.2}x",
        seq_time, par_time, seq_time / par_time);

    // 4. Combined DCT+Quant (fused parallel)
    let start = Instant::now();
    for _ in 0..iters {
        let dct = compute_dct(&y_plane, width, height);
        let _ = quantize_blocks(&dct, &aq_map);
    }
    let seq_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = compute_dct_quant_parallel(&y_plane, width, height, &aq_map);
    }
    let par_time = start.elapsed().as_micros() as f64 / iters as f64 / 1000.0;
    println!("DCT+Quant fused:   seq={:.1}ms  par={:.1}ms  speedup={:.2}x",
        seq_time, par_time, seq_time / par_time);
}

// ===== Parallel implementations =====

fn color_convert_parallel(rgb: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut y = vec![0u8; width * height];
    let mut cb = vec![0u8; width * height];
    let mut cr = vec![0u8; width * height];

    // Process rows in parallel
    y.par_chunks_mut(width)
        .zip(cb.par_chunks_mut(width))
        .zip(cr.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, ((y_row, cb_row), cr_row))| {
            let row_start = row * width * 3;
            for x in 0..width {
                let i = row_start + x * 3;
                let r = rgb[i] as f32;
                let g = rgb[i + 1] as f32;
                let b = rgb[i + 2] as f32;

                y_row[x] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
                cb_row[x] = (128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b).round().clamp(0.0, 255.0) as u8;
                cr_row[x] = (128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b).round().clamp(0.0, 255.0) as u8;
            }
        });

    (y, cb, cr)
}

fn compute_dct_parallel(plane: &[u8], width: usize, height: usize) -> Vec<[i16; 64]> {
    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;

    (0..blocks_h)
        .into_par_iter()
        .flat_map(|by| {
            (0..blocks_w).into_par_iter().map(move |bx| {
                let mut input = [0.0f32; 64];
                for dy in 0..8 {
                    for dx in 0..8 {
                        let x = (bx * 8 + dx).min(width - 1);
                        let y = (by * 8 + dy).min(height - 1);
                        input[dy * 8 + dx] = plane[y * width + x] as f32 - 128.0;
                    }
                }
                let output_f32 = forward_dct_8x8(&input);
                let mut output = [0i16; 64];
                for i in 0..64 {
                    output[i] = output_f32[i].round() as i16;
                }
                output
            })
        })
        .collect()
}

fn quantize_blocks_parallel(dct_blocks: &[[i16; 64]], aq_map: &[f32]) -> Vec<[i16; 64]> {
    dct_blocks
        .par_iter()
        .enumerate()
        .map(|(i, block)| {
            let mut out = [0i16; 64];
            let aq_scale = aq_map.get(i).copied().unwrap_or(1.0);
            for j in 0..64 {
                let q = (QUANT_TABLE[j] as f32 * aq_scale).max(1.0) as i16;
                out[j] = if block[j] >= 0 {
                    (block[j] + q / 2) / q.max(1)
                } else {
                    (block[j] - q / 2) / q.max(1)
                };
            }
            out
        })
        .collect()
}

fn compute_dct_quant_parallel(plane: &[u8], width: usize, height: usize, aq_map: &[f32]) -> Vec<[i16; 64]> {
    let blocks_w = (width + 7) / 8;
    let blocks_h = (height + 7) / 8;

    (0..blocks_h)
        .into_par_iter()
        .flat_map(|by| {
            (0..blocks_w).into_par_iter().map(move |bx| {
                let block_idx = by * blocks_w + bx;

                // DCT
                let mut input = [0.0f32; 64];
                for dy in 0..8 {
                    for dx in 0..8 {
                        let x = (bx * 8 + dx).min(width - 1);
                        let y = (by * 8 + dy).min(height - 1);
                        input[dy * 8 + dx] = plane[y * width + x] as f32 - 128.0;
                    }
                }
                let dct = forward_dct_8x8(&input);

                // Quantize
                let mut out = [0i16; 64];
                let aq_scale = aq_map.get(block_idx).copied().unwrap_or(1.0);
                for j in 0..64 {
                    let q = (QUANT_TABLE[j] as f32 * aq_scale).max(1.0) as i16;
                    let v = dct[j].round() as i16;
                    out[j] = if v >= 0 {
                        (v + q / 2) / q.max(1)
                    } else {
                        (v - q / 2) / q.max(1)
                    };
                }
                out
            })
        })
        .collect()
}

// ===== Full pipeline benchmark =====

fn benchmark_full_pipeline(width: usize, height: usize) {
    use jpegli::{Quality, StreamingEncoder};

    // Create test image
    let mut pixels = vec![0u8; width * height * 3];
    let mut seed = 12345u64;
    for p in pixels.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = (seed >> 33) as u8;
    }

    let iters = 10;

    // Warmup
    for _ in 0..3 {
        let _ = StreamingEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(85.0))
            .encode_all(&pixels);
    }

    // Sequential encode
    let start = Instant::now();
    for _ in 0..iters {
        let _ = StreamingEncoder::new(width as u32, height as u32)
            .quality(Quality::from_quality(85.0))
            .encode_all(&pixels);
    }
    let seq_time = start.elapsed().as_millis() as f64 / iters as f64;
    println!("Full sequential encode: {:.1}ms", seq_time);

    // Measure color conversion portion
    let start = Instant::now();
    for _ in 0..iters {
        let _ = color_convert(&pixels, width, height);
    }
    let color_seq = start.elapsed().as_millis() as f64 / iters as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = color_convert_parallel(&pixels, width, height);
    }
    let color_par = start.elapsed().as_millis() as f64 / iters as f64;

    println!("\n--- Stage breakdown ---");
    println!("Color conversion:       seq={:.1}ms  par={:.1}ms  speedup={:.2}x",
        color_seq, color_par, color_seq / color_par);

    // Get pipeline stages for measuring entropy time
    let (y_plane, _, _) = color_convert(&pixels, width, height);
    let y_dct = compute_dct(&y_plane, width, height);
    let aq_map: Vec<f32> = vec![1.0; y_dct.len()];
    let y_quant = quantize_blocks(&y_dct, &aq_map);

    // Entropy encoding isolated
    let start = Instant::now();
    for _ in 0..iters {
        let _ = entropy_encode(&y_quant, &y_quant, &y_quant);
    }
    let entropy_seq = start.elapsed().as_millis() as f64 / iters as f64;

    #[cfg(feature = "parallel")]
    {
        use jpegli::encode::parallel::{parallel_entropy_encode_444, ParallelEntropyConfig};

        let restart_interval = 64u16; // restart every 64 MCUs

        let config = ParallelEntropyConfig {
            dc_luma: HuffmanEncodeTable::std_dc_luminance().clone(),
            ac_luma: HuffmanEncodeTable::std_ac_luminance().clone(),
            dc_chroma: HuffmanEncodeTable::std_dc_chrominance().clone(),
            ac_chroma: HuffmanEncodeTable::std_ac_chrominance().clone(),
        };

        // Warmup
        let _ = parallel_entropy_encode_444(&y_quant, &y_quant, &y_quant, true, restart_interval, &config);

        let start = Instant::now();
        for _ in 0..iters {
            let _ = parallel_entropy_encode_444(&y_quant, &y_quant, &y_quant, true, restart_interval, &config);
        }
        let entropy_par = start.elapsed().as_millis() as f64 / iters as f64;

        println!("Entropy encoding:       seq={:.1}ms  par={:.1}ms  speedup={:.2}x",
            entropy_seq, entropy_par, entropy_seq / entropy_par);

        // Estimate combined speedup
        let other_stages = seq_time - color_seq - entropy_seq;
        let estimated_combined = color_par + entropy_par + other_stages;
        println!("\n--- Combined estimates ---");
        println!("With parallel color:    {:.1}ms ({:.2}x speedup)",
            seq_time - color_seq + color_par, seq_time / (seq_time - color_seq + color_par));
        println!("With parallel entropy:  {:.1}ms ({:.2}x speedup)",
            seq_time - entropy_seq + entropy_par, seq_time / (seq_time - entropy_seq + entropy_par));
        println!("With both parallel:     {:.1}ms ({:.2}x speedup)",
            estimated_combined, seq_time / estimated_combined);
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("Entropy encoding:       seq={:.1}ms  (parallel feature not enabled)", entropy_seq);
    }
}
