//! Benchmark parallel entropy encoding.
//!
//! Compares sequential vs parallel entropy encoding performance.

use jpegli::foundation::consts::DCT_BLOCK_SIZE;
use jpegli::entropy::encoder::EntropyEncoder;
use jpegli::huffman::HuffmanEncodeTable;
use std::time::Instant;

#[cfg(feature = "parallel")]
use jpegli::encode::parallel::{parallel_entropy_encode_444, ParallelEntropyConfig};

fn create_test_blocks(width: usize, height: usize) -> Vec<[i16; DCT_BLOCK_SIZE]> {
    let blocks_h = (width + 7) / 8;
    let blocks_v = (height + 7) / 8;
    let total = blocks_h * blocks_v;

    // Generate pseudo-random blocks
    let mut blocks = Vec::with_capacity(total);
    let mut seed = 12345u64;

    for _ in 0..total {
        let mut block = [0i16; DCT_BLOCK_SIZE];
        for i in 0..DCT_BLOCK_SIZE {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            // DC values are larger, AC values are smaller
            let val = if i == 0 {
                ((seed >> 32) as i16) % 2048 - 1024
            } else {
                ((seed >> 32) as i16) % 128 - 64
            };
            block[i] = val;
        }
        blocks.push(block);
    }

    blocks
}

fn encode_sequential(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    restart_interval: u16,
) -> Vec<u8> {
    let mut encoder = EntropyEncoder::with_capacity(y_blocks.len() * 100);

    encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
    encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
    encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
    encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());

    if restart_interval > 0 {
        encoder.set_restart_interval(restart_interval);
    }

    for i in 0..y_blocks.len() {
        let _ = encoder.encode_block(&y_blocks[i], 0, 0, 0);
        let _ = encoder.encode_block(&cb_blocks[i], 1, 1, 1);
        let _ = encoder.encode_block(&cr_blocks[i], 2, 1, 1);
        encoder.check_restart();
    }

    encoder.finish()
}

fn main() {
    let (width, height) = (2048, 2048);
    let blocks_h = (width + 7) / 8;
    let blocks_v = (height + 7) / 8;
    let total_blocks = blocks_h * blocks_v;

    println!(
        "Image: {}x{} ({} blocks, {} MCUs)",
        width, height, total_blocks, total_blocks
    );

    // Create test data
    let y_blocks = create_test_blocks(width, height);
    let cb_blocks = create_test_blocks(width, height);
    let cr_blocks = create_test_blocks(width, height);

    // Test different restart intervals
    for &restart_interval in &[0, 64, 128, 256, 512, 1024] {
        println!("\n=== Restart interval: {} ===", restart_interval);

        // Warmup
        for _ in 0..3 {
            let _ = encode_sequential(&y_blocks, &cb_blocks, &cr_blocks, restart_interval);
        }

        // Sequential benchmark
        let iterations = 10;
        let start = Instant::now();
        let mut seq_size = 0;
        for _ in 0..iterations {
            let result = encode_sequential(&y_blocks, &cb_blocks, &cr_blocks, restart_interval);
            seq_size = result.len();
            std::hint::black_box(&result);
        }
        let seq_time = start.elapsed().as_millis() as f64 / iterations as f64;
        println!("Sequential:  {:7.2}ms, {:7} bytes", seq_time, seq_size);

        // Parallel benchmark (if enabled)
        #[cfg(feature = "parallel")]
        if restart_interval > 0 {
            let config = ParallelEntropyConfig {
                dc_luma: HuffmanEncodeTable::std_dc_luminance().clone(),
                ac_luma: HuffmanEncodeTable::std_ac_luminance().clone(),
                dc_chroma: HuffmanEncodeTable::std_dc_chrominance().clone(),
                ac_chroma: HuffmanEncodeTable::std_ac_chrominance().clone(),
            };

            // Warmup
            for _ in 0..3 {
                let _ = parallel_entropy_encode_444(
                    &y_blocks,
                    &cb_blocks,
                    &cr_blocks,
                    true,
                    restart_interval,
                    &config,
                );
            }

            let start = Instant::now();
            let mut par_size = 0;
            for _ in 0..iterations {
                let result = parallel_entropy_encode_444(
                    &y_blocks,
                    &cb_blocks,
                    &cr_blocks,
                    true,
                    restart_interval,
                    &config,
                );
                par_size = result.len();
                std::hint::black_box(&result);
            }
            let par_time = start.elapsed().as_millis() as f64 / iterations as f64;
            let speedup = seq_time / par_time;
            let size_diff = (par_size as f64 - seq_size as f64) / seq_size as f64 * 100.0;
            println!(
                "Parallel:    {:7.2}ms, {:7} bytes ({:+.2}%)",
                par_time, par_size, size_diff
            );
            println!("Speedup:     {:7.2}x", speedup);
        }
    }
}
