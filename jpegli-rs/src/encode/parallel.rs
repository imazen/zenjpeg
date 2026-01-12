//! Parallel encoding support.
//!
//! This module provides parallel implementations of DCT and quantization
//! for improved throughput on multi-core systems.
//!
//! Enable with the `parallel` feature flag.

use crate::dct::simd::forward_dct_8x8_wide;
use crate::simd_types::Block8x8f;
use rayon::prelude::*;

/// Minimum blocks to justify parallel overhead
/// At ~25ns/block, need ~20K blocks for meaningful parallelism
const PARALLEL_THRESHOLD: usize = 16384;  // 1024x1024 equivalent

/// Blocks per parallel task
/// Larger chunks reduce overhead but hurt load balancing
/// Target ~1ms of work per chunk at ~25ns/block = 40K blocks
/// But cap at num_cpus * 4 chunks for load balancing
const CHUNK_SIZE: usize = 4096;

/// Parallel DCT for Y channel blocks.
///
/// Pre-allocates output and uses parallel indexed writes.
/// Falls back to sequential for small block counts.
pub fn parallel_dct_y_blocks(
    strip: &[f32],
    blocks_w: usize,
    strip_blocks_h: usize,
    padded_width: usize,
    output: &mut Vec<Block8x8f>,
) {
    let total_blocks = blocks_w * strip_blocks_h;
    let start_idx = output.len();

    // Pre-allocate space
    output.resize(start_idx + total_blocks, Block8x8f::default());

    if total_blocks < PARALLEL_THRESHOLD {
        // Sequential for small images
        for i in 0..total_blocks {
            let local_by = i / blocks_w;
            let bx = i % blocks_w;
            let block = extract_block_from_strip_wide(strip, bx, local_by, padded_width);
            let dct = forward_dct_8x8_wide(&block);
            output[start_idx + i] = dct;
        }
    } else {
        // Parallel for large images
        let output_slice = &mut output[start_idx..];
        output_slice
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base_i = chunk_idx * CHUNK_SIZE;
                for (j, out) in chunk.iter_mut().enumerate() {
                    let i = base_i + j;
                    if i >= total_blocks {
                        break;
                    }
                    let local_by = i / blocks_w;
                    let bx = i % blocks_w;
                    let block = extract_block_from_strip_wide(strip, bx, local_by, padded_width);
                    *out = forward_dct_8x8_wide(&block);
                }
            });
    }
}

/// Parallel DCT for chroma channel blocks.
///
/// Processes Cb and Cr in parallel, each channel internally parallelized.
pub fn parallel_dct_chroma_blocks(
    cb_strip: &[f32],
    cr_strip: &[f32],
    c_blocks_w: usize,
    c_strip_blocks_h: usize,
    padded_c_width: usize,
    cb_output: &mut Vec<Block8x8f>,
    cr_output: &mut Vec<Block8x8f>,
) {
    let total_blocks = c_blocks_w * c_strip_blocks_h;
    let cb_start = cb_output.len();
    let cr_start = cr_output.len();

    // Pre-allocate
    cb_output.resize(cb_start + total_blocks, Block8x8f::default());
    cr_output.resize(cr_start + total_blocks, Block8x8f::default());

    if total_blocks < PARALLEL_THRESHOLD / 2 {
        // Sequential for small images
        for i in 0..total_blocks {
            let local_by = i / c_blocks_w;
            let bx = i % c_blocks_w;

            let cb_block = extract_block_from_strip_wide(cb_strip, bx, local_by, padded_c_width);
            cb_output[cb_start + i] = forward_dct_8x8_wide(&cb_block);

            let cr_block = extract_block_from_strip_wide(cr_strip, bx, local_by, padded_c_width);
            cr_output[cr_start + i] = forward_dct_8x8_wide(&cr_block);
        }
    } else {
        // Process Cb and Cr in parallel with each other
        let cb_slice = &mut cb_output[cb_start..];
        let cr_slice = &mut cr_output[cr_start..];

        rayon::join(
            || {
                cb_slice
                    .par_chunks_mut(CHUNK_SIZE)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base_i = chunk_idx * CHUNK_SIZE;
                        for (j, out) in chunk.iter_mut().enumerate() {
                            let i = base_i + j;
                            if i >= total_blocks {
                                break;
                            }
                            let local_by = i / c_blocks_w;
                            let bx = i % c_blocks_w;
                            let block = extract_block_from_strip_wide(cb_strip, bx, local_by, padded_c_width);
                            *out = forward_dct_8x8_wide(&block);
                        }
                    });
            },
            || {
                cr_slice
                    .par_chunks_mut(CHUNK_SIZE)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base_i = chunk_idx * CHUNK_SIZE;
                        for (j, out) in chunk.iter_mut().enumerate() {
                            let i = base_i + j;
                            if i >= total_blocks {
                                break;
                            }
                            let local_by = i / c_blocks_w;
                            let bx = i % c_blocks_w;
                            let block = extract_block_from_strip_wide(cr_strip, bx, local_by, padded_c_width);
                            *out = forward_dct_8x8_wide(&block);
                        }
                    });
            },
        );
    }
}

/// Extract an 8×8 block from a strip buffer into wide-native format.
///
/// This is a copy of the function from strip.rs for use in parallel context.
/// IMPORTANT: Applies level shift (-128) as required for JPEG DCT.
#[inline]
fn extract_block_from_strip_wide(
    strip: &[f32],
    block_x: usize,
    block_y: usize,
    padded_width: usize,
) -> Block8x8f {
    use wide::f32x8;

    let level_shift = f32x8::splat(128.0);
    let start_x = block_x * 8;
    let start_y = block_y * 8;

    let mut block = Block8x8f::default();

    for row in 0..8 {
        let y = start_y + row;
        let row_start = y * padded_width + start_x;

        if row_start + 8 <= strip.len() {
            // Fast path: full row available - apply level shift
            let row_slice: [f32; 8] = strip[row_start..row_start + 8].try_into().unwrap();
            block.rows[row] = f32x8::from(row_slice) - level_shift;
        } else if row_start < strip.len() {
            // Partial row: copy what's available, zero-pad rest, apply level shift
            let available = strip.len() - row_start;
            let mut vals = [128.0f32; 8]; // Default to 128 so level shift gives 0
            vals[..available].copy_from_slice(&strip[row_start..row_start + available]);
            block.rows[row] = f32x8::from(vals) - level_shift;
        } else {
            // Entire row missing: level-shifted zero (128 - 128 = 0)
            block.rows[row] = f32x8::ZERO;
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_dct_matches_sequential() {
        // Create a simple test strip
        let width = 256;
        let height = 16;
        let padded_width = ((width + 7) / 8) * 8;
        let strip: Vec<f32> = (0..height * padded_width)
            .map(|i| (i % 256) as f32)
            .collect();

        let blocks_w = (width + 7) / 8;
        let strip_blocks_h = (height + 7) / 8;

        // Sequential reference
        let mut seq_output = Vec::new();
        for local_by in 0..strip_blocks_h {
            for bx in 0..blocks_w {
                let block = extract_block_from_strip_wide(&strip, bx, local_by, padded_width);
                let dct = forward_dct_8x8_wide(&block);
                seq_output.push(dct);
            }
        }

        // Parallel implementation
        let mut par_output = Vec::new();
        parallel_dct_y_blocks(&strip, blocks_w, strip_blocks_h, padded_width, &mut par_output);

        // Compare
        assert_eq!(seq_output.len(), par_output.len());
        for (i, (s, p)) in seq_output.iter().zip(par_output.iter()).enumerate() {
            for row in 0..8 {
                let s_arr: [f32; 8] = s.rows[row].into();
                let p_arr: [f32; 8] = p.rows[row].into();
                for col in 0..8 {
                    assert!(
                        (s_arr[col] - p_arr[col]).abs() < 1e-6,
                        "Mismatch at block {}, row {}, col {}: {} vs {}",
                        i, row, col, s_arr[col], p_arr[col]
                    );
                }
            }
        }
    }
}
