//! Parallel encoding support.
//!
//! This module provides parallel implementations of DCT, quantization,
//! and entropy encoding for improved throughput on multi-core systems.
//!
//! Enable with the `parallel` feature flag.
//!
//! ## Performance Characteristics
//!
//! | Threads | Avg Speedup | Efficiency | Best For |
//! |---------|-------------|------------|----------|
//! | 2 | 1.2-1.6x | 58-81% | Balanced |
//! | 3 | 1.2-1.5x | 40-54% | Diminishing returns |
//! | 4 | 1.3-1.7x | 30-40% | Max throughput |
//!
//! Minimum useful size: ~512x512 (4096 blocks)

use crate::encode::dct::simd::forward_dct_8x8_wide;
use crate::encode::deringing::preprocess_deringing_block;
use crate::entropy::encoder::EntropyEncoder;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::Block8x8f;
use crate::huffman::HuffmanEncodeTable;
use archmage::autoversion;
use rayon::prelude::*;

// Re-export from strip.rs to avoid duplication
use super::strip::extract_block_from_strip_wide;

// =============================================================================
// Constants
// =============================================================================

/// Minimum blocks to justify parallel overhead (~512x512 image)
const PARALLEL_THRESHOLD: usize = 4096;

/// Blocks per parallel task - balances overhead vs load balancing
const CHUNK_SIZE: usize = 4096;

// =============================================================================
// Parallel DCT
// =============================================================================

/// Core parallel DCT loop - processes a plane's blocks in parallel chunks.
///
/// This is the workhorse that both Y and chroma DCT functions delegate to.
/// When `deringing` is `Some(dc_quant)`, applies overshoot deringing before DCT.
#[autoversion]
fn parallel_dct_plane(
    strip: &[f32],
    blocks_w: usize,
    total_blocks: usize,
    padded_width: usize,
    deringing: Option<u16>,
    output: &mut [Block8x8f],
) {
    output
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
                let mut block = extract_block_from_strip_wide(strip, bx, local_by, padded_width);
                if let Some(dc_quant) = deringing {
                    preprocess_deringing_block(&mut block, dc_quant);
                }
                *out = forward_dct_8x8_wide(&block);
            }
        });
}

/// Sequential DCT fallback for small block counts.
#[autoversion]
#[inline]
fn sequential_dct_plane(
    strip: &[f32],
    blocks_w: usize,
    total_blocks: usize,
    padded_width: usize,
    deringing: Option<u16>,
    output: &mut [Block8x8f],
) {
    for i in 0..total_blocks {
        let local_by = i / blocks_w;
        let bx = i % blocks_w;
        let mut block = extract_block_from_strip_wide(strip, bx, local_by, padded_width);
        if let Some(dc_quant) = deringing {
            preprocess_deringing_block(&mut block, dc_quant);
        }
        output[i] = forward_dct_8x8_wide(&block);
    }
}

/// Parallel DCT for Y channel blocks.
///
/// Pre-allocates output and uses parallel indexed writes.
/// Falls back to sequential for small block counts.
/// When `deringing` is `Some(dc_quant)`, applies overshoot deringing before DCT.
pub fn parallel_dct_y_blocks(
    strip: &[f32],
    blocks_w: usize,
    strip_blocks_h: usize,
    padded_width: usize,
    deringing: Option<u16>,
    output: &mut Vec<Block8x8f>,
) {
    let total_blocks = blocks_w * strip_blocks_h;
    let start_idx = output.len();

    // Pre-allocate space
    output.resize(start_idx + total_blocks, Block8x8f::default());
    let output_slice = &mut output[start_idx..];

    if total_blocks < PARALLEL_THRESHOLD {
        sequential_dct_plane(
            strip,
            blocks_w,
            total_blocks,
            padded_width,
            deringing,
            output_slice,
        );
    } else {
        parallel_dct_plane(
            strip,
            blocks_w,
            total_blocks,
            padded_width,
            deringing,
            output_slice,
        );
    }
}

/// Parallel DCT for chroma channel blocks.
///
/// Processes Cb and Cr in parallel with each other using rayon::join.
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

    let cb_slice = &mut cb_output[cb_start..];
    let cr_slice = &mut cr_output[cr_start..];

    if total_blocks < PARALLEL_THRESHOLD / 2 {
        // Sequential for small images
        sequential_dct_plane(
            cb_strip,
            c_blocks_w,
            total_blocks,
            padded_c_width,
            None,
            cb_slice,
        );
        sequential_dct_plane(
            cr_strip,
            c_blocks_w,
            total_blocks,
            padded_c_width,
            None,
            cr_slice,
        );
    } else {
        // Process Cb and Cr in parallel with each other
        rayon::join(
            || {
                parallel_dct_plane(
                    cb_strip,
                    c_blocks_w,
                    total_blocks,
                    padded_c_width,
                    None,
                    cb_slice,
                )
            },
            || {
                parallel_dct_plane(
                    cr_strip,
                    c_blocks_w,
                    total_blocks,
                    padded_c_width,
                    None,
                    cr_slice,
                )
            },
        );
    }
}

// =============================================================================
// Parallel Entropy Encoding
// =============================================================================

/// Parallel entropy encoding configuration.
#[derive(Clone)]
pub struct ParallelEntropyConfig {
    /// DC luminance Huffman table
    pub dc_luma: HuffmanEncodeTable,
    /// AC luminance Huffman table
    pub ac_luma: HuffmanEncodeTable,
    /// DC chrominance Huffman table
    pub dc_chroma: HuffmanEncodeTable,
    /// AC chrominance Huffman table
    pub ac_chroma: HuffmanEncodeTable,
}

impl ParallelEntropyConfig {
    /// Create config with standard JPEG Huffman tables.
    pub fn standard() -> Self {
        Self {
            dc_luma: HuffmanEncodeTable::std_dc_luminance().clone(),
            ac_luma: HuffmanEncodeTable::std_ac_luminance().clone(),
            dc_chroma: HuffmanEncodeTable::std_dc_chrominance().clone(),
            ac_chroma: HuffmanEncodeTable::std_ac_chrominance().clone(),
        }
    }
}

/// Result from encoding one restart segment.
struct SegmentResult {
    /// Encoded bitstream data
    data: Vec<u8>,
    /// Restart marker number (0-7)
    restart_num: u8,
}

/// Creates an entropy encoder configured with the given Huffman tables.
#[inline]
fn create_encoder(config: &ParallelEntropyConfig, capacity: usize) -> EntropyEncoder<'_> {
    let mut encoder = EntropyEncoder::with_capacity(capacity);
    encoder.set_dc_table(0, &config.dc_luma);
    encoder.set_ac_table(0, &config.ac_luma);
    encoder.set_dc_table(1, &config.dc_chroma);
    encoder.set_ac_table(1, &config.ac_chroma);
    encoder
}

/// Encodes a single restart segment for 4:4:4 images.
///
/// Each segment starts with DC predictions reset to 0.
fn encode_segment_444(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    mcu_start: usize,
    mcu_count: usize,
    is_color: bool,
    config: &ParallelEntropyConfig,
    restart_num: u8,
) -> SegmentResult {
    let mut encoder = create_encoder(config, mcu_count * 100);

    let mcu_end = (mcu_start + mcu_count).min(y_blocks.len());
    for i in mcu_start..mcu_end {
        encoder.encode_block(&y_blocks[i], 0, 0, 0);
        if is_color {
            encoder.encode_block(&cb_blocks[i], 1, 1, 1);
            encoder.encode_block(&cr_blocks[i], 2, 1, 1);
        }
    }

    SegmentResult {
        data: encoder.finish(),
        restart_num,
    }
}

/// Encodes a single restart segment for subsampled images.
fn encode_segment_subsampled(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    mcu_start: usize,
    mcu_count: usize,
    mcu_h: usize,
    y_blocks_w: usize,
    y_blocks_h: usize,
    c_blocks_w: usize,
    c_blocks_h: usize,
    h_samp: usize,
    v_samp: usize,
    is_color: bool,
    config: &ParallelEntropyConfig,
    restart_num: u8,
) -> SegmentResult {
    let mut encoder = create_encoder(config, mcu_count * 100 * h_samp * v_samp);

    const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

    for mcu_idx in mcu_start..(mcu_start + mcu_count) {
        let mcu_x = mcu_idx % mcu_h;
        let mcu_y = mcu_idx / mcu_h;

        // Encode Y blocks in this MCU
        for dy in 0..v_samp {
            for dx in 0..h_samp {
                let y_bx = mcu_x * h_samp + dx;
                let y_by = mcu_y * v_samp + dy;
                if y_bx < y_blocks_w && y_by < y_blocks_h {
                    let y_idx = y_by * y_blocks_w + y_bx;
                    encoder.encode_block(&y_blocks[y_idx], 0, 0, 0);
                } else {
                    encoder.encode_block(&ZERO_BLOCK, 0, 0, 0);
                }
            }
        }

        // Encode Cb and Cr blocks
        if is_color {
            if mcu_x < c_blocks_w && mcu_y < c_blocks_h {
                let c_idx = mcu_y * c_blocks_w + mcu_x;
                encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1);
                encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1);
            } else {
                encoder.encode_block(&ZERO_BLOCK, 1, 1, 1);
                encoder.encode_block(&ZERO_BLOCK, 2, 1, 1);
            }
        }
    }

    SegmentResult {
        data: encoder.finish(),
        restart_num,
    }
}

/// Combines encoded segments with RST markers between them.
fn combine_segments(segments: &[SegmentResult]) -> Vec<u8> {
    let total_size: usize = segments.iter().map(|s| s.data.len() + 2).sum();
    let mut output = Vec::with_capacity(total_size);

    for (i, segment) in segments.iter().enumerate() {
        output.extend_from_slice(&segment.data);

        // Add RST marker between segments (not after the last one)
        if i < segments.len() - 1 {
            output.push(0xFF);
            output.push(0xD0 + segment.restart_num);
        }
    }

    output
}

/// Parallel entropy encoding for 4:4:4 images.
///
/// Splits MCUs into restart intervals and encodes in parallel.
/// Returns the combined bitstream with RST markers.
pub fn parallel_entropy_encode_444(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    is_color: bool,
    restart_interval: u16,
    config: &ParallelEntropyConfig,
) -> Vec<u8> {
    let total_mcus = y_blocks.len();
    let interval = restart_interval as usize;
    let num_segments = (total_mcus + interval - 1) / interval;

    if num_segments <= 1 {
        // Single segment - no parallelism benefit
        let result = encode_segment_444(
            y_blocks, cb_blocks, cr_blocks, 0, total_mcus, is_color, config, 0,
        );
        return result.data;
    }

    // Encode segments in parallel
    let segments: Vec<SegmentResult> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_start = seg_idx * interval;
            let mcu_count = interval.min(total_mcus - mcu_start);
            let restart_num = (seg_idx % 8) as u8;

            encode_segment_444(
                y_blocks,
                cb_blocks,
                cr_blocks,
                mcu_start,
                mcu_count,
                is_color,
                config,
                restart_num,
            )
        })
        .collect();

    combine_segments(&segments)
}

/// Parallel entropy encoding for subsampled images (4:2:0, 4:2:2, 4:4:0).
pub fn parallel_entropy_encode_subsampled(
    y_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
    cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
    width: usize,
    height: usize,
    h_samp: usize,
    v_samp: usize,
    is_color: bool,
    restart_interval: u16,
    config: &ParallelEntropyConfig,
) -> Vec<u8> {
    let y_blocks_w = (width + 7) / 8;
    let y_blocks_h = (height + 7) / 8;
    let c_width = (width + h_samp - 1) / h_samp;
    let c_height = (height + v_samp - 1) / v_samp;
    let c_blocks_w = (c_width + 7) / 8;
    let c_blocks_h = (c_height + 7) / 8;

    let mcu_h = (y_blocks_w + h_samp - 1) / h_samp;
    let mcu_v = (y_blocks_h + v_samp - 1) / v_samp;
    let total_mcus = mcu_h * mcu_v;

    let interval = restart_interval as usize;
    let num_segments = (total_mcus + interval - 1) / interval;

    if num_segments <= 1 {
        let result = encode_segment_subsampled(
            y_blocks, cb_blocks, cr_blocks, 0, total_mcus, mcu_h, y_blocks_w, y_blocks_h,
            c_blocks_w, c_blocks_h, h_samp, v_samp, is_color, config, 0,
        );
        return result.data;
    }

    // Encode segments in parallel
    let segments: Vec<SegmentResult> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_start = seg_idx * interval;
            let mcu_count = interval.min(total_mcus - mcu_start);
            let restart_num = (seg_idx % 8) as u8;

            encode_segment_subsampled(
                y_blocks,
                cb_blocks,
                cr_blocks,
                mcu_start,
                mcu_count,
                mcu_h,
                y_blocks_w,
                y_blocks_h,
                c_blocks_w,
                c_blocks_h,
                h_samp,
                v_samp,
                is_color,
                config,
                restart_num,
            )
        })
        .collect();

    combine_segments(&segments)
}

// =============================================================================
// Heuristics
// =============================================================================

/// Determines if parallel encoding is beneficial for the given image size.
///
/// Based on benchmarks across various image sizes and thread counts:
/// - Minimum useful size: ~512x512 (4096 blocks)
/// - 2 threads: 60-80% efficiency, consistent 1.2-1.6x speedup
/// - 4 threads: 30-40% efficiency, 1.3-1.7x speedup
#[inline]
pub fn should_use_parallel(width: u32, height: u32, available_threads: usize) -> bool {
    let blocks = ((width as usize + 7) / 8) * ((height as usize + 7) / 8);
    blocks >= PARALLEL_THRESHOLD && available_threads >= 2
}

/// Returns recommended thread count for the given image size.
///
/// Balances throughput vs efficiency based on benchmarks.
#[inline]
pub fn recommended_threads(width: u32, height: u32, max_threads: usize) -> usize {
    let blocks = ((width as usize + 7) / 8) * ((height as usize + 7) / 8);

    if blocks < PARALLEL_THRESHOLD {
        1
    } else if blocks < 16384 {
        // 512x512 to 1024x1024: 2 threads optimal
        max_threads.min(2)
    } else {
        // Larger images: use available threads up to 4
        max_threads.min(4)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_dct_matches_sequential() {
        let width = 256;
        let height = 16;
        let padded_width = ((width + 7) / 8) * 8;
        let strip: Vec<f32> = (0..height * padded_width)
            .map(|i| (i % 256) as f32)
            .collect();

        let blocks_w = (width + 7) / 8;
        let strip_blocks_h = (height + 7) / 8;
        let total_blocks = blocks_w * strip_blocks_h;

        // Sequential reference
        let mut seq_output = vec![Block8x8f::default(); total_blocks];
        sequential_dct_plane(
            &strip,
            blocks_w,
            total_blocks,
            padded_width,
            None,
            &mut seq_output,
        );

        // Parallel implementation
        let mut par_output = vec![Block8x8f::default(); total_blocks];
        parallel_dct_plane(
            &strip,
            blocks_w,
            total_blocks,
            padded_width,
            None,
            &mut par_output,
        );

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
                        i,
                        row,
                        col,
                        s_arr[col],
                        p_arr[col]
                    );
                }
            }
        }
    }

    #[test]
    fn test_should_use_parallel() {
        // Too small
        assert!(!should_use_parallel(256, 256, 4)); // 1024 blocks

        // Large enough with threads
        assert!(should_use_parallel(512, 512, 2)); // 4096 blocks
        assert!(should_use_parallel(1024, 1024, 4));

        // Large but only 1 thread
        assert!(!should_use_parallel(1024, 1024, 1));
    }

    #[test]
    fn test_recommended_threads() {
        assert_eq!(recommended_threads(256, 256, 8), 1); // Too small
        assert_eq!(recommended_threads(512, 512, 8), 2); // Medium
        assert_eq!(recommended_threads(2048, 2048, 8), 4); // Large
        assert_eq!(recommended_threads(2048, 2048, 2), 2); // Limited threads
    }

    #[test]
    fn test_entropy_config_standard() {
        // Verify construction doesn't panic
        let _config = ParallelEntropyConfig::standard();
    }
}
