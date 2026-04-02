//! Fused parallel encoder: color convert → AQ → DCT → quantize → entropy in one pass per segment.
//!
//! Each rayon task processes a horizontal band of MCU rows independently.
//! AQ at segment boundaries uses edge clamping (same as image edges),
//! producing imperceptible quality differences vs sequential.
//!
//! Requires fixed Huffman tables (no two-pass optimization) and restart markers.

use rayon::prelude::*;

use crate::encode::dct::simd::forward_dct_8x8_wide;
use crate::encode::deringing::preprocess_deringing_block;
use crate::encode::layout::LayoutParams;
use crate::entropy::encoder::EntropyEncoder;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::Block8x8f;
use crate::huffman::optimize::HuffmanTableSet;
use crate::quant::aq::streaming::StreamingAQ;
use crate::types::Subsampling;

use super::parallel::ParallelEntropyConfig;

/// Minimum MCU rows per segment to justify parallel overhead.
const MIN_MCU_ROWS_PER_SEGMENT: usize = 2;

/// Result from encoding one segment.
struct SegmentResult {
    data: Vec<u8>,
    restart_num: u8,
}

/// Shared read-only config for all parallel segments.
struct SharedEncodeConfig {
    width: usize,
    height: usize,
    padded_width: usize,
    blocks_w: usize,
    mcu_height: usize,
    h_samp: usize,
    v_samp: usize,
    subsampling: Subsampling,
    y_quant_values: [u16; DCT_BLOCK_SIZE],
    cb_quant_values: [u16; DCT_BLOCK_SIZE],
    cr_quant_values: [u16; DCT_BLOCK_SIZE],
    y_quant_01: u16,
    deringing: bool,
    aq_enabled: bool,
    entropy_tables: ParallelEntropyConfig,
}

/// Fused parallel encode: processes input RGB → JPEG scan data in parallel segments.
///
/// Returns the encoded scan data (without JPEG headers — caller wraps in headers).
/// Each segment is separated by RST markers.
pub fn fused_parallel_encode(
    rgb_pixels: &[u8],
    width: u32,
    height: u32,
    subsampling: Subsampling,
    tables: &HuffmanTableSet,
    y_quant: &[u16; DCT_BLOCK_SIZE],
    cb_quant: &[u16; DCT_BLOCK_SIZE],
    cr_quant: &[u16; DCT_BLOCK_SIZE],
    restart_mcu_rows: usize,
    deringing: bool,
    aq_enabled: bool,
) -> Result<Vec<u8>> {
    let width = width as usize;
    let height = height as usize;
    let (h_samp, v_samp) = match subsampling {
        Subsampling::S444 => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };
    let mcu_height = v_samp * 8;
    let mcu_rows = (height + mcu_height - 1) / mcu_height;
    let padded_width = ((width + h_samp * 8 - 1) / (h_samp * 8)) * h_samp * 8;
    let blocks_w = padded_width / 8;

    let rows_per_seg = restart_mcu_rows.max(MIN_MCU_ROWS_PER_SEGMENT);
    let num_segments = (mcu_rows + rows_per_seg - 1) / rows_per_seg;

    if num_segments <= 1 {
        return Err(crate::error::Error::unsupported_feature(
            "fused parallel encode requires multiple restart segments",
        ));
    }

    let shared = SharedEncodeConfig {
        width,
        height,
        padded_width,
        blocks_w,
        mcu_height,
        h_samp,
        v_samp,
        subsampling,
        y_quant_values: *y_quant,
        cb_quant_values: *cb_quant,
        cr_quant_values: *cr_quant,
        y_quant_01: y_quant[1],
        deringing,
        aq_enabled,
        entropy_tables: ParallelEntropyConfig {
            dc_luma: tables.dc_luma.table.clone(),
            ac_luma: tables.ac_luma.table.clone(),
            dc_chroma: tables.dc_chroma.table.clone(),
            ac_chroma: tables.ac_chroma.table.clone(),
        },
    };

    // Encode segments in parallel
    let segments: Vec<Result<SegmentResult>> = (0..num_segments)
        .into_par_iter()
        .map(|seg_idx| {
            let mcu_row_start = seg_idx * rows_per_seg;
            let mcu_row_count = rows_per_seg.min(mcu_rows - mcu_row_start);
            let restart_num = (seg_idx % 8) as u8;

            encode_segment(rgb_pixels, &shared, mcu_row_start, mcu_row_count, restart_num)
        })
        .collect();

    // Check for errors and combine
    let mut total_size = 0;
    let mut results = Vec::with_capacity(num_segments);
    for r in segments {
        let seg = r?;
        total_size += seg.data.len() + 2;
        results.push(seg);
    }

    let mut output = Vec::with_capacity(total_size);
    for (i, seg) in results.iter().enumerate() {
        output.extend_from_slice(&seg.data);
        if i < results.len() - 1 {
            output.push(0xFF);
            output.push(0xD0 + seg.restart_num);
        }
    }

    Ok(output)
}

/// Encode a single segment: color convert → AQ → DCT → quantize → entropy.
fn encode_segment(
    rgb_pixels: &[u8],
    shared: &SharedEncodeConfig,
    mcu_row_start: usize,
    mcu_row_count: usize,
    restart_num: u8,
) -> Result<SegmentResult> {
    let width = shared.width;
    let height = shared.height;
    let padded_width = shared.padded_width;
    let mcu_height = shared.mcu_height;
    let h_samp = shared.h_samp;
    let v_samp = shared.v_samp;

    let pixel_row_start = mcu_row_start * mcu_height;
    let pixel_row_end = ((mcu_row_start + mcu_row_count) * mcu_height).min(height);
    let seg_pixel_height = pixel_row_end - pixel_row_start;
    let seg_padded_height = mcu_row_count * mcu_height;

    // 1. Color convert RGB → YCbCr planes
    let y_stride = padded_width;
    let mut y_plane = vec![0.0f32; seg_padded_height * y_stride];

    let c_width = (padded_width + h_samp - 1) / h_samp;
    let c_height = (seg_padded_height + v_samp - 1) / v_samp;
    let mut cb_plane = vec![0.0f32; c_height * c_width];
    let mut cr_plane = vec![0.0f32; c_height * c_width];

    color_convert_segment(
        rgb_pixels, width, height, pixel_row_start, seg_pixel_height, padded_width,
        h_samp, v_samp, &mut y_plane, &mut cb_plane, &mut cr_plane, c_width,
    );

    // 2. Compute AQ strengths
    let seg_blocks_w = shared.blocks_w;
    let seg_blocks_h = mcu_row_count * v_samp; // block rows in this segment
    let aq_strengths = if shared.aq_enabled {
        compute_segment_aq(
            &y_plane, width, seg_pixel_height, y_stride,
            seg_blocks_w, seg_blocks_h, shared.subsampling, shared.y_quant_01,
        )?
    } else {
        vec![0.0f32; seg_blocks_w * seg_blocks_h]
    };

    // 3. DCT + quantize + entropy encode
    let y_dc_quant = shared.y_quant_values[0];
    let mcu_cols = padded_width / (h_samp * 8);
    let total_mcus = mcu_cols * mcu_row_count;

    let est_blocks = total_mcus * (1 + h_samp * v_samp);
    let mut encoder = EntropyEncoder::with_capacity(est_blocks * 3);
    encoder.set_dc_table(0, &shared.entropy_tables.dc_luma);
    encoder.set_ac_table(0, &shared.entropy_tables.ac_luma);
    encoder.set_dc_table(1, &shared.entropy_tables.dc_chroma);
    encoder.set_ac_table(1, &shared.entropy_tables.ac_chroma);

    for local_mcu_row in 0..mcu_row_count {
        for mcu_col in 0..mcu_cols {
            // Y blocks
            for vy in 0..v_samp {
                for hx in 0..h_samp {
                    let block_x = mcu_col * h_samp + hx;
                    let block_y = local_mcu_row * v_samp + vy;

                    let mut block = extract_block_from_plane(
                        &y_plane, y_stride, block_x * 8, block_y * 8, seg_padded_height,
                    );

                    if shared.deringing {
                        preprocess_deringing_block(&mut block, y_dc_quant);
                    }

                    let dct_block = forward_dct_8x8_wide(&block);

                    let aq_idx = block_y * seg_blocks_w + block_x;
                    let aq_strength = aq_strengths.get(aq_idx).copied().unwrap_or(0.0);
                    let quantized = quantize_block_with_aq(
                        &dct_block, &shared.y_quant_values, aq_strength,
                    );

                    encoder.encode_block(&quantized, 0, 0, 0);
                }
            }

            // Cb block
            {
                let cb_block = extract_and_dct_chroma(&cb_plane, c_width, mcu_col, local_mcu_row, c_height);
                let cb_q = quantize_block_no_aq(&cb_block, &shared.cb_quant_values);
                encoder.encode_block(&cb_q, 1, 1, 1);
            }

            // Cr block
            {
                let cr_block = extract_and_dct_chroma(&cr_plane, c_width, mcu_col, local_mcu_row, c_height);
                let cr_q = quantize_block_no_aq(&cr_block, &shared.cr_quant_values);
                encoder.encode_block(&cr_q, 2, 1, 1);
            }

            let mcu_idx = local_mcu_row * mcu_cols + mcu_col;
            if mcu_idx + 1 < total_mcus {
                encoder.check_restart();
            }
        }
    }

    let data = encoder.finish();
    Ok(SegmentResult { data, restart_num })
}

/// Color convert RGB rows to YCbCr planes for a segment.
fn color_convert_segment(
    rgb_pixels: &[u8],
    width: usize,
    height: usize,
    pixel_row_start: usize,
    seg_pixel_height: usize,
    padded_width: usize,
    h_samp: usize,
    v_samp: usize,
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    c_width: usize,
) {
    let rgb_stride = width * 3;

    for local_y in 0..seg_pixel_height {
        let global_y = pixel_row_start + local_y;
        if global_y >= height {
            break;
        }

        let rgb_row = &rgb_pixels[global_y * rgb_stride..(global_y + 1) * rgb_stride];
        let y_row_start = local_y * padded_width;

        for x in 0..width {
            let r = rgb_row[x * 3] as f32;
            let g = rgb_row[x * 3 + 1] as f32;
            let b = rgb_row[x * 3 + 2] as f32;

            y_plane[y_row_start + x] = 0.299 * r + 0.587 * g + 0.114 * b;

            if h_samp == 1 && v_samp == 1 {
                cb_plane[local_y * c_width + x] = 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
                cr_plane[local_y * c_width + x] = 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);
            }
        }

        // Edge-replicate Y to padded width
        if width < padded_width {
            let last_val = y_plane[y_row_start + width - 1];
            for x in width..padded_width {
                y_plane[y_row_start + x] = last_val;
            }
        }
    }

    // Box-filter chroma downsampling for subsampled modes
    if h_samp > 1 || v_samp > 1 {
        box_downsample_chroma(
            rgb_pixels, width, height, pixel_row_start, seg_pixel_height,
            cb_plane, cr_plane, c_width, h_samp, v_samp,
        );
    }
}

/// Box-filter chroma downsampling from RGB input.
fn box_downsample_chroma(
    rgb_pixels: &[u8],
    width: usize,
    height: usize,
    pixel_row_start: usize,
    seg_pixel_height: usize,
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    c_width: usize,
    h_samp: usize,
    v_samp: usize,
) {
    let rgb_stride = width * 3;
    let c_height = (seg_pixel_height + v_samp - 1) / v_samp;

    for cy in 0..c_height {
        for cx in 0..c_width.min((width + h_samp - 1) / h_samp) {
            let mut cb_sum = 0.0f32;
            let mut cr_sum = 0.0f32;
            let mut count = 0.0f32;

            for dy in 0..v_samp {
                for dx in 0..h_samp {
                    let px = cx * h_samp + dx;
                    let py = cy * v_samp + dy;
                    let global_y = pixel_row_start + py;

                    if px < width && global_y < height {
                        let rgb_off = global_y * rgb_stride + px * 3;
                        let r = rgb_pixels[rgb_off] as f32;
                        let g = rgb_pixels[rgb_off + 1] as f32;
                        let b = rgb_pixels[rgb_off + 2] as f32;

                        cb_sum += 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
                        cr_sum += 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                let idx = cy * c_width + cx;
                cb_plane[idx] = cb_sum / count;
                cr_plane[idx] = cr_sum / count;
            }
        }
    }
}

/// Compute AQ strengths for a segment using an independent StreamingAQ instance.
fn compute_segment_aq(
    y_plane: &[f32],
    width: usize,
    seg_height: usize,
    y_stride: usize,
    _blocks_w: usize,
    _blocks_h: usize,
    subsampling: Subsampling,
    y_quant_01: u16,
) -> Result<Vec<f32>> {
    let layout = LayoutParams::new(width, seg_height, subsampling, false);
    let mut aq = StreamingAQ::new(&layout, y_quant_01, true)?;

    let imcu_height = match subsampling {
        Subsampling::S420 | Subsampling::S440 => 16,
        _ => 8,
    };
    let mut all_strengths = Vec::new();

    for strip_y in (0..seg_height).step_by(imcu_height) {
        let strip_h = imcu_height.min(seg_height - strip_y);
        let strip_start = strip_y * y_stride;
        let strip_end = strip_start + strip_h * y_stride;
        let strip_data = &y_plane[strip_start..strip_end];

        if let Some(strengths) = aq.process_y_strip(strip_data, strip_y, strip_h) {
            all_strengths.extend_from_slice(strengths);
        }
    }

    if let Some(strengths) = aq.flush() {
        all_strengths.extend_from_slice(strengths);
    }

    Ok(all_strengths)
}

/// Extract 8x8 f32 block from a contiguous f32 plane.
#[inline]
fn extract_block_from_plane(
    plane: &[f32],
    stride: usize,
    bx_pixels: usize,
    by_pixels: usize,
    plane_height: usize,
) -> Block8x8f {
    let mut block = Block8x8f::ZERO;
    for row in 0..8 {
        let py = by_pixels + row;
        let src_y = if py < plane_height { py } else { plane_height.saturating_sub(1) };
        let src = src_y * stride + bx_pixels;
        if src + 8 <= plane.len() {
            block.rows[row].copy_from_slice(&plane[src..src + 8]);
        }
    }
    block
}

/// Extract and DCT a chroma block.
#[inline]
fn extract_and_dct_chroma(
    plane: &[f32],
    c_width: usize,
    mcu_col: usize,
    local_mcu_row: usize,
    c_height: usize,
) -> Block8x8f {
    let block = extract_block_from_plane(plane, c_width, mcu_col * 8, local_mcu_row * 8, c_height);
    forward_dct_8x8_wide(&block)
}

/// Quantize a DCT block with AQ strength modulation.
#[inline]
fn quantize_block_with_aq(
    dct: &Block8x8f,
    quant: &[u16; DCT_BLOCK_SIZE],
    aq_strength: f32,
) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    let scale = if aq_strength != 0.0 {
        2.0_f32.powf(aq_strength)
    } else {
        1.0
    };
    for row in 0..8 {
        for col in 0..8 {
            let i = row * 8 + col;
            let q = quant[i] as f32 * scale;
            result[i] = if q > 0.0 {
                (dct.rows[row][col] / q).round() as i16
            } else {
                0
            };
        }
    }
    result
}

/// Quantize a DCT block without AQ.
#[inline]
fn quantize_block_no_aq(
    dct: &Block8x8f,
    quant: &[u16; DCT_BLOCK_SIZE],
) -> [i16; DCT_BLOCK_SIZE] {
    let mut result = [0i16; DCT_BLOCK_SIZE];
    for row in 0..8 {
        for col in 0..8 {
            let i = row * 8 + col;
            let q = quant[i] as f32;
            result[i] = if q > 0.0 {
                (dct.rows[row][col] / q).round() as i16
            } else {
                0
            };
        }
    }
    result
}
