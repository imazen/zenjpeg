//! Fused parallel decode: entropy decode + IDCT in a single parallel pass.
//!
//! When DRI is MCU-row-aligned (`restart_interval % mcu_cols == 0`), segments
//! map to disjoint pixel rows. This lets each thread entropy-decode and IDCT
//! immediately (data is cache-hot in L1/L2), writing pixel values directly.
//!
//! ## Paths
//!
//! - **4:4:4 / grayscale**: Single pass — decode → IDCT → color convert → RGB
//! - **4:2:0 + NearestNeighbor**: Single pass — decode → IDCT → fused box upsample+CC → RGB
//! - **4:2:0 + fancy upsample**: Two phases:
//!   1. Parallel decode → IDCT → write to Y/Cb/Cr pixel planes
//!   2. Parallel upsample + color convert → RGB (full ±1 row context available)
//!
//! ## Adaptive Segment Grouping
//!
//! Fine-grained DRI (1 MCU row) is grouped adaptively based on thread count:
//! `rows_per_group = max(1, total_markers / (2 * num_threads))` gives 2× oversubscription
//! for load balancing.

use crate::entropy::EntropyDecoder;
use crate::error::{Result, ScanRead};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;
use crate::quant::dequantize_unzigzag_i32_into_partial;

use super::idct_int::{idct_int_dc_only, idct_int_tiered, idct_int_tiered_libjpeg};
use super::rst_scan::compute_segments;
use super::{ChromaUpsampling, DecodeWarning, Strictness};

use super::parser::JpegParser;

use rayon::prelude::*;

/// Minimum restart segments to justify parallel overhead.
const MIN_SEGMENTS: usize = 4;

/// Minimum total MCUs to justify parallel decode.
const MIN_BLOCKS: usize = 1024;

/// Minimum grouped segments to justify parallel overhead.
const MIN_FUSED_SEGMENTS: usize = 4;

/// Check if parallel decode should be used for this image.
pub(super) fn should_use_parallel(
    restart_interval: u16,
    total_mcus: usize,
    num_rst_markers: usize,
) -> bool {
    if restart_interval == 0 {
        return false;
    }
    let num_segments = num_rst_markers + 1;
    if num_segments < MIN_SEGMENTS {
        return false;
    }
    total_mcus >= MIN_BLOCKS
}

/// Result of a fused decode function: (result, ac_overflow, invalid_huffman, truncation_mcu, padding_error, total_mcus)
type FusedDecodeResult = Result<(FusedResult, bool, bool, Option<u32>, bool, u32)>;

/// IDCT'd pixel planes for the two-phase subsampled path.
#[allow(dead_code)]
pub(super) struct PixelPlanes {
    pub y: Vec<i16>,
    pub cb: Vec<i16>,
    pub cr: Vec<i16>,
    pub y_stride: usize,
    pub c_stride: usize,
    pub y_height: usize,
    pub c_height: usize,
}

/// Result of fused parallel decode.
pub(super) enum FusedResult {
    /// Single-pass: already converted to RGB (4:4:4, grayscale, box-filter 4:2:0)
    Rgb(Vec<u8>),
    /// Two-phase: IDCT'd planes ready for upsample + color convert
    Planes(PixelPlanes),
}

/// Warnings collected from a single segment decode.
struct SegmentWarnings {
    had_ac_overflow: bool,
    had_invalid_huffman: bool,
    truncation_mcu: Option<u32>,
    had_padding_error: bool,
}

impl<'a> JpegParser<'a> {
    /// Try fused parallel decode for baseline JPEG with MCU-row-aligned DRI.
    ///
    /// Returns `Ok(true)` if fused decode was used, `Ok(false)` to fall through
    /// to existing parallel or sequential paths.
    pub(super) fn try_fused_parallel_decode(
        &mut self,
        scan_components: &[(usize, u8, u8)],
    ) -> Result<bool> {
        use super::rst_scan::scan_rst_markers;

        // Quick eligibility checks
        if self.restart_interval == 0 {
            return Ok(false);
        }

        let num_comps = self.num_components as usize;
        if num_comps != 1 && num_comps != 3 {
            return Ok(false);
        }

        // Calculate MCU grid
        let max_h_samp = (0..num_comps)
            .map(|i| self.components[i].h_samp_factor)
            .max()
            .unwrap_or(1) as usize;
        let max_v_samp = (0..num_comps)
            .map(|i| self.components[i].v_samp_factor)
            .max()
            .unwrap_or(1) as usize;
        let mcu_width = max_h_samp * 8;
        let mcu_height = max_v_samp * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;
        let total_mcus = mcu_cols * mcu_rows;

        if total_mcus < 1024 {
            return Ok(false);
        }

        // MCU-row alignment gate
        let ri = self.restart_interval as usize;
        if ri == 0 || ri % mcu_cols != 0 {
            return Ok(false);
        }

        // SIMD RST scan
        let expected_markers = total_mcus / ri;
        let scan_data = &self.data[self.position..];
        let rst_result = scan_rst_markers(scan_data, expected_markers);

        if rst_result.markers.is_empty() {
            return Ok(false);
        }

        // One segment per restart interval (no grouping — rayon handles work distribution)
        let group_stride = 1;
        let (seg_starts, seg_ends) =
            compute_segments(&rst_result.markers, rst_result.entropy_end);
        let num_segments = seg_starts.len();

        // Need enough segments after grouping
        if num_segments < MIN_FUSED_SEGMENTS {
            return Ok(false);
        }

        // Threshold check
        if !should_use_parallel(self.restart_interval, total_mcus, rst_result.markers.len()) {
            return Ok(false);
        }

        // Check for missing DHT
        {
            let mut any_missing = false;
            for (_comp_idx, dc_table, ac_table) in scan_components {
                let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
                let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
                if self.dc_tables[dc_idx].is_none() || self.ac_tables[ac_idx].is_none() {
                    any_missing = true;
                    break;
                }
            }
            if any_missing {
                self.warn(DecodeWarning::MissingHuffmanTables)?;
            }
        }

        // Determine subsampling
        let is_subsampled = num_comps == 3
            && (self.components[1].h_samp_factor != self.components[0].h_samp_factor
                || self.components[1].v_samp_factor != self.components[0].v_samp_factor);

        // Select fused path
        let chroma_upsampling = self.chroma_upsampling;
        let (result, any_ac, any_huff, first_trunc, any_pad, total_mcus) = if !is_subsampled {
            // 4:4:4 or grayscale — single pass
            self.decode_fused_444(
                scan_components,
                scan_data,
                &seg_starts,
                &seg_ends,
                num_segments,
                mcu_cols,
                mcu_rows,
                max_h_samp,
                max_v_samp,
                ri,
                group_stride,
                chroma_upsampling,
            )?
        } else if matches!(chroma_upsampling, ChromaUpsampling::NearestNeighbor) {
            // 4:2:0 + box filter — single pass
            self.decode_fused_subsampled_box(
                scan_components,
                scan_data,
                &seg_starts,
                &seg_ends,
                num_segments,
                mcu_cols,
                mcu_rows,
                max_h_samp,
                max_v_samp,
                ri,
                group_stride,
            )?
        } else {
            // 4:2:0 + fancy upsample — two-phase
            self.decode_fused_subsampled_planes(
                scan_components,
                scan_data,
                &seg_starts,
                &seg_ends,
                num_segments,
                mcu_cols,
                mcu_rows,
                max_h_samp,
                max_v_samp,
                ri,
                group_stride,
                chroma_upsampling,
            )?
        };

        // Advance position past entropy data
        self.position += rst_result.entropy_end;

        // Emit warnings
        if let Some(at_mcu) = first_trunc {
            self.warn(DecodeWarning::TruncatedScan {
                blocks_decoded: at_mcu,
                blocks_expected: total_mcus,
            })?;
        }
        if any_pad {
            self.warn(DecodeWarning::PaddingBlockError)?;
        }
        if any_ac {
            self.warn(DecodeWarning::AcIndexOverflow)?;
        }
        if any_huff {
            self.warn(DecodeWarning::InvalidHuffmanCode)?;
        }

        self.fused_result = Some(result);
        Ok(true)
    }

    /// Build Huffman table arrays for thread-safe parallel access.
    fn build_huffman_tables(
        &self,
        scan_components: &[(usize, u8, u8)],
    ) -> (
        Vec<Option<HuffmanDecodeTable>>,
        Vec<Option<HuffmanDecodeTable>>,
    ) {
        let dc_tables: Vec<Option<HuffmanDecodeTable>> = (0..MAX_HUFFMAN_TABLES)
            .map(|idx| {
                self.dc_tables[idx].clone().or_else(|| {
                    let needed = scan_components
                        .iter()
                        .any(|(_, dc, _)| (*dc as usize).min(MAX_HUFFMAN_TABLES - 1) == idx);
                    if needed {
                        Some(if idx == 0 {
                            HuffmanDecodeTable::std_dc_luminance().clone()
                        } else {
                            HuffmanDecodeTable::std_dc_chrominance().clone()
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        let ac_tables: Vec<Option<HuffmanDecodeTable>> = (0..MAX_HUFFMAN_TABLES)
            .map(|idx| {
                self.ac_tables[idx].clone().or_else(|| {
                    let needed = scan_components
                        .iter()
                        .any(|(_, _, ac)| (*ac as usize).min(MAX_HUFFMAN_TABLES - 1) == idx);
                    if needed {
                        Some(if idx == 0 {
                            HuffmanDecodeTable::std_ac_luminance().clone()
                        } else {
                            HuffmanDecodeTable::std_ac_chrominance().clone()
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        (dc_tables, ac_tables)
    }

    /// Set up an EntropyDecoder with Huffman tables for a segment.
    fn setup_segment_decoder<'d, 't>(
        seg_data: &'d [u8],
        scan_comps: &[(usize, u8, u8)],
        dc_tables: &'t [Option<HuffmanDecodeTable>],
        ac_tables: &'t [Option<HuffmanDecodeTable>],
        lenient: bool,
    ) -> EntropyDecoder<'d, 't> {
        let mut decoder = EntropyDecoder::new(seg_data);
        if lenient {
            decoder.set_lenient(true);
        }
        for (_, dc_table, ac_table) in scan_comps {
            let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
            if let Some(t) = &dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, t);
            }
            if let Some(t) = &ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, t);
            }
        }
        decoder
    }

    /// Compute MCU range for a grouped segment.
    ///
    /// Each grouped segment covers `group_stride` restart intervals.
    fn segment_mcu_range(
        seg_idx: usize,
        num_segments: usize,
        ri: usize,
        group_stride: usize,
        total_mcus: usize,
    ) -> (usize, usize) {
        let mcus_per_group = ri * group_stride;
        let mcu_start = seg_idx * mcus_per_group;
        let mcu_end = if seg_idx + 1 == num_segments {
            total_mcus
        } else {
            ((seg_idx + 1) * mcus_per_group).min(total_mcus)
        };
        (mcu_start, mcu_end)
    }

    /// Single-pass fused decode for 4:4:4 and grayscale.
    ///
    /// Each thread: entropy decode → dequant → IDCT → color convert → write RGB.
    #[allow(clippy::too_many_arguments)]
    fn decode_fused_444(
        &self,
        scan_components: &[(usize, u8, u8)],
        scan_data: &[u8],
        seg_starts: &[usize],
        seg_ends: &[usize],
        num_segments: usize,
        mcu_cols: usize,
        mcu_rows: usize,
        _max_h_samp: usize,
        _max_v_samp: usize,
        ri: usize,
        group_stride: usize,
        chroma_upsampling: ChromaUpsampling,
    ) -> FusedDecodeResult {
        let width = self.width as usize;
        let height = self.height as usize;
        let num_comps = self.num_components as usize;
        let total_mcus = mcu_cols * mcu_rows;
        let strip_width = mcu_cols * 8; // padded width

        // Select IDCT function
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => idct_int_tiered_libjpeg,
            _ => idct_int_tiered,
        };

        // Build thread-safe Huffman tables
        let (dc_tables, ac_tables) = self.build_huffman_tables(scan_components);

        // Get quant tables
        let quant_tables: Vec<&[u16; DCT_BLOCK_SIZE]> = (0..num_comps)
            .map(|ci| {
                self.quant_tables[self.components[ci].quant_table_idx as usize]
                    .as_ref()
                    .unwrap()
            })
            .collect();

        let scan_comps: Vec<(usize, u8, u8)> = scan_components.to_vec();
        let lenient = self.strictness == Strictness::Lenient;

        // Allocate RGB output
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "fused 444 RGB output")?;

        // Compute bytes per pixel row for RGB
        let rgb_row_bytes = width * 3;

        // Each segment covers group_stride restart intervals = group_stride * ri MCUs
        // = group_stride * ri / mcu_cols MCU rows
        // = group_stride * (ri / mcu_cols) MCU rows (ri is MCU-row-aligned)
        let mcu_rows_per_ri = ri / mcu_cols;
        let mcu_rows_per_seg = mcu_rows_per_ri * group_stride;
        let pixel_rows_per_seg = mcu_rows_per_seg * 8; // 4:4:4: MCU height = 8

        // Split RGB output into per-segment chunks
        let seg_rgb_bytes = pixel_rows_per_seg * rgb_row_bytes;
        let rgb_chunks: Vec<&mut [u8]> = rgb.chunks_mut(seg_rgb_bytes).collect();

        // Parallel decode + IDCT + color convert
        let seg_warnings: Vec<Result<SegmentWarnings>> = rgb_chunks
            .into_par_iter()
            .enumerate()
            .map(|(seg_idx, rgb_chunk)| {
                let seg_start = seg_starts[seg_idx];
                let seg_end = seg_ends[seg_idx];
                let seg_data = &scan_data[seg_start..seg_end];

                let (mcu_start, mcu_end) =
                    Self::segment_mcu_range(seg_idx, num_segments, ri, group_stride, total_mcus);

                let mut decoder =
                    Self::setup_segment_decoder(seg_data, &scan_comps, &dc_tables, &ac_tables, lenient);

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;

                // Thread-local strip buffers for one MCU row height
                let strip_pixels = strip_width * 8;
                let mut y_strip: Vec<i16> = vec![0i16; strip_pixels];
                let mut cb_strip: Vec<i16> = if num_comps >= 2 {
                    vec![0i16; strip_pixels]
                } else {
                    Vec::new()
                };
                let mut cr_strip: Vec<i16> = if num_comps >= 3 {
                    vec![0i16; strip_pixels]
                } else {
                    Vec::new()
                };

                // Track MCU row within this segment for strip flushing
                let first_mcu_row = mcu_start / mcu_cols;
                let mut current_mcu_row = first_mcu_row;
                let seg_first_pixel_row = first_mcu_row * 8;

                for mcu_idx in mcu_start..mcu_end {
                    let mcu_row = mcu_idx / mcu_cols;
                    let mcu_col = mcu_idx % mcu_cols;

                    // Flush completed MCU row to RGB
                    if mcu_row != current_mcu_row {
                        // Convert strip to RGB
                        let pixel_row_start = (current_mcu_row * 8).saturating_sub(seg_first_pixel_row);
                        let pixel_rows_this = 8.min(height.saturating_sub(current_mcu_row * 8));
                        let cols_this = width.min(strip_width);

                        for py in 0..pixel_rows_this {
                            let strip_off = py * strip_width;
                            let rgb_off = (pixel_row_start + py) * rgb_row_bytes;
                            if rgb_off + cols_this * 3 > rgb_chunk.len() {
                                break;
                            }

                            if num_comps == 1 {
                                // Grayscale
                                for px in 0..cols_this {
                                    let val = y_strip[strip_off + px].clamp(0, 255) as u8;
                                    let idx = rgb_off + px * 3;
                                    rgb_chunk[idx] = val;
                                    rgb_chunk[idx + 1] = val;
                                    rgb_chunk[idx + 2] = val;
                                }
                            } else {
                                // YCbCr → RGB
                                crate::color::ycbcr_planes_i16_to_rgb_u8(
                                    &y_strip[strip_off..strip_off + cols_this],
                                    &cb_strip[strip_off..strip_off + cols_this],
                                    &cr_strip[strip_off..strip_off + cols_this],
                                    &mut rgb_chunk[rgb_off..rgb_off + cols_this * 3],
                                );
                            }
                        }

                        current_mcu_row = mcu_row;
                    }

                    // Decode blocks for this MCU
                    let base_px = mcu_col * 8;

                    for (sc_idx, (comp_idx, dc_table, ac_table)) in scan_comps.iter().enumerate() {
                        let count = match decoder.decode_block_into(
                            &mut coeffs_buf,
                            prev_coeff_count,
                            *comp_idx,
                            *dc_table as usize,
                            *ac_table as usize,
                        ) {
                            Ok(ScanRead::Value(c)) => c,
                            Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                if truncation_mcu.is_none() {
                                    truncation_mcu = Some(mcu_idx as u32);
                                }
                                coeffs_buf = [0i16; 64];
                                1
                            }
                            Err(e) => return Err(e),
                        };
                        prev_coeff_count = count;

                        // Dequantize + IDCT directly into strip
                        let strip = match sc_idx {
                            0 => &mut y_strip,
                            1 => &mut cb_strip,
                            _ => &mut cr_strip,
                        };

                        if count == 1 {
                            // DC-only fast path
                            let dc = coeffs_buf[0] as i32 * quant_tables[*comp_idx][0] as i32;
                            idct_int_dc_only(dc, &mut strip[base_px..], strip_width);
                        } else {
                            dequantize_unzigzag_i32_into_partial(
                                &coeffs_buf,
                                quant_tables[*comp_idx],
                                &mut dequant_buf,
                                count,
                            );
                            idct_fn(&mut dequant_buf, &mut strip[base_px..], strip_width, count);
                        }
                    }
                }

                // Flush last MCU row
                {
                    let pixel_row_start = (current_mcu_row * 8).saturating_sub(seg_first_pixel_row);
                    let pixel_rows_this = 8.min(height.saturating_sub(current_mcu_row * 8));
                    let cols_this = width.min(strip_width);

                    for py in 0..pixel_rows_this {
                        let strip_off = py * strip_width;
                        let rgb_off = (pixel_row_start + py) * rgb_row_bytes;
                        if rgb_off + cols_this * 3 > rgb_chunk.len() {
                            break;
                        }

                        if num_comps == 1 {
                            for px in 0..cols_this {
                                let val = y_strip[strip_off + px].clamp(0, 255) as u8;
                                let idx = rgb_off + px * 3;
                                rgb_chunk[idx] = val;
                                rgb_chunk[idx + 1] = val;
                                rgb_chunk[idx + 2] = val;
                            }
                        } else {
                            crate::color::ycbcr_planes_i16_to_rgb_u8(
                                &y_strip[strip_off..strip_off + cols_this],
                                &cb_strip[strip_off..strip_off + cols_this],
                                &cr_strip[strip_off..strip_off + cols_this],
                                &mut rgb_chunk[rgb_off..rgb_off + cols_this * 3],
                            );
                        }
                    }
                }

                Ok(SegmentWarnings {
                    had_ac_overflow: decoder.had_ac_overflow,
                    had_invalid_huffman: decoder.had_invalid_huffman,
                    truncation_mcu,
                    had_padding_error: false,
                })
            })
            .collect();

        let (any_ac, any_huff, first_trunc, any_pad) =
            Self::aggregate_fused_warnings(seg_warnings)?;

        Ok((
            FusedResult::Rgb(rgb),
            any_ac,
            any_huff,
            first_trunc,
            any_pad,
            total_mcus as u32,
        ))
    }

    /// Single-pass fused decode for 4:2:0 + NearestNeighbor (box filter).
    ///
    /// Each thread: entropy decode → dequant → IDCT → fused box upsample+CC → write RGB.
    #[allow(clippy::too_many_arguments)]
    fn decode_fused_subsampled_box(
        &self,
        scan_components: &[(usize, u8, u8)],
        scan_data: &[u8],
        seg_starts: &[usize],
        seg_ends: &[usize],
        num_segments: usize,
        mcu_cols: usize,
        mcu_rows: usize,
        max_h_samp: usize,
        max_v_samp: usize,
        ri: usize,
        group_stride: usize,
    ) -> FusedDecodeResult {
        use crate::color::ycbcr::fused_h2v2_box_ycbcr_to_rgb_u8;

        let width = self.width as usize;
        let height = self.height as usize;
        let total_mcus = mcu_cols * mcu_rows;

        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let _mcu_pixel_width = max_h_samp * 8;
        let mcu_pixel_height = max_v_samp * 8;

        // Strip dimensions for one MCU row
        let y_strip_width = mcu_cols * y_h * 8;
        let y_strip_height = y_v * 8;
        let c_strip_width = mcu_cols * self.components[1].h_samp_factor as usize * 8;
        let c_strip_height = self.components[1].v_samp_factor as usize * 8;

        // Select IDCT (box filter always uses standard IDCT)
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = idct_int_tiered;

        let (dc_tables, ac_tables) = self.build_huffman_tables(scan_components);

        let quant_tables: Vec<&[u16; DCT_BLOCK_SIZE]> = (0..3)
            .map(|ci| {
                self.quant_tables[self.components[ci].quant_table_idx as usize]
                    .as_ref()
                    .unwrap()
            })
            .collect();

        let scan_comps: Vec<(usize, u8, u8)> = scan_components.to_vec();
        let lenient = self.strictness == Strictness::Lenient;

        // Component info for sub-block iteration
        let comp_h_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].h_samp_factor as usize)
            .collect();
        let comp_v_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].v_samp_factor as usize)
            .collect();

        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "fused box RGB output")?;

        let rgb_row_bytes = width * 3;
        let mcu_rows_per_ri = ri / mcu_cols;
        let mcu_rows_per_seg = mcu_rows_per_ri * group_stride;
        let pixel_rows_per_seg = mcu_rows_per_seg * mcu_pixel_height;
        let seg_rgb_bytes = pixel_rows_per_seg * rgb_row_bytes;

        let rgb_chunks: Vec<&mut [u8]> = rgb.chunks_mut(seg_rgb_bytes).collect();

        let seg_warnings: Vec<Result<SegmentWarnings>> = rgb_chunks
            .into_par_iter()
            .enumerate()
            .map(|(seg_idx, rgb_chunk)| {
                let seg_start = seg_starts[seg_idx];
                let seg_end = seg_ends[seg_idx];
                let seg_data = &scan_data[seg_start..seg_end];

                let (mcu_start, mcu_end) =
                    Self::segment_mcu_range(seg_idx, num_segments, ri, group_stride, total_mcus);

                let mut decoder =
                    Self::setup_segment_decoder(seg_data, &scan_comps, &dc_tables, &ac_tables, lenient);

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let had_padding_error = false;

                // Thread-local strip buffers
                let mut y_strip: Vec<i16> = vec![0i16; y_strip_width * y_strip_height];
                let mut cb_strip: Vec<i16> = vec![0i16; c_strip_width * c_strip_height];
                let mut cr_strip: Vec<i16> = vec![0i16; c_strip_width * c_strip_height];

                let first_mcu_row = mcu_start / mcu_cols;
                let mut current_mcu_row = first_mcu_row;
                let seg_first_pixel_row = first_mcu_row * mcu_pixel_height;

                // Closure to flush one MCU row of strips to RGB via fused box upsample
                let flush_mcu_row = |current_mcu_row: usize,
                                      y_strip: &[i16],
                                      cb_strip: &[i16],
                                      cr_strip: &[i16],
                                      rgb_chunk: &mut [u8]| {
                    let pixel_row_start =
                        (current_mcu_row * mcu_pixel_height).saturating_sub(seg_first_pixel_row);
                    let pixel_rows_this =
                        mcu_pixel_height.min(height.saturating_sub(current_mcu_row * mcu_pixel_height));
                    let cols_this = width.min(y_strip_width);

                    for py in 0..pixel_rows_this {
                        let y_off = py * y_strip_width;
                        let c_row = py / (max_v_samp / comp_v_samps[1].max(1));
                        let c_off = c_row * c_strip_width;
                        let rgb_off = (pixel_row_start + py) * rgb_row_bytes;
                        if rgb_off + cols_this * 3 > rgb_chunk.len() {
                            break;
                        }
                        fused_h2v2_box_ycbcr_to_rgb_u8(
                            &y_strip[y_off..y_off + cols_this],
                            &cb_strip[c_off..],
                            &cr_strip[c_off..],
                            &mut rgb_chunk[rgb_off..rgb_off + cols_this * 3],
                            cols_this,
                        );
                    }
                };

                for mcu_idx in mcu_start..mcu_end {
                    let mcu_row = mcu_idx / mcu_cols;
                    let mcu_col = mcu_idx % mcu_cols;

                    // Flush completed MCU row
                    if mcu_row != current_mcu_row {
                        flush_mcu_row(current_mcu_row, &y_strip, &cb_strip, &cr_strip, rgb_chunk);
                        current_mcu_row = mcu_row;
                    }

                    // Decode blocks for this MCU (multi-block per component for subsampled)
                    for (sc_idx, (comp_idx, dc_table, ac_table)) in scan_comps.iter().enumerate() {
                        let h_samp = comp_h_samps[*comp_idx];
                        let v_samp = comp_v_samps[*comp_idx];

                        let (strip, strip_stride) = match sc_idx {
                            0 => (&mut y_strip as &mut Vec<i16>, y_strip_width),
                            1 => (&mut cb_strip, c_strip_width),
                            _ => (&mut cr_strip, c_strip_width),
                        };

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let count = match decoder.decode_block_into(
                                    &mut coeffs_buf,
                                    prev_coeff_count,
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(ScanRead::Value(c)) => c,
                                    Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                        if truncation_mcu.is_none() {
                                            truncation_mcu = Some(mcu_idx as u32);
                                        }
                                        coeffs_buf = [0i16; 64];
                                        1
                                    }
                                    Err(e) => return Err(e),
                                };
                                prev_coeff_count = count;

                                let block_px = mcu_col * h_samp * 8 + h * 8;
                                let block_py = v * 8;
                                let strip_off = block_py * strip_stride + block_px;

                                if count == 1 {
                                    let dc = coeffs_buf[0] as i32
                                        * quant_tables[*comp_idx][0] as i32;
                                    idct_int_dc_only(dc, &mut strip[strip_off..], strip_stride);
                                } else {
                                    dequantize_unzigzag_i32_into_partial(
                                        &coeffs_buf,
                                        quant_tables[*comp_idx],
                                        &mut dequant_buf,
                                        count,
                                    );
                                    idct_fn(
                                        &mut dequant_buf,
                                        &mut strip[strip_off..],
                                        strip_stride,
                                        count,
                                    );
                                }
                            }
                        }
                    }
                }

                // Flush last MCU row
                flush_mcu_row(current_mcu_row, &y_strip, &cb_strip, &cr_strip, rgb_chunk);

                Ok(SegmentWarnings {
                    had_ac_overflow: decoder.had_ac_overflow,
                    had_invalid_huffman: decoder.had_invalid_huffman,
                    truncation_mcu,
                    had_padding_error,
                })
            })
            .collect();

        let (any_ac, any_huff, first_trunc, any_pad) =
            Self::aggregate_fused_warnings(seg_warnings)?;

        Ok((
            FusedResult::Rgb(rgb),
            any_ac,
            any_huff,
            first_trunc,
            any_pad,
            total_mcus as u32,
        ))
    }

    /// Two-phase fused decode for subsampled + fancy upsample.
    ///
    /// Phase 1: Parallel entropy decode → dequant → IDCT → write to Y/Cb/Cr planes.
    /// Phase 2: Upsample + color convert in output.rs (convert_from_pixel_planes).
    #[allow(clippy::too_many_arguments)]
    fn decode_fused_subsampled_planes(
        &self,
        scan_components: &[(usize, u8, u8)],
        scan_data: &[u8],
        seg_starts: &[usize],
        seg_ends: &[usize],
        num_segments: usize,
        mcu_cols: usize,
        mcu_rows: usize,
        _max_h_samp: usize,
        _max_v_samp: usize,
        ri: usize,
        group_stride: usize,
        chroma_upsampling: ChromaUpsampling,
    ) -> FusedDecodeResult {
        let _width = self.width as usize;
        let _height = self.height as usize;
        let total_mcus = mcu_cols * mcu_rows;

        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let c_h = self.components[1].h_samp_factor as usize;
        let c_v = self.components[1].v_samp_factor as usize;

        // Plane dimensions (MCU-padded)
        let y_stride = mcu_cols * y_h * 8;
        let y_height = mcu_rows * y_v * 8;
        let c_stride = mcu_cols * c_h * 8;
        let c_height = mcu_rows * c_v * 8;

        let y_plane_size = checked_size_2d(y_stride, y_height)?;
        let c_plane_size = checked_size_2d(c_stride, c_height)?;

        // Select IDCT function
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => idct_int_tiered_libjpeg,
            _ => idct_int_tiered,
        };

        let (dc_tables, ac_tables) = self.build_huffman_tables(scan_components);

        let quant_tables: Vec<&[u16; DCT_BLOCK_SIZE]> = (0..3)
            .map(|ci| {
                self.quant_tables[self.components[ci].quant_table_idx as usize]
                    .as_ref()
                    .unwrap()
            })
            .collect();

        let scan_comps: Vec<(usize, u8, u8)> = scan_components.to_vec();
        let lenient = self.strictness == Strictness::Lenient;

        let comp_h_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].h_samp_factor as usize)
            .collect();
        let comp_v_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].v_samp_factor as usize)
            .collect();

        // Allocate planes
        let mut y_plane: Vec<i16> = try_alloc_maybeuninit(y_plane_size, "fused Y plane")?;
        let mut cb_plane: Vec<i16> = try_alloc_maybeuninit(c_plane_size, "fused Cb plane")?;
        let mut cr_plane: Vec<i16> = try_alloc_maybeuninit(c_plane_size, "fused Cr plane")?;

        // Compute rows per segment for splitting planes
        let mcu_rows_per_ri = ri / mcu_cols;
        let mcu_rows_per_seg = mcu_rows_per_ri * group_stride;

        let y_rows_per_seg = mcu_rows_per_seg * y_v * 8;
        let c_rows_per_seg = mcu_rows_per_seg * c_v * 8;

        let y_pixels_per_seg = y_rows_per_seg * y_stride;
        let c_pixels_per_seg = c_rows_per_seg * c_stride;

        // Split planes into per-segment slices
        let y_chunks: Vec<&mut [i16]> = y_plane.chunks_mut(y_pixels_per_seg).collect();
        let cb_chunks: Vec<&mut [i16]> = cb_plane.chunks_mut(c_pixels_per_seg).collect();
        let cr_chunks: Vec<&mut [i16]> = cr_plane.chunks_mut(c_pixels_per_seg).collect();

        // Zip into (y, cb, cr) per segment
        let plane_chunks: Vec<(&mut [i16], &mut [i16], &mut [i16])> = y_chunks
            .into_iter()
            .zip(cb_chunks)
            .zip(cr_chunks)
            .map(|((y, cb), cr)| (y, cb, cr))
            .collect();

        // Parallel phase 1: decode + IDCT into planes
        let seg_warnings: Vec<Result<SegmentWarnings>> = plane_chunks
            .into_par_iter()
            .enumerate()
            .map(|(seg_idx, (y_seg, cb_seg, cr_seg))| {
                let seg_start = seg_starts[seg_idx];
                let seg_end = seg_ends[seg_idx];
                let seg_data = &scan_data[seg_start..seg_end];

                let (mcu_start, mcu_end) =
                    Self::segment_mcu_range(seg_idx, num_segments, ri, group_stride, total_mcus);

                let mut decoder =
                    Self::setup_segment_decoder(seg_data, &scan_comps, &dc_tables, &ac_tables, lenient);

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let had_padding_error = false;

                // Segment's first MCU row
                let first_mcu_row = mcu_start / mcu_cols;

                for mcu_idx in mcu_start..mcu_end {
                    let mcu_row = mcu_idx / mcu_cols;
                    let mcu_col = mcu_idx % mcu_cols;

                    for (_sc_idx, (comp_idx, dc_table, ac_table)) in scan_comps.iter().enumerate()
                    {
                        let h_samp = comp_h_samps[*comp_idx];
                        let v_samp = comp_v_samps[*comp_idx];

                        let (seg_slice, stride) = match *comp_idx {
                            0 => (&mut *y_seg, y_stride),
                            1 => (&mut *cb_seg, c_stride),
                            _ => (&mut *cr_seg, c_stride),
                        };

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let count = match decoder.decode_block_into(
                                    &mut coeffs_buf,
                                    prev_coeff_count,
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(ScanRead::Value(c)) => c,
                                    Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                        if truncation_mcu.is_none() {
                                            truncation_mcu = Some(mcu_idx as u32);
                                        }
                                        coeffs_buf = [0i16; 64];
                                        1
                                    }
                                    Err(e) => return Err(e),
                                };
                                prev_coeff_count = count;

                                // Compute position within the segment slice
                                let block_x = mcu_col * h_samp * 8 + h * 8;
                                let block_y =
                                    (mcu_row - first_mcu_row) * v_samp * 8 + v * 8;
                                let off = block_y * stride + block_x;

                                if count == 1 {
                                    let dc = coeffs_buf[0] as i32
                                        * quant_tables[*comp_idx][0] as i32;
                                    if off < seg_slice.len() {
                                        idct_int_dc_only(dc, &mut seg_slice[off..], stride);
                                    }
                                } else {
                                    dequantize_unzigzag_i32_into_partial(
                                        &coeffs_buf,
                                        quant_tables[*comp_idx],
                                        &mut dequant_buf,
                                        count,
                                    );
                                    if off < seg_slice.len() {
                                        idct_fn(
                                            &mut dequant_buf,
                                            &mut seg_slice[off..],
                                            stride,
                                            count,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                Ok(SegmentWarnings {
                    had_ac_overflow: decoder.had_ac_overflow,
                    had_invalid_huffman: decoder.had_invalid_huffman,
                    truncation_mcu,
                    had_padding_error,
                })
            })
            .collect();

        let (any_ac, any_huff, first_trunc, any_pad) =
            Self::aggregate_fused_warnings(seg_warnings)?;

        Ok((
            FusedResult::Planes(PixelPlanes {
                y: y_plane,
                cb: cb_plane,
                cr: cr_plane,
                y_stride,
                c_stride,
                y_height,
                c_height,
            }),
            any_ac,
            any_huff,
            first_trunc,
            any_pad,
            total_mcus as u32,
        ))
    }

    /// Aggregate warnings from fused decode segments.
    ///
    /// Returns aggregated warning flags. Propagates any errors from segments.
    fn aggregate_fused_warnings(
        seg_warnings: Vec<Result<SegmentWarnings>>,
    ) -> Result<(bool, bool, Option<u32>, bool)> {
        let mut any_ac_overflow = false;
        let mut any_invalid_huffman = false;
        let mut first_truncation: Option<u32> = None;
        let mut any_padding_error = false;

        for result in seg_warnings {
            let w = result?;
            any_ac_overflow |= w.had_ac_overflow;
            any_invalid_huffman |= w.had_invalid_huffman;
            any_padding_error |= w.had_padding_error;
            if let Some(mcu) = w.truncation_mcu {
                first_truncation = Some(match first_truncation {
                    Some(existing) => existing.min(mcu),
                    None => mcu,
                });
            }
        }

        Ok((
            any_ac_overflow,
            any_invalid_huffman,
            first_truncation,
            any_padding_error,
        ))
    }
}
