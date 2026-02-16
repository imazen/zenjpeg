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
//! - **4:2:0 + fancy upsample**: Single pass with double-buffered extended chroma strips.
//!   Each thread processes MCU rows with a 1-row lag for ±1 chroma context. Boundary
//!   fixup pass after all threads complete corrects the 2 rows per segment junction
//!   where edge replication was used.
//! - **4:2:2 (h2v1)**: Single pass — decode → IDCT → horizontal upsample Cb/Cr → CC → RGB.
//!   No vertical subsampling, so no cross-segment boundary fixup needed.
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
use super::upsample::MAX_UPSAMPLE_SCRATCH;
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

/// Result of fused parallel decode — always single-pass RGB.
pub(super) struct FusedResult(pub Vec<u8>);

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
        if self.num_threads == 1 {
            return Ok(false);
        }
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
        let (seg_starts, seg_ends) = compute_segments(&rst_result.markers, rst_result.entropy_end);
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
            // Subsampled + fancy upsample
            let y_h = self.components[0].h_samp_factor as usize;
            let y_v = self.components[0].v_samp_factor as usize;
            let c_h = self.components[1].h_samp_factor as usize;
            let c_v = self.components[1].v_samp_factor as usize;
            let h_ratio = y_h / c_h.max(1);
            let v_ratio = y_v / c_v.max(1);
            if h_ratio == 2 && v_ratio == 2 {
                // h2v2 (4:2:0) — extended chroma strips with boundary fixup
                self.decode_fused_subsampled_fancy(
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
            } else if h_ratio == 2 && v_ratio == 1 {
                // h2v1 (4:2:2) — horizontal-only upsample, no vertical context
                self.decode_fused_subsampled_h2v1(
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
            } else {
                return Ok(false); // Fall through to sequential for rare modes
            }
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
        permissive_rst: bool,
    ) -> EntropyDecoder<'d, 't> {
        let mut decoder = EntropyDecoder::new(seg_data);
        if lenient {
            decoder.set_lenient(true);
        }
        if permissive_rst {
            decoder.set_permissive_rst(true);
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
        let lenient = matches!(
            self.strictness,
            Strictness::Lenient | Strictness::Permissive
        );
        let permissive_rst = self.strictness == Strictness::Permissive;
        let strict = self.strictness == Strictness::Strict;

        // Pre-compute per-component actual block counts for padding detection.
        // 4:4:4: all components have h_samp=1, v_samp=1, so actual_blocks = ceil(dim/8).
        let actual_blocks_h: Vec<usize> = scan_comps
            .iter()
            .map(|(ci, _, _)| {
                let h = self.components[*ci].h_samp_factor as usize;
                let comp_w = (width * h + _max_h_samp - 1) / _max_h_samp;
                (comp_w + 7) / 8
            })
            .collect();
        let actual_blocks_v: Vec<usize> = scan_comps
            .iter()
            .map(|(ci, _, _)| {
                let v = self.components[*ci].v_samp_factor as usize;
                let comp_h = (height * v + _max_v_samp - 1) / _max_v_samp;
                (comp_h + 7) / 8
            })
            .collect();

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

                let mut decoder = Self::setup_segment_decoder(
                    seg_data,
                    &scan_comps,
                    &dc_tables,
                    &ac_tables,
                    lenient,
                    permissive_rst,
                );

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let mut had_padding_error = false;

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
                        let pixel_row_start =
                            (current_mcu_row * 8).saturating_sub(seg_first_pixel_row);
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
                        // Check if this block is beyond actual image bounds (padding).
                        // Save decoder state only for padding blocks (rare: edge MCUs only).
                        let is_padding = mcu_col >= actual_blocks_h[sc_idx]
                            || mcu_row >= actual_blocks_v[sc_idx];
                        let padding_state = if is_padding && !strict {
                            Some(decoder.save_state())
                        } else {
                            None
                        };

                        let count = match decoder.decode_block_into(
                            &mut coeffs_buf,
                            prev_coeff_count,
                            *comp_idx,
                            *dc_table as usize,
                            *ac_table as usize,
                        ) {
                            Ok(ScanRead::Value(c)) => c,
                            Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                if let Some(state) = padding_state {
                                    decoder.restore_state(state);
                                    coeffs_buf = [0i16; 64];
                                    had_padding_error = true;
                                    prev_coeff_count = 64;
                                    continue;
                                }
                                if truncation_mcu.is_none() {
                                    truncation_mcu = Some(mcu_idx as u32);
                                }
                                coeffs_buf = [0i16; 64];
                                1
                            }
                            Err(e) => {
                                if let Some(state) = padding_state {
                                    decoder.restore_state(state);
                                    coeffs_buf = [0i16; 64];
                                    had_padding_error = true;
                                    prev_coeff_count = 64;
                                    continue;
                                }
                                return Err(e);
                            }
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
                    had_padding_error,
                })
            })
            .collect();

        let (any_ac, any_huff, first_trunc, any_pad) =
            Self::aggregate_fused_warnings(seg_warnings)?;

        Ok((
            FusedResult(rgb),
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
        let lenient = matches!(
            self.strictness,
            Strictness::Lenient | Strictness::Permissive
        );
        let permissive_rst = self.strictness == Strictness::Permissive;
        let strict = self.strictness == Strictness::Strict;

        // Component info for sub-block iteration
        let comp_h_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].h_samp_factor as usize)
            .collect();
        let comp_v_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].v_samp_factor as usize)
            .collect();

        // Pre-compute per-component actual block counts for padding detection
        let actual_blocks_h: Vec<usize> = (0..3)
            .map(|ci| {
                let h = self.components[ci].h_samp_factor as usize;
                let comp_w = (width * h + max_h_samp - 1) / max_h_samp;
                (comp_w + 7) / 8
            })
            .collect();
        let actual_blocks_v: Vec<usize> = (0..3)
            .map(|ci| {
                let v = self.components[ci].v_samp_factor as usize;
                let comp_h = (height * v + max_v_samp - 1) / max_v_samp;
                (comp_h + 7) / 8
            })
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

                let mut decoder = Self::setup_segment_decoder(
                    seg_data,
                    &scan_comps,
                    &dc_tables,
                    &ac_tables,
                    lenient,
                    permissive_rst,
                );

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let mut had_padding_error = false;

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
                    let pixel_rows_this = mcu_pixel_height
                        .min(height.saturating_sub(current_mcu_row * mcu_pixel_height));
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
                                // Check if this sub-block is beyond actual image bounds
                                let block_x = mcu_col * h_samp + h;
                                let block_y = mcu_row * v_samp + v;
                                let is_padding = block_x >= actual_blocks_h[*comp_idx]
                                    || block_y >= actual_blocks_v[*comp_idx];

                                let padding_state = if is_padding && !strict {
                                    Some(decoder.save_state())
                                } else {
                                    None
                                };

                                let count = match decoder.decode_block_into(
                                    &mut coeffs_buf,
                                    prev_coeff_count,
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(ScanRead::Value(c)) => c,
                                    Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                        if let Some(state) = padding_state {
                                            decoder.restore_state(state);
                                            coeffs_buf = [0i16; 64];
                                            had_padding_error = true;
                                            prev_coeff_count = 64;
                                            continue;
                                        }
                                        if truncation_mcu.is_none() {
                                            truncation_mcu = Some(mcu_idx as u32);
                                        }
                                        coeffs_buf = [0i16; 64];
                                        1
                                    }
                                    Err(e) => {
                                        if let Some(state) = padding_state {
                                            decoder.restore_state(state);
                                            coeffs_buf = [0i16; 64];
                                            had_padding_error = true;
                                            prev_coeff_count = 64;
                                            continue;
                                        }
                                        return Err(e);
                                    }
                                };
                                prev_coeff_count = count;

                                let block_px = mcu_col * h_samp * 8 + h * 8;
                                let block_py = v * 8;
                                let strip_off = block_py * strip_stride + block_px;

                                if count == 1 {
                                    let dc =
                                        coeffs_buf[0] as i32 * quant_tables[*comp_idx][0] as i32;
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
            FusedResult(rgb),
            any_ac,
            any_huff,
            first_trunc,
            any_pad,
            total_mcus as u32,
        ))
    }

    /// Single-pass fused decode for 4:2:2 (h2v1) + fancy/libjpeg-compat upsample.
    ///
    /// Simpler than h2v2: no vertical subsampling means no cross-segment chroma
    /// context and no boundary fixup. Each thread: entropy decode → IDCT →
    /// horizontal upsample Cb/Cr → YCbCr→RGB color convert → write output.
    #[allow(clippy::too_many_arguments)]
    fn decode_fused_subsampled_h2v1(
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
        chroma_upsampling: ChromaUpsampling,
    ) -> FusedDecodeResult {
        use super::upsample::{
            upsample_h2v1_i16_fancy, upsample_h2v1_i16_libjpeg, upsample_h2v1_i16_nearest,
        };

        let width = self.width as usize;
        let height = self.height as usize;
        let total_mcus = mcu_cols * mcu_rows;

        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let mcu_pixel_height = max_v_samp * 8; // v_samp=1 for h2v1, so 8

        // Strip dimensions for one MCU row
        let y_strip_width = mcu_cols * y_h * 8;
        let y_strip_height = y_v * 8;
        let c_strip_width = mcu_cols * self.components[1].h_samp_factor as usize * 8;
        let c_strip_height = self.components[1].v_samp_factor as usize * 8;

        // Select IDCT function
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => idct_int_tiered_libjpeg,
            _ => idct_int_tiered,
        };

        // Select h2v1 upsample function
        type UpsampleFn = fn(&[i16], usize, usize, &mut [i16], usize, usize);
        let upsample_fn: UpsampleFn = match chroma_upsampling {
            ChromaUpsampling::NearestNeighbor => upsample_h2v1_i16_nearest,
            ChromaUpsampling::LibjpegCompat => upsample_h2v1_i16_libjpeg,
            _ => upsample_h2v1_i16_fancy,
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
        let lenient = matches!(
            self.strictness,
            Strictness::Lenient | Strictness::Permissive
        );
        let permissive_rst = self.strictness == Strictness::Permissive;
        let strict = self.strictness == Strictness::Strict;

        let comp_h_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].h_samp_factor as usize)
            .collect();
        let comp_v_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].v_samp_factor as usize)
            .collect();

        // Pre-compute per-component actual block counts for padding detection
        let actual_blocks_h: Vec<usize> = (0..3)
            .map(|ci| {
                let h = self.components[ci].h_samp_factor as usize;
                let comp_w = (width * h + max_h_samp - 1) / max_h_samp;
                (comp_w + 7) / 8
            })
            .collect();
        let actual_blocks_v: Vec<usize> = (0..3)
            .map(|ci| {
                let v = self.components[ci].v_samp_factor as usize;
                let comp_h = (height * v + max_v_samp - 1) / max_v_samp;
                (comp_h + 7) / 8
            })
            .collect();

        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "fused h2v1 RGB output")?;

        let rgb_row_bytes = width * 3;
        let mcu_rows_per_ri = ri / mcu_cols;
        let mcu_rows_per_seg = mcu_rows_per_ri * group_stride;
        let pixel_rows_per_seg = mcu_rows_per_seg * mcu_pixel_height;
        let seg_rgb_bytes = pixel_rows_per_seg * rgb_row_bytes;
        let y_cols_this = width.min(y_strip_width);

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

                let mut decoder = Self::setup_segment_decoder(
                    seg_data,
                    &scan_comps,
                    &dc_tables,
                    &ac_tables,
                    lenient,
                    permissive_rst,
                );

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let mut had_padding_error = false;

                // Thread-local strip buffers
                let mut y_strip: Vec<i16> = vec![0i16; y_strip_width * y_strip_height];
                let mut cb_strip: Vec<i16> = vec![0i16; c_strip_width * c_strip_height];
                let mut cr_strip: Vec<i16> = vec![0i16; c_strip_width * c_strip_height];
                // Upsampled chroma (same height, double width)
                let mut cb_up: Vec<i16> = vec![0i16; y_strip_width * c_strip_height];
                let mut cr_up: Vec<i16> = vec![0i16; y_strip_width * c_strip_height];

                let first_mcu_row = mcu_start / mcu_cols;
                let mut current_mcu_row = first_mcu_row;
                let seg_first_pixel_row = first_mcu_row * mcu_pixel_height;

                // Flush one MCU row: upsample chroma horizontally, then color convert
                let flush_mcu_row = |current_mcu_row: usize,
                                     y_strip: &[i16],
                                     cb_strip: &[i16],
                                     cr_strip: &[i16],
                                     cb_up: &mut [i16],
                                     cr_up: &mut [i16],
                                     rgb_chunk: &mut [u8]| {
                    // Horizontal upsample Cb/Cr: c_strip_width → y_strip_width
                    upsample_fn(
                        cb_strip,
                        c_strip_width,
                        c_strip_height,
                        cb_up,
                        y_strip_width,
                        c_strip_height,
                    );
                    upsample_fn(
                        cr_strip,
                        c_strip_width,
                        c_strip_height,
                        cr_up,
                        y_strip_width,
                        c_strip_height,
                    );

                    let pixel_row_start =
                        (current_mcu_row * mcu_pixel_height).saturating_sub(seg_first_pixel_row);
                    let pixel_rows_this = mcu_pixel_height
                        .min(height.saturating_sub(current_mcu_row * mcu_pixel_height));

                    for py in 0..pixel_rows_this {
                        let y_off = py * y_strip_width;
                        let c_off = py * y_strip_width; // same height, upsampled width
                        let rgb_off = (pixel_row_start + py) * rgb_row_bytes;
                        if rgb_off + y_cols_this * 3 > rgb_chunk.len() {
                            break;
                        }
                        crate::color::ycbcr_planes_i16_to_rgb_u8(
                            &y_strip[y_off..y_off + y_cols_this],
                            &cb_up[c_off..c_off + y_cols_this],
                            &cr_up[c_off..c_off + y_cols_this],
                            &mut rgb_chunk[rgb_off..rgb_off + y_cols_this * 3],
                        );
                    }
                };

                for mcu_idx in mcu_start..mcu_end {
                    let mcu_row = mcu_idx / mcu_cols;
                    let mcu_col = mcu_idx % mcu_cols;

                    // Flush completed MCU row
                    if mcu_row != current_mcu_row {
                        flush_mcu_row(
                            current_mcu_row,
                            &y_strip,
                            &cb_strip,
                            &cr_strip,
                            &mut cb_up,
                            &mut cr_up,
                            rgb_chunk,
                        );
                        current_mcu_row = mcu_row;
                    }

                    // Decode blocks for this MCU
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
                                let block_x = mcu_col * h_samp + h;
                                let block_y = mcu_row * v_samp + v;
                                let is_padding = block_x >= actual_blocks_h[*comp_idx]
                                    || block_y >= actual_blocks_v[*comp_idx];

                                let padding_state = if is_padding && !strict {
                                    Some(decoder.save_state())
                                } else {
                                    None
                                };

                                let count = match decoder.decode_block_into(
                                    &mut coeffs_buf,
                                    prev_coeff_count,
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(ScanRead::Value(c)) => c,
                                    Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                        if let Some(state) = padding_state {
                                            decoder.restore_state(state);
                                            coeffs_buf = [0i16; 64];
                                            had_padding_error = true;
                                            prev_coeff_count = 64;
                                            continue;
                                        }
                                        if truncation_mcu.is_none() {
                                            truncation_mcu = Some(mcu_idx as u32);
                                        }
                                        coeffs_buf = [0i16; 64];
                                        1
                                    }
                                    Err(e) => {
                                        if let Some(state) = padding_state {
                                            decoder.restore_state(state);
                                            coeffs_buf = [0i16; 64];
                                            had_padding_error = true;
                                            prev_coeff_count = 64;
                                            continue;
                                        }
                                        return Err(e);
                                    }
                                };
                                prev_coeff_count = count;

                                let block_px = mcu_col * h_samp * 8 + h * 8;
                                let block_py = v * 8;
                                let strip_off = block_py * strip_stride + block_px;

                                if count == 1 {
                                    let dc =
                                        coeffs_buf[0] as i32 * quant_tables[*comp_idx][0] as i32;
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
                flush_mcu_row(
                    current_mcu_row,
                    &y_strip,
                    &cb_strip,
                    &cr_strip,
                    &mut cb_up,
                    &mut cr_up,
                    rgb_chunk,
                );

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
            FusedResult(rgb),
            any_ac,
            any_huff,
            first_trunc,
            any_pad,
            total_mcus as u32,
        ))
    }

    /// Single-pass fused decode for subsampled + fancy upsample (h2v2 only).
    ///
    /// Each thread uses double-buffered extended chroma strips (c_strip_height + 2
    /// rows) with a 1-MCU-row lag: decode MCU row N+1, then output MCU row N
    /// using N+1's first chroma row as below context. At segment boundaries,
    /// edge replication is used (identical to image-edge behavior).
    ///
    /// After all segments complete, a sequential fixup pass corrects the 2
    /// pixel rows per segment junction where edge replication differed from the
    /// real adjacent chroma, achieving exact parity with the sequential path.
    #[allow(clippy::too_many_arguments)]
    fn decode_fused_subsampled_fancy(
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
        use super::upsample::{
            upsample_h2v2_i16_fancy, upsample_h2v2_i16_fancy_reuse_scratch,
            upsample_h2v2_i16_libjpeg, upsample_row_h2_fancy_bilinear,
        };

        let width = self.width as usize;
        let height = self.height as usize;
        let total_mcus = mcu_cols * mcu_rows;

        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let c_v = self.components[1].v_samp_factor as usize;

        let mcu_pixel_height = y_v * 8; // y_v == max_v_samp for h2v2
        let v_ratio = y_v / c_v; // 2 for h2v2

        // Strip dimensions for one MCU row
        let y_strip_width = mcu_cols * y_h * 8;
        let y_strip_height = y_v * 8;
        let c_strip_width = mcu_cols * self.components[1].h_samp_factor as usize * 8;
        let c_strip_height = c_v * 8;

        let ext_height = c_strip_height + 2;

        // Select IDCT function
        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => idct_int_tiered_libjpeg,
            _ => idct_int_tiered,
        };

        // Select upsample function
        type UpsampleFn = fn(&[i16], usize, usize, &mut [i16], usize, usize);
        let upsample_fn: UpsampleFn = match chroma_upsampling {
            ChromaUpsampling::LibjpegCompat => upsample_h2v2_i16_libjpeg,
            _ => upsample_h2v2_i16_fancy,
        };
        let use_scratch_upsample = matches!(chroma_upsampling, ChromaUpsampling::Triangle)
            && c_strip_width <= MAX_UPSAMPLE_SCRATCH;
        let do_fixup = matches!(chroma_upsampling, ChromaUpsampling::Triangle);

        let (dc_tables, ac_tables) = self.build_huffman_tables(scan_components);

        let quant_tables: Vec<&[u16; DCT_BLOCK_SIZE]> = (0..3)
            .map(|ci| {
                self.quant_tables[self.components[ci].quant_table_idx as usize]
                    .as_ref()
                    .unwrap()
            })
            .collect();

        let scan_comps: Vec<(usize, u8, u8)> = scan_components.to_vec();
        let lenient = matches!(
            self.strictness,
            Strictness::Lenient | Strictness::Permissive
        );
        let permissive_rst = self.strictness == Strictness::Permissive;
        let strict = self.strictness == Strictness::Strict;
        let max_h_samp = _max_h_samp;
        let max_v_samp = _max_v_samp;

        let comp_h_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].h_samp_factor as usize)
            .collect();
        let comp_v_samps: Vec<usize> = (0..3)
            .map(|ci| self.components[ci].v_samp_factor as usize)
            .collect();

        // Pre-compute per-component actual block counts for padding detection
        let actual_blocks_h: Vec<usize> = (0..3)
            .map(|ci| {
                let h = self.components[ci].h_samp_factor as usize;
                let comp_w = (width * h + max_h_samp - 1) / max_h_samp;
                (comp_w + 7) / 8
            })
            .collect();
        let actual_blocks_v: Vec<usize> = (0..3)
            .map(|ci| {
                let v = self.components[ci].v_samp_factor as usize;
                let comp_h = (height * v + max_v_samp - 1) / max_v_samp;
                (comp_h + 7) / 8
            })
            .collect();

        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb: Vec<u8> = try_alloc_maybeuninit(rgb_size, "fused fancy RGB output")?;

        let rgb_row_bytes = width * 3;
        let mcu_rows_per_ri = ri / mcu_cols;
        let mcu_rows_per_seg = mcu_rows_per_ri * group_stride;
        let pixel_rows_per_seg = mcu_rows_per_seg * mcu_pixel_height;
        let seg_rgb_bytes = pixel_rows_per_seg * rgb_row_bytes;

        let rgb_chunks: Vec<&mut [u8]> = rgb.chunks_mut(seg_rgb_bytes).collect();

        let upsample_out_height = ext_height * v_ratio;
        let upsample_out_size = upsample_out_height * y_strip_width;
        let y_cols_this_image = width.min(y_strip_width);

        // Boundary data saved per segment for the fixup pass
        struct SegmentBoundary {
            first_cb_row: Vec<i16>,
            first_cr_row: Vec<i16>,
            last_cb_row: Vec<i16>,
            last_cr_row: Vec<i16>,
            first_y_row: Vec<i16>,
            last_y_row: Vec<i16>,
        }

        // Parallel decode + IDCT + upsample + color convert
        let results: Vec<Result<(SegmentWarnings, SegmentBoundary)>> = rgb_chunks
            .into_par_iter()
            .enumerate()
            .map(|(seg_idx, rgb_chunk)| {
                let seg_start = seg_starts[seg_idx];
                let seg_end = seg_ends[seg_idx];
                let seg_data = &scan_data[seg_start..seg_end];

                let (mcu_start, mcu_end) =
                    Self::segment_mcu_range(seg_idx, num_segments, ri, group_stride, total_mcus);

                let mut decoder = Self::setup_segment_decoder(
                    seg_data,
                    &scan_comps,
                    &dc_tables,
                    &ac_tables,
                    lenient,
                    permissive_rst,
                );

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let mut had_padding_error = false;

                // Double-buffered Y strips (one MCU row each)
                let y_strip_size = y_strip_width * y_strip_height;
                let mut y_strip_a: Vec<i16> = vec![0i16; y_strip_size];
                let mut y_strip_b: Vec<i16> = vec![0i16; y_strip_size];

                // Double-buffered extended chroma strips
                let ext_size = ext_height * c_strip_width;
                let mut ext_cb_a: Vec<i16> = vec![0i16; ext_size];
                let mut ext_cb_b: Vec<i16> = vec![0i16; ext_size];
                let mut ext_cr_a: Vec<i16> = vec![0i16; ext_size];
                let mut ext_cr_b: Vec<i16> = vec![0i16; ext_size];

                // Upsampled chroma output
                let mut cb_up: Vec<i16> = vec![0i16; upsample_out_size];
                let mut cr_up: Vec<i16> = vec![0i16; upsample_out_size];

                // Scratch for Triangle upsample
                let mut upsample_scratch = [0i16; MAX_UPSAMPLE_SCRATCH];

                // Boundary data to save
                let mut boundary = SegmentBoundary {
                    first_cb_row: vec![0i16; c_strip_width],
                    first_cr_row: vec![0i16; c_strip_width],
                    last_cb_row: vec![0i16; c_strip_width],
                    last_cr_row: vec![0i16; c_strip_width],
                    first_y_row: vec![0i16; y_strip_width],
                    last_y_row: vec![0i16; y_strip_width],
                };

                let first_mcu_row = mcu_start / mcu_cols;
                let last_mcu_idx = (mcu_end - 1).max(mcu_start);
                let last_mcu_row = last_mcu_idx / mcu_cols;
                let seg_mcu_rows = last_mcu_row - first_mcu_row + 1;
                let seg_first_pixel_row = first_mcu_row * mcu_pixel_height;

                // Helper closure: upsample extended chroma strip and color convert
                // one MCU row to RGB output
                let upsample_and_output =
                    |mcu_row: usize,
                     y_strip: &[i16],
                     ext_cb: &[i16],
                     ext_cr: &[i16],
                     cb_up: &mut [i16],
                     cr_up: &mut [i16],
                     scratch: &mut [i16; MAX_UPSAMPLE_SCRATCH],
                     rgb_chunk: &mut [u8]| {
                        if use_scratch_upsample {
                            upsample_h2v2_i16_fancy_reuse_scratch(
                                ext_cb,
                                c_strip_width,
                                ext_height,
                                cb_up,
                                y_strip_width,
                                upsample_out_height,
                                scratch,
                            );
                            upsample_h2v2_i16_fancy_reuse_scratch(
                                ext_cr,
                                c_strip_width,
                                ext_height,
                                cr_up,
                                y_strip_width,
                                upsample_out_height,
                                scratch,
                            );
                        } else {
                            upsample_fn(
                                ext_cb,
                                c_strip_width,
                                ext_height,
                                cb_up,
                                y_strip_width,
                                upsample_out_height,
                            );
                            upsample_fn(
                                ext_cr,
                                c_strip_width,
                                ext_height,
                                cr_up,
                                y_strip_width,
                                upsample_out_height,
                            );
                        }

                        let pixel_row_start =
                            (mcu_row * mcu_pixel_height).saturating_sub(seg_first_pixel_row);
                        let pixel_rows_this =
                            mcu_pixel_height.min(height.saturating_sub(mcu_row * mcu_pixel_height));

                        for row in 0..pixel_rows_this {
                            let y_off = row * y_strip_width;
                            let up_row = v_ratio + row;
                            let chroma_off = up_row * y_strip_width;
                            let rgb_off = (pixel_row_start + row) * rgb_row_bytes;
                            if rgb_off + y_cols_this_image * 3 > rgb_chunk.len() {
                                break;
                            }
                            crate::color::ycbcr_planes_i16_to_rgb_u8(
                                &y_strip[y_off..y_off + y_cols_this_image],
                                &cb_up[chroma_off..chroma_off + y_cols_this_image],
                                &cr_up[chroma_off..chroma_off + y_cols_this_image],
                                &mut rgb_chunk[rgb_off..rgb_off + y_cols_this_image * 3],
                            );
                        }
                    };

                // Process MCU rows with 1-row lag for chroma context
                for local_row in 0..seg_mcu_rows {
                    let mcu_row = first_mcu_row + local_row;
                    let row_mcu_start = mcu_row * mcu_cols;
                    let row_mcu_end = ((mcu_row + 1) * mcu_cols).min(mcu_end);

                    // Decode all MCUs in this row into b buffers
                    // Y blocks → y_strip_b, chroma blocks → ext_cb_b/ext_cr_b data region
                    for mcu_idx in row_mcu_start..row_mcu_end {
                        let mcu_col = mcu_idx % mcu_cols;

                        for (sc_idx, (comp_idx, dc_table, ac_table)) in
                            scan_comps.iter().enumerate()
                        {
                            let h_samp = comp_h_samps[*comp_idx];
                            let v_samp = comp_v_samps[*comp_idx];

                            for v in 0..v_samp {
                                for h in 0..h_samp {
                                    // Check if this sub-block is beyond actual image bounds
                                    let block_x = mcu_col * h_samp + h;
                                    let block_y = mcu_row * v_samp + v;
                                    let is_padding = block_x >= actual_blocks_h[*comp_idx]
                                        || block_y >= actual_blocks_v[*comp_idx];

                                    let padding_state = if is_padding && !strict {
                                        Some(decoder.save_state())
                                    } else {
                                        None
                                    };

                                    let count = match decoder.decode_block_into(
                                        &mut coeffs_buf,
                                        prev_coeff_count,
                                        *comp_idx,
                                        *dc_table as usize,
                                        *ac_table as usize,
                                    ) {
                                        Ok(ScanRead::Value(c)) => c,
                                        Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                            if let Some(state) = padding_state {
                                                decoder.restore_state(state);
                                                coeffs_buf = [0i16; 64];
                                                had_padding_error = true;
                                                prev_coeff_count = 64;
                                                continue;
                                            }
                                            if truncation_mcu.is_none() {
                                                truncation_mcu = Some(mcu_idx as u32);
                                            }
                                            coeffs_buf = [0i16; 64];
                                            1
                                        }
                                        Err(e) => {
                                            if let Some(state) = padding_state {
                                                decoder.restore_state(state);
                                                coeffs_buf = [0i16; 64];
                                                had_padding_error = true;
                                                prev_coeff_count = 64;
                                                continue;
                                            }
                                            return Err(e);
                                        }
                                    };
                                    prev_coeff_count = count;

                                    if sc_idx == 0 {
                                        // Y block → y_strip_b
                                        let block_px = mcu_col * h_samp * 8 + h * 8;
                                        let block_py = v * 8;
                                        let strip_off = block_py * y_strip_width + block_px;

                                        if count == 1 {
                                            let dc = coeffs_buf[0] as i32
                                                * quant_tables[*comp_idx][0] as i32;
                                            idct_int_dc_only(
                                                dc,
                                                &mut y_strip_b[strip_off..],
                                                y_strip_width,
                                            );
                                        } else {
                                            dequantize_unzigzag_i32_into_partial(
                                                &coeffs_buf,
                                                quant_tables[*comp_idx],
                                                &mut dequant_buf,
                                                count,
                                            );
                                            idct_fn(
                                                &mut dequant_buf,
                                                &mut y_strip_b[strip_off..],
                                                y_strip_width,
                                                count,
                                            );
                                        }
                                    } else {
                                        // Chroma block → ext_cb_b or ext_cr_b data region
                                        // Data region starts at row 1 (row 0 is above context)
                                        let ext = if sc_idx == 1 {
                                            &mut ext_cb_b
                                        } else {
                                            &mut ext_cr_b
                                        };
                                        let block_px = mcu_col * h_samp * 8 + h * 8;
                                        let block_py = v * 8;
                                        let data_offset = c_strip_width; // skip row 0 (context)
                                        let strip_off =
                                            data_offset + block_py * c_strip_width + block_px;

                                        if count == 1 {
                                            let dc = coeffs_buf[0] as i32
                                                * quant_tables[*comp_idx][0] as i32;
                                            idct_int_dc_only(
                                                dc,
                                                &mut ext[strip_off..],
                                                c_strip_width,
                                            );
                                        } else {
                                            dequantize_unzigzag_i32_into_partial(
                                                &coeffs_buf,
                                                quant_tables[*comp_idx],
                                                &mut dequant_buf,
                                                count,
                                            );
                                            idct_fn(
                                                &mut dequant_buf,
                                                &mut ext[strip_off..],
                                                c_strip_width,
                                                count,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // MCU row decoded into b buffers. Handle context and output.

                    if local_row == 0 {
                        // First row in segment: set above context = edge replicate
                        ext_cb_b.copy_within(c_strip_width..2 * c_strip_width, 0);
                        ext_cr_b.copy_within(c_strip_width..2 * c_strip_width, 0);
                    }

                    if local_row > 0 {
                        // Output previous row (in a buffers) — now we have below context
                        let below_ctx_start = (c_strip_height + 1) * c_strip_width;
                        // Set a's below context = b's first data row
                        let src = c_strip_width;
                        ext_cb_a[below_ctx_start..below_ctx_start + c_strip_width]
                            .copy_from_slice(&ext_cb_b[src..src + c_strip_width]);
                        ext_cr_a[below_ctx_start..below_ctx_start + c_strip_width]
                            .copy_from_slice(&ext_cr_b[src..src + c_strip_width]);

                        upsample_and_output(
                            first_mcu_row + local_row - 1,
                            &y_strip_a,
                            &ext_cb_a,
                            &ext_cr_a,
                            &mut cb_up,
                            &mut cr_up,
                            &mut upsample_scratch,
                            rgb_chunk,
                        );

                        // Set b's above context = a's last data row
                        let last_data = c_strip_height * c_strip_width;
                        ext_cb_b[..c_strip_width]
                            .copy_from_slice(&ext_cb_a[last_data..last_data + c_strip_width]);
                        ext_cr_b[..c_strip_width]
                            .copy_from_slice(&ext_cr_a[last_data..last_data + c_strip_width]);
                    }

                    // Save boundary data
                    if local_row == 0 {
                        let src = c_strip_width; // first data row in ext
                        boundary
                            .first_cb_row
                            .copy_from_slice(&ext_cb_b[src..src + c_strip_width]);
                        boundary
                            .first_cr_row
                            .copy_from_slice(&ext_cr_b[src..src + c_strip_width]);
                        boundary
                            .first_y_row
                            .copy_from_slice(&y_strip_b[..y_strip_width]);
                    }
                    if local_row == seg_mcu_rows - 1 {
                        let last_data = c_strip_height * c_strip_width;
                        boundary
                            .last_cb_row
                            .copy_from_slice(&ext_cb_b[last_data..last_data + c_strip_width]);
                        boundary
                            .last_cr_row
                            .copy_from_slice(&ext_cr_b[last_data..last_data + c_strip_width]);
                        let last_y_off = (y_strip_height - 1) * y_strip_width;
                        boundary
                            .last_y_row
                            .copy_from_slice(&y_strip_b[last_y_off..last_y_off + y_strip_width]);
                    }

                    // Swap buffers: b becomes a (pending output), a becomes b (free)
                    core::mem::swap(&mut y_strip_a, &mut y_strip_b);
                    core::mem::swap(&mut ext_cb_a, &mut ext_cb_b);
                    core::mem::swap(&mut ext_cr_a, &mut ext_cr_b);
                }

                // Output last MCU row (now in a buffers after final swap)
                let below_ctx_start = (c_strip_height + 1) * c_strip_width;
                let last_data = c_strip_height * c_strip_width;
                // Set below context = edge replicate (segment boundary)
                ext_cb_a.copy_within(last_data..last_data + c_strip_width, below_ctx_start);
                ext_cr_a.copy_within(last_data..last_data + c_strip_width, below_ctx_start);

                upsample_and_output(
                    last_mcu_row,
                    &y_strip_a,
                    &ext_cb_a,
                    &ext_cr_a,
                    &mut cb_up,
                    &mut cr_up,
                    &mut upsample_scratch,
                    rgb_chunk,
                );

                Ok((
                    SegmentWarnings {
                        had_ac_overflow: decoder.had_ac_overflow,
                        had_invalid_huffman: decoder.had_invalid_huffman,
                        truncation_mcu,
                        had_padding_error,
                    },
                    boundary,
                ))
            })
            .collect();

        // Separate warnings from boundary data
        let mut seg_boundaries: Vec<SegmentBoundary> = Vec::with_capacity(num_segments);
        let mut seg_warnings: Vec<Result<SegmentWarnings>> = Vec::with_capacity(num_segments);
        for result in results {
            match result {
                Ok((warnings, boundary)) => {
                    seg_warnings.push(Ok(warnings));
                    seg_boundaries.push(boundary);
                }
                Err(e) => {
                    seg_warnings.push(Err(e));
                    // Push dummy boundary so indices stay aligned
                    seg_boundaries.push(SegmentBoundary {
                        first_cb_row: Vec::new(),
                        first_cr_row: Vec::new(),
                        last_cb_row: Vec::new(),
                        last_cr_row: Vec::new(),
                        first_y_row: Vec::new(),
                        last_y_row: Vec::new(),
                    });
                }
            }
        }

        let (any_ac, any_huff, first_trunc, any_pad) =
            Self::aggregate_fused_warnings(seg_warnings)?;

        // Boundary fixup pass: correct the 2 pixel rows per segment junction
        // where edge replication was used instead of real adjacent chroma.
        let boundary_count = seg_boundaries.len();
        if do_fixup && boundary_count > 1 {
            let mcu_rows_per_seg_nominal = mcu_rows_per_ri * group_stride;

            let mut cb_up_row = vec![0i16; y_strip_width];
            let mut cr_up_row = vec![0i16; y_strip_width];

            for junc in 0..boundary_count - 1 {
                let seg_n = &seg_boundaries[junc];
                let seg_n1 = &seg_boundaries[junc + 1];

                if seg_n.last_cb_row.is_empty() || seg_n1.first_cb_row.is_empty() {
                    continue;
                }

                // Compute MCU row boundaries for these segments
                let seg_n_last = ((junc + 1) * mcu_rows_per_seg_nominal - 1).min(mcu_rows - 1);
                let seg_n1_first = (junc + 1) * mcu_rows_per_seg_nominal;

                // Fix bottom pixel row of segment N's last MCU row
                // This is the last output pixel row: mcu_pixel_height - 1 within the MCU
                let fix_row_bottom = (seg_n_last + 1) * mcu_pixel_height - 1;
                if fix_row_bottom < height {
                    // Re-upsample with correct below context
                    upsample_row_h2_fancy_bilinear(
                        &seg_n.last_cb_row,
                        &seg_n1.first_cb_row,
                        c_strip_width,
                        &mut cb_up_row,
                        false,
                    );
                    upsample_row_h2_fancy_bilinear(
                        &seg_n.last_cr_row,
                        &seg_n1.first_cr_row,
                        c_strip_width,
                        &mut cr_up_row,
                        false,
                    );
                    let rgb_off = fix_row_bottom * rgb_row_bytes;
                    crate::color::ycbcr_planes_i16_to_rgb_u8(
                        &seg_n.last_y_row[..y_cols_this_image],
                        &cb_up_row[..y_cols_this_image],
                        &cr_up_row[..y_cols_this_image],
                        &mut rgb[rgb_off..rgb_off + y_cols_this_image * 3],
                    );
                }

                // Fix top pixel row of segment N+1's first MCU row
                let fix_row_top = seg_n1_first * mcu_pixel_height;
                if fix_row_top < height {
                    upsample_row_h2_fancy_bilinear(
                        &seg_n1.first_cb_row,
                        &seg_n.last_cb_row,
                        c_strip_width,
                        &mut cb_up_row,
                        true,
                    );
                    upsample_row_h2_fancy_bilinear(
                        &seg_n1.first_cr_row,
                        &seg_n.last_cr_row,
                        c_strip_width,
                        &mut cr_up_row,
                        true,
                    );
                    let rgb_off = fix_row_top * rgb_row_bytes;
                    crate::color::ycbcr_planes_i16_to_rgb_u8(
                        &seg_n1.first_y_row[..y_cols_this_image],
                        &cb_up_row[..y_cols_this_image],
                        &cr_up_row[..y_cols_this_image],
                        &mut rgb[rgb_off..rgb_off + y_cols_this_image * 3],
                    );
                }
            }
        }

        Ok((
            FusedResult(rgb),
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

// ── Wave-based parallel decode state ─────────────────────────────────────────

/// State for wave-based parallel scanline decode.
///
/// Captures everything needed to decode arbitrary restart segments on demand,
/// outliving the `JpegParser`. The scanline reader holds this and decodes
/// `wave_size` segments at a time into a reusable buffer, serving rows from
/// it, then recycling the buffer for the next wave.
///
/// **Prototype scope**: Box filter 4:2:0 path only.
pub(super) struct WaveParallelState {
    /// Byte offset where scan data begins in the JPEG data.
    pub scan_data_start: usize,
    /// Per-segment byte offsets (relative to scan_data_start).
    pub seg_starts: Vec<usize>,
    pub seg_ends: Vec<usize>,

    // Owned Huffman tables
    pub dc_tables: Vec<Option<HuffmanDecodeTable>>,
    pub ac_tables: Vec<Option<HuffmanDecodeTable>>,

    // Per-component quant tables (indexed by component, not table slot)
    pub quant_tables: Vec<[u16; DCT_BLOCK_SIZE]>,

    // Scan component mapping: (comp_idx, dc_table_idx, ac_table_idx)
    pub scan_comps: Vec<(usize, u8, u8)>,

    // Dimensions
    pub width: usize,
    pub height: usize,
    pub mcu_cols: usize,
    pub mcu_rows: usize,
    pub ri: usize,
    pub max_h_samp: usize,
    pub max_v_samp: usize,

    // Per-component sampling
    pub comp_h_samps: Vec<usize>,
    pub comp_v_samps: Vec<usize>,
    pub actual_blocks_h: Vec<usize>,
    pub actual_blocks_v: Vec<usize>,

    // Config
    pub lenient: bool,
    pub permissive_rst: bool,
    pub strict: bool,

    // Wave parameters
    pub wave_size: usize,
    pub num_segments: usize,
    pub mcu_rows_per_ri: usize,
    pub mcu_pixel_height: usize,
    pub pixel_rows_per_seg: usize,
}

impl WaveParallelState {
    /// Create wave state from parsed scan data and RST scan results.
    ///
    /// `scan_comps` is the (comp_idx, dc_table, ac_table) mapping from the SOS header.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scan_data: &super::parser::ParsedScanData<'_>,
        seg_starts: Vec<usize>,
        seg_ends: Vec<usize>,
        scan_comps: Vec<(usize, u8, u8)>,
        dc_tables: Vec<Option<HuffmanDecodeTable>>,
        ac_tables: Vec<Option<HuffmanDecodeTable>>,
        strictness: super::Strictness,
        wave_size: usize,
    ) -> Self {
        let width = scan_data.width as usize;
        let height = scan_data.height as usize;
        let num_comps = scan_data.num_components as usize;

        let max_h_samp = scan_data.h_samp[..num_comps]
            .iter()
            .copied()
            .max()
            .unwrap_or(1) as usize;
        let max_v_samp = scan_data.v_samp[..num_comps]
            .iter()
            .copied()
            .max()
            .unwrap_or(1) as usize;

        let mcu_width = max_h_samp * 8;
        let mcu_height = max_v_samp * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;
        let ri = scan_data.restart_interval as usize;
        let mcu_rows_per_ri = ri / mcu_cols;
        let pixel_rows_per_seg = mcu_rows_per_ri * mcu_height;

        // Build per-component quant tables (owned copies)
        let quant_tables: Vec<[u16; DCT_BLOCK_SIZE]> = (0..num_comps)
            .map(|ci| {
                let qt_idx = scan_data.quant_indices[ci];
                scan_data.quant_tables[qt_idx].unwrap()
            })
            .collect();

        // Per-component sampling factors
        let comp_h_samps: Vec<usize> = (0..num_comps)
            .map(|ci| scan_data.h_samp[ci] as usize)
            .collect();
        let comp_v_samps: Vec<usize> = (0..num_comps)
            .map(|ci| scan_data.v_samp[ci] as usize)
            .collect();

        // Actual block counts for padding detection
        let actual_blocks_h: Vec<usize> = (0..num_comps)
            .map(|ci| {
                let h = scan_data.h_samp[ci] as usize;
                let comp_w = (width * h + max_h_samp - 1) / max_h_samp;
                (comp_w + 7) / 8
            })
            .collect();
        let actual_blocks_v: Vec<usize> = (0..num_comps)
            .map(|ci| {
                let v = scan_data.v_samp[ci] as usize;
                let comp_h = (height * v + max_v_samp - 1) / max_v_samp;
                (comp_h + 7) / 8
            })
            .collect();

        let lenient = matches!(
            strictness,
            super::Strictness::Lenient | super::Strictness::Permissive
        );
        let permissive_rst = strictness == super::Strictness::Permissive;
        let strict = strictness == super::Strictness::Strict;

        let num_segments = seg_starts.len();

        WaveParallelState {
            scan_data_start: scan_data.scan_data_start,
            seg_starts,
            seg_ends,
            dc_tables,
            ac_tables,
            quant_tables,
            scan_comps,
            width,
            height,
            mcu_cols,
            mcu_rows,
            ri,
            max_h_samp,
            max_v_samp,
            comp_h_samps,
            comp_v_samps,
            actual_blocks_h,
            actual_blocks_v,
            lenient,
            permissive_rst,
            strict,
            wave_size,
            num_segments,
            mcu_rows_per_ri,
            mcu_pixel_height: mcu_height,
            pixel_rows_per_seg,
        }
    }

    /// Decode a wave of segments in parallel, writing RGB output to `wave_buf`.
    ///
    /// Decodes segments `seg_range_start..seg_range_end` using rayon, with each
    /// segment writing to its corresponding chunk in `wave_buf`.
    ///
    /// Returns the number of valid pixel rows written.
    pub fn decode_wave_box(
        &self,
        jpeg_data: &[u8],
        seg_range_start: usize,
        seg_range_end: usize,
        wave_buf: &mut [u8],
    ) -> Result<usize> {
        use crate::color::ycbcr::fused_h2v2_box_ycbcr_to_rgb_u8;

        let wave_count = seg_range_end - seg_range_start;
        if wave_count == 0 {
            return Ok(0);
        }

        let scan_data = &jpeg_data[self.scan_data_start..];
        let rgb_row_bytes = self.width * 3;
        let seg_rgb_bytes = self.pixel_rows_per_seg * rgb_row_bytes;
        let total_mcus = self.mcu_cols * self.mcu_rows;

        let y_h = self.comp_h_samps[0];
        let y_v = self.comp_v_samps[0];
        let y_strip_width = self.mcu_cols * y_h * 8;
        let y_strip_height = y_v * 8;
        let c_strip_width = self.mcu_cols * self.comp_h_samps[1] * 8;
        let c_strip_height = self.comp_v_samps[1] * 8;

        let idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8) = idct_int_tiered;

        // Split wave buffer into per-segment chunks
        let chunks: Vec<&mut [u8]> = wave_buf[..wave_count * seg_rgb_bytes]
            .chunks_mut(seg_rgb_bytes)
            .collect();

        let seg_warnings: Vec<Result<SegmentWarnings>> = chunks
            .into_par_iter()
            .enumerate()
            .map(|(i, rgb_chunk)| {
                let raw_idx = seg_range_start + i;
                let seg_start = self.seg_starts[raw_idx];
                let seg_end = self.seg_ends[raw_idx];
                let seg_data = &scan_data[seg_start..seg_end];

                let (mcu_start, mcu_end) =
                    JpegParser::segment_mcu_range(raw_idx, self.num_segments, self.ri, 1, total_mcus);

                let mut decoder = JpegParser::setup_segment_decoder(
                    seg_data,
                    &self.scan_comps,
                    &self.dc_tables,
                    &self.ac_tables,
                    self.lenient,
                    self.permissive_rst,
                );

                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];
                let mut dequant_buf = [0i32; DCT_BLOCK_SIZE];
                let mut prev_coeff_count: u8 = 64;
                let mut truncation_mcu: Option<u32> = None;
                let mut had_padding_error = false;

                let mut y_strip: Vec<i16> = vec![0i16; y_strip_width * y_strip_height];
                let mut cb_strip: Vec<i16> = vec![0i16; c_strip_width * c_strip_height];
                let mut cr_strip: Vec<i16> = vec![0i16; c_strip_width * c_strip_height];

                let first_mcu_row = mcu_start / self.mcu_cols;
                let mut current_mcu_row = first_mcu_row;
                let seg_first_pixel_row = first_mcu_row * self.mcu_pixel_height;

                let flush_mcu_row = |current_mcu_row: usize,
                                     y_strip: &[i16],
                                     cb_strip: &[i16],
                                     cr_strip: &[i16],
                                     rgb_chunk: &mut [u8]| {
                    let pixel_row_start = (current_mcu_row * self.mcu_pixel_height)
                        .saturating_sub(seg_first_pixel_row);
                    let pixel_rows_this = self
                        .mcu_pixel_height
                        .min(self.height.saturating_sub(current_mcu_row * self.mcu_pixel_height));
                    let cols_this = self.width.min(y_strip_width);

                    for py in 0..pixel_rows_this {
                        let y_off = py * y_strip_width;
                        let c_row =
                            py / (self.max_v_samp / self.comp_v_samps[1].max(1));
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
                    let mcu_row = mcu_idx / self.mcu_cols;
                    let mcu_col = mcu_idx % self.mcu_cols;

                    if mcu_row != current_mcu_row {
                        flush_mcu_row(
                            current_mcu_row,
                            &y_strip,
                            &cb_strip,
                            &cr_strip,
                            rgb_chunk,
                        );
                        current_mcu_row = mcu_row;
                    }

                    for (sc_idx, (comp_idx, dc_table, ac_table)) in
                        self.scan_comps.iter().enumerate()
                    {
                        let h_samp = self.comp_h_samps[*comp_idx];
                        let v_samp = self.comp_v_samps[*comp_idx];

                        let (strip, strip_stride) = match sc_idx {
                            0 => (&mut y_strip as &mut Vec<i16>, y_strip_width),
                            1 => (&mut cb_strip, c_strip_width),
                            _ => (&mut cr_strip, c_strip_width),
                        };

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let block_x = mcu_col * h_samp + h;
                                let block_y = mcu_row * v_samp + v;
                                let is_padding = block_x >= self.actual_blocks_h[*comp_idx]
                                    || block_y >= self.actual_blocks_v[*comp_idx];

                                let padding_state = if is_padding && !self.strict {
                                    Some(decoder.save_state())
                                } else {
                                    None
                                };

                                let count = match decoder.decode_block_into(
                                    &mut coeffs_buf,
                                    prev_coeff_count,
                                    *comp_idx,
                                    *dc_table as usize,
                                    *ac_table as usize,
                                ) {
                                    Ok(ScanRead::Value(c)) => c,
                                    Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                        if let Some(state) = padding_state {
                                            decoder.restore_state(state);
                                            coeffs_buf = [0i16; 64];
                                            had_padding_error = true;
                                            prev_coeff_count = 64;
                                            continue;
                                        }
                                        if truncation_mcu.is_none() {
                                            truncation_mcu = Some(mcu_idx as u32);
                                        }
                                        coeffs_buf = [0i16; 64];
                                        1
                                    }
                                    Err(e) => {
                                        if let Some(state) = padding_state {
                                            decoder.restore_state(state);
                                            coeffs_buf = [0i16; 64];
                                            had_padding_error = true;
                                            prev_coeff_count = 64;
                                            continue;
                                        }
                                        return Err(e);
                                    }
                                };
                                prev_coeff_count = count;

                                let block_px = mcu_col * h_samp * 8 + h * 8;
                                let block_py = v * 8;
                                let strip_off = block_py * strip_stride + block_px;

                                if count == 1 {
                                    let dc = coeffs_buf[0] as i32
                                        * self.quant_tables[*comp_idx][0] as i32;
                                    idct_int_dc_only(dc, &mut strip[strip_off..], strip_stride);
                                } else {
                                    dequantize_unzigzag_i32_into_partial(
                                        &coeffs_buf,
                                        &self.quant_tables[*comp_idx],
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
                flush_mcu_row(
                    current_mcu_row,
                    &y_strip,
                    &cb_strip,
                    &cr_strip,
                    rgb_chunk,
                );

                Ok(SegmentWarnings {
                    had_ac_overflow: decoder.had_ac_overflow,
                    had_invalid_huffman: decoder.had_invalid_huffman,
                    truncation_mcu,
                    had_padding_error,
                })
            })
            .collect();

        // Aggregate warnings (ignore for wave mode — just propagate errors)
        for result in seg_warnings {
            result?;
        }

        // Compute valid pixel rows
        let first_row = seg_range_start * self.pixel_rows_per_seg;
        let last_row = (seg_range_end * self.pixel_rows_per_seg).min(self.height);
        Ok(last_row.saturating_sub(first_row))
    }
}

/// Build Huffman table vectors from `ParsedScanData` for thread-safe parallel access.
///
/// Fills in default tables for any missing entries that are referenced by `scan_comps`.
pub(super) fn build_huffman_tables_from_scan_data(
    scan_data: &super::parser::ParsedScanData<'_>,
    scan_comps: &[(usize, u8, u8)],
) -> (
    Vec<Option<HuffmanDecodeTable>>,
    Vec<Option<HuffmanDecodeTable>>,
) {
    let dc_tables: Vec<Option<HuffmanDecodeTable>> = (0..MAX_HUFFMAN_TABLES)
        .map(|idx| {
            scan_data.dc_tables[idx].clone().or_else(|| {
                let needed = scan_comps
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
            scan_data.ac_tables[idx].clone().or_else(|| {
                let needed = scan_comps
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
