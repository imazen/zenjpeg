//! Parallel baseline JPEG decoding using restart marker boundaries.
//!
//! When a JPEG has restart markers (DRI), the entropy-coded data is split
//! into independent segments at each RST boundary. DC predictors reset
//! at each restart, so segments can be decoded in parallel.
//!
//! ## Strategy
//!
//! 1. **SIMD scan**: Find all RST marker byte offsets (see `rst_scan`)
//! 2. **Segment split**: Compute byte ranges for each restart segment
//! 3. **Parallel decode**: Use rayon to entropy-decode segments in parallel
//! 4. **Write-back**: Either direct-write (contiguous case) or scatter
//! 5. **Sequential post-process**: IDCT + upsampling + color convert (unchanged)
//!
//! ## Direct-Write vs Scatter
//!
//! When all scan components have h_samp=v_samp=1 (4:4:4, grayscale), each
//! component's block index equals the linear MCU index. Segments map to
//! contiguous array ranges, enabling zero-copy parallel writes via
//! `chunks_mut`. This eliminates ~30MB of intermediate buffers and scatter
//! copy for a 2048x2048 4:4:4 image.
//!
//! For subsampled images (4:2:0), luma blocks span multiple block rows per
//! MCU row and aren't contiguous. These use the scatter path with per-thread
//! `Vec<DecodedBlock>` buffers.
//!
//! ## Threshold
//!
//! Parallel decode is only used when:
//! - The `parallel` feature is enabled
//! - The image has restart markers (DRI != 0)
//! - There are at least `MIN_SEGMENTS` restart segments
//! - The image has at least `MIN_BLOCKS` total blocks

use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;

use super::rst_scan::{compute_segments, RstScanResult};
use super::{DecodeWarning, Strictness};

// JpegParser is accessible since we're in the same crate
use super::parser::JpegParser;

use rayon::prelude::*;

/// Minimum restart segments to justify parallel overhead.
const MIN_SEGMENTS: usize = 4;

/// Minimum total MCUs to justify parallel decode.
const MIN_BLOCKS: usize = 1024;

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

/// Component decode info needed by each thread.
#[derive(Clone)]
struct CompInfo {
    h_samp: usize,
    v_samp: usize,
}

/// Warnings collected from decoding a single segment.
struct SegmentWarnings {
    had_ac_overflow: bool,
    had_invalid_huffman: bool,
    truncation_mcu: Option<u32>,
    had_padding_error: bool,
}

/// A decoded block with its target position (scatter path only).
struct DecodedBlock {
    /// Component index (0=Y, 1=Cb, 2=Cr)
    comp_idx: usize,
    /// Block index in the component's coefficient array
    block_idx: usize,
    /// The decoded DCT coefficients
    coeffs: [i16; DCT_BLOCK_SIZE],
    /// Coefficient count for tiered IDCT
    coeff_count: u8,
}

impl<'a> JpegParser<'a> {
    /// Parallel decode of a baseline sequential scan.
    ///
    /// Scans for RST markers using SIMD, splits into segments, then
    /// decodes each segment in parallel using rayon.
    pub(super) fn decode_scan_parallel(
        &mut self,
        scan_components: &[(usize, u8, u8)],
        rst_result: RstScanResult,
    ) -> Result<()> {
        let max_h_samp = (0..self.num_components as usize)
            .map(|i| self.components[i].h_samp_factor)
            .max()
            .unwrap_or(1) as usize;
        let max_v_samp = (0..self.num_components as usize)
            .map(|i| self.components[i].v_samp_factor)
            .max()
            .unwrap_or(1) as usize;

        let mcu_width = max_h_samp * 8;
        let mcu_height = max_v_samp * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;
        let restart_interval = self.restart_interval as u32;
        let total_mcus = (mcu_cols * mcu_rows) as u32;

        // Use pre-scanned RST markers (scanned once in try_decode_scan_parallel)
        let rst_markers = &rst_result.markers;
        let entropy_end = rst_result.entropy_end;

        // Compute segment byte ranges (use entropy_end, not full scan_data length)
        let scan_data = &self.data[self.position..];
        let (seg_starts, seg_ends) = compute_segments(rst_markers, entropy_end);
        let found_segments = seg_starts.len();

        // Validate: need at least as many segments as expected from MCU count.
        // Cap to expected count — extra RST markers (from extraneous data or encoder
        // quirks) produce empty segments that would cause u32 underflow in MCU ranges
        // and unwrap panics in the direct-write chunk iterators.
        let expected_segments = ((total_mcus + restart_interval - 1) / restart_interval) as usize;
        if found_segments < expected_segments {
            return Err(Error::internal(
                "parallel decode: fewer RST segments than expected",
            ));
        }
        let num_segments = found_segments.min(expected_segments);

        // Collect Huffman table references (cloned for thread safety)
        let dc_tables: Vec<Option<HuffmanDecodeTable>> = (0..MAX_HUFFMAN_TABLES)
            .map(|idx| {
                self.dc_tables[idx].clone().or_else(|| {
                    // Check if any scan component needs this table
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

        // Compute MCU ranges for each segment
        let mut seg_mcu_ranges: Vec<(u32, u32)> = Vec::with_capacity(num_segments);
        for seg_idx in 0..num_segments {
            let mcu_start = seg_idx as u32 * restart_interval;
            let mcu_end = ((seg_idx as u32 + 1) * restart_interval).min(total_mcus);
            seg_mcu_ranges.push((mcu_start, mcu_end));
        }

        // Collect component info
        let comp_infos: Vec<CompInfo> = (0..self.num_components as usize)
            .map(|i| CompInfo {
                h_samp: self.components[i].h_samp_factor as usize,
                v_samp: self.components[i].v_samp_factor as usize,
            })
            .collect();

        let scan_comps: Vec<(usize, u8, u8)> = scan_components.to_vec();
        let strictness = self.strictness;
        let lenient = strictness == Strictness::Lenient;
        let width = self.width as usize;
        let height = self.height as usize;

        // Check if all scan components are contiguous (block_idx = MCU linear index).
        // True for 4:4:4, grayscale — enables zero-copy direct writes.
        let contiguous = scan_comps
            .iter()
            .all(|(ci, _, _)| comp_infos[*ci].h_samp == 1 && comp_infos[*ci].v_samp == 1);

        let (any_ac_overflow, any_invalid_huffman, first_truncation, any_padding_error) =
            if contiguous {
                self.decode_segments_direct(
                    &scan_comps,
                    &comp_infos,
                    &dc_tables,
                    &ac_tables,
                    scan_data,
                    &seg_starts,
                    &seg_ends,
                    &seg_mcu_ranges,
                    num_segments,
                    restart_interval as usize,
                    mcu_cols,
                    lenient,
                )?
            } else {
                self.decode_segments_scatter(
                    &scan_comps,
                    &comp_infos,
                    &dc_tables,
                    &ac_tables,
                    scan_data,
                    &seg_starts,
                    &seg_ends,
                    &seg_mcu_ranges,
                    num_segments,
                    mcu_cols,
                    max_h_samp,
                    max_v_samp,
                    width,
                    height,
                    lenient,
                )?
            };

        // Advance position past the entropy-coded data to the terminating marker
        self.position += entropy_end;

        // Emit warnings
        if let Some(at_mcu) = first_truncation {
            self.warn(DecodeWarning::TruncatedScan {
                blocks_decoded: at_mcu,
                blocks_expected: total_mcus,
            })?;
        }
        if any_padding_error {
            self.warn(DecodeWarning::PaddingBlockError)?;
        }
        if any_ac_overflow {
            self.warn(DecodeWarning::AcIndexOverflow)?;
        }
        if any_invalid_huffman {
            self.warn(DecodeWarning::InvalidHuffmanCode)?;
        }

        Ok(())
    }

    /// Direct-write parallel decode for contiguous layouts (4:4:4, grayscale).
    ///
    /// Each component's block index equals the MCU linear index, so segments
    /// map to contiguous slices. Threads write directly into coefficient storage
    /// via `chunks_mut`, eliminating intermediate buffers and scatter copy.
    #[allow(clippy::too_many_arguments)]
    fn decode_segments_direct(
        &mut self,
        scan_comps: &[(usize, u8, u8)],
        _comp_infos: &[CompInfo],
        dc_tables: &[Option<HuffmanDecodeTable>],
        ac_tables: &[Option<HuffmanDecodeTable>],
        scan_data: &[u8],
        seg_starts: &[usize],
        seg_ends: &[usize],
        seg_mcu_ranges: &[(u32, u32)],
        num_segments: usize,
        restart_interval: usize,
        _mcu_cols: usize,
        lenient: bool,
    ) -> Result<(bool, bool, Option<u32>, bool)> {
        let num_scan_comps = scan_comps.len();

        // Take coefficient arrays out of self for splitting
        let mut coeffs = core::mem::take(&mut self.coeffs);
        let mut counts = core::mem::take(&mut self.coeff_counts);

        // Split each component's array into per-segment chunks via chunks_mut.
        // iter_mut gives disjoint mutable borrows across components.
        let mut all_coeff_chunks: Vec<Vec<&mut [[i16; DCT_BLOCK_SIZE]]>> = coeffs
            .iter_mut()
            .map(|arr| arr.chunks_mut(restart_interval).collect())
            .collect();

        let mut all_count_chunks: Vec<Vec<&mut [u8]>> = counts
            .iter_mut()
            .map(|arr| arr.chunks_mut(restart_interval).collect())
            .collect();

        // Transpose from [component][segment] to [segment][scan_component]
        // by draining each scan component's chunks into per-segment groups.
        // mem::take avoids multiple mutable borrows of the outer Vec.
        let mut coeff_iters: Vec<alloc::vec::IntoIter<&mut [[i16; DCT_BLOCK_SIZE]]>> =
            Vec::with_capacity(num_scan_comps);
        let mut count_iters: Vec<alloc::vec::IntoIter<&mut [u8]>> =
            Vec::with_capacity(num_scan_comps);

        for &(ci, _, _) in scan_comps {
            coeff_iters.push(core::mem::take(&mut all_coeff_chunks[ci]).into_iter());
            count_iters.push(core::mem::take(&mut all_count_chunks[ci]).into_iter());
        }

        // Build per-segment write targets: Vec<(coeffs_per_comp, counts_per_comp)>
        let mut seg_targets: Vec<(Vec<&mut [[i16; DCT_BLOCK_SIZE]]>, Vec<&mut [u8]>)> =
            Vec::with_capacity(num_segments);

        for _seg_idx in 0..num_segments {
            let mut seg_coeffs = Vec::with_capacity(num_scan_comps);
            let mut seg_counts = Vec::with_capacity(num_scan_comps);
            for sc_idx in 0..num_scan_comps {
                seg_coeffs.push(coeff_iters[sc_idx].next().unwrap());
                seg_counts.push(count_iters[sc_idx].next().unwrap());
            }
            seg_targets.push((seg_coeffs, seg_counts));
        }

        // Parallel decode: each thread writes directly into its segment's slices
        let seg_warnings: Vec<Result<SegmentWarnings>> = seg_targets
            .into_par_iter()
            .enumerate()
            .map(|(seg_idx, (mut coeff_slices, mut count_slices))| {
                let seg_start = seg_starts[seg_idx];
                let seg_end = seg_ends[seg_idx];
                let (mcu_start, mcu_end) = seg_mcu_ranges[seg_idx];
                let seg_data = &scan_data[seg_start..seg_end];

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

                let mut prev_coeff_counts: [u8; 4] = [64; 4];
                let mut truncation_mcu: Option<u32> = None;

                // Contiguous case: no sub-block loops, no padding, direct writes
                for mcu_global in mcu_start..mcu_end {
                    let mcu_local = (mcu_global - mcu_start) as usize;

                    for (sc_idx, (comp_idx, dc_table, ac_table)) in scan_comps.iter().enumerate() {
                        let count = match decoder.decode_block_into(
                            &mut coeff_slices[sc_idx][mcu_local],
                            prev_coeff_counts[*comp_idx],
                            *comp_idx,
                            *dc_table as usize,
                            *ac_table as usize,
                        ) {
                            Ok(ScanRead::Value(c)) => c,
                            Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                if truncation_mcu.is_none() {
                                    truncation_mcu = Some(mcu_global);
                                }
                                coeff_slices[sc_idx][mcu_local] = [0i16; 64];
                                count_slices[sc_idx][mcu_local] = 1;
                                prev_coeff_counts[*comp_idx] = 64;
                                continue;
                            }
                            Err(e) => return Err(e),
                        };
                        count_slices[sc_idx][mcu_local] = count;
                        prev_coeff_counts[*comp_idx] = count;
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

        // Put coefficient arrays back into self
        self.coeffs = coeffs;
        self.coeff_counts = counts;

        // Aggregate warnings
        aggregate_warnings(seg_warnings)
    }

    /// Scatter-based parallel decode for non-contiguous layouts (4:2:0, etc).
    ///
    /// Each thread decodes into a local `Vec<DecodedBlock>`, then the main
    /// thread scatters blocks into coefficient storage at computed positions.
    #[allow(clippy::too_many_arguments)]
    fn decode_segments_scatter(
        &mut self,
        scan_comps: &[(usize, u8, u8)],
        comp_infos: &[CompInfo],
        dc_tables: &[Option<HuffmanDecodeTable>],
        ac_tables: &[Option<HuffmanDecodeTable>],
        scan_data: &[u8],
        seg_starts: &[usize],
        seg_ends: &[usize],
        seg_mcu_ranges: &[(u32, u32)],
        num_segments: usize,
        mcu_cols: usize,
        max_h_samp: usize,
        max_v_samp: usize,
        width: usize,
        height: usize,
        lenient: bool,
    ) -> Result<(bool, bool, Option<u32>, bool)> {
        // Each segment produces a Vec<DecodedBlock> with computed block indices
        let seg_results: Vec<Result<(Vec<DecodedBlock>, SegmentWarnings)>> = (0..num_segments)
            .into_par_iter()
            .map(|seg_idx| {
                let seg_start = seg_starts[seg_idx];
                let seg_end = seg_ends[seg_idx];
                let (mcu_start, mcu_end) = seg_mcu_ranges[seg_idx];
                let seg_data = &scan_data[seg_start..seg_end];

                let num_mcus = (mcu_end - mcu_start) as usize;
                let blocks_per_mcu: usize = scan_comps
                    .iter()
                    .map(|(ci, _, _)| comp_infos[*ci].h_samp * comp_infos[*ci].v_samp)
                    .sum();

                let mut blocks = Vec::with_capacity(num_mcus * blocks_per_mcu);

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

                let mut prev_coeff_counts: [u8; 4] = [64; 4];
                let mut had_padding_error = false;
                let mut truncation_mcu: Option<u32> = None;
                let mut coeffs_buf = [0i16; DCT_BLOCK_SIZE];

                for mcu_global in mcu_start..mcu_end {
                    let mcu_y = mcu_global as usize / mcu_cols;
                    let mcu_x = mcu_global as usize % mcu_cols;

                    for (comp_idx, dc_table, ac_table) in scan_comps {
                        let ci = &comp_infos[*comp_idx];
                        let comp_blocks_h = mcu_cols * ci.h_samp;

                        let comp_width = (width * ci.h_samp + max_h_samp - 1) / max_h_samp;
                        let comp_height = (height * ci.v_samp + max_v_samp - 1) / max_v_samp;
                        let actual_blocks_h = (comp_width + 7) / 8;
                        let actual_blocks_v = (comp_height + 7) / 8;

                        let is_single_component_oversample =
                            scan_comps.len() == 1 && (ci.h_samp > 1 || ci.v_samp > 1);

                        for v in 0..ci.v_samp {
                            for h in 0..ci.h_samp {
                                let block_x = mcu_x * ci.h_samp + h;
                                let block_y = mcu_y * ci.v_samp + v;
                                let block_idx = block_y * comp_blocks_h + block_x;

                                let is_padding =
                                    block_x >= actual_blocks_h || block_y >= actual_blocks_v;

                                if is_padding && is_single_component_oversample {
                                    blocks.push(DecodedBlock {
                                        comp_idx: *comp_idx,
                                        block_idx,
                                        coeffs: [0i16; 64],
                                        coeff_count: 1,
                                    });
                                    continue;
                                }

                                if is_padding {
                                    let saved_state = decoder.save_state();
                                    match decoder.decode_block_into(
                                        &mut coeffs_buf,
                                        prev_coeff_counts[*comp_idx],
                                        *comp_idx,
                                        *dc_table as usize,
                                        *ac_table as usize,
                                    ) {
                                        Ok(ScanRead::Value(count)) => {
                                            blocks.push(DecodedBlock {
                                                comp_idx: *comp_idx,
                                                block_idx,
                                                coeffs: coeffs_buf,
                                                coeff_count: count,
                                            });
                                            prev_coeff_counts[*comp_idx] = count;
                                        }
                                        Ok(ScanRead::EndOfScan | ScanRead::Truncated) | Err(_) => {
                                            decoder.restore_state(saved_state);
                                            blocks.push(DecodedBlock {
                                                comp_idx: *comp_idx,
                                                block_idx,
                                                coeffs: [0i16; 64],
                                                coeff_count: 1,
                                            });
                                            prev_coeff_counts[*comp_idx] = 64;
                                            had_padding_error = true;
                                        }
                                    }
                                } else {
                                    let count = match decoder.decode_block_into(
                                        &mut coeffs_buf,
                                        prev_coeff_counts[*comp_idx],
                                        *comp_idx,
                                        *dc_table as usize,
                                        *ac_table as usize,
                                    ) {
                                        Ok(ScanRead::Value(c)) => c,
                                        Ok(ScanRead::EndOfScan | ScanRead::Truncated) => {
                                            if truncation_mcu.is_none() {
                                                truncation_mcu = Some(mcu_global);
                                            }
                                            blocks.push(DecodedBlock {
                                                comp_idx: *comp_idx,
                                                block_idx,
                                                coeffs: [0i16; 64],
                                                coeff_count: 1,
                                            });
                                            prev_coeff_counts[*comp_idx] = 64;
                                            continue;
                                        }
                                        Err(e) => return Err(e),
                                    };
                                    blocks.push(DecodedBlock {
                                        comp_idx: *comp_idx,
                                        block_idx,
                                        coeffs: coeffs_buf,
                                        coeff_count: count,
                                    });
                                    prev_coeff_counts[*comp_idx] = count;
                                }
                            }
                        }
                    }
                }

                Ok((
                    blocks,
                    SegmentWarnings {
                        had_ac_overflow: decoder.had_ac_overflow,
                        had_invalid_huffman: decoder.had_invalid_huffman,
                        truncation_mcu,
                        had_padding_error,
                    },
                ))
            })
            .collect();

        // Scatter results into coefficient storage
        let mut any_ac_overflow = false;
        let mut any_invalid_huffman = false;
        let mut first_truncation: Option<u32> = None;
        let mut any_padding_error = false;

        for result in seg_results {
            let (blocks, warnings) = result?;
            any_ac_overflow |= warnings.had_ac_overflow;
            any_invalid_huffman |= warnings.had_invalid_huffman;
            any_padding_error |= warnings.had_padding_error;
            if let Some(mcu) = warnings.truncation_mcu {
                first_truncation = Some(match first_truncation {
                    Some(existing) => existing.min(mcu),
                    None => mcu,
                });
            }

            for block in blocks {
                self.coeffs[block.comp_idx][block.block_idx] = block.coeffs;
                self.coeff_counts[block.comp_idx][block.block_idx] = block.coeff_count;
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

/// Aggregate warnings from all segment results into a single tuple.
fn aggregate_warnings(
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
