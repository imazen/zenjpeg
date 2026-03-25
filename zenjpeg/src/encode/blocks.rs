//! Block operations for JPEG encoding.
//!
//! This module contains:
//! - Block quantization functions (YCbCr and XYB)
//! - Block extraction from planes
//! - Huffman table optimization
//! - Scan encoding

use super::config::ComputedConfig;
use crate::entropy::{self, EntropyEncoder};
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::huffman::HuffmanEncodeTable;
use crate::huffman::optimize::{FrequencyCounter, HuffmanTableSet};
use crate::types::Subsampling;
#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;

use wide::{CmpEq, i16x8};

/// Frequency counts from an optimized Huffman encoding pass.
///
/// Contains the raw symbol frequencies for each of the 4 Huffman tables
/// (DC luma, AC luma, DC chroma, AC chroma). These can be aggregated
/// across multiple images to build optimized tables.
#[derive(Clone, Debug)]
pub struct HuffmanSymbolFrequencies {
    /// DC luminance symbol frequencies
    pub dc_luma: FrequencyCounter,
    /// AC luminance symbol frequencies
    pub ac_luma: FrequencyCounter,
    /// DC chrominance symbol frequencies
    pub dc_chroma: FrequencyCounter,
    /// AC chrominance symbol frequencies
    pub ac_chroma: FrequencyCounter,
}

impl HuffmanSymbolFrequencies {
    /// Adds another set of frequency counts into this one.
    pub fn add(&mut self, other: &HuffmanSymbolFrequencies) {
        self.dc_luma.add(&other.dc_luma);
        self.ac_luma.add(&other.ac_luma);
        self.dc_chroma.add(&other.dc_chroma);
        self.ac_chroma.add(&other.ac_chroma);
    }

    /// Generates a `HuffmanTableSet` from the aggregated frequencies.
    ///
    /// Uses the default `JpegliCreateTree` algorithm.
    pub fn generate_tables(&self) -> Result<HuffmanTableSet> {
        self.generate_tables_with_method(crate::types::HuffmanMethod::JpegliCreateTree)
    }

    /// Generates a `HuffmanTableSet` using the specified algorithm.
    pub fn generate_tables_with_method(
        &self,
        method: crate::types::HuffmanMethod,
    ) -> Result<HuffmanTableSet> {
        let dc_luma = self.dc_luma.generate_table_with_method(method)?;
        let ac_luma = self.ac_luma.generate_table_with_method(method)?;

        let (dc_chroma, ac_chroma) = if self.dc_chroma.is_empty_histogram() {
            use crate::huffman::optimize::OptimizedTable;
            use crate::huffman::{
                STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_DC_CHROMINANCE_BITS,
                STD_DC_CHROMINANCE_VALUES,
            };
            (
                OptimizedTable {
                    table: HuffmanEncodeTable::std_dc_chrominance().clone(),
                    bits: STD_DC_CHROMINANCE_BITS,
                    values: STD_DC_CHROMINANCE_VALUES.to_vec(),
                },
                OptimizedTable {
                    table: HuffmanEncodeTable::std_ac_chrominance().clone(),
                    bits: STD_AC_CHROMINANCE_BITS,
                    values: STD_AC_CHROMINANCE_VALUES.to_vec(),
                },
            )
        } else {
            (
                self.dc_chroma.generate_table_with_method(method)?,
                self.ac_chroma.generate_table_with_method(method)?,
            )
        };

        Ok(HuffmanTableSet {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
    }
}

impl ComputedConfig {
    /// Counts symbol frequencies from quantized blocks (for Huffman optimization).
    pub(crate) fn count_block_frequencies(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> HuffmanSymbolFrequencies {
        let mut dc_luma_freq = FrequencyCounter::new();
        let mut dc_chroma_freq = FrequencyCounter::new();
        let mut ac_luma_freq = FrequencyCounter::new();
        let mut ac_chroma_freq = FrequencyCounter::new();

        let width = self.width as usize;
        let height = self.height as usize;
        let (h_samp, v_samp) = match self.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple iteration, no padding needed
            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            // Restart interval tracking (must match encoder behavior exactly)
            let restart_interval = self.restart_interval as usize;
            let total_mcus = y_blocks.len();

            for (i, y_block) in y_blocks.iter().enumerate() {
                Self::collect_block_frequencies(
                    y_block,
                    prev_y_dc,
                    &mut dc_luma_freq,
                    &mut ac_luma_freq,
                );
                prev_y_dc = y_block[0];

                if is_color {
                    Self::collect_block_frequencies(
                        &cb_blocks[i],
                        prev_cb_dc,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    prev_cb_dc = cb_blocks[i][0];

                    Self::collect_block_frequencies(
                        &cr_blocks[i],
                        prev_cr_dc,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    prev_cr_dc = cr_blocks[i][0];
                }

                // Reset DC prediction at restart boundaries (same logic as encoder)
                // This ensures Huffman tables account for DC differences after resets
                if restart_interval > 0 && i + 1 < total_mcus && (i + 1) % restart_interval == 0 {
                    prev_y_dc = 0;
                    prev_cb_dc = 0;
                    prev_cr_dc = 0;
                }
            }
        } else {
            // Subsampled mode - iterate in MCU order with padding
            let y_blocks_w = (width + 7) / 8;
            let y_blocks_h = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_w = (c_width + 7) / 8;
            let c_blocks_h = (c_height + 7) / 8;
            let mcu_h = (y_blocks_w + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_h + v_samp - 1) / v_samp;

            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            // Restart interval tracking (must match encoder behavior exactly)
            let restart_interval = self.restart_interval as usize;
            let total_mcus = mcu_h * mcu_v;
            let mut mcu_idx = 0;

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Y blocks in this MCU
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            let block = if y_bx < y_blocks_w && y_by < y_blocks_h {
                                let y_idx = y_by * y_blocks_w + y_bx;
                                &y_blocks[y_idx]
                            } else {
                                &ZERO_BLOCK
                            };
                            Self::collect_block_frequencies(
                                block,
                                prev_y_dc,
                                &mut dc_luma_freq,
                                &mut ac_luma_freq,
                            );
                            prev_y_dc = block[0];
                        }
                    }

                    // Chroma blocks
                    if is_color {
                        let (cb_block, cr_block) = if mcu_x < c_blocks_w && mcu_y < c_blocks_h {
                            let c_idx = mcu_y * c_blocks_w + mcu_x;
                            (&cb_blocks[c_idx], &cr_blocks[c_idx])
                        } else {
                            (&ZERO_BLOCK, &ZERO_BLOCK)
                        };

                        Self::collect_block_frequencies(
                            cb_block,
                            prev_cb_dc,
                            &mut dc_chroma_freq,
                            &mut ac_chroma_freq,
                        );
                        prev_cb_dc = cb_block[0];

                        Self::collect_block_frequencies(
                            cr_block,
                            prev_cr_dc,
                            &mut dc_chroma_freq,
                            &mut ac_chroma_freq,
                        );
                        prev_cr_dc = cr_block[0];
                    }

                    // Reset DC prediction at restart boundaries (same logic as encoder)
                    mcu_idx += 1;
                    if restart_interval > 0
                        && mcu_idx < total_mcus
                        && mcu_idx % restart_interval == 0
                    {
                        prev_y_dc = 0;
                        prev_cb_dc = 0;
                        prev_cr_dc = 0;
                    }
                }
            }
        }

        HuffmanSymbolFrequencies {
            dc_luma: dc_luma_freq,
            ac_luma: ac_luma_freq,
            dc_chroma: dc_chroma_freq,
            ac_chroma: ac_chroma_freq,
        }
    }

    pub(crate) fn build_optimized_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<HuffmanTableSet> {
        self.count_block_frequencies(y_blocks, cb_blocks, cr_blocks, is_color)
            .generate_tables()
    }

    pub(crate) fn build_optimized_tables_with_counts(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<(HuffmanTableSet, Box<HuffmanSymbolFrequencies>)> {
        let counts = self.count_block_frequencies(y_blocks, cb_blocks, cr_blocks, is_color);
        let tables = counts.generate_tables()?;
        Ok((tables, Box::new(counts)))
    }

    /// Encodes blocks using Huffman tables.
    ///
    /// If `tables` is Some, uses the optimized tables. If None, uses standard (fixed) tables.
    /// Handles MCU interleaving for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    pub(crate) fn encode_with_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
        tables: Option<&HuffmanTableSet>,
    ) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let (h_samp, v_samp) = match self.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        // Use parallel encoding when explicitly enabled
        #[cfg(feature = "parallel")]
        if self.parallel {
            // Auto-set restart interval if not specified
            let restart_interval = if self.restart_interval > 0 {
                self.restart_interval
            } else {
                64 // Default restart interval for parallel encoding
            };
            use super::parallel::{
                ParallelEntropyConfig, parallel_entropy_encode_444,
                parallel_entropy_encode_subsampled,
            };

            let config = if let Some(tables) = tables {
                ParallelEntropyConfig {
                    dc_luma: tables.dc_luma.table.clone(),
                    ac_luma: tables.ac_luma.table.clone(),
                    dc_chroma: tables.dc_chroma.table.clone(),
                    ac_chroma: tables.ac_chroma.table.clone(),
                }
            } else {
                ParallelEntropyConfig {
                    dc_luma: HuffmanEncodeTable::std_dc_luminance().clone(),
                    ac_luma: HuffmanEncodeTable::std_ac_luminance().clone(),
                    dc_chroma: HuffmanEncodeTable::std_dc_chrominance().clone(),
                    ac_chroma: HuffmanEncodeTable::std_ac_chrominance().clone(),
                }
            };

            return if h_samp == 1 && v_samp == 1 {
                Ok(parallel_entropy_encode_444(
                    y_blocks,
                    cb_blocks,
                    cr_blocks,
                    is_color,
                    restart_interval,
                    &config,
                ))
            } else {
                Ok(parallel_entropy_encode_subsampled(
                    y_blocks,
                    cb_blocks,
                    cr_blocks,
                    width,
                    height,
                    h_samp,
                    v_samp,
                    is_color,
                    restart_interval,
                    &config,
                ))
            };
        }

        // Sequential encoding path (default, or when parallel feature disabled)
        // Estimate output size: ~3 bytes/block average; Vec doubles if more needed
        let total_blocks = y_blocks.len() + cb_blocks.len() + cr_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 3);

        // Set up Huffman tables - optimized if provided, standard otherwise
        if let Some(tables) = tables {
            encoder.set_dc_table(0, &tables.dc_luma.table);
            encoder.set_ac_table(0, &tables.ac_luma.table);
            encoder.set_dc_table(1, &tables.dc_chroma.table);
            encoder.set_ac_table(1, &tables.ac_chroma.table);
        } else {
            encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
            encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
            encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
            encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());
        }

        if self.restart_interval > 0 {
            encoder.set_restart_interval(self.restart_interval);
        }

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            let total_mcus = y_blocks.len();
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0);

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1);
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1);
                }

                // Only check restart if not the last MCU
                if i + 1 < total_mcus {
                    encoder.check_restart();
                }
            }
        } else {
            // Subsampled mode - MCU interleaving
            let y_blocks_w = (width + 7) / 8;
            let y_blocks_h = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_w = (c_width + 7) / 8;
            let c_blocks_h = (c_height + 7) / 8;

            let mcu_h = (y_blocks_w + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_h + v_samp - 1) / v_samp;
            let total_mcus = mcu_h * mcu_v;

            // Zero block for padding out-of-bounds MCU positions
            const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

            let mut mcu_idx = 0;
            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Encode Y blocks in this MCU (must encode all even if out of bounds)
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            if y_bx < y_blocks_w && y_by < y_blocks_h {
                                let y_idx = y_by * y_blocks_w + y_bx;
                                encoder.encode_block(&y_blocks[y_idx], 0, 0, 0);
                            } else {
                                // Out of bounds - encode zero block (padding)
                                encoder.encode_block(&ZERO_BLOCK, 0, 0, 0);
                            }
                        }
                    }

                    // Encode Cb and Cr blocks (always, even if out of bounds)
                    if is_color {
                        if mcu_x < c_blocks_w && mcu_y < c_blocks_h {
                            let c_idx = mcu_y * c_blocks_w + mcu_x;
                            encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1);
                            encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1);
                        } else {
                            // Out of bounds - encode zero blocks (padding)
                            encoder.encode_block(&ZERO_BLOCK, 1, 1, 1);
                            encoder.encode_block(&ZERO_BLOCK, 2, 1, 1);
                        }
                    }

                    // Only check restart if not the last MCU
                    mcu_idx += 1;
                    if mcu_idx < total_mcus {
                        encoder.check_restart();
                    }
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Collects symbol frequencies from a block for Huffman optimization.
    /// Uses SIMD to build a nonzero mask and skip zero coefficients.
    fn collect_block_frequencies(
        coeffs: &[i16; DCT_BLOCK_SIZE],
        prev_dc: i16,
        dc_freq: &mut FrequencyCounter,
        ac_freq: &mut FrequencyCounter,
    ) {
        collect_block_frequencies_simd(coeffs, prev_dc, dc_freq, ac_freq);
    }
}

/// SIMD-accelerated frequency collection using nonzero mask.
/// Uses `build_nonzero_mask` (with archmage dispatch) for SIMD — no need for
/// `#[autoversion]` since this function is pure integer bit manipulation.
#[inline]
fn collect_block_frequencies_simd(
    coeffs: &[i16; DCT_BLOCK_SIZE],
    prev_dc: i16,
    dc_freq: &mut FrequencyCounter,
    ac_freq: &mut FrequencyCounter,
) {
    // DC coefficient - must match the unclamped category used in actual encoding
    // (entropy/mod.rs:172, entropy/encoder.rs:64,247,382). XYB can produce DC
    // differences > ±2047 (category 12+) at low quality. If we clamp here but
    // not during encoding, the Huffman table won't have codes for categories
    // 12+, causing (code=0, len=0) writes that corrupt the bitstream.
    let dc_diff = coeffs[0] - prev_dc;
    let dc_category = entropy::category(dc_diff);
    dc_freq.count(dc_category);

    // Build 64-bit mask of non-zero coefficients using SIMD
    let nonzero_mask = build_nonzero_mask(coeffs);

    // Clear DC bit (bit 0), keep only AC bits (1-63)
    let ac_mask = nonzero_mask & !1u64;

    // Fast path: all AC coefficients are zero
    if ac_mask == 0 {
        ac_freq.count(0x00); // EOB
        return;
    }

    // Find position of last non-zero AC coefficient (1-63)
    let last_nonzero_idx = 63 - ac_mask.leading_zeros() as usize;

    // Process each non-zero AC coefficient using bit manipulation
    let mut remaining = ac_mask;
    let mut prev_idx = 0usize;

    while remaining != 0 {
        let idx = remaining.trailing_zeros() as usize;
        let run = (idx - prev_idx - 1) as u8;

        // Encode runs of 16+ zeros (emit ZRL symbols)
        let mut r = run;
        while r >= 16 {
            ac_freq.count(0xF0); // ZRL
            r -= 16;
        }

        // Encode run/size symbol
        let ac = coeffs[idx];
        let ac_category = entropy::category(ac);
        let symbol = (r << 4) | ac_category;
        ac_freq.count(symbol);

        prev_idx = idx;
        remaining &= remaining - 1; // Clear lowest set bit
    }

    // EOB if there are trailing zeros
    if last_nonzero_idx < 63 {
        ac_freq.count(0x00); // EOB
    }
}

/// Build a 64-bit mask of non-zero coefficients using SIMD.
///
/// Archmage AVX2 path processes 16 coefficients per iteration (4 iterations total).
/// Scalar fallback processes 8 per iteration (8 iterations).
#[inline]
pub(crate) fn build_nonzero_mask(coeffs: &[i16; DCT_BLOCK_SIZE]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return mage_build_nonzero_mask(token, coeffs);
        }
    }
    scalar_build_nonzero_mask(coeffs)
}

/// AVX2 nonzero mask: 8 coefficients per iteration via magetypes i16x8.
///
/// Uses i16x8 (128-bit) instead of i16x16 (256-bit) because the i16x16
/// bitmask() implementation has lane-crossing issues with _mm256_packs_epi16
/// that produce incorrect results.
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn mage_build_nonzero_mask(_token: archmage::X64V3Token, coeffs: &[i16; DCT_BLOCK_SIZE]) -> u64 {
    use magetypes::simd::i16x8 as mi16x8;
    let token = _token;
    let zero = mi16x8::zero(token);
    let mut nonzero_mask: u64 = 0;

    // Process 8 coefficients at a time (8 chunks of 8 = 64 total)
    for chunk in 0..8 {
        let start = chunk * 8;
        let v = mi16x8::load(token, coeffs[start..start + 8].try_into().unwrap());
        let is_zero = v.simd_eq(zero);
        let zero_bits = is_zero.bitmask() as u8;
        let nonzero_bits = !zero_bits;
        nonzero_mask |= (nonzero_bits as u64) << start;
    }

    nonzero_mask
}

/// Scalar fallback: 8 coefficients per iteration via wide i16x8.
#[inline]
fn scalar_build_nonzero_mask(coeffs: &[i16; DCT_BLOCK_SIZE]) -> u64 {
    let zero = i16x8::ZERO;
    let mut nonzero_mask: u64 = 0;

    for chunk in 0..8 {
        let start = chunk * 8;
        let v = i16x8::new([
            coeffs[start],
            coeffs[start + 1],
            coeffs[start + 2],
            coeffs[start + 3],
            coeffs[start + 4],
            coeffs[start + 5],
            coeffs[start + 6],
            coeffs[start + 7],
        ]);
        let is_zero = v.simd_eq(zero);
        let zero_bits = is_zero.to_bitmask() as u8;
        let nonzero_bits = !zero_bits;
        nonzero_mask |= (nonzero_bits as u64) << start;
    }

    nonzero_mask
}

impl ComputedConfig {
    /// Builds optimized Huffman tables for XYB mode with raster-ordered blocks.
    ///
    /// This function handles blocks that are stored in raster order (row by row),
    /// as produced by the strip encoder, rather than MCU-interleaved order.
    ///
    /// XYB uses a single shared table for all components (luminance tables).
    pub(crate) fn build_optimized_tables_xyb_raster(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
    ) -> Result<(
        crate::huffman::optimize::OptimizedTable,
        crate::huffman::optimize::OptimizedTable,
    )> {
        let mut dc_freq = FrequencyCounter::new();
        let mut ac_freq = FrequencyCounter::new();

        let width = self.width as usize;
        let height = self.height as usize;

        // X and Y are full resolution
        let xy_blocks_w = (width + 7) / 8;
        let xy_blocks_h = (height + 7) / 8;

        // B is 2x2 downsampled
        let b_blocks_w = (width + 15) / 16;
        let b_blocks_h = (height + 15) / 16;

        // MCU is 16x16 pixels (2x2 blocks for X/Y, 1x1 for B)
        let mcu_h = (xy_blocks_w + 1) / 2;
        let mcu_v = (xy_blocks_h + 1) / 2;

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        // Each component maintains its own DC prediction
        let mut prev_dc_x: i16 = 0;
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_b: i16 = 0;

        for mcu_y in 0..mcu_v {
            for mcu_x in 0..mcu_h {
                // X blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            let idx = by * xy_blocks_w + bx;
                            &x_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        Self::collect_block_frequencies(
                            block,
                            prev_dc_x,
                            &mut dc_freq,
                            &mut ac_freq,
                        );
                        prev_dc_x = block[0];
                    }
                }

                // Y blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            let idx = by * xy_blocks_w + bx;
                            &y_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        Self::collect_block_frequencies(
                            block,
                            prev_dc_y,
                            &mut dc_freq,
                            &mut ac_freq,
                        );
                        prev_dc_y = block[0];
                    }
                }

                // B block (1 per MCU)
                let b_block = if mcu_x < b_blocks_w && mcu_y < b_blocks_h {
                    let idx = mcu_y * b_blocks_w + mcu_x;
                    &b_blocks[idx]
                } else {
                    &ZERO_BLOCK
                };
                Self::collect_block_frequencies(b_block, prev_dc_b, &mut dc_freq, &mut ac_freq);
                prev_dc_b = b_block[0];
            }
        }

        // Use jpegli's Huffman algorithm (matches C++ behavior)
        let huffman_method = crate::types::HuffmanMethod::JpegliCreateTree;

        // Generate optimized tables
        let dc_table = dc_freq.generate_table_with_method(huffman_method)?;
        let ac_table = ac_freq.generate_table_with_method(huffman_method)?;

        Ok((dc_table, ac_table))
    }

    /// Like [`Self::build_optimized_tables_xyb_raster`], but also returns the
    /// raw frequency counters for multi-image aggregation.
    ///
    /// The returned `HuffmanSymbolFrequencies` stores XYB's shared DC/AC
    /// frequencies in the `dc_luma`/`ac_luma` slots. The `dc_chroma` and
    /// `ac_chroma` slots are empty (XYB uses a single table pair for all
    /// three components).
    pub(crate) fn build_optimized_tables_xyb_raster_with_counts(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
    ) -> Result<(
        crate::huffman::optimize::OptimizedTable,
        crate::huffman::optimize::OptimizedTable,
        Box<HuffmanSymbolFrequencies>,
    )> {
        let mut dc_freq = FrequencyCounter::new();
        let mut ac_freq = FrequencyCounter::new();

        let width = self.width as usize;
        let height = self.height as usize;

        let xy_blocks_w = (width + 7) / 8;
        let xy_blocks_h = (height + 7) / 8;
        let b_blocks_w = (width + 15) / 16;
        let b_blocks_h = (height + 15) / 16;
        let mcu_h = (xy_blocks_w + 1) / 2;
        let mcu_v = (xy_blocks_h + 1) / 2;

        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        let mut prev_dc_x: i16 = 0;
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_b: i16 = 0;

        for mcu_y in 0..mcu_v {
            for mcu_x in 0..mcu_h {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            &x_blocks[by * xy_blocks_w + bx]
                        } else {
                            &ZERO_BLOCK
                        };
                        Self::collect_block_frequencies(
                            block,
                            prev_dc_x,
                            &mut dc_freq,
                            &mut ac_freq,
                        );
                        prev_dc_x = block[0];
                    }
                }

                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            &y_blocks[by * xy_blocks_w + bx]
                        } else {
                            &ZERO_BLOCK
                        };
                        Self::collect_block_frequencies(
                            block,
                            prev_dc_y,
                            &mut dc_freq,
                            &mut ac_freq,
                        );
                        prev_dc_y = block[0];
                    }
                }

                let b_block = if mcu_x < b_blocks_w && mcu_y < b_blocks_h {
                    &b_blocks[mcu_y * b_blocks_w + mcu_x]
                } else {
                    &ZERO_BLOCK
                };
                Self::collect_block_frequencies(b_block, prev_dc_b, &mut dc_freq, &mut ac_freq);
                prev_dc_b = b_block[0];
            }
        }

        let huffman_method = crate::types::HuffmanMethod::JpegliCreateTree;
        let dc_table = dc_freq.generate_table_with_method(huffman_method)?;
        let ac_table = ac_freq.generate_table_with_method(huffman_method)?;

        // Store shared XYB frequencies in the luma slots; chroma slots stay empty.
        let frequencies = Box::new(HuffmanSymbolFrequencies {
            dc_luma: dc_freq,
            ac_luma: ac_freq,
            dc_chroma: FrequencyCounter::new(),
            ac_chroma: FrequencyCounter::new(),
        });

        Ok((dc_table, ac_table, frequencies))
    }

    /// Encodes XYB raster-ordered blocks using optimized Huffman tables.
    pub(crate) fn encode_with_tables_xyb_raster(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
        dc_table: &crate::huffman::optimize::OptimizedTable,
        ac_table: &crate::huffman::optimize::OptimizedTable,
    ) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // X and Y are full resolution
        let xy_blocks_w = (width + 7) / 8;
        let xy_blocks_h = (height + 7) / 8;

        // B is 2x2 downsampled
        let b_blocks_w = (width + 15) / 16;
        let b_blocks_h = (height + 15) / 16;

        // MCU is 16x16 pixels
        let mcu_h = (xy_blocks_w + 1) / 2;
        let mcu_v = (xy_blocks_h + 1) / 2;

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        // Estimate output size (~3 bytes/block average; Vec doubles if more needed)
        let total_blocks = x_blocks.len() + y_blocks.len() + b_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 3);

        // Use the same optimized table for all components
        encoder.set_dc_table(0, &dc_table.table);
        encoder.set_ac_table(0, &ac_table.table);

        if self.restart_interval > 0 {
            encoder.set_restart_interval(self.restart_interval);
        }

        let total_mcus = mcu_h * mcu_v;
        let mut mcu_idx = 0;

        for mcu_y in 0..mcu_v {
            for mcu_x in 0..mcu_h {
                // X blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            let idx = by * xy_blocks_w + bx;
                            &x_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 0, 0, 0);
                    }
                }

                // Y blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            let idx = by * xy_blocks_w + bx;
                            &y_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 1, 0, 0);
                    }
                }

                // B block (1 per MCU)
                let b_block = if mcu_x < b_blocks_w && mcu_y < b_blocks_h {
                    let idx = mcu_y * b_blocks_w + mcu_x;
                    &b_blocks[idx]
                } else {
                    &ZERO_BLOCK
                };
                encoder.encode_block(b_block, 2, 0, 0);

                // Only check restart if not the last MCU
                mcu_idx += 1;
                if mcu_idx < total_mcus {
                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Encodes XYB raster-ordered blocks using standard (non-optimized) Huffman tables.
    pub(crate) fn encode_with_tables_xyb_standard_raster(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
    ) -> Result<Vec<u8>> {
        let width = self.width as usize;
        let height = self.height as usize;

        // X and Y are full resolution
        let xy_blocks_w = (width + 7) / 8;
        let xy_blocks_h = (height + 7) / 8;

        // B is 2x2 downsampled
        let b_blocks_w = (width + 15) / 16;
        let b_blocks_h = (height + 15) / 16;

        // MCU is 16x16 pixels
        let mcu_h = (xy_blocks_w + 1) / 2;
        let mcu_v = (xy_blocks_h + 1) / 2;

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        // Estimate output size (~3 bytes/block average; Vec doubles if more needed)
        let total_blocks = x_blocks.len() + y_blocks.len() + b_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 3);

        // Use standard luminance tables for all components in XYB mode
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());

        if self.restart_interval > 0 {
            encoder.set_restart_interval(self.restart_interval);
        }

        let total_mcus = mcu_h * mcu_v;
        let mut mcu_idx = 0;

        for mcu_y in 0..mcu_v {
            for mcu_x in 0..mcu_h {
                // X blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            let idx = by * xy_blocks_w + bx;
                            &x_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 0, 0, 0);
                    }
                }

                // Y blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_w && by < xy_blocks_h {
                            let idx = by * xy_blocks_w + bx;
                            &y_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 1, 0, 0);
                    }
                }

                // B block (1 per MCU)
                let b_block = if mcu_x < b_blocks_w && mcu_y < b_blocks_h {
                    let idx = mcu_y * b_blocks_w + mcu_x;
                    &b_blocks[idx]
                } else {
                    &ZERO_BLOCK
                };
                encoder.encode_block(b_block, 2, 0, 0);

                // Only check restart if not the last MCU
                mcu_idx += 1;
                if mcu_idx < total_mcus {
                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_mask(coeffs: &[i16; 64]) -> u64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if coeffs[i] != 0 {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    #[test]
    fn test_build_nonzero_mask_all_positions() {
        // Single nonzero at each of 64 positions
        for pos in 0..64 {
            let mut b = [0i16; 64];
            b[pos] = 1;
            let expected = 1u64 << pos;
            let got = build_nonzero_mask(&b);
            assert_eq!(
                got, expected,
                "pos {pos}: expected {expected:#066b}, got {got:#066b}"
            );
        }
    }

    #[test]
    fn test_build_nonzero_mask_patterns() {
        // All zeros
        assert_eq!(build_nonzero_mask(&[0i16; 64]), 0);

        // All nonzero
        let mut all = [0i16; 64];
        for i in 0..64 {
            all[i] = (i as i16) + 1;
        }
        assert_eq!(build_nonzero_mask(&all), u64::MAX);

        // Alternating
        let mut alt = [0i16; 64];
        for i in (0..64).step_by(2) {
            alt[i] = 42;
        }
        assert_eq!(build_nonzero_mask(&alt), reference_mask(&alt));

        // Negative values
        let mut neg = [0i16; 64];
        for i in 8..16 {
            neg[i] = -1;
        }
        assert_eq!(build_nonzero_mask(&neg), reference_mask(&neg));
    }

    #[test]
    fn test_build_nonzero_mask_scalar_matches_dispatch() {
        // Verify scalar fallback matches dispatcher on multiple patterns
        let patterns: &[[i16; 64]] = &[
            [0i16; 64],
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b[1] = -50;
                b[8] = 20;
                b[63] = 5;
                b
            },
            {
                let mut b = [0i16; 64];
                for i in 0..64 {
                    b[i] = if i % 3 == 0 { (i as i16) - 20 } else { 0 };
                }
                b
            },
        ];

        for (idx, block) in patterns.iter().enumerate() {
            let scalar = scalar_build_nonzero_mask(block);
            let dispatch = build_nonzero_mask(block);
            assert_eq!(
                scalar, dispatch,
                "pattern {idx}: scalar and dispatch disagree"
            );
        }
    }
}
