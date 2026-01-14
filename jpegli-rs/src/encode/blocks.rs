//! Block operations for JPEG encoding.
//!
//! This module contains:
//! - Block quantization functions (YCbCr and XYB)
//! - Block extraction from planes
//! - Huffman table optimization
//! - Scan encoding

#![allow(deprecated)] // This module implements methods for the deprecated Encoder struct

use super::Encoder;
use crate::foundation::consts::DCT_BLOCK_SIZE;
#[cfg(feature = "experimental-hybrid-trellis")]
use crate::encode::hybrid;
use crate::entropy::{self, EntropyEncoder};
use crate::error::Result;
use crate::huffman::optimize::{FrequencyCounter, OptimizedHuffmanTables};
use crate::huffman::HuffmanEncodeTable;
use crate::types::Subsampling;

impl Encoder {
    pub(crate) fn build_optimized_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<OptimizedHuffmanTables> {
        let mut dc_luma_freq = FrequencyCounter::new();
        let mut dc_chroma_freq = FrequencyCounter::new();
        let mut ac_luma_freq = FrequencyCounter::new();
        let mut ac_chroma_freq = FrequencyCounter::new();

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
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
            let restart_interval = self.config.restart_interval as usize;
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
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;
            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            // Restart interval tracking (must match encoder behavior exactly)
            let restart_interval = self.config.restart_interval as usize;
            let total_mcus = mcu_h * mcu_v;
            let mut mcu_idx = 0;

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Y blocks in this MCU
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            let block = if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
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
                        let (cb_block, cr_block) = if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
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

        // Use jpegli's Huffman algorithm (matches C++ behavior)
        let huffman_method = crate::types::HuffmanMethod::JpegliCreateTree;

        // Build optimized tables with DHT data using selected algorithm
        let dc_luma = dc_luma_freq.generate_table_with_method(huffman_method)?;
        let ac_luma = ac_luma_freq.generate_table_with_method(huffman_method)?;

        let (dc_chroma, ac_chroma) = if is_color {
            (
                dc_chroma_freq.generate_table_with_method(huffman_method)?,
                ac_chroma_freq.generate_table_with_method(huffman_method)?,
            )
        } else {
            // Use standard tables for grayscale (won't be used but needed for structure)
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
        };

        Ok(OptimizedHuffmanTables {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
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
        tables: Option<&OptimizedHuffmanTables>,
    ) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        // Use parallel encoding when explicitly enabled
        #[cfg(feature = "parallel")]
        if self.config.parallel {
            // Auto-set restart interval if not specified
            let restart_interval = if self.config.restart_interval > 0 {
                self.config.restart_interval
            } else {
                64 // Default restart interval for parallel encoding
            };
            use super::parallel::{
                parallel_entropy_encode_444, parallel_entropy_encode_subsampled,
                ParallelEntropyConfig,
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
        // Estimate output size: ~100 bytes per block for typical quality
        let total_blocks = y_blocks.len() + cb_blocks.len() + cr_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 100);

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

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            let total_mcus = y_blocks.len();
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0)?;

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
                }

                // Only check restart if not the last MCU
                if i + 1 < total_mcus {
                    encoder.check_restart();
                }
            }
        } else {
            // Subsampled mode - MCU interleaving
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;

            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;
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
                            if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                encoder.encode_block(&y_blocks[y_idx], 0, 0, 0)?;
                            } else {
                                // Out of bounds - encode zero block (padding)
                                encoder.encode_block(&ZERO_BLOCK, 0, 0, 0)?;
                            }
                        }
                    }

                    // Encode Cb and Cr blocks (always, even if out of bounds)
                    if is_color {
                        if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1)?;
                            encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1)?;
                        } else {
                            // Out of bounds - encode zero blocks (padding)
                            encoder.encode_block(&ZERO_BLOCK, 1, 1, 1)?;
                            encoder.encode_block(&ZERO_BLOCK, 2, 1, 1)?;
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
    fn collect_block_frequencies(
        coeffs: &[i16; DCT_BLOCK_SIZE],
        prev_dc: i16,
        dc_freq: &mut FrequencyCounter,
        ac_freq: &mut FrequencyCounter,
    ) {
        // DC coefficient - limit category to 11 for 8-bit JPEG compatibility
        let dc_diff = coeffs[0] - prev_dc;
        let dc_category = entropy::category(dc_diff).min(11);
        dc_freq.count(dc_category);

        // AC coefficients
        let mut run = 0u8;
        for i in 1..DCT_BLOCK_SIZE {
            let ac = coeffs[i];

            if ac == 0 {
                run += 1;
            } else {
                // Encode runs of 16 zeros (ZRL)
                while run >= 16 {
                    ac_freq.count(0xF0);
                    run -= 16;
                }

                // Encode run/size symbol
                let ac_category = entropy::category(ac);
                let symbol = (run << 4) | ac_category;
                ac_freq.count(symbol);
                run = 0;
            }
        }

        // EOB if trailing zeros
        if run > 0 {
            ac_freq.count(0x00);
        }
    }

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

        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // X and Y are full resolution
        let xy_blocks_h = (width + 7) / 8;
        let xy_blocks_v = (height + 7) / 8;

        // B is 2x2 downsampled
        let b_blocks_h = (width + 15) / 16;
        let b_blocks_v = (height + 15) / 16;

        // MCU is 16x16 pixels (2x2 blocks for X/Y, 1x1 for B)
        let mcu_h = (xy_blocks_h + 1) / 2;
        let mcu_v = (xy_blocks_v + 1) / 2;

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
                        let block = if bx < xy_blocks_h && by < xy_blocks_v {
                            let idx = by * xy_blocks_h + bx;
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
                        let block = if bx < xy_blocks_h && by < xy_blocks_v {
                            let idx = by * xy_blocks_h + bx;
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
                let b_block = if mcu_x < b_blocks_h && mcu_y < b_blocks_v {
                    let idx = mcu_y * b_blocks_h + mcu_x;
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

    /// Encodes XYB raster-ordered blocks using optimized Huffman tables.
    pub(crate) fn encode_with_tables_xyb_raster(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
        dc_table: &crate::huffman::optimize::OptimizedTable,
        ac_table: &crate::huffman::optimize::OptimizedTable,
    ) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // X and Y are full resolution
        let xy_blocks_h = (width + 7) / 8;
        let xy_blocks_v = (height + 7) / 8;

        // B is 2x2 downsampled
        let b_blocks_h = (width + 15) / 16;
        let b_blocks_v = (height + 15) / 16;

        // MCU is 16x16 pixels
        let mcu_h = (xy_blocks_h + 1) / 2;
        let mcu_v = (xy_blocks_v + 1) / 2;

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        // Estimate output size
        let total_blocks = x_blocks.len() + y_blocks.len() + b_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 100);

        // Use the same optimized table for all components
        encoder.set_dc_table(0, &dc_table.table);
        encoder.set_ac_table(0, &ac_table.table);

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        for mcu_y in 0..mcu_v {
            for mcu_x in 0..mcu_h {
                // X blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_h && by < xy_blocks_v {
                            let idx = by * xy_blocks_h + bx;
                            &x_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 0, 0, 0)?;
                    }
                }

                // Y blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_h && by < xy_blocks_v {
                            let idx = by * xy_blocks_h + bx;
                            &y_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 1, 0, 0)?;
                    }
                }

                // B block (1 per MCU)
                let b_block = if mcu_x < b_blocks_h && mcu_y < b_blocks_v {
                    let idx = mcu_y * b_blocks_h + mcu_x;
                    &b_blocks[idx]
                } else {
                    &ZERO_BLOCK
                };
                encoder.encode_block(b_block, 2, 0, 0)?;

                encoder.check_restart();
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
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // X and Y are full resolution
        let xy_blocks_h = (width + 7) / 8;
        let xy_blocks_v = (height + 7) / 8;

        // B is 2x2 downsampled
        let b_blocks_h = (width + 15) / 16;
        let b_blocks_v = (height + 15) / 16;

        // MCU is 16x16 pixels
        let mcu_h = (xy_blocks_h + 1) / 2;
        let mcu_v = (xy_blocks_v + 1) / 2;

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        // Estimate output size
        let total_blocks = x_blocks.len() + y_blocks.len() + b_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 100);

        // Use standard luminance tables for all components in XYB mode
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        for mcu_y in 0..mcu_v {
            for mcu_x in 0..mcu_h {
                // X blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_h && by < xy_blocks_v {
                            let idx = by * xy_blocks_h + bx;
                            &x_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 0, 0, 0)?;
                    }
                }

                // Y blocks (4 per MCU in 2x2 arrangement)
                for dy in 0..2 {
                    for dx in 0..2 {
                        let bx = mcu_x * 2 + dx;
                        let by = mcu_y * 2 + dy;
                        let block = if bx < xy_blocks_h && by < xy_blocks_v {
                            let idx = by * xy_blocks_h + bx;
                            &y_blocks[idx]
                        } else {
                            &ZERO_BLOCK
                        };
                        encoder.encode_block(block, 1, 0, 0)?;
                    }
                }

                // B block (1 per MCU)
                let b_block = if mcu_x < b_blocks_h && mcu_y < b_blocks_v {
                    let idx = mcu_y * b_blocks_h + mcu_x;
                    &b_blocks[idx]
                } else {
                    &ZERO_BLOCK
                };
                encoder.encode_block(b_block, 2, 0, 0)?;

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }
}
