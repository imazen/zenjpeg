//! Block operations for JPEG encoding.
//!
//! This module contains:
//! - Block quantization functions (YCbCr and XYB)
//! - Block extraction from planes
//! - Huffman table optimization
//! - Scan encoding

use super::super::{natural_to_zigzag, natural_to_zigzag_into, Encoder};
use crate::consts::{DCT_BLOCK_SIZE, DCT_SIZE};
use crate::dct::forward_dct_8x8;
#[cfg(feature = "experimental-hybrid-trellis")]
use crate::encode::hybrid;
use crate::entropy::{self, EntropyEncoder};
use crate::error::Result;
use crate::huffman::optimize::{FrequencyCounter, OptimizedHuffmanTables};
use crate::huffman::HuffmanEncodeTable;
use crate::quant::aq::compute_aq_strength_map;
use crate::quant::{self, QuantTable, ZeroBiasParams};
use crate::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::types::{PixelFormat, Subsampling};
use enough::Stop;

impl Encoder {
    /// Quantizes all blocks in the image (4:4:4 only, no subsampling).
    ///
    /// This is separated from encoding to allow Huffman optimization:
    /// 1. Quantize all blocks
    /// 2. Collect frequencies to build optimal tables
    /// 3. Encode with optimal tables
    ///
    /// Note: This function handles non-subsampled 4:4:4 mode. For subsampled modes,
    /// use `quantize_all_blocks_subsampled` instead.
    #[allow(dead_code)] // Kept for potential future use with non-AQ 4:4:4 paths
    pub(crate) fn quantize_all_blocks(
        &self,
        y_plane: &[f32],
        cb_plane: &[f32],
        cr_plane: &[f32],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Zero-bias parameters for each component
        // Use effective distance inferred from quant tables (like C++ QuantValsToDistance)
        // This is important at Q100 where quant values are all 1s but input distance is 0.01
        let _input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map =
            hybrid::get_aq_map_or_compute(&self.config, y_plane, width, height, y_quant_01)?;
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(y_plane, width, height, y_quant_01)?;

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = hybrid::create_hybrid_ctx(&self.config);

        // Pre-allocate block arrays to avoid push() overhead
        let num_blocks = blocks_h * blocks_v;
        let num_chroma_blocks = if is_color { num_blocks } else { 0 };
        let mut y_blocks =
            crate::foundation::alloc::try_alloc_dct_blocks(num_blocks, "y_blocks encode_scan")?;
        let mut cb_blocks = crate::foundation::alloc::try_alloc_dct_blocks(
            num_chroma_blocks,
            "cb_blocks encode_scan",
        )?;
        let mut cr_blocks = crate::foundation::alloc::try_alloc_dct_blocks(
            num_chroma_blocks,
            "cr_blocks encode_scan",
        )?;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                // Get per-block aq_strength
                let aq_strength = aq_map.get(bx, by);

                let y_block =
                    crate::encode_simd::extract_block_simd(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = hybrid::quantize_block_dispatch(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                    true,
                    hybrid_ctx.as_ref(),
                );
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                );

                natural_to_zigzag_into(&y_quant_coeffs, &mut y_blocks[block_idx]);

                if is_color {
                    let cb_block =
                        crate::encode_simd::extract_block_simd(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cb_dct,
                        &cb_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                        false,
                        hybrid_ctx.as_ref(),
                    );
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cb_dct,
                        &cb_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                    );

                    natural_to_zigzag_into(&cb_quant_coeffs, &mut cb_blocks[block_idx]);

                    let cr_block =
                        crate::encode_simd::extract_block_simd(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cr_dct,
                        &cr_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                        false,
                        hybrid_ctx.as_ref(),
                    );
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                        &cr_dct,
                        &cr_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                    );

                    natural_to_zigzag_into(&cr_quant_coeffs, &mut cr_blocks[block_idx]);
                }
            }
        }

        Ok((y_blocks, cb_blocks, cr_blocks))
    }

    /// Quantizes all blocks with subsampling support.
    ///
    /// Unlike `quantize_all_blocks`, this version handles different dimensions
    /// for Y and chroma planes (needed for 4:2:0, 4:2:2, 4:4:0 subsampling).
    ///
    /// Uses SIMD-native types (Block8x8f, QuantTableSimd, ZeroBiasSimd) for
    /// optimized DCT and quantization with minimal load/store overhead.
    ///
    /// If `stop` is Some, checks for cancellation at each MCU row boundary.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_all_blocks_subsampled(
        &self,
        y_plane: &[f32],
        y_width: usize,
        y_height: usize,
        cb_plane: &[f32],
        cr_plane: &[f32],
        c_width: usize,
        c_height: usize,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
        stop: Option<&dyn Stop>,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        let y_blocks_h = (y_width + 7) / 8;
        let y_blocks_v = (y_height + 7) / 8;
        let c_blocks_h = (c_width + 7) / 8;
        let c_blocks_v = (c_height + 7) / 8;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Zero-bias parameters for each component
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Pre-compute SIMD-native quantization tables and zero-bias params
        let y_quant_simd = QuantTableSimd::from_values(&y_quant.values);
        let y_zero_bias_simd = ZeroBiasSimd::from_params(&y_zero_bias);
        let cb_quant_simd = QuantTableSimd::from_values(&cb_quant.values);
        let cb_zero_bias_simd = ZeroBiasSimd::from_params(&cb_zero_bias);
        let cr_quant_simd = QuantTableSimd::from_values(&cr_quant.values);
        let cr_zero_bias_simd = ZeroBiasSimd::from_params(&cr_zero_bias);

        // Compute per-block adaptive quantization strength from Y plane
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map =
            hybrid::get_aq_map_or_compute(&self.config, y_plane, y_width, y_height, y_quant_01)?;
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(y_plane, y_width, y_height, y_quant_01)?;

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = hybrid::create_hybrid_ctx(&self.config);

        // Pre-allocate block arrays to avoid push() overhead
        let num_y_blocks = y_blocks_h * y_blocks_v;
        let num_c_blocks = if is_color { c_blocks_h * c_blocks_v } else { 0 };
        let mut y_blocks = crate::foundation::alloc::try_alloc_dct_blocks(
            num_y_blocks,
            "y_blocks encode_scan_subsampled",
        )?;
        let mut cb_blocks = crate::foundation::alloc::try_alloc_dct_blocks(
            num_c_blocks,
            "cb_blocks encode_scan_subsampled",
        )?;
        let mut cr_blocks = crate::foundation::alloc::try_alloc_dct_blocks(
            num_c_blocks,
            "cr_blocks encode_scan_subsampled",
        )?;

        // Quantize Y blocks using SIMD-optimized pipeline (pre-computed 1/quant, fast_round_int)
        for by in 0..y_blocks_v {
            // Check for cancellation at each MCU row (cooperative cancellation)
            // This is a no-op for Never, optimized out by the compiler
            if let Some(stop) = stop {
                stop.check()?;
            }
            for bx in 0..y_blocks_h {
                let block_idx = by * y_blocks_h + bx;
                let aq_strength = aq_map.get(bx, by);

                // Extract block and perform DCT (returns [f32; 64])
                let y_block =
                    crate::encode_simd::extract_block_simd(y_plane, y_width, y_height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = hybrid::quantize_block_dispatch(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                    true,
                    hybrid_ctx.as_ref(),
                );
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = y_quant_simd.quantize_array_with_zero_bias(
                    &y_dct,
                    &y_zero_bias_simd,
                    aq_strength,
                );

                natural_to_zigzag_into(&y_quant_coeffs, &mut y_blocks[block_idx]);
            }
        }

        // Quantize chroma blocks using SIMD-optimized pipeline
        if is_color {
            for by in 0..c_blocks_v {
                // Check for cancellation at each MCU row
                if let Some(stop) = stop {
                    stop.check()?;
                }
                for bx in 0..c_blocks_h {
                    let block_idx = by * c_blocks_h + bx;
                    // For chroma, use average AQ strength from corresponding Y region
                    // For 4:2:0, each chroma block corresponds to 2x2 Y blocks
                    let y_bx = (bx * y_blocks_h) / c_blocks_h;
                    let y_by = (by * y_blocks_v) / c_blocks_v;
                    let aq_strength =
                        aq_map.get(y_bx.min(y_blocks_h - 1), y_by.min(y_blocks_v - 1));

                    // Extract block and perform DCT for Cb
                    let cb_block =
                        crate::encode_simd::extract_block_simd(cb_plane, c_width, c_height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cb_dct,
                        &cb_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                        false,
                        hybrid_ctx.as_ref(),
                    );
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = cb_quant_simd.quantize_array_with_zero_bias(
                        &cb_dct,
                        &cb_zero_bias_simd,
                        aq_strength,
                    );

                    natural_to_zigzag_into(&cb_quant_coeffs, &mut cb_blocks[block_idx]);

                    // Extract block and perform DCT for Cr
                    let cr_block =
                        crate::encode_simd::extract_block_simd(cr_plane, c_width, c_height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = hybrid::quantize_block_dispatch(
                        &cr_dct,
                        &cr_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                        false,
                        hybrid_ctx.as_ref(),
                    );
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = cr_quant_simd.quantize_array_with_zero_bias(
                        &cr_dct,
                        &cr_zero_bias_simd,
                        aq_strength,
                    );

                    natural_to_zigzag_into(&cr_quant_coeffs, &mut cr_blocks[block_idx]);
                }
            }
        }

        Ok((y_blocks, cb_blocks, cr_blocks))
    }

    /// Builds optimized Huffman tables from quantized blocks.
    ///
    /// Collects symbol frequencies from all blocks and generates optimal
    /// Huffman tables with their DHT marker representations.
    ///
    /// For subsampled modes, this iterates blocks in MCU order to correctly
    /// account for padding blocks.
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

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0)?;

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
                }

                encoder.check_restart();
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

            // Zero block for padding out-of-bounds MCU positions
            const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

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

                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Reorders blocks from MCU order to raster order for XYB progressive encoding.
    ///
    /// For non-interleaved progressive scans, the JPEG decoder expects blocks
    /// in raster order (row by row), not MCU order.
    ///
    /// XYB quantization produces blocks in MCU order:
    /// - MCU 0: (0,0), (1,0), (0,1), (1,1) at indices 0,1,2,3
    /// - MCU 1: (2,0), (3,0), (2,1), (3,1) at indices 4,5,6,7
    ///
    /// But progressive scans need raster order:
    /// - Row 0: (0,0), (1,0), (2,0), (3,0), ... at indices 0,1,2,3,...
    /// - Row 1: (0,1), (1,1), (2,1), (3,1), ... at indices 8,9,10,11,...
    pub(crate) fn reorder_mcu_to_raster(
        mcu_blocks: &[[i16; DCT_BLOCK_SIZE]],
        blocks_x: usize,
        blocks_y: usize,
    ) -> Result<Vec<[i16; DCT_BLOCK_SIZE]>> {
        let total_blocks = blocks_x * blocks_y;
        let mut raster = crate::alloc::try_alloc_dct_blocks(total_blocks, "raster blocks")?;

        let mcu_cols = (blocks_x + 1) / 2;

        // Iterate through MCU-ordered blocks and place in raster order
        for (mcu_idx, chunk) in mcu_blocks.chunks(4).enumerate() {
            let mcu_x = mcu_idx % mcu_cols;
            let mcu_y = mcu_idx / mcu_cols;

            // Within each MCU, blocks are in order: (0,0), (1,0), (0,1), (1,1)
            // which corresponds to positions:
            // [0]: (mcu_x*2 + 0, mcu_y*2 + 0) = top-left
            // [1]: (mcu_x*2 + 1, mcu_y*2 + 0) = top-right
            // [2]: (mcu_x*2 + 0, mcu_y*2 + 1) = bottom-left
            // [3]: (mcu_x*2 + 1, mcu_y*2 + 1) = bottom-right
            for (i, block) in chunk.iter().enumerate() {
                let dx = i % 2;
                let dy = i / 2;
                let bx = mcu_x * 2 + dx;
                let by = mcu_y * 2 + dy;

                if bx < blocks_x && by < blocks_y {
                    let raster_idx = by * blocks_x + bx;
                    raster[raster_idx] = *block;
                }
            }
        }

        Ok(raster)
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

    /// Quantizes all XYB blocks for Huffman optimization.
    ///
    /// Returns quantized blocks for X, Y, and B components.
    /// B component is already downsampled (half resolution).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Reserved for future XYB encoding improvements
    pub(crate) fn quantize_all_blocks_xyb(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        // Pre-allocate block arrays to avoid push() overhead
        let mut x_blocks = crate::alloc::try_alloc_dct_blocks(num_xy_blocks, "x_blocks")?;
        let mut y_blocks = crate::alloc::try_alloc_dct_blocks(num_xy_blocks, "y_blocks")?;
        let mut b_blocks = crate::alloc::try_alloc_dct_blocks(num_b_blocks, "b_blocks")?;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let mcu_idx = mcu_y * mcu_cols + mcu_x;
                let xy_base = mcu_idx * 4; // 4 blocks per MCU for X and Y

                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let block_offset = block_y * 2 + block_x;
                        let x_block = crate::encode_simd::extract_block_xyb_simd(
                            x_plane, width, height, bx, by,
                        );
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block(&x_dct, &x_quant.values);
                        natural_to_zigzag_into(
                            &x_quant_coeffs,
                            &mut x_blocks[xy_base + block_offset],
                        );
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let block_offset = block_y * 2 + block_x;
                        let y_block = crate::encode_simd::extract_block_xyb_simd(
                            y_plane, width, height, bx, by,
                        );
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        natural_to_zigzag_into(
                            &y_quant_coeffs,
                            &mut y_blocks[xy_base + block_offset],
                        );
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = crate::encode_simd::extract_block_xyb_simd(
                    b_plane, b_width, b_height, mcu_x, mcu_y,
                );
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
                natural_to_zigzag_into(&b_quant_coeffs, &mut b_blocks[mcu_idx]);
            }
        }

        Ok((x_blocks, y_blocks, b_blocks))
    }

    /// Quantizes all XYB blocks with jpegli-style adaptive quantization (no trellis).
    ///
    /// This version uses the AQ map for per-block modulation with zero-bias,
    /// matching jpegli's default AQ behavior without hybrid trellis.
    ///
    /// For XYB mode:
    /// - X and Y use luma tables (both are full-resolution "luma-like" channels)
    /// - B uses chroma tables (downsampled blue channel)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_all_blocks_xyb_with_aq_simple(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
        aq_map: &crate::adaptive_quant::AQStrengthMap,
        x_zero_bias: &ZeroBiasParams,
        y_zero_bias: &ZeroBiasParams,
        b_zero_bias: &ZeroBiasParams,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        // Pre-allocate block arrays to avoid push() overhead
        let mut x_blocks = crate::alloc::try_alloc_dct_blocks(num_xy_blocks, "x_blocks")?;
        let mut y_blocks = crate::alloc::try_alloc_dct_blocks(num_xy_blocks, "y_blocks")?;
        let mut b_blocks = crate::alloc::try_alloc_dct_blocks(num_b_blocks, "b_blocks")?;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let mcu_idx = mcu_y * mcu_cols + mcu_x;
                let xy_base = mcu_idx * 4; // 4 blocks per MCU for X and Y

                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let block_offset = block_y * 2 + block_x;
                        let aq_strength = aq_map.get(bx, by);

                        let x_block = crate::encode_simd::extract_block_xyb_simd(
                            x_plane, width, height, bx, by,
                        );
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                            &x_dct,
                            &x_quant.values,
                            x_zero_bias,
                            aq_strength,
                        );
                        natural_to_zigzag_into(
                            &x_quant_coeffs,
                            &mut x_blocks[xy_base + block_offset],
                        );
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let block_offset = block_y * 2 + block_x;
                        let aq_strength = aq_map.get(bx, by);

                        let y_block = crate::encode_simd::extract_block_xyb_simd(
                            y_plane, width, height, bx, by,
                        );
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                            &y_dct,
                            &y_quant.values,
                            y_zero_bias,
                            aq_strength,
                        );
                        natural_to_zigzag_into(
                            &y_quant_coeffs,
                            &mut y_blocks[xy_base + block_offset],
                        );
                    }
                }

                // Process 1 B block (from downsampled plane)
                // For B channel: Average AQ from 4 parent full-res blocks
                let b_aq_strength = {
                    let mut sum = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let bx = mcu_x * 2 + dx;
                            let by = mcu_y * 2 + dy;
                            sum += aq_map.get(bx, by);
                        }
                    }
                    sum / 4.0
                };

                let b_block = crate::encode_simd::extract_block_xyb_simd(
                    b_plane, b_width, b_height, mcu_x, mcu_y,
                );
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block_with_zero_bias_simd(
                    &b_dct,
                    &b_quant.values,
                    b_zero_bias,
                    b_aq_strength,
                );
                natural_to_zigzag_into(&b_quant_coeffs, &mut b_blocks[mcu_idx]);
            }
        }

        Ok((x_blocks, y_blocks, b_blocks))
    }

    /// Builds optimized Huffman tables for XYB mode.
    ///
    /// XYB uses a single shared table for all components (luminance tables).
    /// Returns the optimized DC and AC tables.
    pub(crate) fn build_optimized_tables_xyb(
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

        // Collect frequencies from all components
        // Note: XYB MCU order is 4 X blocks, 4 Y blocks, 1 B block per MCU
        // But since all share the same table, we just iterate through them

        // In XYB mode, we have interleaved blocks per MCU:
        // [X0, X1, X2, X3, Y0, Y1, Y2, Y3, B0] per MCU
        // DC prediction carries across MCUs for each component (standard JPEG behavior)

        let mcu_count = b_blocks.len();

        // Each component maintains its own DC prediction across all MCUs
        let mut prev_dc_x: i16 = 0;
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_b: i16 = 0;

        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                let block = &x_blocks[x_start + i];
                Self::collect_block_frequencies(block, prev_dc_x, &mut dc_freq, &mut ac_freq);
                prev_dc_x = block[0];
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                let block = &y_blocks[y_start + i];
                Self::collect_block_frequencies(block, prev_dc_y, &mut dc_freq, &mut ac_freq);
                prev_dc_y = block[0];
            }

            // B block (1 per MCU)
            Self::collect_block_frequencies(
                &b_blocks[mcu_idx],
                prev_dc_b,
                &mut dc_freq,
                &mut ac_freq,
            );
            prev_dc_b = b_blocks[mcu_idx][0];
        }

        // Use jpegli's Huffman algorithm (matches C++ behavior)
        let huffman_method = crate::types::HuffmanMethod::JpegliCreateTree;

        // Generate optimized tables using selected algorithm
        let dc_table = dc_freq.generate_table_with_method(huffman_method)?;
        let ac_table = ac_freq.generate_table_with_method(huffman_method)?;

        Ok((dc_table, ac_table))
    }

    /// Encodes XYB blocks using optimized Huffman tables.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_with_tables_xyb(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
        dc_table: &crate::huffman::optimize::OptimizedTable,
        ac_table: &crate::huffman::optimize::OptimizedTable,
    ) -> Result<Vec<u8>> {
        // Estimate output size: ~100 bytes per block for typical quality
        let total_blocks = x_blocks.len() + y_blocks.len() + b_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 100);

        // Use the same optimized table for all components
        encoder.set_dc_table(0, &dc_table.table);
        encoder.set_ac_table(0, &ac_table.table);

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let mcu_count = b_blocks.len();
        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&x_blocks[x_start + i], 0, 0, 0)?;
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&y_blocks[y_start + i], 1, 0, 0)?;
            }

            // B block (1 per MCU)
            encoder.encode_block(&b_blocks[mcu_idx], 2, 0, 0)?;

            encoder.check_restart();
        }

        Ok(encoder.finish())
    }

    /// Encodes XYB blocks using standard (non-optimized) Huffman tables.
    pub(crate) fn encode_with_tables_xyb_standard(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
    ) -> Result<Vec<u8>> {
        // Estimate output size: ~100 bytes per block for typical quality
        let total_blocks = x_blocks.len() + y_blocks.len() + b_blocks.len();
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 100);

        // Use standard luminance tables for all components in XYB mode
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let mcu_count = b_blocks.len();
        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&x_blocks[x_start + i], 0, 0, 0)?;
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&y_blocks[y_start + i], 1, 0, 0)?;
            }

            // B block (1 per MCU)
            encoder.encode_block(&b_blocks[mcu_idx], 2, 0, 0)?;

            encoder.check_restart();
        }

        Ok(encoder.finish())
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
                        Self::collect_block_frequencies(block, prev_dc_x, &mut dc_freq, &mut ac_freq);
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
                        Self::collect_block_frequencies(block, prev_dc_y, &mut dc_freq, &mut ac_freq);
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

    /// Encodes scan data for XYB mode with float planes.
    ///
    /// Uses scaled XYB values (in [0, 1] range), converts to [0, 255],
    /// then level shifts by subtracting 128 before DCT.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_scan_xyb_float(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<Vec<u8>> {
        // Estimate output size: 9 blocks per MCU (4+4+1), ~100 bytes per block
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let total_blocks = mcu_cols * mcu_rows * 9;
        let mut encoder = EntropyEncoder::with_capacity(total_blocks * 100);

        // Set up Huffman tables - use luminance tables for all components in XYB mode
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        // Each MCU contains: 4 X blocks + 4 Y blocks + 1 B block = 9 blocks
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let x_block = crate::encode_simd::extract_block_xyb_simd(
                            x_plane, width, height, bx, by,
                        );
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block(&x_dct, &x_quant.values);
                        let x_zigzag = natural_to_zigzag(&x_quant_coeffs);
                        encoder.encode_block(&x_zigzag, 0, 0, 0)?;
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let y_block = crate::encode_simd::extract_block_xyb_simd(
                            y_plane, width, height, bx, by,
                        );
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                        encoder.encode_block(&y_zigzag, 1, 0, 0)?;
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = crate::encode_simd::extract_block_xyb_simd(
                    b_plane, b_width, b_height, mcu_x, mcu_y,
                );
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
                let b_zigzag = natural_to_zigzag(&b_quant_coeffs);
                encoder.encode_block(&b_zigzag, 2, 0, 0)?;

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }

    /// Extracts an 8x8 block from a float plane (scaled XYB values).
    ///
    /// Scaled XYB values are in [0, 1] range. This method:
    /// 1. Multiplies by 255 to get to [0, 255] range
    /// 2. Subtracts 128 for level shifting (DCT input is [-128, 127])
    #[allow(dead_code)]
    fn extract_block_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                let val = plane[idx];
                // XYB scaled values are in range approximately [-2.1, 7.3] after our fix
                // to use C++ jpegli's 0-255 linear RGB convention.
                // After ×255: [-536, 1862]. After -128: [-664, 1734].
                // This is correct for XYB mode - the larger range is expected.
                debug_assert!(
                    val >= -3.0 && val <= 10.0,
                    "extract_block_f32: value {} at ({}, {}) outside expected XYB range [-3, 10]",
                    val,
                    px,
                    py
                );
                // Scale from XYB range to DCT input range, then level shift by -128
                block[y * DCT_SIZE + x] = val * 255.0 - 128.0;
            }
        }

        block
    }

    /// Extracts an 8x8 block from a u8 plane with level shift.
    #[allow(dead_code)]
    fn extract_block(
        &self,
        plane: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Level shift: subtract 128
                block[y * DCT_SIZE + x] = plane[idx] as f32 - 128.0;
            }
        }

        block
    }

    /// Extracts an 8x8 block from a YCbCr f32 plane with level shift.
    /// Input values are in [0, 255] range, output is level-shifted by -128.
    #[allow(dead_code)]
    fn extract_block_ycbcr_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Level shift: subtract 128 (values are already in [0, 255])
                block[y * DCT_SIZE + x] = plane[idx] - 128.0;
            }
        }

        block
    }
}
