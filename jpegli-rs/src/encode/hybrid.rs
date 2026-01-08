//! Hybrid trellis quantization for the encoder.
//!
//! This module contains all hybrid trellis-specific encoding logic,
//! keeping the experimental code separate from the production encoder.
//!
//! The hybrid approach combines:
//! - jpegli's adaptive quantization (WHERE to spend bits)
//! - mozjpeg's trellis quantization (HOW to spend bits)

#![cfg(feature = "experimental-hybrid-trellis")]

use crate::consts::DCT_BLOCK_SIZE;
use crate::dct::forward_dct_8x8;
use crate::hybrid::config::HybridConfig;
use crate::hybrid::core::{hybrid_quantize_block, StandardHuffmanTables};
use crate::quant::aq::AQStrengthMap;
use crate::quant::{self, QuantTable, ZeroBiasParams};

use super::{natural_to_zigzag, natural_to_zigzag_into};

use super::EncoderConfig;

// ============================================================================
// Setup Helpers
// ============================================================================

/// Get the AQ map, using custom if provided or computing from Y plane.
#[inline]
pub(super) fn get_aq_map_or_compute(
    config: &EncoderConfig,
    y_plane: &[f32],
    width: usize,
    height: usize,
    y_quant_01: u16,
) -> crate::Result<AQStrengthMap> {
    if let Some(ref custom) = config.custom_aq_map {
        Ok(custom.clone())
    } else {
        Ok(crate::adaptive_quant::compute_aq_strength_map(
            y_plane, width, height, y_quant_01,
        )?)
    }
}

/// Create hybrid quantization context if enabled in config.
#[inline]
pub(super) fn create_hybrid_ctx(config: &EncoderConfig) -> Option<HybridQuantContext> {
    if config.hybrid_config.enabled {
        Some(HybridQuantContext::new(config.hybrid_config))
    } else {
        None
    }
}

// ============================================================================
// Quantization Dispatch Helper
// ============================================================================

/// Quantize a block, dispatching to hybrid trellis or standard quantization.
///
/// This inline helper centralizes the hybrid vs non-hybrid dispatch logic.
/// When `hybrid_ctx` is Some, uses trellis quantization; otherwise uses
/// standard zero-bias quantization.
#[inline]
pub(super) fn quantize_block_dispatch(
    dct: &[f32; DCT_BLOCK_SIZE],
    quant_values: &[u16; DCT_BLOCK_SIZE],
    zero_bias: &ZeroBiasParams,
    aq_strength: f32,
    is_luma: bool,
    hybrid_ctx: Option<&HybridQuantContext>,
) -> [i16; DCT_BLOCK_SIZE] {
    if let Some(ctx) = hybrid_ctx {
        ctx.quantize_block(dct, quant_values, aq_strength, 1.0, is_luma)
    } else {
        quant::quantize_block_with_zero_bias_simd(dct, quant_values, zero_bias, aq_strength)
    }
}

// ============================================================================
// Hybrid Quantization Context
// ============================================================================

/// Quantization context for hybrid trellis mode.
///
/// This struct holds pre-built Huffman tables and hybrid config for use
/// during hybrid quantization (jpegli AQ + mozjpeg trellis).
pub(crate) struct HybridQuantContext {
    huff_tables: StandardHuffmanTables,
    config: HybridConfig,
}

impl HybridQuantContext {
    /// Creates a new hybrid quantization context with the given config.
    pub(crate) fn new(config: HybridConfig) -> Self {
        Self {
            huff_tables: StandardHuffmanTables::new(),
            config,
        }
    }

    /// Quantize a block using hybrid AQ + trellis.
    ///
    /// # Arguments
    /// * `dct_coeffs` - DCT coefficients
    /// * `quant` - Quantization table
    /// * `aq_strength` - Per-block AQ strength
    /// * `dampen` - Quality-based AQ dampen factor (0-1)
    /// * `is_luma` - True for Y component, false for Cb/Cr
    pub(crate) fn quantize_block(
        &self,
        dct_coeffs: &[f32; DCT_BLOCK_SIZE],
        quant: &[u16; DCT_BLOCK_SIZE],
        aq_strength: f32,
        dampen: f32,
        is_luma: bool,
    ) -> [i16; DCT_BLOCK_SIZE] {
        let ac_table = if is_luma {
            &self.huff_tables.luma_ac
        } else {
            &self.huff_tables.chroma_ac
        };

        // Generate per-block trellis config based on AQ and hybrid settings
        let trellis_config = self.config.to_trellis_config(aq_strength, dampen, !is_luma);

        hybrid_quantize_block(dct_coeffs, quant, aq_strength, ac_table, &trellis_config)
    }
}

// ============================================================================
// XYB Block Quantization with Hybrid Trellis
// ============================================================================

/// Quantizes all XYB blocks with adaptive quantization and optional hybrid trellis.
///
/// This version uses the AQ map for per-block modulation and applies
/// hybrid trellis quantization when enabled via the HybridQuantContext.
///
/// For XYB mode:
/// - X and Y use luma tables (both are full-resolution "luma-like" channels)
/// - B uses chroma tables (downsampled blue channel)
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantize_all_blocks_xyb_with_aq(
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
    aq_map: &AQStrengthMap,
    hybrid_ctx: Option<&HybridQuantContext>,
) -> (
    Vec<[i16; DCT_BLOCK_SIZE]>,
    Vec<[i16; DCT_BLOCK_SIZE]>,
    Vec<[i16; DCT_BLOCK_SIZE]>,
) {
    // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
    let mcu_cols = (width + 15) / 16;
    let mcu_rows = (height + 15) / 16;
    let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
    let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

    // Pre-allocate block arrays to avoid push() overhead
    let mut x_blocks = vec![[0i16; DCT_BLOCK_SIZE]; num_xy_blocks];
    let mut y_blocks = vec![[0i16; DCT_BLOCK_SIZE]; num_xy_blocks];
    let mut b_blocks = vec![[0i16; DCT_BLOCK_SIZE]; num_b_blocks];

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

                    let x_block =
                        crate::encode_simd::extract_block_xyb_simd(x_plane, width, height, bx, by);
                    let x_dct = forward_dct_8x8(&x_block);

                    // X is luma-like in XYB, dampen=1.0
                    let x_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                        ctx.quantize_block(&x_dct, &x_quant.values, aq_strength, 1.0, true)
                    } else {
                        quant::quantize_block(&x_dct, &x_quant.values)
                    };
                    natural_to_zigzag_into(&x_quant_coeffs, &mut x_blocks[xy_base + block_offset]);
                }
            }

            // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
            for block_y in 0..2 {
                for block_x in 0..2 {
                    let bx = mcu_x * 2 + block_x;
                    let by = mcu_y * 2 + block_y;
                    let block_offset = block_y * 2 + block_x;
                    let aq_strength = aq_map.get(bx, by);

                    let y_block =
                        crate::encode_simd::extract_block_xyb_simd(y_plane, width, height, bx, by);
                    let y_dct = forward_dct_8x8(&y_block);

                    // Y is the primary luma channel in XYB, dampen=1.0
                    let y_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                        ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, 1.0, true)
                    } else {
                        quant::quantize_block(&y_dct, &y_quant.values)
                    };
                    natural_to_zigzag_into(&y_quant_coeffs, &mut y_blocks[xy_base + block_offset]);
                }
            }

            // Process 1 B block (from downsampled plane)
            // Average AQ from the 4 corresponding full-res blocks
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

            // B is chroma-like (blue channel), is_luma=false
            let b_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                ctx.quantize_block(&b_dct, &b_quant.values, b_aq_strength, 1.0, false)
            } else {
                quant::quantize_block(&b_dct, &b_quant.values)
            };
            natural_to_zigzag_into(&b_quant_coeffs, &mut b_blocks[mcu_idx]);
        }
    }

    (x_blocks, y_blocks, b_blocks)
}
