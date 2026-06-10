//! Trellis quantization engine: rate tables, block quantization, and the
//! per-block AQ→lambda coupling path.
//!
//! - Core block quantization ([`hybrid_quantize_block`]) — DCT f32 →
//!   trellis DP with a caller-provided [`TrellisConfig`]
//! - Encoder integration (`TrellisContext`) — applies
//!   [`AqCoupling`](super::compat::AqCoupling) per block when active

use super::compat::TrellisConfig;
use super::{RateTable, trellis_quantize_block};
use crate::encode::config::ComputedConfig;
use crate::encode::dct::forward_dct_8x8;
use crate::encode::natural_to_zigzag_into;
use crate::error::Result;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::quant::aq::AQStrengthMap;
use crate::quant::{self, QuantTable, ZeroBiasParams};

// ============================================================================
// Core hybrid functions (from hybrid/core.rs)
// ============================================================================

/// Standard rate tables for trellis rate estimation.
///
/// Trellis quantization needs Huffman code lengths to estimate bit costs.
/// Using standard JPEG tables is a reasonable approximation when
/// optimized tables aren't available yet.
pub struct StandardRateTables {
    pub luma_ac: RateTable,
    pub chroma_ac: RateTable,
    pub luma_dc: RateTable,
    pub chroma_dc: RateTable,
}

impl StandardRateTables {
    /// Create standard rate tables for trellis.
    pub fn new() -> Self {
        Self {
            luma_ac: RateTable::standard_luma_ac(),
            chroma_ac: RateTable::standard_chroma_ac(),
            luma_dc: RateTable::standard_luma_dc(),
            chroma_dc: RateTable::standard_chroma_dc(),
        }
    }
}

impl Default for StandardRateTables {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert f32 DCT coefficients to i32 for trellis quantization.
///
/// jpegli and mozjpeg use different quantization formulas:
/// - jpegli: quantized = round(DCT * 8 / quantval) (DCT at 1/64 scale)
/// - mozjpeg trellis: quantized = round(DCT / (8 * quantval))
///
/// To make trellis produce the same quantized values as jpegli:
/// - We multiply DCT by 64: trellis sees round((64*DCT) / (8*quantval)) = round(DCT*8 / quantval)
/// - This compensates for both the 1/64 DCT scaling and the trellis's 8× divisor
pub fn dct_f32_to_i32(coeffs: &[f32; DCT_BLOCK_SIZE]) -> [i32; DCT_BLOCK_SIZE] {
    let mut result = [0i32; DCT_BLOCK_SIZE];

    for i in 0..DCT_BLOCK_SIZE {
        result[i] = (coeffs[i] * 64.0).round() as i32;
    }

    result
}

/// Hybrid quantization: jpegli AQ + mozjpeg trellis.
///
/// Runs trellis quantization with a pre-configured lambda. The caller
/// (typically `TrellisContext::quantize_block`) is responsible for
/// computing any AQ-adjusted [`TrellisConfig`] before calling this function.
///
/// # Arguments
/// * `dct_coeffs` - DCT coefficients in f32 (jpegli format)
/// * `base_quant` - Base quantization table
/// * `ac_table` - Huffman table for rate estimation
/// * `config` - Trellis configuration (already AQ-adjusted by caller)
///
/// # Returns
/// Quantized coefficients ready for entropy coding
pub fn hybrid_quantize_block(
    dct_coeffs: &[f32; DCT_BLOCK_SIZE],
    base_quant: &[u16; DCT_BLOCK_SIZE],
    ac_table: &RateTable,
    config: &TrellisConfig,
) -> [i16; DCT_BLOCK_SIZE] {
    // Convert f32 DCT to i32 (with 8x scaling to match trellis's 8x quant divisor)
    let dct_i32 = dct_f32_to_i32(dct_coeffs);

    // Run trellis quantization with caller-provided lambda config
    let mut quantized = [0i16; DCT_BLOCK_SIZE];
    trellis_quantize_block(&dct_i32, &mut quantized, base_quant, ac_table, config);

    quantized
}

// ============================================================================
// Encoder integration (from encode/hybrid.rs)
// ============================================================================

// ============================================================================
// Setup Helpers
// ============================================================================

/// Get the AQ map, using custom if provided or computing from Y plane.
#[allow(dead_code)] // XYB block-based encoding path (not yet integrated)
#[inline]
pub(crate) fn get_aq_map_or_compute(
    config: &ComputedConfig,
    y_plane: &[f32],
    width: usize,
    height: usize,
    y_quant_01: u16,
) -> Result<AQStrengthMap> {
    if let Some(ref custom) = config.custom_aq_map {
        Ok(custom.clone())
    } else {
        Ok(crate::quant::aq::compute_aq_strength_map(
            y_plane, width, height, y_quant_01,
        )?)
    }
}

/// Create the trellis quantization context if enabled in config.
#[allow(dead_code)] // XYB block-based encoding path (not yet integrated)
#[inline]
pub(crate) fn create_trellis_ctx(config: &ComputedConfig) -> Option<TrellisContext> {
    config
        .trellis
        .filter(|t| t.is_enabled())
        .map(TrellisContext::new)
}

// ============================================================================
// Quantization Dispatch Helper
// ============================================================================

/// Quantize a block, dispatching to hybrid trellis or standard quantization.
///
/// This inline helper centralizes the hybrid vs non-hybrid dispatch logic.
/// When `trellis_ctx` is Some, uses trellis quantization; otherwise uses
/// standard zero-bias quantization.
#[allow(dead_code)] // XYB block-based encoding path (not yet integrated)
#[inline]
pub(crate) fn quantize_block_dispatch(
    dct: &[f32; DCT_BLOCK_SIZE],
    quant_values: &[u16; DCT_BLOCK_SIZE],
    zero_bias: &ZeroBiasParams,
    aq_strength: f32,
    is_luma: bool,
    trellis_ctx: Option<&TrellisContext>,
) -> [i16; DCT_BLOCK_SIZE] {
    if let Some(ctx) = trellis_ctx {
        ctx.quantize_block(dct, quant_values, aq_strength, is_luma)
    } else {
        quant::quantize_block_with_zero_bias_simd(dct, quant_values, zero_bias, aq_strength)
    }
}

// ============================================================================
// Hybrid Quantization Context
// ============================================================================

/// Quantization context for trellis mode.
///
/// Holds pre-built Huffman rate tables plus the [`TrellisConfig`]. When the
/// config's [`AqCoupling`](super::compat::AqCoupling) is active, the
/// per-block lambda is adjusted from the block's AQ strength; otherwise the
/// config applies verbatim (classic fixed-lambda mozjpeg behaviour).
pub(crate) struct TrellisContext {
    rate_tables: StandardRateTables,
    config: TrellisConfig,
}

impl std::fmt::Debug for TrellisContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrellisContext")
            .field("aq_coupled", &self.config.aq_coupling.is_active())
            .finish_non_exhaustive()
    }
}

impl TrellisContext {
    /// Creates a trellis quantization context.
    pub(crate) fn new(config: TrellisConfig) -> Self {
        Self {
            rate_tables: StandardRateTables::new(),
            config,
        }
    }

    /// Quantize a block using trellis quantization.
    ///
    /// # Arguments
    /// * `dct_coeffs` - DCT coefficients
    /// * `quant` - Quantization table
    /// * `aq_strength` - Per-block AQ strength (used when AQ coupling is active)
    /// * `is_luma` - True for Y component, false for Cb/Cr
    pub(crate) fn quantize_block(
        &self,
        dct_coeffs: &[f32; DCT_BLOCK_SIZE],
        quant: &[u16; DCT_BLOCK_SIZE],
        aq_strength: f32,
        is_luma: bool,
    ) -> [i16; DCT_BLOCK_SIZE] {
        let ac_table = if is_luma {
            &self.rate_tables.luma_ac
        } else {
            &self.rate_tables.chroma_ac
        };

        let trellis_config = self.block_config(aq_strength, !is_luma);
        hybrid_quantize_block(dct_coeffs, quant, ac_table, &trellis_config)
    }

    /// Effective per-block config: lambda adjusted by AQ coupling when active.
    fn block_config(&self, aq_strength: f32, is_chroma: bool) -> TrellisConfig {
        let coupling = &self.config.aq_coupling;
        if !coupling.is_active() {
            return self.config;
        }
        let adjustment = coupling.compute_adjustment(aq_strength, is_chroma);
        let scale1 = if coupling.multiplicative {
            self.config.lambda_log_scale1 * (1.0 + adjustment)
        } else {
            self.config.lambda_log_scale1 + adjustment
        };
        TrellisConfig {
            lambda_log_scale1: scale1,
            ..self.config
        }
    }

    /// Returns true if DC trellis optimization is enabled.
    pub(crate) fn is_dc_trellis_enabled(&self) -> bool {
        self.config.is_dc_enabled()
    }

    /// Returns the base trellis configuration (for DC trellis lambda parameters).
    pub(crate) fn trellis_config(&self) -> TrellisConfig {
        self.config
    }

    /// Returns the luma DC rate table for DC trellis optimization.
    pub(crate) fn luma_dc_rate_table(&self) -> &RateTable {
        &self.rate_tables.luma_dc
    }

    /// Returns the chroma DC rate table for DC trellis optimization.
    pub(crate) fn chroma_dc_rate_table(&self) -> &RateTable {
        &self.rate_tables.chroma_dc
    }
}

// ============================================================================
// XYB Block Quantization with Hybrid Trellis
// ============================================================================

/// Quantizes all XYB blocks with adaptive quantization and optional hybrid trellis.
///
/// This version uses the AQ map for per-block modulation and applies
/// hybrid trellis quantization when enabled via the TrellisContext.
///
/// For XYB mode:
/// - X and Y use luma tables (both are full-resolution "luma-like" channels)
/// - B uses chroma tables (downsampled blue channel)
#[allow(dead_code)] // XYB block-based encoding path (not yet integrated)
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
    trellis_ctx: Option<&TrellisContext>,
) -> crate::error::Result<(
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
    let mut x_blocks = crate::foundation::alloc::try_alloc_dct_blocks(num_xy_blocks, "x_blocks")?;
    let mut y_blocks = crate::foundation::alloc::try_alloc_dct_blocks(num_xy_blocks, "y_blocks")?;
    let mut b_blocks = crate::foundation::alloc::try_alloc_dct_blocks(num_b_blocks, "b_blocks")?;

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
                    let x_quant_coeffs = if let Some(ctx) = trellis_ctx {
                        ctx.quantize_block(&x_dct, &x_quant.values, aq_strength, true)
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
                    let y_quant_coeffs = if let Some(ctx) = trellis_ctx {
                        ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, true)
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
            let b_quant_coeffs = if let Some(ctx) = trellis_ctx {
                ctx.quantize_block(&b_dct, &b_quant.values, b_aq_strength, false)
            } else {
                quant::quantize_block(&b_dct, &b_quant.values)
            };
            natural_to_zigzag_into(&b_quant_coeffs, &mut b_blocks[mcu_idx]);
        }
    }

    Ok((x_blocks, y_blocks, b_blocks))
}

#[cfg(test)]
mod tests {
    use super::super::compat::AqCoupling;
    use super::*;

    #[test]
    fn coupling_off_is_inactive_and_zero() {
        let c = AqCoupling::OFF;
        assert!(!c.is_active());
        assert_eq!(c.compute_adjustment(0.5, false), 0.0);
    }

    #[test]
    fn additive_adjustment_scales_linearly() {
        let c = AqCoupling {
            scale: 4.0,
            ..AqCoupling::OFF
        };
        assert!((c.compute_adjustment(0.5, false) - 2.0).abs() < 1e-6);
        let neg = AqCoupling {
            scale: -4.0,
            ..AqCoupling::OFF
        };
        assert!((neg.compute_adjustment(0.5, false) + 2.0).abs() < 1e-6);
    }

    #[test]
    fn exponent_shapes_aq() {
        let c = AqCoupling {
            scale: 4.0,
            exponent: 2.0,
            ..AqCoupling::OFF
        };
        // 0.5^2 * 4 = 1.0
        assert!((c.compute_adjustment(0.5, false) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn threshold_gates_adjustment() {
        let c = AqCoupling {
            scale: -4.0,
            threshold: 0.3,
            ..AqCoupling::OFF
        };
        assert_eq!(c.compute_adjustment(0.2, false), 0.0);
        assert!(c.compute_adjustment(0.4, false) < 0.0);
    }

    #[test]
    fn max_adjustment_clamps_both_signs() {
        let c = AqCoupling {
            scale: -8.0,
            max_adjustment: 1.0,
            ..AqCoupling::OFF
        };
        assert_eq!(c.compute_adjustment(0.5, false), -1.0);
        let pos = AqCoupling {
            scale: 8.0,
            max_adjustment: 1.0,
            ..AqCoupling::OFF
        };
        assert_eq!(pos.compute_adjustment(0.5, false), 1.0);
    }

    #[test]
    fn chroma_mul_applies_to_chroma_only() {
        let c = AqCoupling {
            scale: 4.0,
            chroma_mul: 0.5,
            ..AqCoupling::OFF
        };
        let luma = c.compute_adjustment(0.5, false);
        let chroma = c.compute_adjustment(0.5, true);
        assert!((chroma - luma * 0.5).abs() < 1e-6);
    }

    #[test]
    fn inactive_coupling_returns_config_verbatim() {
        let cfg = TrellisConfig {
            lambda_log_scale1: 14.5,
            dc_enabled: false,
            ..TrellisConfig::default()
        };
        let ctx = TrellisContext::new(cfg);
        let eff = ctx.block_config(0.5, false);
        assert_eq!(eff, cfg);
    }

    #[test]
    fn active_coupling_adjusts_scale1_only() {
        let cfg = TrellisConfig {
            aq_coupling: AqCoupling {
                scale: -4.0,
                ..AqCoupling::OFF
            },
            ..TrellisConfig::default()
        };
        let ctx = TrellisContext::new(cfg);
        let eff = ctx.block_config(0.5, false);
        assert!((eff.lambda_log_scale1 - (14.75 - 2.0)).abs() < 1e-6);
        assert_eq!(eff.lambda_log_scale2, cfg.lambda_log_scale2);
        assert_eq!(eff.dc_enabled, cfg.dc_enabled);
        assert_eq!(eff.speed_mode, cfg.speed_mode);
    }

    #[test]
    fn multiplicative_mode() {
        let cfg = TrellisConfig {
            aq_coupling: AqCoupling {
                scale: 0.1,
                multiplicative: true,
                ..AqCoupling::OFF
            },
            ..TrellisConfig::default()
        };
        let ctx = TrellisContext::new(cfg);
        let eff = ctx.block_config(0.5, false);
        // adj = 0.5 * 0.1 = 0.05 → scale1 = 14.75 * 1.05
        assert!((eff.lambda_log_scale1 - 14.75 * 1.05).abs() < 1e-4);
    }

    #[test]
    fn test_dct_f32_to_i32() {
        let mut coeffs = [0.0f32; DCT_BLOCK_SIZE];
        coeffs[0] = 1.0;
        coeffs[1] = -0.5;
        let out = dct_f32_to_i32(&coeffs);
        assert_eq!(out[0], 64);
        assert_eq!(out[1], -32);
    }
}
