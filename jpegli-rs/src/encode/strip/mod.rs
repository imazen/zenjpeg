//! Strip-based low-memory JPEG encoding.
//!
//! This module implements a strip-based encoder that processes the image
//! in horizontal strips (MCU rows), avoiding full-plane f32 allocations.
//!
//! # Memory Model
//!
//! Traditional encoder peak memory for 12MP (4000x3000):
//! - f32 YCbCr planes: ~137 MB
//! - f32 downsampled chroma: ~23 MB
//! - i16 quantized blocks: ~36 MB
//! - Total: ~230 MB measured
//!
//! Strip-based encoder with incremental quantization:
//! - f32 strip buffers (reused): ~1 MB
//! - f32 pending iMCU DCT blocks (2x): ~0.7 MB (double-buffered)
//! - i16 quantized blocks: ~36 MB
//! - AQ accumulators: ~2.5 MB
//! - Total: ~40 MB (vs 72 MB without incremental quantization)
//!
//! # Algorithm
//!
//! For each strip of 16 rows (2 MCU rows for 4:2:0):
//! 1. Convert RGB -> YCbCr (f32 strips, reused)
//! 2. Accumulate AQ features for this strip
//! 3. Downsample chroma if needed
//! 4. DCT -> store f32 coefficients in pending buffer
//! 5. If AQ returns strengths for previous iMCU:
//!    - Quantize pending f32 -> i16
//!    - Append to final i16 storage
//! 6. Swap pending buffers
//!
//! After all strips:
//! 1. Flush last iMCU (quantize remaining pending blocks)
//! 2. Build optimized Huffman tables
//! 3. Encode from stored i16 blocks

#![allow(dead_code)]

mod convert;

use crate::encode::encoder_types::DownsamplingMethod;
use crate::error::Result;
use crate::foundation::alloc::{
    try_alloc_filled, try_alloc_zeroed_f32_tracked, try_clone_slice, try_with_capacity_tracked,
    AllocationStats,
};
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::{QuantTable, ZeroBiasParams};
use crate::types::{PixelFormat, Subsampling};

// Trellis quantization support (feature-gated)
#[cfg(feature = "experimental-hybrid-trellis")]
use crate::encode::hybrid::HybridQuantContext;
#[cfg(feature = "experimental-hybrid-trellis")]
use crate::encode::mozjpeg_compat::TrellisConfig;
#[cfg(feature = "experimental-hybrid-trellis")]
use crate::foundation::consts::JPEG_ZIGZAG_ORDER;

/// Quantization context: groups all quantization tables and bias parameters.
///
/// This struct is created once via `set_quant_tables()` and ensures all
/// quantization parameters are set together (no partial initialization).
#[derive(Debug, Clone)]
pub struct QuantContext {
    // SIMD quantization tables (for fast quantization)
    pub y_quant_simd: QuantTableSimd,
    pub cb_quant_simd: QuantTableSimd,
    pub cr_quant_simd: QuantTableSimd,
    pub y_zero_bias_simd: ZeroBiasSimd,
    pub cb_zero_bias_simd: ZeroBiasSimd,
    pub cr_zero_bias_simd: ZeroBiasSimd,

    // Original tables (for progressive encoding and table output)
    pub y_quant: QuantTable,
    pub cb_quant: QuantTable,
    pub cr_quant: QuantTable,
    pub y_zero_bias: ZeroBiasParams,
    pub cb_zero_bias: ZeroBiasParams,
    pub cr_zero_bias: ZeroBiasParams,
}

impl QuantContext {
    /// Creates a new quantization context from the component tables.
    pub fn new(
        y_quant: QuantTable,
        cb_quant: QuantTable,
        cr_quant: QuantTable,
        y_zero_bias: ZeroBiasParams,
        cb_zero_bias: ZeroBiasParams,
        cr_zero_bias: ZeroBiasParams,
    ) -> Self {
        Self {
            y_quant_simd: QuantTableSimd::from_values(&y_quant.values),
            cb_quant_simd: QuantTableSimd::from_values(&cb_quant.values),
            cr_quant_simd: QuantTableSimd::from_values(&cr_quant.values),
            y_zero_bias_simd: ZeroBiasSimd::from_params(&y_zero_bias),
            cb_zero_bias_simd: ZeroBiasSimd::from_params(&cb_zero_bias),
            cr_zero_bias_simd: ZeroBiasSimd::from_params(&cr_zero_bias),
            y_quant,
            cb_quant,
            cr_quant,
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        }
    }
}

use crate::foundation::simd_types::Block8x8f;
use wide::f32x8;

/// Wide-native block extraction: returns Block8x8f directly.
///
/// Assumes strip is properly padded (MCU-aligned) so no bounds checking needed.
/// This is the fast path for the hot encoding loop.
///
/// IMPORTANT: Applies level shift (-128) as required for JPEG DCT.
#[inline]
pub(crate) fn extract_block_from_strip_wide(
    strip: &[f32],
    bx: usize,
    local_by: usize,
    strip_width: usize,
) -> Block8x8f {
    let level_shift = f32x8::splat(128.0);
    let x_start = bx * 8;
    let y_start = local_by * 8;

    let mut rows = [f32x8::ZERO; 8];
    for dy in 0..8 {
        let row_start = (y_start + dy) * strip_width + x_start;
        // Zero-cost load from contiguous padded strip
        let row_slice: [f32; 8] = strip[row_start..row_start + 8].try_into().unwrap();
        rows[dy] = f32x8::from(row_slice) - level_shift;
    }

    Block8x8f { rows }
}

/// Performs forward DCT on a block, dispatching to archmage SIMD when available.
///
/// When the `archmage-simd` feature is enabled and a token is provided,
/// uses the token-based archmage implementation. Otherwise falls back to the
/// portable wide crate implementation.
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
#[inline]
fn forward_dct_dispatch(
    token: Option<crate::encode::mage_simd::Desktop64>,
    block: &Block8x8f,
) -> Block8x8f {
    if let Some(t) = token {
        return crate::encode::mage_simd::mage_forward_dct_8x8_wide(t, block);
    }
    crate::encode::dct::simd::forward_dct_8x8_wide(block)
}

/// Performs forward DCT on a block (non-archmage fallback).
/// The `_token` parameter is ignored but accepted for API consistency.
#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
#[inline]
fn forward_dct_dispatch(_token: (), block: &Block8x8f) -> Block8x8f {
    crate::encode::dct::simd::forward_dct_8x8_wide(block)
}

// StreamingAQ uses rolling buffers for low memory (~2.5 MB for 4K vs 33 MB).

/// Strip-based encoder for low-memory JPEG encoding.
///
/// Processes the image in horizontal strips to avoid materializing
/// full f32 planes in memory.
#[derive(Debug)]
pub struct StripProcessor {
    /// Image width in pixels (original)
    pub(super) width: usize,
    /// Image height in pixels (original)
    pub(super) height: usize,
    /// Padded width (MCU-aligned for block extraction)
    pub(super) padded_width: usize,
    /// Padded chroma width
    pub(super) padded_c_width: usize,
    /// Strip height in pixels (16 for 4:2:0, 8 for 4:4:4)
    strip_height: usize,
    /// Chroma subsampling mode
    pub(super) subsampling: Subsampling,
    /// Pixel format of input data
    pub(super) pixel_format: PixelFormat,
    /// Chroma downsampling method (Box, GammaAware, GammaAwareIterative)
    pub(super) chroma_downsampling: DownsamplingMethod,
    /// Use XYB color space instead of YCbCr
    pub(super) use_xyb: bool,

    // === Reusable strip buffers (f32) ===
    /// Y channel strip buffer
    pub(super) y_strip: Vec<f32>,
    /// Cb channel strip buffer (full res before downsample)
    pub(super) cb_strip: Vec<f32>,
    /// Cr channel strip buffer (full res before downsample)
    pub(super) cr_strip: Vec<f32>,
    /// Cb downsampled strip buffer
    pub(super) cb_down: Vec<f32>,
    /// Cr downsampled strip buffer
    pub(super) cr_down: Vec<f32>,

    // === Final quantized block storage (i16) ===
    /// Y channel quantized blocks (zigzag order)
    y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cb channel quantized blocks
    cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cr channel quantized blocks
    cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,

    // === Pending iMCU DCT blocks (wide-native, double-buffered) ===
    // These hold raw DCT coefficients until AQ strengths are available
    // Double-buffered: [current] and [previous pending quantization]
    // Using Block8x8f for zero-overhead SIMD operations
    pending_y_blocks: [Vec<Block8x8f>; 2],
    pending_cb_blocks: [Vec<Block8x8f>; 2],
    pending_cr_blocks: [Vec<Block8x8f>; 2],
    /// Index of current pending buffer (0 or 1)
    pending_current: usize,

    // === Quantization context (set via set_quant_tables) ===
    quant: Option<QuantContext>,

    // === Block dimension info (for chroma AQ mapping) ===
    y_blocks_h: usize,
    y_blocks_v: usize,
    c_blocks_h: usize,
    c_blocks_v: usize,

    // === Accumulated AQ strengths for batch finalize (debugging) ===
    all_aq_strengths: Vec<f32>,

    // === Streaming AQ state (low memory, rolling buffers) ===
    // Initialized when quant tables are set (needs y_quant_01)
    aq_state: Option<StreamingAQ>,

    // === Allocation tracking ===
    /// Tracks all allocations made by this processor
    alloc_stats: crate::foundation::alloc::AllocationStats,

    // === Optional preprocessing ===
    /// Enable overshoot deringing (on by default)
    deringing: bool,

    // === Trellis quantization (feature-gated) ===
    /// Trellis quantization context for rate-distortion optimization.
    /// When Some, uses trellis quantization instead of standard SIMD quantization.
    #[cfg(feature = "experimental-hybrid-trellis")]
    hybrid_ctx: Option<HybridQuantContext>,

    // === Archmage SIMD token (feature-gated) ===
    /// Desktop64 token for zero-dispatch SIMD operations.
    /// Obtained once at construction, reused for all blocks.
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    simd_token: Option<crate::encode::mage_simd::Desktop64>,
}

impl StripProcessor {
    /// Creates a new strip processor with default settings.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `subsampling` - Chroma subsampling mode
    /// * `pixel_format` - Input pixel format
    pub fn new(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
    ) -> Result<Self> {
        Self::with_options(
            width,
            height,
            subsampling,
            pixel_format,
            DownsamplingMethod::Box,
            0,
        )
    }

    /// Creates a new strip processor with custom chroma downsampling and restart interval.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `subsampling` - Chroma subsampling mode
    /// * `pixel_format` - Input pixel format
    /// * `chroma_downsampling` - Chroma downsampling method
    /// * `_restart_interval` - Restart interval in MCUs (0 = disabled) - reserved for future use
    pub fn with_options(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
        chroma_downsampling: DownsamplingMethod,
        _restart_interval: u16,
    ) -> Result<Self> {
        Self::with_xyb(
            width,
            height,
            subsampling,
            pixel_format,
            chroma_downsampling,
            _restart_interval,
            false, // not XYB mode
        )
    }

    /// Creates a new strip processor with XYB mode support.
    ///
    /// For XYB mode, strip_height is always 16 because the B component is 2x2 downsampled
    /// and needs 16 rows of input to produce an 8x8 DCT block.
    pub fn with_xyb(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
        chroma_downsampling: DownsamplingMethod,
        _restart_interval: u16,
        use_xyb: bool,
    ) -> Result<Self> {
        // Strip height is 16 for:
        // - 4:2:0 and 4:4:0 (2 MCU rows of chroma)
        // - XYB mode (B component is always 2x2 downsampled)
        let strip_height = match (subsampling, use_xyb) {
            (Subsampling::S420 | Subsampling::S440, _) => 16,
            (_, true) => 16, // XYB always needs 16-row strips for B component
            _ => 8,
        };

        // MCU size for padding calculation
        let mcu_size = subsampling.mcu_size();

        // Calculate padded width (MCU-aligned) for parity with full-plane encoder
        let padded_width = (width + mcu_size - 1) / mcu_size * mcu_size;

        // Chroma dimensions for strip allocation
        let (c_width, c_strip_height) = match subsampling {
            Subsampling::S420 => ((width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, strip_height / 2),
            Subsampling::S444 => (width, strip_height),
        };

        // Chroma planes are padded to multiples of 8 (block size)
        let padded_c_width = (c_width + 7) / 8 * 8;

        // Pre-allocate block storage based on image size
        let y_blocks_h = (width + 7) / 8;
        let y_blocks_v = (height + 7) / 8;
        let total_y_blocks = y_blocks_h * y_blocks_v;
        let (c_blocks_h, c_blocks_v, total_c_blocks) = match subsampling {
            Subsampling::S420 => {
                let h = (width + 15) / 16;
                let v = (height + 15) / 16;
                (h, v, h * v)
            }
            Subsampling::S422 => {
                let h = (width + 15) / 16;
                (h, y_blocks_v, h * y_blocks_v)
            }
            Subsampling::S440 => {
                let v = (height + 15) / 16;
                (y_blocks_h, v, y_blocks_h * v)
            }
            Subsampling::S444 => (y_blocks_h, y_blocks_v, total_y_blocks),
        };

        // Pending buffer capacity: one iMCU row of blocks
        // Use padded block counts for pending buffers
        let padded_y_blocks_h = padded_width / 8;
        let v_samp = match subsampling {
            Subsampling::S420 | Subsampling::S440 => 2,
            _ => 1,
        };
        let pending_y_capacity = padded_y_blocks_h * v_samp;
        let padded_c_blocks_h = padded_c_width / 8;
        let pending_c_capacity = padded_c_blocks_h;

        let is_color = !pixel_format.is_grayscale();

        // Track all allocations
        let mut alloc_stats = AllocationStats::new();

        Ok(Self {
            width,
            height,
            padded_width,
            padded_c_width,
            strip_height,
            subsampling,
            pixel_format,
            chroma_downsampling,
            use_xyb,

            // Strip buffers (sized for PADDED width for edge handling parity)
            y_strip: try_alloc_zeroed_f32_tracked(
                padded_width * strip_height,
                "y_strip",
                &mut alloc_stats,
            )?,
            cb_strip: if is_color {
                try_alloc_zeroed_f32_tracked(
                    padded_width * strip_height,
                    "cb_strip",
                    &mut alloc_stats,
                )?
            } else {
                Vec::new()
            },
            cr_strip: if is_color {
                try_alloc_zeroed_f32_tracked(
                    padded_width * strip_height,
                    "cr_strip",
                    &mut alloc_stats,
                )?
            } else {
                Vec::new()
            },
            cb_down: if is_color {
                try_alloc_zeroed_f32_tracked(
                    padded_c_width * c_strip_height,
                    "cb_down",
                    &mut alloc_stats,
                )?
            } else {
                Vec::new()
            },
            cr_down: if is_color {
                try_alloc_zeroed_f32_tracked(
                    padded_c_width * c_strip_height,
                    "cr_down",
                    &mut alloc_stats,
                )?
            } else {
                Vec::new()
            },

            // Final i16 block storage (pre-allocated capacity)
            y_blocks: try_with_capacity_tracked(total_y_blocks, "y_blocks", &mut alloc_stats)?,
            cb_blocks: if is_color {
                try_with_capacity_tracked(total_c_blocks, "cb_blocks", &mut alloc_stats)?
            } else {
                Vec::new()
            },
            cr_blocks: if is_color {
                try_with_capacity_tracked(total_c_blocks, "cr_blocks", &mut alloc_stats)?
            } else {
                Vec::new()
            },

            // Pending f32 DCT blocks (double-buffered, capacity for one iMCU row)
            pending_y_blocks: [
                try_with_capacity_tracked(
                    pending_y_capacity,
                    "pending_y_blocks[0]",
                    &mut alloc_stats,
                )?,
                try_with_capacity_tracked(
                    pending_y_capacity,
                    "pending_y_blocks[1]",
                    &mut alloc_stats,
                )?,
            ],
            pending_cb_blocks: if is_color {
                [
                    try_with_capacity_tracked(
                        pending_c_capacity,
                        "pending_cb_blocks[0]",
                        &mut alloc_stats,
                    )?,
                    try_with_capacity_tracked(
                        pending_c_capacity,
                        "pending_cb_blocks[1]",
                        &mut alloc_stats,
                    )?,
                ]
            } else {
                [Vec::new(), Vec::new()]
            },
            pending_cr_blocks: if is_color {
                [
                    try_with_capacity_tracked(
                        pending_c_capacity,
                        "pending_cr_blocks[0]",
                        &mut alloc_stats,
                    )?,
                    try_with_capacity_tracked(
                        pending_c_capacity,
                        "pending_cr_blocks[1]",
                        &mut alloc_stats,
                    )?,
                ]
            } else {
                [Vec::new(), Vec::new()]
            },
            pending_current: 0,

            // Quantization context (set via set_quant_tables)
            quant: None,

            // Block dimensions for chroma AQ mapping
            y_blocks_h,
            y_blocks_v,
            c_blocks_h,
            c_blocks_v,

            // Accumulated AQ strengths (for output)
            all_aq_strengths: try_with_capacity_tracked(
                total_y_blocks,
                "all_aq_strengths",
                &mut alloc_stats,
            )?,

            // Streaming AQ (initialized when quant tables are set)
            aq_state: None,

            // Allocation tracking
            alloc_stats,

            // Optional preprocessing (deringing on by default)
            deringing: true,

            // Trellis quantization (disabled by default)
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_ctx: None,

            // Archmage SIMD token (obtained once, reused for all blocks)
            #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
            simd_token: {
                use archmage::SimdToken;
                crate::encode::mage_simd::Desktop64::summon()
            },
        })
    }

    /// Returns allocation statistics for this processor.
    #[must_use]
    pub fn allocation_stats(&self) -> &AllocationStats {
        &self.alloc_stats
    }

    /// Returns the archmage SIMD token if available.
    ///
    /// The token is obtained once at construction and can be reused for all blocks
    /// with zero per-call dispatch overhead.
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    #[inline]
    #[must_use]
    pub fn simd_token(&self) -> Option<crate::encode::mage_simd::Desktop64> {
        self.simd_token
    }

    /// Enables or disables XYB color space mode.
    ///
    /// When enabled, strips are converted to scaled XYB instead of YCbCr.
    /// In XYB mode, the B channel is always 2x2 downsampled regardless of
    /// the subsampling setting.
    pub fn set_xyb_mode(&mut self, enable: bool) {
        self.use_xyb = enable;
    }

    /// Enables or disables overshoot deringing (on by default).
    ///
    /// When enabled, hard edges (like text on white) are smoothed by allowing
    /// values to overshoot beyond the maximum. This reduces ringing artifacts
    /// while maintaining visual fidelity (overshoots are clamped on decode).
    ///
    /// This technique was pioneered by @kornel in mozjpeg.
    pub fn set_deringing(&mut self, enable: bool) {
        self.deringing = enable;
    }

    /// Sets trellis quantization configuration.
    ///
    /// When enabled, uses trellis quantization for rate-distortion optimization
    /// instead of standard SIMD quantization. This typically produces 10-15%
    /// smaller files at the same quality.
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub fn set_trellis(&mut self, config: TrellisConfig) {
        if config.is_enabled() {
            self.hybrid_ctx = Some(HybridQuantContext::from_trellis_config(config));
        } else {
            self.hybrid_ctx = None;
        }
    }

    /// Returns whether XYB mode is enabled.
    #[must_use]
    pub fn is_xyb(&self) -> bool {
        self.use_xyb
    }

    /// Sets quantization tables and zero-bias parameters.
    ///
    /// Must be called before processing strips.
    pub fn set_quant_tables(
        &mut self,
        y_quant: QuantTable,
        cb_quant: QuantTable,
        cr_quant: QuantTable,
        y_zero_bias: ZeroBiasParams,
        cb_zero_bias: ZeroBiasParams,
        cr_zero_bias: ZeroBiasParams,
    ) -> Result<()> {
        // Initialize streaming AQ with y_quant_01 for damping calculation
        let y_quant_01 = y_quant.values[1]; // Position [0,1] in zigzag
        let v_samp = self.subsampling.v_samp_factor_luma() as usize;
        let mut aq = StreamingAQ::new(self.width, self.height, y_quant_01, v_samp)?;
        // Y strip is laid out with padded_width stride for edge handling parity
        aq.set_strip_stride(self.padded_width);
        self.aq_state = Some(aq);

        // Create quantization context with all tables
        self.quant = Some(QuantContext::new(
            y_quant,
            cb_quant,
            cr_quant,
            y_zero_bias,
            cb_zero_bias,
            cr_zero_bias,
        ));
        Ok(())
    }

    /// Returns the strip height for iteration.
    pub fn strip_height(&self) -> usize {
        self.strip_height
    }

    /// Returns the subsampling mode.
    pub fn subsampling(&self) -> Subsampling {
        self.subsampling
    }

    /// Processes one strip of RGB input data.
    ///
    /// # Arguments
    /// * `rgb_strip` - RGB pixel data for this strip
    /// * `strip_y` - Starting row index of this strip
    ///
    /// # Returns
    /// Number of blocks added during this strip
    pub fn process_strip(&mut self, rgb_strip: &[u8], strip_y: usize) -> Result<usize> {
        let actual_strip_height = self.strip_height.min(self.height - strip_y);

        // Step 1: Color convert RGB -> YCbCr or XYB into strip buffers
        if self.use_xyb {
            // XYB mode: convert RGB -> scaled XYB
            // X and Y go to y_strip and cb_strip at full res
            // B goes to cr_strip at full res, then is always 2x2 downsampled
            self.convert_strip_to_xyb(rgb_strip, actual_strip_height)?;
        } else {
            // YCbCr mode: choose optimal path based on subsampling
            let uses_gamma_aware_fused = self.chroma_downsampling.uses_gamma_aware()
                && !self.pixel_format.is_grayscale()
                && self.subsampling != Subsampling::S444;

            if uses_gamma_aware_fused {
                self.convert_strip_gamma_aware(rgb_strip, strip_y, actual_strip_height)?;
            } else {
                // Standard path: convert to YCbCr, then downsample separately
                // This matches the fullplane encoder's code path for bitexact output
                self.convert_strip_to_ycbcr(rgb_strip, actual_strip_height)?;
            }
        }

        // Step 1b: Pad strips vertically if this is a partial bottom strip
        // This is needed for vertical downsampling modes (4:2:0, 4:4:0) at image bottom
        if actual_strip_height < self.strip_height {
            self.pad_strips_vertically(actual_strip_height, self.strip_height);
        }

        // Step 2: Process AQ and check if previous iMCU strengths are ready
        let aq_strengths = if let Some(ref mut aq) = self.aq_state {
            aq.process_y_strip(&self.y_strip, strip_y, actual_strip_height)
                .map(|s| s.to_vec())
        } else {
            None
        };

        // Step 3: Downsample chroma if needed
        // - XYB mode: B channel is always 2x2 downsampled (handled in convert_strip_to_xyb)
        // - Gamma-aware fused path already wrote to cb_down/cr_down
        // - Standard path wrote to cb_strip/cr_strip at full res, needs downsampling
        let downsample_height = if actual_strip_height < self.strip_height {
            self.strip_height
        } else {
            actual_strip_height
        };
        let uses_gamma_aware_fused = !self.use_xyb
            && self.chroma_downsampling.uses_gamma_aware()
            && !self.pixel_format.is_grayscale()
            && self.subsampling != Subsampling::S444;
        if !self.pixel_format.is_grayscale() && !uses_gamma_aware_fused && !self.use_xyb {
            self.downsample_chroma_strip(downsample_height)?;
        }

        // Step 4: If we got AQ strengths, quantize the previous pending iMCU
        // This is the key optimization: quantize to i16 immediately instead of storing f32
        if let Some(strengths) = aq_strengths {
            let prev_buffer = 1 - self.pending_current;
            self.quantize_pending_imcu(prev_buffer, &strengths);
            // Clear the previous buffer for reuse
            self.pending_y_blocks[prev_buffer].clear();
            self.pending_cb_blocks[prev_buffer].clear();
            self.pending_cr_blocks[prev_buffer].clear();
        }

        // Step 5: Compute DCT for blocks in this strip into the current pending buffer
        // Use downsample_height (full strip if padded) to include padding in DCT
        let blocks_added = self.dct_strip_blocks_to_pending(strip_y, downsample_height)?;

        // Step 6: Swap pending buffers when iMCU completes
        // (The swap happens via pending_current tracking in dct_strip_blocks_to_pending)

        Ok(blocks_added)
    }

    /// Processes one strip of YCbCr f32 input data.
    ///
    /// This bypasses the RGB->YCbCr conversion, accepting YCbCr data directly.
    /// Values should be in centered range [-128, 127] (will be level-shifted to [0, 255]).
    ///
    /// # Arguments
    /// * `y_row` - Y plane data for this row (width floats)
    /// * `cb_row` - Cb plane data for this row (width floats, full resolution)
    /// * `cr_row` - Cr plane data for this row (width floats, full resolution)
    /// * `strip_y` - Starting row index of this strip
    ///
    /// # Returns
    /// Number of blocks added during this strip
    ///
    /// # Errors
    /// Returns an error if XYB mode is enabled (use RGB input for XYB).
    pub fn process_strip_ycbcr_f32(
        &mut self,
        y_row: &[f32],
        cb_row: &[f32],
        cr_row: &[f32],
        strip_y: usize,
    ) -> Result<usize> {
        // XYB mode requires RGB input for conversion
        if self.use_xyb {
            return Err(crate::error::Error::unsupported_feature(
                "YCbCr input not supported for XYB mode",
            ));
        }

        let actual_strip_height = self.strip_height.min(self.height - strip_y);

        // Step 1: Copy YCbCr data to strip buffers with level shift
        // Convert from centered [-128, 127] to JPEG range [0, 255]
        self.copy_ycbcr_to_strips(y_row, cb_row, cr_row, actual_strip_height)?;

        // Step 1b: Pad strips vertically if this is a partial bottom strip
        if actual_strip_height < self.strip_height {
            self.pad_strips_vertically(actual_strip_height, self.strip_height);
        }

        // Step 2: Process AQ and check if previous iMCU strengths are ready
        let aq_strengths = if let Some(ref mut aq) = self.aq_state {
            aq.process_y_strip(&self.y_strip, strip_y, actual_strip_height)
                .map(|s| s.to_vec())
        } else {
            None
        };

        // Step 3: Downsample chroma if needed
        let downsample_height = if actual_strip_height < self.strip_height {
            self.strip_height
        } else {
            actual_strip_height
        };
        if !self.pixel_format.is_grayscale() {
            self.downsample_chroma_strip(downsample_height)?;
        }

        // Step 4: If we got AQ strengths, quantize the previous pending iMCU
        if let Some(strengths) = aq_strengths {
            let prev_buffer = 1 - self.pending_current;
            self.quantize_pending_imcu(prev_buffer, &strengths);
            self.pending_y_blocks[prev_buffer].clear();
            self.pending_cb_blocks[prev_buffer].clear();
            self.pending_cr_blocks[prev_buffer].clear();
        }

        // Step 5: Compute DCT for blocks in this strip into the current pending buffer
        let blocks_added = self.dct_strip_blocks_to_pending(strip_y, downsample_height)?;

        Ok(blocks_added)
    }

    /// Processes one strip of pre-downsampled YCbCr f32 input data.
    ///
    /// This accepts chroma data that is already downsampled according to the
    /// subsampling mode. Skips the internal chroma downsampling step.
    ///
    /// # Arguments
    /// * `y_row` - Y plane data for this row (width floats)
    /// * `cb_row` - Cb plane data (chroma_width floats, already downsampled)
    /// * `cr_row` - Cr plane data (chroma_width floats, already downsampled)
    /// * `strip_y` - Starting row index of this strip
    ///
    /// # Value Range
    /// Values should be in centered range [-128, 127].
    ///
    /// # Chroma Dimensions
    /// - 4:4:4: cb/cr at full width
    /// - 4:2:2: cb/cr at width/2
    /// - 4:2:0: cb/cr at width/2 (and height/2, but handled row-by-row)
    pub fn process_strip_ycbcr_f32_subsampled(
        &mut self,
        y_row: &[f32],
        cb_row: &[f32],
        cr_row: &[f32],
        strip_y: usize,
    ) -> Result<usize> {
        // XYB mode requires RGB input for conversion
        if self.use_xyb {
            return Err(crate::error::Error::unsupported_feature(
                "YCbCr input not supported for XYB mode",
            ));
        }

        let actual_strip_height = self.strip_height.min(self.height - strip_y);

        // Step 1: Copy Y with level shift, copy chroma directly to downsampled buffers
        self.copy_ycbcr_subsampled_to_strips(y_row, cb_row, cr_row, actual_strip_height)?;

        // Step 1b: Pad strips vertically if this is a partial bottom strip
        if actual_strip_height < self.strip_height {
            self.pad_strips_vertically(actual_strip_height, self.strip_height);
            // Also pad chroma downsampled buffers
            self.pad_chroma_down_vertically(actual_strip_height)?;
        }

        // Step 2: Process AQ
        let aq_strengths = if let Some(ref mut aq) = self.aq_state {
            aq.process_y_strip(&self.y_strip, strip_y, actual_strip_height)
                .map(|s| s.to_vec())
        } else {
            None
        };

        // Step 3: Skip chroma downsampling - already done by caller

        // Step 4: Quantize previous pending iMCU if strengths are ready
        if let Some(strengths) = aq_strengths {
            let prev_buffer = 1 - self.pending_current;
            self.quantize_pending_imcu(prev_buffer, &strengths);
            self.pending_y_blocks[prev_buffer].clear();
            self.pending_cb_blocks[prev_buffer].clear();
            self.pending_cr_blocks[prev_buffer].clear();
        }

        // Step 5: Compute DCT
        let downsample_height = if actual_strip_height < self.strip_height {
            self.strip_height
        } else {
            actual_strip_height
        };
        let blocks_added = self.dct_strip_blocks_to_pending(strip_y, downsample_height)?;

        Ok(blocks_added)
    }

    /// Computes DCT for blocks in the current strip and stores in pending buffer.
    /// This allows quantization to happen incrementally when AQ strengths become available.
    fn dct_strip_blocks_to_pending(
        &mut self,
        strip_y: usize,
        strip_height: usize,
    ) -> Result<usize> {
        // Use original dimensions for block counts (parity with full-plane encoder)
        let blocks_w = (self.width + 7) / 8;
        let strip_blocks_h = (strip_height + 7) / 8;
        let start_block_y = strip_y / 8;
        let height = self.height;
        let pending_idx = self.pending_current;

        // Y strip is now in padded layout (padded_width pixels per row)
        let padded_width = self.padded_width;

        // Extract SIMD token once for all blocks in this strip
        #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
        let simd_token = self.simd_token;
        #[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
        let simd_token = ();

        // y_strip is in padded layout, so use padded_width for sizing
        let y_size = strip_height * padded_width;

        // Calculate actual block rows to process (may be limited by image height)
        let max_block_y = (height + 7) / 8;
        let actual_strip_blocks_h = strip_blocks_h.min(max_block_y.saturating_sub(start_block_y));
        let blocks_added = actual_strip_blocks_h * blocks_w;

        // Compute DCT for Y blocks into pending buffer
        #[cfg(feature = "parallel")]
        {
            super::parallel::parallel_dct_y_blocks(
                &self.y_strip[..y_size],
                blocks_w,
                actual_strip_blocks_h,
                padded_width,
                &mut self.pending_y_blocks[pending_idx],
            );
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Pre-allocate and write directly to avoid push overhead
            let start_idx = self.pending_y_blocks[pending_idx].len();
            self.pending_y_blocks[pending_idx]
                .resize(start_idx + blocks_added, Block8x8f::default());
            let output = &mut self.pending_y_blocks[pending_idx][start_idx..];

            // Get DC quant value for deringing (if enabled)
            let y_dc_quant = self
                .quant
                .as_ref()
                .map(|q| q.y_quant.values[0])
                .unwrap_or(16);

            let mut idx = 0;
            for local_by in 0..actual_strip_blocks_h {
                for bx in 0..blocks_w {
                    // Extract 8x8 block from Y strip and DCT (wide-native path)
                    let mut block = extract_block_from_strip_wide(
                        &self.y_strip[..y_size],
                        bx,
                        local_by,
                        padded_width,
                    );

                    // Apply deringing if enabled (on by default)
                    if self.deringing {
                        super::deringing::preprocess_deringing_block(&mut block, y_dc_quant);
                    }

                    output[idx] = forward_dct_dispatch(simd_token, &block);
                    idx += 1;
                }
            }
        }

        // Compute DCT for Cb/Cr blocks (if color)
        if !self.pixel_format.is_grayscale() {
            let width = self.width;

            if self.use_xyb {
                // XYB mode: Y component is full resolution, B is 2x2 downsampled
                // - Y (cb_strip): full res, same block dimensions as X (y_strip)
                // - B (cr_down): 2x2 downsampled

                // Process Y component (full res from cb_strip which is now in padded layout)
                let y_size = strip_height * padded_width;

                // Calculate actual Cb block count (same as Y - uses actual_strip_blocks_h)
                let cb_blocks_total = actual_strip_blocks_h * blocks_w;
                let cb_start = self.pending_cb_blocks[pending_idx].len();
                self.pending_cb_blocks[pending_idx]
                    .resize(cb_start + cb_blocks_total, Block8x8f::default());

                let mut cb_idx = 0;
                for local_by in 0..strip_blocks_h {
                    let global_by = start_block_y + local_by;
                    if global_by >= (height + 7) / 8 {
                        break;
                    }
                    for bx in 0..blocks_w {
                        let cb_block = extract_block_from_strip_wide(
                            &self.cb_strip[..y_size],
                            bx,
                            local_by,
                            padded_width,
                        );
                        self.pending_cb_blocks[pending_idx][cb_start + cb_idx] =
                            forward_dct_dispatch(simd_token, &cb_block);
                        cb_idx += 1;
                    }
                }

                // Process B component (2x2 downsampled from cr_down)
                // For XYB, B is always 2x2 downsampled, so use c_width/c_height based on that
                let b_width = (width + 1) / 2;
                let b_strip_height = (strip_height + 1) / 2;
                let b_blocks_w = (b_width + 7) / 8;
                let b_strip_blocks_h = (b_strip_height + 7) / 8;
                let b_blocks_total = b_blocks_w * b_strip_blocks_h;
                let padded_c_width = self.padded_c_width;

                // cr_down is sized for c_strip_height rows based on the subsampling mode,
                // but for XYB we compute b_strip_height separately. Use the full cr_down size.
                let c_strip_height_for_buf = match self.subsampling {
                    Subsampling::S420 | Subsampling::S440 => strip_height / 2,
                    _ => strip_height,
                };
                let c_size = c_strip_height_for_buf * padded_c_width;

                let cr_start = self.pending_cr_blocks[pending_idx].len();
                self.pending_cr_blocks[pending_idx]
                    .resize(cr_start + b_blocks_total, Block8x8f::default());

                let mut cr_idx = 0;
                for local_by in 0..b_strip_blocks_h {
                    for bx in 0..b_blocks_w {
                        let cr_block = extract_block_from_strip_wide(
                            &self.cr_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        self.pending_cr_blocks[pending_idx][cr_start + cr_idx] =
                            forward_dct_dispatch(simd_token, &cr_block);
                        cr_idx += 1;
                    }
                }
            } else {
                // YCbCr mode: standard chroma dimensions based on subsampling
                let (c_width, c_strip_height) = match self.subsampling {
                    Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
                    Subsampling::S422 => ((width + 1) / 2, strip_height),
                    Subsampling::S440 => (width, (strip_height + 1) / 2),
                    Subsampling::S444 => (width, strip_height),
                };

                let c_blocks_w = (c_width + 7) / 8;
                let c_strip_blocks_h = (c_strip_height + 7) / 8;
                let c_blocks_total = c_blocks_w * c_strip_blocks_h;

                // cb_down/cr_down are in padded layout (padded_c_width pixels per row)
                let padded_c_width = self.padded_c_width;
                let c_size = c_strip_height * padded_c_width;

                // Pre-allocate chroma buffers
                let cb_start = self.pending_cb_blocks[pending_idx].len();
                let cr_start = self.pending_cr_blocks[pending_idx].len();
                self.pending_cb_blocks[pending_idx]
                    .resize(cb_start + c_blocks_total, Block8x8f::default());
                self.pending_cr_blocks[pending_idx]
                    .resize(cr_start + c_blocks_total, Block8x8f::default());

                let mut idx = 0;
                for local_by in 0..c_strip_blocks_h {
                    for bx in 0..c_blocks_w {
                        // Cb block - DCT only (wide-native path)
                        let cb_block = extract_block_from_strip_wide(
                            &self.cb_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        self.pending_cb_blocks[pending_idx][cb_start + idx] =
                            forward_dct_dispatch(simd_token, &cb_block);

                        // Cr block - DCT only (wide-native path)
                        let cr_block = extract_block_from_strip_wide(
                            &self.cr_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        self.pending_cr_blocks[pending_idx][cr_start + idx] =
                            forward_dct_dispatch(simd_token, &cr_block);
                        idx += 1;
                    }
                }
            }
        }

        // Swap pending buffer for next iMCU
        self.pending_current = 1 - self.pending_current;

        Ok(blocks_added)
    }

    /// Quantizes pending f32 DCT blocks to i16 using AQ strengths.
    ///
    /// This is the key memory optimization: quantize incrementally as soon as
    /// AQ strengths become available, rather than storing all f32 blocks.
    ///
    /// When trellis quantization is enabled (via `set_trellis()`), uses
    /// rate-distortion optimization for better compression. Otherwise uses
    /// fast SIMD quantization.
    fn quantize_pending_imcu(&mut self, buffer_idx: usize, aq_strengths: &[f32]) {
        // Get quantization context (must be set before processing)
        let quant = self.quant.clone().expect("quant context not set");

        // Check if we have trellis context for R-D optimization
        #[cfg(feature = "experimental-hybrid-trellis")]
        let use_trellis = self.hybrid_ctx.is_some();
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let use_trellis = false;

        // Quantize Y blocks (vectors pre-allocated at construction)
        for (i, dct) in self.pending_y_blocks[buffer_idx].iter().enumerate() {
            // Use get() with fallback to avoid branch on common path
            let aq_strength = aq_strengths.get(i).copied().unwrap_or(0.08);

            let zigzag = if use_trellis {
                #[cfg(feature = "experimental-hybrid-trellis")]
                {
                    // Trellis path: convert to array, quantize with R-D, apply zigzag
                    let dct_arr = dct.to_array();
                    let natural = self.hybrid_ctx.as_ref().unwrap().quantize_block(
                        &dct_arr,
                        &quant.y_quant.values,
                        aq_strength,
                        1.0,  // dampen
                        true, // is_luma
                    );
                    // Apply zigzag reordering
                    let mut result = [0i16; DCT_BLOCK_SIZE];
                    for j in 0..DCT_BLOCK_SIZE {
                        result[JPEG_ZIGZAG_ORDER[j] as usize] = natural[j];
                    }
                    result
                }
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                unreachable!()
            } else {
                // Fast SIMD path: fused quantization + zigzag reorder
                quant.y_quant_simd.quantize_with_zero_bias_zigzag(
                    dct,
                    &quant.y_zero_bias_simd,
                    aq_strength,
                )
            };

            self.y_blocks.push(zigzag);
            self.all_aq_strengths.push(aq_strength);
        }

        // Quantize Cb/Cr blocks (always present when quant is set)
        {
            let y_blocks_h = self.y_blocks_h;
            let y_blocks_v = self.y_blocks_v;
            let c_blocks_h = self.c_blocks_h;
            let c_blocks_v = self.c_blocks_v;

            // Compute global chroma by from how many chroma blocks we've already processed
            // For 4:2:0, each iMCU has 1 chroma block row
            let global_chroma_by = self.cb_blocks.len() / c_blocks_h.max(1);

            // Quantize Cb blocks (vectors pre-allocated at construction)
            for (i, dct) in self.pending_cb_blocks[buffer_idx].iter().enumerate() {
                let bx = i % c_blocks_h.max(1);
                let local_by = i / c_blocks_h.max(1);
                // Compute global Y position for this chroma block
                let y_bx = (bx * y_blocks_h) / c_blocks_h.max(1);
                let chroma_by = global_chroma_by + local_by;
                let y_by = (chroma_by * y_blocks_v) / c_blocks_v.max(1);
                // Use global AQ index
                let global_aq_idx = y_by * y_blocks_h + y_bx.min(y_blocks_h.saturating_sub(1));
                let aq_strength = if global_aq_idx < self.all_aq_strengths.len() {
                    self.all_aq_strengths[global_aq_idx]
                } else {
                    0.08 // C++ mean fallback
                };

                let zigzag = if use_trellis {
                    #[cfg(feature = "experimental-hybrid-trellis")]
                    {
                        let dct_arr = dct.to_array();
                        let natural = self.hybrid_ctx.as_ref().unwrap().quantize_block(
                            &dct_arr,
                            &quant.cb_quant.values,
                            aq_strength,
                            1.0,
                            false, // is_chroma
                        );
                        let mut result = [0i16; DCT_BLOCK_SIZE];
                        for j in 0..DCT_BLOCK_SIZE {
                            result[JPEG_ZIGZAG_ORDER[j] as usize] = natural[j];
                        }
                        result
                    }
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    unreachable!()
                } else {
                    quant.cb_quant_simd.quantize_with_zero_bias_zigzag(
                        dct,
                        &quant.cb_zero_bias_simd,
                        aq_strength,
                    )
                };
                self.cb_blocks.push(zigzag);
            }

            // Quantize Cr blocks
            let global_chroma_by_cr = self.cr_blocks.len() / c_blocks_h.max(1);
            for (i, dct) in self.pending_cr_blocks[buffer_idx].iter().enumerate() {
                let bx = i % c_blocks_h.max(1);
                let local_by = i / c_blocks_h.max(1);
                // Compute global Y position for this chroma block
                let y_bx = (bx * y_blocks_h) / c_blocks_h.max(1);
                let chroma_by = global_chroma_by_cr + local_by;
                let y_by = (chroma_by * y_blocks_v) / c_blocks_v.max(1);
                // Use global AQ index
                let global_aq_idx = y_by * y_blocks_h + y_bx.min(y_blocks_h.saturating_sub(1));
                let aq_strength = if global_aq_idx < self.all_aq_strengths.len() {
                    self.all_aq_strengths[global_aq_idx]
                } else {
                    0.08 // C++ mean fallback
                };

                let zigzag = if use_trellis {
                    #[cfg(feature = "experimental-hybrid-trellis")]
                    {
                        let dct_arr = dct.to_array();
                        let natural = self.hybrid_ctx.as_ref().unwrap().quantize_block(
                            &dct_arr,
                            &quant.cr_quant.values,
                            aq_strength,
                            1.0,
                            false, // is_chroma
                        );
                        let mut result = [0i16; DCT_BLOCK_SIZE];
                        for j in 0..DCT_BLOCK_SIZE {
                            result[JPEG_ZIGZAG_ORDER[j] as usize] = natural[j];
                        }
                        result
                    }
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    unreachable!()
                } else {
                    quant.cr_quant_simd.quantize_with_zero_bias_zigzag(
                        dct,
                        &quant.cr_zero_bias_simd,
                        aq_strength,
                    )
                };
                self.cr_blocks.push(zigzag);
            }
        }
    }

    /// Finalizes encoding after all strips have been processed.
    ///
    /// With incremental quantization, most blocks are already quantized.
    /// This method only handles the last pending iMCU.
    pub fn finalize(mut self) -> Result<StripProcessorOutput> {
        // Flush AQ to get the last iMCU's strengths
        if let Some(ref mut aq) = self.aq_state {
            if let Some(last_aq) = aq.flush() {
                // Quantize the last pending iMCU
                let last_strengths = try_clone_slice(last_aq, "last_aq_strengths")?;
                let prev_buffer = 1 - self.pending_current;
                if !self.pending_y_blocks[prev_buffer].is_empty() {
                    self.quantize_pending_imcu(prev_buffer, &last_strengths);
                }
            }
        }

        // Also quantize any blocks remaining in the current pending buffer
        // (for edge cases where we have blocks but no AQ was returned)
        let current_buffer = self.pending_current;
        if !self.pending_y_blocks[current_buffer].is_empty() {
            // Use default AQ strength for remaining blocks
            let default_aq = try_alloc_filled(
                self.pending_y_blocks[current_buffer].len(),
                0.08f32,
                "default_aq_strengths",
            )?;
            self.quantize_pending_imcu(current_buffer, &default_aq);
        }

        Ok(StripProcessorOutput {
            y_blocks: self.y_blocks,
            cb_blocks: self.cb_blocks,
            cr_blocks: self.cr_blocks,
            aq_strengths: self.all_aq_strengths,
            alloc_stats: self.alloc_stats,
        })
    }
}

/// Output from strip processing.
#[derive(Debug)]
pub struct StripProcessorOutput {
    /// Y channel quantized blocks
    pub y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cb channel quantized blocks
    pub cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cr channel quantized blocks
    pub cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Per-block AQ strengths (for optional re-quantization)
    pub aq_strengths: Vec<f32>,
    /// Allocation statistics from the encoding process
    pub alloc_stats: AllocationStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_processor_creation() {
        let processor = StripProcessor::new(1920, 1080, Subsampling::S420, PixelFormat::Rgb);
        assert!(processor.is_ok());
        let processor = processor.unwrap();
        assert_eq!(processor.strip_height(), 16); // 4:2:0 uses 16-row strips
    }

    #[test]
    fn test_strip_processor_444_strip_height() {
        let processor = StripProcessor::new(1920, 1080, Subsampling::S444, PixelFormat::Rgb);
        assert!(processor.is_ok());
        let processor = processor.unwrap();
        assert_eq!(processor.strip_height(), 8); // 4:4:4 uses 8-row strips
    }

    // StreamingAQ tests are in quant/aq/streaming.rs

    /// Test that strip encoder produces valid output for partial MCU heights.
    /// This test checks heights 56-72 which cross the 64-pixel boundary (4 MCU rows).
    /// Validates that decoded output is close to original (PSNR-based check).
    #[test]
    fn test_strip_partial_mcu_heights() {
        use crate::encode::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;

        let width = 64usize;
        let mut results = Vec::new();

        for height in 56..=72 {
            let mut rgb = vec![0u8; width * height * 3];
            // Use a structured pattern with sharp edges to exercise edge handling
            // This pattern has:
            // - Vertical color bands at 8-pixel block boundaries
            // - Horizontal bands at varying positions
            // - Sharp transitions that expose edge padding bugs
            // Unlike smooth gradients, this exercises edge/boundary handling
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 3;
                    // Create color blocks with sharp edges at 8-pixel boundaries
                    let block_x = x / 8;
                    let block_y = y / 8;

                    // Distinct colors for each block - creates sharp edges
                    // Colors chosen to be compressible but distinct
                    let colors: [(u8, u8, u8); 8] = [
                        (200, 100, 80),  // coral
                        (80, 180, 100),  // green
                        (100, 80, 180),  // purple
                        (180, 180, 80),  // yellow
                        (80, 150, 180),  // cyan
                        (180, 80, 150),  // magenta
                        (140, 140, 140), // gray
                        (220, 180, 140), // tan
                    ];

                    // Select color based on block position (varies both x and y)
                    let color_idx = (block_x + block_y * 3) % 8;
                    let (r, g, b) = colors[color_idx];

                    // Add smooth gradient within each block to avoid banding
                    // but keep sharp edges at boundaries
                    let intra_x = (x % 8) as i16;
                    let intra_y = (y % 8) as i16;
                    let grad = ((intra_x + intra_y) / 2) as u8; // 0-7 range

                    rgb[idx] = r.saturating_add(grad);
                    rgb[idx + 1] = g.saturating_sub(grad / 2);
                    rgb[idx + 2] = b.saturating_add(grad / 2);
                }
            }

            // Encode with strip encoder (current default)
            let config = EncoderConfig::ycbcr(85.0, crate::encode::ChromaSubsampling::Quarter)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
                .expect("encoder creation failed");
            enc.push_packed(&rgb, Unstoppable).expect("push failed");
            let jpeg_strip = enc.finish().expect("strip encode failed");

            // Decode and verify
            let decoded_strip = crate::decode::Decoder::new()
                .decode(&jpeg_strip)
                .expect("strip decode failed");

            // Check dimensions match
            assert_eq!(
                decoded_strip.width, width as u32,
                "strip width mismatch at height {}",
                height
            );
            assert_eq!(
                decoded_strip.height, height as u32,
                "strip height mismatch at height {}",
                height
            );

            // Check decoded data size matches
            let expected_size = width * height * 3;
            assert_eq!(
                decoded_strip.data.len(),
                expected_size,
                "strip data size mismatch at height {}: got {} expected {}",
                height,
                decoded_strip.data.len(),
                expected_size
            );

            // Compute mean squared error between original and decoded
            // Note: per-pixel checks removed because sharp synthetic edges create
            // larger compression artifacts than natural images. PSNR is the real
            // quality measure. For real-world parity testing, see
            // tests/strip_edge_cpp_comparison.rs which uses corpus images.
            let mut sum_sq_err: u64 = 0;
            let mut max_diff: i32 = 0;
            for (&orig, &dec) in rgb.iter().zip(decoded_strip.data.iter()) {
                let diff = (orig as i32 - dec as i32).abs();
                sum_sq_err += (diff as u64) * (diff as u64);
                if diff > max_diff {
                    max_diff = diff;
                }
            }

            let mse = sum_sq_err as f64 / expected_size as f64;
            let psnr = if mse > 0.0 {
                10.0 * (255.0 * 255.0 / mse).log10()
            } else {
                100.0
            };

            results.push((height, max_diff, jpeg_strip.len(), psnr));
        }

        // Print summary
        println!("\nHeight  MaxDiff  Size     PSNR");
        for (height, max_diff, size, psnr) in &results {
            let marker = if *psnr < 25.0 { " <-- LOW" } else { "" };
            println!(
                "{:>6} {:>8} {:>8} {:>8.2}{}",
                height, max_diff, size, psnr, marker
            );
        }

        // Quality at 85 should produce PSNR > 25 dB for sharp block patterns
        // (Natural images would be >30 dB, but sharp synthetic edges are harder)
        for (height, _, _, psnr) in &results {
            assert!(
                *psnr > 25.0,
                "strip encoder PSNR {} at height {} is too low (expected > 25)",
                psnr,
                height
            );
        }
    }
}
