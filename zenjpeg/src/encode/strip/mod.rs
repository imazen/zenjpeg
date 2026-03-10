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
use crate::encode::layout::LayoutParams;
use crate::error::Result;
use crate::foundation::alloc::{
    EncodeStats, try_alloc_filled, try_alloc_zeroed_f32_tracked, try_with_capacity_tracked,
};
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::foundation::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::{QuantTable, ZeroBiasParams};
use crate::types::{PixelFormat, Subsampling};

// Trellis quantization support
#[cfg(feature = "trellis")]
use crate::encode::trellis::HybridQuantContext;
#[cfg(feature = "trellis")]
use crate::encode::trellis::TrellisConfig;
#[cfg(feature = "trellis")]
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
    /// Creates a default quantization context for tests (standard JPEG tables at q75).
    #[cfg(test)]
    fn default_for_tests() -> Self {
        let qt = QuantTable {
            values: [16; 64],
            precision: 0,
        };
        let zb = ZeroBiasParams::for_ycbcr(1.0, 0);
        Self::new(qt.clone(), qt.clone(), qt, zb.clone(), zb.clone(), zb)
    }

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
///
/// # Safety invariant
/// `strip` must have at least `(local_by * 8 + 7) * strip_width + bx * 8 + 8` elements.
/// This is guaranteed by MCU-aligned padding in `LayoutParams::padded_width`.
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

    let last_row_end = (y_start + 7) * strip_width + x_start + 8;
    debug_assert!(
        last_row_end <= strip.len(),
        "extract_block_from_strip_wide: block ({bx}, {local_by}) out of bounds \
         (need {last_row_end}, have {}; strip_width={strip_width})",
        strip.len(),
    );

    let mut rows = [f32x8::ZERO; 8];
    for dy in 0..8 {
        let row_start = (y_start + dy) * strip_width + x_start;
        let src = &strip[row_start..row_start + 8];
        // SAFETY: src.len() == 8 is guaranteed by the slice range.
        // Using copy instead of try_into to avoid Result overhead.
        let mut arr = [0.0f32; 8];
        arr.copy_from_slice(src);
        rows[dy] = f32x8::from(arr) - level_shift;
    }

    Block8x8f { rows }
}

/// Performs forward DCT on a block, dispatching to archmage SIMD when available.
///
/// When an archmage token is available (x86_64 with AVX2),
/// uses the token-based archmage implementation. Otherwise falls back to the
/// portable wide crate implementation.
#[cfg(target_arch = "x86_64")]
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
#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn forward_dct_dispatch(_token: (), block: &Block8x8f) -> Block8x8f {
    crate::encode::dct::simd::forward_dct_8x8_wide(block)
}

// StreamingAQ uses rolling buffers for low memory (~2.5 MB for 4K vs 33 MB).

/// Double-buffered pending DCT blocks for iMCU overlap.
///
/// Holds raw f32 DCT coefficients until AQ strengths are available.
/// Two buffers: current (being filled) and previous (awaiting quantization).
/// Using `bool` for the index since there are exactly 2 states.
#[derive(Debug)]
struct PendingBuffers {
    y: [Vec<Block8x8f>; 2],
    cb: [Vec<Block8x8f>; 2],
    cr: [Vec<Block8x8f>; 2],
    current: bool,
}

impl PendingBuffers {
    /// Index of the buffer currently being filled.
    #[inline]
    fn current_idx(&self) -> usize {
        self.current as usize
    }

    /// Index of the previous buffer (awaiting quantization).
    #[inline]
    fn prev_idx(&self) -> usize {
        (!self.current) as usize
    }

    /// Swap current and previous buffers.
    #[inline]
    fn swap(&mut self) {
        self.current = !self.current;
    }

    /// Clear the previous buffer for reuse.
    fn clear_prev(&mut self) {
        let idx = self.prev_idx();
        self.y[idx].clear();
        self.cb[idx].clear();
        self.cr[idx].clear();
    }

    /// Current Y buffer (being filled).
    #[inline]
    fn current_y(&self) -> &Vec<Block8x8f> {
        &self.y[self.current_idx()]
    }

    /// Current Y buffer (mutable, being filled).
    #[inline]
    fn current_y_mut(&mut self) -> &mut Vec<Block8x8f> {
        let idx = self.current_idx();
        &mut self.y[idx]
    }

    /// Current Cb buffer (mutable, being filled).
    #[inline]
    fn current_cb_mut(&mut self) -> &mut Vec<Block8x8f> {
        let idx = self.current_idx();
        &mut self.cb[idx]
    }

    /// Current Cr buffer (mutable, being filled).
    #[inline]
    fn current_cr_mut(&mut self) -> &mut Vec<Block8x8f> {
        let idx = self.current_idx();
        &mut self.cr[idx]
    }

    /// Previous Y buffer (awaiting quantization).
    #[inline]
    fn prev_y(&self) -> &Vec<Block8x8f> {
        &self.y[self.prev_idx()]
    }

    /// Previous Cb buffer (awaiting quantization).
    #[inline]
    fn prev_cb(&self) -> &Vec<Block8x8f> {
        &self.cb[self.prev_idx()]
    }

    /// Previous Cr buffer (awaiting quantization).
    #[inline]
    fn prev_cr(&self) -> &Vec<Block8x8f> {
        &self.cr[self.prev_idx()]
    }
}

/// Quantizes a chroma component's pending DCT blocks to i16.
///
/// Shared logic for both Cb and Cr (and XYB B-channel) quantization.
/// Maps chroma block positions to the corresponding Y-block AQ strength.
///
/// When the `trellis` feature is enabled, accepts optional `HybridQuantContext`
/// for rate-distortion optimized quantization. Without `trellis`, always uses
/// the fast SIMD quantization path.
#[allow(clippy::too_many_arguments)]
#[allow(unused_variables, unused_mut)]
fn quantize_chroma_blocks(
    pending: &[Block8x8f],
    output: &mut Vec<[i16; DCT_BLOCK_SIZE]>,
    mut dc_raw_output: Option<&mut Vec<i32>>,
    all_aq_strengths: &[f32],
    quant_simd: &QuantTableSimd,
    zero_bias_simd: &ZeroBiasSimd,
    quant_values: &[u16; DCT_BLOCK_SIZE],
    #[cfg(feature = "trellis")] hybrid_ctx: Option<&HybridQuantContext>,
    use_trellis: bool,
    chroma_blocks_h: usize,
    chroma_blocks_v: usize,
    y_blocks_w: usize,
    y_blocks_h: usize,
) {
    let blocks_h = chroma_blocks_h.max(1);
    let global_chroma_by = output.len() / blocks_h;

    for (i, dct) in pending.iter().enumerate() {
        // Store raw DC if DC trellis is enabled (scaled by 64 for trellis compatibility)
        #[cfg(feature = "trellis")]
        if let Some(dc_raw) = dc_raw_output.as_deref_mut() {
            let row0: [f32; 8] = dct.rows[0].into();
            let dc_val = (row0[0] * 64.0).round() as i32;
            dc_raw.push(dc_val);
        }

        let bx = i % blocks_h;
        let local_by = i / blocks_h;
        let y_bx = (bx * y_blocks_w) / blocks_h;
        let chroma_by = global_chroma_by + local_by;
        let y_by = (chroma_by * y_blocks_h) / chroma_blocks_v.max(1);
        let global_aq_idx = y_by * y_blocks_w + y_bx.min(y_blocks_w.saturating_sub(1));
        let aq_strength = if global_aq_idx < all_aq_strengths.len() {
            all_aq_strengths[global_aq_idx]
        } else {
            0.08 // C++ mean fallback
        };

        #[cfg(feature = "trellis")]
        let zigzag = if use_trellis {
            let dct_arr = dct.to_array();
            let natural = hybrid_ctx.unwrap().quantize_block(
                &dct_arr,
                quant_values,
                aq_strength,
                1.0,
                false, // is_chroma
            );
            let mut result = [0i16; DCT_BLOCK_SIZE];
            for j in 0..DCT_BLOCK_SIZE {
                result[JPEG_ZIGZAG_ORDER[j] as usize] = natural[j];
            }
            result
        } else {
            quant_simd.quantize_with_zero_bias_zigzag(dct, zero_bias_simd, aq_strength)
        };
        #[cfg(not(feature = "trellis"))]
        let zigzag = quant_simd.quantize_with_zero_bias_zigzag(dct, zero_bias_simd, aq_strength);

        output.push(zigzag);
    }
}

/// Strip-based encoder for low-memory JPEG encoding.
///
/// Processes the image in horizontal strips to avoid materializing
/// full f32 planes in memory.
#[derive(Debug)]
pub struct StripProcessor {
    /// Immutable image layout — single source of truth for all geometry.
    pub(super) layout: LayoutParams,
    /// Pixel format of input data
    pub(super) pixel_format: PixelFormat,
    /// Chroma downsampling method (Box, GammaAware, GammaAwareIterative)
    pub(super) chroma_downsampling: DownsamplingMethod,

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

    // === Raw DC values for DC trellis optimization ===
    /// Y channel raw DC coefficients (scaled by 64 for trellis compatibility).
    /// Only populated when DC trellis is enabled.
    y_dc_raw: Vec<i32>,
    /// Cb channel raw DC coefficients (scaled by 64 for trellis compatibility).
    cb_dc_raw: Vec<i32>,
    /// Cr channel raw DC coefficients (scaled by 64 for trellis compatibility).
    cr_dc_raw: Vec<i32>,

    // === Pending iMCU DCT blocks (wide-native, double-buffered) ===
    pending: PendingBuffers,

    // === Quantization context ===
    quant: QuantContext,

    // Block dimensions accessible via self.layout.*

    // === Accumulated AQ strengths for batch finalize (debugging) ===
    all_aq_strengths: Vec<f32>,

    // === Streaming AQ state (low memory, rolling buffers) ===
    aq_state: StreamingAQ,

    // === Allocation tracking ===
    /// Tracks all allocations made by this processor
    stats: crate::foundation::alloc::EncodeStats,

    // === Optional preprocessing ===
    /// Enable overshoot deringing (on by default)
    deringing: bool,

    // === Trellis quantization ===
    /// Trellis quantization context for rate-distortion optimization.
    /// When Some, uses trellis quantization instead of standard SIMD quantization.
    #[cfg(feature = "trellis")]
    hybrid_ctx: Option<HybridQuantContext>,

    // === Archmage SIMD token (feature-gated) ===
    /// Desktop64 token for zero-dispatch SIMD operations.
    /// Obtained once at construction, reused for all blocks.
    #[cfg(target_arch = "x86_64")]
    simd_token: Option<crate::encode::mage_simd::Desktop64>,

    // === Reusable u8 buffers for yuv crate (yuv feature) ===
    /// Temporary Y buffer for yuv crate conversion (avoids per-strip allocation)
    #[cfg(feature = "yuv")]
    yuv_temp_y: Vec<u8>,
    /// Temporary Cb buffer for yuv crate conversion
    #[cfg(feature = "yuv")]
    yuv_temp_cb: Vec<u8>,
    /// Temporary Cr buffer for yuv crate conversion
    #[cfg(feature = "yuv")]
    yuv_temp_cr: Vec<u8>,

    // === Reusable AQ strengths buffer ===
    /// Buffer for AQ strengths from process_y_strip_into (avoids per-strip allocation)
    aq_strengths_buffer: Vec<f32>,
}

impl StripProcessor {
    /// Creates a new strip processor with default settings (for tests only).
    #[cfg(test)]
    pub fn new(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
    ) -> Result<Self> {
        let quant = QuantContext::default_for_tests();
        Self::with_xyb(
            width,
            height,
            subsampling,
            pixel_format,
            DownsamplingMethod::Box,
            0,
            false,
            quant,
            true, // aq_enabled
        )
    }

    /// Creates a new strip processor with XYB mode support.
    ///
    /// All geometry is computed once by `LayoutParams` — no duplicate calculations.
    /// `quant` provides all quantization tables and zero-bias parameters.
    pub fn with_xyb(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
        chroma_downsampling: DownsamplingMethod,
        _restart_interval: u16,
        use_xyb: bool,
        quant: QuantContext,
        aq_enabled: bool,
    ) -> Result<Self> {
        Self::with_xyb_inner(
            width,
            height,
            subsampling,
            pixel_format,
            chroma_downsampling,
            _restart_interval,
            use_xyb,
            quant,
            aq_enabled,
            false,
        )
    }

    /// Like [`with_xyb`](Self::with_xyb), but when `streaming_through` is true,
    /// only allocates one strip's worth of block storage instead of the full image.
    /// Use this when blocks are drained after each strip (streaming-through mode).
    pub fn with_xyb_streaming(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
        chroma_downsampling: DownsamplingMethod,
        _restart_interval: u16,
        use_xyb: bool,
        quant: QuantContext,
        aq_enabled: bool,
    ) -> Result<Self> {
        Self::with_xyb_inner(
            width,
            height,
            subsampling,
            pixel_format,
            chroma_downsampling,
            _restart_interval,
            use_xyb,
            quant,
            aq_enabled,
            true,
        )
    }

    fn with_xyb_inner(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
        chroma_downsampling: DownsamplingMethod,
        _restart_interval: u16,
        use_xyb: bool,
        quant: QuantContext,
        aq_enabled: bool,
        streaming_through: bool,
    ) -> Result<Self> {
        let layout = LayoutParams::new(width, height, subsampling, use_xyb);

        let strip_height = layout.strip_height;
        let padded_width = layout.padded_width;
        let padded_c_width = layout.padded_c_width;
        let padded_b_width = layout.padded_b_width;
        let c_strip_height = layout.c_strip_height;
        let b_strip_height = layout.b_strip_height;
        let total_y_blocks = layout.total_y_blocks;
        let total_c_blocks = layout.total_c_blocks;
        let pending_y_capacity = layout.pending_y_capacity;
        let pending_c_capacity = layout.pending_c_capacity;

        let is_color = !pixel_format.is_grayscale();

        // Track all allocations
        let mut stats = EncodeStats::new();

        // Initialize streaming AQ from layout and quant tables
        let y_quant_01 = quant.y_quant.values[1]; // Position [0,1] in zigzag
        let aq_state = StreamingAQ::new(&layout, y_quant_01, aq_enabled)?;

        Ok(Self {
            layout,
            pixel_format,
            chroma_downsampling,

            // Strip buffers (sized for PADDED width for edge handling parity)
            y_strip: try_alloc_zeroed_f32_tracked(
                padded_width * strip_height,
                "y_strip",
                &mut stats,
            )?,
            cb_strip: if is_color {
                try_alloc_zeroed_f32_tracked(padded_width * strip_height, "cb_strip", &mut stats)?
            } else {
                Vec::new()
            },
            cr_strip: if is_color {
                try_alloc_zeroed_f32_tracked(padded_width * strip_height, "cr_strip", &mut stats)?
            } else {
                Vec::new()
            },
            cb_down: if is_color {
                try_alloc_zeroed_f32_tracked(
                    padded_c_width * c_strip_height,
                    "cb_down",
                    &mut stats,
                )?
            } else {
                Vec::new()
            },
            cr_down: if is_color {
                // For XYB mode, cr_down holds B channel which is 2x2 downsampled
                let cr_down_size = if use_xyb {
                    padded_b_width * b_strip_height
                } else {
                    padded_c_width * c_strip_height
                };
                try_alloc_zeroed_f32_tracked(cr_down_size, "cr_down", &mut stats)?
            } else {
                Vec::new()
            },

            // Final i16 block storage (pre-allocated capacity).
            // In streaming-through mode, blocks are drained each strip,
            // so we only need one strip's worth of capacity.
            y_blocks: {
                let cap = if streaming_through {
                    pending_y_capacity
                } else {
                    total_y_blocks
                };
                try_with_capacity_tracked(cap, "y_blocks", &mut stats)?
            },
            cb_blocks: if is_color {
                let cap = if streaming_through {
                    pending_c_capacity
                } else {
                    total_c_blocks
                };
                try_with_capacity_tracked(cap, "cb_blocks", &mut stats)?
            } else {
                Vec::new()
            },
            cr_blocks: if is_color {
                let cap = if streaming_through {
                    pending_c_capacity
                } else {
                    total_c_blocks
                };
                try_with_capacity_tracked(cap, "cr_blocks", &mut stats)?
            } else {
                Vec::new()
            },

            // Raw DC values for DC trellis (only populated when DC trellis enabled)
            // Allocation is deferred - we'll grow these lazily if needed
            y_dc_raw: Vec::new(),
            cb_dc_raw: Vec::new(),
            cr_dc_raw: Vec::new(),

            // Pending f32 DCT blocks (double-buffered, capacity for one iMCU row)
            pending: PendingBuffers {
                y: [
                    try_with_capacity_tracked(pending_y_capacity, "pending_y[0]", &mut stats)?,
                    try_with_capacity_tracked(pending_y_capacity, "pending_y[1]", &mut stats)?,
                ],
                cb: if is_color {
                    [
                        try_with_capacity_tracked(pending_c_capacity, "pending_cb[0]", &mut stats)?,
                        try_with_capacity_tracked(pending_c_capacity, "pending_cb[1]", &mut stats)?,
                    ]
                } else {
                    [Vec::new(), Vec::new()]
                },
                cr: if is_color {
                    [
                        try_with_capacity_tracked(pending_c_capacity, "pending_cr[0]", &mut stats)?,
                        try_with_capacity_tracked(pending_c_capacity, "pending_cr[1]", &mut stats)?,
                    ]
                } else {
                    [Vec::new(), Vec::new()]
                },
                current: false,
            },

            quant,

            // Accumulated AQ strengths (for output)
            all_aq_strengths: {
                let cap = if streaming_through {
                    pending_y_capacity
                } else {
                    total_y_blocks
                };
                try_with_capacity_tracked(cap, "all_aq_strengths", &mut stats)?
            },

            aq_state,

            // Allocation tracking
            stats,

            // Optional preprocessing (deringing on by default)
            deringing: true,

            // Trellis quantization (disabled by default)
            #[cfg(feature = "trellis")]
            hybrid_ctx: None,

            // Archmage SIMD token (obtained once, reused for all blocks)
            #[cfg(target_arch = "x86_64")]
            simd_token: {
                use archmage::SimdToken;
                crate::encode::mage_simd::Desktop64::summon()
            },

            // Reusable u8 buffers for yuv crate (one strip worth of pixels)
            // Allocated once, reused for each strip to avoid per-strip allocation
            #[cfg(feature = "yuv")]
            yuv_temp_y: if is_color {
                vec![0u8; padded_width * strip_height]
            } else {
                Vec::new()
            },
            #[cfg(feature = "yuv")]
            yuv_temp_cb: if is_color {
                vec![0u8; padded_width * strip_height]
            } else {
                Vec::new()
            },
            #[cfg(feature = "yuv")]
            yuv_temp_cr: if is_color {
                vec![0u8; padded_width * strip_height]
            } else {
                Vec::new()
            },

            // Reusable AQ strengths buffer (one iMCU row worth of blocks)
            // Size: blocks_per_row * v_samp_factor (max 2 for 4:2:0)
            aq_strengths_buffer: vec![0.0f32; pending_y_capacity],
        })
    }

    /// Returns allocation statistics for this processor.
    #[must_use]
    pub fn encode_stats(&self) -> &EncodeStats {
        &self.stats
    }

    /// Borrow the accumulated Y blocks.
    pub fn y_blocks(&self) -> &[[i16; DCT_BLOCK_SIZE]] {
        &self.y_blocks
    }

    /// Borrow the accumulated Cb blocks.
    pub fn cb_blocks(&self) -> &[[i16; DCT_BLOCK_SIZE]] {
        &self.cb_blocks
    }

    /// Borrow the accumulated Cr blocks.
    pub fn cr_blocks(&self) -> &[[i16; DCT_BLOCK_SIZE]] {
        &self.cr_blocks
    }

    /// Clear accumulated blocks, keeping allocations for reuse.
    pub fn clear_blocks(&mut self) {
        self.y_blocks.clear();
        self.cb_blocks.clear();
        self.cr_blocks.clear();
        self.all_aq_strengths.clear();
        self.y_dc_raw.clear();
        self.cb_dc_raw.clear();
        self.cr_dc_raw.clear();
    }

    #[must_use]
    pub fn take_blocks(&mut self) -> StripProcessorOutput {
        StripProcessorOutput {
            y_blocks: core::mem::take(&mut self.y_blocks),
            cb_blocks: core::mem::take(&mut self.cb_blocks),
            cr_blocks: core::mem::take(&mut self.cr_blocks),
            aq_strengths: core::mem::take(&mut self.all_aq_strengths),
            stats: EncodeStats::new(), // Fresh stats for remaining work
            y_dc_raw: core::mem::take(&mut self.y_dc_raw),
            cb_dc_raw: core::mem::take(&mut self.cb_dc_raw),
            cr_dc_raw: core::mem::take(&mut self.cr_dc_raw),
        }
    }

    /// Returns the archmage SIMD token if available.
    ///
    /// The token is obtained once at construction and can be reused for all blocks
    /// with zero per-call dispatch overhead.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[must_use]
    pub fn simd_token(&self) -> Option<crate::encode::mage_simd::Desktop64> {
        self.simd_token
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
    #[cfg(feature = "trellis")]
    pub fn set_trellis(&mut self, config: TrellisConfig) {
        if config.is_enabled() {
            self.hybrid_ctx = Some(HybridQuantContext::from_trellis_config(config));
        } else {
            self.hybrid_ctx = None;
        }
    }

    /// Sets hybrid quantization configuration (AQ-coupled trellis).
    ///
    /// Hybrid mode adjusts trellis lambda per-block based on AQ strength,
    /// spending more bits on smooth areas and fewer on complex textures.
    #[cfg(feature = "trellis")]
    pub fn set_hybrid(&mut self, config: crate::encode::trellis::HybridConfig) {
        self.hybrid_ctx = Some(HybridQuantContext::new(config));
    }

    /// Returns whether XYB mode is enabled.
    #[must_use]
    pub fn is_xyb(&self) -> bool {
        self.layout.use_xyb
    }

    /// Returns the strip height for iteration.
    pub fn strip_height(&self) -> usize {
        self.layout.strip_height
    }

    /// Returns the subsampling mode.
    pub fn subsampling(&self) -> Subsampling {
        self.layout.subsampling
    }

    /// Returns the strip buffer used as AQ input (luminance channel).
    ///
    /// In YCbCr mode this is `y_strip`. In XYB mode, the Y perceptual channel
    /// is stored in `cb_strip` (component index 1), not `y_strip` (which holds X).
    /// C++ jpegli uses `y_channel = (jpeg_color_space == JCS_RGB) ? 1 : 0`.
    fn aq_input_strip(&self) -> &[f32] {
        if self.layout.use_xyb {
            &self.cb_strip
        } else {
            &self.y_strip
        }
    }

    /// Processes one strip of RGB input data.
    ///
    /// Dispatch color conversion for a strip of RGB input.
    ///
    /// Returns `true` if chroma was already downsampled by a fused path
    /// (XYB, gamma-aware, or fused 420).
    fn color_convert_strip(
        &mut self,
        rgb_strip: &[u8],
        strip_y: usize,
        actual_strip_height: usize,
    ) -> Result<bool> {
        if self.layout.use_xyb {
            self.convert_strip_to_xyb(rgb_strip, actual_strip_height)?;
            return Ok(true); // XYB handles B downsampling internally
        }

        // YCbCr mode: choose optimal path based on subsampling
        let uses_gamma_aware_fused = self.chroma_downsampling.uses_gamma_aware()
            && !self.pixel_format.is_grayscale()
            && self.layout.subsampling != Subsampling::S444;

        if uses_gamma_aware_fused {
            self.convert_strip_gamma_aware(rgb_strip, strip_y, actual_strip_height)?;
            return Ok(true);
        }

        // Try fused 420 path when applicable (significantly faster)
        #[cfg(feature = "yuv")]
        {
            if self.layout.subsampling == Subsampling::S420
                && !self.pixel_format.is_grayscale()
                && self.convert_strip_to_ycbcr_420(rgb_strip, actual_strip_height)?
            {
                return Ok(true);
            }
        }

        // Standard path: convert to YCbCr 444, then downsample separately
        self.convert_strip_to_ycbcr(rgb_strip, actual_strip_height)?;
        Ok(false)
    }

    /// # Arguments
    /// * `rgb_strip` - RGB pixel data for this strip
    /// * `strip_y` - Starting row index of this strip
    ///
    /// # Returns
    /// Number of blocks added during this strip
    pub fn process_strip(&mut self, rgb_strip: &[u8], strip_y: usize) -> Result<usize> {
        let actual_strip_height = self.layout.strip_height.min(self.layout.height - strip_y);

        // Color convert RGB -> YCbCr or XYB into strip buffers
        let chroma_already_downsampled =
            self.color_convert_strip(rgb_strip, strip_y, actual_strip_height)?;

        // Pad strips vertically if this is a partial bottom strip
        // (needed for vertical downsampling modes at image bottom)
        if actual_strip_height < self.layout.strip_height {
            self.pad_strips_vertically(actual_strip_height, self.layout.strip_height);
        }

        let need_chroma_downsample =
            !self.pixel_format.is_grayscale() && !chroma_already_downsampled;
        self.process_strip_common(strip_y, actual_strip_height, need_chroma_downsample)
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
        if self.layout.use_xyb {
            return Err(crate::error::Error::unsupported_feature(
                "YCbCr input not supported for XYB mode",
            ));
        }

        let actual_strip_height = self.layout.strip_height.min(self.layout.height - strip_y);

        // Copy YCbCr data to strip buffers with level shift
        // Convert from centered [-128, 127] to JPEG range [0, 255]
        self.copy_ycbcr_to_strips(y_row, cb_row, cr_row, actual_strip_height)?;

        // Pad strips vertically if this is a partial bottom strip
        if actual_strip_height < self.layout.strip_height {
            self.pad_strips_vertically(actual_strip_height, self.layout.strip_height);
        }

        let need_chroma_downsample = !self.pixel_format.is_grayscale();
        self.process_strip_common(strip_y, actual_strip_height, need_chroma_downsample)
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
        if self.layout.use_xyb {
            return Err(crate::error::Error::unsupported_feature(
                "YCbCr input not supported for XYB mode",
            ));
        }

        let actual_strip_height = self.layout.strip_height.min(self.layout.height - strip_y);

        // Copy Y with level shift, copy chroma directly to downsampled buffers
        self.copy_ycbcr_subsampled_to_strips(y_row, cb_row, cr_row, actual_strip_height)?;

        // Pad strips vertically if this is a partial bottom strip
        if actual_strip_height < self.layout.strip_height {
            self.pad_strips_vertically(actual_strip_height, self.layout.strip_height);
            // Also pad chroma downsampled buffers
            self.pad_chroma_down_vertically(actual_strip_height)?;
        }

        // Chroma already downsampled by caller
        self.process_strip_common(strip_y, actual_strip_height, false)
    }

    /// Shared tail for all process_strip variants: AQ, downsample, quantize, DCT.
    ///
    /// Called after the preamble (color conversion or YCbCr copy) and vertical
    /// padding have been completed. The `need_chroma_downsample` flag encodes
    /// whether the chroma planes still need downsampling:
    /// - `process_strip`: `!grayscale && !fused_path`
    /// - `process_strip_ycbcr_f32`: `!grayscale` (always needs downsample)
    /// - `process_strip_ycbcr_f32_subsampled`: `false` (caller did it)
    fn process_strip_common(
        &mut self,
        strip_y: usize,
        actual_strip_height: usize,
        need_chroma_downsample: bool,
    ) -> Result<usize> {
        // Process AQ and check if previous iMCU strengths are ready.
        // AQ uses the luminance channel: y_strip for YCbCr, cb_strip for XYB
        // (see aq_input_strip() for rationale; inlined here to avoid borrow conflict)
        let aq_input = if self.layout.use_xyb {
            &self.cb_strip
        } else {
            &self.y_strip
        };
        let aq_count = self.aq_state.process_y_strip_into(
            aq_input,
            strip_y,
            actual_strip_height,
            &mut self.aq_strengths_buffer,
        );

        // Use full strip height for downsample/DCT when bottom strip was padded
        let downsample_height = if actual_strip_height < self.layout.strip_height {
            self.layout.strip_height
        } else {
            actual_strip_height
        };

        // Downsample chroma if needed (skipped when fused path or caller handled it)
        if need_chroma_downsample {
            self.downsample_chroma_strip(downsample_height)?;
        }

        // If we got AQ strengths, quantize the previous pending iMCU.
        // This is the key optimization: quantize to i16 immediately instead of storing f32.
        if let Some(count) = aq_count {
            // Use mem::take to avoid borrow conflict (moves buffer, no allocation)
            let temp_buffer = std::mem::take(&mut self.aq_strengths_buffer);
            self.quantize_prev_pending_imcu(&temp_buffer[..count]);
            self.aq_strengths_buffer = temp_buffer;
            self.pending.clear_prev();
        }

        // Compute DCT for blocks in this strip into the current pending buffer.
        // Swap of pending buffers happens inside dct_strip_blocks_to_pending on iMCU boundary.
        self.dct_strip_blocks_to_pending(strip_y, downsample_height)
    }

    /// Computes DCT for blocks in the current strip and stores in pending buffer.
    /// This allows quantization to happen incrementally when AQ strengths become available.
    fn dct_strip_blocks_to_pending(
        &mut self,
        strip_y: usize,
        strip_height: usize,
    ) -> Result<usize> {
        // Use original dimensions for block counts (parity with full-plane encoder)
        let blocks_w = self.layout.blocks_w;
        let strip_blocks_h = (strip_height + 7) / 8;
        let start_block_y = strip_y / 8;
        let height = self.layout.height;
        let pending_idx = self.pending.current_idx();

        // Y strip is now in padded layout (padded_width pixels per row)
        let padded_width = self.layout.padded_width;

        // Extract SIMD token once for all blocks in this strip
        #[cfg(target_arch = "x86_64")]
        let simd_token = self.simd_token;
        #[cfg(not(target_arch = "x86_64"))]
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
            let deringing = if self.deringing {
                Some(self.quant.y_quant.values[0])
            } else {
                None
            };
            super::parallel::parallel_dct_y_blocks(
                &self.y_strip[..y_size],
                blocks_w,
                actual_strip_blocks_h,
                padded_width,
                deringing,
                &mut self.pending.y[pending_idx],
            );
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Pre-allocate and write directly to avoid push overhead
            let start_idx = self.pending.y[pending_idx].len();
            self.pending.y[pending_idx].resize(start_idx + blocks_added, Block8x8f::default());
            let output = &mut self.pending.y[pending_idx][start_idx..];

            // Get DC quant value for deringing (if enabled)
            let y_dc_quant = self.quant.y_quant.values[0];

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
            if self.layout.use_xyb {
                // XYB mode: Y component is full resolution, B is 2x2 downsampled
                // - Y (cb_strip): full res, same block dimensions as X (y_strip)
                // - B (cr_down): 2x2 downsampled

                // Process Y component (full res from cb_strip which is now in padded layout)
                let y_size = strip_height * padded_width;

                // Calculate actual Cb block count (same as Y - uses actual_strip_blocks_h)
                let cb_blocks_total = actual_strip_blocks_h * blocks_w;
                let cb_start = self.pending.cb[pending_idx].len();
                self.pending.cb[pending_idx]
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
                        self.pending.cb[pending_idx][cb_start + cb_idx] =
                            forward_dct_dispatch(simd_token, &cb_block);
                        cb_idx += 1;
                    }
                }

                // Process B component (2x2 downsampled from cr_down)
                let b_blocks_w = self.layout.b_blocks_w;
                let b_strip_height = self.layout.b_strip_height;
                let b_strip_blocks_h = (b_strip_height + 7) / 8;
                let b_blocks_total = b_blocks_w * b_strip_blocks_h;
                let padded_b_width = self.layout.padded_b_width;
                let b_size = b_strip_height * padded_b_width;

                let cr_start = self.pending.cr[pending_idx].len();
                self.pending.cr[pending_idx]
                    .resize(cr_start + b_blocks_total, Block8x8f::default());

                let mut cr_idx = 0;
                for local_by in 0..b_strip_blocks_h {
                    for bx in 0..b_blocks_w {
                        let cr_block = extract_block_from_strip_wide(
                            &self.cr_down[..b_size],
                            bx,
                            local_by,
                            padded_b_width,
                        );
                        self.pending.cr[pending_idx][cr_start + cr_idx] =
                            forward_dct_dispatch(simd_token, &cr_block);
                        cr_idx += 1;
                    }
                }
            } else {
                // YCbCr mode: chroma dimensions from layout (single source of truth)
                let c_blocks_w = self.layout.c_blocks_w;
                let c_strip_height = self.layout.c_strip_height;
                let c_strip_blocks_h = (c_strip_height + 7) / 8;
                let c_blocks_total = c_blocks_w * c_strip_blocks_h;
                let padded_c_width = self.layout.padded_c_width;
                let c_size = c_strip_height * padded_c_width;

                // Pre-allocate chroma buffers
                let cb_start = self.pending.cb[pending_idx].len();
                let cr_start = self.pending.cr[pending_idx].len();
                self.pending.cb[pending_idx]
                    .resize(cb_start + c_blocks_total, Block8x8f::default());
                self.pending.cr[pending_idx]
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
                        self.pending.cb[pending_idx][cb_start + idx] =
                            forward_dct_dispatch(simd_token, &cb_block);

                        // Cr block - DCT only (wide-native path)
                        let cr_block = extract_block_from_strip_wide(
                            &self.cr_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        self.pending.cr[pending_idx][cr_start + idx] =
                            forward_dct_dispatch(simd_token, &cr_block);
                        idx += 1;
                    }
                }
            }
        }

        // Swap pending buffer for next iMCU
        self.pending.swap();

        Ok(blocks_added)
    }

    /// Quantizes the previous pending iMCU's f32 DCT blocks to i16 using AQ strengths.
    ///
    /// This is the key memory optimization: quantize incrementally as soon as
    /// AQ strengths become available, rather than storing all f32 blocks.
    ///
    /// When trellis quantization is enabled (via `set_trellis()`), uses
    /// rate-distortion optimization for better compression. Otherwise uses
    /// fast SIMD quantization.
    fn quantize_prev_pending_imcu(&mut self, aq_strengths: &[f32]) {
        let buffer_idx = self.pending.prev_idx();
        let quant = &self.quant;

        // Check if we have trellis context for R-D optimization
        #[cfg(feature = "trellis")]
        let use_trellis = self.hybrid_ctx.is_some();
        #[cfg(not(feature = "trellis"))]
        let use_trellis = false;

        #[cfg(feature = "trellis")]
        let store_dc_raw = self
            .hybrid_ctx
            .as_ref()
            .is_some_and(|ctx| ctx.is_dc_trellis_enabled());
        #[cfg(not(feature = "trellis"))]
        let store_dc_raw = false;

        // Quantize Y blocks (vectors pre-allocated at construction)
        for (i, dct) in self.pending.y[buffer_idx].iter().enumerate() {
            // Use get() with fallback to avoid branch on common path
            let aq_strength = aq_strengths.get(i).copied().unwrap_or(0.08);

            // Store raw DC if DC trellis is enabled (scaled by 64 for trellis compatibility)
            if store_dc_raw {
                let row0: [f32; 8] = dct.rows[0].into();
                let dc_raw = (row0[0] * 64.0).round() as i32;
                self.y_dc_raw.push(dc_raw);
            }

            #[cfg(feature = "trellis")]
            let zigzag = if use_trellis {
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
            } else {
                // Fast SIMD path: fused quantization + zigzag reorder
                quant.y_quant_simd.quantize_with_zero_bias_zigzag(
                    dct,
                    &quant.y_zero_bias_simd,
                    aq_strength,
                )
            };
            #[cfg(not(feature = "trellis"))]
            let zigzag = quant.y_quant_simd.quantize_with_zero_bias_zigzag(
                dct,
                &quant.y_zero_bias_simd,
                aq_strength,
            );

            self.y_blocks.push(zigzag);
            self.all_aq_strengths.push(aq_strength);
        }

        // Quantize Cb/Cr blocks
        {
            let y_blocks_w = self.layout.y_blocks_w;
            let y_blocks_h = self.layout.y_blocks_h;
            let c_blocks_w = self.layout.c_blocks_w;
            let c_blocks_h = self.layout.c_blocks_h;

            // Cb: always uses c_blocks dimensions
            let cb_dc_raw = if store_dc_raw {
                Some(&mut self.cb_dc_raw)
            } else {
                None
            };
            quantize_chroma_blocks(
                &self.pending.cb[buffer_idx],
                &mut self.cb_blocks,
                cb_dc_raw,
                &self.all_aq_strengths,
                &quant.cb_quant_simd,
                &quant.cb_zero_bias_simd,
                &quant.cb_quant.values,
                #[cfg(feature = "trellis")]
                self.hybrid_ctx.as_ref(),
                use_trellis,
                c_blocks_w,
                c_blocks_h,
                y_blocks_w,
                y_blocks_h,
            );

            // Cr: for XYB mode, use b_blocks dimensions (B channel is 2x2 downsampled)
            let cr_blocks_h = if self.layout.use_xyb {
                self.layout.b_blocks_w
            } else {
                c_blocks_w
            };
            let cr_blocks_v = if self.layout.use_xyb {
                self.layout.b_blocks_h
            } else {
                c_blocks_h
            };
            let cr_dc_raw = if store_dc_raw {
                Some(&mut self.cr_dc_raw)
            } else {
                None
            };
            quantize_chroma_blocks(
                &self.pending.cr[buffer_idx],
                &mut self.cr_blocks,
                cr_dc_raw,
                &self.all_aq_strengths,
                &quant.cr_quant_simd,
                &quant.cr_zero_bias_simd,
                &quant.cr_quant.values,
                #[cfg(feature = "trellis")]
                self.hybrid_ctx.as_ref(),
                use_trellis,
                cr_blocks_h,
                cr_blocks_v,
                y_blocks_w,
                y_blocks_h,
            );
        }
    }

    /// Finalizes encoding after all strips have been processed.
    ///
    /// With incremental quantization, most blocks are already quantized.
    /// This method only handles the last pending iMCU.
    pub fn finalize(mut self) -> Result<StripProcessorOutput> {
        // Flush AQ to get the last iMCU's strengths
        // Use flush_into to write to reusable buffer (zero allocation)
        let flush_count = self.aq_state.flush_into(&mut self.aq_strengths_buffer);
        if let Some(count) = flush_count {
            // Quantize the last pending iMCU
            if !self.pending.prev_y().is_empty() {
                let temp_buffer = std::mem::take(&mut self.aq_strengths_buffer);
                self.quantize_prev_pending_imcu(&temp_buffer[..count]);
                self.aq_strengths_buffer = temp_buffer;
            }
        }

        // Also quantize any blocks remaining in the current pending buffer
        // (for edge cases where we have blocks but no AQ was returned)
        if !self.pending.current_y().is_empty() {
            // Use default AQ strength for remaining blocks
            let default_aq = try_alloc_filled(
                self.pending.current_y().len(),
                0.08f32,
                "default_aq_strengths",
            )?;
            // Swap so current becomes prev, then quantize
            self.pending.swap();
            self.quantize_prev_pending_imcu(&default_aq);
        }

        // Apply DC trellis optimization if enabled
        #[cfg(feature = "trellis")]
        if !self.y_dc_raw.is_empty() {
            self.apply_dc_trellis();
        }

        Ok(StripProcessorOutput {
            y_blocks: self.y_blocks,
            cb_blocks: self.cb_blocks,
            cr_blocks: self.cr_blocks,
            aq_strengths: self.all_aq_strengths,
            stats: self.stats,
            y_dc_raw: self.y_dc_raw,
            cb_dc_raw: self.cb_dc_raw,
            cr_dc_raw: self.cr_dc_raw,
        })
    }

    /// Apply DC trellis optimization to all quantized blocks.
    ///
    /// DC trellis uses dynamic programming to find optimal DC coefficients
    /// that minimize rate (differential encoding cost) + distortion.
    /// Must be called after all blocks are quantized and DC raw values stored.
    ///
    /// Processes each row of blocks independently, propagating `last_dc` from
    /// one row to the next (matching C mozjpeg behavior). When `delta_dc_weight > 0`,
    /// also considers vertical DC gradients from the row above.
    #[cfg(feature = "trellis")]
    fn apply_dc_trellis(&mut self) {
        let Some(ref hybrid_ctx) = self.hybrid_ctx else {
            return;
        };
        if !hybrid_ctx.is_dc_trellis_enabled() {
            return;
        }

        let config = hybrid_ctx.trellis_config();
        let lambda1 = config.lambda_log_scale1();
        let lambda2 = config.lambda_log_scale2();
        let delta_dc_weight = config.get_delta_dc_weight();

        // Y channel DC trellis (row-by-row)
        if !self.y_dc_raw.is_empty() && !self.y_blocks.is_empty() {
            let blocks_w = self.layout.y_blocks_w;
            let dc_quantval = self.quant.y_quant.values[0];
            let dc_table = hybrid_ctx.luma_dc_rate_table();
            dc_trellis_channel_row_by_row(
                &self.y_dc_raw,
                &mut self.y_blocks,
                blocks_w,
                dc_quantval,
                dc_table,
                lambda1,
                lambda2,
                delta_dc_weight,
            );
        }

        // Cb channel DC trellis (row-by-row)
        if !self.cb_dc_raw.is_empty() && !self.cb_blocks.is_empty() {
            let blocks_w = self.layout.c_blocks_w;
            let dc_quantval = self.quant.cb_quant.values[0];
            let dc_table = hybrid_ctx.chroma_dc_rate_table();
            dc_trellis_channel_row_by_row(
                &self.cb_dc_raw,
                &mut self.cb_blocks,
                blocks_w,
                dc_quantval,
                dc_table,
                lambda1,
                lambda2,
                delta_dc_weight,
            );
        }

        // Cr channel DC trellis (row-by-row)
        if !self.cr_dc_raw.is_empty() && !self.cr_blocks.is_empty() {
            let blocks_w = self.layout.c_blocks_w;
            let dc_quantval = self.quant.cr_quant.values[0];
            let dc_table = hybrid_ctx.chroma_dc_rate_table();
            dc_trellis_channel_row_by_row(
                &self.cr_dc_raw,
                &mut self.cr_blocks,
                blocks_w,
                dc_quantval,
                dc_table,
                lambda1,
                lambda2,
                delta_dc_weight,
            );
        }
    }
}

/// Apply DC trellis optimization to one channel, processing row-by-row.
///
/// C mozjpeg processes DC trellis one row at a time, propagating `last_dc`
/// from the end of each row to the start of the next. This matches that
/// behavior and also supports `delta_dc_weight` for vertical DC gradients.
///
/// Blocks are stored in raster order: block index = row * blocks_w + col.
/// The zigzag-to-natural order conversion is done in-place for DC trellis,
/// then converted back.
#[cfg(feature = "trellis")]
#[allow(clippy::too_many_arguments)]
fn dc_trellis_channel_row_by_row(
    dc_raw: &[i32],
    blocks: &mut [[i16; DCT_BLOCK_SIZE]],
    blocks_w: usize,
    dc_quantval: u16,
    dc_table: &crate::encode::trellis::RateTable,
    lambda1: f32,
    lambda2: f32,
    delta_dc_weight: f32,
) {
    if dc_raw.is_empty() || blocks.is_empty() || blocks_w == 0 {
        return;
    }

    let num_blocks = blocks.len();
    let blocks_h = (num_blocks + blocks_w - 1) / blocks_w;

    // Convert DC-only raw values to full blocks for the DC trellis API.
    // AC coefficients are set to 0 since we've already done AC trellis.
    let raw_blocks: Vec<[i32; DCT_BLOCK_SIZE]> = dc_raw
        .iter()
        .map(|&dc| {
            let mut block = [0i32; DCT_BLOCK_SIZE];
            block[0] = dc;
            block
        })
        .collect();

    // Convert zigzag blocks to natural order for DC trellis
    let mut natural_blocks: Vec<[i16; DCT_BLOCK_SIZE]> = blocks
        .iter()
        .map(|zigzag| {
            let mut natural = [0i16; DCT_BLOCK_SIZE];
            for (i, &zz_idx) in JPEG_ZIGZAG_ORDER.iter().enumerate() {
                natural[i] = zigzag[zz_idx as usize];
            }
            natural
        })
        .collect();

    // Double-buffered above-row data (used when delta_dc_weight > 0).
    // We use two buffers and swap: one holds the previous row's data while
    // the current row writes into the other.
    let mut above_raw_dc: Vec<i32> = vec![0; blocks_w];
    let mut above_quant_dc: Vec<i16> = vec![0; blocks_w];
    let mut current_raw_dc: Vec<i32> = vec![0; blocks_w];

    let mut last_dc: i16 = 0;

    for row in 0..blocks_h {
        let row_start = row * blocks_w;
        let row_end = (row_start + blocks_w).min(num_blocks);
        let row_len = row_end - row_start;

        let indices: Vec<usize> = (row_start..row_end).collect();

        // Snapshot this row's raw DC values before trellis (for next row's above_data)
        for col in 0..row_len {
            current_raw_dc[col] = raw_blocks[row_start + col][0];
        }

        let above_data = if delta_dc_weight > 0.0 && row > 0 {
            Some((
                above_raw_dc[..row_len].as_ref(),
                above_quant_dc[..row_len].as_ref(),
            ))
        } else {
            None
        };

        last_dc = crate::encode::trellis::dc_trellis_optimize_indexed(
            &raw_blocks,
            &mut natural_blocks,
            &indices,
            dc_quantval,
            dc_table,
            last_dc,
            lambda1,
            lambda2,
            delta_dc_weight,
            above_data,
        );

        // Save this row's data as "above" for the next row
        above_raw_dc[..row_len].copy_from_slice(&current_raw_dc[..row_len]);
        for col in 0..row_len {
            above_quant_dc[col] = natural_blocks[row_start + col][0];
        }
    }

    // Convert back to zigzag order
    for (i, natural) in natural_blocks.into_iter().enumerate() {
        for (j, &zz_idx) in JPEG_ZIGZAG_ORDER.iter().enumerate() {
            blocks[i][zz_idx as usize] = natural[j];
        }
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
    pub stats: EncodeStats,
    /// Y channel raw DC coefficients (scaled, for DC trellis post-processing).
    /// Empty if DC trellis was not enabled.
    pub y_dc_raw: Vec<i32>,
    /// Cb channel raw DC coefficients.
    pub cb_dc_raw: Vec<i32>,
    /// Cr channel raw DC coefficients.
    pub cr_dc_raw: Vec<i32>,
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
    #[cfg(feature = "decoder")]
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

            // Encode with strip encoder (baseline for test stability)
            let config = EncoderConfig::ycbcr(85.0, crate::encode::ChromaSubsampling::Quarter)
                .progressive(false)
                .optimize_huffman(true);
            let mut enc = config
                .encode_from_bytes(width as u32, height as u32, PixelLayout::Rgb8Srgb)
                .expect("encoder creation failed");
            enc.push_packed(&rgb, Unstoppable).expect("push failed");
            let jpeg_strip = enc.finish().expect("strip encode failed");

            // Decode and verify
            let decoded_strip = crate::decode::Decoder::new()
                .decode(&jpeg_strip, enough::Unstoppable)
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
            let decoded_pixels = decoded_strip.pixels_u8().unwrap();
            assert_eq!(
                decoded_pixels.len(),
                expected_size,
                "strip data size mismatch at height {}: got {} expected {}",
                height,
                decoded_pixels.len(),
                expected_size
            );

            // Compute mean squared error between original and decoded
            // Note: per-pixel checks removed because sharp synthetic edges create
            // larger compression artifacts than natural images. PSNR is the real
            // quality measure. For real-world parity testing, see
            // tests/strip_edge_cpp_comparison.rs which uses corpus images.
            let mut sum_sq_err: u64 = 0;
            let mut max_diff: i32 = 0;
            for (&orig, &dec) in rgb.iter().zip(decoded_pixels.iter()) {
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
