//! Strip-based low-memory JPEG encoding.
//!
//! This module implements a strip-based encoder that processes the image
//! in horizontal strips (MCU rows), avoiding full-plane f32 allocations.
//!
//! # Memory Model
//!
//! Traditional encoder peak memory for 12MP (4000×3000):
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
//! 1. Convert RGB → YCbCr (f32 strips, reused)
//! 2. Accumulate AQ features for this strip
//! 3. Downsample chroma if needed
//! 4. DCT → store f32 coefficients in pending buffer
//! 5. If AQ returns strengths for previous iMCU:
//!    - Quantize pending f32 → i16
//!    - Append to final i16 storage
//! 6. Swap pending buffers
//!
//! After all strips:
//! 1. Flush last iMCU (quantize remaining pending blocks)
//! 2. Build optimized Huffman tables
//! 3. Encode from stored i16 blocks

use crate::alloc::{
    try_alloc_filled, try_alloc_zeroed_f32_tracked, try_clone_slice, try_with_capacity_tracked,
    AllocationStats,
};
use crate::consts::DCT_BLOCK_SIZE;
use crate::error::Result;
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::{QuantTable, ZeroBiasParams};
use crate::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::transfer_functions::srgb_encoded_from_display;
use crate::types::{ChromaDownsampling, PixelFormat, Subsampling};

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

use crate::simd_types::Block8x8f;
use wide::f32x8;

/// Wide-native block extraction: returns Block8x8f directly.
///
/// Assumes strip is properly padded (MCU-aligned) so no bounds checking needed.
/// This is the fast path for the hot encoding loop.
#[inline]
fn extract_block_from_strip_wide(
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

// StreamingAQ uses rolling buffers for low memory (~2.5 MB for 4K vs 33 MB).

/// Strip-based encoder for low-memory JPEG encoding.
///
/// Processes the image in horizontal strips to avoid materializing
/// full f32 planes in memory.
#[derive(Debug)]
pub struct StripProcessor {
    /// Image width in pixels (original)
    width: usize,
    /// Image height in pixels (original)
    height: usize,
    /// Padded width (MCU-aligned for block extraction)
    padded_width: usize,
    /// Padded chroma width
    padded_c_width: usize,
    /// Strip height in pixels (16 for 4:2:0, 8 for 4:4:4)
    strip_height: usize,
    /// Chroma subsampling mode
    subsampling: Subsampling,
    /// Pixel format of input data
    pixel_format: PixelFormat,
    /// Chroma downsampling method (Box, GammaAware, GammaAwareIterative)
    chroma_downsampling: ChromaDownsampling,
    /// Use XYB color space instead of YCbCr
    use_xyb: bool,

    // === Reusable strip buffers (f32) ===
    /// Y channel strip buffer
    y_strip: Vec<f32>,
    /// Cb channel strip buffer (full res before downsample)
    cb_strip: Vec<f32>,
    /// Cr channel strip buffer (full res before downsample)
    cr_strip: Vec<f32>,
    /// Cb downsampled strip buffer
    cb_down: Vec<f32>,
    /// Cr downsampled strip buffer
    cr_down: Vec<f32>,

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
    alloc_stats: crate::alloc::AllocationStats,
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
            ChromaDownsampling::Box,
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
        chroma_downsampling: ChromaDownsampling,
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
        chroma_downsampling: ChromaDownsampling,
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

        let is_color = pixel_format != PixelFormat::Gray;

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
        })
    }

    /// Returns allocation statistics for this processor.
    #[must_use]
    pub fn allocation_stats(&self) -> &AllocationStats {
        &self.alloc_stats
    }

    /// Enables or disables XYB color space mode.
    ///
    /// When enabled, strips are converted to scaled XYB instead of YCbCr.
    /// In XYB mode, the B channel is always 2x2 downsampled regardless of
    /// the subsampling setting.
    pub fn set_xyb_mode(&mut self, enable: bool) {
        self.use_xyb = enable;
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
        let y_quant_01 = y_quant.values[1] as u16; // Position [0,1] in zigzag
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

        // Step 1: Color convert RGB → YCbCr or XYB into strip buffers
        if self.use_xyb {
            // XYB mode: convert RGB → scaled XYB
            // X and Y go to y_strip and cb_strip at full res
            // B goes to cr_strip at full res, then is always 2x2 downsampled
            self.convert_strip_to_xyb(rgb_strip, actual_strip_height)?;
        } else {
            // YCbCr mode: choose optimal path based on subsampling
            let uses_gamma_aware_fused = self.chroma_downsampling.uses_gamma_aware()
                && self.pixel_format != PixelFormat::Gray
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
            && self.pixel_format != PixelFormat::Gray
            && self.subsampling != Subsampling::S444;
        if self.pixel_format != PixelFormat::Gray && !uses_gamma_aware_fused && !self.use_xyb {
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

    /// Converts RGB strip data to YCbCr in the strip buffers.
    ///
    /// Uses strided SIMD conversion that writes Y directly with padded stride,
    /// eliminating the need for a separate rearrange pass.
    fn convert_strip_to_ycbcr(&mut self, rgb_strip: &[u8], strip_height: usize) -> Result<()> {
        let width = self.width;
        let padded_width = self.padded_width;
        let num_pixels = strip_height * width;
        let y_size = strip_height * padded_width;

        match self.pixel_format {
            PixelFormat::Rgb | PixelFormat::Rgba => {
                let bpp = self.pixel_format.bytes_per_pixel();
                crate::encode_simd::rgb_to_ycbcr_strided_inplace(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    padded_width,
                    bpp,
                );
            }
            PixelFormat::Bgr | PixelFormat::Bgra => {
                let bpp = self.pixel_format.bytes_per_pixel();
                crate::encode_simd::bgr_to_ycbcr_strided_inplace(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    padded_width,
                    bpp,
                );
            }
            PixelFormat::Gray => {
                // Grayscale: write Y with strided layout directly
                for row in 0..strip_height {
                    let src_start = row * width;
                    let dst_start = row * padded_width;
                    for x in 0..width {
                        self.y_strip[dst_start + x] = rgb_strip[src_start + x] as f32;
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[dst_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[dst_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::Cmyk => {
                // CMYK: scalar conversion with strided Y output
                use crate::consts::{
                    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR,
                    YCBCR_G_TO_Y, YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
                };
                let bpp = self.pixel_format.bytes_per_pixel();
                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    for x in 0..width {
                        let idx = (row * width + x) * bpp;

                        let c = rgb_strip[idx] as f32 / 255.0;
                        let m = rgb_strip[idx + 1] as f32 / 255.0;
                        let y_val = rgb_strip[idx + 2] as f32 / 255.0;
                        let k = rgb_strip[idx + 3] as f32 / 255.0;
                        let r = 255.0 * (1.0 - c) * (1.0 - k);
                        let g = 255.0 * (1.0 - m) * (1.0 - k);
                        let b = 255.0 * (1.0 - y_val) * (1.0 - k);

                        // Use FMA for accuracy (single rounding)
                        self.y_strip[y_row_start + x] =
                            YCBCR_R_TO_Y.mul_add(r, YCBCR_G_TO_Y.mul_add(g, YCBCR_B_TO_Y * b));
                        self.cb_strip[cbcr_row_start + x] = YCBCR_R_TO_CB
                            .mul_add(r, YCBCR_G_TO_CB.mul_add(g, YCBCR_B_TO_CB.mul_add(b, 128.0)));
                        self.cr_strip[cbcr_row_start + x] = YCBCR_R_TO_CR
                            .mul_add(r, YCBCR_G_TO_CR.mul_add(g, YCBCR_B_TO_CR.mul_add(b, 128.0)));
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[y_row_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[y_row_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::Bgrx => {
                // BGRX: fast path - 4 bytes per pixel, padding ignored
                crate::encode_simd::bgr_to_ycbcr_strided_inplace(
                    rgb_strip,
                    &mut self.y_strip[..y_size],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    padded_width,
                    4, // BGRX is 4 bytes per pixel
                );
            }
            // 16-bit formats (assumed linear, convert through sRGB)
            // High-precision path: all math in f64, single final cast to f32
            PixelFormat::Gray16 => {
                // Gray16: 2 bytes per pixel, native endian, linear
                // Convert: linear -> sRGB -> Y (scale to 0-255 range)
                for row in 0..strip_height {
                    let src_start = row * width * 2;
                    let dst_start = row * padded_width;
                    for x in 0..width {
                        let idx = src_start + x * 2;
                        let linear =
                            u16::from_ne_bytes([rgb_strip[idx], rgb_strip[idx + 1]]) as f64
                                / 65535.0;
                        // sRGB gamma in f64, scale to 0-255, single cast to f32
                        let srgb = srgb_encoded_from_display(linear) * 255.0;
                        self.y_strip[dst_start + x] = srgb as f32;
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[dst_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[dst_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                // RGB16/RGBA16: 6/8 bytes per pixel, native endian, linear
                // High-precision: all math in f64 until final f32 output
                let bpp = self.pixel_format.bytes_per_pixel();

                // YCbCr coefficients as f64 for full precision
                const R_TO_Y: f64 = 0.299;
                const G_TO_Y: f64 = 0.587;
                const B_TO_Y: f64 = 0.114;
                const R_TO_CB: f64 = -0.168_736;
                const G_TO_CB: f64 = -0.331_264;
                const B_TO_CB: f64 = 0.5;
                const R_TO_CR: f64 = 0.5;
                const G_TO_CR: f64 = -0.418_688;
                const B_TO_CR: f64 = -0.081_312;

                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    for x in 0..width {
                        let base = (row * width + x) * bpp;

                        // Read 16-bit values and convert to linear f64
                        let r_linear =
                            u16::from_ne_bytes([rgb_strip[base], rgb_strip[base + 1]]) as f64
                                / 65535.0;
                        let g_linear =
                            u16::from_ne_bytes([rgb_strip[base + 2], rgb_strip[base + 3]]) as f64
                                / 65535.0;
                        let b_linear =
                            u16::from_ne_bytes([rgb_strip[base + 4], rgb_strip[base + 5]]) as f64
                                / 65535.0;

                        // Convert linear to sRGB in f64, scale to 0-255
                        let r = srgb_encoded_from_display(r_linear) * 255.0;
                        let g = srgb_encoded_from_display(g_linear) * 255.0;
                        let b = srgb_encoded_from_display(b_linear) * 255.0;

                        // Convert sRGB to YCbCr in f64, single cast to f32 at the end
                        self.y_strip[y_row_start + x] =
                            (R_TO_Y * r + G_TO_Y * g + B_TO_Y * b) as f32;
                        self.cb_strip[cbcr_row_start + x] =
                            (R_TO_CB * r + G_TO_CB * g + B_TO_CB * b + 128.0) as f32;
                        self.cr_strip[cbcr_row_start + x] =
                            (R_TO_CR * r + G_TO_CR * g + B_TO_CR * b + 128.0) as f32;
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[y_row_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[y_row_start + x] = edge_val;
                        }
                    }
                }
            }
            // 32-bit float formats (linear, 0.0-1.0)
            // High-precision path: all math in f64, single final cast to f32
            PixelFormat::GrayF32 => {
                // GrayF32: 4 bytes per pixel, linear
                // Convert: linear -> sRGB -> Y (scale to 0-255 range)
                for row in 0..strip_height {
                    let src_start = row * width * 4;
                    let dst_start = row * padded_width;
                    for x in 0..width {
                        let idx = src_start + x * 4;
                        let linear = f32::from_ne_bytes([
                            rgb_strip[idx],
                            rgb_strip[idx + 1],
                            rgb_strip[idx + 2],
                            rgb_strip[idx + 3],
                        ])
                        .clamp(0.0, 1.0) as f64;
                        // sRGB gamma in f64, scale to 0-255, single cast to f32
                        let srgb = srgb_encoded_from_display(linear) * 255.0;
                        self.y_strip[dst_start + x] = srgb as f32;
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[dst_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[dst_start + x] = edge_val;
                        }
                    }
                }
            }
            PixelFormat::RgbF32 | PixelFormat::RgbaF32 => {
                // RgbF32/RgbaF32: 12/16 bytes per pixel, linear
                // High-precision: all math in f64 until final f32 output
                let bpp = self.pixel_format.bytes_per_pixel();

                // YCbCr coefficients as f64 for full precision
                const R_TO_Y: f64 = 0.299;
                const G_TO_Y: f64 = 0.587;
                const B_TO_Y: f64 = 0.114;
                const R_TO_CB: f64 = -0.168_736;
                const G_TO_CB: f64 = -0.331_264;
                const B_TO_CB: f64 = 0.5;
                const R_TO_CR: f64 = 0.5;
                const G_TO_CR: f64 = -0.418_688;
                const B_TO_CR: f64 = -0.081_312;

                for row in 0..strip_height {
                    let y_row_start = row * padded_width;
                    let cbcr_row_start = row * width;
                    for x in 0..width {
                        let base = (row * width + x) * bpp;

                        // Read f32 values, clamp, convert to f64
                        let r_linear = f32::from_ne_bytes([
                            rgb_strip[base],
                            rgb_strip[base + 1],
                            rgb_strip[base + 2],
                            rgb_strip[base + 3],
                        ])
                        .clamp(0.0, 1.0) as f64;
                        let g_linear = f32::from_ne_bytes([
                            rgb_strip[base + 4],
                            rgb_strip[base + 5],
                            rgb_strip[base + 6],
                            rgb_strip[base + 7],
                        ])
                        .clamp(0.0, 1.0) as f64;
                        let b_linear = f32::from_ne_bytes([
                            rgb_strip[base + 8],
                            rgb_strip[base + 9],
                            rgb_strip[base + 10],
                            rgb_strip[base + 11],
                        ])
                        .clamp(0.0, 1.0) as f64;

                        // Convert linear to sRGB in f64, scale to 0-255
                        let r = srgb_encoded_from_display(r_linear) * 255.0;
                        let g = srgb_encoded_from_display(g_linear) * 255.0;
                        let b = srgb_encoded_from_display(b_linear) * 255.0;

                        // Convert sRGB to YCbCr in f64, single cast to f32 at the end
                        self.y_strip[y_row_start + x] =
                            (R_TO_Y * r + G_TO_Y * g + B_TO_Y * b) as f32;
                        self.cb_strip[cbcr_row_start + x] =
                            (R_TO_CB * r + G_TO_CB * g + B_TO_CB * b + 128.0) as f32;
                        self.cr_strip[cbcr_row_start + x] =
                            (R_TO_CR * r + G_TO_CR * g + B_TO_CR * b + 128.0) as f32;
                    }
                    // Edge-pad Y row
                    if width < padded_width {
                        let edge_val = self.y_strip[y_row_start + width - 1];
                        for x in width..padded_width {
                            self.y_strip[y_row_start + x] = edge_val;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Converts RGB strip to scaled XYB color space.
    ///
    /// XYB layout in strip buffers:
    /// - y_strip: scaled X component (full res, padded stride)
    /// - cb_strip: scaled Y component (full res)
    /// - cr_strip: scaled B component (full res, before downsampling)
    /// - cb_down: scaled Y component (copied, full res but in downsampled buffer layout)
    /// - cr_down: scaled B component (2x2 downsampled)
    ///
    /// Note: XYB always uses fixed subsampling: X=1x1, Y=1x1, B=2x2
    fn convert_strip_to_xyb(&mut self, rgb_strip: &[u8], strip_height: usize) -> Result<()> {
        use crate::xyb::srgb_to_scaled_xyb;

        let width = self.width;
        let padded_width = self.padded_width;
        let bpp = self.pixel_format.bytes_per_pixel();

        // Only support 8-bit RGB/RGBA/BGR/BGRA/BGRX for XYB (no grayscale, CMYK, or HDR formats)
        match self.pixel_format {
            PixelFormat::Rgb
            | PixelFormat::Rgba
            | PixelFormat::Bgr
            | PixelFormat::Bgra
            | PixelFormat::Bgrx => {}
            PixelFormat::Gray | PixelFormat::Cmyk => {
                return Err(crate::error::Error::UnsupportedFeature {
                    feature: "XYB mode only supports RGB/RGBA pixel formats",
                });
            }
            PixelFormat::Gray16
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
            | PixelFormat::GrayF32
            | PixelFormat::RgbF32
            | PixelFormat::RgbaF32 => {
                return Err(crate::error::Error::UnsupportedFeature {
                    feature: "XYB mode with 16-bit/float formats not yet supported (use YCbCr mode)",
                });
            }
        }

        // Convert RGB to scaled XYB (scalar path for now)
        // XYB values are stored as:
        // - X in y_strip with padded stride
        // - Y in cb_strip with packed stride
        // - B in cr_strip with packed stride
        for row in 0..strip_height {
            let y_row_start = row * padded_width;
            let cbcr_row_start = row * width;

            for x in 0..width {
                let src_idx = (row * width + x) * bpp;
                let (r, g, b) = match self.pixel_format {
                    PixelFormat::Rgb | PixelFormat::Rgba => (
                        rgb_strip[src_idx],
                        rgb_strip[src_idx + 1],
                        rgb_strip[src_idx + 2],
                    ),
                    PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx => (
                        rgb_strip[src_idx + 2],
                        rgb_strip[src_idx + 1],
                        rgb_strip[src_idx],
                    ),
                    _ => unreachable!(),
                };

                // Convert sRGB to scaled XYB
                let (scaled_x, scaled_y, scaled_b) = srgb_to_scaled_xyb(r, g, b);

                // Store: X→y_strip, Y→cb_strip, B→cr_strip
                // Scale to JPEG sample range (multiply by 255) for level shift consistency
                self.y_strip[y_row_start + x] = scaled_x * 255.0;
                self.cb_strip[cbcr_row_start + x] = scaled_y * 255.0;
                self.cr_strip[cbcr_row_start + x] = scaled_b * 255.0;
            }

            // Edge-pad X (y_strip) row
            if width < padded_width {
                let edge_val = self.y_strip[y_row_start + width - 1];
                for x in width..padded_width {
                    self.y_strip[y_row_start + x] = edge_val;
                }
            }
        }

        // For XYB mode, we handle the components differently:
        // - X is in y_strip (full res, already padded)
        // - Y is in cb_strip (full res, needs to stay there for DCT)
        // - B needs 2x2 downsampling (cr_strip → cr_down)
        //
        // Note: The DCT step will need to handle XYB's component structure specially
        // since Y (cb_strip) is full resolution unlike standard chroma.

        // Downsample B channel (cr_strip → cr_down) using 2x2 box filter
        let b_width = (width + 1) / 2;
        let b_height = (strip_height + 1) / 2;
        crate::encode_simd::downsample_2x2_simd_inplace(
            &self.cr_strip[..strip_height * width],
            width,
            strip_height,
            &mut self.cr_down[..b_width * b_height],
        );

        // Rearrange and pad cr_down (B channel)
        self.pad_chroma_down_strip(b_height, b_width);

        // For Y component (cb_strip): rearrange to padded layout directly
        // We'll use cb_strip as the source for DCT in XYB mode
        if padded_width > width {
            for row in (0..strip_height).rev() {
                let src_start = row * width;
                let dst_start = row * padded_width;
                for x in (0..width).rev() {
                    self.cb_strip[dst_start + x] = self.cb_strip[src_start + x];
                }
                let edge_val = self.cb_strip[dst_start + width - 1];
                for x in width..padded_width {
                    self.cb_strip[dst_start + x] = edge_val;
                }
            }
        }

        Ok(())
    }

    /// Converts RGB strip to YCbCr using gamma-aware chroma downsampling.
    ///
    /// This computes Y at full resolution and Cb/Cr directly at the downsampled
    /// resolution using gamma-aware averaging in linear RGB space.
    fn convert_strip_gamma_aware(
        &mut self,
        rgb_strip: &[u8],
        strip_y: usize,
        strip_height: usize,
    ) -> Result<()> {
        let width = self.width;
        let bpp = self.pixel_format.bytes_per_pixel();
        let use_iterative = self.chroma_downsampling == ChromaDownsampling::GammaAwareIterative;

        // Determine chroma strip dimensions
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, (strip_height + 1) / 2),
            Subsampling::S444 => {
                // No downsampling needed for 4:4:4, use standard path
                return self.convert_strip_to_ycbcr(rgb_strip, strip_height);
            }
        };

        let num_pixels = strip_height * width;
        let c_size = c_width * c_strip_height;

        match self.subsampling {
            Subsampling::S420 => {
                crate::chroma::gamma_aware_strip_420(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    strip_y,
                    self.height,
                    bpp,
                    use_iterative,
                );
            }
            Subsampling::S422 => {
                crate::chroma::gamma_aware_strip_422(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                    use_iterative,
                );
            }
            Subsampling::S440 => {
                crate::chroma::gamma_aware_strip_440(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                    use_iterative,
                );
            }
            Subsampling::S444 => unreachable!(), // Handled above
        }

        // Rearrange Y strip from packed to padded layout
        self.rearrange_y_strip_only(strip_height);

        // Pad chroma strips (cb_down, cr_down are already at downsampled resolution)
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
    }

    /// Converts RGB strip using fast fused Box downsampling.
    ///
    /// This computes Y at full resolution and Cb/Cr directly at the downsampled
    /// resolution using simple box averaging (no gamma correction).
    /// Faster than separate convert + downsample steps.
    fn convert_strip_box_fused(&mut self, rgb_strip: &[u8], strip_height: usize) -> Result<()> {
        let width = self.width;
        let bpp = self.pixel_format.bytes_per_pixel();
        let num_pixels = strip_height * width;

        // Determine chroma strip dimensions
        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, (strip_height + 1) / 2),
            Subsampling::S444 => {
                // No downsampling needed for 4:4:4, use standard path
                return self.convert_strip_to_ycbcr(rgb_strip, strip_height);
            }
        };

        let c_size = c_width * c_strip_height;

        match self.subsampling {
            Subsampling::S420 => {
                crate::chroma::box_fused_strip_420(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                );
            }
            Subsampling::S422 => {
                crate::chroma::box_fused_strip_422(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                );
            }
            Subsampling::S440 => {
                crate::chroma::box_fused_strip_440(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_down[..c_size],
                    &mut self.cr_down[..c_size],
                    width,
                    strip_height,
                    bpp,
                );
            }
            Subsampling::S444 => unreachable!(), // Handled above
        }

        // Rearrange Y strip from packed to padded layout
        self.rearrange_y_strip_only(strip_height);

        // Pad chroma strips (cb_down, cr_down are already at downsampled resolution)
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
    }

    /// Rearranges only the Y strip from packed to padded layout.
    /// Used by gamma-aware conversion where Cb/Cr go directly to cb_down/cr_down.
    fn rearrange_y_strip_only(&mut self, strip_height: usize) {
        let width = self.width;
        let padded_width = self.padded_width;

        if padded_width == width {
            return;
        }

        for row in (0..strip_height).rev() {
            let src_start = row * width;
            let dst_start = row * padded_width;

            for x in (0..width).rev() {
                self.y_strip[dst_start + x] = self.y_strip[src_start + x];
            }

            let edge_val = self.y_strip[dst_start + width - 1];
            for x in width..padded_width {
                self.y_strip[dst_start + x] = edge_val;
            }
        }
    }

    /// Pads strips vertically by replicating the last valid row.
    ///
    /// This is needed for the bottom strip when it has fewer rows than strip_height.
    /// Called after color conversion and horizontal padding.
    fn pad_strips_vertically(&mut self, actual_height: usize, target_height: usize) {
        if actual_height >= target_height {
            return;
        }

        let padded_width = self.padded_width;
        let is_color = self.pixel_format != PixelFormat::Gray;

        // Get last valid row index
        let last_row = actual_height - 1;
        let src_start = last_row * padded_width;

        // Replicate to all remaining rows
        for row in actual_height..target_height {
            let dst_start = row * padded_width;
            self.y_strip
                .copy_within(src_start..src_start + padded_width, dst_start);
        }

        if is_color {
            // For cb_strip/cr_strip (if they're in padded layout)
            // Note: these are still in packed layout at this point
            let width = self.width;
            let last_src = last_row * width;
            for row in actual_height..target_height {
                let dst = row * width;
                self.cb_strip.copy_within(last_src..last_src + width, dst);
                self.cr_strip.copy_within(last_src..last_src + width, dst);
            }
        }
    }

    /// Pads chroma down strips (cb_down, cr_down) horizontally.
    fn pad_chroma_down_strip(&mut self, c_strip_height: usize, c_width: usize) {
        let padded_c_width = self.padded_c_width;

        if padded_c_width == c_width {
            return;
        }

        // Rearrange and pad cb_down
        for row in (0..c_strip_height).rev() {
            let src_start = row * c_width;
            let dst_start = row * padded_c_width;

            for x in (0..c_width).rev() {
                self.cb_down[dst_start + x] = self.cb_down[src_start + x];
                self.cr_down[dst_start + x] = self.cr_down[src_start + x];
            }

            let cb_edge = self.cb_down[dst_start + c_width - 1];
            let cr_edge = self.cr_down[dst_start + c_width - 1];
            for x in c_width..padded_c_width {
                self.cb_down[dst_start + x] = cb_edge;
                self.cr_down[dst_start + x] = cr_edge;
            }
        }
    }

    /// Downsamples chroma strips according to subsampling mode.
    ///
    /// Uses SIMD downsampling for floating-point parity with full-plane encoder.
    /// Input cb_strip/cr_strip are in packed layout (width pixels per row).
    /// Output cb_down/cr_down are rearranged to padded layout.
    fn downsample_chroma_strip(&mut self, strip_height: usize) -> Result<()> {
        let width = self.width;
        let num_pixels = strip_height * width;

        let (c_width, c_strip_height) = match self.subsampling {
            Subsampling::S420 => {
                // 2×2 box filter using SIMD
                let c_width = (width + 1) / 2;
                let c_height = (strip_height + 1) / 2;
                let c_size = c_width * c_height;

                crate::encode_simd::downsample_2x2_simd_inplace(
                    &self.cb_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cb_down[..c_size],
                );
                crate::encode_simd::downsample_2x2_simd_inplace(
                    &self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cr_down[..c_size],
                );
                (c_width, c_height)
            }
            Subsampling::S422 => {
                // 2×1 horizontal filter using SIMD
                let c_width = (width + 1) / 2;
                let c_size = c_width * strip_height;

                crate::encode_simd::downsample_2x1_simd_inplace(
                    &self.cb_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cb_down[..c_size],
                );
                crate::encode_simd::downsample_2x1_simd_inplace(
                    &self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cr_down[..c_size],
                );
                (c_width, strip_height)
            }
            Subsampling::S440 => {
                // 1×2 vertical filter using SIMD
                let c_height = (strip_height + 1) / 2;
                let c_size = width * c_height;

                crate::encode_simd::downsample_1x2_simd_inplace(
                    &self.cb_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cb_down[..c_size],
                );
                crate::encode_simd::downsample_1x2_simd_inplace(
                    &self.cr_strip[..num_pixels],
                    width,
                    strip_height,
                    &mut self.cr_down[..c_size],
                );
                (width, c_height)
            }
            Subsampling::S444 => {
                // No downsampling - copy directly
                self.cb_down[..num_pixels].copy_from_slice(&self.cb_strip[..num_pixels]);
                self.cr_down[..num_pixels].copy_from_slice(&self.cr_strip[..num_pixels]);
                (width, strip_height)
            }
        };

        // Rearrange cb_down/cr_down to padded layout for DCT block extraction
        self.pad_chroma_down_strip(c_strip_height, c_width);

        Ok(())
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

        let mut blocks_added = 0;
        // y_strip is in padded layout, so use padded_width for sizing
        let y_size = strip_height * padded_width;

        // Compute DCT for Y blocks into pending buffer
        for local_by in 0..strip_blocks_h {
            let global_by = start_block_y + local_by;
            if global_by >= (height + 7) / 8 {
                break;
            }

            for bx in 0..blocks_w {
                // Extract 8×8 block from Y strip and DCT (wide-native path)
                let block = extract_block_from_strip_wide(
                    &self.y_strip[..y_size],
                    bx,
                    local_by,
                    padded_width,
                );
                let dct = crate::dct::simd::forward_dct_8x8_wide(&block);
                self.pending_y_blocks[pending_idx].push(dct);

                blocks_added += 1;
            }
        }

        // Compute DCT for Cb/Cr blocks (if color)
        if self.pixel_format != PixelFormat::Gray {
            let width = self.width;

            if self.use_xyb {
                // XYB mode: Y component is full resolution, B is 2x2 downsampled
                // - Y (cb_strip): full res, same block dimensions as X (y_strip)
                // - B (cr_down): 2x2 downsampled

                // Process Y component (full res from cb_strip which is now in padded layout)
                let y_size = strip_height * padded_width;
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
                        let cb_dct = crate::dct::simd::forward_dct_8x8_wide(&cb_block);
                        self.pending_cb_blocks[pending_idx].push(cb_dct);
                    }
                }

                // Process B component (2x2 downsampled from cr_down)
                // For XYB, B is always 2x2 downsampled, so use c_width/c_height based on that
                let b_width = (width + 1) / 2;
                let b_strip_height = (strip_height + 1) / 2;
                let b_blocks_w = (b_width + 7) / 8;
                let b_strip_blocks_h = (b_strip_height + 7) / 8;
                let padded_c_width = self.padded_c_width;

                // cr_down is sized for c_strip_height rows based on the subsampling mode,
                // but for XYB we compute b_strip_height separately. Use the full cr_down size.
                let c_strip_height_for_buf = match self.subsampling {
                    Subsampling::S420 | Subsampling::S440 => strip_height / 2,
                    _ => strip_height,
                };
                let c_size = c_strip_height_for_buf * padded_c_width;

                for local_by in 0..b_strip_blocks_h {
                    for bx in 0..b_blocks_w {
                        let cr_block = extract_block_from_strip_wide(
                            &self.cr_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        let cr_dct = crate::dct::simd::forward_dct_8x8_wide(&cr_block);
                        self.pending_cr_blocks[pending_idx].push(cr_dct);
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

                // cb_down/cr_down are in padded layout (padded_c_width pixels per row)
                let padded_c_width = self.padded_c_width;
                let c_size = c_strip_height * padded_c_width;

                for local_by in 0..c_strip_blocks_h {
                    for bx in 0..c_blocks_w {
                        // Cb block - DCT only (wide-native path)
                        let cb_block = extract_block_from_strip_wide(
                            &self.cb_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        let cb_dct = crate::dct::simd::forward_dct_8x8_wide(&cb_block);
                        self.pending_cb_blocks[pending_idx].push(cb_dct);

                        // Cr block - DCT only (wide-native path)
                        let cr_block = extract_block_from_strip_wide(
                            &self.cr_down[..c_size],
                            bx,
                            local_by,
                            padded_c_width,
                        );
                        let cr_dct = crate::dct::simd::forward_dct_8x8_wide(&cr_block);
                        self.pending_cr_blocks[pending_idx].push(cr_dct);
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
    fn quantize_pending_imcu(&mut self, buffer_idx: usize, aq_strengths: &[f32]) {
        // Get quantization context (must be set before processing)
        let quant = self.quant.clone().expect("quant context not set");

        // Quantize Y blocks
        for (i, dct) in self.pending_y_blocks[buffer_idx].iter().enumerate() {
            let aq_strength = if i < aq_strengths.len() {
                aq_strengths[i]
            } else {
                0.08 // C++ mean fallback
            };

            // Fused quantization + zigzag reorder (single pass, no intermediate array)
            let zigzag = quant.y_quant_simd.quantize_with_zero_bias_zigzag(
                dct,
                &quant.y_zero_bias_simd,
                aq_strength,
            );

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

            // Quantize Cb blocks
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

                let zigzag = quant.cb_quant_simd.quantize_with_zero_bias_zigzag(
                    dct,
                    &quant.cb_zero_bias_simd,
                    aq_strength,
                );
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

                let zigzag = quant.cr_quant_simd.quantize_with_zero_bias_zigzag(
                    dct,
                    &quant.cr_zero_bias_simd,
                    aq_strength,
                );
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
        use crate::encode::Encoder;
        use crate::quant::Quality;

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
                        (200, 100, 80),   // coral
                        (80, 180, 100),   // green
                        (100, 80, 180),   // purple
                        (180, 180, 80),   // yellow
                        (80, 150, 180),   // cyan
                        (180, 80, 150),   // magenta
                        (140, 140, 140),  // gray
                        (220, 180, 140),  // tan
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
            let jpeg_strip = Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .pixel_format(PixelFormat::Rgb)
                .subsampling(Subsampling::S420)
                .jpegli_quality(Quality::from_quality(85.0))
                .optimize_huffman(true)
                .encode(&rgb)
                .expect("strip encode failed");

            // Decode and verify
            let decoded_strip = crate::Decoder::new()
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
