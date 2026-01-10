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
//!    - Count Huffman frequencies
//!    - Append to final i16 storage
//! 6. Swap pending buffers
//!
//! After all strips:
//! 1. Flush last iMCU (quantize remaining pending blocks)
//! 2. Build optimized Huffman tables
//! 3. Encode from stored i16 blocks

use crate::alloc::{try_alloc_zeroed_f32_tracked, try_with_capacity_tracked, AllocationStats};
use crate::consts::DCT_BLOCK_SIZE;
use crate::dct::forward_dct_8x8;
use crate::error::Result;
use crate::huffman::optimize::FrequencyCounter;
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::{QuantTable, ZeroBiasParams};
use crate::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::types::{ChromaDownsampling, PixelFormat, Subsampling};

use super::natural_to_zigzag_into;

/// Extracts an 8×8 block from a strip buffer with level shift (free function to avoid borrow issues).
///
/// Applies JPEG level shift (-128) to convert from [0, 255] to [-128, 127] range
/// as required by the DCT.
fn extract_block_from_strip(
    strip: &[f32],
    bx: usize,
    local_by: usize,
    strip_width: usize,
) -> [f32; DCT_BLOCK_SIZE] {
    let mut block = [0.0f32; DCT_BLOCK_SIZE];
    let x_start = bx * 8;
    let y_start = local_by * 8;

    for dy in 0..8 {
        let y = y_start + dy;
        if y * strip_width >= strip.len() {
            // Past end of strip, fill with edge replication
            let last_y = (strip.len() / strip_width).saturating_sub(1);
            for dx in 0..8 {
                let x = (x_start + dx).min(strip_width.saturating_sub(1));
                // Level shift: subtract 128 (values are in [0, 255])
                block[dy * 8 + dx] = strip[last_y * strip_width + x] - 128.0;
            }
        } else {
            for dx in 0..8 {
                let x = (x_start + dx).min(strip_width.saturating_sub(1));
                // Level shift: subtract 128 (values are in [0, 255])
                block[dy * 8 + dx] = strip[y * strip_width + x] - 128.0;
            }
        }
    }

    block
}

// StreamingAQ uses rolling buffers for low memory (~2.5 MB for 4K vs 33 MB).

/// Strip-based encoder for low-memory JPEG encoding.
///
/// Processes the image in horizontal strips to avoid materializing
/// full f32 planes in memory.
#[derive(Debug)]
pub struct StripProcessor {
    /// Image width in pixels
    width: usize,
    /// Image height in pixels
    height: usize,
    /// Strip height in pixels (16 for 4:2:0, 8 for 4:4:4)
    strip_height: usize,
    /// Chroma subsampling mode
    subsampling: Subsampling,
    /// Pixel format of input data
    pixel_format: PixelFormat,
    /// Chroma downsampling method (Box, GammaAware, GammaAwareIterative)
    chroma_downsampling: ChromaDownsampling,

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

    // === Block scratch space ===
    /// DCT buffer (reused per block)
    dct_buf: [f32; DCT_BLOCK_SIZE],

    // === Final quantized block storage (i16) ===
    /// Y channel quantized blocks (zigzag order)
    y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cb channel quantized blocks
    cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cr channel quantized blocks
    cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,

    // === Pending iMCU DCT blocks (f32, double-buffered) ===
    // These hold raw DCT coefficients until AQ strengths are available
    // Double-buffered: [current] and [previous pending quantization]
    pending_y_blocks: [Vec<[f32; DCT_BLOCK_SIZE]>; 2],
    pending_cb_blocks: [Vec<[f32; DCT_BLOCK_SIZE]>; 2],
    pending_cr_blocks: [Vec<[f32; DCT_BLOCK_SIZE]>; 2],
    /// Index of current pending buffer (0 or 1)
    pending_current: usize,

    // === SIMD quantization tables (initialized with quant tables) ===
    y_quant_simd: Option<QuantTableSimd>,
    cb_quant_simd: Option<QuantTableSimd>,
    cr_quant_simd: Option<QuantTableSimd>,
    y_zero_bias_simd: Option<ZeroBiasSimd>,
    cb_zero_bias_simd: Option<ZeroBiasSimd>,
    cr_zero_bias_simd: Option<ZeroBiasSimd>,

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

    // === Huffman frequency accumulators ===
    /// DC luma frequency counter
    dc_luma_freq: FrequencyCounter,
    /// AC luma frequency counter
    ac_luma_freq: FrequencyCounter,
    /// DC chroma frequency counter
    dc_chroma_freq: FrequencyCounter,
    /// AC chroma frequency counter
    ac_chroma_freq: FrequencyCounter,

    // === Quantization parameters (set before processing) ===
    y_quant: Option<QuantTable>,
    cb_quant: Option<QuantTable>,
    cr_quant: Option<QuantTable>,
    y_zero_bias: Option<ZeroBiasParams>,
    cb_zero_bias: Option<ZeroBiasParams>,
    cr_zero_bias: Option<ZeroBiasParams>,

    // === Allocation tracking ===
    /// Tracks all allocations made by this processor
    alloc_stats: crate::alloc::AllocationStats,

    /// Restart interval in MCUs (0 = disabled)
    restart_interval: u16,
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
    /// * `restart_interval` - Restart interval in MCUs (0 = disabled)
    pub fn with_options(
        width: usize,
        height: usize,
        subsampling: Subsampling,
        pixel_format: PixelFormat,
        chroma_downsampling: ChromaDownsampling,
        restart_interval: u16,
    ) -> Result<Self> {
        // Strip height is 16 for 4:2:0 (2 MCU rows), 8 otherwise
        let strip_height = match subsampling {
            Subsampling::S420 | Subsampling::S440 => 16,
            _ => 8,
        };

        // Chroma dimensions
        let (c_width, c_strip_height) = match subsampling {
            Subsampling::S420 => ((width + 1) / 2, strip_height / 2),
            Subsampling::S422 => ((width + 1) / 2, strip_height),
            Subsampling::S440 => (width, strip_height / 2),
            Subsampling::S444 => (width, strip_height),
        };

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
        // For 4:2:0: 2 block rows of Y, 1 block row each of Cb/Cr
        let v_samp = match subsampling {
            Subsampling::S420 | Subsampling::S440 => 2,
            _ => 1,
        };
        let pending_y_capacity = y_blocks_h * v_samp;
        let pending_c_capacity = c_blocks_h; // One chroma block row per iMCU

        let is_color = pixel_format != PixelFormat::Gray;

        // Track all allocations
        let mut alloc_stats = AllocationStats::new();

        Ok(Self {
            width,
            height,
            strip_height,
            subsampling,
            pixel_format,
            chroma_downsampling,
            restart_interval,

            // Strip buffers (sized for one strip)
            y_strip: try_alloc_zeroed_f32_tracked(
                width * strip_height,
                "y_strip",
                &mut alloc_stats,
            )?,
            cb_strip: if is_color {
                try_alloc_zeroed_f32_tracked(width * strip_height, "cb_strip", &mut alloc_stats)?
            } else {
                Vec::new()
            },
            cr_strip: if is_color {
                try_alloc_zeroed_f32_tracked(width * strip_height, "cr_strip", &mut alloc_stats)?
            } else {
                Vec::new()
            },
            cb_down: if is_color {
                try_alloc_zeroed_f32_tracked(c_width * c_strip_height, "cb_down", &mut alloc_stats)?
            } else {
                Vec::new()
            },
            cr_down: if is_color {
                try_alloc_zeroed_f32_tracked(c_width * c_strip_height, "cr_down", &mut alloc_stats)?
            } else {
                Vec::new()
            },

            // Scratch space
            dct_buf: [0.0f32; DCT_BLOCK_SIZE],

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
                Vec::with_capacity(pending_y_capacity),
                Vec::with_capacity(pending_y_capacity),
            ],
            pending_cb_blocks: if is_color {
                [
                    Vec::with_capacity(pending_c_capacity),
                    Vec::with_capacity(pending_c_capacity),
                ]
            } else {
                [Vec::new(), Vec::new()]
            },
            pending_cr_blocks: if is_color {
                [
                    Vec::with_capacity(pending_c_capacity),
                    Vec::with_capacity(pending_c_capacity),
                ]
            } else {
                [Vec::new(), Vec::new()]
            },
            pending_current: 0,

            // SIMD quant tables (initialized in set_quant_tables)
            y_quant_simd: None,
            cb_quant_simd: None,
            cr_quant_simd: None,
            y_zero_bias_simd: None,
            cb_zero_bias_simd: None,
            cr_zero_bias_simd: None,

            // Block dimensions for chroma AQ mapping
            y_blocks_h,
            y_blocks_v,
            c_blocks_h,
            c_blocks_v,

            // Accumulated AQ strengths (for output)
            all_aq_strengths: Vec::with_capacity(total_y_blocks),

            // Streaming AQ (initialized when quant tables are set)
            aq_state: None,

            // Huffman counters
            dc_luma_freq: FrequencyCounter::new(),
            ac_luma_freq: FrequencyCounter::new(),
            dc_chroma_freq: FrequencyCounter::new(),
            ac_chroma_freq: FrequencyCounter::new(),

            // Quant tables (set later)
            y_quant: None,
            cb_quant: None,
            cr_quant: None,
            y_zero_bias: None,
            cb_zero_bias: None,
            cr_zero_bias: None,

            // Allocation tracking
            alloc_stats,
        })
    }

    /// Returns allocation statistics for this processor.
    #[must_use]
    pub fn allocation_stats(&self) -> &AllocationStats {
        &self.alloc_stats
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
        self.aq_state = Some(StreamingAQ::new(
            self.width,
            self.height,
            y_quant_01,
            v_samp,
        )?);

        // Initialize SIMD quant tables for incremental quantization
        self.y_quant_simd = Some(QuantTableSimd::from_values(&y_quant.values));
        self.cb_quant_simd = Some(QuantTableSimd::from_values(&cb_quant.values));
        self.cr_quant_simd = Some(QuantTableSimd::from_values(&cr_quant.values));
        self.y_zero_bias_simd = Some(ZeroBiasSimd::from_params(&y_zero_bias));
        self.cb_zero_bias_simd = Some(ZeroBiasSimd::from_params(&cb_zero_bias));
        self.cr_zero_bias_simd = Some(ZeroBiasSimd::from_params(&cr_zero_bias));

        self.y_quant = Some(y_quant);
        self.cb_quant = Some(cb_quant);
        self.cr_quant = Some(cr_quant);
        self.y_zero_bias = Some(y_zero_bias);
        self.cb_zero_bias = Some(cb_zero_bias);
        self.cr_zero_bias = Some(cr_zero_bias);
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

        // Step 1: Color convert RGB → YCbCr into strip buffers
        // For gamma-aware modes, this computes chroma directly at downsampled resolution
        if self.chroma_downsampling.uses_gamma_aware() && self.pixel_format != PixelFormat::Gray {
            self.convert_strip_gamma_aware(rgb_strip, strip_y, actual_strip_height)?;
        } else {
            self.convert_strip_to_ycbcr(rgb_strip, actual_strip_height)?;
        }

        // Step 2: Process AQ and check if previous iMCU strengths are ready
        let aq_strengths = if let Some(ref mut aq) = self.aq_state {
            aq.process_y_strip(&self.y_strip, strip_y, actual_strip_height)
                .map(|s| s.to_vec())
        } else {
            None
        };

        // Step 3: Downsample chroma if needed (skipped for gamma-aware modes)
        if self.pixel_format != PixelFormat::Gray && !self.chroma_downsampling.uses_gamma_aware() {
            self.downsample_chroma_strip(actual_strip_height)?;
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
        let blocks_added = self.dct_strip_blocks_to_pending(strip_y, actual_strip_height)?;

        // Step 6: Swap pending buffers when iMCU completes
        // (The swap happens via pending_current tracking in dct_strip_blocks_to_pending)

        Ok(blocks_added)
    }

    /// Converts RGB strip data to YCbCr in the strip buffers.
    ///
    /// Uses SIMD conversion for floating-point parity with full-plane encoder.
    fn convert_strip_to_ycbcr(&mut self, rgb_strip: &[u8], strip_height: usize) -> Result<()> {
        let num_pixels = strip_height * self.width;

        // Use the same SIMD conversion as the full-plane encoder for exact floating-point parity
        match self.pixel_format {
            PixelFormat::Rgb => {
                crate::encode_simd::rgb_to_ycbcr_planes_simd_inplace(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    num_pixels,
                );
            }
            PixelFormat::Rgba => {
                crate::encode_simd::rgba_to_ycbcr_planes_simd_inplace(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    num_pixels,
                );
            }
            PixelFormat::Bgr => {
                crate::encode_simd::bgr_to_ycbcr_planes_simd_inplace(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    num_pixels,
                );
            }
            PixelFormat::Bgra => {
                crate::encode_simd::bgra_to_ycbcr_planes_simd_inplace(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    num_pixels,
                );
            }
            PixelFormat::Gray => {
                crate::encode_simd::gray_to_ycbcr_planes_simd_inplace(
                    rgb_strip,
                    &mut self.y_strip[..num_pixels],
                    &mut self.cb_strip[..num_pixels],
                    &mut self.cr_strip[..num_pixels],
                    num_pixels,
                );
            }
            PixelFormat::Cmyk => {
                // CMYK requires scalar conversion (rare format)
                let bpp = self.pixel_format.bytes_per_pixel();
                for y in 0..strip_height {
                    for x in 0..self.width {
                        let idx = (y * self.width + x) * bpp;
                        let plane_idx = y * self.width + x;

                        let c = rgb_strip[idx] as f32 / 255.0;
                        let m = rgb_strip[idx + 1] as f32 / 255.0;
                        let y_val = rgb_strip[idx + 2] as f32 / 255.0;
                        let k = rgb_strip[idx + 3] as f32 / 255.0;
                        let r = 255.0 * (1.0 - c) * (1.0 - k);
                        let g = 255.0 * (1.0 - m) * (1.0 - k);
                        let b = 255.0 * (1.0 - y_val) * (1.0 - k);

                        // Use constants for consistency
                        use crate::consts::{
                            YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB,
                            YCBCR_G_TO_CR, YCBCR_G_TO_Y, YCBCR_R_TO_CB, YCBCR_R_TO_CR,
                            YCBCR_R_TO_Y,
                        };
                        self.y_strip[plane_idx] =
                            YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
                        self.cb_strip[plane_idx] =
                            128.0 + YCBCR_R_TO_CB * r + YCBCR_G_TO_CB * g + YCBCR_B_TO_CB * b;
                        self.cr_strip[plane_idx] =
                            128.0 + YCBCR_R_TO_CR * r + YCBCR_G_TO_CR * g + YCBCR_B_TO_CR * b;
                    }
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

        Ok(())
    }

    /// Downsamples chroma strips according to subsampling mode.
    ///
    /// Uses SIMD downsampling for floating-point parity with full-plane encoder.
    fn downsample_chroma_strip(&mut self, strip_height: usize) -> Result<()> {
        let width = self.width;
        let num_pixels = strip_height * width;

        match self.subsampling {
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
            }
            Subsampling::S444 => {
                // No downsampling - copy directly
                self.cb_down[..num_pixels].copy_from_slice(&self.cb_strip[..num_pixels]);
                self.cr_down[..num_pixels].copy_from_slice(&self.cr_strip[..num_pixels]);
            }
        }

        Ok(())
    }

    /// Computes DCT for blocks in the current strip and stores in pending buffer.
    /// This allows quantization to happen incrementally when AQ strengths become available.
    fn dct_strip_blocks_to_pending(
        &mut self,
        strip_y: usize,
        strip_height: usize,
    ) -> Result<usize> {
        let blocks_w = (self.width + 7) / 8;
        let strip_blocks_h = (strip_height + 7) / 8;
        let start_block_y = strip_y / 8;
        let width = self.width;
        let height = self.height;
        let pending_idx = self.pending_current;

        let mut blocks_added = 0;
        // Only pass the valid portion of the Y buffer for correct edge detection
        let y_size = strip_height * width;

        // Compute DCT for Y blocks into pending buffer
        for local_by in 0..strip_blocks_h {
            let global_by = start_block_y + local_by;
            if global_by >= (height + 7) / 8 {
                break;
            }

            for bx in 0..blocks_w {
                // Extract 8×8 block from Y strip (use valid portion of buffer)
                let block = extract_block_from_strip(&self.y_strip[..y_size], bx, local_by, width);

                // DCT - store raw coefficients in pending buffer
                let dct = forward_dct_8x8(&block);
                self.pending_y_blocks[pending_idx].push(dct);

                blocks_added += 1;
            }
        }

        // Compute DCT for Cb/Cr blocks (if color)
        if self.pixel_format != PixelFormat::Gray {
            // Use ceiling division for chroma dimensions to handle partial strips correctly
            let (c_width, c_strip_height) = match self.subsampling {
                Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
                Subsampling::S422 => ((width + 1) / 2, strip_height),
                Subsampling::S440 => (width, (strip_height + 1) / 2),
                Subsampling::S444 => (width, strip_height),
            };

            let c_blocks_w = (c_width + 7) / 8;
            let c_strip_blocks_h = (c_strip_height + 7) / 8;
            // Only pass the valid portion of the buffer to extract_block_from_strip,
            // so edge detection works correctly for partial strips.
            let c_size = c_width * c_strip_height;

            for local_by in 0..c_strip_blocks_h {
                for bx in 0..c_blocks_w {
                    // Cb block - DCT only (use valid portion of buffer)
                    let cb_block =
                        extract_block_from_strip(&self.cb_down[..c_size], bx, local_by, c_width);
                    let cb_dct = forward_dct_8x8(&cb_block);
                    self.pending_cb_blocks[pending_idx].push(cb_dct);

                    // Cr block - DCT only (use valid portion of buffer)
                    let cr_block =
                        extract_block_from_strip(&self.cr_down[..c_size], bx, local_by, c_width);
                    let cr_dct = forward_dct_8x8(&cr_block);
                    self.pending_cr_blocks[pending_idx].push(cr_dct);
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
        // Clone SIMD tables to avoid borrow issues when calling count_block_frequencies
        let y_quant_simd = self
            .y_quant_simd
            .clone()
            .expect("y_quant_simd not set");
        let y_zero_bias_simd = self
            .y_zero_bias_simd
            .clone()
            .expect("y_zero_bias_simd not set");

        // Quantize Y blocks - first pass: just quantize
        let start_y_idx = self.y_blocks.len();
        let mut y_quantized = Vec::with_capacity(self.pending_y_blocks[buffer_idx].len());

        for (i, dct) in self.pending_y_blocks[buffer_idx].iter().enumerate() {
            let aq_strength = if i < aq_strengths.len() {
                aq_strengths[i]
            } else {
                0.08 // C++ mean fallback
            };

            // SIMD quantization for parity with full-plane encoder
            let quant_coeffs =
                y_quant_simd.quantize_array_with_zero_bias(dct, &y_zero_bias_simd, aq_strength);

            // Convert to zigzag order
            let mut zigzag = [0i16; DCT_BLOCK_SIZE];
            natural_to_zigzag_into(&quant_coeffs, &mut zigzag);

            y_quantized.push((zigzag, aq_strength));
        }

        // Second pass: store and count frequencies
        for (zigzag, aq_strength) in y_quantized {
            self.y_blocks.push(zigzag);
            self.count_block_frequencies(&zigzag, true);
            self.all_aq_strengths.push(aq_strength);
        }

        // Quantize Cb/Cr blocks
        let cb_quant_simd = self.cb_quant_simd.clone();
        let cr_quant_simd = self.cr_quant_simd.clone();
        let cb_zero_bias_simd = self.cb_zero_bias_simd.clone();
        let cr_zero_bias_simd = self.cr_zero_bias_simd.clone();

        if let (Some(cb_qs), Some(cr_qs), Some(cb_zbs), Some(cr_zbs)) = (
            cb_quant_simd,
            cr_quant_simd,
            cb_zero_bias_simd,
            cr_zero_bias_simd,
        ) {
            let y_blocks_h = self.y_blocks_h;
            let y_blocks_v = self.y_blocks_v;
            let c_blocks_h = self.c_blocks_h;
            let c_blocks_v = self.c_blocks_v;

            // Compute global chroma by from how many chroma blocks we've already processed
            // For 4:2:0, each iMCU has 1 chroma block row
            let global_chroma_by = self.cb_blocks.len() / c_blocks_h.max(1);

            // Quantize Cb blocks - first pass
            let mut cb_quantized = Vec::with_capacity(self.pending_cb_blocks[buffer_idx].len());
            for (i, dct) in self.pending_cb_blocks[buffer_idx].iter().enumerate() {
                let bx = i % c_blocks_h.max(1);
                let local_by = i / c_blocks_h.max(1);
                // Compute global Y position for this chroma block
                let y_bx = (bx * y_blocks_h) / c_blocks_h.max(1);
                let chroma_by = global_chroma_by + local_by;
                let y_by = (chroma_by * y_blocks_v) / c_blocks_v.max(1);
                // Use global AQ index
                let global_aq_idx =
                    y_by * y_blocks_h + y_bx.min(y_blocks_h.saturating_sub(1));
                let aq_strength = if global_aq_idx < self.all_aq_strengths.len() {
                    self.all_aq_strengths[global_aq_idx]
                } else {
                    0.08 // C++ mean fallback
                };

                let quant_coeffs = cb_qs.quantize_array_with_zero_bias(dct, &cb_zbs, aq_strength);
                let mut zigzag = [0i16; DCT_BLOCK_SIZE];
                natural_to_zigzag_into(&quant_coeffs, &mut zigzag);
                cb_quantized.push(zigzag);
            }

            // Second pass for Cb
            for zigzag in cb_quantized {
                self.cb_blocks.push(zigzag);
                self.count_block_frequencies(&zigzag, false);
            }

            // Quantize Cr blocks - first pass (use same global chroma by calculation)
            let global_chroma_by_cr = self.cr_blocks.len() / c_blocks_h.max(1);
            let mut cr_quantized = Vec::with_capacity(self.pending_cr_blocks[buffer_idx].len());
            for (i, dct) in self.pending_cr_blocks[buffer_idx].iter().enumerate() {
                let bx = i % c_blocks_h.max(1);
                let local_by = i / c_blocks_h.max(1);
                // Compute global Y position for this chroma block
                let y_bx = (bx * y_blocks_h) / c_blocks_h.max(1);
                let chroma_by = global_chroma_by_cr + local_by;
                let y_by = (chroma_by * y_blocks_v) / c_blocks_v.max(1);
                // Use global AQ index
                let global_aq_idx =
                    y_by * y_blocks_h + y_bx.min(y_blocks_h.saturating_sub(1));
                let aq_strength = if global_aq_idx < self.all_aq_strengths.len() {
                    self.all_aq_strengths[global_aq_idx]
                } else {
                    0.08 // C++ mean fallback
                };

                let quant_coeffs = cr_qs.quantize_array_with_zero_bias(dct, &cr_zbs, aq_strength);
                let mut zigzag = [0i16; DCT_BLOCK_SIZE];
                natural_to_zigzag_into(&quant_coeffs, &mut zigzag);
                cr_quantized.push(zigzag);
            }

            // Second pass for Cr
            for zigzag in cr_quantized {
                self.cr_blocks.push(zigzag);
                self.count_block_frequencies(&zigzag, false);
            }
        }
    }

    /// Counts Huffman frequencies for a quantized block.
    fn count_block_frequencies(&mut self, block: &[i16; DCT_BLOCK_SIZE], is_luma: bool) {
        // DC coefficient - encode as category
        let dc = block[0];
        let dc_cat = crate::entropy::category(dc);
        if is_luma {
            self.dc_luma_freq.count(dc_cat);
        } else {
            self.dc_chroma_freq.count(dc_cat);
        }

        // AC coefficients
        let mut run = 0u8;
        for i in 1..DCT_BLOCK_SIZE {
            let ac = block[i];
            if ac == 0 {
                run += 1;
                if run == 16 {
                    // ZRL symbol
                    if is_luma {
                        self.ac_luma_freq.count(0xF0);
                    } else {
                        self.ac_chroma_freq.count(0xF0);
                    }
                    run = 0;
                }
            } else {
                let cat = crate::entropy::category(ac);
                let symbol = (run << 4) | cat;
                if is_luma {
                    self.ac_luma_freq.count(symbol);
                } else {
                    self.ac_chroma_freq.count(symbol);
                }
                run = 0;
            }
        }

        // EOB if we have trailing zeros
        if run > 0 {
            if is_luma {
                self.ac_luma_freq.count(0x00);
            } else {
                self.ac_chroma_freq.count(0x00);
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
                let last_strengths = last_aq.to_vec();
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
            let default_aq = vec![0.08f32; self.pending_y_blocks[current_buffer].len()];
            self.quantize_pending_imcu(current_buffer, &default_aq);
        }

        Ok(StripProcessorOutput {
            y_blocks: self.y_blocks,
            cb_blocks: self.cb_blocks,
            cr_blocks: self.cr_blocks,
            aq_strengths: self.all_aq_strengths,
            dc_luma_freq: self.dc_luma_freq,
            ac_luma_freq: self.ac_luma_freq,
            dc_chroma_freq: self.dc_chroma_freq,
            ac_chroma_freq: self.ac_chroma_freq,
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
    /// DC luma Huffman frequencies
    pub dc_luma_freq: FrequencyCounter,
    /// AC luma Huffman frequencies
    pub ac_luma_freq: FrequencyCounter,
    /// DC chroma Huffman frequencies
    pub dc_chroma_freq: FrequencyCounter,
    /// AC chroma Huffman frequencies
    pub ac_chroma_freq: FrequencyCounter,
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
}
