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
//! Strip-based encoder peak memory:
//! - f32 strip buffers (reused): ~1 MB
//! - i16 quantized blocks: ~36 MB
//! - AQ accumulators: ~4 MB
//! - Total: ~47 MB target
//!
//! # Algorithm
//!
//! For each strip of 16 rows (2 MCU rows for 4:2:0):
//! 1. Convert RGB → YCbCr (f32 strips, reused)
//! 2. Accumulate AQ features for this strip
//! 3. Downsample chroma if needed
//! 4. DCT + quantize → append to i16 block storage
//! 5. Count Huffman frequencies
//! 6. Release strip buffers (reuse next iteration)
//!
//! After all strips:
//! 1. Finalize AQ map (global normalization)
//! 2. Build optimized Huffman tables
//! 3. Encode from stored i16 blocks

use crate::alloc::try_with_capacity;
use crate::consts::DCT_BLOCK_SIZE;
use crate::dct::forward_dct_8x8;
use crate::error::Result;
use crate::huffman::optimize::FrequencyCounter;
use crate::quant::aq::streaming::StreamingAQ;
use crate::quant::{QuantTable, ZeroBiasParams};
use crate::simd_types::{QuantTableSimd, ZeroBiasSimd};
use crate::types::{PixelFormat, Subsampling};

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

    // === Growing block storage (raw DCT coefficients) ===
    // Stored as f32 to allow quantization with per-block AQ at the end
    /// Y channel raw DCT blocks (natural order, not zigzag)
    y_blocks: Vec<[f32; DCT_BLOCK_SIZE]>,
    /// Cb channel raw DCT blocks
    cb_blocks: Vec<[f32; DCT_BLOCK_SIZE]>,
    /// Cr channel raw DCT blocks
    cr_blocks: Vec<[f32; DCT_BLOCK_SIZE]>,

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
}

impl StripProcessor {
    /// Creates a new strip processor.
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
        let blocks_w = (width + 7) / 8;
        let blocks_h = (height + 7) / 8;
        let total_y_blocks = blocks_w * blocks_h;
        let total_c_blocks = match subsampling {
            Subsampling::S420 => ((width + 15) / 16) * ((height + 15) / 16),
            Subsampling::S422 => ((width + 15) / 16) * blocks_h,
            Subsampling::S440 => blocks_w * ((height + 15) / 16),
            Subsampling::S444 => total_y_blocks,
        };

        let is_color = pixel_format != PixelFormat::Gray;

        Ok(Self {
            width,
            height,
            strip_height,
            subsampling,
            pixel_format,

            // Strip buffers (sized for one strip)
            y_strip: vec![0.0f32; width * strip_height],
            cb_strip: if is_color {
                vec![0.0f32; width * strip_height]
            } else {
                Vec::new()
            },
            cr_strip: if is_color {
                vec![0.0f32; width * strip_height]
            } else {
                Vec::new()
            },
            cb_down: if is_color {
                vec![0.0f32; c_width * c_strip_height]
            } else {
                Vec::new()
            },
            cr_down: if is_color {
                vec![0.0f32; c_width * c_strip_height]
            } else {
                Vec::new()
            },

            // Scratch space
            dct_buf: [0.0f32; DCT_BLOCK_SIZE],

            // Block storage (pre-allocated, stores raw DCT coefficients)
            y_blocks: try_with_capacity(total_y_blocks, "y_blocks")?,
            cb_blocks: if is_color {
                try_with_capacity(total_c_blocks, "cb_blocks")?
            } else {
                Vec::new()
            },
            cr_blocks: if is_color {
                try_with_capacity(total_c_blocks, "cr_blocks")?
            } else {
                Vec::new()
            },

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
        })
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
        self.aq_state = Some(StreamingAQ::new(self.width, self.height, y_quant_01, v_samp)?);

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
        self.convert_strip_to_ycbcr(rgb_strip, actual_strip_height)?;

        // Step 2: Accumulate AQ features from Y strip
        if let Some(ref mut aq) = self.aq_state {
            aq.process_y_strip(&self.y_strip, strip_y, actual_strip_height);
        }

        // Step 3: Downsample chroma if needed
        if self.pixel_format != PixelFormat::Gray {
            self.downsample_chroma_strip(actual_strip_height)?;
        }

        // Step 4: Compute DCT for blocks in this strip
        // Raw DCT coefficients are stored; quantization happens in finalize() with actual AQ values.
        let blocks_added = self.dct_strip_blocks(strip_y, actual_strip_height)?;

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

    /// Computes DCT for blocks in the current strip and stores raw coefficients.
    /// Quantization happens in finalize() with actual per-block AQ values.
    fn dct_strip_blocks(&mut self, strip_y: usize, strip_height: usize) -> Result<usize> {
        let blocks_w = (self.width + 7) / 8;
        let strip_blocks_h = (strip_height + 7) / 8;
        let start_block_y = strip_y / 8;
        let width = self.width;
        let height = self.height;

        let mut blocks_added = 0;
        // Only pass the valid portion of the Y buffer for correct edge detection
        let y_size = strip_height * width;

        // Compute DCT for Y blocks
        for local_by in 0..strip_blocks_h {
            let global_by = start_block_y + local_by;
            if global_by >= (height + 7) / 8 {
                break;
            }

            for bx in 0..blocks_w {
                // Extract 8×8 block from Y strip (use valid portion of buffer)
                let block =
                    extract_block_from_strip(&self.y_strip[..y_size], bx, local_by, width);

                // DCT - store raw coefficients (quantization happens in finalize)
                let dct = forward_dct_8x8(&block);
                self.y_blocks.push(dct);

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
                    self.cb_blocks.push(cb_dct);

                    // Cr block - DCT only (use valid portion of buffer)
                    let cr_block =
                        extract_block_from_strip(&self.cr_down[..c_size], bx, local_by, c_width);
                    let cr_dct = forward_dct_8x8(&cr_block);
                    self.cr_blocks.push(cr_dct);
                }
            }
        }

        Ok(blocks_added)
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
    /// Quantizes all stored raw DCT blocks with per-block AQ values,
    /// counts Huffman frequencies, and returns the quantized blocks.
    pub fn finalize(mut self) -> Result<StripProcessorOutput> {
        // Finalize AQ map (produces identical values to full-plane AQ)
        let aq_strengths = match self.aq_state.take() {
            Some(aq) => aq.finalize()?,
            None => Vec::new(), // AQ not initialized (no quant tables set)
        };

        // Get quant tables and zero bias params
        // Use SIMD types for parity with full-plane encoder (same floating-point operations)
        let y_quant = self.y_quant.as_ref().expect("y_quant not set");
        let y_quant_simd = QuantTableSimd::from_values(&y_quant.values);
        let y_zero_bias = self.y_zero_bias.clone().expect("y_zero_bias not set");
        let y_zero_bias_simd = ZeroBiasSimd::from_params(&y_zero_bias);

        let cb_quant_simd = self.cb_quant.as_ref().map(|q| QuantTableSimd::from_values(&q.values));
        let cr_quant_simd = self.cr_quant.as_ref().map(|q| QuantTableSimd::from_values(&q.values));
        let cb_zero_bias_simd = self.cb_zero_bias.as_ref().map(ZeroBiasSimd::from_params);
        let cr_zero_bias_simd = self.cr_zero_bias.as_ref().map(ZeroBiasSimd::from_params);

        // Quantize Y blocks with per-block AQ strengths
        // Uses the SAME quantization function as the full-plane encoder for identical output
        let num_y_blocks = self.y_blocks.len();
        let mut y_quantized = Vec::with_capacity(num_y_blocks);
        for i in 0..num_y_blocks {
            let dct = &self.y_blocks[i];
            // Use per-block AQ strength if available, otherwise fallback to 0.08
            let aq_strength = if i < aq_strengths.len() {
                aq_strengths[i]
            } else {
                0.08 // C++ mean
            };

            // Use SIMD quantization for parity with full-plane encoder
            let quant_coeffs =
                y_quant_simd.quantize_array_with_zero_bias(dct, &y_zero_bias_simd, aq_strength);

            // Convert to zigzag order
            let mut zigzag = [0i16; DCT_BLOCK_SIZE];
            natural_to_zigzag_into(&quant_coeffs, &mut zigzag);

            y_quantized.push(zigzag);
        }

        // Count Y Huffman frequencies after all blocks are quantized
        for zigzag in &y_quantized {
            self.count_block_frequencies(zigzag, true);
        }

        // Quantize Cb/Cr blocks with per-block AQ derived from corresponding Y blocks
        // (same as full-plane encoder for parity)
        let mut cb_quantized = Vec::with_capacity(self.cb_blocks.len());
        let mut cr_quantized = Vec::with_capacity(self.cr_blocks.len());

        if let (Some(cb_qs), Some(cr_qs), Some(cb_zbs), Some(cr_zbs)) =
            (cb_quant_simd, cr_quant_simd, cb_zero_bias_simd, cr_zero_bias_simd)
        {
            // Compute block dimensions for AQ mapping (same as full-plane encoder)
            let y_blocks_h = (self.width + 7) / 8;
            let y_blocks_v = (self.height + 7) / 8;
            let (c_blocks_h, c_blocks_v) = match self.subsampling {
                Subsampling::S420 => ((self.width + 15) / 16, (self.height + 15) / 16),
                Subsampling::S422 => ((self.width + 15) / 16, y_blocks_v),
                Subsampling::S440 => (y_blocks_h, (self.height + 15) / 16),
                Subsampling::S444 => (y_blocks_h, y_blocks_v),
            };

            for i in 0..self.cb_blocks.len() {
                let dct = &self.cb_blocks[i];
                // Map chroma block to corresponding Y block for AQ strength
                // (same formula as quantize_all_blocks_subsampled)
                let bx = i % c_blocks_h;
                let by = i / c_blocks_h;
                let y_bx = (bx * y_blocks_h) / c_blocks_h;
                let y_by = (by * y_blocks_v) / c_blocks_v;
                let y_idx = y_by.min(y_blocks_v - 1) * y_blocks_h + y_bx.min(y_blocks_h - 1);
                let aq_strength = if y_idx < aq_strengths.len() {
                    aq_strengths[y_idx]
                } else {
                    0.08
                };

                let quant_coeffs =
                    cb_qs.quantize_array_with_zero_bias(dct, &cb_zbs, aq_strength);
                let mut zigzag = [0i16; DCT_BLOCK_SIZE];
                natural_to_zigzag_into(&quant_coeffs, &mut zigzag);
                cb_quantized.push(zigzag);
            }

            for i in 0..self.cr_blocks.len() {
                let dct = &self.cr_blocks[i];
                // Map chroma block to corresponding Y block for AQ strength
                let bx = i % c_blocks_h;
                let by = i / c_blocks_h;
                let y_bx = (bx * y_blocks_h) / c_blocks_h;
                let y_by = (by * y_blocks_v) / c_blocks_v;
                let y_idx = y_by.min(y_blocks_v - 1) * y_blocks_h + y_bx.min(y_blocks_h - 1);
                let aq_strength = if y_idx < aq_strengths.len() {
                    aq_strengths[y_idx]
                } else {
                    0.08
                };

                let quant_coeffs =
                    cr_qs.quantize_array_with_zero_bias(dct, &cr_zbs, aq_strength);
                let mut zigzag = [0i16; DCT_BLOCK_SIZE];
                natural_to_zigzag_into(&quant_coeffs, &mut zigzag);
                cr_quantized.push(zigzag);
            }

            // Count chroma Huffman frequencies
            for zigzag in &cb_quantized {
                self.count_block_frequencies(zigzag, false);
            }
            for zigzag in &cr_quantized {
                self.count_block_frequencies(zigzag, false);
            }
        }

        Ok(StripProcessorOutput {
            y_blocks: y_quantized,
            cb_blocks: cb_quantized,
            cr_blocks: cr_quantized,
            aq_strengths,
            dc_luma_freq: self.dc_luma_freq,
            ac_luma_freq: self.ac_luma_freq,
            dc_chroma_freq: self.dc_chroma_freq,
            ac_chroma_freq: self.ac_chroma_freq,
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
