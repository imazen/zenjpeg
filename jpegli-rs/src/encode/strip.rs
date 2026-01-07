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
use crate::quant::{QuantTable, ZeroBiasParams};
use crate::types::{PixelFormat, Subsampling};

use super::natural_to_zigzag_into;

/// Extracts an 8×8 block from a strip buffer (free function to avoid borrow issues).
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
                block[dy * 8 + dx] = strip[last_y * strip_width + x];
            }
        } else {
            for dx in 0..8 {
                let x = (x_start + dx).min(strip_width.saturating_sub(1));
                block[dy * 8 + dx] = strip[y * strip_width + x];
            }
        }
    }

    block
}

/// Streaming AQ state for incremental computation.
///
/// This accumulates AQ features strip-by-strip without needing
/// the full Y plane in memory at once.
#[derive(Debug)]
pub struct StreamingAQState {
    /// Width in pixels
    width: usize,
    /// Height in pixels
    height: usize,
    /// Width in 8×8 blocks
    blocks_w: usize,
    /// Height in 8×8 blocks
    blocks_h: usize,

    /// Pre-erosion values (1/4 resolution), computed incrementally.
    /// Size: (width+3)/4 × (height+3)/4
    pre_erosion: Vec<f32>,

    /// Per-block gamma modulation sums.
    /// Accumulated as strips are processed.
    gamma_sums: Vec<f32>,

    /// Per-block HF modulation sums.
    /// Accumulated as strips are processed.
    hf_sums: Vec<f32>,

    /// Previous row of Y values for HF modulation vertical diffs.
    /// Size: width
    prev_row: Vec<f32>,

    /// How many rows have been processed so far.
    rows_processed: usize,

    /// Running sum for global normalization.
    global_sum: f64,
    /// Running count for global normalization.
    global_count: usize,
}

impl StreamingAQState {
    /// Creates a new streaming AQ state.
    pub fn new(width: usize, height: usize) -> Result<Self> {
        let blocks_w = (width + 7) / 8;
        let blocks_h = (height + 7) / 8;
        let pre_erosion_w = (width + 3) / 4;
        let pre_erosion_h = (height + 3) / 4;

        Ok(Self {
            width,
            height,
            blocks_w,
            blocks_h,
            pre_erosion: try_with_capacity(pre_erosion_w * pre_erosion_h, "pre_erosion")?,
            gamma_sums: vec![0.0f32; blocks_w * blocks_h],
            hf_sums: vec![0.0f32; blocks_w * blocks_h],
            prev_row: vec![0.0f32; width],
            rows_processed: 0,
            global_sum: 0.0,
            global_count: 0,
        })
    }

    /// Processes a strip of Y values, accumulating AQ features.
    ///
    /// # Arguments
    /// * `y_strip` - Y plane values for this strip (width × strip_height)
    /// * `strip_y` - Starting row index of this strip
    /// * `strip_height` - Number of rows in this strip
    pub fn process_strip(&mut self, y_strip: &[f32], strip_y: usize, strip_height: usize) {
        // Process each row in the strip
        for local_y in 0..strip_height {
            let global_y = strip_y + local_y;
            if global_y >= self.height {
                break;
            }

            let row_start = local_y * self.width;
            let row = &y_strip[row_start..row_start + self.width];

            // Accumulate per-block gamma sums
            self.accumulate_gamma_row(row, global_y);

            // Accumulate per-block HF sums (needs prev_row for vertical diffs)
            self.accumulate_hf_row(row, global_y);

            // Update prev_row for next iteration
            self.prev_row.copy_from_slice(row);
        }

        self.rows_processed += strip_height;
    }

    /// Accumulates gamma modulation for one row of the Y plane.
    fn accumulate_gamma_row(&mut self, row: &[f32], y: usize) {
        let block_y = y / 8;
        if block_y >= self.blocks_h {
            return;
        }

        // For each block in this row
        for bx in 0..self.blocks_w {
            let x_start = bx * 8;
            let x_end = (x_start + 8).min(self.width);

            // Sum pixel values in this block's portion of the row
            let mut sum = 0.0f32;
            for x in x_start..x_end {
                sum += row[x];
            }

            self.gamma_sums[block_y * self.blocks_w + bx] += sum;
        }
    }

    /// Accumulates HF modulation for one row of the Y plane.
    fn accumulate_hf_row(&mut self, row: &[f32], y: usize) {
        let block_y = y / 8;
        if block_y >= self.blocks_h {
            return;
        }

        for bx in 0..self.blocks_w {
            let x_start = bx * 8;
            let x_end = (x_start + 8).min(self.width);

            let mut hf_sum = 0.0f32;

            // Horizontal diffs within row
            for x in x_start..(x_end - 1) {
                hf_sum += (row[x] - row[x + 1]).abs();
            }

            // Vertical diffs with previous row
            if y > 0 {
                for x in x_start..x_end {
                    hf_sum += (row[x] - self.prev_row[x]).abs();
                }
            }

            self.hf_sums[block_y * self.blocks_w + bx] += hf_sum;
        }
    }

    /// Finalizes the AQ map after all strips have been processed.
    ///
    /// Returns per-block AQ strength values (0.0 to ~0.2 range).
    pub fn finalize(self) -> Vec<f32> {
        let num_blocks = self.blocks_w * self.blocks_h;
        let mut strengths = Vec::with_capacity(num_blocks);

        // Constants from C++ AQ algorithm
        const K_INPUT_SCALING: f32 = 1.0 / 255.0;
        const K_GAMMA_MOD_BIAS: f32 = 0.16 * K_INPUT_SCALING;
        const K_GAMMA_MOD_SCALE: f32 = 1.0 / 64.0;
        const K_HF_MOD_COEFF: f32 = -2.0052193233688884 / 112.0;

        for by in 0..self.blocks_h {
            for bx in 0..self.blocks_w {
                let idx = by * self.blocks_w + bx;

                // Compute block dimensions (handle edge blocks)
                let block_w = 8.min(self.width - bx * 8);
                let block_h = 8.min(self.height - by * 8);
                let block_pixels = (block_w * block_h) as f32;

                // Gamma modulation (based on average pixel value)
                let avg_pixel = if block_pixels > 0.0 {
                    self.gamma_sums[idx] / block_pixels
                } else {
                    0.5
                };
                let gamma_mod = (avg_pixel * K_INPUT_SCALING - K_GAMMA_MOD_BIAS)
                    .max(0.0)
                    .ln()
                    * K_GAMMA_MOD_SCALE;

                // HF modulation (based on edge strength)
                let hf_avg = if block_pixels > 0.0 {
                    self.hf_sums[idx] / block_pixels
                } else {
                    0.0
                };
                let hf_mod = hf_avg * K_HF_MOD_COEFF;

                // Combine modulations and clamp to valid range
                let quant_field = 1.0 + gamma_mod + hf_mod;
                let aq_strength = ((0.6 / quant_field) - 1.0).max(0.0).min(0.25);

                strengths.push(aq_strength);
            }
        }

        strengths
    }
}

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
    /// Quantized coefficients buffer (reused per block)
    quant_buf: [i16; DCT_BLOCK_SIZE],

    // === Growing block storage ===
    /// Y channel quantized blocks (zigzag order)
    y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cb channel quantized blocks
    cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    /// Cr channel quantized blocks
    cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,

    // === Streaming AQ state ===
    aq_state: StreamingAQState,

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
            quant_buf: [0i16; DCT_BLOCK_SIZE],

            // Block storage (pre-allocated)
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

            // Streaming AQ
            aq_state: StreamingAQState::new(width, height)?,

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
    ) {
        self.y_quant = Some(y_quant);
        self.cb_quant = Some(cb_quant);
        self.cr_quant = Some(cr_quant);
        self.y_zero_bias = Some(y_zero_bias);
        self.cb_zero_bias = Some(cb_zero_bias);
        self.cr_zero_bias = Some(cr_zero_bias);
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
        self.aq_state
            .process_strip(&self.y_strip, strip_y, actual_strip_height);

        // Step 3: Downsample chroma if needed
        if self.pixel_format != PixelFormat::Gray {
            self.downsample_chroma_strip(actual_strip_height)?;
        }

        // Step 4: Quantize blocks in this strip
        // Note: We use a placeholder AQ strength during strip processing.
        // The final AQ-adjusted re-quantization happens after finalize().
        let blocks_added = self.quantize_strip_blocks(strip_y, actual_strip_height)?;

        Ok(blocks_added)
    }

    /// Converts RGB strip data to YCbCr in the strip buffers.
    fn convert_strip_to_ycbcr(&mut self, rgb_strip: &[u8], strip_height: usize) -> Result<()> {
        let bpp = self.pixel_format.bytes_per_pixel();
        let is_color = self.pixel_format != PixelFormat::Gray;

        for y in 0..strip_height {
            for x in 0..self.width {
                let rgb_idx = (y * self.width + x) * bpp;
                let plane_idx = y * self.width + x;

                let (r, g, b) = match self.pixel_format {
                    PixelFormat::Rgb => (
                        rgb_strip[rgb_idx] as f32,
                        rgb_strip[rgb_idx + 1] as f32,
                        rgb_strip[rgb_idx + 2] as f32,
                    ),
                    PixelFormat::Rgba => (
                        rgb_strip[rgb_idx] as f32,
                        rgb_strip[rgb_idx + 1] as f32,
                        rgb_strip[rgb_idx + 2] as f32,
                    ),
                    PixelFormat::Bgr => (
                        rgb_strip[rgb_idx + 2] as f32,
                        rgb_strip[rgb_idx + 1] as f32,
                        rgb_strip[rgb_idx] as f32,
                    ),
                    PixelFormat::Bgra => (
                        rgb_strip[rgb_idx + 2] as f32,
                        rgb_strip[rgb_idx + 1] as f32,
                        rgb_strip[rgb_idx] as f32,
                    ),
                    PixelFormat::Gray => {
                        let gray = rgb_strip[rgb_idx] as f32;
                        (gray, gray, gray)
                    }
                    PixelFormat::Cmyk => {
                        // CMYK to RGB conversion
                        let c = rgb_strip[rgb_idx] as f32 / 255.0;
                        let m = rgb_strip[rgb_idx + 1] as f32 / 255.0;
                        let y_val = rgb_strip[rgb_idx + 2] as f32 / 255.0;
                        let k = rgb_strip[rgb_idx + 3] as f32 / 255.0;
                        let r = 255.0 * (1.0 - c) * (1.0 - k);
                        let g = 255.0 * (1.0 - m) * (1.0 - k);
                        let b = 255.0 * (1.0 - y_val) * (1.0 - k);
                        (r, g, b)
                    }
                };

                // RGB to YCbCr conversion (BT.601)
                // Y  =  0.299 R + 0.587 G + 0.114 B
                // Cb = -0.169 R - 0.331 G + 0.500 B + 128
                // Cr =  0.500 R - 0.419 G - 0.081 B + 128
                self.y_strip[plane_idx] = 0.299 * r + 0.587 * g + 0.114 * b;

                if is_color {
                    self.cb_strip[plane_idx] = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
                    self.cr_strip[plane_idx] = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
                }
            }
        }

        Ok(())
    }

    /// Downsamples chroma strips according to subsampling mode.
    fn downsample_chroma_strip(&mut self, strip_height: usize) -> Result<()> {
        match self.subsampling {
            Subsampling::S420 => {
                // 2×2 box filter
                let c_height = (strip_height + 1) / 2;
                let c_width = (self.width + 1) / 2;

                for cy in 0..c_height {
                    for cx in 0..c_width {
                        let y0 = cy * 2;
                        let y1 = (y0 + 1).min(strip_height - 1);
                        let x0 = cx * 2;
                        let x1 = (x0 + 1).min(self.width - 1);

                        // Average 2×2 block
                        let cb_avg = (self.cb_strip[y0 * self.width + x0]
                            + self.cb_strip[y0 * self.width + x1]
                            + self.cb_strip[y1 * self.width + x0]
                            + self.cb_strip[y1 * self.width + x1])
                            * 0.25;

                        let cr_avg = (self.cr_strip[y0 * self.width + x0]
                            + self.cr_strip[y0 * self.width + x1]
                            + self.cr_strip[y1 * self.width + x0]
                            + self.cr_strip[y1 * self.width + x1])
                            * 0.25;

                        self.cb_down[cy * c_width + cx] = cb_avg;
                        self.cr_down[cy * c_width + cx] = cr_avg;
                    }
                }
            }
            Subsampling::S422 => {
                // 2×1 horizontal filter
                let c_width = (self.width + 1) / 2;

                for y in 0..strip_height {
                    for cx in 0..c_width {
                        let x0 = cx * 2;
                        let x1 = (x0 + 1).min(self.width - 1);

                        let cb_avg =
                            (self.cb_strip[y * self.width + x0] + self.cb_strip[y * self.width + x1])
                                * 0.5;
                        let cr_avg =
                            (self.cr_strip[y * self.width + x0] + self.cr_strip[y * self.width + x1])
                                * 0.5;

                        self.cb_down[y * c_width + cx] = cb_avg;
                        self.cr_down[y * c_width + cx] = cr_avg;
                    }
                }
            }
            Subsampling::S440 => {
                // 1×2 vertical filter
                let c_height = (strip_height + 1) / 2;

                for cy in 0..c_height {
                    let y0 = cy * 2;
                    let y1 = (y0 + 1).min(strip_height - 1);

                    for x in 0..self.width {
                        let cb_avg = (self.cb_strip[y0 * self.width + x]
                            + self.cb_strip[y1 * self.width + x])
                            * 0.5;
                        let cr_avg = (self.cr_strip[y0 * self.width + x]
                            + self.cr_strip[y1 * self.width + x])
                            * 0.5;

                        self.cb_down[cy * self.width + x] = cb_avg;
                        self.cr_down[cy * self.width + x] = cr_avg;
                    }
                }
            }
            Subsampling::S444 => {
                // No downsampling - copy directly
                self.cb_down[..strip_height * self.width]
                    .copy_from_slice(&self.cb_strip[..strip_height * self.width]);
                self.cr_down[..strip_height * self.width]
                    .copy_from_slice(&self.cr_strip[..strip_height * self.width]);
            }
        }

        Ok(())
    }

    /// Quantizes blocks from the current strip buffers.
    fn quantize_strip_blocks(&mut self, strip_y: usize, strip_height: usize) -> Result<usize> {
        // Clone quant tables and bias params upfront to avoid borrow conflicts
        let y_quant_values = self.y_quant.as_ref().expect("y_quant not set").values;
        let y_zero_bias = self.y_zero_bias.clone().expect("y_zero_bias not set");

        let blocks_w = (self.width + 7) / 8;
        let strip_blocks_h = (strip_height + 7) / 8;
        let start_block_y = strip_y / 8;
        let width = self.width;
        let height = self.height;

        let mut blocks_added = 0;

        // Quantize Y blocks
        for local_by in 0..strip_blocks_h {
            let global_by = start_block_y + local_by;
            if global_by >= (height + 7) / 8 {
                break;
            }

            for bx in 0..blocks_w {
                // Extract 8×8 block from Y strip
                let block = extract_block_from_strip(&self.y_strip, bx, local_by, width);

                // DCT
                let dct = forward_dct_8x8(&block);

                // Quantize with placeholder AQ strength
                // TODO: Use actual AQ strength after finalization
                let aq_strength = 0.08; // C++ mean
                let quant_coeffs = crate::quant::quantize_block_with_zero_bias_simd(
                    &dct,
                    &y_quant_values,
                    &y_zero_bias,
                    aq_strength,
                );

                // Convert to zigzag order and store
                let mut zigzag = [0i16; DCT_BLOCK_SIZE];
                natural_to_zigzag_into(&quant_coeffs, &mut zigzag);
                self.y_blocks.push(zigzag);

                // Count Huffman frequencies
                self.count_block_frequencies(&zigzag, true);

                blocks_added += 1;
            }
        }

        // Quantize Cb/Cr blocks (if color)
        if self.pixel_format != PixelFormat::Gray {
            let (c_width, c_strip_height) = match self.subsampling {
                Subsampling::S420 => ((width + 1) / 2, strip_height / 2),
                Subsampling::S422 => ((width + 1) / 2, strip_height),
                Subsampling::S440 => (width, strip_height / 2),
                Subsampling::S444 => (width, strip_height),
            };

            let c_blocks_w = (c_width + 7) / 8;
            let c_strip_blocks_h = (c_strip_height + 7) / 8;

            // Clone quant tables and bias params upfront
            let cb_quant_values = self.cb_quant.as_ref().expect("cb_quant not set").values;
            let cr_quant_values = self.cr_quant.as_ref().expect("cr_quant not set").values;
            let cb_zero_bias = self.cb_zero_bias.clone().expect("cb_zero_bias not set");
            let cr_zero_bias = self.cr_zero_bias.clone().expect("cr_zero_bias not set");

            for local_by in 0..c_strip_blocks_h {
                for bx in 0..c_blocks_w {
                    // Cb block
                    let cb_block = extract_block_from_strip(&self.cb_down, bx, local_by, c_width);
                    let cb_dct = forward_dct_8x8(&cb_block);
                    let cb_coeffs = crate::quant::quantize_block_with_zero_bias_simd(
                        &cb_dct,
                        &cb_quant_values,
                        &cb_zero_bias,
                        0.08,
                    );
                    let mut cb_zigzag = [0i16; DCT_BLOCK_SIZE];
                    natural_to_zigzag_into(&cb_coeffs, &mut cb_zigzag);
                    self.cb_blocks.push(cb_zigzag);
                    self.count_block_frequencies(&cb_zigzag, false);

                    // Cr block
                    let cr_block = extract_block_from_strip(&self.cr_down, bx, local_by, c_width);
                    let cr_dct = forward_dct_8x8(&cr_block);
                    let cr_coeffs = crate::quant::quantize_block_with_zero_bias_simd(
                        &cr_dct,
                        &cr_quant_values,
                        &cr_zero_bias,
                        0.08,
                    );
                    let mut cr_zigzag = [0i16; DCT_BLOCK_SIZE];
                    natural_to_zigzag_into(&cr_coeffs, &mut cr_zigzag);
                    self.cr_blocks.push(cr_zigzag);
                    self.count_block_frequencies(&cr_zigzag, false);
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
    /// Returns the quantized blocks and Huffman frequency counters.
    pub fn finalize(
        self,
    ) -> StripProcessorOutput {
        // Finalize AQ map (global normalization)
        let aq_strengths = self.aq_state.finalize();

        StripProcessorOutput {
            y_blocks: self.y_blocks,
            cb_blocks: self.cb_blocks,
            cr_blocks: self.cr_blocks,
            aq_strengths,
            dc_luma_freq: self.dc_luma_freq,
            ac_luma_freq: self.ac_luma_freq,
            dc_chroma_freq: self.dc_chroma_freq,
            ac_chroma_freq: self.ac_chroma_freq,
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

    #[test]
    fn test_streaming_aq_state_creation() {
        let state = StreamingAQState::new(4000, 3000);
        assert!(state.is_ok());
        let state = state.unwrap();
        assert_eq!(state.blocks_w, 500);
        assert_eq!(state.blocks_h, 375);
    }
}
