//! Streaming Adaptive Quantization - Low memory, per-iMCU processing.
//!
//! This implementation matches the C++ jpegli memory model by:
//! 1. Using rolling buffers instead of storing the full Y plane
//! 2. Computing AQ values per iMCU row (not all at once at the end)
//! 3. Allowing immediate quantization during strip processing
//!
//! ## Memory Model
//!
//! For a 4K image (3840x2160):
//! - Pre-erosion buffer (rolling): 960 * 12 rows = 45 KB
//! - Y iMCU buffers (2x): 3840 * 16 * 2 = 490 KB
//! - Row buffers: ~60 KB
//! - Total: ~2.5 MB (vs 33 MB for full Y plane)
//!
//! ## Lookahead for Fuzzy Erosion
//!
//! Fuzzy erosion needs a 3x3 neighborhood on the pre_erosion buffer. For the last block row
//! of an iMCU, this requires pre_erosion rows from the next iMCU. To handle this:
//!
//! 1. When an iMCU completes, we DON'T finalize its AQ immediately
//! 2. We wait until the next iMCU has processed enough rows to provide lookahead
//! 3. Double-buffering the Y data allows us to finalize the previous iMCU
//!
//! This adds ~4 rows of latency but produces results matching the full-plane algorithm.

use crate::aligned_alloc::{try_alloc_zeroed, AlignedVec};
use crate::error::Result;

use super::quant_field_to_aq_strength;
use super::simd::{per_block_modulations_row, pre_erosion_row};

/// Streaming AQ with rolling buffers - low memory, high performance.
///
/// Supports two usage modes:
///
/// ## Batch Mode (compatible with existing code)
/// ```ignore
/// let mut aq = StreamingAQ::new(width, height, y_quant_01, v_samp)?;
/// for strip in strips {
///     aq.process_y_strip(&strip, strip_y, strip_height);
/// }
/// let all_strengths = aq.finalize()?; // Get all at once
/// ```
///
/// ## Incremental Mode (lowest memory)
/// ```ignore
/// let mut aq = StreamingAQ::new(width, height, y_quant_01, v_samp)?;
/// for strip in strips {
///     if let Some(strengths) = aq.process_y_strip(&strip, strip_y, strip_height) {
///         // Quantize this iMCU's blocks immediately
///     }
/// }
/// if let Some(strengths) = aq.flush() {
///     // Handle last iMCU
/// }
/// ```
#[derive(Debug)]
pub struct StreamingAQ {
    // Image dimensions
    width: usize,
    height: usize,

    // Block dimensions
    blocks_w: usize,
    blocks_h: usize,

    // Pre-erosion dimensions (4x downsampled)
    pre_erosion_w: usize,
    pre_erosion_h: usize,

    // === Rolling buffers ===
    // Pre-erosion buffer: stores enough rows for fuzzy erosion context
    pre_erosion_buffer: AlignedVec<f32>,
    pre_erosion_buffer_rows: usize,

    // Row buffers for pre_erosion_row with 1-row lookahead
    row_prev_prev: AlignedVec<f32>,
    row_prev: AlignedVec<f32>,
    row_curr: AlignedVec<f32>,
    pending_pre_erosion_row: Option<usize>,

    // Accumulator for 4-row vertical downsampling
    pre_erosion_accum: AlignedVec<f32>,

    // Reusable scratch buffers (avoid per-row allocations)
    pre_erosion_temp: AlignedVec<f32>,

    // Double-buffered Y iMCU data (for lookahead support)
    y_imcu_buffers: [AlignedVec<f32>; 2],
    y_imcu_current: usize,
    y_imcu_height: usize,

    // Intermediate: fuzzy erosion output for current iMCU block rows
    fuzzy_erosion_out: AlignedVec<f32>,

    // Output: per-block AQ strengths for current iMCU (reused each iMCU)
    imcu_aq_strengths: Vec<f32>,

    // Accumulator for batch mode: collects all AQ values
    all_aq_strengths: Vec<f32>,

    // Y quantization table value at position 1
    y_quant_01: f32,

    // Progress tracking
    rows_received: usize,
    current_imcu_row: usize,
    /// Total number of iMCU rows (reserved for progress tracking)
    #[allow(dead_code)]
    total_imcu_rows: usize,

    // Pre-erosion tracking
    pre_erosion_rows_flushed: usize,

    // Pending AQ: iMCU row waiting for lookahead before finalization
    pending_imcu_row: Option<usize>,
}

impl StreamingAQ {
    /// Creates a new streaming AQ state.
    ///
    /// # Arguments
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `y_quant_01` - Y quant table value at position [0,1] (first AC coefficient)
    /// * `v_samp_factor` - Vertical sampling factor (1 for 4:4:4/4:2:2, 2 for 4:2:0/4:4:0)
    ///
    /// # Errors
    /// Returns `AllocError` if buffer allocation fails.
    pub fn new(width: usize, height: usize, y_quant_01: u16, v_samp_factor: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Ok(Self::empty(y_quant_01 as f32));
        }

        let blocks_w = (width + 7) / 8;
        let blocks_h = (height + 7) / 8;
        let pre_erosion_w = (width + 3) / 4;
        let pre_erosion_h = (height + 3) / 4;

        // iMCU height in pixels
        let imcu_height = 8 * v_samp_factor;
        let total_imcu_rows = (height + imcu_height - 1) / imcu_height;

        // Pre-erosion buffer: 12 rows for lookahead
        let pre_erosion_buffer_rows = 12;

        // Blocks per iMCU row
        let blocks_per_imcu = blocks_w * v_samp_factor;

        // Total blocks for accumulator
        let total_blocks = blocks_w * blocks_h;

        Ok(Self {
            width,
            height,
            blocks_w,
            blocks_h,
            pre_erosion_w,
            pre_erosion_h,
            pre_erosion_buffer: try_alloc_zeroed(pre_erosion_w * pre_erosion_buffer_rows)?,
            pre_erosion_buffer_rows,
            row_prev_prev: try_alloc_zeroed(width)?,
            row_prev: try_alloc_zeroed(width)?,
            row_curr: try_alloc_zeroed(width)?,
            pending_pre_erosion_row: None,
            pre_erosion_accum: try_alloc_zeroed(width)?,
            pre_erosion_temp: try_alloc_zeroed(width)?,
            y_imcu_buffers: [
                try_alloc_zeroed(width * imcu_height)?,
                try_alloc_zeroed(width * imcu_height)?,
            ],
            y_imcu_current: 0,
            y_imcu_height: imcu_height,
            fuzzy_erosion_out: try_alloc_zeroed(blocks_per_imcu)?,
            imcu_aq_strengths: vec![0.0f32; blocks_per_imcu],
            all_aq_strengths: Vec::with_capacity(total_blocks),
            y_quant_01: y_quant_01 as f32,
            rows_received: 0,
            current_imcu_row: 0,
            total_imcu_rows,
            pre_erosion_rows_flushed: 0,
            pending_imcu_row: None,
        })
    }

    fn empty(y_quant_01: f32) -> Self {
        Self {
            width: 0,
            height: 0,
            blocks_w: 0,
            blocks_h: 0,
            pre_erosion_w: 0,
            pre_erosion_h: 0,
            pre_erosion_buffer: AlignedVec::new(0),
            pre_erosion_buffer_rows: 0,
            row_prev_prev: AlignedVec::new(0),
            row_prev: AlignedVec::new(0),
            row_curr: AlignedVec::new(0),
            pending_pre_erosion_row: None,
            pre_erosion_accum: AlignedVec::new(0),
            pre_erosion_temp: AlignedVec::new(0),
            y_imcu_buffers: [AlignedVec::new(0), AlignedVec::new(0)],
            y_imcu_current: 0,
            y_imcu_height: 0,
            fuzzy_erosion_out: AlignedVec::new(0),
            imcu_aq_strengths: Vec::new(),
            all_aq_strengths: Vec::new(),
            y_quant_01,
            rows_received: 0,
            current_imcu_row: 0,
            total_imcu_rows: 0,
            pre_erosion_rows_flushed: 0,
            pending_imcu_row: None,
        }
    }

    /// Process Y strip data and compute AQ for completed iMCU rows.
    ///
    /// # Arguments
    /// * `y_strip` - Y plane values for this strip (width × strip_height), 0-255 range
    /// * `strip_y` - Starting row index of this strip
    /// * `strip_height` - Number of rows in this strip
    ///
    /// # Returns
    /// The AQ strengths for a completed iMCU row, or None if no iMCU is ready.
    /// Due to lookahead requirements, AQ output is delayed by ~1 iMCU.
    pub fn process_y_strip(
        &mut self,
        y_strip: &[f32],
        strip_y: usize,
        strip_height: usize,
    ) -> Option<&[f32]> {
        if self.width == 0 || self.height == 0 {
            return None;
        }

        // Process each row in the strip
        for local_y in 0..strip_height {
            let global_y = strip_y + local_y;
            if global_y >= self.height {
                break;
            }

            let row_start = local_y * self.width;
            let row_end = row_start + self.width;
            let row = &y_strip[row_start..row_end];

            // Store in current Y iMCU buffer
            let imcu_local_y = global_y % self.y_imcu_height;
            let dest_start = imcu_local_y * self.width;
            self.y_imcu_buffers[self.y_imcu_current][dest_start..dest_start + self.width]
                .copy_from_slice(row);

            // Process pre-erosion for this row
            self.process_pre_erosion_row(row, global_y);

            self.rows_received = global_y + 1;
        }

        // Check if we completed an iMCU row
        let imcu_height = self.y_imcu_height;
        let next_imcu_boundary = (self.current_imcu_row + 1) * imcu_height;

        if self.rows_received >= next_imcu_boundary.min(self.height) {
            // Edge clamp: fill remaining rows of the iMCU buffer with copies of the last valid row
            // This is needed for partial iMCU rows at the bottom of the image
            let valid_rows_in_imcu = self.rows_received - self.current_imcu_row * imcu_height;
            if valid_rows_in_imcu < imcu_height && valid_rows_in_imcu > 0 {
                let last_valid_row = valid_rows_in_imcu - 1;
                let src_start = last_valid_row * self.width;
                let src_end = src_start + self.width;
                for fill_row in valid_rows_in_imcu..imcu_height {
                    let dest_start = fill_row * self.width;
                    self.y_imcu_buffers[self.y_imcu_current]
                        .copy_within(src_start..src_end, dest_start);
                }
            }
            // Finalize previously pending iMCU (the one waiting for lookahead)
            let valid_count = if let Some(pending) = self.pending_imcu_row.take() {
                let prev_buffer = 1 - self.y_imcu_current;
                let count = self.finalize_imcu_aq_with_buffer(pending, prev_buffer);
                // Accumulate for batch mode - only the valid portion
                self.all_aq_strengths
                    .extend_from_slice(&self.imcu_aq_strengths[..count]);
                Some(count)
            } else {
                None
            };

            // Mark just-completed iMCU as pending
            self.pending_imcu_row = Some(self.current_imcu_row);
            self.current_imcu_row += 1;
            self.y_imcu_current = 1 - self.y_imcu_current;

            if let Some(count) = valid_count {
                return Some(&self.imcu_aq_strengths[..count]);
            }
        }

        None
    }

    /// Flush any pending iMCU AQ at end of image.
    ///
    /// Call after all strips have been processed to get the last iMCU's AQ.
    pub fn flush(&mut self) -> Option<&[f32]> {
        if let Some(pending) = self.pending_imcu_row.take() {
            let prev_buffer = 1 - self.y_imcu_current;
            let count = self.finalize_imcu_aq_with_buffer(pending, prev_buffer);
            // Only append the valid portion for partial iMCU rows
            self.all_aq_strengths
                .extend_from_slice(&self.imcu_aq_strengths[..count]);
            return Some(&self.imcu_aq_strengths[..count]);
        }
        None
    }

    /// Finalize and return all AQ strengths (batch mode).
    ///
    /// This is a drop-in replacement for the old `StreamingAQParity::finalize()`.
    /// Flushes any remaining iMCU and returns all accumulated AQ values.
    pub fn finalize(mut self) -> Result<Vec<f32>> {
        if self.width == 0 || self.height == 0 {
            return Ok(Vec::new());
        }

        // Flush any remaining pending iMCU
        self.flush();

        Ok(self.all_aq_strengths)
    }

    /// Process a single row for pre-erosion computation.
    fn process_pre_erosion_row(&mut self, row: &[f32], global_y: usize) {
        // Shift row buffers
        std::mem::swap(&mut self.row_prev_prev, &mut self.row_prev);
        std::mem::swap(&mut self.row_prev, &mut self.row_curr);
        self.row_curr.copy_from_slice(row);

        // Initialize for first rows (boundary clamping)
        if global_y == 0 {
            self.row_prev.copy_from_slice(row);
            self.row_prev_prev.copy_from_slice(row);
        } else if global_y == 1 {
            self.row_prev_prev.copy_from_slice(&self.row_prev);
        }

        // Process pending row now that we have lookahead
        if let Some(pending_y) = self.pending_pre_erosion_row {
            self.compute_and_accumulate_pre_erosion(pending_y);
        }

        self.pending_pre_erosion_row = Some(global_y);

        // For last row, flush immediately with boundary clamping
        if global_y + 1 == self.height {
            self.compute_last_row_pre_erosion();
            self.pending_pre_erosion_row = None;
            let last_block_y = global_y / 4;
            self.flush_pre_erosion_block(last_block_y);
        }
    }

    fn compute_last_row_pre_erosion(&mut self) {
        let row_above = &self.row_prev;
        let row_curr = &self.row_curr;
        let row_below = &self.row_curr; // Boundary clamping

        self.pre_erosion_temp.fill(0.0);
        pre_erosion_row(row_curr, row_above, row_below, &mut self.pre_erosion_temp);

        for x in 0..self.width {
            self.pre_erosion_accum[x] += self.pre_erosion_temp[x];
        }
    }

    fn compute_and_accumulate_pre_erosion(&mut self, row_y: usize) {
        let row_above = &self.row_prev_prev;
        let row_curr = &self.row_prev;
        let row_below = &self.row_curr;

        self.pre_erosion_temp.fill(0.0);
        pre_erosion_row(row_curr, row_above, row_below, &mut self.pre_erosion_temp);

        for x in 0..self.width {
            self.pre_erosion_accum[x] += self.pre_erosion_temp[x];
        }

        if (row_y + 1) % 4 == 0 && row_y + 1 < self.height {
            self.flush_pre_erosion_block(row_y / 4);
        }
    }

    fn flush_pre_erosion_block(&mut self, block_y: usize) {
        if block_y >= self.pre_erosion_h {
            return;
        }

        let buffer_row = block_y % self.pre_erosion_buffer_rows;
        let out_start = buffer_row * self.pre_erosion_w;

        for x_block in 0..self.pre_erosion_w {
            let in_x = x_block * 4;
            let mut sum = 0.0f32;
            for i in 0..4 {
                if in_x + i < self.width {
                    sum += self.pre_erosion_accum[in_x + i];
                }
            }
            self.pre_erosion_buffer[out_start + x_block] = sum * 0.25;
        }

        self.pre_erosion_accum.fill(0.0);
        self.pre_erosion_rows_flushed = block_y + 1;
    }

    /// Compute AQ strengths for an iMCU row.
    /// Returns the number of valid AQ values computed (may be less than
    /// blocks_per_imcu for partial iMCU rows at the bottom of the image).
    fn finalize_imcu_aq_with_buffer(&mut self, imcu_row: usize, y_buffer_idx: usize) -> usize {
        let v_samp = self.y_imcu_height / 8;
        let blocks_w = self.blocks_w;

        // Compute actual valid pixel height in this iMCU buffer
        // For partial iMCU rows at the bottom, this may be less than y_imcu_height
        let imcu_start_row = imcu_row * self.y_imcu_height;
        let valid_pixel_height = if imcu_start_row + self.y_imcu_height <= self.height {
            self.y_imcu_height
        } else {
            self.height.saturating_sub(imcu_start_row)
        };

        // Damping calculation (from per_block_modulations_simd)
        const K_AC_QUANT: f32 = 0.841;
        const K_DAMPEN_RAMP_START: f32 = 9.0;
        const K_DAMPEN_RAMP_END: f32 = 65.0;
        let base_level = 0.48 * K_AC_QUANT;
        let dampen = if self.y_quant_01 >= K_DAMPEN_RAMP_START {
            let d = 1.0
                - (self.y_quant_01 - K_DAMPEN_RAMP_START)
                    / (K_DAMPEN_RAMP_END - K_DAMPEN_RAMP_START);
            d.max(0.0)
        } else {
            1.0
        };
        let mul = K_AC_QUANT * dampen;
        let add = (1.0 - dampen) * base_level;

        let mut valid_rows = 0;
        for by_offset in 0..v_samp {
            let global_by = imcu_row * v_samp + by_offset;
            if global_by >= self.blocks_h {
                break;
            }
            valid_rows += 1;

            let row_start = by_offset * blocks_w;
            let row_end = row_start + blocks_w;

            // Fuzzy erosion
            let pe_y = global_by * 2;
            for i in row_start..row_end {
                self.fuzzy_erosion_out[i] = 0.0;
            }
            self.compute_fuzzy_erosion_row_into(pe_y, row_start, row_end);

            // Per-block modulations
            // Use valid_pixel_height instead of y_imcu_height for proper boundary handling
            per_block_modulations_row(
                &self.y_imcu_buffers[y_buffer_idx],
                self.width,
                valid_pixel_height,
                by_offset,
                blocks_w,
                &mut self.fuzzy_erosion_out[row_start..row_end],
                mul,
                add,
            );

            // Convert to AQ strength
            for bx in 0..blocks_w {
                let block_idx = row_start + bx;
                let qf = self.fuzzy_erosion_out[block_idx];
                self.imcu_aq_strengths[block_idx] = quant_field_to_aq_strength(qf);
            }
        }

        valid_rows * blocks_w
    }

    fn compute_fuzzy_erosion_row_into(&mut self, pe_y_base: usize, start: usize, end: usize) {
        let pe_w = self.pre_erosion_w;
        let buffer_rows = self.pre_erosion_buffer_rows;
        let max_filled_row = self.pre_erosion_rows_flushed.saturating_sub(1) as isize;

        const MUL0: f32 = 0.125;
        const MUL1: f32 = 0.075;
        const MUL2: f32 = 0.06;
        const MUL3: f32 = 0.05;

        for bx in start..end {
            let pe_x_base = (bx - start) * 2;
            let pe_y = pe_y_base as isize;

            let mut sum = 0.0f32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let cx = (pe_x_base + dx) as isize;
                    let cy = pe_y + dy as isize;

                    let mut vals = [0.0f32; 9];
                    for (i, (ny, nx)) in [
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 0),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1),
                    ]
                    .iter()
                    .enumerate()
                    {
                        let px = (cx + nx).clamp(0, pe_w as isize - 1) as usize;
                        let py = (cy + ny).clamp(0, max_filled_row.max(0)) as usize;
                        let buffer_row = py % buffer_rows;
                        let buf_idx = buffer_row * pe_w + px;
                        vals[i] = if buf_idx < self.pre_erosion_buffer.len() {
                            self.pre_erosion_buffer[buf_idx]
                        } else {
                            0.0
                        };
                    }

                    // Partial sort to get 4 smallest
                    for i in 0..4 {
                        for j in (i + 1)..9 {
                            if vals[j] < vals[i] {
                                vals.swap(i, j);
                            }
                        }
                    }

                    sum += MUL0 * vals[0] + MUL1 * vals[1] + MUL2 * vals[2] + MUL3 * vals[3];
                }
            }

            self.fuzzy_erosion_out[bx] = sum;
        }
    }

    /// Check if all strips have been processed.
    pub fn is_complete(&self) -> bool {
        self.rows_received >= self.height
    }

    /// Returns the number of rows received so far.
    pub fn rows_received(&self) -> usize {
        self.rows_received
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::aq::compute_aq_strength_map;

    #[test]
    fn test_streaming_aq_creation() {
        let aq = StreamingAQ::new(256, 256, 3, 2).unwrap();
        assert_eq!(aq.blocks_w, 32);
        assert_eq!(aq.blocks_h, 32);
        assert_eq!(aq.y_imcu_height, 16);
    }

    #[test]
    fn test_streaming_matches_full_plane_uniform() {
        let width = 64;
        let height = 64;
        let y_quant_01 = 2u16;

        let y_plane = vec![128.0f32; width * height];

        // Full-plane computation
        let full_result = compute_aq_strength_map(&y_plane, width, height, y_quant_01).unwrap();

        // Streaming computation
        let mut streaming = StreamingAQ::new(width, height, y_quant_01, 2).unwrap();
        let strip_height = 16;
        for strip_y in (0..height).step_by(strip_height) {
            let actual_height = strip_height.min(height - strip_y);
            let strip_start = strip_y * width;
            let strip_end = strip_start + actual_height * width;
            streaming.process_y_strip(&y_plane[strip_start..strip_end], strip_y, actual_height);
        }
        let streaming_result = streaming.finalize().unwrap();

        assert_eq!(full_result.strengths.len(), streaming_result.len());

        let max_diff: f32 = full_result
            .strengths
            .iter()
            .zip(streaming_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(max_diff < 0.01, "Max diff {} exceeds threshold", max_diff);
    }

    #[test]
    fn test_streaming_matches_full_plane_gradient() {
        let width = 128;
        let height = 128;
        let y_quant_01 = 3u16;

        let y_plane: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                ((x + y) as f32 / 2.0).min(255.0)
            })
            .collect();

        let full_result = compute_aq_strength_map(&y_plane, width, height, y_quant_01).unwrap();

        let mut streaming = StreamingAQ::new(width, height, y_quant_01, 2).unwrap();
        let strip_height = 16;
        for strip_y in (0..height).step_by(strip_height) {
            let actual_height = strip_height.min(height - strip_y);
            let strip_start = strip_y * width;
            let strip_end = strip_start + actual_height * width;
            streaming.process_y_strip(&y_plane[strip_start..strip_end], strip_y, actual_height);
        }
        let streaming_result = streaming.finalize().unwrap();

        let max_diff: f32 = full_result
            .strengths
            .iter()
            .zip(streaming_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(max_diff < 0.1, "Max diff {} exceeds threshold", max_diff);
    }

    #[test]
    fn test_streaming_incremental_api() {
        let width = 64;
        let height = 64;
        let y_quant_01 = 3u16;

        let y_plane: Vec<f32> = (0..width * height)
            .map(|i| ((i % width + i / width) as f32 / 2.0).min(255.0))
            .collect();

        let mut streaming = StreamingAQ::new(width, height, y_quant_01, 2).unwrap();
        let mut collected = Vec::new();

        let strip_height = 16;
        for strip_y in (0..height).step_by(strip_height) {
            let actual_height = strip_height.min(height - strip_y);
            let strip_start = strip_y * width;
            let strip_end = strip_start + actual_height * width;

            if let Some(aq) =
                streaming.process_y_strip(&y_plane[strip_start..strip_end], strip_y, actual_height)
            {
                collected.extend_from_slice(aq);
            }
        }
        if let Some(aq) = streaming.flush() {
            collected.extend_from_slice(aq);
        }

        // Should have 64 blocks (8x8 grid)
        assert_eq!(collected.len(), 64);

        // All values should be in valid range
        for &v in &collected {
            assert!(v >= 0.0 && v < 1.0, "Invalid AQ value: {}", v);
        }
    }
}
