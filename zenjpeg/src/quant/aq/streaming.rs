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

use crate::encode::layout::LayoutParams;
use crate::error::Result;
use crate::foundation::aligned_alloc::{AlignedVec, try_alloc_zeroed};

use super::autovec::compute_fuzzy_erosion_blocks_autovec;
use super::quant_field_to_aq_strength;
use super::simd::per_block_modulations_row;

// Use autovec pre_erosion as fallback (non-x86_64 targets)
#[cfg(not(target_arch = "x86_64"))]
use super::autovec::pre_erosion_row_autovec_iter as pre_erosion_row_padded;

// AVX-512/AVX2 version - handles its own dispatch (tries V4 first, falls back to V3)
#[cfg(target_arch = "x86_64")]
use super::simd::mage_pre_erosion_row_padded_v4;

#[cfg(target_arch = "x86_64")]
use super::simd::mage_per_block_modulations_row;

// Note: mage_compute_fuzzy_erosion_row exists but is not used - it's 3x slower
// than scalar due to function call overhead. Would need true SIMD partial sort.

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, X64V3Token};

/// Streaming AQ with rolling buffers - low memory, high performance.
///
/// Supports two usage modes:
///
/// ## Batch Mode (compatible with existing code)
/// ```ignore
/// let layout = LayoutParams::new(width, height, subsampling, use_xyb);
/// let mut aq = StreamingAQ::new(&layout, y_quant_01, true)?;
/// for strip in strips {
///     aq.process_y_strip(&strip, strip_y, strip_height);
/// }
/// let all_strengths = aq.finalize()?; // Get all at once
/// ```
///
/// ## Incremental Mode (lowest memory)
/// ```ignore
/// let layout = LayoutParams::new(width, height, subsampling, use_xyb);
/// let mut aq = StreamingAQ::new(&layout, y_quant_01, true)?;
/// for strip in strips {
///     if let Some(strengths) = aq.process_y_strip(&strip, strip_y, strip_height) {
///         // Quantize this iMCU's blocks immediately
///     }
/// }
/// if let Some(strengths) = aq.flush() {
///     // Handle last iMCU
/// }
/// ```
///
/// ## Bypass Mode (AQ disabled)
///
/// When `aq_enabled=false`, the expensive AQ computation is skipped entirely.
/// The struct still tracks iMCU boundaries and returns the correct number of
/// zero-valued AQ strengths at each boundary, preserving pipeline timing.
/// This saves ~600KB-2.5MB of buffer allocations depending on image size.
#[derive(Debug)]
pub struct StreamingAQ {
    /// When false, skips all AQ computation and returns 0.0 for all blocks.
    aq_enabled: bool,
    // Image dimensions
    width: usize,
    height: usize,
    /// MCU-aligned width (blocks_w * 8) - used for internal buffers to avoid boundary checks
    padded_width: usize,
    /// Row stride in Y iMCU buffer (padded_width + 1 for edge replication during HF modulation)
    /// The extra pixel allows `hf_modulation_sum_8x8` to read position 8 of the rightmost block
    /// without wrapping to the next row.
    y_buffer_stride: usize,
    /// Row stride in input data (may be larger than width for padded strips)
    strip_stride: usize,

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

    // Archmage token for optimized SIMD (when feature enabled)
    #[cfg(target_arch = "x86_64")]
    archmage_token: Option<X64V3Token>,
}

impl StreamingAQ {
    /// Creates a new streaming AQ state from layout parameters.
    ///
    /// All geometry (dimensions, block counts, strides) comes from the shared
    /// `LayoutParams` — no independent recomputation.
    ///
    /// # Arguments
    /// * `layout` - Immutable image layout (source of truth for all geometry)
    /// * `y_quant_01` - Y quant table value at position [0,1] (first AC coefficient)
    /// * `aq_enabled` - When false, skips AQ computation and returns 0.0 for all blocks
    ///
    /// # Errors
    /// Returns `AllocError` if buffer allocation fails.
    pub(crate) fn new(layout: &LayoutParams, y_quant_01: u16, aq_enabled: bool) -> Result<Self> {
        let width = layout.width;
        let height = layout.height;
        if width == 0 || height == 0 {
            return Ok(Self::empty(y_quant_01 as f32));
        }

        if !aq_enabled {
            return Self::new_bypass(layout, y_quant_01);
        }

        let blocks_w = layout.blocks_w;
        let blocks_h = layout.blocks_h;
        let padded_width = layout.padded_width;
        let y_buffer_stride = layout.y_buffer_stride;
        let v_samp_factor = layout.v_samp;

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
            aq_enabled: true,
            width,
            height,
            padded_width,
            y_buffer_stride,
            strip_stride: padded_width, // Always padded_width from layout
            blocks_w,
            blocks_h,
            pre_erosion_w,
            pre_erosion_h,
            pre_erosion_buffer: try_alloc_zeroed(pre_erosion_w * pre_erosion_buffer_rows)?,
            pre_erosion_buffer_rows,
            // Allocate with +2 padding (1 on each side) for branchless SIMD access
            // Index 0 = replicated left edge, indices 1..width+1 = data, index width+1 = replicated right edge
            row_prev_prev: try_alloc_zeroed(width + 2)?,
            row_prev: try_alloc_zeroed(width + 2)?,
            row_curr: try_alloc_zeroed(width + 2)?,
            pending_pre_erosion_row: None,
            pre_erosion_accum: try_alloc_zeroed(width)?,
            pre_erosion_temp: try_alloc_zeroed(width)?,
            y_imcu_buffers: [
                // Use y_buffer_stride (padded_width + 1) for proper edge replication
                try_alloc_zeroed(y_buffer_stride * imcu_height)?,
                try_alloc_zeroed(y_buffer_stride * imcu_height)?,
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
            #[cfg(target_arch = "x86_64")]
            archmage_token: X64V3Token::summon(),
        })
    }

    /// Creates a bypass-mode AQ state that skips computation.
    ///
    /// Keeps iMCU tracking fields for correct pipeline timing but skips
    /// expensive buffer allocations (~600KB-2.5MB savings).
    fn new_bypass(layout: &LayoutParams, y_quant_01: u16) -> Result<Self> {
        let blocks_w = layout.blocks_w;
        let blocks_h = layout.blocks_h;
        let v_samp_factor = layout.v_samp;
        let imcu_height = 8 * v_samp_factor;
        let total_imcu_rows = (layout.height + imcu_height - 1) / imcu_height;
        let blocks_per_imcu = blocks_w * v_samp_factor;
        let total_blocks = blocks_w * blocks_h;

        Ok(Self {
            aq_enabled: false,
            width: layout.width,
            height: layout.height,
            padded_width: layout.padded_width,
            y_buffer_stride: 0,
            strip_stride: layout.padded_width,
            blocks_w,
            blocks_h,
            pre_erosion_w: 0,
            pre_erosion_h: 0,
            // No expensive buffer allocations in bypass mode
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
            y_imcu_height: imcu_height,
            fuzzy_erosion_out: AlignedVec::new(0),
            imcu_aq_strengths: vec![0.0f32; blocks_per_imcu],
            all_aq_strengths: Vec::with_capacity(total_blocks),
            y_quant_01: y_quant_01 as f32,
            rows_received: 0,
            current_imcu_row: 0,
            total_imcu_rows,
            pre_erosion_rows_flushed: 0,
            pending_imcu_row: None,
            #[cfg(target_arch = "x86_64")]
            archmage_token: None,
        })
    }

    fn empty(y_quant_01: f32) -> Self {
        Self {
            aq_enabled: false,
            width: 0,
            height: 0,
            padded_width: 0,
            y_buffer_stride: 0,
            strip_stride: 0,
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
            #[cfg(target_arch = "x86_64")]
            archmage_token: None,
        }
    }

    /// Compute the number of valid blocks for an iMCU row.
    ///
    /// The last iMCU may be a partial row (fewer than v_samp block rows).
    fn compute_imcu_block_count(&self, imcu_row: usize) -> usize {
        let v_samp = self.y_imcu_height / 8;
        let mut valid_rows = 0;
        for by_offset in 0..v_samp {
            let global_by = imcu_row * v_samp + by_offset;
            if global_by >= self.blocks_h {
                break;
            }
            valid_rows += 1;
        }
        valid_rows * self.blocks_w
    }

    /// Bypass-mode strip processing: track iMCU boundaries without AQ computation.
    ///
    /// Returns the block count for a completed iMCU (filled with 0.0), or None.
    /// Pipeline timing matches the enabled path exactly: output is delayed by 1 iMCU.
    fn process_strip_bypass(&mut self, strip_y: usize, strip_height: usize) -> Option<usize> {
        // Track rows received (same as enabled path)
        for local_y in 0..strip_height {
            let global_y = strip_y + local_y;
            if global_y >= self.height {
                break;
            }
            self.rows_received = global_y + 1;
        }

        // Check iMCU boundary (same logic as enabled path)
        let imcu_height = self.y_imcu_height;
        let next_imcu_boundary = (self.current_imcu_row + 1) * imcu_height;

        if self.rows_received >= next_imcu_boundary.min(self.height) {
            // Finalize previously pending iMCU
            let valid_count = if let Some(pending) = self.pending_imcu_row.take() {
                let count = self.compute_imcu_block_count(pending);
                // imcu_aq_strengths is pre-initialized to 0.0, just accumulate
                self.all_aq_strengths
                    .extend_from_slice(&self.imcu_aq_strengths[..count]);
                Some(count)
            } else {
                None
            };

            // Mark just-completed iMCU as pending
            self.pending_imcu_row = Some(self.current_imcu_row);
            self.current_imcu_row += 1;

            return valid_count;
        }

        None
    }

    /// Process Y strip data and compute AQ for completed iMCU rows.
    ///
    /// # Arguments
    /// * `y_strip` - Y plane values for this strip (strip_stride × strip_height), 0-255 range.
    ///              Strip stride is `padded_width` from `LayoutParams`.
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

        if !self.aq_enabled {
            let count = self.process_strip_bypass(strip_y, strip_height)?;
            return Some(&self.imcu_aq_strengths[..count]);
        }

        // Use strip_stride for indexing input data
        let stride = self.strip_stride;
        let padded_width = self.padded_width;
        let y_buffer_stride = self.y_buffer_stride;

        // Process each row in the strip
        for local_y in 0..strip_height {
            let global_y = strip_y + local_y;
            if global_y >= self.height {
                break;
            }

            let row_start = local_y * stride;

            // Store padded row in Y iMCU buffer (includes replicated edge pixels for SIMD)
            // The input strip is expected to have stride >= padded_width with edge replication
            let imcu_local_y = global_y % self.y_imcu_height;
            let dest_start = imcu_local_y * y_buffer_stride;
            let padded_row = &y_strip[row_start..row_start + padded_width];
            self.y_imcu_buffers[self.y_imcu_current][dest_start..dest_start + padded_width]
                .copy_from_slice(padded_row);
            // Fill the extra edge pixel (position padded_width) for HF modulation rightmost block
            // This allows hf_modulation_sum_8x8 to safely read 9 consecutive elements
            let edge_val = padded_row[padded_width - 1];
            self.y_imcu_buffers[self.y_imcu_current][dest_start + padded_width] = edge_val;

            // Process pre-erosion using only actual width pixels (not padding)
            let row = &y_strip[row_start..row_start + self.width];
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
            let y_buffer_stride = self.y_buffer_stride;
            if valid_rows_in_imcu < imcu_height && valid_rows_in_imcu > 0 {
                let last_valid_row = valid_rows_in_imcu - 1;
                let src_start = last_valid_row * y_buffer_stride;
                // Copy full y_buffer_stride (includes the extra edge pixel)
                let src_end = src_start + y_buffer_stride;
                for fill_row in valid_rows_in_imcu..imcu_height {
                    let dest_start = fill_row * y_buffer_stride;
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

    /// Process Y strip and write AQ strengths to provided buffer (zero allocation).
    ///
    /// Same as `process_y_strip` but writes to an external buffer instead of
    /// returning a reference. This avoids the `.to_vec()` copy when the caller
    /// needs to hold the results while mutating other state.
    ///
    /// # Arguments
    /// * `y_strip` - Y pixel data for this strip
    /// * `strip_y` - Starting row index
    /// * `strip_height` - Number of rows in the strip
    /// * `out_buffer` - Buffer to write AQ strengths into (must be large enough)
    ///
    /// # Returns
    /// Number of AQ values written to `out_buffer`, or `None` if no iMCU completed.
    pub fn process_y_strip_into(
        &mut self,
        y_strip: &[f32],
        strip_y: usize,
        strip_height: usize,
        out_buffer: &mut [f32],
    ) -> Option<usize> {
        if self.width == 0 || self.height == 0 {
            return None;
        }

        if !self.aq_enabled {
            let count = self.process_strip_bypass(strip_y, strip_height)?;
            // Fill caller's buffer with 0.0 (imcu_aq_strengths is already zero)
            out_buffer[..count].copy_from_slice(&self.imcu_aq_strengths[..count]);
            return Some(count);
        }

        // Use strip_stride for indexing input data
        let stride = self.strip_stride;
        let padded_width = self.padded_width;
        let y_buffer_stride = self.y_buffer_stride;

        // Process each row in the strip
        for local_y in 0..strip_height {
            let global_y = strip_y + local_y;
            if global_y >= self.height {
                break;
            }

            let row_start = local_y * stride;

            // Store padded row in Y iMCU buffer
            let imcu_local_y = global_y % self.y_imcu_height;
            let dest_start = imcu_local_y * y_buffer_stride;
            let padded_row = &y_strip[row_start..row_start + padded_width];
            self.y_imcu_buffers[self.y_imcu_current][dest_start..dest_start + padded_width]
                .copy_from_slice(padded_row);
            // Fill the extra edge pixel (position padded_width) for HF modulation rightmost block
            let edge_val = padded_row[padded_width - 1];
            self.y_imcu_buffers[self.y_imcu_current][dest_start + padded_width] = edge_val;

            // Process pre-erosion using only actual width pixels
            let row = &y_strip[row_start..row_start + self.width];
            self.process_pre_erosion_row(row, global_y);

            self.rows_received = global_y + 1;
        }

        // Check if we completed an iMCU row
        let imcu_height = self.y_imcu_height;
        let next_imcu_boundary = (self.current_imcu_row + 1) * imcu_height;

        if self.rows_received >= next_imcu_boundary.min(self.height) {
            // Edge clamp: fill remaining rows of the iMCU buffer
            let valid_rows_in_imcu = self.rows_received - self.current_imcu_row * imcu_height;
            let y_buffer_stride = self.y_buffer_stride;
            if valid_rows_in_imcu < imcu_height && valid_rows_in_imcu > 0 {
                let last_valid_row = valid_rows_in_imcu - 1;
                let src_start = last_valid_row * y_buffer_stride;
                // Copy full y_buffer_stride (includes the extra edge pixel)
                let src_end = src_start + y_buffer_stride;
                for fill_row in valid_rows_in_imcu..imcu_height {
                    let dest_start = fill_row * y_buffer_stride;
                    self.y_imcu_buffers[self.y_imcu_current]
                        .copy_within(src_start..src_end, dest_start);
                }
            }
            // Finalize previously pending iMCU
            let valid_count = if let Some(pending) = self.pending_imcu_row.take() {
                let prev_buffer = 1 - self.y_imcu_current;
                let count = self.finalize_imcu_aq_with_buffer(pending, prev_buffer);
                // Copy to caller's buffer instead of accumulating
                out_buffer[..count].copy_from_slice(&self.imcu_aq_strengths[..count]);
                // Also accumulate for batch mode
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

            return valid_count;
        }

        None
    }

    /// Flush any pending iMCU AQ at end of image.
    ///
    /// Call after all strips have been processed to get the last iMCU's AQ.
    pub fn flush(&mut self) -> Option<&[f32]> {
        if let Some(pending) = self.pending_imcu_row.take() {
            let count = if self.aq_enabled {
                let prev_buffer = 1 - self.y_imcu_current;
                self.finalize_imcu_aq_with_buffer(pending, prev_buffer)
            } else {
                self.compute_imcu_block_count(pending)
            };
            // Only append the valid portion for partial iMCU rows
            self.all_aq_strengths
                .extend_from_slice(&self.imcu_aq_strengths[..count]);
            return Some(&self.imcu_aq_strengths[..count]);
        }
        None
    }

    /// Flush pending iMCU AQ into provided buffer (zero allocation).
    ///
    /// Same as `flush` but writes to an external buffer.
    ///
    /// # Arguments
    /// * `out_buffer` - Buffer to write AQ strengths into
    ///
    /// # Returns
    /// Number of AQ values written, or `None` if nothing pending.
    pub fn flush_into(&mut self, out_buffer: &mut [f32]) -> Option<usize> {
        if let Some(pending) = self.pending_imcu_row.take() {
            let count = if self.aq_enabled {
                let prev_buffer = 1 - self.y_imcu_current;
                self.finalize_imcu_aq_with_buffer(pending, prev_buffer)
            } else {
                self.compute_imcu_block_count(pending)
            };
            // Copy to caller's buffer (imcu_aq_strengths is 0.0 in bypass mode)
            out_buffer[..count].copy_from_slice(&self.imcu_aq_strengths[..count]);
            // Also accumulate for batch mode
            self.all_aq_strengths
                .extend_from_slice(&self.imcu_aq_strengths[..count]);

            // Debug: dump AQ map when image processing is complete
            if self.rows_received >= self.height && self.pending_imcu_row.is_none() {
                self.dump_aq_map_if_requested();
            }

            return Some(count);
        }
        None
    }

    /// Debug: dump AQ map to file if DUMP_AQ_MAP env var is set.
    fn dump_aq_map_if_requested(&self) {
        if let Ok(path) = std::env::var("DUMP_AQ_MAP")
            && let Ok(mut file) = std::fs::File::create(&path)
        {
            use std::io::Write;
            let w = self.blocks_w as u32;
            let h = self.blocks_h as u32;
            // Write header
            let _ = file.write_all(&w.to_le_bytes());
            let _ = file.write_all(&h.to_le_bytes());
            // Write AQ values
            for val in &self.all_aq_strengths {
                let _ = file.write_all(&val.to_le_bytes());
            }
            eprintln!(
                "AQ map dumped to {} ({}x{} blocks, {} values)",
                path,
                w,
                h,
                self.all_aq_strengths.len()
            );
        }
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

        // Debug: dump AQ map if requested
        self.dump_aq_map_if_requested();

        Ok(self.all_aq_strengths)
    }

    /// Copy row data into padded buffer with edge replication.
    /// Buffer layout: [edge_left, data[0..width], edge_right]
    /// where edge_left = data[0] and edge_right = data[width-1]
    #[inline(always)]
    fn copy_row_with_edge_replication(dst: &mut [f32], src: &[f32]) {
        let width = src.len();
        // Copy data at offset 1
        dst[1..1 + width].copy_from_slice(src);
        // Replicate edges
        dst[0] = src[0];
        dst[width + 1] = src[width - 1];
    }

    /// Process a single row for pre-erosion computation.
    fn process_pre_erosion_row(&mut self, row: &[f32], global_y: usize) {
        // Shift row buffers
        core::mem::swap(&mut self.row_prev_prev, &mut self.row_prev);
        core::mem::swap(&mut self.row_prev, &mut self.row_curr);
        Self::copy_row_with_edge_replication(&mut self.row_curr, row);

        // Initialize for first rows (boundary clamping)
        if global_y == 0 {
            Self::copy_row_with_edge_replication(&mut self.row_prev, row);
            Self::copy_row_with_edge_replication(&mut self.row_prev_prev, row);
        } else if global_y == 1 {
            // row_prev already contains correct data from previous iteration
            // Just need to duplicate it for row_prev_prev
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
        // Padded buffers: data is at indices 1..width+1, edges are replicated at 0 and width+1
        let row_above = &self.row_prev;
        let row_curr = &self.row_curr;
        let row_below = &self.row_curr; // Boundary clamping (same row)

        self.pre_erosion_temp.fill(0.0);

        // Use archmage SIMD when available - V4 (AVX-512) preferred, falls back to V3 (AVX2)
        #[cfg(target_arch = "x86_64")]
        {
            // V4 version handles its own dispatch - tries AVX-512 first, falls back to AVX2
            mage_pre_erosion_row_padded_v4(
                row_curr,
                row_above,
                row_below,
                self.width,
                &mut self.pre_erosion_temp,
            );
        }

        #[cfg(not(target_arch = "x86_64"))]
        pre_erosion_row_padded(
            row_curr,
            row_above,
            row_below,
            self.width,
            &mut self.pre_erosion_temp,
        );

        for x in 0..self.width {
            self.pre_erosion_accum[x] += self.pre_erosion_temp[x];
        }
    }

    fn compute_and_accumulate_pre_erosion(&mut self, row_y: usize) {
        // Padded buffers: data is at indices 1..width+1, edges are replicated at 0 and width+1
        let row_above = &self.row_prev_prev;
        let row_curr = &self.row_prev;
        let row_below = &self.row_curr;

        self.pre_erosion_temp.fill(0.0);

        // Use archmage SIMD when available - V4 (AVX-512) preferred, falls back to V3 (AVX2)
        #[cfg(target_arch = "x86_64")]
        {
            // V4 version handles its own dispatch - tries AVX-512 first, falls back to AVX2
            mage_pre_erosion_row_padded_v4(
                row_curr,
                row_above,
                row_below,
                self.width,
                &mut self.pre_erosion_temp,
            );
        }

        #[cfg(not(target_arch = "x86_64"))]
        pre_erosion_row_padded(
            row_curr,
            row_above,
            row_below,
            self.width,
            &mut self.pre_erosion_temp,
        );

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

            // Fuzzy erosion - compute_fuzzy_erosion_row_into does direct assignment, no zeroing needed
            let pe_y = global_by * 2;
            self.compute_fuzzy_erosion_row_into(pe_y, row_start, row_end);

            // Per-block modulations with padded buffer
            // Use y_buffer_stride (padded_width + 1) to allow reading 9 elements for HF modulation
            // Use archmage SIMD when available (fused HF+gamma, ~2x faster)
            #[cfg(target_arch = "x86_64")]
            if let Some(token) = self.archmage_token {
                mage_per_block_modulations_row(
                    token,
                    &self.y_imcu_buffers[y_buffer_idx],
                    self.y_buffer_stride, // stride (padded_width + 1 for edge replication)
                    by_offset,
                    blocks_w,
                    &mut self.fuzzy_erosion_out[row_start..row_end],
                    mul,
                    add,
                );
            } else {
                per_block_modulations_row(
                    &self.y_imcu_buffers[y_buffer_idx],
                    self.y_buffer_stride,
                    self.width,
                    self.height,
                    by_offset,
                    blocks_w,
                    &mut self.fuzzy_erosion_out[row_start..row_end],
                    mul,
                    add,
                );
            }

            #[cfg(not(target_arch = "x86_64"))]
            per_block_modulations_row(
                &self.y_imcu_buffers[y_buffer_idx],
                self.y_buffer_stride,
                self.width,
                self.height,
                by_offset,
                blocks_w,
                &mut self.fuzzy_erosion_out[row_start..row_end],
                mul,
                add,
            );

            // Convert to AQ strength - use slice iteration to eliminate bounds checks
            let qf_slice = &self.fuzzy_erosion_out[row_start..row_end];
            let aq_slice = &mut self.imcu_aq_strengths[row_start..row_end];
            for (qf, aq) in qf_slice.iter().zip(aq_slice.iter_mut()) {
                *aq = quant_field_to_aq_strength(*qf);
            }
        }

        valid_rows * blocks_w
    }

    fn compute_fuzzy_erosion_row_into(&mut self, pe_y_base: usize, start: usize, end: usize) {
        let pe_w = self.pre_erosion_w;
        let buffer_rows = self.pre_erosion_buffer_rows;
        let max_filled_row = self.pre_erosion_rows_flushed.saturating_sub(1) as isize;

        // Use autovectorized sorting network with runtime AVX2/SSE dispatch
        let simd_processed = compute_fuzzy_erosion_blocks_autovec(
            &self.pre_erosion_buffer,
            pe_w,
            buffer_rows,
            pe_y_base as isize,
            max_filled_row,
            start,
            end,
            &mut self.fuzzy_erosion_out,
        );

        // Handle remaining blocks with scalar code
        let scalar_start = start + simd_processed;
        if scalar_start >= end {
            return;
        }

        const MUL0: f32 = 0.125;
        const MUL1: f32 = 0.075;
        const MUL2: f32 = 0.06;
        const MUL3: f32 = 0.05;

        for bx in scalar_start..end {
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
    use crate::types::Subsampling;

    /// Helper to create a LayoutParams for tests (v_samp=2 like 4:2:0).
    fn test_layout(width: usize, height: usize) -> LayoutParams {
        LayoutParams::new(width, height, Subsampling::S420, false)
    }

    #[test]
    fn test_streaming_aq_creation() {
        let layout = test_layout(256, 256);
        let aq = StreamingAQ::new(&layout, 3, true).unwrap();
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
        let layout = test_layout(width, height);
        let mut streaming = StreamingAQ::new(&layout, y_quant_01, true).unwrap();
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

        let layout = test_layout(width, height);
        let mut streaming = StreamingAQ::new(&layout, y_quant_01, true).unwrap();
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

        let layout = test_layout(width, height);
        let mut streaming = StreamingAQ::new(&layout, y_quant_01, true).unwrap();
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
            assert!((0.0..1.0).contains(&v), "Invalid AQ value: {}", v);
        }
    }
}
