//! Shared strip processing pipeline for JPEG decoding.
//!
//! `StripProcessor` handles stages 2-3 of the decode pipeline:
//! - IDCT + dequantization of coefficients into strip buffers
//! - Chroma upsampling to full resolution
//! - Row accessor for color conversion
//!
//! Both the scanline decoder and buffered decoder share this code path.

use super::config::{ChromaUpsampling, IdctMethod, OutputTarget};
use super::idct_int::{
    idct_int_dc_only, idct_int_dc_only_unclamped, idct_int_tiered, idct_int_tiered_libjpeg,
    idct_int_tiered_libjpeg_unclamped, idct_int_tiered_unclamped,
};
use super::upsample::{
    upsample_h1v2_i16_libjpeg_strided, upsample_h1v2_i16_nearest_strided,
    upsample_h2v1_i16_libjpeg_strided, upsample_h2v1_i16_nearest_strided,
    upsample_h2v2_i16_libjpeg_strided, upsample_h2v2_i16_nearest_strided,
    upsample_h2v2_libjpeg_row,
};
use crate::error::Result;
use crate::foundation::alloc::try_alloc_maybeuninit;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::quant::dequantize_unzigzag_i32_into_partial;
use crate::types::Subsampling;

/// SIMD alignment for strip buffers (32 pixels = 64 bytes for i16).
const STRIP_ALIGNMENT: usize = 32;

/// Round up to next multiple of alignment.
#[inline]
const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Shared strip processing for IDCT, upsampling, and row access.
///
/// Owns the strip buffers and performs dequantization + IDCT into them,
/// then upsamples chroma. The caller handles entropy decoding and color
/// conversion/output formatting.
pub(super) struct StripProcessor {
    // Y strip buffer: full resolution, mcu_height rows
    pub y_strip: Vec<i16>,
    // Cb/Cr strip buffers at native chroma resolution
    pub cb_strip: Vec<i16>,
    pub cr_strip: Vec<i16>,

    // Chroma dimensions
    pub chroma_strip_width: usize,
    pub chroma_strip_stride: usize,
    pub chroma_strip_height: usize,

    // Upsampled chroma buffers (full resolution, for non-4:4:4)
    pub cb_upsampled: Vec<i16>,
    pub cr_upsampled: Vec<i16>,

    // Layout
    pub strip_width: usize,
    pub strip_stride: usize,
    pub mcu_height: usize,

    // Sampling factors
    pub h_samp: [u8; 3],
    pub v_samp: [u8; 3],
    pub max_h_samp: u8,
    pub subsampling: Subsampling,
    #[allow(dead_code)]
    pub num_components: u8,

    // Cross-strip chroma context for vertical upsampling boundary fix.
    // Stores the last chroma row from the previous MCU row's strip so that
    // the top boundary of the current strip uses correct vertical interpolation
    // instead of edge duplication.
    prev_cb_row: Vec<i16>,
    prev_cr_row: Vec<i16>,
    has_prev_context: bool,

    // Next-MCU context for bottom boundary fixup.
    // Stores the first chroma row from the next MCU row's strip so that
    // the bottom boundary of the current strip uses correct vertical interpolation.
    pub next_cb_row: Vec<i16>,
    pub next_cr_row: Vec<i16>,
    pub has_next_context: bool,

    // Deferred bottom row for streaming path.
    // When the streaming decoder pre-decodes the next MCU row to get bottom
    // context, the corrected last-row chroma is stored here.
    deferred_y_row: Vec<i16>,
    deferred_cb_row: Vec<i16>,
    deferred_cr_row: Vec<i16>,
    pub has_deferred_bottom: bool,

    // Reusable IDCT working buffers
    pub dequant_buf: [i32; DCT_BLOCK_SIZE],

    // Config
    pub chroma_upsampling: ChromaUpsampling,
    pub idct_method: IdctMethod,
    pub output_target: OutputTarget,
}

impl StripProcessor {
    /// Create a dummy strip processor for buffered mode (progressive JPEGs).
    ///
    /// In buffered mode, strips are unused — we serve from a pre-decoded buffer.
    pub fn new_dummy(subsampling: Subsampling) -> Self {
        Self {
            y_strip: Vec::new(),
            cb_strip: Vec::new(),
            cr_strip: Vec::new(),
            chroma_strip_width: 0,
            chroma_strip_stride: 0,
            chroma_strip_height: 0,
            cb_upsampled: Vec::new(),
            cr_upsampled: Vec::new(),
            strip_width: 0,
            strip_stride: 0,
            mcu_height: 8,
            h_samp: [1, 1, 1],
            v_samp: [1, 1, 1],
            max_h_samp: 1,
            subsampling,
            num_components: 3,
            prev_cb_row: Vec::new(),
            prev_cr_row: Vec::new(),
            has_prev_context: false,
            next_cb_row: Vec::new(),
            next_cr_row: Vec::new(),
            has_next_context: false,
            deferred_y_row: Vec::new(),
            deferred_cb_row: Vec::new(),
            deferred_cr_row: Vec::new(),
            has_deferred_bottom: false,
            dequant_buf: [0i32; DCT_BLOCK_SIZE],
            chroma_upsampling: ChromaUpsampling::default(),
            idct_method: IdctMethod::default(),
            output_target: OutputTarget::default(),
        }
    }

    /// Create a new strip processor with allocated buffers.
    pub fn new(
        width: u32,
        num_components: u8,
        h_samp: [u8; 3],
        v_samp: [u8; 3],
        chroma_upsampling: ChromaUpsampling,
        idct_method: IdctMethod,
        output_target: OutputTarget,
    ) -> Result<Self> {
        let is_grayscale = num_components == 1;

        let (max_h_samp, max_v_samp) = if is_grayscale {
            (h_samp[0], v_samp[0])
        } else {
            (
                h_samp.iter().copied().max().unwrap_or(1),
                v_samp.iter().copied().max().unwrap_or(1),
            )
        };

        // Determine subsampling by comparing chroma factors to luma factors.
        // Using max factors alone is wrong: e.g. all components at (1,2) has
        // max_v=2 but no actual subsampling — it's 4:4:4 with 8×16 MCUs.
        let subsampling = if is_grayscale {
            Subsampling::S444
        } else {
            let h_ratio = h_samp[0] / h_samp[1].max(1);
            let v_ratio = v_samp[0] / v_samp[1].max(1);
            match (h_ratio, v_ratio) {
                (1, 1) => Subsampling::S444,
                (2, 1) => Subsampling::S422,
                (2, 2) => Subsampling::S420,
                (1, 2) => Subsampling::S440,
                _ => Subsampling::S420,
            }
        };

        let mcu_width = max_h_samp as usize * 8;
        let mcu_cols = (width as usize + mcu_width - 1) / mcu_width;
        let mcu_height = max_v_samp as usize * 8;

        // Y strip: scaled resolution with SIMD-aligned stride
        let strip_width = mcu_cols * mcu_width;
        let strip_stride = align_up(strip_width, STRIP_ALIGNMENT);
        let y_strip_size = strip_stride * mcu_height;

        // Chroma strip: at native (potentially subsampled) resolution.
        // Use actual chroma sampling factors for dimensions, not hardcoded 1×8.
        // For all-same-sampling (e.g. all components 1×2), chroma is full resolution.
        let chroma_h = if is_grayscale { 0 } else { h_samp[1] as usize };
        let chroma_v = if is_grayscale { 0 } else { v_samp[1] as usize };
        let chroma_strip_width = if is_grayscale {
            0
        } else {
            mcu_cols * chroma_h * 8
        };
        let chroma_strip_stride = if is_grayscale {
            0
        } else {
            align_up(chroma_strip_width, STRIP_ALIGNMENT)
        };
        let chroma_strip_height = if is_grayscale { 0 } else { chroma_v * 8 };
        let chroma_strip_size = chroma_strip_stride * chroma_strip_height;

        // Allocate strip buffers
        let y_strip = try_alloc_maybeuninit(y_strip_size, "Y strip buffer")?;

        let (cb_strip, cr_strip) = if is_grayscale {
            (Vec::new(), Vec::new())
        } else {
            (
                try_alloc_maybeuninit(chroma_strip_size, "Cb strip buffer")?,
                try_alloc_maybeuninit(chroma_strip_size, "Cr strip buffer")?,
            )
        };

        // Upsampled chroma buffers (only for non-4:4:4 color images)
        let needs_vertical_upsample = matches!(subsampling, Subsampling::S420 | Subsampling::S440);
        let (cb_upsampled, cr_upsampled) = if !is_grayscale && subsampling != Subsampling::S444 {
            let upsampled_size = strip_stride * mcu_height;
            (
                try_alloc_maybeuninit(upsampled_size, "Cb upsampled buffer")?,
                try_alloc_maybeuninit(upsampled_size, "Cr upsampled buffer")?,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        // Previous/next chroma row context for cross-strip vertical interpolation
        let (prev_cb_row, prev_cr_row, next_cb_row, next_cr_row) =
            if !is_grayscale && needs_vertical_upsample {
                (
                    try_alloc_maybeuninit(chroma_strip_stride, "prev Cb context row")?,
                    try_alloc_maybeuninit(chroma_strip_stride, "prev Cr context row")?,
                    try_alloc_maybeuninit(chroma_strip_stride, "next Cb context row")?,
                    try_alloc_maybeuninit(chroma_strip_stride, "next Cr context row")?,
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };

        // Deferred bottom row buffers for streaming bottom-boundary fixup
        let (deferred_y_row, deferred_cb_row, deferred_cr_row) =
            if !is_grayscale && needs_vertical_upsample {
                (
                    try_alloc_maybeuninit(strip_stride, "deferred Y row")?,
                    try_alloc_maybeuninit(strip_stride, "deferred Cb row")?,
                    try_alloc_maybeuninit(strip_stride, "deferred Cr row")?,
                )
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        Ok(Self {
            y_strip,
            cb_strip,
            cr_strip,
            chroma_strip_width,
            chroma_strip_stride,
            chroma_strip_height,
            cb_upsampled,
            cr_upsampled,
            strip_width,
            strip_stride,
            mcu_height,
            h_samp,
            v_samp,
            max_h_samp,
            subsampling,
            num_components,
            prev_cb_row,
            prev_cr_row,
            has_prev_context: false,
            next_cb_row,
            next_cr_row,
            has_next_context: false,
            deferred_y_row,
            deferred_cb_row,
            deferred_cr_row,
            has_deferred_bottom: false,
            dequant_buf: [0i32; DCT_BLOCK_SIZE],
            chroma_upsampling,
            idct_method,
            output_target,
        })
    }

    /// The number of MCU columns.
    #[inline]
    pub fn mcu_cols(&self) -> usize {
        // strip_width = mcu_cols * mcu_width, mcu_width = max_h_samp * 8
        self.strip_width / (self.max_h_samp as usize * 8)
    }

    /// Perform IDCT on a single block and write to the appropriate strip buffer.
    ///
    /// `comp_idx`: 0=Y, 1=Cb, 2=Cr
    /// `mcu_x`: MCU column index
    /// `h`, `v`: Block position within the MCU (for multi-block components)
    /// `coeffs`: Entropy-decoded coefficients (zigzag order)
    /// `coeff_count`: Number of non-zero coefficients
    /// `quant`: Quantization table for this component
    #[inline(always)]
    pub fn idct_block(
        &mut self,
        comp_idx: usize,
        mcu_x: usize,
        h: usize,
        v: usize,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        coeff_count: u8,
        quant: &[u16; DCT_BLOCK_SIZE],
    ) {
        // Calculate destination in strip buffer.
        // Chroma offsets use per-component h_samp/v_samp to handle both subsampled
        // and all-same-sampling cases (e.g. all components 1×2 = S444 with 8×16 MCUs).
        let (strip, stride) = match comp_idx {
            0 => {
                let x_offset = mcu_x * self.max_h_samp as usize * 8 + h * 8;
                let y_offset = v * 8 * self.strip_stride;
                (&mut self.y_strip[y_offset + x_offset..], self.strip_stride)
            }
            1 => {
                let x_offset = mcu_x * self.h_samp[1] as usize * 8 + h * 8;
                let y_offset = v * 8 * self.chroma_strip_stride;
                (
                    &mut self.cb_strip[y_offset + x_offset..],
                    self.chroma_strip_stride,
                )
            }
            _ => {
                let x_offset = mcu_x * self.h_samp[2] as usize * 8 + h * 8;
                let y_offset = v * 8 * self.chroma_strip_stride;
                (
                    &mut self.cr_strip[y_offset + x_offset..],
                    self.chroma_strip_stride,
                )
            }
        };

        let unclamped = self.output_target.needs_unclamped_idct();

        if coeff_count <= 1 {
            let dc = coeffs[0] as i32 * quant[0] as i32;
            if unclamped {
                idct_int_dc_only_unclamped(dc, strip, stride);
            } else {
                idct_int_dc_only(dc, strip, stride);
            }
        } else {
            dequantize_unzigzag_i32_into_partial(coeffs, quant, &mut self.dequant_buf, coeff_count);
            match (unclamped, self.idct_method) {
                (false, IdctMethod::Libjpeg) => {
                    idct_int_tiered_libjpeg(&mut self.dequant_buf, strip, stride, coeff_count);
                }
                (false, IdctMethod::Jpegli) => {
                    idct_int_tiered(&mut self.dequant_buf, strip, stride, coeff_count);
                }
                (true, IdctMethod::Libjpeg) => {
                    idct_int_tiered_libjpeg_unclamped(
                        &mut self.dequant_buf,
                        strip,
                        stride,
                        coeff_count,
                    );
                }
                (true, IdctMethod::Jpegli) => {
                    idct_int_tiered_unclamped(&mut self.dequant_buf, strip, stride, coeff_count);
                }
            }
        }
    }

    /// Whether this subsampling mode needs vertical chroma upsampling.
    #[inline]
    pub fn needs_vertical_upsample(&self) -> bool {
        matches!(self.subsampling, Subsampling::S420 | Subsampling::S440)
    }

    /// Edge-replicate the last real chroma row/column over MCU padding.
    ///
    /// The encoder pads MCU boundaries by replicating pixel rows/columns
    /// before DCT, but IDCT rounding means decoded padding differs slightly
    /// from the last real sample. libjpeg-turbo handles this via
    /// `set_bottom_pointers()` (vertical) and using `downsampled_width`
    /// (horizontal). We overwrite the padding data to match.
    ///
    /// `image_width`: the actual image width in pixels.
    /// `image_height`: the actual image height in pixels.
    /// `mcu_row`: the current MCU row index (0-based).
    pub fn truncate_chroma_padding(
        &mut self,
        image_width: usize,
        image_height: usize,
        mcu_row: usize,
    ) {
        if !matches!(
            self.subsampling,
            Subsampling::S420 | Subsampling::S440 | Subsampling::S422
        ) {
            return;
        }

        let h_ratio = if self.h_samp[0] > self.h_samp[1] {
            self.h_samp[0] as usize / self.h_samp[1] as usize
        } else {
            1
        };
        let v_ratio = self.mcu_height / self.chroma_strip_height.max(1);
        let stride = self.chroma_strip_stride;
        let strip_w = self.chroma_strip_width;

        // Vertical padding (last MCU row only)
        if v_ratio > 1 {
            let downsampled_h = (image_height + v_ratio - 1) / v_ratio;
            let real_rows = self
                .chroma_strip_height
                .min(downsampled_h.saturating_sub(mcu_row * self.chroma_strip_height));
            if real_rows < self.chroma_strip_height {
                let last_real = (real_rows - 1) * stride;
                for pad_row in real_rows..self.chroma_strip_height {
                    let dst = pad_row * stride;
                    self.cb_strip
                        .copy_within(last_real..last_real + stride, dst);
                    self.cr_strip
                        .copy_within(last_real..last_real + stride, dst);
                }
            }
        }

        // Horizontal padding (all MCU rows if image width not MCU-aligned)
        if h_ratio > 0 {
            let downsampled_w = (image_width + h_ratio - 1) / h_ratio;
            if downsampled_w < strip_w {
                let rows = self.chroma_strip_height;
                for row in 0..rows {
                    let row_off = row * stride;
                    let last_val_cb = self.cb_strip[row_off + downsampled_w - 1];
                    let last_val_cr = self.cr_strip[row_off + downsampled_w - 1];
                    for col in downsampled_w..strip_w {
                        self.cb_strip[row_off + col] = last_val_cb;
                        self.cr_strip[row_off + col] = last_val_cr;
                    }
                }
            }
        }
    }

    /// Upsample chroma buffers to full resolution.
    ///
    /// Call this after all blocks in the MCU row have been IDCT'd.
    /// For vertical upsampling modes (4:2:0, 4:4:0), this also applies
    /// cross-strip boundary correction using the previous strip's last
    /// chroma row, then saves the current strip's last row for next time.
    pub fn upsample_chroma(&mut self) {
        self.upsample_chroma_core();
        match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => self.save_last_chroma_row(),
            _ => {}
        }
        self.has_next_context = false;
    }

    /// Core upsampling: upsample + apply top and bottom boundary fixups.
    ///
    /// Does NOT save last chroma row — caller controls that separately
    /// so the streaming path can defer it.
    fn upsample_chroma_core(&mut self) {
        match self.subsampling {
            Subsampling::S444 => {} // No upsampling needed
            Subsampling::S422 => self.upsample_h2v1(),
            Subsampling::S420 => {
                self.upsample_h2v2();
                self.fixup_vertical_boundary();
                self.fixup_bottom_boundary();
            }
            Subsampling::S440 => {
                self.upsample_h1v2();
                self.fixup_vertical_boundary();
                self.fixup_bottom_boundary();
            }
        }
    }

    /// Get Y/Cb/Cr row slices for a given row within the current MCU row.
    ///
    /// Returns (y_row, cb_row, cr_row) slices of `cols` pixels each.
    /// For subsampled images, cb/cr come from the upsampled buffers.
    /// When a deferred bottom row is available for the last row of an MCU,
    /// returns the corrected chroma from the deferred buffers instead.
    #[inline(always)]
    pub fn row_planes(&self, row_in_mcu: usize, cols: usize) -> (&[i16], &[i16], &[i16]) {
        // Deferred bottom row: return corrected Y/Cb/Cr for last MCU row
        if self.has_deferred_bottom && row_in_mcu == self.mcu_height - 1 {
            return (
                &self.deferred_y_row[..cols],
                &self.deferred_cb_row[..cols],
                &self.deferred_cr_row[..cols],
            );
        }

        let offset = row_in_mcu * self.strip_stride;
        let y = &self.y_strip[offset..offset + cols];
        let (cb, cr) = if self.subsampling == Subsampling::S444 {
            (
                &self.cb_strip[offset..offset + cols],
                &self.cr_strip[offset..offset + cols],
            )
        } else {
            (
                &self.cb_upsampled[offset..offset + cols],
                &self.cr_upsampled[offset..offset + cols],
            )
        };
        (y, cb, cr)
    }

    /// Get Y row slice for grayscale output.
    #[inline(always)]
    pub fn y_row(&self, row_in_mcu: usize, cols: usize) -> &[i16] {
        let offset = row_in_mcu * self.strip_stride;
        &self.y_strip[offset..offset + cols]
    }

    /// Get native-resolution chroma row slices (before upsampling).
    ///
    /// `chroma_row` is the row index within the chroma strip (0..chroma_strip_height).
    /// `cols` is the number of chroma samples to return.
    ///
    /// Returns `(cb_row, cr_row)` at native chroma resolution.
    #[inline(always)]
    pub fn chroma_row_native(&self, chroma_row: usize, cols: usize) -> (&[i16], &[i16]) {
        let offset = chroma_row * self.chroma_strip_stride;
        (
            &self.cb_strip[offset..offset + cols],
            &self.cr_strip[offset..offset + cols],
        )
    }

    // =========================================================================
    // Upsampling implementations
    // =========================================================================

    /// Upsample a single channel using a strided function pointer.
    fn upsample_channel(
        upsample_fn: fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize),
        input: &[i16],
        in_width: usize,
        in_stride: usize,
        in_height: usize,
        output: &mut [i16],
        out_width: usize,
        out_stride: usize,
        out_height: usize,
    ) {
        upsample_fn(
            input, in_width, in_stride, in_height, output, out_width, out_stride, out_height,
        );
    }

    /// Horizontal 2x upsampling (4:2:2) with configurable filter.
    fn upsample_h2v1(&mut self) {
        type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
        let upsample_fn: StridedFn = match self.chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h2v1_i16_libjpeg_strided,
            ChromaUpsampling::NearestNeighbor => upsample_h2v1_i16_nearest_strided,
        };
        self.upsample_both_channels(upsample_fn);
    }

    /// Vertical 2x upsampling (4:4:0) with configurable filter.
    fn upsample_h1v2(&mut self) {
        type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
        let upsample_fn: StridedFn = match self.chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h1v2_i16_libjpeg_strided,
            ChromaUpsampling::NearestNeighbor => upsample_h1v2_i16_nearest_strided,
        };
        self.upsample_both_channels(upsample_fn);
    }

    /// Both horizontal and vertical 2x upsampling (4:2:0) with configurable filter.
    fn upsample_h2v2(&mut self) {
        type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
        let upsample_fn: StridedFn = match self.chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h2v2_i16_libjpeg_strided,
            ChromaUpsampling::NearestNeighbor => upsample_h2v2_i16_nearest_strided,
        };
        self.upsample_both_channels(upsample_fn);
    }

    /// Apply a strided upsample function to both Cb and Cr channels.
    fn upsample_both_channels(
        &mut self,
        upsample_fn: fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize),
    ) {
        let in_width = self.chroma_strip_width;
        let in_stride = self.chroma_strip_stride;
        let in_height = self.chroma_strip_height;
        let out_width = self.strip_width;
        let out_stride = self.strip_stride;
        let out_height = self.mcu_height;

        Self::upsample_channel(
            upsample_fn,
            &self.cb_strip,
            in_width,
            in_stride,
            in_height,
            &mut self.cb_upsampled,
            out_width,
            out_stride,
            out_height,
        );
        Self::upsample_channel(
            upsample_fn,
            &self.cr_strip,
            in_width,
            in_stride,
            in_height,
            &mut self.cr_upsampled,
            out_width,
            out_stride,
            out_height,
        );
    }

    /// Fix the last output row(s) of the upsampled buffers using next strip context.
    ///
    /// Mirrors `fixup_vertical_boundary()` for the bottom edge. The normal
    /// upsampling duplicates the last chroma row as its own vertical neighbor
    /// (edge clamping). When we have the first chroma row from the next MCU
    /// row's strip, we use it as the correct neighbor.
    fn fixup_bottom_boundary(&mut self) {
        if !self.has_next_context {
            return;
        }

        let in_width = self.chroma_strip_width;
        let out_width = self.strip_width;
        let out_stride = self.strip_stride;

        match self.subsampling {
            Subsampling::S420 => {
                self.fixup_h2v2_last_row(in_width, out_width, out_stride);
            }
            Subsampling::S440 => {
                self.fixup_h1v2_last_row(in_width, out_width, out_stride);
            }
            _ => {}
        }
    }

    /// Fix h2v2 last output row using next chroma context.
    fn fixup_h2v2_last_row(&mut self, in_width: usize, out_width: usize, out_stride: usize) {
        let last_out_offset = (self.mcu_height - 1) * out_stride;
        let last_chroma_offset = (self.chroma_strip_height - 1) * self.chroma_strip_stride;

        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                let cb_out = &mut self.cb_upsampled[last_out_offset..last_out_offset + out_stride];
                upsample_h2v2_libjpeg_row(
                    &self.cb_strip[last_chroma_offset..last_chroma_offset + in_width],
                    &self.next_cb_row[..in_width],
                    cb_out,
                    in_width,
                    out_width,
                    false, // is_upper = false → lower half
                );
                let cr_out = &mut self.cr_upsampled[last_out_offset..last_out_offset + out_stride];
                upsample_h2v2_libjpeg_row(
                    &self.cr_strip[last_chroma_offset..last_chroma_offset + in_width],
                    &self.next_cr_row[..in_width],
                    cr_out,
                    in_width,
                    out_width,
                    false,
                );
            }
            ChromaUpsampling::NearestNeighbor => {
                // No interpolation vertically, no fixup needed
            }
        }
    }

    /// Fix h1v2 last output row using next chroma context.
    fn fixup_h1v2_last_row(&mut self, in_width: usize, out_width: usize, _out_stride: usize) {
        let w = in_width.min(out_width);
        let last_out_offset = (self.mcu_height - 1) * self.strip_stride;
        let last_chroma_offset = (self.chroma_strip_height - 1) * self.chroma_strip_stride;

        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                // h1v2 libjpeg: (near * 3 + far + bias) >> 2, bias=2 for lower
                for x in 0..w {
                    let near_cb = self.cb_strip[last_chroma_offset + x] as i32;
                    let far_cb = self.next_cb_row[x] as i32;
                    self.cb_upsampled[last_out_offset + x] =
                        ((near_cb * 3 + far_cb + 2) >> 2) as i16;

                    let near_cr = self.cr_strip[last_chroma_offset + x] as i32;
                    let far_cr = self.next_cr_row[x] as i32;
                    self.cr_upsampled[last_out_offset + x] =
                        ((near_cr * 3 + far_cr + 2) >> 2) as i16;
                }
            }
            ChromaUpsampling::NearestNeighbor => {
                // No interpolation vertically, no fixup needed
            }
        }
    }

    /// Compute deferred bottom row chroma for the streaming path.
    ///
    /// Called after the next MCU row's strip has been IDCT'd into cb_strip/cr_strip.
    /// Uses `prev_cb_row` (still holding the previous MCU's last chroma row)
    /// and `cb_strip[0]` (first chroma row of the just-decoded next MCU) to
    /// recompute the bottom output row with correct interpolation.
    ///
    /// Results are stored in `deferred_cb_row`/`deferred_cr_row`.
    pub fn compute_deferred_bottom(&mut self) {
        let in_width = self.chroma_strip_width;
        let out_width = self.strip_width;

        match self.subsampling {
            Subsampling::S420 => {
                self.compute_deferred_h2v2(in_width, out_width);
            }
            Subsampling::S440 => {
                self.compute_deferred_h1v2(in_width, out_width);
            }
            _ => {}
        }

        self.has_deferred_bottom = true;
    }

    /// Compute deferred h2v2 bottom row.
    fn compute_deferred_h2v2(&mut self, in_width: usize, out_width: usize) {
        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                let cb_out = &mut self.deferred_cb_row[..out_width];
                upsample_h2v2_libjpeg_row(
                    &self.prev_cb_row[..in_width],
                    &self.cb_strip[..in_width],
                    cb_out,
                    in_width,
                    out_width,
                    false, // lower
                );
                let cr_out = &mut self.deferred_cr_row[..out_width];
                upsample_h2v2_libjpeg_row(
                    &self.prev_cr_row[..in_width],
                    &self.cr_strip[..in_width],
                    cr_out,
                    in_width,
                    out_width,
                    false,
                );
            }
            ChromaUpsampling::NearestNeighbor => {
                // No vertical interpolation, just copy from upsampled
                let last_out_offset = (self.mcu_height - 1) * self.strip_stride;
                self.deferred_cb_row[..out_width].copy_from_slice(
                    &self.cb_upsampled[last_out_offset..last_out_offset + out_width],
                );
                self.deferred_cr_row[..out_width].copy_from_slice(
                    &self.cr_upsampled[last_out_offset..last_out_offset + out_width],
                );
            }
        }
    }

    /// Compute deferred h1v2 bottom row.
    fn compute_deferred_h1v2(&mut self, in_width: usize, out_width: usize) {
        let w = in_width.min(out_width);

        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                for x in 0..w {
                    let near_cb = self.prev_cb_row[x] as i32;
                    let far_cb = self.cb_strip[x] as i32;
                    self.deferred_cb_row[x] = ((near_cb * 3 + far_cb + 2) >> 2) as i16;

                    let near_cr = self.prev_cr_row[x] as i32;
                    let far_cr = self.cr_strip[x] as i32;
                    self.deferred_cr_row[x] = ((near_cr * 3 + far_cr + 2) >> 2) as i16;
                }
            }
            ChromaUpsampling::NearestNeighbor => {
                // No vertical interpolation, just copy from upsampled
                let last_out_offset = (self.mcu_height - 1) * self.strip_stride;
                self.deferred_cb_row[..out_width].copy_from_slice(
                    &self.cb_upsampled[last_out_offset..last_out_offset + out_width],
                );
                self.deferred_cr_row[..out_width].copy_from_slice(
                    &self.cr_upsampled[last_out_offset..last_out_offset + out_width],
                );
            }
        }
    }

    /// Save the Y data for the last MCU row into the deferred buffer.
    ///
    /// Called by the streaming path before overwriting y_strip with the next MCU.
    pub fn save_deferred_y_row(&mut self) {
        let last_y_offset = (self.mcu_height - 1) * self.strip_stride;
        let w = self.strip_width;
        self.deferred_y_row[..w].copy_from_slice(&self.y_strip[last_y_offset..last_y_offset + w]);
    }

    /// Save the last chroma row from the current strip for cross-boundary context.
    fn save_last_chroma_row(&mut self) {
        let last_row_offset = (self.chroma_strip_height - 1) * self.chroma_strip_stride;
        let w = self.chroma_strip_width;
        self.prev_cb_row[..w].copy_from_slice(&self.cb_strip[last_row_offset..last_row_offset + w]);
        self.prev_cr_row[..w].copy_from_slice(&self.cr_strip[last_row_offset..last_row_offset + w]);
        self.has_prev_context = true;
    }

    /// Fix output row 0 of the upsampled buffers using previous strip context.
    ///
    /// The normal upsampling duplicates the top chroma row as its own vertical
    /// neighbor (edge clamping). When we have context from the previous strip,
    /// we can use the correct neighbor for proper interpolation.
    fn fixup_vertical_boundary(&mut self) {
        if !self.has_prev_context {
            return;
        }

        let in_width = self.chroma_strip_width;
        let out_width = self.strip_width;
        let out_stride = self.strip_stride;

        match self.subsampling {
            Subsampling::S420 => {
                // h2v2: output row 0 = top half of chroma row 0
                // Vertical neighbor should be prev strip's last row, not chroma row 0
                self.fixup_h2v2_row0(in_width, out_width, out_stride);
            }
            Subsampling::S440 => {
                // h1v2: output row 0 = top half of chroma row 0
                // Vertical neighbor should be prev strip's last row
                self.fixup_h1v2_row0(in_width, out_width, out_stride);
            }
            _ => {}
        }
    }

    /// Fix h2v2 output row 0 using previous chroma context.
    ///
    /// Borrows strip fields directly — `cb_strip`/`cr_strip` (read) and
    /// `cb_upsampled`/`cr_upsampled` (write) are disjoint fields, so no
    /// temporary buffer is needed.
    fn fixup_h2v2_row0(&mut self, in_width: usize, out_width: usize, out_stride: usize) {
        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                let cb_out = &mut self.cb_upsampled[..out_stride];
                upsample_h2v2_libjpeg_row(
                    &self.cb_strip[..in_width],
                    &self.prev_cb_row[..in_width],
                    cb_out,
                    in_width,
                    out_width,
                    true, // is_upper
                );
                let cr_out = &mut self.cr_upsampled[..out_stride];
                upsample_h2v2_libjpeg_row(
                    &self.cr_strip[..in_width],
                    &self.prev_cr_row[..in_width],
                    cr_out,
                    in_width,
                    out_width,
                    true,
                );
            }
            ChromaUpsampling::NearestNeighbor => {
                // No vertical interpolation, no fixup needed
            }
        }
    }

    /// Fix h1v2 output row 0 using previous chroma context.
    fn fixup_h1v2_row0(&mut self, in_width: usize, out_width: usize, out_stride: usize) {
        let _ = out_stride;
        let w = in_width.min(out_width);

        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                // h1v2 libjpeg: (near * 3 + far + bias) >> 2, bias=1 for upper
                for x in 0..w {
                    let near_cb = self.cb_strip[x] as i32;
                    let far_cb = self.prev_cb_row[x] as i32;
                    self.cb_upsampled[x] = ((near_cb * 3 + far_cb + 1) >> 2) as i16;

                    let near_cr = self.cr_strip[x] as i32;
                    let far_cr = self.prev_cr_row[x] as i32;
                    self.cr_upsampled[x] = ((near_cr * 3 + far_cr + 1) >> 2) as i16;
                }
            }
            ChromaUpsampling::NearestNeighbor => {
                // No vertical interpolation, no fixup needed
            }
        }
    }
}
