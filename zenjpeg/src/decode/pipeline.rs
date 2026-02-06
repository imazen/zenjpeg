//! Shared strip processing pipeline for JPEG decoding.
//!
//! `StripProcessor` handles stages 2-3 of the decode pipeline:
//! - IDCT + dequantization of coefficients into strip buffers
//! - Chroma upsampling to full resolution
//! - Row accessor for color conversion
//!
//! Both the scanline decoder and buffered decoder share this code path.

use super::config::ChromaUpsampling;
use super::idct_int::{idct_int_dc_only, idct_int_tiered, idct_int_tiered_libjpeg};
use super::upsample::{
    upsample_h1v2_i16_fancy_strided, upsample_h1v2_i16_libjpeg_strided,
    upsample_h1v2_i16_nearest_strided, upsample_h2v1_i16_fancy_strided,
    upsample_h2v1_i16_libjpeg_strided, upsample_h2v1_i16_nearest_strided,
    upsample_h2v2_i16_fancy_strided, upsample_h2v2_i16_libjpeg_strided,
    upsample_h2v2_i16_nearest_strided, upsample_h2v2_libjpeg_row, upsample_row_h2_fancy_bilinear,
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

    // Reusable IDCT working buffers
    pub dequant_buf: [i32; DCT_BLOCK_SIZE],

    // Config
    pub chroma_upsampling: ChromaUpsampling,
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
            dequant_buf: [0i32; DCT_BLOCK_SIZE],
            chroma_upsampling: ChromaUpsampling::default(),
        }
    }

    /// Create a new strip processor with allocated buffers.
    pub fn new(
        width: u32,
        num_components: u8,
        h_samp: [u8; 3],
        v_samp: [u8; 3],
        chroma_upsampling: ChromaUpsampling,
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

        let subsampling = if is_grayscale {
            Subsampling::S444
        } else {
            match (max_h_samp, max_v_samp) {
                (1, 1) => Subsampling::S444,
                (2, 1) => Subsampling::S422,
                (2, 2) => Subsampling::S420,
                (1, 2) => Subsampling::S440,
                _ => Subsampling::S420,
            }
        };

        let mcu_width = max_h_samp as usize * 8;
        let mcu_height = max_v_samp as usize * 8;
        let mcu_cols = (width as usize + mcu_width - 1) / mcu_width;

        // Y strip: full resolution with SIMD-aligned stride
        let strip_width = mcu_cols * mcu_width;
        let strip_stride = align_up(strip_width, STRIP_ALIGNMENT);
        let y_strip_size = strip_stride * mcu_height;

        // Chroma strip: at native (potentially subsampled) resolution
        let chroma_strip_width = if is_grayscale { 0 } else { mcu_cols * 8 };
        let chroma_strip_stride = if is_grayscale {
            0
        } else {
            align_up(chroma_strip_width, STRIP_ALIGNMENT)
        };
        let chroma_strip_height = if is_grayscale { 0 } else { 8 };
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

        // Previous chroma row context for cross-strip vertical interpolation
        let (prev_cb_row, prev_cr_row) = if !is_grayscale && needs_vertical_upsample {
            (
                vec![0i16; chroma_strip_stride],
                vec![0i16; chroma_strip_stride],
            )
        } else {
            (Vec::new(), Vec::new())
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
            dequant_buf: [0i32; DCT_BLOCK_SIZE],
            chroma_upsampling,
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
        // Calculate destination in strip buffer
        let (strip, stride) = match comp_idx {
            0 => {
                let x_offset = mcu_x * self.max_h_samp as usize * 8 + h * 8;
                let y_offset = v * 8 * self.strip_stride;
                (&mut self.y_strip[y_offset + x_offset..], self.strip_stride)
            }
            1 => {
                let x_offset = mcu_x * 8;
                (&mut self.cb_strip[x_offset..], self.chroma_strip_stride)
            }
            _ => {
                let x_offset = mcu_x * 8;
                (&mut self.cr_strip[x_offset..], self.chroma_strip_stride)
            }
        };

        if coeff_count <= 1 {
            let dc = coeffs[0] as i32 * quant[0] as i32;
            idct_int_dc_only(dc, strip, stride);
        } else {
            dequantize_unzigzag_i32_into_partial(coeffs, quant, &mut self.dequant_buf, coeff_count);
            match self.chroma_upsampling {
                ChromaUpsampling::LibjpegCompat => {
                    idct_int_tiered_libjpeg(&mut self.dequant_buf, strip, stride, coeff_count);
                }
                _ => {
                    idct_int_tiered(&mut self.dequant_buf, strip, stride, coeff_count);
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
        match self.subsampling {
            Subsampling::S444 => {} // No upsampling needed
            Subsampling::S422 => self.upsample_h2v1(),
            Subsampling::S420 => {
                self.upsample_h2v2();
                self.fixup_vertical_boundary();
                self.save_last_chroma_row();
            }
            Subsampling::S440 => {
                self.upsample_h1v2();
                self.fixup_vertical_boundary();
                self.save_last_chroma_row();
            }
        }
    }

    /// Get Y/Cb/Cr row slices for a given row within the current MCU row.
    ///
    /// Returns (y_row, cb_row, cr_row) slices of `cols` pixels each.
    /// For subsampled images, cb/cr come from the upsampled buffers.
    #[inline(always)]
    pub fn row_planes(&self, row_in_mcu: usize, cols: usize) -> (&[i16], &[i16], &[i16]) {
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
            ChromaUpsampling::Triangle => upsample_h2v1_i16_fancy_strided,
            ChromaUpsampling::LibjpegCompat => upsample_h2v1_i16_libjpeg_strided,
            ChromaUpsampling::NearestNeighbor => upsample_h2v1_i16_nearest_strided,
        };
        self.upsample_both_channels(upsample_fn);
    }

    /// Vertical 2x upsampling (4:4:0) with configurable filter.
    fn upsample_h1v2(&mut self) {
        type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
        let upsample_fn: StridedFn = match self.chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h1v2_i16_fancy_strided,
            ChromaUpsampling::LibjpegCompat => upsample_h1v2_i16_libjpeg_strided,
            ChromaUpsampling::NearestNeighbor => upsample_h1v2_i16_nearest_strided,
        };
        self.upsample_both_channels(upsample_fn);
    }

    /// Both horizontal and vertical 2x upsampling (4:2:0) with configurable filter.
    fn upsample_h2v2(&mut self) {
        type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
        let upsample_fn: StridedFn = match self.chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h2v2_i16_fancy_strided,
            ChromaUpsampling::LibjpegCompat => upsample_h2v2_i16_libjpeg_strided,
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
    fn fixup_h2v2_row0(&mut self, in_width: usize, out_width: usize, out_stride: usize) {
        // The current chroma row 0 data
        let cb_row0: [i16; 4096] = {
            let mut buf = [0i16; 4096];
            let w = in_width.min(4096);
            buf[..w].copy_from_slice(&self.cb_strip[..w]);
            buf
        };
        let cr_row0: [i16; 4096] = {
            let mut buf = [0i16; 4096];
            let w = in_width.min(4096);
            buf[..w].copy_from_slice(&self.cr_strip[..w]);
            buf
        };

        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                // Re-compute output row 0 with correct vertical neighbor
                let cb_out = &mut self.cb_upsampled[..out_width];
                upsample_row_h2_fancy_bilinear(
                    &cb_row0[..in_width],
                    &self.prev_cb_row[..in_width],
                    in_width,
                    cb_out,
                    true, // is_top_half
                );
                let cr_out = &mut self.cr_upsampled[..out_width];
                upsample_row_h2_fancy_bilinear(
                    &cr_row0[..in_width],
                    &self.prev_cr_row[..in_width],
                    in_width,
                    cr_out,
                    true,
                );
            }
            ChromaUpsampling::LibjpegCompat => {
                let cb_out = &mut self.cb_upsampled[..out_stride];
                upsample_h2v2_libjpeg_row(
                    &cb_row0[..in_width],
                    &self.prev_cb_row[..in_width],
                    cb_out,
                    in_width,
                    out_width,
                    true, // is_upper
                );
                let cr_out = &mut self.cr_upsampled[..out_stride];
                upsample_h2v2_libjpeg_row(
                    &cr_row0[..in_width],
                    &self.prev_cr_row[..in_width],
                    cr_out,
                    in_width,
                    out_width,
                    true,
                );
            }
            ChromaUpsampling::NearestNeighbor => {
                // Nearest neighbor doesn't interpolate vertically, no fixup needed
            }
        }
    }

    /// Fix h1v2 output row 0 using previous chroma context.
    fn fixup_h1v2_row0(&mut self, in_width: usize, out_width: usize, out_stride: usize) {
        let _ = out_stride;
        let w = in_width.min(out_width);

        match self.chroma_upsampling {
            ChromaUpsampling::Triangle => {
                // h1v2 fancy: (3 * curr + neighbor + 2) >> 2
                for x in 0..w {
                    let curr_cb = self.cb_strip[x] as i32;
                    let prev_cb = self.prev_cb_row[x] as i32;
                    self.cb_upsampled[x] = ((3 * curr_cb + prev_cb + 2) >> 2) as i16;

                    let curr_cr = self.cr_strip[x] as i32;
                    let prev_cr = self.prev_cr_row[x] as i32;
                    self.cr_upsampled[x] = ((3 * curr_cr + prev_cr + 2) >> 2) as i16;
                }
            }
            ChromaUpsampling::LibjpegCompat => {
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
                // No interpolation, no fixup needed
            }
        }
    }
}
