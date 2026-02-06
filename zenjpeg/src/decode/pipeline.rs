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
    upsample_h2v2_i16_fancy_strided, upsample_h2v2_i16_libjpeg_strided,
    upsample_h2v2_i16_nearest_strided,
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
        let (cb_upsampled, cr_upsampled) = if !is_grayscale && subsampling != Subsampling::S444 {
            let upsampled_size = strip_stride * mcu_height;
            (
                try_alloc_maybeuninit(upsampled_size, "Cb upsampled buffer")?,
                try_alloc_maybeuninit(upsampled_size, "Cr upsampled buffer")?,
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
    pub fn upsample_chroma(&mut self) {
        match self.subsampling {
            Subsampling::S444 => {} // No upsampling needed
            Subsampling::S422 => self.upsample_h2v1(),
            Subsampling::S420 => self.upsample_h2v2(),
            Subsampling::S440 => self.upsample_h1v2(),
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

    /// Horizontal 2x upsampling (4:2:2) with configurable filter.
    fn upsample_h2v1(&mut self) {
        let in_width = self.chroma_strip_width;
        let in_stride = self.chroma_strip_stride;
        let out_width = self.strip_width;
        let out_stride = self.strip_stride;
        let height = self.mcu_height;

        for y in 0..height {
            let in_row = y.min(self.chroma_strip_height - 1);

            match self.chroma_upsampling {
                ChromaUpsampling::NearestNeighbor => {
                    for out_x in 0..out_width {
                        let in_x = (out_x / 2).min(in_width - 1);
                        let in_idx = in_row * in_stride + in_x;
                        let out_idx = y * out_stride + out_x;
                        self.cb_upsampled[out_idx] = self.cb_strip[in_idx];
                        self.cr_upsampled[out_idx] = self.cr_strip[in_idx];
                    }
                }
                ChromaUpsampling::Triangle => {
                    for out_x in 0..out_width {
                        let in_x = out_x / 2;
                        let in_idx = in_row * in_stride + in_x.min(in_width - 1);
                        let cb_curr = self.cb_strip[in_idx] as i32;
                        let cr_curr = self.cr_strip[in_idx] as i32;

                        let (cb_val, cr_val) = if out_x % 2 == 0 {
                            let left_idx = in_row * in_stride + in_x.saturating_sub(1);
                            let cb_left = self.cb_strip[left_idx] as i32;
                            let cr_left = self.cr_strip[left_idx] as i32;
                            (
                                ((3 * cb_curr + cb_left + 2) >> 2) as i16,
                                ((3 * cr_curr + cr_left + 2) >> 2) as i16,
                            )
                        } else {
                            let right_idx = in_row * in_stride + (in_x + 1).min(in_width - 1);
                            let cb_right = self.cb_strip[right_idx] as i32;
                            let cr_right = self.cr_strip[right_idx] as i32;
                            (
                                ((3 * cb_curr + cb_right + 2) >> 2) as i16,
                                ((3 * cr_curr + cr_right + 2) >> 2) as i16,
                            )
                        };

                        let out_idx = y * out_stride + out_x;
                        self.cb_upsampled[out_idx] = cb_val;
                        self.cr_upsampled[out_idx] = cr_val;
                    }
                }
                ChromaUpsampling::LibjpegCompat => {
                    // libjpeg-turbo h2v1: alternating +1/+2 bias
                    if in_width == 0 {
                        continue;
                    }
                    let cb_in = &self.cb_strip[in_row * in_stride..];
                    let cr_in = &self.cr_strip[in_row * in_stride..];
                    let out_base = y * out_stride;

                    if in_width == 1 {
                        self.cb_upsampled[out_base] = cb_in[0];
                        self.cr_upsampled[out_base] = cr_in[0];
                        if out_width > 1 {
                            self.cb_upsampled[out_base + 1] = cb_in[0];
                            self.cr_upsampled[out_base + 1] = cr_in[0];
                        }
                        continue;
                    }

                    // First column
                    let cb0 = cb_in[0] as i32;
                    let cr0 = cr_in[0] as i32;
                    self.cb_upsampled[out_base] = cb0 as i16;
                    self.cr_upsampled[out_base] = cr0 as i16;
                    if out_width > 1 {
                        self.cb_upsampled[out_base + 1] =
                            ((cb0 * 3 + cb_in[1] as i32 + 2) >> 2) as i16;
                        self.cr_upsampled[out_base + 1] =
                            ((cr0 * 3 + cr_in[1] as i32 + 2) >> 2) as i16;
                    }

                    // Interior columns
                    for in_x in 1..in_width.saturating_sub(1) {
                        let cb_prev = cb_in[in_x - 1] as i32;
                        let cb_curr = cb_in[in_x] as i32;
                        let cb_next = cb_in[in_x + 1] as i32;
                        let cr_prev = cr_in[in_x - 1] as i32;
                        let cr_curr = cr_in[in_x] as i32;
                        let cr_next = cr_in[in_x + 1] as i32;
                        let left = out_base + in_x * 2;
                        let right = left + 1;
                        if left < out_base + out_width {
                            self.cb_upsampled[left] = ((cb_curr * 3 + cb_prev + 1) >> 2) as i16;
                            self.cr_upsampled[left] = ((cr_curr * 3 + cr_prev + 1) >> 2) as i16;
                        }
                        if right < out_base + out_width {
                            self.cb_upsampled[right] = ((cb_curr * 3 + cb_next + 2) >> 2) as i16;
                            self.cr_upsampled[right] = ((cr_curr * 3 + cr_next + 2) >> 2) as i16;
                        }
                    }

                    // Last column
                    let last = in_width - 1;
                    let cb_prev = cb_in[last - 1] as i32;
                    let cb_curr = cb_in[last] as i32;
                    let cr_prev = cr_in[last - 1] as i32;
                    let cr_curr = cr_in[last] as i32;
                    let left = out_base + last * 2;
                    let right = left + 1;
                    if left < out_base + out_width {
                        self.cb_upsampled[left] = ((cb_curr * 3 + cb_prev + 1) >> 2) as i16;
                        self.cr_upsampled[left] = ((cr_curr * 3 + cr_prev + 1) >> 2) as i16;
                    }
                    if right < out_base + out_width {
                        self.cb_upsampled[right] = cb_curr as i16;
                        self.cr_upsampled[right] = cr_curr as i16;
                    }
                }
            }
        }
    }

    /// Vertical 2x upsampling (4:4:0) with configurable filter.
    fn upsample_h1v2(&mut self) {
        let in_width = self.chroma_strip_width;
        let in_stride = self.chroma_strip_stride;
        let in_height = self.chroma_strip_height;
        let out_width = self.strip_width;
        let out_stride = self.strip_stride;
        let out_height = self.mcu_height;

        for out_y in 0..out_height {
            let in_y = out_y / 2;
            let in_y_clamped = in_y.min(in_height.saturating_sub(1));
            let is_upper = out_y % 2 == 0;

            let far_y = if is_upper {
                in_y_clamped.saturating_sub(1)
            } else {
                (in_y + 1).min(in_height.saturating_sub(1))
            };

            match self.chroma_upsampling {
                ChromaUpsampling::NearestNeighbor => {
                    for out_x in 0..out_width {
                        let in_x = out_x.min(in_width.saturating_sub(1));
                        let in_idx = in_y_clamped * in_stride + in_x;
                        let out_idx = out_y * out_stride + out_x;
                        self.cb_upsampled[out_idx] = self.cb_strip[in_idx];
                        self.cr_upsampled[out_idx] = self.cr_strip[in_idx];
                    }
                }
                ChromaUpsampling::Triangle => {
                    for out_x in 0..out_width {
                        let in_x = out_x.min(in_width.saturating_sub(1));
                        let curr_idx = in_y_clamped * in_stride + in_x;
                        let neighbor_idx = far_y * in_stride + in_x;
                        let cb_curr = self.cb_strip[curr_idx] as i32;
                        let cr_curr = self.cr_strip[curr_idx] as i32;
                        let cb_neighbor = self.cb_strip[neighbor_idx] as i32;
                        let cr_neighbor = self.cr_strip[neighbor_idx] as i32;

                        let out_idx = out_y * out_stride + out_x;
                        self.cb_upsampled[out_idx] = ((3 * cb_curr + cb_neighbor + 2) >> 2) as i16;
                        self.cr_upsampled[out_idx] = ((3 * cr_curr + cr_neighbor + 2) >> 2) as i16;
                    }
                }
                ChromaUpsampling::LibjpegCompat => {
                    let bias = if is_upper { 1i32 } else { 2i32 };
                    for out_x in 0..out_width {
                        let in_x = out_x.min(in_width.saturating_sub(1));
                        let near_idx = in_y_clamped * in_stride + in_x;
                        let far_idx = far_y * in_stride + in_x;
                        let cb_near = self.cb_strip[near_idx] as i32;
                        let cr_near = self.cr_strip[near_idx] as i32;
                        let cb_far = self.cb_strip[far_idx] as i32;
                        let cr_far = self.cr_strip[far_idx] as i32;

                        let out_idx = out_y * out_stride + out_x;
                        self.cb_upsampled[out_idx] = ((cb_near * 3 + cb_far + bias) >> 2) as i16;
                        self.cr_upsampled[out_idx] = ((cr_near * 3 + cr_far + bias) >> 2) as i16;
                    }
                }
            }
        }
    }

    /// Both horizontal and vertical 2x upsampling (4:2:0) with configurable filter.
    fn upsample_h2v2(&mut self) {
        let in_width = self.chroma_strip_width;
        let in_stride = self.chroma_strip_stride;
        let in_height = self.chroma_strip_height;
        let out_width = self.strip_width;
        let out_stride = self.strip_stride;
        let out_height = self.mcu_height;

        type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
        let upsample_fn: StridedFn = match self.chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h2v2_i16_fancy_strided,
            ChromaUpsampling::LibjpegCompat => upsample_h2v2_i16_libjpeg_strided,
            ChromaUpsampling::NearestNeighbor => upsample_h2v2_i16_nearest_strided,
        };

        upsample_fn(
            &self.cb_strip,
            in_width,
            in_stride,
            in_height,
            &mut self.cb_upsampled,
            out_width,
            out_stride,
            out_height,
        );
        upsample_fn(
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
}
