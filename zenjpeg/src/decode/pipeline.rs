//! Shared strip processing pipeline for JPEG decoding.
//!
//! `StripProcessor` handles stages 2-3 of the decode pipeline:
//! - IDCT + dequantization of coefficients into strip buffers
//! - Chroma upsampling to full resolution
//! - Row accessor for color conversion
//!
//! Both the scanline decoder and buffered decoder share this code path.

use super::config::{ChromaUpsampling, DctScale, OutputTarget, ShrinkQuality};
use super::idct_int::{
    idct_int_dc_only, idct_int_dc_only_unclamped, idct_int_tiered, idct_int_tiered_libjpeg,
    idct_int_tiered_libjpeg_unclamped, idct_int_tiered_unclamped,
};
use super::idct_scaled::{
    idct_scaled_1x1_from_dc, idct_scaled_1x1_from_dc_unclamped, idct_scaled_2x2,
    idct_scaled_2x2_unclamped, idct_scaled_4x4, idct_scaled_4x4_unclamped,
};
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
    pub output_target: OutputTarget,
    /// DCT scale for shrink-on-load (Full = default, no scaling).
    pub dct_scale: DctScale,
    /// Quality tier for shrink-on-load. Best = full IDCT + area average.
    pub shrink_quality: ShrinkQuality,
    /// Output pixels per block edge (1, 2, 4, or 8). Cached from dct_scale.
    pub block_size: usize,
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
            output_target: OutputTarget::default(),
            dct_scale: DctScale::Full,
            shrink_quality: ShrinkQuality::Fast,
            block_size: 8,
        }
    }

    /// Create a new strip processor with allocated buffers.
    pub fn new(
        width: u32,
        num_components: u8,
        h_samp: [u8; 3],
        v_samp: [u8; 3],
        chroma_upsampling: ChromaUpsampling,
        output_target: OutputTarget,
        dct_scale: DctScale,
        shrink_quality: ShrinkQuality,
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

        let block_size = dct_scale.block_output_size();

        // MCU column count is based on original dimensions and sampling,
        // independent of DCT scale (we produce fewer pixels per MCU, not fewer MCUs).
        let orig_mcu_width = max_h_samp as usize * 8;
        let mcu_cols = (width as usize + orig_mcu_width - 1) / orig_mcu_width;

        // Scaled MCU dimensions: each block produces block_size pixels instead of 8
        let mcu_width = max_h_samp as usize * block_size;
        let mcu_height = max_v_samp as usize * block_size;

        // Y strip: scaled resolution with SIMD-aligned stride
        let strip_width = mcu_cols * mcu_width;
        let strip_stride = align_up(strip_width, STRIP_ALIGNMENT);
        let y_strip_size = strip_stride * mcu_height;

        // Chroma strip: at native (potentially subsampled) scaled resolution
        let chroma_strip_width = if is_grayscale {
            0
        } else {
            mcu_cols * block_size
        };
        let chroma_strip_stride = if is_grayscale {
            0
        } else {
            align_up(chroma_strip_width, STRIP_ALIGNMENT)
        };
        let chroma_strip_height = if is_grayscale { 0 } else { block_size };
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
            output_target,
            dct_scale,
            shrink_quality,
            block_size,
        })
    }

    /// The number of MCU columns.
    #[inline]
    pub fn mcu_cols(&self) -> usize {
        // strip_width = mcu_cols * mcu_width, mcu_width = max_h_samp * block_size
        self.strip_width / (self.max_h_samp as usize * self.block_size)
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
        let bs = self.block_size;

        // Calculate destination in strip buffer (using block_size instead of 8)
        let (strip, stride) = match comp_idx {
            0 => {
                let x_offset = mcu_x * self.max_h_samp as usize * bs + h * bs;
                let y_offset = v * bs * self.strip_stride;
                (&mut self.y_strip[y_offset + x_offset..], self.strip_stride)
            }
            1 => {
                let x_offset = mcu_x * bs;
                (&mut self.cb_strip[x_offset..], self.chroma_strip_stride)
            }
            _ => {
                let x_offset = mcu_x * bs;
                (&mut self.cr_strip[x_offset..], self.chroma_strip_stride)
            }
        };

        let unclamped = self.output_target.needs_unclamped_idct();

        // ShrinkQuality::Best uses full 8×8 IDCT + area average for all reduced scales.
        // This produces much higher quality than reduced IDCT because it uses all 64
        // DCT coefficients. For aligned 2:1/4:1/8:1 ratios, the averaging regions
        // are fully contained within each 8×8 block, so per-block processing gives
        // identical results to full-image area average.
        if self.shrink_quality == ShrinkQuality::Best && self.dct_scale != DctScale::Full {
            return Self::idct_block_best_quality(
                &mut self.dequant_buf,
                self.chroma_upsampling,
                self.block_size,
                strip,
                stride,
                coeffs,
                coeff_count,
                quant,
                unclamped,
            );
        }

        // Dispatch based on DCT scale
        match self.dct_scale {
            DctScale::Full => {
                // Standard full 8x8 IDCT path (unchanged)
                if coeff_count <= 1 {
                    let dc = coeffs[0] as i32 * quant[0] as i32;
                    if unclamped {
                        idct_int_dc_only_unclamped(dc, strip, stride);
                    } else {
                        idct_int_dc_only(dc, strip, stride);
                    }
                } else {
                    dequantize_unzigzag_i32_into_partial(
                        coeffs,
                        quant,
                        &mut self.dequant_buf,
                        coeff_count,
                    );
                    match (unclamped, self.chroma_upsampling) {
                        (false, ChromaUpsampling::LibjpegCompat) => {
                            idct_int_tiered_libjpeg(
                                &mut self.dequant_buf,
                                strip,
                                stride,
                                coeff_count,
                            );
                        }
                        (false, _) => {
                            idct_int_tiered(&mut self.dequant_buf, strip, stride, coeff_count);
                        }
                        (true, ChromaUpsampling::LibjpegCompat) => {
                            idct_int_tiered_libjpeg_unclamped(
                                &mut self.dequant_buf,
                                strip,
                                stride,
                                coeff_count,
                            );
                        }
                        (true, _) => {
                            idct_int_tiered_unclamped(
                                &mut self.dequant_buf,
                                strip,
                                stride,
                                coeff_count,
                            );
                        }
                    }
                }
            }
            DctScale::Eighth => {
                // 1x1: DC-only, single pixel per block
                let dc = coeffs[0] as i32 * quant[0] as i32;
                if unclamped {
                    idct_scaled_1x1_from_dc_unclamped(dc, strip, stride);
                } else {
                    idct_scaled_1x1_from_dc(dc, strip, stride);
                }
            }
            DctScale::Quarter => {
                // 2x2: uses top-left 2x2 coefficients
                if coeff_count <= 1 {
                    // DC-only: all 4 pixels the same, use 1x1 math replicated
                    let dc = coeffs[0] as i32 * quant[0] as i32;
                    let val = if unclamped {
                        (dc.wrapping_add(4).wrapping_add(1024)).wrapping_shr(3) as i16
                    } else {
                        (dc.wrapping_add(4).wrapping_add(1024))
                            .wrapping_shr(3)
                            .clamp(0, 255) as i16
                    };
                    strip[0] = val;
                    strip[1] = val;
                    strip[stride] = val;
                    strip[stride + 1] = val;
                } else {
                    dequantize_unzigzag_i32_into_partial(
                        coeffs,
                        quant,
                        &mut self.dequant_buf,
                        coeff_count,
                    );
                    if unclamped {
                        idct_scaled_2x2_unclamped(&self.dequant_buf, strip, stride);
                    } else {
                        idct_scaled_2x2(&self.dequant_buf, strip, stride);
                    }
                }
            }
            DctScale::Half => {
                // 4x4: uses top-left 4x4 coefficients
                if coeff_count <= 1 {
                    // DC-only: all 16 pixels the same
                    let dc = coeffs[0] as i32 * quant[0] as i32;
                    let val = if unclamped {
                        (dc.wrapping_add(4).wrapping_add(1024)).wrapping_shr(3) as i16
                    } else {
                        (dc.wrapping_add(4).wrapping_add(1024))
                            .wrapping_shr(3)
                            .clamp(0, 255) as i16
                    };
                    for row in 0..4 {
                        let r = &mut strip[row * stride..];
                        r[..4].fill(val);
                    }
                } else {
                    dequantize_unzigzag_i32_into_partial(
                        coeffs,
                        quant,
                        &mut self.dequant_buf,
                        coeff_count,
                    );
                    if unclamped {
                        idct_scaled_4x4_unclamped(&self.dequant_buf, strip, stride);
                    } else {
                        idct_scaled_4x4(&self.dequant_buf, strip, stride);
                    }
                }
            }
        }
    }

    /// Full 8×8 IDCT + area average for ShrinkQuality::Best.
    ///
    /// Produces scaled output (4×4, 2×2, or 1×1) by first doing a full-resolution
    /// IDCT into a temporary buffer, then area-averaging to the target block size.
    /// Uses all 64 DCT coefficients, giving much better quality than reduced IDCT.
    ///
    /// Takes `dequant_buf` by separate reference to avoid borrowing all of `self`
    /// (since the caller already borrows strip buffers from `self`).
    #[inline(always)]
    fn idct_block_best_quality(
        dequant_buf: &mut [i32; DCT_BLOCK_SIZE],
        chroma_upsampling: ChromaUpsampling,
        block_size: usize,
        strip: &mut [i16],
        stride: usize,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        coeff_count: u8,
        quant: &[u16; DCT_BLOCK_SIZE],
        unclamped: bool,
    ) {
        let bs = block_size;

        if coeff_count <= 1 {
            // DC-only: area average of a uniform block is the same value.
            // This is identical for both reduced IDCT and full IDCT + average.
            let dc = coeffs[0] as i32 * quant[0] as i32;
            let val = if unclamped {
                (dc.wrapping_add(4).wrapping_add(1024)).wrapping_shr(3) as i16
            } else {
                (dc.wrapping_add(4).wrapping_add(1024))
                    .wrapping_shr(3)
                    .clamp(0, 255) as i16
            };
            for row in 0..bs {
                let r = &mut strip[row * stride..];
                r[..bs].fill(val);
            }
            return;
        }

        // Full 8×8 IDCT into temporary stack buffer
        let mut temp = [0i16; 64];
        dequantize_unzigzag_i32_into_partial(
            coeffs,
            quant,
            dequant_buf,
            coeff_count,
        );

        if unclamped {
            match chroma_upsampling {
                ChromaUpsampling::LibjpegCompat => {
                    idct_int_tiered_libjpeg_unclamped(
                        dequant_buf, &mut temp, 8, coeff_count,
                    );
                }
                _ => {
                    idct_int_tiered_unclamped(
                        dequant_buf, &mut temp, 8, coeff_count,
                    );
                }
            }
        } else {
            match chroma_upsampling {
                ChromaUpsampling::LibjpegCompat => {
                    idct_int_tiered_libjpeg(
                        dequant_buf, &mut temp, 8, coeff_count,
                    );
                }
                _ => {
                    idct_int_tiered(
                        dequant_buf, &mut temp, 8, coeff_count,
                    );
                }
            }
        }

        // Area average from 8×8 to bs×bs
        let factor = 8 / bs; // 2 for half, 4 for quarter, 8 for eighth
        let factor_sq = (factor * factor) as i32;
        let round = factor_sq / 2;

        for oy in 0..bs {
            for ox in 0..bs {
                let mut sum = 0i32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        sum += temp[(oy * factor + dy) * 8 + (ox * factor + dx)] as i32;
                    }
                }
                strip[oy * stride + ox] = ((sum + round) / factor_sq) as i16;
            }
        }
    }

    /// Whether this subsampling mode needs vertical chroma upsampling.
    #[inline]
    pub fn needs_vertical_upsample(&self) -> bool {
        matches!(self.subsampling, Subsampling::S420 | Subsampling::S440)
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
                // Re-compute last output row with correct vertical neighbor
                let cb_out = &mut self.cb_upsampled[last_out_offset..last_out_offset + out_width];
                upsample_row_h2_fancy_bilinear(
                    &self.cb_strip[last_chroma_offset..last_chroma_offset + in_width],
                    &self.next_cb_row[..in_width],
                    in_width,
                    cb_out,
                    false, // is_top_half = false → bottom half
                );
                let cr_out = &mut self.cr_upsampled[last_out_offset..last_out_offset + out_width];
                upsample_row_h2_fancy_bilinear(
                    &self.cr_strip[last_chroma_offset..last_chroma_offset + in_width],
                    &self.next_cr_row[..in_width],
                    in_width,
                    cr_out,
                    false,
                );
            }
            ChromaUpsampling::LibjpegCompat => {
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
                // No interpolation, no fixup needed
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
                // h1v2 fancy: (3 * curr + neighbor + 2) >> 2
                for x in 0..w {
                    let curr_cb = self.cb_strip[last_chroma_offset + x] as i32;
                    let next_cb = self.next_cb_row[x] as i32;
                    self.cb_upsampled[last_out_offset + x] =
                        ((3 * curr_cb + next_cb + 2) >> 2) as i16;

                    let curr_cr = self.cr_strip[last_chroma_offset + x] as i32;
                    let next_cr = self.next_cr_row[x] as i32;
                    self.cr_upsampled[last_out_offset + x] =
                        ((3 * curr_cr + next_cr + 2) >> 2) as i16;
                }
            }
            ChromaUpsampling::LibjpegCompat => {
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
                // No interpolation, no fixup needed
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
                // prev_cb_row = last chroma row of previous MCU (the one we're fixing)
                // cb_strip[0..] = first chroma row of next MCU (just decoded)
                let cb_out = &mut self.deferred_cb_row[..out_width];
                upsample_row_h2_fancy_bilinear(
                    &self.prev_cb_row[..in_width],
                    &self.cb_strip[..in_width],
                    in_width,
                    cb_out,
                    false, // bottom half
                );
                let cr_out = &mut self.deferred_cr_row[..out_width];
                upsample_row_h2_fancy_bilinear(
                    &self.prev_cr_row[..in_width],
                    &self.cr_strip[..in_width],
                    in_width,
                    cr_out,
                    false,
                );
            }
            ChromaUpsampling::LibjpegCompat => {
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
                // No interpolation needed, just copy from upsampled
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
                    let curr_cb = self.prev_cb_row[x] as i32;
                    let next_cb = self.cb_strip[x] as i32;
                    self.deferred_cb_row[x] = ((3 * curr_cb + next_cb + 2) >> 2) as i16;

                    let curr_cr = self.prev_cr_row[x] as i32;
                    let next_cr = self.cr_strip[x] as i32;
                    self.deferred_cr_row[x] = ((3 * curr_cr + next_cr + 2) >> 2) as i16;
                }
            }
            ChromaUpsampling::LibjpegCompat => {
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
                // Re-compute output row 0 with correct vertical neighbor
                let cb_out = &mut self.cb_upsampled[..out_width];
                upsample_row_h2_fancy_bilinear(
                    &self.cb_strip[..in_width],
                    &self.prev_cb_row[..in_width],
                    in_width,
                    cb_out,
                    true, // is_top_half
                );
                let cr_out = &mut self.cr_upsampled[..out_width];
                upsample_row_h2_fancy_bilinear(
                    &self.cr_strip[..in_width],
                    &self.prev_cr_row[..in_width],
                    in_width,
                    cr_out,
                    true,
                );
            }
            ChromaUpsampling::LibjpegCompat => {
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
