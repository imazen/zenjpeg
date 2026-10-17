//! The fast integer coefficient path: integer IDCT, integer upsampling and a
//! [`RowKernel`] from stored coefficients straight into the output layout.
//!
//! Serves buffered decodes (progressive, arithmetic, and baseline files that
//! did not stream) whose model and shape [`PathPlan::fast_integer`] accepts:
//! 4:4:4 (any shared sampling) and the 4:2:2 / 4:2:0 / 4:4:0 layouts, for
//! YCbCr and RGB.
//!
//! One routine renders a contiguous range of MCU rows ([`JpegParser::fast_integer_rows`]).
//! The sequential decode is one range; with `parallel`, large images split
//! into one range per thread. Each range builds its own chroma context (the
//! row above its first MCU row, the row below its last), so every split
//! produces the same bytes as the sequential decode by construction — the
//! previous separate parallel twin had drifted (it skipped the right-edge
//! chroma replication and decoded the last column of even, non-MCU-aligned
//! widths differently).
//!
//! [`PathPlan::fast_integer`]: crate::decode::color_model::PathPlan::fast_integer

use super::super::ChromaUpsampling;
use super::super::IdctMethod;
use super::super::color_model::ChromaLayout;
use super::super::idct_int::{idct_int_tiered, idct_int_tiered_libjpeg};
use super::output_helpers::{edge_replicate_h_padding, idct_chroma_into_ext, idct_comp_mcu_row};
use super::{CompInfo, JpegParser};
use crate::color::planes::{Layout8, RowKernel};
use crate::decode::upsample::{
    upsample_h1v2_i16_libjpeg, upsample_h1v2_i16_nearest, upsample_h2v1_i16_libjpeg,
    upsample_h2v1_i16_nearest, upsample_h2v2_i16_libjpeg, upsample_h2v2_i16_libjpeg_turbo,
};
use crate::error::{Error, Result};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit_pref};
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::types::PixelFormat;

type IdctFn = fn(&mut [i32; 64], &mut [i16], usize, u8);
type UpsampleFn = fn(&[i16], usize, usize, &mut [i16], usize, usize);

/// Minimum MCU rows to justify splitting across threads.
#[cfg(feature = "parallel")]
const MIN_MCU_ROWS_PARALLEL: usize = 8;

/// Minimum pixel count (width × height) to justify splitting across threads:
/// below ~2 MP the rayon overhead exceeds the gain.
#[cfg(feature = "parallel")]
const MIN_PIXELS_PARALLEL: usize = 2_000_000;

/// Everything the per-range routine needs, computed once per decode.
pub(super) struct FastIntegerPlan<'p> {
    kernel: RowKernel,
    chroma: ChromaLayout,
    width: usize,
    height: usize,
    mcu_height: usize,
    mcu_rows: usize,
    comp_infos: Vec<CompInfo>,
    quant: [&'p [u16; DCT_BLOCK_SIZE]; 3],
    idct_fn: IdctFn,
    /// Chroma upsampler for the subsampled layouts; `None` for 4:4:4 and for
    /// nearest-neighbour 4:2:0, which converts with [`RowKernel::row_h2`]
    /// straight from the half-width chroma rows.
    upsample: Option<UpsampleFn>,
    /// Triangle 4:2:0 into packed RGB or BGR from YCbCr on x86-64-v3:
    /// upsample and convert in one fused row kernel
    /// (`baseline_streaming::output_fancy_strip_fused`), never materializing
    /// the upsampled chroma planes; BGR swaps R and B per fused row
    /// ([`RowKernel::fused_ycbcr`]). Same math as the upsample-then-convert
    /// path.
    #[cfg(target_arch = "x86_64")]
    fused_h2v2: Option<archmage::X64V3Token>,
}

impl FastIntegerPlan<'_> {
    pub(super) fn bytes_per_pixel(&self) -> usize {
        self.kernel.layout().bytes_per_pixel()
    }

    /// Output bytes, overflow-checked (the frame dimensions are untrusted).
    pub(super) fn output_len(&self) -> Result<usize> {
        checked_size_2d(
            checked_size_2d(self.width, self.height)?,
            self.bytes_per_pixel(),
        )
    }
}

impl<'a> JpegParser<'a> {
    /// The fast integer path for `format`, if this frame's model and shape
    /// have one (RGB-family formats only; dequantization bias and the forced
    /// f32 IDCT stay on the f32 path).
    pub(super) fn fast_integer_plan(
        &self,
        format: PixelFormat,
        chroma_upsampling: ChromaUpsampling,
    ) -> Result<Option<FastIntegerPlan<'_>>> {
        if self.force_f32_idct || self.coeffs.len() < 3 {
            return Ok(None);
        }
        let Some(layout) = Layout8::from_pixel_format(format).filter(|l| *l != Layout8::Gray)
        else {
            return Ok(None);
        };
        let Some(chroma) = self.plan().fast_integer() else {
            return Ok(None);
        };
        let Some(kernel) = self.color_model.row_kernel(layout, self.idct_method) else {
            return Ok(None);
        };

        let width = self.width as usize;
        let height = self.height as usize;
        // Component 0 carries the maximum sampling on every fast-integer shape.
        let y_h = self.components[0].h_samp_factor as usize;
        let y_v = self.components[0].v_samp_factor as usize;
        let mcu_height = y_v * 8;
        let mcu_cols = width.div_ceil(y_h * 8);
        let mcu_rows = height.div_ceil(mcu_height);
        let comp_infos = self.build_comp_infos(mcu_cols, mcu_rows, y_h, y_v, 3)?;
        let quant = |i: usize| {
            self.quant_tables[comp_infos[i].quant_idx]
                .as_ref()
                .ok_or_else(|| Error::internal("missing quantization table"))
        };
        let quant = [quant(0)?, quant(1)?, quant(2)?];
        let idct_fn: IdctFn = match self.idct_method {
            IdctMethod::Libjpeg => idct_int_tiered_libjpeg,
            IdctMethod::Jpegli => idct_int_tiered,
        };
        let triangle = chroma_upsampling == ChromaUpsampling::Triangle;
        let upsample: Option<UpsampleFn> = match chroma {
            ChromaLayout::H2V2 if triangle => Some(match self.idct_method {
                // IdctMethod::Libjpeg => turbo's fixed h2v2 rounding bias.
                IdctMethod::Libjpeg => upsample_h2v2_i16_libjpeg_turbo,
                IdctMethod::Jpegli => upsample_h2v2_i16_libjpeg,
            }),
            ChromaLayout::H2V2 => None,
            ChromaLayout::H2V1 if triangle => Some(upsample_h2v1_i16_libjpeg),
            ChromaLayout::H2V1 => Some(upsample_h2v1_i16_nearest),
            ChromaLayout::H1V2 if triangle => Some(upsample_h1v2_i16_libjpeg),
            ChromaLayout::H1V2 => Some(upsample_h1v2_i16_nearest),
            _ => None,
        };
        #[cfg(target_arch = "x86_64")]
        let fused_h2v2 =
            if chroma == ChromaLayout::H2V2 && triangle && kernel.fused_ycbcr().is_some() {
                <archmage::X64V3Token as archmage::SimdToken>::summon()
            } else {
                None
            };
        Ok(Some(FastIntegerPlan {
            #[cfg(target_arch = "x86_64")]
            fused_h2v2,
            kernel,
            chroma,
            width,
            height,
            mcu_height,
            mcu_rows,
            comp_infos,
            quant,
            idct_fn,
            upsample,
        }))
    }

    /// Render the whole frame into `out` (`plan.output_len()` bytes, rows
    /// packed). Splits across threads when `parallel` is on, threads are
    /// allowed and the image is large enough.
    pub(super) fn fast_integer_into(
        &self,
        plan: &FastIntegerPlan<'_>,
        out: &mut [u8],
    ) -> Result<()> {
        let out = out
            .get_mut(..plan.output_len()?)
            .ok_or_else(|| Error::internal("output too small for the fast integer path"))?;
        #[cfg(feature = "parallel")]
        if self.num_threads != 1
            && plan.mcu_rows >= MIN_MCU_ROWS_PARALLEL
            && plan.width * plan.height >= MIN_PIXELS_PARALLEL
        {
            use rayon::prelude::*;
            let batch = plan.mcu_rows.div_ceil(rayon::current_num_threads());
            let batch_bytes = batch * plan.mcu_height * plan.width * plan.bytes_per_pixel();
            return out
                .par_chunks_mut(batch_bytes)
                .enumerate()
                .try_for_each(|(i, chunk)| {
                    let start = i * batch;
                    self.fast_integer_rows(plan, start, (start + batch).min(plan.mcu_rows), chunk)
                });
        }
        self.fast_integer_rows(plan, 0, plan.mcu_rows, out)
    }

    /// Render MCU rows `start..end` into `out`, whose first byte is pixel row
    /// `start * mcu_height`.
    fn fast_integer_rows(
        &self,
        plan: &FastIntegerPlan<'_>,
        start: usize,
        end: usize,
        out: &mut [u8],
    ) -> Result<()> {
        if plan.chroma == ChromaLayout::S444 {
            self.fast_444_rows(plan, start, end, out)
        } else {
            self.fast_subsampled_rows(plan, start, end, out)
        }
    }

    fn fast_444_rows(
        &self,
        p: &FastIntegerPlan<'_>,
        start: usize,
        end: usize,
        out: &mut [u8],
    ) -> Result<()> {
        let strip_width = p.comp_infos[0].comp_width;
        let strip_size = strip_width * p.mcu_height;
        let alloc =
            |what| try_alloc_maybeuninit_pref::<i16>(self.alloc_pref, false, strip_size, what);
        let mut strips = [
            alloc("Y strip buffer")?,
            alloc("Cb strip buffer")?,
            alloc("Cr strip buffer")?,
        ];
        let mut dequant = [0i32; DCT_BLOCK_SIZE];
        let row_bytes = p.width * p.bytes_per_pixel();
        let cols = p.width.min(strip_width);

        for imcu_row in start..end {
            for (c, strip) in strips.iter_mut().enumerate() {
                idct_comp_mcu_row(
                    &self.coeffs[c],
                    &self.coeff_counts[c],
                    &p.comp_infos[c],
                    p.quant[c],
                    imcu_row,
                    strip,
                    strip_width,
                    p.idct_fn,
                    &mut dequant,
                );
            }
            let y0 = imcu_row * p.mcu_height;
            let rows = p.mcu_height.min(p.height.saturating_sub(y0));
            for row in 0..rows {
                let s = row * strip_width;
                let o = (y0 - start * p.mcu_height + row) * row_bytes;
                p.kernel.row(
                    &strips[0][s..s + cols],
                    &strips[1][s..],
                    &strips[2][s..],
                    &mut out[o..o + row_bytes],
                );
            }
        }
        Ok(())
    }

    /// Subsampled layouts. Chroma is kept in double-buffered extended strips
    /// of `c_strip_height + 2` rows: row 0 is the last chroma row of the MCU
    /// row above (context), rows `1..=c_strip_height` this MCU row's IDCT
    /// output, and the last row the first chroma row of the MCU row below. The
    /// upsamplers' edge handling reads the context rows, so interpolation
    /// across MCU-row boundaries needs no special cases.
    ///
    /// Invariant at the top of each MCU row `r`: `ext_a` holds strip `r` (with
    /// its above context) and `ext_b` holds strip `r + 1` when it exists.
    fn fast_subsampled_rows(
        &self,
        p: &FastIntegerPlan<'_>,
        start: usize,
        end: usize,
        out: &mut [u8],
    ) -> Result<()> {
        let (h_ratio, v_ratio) = self.shape.chroma_ratios();
        let [info_y, info_cb, info_cr] = [&p.comp_infos[0], &p.comp_infos[1], &p.comp_infos[2]];
        let y_strip_width = info_y.comp_width;
        let c_strip_width = info_cb.comp_width;
        let c_strip_height = info_cb.v_samp * 8;
        let ext_height = c_strip_height + 2;
        let ext_size = ext_height * c_strip_width;
        let chroma_height_total = p.height.div_ceil(v_ratio);
        let downsampled_w = p.width.div_ceil(h_ratio);
        let cols = p.width.min(y_strip_width);
        let row_bytes = p.width * p.bytes_per_pixel();
        let pref = self.alloc_pref;

        // Per-range scratch, bounded by the row width × strip height.
        let alloc = |n, what| try_alloc_maybeuninit_pref::<i16>(pref, false, n, what);
        let mut y_strip = alloc(y_strip_width * p.mcu_height, "Y strip buffer")?;
        let mut cb_a = alloc(ext_size, "Cb ext_a buffer")?;
        let mut cb_b = alloc(ext_size, "Cb ext_b buffer")?;
        let mut cr_a = alloc(ext_size, "Cr ext_a buffer")?;
        let mut cr_b = alloc(ext_size, "Cr ext_b buffer")?;
        #[cfg(target_arch = "x86_64")]
        let fused = p.fused_h2v2.is_some();
        #[cfg(not(target_arch = "x86_64"))]
        let fused = false;
        let up_size = if p.upsample.is_some() && !fused {
            ext_height * v_ratio * y_strip_width
        } else {
            0
        };
        let mut cb_up = alloc(up_size, "Cb upsample buffer")?;
        let mut cr_up = alloc(up_size, "Cr upsample buffer")?;
        // Column-sum scratch for the fused h2v2 path (padded: width + 2 taps).
        #[cfg(target_arch = "x86_64")]
        let (mut cs_cb, mut cs_cr) = if p.fused_h2v2.is_some() {
            (
                alloc(c_strip_width + 2, "Cb colsum")?,
                alloc(c_strip_width + 2, "Cr colsum")?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let mut dequant = [0i32; DCT_BLOCK_SIZE];
        let mut chroma_dequant = [0i32; DCT_BLOCK_SIZE];

        let mut idct_chroma = |cb: &mut [i16], cr: &mut [i16], imcu_row: usize| {
            for (ext, c, info) in [(cb, 1, info_cb), (cr, 2, info_cr)] {
                idct_chroma_into_ext(
                    ext,
                    &self.coeffs[c],
                    &self.coeff_counts[c],
                    info,
                    p.quant[c],
                    imcu_row,
                    c_strip_width,
                    c_strip_height,
                    chroma_height_total,
                    p.idct_fn,
                    &mut chroma_dequant,
                );
            }
        };
        let last_data = c_strip_height * c_strip_width;
        let below = (c_strip_height + 1) * c_strip_width;
        let row1 = c_strip_width..2 * c_strip_width;

        idct_chroma(&mut cb_a, &mut cr_a, start);
        if start == 0 {
            // Top of the image: replicate the first chroma row upward.
            cb_a.copy_within(row1.clone(), 0);
            cr_a.copy_within(row1.clone(), 0);
        } else {
            // Above context: the last chroma row of the MCU row above.
            idct_chroma(&mut cb_b, &mut cr_b, start - 1);
            cb_a[..c_strip_width].copy_from_slice(&cb_b[last_data..last_data + c_strip_width]);
            cr_a[..c_strip_width].copy_from_slice(&cr_b[last_data..last_data + c_strip_width]);
        }
        if start + 1 < p.mcu_rows {
            idct_chroma(&mut cb_b, &mut cr_b, start + 1);
        }

        for imcu_row in start..end {
            // Below context: the first chroma row of the next MCU row, or (at
            // the bottom) the last real chroma row. `idct_chroma_into_ext`
            // already replicated that row over the padding rows, so the last
            // data row is the last real one.
            if imcu_row + 1 < p.mcu_rows {
                cb_a[below..below + c_strip_width].copy_from_slice(&cb_b[row1.clone()]);
                cr_a[below..below + c_strip_width].copy_from_slice(&cr_b[row1.clone()]);
            } else {
                cb_a.copy_within(last_data..last_data + c_strip_width, below);
                cr_a.copy_within(last_data..last_data + c_strip_width, below);
            }

            idct_comp_mcu_row(
                &self.coeffs[0],
                &self.coeff_counts[0],
                info_y,
                p.quant[0],
                imcu_row,
                &mut y_strip,
                y_strip_width,
                p.idct_fn,
                &mut dequant,
            );

            let y0 = imcu_row * p.mcu_height;
            let rows = p.mcu_height.min(p.height.saturating_sub(y0));
            let out_row = |row: usize| (y0 - start * p.mcu_height + row) * row_bytes;
            match p.upsample {
                None => {
                    // Nearest 4:2:0: each chroma row covers two luma rows and
                    // each chroma sample two pixels.
                    let c_rows = c_strip_height.min((p.height - y0).div_ceil(v_ratio));
                    for row in 0..rows {
                        let y = row * y_strip_width;
                        let c = (1 + (row / 2).min(c_rows - 1)) * c_strip_width;
                        let o = out_row(row);
                        p.kernel.row_h2(
                            &y_strip[y..y + cols],
                            &cb_a[c..c + c_strip_width],
                            &cr_a[c..c + c_strip_width],
                            &mut out[o..o + row_bytes],
                        );
                    }
                }
                Some(upsample) => {
                    // Upsamplers walk the padded strip width: replicate the
                    // last real chroma column over the padding first, as
                    // libjpeg-turbo upsamples only `downsampled_width`.
                    edge_replicate_h_padding(&mut cb_a, downsampled_w, c_strip_width, ext_height);
                    edge_replicate_h_padding(&mut cr_a, downsampled_w, c_strip_width, ext_height);
                    #[cfg(target_arch = "x86_64")]
                    let fused = if let Some(token) = p.fused_h2v2 {
                        super::baseline_streaming::output_fancy_strip_fused(
                            token,
                            &cb_a,
                            &cr_a,
                            c_strip_width,
                            c_strip_height,
                            &y_strip,
                            y_strip_width,
                            out,
                            p.width,
                            y0 - start * p.mcu_height,
                            rows,
                            self.idct_method == IdctMethod::Libjpeg,
                            p.kernel.fused_ycbcr() == Some(true),
                            &mut cs_cb,
                            &mut cs_cr,
                        );
                        true
                    } else {
                        false
                    };
                    #[cfg(not(target_arch = "x86_64"))]
                    let fused = false;
                    if !fused {
                        let up_height = ext_height * v_ratio;
                        upsample(
                            &cb_a,
                            c_strip_width,
                            ext_height,
                            &mut cb_up,
                            y_strip_width,
                            up_height,
                        );
                        upsample(
                            &cr_a,
                            c_strip_width,
                            ext_height,
                            &mut cr_up,
                            y_strip_width,
                            up_height,
                        );
                        for row in 0..rows {
                            let y = row * y_strip_width;
                            // Skip the upsampled above-context rows.
                            let c = (v_ratio + row) * y_strip_width;
                            let o = out_row(row);
                            p.kernel.row(
                                &y_strip[y..y + cols],
                                &cb_up[c..c + cols],
                                &cr_up[c..c + cols],
                                &mut out[o..o + row_bytes],
                            );
                        }
                    }
                }
            }

            if imcu_row + 1 < end {
                // Next MCU row: its above context is this strip's last row.
                cb_b[..c_strip_width].copy_from_slice(&cb_a[last_data..last_data + c_strip_width]);
                cr_b[..c_strip_width].copy_from_slice(&cr_a[last_data..last_data + c_strip_width]);
                core::mem::swap(&mut cb_a, &mut cb_b);
                core::mem::swap(&mut cr_a, &mut cr_b);
                if imcu_row + 2 < p.mcu_rows {
                    idct_chroma(&mut cb_b, &mut cr_b, imcu_row + 2);
                }
            }
        }
        Ok(())
    }
}
