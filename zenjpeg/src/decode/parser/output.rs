//! Pixel output conversion for decoded JPEG data.
//!
//! This module handles the final conversion from decoded DCT coefficients
//! to pixel output in various formats (RGB u8, RGB f32, YCbCr f32).
//!
//! ## Fast Paths
//!
//! - `fast_integer` (sibling module): integer IDCT, upsampling and row kernels
//!   for the shapes `PathPlan::fast_integer` accepts (YCbCr and RGB)
//!
//! ## Generic Paths
//!
//! - `to_pixels`: General f32 path with bias computation, handles XYB
//! - `to_pixels_f32`: f32 output normalized to [0.0, 1.0]
//! - `to_ycbcr_planes_f32`: Raw YCbCr planes for custom processing

use super::super::idct::inverse_dct_8x8;
use super::super::idct_int::{idct_int_auto, idct_int_libjpeg};
use super::super::upsample::{upsample_libjpeg_f32, upsample_nearest_f32};
use crate::color::planes::{Layout8, relayout_u8};
use crate::color::{
    cmyk_planes_to_cmyk_u8, cmyk_planes_to_rgb_u8, gray_f32_to_gray_f32, gray_f32_to_gray_u8,
    gray_f32_to_rgb_f32, gray_f32_to_rgb_u8, ycbcr::rgb_u8_swap_rb_inplace,
    ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8, ycck_planes_to_cmyk_u8,
    ycck_planes_to_rgb_u8,
};
use crate::decode::color_model::ColorModel;
use crate::error::{Error, Result};
use crate::foundation::alloc::{
    checked_size_2d, try_alloc_maybeuninit_pref, try_alloc_zeroed_f32_pref, try_alloc_zeroed_pref,
};
use crate::foundation::consts::{DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER};
use crate::quant::{
    DequantBiasStats, dequantize_block, dequantize_block_i32, dequantize_block_with_bias,
};
use crate::types::PixelFormat;
use enough::Stop;

use super::{CompInfo, JpegParser};

/// Returns true for formats that decode via the RGB u8 fast paths
/// (i16 IDCT → direct u8 output), then optionally reformat.
fn is_rgb_family_u8(format: PixelFormat) -> bool {
    matches!(
        format,
        PixelFormat::Rgb
            | PixelFormat::Bgr
            | PixelFormat::Rgba
            | PixelFormat::Bgra
            | PixelFormat::Bgrx
    )
}

/// The 8-bit interleaved layout of an RGB-family `format`.
fn rgb_family_layout(format: PixelFormat) -> Result<Layout8> {
    Layout8::from_pixel_format(format)
        .filter(|l| *l != Layout8::Gray)
        .ok_or_else(|| Error::internal("not an RGB-family format"))
}

/// f32 IDCT path: dequantize (optionally with Laplacian bias), inverse DCT,
/// and copy the 8x8 block of f32 pixels into `comp_plane_f32` at
/// `(base_px, base_py)`, clipped to `(cols_to_copy, rows_to_copy)`.
///
/// Used for XYB (extended-gamut precision), `dequant_bias` (fractional bias
/// application), and `force_f32_idct` (dimension-swapping transforms need
/// symmetric IDCT). Hot path — inlined into the MCU-row loop.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn idct_block_f32_into_plane(
    natural_coeffs: &[i16; DCT_BLOCK_SIZE],
    quant: &[u16; DCT_BLOCK_SIZE],
    biases: &[f32; DCT_BLOCK_SIZE],
    is_xyb: bool,
    dequant_bias: bool,
    comp_plane_f32: &mut [f32],
    comp_width: usize,
    base_px: usize,
    base_py: usize,
    rows_to_copy: usize,
    cols_to_copy: usize,
) {
    let dequant = if dequant_bias && !is_xyb {
        dequantize_block_with_bias(natural_coeffs, quant, biases)
    } else {
        dequantize_block(natural_coeffs, quant)
    };
    let pixels = inverse_dct_8x8(&dequant);

    if cols_to_copy == DCT_SIZE {
        for y in 0..rows_to_copy {
            let dst_offset = (base_py + y) * comp_width + base_px;
            let src_offset = y * DCT_SIZE;
            comp_plane_f32[dst_offset..dst_offset + DCT_SIZE]
                .copy_from_slice(&pixels[src_offset..src_offset + DCT_SIZE]);
        }
    } else {
        for y in 0..rows_to_copy {
            for x in 0..cols_to_copy {
                comp_plane_f32[(base_py + y) * comp_width + base_px + x] = pixels[y * DCT_SIZE + x];
            }
        }
    }
}

/// Integer IDCT path: dequantize to i32, run the integer IDCT `idct_method`
/// names (the parser's, gray-normalized), then level-shift the i16 [0,255]
/// result to f32 [-128, 127] and copy into `comp_plane_f32` at
/// `(base_px, base_py)`, clipped to `(cols_to_copy, rows_to_copy)`.
///
/// Used by the per-component path (frames the fast integer path does not
/// serve: CMYK/YCCK, exotic sampling) and by `to_ycbcr_planes_f32`. Hot path —
/// inlined into the MCU-row loop.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn idct_block_i16_into_plane(
    natural_coeffs: &[i16; DCT_BLOCK_SIZE],
    quant: &[u16; DCT_BLOCK_SIZE],
    idct_method: super::super::IdctMethod,
    comp_plane_f32: &mut [f32],
    comp_width: usize,
    base_px: usize,
    base_py: usize,
    rows_to_copy: usize,
    cols_to_copy: usize,
) {
    let mut dequant_i32 = dequantize_block_i32(natural_coeffs, quant);
    let mut pixels_i16 = [0i16; DCT_BLOCK_SIZE];
    // The parser's IDCT method, as on every other integer path (it is
    // already the libjpeg kernel for 1-component sources, #154).
    match idct_method {
        super::super::IdctMethod::Libjpeg => idct_int_libjpeg(&mut dequant_i32, &mut pixels_i16, 8),
        super::super::IdctMethod::Jpegli => idct_int_auto(&mut dequant_i32, &mut pixels_i16, 8),
    }

    if cols_to_copy == DCT_SIZE {
        for y in 0..rows_to_copy {
            let dst_offset = (base_py + y) * comp_width + base_px;
            let src_offset = y * DCT_SIZE;
            for x in 0..DCT_SIZE {
                comp_plane_f32[dst_offset + x] = pixels_i16[src_offset + x] as f32 - 128.0;
            }
        }
    } else {
        for y in 0..rows_to_copy {
            for x in 0..cols_to_copy {
                comp_plane_f32[(base_py + y) * comp_width + base_px + x] =
                    pixels_i16[y * DCT_SIZE + x] as f32 - 128.0;
            }
        }
    }
}

/// Pixel output conversion methods for JpegParser.
impl<'a> JpegParser<'a> {
    // =========================================================================
    // Shared helpers for f32 decode paths
    // =========================================================================

    /// Set up MCU grid, component info, bias tracking, and f32 planes
    /// for all three f32 output paths (to_pixels, to_pixels_f32, to_ycbcr).
    ///
    /// Returns (comp_infos, bias_stats, component_biases, comp_planes_f32,
    /// mcu_rows, max_h_samp, max_v_samp).
    fn setup_f32_decode(
        &self,
        num_components: usize,
    ) -> Result<(
        Vec<CompInfo>,
        DequantBiasStats,
        Vec<[f32; DCT_BLOCK_SIZE]>,
        Vec<Vec<f32>>,
        usize,
        u8,
        u8,
    )> {
        let width = self.width as usize;
        let height = self.height as usize;

        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..num_components {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }

        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        let comp_infos = self.build_comp_infos(
            mcu_cols,
            mcu_rows,
            max_h_samp as usize,
            max_v_samp as usize,
            num_components,
        )?;

        let bias_stats = DequantBiasStats::new(num_components);
        let component_biases = vec![[0.0f32; DCT_BLOCK_SIZE]; num_components];

        let mut comp_planes_f32 = Vec::with_capacity(num_components);
        for info in &comp_infos {
            let size = checked_size_2d(info.comp_width, info.comp_height)?;
            // Full-image per-component f32 plane → default fallible.
            comp_planes_f32.push(try_alloc_zeroed_f32_pref(
                self.alloc_pref,
                true,
                size,
                "f32 component plane",
            )?);
        }

        Ok((
            comp_infos,
            bias_stats,
            component_biases,
            comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ))
    }

    /// Gather bias statistics for full-res components in a single MCU row.
    fn gather_bias_stats(
        &self,
        imcu_row: usize,
        comp_idx: usize,
        info: &CompInfo,
        bias_stats: &mut DequantBiasStats,
    ) {
        if !info.is_full_res {
            return;
        }
        for iy in 0..info.v_samp {
            let by = imcu_row * info.v_samp + iy;
            if by >= info.comp_blocks_v {
                continue;
            }
            for bx in 0..info.comp_blocks_h {
                let block_idx = by * info.comp_blocks_h + bx;
                if block_idx >= self.coeffs[comp_idx].len() {
                    continue;
                }
                let coeffs = &self.coeffs[comp_idx][block_idx];
                let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                    natural_coeffs[zi as usize] = coeffs[i];
                }
                bias_stats.gather_block(comp_idx, &natural_coeffs);
            }
        }
    }

    /// Apply deblocking filters to component planes in-place.
    ///
    /// Planes are centered around 0 (IDCT output: [-128, 127]). Boundary 4-tap
    /// expects [0, 255], so we level-shift before filtering and shift back after.
    /// Knusperli replaces the plane entirely (it does its own IDCT from coefficients).
    fn apply_deblock_to_planes(
        &self,
        comp_planes: &mut [Vec<f32>],
        comp_infos: &[CompInfo],
        mode: super::super::DeblockMode,
    ) -> Result<()> {
        use super::super::DeblockMode;

        if mode == DeblockMode::Off {
            return Ok(());
        }

        for (comp_idx, info) in comp_infos.iter().enumerate() {
            let quant = self.quant_tables[info.quant_idx]
                .as_ref()
                .ok_or(Error::internal("missing quant table for deblock"))?;
            let dc_quant = quant[0];

            // Decide strategy per-component
            let use_knusperli = match mode {
                DeblockMode::Off => unreachable!(),
                DeblockMode::Knusperli => true,
                DeblockMode::Boundary4Tap | DeblockMode::AutoStreamable => false,
                DeblockMode::Auto => {
                    // Simple heuristic: knusperli at low Q (high DC quant), boundary otherwise
                    dc_quant >= 27
                }
            };

            if use_knusperli && comp_idx < self.coeffs.len() {
                // Knusperli: replace plane with its own IDCT + boundary correction.
                // Output is [0, 255] range; shift to [-128, 127] to match pipeline.
                // Coefficients are stored as Vec<[i16; 64]>; flatten to &[i16] for knusperli.
                let flat_coeffs: &[i16] = bytemuck::cast_slice(&self.coeffs[comp_idx]);
                let mut plane = crate::deblock::knusperli::process_component(
                    flat_coeffs,
                    info.comp_blocks_h,
                    info.comp_blocks_v,
                    quant,
                );
                // Level-shift from [0, 255] to [-128, 127] to match the rest of the pipeline
                for v in &mut plane {
                    *v -= 128.0;
                }
                comp_planes[comp_idx] = plane;
            } else {
                // Boundary 4-tap: level-shift to [0, 255], filter in-place, shift back.
                let plane = &mut comp_planes[comp_idx];
                let (w, h) = (info.comp_width, info.comp_height);

                // Shift to [0, 255]
                for v in plane.iter_mut() {
                    *v += 128.0;
                }

                let strength = crate::deblock::BoundaryStrength::from_dc_quant(dc_quant);
                crate::deblock::filter_plane_boundary_4tap(plane, w, h, strength);

                // Shift back to [-128, 127]
                for v in plane.iter_mut() {
                    *v -= 128.0;
                }
            }
        }

        Ok(())
    }

    /// Upsample component f32 planes to full image resolution.
    ///
    /// Handles all chroma upsampling modes (Triangle, Jpegli, NearestNeighbor).
    /// Full-res components are clipped to image dimensions without interpolation.
    fn upsample_planes_f32(
        &self,
        comp_planes_f32: &mut [Vec<f32>],
        comp_infos: &[CompInfo],
        max_h_samp: u8,
        max_v_samp: u8,
        chroma_upsampling: super::super::ChromaUpsampling,
    ) -> Result<Vec<Vec<f32>>> {
        let width = self.width as usize;
        let height = self.height as usize;
        let output_size = checked_size_2d(width, height)?;
        let mut planes_f32 = Vec::with_capacity(comp_infos.len());

        for (comp_idx, info) in comp_infos.iter().enumerate() {
            let plane = if info.h_samp < max_h_samp as usize || info.v_samp < max_v_samp as usize {
                let scale_x = max_h_samp as usize / info.h_samp;
                let scale_y = max_v_samp as usize / info.v_samp;

                // Edge-replicate the block-padding region of the chroma plane
                // before upsampling, mirroring libjpeg-turbo's
                // `set_bottom_pointers`. Non-MCU-aligned images have stale
                // IDCT output in the padding rows/columns, and the fancy
                // upsamplers read `input[(in_y+1) * stride + in_x]` at image
                // boundaries, which otherwise lands inside the garbage region.
                //
                // Only touches O(comp_width + comp_height) elements in the
                // padding area; MCU-aligned files skip it entirely.
                let real_w =
                    (width * info.h_samp + (max_h_samp as usize) - 1) / (max_h_samp as usize);
                let real_h =
                    (height * info.v_samp + (max_v_samp as usize) - 1) / (max_v_samp as usize);
                let pad_right = real_w < info.comp_width && real_w > 0;
                let pad_bottom = real_h < info.comp_height && real_h > 0;

                if pad_right || pad_bottom {
                    let comp_plane = &mut comp_planes_f32[comp_idx];
                    let cw = info.comp_width;

                    // Right-edge column replication over real content rows.
                    if pad_right {
                        for row in 0..real_h {
                            let row_start = row * cw;
                            let last = comp_plane[row_start + real_w - 1];
                            for x in real_w..cw {
                                comp_plane[row_start + x] = last;
                            }
                        }
                    }

                    // Bottom-edge row replication (picks up any right-pad fill
                    // from the previous step).
                    if pad_bottom {
                        let src_start = (real_h - 1) * cw;
                        let src_end = src_start + cw;
                        // Split the vec so we can borrow src immutably and
                        // dst mutably at the same time without allocating.
                        let (before_dst, dst_slab) = comp_plane.split_at_mut(real_h * cw);
                        let src_row = &before_dst[src_start..src_end];
                        for row_idx in 0..(info.comp_height - real_h) {
                            let dst_start = row_idx * cw;
                            dst_slab[dst_start..dst_start + cw].copy_from_slice(src_row);
                        }
                    }
                }

                let comp_plane: &[f32] = &comp_planes_f32[comp_idx];

                match chroma_upsampling {
                    super::super::ChromaUpsampling::Triangle => upsample_libjpeg_f32(
                        comp_plane,
                        info.comp_width,
                        info.comp_height,
                        width,
                        height,
                        scale_x,
                        scale_y,
                    ),
                    super::super::ChromaUpsampling::NearestNeighbor => {
                        // Full-image upsampled f32 plane → default fallible.
                        let mut upsampled = try_alloc_zeroed_f32_pref(
                            self.alloc_pref,
                            true,
                            output_size,
                            "f32 nearest upsample plane",
                        )?;
                        upsample_nearest_f32(
                            comp_plane,
                            info.comp_width,
                            info.comp_height,
                            &mut upsampled,
                            width,
                            height,
                            scale_x,
                            scale_y,
                        );
                        upsampled
                    }
                }
            } else {
                let comp_plane = &comp_planes_f32[comp_idx];
                // Full resolution — clip to image dimensions
                let mut plane = try_alloc_zeroed_f32_pref(
                    self.alloc_pref,
                    true,
                    output_size,
                    "f32 full-res output plane",
                )?;
                for py in 0..height {
                    let src = &comp_plane[py * info.comp_width..py * info.comp_width + width];
                    let dst = &mut plane[py * width..py * width + width];
                    dst.copy_from_slice(src);
                }
                plane
            };

            planes_f32.push(plane);
        }

        Ok(planes_f32)
    }

    // =========================================================================
    // Public output methods
    // =========================================================================

    /// Convert decoded coefficients to pixels in the requested format.
    ///
    /// This is the main entry point for pixel output. It selects the fastest
    /// available path based on the image characteristics:
    ///
    /// 1. **Streaming result** (`streaming_rgb`): Already decoded during `parse_scan()`
    ///    in a single entropy→IDCT→color pass. Zero-copy for RGB, reformat for others.
    /// 2. **Fused parallel result** (`fused_result`): Entropy+IDCT+color per restart
    ///    segment via rayon. Activated by `DecodeMode::Auto` + DRI + `parallel` feature.
    /// 3. **Fast integer path** (`fast_integer_plan`): integer IDCT, upsampling
    ///    and row kernel from buffered coefficients, for the shapes and models
    ///    `PathPlan::fast_integer` accepts, without dequantization bias.
    /// 4. **f32 generic path**: Full f32 pipeline with dequant bias, XYB support.
    ///
    /// Paths 1-2 produce results during entropy decode (no separate output pass).
    /// Wave-parallel scanline decode is handled by `ScanlineReader`, not here.
    ///
    /// The `stop` parameter allows cancellation of long-running operations.
    #[allow(clippy::wrong_self_convention)] // Takes &mut self to take() internal buffer
    pub(in crate::decode) fn to_pixels(
        &mut self,
        format: PixelFormat,
        chroma_upsampling: super::super::ChromaUpsampling,
        output_target: super::super::OutputTarget,
        _stop: &impl Stop,
    ) -> Result<Vec<u8>> {
        let dequant_bias = output_target.uses_dequant_bias();
        let width = self.width as usize;
        let height = self.height as usize;

        // A single-pass result (streaming or fused): convert it to `format`.
        // The target is checked before the buffer is taken, so a format the
        // result cannot serve is refused rather than leaving no data behind.
        if self.has_single_pass(dequant_bias) {
            let target = self
                .single_pass_target(format)
                .ok_or_else(|| Error::unsupported_feature("unsupported color conversion"))?;
            if let Some((buf, source)) = self.take_single_pass(dequant_bias) {
                return self.relayout_owned(buf, source, target, checked_size_2d(width, height)?);
            }
        }

        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        // Full-frame pixel output, sized from the (untrusted) SOF dimensions.
        // Charged here rather than at each of the dozen concrete allocation
        // sites below because they are mutually exclusive alternatives — this
        // is the one buffer every buffered path produces. The streaming and
        // fused-parallel results returned above are charged where they are
        // allocated instead, so neither is counted twice. Transient reformat
        // scratch (one extra frame during a conversion that cannot be done in
        // place) is not separately charged.
        self.charge_memory(
            width
                .saturating_mul(height)
                .saturating_mul(format.bytes_per_pixel()),
            "pixel output buffer",
        )?;

        // Integer IDCT + integer upsampling + row kernel, straight into the
        // requested layout (dequantization bias needs the f32 path).
        if !dequant_bias && let Some(plan) = self.fast_integer_plan(format, chroma_upsampling)? {
            let mut out = try_alloc_maybeuninit_pref(
                self.alloc_pref,
                true,
                plan.output_len()?,
                "pixel output buffer",
            )?;
            self.fast_integer_into(&plan, &mut out)?;
            return Ok(out);
        }

        let is_xyb = self.color_model == ColorModel::Xyb;
        let num_components = self.num_components as usize;
        let (
            comp_infos,
            mut bias_stats,
            mut component_biases,
            mut comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ) = self.setup_f32_decode(num_components)?;

        self.decode_coeffs_to_planes_f32(
            num_components,
            &comp_infos,
            &mut bias_stats,
            &mut component_biases,
            &mut comp_planes_f32,
            mcu_rows,
            is_xyb,
            dequant_bias,
        )?;

        // Upsample and convert to output format
        let planes_f32 = self.upsample_planes_f32(
            &mut comp_planes_f32,
            &comp_infos,
            max_h_samp,
            max_v_samp,
            chroma_upsampling,
        )?;

        self.dispatch_planes_to_output_u8(format, &planes_f32, width, height)
    }

    /// Run the per-MCU-row IDCT loop for `to_pixels`, populating `comp_planes_f32`.
    ///
    /// Per component per MCU row: gather bias stats, refresh per-component
    /// biases every 4 MCU rows (matching C++ incremental bias recomputation),
    /// then IDCT each block via either the f32 path (XYB / dequant_bias /
    /// `force_f32_idct`) or the standard integer path.
    #[allow(clippy::too_many_arguments)]
    fn decode_coeffs_to_planes_f32(
        &self,
        num_components: usize,
        comp_infos: &[CompInfo],
        bias_stats: &mut DequantBiasStats,
        component_biases: &mut [[f32; DCT_BLOCK_SIZE]],
        comp_planes_f32: &mut [Vec<f32>],
        mcu_rows: usize,
        is_xyb: bool,
        dequant_bias: bool,
    ) -> Result<()> {
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..num_components {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                self.gather_bias_stats(imcu_row, comp_idx, info, bias_stats);
                if info.is_full_res && imcu_row % 4 == 3 {
                    component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                }

                let biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    let base_py = by * DCT_SIZE;
                    let rows_to_copy = DCT_SIZE.min(info.comp_height.saturating_sub(base_py));

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));

                        if is_xyb || dequant_bias || self.force_f32_idct {
                            idct_block_f32_into_plane(
                                &natural_coeffs,
                                quant,
                                biases,
                                is_xyb,
                                dequant_bias,
                                comp_plane_f32,
                                info.comp_width,
                                base_px,
                                base_py,
                                rows_to_copy,
                                cols_to_copy,
                            );
                        } else {
                            idct_block_i16_into_plane(
                                &natural_coeffs,
                                quant,
                                self.idct_method,
                                comp_plane_f32,
                                info.comp_width,
                                base_px,
                                base_py,
                                rows_to_copy,
                                cols_to_copy,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a single-pass result [`Self::take_single_pass`] would return
    /// is waiting.
    fn has_single_pass(&self, dequant_bias: bool) -> bool {
        if self.streaming_rgb.is_some() {
            return true;
        }
        #[cfg(feature = "parallel")]
        if !dequant_bias && self.fused_result.is_some() {
            return true;
        }
        let _ = dequant_bias;
        false
    }

    /// Take the result of a single-pass decode, with the layout it was
    /// written in: the streaming path writes [`Self::streaming_layout`], the
    /// fused parallel path packed RGB (its result is not used for
    /// dequantization-bias output, which needs the coefficients).
    fn take_single_pass(&mut self, dequant_bias: bool) -> Option<(Vec<u8>, Layout8)> {
        if !self.has_single_pass(dequant_bias) {
            return None;
        }
        if let Some(buf) = self.streaming_rgb.take() {
            return Some((buf, self.streaming_layout()));
        }
        #[cfg(feature = "parallel")]
        if !dequant_bias
            && let Some(super::super::fused_parallel::FusedResult(rgb)) = self.fused_result.take()
        {
            return Some((rgb, Layout8::Rgb));
        }
        let _ = dequant_bias;
        None
    }

    /// The 8-bit layout a single-pass result can be relaid into for `format`.
    /// A streaming result already written in `Layout8::Gray` serves `Gray`
    /// directly; an RGB single-pass result only relays to `Gray` for grayscale
    /// or RGB frames (for YCbCr frames, gray output is the Y plane, not the
    /// BT.601 luma of clamped RGB).
    fn single_pass_target(&self, format: PixelFormat) -> Option<Layout8> {
        let source_is_gray =
            self.streaming_rgb.is_some() && self.streaming_layout() == Layout8::Gray;
        Layout8::from_pixel_format(format).filter(|&t| {
            t != Layout8::Gray
                || source_is_gray
                || self.num_components == 1
                || self.color_model == ColorModel::Rgb
        })
    }

    /// `buf` (packed `source`, `npixels` pixels) in `target`: a move when the
    /// layouts match, an in-place swap for RGB to BGR, else one new buffer.
    fn relayout_owned(
        &self,
        mut buf: Vec<u8>,
        source: Layout8,
        target: Layout8,
        npixels: usize,
    ) -> Result<Vec<u8>> {
        let have = checked_size_2d(npixels, source.bytes_per_pixel())?;
        if buf.len() < have {
            return Err(Error::internal("pixel buffer smaller than the frame"));
        }
        if source == target {
            return Ok(buf);
        }
        if (source, target) == (Layout8::Rgb, Layout8::Bgr) {
            rgb_u8_swap_rb_inplace(&mut buf[..have]);
            return Ok(buf);
        }
        let mut out = try_alloc_zeroed_pref(
            self.alloc_pref,
            true,
            checked_size_2d(npixels, target.bytes_per_pixel())?,
            "pixel relayout",
        )?;
        relayout_u8(&buf[..have], source, &mut out, target);
        Ok(out)
    }

    /// The layout the streaming baseline path writes: one byte per pixel for
    /// a grayscale frame, else the hinted 8-bit format (RGB without a hint).
    /// The streaming decode picks its row kernel from this and
    /// [`Self::take_single_pass`] reports it with the buffer.
    pub(super) fn streaming_layout(&self) -> Layout8 {
        if self.num_components == 1 {
            return Layout8::Gray;
        }
        self.streaming_output_format
            .and_then(Layout8::from_pixel_format)
            .unwrap_or(Layout8::Rgb)
    }

    /// Dispatch upsampled f32 planes to the requested u8 output format.
    ///
    /// Handles the cross-product of `(num_components, format)`:
    /// - `(1, Gray)`        → level-shifted grayscale
    /// - `(1, RGB-family)`  → gray-expanded RGB, reformatted
    /// - `(3, Gray)`        → Y plane for YCbCr, BT.601 luma of RGB for RGB / XYB
    /// - `(3, RGB-family)`  → XYB / RGB-JPEG / YCbCr → RGB, reformatted
    /// - `(4, RGB-family)`  → CMYK / YCCK → RGB, reformatted
    /// - `(4, Cmyk)`        → raw CMYK / YCCK → CMYK interleave
    fn dispatch_planes_to_output_u8(
        &self,
        format: PixelFormat,
        planes_f32: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let output_size = checked_size_2d(width, height)?;
        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                let mut output = try_alloc_zeroed_pref(
                    self.alloc_pref,
                    true,
                    output_size,
                    "grayscale u8 output",
                )?;
                gray_f32_to_gray_u8(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, f) if is_rgb_family_u8(f) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb =
                    try_alloc_zeroed_pref(self.alloc_pref, true, rgb_size, "gray->RGB u8 output")?;
                gray_f32_to_rgb_u8(&planes_f32[0], &mut rgb);
                self.relayout_owned(rgb, Layout8::Rgb, rgb_family_layout(f)?, output_size)
            }
            (3, PixelFormat::Gray) => {
                if self.color_model == ColorModel::YCbCr {
                    let mut output = try_alloc_zeroed_pref(
                        self.alloc_pref,
                        true,
                        output_size,
                        "grayscale u8 output",
                    )?;
                    gray_f32_to_gray_u8(&planes_f32[0], &mut output);
                    Ok(output)
                } else {
                    let rgb = self.three_chan_planes_to_rgb_u8(planes_f32, output_size)?;
                    self.relayout_owned(rgb, Layout8::Rgb, Layout8::Gray, output_size)
                }
            }
            (3, f) if is_rgb_family_u8(f) => {
                let rgb = self.three_chan_planes_to_rgb_u8(planes_f32, output_size)?;
                self.relayout_owned(rgb, Layout8::Rgb, rgb_family_layout(f)?, output_size)
            }
            (4, f) if is_rgb_family_u8(f) => {
                let rgb = self.four_chan_planes_to_rgb_u8(planes_f32, width, height)?;
                self.relayout_owned(rgb, Layout8::Rgb, rgb_family_layout(f)?, output_size)
            }
            (4, PixelFormat::Cmyk) => self.four_chan_planes_to_cmyk_u8(planes_f32, width, height),
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// XYB / RGB / YCbCr → 3-byte interleaved RGB u8, by the stored model.
    fn three_chan_planes_to_rgb_u8(
        &self,
        planes_f32: &[Vec<f32>],
        output_size: usize,
    ) -> Result<Vec<u8>> {
        let rgb_size = checked_size_2d(output_size, 3)?;
        let mut rgb =
            try_alloc_zeroed_pref(self.alloc_pref, true, rgb_size, "3-channel RGB u8 output")?;
        let [c0, c1, c2] = [&planes_f32[0], &planes_f32[1], &planes_f32[2]];
        match self.color_model {
            // Run the inverse XYB → sRGB transform in-decoder (no CMS
            // required). Output is visibly-correct sRGB.
            ColorModel::Xyb => crate::color::xyb::xyb_planes_to_srgb_u8_simd(c0, c1, c2, &mut rgb),
            // Level-shift from [-128..127] to [0..255] and interleave.
            ColorModel::Rgb => {
                for (px, ((&r, &g), &b)) in rgb
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(c0.iter().zip(c1).zip(c2))
                {
                    let q = |v: f32| (v + 128.0).round().clamp(0.0, 255.0) as u8;
                    px.copy_from_slice(&[q(r), q(g), q(b)]);
                }
            }
            ColorModel::YCbCr => ycbcr_planes_f32_to_rgb_u8(c0, c1, c2, &mut rgb),
            // Not three-plane models: a new model must say how its planes
            // convert here rather than decode as YCbCr.
            ColorModel::Gray | ColorModel::Cmyk | ColorModel::Ycck | ColorModel::Unknown => {
                return Err(Error::unsupported_feature(
                    "three-plane RGB conversion of a frame that is not YCbCr, RGB or XYB",
                ));
            }
        }
        Ok(rgb)
    }

    /// CMYK or YCCK 4-plane → 3-byte interleaved RGB u8.
    ///
    /// Adobe APP14 transform=2 routes through YCCK; otherwise raw CMYK
    /// with Adobe-inverted convention.
    fn four_chan_planes_to_rgb_u8(
        &self,
        planes_f32: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let rgb_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
        let mut rgb =
            try_alloc_zeroed_pref(self.alloc_pref, true, rgb_size, "CMYK/YCCK->RGB u8 output")?;
        match self.color_model {
            ColorModel::Ycck => {
                ycck_planes_to_rgb_u8(
                    &planes_f32[0],
                    &planes_f32[1],
                    &planes_f32[2],
                    &planes_f32[3],
                    &mut rgb,
                );
            }
            ColorModel::Gray
            | ColorModel::YCbCr
            | ColorModel::Rgb
            | ColorModel::Xyb
            | ColorModel::Unknown => {
                return Err(Error::unsupported_feature(
                    "four-plane RGB conversion of a frame that is not CMYK or YCCK",
                ));
            }
            ColorModel::Cmyk => {
                cmyk_planes_to_rgb_u8(
                    &planes_f32[0],
                    &planes_f32[1],
                    &planes_f32[2],
                    &planes_f32[3],
                    &mut rgb,
                );
            }
        }
        Ok(rgb)
    }

    /// CMYK or YCCK 4-plane → raw 4-byte interleaved CMYK u8.
    fn four_chan_planes_to_cmyk_u8(
        &self,
        planes_f32: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let cmyk_size = checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 4))?;
        let mut cmyk_out =
            try_alloc_zeroed_pref(self.alloc_pref, true, cmyk_size, "CMYK u8 output")?;
        match self.color_model {
            ColorModel::Ycck => {
                ycck_planes_to_cmyk_u8(
                    &planes_f32[0],
                    &planes_f32[1],
                    &planes_f32[2],
                    &planes_f32[3],
                    &mut cmyk_out,
                );
            }
            ColorModel::Gray
            | ColorModel::YCbCr
            | ColorModel::Rgb
            | ColorModel::Xyb
            | ColorModel::Unknown => {
                return Err(Error::unsupported_feature(
                    "CMYK output of a frame that is not CMYK or YCCK",
                ));
            }
            ColorModel::Cmyk => {
                cmyk_planes_to_cmyk_u8(
                    &planes_f32[0],
                    &planes_f32[1],
                    &planes_f32[2],
                    &planes_f32[3],
                    &mut cmyk_out,
                );
            }
        }
        Ok(cmyk_out)
    }

    /// Convert decoded data to pixels, writing directly into a caller-provided
    /// buffer. Returns the number of bytes written.
    ///
    /// Single-pass (streaming or fused) results are copied, or relaid out,
    /// into `dst`; the fast integer path renders straight into `dst`; anything
    /// else goes through [`Self::to_pixels`] and a reformat.
    pub(in crate::decode) fn write_pixels_into(
        &mut self,
        format: PixelFormat,
        chroma_upsampling: super::super::ChromaUpsampling,
        output_target: super::super::OutputTarget,
        stop: &impl Stop,
        dst: &mut [u8],
    ) -> Result<usize> {
        let dequant_bias = output_target.uses_dequant_bias();
        let width = self.width as usize;
        let height = self.height as usize;
        let copy = |px: &[u8], dst: &mut [u8]| -> Result<usize> {
            dst.get_mut(..px.len())
                .ok_or_else(|| Error::internal("dst too small"))?
                .copy_from_slice(px);
            Ok(px.len())
        };

        // A single-pass result is relaid straight into `dst`, with no frame
        // in between. The target is checked before the buffer is taken.
        if self.has_single_pass(dequant_bias) {
            let target = self
                .single_pass_target(format)
                .ok_or_else(|| Error::unsupported_feature("unsupported color conversion"))?;
            let (buf, source) = self
                .take_single_pass(dequant_bias)
                .ok_or_else(|| Error::internal("single-pass result vanished"))?;
            let npixels = checked_size_2d(width, height)?;
            let have = checked_size_2d(npixels, source.bytes_per_pixel())?;
            let need = checked_size_2d(npixels, target.bytes_per_pixel())?;
            let src = buf
                .get(..have)
                .ok_or_else(|| Error::internal("single-pass result smaller than the frame"))?;
            let out = dst
                .get_mut(..need)
                .ok_or_else(|| Error::internal("dst too small"))?;
            relayout_u8(src, source, out, target);
            return Ok(need);
        }

        if !dequant_bias && let Some(plan) = self.fast_integer_plan(format, chroma_upsampling)? {
            self.fast_integer_into(&plan, dst)?;
            return plan.output_len();
        }

        // Everything else converts as `decode()` does: to the frame's 8-bit
        // base (gray for one component, RGB otherwise), relaid straight into
        // `dst`, so the only frame-sized buffer besides `dst` is that base.
        // Gray from a colour frame and the non-RGB formats (CMYK) go through
        // `to_pixels(format)` itself, which serves or refuses them.
        let base = if self.num_components == 1 {
            (PixelFormat::Gray, Layout8::Gray)
        } else {
            (PixelFormat::Rgb, Layout8::Rgb)
        };
        let target = Layout8::from_pixel_format(format)
            .filter(|t| *t != Layout8::Gray || base.1 == Layout8::Gray);
        let Some(target) = target else {
            let pixels = self.to_pixels(format, chroma_upsampling, output_target, stop)?;
            return copy(&pixels, dst);
        };
        let pixels = self.to_pixels(base.0, chroma_upsampling, output_target, stop)?;
        let npixels = checked_size_2d(width, height)?;
        let have = checked_size_2d(npixels, base.1.bytes_per_pixel())?;
        let need = checked_size_2d(npixels, target.bytes_per_pixel())?;
        let src = pixels
            .get(..have)
            .ok_or_else(|| Error::internal("decoded frame smaller than the image"))?;
        let out = dst
            .get_mut(..need)
            .ok_or_else(|| Error::internal("dst too small"))?;
        relayout_u8(src, base.1, out, target);
        Ok(need)
    }

    /// Convert decoded coefficients to f32 pixels, normalized to [0, 1].
    pub(in crate::decode) fn to_pixels_f32(
        &self,
        format: PixelFormat,
        chroma_upsampling: super::super::ChromaUpsampling,
        _stop: &impl Stop,
    ) -> Result<Vec<f32>> {
        self.to_pixels_f32_inner(
            format,
            chroma_upsampling,
            super::super::DeblockMode::Off,
            _stop,
        )
    }

    /// Internal f32 pixel conversion with optional deblocking.
    ///
    /// When `deblock_mode` is not `Off`, deblocking is applied to component
    /// planes after IDCT but before chroma upsampling and color conversion.
    pub(in crate::decode) fn to_pixels_f32_deblock(
        &self,
        format: PixelFormat,
        chroma_upsampling: super::super::ChromaUpsampling,
        deblock_mode: super::super::DeblockMode,
        _stop: &impl Stop,
    ) -> Result<Vec<f32>> {
        self.to_pixels_f32_inner(format, chroma_upsampling, deblock_mode, _stop)
    }

    fn to_pixels_f32_inner(
        &self,
        format: PixelFormat,
        chroma_upsampling: super::super::ChromaUpsampling,
        deblock_mode: super::super::DeblockMode,
        _stop: &impl Stop,
    ) -> Result<Vec<f32>> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }
        let is_xyb = self.color_model == ColorModel::Xyb;

        let num_components = self.num_components as usize;
        let (
            comp_infos,
            mut bias_stats,
            mut component_biases,
            mut comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ) = self.setup_f32_decode(num_components)?;

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..num_components {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                self.gather_bias_stats(imcu_row, comp_idx, info, &mut bias_stats);
                if info.is_full_res && imcu_row % 4 == 3 {
                    component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                }

                // IDCT for this component — always f32 for f32 output
                let biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // Always use f32 IDCT for f32 output - preserves fractional precision
                        let dequant = if is_xyb {
                            dequantize_block(&natural_coeffs, quant)
                        } else {
                            dequantize_block_with_bias(&natural_coeffs, quant, biases)
                        };
                        let pixels = inverse_dct_8x8(&dequant);

                        for y in 0..DCT_SIZE {
                            for x in 0..DCT_SIZE {
                                let px = bx * DCT_SIZE + x;
                                let py = by * DCT_SIZE + y;
                                if px < info.comp_width && py < info.comp_height {
                                    comp_plane_f32[py * info.comp_width + px] =
                                        pixels[y * DCT_SIZE + x];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply deblocking to component planes (between IDCT and upsampling).
        // Deblocking operates in [0, 255] pixel domain, so we level-shift
        // the centered IDCT output (+128), filter, then shift back (-128).
        self.apply_deblock_to_planes(&mut comp_planes_f32, &comp_infos, deblock_mode)?;

        // Upsample and convert to output format
        let planes_f32 = self.upsample_planes_f32(
            &mut comp_planes_f32,
            &comp_infos,
            max_h_samp,
            max_v_samp,
            chroma_upsampling,
        )?;

        let width = self.width as usize;
        let height = self.height as usize;
        let output_size = checked_size_2d(width, height)?;

        match (self.num_components, format) {
            (1, PixelFormat::Gray) => {
                let mut output = try_alloc_zeroed_f32_pref(
                    self.alloc_pref,
                    true,
                    output_size,
                    "gray f32 output",
                )?;
                gray_f32_to_gray_f32(&planes_f32[0], &mut output);
                Ok(output)
            }
            (1, PixelFormat::Rgb) => {
                let rgb_size =
                    checked_size_2d(width, height).and_then(|s| checked_size_2d(s, 3))?;
                let mut rgb = try_alloc_zeroed_f32_pref(
                    self.alloc_pref,
                    true,
                    rgb_size,
                    "gray->RGB f32 output",
                )?;
                gray_f32_to_rgb_f32(&planes_f32[0], &mut rgb);
                Ok(rgb)
            }
            (3, PixelFormat::Gray) => {
                let mut output = try_alloc_zeroed_f32_pref(
                    self.alloc_pref,
                    true,
                    output_size,
                    "gray f32 output",
                )?;
                if self.color_model == ColorModel::YCbCr {
                    gray_f32_to_gray_f32(&planes_f32[0], &mut output);
                } else {
                    use crate::foundation::consts::{YCBCR_B_TO_Y, YCBCR_G_TO_Y, YCBCR_R_TO_Y};
                    let rgb = self.three_chan_planes_to_rgb_f32(&planes_f32, output_size)?;
                    for (o, &[r, g, b]) in output.iter_mut().zip(rgb.as_chunks::<3>().0) {
                        *o = YCBCR_R_TO_Y * r + YCBCR_G_TO_Y * g + YCBCR_B_TO_Y * b;
                    }
                }
                Ok(output)
            }
            (3, PixelFormat::Rgb) => self.three_chan_planes_to_rgb_f32(&planes_f32, output_size),
            _ => Err(Error::unsupported_feature("unsupported color conversion")),
        }
    }

    /// XYB / RGB / YCbCr → 3-channel interleaved RGB f32, by the stored model.
    fn three_chan_planes_to_rgb_f32(
        &self,
        planes_f32: &[Vec<f32>],
        output_size: usize,
    ) -> Result<Vec<f32>> {
        let rgb_size = checked_size_2d(output_size, 3)?;
        let mut rgb = try_alloc_zeroed_f32_pref(
            self.alloc_pref,
            true,
            rgb_size,
            "3-channel RGB f32 output",
        )?;

        let [c0, c1, c2] = [&planes_f32[0], &planes_f32[1], &planes_f32[2]];
        match self.color_model {
            // The inverse XYB → sRGB transform in-decoder (no CMS
            // required): sRGB gamma-encoded f32.
            ColorModel::Xyb => {
                crate::color::xyb::xyb_planes_to_srgb_f32_simd(c0, c1, c2, &mut rgb)
            }
            // Level-shift and normalize to [0.0, 1.0].
            ColorModel::Rgb => {
                for (px, ((&r, &g), &b)) in rgb
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(c0.iter().zip(c1).zip(c2))
                {
                    px.copy_from_slice(&[
                        (r + 128.0) / 255.0,
                        (g + 128.0) / 255.0,
                        (b + 128.0) / 255.0,
                    ]);
                }
            }
            ColorModel::YCbCr => ycbcr_planes_f32_to_rgb_f32(c0, c1, c2, &mut rgb),
            ColorModel::Gray | ColorModel::Cmyk | ColorModel::Ycck | ColorModel::Unknown => {
                return Err(Error::unsupported_feature(
                    "three-plane RGB conversion of a frame that is not YCbCr, RGB or XYB",
                ));
            }
        }
        Ok(rgb)
    }

    /// Convert decoded coefficients to YCbCr f32 planes.
    ///
    /// Returns (Y, Cb, Cr) planes, each width×height in size.
    /// Values are in centered range [-128, 127] (raw DCT output).
    /// Chroma planes are upsampled to full resolution. RGB frames are
    /// converted with the BT.601 matrix.
    pub(in crate::decode) fn to_ycbcr_planes_f32(
        &self,
        chroma_upsampling: super::super::ChromaUpsampling,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if self.coeffs.is_empty() {
            return Err(Error::internal("no decoded data"));
        }

        if self.num_components != 3 {
            return Err(Error::unsupported_feature(
                "YCbCr planes require 3-component image",
            ));
        }

        let num_components = self.num_components as usize;
        let (
            comp_infos,
            mut bias_stats,
            mut component_biases,
            mut comp_planes_f32,
            mcu_rows,
            max_h_samp,
            max_v_samp,
        ) = self.setup_f32_decode(num_components)?;

        // Process MCU row by MCU row
        for imcu_row in 0..mcu_rows {
            for comp_idx in 0..num_components {
                let info = &comp_infos[comp_idx];
                let quant = self.quant_tables[info.quant_idx]
                    .as_ref()
                    .ok_or(Error::internal("missing quantization table"))?;

                self.gather_bias_stats(imcu_row, comp_idx, info, &mut bias_stats);
                if info.is_full_res && imcu_row % 4 == 3 {
                    component_biases[comp_idx] = bias_stats.compute_biases(comp_idx);
                }

                // IDCT — integer path for YCbCr output
                let _biases = &component_biases[comp_idx];
                let comp_plane_f32 = &mut comp_planes_f32[comp_idx];

                for iy in 0..info.v_samp {
                    let by = imcu_row * info.v_samp + iy;
                    if by >= info.comp_blocks_v {
                        continue;
                    }

                    let base_py = by * DCT_SIZE;
                    let rows_to_copy = DCT_SIZE.min(info.comp_height.saturating_sub(base_py));

                    for bx in 0..info.comp_blocks_h {
                        let block_idx = by * info.comp_blocks_h + bx;
                        if block_idx >= self.coeffs[comp_idx].len() {
                            continue;
                        }
                        let coeffs = &self.coeffs[comp_idx][block_idx];

                        // Zigzag reorder
                        let mut natural_coeffs = [0i16; DCT_BLOCK_SIZE];
                        for (i, &zi) in JPEG_NATURAL_ORDER[..DCT_BLOCK_SIZE].iter().enumerate() {
                            natural_coeffs[zi as usize] = coeffs[i];
                        }

                        // The parser's IDCT method, as on every other integer
                        // path (decode(), fast integer, the scanline strips).
                        let base_px = bx * DCT_SIZE;
                        let cols_to_copy = DCT_SIZE.min(info.comp_width.saturating_sub(base_px));
                        idct_block_i16_into_plane(
                            &natural_coeffs,
                            quant,
                            self.idct_method,
                            comp_plane_f32,
                            info.comp_width,
                            base_px,
                            base_py,
                            rows_to_copy,
                            cols_to_copy,
                        );
                    }
                }
            }
        }

        // Upsample chroma and return planes
        let mut planes_f32 = self.upsample_planes_f32(
            &mut comp_planes_f32,
            &comp_infos,
            max_h_samp,
            max_v_samp,
            chroma_upsampling,
        )?;

        let (mut y, mut cb, mut cr) = (
            core::mem::take(&mut planes_f32[0]),
            core::mem::take(&mut planes_f32[1]),
            core::mem::take(&mut planes_f32[2]),
        );
        // An RGB frame's planes are R, G, B: convert them, rather than hand
        // them back labelled Y, Cb, Cr.
        if self.color_model == ColorModel::Rgb {
            for ((y, cb), cr) in y.iter_mut().zip(&mut cb).zip(&mut cr) {
                let (yy, cbb, crr) =
                    crate::color::planes::rgb_to_ycbcr_bt601(*y + 128.0, *cb + 128.0, *cr + 128.0);
                (*y, *cb, *cr) = (yy - 128.0, cbb, crr);
            }
        }
        Ok((y, cb, cr))
    }
}
