//! Helpers for `JpegParser::decode_baseline_streaming`.
//!
//! Decomposes the streaming baseline decode loop into named phases:
//! - [`StreamingGeometry`]: MCU/strip dimensions and sampling ratios
//! - [`StreamingBuffers`]: Y/chroma/upsample strip allocations
//! - [`setup_entropy_decoder`]: configure entropy decoder with Huffman tables
//! - [`select_upsample_fn`]: pick chroma upsample kernel
//! - [`decode_mcu_row`]: entropy + dequant + IDCT for one MCU row
//! - [`output_mcu_row`]: YCbCr → RGB/BGRA conversion for one MCU row
//!
//! Each helper is `#[inline]` so the per-MCU and per-block hot paths
//! remain inlined into the parent loop with no added call overhead.

use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::{checked_size_2d, try_alloc_maybeuninit_pref};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;
use crate::quant::dequantize_unzigzag_i32_into_partial;

use super::super::idct_int::{idct_int_dc_only, idct_int_tiered};
use super::super::{ChromaUpsampling, DecodeWarning, IdctMethod};
use super::JpegParser;
use crate::color::{ycbcr_planes_i16_to_rgb_u8, ycbcr_planes_i16_to_xrgba_u8};
use crate::types::PixelFormat;

/// MCU and strip geometry derived from sampling factors and image dimensions.
pub(super) struct StreamingGeometry {
    pub width: usize,
    pub height: usize,
    pub is_grayscale: bool,
    pub max_h_samp: usize,
    pub max_v_samp: usize,
    pub mcu_cols: usize,
    pub mcu_rows: usize,
    pub y_strip_width: usize,
    pub y_strip_height: usize,
    pub c_h_samp: usize,
    #[allow(dead_code)]
    pub c_v_samp: usize,
    pub c_strip_width: usize,
    pub c_strip_height: usize,
    pub h_ratio: usize,
    pub v_ratio: usize,
}

impl StreamingGeometry {
    /// Compute geometry from parser fields. `width`/`height`/`num_components`
    /// and `components` come from the parser; the geometry never mutates them.
    pub(super) fn from_parser(parser: &JpegParser<'_>) -> Self {
        let width = parser.width as usize;
        let height = parser.height as usize;
        let is_grayscale = parser.num_components == 1;

        let max_h_samp = if is_grayscale {
            1usize
        } else {
            parser.components[..3]
                .iter()
                .map(|c| c.h_samp_factor as usize)
                .max()
                .unwrap_or(1)
        };
        let max_v_samp = if is_grayscale {
            1usize
        } else {
            parser.components[..3]
                .iter()
                .map(|c| c.v_samp_factor as usize)
                .max()
                .unwrap_or(1)
        };

        let mcu_width = max_h_samp * 8;
        let mcu_height = max_v_samp * 8;
        let mcu_cols = (width + mcu_width - 1) / mcu_width;
        let mcu_rows = (height + mcu_height - 1) / mcu_height;

        let y_strip_width = mcu_cols * max_h_samp * 8;
        let y_strip_height = max_v_samp * 8;

        let (c_h_samp, c_v_samp, c_strip_width, c_strip_height) = if is_grayscale {
            (1, 1, 0, 0)
        } else {
            let c_h = parser.components[1].h_samp_factor as usize;
            let c_v = parser.components[1].v_samp_factor as usize;
            (c_h, c_v, mcu_cols * c_h * 8, c_v * 8)
        };

        let h_ratio = if is_grayscale {
            1
        } else {
            max_h_samp / c_h_samp
        };
        let v_ratio = if is_grayscale {
            1
        } else {
            max_v_samp / c_v_samp
        };

        Self {
            width,
            height,
            is_grayscale,
            max_h_samp,
            max_v_samp,
            mcu_cols,
            mcu_rows,
            y_strip_width,
            y_strip_height,
            c_h_samp,
            c_v_samp,
            c_strip_width,
            c_strip_height,
            h_ratio,
            v_ratio,
        }
    }
}

/// Y, chroma and upsample scratch buffers for streaming decode.
pub(super) struct StreamingBuffers {
    pub y_strip_a: Vec<i16>,
    pub y_strip_b: Vec<i16>,
    pub cb_a: Vec<i16>,
    pub cr_a: Vec<i16>,
    pub cb_b: Vec<i16>,
    pub cr_b: Vec<i16>,
    pub cb_up: Vec<i16>,
    pub cr_up: Vec<i16>,
    /// Whether fancy h2v2 (double-buffered + 1-row lag) is active.
    pub need_fancy: bool,
    /// Extended chroma row count (data + 1 above + 1 below for fancy, else data).
    pub ext_height: usize,
    /// Upsampled chroma row count.
    pub upsample_out_height: usize,
    /// `out_bpp` — bytes per output pixel (1, 3, or 4).
    pub out_bpp: usize,
    /// Whether output is 4bpp (BGRA/RGBA).
    pub out_4bpp: bool,
    /// Whether to swap R/B channels (true = BGRA, false = RGBA).
    pub swap_rb: bool,
    /// Final RGB/BGRA/gray output buffer.
    pub rgb: Vec<u8>,
}

/// Output pixel layout the streaming path will produce:
/// `(bytes_per_pixel, is_4bpp, swap_rb)`.
///
/// Shared with the caller so it can charge the output buffer against the
/// decode memory budget before [`StreamingBuffers::allocate`] allocates it.
pub(super) fn output_pixel_layout(
    geom: &StreamingGeometry,
    streaming_output_format: Option<PixelFormat>,
) -> (usize, bool, bool) {
    if geom.is_grayscale {
        (1, false, false)
    } else {
        match streaming_output_format {
            Some(PixelFormat::Bgra | PixelFormat::Bgrx) => (4, true, true),
            Some(PixelFormat::Rgba) => (4, true, false),
            _ => (3, false, false),
        }
    }
}

impl StreamingBuffers {
    /// Allocate every buffer needed for streaming decode.
    ///
    /// `alloc_pref` is the per-site fallibility preference: the per-MCU-row
    /// strip / upsample scratch defaults infallible (bounded by the row width),
    /// the full-image output buffer defaults fallible.
    pub(super) fn allocate(
        geom: &StreamingGeometry,
        chroma_upsampling: ChromaUpsampling,
        streaming_output_format: Option<PixelFormat>,
        alloc_pref: zencodec::AllocPreference,
    ) -> Result<Self> {
        // Fancy h2v2 needs double-buffered Y and chroma strips (1-row lag for context)
        let need_fancy = !geom.is_grayscale
            && geom.v_ratio == 2
            && !matches!(chroma_upsampling, ChromaUpsampling::NearestNeighbor);

        let y_strip_size = geom.y_strip_width * geom.y_strip_height;
        let y_strip_a: Vec<i16> =
            try_alloc_maybeuninit_pref(alloc_pref, false, y_strip_size, "Y strip A")?;
        let y_strip_b: Vec<i16> = if need_fancy {
            try_alloc_maybeuninit_pref(alloc_pref, false, y_strip_size, "Y strip B")?
        } else {
            Vec::new()
        };

        // Extended chroma buffers (data + 1 above + 1 below context row for fancy)
        let ext_height = if need_fancy {
            geom.c_strip_height + 2
        } else {
            geom.c_strip_height
        };
        let c_buf_size = if geom.is_grayscale {
            0
        } else {
            geom.c_strip_width * ext_height
        };

        let cb_a: Vec<i16> = if c_buf_size > 0 {
            try_alloc_maybeuninit_pref(alloc_pref, false, c_buf_size, "Cb strip A")?
        } else {
            Vec::new()
        };
        let cr_a: Vec<i16> = if c_buf_size > 0 {
            try_alloc_maybeuninit_pref(alloc_pref, false, c_buf_size, "Cr strip A")?
        } else {
            Vec::new()
        };
        let cb_b: Vec<i16> = if need_fancy && c_buf_size > 0 {
            try_alloc_maybeuninit_pref(alloc_pref, false, c_buf_size, "Cb strip B")?
        } else {
            Vec::new()
        };
        let cr_b: Vec<i16> = if need_fancy && c_buf_size > 0 {
            try_alloc_maybeuninit_pref(alloc_pref, false, c_buf_size, "Cr strip B")?
        } else {
            Vec::new()
        };

        // Upsampled chroma output buffers
        // For fancy h2v2: output is ext_height * v_ratio rows (includes context row output)
        let upsample_out_height = if need_fancy {
            ext_height * geom.v_ratio
        } else {
            geom.y_strip_height
        };
        let up_size = if (!geom.is_grayscale && geom.h_ratio == 2) || need_fancy {
            geom.y_strip_width * upsample_out_height
        } else {
            0
        };
        let cb_up: Vec<i16> = if up_size > 0 {
            try_alloc_maybeuninit_pref(alloc_pref, false, up_size, "Cb upsampled")?
        } else {
            Vec::new()
        };
        let cr_up: Vec<i16> = if up_size > 0 {
            try_alloc_maybeuninit_pref(alloc_pref, false, up_size, "Cr upsampled")?
        } else {
            Vec::new()
        };

        // Determine output pixel format: 4bpp direct BGRA/RGBA when hinted,
        // otherwise 3bpp RGB (grayscale always 1bpp).
        let (out_bpp, out_4bpp, swap_rb) = output_pixel_layout(geom, streaming_output_format);

        // Full-image output buffer sized from the (untrusted) SOF dimensions →
        // default fallible.
        let rgb_size =
            checked_size_2d(geom.width, geom.height).and_then(|s| checked_size_2d(s, out_bpp))?;
        let rgb: Vec<u8> = try_alloc_maybeuninit_pref(alloc_pref, true, rgb_size, "output buffer")?;

        Ok(Self {
            y_strip_a,
            y_strip_b,
            cb_a,
            cr_a,
            cb_b,
            cr_b,
            cb_up,
            cr_up,
            need_fancy,
            ext_height,
            upsample_out_height,
            out_bpp,
            out_4bpp,
            swap_rb,
            rgb,
        })
    }
}

/// Build and configure an `EntropyDecoder` for the scan: lenient/RST flags
/// from strictness, plus DC/AC table mapping per component.
///
/// The returned decoder borrows from `parser.data` AND from
/// `parser.dc_tables`/`parser.ac_tables`, so it ties to the parser's
/// borrow lifetime `'p` (a sub-lifetime of `'a`). Drop the decoder
/// before re-borrowing the parser mutably.
pub(super) fn setup_entropy_decoder<'p, 'a: 'p>(
    parser: &'p JpegParser<'a>,
    scan_components: &[(usize, u8, u8)],
) -> EntropyDecoder<'p, 'p> {
    let scan_data = &parser.data[parser.position..];
    let mut decoder = EntropyDecoder::new(scan_data);
    if parser.strictness.lenient_entropy_recovery() {
        decoder.set_lenient(true);
    }
    // Enable RST resync for all non-Strict modes. Zero overhead on valid
    // input (only gates error-path recovery). On mismatch, resync_to_restart()
    // scans forward for the next RST marker and continues decoding.
    if parser.strictness.recovers_data_errors() {
        decoder.set_permissive_rst(true);
    }
    for (_comp_idx, dc_table, ac_table) in scan_components {
        let dc_idx = (*dc_table as usize).min(MAX_HUFFMAN_TABLES - 1);
        let ac_idx = (*ac_table as usize).min(MAX_HUFFMAN_TABLES - 1);
        let dc_table_ref: &HuffmanDecodeTable = match &parser.dc_tables[dc_idx] {
            Some(table) => table,
            None => {
                if dc_idx == 0 {
                    HuffmanDecodeTable::std_dc_luminance()
                } else {
                    HuffmanDecodeTable::std_dc_chrominance()
                }
            }
        };
        decoder.set_dc_table(dc_idx, dc_table_ref);
        let ac_table_ref: &HuffmanDecodeTable = match &parser.ac_tables[ac_idx] {
            Some(table) => table,
            None => {
                if ac_idx == 0 {
                    HuffmanDecodeTable::std_ac_luminance()
                } else {
                    HuffmanDecodeTable::std_ac_chrominance()
                }
            }
        };
        decoder.set_ac_table(ac_idx, ac_table_ref);
    }
    decoder
}

/// Type alias for the chroma upsample function pointer.
pub(super) type UpsampleFn = fn(&[i16], usize, usize, &mut [i16], usize, usize);

/// Select the chroma upsample function based on subsampling ratios and
/// upsampling policy. Returns `None` for paths that don't need a separate
/// upsample (grayscale, 4:4:4, or fused h2v2+NearestNeighbor box kernel).
#[inline]
pub(super) fn select_upsample_fn(
    geom: &StreamingGeometry,
    chroma_upsampling: ChromaUpsampling,
    idct_method: IdctMethod,
    use_fused_box: bool,
) -> Result<Option<UpsampleFn>> {
    if geom.is_grayscale || (geom.h_ratio == 1 && geom.v_ratio == 1) || use_fused_box {
        return Ok(None);
    }
    use crate::decode::upsample::{
        upsample_h2v1_i16_libjpeg, upsample_h2v1_i16_nearest, upsample_h2v2_i16_libjpeg,
        upsample_h2v2_i16_libjpeg_turbo,
    };
    Ok(Some(match (geom.h_ratio, geom.v_ratio) {
        (2, 2) => match chroma_upsampling {
            // IdctMethod::Libjpeg => turbo's fixed rounding bias (H2v2Bias).
            ChromaUpsampling::Triangle => match idct_method {
                IdctMethod::Libjpeg => upsample_h2v2_i16_libjpeg_turbo,
                _ => upsample_h2v2_i16_libjpeg,
            },
            ChromaUpsampling::NearestNeighbor => {
                unreachable!()
            }
        },
        (2, 1) => match chroma_upsampling {
            ChromaUpsampling::Triangle => upsample_h2v1_i16_libjpeg,
            ChromaUpsampling::NearestNeighbor => upsample_h2v1_i16_nearest,
        },
        _ => {
            return Err(Error::unsupported_feature(
                "unsupported chroma subsampling for streaming decode",
            ));
        }
    }))
}

/// Select the IDCT function pointer based on method config.
#[inline]
pub(super) fn select_idct_fn(method: IdctMethod) -> fn(&mut [i32; 64], &mut [i16], usize, u8) {
    match method {
        IdctMethod::Libjpeg => super::super::idct_int::idct_int_tiered_libjpeg,
        IdctMethod::Jpegli => idct_int_tiered,
    }
}

/// Per-MCU-row decode state mutated across the loop body.
pub(super) struct McuRowState {
    pub mcu_count: u32,
    pub next_restart_num: u8,
    pub prev_coeff_counts: [u8; 4],
    pub streaming_truncation_mcu: Option<u32>,
    pub coeffs: [i16; DCT_BLOCK_SIZE],
    pub dequant_buf: [i32; DCT_BLOCK_SIZE],
}

impl McuRowState {
    pub(super) fn new() -> Self {
        Self {
            mcu_count: 0,
            next_restart_num: 0,
            prev_coeff_counts: [64; 4],
            streaming_truncation_mcu: None,
            coeffs: [0; DCT_BLOCK_SIZE],
            dequant_buf: [0; DCT_BLOCK_SIZE],
        }
    }
}

/// Decode one MCU row of blocks into Y strip and chroma strips.
/// `c_data_offset` is the byte offset into chroma buffers for data (skips
/// context row).
///
/// Hot path: `#[inline(always)]` ensures the per-block inner loop body
/// inlines into the parent function so the IDCT/dequant fast paths stay
/// tight. The closure call to `idct_fn` is a single indirect call per
/// block (same as the prior nested-fn version).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn decode_mcu_row(
    decoder: &mut EntropyDecoder<'_, '_>,
    scan_components: &[(usize, u8, u8)],
    components: &[crate::types::Component; crate::foundation::consts::MAX_COMPONENTS],
    mcu_cols: usize,
    state: &mut McuRowState,
    restart_interval: u32,
    quant_tables: &[&[u16; DCT_BLOCK_SIZE]],
    y_strip: &mut [i16],
    y_strip_width: usize,
    max_h_samp: usize,
    cb: &mut [i16],
    cr: &mut [i16],
    c_strip_width: usize,
    c_h_samp: usize,
    c_data_offset: usize,
    is_grayscale: bool,
    idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
) -> Result<()> {
    for mcu_x in 0..mcu_cols {
        if restart_interval > 0 && state.mcu_count > 0 && state.mcu_count % restart_interval == 0 {
            decoder.align_to_byte();
            if !decoder.read_restart_marker_tolerant(state.next_restart_num)?
                && state.streaming_truncation_mcu.is_none()
            {
                state.streaming_truncation_mcu = Some(state.mcu_count);
            }
            state.next_restart_num = (state.next_restart_num + 1) & 7;
            decoder.reset_dc();
            state.prev_coeff_counts = [64; 4];
        }

        for (comp_idx, dc_table, ac_table) in scan_components {
            let h_samp = components[*comp_idx].h_samp_factor as usize;
            let v_samp = components[*comp_idx].v_samp_factor as usize;

            for by in 0..v_samp {
                for bx in 0..h_samp {
                    let coeff_count = match decoder.decode_block_into(
                        &mut state.coeffs,
                        state.prev_coeff_counts[*comp_idx],
                        *comp_idx,
                        *dc_table as usize,
                        *ac_table as usize,
                    )? {
                        ScanRead::Value(c) => c,
                        ScanRead::EndOfScan | ScanRead::Truncated => {
                            // Fall through with a zero block so the strip gets the
                            // documented zero fill — `continue` left whatever the
                            // previous MCU row had put there in the output (#92).
                            if state.streaming_truncation_mcu.is_none() {
                                state.streaming_truncation_mcu = Some(state.mcu_count);
                            }
                            state.prev_coeff_counts[*comp_idx] = 64;
                            state.coeffs = [0i16; DCT_BLOCK_SIZE];
                            1
                        }
                    };
                    state.prev_coeff_counts[*comp_idx] =
                        state.prev_coeff_counts[*comp_idx].max(coeff_count);

                    let quant = quant_tables[*comp_idx];

                    if *comp_idx == 0 || is_grayscale {
                        let dst_x = mcu_x * max_h_samp * 8 + bx * 8;
                        let dst_y = by * 8;
                        let dst_offset = dst_y * y_strip_width + dst_x;
                        if coeff_count <= 1 {
                            let dc = state.coeffs[0] as i32 * quant[0] as i32;
                            idct_int_dc_only(dc, &mut y_strip[dst_offset..], y_strip_width);
                        } else {
                            dequantize_unzigzag_i32_into_partial(
                                &state.coeffs,
                                quant,
                                &mut state.dequant_buf,
                                coeff_count,
                            );
                            idct_fn(
                                &mut state.dequant_buf,
                                &mut y_strip[dst_offset..],
                                y_strip_width,
                                coeff_count,
                            );
                        }
                    } else {
                        let dst_x = mcu_x * c_h_samp * 8 + bx * 8;
                        let dst_y = by * 8;
                        let dst_offset = c_data_offset + dst_y * c_strip_width + dst_x;
                        let strip = if *comp_idx == 1 { &mut *cb } else { &mut *cr };
                        if coeff_count <= 1 {
                            let dc = state.coeffs[0] as i32 * quant[0] as i32;
                            idct_int_dc_only(dc, &mut strip[dst_offset..], c_strip_width);
                        } else {
                            dequantize_unzigzag_i32_into_partial(
                                &state.coeffs,
                                quant,
                                &mut state.dequant_buf,
                                coeff_count,
                            );
                            idct_fn(
                                &mut state.dequant_buf,
                                &mut strip[dst_offset..],
                                c_strip_width,
                                coeff_count,
                            );
                        }
                    }
                }
            }
        }
        state.mcu_count += 1;
    }
    Ok(())
}

/// Output one MCU row to the output buffer using upsampled chroma.
///
/// `chroma_row_skip` is the number of upsampled rows to skip at the start
/// of the chroma buffer (to skip context row output in fancy mode).
/// When `out_4bpp` is true, writes BGRA/RGBA (4 bytes/pixel) with alpha=255;
/// `swap_rb` controls B,G,R,A vs R,G,B,A order.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn output_mcu_row(
    mcu_y: usize,
    y_strip: &[i16],
    y_strip_width: usize,
    y_strip_height: usize,
    cb_up: &[i16],
    cr_up: &[i16],
    chroma_row_skip: usize,
    rgb: &mut [u8],
    width: usize,
    height: usize,
    is_rgb: bool,
    out_4bpp: bool,
    swap_rb: bool,
    turbo_color: bool,
) {
    let bpp = if out_4bpp { 4 } else { 3 };
    let y_start = mcu_y * y_strip_height;
    let rows = y_strip_height.min(height.saturating_sub(y_start));
    let cols = width.min(y_strip_width);
    for row in 0..rows {
        let y_off = row * y_strip_width;
        let up_off = (chroma_row_skip + row) * y_strip_width;
        let rgb_off = (y_start + row) * width * bpp;
        if is_rgb {
            // RGB JPEG: interleave planes without YCbCr→RGB matrix
            if out_4bpp {
                for px in 0..cols {
                    let o = rgb_off + px * 4;
                    if swap_rb {
                        rgb[o] = cr_up[up_off + px].clamp(0, 255) as u8;
                        rgb[o + 1] = cb_up[up_off + px].clamp(0, 255) as u8;
                        rgb[o + 2] = y_strip[y_off + px].clamp(0, 255) as u8;
                    } else {
                        rgb[o] = y_strip[y_off + px].clamp(0, 255) as u8;
                        rgb[o + 1] = cb_up[up_off + px].clamp(0, 255) as u8;
                        rgb[o + 2] = cr_up[up_off + px].clamp(0, 255) as u8;
                    }
                    rgb[o + 3] = 255;
                }
            } else {
                for px in 0..cols {
                    rgb[rgb_off + px * 3] = y_strip[y_off + px].clamp(0, 255) as u8;
                    rgb[rgb_off + px * 3 + 1] = cb_up[up_off + px].clamp(0, 255) as u8;
                    rgb[rgb_off + px * 3 + 2] = cr_up[up_off + px].clamp(0, 255) as u8;
                }
            }
        } else if out_4bpp {
            ycbcr_planes_i16_to_xrgba_u8(
                &y_strip[y_off..y_off + cols],
                &cb_up[up_off..up_off + cols],
                &cr_up[up_off..up_off + cols],
                &mut rgb[rgb_off..rgb_off + cols * 4],
                swap_rb,
                turbo_color,
            );
        } else {
            ycbcr_planes_i16_to_rgb_u8(
                &y_strip[y_off..y_off + cols],
                &cb_up[up_off..up_off + cols],
                &cr_up[up_off..up_off + cols],
                &mut rgb[rgb_off..rgb_off + cols * 3],
                turbo_color,
            );
        }
    }
}

use super::output_helpers::edge_replicate_h_padding as fixup_h_padding;

/// Inputs to the fancy h2v2 + simple decode loops that don't change
/// across MCU rows. Bundled to keep helper signatures readable.
pub(super) struct LoopInputs<'a, 'b> {
    pub scan_components: &'a [(usize, u8, u8)],
    pub components: &'a [crate::types::Component; crate::foundation::consts::MAX_COMPONENTS],
    pub quant_tables: &'a [&'b [u16; DCT_BLOCK_SIZE]],
    pub restart_interval: u32,
    pub idct_fn: fn(&mut [i32; 64], &mut [i16], usize, u8),
    pub is_rgb: bool,
    /// libjpeg-turbo-exact YCbCr→RGB (IdctMethod::Libjpeg).
    pub turbo_color: bool,
}

/// Fancy h2v2 decode loop: double-buffered Y + chroma strips with 1-row lag
/// so each MCU row's chroma has correct above/below context for the
/// triangle filter.
///
/// Pattern:
///   MCU 0: decode → B, set above ctx = edge repl, swap (B→A)
///   MCU N: decode → B, set A.below = B.first, output A, set B.above = A.last, swap
///   Flush: set A.below = edge repl, output A
#[allow(clippy::too_many_arguments)]
pub(super) fn run_fancy_h2v2_loop(
    decoder: &mut EntropyDecoder<'_, '_>,
    inputs: &LoopInputs<'_, '_>,
    geom: &StreamingGeometry,
    bufs: &mut StreamingBuffers,
    state: &mut McuRowState,
    upsample: UpsampleFn,
    c_data_offset: usize,
    downsampled_w: usize,
    stop: &impl enough::Stop,
) -> Result<()> {
    let mcu_rows = geom.mcu_rows;
    let mcu_cols = geom.mcu_cols;
    let c_strip_width = geom.c_strip_width;
    let c_strip_height = geom.c_strip_height;
    let y_strip_width = geom.y_strip_width;
    let y_strip_height = geom.y_strip_height;
    let max_h_samp = geom.max_h_samp;
    let c_h_samp = geom.c_h_samp;
    let v_ratio = geom.v_ratio;
    let ext_height = bufs.ext_height;
    let upsample_out_height = bufs.upsample_out_height;

    for mcu_y in 0..mcu_rows {
        if stop.should_stop() {
            return Err(Error::cancelled());
        }

        // Decode into B buffers (y_strip_b, cb_b, cr_b)
        decode_mcu_row(
            decoder,
            inputs.scan_components,
            inputs.components,
            mcu_cols,
            state,
            inputs.restart_interval,
            inputs.quant_tables,
            &mut bufs.y_strip_b,
            y_strip_width,
            max_h_samp,
            &mut bufs.cb_b,
            &mut bufs.cr_b,
            c_strip_width,
            c_h_samp,
            c_data_offset,
            false,
            inputs.idct_fn,
        )?;

        if mcu_y == 0 {
            // First row: set above-context = edge replicate
            // (copy first data row to row 0)
            bufs.cb_b.copy_within(c_strip_width..2 * c_strip_width, 0);
            bufs.cr_b.copy_within(c_strip_width..2 * c_strip_width, 0);
        } else {
            // Set A's below-context = B's first data row
            let below_start = (c_strip_height + 1) * c_strip_width;
            bufs.cb_a[below_start..below_start + c_strip_width]
                .copy_from_slice(&bufs.cb_b[c_strip_width..2 * c_strip_width]);
            bufs.cr_a[below_start..below_start + c_strip_width]
                .copy_from_slice(&bufs.cr_b[c_strip_width..2 * c_strip_width]);

            // Output pending MCU row (A has full context now)
            fixup_h_padding(&mut bufs.cb_a, downsampled_w, c_strip_width, ext_height);
            fixup_h_padding(&mut bufs.cr_a, downsampled_w, c_strip_width, ext_height);
            upsample(
                &bufs.cb_a,
                c_strip_width,
                ext_height,
                &mut bufs.cb_up,
                y_strip_width,
                upsample_out_height,
            );
            upsample(
                &bufs.cr_a,
                c_strip_width,
                ext_height,
                &mut bufs.cr_up,
                y_strip_width,
                upsample_out_height,
            );
            output_mcu_row(
                mcu_y - 1,
                &bufs.y_strip_a,
                y_strip_width,
                y_strip_height,
                &bufs.cb_up,
                &bufs.cr_up,
                v_ratio, // skip context rows in upsampled output
                &mut bufs.rgb,
                geom.width,
                geom.height,
                inputs.is_rgb,
                bufs.out_4bpp,
                bufs.swap_rb,
                inputs.turbo_color,
            );

            // Set B's above-context = A's last data row
            let last_data = c_strip_height * c_strip_width;
            bufs.cb_b[..c_strip_width]
                .copy_from_slice(&bufs.cb_a[last_data..last_data + c_strip_width]);
            bufs.cr_b[..c_strip_width]
                .copy_from_slice(&bufs.cr_a[last_data..last_data + c_strip_width]);
        }

        // Swap: B (freshly decoded) → A (pending output)
        core::mem::swap(&mut bufs.y_strip_a, &mut bufs.y_strip_b);
        core::mem::swap(&mut bufs.cb_a, &mut bufs.cb_b);
        core::mem::swap(&mut bufs.cr_a, &mut bufs.cr_b);
    }

    // Flush last pending MCU row (below context = edge replicate)
    if mcu_rows > 0 {
        // For the last MCU row, edge-replicate from the last REAL chroma row,
        // not the last padding row. The encoder pads MCU boundaries by
        // replicating pixel rows before DCT, but IDCT rounding means decoded
        // padding rows differ slightly from the last real row.
        // libjpeg-turbo's set_bottom_pointers() does this same truncation.
        let downsampled_h = (geom.height + v_ratio - 1) / v_ratio;
        let real_rows_in_strip =
            c_strip_height.min(downsampled_h.saturating_sub((mcu_rows - 1) * c_strip_height));
        if real_rows_in_strip < c_strip_height {
            // Edge-replicate last real row over padding rows
            // Data rows are at offset c_data_offset (1 row for fancy context)
            let last_real = c_data_offset + (real_rows_in_strip - 1) * c_strip_width;
            for pad_row in real_rows_in_strip..c_strip_height {
                let dst = c_data_offset + pad_row * c_strip_width;
                bufs.cb_a
                    .copy_within(last_real..last_real + c_strip_width, dst);
                bufs.cr_a
                    .copy_within(last_real..last_real + c_strip_width, dst);
            }
        }

        let last_data = c_strip_height * c_strip_width;
        let below_start = (c_strip_height + 1) * c_strip_width;
        // Now last_data points to the edge-replicated row (== last real row)
        bufs.cb_a
            .copy_within(last_data..last_data + c_strip_width, below_start);
        bufs.cr_a
            .copy_within(last_data..last_data + c_strip_width, below_start);

        fixup_h_padding(&mut bufs.cb_a, downsampled_w, c_strip_width, ext_height);
        fixup_h_padding(&mut bufs.cr_a, downsampled_w, c_strip_width, ext_height);
        upsample(
            &bufs.cb_a,
            c_strip_width,
            ext_height,
            &mut bufs.cb_up,
            y_strip_width,
            upsample_out_height,
        );
        upsample(
            &bufs.cr_a,
            c_strip_width,
            ext_height,
            &mut bufs.cr_up,
            y_strip_width,
            upsample_out_height,
        );
        output_mcu_row(
            mcu_rows - 1,
            &bufs.y_strip_a,
            y_strip_width,
            y_strip_height,
            &bufs.cb_up,
            &bufs.cr_up,
            v_ratio, // skip context rows in upsampled output
            &mut bufs.rgb,
            geom.width,
            geom.height,
            inputs.is_rgb,
            bufs.out_4bpp,
            bufs.swap_rb,
            inputs.turbo_color,
        );
    }
    Ok(())
}

/// Non-fancy paths: grayscale, box h2v2 (fused), h2v1.
/// No double-buffering needed (no vertical chroma context).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_simple_loop(
    decoder: &mut EntropyDecoder<'_, '_>,
    inputs: &LoopInputs<'_, '_>,
    geom: &StreamingGeometry,
    bufs: &mut StreamingBuffers,
    state: &mut McuRowState,
    chroma_upsampling: ChromaUpsampling,
    use_fused_box: bool,
    c_data_offset: usize,
    stop: &impl enough::Stop,
) -> Result<()> {
    use crate::color::ycbcr::fused_h2v2_box_ycbcr_to_rgb_u8;

    let mcu_rows = geom.mcu_rows;
    let mcu_cols = geom.mcu_cols;
    let c_strip_width = geom.c_strip_width;
    let c_strip_height = geom.c_strip_height;
    let y_strip_width = geom.y_strip_width;
    let y_strip_height = geom.y_strip_height;
    let max_h_samp = geom.max_h_samp;
    let c_h_samp = geom.c_h_samp;
    let width = geom.width;
    let height = geom.height;
    let is_grayscale = geom.is_grayscale;
    let out_bpp = bufs.out_bpp;
    let out_4bpp = bufs.out_4bpp;
    let swap_rb = bufs.swap_rb;
    let is_rgb = inputs.is_rgb;

    for mcu_y in 0..mcu_rows {
        if stop.should_stop() {
            return Err(Error::cancelled());
        }

        decode_mcu_row(
            decoder,
            inputs.scan_components,
            inputs.components,
            mcu_cols,
            state,
            inputs.restart_interval,
            inputs.quant_tables,
            &mut bufs.y_strip_a,
            y_strip_width,
            max_h_samp,
            &mut bufs.cb_a,
            &mut bufs.cr_a,
            c_strip_width,
            c_h_samp,
            c_data_offset,
            is_grayscale,
            inputs.idct_fn,
        )?;

        if is_grayscale {
            let y_start = mcu_y * y_strip_height;
            let rows = y_strip_height.min(height.saturating_sub(y_start));
            let cols = width.min(y_strip_width);
            if out_4bpp {
                // Grayscale → BGRA/RGBA: write [v, v, v, 0xFF] per pixel.
                // 16-pixel chunks for auto-vectorization, remainder scalar.
                for row in 0..rows {
                    let strip_off = row * y_strip_width;
                    let out_off = (y_start + row) * width * 4;
                    let src = &bufs.y_strip_a[strip_off..strip_off + cols];
                    let dst = &mut bufs.rgb[out_off..out_off + cols * 4];
                    let chunks = cols / 16;
                    for c in 0..chunks {
                        let si = c * 16;
                        let di = c * 64;
                        let s: &[i16; 16] = src[si..si + 16].try_into().unwrap();
                        let d: &mut [u8; 64] = (&mut dst[di..di + 64]).try_into().unwrap();
                        for i in 0..16 {
                            let v = s[i].clamp(0, 255) as u8;
                            d[i * 4] = v;
                            d[i * 4 + 1] = v;
                            d[i * 4 + 2] = v;
                            d[i * 4 + 3] = 255;
                        }
                    }
                    for px in chunks * 16..cols {
                        let v = src[px].clamp(0, 255) as u8;
                        let o = px * 4;
                        dst[o] = v;
                        dst[o + 1] = v;
                        dst[o + 2] = v;
                        dst[o + 3] = 255;
                    }
                }
            } else {
                // Grayscale → 1bpp gray output
                for row in 0..rows {
                    let strip_off = row * y_strip_width;
                    let out_off = (y_start + row) * width;
                    for px in 0..cols {
                        bufs.rgb[out_off + px] = bufs.y_strip_a[strip_off + px].clamp(0, 255) as u8;
                    }
                }
            }
        } else if use_fused_box {
            let y_start = mcu_y * y_strip_height;
            let y_rows = y_strip_height.min(height.saturating_sub(y_start));
            let c_rows = c_strip_height;
            let cols = width.min(y_strip_width);
            for row in 0..y_rows {
                let c_row = (row / 2).min(c_rows.saturating_sub(1));
                let y_off = row * y_strip_width;
                let c_off = c_row * c_strip_width;
                let rgb_off = (y_start + row) * width * out_bpp;
                if is_rgb && out_4bpp {
                    for px in 0..cols {
                        let cx = px / 2;
                        let o = rgb_off + px * 4;
                        if swap_rb {
                            bufs.rgb[o] = bufs.cr_a[c_off + cx].clamp(0, 255) as u8;
                            bufs.rgb[o + 1] = bufs.cb_a[c_off + cx].clamp(0, 255) as u8;
                            bufs.rgb[o + 2] = bufs.y_strip_a[y_off + px].clamp(0, 255) as u8;
                        } else {
                            bufs.rgb[o] = bufs.y_strip_a[y_off + px].clamp(0, 255) as u8;
                            bufs.rgb[o + 1] = bufs.cb_a[c_off + cx].clamp(0, 255) as u8;
                            bufs.rgb[o + 2] = bufs.cr_a[c_off + cx].clamp(0, 255) as u8;
                        }
                        bufs.rgb[o + 3] = 255;
                    }
                } else if is_rgb {
                    for px in 0..cols {
                        let cx = px / 2;
                        bufs.rgb[rgb_off + px * 3] = bufs.y_strip_a[y_off + px].clamp(0, 255) as u8;
                        bufs.rgb[rgb_off + px * 3 + 1] = bufs.cb_a[c_off + cx].clamp(0, 255) as u8;
                        bufs.rgb[rgb_off + px * 3 + 2] = bufs.cr_a[c_off + cx].clamp(0, 255) as u8;
                    }
                } else if out_4bpp {
                    // No fused h2v2 box→4bpp SIMD kernel yet.
                    // Expand chroma inline (box upsample: duplicate each
                    // sample to 2 pixels) then call the SIMD xrgba function.
                    // This is a cold path (NearestNeighbor + 4bpp).
                    let chroma_w = (cols + 1) / 2;
                    let needed = cols;
                    // Reuse cb_up/cr_up scratch buffers (already allocated
                    // with at least y_strip_width per row)
                    for px in 0..chroma_w {
                        let val_cb = bufs.cb_a[c_off + px];
                        let val_cr = bufs.cr_a[c_off + px];
                        let x0 = px * 2;
                        bufs.cb_up[x0] = val_cb;
                        bufs.cr_up[x0] = val_cr;
                        if x0 + 1 < needed {
                            bufs.cb_up[x0 + 1] = val_cb;
                            bufs.cr_up[x0 + 1] = val_cr;
                        }
                    }
                    ycbcr_planes_i16_to_xrgba_u8(
                        &bufs.y_strip_a[y_off..y_off + cols],
                        &bufs.cb_up[..cols],
                        &bufs.cr_up[..cols],
                        &mut bufs.rgb[rgb_off..rgb_off + cols * 4],
                        swap_rb,
                        inputs.turbo_color,
                    );
                } else {
                    fused_h2v2_box_ycbcr_to_rgb_u8(
                        &bufs.y_strip_a[y_off..y_off + cols],
                        &bufs.cb_a[c_off..c_off + (cols + 1) / 2],
                        &bufs.cr_a[c_off..c_off + (cols + 1) / 2],
                        &mut bufs.rgb[rgb_off..rgb_off + cols * 3],
                        cols,
                        inputs.turbo_color,
                    );
                }
            }
        } else {
            // h2v1 (4:2:2): horizontal chroma upsample, then color convert.
            //
            // Use the STRIDED kernel with the REAL chroma/luma widths (not the
            // MCU-padded strip widths) so the kernel's right-edge replication
            // lands on the last *real* chroma column — byte-identical to the
            // scanline strip path (`StripProcessor::upsample_h2v1`) and to
            // libjpeg-turbo's `h2v1_fancy_upsample`. Feeding the padded strip
            // widths (`c_strip_width`/`y_strip_width`) instead made the final
            // visible column interpolate against MCU-padding chroma, corrupting
            // the rightmost column by up to ~11/255 for even widths (#188).
            use crate::decode::upsample::{
                upsample_h2v1_i16_libjpeg_strided, upsample_h2v1_i16_nearest_strided,
            };
            let upsample = match chroma_upsampling {
                ChromaUpsampling::Triangle => upsample_h2v1_i16_libjpeg_strided,
                ChromaUpsampling::NearestNeighbor => upsample_h2v1_i16_nearest_strided,
            };
            let real_cw = (width + geom.h_ratio - 1) / geom.h_ratio;
            upsample(
                &bufs.cb_a[..c_strip_width * c_strip_height],
                real_cw,
                c_strip_width,
                c_strip_height,
                &mut bufs.cb_up[..y_strip_width * y_strip_height],
                width,
                y_strip_width,
                y_strip_height,
            );
            upsample(
                &bufs.cr_a[..c_strip_width * c_strip_height],
                real_cw,
                c_strip_width,
                c_strip_height,
                &mut bufs.cr_up[..y_strip_width * y_strip_height],
                width,
                y_strip_width,
                y_strip_height,
            );
            output_mcu_row(
                mcu_y,
                &bufs.y_strip_a,
                y_strip_width,
                y_strip_height,
                &bufs.cb_up,
                &bufs.cr_up,
                0, // no context row skip for non-fancy paths
                &mut bufs.rgb,
                width,
                height,
                is_rgb,
                out_4bpp,
                swap_rb,
                inputs.turbo_color,
            );
        }
    }
    Ok(())
}

/// Stats sampled from the entropy decoder before it's dropped, so the
/// parser can emit warnings without needing the decoder borrow.
pub(super) struct DecoderStats {
    pub had_ac_overflow: bool,
    pub had_invalid_huffman: bool,
    pub rst_resyncs: u32,
    pub position: usize,
}

impl DecoderStats {
    pub(super) fn snapshot(decoder: &EntropyDecoder<'_, '_>) -> Self {
        Self {
            had_ac_overflow: decoder.had_ac_overflow,
            had_invalid_huffman: decoder.had_invalid_huffman,
            rst_resyncs: decoder.rst_resync_count(),
            position: decoder.position(),
        }
    }
}

/// Emit decode warnings collected from the entropy decoder and advance the
/// parser position. Caller must have already dropped the decoder.
pub(super) fn finalize_streaming(
    parser: &mut JpegParser<'_>,
    stats: DecoderStats,
    state: &McuRowState,
    mcu_rows: usize,
    mcu_cols: usize,
) -> Result<()> {
    parser.position += stats.position;

    let total_mcus = (mcu_rows * mcu_cols) as u32;
    if let Some(at_mcu) = state.streaming_truncation_mcu {
        parser.warn(DecodeWarning::TruncatedScan {
            blocks_decoded: at_mcu,
            blocks_expected: total_mcus,
        })?;
    }
    if stats.had_ac_overflow {
        parser.warn(DecodeWarning::AcIndexOverflow)?;
    }
    if stats.had_invalid_huffman {
        parser.warn(DecodeWarning::InvalidHuffmanCode)?;
    }
    if stats.rst_resyncs > 0 {
        parser.warn(DecodeWarning::RestartMarkerResync {
            count: stats.rst_resyncs,
        })?;
    }
    Ok(())
}
