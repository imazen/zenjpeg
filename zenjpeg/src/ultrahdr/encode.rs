//! UltraHDR encoding workflow helpers.
//!
//! Provides high-level functions for encoding UltraHDR JPEGs from HDR
//! source images. Inputs are [`zenpixels::PixelBuffer`] (re-exported here
//! as [`super::UhdrRawImage`] for back-compat).

use crate::container::xmp::{create_xmp_app1_marker, generate_gainmap_xmp, generate_primary_xmp};
use crate::encode::extras::EncoderSegments;
use crate::encoder::{EncoderConfig, PixelLayout};
use crate::error::{Error, Result};
use enough::{Stop, Unstoppable};
use ultrahdr_core::{
    ColorPrimaries, GainMap, GainMapChannel, GainMapEncodingFormat, GainMapMetadata,
    LumaGainMapSplitter, LumaToneMap, PixelBuffer, PixelFormat, SplitConfig, SplitStats,
    TransferFunction,
    color::tonemap::{AdaptiveTonemapper, ToneMapConfig, tonemap_image_to_srgb8},
    gainmap::{GainMapConfig, RowEncoder, compute_gain_row, compute_gainmap},
    pixel_buffer_from_vec,
};
use zencodec::Iso21496Format;
use zencodec::gainmap::{ISO_21496_1_PRIMARY_APP2_BODY, serialize_iso21496_fmt};
use zentone::Bt2446C;

/// Encode an HDR image as UltraHDR JPEG.
///
/// Performs the full UltraHDR encoding workflow:
/// 1. Tonemap HDR to SDR via `ultrahdr_core::color::tonemap::tonemap_image_to_srgb8`
/// 2. Compute gain map from HDR/SDR pair
/// 3. Encode SDR base image with jpegli
/// 4. Encode gain map as grayscale JPEG
/// 5. Generate XMP metadata
/// 6. Assemble final UltraHDR JPEG with MPF structure
///
/// `tonemap_config` is currently a no-op placeholder — `tonemap_image_to_srgb8`
/// uses `ToneMapConfig::default()` internally. Issue #71 will replace this
/// signature with one that takes `&dyn LumaToneMap` for full curve choice.
pub fn encode_ultrahdr(
    hdr: &PixelBuffer,
    gainmap_config: &GainMapConfig,
    _tonemap_config: &ToneMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let sdr = tonemap_to_pixel_buffer(hdr, ColorPrimaries::Bt709)?;
    stop.check()?;

    let (gainmap, metadata) =
        compute_gainmap(hdr, &sdr, gainmap_config, &stop).map_err(ultrahdr_to_zenjpeg_error)?;
    stop.check()?;

    encode_with_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        encoder_config,
        gainmap_quality,
        stop,
    )
}

/// Encode UltraHDR using a pre-learned adaptive tonemapper.
///
/// Use this when re-encoding edited HDR content to preserve the original
/// HDR→SDR relationship. The adaptive tonemapper learns the curve from an
/// existing pair and reproduces it for modified content.
pub fn encode_ultrahdr_with_tonemapper(
    hdr: &PixelBuffer,
    tonemapper: &AdaptiveTonemapper,
    gainmap_config: &GainMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let sdr = tonemapper.apply(hdr).map_err(ultrahdr_to_zenjpeg_error)?;
    stop.check()?;

    let (gainmap, metadata) =
        compute_gainmap(hdr, &sdr, gainmap_config, &stop).map_err(ultrahdr_to_zenjpeg_error)?;
    stop.check()?;

    encode_with_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        encoder_config,
        gainmap_quality,
        stop,
    )
}

/// Create a streaming gain map computer for row-by-row processing.
///
/// More memory-efficient than the full-image [`compute_gainmap`] for large
/// images. Inputs are linear f32 RGB for both HDR and SDR.
pub fn create_gainmap_computer(
    width: u32,
    height: u32,
    config: &GainMapConfig,
    hdr_gamut: ColorPrimaries,
) -> Result<RowEncoder> {
    RowEncoder::new(
        width,
        height,
        config.clone(),
        hdr_gamut,
        ColorPrimaries::Bt709,
    )
    .map_err(ultrahdr_to_zenjpeg_error)
}

/// Encode an HDR image as Ultra HDR JPEG using a caller-supplied tone curve.
///
/// Closes #71's "wire `LumaToneMap` into encode_ultrahdr". Derives the SDR
/// base via zentone's [`LumaGainMapSplitter`] (luma-only,
/// chromaticity-preserving) around the supplied curve, then computes the
/// gain map from the resulting (HDR, SDR) pair via
/// [`compute_gainmap`]. Accepts any `LumaToneMap` from zentone
/// (`Bt2446A` / `Bt2446B` / `Bt2446C`, `Bt2408Yrgb`, `CompiledFilmicSpline`,
/// `HableFilmic`, …).
///
/// `gainmap_config.multi_channel` MUST be `false` (the splitter is
/// single-channel by design). Multi-channel callers use
/// [`encode_ultrahdr`] with the tone-mapped SDR baked into the
/// `ToneMapConfig` path instead.
pub fn encode_ultrahdr_with_curve<C: LumaToneMap>(
    hdr: &PixelBuffer,
    curve: &C,
    gainmap_config: &GainMapConfig,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    if gainmap_config.multi_channel {
        return Err(Error::unsupported_feature(
            "encode_ultrahdr_with_curve does not support multi-channel gain maps; \
             use encode_ultrahdr with separate HDR/SDR images instead",
        ));
    }

    // Single-pass: walk HDR rows once, splitting into SDR and quantizing
    // gain-map bytes inline via ultrahdr-core's `compute_gain_row`. Saves a
    // full second pixel walk + materialization vs the previous two-pass flow.
    let (sdr_linear, gainmap, metadata) =
        split_and_compute_gainmap(hdr, curve, gainmap_config, &stop)?;
    stop.check()?;

    // The splitter emits SDR as RgbaF32 / Linear. Encode wants Rgba8 / Srgb.
    let sdr_rgba8 = linear_f32_to_srgb_rgba8(&sdr_linear)?;
    stop.check()?;

    encode_with_gainmap(
        &sdr_rgba8,
        &gainmap,
        &metadata,
        encoder_config,
        gainmap_quality,
        stop,
    )
}

/// Single-pass fused splitter + gain-map quantization.
///
/// Walks each HDR row once: zentone's [`LumaGainMapSplitter`] writes the SDR
/// output, and on the subset of rows that map to a gain-map row,
/// `compute_gain_row` from ultrahdr-core quantizes gain-map bytes from the
/// (HDR, SDR) row pair. Replaces the prior two-pass flow that ran
/// `split_hdr_to_sdr_via_curve` followed by `compute_gainmap` — about 2× the
/// memory bandwidth on encode at typical 4K resolutions.
///
/// Sampling matches `compute_gainmap_slice`: each gain-map cell `(gx, gy)`
/// reads the source pixel at `(gx*scale + scale/2, gy*scale + scale/2)`,
/// clamped to image bounds. `compute_gain_row`'s `observed_min_max`
/// accumulator is tracked across the same pixels, but the declared metadata
/// range is the CONFIG grid the bytes were quantized on — the accumulator
/// only widens the alternate headroom (see [`build_gainmap_metadata`]).
fn split_and_compute_gainmap<C: LumaToneMap>(
    hdr: &PixelBuffer,
    curve: &C,
    gainmap_config: &GainMapConfig,
    stop: &impl Stop,
) -> Result<(PixelBuffer, GainMap, GainMapMetadata)> {
    let width = hdr.width();
    let height = hdr.height();
    let hdr_gamut = hdr.descriptor().primaries;

    let split_cfg = SplitConfig {
        luma_weights: luma_weights_for(hdr_gamut),
        base_offset: gainmap_config.base_offset,
        alternate_offset: gainmap_config.alternate_offset,
        min_log2: gainmap_config.min_boost.log2(),
        max_log2: gainmap_config.max_boost.log2(),
        ..SplitConfig::default()
    };

    let splitter = LumaGainMapSplitter::new(curve, split_cfg);

    let mut sdr_buffer = ultrahdr_core::new_pixel_buffer(
        width,
        height,
        PixelFormat::RgbaF32,
        hdr_gamut,
        TransferFunction::Linear,
    )
    .map_err(ultrahdr_to_zenjpeg_error)?;

    let w = width as usize;
    let scale = gainmap_config.scale_factor.max(1) as u32;
    let gm_width = width.div_ceil(scale);
    let gm_height = height.div_ceil(scale);
    let gm_w = gm_width as usize;

    let mut gainmap = GainMap::new(gm_width, gm_height).map_err(ultrahdr_to_zenjpeg_error)?;

    let mut hdr_row = vec![0.0_f32; w * 4];
    let mut sdr_row = vec![0.0_f32; w * 4];
    let mut gain_row = vec![0.0_f32; w];
    let mut stats = SplitStats::default();

    // Subsampled HDR + SDR rows fed to compute_gain_row at gain-map resolution.
    let mut hdr_gm_row = vec![0.0_f32; gm_w * 4];
    let mut sdr_gm_row = vec![0.0_f32; gm_w * 4];

    let mut min_max = (f32::MAX, f32::MIN);
    // Reverse-map: which HDR row index does each gain-map row pull from?
    // Matches compute_gainmap_slice's center-pixel sampling policy.
    let gm_row_for_y: Vec<u32> = (0..gm_height)
        .map(|gy| (gy * scale + scale / 2).min(height - 1))
        .collect();
    // Forward-map for fast lookup during the row walk.
    let mut gy_for_y: Vec<Option<u32>> = vec![None; height as usize];
    for (gy, &y) in gm_row_for_y.iter().enumerate() {
        gy_for_y[y as usize] = Some(gy as u32);
    }

    for y in 0..height {
        stop.check()?;
        // Linearize one HDR row into interleaved RGBA f32. Mirrors the
        // retired `extract_linear_row_rgba` helper from ultrahdr-core 0.4.
        extract_hdr_row_rgba_linear(hdr, y, &mut hdr_row);

        splitter.split_row(&hdr_row, &mut sdr_row, &mut gain_row, 4, &mut stats);

        // Copy the splitter's SDR row into the RgbaF32 output buffer. Stride
        // is exactly width*16 here (new_pixel_buffer hands back tight rows),
        // but go through row_mut to stay generic over future stride changes.
        let mut sdr_slice = sdr_buffer.as_slice_mut();
        let dst = sdr_slice.row_mut(y);
        let row_bytes: &[u8] = bytemuck::cast_slice(&sdr_row[..w * 4]);
        dst[..row_bytes.len()].copy_from_slice(row_bytes);

        // If this y maps to a gain-map row, subsample columns and quantize.
        if let Some(gy) = gy_for_y[y as usize] {
            for gx in 0..gm_w {
                let x = ((gx as u32 * scale + scale / 2).min(width - 1)) as usize;
                let src_off = x * 4;
                let dst_off = gx * 4;
                hdr_gm_row[dst_off] = hdr_row[src_off];
                hdr_gm_row[dst_off + 1] = hdr_row[src_off + 1];
                hdr_gm_row[dst_off + 2] = hdr_row[src_off + 2];
                hdr_gm_row[dst_off + 3] = hdr_row[src_off + 3];
                sdr_gm_row[dst_off] = sdr_row[src_off];
                sdr_gm_row[dst_off + 1] = sdr_row[src_off + 1];
                sdr_gm_row[dst_off + 2] = sdr_row[src_off + 2];
                sdr_gm_row[dst_off + 3] = sdr_row[src_off + 3];
            }
            let row_start = gy as usize * gm_w;
            let row_end = row_start + gm_w;
            // SDR is in the HDR's gamut here (LumaGainMapSplitter is
            // chromaticity-preserving), so both primaries match.
            compute_gain_row(
                &hdr_gm_row,
                &sdr_gm_row,
                4,
                hdr_gamut,
                hdr_gamut,
                &mut gainmap.data[row_start..row_end],
                gainmap_config,
                &mut min_max,
            );
        }
    }

    let metadata = build_gainmap_metadata(gainmap_config, min_max);
    Ok((sdr_buffer, gainmap, metadata))
}

/// Build the [`GainMapMetadata`] that matches gain-map bytes quantized by
/// `compute_gain_row` on `config`'s boost grid.
///
/// **Contract (zenjpeg #193, ultrahdr #33): the declared per-channel
/// `min`/`max` ARE the dequantization grid.** `compute_gain_row` normalizes
/// each byte over `log(config.min_boost) ..= log(config.max_boost)`, and
/// every conformant reader reconstructs
/// `gain = 2^(min + byte/255 · (max − min))` from the declared range — so
/// the metadata must declare exactly that grid. Declaring the content's
/// observed range instead (what this function did before #193) made every
/// reader dequantize on a narrower grid than the bytes were written on,
/// reconstructing under-boosted HDR whenever the content's gain range was
/// narrower than the configured one.
///
/// `min_max` — the `(min, max)` gain accumulator `compute_gain_row` fills
/// across the image — does not feed the declared range. Its `max` only
/// widens `alternate_hdr_headroom` so full gain application stays reachable
/// when the content exceeds the configured headroom; a non-finite or
/// non-positive value (no pixel observed) falls back to the grid top.
///
/// Mirrors ultrahdr-core's `pub(crate) metadata_for_config_grid`
/// (`a09478f0bfaa`), replicated here because it is not exported.
fn build_gainmap_metadata(config: &GainMapConfig, min_max: (f32, f32)) -> GainMapMetadata {
    let observed_max = min_max.1;
    let observed = if observed_max.is_finite() && observed_max > 0.0 {
        observed_max.clamp(config.min_boost, config.max_boost)
    } else {
        config.max_boost
    };
    let channel = GainMapChannel {
        min: (config.min_boost as f64).log2(),
        max: (config.max_boost as f64).log2(),
        gamma: config.gamma as f64,
        base_offset: config.base_offset as f64,
        alternate_offset: config.alternate_offset as f64,
    };
    let mut metadata = GainMapMetadata::default();
    metadata.channels = [channel; 3];
    metadata.base_hdr_headroom = (config.base_hdr_headroom as f64).log2();
    metadata.alternate_hdr_headroom = (config.alternate_hdr_headroom.max(observed) as f64).log2();
    metadata.use_base_color_space = true;
    metadata.backward_direction = false;
    metadata
}

/// Map a [`ColorPrimaries`] to the matching luma-weights triplet expected
/// by [`SplitConfig::luma_weights`]. Falls back to BT.709 for unknown
/// primaries (matches what `rgb_to_luminance` does internally).
fn luma_weights_for(primaries: ColorPrimaries) -> [f32; 3] {
    match primaries {
        ColorPrimaries::Bt2020 => zentone::LUMA_BT2020,
        ColorPrimaries::DisplayP3 => zentone::LUMA_P3,
        _ => zentone::LUMA_BT709,
    }
}

/// Read one HDR row, linearize via the descriptor's transfer function, and
/// emit interleaved RGBA f32 (alpha forced to 1.0 — the gain map flow is
/// alpha-agnostic). Supports the formats `compute_gainmap` already accepts:
/// `Rgba8`, `Rgb8`, `RgbaF32`, `RgbaF16`, `RgbF16`, `Gray8`.
fn extract_hdr_row_rgba_linear(hdr: &PixelBuffer, y: u32, out: &mut [f32]) {
    use ultrahdr_core::color::transfer::{hlg_eotf, pq_eotf, srgb_eotf};

    let slice = hdr.as_slice();
    let desc = slice.descriptor();
    let format = desc.pixel_format();
    let transfer = desc.transfer();
    let stride = slice.stride();
    let data = slice.as_strided_bytes();
    let width = slice.width() as usize;
    let row_off = y as usize * stride;

    let linearize = |c: f32| -> f32 {
        match transfer {
            TransferFunction::Srgb => srgb_eotf(c),
            TransferFunction::Linear => c,
            TransferFunction::Pq => pq_eotf(c),
            // HLG with default display peak (1000 nits) — same fallback as
            // ultrahdr-core's `apply_transfer_to_linear`.
            TransferFunction::Hlg => hlg_eotf(c, 1000.0),
            _ => srgb_eotf(c),
        }
    };

    for x in 0..width {
        let (r, g, b) = match format {
            PixelFormat::Rgba8 => {
                let i = row_off + x * 4;
                (
                    data[i] as f32 / 255.0,
                    data[i + 1] as f32 / 255.0,
                    data[i + 2] as f32 / 255.0,
                )
            }
            PixelFormat::Rgb8 => {
                let i = row_off + x * 3;
                (
                    data[i] as f32 / 255.0,
                    data[i + 1] as f32 / 255.0,
                    data[i + 2] as f32 / 255.0,
                )
            }
            PixelFormat::RgbaF32 => {
                let i = row_off + x * 16;
                (
                    f32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]),
                    f32::from_le_bytes([data[i + 4], data[i + 5], data[i + 6], data[i + 7]]),
                    f32::from_le_bytes([data[i + 8], data[i + 9], data[i + 10], data[i + 11]]),
                )
            }
            PixelFormat::RgbaF16 => {
                let i = row_off + x * 8;
                (
                    half::f16::from_le_bytes([data[i], data[i + 1]]).to_f32(),
                    half::f16::from_le_bytes([data[i + 2], data[i + 3]]).to_f32(),
                    half::f16::from_le_bytes([data[i + 4], data[i + 5]]).to_f32(),
                )
            }
            PixelFormat::RgbF16 => {
                let i = row_off + x * 6;
                (
                    half::f16::from_le_bytes([data[i], data[i + 1]]).to_f32(),
                    half::f16::from_le_bytes([data[i + 2], data[i + 3]]).to_f32(),
                    half::f16::from_le_bytes([data[i + 4], data[i + 5]]).to_f32(),
                )
            }
            PixelFormat::Gray8 => {
                let i = row_off + x;
                let v = data[i] as f32 / 255.0;
                (v, v, v)
            }
            _ => (0.0, 0.0, 0.0),
        };

        let (lr, lg, lb) = match format {
            // f32/f16 paths already store linear (or PQ/HLG/sRGB-encoded
            // float values) — apply the transfer if present.
            PixelFormat::RgbaF32 | PixelFormat::RgbaF16 | PixelFormat::RgbF16 => {
                (linearize(r), linearize(g), linearize(b))
            }
            // 8-bit paths: 8-bit linear isn't really a thing in practice; if
            // transfer says linear, trust it; otherwise apply sRGB EOTF.
            _ => match transfer {
                TransferFunction::Linear => (r, g, b),
                _ => (srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)),
            },
        };

        let i = x * 4;
        out[i] = lr;
        out[i + 1] = lg;
        out[i + 2] = lb;
        out[i + 3] = 1.0;
    }
}

/// Idiot-proof one-call Ultra HDR encode for the most basic case:
/// **luma-only gain map, default-correct, no config to assemble**.
///
/// Picks `Bt2446C` as the default tone curve (ITU-R BT.2446 Method C —
/// exact algebraic inverse, sensible default for general HDR10 content).
/// Uses [`GainMapConfig::default`] (single-channel gain map) and
/// reasonable JPEG defaults (Q=85 base, Q=75 gain map, 4:2:0 subsampling).
///
/// The HDR input's `descriptor().primaries` chooses the luma weights; the
/// `descriptor().transfer()` selects the EOTF (PQ / HLG / Linear / sRGB).
///
/// Mirrors the libultrahdr default-mode shape (pass HDR pixels, get bytes).
/// For per-curve choice use [`encode_ultrahdr_with_curve`]; for the full
/// matrix of knobs use [`encode_ultrahdr`].
pub fn encode_ultrahdr_luma(hdr: &PixelBuffer) -> Result<Vec<u8>> {
    use crate::encoder::ChromaSubsampling;
    // BT.2446 Method C over the content's MEASURED light (appendix AA):
    // the curve consumes the pixels' own luminance per sample, so the two
    // constructor constants must state FACTS, never an assumed content
    // peak. `Bt2446C::new(input_scale, sdr_ref)`:
    //
    // - `input_scale` is the INPUT NORMALIZATION — "what luma 1.0 means,
    //   in nits" (the zentone `LumaToneMap` contract). It must state the
    //   unit convention of the rows the fused splitter feeds, which
    //   depends on the input transfer (`curve_input_scale_nits`). The
    //   previous `1000.0` asserted a nominal HDR10 content peak instead —
    //   config, not measurement (and the old doc advice of `10000.0` for
    //   PQ content compounded it).
    // - `sdr_ref = 100.0` is the curve's CALIBRATED SDR reference —
    //   BT.2446-1 is specified against a ~100-nit SDR display ("typically
    //   100, or 120 for Method C's super-whites" per zentone's docs),
    //   under which SDR diffuse white (203 nits in) tone-maps to white
    //   (~0.91), not mid-gray. The previous `203.0` pasted BT.2408's
    //   HDR-terms diffuse white into the SDR-reference slot.
    //
    // VERSION FACT (pinned by `bt2446c_params_inert_at_zentone_0_1`): the
    // published zentone 0.1.0 this build resolves RESERVES both params
    // (`let _ = (hdr_peak_nits, sdr_peak_nits)`) — its curve is
    // input-relative (1.0 == "HDR peak", percent domain), so this change
    // is byte-neutral TODAY. The constants go live with zentone's
    // nits-faithful curve (local zentone 0.2.0); stating the correct
    // values now means that dependency bump changes rendering because the
    // CURVE improved, not because stale config constants suddenly started
    // being believed.
    let scale = curve_input_scale_nits(hdr);
    encode_ultrahdr_with_curve(
        hdr,
        &Bt2446C::new(scale, 100.0),
        &GainMapConfig::default(),
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
}

/// What luma value `1.0` means, in cd/m², for the linearized rows
/// [`extract_hdr_row_rgba_linear`] produces from this input — i.e. the
/// [`LumaToneMap`] input normalization that makes a tone curve see the
/// content's TRUE (measured) luminance:
///
/// - Linear float and 8-bit sRGB rows are SDR-white-relative
///   (BT.2408: `1.0` = 203 cd/m² — the crate's `LinearFloat` convention).
/// - PQ rows come out of `pq_eotf` PQ-normalized (`1.0` = 10 000 cd/m²).
/// - HLG rows come out of `hlg_eotf(·, 1000.0)` in absolute display nits
///   at the BT.2100 1000-nit reference (scale `1.0`).
fn curve_input_scale_nits(hdr: &PixelBuffer) -> f32 {
    match hdr.as_slice().descriptor().transfer() {
        TransferFunction::Pq => 10_000.0,
        TransferFunction::Hlg => 1.0,
        _ => 203.0,
    }
}

/// Convert a linear-light f32 RGBA `PixelBuffer` to an sRGB-encoded
/// Rgba8 `PixelBuffer`. Uses the `linear-srgb` crate's OETF directly so
/// we don't need a private duplicate (#71).
fn linear_f32_to_srgb_rgba8(sdr_linear: &PixelBuffer) -> Result<PixelBuffer> {
    let slice = sdr_linear.as_slice();
    let stride = slice.stride();
    let width = slice.width();
    let height = slice.rows();
    let bytes = slice.as_strided_bytes();

    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for y in 0..height as usize {
        let row_start = y * stride;
        for x in 0..width as usize {
            let idx = row_start + x * 16;
            let r =
                f32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
            let g = f32::from_le_bytes([
                bytes[idx + 4],
                bytes[idx + 5],
                bytes[idx + 6],
                bytes[idx + 7],
            ]);
            let b = f32::from_le_bytes([
                bytes[idx + 8],
                bytes[idx + 9],
                bytes[idx + 10],
                bytes[idx + 11],
            ]);
            // Saturate to [0, 1] then sRGB OETF + quantize.
            out.push(
                (linear_srgb::default::linear_to_srgb(r.clamp(0.0, 1.0)) * 255.0).round() as u8,
            );
            out.push(
                (linear_srgb::default::linear_to_srgb(g.clamp(0.0, 1.0)) * 255.0).round() as u8,
            );
            out.push(
                (linear_srgb::default::linear_to_srgb(b.clamp(0.0, 1.0)) * 255.0).round() as u8,
            );
            out.push(0xFF);
        }
    }
    pixel_buffer_from_vec(
        out,
        width,
        height,
        PixelFormat::Rgba8,
        sdr_linear.descriptor().primaries,
        TransferFunction::Srgb,
    )
    .map_err(ultrahdr_to_zenjpeg_error)
}

/// Encode SDR image with pre-computed gain map.
///
/// Uses [`GainMapEncodingFormat::Both`] for maximum compatibility.
pub fn encode_with_gainmap(
    sdr: &PixelBuffer,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    encode_with_gainmap_format(
        sdr,
        gainmap,
        metadata,
        encoder_config,
        gainmap_quality,
        GainMapEncodingFormat::Both,
        stop,
    )
}

/// Encode SDR image with pre-computed gain map and metadata format control.
pub fn encode_with_gainmap_format(
    sdr: &PixelBuffer,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    encoder_config: &EncoderConfig,
    gainmap_quality: f32,
    metadata_format: GainMapEncodingFormat,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let gainmap_jpeg = encode_gainmap_jpeg(gainmap, gainmap_quality, &stop)?;
    stop.check()?;

    let metadata_markers = build_gainmap_metadata_markers(metadata, metadata_format);
    let mut gainmap_final = gainmap_jpeg;
    for marker in metadata_markers.iter().rev() {
        gainmap_final = inject_marker_after_soi(&gainmap_final, marker)?;
    }

    let primary_xmp = generate_primary_xmp(gainmap_final.len());

    let mut segments = EncoderSegments::new().set_xmp(&primary_xmp).add_mpf_image(
        gainmap_final,
        crate::encode::extras::MpfImageType::Undefined,
    );

    let include_iso = matches!(
        metadata_format,
        GainMapEncodingFormat::Iso21496 | GainMapEncodingFormat::Both
    );
    if include_iso {
        segments = segments.add_raw(0xE2, ISO_21496_1_PRIMARY_APP2_BODY.to_vec());
    }

    encode_sdr_base(sdr, encoder_config, segments, stop)
}

/// Tonemap HDR to a Bt709/Srgb `PixelBuffer`. Centralizes the conversion to
/// the pattern ultrahdr-rs uses (`tonemap_image_to_srgb8` + `pixel_buffer_from_vec`).
fn tonemap_to_pixel_buffer(hdr: &PixelBuffer, target_gamut: ColorPrimaries) -> Result<PixelBuffer> {
    let bytes = tonemap_image_to_srgb8(hdr, target_gamut).map_err(ultrahdr_to_zenjpeg_error)?;
    pixel_buffer_from_vec(
        bytes,
        hdr.width(),
        hdr.height(),
        PixelFormat::Rgba8,
        target_gamut,
        TransferFunction::Srgb,
    )
    .map_err(ultrahdr_to_zenjpeg_error)
}

/// Build the gain-map secondary's metadata APP markers (XMP APP1 and/or
/// ISO 21496-1 APP2) in canonical order. Mirrors ultrahdr-rs's local helper
/// since `ultrahdr_core::metadata::xmp::build_gainmap_metadata_markers` was
/// retired in 0.5.
fn build_gainmap_metadata_markers(
    metadata: &GainMapMetadata,
    format: GainMapEncodingFormat,
) -> Vec<Vec<u8>> {
    let mut markers = Vec::with_capacity(2);
    if matches!(
        format,
        GainMapEncodingFormat::Xmp | GainMapEncodingFormat::Both
    ) {
        let xmp = generate_gainmap_xmp(metadata);
        markers.push(create_xmp_app1_marker(&xmp));
    }
    if matches!(
        format,
        GainMapEncodingFormat::Iso21496 | GainMapEncodingFormat::Both
    ) {
        let iso_body = serialize_iso21496_fmt(metadata, Iso21496Format::JpegApp2BodyWithUrn);
        markers.push(wrap_app2(&iso_body));
    }
    markers
}

/// Wrap a body in a JPEG APP2 marker (`FF E2` + big-endian length-with-self
/// + body). Mirrors ultrahdr-rs's local `wrap_app2`.
fn wrap_app2(body: &[u8]) -> Vec<u8> {
    let total_length = 2 + body.len();
    let mut marker = Vec::with_capacity(4 + body.len());
    marker.extend_from_slice(&[
        0xFF,
        0xE2,
        ((total_length >> 8) & 0xFF) as u8,
        (total_length & 0xFF) as u8,
    ]);
    marker.extend_from_slice(body);
    marker
}

/// Inject a complete JPEG marker segment after SOI.
fn inject_marker_after_soi(jpeg: &[u8], marker: &[u8]) -> Result<Vec<u8>> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        // `jpeg` here is always our own freshly-`encode_gainmap_jpeg`-produced
        // buffer (see the sole call site), never caller-supplied — a missing
        // SOI means this encoder's own output is broken, an internal
        // invariant violation, not an unsupported feature (caterr Pattern-B
        // follow-up finding #1 investigation).
        return Err(Error::internal(
            "gain map JPEG missing SOI (internal encoder invariant)",
        ));
    }
    let mut result = Vec::with_capacity(jpeg.len() + marker.len());
    result.extend_from_slice(&jpeg[..2]); // SOI
    result.extend_from_slice(marker); // marker segment
    result.extend_from_slice(&jpeg[2..]); // rest of JPEG
    Ok(result)
}

/// Encode the gain map as a grayscale JPEG.
fn encode_gainmap_jpeg(gainmap: &GainMap, quality: f32, stop: &impl Stop) -> Result<Vec<u8>> {
    let config = EncoderConfig::grayscale(quality);
    let mut encoder = config.encode_from_bytes(
        gainmap.width,
        gainmap.height,
        if gainmap.channels == 1 {
            PixelLayout::Gray8Srgb
        } else {
            PixelLayout::Rgb8Srgb
        },
    )?;
    encoder.push_packed(&gainmap.data, stop)?;
    encoder.finish()
}

/// Encode the SDR base image. The SDR `PixelBuffer` must be `Rgba8` or `Rgb8`.
fn encode_sdr_base(
    sdr: &PixelBuffer,
    config: &EncoderConfig,
    segments: EncoderSegments,
    stop: impl Stop,
) -> Result<Vec<u8>> {
    let format = sdr.descriptor().pixel_format();
    let layout = match format {
        PixelFormat::Rgba8 => PixelLayout::Rgba8Srgb,
        PixelFormat::Rgb8 => PixelLayout::Rgb8Srgb,
        _ => {
            return Err(Error::unsupported_feature(
                "SDR PixelBuffer must be Rgba8 or Rgb8 for UltraHDR encoding",
            ));
        }
    };

    let config_with_segments = config.clone().with_segments(segments);
    let mut encoder = config_with_segments.encode_from_bytes(sdr.width(), sdr.height(), layout)?;

    encoder.push_packed(sdr.as_slice().as_strided_bytes(), stop)?;
    encoder.finish()
}

/// Convert an ultrahdr_core error to a zenjpeg Error.
///
/// Generic over `Display` so it accepts both bare `ultrahdr_core::Error` and
/// the `At<Error>` location wrapper (ultrahdr-core #31).
fn ultrahdr_to_zenjpeg_error(e: impl core::fmt::Display) -> Error {
    Error::decode_error(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf_with_transfer(transfer: TransferFunction) -> PixelBuffer {
        let px: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
        let bytes: Vec<u8> = px.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let desc = zenpixels::PixelDescriptor::RGBAF32_LINEAR.with_transfer(transfer);
        PixelBuffer::from_vec(bytes, 1, 1, desc).expect("fixture")
    }

    /// Appendix AA: the tone curve's input normalization states the ROW
    /// convention (what the linearizer's `1.0` means in nits), never an
    /// assumed content peak. Linear/sRGB rows are SDR-white-relative (203),
    /// PQ rows PQ-normalized (10 000), HLG rows absolute nits (1.0). The
    /// old configured `1000.0` matched none of them.
    #[test]
    fn curve_input_scale_matches_row_convention() {
        assert_eq!(
            curve_input_scale_nits(&buf_with_transfer(TransferFunction::Linear)),
            203.0
        );
        assert_eq!(
            curve_input_scale_nits(&buf_with_transfer(TransferFunction::Srgb)),
            203.0
        );
        assert_eq!(
            curve_input_scale_nits(&buf_with_transfer(TransferFunction::Pq)),
            10_000.0
        );
        assert_eq!(
            curve_input_scale_nits(&buf_with_transfer(TransferFunction::Hlg)),
            1.0
        );
    }

    /// VERSION PIN: the published zentone 0.1.0 this build resolves treats
    /// `Bt2446C::new`'s two parameters as RESERVED (`let _ = ...`) — its
    /// curve is input-relative, so every construction behaves identically.
    /// This pin does two jobs: (1) it documents that the appendix-AA
    /// constant correction in `encode_ultrahdr_luma` is byte-neutral at
    /// this zentone version, and (2) it FAILS the moment a zentone bump
    /// makes the parameters live (the nits-faithful curve in zentone git),
    /// forcing that bump to be taken consciously: re-verify that
    /// `curve_input_scale_nits` + the 100-nit SDR reference then map SDR
    /// diffuse white (input 1.0 = 203 nits) to white (~0.9) in the base —
    /// the assertion to flip this test to.
    #[test]
    fn bt2446c_params_inert_at_zentone_0_1() {
        use zentone::LumaToneMap as _;
        let corrected = Bt2446C::new(203.0, 100.0);
        let legacy = Bt2446C::new(1000.0, 203.0);
        for y in [0.05_f32, 0.25, 1.0, 4.0, 40.0] {
            let (c, l) = (corrected.map_luma(y), legacy.map_luma(y));
            assert!(
                (c - l).abs() < 1e-7,
                "zentone made Bt2446C's constructor params LIVE (map_luma({y}): \
                 corrected {c} vs legacy {l}) — re-verify encode_ultrahdr_luma's \
                 measured-scale constants and flip this test to assert \
                 map_luma(1.0) ≈ 0.9 (SDR white → white) per appendix AA"
            );
        }
        // Monotone into the highlights under the pinned semantics too.
        assert!(corrected.map_luma(4.0) > corrected.map_luma(1.0));
    }
}
