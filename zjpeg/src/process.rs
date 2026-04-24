//! Unified JPEG processing pipeline: optimize, resize, crop, transform.
//!
//! Replaces the separate `optimize`, `transform`, and `restructure` subcommands
//! with a single `process` command that auto-detects the optimal pipeline:
//!
//! - **Lossless**: orient-only or flip/rotate → DCT-domain transform
//! - **Restructure**: progressive/baseline conversion only → lossless restructure
//! - **Lossy**: resize, crop, quality change, deblock, XYB → full decode/encode

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use zenjpeg::deblock::{BoundaryStrength, filter_plane_boundary_4tap};
use zenjpeg::decoder::{
    DecodeConfig, OutputTarget, PreserveConfig, SegmentType, Strictness, Subsampling,
    TargetColorSpace,
};
use zenjpeg::detect::content::{DeblockAction, classify_from_probe, recommend_deblock};
use zenjpeg::detect::{self, QualityScale};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, ProgressiveScanMode,
    Quality, QuantTableConfig, XybSubsampling,
};
use zenjpeg::lossless::{
    self, LosslessTransform, OutputMode, RestartInterval, RestructureConfig, TransformConfig,
};
use zenresize::{FitMode, fit_cover_source_crop, fit_dims};

use crate::batch::{self, BatchSummary, FileResult};
use crate::coord;
use crate::output::OutputConfig;
use crate::{
    DeblockArg, FitArg, FlipArg, IccTargetArg, OrientArg, ProcessArgs, StructureArg,
    SubsamplingArg, TrellisArg,
};

/// Which pipeline to use for a given file.
enum PipelineKind {
    /// Orient-only or flip/rotate → lossless DCT-domain transform.
    Lossless,
    /// Progressive/baseline conversion only → lossless restructure.
    Restructure,
    /// Everything else → full decode/encode.
    Lossy,
}

pub fn run(args: ProcessArgs) -> Result<()> {
    let files = batch::expand_inputs(&args.input)?;
    if files.is_empty() {
        anyhow::bail!("no JPEG files found");
    }

    let output_config = OutputConfig::new(
        args.output.clone(),
        args.in_place,
        args.suffix.clone(),
        args.force,
        args.dry_run,
    )?;

    let is_single = files.len() == 1;
    let mut summary = BatchSummary::new();

    let num_jobs = args.jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(1)
    });

    if files.len() > 1 && num_jobs > 1 {
        use rayon::prelude::*;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_jobs)
            .build()
            .context("failed to create thread pool")?;

        let progress = indicatif::ProgressBar::new(files.len() as u64);
        progress.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );

        let results: Vec<FileResult> = pool.install(|| {
            files
                .par_iter()
                .map(|path| {
                    let result = process_one(path, &args, &output_config, is_single);
                    progress.inc(1);
                    result
                })
                .collect()
        });

        progress.finish_and_clear();

        for r in results {
            summary.add(r);
        }
    } else {
        for path in &files {
            let r = process_one(path, &args, &output_config, is_single);
            summary.add(r);
        }
    }

    if args.report {
        summary.print_report();
    }

    if let Some(ref csv_path) = args.csv {
        summary.write_csv(csv_path)?;
    }

    for r in &summary.results {
        if let Some(ref err) = r.error {
            eprintln!("error: {}: {err}", r.path.display());
        }
    }

    Ok(())
}

fn process_one(
    path: &Path,
    args: &ProcessArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> FileResult {
    let input_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    match process_inner(path, args, output_config, is_single) {
        Ok((output_size, skipped)) => FileResult {
            path: path.to_path_buf(),
            input_size,
            output_size: Some(output_size),
            skipped,
            error: None,
        },
        Err(e) => FileResult {
            path: path.to_path_buf(),
            input_size,
            output_size: None,
            skipped: false,
            error: Some(format!("{e:#}")),
        },
    }
}

fn process_inner(
    path: &Path,
    args: &ProcessArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<(u64, bool)> {
    let data =
        std::fs::read(path).with_context(|| format!("failed to read '{}'", path.display()))?;

    let pipeline = classify_pipeline(args);

    if args.lossless_only && matches!(pipeline, PipelineKind::Lossy) {
        anyhow::bail!(
            "lossy re-encoding required for '{}' but --lossless-only is set",
            path.display()
        );
    }

    match pipeline {
        PipelineKind::Lossless => run_lossless(&data, path, args, output_config, is_single),
        PipelineKind::Restructure => run_restructure(&data, path, args, output_config, is_single),
        PipelineKind::Lossy => run_lossy(&data, path, args, output_config, is_single),
    }
}

// ============================================================================
// Pipeline classification
// ============================================================================

fn classify_pipeline(args: &ProcessArgs) -> PipelineKind {
    let has_resize =
        args.width.is_some() || args.height.is_some() || args.size.is_some() || args.dpr.is_some();
    let has_crop = args.crop.is_some();
    let has_pad = args.pad.is_some();
    let has_quality = args.quality.is_some()
        || args.distance.is_some()
        || args.crush.is_some()
        || args.tolerance.is_some()
        || args.search_ssim2.is_some()
        || args.search_distance.is_some();
    let has_deblock = args.deblock != DeblockArg::Off;
    let has_xyb = args.xyb;
    let has_icc = args.apply_icc.is_some();
    let has_sharp_yuv = args.sharp_yuv;
    let has_subsampling = args.subsampling.is_some();
    let has_quant = args.quant.is_some();
    let has_chroma_tables = args.chroma_tables.is_some();
    let has_preset = args.preset.is_some();
    // optimize_scans requires entropy re-encoding (scan search tries 64 candidates),
    // which currently goes through the lossy encoder. Could be lossless if the
    // restructure module gained scan search support on existing DCT coefficients.
    let has_optimize_scans = args.optimize_scans;
    let has_transform = args.rotate.is_some() || args.flip.is_some();
    let has_structure = args.structure != StructureArg::Auto;
    let auto_orient = args.orient == OrientArg::Auto;

    let lossy_triggers = has_resize
        || has_crop
        || has_pad
        || has_quality
        || has_deblock
        || has_xyb
        || has_icc
        || has_sharp_yuv
        || has_subsampling
        || has_quant
        || has_chroma_tables
        || has_preset
        || has_optimize_scans;

    if lossy_triggers {
        return PipelineKind::Lossy;
    }

    // No lossy triggers: check for restructure-only
    if !has_transform && !auto_orient && has_structure {
        return PipelineKind::Restructure;
    }

    // Has transform/orient, or nothing at all → lossless
    // (auto-orient with no other flags = lossless orient)
    if has_transform || auto_orient {
        return PipelineKind::Lossless;
    }

    // Nothing at all → default to lossy (optimize with auto quality)
    PipelineKind::Lossy
}

// ============================================================================
// Lossless path
// ============================================================================

fn run_lossless(
    data: &[u8],
    path: &Path,
    args: &ProcessArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<(u64, bool)> {
    let output = if args.orient == OrientArg::Auto && args.rotate.is_none() && args.flip.is_none() {
        // Pure auto-orient
        lossless::apply_exif_orientation(data, enough::Unstoppable)
            .map_err(|e| anyhow::anyhow!("orientation failed: {e}"))?
    } else {
        // Explicit transform (may combine with auto-orient in future)
        let xform = determine_lossless_transform(args)?;
        let config = TransformConfig {
            transform: xform,
            ..Default::default()
        };
        lossless::transform(data, &config, enough::Unstoppable)
            .map_err(|e| anyhow::anyhow!("transform failed: {e}"))?
    };

    let output_size = output.len() as u64;
    let input_size = data.len() as u64;

    if args.skip_if_larger && output_size >= input_size {
        return Ok((input_size, true));
    }

    let output_path = output_config.resolve(path, is_single)?;
    output_config.check_writable(&output_path, path)?;

    if !output_config.dry_run {
        OutputConfig::ensure_parent(&output_path)?;
        std::fs::write(&output_path, &output)
            .with_context(|| format!("failed to write '{}'", output_path.display()))?;
    }

    Ok((output_size, false))
}

fn determine_lossless_transform(args: &ProcessArgs) -> Result<LosslessTransform> {
    if let Some(FlipArg::H) = args.flip {
        return Ok(LosslessTransform::FlipHorizontal);
    }
    if let Some(FlipArg::V) = args.flip {
        return Ok(LosslessTransform::FlipVertical);
    }

    if let Some(degrees) = args.rotate {
        return match degrees {
            90 => Ok(LosslessTransform::Rotate90),
            180 => Ok(LosslessTransform::Rotate180),
            270 => Ok(LosslessTransform::Rotate270),
            0 | 360 => Ok(LosslessTransform::None),
            _ => anyhow::bail!("rotation must be 0, 90, 180, 270, or 360"),
        };
    }

    // Auto-orient is handled separately in run_lossless
    anyhow::bail!("no transform specified")
}

// ============================================================================
// Restructure path
// ============================================================================

fn run_restructure(
    data: &[u8],
    path: &Path,
    args: &ProcessArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<(u64, bool)> {
    let output_mode = match args.structure {
        StructureArg::Progressive => OutputMode::Progressive,
        StructureArg::Baseline | StructureArg::Auto => OutputMode::Sequential,
    };

    let config = RestructureConfig {
        output_mode,
        restart_interval: RestartInterval::None,
        ..Default::default()
    };

    let output = lossless::restructure(data, &config, enough::Unstoppable)
        .map_err(|e| anyhow::anyhow!("restructure failed: {e}"))?;

    let output_size = output.len() as u64;
    let input_size = data.len() as u64;

    if args.skip_if_larger && output_size >= input_size {
        return Ok((input_size, true));
    }

    let output_path = output_config.resolve(path, is_single)?;
    output_config.check_writable(&output_path, path)?;

    if !output_config.dry_run {
        OutputConfig::ensure_parent(&output_path)?;
        std::fs::write(&output_path, &output)
            .with_context(|| format!("failed to write '{}'", output_path.display()))?;
    }

    Ok((output_size, false))
}

// ============================================================================
// Lossy path
// ============================================================================

fn run_lossy(
    data: &[u8],
    path: &Path,
    args: &ProcessArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<(u64, bool)> {
    // Step 1: Probe source JPEG
    let probe = detect::probe(data).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Step 2: Determine quality and subsampling
    let (quality, subsampling) = determine_encode_params(args, &probe)?;

    // Step 3: Build layout pipeline if spatial operations requested
    let layout_result = build_layout(args, probe.dimensions.width, probe.dimensions.height)?;

    // Step 4: Decode
    let need_deblock = args.deblock != DeblockArg::Off;
    // XYB wants linear-f32 input end-to-end (sRGB → linear conversion happens
    // in the decoder rather than the encoder). The RgbF32Linear input path to
    // `EncoderConfig::xyb` was broken until zenjpeg commit 04e5e5da (Known Bug
    // #7) — fixed, so this can route through f32 again.
    let decode_to_f32 = need_deblock || args.xyb;

    let mut decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    if decode_to_f32 {
        decoder.output_target = OutputTarget::LinearF32;
    }
    decoder.correct_color = args.apply_icc.map(|t| convert_icc_target(t));
    if args.orient == OrientArg::Auto {
        decoder = decoder.auto_orient(true);
    }
    if let Some(s) = args.strictness {
        decoder.strictness = match s {
            crate::StrictnessArg::Strict => Strictness::Strict,
            crate::StrictnessArg::Balanced => Strictness::Balanced,
            crate::StrictnessArg::Lenient => Strictness::Lenient,
            crate::StrictnessArg::Permissive => Strictness::Permissive,
        };
    }

    let mut result = decoder
        .decode(data, enough::Unstoppable)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    let mut width = result.width();
    let mut height = result.height();

    // Step 5: Extract metadata
    let extras = result.take_extras();

    // Step 6: Apply layout (crop, resize, pad)
    let (output_jpeg, output_size) = if decode_to_f32 {
        let mut pixels_f32 = result
            .into_pixels_f32()
            .context("expected f32 decode output")?;

        // Apply source crop from layout
        if let Some(ref lr) = layout_result
            && let Some(ref crop_rect) = lr.source_crop
        {
            pixels_f32 = crop_f32_pixels(
                &pixels_f32,
                width as usize,
                crop_rect.x as usize,
                crop_rect.y as usize,
                crop_rect.w as usize,
                crop_rect.h as usize,
            );
            width = crop_rect.w;
            height = crop_rect.h;
        }

        // Resize
        if let Some(ref lr) = layout_result
            && lr.needs_resize
        {
            pixels_f32 = resize_f32(&pixels_f32, width, height, lr.target_w, lr.target_h, args)?;
            width = lr.target_w;
            height = lr.target_h;
        }

        // Pad
        if let Some(ref lr) = layout_result
            && let Some(ref pad) = lr.padding
        {
            pixels_f32 = pad_f32_pixels(&pixels_f32, width as usize, height as usize, pad);
            width = width + pad.left + pad.right;
            height = height + pad.top + pad.bottom;
        }

        if need_deblock {
            apply_deblock(
                &mut pixels_f32,
                width as usize,
                height as usize,
                &probe,
                args,
            );
        }

        let jpeg = if let Some((band, metric)) = search_target(args) {
            let (q_lo, q_hi) = args.resolve_quality_range()?;
            let seed = match quality {
                Quality::ApproxJpegli(q) => Some(q),
                _ => None,
            };
            // Metric reference on the f32 encode path: invert the decoder's
            // LinearF32 transfer (sRGB OETF via `linear_srgb` crate, byte-equal
            // to the decoder's own `srgb_u8_to_linear` LUT to ±1 u8). This
            // compares candidates against the ORIGINAL source image in the sRGB
            // u8 domain, rather than against a noisy Q99 re-encode (the
            // encoder's linear→sRGB path has ~1.1 RMSE u8 noise that a
            // Q99-reference would bake into the metric floor).
            let extras_ref = &extras;
            let ref_u8 = linear_f32_to_srgb_u8(&pixels_f32, width, height);
            let result = crate::search::search_for_band(
                &ref_u8,
                width,
                height,
                band,
                metric,
                (q_lo, q_hi),
                seed,
                false,
                args.attempts,
                |q| {
                    let q_typed = Quality::ApproxJpegli(q);
                    let config = build_encoder_config(args, q_typed, subsampling, extras_ref);
                    let pixel_bytes: &[u8] = bytemuck::cast_slice(&pixels_f32);
                    config
                        .encode_bytes(pixel_bytes, width, height, PixelLayout::RgbF32Linear)
                        .map_err(|e| anyhow::anyhow!("encode failed at Q{q:.1}: {e}"))
                },
            )?;
            if !result.in_band {
                eprintln!(
                    "warning: search exhausted {} attempts without hitting band; using \
                     Q{:.1} (metric {:.3}, band {}..{})",
                    result.attempts_used, result.quality, result.metric, band.min, band.max,
                );
            } else if is_single {
                eprintln!(
                    "search: Q{:.1} → metric {:.3} (band {}..{}, {} attempt{})",
                    result.quality,
                    result.metric,
                    band.min,
                    band.max,
                    result.attempts_used,
                    if result.attempts_used == 1 { "" } else { "s" },
                );
            }
            result.jpeg
        } else {
            let config = build_encoder_config(args, quality, subsampling, &extras);
            let pixel_bytes: &[u8] = bytemuck::cast_slice(&pixels_f32);
            config
                .encode_bytes(pixel_bytes, width, height, PixelLayout::RgbF32Linear)
                .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?
        };
        let size = jpeg.len() as u64;
        (jpeg, size)
    } else {
        let mut pixels_u8 = result
            .pixels_u8()
            .context("expected u8 decode output")?
            .to_vec();

        // Apply source crop from layout
        if let Some(ref lr) = layout_result
            && let Some(ref crop_rect) = lr.source_crop
        {
            pixels_u8 = crop_u8_pixels(
                &pixels_u8,
                width as usize,
                crop_rect.x as usize,
                crop_rect.y as usize,
                crop_rect.w as usize,
                crop_rect.h as usize,
            );
            width = crop_rect.w;
            height = crop_rect.h;
        }

        // Resize
        if let Some(ref lr) = layout_result
            && lr.needs_resize
        {
            pixels_u8 = resize_u8(&pixels_u8, width, height, lr.target_w, lr.target_h, args)?;
            width = lr.target_w;
            height = lr.target_h;
        }

        // Pad
        if let Some(ref lr) = layout_result
            && let Some(ref pad) = lr.padding
        {
            pixels_u8 = pad_u8_pixels(&pixels_u8, width as usize, height as usize, pad);
            width = width + pad.left + pad.right;
            height = height + pad.top + pad.bottom;
        }

        let jpeg = if let Some((band, metric)) = search_target(args) {
            let (q_lo, q_hi) = args.resolve_quality_range()?;
            // Seed: smart-detection result from `determine_encode_params` (source-relative
            // proportional-tolerance quality). Typically 1-2 attempts closer to band than
            // a naive midpoint.
            let seed = match quality {
                Quality::ApproxJpegli(q) => Some(q),
                _ => None,
            };
            let extras_ref = &extras;
            let result = crate::search::search_for_band(
                &pixels_u8,
                width,
                height,
                band,
                metric,
                (q_lo, q_hi),
                seed,
                false, // u8 path candidates are sRGB JPEGs — no CMS needed
                args.attempts,
                |q| {
                    let q_typed = Quality::ApproxJpegli(q);
                    let config = build_encoder_config(args, q_typed, subsampling, extras_ref);
                    config
                        .encode_bytes(&pixels_u8, width, height, PixelLayout::Rgb8Srgb)
                        .map_err(|e| anyhow::anyhow!("encode failed at Q{q:.1}: {e}"))
                },
            )?;
            if !result.in_band {
                eprintln!(
                    "warning: search exhausted {} attempts without hitting band; using \
                     Q{:.1} (metric {:.3}, band {}..{})",
                    result.attempts_used, result.quality, result.metric, band.min, band.max,
                );
            } else if is_single {
                eprintln!(
                    "search: Q{:.1} → metric {:.3} (band {}..{}, {} attempt{})",
                    result.quality,
                    result.metric,
                    band.min,
                    band.max,
                    result.attempts_used,
                    if result.attempts_used == 1 { "" } else { "s" },
                );
            }
            result.jpeg
        } else {
            let config = build_encoder_config(args, quality, subsampling, &extras);
            config
                .encode_bytes(&pixels_u8, width, height, PixelLayout::Rgb8Srgb)
                .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?
        };
        let size = jpeg.len() as u64;
        (jpeg, size)
    };

    let input_size = data.len() as u64;

    if args.skip_if_larger && output_size >= input_size {
        return Ok((input_size, true));
    }

    let output_path = output_config.resolve(path, is_single)?;
    output_config.check_writable(&output_path, path)?;

    if !output_config.dry_run {
        OutputConfig::ensure_parent(&output_path)?;
        std::fs::write(&output_path, &output_jpeg)
            .with_context(|| format!("failed to write '{}'", output_path.display()))?;
    }

    Ok((output_size, false))
}

// ============================================================================
// Layout computation
// ============================================================================

/// Resolved layout for the lossy pipeline.
struct LayoutResult {
    source_crop: Option<CropRect>,
    target_w: u32,
    target_h: u32,
    needs_resize: bool,
    padding: Option<PadValues>,
}

#[derive(Clone, Copy)]
struct CropRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

struct PadValues {
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    color: [u8; 3],
}

fn build_layout(args: &ProcessArgs, source_w: u32, source_h: u32) -> Result<Option<LayoutResult>> {
    let (target_w, target_h) = args.resolve_dimensions()?;

    let has_spatial =
        target_w.is_some() || target_h.is_some() || args.crop.is_some() || args.pad.is_some();
    if !has_spatial {
        return Ok(None);
    }

    // Explicit source crop (pixel-only `x,y,w,h`). Chains with later Fit.
    let source_crop = if let Some(ref s) = args.crop {
        Some(parse_crop_pixels(s)?)
    } else {
        None
    };

    // Dimensions of the source-as-seen-by-Fit: post-crop if explicit, else source.
    let fit_src_w = source_crop.map(|c| c.w).unwrap_or(source_w);
    let fit_src_h = source_crop.map(|c| c.h).unwrap_or(source_h);

    // Resolve the Fit mode + target dims. `--no-upscale` forces Within on any mode
    // that would otherwise upscale (Fit/Cover). For `--cover --no-upscale`: if the
    // source already fits inside the target on both axes, skip the crop and resize
    // (matches zenlayout's old `WithinCrop` intent).
    let (resize_to, aspect_crop) = if let (Some(tw), Some(th)) = (target_w, target_h) {
        resolve_fit(args, fit_src_w, fit_src_h, tw, th)
    } else if let Some(tw) = target_w {
        // Width-only: derive height from aspect. `FitMode::Fit` at
        // `(fit_src_w, fit_src_h) → (tw, u32::MAX)` gives a width-bound result.
        let (w, h) = fit_dims(fit_src_w, fit_src_h, tw, u32::MAX, FitMode::Fit);
        (Some((w, h)), None)
    } else if let Some(th) = target_h {
        let (w, h) = fit_dims(fit_src_w, fit_src_h, u32::MAX, th, FitMode::Fit);
        (Some((w, h)), None)
    } else {
        (None, None)
    };

    // Combine explicit crop and aspect crop (from Cover) into a single source_region.
    let combined_crop = match (source_crop, aspect_crop) {
        (Some(c), Some(a)) => Some(CropRect {
            x: c.x + a.x,
            y: c.y + a.y,
            w: a.w,
            h: a.h,
        }),
        (Some(c), None) => Some(c),
        (None, Some(a)) => Some(a),
        (None, None) => None,
    };

    let (target_w, target_h) = resize_to.unwrap_or((fit_src_w, fit_src_h));
    let needs_resize = target_w != fit_src_w || target_h != fit_src_h;

    // Post-resize padding (--pad T,R,B,L, pixels only, black fill).
    let padding = if let Some(ref s) = args.pad {
        let trbl = coord::parse_trbl(s)?;
        if trbl.top.pixels > 0
            || trbl.right.pixels > 0
            || trbl.bottom.pixels > 0
            || trbl.left.pixels > 0
        {
            Some(PadValues {
                top: trbl.top.pixels.max(0) as u32,
                right: trbl.right.pixels.max(0) as u32,
                bottom: trbl.bottom.pixels.max(0) as u32,
                left: trbl.left.pixels.max(0) as u32,
                color: [0, 0, 0],
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(Some(LayoutResult {
        source_crop: combined_crop,
        target_w,
        target_h,
        needs_resize,
        padding,
    }))
}

/// Resolve the fit mode + target dims for a two-axis constraint.
///
/// Returns `(Some(resize_to), Option<aspect_crop>)`. The aspect crop is set
/// only for Cover — it's in coordinates relative to `fit_src_w × fit_src_h`
/// (i.e. post-explicit-crop).
fn resolve_fit(
    args: &ProcessArgs,
    fit_src_w: u32,
    fit_src_h: u32,
    tw: u32,
    th: u32,
) -> (Option<(u32, u32)>, Option<CropRect>) {
    let mode = match args.fit {
        FitArg::Within => FitMode::Within,
        FitArg::Fit => FitMode::Fit,
        FitArg::Cover => FitMode::Cover,
        FitArg::Stretch => FitMode::Stretch,
    };

    // Cover needs an aspect-aligned source crop so the resize exactly fills
    // the target without stretching.
    let aspect_crop = if matches!(mode, FitMode::Cover) {
        let (ax, ay, aw, ah) = fit_cover_source_crop(fit_src_w, fit_src_h, tw, th);
        (aw > 0 && ah > 0).then_some(CropRect {
            x: ax,
            y: ay,
            w: aw,
            h: ah,
        })
    } else {
        None
    };

    let (w, h) = fit_dims(fit_src_w, fit_src_h, tw, th, mode);
    (Some((w, h)), aspect_crop)
}

/// Parse `--crop x,y,w,h` as four pixel values.
fn parse_crop_pixels(s: &str) -> Result<CropRect> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 4 {
        anyhow::bail!(
            "crop rect requires 4 comma-separated pixel values (x,y,w,h), got {}",
            parts.len()
        );
    }
    let parse_u32 = |p: &str| -> Result<u32> {
        p.parse().map_err(|_| {
            anyhow::anyhow!("invalid crop value '{p}' (expected a non-negative integer)")
        })
    };
    Ok(CropRect {
        x: parse_u32(parts[0])?,
        y: parse_u32(parts[1])?,
        w: parse_u32(parts[2])?,
        h: parse_u32(parts[3])?,
    })
}

// ============================================================================
// Encode params
// ============================================================================

fn determine_encode_params(
    args: &ProcessArgs,
    probe: &detect::JpegProbe,
) -> Result<(Quality, ChromaSubsampling)> {
    // Explicit quality/distance: pass through as-is (user knows what they want)
    if let Some(q) = args.quality {
        let sub = user_subsampling(args, probe);
        return Ok((Quality::ApproxJpegli(q), sub));
    }
    if let Some(d) = args.distance {
        let sub = user_subsampling(args, probe);
        return Ok((Quality::ApproxButteraugli(d), sub));
    }

    // Auto quality detection from source.
    //
    // Two modes:
    // 1. Explicit --tolerance: absolute BA delta, uses legacy absolute grid.
    //    Also used by --crush presets which define fixed tolerances.
    // 2. Default: proportional tolerance (factor=0.15), uses calibrated
    //    proportional grid. Allow ba_delta ≤ src_ba × 0.15. This keeps
    //    perceptual impact constant regardless of source quality — a Q30
    //    source (BA~4.2) allows ~0.63 BA delta, while a Q90 source (BA~1.7)
    //    allows ~0.26. The proportional grid was calibrated with per-image
    //    adaptive thresholds (80% pass rate), making it more robust than
    //    converting to absolute tolerance.
    let use_absolute = args.tolerance.is_some() || args.crush.is_some();

    // Per-preset quality offset: compensate for R-D efficiency differences.
    // The calibration grids were built with auto_optimize (hybrid-prog).
    let preset_offset = preset_quality_offset(args.preset);
    if preset_offset >= 5.0
        && let Some(preset) = args.preset
        && let Some(name) = preset.to_possible_value().map(|v| v.get_name().to_string())
    {
        eprintln!("info: preset '{name}' needs +{preset_offset:.0} Q for equivalent quality");
    }

    let reencode_result = if use_absolute {
        let tolerance = if let Some(t) = args.tolerance {
            t
        } else {
            args.crush.unwrap().tolerance()
        };
        probe.reencode_settings(tolerance)
    } else {
        // Default: proportional factor 0.15 (15% of source BA)
        probe.reencode_settings_proportional(0.15)
    };

    let (quality, sub) = match reencode_result {
        Ok(settings) => {
            let mut q = offset_quality(settings.quality, preset_offset);

            // When --skip-if-larger is set, cap quality at the shrink ceiling
            // so we produce the best quality that still fits in the source size.
            if args.skip_if_larger {
                q = apply_shrink_cap(q, &settings);
            }

            let sub = if let Some(user_sub) = args.subsampling {
                convert_subsampling(user_sub)
            } else {
                settings.subsampling
            };
            (q, sub)
        }
        Err(detect::ReencodeError::ToleranceTooTight { best_effort, .. }) => {
            // Use best_effort (Q97 + matching subsampling) instead of falling
            // back to a different tolerance that could produce worse quality.
            let mut q = offset_quality(best_effort.quality, preset_offset);

            if args.skip_if_larger {
                q = apply_shrink_cap(q, &best_effort);
            }

            let sub = if let Some(user_sub) = args.subsampling {
                convert_subsampling(user_sub)
            } else {
                best_effort.subsampling
            };
            eprintln!("warning: tolerance is tighter than achievable; using best effort");
            (q, sub)
        }
        Err(detect::ReencodeError::InvalidTolerance) => {
            anyhow::bail!("invalid tolerance");
        }
        Err(e) => {
            anyhow::bail!("reencode settings error: {e}");
        }
    };

    // Quality floor: prevent catastrophic generation loss.
    let quality = apply_quality_floor(quality, probe);

    let (min_q, max_q) = args.resolve_quality_range()?;
    let quality = clamp_quality(quality, min_q, max_q);
    Ok((quality, sub))
}

/// Per-preset quality offset to compensate for R-D efficiency differences.
///
/// The calibration grids were built with auto_optimize (hybrid-prog). Other
/// presets have different R-D efficiency at the same Q value:
///
/// - Hybrid: calibrated baseline, no offset needed
/// - Jpegli (no trellis): slightly less efficient coding, +1 Q
/// - Mozjpeg (no AQ, Robidoux tables): less perceptually optimized, +3 Q
///
/// Offsets are conservative: better to produce slightly larger files than to
/// risk visible quality degradation from under-quantizing.
fn preset_quality_offset(preset: Option<crate::PresetArg>) -> f32 {
    // Calibrated on 10 gb82 images × 3 encoder families × 6 source qualities.
    // Offset = additional Q points needed to match auto_optimize butteraugli delta.
    // See reencode_calibration --preset-offsets for methodology.
    use crate::PresetArg::*;
    match preset {
        // auto_optimize (hybrid-prog λ=14.5) — calibrated baseline
        None => 0.0,
        // HybridMax ≈ auto_optimize (same tables+trellis, measured +0.1 at tol 0.5)
        Some(HybridMax) => 0.0,
        // Hybrid/HybridProg — very close to auto (measured +1.0 at tol 0.5)
        Some(Hybrid | HybridProg) => 1.0,
        // Jpegli — AQ but no trellis (measured +2.9 at tol 0.5, +2.5 at tol 1.0)
        Some(Jpegli | JpegliProg) => 2.0,
        // Mozjpeg — Robidoux tables, fundamentally different R-D curve
        // (measured +10.5 at tol 0.5, +10.7 at tol 1.0)
        Some(Mozjpeg | MozjpegProg) => 10.0,
        // MozjpegMax — similar to mozjpeg (measured +7.7 at tol 0.5, +13.3 at tol 1.0)
        Some(MozjpegMax) => 10.0,
    }
}

/// Cap quality at the shrink ceiling from calibration data.
///
/// When `--skip-if-larger` is set, we prefer producing the best quality that
/// still fits within the source file size rather than encoding at the
/// tolerance-recommended Q and then skipping because the file grew.
///
/// If no quality level can guarantee a smaller file (e.g., very low-quality
/// mozjpeg sources), the quality is left unchanged — the skip-if-larger check
/// at output time will handle it.
fn apply_shrink_cap(quality: Quality, settings: &detect::ReencodeSettings) -> Quality {
    let q = match quality {
        Quality::ApproxJpegli(q) => q,
        _ => return quality,
    };

    match settings.shrink_cap {
        Some(Quality::ApproxJpegli(cap)) if q > cap => {
            eprintln!(
                "info: capping quality {q:.0} → {cap:.0} for --skip-if-larger \
                 (source encoder is efficient at this quality)"
            );
            Quality::ApproxJpegli(cap)
        }
        None => {
            // No shrink cap means no Q produces smaller files.
            // Leave quality unchanged; skip-if-larger will catch it at output.
            quality
        }
        _ => quality,
    }
}

/// Apply a quality offset, clamping to valid range.
fn offset_quality(quality: Quality, offset: f32) -> Quality {
    if offset == 0.0 {
        return quality;
    }
    match quality {
        Quality::ApproxJpegli(q) => Quality::ApproxJpegli((q + offset).clamp(20.0, 97.0)),
        other => other,
    }
}

/// Quality floor based on detected source quality.
///
/// Prevents catastrophic generation loss when re-encoding without explicit
/// quality. Never lets the recommended Q drop more than 15 points below the
/// source's approximate equivalent zenjpeg Q.
fn apply_quality_floor(quality: Quality, probe: &detect::JpegProbe) -> Quality {
    let q = match quality {
        Quality::ApproxJpegli(q) => q,
        _ => return quality,
    };

    // Estimate source's equivalent zenjpeg Q
    let source_equiv_q = match probe.quality.scale {
        QualityScale::IjgQuality | QualityScale::MozjpegQuality => {
            // IJG/mozjpeg Q maps approximately to zenjpeg Q
            probe.quality.value
        }
        QualityScale::ButteraugliDistance => {
            // Distance → approximate Q: d=1.0≈Q90, d=2.0≈Q80, d=3.0≈Q70
            let d = probe.quality.value;
            if d <= 0.5 {
                97.0
            } else if d <= 1.0 {
                93.0 - (d - 0.5) * 6.0
            } else if d <= 2.0 {
                90.0 - (d - 1.0) * 10.0
            } else if d <= 4.0 {
                80.0 - (d - 2.0) * 5.0
            } else {
                60.0
            }
        }
        // Unknown scale — assume moderate quality, conservative floor
        _ => 75.0,
    };

    // Floor = source equivalent - 15, minimum 20
    let floor = (source_equiv_q - 15.0).max(20.0);

    if q < floor {
        eprintln!(
            "warning: raising quality from {q:.0} to {floor:.0} \
             (source is ~Q{source_equiv_q:.0}, floor prevents generation loss)"
        );
        Quality::ApproxJpegli(floor)
    } else {
        quality
    }
}

fn user_subsampling(args: &ProcessArgs, probe: &detect::JpegProbe) -> ChromaSubsampling {
    if let Some(user_sub) = args.subsampling {
        convert_subsampling(user_sub)
    } else {
        match probe.subsampling {
            Subsampling::S444 => ChromaSubsampling::None,
            Subsampling::S422 => ChromaSubsampling::HalfHorizontal,
            Subsampling::S420 => ChromaSubsampling::Quarter,
            _ => ChromaSubsampling::Quarter,
        }
    }
}

fn convert_subsampling(sub: SubsamplingArg) -> ChromaSubsampling {
    match sub {
        SubsamplingArg::S444 => ChromaSubsampling::None,
        SubsamplingArg::S422 => ChromaSubsampling::HalfHorizontal,
        SubsamplingArg::S420 => ChromaSubsampling::Quarter,
    }
}

fn convert_icc_target(target: IccTargetArg) -> TargetColorSpace {
    match target {
        IccTargetArg::Srgb => TargetColorSpace::Srgb,
        IccTargetArg::P3 => TargetColorSpace::DisplayP3,
        IccTargetArg::Rec2020 => TargetColorSpace::Rec2020,
    }
}

fn clamp_quality(quality: Quality, min_q: f32, max_q: f32) -> Quality {
    match quality {
        Quality::ApproxJpegli(v) => Quality::ApproxJpegli(v.clamp(min_q, max_q)),
        Quality::ApproxMozjpeg(v) => Quality::ApproxMozjpeg((v as f32).clamp(min_q, max_q) as u8),
        _ => quality,
    }
}

// ============================================================================
// Encoder config
// ============================================================================

fn build_encoder_config(
    args: &ProcessArgs,
    quality: Quality,
    subsampling: ChromaSubsampling,
    extras: &Option<zenjpeg::decoder::DecodedExtras>,
) -> EncoderConfig {
    let mut config = if args.xyb {
        EncoderConfig::xyb(quality, XybSubsampling::BQuarter)
    } else {
        EncoderConfig::ycbcr(quality, subsampling)
    };

    // Preset sets the base optimization profile; individual flags override below.
    if let Some(preset) = args.preset {
        config = config.optimization(match preset {
            crate::PresetArg::Jpegli => OptimizationPreset::JpegliBaseline,
            crate::PresetArg::JpegliProg => OptimizationPreset::JpegliProgressive,
            crate::PresetArg::Mozjpeg => OptimizationPreset::MozjpegBaseline,
            crate::PresetArg::MozjpegProg => OptimizationPreset::MozjpegProgressive,
            crate::PresetArg::MozjpegMax => OptimizationPreset::MozjpegMaxCompression,
            crate::PresetArg::Hybrid => OptimizationPreset::HybridBaseline,
            crate::PresetArg::HybridProg => OptimizationPreset::HybridProgressive,
            crate::PresetArg::HybridMax => OptimizationPreset::HybridMaxCompression,
        });
    } else {
        match args.trellis {
            TrellisArg::Off => {}
            TrellisArg::On => {
                #[cfg(feature = "trellis")]
                {
                    config = config.trellis(true);
                }
            }
            TrellisArg::Hybrid => {
                config = config.auto_optimize(true);
            }
        }
    }

    // Quant tables override preset's table selection
    if let Some(qt) = args.quant {
        config = config.quant_table_config(match qt {
            crate::QuantTablesArg::Jpegli => QuantTableConfig::Jpegli,
            crate::QuantTablesArg::Mozjpeg => QuantTableConfig::MozjpegRobidoux,
        });
    }

    // Chroma table layout (applied after quant_tables)
    if let Some(n) = args.chroma_tables {
        config = config.separate_chroma_tables(n == 3);
    }

    match args.structure {
        StructureArg::Progressive => config = config.progressive(true),
        StructureArg::Baseline => config = config.progressive(false),
        StructureArg::Auto => {}
    }
    if args.optimize_scans {
        config = config.progressive(ProgressiveScanMode::ProgressiveSearch);
    }
    if args.sharp_yuv {
        config = config.sharp_yuv(true);
    }
    if args.blur > 0.0 {
        config = config.pre_blur(args.blur);
    }

    let apply_icc_strips_icc = args.apply_icc.is_some();

    if let Some(extras) = extras {
        let (mut strip_exif, mut strip_icc, mut strip_xmp, strip_gainmaps) = args.strip_mask();
        if apply_icc_strips_icc {
            strip_icc = true;
        }

        if strip_exif && strip_icc && strip_xmp && strip_gainmaps {
            // Strip everything — no segments added.
        } else if strip_gainmaps {
            // Keep non-MPF metadata but rebuild from scratch to drop MPF/gain-map segments.
            if !strip_icc && let Some(icc) = extras.icc_profile() {
                config = config.add_segment(0xE2, build_icc_segment(icc));
            }
            if !strip_exif && let Some(exif) = extras.exif() {
                config = config.add_segment(0xE1, build_exif_segment(exif));
            }
            if !strip_xmp && let Some(xmp) = extras.xmp() {
                config = config.add_segment(0xE1, build_xmp_segment(xmp));
            }
        } else if strip_exif || strip_icc || strip_xmp {
            let segments = extras.to_encoder_segments_filtered(|seg| {
                if strip_exif && seg.segment_type == SegmentType::Exif {
                    return false;
                }
                if strip_icc && seg.segment_type == SegmentType::Icc {
                    return false;
                }
                if strip_xmp
                    && (seg.segment_type == SegmentType::Xmp
                        || seg.segment_type == SegmentType::XmpExtended)
                {
                    return false;
                }
                true
            });
            config = config.with_segments(segments);
        } else {
            config = config.with_segments(extras.to_encoder_segments());
        }
    }

    config
}

fn build_icc_segment(icc: &[u8]) -> Vec<u8> {
    let header = b"ICC_PROFILE\0";
    let mut seg = Vec::with_capacity(header.len() + 2 + icc.len());
    seg.extend_from_slice(header);
    seg.push(1);
    seg.push(1);
    seg.extend_from_slice(icc);
    seg
}

fn build_exif_segment(exif: &[u8]) -> Vec<u8> {
    let header = b"Exif\0\0";
    let mut seg = Vec::with_capacity(header.len() + exif.len());
    seg.extend_from_slice(header);
    seg.extend_from_slice(exif);
    seg
}

fn build_xmp_segment(xmp: &str) -> Vec<u8> {
    let header = b"http://ns.adobe.com/xap/1.0/\0";
    let mut seg = Vec::with_capacity(header.len() + xmp.len());
    seg.extend_from_slice(header);
    seg.extend_from_slice(xmp.as_bytes());
    seg
}

// ============================================================================
// Deblock
// ============================================================================

fn apply_deblock(
    pixels: &mut [f32],
    width: usize,
    height: usize,
    probe: &detect::JpegProbe,
    args: &ProcessArgs,
) {
    let content = classify_from_probe(probe);

    let action = match args.deblock {
        DeblockArg::Boundary => DeblockAction::Boundary4Tap,
        DeblockArg::On | DeblockArg::Off => {
            // `On` runs the auto-recommender; `Off` never gets here (caller gates on
            // `args.deblock != Off`), but keep the match exhaustive.
            let zero_ac_frac = estimate_zero_ac_frac(probe);
            let rec = recommend_deblock(probe, content, zero_ac_frac);
            rec.action
        }
    };

    if !matches!(action, DeblockAction::Boundary4Tap) {
        return;
    }

    let dc_quant = probe.dqt_tables.first().map(|t| t.values[0]).unwrap_or(1);
    let strength = BoundaryStrength::from_dc_quant(dc_quant);
    let num_channels = 3;

    for ch in 0..num_channels {
        let mut plane: Vec<f32> = pixels
            .iter()
            .skip(ch)
            .step_by(num_channels)
            .copied()
            .collect();

        filter_plane_boundary_4tap(&mut plane, width, height, strength);

        for (i, &val) in plane.iter().enumerate() {
            pixels[i * num_channels + ch] = val;
        }
    }
}

fn estimate_zero_ac_frac(probe: &detect::JpegProbe) -> f32 {
    let q = match probe.quality.scale {
        QualityScale::IjgQuality | QualityScale::MozjpegQuality => probe.quality.value,
        QualityScale::ButteraugliDistance => 100.0 - probe.quality.value * 10.0,
        _ => 75.0,
    };
    (1.0 - q / 100.0).clamp(0.1, 0.9) * 0.8 + 0.1
}

// ============================================================================
// Pixel operations (crop, resize, pad)
// ============================================================================

fn crop_u8_pixels(
    pixels: &[u8],
    src_width: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let channels = 3;
    let src_stride = src_width * channels;
    let dst_stride = w * channels;
    let mut out = vec![0u8; dst_stride * h];
    for row in 0..h {
        let src_offset = (y + row) * src_stride + x * channels;
        let dst_offset = row * dst_stride;
        out[dst_offset..dst_offset + dst_stride]
            .copy_from_slice(&pixels[src_offset..src_offset + dst_stride]);
    }
    out
}

fn crop_f32_pixels(
    pixels: &[f32],
    src_width: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Vec<f32> {
    let channels = 3;
    let src_stride = src_width * channels;
    let dst_stride = w * channels;
    let mut out = vec![0.0f32; dst_stride * h];
    for row in 0..h {
        let src_offset = (y + row) * src_stride + x * channels;
        let dst_offset = row * dst_stride;
        out[dst_offset..dst_offset + dst_stride]
            .copy_from_slice(&pixels[src_offset..src_offset + dst_stride]);
    }
    out
}

fn resize_u8(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    args: &ProcessArgs,
) -> Result<Vec<u8>> {
    use zenresize::{PixelDescriptor, ResizeConfig, Resizer};

    let filter = args.down_filter.unwrap_or(args.filter).to_zenresize();
    let config = ResizeConfig::builder(src_w, src_h, dst_w, dst_h)
        .filter(filter)
        .format(PixelDescriptor::RGB8_SRGB)
        .sharpen(args.sharpen)
        .linear()
        .build();

    let mut resizer = Resizer::new(&config);
    Ok(resizer.resize(pixels))
}

fn resize_f32(
    pixels: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    args: &ProcessArgs,
) -> Result<Vec<f32>> {
    use zenresize::{PixelDescriptor, ResizeConfig, Resizer};

    let filter = args.down_filter.unwrap_or(args.filter).to_zenresize();
    let config = ResizeConfig::builder(src_w, src_h, dst_w, dst_h)
        .filter(filter)
        .format(PixelDescriptor::RGBF32_LINEAR)
        .sharpen(args.sharpen)
        .build();

    let mut resizer = Resizer::new(&config);
    Ok(resizer.resize_f32(pixels))
}

fn pad_u8_pixels(pixels: &[u8], width: usize, height: usize, pad: &PadValues) -> Vec<u8> {
    let channels = 3;
    let new_w = width + pad.left as usize + pad.right as usize;
    let new_h = height + pad.top as usize + pad.bottom as usize;
    let mut out = vec![0u8; new_w * new_h * channels];

    // Fill with background color
    for pixel in out.chunks_exact_mut(channels) {
        pixel[0] = pad.color[0];
        pixel[1] = pad.color[1];
        pixel[2] = pad.color[2];
    }

    // Copy content into padded buffer
    let src_stride = width * channels;
    let dst_stride = new_w * channels;
    for row in 0..height {
        let src_offset = row * src_stride;
        let dst_offset = (pad.top as usize + row) * dst_stride + pad.left as usize * channels;
        out[dst_offset..dst_offset + src_stride]
            .copy_from_slice(&pixels[src_offset..src_offset + src_stride]);
    }

    out
}

fn pad_f32_pixels(pixels: &[f32], width: usize, height: usize, pad: &PadValues) -> Vec<f32> {
    let channels = 3;
    let new_w = width + pad.left as usize + pad.right as usize;
    let new_h = height + pad.top as usize + pad.bottom as usize;
    let mut out = vec![0.0f32; new_w * new_h * channels];

    // Fill with linear background color (approximate sRGB→linear)
    let to_linear = |v: u8| -> f32 {
        let s = v as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let bg = [
        to_linear(pad.color[0]),
        to_linear(pad.color[1]),
        to_linear(pad.color[2]),
    ];
    for pixel in out.chunks_exact_mut(channels) {
        pixel[0] = bg[0];
        pixel[1] = bg[1];
        pixel[2] = bg[2];
    }

    // Copy content
    let src_stride = width * channels;
    let dst_stride = new_w * channels;
    for row in 0..height {
        let src_offset = row * src_stride;
        let dst_offset = (pad.top as usize + row) * dst_stride + pad.left as usize * channels;
        out[dst_offset..dst_offset + src_stride]
            .copy_from_slice(&pixels[src_offset..src_offset + src_stride]);
    }

    out
}

// ============================================================================
// Target-band search helpers
// ============================================================================

/// Convert a tightly-packed linear-f32 RGB buffer (nominal `[0, 1]`, clamped)
/// to tightly-packed sRGB u8 using the same transfer function zenjpeg's decoder
/// inverts when producing `OutputTarget::LinearF32` — so this is the exact
/// inverse of the decoder's linearization step. Used on the f32 search path
/// (`--deblock`) to build a metric reference that lives in the same sRGB u8
/// domain as the search's decoded candidates.
fn linear_f32_to_srgb_u8(pixels_f32: &[f32], w: u32, h: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize) * 3;
    debug_assert!(pixels_f32.len() >= n);
    pixels_f32
        .iter()
        .take(n)
        .map(|&v| {
            let s = linear_srgb::default::linear_to_srgb(v.clamp(0.0, 1.0));
            (s * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Return the active `--search-*` target, if any.
///
/// `--search-ssim2` wins over `--search-distance` if both are somehow set
/// (the clap `quality_target` group prevents this at the CLI level).
fn search_target(args: &ProcessArgs) -> Option<(crate::search::Band, crate::search::Metric)> {
    if let Some(band) = args.search_ssim2 {
        return Some((band, crate::search::Metric::Ssim2));
    }
    if let Some(band) = args.search_distance {
        return Some((band, crate::search::Metric::Distance));
    }
    None
}
