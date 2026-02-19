use std::path::Path;

use anyhow::{Context, Result};
use zenjpeg::deblock::{filter_plane_boundary_4tap, BoundaryStrength};
use zenjpeg::decoder::{DecodeConfig, OutputTarget, PreserveConfig};
use zenjpeg::detect::content::{classify_from_probe, recommend_deblock, DeblockAction};
use zenjpeg::detect::{self, QualityScale};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Exif, PixelLayout, Quality};
use zenjpeg::decoder::Subsampling;

use crate::batch::{self, BatchSummary, FileResult};
use crate::output::OutputConfig;
use crate::OptimizeArgs;
use crate::SubsamplingArg;

pub fn run(args: OptimizeArgs) -> Result<()> {
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
                    let result = optimize_one(path, &args, &output_config, is_single);
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
            let r = optimize_one(path, &args, &output_config, is_single);
            summary.add(r);
        }
    }

    if args.report {
        summary.print_report();
    }

    if let Some(ref csv_path) = args.csv {
        summary.write_csv(csv_path)?;
    }

    // Print errors to stderr
    for r in &summary.results {
        if let Some(ref err) = r.error {
            eprintln!("error: {}: {err}", r.path.display());
        }
    }

    Ok(())
}

fn optimize_one(
    path: &Path,
    args: &OptimizeArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> FileResult {
    let input_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    match optimize_inner(path, args, output_config, is_single) {
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

fn optimize_inner(
    path: &Path,
    args: &OptimizeArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<(u64, bool)> {
    let data =
        std::fs::read(path).with_context(|| format!("failed to read '{}'", path.display()))?;

    // Step 1: Probe source JPEG
    let probe = detect::probe(&data).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Step 2: Determine quality and subsampling
    let (quality, subsampling) = determine_encode_params(args, &probe)?;

    // Step 3: Build encoder config
    let config = build_encoder_config(args, quality, subsampling);

    // Step 4: Decode
    let need_deblock = args.deblock || args.deblock_boundary;
    let decode_to_f32 = need_deblock;

    let mut decoder = DecodeConfig::new().preserve(PreserveConfig::all());
    if decode_to_f32 {
        decoder.output_target = OutputTarget::SrgbF32;
    }
    if args.auto_orient {
        decoder = decoder.auto_orient(true);
    }

    let mut result = decoder
        .decode(&data, enough::Unstoppable)
        .map_err(|e| anyhow::anyhow!("decode failed: {e}"))?;

    let width = result.width();
    let height = result.height();

    // Extract metadata before consuming pixels
    let extras = result.take_extras();
    let icc_data = if !args.strip_all && !args.strip_icc {
        extras.as_ref().and_then(|e| e.icc_profile()).map(|b| b.to_vec())
    } else {
        None
    };
    let exif_data = if !args.strip_all && !args.strip_exif {
        extras.as_ref().and_then(|e| e.exif()).map(|b| b.to_vec())
    } else {
        None
    };
    let xmp_data = if !args.strip_all {
        extras
            .as_ref()
            .and_then(|e| e.xmp())
            .map(|s| s.as_bytes().to_vec())
    } else {
        None
    };

    // Step 5: Build request with metadata
    let mut req = config.request();
    if let Some(ref icc) = icc_data {
        req = req.icc_profile(icc);
    }
    if let Some(ref exif) = exif_data {
        req = req.exif(Exif::raw(exif.clone()));
    }
    if let Some(ref xmp) = xmp_data {
        req = req.xmp(xmp);
    }

    // Step 6: Encode (with optional deblocking)
    let output_jpeg = if decode_to_f32 {
        let mut pixels_f32 = result
            .into_pixels_f32()
            .context("expected f32 decode output")?;

        if need_deblock {
            apply_deblock(
                &mut pixels_f32,
                width as usize,
                height as usize,
                &probe,
                args,
            );
        }

        // Re-encode from f32 data — f32 pixels are in [0,1] sRGB range, 3 channels
        let pixel_bytes: &[u8] = bytemuck::cast_slice(&pixels_f32);
        req.encode_bytes(pixel_bytes, width, height, PixelLayout::RgbF32Linear)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?
    } else {
        let pixels_u8 = result.pixels_u8().context("expected u8 decode output")?;
        req.encode_bytes(pixels_u8, width, height, PixelLayout::Rgb8Srgb)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?
    };

    let output_size = output_jpeg.len() as u64;
    let input_size = data.len() as u64;

    // Step 7: Skip if larger
    if args.skip_if_larger && output_size >= input_size {
        return Ok((input_size, true));
    }

    // Step 8: Write output
    let output_path = output_config.resolve(path, is_single)?;
    output_config.check_writable(&output_path, path)?;

    if !output_config.dry_run {
        OutputConfig::ensure_parent(&output_path)?;
        std::fs::write(&output_path, &output_jpeg)
            .with_context(|| format!("failed to write '{}'", output_path.display()))?;
    }

    Ok((output_size, false))
}

fn determine_encode_params(
    args: &OptimizeArgs,
    probe: &detect::JpegProbe,
) -> Result<(Quality, ChromaSubsampling)> {
    // User-specified quality overrides everything
    if let Some(q) = args.quality {
        let sub = user_subsampling(args, probe);
        return Ok((Quality::ApproxJpegli(q), sub));
    }
    if let Some(d) = args.distance {
        let sub = user_subsampling(args, probe);
        return Ok((Quality::ApproxButteraugli(d), sub));
    }

    // Determine BA tolerance
    let tolerance = if let Some(t) = args.tolerance {
        t
    } else if let Some(crush) = args.crush {
        crush.tolerance()
    } else {
        0.3 // default
    };

    // Use probe-based smart detection
    match probe.reencode_settings(tolerance) {
        Ok(settings) => {
            let quality = settings.quality;
            let sub = if let Some(user_sub) = args.subsampling {
                convert_subsampling(user_sub)
            } else {
                settings.subsampling
            };
            let quality = clamp_quality(quality, args.min_quality, args.max_quality);
            Ok((quality, sub))
        }
        Err(_) => {
            // Tolerance too tight — fall back to recommended quality
            let quality = probe.recommended_quality();
            let sub = user_subsampling(args, probe);
            let quality = clamp_quality(quality, args.min_quality, args.max_quality);
            Ok((quality, sub))
        }
    }
}

fn user_subsampling(args: &OptimizeArgs, probe: &detect::JpegProbe) -> ChromaSubsampling {
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

fn clamp_quality(quality: Quality, min_q: f32, max_q: f32) -> Quality {
    match quality {
        Quality::ApproxJpegli(v) => Quality::ApproxJpegli(v.clamp(min_q, max_q)),
        Quality::ApproxMozjpeg(v) => {
            Quality::ApproxMozjpeg((v as f32).clamp(min_q, max_q) as u8)
        }
        // Don't clamp distance-based metrics
        _ => quality,
    }
}

fn build_encoder_config(
    args: &OptimizeArgs,
    quality: Quality,
    subsampling: ChromaSubsampling,
) -> EncoderConfig {
    let mut config = EncoderConfig::ycbcr(quality, subsampling);

    // auto_optimize enables hybrid trellis (best R-D)
    if !args.no_optimize {
        config = config.auto_optimize(true);
    }

    if args.progressive {
        config = config.progressive(true);
    }
    if args.baseline {
        config = config.progressive(false);
    }
    if args.sharp_yuv {
        config = config.sharp_yuv(true);
    }

    // Note: --trellis is a no-op when auto_optimize is enabled (which is the default).
    // auto_optimize uses hybrid trellis internally, which is better than standalone trellis.

    config
}

fn apply_deblock(
    pixels: &mut [f32],
    width: usize,
    height: usize,
    probe: &detect::JpegProbe,
    args: &OptimizeArgs,
) {
    let content = classify_from_probe(probe);

    let action = if args.deblock_boundary {
        DeblockAction::Boundary4Tap
    } else {
        let zero_ac_frac = estimate_zero_ac_frac(probe);
        let rec = recommend_deblock(probe, content, zero_ac_frac);
        rec.action
    };

    if !matches!(action, DeblockAction::Boundary4Tap) {
        return;
    }

    let dc_quant = probe
        .dqt_tables
        .first()
        .map(|t| t.values[0])
        .unwrap_or(1);

    let strength = BoundaryStrength::from_dc_quant(dc_quant);
    let num_channels = 3; // RGB

    // Apply deblocking per-channel (filter_plane_boundary_4tap works on single planes)
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

