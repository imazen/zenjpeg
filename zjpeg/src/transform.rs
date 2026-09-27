use std::path::Path;

use anyhow::{Context, Result};
use zenjpeg::lossless::{self, EdgeHandling, LosslessTransform, TransformConfig};

use crate::TransformArgs;
use crate::batch;
use crate::output::OutputConfig;

pub fn run(args: TransformArgs) -> Result<()> {
    let files = batch::expand_inputs(&args.input)?;
    if files.is_empty() {
        anyhow::bail!("no JPEG files found");
    }

    let output_config = OutputConfig::new(
        args.output.clone(),
        args.in_place,
        String::new(), // no suffix for transforms
        args.force,
        false, // no dry-run for transform
    )?;

    let is_single = files.len() == 1;

    let mut failures = 0;
    for path in &files {
        match transform_one(path, &args, &output_config, is_single) {
            Ok(()) => {}
            Err(e) => {
                failures += 1;
                eprintln!("error: {}: {e:#}", path.display());
            }
        }
    }

    if failures > 0 {
        anyhow::bail!("{failures} JPEG transform(s) failed");
    }
    Ok(())
}

fn transform_one(
    path: &Path,
    args: &TransformArgs,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<()> {
    let data =
        std::fs::read(path).with_context(|| format!("failed to read '{}'", path.display()))?;

    let edge_handling = if args.trim {
        EdgeHandling::TrimPartialBlocks
    } else {
        EdgeHandling::RejectPartialBlocks
    };
    let output = if args.auto_orient {
        lossless::apply_exif_orientation_with_edge_handling(
            &data,
            edge_handling,
            enough::Unstoppable,
        )
        .map_err(|e| anyhow::anyhow!("orientation failed: {e}"))?
    } else {
        let xform = determine_transform(args)?;
        let config = TransformConfig {
            transform: xform,
            edge_handling,
        };
        lossless::transform(&data, &config, enough::Unstoppable)
            .map_err(|e| anyhow::anyhow!("transform failed: {e}"))?
    };

    let output_path = output_config.resolve(path, is_single)?;
    output_config.check_writable(&output_path, path)?;

    OutputConfig::ensure_parent(&output_path)?;
    std::fs::write(&output_path, &output)
        .with_context(|| format!("failed to write '{}'", output_path.display()))?;

    Ok(())
}

fn determine_transform(args: &TransformArgs) -> Result<LosslessTransform> {
    if args.transpose {
        return Ok(LosslessTransform::Transpose);
    }
    if args.transverse {
        return Ok(LosslessTransform::Transverse);
    }
    if args.flip_h {
        return Ok(LosslessTransform::FlipHorizontal);
    }
    if args.flip_v {
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

    anyhow::bail!(
        "no transform specified; use --rotate, --flip-h, --flip-v, --transpose, --transverse, or --auto-orient"
    )
}
