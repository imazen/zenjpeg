use std::path::Path;

use anyhow::{Context, Result};
use zenjpeg::lossless::{restructure, OutputMode, RestructureConfig, RestartInterval};

use crate::batch;
use crate::output::OutputConfig;
use crate::RestructureArgs;

pub fn run(args: RestructureArgs) -> Result<()> {
    let files = batch::expand_inputs(&args.input)?;
    if files.is_empty() {
        anyhow::bail!("no JPEG files found");
    }

    let output_config = OutputConfig::new(
        args.output.clone(),
        args.in_place,
        String::new(), // no suffix for restructure
        args.force,
        false,
    )?;

    let is_single = files.len() == 1;

    let config = build_config(&args)?;

    for path in &files {
        match restructure_one(path, &config, &output_config, is_single) {
            Ok(()) => {}
            Err(e) => eprintln!("error: {}: {e:#}", path.display()),
        }
    }

    Ok(())
}

fn build_config(args: &RestructureArgs) -> Result<RestructureConfig> {
    if args.progressive && args.sequential {
        anyhow::bail!("cannot specify both --progressive and --sequential");
    }

    let output_mode = if args.progressive {
        OutputMode::Progressive
    } else {
        OutputMode::Sequential
    };

    let restart_interval = if let Some(rows) = args.restart_rows {
        RestartInterval::EveryMcuRows(rows)
    } else {
        RestartInterval::None
    };

    Ok(RestructureConfig {
        output_mode,
        restart_interval,
        ..Default::default()
    })
}

fn restructure_one(
    path: &Path,
    config: &RestructureConfig,
    output_config: &OutputConfig,
    is_single: bool,
) -> Result<()> {
    let data =
        std::fs::read(path).with_context(|| format!("failed to read '{}'", path.display()))?;

    let output = restructure(&data, config, enough::Unstoppable)
        .map_err(|e| anyhow::anyhow!("restructure failed: {e}"))?;

    let output_path = output_config.resolve(path, is_single)?;
    output_config.check_writable(&output_path, path)?;

    OutputConfig::ensure_parent(&output_path)?;
    std::fs::write(&output_path, &output)
        .with_context(|| format!("failed to write '{}'", output_path.display()))?;

    Ok(())
}
