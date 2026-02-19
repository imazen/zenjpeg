mod batch;
mod info;
mod optimize;
mod output;
mod restructure;
mod transform;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

/// Smart JPEG optimizer, transformer, and inspector.
///
/// zjpeg probes source JPEGs to auto-select quality, re-encodes with perceptual
/// optimization, and supports lossless DCT-domain transforms — all in one tool.
#[derive(Parser)]
#[command(name = "zjpeg", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // When no subcommand, treat positional args as optimize inputs.
    /// Input files (when no subcommand is given, runs optimize)
    #[arg(global = false, trailing_var_arg = true)]
    files: Vec<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Smart JPEG re-encoding with auto quality selection (default).
    Optimize(OptimizeArgs),

    /// Lossless DCT-domain transforms (rotate, flip, transpose).
    Transform(TransformArgs),

    /// Quick JPEG inspection and probe info.
    Info(InfoArgs),

    /// Lossless baseline↔progressive conversion.
    Restructure(RestructureArgs),
}

// ============================================================================
// Optimize
// ============================================================================

#[derive(Parser)]
pub struct OptimizeArgs {
    /// Input JPEG files or glob patterns.
    #[arg(required = true)]
    pub input: Vec<String>,

    /// Output file or directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite input files in-place (requires --force).
    #[arg(long)]
    pub in_place: bool,

    /// Output filename suffix (default: ".optimized").
    #[arg(long, default_value = ".optimized")]
    pub suffix: String,

    /// Allow overwriting existing files.
    #[arg(long)]
    pub force: bool,

    /// Quality/size tradeoff preset.
    #[arg(long, value_enum)]
    pub crush: Option<CrushLevel>,

    /// Exact butteraugli tolerance (overrides --crush).
    #[arg(long)]
    pub tolerance: Option<f32>,

    /// Override quality (0-100, bypasses smart detection).
    #[arg(short, long)]
    pub quality: Option<f32>,

    /// Override butteraugli distance.
    #[arg(short, long)]
    pub distance: Option<f32>,

    /// Maximum quality ceiling.
    #[arg(long, default_value = "97")]
    pub max_quality: f32,

    /// Minimum quality floor.
    #[arg(long, default_value = "50")]
    pub min_quality: f32,

    /// Force progressive output.
    #[arg(long)]
    pub progressive: bool,

    /// Force baseline output.
    #[arg(long)]
    pub baseline: bool,

    /// Force chroma subsampling.
    #[arg(long, value_enum)]
    pub subsampling: Option<SubsamplingArg>,

    /// Disable auto_optimize (hybrid trellis).
    #[arg(long)]
    pub no_optimize: bool,

    /// Enable standalone trellis quantization.
    #[arg(long)]
    pub trellis: bool,

    /// Enable SharpYUV chroma downsampling.
    #[arg(long)]
    pub sharp_yuv: bool,

    /// Enable content-aware deblocking.
    #[arg(long)]
    pub deblock: bool,

    /// Force boundary 4-tap deblocking.
    #[arg(long)]
    pub deblock_boundary: bool,

    /// Strip all metadata.
    #[arg(long)]
    pub strip_all: bool,

    /// Strip EXIF metadata only.
    #[arg(long)]
    pub strip_exif: bool,

    /// Strip ICC profile only.
    #[arg(long)]
    pub strip_icc: bool,

    /// Keep all metadata (default).
    #[arg(long)]
    pub keep_all: bool,

    /// Apply EXIF orientation and reset tag.
    #[arg(long)]
    pub auto_orient: bool,

    /// Don't write output if it would be larger than input.
    #[arg(long)]
    pub skip_if_larger: bool,

    /// Print per-file size comparison table.
    #[arg(long)]
    pub report: bool,

    /// Write CSV report to file.
    #[arg(long)]
    pub csv: Option<PathBuf>,

    /// Show what would happen without writing files.
    #[arg(long)]
    pub dry_run: bool,

    /// Number of parallel jobs (default: num_cpus / 2).
    #[arg(short, long)]
    pub jobs: Option<usize>,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum CrushLevel {
    /// Nearly imperceptible (BA tolerance 0.2).
    Gentle,
    /// Noticeable on close inspection (BA tolerance 0.5).
    Moderate,
    /// Visible but acceptable (BA tolerance 1.0).
    Aggressive,
    /// Significant quality loss (BA tolerance 2.0).
    Max,
}

impl CrushLevel {
    pub fn tolerance(self) -> f32 {
        match self {
            Self::Gentle => 0.2,
            Self::Moderate => 0.5,
            Self::Aggressive => 1.0,
            Self::Max => 2.0,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum SubsamplingArg {
    /// 4:4:4 — no subsampling.
    #[value(name = "444")]
    S444,
    /// 4:2:2 — horizontal subsampling.
    #[value(name = "422")]
    S422,
    /// 4:2:0 — quarter chroma.
    #[value(name = "420")]
    S420,
}

// ============================================================================
// Transform
// ============================================================================

#[derive(Parser)]
pub struct TransformArgs {
    /// Input JPEG files.
    #[arg(required = true)]
    pub input: Vec<String>,

    /// Output file or directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite input files in-place (requires --force).
    #[arg(long)]
    pub in_place: bool,

    /// Allow overwriting existing files.
    #[arg(long)]
    pub force: bool,

    /// Rotate clockwise by degrees.
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=360))]
    pub rotate: Option<u16>,

    /// Flip horizontally (mirror).
    #[arg(long)]
    pub flip_h: bool,

    /// Flip vertically.
    #[arg(long)]
    pub flip_v: bool,

    /// Transpose (reflect across main diagonal).
    #[arg(long)]
    pub transpose: bool,

    /// Transverse (reflect across anti-diagonal).
    #[arg(long)]
    pub transverse: bool,

    /// Apply EXIF orientation and reset tag.
    #[arg(long)]
    pub auto_orient: bool,
}

// ============================================================================
// Info
// ============================================================================

#[derive(Parser)]
pub struct InfoArgs {
    /// Input JPEG files.
    #[arg(required = true)]
    pub input: Vec<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show all details including quant tables.
    #[arg(long)]
    pub all: bool,

    /// Show quantization tables.
    #[arg(long)]
    pub quant: bool,
}

// ============================================================================
// Restructure
// ============================================================================

#[derive(Parser)]
pub struct RestructureArgs {
    /// Input JPEG files.
    #[arg(required = true)]
    pub input: Vec<String>,

    /// Output file or directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite input files in-place (requires --force).
    #[arg(long)]
    pub in_place: bool,

    /// Allow overwriting existing files.
    #[arg(long)]
    pub force: bool,

    /// Convert to progressive.
    #[arg(long)]
    pub progressive: bool,

    /// Convert to sequential (baseline).
    #[arg(long)]
    pub sequential: bool,

    /// Restart marker interval in MCU rows.
    #[arg(long)]
    pub restart_rows: Option<u16>,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Optimize(args)) => optimize::run(args),
        Some(Command::Transform(args)) => transform::run(args),
        Some(Command::Info(args)) => info::run(args),
        Some(Command::Restructure(args)) => restructure::run(args),
        None => {
            if cli.files.is_empty() {
                // No subcommand and no files — show help
                use clap::CommandFactory;
                Cli::command().print_help()?;
                println!();
                Ok(())
            } else {
                // Treat bare files as optimize
                let input: Vec<String> =
                    cli.files.into_iter().map(|p| p.display().to_string()).collect();
                optimize::run(OptimizeArgs {
                    input,
                    output: None,
                    in_place: false,
                    suffix: ".optimized".into(),
                    force: false,
                    crush: None,
                    tolerance: None,
                    quality: None,
                    distance: None,
                    max_quality: 97.0,
                    min_quality: 50.0,
                    progressive: false,
                    baseline: false,
                    subsampling: None,
                    no_optimize: false,
                    trellis: false,
                    sharp_yuv: false,
                    deblock: false,
                    deblock_boundary: false,
                    strip_all: false,
                    strip_exif: false,
                    strip_icc: false,
                    keep_all: false,
                    auto_orient: false,
                    skip_if_larger: false,
                    report: false,
                    csv: None,
                    dry_run: false,
                    jobs: None,
                })
            }
        }
    }
}
