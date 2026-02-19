mod batch;
mod color_parse;
mod coord;
mod info;
mod optimize;
mod output;
mod process;
mod restructure;
mod transform;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

/// Smart JPEG processor: optimize, resize, crop, transform, and inspect.
///
/// zjpeg probes source JPEGs to auto-select quality, re-encodes with perceptual
/// optimization, handles resize/crop via zenlayout, and supports lossless
/// DCT-domain transforms — all in one tool.
#[derive(Parser)]
#[command(name = "zjpeg", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    // When no subcommand, treat positional args as process inputs.
    /// Input files (when no subcommand is given, runs process)
    #[arg(global = false, trailing_var_arg = true)]
    files: Vec<PathBuf>,
}

#[derive(Subcommand)]
enum Command {
    /// Unified JPEG processing: optimize, resize, crop, transform (default).
    Process(Box<ProcessArgs>),

    /// Quick JPEG inspection and probe info.
    Info(InfoArgs),

    /// [deprecated: use `process`] Smart JPEG re-encoding.
    #[command(hide = true)]
    Optimize(OptimizeArgs),

    /// [deprecated: use `process`] Lossless DCT-domain transforms.
    #[command(hide = true)]
    Transform(TransformArgs),

    /// [deprecated: use `process`] Lossless baseline/progressive conversion.
    #[command(hide = true)]
    Restructure(RestructureArgs),
}

// ============================================================================
// Process (unified command)
// ============================================================================

#[derive(Parser)]
pub struct ProcessArgs {
    /// Input JPEG files or glob patterns.
    #[arg(required = true)]
    pub input: Vec<String>,

    // -- Output ----------------------------------------------------------
    /// Output file or directory.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Overwrite input files in-place (requires --force).
    #[arg(long)]
    pub in_place: bool,

    /// Output filename suffix (default: none).
    #[arg(long, default_value = "")]
    pub suffix: String,

    /// Allow overwriting existing files.
    #[arg(long)]
    pub force: bool,

    // -- Fit modes (mutually exclusive) ----------------------------------
    /// Fit inside target dimensions, preserving aspect ratio (may upscale).
    #[arg(long, group = "fit_mode")]
    pub contain: bool,

    /// Fill target dimensions, cropping overflow.
    #[arg(long, group = "fit_mode")]
    pub cover: bool,

    /// Stretch to exact target dimensions (distorts aspect ratio).
    #[arg(long, group = "fit_mode")]
    pub fill: bool,

    /// Fit inside target, never upscale (default when dimensions given).
    #[arg(long, group = "fit_mode")]
    pub scale_down: bool,

    /// Fit inside target with padding to exact dimensions.
    #[arg(long, group = "fit_mode")]
    pub pad: bool,

    /// Modifier: prevent upscaling with --cover or --pad.
    #[arg(long)]
    pub no_upscale: bool,

    // -- Sizing ----------------------------------------------------------
    /// Target width in pixels.
    #[arg(short, long)]
    pub width: Option<u32>,

    /// Target height in pixels.
    #[arg(short, long)]
    pub height: Option<u32>,

    /// Target size as WxH (e.g. 800x600, 800, x600).
    #[arg(long)]
    pub size: Option<String>,

    /// DPR multiplier (scales target dimensions).
    #[arg(long)]
    pub dpr: Option<f32>,

    /// Crop to aspect ratio before resize (e.g. 16:9).
    #[arg(long)]
    pub aspect_crop: Option<String>,

    // -- Spatial operations -----------------------------------------------
    /// Select a crop rectangle: x,y,w,h (px or %).
    #[arg(long)]
    pub crop: Option<String>,

    /// Trim edges (CSS TRBL shorthand, px or %).
    #[arg(long)]
    pub inset: Option<String>,

    /// Viewport region: left,top,right,bottom (px/pct/calc).
    #[arg(long)]
    pub region: Option<String>,

    /// Post-padding (CSS TRBL shorthand, px or %).
    #[arg(long)]
    pub extend: Option<String>,

    /// Anchor position for crop/pad: center, top-left, 30%,70%, etc.
    #[arg(long, default_value = "center")]
    pub position: String,

    /// Background color for padding (CSS hex or named color).
    #[arg(long)]
    pub bg: Option<String>,

    // -- Transforms -------------------------------------------------------
    /// Rotate clockwise by degrees (90, 180, 270).
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=360))]
    pub rotate: Option<u16>,

    /// Flip: h (horizontal) or v (vertical).
    #[arg(long, value_enum)]
    pub flip: Option<FlipArg>,

    /// Apply EXIF orientation and reset tag (default: on).
    #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
    pub auto_orient: bool,

    /// Disable auto-orient.
    #[arg(long, overrides_with = "auto_orient")]
    pub no_auto_orient: bool,

    // -- Encoding ---------------------------------------------------------
    /// Override quality (0-100, bypasses smart detection).
    #[arg(short, long)]
    pub quality: Option<f32>,

    /// Override butteraugli distance.
    #[arg(short, long)]
    pub distance: Option<f32>,

    /// Quality/size tradeoff preset.
    #[arg(long, value_enum)]
    pub crush: Option<CrushLevel>,

    /// Exact butteraugli tolerance (overrides --crush).
    #[arg(long)]
    pub tolerance: Option<f32>,

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

    /// Enable auto_optimize (hybrid trellis, default: on).
    #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
    pub auto_optimize: bool,

    /// Disable auto_optimize.
    #[arg(long, overrides_with = "auto_optimize")]
    pub no_optimize: bool,

    /// Enable SharpYUV chroma downsampling.
    #[arg(long)]
    pub sharp_yuv: bool,

    /// Encode in XYB color space (perceptual, requires linear decode).
    #[arg(long)]
    pub xyb: bool,

    /// Enable content-aware deblocking.
    #[arg(long)]
    pub deblock: bool,

    /// Force boundary 4-tap deblocking.
    #[arg(long)]
    pub deblock_boundary: bool,

    // -- Metadata ---------------------------------------------------------
    /// Strip all metadata.
    #[arg(long)]
    pub strip_all: bool,

    /// Strip EXIF metadata only.
    #[arg(long)]
    pub strip_exif: bool,

    /// Strip ICC profile only.
    #[arg(long)]
    pub strip_icc: bool,

    /// Strip XMP metadata only.
    #[arg(long)]
    pub strip_xmp: bool,

    /// Strip gain maps (UltraHDR).
    #[arg(long)]
    pub strip_gainmaps: bool,

    /// Keep all metadata (default).
    #[arg(long)]
    pub keep_all: bool,

    /// Apply embedded ICC profile, converting to target color space.
    #[arg(long, value_enum)]
    pub apply_icc: Option<IccTargetArg>,

    // -- Resampling -------------------------------------------------------
    /// Resize filter (default: mitchell).
    #[arg(long, value_enum, default_value = "mitchell")]
    pub filter: FilterArg,

    /// Downscale filter override.
    #[arg(long, value_enum)]
    pub down_filter: Option<FilterArg>,

    /// Upscale filter override.
    #[arg(long, value_enum)]
    pub up_filter: Option<FilterArg>,

    /// Post-resize sharpening amount (0.0 = none).
    #[arg(long, default_value = "0.0")]
    pub sharpen: f32,

    // -- Output control ---------------------------------------------------
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

    // -- Escape hatch -----------------------------------------------------
    /// RIAPI query string for advanced layout (overrides spatial flags).
    #[arg(long)]
    pub riapi: Option<String>,
}

impl ProcessArgs {
    /// Whether auto-orient is effectively enabled (--auto-orient minus --no-auto-orient).
    pub fn effective_auto_orient(&self) -> bool {
        self.auto_orient && !self.no_auto_orient
    }

    /// Whether auto-optimize is effectively enabled.
    pub fn effective_auto_optimize(&self) -> bool {
        self.auto_optimize && !self.no_optimize
    }

    /// Resolve target dimensions from --width, --height, --size, --dpr.
    pub fn resolve_dimensions(&self) -> Result<(Option<u32>, Option<u32>)> {
        let (mut w, mut h) = if let Some(ref size_str) = self.size {
            coord::parse_dimensions(size_str)?
        } else {
            (self.width, self.height)
        };

        if let Some(dpr) = self.dpr {
            if dpr <= 0.0 {
                anyhow::bail!("--dpr must be positive");
            }
            w = w.map(|v| (v as f32 * dpr).round() as u32);
            h = h.map(|v| (v as f32 * dpr).round() as u32);
        }

        Ok((w, h))
    }
}

#[derive(Clone, Copy, ValueEnum)]
pub enum FlipArg {
    /// Flip horizontally (mirror).
    H,
    /// Flip vertically.
    V,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum FilterArg {
    Mitchell,
    Lanczos,
    Lanczos2,
    CatmullRom,
    Robidoux,
    RobidouxSharp,
    Hermite,
    Box,
    Triangle,
    Fastest,
}

impl FilterArg {
    pub fn to_zenresize(self) -> zenresize::Filter {
        match self {
            Self::Mitchell => zenresize::Filter::Mitchell,
            Self::Lanczos => zenresize::Filter::Lanczos,
            Self::Lanczos2 => zenresize::Filter::Lanczos2,
            Self::CatmullRom => zenresize::Filter::CatmullRom,
            Self::Robidoux => zenresize::Filter::Robidoux,
            Self::RobidouxSharp => zenresize::Filter::RobidouxSharp,
            Self::Hermite => zenresize::Filter::Hermite,
            Self::Box => zenresize::Filter::Box,
            Self::Triangle => zenresize::Filter::Triangle,
            Self::Fastest => zenresize::Filter::Fastest,
        }
    }
}

// ============================================================================
// Shared enums (used by both Process and legacy Optimize)
// ============================================================================

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

#[derive(Clone, Copy, ValueEnum)]
pub enum IccTargetArg {
    /// Convert to sRGB (standard web color space).
    Srgb,
    /// Convert to Display P3 (wide gamut).
    P3,
    /// Convert to BT.2020/Rec.2020 (wide gamut).
    Rec2020,
}

// ============================================================================
// Legacy subcommand args (hidden, deprecated)
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

    /// Encode in XYB color space (perceptual, requires linear decode).
    #[arg(long)]
    pub xyb: bool,

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

    /// Strip XMP metadata only.
    #[arg(long)]
    pub strip_xmp: bool,

    /// Strip gain maps (UltraHDR).
    #[arg(long)]
    pub strip_gainmaps: bool,

    /// Keep all metadata (default).
    #[arg(long)]
    pub keep_all: bool,

    /// Apply embedded ICC profile, converting to the specified color space.
    #[arg(long, value_enum)]
    pub apply_icc: Option<IccTargetArg>,

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
// Info (unchanged)
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
// Restructure (hidden, deprecated)
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
        Some(Command::Process(args)) => process::run(*args),
        Some(Command::Info(args)) => info::run(args),
        Some(Command::Optimize(args)) => {
            eprintln!("warning: `optimize` is deprecated, use `process` instead");
            optimize::run(args)
        }
        Some(Command::Transform(args)) => {
            eprintln!("warning: `transform` is deprecated, use `process` instead");
            transform::run(args)
        }
        Some(Command::Restructure(args)) => {
            eprintln!("warning: `restructure` is deprecated, use `process` instead");
            restructure::run(args)
        }
        None => {
            if cli.files.is_empty() {
                // No subcommand and no files — show help
                use clap::CommandFactory;
                Cli::command().print_help()?;
                println!();
                Ok(())
            } else {
                // Treat bare files as process with defaults
                let input: Vec<String> = cli
                    .files
                    .into_iter()
                    .map(|p| p.display().to_string())
                    .collect();
                process::run(ProcessArgs {
                    input,
                    output: None,
                    in_place: false,
                    suffix: String::new(),
                    force: false,
                    contain: false,
                    cover: false,
                    fill: false,
                    scale_down: false,
                    pad: false,
                    no_upscale: false,
                    width: None,
                    height: None,
                    size: None,
                    dpr: None,
                    aspect_crop: None,
                    crop: None,
                    inset: None,
                    region: None,
                    extend: None,
                    position: "center".into(),
                    bg: None,
                    rotate: None,
                    flip: None,
                    auto_orient: true,
                    no_auto_orient: false,
                    quality: None,
                    distance: None,
                    crush: None,
                    tolerance: None,
                    max_quality: 97.0,
                    min_quality: 50.0,
                    progressive: false,
                    baseline: false,
                    subsampling: None,
                    auto_optimize: true,
                    no_optimize: false,
                    sharp_yuv: false,
                    xyb: false,
                    deblock: false,
                    deblock_boundary: false,
                    strip_all: false,
                    strip_exif: false,
                    strip_icc: false,
                    strip_xmp: false,
                    strip_gainmaps: false,
                    keep_all: false,
                    apply_icc: None,
                    filter: FilterArg::Mitchell,
                    down_filter: None,
                    up_filter: None,
                    sharpen: 0.0,
                    skip_if_larger: false,
                    report: false,
                    csv: None,
                    dry_run: false,
                    jobs: None,
                    riapi: None,
                })
            }
        }
    }
}
