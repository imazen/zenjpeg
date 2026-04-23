mod batch;
mod coord;
mod info;
mod optimize;
mod output;
mod process;
mod restructure;
mod search;
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

    // -- Sizing / spatial ------------------------------------------------
    /// Fit mode for resizing when width/height/size is set.
    #[arg(long, value_enum, default_value_t = FitArg::default())]
    pub fit: FitArg,

    /// Target width in pixels.
    #[arg(short, long)]
    pub width: Option<u32>,

    /// Target height in pixels.
    #[arg(long)]
    pub height: Option<u32>,

    /// Target size as WxH (e.g. 800x600, 800, x600).
    #[arg(long)]
    pub size: Option<String>,

    /// DPR multiplier (scales target dimensions).
    #[arg(long)]
    pub dpr: Option<f32>,

    /// Source crop rectangle: `x,y,w,h` in pixels.
    #[arg(long)]
    pub crop: Option<String>,

    /// Post-resize padding (CSS TRBL shorthand, pixels). Black fill.
    #[arg(long)]
    pub pad: Option<String>,

    // -- Transforms -------------------------------------------------------
    /// Rotate clockwise by degrees (90, 180, 270).
    #[arg(long, value_parser = clap::value_parser!(u16).range(0..=360))]
    pub rotate: Option<u16>,

    /// Flip: h (horizontal) or v (vertical).
    #[arg(long, value_enum)]
    pub flip: Option<FlipArg>,

    /// EXIF orientation handling.
    #[arg(long, value_enum, default_value_t = OrientArg::default())]
    pub orient: OrientArg,

    // -- Quality ---------------------------------------------------------
    /// Exact quality target (0-100, bypasses smart detection).
    #[arg(short, long, group = "quality_target")]
    pub quality: Option<f32>,

    /// Exact butteraugli distance target (alternative to --quality).
    #[arg(short, long, group = "quality_target")]
    pub distance: Option<f32>,

    /// Target SSIM2 band as `MIN..MAX` — search for smallest file in band.
    /// Example: `--search-ssim2 85..92`.
    #[arg(long, group = "quality_target")]
    pub search_ssim2: Option<search::Band>,

    /// Target butteraugli distance band as `MIN..MAX` — search for smallest file in band.
    /// Example: `--search-distance 0.8..1.5`.
    #[arg(long, group = "quality_target")]
    pub search_distance: Option<search::Band>,

    /// Max encode+measure iterations for --search-* (default 3).
    #[arg(long, default_value_t = 3)]
    pub attempts: u32,

    /// Quality/size tradeoff preset for smart detection.
    #[arg(long, value_enum, group = "quality_target")]
    pub crush: Option<CrushLevel>,

    /// Smart-mode butteraugli tolerance override (overrides --crush).
    #[arg(long)]
    pub tolerance: Option<f32>,

    /// Smart-mode quality range as `MIN:MAX` (default `50:97`).
    /// Also used as the quality search window for `--search-*`.
    #[arg(long, default_value = "50:97")]
    pub quality_range: String,

    // -- Format & structure ----------------------------------------------
    /// Output structure.
    #[arg(long, value_enum, default_value_t = StructureArg::default())]
    pub structure: StructureArg,

    /// Search 64 progressive scan scripts for smallest output (~2% smaller, ~2x slower).
    #[arg(long)]
    pub optimize_scans: bool,

    /// Force chroma subsampling.
    #[arg(long, value_enum)]
    pub subsampling: Option<SubsamplingArg>,

    /// Quantization table family.
    #[arg(long, value_enum)]
    pub quant: Option<QuantTablesArg>,

    /// Number of chroma quantization tables: 3 (separate Cb/Cr, default) or 2 (shared).
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=3))]
    pub chroma_tables: Option<u8>,

    /// Preset bundle (overrides individual tuning flags).
    #[arg(long, value_enum)]
    pub preset: Option<PresetArg>,

    /// Trellis rate-distortion optimization.
    #[arg(long, value_enum, default_value_t = TrellisArg::default())]
    pub trellis: TrellisArg,

    /// Enable SharpYUV chroma downsampling.
    #[arg(long)]
    pub sharp_yuv: bool,

    /// Encode in XYB color space (perceptual, requires linear decode).
    #[arg(long)]
    pub xyb: bool,

    /// Deblocking mode.
    #[arg(long, value_enum, default_value_t = DeblockArg::default())]
    pub deblock: DeblockArg,

    /// Pre-encode Gaussian blur sigma (0.0 = disabled).
    ///
    /// A mild blur (σ ≈ 0.4) before JPEG encoding reduces file size ~5%
    /// with negligible perceptual quality loss.
    #[arg(long, default_value = "0.0")]
    pub blur: f32,

    // -- Decoder ---------------------------------------------------------
    /// Decoder error tolerance for damaged/non-conformant JPEGs.
    #[arg(long, value_enum)]
    pub strictness: Option<StrictnessArg>,

    // -- Metadata --------------------------------------------------------
    /// Metadata to strip (comma-separated). Values: all, none, exif, icc, xmp, gainmaps.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub strip: Vec<StripArg>,

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
    /// Error instead of re-encoding when lossy pipeline would be required.
    #[arg(long)]
    pub lossless_only: bool,

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

impl ProcessArgs {
    /// Parse `--quality-range MIN:MAX`, returning `(min, max)` as f32.
    pub fn resolve_quality_range(&self) -> Result<(f32, f32)> {
        let (lo, hi) = self
            .quality_range
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("--quality-range must be MIN:MAX (e.g. 50:97)"))?;
        let lo: f32 = lo
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid quality-range MIN: '{lo}'"))?;
        let hi: f32 = hi
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid quality-range MAX: '{hi}'"))?;
        if lo > hi {
            anyhow::bail!("--quality-range MIN ({lo}) exceeds MAX ({hi})");
        }
        Ok((lo, hi))
    }

    /// Effective strip decisions: `(exif, icc, xmp, gainmaps)`.
    ///
    /// `All` turns everything on; `None` turns everything off; individual
    /// values toggle only that target. Later list entries take precedence
    /// over earlier ones (so `all,none` = keep everything; `none,exif` =
    /// strip only EXIF).
    pub fn strip_mask(&self) -> (bool, bool, bool, bool) {
        let mut exif = false;
        let mut icc = false;
        let mut xmp = false;
        let mut gm = false;
        for s in &self.strip {
            match s {
                StripArg::All => {
                    exif = true;
                    icc = true;
                    xmp = true;
                    gm = true;
                }
                StripArg::None => {
                    exif = false;
                    icc = false;
                    xmp = false;
                    gm = false;
                }
                StripArg::Exif => exif = true,
                StripArg::Icc => icc = true,
                StripArg::Xmp => xmp = true,
                StripArg::Gainmaps => gm = true,
            }
        }
        (exif, icc, xmp, gm)
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

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum PresetArg {
    /// Jpegli tables + AQ, no trellis, baseline.
    Jpegli,
    /// Jpegli tables + AQ, no trellis, progressive.
    JpegliProg,
    /// Mozjpeg Robidoux tables + trellis, no AQ, baseline.
    Mozjpeg,
    /// Mozjpeg tables + trellis, no AQ, mozjpeg progressive script.
    MozjpegProg,
    /// Mozjpeg tables + trellis + scan search + deringing, no AQ.
    MozjpegMax,
    /// Jpegli tables + trellis + AQ, baseline.
    Hybrid,
    /// Jpegli tables + trellis + AQ, progressive (default).
    HybridProg,
    /// Jpegli tables + trellis + AQ + scan search + deringing.
    HybridMax,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum QuantTablesArg {
    /// Jpegli perceptual tables (default).
    Jpegli,
    /// Mozjpeg Robidoux psychovisual tables.
    Mozjpeg,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum StrictnessArg {
    /// Fail on any spec violation.
    Strict,
    /// Match libjpeg-turbo error handling (default).
    Balanced,
    /// Recover from all errors when possible.
    Lenient,
    /// Maximum compatibility with damaged files.
    Permissive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum FitArg {
    /// Fit inside target, never upscale past source (default).
    #[default]
    Within,
    /// Fit inside target, preserve aspect; may upscale.
    Fit,
    /// Fill target, center-cropping source to target aspect.
    Cover,
    /// Stretch to exact target (ignores aspect).
    Stretch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum OrientArg {
    /// Apply EXIF orientation and reset the tag (default).
    #[default]
    Auto,
    /// Leave pixels as-is; keep the EXIF tag intact.
    Keep,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum StructureArg {
    /// Let the encoder choose based on preset / quality (default).
    #[default]
    Auto,
    /// Force progressive output.
    Progressive,
    /// Force baseline (sequential) output.
    Baseline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum TrellisArg {
    /// Disable trellis entirely.
    Off,
    /// Standalone mozjpeg-style trellis.
    On,
    /// Hybrid trellis with RD lambda tuning (default, best quality).
    #[default]
    Hybrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum DeblockArg {
    /// No deblocking (default).
    #[default]
    Off,
    /// Content-aware deblocking.
    On,
    /// Force boundary 4-tap deblocking.
    Boundary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum StripArg {
    /// Strip all metadata (EXIF, ICC, XMP, gain maps).
    All,
    /// Keep all metadata.
    None,
    Exif,
    Icc,
    Xmp,
    /// Strip UltraHDR gain map (demotes file to SDR).
    Gainmaps,
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

    /// Pre-encode Gaussian blur sigma (0.0 = disabled).
    #[arg(long, default_value = "0.0")]
    pub blur: f32,

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
                    fit: FitArg::default(),
                    width: None,
                    height: None,
                    size: None,
                    dpr: None,
                    crop: None,
                    pad: None,
                    rotate: None,
                    flip: None,
                    orient: OrientArg::default(),
                    quality: None,
                    distance: None,
                    search_ssim2: None,
                    search_distance: None,
                    attempts: 3,
                    crush: None,
                    tolerance: None,
                    quality_range: "50:97".to_string(),
                    structure: StructureArg::default(),
                    optimize_scans: false,
                    subsampling: None,
                    quant: None,
                    chroma_tables: None,
                    preset: None,
                    trellis: TrellisArg::default(),
                    sharp_yuv: false,
                    xyb: false,
                    deblock: DeblockArg::default(),
                    blur: 0.0,
                    strictness: None,
                    strip: Vec::new(),
                    apply_icc: None,
                    filter: FilterArg::Mitchell,
                    down_filter: None,
                    up_filter: None,
                    sharpen: 0.0,
                    lossless_only: false,
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
