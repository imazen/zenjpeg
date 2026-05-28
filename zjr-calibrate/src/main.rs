//! `zjr-calibrate` — expert-only sweep driver that builds calibration
//! tables consumed by `zenjpeg-recompress`.
//!
//! Subcommands:
//!
//! - `inspect <path>` — print the encoder-family / quality / structure
//!   probe for a single JPEG.
//! - `sweep --encoded-corpus DIR [...]` — drive `recompress()` over a
//!   directory of JPEGs across a target zensim-A grid, measure
//!   generation loss with zensim Profile A vs the *source* JPEG (not
//!   the unknown reference original), and emit TSV.
//! - `cumulative-sweep --references DIR --encoder ENC --output FILE` —
//!   generate the encoded corpus internally by re-encoding each
//!   reference PNG/PPM through zenjpeg at a Q grid, then run the same
//!   recompression sweep but score against the *known* reference. This
//!   produces the cumulative-zensim-A data that the calibration
//!   tables actually want.
//!
//! TSV schema (sweep + cumulative-sweep, identical columns):
//!
//! ```
//! reference_id  source_jpeg_id  encoder  source_q  quality_scale
//! subsampling   width  height  target_zensim_a  strategy
//! output_len  source_len  size_ratio  zensim_a_vs_source
//! zensim_a_vs_reference  projected_zensim_a  source_estimated_zensim_a
//! ```
//!
//! Empty cells (`-`) indicate "not measured for this row".

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use enough::Unstoppable;
use zenjpeg::decoder::{DeblockMode, DecodeConfig, OutputTarget};
use zenjpeg::detect::{EncoderFamily, QualityScale};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};

use zenjpeg::recompress::{
    Budget, RecompressOptions, RecompressResult, StrategyKind,
    expert::{
        CellCi, SourceAnalysis, StrategyParams, analyze_source, run_deblock, run_preserve,
        run_tuned, score_against_reference, score_recompression, target_zensim_a_to_ba_distance,
        target_zensim_a_to_ijg_q,
    },
    recompress,
};

#[derive(Parser, Debug)]
#[command(name = "zjr-calibrate", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Inspect a single JPEG: encoder, quality, dimensions, subsampling.
    Inspect { path: PathBuf },

    /// Sweep `recompress()` over a directory of pre-encoded JPEGs.
    /// Measures generation loss vs the source JPEG.
    Sweep(SweepArgs),

    /// Generate an encoded corpus from reference images, then sweep
    /// `recompress()` and measure against the **reference**. Produces
    /// cumulative-zensim-A data — the columns the calibration tables
    /// actually fit against.
    CumulativeSweep(CumulativeSweepArgs),

    /// Sweep `recompress()` over **externally-encoded** source JPEGs
    /// (from cjpeg / mozjpeg / cjpegli / any encoder) and score against
    /// the matching original PNG. This is how we validate the
    /// per-encoder calibration on REAL encoder output rather than
    /// zenjpeg's internal synthesis. Sources are matched to originals
    /// by filename stem prefix: a source `1001682__turbo__q60.jpg`
    /// matches original `1001682.png`.
    RecompressSweep(RecompressSweepArgs),
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// Directory of pre-encoded JPEGs to sweep.
    #[arg(long)]
    encoded_corpus: PathBuf,
    /// Output TSV path.
    #[arg(long)]
    output: PathBuf,
    /// Comma-separated zensim-A targets.
    #[arg(long, default_value = "30,40,50,60,70,80,90")]
    targets: String,
}

#[derive(Parser, Debug)]
struct CumulativeSweepArgs {
    /// Directory of reference images (PPM rgb8, or P6 / .ppm).
    /// We use PPM rather than PNG to keep zjr-calibrate dependency-light;
    /// alltheimages already produces PPM intermediates.
    #[arg(long)]
    references: PathBuf,
    /// Output TSV path.
    #[arg(long)]
    output: PathBuf,
    /// Comma-separated source IJG-quality grid for the *internal*
    /// encode step.
    #[arg(long, default_value = "20,30,40,50,60,70,80,85,90,95")]
    source_qs: String,
    /// Comma-separated zensim-A target grid for the recompression step.
    #[arg(long, default_value = "30,40,50,60,70,80,90")]
    targets: String,
    /// Chroma subsampling to use for the internal encode step
    /// (`444`, `422`, `420`, `440`).
    #[arg(long, default_value = "420")]
    subsampling: String,
    /// Force the Tuned strategy at every cell instead of the smart
    /// router. Required for fitting the `data.rs` Tuned fallback grids
    /// (`fit_calibration.py`): the smart router routes most cells to
    /// noop/lossless/preserve, starving the per-cell Tuned medians and
    /// collapsing the fit to a degenerate grid.
    #[arg(long)]
    force_tuned: bool,
}

#[derive(Parser, Debug)]
struct RecompressSweepArgs {
    /// Directory of externally-encoded source JPEGs. Filenames must
    /// start with the matching original's stem, e.g.
    /// `1001682__turbo__q60.jpg` → original `1001682.png`.
    #[arg(long)]
    sources: PathBuf,
    /// Directory of original PNG/PPM references.
    #[arg(long)]
    originals: PathBuf,
    /// Output TSV path.
    #[arg(long)]
    output: PathBuf,
    /// Comma-separated zensim-A target grid.
    #[arg(long, default_value = "30,40,50,60,70,80,90")]
    targets: String,
    /// Force a single strategy (`tuned`|`deblock`|`preserve`) instead of
    /// letting the router pick. Used to measure per-strategy achieved
    /// quality for calibration fitting.
    #[arg(long)]
    force_strategy: Option<String>,
    /// Baseline mode: instead of the smart router, do a NAIVE
    /// deblock-and-re-encode — decode with content-aware deblock, then
    /// re-encode at the target-q with default (non-hybrid) zenjpeg,
    /// unconditionally (no NoOp, no Lossless, no Preserve, no
    /// per-encoder calibration). This is the "just deblock and re-save"
    /// baseline the smart router is measured against.
    #[arg(long)]
    naive_deblock: bool,
    /// Iteration budget for the smart-router path (ignored by
    /// `--force-strategy` / `--naive-deblock`). `1` = one-shot measure
    /// (no bump, default). `>1` enables the Lever-4 closed loop:
    /// measure generation loss, predict achieved quality, bump the dial
    /// when a pass lands short of target.
    #[arg(long, default_value = "1")]
    max_iterations: u32,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let res = match cli.command {
        Command::Inspect { path } => inspect(&path),
        Command::Sweep(args) => sweep(&args),
        Command::CumulativeSweep(args) => cumulative_sweep(&args),
        Command::RecompressSweep(args) => recompress_sweep(&args),
    };
    match res {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zjr-calibrate: {e}");
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let probe = zenjpeg::detect::probe(&bytes).map_err(|e| format!("probe: {e}"))?;
    println!("file:           {path:?}");
    println!("size:           {} bytes", bytes.len());
    println!(
        "dimensions:     {}x{}",
        probe.dimensions.width, probe.dimensions.height
    );
    println!("encoder:        {:?}", probe.encoder);
    println!(
        "quality:        {} ({:?}, {:?})",
        probe.quality.value, probe.quality.scale, probe.quality.confidence
    );
    println!("subsampling:    {:?}", probe.subsampling);
    println!("mode:           {:?}", probe.mode);
    println!("components:     {}", probe.num_components);
    Ok(())
}

const TSV_HEADER: &str = "reference_id\tsource_jpeg_id\tencoder\tsource_q\tquality_scale\tsubsampling\twidth\theight\ttarget_zensim_a\tstrategy\toutput_len\tsource_len\tsize_ratio\tzensim_a_vs_source\tzensim_a_vs_reference\tprojected_zensim_a\tsource_estimated_zensim_a";

fn sweep(args: &SweepArgs) -> Result<(), String> {
    let targets = parse_targets(&args.targets)?;
    let jpegs = collect_jpegs(&args.encoded_corpus)?;

    if jpegs.is_empty() {
        return Err(format!(
            "no .jpg/.jpeg files under {:?}",
            args.encoded_corpus
        ));
    }
    let mut out = open_tsv(&args.output)?;
    writeln!(out, "{TSV_HEADER}").map_err(|e| format!("write header: {e}"))?;

    let mut rows = 0_u32;
    for jpeg_path in &jpegs {
        let source_bytes = fs::read(jpeg_path).map_err(|e| format!("read {jpeg_path:?}: {e}"))?;
        let analysis = match analyze_source(&source_bytes) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skip {jpeg_path:?}: analyze error {e}");
                continue;
            }
        };
        let id = jpeg_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        for &target in &targets {
            let result = recompress(
                &source_bytes,
                &RecompressOptions::new(target).with_budget(Budget::MaxIterations(1)),
            )
            .map_err(|e| format!("recompress {jpeg_path:?} @ {target}: {e}"))?;
            write_row(
                &mut out,
                None,
                &id,
                &analysis,
                target,
                &source_bytes,
                &result,
                None,
            )?;
            rows += 1;
        }
    }
    eprintln!(
        "zjr-calibrate sweep: wrote {rows} rows to {:?}",
        args.output
    );
    Ok(())
}

fn cumulative_sweep(args: &CumulativeSweepArgs) -> Result<(), String> {
    let source_qs: Vec<u8> = args
        .source_qs
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<u8>()
                .map_err(|e| format!("--source-qs {e}"))
        })
        .collect::<Result<_, _>>()?;
    let targets = parse_targets(&args.targets)?;
    let chroma = parse_subsampling(&args.subsampling)?;
    let refs = collect_references(&args.references)?;
    if refs.is_empty() {
        return Err(format!("no .ppm or .png files under {:?}", args.references));
    }
    let mut out = open_tsv(&args.output)?;
    writeln!(out, "{TSV_HEADER}").map_err(|e| format!("write header: {e}"))?;

    let mut rows = 0_u32;
    for ref_path in &refs {
        let (width, height, ref_bytes) = read_reference_rgb8(ref_path)?;
        let ref_id = ref_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        for &source_q in &source_qs {
            let source_jpeg =
                encode_ycbcr_progressive(&ref_bytes, width, height, source_q, chroma)?;
            let analysis =
                analyze_source(&source_jpeg).map_err(|e| format!("analyze synthesized: {e}"))?;
            let source_jpeg_id = format!("{ref_id}_q{source_q:02}");
            for &target in &targets {
                let result = if args.force_tuned {
                    // Forced Tuned: populate every cell so the data.rs
                    // fallback fit has a full grid (the smart router would
                    // route most cells away from Tuned).
                    let params = StrategyParams {
                        target_ijg_q: target_zensim_a_to_ijg_q(target),
                        target_ba_distance: target_zensim_a_to_ba_distance(target),
                        ci: CellCi::Moderate,
                        target_zensim_a: target,
                        projected_zensim_a: target,
                        dial_zensim_a: target,
                    };
                    let opts = RecompressOptions::new(target).with_budget(Budget::OneShot);
                    match run_tuned(&source_jpeg, &analysis, &params, &opts) {
                        Ok(o) => {
                            let ratio = o.bytes.len() as f32 / source_jpeg.len() as f32;
                            RecompressResult::Recompressed {
                                bytes: o.bytes,
                                strategy: StrategyKind::Tuned,
                                projected_zensim_a: target,
                                measured_zensim_a: o.measured_zensim_a,
                                source_to_output_ratio: ratio,
                            }
                        }
                        Err(_) => RecompressResult::NoOp {
                            reason: zenjpeg::recompress::NoOpReason::SourceAlreadyMeetsTarget,
                        },
                    }
                } else {
                    recompress(
                        &source_jpeg,
                        &RecompressOptions::new(target).with_budget(Budget::MaxIterations(1)),
                    )
                    .map_err(|e| format!("recompress {ref_id} q{source_q} → t{target}: {e}"))?
                };
                let zensim_a_vs_reference = match &result {
                    RecompressResult::Recompressed { bytes, .. }
                    | RecompressResult::LosslessOnly { bytes, .. } => Some(
                        score_against_reference(&ref_bytes, width, height, bytes)
                            .map_err(|e| format!("score vs reference: {e}"))?,
                    ),
                    _ => None,
                };
                write_row(
                    &mut out,
                    Some(&ref_id),
                    &source_jpeg_id,
                    &analysis,
                    target,
                    &source_jpeg,
                    &result,
                    zensim_a_vs_reference,
                )?;
                rows += 1;
            }
        }
    }
    eprintln!(
        "zjr-calibrate cumulative-sweep: wrote {rows} rows to {:?}",
        args.output
    );
    Ok(())
}

fn recompress_sweep(args: &RecompressSweepArgs) -> Result<(), String> {
    let targets = parse_targets(&args.targets)?;
    let sources = collect_jpegs(&args.sources)?;
    if sources.is_empty() {
        return Err(format!("no .jpg/.jpeg under {:?}", args.sources));
    }
    let originals = collect_references(&args.originals)?;
    if originals.is_empty() {
        return Err(format!("no .png/.ppm under {:?}", args.originals));
    }

    // Index originals by stem for prefix matching.
    let orig_index: Vec<(String, PathBuf)> = originals
        .iter()
        .filter_map(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .map(|s| (s.to_string(), p.clone()))
        })
        .collect();

    // Cache decoded originals (keyed by path) — many sources share one.
    let mut orig_cache: std::collections::HashMap<PathBuf, (u32, u32, Vec<u8>)> =
        std::collections::HashMap::new();

    let mut out = open_tsv(&args.output)?;
    writeln!(out, "{TSV_HEADER}").map_err(|e| format!("write header: {e}"))?;

    let mut rows = 0_u32;
    let mut unmatched = 0_u32;
    for src_path in &sources {
        let src_stem = src_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        // Find the original whose stem is the longest prefix of src_stem.
        let orig = orig_index
            .iter()
            .filter(|(ostem, _)| src_stem.starts_with(ostem.as_str()))
            .max_by_key(|(ostem, _)| ostem.len());
        let Some((_, orig_path)) = orig else {
            unmatched += 1;
            eprintln!("skip {src_path:?}: no matching original");
            continue;
        };

        let (width, height, ref_bytes) = match orig_cache.get(orig_path) {
            Some(v) => v.clone(),
            None => {
                let v = read_reference_rgb8(orig_path)?;
                orig_cache.insert(orig_path.clone(), v.clone());
                v
            }
        };

        let source_jpeg = fs::read(src_path).map_err(|e| format!("read {src_path:?}: {e}"))?;
        let analysis = match analyze_source(&source_jpeg) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skip {src_path:?}: analyze {e}");
                continue;
            }
        };
        // Dimensions must match the original (decoder vs PNG).
        if analysis.width != width || analysis.height != height {
            eprintln!(
                "skip {src_path:?}: dim {}x{} != original {width}x{height}",
                analysis.width, analysis.height
            );
            continue;
        }
        let src_id = src_stem.to_string();
        for &target in &targets {
            let result = if args.naive_deblock {
                // NAIVE baseline: decode + deblock + plain re-encode at
                // target-q, unconditionally. No router intelligence.
                let bytes = naive_deblock_reencode(&source_jpeg, target)
                    .map_err(|e| format!("naive {src_id} → t{target}: {e}"))?;
                let ratio = bytes.len() as f32 / source_jpeg.len() as f32;
                RecompressResult::Recompressed {
                    bytes,
                    strategy: StrategyKind::Deblock,
                    projected_zensim_a: target,
                    measured_zensim_a: None,
                    source_to_output_ratio: ratio,
                }
            } else if let Some(force) = &args.force_strategy {
                // Forced-strategy path: call the expert strategy directly
                // and wrap as a Recompressed result for write_row. Used to
                // measure per-strategy achieved quality for calibration.
                let params = StrategyParams {
                    target_ijg_q: target_zensim_a_to_ijg_q(target),
                    target_ba_distance: target_zensim_a_to_ba_distance(target),
                    ci: CellCi::Moderate,
                    target_zensim_a: target,
                    projected_zensim_a: target,
                    dial_zensim_a: target,
                };
                let opts = RecompressOptions::new(target).with_budget(Budget::OneShot);
                let outcome = match force.as_str() {
                    "tuned" => run_tuned(&source_jpeg, &analysis, &params, &opts),
                    "deblock" => run_deblock(&source_jpeg, &analysis, &params, &opts),
                    "preserve" => run_preserve(&source_jpeg, &analysis, &params, &opts),
                    other => return Err(format!("unknown --force-strategy {other}")),
                };
                match outcome {
                    Ok(o) => {
                        let ratio = o.bytes.len() as f32 / source_jpeg.len() as f32;
                        RecompressResult::Recompressed {
                            bytes: o.bytes,
                            strategy: match force.as_str() {
                                "tuned" => StrategyKind::Tuned,
                                "deblock" => StrategyKind::Deblock,
                                _ => StrategyKind::Preserve,
                            },
                            projected_zensim_a: target,
                            measured_zensim_a: o.measured_zensim_a,
                            source_to_output_ratio: ratio,
                        }
                    }
                    // Strategy declined (e.g. preserve roundtrip guard) —
                    // record as a NoOp so the cell is visible.
                    Err(_) => RecompressResult::NoOp {
                        reason: zenjpeg::recompress::NoOpReason::SourceAlreadyMeetsTarget,
                    },
                }
            } else {
                recompress(
                    &source_jpeg,
                    &RecompressOptions::new(target)
                        .with_budget(Budget::MaxIterations(args.max_iterations.max(1))),
                )
                .map_err(|e| format!("recompress {src_id} → t{target}: {e}"))?
            };
            let zensim_a_vs_reference = match &result {
                RecompressResult::Recompressed { bytes, .. }
                | RecompressResult::LosslessOnly { bytes, .. } => Some(
                    score_against_reference(&ref_bytes, width, height, bytes)
                        .map_err(|e| format!("score vs original: {e}"))?,
                ),
                _ => None,
            };
            let ref_stem = orig_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?");
            write_row(
                &mut out,
                Some(ref_stem),
                &src_id,
                &analysis,
                target,
                &source_jpeg,
                &result,
                zensim_a_vs_reference,
            )?;
            rows += 1;
        }
    }
    eprintln!(
        "zjr-calibrate recompress-sweep: wrote {rows} rows ({unmatched} sources unmatched) to {:?}",
        args.output
    );
    Ok(())
}

/// Naive baseline: decode (content-aware deblock) → re-encode at the
/// target-q with DEFAULT zenjpeg (JpegliProgressive, no hybrid/trellis).
/// Matches what a naive "deblock and re-save" tool does — no router, no
/// Preserve, no Lossless, no NoOp, no per-encoder calibration.
fn naive_deblock_reencode(source_jpeg: &[u8], target: f32) -> Result<Vec<u8>, String> {
    use enough::Unstoppable;
    let decoded = DecodeConfig::new()
        .output_target(OutputTarget::Srgb8)
        .deblock(DeblockMode::Auto)
        .decode(source_jpeg, Unstoppable)
        .map_err(|e| format!("decode: {e}"))?;
    let pixels = decoded.pixels_u8().ok_or("no u8 pixels")?;
    let q = target_zensim_a_to_ijg_q(target);
    // Plain progressive YCbCr — the naive default. NO HybridMaxCompression.
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(true);
    let mut enc = cfg
        .encode_from_bytes(decoded.width, decoded.height, PixelLayout::Rgb8Srgb)
        .map_err(|e| format!("enc setup: {e}"))?;
    enc.push_packed(pixels, Unstoppable)
        .map_err(|e| format!("enc push: {e}"))?;
    enc.finish().map_err(|e| format!("enc finish: {e}"))
}

fn write_row<W: Write>(
    out: &mut W,
    reference_id: Option<&str>,
    source_id: &str,
    analysis: &SourceAnalysis,
    target: f32,
    source_bytes: &[u8],
    result: &RecompressResult,
    zensim_a_vs_reference: Option<f32>,
) -> Result<(), String> {
    let (strategy, output_len, measured, projected) = match result {
        RecompressResult::Recompressed {
            bytes,
            strategy,
            projected_zensim_a,
            measured_zensim_a,
            ..
        } => (
            strategy_name(*strategy),
            bytes.len(),
            *measured_zensim_a,
            Some(*projected_zensim_a),
        ),
        RecompressResult::LosslessOnly { bytes, .. } => ("lossless", bytes.len(), None, None),
        RecompressResult::NoOp { .. } => ("noop", source_bytes.len(), None, None),
        _ => ("unknown", source_bytes.len(), None, None),
    };
    let size_ratio = output_len as f32 / source_bytes.len() as f32;
    writeln!(
        out,
        "{ref_id}\t{src}\t{enc:?}\t{sq}\t{qs:?}\t{sub:?}\t{w}\t{h}\t{t:.2}\t{strat}\t{olen}\t{slen}\t{sr:.4}\t{m}\t{c}\t{p}\t{est:.2}",
        ref_id = reference_id.unwrap_or("-"),
        src = source_id,
        enc = analysis.encoder,
        sq = analysis.quality.value,
        qs = analysis.quality.scale,
        sub = analysis.subsampling,
        w = analysis.width,
        h = analysis.height,
        t = target,
        strat = strategy,
        olen = output_len,
        slen = source_bytes.len(),
        sr = size_ratio,
        m = opt_f32(measured),
        c = opt_f32(zensim_a_vs_reference),
        p = opt_f32(projected),
        est = analysis.estimated_zensim_a_vs_reference(),
    )
    .map_err(|e| format!("write row: {e}"))?;
    let _ = (EncoderFamily::Mozjpeg, QualityScale::IjgQuality); // suppress imports if pruned
    Ok(())
}

fn opt_f32(x: Option<f32>) -> String {
    x.map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "-".to_string())
}

fn strategy_name(k: StrategyKind) -> &'static str {
    match k {
        StrategyKind::Preserve => "preserve",
        StrategyKind::Deblock => "deblock",
        StrategyKind::Tuned => "tuned",
        StrategyKind::Lossless => "lossless",
        _ => "unknown",
    }
}

fn parse_targets(s: &str) -> Result<Vec<f32>, String> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<f32>()
                .map_err(|e| format!("--targets {e}"))
        })
        .collect()
}

fn parse_subsampling(s: &str) -> Result<ChromaSubsampling, String> {
    match s {
        "444" => Ok(ChromaSubsampling::None),
        "422" => Ok(ChromaSubsampling::HalfHorizontal),
        "420" => Ok(ChromaSubsampling::Quarter),
        "440" => Ok(ChromaSubsampling::HalfVertical),
        other => Err(format!("unknown subsampling {other}; want 444|422|420|440")),
    }
}

fn open_tsv(path: &Path) -> Result<File, String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {parent:?}: {e}"))?;
        }
    }
    File::create(path).map_err(|e| format!("create {path:?}: {e}"))
}

fn collect_jpegs(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "jpg" | "jpeg") {
            out.push(entry.path().to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

fn collect_references(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .max_depth(2)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry
            .path()
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "ppm" | "png") {
            out.push(entry.path().to_path_buf());
        }
    }
    out.sort();
    Ok(out)
}

fn read_reference_rgb8(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "ppm" => read_ppm_rgb8(path),
        "png" => read_png_rgb8(path),
        other => Err(format!(
            "{path:?}: unsupported reference extension {other:?}"
        )),
    }
}

fn read_png_rgb8(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let f = File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(f));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header {path:?}: {e}"))?;
    let info = reader.info().clone();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png decode {path:?}: {e}"))?;
    buf.truncate(frame.buffer_size());

    let width = info.width;
    let height = info.height;
    let need = (width as usize) * (height as usize) * 3;

    let pixels = match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => {
            if buf.len() != need {
                return Err(format!("{path:?}: rgb8 size mismatch"));
            }
            buf
        }
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            let stride_in = 4 * (width as usize);
            let mut out = Vec::with_capacity(need);
            for row in buf.chunks_exact(stride_in) {
                for px in row.chunks_exact(4) {
                    out.extend_from_slice(&px[..3]);
                }
            }
            out
        }
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            let mut out = Vec::with_capacity(need);
            for &v in &buf {
                out.extend_from_slice(&[v, v, v]);
            }
            out
        }
        (ct, bd) => {
            return Err(format!(
                "{path:?}: unsupported PNG ({ct:?}, {bd:?}); convert to RGB8 first",
            ));
        }
    };
    Ok((width, height, pixels))
}

fn encode_ycbcr_progressive(
    rgb8: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    chroma: ChromaSubsampling,
) -> Result<Vec<u8>, String> {
    let cfg = EncoderConfig::ycbcr(Quality::ApproxJpegli(quality as f32), chroma).progressive(true);
    let mut enc = cfg
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| format!("encoder setup: {e}"))?;
    enc.push_packed(rgb8, Unstoppable)
        .map_err(|e| format!("encoder push: {e}"))?;
    enc.finish().map_err(|e| format!("encoder finish: {e}"))
}

/// Minimal PPM (P6) reader. Returns `(width, height, rgb8_bytes)`.
fn read_ppm_rgb8(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let data = fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut idx = 0;
    let mut tokens = Vec::with_capacity(4);
    while tokens.len() < 4 && idx < data.len() {
        // Skip whitespace and comments.
        while idx < data.len() && (data[idx] as char).is_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
            continue;
        }
        let start = idx;
        while idx < data.len() && !(data[idx] as char).is_whitespace() {
            idx += 1;
        }
        if start == idx {
            break;
        }
        tokens.push(&data[start..idx]);
    }
    if tokens.len() < 4 {
        return Err(format!("{path:?}: not a valid PPM (P6) header"));
    }
    if tokens[0] != b"P6" {
        return Err(format!("{path:?}: only P6 PPM is supported"));
    }
    let width: u32 = std::str::from_utf8(tokens[1])
        .map_err(|e| format!("ppm width utf8: {e}"))?
        .parse()
        .map_err(|e| format!("ppm width parse: {e}"))?;
    let height: u32 = std::str::from_utf8(tokens[2])
        .map_err(|e| format!("ppm height utf8: {e}"))?
        .parse()
        .map_err(|e| format!("ppm height parse: {e}"))?;
    let max_val: u32 = std::str::from_utf8(tokens[3])
        .map_err(|e| format!("ppm maxval utf8: {e}"))?
        .parse()
        .map_err(|e| format!("ppm maxval parse: {e}"))?;
    if max_val != 255 {
        return Err(format!(
            "{path:?}: PPM maxval {max_val} unsupported (need 255)"
        ));
    }
    // Skip the single whitespace after maxval, then read width*height*3 bytes.
    if idx < data.len() && (data[idx] as char).is_whitespace() {
        idx += 1;
    }
    let needed = (width as usize) * (height as usize) * 3;
    if data.len() - idx < needed {
        return Err(format!(
            "{path:?}: ppm payload short ({} bytes available, {needed} expected)",
            data.len() - idx
        ));
    }
    let pixels = data[idx..idx + needed].to_vec();
    let _ = score_recompression; // silence unused if features prune
    Ok((width, height, pixels))
}
