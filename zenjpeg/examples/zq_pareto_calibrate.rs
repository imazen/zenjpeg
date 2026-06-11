//! Comprehensive Pareto-front calibration harness for the Zq /
//! perceptual-target controller.
//!
//! Sweeps four axes per the global benchmark-discipline rule:
//!   - **size**: tiny (64), small (256), medium (1024), large (native)
//!   - **config**: ycbcr/xyb × 4:4:4/4:2:0 × baseline/progressive ±
//!     auto-optimize (8 representative configs covering the knobs that
//!     materially shift the q→zq curve and encoded bytes)
//!   - **q**: 0..=100 step 5 (21 points; full coverage including the
//!     low-q regime that web-focused codecs ship for)
//!   - **content**: tiny + photo + screen + line-art + mixed via the
//!     usual codec-eval corpora
//!
//! Per (image, size, config, q) we emit one TSV row capturing
//! `bytes + zensim + encode_ms`. Per-image zenanalyze features are
//! captured separately so offline regression can join them in.
//!
//! This harness produces the data; the offline analysis (in a sibling
//! script / Rust binary) computes per-(image, size, config, target_zq)
//! Pareto-optimal `(config, q)` and trains the regression that the
//! perceptual-target controller bakes in.
//!
//! Usage:
//!   cargo run --release -p zenjpeg --features target-zq \
//!     --example zq_pareto_calibrate -- \
//!       --corpus /path/to/photos \
//!       --corpus /path/to/screens \
//!       --output benchmarks/zq_pareto_<DATE>.tsv \
//!       --features-output benchmarks/zq_pareto_features_<DATE>.tsv \
//!       [--max-images N] [--sizes 64,256,1024,native] [--threads N]
//!
//! Output is appended row-by-row so a partial run is still useful;
//! a crash mid-sweep doesn't lose prior cells.

#![cfg(feature = "target-zq")]

use enough::Unstoppable;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;
use zenanalyze::analyze_features_rgb8;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenjpeg::decode::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality, XybSubsampling};
use zensim::{DiffmapWeighting, RgbSlice, Zensim, ZensimProfile};

// ---------------------------------------------------------------------
// Sweep grid
// ---------------------------------------------------------------------

/// Quality grid: full 0..100 step 5 per the source-informing-benchmark rule.
const Q_GRID: &[u8] = &[
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
];

/// Default size axis. `0` means "use native dimensions" (large).
/// Sized by max(width, height) — preserve aspect via image::imageops::resize
/// with Lanczos3 (high-quality downsample).
const DEFAULT_SIZES: &[u32] = &[64, 256, 1024, 0];

/// One encoder configuration variant. The set below covers the knobs
/// that materially shift bytes and the q→zq mapping. Trellis lambda
/// and dequant bias are reachable as further axes; start with these
/// 8 because each is a single-flag flip with an obvious user-visible
/// meaning, and we can extend later from the same TSV without a
/// re-run if we add new configs that happen to include any in this
/// initial set.
#[derive(Clone, Copy)]
struct ConfigSpec {
    name: &'static str,
    /// Encoded as a small-int feature for the trained model.
    id: u8,
    color_mode: ColorMode,
    sub: SubChoice,
    progressive: bool,
    auto_optimize: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorMode {
    YCbCr,
    Xyb,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SubChoice {
    /// 4:4:4 — no chroma downsample
    Full,
    /// 4:2:0 — h2v2
    Quarter,
}

const CONFIGS: &[ConfigSpec] = &[
    ConfigSpec {
        name: "ycbcr_444_baseline",
        id: 0,
        color_mode: ColorMode::YCbCr,
        sub: SubChoice::Full,
        progressive: false,
        auto_optimize: false,
    },
    ConfigSpec {
        name: "ycbcr_420_baseline",
        id: 1,
        color_mode: ColorMode::YCbCr,
        sub: SubChoice::Quarter,
        progressive: false,
        auto_optimize: false,
    },
    ConfigSpec {
        name: "ycbcr_444_progressive",
        id: 2,
        color_mode: ColorMode::YCbCr,
        sub: SubChoice::Full,
        progressive: true,
        auto_optimize: false,
    },
    ConfigSpec {
        name: "ycbcr_420_progressive",
        id: 3,
        color_mode: ColorMode::YCbCr,
        sub: SubChoice::Quarter,
        progressive: true,
        auto_optimize: false,
    },
    ConfigSpec {
        name: "xyb_444_baseline",
        id: 4,
        color_mode: ColorMode::Xyb,
        sub: SubChoice::Full,
        progressive: false,
        auto_optimize: false,
    },
    ConfigSpec {
        name: "xyb_420_baseline",
        id: 5,
        color_mode: ColorMode::Xyb,
        sub: SubChoice::Quarter,
        progressive: false,
        auto_optimize: false,
    },
    ConfigSpec {
        name: "ycbcr_420_auto_optimize",
        id: 6,
        color_mode: ColorMode::YCbCr,
        sub: SubChoice::Quarter,
        progressive: true,
        auto_optimize: true,
    },
    ConfigSpec {
        name: "ycbcr_444_auto_optimize",
        id: 7,
        color_mode: ColorMode::YCbCr,
        sub: SubChoice::Full,
        progressive: true,
        auto_optimize: true,
    },
];

// ---------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------

struct Args {
    corpora: Vec<PathBuf>,
    sizes: Vec<u32>,
    output: PathBuf,
    features_output: PathBuf,
    max_images: usize,
    threads: usize,
    /// When true, skip the per-config encode + zensim loop and emit
    /// only the per-(image, size) features TSV. Used to extend an
    /// existing Pareto sweep with new analyzer features without
    /// re-running the expensive encode pass.
    features_only: bool,
}

fn parse_args() -> Args {
    let mut corpora: Vec<PathBuf> = Vec::new();
    let mut sizes: Vec<u32> = Vec::new();
    let mut max_images = 1024;
    let mut threads = 0;
    let mut features_only = false;
    let date = chrono_today();
    let mut output = PathBuf::from(format!("benchmarks/zq_pareto_{date}.tsv"));
    let mut features_output = PathBuf::from(format!("benchmarks/zq_pareto_features_{date}.tsv"));

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--corpus" => corpora.push(PathBuf::from(it.next().unwrap())),
            "--sizes" => {
                let s = it.next().unwrap();
                for tok in s.split(',') {
                    if tok == "native" {
                        sizes.push(0);
                    } else {
                        sizes.push(tok.parse().expect("size must be uint or 'native'"));
                    }
                }
            }
            "--output" => output = PathBuf::from(it.next().unwrap()),
            "--features-output" => features_output = PathBuf::from(it.next().unwrap()),
            "--max-images" => max_images = it.next().unwrap().parse().expect("max-images uint"),
            "--threads" => threads = it.next().unwrap().parse().expect("threads uint"),
            "--features-only" => features_only = true,
            other => panic!("unknown arg: {other}"),
        }
    }
    if corpora.is_empty() {
        // Default mixed corpus covering all 5 content classes.
        for d in [
            "/home/lilith/work/codec-eval/codec-corpus/CID22/CID22-512/validation",
            "/home/lilith/work/codec-eval/codec-corpus/CID22/CID22-512/training",
            "/home/lilith/work/codec-eval/codec-corpus/gb82",
            "/home/lilith/work/codec-eval/codec-corpus/gb82-sc",
            "/home/lilith/work/codec-eval/codec-corpus/clic2025/training",
            "/home/lilith/work/codec-eval/codec-corpus/clic2025/final-test",
        ] {
            corpora.push(PathBuf::from(d));
        }
    }
    if sizes.is_empty() {
        sizes = DEFAULT_SIZES.to_vec();
    }
    Args {
        corpora,
        sizes,
        output,
        features_output,
        features_only,
        max_images,
        threads,
    }
}

fn chrono_today() -> String {
    // Use system date — we don't pull chrono just for this.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Compute YYYY-MM-DD from epoch seconds (UTC, naive).
    let days = (secs / 86400) as i64;
    // Days from 1970-01-01 → calendar date (Howard Hinnant's algorithm).
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ---------------------------------------------------------------------
// Image loading + size-axis variants
// ---------------------------------------------------------------------

fn load_png(path: &std::path::Path) -> (Vec<u8>, u32, u32) {
    let img =
        image::open(path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
    let rgb = img.to_rgb8();
    (rgb.as_raw().clone(), rgb.width(), rgb.height())
}

/// Resize an RGB8 image to fit within `target_max` on its longer side
/// while preserving aspect. `target_max == 0` returns the input as-is.
fn resize_to(rgb: &[u8], w: u32, h: u32, target_max: u32) -> (Vec<u8>, u32, u32) {
    if target_max == 0 || (w.max(h) <= target_max) {
        return (rgb.to_vec(), w, h);
    }
    let scale = target_max as f32 / w.max(h) as f32;
    let new_w = ((w as f32 * scale).round() as u32).max(1);
    let new_h = ((h as f32 * scale).round() as u32).max(1);
    let buf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(w, h, rgb.to_vec())
        .expect("rgb8 buffer");
    let resized =
        image::imageops::resize(&buf, new_w, new_h, image::imageops::FilterType::Lanczos3);
    (resized.into_raw(), new_w, new_h)
}

// ---------------------------------------------------------------------
// Encoding + scoring
// ---------------------------------------------------------------------

fn build_encoder(spec: ConfigSpec, q: u8) -> EncoderConfig {
    let quality = Quality::ApproxJpegli(q as f32);
    let mut cfg = match spec.color_mode {
        ColorMode::YCbCr => {
            let sub = match spec.sub {
                SubChoice::Full => ChromaSubsampling::None,
                SubChoice::Quarter => ChromaSubsampling::Quarter,
            };
            EncoderConfig::ycbcr(quality, sub)
        }
        ColorMode::Xyb => {
            // XYB has its own subsampling enum (only B-channel).
            let xyb_sub = match spec.sub {
                SubChoice::Full => XybSubsampling::Full,
                SubChoice::Quarter => XybSubsampling::BQuarter,
            };
            EncoderConfig::xyb(quality, xyb_sub)
        }
    };
    // NOTE: zenjpeg's `EncoderConfig::ycbcr/xyb` defaults to
    // `ProgressiveScanMode::Progressive`. To actually compare progressive
    // vs baseline we must EXPLICITLY set Baseline when spec.progressive
    // is false; passing Progressive when already-default is a no-op and
    // produces byte-identical output.
    if spec.progressive {
        cfg = cfg.progressive(zenjpeg::encode::ProgressiveScanMode::Progressive);
    } else {
        cfg = cfg.progressive(zenjpeg::encode::ProgressiveScanMode::Baseline);
    }
    if spec.auto_optimize {
        cfg = cfg.auto_optimize(true);
    }
    cfg
}

/// One encode + decode + zensim score.
/// Returns (encoded_bytes, zensim_score, encode_ms, total_ms).
fn encode_decode_score(
    z: &Zensim,
    pre: &zensim::PrecomputedReference,
    rgb: &[u8],
    w: u32,
    h: u32,
    spec: ConfigSpec,
    q: u8,
) -> Option<(usize, f32, f64, f64)> {
    let cfg = build_encoder(spec, q);
    let total_start = Instant::now();
    let encode_start = Instant::now();
    let jpeg = match cfg.encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb) {
        Ok(j) => j,
        Err(_) => return None, // some configs reject some inputs (e.g., XYB at q=0)
    };
    let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
    let dec = match Decoder::new().decode(&jpeg, Unstoppable) {
        Ok(d) => match d.into_pixels_u8() {
            Some(p) => p,
            None => return None,
        },
        Err(_) => return None,
    };
    let chunks: &[[u8; 3]] = dec.as_chunks::<3>().0;
    let dec_slice = RgbSlice::new(chunks, w as usize, h as usize);
    let zensim_score =
        match z.compute_with_ref_and_diffmap(pre, &dec_slice, DiffmapWeighting::Trained) {
            Ok(r) => r.score() as f32,
            Err(_) => return None,
        };
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    Some((jpeg.len(), zensim_score, encode_ms, total_ms))
}

// ---------------------------------------------------------------------
// Feature extraction (per image-size; identical across configs+q)
// ---------------------------------------------------------------------

/// Every feature we want as a column. Order is the column order in the
/// features TSV. Composites + experimental land here so the regression
/// can learn from them; if those cargo features aren't on, the
/// corresponding columns get None and we emit empty.
fn full_feature_set() -> FeatureSet {
    FeatureSet::SUPPORTED
}

fn feature_columns() -> Vec<AnalysisFeature> {
    // Walk every analyzer feature this zenanalyze build supports.
    // FeatureSet::SUPPORTED reflects compile-time feature gates, so a
    // build without the `composites` cargo feature emits a strictly
    // smaller TSV — automatic + future-proof.
    //
    // Picker training scripts read column names; new features land
    // here without code changes the moment they ship in zenanalyze.
    zenanalyze::feature::FeatureSet::SUPPORTED.iter().collect()
}

/// Whether an append-mode TSV needs its header written, refusing to
/// proceed when an existing header doesn't match (#133).
///
/// Appending rows under a stale header silently scrambles the
/// downstream `csv.DictReader` column mapping — picker training then
/// succeeds on garbage. This bites whenever zenanalyze ships new
/// `AnalysisFeature` variants between runs against the same output
/// path, in both `--features-only` and sweep-resume modes. A
/// header-equal file appends as before (legit resume); a missing or
/// empty file gets a fresh header (an empty file previously collected
/// headerless rows).
fn tsv_needs_header(path: &std::path::Path, expected_header: &str) -> bool {
    use std::io::BufRead;
    let Ok(file) = std::fs::File::open(path) else {
        return true; // missing → write header
    };
    let mut existing = String::new();
    if std::io::BufReader::new(file)
        .read_line(&mut existing)
        .unwrap_or(0)
        == 0
    {
        return true; // empty → write header
    }
    let existing = existing.trim_end_matches(['\r', '\n']);
    if existing == expected_header {
        return false; // schema matches → append rows, no header
    }
    eprintln!(
        "ERROR: {} was written for a different schema (existing header: {} cols, \
         current: {} cols).",
        path.display(),
        existing.split('\t').count(),
        expected_header.split('\t').count(),
    );
    eprintln!(
        "Appending would silently scramble downstream column mapping (#133). \
         Re-run with a fresh output path or remove the file."
    );
    std::process::exit(1);
}

fn feature_value_str(
    analysis: &zenanalyze::feature::AnalysisResults,
    f: AnalysisFeature,
) -> String {
    if let Some(v) = analysis.get_f32(f) {
        format!("{v:.6}")
    } else if let Some(v) = analysis.get(f) {
        // Could be U32 or Bool — render generically.
        match v {
            zenanalyze::feature::FeatureValue::F32(x) => format!("{x:.6}"),
            zenanalyze::feature::FeatureValue::U32(x) => format!("{x}"),
            zenanalyze::feature::FeatureValue::Bool(b) => format!("{}", b as u8),
            _ => String::new(),
        }
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------

fn main() {
    let args = parse_args();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let z = Zensim::new(ZensimProfile::latest());

    let mut paths: Vec<PathBuf> = Vec::new();
    for corpus in &args.corpora {
        let entries = std::fs::read_dir(corpus)
            .unwrap_or_else(|e| panic!("read_dir {}: {e}", corpus.display()));
        for entry in entries.filter_map(|r| r.ok()) {
            let p = entry.path();
            match p.extension().and_then(|s| s.to_str()) {
                Some("png") | Some("jpg") | Some("jpeg") => paths.push(p),
                _ => {}
            }
        }
    }
    paths.sort();
    paths.truncate(args.max_images);

    let cells = paths.len() * args.sizes.len() * CONFIGS.len() * Q_GRID.len();
    eprintln!(
        "[zq_pareto_calibrate] {} images × {} sizes × {} configs × {} q values = {} cells",
        paths.len(),
        args.sizes.len(),
        CONFIGS.len(),
        Q_GRID.len(),
        cells,
    );
    eprintln!("[zq_pareto_calibrate] output: {}", args.output.display());
    eprintln!(
        "[zq_pareto_calibrate] features:  {}",
        args.features_output.display()
    );

    // Open output files in append mode. Write headers if files are new.
    // In features-only mode the main pareto TSV is never opened.
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let main_file: Option<Mutex<std::fs::File>> = if args.features_only {
        None
    } else {
        const MAIN_HEADER: &str = "image_path\tsize_class\twidth\theight\tconfig_id\tconfig_name\tq\tbytes\tzensim\tencode_ms\ttotal_ms";
        let main_needs_header = tsv_needs_header(&args.output, MAIN_HEADER);
        let main_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)
            .expect("open output");
        let main_file = Mutex::new(main_file);
        if main_needs_header {
            let mut f = main_file.lock().unwrap();
            writeln!(f, "{MAIN_HEADER}").ok();
        }
        Some(main_file)
    };

    let cols = feature_columns();
    let feat_header = {
        let mut h = String::from("image_path\tsize_class\twidth\theight");
        for c in &cols {
            h.push_str("\tfeat_");
            h.push_str(c.name());
        }
        h
    };
    let feat_needs_header = tsv_needs_header(&args.features_output, &feat_header);
    let feat_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.features_output)
        .expect("open features output");
    let feat_file = Mutex::new(feat_file);
    if feat_needs_header {
        let mut f = feat_file.lock().unwrap();
        writeln!(f, "{feat_header}").ok();
    }

    // Build the work units. We unwrap (image, size) up-front (to avoid
    // re-reading + re-resizing for every config). Each work unit is one
    // (image, size) cell, expanded to (CONFIGS × Q_GRID) inner work.
    let query = AnalysisQuery::new(full_feature_set());
    let started = Instant::now();
    let unit_count = paths.len() * args.sizes.len();
    let done = std::sync::atomic::AtomicUsize::new(0);

    let work_units: Vec<(PathBuf, u32)> = paths
        .iter()
        .flat_map(|path| args.sizes.iter().map(move |&sz| (path.clone(), sz)))
        .collect();
    work_units.par_iter().for_each(|(path, target_size)| {
        let target_size = *target_size;
        // Load + resize.
        let (rgb_native, w_native, h_native) = load_png(path);
        let (rgb, w, h) = resize_to(&rgb_native, w_native, h_native, target_size);
        let size_class = match target_size {
            64 => "tiny",
            256 => "small",
            1024 => "medium",
            0 => "large",
            _ => "custom",
        };

        // Per-image features (analyzed once at this size).
        let analysis = analyze_features_rgb8(&rgb, w, h, &query);
        {
            let mut f = feat_file.lock().unwrap();
            write!(f, "{}\t{}\t{}\t{}", path.display(), size_class, w, h).ok();
            for c in &cols {
                write!(f, "\t{}", feature_value_str(&analysis, *c)).ok();
            }
            writeln!(f).ok();
            f.flush().ok();
        }

        // Skip the expensive encode + zensim loop when --features-only.
        if let Some(main_file) = main_file.as_ref() {
            // Pre-compute zensim reference once.
            let src_chunks: &[[u8; 3]] = rgb.as_chunks::<3>().0;
            let src_slice = RgbSlice::new(src_chunks, w as usize, h as usize);
            let pre = z.precompute_reference(&src_slice).expect("precompute");

            // Cartesian over configs × q.
            for spec in CONFIGS {
                for &q in Q_GRID {
                    let row = encode_decode_score(&z, &pre, &rgb, w, h, *spec, q);
                    let mut f = main_file.lock().unwrap();
                    match row {
                        Some((bytes, zensim, encode_ms, total_ms)) => {
                            writeln!(
                                f,
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.3}\t{:.3}",
                                path.display(),
                                size_class,
                                w,
                                h,
                                spec.id,
                                spec.name,
                                q,
                                bytes,
                                zensim,
                                encode_ms,
                                total_ms,
                            )
                            .ok();
                        }
                        None => {
                            // Record a row with empty bytes/zensim to mark
                            // the failure — keeps the cell index dense.
                            writeln!(
                                f,
                                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t\t\t\t",
                                path.display(),
                                size_class,
                                w,
                                h,
                                spec.id,
                                spec.name,
                                q,
                            )
                            .ok();
                        }
                    }
                }
                main_file.lock().unwrap().flush().ok();
            }
        }

        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n % 4 == 0 || n == unit_count {
            let dt = started.elapsed().as_secs_f64();
            let rate = n as f64 / dt;
            let eta = (unit_count - n) as f64 / rate;
            eprintln!(
                "  progress: {}/{}  ({:.1}/sec, ETA {:.0}s)",
                n, unit_count, rate, eta
            );
        }
    });

    let primary = if args.features_only {
        args.features_output.display().to_string()
    } else {
        args.output.display().to_string()
    };
    eprintln!(
        "[zq_pareto_calibrate] done in {:.0}s, output at {primary}",
        started.elapsed().as_secs_f64(),
    );
}
