//! Corpus evaluation harness — runs `analyze_features` over one or
//! more corpora with `FeatureSet::SUPPORTED` and dumps every feature
//! to CSV on stdout. Used to reproduce the empirical calibration
//! baseline in `docs/calibration-corpus-2026-04-27.md`.
//!
//! Two modes, picked via env vars:
//!
//! * **Labeled mode** (preferred for calibration). Set
//!   `LABELS_TSV=/path/to/labels.tsv` to a TSV with at least the
//!   columns `corpus`, `image`, `primary_category`, `is_synthetic`,
//!   `palette_size`, `dominant_chroma`, `has_text` (the
//!   coefficient `benchmarks/classifier-eval/labels.tsv` schema).
//!   Each row's image is resolved against `CORPUS_ROOT` (defaults
//!   to `~/work/codec-eval/codec-corpus`) and the per-corpus
//!   sub-directories listed in `RESOLVE_DIRS` below. Output CSV
//!   has the label columns inserted before the feature columns.
//!
//! * **Unlabeled walk mode** (default if `LABELS_TSV` is unset).
//!   Walks each corpus root in `CORPORA` and runs the analyzer on
//!   every PNG / JPEG up to `MAX_PER_CORPUS` images
//!   (default 120, override via env). Override the corpus list at
//!   the top of `main()` for one-off runs.
//!
//! Build:
//!
//! ```sh
//! cargo build --release -p zenanalyze --features experimental \
//!   --example corpus_eval
//! ```
//!
//! Run (labeled mode):
//!
//! ```sh
//! LABELS_TSV=path/to/labels.tsv \
//! CORPUS_ROOT=/path/to/codec-corpus \
//!   ./target/release/examples/corpus_eval > /tmp/zenanalyze_labeled.csv
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::ImageReader;
use zenanalyze::feature::{
    AnalysisFeature, AnalysisQuery, AnalysisResults, FeatureSet, FeatureValue,
};
use zenpixels::{PixelDescriptor, PixelSlice};

// Hand-list every supported feature. Must match feature.rs table.
fn all_features() -> Vec<AnalysisFeature> {
    use AnalysisFeature::*;
    vec![
        Variance,
        EdgeDensity,
        ChromaComplexity,
        CbSharpness,
        CrSharpness,
        Uniformity,
        FlatColorBlockRatio,
        Colourfulness,
        LaplacianVariance,
        VarianceSpread,
        DistinctColorBins,
        PaletteDensity,
        CbHorizSharpness,
        CbVertSharpness,
        CbPeakSharpness,
        CrHorizSharpness,
        CrVertSharpness,
        CrPeakSharpness,
        HighFreqEnergyRatio,
        LumaHistogramEntropy,
        DctCompressibilityY,
        DctCompressibilityUV,
        PatchFraction,
        AlphaPresent,
        AlphaUsedFraction,
        AlphaBimodalScore,
        TextLikelihood,
        ScreenContentLikelihood,
        NaturalLikelihood,
        IndexedPaletteWidth,
        PaletteFitsIn256,
        PeakLuminanceNits,
        P99LuminanceNits,
        HdrHeadroomStops,
        HdrPixelFraction,
        WideGamutPeak,
        WideGamutFraction,
        EffectiveBitDepth,
        HdrPresent,
        GrayscaleScore,
        AqMapMean,
        AqMapStd,
        NoiseFloorY,
        NoiseFloorUV,
        LineArtScore,
        SkinToneFraction,
        EdgeSlopeStdev,
    ]
}

struct Row {
    corpus: String,
    file: String,
    width: u32,
    height: u32,
    elapsed_us: u128,
    /// One f32 per feature, NaN if absent.
    values: Vec<f32>,
}

fn analyze_path(
    path: &Path,
    corpus: &str,
    query: &AnalysisQuery,
    features: &[AnalysisFeature],
) -> Option<Row> {
    let img = ImageReader::open(path).ok()?.with_guessed_format().ok()?;
    let dyn_img = img.decode().ok()?;
    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let buf = rgb.as_raw();
    let stride = (w as usize) * 3;
    let slice = PixelSlice::new(buf, w, h, stride, PixelDescriptor::RGB8_SRGB).ok()?;

    let start = Instant::now();
    let r: AnalysisResults = zenanalyze::analyze_features(slice, query).ok()?;
    let elapsed = start.elapsed().as_micros();

    let mut values: Vec<f32> = Vec::with_capacity(features.len());
    for &f in features {
        let v = match r.get(f) {
            Some(FeatureValue::F32(x)) => x,
            Some(FeatureValue::U32(x)) => x as f32,
            Some(FeatureValue::Bool(true)) => 1.0,
            Some(FeatureValue::Bool(false)) => 0.0,
            _ => f32::NAN,
        };
        values.push(v);
    }

    Some(Row {
        corpus: corpus.to_string(),
        file: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        width: w,
        height: h,
        elapsed_us: elapsed,
        values,
    })
}

fn list_pngs(dir: &Path, max: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk(dir, &mut out, &["png", "jpg", "jpeg"]);
    out.sort();
    if out.len() > max {
        out.truncate(max);
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>, exts: &[&str]) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out, exts);
        } else if let Some(ext) = p.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if exts.iter().any(|x| x == &ext) {
                out.push(p);
            }
        }
    }
}

fn main() {
    let features = all_features();
    let mut set = FeatureSet::new();
    for &f in &features {
        set = set.with(f);
    }
    let query = AnalysisQuery::new(set);

    // Labeled mode: LABELS_TSV=/path/to/labels.tsv emits CSV with label
    // columns appended. Used for empirical class-conditional recalibration.
    if let Ok(tsv) = std::env::var("LABELS_TSV") {
        run_labeled(&tsv, &query, &features);
        return;
    }

    let mut all_rows: Vec<Row> = Vec::new();
    let max_per_corpus: usize = std::env::var("MAX_PER_CORPUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    let cc = corpus_root();
    let corpora: Vec<(String, PathBuf)> = vec![
        ("CID22-512".into(), cc.join("CID22/CID22-512")),
        ("clic2025-final-test".into(), cc.join("clic2025/final-test")),
        ("gb82".into(), cc.join("gb82")),
        ("gb82-sc".into(), cc.join("gb82-sc")),
    ];

    for (name, root) in &corpora {
        let files = list_pngs(root, max_per_corpus);
        eprintln!("{}: {} files", name, files.len());
        for (i, f) in files.iter().enumerate() {
            if let Some(row) = analyze_path(f, name, &query, &features) {
                all_rows.push(row);
            } else {
                eprintln!("  fail: {}", f.display());
            }
            if i % 20 == 0 {
                eprintln!("  {}/{}", i + 1, files.len());
            }
        }
    }

    // Header
    let mut header = String::from("corpus,file,width,height,elapsed_us");
    for &f in &features {
        header.push(',');
        header.push_str(f.name());
    }
    println!("{}", header);

    for r in &all_rows {
        let mut line = format!("{},{},{},{},{}", r.corpus, r.file, r.width, r.height, r.elapsed_us);
        for v in &r.values {
            line.push(',');
            if v.is_nan() {
                line.push_str("NA");
            } else {
                line.push_str(&format!("{}", v));
            }
        }
        println!("{}", line);
    }

    eprintln!("done — {} rows", all_rows.len());
}

// ---- labeled mode -----------------------------------------------------

/// Resolve the codec-corpus root from `CORPUS_ROOT` or fall back to
/// the standard layout under `~/work/codec-eval/codec-corpus`.
fn corpus_root() -> PathBuf {
    if let Ok(p) = std::env::var("CORPUS_ROOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
    PathBuf::from(home).join("work/codec-eval/codec-corpus")
}

fn run_labeled(tsv: &str, query: &AnalysisQuery, features: &[AnalysisFeature]) {
    let cc = corpus_root();
    // (corpus -> [search dirs relative to corpus_root])
    let resolve_dirs: &[(&str, &[&str])] = &[
        ("cid22-train", &["CID22/CID22-512/training", "CID22/CID22-512"]),
        ("cid22-val", &["CID22/CID22-512/validation", "CID22/CID22-512"]),
        ("clic2025-1024", &["clic2025-1024", "clic2025/final-test", "clic2025/training"]),
        ("gb82", &["gb82"]),
        ("gb82-sc", &["gb82-sc"]),
        ("imageflow", &["imageflow/test_inputs", "imageflow"]),
        ("kadid10k", &["kadid10k"]),
        ("qoi-benchmark", &["qoi-benchmark/screenshot_web", "qoi-benchmark"]),
        ("corpus", &[""]),
    ];

    let labels = fs::read_to_string(tsv).expect("read labels");
    let mut lines = labels.lines();
    let header = lines.next().expect("header");
    let cols: Vec<&str> = header.split('\t').collect();
    let idx_corpus = cols.iter().position(|s| *s == "corpus").unwrap();
    let idx_image = cols.iter().position(|s| *s == "image").unwrap();
    let idx_cat = cols.iter().position(|s| *s == "primary_category").unwrap();
    let idx_synth = cols.iter().position(|s| *s == "is_synthetic").unwrap();
    let idx_palette = cols.iter().position(|s| *s == "palette_size").unwrap();
    let idx_chroma = cols.iter().position(|s| *s == "dominant_chroma").unwrap();
    let idx_text = cols.iter().position(|s| *s == "has_text").unwrap();

    // emit header
    let mut out_h = String::from("corpus,file,width,height,elapsed_us,primary_category,is_synthetic,palette_size,dominant_chroma,has_text");
    for &f in features {
        out_h.push(',');
        out_h.push_str(f.name());
    }
    println!("{}", out_h);

    let mut found = 0usize;
    let mut missing = 0usize;
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() <= idx_text { continue; }
        let corpus = f[idx_corpus];
        let img = f[idx_image];
        let cat = f[idx_cat];
        let synth = f[idx_synth];
        let palette = f[idx_palette];
        let chroma = f[idx_chroma];
        let text = f[idx_text];

        // resolve path
        let dirs = resolve_dirs.iter().find(|(c, _)| *c == corpus).map(|(_, d)| *d).unwrap_or(&[]);
        let mut path: Option<PathBuf> = None;
        for sub in dirs {
            let dir = if sub.is_empty() {
                cc.clone()
            } else {
                cc.join(sub)
            };
            if !dir.is_dir() { continue; }
            let mut found_path: Option<PathBuf> = None;
            walk_find(&dir, img, &mut found_path);
            if let Some(p) = found_path { path = Some(p); break; }
        }
        let Some(p) = path else { missing += 1; eprintln!("MISSING: {}/{}", corpus, img); continue };

        let Some(row) = analyze_path(&p, corpus, query, features) else {
            eprintln!("ANALYZE_FAIL: {}", p.display());
            continue;
        };
        found += 1;
        let mut line = format!(
            "{},{},{},{},{},{},{},{},{},{}",
            row.corpus, row.file, row.width, row.height, row.elapsed_us,
            cat, synth, palette, chroma, text,
        );
        for v in &row.values {
            line.push(',');
            if v.is_nan() { line.push_str("NA"); } else { line.push_str(&format!("{}", v)); }
        }
        println!("{}", line);
        if found % 25 == 0 { eprintln!("  {} rows", found); }
    }
    eprintln!("done — {} found, {} missing", found, missing);
}

fn walk_find(dir: &Path, name: &str, out: &mut Option<PathBuf>) {
    if out.is_some() { return; }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        if out.is_some() { return; }
        let p = e.path();
        if p.is_dir() {
            walk_find(&p, name, out);
        } else if p.file_name().and_then(|s| s.to_str()) == Some(name) {
            *out = Some(p);
            return;
        }
    }
}
