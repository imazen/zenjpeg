//! Corpus evaluation harness — runs analyze_features over multiple
//! corpora with FeatureSet::SUPPORTED and dumps every feature to CSV.
//!
//! NOT FOR COMMIT. Temporary tool used to evaluate just-landed
//! HDR/depth + Tier1/Tier3 piggyback features.

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

    let mut all_rows: Vec<Row> = Vec::new();
    let max_per_corpus: usize = std::env::var("MAX_PER_CORPUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    let corpora: &[(&str, &str)] = &[
        (
            "CID22-512",
            "/home/lilith/work/codec-eval/codec-corpus/CID22/CID22-512",
        ),
        (
            "clic2025-final-test",
            "/home/lilith/work/codec-eval/codec-corpus/clic2025/final-test",
        ),
        ("gb82", "/home/lilith/work/codec-eval/codec-corpus/gb82"),
        (
            "gb82-sc",
            "/home/lilith/work/codec-eval/codec-corpus/gb82-sc",
        ),
    ];

    for (name, root) in corpora {
        let files = list_pngs(Path::new(root), max_per_corpus);
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
