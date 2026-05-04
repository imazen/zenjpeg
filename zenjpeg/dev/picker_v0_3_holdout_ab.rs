//! Held-out re-encode A/B for the v0.3 zenjpeg picker.
//!
//! Compares two encoder arms on a held-out PNG corpus at multiple
//! `target_zq` levels:
//!
//!   * **picker**: 51-feature zenanalyze vector → engineered 112-vec
//!     (same layout as the v0.3 manifest's `feat_cols + extra_axes`)
//!     → externally-loaded v0.3 `.bin` via `zenpredict::Model::from_bytes`
//!     → `predict_transformed` → cell argmin (12-cell taxonomy
//!     `color × sub × trellis_on × sa`) + scalar heads
//!     (`chroma_scale`, `lambda`) → `EncoderConfig::ycbcr`/`xyb`
//!     with the picker-derived `(ChromaSubsampling, hybrid_config,
//!     chroma_distance_scale, optimize_scans)` knobs.
//!   * **bucket**: `EncoderConfig::ycbcr(Quality::Zq(target),
//!     ChromaSubsampling::Quarter)` with codec defaults — the simplest
//!     baseline (no analyzer, no trellis, no chroma scaling).
//!
//! Both arms ride the same `Quality::Zq` closed-loop iteration so they
//! hit comparable achieved zensim scores; the differential is bytes at
//! matched quality.
//!
//! This harness lives in `zenjpeg/dev/` (not registered in Cargo.toml)
//! because it loads the v0.3 `.bin` externally and is not part of the
//! production library API. To run, copy or symlink into
//! `zenjpeg/examples/` and add a `[[example]]` stanza, or pass
//! `--manifest-path` from outside.
//!
//! Usage:
//!   cargo run --release -p zenjpeg \
//!     --features "target-zq trellis" \
//!     --example picker_v0_3_holdout_ab -- \
//!     --bin benchmarks/zenjpeg_picker_v0.3_2026-05-04.bin \
//!     --corpus ~/work/zentrain-corpus/mlp-validate/cid22-val \
//!     --targets 30,35,40,45,50,55,60,65,70,75,80,85,90 \
//!     --out-md benchmarks/picker_v0.3_zenjpeg_2026-05-04.md \
//!     --out-tsv benchmarks/picker_v0.3_holdout_ab_2026-05-04.tsv

#![cfg(all(feature = "target-zq", feature = "trellis"))]
#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use enough::Unstoppable;
use zenanalyze::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
use zenpredict::{AllowedMask, Model, Predictor, ScoreTransform, argmin_masked_in_range};

use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, PixelLayout, Quality, XybSubsampling,
};
use zenjpeg::encode::trellis::HybridConfig;
use zenjpeg::encode::zq::ZqTarget;

// -----------------------------------------------------------------------
// Schema (mirrors the v0.3 manifest.json::feat_cols ordering exactly).
// -----------------------------------------------------------------------

/// 51 features in the order the trainer / .bin expects.
const FEAT_COLS: &[&str] = &[
    "feat_variance",
    "feat_edge_density",
    "feat_uniformity",
    "feat_chroma_complexity",
    "feat_cb_sharpness",
    "feat_cr_sharpness",
    "feat_flat_color_block_ratio",
    "feat_colourfulness",
    "feat_laplacian_variance",
    "feat_variance_spread",
    "feat_grayscale_score",
    "feat_cb_horiz_sharpness",
    "feat_cb_vert_sharpness",
    "feat_cb_peak_sharpness",
    "feat_cr_horiz_sharpness",
    "feat_cr_vert_sharpness",
    "feat_cr_peak_sharpness",
    "feat_high_freq_energy_ratio",
    "feat_luma_histogram_entropy",
    "feat_dct_compressibility_y",
    "feat_dct_compressibility_uv",
    "feat_patch_fraction_fast",
    "feat_quant_survival_y",
    "feat_quant_survival_uv",
    "feat_aq_map_mean",
    "feat_aq_map_std",
    "feat_noise_floor_y",
    "feat_noise_floor_uv",
    "feat_edge_slope_stdev",
    "feat_gradient_fraction",
    "feat_palette_density",
    "feat_alpha_used_fraction",
    "feat_alpha_bimodal_score",
    "feat_pixel_count",
    "feat_log_pixels",
    "feat_aspect_min_over_max",
    "feat_channel_count",
    "feat_aq_map_p75",
    "feat_aq_map_p90",
    "feat_aq_map_p95",
    "feat_aq_map_p99",
    "feat_noise_floor_y_p50",
    "feat_noise_floor_y_p90",
    "feat_laplacian_variance_p50",
    "feat_laplacian_variance_p75",
    "feat_laplacian_variance_p90",
    "feat_laplacian_variance_p99",
    "feat_laplacian_variance_peak",
    "feat_quant_survival_y_p10",
    "feat_luma_kurtosis",
    "feat_gradient_fraction_smooth",
];

const N_FEAT: usize = 51;

/// Per-FEAT_COL `Option<AnalysisFeature>` parallel to FEAT_COLS.
const ANALYSIS_FEATURES: &[Option<AnalysisFeature>] = &[
    Some(AnalysisFeature::Variance),
    Some(AnalysisFeature::EdgeDensity),
    Some(AnalysisFeature::Uniformity),
    Some(AnalysisFeature::ChromaComplexity),
    Some(AnalysisFeature::CbSharpness),
    Some(AnalysisFeature::CrSharpness),
    Some(AnalysisFeature::FlatColorBlockRatio),
    Some(AnalysisFeature::Colourfulness),
    Some(AnalysisFeature::LaplacianVariance),
    Some(AnalysisFeature::VarianceSpread),
    Some(AnalysisFeature::GrayscaleScore),
    Some(AnalysisFeature::CbHorizSharpness),
    Some(AnalysisFeature::CbVertSharpness),
    Some(AnalysisFeature::CbPeakSharpness),
    Some(AnalysisFeature::CrHorizSharpness),
    Some(AnalysisFeature::CrVertSharpness),
    Some(AnalysisFeature::CrPeakSharpness),
    Some(AnalysisFeature::HighFreqEnergyRatio),
    Some(AnalysisFeature::LumaHistogramEntropy),
    Some(AnalysisFeature::DctCompressibilityY),
    Some(AnalysisFeature::DctCompressibilityUV),
    Some(AnalysisFeature::PatchFractionFast),
    Some(AnalysisFeature::QuantSurvivalY),
    Some(AnalysisFeature::QuantSurvivalUv),
    Some(AnalysisFeature::AqMapMean),
    Some(AnalysisFeature::AqMapStd),
    Some(AnalysisFeature::NoiseFloorY),
    Some(AnalysisFeature::NoiseFloorUV),
    Some(AnalysisFeature::EdgeSlopeStdev),
    Some(AnalysisFeature::GradientFraction),
    Some(AnalysisFeature::PaletteDensity),
    Some(AnalysisFeature::AlphaUsedFraction),
    Some(AnalysisFeature::AlphaBimodalScore),
    Some(AnalysisFeature::PixelCount),
    Some(AnalysisFeature::LogPixels),
    Some(AnalysisFeature::AspectMinOverMax),
    Some(AnalysisFeature::ChannelCount),
    Some(AnalysisFeature::AqMapP75),
    Some(AnalysisFeature::AqMapP90),
    Some(AnalysisFeature::AqMapP95),
    Some(AnalysisFeature::AqMapP99),
    Some(AnalysisFeature::NoiseFloorYP50),
    Some(AnalysisFeature::NoiseFloorYP90),
    Some(AnalysisFeature::LaplacianVarianceP50),
    Some(AnalysisFeature::LaplacianVarianceP75),
    Some(AnalysisFeature::LaplacianVarianceP90),
    Some(AnalysisFeature::LaplacianVarianceP99),
    Some(AnalysisFeature::LaplacianVariancePeak),
    Some(AnalysisFeature::QuantSurvivalYP10),
    Some(AnalysisFeature::LumaKurtosis),
    Some(AnalysisFeature::GradientFractionSmooth),
];

/// 12-cell taxonomy in the trainer's lex-sorted order
/// (color, sub, trellis_on, sa). Verified against
/// `hybrid_heads_manifest.cells` in the .json.
#[derive(Clone, Copy, Debug)]
struct CellSpec {
    color: ColorChoice,
    sub_420: bool,
    trellis_on: bool,
    sa: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColorChoice {
    Ycbcr,
    Xyb,
}

const CELLS: &[CellSpec] = &[
    CellSpec { color: ColorChoice::Xyb,   sub_420: true,  trellis_on: false, sa: false }, // 0  xyb_420_noT
    CellSpec { color: ColorChoice::Xyb,   sub_420: true,  trellis_on: true,  sa: false }, // 1  xyb_420_trellis
    CellSpec { color: ColorChoice::Xyb,   sub_420: false, trellis_on: false, sa: false }, // 2  xyb_444_noT
    CellSpec { color: ColorChoice::Xyb,   sub_420: false, trellis_on: true,  sa: false }, // 3  xyb_444_trellis
    CellSpec { color: ColorChoice::Ycbcr, sub_420: true,  trellis_on: false, sa: false }, // 4  ycbcr_420_noT
    CellSpec { color: ColorChoice::Ycbcr, sub_420: true,  trellis_on: false, sa: true  }, // 5  ycbcr_420_noT_sa
    CellSpec { color: ColorChoice::Ycbcr, sub_420: true,  trellis_on: true,  sa: false }, // 6  ycbcr_420_trellis
    CellSpec { color: ColorChoice::Ycbcr, sub_420: true,  trellis_on: true,  sa: true  }, // 7  ycbcr_420_trellis_sa
    CellSpec { color: ColorChoice::Ycbcr, sub_420: false, trellis_on: false, sa: false }, // 8  ycbcr_444_noT
    CellSpec { color: ColorChoice::Ycbcr, sub_420: false, trellis_on: false, sa: true  }, // 9  ycbcr_444_noT_sa
    CellSpec { color: ColorChoice::Ycbcr, sub_420: false, trellis_on: true,  sa: false }, // 10 ycbcr_444_trellis
    CellSpec { color: ColorChoice::Ycbcr, sub_420: false, trellis_on: true,  sa: true  }, // 11 ycbcr_444_trellis_sa
];

const N_CELLS: usize = 12;

// Output layout (verified from train_hybrid.py output_layout build):
//   bytes_log[0..12], time_log[12..24], chroma_scale[24..36], lambda[36..48]
const RANGE_BYTES_LOG: (usize, usize) = (0, 12);
const OFF_CHROMA_SCALE: usize = 24;
const OFF_LAMBDA: usize = 36;

const N_OUTPUTS: usize = 48;

// -----------------------------------------------------------------------
// Per-feature pre-engineering transform (from .bin's stripped
// `zentrain.feature_transforms` metadata, dumped from the JSON).
// 51 entries, parallel to FEAT_COLS / ANALYSIS_FEATURES.
// -----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RawTransform {
    Identity,
    Log,
    Log1p,
}

impl RawTransform {
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::Identity => x,
            Self::Log => x.ln(),
            Self::Log1p => x.ln_1p(),
        }
    }
}

const RAW_TRANSFORMS_V0_3: &[RawTransform] = &[
    RawTransform::Log1p,    // 0  feat_variance
    RawTransform::Identity, // 1  feat_edge_density
    RawTransform::Identity, // 2  feat_uniformity
    RawTransform::Identity, // 3  feat_chroma_complexity
    RawTransform::Identity, // 4  feat_cb_sharpness
    RawTransform::Identity, // 5  feat_cr_sharpness
    RawTransform::Identity, // 6  feat_flat_color_block_ratio
    RawTransform::Identity, // 7  feat_colourfulness
    RawTransform::Log1p,    // 8  feat_laplacian_variance
    RawTransform::Log1p,    // 9  feat_variance_spread
    RawTransform::Identity, // 10 feat_grayscale_score
    RawTransform::Identity, // 11 feat_cb_horiz_sharpness
    RawTransform::Identity, // 12 feat_cb_vert_sharpness
    RawTransform::Identity, // 13 feat_cb_peak_sharpness
    RawTransform::Identity, // 14 feat_cr_horiz_sharpness
    RawTransform::Identity, // 15 feat_cr_vert_sharpness
    RawTransform::Identity, // 16 feat_cr_peak_sharpness
    RawTransform::Identity, // 17 feat_high_freq_energy_ratio
    RawTransform::Identity, // 18 feat_luma_histogram_entropy
    RawTransform::Identity, // 19 feat_dct_compressibility_y
    RawTransform::Identity, // 20 feat_dct_compressibility_uv
    RawTransform::Identity, // 21 feat_patch_fraction_fast
    RawTransform::Identity, // 22 feat_quant_survival_y
    RawTransform::Identity, // 23 feat_quant_survival_uv
    RawTransform::Identity, // 24 feat_aq_map_mean
    RawTransform::Identity, // 25 feat_aq_map_std
    RawTransform::Identity, // 26 feat_noise_floor_y
    RawTransform::Identity, // 27 feat_noise_floor_uv
    RawTransform::Log1p,    // 28 feat_edge_slope_stdev
    RawTransform::Identity, // 29 feat_gradient_fraction
    RawTransform::Identity, // 30 feat_palette_density
    RawTransform::Identity, // 31 feat_alpha_used_fraction
    RawTransform::Identity, // 32 feat_alpha_bimodal_score
    RawTransform::Log,      // 33 feat_pixel_count
    RawTransform::Identity, // 34 feat_log_pixels
    RawTransform::Identity, // 35 feat_aspect_min_over_max
    RawTransform::Identity, // 36 feat_channel_count
    RawTransform::Identity, // 37 feat_aq_map_p75
    RawTransform::Identity, // 38 feat_aq_map_p90
    RawTransform::Identity, // 39 feat_aq_map_p95
    RawTransform::Log1p,    // 40 feat_aq_map_p99
    RawTransform::Identity, // 41 feat_noise_floor_y_p50
    RawTransform::Identity, // 42 feat_noise_floor_y_p90
    RawTransform::Log1p,    // 43 feat_laplacian_variance_p50
    RawTransform::Log1p,    // 44 feat_laplacian_variance_p75
    RawTransform::Log1p,    // 45 feat_laplacian_variance_p90
    RawTransform::Log1p,    // 46 feat_laplacian_variance_p99
    RawTransform::Log1p,    // 47 feat_laplacian_variance_peak
    RawTransform::Identity, // 48 feat_quant_survival_y_p10
    RawTransform::Identity, // 49 feat_luma_kurtosis
    RawTransform::Identity, // 50 feat_gradient_fraction_smooth
];

// -----------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------

#[derive(Default)]
struct Cli {
    bin: PathBuf,
    corpus: PathBuf,
    targets: Vec<f32>,
    out_md: Option<PathBuf>,
    out_tsv: Option<PathBuf>,
    max_passes: u8,
    max_images: usize,
}

fn parse_args() -> Cli {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut cli = Cli {
        targets: vec![
            30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0,
        ],
        max_passes: 3,
        max_images: 0,
        ..Default::default()
    };
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        let next = || -> &str { argv.get(i + 1).map(String::as_str).unwrap_or("") };
        match a.as_str() {
            "--bin" => {
                cli.bin = PathBuf::from(next());
                i += 2;
            }
            "--corpus" => {
                cli.corpus = PathBuf::from(next());
                i += 2;
            }
            "--targets" => {
                cli.targets = next()
                    .split(',')
                    .map(|s| s.trim().parse::<f32>().expect("bad --targets value"))
                    .collect();
                i += 2;
            }
            "--out-md" => {
                cli.out_md = Some(PathBuf::from(next()));
                i += 2;
            }
            "--out-tsv" => {
                cli.out_tsv = Some(PathBuf::from(next()));
                i += 2;
            }
            "--max-passes" => {
                cli.max_passes = next().parse().expect("bad --max-passes");
                i += 2;
            }
            "--max-images" => {
                cli.max_images = next().parse().expect("bad --max-images");
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                usage_and_exit();
            }
        }
    }
    if cli.bin.as_os_str().is_empty() || cli.corpus.as_os_str().is_empty() {
        usage_and_exit();
    }
    cli
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage: picker_v0_3_holdout_ab \n\
         \t--bin <v0.3.bin> --corpus <dir> [--targets 30,35,...,90] \n\
         \t[--out-md path] [--out-tsv path] [--max-passes 3] [--max-images N]"
    );
    std::process::exit(2);
}

// -----------------------------------------------------------------------
// PNG load
// -----------------------------------------------------------------------

struct DecodedPng {
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn decode_png(path: &Path) -> Option<DecodedPng> {
    let img = image::open(path).ok()?.to_rgb8();
    let w = img.width();
    let h = img.height();
    Some(DecodedPng {
        rgb: img.into_raw(),
        w,
        h,
    })
}

fn list_pngs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .is_some_and(|e| e == "png" || e == "PNG")
        })
        .collect();
    v.sort();
    v
}

// -----------------------------------------------------------------------
// Feature extraction + engineered vector
// -----------------------------------------------------------------------

fn extract_raw_features_rgb8(rgb: &[u8], width: u32, height: u32) -> Vec<f32> {
    let mut feats = FeatureSet::new();
    for slot in ANALYSIS_FEATURES {
        if let Some(f) = slot {
            feats = feats.with(*f);
        }
    }
    let query = AnalysisQuery::new(feats);
    let analysis = zenanalyze::analyze_features_rgb8(rgb, width, height, &query);
    ANALYSIS_FEATURES
        .iter()
        .map(|slot| match slot {
            Some(f) => analysis.get_f32(*f).unwrap_or(0.0),
            None => 0.0,
        })
        .collect()
}

/// Apply per-feature transforms (log/log1p/identity) in-place, mirror
/// of the trainer's pre-engineering transform pass.
fn transform_raw_in_place(raw: &mut [f32]) {
    debug_assert_eq!(raw.len(), RAW_TRANSFORMS_V0_3.len());
    for (i, x) in raw.iter_mut().enumerate() {
        *x = RAW_TRANSFORMS_V0_3[i].apply(*x);
    }
}

/// Engineered feature vector for the v0.3 zenjpeg picker.
///
/// Layout (matches manifest.json::extra_axes):
///   raw[51] || size_oh[4] || poly[5] || cross[51] || icc[1]  = 112
/// where:
///   size_oh = 4-bucket one-hot (`tiny<64*64`, `small<256*256`, `medium<1024*1024`, `large`)
///   poly    = `[log_px, log_px^2, zq_norm, zq_norm^2, zq_norm * log_px]`
///   cross   = `zq_norm * raw[i]`
fn engineered_features(raw_feats: &[f32], width: u32, height: u32, target_zq: f32) -> Vec<f32> {
    debug_assert_eq!(raw_feats.len(), N_FEAT);
    let pixels = (width as f32) * (height as f32);
    let log_px = pixels.max(1.0).ln();
    let zq_norm = target_zq / 100.0;

    let size_oh = match (width as u64) * (height as u64) {
        n if n < 64 * 64 => [1.0_f32, 0.0, 0.0, 0.0],
        n if n < 256 * 256 => [0.0, 1.0, 0.0, 0.0],
        n if n < 1024 * 1024 => [0.0, 0.0, 1.0, 0.0],
        _ => [0.0, 0.0, 0.0, 1.0],
    };

    let mut out = Vec::with_capacity(N_FEAT + 4 + 5 + N_FEAT + 1);
    out.extend_from_slice(raw_feats);
    out.extend_from_slice(&size_oh);
    out.extend_from_slice(&[
        log_px,
        log_px * log_px,
        zq_norm,
        zq_norm * zq_norm,
        zq_norm * log_px,
    ]);
    for f in raw_feats {
        out.push(zq_norm * f);
    }
    out.push(0.0); // icc_bytes — not plumbed (matches runtime convention)
    out
}

// -----------------------------------------------------------------------
// Picker inference → encoder knobs
// -----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct PickerKnobs {
    cell_idx: usize,
    spec: CellSpec,
    chroma_scale: f32,
    lambda: f32,
}

fn pick_knobs(predictor: &mut Predictor<'_>, feats: &[f32]) -> PickerKnobs {
    // Feed pre-transformed features (transforms applied at raw extraction
    // time, before engineering). The .bin's `feature_transforms`
    // metadata mirrors RAW_TRANSFORMS_V0_3 but `predict()` is the safer
    // entry — it does not re-apply transforms even if metadata is
    // present.
    let output = predictor.predict(feats).expect("predict");
    assert_eq!(output.len(), N_OUTPUTS);
    let mask_arr = [true; N_CELLS];
    let mask = AllowedMask::new(&mask_arr);
    let cell_idx = argmin_masked_in_range(output, RANGE_BYTES_LOG, &mask, ScoreTransform::Exp, None)
        .expect("argmin");
    assert!(cell_idx < N_CELLS);
    let spec = CELLS[cell_idx];
    let chroma_scale = clamp_f32(output[OFF_CHROMA_SCALE + cell_idx], 0.6, 1.5);
    // Snap lambda to the trainer's discrete set {0, 8.0, 14.5, 25.0}.
    let lambda_raw = clamp_f32(output[OFF_LAMBDA + cell_idx], 0.0, 25.0);
    let lambda = if !spec.trellis_on {
        0.0
    } else {
        snap_to_set(lambda_raw, &[8.0, 14.5, 25.0])
    };
    PickerKnobs {
        cell_idx,
        spec,
        chroma_scale,
        lambda,
    }
}

fn clamp_f32(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() {
        lo
    } else {
        v.max(lo).min(hi)
    }
}

fn snap_to_set(v: f32, set: &[f32]) -> f32 {
    let mut best = set[0];
    let mut best_d = (v - best).abs();
    for &s in &set[1..] {
        let d = (v - s).abs();
        if d < best_d {
            best = s;
            best_d = d;
        }
    }
    best
}

// -----------------------------------------------------------------------
// Encode arms (each rides Quality::Zq closed loop)
// -----------------------------------------------------------------------

struct EncodeOutcome {
    bytes: usize,
    achieved: f32,
    passes: u8,
    targets_met: bool,
    elapsed_ms: f64,
}

fn build_picker_config(spec: CellSpec, lambda: f32, chroma_scale: f32, target: f32) -> EncoderConfig {
    let zq = Quality::ZqExplicit(
        ZqTarget::new(target)
            .with_max_overshoot(Some(1.5))
            .with_max_passes(3),
    );
    let mut cfg = match spec.color {
        ColorChoice::Ycbcr => {
            let sub = if spec.sub_420 {
                ChromaSubsampling::Quarter
            } else {
                ChromaSubsampling::None
            };
            EncoderConfig::ycbcr(zq, sub)
        }
        ColorChoice::Xyb => {
            let xyb_sub = if spec.sub_420 {
                XybSubsampling::BQuarter
            } else {
                XybSubsampling::Full
            };
            EncoderConfig::xyb(zq, xyb_sub)
        }
    };
    if spec.trellis_on && lambda > 0.0 {
        cfg = cfg.hybrid_config(HybridConfig {
            enabled: true,
            base_lambda_scale1: lambda,
            chroma_scale,
            ..HybridConfig::default()
        });
    } else {
        // Even without trellis, apply chroma_scale via
        // chroma_distance_scale so the picker's chroma signal still
        // affects the output. (HybridConfig.chroma_scale only fires
        // when trellis is on.)
        cfg = cfg.chroma_distance_scale(chroma_scale);
    }
    if spec.sa {
        // SA-piecewise tables (CID22-tuned). When the picker selects a
        // `sa` cell the trainer expects `optimize_scans` to be on too,
        // matching `_sa` config-name semantics.
        cfg = cfg.optimize_scans(true);
    }
    cfg
}

fn build_bucket_config(target: f32) -> EncoderConfig {
    // The simplest baseline: ycbcr 4:2:0 with default Jpegli tables and
    // closed-loop Zq targeting. No analyzer, no trellis, no chroma
    // scaling — what a caller gets from
    // `EncoderConfig::ycbcr(Quality::Zq(t), ChromaSubsampling::Quarter)`.
    let zq = Quality::ZqExplicit(
        ZqTarget::new(target)
            .with_max_overshoot(Some(1.5))
            .with_max_passes(3),
    );
    EncoderConfig::ycbcr(zq, ChromaSubsampling::Quarter)
}

fn encode_with(cfg: &EncoderConfig, rgb: &[u8], w: u32, h: u32) -> Option<EncodeOutcome> {
    let t0 = Instant::now();
    let mut enc = match cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("ERR encode_from_bytes: {e:?}");
            return None;
        }
    };
    if let Err(e) = enc.push_packed(rgb, Unstoppable) {
        eprintln!("ERR push_packed: {e:?}");
        return None;
    }
    match enc.finish_with_metrics() {
        Ok((bytes, m)) => Some(EncodeOutcome {
            bytes: bytes.len(),
            achieved: m.achieved_score,
            passes: m.passes_used,
            targets_met: m.targets_met,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }),
        Err(e) => {
            eprintln!("ERR finish_with_metrics: {e:?}");
            None
        }
    }
}

// -----------------------------------------------------------------------
// Aggregate row + report
// -----------------------------------------------------------------------

#[derive(Default, Clone)]
struct CellStats {
    n: u32,
    bytes_sum: u64,
    achieved_sum: f64,
}

impl CellStats {
    fn add(&mut self, bytes: usize, achieved: f32) {
        self.n += 1;
        self.bytes_sum += bytes as u64;
        self.achieved_sum += f64::from(achieved);
    }
    fn mean_achieved(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.achieved_sum / f64::from(self.n)
        }
    }
}

fn fmt_bytes(b: u64) -> String {
    if b > 1_000_000 {
        format!("{:.2} MB", b as f64 / 1_000_000.0)
    } else {
        format!("{} B", b)
    }
}

fn band_for(target: f32) -> &'static str {
    if target < 50.0 {
        "low"
    } else if target < 75.0 {
        "mid"
    } else {
        "high"
    }
}

fn main() {
    let cli = parse_args();
    eprintln!("v0.3 zenjpeg picker A/B held-out:");
    eprintln!("  bin: {}", cli.bin.display());
    eprintln!("  corpus: {}", cli.corpus.display());
    eprintln!("  targets: {:?}", cli.targets);

    let bin_bytes = fs::read(&cli.bin).expect("read --bin");
    let model = Model::from_bytes(&bin_bytes).expect("Model::from_bytes (v0.3)");
    eprintln!(
        "  model: n_inputs={}, n_outputs={}, schema_hash=0x{:016x}",
        model.n_inputs(),
        model.n_outputs(),
        model.schema_hash()
    );
    assert_eq!(model.n_inputs(), 112, "expected 112 inputs (51 + 4 + 5 + 51 + 1)");
    assert_eq!(model.n_outputs(), 48, "expected 48 outputs (12 cells × 4 heads)");
    let schema_hash = model.schema_hash();
    let mut predictor = Predictor::new(model);

    let mut images_paths = list_pngs(&cli.corpus);
    if cli.max_images > 0 && images_paths.len() > cli.max_images {
        images_paths.truncate(cli.max_images);
    }
    eprintln!("  found {} PNGs", images_paths.len());
    let mut images: Vec<(PathBuf, DecodedPng)> = Vec::new();
    for p in images_paths {
        if let Some(d) = decode_png(&p) {
            images.push((p, d));
        }
    }
    eprintln!("  loaded {} images", images.len());

    let mut tsv_writer: Option<fs::File> = cli.out_tsv.as_ref().map(|p| {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).ok();
        }
        let mut f = fs::File::create(p).expect("open --out-tsv");
        writeln!(
            f,
            "arm\timage\twidth\theight\ttarget\tcell_idx\tcolor\tsub\ttrellis_on\tsa\tchroma_scale\tlambda\tbytes\tachieved\tpasses\tmet\tencode_ms"
        )
        .unwrap();
        f
    });

    let t0 = Instant::now();
    let mut picker_per_target: Vec<CellStats> = vec![CellStats::default(); cli.targets.len()];
    let mut bucket_per_target: Vec<CellStats> = vec![CellStats::default(); cli.targets.len()];
    let mut picker_total = CellStats::default();
    let mut bucket_total = CellStats::default();
    let mut picker_wins_per_target: Vec<u32> = vec![0; cli.targets.len()];
    let mut paired_count_per_target: Vec<u32> = vec![0; cli.targets.len()];
    // Per-band totals (low <50, mid 50-74, high 75+).
    let mut picker_band: std::collections::BTreeMap<&'static str, CellStats> =
        std::collections::BTreeMap::new();
    let mut bucket_band: std::collections::BTreeMap<&'static str, CellStats> =
        std::collections::BTreeMap::new();

    for (idx, (path, d)) in images.iter().enumerate() {
        if idx % 4 == 0 {
            // Refresh workongoing every few images.
            let _ = std::fs::write(
                "/home/lilith/work/zen/zenjpeg/.workongoing",
                format!(
                    "{} claude-zenjpeg-picker-v0.3 phase-3-encoding {}/{}\n",
                    chrono_now_iso(),
                    idx,
                    images.len()
                ),
            );
        }
        let mut raw_feats = extract_raw_features_rgb8(&d.rgb, d.w, d.h);
        transform_raw_in_place(&mut raw_feats);

        for (ti, &target) in cli.targets.iter().enumerate() {
            // Picker arm: re-engineer features per target (zq_norm differs).
            let feats = engineered_features(&raw_feats, d.w, d.h, target);
            let pk = pick_knobs(&mut predictor, &feats);
            let cfg_p = build_picker_config(pk.spec, pk.lambda, pk.chroma_scale, target);
            let p = encode_with(&cfg_p, &d.rgb, d.w, d.h);

            let cfg_b = build_bucket_config(target);
            let b = encode_with(&cfg_b, &d.rgb, d.w, d.h);

            if let Some(po) = &p {
                picker_per_target[ti].add(po.bytes, po.achieved);
                picker_total.add(po.bytes, po.achieved);
                picker_band
                    .entry(band_for(target))
                    .or_default()
                    .add(po.bytes, po.achieved);
                if let Some(f) = tsv_writer.as_mut() {
                    writeln!(
                        f,
                        "picker\t{}\t{}\t{}\t{}\t{}\t{:?}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{:.3}\t{}\t{}\t{:.2}",
                        path.display(),
                        d.w,
                        d.h,
                        target,
                        pk.cell_idx,
                        pk.spec.color,
                        if pk.spec.sub_420 { "420" } else { "444" },
                        pk.spec.trellis_on as u8,
                        pk.spec.sa as u8,
                        pk.chroma_scale,
                        pk.lambda,
                        po.bytes,
                        po.achieved,
                        po.passes,
                        po.targets_met as u8,
                        po.elapsed_ms,
                    )
                    .unwrap();
                }
            }
            if let Some(bo) = &b {
                bucket_per_target[ti].add(bo.bytes, bo.achieved);
                bucket_total.add(bo.bytes, bo.achieved);
                bucket_band
                    .entry(band_for(target))
                    .or_default()
                    .add(bo.bytes, bo.achieved);
                if let Some(f) = tsv_writer.as_mut() {
                    writeln!(
                        f,
                        "bucket\t{}\t{}\t{}\t{}\t-\tycbcr\t420\t0\t0\t1.0000\t0.0000\t{}\t{:.3}\t{}\t{}\t{:.2}",
                        path.display(),
                        d.w,
                        d.h,
                        target,
                        bo.bytes,
                        bo.achieved,
                        bo.passes,
                        bo.targets_met as u8,
                        bo.elapsed_ms,
                    )
                    .unwrap();
                }
            }
            if let (Some(po), Some(bo)) = (&p, &b) {
                paired_count_per_target[ti] += 1;
                if po.bytes <= bo.bytes {
                    picker_wins_per_target[ti] += 1;
                }
            }
        }
        if let Some(f) = tsv_writer.as_mut() {
            f.flush().ok();
        }
    }

    if let Some(mut f) = tsv_writer {
        f.flush().ok();
    }

    // ---------- Markdown report ----------
    let elapsed_total = t0.elapsed().as_secs_f64();
    let mut md = String::new();
    md.push_str("# Picker v0.3 — held-out re-encode A/B (zenjpeg)\n\n");
    md.push_str(&format!(
        "* Date: {}\n* Corpus: `{}` ({} images)\n* Picker bin: `{}` (n_inputs=112, n_outputs=48, schema_hash=`0x{:016x}`)\n* Targets: {:?}\n* Encoder closed-loop: `Quality::ZqExplicit` with `max_passes={}`, `max_overshoot=1.5`\n* Wall: {:.1} s\n\n",
        chrono_now_iso(),
        cli.corpus.display(),
        images.len(),
        cli.bin.display(),
        schema_hash,
        cli.targets,
        cli.max_passes,
        elapsed_total,
    ));

    md.push_str("## Per-target table\n\n");
    md.push_str("| target_zq | n | bytes_picker | bytes_bucket | Δ% (picker − bucket) | win_rate | achieved_picker | achieved_bucket |\n");
    md.push_str("|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (ti, &target) in cli.targets.iter().enumerate() {
        let p = &picker_per_target[ti];
        let b = &bucket_per_target[ti];
        let bp = p.bytes_sum as f64;
        let bb = b.bytes_sum as f64;
        let delta_pct = if bb > 0.0 { (bp - bb) / bb * 100.0 } else { 0.0 };
        let wins = picker_wins_per_target[ti];
        let paired = paired_count_per_target[ti].max(1);
        let win_rate = wins as f64 / paired as f64 * 100.0;
        md.push_str(&format!(
            "| {:.0} | {} | {} | {} | {:+.2}% | {:.0}% ({}/{}) | {:.2} | {:.2} |\n",
            target,
            p.n,
            fmt_bytes(p.bytes_sum),
            fmt_bytes(b.bytes_sum),
            delta_pct,
            win_rate,
            wins,
            paired,
            p.mean_achieved(),
            b.mean_achieved(),
        ));
    }

    md.push_str("\n## Per-band totals\n\n");
    md.push_str("| band | range | bytes_picker | bytes_bucket | Δ% | achieved_picker | achieved_bucket |\n");
    md.push_str("|:--|:--|---:|---:|---:|---:|---:|\n");
    let band_ranges = [("low", "zq < 50"), ("mid", "50 ≤ zq < 75"), ("high", "zq ≥ 75")];
    for (band, range) in band_ranges {
        let p = picker_band.get(band).cloned().unwrap_or_default();
        let b = bucket_band.get(band).cloned().unwrap_or_default();
        let bp = p.bytes_sum as f64;
        let bb = b.bytes_sum as f64;
        let delta_pct = if bb > 0.0 { (bp - bb) / bb * 100.0 } else { 0.0 };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:+.2}% | {:.2} | {:.2} |\n",
            band,
            range,
            fmt_bytes(p.bytes_sum),
            fmt_bytes(b.bytes_sum),
            delta_pct,
            p.mean_achieved(),
            b.mean_achieved(),
        ));
    }

    let bp = picker_total.bytes_sum as f64;
    let bb = bucket_total.bytes_sum as f64;
    let total_delta_pct = if bb > 0.0 { (bp - bb) / bb * 100.0 } else { 0.0 };
    md.push_str(&format!(
        "\n## Total\n\n* Picker total bytes: **{}** (mean achieved zensim {:.2})\n* Bucket total bytes: **{}** (mean achieved zensim {:.2})\n* Δ bytes: **{:+.2}%** (picker − bucket)\n* Δ achieved zensim: **{:+.3}** pp\n",
        fmt_bytes(picker_total.bytes_sum),
        picker_total.mean_achieved(),
        fmt_bytes(bucket_total.bytes_sum),
        bucket_total.mean_achieved(),
        total_delta_pct,
        picker_total.mean_achieved() - bucket_total.mean_achieved(),
    ));

    let achieved_gap = (picker_total.mean_achieved() - bucket_total.mean_achieved()).abs();
    let verdict = if total_delta_pct <= 0.0 && achieved_gap <= 0.5 {
        "**SHIP**"
    } else if total_delta_pct < 0.0 && achieved_gap > 0.5 {
        "HOLD (achieved-zensim gap > 0.5pp invalidates byte comparison)"
    } else {
        "**HOLD**"
    };
    md.push_str(&format!(
        "\n## Verdict\n\n{}\n\n* Threshold: SHIP if total bytes (picker) ≤ total bytes (bucket) within ±0.5pp achieved-zensim parity.\n",
        verdict,
    ));

    md.push_str("\n## Method notes\n\n");
    md.push_str(&format!(
        "* Picker arm: extracted 51-feature zenanalyze vector in `FEAT_COLS` order, applied per-feature transforms (log/log1p/identity per the v0.3 trainer's `feature_transforms`), built the engineered 112-vec via `feats[51] || size_oh[4] || poly[5] || zq*feats[51] || icc[1]` (mirror of the v0.3 manifest's `extra_axes`), ran `Predictor::predict` against the externally-loaded v0.3 `.bin`, decoded the `bytes_log[0..12]` argmin → cell index → `(color, sub, trellis_on, sa)` from the lex-sorted 12-cell taxonomy, and read `chroma_scale` (clamped to [0.6, 1.5]) and `lambda` (snapped to {{0, 8.0, 14.5, 25.0}}) from the per-cell scalar heads at offsets {} and {}.\n",
        OFF_CHROMA_SCALE, OFF_LAMBDA,
    ));
    md.push_str("* Cell → encoder mapping: `EncoderConfig::ycbcr(zq, sub)` or `EncoderConfig::xyb(zq, b_sub)` based on cell color; if `trellis_on` and `lambda > 0`, apply `hybrid_config(HybridConfig { base_lambda_scale1: lambda, chroma_scale, .. })`; otherwise apply `chroma_distance_scale(chroma_scale)`. `sa` cells additionally enable `optimize_scans(true)`.\n");
    md.push_str("* Bucket arm: `EncoderConfig::ycbcr(Quality::ZqExplicit(target), ChromaSubsampling::Quarter)` — codec defaults, no analyzer-derived knobs. The simplest baseline a caller would get from a one-line `EncoderConfig::ycbcr(...)` call.\n");
    md.push_str("* Both arms ride the same `Quality::ZqExplicit` closed loop so the iteration adapts the underlying jpegli quality to land in the target zensim band.\n");
    md.push_str("* FEAT_COLS source: hardcoded from `benchmarks/zenjpeg_hybrid_v0.3_2026-05-04.json::feat_cols` (51 entries, matches `zenjpeg_picker_v0.3_2026-05-04.manifest.json::feat_cols` exactly). Engineered axes (61 = size_oh[4] + poly[5] + zq×feats[51] + icc[1]) match `manifest.json::extra_axes` order.\n");

    if let Some(p) = &cli.out_md {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(p, &md).expect("write --out-md");
        eprintln!("wrote markdown report: {}", p.display());
    } else {
        println!("{md}");
    }

    eprintln!(
        "summary: picker {} ({:.2}) vs bucket {} ({:.2}) delta {:+.2}% over {} images × {} targets in {:.1}s",
        fmt_bytes(picker_total.bytes_sum),
        picker_total.mean_achieved(),
        fmt_bytes(bucket_total.bytes_sum),
        bucket_total.mean_achieved(),
        total_delta_pct,
        images.len(),
        cli.targets.len(),
        elapsed_total,
    );
}

// Tiny ISO-8601 UTC formatter (no chrono dep).
fn chrono_now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let mut hms = secs % 86_400;
    let h = hms / 3600;
    hms %= 3600;
    let m = hms / 60;
    let s = hms % 60;
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m_civil = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m_civil <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m_civil:02}-{d:02}T{h:02}:{mm:02}:{ss:02}Z",
        m_civil = m_civil,
        d = d,
        h = h,
        mm = m,
        ss = s,
    )
}
