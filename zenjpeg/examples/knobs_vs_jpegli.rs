//! Rate-distortion comparison of zenjpeg encoding approaches vs C++ jpegli.
//!
//! Compares jpegli (AQ, no trellis), mozjpeg-style trellis (no AQ), and hybrid
//! (AQ + trellis) approaches on both SSIMULACRA2 and Butteraugli metrics.
//!
//! Usage:
//! ```bash
//! # gb82 corpus (25 images, 576x576) — default
//! cargo run --release -p zenjpeg --example knobs_vs_jpegli
//!
//! # CID22 corpus
//! cargo run --release -p zenjpeg --example knobs_vs_jpegli -- \
//!     --corpus ~/work/codec-eval/codec-corpus/CID22/CID22-512/validation
//!
//! # With CSV output
//! cargo run --release -p zenjpeg --example knobs_vs_jpegli -- \
//!     --output /mnt/v/output/zenjpeg/approach_rd.csv
//! ```

use enough::Unstoppable;
use std::io::Write;
use std::path::PathBuf;
#[cfg(feature = "optimized-tables")]
use zenjpeg::encode::OptimizedTables;
use zenjpeg::encode::search::ExpertConfig;
use zenjpeg::encode::{
    ChromaSubsampling, ColorMode, EncoderConfig, OptimizationPreset, PixelLayout,
};
use zenjpeg_bench_utils::{
    ChromaSubsampling as BenchSub, ColorMode as BenchColor, EncoderConfig as BenchEncoderConfig,
    EncoderImpl, ImageData, QualityMetrics, RgbImage, ScanMode, bytes_to_rgb, decode_jpeg_with_icc,
};

fn default_corpus_dir() -> PathBuf {
    let cc = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    cc.get("gb82").expect("gb82 corpus not available")
}

/// Quality levels spanning the useful range for JPEG.
const QUALITY_LEVELS: [f32; 10] = [98.0, 95.0, 92.0, 90.0, 88.0, 85.0, 80.0, 75.0, 65.0, 50.0];

/// C++ distances matching approximately the same quality range.
const CPP_DISTANCES: [f32; 10] = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0];

/// Lambda values for trellis presets.
const LAMBDAS: [f32; 3] = [12.0, 14.5, 16.0];

struct Args {
    corpus: PathBuf,
    output: Option<PathBuf>,
    max_images: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        corpus: default_corpus_dir(),
        output: None,
        max_images: 50,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--corpus" => {
                if let Some(s) = iter.next() {
                    let expanded = if s.starts_with('~') {
                        if let Some(home) = std::env::var_os("HOME") {
                            PathBuf::from(home).join(&s[2..])
                        } else {
                            PathBuf::from(s)
                        }
                    } else {
                        PathBuf::from(s)
                    };
                    args.corpus = expanded;
                }
            }
            "--output" => args.output = iter.next().map(PathBuf::from),
            "--images" => {
                args.max_images = iter.next().and_then(|s| s.parse().ok()).unwrap_or(50);
            }
            "--help" | "-h" => {
                eprintln!("Usage: knobs_vs_jpegli [OPTIONS]");
                eprintln!("  --corpus <dir>     Image directory (default: gb82)");
                eprintln!("  --output <csv>     Write CSV results");
                eprintln!("  --images <N>       Max images (default: 50)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }
    args
}

struct Metrics {
    bpp: f64,
    ssim2: f64,
    ba: f64,
    bytes: usize,
}

fn compute_metrics(img: &ImageData, jpeg: &[u8]) -> Option<Metrics> {
    let px = (img.width * img.height) as f64;
    let orig = bytes_to_rgb(&img.pixels, img.width, img.height);
    // Use zenjpeg decoder — zune-jpeg fails on zenjpeg's progressive scan structure
    let dec: RgbImage = decode_jpeg_with_icc(jpeg).ok()?;
    Some(Metrics {
        bpp: jpeg.len() as f64 * 8.0 / px,
        ssim2: QualityMetrics::ssimulacra2(orig.as_ref(), dec.as_ref()),
        ba: QualityMetrics::butteraugli(orig.as_ref(), dec.as_ref()),
        bytes: jpeg.len(),
    })
}

fn encode_zen(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut e = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    e.push_packed(&img.pixels, Unstoppable).ok()?;
    e.finish().ok()
}

fn encode_cpp(dist: f32, img: &ImageData) -> Option<Vec<u8>> {
    BenchEncoderConfig::new(EncoderImpl::CJpegli)
        .color(BenchColor::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(BenchSub::S444)
        .distance(dist)
        .encode(img)
        .ok()
}

struct RDPoint {
    config: String,
    param: String, // quality or distance value
    avg_bpp: f64,
    avg_ssim2: f64,
    avg_ba: f64,
    avg_bytes: f64,
    n: usize,
}

fn sweep_zen(
    label: &str,
    configs: Vec<(String, EncoderConfig)>,
    images: &[ImageData],
) -> Vec<RDPoint> {
    let mut results = Vec::new();
    for (param, cfg) in &configs {
        let mut bpps = Vec::new();
        let mut ssim2s = Vec::new();
        let mut bas = Vec::new();
        let mut bytess = Vec::new();
        for img in images {
            if let Some(jpeg) = encode_zen(cfg, img) {
                if let Some(m) = compute_metrics(img, &jpeg) {
                    bpps.push(m.bpp);
                    ssim2s.push(m.ssim2);
                    bas.push(m.ba);
                    bytess.push(m.bytes as f64);
                }
            }
        }
        let n = bpps.len();
        if n > 0 {
            results.push(RDPoint {
                config: label.to_string(),
                param: param.clone(),
                avg_bpp: bpps.iter().sum::<f64>() / n as f64,
                avg_ssim2: ssim2s.iter().sum::<f64>() / n as f64,
                avg_ba: bas.iter().sum::<f64>() / n as f64,
                avg_bytes: bytess.iter().sum::<f64>() / n as f64,
                n,
            });
        }
    }
    results
}

fn main() {
    let args = parse_args();

    // Load images
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.corpus)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read {}: {}", args.corpus.display(), e);
            std::process::exit(1);
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("png"))
        })
        .map(|e| e.path())
        .collect();
    paths.sort();
    paths.truncate(args.max_images);

    let images: Vec<ImageData> = paths
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    let corpus_name = args
        .corpus
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    println!(
        "=== R-D Comparison: zenjpeg approaches vs C++ jpegli 444 ===\n\
         Corpus: {} ({} images)\n\
         Metrics: SSIMULACRA2 (higher=better), Butteraugli (lower=better)\n",
        corpus_name,
        images.len()
    );

    let color_444 = ColorMode::YCbCr {
        subsampling: ChromaSubsampling::None,
    };

    let mut all_results: Vec<RDPoint> = Vec::new();

    // --- C++ jpegli 444 progressive (reference) ---
    print!("C++ jpegli 444 progressive...");
    std::io::stdout().flush().ok();
    {
        let mut results = Vec::new();
        for &dist in &CPP_DISTANCES {
            let mut bpps = Vec::new();
            let mut ssim2s = Vec::new();
            let mut bas = Vec::new();
            let mut bytess = Vec::new();
            for img in &images {
                if let Some(jpeg) = encode_cpp(dist, img) {
                    if let Some(m) = compute_metrics(img, &jpeg) {
                        bpps.push(m.bpp);
                        ssim2s.push(m.ssim2);
                        bas.push(m.ba);
                        bytess.push(m.bytes as f64);
                    }
                }
            }
            let n = bpps.len();
            if n > 0 {
                results.push(RDPoint {
                    config: "cjpegli-444".to_string(),
                    param: format!("d{:.1}", dist),
                    avg_bpp: bpps.iter().sum::<f64>() / n as f64,
                    avg_ssim2: ssim2s.iter().sum::<f64>() / n as f64,
                    avg_ba: bas.iter().sum::<f64>() / n as f64,
                    avg_bytes: bytess.iter().sum::<f64>() / n as f64,
                    n,
                });
            }
        }
        println!(" done ({} points)", results.len());
        all_results.extend(results);
    }

    // --- JpegliProgressive 444 (AQ, no trellis) ---
    print!("JpegliProg 444...");
    std::io::stdout().flush().ok();
    let jpegli_results = sweep_zen(
        "JpegliProg",
        QUALITY_LEVELS
            .iter()
            .map(|&q| {
                (
                    format!("q{:.0}", q),
                    EncoderConfig::ycbcr(q, ChromaSubsampling::None)
                        .optimization(OptimizationPreset::JpegliProgressive),
                )
            })
            .collect(),
        &images,
    );
    println!(" done ({} points)", jpegli_results.len());
    all_results.extend(jpegli_results);

    // --- MozjpegMaxCompression 444 (trellis, no AQ) at multiple lambdas ---
    for &lam in &LAMBDAS {
        let label = format!("MozMax-L{:.1}", lam);
        print!("{}...", label);
        std::io::stdout().flush().ok();
        let results = sweep_zen(
            &label,
            QUALITY_LEVELS
                .iter()
                .map(|&q| {
                    let mut expert =
                        ExpertConfig::from_preset(OptimizationPreset::MozjpegMaxCompression, q);
                    expert.trellis_lambda_log_scale1 = lam;
                    (format!("q{:.0}", q), expert.to_encoder_config(color_444))
                })
                .collect(),
            &images,
        );
        println!(" done ({} points)", results.len());
        all_results.extend(results);
    }

    // --- MozjpegProgressive 444 (trellis, no AQ) at best lambda ---
    print!("MozProg-L14.5...");
    std::io::stdout().flush().ok();
    let moz_prog_results = sweep_zen(
        "MozProg-L14.5",
        QUALITY_LEVELS
            .iter()
            .map(|&q| {
                let mut expert =
                    ExpertConfig::from_preset(OptimizationPreset::MozjpegProgressive, q);
                expert.trellis_lambda_log_scale1 = 14.5;
                (format!("q{:.0}", q), expert.to_encoder_config(color_444))
            })
            .collect(),
        &images,
    );
    println!(" done ({} points)", moz_prog_results.len());
    all_results.extend(moz_prog_results);

    // --- HybridMaxCompression 444 (AQ + trellis) at multiple lambdas ---
    for &lam in &LAMBDAS {
        let label = format!("HybMax-L{:.1}", lam);
        print!("{}...", label);
        std::io::stdout().flush().ok();
        let results = sweep_zen(
            &label,
            QUALITY_LEVELS
                .iter()
                .map(|&q| {
                    let mut expert =
                        ExpertConfig::from_preset(OptimizationPreset::HybridMaxCompression, q);
                    expert.trellis_lambda_log_scale1 = lam;
                    (format!("q{:.0}", q), expert.to_encoder_config(color_444))
                })
                .collect(),
            &images,
        );
        println!(" done ({} points)", results.len());
        all_results.extend(results);
    }

    // --- SA-Optimized Tables 444 (jpegli AQ + SA-tuned quant tables) ---
    #[cfg(feature = "optimized-tables")]
    {
        // With JpegliProgressive base (AQ enabled, progressive scan)
        print!("SA-Opt-JpegliProg...");
        std::io::stdout().flush().ok();
        let sa_jpegli_results = sweep_zen(
            "SA-Opt-JpegliProg",
            QUALITY_LEVELS
                .iter()
                .map(|&q| {
                    let tables = OptimizedTables::generate(q as u8);
                    (
                        format!("q{:.0}", q),
                        EncoderConfig::ycbcr(q, ChromaSubsampling::None)
                            .optimization(OptimizationPreset::JpegliProgressive)
                            .tables(tables),
                    )
                })
                .collect(),
            &images,
        );
        println!(" done ({} points)", sa_jpegli_results.len());
        all_results.extend(sa_jpegli_results);

        // With HybridMaxCompression base (AQ + trellis + SA tables) at best lambda
        print!("SA-Opt-HybMax-L14.5...");
        std::io::stdout().flush().ok();
        let sa_hyb_results = sweep_zen(
            "SA-Opt-HybMax-L14.5",
            QUALITY_LEVELS
                .iter()
                .map(|&q| {
                    let tables = OptimizedTables::generate(q as u8);
                    let mut expert =
                        ExpertConfig::from_preset(OptimizationPreset::HybridMaxCompression, q);
                    expert.trellis_lambda_log_scale1 = 14.5;
                    let config = expert.to_encoder_config(color_444).tables(tables);
                    (format!("q{:.0}", q), config)
                })
                .collect(),
            &images,
        );
        println!(" done ({} points)", sa_hyb_results.len());
        all_results.extend(sa_hyb_results);
    }

    // --- AutoOptimize 444 (auto_optimize(true)) ---
    print!("AutoOptimize 444...");
    std::io::stdout().flush().ok();
    let auto_results = sweep_zen(
        "AutoOptimize",
        QUALITY_LEVELS
            .iter()
            .map(|&q| {
                (
                    format!("q{:.0}", q),
                    EncoderConfig::ycbcr(q, ChromaSubsampling::None).auto_optimize(true),
                )
            })
            .collect(),
        &images,
    );
    println!(" done ({} points)", auto_results.len());
    all_results.extend(auto_results);

    // --- Print grouped R-D tables ---
    println!("\n{}", "=".repeat(90));

    // Group by config
    let mut configs: Vec<String> = Vec::new();
    for r in &all_results {
        if !configs.contains(&r.config) {
            configs.push(r.config.clone());
        }
    }

    for config in &configs {
        let pts: Vec<&RDPoint> = all_results.iter().filter(|r| &r.config == config).collect();
        println!("\n{}", config);
        println!(
            "  {:>6} {:>8} {:>8} {:>8} {:>8}",
            "Param", "BPP", "SSIM2", "BA", "Bytes"
        );
        println!("  {}", "-".repeat(44));
        for pt in &pts {
            println!(
                "  {:>6} {:>8.4} {:>8.2} {:>8.3} {:>8.0}",
                pt.param, pt.avg_bpp, pt.avg_ssim2, pt.avg_ba, pt.avg_bytes,
            );
        }
    }

    // --- Interpolated comparison at matched BPP ---
    println!("\n{}", "=".repeat(90));
    println!("\nInterpolated comparison at matched BPP:");
    println!("(shows quality metrics where each config achieves a given BPP)\n");

    let target_bpps = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];

    println!(" {:>20} | {:>5} |", "Config", "Metric");
    print!(" {:>20} | {:>5} |", "", "");
    for &bpp in &target_bpps {
        print!(" {:>6.1} |", bpp);
    }
    println!();
    println!(" {}", "-".repeat(20 + 3 + 5 + 3 + target_bpps.len() * 9));

    for config in &configs {
        let pts: Vec<&RDPoint> = all_results.iter().filter(|r| &r.config == config).collect();
        if pts.len() < 2 {
            continue;
        }

        // Sort by bpp for interpolation
        let mut sorted: Vec<(f64, f64, f64)> = pts
            .iter()
            .map(|p| (p.avg_bpp, p.avg_ssim2, p.avg_ba))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // SSIM2 row
        print!(" {:>20} | {:>5} |", config, "SSIM2");
        for &target in &target_bpps {
            let val = interpolate(&sorted, target, |p| p.1);
            if let Some(v) = val {
                print!(" {:>6.1} |", v);
            } else {
                print!("      - |");
            }
        }
        println!();

        // Butteraugli row
        print!(" {:>20} | {:>5} |", "", "BA");
        for &target in &target_bpps {
            let val = interpolate(&sorted, target, |p| p.2);
            if let Some(v) = val {
                print!(" {:>6.2} |", v);
            } else {
                print!("      - |");
            }
        }
        println!();
    }
    println!("\nSSIM2: higher=better, BA: lower=better");

    // --- CSV output ---
    if let Some(ref path) = args.output {
        write_csv(path, &all_results);
    }
}

/// Linear interpolation on sorted (bpp, ssim2, ba) points.
fn interpolate(
    sorted: &[(f64, f64, f64)],
    target_bpp: f64,
    accessor: fn(&(f64, f64, f64)) -> f64,
) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    // Outside range
    if target_bpp < sorted.first()?.0 || target_bpp > sorted.last()?.0 {
        return None;
    }
    // Find bracketing pair
    for w in sorted.windows(2) {
        let (b1, b2) = (w[0].0, w[1].0);
        if b1 <= target_bpp && target_bpp <= b2 {
            if (b2 - b1).abs() < 1e-9 {
                return Some(accessor(&w[0]));
            }
            let t = (target_bpp - b1) / (b2 - b1);
            return Some(accessor(&w[0]) + t * (accessor(&w[1]) - accessor(&w[0])));
        }
    }
    None
}

fn write_csv(path: &std::path::Path, results: &[RDPoint]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot create {}: {}", path.display(), e);
            return;
        }
    };
    writeln!(f, "config,param,avg_bpp,avg_ssim2,avg_ba,avg_bytes,n").ok();
    for r in results {
        writeln!(
            f,
            "{},{},{:.6},{:.4},{:.4},{:.0},{}",
            r.config, r.param, r.avg_bpp, r.avg_ssim2, r.avg_ba, r.avg_bytes, r.n
        )
        .ok();
    }
    println!("CSV written to {}", path.display());
}
