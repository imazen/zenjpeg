//! SSIM2 Pareto Sweep: find zenjpeg configs that beat C++ jpegli at equivalent quality.
//!
//! Uses codec-eval's RDKnee angular buckets: first computes the C++ jpegli R-D knee
//! from corpus-averaged SSIMULACRA2 data, then bins all encode results by angle from
//! that knee. For each angular bin, finds the zenjpeg configuration producing the
//! smallest files compared to C++ jpegli 4:4:4 progressive (max effort).
//!
//! Usage:
//! ```bash
//! # Quick smoke test (3 images)
//! cargo run --release -p zenjpeg --example ssim2_pareto_sweep -- --images 3 --quick
//!
//! # Full run (20 images, ~13k encodes)
//! cargo run --release -p zenjpeg --example ssim2_pareto_sweep
//!
//! # With CSV output
//! cargo run --release -p zenjpeg --example ssim2_pareto_sweep -- \
//!     --output /mnt/v/output/zenjpeg/ssim2_pareto.csv \
//!     --output-detail /mnt/v/output/zenjpeg/ssim2_pareto_detail.csv
//! ```

use codec_eval::stats::rd_knee::{BinScheme, CorpusAggregate, FixedFrame, RDKnee};
use enough::Unstoppable;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use zenjpeg::encode::search::ExpertConfig;
use zenjpeg::encode::{ChromaSubsampling, ColorMode, OptimizationPreset, PixelLayout};
use zenjpeg_bench_utils::{
    ChromaSubsampling as BenchSub, ColorMode as BenchColor, EncoderConfig as BenchEncoderConfig,
    EncoderImpl, ImageData, QualityMetrics, ScanMode, bytes_to_rgb, decode_jpeg_to_rgb,
};

// --- Constants ---

fn corpus_cid22_dir() -> PathBuf {
    let cc = codec_corpus::Corpus::new().expect("codec-corpus unavailable");
    cc.get("CID22/CID22-512/validation")
        .expect("CID22 corpus not available")
}

/// C++ distances spanning the full R-D curve (low quality → near-lossless).
const CPP_DISTANCES: [f32; 13] = [
    0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0,
];

const QUALITY_LEVELS: [f32; 13] = [
    98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 85.0, 82.0, 78.0, 72.0, 65.0, 55.0, 40.0,
];

const QUALITY_LEVELS_QUICK: [f32; 7] = [98.0, 94.0, 90.0, 85.0, 78.0, 65.0, 40.0];

const LAMBDA_VALUES: [f32; 8] = [12.0, 13.0, 14.0, 14.5, 15.0, 15.5, 16.0, 17.0];

// Presets that don't use trellis (no lambda sweep)
const NO_TRELLIS_PRESETS: [(OptimizationPreset, &str); 2] = [
    (OptimizationPreset::JpegliBaseline, "JpegliBase"),
    (OptimizationPreset::JpegliProgressive, "JpegliProg"),
];

// Presets that use trellis (sweep lambda)
const TRELLIS_PRESETS: [(OptimizationPreset, &str); 6] = [
    (OptimizationPreset::MozjpegBaseline, "MozBase"),
    (OptimizationPreset::MozjpegProgressive, "MozProg"),
    (OptimizationPreset::MozjpegMaxCompression, "MozMax"),
    (OptimizationPreset::HybridBaseline, "HybBase"),
    (OptimizationPreset::HybridProgressive, "HybProg"),
    (OptimizationPreset::HybridMaxCompression, "HybMax"),
];

// --- Data types ---

/// A single encode data point with angular position from R-D knee.
#[derive(Clone, Debug)]
struct DataPoint {
    config_name: String,
    image_idx: usize,
    ssim2: f64,
    bytes: usize,
    bpp: f64,
    /// Angle from the C++ R-D knee (degrees). Populated after knee computation.
    angle: f64,
}

/// Parsed CLI arguments.
struct Args {
    output: Option<PathBuf>,
    output_detail: Option<PathBuf>,
    images: usize,
    verbose: bool,
    quick: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        output: None,
        output_detail: None,
        images: 20,
        verbose: false,
        quick: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--output" => args.output = iter.next().map(PathBuf::from),
            "--output-detail" => args.output_detail = iter.next().map(PathBuf::from),
            "--images" => {
                args.images = iter.next().and_then(|s| s.parse().ok()).unwrap_or(20);
            }
            "--verbose" => args.verbose = true,
            "--quick" => args.quick = true,
            "--help" | "-h" => {
                eprintln!("Usage: ssim2_pareto_sweep [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --output <file.csv>         Bucket summary CSV");
                eprintln!("  --output-detail <file.csv>  Per-point detail CSV");
                eprintln!("  --images <N>                Number of CID22 images (default: 20)");
                eprintln!("  --verbose                   Print per-image progress");
                eprintln!("  --quick                     7 quality levels instead of 13");
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

// --- Encode helpers ---

fn encode_expert(expert: &ExpertConfig, color_mode: ColorMode, img: &ImageData) -> Option<Vec<u8>> {
    let config = expert.to_encoder_config(color_mode);
    let mut enc = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    enc.push_packed(&img.pixels, Unstoppable).ok()?;
    enc.finish().ok()
}

fn encode_cpp(dist: f32, img: &ImageData) -> Option<Vec<u8>> {
    let config = BenchEncoderConfig::new(EncoderImpl::CJpegli)
        .color(BenchColor::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(BenchSub::S444)
        .distance(dist);
    config.encode(img).ok()
}

fn compute_ssim2(img: &ImageData, jpeg: &[u8]) -> Option<f64> {
    let orig = bytes_to_rgb(&img.pixels, img.width, img.height);
    let dec = decode_jpeg_to_rgb(jpeg).ok()?;
    Some(QualityMetrics::ssimulacra2(orig.as_ref(), dec.as_ref()))
}

fn pixels_for(img: &ImageData) -> usize {
    img.width * img.height
}

// --- Knee computation ---

/// Build the C++ jpegli SSIMULACRA2 R-D knee from corpus-averaged data.
///
/// Encodes all images at each distance, averages bpp and SSIM2 across images,
/// then finds the 45° tangent point on the normalized R-D curve.
fn compute_cpp_knee(cpp_points: &[DataPoint], images: &[ImageData]) -> Option<RDKnee> {
    // Group by distance index: points are pushed per-image in distance order.
    // Compute per-distance averages across images for the R-D curve.
    let n_images = images.len();
    let n_distances = CPP_DISTANCES.len();

    // Points are ordered: image0_dist0, image0_dist1, ..., image1_dist0, ...
    // But they might have gaps if encodes failed. Group by distance index.
    let mut dist_bpp: Vec<Vec<f64>> = vec![Vec::new(); n_distances];
    let mut dist_ssim2: Vec<Vec<f64>> = vec![Vec::new(); n_distances];

    for pt in cpp_points {
        // Find which distance bucket this point belongs to by matching ssim2/bpp
        // Since points are pushed in order, we can use position
        let points_per_image: Vec<&DataPoint> = cpp_points
            .iter()
            .filter(|p| p.image_idx == pt.image_idx)
            .collect();
        if let Some(di) = points_per_image.iter().position(|p| std::ptr::eq(*p, pt)) {
            if di < n_distances {
                dist_bpp[di].push(pt.bpp);
                dist_ssim2[di].push(pt.ssim2);
            }
        }
    }

    // Build averaged curve: (bpp, ssim2, 0.0) — no butteraugli
    let mut curve: Vec<(f64, f64, f64)> = Vec::new();
    for di in 0..n_distances {
        if dist_bpp[di].is_empty() {
            continue;
        }
        let avg_bpp = dist_bpp[di].iter().sum::<f64>() / dist_bpp[di].len() as f64;
        let avg_s2 = dist_ssim2[di].iter().sum::<f64>() / dist_ssim2[di].len() as f64;
        curve.push((avg_bpp, avg_s2, 0.0));
    }

    // Sort by bpp ascending (required by knee detection)
    curve.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    if curve.len() < 3 {
        eprintln!(
            "Warning: only {} R-D curve points, need >= 3 for knee detection",
            curve.len()
        );
        return None;
    }

    let agg = CorpusAggregate {
        corpus: "CID22-512".to_string(),
        codec: "cjpegli-444-progressive".to_string(),
        curve,
        image_count: n_images,
    };

    agg.ssimulacra2_knee(&FixedFrame::WEB)
}

// --- Main ---

fn main() {
    let args = parse_args();

    // Load CID22 images
    let cid22_dir = corpus_cid22_dir();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&cid22_dir)
        .unwrap_or_else(|e| {
            eprintln!("Cannot read {}: {}", cid22_dir.display(), e);
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
    paths.truncate(args.images);

    let images: Vec<ImageData> = paths
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    if images.is_empty() {
        eprintln!("No images loaded from {}", cid22_dir.display());
        std::process::exit(1);
    }

    println!(
        "=== SSIM2 Pareto Sweep: zenjpeg vs C++ jpegli 444 ===\n\
         Images: {} from CID22-512\n\
         Baseline: cjpegli progressive 4:4:4 (distance mode)",
        images.len()
    );

    let quality_levels: &[f32] = if args.quick {
        &QUALITY_LEVELS_QUICK
    } else {
        &QUALITY_LEVELS
    };

    let color_mode = ColorMode::YCbCr {
        subsampling: ChromaSubsampling::None,
    };

    // Phase 1a: C++ baseline data (444 progressive, distance mode)
    let mut cpp_points: Vec<DataPoint> = Vec::new();
    print!(
        "\nC++ jpegli 444: encoding {} images × {} distances...",
        images.len(),
        CPP_DISTANCES.len()
    );
    std::io::stdout().flush().ok();

    for (img_idx, img) in images.iter().enumerate() {
        let px = pixels_for(img);
        for &dist in &CPP_DISTANCES {
            if let Some(jpeg) = encode_cpp(dist, img) {
                if let Some(ssim2) = compute_ssim2(img, &jpeg) {
                    let bpp = jpeg.len() as f64 * 8.0 / px as f64;
                    cpp_points.push(DataPoint {
                        config_name: "cjpegli".to_string(),
                        image_idx: img_idx,
                        ssim2,
                        bytes: jpeg.len(),
                        bpp,
                        angle: 0.0, // filled in after knee computation
                    });
                }
            }
        }
        if args.verbose {
            print!(" {}", img_idx + 1);
            std::io::stdout().flush().ok();
        }
    }
    println!(" done ({} points)", cpp_points.len());

    // Compute R-D knee from C++ data
    let knee = match compute_cpp_knee(&cpp_points, &images) {
        Some(k) => {
            println!(
                "\nR-D knee: bpp = {:.4}, SSIM2 = {:.2} (norm: bpp={:.3}, q={:.3})",
                k.bpp,
                k.quality,
                k.norm.normalize_bpp(k.bpp),
                k.norm.normalize_quality(k.quality),
            );
            println!(
                "  bpp range: [{:.4}, {:.4}], SSIM2 range: [{:.2}, {:.2}]",
                k.norm.bpp_range.min,
                k.norm.bpp_range.max,
                k.norm.quality_range.min,
                k.norm.quality_range.max,
            );
            k
        }
        None => {
            eprintln!("ERROR: Could not compute R-D knee from C++ data. Need more images.");
            std::process::exit(1);
        }
    };

    // Compute angles for C++ points
    for pt in &mut cpp_points {
        pt.angle = FixedFrame::WEB.s2_angle(pt.bpp, pt.ssim2);
    }

    // Phase 1b: zenjpeg sweep (444 to match C++ baseline)
    let mut zen_points: Vec<DataPoint> = Vec::new();
    let mut config_list: Vec<(String, ExpertConfig)> = Vec::new();

    // No-trellis presets (no lambda sweep)
    for (preset, name) in &NO_TRELLIS_PRESETS {
        for &q in quality_levels {
            let expert = ExpertConfig::from_preset(*preset, q);
            config_list.push((name.to_string(), expert));
        }
    }

    // Trellis presets (sweep lambda)
    for (preset, name) in &TRELLIS_PRESETS {
        for &lam in &LAMBDA_VALUES {
            for &q in quality_levels {
                let mut expert = ExpertConfig::from_preset(*preset, q);
                expert.trellis_lambda_log_scale1 = lam;
                config_list.push((format!("{}-L{:.2}", name, lam), expert));
            }
        }
    }

    let total_encodes = config_list.len() * images.len();
    let n_configs = config_list.len() / quality_levels.len();
    println!(
        "\nzenjpeg 444: {} configs × {} qualities × {} images = {} encodes",
        n_configs,
        quality_levels.len(),
        images.len(),
        total_encodes
    );
    print!("Encoding...");
    std::io::stdout().flush().ok();

    let mut encode_count = 0usize;
    let report_interval = (total_encodes / 20).max(1);

    for (img_idx, img) in images.iter().enumerate() {
        let px = pixels_for(img);
        for (cfg_name, expert) in &config_list {
            if let Some(jpeg) = encode_expert(expert, color_mode, img) {
                if let Some(ssim2) = compute_ssim2(img, &jpeg) {
                    let bpp = jpeg.len() as f64 * 8.0 / px as f64;
                    let angle = FixedFrame::WEB.s2_angle(bpp, ssim2);
                    zen_points.push(DataPoint {
                        config_name: cfg_name.clone(),
                        image_idx: img_idx,
                        ssim2,
                        bytes: jpeg.len(),
                        bpp,
                        angle,
                    });
                }
            }
            encode_count += 1;
            if encode_count % report_interval == 0 {
                let pct = encode_count * 100 / total_encodes;
                print!(" {}%", pct);
                std::io::stdout().flush().ok();
            }
        }
    }
    println!(" done ({} points)", zen_points.len());

    // Phase 2: Angular bucket analysis
    let scheme = BinScheme::default_18(); // 18 bins × 10° covering [-90°, +90°]
    println!();
    analyze_angular_buckets(&cpp_points, &zen_points, &scheme, &knee, &images);

    // Write CSVs
    if let Some(ref path) = args.output {
        write_summary_csv(path, &cpp_points, &zen_points, &scheme, &knee);
    }
    if let Some(ref path) = args.output_detail {
        write_detail_csv(path, &cpp_points, &zen_points, &images, &scheme);
    }
}

fn analyze_angular_buckets(
    cpp_points: &[DataPoint],
    zen_points: &[DataPoint],
    scheme: &BinScheme,
    knee: &RDKnee,
    images: &[ImageData],
) {
    let n_bins = scheme.count;
    let n_images = images.len();

    // Group C++ points by bin → bpp values
    let mut cpp_bin_bpp: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    for pt in cpp_points {
        let bin = scheme.bin_for(pt.angle);
        cpp_bin_bpp[bin.index].push(pt.bpp);
    }

    // Group zenjpeg points by (config_name, bin) → bpp values
    let mut zen_by_config: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    // Track per-image best bpp for wins calculation
    let mut zen_image_bpp: HashMap<String, Vec<HashMap<usize, f64>>> = HashMap::new();

    for pt in zen_points {
        let bin = scheme.bin_for(pt.angle);
        let bi = bin.index;

        let entry = zen_by_config
            .entry(pt.config_name.clone())
            .or_insert_with(|| vec![Vec::new(); n_bins]);
        entry[bi].push(pt.bpp);

        let img_entry = zen_image_bpp
            .entry(pt.config_name.clone())
            .or_insert_with(|| vec![HashMap::new(); n_bins]);
        let current = img_entry[bi].entry(pt.image_idx).or_insert(f64::MAX);
        if pt.bpp < *current {
            *current = pt.bpp;
        }
    }

    // C++ per-image best bpp per bin
    let mut cpp_image_bpp: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_bins];
    for pt in cpp_points {
        let bin = scheme.bin_for(pt.angle);
        let current = cpp_image_bpp[bin.index]
            .entry(pt.image_idx)
            .or_insert(f64::MAX);
        if pt.bpp < *current {
            *current = pt.bpp;
        }
    }

    // Print table
    println!(
        " {:>14} | {:>5} | {:>9} | {:>28} | {:>9} | {:>8} | {:>6}",
        "Angle Range", "N(C++)", "C++ bpp", "Best zenjpeg Config", "Zen bpp", "Savings", "Wins"
    );
    println!(
        " {:-<14}-+-{:-<5}-+-{:-<9}-+-{:-<28}-+-{:-<9}-+-{:-<8}-+-{:-<6}",
        "", "", "", "", "", "", ""
    );

    for bin in scheme.bins() {
        let bi = bin.index;
        let cpp_bpps = &cpp_bin_bpp[bi];
        if cpp_bpps.is_empty() {
            continue;
        }
        let cpp_avg = cpp_bpps.iter().sum::<f64>() / cpp_bpps.len() as f64;

        // Find best zenjpeg config for this bin
        let mut best_name = String::new();
        let mut best_avg = f64::MAX;

        for (cfg_name, bin_data) in &zen_by_config {
            let bpps = &bin_data[bi];
            if bpps.is_empty() {
                continue;
            }
            let avg = bpps.iter().sum::<f64>() / bpps.len() as f64;
            if avg < best_avg {
                best_avg = avg;
                best_name = cfg_name.clone();
            }
        }

        let bin_label = format!("[{:+.0}°,{:+.0}°)", bin.lo(), bin.hi());

        if best_name.is_empty() {
            println!(
                " {:>14} | {:>5} | {:>9.4} | {:>28} | {:>9} | {:>8} | {:>6}",
                bin_label,
                cpp_bpps.len(),
                cpp_avg,
                "(no data)",
                "-",
                "-",
                "-"
            );
            continue;
        }

        let savings_pct = (cpp_avg - best_avg) / cpp_avg * 100.0;

        // Count per-image wins (zen config beats C++ for same image in same bin)
        let wins = if let Some(zen_imgs) = zen_image_bpp.get(&best_name) {
            let zen_bi = &zen_imgs[bi];
            let cpp_bi = &cpp_image_bpp[bi];
            cpp_bi
                .iter()
                .filter(|(&img_idx, &cpp_bpp)| {
                    zen_bi
                        .get(&img_idx)
                        .is_some_and(|&zen_bpp| zen_bpp < cpp_bpp)
                })
                .count()
        } else {
            0
        };
        let total_in_bin = cpp_image_bpp[bi].len();

        let savings_str = format!("{:+.1}%", -savings_pct);
        let wins_str = format!("{}/{}", wins, total_in_bin);

        // Mark the knee bin
        let knee_marker = if bin.contains(0.0) { " ←knee" } else { "" };

        println!(
            " {:>14} | {:>5} | {:>9.4} | {:>28} | {:>9.4} | {:>8} | {:>6}{}",
            bin_label,
            cpp_bpps.len(),
            cpp_avg,
            &best_name,
            best_avg,
            savings_str,
            wins_str,
            knee_marker,
        );
    }

    // Summary
    println!(
        "\nKnee: bpp={:.4}, SSIM2={:.2}  |  {} images  |  Bins: {} × {:.0}°",
        knee.bpp, knee.quality, n_images, scheme.count, scheme.width
    );
    println!(
        "Negative angle = efficient compression (below knee), positive = diminishing returns (above knee)"
    );
}

fn write_summary_csv(
    path: &std::path::Path,
    cpp_points: &[DataPoint],
    zen_points: &[DataPoint],
    scheme: &BinScheme,
    _knee: &RDKnee,
) {
    let n_bins = scheme.count;

    let mut cpp_bin_bpp: Vec<Vec<f64>> = vec![Vec::new(); n_bins];
    for pt in cpp_points {
        let bin = scheme.bin_for(pt.angle);
        cpp_bin_bpp[bin.index].push(pt.bpp);
    }

    let mut zen_by_config: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for pt in zen_points {
        let bin = scheme.bin_for(pt.angle);
        zen_by_config
            .entry(pt.config_name.clone())
            .or_insert_with(|| vec![Vec::new(); n_bins])[bin.index]
            .push(pt.bpp);
    }

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

    writeln!(
        f,
        "bin_lo,bin_hi,bin_center,cpp_bpp,best_zen_config,zen_bpp,savings_pct,cpp_points,zen_points"
    )
    .ok();

    for bin in scheme.bins() {
        let bi = bin.index;
        let cpp_bpps = &cpp_bin_bpp[bi];
        if cpp_bpps.is_empty() {
            continue;
        }
        let cpp_avg = cpp_bpps.iter().sum::<f64>() / cpp_bpps.len() as f64;

        let mut best_name = String::new();
        let mut best_avg = f64::MAX;
        let mut best_count = 0usize;

        for (cfg_name, bin_data) in &zen_by_config {
            let bpps = &bin_data[bi];
            if bpps.is_empty() {
                continue;
            }
            let avg = bpps.iter().sum::<f64>() / bpps.len() as f64;
            if avg < best_avg {
                best_avg = avg;
                best_name = cfg_name.clone();
                best_count = bpps.len();
            }
        }

        if best_name.is_empty() {
            continue;
        }

        let savings = (cpp_avg - best_avg) / cpp_avg * 100.0;
        writeln!(
            f,
            "{:.1},{:.1},{:.1},{:.6},{},{:.6},{:.2},{},{}",
            bin.lo(),
            bin.hi(),
            bin.center,
            cpp_avg,
            best_name,
            best_avg,
            savings,
            cpp_bpps.len(),
            best_count,
        )
        .ok();
    }

    println!("Summary CSV written to {}", path.display());
}

fn write_detail_csv(
    path: &std::path::Path,
    cpp_points: &[DataPoint],
    zen_points: &[DataPoint],
    images: &[ImageData],
    scheme: &BinScheme,
) {
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

    writeln!(
        f,
        "config,image_idx,image_name,ssim2,bytes,bpp,angle,bin_center"
    )
    .ok();

    for pt in cpp_points.iter().chain(zen_points.iter()) {
        let bin = scheme.bin_for(pt.angle);
        writeln!(
            f,
            "{},{},{},{:.4},{},{:.6},{:.2},{:.1}",
            pt.config_name,
            pt.image_idx,
            images[pt.image_idx].name,
            pt.ssim2,
            pt.bytes,
            pt.bpp,
            pt.angle,
            bin.center,
        )
        .ok();
    }

    println!("Detail CSV written to {}", path.display());
}
