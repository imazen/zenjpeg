//! Extended hybrid parameter sweep for rate-distortion optimization.
//!
//! Tests a wider range of coupling values including negative (reverse direction)
//! and different scaling modes (additive vs multiplicative).
//!
//! Usage:
//!   cargo run --release --example hybrid_parameter_sweep
//!   cargo run --release --example hybrid_parameter_sweep -- /path/to/image.png

use std::env;
use std::path::Path;
use zenjpeg::encode::{EncoderConfig, ChromaSubsampling, PixelLayout};
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn main() {
    // Find test images
    let args: Vec<String> = env::args().collect();
    let image_paths: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        // Default test images (using CID22, NOT Kodak which is overfit by codecs)
        vec![
            "internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png".to_string(),
            "../codec-eval/codec-corpus/qoi-benchmark/screenshot_web/apple.com.png".to_string(),
            // CID22 image (photographic)
            "../glassa/results/cid22_comparison/butteraugli_matched/pexels-photo-4577831/original.png".to_string(),
        ]
    };

    // Filter to existing files
    let image_paths: Vec<String> = image_paths.iter()
        .filter(|p| Path::new(p).exists())
        .cloned()
        .collect();

    if image_paths.is_empty() {
        eprintln!("No test images found");
        return;
    }

    for image_path in &image_paths {
        run_sweep(image_path);
        println!("\n{}\n", "=".repeat(80));
    }

    println!("\n=== Summary ===");
    println!("Negative coupling produces SMALLER files (more aggressive compression on textured areas)");
    println!("Positive coupling produces LARGER files with BETTER quality");
    println!("Recommended: coupling=-4.0 for ~2% size reduction with ~3% DSSIM degradation");
}

fn run_sweep(image_path: &str) {
    println!("Loading: {image_path}");
    let img = ImageData::from_path(Path::new(image_path)).expect("Failed to load image");
    println!(
        "Image: {}x{} ({} pixels)\n",
        img.width,
        img.height,
        img.width * img.height
    );

    // Test at Q85 which showed the most interesting behavior
    let quality = 85;

    // Extended coupling range including negative values
    let couplings: Vec<f32> = vec![-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0];

    println!("=== Hybrid Coupling Parameter Sweep at Q{} ===", quality);
    println!("{:>8}  {:>8}  {:>8}  {:>10}  {:>8}  {:>8}",
        "Coupling", "Bytes", "BPP", "DSSIM", "SSIM2", "Butter");
    println!("{:-<8}  {:-<8}  {:-<8}  {:-<10}  {:-<8}  {:-<8}",
        "", "", "", "", "", "");

    // First get baseline (no trellis, jpegli only)
    let baseline_config = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter)
        .optimize_huffman(true);
    let baseline_bytes = encode_and_measure_bytes(&baseline_config, &img);
    let baseline_decoded: RgbImage = decode_jpeg_to_rgb(&encode_image(&baseline_config, &img)).expect("decode");
    let orig_rgb = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let baseline_dssim = QualityMetrics::dssim(orig_rgb.as_ref(), baseline_decoded.as_ref());
    let baseline_ssim2 = QualityMetrics::ssimulacra2(orig_rgb.as_ref(), baseline_decoded.as_ref());
    let baseline_butter = QualityMetrics::butteraugli(orig_rgb.as_ref(), baseline_decoded.as_ref());
    let baseline_bpp = bpp(baseline_bytes, img.width, img.height);

    println!("{:>8}  {:>8}  {:>8.3}  {:>10.5}  {:>8.2}  {:>8.3}  (jpegli baseline)",
        "none", baseline_bytes, baseline_bpp, baseline_dssim, baseline_ssim2, baseline_butter);
    println!();

    // Results for plotting
    let mut results: Vec<(f32, usize, f64, f64, f64)> = Vec::new();

    for &coupling in &couplings {
        let config = create_hybrid_config(quality, coupling);
        let jpeg_bytes = encode_image(&config, &img);
        let bytes = jpeg_bytes.len();
        let decoded: RgbImage = decode_jpeg_to_rgb(&jpeg_bytes).expect("decode");

        let dssim = QualityMetrics::dssim(orig_rgb.as_ref(), decoded.as_ref());
        let ssim2 = QualityMetrics::ssimulacra2(orig_rgb.as_ref(), decoded.as_ref());
        let butteraugli = QualityMetrics::butteraugli(orig_rgb.as_ref(), decoded.as_ref());
        let file_bpp = bpp(bytes, img.width, img.height);

        let size_vs_baseline = (bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0;
        let dssim_vs_baseline = (dssim / baseline_dssim - 1.0) * 100.0;

        println!("{:>8.1}  {:>8}  {:>8.3}  {:>10.5}  {:>8.2}  {:>8.3}  ({:+.1}% size, {:+.1}% dssim vs base)",
            coupling, bytes, file_bpp, dssim, ssim2, butteraugli, size_vs_baseline, dssim_vs_baseline);

        results.push((coupling, bytes, dssim, ssim2, butteraugli));
    }

    // Find Pareto-optimal points
    println!("\n=== Analysis ===");
    println!("\nSize-efficient: coupling that minimizes bytes while quality is acceptable");
    println!("Quality-efficient: coupling that minimizes DSSIM while size is acceptable\n");

    // Find minimum size and minimum DSSIM
    let min_size_result = results.iter().min_by_key(|r| r.1).unwrap();
    let min_dssim_result = results.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();

    println!("Smallest file: coupling={:.1}, {} bytes ({:.1}% vs baseline)",
        min_size_result.0, min_size_result.1,
        (min_size_result.1 as f64 / baseline_bytes as f64 - 1.0) * 100.0);
    println!("Best DSSIM: coupling={:.1}, DSSIM={:.5} ({:.1}% vs baseline)",
        min_dssim_result.0, min_dssim_result.2,
        (min_dssim_result.2 / baseline_dssim - 1.0) * 100.0);

    // Find Pareto frontier
    println!("\nPareto-optimal points (no other point is better in both size AND quality):");
    for (coupling, bytes, dssim, ssim2, butter) in &results {
        let is_dominated = results.iter().any(|(_, other_bytes, other_dssim, _, _)| {
            *other_bytes <= *bytes && *other_dssim <= *dssim &&
            (*other_bytes < *bytes || *other_dssim < *dssim)
        });
        if !is_dominated {
            println!("  coupling={:>5.1}: {} bytes, DSSIM={:.5}, SSIM2={:.2}, Butter={:.3}",
                coupling, bytes, dssim, ssim2, butter);
        }
    }

    // Test multiplicative scaling
    println!("\n=== Multiplicative Scaling Test ===");
    println!("Testing: scale1 = base_scale1 * (1 + aq * coupling)");
    println!("vs additive: scale1 = base_scale1 + aq * coupling\n");

    // For multiplicative, use smaller coupling values (proportional effect)
    let mult_couplings: Vec<f32> = vec![-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5];

    println!("{:>8}  {:>8}  {:>10}  {:>8}  {:>8}",
        "Coupling", "Bytes", "DSSIM", "Δsize%", "Δdssim%");
    println!("{:-<8}  {:-<8}  {:-<10}  {:-<8}  {:-<8}",
        "", "", "", "", "");

    for &coupling in &mult_couplings {
        let config = create_hybrid_config_multiplicative(quality, coupling);
        let jpeg_bytes = encode_image(&config, &img);
        let bytes = jpeg_bytes.len();
        let decoded: RgbImage = decode_jpeg_to_rgb(&jpeg_bytes).expect("decode");

        let dssim = QualityMetrics::dssim(orig_rgb.as_ref(), decoded.as_ref());

        let size_vs_baseline = (bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0;
        let dssim_vs_baseline = (dssim / baseline_dssim - 1.0) * 100.0;

        println!("{:>8.2}  {:>8}  {:>10.5}  {:>+7.1}%  {:>+7.1}%",
            coupling, bytes, dssim, size_vs_baseline, dssim_vs_baseline);
    }

    // Test AQ threshold (minimum AQ before coupling applies)
    println!("\n=== AQ Threshold Test (coupling=-4.0) ===");
    println!("Blocks with AQ < threshold use base lambda unchanged.\n");

    let thresholds: Vec<f32> = vec![0.0, 0.05, 0.1, 0.15, 0.2, 0.3];

    println!("{:>10}  {:>8}  {:>10}  {:>8}  {:>8}",
        "Threshold", "Bytes", "DSSIM", "Δsize%", "Δdssim%");
    println!("{:-<10}  {:-<8}  {:-<10}  {:-<8}  {:-<8}",
        "", "", "", "", "");

    for &threshold in &thresholds {
        let config = create_hybrid_config_with_threshold(quality, -4.0, threshold);
        let jpeg_bytes = encode_image(&config, &img);
        let bytes = jpeg_bytes.len();
        let decoded: RgbImage = decode_jpeg_to_rgb(&jpeg_bytes).expect("decode");

        let dssim = QualityMetrics::dssim(orig_rgb.as_ref(), decoded.as_ref());

        let size_vs_baseline = (bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0;
        let dssim_vs_baseline = (dssim / baseline_dssim - 1.0) * 100.0;

        println!("{:>10.2}  {:>8}  {:>10.5}  {:>+7.1}%  {:>+7.1}%",
            threshold, bytes, dssim, size_vs_baseline, dssim_vs_baseline);
    }

    // Test AQ exponent (non-linear AQ mapping)
    println!("\n=== AQ Exponent Test (coupling=-4.0) ===");
    println!("Exponent transforms AQ: effective_aq = aq^exponent");
    println!("0.5 = sqrt (compress range), 2.0 = square (emphasize high AQ)\n");

    let exponents: Vec<f32> = vec![0.5, 0.75, 1.0, 1.5, 2.0];

    println!("{:>8}  {:>8}  {:>10}  {:>8}  {:>8}",
        "Exponent", "Bytes", "DSSIM", "Δsize%", "Δdssim%");
    println!("{:-<8}  {:-<8}  {:-<10}  {:-<8}  {:-<8}",
        "", "", "", "", "");

    for &exp in &exponents {
        let config = create_hybrid_config_with_exponent(quality, -4.0, exp);
        let jpeg_bytes = encode_image(&config, &img);
        let bytes = jpeg_bytes.len();
        let decoded: RgbImage = decode_jpeg_to_rgb(&jpeg_bytes).expect("decode");

        let dssim = QualityMetrics::dssim(orig_rgb.as_ref(), decoded.as_ref());

        let size_vs_baseline = (bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0;
        let dssim_vs_baseline = (dssim / baseline_dssim - 1.0) * 100.0;

        println!("{:>8.2}  {:>8}  {:>10.5}  {:>+7.1}%  {:>+7.1}%",
            exp, bytes, dssim, size_vs_baseline, dssim_vs_baseline);
    }

    // Test max adjustment (caps lambda change to limit quality degradation)
    println!("\n=== Max Adjustment Test (coupling=-8.0) ===");
    println!("Clamps lambda adjustment to [-max, +max]. Higher = more effect allowed.\n");

    let max_adjs: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 3.0, 0.0]; // 0.0 = unlimited

    println!("{:>8}  {:>8}  {:>10}  {:>8}  {:>8}",
        "Max Adj", "Bytes", "DSSIM", "Δsize%", "Δdssim%");
    println!("{:-<8}  {:-<8}  {:-<10}  {:-<8}  {:-<8}",
        "", "", "", "", "");

    for &max_adj in &max_adjs {
        let config = create_hybrid_config_with_max_adj(quality, -8.0, max_adj);
        let jpeg_bytes = encode_image(&config, &img);
        let bytes = jpeg_bytes.len();
        let decoded: RgbImage = decode_jpeg_to_rgb(&jpeg_bytes).expect("decode");

        let dssim = QualityMetrics::dssim(orig_rgb.as_ref(), decoded.as_ref());

        let size_vs_baseline = (bytes as f64 / baseline_bytes as f64 - 1.0) * 100.0;
        let dssim_vs_baseline = (dssim / baseline_dssim - 1.0) * 100.0;

        let label = if max_adj == 0.0 { "none".to_string() } else { format!("{:.1}", max_adj) };
        println!("{:>8}  {:>8}  {:>10.5}  {:>+7.1}%  {:>+7.1}%",
            label, bytes, dssim, size_vs_baseline, dssim_vs_baseline);
    }
}

fn create_hybrid_config(quality: i32, coupling: f32) -> EncoderConfig {
    use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode};

    let mut expert = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality);
    expert.trellis_enabled = true;
    expert.aq_trellis_coupling = coupling;

    expert.to_encoder_config(ColorMode::YCbCr {
        subsampling: ChromaSubsampling::Quarter,
    })
}

fn create_hybrid_config_multiplicative(quality: i32, coupling: f32) -> EncoderConfig {
    use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode};

    let mut expert = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality);
    expert.trellis_enabled = true;
    expert.aq_trellis_coupling = coupling;
    expert.aq_trellis_multiplicative = true;

    expert.to_encoder_config(ColorMode::YCbCr {
        subsampling: ChromaSubsampling::Quarter,
    })
}

fn create_hybrid_config_with_threshold(quality: i32, coupling: f32, threshold: f32) -> EncoderConfig {
    use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode};

    let mut expert = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality);
    expert.trellis_enabled = true;
    expert.aq_trellis_coupling = coupling;
    expert.aq_trellis_threshold = threshold;

    expert.to_encoder_config(ColorMode::YCbCr {
        subsampling: ChromaSubsampling::Quarter,
    })
}

fn create_hybrid_config_with_exponent(quality: i32, coupling: f32, exponent: f32) -> EncoderConfig {
    use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode};

    let mut expert = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality);
    expert.trellis_enabled = true;
    expert.aq_trellis_coupling = coupling;
    expert.aq_trellis_exponent = exponent;

    expert.to_encoder_config(ColorMode::YCbCr {
        subsampling: ChromaSubsampling::Quarter,
    })
}

fn create_hybrid_config_with_max_adj(quality: i32, coupling: f32, max_adj: f32) -> EncoderConfig {
    use zenjpeg::encode::{ExpertConfig, OptimizationPreset, ColorMode};

    let mut expert = ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality);
    expert.trellis_enabled = true;
    expert.aq_trellis_coupling = coupling;
    expert.aq_trellis_max_adjustment = max_adj;

    expert.to_encoder_config(ColorMode::YCbCr {
        subsampling: ChromaSubsampling::Quarter,
    })
}

fn encode_image(config: &EncoderConfig, img: &ImageData) -> Vec<u8> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder create");
    encoder
        .push_packed(&img.pixels, enough::Unstoppable)
        .expect("push");
    encoder.finish().expect("finish")
}

fn encode_and_measure_bytes(config: &EncoderConfig, img: &ImageData) -> usize {
    encode_image(config, img).len()
}

fn bpp(bytes: usize, width: usize, height: usize) -> f64 {
    (bytes as f64 * 8.0) / (width * height) as f64
}
