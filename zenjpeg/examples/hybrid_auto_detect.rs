//! Test hybrid trellis auto-detection based on AQ statistics.
//!
//! This example validates that image type detection works correctly:
//! - Photos get aggressive compression (smaller files)
//! - Screenshots get protected compression (quality preserved)
//!
//! Usage:
//!   cargo run --release --example hybrid_auto_detect
//!   cargo run --release --example hybrid_auto_detect -- /path/to/image.png

use enough::Unstoppable;
use std::env;
use std::path::Path;
use zenjpeg::encode::trellis::hybrid::{adaptive_config, detect_image_type, ImageType};
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::quant::aq::compute_aq_strength_map;
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn main() {
    let args: Vec<String> = env::args().collect();
    let image_paths: Vec<String> = if args.len() > 1 {
        args[1..].to_vec()
    } else {
        // Default test images - mix of photos and screenshots
        let mut defaults = vec![
            // Local testdata
            "internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png".to_string(),
        ];
        if let Ok(corpus) = codec_corpus::Corpus::new() {
            if let Ok(d) = corpus.get("cid22/cid22-source") {
                for f in ["p01.png", "p02.png", "p05.png"] {
                    defaults.push(d.join(f).to_string_lossy().to_string());
                }
            }
            if let Ok(d) = corpus.get("qoi-benchmark/screenshot_web") {
                defaults.push(d.join("apple.com.png").to_string_lossy().to_string());
            }
            if let Ok(d) = corpus.get("qoi-benchmark/screenshot_game") {
                defaults.push(d.join("rb3_1.png").to_string_lossy().to_string());
            }
        }
        defaults
    };

    // Filter to existing files
    let image_paths: Vec<String> = image_paths
        .iter()
        .filter(|p| Path::new(p).exists())
        .cloned()
        .collect();

    if image_paths.is_empty() {
        eprintln!("No test images found. Run from repo root or provide image paths.");
        return;
    }

    println!("=== Hybrid Auto-Detection Test ===\n");
    println!(
        "{:>40}  {:>8}  {:>8}  {:>6}  {:>12}  {:>10}  {:>10}  {:>10}",
        "Image", "AQ Mean", "AQ Std", "CV", "Detected", "Baseline", "Adaptive", "Δ Size"
    );
    println!(
        "{:-<40}  {:-<8}  {:-<8}  {:-<6}  {:-<12}  {:-<10}  {:-<10}  {:-<10}",
        "", "", "", "", "", "", "", ""
    );

    let mut results: Vec<DetectionResult> = Vec::new();

    for image_path in &image_paths {
        if let Some(result) = analyze_image(image_path) {
            let name = Path::new(image_path)
                .file_name()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default();

            let type_str = match result.detected_type {
                ImageType::Photo => "Photo",
                ImageType::Screenshot => "Screenshot",
                ImageType::Mixed => "Mixed",
            };

            let delta_pct =
                (result.adaptive_bytes as f64 / result.baseline_bytes as f64 - 1.0) * 100.0;

            println!(
                "{:>40}  {:>8.4}  {:>8.4}  {:>6.2}  {:>12}  {:>10}  {:>10}  {:>9.1}%",
                truncate(&name, 40),
                result.aq_mean,
                result.aq_std,
                result.cv,
                type_str,
                result.baseline_bytes,
                result.adaptive_bytes,
                delta_pct
            );

            results.push(result);
        }
    }

    // Summary
    println!("\n=== Summary ===\n");

    let photos: Vec<_> = results
        .iter()
        .filter(|r| r.detected_type == ImageType::Photo)
        .collect();
    let screenshots: Vec<_> = results
        .iter()
        .filter(|r| r.detected_type == ImageType::Screenshot)
        .collect();

    if !photos.is_empty() {
        let avg_delta: f64 = photos
            .iter()
            .map(|r| (r.adaptive_bytes as f64 / r.baseline_bytes as f64 - 1.0) * 100.0)
            .sum::<f64>()
            / photos.len() as f64;
        let avg_dssim_delta: f64 = photos
            .iter()
            .map(|r| (r.adaptive_dssim / r.baseline_dssim - 1.0) * 100.0)
            .sum::<f64>()
            / photos.len() as f64;
        println!(
            "Photos ({} images): avg size {:.1}%, avg DSSIM delta {:.1}%",
            photos.len(),
            avg_delta,
            avg_dssim_delta
        );
    }

    if !screenshots.is_empty() {
        let avg_delta: f64 = screenshots
            .iter()
            .map(|r| (r.adaptive_bytes as f64 / r.baseline_bytes as f64 - 1.0) * 100.0)
            .sum::<f64>()
            / screenshots.len() as f64;
        let avg_dssim_delta: f64 = screenshots
            .iter()
            .map(|r| (r.adaptive_dssim / r.baseline_dssim - 1.0) * 100.0)
            .sum::<f64>()
            / screenshots.len() as f64;
        println!(
            "Screenshots ({} images): avg size {:.1}%, avg DSSIM delta {:.1}%",
            screenshots.len(),
            avg_delta,
            avg_dssim_delta
        );
    }

    println!("\nExpected behavior:");
    println!("  - Photos: Size should decrease (negative %), DSSIM slightly worse");
    println!("  - Screenshots: Size may increase due to trellis overhead, DSSIM protected");
}

struct DetectionResult {
    aq_mean: f32,
    aq_std: f32,
    cv: f32,
    detected_type: ImageType,
    baseline_bytes: usize,
    adaptive_bytes: usize,
    baseline_dssim: f64,
    adaptive_dssim: f64,
}

fn analyze_image(path: &str) -> Option<DetectionResult> {
    let img = ImageData::from_path(Path::new(path))?;

    // Compute AQ statistics
    let y_plane = extract_y_plane(&img);
    let aq_map = compute_aq_strength_map(&y_plane, img.width, img.height, 1).ok()?;
    let (_, _, aq_mean, aq_std) = aq_map.stats();
    let cv = if aq_mean > 0.001 {
        aq_std / aq_mean
    } else {
        0.0
    };

    // Detect image type
    let detected_type = detect_image_type(aq_mean, aq_std);

    // Encode baseline (no hybrid)
    let baseline_config =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).optimize_huffman(true);
    let baseline_jpeg = encode_image(&baseline_config, &img)?;
    let baseline_bytes = baseline_jpeg.len();

    // Encode with adaptive config
    let hybrid = adaptive_config(aq_mean, aq_std);
    let adaptive_config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .optimize_huffman(true)
        .hybrid_config(hybrid);
    let adaptive_jpeg = encode_image(&adaptive_config, &img)?;
    let adaptive_bytes = adaptive_jpeg.len();

    // Compute DSSIM for both
    let orig_rgb = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let baseline_decoded: RgbImage = decode_jpeg_to_rgb(&baseline_jpeg).ok()?;
    let adaptive_decoded: RgbImage = decode_jpeg_to_rgb(&adaptive_jpeg).ok()?;
    let baseline_dssim = QualityMetrics::dssim(orig_rgb.as_ref(), baseline_decoded.as_ref());
    let adaptive_dssim = QualityMetrics::dssim(orig_rgb.as_ref(), adaptive_decoded.as_ref());

    Some(DetectionResult {
        aq_mean,
        aq_std,
        cv,
        detected_type,
        baseline_bytes,
        adaptive_bytes,
        baseline_dssim,
        adaptive_dssim,
    })
}

fn extract_y_plane(img: &ImageData) -> Vec<f32> {
    // Convert RGB to Y plane (BT.601 luma), 0-255 range
    let mut y = Vec::with_capacity(img.width * img.height);
    for chunk in img.pixels.chunks(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        // BT.601: Y = 0.299*R + 0.587*G + 0.114*B
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        y.push(luma);
    }
    y
}

fn encode_image(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    encoder.push_packed(&img.pixels, Unstoppable).ok()?;
    encoder.finish().ok()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max + 3..])
    }
}
