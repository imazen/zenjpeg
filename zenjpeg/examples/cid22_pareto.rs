//! Check if hybrid offers Pareto improvement over jpegli baseline
//! by comparing at multiple quality levels.

use std::path::Path;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::hybrid::config::adaptive_config;
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};
use enough::Unstoppable;

fn main() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";

    let mut images: Vec<_> = std::fs::read_dir(base_dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("original.png").exists())
        .map(|e| e.path().join("original.png"))
        .collect();
    images.sort();

    if images.is_empty() {
        eprintln!("No CID22 images found");
        return;
    }

    // Test at multiple quality levels
    let qualities = [75, 80, 85, 90, 95];

    println!("=== Pareto Analysis: Hybrid vs Jpegli on CID22 ===\n");

    // Collect all data points
    let mut jpegli_points: Vec<(f64, f64)> = Vec::new(); // (size, butteraugli)
    let mut hybrid_points: Vec<(f64, f64)> = Vec::new();

    for &quality in &qualities {
        let mut jpegli_size_sum = 0usize;
        let mut jpegli_butter_sum = 0.0f64;
        let mut hybrid_size_sum = 0usize;
        let mut hybrid_butter_sum = 0.0f64;
        let mut count = 0;

        for img_path in images.iter().take(20) {
            let img = match ImageData::from_path(img_path) {
                Some(i) => i,
                None => continue,
            };

            // Compute AQ stats
            let y_plane = extract_y_plane(&img);
            let aq_map = match zenjpeg::quant::aq::compute_aq_strength_map(&y_plane, img.width, img.height, 1) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let (_, _, aq_mean, aq_std) = aq_map.stats();

            // Jpegli baseline
            let jpegli_config = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter)
                .optimize_huffman(true);
            let jpegli_jpeg = match encode_image(&jpegli_config, &img) {
                Some(j) => j,
                None => continue,
            };

            // Hybrid
            let hybrid = adaptive_config(aq_mean, aq_std);
            let hybrid_config = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter)
                .optimize_huffman(true)
                .hybrid_config(hybrid);
            let hybrid_jpeg = match encode_image(&hybrid_config, &img) {
                Some(j) => j,
                None => continue,
            };

            // Compute butteraugli
            let orig_rgb = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
            let jpegli_decoded: RgbImage = match decode_jpeg_to_rgb(&jpegli_jpeg) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let hybrid_decoded: RgbImage = match decode_jpeg_to_rgb(&hybrid_jpeg) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let jpegli_butter = QualityMetrics::butteraugli(orig_rgb.as_ref(), jpegli_decoded.as_ref());
            let hybrid_butter = QualityMetrics::butteraugli(orig_rgb.as_ref(), hybrid_decoded.as_ref());

            jpegli_size_sum += jpegli_jpeg.len();
            jpegli_butter_sum += jpegli_butter;
            hybrid_size_sum += hybrid_jpeg.len();
            hybrid_butter_sum += hybrid_butter;
            count += 1;
        }

        if count > 0 {
            let jpegli_avg_size = jpegli_size_sum as f64 / count as f64;
            let jpegli_avg_butter = jpegli_butter_sum / count as f64;
            let hybrid_avg_size = hybrid_size_sum as f64 / count as f64;
            let hybrid_avg_butter = hybrid_butter_sum / count as f64;

            jpegli_points.push((jpegli_avg_size, jpegli_avg_butter));
            hybrid_points.push((hybrid_avg_size, hybrid_avg_butter));

            println!("Q{}: Jpegli={:.0}B/{:.3}ba, Hybrid={:.0}B/{:.3}ba",
                quality, jpegli_avg_size, jpegli_avg_butter,
                hybrid_avg_size, hybrid_avg_butter);
        }
    }

    println!("\n=== Pareto Comparison ===\n");
    println!("For each hybrid point, find if it dominates any jpegli point:");
    println!("(Dominates = smaller size AND better/equal quality, or equal size AND better quality)\n");

    let mut any_pareto_win = false;

    for (i, &quality) in qualities.iter().enumerate() {
        let (h_size, h_butter) = hybrid_points[i];

        // Check if this hybrid point dominates any jpegli point
        for (j, &jq) in qualities.iter().enumerate() {
            let (j_size, j_butter) = jpegli_points[j];

            // Hybrid dominates if: smaller size AND <= butteraugli
            // OR: <= size AND smaller butteraugli
            let size_better = h_size < j_size;
            let size_equal = (h_size - j_size).abs() / j_size < 0.01; // within 1%
            let quality_better = h_butter < j_butter;
            let quality_equal = (h_butter - j_butter).abs() / j_butter < 0.01; // within 1%

            if (size_better && (quality_better || quality_equal)) ||
               (size_equal && quality_better) {
                println!("✓ Hybrid Q{} ({:.0}B, {:.3}ba) dominates Jpegli Q{} ({:.0}B, {:.3}ba)",
                    quality, h_size, h_butter, jq, j_size, j_butter);
                any_pareto_win = true;
            }
        }
    }

    if !any_pareto_win {
        println!("✗ No Pareto dominance found - hybrid is always worse on at least one axis");
    }

    println!("\n=== Rate-Distortion Table ===\n");
    println!("{:>5}  {:>12}  {:>10}  {:>12}  {:>10}  {:>8}  {:>8}",
        "Q", "Jpegli Size", "Jpegli BA", "Hybrid Size", "Hybrid BA", "ΔSize%", "ΔBA%");
    println!("{:-<5}  {:-<12}  {:-<10}  {:-<12}  {:-<10}  {:-<8}  {:-<8}",
        "", "", "", "", "", "", "");

    for (i, &quality) in qualities.iter().enumerate() {
        let (j_size, j_butter) = jpegli_points[i];
        let (h_size, h_butter) = hybrid_points[i];
        let size_delta = (h_size / j_size - 1.0) * 100.0;
        let butter_delta = (h_butter / j_butter - 1.0) * 100.0;

        println!("{:>5}  {:>12.0}  {:>10.3}  {:>12.0}  {:>10.3}  {:>+8.1}  {:>+8.1}",
            quality, j_size, j_butter, h_size, h_butter, size_delta, butter_delta);
    }

    // Check cross-quality comparisons
    println!("\n=== Cross-Quality Matches ===");
    println!("Can hybrid at Q_h match jpegli at Q_j with better size or quality?\n");

    for (i, &hq) in qualities.iter().enumerate() {
        let (h_size, h_butter) = hybrid_points[i];

        // Find jpegli quality with similar butteraugli
        for (j, &jq) in qualities.iter().enumerate() {
            let (j_size, j_butter) = jpegli_points[j];

            // If butteraugli is within 5%, compare sizes
            if (h_butter - j_butter).abs() / j_butter < 0.05 {
                let size_delta = (h_size / j_size - 1.0) * 100.0;
                if size_delta.abs() > 1.0 { // Only show if meaningful difference
                    println!("Hybrid Q{} ≈ Jpegli Q{} quality ({:.3} vs {:.3} BA), size: {:+.1}%",
                        hq, jq, h_butter, j_butter, size_delta);
                }
            }
        }
    }
}

fn extract_y_plane(img: &ImageData) -> Vec<f32> {
    let mut y = Vec::with_capacity(img.width * img.height);
    for chunk in img.pixels.chunks(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;
        y.push(0.299 * r + 0.587 * g + 0.114 * b);
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
