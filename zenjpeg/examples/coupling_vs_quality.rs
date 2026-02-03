//! Does coupling give a different rate-distortion curve than just changing quality?

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};
use zenjpeg::encode::trellis::HybridConfig;
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn encode(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    encoder.push_packed(&img.pixels, Unstoppable).ok()?;
    encoder.finish().ok()
}

fn butteraugli(img: &ImageData, jpeg: &[u8]) -> Option<f64> {
    let orig = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let dec: RgbImage = decode_jpeg_to_rgb(jpeg).ok()?;
    Some(QualityMetrics::butteraugli(orig.as_ref(), dec.as_ref()))
}

fn main() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";
    let images: Vec<_> = if Path::new(base_dir).exists() {
        std::fs::read_dir(base_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png"))
            .take(15)
            .collect()
    } else {
        let fb = "/home/lilith/work/codec-eval/codec-corpus/cid22";
        std::fs::read_dir(fb)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .map(|e| e.path())
            .take(15)
            .collect()
    };
    let loaded: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    println!("=== Coupling vs Quality: Same Curve? ===\n");
    println!("If coupling just shifts along the same curve as quality, it's pointless.\n");

    // Collect data points: (size, butteraugli, label)
    let mut points: Vec<(f64, f64, String)> = Vec::new();

    // JpegliProgressive at various qualities
    println!("JpegliProgressive quality sweep:");
    for q in [80.0, 82.0, 84.0, 85.0, 86.0, 88.0, 90.0] {
        let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliProgressive);
        let (mut sz, mut ba, mut n) = (0usize, 0.0, 0);
        for img in &loaded {
            if let Some(j) = encode(&cfg, img) {
                if let Some(b) = butteraugli(img, &j) {
                    sz += j.len();
                    ba += b;
                    n += 1;
                }
            }
        }
        let (sz_avg, ba_avg) = (sz as f64 / n as f64, ba / n as f64);
        println!("  Q{:>4.0}: {:>7.0} bytes, BA {:.3}", q, sz_avg, ba_avg);
        points.push((sz_avg, ba_avg, format!("Jpegli Q{}", q)));
    }

    // HybridProgressive with positive coupling at Q85
    println!("\nHybridProgressive +5 coupling, quality sweep:");
    for q in [82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0] {
        let hybrid = HybridConfig {
            enabled: true,
            aq_lambda_scale: 5.0,
            ..Default::default()
        };
        let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive)
            .hybrid_config(hybrid);
        let (mut sz, mut ba, mut n) = (0usize, 0.0, 0);
        for img in &loaded {
            if let Some(j) = encode(&cfg, img) {
                if let Some(b) = butteraugli(img, &j) {
                    sz += j.len();
                    ba += b;
                    n += 1;
                }
            }
        }
        let (sz_avg, ba_avg) = (sz as f64 / n as f64, ba / n as f64);
        println!("  Q{:>4.0}: {:>7.0} bytes, BA {:.3}", q, sz_avg, ba_avg);
        points.push((sz_avg, ba_avg, format!("Hybrid+5 Q{}", q)));
    }

    // HybridProgressive with negative coupling
    println!("\nHybridProgressive -4 coupling, quality sweep:");
    for q in [85.0, 86.0, 87.0, 88.0, 89.0, 90.0] {
        let hybrid = HybridConfig {
            enabled: true,
            aq_lambda_scale: -4.0,
            ..Default::default()
        };
        let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive)
            .hybrid_config(hybrid);
        let (mut sz, mut ba, mut n) = (0usize, 0.0, 0);
        for img in &loaded {
            if let Some(j) = encode(&cfg, img) {
                if let Some(b) = butteraugli(img, &j) {
                    sz += j.len();
                    ba += b;
                    n += 1;
                }
            }
        }
        let (sz_avg, ba_avg) = (sz as f64 / n as f64, ba / n as f64);
        println!("  Q{:>4.0}: {:>7.0} bytes, BA {:.3}", q, sz_avg, ba_avg);
        points.push((sz_avg, ba_avg, format!("Hybrid-4 Q{}", q)));
    }

    // Find comparable points (similar file size, compare BA)
    println!("\n=== Comparable Points (similar size, which has better BA?) ===\n");
    println!(
        "{:>20} {:>10} {:>8}  vs  {:>20} {:>10} {:>8}  {:>10}",
        "Config A", "Size", "BA", "Config B", "Size", "BA", "Winner"
    );
    println!("{}", "-".repeat(100));

    // Compare Jpegli Q85 vs Hybrid+5 at similar size
    let jp85 = points.iter().find(|p| p.2 == "Jpegli Q85").unwrap();

    // Find hybrid+5 point closest in size to jp85
    let hybrid_plus: Vec<_> = points
        .iter()
        .filter(|p| p.2.starts_with("Hybrid+5"))
        .collect();
    let closest_plus = hybrid_plus
        .iter()
        .min_by_key(|p| ((p.0 - jp85.0).abs() * 100.0) as i64)
        .unwrap();

    let ba_diff = (closest_plus.1 - jp85.1) / jp85.1 * 100.0;
    let winner = if ba_diff < -1.0 {
        "Hybrid+5"
    } else if ba_diff > 1.0 {
        "Jpegli"
    } else {
        "~same"
    };
    println!(
        "{:>20} {:>10.0} {:>8.3}  vs  {:>20} {:>10.0} {:>8.3}  {:>10} ({:+.1}% BA)",
        jp85.2, jp85.0, jp85.1, closest_plus.2, closest_plus.0, closest_plus.1, winner, ba_diff
    );

    // Find hybrid-4 point closest in size to jp85
    let hybrid_minus: Vec<_> = points
        .iter()
        .filter(|p| p.2.starts_with("Hybrid-4"))
        .collect();
    let closest_minus = hybrid_minus
        .iter()
        .min_by_key(|p| ((p.0 - jp85.0).abs() * 100.0) as i64)
        .unwrap();

    let ba_diff = (closest_minus.1 - jp85.1) / jp85.1 * 100.0;
    let winner = if ba_diff < -1.0 {
        "Hybrid-4"
    } else if ba_diff > 1.0 {
        "Jpegli"
    } else {
        "~same"
    };
    println!(
        "{:>20} {:>10.0} {:>10.3}  vs  {:>20} {:>10.0} {:>8.3}  {:>10} ({:+.1}% BA)",
        jp85.2, jp85.0, jp85.1, closest_minus.2, closest_minus.0, closest_minus.1, winner, ba_diff
    );

    println!("\n=== Conclusion ===");
    println!("If coupling traces the SAME curve as quality changes, it's redundant.");
    println!("If it traces a DIFFERENT curve, there may be value at specific operating points.");
}
