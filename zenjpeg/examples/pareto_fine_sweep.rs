//! Fine sweep around the Pareto-optimal point found in pareto_approaches.rs
//!
//! Found: aq_lambda_scale=+4, Q84 gives -0.9% size, -1.5% butteraugli vs Q85 baseline
//! Let's find the exact optimal.

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::encode::trellis::HybridConfig;
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn encode_with_config(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    encoder.push_packed(&img.pixels, Unstoppable).ok()?;
    encoder.finish().ok()
}

fn create_config(quality: f32, coupling: f32) -> EncoderConfig {
    let mut config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .progressive(true)
        .optimize_huffman(true);

    if coupling.abs() > 0.001 {
        let hybrid = HybridConfig {
            enabled: true,
            aq_lambda_scale: coupling,
            max_adjustment: 0.0,
            ..Default::default()
        };
        config = config.hybrid_config(hybrid);
    }

    config
}

fn compute_butteraugli(img: &ImageData, jpeg: &[u8]) -> Option<f64> {
    let orig_rgb = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let decoded: RgbImage = decode_jpeg_to_rgb(jpeg).ok()?;
    Some(QualityMetrics::butteraugli(
        orig_rgb.as_ref(),
        decoded.as_ref(),
    ))
}

fn main() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";
    let corpus = Path::new(base_dir);

    let images: Vec<_> = if corpus.exists() {
        std::fs::read_dir(corpus)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png"))
            .take(10)
            .collect()
    } else {
        let fallback = Path::new("/home/lilith/work/codec-eval/codec-corpus/cid22");
        if fallback.exists() {
            std::fs::read_dir(fallback)
                .expect("read dir")
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|x| x == "png" || x == "jpg")
                        .unwrap_or(false)
                })
                .map(|e| e.path())
                .take(10)
                .collect()
        } else {
            eprintln!("No corpus found");
            return;
        }
    };

    let base_quality = 85.0;
    let jpegli_config = create_config(base_quality, 0.0);

    println!("=== Fine Sweep: Positive Coupling + Quality Reduction ===\n");
    println!("Baseline: jpegli Q{} (no hybrid)\n", base_quality);
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Coupling", "Quality", "Hybrid sz", "Hybrid BA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(66));

    // Calculate baseline once
    let mut jpegli_size_sum = 0usize;
    let mut jpegli_ba_sum = 0.0f64;
    let mut count = 0;

    let loaded_images: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    for img in &loaded_images {
        if let Some(jpegli_bytes) = encode_with_config(&jpegli_config, img) {
            let jpegli_ba = compute_butteraugli(img, &jpegli_bytes).unwrap_or(999.0);
            jpegli_size_sum += jpegli_bytes.len();
            jpegli_ba_sum += jpegli_ba;
            count += 1;
        }
    }

    let jpegli_size_avg = jpegli_size_sum as f64 / count as f64;
    let jpegli_ba_avg = jpegli_ba_sum / count as f64;

    println!(
        "{:>8} {:>8} {:>10.0} {:>10.3} {:>10} {:>10} (baseline)",
        "-", base_quality, jpegli_size_avg, jpegli_ba_avg, "-", "-"
    );
    println!();

    // Fine sweep around the optimal
    let couplings = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0];
    let quality_offsets = [0.0, -0.5, -1.0, -1.5, -2.0];

    let mut best_pareto: Option<(f32, f32, f64, f64)> = None;

    for coupling in couplings {
        for offset in quality_offsets {
            let quality = base_quality + offset;
            let hybrid_config = create_config(quality, coupling);

            let mut hybrid_size_sum = 0usize;
            let mut hybrid_ba_sum = 0.0f64;

            for img in &loaded_images {
                if let Some(hybrid_bytes) = encode_with_config(&hybrid_config, img) {
                    let hybrid_ba = compute_butteraugli(img, &hybrid_bytes).unwrap_or(999.0);
                    hybrid_size_sum += hybrid_bytes.len();
                    hybrid_ba_sum += hybrid_ba;
                }
            }

            let hybrid_size_avg = hybrid_size_sum as f64 / count as f64;
            let hybrid_ba_avg = hybrid_ba_sum / count as f64;

            let size_delta = (hybrid_size_avg - jpegli_size_avg) / jpegli_size_avg * 100.0;
            let ba_delta = (hybrid_ba_avg - jpegli_ba_avg) / jpegli_ba_avg * 100.0;

            // Pareto: smaller AND better quality
            let is_pareto = size_delta < 0.0 && ba_delta < 0.0;
            let pareto_marker = if is_pareto { " ★" } else { "" };

            // Track best Pareto point (maximize size savings with better quality)
            if is_pareto {
                let score = -size_delta - ba_delta; // Both negative is good
                if best_pareto.is_none() || score > best_pareto.unwrap().2 + best_pareto.unwrap().3
                {
                    best_pareto = Some((coupling, offset, -size_delta, -ba_delta));
                }
            }

            println!(
                "{:>8.1} {:>8.1} {:>10.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
                coupling,
                quality,
                hybrid_size_avg,
                hybrid_ba_avg,
                size_delta,
                ba_delta,
                pareto_marker
            );
        }
    }

    println!("\n★ = Pareto improvement (smaller size AND better quality)\n");

    if let Some((c, q_off, sz, ba)) = best_pareto {
        println!(
            "BEST PARETO: aq_lambda_scale={}, Q{} → {:.1}% smaller, {:.1}% better BA",
            c,
            base_quality + q_off,
            sz,
            ba
        );
    }
}
