//! Test Approach A (Quality-Neutral) and D (Positive Coupling) for Pareto optimization
//!
//! Approach A: Hybrid at Q+N with negative coupling to match jpegli Q butteraugli
//! Approach D: Hybrid at Q-N with positive coupling to match jpegli Q size

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

    // Add hybrid config if coupling is non-zero
    if coupling.abs() > 0.001 {
        let hybrid = HybridConfig {
            enabled: true,
            aq_lambda_scale: coupling,
            max_adjustment: 0.0, // No cap for photos
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
    // Try to find corpus
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
        // Fallback to codec-corpus
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
            eprintln!("No corpus found at {} or {}", base_dir, fallback.display());
            return;
        }
    };

    if images.is_empty() {
        eprintln!("No images found");
        return;
    }

    println!("Testing {} images\n", images.len());

    let base_quality = 85.0;

    println!("=== APPROACH A: Quality-Neutral (boost quality to match butteraugli) ===\n");
    println!(
        "Goal: Same butteraugli as jpegli Q{}, smaller file\n",
        base_quality
    );

    // Test: hybrid at Q87 with negative coupling vs jpegli at Q85
    let quality_boosts = [1.0, 2.0, 3.0, 4.0, 5.0];
    let negative_coupling = -4.0; // Negative = aggressive compression

    println!("Baseline: jpegli Q{} (no hybrid)", base_quality);
    println!(
        "Test: hybrid Q{}+N with aq_lambda_scale={}\n",
        base_quality, negative_coupling
    );

    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Boost", "Jpegli sz", "Jpegli BA", "Hybrid sz", "Hybrid BA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(76));

    for boost in quality_boosts {
        let boosted_q = base_quality + boost;

        let jpegli_config = create_config(base_quality, 0.0);
        let hybrid_config = create_config(boosted_q, negative_coupling);

        let mut jpegli_size_sum = 0usize;
        let mut jpegli_ba_sum = 0.0f64;
        let mut hybrid_size_sum = 0usize;
        let mut hybrid_ba_sum = 0.0f64;
        let mut count = 0;

        for img_path in &images {
            let img = match ImageData::from_path(img_path) {
                Some(i) => i,
                None => continue,
            };

            let jpegli_bytes = match encode_with_config(&jpegli_config, &img) {
                Some(j) => j,
                None => continue,
            };
            let hybrid_bytes = match encode_with_config(&hybrid_config, &img) {
                Some(j) => j,
                None => continue,
            };

            let jpegli_ba = compute_butteraugli(&img, &jpegli_bytes).unwrap_or(999.0);
            let hybrid_ba = compute_butteraugli(&img, &hybrid_bytes).unwrap_or(999.0);

            jpegli_size_sum += jpegli_bytes.len();
            jpegli_ba_sum += jpegli_ba;
            hybrid_size_sum += hybrid_bytes.len();
            hybrid_ba_sum += hybrid_ba;
            count += 1;
        }

        if count > 0 {
            let jpegli_size_avg = jpegli_size_sum as f64 / count as f64;
            let jpegli_ba_avg = jpegli_ba_sum / count as f64;
            let hybrid_size_avg = hybrid_size_sum as f64 / count as f64;
            let hybrid_ba_avg = hybrid_ba_sum / count as f64;

            let size_delta = (hybrid_size_avg - jpegli_size_avg) / jpegli_size_avg * 100.0;
            let ba_delta = (hybrid_ba_avg - jpegli_ba_avg) / jpegli_ba_avg * 100.0;

            // Highlight Pareto-optimal results
            let pareto = if size_delta < 0.0 && ba_delta <= 0.5 {
                " ★"
            } else {
                ""
            };

            println!(
                "{:>6.1} {:>10.0} {:>10.3} {:>10.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
                boost,
                jpegli_size_avg,
                jpegli_ba_avg,
                hybrid_size_avg,
                hybrid_ba_avg,
                size_delta,
                ba_delta,
                pareto
            );
        }
    }

    println!("\n=== APPROACH D: Positive Coupling (preserve texture, reduce base quality) ===\n");
    println!(
        "Goal: Same file size as jpegli Q{}, better quality\n",
        base_quality
    );

    // Test: hybrid at Q83 with positive coupling vs jpegli at Q85
    let quality_reductions = [1.0, 2.0, 3.0, 4.0, 5.0];
    let positive_couplings = [1.0, 2.0, 4.0];

    println!("Baseline: jpegli Q{} (no hybrid)", base_quality);
    println!(
        "Test: hybrid Q{}-N with positive aq_lambda_scale\n",
        base_quality
    );

    for coupling in positive_couplings {
        println!("\n--- aq_lambda_scale = +{} ---\n", coupling);
        println!(
            "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "QReduc", "Jpegli sz", "Jpegli BA", "Hybrid sz", "Hybrid BA", "ΔSize%", "ΔBA%"
        );
        println!("{}", "-".repeat(76));

        for reduction in &quality_reductions {
            let reduced_q = base_quality - reduction;

            let jpegli_config = create_config(base_quality, 0.0);
            let hybrid_config = create_config(reduced_q, coupling);

            let mut jpegli_size_sum = 0usize;
            let mut jpegli_ba_sum = 0.0f64;
            let mut hybrid_size_sum = 0usize;
            let mut hybrid_ba_sum = 0.0f64;
            let mut count = 0;

            for img_path in &images {
                let img = match ImageData::from_path(img_path) {
                    Some(i) => i,
                    None => continue,
                };

                let jpegli_bytes = match encode_with_config(&jpegli_config, &img) {
                    Some(j) => j,
                    None => continue,
                };
                let hybrid_bytes = match encode_with_config(&hybrid_config, &img) {
                    Some(j) => j,
                    None => continue,
                };

                let jpegli_ba = compute_butteraugli(&img, &jpegli_bytes).unwrap_or(999.0);
                let hybrid_ba = compute_butteraugli(&img, &hybrid_bytes).unwrap_or(999.0);

                jpegli_size_sum += jpegli_bytes.len();
                jpegli_ba_sum += jpegli_ba;
                hybrid_size_sum += hybrid_bytes.len();
                hybrid_ba_sum += hybrid_ba;
                count += 1;
            }

            if count > 0 {
                let jpegli_size_avg = jpegli_size_sum as f64 / count as f64;
                let jpegli_ba_avg = jpegli_ba_sum / count as f64;
                let hybrid_size_avg = hybrid_size_sum as f64 / count as f64;
                let hybrid_ba_avg = hybrid_ba_sum / count as f64;

                let size_delta = (hybrid_size_avg - jpegli_size_avg) / jpegli_size_avg * 100.0;
                let ba_delta = (hybrid_ba_avg - jpegli_ba_avg) / jpegli_ba_avg * 100.0;

                // Highlight Pareto-optimal results (similar size AND better quality)
                let pareto = if size_delta.abs() < 2.0 && ba_delta < 0.0 {
                    " ★"
                } else {
                    ""
                };

                println!(
                    "{:>6.1} {:>10.0} {:>10.3} {:>10.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
                    reduction,
                    jpegli_size_avg,
                    jpegli_ba_avg,
                    hybrid_size_avg,
                    hybrid_ba_avg,
                    size_delta,
                    ba_delta,
                    pareto
                );
            }
        }
    }

    println!("\n★ = Near-Pareto (Approach A: smaller + similar quality, Approach D: similar size + better quality)");
}
