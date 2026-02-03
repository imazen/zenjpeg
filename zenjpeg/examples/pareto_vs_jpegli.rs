//! Compare hybrid trellis against JpegliProgressive (the zenjpeg mode that matches C++ cjpegli)

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};
use zenjpeg::encode::trellis::HybridConfig;
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn encode_with_config(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    encoder.push_packed(&img.pixels, Unstoppable).ok()?;
    encoder.finish().ok()
}

fn create_jpegli_config(quality: f32) -> EncoderConfig {
    EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::JpegliProgressive)
}

fn create_hybrid_config(quality: f32, coupling: f32) -> EncoderConfig {
    let hybrid = HybridConfig {
        enabled: true,
        aq_lambda_scale: coupling,
        max_adjustment: 0.0,
        ..Default::default()
    };
    EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive)
        .hybrid_config(hybrid)
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
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png"))
            .take(20)
            .collect()
    } else {
        let fallback = Path::new("/home/lilith/work/codec-eval/codec-corpus/cid22");
        std::fs::read_dir(fallback)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .map(|e| e.path())
            .take(20)
            .collect()
    };

    let loaded: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();
    println!("=== Hybrid vs JpegliProgressive (the real baseline) ===\n");
    println!("Testing {} images\n", loaded.len());

    // Calculate jpegli baseline at Q85
    let jpegli_cfg = create_jpegli_config(85.0);
    let mut jp_sizes = Vec::new();
    let mut jp_bas = Vec::new();

    for img in &loaded {
        if let Some(jpeg) = encode_with_config(&jpegli_cfg, img) {
            if let Some(ba) = compute_butteraugli(img, &jpeg) {
                jp_sizes.push(jpeg.len());
                jp_bas.push(ba);
            }
        }
    }

    let jp_sz_avg = jp_sizes.iter().sum::<usize>() as f64 / jp_sizes.len() as f64;
    let jp_ba_avg: f64 = jp_bas.iter().sum::<f64>() / jp_bas.len() as f64;

    println!(
        "{:>35} {:>10} {:>10} {:>10} {:>10}",
        "Config", "Avg Size", "Avg BA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(80));
    println!(
        "{:>35} {:>10.0} {:>10.3} {:>10} {:>10}",
        "JpegliProgressive Q85 (baseline)", jp_sz_avg, jp_ba_avg, "-", "-"
    );

    // Test hybrid configs
    let test_configs: Vec<(&str, f32, f32)> = vec![
        // (name, coupling, quality)
        ("HybridProgressive Q85 (no coupling)", 0.0, 85.0),
        ("Hybrid +5 coupling, Q85", 5.0, 85.0),
        ("Hybrid +5 coupling, Q84", 5.0, 84.0),
        ("Hybrid +5 coupling, Q83.5", 5.0, 83.5),
        ("Hybrid -4 coupling, Q85", -4.0, 85.0),
        ("Hybrid -4 coupling, Q87", -4.0, 87.0),
        ("Hybrid -4 coupling, Q88", -4.0, 88.0),
    ];

    for (name, coupling, quality) in &test_configs {
        let config = if coupling.abs() < 0.001 {
            EncoderConfig::ycbcr(*quality, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::HybridProgressive)
        } else {
            create_hybrid_config(*quality, *coupling)
        };

        let mut sizes = Vec::new();
        let mut bas = Vec::new();

        for img in &loaded {
            if let Some(jpeg) = encode_with_config(&config, img) {
                if let Some(ba) = compute_butteraugli(img, &jpeg) {
                    sizes.push(jpeg.len());
                    bas.push(ba);
                }
            }
        }

        let sz_avg = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        let ba_avg: f64 = bas.iter().sum::<f64>() / bas.len() as f64;

        let sz_d = (sz_avg - jp_sz_avg) / jp_sz_avg * 100.0;
        let ba_d = (ba_avg - jp_ba_avg) / jp_ba_avg * 100.0;

        let pareto = if sz_d < -0.5 && ba_d < -0.5 {
            " ★"
        } else {
            ""
        };

        println!(
            "{:>35} {:>10.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
            name, sz_avg, ba_avg, sz_d, ba_d, pareto
        );
    }

    println!("\n★ = Beats JpegliProgressive on BOTH size (>0.5%) AND quality (>0.5%)\n");

    // Now test across quality levels
    println!("\n=== Hybrid vs Jpegli Across Quality Levels ===\n");
    println!(
        "{:>6} {:>12} {:>10} {:>12} {:>10} {:>10} {:>10}",
        "Q", "Jpegli sz", "Jpegli BA", "Hybrid sz", "Hybrid BA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(82));

    for q in [75.0, 80.0, 85.0, 90.0, 95.0] {
        let jpegli_cfg = create_jpegli_config(q);
        // Hybrid with +5 coupling, same quality (should be larger but better quality)
        let hybrid_cfg = create_hybrid_config(q, 5.0);

        let mut jp_sz = 0usize;
        let mut jp_ba = 0.0;
        let mut hy_sz = 0usize;
        let mut hy_ba = 0.0;
        let mut cnt = 0;

        for img in &loaded {
            if let (Some(jp), Some(hy)) = (
                encode_with_config(&jpegli_cfg, img),
                encode_with_config(&hybrid_cfg, img),
            ) {
                if let (Some(jp_b), Some(hy_b)) =
                    (compute_butteraugli(img, &jp), compute_butteraugli(img, &hy))
                {
                    jp_sz += jp.len();
                    jp_ba += jp_b;
                    hy_sz += hy.len();
                    hy_ba += hy_b;
                    cnt += 1;
                }
            }
        }

        let jp_sz_avg = jp_sz as f64 / cnt as f64;
        let jp_ba_avg = jp_ba / cnt as f64;
        let hy_sz_avg = hy_sz as f64 / cnt as f64;
        let hy_ba_avg = hy_ba / cnt as f64;

        let sz_d = (hy_sz_avg - jp_sz_avg) / jp_sz_avg * 100.0;
        let ba_d = (hy_ba_avg - jp_ba_avg) / jp_ba_avg * 100.0;

        let p = if sz_d < -0.5 && ba_d < -0.5 {
            " ★"
        } else {
            ""
        };

        println!(
            "{:>6.0} {:>12.0} {:>10.3} {:>12.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
            q, jp_sz_avg, jp_ba_avg, hy_sz_avg, hy_ba_avg, sz_d, ba_d, p
        );
    }

    println!("\n=== Fine Search: Can we beat Jpegli Q85? ===\n");
    println!("Looking for hybrid config that beats both size AND quality...\n");

    // Grid search
    let couplings: [f32; 7] = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0];
    let quality_offsets: [f32; 6] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    let mut best: Option<(f32, f32, f64, f64)> = None;

    for coupling in couplings {
        for offset in quality_offsets {
            let quality = 85.0 + offset;
            let config = if coupling.abs() < 0.001 {
                EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
                    .optimization(OptimizationPreset::HybridProgressive)
            } else {
                create_hybrid_config(quality, coupling)
            };

            let mut sizes = Vec::new();
            let mut bas = Vec::new();

            for img in &loaded {
                if let Some(jpeg) = encode_with_config(&config, img) {
                    if let Some(ba) = compute_butteraugli(img, &jpeg) {
                        sizes.push(jpeg.len());
                        bas.push(ba);
                    }
                }
            }

            let sz_avg = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
            let ba_avg: f64 = bas.iter().sum::<f64>() / bas.len() as f64;

            let sz_d = (sz_avg - jp_sz_avg) / jp_sz_avg * 100.0;
            let ba_d = (ba_avg - jp_ba_avg) / jp_ba_avg * 100.0;

            // Look for Pareto improvement
            if sz_d < 0.0 && ba_d < 0.0 {
                let score = -sz_d - ba_d;
                if best.is_none() || score > best.unwrap().2 + best.unwrap().3 {
                    best = Some((coupling, offset, -sz_d, -ba_d));
                    println!(
                        "★ PARETO: coupling={:+.1}, Q{:.1} → {:.1}% smaller, {:.1}% better BA",
                        coupling, quality, -sz_d, -ba_d
                    );
                }
            }
        }
    }

    if best.is_none() {
        println!("No Pareto improvement found over JpegliProgressive Q85.");
        println!("\nHybrid shifts the rate-distortion curve but doesn't dominate it.");
    }
}
