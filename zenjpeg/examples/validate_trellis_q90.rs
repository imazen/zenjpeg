//! Validate: Does trellis beat jpegli at Q90+?

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn encode(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut e = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    e.push_packed(&img.pixels, Unstoppable).ok()?;
    e.finish().ok()
}

fn ba(img: &ImageData, jpeg: &[u8]) -> Option<f64> {
    let o = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let d: RgbImage = decode_jpeg_to_rgb(jpeg).ok()?;
    Some(QualityMetrics::butteraugli(o.as_ref(), d.as_ref()))
}

fn main() {
    // Use ALL available CID22 images
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";
    let images: Vec<_> = if Path::new(base_dir).exists() {
        std::fs::read_dir(base_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png"))
            .collect()
    } else {
        let fb = "/home/lilith/work/codec-eval/codec-corpus/cid22";
        std::fs::read_dir(fb)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .map(|e| e.path())
            .collect()
    };
    let loaded: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    println!("=== Validate: Trellis vs Jpegli at Q90+ ===");
    println!("Testing {} images\n", loaded.len());

    println!(
        "{:>4} {:>12} {:>8} {:>12} {:>8} {:>10} {:>10} {:>8}",
        "Q", "Jpegli sz", "Jpegli BA", "Trellis sz", "Trellis BA", "ΔSize%", "ΔBA%", "Pareto?"
    );
    println!("{}", "-".repeat(90));

    let mut pareto_wins = 0;
    let mut pareto_losses = 0;

    for q in [88, 89, 90, 91, 92, 93, 94, 95] {
        let q = q as f32;

        let jpegli_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::JpegliProgressive);
        let trellis_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .optimization(OptimizationPreset::HybridProgressive);

        let (mut jsz, mut jba, mut tsz, mut tba, mut n) = (0usize, 0.0, 0usize, 0.0, 0);

        for img in &loaded {
            if let (Some(jj), Some(tj)) = (encode(&jpegli_cfg, img), encode(&trellis_cfg, img)) {
                if let (Some(jb), Some(tb)) = (ba(img, &jj), ba(img, &tj)) {
                    jsz += jj.len();
                    jba += jb;
                    tsz += tj.len();
                    tba += tb;
                    n += 1;
                }
            }
        }

        let (jsz_avg, jba_avg) = (jsz as f64 / n as f64, jba / n as f64);
        let (tsz_avg, tba_avg) = (tsz as f64 / n as f64, tba / n as f64);

        let sz_d = (tsz_avg - jsz_avg) / jsz_avg * 100.0;
        let ba_d = (tba_avg - jba_avg) / jba_avg * 100.0;

        // Pareto win: smaller AND (same or better quality)
        // Allow 1% BA tolerance for "same quality"
        let pareto = if sz_d < -0.5 && ba_d < 1.0 {
            pareto_wins += 1;
            "★ WIN"
        } else if sz_d > 0.5 && ba_d > 1.0 {
            pareto_losses += 1;
            "LOSE"
        } else {
            "~same"
        };

        println!(
            "{:>4.0} {:>12.0} {:>8.3} {:>12.0} {:>8.3} {:>+10.1}% {:>+10.1}% {:>8}",
            q, jsz_avg, jba_avg, tsz_avg, tba_avg, sz_d, ba_d, pareto
        );
    }

    println!("\n=== Per-Image Analysis at Q90 ===\n");

    let q = 90.0;
    let jpegli_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::JpegliProgressive);
    let trellis_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive);

    println!(
        "{:>30} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Image", "Jpegli sz", "Trellis sz", "ΔSize%", "ΔBA%", "Winner"
    );
    println!("{}", "-".repeat(85));

    let mut jpegli_wins = 0;
    let mut trellis_wins = 0;
    let mut ties = 0;

    for img in &loaded {
        if let (Some(jj), Some(tj)) = (encode(&jpegli_cfg, img), encode(&trellis_cfg, img)) {
            if let (Some(jb), Some(tb)) = (ba(img, &jj), ba(img, &tj)) {
                let sz_d = (tj.len() as f64 - jj.len() as f64) / jj.len() as f64 * 100.0;
                let ba_d = (tb - jb) / jb * 100.0;

                let winner = if sz_d < -1.0 && ba_d < 2.0 {
                    trellis_wins += 1;
                    "Trellis"
                } else if sz_d > 1.0 && ba_d > 2.0 {
                    jpegli_wins += 1;
                    "Jpegli"
                } else {
                    ties += 1;
                    "~tie"
                };

                let name = img.name.chars().take(30).collect::<String>();
                println!(
                    "{:>30} {:>10} {:>10} {:>+10.1}% {:>+10.1}% {:>8}",
                    name,
                    jj.len(),
                    tj.len(),
                    sz_d,
                    ba_d,
                    winner
                );
            }
        }
    }

    println!("\n=== Summary ===\n");
    println!(
        "Q88-95 Pareto analysis: {} wins, {} losses",
        pareto_wins, pareto_losses
    );
    println!(
        "Q90 per-image: Trellis wins {}, Jpegli wins {}, Ties {}",
        trellis_wins, jpegli_wins, ties
    );

    if pareto_wins > pareto_losses && trellis_wins > jpegli_wins {
        println!("\n✓ VALIDATED: Trellis helps at Q90+");
    } else {
        println!("\n✗ NOT VALIDATED: Results inconclusive or negative");
    }
}
