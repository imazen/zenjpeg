//! Which knobs compose well with jpegli? (give different R-D curves)

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, ScanMode,
};
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

fn test_curve(name: &str, configs: Vec<(f32, EncoderConfig)>, images: &[ImageData]) {
    println!("\n{}", name);
    println!("{:>6} {:>10} {:>8}", "Q", "Size", "BA");
    for (q, cfg) in configs {
        let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
        for img in images {
            if let Some(j) = encode(&cfg, img) {
                if let Some(v) = ba(img, &j) {
                    sz += j.len();
                    b += v;
                    n += 1;
                }
            }
        }
        println!(
            "{:>6.0} {:>10.0} {:>8.3}",
            q,
            sz as f64 / n as f64,
            b / n as f64
        );
    }
}

fn main() {
    let base_dir = "../glassa/results/cid22_comparison/butteraugli_matched";
    let images: Vec<_> = if Path::new(base_dir).exists() {
        std::fs::read_dir(base_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png"))
            .take(10)
            .collect()
    } else {
        let fb = "/home/lilith/work/codec-eval/codec-corpus/cid22";
        std::fs::read_dir(fb)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
            .map(|e| e.path())
            .take(10)
            .collect()
    };
    let loaded: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    println!("=== Which Knobs Give Different R-D Curves vs Jpegli? ===");
    println!("Testing {} images\n", loaded.len());

    let qs = [80.0, 85.0, 90.0];

    // 1. Baseline: JpegliProgressive
    test_curve(
        "1. JpegliProgressive (baseline)",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::JpegliProgressive),
                )
            })
            .collect(),
        &loaded,
    );

    // 2. Trellis (HybridProgressive without coupling)
    test_curve(
        "2. Trellis (HybridProgressive, no coupling)",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::HybridProgressive),
                )
            })
            .collect(),
        &loaded,
    );

    // 3. Scan optimization (progressive search)
    test_curve(
        "3. Jpegli + Scan Optimization",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::JpegliProgressive)
                        .optimize_scans(true),
                )
            })
            .collect(),
        &loaded,
    );

    // 4. Mozjpeg tables (Robidoux) instead of Jpegli tables
    test_curve(
        "4. MozjpegProgressive (different tables, has trellis)",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::MozjpegProgressive),
                )
            })
            .collect(),
        &loaded,
    );

    // 5. 4:4:4 subsampling
    test_curve(
        "5. Jpegli 4:4:4 (no chroma subsampling)",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::None)
                        .optimization(OptimizationPreset::JpegliProgressive),
                )
            })
            .collect(),
        &loaded,
    );

    // 6. Deringing
    test_curve(
        "6. Jpegli + Deringing",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::JpegliProgressive)
                        .deringing(true),
                )
            })
            .collect(),
        &loaded,
    );

    // 7. Combined: Trellis + Scan Optimization
    test_curve(
        "7. Trellis + Scan Optimization",
        qs.iter()
            .map(|&q| {
                (
                    q,
                    EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                        .optimization(OptimizationPreset::HybridProgressive)
                        .optimize_scans(true),
                )
            })
            .collect(),
        &loaded,
    );

    println!("\n=== Summary at Q85 ===\n");
    println!("Comparing at Q85 to see which give DIFFERENT curves:\n");

    let q = 85.0;
    let configs: Vec<(&str, EncoderConfig)> = vec![
        (
            "JpegliProgressive",
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::JpegliProgressive),
        ),
        (
            "+ Trellis",
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::HybridProgressive),
        ),
        (
            "+ Scan Opt",
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::JpegliProgressive)
                .optimize_scans(true),
        ),
        (
            "+ Deringing",
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::JpegliProgressive)
                .deringing(true),
        ),
        (
            "MozjpegProg",
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::MozjpegProgressive),
        ),
        (
            "Trellis+ScanOpt",
            EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::HybridProgressive)
                .optimize_scans(true),
        ),
    ];

    // Get baseline
    let base_cfg = &configs[0].1;
    let (mut bsz, mut bba, mut bn) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(base_cfg, img) {
            if let Some(v) = ba(img, &j) {
                bsz += j.len();
                bba += v;
                bn += 1;
            }
        }
    }
    let (base_sz, base_ba) = (bsz as f64 / bn as f64, bba / bn as f64);

    println!(
        "{:>20} {:>10} {:>8} {:>10} {:>10}",
        "Config", "Size", "BA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(62));

    for (name, cfg) in &configs {
        let (mut sz, mut b, mut n) = (0usize, 0.0, 0);
        for img in &loaded {
            if let Some(j) = encode(cfg, img) {
                if let Some(v) = ba(img, &j) {
                    sz += j.len();
                    b += v;
                    n += 1;
                }
            }
        }
        let (s, ba_v) = (sz as f64 / n as f64, b / n as f64);
        let sd = (s - base_sz) / base_sz * 100.0;
        let bd = (ba_v - base_ba) / base_ba * 100.0;
        let marker = if sd < -1.0 && bd < 1.0 {
            " ★"
        } else if sd.abs() < 0.5 && bd.abs() < 0.5 {
            " (same)"
        } else {
            ""
        };
        println!(
            "{:>20} {:>10.0} {:>8.3} {:>+10.1}% {:>+10.1}%{}",
            name, s, ba_v, sd, bd, marker
        );
    }

    println!("\n★ = Smaller files with similar or better quality (Pareto win)");
}
