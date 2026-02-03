//! Validate Pareto findings on full CID22 (20 images)

use enough::Unstoppable;
use std::path::Path;
use zenjpeg::encode::trellis::HybridConfig;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};
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
        config = config.hybrid_config(HybridConfig {
            enabled: true,
            aq_lambda_scale: coupling,
            max_adjustment: 0.0,
            ..Default::default()
        });
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
            .expect("read")
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join("original.png").exists())
            .map(|e| e.path().join("original.png"))
            .collect()
    } else {
        let fallback = Path::new("/home/lilith/work/codec-eval/codec-corpus/cid22");
        std::fs::read_dir(fallback)
            .expect("read")
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "png" || x == "jpg")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect()
    };

    let loaded: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();
    println!("Testing {} images\n", loaded.len());

    // Test configurations: (name, coupling, quality)
    let configs = [
        ("baseline Q85", 0.0, 85.0),
        ("quality-focus (4.0, Q84)", 4.0, 84.0),
        ("balanced (5.0, Q83.5)", 5.0, 83.5),
        ("size-focus (6.0, Q83.5)", 6.0, 83.5),
    ];

    println!(
        "{:>30} {:>10} {:>10} {:>10} {:>10}",
        "Configuration", "Avg Size", "Avg BA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(76));

    let mut baseline_size = 0.0;
    let mut baseline_ba = 0.0;

    for (name, coupling, quality) in &configs {
        let config = create_config(*quality, *coupling);

        let mut size_sum = 0usize;
        let mut ba_sum = 0.0f64;
        let mut count = 0;

        for img in &loaded {
            if let Some(bytes) = encode_with_config(&config, img) {
                if let Some(ba) = compute_butteraugli(img, &bytes) {
                    size_sum += bytes.len();
                    ba_sum += ba;
                    count += 1;
                }
            }
        }

        let size_avg = size_sum as f64 / count as f64;
        let ba_avg = ba_sum / count as f64;

        if *coupling == 0.0 {
            baseline_size = size_avg;
            baseline_ba = ba_avg;
            println!(
                "{:>30} {:>10.0} {:>10.3} {:>10} {:>10}",
                name, size_avg, ba_avg, "-", "-"
            );
        } else {
            let size_delta = (size_avg - baseline_size) / baseline_size * 100.0;
            let ba_delta = (ba_avg - baseline_ba) / baseline_ba * 100.0;
            let pareto = if size_delta < 0.0 && ba_delta < 0.0 {
                " ★"
            } else {
                ""
            };
            println!(
                "{:>30} {:>10.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
                name, size_avg, ba_avg, size_delta, ba_delta, pareto
            );
        }
    }

    println!("\n★ = Pareto improvement\n");

    // Test across multiple quality levels
    println!("\n=== Pareto Check Across Quality Levels ===\n");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "BaseQ", "Baseline", "BaseBA", "Hybrid", "HybridBA", "ΔSize%", "ΔBA%"
    );
    println!("{}", "-".repeat(76));

    for base_q in [75.0, 80.0, 85.0, 90.0, 95.0] {
        let baseline_cfg = create_config(base_q, 0.0);
        // Use the balanced config: coupling=5, quality-1.5
        let hybrid_cfg = create_config(base_q - 1.5, 5.0);

        let mut bl_sz = 0usize;
        let mut bl_ba = 0.0;
        let mut hy_sz = 0usize;
        let mut hy_ba = 0.0;
        let mut cnt = 0;

        for img in &loaded {
            if let (Some(bl), Some(hy)) = (
                encode_with_config(&baseline_cfg, img),
                encode_with_config(&hybrid_cfg, img),
            ) {
                if let (Some(bl_b), Some(hy_b)) =
                    (compute_butteraugli(img, &bl), compute_butteraugli(img, &hy))
                {
                    bl_sz += bl.len();
                    bl_ba += bl_b;
                    hy_sz += hy.len();
                    hy_ba += hy_b;
                    cnt += 1;
                }
            }
        }

        let bl_sz_avg = bl_sz as f64 / cnt as f64;
        let bl_ba_avg = bl_ba / cnt as f64;
        let hy_sz_avg = hy_sz as f64 / cnt as f64;
        let hy_ba_avg = hy_ba / cnt as f64;

        let sz_d = (hy_sz_avg - bl_sz_avg) / bl_sz_avg * 100.0;
        let ba_d = (hy_ba_avg - bl_ba_avg) / bl_ba_avg * 100.0;
        let p = if sz_d < 0.0 && ba_d < 0.0 { " ★" } else { "" };

        println!(
            "{:>6.0} {:>10.0} {:>10.3} {:>10.0} {:>10.3} {:>+10.1}% {:>+10.1}%{}",
            base_q, bl_sz_avg, bl_ba_avg, hy_sz_avg, hy_ba_avg, sz_d, ba_d, p
        );
    }
}
