//! Does trellis help with 4:4:4 subsampling?

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

    println!("=== Trellis with 4:4:4 (no chroma subsampling) ===");
    println!("Testing {} images\n", loaded.len());

    println!(
        "{:>4} {:>12} {:>8} {:>12} {:>8} {:>10} {:>10} {:>8}",
        "Q", "Jpegli sz", "Jpegli BA", "Trellis sz", "Trellis BA", "ΔSize%", "ΔBA%", "Pareto?"
    );
    println!("{}", "-".repeat(90));

    for q in [85, 90, 95] {
        let q = q as f32;

        let jpegli_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::None) // 4:4:4
            .optimization(OptimizationPreset::JpegliProgressive);
        let trellis_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::None) // 4:4:4
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

        let pareto = if sz_d < -0.5 && ba_d < 1.0 {
            "★ WIN"
        } else {
            "~same"
        };

        println!(
            "{:>4.0} {:>12.0} {:>8.3} {:>12.0} {:>8.3} {:>+10.1}% {:>+10.1}% {:>8}",
            q, jsz_avg, jba_avg, tsz_avg, tba_avg, sz_d, ba_d, pareto
        );
    }

    // Compare 4:2:0 vs 4:4:4 at same quality
    println!("\n=== 4:2:0 vs 4:4:4 Comparison ===\n");
    println!(
        "{:>15} {:>12} {:>8} {:>12} {:>8}",
        "Config @ Q90", "Size", "BA", "vs 420 sz%", "vs 420 BA%"
    );
    println!("{}", "-".repeat(60));

    // 4:2:0 baseline
    let cfg_420 = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .optimization(OptimizationPreset::HybridProgressive);
    let (mut sz, mut ba_v, mut n) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(&cfg_420, img) {
            if let Some(b) = ba(img, &j) {
                sz += j.len();
                ba_v += b;
                n += 1;
            }
        }
    }
    let (base_sz, base_ba) = (sz as f64 / n as f64, ba_v / n as f64);
    println!(
        "{:>15} {:>12.0} {:>8.3} {:>12} {:>8}",
        "Trellis 4:2:0", base_sz, base_ba, "-", "-"
    );

    // 4:4:4
    let cfg_444 = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None)
        .optimization(OptimizationPreset::HybridProgressive);
    let (mut sz, mut ba_v, mut n) = (0usize, 0.0, 0);
    for img in &loaded {
        if let Some(j) = encode(&cfg_444, img) {
            if let Some(b) = ba(img, &j) {
                sz += j.len();
                ba_v += b;
                n += 1;
            }
        }
    }
    let sz_d = (sz as f64 / n as f64 - base_sz) / base_sz * 100.0;
    let ba_d = (ba_v / n as f64 - base_ba) / base_ba * 100.0;
    println!(
        "{:>15} {:>12.0} {:>8.3} {:>+12.1}% {:>+8.1}%",
        "Trellis 4:4:4",
        sz as f64 / n as f64,
        ba_v / n as f64,
        sz_d,
        ba_d
    );
}
