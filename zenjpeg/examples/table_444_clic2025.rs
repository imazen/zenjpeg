//! Butteraugli tables for 4:4:4 jpegli with/without trellis Q90+ on CLIC2025

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
    let clic_path = "/home/lilith/work/codec-eval/codec-corpus/clic2025/final-test";

    let images: Vec<_> = std::fs::read_dir(clic_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "png").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let loaded: Vec<_> = images
        .iter()
        .filter_map(|p| ImageData::from_path(p))
        .collect();

    println!(
        "=== 4:4:4 Jpegli vs Trellis Q90-99 on CLIC2025 ({} images) ===\n",
        loaded.len()
    );

    println!("| Q | Jpegli Size | Jpegli BA | Trellis Size | Trellis BA | Size Δ% | BA Δ% |");
    println!("|---|-------------|-----------|--------------|------------|---------|-------|");

    for q in 90..=99 {
        let q = q as f32;

        let jpegli_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::None)
            .optimization(OptimizationPreset::JpegliProgressive);
        let trellis_cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::None)
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

        println!(
            "| {:>2} | {:>11.0} | {:>9.3} | {:>12.0} | {:>10.3} | {:>+6.1}% | {:>+5.1}% |",
            q as i32, jsz_avg, jba_avg, tsz_avg, tba_avg, sz_d, ba_d
        );
    }

    println!("\n(Negative Size Δ = trellis smaller, Positive BA Δ = trellis worse quality)");
}
