//! Compare glassa low-BPP tables vs jpegli and mozjpeg at extreme compression.
//!
//! Glassa tables are optimized for QUALITY at a given BPP, not for minimum file size.
//! This example compares quality (SSIMULACRA2) at matched file sizes.
//!
//! Run: cargo run --release -p zenjpeg --example glassa_lowbpp_compare

use zenjpeg::encode::{
    ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout, QuantTableConfig,
};
use zenjpeg_bench_utils::{ImageData, QualityMetrics, decode_jpeg_to_rgb};

fn encode(config: &EncoderConfig, img: &ImageData) -> Option<Vec<u8>> {
    let mut enc = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .ok()?;
    enc.push_packed(&img.pixels, enough::Unstoppable).ok()?;
    enc.finish().ok()
}

fn compute_ssim2(img: &ImageData, jpeg: &[u8]) -> Option<f64> {
    let original = zenjpeg_bench_utils::bytes_to_rgb(&img.pixels, img.width, img.height);
    let decoded = decode_jpeg_to_rgb(jpeg).ok()?;
    Some(QualityMetrics::ssimulacra2(original.as_ref(), decoded.as_ref()))
}

fn main() {
    // Find test images
    let corpus_path = std::env::var("CORPUS_PATH")
        .unwrap_or_else(|_| "/home/lilith/work/codec-eval/codec-corpus/CID22/CID22-512/training".to_string());

    let images: Vec<_> = std::fs::read_dir(&corpus_path)
        .expect("Cannot find corpus; set CORPUS_PATH env var")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == "png" || ext == "jpg")
        })
        .filter_map(|e| ImageData::from_path(&e.path()))
        .take(5)
        .collect();

    if images.is_empty() {
        eprintln!("No images found in {corpus_path}");
        return;
    }

    println!("=== Glassa Low-BPP Quality Comparison ({} images, 4:2:0) ===\n", images.len());
    println!("Glassa tables are optimized for QUALITY at a given file size, not minimum size.");
    println!("This compares SSIMULACRA2 (higher=better) at similar file sizes.\n");

    // Target BPPs that match glassa table anchors
    let targets = [
        (5u8, "~0.21 BPP"),
        (10, "~0.27 BPP"),
        (15, "~0.26 BPP"),
        (20, "~0.58 BPP"),
        (25, "~0.48 BPP"),
    ];

    println!("| Glassa Q | Glassa SSIM2 | Glassa Size | Mozjpeg Q (matched) | Mozjpeg SSIM2 | Mozjpeg Size | SSIM2 Δ |");
    println!("|----------|--------------|-------------|---------------------|---------------|--------------|---------|");

    for (glassa_q, _bpp_label) in targets {
        let mut glassa_ssim_sum = 0.0f64;
        let mut glassa_size_sum = 0usize;
        let mut moz_ssim_sum = 0.0f64;
        let mut moz_size_sum = 0usize;
        let mut moz_q_sum = 0u32;
        let mut count = 0usize;

        for img in &images {
            // Encode with glassa
            let glassa_cfg = EncoderConfig::ycbcr(glassa_q as f32, ChromaSubsampling::Quarter)
                .quant_table_config(QuantTableConfig::GlassaLowBpp(glassa_q))
                .progressive(false);

            let Some(glassa_jpg) = encode(&glassa_cfg, img) else {
                continue;
            };
            let Some(glassa_ssim) = compute_ssim2(img, &glassa_jpg) else {
                continue;
            };

            // Find mozjpeg Q that produces similar file size
            let target_size = glassa_jpg.len();
            let mut best_moz_q = 1u8;
            let mut best_diff = usize::MAX;

            for q in 1..=50 {
                let moz_cfg = EncoderConfig::ycbcr(q as f32, ChromaSubsampling::Quarter)
                    .optimization(OptimizationPreset::MozjpegBaseline);
                if let Some(moz_jpg) = encode(&moz_cfg, img) {
                    let diff = (moz_jpg.len() as isize - target_size as isize).unsigned_abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_moz_q = q;
                    }
                }
            }

            // Encode at matched Q
            let moz_cfg = EncoderConfig::ycbcr(best_moz_q as f32, ChromaSubsampling::Quarter)
                .optimization(OptimizationPreset::MozjpegBaseline);
            let Some(moz_jpg) = encode(&moz_cfg, img) else {
                continue;
            };
            let Some(moz_ssim) = compute_ssim2(img, &moz_jpg) else {
                continue;
            };

            glassa_ssim_sum += glassa_ssim;
            glassa_size_sum += glassa_jpg.len();
            moz_ssim_sum += moz_ssim;
            moz_size_sum += moz_jpg.len();
            moz_q_sum += best_moz_q as u32;
            count += 1;
        }

        if count == 0 {
            continue;
        }

        let glassa_ssim_avg = glassa_ssim_sum / count as f64;
        let glassa_size_avg = glassa_size_sum / count;
        let moz_ssim_avg = moz_ssim_sum / count as f64;
        let moz_size_avg = moz_size_sum / count;
        let moz_q_avg = moz_q_sum / count as u32;
        let ssim_delta = glassa_ssim_avg - moz_ssim_avg;

        println!(
            "|    Q{:<4} |       {:>5.1} |      {:>6} |              Q{:<5} |        {:>5.1} |       {:>6} |  {:>+5.1} |",
            glassa_q, glassa_ssim_avg, glassa_size_avg,
            moz_q_avg, moz_ssim_avg, moz_size_avg, ssim_delta
        );
    }

    println!("\nPositive SSIM2 Δ = glassa has better quality at same size.");
    println!("SSIM2: higher is better (100 = identical to original).");
}
