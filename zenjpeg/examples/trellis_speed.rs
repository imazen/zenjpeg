//! Speed cost of trellis

use enough::Unstoppable;
use std::time::Instant;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, OptimizationPreset, PixelLayout};
use zenjpeg_bench_utils::SyntheticPattern;

fn main() {
    let img = SyntheticPattern::PhotoLike.generate(1024, 1024);
    let pixels: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();

    println!("=== Trellis Speed Cost (1024x1024) ===\n");

    for (name, preset) in [
        ("JpegliProgressive", OptimizationPreset::JpegliProgressive),
        (
            "HybridProgressive (trellis)",
            OptimizationPreset::HybridProgressive,
        ),
    ] {
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).optimization(preset);

        // Warmup
        for _ in 0..3 {
            let mut e = config
                .encode_from_bytes(1024, 1024, PixelLayout::Rgb8Srgb)
                .unwrap();
            e.push_packed(&pixels, Unstoppable).unwrap();
            let _ = e.finish();
        }

        // Measure
        let start = Instant::now();
        let iters = 10;
        for _ in 0..iters {
            let mut e = config
                .encode_from_bytes(1024, 1024, PixelLayout::Rgb8Srgb)
                .unwrap();
            e.push_packed(&pixels, Unstoppable).unwrap();
            let _ = e.finish();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_secs_f64() / iters as f64 * 1000.0;

        println!("{:>30}: {:.1} ms/encode", name, per_iter);
    }
}
