//! Benchmark hybrid trellis vs standalone trellis for quality/size trade-off.
//!
//! Tests whether AQ-coupled trellis provides better rate-distortion than
//! standalone trellis or no-trellis jpegli mode.
//!
//! Usage:
//!   cargo run --release --example hybrid_trellis_benchmark
//!   cargo run --release --example hybrid_trellis_benchmark -- /path/to/image.png

use std::env;
use std::path::Path;
use zenjpeg::encode::{ExpertConfig, OptimizationPreset, PixelLayout};
use zenjpeg_bench_utils::{decode_jpeg_to_rgb, ImageData, QualityMetrics, RgbImage};

fn main() {
    // Find test image
    let args: Vec<String> = env::args().collect();
    let image_path = if args.len() > 1 {
        args[1].clone()
    } else {
        // Try default test image
        let candidates = [
            "internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png",
            "internal/jpegli-cpp/testdata/jxl/flower/flower.png",
            "../codec-eval/codec-corpus/qoi-benchmark/screenshot_web/apple.com.png",
        ];
        candidates
            .iter()
            .find(|p| Path::new(p).exists())
            .map(|s| s.to_string())
            .expect("No test image found. Pass path as argument.")
    };

    println!("Loading: {image_path}");
    let img = ImageData::from_path(Path::new(&image_path)).expect("Failed to load image");
    println!(
        "Image: {}x{} ({} pixels)",
        img.width,
        img.height,
        img.width * img.height
    );

    let qualities = [50, 70, 85, 90, 95];
    let couplings = [0.0, 0.5, 1.0, 2.0, 4.0];

    println!("\n=== Rate-Distortion Comparison ===");
    println!("   Q   Mode        Coupling   Size     BPP    DSSIM       SSIM2   Butteraugli");
    println!("  ---  ----------  --------  ------  ------  --------  --------  -----------");

    for &quality in &qualities {
        // Baseline: jpegli (no trellis)
        let (jpegli_bytes, jpegli_dssim, jpegli_ssim2, jpegli_butter) =
            encode_and_measure(&img, quality, Mode::Jpegli);
        let jpegli_bpp = bpp(jpegli_bytes, img.width, img.height);
        println!(
            "  {:3}  jpegli      -         {:6}  {:6.3}  {:8.5}  {:8.2}  {:11.3}",
            quality, jpegli_bytes, jpegli_bpp, jpegli_dssim, jpegli_ssim2, jpegli_butter
        );

        // Standalone trellis (mozjpeg-style)
        let (standalone_bytes, standalone_dssim, standalone_ssim2, standalone_butter) =
            encode_and_measure(&img, quality, Mode::Standalone);
        let standalone_bpp = bpp(standalone_bytes, img.width, img.height);
        let standalone_vs_jpegli = (standalone_bytes as f64 / jpegli_bytes as f64 - 1.0) * 100.0;
        println!(
            "  {:3}  standalone  0.0       {:6}  {:6.3}  {:8.5}  {:8.2}  {:11.3}  ({:+.1}% vs jpegli)",
            quality, standalone_bytes, standalone_bpp, standalone_dssim, standalone_ssim2, standalone_butter,
            standalone_vs_jpegli
        );

        // Hybrid with different couplings
        for &coupling in &couplings[1..] {
            // Skip 0.0, same as standalone
            let (hybrid_bytes, hybrid_dssim, hybrid_ssim2, hybrid_butter) =
                encode_and_measure(&img, quality, Mode::Hybrid(coupling));
            let hybrid_bpp = bpp(hybrid_bytes, img.width, img.height);
            let vs_standalone = (hybrid_bytes as f64 / standalone_bytes as f64 - 1.0) * 100.0;
            let dssim_vs = (hybrid_dssim / standalone_dssim - 1.0) * 100.0;
            println!(
                "  {:3}  hybrid      {:4.1}      {:6}  {:6.3}  {:8.5}  {:8.2}  {:11.3}  ({:+.1}% size, {:+.1}% dssim vs standalone)",
                quality, coupling, hybrid_bytes, hybrid_bpp, hybrid_dssim, hybrid_ssim2, hybrid_butter,
                vs_standalone, dssim_vs
            );
        }
        println!();
    }

    println!("\nInterpretation:");
    println!("- DSSIM: lower = better (0 = identical)");
    println!("- SSIM2: higher = better (100 = identical)");
    println!("- Butteraugli: lower = better (<1.0 is good)");
    println!("- If hybrid has larger files AND worse quality, it's not useful.");
    println!("- If hybrid has larger files but BETTER quality (lower DSSIM), might be worthwhile.");
}

enum Mode {
    Jpegli,
    Standalone,
    Hybrid(f32),
}

fn encode_and_measure(img: &ImageData, quality: u8, mode: Mode) -> (usize, f64, f64, f64) {
    // Create config based on mode
    let config = match mode {
        Mode::Jpegli => {
            // Jpegli preset: no trellis, jpegli tables
            ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality as i32)
                .to_encoder_config(zenjpeg::encode::ColorMode::YCbCr {
                    subsampling: zenjpeg::encode::ChromaSubsampling::Quarter,
                })
        }
        Mode::Standalone => {
            // Mozjpeg preset: trellis enabled, coupling=0
            ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, quality as i32)
                .to_encoder_config(zenjpeg::encode::ColorMode::YCbCr {
                    subsampling: zenjpeg::encode::ChromaSubsampling::Quarter,
                })
        }
        Mode::Hybrid(coupling) => {
            // Hybrid: jpegli tables + trellis + AQ coupling
            let mut expert =
                ExpertConfig::from_preset(OptimizationPreset::JpegliBaseline, quality as i32);
            expert.trellis_enabled = true;
            expert.aq_trellis_coupling = coupling;
            expert.to_encoder_config(zenjpeg::encode::ColorMode::YCbCr {
                subsampling: zenjpeg::encode::ChromaSubsampling::Quarter,
            })
        }
    };

    // Encode
    let jpeg_bytes = encode_image(&config, img);

    // Decode and measure quality
    let decoded: RgbImage = decode_jpeg_to_rgb(&jpeg_bytes).expect("decode failed");
    let (dssim, ssim2, butteraugli) = measure_quality(img, &decoded);

    (jpeg_bytes.len(), dssim, ssim2, butteraugli)
}

fn encode_image(config: &zenjpeg::encode::EncoderConfig, img: &ImageData) -> Vec<u8> {
    let mut encoder = config
        .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
        .expect("encoder create");
    encoder
        .push_packed(&img.pixels, enough::Unstoppable)
        .expect("push");
    encoder.finish().expect("finish")
}

fn measure_quality(original: &ImageData, decoded: &RgbImage) -> (f64, f64, f64) {
    let orig = zenjpeg_bench_utils::bytes_to_rgb(&original.pixels, original.width, original.height);

    let dssim = QualityMetrics::dssim(orig.as_ref(), decoded.as_ref());
    let ssim2 = QualityMetrics::ssimulacra2(orig.as_ref(), decoded.as_ref());
    let butteraugli = QualityMetrics::butteraugli(orig.as_ref(), decoded.as_ref());

    (dssim, ssim2, butteraugli)
}

fn bpp(bytes: usize, width: usize, height: usize) -> f64 {
    (bytes as f64 * 8.0) / (width * height) as f64
}
