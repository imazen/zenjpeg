//! Pareto benchmark comparing zenjpeg, mozjpeg-oxide, and jpegli.
//!
//! Sweeps quality levels on test images and generates rate-distortion curves.
//!
//! Usage:
//!   cargo run --release --example pareto_benchmark
//!   cargo run --release --example pareto_benchmark -- --xyb-fair
//!
//! Options:
//!   --xyb-fair  Use XYB roundtrip for fair jpegli comparison.
//!               This isolates compression error from color space conversion error
//!               for codecs that operate in XYB color space internally.
//!
//! Outputs:
//!   - comparison_outputs/pareto_results.csv
//!   - comparison_outputs/pareto_results.json

use codec_eval::{
    EvalConfig, EvalSession, ImageData, MetricConfig, ParetoFront, RDPoint, ViewingCondition,
};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let use_xyb_fair = args.iter().any(|a| a == "--xyb-fair");

    let output_dir = PathBuf::from("comparison_outputs");
    fs::create_dir_all(&output_dir)?;

    // Configure evaluation
    // Use XYB roundtrip when --xyb-fair is specified for fair jpegli comparison.
    // This isolates compression error from color space conversion error for XYB codecs.
    let metrics = if use_xyb_fair {
        println!("Using XYB roundtrip for fair comparison (--xyb-fair)");
        MetricConfig::perceptual_xyb()
    } else {
        MetricConfig::perceptual()
    };

    let config = EvalConfig::builder()
        .report_dir(&output_dir)
        .viewing(ViewingCondition::desktop())
        .metrics(metrics)
        .quality_levels(vec![
            30.0, 40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
        ])
        .build();

    let mut session = EvalSession::new(config);

    // Register zenjpeg encoder
    session.add_codec_with_decode(
        "zenjpeg",
        env!("CARGO_PKG_VERSION"),
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width();
            let height = image.height();
            let quality = request.quality as u8;

            let encoder =
                zenjpeg::Encoder::new().quality(zenjpeg::Quality::Standard(quality));
            encoder.encode_rgb(&rgb, width, height).map_err(|e| {
                codec_eval::Error::Codec {
                    codec: "zenjpeg".to_string(),
                    message: e.to_string(),
                }
            })
        }),
        Box::new(|data| {
            let mut decoder = jpeg_decoder::Decoder::new(data);
            let pixels = decoder.decode().map_err(|e| codec_eval::Error::Codec {
                codec: "jpeg-decoder".to_string(),
                message: e.to_string(),
            })?;
            let info = decoder.info().unwrap();
            Ok(ImageData::RgbSlice {
                data: pixels,
                width: info.width as usize,
                height: info.height as usize,
            })
        }),
    );

    // Register mozjpeg-oxide encoder
    session.add_codec_with_decode(
        "mozjpeg-oxide",
        "0.1.0",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as u8;

            let encoder = mozjpeg_oxide::Encoder::new()
                .quality(quality)
                .subsampling(mozjpeg_oxide::Subsampling::S444);
            encoder.encode_rgb(&rgb, width, height).map_err(|e| {
                codec_eval::Error::Codec {
                    codec: "mozjpeg-oxide".to_string(),
                    message: e.to_string(),
                }
            })
        }),
        Box::new(|data| {
            let mut decoder = jpeg_decoder::Decoder::new(data);
            let pixels = decoder.decode().map_err(|e| codec_eval::Error::Codec {
                codec: "jpeg-decoder".to_string(),
                message: e.to_string(),
            })?;
            let info = decoder.info().unwrap();
            Ok(ImageData::RgbSlice {
                data: pixels,
                width: info.width as usize,
                height: info.height as usize,
            })
        }),
    );

    // Register jpegli encoder
    session.add_codec_with_decode(
        "jpegli",
        "0.1.0",
        Box::new(|image, request| {
            let rgb = image.to_rgb8_vec();
            let width = image.width() as u32;
            let height = image.height() as u32;
            let quality = request.quality as f32;

            let encoder = jpegli::Encoder::new()
                .width(width)
                .height(height)
                .pixel_format(jpegli::PixelFormat::Rgb)
                .quality(jpegli::Quality::Traditional(quality));
            encoder.encode(&rgb).map_err(|e| codec_eval::Error::Codec {
                codec: "jpegli".to_string(),
                message: e.to_string(),
            })
        }),
        Box::new(|data| {
            let mut decoder = jpeg_decoder::Decoder::new(data);
            let pixels = decoder.decode().map_err(|e| codec_eval::Error::Codec {
                codec: "jpeg-decoder".to_string(),
                message: e.to_string(),
            })?;
            let info = decoder.info().unwrap();
            Ok(ImageData::RgbSlice {
                data: pixels,
                width: info.width as usize,
                height: info.height as usize,
            })
        }),
    );

    println!("Registered {} codecs", session.codec_count());

    // Find test images
    let corpus_dir = PathBuf::from("../codec-eval/codec-corpus/kodak");
    let test_images: Vec<PathBuf> = if corpus_dir.exists() {
        fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "png"))
            .take(6) // Use first 6 images for reasonably comprehensive benchmark
            .collect()
    } else {
        println!("Kodak corpus not found at {:?}, using synthetic image", corpus_dir);
        vec![]
    };

    // Collect all RD points for Pareto analysis
    let mut all_points: Vec<RDPoint> = Vec::new();

    if test_images.is_empty() {
        // Create synthetic test image
        println!("Using synthetic 256x256 gradient image");
        let (width, height) = (256, 256);
        let mut pixels = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.push((x * 255 / width) as u8);
                pixels.push((y * 255 / height) as u8);
                pixels.push(128u8);
            }
        }
        let image = ImageData::RgbSlice {
            data: pixels,
            width,
            height,
        };

        let report = session.evaluate_image("synthetic", image)?;
        println!(
            "\n=== {} ({}x{}) ===",
            report.name, report.width, report.height
        );

        for result in &report.results {
            let dssim = result.metrics.dssim.unwrap_or(0.0);
            let ssim2 = result.metrics.ssimulacra2.unwrap_or(0.0);
            let bfly = result.metrics.butteraugli.unwrap_or(0.0);
            println!(
                "{:15} Q{:3}: {:6} bytes, {:5.3} bpp, DSSIM={:.6}, SSIM2={:.2}, Bfly={:.3}",
                result.codec_id,
                result.quality as u8,
                result.file_size,
                result.bits_per_pixel,
                dssim,
                ssim2,
                bfly
            );

            // Use SSIMULACRA2 for Pareto (higher is better)
            if let Some(ssim2) = result.metrics.ssimulacra2 {
                all_points.push(RDPoint {
                    codec: result.codec_id.clone(),
                    quality_setting: result.quality,
                    bpp: result.bits_per_pixel,
                    quality: ssim2,
                    encode_time_ms: Some(result.encode_time.as_millis() as f64),
                    image: Some(report.name.clone()),
                });
            }
        }

        session.write_image_report(&report)?;
    } else {
        for img_path in &test_images {
            let img_name = img_path.file_stem().unwrap().to_string_lossy();
            println!("\nProcessing: {}", img_name);

            // Load PNG
            let decoder = png::Decoder::new(fs::File::open(img_path)?);
            let mut reader = decoder.read_info()?;
            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf)?;
            let bytes = &buf[..info.buffer_size()];

            // Convert to RGB if needed
            let rgb_data = match info.color_type {
                png::ColorType::Rgb => bytes.to_vec(),
                png::ColorType::Rgba => bytes
                    .chunks_exact(4)
                    .flat_map(|c| [c[0], c[1], c[2]])
                    .collect(),
                _ => {
                    println!("  Skipping unsupported color type: {:?}", info.color_type);
                    continue;
                }
            };

            let image = ImageData::RgbSlice {
                data: rgb_data,
                width: info.width as usize,
                height: info.height as usize,
            };

            let report = session.evaluate_image(&img_name, image)?;
            println!(
                "=== {} ({}x{}) ===",
                report.name, report.width, report.height
            );

            for result in &report.results {
                let dssim = result.metrics.dssim.unwrap_or(0.0);
                let ssim2 = result.metrics.ssimulacra2.unwrap_or(0.0);
                let bfly = result.metrics.butteraugli.unwrap_or(0.0);
                println!(
                    "{:15} Q{:3}: {:6} bytes, {:5.3} bpp, DSSIM={:.6}, SSIM2={:.2}, Bfly={:.3}",
                    result.codec_id,
                    result.quality as u8,
                    result.file_size,
                    result.bits_per_pixel,
                    dssim,
                    ssim2,
                    bfly
                );

                // Use SSIMULACRA2 for Pareto (higher is better)
                if let Some(ssim2) = result.metrics.ssimulacra2 {
                    all_points.push(RDPoint {
                        codec: result.codec_id.clone(),
                        quality_setting: result.quality,
                        bpp: result.bits_per_pixel,
                        quality: ssim2,
                        encode_time_ms: Some(result.encode_time.as_millis() as f64),
                        image: Some(report.name.clone()),
                    });
                }
            }

            session.write_image_report(&report)?;
        }
    }

    // Compute overall Pareto front
    println!("\n=== Pareto Front Analysis ===");
    let pareto = ParetoFront::compute(&all_points);
    println!("Total points: {}", all_points.len());
    println!("Pareto-optimal points: {}", pareto.len());

    println!("\nPareto front (by BPP):");
    for point in &pareto.points {
        println!(
            "  {} Q{:.0}: {:.3} bpp @ SSIM2={:.2}",
            point.codec, point.quality_setting, point.bpp, point.quality
        );
    }

    // Codecs on the Pareto front
    let front_codecs = pareto.codecs();
    println!("\nCodecs appearing on Pareto front: {:?}", front_codecs);

    // Per-codec Pareto fronts
    let per_codec = ParetoFront::per_codec(&all_points);
    println!("\nPer-codec efficiency:");
    for (codec, front) in &per_codec {
        if let Some(best_q80) = front.best_at_quality(80.0) {
            println!(
                "  {} best at SSIM2>=80: {:.3} bpp (Q{:.0})",
                codec, best_q80.bpp, best_q80.quality_setting
            );
        }
    }

    // Write Pareto data
    let pareto_json = serde_json::to_string_pretty(&pareto)?;
    fs::write(output_dir.join("pareto_front.json"), pareto_json)?;

    // Write all points as CSV
    let mut csv = String::from("codec,quality,bpp,ssimulacra2,image\n");
    for p in &all_points {
        csv.push_str(&format!(
            "{},{:.0},{:.4},{:.2},{}\n",
            p.codec,
            p.quality_setting,
            p.bpp,
            p.quality,
            p.image.as_deref().unwrap_or("")
        ));
    }
    fs::write(output_dir.join("all_points.csv"), csv)?;

    println!("\nResults written to {:?}", output_dir);
    Ok(())
}
