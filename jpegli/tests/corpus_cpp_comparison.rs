//! Corpus comparison test using codec-eval.
//!
//! Compares Rust jpegli against C++ jpegli on a large corpus.
//! Run with: cargo test --test corpus_cpp_comparison -- --ignored --nocapture

use codec_eval::{EvalConfig, EvalSession, ImageData, ViewingCondition};
use jpegli::quant::Quality;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Path to C++ cjpegli binary
const CJPEGLI_PATH: &str = "/home/lilith/work/jpegli/build/tools/cjpegli";

/// Path to CID22-512 corpus
const CORPUS_PATH: &str = "/mnt/v/work/corpus/CID22-512";

/// Generic JPEG decoder using jpeg-decoder crate
fn decode_jpeg(data: &[u8]) -> codec_eval::Result<ImageData> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    let pixels = decoder.decode().map_err(|e| codec_eval::Error::Codec {
        codec: "jpeg-decoder".to_string(),
        message: format!("{}", e),
    })?;

    let info = decoder.info().ok_or_else(|| codec_eval::Error::Codec {
        codec: "jpeg-decoder".to_string(),
        message: "No image info".to_string(),
    })?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if needed
    let rgb_data = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => {
            let mut rgb = Vec::with_capacity(width * height * 3);
            for &g in &pixels {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        _ => {
            return Err(codec_eval::Error::Codec {
                codec: "jpeg-decoder".to_string(),
                message: format!("Unsupported pixel format: {:?}", info.pixel_format),
            });
        }
    };

    Ok(ImageData::RgbSlice {
        data: rgb_data,
        width,
        height,
    })
}

/// Register Rust jpegli encoder with decoder
fn register_rust_jpegli(session: &mut EvalSession) {
    session.add_codec_with_decode(
        "jpegli-rs",
        env!("CARGO_PKG_VERSION"),
        // Encoder
        Box::new(|image, request| {
            let width = image.width();
            let height = image.height();
            let rgb_data = image.to_rgb8_vec();

            let quality = request.quality as f32;

            let encoder = jpegli::Encoder::new()
                .width(width as u32)
                .height(height as u32)
                .pixel_format(jpegli::PixelFormat::Rgb)
                .quality(Quality::from_quality(quality));

            let encoded = encoder
                .encode(&rgb_data)
                .map_err(|e| codec_eval::Error::Codec {
                    codec: "jpegli-rs".to_string(),
                    message: format!("{}", e),
                })?;

            Ok(encoded)
        }),
        // Decoder
        Box::new(|data| decode_jpeg(data)),
    );
}

/// Register C++ jpegli encoder with decoder
fn register_cpp_jpegli(session: &mut EvalSession) {
    session.add_codec_with_decode(
        "jpegli-cpp",
        "latest",
        // Encoder
        Box::new(|image, request| {
            let width = image.width();
            let height = image.height();
            let rgb_data = image.to_rgb8_vec();

            // Write PPM to temp file
            let ppm_path = format!("/tmp/cpp_eval_{}.ppm", std::process::id());
            let jpg_path = format!("/tmp/cpp_eval_{}.jpg", std::process::id());

            // Write PPM header and data
            let mut ppm_data = format!("P6\n{} {}\n255\n", width, height).into_bytes();
            ppm_data.extend_from_slice(&rgb_data);
            std::fs::write(&ppm_path, &ppm_data)?;

            // Run cjpegli with matching settings
            let quality = request.quality as u32;
            let output = Command::new(CJPEGLI_PATH)
                .args([
                    "--chroma_subsampling=444",
                    "-p",
                    "0",
                    "--fixed_code",
                    &ppm_path,
                    &jpg_path,
                    "-q",
                    &quality.to_string(),
                ])
                .output()?;

            if !output.status.success() {
                return Err(codec_eval::Error::Codec {
                    codec: "jpegli-cpp".to_string(),
                    message: String::from_utf8_lossy(&output.stderr).to_string(),
                });
            }

            let encoded = std::fs::read(&jpg_path)?;

            // Cleanup
            let _ = std::fs::remove_file(&ppm_path);
            let _ = std::fs::remove_file(&jpg_path);

            Ok(encoded)
        }),
        // Decoder (use same generic JPEG decoder)
        Box::new(|data| decode_jpeg(data)),
    );
}

/// Load PNG image as ImageData
fn load_png(path: &Path) -> Result<ImageData, codec_eval::Error> {
    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().map_err(|e| codec_eval::Error::Codec {
        codec: "png".to_string(),
        message: format!("PNG decode error: {}", e),
    })?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| codec_eval::Error::Codec {
            codec: "png".to_string(),
            message: format!("PNG frame error: {}", e),
        })?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if needed
    let rgb_data = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => {
            let mut rgb = Vec::with_capacity(width * height * 3);
            for chunk in buf[..width * height * 4].chunks_exact(4) {
                rgb.push(chunk[0]);
                rgb.push(chunk[1]);
                rgb.push(chunk[2]);
            }
            rgb
        }
        png::ColorType::Grayscale => {
            let mut rgb = Vec::with_capacity(width * height * 3);
            for &g in &buf[..width * height] {
                rgb.push(g);
                rgb.push(g);
                rgb.push(g);
            }
            rgb
        }
        _ => {
            return Err(codec_eval::Error::UnsupportedFormat(
                "Unsupported PNG color type".into(),
            ))
        }
    };

    Ok(ImageData::RgbSlice {
        data: rgb_data,
        width,
        height,
    })
}

#[test]
#[ignore = "requires C++ cjpegli build and CID22-512 corpus"]
fn test_corpus_comparison() {
    // Check prerequisites
    if !Path::new(CJPEGLI_PATH).exists() {
        eprintln!("Skipping: C++ cjpegli not available at {}", CJPEGLI_PATH);
        return;
    }

    let corpus_path = Path::new(CORPUS_PATH);
    if !corpus_path.exists() {
        eprintln!("Skipping: Corpus not available at {}", CORPUS_PATH);
        return;
    }

    // Count available images
    let images: Vec<_> = std::fs::read_dir(corpus_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
        .collect();

    let max_images = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let images: Vec<_> = images.into_iter().take(max_images).collect();

    println!("\n=== CORPUS COMPARISON: Rust vs C++ jpegli ===\n");
    println!("Images to test: {}", images.len());
    println!("Quality levels: 60, 80, 90");
    println!("Settings: 4:4:4, AQ enabled, sequential, fixed Huffman\n");

    // Configure evaluation
    let config = EvalConfig::builder()
        .report_dir(PathBuf::from("/tmp/jpegli-corpus-comparison"))
        .viewing(ViewingCondition::desktop())
        .quality_levels(vec![60.0, 80.0, 90.0])
        .build();

    let mut session = EvalSession::new(config);

    // Register codecs
    register_rust_jpegli(&mut session);
    register_cpp_jpegli(&mut session);

    // Track results
    let mut total_rust_bytes = 0u64;
    let mut total_cpp_bytes = 0u64;
    let mut total_rust_dssim = 0.0f64;
    let mut total_cpp_dssim = 0.0f64;
    let mut count = 0u64;

    // Evaluate each image
    for entry in &images {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy();

        let image_data = match load_png(&path) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Skipping {}: {}", name, e);
                continue;
            }
        };

        // Evaluate at Q90 for summary stats
        match session.evaluate_image(&name, image_data.clone()) {
            Ok(report) => {
                // Find results for Q90
                for result in &report.results {
                    if (result.quality - 90.0).abs() < 0.1 {
                        if result.codec_id == "jpegli-rs" {
                            total_rust_bytes += result.file_size as u64;
                            if let Some(dssim) = result.metrics.dssim {
                                total_rust_dssim += dssim;
                            }
                        } else if result.codec_id == "jpegli-cpp" {
                            total_cpp_bytes += result.file_size as u64;
                            if let Some(dssim) = result.metrics.dssim {
                                total_cpp_dssim += dssim;
                            }
                        }
                    }
                }
                count += 1;
            }
            Err(e) => {
                eprintln!("Error evaluating {}: {}", name, e);
            }
        }

        // Progress indicator
        if count % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }

    println!("\n");

    // Summary
    if count > 0 {
        let avg_rust_bytes = total_rust_bytes as f64 / count as f64;
        let avg_cpp_bytes = total_cpp_bytes as f64 / count as f64;
        let avg_rust_dssim = total_rust_dssim / count as f64;
        let avg_cpp_dssim = total_cpp_dssim / count as f64;

        let size_diff_pct = (avg_rust_bytes - avg_cpp_bytes) / avg_cpp_bytes * 100.0;

        println!("=== SUMMARY ({} images at Q90) ===\n", count);
        println!("| Codec | Avg Size | Avg DSSIM |");
        println!("|-------|----------|-----------|");
        println!(
            "| jpegli-rs  | {:.0} bytes | {:.6} |",
            avg_rust_bytes, avg_rust_dssim
        );
        println!(
            "| jpegli-cpp | {:.0} bytes | {:.6} |",
            avg_cpp_bytes, avg_cpp_dssim
        );
        println!();
        println!(
            "Size difference: {:.1}% ({})",
            size_diff_pct,
            if size_diff_pct < 0.0 {
                "Rust smaller"
            } else {
                "C++ smaller"
            }
        );

        // Assert reasonable parity
        assert!(
            size_diff_pct.abs() < 15.0,
            "Size difference too large: {:.1}%",
            size_diff_pct
        );
    } else {
        panic!("No images were successfully evaluated");
    }
}

/// Quick test with just a few images
#[test]
#[ignore = "requires C++ cjpegli build and CID22-512 corpus"]
fn test_corpus_quick() {
    std::env::set_var("MAX_IMAGES", "10");
    test_corpus_comparison();
}
