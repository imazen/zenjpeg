//! Parity tests against C++ jpegli reference data.
//!
//! This test compares Rust jpegli output against pre-captured C++ reference data.
//! The reference data includes file sizes and quality metrics for the Kodak corpus
//! at various quality levels.
//!
//! Thresholds are intentionally tight to catch regressions.

use jpegli::decoder::Decoder;
use jpegli::decoder::PixelFormat;
use jpegli::encoder::{EncoderConfig, PixelLayout};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Reference data for a single image at a single quality level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPoint {
    pub quality: u8,
    pub file_size: usize,
    pub dssim: f64,
    pub ssimulacra2: f64,
    pub butteraugli: f64,
}

/// Reference data for a single image across all quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageReference {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub original_size: usize,
    pub points: Vec<QualityPoint>,
}

/// Complete reference dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CppReferenceData {
    pub version: String,
    pub cjpegli_version: String,
    pub generated_at: String,
    pub images: Vec<ImageReference>,
}

fn load_reference_data() -> Option<CppReferenceData> {
    let paths = [
        "testdata/cpp_reference_kodak.json",
        "jpegli/testdata/cpp_reference_kodak.json",
    ];

    for path in paths {
        if let Ok(data) = fs::read_to_string(path) {
            if let Ok(reference) = serde_json::from_str(&data) {
                return Some(reference);
            }
        }
    }
    None
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    let rgb = match info.color_type {
        png::ColorType::Rgb => buf,
        png::ColorType::Rgba => buf.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        png::ColorType::Grayscale => buf.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buf.chunks(2).flat_map(|c| [c[0], c[0], c[0]]).collect(),
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

fn compute_dssim(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;
    use rgb::RGBA;

    let attr = Dssim::new();

    let orig_rgba: Vec<RGBA<u8>> = orig
        .chunks(3)
        .map(|c| RGBA::new(c[0], c[1], c[2], 255))
        .collect();
    let comp_rgba: Vec<RGBA<u8>> = comp
        .chunks(3)
        .map(|c| RGBA::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp_img = attr.create_image_rgba(&comp_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, comp_img);
    dssim.into()
}

#[allow(dead_code)] // Available for quality metric comparisons when needed
fn compute_ssimulacra2(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

    let orig_rgb: Vec<[f32; 3]> = orig
        .chunks(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();
    let comp_rgb: Vec<[f32; 3]> = comp
        .chunks(3)
        .map(|c| {
            [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ]
        })
        .collect();

    let orig_frame = Rgb::new(
        orig_rgb,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let comp_frame = Rgb::new(
        comp_rgb,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_frame, comp_frame).unwrap_or(0.0)
}

#[allow(dead_code)] // Available for quality metric comparisons when needed
fn compute_butteraugli_score(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli::{compute_butteraugli, ButteraugliParams};

    let params = ButteraugliParams::default();
    match compute_butteraugli(orig, comp, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
}

/// Test that Rust jpegli produces similar file sizes to C++ jpegli.
/// Threshold: within 5% of C++ file size.
#[test]
#[ignore = "requires kodak corpus"]
fn test_file_size_parity() {
    let reference = match load_reference_data() {
        Some(r) => r,
        None => {
            eprintln!("Reference data not found, skipping test");
            return;
        }
    };

    let corpus_paths = [
        "../codec-eval/codec-corpus/kodak",
        "../codec-corpus/kodak",
        "codec-corpus/kodak",
    ];

    let corpus_dir = corpus_paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(Path::new);

    let Some(corpus_dir) = corpus_dir else {
        eprintln!("Kodak corpus not found, skipping test");
        return;
    };

    let mut total_cpp_size = 0usize;
    let mut total_rust_size = 0usize;
    let mut failures = Vec::new();

    for img_ref in &reference.images {
        let png_path = corpus_dir.join(format!("{}.png", img_ref.name));
        let Some((pixels, width, height)) = load_png(&png_path) else {
            eprintln!("Skipping {}: failed to load", img_ref.name);
            continue;
        };

        for point in &img_ref.points {
            let config = EncoderConfig::new().quality(point.quality as f32);
            let rust_jpeg = match config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb) {
                Ok(mut enc) => {
                    if let Err(e) = enc.push_packed(&pixels, enough::Unstoppable) {
                        failures.push(format!(
                            "{} Q{}: push failed: {:?}",
                            img_ref.name, point.quality, e
                        ));
                        continue;
                    }
                    match enc.finish() {
                        Ok(data) => data,
                        Err(e) => {
                            failures.push(format!(
                                "{} Q{}: finish failed: {:?}",
                                img_ref.name, point.quality, e
                            ));
                            continue;
                        }
                    }
                }
                Err(e) => {
                    failures.push(format!(
                        "{} Q{}: encode failed: {:?}",
                        img_ref.name, point.quality, e
                    ));
                    continue;
                }
            };

            let rust_size = rust_jpeg.len();
            let cpp_size = point.file_size;
            let diff_pct = ((rust_size as f64 - cpp_size as f64) / cpp_size as f64) * 100.0;

            total_cpp_size += cpp_size;
            total_rust_size += rust_size;

            // 5% threshold
            if diff_pct.abs() > 5.0 {
                failures.push(format!(
                    "{} Q{}: Rust {} bytes vs C++ {} bytes ({:+.1}%)",
                    img_ref.name, point.quality, rust_size, cpp_size, diff_pct
                ));
            }
        }
    }

    let overall_diff =
        ((total_rust_size as f64 - total_cpp_size as f64) / total_cpp_size as f64) * 100.0;
    eprintln!(
        "\nOverall: Rust {} bytes vs C++ {} bytes ({:+.2}%)",
        total_rust_size, total_cpp_size, overall_diff
    );

    if !failures.is_empty() {
        eprintln!("\nFailures ({}):", failures.len());
        for f in &failures {
            eprintln!("  {}", f);
        }
        panic!("{} file size parity failures", failures.len());
    }
}

/// Test that Rust jpegli produces similar quality (DSSIM) to C++ jpegli.
/// Threshold: DSSIM within 20% of C++ (relative).
#[test]
#[ignore = "requires kodak corpus"]
fn test_dssim_parity() {
    let reference = match load_reference_data() {
        Some(r) => r,
        None => {
            eprintln!("Reference data not found, skipping test");
            return;
        }
    };

    let corpus_paths = [
        "../codec-eval/codec-corpus/kodak",
        "../codec-corpus/kodak",
        "codec-corpus/kodak",
    ];

    let corpus_dir = corpus_paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(Path::new);

    let Some(corpus_dir) = corpus_dir else {
        eprintln!("Kodak corpus not found, skipping test");
        return;
    };

    let mut failures = Vec::new();

    for img_ref in &reference.images {
        let png_path = corpus_dir.join(format!("{}.png", img_ref.name));
        let Some((pixels, width, height)) = load_png(&png_path) else {
            continue;
        };

        for point in &img_ref.points {
            let config = EncoderConfig::new().quality(point.quality as f32);
            let rust_jpeg = match config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb) {
                Ok(mut enc) => {
                    if enc.push_packed(&pixels, enough::Unstoppable).is_err() {
                        continue;
                    }
                    match enc.finish() {
                        Ok(data) => data,
                        Err(_) => continue,
                    }
                }
                Err(_) => continue,
            };

            // Decode and compute DSSIM
            let decoder = Decoder::new().output_format(PixelFormat::Rgb);
            let decoded = match decoder.decode(&rust_jpeg) {
                Ok(img) => img,
                Err(_) => continue,
            };

            let rust_dssim = compute_dssim(&pixels, &decoded.data, width as usize, height as usize);
            let cpp_dssim = point.dssim;

            // 20% relative threshold (DSSIM values are small, so relative comparison)
            let diff_pct = if cpp_dssim > 0.0 {
                ((rust_dssim - cpp_dssim) / cpp_dssim) * 100.0
            } else {
                0.0
            };

            if diff_pct.abs() > 20.0 {
                failures.push(format!(
                    "{} Q{}: Rust DSSIM {:.6} vs C++ {:.6} ({:+.1}%)",
                    img_ref.name, point.quality, rust_dssim, cpp_dssim, diff_pct
                ));
            }
        }
    }

    if !failures.is_empty() {
        eprintln!("\nDSSIM Parity Failures ({}):", failures.len());
        for f in &failures {
            eprintln!("  {}", f);
        }
        panic!("{} DSSIM parity failures", failures.len());
    }
}

/// Quick sanity test with a single image.
#[test]
fn test_reference_data_loads() {
    if let Some(reference) = load_reference_data() {
        assert!(
            !reference.images.is_empty(),
            "Reference data should have images"
        );
        assert_eq!(reference.version, "1.0");

        // Check structure
        for img in &reference.images {
            assert!(img.width > 0);
            assert!(img.height > 0);
            assert!(!img.points.is_empty());

            for pt in &img.points {
                assert!(pt.file_size > 0);
                assert!(pt.dssim >= 0.0);
                assert!(pt.ssimulacra2 <= 100.0);
                assert!(pt.butteraugli >= 0.0);
            }
        }

        eprintln!(
            "Reference data: {} images, {} total data points",
            reference.images.len(),
            reference
                .images
                .iter()
                .map(|i| i.points.len())
                .sum::<usize>()
        );
    } else {
        eprintln!("Reference data not found - generate with:");
        eprintln!("  cargo run --example generate_cpp_reference");
    }
}
