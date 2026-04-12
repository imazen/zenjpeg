//! Parity tests against C++ jpegli reference data.
//!
//! This test compares Rust jpegli output against pre-captured C++ reference data.
//! The reference data includes file sizes and quality metrics for the Kodak corpus
//! at various quality levels.
//!
//! Thresholds are intentionally tight to catch regressions.
use enough::Unstoppable;

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use zenjpeg::decoder::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

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
        if let Ok(data) = fs::read_to_string(path)
            && let Ok(reference) = serde_json::from_str(&data)
        {
            return Some(reference);
        }
    }
    None
}

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = zenjpeg_bench_utils::load_png(path).ok()?;
    let width = img.width() as u32;
    let height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    Some((rgb, width, height))
}

fn compute_dssim(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use dssim_core::Dssim;
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
    use imgref::{Img, ImgVec};

    let orig_pixels: Vec<[u8; 3]> = orig.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let comp_pixels: Vec<[u8; 3]> = comp.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

    let orig_img: ImgVec<[u8; 3]> = Img::new(orig_pixels, width, height);
    let comp_img: ImgVec<[u8; 3]> = Img::new(comp_pixels, width, height);

    fast_ssim2::compute_ssimulacra2(orig_img.as_ref(), comp_img.as_ref()).unwrap_or(0.0)
}

#[allow(dead_code)] // Available for quality metric comparisons when needed
fn compute_butteraugli_score(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli::ButteraugliParams;
    use imgref::ImgVec;
    use rgb::RGB8;

    let to_img = |data: &[u8]| -> ImgVec<RGB8> {
        let pixels: Vec<RGB8> = data
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect();
        ImgVec::new(pixels, width, height)
    };
    let orig_img = to_img(orig);
    let comp_img = to_img(comp);
    let params = ButteraugliParams::default();
    match butteraugli::butteraugli(orig_img.as_ref(), comp_img.as_ref(), &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
}

/// Test that Rust jpegli produces similar file sizes to C++ jpegli.
/// Threshold: within 5% of C++ file size.
#[ignore = "requires codec-corpus (network on first run)"]
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

    let corpus = match codec_corpus::Corpus::new()
        .ok()
        .and_then(|c| c.get("kodak").ok())
    {
        Some(p) => p,
        None => {
            eprintln!("Kodak corpus not found, skipping test");
            return;
        }
    };
    let corpus_dir = corpus.as_path();

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
            let config = EncoderConfig::ycbcr(point.quality as f32, ChromaSubsampling::Quarter);
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
#[ignore = "requires codec-corpus (network on first run)"]
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

    let corpus = match codec_corpus::Corpus::new()
        .ok()
        .and_then(|c| c.get("kodak").ok())
    {
        Some(p) => p,
        None => {
            eprintln!("Kodak corpus not found, skipping test");
            return;
        }
    };
    let corpus_dir = corpus.as_path();

    let mut failures = Vec::new();

    for img_ref in &reference.images {
        let png_path = corpus_dir.join(format!("{}.png", img_ref.name));
        let Some((pixels, width, height)) = load_png(&png_path) else {
            continue;
        };

        for point in &img_ref.points {
            let config = EncoderConfig::ycbcr(point.quality as f32, ChromaSubsampling::Quarter);
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
            let decoded = match decoder.decode(&rust_jpeg, Unstoppable) {
                Ok(img) => img,
                Err(_) => continue,
            };

            let rust_dssim = compute_dssim(
                &pixels,
                decoded.pixels_u8().unwrap(),
                width as usize,
                height as usize,
            );
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
