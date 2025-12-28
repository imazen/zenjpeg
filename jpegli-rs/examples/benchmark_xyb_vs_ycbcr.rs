//! Benchmark C++ jpegli XYB vs YCbCr modes.
//!
//! Compares file sizes and quality metrics between XYB and YCbCr color spaces
//! using C++ cjpegli as the encoder.
//!
//! XYB decoding requires ICC profile application. This benchmark uses:
//! - For YCbCr: standard jpeg-decoder (no color management needed)
//! - For XYB: jpeg-decoder + ICC profile application via jpegli::icc

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use jpegli::icc::{apply_icc_transform, extract_icc_profile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeComparison {
    pub image: String,
    pub quality: u8,
    pub ycbcr_size: usize,
    pub xyb_size: usize,
    pub size_diff_pct: f64,
    pub ycbcr_dssim: f64,
    pub xyb_dssim: f64,
    pub dssim_diff_pct: f64,
    pub ycbcr_ssim2: f64,
    pub xyb_ssim2: f64,
    pub ycbcr_butteraugli: f64,
    pub xyb_butteraugli: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub generated_at: String,
    pub comparisons: Vec<ModeComparison>,
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

fn encode_with_cjpegli(
    cjpegli_path: &str,
    input_path: &Path,
    quality: u8,
    xyb: bool,
) -> Option<Vec<u8>> {
    let mode = if xyb { "xyb" } else { "ycbcr" };
    let temp_output = std::env::temp_dir().join(format!(
        "cjpegli_bench_{}_{}_{}.jpg",
        input_path.file_stem()?.to_str()?,
        quality,
        mode
    ));

    let mut cmd = Command::new(cjpegli_path);
    cmd.arg(input_path)
        .arg(&temp_output)
        .arg(format!("--quality={}", quality));

    if xyb {
        cmd.arg("--xyb");
    }

    let status = cmd.status().ok()?;

    if !status.success() {
        return None;
    }

    let data = fs::read(&temp_output).ok()?;
    let _ = fs::remove_file(&temp_output);
    Some(data)
}

fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;

    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => return None,
    };

    Some((rgb, info.width as u32, info.height as u32))
}

/// Decode XYB JPEG with ICC profile application.
///
/// XYB JPEGs store data in XYB color space with an embedded ICC profile.
/// We need to apply this profile to convert back to sRGB for quality comparison.
fn decode_jpeg_with_icc(data: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    // Extract ICC profile
    let icc_profile = extract_icc_profile(data);

    // Decode JPEG
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;

    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
        _ => return None,
    };

    let width = info.width as u32;
    let height = info.height as u32;

    // Apply ICC profile if present (required for XYB)
    let output = if let Some(ref profile) = icc_profile {
        match apply_icc_transform(&rgb, width as usize, height as usize, profile) {
            Ok(converted) => converted,
            Err(e) => {
                eprintln!("Warning: ICC transform failed: {:?}", e);
                rgb
            }
        }
    } else {
        rgb
    };

    Some((output, width, height))
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

fn compute_ssimulacra2(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

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

fn compute_butteraugli_score(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    use butteraugli::{compute_butteraugli, ButteraugliParams};

    let params = ButteraugliParams::default();
    match compute_butteraugli(orig, comp, width, height, &params) {
        Ok(result) => result.score,
        Err(_) => f64::NAN,
    }
}

fn main() {
    let cjpegli_path = std::env::var("CJPEGLI_PATH")
        .map(PathBuf::from)
        .or_else(|_| jpegli::test_utils::find_cjpegli().ok_or(()))
        .expect("cjpegli not found. Set CJPEGLI_PATH or build internal/jpegli-cpp");
    let cjpegli_path = cjpegli_path.to_string_lossy().to_string();

    // Check for corpus directories
    let testdata_dir = jpegli::test_utils::get_testdata_dir();
    let mut corpus_paths: Vec<PathBuf> = Vec::new();
    let flower_dir = testdata_dir.join("jxl/flower");
    if flower_dir.exists() {
        corpus_paths.push(flower_dir);
    }
    for c in [
        "../codec-eval/codec-corpus/kodak",
        "../codec-corpus/kodak",
        "codec-corpus/kodak",
    ] {
        let p = PathBuf::from(c);
        if p.exists() {
            corpus_paths.push(p);
        }
    }

    let corpus_dir = corpus_paths
        .into_iter()
        .find(|p| p.exists())
        .expect("No corpus found. Set JPEGLI_TESTDATA env var");

    let qualities: Vec<u8> = vec![30, 50, 70, 80, 85, 90, 95];

    eprintln!("Benchmarking C++ jpegli: XYB vs YCbCr");
    eprintln!("  cjpegli: {}", cjpegli_path);
    eprintln!("  Using Rust ICC decoding for XYB");
    eprintln!("  corpus: {}", corpus_dir.display());
    eprintln!("  qualities: {:?}\n", qualities);

    // Print header
    println!(
        "{:<8} {:>3} {:>8} {:>8} {:>7} {:>8} {:>8} {:>7} {:>6} {:>6}",
        "Image", "Q", "YCbCr", "XYB", "Δ Size", "DSSIM_Y", "DSSIM_X", "Δ DSSIM", "Butt_Y", "Butt_X"
    );
    println!("{}", "-".repeat(95));

    let mut png_files: Vec<_> = fs::read_dir(&corpus_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "png"))
        .map(|e| e.path())
        .collect();
    png_files.sort();

    let max_images = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    png_files.truncate(max_images);

    let mut comparisons = Vec::new();
    let mut total_ycbcr_size = 0usize;
    let mut total_xyb_size = 0usize;
    let mut total_ycbcr_dssim = 0.0f64;
    let mut total_xyb_dssim = 0.0f64;
    let mut count = 0usize;

    for png_path in &png_files {
        let name = png_path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let Some((orig_rgb, width, height)) = load_png(png_path) else {
            continue;
        };

        for &quality in &qualities {
            // Encode with YCbCr
            let Some(ycbcr_jpeg) = encode_with_cjpegli(&cjpegli_path, png_path, quality, false)
            else {
                continue;
            };
            let ycbcr_size = ycbcr_jpeg.len();

            // Encode with XYB
            let Some(xyb_jpeg) = encode_with_cjpegli(&cjpegli_path, png_path, quality, true) else {
                continue;
            };
            let xyb_size = xyb_jpeg.len();

            // Decode YCbCr with jpeg-decoder (no ICC needed)
            let Some((ycbcr_decoded, _, _)) = decode_jpeg(&ycbcr_jpeg) else {
                continue;
            };

            // Decode XYB with ICC profile application
            let Some((xyb_decoded, _, _)) = decode_jpeg_with_icc(&xyb_jpeg) else {
                continue;
            };

            let ycbcr_dssim =
                compute_dssim(&orig_rgb, &ycbcr_decoded, width as usize, height as usize);
            let xyb_dssim = compute_dssim(&orig_rgb, &xyb_decoded, width as usize, height as usize);

            let ycbcr_ssim2 =
                compute_ssimulacra2(&orig_rgb, &ycbcr_decoded, width as usize, height as usize);
            let xyb_ssim2 =
                compute_ssimulacra2(&orig_rgb, &xyb_decoded, width as usize, height as usize);

            let ycbcr_butt = compute_butteraugli_score(
                &orig_rgb,
                &ycbcr_decoded,
                width as usize,
                height as usize,
            );
            let xyb_butt =
                compute_butteraugli_score(&orig_rgb, &xyb_decoded, width as usize, height as usize);

            let size_diff_pct = ((xyb_size as f64 - ycbcr_size as f64) / ycbcr_size as f64) * 100.0;
            let dssim_diff_pct = if ycbcr_dssim > 0.0 {
                ((xyb_dssim - ycbcr_dssim) / ycbcr_dssim) * 100.0
            } else {
                0.0
            };

            println!(
                "{:<8} {:>3} {:>8} {:>8} {:>+6.1}% {:>.6} {:>.6} {:>+6.1}% {:>6.2} {:>6.2}",
                name,
                quality,
                ycbcr_size,
                xyb_size,
                size_diff_pct,
                ycbcr_dssim,
                xyb_dssim,
                dssim_diff_pct,
                ycbcr_butt,
                xyb_butt
            );

            total_ycbcr_size += ycbcr_size;
            total_xyb_size += xyb_size;
            total_ycbcr_dssim += ycbcr_dssim;
            total_xyb_dssim += xyb_dssim;
            count += 1;

            comparisons.push(ModeComparison {
                image: name.to_string(),
                quality,
                ycbcr_size,
                xyb_size,
                size_diff_pct,
                ycbcr_dssim,
                xyb_dssim,
                dssim_diff_pct,
                ycbcr_ssim2,
                xyb_ssim2,
                ycbcr_butteraugli: ycbcr_butt,
                xyb_butteraugli: xyb_butt,
            });
        }
    }

    println!("{}", "-".repeat(95));

    let overall_size_diff =
        ((total_xyb_size as f64 - total_ycbcr_size as f64) / total_ycbcr_size as f64) * 100.0;
    let avg_ycbcr_dssim = total_ycbcr_dssim / count as f64;
    let avg_xyb_dssim = total_xyb_dssim / count as f64;
    let avg_dssim_diff = ((avg_xyb_dssim - avg_ycbcr_dssim) / avg_ycbcr_dssim) * 100.0;

    println!(
        "{:<8} {:>3} {:>8} {:>8} {:>+6.1}% {:>.6} {:>.6} {:>+6.1}%",
        "TOTAL",
        "",
        total_ycbcr_size,
        total_xyb_size,
        overall_size_diff,
        avg_ycbcr_dssim,
        avg_xyb_dssim,
        avg_dssim_diff
    );

    eprintln!("\n=== Summary ===");
    eprintln!("XYB vs YCbCr file size: {:+.2}%", overall_size_diff);
    eprintln!(
        "XYB vs YCbCr avg DSSIM: {:+.2}% (lower is better)",
        avg_dssim_diff
    );
    eprintln!("\nPositive size diff = XYB is larger");
    eprintln!("Negative DSSIM diff = XYB has better quality");

    // Save results
    let results = BenchmarkResults {
        generated_at: chrono::Utc::now().to_rfc3339(),
        comparisons,
    };
    let json = serde_json::to_string_pretty(&results).unwrap();
    fs::write("xyb_vs_ycbcr_benchmark.json", &json).ok();
}
