//! Comprehensive comparison of Rust vs C++ jpegli across corpora.
//!
//! Measures: timing, file size, DSSIM, butteraugli at quality levels 2, 4, 6, ..., 100

use jpegli::types::{JpegMode, PixelFormat};
use jpegli::{Encoder, Quality};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let width = info.width;
    let height = info.height;

    // Convert to RGB if needed
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..info.buffer_size()]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn compute_dssim(orig: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = dssim::Dssim::new();

    let orig_rgba: Vec<rgb::RGBA8> = orig
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let dec_rgba: Vec<rgb::RGBA8> = decoded
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let dec_img = attr.create_image_rgba(&dec_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(data).decode().unwrap()
}

struct ComparisonResult {
    quality: u8,
    rust_size: usize,
    cpp_size: usize,
    rust_time_ms: f64,
    cpp_time_ms: f64,
    rust_dssim: f64,
    cpp_dssim: f64,
    rust_butteraugli: f64,
    cpp_butteraugli: f64,
}

fn compare_image(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    cjpegli_path: &Path,
    png_path: &Path,
) -> Option<ComparisonResult> {
    // Rust encoding with timing
    let rust_start = Instant::now();
    let rust_jpeg = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality as f32))
        .optimize_huffman(true)
        .mode(JpegMode::Progressive)
        .encode(rgb)
        .ok()?;
    let rust_time_ms = rust_start.elapsed().as_secs_f64() * 1000.0;
    let rust_size = rust_jpeg.len();

    // Decode Rust JPEG for quality metrics
    let rust_decoded = decode_jpeg(&rust_jpeg);
    let rust_dssim = compute_dssim(rgb, &rust_decoded, width as usize, height as usize);

    // Rust butteraugli
    let bfly_params = butteraugli_oxide::ButteraugliParams::default();
    let rust_butteraugli = butteraugli_oxide::compute_butteraugli(
        rgb,
        &rust_decoded,
        width as usize,
        height as usize,
        &bfly_params,
    )
    .score;

    // C++ encoding with timing
    let cpp_out = format!("/tmp/cpp_compare_q{}.jpg", quality);
    let cpp_start = Instant::now();
    let status = Command::new(cjpegli_path)
        .args([
            png_path.to_str().unwrap(),
            &cpp_out,
            "-q",
            &quality.to_string(),
            "--progressive_level=2",
        ])
        .output()
        .ok()?;
    let cpp_time_ms = cpp_start.elapsed().as_secs_f64() * 1000.0;

    if !status.status.success() {
        return None;
    }

    let cpp_jpeg = fs::read(&cpp_out).ok()?;
    let cpp_size = cpp_jpeg.len();

    // Decode C++ JPEG for quality metrics
    let cpp_decoded = decode_jpeg(&cpp_jpeg);
    let cpp_dssim = compute_dssim(rgb, &cpp_decoded, width as usize, height as usize);

    // C++ butteraugli
    let cpp_butteraugli = butteraugli_oxide::compute_butteraugli(
        rgb,
        &cpp_decoded,
        width as usize,
        height as usize,
        &bfly_params,
    )
    .score;

    // Cleanup
    let _ = fs::remove_file(&cpp_out);

    Some(ComparisonResult {
        quality,
        rust_size,
        cpp_size,
        rust_time_ms,
        cpp_time_ms,
        rust_dssim,
        cpp_dssim,
        rust_butteraugli,
        cpp_butteraugli,
    })
}

fn find_cjpegli() -> Option<std::path::PathBuf> {
    let paths = [
        "/home/lilith/work/jpegli/build/tools/cjpegli",
        "../../../build/tools/cjpegli",
        "../../build/tools/cjpegli",
    ];
    paths
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
}

fn find_corpus_images(max_images: usize) -> Vec<std::path::PathBuf> {
    let mut images = Vec::new();

    // Try CID22-512
    if let Ok(entries) = fs::read_dir("/mnt/v/work/corpus/CID22-512") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "png") {
                images.push(path);
                if images.len() >= max_images {
                    break;
                }
            }
        }
    }

    // Try testdata
    if images.len() < max_images {
        if let Ok(entries) = fs::read_dir("/home/lilith/work/jpegli/testdata/jxl/flower") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "png") {
                    images.push(path);
                    if images.len() >= max_images {
                        break;
                    }
                }
            }
        }
    }

    images
}

#[test]
#[ignore] // Requires C++ cjpegli and corpus images
fn test_comprehensive_cpp_comparison() {
    let cjpegli_path = match find_cjpegli() {
        Some(p) => p,
        None => {
            println!("Skipping: cjpegli not found");
            return;
        }
    };

    let max_images = std::env::var("MAX_IMAGES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let images = find_corpus_images(max_images);
    if images.is_empty() {
        println!("Skipping: no corpus images found");
        return;
    }

    // Quality levels: 2, 4, 6, ..., 100
    let qualities: Vec<u8> = (1..=50).map(|i| i * 2).collect();

    println!("\n{}", "=".repeat(120));
    println!(" COMPREHENSIVE RUST vs C++ JPEGLI COMPARISON ");
    println!("{}\n", "=".repeat(120));
    println!("Images: {}", images.len());
    println!("Quality levels: {:?}\n", qualities);

    // Aggregate results per quality level
    let mut aggregated: std::collections::HashMap<u8, Vec<ComparisonResult>> =
        std::collections::HashMap::new();

    for (img_idx, img_path) in images.iter().enumerate() {
        let img_name = img_path.file_name().unwrap().to_str().unwrap();
        println!(
            "[{}/{}] Processing: {}",
            img_idx + 1,
            images.len(),
            img_name
        );

        let (rgb, width, height) = match load_png(img_path) {
            Some(data) => data,
            None => {
                println!("  Skipping: failed to load");
                continue;
            }
        };

        // Save as PNG for C++ input
        let tmp_png = format!("/tmp/rust_compare_{}.png", img_idx);
        {
            let file = fs::File::create(&tmp_png).unwrap();
            let mut encoder = png::Encoder::new(file, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgb).unwrap();
        }

        for &q in &qualities {
            if let Some(result) =
                compare_image(&rgb, width, height, q, &cjpegli_path, Path::new(&tmp_png))
            {
                aggregated.entry(q).or_default().push(result);
            }
        }

        let _ = fs::remove_file(&tmp_png);
    }

    // Print summary table
    println!("\n{}", "=".repeat(140));
    println!(" SUMMARY (averaged across {} images) ", images.len());
    println!("{}\n", "=".repeat(140));

    println!(
        "{:>4} | {:>10} {:>10} {:>7} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7}",
        "Q",
        "Rust Size",
        "C++ Size",
        "Δ%",
        "Rust ms",
        "C++ ms",
        "Δ%",
        "Rust DSSIM",
        "C++ DSSIM",
        "Δ%",
        "Rust Bfly",
        "C++ Bfly",
        "Δ%"
    );
    println!("{:-<140}", "");

    let mut all_size_diffs = Vec::new();
    let mut all_dssim_diffs = Vec::new();
    let mut all_bfly_diffs = Vec::new();

    for q in &qualities {
        if let Some(results) = aggregated.get(q) {
            let n = results.len() as f64;

            let avg_rust_size: f64 = results.iter().map(|r| r.rust_size as f64).sum::<f64>() / n;
            let avg_cpp_size: f64 = results.iter().map(|r| r.cpp_size as f64).sum::<f64>() / n;
            let size_diff = (avg_rust_size - avg_cpp_size) / avg_cpp_size * 100.0;

            let avg_rust_time: f64 = results.iter().map(|r| r.rust_time_ms).sum::<f64>() / n;
            let avg_cpp_time: f64 = results.iter().map(|r| r.cpp_time_ms).sum::<f64>() / n;
            let time_diff = (avg_rust_time - avg_cpp_time) / avg_cpp_time * 100.0;

            let avg_rust_dssim: f64 = results.iter().map(|r| r.rust_dssim).sum::<f64>() / n;
            let avg_cpp_dssim: f64 = results.iter().map(|r| r.cpp_dssim).sum::<f64>() / n;
            let dssim_diff = if avg_cpp_dssim > 0.0 {
                (avg_rust_dssim - avg_cpp_dssim) / avg_cpp_dssim * 100.0
            } else {
                0.0
            };

            let avg_rust_bfly: f64 = results.iter().map(|r| r.rust_butteraugli).sum::<f64>() / n;
            let avg_cpp_bfly: f64 = results.iter().map(|r| r.cpp_butteraugli).sum::<f64>() / n;
            let bfly_diff = if avg_cpp_bfly > 0.0 {
                (avg_rust_bfly - avg_cpp_bfly) / avg_cpp_bfly * 100.0
            } else {
                0.0
            };

            all_size_diffs.push(size_diff);
            all_dssim_diffs.push(dssim_diff);
            all_bfly_diffs.push(bfly_diff);

            println!(
                "{:>4} | {:>10.0} {:>10.0} {:>+6.1}% | {:>8.2} {:>8.2} {:>+6.1}% | {:>8.6} {:>8.6} {:>+6.1}% | {:>8.4} {:>8.4} {:>+6.1}%",
                q, avg_rust_size, avg_cpp_size, size_diff,
                avg_rust_time, avg_cpp_time, time_diff,
                avg_rust_dssim, avg_cpp_dssim, dssim_diff,
                avg_rust_bfly, avg_cpp_bfly, bfly_diff
            );
        }
    }

    println!("{:-<140}", "");

    // Overall summary
    let avg_size_diff: f64 = all_size_diffs.iter().sum::<f64>() / all_size_diffs.len() as f64;
    let avg_dssim_diff: f64 = all_dssim_diffs.iter().sum::<f64>() / all_dssim_diffs.len() as f64;
    let avg_bfly_diff: f64 = all_bfly_diffs.iter().sum::<f64>() / all_bfly_diffs.len() as f64;

    println!("\nOVERALL AVERAGES:");
    println!(
        "  Size difference:       {:>+.2}% (positive = Rust larger)",
        avg_size_diff
    );
    println!(
        "  DSSIM difference:      {:>+.2}% (positive = Rust worse)",
        avg_dssim_diff
    );
    println!(
        "  Butteraugli difference: {:>+.2}% (positive = Rust worse)",
        avg_bfly_diff
    );

    // Quality parity assessment
    println!("\nQUALITY PARITY ASSESSMENT:");
    let dssim_match_count = all_dssim_diffs.iter().filter(|d| d.abs() < 5.0).count();
    let bfly_match_count = all_bfly_diffs.iter().filter(|d| d.abs() < 5.0).count();
    println!(
        "  DSSIM within 5%: {}/{} quality levels",
        dssim_match_count,
        all_dssim_diffs.len()
    );
    println!(
        "  Butteraugli within 5%: {}/{} quality levels",
        bfly_match_count,
        all_bfly_diffs.len()
    );
}
