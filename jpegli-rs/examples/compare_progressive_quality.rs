//! Compare decoded pixels between C++ and Rust progressive JPEGs

use std::fs;
use std::io::{Cursor, Write};
use std::process::Command;

fn main() {
    // Build test image list dynamically
    let flower_path = jpegli::test_utils::require_flower_small_path();
    let flower_str = flower_path.to_string_lossy().to_string();

    let mut test_images: Vec<(String, &str)> = vec![(flower_str.clone(), "flower")];

    // Add optional corpus images if they exist
    for (path, name) in [
        ("/mnt/v/work/corpus/CID22-512/1459534.png", "cid22_large"),
        (
            "/mnt/v/work/corpus/CID22-512/2504911.png",
            "cid22_medium_large",
        ),
        ("/mnt/v/work/corpus/CID22-512/3616956.png", "cid22_medium"),
        (
            "/mnt/v/work/corpus/CID22-512/nicubunu_Game_baddie_Policeman.png",
            "cid22_small",
        ),
    ] {
        if std::path::Path::new(path).exists() {
            test_images.push((path.to_string(), name));
        }
    }

    println!("=== Progressive JPEG Quality Comparison (C++ vs Rust) ===\n");

    for q in [50, 70, 90] {
        println!("\n--- Quality {} ---", q);
        println!(
            "{:<20} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Image", "C++ size", "Rust size", "Exact%", "±1", "±2", "±3+", "MaxDiff"
        );
        println!("{}", "-".repeat(96));

        for (png_path, name) in &test_images {
            if let Some(result) = compare_image(png_path, name, q) {
                let total = result.total_pixels as f64;
                println!(
                    "{:<20} {:>10} {:>10} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
                    name,
                    result.cpp_size,
                    result.rust_size,
                    100.0 * result.exact_match as f64 / total,
                    100.0 * result.off_by_1 as f64 / total,
                    100.0 * result.off_by_2 as f64 / total,
                    100.0 * result.off_by_3_plus as f64 / total,
                    result.max_diff,
                );
            } else {
                println!("{:<20} SKIPPED", name);
            }
        }
    }

    println!("\n=== Detailed Histogram for flower at Q50 ===\n");

    // Just do flower for detailed histogram
    if let Some((hist, max_diff)) = detailed_histogram(&flower_str, "flower", 50) {
        println!("Flower image difference histogram:");
        for (diff, count) in hist.iter().enumerate().take(max_diff as usize + 1) {
            if *count > 0 {
                let pct = 100.0 * *count as f64 / hist.iter().sum::<usize>() as f64;
                let bar = "█".repeat((pct * 2.0).min(60.0) as usize);
                println!("  {:>2}: {:>8} ({:>5.2}%) {}", diff, count, pct, bar);
            }
        }
    }
}

struct CompareResult {
    cpp_size: usize,
    rust_size: usize,
    total_pixels: usize,
    exact_match: usize,
    off_by_1: usize,
    off_by_2: usize,
    off_by_3_plus: usize,
    max_diff: u8,
    cpp_vs_orig_dssim: f64,
    rust_vs_orig_dssim: f64,
}

fn compare_image(png_path: &str, name: &str, quality: u32) -> Option<CompareResult> {
    let (rgb, width, height) = load_png(png_path)?;

    let ppm_path = format!("/tmp/{}_compare.ppm", name);
    write_ppm(&ppm_path, &rgb, width as usize, height as usize).ok()?;

    let cpp_jpg_path = format!("/tmp/{}_cpp_prog_q{}.jpg", name, quality);
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;

    // C++ progressive mode with -p 2 (level 2 = successive approximation)
    // Note: --fixed_code can't be used with progressive, so both use Huffman optimization
    // Both C++ and Rust use adaptive quantization (default)
    let output = Command::new(&cjpegli_path)
        .args([
            "--chroma_subsampling=444",
            "-p", "2",  // Progressive level 2
            &ppm_path,
            &cpp_jpg_path,
            "-q",
            &quality.to_string(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!("cjpegli failed: {}", String::from_utf8_lossy(&output.stderr));
        return None;
    }

    // Rust progressive mode with Huffman optimization
    // Match C++ settings: with adaptive quantization, 4:4:4 subsampling
    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(true)  // Progressive uses optimized Huffman
        .subsampling(jpegli::types::Subsampling::S444)  // Match C++ --chroma_subsampling=444
        .jpegli_quality(jpegli::quant::Quality::Traditional(quality as f32))
        .encode(&rgb)
        .ok()?;

    let cpp_jpeg = fs::read(&cpp_jpg_path).ok()?;

    // Use zune-jpeg for decoding (more tolerant than jpeg-decoder)
    let cpp_decoded = zune_jpeg::JpegDecoder::new(Cursor::new(&cpp_jpeg)).decode().ok()?;
    let rust_decoded = zune_jpeg::JpegDecoder::new(Cursor::new(&rust_jpeg)).decode().ok()?;

    if cpp_decoded.len() != rust_decoded.len() {
        eprintln!("Size mismatch: C++ {} vs Rust {}", cpp_decoded.len(), rust_decoded.len());
        return None;
    }

    let total_pixels = cpp_decoded.len();
    let mut exact_match = 0usize;
    let mut off_by_1 = 0usize;
    let mut off_by_2 = 0usize;
    let mut off_by_3_plus = 0usize;
    let mut max_diff = 0u8;

    for i in 0..total_pixels {
        let diff = (cpp_decoded[i] as i16 - rust_decoded[i] as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);

        match diff {
            0 => exact_match += 1,
            1 => off_by_1 += 1,
            2 => off_by_2 += 1,
            _ => off_by_3_plus += 1,
        }
    }

    Some(CompareResult {
        cpp_size: cpp_jpeg.len(),
        rust_size: rust_jpeg.len(),
        total_pixels,
        exact_match,
        off_by_1,
        off_by_2,
        off_by_3_plus,
        max_diff,
        cpp_vs_orig_dssim: 0.0,
        rust_vs_orig_dssim: 0.0,
    })
}

fn detailed_histogram(png_path: &str, name: &str, quality: u32) -> Option<(Vec<usize>, u8)> {
    let (rgb, width, height) = load_png(png_path)?;

    let ppm_path = format!("/tmp/{}_compare.ppm", name);
    write_ppm(&ppm_path, &rgb, width as usize, height as usize).ok()?;

    let cpp_jpg_path = format!("/tmp/{}_cpp_prog_q{}.jpg", name, quality);
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;
    Command::new(&cjpegli_path)
        .args([
            "--noadaptive_quantization",
            "--chroma_subsampling=444",
            "-p", "2",
            &ppm_path,
            &cpp_jpg_path,
            "-q",
            &quality.to_string(),
        ])
        .output()
        .ok()?;

    let rust_jpeg = jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .mode(jpegli::types::JpegMode::Progressive)
        .optimize_huffman(true)
        .jpegli_quality(jpegli::quant::Quality::Traditional(quality as f32))
        .encode(&rgb)
        .ok()?;

    let cpp_jpeg = fs::read(&cpp_jpg_path).ok()?;
    let cpp_decoded = zune_jpeg::JpegDecoder::new(Cursor::new(&cpp_jpeg)).decode().ok()?;
    let rust_decoded = zune_jpeg::JpegDecoder::new(Cursor::new(&rust_jpeg)).decode().ok()?;

    let mut histogram = vec![0usize; 256];
    let mut max_diff = 0u8;

    for i in 0..cpp_decoded.len() {
        let diff = (cpp_decoded[i] as i16 - rust_decoded[i] as i16).unsigned_abs() as u8;
        histogram[diff as usize] += 1;
        max_diff = max_diff.max(diff);
    }

    Some((histogram, max_diff))
}

fn load_png(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let decoder = png::Decoder::new(fs::File::open(path).ok()?);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => return None,
    };
    Some((rgb, info.width, info.height))
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6\n{} {}\n255", width, height)?;
    file.write_all(rgb)
}
