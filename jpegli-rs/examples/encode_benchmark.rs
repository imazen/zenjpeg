//! Comprehensive encoding benchmark: jpegli-rs vs C jpegli
//!
//! Tests 20 quality levels across CLIC2025 and Kodak corpuses.
//! Runs each encoding twice and takes the lowest time.
//!
//! Usage:
//!   cargo run --release --example encode_benchmark
//!
//! Environment variables:
//!   KODAK_DIR - Path to Kodak corpus (default: /home/lilith/work/codec-corpus/kodak)
//!   CLIC_DIR - Path to CLIC2025 validation (default: /home/lilith/work/codec-corpus/clic2025/validation)
//!   MAX_FILES - Limit number of files per corpus (default: all)
//!   QUALITIES - Comma-separated quality values (default: 20 values from 30-97)

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

fn main() {
    let kodak_dir = PathBuf::from(
        env::var("KODAK_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus/kodak".to_string()),
    );
    let clic_dir = PathBuf::from(
        env::var("CLIC_DIR")
            .unwrap_or_else(|_| "/home/lilith/work/codec-corpus/clic2025/validation".to_string()),
    );

    let max_files: Option<usize> = env::var("MAX_FILES").ok().and_then(|s| s.parse().ok());

    // 20 quality levels spread across the useful range
    let qualities: Vec<u8> = env::var("QUALITIES")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| {
            vec![
                30, 35, 40, 45, 50, 55, 60, 65, 70, 72, 75, 78, 80, 82, 85, 88, 90, 92, 95, 97,
            ]
        });

    // Collect images from both corpuses
    let mut files: Vec<(PathBuf, &str)> = Vec::new();

    // Kodak corpus
    if kodak_dir.exists() {
        let mut kodak_files: Vec<PathBuf> = std::fs::read_dir(&kodak_dir)
            .unwrap_or_else(|_| panic!("Failed to read Kodak dir: {}", kodak_dir.display()))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        kodak_files.sort();
        if let Some(max) = max_files {
            kodak_files.truncate(max);
        }
        for f in kodak_files {
            files.push((f, "kodak"));
        }
    } else {
        eprintln!("Warning: Kodak dir not found: {}", kodak_dir.display());
    }

    // CLIC2025 corpus
    if clic_dir.exists() {
        let mut clic_files: Vec<PathBuf> = std::fs::read_dir(&clic_dir)
            .unwrap_or_else(|_| panic!("Failed to read CLIC dir: {}", clic_dir.display()))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "png"))
            .collect();
        clic_files.sort();
        if let Some(max) = max_files {
            clic_files.truncate(max);
        }
        for f in clic_files {
            files.push((f, "clic2025"));
        }
    } else {
        eprintln!("Warning: CLIC dir not found: {}", clic_dir.display());
    }

    if files.is_empty() {
        eprintln!("No PNG files found!");
        return;
    }

    eprintln!(
        "Benchmark: {} images × {} quality levels = {} encodings per encoder",
        files.len(),
        qualities.len(),
        files.len() * qualities.len()
    );
    eprintln!("Quality levels: {:?}", qualities);
    eprintln!();

    // CSV header
    println!("corpus,image,width,height,pixels,quality,encoder,bytes,bpp,encode_ms,mpixels_per_sec,dssim,ssimulacra2");

    for (path, corpus) in &files {
        if let Some(img) = load_image(path) {
            for &q in &qualities {
                // jpegli-rs encoding (2x, take lowest)
                let rust_result = benchmark_rust_encode(&img, q);
                print_result(corpus, &img, q, "jpegli-rs", &rust_result);

                // C jpegli encoding (2x, take lowest)
                let cpp_result = benchmark_cpp_encode(path, &img, q);
                print_result(corpus, &img, q, "cjpegli", &cpp_result);
            }
            // Flush after each image for progress visibility
            std::io::stdout().flush().ok();
        }
    }
}

struct ImageData {
    name: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

struct BenchmarkResult {
    bytes: usize,
    encode_time: Duration,
    dssim: f64,
    ssimulacra2: f64,
}

fn print_result(
    corpus: &str,
    img: &ImageData,
    quality: u8,
    encoder: &str,
    result: &BenchmarkResult,
) {
    let pixels = img.width * img.height;
    let bpp = 8.0 * result.bytes as f64 / pixels as f64;
    let encode_ms = result.encode_time.as_secs_f64() * 1000.0;
    let mpixels_per_sec = (pixels as f64 / 1_000_000.0) / result.encode_time.as_secs_f64();

    println!(
        "{},{},{},{},{},{},{},{},{:.4},{:.2},{:.2},{:.8},{:.4}",
        corpus,
        img.name,
        img.width,
        img.height,
        pixels,
        quality,
        encoder,
        result.bytes,
        bpp,
        encode_ms,
        mpixels_per_sec,
        result.dssim,
        result.ssimulacra2,
    );
}

fn load_image(path: &PathBuf) -> Option<ImageData> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    // Handle different color types
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            // Convert RGBA to RGB
            buf[..info.buffer_size()]
                .chunks(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect()
        }
        _ => return None,
    };

    Some(ImageData {
        name: path.file_name()?.to_string_lossy().to_string(),
        pixels,
        width: info.width as usize,
        height: info.height as usize,
    })
}

fn benchmark_rust_encode(img: &ImageData, quality: u8) -> BenchmarkResult {
    // Run twice, take lowest time
    let mut best_time = Duration::MAX;
    let mut jpeg_data = Vec::new();

    for _ in 0..2 {
        let start = Instant::now();
        let data = jpegli::Encoder::new()
            .width(img.width as u32)
            .height(img.height as u32)
            .jpegli_quality(jpegli::quant::Quality::from_quality(quality as f32))
            .encode(&img.pixels)
            .expect("jpegli-rs encode failed");
        let elapsed = start.elapsed();

        if elapsed < best_time {
            best_time = elapsed;
            jpeg_data = data;
        }
    }

    let decoded = decode_jpeg(&jpeg_data);
    let (dssim, ssim2) = compute_metrics(&img.pixels, &decoded, img.width, img.height);

    BenchmarkResult {
        bytes: jpeg_data.len(),
        encode_time: best_time,
        dssim,
        ssimulacra2: ssim2,
    }
}

fn benchmark_cpp_encode(input_path: &PathBuf, img: &ImageData, quality: u8) -> BenchmarkResult {
    // Create temp file for output
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join(format!("cjpegli_bench_{}.jpg", std::process::id()));

    // Run twice, take lowest time
    let mut best_time = Duration::MAX;

    for _ in 0..2 {
        let start = Instant::now();
        let status = Command::new("cjpegli")
            .arg(input_path)
            .arg(&output_path)
            .arg("-q")
            .arg(quality.to_string())
            .arg("--chroma_subsampling=444") // Match jpegli-rs default
            .arg("-p")
            .arg("0") // Sequential (no progressive) to match jpegli-rs
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("Failed to run cjpegli");
        let elapsed = start.elapsed();

        if status.success() && elapsed < best_time {
            best_time = elapsed;
        }
    }

    // Read output file
    let jpeg_data = std::fs::read(&output_path).unwrap_or_default();
    let _ = std::fs::remove_file(&output_path);

    if jpeg_data.is_empty() {
        return BenchmarkResult {
            bytes: 0,
            encode_time: best_time,
            dssim: 99.0,
            ssimulacra2: 0.0,
        };
    }

    let decoded = decode_jpeg(&jpeg_data);
    let (dssim, ssim2) = compute_metrics(&img.pixels, &decoded, img.width, img.height);

    BenchmarkResult {
        bytes: jpeg_data.len(),
        encode_time: best_time,
        dssim,
        ssimulacra2: ssim2,
    }
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().unwrap_or_default()
}

fn compute_metrics(original: &[u8], decoded: &[u8], width: usize, height: usize) -> (f64, f64) {
    if decoded.is_empty() || decoded.len() != original.len() {
        return (99.0, 0.0);
    }

    let dssim = compute_dssim(original, decoded, width, height);
    let ssim2 = compute_ssimulacra2(original, decoded, width, height);
    (dssim, ssim2)
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use dssim::Dssim;

    let attr = Dssim::new();

    let orig_rgba: Vec<rgb::RGBA<u8>> = original
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = match attr.create_image_rgba(&orig_rgba, width, height) {
        Some(img) => img,
        None => return 99.0,
    };

    let decoded_rgba: Vec<rgb::RGBA<u8>> = decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let decoded_img = match attr.create_image_rgba(&decoded_rgba, width, height) {
        Some(img) => img,
        None => return 99.0,
    };

    let (dssim, _) = attr.compare(&orig_img, decoded_img);
    dssim.into()
}

fn compute_ssimulacra2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

    let orig_rgb = match Rgb::new(
        original
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    ) {
        Ok(rgb) => rgb,
        Err(_) => return 0.0,
    };

    let decoded_rgb = match Rgb::new(
        decoded
            .chunks(3)
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]
            })
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    ) {
        Ok(rgb) => rgb,
        Err(_) => return 0.0,
    };

    compute_frame_ssimulacra2(orig_rgb, decoded_rgb).unwrap_or(0.0)
}
