//! AQ Tuning for Sharpened Images
//!
//! Binary search for optimal AQ scale factors on sharpened/contrast-boosted images.
//! Uses DSSIM and SSIMULACRA2 for quality measurement.
//!
//! Usage:
//!   cargo run --release --example aq_sharpened_tuning -- /path/to/corpus [output.csv]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use dssim::Dssim;
use jpegli::adaptive_quant::compute_aq_strength_map;
use jpegli::{Encoder, PixelFormat, Quality};
use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

/// Quality metrics for a single encode
#[derive(Debug, Clone)]
struct EncodeResult {
    file_size: usize,
    bpp: f64,
    dssim: f64,
    ssimulacra2: f64,
    encode_time_ms: u64,
}

/// Result of testing one AQ configuration
#[derive(Debug, Clone)]
struct AQTestResult {
    aq_scale: f32,
    distance: f32,
    aq_mean: f32,
    encode: EncodeResult,
}

/// Load a PNG image
fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if needed
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

/// Compute DSSIM between two RGB images
fn compute_dssim(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    let orig_rgba: Vec<rgb::RGBA<u8>> = orig
        .chunks(3)
        .map(|c| rgb::RGBA::new(c[0], c[1], c[2], 255))
        .collect();
    let comp_rgba: Vec<rgb::RGBA<u8>> = comp
        .chunks(3)
        .map(|c| rgb::RGBA::new(c[0], c[1], c[2], 255))
        .collect();

    let orig_img = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
    let comp_img = attr.create_image_rgba(&comp_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&orig_img, comp_img);
    dssim.into()
}

/// Compute SSIMULACRA2 between two RGB images
fn compute_ssim2(orig: &[u8], comp: &[u8], width: usize, height: usize) -> f64 {
    let orig_rgb = Rgb::new(
        orig.chunks(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("create orig rgb");

    let comp_rgb = Rgb::new(
        comp.chunks(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("create comp rgb");

    compute_frame_ssimulacra2(orig_rgb, comp_rgb).unwrap_or(0.0)
}

/// Decode JPEG to RGB
fn decode_jpeg(data: &[u8]) -> Option<(Vec<u8>, usize, usize)> {
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let width = info.width as usize;
    let height = info.height as usize;

    // Convert to RGB if grayscale
    let rgb = if info.pixel_format == jpeg_decoder::PixelFormat::L8 {
        pixels.iter().flat_map(|&g| [g, g, g]).collect()
    } else {
        pixels
    };

    Some((rgb, width, height))
}

/// Encode image with specific AQ scale
fn encode_with_aq_scale(
    pixels: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    aq_scale: f32,
) -> Result<(Vec<u8>, f32), String> {
    // Extract Y plane for AQ computation
    let y_plane: Vec<f32> = pixels
        .chunks(3)
        .map(|rgb| 0.299 * rgb[0] as f32 + 0.587 * rgb[1] as f32 + 0.114 * rgb[2] as f32)
        .collect();

    // Compute AQ map - use approximate y_quant_01 based on distance
    let y_quant_01 = (distance * 8.0).max(1.0) as u16;
    let mut aq_map = compute_aq_strength_map(&y_plane, width, height, y_quant_01);

    // Apply the scale factor
    aq_map.scale(aq_scale);
    let aq_mean = aq_map.mean();

    // Encode with custom AQ map
    let jpeg_data = Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_distance(distance))
        .aq_map(aq_map)
        .encode(pixels)
        .map_err(|e| format!("encode error: {e:?}"))?;

    Ok((jpeg_data, aq_mean))
}

/// Test one image at one configuration
fn test_config(
    pixels: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    aq_scale: f32,
) -> Result<AQTestResult, String> {
    let start = Instant::now();

    // Encode
    let (jpeg_data, aq_mean) = encode_with_aq_scale(pixels, width, height, distance, aq_scale)?;

    let encode_time = start.elapsed().as_millis() as u64;
    let file_size = jpeg_data.len();
    let total_pixels = width * height;
    let bpp = (file_size * 8) as f64 / total_pixels as f64;

    // Decode and measure quality
    let (decoded, dec_w, dec_h) =
        decode_jpeg(&jpeg_data).ok_or_else(|| "failed to decode".to_string())?;

    if dec_w != width || dec_h != height {
        return Err(format!(
            "size mismatch: {width}x{height} vs {dec_w}x{dec_h}"
        ));
    }

    let dssim = compute_dssim(pixels, &decoded, width, height);
    let ssimulacra2 = compute_ssim2(pixels, &decoded, width, height);

    Ok(AQTestResult {
        aq_scale,
        distance,
        aq_mean,
        encode: EncodeResult {
            file_size,
            bpp,
            dssim,
            ssimulacra2,
            encode_time_ms: encode_time,
        },
    })
}

/// Sweep AQ scales and find Pareto-optimal points
fn aq_scale_sweep(pixels: &[u8], width: usize, height: usize, distance: f32) -> Vec<AQTestResult> {
    let scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0];
    let mut results = Vec::new();

    for &scale in &scales {
        match test_config(pixels, width, height, distance, scale) {
            Ok(result) => results.push(result),
            Err(e) => eprintln!("  scale={scale:.2}: {e}"),
        }
    }

    results
}

/// Find optimal AQ scale - best quality per byte
fn find_optimal_aq_for_quality(results: &[AQTestResult]) -> Option<(f32, f64)> {
    // Quality efficiency = ssimulacra2 / bpp (higher is better)
    results
        .iter()
        .map(|r| {
            let efficiency = r.encode.ssimulacra2 / r.encode.bpp;
            (r.aq_scale, efficiency)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

/// Find optimal AQ scale minimizing DSSIM * bpp (rate-distortion)
fn find_optimal_rd(results: &[AQTestResult]) -> Option<(f32, f64)> {
    results
        .iter()
        .map(|r| {
            let rd = r.encode.dssim * r.encode.bpp;
            (r.aq_scale, rd)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

/// Binary search for AQ scale targeting a specific DSSIM
fn binary_search_aq_for_dssim(
    pixels: &[u8],
    width: usize,
    height: usize,
    distance: f32,
    target_dssim: f64,
    tolerance: f64,
) -> Option<(f32, AQTestResult)> {
    let mut low = 0.25f32;
    let mut high = 3.0f32;
    let mut best_result: Option<AQTestResult> = None;
    let mut best_scale = 1.0f32;

    // Binary search for ~12 iterations
    for _ in 0..12 {
        let mid = (low + high) / 2.0;

        if let Ok(result) = test_config(pixels, width, height, distance, mid) {
            let diff = result.encode.dssim - target_dssim;

            if diff.abs() < tolerance {
                return Some((mid, result));
            }

            // Higher AQ scale = more bits to detail = lower DSSIM (better quality)
            // If DSSIM is too high (worse quality), increase scale
            if diff > 0.0 {
                low = mid;
            } else {
                high = mid;
            }

            let abs_diff = diff.abs();
            if best_result.is_none()
                || abs_diff < (best_result.as_ref().unwrap().encode.dssim - target_dssim).abs()
            {
                best_result = Some(result);
                best_scale = mid;
            }
        }
    }

    best_result.map(|r| (best_scale, r))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <corpus_dir> [output.csv]", args[0]);
        eprintln!("\nExpected corpus structure:");
        eprintln!("  corpus_dir/*.png - sharpened source images");
        std::process::exit(1);
    }

    let corpus_dir = PathBuf::from(&args[1]);
    let output_csv = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| corpus_dir.join("aq_tuning_results.csv"));

    // Collect PNG files
    let mut png_files: Vec<PathBuf> = fs::read_dir(&corpus_dir)
        .expect("read corpus dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();

    png_files.sort();

    println!("Found {} PNG files in {:?}", png_files.len(), corpus_dir);

    // Quality distances to test
    let distances = [0.5, 1.0, 1.5, 2.0, 3.0];

    // Open CSV for results
    let mut csv = fs::File::create(&output_csv).expect("create csv");
    writeln!(
        csv,
        "image,distance,aq_scale,aq_mean,file_size,bpp,dssim,ssimulacra2,encode_ms"
    )
    .unwrap();

    // Track aggregated results per distance
    let mut per_distance_optimal: std::collections::HashMap<String, Vec<f32>> =
        std::collections::HashMap::new();

    for (idx, png_path) in png_files.iter().enumerate() {
        let filename = png_path.file_name().unwrap().to_string_lossy().to_string();
        println!(
            "\n[{}/{}] Processing: {}",
            idx + 1,
            png_files.len(),
            filename
        );

        let (pixels, width, height) = match load_png(png_path) {
            Some(data) => data,
            None => {
                eprintln!("  Failed to load, skipping");
                continue;
            }
        };

        println!("  Size: {}x{}", width, height);

        for &distance in &distances {
            print!("  d={distance}: ");
            let results = aq_scale_sweep(&pixels, width, height, distance);

            if results.is_empty() {
                println!("no valid results");
                continue;
            }

            // Find optimal (best RD efficiency)
            if let Some((opt_scale, opt_rd)) = find_optimal_rd(&results) {
                print!("optimal={opt_scale:.2} (rd={opt_rd:.6}) ");
                per_distance_optimal
                    .entry(format!("d{distance}"))
                    .or_default()
                    .push(opt_scale);
            }

            // Write to CSV
            for r in &results {
                writeln!(
                    csv,
                    "{},{},{:.3},{:.4},{},{:.4},{:.6},{:.2},{}",
                    filename,
                    r.distance,
                    r.aq_scale,
                    r.aq_mean,
                    r.encode.file_size,
                    r.encode.bpp,
                    r.encode.dssim,
                    r.encode.ssimulacra2,
                    r.encode.encode_time_ms
                )
                .unwrap();
            }

            // Print summary
            if let (Some(first), Some(last)) = (results.first(), results.last()) {
                println!(
                    "bpp {:.2}-{:.2}, dssim {:.5}-{:.5}",
                    first.encode.bpp,
                    last.encode.bpp,
                    first.encode.dssim,
                    last.encode.dssim
                );
            }
        }
    }

    println!("\n=== Summary ===");
    println!("Results written to: {:?}", output_csv);

    // Print average optimal AQ scales per distance
    println!("\nOptimal AQ scale (minimizing DSSIM*bpp):");
    for (key, values) in per_distance_optimal.iter() {
        if !values.is_empty() {
            let avg = values.iter().sum::<f32>() / values.len() as f32;
            let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!("  {key}: avg={avg:.2}, range=[{min:.2}, {max:.2}] (n={})", values.len());
        }
    }

    // Binary search demo on first image
    if let Some(first_path) = png_files.first() {
        if let Some((pixels, width, height)) = load_png(first_path) {
            println!("\n=== Binary Search Demo ===");
            let target_dssim = 0.002; // Target a specific quality
            println!("Finding AQ scale for DSSIM ~{target_dssim} at distance 1.0");

            if let Some((optimal_scale, result)) =
                binary_search_aq_for_dssim(&pixels, width, height, 1.0, target_dssim, 0.0002)
            {
                println!("Found optimal AQ scale: {optimal_scale:.3}");
                println!("  DSSIM: {:.6}", result.encode.dssim);
                println!("  SSIMULACRA2: {:.1}", result.encode.ssimulacra2);
                println!("  File size: {} bytes", result.encode.file_size);
                println!("  BPP: {:.3}", result.encode.bpp);
                println!("  AQ mean: {:.4}", result.aq_mean);
            } else {
                println!("Binary search did not converge");
            }
        }
    }
}
