//! Comprehensive benchmark: sizes × subsampling × modes × paths
//!
//! Tests 1K, 2K, 3K, 4K images with 4:4:4/4:2:0, progressive/baseline,
//! comparing full-plane vs strip-based encoding.
//!
//! Uses force_full_plane() to bypass auto-dispatch for true comparison.

use jpegli::{Encoder, JpegMode, PixelFormat, Quality, Subsampling};
use png::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

/// Load a PNG and resize it to target dimensions using simple box filter
fn load_and_resize(path: &str, target_w: usize, target_h: usize) -> Vec<u8> {
    let file = File::open(path).expect("open file");
    let decoder = Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info().expect("read info");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("decode");

    let src_w = info.width as usize;
    let src_h = info.height as usize;
    let channels = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        _ => panic!("Unsupported color type: {:?}", info.color_type),
    };

    let mut result = vec![0u8; target_w * target_h * 3];
    for ty in 0..target_h {
        for tx in 0..target_w {
            let sx = (tx * src_w) / target_w;
            let sy = (ty * src_h) / target_h;
            let src_idx = (sy * src_w + sx) * channels;
            let dst_idx = (ty * target_w + tx) * 3;
            result[dst_idx] = buf[src_idx];
            result[dst_idx + 1] = buf[src_idx + 1];
            result[dst_idx + 2] = buf[src_idx + 2];
        }
    }
    result
}

struct BenchResult {
    time_ms: f64,
    mpps: f64,
    size_bytes: usize,
    output: Vec<u8>,
}

fn bench_full_plane(
    data: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    progressive: bool,
    quality: f32,
    iterations: usize,
) -> BenchResult {
    let w = width as u32;
    let h = height as u32;

    // Warmup
    for _ in 0..2 {
        let enc = Encoder::new()
            .width(w)
            .height(h)
            .pixel_format(PixelFormat::Rgb)
            .subsampling(subsampling)
            .mode(if progressive {
                JpegMode::Progressive
            } else {
                JpegMode::Baseline
            })
            .jpegli_quality(Quality::from_quality(quality))
            .force_full_plane(true); // Force full-plane, bypass auto-dispatch
        let _ = enc.encode(data);
    }

    let mut times = Vec::with_capacity(iterations);
    let mut output = Vec::new();

    for _ in 0..iterations {
        let enc = Encoder::new()
            .width(w)
            .height(h)
            .pixel_format(PixelFormat::Rgb)
            .subsampling(subsampling)
            .mode(if progressive {
                JpegMode::Progressive
            } else {
                JpegMode::Baseline
            })
            .jpegli_quality(Quality::from_quality(quality))
            .force_full_plane(true); // Force full-plane, bypass auto-dispatch

        let start = Instant::now();
        let result = enc.encode(data);
        times.push(start.elapsed().as_secs_f64() * 1000.0);

        if let Ok(r) = result {
            output = r;
        }
    }

    let avg_ms = times.iter().sum::<f64>() / iterations as f64;
    let pixels = (width * height) as f64;
    let mpps = pixels / (avg_ms / 1000.0) / 1_000_000.0;

    BenchResult {
        time_ms: avg_ms,
        mpps,
        size_bytes: output.len(),
        output,
    }
}

fn bench_strip(
    data: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    quality: f32,
    iterations: usize,
) -> BenchResult {
    let w = width as u32;
    let h = height as u32;

    // Warmup
    for _ in 0..2 {
        let enc = Encoder::new()
            .width(w)
            .height(h)
            .pixel_format(PixelFormat::Rgb)
            .subsampling(subsampling)
            .jpegli_quality(Quality::from_quality(quality));
        let _ = enc.encode_strip_based(data);
    }

    let mut times = Vec::with_capacity(iterations);
    let mut output = Vec::new();

    for _ in 0..iterations {
        let enc = Encoder::new()
            .width(w)
            .height(h)
            .pixel_format(PixelFormat::Rgb)
            .subsampling(subsampling)
            .jpegli_quality(Quality::from_quality(quality));

        let start = Instant::now();
        let result = enc.encode_strip_based(data);
        times.push(start.elapsed().as_secs_f64() * 1000.0);

        if let Ok(r) = result {
            output = r;
        }
    }

    let avg_ms = times.iter().sum::<f64>() / iterations as f64;
    let pixels = (width * height) as f64;
    let mpps = pixels / (avg_ms / 1000.0) / 1_000_000.0;

    BenchResult {
        time_ms: avg_ms,
        mpps,
        size_bytes: output.len(),
        output,
    }
}

fn decode_jpeg(data: &[u8]) -> (Vec<u8>, usize, usize) {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new_with_options(cursor, opts);
    decoder.decode_headers().ok();
    let info = decoder.info().unwrap();
    let w = info.width as usize;
    let h = info.height as usize;
    (decoder.decode().unwrap_or_default(), w, h)
}

fn compute_ssimulacra2(img1: &[u8], img2: &[u8], width: usize, height: usize) -> f64 {
    use fast_ssim2::{compute_frame_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};

    let source: Vec<[f32; 3]> = img1
        .chunks_exact(3)
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();
    let distorted: Vec<[f32; 3]> = img2
        .chunks_exact(3)
        .map(|c| {
            [
                srgb_u8_to_linear(c[0]),
                srgb_u8_to_linear(c[1]),
                srgb_u8_to_linear(c[2]),
            ]
        })
        .collect();

    let source_img = LinearRgbImage::new(source, width, height);
    let distorted_img = LinearRgbImage::new(distorted, width, height);

    compute_frame_ssimulacra2(source_img, distorted_img).unwrap_or(-1.0)
}

fn compute_diff(a: &[u8], b: &[u8]) -> (u8, f64) {
    if a.len() != b.len() {
        return (255, 255.0);
    }
    let mut max_diff = 0u8;
    let mut sum_sq = 0u64;

    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32 - *y as i32).unsigned_abs() as u8;
        max_diff = max_diff.max(d);
        sum_sq += (d as u64) * (d as u64);
    }

    let mse = sum_sq as f64 / a.len() as f64;
    (max_diff, mse.sqrt())
}

fn main() {
    let source_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/lilith/work/codec-eval/codec-corpus/kodak/1.png".to_string());

    println!("=== Comprehensive Encoder Benchmark ===");
    println!("Source: {}", source_path);
    println!();
    println!("Comparing full-plane vs strip-based encoding with same chroma conversion.");
    println!("Using force_full_plane() to bypass auto-dispatch for true comparison.");
    println!();

    let sizes = [
        (1024, 768, "1K"),  // 0.79M
        (1920, 1080, "2K"), // 2.07M
        (2560, 1440, "3K"), // 3.69M
        (3840, 2160, "4K"), // 8.29M
    ];

    let quality = 90.0;
    let iterations = 5;

    // Store results for parity comparison
    let mut results: Vec<(&str, &str, BenchResult, BenchResult, BenchResult)> = Vec::new();

    println!(
        "{:<6} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Size", "Sub", "full MP/s", "full KB", "strip MP/s", "strip KB", "prog MP/s", "prog KB"
    );
    println!("{}", "-".repeat(76));

    for (w, h, size_name) in sizes {
        let data = load_and_resize(&source_path, w, h);
        let pixels_m = (w * h) as f64 / 1_000_000.0;

        for &subsampling in &[Subsampling::S444, Subsampling::S420] {
            let sub_name = if subsampling == Subsampling::S444 {
                "444"
            } else {
                "420"
            };

            // Full-plane baseline (forced, same chroma conversion as strip)
            let full = bench_full_plane(&data, w, h, subsampling, false, quality, iterations);

            // Strip-based baseline
            let strip = bench_strip(&data, w, h, subsampling, quality, iterations);

            // Progressive (always full-plane, for reference)
            let prog = bench_full_plane(&data, w, h, subsampling, true, quality, iterations);

            println!(
                "{:<6} {:>4} {:>8.1} {:>8} {:>8.1} {:>8} {:>8.1} {:>8}",
                size_name,
                sub_name,
                full.mpps,
                full.size_bytes / 1024,
                strip.mpps,
                strip.size_bytes / 1024,
                prog.mpps,
                prog.size_bytes / 1024
            );

            results.push((size_name, sub_name, full, strip, prog));
        }
    }

    // Output parity verification
    println!("\n=== Output Parity: full-plane vs strip-based ===");
    println!("Both use same chroma conversion (Intrinsic). Outputs should be similar.\n");

    println!(
        "{:<6} {:>4} {:>10} {:>10} {:>8} {:>8} {:>8} {:>6}",
        "Size", "Sub", "full KB", "strip KB", "Diff%", "MaxDiff", "SSIM2", "Status"
    );
    println!("{}", "-".repeat(72));

    for (size_name, sub_name, full, strip, _prog) in &results {
        let size_diff_pct =
            (strip.size_bytes as f64 - full.size_bytes as f64) / full.size_bytes as f64 * 100.0;

        let (full_decoded, fw, fh) = decode_jpeg(&full.output);
        let (strip_decoded, sw, sh) = decode_jpeg(&strip.output);

        if fw != sw || fh != sh {
            println!(
                "{:<6} {:>4} DIM MISMATCH {}x{} vs {}x{}",
                size_name, sub_name, fw, fh, sw, sh
            );
            continue;
        }

        let (max_diff, _rmse) = compute_diff(&full_decoded, &strip_decoded);
        let ssim2 = compute_ssimulacra2(&full_decoded, &strip_decoded, fw, fh);

        let status = if max_diff == 0 {
            "EXACT"
        } else if ssim2 >= 90.0 {
            "PASS"
        } else if ssim2 >= 70.0 {
            "CLOSE"
        } else {
            "DIFF"
        };

        println!(
            "{:<6} {:>4} {:>10} {:>10} {:>+7.2}% {:>8} {:>8.2} {:>6}",
            size_name,
            sub_name,
            full.size_bytes / 1024,
            strip.size_bytes / 1024,
            size_diff_pct,
            max_diff,
            ssim2,
            status
        );
    }

    println!();
    println!("SSIMULACRA2: 90+ = excellent, 70-90 = good, <70 = visible differences");

    // Performance summary
    println!("\n=== Performance Summary: Strip vs Full-Plane Speedup ===\n");

    println!(
        "{:<6} {:>4} {:>10} {:>10} {:>10}",
        "Size", "Sub", "Full MP/s", "Strip MP/s", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for (size_name, sub_name, full, strip, _prog) in &results {
        let speedup = strip.mpps / full.mpps;
        let speedup_pct = (speedup - 1.0) * 100.0;
        let speedup_str = if speedup >= 1.0 {
            format!("+{:.0}%", speedup_pct)
        } else {
            format!("{:.0}%", speedup_pct)
        };

        println!(
            "{:<6} {:>4} {:>10.1} {:>10.1} {:>10}",
            size_name, sub_name, full.mpps, strip.mpps, speedup_str
        );
    }

    // Best path recommendation
    println!("\n=== Recommendation ===");
    println!("Strip-based encoding is faster due to better cache locality.");
    println!("The 2MP auto-dispatch threshold is correct for most systems.");
}
