//! Unified butteraugli debugging tool.
//!
//! # Usage
//!
//! ```bash
//! # Compare two images
//! cargo run --release --example butteraugli_debug -- compare img1.png img2.png
//!
//! # Analyze JPEG compression quality
//! cargo run --release --example butteraugli_debug -- jpeg image.png 90
//!
//! # Test with uniform color
//! cargo run --release --example butteraugli_debug -- uniform 128 128 128
//! ```

use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        return;
    }

    match args[1].as_str() {
        "compare" => cmd_compare(&args[2..]),
        "jpeg" => cmd_jpeg(&args[2..]),
        "uniform" => cmd_uniform(&args[2..]),
        "sweep" => cmd_sweep(&args[2..]),
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_help();
        }
    }
}

fn print_help() {
    println!(
        r#"butteraugli-debug: Perceptual image quality analysis

USAGE:
    cargo run --example butteraugli_debug -- <COMMAND> [OPTIONS]

COMMANDS:
    compare <img1> <img2>     Compare two images with butteraugli
    jpeg <image> [quality]    Analyze JPEG compression at quality level(s)
    uniform <r> <g> <b>       Test with uniform color images
    sweep <image>             Sweep quality levels 50-100

EXAMPLES:
    # Compare original vs compressed
    cargo run --release --example butteraugli_debug -- compare orig.png compressed.png

    # Test JPEG quality at Q90
    cargo run --release --example butteraugli_debug -- jpeg photo.png 90

    # Sweep all quality levels
    cargo run --release --example butteraugli_debug -- sweep photo.png
"#
    );
}

// ============================================================================
// COMPARE: Two images
// ============================================================================

fn cmd_compare(args: &[String]) {
    if args.len() < 2 {
        println!("Usage: compare <image1> <image2>");
        return;
    }

    let (rgb1, w1, h1) = match load_image(&args[0]) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load: {}", args[0]);
            return;
        }
    };

    let (rgb2, w2, h2) = match load_image(&args[1]) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load: {}", args[1]);
            return;
        }
    };

    if w1 != w2 || h1 != h2 {
        eprintln!(
            "Image dimensions don't match: {}x{} vs {}x{}",
            w1, h1, w2, h2
        );
        return;
    }

    println!("=== BUTTERAUGLI COMPARISON ===");
    println!("Image 1: {}", args[0]);
    println!("Image 2: {}", args[1]);
    println!("Size: {}x{}", w1, h1);
    println!();

    // Compute butteraugli
    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, w1, h1, &params);

    println!("Butteraugli score: {:.6}", result.score);
    println!();

    // Quality interpretation
    println!("Quality interpretation:");
    if result.score < 0.5 {
        println!("  Excellent - imperceptible difference");
    } else if result.score < 1.0 {
        println!("  Good - barely noticeable");
    } else if result.score < 2.0 {
        println!("  Acceptable - noticeable on close inspection");
    } else if result.score < 3.0 {
        println!("  Poor - clearly visible artifacts");
    } else {
        println!("  Bad - significant visual degradation");
    }

    // Diffmap analysis
    if let Some(ref diffmap) = result.diffmap {
        analyze_diffmap(diffmap);
    }

    // Pixel-level stats
    analyze_pixels(&rgb1, &rgb2);
}

fn analyze_diffmap(diffmap: &butteraugli_oxide::image::ImageF) {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut histogram = [0usize; 10];

    for y in 0..diffmap.height() {
        for x in 0..diffmap.width() {
            let v = diffmap.get(x, y);
            max = max.max(v);
            sum += v as f64;

            let bucket = (v.min(9.9) as usize).min(9);
            histogram[bucket] += 1;
        }
    }

    let count = (diffmap.width() * diffmap.height()) as f64;
    let mean = sum / count;

    println!("\nDiffmap analysis:");
    println!("  Max: {:.4}", max);
    println!("  Mean: {:.4}", mean);
    println!("  Distribution:");
    for i in 0..10 {
        let pct = histogram[i] as f64 / count * 100.0;
        let bar = "█".repeat((pct / 5.0) as usize);
        println!("    [{}-{}): {:5.1}% {}", i, i + 1, pct, bar);
    }
}

fn analyze_pixels(a: &[u8], b: &[u8]) {
    let mut max_diff = 0u8;
    let mut sum_diff = 0u64;
    let mut histogram = [0usize; 256];

    for i in 0..a.len() {
        let diff = (a[i] as i16 - b[i] as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
        sum_diff += diff as u64;
        histogram[diff as usize] += 1;
    }

    let mean = sum_diff as f64 / a.len() as f64;

    println!("\nPixel difference:");
    println!("  Max: {}", max_diff);
    println!("  Mean: {:.3}", mean);
    println!(
        "  Identical pixels: {} ({:.1}%)",
        histogram[0],
        histogram[0] as f64 / a.len() as f64 * 100.0
    );
}

// ============================================================================
// JPEG: Analyze compression quality
// ============================================================================

fn cmd_jpeg(args: &[String]) {
    if args.is_empty() {
        println!("Usage: jpeg <image.png> [quality]");
        println!("\nTests JPEG compression at the specified quality level(s).");
        return;
    }

    let path = &args[0];
    let quality: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(90);

    let (original, width, height) = match load_image(path) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load: {}", path);
            return;
        }
    };

    println!("=== JPEG QUALITY ANALYSIS ===");
    println!("Image: {} ({}x{})", path, width, height);
    println!("Quality: {}", quality);
    println!();

    // Encode with mozjpeg
    let jpeg_data = encode_mozjpeg(&original, width as u32, height as u32, quality);
    let decoded = decode_jpeg(&jpeg_data);

    if decoded.len() != original.len() {
        eprintln!("Decoded size mismatch");
        return;
    }

    // File size
    let orig_size = original.len();
    let jpeg_size = jpeg_data.len();
    let ratio = orig_size as f64 / jpeg_size as f64;
    let bpp = jpeg_size as f64 * 8.0 / (width * height) as f64;

    println!("Size:");
    println!("  Original: {} bytes", orig_size);
    println!("  JPEG: {} bytes ({:.1}:1 compression)", jpeg_size, ratio);
    println!("  BPP: {:.3}", bpp);

    // Butteraugli
    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&original, &decoded, width, height, &params);

    println!("\nButteraugli score: {:.6}", result.score);

    // Pixel stats
    analyze_pixels(&original, &decoded);
}

// ============================================================================
// SWEEP: Quality sweep
// ============================================================================

fn cmd_sweep(args: &[String]) {
    if args.is_empty() {
        println!("Usage: sweep <image.png>");
        return;
    }

    let path = &args[0];
    let (original, width, height) = match load_image(path) {
        Some(d) => d,
        None => {
            eprintln!("Failed to load: {}", path);
            return;
        }
    };

    println!("=== QUALITY SWEEP ===");
    println!("Image: {} ({}x{})", path, width, height);
    println!();
    println!(
        "{:>4} {:>10} {:>8} {:>10}",
        "Q", "Size", "BPP", "Butteraugli"
    );
    println!("{:-<4} {:-<10} {:-<8} {:-<10}", "", "", "", "");

    for q in (50..=100).step_by(5) {
        let jpeg_data = encode_mozjpeg(&original, width as u32, height as u32, q);
        let decoded = decode_jpeg(&jpeg_data);

        if decoded.len() != original.len() {
            continue;
        }

        let bpp = jpeg_data.len() as f64 * 8.0 / (width * height) as f64;
        let params = ButteraugliParams::default();
        let result = compute_butteraugli(&original, &decoded, width, height, &params);

        println!(
            "{:>4} {:>10} {:>8.3} {:>10.4}",
            q,
            jpeg_data.len(),
            bpp,
            result.score
        );
    }
}

// ============================================================================
// UNIFORM: Test with uniform color
// ============================================================================

fn cmd_uniform(args: &[String]) {
    let r: u8 = args.get(0).and_then(|s| s.parse().ok()).unwrap_or(128);
    let g: u8 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let b: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    println!("=== UNIFORM COLOR TEST ===");
    println!("Color: RGB({}, {}, {})", r, g, b);
    println!();

    // Create 64x64 uniform image
    let size = 64;
    let img1: Vec<u8> = (0..size * size).flat_map(|_| [r, g, b]).collect();

    // Create slightly different image (±1 per channel)
    let img2: Vec<u8> = (0..size * size)
        .flat_map(|_| {
            [
                r.saturating_add(1),
                g.saturating_add(1),
                b.saturating_add(1),
            ]
        })
        .collect();

    // Compare identical
    let params = ButteraugliParams::default();
    let result1 = compute_butteraugli(&img1, &img1, size, size, &params);
    println!("Identical images: {:.6}", result1.score);

    // Compare ±1
    let result2 = compute_butteraugli(&img1, &img2, size, size, &params);
    println!("±1 difference: {:.6}", result2.score);

    // JPEG roundtrip
    for q in [90, 95, 100] {
        let jpeg = encode_mozjpeg(&img1, size as u32, size as u32, q);
        let decoded = decode_jpeg(&jpeg);

        if decoded.len() == img1.len() {
            let result = compute_butteraugli(&img1, &decoded, size, size, &params);
            let max_diff = img1
                .iter()
                .zip(decoded.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs())
                .max()
                .unwrap_or(0);
            println!(
                "JPEG Q{}: butteraugli={:.6}, max_pixel_diff={}",
                q, result.score, max_diff
            );
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn load_image(path: &str) -> Option<(Vec<u8>, usize, usize)> {
    let p = Path::new(path);

    // Try PNG
    if let Ok(file) = fs::File::open(p) {
        let decoder = png::Decoder::new(file);
        if let Ok(mut reader) = decoder.read_info() {
            let mut buf = vec![0; reader.output_buffer_size()];
            if let Ok(info) = reader.next_frame(&mut buf) {
                let (w, h) = (info.width as usize, info.height as usize);
                let rgb = match info.color_type {
                    png::ColorType::Rgb => buf[..w * h * 3].to_vec(),
                    png::ColorType::Rgba => buf[..w * h * 4]
                        .chunks(4)
                        .flat_map(|c| [c[0], c[1], c[2]])
                        .collect(),
                    png::ColorType::Grayscale => {
                        buf[..w * h].iter().flat_map(|&g| [g, g, g]).collect()
                    }
                    _ => return None,
                };
                return Some((rgb, w, h));
            }
        }
    }

    // Try JPEG
    if let Ok(data) = fs::read(p) {
        let mut decoder = jpeg_decoder::Decoder::new(&data[..]);
        if let Ok(pixels) = decoder.decode() {
            let info = decoder.info().unwrap();
            return Some((pixels, info.width as usize, info.height as usize));
        }
    }

    None
}

fn encode_mozjpeg(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    use std::io::Cursor;
    let mut output = Vec::new();
    let mut comp = mozjpeg::Compress::new(mozjpeg::ColorSpace::JCS_RGB);
    comp.set_size(width as usize, height as usize);
    comp.set_quality(quality as f32);
    let mut started = comp
        .start_compress(Cursor::new(&mut output))
        .expect("start");
    let row_stride = width as usize * 3;
    for row in rgb.chunks(row_stride) {
        started.write_scanlines(row).expect("write");
    }
    started.finish().expect("finish");
    output
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    let mut decoder = jpeg_decoder::Decoder::new(data);
    decoder.decode().unwrap_or_default()
}
