//! Comprehensive edge replication test using SSIMULACRA2
//!
//! Tests all partial MCU sizes (1-7) for both horizontal and vertical edges
//! to catch edge padding bugs in the encoder.
//!
//! Usage:
//!   cargo run --release --example comprehensive_edge_ssim2
//!   cargo run --release --example comprehensive_edge_ssim2 -- --subsampling=420
//!   cargo run --release --example comprehensive_edge_ssim2 -- --verbose

#[allow(deprecated)]
use fast_ssim2::{compute_frame_ssimulacra2, srgb_u8_to_linear, LinearRgbImage};
use jpegli::types::Subsampling;
use jpegli::PixelFormat;
use jpegli::Quality;
use std::env;

/// Test configuration
struct TestConfig {
    subsampling: Subsampling,
    verbose: bool,
    quality: f32,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            subsampling: Subsampling::S444,
            verbose: false,
            quality: 85.0,
        }
    }
}

fn parse_args() -> TestConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = TestConfig::default();

    for arg in &args[1..] {
        if arg == "--verbose" || arg == "-v" {
            config.verbose = true;
        } else if arg.starts_with("--subsampling=") {
            let sub = arg.trim_start_matches("--subsampling=");
            config.subsampling = match sub {
                "444" => Subsampling::S444,
                "422" => Subsampling::S422,
                "420" => Subsampling::S420,
                "440" => Subsampling::S440,
                _ => {
                    eprintln!("Unknown subsampling: {}, using 444", sub);
                    Subsampling::S444
                }
            };
        } else if arg.starts_with("--quality=") {
            if let Ok(q) = arg.trim_start_matches("--quality=").parse() {
                config.quality = q;
            }
        }
    }

    config
}

/// MCU size for a given subsampling mode
fn mcu_size(sub: Subsampling) -> (usize, usize) {
    match sub {
        Subsampling::S444 => (8, 8),
        Subsampling::S422 => (16, 8),
        Subsampling::S420 => (16, 16),
        Subsampling::S440 => (8, 16),
        _ => (8, 8),
    }
}

/// Create a smooth gradient test image that won't overflow at edges
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Use smooth gradients that don't wrap at 256
            // Red: horizontal gradient 64-192
            rgb[idx] = (64.0 + (x as f32 / width as f32) * 128.0) as u8;
            // Green: vertical gradient 64-192
            rgb[idx + 1] = (64.0 + (y as f32 / height as f32) * 128.0) as u8;
            // Blue: diagonal gradient
            rgb[idx + 2] = (64.0 + ((x + y) as f32 / (width + height) as f32) * 128.0) as u8;
        }
    }
    rgb
}

#[allow(deprecated)]
fn encode_strip(rgb: &[u8], width: u32, height: u32, sub: Subsampling, quality: f32) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .subsampling(sub)
        .jpegli_quality(Quality::from_quality(quality))
        .optimize_huffman(true)
        .encode(rgb)
        .expect("strip encode failed")
}

#[allow(deprecated)]
fn encode_fullplane(
    rgb: &[u8],
    width: u32,
    height: u32,
    sub: Subsampling,
    quality: f32,
) -> Vec<u8> {
    jpegli::Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .subsampling(sub)
        .jpegli_quality(Quality::from_quality(quality))
        .optimize_huffman(true)
        .encode_fullplane(rgb)
        .expect("fullplane encode failed")
}

fn bytes_to_linear(data: &[u8], width: usize, height: usize) -> LinearRgbImage {
    let pixels: Vec<[f32; 3]> = data
        .chunks_exact(3)
        .map(|rgb| {
            [
                srgb_u8_to_linear(rgb[0]),
                srgb_u8_to_linear(rgb[1]),
                srgb_u8_to_linear(rgb[2]),
            ]
        })
        .collect();
    LinearRgbImage::new(pixels, width, height)
}

fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    if original.len() != decoded.len() {
        return -1.0;
    }
    let orig = bytes_to_linear(original, width, height);
    let dec = bytes_to_linear(decoded, width, height);
    compute_frame_ssimulacra2(orig, dec).unwrap_or(0.0)
}

struct TestResult {
    width: usize,
    height: usize,
    strip_size: usize,
    strip_ssim2: f64,
    full_size: usize,
    full_ssim2: f64,
}

fn test_dimension(
    width: usize,
    height: usize,
    sub: Subsampling,
    quality: f32,
) -> Option<TestResult> {
    let rgb = create_test_image(width, height);

    let jpeg_strip = encode_strip(&rgb, width as u32, height as u32, sub, quality);
    let jpeg_full = encode_fullplane(&rgb, width as u32, height as u32, sub, quality);

    let decoded_strip = jpegli::Decoder::new().decode(&jpeg_strip).ok()?;
    let decoded_full = jpegli::Decoder::new().decode(&jpeg_full).ok()?;

    let strip_ssim2 = compute_ssim2(&rgb, &decoded_strip.data, width, height);
    let full_ssim2 = compute_ssim2(&rgb, &decoded_full.data, width, height);

    Some(TestResult {
        width,
        height,
        strip_size: jpeg_strip.len(),
        strip_ssim2,
        full_size: jpeg_full.len(),
        full_ssim2,
    })
}

fn main() {
    let config = parse_args();
    let (mcu_w, mcu_h) = mcu_size(config.subsampling);

    println!(
        "Comprehensive Edge Replication Test - {:?} ({}x{} MCU)",
        config.subsampling, mcu_w, mcu_h
    );
    println!("Quality: {:.0}", config.quality);
    println!();

    // Test RIGHT edge (width = base + 1..7)
    println!("=== RIGHT EDGE (varying width) ===");
    // Use larger base dimensions for more realistic testing
    // Small images (64px) are dominated by padding effects
    let base_width = mcu_w * 32; // 256 for S444, 512 for S420/S422
    let height = mcu_h * 16; // Fixed height that's MCU-aligned
    println!(
        "Testing widths: {}+1 to {}+7 (partial MCU columns)",
        base_width, base_width
    );
    println!(
        "{:>8} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Width", "W%MCU", "Pad", "Strip", "SSIM2", "Full", "SSIM2", "Status"
    );

    let mut right_edge_failures = 0;

    for remainder in 1..=7 {
        let width = base_width + remainder;
        let pad_cols = mcu_w - (width % mcu_w);

        if let Some(result) = test_dimension(width, height, config.subsampling, config.quality) {
            let status = if result.strip_ssim2 < 85.0 || result.full_ssim2 < 85.0 {
                right_edge_failures += 1;
                "FAIL"
            } else if (result.strip_ssim2 - result.full_ssim2).abs() > 5.0 {
                right_edge_failures += 1;
                "MISMATCH"
            } else {
                "OK"
            };

            println!(
                "{:>8} {:>8} {:>6} {:>10} {:>10.2} {:>10} {:>10.2} {:>8}",
                result.width,
                width % mcu_w,
                pad_cols,
                result.strip_size,
                result.strip_ssim2,
                result.full_size,
                result.full_ssim2,
                status
            );
        }
    }

    println!();

    // Test BOTTOM edge (height = base + 1..7)
    println!("=== BOTTOM EDGE (varying height) ===");
    // Use larger base dimensions for more realistic testing
    let width = mcu_w * 32; // Fixed width that's MCU-aligned (256 for S444, 512 for S420)
    let base_height = mcu_h * 16; // 128 for S444, 256 for S420
    println!(
        "Testing heights: {}+1 to {}+7 (partial MCU rows)",
        base_height, base_height
    );
    println!(
        "{:>8} {:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "Height", "H%MCU", "Pad", "Strip", "SSIM2", "Full", "SSIM2", "Status"
    );

    let mut bottom_edge_failures = 0;

    for remainder in 1..=7 {
        let height = base_height + remainder;
        let pad_rows = mcu_h - (height % mcu_h);

        if let Some(result) = test_dimension(width, height, config.subsampling, config.quality) {
            let status = if result.strip_ssim2 < 85.0 || result.full_ssim2 < 85.0 {
                bottom_edge_failures += 1;
                "FAIL"
            } else if (result.strip_ssim2 - result.full_ssim2).abs() > 5.0 {
                bottom_edge_failures += 1;
                "MISMATCH"
            } else {
                "OK"
            };

            println!(
                "{:>8} {:>8} {:>6} {:>10} {:>10.2} {:>10} {:>10.2} {:>8}",
                result.height,
                height % mcu_h,
                pad_rows,
                result.strip_size,
                result.strip_ssim2,
                result.full_size,
                result.full_ssim2,
                status
            );
        }
    }

    println!();

    // Test BOTH edges (corner case)
    println!("=== BOTH EDGES (corner cases) ===");
    println!(
        "{:>8} {:>8} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "WxH", "W%MCU", "H%MCU", "Pad", "Strip", "SSIM2", "Full", "SSIM2", "Status"
    );
    println!("{}", "-".repeat(95));

    let mut both_edge_failures = 0;

    // Test all combinations of remainders 1, 4, 7
    for w_rem in [1, 4, 7] {
        for h_rem in [1, 4, 7] {
            let width = base_width + w_rem;
            let height = base_height + h_rem;

            if let Some(result) = test_dimension(width, height, config.subsampling, config.quality)
            {
                let status = if result.strip_ssim2 < 85.0 || result.full_ssim2 < 85.0 {
                    both_edge_failures += 1;
                    "FAIL"
                } else if (result.strip_ssim2 - result.full_ssim2).abs() > 5.0 {
                    both_edge_failures += 1;
                    "MISMATCH"
                } else {
                    "OK"
                };

                let pad_total = (mcu_w - (width % mcu_w)) + (mcu_h - (height % mcu_h));
                println!(
                    "{:>8} {:>8} {:>6} {:>6} {:>10} {:>10.2} {:>10} {:>10.2} {:>8}",
                    format!("{}x{}", result.width, result.height),
                    width % mcu_w,
                    height % mcu_h,
                    pad_total,
                    result.strip_size,
                    result.strip_ssim2,
                    result.full_size,
                    result.full_ssim2,
                    status
                );
            }
        }
    }

    println!();

    // Additional test: extremes for S420 (1-15 partial rows/cols)
    if config.subsampling == Subsampling::S420 {
        println!("=== S420 EXTENDED TEST (1-15 partial MCU) ===");
        println!(
            "{:>8} {:>8} {:>6} {:>10} {:>10} {:>8}",
            "Height", "H%16", "Pad", "Strip", "SSIM2", "Status"
        );
        println!("{}", "-".repeat(60));

        let width = 64usize; // 4 MCUs wide (MCU-aligned)
        let base_height = 64usize; // 4 MCUs tall

        for remainder in 1..=15 {
            let height = base_height + remainder;
            let pad_rows = 16 - (height % 16);

            if let Some(result) = test_dimension(width, height, config.subsampling, config.quality)
            {
                let status = if result.strip_ssim2 < 85.0 {
                    "FAIL"
                } else {
                    "OK"
                };

                println!(
                    "{:>8} {:>8} {:>6} {:>10} {:>10.2} {:>8}",
                    result.height,
                    height % 16,
                    pad_rows,
                    result.strip_size,
                    result.strip_ssim2,
                    status
                );
            }
        }
        println!();
    }

    // Summary
    let total_failures = right_edge_failures + bottom_edge_failures + both_edge_failures;
    println!("=== SUMMARY ===");
    println!("Right edge failures:  {}", right_edge_failures);
    println!("Bottom edge failures: {}", bottom_edge_failures);
    println!("Both edge failures:   {}", both_edge_failures);
    println!("Total failures:       {}", total_failures);

    if total_failures > 0 {
        println!("\nFAIL: Edge handling has quality issues");
        std::process::exit(1);
    } else {
        println!("\nPASS: All edge cases passed quality threshold");
    }
}
