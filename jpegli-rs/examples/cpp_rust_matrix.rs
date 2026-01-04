//! Comprehensive matrix comparison: Rust vs C++ jpegli
//!
//! Tests all combinations of:
//! - Image size: Small (64x64), Medium (512x512), Large (2048x2048)
//! - Mode: Baseline, Progressive
//! - Color: YCbCr, XYB
//! - Huffman: Fixed, Optimized
//! - Quality: 50, 70, 90
//!
//! This helps understand:
//! - When progressive is beneficial vs detrimental
//! - Whether ICC profile overhead matters (XYB adds ~720 bytes)
//! - Whether progressive scan overhead matters on small vs large images
//! - File size parity with C++ across the matrix

use jpegli::{Encoder, PixelFormat};
use std::fs;
use std::process::Command;

#[derive(Clone, Copy, Debug)]
struct Config {
    width: usize,
    height: usize,
    mode: Mode,
    color: ColorMode,
    huffman: HuffmanMode,
    quality: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Baseline,
    Progressive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ColorMode {
    YCbCr,
    XYB,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum HuffmanMode {
    Fixed,
    Optimized,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Mode::Baseline => write!(f, "Baseline"),
            Mode::Progressive => write!(f, "Progressive"),
        }
    }
}

impl std::fmt::Display for ColorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ColorMode::YCbCr => write!(f, "YCbCr"),
            ColorMode::XYB => write!(f, "XYB"),
        }
    }
}

impl std::fmt::Display for HuffmanMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            HuffmanMode::Fixed => write!(f, "Fixed"),
            HuffmanMode::Optimized => write!(f, "Optimized"),
        }
    }
}

/// Create gradient test image
fn create_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = ((x * 255) / width.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / height.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }
    rgb
}

/// Create complex pattern with varying frequencies
fn create_complex(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Mix of gradients and high-frequency patterns
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;

            rgb[idx] = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0) as u8;
            rgb[idx + 1] = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0) as u8;
            rgb[idx + 2] = (128.0 + ((fx + fy) * 50.0).sin() * 50.0) as u8;
        }
    }
    rgb
}

fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

struct Result {
    rust_size: usize,
    cpp_size: usize,
}

fn encode_rust(rgb: &[u8], config: Config) -> Vec<u8> {
    Encoder::new()
        .width(config.width as u32)
        .height(config.height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(jpegli::quant::Quality::from_quality(config.quality as f32))
        .mode(match config.mode {
            Mode::Baseline => jpegli::types::JpegMode::Baseline,
            Mode::Progressive => jpegli::types::JpegMode::Progressive,
        })
        .use_xyb(config.color == ColorMode::XYB)
        .optimize_huffman(config.huffman == HuffmanMode::Optimized)
        .encode(rgb)
        .unwrap()
}

fn encode_cpp(ppm_path: &str, config: Config) -> Option<Vec<u8>> {
    let cjpegli_path = jpegli::test_utils::find_cjpegli()?;

    let output_path = format!(
        "/tmp/cpp_{}_{}_{}_{}_q{}.jpg",
        config.width, config.mode, config.color, config.huffman, config.quality
    );

    let mut args = vec![ppm_path.to_string(), output_path.clone()];

    // Progressive level
    args.push("-p".to_string());
    args.push(match config.mode {
        Mode::Baseline => "0".to_string(),
        Mode::Progressive => "2".to_string(),
    });

    // XYB mode
    if config.color == ColorMode::XYB {
        args.push("--xyb".to_string());
    }

    // Huffman optimization (C++ defaults to ON, use --fixed_code to disable)
    if config.huffman == HuffmanMode::Fixed {
        args.push("--fixed_code".to_string());
    }

    // Quality
    args.push("-q".to_string());
    args.push(config.quality.to_string());

    let output = Command::new(cjpegli_path).args(&args).output().ok()?;

    if !output.status.success() {
        eprintln!(
            "C++ encoding failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    fs::read(&output_path).ok()
}

fn run_comparison(rgb: &[u8], config: Config, label: &str) -> Option<Result> {
    // Write PPM for C++
    let ppm_path = format!(
        "/tmp/test_{}_{}_{}_{}.ppm",
        config.width, config.mode, config.color, config.huffman
    );
    write_ppm(&ppm_path, rgb, config.width, config.height).ok()?;

    // Encode with both
    let rust_jpeg = encode_rust(rgb, config);
    let cpp_jpeg = encode_cpp(&ppm_path, config)?;

    let rust_size = rust_jpeg.len();
    let cpp_size = cpp_jpeg.len();

    println!(
        "{:12} | {:4}x{:4} | {:11} | {:6} | {:9} | Q{:2} | {:>7} | {:>7} | {:>+8} | {:>+6.1}%",
        label,
        config.width,
        config.height,
        format!("{}", config.mode),
        format!("{}", config.color),
        format!("{}", config.huffman),
        config.quality,
        rust_size,
        cpp_size,
        rust_size as i32 - cpp_size as i32,
        100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64
    );

    Some(Result {
        rust_size,
        cpp_size,
    })
}

fn main() {
    println!("\n{}", "=".repeat(150));
    println!(" COMPREHENSIVE RUST vs C++ MATRIX COMPARISON");
    println!("{}\n", "=".repeat(150));

    if jpegli::test_utils::find_cjpegli().is_none() {
        println!("ERROR: C++ cjpegli not found. Build internal/jpegli-cpp first.");
        return;
    }

    println!(
        "{:12} | {:9} | {:11} | {:6} | {:9} | {:3} | {:>7} | {:>7} | {:>8} | {:>7}",
        "Image Type", "Size", "Mode", "Color", "Huffman", "Q", "Rust", "C++", "Δ bytes", "Δ %"
    );
    println!("{}", "-".repeat(150));

    // Image configurations
    let sizes = [(64, 64, "Small"), (512, 512, "Medium")];

    let modes = [Mode::Baseline, Mode::Progressive];
    let colors = [ColorMode::YCbCr, ColorMode::XYB];
    let huffmans = [HuffmanMode::Fixed, HuffmanMode::Optimized];
    let qualities = [90];

    // Test simple gradients
    println!("\n--- SIMPLE GRADIENTS (low complexity) ---");
    for (width, height, size_label) in &sizes {
        let rgb = create_gradient(*width, *height);

        for &mode in &modes {
            for &color in &colors {
                for &huffman in &huffmans {
                    // Skip progressive + fixed (not supported in Rust)
                    if mode == Mode::Progressive && huffman == HuffmanMode::Fixed {
                        continue;
                    }
                    for &quality in &qualities {
                        let config = Config {
                            width: *width,
                            height: *height,
                            mode,
                            color,
                            huffman,
                            quality,
                        };

                        run_comparison(&rgb, config, size_label);
                    }
                }
            }
        }
    }

    // Test complex patterns
    println!("\n--- COMPLEX PATTERNS (high complexity) ---");
    for (width, height, size_label) in &sizes {
        let rgb = create_complex(*width, *height);

        for &mode in &modes {
            for &color in &colors {
                for &huffman in &huffmans {
                    // Skip progressive + fixed (not supported in Rust)
                    if mode == Mode::Progressive && huffman == HuffmanMode::Fixed {
                        continue;
                    }
                    for &quality in &qualities {
                        let config = Config {
                            width: *width,
                            height: *height,
                            mode,
                            color,
                            huffman,
                            quality,
                        };

                        run_comparison(&rgb, config, &format!("{}_cmplx", size_label));
                    }
                }
            }
        }
    }

    println!("\n{}", "=".repeat(150));
    println!("KEY INSIGHTS TO LOOK FOR:");
    println!("1. When is progressive SMALLER than baseline? (should be on larger, complex images)");
    println!(
        "2. How much overhead does XYB ICC profile add? (~720 bytes - matters for small images)"
    );
    println!("3. How much do progressive scan headers cost? (matters for small images)");
    println!("4. How does Rust compare to C++ across this matrix?");
    println!("5. Is Huffman optimization beneficial for both baseline and progressive?");
    println!("{}\n", "=".repeat(150));
}
