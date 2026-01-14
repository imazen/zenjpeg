//! File size and quality parity comparison: Rust jpegli-rs vs C++ cjpegli
//!
//! Compares output file sizes and SSIMULACRA2 quality scores.
//! Does NOT compare timing (unfair: subprocess vs library call).
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings (512, 2K, 4K synthetic images)
//! cargo run --release --example cpp_parity_matrix
//!
//! # Run with a specific image
//! cargo run --release --example cpp_parity_matrix -- path/to/image.png
//!
//! # Save results to CSV
//! cargo run --release --example cpp_parity_matrix -- --csv results.csv
//! ```

use enough::Unstoppable;
use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use jpegli::decoder::{decode_jpeg_with_icc, JpegMode, Subsampling};
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use jpegli::test_utils::find_cjpegli;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

const SYNTHETIC_SIZES: &[(u32, u32, &str)] =
    &[(512, 512, "512"), (2048, 2048, "2K"), (4096, 4096, "4K")];

// ============================================================================
// Configuration Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ScanMode {
    Baseline,
    Progressive,
}

impl ScanMode {
    fn name(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Progressive => "progressive",
        }
    }

    fn to_jpegli(&self) -> JpegMode {
        match self {
            Self::Baseline => JpegMode::Baseline,
            Self::Progressive => JpegMode::Progressive,
        }
    }

    fn cpp_args(&self) -> Vec<&'static str> {
        match self {
            Self::Baseline => vec!["-p", "0"],
            Self::Progressive => vec!["-p", "2"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HuffmanMode {
    Fixed,
    Optimized,
}

impl HuffmanMode {
    fn name(&self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Optimized => "opt",
        }
    }

    fn cpp_args(&self) -> Vec<&'static str> {
        match self {
            Self::Fixed => vec!["--fixed_code"],
            Self::Optimized => vec![],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ChromaSampling {
    S444,
    S420,
}

impl ChromaSampling {
    fn name(&self) -> &'static str {
        match self {
            Self::S444 => "444",
            Self::S420 => "420",
        }
    }

    fn to_jpegli(&self) -> Subsampling {
        match self {
            Self::S444 => Subsampling::S444,
            Self::S420 => Subsampling::S420,
        }
    }

    fn cpp_args(&self) -> Vec<&'static str> {
        match self {
            Self::S444 => vec!["--chroma_subsampling=444"],
            Self::S420 => vec!["--chroma_subsampling=420"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ColorMode {
    YCbCr,
    Xyb,
}

impl ColorMode {
    fn name(&self) -> &'static str {
        match self {
            Self::YCbCr => "ycbcr",
            Self::Xyb => "xyb",
        }
    }

    fn cpp_args(&self) -> Vec<&'static str> {
        match self {
            Self::YCbCr => vec![],
            Self::Xyb => vec!["--xyb"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Config {
    scan: ScanMode,
    huffman: HuffmanMode,
    chroma: ChromaSampling,
    color: ColorMode,
}

impl Config {
    fn name(&self) -> String {
        format!(
            "{}-{}-{}-{}",
            self.scan.name(),
            self.huffman.name(),
            self.chroma.name(),
            self.color.name()
        )
    }

    fn short_name(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            match self.color {
                ColorMode::YCbCr => "YUV",
                ColorMode::Xyb => "XYB",
            },
            match self.scan {
                ScanMode::Baseline => "SEQ",
                ScanMode::Progressive => "PRO",
            },
            match self.huffman {
                HuffmanMode::Fixed => "FIX",
                HuffmanMode::Optimized => "OPT",
            },
            self.chroma.name(),
        )
    }
}

// ============================================================================
// Parity Results
// ============================================================================

#[derive(Debug, Clone)]
struct ParityResult {
    rust_size: usize,
    cpp_size: usize,
    rust_ssim2: f64,
    cpp_ssim2: f64,
    max_pixel_diff: u8,
}

impl ParityResult {
    fn size_diff_pct(&self) -> f64 {
        (self.rust_size as f64 - self.cpp_size as f64) / self.cpp_size as f64 * 100.0
    }

    fn ssim2_diff(&self) -> f64 {
        self.rust_ssim2 - self.cpp_ssim2
    }
}

// ============================================================================
// Verification Functions
// ============================================================================

fn decode_jpeg(data: &[u8], color: ColorMode) -> Vec<u8> {
    match color {
        ColorMode::Xyb => decode_jpeg_with_icc(data)
            .map(|(pixels, _, _)| pixels)
            .unwrap_or_else(|_| {
                use zune_jpeg::zune_core::bytestream::ZCursor;
                use zune_jpeg::JpegDecoder;
                let cursor = ZCursor::new(data);
                let mut decoder = JpegDecoder::new(cursor);
                decoder.decode().expect("JPEG decode failed")
            }),
        ColorMode::YCbCr => {
            use zune_jpeg::zune_core::bytestream::ZCursor;
            use zune_jpeg::JpegDecoder;
            let cursor = ZCursor::new(data);
            let mut decoder = JpegDecoder::new(cursor);
            decoder.decode().expect("JPEG decode failed")
        }
    }
}

fn compute_max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn compute_ssim2(orig_rgb: &[u8], decoded_rgb: &[u8], width: usize, height: usize) -> f64 {
    let orig = Rgb::new(
        orig_rgb
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
    )
    .unwrap();

    let dec = Rgb::new(
        decoded_rgb
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
    )
    .unwrap();

    compute_frame_ssimulacra2(orig, dec).unwrap_or(-1.0)
}

// ============================================================================
// Encoding Functions
// ============================================================================

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, config: &Config) -> Vec<u8> {
    let sub = match config.chroma.to_jpegli() {
        Subsampling::S444 => ChromaSubsampling::Full,
        Subsampling::S422 => ChromaSubsampling::HalfHorizontal,
        Subsampling::S420 => ChromaSubsampling::Quarter,
        Subsampling::S440 => ChromaSubsampling::HalfVertical,
        _ => ChromaSubsampling::Quarter,
    };
    let mut enc_config = EncoderConfig::new()
        .quality(quality as f32)
        .progressive(config.scan.to_jpegli() == JpegMode::Progressive)
        .optimize_huffman(config.huffman == HuffmanMode::Optimized)
        .ycbcr(sub);
    if config.color == ColorMode::Xyb {
        enc_config = enc_config.xyb();
    }
    let mut enc = enc_config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("Rust encoding failed")
}

fn encode_cpp(cjpegli: &Path, input_path: &Path, quality: u8, config: &Config) -> Option<Vec<u8>> {
    let output_path = format!("/tmp/cpp_parity_{}.jpg", config.name());

    let mut args: Vec<String> = vec![
        input_path.to_str().unwrap().to_string(),
        output_path.clone(),
        "-q".to_string(),
        quality.to_string(),
    ];

    for arg in config.scan.cpp_args() {
        args.push(arg.to_string());
    }
    for arg in config.huffman.cpp_args() {
        args.push(arg.to_string());
    }
    for arg in config.chroma.cpp_args() {
        args.push(arg.to_string());
    }
    for arg in config.color.cpp_args() {
        args.push(arg.to_string());
    }

    let output = Command::new(cjpegli).args(&args).output().ok()?;

    if !output.status.success() {
        eprintln!(
            "C++ encoding failed for {}: {}",
            config.name(),
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    let data = fs::read(&output_path).ok()?;
    let _ = fs::remove_file(&output_path);
    Some(data)
}

fn compare_config(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    config: &Config,
    cjpegli: &Path,
    ppm_path: &Path,
) -> Option<ParityResult> {
    let rust_jpeg = encode_rust(rgb, width, height, quality, config);
    let cpp_jpeg = encode_cpp(cjpegli, ppm_path, quality, config)?;

    let rust_decoded = decode_jpeg(&rust_jpeg, config.color);
    let cpp_decoded = decode_jpeg(&cpp_jpeg, config.color);

    let rust_ssim2 = compute_ssim2(rgb, &rust_decoded, width as usize, height as usize);
    let cpp_ssim2 = compute_ssim2(rgb, &cpp_decoded, width as usize, height as usize);
    let max_pixel_diff = compute_max_pixel_diff(&rust_decoded, &cpp_decoded);

    Some(ParityResult {
        rust_size: rust_jpeg.len(),
        cpp_size: cpp_jpeg.len(),
        rust_ssim2,
        cpp_ssim2,
        max_pixel_diff,
    })
}

// ============================================================================
// Image Loading
// ============================================================================

fn load_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

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

    Some((rgb, info.width, info.height))
}

fn generate_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;

            rgb[idx] = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 1] = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 2] = (128.0 + ((fx + fy) * 50.0).sin() * 50.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

fn write_ppm(path: &Path, rgb: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn run_comparison_for_size(
    rgb: &[u8],
    width: u32,
    height: u32,
    size_name: &str,
    quality: u8,
    cjpegli: &Path,
    configs: &[Config],
) -> BTreeMap<Config, ParityResult> {
    let ppm_path_str = format!("/tmp/cpp_parity_{}x{}.ppm", width, height);
    let ppm_path = Path::new(&ppm_path_str);
    write_ppm(ppm_path, rgb, width, height).expect("Failed to write PPM");

    println!("\n{}", "=".repeat(100));
    println!(
        " SIZE: {} ({}x{} = {:.2} Mpx)",
        size_name,
        width,
        height,
        (width as f64 * height as f64) / 1_000_000.0
    );
    println!("{}", "=".repeat(100));
    println!();

    println!(
        "{:<16} | {:>9} {:>9} {:>8} | {:>7} {:>7} {:>7} | {:>6}",
        "Mode", "Rust KB", "C++ KB", "Δ Size", "R SSIM2", "C SSIM2", "Δ SSIM2", "Δ px"
    );
    println!("{:-<100}", "");

    let mut results: BTreeMap<Config, ParityResult> = BTreeMap::new();

    for config in configs {
        print!("{:<16} | ", config.short_name());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match compare_config(rgb, width, height, quality, config, cjpegli, ppm_path) {
            Some(result) => {
                println!(
                    "{:>9.1} {:>9.1} {:>+7.2}% | {:>7.2} {:>7.2} {:>+6.2} | {:>6}",
                    result.rust_size as f64 / 1024.0,
                    result.cpp_size as f64 / 1024.0,
                    result.size_diff_pct(),
                    result.rust_ssim2,
                    result.cpp_ssim2,
                    result.ssim2_diff(),
                    result.max_pixel_diff
                );
                results.insert(*config, result);
            }
            None => {
                println!("FAILED (C++ encoding error)");
            }
        }
    }

    let _ = fs::remove_file(ppm_path);
    results
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut image_path: Option<String> = None;
    let mut csv_path: Option<String> = None;
    let mut quality = 90u8;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" | "-c" => {
                csv_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--quality" | "-q" => {
                quality = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(90);
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: cpp_parity_matrix [OPTIONS] [IMAGE]");
                println!();
                println!("Compares Rust jpegli-rs vs C++ cjpegli output parity.");
                println!("Does NOT compare timing (unfair: subprocess vs library).");
                println!();
                println!("Options:");
                println!("  -q, --quality Q   Quality level 1-100 (default: 90)");
                println!("  -c, --csv FILE    Save results to CSV file");
                println!("  -h, --help        Show this help");
                println!();
                println!("Arguments:");
                println!("  IMAGE             Path to PNG image (uses that size only)");
                println!("                    Default: runs 512, 2K, and 4K synthetic images");
                return;
            }
            arg if !arg.starts_with('-') => {
                image_path = Some(arg.to_string());
                i += 1;
            }
            _ => i += 1,
        }
    }

    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("ERROR: cjpegli not found. Build internal/jpegli-cpp first:");
            eprintln!("  cd internal/jpegli-cpp && mkdir -p build && cd build");
            eprintln!("  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j");
            return;
        }
    };

    println!();
    println!("{}", "=".repeat(100));
    println!(" RUST vs C++ JPEGLI PARITY: File Size + SSIMULACRA2");
    println!("{}", "=".repeat(100));
    println!();
    println!("Quality: {}", quality);

    let configs: Vec<Config> = vec![
        Config {
            scan: ScanMode::Baseline,
            huffman: HuffmanMode::Fixed,
            chroma: ChromaSampling::S444,
            color: ColorMode::YCbCr,
        },
        Config {
            scan: ScanMode::Baseline,
            huffman: HuffmanMode::Optimized,
            chroma: ChromaSampling::S444,
            color: ColorMode::YCbCr,
        },
        Config {
            scan: ScanMode::Progressive,
            huffman: HuffmanMode::Optimized,
            chroma: ChromaSampling::S444,
            color: ColorMode::YCbCr,
        },
        Config {
            scan: ScanMode::Baseline,
            huffman: HuffmanMode::Optimized,
            chroma: ChromaSampling::S420,
            color: ColorMode::YCbCr,
        },
        Config {
            scan: ScanMode::Progressive,
            huffman: HuffmanMode::Optimized,
            chroma: ChromaSampling::S420,
            color: ColorMode::YCbCr,
        },
        Config {
            scan: ScanMode::Baseline,
            huffman: HuffmanMode::Optimized,
            chroma: ChromaSampling::S444,
            color: ColorMode::Xyb,
        },
        Config {
            scan: ScanMode::Progressive,
            huffman: HuffmanMode::Optimized,
            chroma: ChromaSampling::S444,
            color: ColorMode::Xyb,
        },
    ];

    println!("Configs: {}", configs.len());

    let mut all_results: Vec<(String, u32, u32, BTreeMap<Config, ParityResult>)> = Vec::new();

    if let Some(path) = &image_path {
        match load_png(Path::new(path)) {
            Some((rgb, width, height)) => {
                println!("Image:   {} ({}x{})", path, width, height);
                let results = run_comparison_for_size(
                    &rgb, width, height, "custom", quality, &cjpegli, &configs,
                );
                all_results.push((path.clone(), width, height, results));
            }
            None => {
                eprintln!("Failed to load image: {}", path);
                return;
            }
        }
    } else {
        println!("Sizes:   512, 2K, 4K synthetic images");

        for &(width, height, name) in SYNTHETIC_SIZES {
            print!("\nGenerating {}x{} synthetic image...", width, height);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            let rgb = generate_test_image(width as usize, height as usize);
            println!(" done");

            let results =
                run_comparison_for_size(&rgb, width, height, name, quality, &cjpegli, &configs);
            all_results.push((name.to_string(), width, height, results));
        }
    }

    // Summary
    println!("\n{}", "=".repeat(100));
    println!(" SUMMARY ACROSS ALL SIZES");
    println!("{}", "=".repeat(100));

    let all_parity_results: Vec<&ParityResult> = all_results
        .iter()
        .flat_map(|(_, _, _, results)| results.values())
        .collect();

    if !all_parity_results.is_empty() {
        println!();
        println!(
            "{:<16} | {:>12} {:>12} | {:>10}",
            "Mode", "Avg Δ Size", "Avg Δ SSIM2", "Max Δ px"
        );
        println!("{:-<60}", "");

        for config in &configs {
            let config_results: Vec<&ParityResult> = all_results
                .iter()
                .filter_map(|(_, _, _, results)| results.get(config))
                .collect();

            if !config_results.is_empty() {
                let avg_size_diff: f64 = config_results
                    .iter()
                    .map(|r| r.size_diff_pct())
                    .sum::<f64>()
                    / config_results.len() as f64;
                let avg_ssim2_diff: f64 =
                    config_results.iter().map(|r| r.ssim2_diff()).sum::<f64>()
                        / config_results.len() as f64;
                let max_parity_diff: u8 = config_results
                    .iter()
                    .map(|r| r.max_pixel_diff)
                    .max()
                    .unwrap_or(0);

                println!(
                    "{:<16} | {:>+11.2}% {:>+11.2} | {:>10}",
                    config.short_name(),
                    avg_size_diff,
                    avg_ssim2_diff,
                    max_parity_diff
                );
            }
        }

        let avg_size_diff: f64 = all_parity_results
            .iter()
            .map(|r| r.size_diff_pct())
            .sum::<f64>()
            / all_parity_results.len() as f64;
        let avg_ssim2_diff: f64 = all_parity_results
            .iter()
            .map(|r| r.ssim2_diff())
            .sum::<f64>()
            / all_parity_results.len() as f64;

        println!("{:-<60}", "");
        println!(
            "{:<16} | {:>+11.2}% {:>+11.2} |",
            "OVERALL", avg_size_diff, avg_ssim2_diff
        );
    }

    if let Some(csv_file) = csv_path {
        let mut csv = String::new();
        csv.push_str("size,width,height,mode,rust_kb,cpp_kb,size_diff_pct,rust_ssim2,cpp_ssim2,ssim2_diff,parity_px\n");

        for (name, width, height, results) in &all_results {
            for (config, result) in results {
                csv.push_str(&format!(
                    "{},{},{},{},{:.1},{:.1},{:+.2},{:.2},{:.2},{:+.2},{}\n",
                    name,
                    width,
                    height,
                    config.short_name(),
                    result.rust_size as f64 / 1024.0,
                    result.cpp_size as f64 / 1024.0,
                    result.size_diff_pct(),
                    result.rust_ssim2,
                    result.cpp_ssim2,
                    result.ssim2_diff(),
                    result.max_pixel_diff
                ));
            }
        }

        fs::write(&csv_file, csv).expect("Failed to write CSV");
        println!("\nResults saved to: {}", csv_file);
    }

    println!();
    println!("{}", "=".repeat(100));
    println!("LEGEND:");
    println!("  Δ Size:   positive = Rust larger, negative = Rust smaller");
    println!("  Δ SSIM2:  positive = Rust better quality, negative = C++ better");
    println!("  Δ px:     max pixel difference between Rust and C++ decoded outputs");
    println!("{}", "=".repeat(100));
}
