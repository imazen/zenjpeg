//! Comprehensive timing benchmark: Rust jpegli-rs vs C++ cjpegli
//!
//! Tests all combinations of encoding modes with timing measurements.
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings (512x512 test image)
//! cargo run --release --example cpp_timing_matrix
//!
//! # Run with a specific image
//! cargo run --release --example cpp_timing_matrix -- path/to/image.png
//!
//! # Run with iterations for more stable timing
//! cargo run --release --example cpp_timing_matrix -- --iterations 5
//!
//! # Save results to CSV for tracking over time
//! cargo run --release --example cpp_timing_matrix -- --csv results.csv
//! ```

use jpegli::test_utils::find_cjpegli;
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

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
            match self.scan {
                ScanMode::Baseline => "B",
                ScanMode::Progressive => "P",
            },
            match self.huffman {
                HuffmanMode::Fixed => "F",
                HuffmanMode::Optimized => "O",
            },
            self.chroma.name(),
            match self.color {
                ColorMode::YCbCr => "Y",
                ColorMode::Xyb => "X",
            }
        )
    }

    fn is_valid(&self) -> bool {
        // Progressive + Fixed huffman is not a common/supported combination
        // XYB mode always uses 444 subsampling
        if self.color == ColorMode::Xyb && self.chroma == ChromaSampling::S420 {
            return false;
        }
        true
    }
}

// ============================================================================
// Benchmark Results
// ============================================================================

#[derive(Debug, Clone)]
struct TimingResult {
    rust_time_ms: f64,
    cpp_time_ms: f64,
    rust_size: usize,
    cpp_size: usize,
    // Verification metrics
    max_pixel_diff: u8,
    dssim: f64,
}

impl TimingResult {
    fn speedup(&self) -> f64 {
        self.cpp_time_ms / self.rust_time_ms
    }

    fn size_diff_pct(&self) -> f64 {
        (self.rust_size as f64 - self.cpp_size as f64) / self.cpp_size as f64 * 100.0
    }

    fn is_valid(&self) -> bool {
        // Results are valid if max pixel diff is small and DSSIM is very low
        self.max_pixel_diff <= 2 && self.dssim < 0.0001
    }
}

// ============================================================================
// Verification Functions
// ============================================================================

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("JPEG decode failed")
}

fn compute_max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

fn compute_dssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    let attr = dssim::Dssim::new();

    let a_rgba: Vec<rgb::RGBA8> = a
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();
    let b_rgba: Vec<rgb::RGBA8> = b
        .chunks(3)
        .map(|c| rgb::RGBA8::new(c[0], c[1], c[2], 255))
        .collect();

    let a_img = attr.create_image_rgba(&a_rgba, width, height).unwrap();
    let b_img = attr.create_image_rgba(&b_rgba, width, height).unwrap();

    let (dssim, _) = attr.compare(&a_img, b_img);
    dssim.into()
}

// ============================================================================
// Encoding Functions
// ============================================================================

fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: u8, config: &Config) -> Vec<u8> {
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .mode(config.scan.to_jpegli())
        .optimize_huffman(config.huffman == HuffmanMode::Optimized)
        .subsampling(config.chroma.to_jpegli())
        .use_xyb(config.color == ColorMode::Xyb)
        .encode(rgb)
        .expect("Rust encoding failed")
}

fn encode_cpp(cjpegli: &Path, input_path: &Path, quality: u8, config: &Config) -> Option<Vec<u8>> {
    let output_path = format!("/tmp/cpp_timing_{}.jpg", config.name());

    let mut args: Vec<String> = vec![
        input_path.to_str().unwrap().to_string(),
        output_path.clone(),
        "-q".to_string(),
        quality.to_string(),
    ];

    // Add mode-specific arguments
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

fn benchmark_config(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    config: &Config,
    cjpegli: &Path,
    ppm_path: &Path,
    iterations: usize,
) -> Option<TimingResult> {
    // Warmup
    let _ = encode_rust(rgb, width, height, quality, config);
    let _ = encode_cpp(cjpegli, ppm_path, quality, config)?;

    // Benchmark Rust
    let rust_start = Instant::now();
    let mut rust_jpeg = Vec::new();
    for _ in 0..iterations {
        rust_jpeg = encode_rust(rgb, width, height, quality, config);
    }
    let rust_time_ms = rust_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // Benchmark C++
    let cpp_start = Instant::now();
    let mut cpp_jpeg = Vec::new();
    for _ in 0..iterations {
        cpp_jpeg = encode_cpp(cjpegli, ppm_path, quality, config)?;
    }
    let cpp_time_ms = cpp_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // Verify outputs are near-identical by decoding and comparing
    let rust_decoded = decode_jpeg(&rust_jpeg);
    let cpp_decoded = decode_jpeg(&cpp_jpeg);

    let max_pixel_diff = compute_max_pixel_diff(&rust_decoded, &cpp_decoded);
    let dssim = compute_dssim(&rust_decoded, &cpp_decoded, width as usize, height as usize);

    Some(TimingResult {
        rust_time_ms,
        cpp_time_ms,
        rust_size: rust_jpeg.len(),
        cpp_size: cpp_jpeg.len(),
        max_pixel_diff,
        dssim,
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

            // Mix of gradients and patterns for realistic content
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut image_path: Option<String> = None;
    let mut iterations = 3;
    let mut csv_path: Option<String> = None;
    let mut quality = 90u8;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" | "-i" => {
                iterations = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(3);
                i += 2;
            }
            "--csv" | "-c" => {
                csv_path = args.get(i + 1).cloned();
                i += 2;
            }
            "--quality" | "-q" => {
                quality = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(90);
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: cpp_timing_matrix [OPTIONS] [IMAGE]");
                println!();
                println!("Options:");
                println!("  -i, --iterations N   Number of iterations per config (default: 3)");
                println!("  -q, --quality Q      Quality level 1-100 (default: 90)");
                println!("  -c, --csv FILE       Save results to CSV file");
                println!("  -h, --help           Show this help");
                println!();
                println!("Arguments:");
                println!("  IMAGE                Path to PNG image (default: synthetic 512x512)");
                return;
            }
            arg if !arg.starts_with('-') => {
                image_path = Some(arg.to_string());
                i += 1;
            }
            _ => i += 1,
        }
    }

    // Check for cjpegli
    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("ERROR: cjpegli not found. Build internal/jpegli-cpp first:");
            eprintln!("  cd internal/jpegli-cpp && mkdir -p build && cd build");
            eprintln!("  cmake .. -DCMAKE_BUILD_TYPE=Release && make -j");
            return;
        }
    };

    // Load or generate test image
    let (rgb, width, height) = if let Some(path) = &image_path {
        match load_png(Path::new(path)) {
            Some(data) => {
                println!("Loaded image: {} ({}x{})", path, data.1, data.2);
                data
            }
            None => {
                eprintln!("Failed to load image: {}", path);
                return;
            }
        }
    } else {
        let (w, h) = (512u32, 512u32);
        println!("Using synthetic test image ({}x{})", w, h);
        (generate_test_image(w as usize, h as usize), w, h)
    };

    // Write PPM for C++ encoder
    let ppm_path = Path::new("/tmp/cpp_timing_input.ppm");
    write_ppm(ppm_path, &rgb, width, height).expect("Failed to write PPM");

    // Generate all valid configurations
    let configs: Vec<Config> = {
        let mut configs = Vec::new();
        for scan in [ScanMode::Baseline, ScanMode::Progressive] {
            for huffman in [HuffmanMode::Fixed, HuffmanMode::Optimized] {
                for chroma in [ChromaSampling::S444, ChromaSampling::S420] {
                    for color in [ColorMode::YCbCr, ColorMode::Xyb] {
                        let config = Config {
                            scan,
                            huffman,
                            chroma,
                            color,
                        };
                        if config.is_valid() {
                            configs.push(config);
                        }
                    }
                }
            }
        }
        configs
    };

    println!();
    println!("{}", "=".repeat(100));
    println!(" RUST vs C++ JPEGLI TIMING BENCHMARK");
    println!("{}", "=".repeat(100));
    println!();
    println!(
        "Image:      {}x{} ({} pixels)",
        width,
        height,
        width * height
    );
    println!("Quality:    {}", quality);
    println!("Iterations: {}", iterations);
    println!("Configs:    {}", configs.len());
    println!();

    // Print header
    println!(
        "{:<20} | {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7} | {:>4} {:>8} {:>6}",
        "Config",
        "Rust ms",
        "C++ ms",
        "Speedup",
        "Rust KB",
        "C++ KB",
        "Δ Size",
        "Δpx",
        "DSSIM",
        "Valid"
    );
    println!("{}", "-".repeat(115));

    // Run benchmarks and collect results
    let mut results: BTreeMap<Config, TimingResult> = BTreeMap::new();

    for config in &configs {
        print!("{:<20} | ", config.short_name());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        match benchmark_config(
            &rgb, width, height, quality, config, &cjpegli, ppm_path, iterations,
        ) {
            Some(result) => {
                let speedup = result.speedup();
                let speedup_str = if speedup >= 1.0 {
                    format!("{:.2}x", speedup)
                } else {
                    format!("{:.2}x", speedup)
                };

                let valid_str = if result.is_valid() { "✓" } else { "✗" };

                println!(
                    "{:>8.2} {:>8.2} {:>7} | {:>8.1} {:>8.1} {:>+6.1}% | {:>4} {:>8.6} {:>6}",
                    result.rust_time_ms,
                    result.cpp_time_ms,
                    speedup_str,
                    result.rust_size as f64 / 1024.0,
                    result.cpp_size as f64 / 1024.0,
                    result.size_diff_pct(),
                    result.max_pixel_diff,
                    result.dssim,
                    valid_str
                );

                results.insert(*config, result);
            }
            None => {
                println!("FAILED (C++ encoding error)");
            }
        }
    }

    println!("{}", "-".repeat(115));

    // Summary statistics
    if !results.is_empty() {
        let avg_speedup: f64 =
            results.values().map(|r| r.speedup()).sum::<f64>() / results.len() as f64;
        let min_speedup = results
            .values()
            .map(|r| r.speedup())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_speedup = results
            .values()
            .map(|r| r.speedup())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let avg_size_diff: f64 =
            results.values().map(|r| r.size_diff_pct()).sum::<f64>() / results.len() as f64;

        // Validation stats
        let valid_count = results.values().filter(|r| r.is_valid()).count();
        let max_pixel_diff_overall = results
            .values()
            .map(|r| r.max_pixel_diff)
            .max()
            .unwrap_or(0);
        let max_dssim_overall = results
            .values()
            .map(|r| r.dssim)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        println!();
        println!("SUMMARY:");
        println!(
            "  Speedup (Rust vs C++): avg {:.2}x, min {:.2}x, max {:.2}x",
            avg_speedup, min_speedup, max_speedup
        );
        println!("  Average size difference: {:+.2}%", avg_size_diff);
        println!();
        println!("VALIDATION (Rust vs C++ output similarity):");
        println!(
            "  Valid configs: {}/{} (max pixel diff ≤2, DSSIM <0.0001)",
            valid_count,
            results.len()
        );
        println!(
            "  Worst case: max_pixel_diff={}, max_dssim={:.6}",
            max_pixel_diff_overall, max_dssim_overall
        );
        if valid_count < results.len() {
            println!("  WARNING: Some configs produced different output than C++!");
        }

        // Breakdown by mode
        println!();
        println!("BREAKDOWN BY MODE:");

        // By scan mode
        for scan in [ScanMode::Baseline, ScanMode::Progressive] {
            let filtered: Vec<_> = results
                .iter()
                .filter(|(c, _)| c.scan == scan)
                .map(|(_, r)| r)
                .collect();
            if !filtered.is_empty() {
                let avg: f64 =
                    filtered.iter().map(|r| r.speedup()).sum::<f64>() / filtered.len() as f64;
                println!("  {:12}: {:.2}x avg speedup", scan.name(), avg);
            }
        }

        // By huffman mode
        for huffman in [HuffmanMode::Fixed, HuffmanMode::Optimized] {
            let filtered: Vec<_> = results
                .iter()
                .filter(|(c, _)| c.huffman == huffman)
                .map(|(_, r)| r)
                .collect();
            if !filtered.is_empty() {
                let avg: f64 =
                    filtered.iter().map(|r| r.speedup()).sum::<f64>() / filtered.len() as f64;
                println!("  {:12}: {:.2}x avg speedup", huffman.name(), avg);
            }
        }

        // By chroma
        for chroma in [ChromaSampling::S444, ChromaSampling::S420] {
            let filtered: Vec<_> = results
                .iter()
                .filter(|(c, _)| c.chroma == chroma)
                .map(|(_, r)| r)
                .collect();
            if !filtered.is_empty() {
                let avg: f64 =
                    filtered.iter().map(|r| r.speedup()).sum::<f64>() / filtered.len() as f64;
                println!("  {:12}: {:.2}x avg speedup", chroma.name(), avg);
            }
        }

        // By color
        for color in [ColorMode::YCbCr, ColorMode::Xyb] {
            let filtered: Vec<_> = results
                .iter()
                .filter(|(c, _)| c.color == color)
                .map(|(_, r)| r)
                .collect();
            if !filtered.is_empty() {
                let avg: f64 =
                    filtered.iter().map(|r| r.speedup()).sum::<f64>() / filtered.len() as f64;
                println!("  {:12}: {:.2}x avg speedup", color.name(), avg);
            }
        }
    }

    // Save to CSV if requested
    if let Some(csv_file) = csv_path {
        let mut csv = String::new();
        csv.push_str("timestamp,image,width,height,quality,scan,huffman,chroma,color,rust_ms,cpp_ms,speedup,rust_bytes,cpp_bytes,size_diff_pct\n");

        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let image_name = image_path.as_deref().unwrap_or("synthetic");

        for (config, result) in &results {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{},{},{:.2}\n",
                timestamp,
                image_name,
                width,
                height,
                quality,
                config.scan.name(),
                config.huffman.name(),
                config.chroma.name(),
                config.color.name(),
                result.rust_time_ms,
                result.cpp_time_ms,
                result.speedup(),
                result.rust_size,
                result.cpp_size,
                result.size_diff_pct()
            ));
        }

        // Append to file if it exists, otherwise create with header
        let path = Path::new(&csv_file);
        if path.exists() {
            // Read existing, skip our header if file already has one
            let existing = fs::read_to_string(path).unwrap_or_default();
            if existing.starts_with("timestamp,") {
                // Append without header
                let lines: Vec<_> = csv.lines().skip(1).collect();
                fs::write(path, format!("{}{}\n", existing, lines.join("\n")))
                    .expect("Failed to write CSV");
            } else {
                fs::write(path, csv).expect("Failed to write CSV");
            }
        } else {
            fs::write(path, csv).expect("Failed to write CSV");
        }

        println!();
        println!("Results saved to: {}", csv_file);
    }

    // Cleanup
    let _ = fs::remove_file(ppm_path);

    println!();
    println!("Legend: B=Baseline, P=Progressive, F=Fixed, O=Optimized, Y=YCbCr, X=XYB");
    println!("Speedup > 1.0 means Rust is faster than C++");
}
