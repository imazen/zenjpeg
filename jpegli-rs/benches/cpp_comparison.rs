//! Criterion benchmarks comparing Rust jpegli-rs vs C++ cjpegli.
//!
//! Provides statistically valid timing comparisons with noise detection.
//!
//! # Usage
//!
//! ```bash
//! # Run all comparison benchmarks
//! cargo bench --bench cpp_comparison
//!
//! # Run specific benchmark
//! cargo bench --bench cpp_comparison -- "baseline"
//!
//! # Save baseline for regression detection
//! cargo bench --bench cpp_comparison -- --save-baseline main
//!
//! # Compare against baseline
//! cargo bench --bench cpp_comparison -- --baseline main
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use jpegli::test_utils::find_cjpegli;
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

// ============================================================================
// Test Data
// ============================================================================

struct TestImage {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    ppm_path: PathBuf,
}

impl TestImage {
    fn new(width: u32, height: u32) -> Self {
        let rgb = generate_complex_image(width as usize, height as usize);
        let ppm_path = PathBuf::from(format!("/tmp/bench_cpp_{}x{}.ppm", width, height));
        write_ppm(&ppm_path, &rgb, width, height).expect("write ppm");
        Self {
            rgb,
            width,
            height,
            ppm_path,
        }
    }

    fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }
}

impl Drop for TestImage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.ppm_path);
    }
}

fn generate_complex_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;

            // Complex pattern with multiple frequencies
            rgb[idx] = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 1] = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0).clamp(0.0, 255.0) as u8;
            rgb[idx + 2] = (128.0 + ((fx + fy) * 50.0).sin() * 50.0).clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}

fn write_ppm(path: &PathBuf, rgb: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

// ============================================================================
// Encoder Wrappers
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct EncodeConfig {
    mode: JpegMode,
    optimize_huffman: bool,
    subsampling: Subsampling,
    use_xyb: bool,
    quality: u8,
}

impl EncodeConfig {
    fn name(&self) -> String {
        let mode = match self.mode {
            JpegMode::Baseline => "base",
            JpegMode::Progressive => "prog",
            JpegMode::Lossless => "lossless",
            _ => "other",
        };
        let huff = if self.optimize_huffman { "opt" } else { "fix" };
        let sub = match self.subsampling {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S420 => "420",
            Subsampling::S440 => "440",
            _ => "other",
        };
        let color = if self.use_xyb { "xyb" } else { "ycbcr" };
        format!("{}-{}-{}-{}", mode, huff, sub, color)
    }

    fn cpp_args(&self) -> Vec<String> {
        let mut args = vec!["-q".to_string(), self.quality.to_string()];

        // Progressive level
        match self.mode {
            JpegMode::Baseline => args.extend(["-p".to_string(), "0".to_string()]),
            JpegMode::Progressive => args.extend(["-p".to_string(), "2".to_string()]),
            JpegMode::Lossless => {}
            _ => {}
        }

        // Huffman
        if !self.optimize_huffman {
            args.push("--fixed_code".to_string());
        }

        // Subsampling
        let sub = match self.subsampling {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S420 => "420",
            Subsampling::S440 => "440",
            _ => "420",
        };
        args.push(format!("--chroma_subsampling={}", sub));

        // XYB
        if self.use_xyb {
            args.push("--xyb".to_string());
        }

        args
    }
}

fn encode_rust(image: &TestImage, config: &EncodeConfig) -> Vec<u8> {
    Encoder::new()
        .width(image.width)
        .height(image.height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(config.quality as f32))
        .mode(config.mode)
        .optimize_huffman(config.optimize_huffman)
        .subsampling(config.subsampling)
        .use_xyb(config.use_xyb)
        .encode(&image.rgb)
        .expect("Rust encode failed")
}

fn encode_cpp(cjpegli: &PathBuf, image: &TestImage, config: &EncodeConfig) -> Vec<u8> {
    let output_path = format!("/tmp/bench_cpp_out_{}.jpg", std::process::id());

    let mut args = config.cpp_args();
    args.insert(0, image.ppm_path.to_str().unwrap().to_string());
    args.insert(1, output_path.clone());

    let result = Command::new(cjpegli).args(&args).output().expect("run cjpegli");

    if !result.status.success() {
        panic!(
            "cjpegli failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }

    let data = fs::read(&output_path).expect("read cpp output");
    let _ = fs::remove_file(&output_path);
    data
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_rust_vs_cpp(c: &mut Criterion) {
    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpegli not found. Build internal/jpegli-cpp first.");
            return;
        }
    };

    // Create test image (512x512 is a good balance of speed and realism)
    let image = TestImage::new(512, 512);

    // Test configurations
    let configs = vec![
        // Baseline YCbCr
        EncodeConfig {
            mode: JpegMode::Baseline,
            optimize_huffman: false,
            subsampling: Subsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        EncodeConfig {
            mode: JpegMode::Baseline,
            optimize_huffman: true,
            subsampling: Subsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        EncodeConfig {
            mode: JpegMode::Baseline,
            optimize_huffman: true,
            subsampling: Subsampling::S444,
            use_xyb: false,
            quality: 90,
        },
        // Progressive YCbCr
        EncodeConfig {
            mode: JpegMode::Progressive,
            optimize_huffman: true,
            subsampling: Subsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        EncodeConfig {
            mode: JpegMode::Progressive,
            optimize_huffman: true,
            subsampling: Subsampling::S444,
            use_xyb: false,
            quality: 90,
        },
        // XYB mode (always 444)
        EncodeConfig {
            mode: JpegMode::Baseline,
            optimize_huffman: true,
            subsampling: Subsampling::S444,
            use_xyb: true,
            quality: 90,
        },
        EncodeConfig {
            mode: JpegMode::Progressive,
            optimize_huffman: true,
            subsampling: Subsampling::S444,
            use_xyb: true,
            quality: 90,
        },
    ];

    let mut group = c.benchmark_group("rust_vs_cpp");
    group.throughput(Throughput::Elements(image.pixels()));
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for config in &configs {
        // Benchmark Rust
        group.bench_with_input(
            BenchmarkId::new("rust", config.name()),
            &(&image, config),
            |b, (img, cfg)| {
                b.iter(|| encode_rust(black_box(img), black_box(cfg)));
            },
        );

        // Benchmark C++
        group.bench_with_input(
            BenchmarkId::new("cpp", config.name()),
            &(&image, config, &cjpegli),
            |b, (img, cfg, cjpegli)| {
                b.iter(|| encode_cpp(black_box(cjpegli), black_box(img), black_box(cfg)));
            },
        );
    }

    group.finish();
}

fn bench_sizes(c: &mut Criterion) {
    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpegli not found.");
            return;
        }
    };

    let config = EncodeConfig {
        mode: JpegMode::Progressive,
        optimize_huffman: true,
        subsampling: Subsampling::S420,
        use_xyb: false,
        quality: 90,
    };

    let mut group = c.benchmark_group("size_scaling");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for size in [256, 512, 1024] {
        let image = TestImage::new(size, size);
        group.throughput(Throughput::Elements(image.pixels()));

        group.bench_with_input(
            BenchmarkId::new("rust", format!("{}x{}", size, size)),
            &(&image, &config),
            |b, (img, cfg)| {
                b.iter(|| encode_rust(black_box(img), black_box(cfg)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpp", format!("{}x{}", size, size)),
            &(&image, &config, &cjpegli),
            |b, (img, cfg, cjpegli)| {
                b.iter(|| encode_cpp(black_box(cjpegli), black_box(img), black_box(cfg)));
            },
        );
    }

    group.finish();
}

fn bench_quality_levels(c: &mut Criterion) {
    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpegli not found.");
            return;
        }
    };

    let image = TestImage::new(512, 512);

    let mut group = c.benchmark_group("quality_levels");
    group.throughput(Throughput::Elements(image.pixels()));
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for quality in [50, 75, 90, 95] {
        let config = EncodeConfig {
            mode: JpegMode::Progressive,
            optimize_huffman: true,
            subsampling: Subsampling::S420,
            use_xyb: false,
            quality,
        };

        group.bench_with_input(
            BenchmarkId::new("rust", format!("q{}", quality)),
            &(&image, &config),
            |b, (img, cfg)| {
                b.iter(|| encode_rust(black_box(img), black_box(cfg)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpp", format!("q{}", quality)),
            &(&image, &config, &cjpegli),
            |b, (img, cfg, cjpegli)| {
                b.iter(|| encode_cpp(black_box(cjpegli), black_box(img), black_box(cfg)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Verification (run once before benchmarks)
// ============================================================================

fn verify_outputs_match(c: &mut Criterion) {
    let cjpegli = match find_cjpegli() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpegli not found.");
            return;
        }
    };

    let image = TestImage::new(512, 512);
    let config = EncodeConfig {
        mode: JpegMode::Progressive,
        optimize_huffman: true,
        subsampling: Subsampling::S420,
        use_xyb: false,
        quality: 90,
    };

    // Encode both
    let rust_jpeg = encode_rust(&image, &config);
    let cpp_jpeg = encode_cpp(&cjpegli, &image, &config);

    // Decode both
    let rust_decoded = decode_jpeg(&rust_jpeg);
    let cpp_decoded = decode_jpeg(&cpp_jpeg);

    // Compare
    let max_diff = compute_max_diff(&rust_decoded, &cpp_decoded);
    let size_diff_pct =
        (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64 * 100.0;

    println!("\n=== OUTPUT VERIFICATION ===");
    println!("Rust size:  {} bytes", rust_jpeg.len());
    println!("C++ size:   {} bytes", cpp_jpeg.len());
    println!("Size diff:  {:+.2}%", size_diff_pct);
    println!("Max pixel diff (Rust vs C++ decoded): {}", max_diff);

    if max_diff > 2 {
        println!("WARNING: Outputs differ significantly!");
    } else {
        println!("OK: Outputs are nearly identical");
    }
    println!();

    // Dummy benchmark just to include this in the benchmark run
    let mut group = c.benchmark_group("verification");
    group.bench_function("check", |b| b.iter(|| black_box(1 + 1)));
    group.finish();
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;
    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("JPEG decode failed")
}

fn compute_max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

criterion_group!(
    benches,
    verify_outputs_match,
    bench_rust_vs_cpp,
    bench_sizes,
    bench_quality_levels,
);
criterion_main!(benches);
