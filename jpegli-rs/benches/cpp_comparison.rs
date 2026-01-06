//! Criterion benchmarks comparing Rust jpegli-rs vs C++ cjpegli.
//!
//! Uses FFI encoding (not CLI subprocess) for accurate timing comparisons.
//!
//! # Usage
//!
//! ```bash
//! # Run all comparison benchmarks (requires cjpegli-ffi feature in jpegli-bench-utils)
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
//!
//! # Requirements
//!
//! This benchmark requires building jpegli-bench-utils with the `cjpegli-ffi` feature,
//! which in turn requires building internal/jpegli-cpp:
//!
//! ```bash
//! git submodule update --init --recursive
//! cd internal/jpegli-cpp
//! mkdir -p build && cd build
//! cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
//! ninja jpegli
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jpegli::types::{JpegMode, PixelFormat, Subsampling};
use jpegli::{Encoder, Quality};
use jpegli_bench_utils::{ChromaSubsampling, ColorMode, EncoderConfig, EncoderImpl, ImageData, ScanMode, SyntheticPattern};
use std::time::Duration;

// ============================================================================
// Test Data
// ============================================================================

fn create_test_image(width: u32, height: u32) -> ImageData {
    let pattern = SyntheticPattern::Complex;
    let img = pattern.generate(width, height);
    ImageData {
        name: format!("complex_{}x{}", width, height),
        pixels: img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect(),
        width: width as usize,
        height: height as usize,
    }
}

// ============================================================================
// Encoder Configurations
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct BenchConfig {
    mode: JpegMode,
    subsampling: Subsampling,
    use_xyb: bool,
    quality: u8,
}

impl BenchConfig {
    fn name(&self) -> String {
        let mode = match self.mode {
            JpegMode::Baseline => "base",
            JpegMode::Progressive => "prog",
            _ => "other",
        };
        let sub = match self.subsampling {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S420 => "420",
            Subsampling::S440 => "440",
            _ => "other",
        };
        let color = if self.use_xyb { "xyb" } else { "ycbcr" };
        format!("{}-{}-{}-q{}", mode, sub, color, self.quality)
    }

    fn to_encoder_config(&self) -> EncoderConfig {
        EncoderConfig::new(EncoderImpl::CJpegli)
            .color(if self.use_xyb { ColorMode::Xyb } else { ColorMode::YCbCr })
            .scan(match self.mode {
                JpegMode::Baseline => ScanMode::Baseline,
                JpegMode::Progressive => ScanMode::Progressive,
                _ => ScanMode::Progressive,
            })
            .subsampling(match self.subsampling {
                Subsampling::S444 => ChromaSubsampling::S444,
                Subsampling::S422 => ChromaSubsampling::S422,
                Subsampling::S420 => ChromaSubsampling::S420,
                Subsampling::S440 => ChromaSubsampling::S440,
                _ => ChromaSubsampling::S420,
            })
            .quality(self.quality)
    }
}

fn encode_rust(image: &ImageData, config: &BenchConfig) -> Vec<u8> {
    Encoder::new()
        .width(image.width as u32)
        .height(image.height as u32)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(config.quality as f32))
        .mode(config.mode)
        .optimize_huffman(true)
        .subsampling(config.subsampling)
        .use_xyb(config.use_xyb)
        .encode(&image.pixels)
        .expect("Rust encode failed")
}

fn encode_cpp_ffi(image: &ImageData, config: &BenchConfig) -> Vec<u8> {
    config.to_encoder_config()
        .encode(image)
        .expect("C++ FFI encode failed")
}

// ============================================================================
// Check if FFI is available
// ============================================================================

fn ffi_available() -> bool {
    // Try encoding a small test image with C++ FFI
    let test_img = ImageData {
        name: "test".to_string(),
        pixels: vec![128u8; 8 * 8 * 3],
        width: 8,
        height: 8,
    };

    let config = EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .quality(90);

    config.encode(&test_img).is_ok()
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_rust_vs_cpp(c: &mut Criterion) {
    if !ffi_available() {
        eprintln!("SKIP: C++ jpegli FFI not available.");
        eprintln!("Build jpegli-bench-utils with --features cjpegli-ffi");
        eprintln!("And ensure internal/jpegli-cpp is built.");
        return;
    }

    // Create test image (512x512 is a good balance of speed and realism)
    let image = create_test_image(512, 512);

    // Test configurations (YCbCr only - XYB not supported via FFI)
    let configs = vec![
        // Baseline YCbCr
        BenchConfig {
            mode: JpegMode::Baseline,
            subsampling: Subsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        BenchConfig {
            mode: JpegMode::Baseline,
            subsampling: Subsampling::S444,
            use_xyb: false,
            quality: 90,
        },
        // Progressive YCbCr
        BenchConfig {
            mode: JpegMode::Progressive,
            subsampling: Subsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        BenchConfig {
            mode: JpegMode::Progressive,
            subsampling: Subsampling::S444,
            use_xyb: false,
            quality: 90,
        },
    ];

    let mut group = c.benchmark_group("rust_vs_cpp_ffi");
    group.throughput(Throughput::Elements((image.width * image.height) as u64));
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

        // Benchmark C++ FFI
        group.bench_with_input(
            BenchmarkId::new("cpp_ffi", config.name()),
            &(&image, config),
            |b, (img, cfg)| {
                b.iter(|| encode_cpp_ffi(black_box(img), black_box(cfg)));
            },
        );
    }

    group.finish();
}

fn bench_sizes(c: &mut Criterion) {
    if !ffi_available() {
        eprintln!("SKIP: C++ jpegli FFI not available.");
        return;
    }

    let config = BenchConfig {
        mode: JpegMode::Progressive,
        subsampling: Subsampling::S420,
        use_xyb: false,
        quality: 90,
    };

    let mut group = c.benchmark_group("size_scaling_ffi");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for size in [256, 512, 1024] {
        let image = create_test_image(size, size);
        group.throughput(Throughput::Elements((image.width * image.height) as u64));

        group.bench_with_input(
            BenchmarkId::new("rust", format!("{}x{}", size, size)),
            &(&image, &config),
            |b, (img, cfg)| {
                b.iter(|| encode_rust(black_box(img), black_box(cfg)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpp_ffi", format!("{}x{}", size, size)),
            &(&image, &config),
            |b, (img, cfg)| {
                b.iter(|| encode_cpp_ffi(black_box(img), black_box(cfg)));
            },
        );
    }

    group.finish();
}

fn bench_quality_levels(c: &mut Criterion) {
    if !ffi_available() {
        eprintln!("SKIP: C++ jpegli FFI not available.");
        return;
    }

    let image = create_test_image(512, 512);

    let mut group = c.benchmark_group("quality_levels_ffi");
    group.throughput(Throughput::Elements((image.width * image.height) as u64));
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for quality in [50, 75, 90, 95] {
        let config = BenchConfig {
            mode: JpegMode::Progressive,
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
            BenchmarkId::new("cpp_ffi", format!("q{}", quality)),
            &(&image, &config),
            |b, (img, cfg)| {
                b.iter(|| encode_cpp_ffi(black_box(img), black_box(cfg)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Verification (run once before benchmarks)
// ============================================================================

fn verify_outputs_match(c: &mut Criterion) {
    if !ffi_available() {
        eprintln!("SKIP: C++ jpegli FFI not available.");
        // Still create a dummy benchmark
        let mut group = c.benchmark_group("verification");
        group.bench_function("skip", |b| b.iter(|| black_box(1 + 1)));
        group.finish();
        return;
    }

    let image = create_test_image(512, 512);
    let config = BenchConfig {
        mode: JpegMode::Progressive,
        subsampling: Subsampling::S420,
        use_xyb: false,
        quality: 90,
    };

    // Encode both
    let rust_jpeg = encode_rust(&image, &config);
    let cpp_jpeg = encode_cpp_ffi(&image, &config);

    // Decode both
    let rust_decoded = decode_jpeg(&rust_jpeg);
    let cpp_decoded = decode_jpeg(&cpp_jpeg);

    // Compare
    let max_diff = compute_max_diff(&rust_decoded, &cpp_decoded);
    let size_diff_pct =
        (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64) / cpp_jpeg.len() as f64 * 100.0;

    println!("\n=== FFI OUTPUT VERIFICATION ===");
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
