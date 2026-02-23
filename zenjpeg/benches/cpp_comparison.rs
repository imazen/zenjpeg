//! Criterion benchmarks comparing Rust zenjpeg vs C++ cjpegli via FFI.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench cpp_comparison
//! cargo bench --bench cpp_comparison -- --save-baseline main
//! cargo bench --bench cpp_comparison -- --baseline main
//! ```
//!
//! IMPORTANT: Uses distance-based encoding for fair comparison.
//! C++ jpegli's `jpeg_set_quality()` uses 2 chroma tables, while
//! `jpegli_set_distance()` uses 3 tables matching Rust's behavior.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;
use zenjpeg::encoder::{
    ChromaSubsampling as RustSubsampling, EncoderConfig as RustConfig, PixelLayout, Quality,
    Unstoppable,
};
use zenjpeg_bench_utils::{
    ChromaSubsampling, ColorMode, EncoderConfig, EncoderImpl, ImageData, ScanMode, SyntheticPattern,
};

/// Convert quality (0-100) to butteraugli distance.
/// Same formula as C++ jpegli_quality_to_distance.
fn quality_to_distance(q: f32) -> f32 {
    if q >= 100.0 {
        0.01
    } else if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}

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
    progressive: bool,
    subsampling: ChromaSubsampling,
    use_xyb: bool,
    quality: u8,
}

impl BenchConfig {
    fn name(&self) -> String {
        let mode = if self.progressive { "prog" } else { "base" };
        let sub = match self.subsampling {
            ChromaSubsampling::S444 => "444",
            ChromaSubsampling::S422 => "422",
            ChromaSubsampling::S420 => "420",
            ChromaSubsampling::S440 => "440",
        };
        let color = if self.use_xyb { "xyb" } else { "ycbcr" };
        format!("{}-{}-{}-q{}", mode, sub, color, self.quality)
    }

    /// Get butteraugli distance (for 3-table parity with Rust)
    fn distance(self) -> f32 {
        quality_to_distance(self.quality as f32)
    }

    fn to_encoder_config(self) -> EncoderConfig {
        EncoderConfig::new(EncoderImpl::CJpegli)
            .color(if self.use_xyb {
                ColorMode::Xyb
            } else {
                ColorMode::YCbCr
            })
            .scan(if self.progressive {
                ScanMode::Progressive
            } else {
                ScanMode::Baseline
            })
            .subsampling(self.subsampling)
            .distance(self.distance()) // Use distance for 3-table parity
    }

    fn to_rust_subsampling(self) -> RustSubsampling {
        match self.subsampling {
            ChromaSubsampling::S444 => RustSubsampling::None,
            ChromaSubsampling::S422 => RustSubsampling::HalfHorizontal,
            ChromaSubsampling::S420 => RustSubsampling::Quarter,
            ChromaSubsampling::S440 => RustSubsampling::HalfVertical,
        }
    }
}

fn encode_rust(image: &ImageData, config: &BenchConfig) -> Vec<u8> {
    let rust_config = RustConfig::ycbcr(
        Quality::ApproxButteraugli(config.distance()),
        config.to_rust_subsampling(),
    )
    .progressive(config.progressive);

    let mut enc = rust_config
        .encode_from_bytes(
            image.width as u32,
            image.height as u32,
            PixelLayout::Rgb8Srgb,
        )
        .expect("Failed to create encoder");
    enc.push_packed(&image.pixels, Unstoppable)
        .expect("Rust encode failed");
    enc.finish().expect("Rust finish failed")
}

fn encode_cpp_ffi(image: &ImageData, config: &BenchConfig) -> Vec<u8> {
    config
        .to_encoder_config()
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
        .distance(quality_to_distance(90.0)); // Use distance for 3-table parity

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
            progressive: false,
            subsampling: ChromaSubsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        BenchConfig {
            progressive: false,
            subsampling: ChromaSubsampling::S444,
            use_xyb: false,
            quality: 90,
        },
        // Progressive YCbCr
        BenchConfig {
            progressive: true,
            subsampling: ChromaSubsampling::S420,
            use_xyb: false,
            quality: 90,
        },
        BenchConfig {
            progressive: true,
            subsampling: ChromaSubsampling::S444,
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
        progressive: true,
        subsampling: ChromaSubsampling::S420,
        use_xyb: false,
        quality: 90,
    };

    let mut group = c.benchmark_group("size_scaling_ffi");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for size in [1024, 2048, 4096] {
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
            progressive: true,
            subsampling: ChromaSubsampling::S420,
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
        progressive: true,
        subsampling: ChromaSubsampling::S420,
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
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::bytestream::ZCursor;
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
