//! Comprehensive encoding benchmark matrix.
//!
//! Tests 2K/4K resolutions with different configurations:
//! - Subsampling: 4:4:4 and 4:2:0
//! - Mode: Baseline and Progressive
//! - Huffman: Standard and Optimized
//!
//! No I/O or decode operations - pure encode timing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jpegli::{Encoder, JpegMode, PixelFormat, Quality, Subsampling};

/// Create a test image with realistic pixel patterns.
///
/// Uses gradients and patterns that exercise the encoder's DCT and quantization
/// without the overhead of loading actual image files.
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            // Create gradients and patterns that test edge detection
            let x_grad = ((x * 255) / width) as u8;
            let y_grad = ((y * 255) / height) as u8;
            // Add some high-frequency patterns
            let checker = if ((x / 8) + (y / 8)) % 2 == 0 { 32 } else { 0 };
            data[idx] = x_grad.saturating_add(checker); // R
            data[idx + 1] = y_grad; // G
            data[idx + 2] = ((x_grad as u16 + y_grad as u16) / 2) as u8; // B
        }
    }
    data
}

/// Encode configuration for benchmarking.
#[derive(Clone, Copy)]
struct EncodeConfig {
    subsampling: Subsampling,
    mode: JpegMode,
    optimize_huffman: bool,
}

impl EncodeConfig {
    fn name(&self) -> String {
        let sub = match self.subsampling {
            Subsampling::S444 => "444",
            Subsampling::S420 => "420",
            _ => "other",
        };
        let mode = match self.mode {
            JpegMode::Baseline => "baseline",
            JpegMode::Progressive => "prog",
            _ => "other",
        };
        let huff = if self.optimize_huffman { "opt" } else { "std" };
        format!("{}_{}_{}", sub, mode, huff)
    }
}

fn bench_encode_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_matrix");
    group.sample_size(20); // Reduce samples for large images

    // Resolution presets
    let resolutions = [("2k", 1920, 1080), ("4k", 3840, 2160)];

    // Configuration matrix
    let configs = [
        // YCbCr 4:4:4 configurations
        EncodeConfig {
            subsampling: Subsampling::S444,
            mode: JpegMode::Baseline,
            optimize_huffman: false,
        },
        EncodeConfig {
            subsampling: Subsampling::S444,
            mode: JpegMode::Baseline,
            optimize_huffman: true,
        },
        EncodeConfig {
            subsampling: Subsampling::S444,
            mode: JpegMode::Progressive,
            optimize_huffman: false,
        },
        EncodeConfig {
            subsampling: Subsampling::S444,
            mode: JpegMode::Progressive,
            optimize_huffman: true,
        },
        // YCbCr 4:2:0 configurations
        EncodeConfig {
            subsampling: Subsampling::S420,
            mode: JpegMode::Baseline,
            optimize_huffman: false,
        },
        EncodeConfig {
            subsampling: Subsampling::S420,
            mode: JpegMode::Baseline,
            optimize_huffman: true,
        },
        EncodeConfig {
            subsampling: Subsampling::S420,
            mode: JpegMode::Progressive,
            optimize_huffman: false,
        },
        EncodeConfig {
            subsampling: Subsampling::S420,
            mode: JpegMode::Progressive,
            optimize_huffman: true,
        },
    ];

    for (res_name, width, height) in resolutions {
        // Pre-allocate test image once per resolution
        let data = create_test_image(width, height);
        let megapixels = (width * height) as f64 / 1_000_000.0;

        for config in &configs {
            let bench_name = format!("{}/{}", res_name, config.name());

            group.throughput(Throughput::Elements(1)); // One image per iteration

            group.bench_with_input(BenchmarkId::new("encode", &bench_name), &data, |b, data| {
                b.iter(|| {
                    let encoder = Encoder::new()
                        .width(width as u32)
                        .height(height as u32)
                        .pixel_format(PixelFormat::Rgb)
                        .jpegli_quality(Quality::from_quality(90.0))
                        .subsampling(config.subsampling)
                        .mode(config.mode)
                        .optimize_huffman(config.optimize_huffman);
                    encoder.encode(black_box(data))
                });
            });
        }

        // Print megapixels for context
        eprintln!("{}: {:.2} MP", res_name, megapixels);
    }

    group.finish();
}

/// Focused benchmark comparing only the most common configurations.
fn bench_encode_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_quick");
    group.sample_size(30);

    // 4K image for realistic workload
    let width = 3840;
    let height = 2160;
    let data = create_test_image(width, height);

    // Most common production configurations
    let configs = [
        (
            "420_prog_opt",
            Subsampling::S420,
            JpegMode::Progressive,
            true,
        ),
        (
            "444_baseline_opt",
            Subsampling::S444,
            JpegMode::Baseline,
            true,
        ),
    ];

    for (name, sub, mode, opt) in configs {
        group.bench_with_input(BenchmarkId::new("4k", name), &data, |b, data| {
            b.iter(|| {
                let encoder = Encoder::new()
                    .width(width as u32)
                    .height(height as u32)
                    .pixel_format(PixelFormat::Rgb)
                    .jpegli_quality(Quality::from_quality(90.0))
                    .subsampling(sub)
                    .mode(mode)
                    .optimize_huffman(opt);
                encoder.encode(black_box(data))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_encode_matrix, bench_encode_quick);
criterion_main!(benches);
