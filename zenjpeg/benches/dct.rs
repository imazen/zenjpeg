//! DCT benchmark comparing recursive (jpegli) and AAN (libjpeg) implementations.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench dct
//! ```

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use zenjpeg::encode::dct::{aan_forward_dct_8x8, forward_dct_8x8};

/// Generate test blocks with realistic patterns
fn generate_test_blocks(count: usize) -> Vec<[f32; 64]> {
    let mut blocks = Vec::with_capacity(count);
    for i in 0..count {
        let mut block = [0.0f32; 64];
        for row in 0..8 {
            for col in 0..8 {
                let idx = row * 8 + col;
                // Mix of gradient, noise, and block-specific variation
                let base = ((i * 17 + row * 13 + col * 7) % 256) as f32;
                let gradient = (row * 8 + col) as f32 * 0.5;
                let noise = ((i ^ row ^ col) % 32) as f32;
                block[idx] = (base + gradient + noise - 128.0).clamp(-128.0, 127.0);
            }
        }
        blocks.push(block);
    }
    blocks
}

fn dct_bench(c: &mut Criterion) {
    // Simulate 4K image: 3840x2160 = 8.3M pixels = ~130k blocks (for Y plane at 4:2:0)
    // Using fewer blocks for reasonable benchmark time
    let num_blocks = 16384; // ~1 megapixel worth
    let blocks = generate_test_blocks(num_blocks);

    let mut group = c.benchmark_group("dct");
    group.throughput(Throughput::Elements(num_blocks as u64));
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(100);

    // Benchmark recursive DCT (current implementation)
    group.bench_function("recursive", |b| {
        b.iter(|| {
            for block in &blocks {
                black_box(forward_dct_8x8(black_box(block)));
            }
        })
    });

    // Benchmark AAN DCT
    group.bench_function("aan", |b| {
        b.iter(|| {
            for block in &blocks {
                black_box(aan_forward_dct_8x8(black_box(block)));
            }
        })
    });

    group.finish();
}

fn dct_single_block_bench(c: &mut Criterion) {
    // Single block benchmark for latency measurement
    let block = generate_test_blocks(1)[0];

    let mut group = c.benchmark_group("dct_single");
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("recursive", |b| {
        b.iter(|| black_box(forward_dct_8x8(black_box(&block))))
    });

    group.bench_function("aan", |b| {
        b.iter(|| black_box(aan_forward_dct_8x8(black_box(&block))))
    });

    group.finish();
}

criterion_group!(benches, dct_bench, dct_single_block_bench);
criterion_main!(benches);
