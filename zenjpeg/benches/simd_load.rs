//! Benchmark: Safe vs Unsafe SIMD Loading Patterns
//!
//! Tests whether `f32x8::from(slice.try_into().unwrap())` optimizes to the same
//! code as `f32x8::from(unsafe { *(ptr as *const [f32; 8]) })`.
//!
//! Run with: cargo bench --bench simd_load

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use wide::f32x8;

// Test buffer size - large enough to be realistic, small enough to fit in L2 cache
const BUFFER_SIZE: usize = 256 * 256; // 64K floats = 256KB

/// Generate test data - smooth gradient with some variation
fn generate_test_data() -> Vec<f32> {
    (0..BUFFER_SIZE)
        .map(|i| 100.0 + (i as f32) * 0.01 + ((i as f32) * 0.1).sin() * 10.0)
        .collect()
}

/// Approach 1: Unsafe pointer cast (current production code)
#[inline(always)]
fn load_unsafe(data: &[f32], offset: usize) -> f32x8 {
    f32x8::from(unsafe { *(data.as_ptr().add(offset) as *const [f32; 8]) })
}

/// Approach 2: Safe try_into with unwrap
#[inline(always)]
fn load_safe_try_into(data: &[f32], offset: usize) -> f32x8 {
    f32x8::from(<[f32; 8]>::try_from(&data[offset..offset + 8]).unwrap())
}

/// Approach 3: Safe with explicit indexing (worst case baseline)
#[inline(always)]
fn load_indexed(data: &[f32], offset: usize) -> f32x8 {
    f32x8::from([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ])
}

/// Approach 4: Copy to stack array first
#[inline(always)]
fn load_copy_to_array(data: &[f32], offset: usize) -> f32x8 {
    let mut arr = [0.0f32; 8];
    arr.copy_from_slice(&data[offset..offset + 8]);
    f32x8::from(arr)
}

/// Process entire buffer using a given loading strategy.
/// Does actual SIMD work (multiply-accumulate) to prevent dead code elimination.
fn process_buffer<F>(data: &[f32], load_fn: F) -> f32
where
    F: Fn(&[f32], usize) -> f32x8,
{
    let mut acc = f32x8::ZERO;
    let scale = f32x8::splat(0.5);

    // Process in chunks of 8
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let v = load_fn(data, i * 8);
        acc += v * scale; // Actual SIMD work
    }

    // Reduce to scalar (prevents acc from being optimized away)
    let arr: [f32; 8] = acc.into();
    arr.iter().sum()
}

/// More realistic workload: simulate pre_erosion_row pattern
/// Loads center, left, right, top, bottom - 5 loads per iteration
fn simulate_row_processing<F>(row: &[f32], row_above: &[f32], row_below: &[f32], load_fn: F) -> f32
where
    F: Fn(&[f32], usize) -> f32x8,
{
    let mut acc = f32x8::ZERO;
    let quarter = f32x8::splat(0.25);

    let chunks = (row.len() / 8).saturating_sub(1); // Leave margin for left/right

    for i in 1..chunks {
        let x = i * 8;

        let center = load_fn(row, x);
        let left = load_fn(row, x - 1);
        let right = load_fn(row, x + 1);
        let top = load_fn(row_above, x);
        let bottom = load_fn(row_below, x);

        // Simulate base calculation: 0.25 * (left + right + top + bottom)
        let base = (left + right + top + bottom) * quarter;
        let diff = center - base;
        acc += diff * diff; // Accumulate squared differences
    }

    let arr: [f32; 8] = acc.into();
    arr.iter().sum()
}

fn bench_simple_load(c: &mut Criterion) {
    let data = generate_test_data();

    let mut group = c.benchmark_group("simd_load_simple");
    group.throughput(Throughput::Bytes((BUFFER_SIZE * 4) as u64));

    group.bench_function("unsafe_ptr", |b| {
        b.iter(|| process_buffer(black_box(&data), load_unsafe))
    });

    group.bench_function("safe_try_into", |b| {
        b.iter(|| process_buffer(black_box(&data), load_safe_try_into))
    });

    group.bench_function("indexed", |b| {
        b.iter(|| process_buffer(black_box(&data), load_indexed))
    });

    group.bench_function("copy_to_array", |b| {
        b.iter(|| process_buffer(black_box(&data), load_copy_to_array))
    });

    group.finish();
}

fn bench_row_processing(c: &mut Criterion) {
    // Simulate image row processing (1920 width = typical HD)
    let width = 1920;
    let row: Vec<f32> = (0..width).map(|i| 128.0 + (i as f32) * 0.5).collect();
    let row_above: Vec<f32> = (0..width).map(|i| 126.0 + (i as f32) * 0.5).collect();
    let row_below: Vec<f32> = (0..width).map(|i| 130.0 + (i as f32) * 0.5).collect();

    let mut group = c.benchmark_group("simd_load_row");
    group.throughput(Throughput::Elements(width as u64));

    group.bench_function("unsafe_ptr", |b| {
        b.iter(|| {
            simulate_row_processing(
                black_box(&row),
                black_box(&row_above),
                black_box(&row_below),
                load_unsafe,
            )
        })
    });

    group.bench_function("safe_try_into", |b| {
        b.iter(|| {
            simulate_row_processing(
                black_box(&row),
                black_box(&row_above),
                black_box(&row_below),
                load_safe_try_into,
            )
        })
    });

    group.bench_function("indexed", |b| {
        b.iter(|| {
            simulate_row_processing(
                black_box(&row),
                black_box(&row_above),
                black_box(&row_below),
                load_indexed,
            )
        })
    });

    group.bench_function("copy_to_array", |b| {
        b.iter(|| {
            simulate_row_processing(
                black_box(&row),
                black_box(&row_above),
                black_box(&row_below),
                load_copy_to_array,
            )
        })
    });

    group.finish();
}

fn bench_varying_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_load_sizes");

    for size in [1024, 4096, 16384, 65536, 262144] {
        let data: Vec<f32> = (0..size).map(|i| 100.0 + (i as f32) * 0.01).collect();

        group.throughput(Throughput::Bytes((size * 4) as u64));

        group.bench_with_input(BenchmarkId::new("unsafe_ptr", size), &data, |b, data| {
            b.iter(|| process_buffer(black_box(data), load_unsafe))
        });

        group.bench_with_input(BenchmarkId::new("safe_try_into", size), &data, |b, data| {
            b.iter(|| process_buffer(black_box(data), load_safe_try_into))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simple_load,
    bench_row_processing,
    bench_varying_sizes,
);

criterion_main!(benches);
