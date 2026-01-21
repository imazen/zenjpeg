//! Benchmark comparing wide vs archmage for outer-level AQ functions
//!
//! Run with: cargo bench -p jpegli-rs --bench aq_simd --features "archmage-simd,test-utils"

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

// Import production AQ functions (wide-based, multiversed)
use jpegli::quant::aq::simd::{pre_erosion_row, pre_erosion_row_padded, per_block_modulations_row};

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use jpegli::quant::aq::simd::mage_pre_erosion_row_padded;

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use archmage::{arcane, mem::avx, Avx2FmaToken, HasAvx2, HasFma, SimdToken};

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use jpegli::encode::mage_simd::mage_pre_erosion_pixel_x8;

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Archmage version of pre_erosion_row - mirrors production loop structure
#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
#[arcane]
fn mage_pre_erosion_row<T: HasAvx2 + HasFma + Copy>(
    token: T,
    row: &[f32],
    row_above: &[f32],
    row_below: &[f32],
    output: &mut [f32],
) {
    let width = row.len();
    if width == 0 {
        return;
    }

    let chunks = width / 8;

    for chunk in 0..chunks {
        let x = chunk * 8;

        // Load center pixels
        let pixels = avx::_mm256_loadu_ps(token, (&row[x..x+8]).try_into().unwrap());

        // Load neighbors (simplified - skip boundary handling for benchmark)
        let left = if x == 0 {
            avx::_mm256_loadu_ps(token, (&row[0..8]).try_into().unwrap())
        } else {
            avx::_mm256_loadu_ps(token, (&row[x-1..x+7]).try_into().unwrap())
        };

        let right = if x + 9 > width {
            avx::_mm256_loadu_ps(token, (&row[x..x+8]).try_into().unwrap())
        } else {
            avx::_mm256_loadu_ps(token, (&row[x+1..x+9]).try_into().unwrap())
        };

        let top = avx::_mm256_loadu_ps(token, (&row_above[x..x+8]).try_into().unwrap());
        let bottom = avx::_mm256_loadu_ps(token, (&row_below[x..x+8]).try_into().unwrap());

        // Compute using archmage primitive
        let result = mage_pre_erosion_pixel_x8(token, pixels, left, right, top, bottom);

        // Load existing, add result, store back
        let existing = avx::_mm256_loadu_ps(token, (&output[x..x+8]).try_into().unwrap());
        let sum = _mm256_add_ps(existing, result);
        avx::_mm256_storeu_ps(token, (&mut output[x..x+8]).try_into().unwrap(), sum);
    }
}

fn bench_pre_erosion_row(c: &mut Criterion) {
    for width in [64, 256, 1024, 4096] {
        let mut group = c.benchmark_group(format!("AQ pre_erosion_row width={}", width));
        group.throughput(Throughput::Elements(width as u64));

        let row: Vec<f32> = (0..width).map(|i| (i % 256) as f32).collect();
        let row_above: Vec<f32> = (0..width).map(|i| ((i + 1) % 256) as f32).collect();
        let row_below: Vec<f32> = (0..width).map(|i| ((i + 2) % 256) as f32).collect();
        let mut output = vec![0.0f32; width];

        group.bench_function("wide+multiversed", |b| {
            b.iter(|| {
                output.fill(0.0);
                pre_erosion_row(
                    black_box(&row),
                    black_box(&row_above),
                    black_box(&row_below),
                    &mut output,
                );
                black_box(output[0])
            })
        });

        #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
        if let Some(token) = Avx2FmaToken::try_new() {
            let mut output_mage = vec![0.0f32; width];
            group.bench_function("archmage loop", |b| {
                b.iter(|| {
                    output_mage.fill(0.0);
                    mage_pre_erosion_row(
                        token,
                        black_box(&row),
                        black_box(&row_above),
                        black_box(&row_below),
                        &mut output_mage,
                    );
                    black_box(output_mage[0])
                })
            });
        }

        group.finish();
    }
}

fn bench_pre_erosion_row_padded(c: &mut Criterion) {
    for width in [64, 256, 1024, 4096] {
        let mut group = c.benchmark_group(format!("AQ pre_erosion_row_padded width={}", width));
        group.throughput(Throughput::Elements(width as u64));

        let row: Vec<f32> = (0..width).map(|i| (i % 256) as f32).collect();
        let row_above: Vec<f32> = (0..width).map(|i| ((i + 1) % 256) as f32).collect();
        let row_below: Vec<f32> = (0..width).map(|i| ((i + 2) % 256) as f32).collect();

        // Create padded buffers with edge replication
        let mut row_padded = vec![0.0f32; width + 2];
        let mut row_above_padded = vec![0.0f32; width + 2];
        let mut row_below_padded = vec![0.0f32; width + 2];

        row_padded[1..1 + width].copy_from_slice(&row);
        row_padded[0] = row[0];
        row_padded[width + 1] = row[width - 1];

        row_above_padded[1..1 + width].copy_from_slice(&row_above);
        row_above_padded[0] = row_above[0];
        row_above_padded[width + 1] = row_above[width - 1];

        row_below_padded[1..1 + width].copy_from_slice(&row_below);
        row_below_padded[0] = row_below[0];
        row_below_padded[width + 1] = row_below[width - 1];

        let mut output = vec![0.0f32; width];

        group.bench_function("wide+multiversed padded", |b| {
            b.iter(|| {
                output.fill(0.0);
                pre_erosion_row_padded(
                    black_box(&row_padded),
                    black_box(&row_above_padded),
                    black_box(&row_below_padded),
                    black_box(width),
                    &mut output,
                );
                black_box(output[0])
            })
        });

        #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
        if let Some(token) = Avx2FmaToken::try_new() {
            let mut output_mage = vec![0.0f32; width];
            group.bench_function("archmage padded", |b| {
                b.iter(|| {
                    output_mage.fill(0.0);
                    mage_pre_erosion_row_padded(
                        token,
                        black_box(&row_padded),
                        black_box(&row_above_padded),
                        black_box(&row_below_padded),
                        black_box(width),
                        &mut output_mage,
                    );
                    black_box(output_mage[0])
                })
            });
        }

        group.finish();
    }
}

fn bench_per_block_modulations_row(c: &mut Criterion) {
    for blocks_w in [8, 32, 128, 512] {
        let width = blocks_w * 8;
        let height = 64;
        let mut group = c.benchmark_group(format!("AQ per_block_modulations_row blocks={}", blocks_w));
        group.throughput(Throughput::Elements((blocks_w * 64) as u64));

        let stride = width + 8;
        let y_data: Vec<f32> = (0..(stride * (height + 1)))
            .map(|i| (i % 256) as f32)
            .collect();

        let mut aq_row: Vec<f32> = (0..blocks_w)
            .map(|i| 0.5 + (i % 100) as f32 * 0.01)
            .collect();

        let mul = 1.0f32;
        let add = 0.0f32;

        group.bench_function("wide+multiversed", |b| {
            b.iter(|| {
                per_block_modulations_row(
                    black_box(&y_data),
                    black_box(stride),
                    black_box(width),
                    black_box(height),
                    black_box(0),
                    black_box(blocks_w),
                    &mut aq_row,
                    black_box(mul),
                    black_box(add),
                );
                black_box(aq_row[0])
            })
        });

        // Note: No archmage version of per_block_modulations_row yet
        // Would require porting the full function with all its complexity

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_pre_erosion_row,
    bench_pre_erosion_row_padded,
    bench_per_block_modulations_row,
);
criterion_main!(benches);
