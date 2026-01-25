//! Benchmark comparing wide crate vs archmage-simd implementations
//!
//! Run with: cargo bench -p zenjpeg --bench mage_simd --features "archmage-simd"

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use archmage::{mem::avx, Avx2FmaToken, Avx2Token, AvxToken, SimdToken};

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use zenjpeg::encode::mage_simd::{
    mage_box_filter_2x2, mage_forward_dct_8x8, mage_gather_even_odd_x8, mage_rgb_to_ycbcr_8px,
    mage_transpose_8x8_inplace,
};

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
use std::arch::x86_64::__m256;

fn bench_dct(c: &mut Criterion) {
    let mut group = c.benchmark_group("DCT 8x8");
    group.throughput(Throughput::Elements(1));

    // Test data - typical image block values
    let input: [f32; 64] = std::array::from_fn(|i| (i as f32) * 0.5 - 16.0);

    // Benchmark the wide crate version (production path)
    group.bench_function("wide (production)", |b| {
        b.iter(|| {
            let result = zenjpeg::encode::dct::forward_dct_8x8(black_box(&input));
            black_box(result)
        })
    });

    // Benchmark archmage-simd version
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    if let Some(token) = Avx2FmaToken::try_new() {
        group.bench_function("archmage-simd", |b| {
            b.iter(|| {
                let mut output = [0.0f32; 64];
                mage_forward_dct_8x8(token, black_box(&input), &mut output);
                black_box(output)
            })
        });
    }

    group.finish();
}

fn bench_gather_even_odd(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gather Even/Odd x8");
    group.throughput(Throughput::Elements(16));

    let data: [f32; 16] = std::array::from_fn(|i| i as f32);

    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    if let Some(token) = Avx2Token::try_new() {
        group.bench_function("archmage-simd", |b| {
            b.iter(|| {
                let (evens, odds) = mage_gather_even_odd_x8(token, black_box(&data));
                black_box((evens, odds))
            })
        });

        // Scalar baseline for comparison
        group.bench_function("scalar", |b| {
            b.iter(|| {
                let data = black_box(&data);
                let evens = [
                    data[0], data[2], data[4], data[6], data[8], data[10], data[12], data[14],
                ];
                let odds = [
                    data[1], data[3], data[5], data[7], data[9], data[11], data[13], data[15],
                ];
                black_box((evens, odds))
            })
        });
    }

    group.finish();
}

fn bench_rgb_to_ycbcr(c: &mut Criterion) {
    let mut group = c.benchmark_group("RGB to YCbCr 8px");
    group.throughput(Throughput::Elements(8));

    let r_in = [128.0f32; 8];
    let g_in = [128.0f32; 8];
    let b_in = [128.0f32; 8];

    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    if let Some(token) = Avx2FmaToken::try_new() {
        group.bench_function("archmage-simd", |bencher| {
            bencher.iter(|| {
                let mut y = [0.0f32; 8];
                let mut cb = [0.0f32; 8];
                let mut cr = [0.0f32; 8];
                mage_rgb_to_ycbcr_8px(
                    token,
                    black_box(&r_in),
                    black_box(&g_in),
                    black_box(&b_in),
                    &mut y,
                    &mut cb,
                    &mut cr,
                );
                black_box((y, cb, cr))
            })
        });

        // Scalar baseline
        group.bench_function("scalar", |bencher| {
            bencher.iter(|| {
                let r = black_box(&r_in);
                let g = black_box(&g_in);
                let b = black_box(&b_in);
                let mut y = [0.0f32; 8];
                let mut cb = [0.0f32; 8];
                let mut cr = [0.0f32; 8];
                for i in 0..8 {
                    y[i] = 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
                    cb[i] = 128.0 - 0.168736 * r[i] - 0.331264 * g[i] + 0.5 * b[i];
                    cr[i] = 128.0 + 0.5 * r[i] - 0.418688 * g[i] - 0.081312 * b[i];
                }
                black_box((y, cb, cr))
            })
        });
    }

    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transpose 8x8");
    group.throughput(Throughput::Elements(64));

    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    if let Some(token) = AvxToken::try_new() {
        let data: [f32; 64] = std::array::from_fn(|i| i as f32);

        group.bench_function("archmage-simd", |b| {
            b.iter(|| {
                let mut reg: [__m256; 8] = [
                    avx::_mm256_loadu_ps(token, data[0..8].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[8..16].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[16..24].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[24..32].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[32..40].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[40..48].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[48..56].try_into().unwrap()),
                    avx::_mm256_loadu_ps(token, data[56..64].try_into().unwrap()),
                ];
                mage_transpose_8x8_inplace(token, black_box(&mut reg));
                black_box(reg)
            })
        });

        // Scalar transpose for baseline
        group.bench_function("scalar", |b| {
            b.iter(|| {
                let data = black_box(&data);
                let mut out = [0.0f32; 64];
                for row in 0..8 {
                    for col in 0..8 {
                        out[col * 8 + row] = data[row * 8 + col];
                    }
                }
                black_box(out)
            })
        });
    }

    group.finish();
}

fn bench_box_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("Box Filter 2x2");
    group.throughput(Throughput::Elements(8));

    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    if let Some(token) = AvxToken::try_new() {
        let row0: [f32; 16] = std::array::from_fn(|i| i as f32);
        let row1: [f32; 16] = std::array::from_fn(|i| (i + 16) as f32);

        group.bench_function("archmage-simd", |b| {
            b.iter(|| {
                let row0_evens = avx::_mm256_loadu_ps(token, row0[0..8].try_into().unwrap());
                let row0_odds = avx::_mm256_loadu_ps(token, row0[8..16].try_into().unwrap());
                let row1_evens = avx::_mm256_loadu_ps(token, row1[0..8].try_into().unwrap());
                let row1_odds = avx::_mm256_loadu_ps(token, row1[8..16].try_into().unwrap());

                let result = mage_box_filter_2x2(
                    token,
                    black_box(row0_evens),
                    black_box(row0_odds),
                    black_box(row1_evens),
                    black_box(row1_odds),
                );
                black_box(result)
            })
        });

        // Scalar baseline
        group.bench_function("scalar", |b| {
            b.iter(|| {
                let row0 = black_box(&row0);
                let row1 = black_box(&row1);
                let mut out = [0.0f32; 8];
                for i in 0..8 {
                    out[i] = (row0[i * 2] + row0[i * 2 + 1] + row1[i * 2] + row1[i * 2 + 1]) * 0.25;
                }
                black_box(out)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_dct,
    bench_gather_even_odd,
    bench_rgb_to_ycbcr,
    bench_transpose,
    bench_box_filter
);
criterion_main!(benches);
