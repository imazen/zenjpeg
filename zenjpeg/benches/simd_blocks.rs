//! Benchmark comparing SIMD block operations vs scalar.
//!
//! Tests the new Block8x8f and QuantTableSimd types.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use wide::f32x8;
use zenjpeg::foundation::simd_types::{Block8x8f, QuantTableSimd};

/// Old-style quantization (load/store dance)
fn quantize_scalar(block: &[f32; 64], quant: &[f32; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for i in 0..64 {
        result[i] = (block[i] * quant[i]).round() as i16;
    }
    result
}

/// Old-style with SIMD but array-based storage
fn quantize_simd_array(block: &[f32; 64], quant_recip: &[f32; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for row in 0..8 {
        let k = row * 8;
        // Load block coefficients
        let b = f32x8::from([
            block[k],
            block[k + 1],
            block[k + 2],
            block[k + 3],
            block[k + 4],
            block[k + 5],
            block[k + 6],
            block[k + 7],
        ]);
        // Load quant reciprocals
        let q = f32x8::from([
            quant_recip[k],
            quant_recip[k + 1],
            quant_recip[k + 2],
            quant_recip[k + 3],
            quant_recip[k + 4],
            quant_recip[k + 5],
            quant_recip[k + 6],
            quant_recip[k + 7],
        ]);
        // Multiply and round
        let quantized = (b * q).round_int();
        let arr: [i32; 8] = quantized.into();
        for (i, &v) in arr.iter().enumerate() {
            result[k + i] = v as i16;
        }
    }
    result
}

/// New-style with Block8x8f (no load overhead)
fn quantize_simd_native(block: &Block8x8f, quant: &QuantTableSimd) -> [i16; 64] {
    quant.quantize(block).to_i16_array()
}

fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_block");

    // Create test data
    let mut block_arr = [0.0f32; 64];
    let mut quant_arr = [0.0f32; 64];
    let mut quant_u16 = [1u16; 64];
    for i in 0..64 {
        block_arr[i] = ((i as f32) * 7.3 - 200.0).sin() * 500.0;
        quant_u16[i] = (i + 1) as u16;
        quant_arr[i] = 1.0 / quant_u16[i] as f32;
    }

    let block_simd = Block8x8f::from_array(&block_arr);
    let quant_simd = QuantTableSimd::from_values(&quant_u16);

    // Benchmark scalar
    group.bench_function("scalar", |b| {
        b.iter(|| quantize_scalar(black_box(&block_arr), black_box(&quant_arr)))
    });

    // Benchmark SIMD with array storage (old style)
    group.bench_function("simd_array", |b| {
        b.iter(|| quantize_simd_array(black_box(&block_arr), black_box(&quant_arr)))
    });

    // Benchmark SIMD with native storage (new style)
    group.bench_function("simd_native", |b| {
        b.iter(|| quantize_simd_native(black_box(&block_simd), black_box(&quant_simd)))
    });

    group.finish();
}

fn bench_block_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_ops");

    // Create test data
    let mut arr1 = [0.0f32; 64];
    let mut arr2 = [0.0f32; 64];
    for i in 0..64 {
        arr1[i] = (i as f32) * 1.5;
        arr2[i] = (i as f32) * 0.5 + 10.0;
    }

    let block1 = Block8x8f::from_array(&arr1);
    let block2 = Block8x8f::from_array(&arr2);

    // Benchmark from_array
    group.bench_function("from_array", |b| {
        b.iter(|| Block8x8f::from_array(black_box(&arr1)))
    });

    // Benchmark to_array
    group.bench_function("to_array", |b| b.iter(|| black_box(&block1).to_array()));

    // Benchmark mul
    group.bench_function("mul", |b| {
        b.iter(|| black_box(&block1).mul(black_box(&block2)))
    });

    // Benchmark scale
    group.bench_function("scale", |b| {
        b.iter(|| black_box(&block1).scale(black_box(0.125)))
    });

    group.finish();
}

criterion_group!(benches, bench_quantization, bench_block_operations);
criterion_main!(benches);
