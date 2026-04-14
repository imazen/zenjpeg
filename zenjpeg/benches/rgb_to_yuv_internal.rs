//! Head-to-head benchmark: internal magetypes RGB→YUV vs the `yuv` crate's
//! Professional mode, across widths 256..4096.
//!
//! Run: `cargo bench --bench rgb_to_yuv_internal`
//! Results land in `target/criterion/rgb_to_yuv_444/*/new/estimates.json`.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use zenjpeg::color::rgb_to_yuv::{rgb_to_yuv420, rgb_to_yuv444};

use yuv::{
    BufferStoreMut, YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix, rgb_to_yuv420 as yuv_rgb_to_yuv420, rgb_to_yuv444 as yuv_rgb_to_yuv444,
};

/// Noise+patches pattern. Avoid smooth gradients — they produce degenerate
/// coefficient distributions that don't stress a codec path.
fn noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e3779b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let r = ((state >> 24) as u8).wrapping_add(patch * 40);
            let g = ((state >> 16) as u8).wrapping_add(patch * 80);
            let b = ((state >> 8) as u8).wrapping_add(patch * 120);
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    rgb
}

fn bench_444(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb_to_yuv_444");
    for &size in &[256usize, 512, 1024, 2048, 4096] {
        let rgb = noise_patches(size, size);
        let n = size * size;
        let mut y = vec![0u8; n];
        let mut u = vec![0u8; n];
        let mut v = vec![0u8; n];

        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("internal/{size}"), |b| {
            b.iter(|| {
                rgb_to_yuv444(
                    black_box(&rgb),
                    &mut y,
                    &mut u,
                    &mut v,
                    size,
                    size,
                );
                black_box(&y);
            })
        });

        group.bench_function(format!("yuv_crate/{size}"), |b| {
            b.iter(|| {
                let mut img = YuvPlanarImageMut {
                    y_plane: BufferStoreMut::Borrowed(&mut y[..]),
                    y_stride: size as u32,
                    u_plane: BufferStoreMut::Borrowed(&mut u[..]),
                    u_stride: size as u32,
                    v_plane: BufferStoreMut::Borrowed(&mut v[..]),
                    v_stride: size as u32,
                    width: size as u32,
                    height: size as u32,
                };
                yuv_rgb_to_yuv444(
                    &mut img,
                    black_box(&rgb),
                    (size * 3) as u32,
                    YuvRange::Full,
                    YuvStandardMatrix::Bt601,
                    YuvConversionMode::Professional,
                )
                .unwrap();
            })
        });
    }
    group.finish();
}

fn bench_420(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb_to_yuv_420");
    for &size in &[256usize, 512, 1024, 2048, 4096] {
        let rgb = noise_patches(size, size);
        let n = size * size;
        let cw = size / 2;
        let cn = cw * cw;
        let mut y = vec![0u8; n];
        let mut u = vec![0u8; cn];
        let mut v = vec![0u8; cn];

        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("internal/{size}"), |b| {
            b.iter(|| {
                rgb_to_yuv420(
                    black_box(&rgb),
                    &mut y,
                    &mut u,
                    &mut v,
                    size,
                    size,
                );
                black_box(&y);
            })
        });

        group.bench_function(format!("yuv_crate/{size}"), |b| {
            b.iter(|| {
                let mut img = YuvPlanarImageMut {
                    y_plane: BufferStoreMut::Borrowed(&mut y[..]),
                    y_stride: size as u32,
                    u_plane: BufferStoreMut::Borrowed(&mut u[..]),
                    u_stride: cw as u32,
                    v_plane: BufferStoreMut::Borrowed(&mut v[..]),
                    v_stride: cw as u32,
                    width: size as u32,
                    height: size as u32,
                };
                yuv_rgb_to_yuv420(
                    &mut img,
                    black_box(&rgb),
                    (size * 3) as u32,
                    YuvRange::Full,
                    YuvStandardMatrix::Bt601,
                    YuvConversionMode::Professional,
                )
                .unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_444, bench_420);
criterion_main!(benches);
