//! Head-to-head: internal magetypes RGB→YCbCr vs `yuv` crate Professional,
//! measured with zenbench (interleaved paired execution, not back-to-back
//! runs — kills thermal/turbo bias that criterion bakes in).
//!
//! Run: `cargo bench --bench rgb_to_yuv_internal`

use zenbench::prelude::*;

use zenyuv::{rgb_to_yuv420, rgb_to_yuv444};

use yuv::{
    BufferStoreMut, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
    rgb_to_yuv420 as yuv_rgb_to_yuv420, rgb_to_yuv444 as yuv_rgb_to_yuv444,
};

/// Noise + patches — realistic DCT coefficient distribution, avoids
/// the degenerate 0/±1 coefficients smooth gradients produce.
fn noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let r = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            let g = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            let b = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    rgb
}

fn bench_444(suite: &mut Suite) {
    for &size in &[256usize, 512, 1024, 2048, 4096] {
        suite.group(format!("rgb_to_yuv_444/{size}"), |g| {
            let rgb: &'static [u8] = Box::leak(noise_patches(size, size).into_boxed_slice());
            let n = size * size;

            g.bench("internal (magetypes f32+FMA)", move |b| {
                let mut y = vec![0u8; n];
                let mut u = vec![0u8; n];
                let mut v = vec![0u8; n];
                b.iter(|| {
                    rgb_to_yuv444(rgb, &mut y, &mut u, &mut v, size, size);
                })
            });

            g.bench("yuv crate Professional", move |b| {
                let mut y = vec![0u8; n];
                let mut u = vec![0u8; n];
                let mut v = vec![0u8; n];
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
                        rgb,
                        (size * 3) as u32,
                        YuvRange::Full,
                        YuvStandardMatrix::Bt601,
                        YuvConversionMode::Professional,
                    )
                    .unwrap();
                })
            });
        });
    }
}

fn bench_420(suite: &mut Suite) {
    for &size in &[256usize, 512, 1024, 2048, 4096] {
        suite.group(format!("rgb_to_yuv_420/{size}"), |g| {
            let rgb: &'static [u8] = Box::leak(noise_patches(size, size).into_boxed_slice());
            let n = size * size;
            let cw = size / 2;
            let cn = cw * cw;

            g.bench("internal (magetypes f32+FMA)", move |b| {
                let mut y = vec![0u8; n];
                let mut u = vec![0u8; cn];
                let mut v = vec![0u8; cn];
                b.iter(|| {
                    rgb_to_yuv420(rgb, &mut y, &mut u, &mut v, size, size);
                })
            });

            g.bench("yuv crate Professional", move |b| {
                let mut y = vec![0u8; n];
                let mut u = vec![0u8; cn];
                let mut v = vec![0u8; cn];
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
                        rgb,
                        (size * 3) as u32,
                        YuvRange::Full,
                        YuvStandardMatrix::Bt601,
                        YuvConversionMode::Professional,
                    )
                    .unwrap();
                })
            });
        });
    }
}

fn bench_all(suite: &mut Suite) {
    bench_444(suite);
    bench_420(suite);
}

zenbench::main!(bench_all);
