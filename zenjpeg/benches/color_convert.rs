//! Focused micro-benchmarks for YCbCr→RGB color conversion.
//!
//! These benchmarks isolate the color conversion hot paths to detect
//! regressions when porting from hand-written AVX2 intrinsics to
//! magetypes generics.
//!
//! # Usage
//!
//! ```bash
//! # Run all color conversion benchmarks
//! cargo bench --bench color_convert -p zenjpeg
//!
//! # Run and save baseline before refactoring
//! cargo bench --bench color_convert -p zenjpeg -- --save-baseline avx2-baseline
//!
//! # Compare after refactoring
//! cargo bench --bench color_convert -p zenjpeg -- --baseline avx2-baseline
//! ```

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use zenjpeg::color::ycbcr;

/// Generate realistic i16 YCbCr test data (post-IDCT values).
/// Y range: roughly 0-255 (luma), Cb/Cr: roughly 0-255 (centered at 128).
fn generate_ycbcr_data(width: usize) -> (Vec<i16>, Vec<i16>, Vec<i16>) {
    let mut y = vec![0i16; width];
    let mut cb = vec![0i16; width];
    let mut cr = vec![0i16; width];
    for i in 0..width {
        // Varied but realistic post-IDCT values
        y[i] = ((i * 7 + 30) % 256) as i16;
        cb[i] = (((i * 3 + 100) % 256) as i16).wrapping_sub(0); // 0-255 range
        cr[i] = (((i * 11 + 60) % 256) as i16).wrapping_sub(0);
    }
    (y, cb, cr)
}

fn bench_ycbcr_to_rgb_i16_x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("ycbcr_to_rgb_i16_x16");

    // Bench at multiple widths to see scaling
    for &width in &[256, 1024, 4096] {
        let (y_data, cb_data, cr_data) = generate_ycbcr_data(width);
        let pixels = width as u64;
        group.throughput(Throughput::Elements(pixels));

        group.bench_function(format!("{}px", width), |b| {
            let mut rgb = vec![0u8; width * 3];
            b.iter(|| {
                let mut offset = 0usize;
                for chunk_start in (0..width).step_by(16) {
                    let y16: &[i16; 16] = y_data[chunk_start..chunk_start + 16].try_into().unwrap();
                    let cb16: &[i16; 16] =
                        cb_data[chunk_start..chunk_start + 16].try_into().unwrap();
                    let cr16: &[i16; 16] =
                        cr_data[chunk_start..chunk_start + 16].try_into().unwrap();
                    ycbcr::ycbcr_to_rgb_i16_x16(
                        black_box(y16),
                        black_box(cb16),
                        black_box(cr16),
                        &mut rgb,
                        &mut offset,
                    );
                }
                black_box(&rgb);
            });
        });
    }

    group.finish();
}

fn bench_fused_h2v2_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("fused_h2v2_box_ycbcr_to_rgb");

    for &width in &[256, 1024, 4096] {
        let (y_data, cb_data_full, cr_data_full) = generate_ycbcr_data(width);
        // 4:2:0 box: chroma is half-width
        let chroma_width = (width + 1) / 2;
        let cb_data = &cb_data_full[..chroma_width];
        let cr_data = &cr_data_full[..chroma_width];
        let pixels = width as u64;
        group.throughput(Throughput::Elements(pixels));

        group.bench_function(format!("{}px", width), |b| {
            let mut rgb = vec![0u8; width * 3];
            b.iter(|| {
                ycbcr::fused_h2v2_box_ycbcr_to_rgb_u8(
                    black_box(&y_data),
                    black_box(cb_data),
                    black_box(cr_data),
                    &mut rgb,
                    width,
                    false,
                );
                black_box(&rgb);
            });
        });
    }

    group.finish();
}

/// Full decode benchmark — measures end-to-end impact including color conversion.
fn bench_decode_420(c: &mut Criterion) {
    use enough::Unstoppable;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    let mut group = c.benchmark_group("decode_420_e2e");

    let width = 1024u32;
    let height = 1024u32;
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            pixels[idx] = ((x * 7 + y * 3) % 256) as u8;
            pixels[idx + 1] = ((x * 3 + y * 11 + 128) % 256) as u8;
            pixels[idx + 2] = ((x * 13 + y * 5 + 64) % 256) as u8;
        }
    }

    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();

    let mpix = (width as u64 * height as u64) / 1_000_000;
    group.throughput(Throughput::Elements(mpix));

    group.bench_function("1024x1024", |b| {
        b.iter(|| {
            let decoder = zenjpeg::decoder::Decoder::new().auto_orient(false);
            decoder.decode(black_box(&jpeg), Unstoppable).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ycbcr_to_rgb_i16_x16,
    bench_fused_h2v2_box,
    bench_decode_420,
);
criterion_main!(benches);
