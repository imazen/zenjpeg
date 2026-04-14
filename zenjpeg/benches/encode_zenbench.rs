//! Full-pipeline encode benchmark: measures complete RGB→JPEG at Q85.
//! Compares 4:4:4 vs 4:2:0, with and without sharp_yuv.
//!
//! Run: `cargo bench --bench encode_zenbench`

use zenbench::prelude::*;

fn noise_patches(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    let mut state = 0x9e37_79b9u32;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let patch = ((x / 32 + y / 32) & 3) as u8;
            let i = (y * w + x) * 3;
            rgb[i] = ((state >> 24) as u8).wrapping_add(patch.wrapping_mul(40));
            rgb[i + 1] = ((state >> 16) as u8).wrapping_add(patch.wrapping_mul(80));
            rgb[i + 2] = ((state >> 8) as u8).wrapping_add(patch.wrapping_mul(120));
        }
    }
    rgb
}

fn bench_encode(suite: &mut Suite) {
    use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
    use zenjpeg::encode::EncoderConfig;

    for &size in &[256usize, 512, 1024] {
        let rgb: &'static [u8] = Box::leak(noise_patches(size, size).into_boxed_slice());
        let w = size as u32;
        let h = size as u32;

        suite.group(format!("encode_q85/{size}"), |g| {
            g.bench("4:4:4", move |b| {
                b.iter(|| {
                    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
                    config.encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb).unwrap()
                })
            });

            g.bench("4:2:0", move |b| {
                b.iter(|| {
                    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
                    config.encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb).unwrap()
                })
            });

            g.bench("4:2:0 sharp", move |b| {
                b.iter(|| {
                    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                        .sharp_yuv(true);
                    config.encode_bytes(rgb, w, h, PixelLayout::Rgb8Srgb).unwrap()
                })
            });
        });
    }
}

zenbench::main!(bench_encode);
