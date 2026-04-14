//! Full-pipeline encode benchmark: measures complete RGB→JPEG at Q85.
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

    // 4K UHD: 3840x2160 = 8.3M pixels
    let rgb_4k: &'static [u8] = Box::leak(noise_patches(3840, 2160).into_boxed_slice());

    suite.group("encode_q85_4k", |g| {
        g.bench("4:4:4 progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
                config.encode_bytes(rgb_4k, 3840, 2160, PixelLayout::Rgb8Srgb).unwrap()
            })
        });

        g.bench("4:2:0 progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
                config.encode_bytes(rgb_4k, 3840, 2160, PixelLayout::Rgb8Srgb).unwrap()
            })
        });

        g.bench("4:2:0 baseline", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                    .progressive(false);
                config.encode_bytes(rgb_4k, 3840, 2160, PixelLayout::Rgb8Srgb).unwrap()
            })
        });

        g.bench("4:2:0 sharp progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
                    .sharp_yuv(true);
                config.encode_bytes(rgb_4k, 3840, 2160, PixelLayout::Rgb8Srgb).unwrap()
            })
        });
    });
}

zenbench::main!(bench_encode);
