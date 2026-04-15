//! Full-pipeline encode benchmark + isolated sharp YUV component timing.
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
    use zenjpeg::encode::EncoderConfig;
    use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout, XybSubsampling};

    let rgb_4k: &'static [u8] = Box::leak(noise_patches(3840, 2160).into_boxed_slice());
    let rgb_1k_xyb: &'static [u8] = Box::leak(noise_patches(1024, 1024).into_boxed_slice());

    suite.group("encode_q85_4k", |g| {
        g.bench("4:2:0 progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
                config
                    .encode_bytes(rgb_4k, 3840, 2160, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });

        g.bench("4:2:0 sharp progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).sharp_yuv(true);
                config
                    .encode_bytes(rgb_4k, 3840, 2160, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });
    });

    // XYB encode benchmarks at 1024×1024 — used to track XYB-vs-YCbCr gap
    // and measure the impact of XYB-specific perf work (streaming-through
    // gating, color conversion, parallel encoding).
    suite.group("encode_q85_1k_xyb", |g| {
        g.bench("ycbcr 4:2:0 progressive (baseline)", move |b| {
            b.iter(|| {
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
                config
                    .encode_bytes(rgb_1k_xyb, 1024, 1024, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });

        g.bench("xyb BQuarter progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::xyb(85.0, XybSubsampling::BQuarter);
                config
                    .encode_bytes(rgb_1k_xyb, 1024, 1024, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });

        g.bench("xyb Full progressive", move |b| {
            b.iter(|| {
                let config = EncoderConfig::xyb(85.0, XybSubsampling::Full);
                config
                    .encode_bytes(rgb_1k_xyb, 1024, 1024, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });

        g.bench("xyb BQuarter baseline (non-progressive)", move |b| {
            b.iter(|| {
                let config = EncoderConfig::xyb(85.0, XybSubsampling::BQuarter).progressive(false);
                config
                    .encode_bytes(rgb_1k_xyb, 1024, 1024, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });

        g.bench("xyb Full baseline (non-progressive)", move |b| {
            b.iter(|| {
                let config = EncoderConfig::xyb(85.0, XybSubsampling::Full).progressive(false);
                config
                    .encode_bytes(rgb_1k_xyb, 1024, 1024, PixelLayout::Rgb8Srgb)
                    .unwrap()
            })
        });
    });

    // Isolated sharp YUV benchmark.
    let rgb_1k: &'static [u8] = Box::leak(noise_patches(1024, 1024).into_boxed_slice());

    suite.group("sharp_yuv_isolated/1024", |g| {
        let (w, h) = (1024usize, 1024);
        let cw = w / 2;
        let ch = h / 2;

        g.bench("plain 4:2:0", move |b| {
            let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
            let mut y = vec![0u8; w * h];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            b.iter(|| ctx.encode_420_u8(rgb_1k, &mut y, &mut cb, &mut cr, w, h))
        });

        g.bench("sharp iter=4 (default)", move |b| {
            let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
            let mut y = vec![0u8; w * h];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            let config = zenyuv::SharpYuvConfig {
                max_iterations: 4,
                ..Default::default()
            };
            b.iter(|| ctx.encode_sharp_420_u8(rgb_1k, &mut y, &mut cb, &mut cr, w, h, &config))
        });

        g.bench("sharp iter=1", move |b| {
            let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
            let mut y = vec![0u8; w * h];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            let config = zenyuv::SharpYuvConfig {
                max_iterations: 1,
                ..Default::default()
            };
            b.iter(|| ctx.encode_sharp_420_u8(rgb_1k, &mut y, &mut cb, &mut cr, w, h, &config))
        });

        g.bench("sharp iter=0 (gamma-aware only)", move |b| {
            let mut ctx = zenyuv::YuvContext::new(zenyuv::Range::Full, zenyuv::Matrix::Bt601);
            let mut y = vec![0u8; w * h];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            let config = zenyuv::SharpYuvConfig {
                max_iterations: 0,
                ..Default::default()
            };
            b.iter(|| ctx.encode_sharp_420_u8(rgb_1k, &mut y, &mut cb, &mut cr, w, h, &config))
        });
    });
}

zenbench::main!(bench_encode);
