//! Neutral RGB must remain neutral through subsampled partial strips.
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::types::PixelFormat;

fn neutral_rgb(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .flat_map(|i| {
            let x = i % width;
            let y = i / width;
            let v = ((x * 29 + y * 17 + 31) % 192 + 32) as u8;
            [v, v, v]
        })
        .collect()
}

fn check_neutral(width: usize, height: usize, sub: ChromaSubsampling) {
    let source = neutral_rgb(width, height);
    let encoded = EncoderConfig::ycbcr(40, sub)
        .encode_bytes(&source, width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    let decoded = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&encoded, enough::Unstoppable)
        .unwrap();
    let pixels = decoded.pixels_u8().unwrap();
    assert_eq!(pixels.len(), width * height * 3);
    for (i, p) in pixels.chunks_exact(3).enumerate() {
        assert!(
            p[0] == p[1] && p[1] == p[2],
            "{width}x{height} {sub:?} pixel ({},{}) gained chroma: {p:?}",
            i % width,
            i / width
        );
    }
}

#[test]
fn tiny_420_neutral_chroma_is_preserved() {
    check_neutral(32, 8, ChromaSubsampling::Quarter);
}

#[test]
fn neutral_chroma_at_all_partial_strip_edges() {
    for sub in [
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::HalfVertical,
        ChromaSubsampling::None,
    ] {
        for width in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 129] {
            for height in 1..=33 {
                check_neutral(width, height, sub);
            }
        }
    }
}

#[test]
fn pre_subsampled_strided_edges_match_full_resolution_planes() {
    use zenjpeg::encoder::YCbCrPlanes;
    for sub in [
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::HalfVertical,
        ChromaSubsampling::None,
    ] {
        for width in [1usize, 7, 9, 17, 31, 33] {
            for height in [1usize, 7, 8, 9, 15, 16, 17, 31, 33] {
                let cw = width.div_ceil(sub.h_factor() as usize);
                let ch = height.div_ceil(sub.v_factor() as usize);
                for extra in [0usize, 3] {
                    fn plane(w: usize, h: usize, extra: usize, value: f32) -> Vec<f32> {
                        let mut p = vec![1234.; (w + extra) * h];
                        for row in p.chunks_exact_mut(w + extra) {
                            row[..w].fill(value);
                        }
                        p
                    }
                    let y = plane(width, height, extra, 0.);
                    let cb = plane(width, height, extra, 20.);
                    let cr = plane(width, height, extra, -10.);
                    let scb = plane(cw, ch, extra, 20.);
                    let scr = plane(cw, ch, extra, -10.);
                    let full = YCbCrPlanes {
                        y: &y,
                        y_stride: width + extra,
                        cb: &cb,
                        cb_stride: width + extra,
                        cr: &cr,
                        cr_stride: width + extra,
                    };
                    let sampled = YCbCrPlanes {
                        y: &y,
                        y_stride: width + extra,
                        cb: &scb,
                        cb_stride: cw + extra,
                        cr: &scr,
                        cr_stride: cw + extra,
                    };
                    let config = EncoderConfig::ycbcr(40, sub);
                    let mut a = config
                        .encode_from_ycbcr_planar(width as u32, height as u32)
                        .unwrap();
                    let mut b = config
                        .encode_from_ycbcr_planar(width as u32, height as u32)
                        .unwrap();
                    a.push(&full, height, enough::Unstoppable).unwrap();
                    b.push_subsampled(&sampled, height, enough::Unstoppable)
                        .unwrap();
                    let a = Decoder::new()
                        .output_format(PixelFormat::Rgb)
                        .decode(&a.finish().unwrap(), enough::Unstoppable)
                        .unwrap();
                    let b = Decoder::new()
                        .output_format(PixelFormat::Rgb)
                        .decode(&b.finish().unwrap(), enough::Unstoppable)
                        .unwrap();
                    assert_eq!(
                        a.pixels_u8().unwrap(),
                        b.pixels_u8().unwrap(),
                        "{width}x{height} {sub:?} stride padding {extra}"
                    );
                }
            }
        }
    }
}

#[test]
fn gamma_aware_partial_strips_remain_neutral() {
    use zenjpeg::encoder::DownsamplingMethod;
    for method in [
        DownsamplingMethod::GammaAware,
        DownsamplingMethod::GammaAwareIterative,
    ] {
        for sub in [
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::HalfVertical,
        ] {
            for (w, h) in [
                (1, 1),
                (7, 3),
                (9, 7),
                (17, 9),
                (31, 15),
                (33, 17),
                (65, 33),
            ] {
                let source = neutral_rgb(w, h);
                let encoded = EncoderConfig::ycbcr(40, sub)
                    .downsampling_method(method)
                    .encode_bytes(&source, w as u32, h as u32, PixelLayout::Rgb8Srgb)
                    .unwrap();
                let decoded = Decoder::new()
                    .output_format(PixelFormat::Rgb)
                    .decode(&encoded, enough::Unstoppable)
                    .unwrap();
                for (i, p) in decoded.pixels_u8().unwrap().chunks_exact(3).enumerate() {
                    assert!(
                        p[0] == p[1] && p[1] == p[2],
                        "{w}x{h} {sub:?} {method:?} pixel {i}: {p:?}"
                    );
                }
            }
        }
    }
}
