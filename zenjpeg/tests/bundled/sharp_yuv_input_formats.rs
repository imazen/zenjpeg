//! Regression for SharpYUV interpreting linear u16/f32 samples as RGB bytes.
//! All images are generated; no external corpus or native decoder is required.

use std::io::Cursor;
use zenjpeg::encoder::{
    ChromaSubsampling, DownsamplingMethod, EncoderConfig, PixelLayout, ProgressiveScanMode,
    Unstoppable,
};

fn linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn source(w: usize, h: usize) -> Vec<u8> {
    // Coloured patches with deterministic fine texture exercise channel order,
    // clipping, DCT coefficients, odd edges and the final partial strip.
    let mut out = Vec::new();
    let colours = [
        [45u8, 170, 220],
        [210, 65, 90],
        [125, 200, 40],
        [235, 225, 170],
    ];
    for y in 0..h {
        for x in 0..w {
            let p = colours[(x / 8 + y / 8) % colours.len()];
            let noise = ((x * 13 + y * 7) % 5) as i16 - 2;
            out.extend(p.map(|v| (v as i16 + noise) as u8));
        }
    }
    out
}

fn pixels(rgb: &[u8], layout: PixelLayout) -> Vec<u8> {
    let mut out = Vec::new();
    for p in rgb.chunks_exact(3) {
        match layout {
            PixelLayout::Rgb8Srgb => out.extend_from_slice(p),
            PixelLayout::Rgba8Srgb => out.extend([p[0], p[1], p[2], 17]),
            PixelLayout::Bgr8Srgb => out.extend([p[2], p[1], p[0]]),
            PixelLayout::Bgra8Srgb => out.extend([p[2], p[1], p[0], 17]),
            PixelLayout::Rgb16Linear | PixelLayout::Rgba16Linear => {
                for &v in p {
                    out.extend(((linear(v as f32 / 255.0) * 65535.0).round() as u16).to_ne_bytes());
                }
                if layout == PixelLayout::Rgba16Linear {
                    out.extend(12345u16.to_ne_bytes());
                }
            }
            PixelLayout::RgbF32Linear | PixelLayout::RgbaF32Linear => {
                for &v in p {
                    out.extend(linear(v as f32 / 255.0).to_ne_bytes());
                }
                if layout == PixelLayout::RgbaF32Linear {
                    out.extend(0.17f32.to_ne_bytes());
                }
            }
            _ => unreachable!(),
        }
    }
    out
}

fn encode(
    data: &[u8],
    w: usize,
    h: usize,
    layout: PixelLayout,
    chroma: ChromaSubsampling,
    strided: bool,
    method: DownsamplingMethod,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(99, chroma)
        .auto_optimize(true)
        .scan_mode(ProgressiveScanMode::ProgressiveSearch)
        .downsampling_method(method);
    let mut enc = config
        .encode_from_bytes(w as u32, h as u32, layout)
        .unwrap();
    if strided {
        let row_bytes = w * layout.bytes_per_pixel();
        let stride = row_bytes + 3 * layout.bytes_per_pixel();
        for start in (0..h).step_by(7) {
            let rows = (h - start).min(7);
            let mut padded = vec![0xff; rows * stride];
            for y in 0..rows {
                padded[y * stride..y * stride + row_bytes]
                    .copy_from_slice(&data[(start + y) * row_bytes..(start + y + 1) * row_bytes]);
            }
            enc.push(&padded, rows, stride, Unstoppable).unwrap();
        }
    } else {
        enc.push_packed(data, Unstoppable).unwrap();
    }
    enc.finish().unwrap()
}

fn compare_layout(layout: PixelLayout) {
    for method in [
        DownsamplingMethod::GammaAware,
        DownsamplingMethod::GammaAwareIterative,
    ] {
        compare_layout_with_method(layout, method);
    }
}

fn compare_layout_with_method(layout: PixelLayout, method: DownsamplingMethod) {
    for (w, h) in [(1, 1), (1, 35), (33, 1), (33, 35), (64, 67)] {
        let rgb = source(w, h);
        let data = pixels(&rgb, layout);
        for chroma in [
            ChromaSubsampling::None,
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::HalfVertical,
        ] {
            let reference = encode(&rgb, w, h, PixelLayout::Rgb8Srgb, chroma, false, method);
            let candidate = encode(&data, w, h, layout, chroma, false, method);
            assert_eq!(
                candidate,
                encode(&data, w, h, layout, chroma, true, method),
                "padding/chunking changed pixels: {layout:?} {chroma:?} {method:?} {w}x{h}"
            );
            let decode = |bytes: &[u8]| {
                jpeg_decoder::Decoder::new(Cursor::new(bytes))
                    .decode()
                    .unwrap()
            };
            let a = decode(&reference);
            let b = decode(&candidate);
            assert_eq!(a.len(), w * h * 3);
            assert_eq!(b.len(), a.len());
            let mae = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.abs_diff(y) as f64)
                .sum::<f64>()
                / a.len() as f64;
            let max = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.abs_diff(y))
                .max()
                .unwrap();
            assert!(
                mae < 0.8 && max <= 6,
                "{layout:?} {chroma:?} {method:?} {w}x{h}: MAE {mae}, max {max}"
            );
        }
    }
}

#[test]
fn sharp_yuv_rgb_f32() {
    compare_layout(PixelLayout::RgbF32Linear);
}
#[test]
fn sharp_yuv_rgba_f32() {
    compare_layout(PixelLayout::RgbaF32Linear);
}
#[test]
fn sharp_yuv_rgb_u16() {
    compare_layout(PixelLayout::Rgb16Linear);
}
#[test]
fn sharp_yuv_rgba_u16() {
    compare_layout(PixelLayout::Rgba16Linear);
}
#[test]
fn sharp_yuv_rgb_byte_layouts() {
    for layout in [
        PixelLayout::Rgb8Srgb,
        PixelLayout::Rgba8Srgb,
        PixelLayout::Bgr8Srgb,
        PixelLayout::Bgra8Srgb,
    ] {
        compare_layout(layout);
    }
}

#[test]
fn sharp_yuv_preserves_fractional_input() {
    for chroma in [
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfVertical,
    ] {
        let encode_gray = |v: f32| {
            let data: Vec<u8> = (0..16 * 16 * 3)
                .flat_map(|_| linear(v / 255.0).to_ne_bytes())
                .collect();
            encode(
                &data,
                16,
                16,
                PixelLayout::RgbF32Linear,
                chroma,
                false,
                DownsamplingMethod::GammaAwareIterative,
            )
        };
        // Both values become 127 if input is rounded to eight bits first.
        assert_ne!(encode_gray(127.05), encode_gray(127.45));
    }
}
