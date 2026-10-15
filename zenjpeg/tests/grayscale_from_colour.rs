//! `EncoderConfig::grayscale` encodes only luminance, whatever the input:
//! colour input is converted (its documented contract). It wrote a
//! three-component JPEG for every colour input layout (#235): the byte
//! encoder never told the streaming encoder to write one component.
//!
//! Oracle: the grayscale encode of colour input decodes to the grayscale
//! encode of the same image converted to BT.601 luma first, within the
//! rounding of that conversion (Y rounded to 8 bits, plus the linear
//! round trip for the 16-bit and f32 layouts).

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, PixelFormat};
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

const W: u32 = 37;
const H: u32 = 23;

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn rgb() -> Vec<[u8; 3]> {
    (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            let n = (i.wrapping_mul(2_654_435_761) >> 27) as u8;
            [
                (x * 6) as u8 ^ n,
                (y * 9) as u8 ^ (n >> 1),
                ((x + y) * 4) as u8,
            ]
        })
        .collect()
}

fn bytes(layout: PixelLayout, px: &[[u8; 3]]) -> Vec<u8> {
    let lin = |v: u8| srgb_to_linear(f32::from(v) / 255.0);
    let mut out = Vec::new();
    for &[r, g, b] in px {
        match layout {
            PixelLayout::Rgb8Srgb => out.extend_from_slice(&[r, g, b]),
            PixelLayout::Bgr8Srgb => out.extend_from_slice(&[b, g, r]),
            PixelLayout::Rgbx8Srgb | PixelLayout::Rgba8Srgb => {
                out.extend_from_slice(&[r, g, b, 255])
            }
            PixelLayout::Bgrx8Srgb | PixelLayout::Bgra8Srgb => {
                out.extend_from_slice(&[b, g, r, 255])
            }
            PixelLayout::Rgb16Linear | PixelLayout::Rgbx16Linear | PixelLayout::Rgba16Linear => {
                let n = if layout == PixelLayout::Rgb16Linear {
                    3
                } else {
                    4
                };
                for c in [r, g, b, 255].into_iter().take(n) {
                    let v = (lin(c) * 65535.0).round() as u16;
                    out.extend_from_slice(&v.to_ne_bytes());
                }
            }
            PixelLayout::RgbF32Linear | PixelLayout::RgbxF32Linear | PixelLayout::RgbaF32Linear => {
                let n = if layout == PixelLayout::RgbF32Linear {
                    3
                } else {
                    4
                };
                for c in [r, g, b, 255].into_iter().take(n) {
                    out.extend_from_slice(&lin(c).to_ne_bytes());
                }
            }
            _ => unreachable!(),
        }
    }
    out
}

fn encode(config: &EncoderConfig, layout: PixelLayout, data: &[u8]) -> Vec<u8> {
    let mut e = config.encode_from_bytes(W, H, layout).unwrap();
    e.push_packed(data, Unstoppable).unwrap();
    e.finish().unwrap()
}

fn components(jpeg: &[u8]) -> u8 {
    let sof = jpeg
        .windows(2)
        .position(|w| w[0] == 0xFF && matches!(w[1], 0xC0..=0xC2))
        .unwrap();
    jpeg[sof + 9]
}

fn gray(jpeg: &[u8]) -> Vec<u8> {
    Decoder::new()
        .output_format(PixelFormat::Gray)
        .decode(jpeg, Unstoppable)
        .unwrap()
        .pixels_u8()
        .unwrap()
        .to_vec()
}

#[test]
fn grayscale_config_writes_luma_for_colour_input() {
    let px = rgb();
    let luma: Vec<u8> = px
        .iter()
        .map(|&[r, g, b]| {
            (0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b)).round() as u8
        })
        .collect();
    for progressive in [false, true] {
        let config = EncoderConfig::grayscale(90).progressive(progressive);
        let want = gray(&encode(&config, PixelLayout::Gray8Srgb, &luma));
        for layout in [
            PixelLayout::Rgb8Srgb,
            PixelLayout::Bgr8Srgb,
            PixelLayout::Rgbx8Srgb,
            PixelLayout::Rgba8Srgb,
            PixelLayout::Bgrx8Srgb,
            PixelLayout::Bgra8Srgb,
            PixelLayout::Rgb16Linear,
            PixelLayout::Rgbx16Linear,
            PixelLayout::Rgba16Linear,
            PixelLayout::RgbF32Linear,
            PixelLayout::RgbxF32Linear,
            PixelLayout::RgbaF32Linear,
        ] {
            let jpeg = encode(&config, layout, &bytes(layout, &px));
            assert_eq!(components(&jpeg), 1, "{layout:?} progressive {progressive}");
            let got = gray(&jpeg);
            let max = got
                .iter()
                .zip(&want)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(
                max <= 3,
                "{layout:?} progressive {progressive}: max diff {max}"
            );
        }
    }
}
