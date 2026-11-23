//! Fixed, committed decode fixtures and explicit linear/sRGB encode witnesses.
#![forbid(unsafe_code)]
use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

pub fn decode_cases() -> Vec<(&'static str, &'static [u8])> {
    vec![
        (
            "photo420",
            include_bytes!("../../../internal/native-bench/evidence/flower-420-base.jpg"),
        ),
        (
            "photo444",
            include_bytes!("../../../internal/native-bench/evidence/flower-444-base.jpg"),
        ),
        (
            "photo_progressive",
            include_bytes!("../../../internal/native-bench/evidence/flower-420-prog.jpg"),
        ),
        (
            "noise420",
            include_bytes!("../../../internal/native-bench/evidence/noise-420-base-q85-256.jpg"),
        ),
        (
            "noise_progressive",
            include_bytes!("../../../internal/native-bench/evidence/noise-420-prog-q85-256.jpg"),
        ),
        (
            "dc_odd",
            include_bytes!("../../../internal/native-bench/evidence/dc-17x33-q85-420-base.jpg"),
        ),
    ]
}

pub struct EncodeCase {
    // Labels are used by native file/benchmark output; the unknown target only retains code.
    #[cfg_attr(target_os = "unknown", allow(dead_code))]
    pub name: &'static str,
    pub input: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub config: EncoderConfig,
}

impl EncodeCase {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = self
            .config
            .encode_from_bytes(self.width, self.height, self.layout)
            .unwrap();
        encoder.push_packed(&self.input, Unstoppable).unwrap();
        encoder.finish().unwrap()
    }
}

pub fn encode_cases() -> Vec<EncodeCase> {
    let mut oracle = jpeg_decoder::Decoder::new(std::io::Cursor::new(decode_cases()[0].1));
    let photo = oracle.decode().unwrap();
    let info = oracle.info().unwrap();
    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
    let width = u32::from(info.width).min(512);
    let height = u32::from(info.height).min(384);
    let rgb: Vec<u8> = (0..height as usize)
        .flat_map(|y| {
            photo[y * info.width as usize * 3..][..width as usize * 3]
                .iter()
                .copied()
        })
        .collect();
    let mut cases = Vec::new();
    for (name, sampling) in [
        ("srgb8_fixed420", ChromaSubsampling::Quarter),
        ("srgb8_fixed444", ChromaSubsampling::None),
    ] {
        cases.push(EncodeCase {
            name,
            input: rgb.clone(),
            width,
            height,
            layout: PixelLayout::Rgb8Srgb,
            config: EncoderConfig::ycbcr(85.0, sampling)
                .progressive(false)
                .optimize_huffman(false),
        });
    }
    cases.push(EncodeCase {
        name: "srgb8_fixed_xyb",
        input: rgb,
        width,
        height,
        layout: PixelLayout::Rgb8Srgb,
        config: EncoderConfig::xyb(85.0, XybSubsampling::Full)
            .progressive(false)
            .optimize_huffman(false),
    });
    for (name, layout, channels, float) in [
        ("linear_rgb16", PixelLayout::Rgb16Linear, 3, false),
        ("linear_gray16", PixelLayout::Gray16Linear, 1, false),
        ("linear_rgbf32", PixelLayout::RgbF32Linear, 3, true),
        ("linear_grayf32", PixelLayout::GrayF32Linear, 1, true),
    ] {
        let (width, height) = (257, 193);
        let mut input = Vec::new();
        let mut state = 19u32;
        for i in 0..width * height * channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = if i % 17 == 0 { 0 } else { (state >> 16) as u16 };
            if float {
                input.extend_from_slice(&(f32::from(value) / 32767.5 - 0.1).to_ne_bytes());
            } else {
                input.extend_from_slice(&value.to_ne_bytes());
            }
        }
        let config = if channels == 1 {
            EncoderConfig::grayscale(85.0)
        } else {
            EncoderConfig::ycbcr(85.0, ChromaSubsampling::None)
        };
        cases.push(EncodeCase {
            name,
            input,
            width,
            height,
            layout,
            config: config.progressive(false).optimize_huffman(false),
        });
    }
    cases
}

pub fn decode(bytes: &[u8]) -> Vec<u8> {
    Decoder::new()
        .decode(bytes, Unstoppable)
        .unwrap()
        .pixels_u8()
        .unwrap()
        .to_vec()
}
