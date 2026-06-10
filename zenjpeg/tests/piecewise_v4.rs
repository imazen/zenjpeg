//! PiecewiseV4 table family: encode smoke + quality progression.
//!
//! The anchors were SA-trained on CID22-512 photos (GPU butteraugli,
//! jpegli encoder, 4:2:0). These tests cover wiring, not pareto claims —
//! re-run coefficient's sweep harness for quality numbers.

use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, QuantTableConfig};

fn photo_ish_rgb(w: u32, h: u32) -> Vec<u8> {
    // Deterministic noise + low-frequency ramps; NOT a smooth gradient
    // (banned: degenerate DCT statistics).
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    let mut state = 0x2545F4914F6CDD1Du64;
    for y in 0..h {
        for x in 0..w {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let n = (state >> 32) as u32;
            let r = ((x * 3 + y) % 256) as u8 ^ (n & 0x1F) as u8;
            let g = ((x + y * 2) % 256) as u8 ^ ((n >> 5) & 0x1F) as u8;
            let b = ((x * 2 + y * 3) % 256) as u8 ^ ((n >> 10) & 0x1F) as u8;
            v.extend_from_slice(&[r, g, b]);
        }
    }
    v
}

fn encode(config: &EncoderConfig, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

#[test]
fn piecewise_v4_encodes_valid_decodable_jpegs() {
    let (w, h) = (160, 120);
    let rgb = photo_ish_rgb(w, h);

    for q in [10.0f32, 50.0, 90.0] {
        let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
            .quant_table_config(QuantTableConfig::PiecewiseV4);
        let jpeg = encode(&cfg, &rgb, w, h);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "SOI at q={q}");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI at q={q}");

        let decoder = zenjpeg::decoder::Decoder::new();
        let decoded = decoder.decode(&jpeg, Unstoppable).expect("decodable");
        assert_eq!(decoded.width(), w);
        assert_eq!(decoded.height(), h);
    }
}

#[test]
fn piecewise_v4_size_tracks_quality() {
    let (w, h) = (160, 120);
    let rgb = photo_ish_rgb(w, h);
    let size = |q: f32| {
        encode(
            &EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter)
                .quant_table_config(QuantTableConfig::PiecewiseV4),
            &rgb,
            w,
            h,
        )
        .len()
    };
    let (s10, s50, s90) = (size(10.0), size(50.0), size(90.0));
    assert!(s10 < s50, "q10 ({s10}) should be smaller than q50 ({s50})");
    assert!(s50 < s90, "q50 ({s50}) should be smaller than q90 ({s90})");
}

#[test]
fn piecewise_v4_differs_from_jpegli_defaults() {
    let (w, h) = (160, 120);
    let rgb = photo_ish_rgb(w, h);
    let cfg_pw = EncoderConfig::ycbcr(50.0, ChromaSubsampling::Quarter)
        .quant_table_config(QuantTableConfig::PiecewiseV4);
    let cfg_jp = EncoderConfig::ycbcr(50.0, ChromaSubsampling::Quarter);
    assert_ne!(
        encode(&cfg_pw, &rgb, w, h),
        encode(&cfg_jp, &rgb, w, h),
        "PiecewiseV4 must actually select different tables"
    );
}
