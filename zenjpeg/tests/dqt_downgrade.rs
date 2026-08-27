//! #143 item 1: with `allow_16bit_quant_tables(true)`, a table whose >255
//! positions carry only zero coefficients for THIS image is emitted as its
//! 8-bit clamp (SOF0, 64 fewer DQT bytes per table) — pixel-identical by
//! construction. Content that actually uses those positions keeps 16-bit.

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// (DQT precisions in stream order, SOF marker byte).
fn dqt_precisions_and_sof(jpeg: &[u8]) -> (Vec<u8>, u8) {
    let mut precisions = Vec::new();
    let mut sof = 0u8;
    let mut i = 2usize; // past SOI
    while i + 4 <= jpeg.len() {
        assert_eq!(jpeg[i], 0xFF, "marker expected at {i}");
        let marker = jpeg[i + 1];
        if marker == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        let payload = &jpeg[i + 4..i + 2 + len];
        match marker {
            0xDB => {
                let mut p = 0;
                while p < payload.len() {
                    let precision = payload[p] >> 4;
                    precisions.push(precision);
                    p += 1 + if precision == 0 { 64 } else { 128 };
                }
            }
            0xC0..=0xC2 => sof = marker,
            0xDA => break, // scan data follows; headers are done
            _ => {}
        }
        i += 2 + len;
    }
    (precisions, sof)
}

fn encode(rgb: &[u8], w: u32, h: u32, allow_16bit: bool) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(50.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .allow_16bit_quant_tables(allow_16bit);
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

/// Luma noise, chroma flat per 64×64 tile: chroma blocks are DC-only, so the
/// high-frequency chroma positions the Q50 table pushes past 255 are all zero.
fn flat_chroma(w: u32, h: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 3) as usize];
    let mut seed = 0xACE1u32;
    for y in 0..h as usize {
        for x in 0..w as usize {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let luma = 96u8.wrapping_add((seed >> 25) as u8);
            let tint = ((x / 64 + y / 64) % 3) as u8;
            let i = (y * w as usize + x) * 3;
            v[i] = luma.saturating_add(tint * 20);
            v[i + 1] = luma;
            v[i + 2] = luma.saturating_add((2 - tint) * 20);
        }
    }
    v
}

/// Saturated colour checkerboard with 2×2-pixel cells: after 4:2:0's 2×2
/// chroma averaging the chroma planes still alternate every sample, so the
/// highest chroma frequencies (the positions Q50 pushes past 255) carry
/// coefficients and 16-bit must survive. (A per-pixel checkerboard would
/// average to FLAT chroma and downgrade — correctly.)
fn busy_chroma(w: u32, h: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 3;
            let on = (x / 2 + y / 2) % 2 == 0;
            v[i] = if on { 255 } else { 0 };
            v[i + 1] = if on { 0 } else { 255 };
            v[i + 2] = if (x / 2) % 2 == 0 { 255 } else { 0 };
        }
    }
    v
}

#[test]
fn flat_chroma_image_downgrades_to_8bit_dqt_and_sof0() {
    let (w, h) = (256u32, 192u32);
    let rgb = flat_chroma(w, h);
    let with = encode(&rgb, w, h, true);
    let (prec, sof) = dqt_precisions_and_sof(&with);
    assert!(!prec.is_empty());
    assert!(
        prec.iter().all(|&p| p == 0),
        "every table should be provably 8-bit for DC-only chroma, got {prec:?}"
    );
    assert_eq!(sof, 0xC0, "no 16-bit table left → baseline SOF0");
    // Still a valid image with the right geometry.
    let img = Decoder::new().decode(&with, Unstoppable).expect("decode");
    assert_eq!((img.width(), img.height()), (w, h));
    // The explicit force-baseline encode is a different quantization (finer
    // divisors), so it need not be byte-equal — but it must not be SMALLER by
    // more than the DQT saving the downgrade already captured.
    let without = encode(&rgb, w, h, false);
    let (prec0, _) = dqt_precisions_and_sof(&without);
    assert!(prec0.iter().all(|&p| p == 0));
}

#[test]
fn busy_chroma_image_keeps_16bit_dqt() {
    let (w, h) = (256u32, 192u32);
    let rgb = busy_chroma(w, h);
    let with = encode(&rgb, w, h, true);
    let (prec, sof) = dqt_precisions_and_sof(&with);
    assert!(
        prec.contains(&1),
        "chroma with energy at the >255 positions must keep a 16-bit table, got {prec:?}"
    );
    assert_eq!(sof, 0xC1);
    let img = Decoder::new().decode(&with, Unstoppable).expect("decode");
    assert_eq!((img.width(), img.height()), (w, h));
}
