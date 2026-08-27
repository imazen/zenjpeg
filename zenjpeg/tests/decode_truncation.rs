//! #92: the "decode a growing prefix" contract, checked at EVERY byte offset
//! of small baseline / progressive / restart-interval / grayscale fixtures.
//!
//! For each prefix `data[..n]` under the default (`Balanced`) strictness:
//! - decoding never panics (the whole test is the assertion);
//! - an `Ok` carries the header dimensions;
//! - **monotone acceptance**: once some prefix decodes, every longer prefix
//!   decodes too — a consumer re-decoding on each network chunk must never
//!   see a partial image turn back into an error;
//! - **differential**: if `Strict` accepts the prefix, the `Balanced` pixmap
//!   is identical to Strict's;
//! - a prefix that decodes but is shorter than the last scan's end reports
//!   at least one `Truncated*` warning (the signal a streaming consumer keys
//!   off), and the complete stream reports none of them.

use enough::Unstoppable;
use zenjpeg::decoder::{DecodeWarning, Decoder, Strictness};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn noise_rgb(w: u32, h: u32, seed0: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 3) as usize];
    let mut seed = seed0;
    for (i, b) in v.iter_mut().enumerate() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        // Blocky low-frequency structure + noise so every scan carries data.
        let x = (i / 3) % w as usize;
        let y = (i / 3) / w as usize;
        *b = (((x / 8 + y / 8) * 37) as u8).wrapping_add((seed >> 27) as u8);
    }
    v
}

fn fixture(name: &str, w: u32, h: u32, cfg: EncoderConfig, gray: bool) -> (String, Vec<u8>) {
    let layout = if gray {
        PixelLayout::Gray8Srgb
    } else {
        PixelLayout::Rgb8Srgb
    };
    let rgb = noise_rgb(w, h, 0xC0FFEE);
    let px: Vec<u8> = if gray {
        rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect()
    } else {
        rgb
    };
    let mut enc = cfg.encode_from_bytes(w, h, layout).expect("encoder");
    enc.push_packed(&px, Unstoppable).expect("push");
    (name.to_string(), enc.finish().expect("finish"))
}

fn fixtures() -> Vec<(String, Vec<u8>)> {
    let (w, h) = (72u32, 56u32); // non-MCU-aligned on both axes
    vec![
        fixture(
            "baseline-420",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter).progressive(false),
            false,
        ),
        fixture(
            "baseline-444-dri",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::None)
                .progressive(false)
                .restart_mcu_rows(1),
            false,
        ),
        fixture(
            "progressive-420",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter).progressive(true),
            false,
        ),
        fixture(
            "progressive-420-dri",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::Quarter)
                .progressive(true)
                .restart_mcu_rows(1),
            false,
        ),
        fixture(
            "gray-baseline",
            w,
            h,
            EncoderConfig::ycbcr(80.0, ChromaSubsampling::None).progressive(false),
            true,
        ),
    ]
}

fn is_truncation_warning(w: &DecodeWarning) -> bool {
    matches!(
        w,
        DecodeWarning::TruncatedScan { .. }
            | DecodeWarning::TruncatedBetweenScans { .. }
            | DecodeWarning::TruncatedProgressiveScan
    )
}

/// Offset one past the last entropy-coded byte: everything from here on is
/// trailer (the EOI marker), so a prefix at or beyond it lost no image data.
fn end_of_last_scan(jpeg: &[u8]) -> usize {
    assert_eq!(
        &jpeg[jpeg.len() - 2..],
        &[0xFF, 0xD9],
        "fixture must end in EOI"
    );
    jpeg.len() - 2
}

#[test]
fn every_prefix_decodes_consistently() {
    for (name, jpeg) in fixtures() {
        let full = Decoder::new()
            .decode(&jpeg, Unstoppable)
            .expect("full decode");
        let (fw, fh) = (full.width(), full.height());
        assert!(
            !full.warnings().iter().any(is_truncation_warning),
            "{name}: complete stream reports a truncation warning: {:?}",
            full.warnings()
        );
        let scan_end = end_of_last_scan(&jpeg);

        let mut first_ok: Option<usize> = None;
        let mut strict_ok_count = 0usize;
        for n in 0..=jpeg.len() {
            let prefix = &jpeg[..n];
            let balanced = Decoder::new().decode(prefix, Unstoppable);
            let strict = Decoder::new()
                .strictness(Strictness::Strict)
                .decode(prefix, Unstoppable);

            match &balanced {
                Ok(img) => {
                    assert_eq!((img.width(), img.height()), (fw, fh), "{name}@{n}: dims");
                    if first_ok.is_none() {
                        first_ok = Some(n);
                    }
                    if n < scan_end {
                        assert!(
                            img.warnings().iter().any(is_truncation_warning),
                            "{name}@{n}: prefix lost scan data but reports no Truncated* warning: {:?}",
                            img.warnings()
                        );
                    }
                }
                Err(e) => {
                    if let Some(f) = first_ok {
                        panic!("{name}: prefix {f} decoded but longer prefix {n} errored: {e}");
                    }
                }
            }
            if let Ok(s) = &strict {
                strict_ok_count += 1;
                let b = balanced
                    .as_ref()
                    .unwrap_or_else(|e| panic!("{name}@{n}: Strict ok but Balanced err: {e}"));
                assert_eq!(
                    s.pixels_u8(),
                    b.pixels_u8(),
                    "{name}@{n}: Strict/Balanced pixels"
                );
            }
        }
        let f = first_ok.unwrap_or_else(|| panic!("{name}: no prefix ever decoded"));
        assert!(f < jpeg.len(), "{name}: only the complete stream decodes");
        assert!(
            strict_ok_count >= 1,
            "{name}: Strict never accepted even the full stream"
        );
        eprintln!(
            "{name}: {} bytes, first decodable prefix {f} ({:.0}%), strict-ok prefixes {strict_ok_count}",
            jpeg.len(),
            100.0 * f as f64 / jpeg.len() as f64
        );
    }
}
