//! RGB passthrough mode tests (issue #185).
//!
//! `EncoderConfig::rgb()` stores channels without color transformation:
//! JPEG components R, G, B at 4:4:4, signaled via component IDs
//! 'R','G','B' and an Adobe APP14 marker with transform=0. The tests
//! verify channel purity (no cross-channel bleed), marker structure,
//! roundtrip accuracy through zenjpeg's own decoder, and interop with
//! third-party decoders (jpeg-decoder, zune-jpeg).

use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};
use zenjpeg::types::PixelFormat;

fn solid_rgb(width: usize, height: usize, rgb: [u8; 3]) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        data.extend_from_slice(&rgb);
    }
    data
}

/// Deterministic noise+patches image (gradients are banned — degenerate DCT).
fn noise_image(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    let mut seed: u32 = 0xC0FF_EE00;
    for y in 0..height {
        for x in 0..width {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let r = ((seed >> 16) & 0xFF) as u8;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let g = ((seed >> 16) & 0xFF) as u8;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let b = ((seed >> 16) & 0xFF) as u8;
            let idx = (y * width + x) * 3;
            // Patches of near-solid color mixed with full noise, so blocks
            // have both low- and high-frequency content.
            if (x / 16 + y / 16) % 3 == 0 {
                rgb[idx] = r;
                rgb[idx + 1] = g;
                rgb[idx + 2] = b;
            } else {
                rgb[idx] = 180u8.wrapping_add(r / 8);
                rgb[idx + 1] = 30u8.wrapping_add(g / 8);
                rgb[idx + 2] = 90u8.wrapping_add(b / 8);
            }
        }
    }
    rgb
}

fn decode_zen(encoded: &[u8]) -> (Vec<u8>, u32, u32) {
    let decoded = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(encoded, enough::Unstoppable)
        .expect("decode failed");
    let (w, h) = (decoded.width(), decoded.height());
    (decoded.pixels_u8().expect("u8 pixels").to_vec(), w, h)
}

/// Solid single-channel images must come back with the other channels
/// (nearly) untouched — the whole point of skipping the color transform.
/// A YCbCr roundtrip of pure red bleeds into G/B by design; passthrough
/// must not.
#[test]
fn rgb_solid_channel_purity() {
    let (w, h) = (64usize, 48usize);
    for (name, src, on_idx) in [
        ("red", solid_rgb(w, h, [200, 0, 0]), 0usize),
        ("green", solid_rgb(w, h, [0, 200, 0]), 1),
        ("blue", solid_rgb(w, h, [0, 0, 200]), 2),
    ] {
        for progressive in [false, true] {
            let config = EncoderConfig::rgb(90).progressive(progressive);
            let encoded = config
                .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
                .unwrap_or_else(|e| panic!("{name} prog={progressive} encode failed: {e}"));
            let (pixels, dw, dh) = decode_zen(&encoded);
            assert_eq!((dw, dh), (w as u32, h as u32));

            for (i, px) in pixels.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    let expected = if c == on_idx { 200i32 } else { 0 };
                    let diff = (px[c] as i32 - expected).abs();
                    assert!(
                        diff <= 2,
                        "{name} prog={progressive}: pixel {i} channel {c} = {} (expected ~{expected}); cross-channel bleed?",
                        px[c]
                    );
                }
            }
        }
    }
}

/// Locate an APP14 Adobe segment and return its transform byte.
fn find_adobe_transform(bytes: &[u8]) -> Option<u8> {
    let mut i = 2; // skip SOI
    while i + 4 <= bytes.len() {
        if bytes[i] != 0xFF {
            return None; // lost sync — headers only
        }
        let marker = bytes[i + 1];
        if marker == 0xDA {
            return None; // SOS reached
        }
        let len = ((bytes[i + 2] as usize) << 8) | bytes[i + 3] as usize;
        if marker == 0xEE && bytes[i + 4..].starts_with(b"Adobe") {
            return Some(bytes[i + 2 + len - 1]);
        }
        i += 2 + len;
    }
    None
}

/// Collect (marker, segment_payload) pairs up to SOS.
fn header_segments(bytes: &[u8]) -> Vec<(u8, Vec<u8>)> {
    let mut segs = Vec::new();
    let mut i = 2;
    while i + 4 <= bytes.len() {
        if bytes[i] != 0xFF {
            break;
        }
        let marker = bytes[i + 1];
        if marker == 0xDA {
            break;
        }
        let len = ((bytes[i + 2] as usize) << 8) | bytes[i + 3] as usize;
        segs.push((marker, bytes[i + 4..i + 2 + len].to_vec()));
        i += 2 + len;
    }
    segs
}

/// Structural checks: Adobe APP14 transform=0, no JFIF, component IDs
/// 'R','G','B' at 1×1 sampling all referencing quant table 0, exactly one
/// 8-bit DQT table, SOF0 for baseline / SOF2 for progressive.
#[test]
fn rgb_marker_structure() {
    let (w, h) = (64usize, 64usize);
    let src = noise_image(w, h);

    for (progressive, want_sof) in [(false, 0xC0u8), (true, 0xC2u8)] {
        let encoded = EncoderConfig::rgb(90)
            .progressive(progressive)
            .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("encode failed");

        assert_eq!(
            find_adobe_transform(&encoded),
            Some(0),
            "prog={progressive}: missing Adobe APP14 transform=0"
        );

        let segs = header_segments(&encoded);
        assert!(
            !segs
                .iter()
                .any(|(m, p)| *m == 0xE0 && p.starts_with(b"JFIF")),
            "prog={progressive}: JFIF APP0 must not be written for RGB"
        );

        let dqts: Vec<&Vec<u8>> = segs
            .iter()
            .filter(|(m, _)| *m == 0xDB)
            .map(|(_, p)| p)
            .collect();
        assert_eq!(
            dqts.len(),
            1,
            "prog={progressive}: expected one DQT segment"
        );
        assert_eq!(
            dqts[0].len(),
            65,
            "prog={progressive}: expected a single 8-bit quant table"
        );
        assert_eq!(
            dqts[0][0], 0x00,
            "prog={progressive}: table 0, 8-bit precision"
        );

        let sof = segs
            .iter()
            .find(|(m, _)| *m == want_sof)
            .unwrap_or_else(|| panic!("prog={progressive}: SOF{:02X} not found", want_sof));
        let p = &sof.1;
        assert_eq!(p[5], 3, "3 components");
        for (comp, id) in [(0usize, b'R'), (1, b'G'), (2, b'B')] {
            let off = 6 + comp * 3;
            assert_eq!(p[off], id, "component {comp} ID");
            assert_eq!(p[off + 1], 0x11, "component {comp} sampling 1x1");
            assert_eq!(p[off + 2], 0, "component {comp} quant table 0");
        }
    }
}

/// Noise roundtrip accuracy on non-MCU-aligned dimensions (exercises the
/// padded-stride conversion, right-edge replication, and bottom-strip
/// vertical padding).
#[test]
fn rgb_noise_roundtrip_accuracy() {
    let (w, h) = (130usize, 67usize);
    let src = noise_image(w, h);

    for progressive in [false, true] {
        let encoded = EncoderConfig::rgb(95)
            .progressive(progressive)
            .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .expect("encode failed");
        let (pixels, dw, dh) = decode_zen(&encoded);
        assert_eq!((dw, dh), (w as u32, h as u32));
        assert_eq!(pixels.len(), src.len());

        let mut sum_abs = [0u64; 3];
        let mut max_abs = [0u32; 3];
        for (s, d) in src.chunks_exact(3).zip(pixels.chunks_exact(3)) {
            for c in 0..3 {
                let diff = (s[c] as i32 - d[c] as i32).unsigned_abs();
                sum_abs[c] += diff as u64;
                max_abs[c] = max_abs[c].max(diff);
            }
        }
        let n = (w * h) as u64;
        for c in 0..3 {
            let mean = sum_abs[c] as f64 / n as f64;
            assert!(
                mean < 6.0,
                "prog={progressive}: channel {c} mean abs diff {mean:.2} too high"
            );
            assert!(
                max_abs[c] < 64,
                "prog={progressive}: channel {c} max abs diff {} too high",
                max_abs[c]
            );
        }
    }

    // Lower qualities must still roundtrip (16-bit DQT / SOF1 territory
    // when allow_16bit is on).
    for q in [5, 25, 50, 75] {
        let encoded = EncoderConfig::rgb(q)
            .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap_or_else(|e| panic!("q{q} encode failed: {e}"));
        let (pixels, ..) = decode_zen(&encoded);
        assert_eq!(pixels.len(), src.len(), "q{q} roundtrip size");
    }
}

/// BGRA input must land in the right components (catches swizzle bugs),
/// and the alpha byte must be ignored.
#[test]
fn rgb_bgra_input_swizzle() {
    let (w, h) = (32usize, 32usize);
    // BGRA bytes for the color (r=60, g=20, b=180)
    let mut bgra = Vec::with_capacity(w * h * 4);
    for _ in 0..w * h {
        bgra.extend_from_slice(&[180, 20, 60, 255]);
    }
    let encoded = EncoderConfig::rgb(90)
        .encode_bytes(&bgra, w as u32, h as u32, PixelLayout::Bgra8Srgb)
        .expect("BGRA encode failed");
    let (pixels, ..) = decode_zen(&encoded);
    let px = &pixels[..3];
    assert!(
        (px[0] as i32 - 60).abs() <= 2
            && (px[1] as i32 - 20).abs() <= 2
            && (px[2] as i32 - 180).abs() <= 2,
        "BGRA swizzle wrong: got {px:?}, expected ~[60, 20, 180]"
    );
}

/// Third-party decoders must see an RGB (no transform) JPEG and return
/// unmixed channels.
#[test]
fn rgb_interop_third_party_decoders() {
    let (w, h) = (48usize, 40usize);
    let src = solid_rgb(w, h, [0, 200, 0]);
    let encoded = EncoderConfig::rgb(90)
        .progressive(false)
        .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode failed");

    // jpeg-decoder
    {
        let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(&encoded));
        let pixels = dec.decode().expect("jpeg-decoder failed");
        let info = dec.info().unwrap();
        assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
        for (i, px) in pixels.chunks_exact(3).enumerate() {
            assert!(
                px[0] <= 2 && (px[1] as i32 - 200).abs() <= 2 && px[2] <= 2,
                "jpeg-decoder pixel {i}: {px:?} not pure green"
            );
        }
    }

    // zune-jpeg
    {
        let mut dec =
            zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&encoded));
        let pixels = dec.decode().expect("zune-jpeg failed");
        assert_eq!(pixels.len(), w * h * 3);
        for (i, px) in pixels.chunks_exact(3).enumerate() {
            assert!(
                px[0] <= 2 && (px[1] as i32 - 200).abs() <= 2 && px[2] <= 2,
                "zune-jpeg pixel {i}: {px:?} not pure green"
            );
        }
    }
}

/// Non-RGB input layouts are rejected with a clear error instead of
/// silently converting.
#[test]
fn rgb_rejects_non_rgb_input() {
    let (w, h) = (16usize, 16usize);
    let gray = vec![128u8; w * h];
    let err = EncoderConfig::rgb(90)
        .encode_bytes(&gray, w as u32, h as u32, PixelLayout::Gray8Srgb)
        .expect_err("grayscale input must be rejected in RGB mode");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("rgb"),
        "error should mention RGB mode: {msg}"
    );
}

/// AQ off must also roundtrip (flat zero-bias path without AQ strengths).
#[test]
fn rgb_aq_disabled_roundtrip() {
    let (w, h) = (64usize, 64usize);
    let src = noise_image(w, h);
    let encoded = EncoderConfig::rgb(85)
        .aq_enabled(false)
        .encode_bytes(&src, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .expect("encode failed");
    let (pixels, ..) = decode_zen(&encoded);
    assert_eq!(pixels.len(), src.len());
}
