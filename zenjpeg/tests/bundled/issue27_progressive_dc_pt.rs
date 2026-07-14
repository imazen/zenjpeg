//! Issue #27: Progressive JPEG DC point transform produces wrong pixel values.
//!
//! When a progressive DC-only scan has point transform Al > 0 and there is no
//! subsequent refining scan, the decoded coefficients must be left-shifted by Al
//! before dequantization and IDCT.
//!
//! Run: cargo test --release -p zenjpeg --test issue27_progressive_dc_pt -- --nocapture

use enough::Unstoppable;
use zenjpeg::decoder::{Decoder, PixelFormat};

/// Minimal progressive JPEG: 8x8 grayscale, DC-only scan with Al=1.
///
/// JPEG structure:
/// - SOF2 (progressive): 8x8 grayscale, 1 component, sampling 1x1
/// - DQT: Q[0] = 3
/// - DHT: DC table with single 1-bit code `0` -> category 8
/// - SOS: Ss=0, Se=0, Ah=0, Al=1 (DC-only scan, point transform shift=1)
/// - Entropy data: `0x54 0xFF 0x00` (2 decoded bytes after byte-stuffing)
/// - Only one scan, no AC coefficients, no refining DC scan, then EOI
///
/// Correct decode (libjpeg-turbo verified):
/// 1. Huffman: code `0` -> category 8, extra bits `10101001` = 169 -> DC diff = +169
/// 2. Point transform: coefficient = 169 << 1 = 338
/// 3. Dequantize: 338 * Q[0]=3 = 1014
/// 4. IDCT DC-only: pixel = 1014/8 + 128 = 254.75 -> clamp to 255
const CRASH_JPEG: &[u8] = &[
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48,
    0x00, 0x48, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43, 0x00, 0x03, 0x02, 0x02, 0x03, 0x02, 0x02, 0x03,
    0x03, 0x03, 0x03, 0x04, 0x03, 0x03, 0x04, 0x05, 0x08, 0x05, 0x05, 0x04, 0x04, 0x05, 0x0a, 0x07,
    0x07, 0x06, 0x08, 0x0c, 0x0a, 0x0c, 0x0c, 0x0b, 0x0a, 0x0b, 0x0b, 0x0d, 0x0e, 0x12, 0x10, 0x0d,
    0x0e, 0x11, 0x0e, 0x0b, 0x0b, 0x10, 0x16, 0x10, 0x11, 0x13, 0x14, 0x15, 0x15, 0x15, 0x0c, 0x0f,
    0x17, 0x18, 0x16, 0x14, 0x18, 0x12, 0x14, 0x15, 0x14, 0xff, 0xc2, 0x00, 0x0b, 0x08, 0x00, 0x08,
    0x00, 0x08, 0x01, 0x01, 0x11, 0x00, 0xff, 0xc4, 0x00, 0x14, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0xff, 0xda, 0x00, 0x08,
    0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x54, 0xff, 0x00, 0xff, 0xd9,
];

#[test]
fn test_issue27_progressive_dc_point_transform_al1() {
    // Match the fuzz target configuration that found the bug
    let decoder = Decoder::new().output_format(PixelFormat::Rgb);
    let result = decoder
        .decode(CRASH_JPEG, Unstoppable)
        .expect("decode failed");

    assert_eq!(result.width, 8);
    assert_eq!(result.height, 8);

    let pixels = result.pixels_u8().expect("should have u8 pixels");
    eprintln!("Format: {:?}", result.format);
    eprintln!("Pixel data length: {}", pixels.len());

    // Print actual pixel values for debugging
    let bpp = match result.format {
        PixelFormat::Rgb => 3,
        PixelFormat::Gray => 1,
        _ => panic!("unexpected format {:?}", result.format),
    };

    for y in 0..8 {
        for x in 0..8 {
            let idx = (y * 8 + x) * bpp;
            if bpp == 3 {
                eprint!("({},{},{}) ", pixels[idx], pixels[idx + 1], pixels[idx + 2]);
            } else {
                eprint!("{} ", pixels[idx]);
            }
        }
        eprintln!();
    }

    // All pixels should be 255 (from DC value 1014 -> 1014/8 + 128 = 254.75 -> 255)
    let expected = 255u8;
    let mut max_diff = 0u8;
    for (i, &px) in pixels.iter().enumerate() {
        let diff = px.abs_diff(expected);
        if diff > max_diff {
            max_diff = diff;
            eprintln!(
                "pixel[{}] = {}, expected {}, diff {}",
                i, px, expected, diff
            );
        }
    }
    eprintln!("max_diff = {}", max_diff);

    // All 64 pixels should be 255
    assert!(
        max_diff <= 1,
        "max_diff={max_diff}, expected all pixels to be 255 (or 254 due to rounding)"
    );
}

/// Build a minimal progressive JPEG with configurable DC value and point transform.
///
/// Creates an 8x8 grayscale progressive JPEG with a single DC-only scan.
/// The Huffman table maps code `0` (1-bit) to the given category.
/// Extra bits encode the desired DC diff value.
fn build_progressive_jpeg_dc_only(dc_diff: i16, al: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256);

    // SOI
    buf.extend_from_slice(&[0xFF, 0xD8]);

    // JFIF APP0
    buf.extend_from_slice(&[
        0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00,
        0x48, 0x00, 0x00,
    ]);

    // DQT: Q[0] = 1 (identity quantization for easy verification)
    buf.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    buf.extend(std::iter::repeat_n(1u8, 64)); // All quant values = 1

    // SOF2 (progressive): 8x8 grayscale, 1 component
    buf.extend_from_slice(&[
        0xFF, 0xC2, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
    ]);

    // Compute category and extra bits for the DC diff
    let abs_val = dc_diff.unsigned_abs();
    let category = if abs_val == 0 {
        0u8
    } else {
        16 - abs_val.leading_zeros() as u8
    };

    let extra_bits = if dc_diff >= 0 {
        dc_diff as u16
    } else {
        // Negative: one's complement encoding
        (dc_diff - 1) as u16 & ((1u16 << category) - 1)
    };

    // DHT: DC table with single 1-bit code `0` -> category
    buf.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00]);
    // BITS: 1 code of length 1, rest 0
    buf.push(0x01); // 1 code of length 1
    buf.extend(std::iter::repeat_n(0x00u8, 15));
    // VALUES: the single code maps to our category
    buf.push(category);

    // SOS: Ss=0, Se=0, Ah=0, Al=al
    buf.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x00, al]);

    // Build bit stream: code 0 (1 bit) + extra_bits (category bits)
    // MSB-first packing into bytes
    let mut bit_accum: u32 = 0;
    let mut bits_in_accum: u32 = 0;

    // Huffman code: single bit `0`
    bit_accum <<= 1;
    bits_in_accum += 1;

    // Extra bits
    if category > 0 {
        bit_accum = (bit_accum << category) | (extra_bits as u32);
        bits_in_accum += category as u32;
    }

    // Pad remaining bits with 1s (JPEG fill bits)
    while bits_in_accum % 8 != 0 {
        bit_accum = (bit_accum << 1) | 1;
        bits_in_accum += 1;
    }

    // Write bytes with byte-stuffing
    let num_bytes = bits_in_accum / 8;
    for i in (0..num_bytes).rev() {
        let byte = ((bit_accum >> (i * 8)) & 0xFF) as u8;
        buf.push(byte);
        if byte == 0xFF {
            buf.push(0x00); // byte-stuffing
        }
    }

    // EOI
    buf.extend_from_slice(&[0xFF, 0xD9]);

    buf
}

/// Compute expected pixel value for DC-only progressive decode with point transform.
fn expected_pixel_for_dc(dc_diff: i16, al: u8, quant_dc: u16) -> u8 {
    // 1. Point transform: coefficient = dc_diff << al
    let coeff = (dc_diff as i32) << al;
    // 2. Dequantize: coeff * Q[0]
    let dequant = coeff * quant_dc as i32;
    // 3. IDCT DC-only: pixel = (dequant + 4 + 1024) >> 3
    //    (rounding: +4 is half of 8; level shift: 1024 = 128 << 3)
    let pixel = (dequant + 4 + 1024) >> 3;
    // 4. Clamp to [0, 255]
    pixel.clamp(0, 255) as u8
}

#[test]
fn test_progressive_dc_al0() {
    // Al=0: no point transform, basic DC decode
    let dc_diff = 100i16;
    let al = 0u8;
    let jpeg = build_progressive_jpeg_dc_only(dc_diff, al);
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    let pixels = result.pixels_u8().expect("u8 pixels");
    let expected = expected_pixel_for_dc(dc_diff, al, 1);
    eprintln!("Al={al}, dc_diff={dc_diff}, expected pixel={expected}");
    eprintln!("Actual first pixel: R={}", pixels[0]);

    for (i, &px) in pixels.iter().enumerate() {
        let diff = (px as i16 - expected as i16).unsigned_abs() as u8;
        assert!(
            diff <= 1,
            "pixel[{i}]={px}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_progressive_dc_al1() {
    // Al=1: point transform shift=1
    let dc_diff = 50i16;
    let al = 1u8;
    let jpeg = build_progressive_jpeg_dc_only(dc_diff, al);
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    let pixels = result.pixels_u8().expect("u8 pixels");
    let expected = expected_pixel_for_dc(dc_diff, al, 1);
    eprintln!("Al={al}, dc_diff={dc_diff}, expected pixel={expected}");
    eprintln!("Actual first pixel: R={}", pixels[0]);

    for (i, &px) in pixels.iter().enumerate() {
        let diff = (px as i16 - expected as i16).unsigned_abs() as u8;
        assert!(
            diff <= 1,
            "pixel[{i}]={px}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_progressive_dc_al2() {
    // Al=2: point transform shift=2
    let dc_diff = 25i16;
    let al = 2u8;
    let jpeg = build_progressive_jpeg_dc_only(dc_diff, al);
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    let pixels = result.pixels_u8().expect("u8 pixels");
    let expected = expected_pixel_for_dc(dc_diff, al, 1);
    eprintln!("Al={al}, dc_diff={dc_diff}, expected pixel={expected}");
    eprintln!("Actual first pixel: R={}", pixels[0]);

    for (i, &px) in pixels.iter().enumerate() {
        let diff = (px as i16 - expected as i16).unsigned_abs() as u8;
        assert!(
            diff <= 1,
            "pixel[{i}]={px}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_progressive_dc_al3() {
    // Al=3: point transform shift=3
    let dc_diff = 10i16;
    let al = 3u8;
    let jpeg = build_progressive_jpeg_dc_only(dc_diff, al);
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    let pixels = result.pixels_u8().expect("u8 pixels");
    let expected = expected_pixel_for_dc(dc_diff, al, 1);
    eprintln!("Al={al}, dc_diff={dc_diff}, expected pixel={expected}");
    eprintln!("Actual first pixel: R={}", pixels[0]);

    for (i, &px) in pixels.iter().enumerate() {
        let diff = (px as i16 - expected as i16).unsigned_abs() as u8;
        assert!(
            diff <= 1,
            "pixel[{i}]={px}, expected={expected}, diff={diff}"
        );
    }
}

#[test]
fn test_progressive_dc_negative_al1() {
    // Al=1 with negative DC diff
    let dc_diff = -50i16;
    let al = 1u8;
    let jpeg = build_progressive_jpeg_dc_only(dc_diff, al);
    let decoder = Decoder::new();
    let result = decoder.decode(&jpeg, Unstoppable).expect("decode failed");

    let pixels = result.pixels_u8().expect("u8 pixels");
    let expected = expected_pixel_for_dc(dc_diff, al, 1);
    eprintln!("Al={al}, dc_diff={dc_diff}, expected pixel={expected}");
    eprintln!("Actual first pixel: R={}", pixels[0]);

    for (i, &px) in pixels.iter().enumerate() {
        let diff = (px as i16 - expected as i16).unsigned_abs() as u8;
        assert!(
            diff <= 1,
            "pixel[{i}]={px}, expected={expected}, diff={diff}"
        );
    }
}

/// Sweep all Al values and various DC diffs to systematically test point transform.
#[test]
fn test_progressive_dc_point_transform_sweep() {
    let dc_diffs: &[i16] = &[0, 1, -1, 10, -10, 50, -50, 100, -100, 127, -128];
    let al_values: &[u8] = &[0, 1, 2, 3];

    let mut failures = Vec::new();

    for &dc_diff in dc_diffs {
        for &al in al_values {
            let jpeg = build_progressive_jpeg_dc_only(dc_diff, al);
            let decoder = Decoder::new();
            let result = match decoder.decode(&jpeg, Unstoppable) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("dc_diff={dc_diff}, al={al}: decode error: {e}"));
                    continue;
                }
            };

            let pixels = result.pixels_u8().expect("u8 pixels");
            let expected = expected_pixel_for_dc(dc_diff, al, 1);

            let mut max_diff = 0u8;
            for &px in pixels.iter() {
                let diff = (px as i16 - expected as i16).unsigned_abs() as u8;
                max_diff = max_diff.max(diff);
            }

            if max_diff > 1 {
                failures.push(format!(
                    "dc_diff={dc_diff}, al={al}: expected={expected}, got={}, max_diff={max_diff}",
                    pixels[0]
                ));
            } else {
                eprintln!(
                    "OK: dc_diff={dc_diff}, al={al}: expected={expected}, got={}, max_diff={max_diff}",
                    pixels[0]
                );
            }
        }
    }

    if !failures.is_empty() {
        panic!("Point transform failures:\n{}", failures.join("\n"));
    }
}
