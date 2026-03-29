//! Smoke tests for the Decoder deblocking API.

use enough::Unstoppable;
use zenjpeg::decode::DeblockMode;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn make_test_jpeg(quality: u8) -> Vec<u8> {
    make_test_jpeg_progressive(quality, true)
}

/// Create test JPEG with configurable progressive mode.
fn make_test_jpeg_progressive(quality: u8, progressive: bool) -> Vec<u8> {
    // Use a simple gradient pattern that produces visible blocking at low Q
    let w = 128u32;
    let h = 128;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 3) as usize;
            pixels[idx] = (x * 2) as u8;
            pixels[idx + 1] = (y * 2) as u8;
            pixels[idx + 2] = 128;
        }
    }
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).progressive(progressive);
    config
        .encode_bytes(&pixels, w, h, PixelLayout::Rgb8Srgb)
        .unwrap()
}

/// Create a baseline (non-progressive) test JPEG for scanline streaming tests.
fn make_baseline_test_jpeg(quality: u8) -> Vec<u8> {
    make_test_jpeg_progressive(quality, false)
}

#[test]
fn deblock_off_matches_default() {
    let jpeg = make_test_jpeg(50);
    let default = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();
    let off = Decoder::new()
        .deblock(DeblockMode::Off)
        .decode(&jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();
    assert_eq!(default, off, "DeblockMode::Off should match default decode");
}

#[test]
fn deblock_boundary_produces_different_output() {
    let jpeg = make_test_jpeg(20); // Low Q to make blocking visible
    let plain = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();
    let deblocked = Decoder::new()
        .deblock(DeblockMode::Boundary4Tap)
        .decode(&jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();

    assert_eq!(plain.len(), deblocked.len(), "same output size");
    let diffs = plain
        .iter()
        .zip(deblocked.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(diffs > 0, "Boundary4Tap should modify some pixels at Q20");
    println!(
        "{diffs} pixels differ out of {} ({:.1}%)",
        plain.len(),
        diffs as f64 / plain.len() as f64 * 100.0
    );
}

#[test]
fn deblock_knusperli_produces_different_output() {
    let jpeg = make_test_jpeg(10); // Very low Q
    let plain = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();
    let deblocked = Decoder::new()
        .deblock(DeblockMode::Knusperli)
        .decode(&jpeg, Unstoppable)
        .unwrap()
        .into_pixels_u8()
        .unwrap();

    assert_eq!(plain.len(), deblocked.len(), "same output size");
    let diffs = plain
        .iter()
        .zip(deblocked.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(diffs > 0, "Knusperli should modify pixels at Q10");
    println!(
        "{diffs} pixels differ out of {} ({:.1}%)",
        plain.len(),
        diffs as f64 / plain.len() as f64 * 100.0
    );
}

#[test]
fn deblock_auto_works() {
    let jpeg = make_test_jpeg(30);
    let result = Decoder::new()
        .deblock(DeblockMode::Auto)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let pixels = result.into_pixels_u8().unwrap();
    assert!(!pixels.is_empty(), "Auto deblock should produce output");
}

#[test]
fn deblock_scanline_knusperli_falls_back_to_buffered() {
    let jpeg = make_baseline_test_jpeg(50);
    // Knusperli in scanline_reader transparently falls back to decode() + buffered
    let mut reader = Decoder::new()
        .deblock(DeblockMode::Knusperli)
        .scanline_reader(&jpeg)
        .expect("Knusperli should work via fallback, not error");
    let mut buf = vec![0u8; 128 * 128 * 3];
    let rows = reader
        .read_rows_rgb8(imgref::ImgRefMut::new(&mut buf, 128 * 3, 128))
        .unwrap();
    assert_eq!(rows, 128);
}

#[test]
fn deblock_scanline_boundary4tap_succeeds() {
    let jpeg = make_baseline_test_jpeg(20); // Low Q baseline so deblocking has effect
    let mut reader = Decoder::new()
        .deblock(DeblockMode::Boundary4Tap)
        .scanline_reader(&jpeg)
        .expect("Boundary4Tap should be supported in scanline_reader");

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut pixels = vec![0u8; w * h * 3];
    let mut rows_read = 0;
    while rows_read < h {
        let remaining = h - rows_read;
        let output = imgref::ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
        rows_read += reader.read_rows_rgb8(output).unwrap();
    }
    assert_eq!(rows_read, h, "should read all rows");
    assert!(
        !pixels.iter().all(|&v| v == 0),
        "should produce non-zero pixels"
    );
}

#[test]
fn deblock_scanline_auto_succeeds() {
    let jpeg = make_baseline_test_jpeg(30);
    let mut reader = Decoder::new()
        .deblock(DeblockMode::Auto)
        .scanline_reader(&jpeg)
        .expect("Auto should be supported in scanline_reader (resolves to Boundary4Tap)");

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut pixels = vec![0u8; w * h * 3];
    let mut rows_read = 0;
    while rows_read < h {
        let remaining = h - rows_read;
        let output = imgref::ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
        rows_read += reader.read_rows_rgb8(output).unwrap();
    }
    assert_eq!(rows_read, h);
}

#[test]
fn deblock_scanline_off_matches_default_scanline() {
    // Verify that DeblockMode::Off produces byte-identical output to the default
    let jpeg = make_baseline_test_jpeg(50);

    let decode_scanline = |mode: DeblockMode| -> Vec<u8> {
        let mut reader = Decoder::new().deblock(mode).scanline_reader(&jpeg).unwrap();
        let w = reader.width() as usize;
        let h = reader.height() as usize;
        let mut pixels = vec![0u8; w * h * 3];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let output = imgref::ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
            rows_read += reader.read_rows_rgb8(output).unwrap();
        }
        pixels
    };

    let default_pixels = decode_scanline(DeblockMode::Off);
    let no_deblock = {
        let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();
        let w = reader.width() as usize;
        let h = reader.height() as usize;
        let mut pixels = vec![0u8; w * h * 3];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let output = imgref::ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
            rows_read += reader.read_rows_rgb8(output).unwrap();
        }
        pixels
    };

    assert_eq!(
        default_pixels, no_deblock,
        "DeblockMode::Off scanline output must be byte-identical to default (no deblock)"
    );
}

#[test]
fn deblock_scanline_boundary_differs_from_off() {
    let jpeg = make_baseline_test_jpeg(5); // Very low Q baseline for maximum blocking

    let decode_scanline = |mode: DeblockMode| -> Vec<u8> {
        let mut reader = Decoder::new().deblock(mode).scanline_reader(&jpeg).unwrap();
        let w = reader.width() as usize;
        let h = reader.height() as usize;
        let mut pixels = vec![0u8; w * h * 3];
        let mut rows_read = 0;
        while rows_read < h {
            let remaining = h - rows_read;
            let output = imgref::ImgRefMut::new(&mut pixels[rows_read * w * 3..], w * 3, remaining);
            rows_read += reader.read_rows_rgb8(output).unwrap();
        }
        pixels
    };

    let off = decode_scanline(DeblockMode::Off);
    let deblocked = decode_scanline(DeblockMode::Boundary4Tap);

    assert_eq!(off.len(), deblocked.len(), "same output size");
    let diffs = off
        .iter()
        .zip(deblocked.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diffs > 0,
        "Boundary4Tap scanline should produce different pixels at Q20"
    );
    println!(
        "scanline deblock: {diffs} pixels differ out of {} ({:.1}%)",
        off.len(),
        diffs as f64 / off.len() as f64 * 100.0
    );
}

#[test]
fn deblock_all_modes_same_dimensions() {
    let jpeg = make_test_jpeg(50);
    let modes = [
        DeblockMode::Off,
        DeblockMode::Boundary4Tap,
        DeblockMode::Knusperli,
        DeblockMode::Auto,
        DeblockMode::AutoStreamable,
    ];
    let mut sizes: Vec<(DeblockMode, usize)> = Vec::new();
    for mode in modes {
        let result = Decoder::new()
            .deblock(mode)
            .decode(&jpeg, Unstoppable)
            .unwrap();
        let pixels = result.into_pixels_u8().unwrap();
        sizes.push((mode, pixels.len()));
    }
    let expected = sizes[0].1;
    for (mode, size) in &sizes {
        assert_eq!(
            *size, expected,
            "Mode {mode:?} produced different pixel count"
        );
    }
}
