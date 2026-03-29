//! Smoke tests for the Decoder deblocking API.

use enough::Unstoppable;
use zenjpeg::decode::DeblockMode;
use zenjpeg::decoder::Decoder;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

fn make_test_jpeg(quality: u8) -> Vec<u8> {
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
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    config.encode_bytes(&pixels, w, h, PixelLayout::Rgb8Srgb).unwrap()
}

#[test]
fn deblock_off_matches_default() {
    let jpeg = make_test_jpeg(50);
    let default = Decoder::new().apply_icc(false)
        .decode(&jpeg, Unstoppable).unwrap().into_pixels_u8().unwrap();
    let off = Decoder::new().apply_icc(false).deblock(DeblockMode::Off)
        .decode(&jpeg, Unstoppable).unwrap().into_pixels_u8().unwrap();
    assert_eq!(default, off, "DeblockMode::Off should match default decode");
}

#[test]
fn deblock_boundary_produces_different_output() {
    let jpeg = make_test_jpeg(20); // Low Q to make blocking visible
    let plain = Decoder::new().apply_icc(false)
        .decode(&jpeg, Unstoppable).unwrap().into_pixels_u8().unwrap();
    let deblocked = Decoder::new().apply_icc(false).deblock(DeblockMode::Boundary4Tap)
        .decode(&jpeg, Unstoppable).unwrap().into_pixels_u8().unwrap();

    assert_eq!(plain.len(), deblocked.len(), "same output size");
    let diffs = plain.iter().zip(deblocked.iter())
        .filter(|(a, b)| a != b).count();
    assert!(diffs > 0, "Boundary4Tap should modify some pixels at Q20");
    println!("{diffs} pixels differ out of {} ({:.1}%)",
        plain.len(), diffs as f64 / plain.len() as f64 * 100.0);
}

#[test]
fn deblock_knusperli_produces_different_output() {
    let jpeg = make_test_jpeg(10); // Very low Q
    let plain = Decoder::new().apply_icc(false)
        .decode(&jpeg, Unstoppable).unwrap().into_pixels_u8().unwrap();
    let deblocked = Decoder::new().apply_icc(false).deblock(DeblockMode::Knusperli)
        .decode(&jpeg, Unstoppable).unwrap().into_pixels_u8().unwrap();

    assert_eq!(plain.len(), deblocked.len(), "same output size");
    let diffs = plain.iter().zip(deblocked.iter())
        .filter(|(a, b)| a != b).count();
    assert!(diffs > 0, "Knusperli should modify pixels at Q10");
    println!("{diffs} pixels differ out of {} ({:.1}%)",
        plain.len(), diffs as f64 / plain.len() as f64 * 100.0);
}

#[test]
fn deblock_auto_works() {
    let jpeg = make_test_jpeg(30);
    let result = Decoder::new().apply_icc(false).deblock(DeblockMode::Auto)
        .decode(&jpeg, Unstoppable).unwrap();
    let pixels = result.into_pixels_u8().unwrap();
    assert!(!pixels.is_empty(), "Auto deblock should produce output");
}

#[test]
fn deblock_all_modes_same_dimensions() {
    let jpeg = make_test_jpeg(50);
    let modes = [DeblockMode::Off, DeblockMode::Boundary4Tap, DeblockMode::Knusperli, DeblockMode::Auto];
    let mut sizes: Vec<(DeblockMode, usize)> = Vec::new();
    for mode in modes {
        let result = Decoder::new().apply_icc(false).deblock(mode)
            .decode(&jpeg, Unstoppable).unwrap();
        let pixels = result.into_pixels_u8().unwrap();
        sizes.push((mode, pixels.len()));
    }
    let expected = sizes[0].1;
    for (mode, size) in &sizes {
        assert_eq!(*size, expected, "Mode {mode:?} produced different pixel count");
    }
}
