//! Decoder consistency tests.
//!
//! Verifies that the decoder produces consistent output across multiple roundtrips
//! and that encode→decode→encode→decode produces stable results.

use enough::Unstoppable;
use jpegli::{encoder::{EncoderConfig, PixelLayout}, decoder::Decoder};

/// Generate a gradient test image
fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            rgb[idx] = ((x * 255) / width.max(1)) as u8;
            rgb[idx + 1] = ((y * 255) / height.max(1)) as u8;
            rgb[idx + 2] = 128;
        }
    }
    rgb
}

/// Compute maximum pixel difference between two images
fn max_pixel_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x.abs_diff(y))
        .max()
        .unwrap_or(0)
}

fn encode_rgb(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(data, enough::Unstoppable).expect("push failed");
    enc.finish().expect("finish failed")
}

#[test]
fn test_baseline_roundtrip_consistency() {
    let width = 128u32;
    let height = 128u32;
    let rgb = generate_gradient(width as usize, height as usize);

    let config = EncoderConfig::new().quality(90.0);

    // First encode
    let jpeg1 = encode_rgb(width, height, &rgb, &config);

    // Decode
    let decoded1 = Decoder::new().decode(&jpeg1).expect("decode 1 failed");

    // Re-encode from decoded
    let jpeg2 = encode_rgb(width, height, &decoded1.data, &config);

    // Decode again
    let decoded2 = Decoder::new().decode(&jpeg2).expect("decode 2 failed");

    // The decoder should produce identical output when decoding the same JPEG
    // Re-encoded JPEGs will differ due to quantization, but decoding identical
    // bitstreams should always produce identical pixels
    let decode_same_1 = Decoder::new().decode(&jpeg1).expect("decode same 1 failed");
    let decode_same_2 = Decoder::new().decode(&jpeg1).expect("decode same 2 failed");
    assert_eq!(
        decode_same_1.data, decode_same_2.data,
        "Decoding same JPEG twice should produce identical pixels"
    );

    // After one roundtrip, quality should stabilize
    // The second roundtrip shouldn't introduce additional significant error
    let max_diff = max_pixel_diff(&decoded1.data, &decoded2.data);
    assert!(
        max_diff <= 10,
        "Roundtrip stability: max diff {} too high (expected <= 10)",
        max_diff
    );
}

#[test]
fn test_progressive_roundtrip_consistency() {
    let width = 128u32;
    let height = 128u32;
    let rgb = generate_gradient(width as usize, height as usize);

    let config = EncoderConfig::new().quality(90.0).progressive(true);

    // First encode
    let jpeg1 = encode_rgb(width, height, &rgb, &config);

    // Decode
    let decoded1 = Decoder::new().decode(&jpeg1).expect("decode 1 failed");

    // Re-encode from decoded
    let jpeg2 = encode_rgb(width, height, &decoded1.data, &config);

    // Decode again
    let decoded2 = Decoder::new().decode(&jpeg2).expect("decode 2 failed");

    // After one roundtrip, quality should stabilize
    let max_diff = max_pixel_diff(&decoded1.data, &decoded2.data);
    assert!(
        max_diff <= 10,
        "Progressive roundtrip stability: max diff {} too high",
        max_diff
    );
}

#[test]
fn test_decoder_produces_correct_dimensions() {
    let config = EncoderConfig::new().quality(85.0);

    for (width, height) in [(64, 64), (100, 75), (256, 128), (17, 33)] {
        let rgb = generate_gradient(width, height);

        let jpeg = encode_rgb(width as u32, height as u32, &rgb, &config);

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");

        assert_eq!(
            decoded.width as usize, width,
            "Decoded width mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            decoded.height as usize, height,
            "Decoded height mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            decoded.data.len(),
            width * height * 3,
            "Decoded data size mismatch"
        );
    }
}

#[test]
fn test_multiple_roundtrips_converge() {
    let width = 64u32;
    let height = 64u32;
    let rgb = generate_gradient(width as usize, height as usize);

    let config = EncoderConfig::new().quality(90.0);

    let mut current_pixels = rgb.clone();
    let mut previous_diff = u8::MAX;

    // Do 5 roundtrips - error should stabilize/decrease
    for i in 0..5 {
        let jpeg = encode_rgb(width, height, &current_pixels, &config);

        let decoded = Decoder::new().decode(&jpeg).expect("decode failed");
        let max_diff = max_pixel_diff(&current_pixels, &decoded.data);

        // After first roundtrip, differences should not grow significantly
        if i > 0 {
            assert!(
                max_diff <= previous_diff + 2,
                "Roundtrip {} introduced unexpected error: {} vs previous {}",
                i,
                max_diff,
                previous_diff
            );
        }

        previous_diff = max_diff;
        current_pixels = decoded.data;
    }
}

#[test]
fn test_high_quality_roundtrip_low_error() {
    let width = 128u32;
    let height = 128u32;
    let rgb = generate_gradient(width as usize, height as usize);

    let config = EncoderConfig::new().quality(100.0);
    let jpeg = encode_rgb(width, height, &rgb, &config);

    let decoded = Decoder::new().decode(&jpeg).expect("decode failed");

    let max_diff = max_pixel_diff(&rgb, &decoded.data);

    // Q100 should have very low per-pixel error
    // JPEG is lossy, so some pixels will differ, but max difference should be small
    assert!(
        max_diff <= 5,
        "Q100 max diff {} too high (expected <= 5)",
        max_diff
    );
}
