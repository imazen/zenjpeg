//! Basic XYB encoding tests.
//!
//! Verifies that the XYB encoder produces valid, decodable output.

use enough::Unstoppable;
use jpegli::{ChromaSubsampling, Decoder, EncoderConfig, PixelLayout};

/// Encode with XYB color mode
fn encode_xyb(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::new()
        .quality(quality)
        .xyb();

    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder.push_packed(rgb, Never).expect("push failed");
    encoder.finish().expect("XYB encode failed")
}

#[test]
fn test_xyb_basic() {
    let width = 64u32;
    let height = 64u32;

    // Create gradient test image
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            rgb[idx] = ((x * 4) % 256) as u8; // R
            rgb[idx + 1] = ((y * 4) % 256) as u8; // G
            rgb[idx + 2] = (((x + y) * 2) % 256) as u8; // B
        }
    }

    for quality in [70.0, 85.0, 95.0] {
        let jpeg = encode_xyb(&rgb, width, height, quality);

        println!("Q{}: XYB encoder = {} bytes", quality as i32, jpeg.len());

        // Verify valid JPEG
        assert!(!jpeg.is_empty(), "JPEG is empty");
        assert_eq!(jpeg[0..2], [0xFF, 0xD8], "not valid JPEG SOI");

        // Verify it can be decoded
        let decoded = Decoder::new().decode(&jpeg).expect("failed to decode JPEG");
        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
    }
}

#[test]
fn test_xyb_output_quality() {
    use dssim::Dssim;
    use rgb::RGBA8;

    fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
        data.chunks(3)
            .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
            .collect()
    }

    let width = 128u32;
    let height = 128u32;

    // Create more complex test pattern
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            // Gradient with some variation
            rgb[idx] = ((x * 2 + (y % 8) * 4) % 256) as u8;
            rgb[idx + 1] = ((y * 2 + (x % 8) * 4) % 256) as u8;
            rgb[idx + 2] = (((x ^ y) * 3) % 256) as u8;
        }
    }

    let quality = 90.0;
    let jpeg = encode_xyb(&rgb, width, height, quality);

    // Decode and measure quality
    let decoded = Decoder::new()
        .apply_icc(true)
        .decode(&jpeg)
        .expect("decode failed");

    // Compute DSSIM
    let dssim = Dssim::new();
    let orig_rgba = rgb_to_rgba(&rgb);
    let decoded_rgba = rgb_to_rgba(&decoded.data);

    let orig_img = dssim
        .create_image_rgba(&orig_rgba, width as usize, height as usize)
        .expect("create orig");
    let decoded_img = dssim
        .create_image_rgba(&decoded_rgba, width as usize, height as usize)
        .expect("create decoded");

    let (dssim_val, _) = dssim.compare(&orig_img, decoded_img);

    println!(
        "XYB Q{}: {} bytes, DSSIM: {:.6}",
        quality as i32,
        jpeg.len(),
        dssim_val
    );
}
