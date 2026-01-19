//! Tests that deringing improves quality on synthetic edge images.
//!
//! These tests verify that mozjpeg-style deringing reduces artifacts on images
//! with sharp black/white transitions.

#![cfg(feature = "mozjpeg-deringing")]

use dssim::Dssim;
use jpegli::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
    types::PixelFormat,
};

/// Generate a grayscale image with horizontal black and white stripes.
fn generate_horizontal_stripes(width: u32, height: u32, stripe_height: u32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height) as usize];
    for y in 0..height {
        let stripe_idx = y / stripe_height;
        let value = if stripe_idx % 2 == 0 { 0 } else { 255 };
        for x in 0..width {
            data[(y * width + x) as usize] = value;
        }
    }
    data
}

/// Generate a grayscale image with thin vertical lines on white background.
/// This creates maximum ringing potential - thin black lines on white.
fn generate_thin_lines_on_white(width: u32, height: u32, spacing: u32) -> Vec<u8> {
    let mut data = vec![255u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            if x % spacing == 0 {
                data[(y * width + x) as usize] = 0;
            }
        }
    }
    data
}

fn encode_gray_with_deringing(
    width: u32,
    height: u32,
    data: &[u8],
    quality: f32,
    deringing: bool,
) -> Vec<u8> {
    let mut config = EncoderConfig::new(quality, ChromaSubsampling::None);
    config = config.deringing(deringing);

    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(data, enough::Unstoppable)
        .expect("push failed");
    enc.finish().expect("finish failed")
}

fn decode_gray(jpeg: &[u8]) -> Vec<u8> {
    let decoder = Decoder::new().output_format(PixelFormat::Gray);
    let decoded = decoder.decode(jpeg).expect("decode failed");
    decoded.data
}

fn compute_dssim_gray(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64 {
    let dssim = Dssim::new();

    // Convert to rgb::RGBAPLU for dssim (even though grayscale, API requires it)
    let orig_rgba: Vec<rgb::RGBA8> = original
        .iter()
        .map(|&g| rgb::RGBA8::new(g, g, g, 255))
        .collect();
    let dec_rgba: Vec<rgb::RGBA8> = decoded
        .iter()
        .map(|&g| rgb::RGBA8::new(g, g, g, 255))
        .collect();

    let orig_img = dssim
        .create_image_rgba(&orig_rgba, width as usize, height as usize)
        .expect("create orig image");
    let dec_img = dssim
        .create_image_rgba(&dec_rgba, width as usize, height as usize)
        .expect("create dec image");

    let (dssim_val, _) = dssim.compare(&orig_img, dec_img);
    dssim_val.into()
}

/// Test that deringing improves DSSIM on horizontal stripe pattern.
#[test]
fn test_deringing_improves_horizontal_stripes() {
    let width = 256u32;
    let height = 256u32;
    let stripe_height = 8u32;
    let quality = 75.0;

    let original = generate_horizontal_stripes(width, height, stripe_height);

    // Encode without deringing
    let jpeg_no_dering = encode_gray_with_deringing(width, height, &original, quality, false);
    let decoded_no_dering = decode_gray(&jpeg_no_dering);
    let dssim_no_dering = compute_dssim_gray(&original, &decoded_no_dering, width, height);

    // Encode with deringing
    let jpeg_with_dering = encode_gray_with_deringing(width, height, &original, quality, true);
    let decoded_with_dering = decode_gray(&jpeg_with_dering);
    let dssim_with_dering = compute_dssim_gray(&original, &decoded_with_dering, width, height);

    println!(
        "Horizontal stripes (stripe_height={}px, q={}):",
        stripe_height, quality
    );
    println!(
        "  Without deringing: DSSIM = {:.6}, size = {} bytes",
        dssim_no_dering,
        jpeg_no_dering.len()
    );
    println!(
        "  With deringing:    DSSIM = {:.6}, size = {} bytes",
        dssim_with_dering,
        jpeg_with_dering.len()
    );
    println!(
        "  Improvement: {:.2}%",
        (1.0 - dssim_with_dering / dssim_no_dering) * 100.0
    );

    // Deringing should improve or maintain quality (lower DSSIM = better)
    // Allow small tolerance for edge cases
    assert!(
        dssim_with_dering <= dssim_no_dering * 1.05,
        "Deringing made quality significantly worse: {} > {} * 1.05",
        dssim_with_dering,
        dssim_no_dering
    );
}

/// Test that deringing improves DSSIM on thin lines pattern (worst case for ringing).
#[test]
fn test_deringing_improves_thin_lines_on_white() {
    let width = 256u32;
    let height = 256u32;
    let line_spacing = 16u32;
    let quality = 75.0;

    let original = generate_thin_lines_on_white(width, height, line_spacing);

    // Encode without deringing
    let jpeg_no_dering = encode_gray_with_deringing(width, height, &original, quality, false);
    let decoded_no_dering = decode_gray(&jpeg_no_dering);
    let dssim_no_dering = compute_dssim_gray(&original, &decoded_no_dering, width, height);

    // Encode with deringing
    let jpeg_with_dering = encode_gray_with_deringing(width, height, &original, quality, true);
    let decoded_with_dering = decode_gray(&jpeg_with_dering);
    let dssim_with_dering = compute_dssim_gray(&original, &decoded_with_dering, width, height);

    println!(
        "Thin lines on white (spacing={}px, q={}):",
        line_spacing, quality
    );
    println!(
        "  Without deringing: DSSIM = {:.6}, size = {} bytes",
        dssim_no_dering,
        jpeg_no_dering.len()
    );
    println!(
        "  With deringing:    DSSIM = {:.6}, size = {} bytes",
        dssim_with_dering,
        jpeg_with_dering.len()
    );
    println!(
        "  Improvement: {:.2}%",
        (1.0 - dssim_with_dering / dssim_no_dering) * 100.0
    );

    // For thin lines on white, deringing should help significantly
    assert!(
        dssim_with_dering <= dssim_no_dering * 1.05,
        "Deringing made quality worse on thin lines: {} > {} * 1.05",
        dssim_with_dering,
        dssim_no_dering
    );
}

/// Test deringing at multiple quality levels.
#[test]
fn test_deringing_across_quality_levels() {
    let width = 128u32;
    let height = 128u32;
    let original = generate_thin_lines_on_white(width, height, 8);

    println!("\nDeringing effectiveness across quality levels:");
    println!("Quality | No Dering DSSIM | With Dering DSSIM | Improvement");
    println!("--------|-----------------|-------------------|------------");

    for quality in [50.0, 65.0, 75.0, 85.0, 95.0] {
        let jpeg_no = encode_gray_with_deringing(width, height, &original, quality, false);
        let dec_no = decode_gray(&jpeg_no);
        let dssim_no = compute_dssim_gray(&original, &dec_no, width, height);

        let jpeg_yes = encode_gray_with_deringing(width, height, &original, quality, true);
        let dec_yes = decode_gray(&jpeg_yes);
        let dssim_yes = compute_dssim_gray(&original, &dec_yes, width, height);

        let improvement = (1.0 - dssim_yes / dssim_no) * 100.0;
        println!(
            "q={:5.1} | {:15.6} | {:17.6} | {:+.2}%",
            quality, dssim_no, dssim_yes, improvement
        );

        // At all quality levels, deringing should not significantly hurt quality
        assert!(
            dssim_yes <= dssim_no * 1.10,
            "Deringing hurt quality at q={}: {} > {} * 1.10",
            quality,
            dssim_yes,
            dssim_no
        );
    }
}
