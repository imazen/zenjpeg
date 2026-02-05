//! Integration tests for roundtrip quality verification.
//!
//! These tests encode with zenjpeg and decode with jpeg-decoder (reference),
//! then verify quality using DSSIM.
use enough::Unstoppable;

use dssim::Dssim;
use rgb::RGBA8;
use std::fs;
use std::path::Path;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

/// Maximum acceptable DSSIM for quality 90 encoding.
/// Lower is better; 0 = identical, typical good JPEG is < 0.01
const MAX_DSSIM_Q90: f64 = 0.005;

/// Helper function to encode RGB data with given config
fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    quality: f32,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn load_png(path: &Path) -> Option<(Vec<u8>, usize, usize)> {
    let file = fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let (width, height) = (info.width as usize, info.height as usize);

    // Convert to RGB
    let rgb = match info.color_type {
        png::ColorType::Rgb => buf[..width * height * 3].to_vec(),
        png::ColorType::Rgba => buf[..width * height * 4]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..width * height]
            .iter()
            .flat_map(|&g| [g, g, g])
            .collect(),
        png::ColorType::GrayscaleAlpha => buf[..width * height * 2]
            .chunks(2)
            .flat_map(|c| [c[0], c[0], c[0]])
            .collect(),
        _ => return None,
    };

    Some((rgb, width, height))
}

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);

    let orig = attr
        .create_image_rgba(&orig_rgba, width, height)
        .expect("create orig image");
    let comp = attr
        .create_image_rgba(&dec_rgba, width, height)
        .expect("create comp image");

    let (dssim, _) = attr.compare(&orig, comp);
    dssim.into()
}

fn decode_with_jpeg_decoder(jpeg: &[u8]) -> (Vec<u8>, usize, usize) {
    let mut decoder =
        zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg));
    let pixels = decoder.decode().expect("jpeg-decoder decode failed");
    let (width, height) = decoder.dimensions().unwrap();
    (pixels, width, height)
}

/// Test roundtrip quality on the flower_small test image.
#[test]
fn test_roundtrip_flower_small() {
    let path = zenjpeg::test_utils::get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        eprintln!("Skipping test: test image not found. Set JPEGLI_TESTDATA env var.");
        return;
    }

    let (original_rgb, width, height) = load_png(&path).expect("Failed to load test image");

    // Encode with jpegli at quality 90
    let jpeg_data =
        encode_rgb(width as u32, height as u32, &original_rgb, 90.0).expect("jpegli encode failed");

    // Decode with reference decoder (jpeg-decoder)
    let (decoded_rgb, dec_width, dec_height) = decode_with_jpeg_decoder(&jpeg_data);

    assert_eq!(width, dec_width, "Width mismatch");
    assert_eq!(height, dec_height, "Height mismatch");

    // Compute DSSIM
    let dssim = compute_dssim(&original_rgb, &decoded_rgb, width, height);

    println!(
        "Roundtrip test: {}x{}, JPEG size: {} bytes, DSSIM: {:.6}",
        width,
        height,
        jpeg_data.len(),
        dssim
    );

    assert!(
        dssim < MAX_DSSIM_Q90,
        "DSSIM {} exceeds threshold {} for quality 90",
        dssim,
        MAX_DSSIM_Q90
    );
}

/// Test roundtrip quality on a generated gradient image.
#[test]
fn test_roundtrip_gradient() {
    let width = 256;
    let height = 256;

    // Create RGB gradient: R increases left to right, G increases top to bottom
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push(x as u8); // R
            rgb.push(y as u8); // G
            rgb.push(128); // B constant
        }
    }

    // Encode with jpegli at quality 90
    let jpeg_data =
        encode_rgb(width as u32, height as u32, &rgb, 90.0).expect("jpegli encode failed");

    // Decode with reference decoder
    let (decoded_rgb, dec_width, dec_height) = decode_with_jpeg_decoder(&jpeg_data);

    assert_eq!(width, dec_width);
    assert_eq!(height, dec_height);

    let dssim = compute_dssim(&rgb, &decoded_rgb, width, height);

    println!(
        "Gradient test: {}x{}, JPEG size: {} bytes, DSSIM: {:.6}",
        width,
        height,
        jpeg_data.len(),
        dssim
    );

    // Gradients should compress very well
    assert!(dssim < 0.002, "DSSIM {} too high for gradient image", dssim);
}

/// Test roundtrip quality on a solid color image.
#[test]
fn test_roundtrip_solid_color() {
    let width = 64;
    let height = 64;

    // Solid magenta
    let rgb: Vec<u8> = (0..width * height).flat_map(|_| [200u8, 50, 180]).collect();

    let jpeg_data =
        encode_rgb(width as u32, height as u32, &rgb, 90.0).expect("jpegli encode failed");
    let (decoded_rgb, _, _) = decode_with_jpeg_decoder(&jpeg_data);

    let dssim = compute_dssim(&rgb, &decoded_rgb, width, height);

    println!(
        "Solid color test: {}x{}, JPEG size: {} bytes, DSSIM: {:.6}",
        width,
        height,
        jpeg_data.len(),
        dssim
    );

    // Solid colors should be nearly lossless
    assert!(dssim < 0.0001, "DSSIM {} too high for solid color", dssim);
}

/// Test multiple quality levels.
#[test]
fn test_quality_levels() {
    let width = 128;
    let height = 128;

    // Create a more complex pattern
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32 * 255.0).sin().abs() * 255.0) as u8;
            let g = ((y as f32 / height as f32 * 255.0).cos().abs() * 255.0) as u8;
            let b = (((x + y) as f32 / (width + height) as f32).sin().abs() * 255.0) as u8;
            rgb.push(r);
            rgb.push(g);
            rgb.push(b);
        }
    }

    println!("\nQuality level test:");
    let mut prev_size = usize::MAX;
    let mut prev_dssim = f64::MAX;

    for quality in [60, 75, 85, 95] {
        let jpeg_data = encode_rgb(width as u32, height as u32, &rgb, quality as f32)
            .expect("jpegli encode failed");
        let (decoded_rgb, _, _) = decode_with_jpeg_decoder(&jpeg_data);
        let dssim = compute_dssim(&rgb, &decoded_rgb, width, height);

        println!(
            "  Q{}: {} bytes, DSSIM: {:.6}",
            quality,
            jpeg_data.len(),
            dssim
        );

        // Higher quality should mean larger file (roughly)
        if quality > 60 {
            // Allow some variance but generally should increase
            assert!(
                jpeg_data.len() >= prev_size * 8 / 10,
                "Size should roughly increase with quality"
            );
        }

        // Higher quality should mean lower DSSIM (better quality)
        if quality > 60 {
            assert!(
                dssim <= prev_dssim * 1.1, // Allow 10% variance
                "DSSIM should decrease with higher quality"
            );
        }

        prev_size = jpeg_data.len();
        prev_dssim = dssim;
    }
}
