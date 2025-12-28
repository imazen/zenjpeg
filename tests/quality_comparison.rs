//! Quality comparison tests using perceptual metrics

use dssim::Dssim;
use rgb::RGBA8;
use zenjpeg::{Encoder, Quality};

/// Convert RGB bytes to RGBA for DSSIM
fn rgb_to_rgba(rgb: &[u8]) -> Vec<RGBA8> {
    rgb.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

/// Create a natural-looking test image with gradients and edges
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);

    for y in 0..height {
        for x in 0..width {
            // Mix of gradients and step functions
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Horizontal gradient
            let r = (fx * 255.0) as u8;

            // Vertical gradient with edge
            let g = if fy < 0.5 {
                (fy * 2.0 * 200.0) as u8
            } else {
                200 + ((fy - 0.5) * 2.0 * 55.0) as u8
            };

            // Diagonal pattern
            let b = (((x + y) % 32) as f32 / 32.0 * 128.0 + 64.0) as u8;

            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Compute DSSIM between original and encoded/decoded image
fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let attr = Dssim::new();

    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);

    let orig_img = attr
        .create_image_rgba(&orig_rgba, width, height)
        .expect("Failed to create original image");

    let dec_img = attr
        .create_image_rgba(&dec_rgba, width, height)
        .expect("Failed to create decoded image");

    let (dssim, _) = attr.compare(&orig_img, dec_img);
    dssim.into()
}

#[test]
fn test_quality_vs_dssim() {
    let width = 128;
    let height = 128;
    let original = create_test_image(width, height);

    let qualities = [30, 50, 70, 85, 95];
    let mut prev_dssim = f64::MAX;
    let mut prev_size = 0usize;

    for q in qualities {
        let encoder = Encoder::new().quality(Quality::Standard(q));
        let jpeg_data = encoder.encode_rgb(&original, width, height).unwrap();

        // Decode
        let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
        let decoded = decoder.decode().unwrap();

        let dssim = compute_dssim(&original, &decoded, width, height);
        let size = jpeg_data.len();

        println!("Q{}: size={} bytes, DSSIM={:.6}", q, size, dssim);

        // Higher quality should give lower DSSIM (better)
        if q > 30 {
            assert!(
                dssim <= prev_dssim * 1.1, // Allow 10% tolerance for noise
                "Q{} DSSIM ({:.6}) should be <= Q{} DSSIM ({:.6})",
                q,
                dssim,
                q - 20,
                prev_dssim
            );
        }

        // Higher quality should give larger size
        if q > 30 {
            assert!(
                size >= prev_size,
                "Q{} size ({}) should be >= Q{} size ({})",
                q,
                size,
                q - 20,
                prev_size
            );
        }

        prev_dssim = dssim;
        prev_size = size;
    }
}

#[test]
fn test_high_quality_dssim_threshold() {
    let width = 128;
    let height = 128;
    let original = create_test_image(width, height);

    // At Q95, DSSIM should be very low
    let encoder = Encoder::new().quality(Quality::Standard(95));
    let jpeg_data = encoder.encode_rgb(&original, width, height).unwrap();

    let mut decoder = jpeg_decoder::Decoder::new(&jpeg_data[..]);
    let decoded = decoder.decode().unwrap();

    let dssim = compute_dssim(&original, &decoded, width, height);

    // Based on codec-eval thresholds:
    // < 0.003 is acceptable for "noticeable but okay" quality
    assert!(
        dssim < 0.01,
        "Q95 DSSIM {:.6} should be < 0.01 for good quality",
        dssim
    );

    println!("Q95 DSSIM: {:.6}", dssim);
}

#[test]
fn test_strategy_selection_affects_quality() {
    let width = 128;
    let height = 128;
    let original = create_test_image(width, height);

    // Compare Low vs High mode at same numeric quality
    let low_encoder = Encoder::new().quality(Quality::Low(70));
    let high_encoder = Encoder::new().quality(Quality::High(70));

    let low_jpeg = low_encoder.encode_rgb(&original, width, height).unwrap();
    let high_jpeg = high_encoder.encode_rgb(&original, width, height).unwrap();

    // Decode both
    let low_decoded = {
        let mut decoder = jpeg_decoder::Decoder::new(&low_jpeg[..]);
        decoder.decode().unwrap()
    };
    let high_decoded = {
        let mut decoder = jpeg_decoder::Decoder::new(&high_jpeg[..]);
        decoder.decode().unwrap()
    };

    let low_dssim = compute_dssim(&original, &low_decoded, width, height);
    let high_dssim = compute_dssim(&original, &high_decoded, width, height);

    println!(
        "Low mode Q70: {} bytes, DSSIM={:.6}",
        low_jpeg.len(),
        low_dssim
    );
    println!(
        "High mode Q70: {} bytes, DSSIM={:.6}",
        high_jpeg.len(),
        high_dssim
    );

    // Both should produce reasonable quality
    assert!(
        low_dssim < 0.05,
        "Low mode DSSIM too high: {:.6}",
        low_dssim
    );
    assert!(
        high_dssim < 0.05,
        "High mode DSSIM too high: {:.6}",
        high_dssim
    );
}

#[test]
#[ignore] // Run with --ignored for full comparison
fn test_compare_with_reference_encoders() {
    let width = 256;
    let height = 256;
    let original = create_test_image(width, height);

    let quality = 85;

    // zenjpeg
    let zen_encoder = Encoder::new().quality(Quality::Standard(quality));
    let zen_jpeg = zen_encoder.encode_rgb(&original, width, height).unwrap();
    let zen_decoded = jpeg_decoder::Decoder::new(&zen_jpeg[..]).decode().unwrap();
    let zen_dssim = compute_dssim(&original, &zen_decoded, width, height);

    println!("=== Quality {} Comparison ===", quality);
    println!("zenjpeg:  {} bytes, DSSIM={:.6}", zen_jpeg.len(), zen_dssim);

    // mozjpeg-oxide (if available)
    // TODO: Add when API is stable
    // let moz_encoder = mozjpeg_oxide::Encoder::new().quality(quality);
    // let moz_jpeg = moz_encoder.encode_rgb(&original, width, height).unwrap();

    // jpegli (if available)
    // TODO: Add when API is stable
    // let jpegli_encoder = jpegli::Encoder::new().quality(quality);
    // let jpegli_jpeg = jpegli_encoder.encode_rgb(&original, width, height).unwrap();
}
