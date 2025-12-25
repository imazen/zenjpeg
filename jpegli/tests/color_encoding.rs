//! Color encoding roundtrip tests.
//!
//! These tests validate that color space conversions are invertible and
//! that ICC profile handling works correctly.

use jpegli::color::{rgb_to_ycbcr, ycbcr_to_rgb};
use jpegli::transfer_functions::{
    hlg_display_from_encoded, hlg_encoded_from_display, srgb_display_from_encoded,
    srgb_encoded_from_display, PQ,
};
use jpegli::xyb::{linear_rgb_to_xyb, scale_xyb, unscale_xyb, xyb_to_linear_rgb};

// ============================================================================
// RGB ↔ YCbCr Roundtrip
// ============================================================================

/// Test RGB to YCbCr roundtrip.
#[test]
fn test_rgb_ycbcr_roundtrip() {
    let mut max_error: f64 = 0.0;

    // Test various RGB values
    for r in (0..=255).step_by(17) {
        for g in (0..=255).step_by(17) {
            for b in (0..=255).step_by(17) {
                let (y, cb, cr) = rgb_to_ycbcr(r, g, b);
                let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);

                let error_r = (r as i16 - r2 as i16).abs() as f64;
                let error_g = (g as i16 - g2 as i16).abs() as f64;
                let error_b = (b as i16 - b2 as i16).abs() as f64;
                max_error = max_error.max(error_r).max(error_g).max(error_b);
            }
        }
    }

    println!("RGB↔YCbCr max roundtrip error: {}", max_error);
    assert!(
        max_error <= 1.0,
        "RGB↔YCbCr roundtrip error {} exceeds threshold 1.0",
        max_error
    );
}

/// Test specific RGB values for YCbCr conversion.
#[test]
fn test_rgb_ycbcr_known_values() {
    // Black
    let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
    assert_eq!(y, 0);
    assert!((cb as i16 - 128).abs() <= 1);
    assert!((cr as i16 - 128).abs() <= 1);

    // White
    let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
    assert_eq!(y, 255);
    assert!((cb as i16 - 128).abs() <= 1);
    assert!((cr as i16 - 128).abs() <= 1);

    // Pure red
    let (y, cb, cr) = rgb_to_ycbcr(255, 0, 0);
    assert!((y as i16 - 76).abs() <= 1); // ~0.299 * 255 = 76
    assert!(cr > 128); // Red increases Cr

    // Pure green
    let (y, cb, cr) = rgb_to_ycbcr(0, 255, 0);
    assert!((y as i16 - 150).abs() <= 1); // ~0.587 * 255 = 150
    assert!(cb < 128); // Green decreases Cb
    assert!(cr < 128); // Green decreases Cr

    // Pure blue
    let (y, cb, cr) = rgb_to_ycbcr(0, 0, 255);
    assert!((y as i16 - 29).abs() <= 1); // ~0.114 * 255 = 29
    assert!(cb > 128); // Blue increases Cb
}

// ============================================================================
// XYB Roundtrip
// ============================================================================

/// Test linear RGB to XYB roundtrip.
#[test]
fn test_xyb_roundtrip() {
    let mut max_error: f64 = 0.0;
    const NUM_SAMPLES: usize = 32;

    for i in 0..NUM_SAMPLES {
        for j in 0..NUM_SAMPLES {
            for k in 0..NUM_SAMPLES {
                let r = i as f32 / (NUM_SAMPLES - 1) as f32;
                let g = j as f32 / (NUM_SAMPLES - 1) as f32;
                let b = k as f32 / (NUM_SAMPLES - 1) as f32;

                let (x, y, b_xyb) = linear_rgb_to_xyb(r, g, b);
                let (r2, g2, b2) = xyb_to_linear_rgb(x, y, b_xyb);

                let error_r = (r - r2).abs() as f64;
                let error_g = (g - g2).abs() as f64;
                let error_b = (b - b2).abs() as f64;
                max_error = max_error.max(error_r).max(error_g).max(error_b);
            }
        }
    }

    println!("XYB roundtrip max error: {:.2e}", max_error);
    assert!(
        max_error < 1e-5,
        "XYB roundtrip error {} exceeds threshold",
        max_error
    );
}

/// Test XYB scaling and unscaling.
#[test]
fn test_xyb_scale_roundtrip() {
    let mut max_error: f64 = 0.0;
    const NUM_SAMPLES: usize = 32;

    for i in 0..NUM_SAMPLES {
        for j in 0..NUM_SAMPLES {
            for k in 0..NUM_SAMPLES {
                let x = (i as f32 / (NUM_SAMPLES - 1) as f32 - 0.5) * 2.0;
                let y = j as f32 / (NUM_SAMPLES - 1) as f32;
                let b = (k as f32 / (NUM_SAMPLES - 1) as f32 - 0.5) * 2.0;

                let (sx, sy, sb) = scale_xyb(x, y, b);
                let (x2, y2, b2) = unscale_xyb(sx, sy, sb);

                let error_x = (x - x2).abs() as f64;
                let error_y = (y - y2).abs() as f64;
                let error_b = (b - b2).abs() as f64;
                max_error = max_error.max(error_x).max(error_y).max(error_b);
            }
        }
    }

    println!("XYB scale roundtrip max error: {:.2e}", max_error);
    assert!(
        max_error < 1e-5,
        "XYB scale roundtrip error {} exceeds threshold",
        max_error
    );
}

// ============================================================================
// Transfer Function Roundtrip
// ============================================================================

/// Test sRGB transfer function roundtrip.
#[test]
fn test_srgb_transfer_roundtrip() {
    let mut max_error: f64 = 0.0;

    for i in 0..=1000 {
        let linear = i as f64 / 1000.0;
        let encoded = srgb_encoded_from_display(linear);
        let decoded = srgb_display_from_encoded(encoded);
        let error = (linear - decoded).abs();
        max_error = max_error.max(error);
    }

    println!("sRGB transfer roundtrip max error: {:.2e}", max_error);
    assert!(max_error < 1e-10);
}

/// Test PQ transfer function roundtrip.
#[test]
fn test_pq_transfer_roundtrip() {
    let pq = PQ::new(10000.0);
    let mut max_error: f64 = 0.0;

    for i in 0..=1000 {
        let linear = i as f64 / 1000.0;
        let encoded = pq.encoded_from_display(linear);
        let decoded = pq.display_from_encoded(encoded);
        let error = (linear - decoded).abs();
        max_error = max_error.max(error);
    }

    println!("PQ transfer roundtrip max error: {:.2e}", max_error);
    assert!(max_error < 1e-10);
}

/// Test HLG transfer function roundtrip.
#[test]
fn test_hlg_transfer_roundtrip() {
    let mut max_error: f64 = 0.0;

    for i in 0..=1000 {
        let linear = i as f64 / 1000.0;
        let encoded = hlg_encoded_from_display(linear);
        let decoded = hlg_display_from_encoded(encoded);
        let error = (linear - decoded).abs();
        max_error = max_error.max(error);
    }

    println!("HLG transfer roundtrip max error: {:.2e}", max_error);
    assert!(max_error < 1e-10);
}

// ============================================================================
// Color Space Description Tests
// ============================================================================

/// Test color space enum properties.
#[test]
fn test_color_space_properties() {
    use jpegli::types::ColorSpace;

    // RGB has 3 components
    assert_eq!(ColorSpace::Rgb.num_components(), 3);
    // Grayscale has 1 component
    assert_eq!(ColorSpace::Grayscale.num_components(), 1);
    // YCbCr has 3 components
    assert_eq!(ColorSpace::YCbCr.num_components(), 3);
    // CMYK has 4 components
    assert_eq!(ColorSpace::Cmyk.num_components(), 4);
}

/// Test pixel format byte sizes.
#[test]
fn test_pixel_format_sizes() {
    use jpegli::types::PixelFormat;

    // Standard formats
    assert_eq!(PixelFormat::Gray.bytes_per_pixel(), 1);
    assert_eq!(PixelFormat::Rgb.bytes_per_pixel(), 3);
    assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Bgr.bytes_per_pixel(), 3);
    assert_eq!(PixelFormat::Bgra.bytes_per_pixel(), 4);
    assert_eq!(PixelFormat::Cmyk.bytes_per_pixel(), 4);
}
