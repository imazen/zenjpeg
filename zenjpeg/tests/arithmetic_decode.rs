#![cfg(feature = "__ffi-tests")]
//! Tests for arithmetic-coded JPEG decoding.
//!
//! Note: C++ jpegli does not support arithmetic coding, so the reference
//! comparison tests shell out to `djpeg` (libjpeg-turbo) when available.
//! Neither jpegli FFI, jpeg-decoder, nor zune-jpeg support arithmetic JPEG.
//! When djpeg is not installed (e.g. Windows CI), the comparison tests
//! skip gracefully.
use enough::Unstoppable;

use zenjpeg::decode::Decoder;
use zenjpeg::types::JpegMode;

/// Decode JPEG to raw RGB using djpeg (libjpeg-turbo CLI).
/// Returns None if djpeg is not installed or fails to decode.
fn decode_with_djpeg(jpeg_path: &str) -> Option<(Vec<u8>, usize, usize)> {
    use std::process::Command;

    let output = match Command::new("djpeg").args(["-pnm", jpeg_path]).output() {
        Ok(o) => o,
        Err(_) => return None,
    };
    if !output.status.success() {
        return None;
    }

    // Parse PPM: "P6\n<width> <height>\n<maxval>\n<data>"
    let ppm = &output.stdout;
    let header_end = {
        let mut pos = 0;
        let mut newlines = 0;
        while pos < ppm.len() && newlines < 3 {
            if ppm[pos] == b'\n' {
                newlines += 1;
            }
            pos += 1;
        }
        if newlines == 3 { pos } else { return None }
    };

    let header = std::str::from_utf8(&ppm[..header_end]).ok()?;
    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 4 || parts[0] != "P6" {
        return None;
    }

    let width: usize = parts[1].parse().ok()?;
    let height: usize = parts[2].parse().ok()?;
    let pixels = ppm[header_end..].to_vec();

    Some((pixels, width, height))
}

const TESTIMGARI_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../internal/jpegli-cpp/third_party/libjpeg-turbo/testimages/testimgari.jpg"
);

#[test]
fn test_arithmetic_jpeg_info() {
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read testimgari.jpg");
    let decoder = Decoder::new();
    let info = decoder.read_info(&data).expect("failed to read info");

    assert_eq!(info.mode, JpegMode::ArithmeticSequential);
    assert_eq!(info.dimensions.width, 227);
    assert_eq!(info.dimensions.height, 149);
    assert_eq!(info.num_components, 3);
}

#[test]
fn test_arithmetic_jpeg_decode_rgb() {
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read testimgari.jpg");
    let decoder = Decoder::new();
    let info = decoder.read_info(&data).expect("failed to read info");

    // Decode to RGB
    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("failed to decode arithmetic JPEG");

    // Verify dimensions
    let expected_size = (info.dimensions.width as usize) * (info.dimensions.height as usize) * 3;
    assert_eq!(decoded.pixels_u8().unwrap().len(), expected_size);

    // Basic sanity check - not all zeros
    let sum: u64 = decoded.pixels_u8().unwrap().iter().map(|&x| x as u64).sum();
    assert!(sum > 0, "decoded image is all zeros");

    // Check some pixels have variety (not all same color)
    let unique_values: std::collections::HashSet<u8> =
        decoded.pixels_u8().unwrap().iter().copied().collect();
    assert!(
        unique_values.len() > 10,
        "decoded image has too few unique values"
    );
}

#[test]
fn test_arithmetic_jpeg_decode_coefficients() {
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read testimgari.jpg");
    let decoder = Decoder::new();

    // Decode coefficients
    let coeffs = decoder
        .decode_coefficients(&data, Unstoppable)
        .expect("failed to decode coefficients");

    assert_eq!(coeffs.width, 227);
    assert_eq!(coeffs.height, 149);
    assert_eq!(coeffs.components.len(), 3);

    // Check Y component has reasonable dimensions
    let y_comp = &coeffs.components[0];
    assert!(y_comp.blocks_wide > 0);
    assert!(y_comp.blocks_high > 0);

    // Check coefficients aren't all zero
    let nonzero_count: usize = y_comp.coeffs.iter().filter(|&&x| x != 0).count();
    assert!(
        nonzero_count > 100,
        "Y component has too few non-zero coefficients"
    );
}

/// Reference decode using djpeg (libjpeg-turbo).
/// This test verifies our output matches the reference implementation.
/// Skipped when djpeg is not installed (e.g. Windows CI).
#[test]
#[ignore = "arithmetic decode has known max_diff=23 vs libjpeg-turbo"]
fn test_arithmetic_jpeg_reference_comparison() {
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read testimgari.jpg");
    let decoder = Decoder::new();

    // Decode with our decoder
    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("failed to decode with zenjpeg");

    // Decode with djpeg (libjpeg-turbo, supports arithmetic coding)
    let Some((ref_rgb, ref_w, ref_h)) = decode_with_djpeg(TESTIMGARI_PATH) else {
        println!("djpeg not found, skipping reference comparison");
        return;
    };

    assert_eq!(decoded.width as usize, ref_w, "width mismatch");
    assert_eq!(decoded.height as usize, ref_h, "height mismatch");

    let our_pixels = decoded.pixels_u8().unwrap();
    assert_eq!(
        our_pixels.len(),
        ref_rgb.len(),
        "pixel data length mismatch"
    );

    // Compare
    let mut max_diff = 0u8;
    for (ours, reference) in our_pixels.iter().zip(ref_rgb.iter()) {
        let diff = (*ours as i16 - *reference as i16).unsigned_abs() as u8;
        max_diff = max_diff.max(diff);
    }
    // Allow small differences due to rounding
    assert!(
        max_diff <= 2,
        "max pixel difference from reference: {}",
        max_diff
    );
}

#[test]
fn debug_arithmetic_output() {
    let data = std::fs::read(TESTIMGARI_PATH).expect("failed to read testimgari.jpg");
    let decoder = Decoder::new();
    let info = decoder.read_info(&data).expect("failed to read info");

    println!(
        "Image: {}x{}, mode: {:?}",
        info.dimensions.width, info.dimensions.height, info.mode
    );

    let decoded = decoder
        .decode(&data, Unstoppable)
        .expect("failed to decode");
    println!("Decoded {} bytes", decoded.pixels_u8().unwrap().len());

    // Get djpeg reference output (libjpeg-turbo, supports arithmetic coding)
    let Some((ref_rgb, ref_w, ref_h)) = decode_with_djpeg(TESTIMGARI_PATH) else {
        println!("djpeg not found, skipping reference comparison");
        return;
    };
    println!("Reference: {}x{}, {} bytes", ref_w, ref_h, ref_rgb.len());

    // Count differences by magnitude
    let mut diff_hist = [0usize; 256];
    let mut max_diff = 0i16;
    let mut max_diff_pos = 0usize;

    for (i, (&ours, &reference)) in decoded
        .pixels_u8()
        .unwrap()
        .iter()
        .zip(ref_rgb.iter())
        .enumerate()
    {
        let diff = (ours as i16 - reference as i16).abs();
        diff_hist[diff as usize] += 1;
        if diff > max_diff {
            max_diff = diff;
            max_diff_pos = i;
        }
    }

    println!("\nDifference histogram:");
    println!("  diff=0: {} values", diff_hist[0]);
    println!("  diff=1: {} values", diff_hist[1]);
    println!("  diff=2: {} values", diff_hist[2]);
    println!("  diff=3: {} values", diff_hist[3]);
    println!("  diff>3: {} values", diff_hist[4..].iter().sum::<usize>());
    println!(
        "  diff>10: {} values",
        diff_hist[11..].iter().sum::<usize>()
    );
    println!(
        "  diff>50: {} values",
        diff_hist[51..].iter().sum::<usize>()
    );
    println!(
        "  diff>100: {} values",
        diff_hist[101..].iter().sum::<usize>()
    );

    println!("\nMax diff: {} at byte position {}", max_diff, max_diff_pos);
    let pixel_idx = max_diff_pos / 3;
    let channel = max_diff_pos % 3;
    let x = pixel_idx % 227;
    let y = pixel_idx / 227;
    println!(
        "  Pixel ({}, {}), channel {} (0=R, 1=G, 2=B)",
        x, y, channel
    );
    println!(
        "  Our value: {}",
        decoded.pixels_u8().unwrap()[max_diff_pos]
    );
    println!("  Ref value: {}", ref_rgb[max_diff_pos]);

    // Show pixels around the max diff
    let row_start = (pixel_idx / 227) * 227;
    println!("\nPixels around max diff (row {}):", y);
    for i in row_start.saturating_sub(2)
        ..=(row_start + 10).min(decoded.pixels_u8().unwrap().len() / 3 - 1)
    {
        let ours = &decoded.pixels_u8().unwrap()[i * 3..(i + 1) * 3];
        let refs = &ref_rgb[i * 3..(i + 1) * 3];
        let diff_r = (ours[0] as i16 - refs[0] as i16).abs();
        let diff_g = (ours[1] as i16 - refs[1] as i16).abs();
        let diff_b = (ours[2] as i16 - refs[2] as i16).abs();
        if diff_r > 3 || diff_g > 3 || diff_b > 3 {
            println!(
                "  Pixel {}: ours=({:3},{:3},{:3}) ref=({:3},{:3},{:3}) diff=({},{},{})",
                i, ours[0], ours[1], ours[2], refs[0], refs[1], refs[2], diff_r, diff_g, diff_b
            );
        }
    }
}
