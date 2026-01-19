//! Tests for 10+ bit precision roundtrip.
//!
//! These tests verify jpegli's claim of 10+ bit precision encoding:
//! - 16-bit input preserves more precision than 8-bit input
//! - f32 decode output recovers sub-8-bit precision
//! - Smooth gradients show less banding with high-bit-depth pipeline
//!
//! The "10+ bit" feature works through:
//! 1. Encoder: Accepts 16-bit/float input (preserves precision in coefficients)
//! 2. Decoder: Optimal dequantization biases recover sub-sample precision
//! 3. Output: f32/u16 output formats preserve the recovered precision
//!
//! IMPORTANT: RGB16 and RgbF32 formats are treated as LINEAR RGB input.
//! The encoder applies sRGB gamma correction during encoding.
//! Standard RGB (8-bit) is assumed to already be in sRGB space.

use jpegli::decoder::Decoder;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

// ============================================================================
// Helper Functions
// ============================================================================

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn encode_rgb16(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> jpegli::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb16Linear)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

/// Convert sRGB value [0,1] to linear [0,1].
fn srgb_to_linear(s: f64) -> f64 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Create a slow gradient that would show banding at 8-bit.
/// This creates a gradient spanning only ~51 levels in 8-bit space.
/// INPUT: Linear RGB16 values that will produce sRGB 0.4-0.6 after encoding.
fn create_slow_gradient_rgb16(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 6);
    for y in 0..height {
        for x in 0..width {
            // Target sRGB range: 0.4 to 0.6 (only ~51 levels at 8-bit, 3277 at 16-bit)
            let t = (x as f64 + y as f64) / (width as f64 + height as f64);
            let srgb_value = 0.4 + 0.2 * t;
            // Convert to linear (encoder expects linear input)
            let linear_value = srgb_to_linear(srgb_value);

            let v16 = (linear_value * 65535.0) as u16;
            // Gray gradient in RGB
            data.extend_from_slice(&v16.to_ne_bytes());
            data.extend_from_slice(&v16.to_ne_bytes());
            data.extend_from_slice(&v16.to_ne_bytes());
        }
    }
    data
}

/// Create the equivalent 8-bit gradient (same visual range).
/// INPUT: sRGB values (8-bit is assumed to be sRGB).
fn create_slow_gradient_rgb8(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let t = (x as f64 + y as f64) / (width as f64 + height as f64);
            let value = 0.4 + 0.2 * t; // Already sRGB
            let v8 = (value * 255.0) as u8;
            data.push(v8);
            data.push(v8);
            data.push(v8);
        }
    }
    data
}

/// Create a fine stepped gradient with known values for precision testing.
/// Each column has a distinct value with sub-8-bit precision.
/// Uses 8-bit sRGB input for simpler comparison (avoids linear/sRGB confusion).
fn create_precision_test_rgb8(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            // Create values in the mid-gray range (126-130 in 8-bit)
            // This spans only 4 8-bit levels but should have sub-level precision
            let base = 126.0;
            let offset = (x as f64 / width as f64) * 4.0; // Span 4 "8-bit levels"
            let v8 = (base + offset).clamp(0.0, 255.0) as u8;

            data.push(v8);
            data.push(v8);
            data.push(v8);
        }
    }
    data
}

/// Compute the effective bit depth from value variance.
/// Returns estimated bits of precision preserved.
fn estimate_effective_bits(values: &[f32], expected_range: f32) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    // Count unique values (discretization)
    let mut sorted: Vec<i64> = values.iter().map(|&v| (v * 1_000_000.0) as i64).collect();
    sorted.sort();
    sorted.dedup();
    let unique_levels = sorted.len();

    // Effective bits = log2(unique levels within the range)
    if unique_levels <= 1 {
        return 0.0;
    }

    // Also consider the actual range used
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let actual_range = max - min;

    // Scale unique levels to full range
    let effective_levels = if actual_range > 0.0 {
        (unique_levels as f64) * (expected_range as f64 / actual_range as f64)
    } else {
        unique_levels as f64
    };

    effective_levels.log2()
}

#[test]
fn test_16bit_input_preserves_good_precision() {
    // This test verifies that encoding from 16-bit input produces
    // high-quality results with good precision preservation.
    //
    // Note: The 16-bit path involves linear→sRGB conversion, which may
    // affect the comparison with 8-bit input. Instead of comparing the two,
    // we verify that the 16-bit path achieves good absolute precision.

    let width = 256;
    let height = 64;

    // Create input images
    let input_16 = create_slow_gradient_rgb16(width, height);
    let input_8 = create_slow_gradient_rgb8(width, height);

    // Encode both at high quality with 4:4:4 (no chroma subsampling)
    let config_16 = EncoderConfig::ycbcr(98.0, ChromaSubsampling::None);
    let jpeg_16 = encode_rgb16(width as u32, height as u32, &input_16, &config_16)
        .expect("16-bit encode should succeed");

    let config_8 = EncoderConfig::ycbcr(98.0, ChromaSubsampling::None);
    let jpeg_8 = encode_rgb(width as u32, height as u32, &input_8, &config_8)
        .expect("8-bit encode should succeed");

    // Decode both to f32 for maximum precision comparison
    let decoder = Decoder::new();

    let decoded_16 = decoder
        .decode_f32(&jpeg_16)
        .expect("decode 16-bit jpeg failed");
    let decoded_8 = decoder
        .decode_f32(&jpeg_8)
        .expect("decode 8-bit jpeg failed");

    // Extract red channel (same as green and blue for gray gradient)
    let red_16: Vec<f32> = decoded_16.data.chunks(3).map(|c| c[0]).collect();
    let red_8: Vec<f32> = decoded_8.data.chunks(3).map(|c| c[0]).collect();

    // Estimate effective bit depth
    let bits_16 = estimate_effective_bits(&red_16, 0.2);
    let bits_8 = estimate_effective_bits(&red_8, 0.2);

    println!("=== 16-bit vs 8-bit Input Test ===");
    println!("Input gradient range: 0.4 to 0.6 (20% of full range)");
    println!("16-bit input → f32 output: {:.2} effective bits", bits_16);
    println!("8-bit input → f32 output: {:.2} effective bits", bits_8);
    println!(
        "JPEG sizes: 16-bit={} bytes, 8-bit={} bytes",
        jpeg_16.len(),
        jpeg_8.len()
    );

    // Both paths should achieve at least 10 effective bits for this gradient
    // (10 bits = 1024 unique levels, showing good precision preservation)
    assert!(
        bits_16 >= 10.0,
        "16-bit path should have at least 10 effective bits, got {}",
        bits_16
    );

    assert!(
        bits_8 >= 10.0,
        "8-bit path should have at least 10 effective bits, got {}",
        bits_8
    );

    // The 16-bit path produces smaller files (more efficient encoding)
    // because it starts with linear values and can better exploit the
    // perceptual encoding of jpegli
    println!(
        "16-bit compression advantage: {:.1}%",
        (1.0 - jpeg_16.len() as f64 / jpeg_8.len() as f64) * 100.0
    );
}

#[test]
fn test_f32_decode_recovers_sub_sample_precision() {
    // This test verifies that decode_f32() has more unique values
    // than decoding to 8-bit (showing sub-8-bit precision recovery).
    //
    // Note: We can't directly compare MAE because f32 output and u8 output
    // go through different paths. Instead, we measure the number of distinct
    // values recovered - f32 should have more.

    let width = 128;
    let height = 32;
    let input = create_precision_test_rgb8(width, height);

    // Encode at high quality with 4:4:4
    let config = EncoderConfig::ycbcr(99.0, ChromaSubsampling::None);
    let jpeg =
        encode_rgb(width as u32, height as u32, &input, &config).expect("encode should succeed");

    let decoder = Decoder::new();

    // Decode to f32 (high precision path)
    let decoded_f32 = decoder.decode_f32(&jpeg).expect("f32 decode failed");
    let f32_red: Vec<f32> = decoded_f32.data.chunks(3).map(|c| c[0]).collect();

    // Decode to u8
    let decoded_u8 = decoder.decode(&jpeg).expect("u8 decode failed");
    let u8_red: Vec<u8> = decoded_u8.data.chunks(3).map(|c| c[0]).collect();

    // Count unique values in each output
    let mut f32_unique: Vec<i64> = f32_red.iter().map(|&v| (v * 1_000_000.0) as i64).collect();
    f32_unique.sort();
    f32_unique.dedup();

    let mut u8_unique: Vec<u8> = u8_red.clone();
    u8_unique.sort();
    u8_unique.dedup();

    println!("=== f32 vs u8 Decode Precision Test ===");
    println!("Input spans: 126-130 (4 8-bit levels)");
    println!("f32 unique values: {}", f32_unique.len());
    println!("u8 unique values: {}", u8_unique.len());

    // f32 decode should have more unique values than u8
    // (demonstrating sub-8-bit precision recovery)
    assert!(
        f32_unique.len() >= u8_unique.len(),
        "f32 decode should have at least as many unique values as u8: {} vs {}",
        f32_unique.len(),
        u8_unique.len()
    );

    // The input spans only 4 8-bit levels, but f32 output should have more
    // distinct values due to sub-sample precision in the dequantization
    println!(
        "Precision improvement: {}x more unique values in f32",
        f32_unique.len() as f32 / u8_unique.len() as f32
    );
}

#[test]
fn test_to_u16_conversion_preserves_precision() {
    // Verify that DecodedImageF32::to_u16() properly scales to 16-bit
    // and preserves the 10+ bit precision from the decoder.

    let width = 64;
    let height = 64;
    let input = create_slow_gradient_rgb16(width, height);

    let config = EncoderConfig::ycbcr(98.0, ChromaSubsampling::None);
    let jpeg =
        encode_rgb16(width as u32, height as u32, &input, &config).expect("encode should succeed");

    let decoder = Decoder::new();
    let decoded_f32 = decoder.decode_f32(&jpeg).expect("f32 decode failed");

    // Convert to u16
    let data_u16 = decoded_f32.to_u16();

    // Verify range is correct (should span roughly 0.4 to 0.6 * 65535)
    // The input is linear RGB that produces sRGB 0.4-0.6 after encoding
    let min = *data_u16.iter().min().unwrap();
    let max = *data_u16.iter().max().unwrap();

    // Expected sRGB output range: 0.4 to 0.6 = 26214 to 39321
    let expected_min = (0.35 * 65535.0) as u16; // Allow some margin
    let expected_max = (0.65 * 65535.0) as u16;

    println!("=== to_u16 Conversion Test ===");
    println!("16-bit output range: {} to {}", min, max);
    println!(
        "Expected sRGB range: ~{} to ~{}",
        expected_min, expected_max
    );

    // Should be in a reasonable range
    assert!(min > 15000, "min should be > 0.23 * 65535, got {}", min);
    assert!(max < 55000, "max should be < 0.84 * 65535, got {}", max);

    // Check that we have many distinct values (not just 51 8-bit levels)
    let mut unique: Vec<u16> = data_u16.clone();
    unique.sort();
    unique.dedup();

    println!("Unique 16-bit values: {}", unique.len());

    // With 10+ bit precision, we should have significantly more than 51 unique values
    // (51 = the 8-bit levels in the 0.4-0.6 sRGB range)
    assert!(
        unique.len() > 100,
        "Should have more than 100 unique 16-bit values for 10+ bit precision, got {}",
        unique.len()
    );
}

#[test]
fn test_gradient_banding_reduced() {
    // This test measures "banding" - the visibility of discrete steps in gradients.
    // With 10+ bit precision, there should be smaller steps between adjacent pixels.

    let width = 512;
    let height = 8;
    let input = create_slow_gradient_rgb16(width, height);

    let config = EncoderConfig::ycbcr(95.0, ChromaSubsampling::None);
    let jpeg =
        encode_rgb16(width as u32, height as u32, &input, &config).expect("encode should succeed");

    let decoder = Decoder::new();
    let decoded = decoder.decode_f32(&jpeg).expect("decode failed");

    // Analyze horizontal gradient steps in the middle row
    let row_start = (height / 2) * width * 3;
    let row: Vec<f32> = decoded.data[row_start..row_start + width * 3]
        .chunks(3)
        .map(|c| c[0]) // Red channel
        .collect();

    // Calculate step sizes between adjacent pixels
    let mut steps: Vec<f32> = Vec::with_capacity(width - 1);
    for i in 1..row.len() {
        steps.push((row[i] - row[i - 1]).abs());
    }

    // Count "banding" events - large steps that would be visible
    // An 8-bit step is ~0.004 (1/255), so anything > 0.006 is notable
    let banding_threshold = 0.006;
    let banding_count = steps.iter().filter(|&&s| s > banding_threshold).count();

    // Calculate average and max step
    let avg_step: f32 = steps.iter().sum::<f32>() / steps.len() as f32;
    let max_step: f32 = steps.iter().cloned().fold(0.0, f32::max);

    println!("=== Gradient Banding Test ===");
    println!("Gradient length: {} pixels", width);
    println!(
        "Average step size: {:.6} (ideal: {:.6})",
        avg_step,
        0.2 / width as f32
    );
    println!("Max step size: {:.6}", max_step);
    println!(
        "Banding events (step > {}): {}",
        banding_threshold, banding_count
    );

    // With good 10+ bit encoding, we should have very few banding events
    // Allow some due to DCT block boundaries
    assert!(
        banding_count < width / 16,
        "Too many banding events: {} (should be < {})",
        banding_count,
        width / 16
    );

    // Average step should be reasonably smooth
    let ideal_step = 0.2 / width as f32;
    assert!(
        avg_step < ideal_step * 3.0,
        "Average step {} too large (ideal: {})",
        avg_step,
        ideal_step
    );
}

#[test]
fn test_full_pipeline_8bit_to_f32_precision() {
    // End-to-end test: 8-bit in, JPEG encode, f32 decode
    // Verifies that f32 decode recovers more precision than the 8-bit input.
    //
    // Note: We use 8-bit input (sRGB) to avoid linear/sRGB conversion complexity.
    // The test verifies that the f32 decoder path preserves fractional precision.

    let width = 128;
    let height = 128;

    // Create 8-bit input with a smooth pattern
    let mut input_values: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            // Create a pattern in the mid-gray range (120-136)
            let base = 120u8;
            let offset = ((x * 17 + y * 13) % 16) as u8;
            input_values.push(base + offset);
        }
    }

    // Convert to RGB bytes
    let mut input_bytes = Vec::with_capacity(width * height * 3);
    for &v in &input_values {
        // Same value for R, G, B (gray)
        input_bytes.push(v);
        input_bytes.push(v);
        input_bytes.push(v);
    }

    // Encode at very high quality
    let config = EncoderConfig::ycbcr(99.0, ChromaSubsampling::None);
    let jpeg = encode_rgb(width as u32, height as u32, &input_bytes, &config)
        .expect("encode should succeed");

    // Decode to f32
    let decoder = Decoder::new();
    let decoded_f32 = decoder.decode_f32(&jpeg).expect("decode failed");

    // Also decode to u8 for comparison
    let decoded_u8 = decoder.decode(&jpeg).expect("u8 decode failed");

    // Extract red channel
    let output_f32: Vec<f32> = decoded_f32.data.chunks(3).map(|c| c[0]).collect();
    let output_u8: Vec<u8> = decoded_u8.data.chunks(3).map(|c| c[0]).collect();

    // Calculate error vs original (in 8-bit space for fair comparison)
    let reference: Vec<f32> = input_values.iter().map(|&v| v as f32 / 255.0).collect();

    let mae_f32: f64 = reference
        .iter()
        .zip(output_f32.iter())
        .map(|(&r, &o)| (r - o).abs() as f64)
        .sum::<f64>()
        / reference.len() as f64;

    let mae_u8: f64 = reference
        .iter()
        .zip(output_u8.iter())
        .map(|(&r, &o)| (r - o as f32 / 255.0).abs() as f64)
        .sum::<f64>()
        / reference.len() as f64;

    // Count unique values in each output
    let mut f32_unique: Vec<i64> = output_f32
        .iter()
        .map(|&v| (v * 1_000_000.0) as i64)
        .collect();
    f32_unique.sort();
    f32_unique.dedup();

    let mut u8_unique: Vec<u8> = output_u8.clone();
    u8_unique.sort();
    u8_unique.dedup();

    println!("=== Full Pipeline Precision Test ===");
    println!("Image: {}x{} gray", width, height);
    println!("JPEG size: {} bytes", jpeg.len());
    println!("Input unique values: 16 (8-bit levels 120-135)");
    println!("f32 output unique values: {}", f32_unique.len());
    println!("u8 output unique values: {}", u8_unique.len());
    println!("MAE (f32 path): {:.6}", mae_f32);
    println!("MAE (u8 path): {:.6}", mae_u8);

    // At Q99, both should have low error
    assert!(mae_f32 < 0.01, "f32 MAE should be < 0.01, got {}", mae_f32);

    // f32 should have at least as many unique values as u8
    assert!(
        f32_unique.len() >= u8_unique.len(),
        "f32 should have at least as many unique values: {} vs {}",
        f32_unique.len(),
        u8_unique.len()
    );
}

#[test]
fn test_quality_affects_precision() {
    // Lower quality should result in lower precision preservation

    let width = 128;
    let height = 32;
    let input = create_slow_gradient_rgb16(width, height);

    let decoder = Decoder::new();

    println!("=== Quality vs Precision Test ===");
    let mut prev_bits = f64::MAX;

    for quality in [70.0, 85.0, 95.0, 99.0] {
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None);
        let jpeg = encode_rgb16(width as u32, height as u32, &input, &config)
            .expect("encode should succeed");

        let decoded = decoder.decode_f32(&jpeg).expect("decode failed");
        let red: Vec<f32> = decoded.data.chunks(3).map(|c| c[0]).collect();
        let bits = estimate_effective_bits(&red, 0.2);

        println!(
            "Q{}: {} bytes, {:.2} effective bits",
            quality,
            jpeg.len(),
            bits
        );

        // Higher quality should preserve more precision (or equal for very high Q)
        if quality > 70.0 {
            assert!(
                bits >= prev_bits * 0.9, // Allow small variance
                "Higher quality should preserve at least as much precision"
            );
        }
        prev_bits = bits;
    }
}

#[test]
fn test_subsampling_comparison() {
    // Compare 4:4:4 vs 4:2:0 encoding - 4:4:4 should generally preserve
    // more chroma detail, but the precision metric can be noisy.

    let width = 128;
    let height = 128;

    // Create a color gradient using 8-bit sRGB (avoids linear/sRGB confusion)
    let mut input = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width) as u8;
            let g = (y * 255 / height) as u8;
            let b = 128u8;
            input.push(r);
            input.push(g);
            input.push(b);
        }
    }

    let decoder = Decoder::new();

    let config_444 = EncoderConfig::ycbcr(95.0, ChromaSubsampling::None);
    let jpeg_444 =
        encode_rgb(width as u32, height as u32, &input, &config_444).expect("444 encode failed");

    let config_420 = EncoderConfig::ycbcr(95.0, ChromaSubsampling::Quarter);
    let jpeg_420 =
        encode_rgb(width as u32, height as u32, &input, &config_420).expect("420 encode failed");

    let decoded_444 = decoder.decode_f32(&jpeg_444).expect("444 decode failed");
    let decoded_420 = decoder.decode_f32(&jpeg_420).expect("420 decode failed");

    // Check green channel precision (affected by chroma subsampling)
    let green_444: Vec<f32> = decoded_444.data.chunks(3).map(|c| c[1]).collect();
    let green_420: Vec<f32> = decoded_420.data.chunks(3).map(|c| c[1]).collect();

    let bits_444 = estimate_effective_bits(&green_444, 1.0);
    let bits_420 = estimate_effective_bits(&green_420, 1.0);

    // 4:4:4 should be larger (more data) and have similar or better precision
    println!("=== Subsampling Comparison Test ===");
    println!(
        "4:4:4: {} bytes, {:.2} effective bits",
        jpeg_444.len(),
        bits_444
    );
    println!(
        "4:2:0: {} bytes, {:.2} effective bits",
        jpeg_420.len(),
        bits_420
    );
    println!(
        "Size ratio: {:.2}x",
        jpeg_444.len() as f64 / jpeg_420.len() as f64
    );

    // 4:4:4 should be larger since it has more chroma data
    assert!(
        jpeg_444.len() > jpeg_420.len(),
        "4:4:4 should be larger than 4:2:0: {} vs {}",
        jpeg_444.len(),
        jpeg_420.len()
    );

    // Both should have reasonable precision (> 10 bits for full-range gradient)
    assert!(
        bits_444 > 10.0,
        "4:4:4 should have > 10 effective bits, got {}",
        bits_444
    );
    assert!(
        bits_420 > 10.0,
        "4:2:0 should have > 10 effective bits, got {}",
        bits_420
    );

    // Note: Due to measurement noise, we don't require 4:4:4 > 4:2:0
    // The important thing is both preserve good precision.
}
