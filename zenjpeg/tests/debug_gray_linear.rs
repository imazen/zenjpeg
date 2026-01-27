//! Debug test for Gray16 linear encoding issue

use std::collections::HashSet;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

fn encode(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
    layout: PixelLayout,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, layout)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn srgb_to_linear(s: f64) -> f64 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn debug_gray_linear_encoding() {
    let width = 64;
    let height = 64;
    let n_pixels = width * height;

    // Create Gray8 (sRGB) - full range 0-255
    let gray8: Vec<u8> = (0..n_pixels).map(|i| (i % 256) as u8).collect();
    let unique_input_8: HashSet<u8> = gray8.iter().cloned().collect();

    // Create Gray16 (linear) - convert sRGB to linear
    let gray16_bytes: Vec<u8> = gray8
        .iter()
        .flat_map(|&v| {
            let srgb = v as f64 / 255.0;
            let linear = srgb_to_linear(srgb);
            let v16 = (linear * 65535.0) as u16;
            v16.to_ne_bytes()
        })
        .collect();

    // Check linear values distribution
    let linear_vals: Vec<u16> = gray16_bytes
        .chunks(2)
        .map(|b| u16::from_ne_bytes([b[0], b[1]]))
        .collect();
    let unique_linear: HashSet<u16> = linear_vals.iter().cloned().collect();

    println!("=== INPUT ANALYSIS ===");
    println!("Gray8 (sRGB) input:");
    println!(
        "  Range: {} to {}",
        gray8.iter().min().unwrap(),
        gray8.iter().max().unwrap()
    );
    println!("  Unique values: {}", unique_input_8.len());

    println!("\nGray16 (linear) input:");
    println!(
        "  Range: {} to {} (in u16)",
        linear_vals.iter().min().unwrap(),
        linear_vals.iter().max().unwrap()
    );
    println!("  Unique values: {}", unique_linear.len());

    // Show some example mappings
    println!("\n  sRGB → Linear mapping examples:");
    for srgb in [0, 64, 128, 192, 255] {
        let linear = srgb_to_linear(srgb as f64 / 255.0);
        println!(
            "    sRGB {} → linear {:.4} → u16 {}",
            srgb,
            linear,
            (linear * 65535.0) as u16
        );
    }

    // Encode Gray8
    let config = EncoderConfig::grayscale(98.0);
    let jpeg8 = encode(
        width as u32,
        height as u32,
        &gray8,
        &config,
        PixelLayout::Gray8Srgb,
    )
    .expect("Gray8 encode failed");

    // Encode Gray16
    let jpeg16 = encode(
        width as u32,
        height as u32,
        &gray16_bytes,
        &config,
        PixelLayout::Gray16Linear,
    )
    .expect("Gray16 encode failed");

    println!("\n=== ENCODED JPEG ===");
    println!("Gray8 JPEG size: {} bytes", jpeg8.len());
    println!("Gray16 JPEG size: {} bytes", jpeg16.len());

    // Decode both
    let decoder = Decoder::new();
    let dec8 = decoder.decode_f32(&jpeg8).expect("Gray8 decode failed");
    let dec16 = decoder.decode_f32(&jpeg16).expect("Gray16 decode failed");

    // Extract just the red channel (grayscale outputs RGB with R=G=B)
    let vals8: Vec<f32> = dec8.data.iter().step_by(3).cloned().collect();
    let vals16: Vec<f32> = dec16.data.iter().step_by(3).cloned().collect();

    println!("\n=== DECODED OUTPUT ===");
    println!("Gray8 decoded (f32):");
    println!(
        "  Range: {:.4} to {:.4}",
        vals8.iter().cloned().fold(f32::INFINITY, f32::min),
        vals8.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    println!("\nGray16 decoded (f32):");
    println!(
        "  Range: {:.4} to {:.4}",
        vals16.iter().cloned().fold(f32::INFINITY, f32::min),
        vals16.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Count unique values at different precisions
    println!("\n=== UNIQUE VALUES BY PRECISION ===");
    println!("{:>8} {:>12} {:>12}", "Bits", "Gray8", "Gray16");
    println!("{:->8} {:->12} {:->12}", "", "", "");

    for bits in [8, 10, 12, 14, 16] {
        let scale = (1u64 << bits) as f32;
        let unique8: HashSet<i64> = vals8.iter().map(|&v| (v * scale) as i64).collect();
        let unique16: HashSet<i64> = vals16.iter().map(|&v| (v * scale) as i64).collect();
        println!("{:>8} {:>12} {:>12}", bits, unique8.len(), unique16.len());
    }

    // Compare the actual output values for the same input
    println!("\n=== SAMPLE OUTPUT COMPARISON ===");
    println!("For input sRGB values, what does each encoder output?");
    println!("{:>8} {:>12} {:>12}", "Input", "Gray8 out", "Gray16 out");
    println!("{:->8} {:->12} {:->12}", "", "", "");

    // Find pixels with specific input values
    for target_srgb in [0u8, 64, 128, 192, 255] {
        // Find a pixel with this value in gray8
        if let Some(idx) = gray8.iter().position(|&v| v == target_srgb) {
            let out8 = vals8[idx];
            let out16 = vals16[idx];
            println!("{:>8} {:>12.4} {:>12.4}", target_srgb, out8, out16);
        }
    }

    // The key question: is Gray16 linear being converted correctly?
    // If the encoder properly converts linear→sRGB, output ranges should be similar
    let range8 = vals8.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - vals8.iter().cloned().fold(f32::INFINITY, f32::min);
    let range16 = vals16.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - vals16.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("\n=== DIAGNOSIS ===");
    println!("Gray8 output range span: {:.4}", range8);
    println!("Gray16 output range span: {:.4}", range16);
    println!("Ratio (Gray16/Gray8): {:.2}x", range16 / range8);

    if range16 < range8 * 0.5 {
        println!("\n⚠️  Gray16 has significantly smaller output range!");
        println!("   This suggests linear→sRGB conversion may not be happening.");
    } else {
        println!("\n✓ Output ranges are similar - conversion looks correct.");
    }
}
