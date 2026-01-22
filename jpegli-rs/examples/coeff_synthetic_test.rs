//! Synthetic image test for coefficient comparison.
//!
//! Uses simple solid color images to isolate systematic offsets.
//!
//! IMPORTANT: Uses distance-based encoding (`jpegli_set_distance`) for both
//! encoders to ensure identical quant table configurations (3 tables).

use enough::Unstoppable;
use jpegli::decode::Decoder;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Quality};
use jpegli_bench_utils::{
    ChromaSubsampling as BenchChromaSubsampling, ColorMode, EncoderConfig as BenchEncoderConfig,
    EncoderImpl, ImageData, ScanMode,
};

fn encode_rust(pixels: &[u8], width: u32, height: u32, distance: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(
        Quality::ApproxButteraugli(distance),
        ChromaSubsampling::Quarter,
    )
    .progressive(false)
    .optimize_huffman(true);
    let mut encoder = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    encoder
        .push_packed(pixels, Unstoppable)
        .expect("push failed");
    encoder.finish().expect("finish failed")
}

fn encode_cpp_ffi(pixels: &[u8], width: u32, height: u32, distance: f32) -> Vec<u8> {
    let img = ImageData {
        name: "test".to_string(),
        pixels: pixels.to_vec(),
        width: width as usize,
        height: height as usize,
    };
    BenchEncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Baseline)
        .subsampling(BenchChromaSubsampling::S420)
        .distance(distance) // Use distance, not quality!
        .encode(&img)
        .expect("C++ jpegli FFI encode failed")
}

fn test_solid_color(r: u8, g: u8, b: u8, name: &str, distance: f32) {
    let width = 64u32;
    let height = 64u32;
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for _ in 0..(width * height) {
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
    }

    let rust_jpeg = encode_rust(&pixels, width, height, distance);
    let cpp_jpeg = encode_cpp_ffi(&pixels, width, height, distance);

    let decoder = Decoder::new();
    let rust_coeffs = decoder
        .decode_coefficients(&rust_jpeg)
        .expect("decode rust");
    let cpp_coeffs = decoder.decode_coefficients(&cpp_jpeg).expect("decode cpp");

    println!(
        "=== {} (RGB={},{},{}) distance={:.2} ===",
        name, r, g, b, distance
    );

    // Print quant tables
    println!("  Quant tables (DC position [0]):");
    for (i, (rust_qt, cpp_qt)) in rust_coeffs
        .quant_tables
        .iter()
        .zip(&cpp_coeffs.quant_tables)
        .enumerate()
    {
        if let (Some(rq), Some(cq)) = (rust_qt, cpp_qt) {
            if rq[0] != cq[0] {
                println!(
                    "    Table {}: rust={}, cpp={} (DIFFERENT!)",
                    i, rq[0], cq[0]
                );
            } else {
                println!("    Table {}: rust={}, cpp={} (same)", i, rq[0], cq[0]);
            }
        }
    }

    for comp_idx in 0..3 {
        let rc = &rust_coeffs.components[comp_idx];
        let cc = &cpp_coeffs.components[comp_idx];

        // For solid color, all blocks should have the same DC
        let rust_dc = rc.block(0)[0];
        let cpp_dc = cc.block(0)[0];
        let diff = rust_dc as i32 - cpp_dc as i32;

        let comp_name = match comp_idx {
            0 => "Y ",
            1 => "Cb",
            2 => "Cr",
            _ => "??",
        };

        // Check if all DCs are the same
        let all_same_rust = rc.coeffs.chunks(64).all(|b| b[0] == rust_dc);
        let all_same_cpp = cc.coeffs.chunks(64).all(|b| b[0] == cpp_dc);

        // Calculate ratio if both non-zero
        let ratio = if cpp_dc != 0 && rust_dc != 0 {
            format!(" ratio={:.3}", rust_dc as f64 / cpp_dc as f64)
        } else {
            String::new()
        };

        println!(
            "  {}: rust_DC={:5}, cpp_DC={:5}, diff={:+4}{} (uniform: rust={}, cpp={})",
            comp_name, rust_dc, cpp_dc, diff, ratio, all_same_rust, all_same_cpp
        );
    }
    println!();
}

fn main() {
    println!("=== Synthetic Solid Color Coefficient Test (Distance-based) ===\n");
    println!("NOTE: Using jpegli_set_distance for C++ and Quality::ApproxButteraugli for Rust.");
    println!("      Both use 3 quant tables (add_two_chroma_tables=true).\n");

    // Test at distance ~1.0 (roughly q90)
    let distance = 1.0;

    // Test common colors
    test_solid_color(128, 128, 128, "Gray", distance); // Neutral gray
    test_solid_color(0, 0, 0, "Black", distance); // Pure black
    test_solid_color(255, 255, 255, "White", distance); // Pure white
    test_solid_color(255, 0, 0, "Red", distance); // Pure red
    test_solid_color(0, 255, 0, "Green", distance); // Pure green
    test_solid_color(0, 0, 255, "Blue", distance); // Pure blue
    test_solid_color(255, 255, 0, "Yellow", distance); // Yellow
    test_solid_color(0, 255, 255, "Cyan", distance); // Cyan

    // Test at different distance levels
    println!("=== Distance Level Comparison (Gray) ===\n");
    for d in [0.5, 1.0, 2.0, 4.0] {
        test_solid_color(128, 128, 128, "Gray", d);
    }
}
