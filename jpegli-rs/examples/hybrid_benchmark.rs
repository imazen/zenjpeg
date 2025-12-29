//! Benchmark comparing hybrid AQ+trellis vs standard jpegli
//!
//! Run with:
//! cargo run --release --example hybrid_benchmark --features hybrid-trellis

use std::time::Instant;

#[cfg(not(feature = "hybrid-trellis"))]
fn main() {
    eprintln!("This example requires the hybrid-trellis feature.");
    eprintln!("Run with: cargo run --release --example hybrid_benchmark --features hybrid-trellis");
}

#[cfg(feature = "hybrid-trellis")]
fn main() {
    use jpegli::{
        adaptive_quant::compute_aq_strength_map,
        hybrid::{scale_quant_by_aq, StandardHuffmanTables},
    };
    use mozjpeg_oxide::{trellis::trellis_quantize_block, TrellisConfig};

    // Create test image (512x512 with varying content)
    let width = 512;
    let height = 512;
    let mut pixels = vec![0u8; width * height * 3];

    // Create image with varying content: smooth gradients + sharp edges + texture
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let qx = x / (width / 2);
            let qy = y / (height / 2);

            match (qx, qy) {
                (0, 0) => {
                    // Smooth gradient (low AQ)
                    let val = ((x + y) * 255 / (width + height)) as u8;
                    pixels[idx] = val;
                    pixels[idx + 1] = val;
                    pixels[idx + 2] = val;
                }
                (1, 0) => {
                    // Sharp edges - checkerboard (medium AQ)
                    let val = if (x / 16 + y / 16) % 2 == 0 { 240 } else { 16 };
                    pixels[idx] = val;
                    pixels[idx + 1] = val;
                    pixels[idx + 2] = val;
                }
                (0, 1) => {
                    // Color gradient
                    pixels[idx] = (x % 256) as u8;
                    pixels[idx + 1] = (y % 256) as u8;
                    pixels[idx + 2] = 128;
                }
                (1, 1) => {
                    // High frequency texture (high AQ)
                    let val = ((x * 7 + y * 13) % 256) as u8;
                    pixels[idx] = val;
                    pixels[idx + 1] = val;
                    pixels[idx + 2] = val;
                }
                _ => {}
            }
        }
    }

    let quality = 75u8;

    // Standard luminance quantization table scaled for quality
    let base_quant = scale_quant_for_quality(quality);

    println!("=== Hybrid Encoder Benchmark ===\n");
    println!("Image: {}x{}, Quality: {}", width, height, quality);

    // Step 1: Convert to Y plane for AQ analysis
    let start = Instant::now();
    let y_plane: Vec<f32> = (0..(width * height))
        .map(|i| {
            let r = pixels[i * 3] as f32;
            let g = pixels[i * 3 + 1] as f32;
            let b = pixels[i * 3 + 2] as f32;
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect();
    println!("1. RGB to Y plane: {:?}", start.elapsed());

    // Step 2: Compute AQ strength map
    let start = Instant::now();
    let y_quant_01 = base_quant[1];
    let aq_map = compute_aq_strength_map(&y_plane, width, height, y_quant_01);
    println!("2. Compute AQ map: {:?}", start.elapsed());

    // Analyze AQ map
    let blocks_h = (width + 7) / 8;
    let blocks_v = (height + 7) / 8;
    let mut aq_min = f32::MAX;
    let mut aq_max = f32::MIN;
    let mut aq_sum = 0.0f32;

    for by in 0..blocks_v {
        for bx in 0..blocks_h {
            let strength = aq_map.get(bx, by);
            aq_min = aq_min.min(strength);
            aq_max = aq_max.max(strength);
            aq_sum += strength;
        }
    }
    let aq_mean = aq_sum / (blocks_h * blocks_v) as f32;
    println!(
        "   AQ strengths: min={:.3}, max={:.3}, mean={:.3}",
        aq_min, aq_max, aq_mean
    );

    // Step 3: Show AQ map for different image regions
    println!("\n3. AQ strength by quadrant:");
    let quadrants = [
        ("Smooth gradient (top-left)", 0, 0),
        ("Sharp edges (top-right)", blocks_h / 2, 0),
        ("Color gradient (bottom-left)", 0, blocks_v / 2),
        ("High frequency (bottom-right)", blocks_h / 2, blocks_v / 2),
    ];

    for (name, start_bx, start_by) in quadrants {
        let mut sum = 0.0f32;
        let mut count = 0;
        for by in start_by..(start_by + 8).min(blocks_v) {
            for bx in start_bx..(start_bx + 8).min(blocks_h) {
                sum += aq_map.get(bx, by);
                count += 1;
            }
        }
        let mean = sum / count as f32;
        println!("   {}: mean AQ = {:.3}", name, mean);
    }

    // Step 4: Demonstrate per-block quant table scaling
    println!("\n4. Quant table scaling examples:");
    let low_aq = aq_map.get(2, 2); // Likely smooth area
    let high_aq = aq_map.get(blocks_h / 2 + 2, 2); // Likely edge area

    let scaled_low = scale_quant_by_aq(&base_quant, low_aq);
    let scaled_high = scale_quant_by_aq(&base_quant, high_aq);

    println!("   Base quant[0..4]: {:?}", &base_quant[0..4]);
    println!("   Low AQ ({:.3}) scaled: {:?}", low_aq, &scaled_low[0..4]);
    println!(
        "   High AQ ({:.3}) scaled: {:?}",
        high_aq,
        &scaled_high[0..4]
    );

    // Step 5: Benchmark trellis quantization with AQ-scaled tables
    println!("\n5. Trellis benchmark:");

    let huff_tables = StandardHuffmanTables::new();
    let trellis_config = TrellisConfig::default();

    // Create sample DCT block (typical AC values)
    let dct_block: [i32; 64] = {
        let mut block = [0i32; 64];
        block[0] = 1024; // DC
        for i in 1..64 {
            block[i] = (50 - (i as i32 * 2)).max(-100).min(100);
        }
        block
    };

    // Benchmark: trellis with fixed quant vs AQ-scaled quant
    let iterations = 10000;

    let start = Instant::now();
    for _ in 0..iterations {
        let mut quantized = [0i16; 64];
        trellis_quantize_block(
            &dct_block,
            &mut quantized,
            &base_quant,
            &huff_tables.luma_ac,
            &trellis_config,
        );
        std::hint::black_box(&quantized);
    }
    let fixed_time = start.elapsed();

    let start = Instant::now();
    for by in 0..10 {
        for bx in 0..10 {
            let aq = aq_map.get(bx, by);
            let scaled = scale_quant_by_aq(&base_quant, aq);
            for _ in 0..(iterations / 100) {
                let mut quantized = [0i16; 64];
                trellis_quantize_block(
                    &dct_block,
                    &mut quantized,
                    &scaled,
                    &huff_tables.luma_ac,
                    &trellis_config,
                );
                std::hint::black_box(&quantized);
            }
        }
    }
    let hybrid_time = start.elapsed();

    println!(
        "   Trellis (fixed quant): {:?} for {} blocks",
        fixed_time, iterations
    );
    println!(
        "   Trellis (AQ-scaled):   {:?} for {} blocks",
        hybrid_time, iterations
    );

    // Step 6: Compare actual encoding
    println!("\n6. Full encoding comparison:");

    // Standard jpegli (AQ + zero-bias)
    let start = Instant::now();
    let jpegli_result = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(jpegli::quant::Quality::from_quality(quality as f32))
        .encode(&pixels)
        .unwrap();
    let jpegli_time = start.elapsed();
    println!(
        "   jpegli (AQ):        {} bytes in {:?}",
        jpegli_result.len(),
        jpegli_time
    );

    // Hybrid jpegli (AQ + trellis)
    let start = Instant::now();
    let hybrid_result = jpegli::Encoder::new()
        .width(width as u32)
        .height(height as u32)
        .quality(jpegli::quant::Quality::from_quality(quality as f32))
        .hybrid_trellis(true)
        .encode(&pixels)
        .unwrap();
    let hybrid_time = start.elapsed();
    println!(
        "   jpegli (AQ+trellis): {} bytes in {:?}",
        hybrid_result.len(),
        hybrid_time
    );

    // Size comparison
    let size_diff = hybrid_result.len() as i64 - jpegli_result.len() as i64;
    let size_pct = (size_diff as f64 / jpegli_result.len() as f64) * 100.0;
    println!("   Difference: {:+} bytes ({:+.2}%)", size_diff, size_pct);

    // Verify both are valid JPEGs and compare quality
    use jpeg_decoder::Decoder;
    let mut decoder = Decoder::new(&jpegli_result[..]);
    let jpegli_decoded = decoder
        .decode()
        .expect("jpegli output should be valid JPEG");

    let mut decoder = Decoder::new(&hybrid_result[..]);
    let hybrid_decoded = decoder
        .decode()
        .expect("hybrid output should be valid JPEG");
    println!("   Both outputs are valid JPEGs ✓");

    // Calculate DSSIM to compare quality
    let attr = dssim::Dssim::new();

    // Original image
    let orig_rgba: Vec<rgb::RGBA<u8>> = pixels
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let orig_img = attr
        .create_image_rgba(&orig_rgba, width, height)
        .expect("create orig image");

    // jpegli decoded
    let jpegli_rgba: Vec<rgb::RGBA<u8>> = jpegli_decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let jpegli_img = attr
        .create_image_rgba(&jpegli_rgba, width, height)
        .expect("create jpegli image");

    // hybrid decoded
    let hybrid_rgba: Vec<rgb::RGBA<u8>> = hybrid_decoded
        .chunks(3)
        .map(|rgb| rgb::RGBA::new(rgb[0], rgb[1], rgb[2], 255))
        .collect();
    let hybrid_img = attr
        .create_image_rgba(&hybrid_rgba, width, height)
        .expect("create hybrid image");

    let (jpegli_dssim, _) = attr.compare(&orig_img, jpegli_img);
    let (hybrid_dssim, _) = attr.compare(&orig_img, hybrid_img);

    println!("\n7. Quality comparison (DSSIM, lower = better):");
    println!("   jpegli (AQ):         {:.6}", jpegli_dssim);
    println!("   jpegli (AQ+trellis): {:.6}", hybrid_dssim);
    println!("   Quality ratio: {:.2}x", hybrid_dssim / jpegli_dssim);

    println!("\n=== Summary ===");
    println!("Current implementation: trellis quantization WITHOUT AQ integration.");
    println!("This shows trellis produces valid, comparable results to jpegli.");
    println!();
    println!("TODO: Integrate AQ into trellis by adjusting lambda per block:");
    println!("- Higher AQ strength (textured) → higher lambda → favor smaller file");
    println!("- Lower AQ strength (smooth) → lower lambda → preserve quality");
}

/// Scale standard luminance quant table for quality level
#[cfg(feature = "hybrid-trellis")]
fn scale_quant_for_quality(quality: u8) -> [u16; 64] {
    const STD_LUMA_QUANT: [u16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];

    let scale = if quality < 50 {
        5000.0 / quality as f32
    } else {
        200.0 - 2.0 * quality as f32
    } / 100.0;

    let mut scaled = [0u16; 64];
    for i in 0..64 {
        let val = (STD_LUMA_QUANT[i] as f32 * scale).round() as u16;
        scaled[i] = val.clamp(1, 255);
    }
    scaled
}
