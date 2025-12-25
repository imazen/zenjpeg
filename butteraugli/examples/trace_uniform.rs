//! Trace butteraugli values for uniform images to debug score discrepancy
//!
//! Run with: cargo run --example trace_uniform

use butteraugli_oxide::opsin::{gamma, opsin_absorbance, srgb_to_xyb_butteraugli};
use butteraugli_oxide::{compute_butteraugli, ButteraugliParams, ImageF, PsychoImage};

/// sRGB transfer function (gamma decoding)
fn srgb_to_linear(v: u8) -> f32 {
    let v = v as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn main() {
    println!("=== Butteraugli Uniform Image Debug ===\n");

    // Test basic gamma function
    println!("--- Gamma function test ---");
    let test_vals = [1.0, 10.0, 20.0, 50.0, 100.0];
    for v in test_vals {
        let g = gamma(v);
        println!("Gamma({:.1}) = {:.6}", v, g);
    }

    // Test OpsinAbsorbance
    println!("\n--- OpsinAbsorbance test ---");
    let gray128_linear = srgb_to_linear(128);
    let gray138_linear = srgb_to_linear(138);
    println!("Gray 128: sRGB->linear = {:.6}", gray128_linear);
    println!("Gray 138: sRGB->linear = {:.6}", gray138_linear);

    let intensity_target = 80.0;
    let r128 = gray128_linear * intensity_target;
    let r138 = gray138_linear * intensity_target;
    println!("Gray 128 * intensity_target: {:.6}", r128);
    println!("Gray 138 * intensity_target: {:.6}", r138);

    let (o0_128, o1_128, o2_128) = opsin_absorbance(r128, r128, r128, true);
    let (o0_138, o1_138, o2_138) = opsin_absorbance(r138, r138, r138, true);
    println!(
        "OpsinAbsorbance(gray128): [{:.6}, {:.6}, {:.6}]",
        o0_128, o1_128, o2_128
    );
    println!(
        "OpsinAbsorbance(gray138): [{:.6}, {:.6}, {:.6}]",
        o0_138, o1_138, o2_138
    );

    // Compute sensitivity
    let min_val = 1e-4_f32;
    let sens0_128 = (gamma(o0_128) / o0_128).max(min_val);
    let sens1_128 = (gamma(o1_128) / o1_128).max(min_val);
    let sens2_128 = (gamma(o2_128) / o2_128).max(min_val);
    println!(
        "Sensitivity(gray128): [{:.6}, {:.6}, {:.6}]",
        sens0_128, sens1_128, sens2_128
    );

    // Apply sensitivity and convert to XYB
    let cur0_128 = o0_128 * sens0_128;
    let cur1_128 = o1_128 * sens1_128;
    let cur2_128 = o2_128 * sens2_128;
    let x_128 = cur0_128 - cur1_128;
    let y_128 = cur0_128 + cur1_128;
    let b_128 = cur2_128;
    println!(
        "XYB(gray128): X={:.6}, Y={:.6}, B={:.6}",
        x_128, y_128, b_128
    );

    let sens0_138 = (gamma(o0_138) / o0_138).max(min_val);
    let sens1_138 = (gamma(o1_138) / o1_138).max(min_val);
    let sens2_138 = (gamma(o2_138) / o2_138).max(min_val);
    let cur0_138 = o0_138 * sens0_138;
    let cur1_138 = o1_138 * sens1_138;
    let cur2_138 = o2_138 * sens2_138;
    let x_138 = cur0_138 - cur1_138;
    let y_138 = cur0_138 + cur1_138;
    let b_138 = cur2_138;
    println!(
        "XYB(gray138): X={:.6}, Y={:.6}, B={:.6}",
        x_138, y_138, b_138
    );

    println!(
        "\nXYB difference: dX={:.6}, dY={:.6}, dB={:.6}",
        x_138 - x_128,
        y_138 - y_128,
        b_138 - b_128
    );

    // Test actual XYB conversion
    println!("\n--- Full XYB conversion test (single pixel) ---");
    let width = 4;
    let height = 4;
    let rgb128: Vec<u8> = vec![128; width * height * 3];
    let rgb138: Vec<u8> = vec![138; width * height * 3];

    let xyb128 = srgb_to_xyb_butteraugli(&rgb128, width, height, intensity_target);
    let xyb138 = srgb_to_xyb_butteraugli(&rgb138, width, height, intensity_target);

    println!(
        "srgb_to_xyb_butteraugli(gray128)[0,0]: X={:.6}, Y={:.6}, B={:.6}",
        xyb128.plane(0).get(0, 0),
        xyb128.plane(1).get(0, 0),
        xyb128.plane(2).get(0, 0)
    );
    println!(
        "srgb_to_xyb_butteraugli(gray138)[0,0]: X={:.6}, Y={:.6}, B={:.6}",
        xyb138.plane(0).get(0, 0),
        xyb138.plane(1).get(0, 0),
        xyb138.plane(2).get(0, 0)
    );

    // Full butteraugli computation
    println!("\n--- Full butteraugli computation ---");
    let params = ButteraugliParams::default();

    // 64x64 uniform gray images
    let width = 64;
    let height = 64;
    let rgb1: Vec<u8> = vec![128; width * height * 3];
    let rgb2: Vec<u8> = vec![138; width * height * 3];

    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params).expect("valid input");
    println!("Uniform 128 vs 138 (64x64): score = {:.6}", result.score);

    if let Some(dm) = result.diffmap.as_ref() {
        let sum: f64 = (0..dm.height())
            .flat_map(|y| (0..dm.width()).map(move |x| dm.get(x, y) as f64))
            .sum();
        let mean = sum / (dm.width() * dm.height()) as f64;
        let max = (0..dm.height())
            .flat_map(|y| (0..dm.width()).map(move |x| dm.get(x, y)))
            .fold(0.0f32, f32::max);
        println!("  diffmap mean = {:.6}", mean);
        println!("  diffmap max = {:.6}", max);
        println!("  diffmap[32,32] = {:.6}", dm.get(32, 32));
    }

    // Test MaskY and MaskDcY
    println!("\n--- MaskY/MaskDcY test ---");
    // Constants from consts.rs
    const GLOBAL_SCALE: f32 = 1.0 / (17.83 * 0.790799174);
    const MASK_Y_OFFSET: f64 = 0.829591754942;
    const MASK_Y_SCALER: f64 = 0.451936922203;
    const MASK_Y_MUL: f64 = 2.5485944793;
    const MASK_DC_Y_OFFSET: f64 = 0.20025578522;
    const MASK_DC_Y_SCALER: f64 = 3.87449418804;
    const MASK_DC_Y_MUL: f64 = 0.505054525019;

    fn mask_y(delta: f64) -> f64 {
        let c = MASK_Y_MUL / (MASK_Y_SCALER * delta + MASK_Y_OFFSET);
        let retval = GLOBAL_SCALE as f64 * (1.0 + c);
        retval * retval
    }

    fn mask_dc_y(delta: f64) -> f64 {
        let c = MASK_DC_Y_MUL / (MASK_DC_Y_SCALER * delta + MASK_DC_Y_OFFSET);
        let retval = GLOBAL_SCALE as f64 * (1.0 + c);
        retval * retval
    }

    for mask_val in [0.0, 0.5, 1.0, 2.0, 5.0] {
        println!(
            "MaskY({:.1}) = {:.6}, MaskDcY({:.1}) = {:.6}",
            mask_val,
            mask_y(mask_val),
            mask_val,
            mask_dc_y(mask_val)
        );
    }

    // Compare different gray levels
    println!("\n--- Gray level comparison ---");
    for diff in [1u8, 2, 5, 10, 20, 50] {
        let rgb1: Vec<u8> = vec![128; width * height * 3];
        let rgb2: Vec<u8> = vec![128 + diff; width * height * 3];
        let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params).expect("valid input");
        println!("Gray 128 vs {}: score = {:.4}", 128 + diff, result.score);
    }

    // Expected C++ values (approximate, for reference)
    println!("\n--- Expected C++ reference (approximate) ---");
    println!("For Q90 JPEG (flower_small.png): ~1.8");
    println!("For uniform gray difference of 10: should be < 1.0 for imperceptible");
}
