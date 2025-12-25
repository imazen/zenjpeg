//! Detailed tracing of butteraugli computation for uniform images
//!
//! Run with: cargo run --example trace_detailed

use butteraugli_oxide::opsin::srgb_to_xyb_butteraugli;
use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};

mod consts {
    pub const SIGMA_LF: f32 = 7.15593339443;
    pub const SIGMA_HF: f32 = 3.22489901262;
    pub const SIGMA_UHF: f32 = 1.56416327805;
    pub const XMUL_LF_TO_VALS: f32 = 33.832837186260;
    pub const YMUL_LF_TO_VALS: f32 = 14.458268100570;
    pub const BMUL_LF_TO_VALS: f32 = 49.87984651440;
    pub const Y_TO_B_MUL_LF_TO_VALS: f32 = -0.362267051518;
    pub const WMUL: [f64; 9] = [
        400.0,
        1.50815703118,
        0.0,
        2150.0,
        10.6195433239,
        16.2176043152,
        29.2353797994,
        0.844626970982,
        0.703646627719,
    ];
    pub const GLOBAL_SCALE: f32 = 1.0 / (17.83 * 0.790799174);
    pub const MASK_DC_Y_OFFSET: f64 = 0.20025578522;
    pub const MASK_DC_Y_SCALER: f64 = 3.87449418804;
    pub const MASK_DC_Y_MUL: f64 = 0.505054525019;
}

use consts::*;

fn mask_dc_y(delta: f64) -> f64 {
    let c = MASK_DC_Y_MUL / (MASK_DC_Y_SCALER * delta + MASK_DC_Y_OFFSET);
    let retval = GLOBAL_SCALE as f64 * (1.0 + c);
    retval * retval
}

fn main() {
    println!("=== Detailed Butteraugli Trace ===\n");

    let width = 16;
    let height = 16;
    let intensity_target = 80.0;

    // Test various gray differences
    for diff in [1u8, 2, 5, 10, 20] {
        let gray1 = 128u8;
        let gray2 = 128u8.saturating_add(diff);

        println!("\n--- Gray {} vs {} (diff={}) ---", gray1, gray2, diff);

        // Create uniform images
        let rgb1: Vec<u8> = vec![gray1; width * height * 3];
        let rgb2: Vec<u8> = vec![gray2; width * height * 3];

        // Convert to XYB
        let xyb1 = srgb_to_xyb_butteraugli(&rgb1, width, height, intensity_target);
        let xyb2 = srgb_to_xyb_butteraugli(&rgb2, width, height, intensity_target);

        // Get center pixel values (should be same as any pixel for uniform)
        let cx = width / 2;
        let cy = height / 2;

        let x1 = xyb1.plane(0).get(cx, cy);
        let y1 = xyb1.plane(1).get(cx, cy);
        let b1 = xyb1.plane(2).get(cx, cy);

        let x2 = xyb2.plane(0).get(cx, cy);
        let y2 = xyb2.plane(1).get(cx, cy);
        let b2 = xyb2.plane(2).get(cx, cy);

        println!("XYB1: X={:.4}, Y={:.4}, B={:.4}", x1, y1, b1);
        println!("XYB2: X={:.4}, Y={:.4}, B={:.4}", x2, y2, b2);

        let dx = x2 - x1;
        let dy = y2 - y1;
        let db = b2 - b1;
        println!("dXYB: dX={:.6}, dY={:.6}, dB={:.6}", dx, dy, db);

        // For uniform images, LF = XYB, so after xyb_low_freq_to_vals:
        // val_x = x * XMUL_LF_TO_VALS
        // val_y = y * YMUL_LF_TO_VALS
        // val_b = (Y_TO_B_MUL * y + b) * BMUL_LF_TO_VALS

        let val_x1 = x1 * XMUL_LF_TO_VALS;
        let val_y1 = y1 * YMUL_LF_TO_VALS;
        let val_b1 = (Y_TO_B_MUL_LF_TO_VALS * y1 + b1) * BMUL_LF_TO_VALS;

        let val_x2 = x2 * XMUL_LF_TO_VALS;
        let val_y2 = y2 * YMUL_LF_TO_VALS;
        let val_b2 = (Y_TO_B_MUL_LF_TO_VALS * y2 + b2) * BMUL_LF_TO_VALS;

        println!(
            "LF Vals1: x={:.4}, y={:.4}, b={:.4}",
            val_x1, val_y1, val_b1
        );
        println!(
            "LF Vals2: x={:.4}, y={:.4}, b={:.4}",
            val_x2, val_y2, val_b2
        );

        let d_val_x = val_x2 - val_x1;
        let d_val_y = val_y2 - val_y1;
        let d_val_b = val_b2 - val_b1;
        println!(
            "dVals: dX={:.6}, dY={:.6}, dB={:.6}",
            d_val_x, d_val_y, d_val_b
        );

        // block_diff_dc = sum of (diff^2 * WMUL[6+c])
        let dc_x = d_val_x * d_val_x * WMUL[6] as f32;
        let dc_y = d_val_y * d_val_y * WMUL[7] as f32;
        let dc_b = d_val_b * d_val_b * WMUL[8] as f32;
        let dc_total = dc_x + dc_y + dc_b;
        println!(
            "DC diff: X={:.4}, Y={:.4}, B={:.4}, total={:.4}",
            dc_x, dc_y, dc_b, dc_total
        );

        // For uniform images, mask should be 0 (no HF/UHF content)
        let mask = 0.0;
        let dc_mask_val = mask_dc_y(mask);
        println!("Mask=0, MaskDcY(0)={:.6}", dc_mask_val);

        // Expected diffmap (if block_diff_ac = 0)
        let expected_diffmap = (dc_total as f64 * dc_mask_val).sqrt();
        println!("Expected diffmap (DC only): {:.6}", expected_diffmap);

        // Actual butteraugli score
        let params = ButteraugliParams::default();
        let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
        println!("Actual butteraugli score: {:.6}", result.score);

        if let Some(dm) = result.diffmap.as_ref() {
            // Check if diffmap is uniform
            let min = (0..height)
                .flat_map(|y| (0..width).map(move |x| dm.get(x, y)))
                .fold(f32::MAX, f32::min);
            let max = (0..height)
                .flat_map(|y| (0..width).map(move |x| dm.get(x, y)))
                .fold(f32::MIN, f32::max);
            println!("Diffmap min={:.6}, max={:.6}", min, max);

            // If not uniform, check corners
            if (max - min).abs() > 0.01 {
                println!(
                    "Diffmap corners: [0,0]={:.4}, [w-1,0]={:.4}, [0,h-1]={:.4}, [w-1,h-1]={:.4}",
                    dm.get(0, 0),
                    dm.get(width - 1, 0),
                    dm.get(0, height - 1),
                    dm.get(width - 1, height - 1)
                );
            }
        }
    }

    // Now test with a real image pattern to see if frequency content changes things
    println!("\n\n=== Test with frequency content ===");

    // Create a gradient image
    let mut rgb_grad: Vec<u8> = vec![0; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let v = (128 + (x as i32 - 8) * 4).clamp(0, 255) as u8;
            let idx = (y * width + x) * 3;
            rgb_grad[idx] = v;
            rgb_grad[idx + 1] = v;
            rgb_grad[idx + 2] = v;
        }
    }

    // Slightly modified gradient
    let mut rgb_grad2: Vec<u8> = rgb_grad.clone();
    for i in 0..rgb_grad2.len() {
        rgb_grad2[i] = rgb_grad2[i].saturating_add(5);
    }

    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb_grad, &rgb_grad2, width, height, &params);
    println!("Gradient +5: score = {:.6}", result.score);

    if let Some(dm) = result.diffmap.as_ref() {
        let min = (0..height)
            .flat_map(|y| (0..width).map(move |x| dm.get(x, y)))
            .fold(f32::MAX, f32::min);
        let max = (0..height)
            .flat_map(|y| (0..width).map(move |x| dm.get(x, y)))
            .fold(f32::MIN, f32::max);
        let center = dm.get(width / 2, height / 2);
        println!(
            "Diffmap min={:.6}, max={:.6}, center={:.6}",
            min, max, center
        );
    }
}
