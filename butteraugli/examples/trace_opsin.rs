//! Trace OpsinDynamicsImage to find the non-monotonicity bug
//!
//! Run with: cargo run --example trace_opsin

use butteraugli::opsin::{gamma, opsin_absorbance};

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
    println!("=== OpsinDynamicsImage Trace ===\n");

    let intensity_target = 80.0;

    println!("Gray Level | Linear | Opsin0   | Gamma(O) | Sens0    | Cur0     | Y");
    println!("{}", "-".repeat(80));

    for gray in [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140] {
        let linear = srgb_to_linear(gray);
        let scaled = linear * intensity_target;

        // OpsinAbsorbance
        let (o0, o1, o2) = opsin_absorbance(scaled, scaled, scaled, true);

        // Gamma
        let g0 = gamma(o0);

        // Sensitivity
        let min_val = 1e-4_f32;
        let s0 = (g0 / o0).max(min_val);

        // cur_mixed (unclamped opsin * sensitivity)
        let (u0, u1, _) = opsin_absorbance(scaled, scaled, scaled, false);
        let cur0 = (u0 * s0).max(1.7557483643287353); // clamp to min
        let cur1 = (u1 * s0).max(1.7557483643287353);

        // Y = cur0 + cur1
        let y = cur0 + cur1;

        println!(
            "{:>10} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4}",
            gray, linear, o0, g0, s0, cur0, y
        );
    }

    println!("\n--- Expected monotonicity check ---");
    println!("Y should increase with gray level.");
    println!();

    // Now check what's happening with the blur
    println!("=== Checking blur effect on sensitivity ===\n");

    // For a uniform image of size 16x16, the blur should NOT change the values
    // because every pixel is the same. Let's verify this.
    use butteraugli::opsin::srgb_to_xyb_butteraugli;

    for gray in [128u8, 133, 138] {
        let width = 4;
        let height = 4;
        let rgb: Vec<u8> = vec![gray; width * height * 3];
        let xyb = srgb_to_xyb_butteraugli(&rgb, width, height, intensity_target);

        // Check corner vs center values
        println!("Gray {}: ", gray);
        println!("  XYB[0,0]: X={:.4}, Y={:.4}, B={:.4}",
                 xyb.plane(0).get(0, 0),
                 xyb.plane(1).get(0, 0),
                 xyb.plane(2).get(0, 0));
        println!("  XYB[1,1]: X={:.4}, Y={:.4}, B={:.4}",
                 xyb.plane(0).get(1, 1),
                 xyb.plane(1).get(1, 1),
                 xyb.plane(2).get(1, 1));

        // Also check what raw opsin values would give
        let linear = srgb_to_linear(gray);
        let scaled = linear * intensity_target;
        let (o0, o1, _) = opsin_absorbance(scaled, scaled, scaled, true);
        let g0 = gamma(o0);
        let s0 = (g0 / o0).max(1e-4);
        let (u0, u1, _) = opsin_absorbance(scaled, scaled, scaled, false);
        let cur0 = (u0 * s0).max(1.7557483643287353);
        let cur1 = (u1 * s0).max(1.7557483643287353);
        let expected_y = cur0 + cur1;
        println!("  Expected Y (no blur): {:.4}", expected_y);
        println!();
    }

    // The issue might be that for uniform images, the BLURRED value is the same
    // as the original, but the sensitivity is computed from BLURRED which goes
    // through a different code path...

    println!("\n=== Testing sensitivity computation ===\n");

    // In opsin_dynamics_image, sensitivity is computed from BLURRED pre_mixed values
    // Then applied to ORIGINAL pre_mixed values

    // For uniform images, blurred = original, so:
    // sensitivity = Gamma(opsin) / opsin
    // cur = opsin * sensitivity = opsin * Gamma(opsin) / opsin = Gamma(opsin)

    // Wait, that's not right either. Let me trace through more carefully...

    // Actually, I think the issue is that I'm using sensitivity0 for both cur_mixed0 AND cur_mixed1
    // But in the C++ code, there are SEPARATE sensitivities for each channel

    println!("Tracing per-channel sensitivity:");
    for gray in [128u8, 133, 138] {
        let linear = srgb_to_linear(gray);
        let scaled = linear * intensity_target;

        let (o0, o1, o2) = opsin_absorbance(scaled, scaled, scaled, true);

        // Each channel has its own sensitivity
        let g0 = gamma(o0);
        let g1 = gamma(o1);
        let g2 = gamma(o2);

        let s0 = (g0 / o0).max(1e-4);
        let s1 = (g1 / o1).max(1e-4);
        let s2 = (g2 / o2).max(1e-4);

        let (u0, u1, u2) = opsin_absorbance(scaled, scaled, scaled, false);
        let cur0 = (u0 * s0).max(1.7557483643287353);
        let cur1 = (u1 * s1).max(1.7557483643287353);
        let cur2 = (u2 * s2).max(12.226454707163354);

        let x = cur0 - cur1;
        let y = cur0 + cur1;
        let b = cur2;

        println!("Gray {}: opsin=[{:.4}, {:.4}, {:.4}], sens=[{:.4}, {:.4}, {:.4}]",
                 gray, o0, o1, o2, s0, s1, s2);
        println!("        cur=[{:.4}, {:.4}, {:.4}], XYB=[{:.4}, {:.4}, {:.4}]",
                 cur0, cur1, cur2, x, y, b);
        println!();
    }
}
