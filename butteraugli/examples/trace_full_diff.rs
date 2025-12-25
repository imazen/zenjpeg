//! Trace full diff computation to find where extra score comes from

use butteraugli_oxide::consts::WMUL;
use butteraugli_oxide::mask::{mask_dc_y, mask_y};
use butteraugli_oxide::opsin::srgb_to_xyb_butteraugli;
use butteraugli_oxide::psycho::separate_frequencies;
use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};

fn main() {
    let width = 16;
    let height = 16;
    let intensity_target = 80.0;

    // Uniform gray images
    let rgb1: Vec<u8> = vec![128; width * height * 3];
    let rgb2: Vec<u8> = vec![138; width * height * 3];

    // Convert to XYB
    let xyb1 = srgb_to_xyb_butteraugli(&rgb1, width, height, intensity_target);
    let xyb2 = srgb_to_xyb_butteraugli(&rgb2, width, height, intensity_target);

    // Separate frequencies
    let ps1 = separate_frequencies(&xyb1);
    let ps2 = separate_frequencies(&xyb2);

    let cx = 8;
    let cy = 8;

    // Compute DC differences manually
    println!("=== DC (LF) Differences ===");
    let mut dc_total = 0.0_f32;
    for c in 0..3 {
        let d = ps1.lf.plane(c).get(cx, cy) - ps2.lf.plane(c).get(cx, cy);
        let d2w = d * d * WMUL[6 + c] as f32;
        dc_total += d2w;
        println!(
            "  LF[{}]: d={:.4}, d^2*w={:.4} (w={:.4})",
            c,
            d,
            d2w,
            WMUL[6 + c]
        );
    }
    println!("  DC total = {:.4}", dc_total);

    // Compute AC differences for MF
    println!("\n=== AC (MF) Differences ===");
    let mut mf_total = 0.0_f32;
    for c in 0..3 {
        let d = ps1.mf.plane(c).get(cx, cy) - ps2.mf.plane(c).get(cx, cy);
        let d2w = d * d * WMUL[3 + c] as f32;
        mf_total += d2w;
        println!(
            "  MF[{}]: d={:.6}, d^2*w={:.9} (w={:.4})",
            c,
            d,
            d2w,
            WMUL[3 + c]
        );
    }
    println!("  MF total = {:.9}", mf_total);

    // For uniform images, mask should be 0
    let mask_val = 0.0;
    let mask_y_val = mask_y(mask_val);
    let mask_dc_y_val = mask_dc_y(mask_val);
    println!("\n=== Masking ===");
    println!("  mask = {}", mask_val);
    println!("  MaskY(0) = {:.6}", mask_y_val);
    println!("  MaskDcY(0) = {:.6}", mask_dc_y_val);

    // Expected diffmap
    let dc_masked = dc_total as f64 * mask_dc_y_val;
    let ac_masked = mf_total as f64 * mask_y_val;
    println!("\n=== Expected Score (DC only) ===");
    println!("  DC * MaskDcY = {:.4}", dc_masked);
    println!("  MF * MaskY = {:.9}", ac_masked);
    println!(
        "  Expected diffmap = sqrt({:.4} + {:.9}) = {:.4}",
        dc_masked,
        ac_masked,
        (dc_masked + ac_masked).sqrt()
    );

    // Actual butteraugli
    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
    println!("\n=== Actual ===");
    println!("  Butteraugli score = {:.4}", result.score);

    // The difference
    println!("\n=== Gap Analysis ===");
    let expected = (dc_masked + ac_masked).sqrt();
    let actual = result.score;
    println!("  Expected: {:.4}", expected);
    println!("  Actual: {:.4}", actual);
    println!("  Ratio: {:.4}", actual / expected);

    println!("\nNote: For uniform images, AC should contribute 0.");
    println!(
        "The {} gap suggests extra AC contributions.",
        (actual / expected - 1.0) * 100.0
    );
}
