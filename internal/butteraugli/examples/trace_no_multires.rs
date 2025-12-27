//! Test butteraugli without multiresolution to understand base score

use butteraugli_oxide::consts::WMUL;
use butteraugli_oxide::mask::{mask_dc_y, mask_y};

fn main() {
    // For uniform 128 vs 138 gray:
    // DC differences (computed from LF vals):
    // dLF_X = 0.0804, dLF_Y = 56.1577, dLF_B = 43.3232
    //
    // DC contribution = sum of (d^2 * WMUL[6+c])
    // X: 0.0804^2 * 29.2354 = 0.189
    // Y: 56.1577^2 * 0.8446 = 2663.69
    // B: 43.3232^2 * 0.7036 = 1320.67
    // Total DC = 3984.55

    let dc_total = 3984.55_f64;

    // For uniform images, mask value = 0
    // MaskDcY(0) = 0.0624
    let mask_dc = mask_dc_y(0.0);

    // Single resolution score = sqrt(DC * MaskDcY)
    let single_res_score = (dc_total * mask_dc).sqrt();
    println!("Single resolution score: {:.4}", single_res_score);

    // Multi-resolution blending:
    // At each level, the score is the same for uniform images
    // Blending: new = old * (1 - 0.3 * 0.5) + 0.5 * sub_score
    //         = old * 0.85 + 0.5 * old (since sub_score = old for uniform)
    //         = old * 1.35

    let multi_res_score = single_res_score * 1.35;
    println!("Multi-resolution score (1 level): {:.4}", multi_res_score);

    // For 16x16 image:
    // - Level 0: 16x16 → score = S
    // - Level 1: 8x8 → score = S
    // - Level 2: would be 4x4, but MIN_SIZE = 8, so no more recursion
    // Only 1 level of multiresolution for 16x16 image

    // For 64x64 image:
    // - Level 0: 64x64 → score = S0
    // - Level 1: 32x32 → blend: S0 * 0.85 + 0.5 * S1
    // - Level 2: 16x16 → blend: ... + 0.5 * S2
    // - Level 3: 8x8 → blend: ... + 0.5 * S3
    //
    // If all S are equal (uniform):
    // S0 = base
    // After L1: base * 0.85 + 0.5 * base = 1.35 * base
    // After L2: 1.35 * base * 0.85 + 0.5 * base = 1.1475 * base + 0.5 * base = 1.6475 * base
    // After L3: 1.6475 * base * 0.85 + 0.5 * base = 1.4 * base + 0.5 * base = 1.9 * base

    println!("\nExpected scores for 64x64 uniform image:");
    let s = single_res_score;
    let l1 = s * 0.85 + 0.5 * s;
    let l2 = l1 * 0.85 + 0.5 * s;
    let l3 = l2 * 0.85 + 0.5 * s;
    println!("  After L0 (64x64): {:.4}", s);
    println!("  After L1 (32x32): {:.4}", l1);
    println!("  After L2 (16x16): {:.4}", l2);
    println!("  After L3 (8x8):   {:.4}", l3);
    println!("  Factor: {:.4}", l3 / s);

    // Let's verify by computing what factor we'd expect for 16x16 vs 64x64
    println!(
        "\nFor 16x16 uniform: factor = 1.35, expected score = {:.4}",
        s * 1.35
    );
    println!(
        "For 64x64 uniform: factor ≈ 1.9, expected score = {:.4}",
        s * 1.9
    );

    println!("\nConclusion: For uniform images, the butteraugli score increases");
    println!("with image size due to multi-resolution blending. This is expected!");
}
