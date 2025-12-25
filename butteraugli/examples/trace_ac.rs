//! Trace AC components for uniform images to debug high scores

use butteraugli::opsin::srgb_to_xyb_butteraugli;
use butteraugli::psycho::separate_frequencies;
use butteraugli::{compute_butteraugli, ButteraugliParams, ImageF};

fn check_range(img: &ImageF, name: &str) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let v = img.get(x, y);
            min = min.min(v);
            max = max.max(v);
        }
    }
    println!("  {} range: [{:.6}, {:.6}]", name, min, max);
}

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

    println!(
        "XYB1 center: X={:.4}, Y={:.4}, B={:.4}",
        xyb1.plane(0).get(8, 8),
        xyb1.plane(1).get(8, 8),
        xyb1.plane(2).get(8, 8)
    );
    println!(
        "XYB2 center: X={:.4}, Y={:.4}, B={:.4}",
        xyb2.plane(0).get(8, 8),
        xyb2.plane(1).get(8, 8),
        xyb2.plane(2).get(8, 8)
    );

    // Separate frequencies
    let ps1 = separate_frequencies(&xyb1);
    let ps2 = separate_frequencies(&xyb2);

    println!("\nPs1 frequency components at center (8,8):");
    println!(
        "  UHF: X={:.6}, Y={:.6}",
        ps1.uhf[0].get(8, 8),
        ps1.uhf[1].get(8, 8)
    );
    println!(
        "  HF:  X={:.6}, Y={:.6}",
        ps1.hf[0].get(8, 8),
        ps1.hf[1].get(8, 8)
    );
    println!(
        "  MF:  X={:.6}, Y={:.6}, B={:.6}",
        ps1.mf.plane(0).get(8, 8),
        ps1.mf.plane(1).get(8, 8),
        ps1.mf.plane(2).get(8, 8)
    );
    println!(
        "  LF:  X={:.6}, Y={:.6}, B={:.6}",
        ps1.lf.plane(0).get(8, 8),
        ps1.lf.plane(1).get(8, 8),
        ps1.lf.plane(2).get(8, 8)
    );

    println!("\nPs2 frequency components at center (8,8):");
    println!(
        "  UHF: X={:.6}, Y={:.6}",
        ps2.uhf[0].get(8, 8),
        ps2.uhf[1].get(8, 8)
    );
    println!(
        "  HF:  X={:.6}, Y={:.6}",
        ps2.hf[0].get(8, 8),
        ps2.hf[1].get(8, 8)
    );
    println!(
        "  MF:  X={:.6}, Y={:.6}, B={:.6}",
        ps2.mf.plane(0).get(8, 8),
        ps2.mf.plane(1).get(8, 8),
        ps2.mf.plane(2).get(8, 8)
    );
    println!(
        "  LF:  X={:.6}, Y={:.6}, B={:.6}",
        ps2.lf.plane(0).get(8, 8),
        ps2.lf.plane(1).get(8, 8),
        ps2.lf.plane(2).get(8, 8)
    );

    // Compute differences
    println!("\nFrequency differences:");
    println!(
        "  dUHF: X={:.6}, Y={:.6}",
        ps2.uhf[0].get(8, 8) - ps1.uhf[0].get(8, 8),
        ps2.uhf[1].get(8, 8) - ps1.uhf[1].get(8, 8)
    );
    println!(
        "  dHF:  X={:.6}, Y={:.6}",
        ps2.hf[0].get(8, 8) - ps1.hf[0].get(8, 8),
        ps2.hf[1].get(8, 8) - ps1.hf[1].get(8, 8)
    );
    println!(
        "  dMF:  X={:.6}, Y={:.6}, B={:.6}",
        ps2.mf.plane(0).get(8, 8) - ps1.mf.plane(0).get(8, 8),
        ps2.mf.plane(1).get(8, 8) - ps1.mf.plane(1).get(8, 8),
        ps2.mf.plane(2).get(8, 8) - ps1.mf.plane(2).get(8, 8)
    );
    println!(
        "  dLF:  X={:.6}, Y={:.6}, B={:.6}",
        ps2.lf.plane(0).get(8, 8) - ps1.lf.plane(0).get(8, 8),
        ps2.lf.plane(1).get(8, 8) - ps1.lf.plane(1).get(8, 8),
        ps2.lf.plane(2).get(8, 8) - ps1.lf.plane(2).get(8, 8)
    );

    println!("\nPs1 AC ranges (should be near 0 for uniform):");
    check_range(&ps1.uhf[0], "UHF_X");
    check_range(&ps1.uhf[1], "UHF_Y");
    check_range(&ps1.hf[0], "HF_X");
    check_range(&ps1.hf[1], "HF_Y");
    check_range(ps1.mf.plane(0), "MF_X");
    check_range(ps1.mf.plane(1), "MF_Y");

    // Compute butteraugli
    let params = ButteraugliParams::default();
    let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params);
    println!("\nButteraugli score: {:.4}", result.score);

    if let Some(dm) = result.diffmap.as_ref() {
        let center = dm.get(8, 8);
        let corner = dm.get(0, 0);
        println!("  diffmap[8,8] = {:.4}", center);
        println!("  diffmap[0,0] = {:.4}", corner);
    }
}
