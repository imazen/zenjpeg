//! Verify [`zenjpeg::color::rgb_to_ycbcr_f32`] matches ITU-R BT.601 Full-range
//! reference values.
//!
//! `rgb_to_ycbcr_f32` is live production code — called from the gamma-aware
//! chroma helpers in `encode/chroma.rs` (4:2:2 and 4:4:0 paths). This test
//! guards against silent regression of the BT.601 coefficients in
//! `foundation/consts.rs`.
//!
//! Replaces the previous `yuv_crate_comparison.rs` which used the external
//! `yuv` crate as an oracle. zenyuv-vs-yuv-crate parity is now covered by
//! zenyuv's own tests (`zenyuv/src/lib.rs` `#[cfg(test)]`) plus the
//! `precision_vs_yuv_crate` example in zenyuv. zenjpeg no longer depends
//! on the external `yuv` crate in any build.

use zenjpeg::color::rgb_to_ycbcr_f32;

/// Pure-color ITU-R BT.601 Full-range reference values, hand-computed from
/// the spec coefficients:
///
/// ```text
/// Y  =  0.299  R + 0.587  G + 0.114  B
/// Cb = -0.1687 R - 0.3313 G + 0.5    B + 128
/// Cr =  0.5    R - 0.4187 G - 0.0813 B + 128
/// ```
#[test]
fn bt601_pure_colors() {
    // (R, G, B, expected Y, Cb, Cr). Values hand-computed from zenjpeg's
    // BT.601 constants in `foundation/consts.rs` (0.299/0.587/0.114,
    // -0.168_736/-0.331_264/0.5, 0.5/-0.418_688/-0.081_312). Function
    // returns unclamped f32 — red Cr and blue Cb exceed 255.
    let cases = [
        ("black",   (  0.0,   0.0,   0.0), (  0.000, 128.00000, 128.00000)),
        ("white",   (255.0, 255.0, 255.0), (255.000, 128.00000, 128.00000)),
        ("red",     (255.0,   0.0,   0.0), ( 76.245,  84.97232, 255.50000)),
        ("green",   (  0.0, 255.0,   0.0), (149.685,  43.52768,  21.23456)),
        ("blue",    (  0.0,   0.0, 255.0), ( 29.070, 255.50000, 107.26544)),
        ("gray128", (128.0, 128.0, 128.0), (128.000, 128.00000, 128.00000)),
        ("yellow",  (255.0, 255.0,   0.0), (225.930,   0.50000, 148.73456)),
        ("cyan",    (  0.0, 255.0, 255.0), (178.755, 171.02768,   0.50000)),
        ("magenta", (255.0,   0.0, 255.0), (105.315, 212.47232, 234.76544)),
    ];

    const TOL: f32 = 0.05;
    for (name, (r, g, b), (ey, ecb, ecr)) in cases {
        let (y, cb, cr) = rgb_to_ycbcr_f32(r, g, b);
        assert!(
            (y - ey).abs() < TOL,
            "{name}: Y={y} expected {ey} (diff {:.4})",
            (y - ey).abs()
        );
        assert!(
            (cb - ecb).abs() < TOL,
            "{name}: Cb={cb} expected {ecb} (diff {:.4})",
            (cb - ecb).abs()
        );
        assert!(
            (cr - ecr).abs() < TOL,
            "{name}: Cr={cr} expected {ecr} (diff {:.4})",
            (cr - ecr).abs()
        );
    }
}

/// Verify the function doesn't scramble Y/Cb/Cr output positions — check that
/// grayscale inputs yield Cb=Cr=128 and Y=input, while saturated colors pin
/// one chroma channel at an extreme. Catches accidental coefficient swaps
/// or tuple-order regressions.
#[test]
fn bt601_structural_invariants() {
    // Grayscale: Cb and Cr must both be 128.0 (the achromatic axis).
    for v in [0.0, 64.0, 128.0, 192.0, 255.0] {
        let (y, cb, cr) = rgb_to_ycbcr_f32(v, v, v);
        assert!((y - v).abs() < 0.001, "gray({v}) Y should equal input");
        assert!((cb - 128.0).abs() < 0.001, "gray({v}) Cb should be 128");
        assert!((cr - 128.0).abs() < 0.001, "gray({v}) Cr should be 128");
    }

    // Saturated blue pins Cb at 255 (blue adds +0.5 to Cb, max input 255 → +127.5 → 255.5 clamps).
    let (_, cb, _) = rgb_to_ycbcr_f32(0.0, 0.0, 255.0);
    assert!(cb > 254.0, "saturated blue Cb should be ~255, got {cb}");

    // Saturated red pins Cr at 255 (red adds +0.5 to Cr).
    let (_, _, cr) = rgb_to_ycbcr_f32(255.0, 0.0, 0.0);
    assert!(cr > 254.0, "saturated red Cr should be ~255, got {cr}");

    // Y weights: green contributes most (0.587), blue least (0.114).
    let (yg, _, _) = rgb_to_ycbcr_f32(0.0, 255.0, 0.0);
    let (yr, _, _) = rgb_to_ycbcr_f32(255.0, 0.0, 0.0);
    let (yb, _, _) = rgb_to_ycbcr_f32(0.0, 0.0, 255.0);
    assert!(yg > yr && yr > yb, "Y weight order G>R>B violated: g={yg} r={yr} b={yb}");
}

/// Brute force: for 4096 sampled RGB values, verify `rgb_to_ycbcr_f32`
/// matches a naive inline BT.601 formula to within floating-point noise.
/// Catches any divergence between the shared constants and the ITU-R
/// reference formula.
#[test]
fn bt601_brute_force_matches_inline_formula() {
    let mut max_y = 0.0f32;
    let mut max_cb = 0.0f32;
    let mut max_cr = 0.0f32;
    let mut worst_rgb = (0u8, 0u8, 0u8);

    for r in (0..=255u8).step_by(17) {
        for g in (0..=255u8).step_by(17) {
            for b in (0..=255u8).step_by(17) {
                let (y1, cb1, cr1) = rgb_to_ycbcr_f32(r as f32, g as f32, b as f32);

                // Inline BT.601 — same rounded-to-6-digit constants zenjpeg ships
                // in foundation/consts.rs. Expresses the formula independently of
                // the FMA reassociation zenjpeg's mul_add chain uses.
                let rf = r as f32;
                let gf = g as f32;
                let bf = b as f32;
                let y2 = 0.299 * rf + 0.587 * gf + 0.114 * bf;
                let cb2 = -0.168_736 * rf - 0.331_264 * gf + 0.5 * bf + 128.0;
                let cr2 = 0.5 * rf - 0.418_688 * gf - 0.081_312 * bf + 128.0;

                let dy = (y1 - y2).abs();
                let dcb = (cb1 - cb2).abs();
                let dcr = (cr1 - cr2).abs();
                if dy > max_y {
                    max_y = dy;
                    worst_rgb = (r, g, b);
                }
                max_cb = max_cb.max(dcb);
                max_cr = max_cr.max(dcr);
            }
        }
    }

    // FMA/rounding differences — should be well under 0.01 levels.
    const TOL: f32 = 0.01;
    assert!(
        max_y < TOL,
        "Y diff {max_y} too large at RGB {worst_rgb:?} (tolerance {TOL})"
    );
    assert!(max_cb < TOL, "Cb diff {max_cb} too large");
    assert!(max_cr < TOL, "Cr diff {max_cr} too large");
}
