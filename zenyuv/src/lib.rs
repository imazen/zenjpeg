//! SIMD-optimized YUV↔RGB color matrix conversion.
//!
//! Supports BT.601, BT.709, and BT.2020 matrices in both full and limited
//! (studio) range. Covers 4:4:4, 4:2:0, 4:2:2, and 4:0:0 (grayscale) for
//! both encode (RGB->YCbCr) and decode (YCbCr->RGB) directions.
//!
//! # Dispatch tiers
//!
//! - **x86-64 AVX2** -- `#[arcane]` pmaddwd kernel (32 pixels/iter for 4:4:4,
//!   2x32 fused Y+chroma for 4:2:0). Entry point selected at runtime via
//!   `X64V3Token::summon()`.
//! - **Generic fallback** -- magetypes `f32x8` FMA via `#[magetypes(v3, neon,
//!   wasm128, scalar)]`. Covers NEON, WASM SIMD128, and scalar.
//!
//! # Example
//!
//! ```
//! // Encode: RGB -> YCbCr 4:4:4
//! let rgb = vec![128u8; 64 * 64 * 3];
//! let mut y = vec![0u8; 64 * 64];
//! let mut cb = vec![0u8; 64 * 64];
//! let mut cr = vec![0u8; 64 * 64];
//! zenyuv::rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, 64, 64);
//!
//! // Decode: YCbCr 4:4:4 -> RGB
//! let mut out = vec![0u8; 64 * 64 * 3];
//! zenyuv::yuv444_to_rgb(&y, &cb, &cr, &mut out, 64, 64);
//! ```

#![no_std]
#![forbid(unsafe_code)]

// ── Modules ────────────────────────────────────────────────────────────────

pub mod types;
pub mod gamma;
mod encode;
mod decode;
mod encode_generic;
mod decode_generic;
pub mod sharp;

#[cfg(target_arch = "x86_64")]
mod avx2_encode;
#[cfg(target_arch = "x86_64")]
mod avx2_decode;

#[cfg(target_arch = "aarch64")]
mod neon_encode;

#[cfg(target_arch = "wasm32")]
mod wasm_encode;

// ── Public re-exports ──────────────────────────────────────────────────────

pub use types::{Matrix, Range, ForwardCoeffs, InverseCoeffs};

// Encode (RGB -> YCbCr)
pub use encode::{
    rgb_to_yuv444, rgb_to_yuv444_with,
    rgb_to_yuv420, rgb_to_yuv420_with,
};

// Sharp YUV (iterative perceptual chroma optimization)
pub use sharp::{rgb_to_yuv420_sharp, SharpYuvConfig};
pub use gamma::GammaLuts;

// Decode (YCbCr -> RGB)
pub use decode::{
    yuv444_to_rgb, yuv444_to_rgb_with,
    yuv420_to_rgb, yuv420_to_rgb_with,
    yuv420_to_rgb_bilinear, yuv420_to_rgb_bilinear_with,
    yuv422_to_rgb, yuv422_to_rgb_with,
    yuv400_to_rgb, yuv400_to_rgb_with,
};

// ── Shared utilities ───────────────────────────────────────────────────────

#[inline(always)]
fn clamp_round(v: f32) -> u8 {
    let r = v.round() as i32;
    r.clamp(0, 255) as u8
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
extern crate std;
#[cfg(test)]
extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::ToString, vec, vec::Vec};
    use std::eprintln;

    fn make_pattern(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = ((x * 7 + y * 3) & 0xff) as u8;
                rgb[i + 1] = ((x * 3 ^ y * 11) & 0xff) as u8;
                rgb[i + 2] = (((x + y) * 5) & 0xff) as u8;
            }
        }
        rgb
    }

    fn mean_abs_err(a: &[u8], b: &[u8]) -> f64 {
        let mut s = 0u64;
        for (x, y) in a.iter().zip(b.iter()) {
            s += x.abs_diff(*y) as u64;
        }
        s as f64 / a.len() as f64
    }

    fn max_abs_err(a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| x.abs_diff(*y) as u32)
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn yuv444_matches_yuv_crate() {
        let (w, h) = (123, 45);
        let rgb = make_pattern(w, h);
        let n = w * h;

        let mut y = vec![0u8; n];
        let mut cb = vec![0u8; n];
        let mut cr = vec![0u8; n];
        rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, w, h);

        let mut ref_img = yuv::YuvPlanarImageMut::alloc(
            w as u32,
            h as u32,
            yuv::YuvChromaSubsampling::Yuv444,
        );
        yuv::rgb_to_yuv444(
            &mut ref_img,
            &rgb,
            (w * 3) as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
            yuv::YuvConversionMode::Professional,
        )
        .unwrap();

        let ry = ref_img.y_plane.borrow();
        let ru = ref_img.u_plane.borrow();
        let rv = ref_img.v_plane.borrow();

        assert!(max_abs_err(&y, ry) <= 1, "Y max err > 1");
        assert!(max_abs_err(&cb, ru) <= 1, "Cb max err > 1");
        assert!(max_abs_err(&cr, rv) <= 1, "Cr max err > 1");
        assert!(mean_abs_err(&y, ry) < 0.05);
        assert!(mean_abs_err(&cb, ru) < 0.05);
        assert!(mean_abs_err(&cr, rv) < 0.05);
    }

    #[test]
    fn yuv420_matches_yuv_crate() {
        let (w, h) = (124, 46);
        let rgb = make_pattern(w, h);
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        rgb_to_yuv420(&rgb, &mut y, &mut cb, &mut cr, w, h);

        let mut ref_img = yuv::YuvPlanarImageMut::alloc(
            w as u32,
            h as u32,
            yuv::YuvChromaSubsampling::Yuv420,
        );
        yuv::rgb_to_yuv420(
            &mut ref_img,
            &rgb,
            (w * 3) as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
            yuv::YuvConversionMode::Professional,
        )
        .unwrap();

        let ry = ref_img.y_plane.borrow();
        let ru = ref_img.u_plane.borrow();
        let rv = ref_img.v_plane.borrow();

        let y_max = max_abs_err(&y, ry);
        let cb_max = max_abs_err(&cb, ru);
        let cr_max = max_abs_err(&cr, rv);
        let cb_mean = mean_abs_err(&cb, ru);
        let cr_mean = mean_abs_err(&cr, rv);
        eprintln!(
            "420 parity: Y max={y_max} Cb max={cb_max} mean={cb_mean:.4} Cr max={cr_max} mean={cr_mean:.4}"
        );
        assert!(y_max <= 1, "Y max err {y_max} > 1");
        assert!(cb_max <= 3, "Cb max err {cb_max} > 3");
        assert!(cr_max <= 3, "Cr max err {cr_max} > 3");
        assert!(cb_mean < 0.3, "Cb mean err {cb_mean} > 0.3");
        assert!(cr_mean < 0.3, "Cr mean err {cr_mean} > 0.3");
    }

    #[test]
    fn clamp_boundaries_white_and_black() {
        let rgb: Vec<u8> = [[0u8, 0, 0], [255, 255, 255]]
            .iter()
            .flatten()
            .copied()
            .collect();
        let mut y = [0u8; 2];
        let mut cb = [0u8; 2];
        let mut cr = [0u8; 2];
        rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, 2, 1);
        assert_eq!(y, [0, 255]);
        assert_eq!(cb, [128, 128]);
        assert_eq!(cr, [128, 128]);
    }

    /// Verify all SIMD dispatch tiers produce output within +/-1 of each other
    /// for 4:4:4. AVX2 uses 15-bit fixed-point while generic uses f32 FMA.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn yuv444_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let rgb = make_pattern(256, 256);
        let n = 256 * 256;
        let mut y_ref = vec![0u8; n];
        let mut cb_ref = vec![0u8; n];
        let mut cr_ref = vec![0u8; n];
        rgb_to_yuv444(&rgb, &mut y_ref, &mut cb_ref, &mut cr_ref, 256, 256);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut y = vec![0u8; n];
            let mut cb = vec![0u8; n];
            let mut cr = vec![0u8; n];
            rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, 256, 256);
            let ym = max_abs_err(&y, &y_ref);
            let cbm = max_abs_err(&cb, &cb_ref);
            let crm = max_abs_err(&cr, &cr_ref);
            assert!(
                ym <= 1 && cbm <= 1 && crm <= 1,
                "tier parity exceeded +/-1 at {perm}: Y={ym} Cb={cbm} Cr={crm}"
            );
        });
        std::eprintln!("yuv444 dispatch parity: {report}");
        assert!(report.permutations_run >= 2, "need at least 2 permutations");
    }

    /// Verify all SIMD dispatch tiers produce output within +/-1 for 4:2:0.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn yuv420_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let (w, h) = (256, 256);
        let rgb = make_pattern(w, h);
        let cw = w / 2;
        let ch = h / 2;
        let mut y_ref = vec![0u8; w * h];
        let mut cb_ref = vec![0u8; cw * ch];
        let mut cr_ref = vec![0u8; cw * ch];
        rgb_to_yuv420(&rgb, &mut y_ref, &mut cb_ref, &mut cr_ref, w, h);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut y = vec![0u8; w * h];
            let mut cb = vec![0u8; cw * ch];
            let mut cr = vec![0u8; cw * ch];
            rgb_to_yuv420(&rgb, &mut y, &mut cb, &mut cr, w, h);
            let ym = max_abs_err(&y, &y_ref);
            let cbm = max_abs_err(&cb, &cb_ref);
            let crm = max_abs_err(&cr, &cr_ref);
            assert!(
                ym <= 1 && cbm <= 1 && crm <= 1,
                "tier parity exceeded +/-1 at {perm}: Y={ym} Cb={cbm} Cr={crm}"
            );
        });
        std::eprintln!("yuv420 dispatch parity: {report}");
        assert!(report.permutations_run >= 2, "need at least 2 permutations");
    }

    /// Exhaustive single-pixel precision: all 256^3 RGB inputs through 4:4:4.
    #[test]
    #[ignore]
    fn exhaustive_all_rgb_values() {
        let mut max_diff_y = 0u8;
        let mut max_diff_cb = 0u8;
        let mut max_diff_cr = 0u8;

        for g in 0..=255u8 {
            for b in 0..=255u8 {
                let mut rgb = [0u8; 256 * 3];
                for r in 0..=255u8 {
                    rgb[r as usize * 3] = r;
                    rgb[r as usize * 3 + 1] = g;
                    rgb[r as usize * 3 + 2] = b;
                }
                let mut y = [0u8; 256];
                let mut cb_arr = [0u8; 256];
                let mut cr_arr = [0u8; 256];
                rgb_to_yuv444(&rgb, &mut y, &mut cb_arr, &mut cr_arr, 256, 1);

                for r in 0..=255u8 {
                    let rf = r as f32;
                    let gf = g as f32;
                    let bf = b as f32;
                    let y_ref = clamp_round(0.299 * rf + 0.587 * gf + 0.114 * bf);
                    let cb_ref =
                        clamp_round(-0.168_736 * rf + -0.331_264 * gf + 0.5 * bf + 128.0);
                    let cr_ref =
                        clamp_round(0.5 * rf + -0.418_688 * gf + -0.081_312 * bf + 128.0);

                    let dy = y[r as usize].abs_diff(y_ref);
                    let dcb = cb_arr[r as usize].abs_diff(cb_ref);
                    let dcr = cr_arr[r as usize].abs_diff(cr_ref);
                    max_diff_y = max_diff_y.max(dy);
                    max_diff_cb = max_diff_cb.max(dcb);
                    max_diff_cr = max_diff_cr.max(dcr);

                    assert!(
                        dy <= 1 && dcb <= 1 && dcr <= 1,
                        "R={r} G={g} B={b}: Y {}/{y_ref} Cb {}/{cb_ref} Cr {}/{cr_ref}",
                        y[r as usize],
                        cb_arr[r as usize],
                        cr_arr[r as usize],
                    );
                }
            }
        }
        std::eprintln!(
            "exhaustive 256^3: max diff Y={max_diff_y} Cb={max_diff_cb} Cr={max_diff_cr}"
        );
    }

    // ── Decode tests ───────────────────────────────────────────────────────

    #[test]
    fn yuv444_roundtrip() {
        let (w, h) = (64, 64);
        let rgb = make_pattern(w, h);
        let n = w * h;

        let mut y = vec![0u8; n];
        let mut cb = vec![0u8; n];
        let mut cr = vec![0u8; n];
        rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, w, h);

        let mut out = vec![0u8; n * 3];
        yuv444_to_rgb(&y, &cb, &cr, &mut out, w, h);

        // Roundtrip error: encode + decode should be <= 2 levels max.
        let max_err = max_abs_err(&rgb, &out);
        let mean_err = mean_abs_err(&rgb, &out);
        eprintln!("444 roundtrip: max={max_err} mean={mean_err:.4}");
        assert!(max_err <= 2, "roundtrip max err {max_err} > 2");
        assert!(mean_err < 0.5, "roundtrip mean err {mean_err} > 0.5");
    }

    #[test]
    fn yuv420_roundtrip() {
        // Use a smooth gradient to avoid extreme chroma errors from high-frequency
        // content. make_pattern() has rapidly varying colors that cause large
        // roundtrip errors with 4:2:0 (inherent to chroma subsampling).
        let (w, h) = (64, 64);
        let n = w * h;
        let mut rgb = vec![0u8; n * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = ((x * 4) & 0xff) as u8;
                rgb[i + 1] = ((y * 4) & 0xff) as u8;
                rgb[i + 2] = (((x + y) * 2) & 0xff) as u8;
            }
        }
        let cw = w / 2;
        let ch = h / 2;

        let mut y_plane = vec![0u8; n];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];
        rgb_to_yuv420(&rgb, &mut y_plane, &mut cb, &mut cr, w, h);

        let mut out = vec![0u8; n * 3];
        yuv420_to_rgb(&y_plane, &cb, &cr, &mut out, w, h);

        // 4:2:0 loses chroma resolution. Smooth gradients should roundtrip
        // with modest error since adjacent 2x2 blocks are similar.
        let max_err = max_abs_err(&rgb, &out);
        let mean_err = mean_abs_err(&rgb, &out);
        eprintln!("420 roundtrip: max={max_err} mean={mean_err:.4}");
        assert!(max_err <= 10, "roundtrip max err {max_err} > 10");
        assert!(mean_err < 3.0, "roundtrip mean err {mean_err} > 3.0");
    }

    #[test]
    fn yuv400_grayscale() {
        let y = [0u8, 128, 255];
        let mut rgb = [0u8; 9];
        yuv400_to_rgb(&y, &mut rgb, 3, 1);
        assert_eq!(rgb, [0, 0, 0, 128, 128, 128, 255, 255, 255]);
    }

    #[test]
    fn yuv422_roundtrip_basic() {
        // Pure gray: should roundtrip perfectly.
        let (w, h) = (4, 2);
        let rgb = [128u8; 4 * 2 * 3];
        let n = w * h;
        let cw = w / 2;

        let mut y = vec![0u8; n];
        let mut cb = vec![0u8; cw * h];
        let mut cr = vec![0u8; cw * h];
        // Manually set YCbCr for gray: Y=128, Cb=128, Cr=128
        for i in 0..n {
            y[i] = 128;
        }
        for i in 0..cw * h {
            cb[i] = 128;
            cr[i] = 128;
        }

        let mut out = vec![0u8; n * 3];
        yuv422_to_rgb(&y, &cb, &cr, &mut out, w, h);

        assert_eq!(&out, &rgb);
    }

    #[test]
    fn decode_white_and_black() {
        // White: Y=255, Cb=128, Cr=128
        // Black: Y=0, Cb=128, Cr=128
        let y = [0u8, 255];
        let cb = [128u8, 128];
        let cr = [128u8, 128];
        let mut rgb = [0u8; 6];
        yuv444_to_rgb(&y, &cb, &cr, &mut rgb, 2, 1);
        assert_eq!(rgb, [0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn limited_range_encode_decode() {
        // Limited range: Y should be in [16, 235].
        let rgb = [255u8, 255, 255, 0, 0, 0];
        let mut y = [0u8; 2];
        let mut cb = [0u8; 2];
        let mut cr = [0u8; 2];
        encode::rgb_to_yuv444_with(&rgb, &mut y, &mut cb, &mut cr, 2, 1, Range::Limited, Matrix::Bt601);

        // White should map to Y~235, black to Y~16.
        assert!(
            y[0] >= 233 && y[0] <= 237,
            "white Y={}, expected ~235",
            y[0]
        );
        assert!(y[1] >= 14 && y[1] <= 18, "black Y={}, expected ~16", y[1]);

        // Roundtrip through limited range decode.
        let mut out = [0u8; 6];
        decode::yuv444_to_rgb_with(&y, &cb, &cr, &mut out, 2, 1, Range::Limited, Matrix::Bt601);

        // Should recover close to original.
        let max_err = max_abs_err(&rgb, &out);
        eprintln!("limited roundtrip: max={max_err}, out={out:?}");
        assert!(max_err <= 2, "limited roundtrip max err {max_err} > 2");
    }

    #[test]
    fn bt709_coefficients() {
        // Just verify BT.709 produces different output than BT.601.
        let rgb = [200u8, 100, 50];
        let mut y601 = [0u8; 1];
        let mut cb601 = [0u8; 1];
        let mut cr601 = [0u8; 1];
        let mut y709 = [0u8; 1];
        let mut cb709 = [0u8; 1];
        let mut cr709 = [0u8; 1];

        encode::rgb_to_yuv444_with(&rgb, &mut y601, &mut cb601, &mut cr601, 1, 1, Range::Full, Matrix::Bt601);
        encode::rgb_to_yuv444_with(&rgb, &mut y709, &mut cb709, &mut cr709, 1, 1, Range::Full, Matrix::Bt709);

        // BT.709 has less weight on red for Y, so Y should differ.
        assert_ne!(y601[0], y709[0], "BT.601 and BT.709 should produce different Y");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn yuv444_decode_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let (w, h) = (256, 256);
        let rgb = make_pattern(w, h);
        let n = w * h;

        // Encode first.
        let mut y = vec![0u8; n];
        let mut cb = vec![0u8; n];
        let mut cr = vec![0u8; n];
        rgb_to_yuv444(&rgb, &mut y, &mut cb, &mut cr, w, h);

        // Reference decode.
        let mut ref_out = vec![0u8; n * 3];
        yuv444_to_rgb(&y, &cb, &cr, &mut ref_out, w, h);

        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let mut out = vec![0u8; n * 3];
            yuv444_to_rgb(&y, &cb, &cr, &mut out, w, h);
            let max_err = max_abs_err(&out, &ref_out);
            assert!(
                max_err <= 1,
                "decode tier parity exceeded +/-1 at {perm}: max={max_err}"
            );
        });
        std::eprintln!("yuv444 decode dispatch parity: {report}");
        assert!(report.permutations_run >= 2);
    }

    #[test]
    fn sharp_yuv_420_basic() {
        let (w, h) = (64, 64);
        let rgb = make_pattern(w, h);
        let cw = w / 2;
        let ch = h / 2;
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];

        let luts = GammaLuts::srgb();
        let config = SharpYuvConfig::default();
        sharp::rgb_to_yuv420_sharp(
            &rgb, &mut y, &mut cb, &mut cr, w, h,
            Range::Full, Matrix::Bt601, &luts, &config,
        );

        // Y should match the non-sharp 4:4:4 Y exactly (same kernel).
        let mut y_ref = vec![0u8; w * h];
        let mut cb_ref = vec![0u8; w * h];
        let mut cr_ref = vec![0u8; w * h];
        rgb_to_yuv444(&rgb, &mut y_ref, &mut cb_ref, &mut cr_ref, w, h);
        // Fused scalar Y vs SIMD Y may differ by ±1 (different rounding).
        let y_max = max_abs_err(&y, &y_ref);
        assert!(y_max <= 1, "Y max err {y_max} > 1 between sharp and standard");

        // Cb/Cr from sharp should differ from simple box-average 4:2:0
        // (the whole point of iterative refinement).
        let mut cb_box = vec![0u8; cw * ch];
        let mut cr_box = vec![0u8; cw * ch];
        let mut y_box = vec![0u8; w * h];
        rgb_to_yuv420(&rgb, &mut y_box, &mut cb_box, &mut cr_box, w, h);

        // Sharp should NOT be identical to box (it refines).
        let cb_diff: usize = cb.iter().zip(cb_box.iter()).map(|(a, b)| a.abs_diff(*b) as usize).sum();
        let cr_diff: usize = cr.iter().zip(cr_box.iter()).map(|(a, b)| a.abs_diff(*b) as usize).sum();
        eprintln!("sharp vs box: cb_diff={cb_diff} cr_diff={cr_diff}");
        assert!(cb_diff > 0 || cr_diff > 0, "sharp should differ from box");
    }

    #[test]
    fn sharp_yuv_420_limited_range() {
        let (w, h) = (32, 32);
        let rgb = make_pattern(w, h);
        let cw = w / 2;
        let ch = h / 2;
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];

        let luts = GammaLuts::srgb();
        let config = SharpYuvConfig::default();
        sharp::rgb_to_yuv420_sharp(
            &rgb, &mut y, &mut cb, &mut cr, w, h,
            Range::Limited, Matrix::Bt601, &luts, &config,
        );

        // Limited range Y should be in [16, 235]
        for &yv in y.iter() {
            assert!(yv >= 16 && yv <= 235, "Y={yv} outside limited range");
        }
    }

    #[test]
    fn sharp_yuv_420_libwebp_gamma() {
        let (w, h) = (32, 32);
        let rgb = make_pattern(w, h);
        let cw = w / 2;
        let ch = h / 2;
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; cw * ch];
        let mut cr = vec![0u8; cw * ch];

        let luts = GammaLuts::libwebp();
        let config = SharpYuvConfig { srgb_delinearize: false, ..Default::default() };
        sharp::rgb_to_yuv420_sharp(
            &rgb, &mut y, &mut cb, &mut cr, w, h,
            Range::Limited, Matrix::Bt601, &luts, &config,
        );

        // Should not panic, output should be valid
        for &yv in y.iter() {
            assert!(yv >= 16 && yv <= 235, "Y={yv} outside limited range");
        }
    }

    /// Compare sharp YUV quality on real CID22 photos: measure reconstruction
    /// error for box-average, our sharp, and the yuv crate's sharp.
    #[test]
    fn sharp_yuv_quality_comparison() {
        let corpus_dir = std::path::Path::new(
            &std::env::var("HOME").unwrap_or_else(|_| "/home/lilith".into()),
        )
        .join("work/codec-eval/codec-corpus/CID22/CID22-512/training");

        let mut paths: Vec<_> = std::fs::read_dir(&corpus_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("png"))
            })
            .map(|e| e.path())
            .collect();
        paths.sort();
        paths.truncate(10);

        if paths.is_empty() {
            eprintln!("No CID22 corpus found, using synthetic pattern");
            paths.clear();
        }

        // Aggregate stats across all images.
        let mut total_box_sum = 0.0f64;
        let mut total_sharp_sum = 0.0f64;
        let mut total_yuv_sum = 0.0f64;
        let mut total_pixels = 0u64;
        let mut images_tested = 0u32;

        // If no corpus, fall back to synthetic.
        let synthetic;
        let test_images: Vec<(&[u8], usize, usize)> = if paths.is_empty() {
            synthetic = make_pattern(512, 512);
            vec![(&synthetic, 512, 512)]
        } else {
            vec![] // filled below
        };

        let mut loaded: Vec<(Vec<u8>, usize, usize)> = Vec::new();
        for p in &paths {
            if let Some((rgb, w, h)) = load_png_rgb(p) {
                loaded.push((rgb, w as usize, h as usize));
            }
        }
        let test_data: Vec<(&[u8], usize, usize)> = if loaded.is_empty() {
            test_images
        } else {
            loaded.iter().map(|(r, w, h)| (r.as_slice(), *w, *h)).collect()
        };

        eprintln!("=== Sharp YUV Quality Comparison (BT.601 Full, {} images) ===", test_data.len());
        eprintln!("{:>30} {:>8} {:>8} {:>8}", "image", "box", "sharp", "yuv_shp");

        for (rgb, w, h) in &test_data {
            let (w, h) = (*w, *h);
            let n = w * h;
            let cw = w / 2;
            let ch = h / 2;

            // Box average 4:2:0.
            let mut y_box = vec![0u8; n];
            let mut cb_box = vec![0u8; cw * ch];
            let mut cr_box = vec![0u8; cw * ch];
            rgb_to_yuv420(rgb, &mut y_box, &mut cb_box, &mut cr_box, w, h);
            let mut rt_box = vec![0u8; n * 3];
            yuv420_to_rgb(&y_box, &cb_box, &cr_box, &mut rt_box, w, h);
            let box_mean = mean_abs_err(rgb, &rt_box);

            // Our sharp.
            let mut y_sharp = vec![0u8; n];
            let mut cb_sharp = vec![0u8; cw * ch];
            let mut cr_sharp = vec![0u8; cw * ch];
            let luts = GammaLuts::srgb();
            let config = SharpYuvConfig::default();
            sharp::rgb_to_yuv420_sharp(
                rgb, &mut y_sharp, &mut cb_sharp, &mut cr_sharp, w, h,
                Range::Full, Matrix::Bt601, &luts, &config,
            );
            let mut rt_sharp = vec![0u8; n * 3];
            yuv420_to_rgb(&y_sharp, &cb_sharp, &cr_sharp, &mut rt_sharp, w, h);
            let sharp_mean = mean_abs_err(rgb, &rt_sharp);

            // yuv crate sharp.
            let mut ref_img = yuv::YuvPlanarImageMut::alloc(
                w as u32, h as u32,
                yuv::YuvChromaSubsampling::Yuv420,
            );
            yuv::rgb_to_sharp_yuv420(
                &mut ref_img,
                rgb,
                (w * 3) as u32,
                yuv::YuvRange::Full,
                yuv::YuvStandardMatrix::Bt601,
                yuv::SharpYuvGammaTransfer::Srgb,
            ).unwrap();
            let ry = ref_img.y_plane.borrow();
            let ru = ref_img.u_plane.borrow();
            let rv = ref_img.v_plane.borrow();
            let mut rt_yuv = vec![0u8; n * 3];
            yuv420_to_rgb(ry, ru, rv, &mut rt_yuv, w, h);
            let yuv_mean = mean_abs_err(rgb, &rt_yuv);

            let name = paths.get(images_tested as usize)
                .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
                .unwrap_or_else(|| "synthetic".into());
            eprintln!("{name:>30} {box_mean:8.4} {sharp_mean:8.4} {yuv_mean:8.4}");

            total_box_sum += box_mean * n as f64;
            total_sharp_sum += sharp_mean * n as f64;
            total_yuv_sum += yuv_mean * n as f64;
            total_pixels += n as u64;
            images_tested += 1;
        }

        let avg_box = total_box_sum / total_pixels as f64;
        let avg_sharp = total_sharp_sum / total_pixels as f64;
        let avg_yuv = total_yuv_sum / total_pixels as f64;
        eprintln!("{:>30} {avg_box:8.4} {avg_sharp:8.4} {avg_yuv:8.4}", "MEAN");
        eprintln!();
        eprintln!("sharp vs box: {:.2}%", (avg_sharp - avg_box) / avg_box * 100.0);
        eprintln!("sharp vs yuv: {:.2}%", (avg_sharp - avg_yuv) / avg_yuv * 100.0);
    }

    fn load_png_rgb(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
        let file = std::fs::File::open(path).ok()?;
        let dec = png::Decoder::new(std::io::BufReader::new(file));
        let mut reader = dec.read_info().ok()?;
        let mut buf = vec![0u8; reader.output_buffer_size()?];
        let info = reader.next_frame(&mut buf).ok()?;
        let rgb = match info.color_type {
            png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
            png::ColorType::Rgba => {
                let src = &buf[..info.buffer_size()];
                let mut out = Vec::with_capacity((info.width * info.height * 3) as usize);
                for c in src.chunks_exact(4) {
                    out.extend_from_slice(&c[..3]);
                }
                out
            }
            _ => return None,
        };
        Some((rgb, info.width, info.height))
    }
}
