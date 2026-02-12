//! Reduced IDCT kernels for shrink-on-load decoding.
//!
//! These kernels produce NxN output from the top-left NxN DCT coefficients,
//! skipping high-frequency data entirely. This is the "speed path" for
//! shrink-on-load: fewer multiplies, smaller output, no cross-block filtering.
//!
//! Reference: libjpeg-turbo `jidctred.c` (jpeg_idct_NxN functions).
//!
//! # Kernel summary
//!
//! | Scale | Output | Coefficients used | Operations |
//! |-------|--------|-------------------|------------|
//! | 1/8   | 1x1    | DC only           | 1 add + shift |
//! | 1/4   | 2x2    | Top-left 2x2      | 2-point butterfly × 2 |
//! | 1/2   | 4x4    | Top-left 4x4      | 4-point Loeffler × 2 |
//!
//! All kernels take dequantized coefficients in natural (raster) order.
//! The `_unclamped` variants skip the [0, 255] clamp for f32 output targets.

/// Fixed-point multiplication factor.
#[inline(always)]
const fn f(x: f32) -> i32 {
    (x * 4096.0 + 0.5) as i32
}

#[inline(always)]
const fn wa(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline(always)]
const fn ws(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline(always)]
const fn wm(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

#[inline(always)]
fn clamp_u8(a: i32) -> i16 {
    a.clamp(0, 255) as i16
}

// ============================================================================
// 1x1 (DC only) — 1/8 scale
// ============================================================================

/// 1x1 reduced IDCT: DC coefficient only.
///
/// Input: dequantized coefficients in natural order (only [0] is used).
/// Output: single pixel written at out[0].
///
/// Math: `pixel = (DC + 4) / 8 + 128`, clamped to [0, 255].
#[inline]
#[allow(dead_code)]
pub fn idct_scaled_1x1(dequant: &[i32; 64], out: &mut [i16], _stride: usize) {
    let dc = dequant[0];
    out[0] = wa(wa(dc, 4), 1024).wrapping_shr(3).clamp(0, 255) as i16;
}

/// 1x1 reduced IDCT without clamping (for f32 output targets).
#[inline]
#[allow(dead_code)]
pub fn idct_scaled_1x1_unclamped(dequant: &[i32; 64], out: &mut [i16], _stride: usize) {
    let dc = dequant[0];
    out[0] = wa(wa(dc, 4), 1024).wrapping_shr(3) as i16;
}

/// 1x1 DC-only fast path: takes raw DC*quant directly (skips dequant buffer).
///
/// Same math as `idct_int_dc_only` but writes 1 pixel instead of 8x8.
#[inline]
pub fn idct_scaled_1x1_from_dc(dc_coeff: i32, out: &mut [i16], _stride: usize) {
    out[0] = wa(wa(dc_coeff, 4), 1024).wrapping_shr(3).clamp(0, 255) as i16;
}

/// 1x1 DC-only fast path without clamping.
#[inline]
pub fn idct_scaled_1x1_from_dc_unclamped(dc_coeff: i32, out: &mut [i16], _stride: usize) {
    out[0] = wa(wa(dc_coeff, 4), 1024).wrapping_shr(3) as i16;
}

// ============================================================================
// 2x2 — 1/4 scale
// ============================================================================
//
// Uses top-left 2x2 frequencies: positions [0,0], [0,1], [1,0], [1,1] in
// natural order = indices 0, 1, 8, 9 in the 8x8 block.
//
// 2-point IDCT per axis:
//   even = DC
//   odd  = AC[1] * cos(pi/4) = AC[1] * 0.707...
//
//   out[0] = (even + odd)
//   out[1] = (even - odd)
//
// Reference: libjpeg-turbo jpeg_idct_2x2 in jidctred.c
//
// The 2-point IDCT is just a butterfly: for input [a, b]:
//   X[0] = a + b * C
//   X[1] = a - b * C
// where C = cos(pi/4) = sqrt(2)/2 ≈ 0.7071
//
// In fixed-point with 12-bit precision: C = f(0.707106781) = 2896

/// cos(pi/4) in 12-bit fixed point
const COS_PI_4: i32 = f(core::f32::consts::FRAC_1_SQRT_2); // 2896

/// Rounding + level shift for 2x2 IDCT.
/// The 2-point transform has different scaling than 8-point.
/// After column pass (>>10) and row pass (>>17), we need:
/// - Rounding for row pass: 65536 (1 << 16)
/// - Level shift: 128 << 17 = 16777216
const SCALE_2X2: i32 = 65536 + (128 << 17);

/// 2x2 reduced IDCT.
///
/// Input: dequantized coefficients in natural order (uses [0], [1], [8], [9]).
/// Output: 2x2 pixels written to out[0..2] and out[stride..stride+2].
#[inline]
pub fn idct_scaled_2x2(dequant: &[i32; 64], out: &mut [i16], stride: usize) {
    // Coefficients in natural order:
    // [0] = DC,      [1] = freq(0,1)
    // [8] = freq(1,0), [9] = freq(1,1)

    let dc = dequant[0];
    let c01 = dequant[1];
    let c10 = dequant[8];
    let c11 = dequant[9];

    // Column pass: 2-point IDCT on each of 2 columns
    // Column 0: inputs dc, c10
    let col0_even = dc << 12; // scale up to fixed-point
    let col0_odd = wm(c10, COS_PI_4);
    let tmp00 = wa(col0_even, col0_odd) >> 10; // row 0, col 0 intermediate
    let tmp10 = ws(col0_even, col0_odd) >> 10; // row 1, col 0 intermediate

    // Column 1: inputs c01, c11
    let col1_even = c01 << 12;
    let col1_odd = wm(c11, COS_PI_4);
    let tmp01 = wa(col1_even, col1_odd) >> 10; // row 0, col 1 intermediate
    let tmp11 = ws(col1_even, col1_odd) >> 10; // row 1, col 1 intermediate

    // Row pass: 2-point IDCT on each of 2 rows
    // Row 0: inputs tmp00, tmp01
    let row0_even = wa(tmp00 << 12, SCALE_2X2);
    let row0_odd = wm(tmp01, COS_PI_4);
    out[0] = clamp_u8(wa(row0_even, row0_odd) >> 17);
    out[1] = clamp_u8(ws(row0_even, row0_odd) >> 17);

    // Row 1: inputs tmp10, tmp11
    let row1_even = wa(tmp10 << 12, SCALE_2X2);
    let row1_odd = wm(tmp11, COS_PI_4);
    out[stride] = clamp_u8(wa(row1_even, row1_odd) >> 17);
    out[stride + 1] = clamp_u8(ws(row1_even, row1_odd) >> 17);
}

/// 2x2 reduced IDCT without clamping (for f32 output targets).
#[inline]
pub fn idct_scaled_2x2_unclamped(dequant: &[i32; 64], out: &mut [i16], stride: usize) {
    let dc = dequant[0];
    let c01 = dequant[1];
    let c10 = dequant[8];
    let c11 = dequant[9];

    let col0_even = dc << 12;
    let col0_odd = wm(c10, COS_PI_4);
    let tmp00 = wa(col0_even, col0_odd) >> 10;
    let tmp10 = ws(col0_even, col0_odd) >> 10;

    let col1_even = c01 << 12;
    let col1_odd = wm(c11, COS_PI_4);
    let tmp01 = wa(col1_even, col1_odd) >> 10;
    let tmp11 = ws(col1_even, col1_odd) >> 10;

    let row0_even = wa(tmp00 << 12, SCALE_2X2);
    let row0_odd = wm(tmp01, COS_PI_4);
    out[0] = wa(row0_even, row0_odd).wrapping_shr(17) as i16;
    out[1] = ws(row0_even, row0_odd).wrapping_shr(17) as i16;

    let row1_even = wa(tmp10 << 12, SCALE_2X2);
    let row1_odd = wm(tmp11, COS_PI_4);
    out[stride] = wa(row1_even, row1_odd).wrapping_shr(17) as i16;
    out[stride + 1] = ws(row1_even, row1_odd).wrapping_shr(17) as i16;
}

// ============================================================================
// 4x4 — 1/2 scale
// ============================================================================
//
// Uses top-left 4x4 frequencies: rows 0-3, columns 0-3 in natural order.
// That's indices [0..4], [8..12], [16..20], [24..28].
//
// 4-point Loeffler IDCT per axis.
// Reference: libjpeg-turbo jpeg_idct_4x4 in jidctred.c
//
// Constants:
//   cos(pi/8) = 0.92388 → f2f = 3784
//   cos(3pi/8) = 0.38268 → f2f = 1567
//   cos(pi/4) = 0.70711 → f2f = 2896
//   1.175876 (combined rotation constant) → f2f = 4816

/// cos(pi/8) ≈ 0.92388, in 12-bit fixed point
const COS_PI_8: i32 = f(0.92387953);
/// cos(3pi/8) ≈ 0.38268, in 12-bit fixed point
const COS_3PI_8: i32 = f(0.38268343);

/// 4x4 reduced IDCT.
///
/// Input: dequantized coefficients in natural order (uses top-left 4x4).
/// Output: 4x4 pixels written to out with given stride.
///
/// Based on libjpeg-turbo jpeg_idct_4x4 (jidctred.c).
#[inline]
pub fn idct_scaled_4x4(dequant: &[i32; 64], out: &mut [i16], stride: usize) {
    // We need a 4x4 intermediate buffer for column pass results.
    // Column pass processes 4 columns of 4 rows each.
    let mut tmp = [0i32; 16]; // 4 columns × 4 rows

    // Column pass: 4-point IDCT on each of 4 columns
    for col in 0..4 {
        let c0 = dequant[col]; // row 0
        let c1 = dequant[8 + col]; // row 1
        let c2 = dequant[16 + col]; // row 2
        let c3 = dequant[24 + col]; // row 3

        // Even part: butterfly on c0, c2
        let even_sum = wa(c0, c2) << 12;
        let even_diff = ws(c0, c2) << 12;

        // Odd part: rotation of (c1, c3) by pi/8
        let z1 = wa(c1, c3);
        let z2 = ws(c1, c3);
        let odd0 = wa(wm(z1, COS_3PI_8), wm(z2, COS_PI_8));
        let odd1 = ws(wm(z1, COS_PI_8), wm(z2, COS_3PI_8));

        // Combine
        tmp[col] = wa(even_sum, odd0) >> 10;
        tmp[4 + col] = wa(even_diff, odd1) >> 10;
        tmp[8 + col] = ws(even_diff, odd1) >> 10;
        tmp[12 + col] = ws(even_sum, odd0) >> 10;
    }

    // Rounding constant for row pass: includes level shift
    const ROW_SCALE: i32 = 65536 + (128 << 17);

    // Row pass: 4-point IDCT on each of 4 rows
    for row in 0..4 {
        let base = row * 4;
        let c0 = tmp[base];
        let c1 = tmp[base + 1];
        let c2 = tmp[base + 2];
        let c3 = tmp[base + 3];

        let even_sum = wa(wa(c0, c2) << 12, ROW_SCALE);
        let even_diff = wa(ws(c0, c2) << 12, ROW_SCALE);

        let z1 = wa(c1, c3);
        let z2 = ws(c1, c3);
        let odd0 = wa(wm(z1, COS_3PI_8), wm(z2, COS_PI_8));
        let odd1 = ws(wm(z1, COS_PI_8), wm(z2, COS_3PI_8));

        let out_row = &mut out[row * stride..];
        out_row[0] = clamp_u8(wa(even_sum, odd0) >> 17);
        out_row[1] = clamp_u8(wa(even_diff, odd1) >> 17);
        out_row[2] = clamp_u8(ws(even_diff, odd1) >> 17);
        out_row[3] = clamp_u8(ws(even_sum, odd0) >> 17);
    }
}

/// 4x4 reduced IDCT without clamping (for f32 output targets).
#[inline]
pub fn idct_scaled_4x4_unclamped(dequant: &[i32; 64], out: &mut [i16], stride: usize) {
    let mut tmp = [0i32; 16];

    for col in 0..4 {
        let c0 = dequant[col];
        let c1 = dequant[8 + col];
        let c2 = dequant[16 + col];
        let c3 = dequant[24 + col];

        let even_sum = wa(c0, c2) << 12;
        let even_diff = ws(c0, c2) << 12;

        let z1 = wa(c1, c3);
        let z2 = ws(c1, c3);
        let odd0 = wa(wm(z1, COS_3PI_8), wm(z2, COS_PI_8));
        let odd1 = ws(wm(z1, COS_PI_8), wm(z2, COS_3PI_8));

        tmp[col] = wa(even_sum, odd0) >> 10;
        tmp[4 + col] = wa(even_diff, odd1) >> 10;
        tmp[8 + col] = ws(even_diff, odd1) >> 10;
        tmp[12 + col] = ws(even_sum, odd0) >> 10;
    }

    const ROW_SCALE: i32 = 65536 + (128 << 17);

    for row in 0..4 {
        let base = row * 4;
        let c0 = tmp[base];
        let c1 = tmp[base + 1];
        let c2 = tmp[base + 2];
        let c3 = tmp[base + 3];

        let even_sum = wa(wa(c0, c2) << 12, ROW_SCALE);
        let even_diff = wa(ws(c0, c2) << 12, ROW_SCALE);

        let z1 = wa(c1, c3);
        let z2 = ws(c1, c3);
        let odd0 = wa(wm(z1, COS_3PI_8), wm(z2, COS_PI_8));
        let odd1 = ws(wm(z1, COS_PI_8), wm(z2, COS_3PI_8));

        let out_row = &mut out[row * stride..];
        out_row[0] = wa(even_sum, odd0).wrapping_shr(17) as i16;
        out_row[1] = wa(even_diff, odd1).wrapping_shr(17) as i16;
        out_row[2] = ws(even_diff, odd1).wrapping_shr(17) as i16;
        out_row[3] = ws(even_sum, odd0).wrapping_shr(17) as i16;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::idct_int::{idct_int_dc_only, idct_int_tiered};

    /// Helper: do full 8x8 IDCT then downsample to NxN by area averaging.
    fn full_idct_then_downsample(dequant: &mut [i32; 64], scale: usize) -> Vec<i16> {
        let mut full = [0i16; 64];
        let coeff_count = dequant
            .iter()
            .rposition(|&x| x != 0)
            .map(|p| (p + 1) as u8)
            .unwrap_or(0);

        if coeff_count <= 1 {
            idct_int_dc_only(dequant[0], &mut full, 8);
        } else {
            idct_int_tiered(dequant, &mut full, 8, coeff_count);
        }

        let block_size = 8 / scale;
        let mut result = vec![0i16; scale * scale];
        for oy in 0..scale {
            for ox in 0..scale {
                let mut sum = 0i32;
                for iy in 0..block_size {
                    for ix in 0..block_size {
                        sum += full[(oy * block_size + iy) * 8 + ox * block_size + ix] as i32;
                    }
                }
                result[oy * scale + ox] = (sum / (block_size * block_size) as i32) as i16;
            }
        }
        result
    }

    /// Create a simple test block with known coefficients.
    fn make_test_block(dc: i32, ac_values: &[(usize, i32)]) -> [i32; 64] {
        let mut block = [0i32; 64];
        block[0] = dc;
        for &(idx, val) in ac_values {
            block[idx] = val;
        }
        block
    }

    #[test]
    fn scaled_1x1_matches_dc_only() {
        // DC-only block: reduced 1x1 should produce same value as full DC-only
        // but as a single pixel instead of 8x8
        for dc in [0, 100, 500, 1000, -500, 2000] {
            let block = make_test_block(dc, &[]);
            let mut out_scaled = [0i16; 1];
            idct_scaled_1x1(&block, &mut out_scaled, 1);

            let mut full_out = [0i16; 64];
            idct_int_dc_only(dc, &mut full_out, 8);

            // 1x1 should equal the full DC-only value (they're all the same)
            assert_eq!(
                out_scaled[0], full_out[0],
                "1x1 mismatch for DC={dc}: scaled={}, full={}",
                out_scaled[0], full_out[0],
            );
        }
    }

    #[test]
    fn scaled_1x1_from_dc_matches() {
        for dc in [0, 100, 500, 1000, -500, 2000] {
            let block = make_test_block(dc, &[]);
            let mut out_buf = [0i16; 1];
            idct_scaled_1x1(&block, &mut out_buf, 1);

            let mut out_direct = [0i16; 1];
            idct_scaled_1x1_from_dc(dc, &mut out_direct, 1);

            assert_eq!(out_buf[0], out_direct[0], "DC={dc}");
        }
    }

    #[test]
    fn scaled_2x2_dc_only_uniform() {
        // DC-only: all 4 output pixels should be the same value
        let block = make_test_block(800, &[]);
        let mut out = [0i16; 4];
        idct_scaled_2x2(&block, &mut out, 2);

        assert_eq!(out[0], out[1], "row 0 not uniform");
        assert_eq!(out[2], out[3], "row 1 not uniform");
        assert_eq!(out[0], out[2], "rows not uniform");

        // Should match the DC-only 8x8 value
        let mut full = [0i16; 64];
        idct_int_dc_only(800, &mut full, 8);
        assert_eq!(out[0], full[0], "DC-only value mismatch");
    }

    #[test]
    fn scaled_4x4_dc_only_uniform() {
        // DC-only: all 16 output pixels should be the same value
        let block = make_test_block(600, &[]);
        let mut out = [0i16; 16];
        idct_scaled_4x4(&block, &mut out, 4);

        let expected = out[0];
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, expected, "pixel {i} differs: {v} != {expected}");
        }

        // Should match the DC-only 8x8 value
        let mut full = [0i16; 64];
        idct_int_dc_only(600, &mut full, 8);
        assert_eq!(out[0], full[0], "DC-only value mismatch");
    }

    #[test]
    fn scaled_2x2_with_ac_reasonable() {
        // Block with some AC content: reduced output should be close to
        // full IDCT + 4x4 area average (not exact due to different math).
        let mut block = make_test_block(
            800,
            &[
                (1, 50),  // freq(0,1)
                (8, -30), // freq(1,0)
                (9, 20),  // freq(1,1)
            ],
        );

        let mut out_scaled = [0i16; 4];
        idct_scaled_2x2(&block, &mut out_scaled, 2);

        // Check that we get reasonable output: all pixels in valid range
        for &v in &out_scaled {
            assert!((0..=255).contains(&v), "pixel out of range: {v}");
        }

        // Check that AC content produces variation (not all same)
        let has_variation = out_scaled.windows(2).any(|w| w[0] != w[1]);
        assert!(has_variation, "2x2 with AC should have pixel variation");

        // Compare with full IDCT + downsample. Allow tolerance since
        // reduced IDCT is mathematically different from full IDCT + area average.
        let reference = full_idct_then_downsample(&mut block, 2);
        for i in 0..4 {
            let diff = (out_scaled[i] as i32 - reference[i] as i32).abs();
            assert!(
                diff <= 5,
                "2x2 pixel {i}: scaled={} ref={} diff={diff}",
                out_scaled[i],
                reference[i],
            );
        }
    }

    #[test]
    fn scaled_4x4_with_ac_reasonable() {
        let mut block = make_test_block(
            700,
            &[
                (1, 40),   // freq(0,1)
                (2, -20),  // freq(0,2)
                (8, 30),   // freq(1,0)
                (9, -15),  // freq(1,1)
                (16, -25), // freq(2,0)
                (17, 10),  // freq(2,1)
                (24, 5),   // freq(3,0)
            ],
        );

        let mut out_scaled = [0i16; 16];
        idct_scaled_4x4(&block, &mut out_scaled, 4);

        // All pixels should be in valid range
        for (i, &v) in out_scaled.iter().enumerate() {
            assert!((0..=255).contains(&v), "pixel {i} out of range: {v}");
        }

        // Should have spatial variation
        let has_variation = out_scaled.windows(2).any(|w| w[0] != w[1]);
        assert!(has_variation, "4x4 with AC should have pixel variation");

        // Compare with full IDCT + downsample. Allow more tolerance since
        // reduced IDCT is a different transform than full IDCT + area average.
        let reference = full_idct_then_downsample(&mut block, 4);
        for i in 0..16 {
            let diff = (out_scaled[i] as i32 - reference[i] as i32).abs();
            assert!(
                diff <= 5,
                "4x4 pixel {i}: scaled={} ref={} diff={diff}",
                out_scaled[i],
                reference[i],
            );
        }
    }

    #[test]
    fn scaled_clamping_works() {
        // Very large DC: should clamp to 255
        let block = make_test_block(4000, &[]);
        let mut out1 = [0i16; 1];
        idct_scaled_1x1(&block, &mut out1, 1);
        assert_eq!(out1[0], 255);

        let mut out2 = [0i16; 4];
        idct_scaled_2x2(&block, &mut out2, 2);
        assert_eq!(out2[0], 255);

        let mut out4 = [0i16; 16];
        idct_scaled_4x4(&block, &mut out4, 4);
        assert_eq!(out4[0], 255);

        // Very negative DC: should clamp to 0
        let block_neg = make_test_block(-4000, &[]);
        idct_scaled_1x1(&block_neg, &mut out1, 1);
        assert_eq!(out1[0], 0);

        idct_scaled_2x2(&block_neg, &mut out2, 2);
        assert_eq!(out2[0], 0);

        idct_scaled_4x4(&block_neg, &mut out4, 4);
        assert_eq!(out4[0], 0);
    }

    #[test]
    fn scaled_unclamped_allows_out_of_range() {
        let block = make_test_block(4000, &[]);
        let mut out = [0i16; 1];
        idct_scaled_1x1_unclamped(&block, &mut out, 1);
        assert!(out[0] > 255, "unclamped should exceed 255: {}", out[0]);

        let block_neg = make_test_block(-4000, &[]);
        idct_scaled_1x1_unclamped(&block_neg, &mut out, 1);
        assert!(out[0] < 0, "unclamped should be negative: {}", out[0]);
    }

    #[test]
    fn scaled_stride_respected() {
        // 2x2 with stride > 2
        let block = make_test_block(500, &[(1, 30), (8, -20)]);
        let mut out = [0i16; 20]; // stride=10, 2 rows
        idct_scaled_2x2(&block, &mut out, 10);

        // Row 0 at offset 0,1; Row 1 at offset 10,11
        assert_ne!(out[0], 0);
        assert_ne!(out[10], 0);
        // Pixels between rows should be untouched (still 0)
        assert_eq!(out[2], 0);
        assert_eq!(out[5], 0);

        // 4x4 with stride > 4
        let mut out4 = [0i16; 100]; // stride=25, 4 rows
        idct_scaled_4x4(&block, &mut out4, 25);
        assert_ne!(out4[0], 0);
        assert_ne!(out4[25], 0);
        assert_ne!(out4[50], 0);
        assert_ne!(out4[75], 0);
    }
}
