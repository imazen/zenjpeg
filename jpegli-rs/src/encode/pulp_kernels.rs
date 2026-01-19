//! SIMD kernels using the pulp crate
//!
//! This module provides implementations using pulp's generic SIMD abstraction,
//! which provides proper runtime dispatch to the best available SIMD instruction set.
//!
//! Unlike `wide` + `multiversed`, pulp's dispatch actually changes the SIMD width
//! at runtime (SSE4=128-bit, AVX2=256-bit, AVX-512=512-bit).

use pulp::{Arch, Simd, WithSimd};

use crate::foundation::consts::DCT_BLOCK_SIZE;

// ============================================================================
// DCT Constants
// ============================================================================

// AAN DCT constants
const C_PI_4: f32 = core::f32::consts::FRAC_1_SQRT_2; // cos(π/4) = √2/2
const C_PI_8: f32 = 0.382_683_43; // sin(π/8)
const C2_M_C6: f32 = 0.541_196_1; // cos(π/8) - cos(3π/8)
const C2_P_C6: f32 = 1.306_563; // cos(π/8) + cos(3π/8)

/// AAN scale factors (reciprocals for multiplication instead of division)
const AAN_INV_SCALES: [f32; 8] = [
    1.0,
    1.0 / 1.387_039_9,
    1.0 / 1.306_563,
    1.0 / 1.175_875_5,
    1.0,
    1.0 / 0.785_694_96,
    1.0 / 0.541_196_1,
    1.0 / 0.275_899_38,
];

// ============================================================================
// Color Conversion Constants (BT.601)
// ============================================================================

const Y_R: f32 = 0.299;
const Y_G: f32 = 0.587;
const Y_B: f32 = 0.114;

const CB_R: f32 = -0.168736;
const CB_G: f32 = -0.331264;
const CB_B: f32 = 0.5;

const CR_R: f32 = 0.5;
const CR_G: f32 = -0.418688;
const CR_B: f32 = -0.081312;

// ============================================================================
// DCT Implementation
// ============================================================================

/// 2D AAN DCT on an 8x8 block using pulp runtime dispatch.
///
/// Uses `Arch::new().dispatch()` to select the best SIMD at runtime.
#[inline]
pub fn aan_dct_2d_pulp(input: &[f32; DCT_BLOCK_SIZE]) -> [f32; DCT_BLOCK_SIZE] {
    struct DctImpl<'a> {
        input: &'a [f32; DCT_BLOCK_SIZE],
    }

    impl WithSimd for DctImpl<'_> {
        type Output = [f32; DCT_BLOCK_SIZE];

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            aan_dct_2d_generic(simd, self.input)
        }
    }

    Arch::new().dispatch(DctImpl { input })
}

/// Generic 2D AAN DCT implementation
#[inline(always)]
fn aan_dct_2d_generic<S: Simd>(simd: S, input: &[f32; DCT_BLOCK_SIZE]) -> [f32; DCT_BLOCK_SIZE] {
    let mut data = *input;

    // Pass 1: Process rows
    for row in 0..8 {
        let offset = row * 8;
        let mut row_data: [f32; 8] = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ];
        aan_dct_1d_scalar(&mut row_data);
        data[offset..offset + 8].copy_from_slice(&row_data);
    }

    // Pass 2: Process columns
    for col in 0..8 {
        let mut col_data: [f32; 8] = [
            data[col],
            data[col + 8],
            data[col + 16],
            data[col + 24],
            data[col + 32],
            data[col + 40],
            data[col + 48],
            data[col + 56],
        ];
        aan_dct_1d_scalar(&mut col_data);
        data[col] = col_data[0];
        data[col + 8] = col_data[1];
        data[col + 16] = col_data[2];
        data[col + 24] = col_data[3];
        data[col + 32] = col_data[4];
        data[col + 40] = col_data[5];
        data[col + 48] = col_data[6];
        data[col + 56] = col_data[7];
    }

    // Apply 1/8 scaling using SIMD
    let scale = simd.splat_f32s(0.125);
    let (head, tail) = S::as_mut_simd_f32s(&mut data);
    for chunk in head.iter_mut() {
        *chunk = simd.mul_f32s(*chunk, scale);
    }
    for val in tail {
        *val *= 0.125;
    }

    data
}

/// 1D AAN DCT on a single row of 8 elements (scalar)
#[inline(always)]
fn aan_dct_1d_scalar(d: &mut [f32; 8]) {
    // Stage 1: Butterfly
    let tmp0 = d[0] + d[7];
    let tmp7 = d[0] - d[7];
    let tmp1 = d[1] + d[6];
    let tmp6 = d[1] - d[6];
    let tmp2 = d[2] + d[5];
    let tmp5 = d[2] - d[5];
    let tmp3 = d[3] + d[4];
    let tmp4 = d[3] - d[4];

    // Stage 2: Even part
    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    d[0] = tmp10 + tmp11;
    d[4] = tmp10 - tmp11;

    let z1 = (tmp12 + tmp13) * C_PI_4;
    d[2] = tmp13 + z1;
    d[6] = tmp13 - z1;

    // Stage 3: Odd part
    let tmp10 = tmp4 + tmp5;
    let tmp11 = tmp5 + tmp6;
    let tmp12 = tmp6 + tmp7;

    let z5 = (tmp10 - tmp12) * C_PI_8;
    let z2 = C2_M_C6 * tmp10 + z5;
    let z4 = C2_P_C6 * tmp12 + z5;
    let z3 = tmp11 * C_PI_4;

    let z11 = tmp7 + z3;
    let z13 = tmp7 - z3;

    d[5] = z13 + z2;
    d[3] = z13 - z2;
    d[1] = z11 + z4;
    d[7] = z11 - z4;

    // Descale
    d[0] *= AAN_INV_SCALES[0];
    d[1] *= AAN_INV_SCALES[1];
    d[2] *= AAN_INV_SCALES[2];
    d[3] *= AAN_INV_SCALES[3];
    d[4] *= AAN_INV_SCALES[4];
    d[5] *= AAN_INV_SCALES[5];
    d[6] *= AAN_INV_SCALES[6];
    d[7] *= AAN_INV_SCALES[7];
}

/// Batch DCT processing using pulp with ILP (instruction-level parallelism)
#[inline]
pub fn aan_dct_2d_batch_pulp(
    inputs: &[[f32; DCT_BLOCK_SIZE]],
    outputs: &mut [[f32; DCT_BLOCK_SIZE]],
) {
    struct BatchDctImpl<'a> {
        inputs: &'a [[f32; DCT_BLOCK_SIZE]],
        outputs: &'a mut [[f32; DCT_BLOCK_SIZE]],
    }

    impl WithSimd for BatchDctImpl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            for (input, output) in self.inputs.iter().zip(self.outputs.iter_mut()) {
                *output = aan_dct_2d_generic(simd, input);
            }
        }
    }

    Arch::new().dispatch(BatchDctImpl { inputs, outputs });
}

// ============================================================================
// Color Conversion Implementation
// ============================================================================

/// RGB to YCbCr planar conversion using pulp runtime dispatch.
///
/// Uses ILP (4 vectors at a time) for better throughput.
#[inline]
pub fn rgb_to_ycbcr_planar_pulp(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
) {
    struct ColorImpl<'a> {
        r: &'a [f32],
        g: &'a [f32],
        b: &'a [f32],
        y: &'a mut [f32],
        cb: &'a mut [f32],
        cr: &'a mut [f32],
    }

    impl WithSimd for ColorImpl<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            rgb_to_ycbcr_generic(simd, self.r, self.g, self.b, self.y, self.cb, self.cr);
        }
    }

    Arch::new().dispatch(ColorImpl { r, g, b, y, cb, cr });
}

/// Generic RGB to YCbCr implementation with ILP
#[inline(always)]
fn rgb_to_ycbcr_generic<S: Simd>(
    simd: S,
    r: &[f32],
    g: &[f32],
    b: &[f32],
    y: &mut [f32],
    cb: &mut [f32],
    cr: &mut [f32],
) {
    // Splat constants
    let yr = simd.splat_f32s(Y_R);
    let yg = simd.splat_f32s(Y_G);
    let yb = simd.splat_f32s(Y_B);

    let cbr = simd.splat_f32s(CB_R);
    let cbg = simd.splat_f32s(CB_G);
    let cbb = simd.splat_f32s(CB_B);

    let crr = simd.splat_f32s(CR_R);
    let crg = simd.splat_f32s(CR_G);
    let crb = simd.splat_f32s(CR_B);

    let bias = simd.splat_f32s(128.0);

    // Process SIMD chunks
    let (r_simd, r_tail) = S::as_simd_f32s(r);
    let (g_simd, g_tail) = S::as_simd_f32s(g);
    let (b_simd, b_tail) = S::as_simd_f32s(b);
    let (y_simd, y_tail) = S::as_mut_simd_f32s(y);
    let (cb_simd, cb_tail) = S::as_mut_simd_f32s(cb);
    let (cr_simd, cr_tail) = S::as_mut_simd_f32s(cr);

    // ILP: process 4 vectors at a time when possible
    let (r4, r1) = pulp::as_arrays::<4, _>(r_simd);
    let (g4, g1) = pulp::as_arrays::<4, _>(g_simd);
    let (b4, b1) = pulp::as_arrays::<4, _>(b_simd);
    let (y4, y1) = pulp::as_arrays_mut::<4, _>(y_simd);
    let (cb4, cb1) = pulp::as_arrays_mut::<4, _>(cb_simd);
    let (cr4, cr1) = pulp::as_arrays_mut::<4, _>(cr_simd);

    for (
        ((([r0, r1, r2, r3], [g0, g1, g2, g3]), [b0, b1, b2, b3]), [y0, y1, y2, y3]),
        ([cb0, cb1, cb2, cb3], [cr0, cr1, cr2, cr3]),
    ) in r4
        .iter()
        .zip(g4.iter())
        .zip(b4.iter())
        .zip(y4.iter_mut())
        .zip(cb4.iter_mut().zip(cr4.iter_mut()))
    {
        // Y = R*Yr + G*Yg + B*Yb (FMA chain)
        *y0 = simd.mul_add_f32s(*r0, yr, simd.mul_add_f32s(*g0, yg, simd.mul_f32s(*b0, yb)));
        *y1 = simd.mul_add_f32s(*r1, yr, simd.mul_add_f32s(*g1, yg, simd.mul_f32s(*b1, yb)));
        *y2 = simd.mul_add_f32s(*r2, yr, simd.mul_add_f32s(*g2, yg, simd.mul_f32s(*b2, yb)));
        *y3 = simd.mul_add_f32s(*r3, yr, simd.mul_add_f32s(*g3, yg, simd.mul_f32s(*b3, yb)));

        // Cb = 128 + R*CBr + G*CBg + B*CBb
        *cb0 = simd.mul_add_f32s(
            *r0,
            cbr,
            simd.mul_add_f32s(*g0, cbg, simd.mul_add_f32s(*b0, cbb, bias)),
        );
        *cb1 = simd.mul_add_f32s(
            *r1,
            cbr,
            simd.mul_add_f32s(*g1, cbg, simd.mul_add_f32s(*b1, cbb, bias)),
        );
        *cb2 = simd.mul_add_f32s(
            *r2,
            cbr,
            simd.mul_add_f32s(*g2, cbg, simd.mul_add_f32s(*b2, cbb, bias)),
        );
        *cb3 = simd.mul_add_f32s(
            *r3,
            cbr,
            simd.mul_add_f32s(*g3, cbg, simd.mul_add_f32s(*b3, cbb, bias)),
        );

        // Cr = 128 + R*CRr + G*CRg + B*CRb
        *cr0 = simd.mul_add_f32s(
            *r0,
            crr,
            simd.mul_add_f32s(*g0, crg, simd.mul_add_f32s(*b0, crb, bias)),
        );
        *cr1 = simd.mul_add_f32s(
            *r1,
            crr,
            simd.mul_add_f32s(*g1, crg, simd.mul_add_f32s(*b1, crb, bias)),
        );
        *cr2 = simd.mul_add_f32s(
            *r2,
            crr,
            simd.mul_add_f32s(*g2, crg, simd.mul_add_f32s(*b2, crb, bias)),
        );
        *cr3 = simd.mul_add_f32s(
            *r3,
            crr,
            simd.mul_add_f32s(*g3, crg, simd.mul_add_f32s(*b3, crb, bias)),
        );
    }

    // Process remaining SIMD vectors
    for ((((rv, gv), bv), yv), (cbv, crv)) in r1
        .iter()
        .zip(g1.iter())
        .zip(b1.iter())
        .zip(y1.iter_mut())
        .zip(cb1.iter_mut().zip(cr1.iter_mut()))
    {
        *yv = simd.mul_add_f32s(*rv, yr, simd.mul_add_f32s(*gv, yg, simd.mul_f32s(*bv, yb)));
        *cbv = simd.mul_add_f32s(
            *rv,
            cbr,
            simd.mul_add_f32s(*gv, cbg, simd.mul_add_f32s(*bv, cbb, bias)),
        );
        *crv = simd.mul_add_f32s(
            *rv,
            crr,
            simd.mul_add_f32s(*gv, crg, simd.mul_add_f32s(*bv, crb, bias)),
        );
    }

    // Handle scalar remainder
    for i in 0..r_tail.len() {
        let ri = r_tail[i];
        let gi = g_tail[i];
        let bi = b_tail[i];
        y_tail[i] = ri * Y_R + gi * Y_G + bi * Y_B;
        cb_tail[i] = 128.0 + ri * CB_R + gi * CB_G + bi * CB_B;
        cr_tail[i] = 128.0 + ri * CR_R + gi * CR_G + bi * CR_B;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_only() {
        let input = [128.0f32; DCT_BLOCK_SIZE];
        let output = aan_dct_2d_pulp(&input);

        // DC coefficient should be non-zero
        assert!(output[0].abs() > 1.0, "DC should be non-zero");

        // AC coefficients should be near zero
        for (i, &val) in output.iter().enumerate().skip(1) {
            assert!(val.abs() < 0.01, "AC[{}] = {} should be ~0", i, val);
        }
    }

    #[test]
    fn test_dct_matches_scalar_aan() {
        use crate::encode::dct::aan_forward_dct_8x8;

        let input: [f32; DCT_BLOCK_SIZE] =
            core::array::from_fn(|i| ((i * 7 + 13) % 256) as f32 - 128.0);

        let scalar_out = aan_forward_dct_8x8(&input);
        let pulp_out = aan_dct_2d_pulp(&input);

        for i in 0..DCT_BLOCK_SIZE {
            assert!(
                (scalar_out[i] - pulp_out[i]).abs() < 0.01,
                "Mismatch at {}: scalar={}, pulp={}",
                i,
                scalar_out[i],
                pulp_out[i]
            );
        }
    }

    #[test]
    fn test_rgb_to_ycbcr_gray() {
        // Gray (R=G=B=128) should give Y≈128, Cb≈128, Cr≈128
        let r = vec![128.0f32; 64];
        let g = vec![128.0f32; 64];
        let b = vec![128.0f32; 64];

        let mut y = vec![0.0f32; 64];
        let mut cb = vec![0.0f32; 64];
        let mut cr = vec![0.0f32; 64];

        rgb_to_ycbcr_planar_pulp(&r, &g, &b, &mut y, &mut cb, &mut cr);

        for i in 0..64 {
            assert!((y[i] - 128.0).abs() < 0.01, "Y[{}] = {}", i, y[i]);
            assert!((cb[i] - 128.0).abs() < 0.01, "Cb[{}] = {}", i, cb[i]);
            assert!((cr[i] - 128.0).abs() < 0.01, "Cr[{}] = {}", i, cr[i]);
        }
    }

    #[test]
    fn test_batch_dct() {
        let inputs: Vec<[f32; DCT_BLOCK_SIZE]> = (0..16)
            .map(|b| core::array::from_fn(|i| ((i + b * 7) % 256) as f32 - 128.0))
            .collect();
        let mut outputs = vec![[0.0f32; DCT_BLOCK_SIZE]; 16];

        aan_dct_2d_batch_pulp(&inputs, &mut outputs);

        // Verify each output matches individual processing
        for (input, output) in inputs.iter().zip(outputs.iter()) {
            let expected = aan_dct_2d_pulp(input);
            for i in 0..DCT_BLOCK_SIZE {
                assert!(
                    (expected[i] - output[i]).abs() < 0.001,
                    "Batch mismatch at {}: expected={}, got={}",
                    i,
                    expected[i],
                    output[i]
                );
            }
        }
    }
}
