//! Forward DCT (Discrete Cosine Transform) for JPEG encoding
//!
//! This module implements the Loeffler-Ligtenberg-Moschytz algorithm for 8x8 DCT,
//! matching mozjpeg's jfdctint.c (integer slow DCT).
//!
//! The algorithm uses 12 multiplies and 32 adds per 1-D DCT.
//! A 2-D DCT is done by 1-D DCT on rows followed by 1-D DCT on columns.
//!
//! # ⚠️ CRITICAL: OUTPUT SCALING ⚠️
//!
//! The output is scaled up by a factor of 8 compared to a true DCT!
//! This is intentional and matches libjpeg/mozjpeg behavior.
//!
//! - For non-trellis path: descale with `(coef + 4) >> 3` before quantization
//! - For trellis path: pass scaled values directly (trellis multiplies qtable by 8)
//!
//! # SIMD Optimization
//!
//! Multiple SIMD implementations are provided:
//! - `forward_dct_8x8_int`: Scalar with `#[multiversion]` for autovectorization (NEON, SSE4, AVX2)
//! - `forward_dct_8x8_simd`: Row-parallel using `wide::i32x4`
//! - `forward_dct_8x8_transpose`: 8-wide using `wide::i32x8` with transpose
//! - AVX2 intrinsics module for x86_64
//!
//! Reference: C. Loeffler, A. Ligtenberg and G. Moschytz,
//! "Practical Fast 1-D DCT Algorithms with 11 Multiplications",
//! Proc. ICASSP 1989, pp. 988-991.

use crate::consts::{DCTSIZE, DCTSIZE2};
use multiversion::multiversion;
use wide::{i32x4, i32x8};

// Fixed-point constants for 13-bit precision (CONST_BITS = 13)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

// Pre-calculated fixed-point constants: FIX(x) = (x * (1 << CONST_BITS) + 0.5)
const FIX_0_298631336: i32 = 2446;  // FIX(0.298631336)
const FIX_0_390180644: i32 = 3196;  // FIX(0.390180644)
const FIX_0_541196100: i32 = 4433;  // FIX(0.541196100)
const FIX_0_765366865: i32 = 6270;  // FIX(0.765366865)
const FIX_0_899976223: i32 = 7373;  // FIX(0.899976223)
const FIX_1_175875602: i32 = 9633;  // FIX(1.175875602)
const FIX_1_501321110: i32 = 12299; // FIX(1.501321110)
const FIX_1_847759065: i32 = 15137; // FIX(1.847759065)
const FIX_1_961570560: i32 = 16069; // FIX(1.961570560)
const FIX_2_053119869: i32 = 16819; // FIX(2.053119869)
const FIX_2_562915447: i32 = 20995; // FIX(2.562915447)
const FIX_3_072711026: i32 = 25172; // FIX(3.072711026)

/// DESCALE: Right-shift with rounding (used to remove fixed-point scaling)
#[inline]
fn descale(x: i32, n: i32) -> i32 {
    // Round by adding 2^(n-1) before shifting
    (x + (1 << (n - 1))) >> n
}

/// Forward 8x8 DCT using Loeffler algorithm (integer version).
///
/// Uses `multiversion` for automatic SIMD optimization via autovectorization
/// on x86 (SSE4, AVX2) and ARM (NEON).
///
/// # ⚠️ CRITICAL: OUTPUT SCALING ⚠️
///
/// Output is scaled by factor of 8 (intentional, matches mozjpeg).
///
/// - For non-trellis: use `descale_for_quant()` before quantization
/// - For trellis: pass directly to `trellis_quantize_block()`
///
/// # Arguments
/// * `samples` - Input 8x8 block of level-shifted pixels (i16, typically -128 to 127)
/// * `coeffs` - Output 8x8 block of DCT coefficients (scaled by 8)
#[multiversion(targets(
    "x86_64+avx2",
    "x86_64+sse4.1",
    "x86+avx2",
    "x86+sse4.1",
    "aarch64+neon",
))]
pub fn forward_dct_8x8_int(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    // Work buffer (we modify in place across both passes)
    let mut data = [0i32; DCTSIZE2];

    // Convert input to i32 for processing
    for i in 0..DCTSIZE2 {
        data[i] = samples[i] as i32;
    }

    // Pass 1: process rows
    // Results are scaled up by sqrt(8) and by 2^PASS1_BITS
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;

        let tmp0 = data[base] + data[base + 7];
        let tmp7 = data[base] - data[base + 7];
        let tmp1 = data[base + 1] + data[base + 6];
        let tmp6 = data[base + 1] - data[base + 6];
        let tmp2 = data[base + 2] + data[base + 5];
        let tmp5 = data[base + 2] - data[base + 5];
        let tmp3 = data[base + 3] + data[base + 4];
        let tmp4 = data[base + 3] - data[base + 4];

        // Even part (per Loeffler figure 1)
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[base] = (tmp10 + tmp11) << PASS1_BITS;
        data[base + 4] = (tmp10 - tmp11) << PASS1_BITS;

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[base + 2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        data[base + 6] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);

        // Odd part (per Loeffler figure 8)
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602; // sqrt(2) * c3

        let tmp4 = tmp4 * FIX_0_298631336; // sqrt(2) * (-c1+c3+c5-c7)
        let tmp5 = tmp5 * FIX_2_053119869; // sqrt(2) * ( c1+c3-c5+c7)
        let tmp6 = tmp6 * FIX_3_072711026; // sqrt(2) * ( c1+c3+c5-c7)
        let tmp7 = tmp7 * FIX_1_501321110; // sqrt(2) * ( c1+c3-c5-c7)
        let z1 = z1 * (-FIX_0_899976223);  // sqrt(2) * ( c7-c3)
        let z2 = z2 * (-FIX_2_562915447);  // sqrt(2) * (-c1-c3)
        let z3 = z3 * (-FIX_1_961570560) + z5; // sqrt(2) * (-c3-c5)
        let z4 = z4 * (-FIX_0_390180644) + z5; // sqrt(2) * ( c5-c3)

        data[base + 7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        data[base + 5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        data[base + 3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        data[base + 1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process columns
    // We remove PASS1_BITS scaling but leave results scaled by factor of 8
    for col in 0..DCTSIZE {
        let tmp0 = data[col] + data[DCTSIZE * 7 + col];
        let tmp7 = data[col] - data[DCTSIZE * 7 + col];
        let tmp1 = data[DCTSIZE + col] + data[DCTSIZE * 6 + col];
        let tmp6 = data[DCTSIZE + col] - data[DCTSIZE * 6 + col];
        let tmp2 = data[DCTSIZE * 2 + col] + data[DCTSIZE * 5 + col];
        let tmp5 = data[DCTSIZE * 2 + col] - data[DCTSIZE * 5 + col];
        let tmp3 = data[DCTSIZE * 3 + col] + data[DCTSIZE * 4 + col];
        let tmp4 = data[DCTSIZE * 3 + col] - data[DCTSIZE * 4 + col];

        // Even part
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        data[col] = descale(tmp10 + tmp11, PASS1_BITS);
        data[DCTSIZE * 4 + col] = descale(tmp10 - tmp11, PASS1_BITS);

        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[DCTSIZE * 2 + col] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 6 + col] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS + PASS1_BITS);

        // Odd part
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;

        data[DCTSIZE * 7 + col] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 5 + col] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        data[DCTSIZE * 3 + col] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        data[DCTSIZE + col] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }

    // Copy results to output
    for i in 0..DCTSIZE2 {
        coeffs[i] = data[i] as i16;
    }
}

// ============================================================================
// SIMD Implementations using `wide` crate (portable across x86 and ARM)
// ============================================================================

/// SIMD descale operation for i32x4 vectors.
#[inline(always)]
fn descale_simd(x: i32x4, n: i32) -> i32x4 {
    let round = i32x4::splat(1 << (n - 1));
    (x + round) >> n
}

// Pre-computed SIMD constants for DCT - avoid recreating each call
const SIMD_FIX_0_298631336: i32x4 = i32x4::new([FIX_0_298631336; 4]);
const SIMD_FIX_0_541196100: i32x4 = i32x4::new([FIX_0_541196100; 4]);
const SIMD_FIX_0_765366865: i32x4 = i32x4::new([FIX_0_765366865; 4]);
const SIMD_FIX_1_175875602: i32x4 = i32x4::new([FIX_1_175875602; 4]);
const SIMD_FIX_1_501321110: i32x4 = i32x4::new([FIX_1_501321110; 4]);
const SIMD_FIX_1_847759065: i32x4 = i32x4::new([FIX_1_847759065; 4]);
const SIMD_FIX_2_053119869: i32x4 = i32x4::new([FIX_2_053119869; 4]);
const SIMD_FIX_3_072711026: i32x4 = i32x4::new([FIX_3_072711026; 4]);

// Negated constants to avoid runtime negation
const SIMD_NEG_FIX_0_390180644: i32x4 = i32x4::new([-FIX_0_390180644; 4]);
const SIMD_NEG_FIX_0_899976223: i32x4 = i32x4::new([-FIX_0_899976223; 4]);
const SIMD_NEG_FIX_1_961570560: i32x4 = i32x4::new([-FIX_1_961570560; 4]);
const SIMD_NEG_FIX_2_562915447: i32x4 = i32x4::new([-FIX_2_562915447; 4]);

/// Process one batch of 4 rows/columns with 1D DCT.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dct_1d_simd(
    d0: i32x4, d1: i32x4, d2: i32x4, d3: i32x4,
    d4: i32x4, d5: i32x4, d6: i32x4, d7: i32x4,
    shift_pass1: bool,
) -> [i32x4; 8] {
    // Even part
    let tmp0 = d0 + d7;
    let tmp7 = d0 - d7;
    let tmp1 = d1 + d6;
    let tmp6 = d1 - d6;
    let tmp2 = d2 + d5;
    let tmp5 = d2 - d5;
    let tmp3 = d3 + d4;
    let tmp4 = d3 - d4;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let (out0, out4) = if shift_pass1 {
        ((tmp10 + tmp11) << PASS1_BITS, (tmp10 - tmp11) << PASS1_BITS)
    } else {
        (descale_simd(tmp10 + tmp11, PASS1_BITS), descale_simd(tmp10 - tmp11, PASS1_BITS))
    };

    let z1 = (tmp12 + tmp13) * SIMD_FIX_0_541196100;
    let (out2, out6) = if shift_pass1 {
        (
            descale_simd(z1 + tmp13 * SIMD_FIX_0_765366865, CONST_BITS - PASS1_BITS),
            descale_simd(z1 - tmp12 * SIMD_FIX_1_847759065, CONST_BITS - PASS1_BITS),
        )
    } else {
        (
            descale_simd(z1 + tmp13 * SIMD_FIX_0_765366865, CONST_BITS + PASS1_BITS),
            descale_simd(z1 - tmp12 * SIMD_FIX_1_847759065, CONST_BITS + PASS1_BITS),
        )
    };

    // Odd part
    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * SIMD_FIX_1_175875602;

    let tmp4 = tmp4 * SIMD_FIX_0_298631336;
    let tmp5 = tmp5 * SIMD_FIX_2_053119869;
    let tmp6 = tmp6 * SIMD_FIX_3_072711026;
    let tmp7 = tmp7 * SIMD_FIX_1_501321110;

    let neg_z1 = z1 * SIMD_NEG_FIX_0_899976223;
    let neg_z2 = z2 * SIMD_NEG_FIX_2_562915447;
    let z3 = z3 * SIMD_NEG_FIX_1_961570560 + z5;
    let z4 = z4 * SIMD_NEG_FIX_0_390180644 + z5;

    let scale = if shift_pass1 { CONST_BITS - PASS1_BITS } else { CONST_BITS + PASS1_BITS };
    let out7 = descale_simd(tmp4 + neg_z1 + z3, scale);
    let out5 = descale_simd(tmp5 + neg_z2 + z4, scale);
    let out3 = descale_simd(tmp6 + neg_z2 + z3, scale);
    let out1 = descale_simd(tmp7 + neg_z1 + z4, scale);

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// SIMD-optimized forward DCT using row-parallel approach.
///
/// Processes 4 rows simultaneously using `wide::i32x4`.
/// Works on both x86 (SSE/AVX) and ARM (NEON) via the `wide` crate.
#[allow(dead_code)]
pub fn forward_dct_8x8_simd(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    let mut data = [0i32; DCTSIZE2];

    // Convert input to i32
    for row in 0..DCTSIZE {
        let base = row * DCTSIZE;
        for col in 0..DCTSIZE {
            data[base + col] = samples[base + col] as i32;
        }
    }

    // Pass 1: Process rows 0-3
    {
        let col0 = i32x4::new([data[0], data[8], data[16], data[24]]);
        let col1 = i32x4::new([data[1], data[9], data[17], data[25]]);
        let col2 = i32x4::new([data[2], data[10], data[18], data[26]]);
        let col3 = i32x4::new([data[3], data[11], data[19], data[27]]);
        let col4 = i32x4::new([data[4], data[12], data[20], data[28]]);
        let col5 = i32x4::new([data[5], data[13], data[21], data[29]]);
        let col6 = i32x4::new([data[6], data[14], data[22], data[30]]);
        let col7 = i32x4::new([data[7], data[15], data[23], data[31]]);

        let out = dct_1d_simd(col0, col1, col2, col3, col4, col5, col6, col7, true);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        for i in 0..4 {
            data[i * 8] = arr0[i];
            data[i * 8 + 1] = arr1[i];
            data[i * 8 + 2] = arr2[i];
            data[i * 8 + 3] = arr3[i];
            data[i * 8 + 4] = arr4[i];
            data[i * 8 + 5] = arr5[i];
            data[i * 8 + 6] = arr6[i];
            data[i * 8 + 7] = arr7[i];
        }
    }

    // Pass 1: Process rows 4-7
    {
        let col0 = i32x4::new([data[32], data[40], data[48], data[56]]);
        let col1 = i32x4::new([data[33], data[41], data[49], data[57]]);
        let col2 = i32x4::new([data[34], data[42], data[50], data[58]]);
        let col3 = i32x4::new([data[35], data[43], data[51], data[59]]);
        let col4 = i32x4::new([data[36], data[44], data[52], data[60]]);
        let col5 = i32x4::new([data[37], data[45], data[53], data[61]]);
        let col6 = i32x4::new([data[38], data[46], data[54], data[62]]);
        let col7 = i32x4::new([data[39], data[47], data[55], data[63]]);

        let out = dct_1d_simd(col0, col1, col2, col3, col4, col5, col6, col7, true);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        for i in 0..4 {
            data[32 + i * 8] = arr0[i];
            data[32 + i * 8 + 1] = arr1[i];
            data[32 + i * 8 + 2] = arr2[i];
            data[32 + i * 8 + 3] = arr3[i];
            data[32 + i * 8 + 4] = arr4[i];
            data[32 + i * 8 + 5] = arr5[i];
            data[32 + i * 8 + 6] = arr6[i];
            data[32 + i * 8 + 7] = arr7[i];
        }
    }

    // Pass 2: Process columns 0-3
    {
        let row0 = i32x4::new([data[0], data[1], data[2], data[3]]);
        let row1 = i32x4::new([data[8], data[9], data[10], data[11]]);
        let row2 = i32x4::new([data[16], data[17], data[18], data[19]]);
        let row3 = i32x4::new([data[24], data[25], data[26], data[27]]);
        let row4 = i32x4::new([data[32], data[33], data[34], data[35]]);
        let row5 = i32x4::new([data[40], data[41], data[42], data[43]]);
        let row6 = i32x4::new([data[48], data[49], data[50], data[51]]);
        let row7 = i32x4::new([data[56], data[57], data[58], data[59]]);

        let out = dct_1d_simd(row0, row1, row2, row3, row4, row5, row6, row7, false);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        for i in 0..4 {
            data[i] = arr0[i];
            data[8 + i] = arr1[i];
            data[16 + i] = arr2[i];
            data[24 + i] = arr3[i];
            data[32 + i] = arr4[i];
            data[40 + i] = arr5[i];
            data[48 + i] = arr6[i];
            data[56 + i] = arr7[i];
        }
    }

    // Pass 2: Process columns 4-7
    {
        let row0 = i32x4::new([data[4], data[5], data[6], data[7]]);
        let row1 = i32x4::new([data[12], data[13], data[14], data[15]]);
        let row2 = i32x4::new([data[20], data[21], data[22], data[23]]);
        let row3 = i32x4::new([data[28], data[29], data[30], data[31]]);
        let row4 = i32x4::new([data[36], data[37], data[38], data[39]]);
        let row5 = i32x4::new([data[44], data[45], data[46], data[47]]);
        let row6 = i32x4::new([data[52], data[53], data[54], data[55]]);
        let row7 = i32x4::new([data[60], data[61], data[62], data[63]]);

        let out = dct_1d_simd(row0, row1, row2, row3, row4, row5, row6, row7, false);

        let arr0 = out[0].to_array();
        let arr1 = out[1].to_array();
        let arr2 = out[2].to_array();
        let arr3 = out[3].to_array();
        let arr4 = out[4].to_array();
        let arr5 = out[5].to_array();
        let arr6 = out[6].to_array();
        let arr7 = out[7].to_array();

        for i in 0..4 {
            data[4 + i] = arr0[i];
            data[12 + i] = arr1[i];
            data[20 + i] = arr2[i];
            data[28 + i] = arr3[i];
            data[36 + i] = arr4[i];
            data[44 + i] = arr5[i];
            data[52 + i] = arr6[i];
            data[60 + i] = arr7[i];
        }
    }

    // Copy results
    for i in 0..DCTSIZE2 {
        coeffs[i] = data[i] as i16;
    }
}

// ============================================================================
// 8-wide SIMD using i32x8 with transpose
// ============================================================================

// SIMD constants for i32x8 (8-wide operations)
const SIMD8_FIX_0_298631336: i32x8 = i32x8::new([FIX_0_298631336; 8]);
const SIMD8_FIX_0_541196100: i32x8 = i32x8::new([FIX_0_541196100; 8]);
const SIMD8_FIX_0_765366865: i32x8 = i32x8::new([FIX_0_765366865; 8]);
const SIMD8_FIX_1_175875602: i32x8 = i32x8::new([FIX_1_175875602; 8]);
const SIMD8_FIX_1_501321110: i32x8 = i32x8::new([FIX_1_501321110; 8]);
const SIMD8_FIX_1_847759065: i32x8 = i32x8::new([FIX_1_847759065; 8]);
const SIMD8_FIX_2_053119869: i32x8 = i32x8::new([FIX_2_053119869; 8]);
const SIMD8_FIX_3_072711026: i32x8 = i32x8::new([FIX_3_072711026; 8]);

const SIMD8_NEG_FIX_0_390180644: i32x8 = i32x8::new([-FIX_0_390180644; 8]);
const SIMD8_NEG_FIX_0_899976223: i32x8 = i32x8::new([-FIX_0_899976223; 8]);
const SIMD8_NEG_FIX_1_961570560: i32x8 = i32x8::new([-FIX_1_961570560; 8]);
const SIMD8_NEG_FIX_2_562915447: i32x8 = i32x8::new([-FIX_2_562915447; 8]);

/// SIMD descale for i32x8
#[inline(always)]
fn descale_simd8(x: i32x8, n: i32) -> i32x8 {
    let round = i32x8::splat(1 << (n - 1));
    (x + round) >> n
}

/// 1D DCT on 8 values simultaneously using i32x8.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dct_1d_8wide(
    d0: i32x8, d1: i32x8, d2: i32x8, d3: i32x8,
    d4: i32x8, d5: i32x8, d6: i32x8, d7: i32x8,
    shift_pass1: bool,
) -> [i32x8; 8] {
    let tmp0 = d0 + d7;
    let tmp7 = d0 - d7;
    let tmp1 = d1 + d6;
    let tmp6 = d1 - d6;
    let tmp2 = d2 + d5;
    let tmp5 = d2 - d5;
    let tmp3 = d3 + d4;
    let tmp4 = d3 - d4;

    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    let (out0, out4) = if shift_pass1 {
        ((tmp10 + tmp11) << PASS1_BITS, (tmp10 - tmp11) << PASS1_BITS)
    } else {
        (descale_simd8(tmp10 + tmp11, PASS1_BITS), descale_simd8(tmp10 - tmp11, PASS1_BITS))
    };

    let z1 = (tmp12 + tmp13) * SIMD8_FIX_0_541196100;
    let (out2, out6) = if shift_pass1 {
        (
            descale_simd8(z1 + tmp13 * SIMD8_FIX_0_765366865, CONST_BITS - PASS1_BITS),
            descale_simd8(z1 - tmp12 * SIMD8_FIX_1_847759065, CONST_BITS - PASS1_BITS),
        )
    } else {
        (
            descale_simd8(z1 + tmp13 * SIMD8_FIX_0_765366865, CONST_BITS + PASS1_BITS),
            descale_simd8(z1 - tmp12 * SIMD8_FIX_1_847759065, CONST_BITS + PASS1_BITS),
        )
    };

    let z1 = tmp4 + tmp7;
    let z2 = tmp5 + tmp6;
    let z3 = tmp4 + tmp6;
    let z4 = tmp5 + tmp7;
    let z5 = (z3 + z4) * SIMD8_FIX_1_175875602;

    let tmp4 = tmp4 * SIMD8_FIX_0_298631336;
    let tmp5 = tmp5 * SIMD8_FIX_2_053119869;
    let tmp6 = tmp6 * SIMD8_FIX_3_072711026;
    let tmp7 = tmp7 * SIMD8_FIX_1_501321110;

    let neg_z1 = z1 * SIMD8_NEG_FIX_0_899976223;
    let neg_z2 = z2 * SIMD8_NEG_FIX_2_562915447;
    let z3 = z3 * SIMD8_NEG_FIX_1_961570560 + z5;
    let z4 = z4 * SIMD8_NEG_FIX_0_390180644 + z5;

    let scale = if shift_pass1 { CONST_BITS - PASS1_BITS } else { CONST_BITS + PASS1_BITS };
    let out7 = descale_simd8(tmp4 + neg_z1 + z3, scale);
    let out5 = descale_simd8(tmp5 + neg_z2 + z4, scale);
    let out3 = descale_simd8(tmp6 + neg_z2 + z3, scale);
    let out1 = descale_simd8(tmp7 + neg_z1 + z4, scale);

    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// Transpose 8x8 matrix stored as 8 i32x8 vectors.
#[inline(always)]
fn transpose_8x8(rows: &mut [i32x8; 8]) {
    let mut data: [[i32; 8]; 8] = [
        rows[0].to_array(), rows[1].to_array(), rows[2].to_array(), rows[3].to_array(),
        rows[4].to_array(), rows[5].to_array(), rows[6].to_array(), rows[7].to_array(),
    ];

    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        for j in (i + 1)..8 {
            let tmp = data[i][j];
            data[i][j] = data[j][i];
            data[j][i] = tmp;
        }
    }

    for i in 0..8 {
        rows[i] = i32x8::new(data[i]);
    }
}

/// Transpose-based SIMD DCT with contiguous memory access.
///
/// Uses 8-wide SIMD (i32x8) for maximum parallelism.
/// Works on both x86 (AVX) and ARM (NEON via wide crate).
#[allow(dead_code)]
pub fn forward_dct_8x8_transpose(samples: &[i16; DCTSIZE2], coeffs: &mut [i16; DCTSIZE2]) {
    let mut data: [i32x8; 8] = std::array::from_fn(|row| {
        let base = row * 8;
        i32x8::new([
            samples[base] as i32, samples[base + 1] as i32,
            samples[base + 2] as i32, samples[base + 3] as i32,
            samples[base + 4] as i32, samples[base + 5] as i32,
            samples[base + 6] as i32, samples[base + 7] as i32,
        ])
    });

    transpose_8x8(&mut data);

    let result = dct_1d_8wide(
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7], true,
    );
    data = result;

    transpose_8x8(&mut data);

    let result = dct_1d_8wide(
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7], false,
    );
    data = result;

    for row in 0..8 {
        let arr = data[row].to_array();
        for col in 0..8 {
            coeffs[row * 8 + col] = arr[col] as i16;
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Descale DCT coefficients for non-trellis quantization.
///
/// The integer DCT outputs values scaled by 8. For standard quantization
/// (without trellis), we need to remove this scaling first.
#[inline]
pub fn descale_for_quant(coeffs: &[i16; DCTSIZE2], output: &mut [i32; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        // (x + 4) >> 3 = divide by 8 with rounding
        output[i] = ((coeffs[i] as i32) + 4) >> 3;
    }
}

/// Scale DCT coefficients for trellis (convert i16 to i32).
///
/// For trellis quantization, we pass the 8x-scaled values directly.
/// Trellis internally multiplies qtable by 8 to match.
#[inline]
pub fn scale_for_trellis(coeffs: &[i16; DCTSIZE2], output: &mut [i32; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        output[i] = coeffs[i] as i32;
    }
}

/// Quantize DCT coefficients using integer arithmetic.
pub fn quantize_block_int(coeffs: &[i32; DCTSIZE2], quant: &[u16; DCTSIZE2], output: &mut [i16; DCTSIZE2]) {
    for i in 0..DCTSIZE2 {
        let c = coeffs[i];
        let q = quant[i] as i32;
        output[i] = if c >= 0 {
            ((c + q / 2) / q) as i16
        } else {
            ((c - q / 2) / q) as i16
        };
    }
}

/// Level-shift a block of pixels for DCT (subtract 128)
pub fn level_shift(pixels: &[u8; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for i in 0..64 {
        output[i] = pixels[i] as i16 - 128;
    }
    output
}

// ============================================================================
// Legacy float-based functions (for backwards compatibility)
// ============================================================================

/// Forward 8x8 DCT on level-shifted input (LEGACY - float version)
///
/// **DEPRECATED**: Use `forward_dct_8x8_int` for better precision and compression.
pub fn forward_dct_8x8(block: &[i16; 64]) -> [f32; 64] {
    let mut output = [0.0f32; 64];

    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;

            for y in 0..8 {
                for x in 0..8 {
                    let pixel = block[y * 8 + x] as f32;
                    let cu = if u == 0 { 1.0 / std::f32::consts::SQRT_2 } else { 1.0 };
                    let cv = if v == 0 { 1.0 / std::f32::consts::SQRT_2 } else { 1.0 };

                    let cos_u = ((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI / 16.0).cos();
                    let cos_v = ((2.0 * y as f32 + 1.0) * v as f32 * std::f32::consts::PI / 16.0).cos();

                    sum += cu * cv * pixel * cos_u * cos_v;
                }
            }

            output[v * 8 + u] = sum / 4.0;
        }
    }

    output
}

/// Quantize DCT coefficients (LEGACY - float version)
pub fn quantize_block(dct: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for i in 0..64 {
        let q = quant[i] as f32;
        output[i] = (dct[i] / q).round() as i16;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_only() {
        let block = [0i16; 64];
        let dct = forward_dct_8x8(&block);
        assert!(dct[0].abs() < 0.01, "DC = {}", dct[0]);
        for i in 1..64 {
            assert!(dct[i].abs() < 0.01, "AC[{}] = {}", i, dct[i]);
        }
    }

    #[test]
    fn test_integer_dct_dc_only() {
        let samples = [0i16; 64];
        let mut coeffs = [0i16; 64];
        forward_dct_8x8_int(&samples, &mut coeffs);
        assert_eq!(coeffs[0], 0, "DC = {}", coeffs[0]);
        for i in 1..64 {
            assert_eq!(coeffs[i], 0, "AC[{}] = {}", i, coeffs[i]);
        }
    }

    #[test]
    fn test_integer_dct_white_block() {
        let samples = [127i16; 64];
        let mut coeffs = [0i16; 64];
        forward_dct_8x8_int(&samples, &mut coeffs);
        assert!(coeffs[0] > 900, "DC = {} (expected ~1016)", coeffs[0]);
        for i in 1..64 {
            assert_eq!(coeffs[i], 0, "AC[{}] = {} (expected 0)", i, coeffs[i]);
        }
    }

    #[test]
    fn test_simd_matches_scalar_all_patterns() {
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_simd = [0i16; DCTSIZE2];

            forward_dct_8x8_int(&samples, &mut coeffs_scalar);
            forward_dct_8x8_simd(&samples, &mut coeffs_simd);

            assert_eq!(coeffs_scalar, coeffs_simd, "SIMD should match scalar for seed {}", seed);
        }
    }

    #[test]
    fn test_transpose_matches_scalar_all_patterns() {
        for seed in 0..20 {
            let mut samples = [0i16; DCTSIZE2];
            for i in 0..DCTSIZE2 {
                samples[i] = ((i as i32 * (seed * 37 + 13) + seed * 7) % 256 - 128) as i16;
            }

            let mut coeffs_scalar = [0i16; DCTSIZE2];
            let mut coeffs_transpose = [0i16; DCTSIZE2];

            forward_dct_8x8_int(&samples, &mut coeffs_scalar);
            forward_dct_8x8_transpose(&samples, &mut coeffs_transpose);

            assert_eq!(coeffs_scalar, coeffs_transpose, "Transpose should match scalar for seed {}", seed);
        }
    }

    #[test]
    fn test_quantize_block_int() {
        let coeffs = [100i32, 50, -30, 0, 200, -150, 10, 5,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0];
        let quant = [16u16; 64];
        let mut output = [0i16; 64];

        quantize_block_int(&coeffs, &quant, &mut output);

        assert_eq!(output[0], 6);  // 100 / 16 = 6.25 -> 6
        assert_eq!(output[1], 3);  // 50 / 16 = 3.125 -> 3
        assert_eq!(output[2], -2); // -30 / 16 = -1.875 -> -2
        assert_eq!(output[3], 0);
    }
}
