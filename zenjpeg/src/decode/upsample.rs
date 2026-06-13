//! Chroma upsampling for JPEG decoding.
//!
//! Implements libjpeg-turbo compatible upsampling and nearest-neighbor (box filter)
//! upsampling for various chroma subsampling modes (4:2:2, 4:4:0, 4:2:0).

#[cfg(target_arch = "x86_64")]
use archmage::{SimdToken, arcane};

#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

use archmage::prelude::*;
use magetypes::simd::generic::i32x8 as GenericI32x8;

// Nearest-Neighbor Upsampling (Box Filter)
// ============================================================================

/// Horizontal 2x + vertical 2x nearest-neighbor upsampling in i16 (4:2:0 → 4:4:4).
///
/// Each chroma sample is replicated to fill the corresponding 2x2 output area.
pub fn upsample_h2v2_i16_nearest(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let out_row = out_y * out_width;
        let in_row = in_y * in_width;

        for out_x in 0..out_width {
            let in_x = (out_x / 2).min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

/// Horizontal 2x nearest-neighbor upsampling in i16 (4:2:2 → 4:4:4).
pub fn upsample_h2v1_i16_nearest(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v1_i16_nearest_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Vertical 2x nearest-neighbor upsampling in i16 (4:4:0 → 4:4:4).
pub fn upsample_h1v2_i16_nearest(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h1v2_i16_nearest_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Strided horizontal 2x nearest-neighbor upsampling in i16 (4:2:2 → 4:4:4).
pub fn upsample_h2v1_i16_nearest_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = (out_x / 2).min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

/// Strided vertical 2x nearest-neighbor upsampling in i16 (4:4:0 → 4:4:4).
pub fn upsample_h1v2_i16_nearest_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

/// Strided horizontal 2x + vertical 2x nearest-neighbor upsampling in i16 (4:2:0 → 4:4:4).
pub fn upsample_h2v2_i16_nearest_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    for out_y in 0..out_height {
        let in_y = (out_y / 2).min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        for out_x in 0..out_width {
            let in_x = (out_x / 2).min(in_width.saturating_sub(1));
            output[out_row + out_x] = input[in_row + in_x];
        }
    }
}

// ============================================================================
// libjpeg-turbo Compatible Upsampling
// ============================================================================

/// Rounding-bias mode for the fused h2v2 (4:2:0) triangle row kernel.
///
/// `Alternating` is zenjpeg's default Triangle behavior: biases (8, 7) on
/// upper output rows and (7, 8) on lower rows, giving the half-case
/// rounding a checkerboard arrangement. `Turbo` is libjpeg-turbo's
/// `h2v2_fancy_upsample`: fixed (8, 7) on every row — bit-exact with
/// turbo. Selected by [`IdctMethod::Libjpeg`](super::IdctMethod::Libjpeg),
/// whose contract is libjpeg-turbo-exact decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum H2v2Bias {
    /// Row-alternating bias (default Triangle).
    Alternating {
        /// True on even output rows (nearer chroma row is above).
        is_upper: bool,
    },
    /// libjpeg-turbo fixed bias: (8, 7) on every row.
    Turbo,
}

impl H2v2Bias {
    /// Select the mode for a decode: `Turbo` under [`super::IdctMethod::Libjpeg`].
    #[inline]
    pub(crate) fn for_idct_method(method: super::IdctMethod, is_upper: bool) -> Self {
        match method {
            super::IdctMethod::Libjpeg => H2v2Bias::Turbo,
            _ => H2v2Bias::Alternating { is_upper },
        }
    }

    /// (left-output bias, right-output bias) for this row.
    #[inline(always)]
    fn pair(self) -> (i32, i32) {
        match self {
            H2v2Bias::Alternating { is_upper: false } => (7, 8),
            _ => (8, 7),
        }
    }

    /// Bias of the second output when the chroma plane is a single column.
    /// Alternating keeps zenjpeg's legacy duplicated first output (bias 8);
    /// turbo emits the `(4c + 7) >> 4` right-output form its padded
    /// buffers produce.
    #[inline(always)]
    fn single_col_second(self) -> i32 {
        match self {
            H2v2Bias::Turbo => 7,
            H2v2Bias::Alternating { .. } => 8,
        }
    }
}

/// Horizontal 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:2:2 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for left pixel, +2 for right pixel.
/// Matches libjpeg-turbo's `jdsample.c` h2v1_fancy_upsample.
pub fn upsample_h2v1_i16_libjpeg(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v1_i16_libjpeg_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:4:0 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for upper row (v=0), +2 for lower row (v=1).
pub fn upsample_h1v2_i16_libjpeg(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h1v2_i16_libjpeg_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Strided horizontal 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:2:2 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for left pixel, +2 for right pixel.
/// Matches libjpeg-turbo's `jdsample.c` h2v1_fancy_upsample.
#[doc(hidden)]
pub fn upsample_h2v1_i16_libjpeg_strided_scalar(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        if in_width == 1 {
            let val = input[in_row];
            if out_width > 0 {
                output[out_row] = val;
            }
            if out_width > 1 {
                output[out_row + 1] = val;
            }
            continue;
        }

        // First column
        let curr = input[in_row] as i32;
        let next = input[in_row + 1] as i32;
        output[out_row] = curr as i16;
        if out_width > 1 {
            output[out_row + 1] = ((curr * 3 + next + 2) >> 2) as i16;
        }

        // Interior columns
        for in_x in 1..in_width.saturating_sub(1) {
            let prev = input[in_row + in_x - 1] as i32;
            let curr = input[in_row + in_x] as i32;
            let next = input[in_row + in_x + 1] as i32;
            let left_out = in_x * 2;
            let right_out = left_out + 1;
            if left_out < out_width {
                output[out_row + left_out] = ((curr * 3 + prev + 1) >> 2) as i16;
            }
            if right_out < out_width {
                output[out_row + right_out] = ((curr * 3 + next + 2) >> 2) as i16;
            }
        }

        // Last column
        let last = in_width - 1;
        let prev = input[in_row + last - 1] as i32;
        let curr = input[in_row + last] as i32;
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[out_row + left_out] = ((curr * 3 + prev + 1) >> 2) as i16;
        }
        if right_out < out_width {
            output[out_row + right_out] = curr as i16;
        }
    }
}

/// 4:2:2 (h2v1) fancy upsampler. The interior columns run in i32x8 SIMD
/// (AVX2/NEON/wasm128 via magetypes); edges and narrow rows fall to the
/// bit-identical scalar path. The scalar kernel does NOT autovectorize on
/// x86 (177 scalar instrs / 0 vector, measured 2026-06-13), so the SIMD
/// interior is a real all-arch win on this otherwise-scalar path.
pub fn upsample_h2v1_i16_libjpeg_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width < 10 {
        // Too narrow for an 8-wide interior chunk; scalar is fine.
        upsample_h2v1_i16_libjpeg_strided_scalar(
            input, in_width, in_stride, in_height, output, out_width, out_stride, out_height,
        );
        return;
    }
    incant!(upsample_h2v1_generic_impl(
        input, in_width, in_stride, in_height, output, out_width, out_stride, out_height
    ));
}

#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn upsample_h2v1_generic_impl(
    token: Token,
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;
    if in_width == 0 || in_height == 0 {
        return;
    }
    let three = i32x8::splat(token, 3);
    let one = i32x8::splat(token, 1);
    let two = i32x8::splat(token, 2);
    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_stride;
        let in_row = in_y * in_stride;

        // First column (bit-identical to scalar).
        let curr0 = input[in_row] as i32;
        let next0 = input[in_row + 1] as i32;
        output[out_row] = curr0 as i16;
        if out_width > 1 {
            output[out_row + 1] = ((curr0 * 3 + next0 + 2) >> 2) as i16;
        }

        // Interior columns, SIMD in chunks of 8 input positions from x=1.
        // Needs input[x-1 ..= x+8] and output[2x ..= 2x+15] in bounds.
        let mut x = 1usize;
        while x + 9 <= in_width && 2 * x + 15 < out_width {
            let prev = i32x8::from_array(
                token,
                core::array::from_fn(|i| input[in_row + x - 1 + i] as i32),
            );
            let curr = i32x8::from_array(
                token,
                core::array::from_fn(|i| input[in_row + x + i] as i32),
            );
            let next = i32x8::from_array(
                token,
                core::array::from_fn(|i| input[in_row + x + 1 + i] as i32),
            );
            let curr3 = curr * three;
            let left = (curr3 + prev + one).shr_arithmetic_const::<2>().to_array();
            let right = (curr3 + next + two).shr_arithmetic_const::<2>().to_array();
            let mut o = out_row + x * 2;
            for i in 0..8 {
                output[o] = left[i] as i16;
                output[o + 1] = right[i] as i16;
                o += 2;
            }
            x += 8;
        }

        // Scalar interior remainder.
        while x < in_width - 1 {
            let prev = input[in_row + x - 1] as i32;
            let curr = input[in_row + x] as i32;
            let next = input[in_row + x + 1] as i32;
            let lo = x * 2;
            let ro = lo + 1;
            if lo < out_width {
                output[out_row + lo] = ((curr * 3 + prev + 1) >> 2) as i16;
            }
            if ro < out_width {
                output[out_row + ro] = ((curr * 3 + next + 2) >> 2) as i16;
            }
            x += 1;
        }

        // Last column (bit-identical to scalar).
        let last = in_width - 1;
        let prev = input[in_row + last - 1] as i32;
        let curr = input[in_row + last] as i32;
        let lo = last * 2;
        let ro = lo + 1;
        if lo < out_width {
            output[out_row + lo] = ((curr * 3 + prev + 1) >> 2) as i16;
        }
        if ro < out_width {
            output[out_row + ro] = curr as i16;
        }
    }
}

/// Strided vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding (4:4:0 → 4:4:4).
///
/// Uses alternating rounding bias: +1 for upper row (v=0), +2 for lower row (v=1).
pub fn upsample_h1v2_i16_libjpeg_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;
        let out_row = out_y * out_stride;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_stride;
        let far_row = far_y * in_stride;

        let bias = if is_upper { 1i16 } else { 2i16 };
        let w = out_width.min(in_width);

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(token) = archmage::X64V3Token::summon() {
                upsample_h1v2_row_avx2(
                    token,
                    &input[near_row..],
                    &input[far_row..],
                    &mut output[out_row..],
                    w,
                    bias,
                );
                // Handle edge replication for out_width > in_width
                if out_width > in_width && in_width > 0 {
                    let edge = output[out_row + in_width - 1];
                    for x in in_width..out_width {
                        output[out_row + x] = edge;
                    }
                }
                continue;
            }
        }

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            let near = input[near_row + in_x] as i32;
            let far = input[far_row + in_x] as i32;
            output[out_row + out_x] = ((near * 3 + far + bias as i32) >> 2) as i16;
        }
    }
}

/// AVX2 implementation of one row of vertical 2x upsampling.
///
/// Computes `(near*3 + far + bias) >> 2` for 16 i16 elements at a time.
/// All arithmetic stays within i16 range: max magnitude is
/// `(2048*3 + 2048 + 2) >> 2 = 2049`, well within i16.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_h1v2_row_avx2(
    _token: archmage::X64V3Token,
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    width: usize,
    bias: i16,
) {
    use core::arch::x86_64::*;

    let v_three = _mm256_set1_epi16(3);
    let v_bias = _mm256_set1_epi16(bias);

    let mut x = 0;
    while x + 16 <= width {
        let v_near =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x..x + 16]).unwrap());
        let v_far = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x..x + 16]).unwrap());
        // (near * 3 + far + bias) >> 2
        let v_result = _mm256_srai_epi16(
            _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(v_near, v_three), v_far),
                v_bias,
            ),
            2,
        );
        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[x..x + 16]).unwrap(),
            v_result,
        );
        x += 16;
    }

    // Scalar tail
    for x in x..width {
        let n = near[x] as i32;
        let f = far[x] as i32;
        output[x] = ((n * 3 + f + bias as i32) >> 2) as i16;
    }
}

/// Horizontal 2x + vertical 2x triangle upsampling in i16 (4:2:0 → 4:4:4).
///
/// Same fused 9:3:3:1 filter as libjpeg-turbo's `jdsample.c`
/// `h2v2_fancy_upsample`, but NOT bit-identical to it: turbo uses fixed
/// rounding biases (+8 left output, +7 right output) on both rows of a
/// pair, while this implementation row-alternates the pair to (7, 8) on
/// lower rows — the same vertical alternation turbo itself applies in
/// `h1v2_fancy_upsample` (+1/+2). Measured vs a turbo reference
/// (`h2v2_triangle_vs_libjpeg_turbo_reference`): even output rows are
/// bit-identical; ~6.7% of odd-row pixels differ by exactly ±1 (the
/// half-boundary cases). Both schemes have max error 0.5 and ~zero global
/// bias vs the exact real-valued filter; they differ only in the spatial
/// arrangement of half-case rounding (checkerboard here, column stripes
/// in turbo).
pub fn upsample_h2v2_i16_libjpeg(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v2_i16_libjpeg_strided(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Like [`upsample_h2v2_i16_libjpeg`] but with libjpeg-turbo's fixed
/// rounding biases — bit-exact with turbo's `h2v2_fancy_upsample`.
/// Selected for [`IdctMethod::Libjpeg`](super::IdctMethod::Libjpeg) decodes.
pub fn upsample_h2v2_i16_libjpeg_turbo(
    input: &[i16],
    in_width: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_height: usize,
) {
    upsample_h2v2_i16_libjpeg_strided_turbo(
        input, in_width, in_width, in_height, output, out_width, out_width, out_height,
    );
}

/// Process one output row of fused h2v2 libjpeg-compat upsampling.
///
/// `bias` selects the rounding-bias scheme (row-alternating default or
/// libjpeg-turbo fixed — see [`H2v2Bias`]).
///
/// Dispatches to AVX2 SIMD on x86_64 when available, with scalar fallback.
#[inline]
#[doc(hidden)]
pub fn upsample_h2v2_libjpeg_row(
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    bias: H2v2Bias,
) {
    // Try AVX2 SIMD path on x86_64 (hand-tuned, beats the generic here).
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            upsample_h2v2_libjpeg_row_avx2(token, near, far, output, in_width, out_width, bias);
            return;
        }
    }

    // Non-x86: magetypes-generic (NEON/wasm128). The scalar row does not
    // autovectorize (0 vector instrs, measured), so the SIMD interior is a
    // real ARM/wasm win (~+45% NEON, like the h2v1 case). Narrow rows + edges
    // stay scalar inside the generic.
    #[cfg(not(target_arch = "x86_64"))]
    {
        incant!(upsample_h2v2_libjpeg_row_generic(
            near, far, output, in_width, out_width, bias
        ));
        return;
    }
    #[cfg(target_arch = "x86_64")]
    upsample_h2v2_libjpeg_row_scalar(near, far, output, in_width, out_width, bias);
}

/// Magetypes-generic one-row h2v2 fancy upsampler — bit-identical to
/// `upsample_h2v2_libjpeg_row_scalar`, interior columns vectorized in i32x8.
/// Used on non-x86 (NEON/wasm128); x86 uses the hand AVX2 kernel.
#[magetypes(v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn upsample_h2v2_libjpeg_row_generic(
    token: Token,
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    bias: H2v2Bias,
) {
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;
    if in_width == 1 {
        let colsum = near[0] as i32 * 3 + far[0] as i32;
        if out_width > 0 {
            output[0] = ((colsum * 4 + 8) >> 4) as i16;
        }
        if out_width > 1 {
            output[1] = ((colsum * 4 + bias.single_col_second()) >> 4) as i16;
        }
        return;
    }
    let (bias_left, bias_right) = bias.pair();

    // First column.
    let cs0 = near[0] as i32 * 3 + far[0] as i32;
    let cs1 = near[1] as i32 * 3 + far[1] as i32;
    output[0] = ((cs0 * 4 + 8) >> 4) as i16;
    if out_width > 1 {
        output[1] = ((cs0 * 3 + cs1 + bias_right) >> 4) as i16;
    }

    // Interior, SIMD in chunks of 8 input positions from x=1.
    // colsum[i] = near[i]*3 + far[i]; needs i in [x-1 ..= x+8].
    let three = i32x8::splat(token, 3);
    let vbl = i32x8::splat(token, bias_left);
    let vbr = i32x8::splat(token, bias_right);
    let mut x = 1usize;
    while x + 9 <= in_width && 2 * x + 15 < out_width {
        // Load colsum ONCE for the [x-1 ..= x+8] window (10 values), reading
        // near/far from memory a single time, then build the prev/this/next
        // vectors from that local array. The previous code re-gathered near
        // and far 3× (6 scalar-widen gathers/chunk → ~48 element loads); this
        // is ~20 loads + 3 contiguous reads of an i32 array. (The h2v1 kernel
        // is cheaper per output — one plane, one window — so its 3-gather form
        // already beats scalar; h2v2's 2 planes × 3 windows did not.)
        let cw: [i32; 10] =
            core::array::from_fn(|i| near[x - 1 + i] as i32 * 3 + far[x - 1 + i] as i32);
        let prev_cs = i32x8::from_array(token, core::array::from_fn(|i| cw[i]));
        let this_cs = i32x8::from_array(token, core::array::from_fn(|i| cw[1 + i]));
        let next_cs = i32x8::from_array(token, core::array::from_fn(|i| cw[2 + i]));
        let this3 = this_cs * three;
        let left = (this3 + prev_cs + vbl)
            .shr_arithmetic_const::<4>()
            .to_array();
        let right = (this3 + next_cs + vbr)
            .shr_arithmetic_const::<4>()
            .to_array();
        let mut o = x * 2;
        for i in 0..8 {
            output[o] = left[i] as i16;
            output[o + 1] = right[i] as i16;
            o += 2;
        }
        x += 8;
    }

    // Scalar interior remainder.
    let mut last_colsum = if x == 1 {
        cs0
    } else {
        near[x - 1] as i32 * 3 + far[x - 1] as i32
    };
    while x < in_width - 1 {
        let this_colsum = near[x] as i32 * 3 + far[x] as i32;
        let next_colsum = near[x + 1] as i32 * 3 + far[x + 1] as i32;
        let lo = x * 2;
        let ro = lo + 1;
        if lo < out_width {
            output[lo] = ((this_colsum * 3 + last_colsum + bias_left) >> 4) as i16;
        }
        if ro < out_width {
            output[ro] = ((this_colsum * 3 + next_colsum + bias_right) >> 4) as i16;
        }
        last_colsum = this_colsum;
        x += 1;
    }

    // Last column.
    let last = in_width - 1;
    let this_colsum = near[last] as i32 * 3 + far[last] as i32;
    let lo = last * 2;
    let ro = lo + 1;
    if lo < out_width {
        output[lo] = ((this_colsum * 3 + last_colsum + bias_left) >> 4) as i16;
    }
    if ro < out_width {
        output[ro] = ((this_colsum * 4 + bias_right) >> 4) as i16;
    }
}

/// Scalar implementation of one output row of fused h2v2 libjpeg-compat upsampling.
#[doc(hidden)]
#[inline]
pub fn upsample_h2v2_libjpeg_row_scalar(
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    bias: H2v2Bias,
) {
    if in_width == 1 {
        // Single column: just vertical filter
        let colsum = near[0] as i32 * 3 + far[0] as i32;
        if out_width > 0 {
            output[0] = ((colsum * 4 + 8) >> 4) as i16;
        }
        if out_width > 1 {
            output[1] = ((colsum * 4 + bias.single_col_second()) >> 4) as i16;
        }
        return;
    }

    // Rounding biases: libjpeg-turbo's h2v2 uses FIXED (left 8, right 7)
    // for both rows of a pair; the default Triangle mode alternates to
    // (7, 8) on lower rows (like turbo's own h1v2 +1/+2 scheme), which
    // turns the half-case rounding pattern from column stripes into a
    // checkerboard but makes odd rows differ from turbo by ±1 on ~6.7% of
    // pixels. `H2v2Bias::Turbo` (used by IdctMethod::Libjpeg) keeps the
    // fixed pair and is bit-exact with turbo — see
    // h2v2_triangle_vs_libjpeg_turbo_reference.
    let (bias_left, bias_right) = bias.pair();

    // Column sums: near * 3 + far
    let this_colsum = near[0] as i32 * 3 + far[0] as i32;
    let next_colsum = near[1] as i32 * 3 + far[1] as i32;

    // First column
    output[0] = ((this_colsum * 4 + 8) >> 4) as i16;
    if out_width > 1 {
        output[1] = ((this_colsum * 3 + next_colsum + bias_right) >> 4) as i16;
    }

    // Interior columns
    let mut last_colsum = this_colsum;
    for in_x in 1..in_width.saturating_sub(1) {
        let this_colsum = near[in_x] as i32 * 3 + far[in_x] as i32;
        let next_colsum = near[in_x + 1] as i32 * 3 + far[in_x + 1] as i32;

        let left_out = in_x * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[left_out] = ((this_colsum * 3 + last_colsum + bias_left) >> 4) as i16;
        }
        if right_out < out_width {
            output[right_out] = ((this_colsum * 3 + next_colsum + bias_right) >> 4) as i16;
        }
        last_colsum = this_colsum;
    }

    // Last column
    let last = in_width - 1;
    let this_colsum = near[last] as i32 * 3 + far[last] as i32;
    let left_out = last * 2;
    let right_out = left_out + 1;
    if left_out < out_width {
        output[left_out] = ((this_colsum * 3 + last_colsum + bias_left) >> 4) as i16;
    }
    if right_out < out_width {
        output[right_out] = ((this_colsum * 4 + bias_right) >> 4) as i16;
    }
}

/// AVX2 SIMD implementation of one output row of fused h2v2 libjpeg-compat upsampling.
///
/// Processes 16 input chroma samples at a time → 32 output pixels.
/// Produces bit-exact output matching the scalar `upsample_h2v2_libjpeg_row_scalar`.
///
/// All arithmetic stays within i16 range: colsum max magnitude is 8192
/// (`2048*3 + 2048`), and `colsum*3 + colsum_neighbor + 8` max is 32760.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn upsample_h2v2_libjpeg_row_avx2(
    _token: archmage::X64V3Token,
    near: &[i16],
    far: &[i16],
    output: &mut [i16],
    in_width: usize,
    out_width: usize,
    bias: H2v2Bias,
) {
    use core::arch::x86_64::*;

    // For very small widths, fall back to scalar (not worth SIMD overhead)
    if in_width < 18 {
        upsample_h2v2_libjpeg_row_scalar(near, far, output, in_width, out_width, bias);
        return;
    }

    let (bias_left, bias_right) = bias.pair();
    let (bias_left, bias_right) = (bias_left as i16, bias_right as i16);

    let v_three = _mm256_set1_epi16(3);
    let v_bias_left = _mm256_set1_epi16(bias_left);
    let v_bias_right = _mm256_set1_epi16(bias_right);

    // --- First column (scalar, special edge handling) ---
    let colsum_0 = near[0] as i32 * 3 + far[0] as i32;
    let colsum_1 = near[1] as i32 * 3 + far[1] as i32;
    output[0] = ((colsum_0 * 4 + 8) >> 4) as i16;
    if out_width > 1 {
        output[1] = ((colsum_0 * 3 + colsum_1 + bias_right as i32) >> 4) as i16;
    }

    // --- Interior: SIMD processing ---
    // Process chunks of 16 input pixels starting from position 1.
    // For each chunk we need colsum[x-1..x+16] and colsum[x..x+17],
    // so we need near/far[x-1..x+17] accessible. We process up to
    // the point where x+17 <= in_width (i.e., x <= in_width - 17).
    let simd_start = 1usize;
    let simd_end_exclusive = if in_width >= 17 {
        // Last chunk starts at x where x+16 <= in_width-1 (need next neighbor)
        // i.e., x <= in_width - 17
        let max_start = in_width - 17;
        // Round down to chunk boundary relative to simd_start
        let num_chunks = (max_start - simd_start + 16) / 16;
        simd_start + num_chunks * 16
    } else {
        simd_start
    };

    let mut x = simd_start;
    while x + 16 <= simd_end_exclusive {
        let out_base = x * 2;
        if out_base + 32 > out_width {
            break;
        }

        // Load near[x..x+16], far[x..x+16] → colsum[x..x+16]
        let v_near =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x..x + 16]).unwrap());
        let v_far = safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x..x + 16]).unwrap());
        let v_colsum = _mm256_add_epi16(_mm256_mullo_epi16(v_near, v_three), v_far);

        // Load colsum for x-1 (prev neighbor)
        let v_near_prev =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x - 1..x + 15]).unwrap());
        let v_far_prev =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x - 1..x + 15]).unwrap());
        let v_colsum_prev = _mm256_add_epi16(_mm256_mullo_epi16(v_near_prev, v_three), v_far_prev);

        // Load colsum for x+1 (next neighbor)
        let v_near_next =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&near[x + 1..x + 17]).unwrap());
        let v_far_next =
            safe_simd::_mm256_loadu_si256(<&[i16; 16]>::try_from(&far[x + 1..x + 17]).unwrap());
        let v_colsum_next = _mm256_add_epi16(_mm256_mullo_epi16(v_near_next, v_three), v_far_next);

        // colsum * 3
        let v_colsum3 = _mm256_mullo_epi16(v_colsum, v_three);

        // Left output: (colsum*3 + colsum_prev + bias_left) >> 4
        let v_left = _mm256_srai_epi16(
            _mm256_add_epi16(_mm256_add_epi16(v_colsum3, v_colsum_prev), v_bias_left),
            4,
        );

        // Right output: (colsum*3 + colsum_next + bias_right) >> 4
        let v_right = _mm256_srai_epi16(
            _mm256_add_epi16(_mm256_add_epi16(v_colsum3, v_colsum_next), v_bias_right),
            4,
        );

        // Interleave left and right: [L0, R0, L1, R1, ...]
        // unpacklo/hi work on 128-bit lanes, so we need permute to fix order
        let lo = _mm256_unpacklo_epi16(v_left, v_right);
        let hi = _mm256_unpackhi_epi16(v_left, v_right);
        let out0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        let out1 = _mm256_permute2x128_si256(lo, hi, 0x31);

        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_base..out_base + 16]).unwrap(),
            out0,
        );
        safe_simd::_mm256_storeu_si256(
            <&mut [i16; 16]>::try_from(&mut output[out_base + 16..out_base + 32]).unwrap(),
            out1,
        );

        x += 16;
    }

    // --- Scalar remainder for interior pixels not covered by SIMD ---
    let mut last_colsum_i32 = if x > 1 {
        near[x - 1] as i32 * 3 + far[x - 1] as i32
    } else {
        colsum_0
    };

    for in_x in x..in_width.saturating_sub(1) {
        let this_colsum = near[in_x] as i32 * 3 + far[in_x] as i32;
        let next_colsum = near[in_x + 1] as i32 * 3 + far[in_x + 1] as i32;

        let left_out = in_x * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[left_out] = ((this_colsum * 3 + last_colsum_i32 + bias_left as i32) >> 4) as i16;
        }
        if right_out < out_width {
            output[right_out] = ((this_colsum * 3 + next_colsum + bias_right as i32) >> 4) as i16;
        }
        last_colsum_i32 = this_colsum;
    }

    // --- Last column (scalar, special edge handling) ---
    let last = in_width - 1;
    if last >= x || x == simd_start {
        // Only emit last column if not already covered
        let this_colsum = near[last] as i32 * 3 + far[last] as i32;
        let prev_colsum = if last > 0 {
            near[last - 1] as i32 * 3 + far[last - 1] as i32
        } else {
            this_colsum
        };
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[left_out] = ((this_colsum * 3 + prev_colsum + bias_left as i32) >> 4) as i16;
        }
        if right_out < out_width {
            output[right_out] = ((this_colsum * 4 + bias_right as i32) >> 4) as i16;
        }
    }
}

/// Strided horizontal 2x + vertical 2x upsampling in i16 with libjpeg-turbo compatible rounding.
///
/// Same algorithm as `upsample_h2v2_i16_libjpeg` but supports SIMD-aligned stride > width.
pub fn upsample_h2v2_i16_libjpeg_strided(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    upsample_h2v2_i16_libjpeg_strided_with(
        input, in_width, in_stride, in_height, output, out_width, out_stride, out_height, false,
    );
}

/// Like [`upsample_h2v2_i16_libjpeg_strided`] but with libjpeg-turbo's
/// fixed rounding biases — bit-exact with turbo's `h2v2_fancy_upsample`.
pub fn upsample_h2v2_i16_libjpeg_strided_turbo(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
) {
    upsample_h2v2_i16_libjpeg_strided_with(
        input, in_width, in_stride, in_height, output, out_width, out_stride, out_height, true,
    );
}

#[allow(clippy::too_many_arguments)]
fn upsample_h2v2_i16_libjpeg_strided_with(
    input: &[i16],
    in_width: usize,
    in_stride: usize,
    in_height: usize,
    output: &mut [i16],
    out_width: usize,
    out_stride: usize,
    out_height: usize,
    turbo_bias: bool,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_stride;
        let far_row = far_y * in_stride;
        let out_row = out_y * out_stride;

        let bias = if turbo_bias {
            H2v2Bias::Turbo
        } else {
            H2v2Bias::Alternating { is_upper }
        };
        upsample_h2v2_libjpeg_row(
            &input[near_row..near_row + in_width],
            &input[far_row..far_row + in_width],
            &mut output[out_row..],
            in_width,
            out_width,
            bias,
        );
    }
}

// ============================================================================
// f32 Nearest-Neighbor and libjpeg-compat Upsampling
// ============================================================================

/// Nearest-neighbor upsampling for f32 planes.
///
/// Replaces the inline box filter code in output.rs.
pub fn upsample_nearest_f32(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
    scale_x: usize,
    scale_y: usize,
) {
    for py in 0..out_height {
        let sy = (py / scale_y).min(in_height.saturating_sub(1));
        let out_row = py * out_width;
        let in_row = sy * in_width;
        for px in 0..out_width {
            let sx = (px / scale_x).min(in_width.saturating_sub(1));
            output[out_row + px] = input[in_row + sx];
        }
    }
}

/// libjpeg-turbo compatible upsampling for f32 planes.
///
/// Dispatches to the appropriate algorithm based on scale factors.
pub fn upsample_libjpeg_f32(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
    scale_x: usize,
    scale_y: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; out_width * out_height];
    match (scale_x, scale_y) {
        (2, 2) => upsample_h2v2_f32_libjpeg(
            input,
            in_width,
            in_height,
            &mut output,
            out_width,
            out_height,
        ),
        (2, 1) => upsample_h2v1_f32_libjpeg(
            input,
            in_width,
            in_height,
            &mut output,
            out_width,
            out_height,
        ),
        (1, 2) => upsample_h1v2_f32_libjpeg(
            input,
            in_width,
            in_height,
            &mut output,
            out_width,
            out_height,
        ),
        (1, 1) => {
            // No upsampling, just crop
            for y in 0..out_height {
                let in_y = y.min(in_height.saturating_sub(1));
                for x in 0..out_width {
                    let in_x = x.min(in_width.saturating_sub(1));
                    output[y * out_width + x] = input[in_y * in_width + in_x];
                }
            }
        }
        _ => {
            // Fall back to nearest-neighbor for unsupported ratios
            upsample_nearest_f32(
                input,
                in_width,
                in_height,
                &mut output,
                out_width,
                out_height,
                scale_x,
                scale_y,
            );
        }
    }
    output
}

/// f32 version of libjpeg-turbo h2v1 upsampling with alternating bias.
fn upsample_h2v1_f32_libjpeg(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y.min(in_height.saturating_sub(1));
        let out_row = out_y * out_width;
        let in_row = in_y * in_width;

        if in_width == 1 {
            let val = input[in_row];
            if out_width > 0 {
                output[out_row] = val;
            }
            if out_width > 1 {
                output[out_row + 1] = val;
            }
            continue;
        }

        // First column
        let curr = input[in_row];
        let next = input[in_row + 1];
        output[out_row] = curr;
        if out_width > 1 {
            output[out_row + 1] = curr * 0.75 + next * 0.25;
        }

        // Interior
        for in_x in 1..in_width.saturating_sub(1) {
            let prev = input[in_row + in_x - 1];
            let curr = input[in_row + in_x];
            let next = input[in_row + in_x + 1];
            let left_out = in_x * 2;
            let right_out = left_out + 1;
            if left_out < out_width {
                output[out_row + left_out] = curr * 0.75 + prev * 0.25;
            }
            if right_out < out_width {
                output[out_row + right_out] = curr * 0.75 + next * 0.25;
            }
        }

        // Last column
        let last = in_width - 1;
        let prev = input[in_row + last - 1];
        let curr = input[in_row + last];
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[out_row + left_out] = curr * 0.75 + prev * 0.25;
        }
        if right_out < out_width {
            output[out_row + right_out] = curr;
        }
    }
}

/// f32 version of libjpeg-turbo h1v2 upsampling with alternating bias.
fn upsample_h1v2_f32_libjpeg(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;
        let out_row = out_y * out_width;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_width;
        let far_row = far_y * in_width;

        for out_x in 0..out_width {
            let in_x = out_x.min(in_width.saturating_sub(1));
            let near = input[near_row + in_x];
            let far = input[far_row + in_x];
            output[out_row + out_x] = near * 0.75 + far * 0.25;
        }
    }
}

/// f32 version of libjpeg-turbo fused h2v2 upsampling.
fn upsample_h2v2_f32_libjpeg(
    input: &[f32],
    in_width: usize,
    in_height: usize,
    output: &mut [f32],
    out_width: usize,
    out_height: usize,
) {
    if in_width == 0 || in_height == 0 || out_width == 0 || out_height == 0 {
        return;
    }

    for out_y in 0..out_height {
        let in_y = out_y / 2;
        let in_y_clamped = in_y.min(in_height.saturating_sub(1));
        let is_upper = out_y % 2 == 0;

        let far_y = if is_upper {
            in_y_clamped.saturating_sub(1)
        } else {
            (in_y + 1).min(in_height.saturating_sub(1))
        };

        let near_row = in_y_clamped * in_width;
        let far_row = far_y * in_width;
        let out_row = out_y * out_width;

        if in_width == 1 {
            let colsum = input[near_row] * 3.0 + input[far_row];
            let val = colsum * 0.25;
            if out_width > 0 {
                output[out_row] = val;
            }
            if out_width > 1 {
                output[out_row + 1] = val;
            }
            continue;
        }

        // Column sums
        let this_colsum = input[near_row] * 3.0 + input[far_row];
        let next_colsum = input[near_row + 1] * 3.0 + input[far_row + 1];

        // First column
        output[out_row] = this_colsum * 0.25;
        if out_width > 1 {
            output[out_row + 1] = (this_colsum * 3.0 + next_colsum) / 16.0;
        }

        let mut last_colsum = this_colsum;
        for in_x in 1..in_width.saturating_sub(1) {
            let this_colsum = input[near_row + in_x] * 3.0 + input[far_row + in_x];
            let next_colsum = input[near_row + in_x + 1] * 3.0 + input[far_row + in_x + 1];
            let left_out = in_x * 2;
            let right_out = left_out + 1;
            if left_out < out_width {
                output[out_row + left_out] = (this_colsum * 3.0 + last_colsum) / 16.0;
            }
            if right_out < out_width {
                output[out_row + right_out] = (this_colsum * 3.0 + next_colsum) / 16.0;
            }
            last_colsum = this_colsum;
        }

        // Last column
        let last = in_width - 1;
        let this_colsum = input[near_row + last] * 3.0 + input[far_row + last];
        let left_out = last * 2;
        let right_out = left_out + 1;
        if left_out < out_width {
            output[out_row + left_out] = (this_colsum * 3.0 + last_colsum) / 16.0;
        }
        if right_out < out_width {
            output[out_row + right_out] = this_colsum * 0.25;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2v2_i16_nearest_basic() {
        let input: Vec<i16> = vec![750; 4 * 4];
        let mut output = vec![0i16; 8 * 8];
        upsample_h2v2_i16_nearest(&input, 4, 4, &mut output, 8, 8);
        for &v in &output {
            assert_eq!(v, 750);
        }
    }

    #[test]
    fn h2v1_i16_nearest_basic() {
        let input: Vec<i16> = vec![300; 4];
        let mut output = vec![0i16; 8];
        upsample_h2v1_i16_nearest(&input, 4, 1, &mut output, 8, 1);
        for &v in &output {
            assert_eq!(v, 300);
        }
    }

    #[test]
    fn h1v2_i16_nearest_basic() {
        let input: Vec<i16> = vec![200; 4 * 4];
        let mut output = vec![0i16; 4 * 8];
        upsample_h1v2_i16_nearest(&input, 4, 4, &mut output, 4, 8);
        for &v in &output {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn h2v1_i16_libjpeg_basic() {
        let input: Vec<i16> = vec![400; 8];
        let mut output = vec![0i16; 16];
        upsample_h2v1_i16_libjpeg(&input, 8, 1, &mut output, 16, 1);
        for (i, &v) in output.iter().enumerate() {
            assert!((v - 400).abs() <= 1, "libjpeg h2v1 pixel {i}: {v} != ~400");
        }
    }

    /// Test data: gradient pattern with varying values to exercise edge handling
    fn gradient_test_data(width: usize, height: usize) -> Vec<i16> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                ((x as i32 * 37 + y as i32 * 53) % 500 - 250) as i16
            })
            .collect()
    }

    /// Test data: extreme chroma transitions (worst case for rounding differences)
    fn extreme_test_data(width: usize, height: usize) -> Vec<i16> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                // Alternate between extreme values at block boundaries
                if (x / 4 + y / 4) % 2 == 0 {
                    2000
                } else {
                    -2000
                }
            })
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_libjpeg_row_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test that AVX2 and scalar paths produce identical output for
        // the libjpeg-compat h2v2 fused row upsampler.
        let widths: &[usize] = &[2, 4, 8, 15, 16, 17, 18, 32, 33, 64, 128, 255, 256, 512];

        for &in_width in widths {
            let out_width = in_width * 2;

            for (label, near_data, far_data) in [
                (
                    "gradient",
                    (0..in_width)
                        .map(|x| (x as i32 * 37 % 500 - 250) as i16)
                        .collect::<Vec<_>>(),
                    (0..in_width)
                        .map(|x| (x as i32 * 53 % 500 - 250) as i16)
                        .collect::<Vec<_>>(),
                ),
                (
                    "extreme",
                    (0..in_width)
                        .map(|x| if x % 2 == 0 { 2000i16 } else { -2000 })
                        .collect(),
                    (0..in_width)
                        .map(|x| if x % 2 == 0 { -2000i16 } else { 2000 })
                        .collect(),
                ),
                ("constant", vec![1000i16; in_width], vec![500i16; in_width]),
            ] {
                for bias in [
                    H2v2Bias::Alternating { is_upper: true },
                    H2v2Bias::Alternating { is_upper: false },
                    H2v2Bias::Turbo,
                ] {
                    // Compute reference using scalar path
                    let mut reference = vec![0i16; out_width];
                    upsample_h2v2_libjpeg_row_scalar(
                        &near_data,
                        &far_data,
                        &mut reference,
                        in_width,
                        out_width,
                        bias,
                    );

                    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                        let mut result = vec![0i16; out_width];
                        upsample_h2v2_libjpeg_row(
                            &near_data,
                            &far_data,
                            &mut result,
                            in_width,
                            out_width,
                            bias,
                        );

                        assert_eq!(
                            result, reference,
                            "h2v2_libjpeg_row mismatch: {label} width={in_width} \
                             bias={bias:?} at {perm}"
                        );
                    });

                    if label == "gradient"
                        && in_width == 32
                        && bias == (H2v2Bias::Alternating { is_upper: true })
                    {
                        eprintln!("h2v2_libjpeg_row dispatch: {report}");
                        assert!(
                            report.permutations_run >= 2,
                            "expected at least 2 permutations"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn h2v2_libjpeg_full_dispatch_parity() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test the full h2v2 libjpeg upsampler (multiple rows) for SIMD parity.
        let sizes: &[(usize, usize)] = &[
            (4, 4),
            (8, 8),
            (16, 16),
            (17, 9),
            (32, 32),
            (64, 16),
            (33, 33),
            (128, 64),
        ];

        for &(in_w, in_h) in sizes {
            let out_w = in_w * 2;
            let out_h = in_h * 2;

            for (label, input) in [
                ("gradient", gradient_test_data(in_w, in_h)),
                ("extreme", extreme_test_data(in_w, in_h)),
                ("constant", vec![1000i16; in_w * in_h]),
            ] {
                // Compute reference using scalar path directly
                let mut reference = vec![0i16; out_w * out_h];
                for out_y in 0..out_h {
                    let in_y = out_y / 2;
                    let in_y_clamped = in_y.min(in_h.saturating_sub(1));
                    let is_upper = out_y % 2 == 0;

                    let far_y = if is_upper {
                        in_y_clamped.saturating_sub(1)
                    } else {
                        (in_y + 1).min(in_h.saturating_sub(1))
                    };

                    let near_row = in_y_clamped * in_w;
                    let far_row = far_y * in_w;
                    let out_row = out_y * out_w;

                    upsample_h2v2_libjpeg_row_scalar(
                        &input[near_row..near_row + in_w],
                        &input[far_row..far_row + in_w],
                        &mut reference[out_row..],
                        in_w,
                        out_w,
                        H2v2Bias::Alternating { is_upper },
                    );
                }

                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let mut result = vec![0i16; out_w * out_h];
                    upsample_h2v2_i16_libjpeg(&input, in_w, in_h, &mut result, out_w, out_h);

                    assert_eq!(
                        result, reference,
                        "h2v2_libjpeg_full mismatch: {label} {in_w}x{in_h} at {perm}"
                    );
                });

                if label == "gradient" && in_w == 32 {
                    eprintln!("h2v2_libjpeg_full dispatch: {report}");
                    assert!(
                        report.permutations_run >= 2,
                        "expected at least 2 permutations"
                    );
                }
            }
        }
    }

    /// Simple LCG for reproducible random planes.
    struct Lcg(u64);
    impl Lcg {
        fn next_u8(&mut self) -> i16 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) & 0xFF) as i16
        }
    }

    /// Exact port of libjpeg-turbo's `h2v2_fancy_upsample` (jdsample.c) with
    /// edge-replicated context rows. Turbo uses FIXED rounding biases —
    /// +8 on left outputs, +7 on right outputs — for BOTH rows of a pair.
    /// zenjpeg's Triangle additionally alternates the pair to (7, 8) on
    /// lower rows (the same vertical alternation turbo itself uses in
    /// `h1v2_fancy_upsample`'s +1/+2 biases), so the two implementations
    /// differ by at most ±1, only on odd output rows, only where the
    /// 9:3:3:1 sum lands exactly on the rounding boundary.
    fn turbo_h2v2_fancy_reference(
        input: &[i16],
        in_w: usize,
        in_h: usize,
        out_w: usize,
        out_h: usize,
    ) -> Vec<i16> {
        let mut out = vec![0i16; out_w * out_h];
        for oy in 0..out_h {
            let iy = (oy / 2).min(in_h - 1);
            let fy = if oy % 2 == 0 {
                iy.saturating_sub(1)
            } else {
                (iy + 1).min(in_h - 1)
            };
            let near = &input[iy * in_w..][..in_w];
            let far = &input[fy * in_w..][..in_w];
            let colsum = |x: usize| near[x] as i32 * 3 + far[x] as i32;
            let row = &mut out[oy * out_w..][..out_w];

            // First column (turbo reads the duplicated edge sample when
            // in_w == 1, via its padded row buffers).
            let mut this = colsum(0);
            row[0] = ((this * 4 + 8) >> 4) as i16;
            if out_w > 1 {
                let next = colsum(1.min(in_w - 1));
                row[1] = ((this * 3 + next + 7) >> 4) as i16;
            }
            let mut last = this;
            if in_w >= 2 {
                this = colsum(1);
            }
            for ix in 1..in_w.saturating_sub(1) {
                let next = colsum(ix + 1);
                if ix * 2 < out_w {
                    row[ix * 2] = ((this * 3 + last + 8) >> 4) as i16;
                }
                if ix * 2 + 1 < out_w {
                    row[ix * 2 + 1] = ((this * 3 + next + 7) >> 4) as i16;
                }
                last = this;
                this = next;
            }
            if in_w >= 2 {
                let lx = in_w - 1;
                if lx * 2 < out_w {
                    row[lx * 2] = ((this * 3 + last + 8) >> 4) as i16;
                }
                if lx * 2 + 1 < out_w {
                    row[lx * 2 + 1] = ((this * 4 + 7) >> 4) as i16;
                }
            }
        }
        out
    }

    /// Pin the exact relationship between zenjpeg's Triangle h2v2 and
    /// libjpeg-turbo's fancy upsample: even output rows are bit-identical,
    /// odd rows differ by at most ±1 (the row-alternated rounding bias),
    /// and the divergence rate is reported.
    #[test]
    fn h2v2_triangle_vs_libjpeg_turbo_reference() {
        let mut rng = Lcg(0xFA9C_75A3_11E0_2B4D);
        let mut diffs = 0u64;
        let mut total = 0u64;
        let mut odd_row_pixels = 0u64;

        let sizes: &[(usize, usize)] = &[(32, 24), (31, 17), (8, 8), (64, 48), (2, 2), (16, 1)];
        for &(in_w, in_h) in sizes {
            for trial in 0..40 {
                let input: Vec<i16> = (0..in_w * in_h).map(|_| rng.next_u8()).collect();
                // Even and odd output dims (odd = image width/height 2w-1)
                let (out_w, out_h) = if trial % 2 == 0 {
                    (in_w * 2, in_h * 2)
                } else {
                    (in_w * 2 - 1, (in_h * 2).saturating_sub(1).max(1))
                };

                let mut ours = vec![0i16; out_w * out_h];
                upsample_h2v2_i16_libjpeg(&input, in_w, in_h, &mut ours, out_w, out_h);
                let turbo = turbo_h2v2_fancy_reference(&input, in_w, in_h, out_w, out_h);

                // The Turbo bias variant (selected by IdctMethod::Libjpeg)
                // must be bit-identical to libjpeg-turbo, every row, every
                // size — including single-column planes.
                let mut ours_turbo = vec![0i16; out_w * out_h];
                upsample_h2v2_i16_libjpeg_turbo(&input, in_w, in_h, &mut ours_turbo, out_w, out_h);
                assert_eq!(
                    ours_turbo, turbo,
                    "turbo-bias variant diverges from the turbo reference \
                     ({in_w}x{in_h} trial={trial})"
                );

                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let a = ours[oy * out_w + ox];
                        let b = turbo[oy * out_w + ox];
                        let d = (a - b).abs();
                        assert!(
                            d <= 1,
                            "h2v2 vs turbo diff > 1 at ({ox},{oy}) {in_w}x{in_h}: {a} vs {b}"
                        );
                        if oy % 2 == 0 {
                            assert_eq!(
                                a, b,
                                "even rows must be bit-identical, ({ox},{oy}) {in_w}x{in_h}"
                            );
                        } else {
                            odd_row_pixels += 1;
                            if d != 0 {
                                diffs += 1;
                            }
                        }
                        total += 1;
                    }
                }
            }
        }
        // Single-column planes (in_w == 1): the turbo variant matches the
        // reference's padded-buffer behavior ((4c+8)/(4c+7)); the
        // alternating mode keeps the legacy duplicated first output there,
        // so only the turbo variant is asserted exact.
        for trial in 0..10 {
            let input: Vec<i16> = (0..16).map(|_| rng.next_u8()).collect();
            let (in_w, in_h, out_w, out_h) = (1usize, 16usize, 2usize, 32usize);
            let turbo_ref = turbo_h2v2_fancy_reference(&input, in_w, in_h, out_w, out_h);
            let mut ours_turbo = vec![0i16; out_w * out_h];
            upsample_h2v2_i16_libjpeg_turbo(&input, in_w, in_h, &mut ours_turbo, out_w, out_h);
            assert_eq!(
                ours_turbo, turbo_ref,
                "single-column turbo mismatch trial={trial}"
            );
        }

        eprintln!(
            "h2v2 Triangle vs libjpeg-turbo fancy: {diffs}/{total} pixels differ \
             ({:.2}% overall, {:.2}% of odd rows), all by ±1, even rows exact",
            100.0 * diffs as f64 / total as f64,
            100.0 * diffs as f64 / odd_row_pixels as f64
        );
        assert!(diffs > 0, "expected the documented odd-row bias divergence");
    }

    /// Accuracy of the two h2v2 rounding-bias schemes against the exact
    /// real-valued 9:3:3:1 filter. Both schemes have max error 0.5 and
    /// near-zero global bias; they differ only in the spatial STRUCTURE of
    /// the half-case rounding (turbo: constant per column parity; zenjpeg:
    /// alternating per row, checkerboard-like).
    #[test]
    fn h2v2_bias_schemes_accuracy_vs_ideal() {
        let mut rng = Lcg(0x1DEA_1DEA_1DEA_1DEA);
        let (in_w, in_h) = (64, 64);
        let (out_w, out_h) = (128, 128);

        // err[scheme][row parity][col parity] -> (sum, n)
        let mut sums = [[[0.0f64; 2]; 2]; 2];
        let mut counts = [[[0u64; 2]; 2]; 2];
        let mut max_abs = [0.0f64; 2];

        for _ in 0..30 {
            let input: Vec<i16> = (0..in_w * in_h).map(|_| rng.next_u8()).collect();

            let mut zen = vec![0i16; out_w * out_h];
            upsample_h2v2_i16_libjpeg(&input, in_w, in_h, &mut zen, out_w, out_h);
            let turbo = turbo_h2v2_fancy_reference(&input, in_w, in_h, out_w, out_h);

            // Exact real-valued reference via the f32 path's formula in f64.
            for oy in 0..out_h {
                let iy = (oy / 2).min(in_h - 1);
                let fy = if oy % 2 == 0 {
                    iy.saturating_sub(1)
                } else {
                    (iy + 1).min(in_h - 1)
                };
                for ox in 0..out_w {
                    let ix = (ox / 2).min(in_w - 1);
                    let hx = if ox % 2 == 0 {
                        ix.saturating_sub(1)
                    } else {
                        (ix + 1).min(in_w - 1)
                    };
                    let near_this = input[iy * in_w + ix] as f64;
                    let near_adj = input[iy * in_w + hx] as f64;
                    let far_this = input[fy * in_w + ix] as f64;
                    let far_adj = input[fy * in_w + hx] as f64;
                    let ideal =
                        (9.0 * near_this + 3.0 * near_adj + 3.0 * far_this + far_adj) / 16.0;

                    for (s, out) in [(0usize, &zen), (1usize, &turbo)] {
                        let e = out[oy * out_w + ox] as f64 - ideal;
                        sums[s][oy % 2][ox % 2] += e;
                        counts[s][oy % 2][ox % 2] += 1;
                        max_abs[s] = max_abs[s].max(e.abs());
                    }
                }
            }
        }

        for (s, name) in [(0usize, "zen-alternating"), (1usize, "turbo-fixed")] {
            let cell = |r: usize, c: usize| sums[s][r][c] / counts[s][r][c] as f64;
            let total_n: u64 = counts[s].iter().flatten().sum();
            let total: f64 = sums[s].iter().flatten().sum::<f64>() / total_n as f64;
            eprintln!(
                "{name}: max|err|={:.3} global mean={total:+.5} \
                 per-cell [r0c0 {:+.4}, r0c1 {:+.4}, r1c0 {:+.4}, r1c1 {:+.4}]",
                max_abs[s],
                cell(0, 0),
                cell(0, 1),
                cell(1, 0),
                cell(1, 1)
            );
            assert!(max_abs[s] <= 0.5 + 1e-9, "{name} exceeds half-step error");
            assert!(total.abs() < 0.01, "{name} global bias too large");
        }
    }

    /// The magetypes-generic 4:2:2 (h2v1) upsampler must be BYTE-IDENTICAL to
    /// the scalar reference across widths (incl. the SIMD-chunk boundary at
    /// 10), heights, strides, and odd output widths — and across every SIMD
    /// tier. Decoded pixels are sacred; the SIMD interior is a speed-only swap.
    #[test]
    fn h2v1_generic_matches_scalar_bit_exact() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            for in_w in [1usize, 2, 7, 9, 10, 11, 16, 17, 31, 64, 100] {
                for in_h in [1usize, 3] {
                    for (in_stride, pad) in [(in_w, 0usize), (in_w + 5, 5)] {
                        let _ = pad;
                        let input = gradient_test_data(in_stride, in_h);
                        for out_w in [in_w * 2, (in_w * 2).saturating_sub(1).max(1)] {
                            let out_stride = out_w + 3;
                            let n = out_stride * in_h;
                            let mut a = vec![0i16; n];
                            let mut b = vec![0i16; n];
                            upsample_h2v1_i16_libjpeg_strided_scalar(
                                &input, in_w, in_stride, in_h, &mut a, out_w, out_stride, in_h,
                            );
                            upsample_h2v1_i16_libjpeg_strided(
                                &input, in_w, in_stride, in_h, &mut b, out_w, out_stride, in_h,
                            );
                            assert_eq!(
                                a, b,
                                "h2v1 generic != scalar: in_w={in_w} in_h={in_h} \
                                 in_stride={in_stride} out_w={out_w} at {perm}"
                            );
                        }
                    }
                }
            }
        });
        eprintln!("h2v1 generic vs scalar bit-exact: {report}");
        assert!(report.permutations_run >= 2, "expected >=2 SIMD tiers");
    }

    /// The magetypes-generic h2v2 (4:2:0) row must be BYTE-IDENTICAL to the
    /// scalar row across widths, out-widths, and both bias modes, on every
    /// SIMD tier. (x86 production uses the hand AVX2 kernel; the generic is
    /// the non-x86 path — this test exercises it directly via incant!.)
    #[test]
    fn h2v2_row_generic_matches_scalar_bit_exact() {
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            for in_w in [1usize, 2, 7, 9, 10, 11, 16, 17, 31, 64] {
                let near = gradient_test_data(in_w, 1);
                let far = extreme_test_data(in_w, 1);
                for out_w in [in_w * 2, (in_w * 2).saturating_sub(1).max(1)] {
                    for bias in [
                        H2v2Bias::Alternating { is_upper: true },
                        H2v2Bias::Alternating { is_upper: false },
                        H2v2Bias::Turbo,
                    ] {
                        let mut a = vec![0i16; out_w];
                        let mut b = vec![0i16; out_w];
                        upsample_h2v2_libjpeg_row_scalar(&near, &far, &mut a, in_w, out_w, bias);
                        incant!(upsample_h2v2_libjpeg_row_generic(
                            &near, &far, &mut b, in_w, out_w, bias
                        ));
                        assert_eq!(
                            a, b,
                            "h2v2 generic != scalar: in_w={in_w} out_w={out_w} bias={bias:?} at {perm}"
                        );
                    }
                }
            }
        });
        eprintln!("h2v2 row generic vs scalar bit-exact: {report}");
        assert!(report.permutations_run >= 2, "expected >=2 SIMD tiers");
    }
}
