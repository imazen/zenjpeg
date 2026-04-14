//! Color matrix standards, signal ranges, and coefficient computation.

/// Color matrix standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    /// ITU-R BT.601 (SD video, JFIF JPEG). Kr=0.299, Kb=0.114.
    Bt601,
    /// ITU-R BT.709 (HD video). Kr=0.2126, Kb=0.0722.
    Bt709,
    /// ITU-R BT.2020 (UHD/HDR video). Kr=0.2627, Kb=0.0593.
    Bt2020,
}

/// Signal range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    /// Full range: Y `[0,255]`, Cb/Cr `[0,255]` centered at 128.
    Full,
    /// Limited (studio) range: Y `[16,235]`, Cb/Cr `[16,240]`.
    Limited,
}

/// Forward (RGB->YCbCr) coefficients, precomputed at 15-bit fixed-point.
#[derive(Clone, Copy, Debug)]
pub struct ForwardCoeffs {
    pub yr: i16,
    pub yg: i16,
    pub yb: i16,
    pub cb_r: i16,
    pub cb_g: i16,
    pub cb_b: i16,
    pub cr_r: i16,
    pub cr_g: i16,
    pub cr_b: i16,
    /// bias_y * (1<<PREC) + round
    pub y_bias: i32,
    /// bias_uv * (1<<PREC) + round
    pub uv_bias: i32,
    /// bias_uv * (1<<(PREC+1)) + round (for 4:2:0 fused kernel)
    pub uv_bias_420: i32,
    /// Floating-point versions for generic kernels.
    pub yr_f: f32,
    pub yg_f: f32,
    pub yb_f: f32,
    pub cb_r_f: f32,
    pub cb_g_f: f32,
    pub cb_b_f: f32,
    pub cr_r_f: f32,
    pub cr_g_f: f32,
    pub cr_b_f: f32,
    pub y_bias_f: f32,
    pub uv_bias_f: f32,
}

/// Inverse (YCbCr->RGB) coefficients for decode.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct InverseCoeffs {
    /// Y scale factor (1.0 for full range, 255/219 for limited)
    pub y_coeff: f32,
    /// Cr contribution to R channel
    pub cr_to_r: f32,
    /// Cr contribution to G channel
    pub cr_to_g: f32,
    /// Cb contribution to G channel
    pub cb_to_g: f32,
    /// Cb contribution to B channel
    pub cb_to_b: f32,
    /// Y offset before scaling (0 for full, -16 for limited)
    pub y_offset: f32,
    /// UV offset (always -128)
    pub uv_offset: f32,
    /// 15-bit fixed-point versions for integer kernels.
    #[allow(dead_code)]
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub y_coeff_i: i32,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub cr_to_r_i: i32,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub cr_to_g_i: i32,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub cb_to_g_i: i32,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub cb_to_b_i: i32,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub y_offset_i: i32,
}

/// Fixed-point precision used for integer kernels.
pub(crate) const PREC: i32 = 15;

/// Pack a pair of i16 coefficients into a 32-bit value so pmaddwd reads them
/// as `(low, high)` and computes `a*x + b*y` per i32 lane.
pub(crate) const fn pack_i16_pair(a: i16, b: i16) -> i32 {
    ((a as u16 as u32) | ((b as u16 as u32) << 16)) as i32
}

/// const-compatible f32 round-to-nearest (ties away from zero).
/// Equivalent to `f32::round()` but works in `const fn` on MSRV 1.85.
const fn const_round(v: f32) -> i32 {
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
}

/// Precomputed coefficient tables for all 6 (Matrix, Range) combinations.
/// Avoids runtime computation entirely.
pub(crate) const BT601_FULL: ForwardCoeffs = ForwardCoeffs::compute(Matrix::Bt601, Range::Full);
pub(crate) const BT601_LIMITED: ForwardCoeffs =
    ForwardCoeffs::compute(Matrix::Bt601, Range::Limited);
pub(crate) const BT709_FULL: ForwardCoeffs = ForwardCoeffs::compute(Matrix::Bt709, Range::Full);
pub(crate) const BT709_LIMITED: ForwardCoeffs =
    ForwardCoeffs::compute(Matrix::Bt709, Range::Limited);
pub(crate) const BT2020_FULL: ForwardCoeffs = ForwardCoeffs::compute(Matrix::Bt2020, Range::Full);
pub(crate) const BT2020_LIMITED: ForwardCoeffs =
    ForwardCoeffs::compute(Matrix::Bt2020, Range::Limited);

pub(crate) const INV_BT601_FULL: InverseCoeffs = InverseCoeffs::compute(Matrix::Bt601, Range::Full);
pub(crate) const INV_BT601_LIMITED: InverseCoeffs =
    InverseCoeffs::compute(Matrix::Bt601, Range::Limited);
pub(crate) const INV_BT709_FULL: InverseCoeffs = InverseCoeffs::compute(Matrix::Bt709, Range::Full);
pub(crate) const INV_BT709_LIMITED: InverseCoeffs =
    InverseCoeffs::compute(Matrix::Bt709, Range::Limited);
pub(crate) const INV_BT2020_FULL: InverseCoeffs =
    InverseCoeffs::compute(Matrix::Bt2020, Range::Full);
pub(crate) const INV_BT2020_LIMITED: InverseCoeffs =
    InverseCoeffs::compute(Matrix::Bt2020, Range::Limited);

impl ForwardCoeffs {
    /// Look up precomputed coefficients. Zero runtime cost.
    pub const fn new(matrix: Matrix, range: Range) -> Self {
        match (matrix, range) {
            (Matrix::Bt601, Range::Full) => BT601_FULL,
            (Matrix::Bt601, Range::Limited) => BT601_LIMITED,
            (Matrix::Bt709, Range::Full) => BT709_FULL,
            (Matrix::Bt709, Range::Limited) => BT709_LIMITED,
            (Matrix::Bt2020, Range::Full) => BT2020_FULL,
            (Matrix::Bt2020, Range::Limited) => BT2020_LIMITED,
        }
    }

    /// Compute coefficients at compile time.
    const fn compute(matrix: Matrix, range: Range) -> Self {
        let (kr_f64, kb_f64) = matrix.kr_kb();
        // Use f32 for coefficient computation to match yuv crate behavior.
        let kr = kr_f64 as f32;
        let kb = kb_f64 as f32;
        let kg = 1.0f32 - kr - kb;

        match range {
            Range::Full => {
                // Full range (range_y = range_uv = range_rgba = 255).
                // yuv crate: yr = kr * range_y / range_rgba = kr * 1.0 = kr
                let yr_f = kr;
                let yg_f = kg;
                let yb_f = kb;
                let cb_r_f = -0.5f32 * kr / (1.0 - kb);
                let cb_g_f = -0.5f32 * kg / (1.0 - kb);
                let cb_b_f = 0.5f32;
                let cr_r_f = 0.5f32;
                let cr_g_f = -0.5f32 * kg / (1.0 - kr);
                let cr_b_f = -0.5f32 * kb / (1.0 - kr);

                let scale = (1 << PREC) as f32;
                let round = (1i32 << (PREC - 1)) - 1;
                let round_420 = (1i32 << PREC) - 1;

                Self {
                    yr: const_round(yr_f * scale) as i16,
                    yg: const_round(yg_f * scale) as i16,
                    yb: const_round(yb_f * scale) as i16,
                    cb_r: const_round(cb_r_f * scale) as i16,
                    cb_g: const_round(cb_g_f * scale) as i16,
                    cb_b: const_round(cb_b_f * scale) as i16,
                    cr_r: const_round(cr_r_f * scale) as i16,
                    cr_g: const_round(cr_g_f * scale) as i16,
                    cr_b: const_round(cr_b_f * scale) as i16,
                    y_bias: round,
                    uv_bias: (128i32 << PREC) + round,
                    uv_bias_420: (128i32 << (PREC + 1)) + round_420,
                    yr_f,
                    yg_f,
                    yb_f,
                    cb_r_f,
                    cb_g_f,
                    cb_b_f,
                    cr_r_f,
                    cr_g_f,
                    cr_b_f,
                    y_bias_f: 0.0,
                    uv_bias_f: 128.0,
                }
            }
            Range::Limited => {
                // Limited range: Y [16, 235], Cb/Cr [16, 240].
                // yuv crate: range_y=219, range_uv=224, range_rgba=255.
                let range_y = 219.0f32;
                let range_uv = 224.0f32;
                let range_rgba = 255.0f32;

                let yr_f = kr * range_y / range_rgba;
                let yg_f = kg * range_y / range_rgba;
                let yb_f = kb * range_y / range_rgba;
                let cb_r_f = -0.5f32 * kr / (1.0 - kb) * range_uv / range_rgba;
                let cb_g_f = -0.5f32 * kg / (1.0 - kb) * range_uv / range_rgba;
                let cb_b_f = 0.5f32 * range_uv / range_rgba;
                let cr_r_f = 0.5f32 * range_uv / range_rgba;
                let cr_g_f = -0.5f32 * kg / (1.0 - kr) * range_uv / range_rgba;
                let cr_b_f = -0.5f32 * kb / (1.0 - kr) * range_uv / range_rgba;

                let scale = (1 << PREC) as f32;
                let round = (1i32 << (PREC - 1)) - 1;
                let round_420 = (1i32 << PREC) - 1;

                Self {
                    yr: const_round(yr_f * scale) as i16,
                    yg: const_round(yg_f * scale) as i16,
                    yb: const_round(yb_f * scale) as i16,
                    cb_r: const_round(cb_r_f * scale) as i16,
                    cb_g: const_round(cb_g_f * scale) as i16,
                    cb_b: const_round(cb_b_f * scale) as i16,
                    cr_r: const_round(cr_r_f * scale) as i16,
                    cr_g: const_round(cr_g_f * scale) as i16,
                    cr_b: const_round(cr_b_f * scale) as i16,
                    y_bias: (16i32 << PREC) + round,
                    uv_bias: (128i32 << PREC) + round,
                    uv_bias_420: (128i32 << (PREC + 1)) + round_420,
                    yr_f,
                    yg_f,
                    yb_f,
                    cb_r_f,
                    cb_g_f,
                    cb_b_f,
                    cr_r_f,
                    cr_g_f,
                    cr_b_f,
                    y_bias_f: 16.0,
                    uv_bias_f: 128.0,
                }
            }
        }
    }
}

impl InverseCoeffs {
    /// Look up precomputed inverse coefficients. Zero runtime cost.
    pub const fn new(matrix: Matrix, range: Range) -> Self {
        match (matrix, range) {
            (Matrix::Bt601, Range::Full) => INV_BT601_FULL,
            (Matrix::Bt601, Range::Limited) => INV_BT601_LIMITED,
            (Matrix::Bt709, Range::Full) => INV_BT709_FULL,
            (Matrix::Bt709, Range::Limited) => INV_BT709_LIMITED,
            (Matrix::Bt2020, Range::Full) => INV_BT2020_FULL,
            (Matrix::Bt2020, Range::Limited) => INV_BT2020_LIMITED,
        }
    }

    /// Compute inverse coefficients at compile time.
    const fn compute(matrix: Matrix, range: Range) -> Self {
        let (kr, kb) = matrix.kr_kb();
        let kg = 1.0 - kr - kb;

        // Base inverse matrix (from full-range):
        // R = Y + (2*(1-Kr))*Cr'
        // G = Y - (2*Kb*(1-Kb)/Kg)*Cb' - (2*Kr*(1-Kr)/Kg)*Cr'
        // B = Y + (2*(1-Kb))*Cb'
        // where Cb' = Cb - 128, Cr' = Cr - 128

        let cr_to_r_base = 2.0 * (1.0 - kr);
        let cb_to_g_base = -2.0 * kb * (1.0 - kb) / kg;
        let cr_to_g_base = -2.0 * kr * (1.0 - kr) / kg;
        let cb_to_b_base = 2.0 * (1.0 - kb);

        match range {
            Range::Full => {
                let scale = (1 << PREC) as f64;
                Self {
                    y_coeff: 1.0,
                    cr_to_r: cr_to_r_base as f32,
                    cr_to_g: cr_to_g_base as f32,
                    cb_to_g: cb_to_g_base as f32,
                    cb_to_b: cb_to_b_base as f32,
                    y_offset: 0.0,
                    uv_offset: -128.0,
                    y_coeff_i: (1.0 * scale + 0.5) as i32,
                    cr_to_r_i: (cr_to_r_base * scale + 0.5) as i32,
                    cr_to_g_i: (cr_to_g_base * scale - 0.5) as i32,
                    cb_to_g_i: (cb_to_g_base * scale - 0.5) as i32,
                    cb_to_b_i: (cb_to_b_base * scale + 0.5) as i32,
                    y_offset_i: 0,
                }
            }
            Range::Limited => {
                // For limited range:
                // Y_full = (Y_limited - 16) * 255/219
                // Cb_full = (Cb_limited - 128) * 255/224
                // Cr_full = (Cr_limited - 128) * 255/224
                let y_scale = 255.0 / 219.0;
                let uv_scale = 255.0 / 224.0;

                let cr_to_r = cr_to_r_base * uv_scale;
                let cr_to_g = cr_to_g_base * uv_scale;
                let cb_to_g = cb_to_g_base * uv_scale;
                let cb_to_b = cb_to_b_base * uv_scale;

                let scale = (1 << PREC) as f64;
                Self {
                    y_coeff: y_scale as f32,
                    cr_to_r: cr_to_r as f32,
                    cr_to_g: cr_to_g as f32,
                    cb_to_g: cb_to_g as f32,
                    cb_to_b: cb_to_b as f32,
                    y_offset: -16.0,
                    uv_offset: -128.0,
                    y_coeff_i: (y_scale * scale + 0.5) as i32,
                    cr_to_r_i: (cr_to_r * scale + 0.5) as i32,
                    cr_to_g_i: (cr_to_g * scale - 0.5) as i32,
                    cb_to_g_i: (cb_to_g * scale - 0.5) as i32,
                    cb_to_b_i: (cb_to_b * scale + 0.5) as i32,
                    y_offset_i: -16,
                }
            }
        }
    }
}

impl Matrix {
    /// Return (Kr, Kb) for this matrix standard as f64 for precision.
    const fn kr_kb(self) -> (f64, f64) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt2020 => (0.2627, 0.0593),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt601_full_coefficients_match_yuv_crate() {
        let c = ForwardCoeffs::new(Matrix::Bt601, Range::Full);
        // Verify these match the yuv crate's f32-rounded computation.
        // yuv crate: (coefficient_f32 * (1 << 15) as f32).round() as i32
        assert_eq!(c.yr, 9798);
        assert_eq!(c.yg, 19235);
        assert_eq!(c.yb, 3736); // 0.114 * 32768 = 3735.55 -> round = 3736
        assert_eq!(c.cb_r, -5529);
        assert_eq!(c.cb_g, -10855);
        assert_eq!(c.cb_b, 16384);
        assert_eq!(c.cr_r, 16384);
        assert_eq!(c.cr_g, -13720);
        assert_eq!(c.cr_b, -2664);
    }

    #[test]
    fn inverse_full_bt601_roundtrip() {
        let inv = InverseCoeffs::new(Matrix::Bt601, Range::Full);
        // Pure white: Y=255, Cb=128, Cr=128 -> R=255, G=255, B=255
        let y = 255.0f32;
        let cb = 128.0f32 + inv.uv_offset;
        let cr = 128.0f32 + inv.uv_offset;
        let r = (y * inv.y_coeff + cr * inv.cr_to_r).round();
        let g = (y * inv.y_coeff + cb * inv.cb_to_g + cr * inv.cr_to_g).round();
        let b = (y * inv.y_coeff + cb * inv.cb_to_b).round();
        assert!((r - 255.0).abs() < 1.0, "R={r}");
        assert!((g - 255.0).abs() < 1.0, "G={g}");
        assert!((b - 255.0).abs() < 1.0, "B={b}");
    }
}
