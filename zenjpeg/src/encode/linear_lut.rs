//! Fast linear RGB to sRGB/YCbCr conversion using the linear-srgb crate.
//!
//! The linear-srgb crate provides SIMD-optimized rational polynomial approximations
//! for the sRGB transfer function, dispatched via archmage tokens (AVX2/NEON/WASM128).
//!
//! The x8 functions use archmage `#[arcane]` entry points to fuse u16->f32 conversion,
//! sRGB transfer, x255 scaling, and YCbCr matrix multiply into a single SIMD pass,
//! eliminating per-call dispatch overhead from `linear_to_srgb_slice`.

use crate::foundation::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
};
#[cfg(target_arch = "x86_64")]
use archmage::SimdToken;

/// Cb/Cr offset (128.0 for 8-bit JPEG)
const CHROMA_OFFSET: f32 = 128.0;

/// sRGB transfer function (linear -> sRGB) via linear-srgb crate.
/// Uses rational polynomial approximation (no powf).
#[inline]
fn linear_to_srgb(x: f32) -> f32 {
    linear_srgb::default::linear_to_srgb(x)
}

// ============================================================================
// 16-bit input conversion
// ============================================================================

/// Convert linear u16 [0, 65535] to sRGB f32 [0, 255].
#[inline]
pub fn linear_u16_to_srgb_255(value: u16) -> f32 {
    let linear = value as f32 / 65535.0;
    linear_to_srgb_255(linear)
}

/// Convert linear RGB16 pixel to YCbCr f32.
#[inline]
pub fn linear_rgb16_to_ycbcr(r: u16, g: u16, b: u16) -> (f32, f32, f32) {
    let r = linear_u16_to_srgb_255(r);
    let g = linear_u16_to_srgb_255(g);
    let b = linear_u16_to_srgb_255(b);

    let y = YCBCR_R_TO_Y.mul_add(r, YCBCR_G_TO_Y.mul_add(g, YCBCR_B_TO_Y * b));
    let cb = YCBCR_R_TO_CB.mul_add(r, YCBCR_G_TO_CB.mul_add(g, YCBCR_B_TO_CB * b)) + CHROMA_OFFSET;
    let cr = YCBCR_R_TO_CR.mul_add(r, YCBCR_G_TO_CR.mul_add(g, YCBCR_B_TO_CR * b)) + CHROMA_OFFSET;

    (y, cb, cr)
}

// ============================================================================
// f32 input conversion
// ============================================================================

/// Convert linear f32 [0, 1] to sRGB [0, 255].
///
/// Uses direct transfer function computation (no LUTs).
/// Values > 1.0 are tone-mapped with Reinhard.
#[inline]
pub fn linear_to_srgb_255(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    // Handle HDR with Reinhard tone mapping
    let x = if x > 1.0 { x / (1.0 + x) } else { x };

    linear_to_srgb(x) * 255.0
}

/// Fast sRGB conversion using direct computation.
#[inline]
#[allow(dead_code)]
pub fn linear_to_srgb_fast(x: f32) -> f32 {
    linear_to_srgb_255(x) / 255.0
}

/// Convert linear f32 [0,1] to sRGB f32 [0, 255] using fast computation.
#[inline]
pub fn linear_f32_to_srgb_255_fast(x: f32) -> f32 {
    linear_to_srgb_255(x)
}

/// Alias for `linear_to_srgb_255` (compatibility).
#[cfg(test)]
#[inline]
pub fn linear_f32_to_srgb_255_lut(x: f32) -> f32 {
    linear_to_srgb_255(x)
}

/// Convert linear RGB f32 pixel to YCbCr f32.
#[inline]
pub fn linear_rgbf32_to_ycbcr_fast(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r = linear_to_srgb_255(r);
    let g = linear_to_srgb_255(g);
    let b = linear_to_srgb_255(b);

    let y = YCBCR_R_TO_Y.mul_add(r, YCBCR_G_TO_Y.mul_add(g, YCBCR_B_TO_Y * b));
    let cb = YCBCR_R_TO_CB.mul_add(r, YCBCR_G_TO_CB.mul_add(g, YCBCR_B_TO_CB * b)) + CHROMA_OFFSET;
    let cr = YCBCR_R_TO_CR.mul_add(r, YCBCR_G_TO_CR.mul_add(g, YCBCR_B_TO_CR * b)) + CHROMA_OFFSET;

    (y, cb, cr)
}

/// Alias for `linear_rgbf32_to_ycbcr_fast` (compatibility).
#[inline]
#[allow(dead_code)]
pub fn linear_rgbf32_to_ycbcr_lut(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    linear_rgbf32_to_ycbcr_fast(r, g, b)
}

// ============================================================================
// SIMD implementations (8-wide)
// ============================================================================

// x86_64 SIMD: fused u16->sRGB->x255->YCbCr using archmage #[arcane] + linear-srgb tokens.
// Eliminates per-call incant! dispatch overhead from linear_to_srgb_slice and ensures
// u16->f32 and x255 scaling are vectorized (they don't autovectorize at trip count 8).
#[cfg(target_arch = "x86_64")]
mod simd_fused {
    use super::*;
    use archmage::{X64V3Token, arcane, rite};
    use magetypes::simd::f32x8 as mt_f32x8;

    /// Convert 8 u16 values to f32 and divide by 65535.0, all in SIMD.
    #[rite]
    fn u16x8_to_linear_f32(token: X64V3Token, values: &[u16; 8]) -> mt_f32x8 {
        // Convert u16 to f32 array, then load + scale in SIMD.
        // The scalar u16->f32 casts are cheap (single vcvtsi2ss each) and the
        // subsequent SIMD multiply ensures the division is vectorized.
        let arr = [
            values[0] as f32,
            values[1] as f32,
            values[2] as f32,
            values[3] as f32,
            values[4] as f32,
            values[5] as f32,
            values[6] as f32,
            values[7] as f32,
        ];
        let v = mt_f32x8::from_array(token, arr);
        v * mt_f32x8::splat(token, 1.0 / 65535.0)
    }

    /// Apply sRGB transfer and x255 scaling to an f32x8 of linear [0,1] values.
    /// Returns sRGB values in [0, 255].
    #[rite]
    fn linear_to_srgb_255_simd(token: X64V3Token, linear: mt_f32x8) -> mt_f32x8 {
        let srgb_arr = linear_srgb::tokens::x8::linear_to_srgb_v3(token, linear.to_array());
        mt_f32x8::from_array(token, srgb_arr) * mt_f32x8::splat(token, 255.0)
    }

    /// Fused linear u16->sRGB->x255 for 8 values.
    #[arcane]
    pub(super) fn linear_u16_to_srgb_255_x8_v3(token: X64V3Token, values: &[u16; 8]) -> [f32; 8] {
        let linear = u16x8_to_linear_f32(token, values);
        linear_to_srgb_255_simd(token, linear).to_array()
    }

    /// Fused linear f32->Reinhard->sRGB->x255 for 8 values.
    #[arcane]
    pub(super) fn linear_to_srgb_255_x8_v3(token: X64V3Token, x: &[f32; 8]) -> [f32; 8] {
        let zero = mt_f32x8::zero(token);
        let one = mt_f32x8::splat(token, 1.0);
        let v = mt_f32x8::from_array(token, *x).max(zero);

        // Reinhard tone mapping for HDR values: x / (1 + x)
        // For values <= 1.0, use as-is
        let reinhard = v / (one + v);
        let mask = v.simd_gt(one);
        let clamped = mt_f32x8::blend(mask, reinhard, v);

        linear_to_srgb_255_simd(token, clamped).to_array()
    }

    /// Fused linear RGB16->sRGB->YCbCr for 8 pixels.
    /// Single dispatch, single target_feature context for all operations.
    #[arcane]
    pub(super) fn linear_rgb16_to_ycbcr_x8_v3(
        token: X64V3Token,
        r: &[u16; 8],
        g: &[u16; 8],
        b: &[u16; 8],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        // u16->f32->sRGB->x255 for each channel
        let r = linear_to_srgb_255_simd(token, u16x8_to_linear_f32(token, r));
        let g = linear_to_srgb_255_simd(token, u16x8_to_linear_f32(token, g));
        let b = linear_to_srgb_255_simd(token, u16x8_to_linear_f32(token, b));

        // BT.601 RGB->YCbCr matrix multiply in SIMD
        let kr_y = mt_f32x8::splat(token, YCBCR_R_TO_Y);
        let kg_y = mt_f32x8::splat(token, YCBCR_G_TO_Y);
        let kb_y = mt_f32x8::splat(token, YCBCR_B_TO_Y);
        let kr_cb = mt_f32x8::splat(token, YCBCR_R_TO_CB);
        let kg_cb = mt_f32x8::splat(token, YCBCR_G_TO_CB);
        let kb_cb = mt_f32x8::splat(token, YCBCR_B_TO_CB);
        let kr_cr = mt_f32x8::splat(token, YCBCR_R_TO_CR);
        let kg_cr = mt_f32x8::splat(token, YCBCR_G_TO_CR);
        let kb_cr = mt_f32x8::splat(token, YCBCR_B_TO_CR);
        let offset = mt_f32x8::splat(token, CHROMA_OFFSET);

        let y = kr_y.mul_add(r, kg_y.mul_add(g, kb_y * b));
        let cb = kr_cb.mul_add(r, kg_cb.mul_add(g, kb_cb * b)) + offset;
        let cr = kr_cr.mul_add(r, kg_cr.mul_add(g, kb_cr * b)) + offset;

        (y.to_array(), cb.to_array(), cr.to_array())
    }

    /// Fused linear RGB f32->sRGB->YCbCr for 8 pixels.
    #[arcane]
    pub(super) fn linear_rgbf32_to_ycbcr_x8_v3(
        token: X64V3Token,
        r: &[f32; 8],
        g: &[f32; 8],
        b: &[f32; 8],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let zero = mt_f32x8::zero(token);
        let one = mt_f32x8::splat(token, 1.0);

        // Reinhard tone mapping + sRGB for each channel
        let process = |x: &[f32; 8]| -> mt_f32x8 {
            let v = mt_f32x8::from_array(token, *x).max(zero);
            let reinhard = v / (one + v);
            let mask = v.simd_gt(one);
            let clamped = mt_f32x8::blend(mask, reinhard, v);
            linear_to_srgb_255_simd(token, clamped)
        };

        let r = process(r);
        let g = process(g);
        let b = process(b);

        // BT.601 RGB->YCbCr
        let kr_y = mt_f32x8::splat(token, YCBCR_R_TO_Y);
        let kg_y = mt_f32x8::splat(token, YCBCR_G_TO_Y);
        let kb_y = mt_f32x8::splat(token, YCBCR_B_TO_Y);
        let kr_cb = mt_f32x8::splat(token, YCBCR_R_TO_CB);
        let kg_cb = mt_f32x8::splat(token, YCBCR_G_TO_CB);
        let kb_cb = mt_f32x8::splat(token, YCBCR_B_TO_CB);
        let kr_cr = mt_f32x8::splat(token, YCBCR_R_TO_CR);
        let kg_cr = mt_f32x8::splat(token, YCBCR_G_TO_CR);
        let kb_cr = mt_f32x8::splat(token, YCBCR_B_TO_CR);
        let offset = mt_f32x8::splat(token, CHROMA_OFFSET);

        let y = kr_y.mul_add(r, kg_y.mul_add(g, kb_y * b));
        let cb = kr_cb.mul_add(r, kg_cb.mul_add(g, kb_cb * b)) + offset;
        let cr = kr_cr.mul_add(r, kg_cr.mul_add(g, kb_cr * b)) + offset;

        (y.to_array(), cb.to_array(), cr.to_array())
    }
}

/// Convert 8 linear f32 values [0,infinity) to sRGB [0, 255].
///
/// Values > 1.0 are tone-mapped with Reinhard: x / (1 + x).
/// Uses fused SIMD on x86_64 (archmage + linear-srgb tokens), scalar elsewhere.
#[inline(always)]
pub fn linear_to_srgb_255_x8(x: &[f32; 8]) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return simd_fused::linear_to_srgb_255_x8_v3(token, x);
        }
    }
    linear_to_srgb_255_x8_scalar(x)
}

/// Scalar fallback for linear f32->sRGB x255.
#[inline(always)]
fn linear_to_srgb_255_x8_scalar(x: &[f32; 8]) -> [f32; 8] {
    let mut buf = [0.0f32; 8];
    for i in 0..8 {
        let v = x[i].max(0.0);
        buf[i] = if v > 1.0 { v / (1.0 + v) } else { v };
    }
    linear_srgb::default::linear_to_srgb_slice(&mut buf);
    for i in 0..8 {
        buf[i] *= 255.0;
    }
    buf
}

/// Convert 8 linear u16 values [0, 65535] to sRGB [0, 255].
///
/// Uses fused SIMD on x86_64 (archmage + linear-srgb tokens), scalar elsewhere.
#[inline(always)]
pub fn linear_u16_to_srgb_255_x8(values: &[u16; 8]) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return simd_fused::linear_u16_to_srgb_255_x8_v3(token, values);
        }
    }
    linear_u16_to_srgb_255_x8_scalar(values)
}

/// Scalar fallback for linear u16->sRGB x255.
#[inline(always)]
fn linear_u16_to_srgb_255_x8_scalar(values: &[u16; 8]) -> [f32; 8] {
    let mut buf = [0.0f32; 8];
    for i in 0..8 {
        buf[i] = values[i] as f32 / 65535.0;
    }
    linear_srgb::default::linear_to_srgb_slice(&mut buf);
    for i in 0..8 {
        buf[i] *= 255.0;
    }
    buf
}

/// Convert 8 linear RGB16 pixels to 8 Y, 8 Cb, 8 Cr values.
///
/// Takes R, G, B as separate arrays of 8 u16 values.
/// Returns (Y, Cb, Cr) as [f32; 8] arrays.
/// Uses a single SIMD dispatch on x86_64 for the entire u16->sRGB->YCbCr pipeline.
#[inline(always)]
pub fn linear_rgb16_to_ycbcr_x8(
    r: &[u16; 8],
    g: &[u16; 8],
    b: &[u16; 8],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return simd_fused::linear_rgb16_to_ycbcr_x8_v3(token, r, g, b);
        }
    }
    // Scalar fallback
    let r = linear_u16_to_srgb_255_x8_scalar(r);
    let g = linear_u16_to_srgb_255_x8_scalar(g);
    let b = linear_u16_to_srgb_255_x8_scalar(b);
    rgb_to_ycbcr_x8_scalar(&r, &g, &b)
}

/// Convert 8 linear RGB f32 pixels to 8 Y, 8 Cb, 8 Cr values.
///
/// Takes R, G, B as separate [f32; 8] arrays (structure-of-arrays layout).
/// Returns (Y, Cb, Cr) as [f32; 8] arrays.
/// Uses a single SIMD dispatch on x86_64 for the entire f32->sRGB->YCbCr pipeline.
#[inline(always)]
pub fn linear_rgbf32_to_ycbcr_x8(
    r: &[f32; 8],
    g: &[f32; 8],
    b: &[f32; 8],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return simd_fused::linear_rgbf32_to_ycbcr_x8_v3(token, r, g, b);
        }
    }
    // Scalar fallback
    let r = linear_to_srgb_255_x8_scalar(r);
    let g = linear_to_srgb_255_x8_scalar(g);
    let b = linear_to_srgb_255_x8_scalar(b);
    rgb_to_ycbcr_x8_scalar(&r, &g, &b)
}

/// BT.601 RGB->YCbCr matrix multiply for 8 pixels (SoA layout). Scalar fallback.
#[inline(always)]
fn rgb_to_ycbcr_x8_scalar(
    r: &[f32; 8],
    g: &[f32; 8],
    b: &[f32; 8],
) -> ([f32; 8], [f32; 8], [f32; 8]) {
    let mut y = [0.0f32; 8];
    let mut cb = [0.0f32; 8];
    let mut cr = [0.0f32; 8];
    for i in 0..8 {
        y[i] = YCBCR_R_TO_Y * r[i] + YCBCR_G_TO_Y * g[i] + YCBCR_B_TO_Y * b[i];
        cb[i] = YCBCR_R_TO_CB * r[i] + YCBCR_G_TO_CB * g[i] + YCBCR_B_TO_CB * b[i] + CHROMA_OFFSET;
        cr[i] = YCBCR_R_TO_CR * r[i] + YCBCR_G_TO_CR * g[i] + YCBCR_B_TO_CR * b[i] + CHROMA_OFFSET;
    }
    (y, cb, cr)
}

// ============================================================================
// Reference implementation (accurate, slow)
// ============================================================================

/// Reference sRGB conversion using standard formula with powf.
#[inline]
#[allow(dead_code)]
pub fn linear_to_srgb_reference(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }

    let x = if x > 1.0 { x / (1.0 + x) } else { x };

    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Reference: convert linear f32 to sRGB [0, 255].
#[inline]
#[allow(dead_code)]
pub fn linear_f32_to_srgb_255_reference(x: f32) -> f32 {
    linear_to_srgb_reference(x) * 255.0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u16_endpoints() {
        // Black
        assert!((linear_u16_to_srgb_255(0) - 0.0).abs() < 0.001);
        // White
        assert!((linear_u16_to_srgb_255(65535) - 255.0).abs() < 0.001);
        // Linear 0.5 should be ~186 in sRGB (brighter due to gamma)
        let mid = linear_u16_to_srgb_255(32768);
        assert!(mid > 180.0 && mid < 195.0, "mid gray = {}", mid);
    }

    #[test]
    fn test_u16_matches_reference() {
        for i in (0..65536).step_by(256) {
            let linear = i as f32 / 65535.0;
            let result = linear_u16_to_srgb_255(i as u16);
            let ref_val = linear_f32_to_srgb_255_reference(linear);
            let diff = (result - ref_val).abs();
            assert!(
                diff < 0.1,
                "u16 mismatch at {}: result={}, ref={}, diff={}",
                i,
                result,
                ref_val,
                diff
            );
        }
    }

    #[test]
    fn test_f32_fast_accuracy() {
        let mut max_error = 0.0f32;
        let mut max_error_at = 0.0f32;

        for i in 0..1000 {
            let linear = i as f32 / 999.0;
            let fast = linear_f32_to_srgb_255_fast(linear);
            let reference = linear_f32_to_srgb_255_reference(linear);
            let error = (fast - reference).abs();

            if error > max_error {
                max_error = error;
                max_error_at = linear;
            }
        }

        println!("Fast max error: {} at linear={}", max_error, max_error_at);
        assert!(
            max_error < 0.1,
            "Fast approximation error too high: {} at {}",
            max_error,
            max_error_at
        );
    }

    #[test]
    fn test_ycbcr_conversion_u16() {
        // Test that YCbCr conversion produces valid ranges
        let (y, cb, cr) = linear_rgb16_to_ycbcr(32768, 32768, 32768);
        // Mid gray should have Y around 128, Cb/Cr around 128
        assert!(y > 100.0 && y < 200.0, "Y = {}", y);
        assert!(cb > 120.0 && cb < 136.0, "Cb = {}", cb);
        assert!(cr > 120.0 && cr < 136.0, "Cr = {}", cr);

        // Pure red
        let (y, cb, cr) = linear_rgb16_to_ycbcr(65535, 0, 0);
        assert!(y > 50.0 && y < 100.0, "Red Y = {}", y);
        assert!(cb < 128.0, "Red Cb = {}", cb); // Cb should be below neutral
        assert!(cr > 200.0, "Red Cr = {}", cr); // Cr should be high for red
    }

    #[test]
    fn test_hdr_handling() {
        // Values > 1.0 should be tone-mapped, not clipped
        let hdr_2 = linear_f32_to_srgb_255_fast(2.0);
        let hdr_10 = linear_f32_to_srgb_255_fast(10.0);

        // Both should be < 255 (tone mapped)
        assert!(hdr_2 < 255.0 && hdr_2 > 200.0, "HDR 2.0 = {}", hdr_2);
        assert!(hdr_10 < 255.0 && hdr_10 > hdr_2, "HDR 10.0 = {}", hdr_10);

        // Should be monotonically increasing
        assert!(hdr_10 > hdr_2);
    }

    #[test]
    fn test_negative_handling() {
        assert_eq!(linear_f32_to_srgb_255_fast(-0.5), 0.0);
        assert_eq!(linear_f32_to_srgb_255_lut(-0.5), 0.0);
        assert_eq!(linear_u16_to_srgb_255(0), 0.0);
    }

    /// Verify that SIMD and scalar paths produce identical results.
    #[test]
    fn test_simd_scalar_parity_u16() {
        let values: [u16; 8] = [0, 1000, 10000, 20000, 32768, 50000, 60000, 65535];
        let simd_result = linear_u16_to_srgb_255_x8(&values);
        for (i, &v) in values.iter().enumerate() {
            let scalar_result = linear_u16_to_srgb_255(v);
            let diff = (simd_result[i] - scalar_result).abs();
            assert!(
                diff < 0.01,
                "u16 SIMD/scalar mismatch at {}: simd={}, scalar={}, diff={}",
                v,
                simd_result[i],
                scalar_result,
                diff
            );
        }
    }

    /// Verify that SIMD and scalar YCbCr paths produce identical results.
    #[test]
    fn test_simd_scalar_parity_rgb16_ycbcr() {
        let r: [u16; 8] = [65535, 0, 0, 32768, 10000, 50000, 20000, 40000];
        let g: [u16; 8] = [0, 65535, 0, 32768, 20000, 30000, 50000, 10000];
        let b: [u16; 8] = [0, 0, 65535, 32768, 40000, 10000, 30000, 60000];

        let (y_simd, cb_simd, cr_simd) = linear_rgb16_to_ycbcr_x8(&r, &g, &b);

        for i in 0..8 {
            let (y_scalar, cb_scalar, cr_scalar) = linear_rgb16_to_ycbcr(r[i], g[i], b[i]);
            let y_diff = (y_simd[i] - y_scalar).abs();
            let cb_diff = (cb_simd[i] - cb_scalar).abs();
            let cr_diff = (cr_simd[i] - cr_scalar).abs();
            assert!(
                y_diff < 0.05 && cb_diff < 0.05 && cr_diff < 0.05,
                "RGB16 YCbCr SIMD/scalar mismatch at pixel {}: Y diff={}, Cb diff={}, Cr diff={}",
                i,
                y_diff,
                cb_diff,
                cr_diff
            );
        }
    }

    /// Benchmark-style test to compare performance
    #[test]
    fn bench_conversion_methods() {
        use std::time::Instant;

        const ITERATIONS: usize = 1_000_000;

        // Generate test data
        let test_values: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let test_u16: Vec<u16> = (0..1000).map(|i| (i * 65) as u16).collect();

        // Warm up caches
        for &v in &test_values {
            let _ = linear_f32_to_srgb_255_reference(v);
            let _ = linear_f32_to_srgb_255_fast(v);
        }
        for &v in &test_u16 {
            let _ = linear_u16_to_srgb_255(v);
        }

        // Benchmark reference (powf)
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..ITERATIONS / 1000 {
            for &v in &test_values {
                sum += linear_f32_to_srgb_255_reference(v);
            }
        }
        let ref_time = start.elapsed();
        println!("Reference (powf): {:?}, sum={}", ref_time, sum);

        // Benchmark fast (linear-srgb crate)
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..ITERATIONS / 1000 {
            for &v in &test_values {
                sum += linear_f32_to_srgb_255_fast(v);
            }
        }
        let fast_time = start.elapsed();
        println!("Fast (linear-srgb): {:?}, sum={}", fast_time, sum);

        // Benchmark u16 conversion
        let start = Instant::now();
        let mut sum = 0.0f32;
        for _ in 0..ITERATIONS / 1000 {
            for &v in &test_u16 {
                sum += linear_u16_to_srgb_255(v);
            }
        }
        let u16_time = start.elapsed();
        println!("U16 conversion: {:?}, sum={}", u16_time, sum);

        // Print speedups
        let ref_ns = ref_time.as_nanos() as f64;
        println!(
            "\nSpeedups vs reference:\n  Fast: {:.1}x\n  U16: {:.1}x",
            ref_ns / fast_time.as_nanos() as f64,
            ref_ns / u16_time.as_nanos() as f64
        );
    }
}
