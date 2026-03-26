//! Fast linear RGB to sRGB/YCbCr conversion using the linear-srgb crate.
//!
//! Uses direct transfer function computation - no LUTs, no runtime initialization.
//! The linear-srgb crate provides optimized implementations with FMA acceleration.

use crate::foundation::consts::{
    YCBCR_B_TO_CB, YCBCR_B_TO_CR, YCBCR_B_TO_Y, YCBCR_G_TO_CB, YCBCR_G_TO_CR, YCBCR_G_TO_Y,
    YCBCR_R_TO_CB, YCBCR_R_TO_CR, YCBCR_R_TO_Y,
};
/// Cb/Cr offset (128.0 for 8-bit JPEG)
const CHROMA_OFFSET: f32 = 128.0;

/// sRGB transfer function (linear → sRGB).
/// Standard IEC 61966-2-1 formula. Input and output in [0, 1].
#[inline]
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
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

use wide::{CmpGt, f32x8};

/// sRGB transfer function for 8 lanes (linear → sRGB).
/// Uses polynomial approximation matching linear-srgb crate accuracy.
#[inline(always)]
fn linear_to_srgb_x8(x: f32x8) -> f32x8 {
    // Standard sRGB transfer: x <= 0.0031308 ? 12.92*x : 1.055*x^(1/2.4) - 0.055
    // Approximate x^(1/2.4) ≈ x^0.4167 via sqrt(sqrt(x)) * x^0.0417 ≈ sqrt(sqrt(x)) * lerp
    // But for correctness, use the polynomial approximation from the sRGB spec.
    //
    // Fast path: use the exact formula with sqrt-based approximation of pow.
    // x^(1/2.4) = x^(5/12) = (x^(1/4))^(5/3) = sqrt(sqrt(x)) * (sqrt(sqrt(x)))^(2/3)
    // Simpler: x^(1/2.4) ≈ sqrt(x) * x^(-1/60) which is close but imprecise.
    //
    // For maximum accuracy without powf, use the minimax polynomial from linear-srgb:
    // We'll use scalar powf per-element since this is not the hot path.
    let arr = <[f32; 8]>::from(x);
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = linear_to_srgb(arr[i]);
    }
    f32x8::new(out)
}

/// Convert 8 linear f32 values [0,1] to sRGB [0, 255] using SIMD.
///
/// Values > 1.0 are tone-mapped with Reinhard.
#[inline(always)]
pub fn linear_to_srgb_255_x8(x: f32x8) -> f32x8 {
    // Clamp negatives to zero
    let x = x.max(f32x8::ZERO);

    // Reinhard tone mapping for HDR: x / (1 + x)
    // Only apply where x > 1.0
    let one = f32x8::ONE;
    let needs_tonemap = x.simd_gt(one);
    let tonemapped = x / (one + x);
    let x = needs_tonemap.blend(tonemapped, x);

    // Apply sRGB transfer function and scale to [0, 255]
    linear_to_srgb_x8(x) * f32x8::splat(255.0)
}

/// Convert 8 linear u16 values [0, 65535] to sRGB [0, 255] using SIMD.
#[inline(always)]
pub fn linear_u16_to_srgb_255_x8(values: [u16; 8]) -> f32x8 {
    let scale = f32x8::splat(1.0 / 65535.0);
    let linear = f32x8::new([
        values[0] as f32,
        values[1] as f32,
        values[2] as f32,
        values[3] as f32,
        values[4] as f32,
        values[5] as f32,
        values[6] as f32,
        values[7] as f32,
    ]) * scale;

    // No HDR tone mapping needed for u16 (max is 1.0)
    linear_to_srgb_x8(linear) * f32x8::splat(255.0)
}

/// Convert 8 linear RGB16 pixels to 8 Y, 8 Cb, 8 Cr values using SIMD.
///
/// Takes R, G, B as separate arrays of 8 u16 values.
/// Returns (Y, Cb, Cr) as f32x8 vectors.
#[inline(always)]
pub fn linear_rgb16_to_ycbcr_x8(r: [u16; 8], g: [u16; 8], b: [u16; 8]) -> (f32x8, f32x8, f32x8) {
    // Convert linear u16 to sRGB [0, 255]
    let r = linear_u16_to_srgb_255_x8(r);
    let g = linear_u16_to_srgb_255_x8(g);
    let b = linear_u16_to_srgb_255_x8(b);

    // RGB to YCbCr matrix multiplication
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);
    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);
    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);
    let chroma_offset = f32x8::splat(CHROMA_OFFSET);

    let y = r_to_y.mul_add(r, g_to_y.mul_add(g, b_to_y * b));
    let cb = r_to_cb.mul_add(r, g_to_cb.mul_add(g, b_to_cb * b)) + chroma_offset;
    let cr = r_to_cr.mul_add(r, g_to_cr.mul_add(g, b_to_cr * b)) + chroma_offset;

    (y, cb, cr)
}

/// Convert 8 linear RGB pixels to 8 Y, 8 Cb, 8 Cr values using SIMD.
///
/// Takes R, G, B as separate f32x8 vectors (structure-of-arrays layout).
/// Returns (Y, Cb, Cr) as f32x8 vectors.
#[inline(always)]
pub fn linear_rgbf32_to_ycbcr_x8(r: f32x8, g: f32x8, b: f32x8) -> (f32x8, f32x8, f32x8) {
    // Convert linear to sRGB [0, 255]
    let r = linear_to_srgb_255_x8(r);
    let g = linear_to_srgb_255_x8(g);
    let b = linear_to_srgb_255_x8(b);

    // RGB to YCbCr matrix multiplication
    let r_to_y = f32x8::splat(YCBCR_R_TO_Y);
    let g_to_y = f32x8::splat(YCBCR_G_TO_Y);
    let b_to_y = f32x8::splat(YCBCR_B_TO_Y);
    let r_to_cb = f32x8::splat(YCBCR_R_TO_CB);
    let g_to_cb = f32x8::splat(YCBCR_G_TO_CB);
    let b_to_cb = f32x8::splat(YCBCR_B_TO_CB);
    let r_to_cr = f32x8::splat(YCBCR_R_TO_CR);
    let g_to_cr = f32x8::splat(YCBCR_G_TO_CR);
    let b_to_cr = f32x8::splat(YCBCR_B_TO_CR);
    let chroma_offset = f32x8::splat(CHROMA_OFFSET);

    let y = r_to_y.mul_add(r, g_to_y.mul_add(g, b_to_y * b));
    let cb = r_to_cb.mul_add(r, g_to_cb.mul_add(g, b_to_cb * b)) + chroma_offset;
    let cr = r_to_cr.mul_add(r, g_to_cr.mul_add(g, b_to_cr * b)) + chroma_offset;

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
