//! Gamma transfer functions for chroma downsampling.
//!
//! No traits — just `#[inline(always)]` free functions and precomputed LUTs.
//! Works inside `#[arcane]` regions because inlined functions inherit the
//! caller's target features.

/// Precomputed gamma lookup tables for linearize (u8→f32) and delinearize
/// (f32→u8). Passed by reference into chroma kernels — no trait dispatch.
pub struct GammaLuts {
    /// sRGB u8 → linear f32. 256 entries.
    pub to_linear: [f32; 256],
}

impl GammaLuts {
    /// Build sRGB gamma LUTs using `linear-srgb` crate's exact tables.
    pub fn srgb() -> Self {
        let mut to_linear = [0.0f32; 256];
        for i in 0..256 {
            to_linear[i] = linear_srgb::default::srgb_u8_to_linear(i as u8);
        }
        Self { to_linear }
    }

    /// Build libwebp-style gamma^0.80 LUTs (for VP8/WebP compatibility).
    ///
    /// libwebp uses `v^(1/0.45)` for linearize and `v^0.45` for delinearize,
    /// which is approximately gamma 2.22 encode / 0.45 decode. The effective
    /// averaging exponent is 0.80 (compromise between gamma and perceptual).
    pub fn libwebp() -> Self {
        let mut to_linear = [0.0f32; 256];
        for i in 0..256 {
            let v = i as f32 / 255.0;
            // libwebp uses exponent 1/0.45 ≈ 2.222 for linearization
            to_linear[i] = powf_nostd(v, 1.0 / 0.45);
        }
        Self { to_linear }
    }
}

/// Linearize a u8 value using the provided LUT.
#[inline(always)]
pub fn linearize(luts: &GammaLuts, v: u8) -> f32 {
    luts.to_linear[v as usize]
}

/// Delinearize a linear f32 [0,1] value to sRGB u8 [0,255].
/// Uses `linear-srgb`'s optimized scalar path (rational polynomial).
#[inline(always)]
pub fn delinearize_srgb(v: f32) -> f32 {
    linear_srgb::default::linear_to_srgb(v)
}

/// Delinearize with libwebp's gamma^0.45 curve.
#[inline(always)]
pub fn delinearize_libwebp(v: f32) -> f32 {
    powf_nostd(v, 0.45)
}

/// `powf` for no_std via libm.
#[inline(always)]
fn powf_nostd(base: f32, exp: f32) -> f32 {
    libm::powf(base, exp)
}
