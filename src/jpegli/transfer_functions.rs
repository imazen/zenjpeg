//! Transfer functions for HDR color encodings.
//!
//! Implements PQ (Perceptual Quantizer) and HLG (Hybrid Log-Gamma) transfer
//! functions as defined in BT.2100-2.
//!
//! Terminology:
//! - "display" is linear light (nits) normalized to [0, 1]
//! - "encoded" is a nonlinear encoding (e.g. PQ/HLG) in [0, 1]
//! - "scene" is a linear function of photon counts, normalized to [0, 1]
//!
//! The functions support unbounded inputs (negative or >1) by mirroring
//! negative values via copysign: f(-x) = -f(x). This is needed for chromatic
//! adaptation in color management.

/// Default display intensity target in nits.
pub const DEFAULT_INTENSITY_TARGET: f32 = 255.0;

// ============================================================================
// HLG (Hybrid Log-Gamma) Constants
// ============================================================================

/// HLG constant a (from BT.2100-2)
const HLG_A: f64 = 0.17883277;
/// 1/a for inverse calculations
const HLG_RA: f64 = 1.0 / HLG_A;
/// HLG constant b = 1 - 4a
const HLG_B: f64 = 1.0 - 4.0 * HLG_A;
/// HLG constant c
const HLG_C: f64 = 0.5599107295;
/// 1/12 threshold for HLG OETF
const HLG_INV12: f64 = 1.0 / 12.0;

// ============================================================================
// PQ (Perceptual Quantizer) Constants
// ============================================================================

/// PQ exponent m1 = 2610/16384
const PQ_M1: f64 = 2610.0 / 16384.0;
/// PQ exponent m2 = (2523/4096) * 128
const PQ_M2: f64 = (2523.0 / 4096.0) * 128.0;
/// PQ constant c1 = 3424/4096
const PQ_C1: f64 = 3424.0 / 4096.0;
/// PQ constant c2 = (2413/4096) * 32
const PQ_C2: f64 = (2413.0 / 4096.0) * 32.0;
/// PQ constant c3 = (2392/4096) * 32
const PQ_C3: f64 = (2392.0 / 4096.0) * 32.0;

// ============================================================================
// HLG Transfer Functions
// ============================================================================

/// HLG OETF (Opto-Electronic Transfer Function).
///
/// Converts scene light (linear) to encoded HLG signal.
/// s = scene [0,1] -> encoded [0,1]
#[inline]
pub fn hlg_oetf(s: f64) -> f64 {
    if s == 0.0 {
        return 0.0;
    }
    let original_sign = s.signum();
    let s_abs = s.abs();

    let encoded = if s_abs <= HLG_INV12 {
        (3.0 * s_abs).sqrt()
    } else {
        HLG_A * (12.0 * s_abs - HLG_B).ln() + HLG_C
    };

    original_sign * encoded
}

/// HLG Inverse OETF.
///
/// Converts encoded HLG signal to scene light (linear).
/// e = encoded [0,1] -> scene [0,1]
#[inline]
pub fn hlg_inv_oetf(e: f64) -> f64 {
    if e == 0.0 {
        return 0.0;
    }
    let original_sign = e.signum();
    let e_abs = e.abs();

    let scene = if e_abs <= 0.5 {
        e_abs * e_abs / 3.0
    } else {
        (((e_abs - HLG_C) * HLG_RA).exp() + HLG_B) * HLG_INV12
    };

    original_sign * scene
}

/// HLG OOTF (Opto-Optical Transfer Function).
///
/// For system gamma of 1.0 at 334 nits, this is the identity function.
/// s = scene -> display
#[inline]
pub fn hlg_ootf(s: f64) -> f64 {
    // At 334 nits reference, gamma = 1.0, so OOTF is identity
    s
}

/// HLG Inverse OOTF.
///
/// For system gamma of 1.0 at 334 nits, this is the identity function.
/// d = display -> scene
#[inline]
pub fn hlg_inv_ootf(d: f64) -> f64 {
    // At 334 nits reference, gamma = 1.0, so inverse OOTF is identity
    d
}

/// HLG EOTF (Electro-Optical Transfer Function).
///
/// Converts encoded HLG signal to display light.
/// e = encoded [0,1] -> display [0,1]
#[inline]
pub fn hlg_display_from_encoded(e: f64) -> f64 {
    hlg_ootf(hlg_inv_oetf(e))
}

/// HLG Inverse EOTF.
///
/// Converts display light to encoded HLG signal.
/// d = display [0,1] -> encoded [0,1]
#[inline]
pub fn hlg_encoded_from_display(d: f64) -> f64 {
    hlg_oetf(hlg_inv_ootf(d))
}

/// HLG encoded from display (f32 version).
#[inline]
pub fn hlg_encoded_from_display_f32(d: f32) -> f32 {
    hlg_encoded_from_display(d as f64) as f32
}

/// HLG display from encoded (f32 version).
#[inline]
pub fn hlg_display_from_encoded_f32(e: f32) -> f32 {
    hlg_display_from_encoded(e as f64) as f32
}

// ============================================================================
// PQ (Perceptual Quantizer) Transfer Functions
// ============================================================================

/// PQ struct with configurable display intensity target.
#[derive(Debug, Clone, Copy)]
pub struct PQ {
    /// Scaling factor for converting to 10000 nits reference
    to_10000_nits: f64,
    /// Scaling factor for converting from 10000 nits reference
    from_10000_nits: f64,
}

impl Default for PQ {
    fn default() -> Self {
        Self::new(DEFAULT_INTENSITY_TARGET)
    }
}

impl PQ {
    /// Create a PQ transfer function with the given display intensity target.
    ///
    /// The intensity target is in nits (cd/m²). The PQ curve is defined
    /// relative to 10000 nits, so we scale appropriately.
    #[inline]
    pub fn new(display_intensity_target: f32) -> Self {
        Self {
            to_10000_nits: display_intensity_target as f64 / 10000.0,
            from_10000_nits: 10000.0 / display_intensity_target as f64,
        }
    }

    /// PQ EOTF (Electro-Optical Transfer Function).
    ///
    /// Converts encoded PQ signal to display light.
    /// e = encoded [0,1] -> display [0,1]
    #[inline]
    pub fn display_from_encoded(&self, e: f64) -> f64 {
        if e == 0.0 {
            return 0.0;
        }
        let original_sign = e.signum();
        let e_abs = e.abs();

        let xp = e_abs.powf(1.0 / PQ_M2);
        let num = (xp - PQ_C1).max(0.0);
        let den = PQ_C2 - PQ_C3 * xp;

        let d = if den != 0.0 {
            (num / den).powf(1.0 / PQ_M1)
        } else {
            0.0
        };

        original_sign * d * self.from_10000_nits
    }

    /// PQ Inverse EOTF.
    ///
    /// Converts display light to encoded PQ signal.
    /// d = display [0,1] -> encoded [0,1]
    #[inline]
    pub fn encoded_from_display(&self, d: f64) -> f64 {
        if d == 0.0 {
            return 0.0;
        }
        let original_sign = d.signum();
        let d_abs = d.abs();

        let xp = (d_abs * self.to_10000_nits).powf(PQ_M1);
        let num = PQ_C1 + xp * PQ_C2;
        let den = 1.0 + xp * PQ_C3;

        let e = (num / den).powf(PQ_M2);

        original_sign * e
    }

    /// PQ encoded from display (f32 version).
    #[inline]
    pub fn encoded_from_display_f32(&self, d: f32) -> f32 {
        self.encoded_from_display(d as f64) as f32
    }

    /// PQ display from encoded (f32 version).
    #[inline]
    pub fn display_from_encoded_f32(&self, e: f32) -> f32 {
        self.display_from_encoded(e as f64) as f32
    }
}

/// PQ EOTF with default intensity target.
#[inline]
pub fn pq_display_from_encoded(e: f64) -> f64 {
    PQ::default().display_from_encoded(e)
}

/// PQ Inverse EOTF with default intensity target.
#[inline]
pub fn pq_encoded_from_display(d: f64) -> f64 {
    PQ::default().encoded_from_display(d)
}

/// PQ EOTF with custom intensity target.
#[inline]
pub fn pq_display_from_encoded_with_target(intensity_target: f32, e: f64) -> f64 {
    PQ::new(intensity_target).display_from_encoded(e)
}

/// PQ Inverse EOTF with custom intensity target.
#[inline]
pub fn pq_encoded_from_display_with_target(intensity_target: f32, d: f64) -> f64 {
    PQ::new(intensity_target).encoded_from_display(d)
}

// ============================================================================
// sRGB Transfer Functions
// ============================================================================

/// sRGB threshold for linear-to-encoded transition
const SRGB_THRESH_LINEAR_TO_ENCODED: f64 = 0.0031308;
/// sRGB threshold for encoded-to-linear transition
const SRGB_THRESH_ENCODED_TO_LINEAR: f64 = 0.04045;
/// sRGB linear segment multiplier
const SRGB_LINEAR_MUL: f64 = 12.92;
/// 1/12.92 for inverse
const SRGB_LINEAR_MUL_INV: f64 = 1.0 / 12.92;

/// sRGB EOTF (encoded to linear).
///
/// Converts sRGB encoded value to linear light.
#[inline]
pub fn srgb_display_from_encoded(e: f64) -> f64 {
    if e == 0.0 {
        return 0.0;
    }
    let original_sign = e.signum();
    let e_abs = e.abs();

    let linear = if e_abs <= SRGB_THRESH_ENCODED_TO_LINEAR {
        e_abs * SRGB_LINEAR_MUL_INV
    } else {
        ((e_abs + 0.055) / 1.055).powf(2.4)
    };

    original_sign * linear
}

/// sRGB Inverse EOTF (linear to encoded).
///
/// Converts linear light to sRGB encoded value.
#[inline]
pub fn srgb_encoded_from_display(d: f64) -> f64 {
    if d == 0.0 {
        return 0.0;
    }
    let original_sign = d.signum();
    let d_abs = d.abs();

    let encoded = if d_abs <= SRGB_THRESH_LINEAR_TO_ENCODED {
        d_abs * SRGB_LINEAR_MUL
    } else {
        1.055 * d_abs.powf(1.0 / 2.4) - 0.055
    };

    original_sign * encoded
}

/// sRGB encoded from display (f32 version).
#[inline]
pub fn srgb_encoded_from_display_f32(d: f32) -> f32 {
    srgb_encoded_from_display(d as f64) as f32
}

/// sRGB display from encoded (f32 version).
#[inline]
pub fn srgb_display_from_encoded_f32(e: f32) -> f32 {
    srgb_display_from_encoded(e as f64) as f32
}

// ============================================================================
// BT.709 Transfer Functions
// ============================================================================

/// BT.709 threshold
const BT709_THRESH: f64 = 0.018;
/// BT.709 low multiplier
const BT709_MUL_LOW: f64 = 4.5;
/// BT.709 high multiplier
const BT709_MUL_HI: f64 = 1.099;
/// BT.709 high exponent
const BT709_POW_HI: f64 = 0.45;
/// BT.709 subtraction constant
const BT709_SUB: f64 = -0.099;

/// BT.709 inverse threshold
const BT709_INV_THRESH: f64 = 0.081;
/// BT.709 inverse low multiplier
const BT709_INV_MUL_LOW: f64 = 1.0 / 4.5;
/// BT.709 inverse high multiplier
const BT709_INV_MUL_HI: f64 = 1.0 / 1.099;
/// BT.709 inverse high exponent
const BT709_INV_POW_HI: f64 = 1.0 / 0.45;
/// BT.709 inverse add constant
const BT709_INV_ADD: f64 = 0.099 * BT709_INV_MUL_HI;

/// BT.709 OETF (linear to encoded).
#[inline]
pub fn bt709_encoded_from_display(d: f64) -> f64 {
    if d < BT709_THRESH {
        BT709_MUL_LOW * d
    } else {
        BT709_MUL_HI * d.powf(BT709_POW_HI) + BT709_SUB
    }
}

/// BT.709 Inverse OETF (encoded to linear).
#[inline]
pub fn bt709_display_from_encoded(e: f64) -> f64 {
    if e < BT709_INV_THRESH {
        e * BT709_INV_MUL_LOW
    } else {
        (e * BT709_INV_MUL_HI + BT709_INV_ADD).powf(BT709_INV_POW_HI)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlg_roundtrip() {
        for i in 0..=100 {
            let d = i as f64 / 100.0;
            let encoded = hlg_encoded_from_display(d);
            let decoded = hlg_display_from_encoded(encoded);
            let error = (d - decoded).abs();
            assert!(
                error < 1e-10,
                "HLG roundtrip error at d={}: error={}",
                d,
                error
            );
        }
    }

    #[test]
    fn test_hlg_negative() {
        // Test unbounded (negative) inputs
        let d = -0.5;
        let encoded = hlg_encoded_from_display(d);
        let decoded = hlg_display_from_encoded(encoded);
        let error = (d - decoded).abs();
        assert!(error < 1e-10, "HLG negative roundtrip error: {}", error);
        assert!(encoded < 0.0, "HLG should preserve sign");
    }

    #[test]
    fn test_pq_roundtrip() {
        let pq = PQ::new(11000.0);
        for i in 0..=100 {
            let d = i as f64 / 100.0;
            let encoded = pq.encoded_from_display(d);
            let decoded = pq.display_from_encoded(encoded);
            let error = (d - decoded).abs();
            assert!(
                error < 1e-10,
                "PQ roundtrip error at d={}: error={}",
                d,
                error
            );
        }
    }

    #[test]
    fn test_pq_negative() {
        let pq = PQ::default();
        let d = -0.5;
        let encoded = pq.encoded_from_display(d);
        let decoded = pq.display_from_encoded(encoded);
        let error = (d - decoded).abs();
        assert!(error < 1e-10, "PQ negative roundtrip error: {}", error);
        assert!(encoded < 0.0, "PQ should preserve sign");
    }

    #[test]
    fn test_srgb_roundtrip() {
        for i in 0..=100 {
            let d = i as f64 / 100.0;
            let encoded = srgb_encoded_from_display(d);
            let decoded = srgb_display_from_encoded(encoded);
            let error = (d - decoded).abs();
            assert!(
                error < 1e-10,
                "sRGB roundtrip error at d={}: error={}",
                d,
                error
            );
        }
    }

    #[test]
    fn test_bt709_roundtrip() {
        for i in 0..=100 {
            let d = i as f64 / 100.0;
            let encoded = bt709_encoded_from_display(d);
            let decoded = bt709_display_from_encoded(encoded);
            let error = (d - decoded).abs();
            assert!(
                error < 1e-10,
                "BT.709 roundtrip error at d={}: error={}",
                d,
                error
            );
        }
    }

    #[test]
    fn test_pq_intensity_target() {
        // Verify that intensity target affects the output
        let pq_default = PQ::new(255.0);
        let pq_high = PQ::new(10000.0);

        let d = 0.5;
        let encoded_default = pq_default.encoded_from_display(d);
        let encoded_high = pq_high.encoded_from_display(d);

        // Higher target should produce higher encoded values
        // (same display nits normalized to higher max)
        assert!(
            encoded_default < encoded_high,
            "Higher target should produce higher encoded: {} < {}",
            encoded_default,
            encoded_high
        );
    }
}
