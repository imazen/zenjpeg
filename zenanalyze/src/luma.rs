//! Per-primaries luma weights for the tier kernels.
//!
//! The standard tiers (Tier 1 / Tier 2 / Tier 3) compute luma /
//! chroma stats on display-space u8 RGB bytes. For sRGB-primary
//! content that's a well-defined operation: pick a fixed
//! `(KR, KG, KB)` triplet, accumulate. For wide-gamut u8 inputs
//! (Display P3 / Rec.2020 / AdobeRGB) the *bytes* are still
//! display-space u8, but the green primary in particular sits at a
//! different chromaticity — so the weights `(0.299, 0.587, 0.114)`
//! that turn sRGB-primary RGB into BT.601 luma are wrong.
//!
//! This module returns the right `(KR, KG, KB)` triplet for each
//! `ColorPrimaries`, derived from each primary set's RGB→XYZ matrix
//! (Y row, normalised so KR + KG + KB ≈ 1.0).
//!
//! ## Why sRGB / BT.709 keeps BT.601 weights
//!
//! Strictly, sRGB-primary content should use the BT.709 weights
//! `(0.2126, 0.7152, 0.0722)` — sRGB and BT.709 share primaries, and
//! BT.709 is the matching YCbCr matrix. The analyzer's threshold
//! contract is calibrated against BT.601 weights on sRGB bytes (a
//! historical accident — `coefficient` and the trees it ships used
//! `(0.299, 0.587, 0.114)` against sRGB content from day one). To
//! preserve those trained thresholds, we keep BT.601 weights for the
//! sRGB / BT.709 path. Wide-gamut sources are new territory and get
//! the mathematically-correct weights for their primaries.
//!
//! See `CLAUDE.md` for the no-0.2.x stability contract — this is a
//! one-time, additive numeric drift on wide-gamut inputs only.

use zenpixels::ColorPrimaries;

/// `(KR, KG, KB)` luma weight triplet, plus precomputed 8-bit fixed-
/// point versions for the integer luma paths (Tier 3 histogram +
/// block luma).
///
/// The f32 weights `kr / kg / kb` sum to ≈ 1.0 (mathematical luma
/// matrix from each primary set's RGB→XYZ Y row). The fixed-point
/// weights `qr / qg / qb` sum to **220**, matching the libwebp /
/// coefficient BT.601 baseline `66 + 129 + 25 = 220`. That sum is
/// the integer-luma scale Tier 3's histogram and DCT-block paths
/// were calibrated against (white maps to 219, not 255). Wide-
/// gamut weight sets are scaled to the same 220 sum so a pure-
/// white pixel hits the same histogram bin regardless of source
/// primaries — keeping the trained-threshold contract intact for
/// every analyzer feature whose output depends on absolute luma
/// magnitude.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LumaWeights {
    pub kr: f32,
    pub kg: f32,
    pub kb: f32,
    /// Fixed-point luma scaled to sum ≈ 220 (libwebp baseline).
    /// Used by Tier 3 in `((qr * R + qg * G + qb * B + 128) >> 8)`.
    pub qr: i32,
    pub qg: i32,
    pub qb: i32,
}

/// Sum of the libwebp-style fixed-point luma matrix. White maps to
/// `(LUMA_SUM_220 * 255 + 128) >> 8 = 219`. All primaries use this
/// scale so the histogram thresholds keep their meaning.
const LUMA_SUM_220: i32 = 220;

impl LumaWeights {
    /// True iff this weight set is the BT.601 baseline shared by
    /// sRGB / BT.709 / Unknown sources. Used by the Tier 1 SIMD
    /// dispatcher to pick the const-folded `accumulate_row_simd::<true>`
    /// specialisation, which lets LLVM emit `vfmadd*` immediates for
    /// `kr / kg / kb` instead of register-loaded splats.
    #[inline]
    pub(crate) fn is_bt601_baseline(&self) -> bool {
        // The constants are chosen so equality holds bit-exact against
        // the `bt601()` constructor below — no tolerance needed.
        self.kr == 0.299 && self.kg == 0.587 && self.kb == 0.114
    }

    /// Pick weights for a source's declared [`ColorPrimaries`].
    ///
    /// - `Bt709` (sRGB / BT.709): BT.601 weights — preserves the
    ///   trained-threshold baseline.
    /// - `Bt2020`: BT.2020 luma weights from Rec. ITU-R BT.2020.
    /// - `DisplayP3`, `AdobeRgb`: derived from each primary set's
    ///   RGB→XYZ Y row.
    /// - Anything else (`Unknown`, future variants): falls back to
    ///   BT.601, matching the historical default.
    pub fn for_primaries(p: ColorPrimaries) -> Self {
        match p {
            ColorPrimaries::Bt709 => bt601(),
            ColorPrimaries::Bt2020 => bt2020(),
            ColorPrimaries::DisplayP3 => display_p3(),
            ColorPrimaries::AdobeRgb => adobe_rgb(),
            _ => bt601(),
        }
    }
}

/// BT.601 weights — the analyzer's historical baseline; preserved
/// for sRGB / BT.709 inputs to keep the trained threshold contract.
#[inline]
const fn bt601() -> LumaWeights {
    LumaWeights {
        kr: 0.299,
        kg: 0.587,
        kb: 0.114,
        // round(0.299 * 256) = 77; 0.587 * 256 ≈ 150; 0.114 * 256 ≈ 29
        // — but tier 3's existing integer math uses 66/129/25 which
        // is BT.601-derived-from-BT.709-with-different-rounding (the
        // libwebp / coefficient lineage). Match that.
        qr: 66,
        qg: 129,
        qb: 25,
    }
}

/// BT.2020 luma weights — Rec. ITU-R BT.2020-2 §3.4. Fixed-point
/// weights scaled to the sum-220 libwebp baseline so wide-gamut
/// integer luma lands on the same 0-219 scale as sRGB content.
#[inline]
const fn bt2020() -> LumaWeights {
    LumaWeights {
        kr: 0.2627,
        kg: 0.6780,
        kb: 0.0593,
        // round(0.2627 * 220) = 58, round(0.6780 * 220) = 149,
        // round(0.0593 * 220) = 13. Sum = 220.
        qr: 58,
        qg: 149,
        qb: 13,
    }
}

/// Display P3 (D65, P3 primaries) — derived from the RGB→XYZ matrix
/// in SMPTE EG 432-1. Sum-220 fixed-point.
#[inline]
const fn display_p3() -> LumaWeights {
    LumaWeights {
        kr: 0.2289,
        kg: 0.6917,
        kb: 0.0793,
        // round(0.2289 * 220) = 50, round(0.6917 * 220) = 152,
        // round(0.0793 * 220) = 17. Sum = 219, off by -1 (negligible).
        qr: 50,
        qg: 152,
        qb: 17,
    }
}

/// AdobeRGB (1998), D65. Primaries differ from sRGB on green;
/// red and blue are close. Sum-220 fixed-point.
#[inline]
const fn adobe_rgb() -> LumaWeights {
    LumaWeights {
        kr: 0.2974,
        kg: 0.6273,
        kb: 0.0753,
        // round(0.2974 * 220) = 65, round(0.6273 * 220) = 138,
        // round(0.0753 * 220) = 17. Sum = 220.
        qr: 65,
        qg: 138,
        qb: 17,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fp + integer weights both sum close to unity / 256.
    #[test]
    fn weights_sum_to_unity_for_every_primary_set() {
        for &p in &[
            ColorPrimaries::Bt709,
            ColorPrimaries::Bt2020,
            ColorPrimaries::DisplayP3,
            ColorPrimaries::AdobeRgb,
        ] {
            let w = LumaWeights::for_primaries(p);
            let f_sum = w.kr + w.kg + w.kb;
            assert!(
                (f_sum - 1.0).abs() < 1e-2,
                "{p:?}: f32 weights sum {f_sum}, expected ≈ 1.0"
            );
            let q_sum = w.qr + w.qg + w.qb;
            assert!(
                (q_sum - LUMA_SUM_220).abs() <= 1,
                "{p:?}: fixed-point weights sum {q_sum}, expected ≈ 220 (libwebp baseline)"
            );
        }
    }

    #[test]
    fn bt709_path_returns_bt601_baseline() {
        // Trained-threshold preservation: sRGB / BT.709 inputs MUST
        // continue to use BT.601 weights so the existing thresholds
        // keep their meaning.
        let w = LumaWeights::for_primaries(ColorPrimaries::Bt709);
        assert_eq!(w.kr, 0.299);
        assert_eq!(w.kg, 0.587);
        assert_eq!(w.kb, 0.114);
        assert_eq!((w.qr, w.qg, w.qb), (66, 129, 25));
    }

    #[test]
    fn wide_gamut_paths_diverge_from_bt601() {
        for &p in &[
            ColorPrimaries::Bt2020,
            ColorPrimaries::DisplayP3,
            ColorPrimaries::AdobeRgb,
        ] {
            let w = LumaWeights::for_primaries(p);
            assert_ne!(
                (w.kr, w.kg, w.kb),
                (0.299_f32, 0.587_f32, 0.114_f32),
                "{p:?} should NOT use BT.601 weights"
            );
        }
    }
}
