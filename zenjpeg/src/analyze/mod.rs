//! Image content analysis for `EncoderConfig::auto_for`.
//!
//! Computes the numeric features the oracle decision trees split on
//! (`scripts/fit_oracle_tree.py` in `coefficient`). The set is the
//! "statistically relevant" subset — every feature appears in at
//! least one fitted tree's splits per the 2026-04-25 audit (70 trees,
//! 5 buckets × 7 q-bins × 2 metrics).
//!
//! # Entry points
//!
//! - [`analyze`] — preferred. Takes any [`PixelSlice`]; pulls RGB8
//!   rows on demand via [`row_stream::RowStream`]. Native (zero-copy)
//!   on RGB8/RGB8_SRGB inputs; one-row scratch + `RowConverter` on
//!   anything else.
//! - [`analyze_rgb8`] — convenience for callers who already hold a
//!   packed RGB8 `&[u8]` (e.g., the parity harness, internal
//!   tests). Builds a `PixelSlice<Rgb<u8>>` and forwards.
//!
//! # Layering
//!
//! - [`tier1`]: variance, edges, chroma stats, uniformity, palette.
//!   Sparse stripe sampling (8-row blocks, ~500k pixel budget).
//!   `#[archmage::autoversion]` autovec on the inner row scan.
//! - [`tier2_chroma`]: per-channel per-axis chroma sharpness, forked
//!   from evalchroma 1.0.3 `image_sharpness`. Three-row sliding
//!   window over the full image with fragment-based normalization.
//! - [`tier3`]: 32-bin luma histogram entropy, naive 8×8 DCT
//!   high-freq energy ratio (caps at 256 sampled blocks), derived
//!   text/screen-content/natural likelihoods.
//!
//! # Encapsulation
//!
//! Public surface is just [`analyze`] / [`AnalyzerOutput`] (and the
//! convenience [`analyze_rgb8`]). The taxonomy that the trees were
//! trained against is an implementation detail — there is no public
//! `ContentBucket` enum, no public per-feature struct on the
//! `EncoderConfig::auto_for` API. See
//! `src/encode/auto_for_design.md` for the rationale.

#![allow(dead_code)] // Some pub helpers ride along for the parity harness.

pub mod row_stream;
pub mod tier1;
pub mod tier2_chroma;
pub mod tier3;

use zenpixels::{PixelDescriptor, PixelSlice};

use row_stream::RowStream;

/// Numeric image features consumed by the auto_for decision trees.
///
/// Field names match the keys in `oracle-d2/source_features.json`
/// produced by `coefficient::examples::oracle_extract_features`, so
/// the codegen pass (`scripts/gen_auto_for.py`) can emit
/// `if features.<name> <= T` without an extra mapping layer.
///
/// All values are normalized into stable ranges (see per-field docs).
/// Default = 0 for every field, which matches what the fitter's
/// sidecar loader does for missing keys.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnalyzerOutput {
    // ---------------- Tier 1: sparse stripe scan -----------------
    /// Luma variance on the BT.601 [0, 255] scale. ~1000 ⇒ complex.
    pub variance: f32,
    /// Fraction of sampled interior pixels with luma gradient² > 400 (= |∇L| > 20).
    pub edge_density: f32,
    /// √(Var(Cb) + Var(Cr)) where Cb = (B-L)/255, Cr = (R-L)/255.
    pub chroma_complexity: f32,
    /// Mean |∇Cb| over horizontally-paired sampled pixels (Cb ∈ [0, 1]).
    pub cb_sharpness: f32,
    /// Mean |∇Cr| over horizontally-paired sampled pixels.
    pub cr_sharpness: f32,
    /// Fraction of 8×8 blocks with luma variance < 25.
    pub uniformity: f32,
    /// Fraction of 8×8 blocks with R, G, B ranges all ≤ 4 (solid color).
    pub flat_color_block_ratio: f32,
    /// Distinct 5-bit-per-channel color bins (RGB → 32k bin space, popcount).
    pub distinct_color_bins: u32,

    // ---------------- Tier 2: per-channel per-axis chroma --------
    /// Cb horizontal gradient energy / 1e5 (evalchroma scale).
    pub cb_horiz_sharpness: f32,
    /// Cb vertical gradient energy / 1e5.
    pub cb_vert_sharpness: f32,
    /// Cb peak gradient magnitude on [0, 100].
    pub cb_peak_sharpness: f32,
    /// Cr horizontal gradient energy / 1e5.
    pub cr_horiz_sharpness: f32,
    /// Cr vertical gradient energy / 1e5.
    pub cr_vert_sharpness: f32,
    /// Cr peak gradient magnitude on [0, 100].
    pub cr_peak_sharpness: f32,

    // ---------------- Tier 3: DCT energy + entropy + likelihoods -
    /// Σ AC[k≥16] / Σ AC[k∈1..16] over sampled 8×8 luma blocks.
    pub high_freq_energy_ratio: f32,
    /// Shannon entropy (bits) of a 32-bin luma histogram. [0, 5].
    pub luma_histogram_entropy: f32,
    /// Soft score [0, 1]: rendered text / document content.
    pub text_likelihood: f32,
    /// Soft score [0, 1]: UI / chart / synthetic content.
    pub screen_content_likelihood: f32,
    /// Soft score [0, 1]: natural photographic content.
    pub natural_likelihood: f32,

    // ---------------- Geometry (free from inputs) ----------------
    /// width × height / 1e6.
    pub megapixels: f32,
    /// width / max(1, height).
    pub aspect_ratio: f32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}

/// Run Tier 1 + Tier 2 + Tier 3 over any [`PixelSlice`]. Pulls rows
/// on demand — never materializes a full RGB8 buffer.
///
/// Native zero-copy when the slice descriptor is layout-compatible
/// with `RGB8`; transparent row-by-row conversion to `RGB8_SRGB`
/// otherwise. For non-sRGB transfer functions the analyzer treats
/// the converted bytes as display-space sRGB (matching the trees'
/// training data — the bytes that come out of `image::open(...).
/// to_rgb8()`).
///
/// # Errors
///
/// Returns an error string if the source descriptor isn't
/// convertible (e.g. CMYK without a CMS plugin loaded into
/// `RowConverter`).
pub fn analyze(slice: PixelSlice<'_>) -> Result<AnalyzerOutput, String> {
    let width = slice.width();
    let height = slice.rows();
    let mut stream = RowStream::new(slice)?;

    let mut out = AnalyzerOutput::default();
    if width >= 2 && height >= 2 {
        tier1::extract_tier1_into(&mut out, &mut stream);
    }
    if width >= 3 && height >= 3 {
        tier2_chroma::populate_tier2(&mut out, &mut stream);
    }
    if width >= 8 && height >= 8 {
        tier3::populate_tier3(&mut out, &mut stream);
    }
    tier3::compute_derived_likelihoods(&mut out);

    out.width = width;
    out.height = height;
    out.megapixels = (width as f64 * height as f64 / 1_000_000.0) as f32;
    out.aspect_ratio = (width as f64 / (height as f64).max(1.0)) as f32;

    Ok(out)
}

/// Convenience entry for callers holding a packed RGB8 buffer
/// (`width * height * 3` bytes, no row stride). Panics on length
/// mismatch — same contract as the prior version of this function.
///
/// # Panics
///
/// Panics if `rgb.len() != width * height * 3`.
pub fn analyze_rgb8(rgb: &[u8], width: u32, height: u32) -> AnalyzerOutput {
    let w = width as usize;
    let h = height as usize;
    assert_eq!(
        rgb.len(),
        w * h * 3,
        "analyze_rgb8: RGB8 buffer size mismatch"
    );
    let stride = w * 3;
    let slice = PixelSlice::new(rgb, width, height, stride, PixelDescriptor::RGB8_SRGB)
        .expect("RGB8 PixelSlice from packed buffer");
    analyze(slice).expect("analyze never fails on RGB8")
}

#[cfg(test)]
mod tests;
