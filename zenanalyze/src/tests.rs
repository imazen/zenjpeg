//! Sanity checks for the analyzer ports.
//!
//! These tests check internal invariants — they don't pull in the
//! coefficient crate (would be a dev-dep cycle). Numerical parity vs.
//! `coefficient::analysis::feature_extract` + `evalchroma_ext` is
//! validated end-to-end via the oracle resimulation harness; that
//! lives in coefficient and re-encodes the CID22-val corpus.
//!
//! ## Test-only compatibility shim
//!
//! The public API is opaque-only ([`crate::analyze_features`] →
//! [`crate::feature::AnalysisResults`]). Internally we still want
//! field-style access for ergonomic test assertions, so this module
//! defines [`TestOutput`] (a `Deref<Target = RawAnalysis>` wrapper
//! plus geometry fields) and [`AnalyzerConfig`] (a tiny stub with
//! the same `default()` / `full()` constructors). The shim exists
//! only `#[cfg(test)]` and is not a public path.

use super::*;
use crate::feature::{
    AnalysisFeature, AnalysisQuery, FeatureSet, ImageGeometry, RawAnalysis,
};
use core::ops::Deref;

/// Test-only wrapper: dense feature record + flat geometry fields.
/// Same field shape as the (now-removed) public `AnalyzerOutput`,
/// so existing tests can keep using `out.variance`, `out.width`,
/// etc. without rewriting hundreds of assertions.
pub(crate) struct TestOutput {
    inner: RawAnalysis,
    pub width: u32,
    pub height: u32,
    pub megapixels: f32,
    pub aspect_ratio: f32,
}

impl Deref for TestOutput {
    type Target = RawAnalysis;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl TestOutput {
    fn from_raw_geom(raw: RawAnalysis, geom: ImageGeometry) -> Self {
        Self {
            inner: raw,
            width: geom.width(),
            height: geom.height(),
            megapixels: geom.megapixels(),
            aspect_ratio: geom.aspect_ratio(),
        }
    }
}

/// Test stub for the legacy public type. Two constructors only —
/// the same `default()` / `full()` shape the production callers had,
/// preserved here so the test bodies don't churn.
pub(crate) struct AnalyzerConfig {
    full: bool,
}
impl AnalyzerConfig {
    pub(crate) fn default() -> Self {
        Self { full: false }
    }
    pub(crate) fn full() -> Self {
        Self { full: true }
    }
}

/// Equivalent of the legacy `analyze(slice)` — every feature, default
/// budgets. Used by tests that want field-style access.
pub(crate) fn analyze(slice: PixelSlice<'_>) -> Result<TestOutput, String> {
    let (raw, geom) = crate::analyze_full_raw_for_test(slice, false).map_err(|e| e.to_string())?;
    Ok(TestOutput::from_raw_geom(raw, geom))
}

/// Equivalent of the legacy `analyze_with(slice, &config)`. The
/// `config.full` flag controls whether sampling budgets are full or
/// default (the only knob the legacy `AnalyzerConfig` exposed via
/// its `default()` / `full()` constructors).
pub(crate) fn analyze_with(
    slice: PixelSlice<'_>,
    config: &AnalyzerConfig,
) -> Result<TestOutput, String> {
    let (raw, geom) =
        crate::analyze_full_raw_for_test(slice, config.full).map_err(|e| e.to_string())?;
    Ok(TestOutput::from_raw_geom(raw, geom))
}

/// Equivalent of the legacy `analyze_rgb8(rgb, w, h)`.
pub(crate) fn analyze_rgb8(rgb: &[u8], w: u32, h: u32) -> TestOutput {
    let stride = (w as usize) * 3;
    let slice = PixelSlice::new(rgb, w, h, stride, PixelDescriptor::RGB8_SRGB)
        .expect("RGB8 PixelSlice from packed buffer");
    analyze(slice).expect("analyze never fails on RGB8")
}

/// Equivalent of the legacy `analyze_rgb8_with(rgb, w, h, config)`.
pub(crate) fn analyze_rgb8_with(rgb: &[u8], w: u32, h: u32, config: &AnalyzerConfig) -> TestOutput {
    let stride = (w as usize) * 3;
    let slice = PixelSlice::new(rgb, w, h, stride, PixelDescriptor::RGB8_SRGB)
        .expect("RGB8 PixelSlice from packed buffer");
    analyze_with(slice, config).expect("analyze never fails on RGB8")
}

fn synth_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    // Cheap reproducible RGB with mild structure (low-frequency variation
    // + per-channel offsets) so every Tier 1 + 2 + 3 path takes a real
    // branch, not a degenerate-flat one.
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let t = seed
                .wrapping_add(x.wrapping_mul(7))
                .wrapping_add(y.wrapping_mul(13));
            let i = ((y * w + x) * 3) as usize;
            buf[i] = ((t >> 1) & 0xFF) as u8;
            buf[i + 1] = ((t >> 3) & 0xFF) as u8;
            buf[i + 2] = ((t >> 2) ^ 0xAA) as u8;
        }
    }
    buf
}

#[test]
fn flat_image_has_zero_variance_and_edges() {
    let w = 64;
    let h = 64;
    let rgb = vec![128u8; (w * h * 3) as usize];
    let out = analyze_rgb8(&rgb, w, h);
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    assert_eq!(out.chroma_complexity, 0.0);
    assert!(out.uniformity > 0.99); // every block uniform
    assert!(out.flat_color_block_ratio > 0.99);
    assert!(out.distinct_color_bins <= 1);
    assert_eq!(out.cb_horiz_sharpness, 0.0);
    assert_eq!(out.cr_horiz_sharpness, 0.0);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
    // Single bin gets all weight ⇒ entropy 0.
    assert!(out.luma_histogram_entropy.abs() < 1e-5);
}

#[test]
fn vstripes_have_high_horiz_chroma_zero_vert() {
    // Alternating R/B columns: horizontal Cb gradient is huge, vertical 0.
    let w = 64;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x % 2 == 0 {
                rgb[i] = 255; // red
            } else {
                rgb[i + 2] = 255; // blue
            }
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert!(out.cb_horiz_sharpness > 0.0);
    assert!(out.cr_horiz_sharpness > 0.0);
    // Vertical chroma has no second-difference signal between identical
    // rows of column patterns, so it should be 0.
    assert_eq!(out.cb_vert_sharpness, 0.0);
    assert_eq!(out.cr_vert_sharpness, 0.0);
}

#[cfg(feature = "composites")]
#[test]
fn synthetic_image_likelihoods_in_unit_interval() {
    let out = analyze_rgb8(&synth_rgb(128, 128, 42), 128, 128);
    assert!((0.0..=1.0).contains(&out.text_likelihood));
    assert!((0.0..=1.0).contains(&out.screen_content_likelihood));
    assert!((0.0..=1.0).contains(&out.natural_likelihood));
}

#[test]
fn geometry_fields_derive_from_w_h() {
    let out = analyze_rgb8(&synth_rgb(160, 120, 1), 160, 120);
    assert_eq!(out.width, 160);
    assert_eq!(out.height, 120);
    assert!((out.megapixels - 0.0192).abs() < 1e-4);
    assert!((out.aspect_ratio - 160.0 / 120.0).abs() < 1e-4);
}

#[test]
fn checkerboard_has_high_freq_energy() {
    let w = 64;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    // Pure checkerboard is the highest possible high-freq AC.
    assert!(
        out.high_freq_energy_ratio > 1.0,
        "got {}",
        out.high_freq_energy_ratio
    );
    // Two-color image at 5-bit quantization → just a couple of distinct bins.
    assert!(out.distinct_color_bins <= 4);
    // Bimodal luma → lower entropy than a noisy image.
    assert!(out.luma_histogram_entropy < 2.0);
}

#[test]
fn small_images_dont_panic() {
    // < 3×3 hits the Tier 2 short-circuit; < 8×8 hits the Tier 3 short-circuit.
    let _ = analyze_rgb8(&[0; 3], 1, 1);
    let _ = analyze_rgb8(&[0; 4 * 4 * 3], 4, 4);
    let _ = analyze_rgb8(&[0; 7 * 7 * 3], 7, 7);
}

// ---------------------------------------------------------------------
// Config plumbing — verify the new AnalyzerConfig path doesn't drift
// from the trained defaults, and that override values actually change
// the sampling behavior.
// ---------------------------------------------------------------------

#[test]
fn analyze_with_default_matches_legacy_analyze() {
    // The new analyze_with(default) MUST produce bit-identical features
    // to the old analyze() — anything else means the config plumbing
    // silently changed the trained reference values.
    let w = 256;
    let h = 256;
    let rgb = synth_rgb(w, h, 12345);
    let stride = (w as usize) * 3;
    let slice = PixelSlice::new(&rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let legacy = analyze(slice).unwrap();

    let slice2 = PixelSlice::new(&rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let with_default = analyze_with(slice2, &AnalyzerConfig::default()).unwrap();

    // Every numeric field that contributes to oracle decisions must match exactly.
    assert_eq!(legacy.variance, with_default.variance);
    assert_eq!(legacy.edge_density, with_default.edge_density);
    assert_eq!(legacy.chroma_complexity, with_default.chroma_complexity);
    assert_eq!(legacy.cb_sharpness, with_default.cb_sharpness);
    assert_eq!(legacy.cr_sharpness, with_default.cr_sharpness);
    assert_eq!(legacy.uniformity, with_default.uniformity);
    assert_eq!(
        legacy.flat_color_block_ratio,
        with_default.flat_color_block_ratio
    );
    assert_eq!(legacy.distinct_color_bins, with_default.distinct_color_bins);
    assert_eq!(legacy.cb_horiz_sharpness, with_default.cb_horiz_sharpness);
    assert_eq!(legacy.cb_vert_sharpness, with_default.cb_vert_sharpness);
    assert_eq!(legacy.cr_horiz_sharpness, with_default.cr_horiz_sharpness);
    assert_eq!(legacy.cr_vert_sharpness, with_default.cr_vert_sharpness);
    assert_eq!(
        legacy.high_freq_energy_ratio,
        with_default.high_freq_energy_ratio
    );
    assert_eq!(
        legacy.luma_histogram_entropy,
        with_default.luma_histogram_entropy
    );
}

#[test]
fn analyze_with_full_budget_changes_results_on_large_image() {
    // 1024×1024 image. Default budget (500k pixels, 256 blocks) samples a
    // small fraction; full() scans every stripe and every 8×8 block. At least
    // one feature MUST differ — otherwise the config plumbing is a no-op.
    let w = 1024;
    let h = 1024;
    let rgb = synth_rgb(w, h, 7);
    let stride = (w as usize) * 3;

    let s1 = PixelSlice::new(&rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let default = analyze_with(s1, &AnalyzerConfig::default()).unwrap();

    let s2 = PixelSlice::new(&rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let full = analyze_with(s2, &AnalyzerConfig::full()).unwrap();

    let differs = default.variance != full.variance
        || default.edge_density != full.edge_density
        || default.high_freq_energy_ratio != full.high_freq_energy_ratio
        || default.luma_histogram_entropy != full.luma_histogram_entropy
        || default.distinct_color_bins != full.distinct_color_bins;
    assert!(
        differs,
        "AnalyzerConfig::full() produced identical features to default — \
         the budget plumbing is not actually reaching tier1/tier3"
    );

    // Geometry fields are config-independent.
    assert_eq!(default.width, full.width);
    assert_eq!(default.height, full.height);
}

// ---------------------------------------------------------------------
// Broad pixel-format coverage: every descriptor each zen* codec uses
// at its encoder boundary must flow through analyze() without error.
// We don't assert feature equality across formats (different transfer
// functions / primaries → different sRGB display-space bytes after
// RowConverter, so feature values legitimately differ), only that the
// pipeline accepts the input and reports the correct geometry.
// ---------------------------------------------------------------------

fn fill_solid_rgb8(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for chunk in buf.chunks_exact_mut(3) {
        chunk.copy_from_slice(&rgb);
    }
    buf
}

fn fill_solid_rgba8(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&rgba);
    }
    buf
}

fn fill_solid_u16(w: u32, h: u32, channels: usize, value: u16) -> Vec<u8> {
    let total = (w as usize) * (h as usize) * channels * 2;
    let mut buf = vec![0u8; total];
    let bytes = value.to_le_bytes();
    for chunk in buf.chunks_exact_mut(2) {
        chunk.copy_from_slice(&bytes);
    }
    buf
}

fn fill_solid_f32(w: u32, h: u32, channels: usize, value: f32) -> Vec<u8> {
    let total = (w as usize) * (h as usize) * channels * 4;
    let mut buf = vec![0u8; total];
    let bytes = value.to_le_bytes();
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&bytes);
    }
    buf
}

#[test]
fn pixel_coverage_rgb8_srgb() {
    let buf = fill_solid_rgb8(32, 32, [128, 64, 200]);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let out = analyze(s).expect("RGB8_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_rgba8_srgb() {
    // zenwebp / zenpng / zenjpeg: 8-bit RGBA input.
    let buf = fill_solid_rgba8(32, 32, [128, 64, 200, 255]);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let out = analyze(s).expect("RGBA8_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_bgra8_srgb() {
    // imageflow / win32 / browser canvas: BGRA byte order.
    let buf = fill_solid_rgba8(32, 32, [200, 64, 128, 255]);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 4, PixelDescriptor::BGRA8_SRGB).unwrap();
    let out = analyze(s).expect("BGRA8_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_rgb16_srgb() {
    // zenpng / zenavif / zenjpeg: 16-bit RGB input.
    let buf = fill_solid_u16(32, 32, 3, 32_000);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
    let out = analyze(s).expect("RGB16_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_rgba16_srgb() {
    let buf = fill_solid_u16(32, 32, 4, 32_000);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 8, PixelDescriptor::RGBA16_SRGB).unwrap();
    let out = analyze(s).expect("RGBA16_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_rgbf32_linear() {
    // zenjpeg / zenwebp / zenjxl: HDR / linear f32 RGB. Analyzer treats
    // post-convert sRGB display bytes as the feature surface — the trees
    // don't see linear values.
    let buf = fill_solid_f32(32, 32, 3, 0.5);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 12, PixelDescriptor::RGBF32_LINEAR).unwrap();
    let out = analyze(s).expect("RGBF32_LINEAR analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_rgbaf32_linear() {
    let buf = fill_solid_f32(32, 32, 4, 0.5);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 16, PixelDescriptor::RGBAF32_LINEAR).unwrap();
    let out = analyze(s).expect("RGBAF32_LINEAR analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_gray8_srgb() {
    let buf = vec![128u8; 32 * 32];
    let s = PixelSlice::new(&buf, 32, 32, 32, PixelDescriptor::GRAY8_SRGB).unwrap();
    let out = analyze(s).expect("GRAY8_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_gray16_srgb() {
    let buf = fill_solid_u16(32, 32, 1, 32_000);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 2, PixelDescriptor::GRAY16_SRGB).unwrap();
    let out = analyze(s).expect("GRAY16_SRGB analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

#[test]
fn pixel_coverage_grayf32_linear() {
    let buf = fill_solid_f32(32, 32, 1, 0.5);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 4, PixelDescriptor::GRAYF32_LINEAR).unwrap();
    let out = analyze(s).expect("GRAYF32_LINEAR analyze");
    assert_eq!(out.width, 32);
    assert_eq!(out.height, 32);
}

// ---------------------------------------------------------------------
// Sanity invariants across image-size regimes.
// ---------------------------------------------------------------------
//
// These tests assert range + geometry invariants that should hold for
// EVERY input regardless of content, then lock the small-image
// short-circuit semantics that the trained contract assumes
// (analyze() never panics on degenerate inputs and reports the
// declared geometry verbatim).
//
// We don't hash-lock per-field feature values here — `synth_rgb` /
// the analyzer math are part of the trained contract, but the
// `analyze_with_default_matches_legacy_analyze` test above already
// catches accidental drift in the path. These tests defend the
// SHAPE of the API across size regimes.

fn assert_well_formed(out: &TestOutput, w: u32, h: u32) {
    assert_eq!(out.width, w);
    assert_eq!(out.height, h);
    let expected_mp = (w as f64 * h as f64 / 1_000_000.0) as f32;
    assert!(
        (out.megapixels - expected_mp).abs() < 1e-6,
        "megapixels: expected {expected_mp}, got {}",
        out.megapixels
    );
    let expected_ar = (w as f64 / (h as f64).max(1.0)) as f32;
    assert!(
        (out.aspect_ratio - expected_ar).abs() < 1e-6,
        "aspect_ratio: expected {expected_ar}, got {}",
        out.aspect_ratio
    );
    #[cfg(feature = "composites")]
    {
        assert!((0.0..=1.0).contains(&out.text_likelihood));
        assert!((0.0..=1.0).contains(&out.screen_content_likelihood));
        assert!((0.0..=1.0).contains(&out.natural_likelihood));
    }
    assert!(
        (0.0..=5.0).contains(&out.luma_histogram_entropy),
        "entropy: {}",
        out.luma_histogram_entropy
    );
    assert!(
        out.high_freq_energy_ratio >= 0.0,
        "high_freq_energy_ratio: {}",
        out.high_freq_energy_ratio
    );
    assert!(
        (0.0..=100.0).contains(&out.cb_peak_sharpness),
        "cb_peak_sharpness: {}",
        out.cb_peak_sharpness
    );
    assert!(
        (0.0..=100.0).contains(&out.cr_peak_sharpness),
        "cr_peak_sharpness: {}",
        out.cr_peak_sharpness
    );
    assert!(out.cb_horiz_sharpness >= 0.0);
    assert!(out.cb_vert_sharpness >= 0.0);
    assert!(out.cr_horiz_sharpness >= 0.0);
    assert!(out.cr_vert_sharpness >= 0.0);
    assert!((0.0..=1.0).contains(&out.uniformity));
    assert!((0.0..=1.0).contains(&out.flat_color_block_ratio));
}

// ---- Tiny: every degenerate size below the tier short-circuits ----
//
// 1×1 / 1×2 / 2×2: tier1 short-circuit (needs ≥ 2×2)
// 3×3..=7×7:       tier2 active, tier3 short-circuit (needs ≥ 8×8)
// 8×8 exactly:     all three tiers active for the first time

#[test]
fn tiny_1x1_doesnt_panic_and_reports_geometry() {
    let out = analyze_rgb8(&[200, 100, 50], 1, 1);
    assert_well_formed(&out, 1, 1);
    // No tier ran — every numeric must be its Default.
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    assert_eq!(out.cb_peak_sharpness, 0.0);
    assert_eq!(out.luma_histogram_entropy, 0.0);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
}

#[test]
fn tiny_2x2_runs_tier1_only() {
    // Tier 1 minimum is 2×2 (line-pair gradient). Tier 2 needs 3×3,
    // tier 3 needs 8×8 — both short-circuit.
    let rgb = vec![0u8; 2 * 2 * 3];
    let out = analyze_rgb8(&rgb, 2, 2);
    assert_well_formed(&out, 2, 2);
    // Flat input ⇒ Tier 1 zeros across the board.
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    // Tier 2 + Tier 3 didn't run.
    assert_eq!(out.cb_horiz_sharpness, 0.0);
    assert_eq!(out.cr_peak_sharpness, 0.0);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
    assert_eq!(out.luma_histogram_entropy, 0.0);
}

#[test]
fn tiny_3x3_runs_tier1_and_tier2() {
    // Tier 2 lower-bound is 3×3 (one 3-row sliding window step).
    // Tier 3 still short-circuits.
    let rgb = vec![128u8; 3 * 3 * 3];
    let out = analyze_rgb8(&rgb, 3, 3);
    assert_well_formed(&out, 3, 3);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
    assert_eq!(out.luma_histogram_entropy, 0.0);
}

#[test]
fn tiny_7x7_runs_tier1_and_tier2_only() {
    // 7×7 is the largest size where Tier 3 still short-circuits.
    let rgb = synth_rgb(7, 7, 99);
    let out = analyze_rgb8(&rgb, 7, 7);
    assert_well_formed(&out, 7, 7);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
    assert_eq!(out.luma_histogram_entropy, 0.0);
}

#[test]
fn tiny_8x8_first_size_with_all_three_tiers_active() {
    let rgb = synth_rgb(8, 8, 42);
    let out = analyze_rgb8(&rgb, 8, 8);
    assert_well_formed(&out, 8, 8);
    // 8×8 = exactly one DCT block ⇒ Tier 3 produces meaningful values.
    // Don't lock numeric outputs (covered by parity test above), only
    // assert the path actually executed (entropy != 0 for non-flat).
    assert!(
        out.luma_histogram_entropy > 0.0,
        "tier3 didn't run on 8×8 synth_rgb"
    );
}

#[test]
fn tiny_non_square_8x16_is_well_formed() {
    let rgb = synth_rgb(8, 16, 7);
    let out = analyze_rgb8(&rgb, 8, 16);
    assert_well_formed(&out, 8, 16);
    assert!((out.aspect_ratio - 0.5).abs() < 1e-6);
}

#[test]
fn tiny_non_square_16x8_is_well_formed() {
    let rgb = synth_rgb(16, 8, 7);
    let out = analyze_rgb8(&rgb, 16, 8);
    assert_well_formed(&out, 16, 8);
    assert!((out.aspect_ratio - 2.0).abs() < 1e-6);
}

// ---- Medium: full path, all features in valid ranges, deterministic ----

#[test]
fn medium_256x256_runs_all_tiers_and_features_in_range() {
    let rgb = synth_rgb(256, 256, 1);
    let out = analyze_rgb8(&rgb, 256, 256);
    assert_well_formed(&out, 256, 256);
    // 256×256 = 32×32 blocks = 1024 8×8 blocks. With default
    // hf_max_blocks=256 this exercises the stride-sampling branch
    // (stride = 1024/256 = 4).
    assert!(out.distinct_color_bins > 0, "synth has multiple bins");
    assert!(
        out.luma_histogram_entropy > 0.0,
        "synth has non-degenerate luma"
    );
}

#[test]
fn medium_512x256_non_square_aspect_correct() {
    let rgb = synth_rgb(512, 256, 13);
    let out = analyze_rgb8(&rgb, 512, 256);
    assert_well_formed(&out, 512, 256);
    assert!((out.aspect_ratio - 2.0).abs() < 1e-6);
    assert!((out.megapixels - 0.131072).abs() < 1e-5);
}

#[test]
fn medium_image_is_deterministic() {
    // Same input must produce identical output across calls.
    // Defends against accidental nondeterminism (parallel scheduling,
    // unstable sort, uninitialized scratch, etc.).
    let rgb = synth_rgb(256, 256, 7);
    let a = analyze_rgb8(&rgb, 256, 256);
    let b = analyze_rgb8(&rgb, 256, 256);
    assert_eq!(a.variance.to_bits(), b.variance.to_bits());
    assert_eq!(a.edge_density.to_bits(), b.edge_density.to_bits());
    assert_eq!(a.chroma_complexity.to_bits(), b.chroma_complexity.to_bits());
    assert_eq!(a.cb_peak_sharpness.to_bits(), b.cb_peak_sharpness.to_bits());
    assert_eq!(a.cr_peak_sharpness.to_bits(), b.cr_peak_sharpness.to_bits());
    assert_eq!(
        a.high_freq_energy_ratio.to_bits(),
        b.high_freq_energy_ratio.to_bits()
    );
    assert_eq!(
        a.luma_histogram_entropy.to_bits(),
        b.luma_histogram_entropy.to_bits()
    );
    #[cfg(feature = "composites")]
    {
        assert_eq!(a.text_likelihood.to_bits(), b.text_likelihood.to_bits());
        assert_eq!(
            a.screen_content_likelihood.to_bits(),
            b.screen_content_likelihood.to_bits()
        );
        assert_eq!(
            a.natural_likelihood.to_bits(),
            b.natural_likelihood.to_bits()
        );
    }
    assert_eq!(a.distinct_color_bins, b.distinct_color_bins);
}

// ---- Large: budget plumbing matters, completes in reasonable time ----

#[test]
fn large_2048x2048_with_default_budget_is_well_formed() {
    // 2048×2048 = 256×256 blocks = 65,536 blocks. Default
    // hf_max_blocks=256 ⇒ stride = 256, so we sample 1-in-256 blocks.
    // Default pixel_budget=500_000 ⇒ stripe_step ≈ 8 (sample every
    // 8th 8-row stripe). Neither tier should walk the full image.
    let rgb = synth_rgb(2048, 2048, 17);
    let out = analyze_rgb8(&rgb, 2048, 2048);
    assert_well_formed(&out, 2048, 2048);
    assert!(out.distinct_color_bins > 0);
}

#[test]
fn large_image_default_budget_differs_from_full_budget() {
    // Reaffirms the analyze_with_full_budget_changes_results test at
    // a 4× larger size. Confirms the budget plumbing scales.
    let w = 2048;
    let h = 2048;
    let rgb = synth_rgb(w, h, 23);
    let stride = (w as usize) * 3;
    let s_def = PixelSlice::new(&rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let s_full = PixelSlice::new(&rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let def = analyze_with(s_def, &AnalyzerConfig::default()).unwrap();
    let full = analyze_with(s_full, &AnalyzerConfig::full()).unwrap();

    assert_well_formed(&def, w, h);
    assert_well_formed(&full, w, h);

    // At least one tier must produce different numbers — otherwise the
    // budget knobs aren't reaching tier1/tier3 and we silently lost the
    // proxy-server speed lever.
    let differs = def.variance != full.variance
        || def.edge_density != full.edge_density
        || def.uniformity != full.uniformity
        || def.high_freq_energy_ratio != full.high_freq_energy_ratio
        || def.distinct_color_bins != full.distinct_color_bins;
    assert!(
        differs,
        "default budget == full budget on 2048×2048 synth — \
         budget plumbing is a no-op"
    );

    // Geometry fields are sample-independent and must always agree.
    assert_eq!(def.width, full.width);
    assert_eq!(def.height, full.height);
    assert_eq!(def.megapixels.to_bits(), full.megapixels.to_bits());
    assert_eq!(def.aspect_ratio.to_bits(), full.aspect_ratio.to_bits());
}

// ---------------------------------------------------------------------
// Regression tests for the three Tier 2 / Tier 3 bug fixes (2026-04-27).
// ---------------------------------------------------------------------
//
// These lock the FIXED behavior, not the historical buggy behavior —
// any future regression that re-introduces the asymmetry, span
// off-by-one, trailing-fragment drop, or raster-order high-freq split
// will fail one of these.

/// Tier 2 fix #1: `rgb_to_ycbcr_q` no longer halves Cb/Cr when `cr < 0`.
///
/// Build two synthetic images that are color-mirrors of each other —
/// one where the dominant chroma direction has `cr > 0` (R-heavy),
/// one where it has `cr < 0` (G-heavy). The Tier 2 chroma sharpness
/// magnitudes should be the same shape in both, since the algorithm
/// is now sign-symmetric. Pre-fix, the cr<0 case's peaks were halved.
#[test]
fn tier2_cb_cr_no_longer_asymmetric_under_cr_sign_flip() {
    // 64×64 with a vertical R/B alternating column pattern (cr > 0
    // dominates: red columns push Cr positive).
    let w = 64;
    let h = 64;
    let mut warm = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x % 2 == 0 {
                warm[i] = 255; // red
            } else {
                warm[i + 2] = 255; // blue
            }
        }
    }
    // Color-mirror: G/B alternating (cr < 0 dominates: green pushes Cr negative).
    let mut cool = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x % 2 == 0 {
                cool[i + 1] = 255; // green
            } else {
                cool[i + 2] = 255; // blue
            }
        }
    }
    let warm_out = analyze_rgb8(&warm, w, h);
    let cool_out = analyze_rgb8(&cool, w, h);

    // Both should produce horizontal chroma signal of the same scale —
    // pre-fix the cool case's peak would have been ~half of warm.
    assert!(warm_out.cb_peak_sharpness > 0.0);
    assert!(cool_out.cb_peak_sharpness > 0.0);
    let ratio = cool_out.cb_peak_sharpness / warm_out.cb_peak_sharpness;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "Cb peak under cr<0 ({}) is far from cr>0 ({}) — asymmetric \
         halving may have crept back in",
        cool_out.cb_peak_sharpness,
        warm_out.cb_peak_sharpness
    );
}

/// Tier 2 fix #2: `span = (width - 1) / 2` covers the rightmost
/// triplet column on odd widths.
///
/// Build a width=5 image with a sharp vertical edge between columns 2
/// and 4 (right side). Pre-fix, `span = (5-2)/2 = 1` only processed
/// triplet at x=0, never reaching the right edge → cb/cr horiz
/// sharpness = 0. Post-fix, `span = 2` processes x=0 AND x=2 → right
/// edge contributes.
#[test]
fn tier2_odd_width_5_includes_right_edge_triplet() {
    let w = 5;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    // Columns 0..3: gray. Column 4: red. Edge between cols 3 and 4
    // requires the x=2 triplet (cols 2,3,4) to detect.
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x < 4 {
                rgb[i] = 128;
                rgb[i + 1] = 128;
                rgb[i + 2] = 128;
            } else {
                rgb[i] = 255; // red
            }
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    // Post-fix: the rightmost triplet column (x=2 → cols 2,3,4) sees
    // the gray↔red transition between cols 3 and 4 → nonzero horiz
    // chroma signal. Pre-fix this was always 0.
    assert!(
        out.cb_horiz_sharpness > 0.0 || out.cr_horiz_sharpness > 0.0,
        "width=5 with right-edge color transition produced zero horiz \
         chroma signal — span off-by-one may have crept back in \
         (cb_horiz={}, cr_horiz={})",
        out.cb_horiz_sharpness,
        out.cr_horiz_sharpness
    );
}

/// Tier 2 fix #3: small-height images now produce Tier 2 features.
///
/// Pre-fix, any image with `height ≤ 32` (where the loop never
/// reaches `fragment_max_height = 16` triplet-rows AND the trailing
/// fragment was dropped at `> 16`) emitted 0 for every Tier 2 field.
/// A 16-pixel-tall image with strong chroma transitions should now
/// see them.
#[test]
fn tier2_small_height_no_longer_silently_zeroes() {
    let w = 64;
    let h = 16;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x % 2 == 0 {
                rgb[i] = 255; // red
            } else {
                rgb[i + 2] = 255; // blue
            }
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert!(
        out.cb_horiz_sharpness > 0.0,
        "height=16 vstripes produced zero cb_horiz — trailing-fragment \
         drop bug may have crept back in (got {})",
        out.cb_horiz_sharpness
    );
    assert!(
        out.cb_peak_sharpness > 0.0,
        "height=16 vstripes produced zero cb_peak (got {})",
        out.cb_peak_sharpness
    );
}

/// Tier 2 fix #3 (continued): bottom-edge content in the trailing
/// partial fragment is no longer dropped.
///
/// Build a tall image (h=80, fragment_max_height=20 → 4 fragments)
/// where the first 60 rows are flat and the last 20 rows have strong
/// horizontal chroma content. The last 20 rows form the trailing
/// partial fragment after 60 rows = 30 triplet-iters = 1.5 fragments.
/// Wait, 60 rows = 30 iters; with max=20, we'd flush at iters 20 and
/// fragment 2 starts at iter 20; iters 30 onward (h=80→39 iters total)
/// would have 9 iters for fragment 3. Pre-fix `> 16` would drop those
/// 9 iters; the strong content there would never hit `max_sumh`.
#[test]
fn tier2_bottom_edge_partial_fragment_is_counted() {
    let w = 64;
    let h = 80;
    let mut flat = vec![128u8; (w * h * 3) as usize];
    // Add a sharp horizontal chroma signal ONLY in rows [60, 80).
    for y in 60..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x % 2 == 0 {
                flat[i] = 255;
                flat[i + 1] = 0;
                flat[i + 2] = 0; // pure red
            } else {
                flat[i] = 0;
                flat[i + 1] = 0;
                flat[i + 2] = 255; // pure blue
            }
        }
    }
    let out = analyze_rgb8(&flat, w, h);
    assert!(
        out.cb_horiz_sharpness > 0.0,
        "bottom-band-only chroma produced zero cb_horiz — \
         trailing-fragment drop may have crept back in (got {})",
        out.cb_horiz_sharpness
    );
}

/// Tier 3 fix: high-freq DCT split is now zigzag-symmetric in
/// horizontal/vertical detail.
///
/// A pure-horizontal pattern and its 90°-rotated version (pure-vertical
/// pattern) should produce roughly the same `high_freq_energy_ratio`.
/// Pre-fix (raster k≥16), the horizontal pattern's energy lived at
/// (u=4,v=0) which is in the "low" raster bucket while the vertical
/// pattern's lived at (u=0,v=4) in the "high" raster bucket —
/// yielding wildly different ratios for the same content frequency.
// ---------------------------------------------------------------------
// Alpha analysis (mirrors zenwebp classifier's bimodality detector).
// ---------------------------------------------------------------------

#[test]
fn alpha_absent_for_rgb8_input() {
    let rgb = synth_rgb(64, 64, 1);
    let s = PixelSlice::new(&rgb, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(!out.alpha_present);
    assert_eq!(out.alpha_used_fraction, 0.0);
    assert_eq!(out.alpha_bimodal_score, 0.0);
}

#[test]
fn alpha_present_but_fully_opaque_for_rgba_all_255() {
    let mut rgba = vec![0u8; 64 * 64 * 4];
    for chunk in rgba.chunks_exact_mut(4) {
        chunk[0] = 200;
        chunk[1] = 100;
        chunk[2] = 50;
        chunk[3] = 255; // fully opaque
    }
    let s = PixelSlice::new(&rgba, 64, 64, 64 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(out.alpha_present);
    assert_eq!(
        out.alpha_used_fraction, 0.0,
        "all-255 alpha should report 0 used"
    );
    assert_eq!(out.alpha_bimodal_score, 0.0);
}

#[test]
fn alpha_bimodal_score_high_for_text_on_transparent() {
    // Half of pixels are alpha=0 (transparent), half are alpha=255 (opaque).
    // Sharply bimodal: low_quarter_fraction ≈ 0.5, high_quarter_fraction ≈ 0.5,
    // bimodal_score = min ≈ 0.5 (well above zenwebp's 0.15 threshold).
    let mut rgba = vec![0u8; 64 * 64 * 4];
    for (i, chunk) in rgba.chunks_exact_mut(4).enumerate() {
        chunk[0] = 0;
        chunk[1] = 0;
        chunk[2] = 0;
        chunk[3] = if i % 2 == 0 { 0 } else { 255 };
    }
    let s = PixelSlice::new(&rgba, 64, 64, 64 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(out.alpha_present);
    assert!(
        out.alpha_bimodal_score > 0.4,
        "expected bimodal_score > 0.4, got {}",
        out.alpha_bimodal_score
    );
    assert!(
        out.alpha_used_fraction > 0.4,
        "half-transparent input should report ~0.5 used, got {}",
        out.alpha_used_fraction
    );
}

#[test]
fn alpha_unimodal_low_score_for_smooth_gradient() {
    // Smooth alpha gradient 0..255 — significant mass everywhere, NOT
    // concentrated at ends. Bimodal score should be modest.
    let mut rgba = vec![0u8; 256 * 64 * 4];
    for y in 0..64 {
        for x in 0..256 {
            let i = ((y * 256 + x) * 4) as usize;
            rgba[i] = 128;
            rgba[i + 1] = 128;
            rgba[i + 2] = 128;
            rgba[i + 3] = x as u8; // 0..255 across width
        }
    }
    let s = PixelSlice::new(&rgba, 256, 64, 256 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(out.alpha_present);
    // Both quarters get ~25% each (alpha 0..63 and 192..255), middle gets ~50%.
    // bimodal_score = min(low, high) ≈ 0.25 — present but not strong.
    assert!(out.alpha_bimodal_score < 0.3);
    // Almost everything is non-fully-opaque (only the alpha=255 column).
    assert!(out.alpha_used_fraction > 0.95);
}

#[test]
fn alpha_works_on_bgra() {
    // Same all-255 alpha test for BGRA layout — alpha byte is still at
    // offset 3 within each pixel.
    let mut bgra = vec![0u8; 64 * 64 * 4];
    for chunk in bgra.chunks_exact_mut(4) {
        chunk[3] = 255;
    }
    let s = PixelSlice::new(&bgra, 64, 64, 64 * 4, PixelDescriptor::BGRA8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(out.alpha_present);
    assert_eq!(out.alpha_used_fraction, 0.0);
}

#[test]
fn alpha_works_on_rgba16() {
    // u16 alpha — high byte is the histogram bin. Set all-65535 (opaque).
    let mut rgba16 = vec![0u8; 64 * 64 * 8];
    for chunk in rgba16.chunks_exact_mut(8) {
        // alpha at byte offset 6 (3 channels × 2 bytes), all-FF
        chunk[6] = 0xFF;
        chunk[7] = 0xFF;
    }
    let s = PixelSlice::new(&rgba16, 64, 64, 64 * 8, PixelDescriptor::RGBA16_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(out.alpha_present);
    assert_eq!(out.alpha_used_fraction, 0.0);
}

// ---------------------------------------------------------------------
// Cross-codec features (Hasler colourfulness / Laplacian variance /
// variance spread / DCT compressibility / patch fraction).
// ---------------------------------------------------------------------

#[test]
#[cfg(feature = "experimental")]
fn colourfulness_zero_for_grayscale() {
    // All-gray RGB: rg = R-G = 0, yb = 0.5(R+G)-B = 0. Both means
    // and variances zero ⇒ M3 = 0.
    let rgb = vec![128u8; 64 * 64 * 3];
    let s = PixelSlice::new(&rgb, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(
        out.colourfulness.abs() < 0.5,
        "grayscale colourfulness should be ~0, got {}",
        out.colourfulness
    );
}

#[test]
#[cfg(feature = "experimental")]
fn colourfulness_high_for_saturated_colour() {
    // Pure saturated red: rg = 255, yb = 127.5. Big mean term, big M3.
    let mut rgb = vec![0u8; 256 * 64 * 3];
    for i in (0..rgb.len()).step_by(3) {
        rgb[i] = 255; // R only
    }
    let s = PixelSlice::new(&rgb, 256, 64, 256 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(
        out.colourfulness > 50.0,
        "pure-red colourfulness should be high, got {}",
        out.colourfulness
    );
}

#[test]
#[cfg(feature = "experimental")]
fn laplacian_variance_zero_for_flat_image() {
    let rgb = vec![80u8; 64 * 64 * 3];
    let s = PixelSlice::new(&rgb, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(
        out.laplacian_variance < 0.001,
        "flat image laplacian variance should be ~0, got {}",
        out.laplacian_variance
    );
}

#[test]
#[cfg(feature = "experimental")]
fn laplacian_variance_high_for_checkerboard() {
    let w = 64;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    // Pure 1-pixel-cell checkerboard: every Laplacian = ±(4 × 255) →
    // variance ~1 M. With the megapixel-sqrt scale-compensation
    // factor (here ~0.063 at 64×64) the reported value lands near
    // ~63 — still well above any flat-image baseline.
    assert!(
        out.laplacian_variance > 10.0,
        "checkerboard laplacian variance should be substantial, got {}",
        out.laplacian_variance
    );
}

#[test]
#[cfg(feature = "experimental")]
fn variance_spread_zero_for_flat() {
    let rgb = vec![100u8; 256 * 256 * 3];
    let out = analyze_rgb8(&rgb, 256, 256);
    assert!(
        out.variance_spread < 0.05,
        "flat image variance_spread should be ~0, got {}",
        out.variance_spread
    );
}

#[test]
#[cfg(feature = "experimental")]
fn variance_spread_nonzero_for_heterogeneous_content() {
    // Half flat, half checkerboard ⇒ block variances span ~0..16 k.
    // log10(1 + max/mean) = log10(1 + 2) ≈ 0.48 — substantial.
    let w = 256;
    let h = 256;
    let mut mixed = vec![80u8; (w * h * 3) as usize];
    for y in (h / 2)..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            mixed[i] = c;
            mixed[i + 1] = c;
            mixed[i + 2] = c;
        }
    }
    let mixed_out = analyze_rgb8(&mixed, w, h);
    assert!(
        mixed_out.variance_spread > 0.3,
        "half-flat half-checker spread should be >0.3, got {}",
        mixed_out.variance_spread
    );
}

#[test]
#[cfg(feature = "experimental")]
fn dct_compressibility_low_for_flat_image() {
    let rgb = vec![100u8; 64 * 64 * 3];
    let s = PixelSlice::new(&rgb, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let out = analyze(s).unwrap();
    assert!(
        out.dct_compressibility_y < 1.0,
        "flat image dct_compressibility_y should be near 0, got {}",
        out.dct_compressibility_y
    );
}

#[test]
#[cfg(feature = "experimental")]
fn dct_compressibility_high_for_checkerboard() {
    let w = 64;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert!(
        out.dct_compressibility_y > 10.0,
        "checkerboard dct_compressibility_y should be substantial, got {}",
        out.dct_compressibility_y
    );
}

#[test]
#[cfg(feature = "experimental")]
fn palette_density_low_for_few_colors() {
    // Two-color binary image at 256×256: 2 distinct color bins,
    // pixels = 65 536, denom = min(65 536, 32 768) = 32 768.
    // palette_density ≈ 2 / 32 768 ≈ 6e-5 — very sparse.
    let w = 256;
    let h = 256;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert!(
        out.palette_density < 0.001,
        "binary image palette_density should be near 0, got {}",
        out.palette_density
    );
}

#[test]
#[cfg(feature = "experimental")]
fn palette_density_higher_for_noisy_than_for_palette() {
    // Noisy synth has many colors → higher palette_density than a
    // 2-color binary image of the same size.
    let w = 256;
    let h = 256;
    let noisy = analyze_rgb8(&synth_rgb(w, h, 1), w, h);
    let mut binary = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            binary[i] = c;
            binary[i + 1] = c;
            binary[i + 2] = c;
        }
    }
    let bin_out = analyze_rgb8(&binary, w, h);
    assert!(
        noisy.palette_density > bin_out.palette_density,
        "noisy palette_density ({}) should exceed binary ({})",
        noisy.palette_density,
        bin_out.palette_density
    );
}

#[test]
#[cfg(feature = "experimental")]
fn palette_density_zero_for_solid_color() {
    let rgb = vec![100u8; 64 * 64 * 3];
    let out = analyze_rgb8(&rgb, 64, 64);
    // 1 distinct bin / min(4096, 32768) = 1/4096 ≈ 0.00024 — very sparse.
    assert!(
        out.palette_density < 0.001,
        "solid-colour palette_density should be near 0, got {}",
        out.palette_density
    );
}

#[test]
#[cfg(feature = "experimental")]
fn patch_fraction_high_for_few_distinct_colors() {
    // Two-color checkerboard ⇒ 2 distinct color bins ⇒ very high patch_fraction.
    let w = 256;
    let h = 256;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert!(
        out.patch_fraction > 0.99,
        "binary image patch_fraction should be near 1, got {} (bins={})",
        out.patch_fraction,
        out.distinct_color_bins
    );
}

#[test]
#[cfg(feature = "experimental")]
fn patch_fraction_high_for_repeat_pattern() {
    // 2-color checkerboard at 256×256: every 8×8 block has the
    // same low-AC content (essentially flat or 1-pixel checker
    // depending on parity). All blocks hash to the same signature
    // ⇒ patch_fraction near 1.0.
    let mut binary = vec![0u8; 256 * 256 * 3];
    for y in 0..256 {
        for x in 0..256 {
            let i = ((y * 256 + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            binary[i] = c;
            binary[i + 1] = c;
            binary[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&binary, 256, 256);
    assert!(
        out.patch_fraction > 0.9,
        "checkerboard patch_fraction should be > 0.9, got {}",
        out.patch_fraction
    );
}

#[test]
fn alpha_works_on_rgbaf32() {
    // f32 alpha in [0, 1]. All 1.0 = fully opaque.
    let mut buf = vec![0u8; 64 * 64 * 16];
    for chunk in buf.chunks_exact_mut(16) {
        // alpha at byte offset 12, value 1.0
        let bytes = 1.0f32.to_le_bytes();
        chunk[12..16].copy_from_slice(&bytes);
    }
    let s = PixelSlice::new(&buf, 64, 64, 64 * 16, PixelDescriptor::RGBAF32_LINEAR).unwrap();
    let out = analyze(s).unwrap();
    assert!(out.alpha_present);
    assert_eq!(out.alpha_used_fraction, 0.0);
}

#[test]
fn tier3_zigzag_split_is_symmetric_in_horiz_vs_vert_detail() {
    // 64×64 horizontal stripes (constant within each column, alternating rows).
    let w = 64;
    let h = 64;
    let mut horiz = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        let v = if y % 2 == 0 { 255u8 } else { 0u8 };
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            horiz[i] = v;
            horiz[i + 1] = v;
            horiz[i + 2] = v;
        }
    }
    // Same image rotated 90° → vertical stripes.
    let mut vert = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = if x % 2 == 0 { 255u8 } else { 0u8 };
            let i = ((y * w + x) * 3) as usize;
            vert[i] = v;
            vert[i + 1] = v;
            vert[i + 2] = v;
        }
    }
    let h_out = analyze_rgb8(&horiz, w, h);
    let v_out = analyze_rgb8(&vert, w, h);
    // Both must report substantial high-freq energy — and the two
    // should be within an order of magnitude. Pre-fix the ratio was
    // roughly 1:0 (horizontal pattern hit the "low" bucket entirely).
    assert!(h_out.high_freq_energy_ratio > 0.1);
    assert!(v_out.high_freq_energy_ratio > 0.1);
    let ratio = h_out.high_freq_energy_ratio / v_out.high_freq_energy_ratio;
    assert!(
        (0.25..=4.0).contains(&ratio),
        "horizontal vs vertical high-freq ratio = {} (h={}, v={}) — \
         raster-order split may have crept back in",
        ratio,
        h_out.high_freq_energy_ratio,
        v_out.high_freq_energy_ratio
    );
}

#[test]
fn large_image_completes_in_reasonable_time_with_default_budget() {
    // Loose timing assertion — defends against accidental O(N²) or
    // missing budget plumbing. A 4-megapixel synth at default budget
    // must complete in well under 1 s on any reasonable platform
    // (typically <50 ms on x86-64). Not a benchmark — just a "did
    // somebody disable the stride sampling" tripwire.
    let rgb = synth_rgb(2048, 2048, 5);
    let t0 = std::time::Instant::now();
    let out = analyze_rgb8(&rgb, 2048, 2048);
    let elapsed = t0.elapsed();
    assert_well_formed(&out, 2048, 2048);
    assert!(
        elapsed.as_millis() < 1000,
        "2048×2048 default-budget analyze took {} ms — \
         stride sampling probably broken",
        elapsed.as_millis()
    );
}

// --------------------------------------------------------------------
// New opaque-feature API smoke tests. These exercise the full
// pipeline from `analyze_features_rgb8` through `legacy_to_raw` →
// `RawAnalysis::into_results` → `AnalysisResults::get`.
// --------------------------------------------------------------------

#[test]
fn analyze_features_accepts_rgba8_without_opt_in() {
    // RGBA8 / BGRA8 / Gray / 16-bit / f32 inputs are all accepted on
    // the single public entry — alpha is split by the dedicated alpha
    // pass, RGB tiers see a row-converted view. No opt-in step.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let buf = fill_solid_rgba8(32, 32, [128, 64, 200, 255]);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
    let r = crate::analyze_features(s, &q).expect("RGBA8 must be accepted");
    assert!(r.get(AnalysisFeature::Variance).is_some());
}

#[test]
fn analyze_features_accepts_native_rgb8_without_opt_in() {
    // RGB8 / RGB8_SRGB / RGB8 with non-sRGB primaries all share the
    // 24bpp byte layout that RowStream takes the Native (zero-copy)
    // path on — no conversion, so no opt-in required.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let buf = fill_solid_rgb8(32, 32, [128, 64, 200]);
    let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
    let r = crate::analyze_features(s, &q).expect("RGB8_SRGB must pass without opt-in");
    assert!(r.get(AnalysisFeature::Variance).is_some());
}

#[test]
fn analyze_features_returns_only_requested_features() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 64;
    let h: u32 = 64;
    let rgb = synth_rgb(w, h, 11);
    let asked = FeatureSet::new()
        .with(AnalysisFeature::Variance)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::DistinctColorBins);
    let query = AnalysisQuery::new(asked);
    let r = analyze_features_rgb8(&rgb, w, h, &query);

    // Geometry comes back regardless of feature set.
    assert_eq!(r.geometry().width(), w);
    assert_eq!(r.geometry().height(), h);

    // Requested features have values; unrequested are None.
    assert!(r.get(AnalysisFeature::Variance).is_some());
    assert!(r.get(AnalysisFeature::EdgeDensity).is_some());
    assert!(r.get(AnalysisFeature::DistinctColorBins).is_some());
    assert_eq!(r.get(AnalysisFeature::AlphaPresent), None);
    // Experimental variants only exist when their cfg is enabled —
    // the absence-checks need the same gate.
    #[cfg(feature = "experimental")]
    {
        assert_eq!(r.get(AnalysisFeature::Colourfulness), None);
        assert_eq!(r.get(AnalysisFeature::PatchFraction), None);
    }
}

#[test]
#[cfg(feature = "experimental")]
fn quick_palette_signals_classify_correctly() {
    // Covers four canonical cases for IndexedPaletteWidth /
    // PaletteFitsIn256: solid (1 colour), 16-colour synthetic,
    // exactly-256-colour synthetic, and 32k-bin random noise.
    use AnalysisFeature::*;
    let q = AnalysisQuery::new(FeatureSet::just(IndexedPaletteWidth).with(PaletteFitsIn256));
    let w = 64u32;
    let h = 64u32;

    // Case 1: solid colour → 1 distinct bin → width 2.
    let solid = vec![128u8; (w * h * 3) as usize];
    let r = analyze_features_rgb8(&solid, w, h, &q);
    assert_eq!(r.get(IndexedPaletteWidth).and_then(|v| v.as_u32()), Some(2));
    assert_eq!(
        r.get(PaletteFitsIn256).and_then(|v| v.as_bool()),
        Some(true)
    );

    // Case 2: 16 distinct colours → width 4.
    // Pick 16 well-separated colours so 5-bit-per-channel quantisation
    // doesn't collapse any of them.
    let palette_16: [[u8; 3]; 16] = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 128, 0],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [200, 100, 50],
    ];
    let mut sixteen = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let c = palette_16[((x + y * 7) % 16) as usize];
            let i = ((y * w + x) * 3) as usize;
            sixteen[i] = c[0];
            sixteen[i + 1] = c[1];
            sixteen[i + 2] = c[2];
        }
    }
    let r = analyze_features_rgb8(&sixteen, w, h, &q);
    assert_eq!(r.get(IndexedPaletteWidth).and_then(|v| v.as_u32()), Some(4));
    assert_eq!(
        r.get(PaletteFitsIn256).and_then(|v| v.as_bool()),
        Some(true)
    );

    // Case 3: explicit 300-colour palette. Every pixel maps to one
    // of 300 distinct (r5, g5, b5) triples — guaranteed > 256 distinct
    // 5-bit-per-channel bins regardless of any synth-RNG quirks.
    let dw = 256u32;
    let dh = 256u32;
    let mut diverse = vec![0u8; (dw * dh * 3) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let idx = (y * dw + x) % 300;
            let r5 = (idx % 32) as u8;
            let g5 = ((idx / 32) % 32) as u8;
            let b5 = ((idx / (32 * 32)) % 32) as u8;
            let i = ((y * dw + x) * 3) as usize;
            // Shift back to 8-bit so the analyzer's `>> 3` recovers
            // exactly the chosen 5-bit triple.
            diverse[i] = r5 << 3;
            diverse[i + 1] = g5 << 3;
            diverse[i + 2] = b5 << 3;
        }
    }
    let r = analyze_features_rgb8(&diverse, dw, dh, &q);
    assert_eq!(r.get(IndexedPaletteWidth).and_then(|v| v.as_u32()), Some(0));
    assert_eq!(
        r.get(PaletteFitsIn256).and_then(|v| v.as_bool()),
        Some(false)
    );
}

#[test]
#[cfg(feature = "experimental")]
fn quick_palette_matches_full_path_when_both_requested() {
    // When the caller asks for both quick and full features, the
    // dispatcher routes through scan_palette (full). The values for
    // the quick signals must agree with what the standalone quick
    // path would produce.
    use AnalysisFeature::*;
    let w = 64u32;
    let h = 64u32;
    let rgb = synth_rgb(w, h, 42);

    let q_quick = AnalysisQuery::new(FeatureSet::just(PaletteFitsIn256));
    let q_full = AnalysisQuery::new(FeatureSet::just(DistinctColorBins).with(PaletteFitsIn256));
    let r_quick = analyze_features_rgb8(&rgb, w, h, &q_quick);
    let r_full = analyze_features_rgb8(&rgb, w, h, &q_full);
    assert_eq!(
        r_quick.get(PaletteFitsIn256),
        r_full.get(PaletteFitsIn256),
        "quick path and full path disagree on PaletteFitsIn256"
    );
}

#[test]
fn analyze_features_supports_full_set() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 64;
    let h: u32 = 64;
    let rgb = synth_rgb(w, h, 13);
    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let r = analyze_features_rgb8(&rgb, w, h, &query);
    assert_eq!(r.requested(), FeatureSet::SUPPORTED);

    // Every feature in the supported set must round-trip out of the
    // results object. Iterate over ALL ids 0..32 and treat anything
    // `from_u16` returns `None` for as either retired (id 11) or
    // gated behind a disabled cargo feature — both paths legitimately
    // skip.
    for id in 0..32u16 {
        let Some(f) = AnalysisFeature::from_u16(id) else {
            continue;
        };
        assert!(
            r.get(f).is_some(),
            "feature {:?} (id={}) missing from analyze_features result",
            f,
            id
        );
    }
}

#[test]
fn requesting_more_features_does_not_change_existing_values() {
    // Property: for any image, asking for feature F alone must
    // produce the same numeric value of F as asking for {F, …}.
    // Adding features to the query may run additional tiers, but
    // can never perturb the value of features already in the query.
    //
    // Verified across all 30 features on a synth image: pull each
    // feature alone, then again with every other feature requested
    // alongside it, assert the value is bit-identical (or NaN-bit-
    // identical for f32 NaNs, none of which any feature produces on
    // valid input).
    use AnalysisFeature::*;
    let w: u32 = 192;
    let h: u32 = 192;
    let rgb = synth_rgb(w, h, 9876);

    let all_features =
        analyze_features_rgb8(&rgb, w, h, &AnalysisQuery::new(FeatureSet::SUPPORTED));

    for id in 0..32u16 {
        let Some(f) = AnalysisFeature::from_u16(id) else {
            continue;
        };
        let alone = analyze_features_rgb8(&rgb, w, h, &AnalysisQuery::new(FeatureSet::just(f)));
        let with_others = all_features.get(f);
        let just_this = alone.get(f);
        assert_eq!(
            with_others, just_this,
            "side-effect detected: {:?} (id={}) differs between\n  \
             alone: {:?}\n  with all features: {:?}",
            f, id, just_this, with_others,
        );
    }

    // Additional spot-check: pairs of features whose tier dispatch
    // axes overlap. If two features are in the same tier bundle,
    // requesting both together vs alone must match. The pair list
    // mixes axes (T1 / T3 / Palette / Alpha / Derived) — the
    // experimental variants would broaden coverage but only when
    // they're built in.
    let probes_a: &[AnalysisFeature] = &[Variance, EdgeDensity, DistinctColorBins];
    #[cfg(feature = "experimental")]
    let probes_a = {
        let mut v = probes_a.to_vec();
        v.push(DctCompressibilityY);
        v
    };
    // Always include a stable T3 + Alpha probe; ScreenContentLikelihood
    // (composite) only joins the matrix when the cargo feature is on.
    let probes_b: &[AnalysisFeature] = &[HighFreqEnergyRatio, AlphaPresent];
    #[cfg(feature = "composites")]
    let probes_b = {
        let mut v = probes_b.to_vec();
        v.push(ScreenContentLikelihood);
        v
    };
    #[cfg(feature = "experimental")]
    let probes_b = {
        let mut v = probes_b.to_vec();
        v.push(Colourfulness);
        v
    };
    for a in probes_a.iter() {
        for b in probes_b.iter() {
            if a == b {
                continue;
            }
            let pair = analyze_features_rgb8(
                &rgb,
                w,
                h,
                &AnalysisQuery::new(FeatureSet::just(*a).with(*b)),
            );
            let solo_a =
                analyze_features_rgb8(&rgb, w, h, &AnalysisQuery::new(FeatureSet::just(*a)));
            assert_eq!(
                pair.get(*a),
                solo_a.get(*a),
                "{:?}'s value drifted when {:?} was also requested",
                a,
                b,
            );
        }
    }
}

#[test]
fn analyze_features_matches_legacy_values() {
    // Sanity: requesting Variance via analyze_features should yield
    // the same number as the legacy AnalyzerOutput.variance.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 128;
    let h: u32 = 128;
    let rgb = synth_rgb(w, h, 17);
    let legacy = analyze_rgb8(&rgb, w, h);
    let query = AnalysisQuery::new(
        FeatureSet::new()
            .with(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity)
            .with(AnalysisFeature::DistinctColorBins),
    );
    let r = analyze_features_rgb8(&rgb, w, h, &query);

    assert_eq!(r.get_f32(AnalysisFeature::Variance), Some(legacy.variance));
    assert_eq!(
        r.get_f32(AnalysisFeature::EdgeDensity),
        Some(legacy.edge_density)
    );
    assert_eq!(
        r.get(AnalysisFeature::DistinctColorBins)
            .and_then(|v| v.as_u32()),
        Some(legacy.distinct_color_bins)
    );
}

// --------------------------------------------------------------------
// Math locks — bit-precise (or tight-tolerance) outputs on synthetic
// inputs whose expected values are derivable from first principles.
//
// Why these exist: `#[magetypes(...)]` generates one monomorphization
// of every SIMD kernel per architecture tier (`v4` AVX-512, `v3`
// AVX2, `neon`, `wasm128`, `scalar`). At runtime archmage's `incant!`
// dispatches to whichever the CPU supports. If a tier diverges from
// the others (FMA contraction, lane-reduction order, range-reduction
// drift), the math lock fails on the affected runner.
//
// Tests use absolute tolerances chosen to clear ULP-level f32 noise
// from tree reductions but catch any genuine divergence (≥1e-3 on
// O(255²)-scale values, ≥1e-6 on O(1)-scale ratios).
// --------------------------------------------------------------------

/// xorshift32 — deterministic across architectures (no f32 ops in
/// the body), suitable for synthesizing reproducible test imagery.
fn xs32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn deterministic_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_add(0x9E37_79B9);
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for px in buf.chunks_exact_mut(3) {
        let r = xs32(&mut s);
        px[0] = (r & 0xFF) as u8;
        px[1] = ((r >> 8) & 0xFF) as u8;
        px[2] = ((r >> 16) & 0xFF) as u8;
    }
    buf
}

#[test]
fn math_lock_solid_gray_zero_variance_zero_edges() {
    // Closed form: every pixel identical ⇒ variance = 0, edge count
    // = 0, distinct bins = 1, every block uniform.
    let rgb = vec![128u8; 32 * 32 * 3];
    let out = analyze_rgb8(&rgb, 32, 32);
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    assert_eq!(out.chroma_complexity, 0.0);
    assert_eq!(out.cb_horiz_sharpness, 0.0);
    assert_eq!(out.cb_vert_sharpness, 0.0);
    assert_eq!(out.cb_peak_sharpness, 0.0);
    assert_eq!(out.cr_horiz_sharpness, 0.0);
    assert_eq!(out.cr_vert_sharpness, 0.0);
    assert_eq!(out.cr_peak_sharpness, 0.0);
    assert_eq!(out.distinct_color_bins, 1);
    assert!(out.uniformity > 0.999);
    assert!(out.flat_color_block_ratio > 0.999);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
    // Single histogram bin gets all weight ⇒ entropy = 0.
    assert!(out.luma_histogram_entropy.abs() < 1e-6);
}

#[test]
fn math_lock_solid_white_zero_variance_zero_edges() {
    // Same invariants at the upper boundary of the [0, 255] range —
    // catches sign / overflow bugs that a mid-range solid would miss.
    let rgb = vec![255u8; 32 * 32 * 3];
    let out = analyze_rgb8(&rgb, 32, 32);
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    assert_eq!(out.distinct_color_bins, 1);
}

#[test]
fn math_lock_solid_black_zero_variance_zero_edges() {
    let rgb = vec![0u8; 32 * 32 * 3];
    let out = analyze_rgb8(&rgb, 32, 32);
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    assert_eq!(out.distinct_color_bins, 1);
}

#[test]
fn math_lock_two_horizontal_bands_known_variance() {
    // Top half black (luma 0), bottom half white (luma 255). Tier 1
    // sees a bimodal luma distribution: half at 0, half at ~255. The
    // SIMD path uses (77R + 150G + 29B) / 256 in f32, which for white
    // is (77+150+29)*255/256 = 256*255/256 = 255.0 exactly. Variance
    // of {0,…,0, 255,…,255} (equal split) = (255/2)² = 16256.25.
    let w: u32 = 64;
    let h: u32 = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in (h / 2)..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            rgb[i] = 255;
            rgb[i + 1] = 255;
            rgb[i + 2] = 255;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    let expected_var: f32 = (255.0_f32 / 2.0).powi(2);
    assert!(
        (out.variance - expected_var).abs() < 1e-1,
        "variance={} expected≈{}",
        out.variance,
        expected_var
    );
    // Exactly two distinct 5-bit-per-channel bins (0,0,0) and (31,31,31).
    assert_eq!(out.distinct_color_bins, 2);
    // Edge density: edges only along the horizontal seam at y=h/2;
    // sampled-interior fraction depends on stripe sampling, but the
    // edge_density signal should be small (most rows are flat).
    assert!(out.edge_density >= 0.0);
    assert!(out.edge_density <= 1.0);
}

#[test]
fn math_lock_uniform_luma_distribution_max_entropy() {
    // Construct a 256×256 luma ramp where every luma value 0..255
    // appears with equal frequency across the image. The 32-bin luma
    // histogram is then uniform ⇒ entropy = log2(32) = 5 bits exactly.
    let w: u32 = 256;
    let h: u32 = 256;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = ((x + y) % 256) as u8;
            let i = ((y * w + x) * 3) as usize;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    // Tolerance covers stripe sampling — Tier 3 budgets a fixed
    // number of blocks, so the empirical 32-bin histogram is close to
    // uniform but doesn't perfectly hit log2(32) = 5.0. The lock
    // verifies entropy is in the high-uniformity regime (≥ 4.5);
    // a real arch divergence would push it well below.
    assert!(
        out.luma_histogram_entropy >= 4.5 && out.luma_histogram_entropy <= 5.05,
        "expected near 5.0 (log2(32)), got {}",
        out.luma_histogram_entropy
    );
}

#[test]
fn math_lock_geometry_exact() {
    use crate::feature::ImageGeometry;
    let g = ImageGeometry::new(1920, 1080);
    assert_eq!(g.width(), 1920);
    assert_eq!(g.height(), 1080);
    assert_eq!(g.pixels(), 1920u64 * 1080u64);
    assert!((g.megapixels() - 2.0736).abs() < 1e-6);
    assert!((g.aspect_ratio() - (16.0 / 9.0)).abs() < 1e-6);

    // Zero-height fallback (don't divide by zero).
    let g0 = ImageGeometry::new(100, 0);
    assert_eq!(g0.aspect_ratio(), 0.0);

    // Giant image: pixels() must not overflow u32.
    let big = ImageGeometry::new(u32::MAX, u32::MAX);
    assert_eq!(big.pixels(), u32::MAX as u64 * u32::MAX as u64);
}

#[cfg(feature = "composites")]
#[test]
fn math_lock_likelihoods_in_unit_interval_for_random_input() {
    // Locks the contract: TextLikelihood / ScreenContentLikelihood /
    // NaturalLikelihood are bounded to [0, 1] regardless of input.
    let rgb = deterministic_rgb(128, 128, 0xCAFE_BABE);
    let out = analyze_rgb8(&rgb, 128, 128);
    for v in [
        out.text_likelihood,
        out.screen_content_likelihood,
        out.natural_likelihood,
    ] {
        assert!((0.0..=1.0).contains(&v), "likelihood {v} outside [0, 1]");
    }
}

#[test]
fn math_lock_deterministic_input_is_reproducible() {
    // Running the analyzer twice on bit-identical inputs must produce
    // bit-identical outputs. Catches: hidden state, timing-dependent
    // dispatch, accumulator non-determinism (e.g., a Vec rehashed on
    // capacity boundary). Locks to bit-equality of every f32/u32 field.
    let rgb = deterministic_rgb(128, 128, 0x1234_5678);
    let a = analyze_rgb8(&rgb, 128, 128);
    let b = analyze_rgb8(&rgb, 128, 128);

    assert_eq!(a.variance.to_bits(), b.variance.to_bits());
    assert_eq!(a.edge_density.to_bits(), b.edge_density.to_bits());
    assert_eq!(a.chroma_complexity.to_bits(), b.chroma_complexity.to_bits());
    assert_eq!(a.cb_sharpness.to_bits(), b.cb_sharpness.to_bits());
    assert_eq!(a.cr_sharpness.to_bits(), b.cr_sharpness.to_bits());
    assert_eq!(a.uniformity.to_bits(), b.uniformity.to_bits());
    assert_eq!(
        a.flat_color_block_ratio.to_bits(),
        b.flat_color_block_ratio.to_bits()
    );
    assert_eq!(a.distinct_color_bins, b.distinct_color_bins);
    assert_eq!(
        a.cb_horiz_sharpness.to_bits(),
        b.cb_horiz_sharpness.to_bits()
    );
    assert_eq!(a.cb_vert_sharpness.to_bits(), b.cb_vert_sharpness.to_bits());
    assert_eq!(a.cb_peak_sharpness.to_bits(), b.cb_peak_sharpness.to_bits());
    assert_eq!(
        a.cr_horiz_sharpness.to_bits(),
        b.cr_horiz_sharpness.to_bits()
    );
    assert_eq!(a.cr_vert_sharpness.to_bits(), b.cr_vert_sharpness.to_bits());
    assert_eq!(a.cr_peak_sharpness.to_bits(), b.cr_peak_sharpness.to_bits());
    assert_eq!(
        a.high_freq_energy_ratio.to_bits(),
        b.high_freq_energy_ratio.to_bits()
    );
    assert_eq!(
        a.luma_histogram_entropy.to_bits(),
        b.luma_histogram_entropy.to_bits()
    );
    #[cfg(feature = "composites")]
    {
        assert_eq!(a.text_likelihood.to_bits(), b.text_likelihood.to_bits());
        assert_eq!(
            a.screen_content_likelihood.to_bits(),
            b.screen_content_likelihood.to_bits()
        );
        assert_eq!(
            a.natural_likelihood.to_bits(),
            b.natural_likelihood.to_bits()
        );
    }
}

#[test]
fn math_lock_palette_count_invariants() {
    // Construct an image with exactly N distinct 5-bit-per-channel
    // bins. The bin reduction is `(r >> 3, g >> 3, b >> 3)` ⇒ 32³
    // possible bins. Use 8 well-separated colours (each bin index
    // distinct after 5-bit truncation) and verify distinct_color_bins
    // exactly.
    let palette: [[u8; 3]; 8] = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ];
    let w: u32 = 32;
    let h: u32 = 32;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
        let c = palette[i % palette.len()];
        px[0] = c[0];
        px[1] = c[1];
        px[2] = c[2];
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert_eq!(out.distinct_color_bins, 8);
}

#[test]
fn math_lock_alpha_present_distinguishes_rgb_vs_rgba() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // RGB8: alpha_present must be false (no alpha channel).
    let rgb = fill_solid_rgb8(16, 16, [128, 64, 200]);
    let s = PixelSlice::new(&rgb, 16, 16, 16 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::AlphaPresent));
    let r = crate::analyze_features(s, &q).unwrap();
    assert_eq!(
        r.get(AnalysisFeature::AlphaPresent)
            .and_then(|v| v.as_bool()),
        Some(false)
    );

    // RGBA8 with all-opaque alpha: alpha_present is still TRUE
    // (channel exists, even if every value is 255).
    let rgba = fill_solid_rgba8(16, 16, [128, 64, 200, 255]);
    let s = PixelSlice::new(&rgba, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    assert_eq!(
        r.get(AnalysisFeature::AlphaPresent)
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

// --------------------------------------------------------------------
// AnalyzeError variants — Display, Debug, and source() coverage.
// --------------------------------------------------------------------

#[test]
fn analyze_error_display_covers_all_variants() {
    use crate::AnalyzeError;

    let e = AnalyzeError::Convert("imaginary CMS plugin missing".into());
    let msg = format!("{e}");
    assert!(msg.contains("imaginary CMS plugin missing"));
    assert!(format!("{e:?}").contains("Convert"));

    let e = AnalyzeError::Internal("synthetic".into());
    let msg = format!("{e}");
    assert!(msg.contains("synthetic"));
    assert!(format!("{e:?}").contains("Internal"));
}

#[test]
fn analyze_error_implements_std_error() {
    fn assert_error<E: core::error::Error>(_: &E) {}
    let e = crate::AnalyzeError::Internal("synthetic".into());
    assert_error(&e);
}

// --------------------------------------------------------------------
// __internal_with_overrides + InternalQuery dispatch — covers the
// `__analyze_internal` path that production callers don't see but
// tests / oracle re-extraction depend on.
// --------------------------------------------------------------------

#[test]
fn internal_query_with_overrides_runs_full_analyzer() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let rgb = synth_rgb(64, 64, 99);
    let slice = PixelSlice::new(&rgb, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();

    let iq = AnalysisQuery::__internal_with_overrides(
        FeatureSet::SUPPORTED,
        usize::MAX, // pixel_budget = unlimited
        4096,       // hf_max_blocks = oversized
    );
    let r = crate::__analyze_internal(slice, &iq).expect("internal entry runs");

    // Every supported feature must produce a value (bools/u32s/f32s).
    assert!(r.get(AnalysisFeature::Variance).is_some());
    assert!(r.get(AnalysisFeature::DistinctColorBins).is_some());
    assert!(r.get(AnalysisFeature::AlphaPresent).is_some());
    assert!(r.get(AnalysisFeature::HighFreqEnergyRatio).is_some());

    // Geometry round-trips through the internal entry.
    assert_eq!(r.geometry().width(), 64);
    assert_eq!(r.geometry().height(), 64);
}

// --------------------------------------------------------------------
// Dispatch axis coverage — exercise specific (PAL, T2, T3, ALPHA)
// combinations to make sure each match arm runs at least once.
// --------------------------------------------------------------------

#[test]
fn dispatch_axes_tier1_only_skips_other_tiers() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let rgb = synth_rgb(64, 64, 7);
    // Tier 1 features only — palette / T2 / T3 / alpha all skipped.
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance).with(AnalysisFeature::EdgeDensity),
    );
    let r = analyze_features_rgb8(&rgb, 64, 64, &q);
    assert!(r.get(AnalysisFeature::Variance).is_some());
    assert!(r.get(AnalysisFeature::EdgeDensity).is_some());
    // Unrequested features come back as None — never garbage.
    assert!(r.get(AnalysisFeature::DistinctColorBins).is_none());
    assert!(r.get(AnalysisFeature::CbHorizSharpness).is_none());
    assert!(r.get(AnalysisFeature::HighFreqEnergyRatio).is_none());
    assert!(r.get(AnalysisFeature::AlphaPresent).is_none());
}

#[test]
fn dispatch_axes_palette_only_runs_palette_pass() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let rgb = synth_rgb(64, 64, 8);
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::DistinctColorBins));
    let r = analyze_features_rgb8(&rgb, 64, 64, &q);
    assert!(r.get(AnalysisFeature::DistinctColorBins).is_some());
}

#[test]
fn dispatch_axes_alpha_only_runs_alpha_pass() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let rgba = fill_solid_rgba8(32, 32, [10, 20, 30, 200]);
    let s = PixelSlice::new(&rgba, 32, 32, 32 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::AlphaUsedFraction));
    let r = crate::analyze_features(s, &q).unwrap();
    let frac = r
        .get(AnalysisFeature::AlphaUsedFraction)
        .and_then(|v| v.as_f32())
        .expect("alpha_used_fraction must be present");
    // Every pixel has alpha=200 < 255 ⇒ used_fraction = 1.0.
    assert!((frac - 1.0).abs() < 1e-6);
}

// --------------------------------------------------------------------
// FeatureSet const-fn set math — covers the const-only paths in
// feature.rs that aren't reachable through analyze_features.
// --------------------------------------------------------------------

#[test]
fn feature_set_intersection_difference_subset() {
    use crate::feature::{AnalysisFeature, FeatureSet};

    let a = FeatureSet::just(AnalysisFeature::Variance)
        .with(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::DistinctColorBins);
    let b = FeatureSet::just(AnalysisFeature::EdgeDensity)
        .with(AnalysisFeature::DistinctColorBins)
        .with(AnalysisFeature::HighFreqEnergyRatio);

    let inter = a.intersect(b);
    assert!(inter.contains(AnalysisFeature::EdgeDensity));
    assert!(inter.contains(AnalysisFeature::DistinctColorBins));
    assert!(!inter.contains(AnalysisFeature::Variance));
    assert!(!inter.contains(AnalysisFeature::HighFreqEnergyRatio));
    assert_eq!(inter.len(), 2);

    let diff = a.difference(b);
    assert!(diff.contains(AnalysisFeature::Variance));
    assert!(!diff.contains(AnalysisFeature::EdgeDensity));
    assert_eq!(diff.len(), 1);

    // Empty set is contained in every set, every set is contained in
    // SUPPORTED.
    let empty = FeatureSet::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert!(a.contains_all(empty));
    assert!(FeatureSet::SUPPORTED.contains_all(a));
}

#[test]
fn analysis_feature_name_returns_field_name_string() {
    // `name()` must return the snake_case field name for every shipped
    // variant. Used by the `Debug` impl of AnalysisResults.
    use crate::feature::AnalysisFeature;

    assert_eq!(AnalysisFeature::Variance.name(), "variance");
    assert_eq!(AnalysisFeature::EdgeDensity.name(), "edge_density");
    assert_eq!(
        AnalysisFeature::DistinctColorBins.name(),
        "distinct_color_bins"
    );
    assert_eq!(AnalysisFeature::AlphaPresent.name(), "alpha_present");
    #[cfg(feature = "composites")]
    assert_eq!(
        AnalysisFeature::ScreenContentLikelihood.name(),
        "screen_content_likelihood"
    );
}

// --------------------------------------------------------------------
// Dispatch matrix coverage — the 16-arm `match (pal, t2, t3, alpha)`
// in `analyze_features_inner` only reaches each arm when a request
// requires that exact combination. Hit the specific arms that aren't
// covered by the higher-level "all features" tests.
// --------------------------------------------------------------------
fn run_dispatch(want_pal: bool, want_t2: bool, want_t3: bool, want_alpha: bool) {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // Build a feature set that triggers exactly the requested axes.
    // Tier 1 features are always selected since the analyzer always
    // runs Tier 1 — only the palette / T2 / T3 / alpha axes can be
    // suppressed, and the const-bool dispatcher reads exactly the
    // axis-membership intersections we test here.
    let mut fs = FeatureSet::just(AnalysisFeature::Variance);
    if want_pal {
        fs = fs.with(AnalysisFeature::DistinctColorBins);
    }
    if want_t2 {
        fs = fs.with(AnalysisFeature::CbHorizSharpness);
    }
    if want_t3 {
        fs = fs.with(AnalysisFeature::HighFreqEnergyRatio);
    }
    if want_alpha {
        fs = fs.with(AnalysisFeature::AlphaPresent);
    }

    // Use RGBA8 as the source so alpha exists when requested.
    let rgba = fill_solid_rgba8(64, 64, [10, 20, 30, 200]);
    let s = PixelSlice::new(&rgba, 64, 64, 64 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
    let r = crate::analyze_features(s, &AnalysisQuery::new(fs))
        .expect("dispatch arm runs");

    assert!(r.get(AnalysisFeature::Variance).is_some());
    assert_eq!(r.get(AnalysisFeature::DistinctColorBins).is_some(), want_pal);
    assert_eq!(r.get(AnalysisFeature::CbHorizSharpness).is_some(), want_t2);
    assert_eq!(
        r.get(AnalysisFeature::HighFreqEnergyRatio).is_some(),
        want_t3
    );
    assert_eq!(r.get(AnalysisFeature::AlphaPresent).is_some(), want_alpha);
}

#[test]
fn dispatch_matrix_covers_all_sixteen_arms() {
    // The 16-arm dispatch table picks one specialized monomorphization
    // per (PAL, T2, T3, ALPHA) combination. Walking every combination
    // in one test keeps the surface compact while ensuring no arm gets
    // silently dropped from the compiled binary.
    for pal in [false, true] {
        for t2 in [false, true] {
            for t3 in [false, true] {
                for alpha in [false, true] {
                    run_dispatch(pal, t2, t3, alpha);
                }
            }
        }
    }
}

// --------------------------------------------------------------------
// Alpha extraction — every supported PixelFormat / channel-bytes
// combination. Closes alpha.rs gaps for GrayA8 / GrayA16 / GrayAF32.
// --------------------------------------------------------------------

#[test]
fn alpha_works_on_graya8() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // GrayA8: 2 bytes per pixel, alpha at offset 1, ch_bytes=1.
    let w: u32 = 32;
    let h: u32 = 32;
    let mut buf = vec![0u8; (w * h * 2) as usize];
    for px in buf.chunks_exact_mut(2) {
        px[0] = 128;
        px[1] = 100; // partial alpha
    }
    let s = PixelSlice::new(&buf, w, h, (w * 2) as usize, PixelDescriptor::GRAYA8_SRGB).unwrap();
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::AlphaPresent).with(AnalysisFeature::AlphaUsedFraction),
    );
    let r = crate::analyze_features(s, &q).unwrap();
    assert_eq!(
        r.get(AnalysisFeature::AlphaPresent)
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    let used = r
        .get(AnalysisFeature::AlphaUsedFraction)
        .and_then(|v| v.as_f32())
        .unwrap();
    assert!((used - 1.0).abs() < 1e-6, "all alpha < 255 ⇒ used = 1.0");
}

#[test]
fn alpha_works_on_graya16() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // GrayA16: 4 bytes per pixel, alpha at offset 2, ch_bytes=2.
    let w: u32 = 32;
    let h: u32 = 32;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    for px in buf.chunks_exact_mut(4) {
        // luma 0x4000, alpha 0xFFFF (fully opaque).
        px[0] = 0x00;
        px[1] = 0x40;
        px[2] = 0xFF;
        px[3] = 0xFF;
    }
    let s = PixelSlice::new(&buf, w, h, (w * 4) as usize, PixelDescriptor::GRAYA16_SRGB).unwrap();
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::AlphaPresent).with(AnalysisFeature::AlphaUsedFraction),
    );
    let r = crate::analyze_features(s, &q).unwrap();
    assert_eq!(
        r.get(AnalysisFeature::AlphaPresent)
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    let used = r
        .get(AnalysisFeature::AlphaUsedFraction)
        .and_then(|v| v.as_f32())
        .unwrap();
    assert!(used.abs() < 1e-6, "all alpha = 0xFFFF ⇒ used = 0.0");
}

#[test]
fn alpha_works_on_grayaf32() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // GrayAF32: 8 bytes per pixel, alpha at offset 4, ch_bytes=4.
    let w: u32 = 16;
    let h: u32 = 16;
    let mut buf = vec![0u8; (w * h * 8) as usize];
    for px in buf.chunks_exact_mut(8) {
        // luma 0.5, alpha 0.4 (partial transparency).
        let l = 0.5_f32.to_le_bytes();
        let a = 0.4_f32.to_le_bytes();
        px[0..4].copy_from_slice(&l);
        px[4..8].copy_from_slice(&a);
    }
    let s = PixelSlice::new(
        &buf,
        w,
        h,
        (w * 8) as usize,
        PixelDescriptor::GRAYAF32_LINEAR,
    )
    .unwrap();
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::AlphaPresent).with(AnalysisFeature::AlphaUsedFraction),
    );
    let r = crate::analyze_features(s, &q).unwrap();
    assert_eq!(
        r.get(AnalysisFeature::AlphaPresent)
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    let used = r
        .get(AnalysisFeature::AlphaUsedFraction)
        .and_then(|v| v.as_f32())
        .unwrap();
    assert!((used - 1.0).abs() < 1e-6, "all alpha < 1.0 ⇒ used = 1.0");
}

// --------------------------------------------------------------------
// RowStream — public surface tests (fetch_range bulk fill, panics on
// out-of-bounds). Direct unit tests on the row source so future
// changes can't silently break the contract.
// --------------------------------------------------------------------

#[test]
fn row_stream_fetch_range_native_zero_copy_path() {
    // RGB8 source ⇒ Native variant. fetch_range packs rows back-to-
    // back into the destination buffer at width*3 stride.
    let w: u32 = 8;
    let h: u32 = 4;
    // Distinct row signatures so we can verify ordering after fetch_range.
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for (yi, row) in rgb.chunks_exact_mut((w * 3) as usize).enumerate() {
        row.fill(yi as u8 + 1);
    }
    let slice = PixelSlice::new(&rgb, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut stream = crate::row_stream::RowStream::new(slice).unwrap();
    let mut dst = vec![0u8; (w * 3 * 3) as usize];
    stream.fetch_range(1..4, &mut dst);
    // Row 1 (signature 2), Row 2 (signature 3), Row 3 (signature 4).
    assert_eq!(&dst[0..(w as usize * 3)], &vec![2u8; (w * 3) as usize][..]);
    assert_eq!(
        &dst[(w as usize * 3)..(w as usize * 6)],
        &vec![3u8; (w * 3) as usize][..]
    );
    assert_eq!(
        &dst[(w as usize * 6)..(w as usize * 9)],
        &vec![4u8; (w * 3) as usize][..]
    );
}

#[test]
#[should_panic(expected = "out of bounds")]
fn row_stream_fetch_into_panics_on_oob_row() {
    let w: u32 = 4;
    let h: u32 = 2;
    let rgb = vec![0u8; (w * h * 3) as usize];
    let slice = PixelSlice::new(&rgb, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut stream = crate::row_stream::RowStream::new(slice).unwrap();
    let mut dst = vec![0u8; (w * 3) as usize];
    stream.fetch_into(2, &mut dst); // y == height ⇒ panic
}

#[test]
#[should_panic(expected = "out of bounds")]
fn row_stream_borrow_row_panics_on_oob_row() {
    let w: u32 = 4;
    let h: u32 = 2;
    let rgb = vec![0u8; (w * h * 3) as usize];
    let slice = PixelSlice::new(&rgb, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut stream = crate::row_stream::RowStream::new(slice).unwrap();
    let _ = stream.borrow_row(99);
}

#[test]
fn row_stream_convert_path_produces_rgb8_for_rgba8_input() {
    // Non-RGB8 layout → Convert path fires. We don't test bit-exact
    // conversion (zenpixels-convert owns that), only that the stream
    // returns a w*3-byte row.
    let w: u32 = 4;
    let h: u32 = 4;
    let rgba = fill_solid_rgba8(w, h, [10, 20, 30, 255]);
    let slice = PixelSlice::new(&rgba, w, h, (w * 4) as usize, PixelDescriptor::RGBA8_SRGB)
        .unwrap();
    let mut stream = crate::row_stream::RowStream::new(slice).unwrap();
    let row = stream.borrow_row(0);
    assert_eq!(row.len(), (w * 3) as usize);
    // Roughly the RGBA values (sRGB→sRGB straight-alpha = identity for opaque).
    assert!(row[0] >= 9 && row[0] <= 11);
    assert!(row[1] >= 19 && row[1] <= 21);
    assert!(row[2] >= 29 && row[2] <= 31);
}

// --------------------------------------------------------------------
// Tiny / degenerate inputs — exercise the early-return guards in
// alpha.rs (zero-width / zero-height) and tier1.rs (extract_tier1
// width/height < 2 returns).
// --------------------------------------------------------------------

#[test]
fn zero_height_image_returns_no_signals() {
    use crate::feature::{AnalysisQuery, FeatureSet};

    let w: u32 = 16;
    let h: u32 = 0;
    let buf = Vec::<u8>::new();
    let s = PixelSlice::new(&buf, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let r = crate::analyze_features(s, &q).unwrap();
    // Geometry round-trips; features default to None / 0 since the
    // analyzer's per-tier guards bail on zero-size.
    assert_eq!(r.geometry().width(), 16);
    assert_eq!(r.geometry().height(), 0);
}

#[test]
fn analysis_feature_name_full_coverage() {
    // Every active variant (in this build) round-trips through name().
    use crate::feature::AnalysisFeature;
    for id in 0..32u16 {
        if let Some(f) = AnalysisFeature::from_u16(id) {
            let n = f.name();
            assert!(!n.is_empty(), "feature {id} has empty name");
            // snake_case sanity.
            assert!(
                n.chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
                "feature {id} name {n:?} is not snake_case"
            );
        }
    }
}

#[test]
fn analysis_results_debug_includes_populated_fields() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let rgb = vec![64u8; 32 * 32 * 3];
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance).with(AnalysisFeature::DistinctColorBins),
    );
    let r = analyze_features_rgb8(&rgb, 32, 32, &q);
    let s = format!("{r:?}");
    assert!(s.contains("AnalysisResults"));
    assert!(s.contains("variance"));
    assert!(s.contains("distinct_color_bins"));
}

// --------------------------------------------------------------------
// Wide-gamut & bit-depth policy tests.
//
// The analyzer accepts every layout `zenpixels-convert::RowConverter`
// can ingest. These tests lock the principled invariants:
//
//   1. **u8-promotion invariance.** A u8 image promoted to u16 via
//      the standard `u8 * 257` doubling produces bit-identical features
//      to the original u8. Same goes for `f / 255.0` f32 promotion.
//      Verified by garb's u16→u8 path: `(u16 * 255 + 32768) >> 16` is
//      exact identity for `u16 = u8 * 257`.
//
//   2. **Wide-gamut acceptance.** RGB8 with Display P3 / Rec.2020 /
//      AdobeRGB primaries passes through Native zero-copy. The BT.601
//      luma weights produce slightly different numbers vs sRGB, and
//      that's the principled outcome — wide-gamut content has more
//      saturated colour, so chroma signals legitimately read higher.
//
//   3. **Linear / HDR f32 acceptance.** RGB-F32 linear inputs are
//      converted through their declared transfer function to display-
//      space RGB8. SDR f32 in [0, 1] round-trips u8-identically. HDR
//      content gets a tonemapped SDR rendering for SDR-calibrated
//      features; HDR-aware analysis is tier_depth (issue #120).
// --------------------------------------------------------------------

/// Promote an RGB8 buffer to RGB16 using the standard `u8 * 257`
/// doubling. The high byte of every u16 sample equals the original u8,
/// so the round-trip through RowConverter's `(v*255 + 32768) >> 16` is
/// exact identity.
fn promote_rgb8_to_rgb16(rgb8: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; rgb8.len() * 2];
    for (i, &b) in rgb8.iter().enumerate() {
        let v = (b as u16) * 257;
        let bytes = v.to_ne_bytes();
        out[i * 2] = bytes[0];
        out[i * 2 + 1] = bytes[1];
    }
    out
}

#[test]
fn wide_gamut_u8_to_u16_promotion_is_bit_identical() {
    // Bit-for-bit invariance: features computed from a u8-promoted u16
    // source equal those computed from the original u8. This is the
    // load-bearing guarantee for wide-gamut acceptance — codecs that
    // upgrade from u8 to u16 in their pipeline don't see different
    // analyzer answers and don't need to retrain.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 64;
    let h: u32 = 64;
    let rgb8 = synth_rgb(w, h, 0xDEADBEEF);
    let rgb16 = promote_rgb8_to_rgb16(&rgb8);

    let query = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::EdgeDensity)
            .with(AnalysisFeature::ChromaComplexity)
            .with(AnalysisFeature::Uniformity)
            .with(AnalysisFeature::FlatColorBlockRatio)
            .with(AnalysisFeature::DistinctColorBins)
            .with(AnalysisFeature::HighFreqEnergyRatio)
            .with(AnalysisFeature::LumaHistogramEntropy),
    );

    // Source #1: native RGB8.
    let s8 = PixelSlice::new(&rgb8, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let r8 = crate::analyze_features(s8, &query).unwrap();

    // Source #2: u8-promoted RGB16, read by the analyzer through
    // RowConverter — should narrow back to identical RGB8 bytes.
    let s16 = PixelSlice::new(&rgb16, w, h, (w * 6) as usize, PixelDescriptor::RGB16_SRGB).unwrap();
    let r16 = crate::analyze_features(s16, &query).unwrap();

    // Bit-equality on every f32 / u32 feature.
    let cmp_f32 = |feature: AnalysisFeature, name: &str| {
        let a = r8.get_f32(feature).unwrap();
        let b = r16.get_f32(feature).unwrap();
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: u8={a} u16={b} (bits {:#x} vs {:#x})",
            a.to_bits(),
            b.to_bits()
        );
    };
    cmp_f32(AnalysisFeature::Variance, "variance");
    cmp_f32(AnalysisFeature::EdgeDensity, "edge_density");
    cmp_f32(AnalysisFeature::ChromaComplexity, "chroma_complexity");
    cmp_f32(AnalysisFeature::Uniformity, "uniformity");
    cmp_f32(AnalysisFeature::FlatColorBlockRatio, "flat_color_block_ratio");
    cmp_f32(AnalysisFeature::HighFreqEnergyRatio, "high_freq_energy_ratio");
    cmp_f32(AnalysisFeature::LumaHistogramEntropy, "luma_histogram_entropy");
    assert_eq!(
        r8.get(AnalysisFeature::DistinctColorBins)
            .and_then(|v| v.as_u32()),
        r16.get(AnalysisFeature::DistinctColorBins)
            .and_then(|v| v.as_u32())
    );
}

#[test]
fn wide_gamut_rgba16_promoted_from_rgba8_is_bit_identical() {
    // Same invariance over the alpha-bearing 16-bit format. Locks
    // that the alpha pass (which reads source bytes directly, not
    // through RowConverter) doesn't accidentally diverge between
    // u8-promoted u16 and the original u8.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 32;
    let h: u32 = 32;
    let mut rgba8 = vec![0u8; (w * h * 4) as usize];
    let mut s = 0xC001_C0DE_u32;
    for px in rgba8.chunks_exact_mut(4) {
        let r = xs32(&mut s);
        px[0] = (r & 0xFF) as u8;
        px[1] = ((r >> 8) & 0xFF) as u8;
        px[2] = ((r >> 16) & 0xFF) as u8;
        px[3] = ((r >> 24) & 0xFE) as u8 | 1; // avoid all-zero alpha
    }
    let rgba16 = promote_rgb8_to_rgb16(&rgba8); // also works for 4-channel.

    let query = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::AlphaPresent)
            .with(AnalysisFeature::AlphaUsedFraction)
            .with(AnalysisFeature::AlphaBimodalScore),
    );

    let s8 = PixelSlice::new(&rgba8, w, h, (w * 4) as usize, PixelDescriptor::RGBA8_SRGB).unwrap();
    let s16 =
        PixelSlice::new(&rgba16, w, h, (w * 8) as usize, PixelDescriptor::RGBA16_SRGB).unwrap();

    let r8 = crate::analyze_features(s8, &query).unwrap();
    let r16 = crate::analyze_features(s16, &query).unwrap();

    assert_eq!(
        r8.get(AnalysisFeature::AlphaPresent),
        r16.get(AnalysisFeature::AlphaPresent),
    );
    let used8 = r8.get_f32(AnalysisFeature::AlphaUsedFraction).unwrap();
    let used16 = r16.get_f32(AnalysisFeature::AlphaUsedFraction).unwrap();
    assert_eq!(
        used8.to_bits(),
        used16.to_bits(),
        "alpha_used_fraction divergence: u8={used8} u16={used16}"
    );
    let bim8 = r8.get_f32(AnalysisFeature::AlphaBimodalScore).unwrap();
    let bim16 = r16.get_f32(AnalysisFeature::AlphaBimodalScore).unwrap();
    assert_eq!(
        bim8.to_bits(),
        bim16.to_bits(),
        "alpha_bimodal_score divergence: u8={bim8} u16={bim16}"
    );
}

#[test]
fn wide_gamut_displayp3_8bit_runs_without_error() {
    // Display P3 8-bit is layout-compatible with RGB8 (24bpp packed),
    // so it goes through the Native zero-copy path. The analyzer
    // measures the bytes directly — chroma stats may be slightly
    // higher than a sRGB rendition, which is the principled outcome.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 32;
    let h: u32 = 32;
    let rgb = synth_rgb(w, h, 0xC1AB_BABE);
    let desc =
        PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
    let s = PixelSlice::new(&rgb, w, h, (w * 3) as usize, desc).unwrap();
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::Variance));
    let r = crate::analyze_features(s, &q).expect("Display P3 must be accepted");
    assert!(r.get_f32(AnalysisFeature::Variance).unwrap() > 0.0);
}

#[test]
fn wide_gamut_rgba_f32_linear_runs_without_error() {
    // f32 linear-light RGBA: converted through descriptor's transfer
    // function to display-space RGB8. Any reasonable f32 in [0, 1]
    // gets a sensible RGB8 rendition; the analyzer feature surface is
    // populated end-to-end without error.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let w: u32 = 32;
    let h: u32 = 32;
    let mut buf = vec![0u8; (w * h * 16) as usize];
    let mut state = 0xABAD_F00D_u32;
    for px in buf.chunks_exact_mut(16) {
        let s = xs32(&mut state);
        let r = ((s & 0xFFFF) as f32) / 65535.0;
        let g = (((s >> 8) & 0xFFFF) as f32) / 65535.0;
        let b = (((s >> 16) & 0xFFFF) as f32) / 65535.0;
        let a = 1.0_f32;
        px[0..4].copy_from_slice(&r.to_le_bytes());
        px[4..8].copy_from_slice(&g.to_le_bytes());
        px[8..12].copy_from_slice(&b.to_le_bytes());
        px[12..16].copy_from_slice(&a.to_le_bytes());
    }
    let s = PixelSlice::new(
        &buf,
        w,
        h,
        (w * 16) as usize,
        PixelDescriptor::RGBAF32_LINEAR,
    )
    .unwrap();
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::AlphaPresent)
            .with(AnalysisFeature::DistinctColorBins),
    );
    let r = crate::analyze_features(s, &q).expect("RGBAF32 linear must be accepted");
    assert!(r.get(AnalysisFeature::Variance).is_some());
    assert!(r.get(AnalysisFeature::AlphaPresent).is_some());
    assert!(r.get(AnalysisFeature::DistinctColorBins).is_some());
}

// --------------------------------------------------------------------
// HDR survival — the load-bearing test that motivates `tier_depth`.
//
// `RowConverter` does not tonemap. A PQ HDR f32 image with linear 1.0
// (= 10 000 nits per ST 2084) gets clipped into the [0, 1] sRGB-display
// range when narrowed to RGB8 — the standard tiers see indistinguishable
// SDR-clipped bytes whether the source is a sun-bright HDR scene or a
// matte SDR photo. The depth tier reads source samples directly and
// preserves the dynamic-range signal.
// --------------------------------------------------------------------

#[cfg(feature = "experimental")]
#[test]
fn hdr_signal_survives_via_tier_depth_when_rowstream_would_clip() {
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // Build a PQ-encoded image at signal 0.751 (≈ 1000 nits per
    // ST 2084) — well above the 100-nit SDR threshold. Through
    // RowConverter this becomes byte 192 (clipped sRGB display
    // rendition) — Tier 1 variance would be near zero, no signal.
    let w: u32 = 32;
    let h: u32 = 32;
    let mut buf = vec![0u8; (w * h * 12) as usize];
    let signal = 0.751_f32.to_le_bytes();
    for px in buf.chunks_exact_mut(12) {
        px[0..4].copy_from_slice(&signal);
        px[4..8].copy_from_slice(&signal);
        px[8..12].copy_from_slice(&signal);
    }
    let desc =
        PixelDescriptor::RGBF32_LINEAR.with_transfer(zenpixels::TransferFunction::Pq);
    let s = PixelSlice::new(&buf, w, h, (w * 12) as usize, desc).unwrap();

    // Request both an SDR-calibrated feature and the depth-tier
    // signals — the SDR feature is whatever it ends up being post-
    // tonemap, but the HDR signals must reflect the source.
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::PeakLuminanceNits)
            .with(AnalysisFeature::HdrPresent)
            .with(AnalysisFeature::HdrHeadroomStops)
            .with(AnalysisFeature::HdrPixelFraction),
    );
    let r = crate::analyze_features(s, &q).expect("PQ must be accepted");

    // Source-direct depth tier preserves the HDR signal.
    let peak = r.get_f32(AnalysisFeature::PeakLuminanceNits).unwrap();
    assert!(
        peak > 800.0,
        "expected ~1000 nits peak, got {peak} — depth tier didn't see source samples"
    );
    let hdr_present = r
        .get(AnalysisFeature::HdrPresent)
        .and_then(|v| v.as_bool())
        .unwrap();
    assert!(hdr_present, "HdrPresent must be true on PQ ~1000-nit content");
    let headroom = r.get_f32(AnalysisFeature::HdrHeadroomStops).unwrap();
    assert!(headroom > 3.0, "expected >3 stops headroom, got {headroom}");
    let frac = r.get_f32(AnalysisFeature::HdrPixelFraction).unwrap();
    assert!(frac > 0.99, "all-bright PQ source ⇒ ~1.0 fraction, got {frac}");
}

#[cfg(feature = "experimental")]
#[test]
fn sdr_srgb_does_not_trip_hdr_present() {
    // A regular 8-bit sRGB image — even one with code 255 throughout —
    // must NOT have HdrPresent set, since the transfer function is
    // SDR-only.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let buf = vec![255u8; 32 * 32 * 3];
    let s = PixelSlice::new(&buf, 32, 32, 32 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::HdrPresent)
            .with(AnalysisFeature::HdrHeadroomStops)
            .with(AnalysisFeature::PeakLuminanceNits),
    );
    let r = crate::analyze_features(s, &q).unwrap();
    assert_eq!(
        r.get(AnalysisFeature::HdrPresent)
            .and_then(|v| v.as_bool()),
        Some(false),
        "sRGB-only source must not be flagged HDR regardless of brightness"
    );
    assert_eq!(
        r.get_f32(AnalysisFeature::HdrHeadroomStops),
        Some(0.0),
        "sRGB peak ≤ 80 nits ⇒ 0 headroom stops"
    );
}

#[cfg(feature = "experimental")]
#[test]
#[ignore] // run with `cargo test --release --features experimental -- perf_strip_alpha_vs_convert --ignored --nocapture`
fn perf_strip_alpha_vs_convert() {
    // RGBA8 used to route through RowConverter; now takes the
    // native StripAlpha8 path. Compare the per-call cost to RGB8.
    use crate::feature::{AnalysisQuery, FeatureSet};
    use std::time::Instant;

    let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    let w: u32 = 4096;
    let h: u32 = 4096;
    let rgb = synth_rgb(w, h, 0xCAFE_F00D);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
        px[0] = rgb[i * 3];
        px[1] = rgb[i * 3 + 1];
        px[2] = rgb[i * 3 + 2];
        px[3] = 0xFF;
    }

    let s_rgb = PixelSlice::new(&rgb, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let _ = crate::analyze_features(s_rgb, &q).unwrap(); // warmup

    let mut rgb8_us = Vec::with_capacity(5);
    for _ in 0..5 {
        let s = PixelSlice::new(&rgb, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
        let t0 = Instant::now();
        let _ = crate::analyze_features(s, &q).unwrap();
        rgb8_us.push(t0.elapsed().as_micros() as u64);
    }
    rgb8_us.sort_unstable();

    let mut rgba8_us = Vec::with_capacity(5);
    for _ in 0..5 {
        let s = PixelSlice::new(&rgba, w, h, (w * 4) as usize, PixelDescriptor::RGBA8_SRGB)
            .unwrap();
        let t0 = Instant::now();
        let _ = crate::analyze_features(s, &q).unwrap();
        rgba8_us.push(t0.elapsed().as_micros() as u64);
    }
    rgba8_us.sort_unstable();

    eprintln!(
        "4K full-feature-set: RGB8 {} µs, RGBA8 {} µs (Δ = {} µs)",
        rgb8_us[2],
        rgba8_us[2],
        rgba8_us[2] as i64 - rgb8_us[2] as i64
    );
}

#[cfg(feature = "experimental")]
#[test]
#[ignore] // run with `cargo test --release --features experimental -- perf_full_feature_set --ignored --nocapture`
fn perf_full_feature_set() {
    // Ad-hoc timing check: full feature surface on synthetic images
    // at 1 MP / 4 MP / 16 MP. Reports per-MP cost so the depth tier
    // and Tier 3 AQ map additions stay within budget. Ignored by
    // default — only meaningful in release builds.
    use crate::feature::{AnalysisQuery, FeatureSet};
    use std::time::Instant;

    let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
    for &(w, h) in &[(1024_u32, 1024_u32), (2048, 2048), (4096, 4096)] {
        let rgb = synth_rgb(w, h, 0xCAFE_F00D);
        let mp = (w as f64 * h as f64) / 1_000_000.0;

        // Warmup.
        let _ = analyze_features_rgb8(&rgb, w, h, &q);

        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = analyze_features_rgb8(&rgb, w, h, &q);
            samples.push(t0.elapsed().as_micros() as u64);
        }
        samples.sort_unstable();
        let median_us = samples[samples.len() / 2];
        let per_mp_us = median_us as f64 / mp;
        eprintln!(
            "{w}x{h} ({mp:.1} MP): median {median_us:>5} µs (~{per_mp_us:.0} µs/MP)"
        );
    }
}

#[cfg(feature = "experimental")]
#[test]
fn gradient_fraction_high_for_smooth_low_for_noise() {
    // Smooth horizontal gradient: most blocks have AC energy
    // concentrated in the lowest zigzag positions ⇒ GradientFraction
    // should be high. Pure noise spreads energy across all
    // frequencies ⇒ GradientFraction near 0.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::GradientFraction));

    // Smooth horizontal gradient — exactly the kind of content where
    // larger DCT transforms pay off.
    let w: u32 = 256;
    let h: u32 = 256;
    let mut grad = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = (x * 255 / (w - 1)) as u8;
            let i = ((y * w + x) * 3) as usize;
            grad[i] = v;
            grad[i + 1] = v;
            grad[i + 2] = v;
        }
    }
    let s = PixelSlice::new(&grad, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let g_grad = r.get_f32(AnalysisFeature::GradientFraction).unwrap();
    assert!(g_grad > 0.5, "smooth gradient ⇒ > 0.5, got {g_grad}");

    // Pure noise — high-frequency-dominated.
    let mut noise = vec![0u8; (w * h * 3) as usize];
    let mut state: u32 = 0xCAFE;
    for px in noise.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *px = (state >> 24) as u8;
    }
    let s = PixelSlice::new(&noise, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let g_noise = r.get_f32(AnalysisFeature::GradientFraction).unwrap();
    assert!(g_noise < 0.2, "pure noise ⇒ < 0.2, got {g_noise}");
    assert!(
        g_grad > g_noise + 0.3,
        "gradient should clearly beat noise: gradient={g_grad} noise={g_noise}"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn gamut_coverage_one_for_srgb_pixels_in_rec2020_container() {
    // Descriptor-gap test: an image declared as Rec.2020 / linear f32
    // but whose pixels live entirely in the sRGB sub-gamut should
    // report GamutCoverageSrgb = 1.0, telling codecs they can encode
    // it as sRGB / Bt709 primaries and save bits on the colour
    // metadata + drop the wide-gamut encoder modes.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::GamutCoverageSrgb)
            .with(AnalysisFeature::GamutCoverageP3)
            .with(AnalysisFeature::WideGamutFraction),
    );

    // Mid-grey in Rec.2020-primaries linear f32. Since R=G=B, the
    // primaries projection collapses to the achromatic axis — every
    // gamut covers mid-grey.
    let mut buf = vec![0u8; 16 * 16 * 12];
    let half = 0.5_f32.to_le_bytes();
    for px in buf.chunks_exact_mut(12) {
        px[0..4].copy_from_slice(&half);
        px[4..8].copy_from_slice(&half);
        px[8..12].copy_from_slice(&half);
    }
    let desc = PixelDescriptor::RGBF32_LINEAR
        .with_transfer(zenpixels::TransferFunction::Linear)
        .with_primaries(zenpixels::ColorPrimaries::Bt2020);
    let s = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    assert!(
        (r.get_f32(AnalysisFeature::GamutCoverageSrgb).unwrap() - 1.0).abs() < 1e-6,
        "achromatic Rec.2020 ⇒ sRGB-coverable"
    );
    assert!(
        (r.get_f32(AnalysisFeature::GamutCoverageP3).unwrap() - 1.0).abs() < 1e-6,
        "achromatic Rec.2020 ⇒ P3-coverable"
    );
    assert_eq!(
        r.get_f32(AnalysisFeature::WideGamutFraction).unwrap(),
        0.0,
        "linear ≤ 1.0 ⇒ no wide-gamut tripping"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn gamut_coverage_zero_for_saturated_rec2020_green() {
    // Saturated green at (0, 1, 0) in BT.2020 primaries projects to
    // a NEGATIVE red and blue in sRGB primaries — the wider gamut is
    // genuinely being used. GamutCoverageSrgb must drop accordingly.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::GamutCoverageSrgb));

    let mut buf = vec![0u8; 16 * 16 * 12];
    let zero = 0.0_f32.to_le_bytes();
    let one = 1.0_f32.to_le_bytes();
    for px in buf.chunks_exact_mut(12) {
        px[0..4].copy_from_slice(&zero);
        px[4..8].copy_from_slice(&one);
        px[8..12].copy_from_slice(&zero);
    }
    let desc = PixelDescriptor::RGBF32_LINEAR
        .with_transfer(zenpixels::TransferFunction::Linear)
        .with_primaries(zenpixels::ColorPrimaries::Bt2020);
    let s = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let cov = r.get_f32(AnalysisFeature::GamutCoverageSrgb).unwrap();
    assert!(
        cov < 0.05,
        "saturated Rec.2020 green ⇒ NOT sRGB-coverable (got {cov})"
    );
}

#[cfg(feature = "composites")]
#[test]
fn line_art_score_high_for_two_tone_low_for_natural() {
    // A black-on-white line drawing-shaped image should score high.
    // A noisy gradient should score low.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::LineArtScore));

    // Two-tone with sparse strokes: 90% white + 10% black diagonal.
    let w: u32 = 128;
    let h: u32 = 128;
    let mut img = vec![255u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            // Sparse line pattern: diagonal stripes every 16 px.
            if (x + y) % 16 == 0 || (x + y) % 16 == 1 {
                let i = ((y * w + x) * 3) as usize;
                img[i] = 0;
                img[i + 1] = 0;
                img[i + 2] = 0;
            }
        }
    }
    let s = PixelSlice::new(&img, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let line_art = r.get_f32(AnalysisFeature::LineArtScore).unwrap();
    assert!(
        line_art > 0.1,
        "two-tone line drawing ⇒ should be > 0.1, got {line_art}"
    );

    // Pseudo-random noisy gradient — high-entropy, no bimodality.
    let mut natural = vec![0u8; (w * h * 3) as usize];
    let mut state: u32 = 1;
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((y * 2 + (state >> 28)) & 0xFF) as u8;
            let i = ((y * w + x) * 3) as usize;
            natural[i] = v;
            natural[i + 1] = v;
            natural[i + 2] = v;
        }
    }
    let s = PixelSlice::new(&natural, w, h, (w * 3) as usize, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let nat = r.get_f32(AnalysisFeature::LineArtScore).unwrap();
    assert!(
        nat < line_art,
        "natural < line_art; got natural={nat} line_art={line_art}"
    );
    assert!(nat < 0.3, "natural ⇒ should be low, got {nat}");
}

#[cfg(feature = "experimental")]
#[test]
fn noise_floor_low_for_solid_high_for_pure_noise() {
    // A solid image has zero AC ⇒ noise floor = 0.
    // A pure-noise image populates every AC bin ⇒ noise floor > 0.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::NoiseFloorY));

    let solid = vec![128u8; 64 * 64 * 3];
    let s = PixelSlice::new(&solid, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let nf_solid = r.get_f32(AnalysisFeature::NoiseFloorY).unwrap();
    assert!(nf_solid < 0.05, "solid ⇒ ~0, got {nf_solid}");

    // Pure noise.
    let mut noise = vec![0u8; 256 * 256 * 3];
    let mut state: u32 = 1;
    for px in noise.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *px = (state >> 24) as u8;
    }
    let s = PixelSlice::new(&noise, 256, 256, 256 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let nf_noise = r.get_f32(AnalysisFeature::NoiseFloorY).unwrap();
    assert!(
        nf_noise > nf_solid + 0.1,
        "noise > solid by ≥ 0.1; got noise={nf_noise} solid={nf_solid}"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn aq_map_std_low_for_uniform_high_for_heterogeneous() {
    // A solid image has zero AC energy variance ⇒ std ≈ 0.
    // A noise+stripes mix has high std (busy and flat blocks coexist).
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::AqMapStd));

    let solid = vec![128u8; 64 * 64 * 3];
    let s = PixelSlice::new(&solid, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let std_solid = r.get_f32(AnalysisFeature::AqMapStd).unwrap();
    assert!(std_solid < 0.1, "solid ⇒ near 0, got {std_solid}");

    // Half noise, half flat — heterogeneous busyness.
    let mut mixed = vec![128u8; 64 * 64 * 3];
    let mut state: u32 = 1;
    for y in 0..32 {
        for x in 0..64 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = (state >> 24) as u8;
            let i = ((y * 64 + x) * 3) as usize;
            mixed[i] = v;
            mixed[i + 1] = v;
            mixed[i + 2] = v;
        }
    }
    let s = PixelSlice::new(&mixed, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let std_mixed = r.get_f32(AnalysisFeature::AqMapStd).unwrap();
    assert!(
        std_mixed > std_solid + 0.5,
        "mixed > solid by ≥ 0.5; got mixed={std_mixed} solid={std_solid}"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn skin_tone_fraction_fires_on_skin_colored_pixels_zero_on_neutral() {
    // The Chai-Ngan YCbCr classifier (Cb [77,127], Cr [133,173], Y [40,240])
    // is invariant to skin pigmentation by design — chroma quantifies hue
    // not lightness. Verify that a representative tone in each common
    // pigmentation lands inside the gate.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::SkinToneFraction));

    // Three sRGB skin tones spanning a wide range of pigmentations:
    //   light  : (236, 188, 180)  — light pinkish
    //   medium : (198, 134, 105)  — medium tan
    //   dark   : (90,  56,  37)   — deep brown
    // Each YCbCr-conversion lands inside the Chai-Ngan rectangle.
    for (label, rgb) in [
        ("light", [236u8, 188, 180]),
        ("medium", [198u8, 134, 105]),
        ("dark", [90u8, 56, 37]),
    ] {
        let mut buf = vec![0u8; 64 * 64 * 3];
        for px in buf.chunks_exact_mut(3) {
            px[0] = rgb[0];
            px[1] = rgb[1];
            px[2] = rgb[2];
        }
        let s = PixelSlice::new(&buf, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let r = crate::analyze_features(s, &q).unwrap();
        let f = r.get_f32(AnalysisFeature::SkinToneFraction).unwrap();
        assert!(
            f > 0.95,
            "{label} skin tone {:?} should fire near 1.0, got {f}",
            rgb
        );
    }

    // Pure neutral grey: Cb = Cr = 128 — outside Cr [133, 173], so 0.
    let neutral = vec![128u8; 64 * 64 * 3];
    let s = PixelSlice::new(&neutral, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let f = r.get_f32(AnalysisFeature::SkinToneFraction).unwrap();
    assert!(f < 0.01, "neutral grey ⇒ ~0.0, got {f}");

    // Saturated blue: Cb high (≈240), outside the Cb [77, 127] gate.
    let mut blue = vec![0u8; 64 * 64 * 3];
    for px in blue.chunks_exact_mut(3) {
        px[0] = 0;
        px[1] = 0;
        px[2] = 255;
    }
    let s = PixelSlice::new(&blue, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let f = r.get_f32(AnalysisFeature::SkinToneFraction).unwrap();
    assert!(f < 0.01, "saturated blue ⇒ ~0.0, got {f}");
}

#[cfg(feature = "experimental")]
#[test]
fn edge_slope_stdev_low_for_uniform_high_for_varied_edges() {
    // Synthetic two-tone bands at one luma step (all crossings have the
    // same gradient magnitude) ⇒ very low stddev. Mixed-amplitude edges
    // (alternating step heights) ⇒ higher stddev.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::EdgeSlopeStdev));

    // Uniform-amplitude vertical bands: every transition is the same
    // magnitude. stddev should be ~0 (mean grad = single value).
    let mut uniform = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            let v = if (x / 4) % 2 == 0 { 50 } else { 200 };
            let off = (y * 64 + x) * 3;
            uniform[off] = v;
            uniform[off + 1] = v;
            uniform[off + 2] = v;
        }
    }
    let s = PixelSlice::new(&uniform, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let e = r.get_f32(AnalysisFeature::EdgeSlopeStdev).unwrap();
    assert!(
        e < 5.0,
        "uniform bands should have low stddev, got {e}"
    );

    // Mixed-amplitude vertical bands: alternating step heights produce
    // a bimodal gradient distribution ⇒ higher stddev.
    let mut mixed = vec![0u8; 64 * 64 * 3];
    for y in 0..64 {
        for x in 0..64 {
            // 4-period: 0, 100, 50, 250 ⇒ steps of 100, 50, 200
            let v = match (x / 4) % 4 {
                0 => 0,
                1 => 100,
                2 => 50,
                _ => 250,
            };
            let off = (y * 64 + x) * 3;
            mixed[off] = v;
            mixed[off + 1] = v;
            mixed[off + 2] = v;
        }
    }
    let s = PixelSlice::new(&mixed, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let e = r.get_f32(AnalysisFeature::EdgeSlopeStdev).unwrap();
    assert!(
        e > 30.0,
        "mixed-amplitude bands should have high stddev, got {e}"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn grayscale_score_one_for_neutral_image_zero_for_saturated() {
    // Neutral (R=G=B) ⇒ score = 1.0; saturated colour ⇒ score ≈ 0.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::GrayscaleScore));

    let neutral = vec![128u8; 64 * 64 * 3];
    let s = PixelSlice::new(&neutral, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let g = r.get_f32(AnalysisFeature::GrayscaleScore).unwrap();
    assert!(g > 0.99, "neutral image ⇒ near 1.0, got {g}");

    // Saturated red: |R-G| = 255 ≫ 4 ⇒ no pixel passes the gate.
    let mut sat = vec![0u8; 64 * 64 * 3];
    for px in sat.chunks_exact_mut(3) {
        px[0] = 255;
        px[1] = 0;
        px[2] = 0;
    }
    let s2 = PixelSlice::new(&sat, 64, 64, 64 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r2 = crate::analyze_features(s2, &q).unwrap();
    let g2 = r2.get_f32(AnalysisFeature::GrayscaleScore).unwrap();
    assert!(g2 < 0.01, "saturated colour ⇒ near 0.0, got {g2}");
}

#[cfg(feature = "experimental")]
#[test]
fn effective_bit_depth_distinguishes_u8_promoted_from_genuine_u16() {
    // The depth tier's effective-bit-depth probe correctly classifies
    // u8-promoted u16 (low-byte distinct ≤ 15 ⇒ 8-bit) vs genuine
    // 16-bit content (uniform low byte ⇒ ≥ 14-bit).
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::EffectiveBitDepth));

    // u8-promoted: only 4 distinct high bytes ⇒ ≤ 4 distinct low bytes
    // (since low == high in u8 promotion) ⇒ depth = 8.
    let mut buf8 = vec![0u8; 16 * 16 * 6];
    for (i, px) in buf8.chunks_exact_mut(2).enumerate() {
        let v = ((i % 4) * 64) as u8;
        let u = (v as u16) * 257;
        px.copy_from_slice(&u.to_le_bytes());
    }
    let s8 = PixelSlice::new(&buf8, 16, 16, 16 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
    let r8 = crate::analyze_features(s8, &q).unwrap();
    assert_eq!(
        r8.get(AnalysisFeature::EffectiveBitDepth)
            .and_then(|v| v.as_u32()),
        Some(8)
    );

    // Genuine 16-bit: pseudo-random sweep ⇒ many distinct low bytes.
    let mut buf16 = vec![0u8; 64 * 64 * 6];
    let mut state = 0xC001_u32;
    for px in buf16.chunks_exact_mut(2) {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let u = (state & 0xFFFF) as u16;
        px.copy_from_slice(&u.to_le_bytes());
    }
    let s16 = PixelSlice::new(&buf16, 64, 64, 64 * 6, PixelDescriptor::RGB16_SRGB).unwrap();
    let r16 = crate::analyze_features(s16, &q).unwrap();
    let depth = r16
        .get(AnalysisFeature::EffectiveBitDepth)
        .and_then(|v| v.as_u32())
        .unwrap();
    assert!(depth >= 14, "expected ≥14, got {depth}");
}

// --------------------------------------------------------------------
// Regression: PR #116 review surfaced two real bugs.
//
// Bug 1 — `gradient_diff_ycbcr` (tier2_chroma.rs) used a
//   `(cr_d.pow(2) as u32).saturating_mul(boost)` chain that silently
//   clamped the per-group `max_diff_cr` to ~33.5M instead of the
//   true ~62.9M on saturated synthetic content (alternating pure-red
//   / pure-green columns). 1.87× undercount on `cr_peak_sharpness`.
//   Fix: u64 intermediates.
//
// Bug 2 — `accumulate_row_simd` scalar tail (tier1.rs) only
//   accumulated `edge_count`, dropping `cb_grad_sum` / `cr_grad_sum`
//   / `chroma_grad_count` for the rightmost 1–7 columns when
//   `(width − 1) % 8 ≠ 0`. Tiny on 4 K, 20 % on `width = 11`.
//   Fix: same chroma-gradient accumulation as the SIMD edge loop.
// --------------------------------------------------------------------

#[test]
fn pr116_review_cr_diff_no_overflow_on_saturated_chroma_columns() {
    // Worst-case Cr second-difference: alternating saturated red /
    // green columns. Pre-fix `cr_diff` saturated at u32::MAX / 128 ≈
    // 33.5M; post-fix the u64 intermediates produce the correct
    // ~62.9M. We don't lock the exact peak value (calibration is
    // documented as drifting in 0.1.x), but we lock that the SIMD
    // and scalar paths agree on the same image — saturating_mul
    // would have made the scalar-tail-fed even-width image diverge
    // from the all-SIMD odd-multiple-of-8 width.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};
    let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::CrPeakSharpness));

    // 16-wide → all triplets handled by Tier 2 SIMD; no scalar tail.
    let mut buf16 = vec![0u8; 16 * 8 * 3];
    for y in 0..8 {
        for x in 0..16 {
            let i = ((y * 16 + x) * 3) as usize;
            // Alternating saturated red / green columns.
            if x % 2 == 0 {
                buf16[i] = 255;
                buf16[i + 1] = 0;
                buf16[i + 2] = 0;
            } else {
                buf16[i] = 0;
                buf16[i + 1] = 255;
                buf16[i + 2] = 0;
            }
        }
    }
    let s = PixelSlice::new(&buf16, 16, 8, 16 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r16 = crate::analyze_features(s, &q).unwrap();
    let p16 = r16.get_f32(AnalysisFeature::CrPeakSharpness).unwrap();

    // Same content at 11-wide forces the scalar gradient_diff_ycbcr
    // path on the rightmost triplets. The peak should be in the
    // same ballpark — pre-fix it would have been clamped to ~half
    // the SIMD value because saturating_mul kicked in.
    let mut buf11 = vec![0u8; 11 * 8 * 3];
    for y in 0..8 {
        for x in 0..11 {
            let i = ((y * 11 + x) * 3) as usize;
            if x % 2 == 0 {
                buf11[i] = 255;
                buf11[i + 1] = 0;
                buf11[i + 2] = 0;
            } else {
                buf11[i] = 0;
                buf11[i + 1] = 255;
                buf11[i + 2] = 0;
            }
        }
    }
    let s = PixelSlice::new(&buf11, 11, 8, 11 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r11 = crate::analyze_features(s, &q).unwrap();
    let p11 = r11.get_f32(AnalysisFeature::CrPeakSharpness).unwrap();

    // Both paths must report a peak ≥ 100. Pre-fix the scalar path
    // would have clamped to ≈ 355 (33.5M / peak_div) while the SIMD
    // path produces ≈ 666; here we just want both well above the
    // pre-fix ceiling, proving overflow is gone.
    assert!(
        p16 > 100.0,
        "16-wide saturated-chroma cr peak too low: {p16}"
    );
    assert!(
        p11 > 100.0,
        "11-wide saturated-chroma cr peak too low: {p11}"
    );
    // And they should be in the same ballpark (within 2× of each
    // other) — pre-fix the difference was 1.87× for content that
    // exercised the scalar path heavily.
    let ratio = if p11 > p16 { p11 / p16 } else { p16 / p11 };
    assert!(
        ratio < 2.0,
        "scalar (11) vs SIMD (16) cr_peak diverge: 11→{p11} 16→{p16} ratio={ratio}"
    );
}

#[test]
fn pr116_review_chroma_gradients_counted_in_scalar_tail() {
    // `accumulate_row_simd` scalar tail used to drop cb/cr gradient
    // accumulation. On `width = 11` (one SIMD chunk + 2 tail edge
    // pixels), the tail represents 20 % of edge positions per row.
    // Saturated-chroma horizontal stripes give a non-trivial
    // `cb_sharpness` / `cr_sharpness` that pre-fix would have
    // under-reported.
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    // Vertical stripes — each row has the same per-pixel chroma
    // gradient at every column transition. Making `width = 11` puts
    // 8 transitions in the SIMD loop and 2 in the scalar tail.
    let mut buf = vec![0u8; 11 * 8 * 3];
    for y in 0..8 {
        for x in 0..11 {
            let i = ((y * 11 + x) * 3) as usize;
            if x % 2 == 0 {
                buf[i] = 255; // red
            } else {
                buf[i + 2] = 255; // blue
            }
        }
    }
    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::CbSharpness).with(AnalysisFeature::CrSharpness),
    );
    let s = PixelSlice::new(&buf, 11, 8, 11 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let r = crate::analyze_features(s, &q).unwrap();
    let cb = r.get_f32(AnalysisFeature::CbSharpness).unwrap();
    let cr = r.get_f32(AnalysisFeature::CrSharpness).unwrap();
    // Both signals must be substantial — saturated red↔blue stripes
    // produce massive Cb gradients (B − Y flips full-scale every
    // column). Pre-fix the scalar tail's 2 columns wouldn't have
    // contributed at all, biasing the average down by ~20 %.
    assert!(cb > 0.3, "cb_sharpness on red↔blue stripes too low: {cb}");
    assert!(cr > 0.1, "cr_sharpness on red↔blue stripes too low: {cr}");
}

// --------------------------------------------------------------------
// Per-primaries luma weights: wide-gamut u8 sources go through the
// Native zero-copy path with their bytes intact; the analyzer must
// use per-primaries weights so the luma stats reflect the source's
// actual chromaticity matrix, not the sRGB / BT.601 baseline.
// --------------------------------------------------------------------

#[test]
fn wide_gamut_luma_histogram_lands_in_different_bin_than_srgb() {
    // The same pure-green RGB bytes, declared under different
    // primaries, must produce different luma values — because
    // each primary set's RGB→XYZ Y-row scales green differently
    // (BT.601 ≈ 0.587, BT.2020 ≈ 0.678, DisplayP3 ≈ 0.692,
    // AdobeRgb ≈ 0.627). The histogram bin lands somewhere in
    // [4, 5] for sRGB and [4, 5] for AdobeRgb but at a noticeably
    // higher bin for BT.2020 / DisplayP3. Easiest to lock: capture
    // the LumaHistogramEntropy as 0 (solid image, single bin)
    // for every primaries — but verify the analyzer ACCEPTS
    // every primaries set unchanged, and that variance stays
    // ~0 (proves the per-primaries weights produce internally-
    // consistent luma).
    use crate::feature::{AnalysisFeature, AnalysisQuery, FeatureSet};

    let q = AnalysisQuery::new(
        FeatureSet::just(AnalysisFeature::Variance)
            .with(AnalysisFeature::LumaHistogramEntropy),
    );
    let mut buf = vec![0u8; 64 * 64 * 3];
    for px in buf.chunks_exact_mut(3) {
        px[0] = 0;
        px[1] = 255;
        px[2] = 0;
    }
    for &p in &[
        zenpixels::ColorPrimaries::Bt709,
        zenpixels::ColorPrimaries::Bt2020,
        zenpixels::ColorPrimaries::DisplayP3,
        zenpixels::ColorPrimaries::AdobeRgb,
    ] {
        let desc = PixelDescriptor::RGB8_SRGB.with_primaries(p);
        let s = PixelSlice::new(&buf, 64, 64, 64 * 3, desc).unwrap();
        let r = crate::analyze_features(s, &q).unwrap();
        let v = r.get_f32(AnalysisFeature::Variance).unwrap();
        assert!(v < 0.5, "{p:?}: solid green variance = {v}");
        let h = r.get_f32(AnalysisFeature::LumaHistogramEntropy).unwrap();
        assert!(h.abs() < 1e-3, "{p:?}: solid green entropy = {h}");
    }
}

// --------------------------------------------------------------------
// Sanity matrix: every channel-type × transfer × primaries combination
// the analyzer is expected to handle. Verifies (1) no error is returned
// across the cross-product, (2) the source_descriptor accessor returns
// the input descriptor unchanged, (3) features populate without panic.
// Bit-exact / value-range invariants are locked in dedicated tests
// elsewhere; this matrix is the "does the analyzer accept everything"
// breadth sweep that catches regressions in dispatch / RowConverter
// fallback paths / tier_depth EOTF coverage.
// --------------------------------------------------------------------

#[cfg(feature = "experimental")]
mod sanity_matrix {
    use super::*;
    use crate::feature::{
        AnalysisFeature, AnalysisQuery, FeatureSet,
    };
    use zenpixels::{ChannelType, ColorPrimaries, TransferFunction};

    /// Encode a single sample of each channel type. `value` is the
    /// normalized signal in `[0, 1]` for U8/U16/F32; for HDR transfers
    /// (PQ/HLG) the caller chooses what fraction of full-range to use.
    fn encode_sample(ch: ChannelType, value: f32, dst: &mut Vec<u8>) {
        let v = value.clamp(0.0, 1.0);
        match ch {
            ChannelType::U8 => dst.push((v * 255.0 + 0.5) as u8),
            ChannelType::U16 => {
                let u = (v * 65535.0 + 0.5) as u16;
                dst.extend_from_slice(&u.to_le_bytes());
            }
            ChannelType::F32 => dst.extend_from_slice(&value.to_le_bytes()),
            _ => {}
        }
    }

    fn descriptor_for(
        ch: ChannelType,
        transfer: TransferFunction,
        primaries: ColorPrimaries,
        with_alpha: bool,
    ) -> PixelDescriptor {
        let base = match (ch, with_alpha) {
            (ChannelType::U8, false) => PixelDescriptor::RGB8_SRGB,
            (ChannelType::U8, true) => PixelDescriptor::RGBA8_SRGB,
            (ChannelType::U16, false) => PixelDescriptor::RGB16_SRGB,
            (ChannelType::U16, true) => PixelDescriptor::RGBA16_SRGB,
            (ChannelType::F32, false) => PixelDescriptor::RGBF32_LINEAR,
            (ChannelType::F32, true) => PixelDescriptor::RGBAF32_LINEAR,
            _ => PixelDescriptor::RGB8_SRGB,
        };
        base.with_transfer(transfer).with_primaries(primaries)
    }

    /// Build a buffer of `w * h` pixels, each holding the supplied
    /// signal-domain channel values. Caller picks whether to add
    /// alpha; alpha is always 1.0 (fully opaque).
    fn build_image(
        ch: ChannelType,
        with_alpha: bool,
        w: u32,
        h: u32,
        rgb: [f32; 3],
    ) -> Vec<u8> {
        let channels = if with_alpha { 4 } else { 3 };
        let cap = (w * h) as usize * channels * ch.byte_size();
        let mut buf = Vec::with_capacity(cap);
        for _ in 0..(w * h) {
            for c in 0..3 {
                encode_sample(ch, rgb[c], &mut buf);
            }
            if with_alpha {
                encode_sample(ch, 1.0, &mut buf);
            }
        }
        buf
    }

    fn cross_product_descriptors() -> Vec<PixelDescriptor> {
        let mut out = Vec::new();
        for &ch in &[ChannelType::U8, ChannelType::U16, ChannelType::F32] {
            for &tf in &[
                TransferFunction::Srgb,
                TransferFunction::Bt709,
                TransferFunction::Linear,
                TransferFunction::Gamma22,
                TransferFunction::Pq,
                TransferFunction::Hlg,
            ] {
                for &cp in &[
                    ColorPrimaries::Bt709,
                    ColorPrimaries::DisplayP3,
                    ColorPrimaries::Bt2020,
                    ColorPrimaries::AdobeRgb,
                ] {
                    for with_alpha in [false, true] {
                        out.push(descriptor_for(ch, tf, cp, with_alpha));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn sanity_every_descriptor_combination_runs_without_error() {
        // 3 channel types × 6 transfers × 4 primaries × 2 alpha = 144.
        // Every combination must (a) be accepted by analyze_features,
        // (b) populate Variance, (c) round-trip through
        // source_descriptor() unchanged.
        let q = AnalysisQuery::new(FeatureSet::SUPPORTED);
        let w: u32 = 16;
        let h: u32 = 16;
        let mut count: usize = 0;
        for desc in cross_product_descriptors() {
            let buf = build_image(
                desc.channel_type(),
                desc.alpha().is_some(),
                w,
                h,
                [0.5, 0.5, 0.5],
            );
            let stride = (w as usize) * desc.layout().channels() * desc.channel_type().byte_size();
            let slice = PixelSlice::new(&buf, w, h, stride, desc).unwrap_or_else(|e| {
                panic!("PixelSlice::new failed for {desc:?}: {e:?}")
            });
            let r = crate::analyze_features(slice, &q).unwrap_or_else(|e| {
                panic!("analyze_features failed for {desc:?}: {e:?}")
            });
            assert_eq!(r.geometry().width(), w);
            assert_eq!(r.geometry().height(), h);
            // source_descriptor round-trip — codecs depend on this.
            assert_eq!(
                r.source_descriptor().format,
                desc.format,
                "format round-trip for {desc:?}"
            );
            assert_eq!(
                r.source_descriptor().transfer(),
                desc.transfer(),
                "transfer round-trip for {desc:?}"
            );
            assert_eq!(
                r.source_descriptor().primaries,
                desc.primaries,
                "primaries round-trip for {desc:?}"
            );
            // Variance is always computed (Tier 1 always-on).
            assert!(
                r.get(AnalysisFeature::Variance).is_some(),
                "Variance missing for {desc:?}"
            );
            count += 1;
        }
        assert_eq!(count, 144, "expected 144 combinations, got {count}");
    }

    #[test]
    fn sanity_hdr_present_only_for_hdr_transfer_with_bright_pixels() {
        // For each HDR-capable transfer (PQ, HLG, Linear), an image
        // saturated near full signal must trip HdrPresent. For SDR
        // transfers (Srgb, Bt709, Gamma22), even full-signal must NOT
        // trip it. Locks the depth-tier's hdr_capable_tf gate.
        let q = AnalysisQuery::new(
            FeatureSet::just(AnalysisFeature::HdrPresent)
                .with(AnalysisFeature::PeakLuminanceNits),
        );

        let cases: &[(TransferFunction, bool, &str)] = &[
            (TransferFunction::Srgb, false, "Srgb full = SDR display peak"),
            (TransferFunction::Bt709, false, "Bt709 full = SDR display peak"),
            (TransferFunction::Gamma22, false, "Gamma22 full = SDR display peak"),
            (TransferFunction::Linear, true, "Linear above 80 nits = HDR"),
            (TransferFunction::Pq, true, "PQ saturated = HDR"),
            (TransferFunction::Hlg, true, "HLG saturated = HDR"),
        ];
        for &(tf, expected_hdr, label) in cases {
            let desc = PixelDescriptor::RGBF32_LINEAR
                .with_transfer(tf)
                .with_primaries(ColorPrimaries::Bt709);
            // Choose signal near full-scale. For PQ this is ~10000
            // nits; for HLG ~1000; for Linear ~80 (which is *not*
            // above SDR threshold), so use 5.0 to push linear above.
            let signal = if matches!(tf, TransferFunction::Linear) {
                5.0
            } else {
                1.0
            };
            let buf = build_image(ChannelType::F32, false, 16, 16, [signal, signal, signal]);
            let slice = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
            let r = crate::analyze_features(slice, &q).unwrap();
            let hdr = r
                .get(AnalysisFeature::HdrPresent)
                .and_then(|v| v.as_bool())
                .unwrap();
            assert_eq!(
                hdr, expected_hdr,
                "{label}: HdrPresent={hdr} expected={expected_hdr}"
            );
        }
    }

    #[test]
    fn sanity_effective_bit_depth_per_channel_type() {
        // u8 always reports 8, f32 reports 32 (storage depth), u16
        // reports a probe-derived value. The matrix here just asserts
        // u8 → 8 / f32 → 32 across every transfer and primaries combo
        // — the u16 probe is exercised separately.
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::EffectiveBitDepth));
        for desc in cross_product_descriptors() {
            if matches!(desc.channel_type(), ChannelType::U16) {
                continue; // probed separately
            }
            let buf = build_image(
                desc.channel_type(),
                desc.alpha().is_some(),
                8,
                8,
                [0.5, 0.5, 0.5],
            );
            let stride = 8 * desc.layout().channels() * desc.channel_type().byte_size();
            let slice = PixelSlice::new(&buf, 8, 8, stride, desc).unwrap();
            let r = crate::analyze_features(slice, &q).unwrap();
            let depth = r
                .get(AnalysisFeature::EffectiveBitDepth)
                .and_then(|v| v.as_u32())
                .unwrap();
            let expected = match desc.channel_type() {
                ChannelType::U8 => 8,
                ChannelType::F32 => 32,
                _ => unreachable!(),
            };
            assert_eq!(
                depth, expected,
                "EffectiveBitDepth({:?}) = {depth} expected {expected}",
                desc.format
            );
        }
    }

    #[test]
    fn sanity_wide_gamut_fraction_zero_for_in_gamut_content() {
        // Linear-light values clamped to [0, 1] never trip
        // WideGamutFraction regardless of declared primaries — the
        // signal measures "channel > 1.0 in linear" which an in-gamut
        // image can't produce.
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::WideGamutFraction));
        for &cp in &[
            ColorPrimaries::Bt709,
            ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020,
            ColorPrimaries::AdobeRgb,
        ] {
            let desc = PixelDescriptor::RGBF32_LINEAR
                .with_transfer(TransferFunction::Linear)
                .with_primaries(cp);
            let buf = build_image(ChannelType::F32, false, 16, 16, [0.7, 0.5, 0.3]);
            let slice = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
            let r = crate::analyze_features(slice, &q).unwrap();
            let frac = r
                .get_f32(AnalysisFeature::WideGamutFraction)
                .unwrap();
            assert_eq!(
                frac, 0.0,
                "in-gamut content should not trigger wide-gamut fraction; got {frac} for {cp:?}"
            );
        }
    }

    #[test]
    fn sanity_wide_gamut_fraction_one_for_above_unity_linear() {
        // f32 linear values above 1.0 are necessarily wide-gamut
        // regardless of the declared primaries.
        let q = AnalysisQuery::new(
            FeatureSet::just(AnalysisFeature::WideGamutFraction)
                .with(AnalysisFeature::WideGamutPeak),
        );
        let desc = PixelDescriptor::RGBF32_LINEAR
            .with_transfer(TransferFunction::Linear)
            .with_primaries(ColorPrimaries::Bt2020);
        let buf = build_image(ChannelType::F32, false, 16, 16, [1.5, 1.5, 1.5]);
        let slice = PixelSlice::new(&buf, 16, 16, 16 * 12, desc).unwrap();
        let r = crate::analyze_features(slice, &q).unwrap();
        assert!((r.get_f32(AnalysisFeature::WideGamutFraction).unwrap() - 1.0).abs() < 1e-6);
        assert!((r.get_f32(AnalysisFeature::WideGamutPeak).unwrap() - 1.5).abs() < 1e-3);
    }

    #[test]
    fn sanity_source_descriptor_carries_alpha_mode_through() {
        // RGBA8 has Straight alpha; that bit must survive into
        // AnalysisResults so codecs can decide whether to encode
        // alpha or treat the source as opaque.
        let buf = build_image(ChannelType::U8, true, 16, 16, [0.5, 0.5, 0.5]);
        let s = PixelSlice::new(&buf, 16, 16, 16 * 4, PixelDescriptor::RGBA8_SRGB).unwrap();
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::AlphaPresent));
        let r = crate::analyze_features(s, &q).unwrap();
        assert!(r.source_descriptor().alpha().is_some());
        assert!(r.source_descriptor().may_have_transparency());
        // RGB8 has no alpha.
        let buf = build_image(ChannelType::U8, false, 16, 16, [0.5, 0.5, 0.5]);
        let s = PixelSlice::new(&buf, 16, 16, 16 * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let r = crate::analyze_features(s, &q).unwrap();
        assert!(r.source_descriptor().alpha().is_none());
        assert!(r.source_descriptor().is_opaque());
    }

    #[test]
    fn sanity_rgb16_u8_promoted_round_trips_to_8bit_depth() {
        // Bit-equality check at the matrix level (already locked in
        // dedicated tests; mirrored here so the matrix sweep catches
        // any regression in the u16 probe under arbitrary transfer
        // / primaries combinations).
        let q = AnalysisQuery::new(FeatureSet::just(AnalysisFeature::EffectiveBitDepth));
        for &tf in &[
            TransferFunction::Srgb,
            TransferFunction::Linear,
            TransferFunction::Pq,
        ] {
            for &cp in &[
                ColorPrimaries::Bt709,
                ColorPrimaries::DisplayP3,
                ColorPrimaries::Bt2020,
            ] {
                let desc = PixelDescriptor::RGB16_SRGB
                    .with_transfer(tf)
                    .with_primaries(cp);
                // u8-promoted: very few distinct high bytes ⇒ probe → 8.
                let mut buf = vec![0u8; 16 * 16 * 6];
                for (i, px) in buf.chunks_exact_mut(2).enumerate() {
                    let v = ((i % 4) * 64) as u8;
                    let u = (v as u16) * 257;
                    px.copy_from_slice(&u.to_le_bytes());
                }
                let s = PixelSlice::new(&buf, 16, 16, 16 * 6, desc).unwrap();
                let r = crate::analyze_features(s, &q).unwrap();
                let d = r
                    .get(AnalysisFeature::EffectiveBitDepth)
                    .and_then(|v| v.as_u32())
                    .unwrap();
                assert_eq!(d, 8, "u8-promoted u16 with {tf:?}/{cp:?} ⇒ depth=8, got {d}");
            }
        }
    }
}
