//! Smoke test for the unstable `__diagnostics` capture feature.
//!
//! Verifies that an encode with `with_diagnostics(true)` populates
//! the per-component block lists with non-default data, and that the
//! population covers every block in the grid.

#![cfg(feature = "__diagnostics")]

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};

/// Generate a deterministic non-trivial 32×32 RGB pattern. Uses a sine
/// modulation per-channel so every 8×8 block has non-zero AC energy.
fn make_test_image(width: u32, height: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            // Channel-dependent low-freq + high-freq mixture.
            let fx = x as f32;
            let fy = y as f32;
            let r = (128.0
                + 64.0 * (fx * 0.3).sin()
                + 32.0 * ((fx + fy) * 0.9).cos())
                .clamp(0.0, 255.0) as u8;
            let g = (128.0
                + 64.0 * (fy * 0.25).cos()
                + 32.0 * ((fx - fy) * 0.7).sin())
                .clamp(0.0, 255.0) as u8;
            let b = (128.0 + 96.0 * ((fx + fy) * 0.15).sin()).clamp(0.0, 255.0) as u8;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

fn assert_blocks_populated(
    diag: &zenjpeg::encode::diagnostics::EncodeDiagnostics,
    label: &str,
) {
    assert!(
        !diag.components.is_empty(),
        "[{label}] expected at least one component"
    );
    for (ci, comp) in diag.components.iter().enumerate() {
        let expected = (comp.block_grid.0 as usize) * (comp.block_grid.1 as usize);
        assert_eq!(
            comp.blocks.len(),
            expected,
            "[{label}] component {ci} block count mismatch: \
             grid {:?} → {expected} expected, got {}",
            comp.block_grid,
            comp.blocks.len()
        );
        let populated = comp
            .blocks
            .iter()
            .filter(|b| b.coef_pre_quant.iter().any(|&c| c != 0.0))
            .count();
        assert!(
            populated > 0,
            "[{label}] component {ci} has zero populated pre-quant DCT blocks \
             (expected at least one block with non-zero AC energy)"
        );
        // For the synthetic pattern, every block should have non-trivial
        // energy. Allow a small tolerance for edge effects but be loud
        // about big gaps.
        let populated_ratio = populated as f32 / expected.max(1) as f32;
        assert!(
            populated_ratio >= 0.9,
            "[{label}] component {ci}: only {populated}/{expected} \
             ({:.1}%) blocks populated — diagnostics capture coverage gap",
            populated_ratio * 100.0
        );
        // AQ multipliers should be non-zero (default is 0.08 mean).
        let aq_nonzero = comp
            .blocks
            .iter()
            .filter(|b| b.aq_multiplier > 0.0)
            .count();
        assert!(
            aq_nonzero > 0,
            "[{label}] component {ci} has no blocks with non-zero AQ multiplier"
        );
    }
}

#[test]
fn ycbcr_444_baseline_aq_only_smoke() {
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None)
        .aq_enabled(true)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    assert_eq!(diag.image.width, w);
    assert_eq!(diag.image.height, h);
    // 32×32 → 4×4 Y blocks, 4×4 chroma blocks (no subsampling).
    assert_eq!(diag.components.len(), 3);
    assert_eq!(diag.components[0].block_grid, (4, 4));
    assert_eq!(diag.components[1].block_grid, (4, 4));
    assert_eq!(diag.components[2].block_grid, (4, 4));
    assert_blocks_populated(&diag, "ycbcr_444_aq_only");
}

#[test]
fn ycbcr_420_baseline_aq_only_smoke() {
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .aq_enabled(true)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    // 32×32 with 4:2:0: Y is 4×4 blocks, chroma is 2×2 blocks.
    assert_eq!(diag.components.len(), 3);
    assert_eq!(diag.components[0].block_grid, (4, 4));
    assert_eq!(diag.components[1].block_grid, (2, 2));
    assert_eq!(diag.components[2].block_grid, (2, 2));
    assert_blocks_populated(&diag, "ycbcr_420_aq_only");
}

/// 4:2:2 — h_samp_y=2, v_samp_y=1. Should populate Y in MCU-traversal
/// order and chroma in raster order with half-width grids.
#[test]
fn ycbcr_422_aq_only_smoke() {
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::HalfHorizontal)
        .aq_enabled(true)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    // 4:2:2: Y is 4×4 blocks, chroma is 2×4 blocks.
    assert_eq!(diag.components.len(), 3);
    assert_eq!(diag.components[0].block_grid, (4, 4));
    assert_eq!(diag.components[1].block_grid, (2, 4));
    assert_eq!(diag.components[2].block_grid, (2, 4));
    assert_blocks_populated(&diag, "ycbcr_422_aq_only");
}

/// AQ disabled: blocks still populate with neutral AQ (encoder still
/// pushes constant 0.0 strengths through the same path).
#[test]
fn ycbcr_444_aq_disabled_smoke() {
    let w = 24u32;
    let h = 24u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None)
        .aq_enabled(false)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    // 24×24 → 3×3 Y blocks per component (no subsampling).
    assert_eq!(diag.components.len(), 3);
    assert_eq!(diag.components[0].block_grid, (3, 3));
    // Pre-quant DCT and levels still populated even with AQ off.
    let populated = diag.components[0]
        .blocks
        .iter()
        .filter(|b| b.coef_pre_quant.iter().any(|&c| c != 0.0))
        .count();
    assert!(populated >= 8, "expected ~9 blocks populated, got {populated}");
}

/// Trellis-only mode (mozjpeg-compatible R-D). The same per-block
/// capture point handles both the SIMD fast path and the trellis
/// branch, so post-quant levels should reflect trellis decisions.
#[cfg(feature = "trellis")]
#[test]
fn ycbcr_444_trellis_only_smoke() {
    use zenjpeg::encode::trellis::TrellisConfig;
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let trellis_cfg = TrellisConfig::new();
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None)
        .aq_enabled(true)
        .trellis(trellis_cfg)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    assert_eq!(diag.components.len(), 3);
    assert_blocks_populated(&diag, "ycbcr_444_trellis_only");
}

/// Hybrid (AQ-coupled trellis) mode via auto_optimize.
#[cfg(feature = "trellis")]
#[test]
fn ycbcr_420_hybrid_smoke() {
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .auto_optimize(true)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    assert_eq!(diag.components.len(), 3);
    assert_blocks_populated(&diag, "ycbcr_420_hybrid");
}

/// XYB color path, no chroma subsampling. Verifies the diagnostics
/// captures work for the perceptual color-space encoder too.
#[test]
fn xyb_full_smoke() {
    use zenjpeg::encoder::XybSubsampling;
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::xyb(85, XybSubsampling::Full)
        .aq_enabled(true)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    assert_eq!(
        diag.image.color_path,
        zenjpeg::encode::diagnostics::ColorPathTag::Xyb
    );
    // XYB Full: all three components at full resolution.
    assert_eq!(diag.components.len(), 3);
    assert_eq!(diag.components[0].block_grid, (4, 4));
    assert_blocks_populated(&diag, "xyb_full");
}

/// XYB with B-channel quartered (default XYB subsampling). The B
/// component grid should be half-resolution.
#[test]
fn xyb_b_quarter_smoke() {
    use zenjpeg::encoder::XybSubsampling;
    let w = 32u32;
    let h = 32u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::xyb(85, XybSubsampling::BQuarter)
        .aq_enabled(true)
        .with_diagnostics(true);
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    let diag = diag.expect("diagnostics enabled → Some");
    assert_eq!(
        diag.image.color_path,
        zenjpeg::encode::diagnostics::ColorPathTag::Xyb
    );
    assert_eq!(diag.components.len(), 3);
    // X (luma-ish): 4×4. Y component (Cb-slot): 4×4. B (Cr-slot): 2×2.
    assert_eq!(diag.components[0].block_grid, (4, 4));
    assert_eq!(diag.components[2].block_grid, (2, 2));
    assert_blocks_populated(&diag, "xyb_b_quarter");
}

#[test]
fn diagnostics_off_returns_none() {
    let w = 16u32;
    let h = 16u32;
    let pixels = make_test_image(w, h);
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
    // No `.with_diagnostics(true)` — feature is on but capture is off.
    let request = config.request();
    let mut encoder = request
        .encode_from_bytes(w, h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
        .expect("build encoder");
    encoder
        .push_packed(&pixels, enough::Unstoppable)
        .expect("push pixels");
    let (bytes, diag) = encoder.finish_with_diagnostics().expect("finish");
    assert!(!bytes.is_empty());
    assert!(diag.is_none(), "without with_diagnostics(true), no diag");
}
