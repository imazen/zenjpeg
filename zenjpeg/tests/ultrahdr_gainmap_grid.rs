//! Regression gate for #193: the fused Ultra HDR encoder must declare the
//! gain-map metadata on the CONFIG quantization grid, not the content's
//! observed range.
//!
//! Standalone (not under `tests/bundled/`) because the `bundled` target links
//! `jpegli-internals-sys`, which cannot link on every host; this test only
//! needs the `ultrahdr` feature.
#![cfg(feature = "ultrahdr")]

use enough::Unstoppable;
use ultrahdr_core::pixel_buffer_from_vec;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig};
use zenjpeg::ultrahdr::{
    GainMapConfig, UhdrColorGamut, UhdrColorTransfer, UhdrPixelFormat, UltraHdrExtras,
};

/// Regression gate for #193 (the ultrahdr#33 defect class in the fused
/// encoder): `build_gainmap_metadata` used to declare the CONTENT's observed
/// gain range while `compute_gain_row` had quantized the bytes on the CONFIG
/// grid. Readers dequantize on the declared range, so every file whose
/// content range was narrower than the configured range came back
/// under-boosted.
///
/// Two gates: (1) structural — the declared per-channel `min`/`max` equal
/// the config grid; (2) round-trip — a known-peak HDR patch decoded at full
/// gain weight reconstructs its peak. With the default grid
/// (`max_boost = 6.0`) and a 4.0× patch, the pre-fix code declared
/// `max = log2(4)` for bytes written on a `log2(6)` grid and reconstructed
/// the patch at ~2.9× — well outside the tolerance below.
#[test]
fn fused_gainmap_metadata_declares_config_grid_and_roundtrips_peak() {
    use ultrahdr_core::gainmap::HdrOutputFormat;
    use zenjpeg::ultrahdr::{decode_ultrahdr_hdr, encode_ultrahdr_with_curve};
    use zentone::Bt2446C;

    const PEAK: f32 = 4.0;
    let (width, height) = (64u32, 64u32);
    let mut data = Vec::with_capacity((width * height * 16) as usize);
    for _y in 0..height {
        for x in 0..width {
            // Left half SDR white, right half a flat 4× patch (gray, so
            // 4:2:0 chroma subsampling cannot smear the peak).
            let v = if x < width / 2 { 1.0 } else { PEAK };
            for c in [v, v, v, 1.0f32] {
                data.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    let hdr = pixel_buffer_from_vec(
        data,
        width,
        height,
        UhdrPixelFormat::RgbaF32,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Linear,
    )
    .expect("HDR buffer");

    let config = GainMapConfig::default();
    let jpeg = encode_ultrahdr_with_curve(
        &hdr,
        &Bt2446C::new(203.0, 100.0),
        &config,
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("encode_ultrahdr_with_curve should succeed");

    // (1) Structural: declared range == quantization grid, on every channel.
    let decoded = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let extras = decoded.extras().expect("extras");
    let (metadata, _) = extras
        .ultrahdr_metadata()
        .expect("ultrahdr metadata present")
        .expect("ultrahdr metadata parses");
    let grid_min = (config.min_boost as f64).log2();
    let grid_max = (config.max_boost as f64).log2();
    for (i, ch) in metadata.channels.iter().enumerate() {
        assert!(
            (ch.min - grid_min).abs() < 1e-3,
            "channel {i}: declared min {} != config grid min {grid_min}",
            ch.min
        );
        assert!(
            (ch.max - grid_max).abs() < 1e-3,
            "channel {i}: declared max {} != config grid max {grid_max}",
            ch.max
        );
    }
    // Headroom keeps the grid top when the content stays inside the grid.
    assert!(
        metadata.alternate_hdr_headroom >= (config.alternate_hdr_headroom as f64).log2() - 1e-3,
        "alternate headroom {} narrower than config {}",
        metadata.alternate_hdr_headroom,
        (config.alternate_hdr_headroom as f64).log2()
    );

    // (2) Round-trip at full weight: the 4× patch must come back at ~4×.
    let hdr_back = decode_ultrahdr_hdr(
        &jpeg,
        config.alternate_hdr_headroom,
        HdrOutputFormat::LinearFloat,
    )
    .expect("decode_ultrahdr_hdr should succeed");
    assert_eq!(hdr_back.width(), width);
    assert_eq!(hdr_back.height(), height);
    let slice = hdr_back.as_slice();
    let mut peak = 0.0f32;
    for y in 0..height {
        let row = slice.row(y);
        for px in row[..(width as usize) * 16].as_chunks::<16>().0 {
            let r = f32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            peak = peak.max(r);
        }
    }
    let rel = (peak - PEAK).abs() / PEAK;
    assert!(
        rel < 0.10,
        "reconstructed peak {peak} vs encoded {PEAK} (rel err {rel:.3}); \
         metadata channel max {} — under-boost means the declared range is not the quantization grid",
        metadata.channels[0].max
    );
}
