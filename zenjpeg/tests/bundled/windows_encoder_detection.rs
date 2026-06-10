//! Windows (GDI+/WIC) encoder detection against real Windows-encoded JPEGs.
//!
//! Fixtures in `tests/testdata/windows_encoder/` were produced by a
//! Windows JPEG encoder via `https://z.zr.io/ri/red-leaf.jpg;width=8;quality=Q`
//! (`win_*` = default GDI+ builder; `wic*` = `;builder=wic` with
//! `;subsampling=444|422`; fetched 2026-06-10 — headers verified
//! byte-identical to the 256px corpus apart from SOF dimensions; the
//! probe is header-only so pixel count is irrelevant). The full
//! q=1..=100 sweeps at width=256 (GDI+ + WIC×{420,444,422}, 400
//! files) live in `/mnt/v/input/zenjpeg/windows-encoder/` with the
//! analysis scripts; every file carries byte-exact IJG tables (GDI+:
//! index `k = q - 1` except `k = q` at multiples of 25; WIC: `k = q`
//! except 53/59), standard Huffman, baseline, and JFIF density
//! 96×96 DPI.

use zenjpeg::detect::{Confidence, EncoderFamily, QualityScale, probe};

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/windows_encoder")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn windows_fixtures_detect_as_windows_imaging() {
    // (file, expected reported quality, expected subsampling)
    // win_q26 reports 25: GDI+ q=25 and q=26 emit byte-identical
    // tables (IJG index 25); the estimator reports the round number.
    // wic422_q90 reports 91: WIC q=90 emits IJG index 90, and reports
    // follow the GDI+ convention (k + 1).
    use zenjpeg::types::Subsampling;
    let cases = [
        ("win_q10.jpg", 10.0f32, Subsampling::S420),
        ("win_q26.jpg", 25.0, Subsampling::S420),
        ("win_q50.jpg", 50.0, Subsampling::S420),
        ("win_q85.jpg", 85.0, Subsampling::S420),
        ("wic444_q24.jpg", 24.0, Subsampling::S444),
        ("wic422_q90.jpg", 91.0, Subsampling::S422),
    ];

    for (name, want_quality, want_subsampling) in cases {
        let data = fixture(name);
        let result = probe(&data).unwrap_or_else(|e| panic!("probe {name}: {e}"));

        assert_eq!(
            result.encoder,
            EncoderFamily::WindowsImaging,
            "{name}: expected WindowsImaging, got {:?}",
            result.encoder
        );
        assert_eq!(result.quality.scale, QualityScale::WindowsQuality, "{name}");
        assert_eq!(
            result.quality.value, want_quality,
            "{name}: expected quality {want_quality}, got {}",
            result.quality.value
        );
        assert_eq!(result.quality.confidence, Confidence::Exact, "{name}");
        assert_eq!(result.mode, zenjpeg::types::JpegMode::Baseline, "{name}");
        assert_eq!(result.subsampling, want_subsampling, "{name}");
        assert_eq!(result.num_components, 3, "{name}");
        assert_eq!(result.dimensions.width, 8, "{name}");
    }
}

#[test]
fn windows_generic_quality_maps_to_value() {
    use zencodec::SourceEncodingDetails;

    let data = fixture("win_q85.jpg");
    let result = probe(&data).unwrap();
    assert_eq!(result.source_generic_quality(), Some(85.0));
}
