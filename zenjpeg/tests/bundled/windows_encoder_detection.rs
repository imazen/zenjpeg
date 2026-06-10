//! Windows (GDI+/WIC) encoder detection against real Windows-encoded JPEGs.
//!
//! Fixtures in `tests/testdata/windows_encoder/` were produced by a
//! Windows JPEG encoder via `https://z.zr.io/ri/red-leaf.jpg;width=256;quality=Q`
//! (fetched 2026-06-09). The full q=1..=100 sweep lives in
//! `/mnt/v/input/zenjpeg/windows-encoder/` with the analysis scripts;
//! every file in that sweep carries IJG tables at index `k = q - 1`
//! (except `k = q` for multiples of 25), standard Huffman, baseline,
//! 4:2:0, and JFIF density 96×96 DPI.

use zenjpeg::detect::{Confidence, EncoderFamily, QualityScale, probe};

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/testdata/windows_encoder")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn windows_fixtures_detect_as_windows_imaging() {
    // (file, encoded-at GDI+ quality, expected reported quality)
    // win_q26 reports 25: GDI+ q=25 and q=26 emit byte-identical
    // tables (IJG index 25); the estimator reports the round number.
    let cases = [
        ("win_q10.jpg", 10.0f32),
        ("win_q26.jpg", 25.0),
        ("win_q50.jpg", 50.0),
        ("win_q85.jpg", 85.0),
    ];

    for (name, want_quality) in cases {
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
        assert_eq!(
            result.subsampling,
            zenjpeg::types::Subsampling::S420,
            "{name}"
        );
        assert_eq!(result.num_components, 3, "{name}");
        assert_eq!(result.dimensions.width, 256, "{name}");
    }
}

#[test]
fn windows_generic_quality_maps_to_value() {
    use zencodec::SourceEncodingDetails;

    let data = fixture("win_q85.jpg");
    let result = probe(&data).unwrap();
    assert_eq!(result.source_generic_quality(), Some(85.0));
}
