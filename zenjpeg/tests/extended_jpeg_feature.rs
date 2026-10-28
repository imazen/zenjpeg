//! Feature admission and exact retained rasters, including XYB.
use enough::Unstoppable;
use sha2::{Digest, Sha256};
use zenjpeg::decoder::Decoder;

#[test]
fn capability_matches_measured_rasters_and_refusals() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let report: serde_json::Value = serde_json::from_str(include_str!(
        "../../benchmarks/wasm_decoder_size_2026-10-27.json"
    ))
    .unwrap();
    let rows = report["raster_checks"][0]["results"].as_array().unwrap();
    let variant = if cfg!(feature = "extended-jpeg") {
        "full"
    } else {
        "no-both"
    };
    let mut checked = 0;
    for row in rows.iter().filter(|r| r["variant"] == variant) {
        let name = row["fixture"].as_str().unwrap();
        let path = if name.starts_with("xyb") {
            root.join("tests/testdata/permutation_regression")
                .join(name)
        } else {
            root.join("fuzz/seeds/native").join(name)
        };
        let data = std::fs::read(path).unwrap();
        let decoded = Decoder::new().decode(&data, Unstoppable);
        if row["refused"].as_bool().unwrap() {
            let err = decoded.unwrap_err();
            assert!(matches!(
                err.kind(),
                zenjpeg::decoder::ErrorKind::UnsupportedFeature { .. }
            ));
            assert!(err.to_string().contains("extended-jpeg"), "{name}: {err}");
            assert!(Decoder::new().scanline_reader(&data).is_err(), "{name}");
        } else {
            let decoded = decoded.unwrap();
            let mut hash = Sha256::new();
            hash.update(decoded.width().to_le_bytes());
            hash.update(decoded.height().to_le_bytes());
            hash.update(decoded.pixels_u8().unwrap());
            assert_eq!(
                format!("{:x}", hash.finalize()),
                row["raster_sha256"].as_str().unwrap(),
                "{name}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 16);
}

#[test]
fn unsupported_capability_does_not_bypass_frame_validation() {
    let mut data = include_bytes!("../fuzz/seeds/native/01-p12-c1-sof30-mode0-s1.jpg").to_vec();
    let offset = zenjpeg::container::iter(&data)
        .find(|s| matches!(s.kind, zenjpeg::container::MarkerKind::Sof(_)))
        .unwrap()
        .offset;
    // A zero sampling factor is invalid in both builds, before admission.
    data[offset + 11] = 0;
    let err = Decoder::new().decode(&data, Unstoppable).unwrap_err();
    assert!(err.to_string().contains("sampling factor is zero"), "{err}");
}

#[path = "support/native_fixture_rasters.rs"]
mod native_fixture_rasters;

#[test]
fn every_native_fixture_obeys_the_capability_and_retains_exact_samples() {
    native_fixture_rasters::check(cfg!(feature = "extended-jpeg"));
}
