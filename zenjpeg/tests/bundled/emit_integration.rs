//! Exercises the zencodec `resolve_color_emit` color/metadata path wired into
//! the `JpegEncoder` trait chain (`with_metadata_policy` →
//! `build_request_from` → `resolve_color_emit`). The existing `encode_api` /
//! `encode_request_guide` tests drive the low-level `config.request()` builder
//! directly and so bypass the color-emit resolver; these tests go through the
//! full `EncoderConfig::job().with_metadata_policy().encoder().encode()` chain
//! so the [`zencodec::ColorEmitPlan`] actually drives ICC / EXIF emission.

#![cfg(feature = "zencodec")]

use zencodec::{Metadata, MetadataPolicy};
use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
use zenjpeg::JpegEncoderConfig;
use zenpixels::{Orientation as ZpOrientation, PixelDescriptor, PixelSlice};

/// 4×4 RGB8 checkerboard, tightly packed.
fn rgb_4x4() -> Vec<u8> {
    let mut v = Vec::with_capacity(4 * 4 * 3);
    for y in 0..4 {
        for x in 0..4 {
            let on = (x + y) % 2 == 0;
            let (r, g, b) = if on { (200, 40, 40) } else { (40, 40, 200) };
            v.extend_from_slice(&[r, g, b]);
        }
    }
    v
}

fn encode_with_meta(meta: Metadata) -> Vec<u8> {
    let pixels = rgb_4x4();
    let slice =
        PixelSlice::new(&pixels, 4, 4, 4 * 3, PixelDescriptor::RGB8_SRGB).expect("pixel slice");
    let enc = JpegEncoderConfig::new()
        .job()
        .with_metadata_policy(meta, MetadataPolicy::PreserveExact)
        .encoder()
        .expect("encoder build");
    enc.encode(slice).expect("encode").into_vec()
}

fn has_marker(jpeg: &[u8], needle: &[u8]) -> bool {
    jpeg.windows(needle.len()).any(|w| w == needle)
}

/// A non-sRGB ICC profile present in the metadata must be embedded
/// (`IccDisposition::KeepSource`) via the color-emit plan.
#[test]
fn emit_embeds_source_icc() {
    // 132-byte minimal-length ICC-ish blob (not recognized as sRGB → kept).
    let icc = vec![0u8; 132];
    let meta = Metadata::none().with_icc(icc);
    let jpeg = encode_with_meta(meta);
    assert!(
        has_marker(&jpeg, b"ICC_PROFILE\0"),
        "color-emit plan should have embedded the source ICC via APP2"
    );
}

/// An sRGB-tagged CICP with no ICC: JPEG has no CICP carrier and sRGB is the
/// assumed default, so the plan synthesizes nothing — no ICC marker, encode
/// still valid.
#[test]
fn emit_srgb_cicp_no_icc_marker() {
    let meta = Metadata::none().with_cicp(zencodec::Cicp::SRGB);
    let jpeg = encode_with_meta(meta);
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "valid JPEG SOI");
    assert!(
        !has_marker(&jpeg, b"ICC_PROFILE\0"),
        "sRGB default must not synthesize a redundant ICC"
    );
}

/// A CICP-only Display-P3 source must synthesize an embedded ICC
/// (`IccDisposition::SynthesizeFrom` lowered via zenpixels-convert's
/// `synthesize_icc_for_cicp`): JPEG has no CICP carrier, so an embedded ICC is
/// the only way the wide-gamut description survives — otherwise the file is
/// silently relabeled sRGB. The oracle is the bundled `DISPLAY_P3_V4` profile.
#[test]
fn emit_cicp_only_display_p3_synthesizes_icc() {
    let meta = Metadata::none().with_cicp(zencodec::Cicp::DISPLAY_P3);
    let jpeg = encode_with_meta(meta);
    assert!(
        has_marker(&jpeg, b"ICC_PROFILE\0"),
        "CICP-only Display-P3 source must synthesize an embedded ICC"
    );
    // The synthesized bytes are the bundled Display-P3 profile (it fits in a
    // single APP2 segment at 480 bytes, so it appears contiguously).
    let p3 = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    assert!(
        jpeg.windows(p3.len()).any(|w| w == p3),
        "embedded ICC should be the bundled Display-P3 profile"
    );
}

/// A CICP-only source on a grayscale encode must NOT synthesize an RGB ICC —
/// an RGB profile over gray pixels would recolor them (the resolver's
/// grayscale terminal state).
#[test]
fn emit_gray_cicp_only_synthesizes_nothing() {
    let pixels: Vec<u8> = (0..16u8).map(|i| i * 16).collect();
    let slice =
        PixelSlice::new(&pixels, 4, 4, 4, PixelDescriptor::GRAY8_SRGB).expect("pixel slice");
    let meta = Metadata::none().with_cicp(zencodec::Cicp::DISPLAY_P3);
    let enc = JpegEncoderConfig::new()
        .job()
        .with_metadata_policy(meta, MetadataPolicy::PreserveExact)
        .encoder()
        .expect("encoder build");
    let jpeg = enc.encode(slice).expect("encode").into_vec();
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "valid JPEG SOI");
    assert!(
        !has_marker(&jpeg, b"ICC_PROFILE\0"),
        "grayscale encode must not synthesize an RGB ICC from CICP"
    );
}

/// EXIF (including an orientation tag) provided via metadata must be carried
/// through to an APP1 Exif segment.
#[test]
fn emit_carries_exif_with_orientation() {
    // Build a metadata with an EXIF blob whose orientation tag is Rotate90 (6).
    let exif = exif_orientation_blob(6);
    let meta = Metadata::none().with_exif(exif);
    assert_eq!(meta.orientation, ZpOrientation::Rotate90);
    let jpeg = encode_with_meta(meta);
    assert!(
        has_marker(&jpeg, b"Exif\0\0"),
        "EXIF blob should be carried via APP1"
    );
}

/// ICC + EXIF together both survive the color-emit-driven encode.
#[test]
fn emit_icc_and_exif_together() {
    let icc = vec![1u8; 200];
    let exif = exif_orientation_blob(1);
    let meta = Metadata::none().with_icc(icc).with_exif(exif);
    let jpeg = encode_with_meta(meta);
    assert!(has_marker(&jpeg, b"ICC_PROFILE\0"), "ICC embedded");
    assert!(has_marker(&jpeg, b"Exif\0\0"), "EXIF carried");
}

/// Minimal little-endian TIFF with a single Orientation (0x0112) SHORT entry.
fn exif_orientation_blob(value: u16) -> Vec<u8> {
    let mut v = b"II".to_vec();
    v.extend_from_slice(&42u16.to_le_bytes()); // magic
    v.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    v.extend_from_slice(&1u16.to_le_bytes()); // entry count
    v.extend_from_slice(&0x0112u16.to_le_bytes()); // tag = Orientation
    v.extend_from_slice(&3u16.to_le_bytes()); // type = SHORT
    v.extend_from_slice(&1u32.to_le_bytes()); // count
    v.extend_from_slice(&value.to_le_bytes()); // value
    v.extend_from_slice(&0u16.to_le_bytes()); // value padding
    v.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0
    v
}
