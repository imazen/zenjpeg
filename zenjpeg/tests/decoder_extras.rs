//! Tests for decoder extras preservation.
use enough::Unstoppable;

use zenjpeg::decoder::{Decoder, MpfImageType, PreserveConfig, SegmentType};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Create a simple test image.
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            data[idx] = (x * 255 / width) as u8; // R
            data[idx + 1] = (y * 255 / height) as u8; // G
            data[idx + 2] = 128; // B
        }
    }
    data
}

#[test]
fn test_default_preservation_config() {
    let config = PreserveConfig::default();

    // By default, most metadata should be preserved
    assert!(config.jfif);
    assert!(config.exif);
    assert!(config.xmp);
    assert!(config.iptc);
    assert!(config.adobe);
    assert!(config.com);

    // Gain maps should be preserved
    assert!(config.mpf_gainmaps);

    // Thumbnails should not be preserved by default
    assert!(!config.mpf_thumbnails);
}

#[test]
fn test_preserve_none_config() {
    let config = PreserveConfig::none();

    assert!(!config.jfif);
    assert!(!config.exif);
    assert!(!config.xmp);
    assert!(!config.iptc);
    assert!(!config.adobe);
    assert!(!config.com);
    assert!(!config.mpf_gainmaps);
}

#[test]
fn test_preserve_all_config() {
    let config = PreserveConfig::all();

    assert!(config.jfif);
    assert!(config.exif);
    assert!(config.xmp);
    assert!(config.iptc);
    assert!(config.adobe);
    assert!(config.com);
    assert!(config.app_unknown);
    assert!(config.mpf_gainmaps);
    assert!(config.mpf_thumbnails);
}

#[test]
fn test_extras_empty_when_preserve_none() {
    let pixels = create_test_image(64, 64);

    // Encode
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&pixels, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("encode");

    // Decode with no preservation
    let decoded = Decoder::new()
        .preserve_none()
        .decode(&jpeg, Unstoppable)
        .expect("decode");

    // Should have no extras (or empty extras)
    match decoded.extras() {
        None => {} // Expected
        Some(extras) => assert!(extras.is_empty(), "extras should be empty"),
    }
}

#[test]
fn test_extras_preserved_with_default_config() {
    let pixels = create_test_image(64, 64);

    // Encode
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&pixels, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("encode");

    // Decode with default preservation
    let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");

    // Should have extras if there are any segments to preserve
    // Note: A minimal JPEG might not have JFIF, EXIF, etc.
    // The extras might be None if there's nothing to preserve
    if let Some(extras) = decoded.extras() {
        // If we have extras, segments should be accessible
        let _segments = extras.segments();
    }
}

#[test]
fn test_mpf_image_type_codes() {
    // Undefined (gain maps)
    let t = MpfImageType::from_type_code(0x000000);
    assert!(t.is_gainmap());
    assert!(!t.is_thumbnail());

    // VGA thumbnail
    let t = MpfImageType::from_type_code(0x010001);
    assert!(t.is_thumbnail());
    assert!(!t.is_gainmap());

    // Disparity
    let t = MpfImageType::from_type_code(0x020002);
    assert!(t.is_depth());

    // Panorama
    let t = MpfImageType::from_type_code(0x020001);
    assert!(t.is_multiframe());

    // Round-trip
    assert_eq!(MpfImageType::Undefined.to_type_code(), 0x000000);
    assert_eq!(MpfImageType::LargeThumbnailVga.to_type_code(), 0x010001);
    assert_eq!(MpfImageType::Disparity.to_type_code(), 0x020002);
}

#[test]
fn test_preserve_config_builder() {
    let config = PreserveConfig::default()
        .jfif(false)
        .exif(false)
        .xmp(true)
        .mpf_gainmaps(true)
        .mpf_thumbnails(true);

    assert!(!config.jfif);
    assert!(!config.exif);
    assert!(config.xmp);
    assert!(config.mpf_gainmaps);
    assert!(config.mpf_thumbnails);
}

#[test]
fn test_segment_type_values() {
    // Just verify the enum variants exist and are distinct
    let jfif = SegmentType::Jfif;
    let exif = SegmentType::Exif;
    let xmp = SegmentType::Xmp;
    let icc = SegmentType::Icc;
    let mpf = SegmentType::Mpf;
    let iptc = SegmentType::Iptc;
    let adobe = SegmentType::Adobe;
    let comment = SegmentType::Comment;
    let unknown = SegmentType::Unknown;

    assert_ne!(jfif, exif);
    assert_ne!(xmp, icc);
    assert_ne!(mpf, iptc);
    assert_ne!(adobe, comment);
    assert_ne!(comment, unknown);
}

#[test]
fn test_decoded_image_has_extras_method() {
    let pixels = create_test_image(32, 32);

    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::None);
    let mut enc = config
        .encode_from_bytes(32, 32, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    enc.push_packed(&pixels, Unstoppable).expect("push");
    let jpeg = enc.finish().expect("encode");

    let decoded = Decoder::new().decode(&jpeg, Unstoppable).expect("decode");

    // The extras() method should be callable
    let _extras_opt = decoded.extras();

    // The into_parts() method should work
    let (data, w, h, fmt, extras) = decoded.into_parts();
    assert_eq!(w, 32);
    assert_eq!(h, 32);
    assert!(!data.is_empty());
    let _ = (fmt, extras);
}
