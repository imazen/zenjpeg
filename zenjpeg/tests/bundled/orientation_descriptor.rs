//! Regression tests for orientation reporting and PixelDescriptor accuracy
//! across all zencodec decode paths.
//!
//! Creates 4x2 test JPEGs with every EXIF orientation (1-8) and verifies:
//! - Dimensions are correct (swapped for orientations 5-8 when auto-orienting)
//! - `ImageInfo.orientation` is Identity when auto-orient was applied
//! - `ImageInfo.orientation` is the source EXIF value when preserved
//! - `PixelDescriptor` reflects source color metadata (not hardcoded sRGB)
//! - `OutputInfo.orientation_applied` matches what was actually applied

#![cfg(feature = "zencodec")]

use std::borrow::Cow;

use zenjpeg::JpegDecoderConfig;
use zenjpeg::encode::encoder_config::EncoderConfig;
use zenjpeg::encode::encoder_types::{ChromaSubsampling, PixelLayout};
use zenjpeg::encode::exif::{Exif, Orientation};

use zencodec::OrientationHint;
use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
use zenpixels::{ColorPrimaries, TransferFunction};

/// All 8 EXIF orientation values.
const ALL_ORIENTATIONS: &[Orientation] = &[
    Orientation::Normal,
    Orientation::FlipHorizontal,
    Orientation::Rotate180,
    Orientation::FlipVertical,
    Orientation::Transpose,
    Orientation::Rotate90,
    Orientation::Transverse,
    Orientation::Rotate270,
];

/// Whether this EXIF orientation swaps width/height.
fn swaps_axes(orient: Orientation) -> bool {
    matches!(
        orient,
        Orientation::Transpose
            | Orientation::Rotate90
            | Orientation::Transverse
            | Orientation::Rotate270
    )
}

/// Convert encoder Orientation to zencodec Orientation.
fn to_zencodec_orient(orient: Orientation) -> zencodec::Orientation {
    zencodec::Orientation::from_exif(orient as u8).unwrap_or_default()
}

/// Encode a 4x2 test JPEG with the given EXIF orientation and subsampling.
fn encode_4x2(orient: Orientation, ss: ChromaSubsampling) -> Vec<u8> {
    // 8 distinct pixels so rotations produce unique patterns.
    // R G B W / C M Y K
    #[rustfmt::skip]
    let pixels: Vec<u8> = vec![
        255,   0,   0,    0, 255,   0,    0,   0, 255,  255, 255, 255,
          0, 255, 255,  255,   0, 255,  255, 255,   0,    0,   0,   0,
    ];
    let config = EncoderConfig::ycbcr(95.0, ss);
    config
        .request()
        .exif(Exif::build().orientation(orient))
        .encode_bytes(&pixels, 4, 2, PixelLayout::Rgb8Srgb)
        .expect("encode failed")
}

// ── Orientation: Preserve (default) ────────────────────────────────────

#[test]
fn preserve_orientation_reports_source_exif() {
    for &orient in ALL_ORIENTATIONS {
        let jpeg = encode_4x2(orient, ChromaSubsampling::None);
        let config = JpegDecoderConfig::new();
        // Default OrientationHint is Preserve → no auto-orient
        let output = config.decode(&jpeg).expect("decode failed");
        let info = output.info();

        // Dimensions always 4x2 (no rotation applied)
        assert_eq!(info.width, 4, "orient={orient:?}: wrong width");
        assert_eq!(info.height, 2, "orient={orient:?}: wrong height");

        // Orientation should be the source EXIF value
        let expected = to_zencodec_orient(orient);
        assert_eq!(
            info.orientation, expected,
            "orient={orient:?}: expected source orientation"
        );
    }
}

// ── Orientation: Correct (auto-orient) ─────────────────────────────────

#[test]
fn correct_orientation_reports_identity() {
    for &orient in ALL_ORIENTATIONS {
        let jpeg = encode_4x2(orient, ChromaSubsampling::None);
        let config = JpegDecoderConfig::new();

        let output = config
            .clone()
            .job()
            .with_orientation(OrientationHint::Correct)
            .decoder(Cow::Borrowed(&jpeg), &[])
            .expect("decoder failed")
            .decode()
            .expect("decode failed");

        let info = output.info();

        // After auto-orient, orientation should be Identity
        assert_eq!(
            info.orientation,
            zencodec::Orientation::Identity,
            "orient={orient:?}: expected Identity after auto-orient"
        );

        // Dimensions should be swapped for orientations 5-8
        if swaps_axes(orient) {
            assert_eq!(info.width, 2, "orient={orient:?}: expected swapped width");
            assert_eq!(info.height, 4, "orient={orient:?}: expected swapped height");
        } else {
            assert_eq!(info.width, 4, "orient={orient:?}: expected original width");
            assert_eq!(
                info.height, 2,
                "orient={orient:?}: expected original height"
            );
        }
    }
}

// ── OutputInfo orientation_applied ──────────────────────────────────────

#[test]
fn output_info_reports_orientation_applied() {
    for &orient in ALL_ORIENTATIONS {
        let jpeg = encode_4x2(orient, ChromaSubsampling::None);
        let config = JpegDecoderConfig::new();

        let job = config
            .clone()
            .job()
            .with_orientation(OrientationHint::Correct);
        let out_info = job.output_info(&jpeg).expect("output_info failed");

        let expected = to_zencodec_orient(orient);
        assert_eq!(
            out_info.orientation_applied, expected,
            "orient={orient:?}: output_info should report orientation_applied"
        );

        if swaps_axes(orient) {
            assert_eq!(out_info.width, 2, "orient={orient:?}: output_info width");
            assert_eq!(out_info.height, 4, "orient={orient:?}: output_info height");
        } else {
            assert_eq!(out_info.width, 4, "orient={orient:?}: output_info width");
            assert_eq!(out_info.height, 2, "orient={orient:?}: output_info height");
        }
    }
}

#[test]
fn output_info_preserve_no_orientation_applied() {
    for &orient in ALL_ORIENTATIONS {
        let jpeg = encode_4x2(orient, ChromaSubsampling::None);
        let config = JpegDecoderConfig::new();
        // Default is Preserve
        let job = config.clone().job();
        let out_info = job.output_info(&jpeg).expect("output_info failed");

        assert_eq!(
            out_info.orientation_applied,
            zencodec::Orientation::Identity,
            "orient={orient:?}: Preserve should not apply orientation"
        );
        assert_eq!(out_info.width, 4);
        assert_eq!(out_info.height, 2);
    }
}

// ── PixelDescriptor: no ICC → sRGB ─────────────────────────────────────

#[test]
fn descriptor_srgb_for_plain_jpeg() {
    let jpeg = encode_4x2(Orientation::Normal, ChromaSubsampling::None);
    let config = JpegDecoderConfig::new();
    let output = config.decode(&jpeg).expect("decode failed");

    let desc = output.pixels().descriptor();
    assert_eq!(desc.transfer, TransferFunction::Srgb);
    assert_eq!(desc.primaries, ColorPrimaries::Bt709);
}

#[test]
fn descriptor_srgb_across_subsampling_modes() {
    for ss in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        let jpeg = encode_4x2(Orientation::Normal, ss);
        let config = JpegDecoderConfig::new();
        let output = config.decode(&jpeg).expect("decode failed");

        let desc = output.pixels().descriptor();
        assert_eq!(
            desc.transfer,
            TransferFunction::Srgb,
            "ss={ss:?}: expected sRGB transfer"
        );
        assert_eq!(
            desc.primaries,
            ColorPrimaries::Bt709,
            "ss={ss:?}: expected BT.709 primaries"
        );
    }
}

// ── PixelDescriptor: with ICC profile ──────────────────────────────────

#[test]
fn descriptor_unknown_for_non_srgb_icc() {
    // Encode a JPEG with a fake (non-sRGB) ICC profile
    let pixels: Vec<u8> = vec![128u8; 4 * 2 * 3];
    let fake_icc = vec![0u8; 256]; // Not a known sRGB profile
    let config = EncoderConfig::ycbcr(95.0, ChromaSubsampling::None);
    let jpeg = config
        .request()
        .icc_profile(&fake_icc[..])
        .encode_bytes(&pixels, 4, 2, PixelLayout::Rgb8Srgb)
        .expect("encode failed");

    let decoder_config = JpegDecoderConfig::new();
    let output = decoder_config.decode(&jpeg).expect("decode failed");

    let desc = output.pixels().descriptor();
    assert_eq!(
        desc.transfer,
        TransferFunction::Unknown,
        "non-sRGB ICC should yield Unknown transfer"
    );
    assert_eq!(
        desc.primaries,
        ColorPrimaries::Unknown,
        "non-sRGB ICC should yield Unknown primaries"
    );

    // ICC profile should still be preserved in source_color
    assert!(
        output.info().source_color.icc_profile.is_some(),
        "ICC profile should be preserved in source_color"
    );
}

// ── Orientation × subsampling matrix ───────────────────────────────────

#[test]
fn orientation_correct_all_subsampling_modes() {
    for ss in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        let jpeg = encode_4x2(Orientation::Rotate90, ss);
        let config = JpegDecoderConfig::new();

        let output = config
            .clone()
            .job()
            .with_orientation(OrientationHint::Correct)
            .decoder(Cow::Borrowed(&jpeg), &[])
            .expect("decoder failed")
            .decode()
            .expect("decode failed");

        let info = output.info();
        assert_eq!(
            info.orientation,
            zencodec::Orientation::Identity,
            "ss={ss:?}: expected Identity after Rotate90 auto-orient"
        );
        // Rotate90 swaps axes: 4x2 → 2x4
        assert_eq!(info.width, 2, "ss={ss:?}: expected width=2");
        assert_eq!(info.height, 4, "ss={ss:?}: expected height=4");
    }
}

// ── Streaming decoder orientation ──────────────────────────────────────

#[test]
fn streaming_decoder_reports_identity_after_auto_orient() {
    use zencodec::decode::StreamingDecode as _;

    let jpeg = encode_4x2(Orientation::Rotate90, ChromaSubsampling::None);
    let config = JpegDecoderConfig::new();

    let mut streamer = config
        .clone()
        .job()
        .with_orientation(OrientationHint::Correct)
        .streaming_decoder(Cow::Borrowed(&jpeg), &[])
        .expect("streaming_decoder failed");

    let info = streamer.info();
    assert_eq!(
        info.orientation,
        zencodec::Orientation::Identity,
        "streaming: expected Identity after auto-orient"
    );
    // NOTE: ImageInfo.width/height come from the JPEG header (pre-transform).
    // The actual output dimensions from next_batch() reflect the transform.
    // This is a known limitation — ImageInfo dimensions are source dimensions.

    // Drain the stream and verify actual output dimensions
    let mut total_rows = 0u32;
    let mut batch_width = 0u32;
    while let Some((_, batch)) = streamer.next_batch().expect("batch failed") {
        batch_width = batch.width();
        total_rows += batch.rows();
    }
    // Rotate90 swaps: 4x2 → 2x4
    assert_eq!(batch_width, 2, "streaming batch width after Rotate90");
    assert_eq!(total_rows, 4, "streaming total rows after Rotate90");
}

#[test]
fn streaming_decoder_preserve_reports_source() {
    use zencodec::decode::StreamingDecode as _;

    let jpeg = encode_4x2(Orientation::Rotate270, ChromaSubsampling::None);
    let config = JpegDecoderConfig::new();

    // Default Preserve
    let mut streamer = config
        .clone()
        .job()
        .streaming_decoder(Cow::Borrowed(&jpeg), &[])
        .expect("streaming_decoder failed");

    let info = streamer.info();
    assert_eq!(
        info.orientation,
        to_zencodec_orient(Orientation::Rotate270),
        "streaming preserve: expected source orientation"
    );
    assert_eq!(info.width, 4);
    assert_eq!(info.height, 2);

    while let Some(_batch) = streamer.next_batch().expect("batch failed") {}
}

// ── Push decoder orientation ───────────────────────────────────────────

/// Simple sink that uses PixelSliceMut from a backing buffer.
struct CollectSink {
    data: Vec<u8>,
    stride: usize,
}

impl CollectSink {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            stride: 0,
        }
    }
}

impl zencodec::decode::DecodeRowSink for CollectSink {
    fn begin(
        &mut self,
        width: u32,
        height: u32,
        descriptor: zenpixels::PixelDescriptor,
    ) -> Result<(), zencodec::decode::SinkError> {
        let bpp = descriptor.bytes_per_pixel();
        self.stride = width as usize * bpp;
        self.data.resize(self.stride * height as usize, 0);
        Ok(())
    }

    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        descriptor: zenpixels::PixelDescriptor,
    ) -> Result<zenpixels::PixelSliceMut<'_>, zencodec::decode::SinkError> {
        let start = y as usize * self.stride;
        let end = start + height as usize * self.stride;
        zenpixels::PixelSliceMut::new(
            &mut self.data[start..end],
            width,
            height,
            self.stride,
            descriptor,
        )
        .map_err(|e| -> zencodec::decode::SinkError { e.to_string().into() })
    }

    fn finish(&mut self) -> Result<(), zencodec::decode::SinkError> {
        Ok(())
    }
}

#[test]
fn push_decoder_reports_orientation_applied() {
    let jpeg = encode_4x2(Orientation::Rotate90, ChromaSubsampling::None);
    let config = JpegDecoderConfig::new();
    let mut sink = CollectSink::new();

    let out_info = config
        .clone()
        .job()
        .with_orientation(OrientationHint::Correct)
        .push_decoder(Cow::Borrowed(&jpeg), &mut sink, &[])
        .expect("push_decoder failed");

    assert_eq!(
        out_info.orientation_applied,
        to_zencodec_orient(Orientation::Rotate90),
        "push_decoder should report Rotate90 applied"
    );
    assert_eq!(out_info.width, 2);
    assert_eq!(out_info.height, 4);
}

#[test]
fn push_decoder_preserve_no_orientation() {
    let jpeg = encode_4x2(Orientation::Rotate90, ChromaSubsampling::None);
    let config = JpegDecoderConfig::new();
    let mut sink = CollectSink::new();

    let out_info = config
        .clone()
        .job()
        .push_decoder(Cow::Borrowed(&jpeg), &mut sink, &[])
        .expect("push_decoder failed");

    assert_eq!(
        out_info.orientation_applied,
        zencodec::Orientation::Identity,
        "push_decoder Preserve should not apply orientation"
    );
    assert_eq!(out_info.width, 4);
    assert_eq!(out_info.height, 2);
}

/// Regression test: `push_decoder` must accept `Cow::Owned` and produce
/// identical output to `Cow::Borrowed`. The slice only needs scope-local
/// lifetime within `push_decoder_native`, so both Cow variants are valid.
#[test]
fn push_decoder_accepts_cow_owned() {
    let jpeg = encode_4x2(Orientation::Normal, ChromaSubsampling::None);
    let config = JpegDecoderConfig::new();

    // Decode with Cow::Borrowed
    let mut borrowed_sink = CollectSink::new();
    let borrowed_info = config
        .clone()
        .job()
        .push_decoder(Cow::Borrowed(&jpeg), &mut borrowed_sink, &[])
        .expect("push_decoder with Cow::Borrowed failed");

    // Decode with Cow::Owned (clone the bytes into a fresh Vec)
    let mut owned_sink = CollectSink::new();
    let owned_info = config
        .clone()
        .job()
        .push_decoder(Cow::Owned(jpeg.clone()), &mut owned_sink, &[])
        .expect("push_decoder with Cow::Owned must succeed");

    // Output dimensions and orientation should match
    assert_eq!(owned_info.width, borrowed_info.width);
    assert_eq!(owned_info.height, borrowed_info.height);
    assert_eq!(
        owned_info.orientation_applied,
        borrowed_info.orientation_applied
    );

    // Decoded pixels must be byte-identical
    assert_eq!(
        owned_sink.data, borrowed_sink.data,
        "Cow::Owned and Cow::Borrowed must produce identical pixel output"
    );
    assert_eq!(owned_sink.stride, borrowed_sink.stride);
}
