//! Shared JPEG pixel interpretation and bounded row output.
#![cfg(feature = "zencodec")]

use std::{borrow::Cow, num::NonZeroU32};

use zencodec::decode::{Decode, DecodeJob, DecodeRowSink, DecoderConfig, SinkError};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig as NativeEncoder, PixelLayout, Unstoppable,
};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{ColorSpace, PixelFormat, PixelView, PixelViewMut, Primaries};

struct BoundedSink {
    cap: u32,
    image: Vec<u8>,
    stride: usize,
    spec: Option<PixelFormat>,
    next: u32,
}

impl DecodeRowSink for BoundedSink {
    fn begin(&mut self, width: u32, height: u32, spec: PixelFormat) -> Result<(), SinkError> {
        assert!(self.spec.is_none(), "a sink begins once");
        self.stride = width as usize * spec.layout.bytes_per_pixel() + 7;
        self.image.resize(self.stride * height as usize, 0xa5);
        self.spec = Some(spec);
        Ok(())
    }

    fn max_buffer_rows(&self) -> NonZeroU32 {
        if self.spec.is_some() {
            NonZeroU32::new(self.cap).unwrap()
        } else {
            NonZeroU32::MAX
        }
    }

    fn provide_next_buffer(
        &mut self,
        y: u32,
        height: u32,
        width: u32,
        spec: PixelFormat,
    ) -> Result<PixelViewMut<'_>, SinkError> {
        assert!(height <= self.cap);
        assert_eq!(self.next, y);
        self.next += height;
        assert_eq!(self.spec.as_ref(), Some(&spec));
        let start = self.stride * y as usize;
        let end = start + self.stride * height as usize;
        Ok(PixelViewMut::new_with_format_cow(
            &mut self.image[start..end],
            width,
            height,
            self.stride,
            zenpixels::FormatCow::from_owned(spec),
        )?)
    }
}

fn pattern(width: u32, height: u32) -> Vec<u8> {
    (0..width * height * 3)
        .map(|i| (i.wrapping_mul(107) ^ (i >> 3)) as u8)
        .collect()
}

#[test]
fn bounded_padded_sink_matches_one_shot_at_odd_edges() {
    let (w, h) = (35, 39);
    let rgb = pattern(w, h);
    for sampling in [
        ChromaSubsampling::None,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfVertical,
    ] {
        for progressive in [false, true] {
            let mut encoder = NativeEncoder::ycbcr(86, sampling)
                .progressive(progressive)
                .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            encoder.push_packed(&rgb, Unstoppable).unwrap();
            let jpeg = encoder.finish().unwrap();
            let config = JpegDecoderConfig::new();
            let one_shot = config
                .clone()
                .job()
                .decoder(Cow::Borrowed(&jpeg), &[])
                .unwrap()
                .decode()
                .unwrap();
            for cap in [1, 3, 8, 16, h] {
                let mut sink = BoundedSink {
                    cap,
                    image: Vec::new(),
                    stride: 0,
                    spec: None,
                    next: 0,
                };
                config
                    .clone()
                    .job()
                    .push_decoder(Cow::Borrowed(&jpeg), &mut sink, &[])
                    .unwrap();
                assert_eq!(sink.next, h);
                let sink_spec = sink.spec.unwrap();
                let actual = PixelView::new(&sink.image, w, h, sink.stride, &sink_spec).unwrap();
                for y in 0..h {
                    assert_eq!(
                        actual.row(y),
                        one_shot.pixels().row(y),
                        "{sampling:?} progressive={progressive} cap={cap} row={y}"
                    );
                    let end = (y as usize + 1) * sink.stride;
                    assert_eq!(&sink.image[end - 7..end], &[0xa5; 7]);
                }
            }
        }
    }
}

#[test]
fn icc_only_p3_and_xyb_bind_current_rgb_and_source_icc() {
    let (w, h) = (17, 19);
    let rgb = pattern(w, h);
    let p3 = PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3);
    let jpeg = JpegEncoderConfig::new()
        .job()
        .encoder()
        .unwrap()
        .encode(PixelView::from_bytes(&rgb, w, h, &p3).unwrap())
        .unwrap();
    let mut oracle = jpeg_decoder::Decoder::new(jpeg.data());
    oracle.decode().unwrap();
    assert!(oracle.icc_profile().is_some());
    let decoded = JpegDecoderConfig::new().decode(jpeg.data()).unwrap();
    let pixels = decoded.pixels();
    let ColorSpace::Icc {
        profile,
        channels: 3,
        ..
    } = &pixels.format().color
    else {
        panic!("the stored Display P3 ICC must remain authoritative for the RGB samples");
    };
    assert_eq!(Some(profile.as_ref()), oracle.icc_profile().as_deref());

    let mut encoder = NativeEncoder::xyb(90, zenjpeg::encoder::XybSubsampling::BQuarter)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    encoder.push_packed(&rgb, Unstoppable).unwrap();
    let jpeg = encoder.finish().unwrap();
    let decoded = JpegDecoderConfig::new().decode(&jpeg).unwrap();
    assert_eq!(decoded.pixels().format(), &PixelFormat::RGB8_SRGB);
    assert!(decoded.info().source_color.icc_profile.is_some());
}

#[cfg(feature = "ultrahdr")]
#[test]
fn ultrahdr_reconstruction_bakes_exif_after_applying_the_map() {
    use img_parts::{ImageEXIF, ImageICC};
    use zencodec::{GainMapRender, Metadata, MetadataPolicy, Orientation, OrientationHint};
    use zenjpeg::ultrahdr::{
        GainMap, GainMapEncodingFormat, GainMapMetadata, encode_with_gainmap_metadata,
    };
    use zenpixels::{DiffuseWhite, PixelBuf, Transfer};
    use zenpixels_convert::icc_profiles::DISPLAY_P3_V4;

    let (w, h) = (17, 19);
    let base_spec = PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3);
    let base = PixelBuf::from_bytes(pattern(w, h), w, h, base_spec).unwrap();
    let mut exif = zencodec::exif::Exif::new(zencodec::exif::TextEncoding::Ascii);
    exif.set_orientation(Orientation::Rotate90);
    let exif = zenmeta::exif::serialize(&exif);
    let source = Metadata::none()
        .with_exif(exif.as_slice())
        .with_icc(DISPLAY_P3_V4);
    let gain = GainMap {
        width: 5,
        height: 6,
        channels: 1,
        data: (0..30).map(|n| (n * 37) as u8).collect(),
    };
    let mut params = GainMapMetadata::default();
    params.alternate_hdr_headroom = 2.0;
    for channel in &mut params.channels {
        channel.max = 2.0;
    }
    let jpeg = encode_with_gainmap_metadata(
        &base,
        &gain,
        &params,
        &NativeEncoder::ycbcr(90, ChromaSubsampling::Quarter),
        90.0,
        GainMapEncodingFormat::Both,
        &source,
        &MetadataPolicy::PreserveExact,
        Unstoppable,
    )
    .unwrap();

    let mut oracle = jpeg_decoder::Decoder::new(jpeg.as_slice());
    oracle.decode().unwrap();
    assert_eq!(oracle.icc_profile().as_deref(), Some(DISPLAY_P3_V4));
    let source_jpeg = img_parts::jpeg::Jpeg::from_bytes(jpeg.clone().into()).unwrap();
    assert_eq!(source_jpeg.exif().as_deref(), Some(exif.as_slice()));
    let config = JpegDecoderConfig::new();
    let decode = |orient| {
        config
            .clone()
            .job()
            .with_orientation(orient)
            .with_gain_map_render(GainMapRender::ReconstructHdr {
                target_headroom: None,
            })
            .decoder(Cow::Borrowed(&jpeg), &[])
            .unwrap()
            .decode()
            .unwrap()
    };
    let stored = decode(OrientationHint::Preserve);
    assert_eq!(stored.info().orientation, Orientation::Rotate90);
    let corrected = decode(OrientationHint::Correct);
    assert_eq!(corrected.info().orientation, Orientation::Identity);
    for output in [&stored, &corrected] {
        assert_eq!(
            output.info().embedded_metadata.exif.as_deref(),
            Some(exif.as_slice())
        );
        assert_eq!(
            output.info().source_color.icc_profile.as_deref(),
            Some(DISPLAY_P3_V4)
        );
        let pixels = output.pixels();
        assert_eq!(pixels.origin().unwrap().icc.as_deref(), Some(DISPLAY_P3_V4));
        assert!(matches!(pixels.format().color, ColorSpace::Rgb {
            primaries: Primaries::DisplayP3, transfer: Transfer::Linear,
            white: Some(white), .. } if white == DiffuseWhite::BT2408));
        let sc = &output.info().source_color;
        assert!(sc.content_light_level.is_some());
        assert!(sc.mastering_display.is_some());
        assert_eq!(pixels.hdr().cll, sc.content_light_level);
        assert_eq!(pixels.hdr().mastering, sc.mastering_display);
    }
    assert_eq!(corrected.pixels().hdr(), stored.pixels().hdr());
    assert_eq!(
        (corrected.pixels().width(), corrected.pixels().rows()),
        (h, w)
    );
    let reference =
        zenpixels_convert::apply_orientation(&stored.pixels(), Orientation::Rotate90).unwrap();
    assert_eq!(
        corrected.pixels().as_strided_bytes(),
        reference.as_view().as_strided_bytes()
    );

    // Re-encode the already-oriented SDR base with the source metadata
    // provenance. The checked metadata path reconciles the orientation tag.
    let base = config
        .job()
        .with_orientation(OrientationHint::Correct)
        .decoder(Cow::Borrowed(&jpeg), &[])
        .unwrap()
        .decode()
        .unwrap();
    let mut metadata = base.info().metadata();
    metadata.xmp = None;
    let mut job = JpegEncoderConfig::new().job();
    job.try_set_metadata_policy(metadata, MetadataPolicy::PreserveExact)
        .unwrap();
    let output = job.encoder().unwrap().encode(base.pixels()).unwrap();
    let mut reader = jpeg_decoder::Decoder::new(output.data());
    reader.decode().unwrap();
    let info = reader.info().unwrap();
    assert_eq!((u32::from(info.width), u32::from(info.height)), (h, w));
    assert_eq!(reader.icc_profile().as_deref(), Some(DISPLAY_P3_V4));
    let parsed = img_parts::jpeg::Jpeg::from_bytes(output.data().to_vec().into()).unwrap();
    assert_eq!(parsed.icc_profile().as_deref(), Some(DISPLAY_P3_V4));
    let exif = exif::Reader::new()
        .read_raw(parsed.exif().unwrap().to_vec())
        .unwrap();
    let orientation = exif
        .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .unwrap();
    assert_eq!(orientation.value.get_uint(0), Some(1));
}

#[path = "support/jpeg_writer.rs"]
#[allow(dead_code)]
mod jpeg_writer;

#[test]
fn adobe_cmyk_and_ycck_passthrough_declare_inverted_ink() {
    use jpeg_writer::{Comp, Segment, Spec, app14, dc_block, write};
    use zenpixels::ChannelOrder;

    for (transform, expected) in [(0, [96, 128, 128, 200]), (2, [159, 159, 159, 200])] {
        let jpeg = write(&Spec {
            width: 9,
            height: 11,
            comps: (1..=4).map(|id| Comp { id, h: 1, v: 1 }).collect(),
            block: &|component, _, _| dc_block([96, 128, 128, 200][component]),
            restart_interval: 0,
            quant: 8,
            segments: vec![Segment::PreFrame(app14(transform))],
        });
        let decoded = JpegDecoderConfig::new().decode(&jpeg).unwrap();
        let pixels = decoded.pixels();
        assert_eq!(pixels.format().layout.order, ChannelOrder::CmykInverted);
        assert!(matches!(
            pixels.format().color,
            ColorSpace::Cmyk { inverted: true, .. }
        ));
        for y in 0..pixels.rows() {
            assert!(
                pixels
                    .row(y)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|&pixel| pixel == expected)
            );
        }
    }
}

#[test]
fn linear_gray_codes_keep_their_gray_icc_carrier() {
    use std::sync::Arc;

    use zenpixels::{AlphaMode, Cicp, RenderingIntent, SampleLayout};
    use zenpixels_convert::icc_profiles::SynthesizedIcc;

    let SynthesizedIcc::Profile(profile) = SynthesizedIcc::for_gray_cicp(Cicp::new(1, 8, 0, true))
    else {
        panic!("the pinned database must supply the linear GRAY profile");
    };
    let spec = PixelFormat::new(
        SampleLayout::GRAY8,
        ColorSpace::Icc {
            profile: Arc::from(profile.as_ref()),
            channels: 1,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let samples = [72u8; 64];
    let jpeg = JpegEncoderConfig::grayscale(90.0)
        .job()
        .encoder()
        .unwrap()
        .encode(PixelView::from_bytes(&samples, 8, 8, &spec).unwrap())
        .unwrap();
    let mut oracle = jpeg_decoder::Decoder::new(jpeg.data());
    assert_eq!(oracle.decode().unwrap(), samples);
    assert_eq!(oracle.icc_profile().as_deref(), Some(profile.as_ref()));
}
