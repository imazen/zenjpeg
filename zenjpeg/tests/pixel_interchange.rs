//! Shared JPEG pixel interpretation and bounded row output.
#![cfg(feature = "zencodec")]

use std::{borrow::Cow, num::NonZeroU32};

use zencodec::decode::{Decode, DecodeJob, DecodeRowSink, DecoderConfig, SinkError};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig as NativeEncoder, PixelLayout, Unstoppable,
};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{ColorSpace, ImageSpec, PixelView, PixelViewMut, Primaries};

struct BoundedSink {
    cap: u32,
    image: Vec<u8>,
    stride: usize,
    spec: Option<ImageSpec>,
    next: u32,
}

impl DecodeRowSink for BoundedSink {
    fn begin(&mut self, width: u32, height: u32, spec: ImageSpec) -> Result<(), SinkError> {
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
        spec: ImageSpec,
    ) -> Result<PixelViewMut<'_>, SinkError> {
        assert!(height <= self.cap);
        assert_eq!(self.next, y);
        self.next += height;
        assert_eq!(self.spec.as_ref(), Some(&spec));
        let start = self.stride * y as usize;
        let end = start + self.stride * height as usize;
        Ok(PixelViewMut::new(
            &mut self.image[start..end],
            width,
            height,
            self.stride,
            spec,
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
                let actual =
                    PixelView::new(&sink.image, w, h, sink.stride, sink.spec.unwrap()).unwrap();
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
    let p3 = ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3);
    let jpeg = JpegEncoderConfig::new()
        .job()
        .encoder()
        .unwrap()
        .encode(PixelView::from_packed(&rgb, w, h, p3).unwrap())
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
    } = &pixels.spec().color
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
    assert_eq!(decoded.pixels().spec(), &ImageSpec::rgb8_srgb());
    assert!(decoded.info().source_color.icc_profile.is_some());
}

#[cfg(feature = "ultrahdr")]
#[test]
fn ultrahdr_reconstruction_bakes_exif_after_applying_the_map() {
    use zencodec::{GainMapRender, Metadata, MetadataPolicy, Orientation, OrientationHint};
    use zenjpeg::ultrahdr::{
        GainMap, GainMapEncodingFormat, GainMapMetadata, encode_with_gainmap_metadata,
    };
    use zenpixels::PixelBuf;

    let (w, h) = (17, 19);
    let base = PixelBuf::from_bytes(pattern(w, h), w, h, ImageSpec::rgb8_srgb()).unwrap();
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
        &Metadata::none().with_orientation(Orientation::Rotate90),
        &MetadataPolicy::PreserveExact,
        Unstoppable,
    )
    .unwrap();

    let mut oracle = jpeg_decoder::Decoder::new(jpeg.as_slice());
    oracle.decode().unwrap();
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
        assert_eq!(pixels.spec().layout.order, ChannelOrder::CmykInverted);
        assert!(matches!(
            pixels.spec().color,
            ColorSpace::Cmyk { inverted: true, .. }
        ));
        for y in 0..pixels.rows() {
            assert!(pixels.row(y).chunks_exact(4).all(|pixel| pixel == expected));
        }
    }
}
