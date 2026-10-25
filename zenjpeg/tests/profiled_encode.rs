//! Profile-tagged bytes keep their original interpretation through the adapter.
#![cfg(feature = "zencodec")]
use std::{io::Cursor, sync::Arc};
use zencodec::decode::{DecodeJob, DecoderConfig};
use zencodec::encode::{EncodeJob, EncodePolicy, Encoder, EncoderConfig};
use zencodec::{Metadata, MetadataPolicy};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{
    ColorContext, ColorPrimaries, ColorProfileSource, PixelDescriptor, PixelSlice, TransferFunction,
};
use zenpixels_convert::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V4};

fn descriptor(base: PixelDescriptor, profile: &[u8]) -> PixelDescriptor {
    let (p, t) = ColorProfileSource::Icc(profile)
        .resolve()
        .unwrap_or((ColorPrimaries::Unknown, TransferFunction::Unknown));
    base.with_primaries(p).with_transfer(t)
}
#[test]
fn profile_tagged_bytes_match_legacy_packing_and_strided_rows_exactly() {
    for base in [
        PixelDescriptor::RGB8_SRGB,
        PixelDescriptor::RGBA8_SRGB,
        PixelDescriptor::BGRA8_SRGB,
    ] {
        let width = 97;
        let height = 65;
        let bpp = base.bytes_per_pixel();
        let stride = width as usize * bpp + 7 * bpp;
        let mut raw = vec![0xa5; stride * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel = &mut raw[y * stride + x * bpp..][..bpp];
                pixel[..3].copy_from_slice(&[(x * 7) as u8, (y * 11) as u8, ((x + y) * 3) as u8]);
                if bpp == 4 {
                    pixel[3] = 255;
                }
            }
        }
        let profile: Arc<[u8]> = Arc::from(ADOBE_RGB);
        let desc = descriptor(base, &profile);
        assert_ne!(desc.transfer(), TransferFunction::Srgb);
        let pixels = || {
            PixelSlice::new(&raw, width, height, stride, desc)
                .unwrap()
                .with_color_context(Arc::new(ColorContext::from_icc(profile.clone())))
        };
        for progressive in [false, true] {
            for huffman in [false, true] {
                let config = JpegEncoderConfig::new()
                    .with_calibrated_quality(87.0)
                    .with_progressive(progressive)
                    .with_huffman(huffman);
                let job = || {
                    config.clone().job().with_metadata_policy(
                        Metadata::none().with_icc(profile.clone()),
                        MetadataPolicy::PreserveExact,
                    )
                };
                let legacy = job()
                    .encoder()
                    .unwrap()
                    .encode(PixelSlice::new(&raw, width, height, stride, base).unwrap())
                    .unwrap();
                let direct = job().encoder().unwrap().encode(pixels()).unwrap();
                assert_eq!(direct.data(), legacy.data());
                for strip in [1, 7, 16, 31, 65] {
                    let mut encoder = job().encoder().unwrap();
                    encoder.begin(width, height, desc).unwrap();
                    for y in (0..height).step_by(strip as usize) {
                        encoder
                            .push_rows(pixels().sub_rows(y, strip.min(height - y)))
                            .unwrap();
                    }
                    let output = encoder.finish().unwrap();
                    assert_eq!(
                        output.data(),
                        direct.data(),
                        "{base:?} progressive={progressive} huffman={huffman} strip={strip}"
                    );
                }
                let mut oracle = jpeg_decoder::Decoder::new(Cursor::new(direct.data()));
                let decoded = oracle
                    .decode()
                    .expect("independent JPEG entropy/geometry decode");
                assert_eq!(decoded.len(), width as usize * height as usize * 3);
                assert_eq!(
                    JpegDecoderConfig::new()
                        .job()
                        .probe_full(direct.data())
                        .unwrap()
                        .source_color
                        .icc_profile
                        .as_deref(),
                    Some(profile.as_ref())
                );
            }
        }
    }
}

#[test]
fn profile_tagged_bytes_refuse_missing_discarded_and_conflicting_interpretation() {
    let desc = descriptor(PixelDescriptor::RGB8_SRGB, ADOBE_RGB);
    let raw = [17, 31, 61].repeat(16);
    let pixels = |d| PixelSlice::new(&raw, 4, 4, 12, d).unwrap();
    let job = || {
        JpegEncoderConfig::new().job().with_metadata_policy(
            Metadata::none().with_icc(ADOBE_RGB),
            MetadataPolicy::PreserveExact,
        )
    };
    assert!(JpegEncoderConfig::new().encode(pixels(desc)).is_err());
    assert!(
        job()
            .with_policy(EncodePolicy::none().with_embed_icc(false))
            .encoder()
            .unwrap()
            .encode(pixels(desc))
            .is_err()
    );
    let conflicting =
        pixels(desc).with_color_context(Arc::new(ColorContext::from_icc(DISPLAY_P3_V4)));
    assert!(job().encoder().unwrap().encode(conflicting).is_err());
    assert!(
        job()
            .encoder()
            .unwrap()
            .encode(pixels(desc.with_transfer(TransferFunction::Pq)))
            .is_err()
    );
    assert!(
        job()
            .encoder()
            .unwrap()
            .encode(pixels(desc.with_transfer(TransferFunction::Linear)))
            .is_err()
    );
    assert!(
        job()
            .encoder()
            .unwrap()
            .encode(pixels(
                desc.with_signal_range(zenpixels::SignalRange::Narrow)
            ))
            .is_err()
    );
    let mut xyb = JpegEncoderConfig::new();
    *xyb.inner_mut() =
        zenjpeg::encode::EncoderConfig::xyb(87.0, zenjpeg::encode::XybSubsampling::Full);
    assert!(
        xyb.job()
            .with_metadata_policy(
                Metadata::none().with_icc(ADOBE_RGB),
                MetadataPolicy::PreserveExact
            )
            .encoder()
            .unwrap()
            .encode(pixels(desc))
            .is_err()
    );
    let mut wrong_channels = ADOBE_RGB.to_vec();
    wrong_channels[16..20].copy_from_slice(b"GRAY");
    assert!(
        JpegEncoderConfig::new()
            .job()
            .with_metadata_policy(
                Metadata::none().with_icc(wrong_channels),
                MetadataPolicy::PreserveExact
            )
            .encoder()
            .unwrap()
            .encode(pixels(desc))
            .is_err()
    );
    // A failed begin is recoverable and must not mutate the previously valid job.
    let mut encoder = job().encoder().unwrap();
    assert!(
        encoder
            .begin(4, 4, desc.with_transfer(TransferFunction::Pq))
            .is_err()
    );
    encoder.begin(4, 4, desc).unwrap();
    encoder.push_rows(pixels(desc)).unwrap();
    assert!(!encoder.finish().unwrap().data().is_empty());
}
