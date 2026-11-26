//! Profile-tagged bytes keep their original interpretation through the adapter.
#![cfg(feature = "zencodec")]
use std::{io::Cursor, sync::Arc};
use zencodec::decode::{DecodeJob, DecoderConfig};
use zencodec::encode::{EncodeJob, EncodePolicy, Encoder, EncoderConfig};
use zencodec::{Metadata, MetadataPolicy};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{ColorSpace, PixelFormat, PixelView, Primaries, Transfer};
use zenpixels_convert::icc_profiles::{ADOBE_RGB, DISPLAY_P3_V4};

fn descriptor(base: PixelFormat, profile: &[u8]) -> PixelFormat {
    let (p, t) = zenpixels::icc::IccIdentification::from_icc(profile)
        .map_or((Primaries::Unknown, Transfer::Unknown), |id| {
            (id.primaries, id.transfer)
        });
    base.with_primaries(p).with_transfer(t)
}
fn active_icc(spec: PixelFormat, profile: impl Into<Arc<[u8]>>) -> PixelFormat {
    let alpha = spec.color.alpha();
    PixelFormat::new(
        spec.layout,
        ColorSpace::Icc {
            profile: profile.into(),
            channels: 3,
            alpha,
            intent: zenpixels::RenderingIntent::RelativeColorimetric,
        },
    )
}

#[test]
#[allow(deprecated)] // Exercises the legacy metadata setter.
fn profile_tagged_bytes_match_legacy_packing_and_strided_rows_exactly() {
    for base in [
        PixelFormat::RGB8_SRGB,
        PixelFormat::RGBA8_SRGB,
        PixelFormat::BGRA8_SRGB,
        PixelFormat::RGBX8_SRGB,
        PixelFormat::BGRX8_SRGB,
    ] {
        let width = 97;
        let height = 65;
        let bpp = base.layout.bytes_per_pixel();
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
        let desc = active_icc(base.clone(), profile.clone());
        let pixels = || PixelView::new(&raw, width, height, stride, &desc).unwrap();
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
                    .encode(PixelView::new(&raw, width, height, stride, &base).unwrap())
                    .unwrap();
                let direct = job().encoder().unwrap().encode(pixels()).unwrap();
                assert_eq!(direct.data(), legacy.data());
                for strip in [1, 7, 16, 31, 65] {
                    let mut encoder = job().encoder().unwrap();
                    encoder.begin(width, height, desc.clone()).unwrap();
                    for y in (0..height).step_by(strip as usize) {
                        encoder
                            .push_rows(pixels().sub_rows(y, strip.min(height - y)).unwrap())
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
#[allow(deprecated)] // Exercises the legacy best-effort policy setter on purpose.
fn profile_tagged_bytes_refuse_missing_discarded_and_conflicting_interpretation() {
    let desc = descriptor(PixelFormat::RGB8_SRGB, ADOBE_RGB);
    let raw = [17, 31, 61].repeat(16);
    let pixels = |d| {
        PixelView::new_with_format_cow(&raw, 4, 4, 12, zenpixels::FormatCow::from_owned(d)).unwrap()
    };
    let job = || {
        JpegEncoderConfig::new().job().with_metadata_policy(
            Metadata::none().with_icc(ADOBE_RGB),
            MetadataPolicy::PreserveExact,
        )
    };
    assert!(
        JpegEncoderConfig::new()
            .encode(pixels(desc.clone()))
            .is_err()
    );
    assert!(
        job()
            .with_policy(EncodePolicy::none().with_embed_icc(false))
            .encoder()
            .unwrap()
            .encode(pixels(desc.clone()))
            .is_err()
    );
    let conflicting = pixels(active_icc(desc.clone(), DISPLAY_P3_V4));
    assert!(job().encoder().unwrap().encode(conflicting).is_err());
    assert!(
        job()
            .encoder()
            .unwrap()
            .encode(pixels(desc.clone().with_transfer(Transfer::Pq)))
            .is_err()
    );
    assert!(
        job()
            .encoder()
            .unwrap()
            .encode(pixels(desc.clone().with_transfer(Transfer::Linear)))
            .is_err()
    );
    assert!(
        job()
            .encoder()
            .unwrap()
            .encode(pixels(
                desc.clone().with_range(zenpixels::SignalRange::Narrow)
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
            .encode(pixels(desc.clone()))
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
            .encode(pixels(desc.clone()))
            .is_err()
    );
    // A failed begin is recoverable and must not mutate the previously valid job.
    let mut encoder = job().encoder().unwrap();
    assert!(
        encoder
            .begin(4, 4, desc.clone().with_transfer(Transfer::Pq))
            .is_err()
    );
    encoder.begin(4, 4, desc.clone()).unwrap();
    encoder.push_rows(pixels(desc.clone())).unwrap();
    assert!(!encoder.finish().unwrap().data().is_empty());
}

#[test]
#[allow(deprecated)] // Exercises the legacy metadata setter.
fn checked_verbatim_color_keeps_profile_across_erasure_and_rows() {
    use zencodec::{CodecSet, ColorEmitPolicy, ImageFormat, UnsupportedOperation};
    let policy = EncodePolicy::none().with_color(ColorEmitPolicy::Verbatim);
    for icc in [ADOBE_RGB, DISPLAY_P3_V4] {
        let raw = [17, 31, 61, 7, 11, 13, 0xee, 0xee, 0xee].repeat(9);
        let desc = active_icc(PixelFormat::RGB8_SRGB, icc);
        let pixels = || PixelView::new(&raw, 2, 9, 9, &desc).unwrap();
        for progressive in [false, true] {
            let config = JpegEncoderConfig::new().with_progressive(progressive);
            let metadata = Metadata::none().with_icc(icc);
            let mut job = config
                .clone()
                .job()
                .with_metadata_policy(metadata.clone(), MetadataPolicy::PreserveExact)
                .try_with_policy(policy)
                .unwrap();
            for refused in [
                policy.with_strict(true),
                policy.with_embed_exif(false),
                policy.with_embed_xmp(false),
                policy.with_embed_icc(false),
            ] {
                assert_eq!(
                    job.try_set_policy(refused),
                    Err(UnsupportedOperation::EncodePolicy)
                );
            }
            job.try_set_policy(EncodePolicy::none()).unwrap();
            let direct = job.encoder().unwrap().encode(pixels()).unwrap();
            let registry = CodecSet::new().with_send_encoder(config);
            let mut job = registry.encode_job(ImageFormat::Jpeg).unwrap();
            job.try_set_policy(policy).unwrap();
            job.set_metadata_policy(metadata, MetadataPolicy::PreserveExact);
            let mut rows = job.into_encoder().unwrap();
            rows.begin(2, 9, desc.clone()).unwrap();
            for y in 0..9 {
                rows.push_rows(pixels().sub_rows(y, 1).unwrap()).unwrap();
            }
            let output = rows.finish().unwrap();
            assert_eq!(output.data(), direct.data());
            let info = JpegDecoderConfig::new()
                .job()
                .probe_full(output.data())
                .unwrap();
            assert_eq!(info.source_color.icc_profile.as_deref(), Some(icc));
            assert_eq!(info.source_color.color_origin().icc.as_deref(), Some(icc));
            let mut oracle = jpeg_decoder::Decoder::new(Cursor::new(output.data()));
            assert_eq!(oracle.decode().unwrap().len(), 2 * 9 * 3);
        }
    }
}

#[test]
#[allow(deprecated)] // Exercises the legacy metadata setter.
fn profile_descriptor_admission_checks_each_interpretation_guard() {
    let known = descriptor(PixelFormat::RGB8_SRGB, ADOBE_RGB);
    let unknown = known
        .clone()
        .with_primaries(Primaries::Unknown)
        .with_transfer(Transfer::Unknown);
    let mut opaque = vec![0u8; 132];
    opaque[..4].copy_from_slice(&132u32.to_be_bytes());
    opaque[12..16].copy_from_slice(b"mntr");
    opaque[16..20].copy_from_slice(b"RGB ");
    opaque[20..24].copy_from_slice(b"XYZ ");
    opaque[36..40].copy_from_slice(b"acsp");
    assert!(zenpixels::icc::IccIdentification::from_icc(&opaque).is_none());
    for (profile, desc, accepted) in [
        (ADOBE_RGB, unknown.clone(), true),
        (
            ADOBE_RGB,
            known.clone().with_transfer(Transfer::Unknown),
            true,
        ),
        (
            ADOBE_RGB,
            known.clone().with_primaries(Primaries::Bt709),
            false,
        ),
        (opaque.as_slice(), unknown.clone(), true),
        (
            opaque.as_slice(),
            known.clone().with_transfer(Transfer::Unknown),
            false,
        ),
        (
            opaque.as_slice(),
            unknown.clone().with_transfer(Transfer::AdobeRgb),
            false,
        ),
        (
            ADOBE_RGB,
            descriptor(PixelFormat::RGBA8_SRGB, ADOBE_RGB)
                .with_alpha(zenpixels::AlphaMode::Premultiplied),
            false,
        ),
        (
            ADOBE_RGB,
            descriptor(PixelFormat::RGBF32_LINEAR, ADOBE_RGB),
            false,
        ),
        (
            ADOBE_RGB,
            descriptor(PixelFormat::GRAY8_SRGB, ADOBE_RGB),
            false,
        ),
    ] {
        let mut encoder = JpegEncoderConfig::new()
            .job()
            .with_metadata_policy(
                Metadata::none().with_icc(profile),
                MetadataPolicy::PreserveExact,
            )
            .encoder()
            .unwrap();
        assert_eq!(
            encoder.begin(4, 4, desc.clone()).is_ok(),
            accepted,
            "{desc:?}"
        );
    }
    let mut encoder = JpegEncoderConfig::new()
        .job()
        .with_metadata_policy(Metadata::none(), MetadataPolicy::PreserveExact)
        .encoder()
        .unwrap();
    assert!(encoder.begin(4, 4, known).is_err());
}

#[test]
#[allow(deprecated)] // Exercises the legacy best-effort policy setter on purpose.
fn profile_bytes_refuse_color_carrier_removal_and_cicp_override() {
    let desc = descriptor(PixelFormat::RGB8_SRGB, ADOBE_RGB);
    let job = || {
        JpegEncoderConfig::new().job().with_metadata_policy(
            Metadata::none().with_icc(ADOBE_RGB),
            MetadataPolicy::PreserveExact,
        )
    };
    let mut encoder = job()
        .with_policy(
            EncodePolicy::none().with_color(zencodec::ColorEmitPolicy::Custom(
                zencodec::ColorEmitFields::new(
                    zencodec::IccRetention::Drop,
                    zencodec::CicpEmission::Never,
                ),
            )),
        )
        .encoder()
        .unwrap();
    assert!(encoder.begin(4, 4, desc.clone()).is_err());
    let raw = [17, 31, 61].repeat(16);
    let spec = active_icc(desc, DISPLAY_P3_V4);
    let pixels = PixelView::new(&raw, 4, 4, 12, &spec).unwrap();
    assert!(job().encoder().unwrap().encode(pixels).is_err());
}
