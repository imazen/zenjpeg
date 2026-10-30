//! Checked metadata admission must retain the policy-filtered record or refuse.
#![cfg(feature = "zencodec")]

use img_parts::{ImageEXIF, ImageICC};
use std::{io::Cursor, sync::Arc};
use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zencodec::{Metadata, MetadataPolicy, UnsupportedOperation};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{ColorContext, ColorProfileSource, PixelDescriptor, PixelSlice};
use zenpixels_convert::icc_profiles::ADOBE_RGB;

fn record() -> Metadata {
    let exif = [b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    Metadata::none()
        .with_exif(exif.as_slice())
        .with_xmp(b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".as_slice())
        .with_icc(ADOBE_RGB)
}
fn pixels(data: &[u8]) -> PixelSlice<'_> {
    let (p, t) = ColorProfileSource::Icc(ADOBE_RGB).resolve().unwrap();
    let desc = PixelDescriptor::RGB8_SRGB
        .with_primaries(p)
        .with_transfer(t);
    PixelSlice::new(data, 16, 17, 48, desc)
        .unwrap()
        .with_color_context(Arc::new(ColorContext::from_icc(Arc::<[u8]>::from(
            ADOBE_RGB,
        ))))
}
fn assert_retained(bytes: &[u8], expected: &Metadata) {
    let info = JpegDecoderConfig::new().probe_full_metadata(bytes).unwrap();
    zencodec_testkit::check_metadata_retained(expected, &info.metadata()).unwrap();
    assert_eq!(info.metadata().exif.as_deref(), expected.exif.as_deref());
    assert_eq!(info.metadata().xmp.as_deref(), expected.xmp.as_deref());
    assert_eq!(
        info.metadata().icc_profile.as_deref(),
        expected.icc_profile.as_deref()
    );
    let jpeg = img_parts::jpeg::Jpeg::from_bytes(bytes.to_vec().into()).unwrap();
    if let Some(xmp) = expected.xmp.as_deref() {
        const XMP_APP1_PREFIX: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
        let payloads: Vec<&[u8]> = jpeg
            .segments_by_marker(0xe1)
            .filter_map(|segment| segment.contents().strip_prefix(XMP_APP1_PREFIX))
            .collect();
        assert_eq!(payloads, [xmp], "independent XMP APP1 payload");
    }
    assert_eq!(jpeg.exif().as_deref(), expected.exif.as_deref());
    assert_eq!(
        jpeg.icc_profile().as_deref(),
        expected.icc_profile.as_deref()
    );
    let mut oracle = jpeg_decoder::Decoder::new(Cursor::new(bytes));
    assert_eq!(oracle.decode().unwrap().len(), 16 * 17 * 3);
}
fn independent_samples(bytes: &[u8]) -> Vec<u8> {
    jpeg_decoder::Decoder::new(Cursor::new(bytes))
        .decode()
        .unwrap()
}
#[test]
#[allow(deprecated)] // Compare to the established legacy Verbatim route.
fn checked_record_is_in_real_one_shot_stream_and_dyn_outputs() {
    let data: Vec<u8> = (0..16 * 17 * 3).map(|i| (i * 71) as u8).collect();
    let record = record();
    #[allow(deprecated)]
    let expected = record.filtered(&MetadataPolicy::PreserveExact);
    for progressive in [false, true] {
        let config = JpegEncoderConfig::new()
            .with_progressive(progressive)
            .with_huffman(false);
        let job = || {
            config
                .clone()
                .job()
                .try_with_metadata_policy(record.clone(), MetadataPolicy::PreserveExact)
                .unwrap()
        };
        let legacy = config
            .clone()
            .job()
            .with_metadata_policy(record.clone(), MetadataPolicy::PreserveExact)
            .with_policy(
                zencodec::encode::EncodePolicy::none()
                    .with_color(zencodec::ColorEmitPolicy::Verbatim),
            )
            .encoder()
            .unwrap()
            .encode(pixels(&data))
            .unwrap();
        let expected_samples = independent_samples(legacy.data());
        let direct = job().encoder().unwrap().encode(pixels(&data)).unwrap();
        assert_retained(direct.data(), &expected);
        assert_eq!(
            independent_samples(direct.data()),
            expected_samples,
            "checked one-shot vs legacy Verbatim"
        );
        let mut encoder = job().encoder().unwrap();
        let px = pixels(&data);
        encoder.begin(16, 17, px.descriptor()).unwrap();
        for y in 0..17 {
            encoder.push_rows(px.sub_rows(y, 1)).unwrap();
        }
        let streamed = encoder.finish().unwrap();
        assert_retained(streamed.data(), &expected);
        assert_eq!(
            independent_samples(streamed.data()),
            expected_samples,
            "checked rows vs legacy Verbatim"
        );
        let registry = zencodec::CodecSet::new().with_send_encoder(config.clone());
        let mut dyn_job = registry.encode_job(zencodec::ImageFormat::Jpeg).unwrap();
        dyn_job
            .try_set_metadata_policy(record.clone(), MetadataPolicy::PreserveExact)
            .unwrap();
        let output = dyn_job
            .into_encoder()
            .unwrap()
            .encode(pixels(&data))
            .unwrap();
        assert_retained(output.data(), &expected);
        assert_eq!(
            independent_samples(output.data()),
            expected_samples,
            "checked dyn vs legacy Verbatim"
        );
    }
}
#[test]
fn unsupported_hdr_is_atomic_and_checked_clear_replaces_old_record() {
    use zencodec::{ContentLightLevel, MasteringDisplay};
    use zenpixels::{Cicp, hdr::DiffuseWhite};
    let data = vec![42; 16 * 17 * 3];
    let original = record();
    let mut job = JpegEncoderConfig::new().job();
    job.try_set_metadata_policy(original.clone(), MetadataPolicy::PreserveExact)
        .unwrap();
    let unsupported = [
        Metadata::none().with_content_light_level(ContentLightLevel::new(1000, 400)),
        Metadata::none().with_mastering_display(MasteringDisplay::default()),
        Metadata::none().with_diffuse_white(DiffuseWhite::BT2408),
        Metadata::none().with_cicp(Cicp::new(1, 13, 0, true)),
    ];
    for metadata in unsupported {
        let error = job
            .try_set_metadata_policy(metadata, MetadataPolicy::PreserveExact)
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<UnsupportedOperation>(),
            Some(&UnsupportedOperation::MetadataRetention)
        );
    }
    #[allow(deprecated)]
    let expected = original.filtered(&MetadataPolicy::PreserveExact);
    let output = job
        .clone()
        .encoder()
        .unwrap()
        .encode(pixels(&data))
        .unwrap();
    assert_retained(output.data(), &expected);
    job.try_set_metadata_policy(Metadata::none(), MetadataPolicy::PreserveExact)
        .unwrap();
    let plain = PixelSlice::new(&data, 16, 17, 48, PixelDescriptor::RGB8_SRGB).unwrap();
    let output = job.encoder().unwrap().encode(plain).unwrap();
    let info = JpegDecoderConfig::new()
        .probe_full_metadata(output.data())
        .unwrap();
    assert!(info.metadata().exif.is_none());
    assert!(info.metadata().xmp.is_none());
    assert!(info.metadata().icc_profile.is_none());
}
#[test]
#[allow(deprecated)]
fn later_legacy_setters_cannot_discard_checked_requirements() {
    use zencodec::encode::EncodePolicy;
    let data = vec![12; 16 * 17 * 3];
    let make = || {
        JpegEncoderConfig::new()
            .job()
            .try_with_metadata_policy(record(), MetadataPolicy::PreserveExact)
            .unwrap()
    };
    assert!(make().with_metadata(Metadata::none()).encoder().is_err());
    for policy in [
        EncodePolicy::none().with_embed_exif(false),
        EncodePolicy::none().with_embed_xmp(false),
        EncodePolicy::none().with_embed_icc(false),
    ] {
        let mut encoder = make().with_policy(policy).encoder().unwrap();
        assert!(encoder.begin(16, 17, pixels(&data).descriptor()).is_err());
        assert!(encoder.pending_output().is_empty());
        assert!(
            make()
                .with_policy(policy)
                .encoder()
                .unwrap()
                .encode(pixels(&data))
                .is_err()
        );
    }
}
#[test]
fn unsupported_opaque_payloads_are_rejected_before_mutation() {
    let mut job = JpegEncoderConfig::new().job();
    for metadata in [
        Metadata::none().with_xmp([0xff]),
        Metadata::none().with_xmp(Vec::<u8>::new()),
        Metadata::none().with_icc(Vec::<u8>::new()),
        Metadata::none().with_exif(b"invalid exif".as_slice()),
    ] {
        let error = job
            .try_set_metadata_policy(metadata, MetadataPolicy::PreserveExact)
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<UnsupportedOperation>(),
            Some(&UnsupportedOperation::MetadataRetention)
        );
    }
}

#[test]
fn checked_requests_forward_ordinary_and_send_and_refuse_hdr() {
    use zencodec::ContentLightLevel;
    use zencodec::operation::EncodeRequest;
    use zencodec::{CodecSet, ImageFormat};
    let data = vec![23; 16 * 17 * 3];
    let metadata = record();
    #[allow(deprecated)]
    let expected = metadata.filtered(&MetadataPolicy::PreserveExact);
    let request = EncodeRequest::new(ImageFormat::Jpeg)
        .with_metadata_policy(metadata, MetadataPolicy::PreserveExact);
    let ordinary = CodecSet::new().with_encoder(JpegEncoderConfig::new());
    let job = request.prepare(&ordinary).unwrap().remove(0).into_job();
    let output = job.into_encoder().unwrap().encode(pixels(&data)).unwrap();
    assert_retained(output.data(), &expected);
    let send = CodecSet::new().with_send_encoder(JpegEncoderConfig::new());
    let job = request.prepare_send(&send).unwrap().remove(0).into_job();
    let output = job.into_encoder().unwrap().encode(pixels(&data)).unwrap();
    assert_retained(output.data(), &expected);
    let hdr = EncodeRequest::new(ImageFormat::Jpeg).with_metadata_policy(
        Metadata::none().with_content_light_level(ContentLightLevel::new(1200, 500)),
        MetadataPolicy::PreserveExact,
    );
    assert!(hdr.prepare(&ordinary).is_err());
    assert!(hdr.prepare_send(&send).is_err());
}

#[test]
#[allow(deprecated)]
fn later_dynamic_replacement_and_policy_conflict_refuse_before_output() {
    use zencodec::encode::EncodePolicy;
    use zencodec::{CodecSet, ImageFormat};
    let registry = CodecSet::new().with_send_encoder(JpegEncoderConfig::new());
    let mut job = registry.encode_job(ImageFormat::Jpeg).unwrap();
    job.try_set_metadata_policy(record(), MetadataPolicy::PreserveExact)
        .unwrap();
    job.set_metadata(Metadata::none());
    assert!(job.into_encoder().is_err());
    let mut job = registry.encode_job(ImageFormat::Jpeg).unwrap();
    job.try_set_metadata_policy(record(), MetadataPolicy::PreserveExact)
        .unwrap();
    job.set_policy(EncodePolicy::none().with_embed_xmp(false));
    let mut encoder = job.into_encoder().unwrap();
    let data = vec![0; 16 * 17 * 3];
    assert!(encoder.begin(16, 17, pixels(&data).descriptor()).is_err());
    assert!(encoder.pending_output().is_empty());
}

#[test]
fn prepared_segments_cannot_sneak_past_checked_clear() {
    let mut config = JpegEncoderConfig::new();
    *config.inner_mut() = config
        .inner()
        .clone()
        .with_segments(zenjpeg::encode::EncoderSegments::new().set_xmp("<old/>"));
    let mut job = config.job();
    let error = job
        .try_set_metadata_policy(Metadata::none(), MetadataPolicy::PreserveExact)
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<UnsupportedOperation>(),
        Some(&UnsupportedOperation::MetadataRetention)
    );
}

#[test]
fn checked_exif_xmp_reach_pull_and_srgba_paths() {
    let exif = [b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let metadata = Metadata::none()
        .with_exif(exif.as_slice())
        .with_xmp(b"<fixture/>".as_slice());
    #[allow(deprecated)]
    let expected = metadata.filtered(&MetadataPolicy::PreserveExact);
    let job = || {
        JpegEncoderConfig::new()
            .job()
            .try_with_metadata_policy(metadata.clone(), MetadataPolicy::PreserveExact)
            .unwrap()
    };
    let raw = vec![42u8; 16 * 17 * 3];
    let pulled = job()
        .with_canvas_size(16, 17)
        .encoder()
        .unwrap()
        .encode_from(&mut |y, mut out: zenpixels::PixelSliceMut<'_>| {
            for row in 0..out.rows() {
                let offset = (y + row) as usize * 48;
                out.row_mut(row).copy_from_slice(&raw[offset..offset + 48]);
            }
            out.rows() as usize
        })
        .unwrap();
    assert_retained(pulled.data(), &expected);
    let mut rgba = vec![64u8; 16 * 17 * 4];
    let output = job()
        .encoder()
        .unwrap()
        .encode_srgba8(&mut rgba, true, 16, 17, 16)
        .unwrap();
    assert_retained(output.data(), &expected);
}

#[test]
#[allow(deprecated)]
fn conflicts_refuse_plan_and_pull_before_source_access() {
    use zencodec::encode::{EncodeDeclaration, EncodePolicy};
    let job = JpegEncoderConfig::new()
        .job()
        .try_with_metadata_policy(record(), MetadataPolicy::PreserveExact)
        .unwrap()
        .with_policy(EncodePolicy::none().with_embed_xmp(false));
    let desc = pixels(&vec![0; 16 * 17 * 3]).descriptor();
    assert!(
        job.plan_encode(&EncodeDeclaration::new(16, 17, desc))
            .is_err()
    );
    let mut called = false;
    let result = job
        .with_canvas_size(16, 17)
        .encoder()
        .unwrap()
        .encode_from(&mut |_, _| {
            called = true;
            0
        });
    assert!(result.is_err());
    assert!(!called);
}

#[test]
fn later_conflicting_color_declaration_refuses_before_header() {
    use zencodec::encode::EncodeDeclaration;
    use zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
    let job = JpegEncoderConfig::new()
        .with_huffman(false)
        .job()
        .try_with_metadata_policy(record(), MetadataPolicy::PreserveExact)
        .unwrap();
    let conflicting = ColorContext::from_icc(Arc::<[u8]>::from(DISPLAY_P3_V4));
    let declaration =
        EncodeDeclaration::new(16, 17, PixelDescriptor::RGB8_SRGB).with_color(&conflicting);
    assert!(job.plan_encode(&declaration).is_err());
    let mut encoder = job.encoder().unwrap();
    assert!(encoder.begin_declared(&declaration).is_err());
    assert!(encoder.pending_output().is_empty());
}

#[test]
fn exif_without_required_orientation_tag_is_refused() {
    let exif = [b'I', b'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let metadata = Metadata::none()
        .with_exif(exif.as_slice())
        .with_orientation(zencodec::Orientation::Rotate90);
    let error = JpegEncoderConfig::new()
        .job()
        .try_set_metadata_policy(metadata, MetadataPolicy::PreserveExact)
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<UnsupportedOperation>(),
        Some(&UnsupportedOperation::MetadataRetention)
    );
}

#[test]
#[allow(deprecated)]
fn required_srgb_icc_is_verbatim_by_default_but_explicit_drop_refuses() {
    use zencodec::encode::EncodePolicy;
    use zencodec::{CicpEmission, ColorEmitFields, ColorEmitPolicy, IccRetention};
    let sample = include_bytes!("images/ultrahdr_sample.jpg");
    let input = img_parts::jpeg::Jpeg::from_bytes(sample.as_slice().to_vec().into()).unwrap();
    let icc = input.icc_profile().expect("fixture contains the sRGB ICC");
    assert_eq!(
        ColorProfileSource::Icc(&icc).resolve(),
        Some((
            zenpixels::ColorPrimaries::Bt709,
            zenpixels::TransferFunction::Srgb
        ))
    );
    let meta = Metadata::none().with_icc(icc.as_ref());
    let expected = meta.filtered(&MetadataPolicy::PreserveExact);
    let data = vec![51u8; 16 * 17 * 3];
    let px = || PixelSlice::new(&data, 16, 17, 48, PixelDescriptor::RGB8_SRGB).unwrap();
    let job = || {
        JpegEncoderConfig::new()
            .with_huffman(false)
            .job()
            .try_with_metadata_policy(meta.clone(), MetadataPolicy::PreserveExact)
            .unwrap()
    };
    let legacy = JpegEncoderConfig::new()
        .with_huffman(false)
        .job()
        .with_metadata_policy(meta.clone(), MetadataPolicy::PreserveExact)
        .with_policy(EncodePolicy::none().with_color(ColorEmitPolicy::Verbatim))
        .encoder()
        .unwrap()
        .encode(px())
        .unwrap();
    let expected_samples = independent_samples(legacy.data());
    let direct = job().encoder().unwrap().encode(px()).unwrap();
    assert_retained(direct.data(), &expected);
    assert_eq!(
        independent_samples(direct.data()),
        expected_samples,
        "sRGB checked one-shot vs legacy Verbatim"
    );
    let mut encoder = job().encoder().unwrap();
    encoder.begin(16, 17, PixelDescriptor::RGB8_SRGB).unwrap();
    encoder.push_rows(px()).unwrap();
    let streamed = encoder.finish().unwrap();
    assert_retained(streamed.data(), &expected);
    assert_eq!(
        independent_samples(streamed.data()),
        expected_samples,
        "sRGB checked rows vs legacy Verbatim"
    );
    let drop_icc = ColorEmitPolicy::Custom(ColorEmitFields::new(
        IccRetention::Drop,
        CicpEmission::WhereValidCarrier,
    ));
    let conflicting = EncodePolicy::none().with_color(drop_icc);
    let mut encoder = job().with_policy(conflicting).encoder().unwrap();
    assert!(encoder.begin(16, 17, PixelDescriptor::RGB8_SRGB).is_err());
    assert!(encoder.pending_output().is_empty());
    assert!(
        job()
            .with_policy(conflicting)
            .encoder()
            .unwrap()
            .encode(px())
            .is_err()
    );
}

#[test]
#[allow(deprecated)]
fn required_field_only_orientation_cannot_be_discarded_later() {
    use zencodec::encode::EncodePolicy;
    let job = JpegEncoderConfig::new()
        .with_huffman(false)
        .job()
        .try_with_metadata_policy(
            Metadata::none().with_orientation(zencodec::Orientation::Rotate90),
            MetadataPolicy::PreserveExact,
        )
        .unwrap()
        .with_policy(EncodePolicy::none().with_embed_exif(false));
    let mut encoder = job.encoder().unwrap();
    assert!(encoder.begin(16, 17, PixelDescriptor::RGB8_SRGB).is_err());
    assert!(encoder.pending_output().is_empty());
}
