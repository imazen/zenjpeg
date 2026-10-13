#![cfg(feature = "zencodec")]
use std::borrow::Cow;
use zencodec::decode::{Decode, DecodeJob, DecodePolicy, DecoderConfig};
use zencodec::encode::{EncodeJob, EncodePolicy, EncoderConfig};
use zencodec::{Metadata, MetadataPolicy};
use zencodec_testkit as tk;
use zenjpeg::{JpegDecoderConfig as D, JpegEncoderConfig as E};

#[test]
fn policy_and_inventory_contracts() {
    let img = tk::TestImage::rgb8_gradient(16, 16);
    let enc = E::new();
    let metadata = Metadata::none().with_xmp(b"<private>SECRET</private>".as_slice());
    let bytes = enc
        .clone()
        .job()
        .with_metadata_policy(metadata.clone(), MetadataPolicy::PreserveExact)
        .encode(img.as_slice())
        .unwrap();
    let expected = D::new()
        .job()
        .decoder(Cow::Borrowed(bytes.data()), &[])
        .unwrap()
        .decode()
        .unwrap()
        .metadata();
    assert_eq!(
        expected.xmp, metadata.xmp,
        "fixture must exercise real metadata"
    );
    for (policy, accepted) in [
        (EncodePolicy::none(), true),
        (EncodePolicy::strip_all(), false),
    ] {
        tk::check_encode_policy_admission(
            enc.clone(),
            D::new(),
            img.as_slice(),
            metadata.clone(),
            EncodePolicy::none(),
            policy,
            accepted,
            &expected,
        )
        .unwrap();
    }
    for (policy, accepted) in [
        (DecodePolicy::none(), true),
        (DecodePolicy::strict(), false),
    ] {
        tk::check_decode_policy_admission(
            D::new(),
            bytes.data(),
            DecodePolicy::none(),
            policy,
            accepted,
            &expected,
        )
        .unwrap();
    }
    tk::check_inventory_dispatch(D::new(), bytes.data()).unwrap();
    #[cfg(feature = "inventory")]
    tk::check_inventory_input_limit(D::new(), bytes.data()).unwrap();
}

#[cfg(feature = "metadata-rewrite")]
#[test]
fn rewrite_preserves_independently_decoded_pixels_and_requested_xmp() {
    let img = tk::TestImage::rgb8_gradient(16, 16);
    let bytes = E::new()
        .job()
        .with_metadata_policy(
            Metadata::none().with_xmp(b"<private>SECRET</private>".as_slice()),
            MetadataPolicy::PreserveExact,
        )
        .encode(img.as_slice())
        .unwrap();
    let pixels = |data: &[u8]| {
        jpeg_decoder::Decoder::new(std::io::Cursor::new(data))
            .decode()
            .unwrap()
    };
    let before = pixels(bytes.data());
    for (policy, keep) in [
        (MetadataPolicy::ColorAndRotation, false),
        (MetadataPolicy::PreserveExact, true),
    ] {
        tk::check_metadata_rewrite(
            zencodec::metadata_rewrite::Request::new(bytes.data()).with_policy(policy),
            zenjpeg::metadata::rewrite,
            |out| {
                assert_eq!(pixels(out), before);
                assert_eq!(out.windows(6).any(|w| w == b"SECRET"), keep);
                Ok(())
            },
        )
        .unwrap();
    }
}
