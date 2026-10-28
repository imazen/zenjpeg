//! A field-only orientation (`Metadata::orientation` with no EXIF blob, as
//! formats whose orientation is a container property and callers produce)
//! must survive JPEG encoding: JPEG carries orientation only in EXIF, so the
//! encoder writes a minimal EXIF holding the tag. A policy that forbids EXIF
//! refuses before any output instead of displaying the pixels re-oriented.
#![cfg(feature = "zencodec")]

use std::borrow::Cow;

use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zencodec::encode::{EncodeJob, EncodePolicy, Encoder, EncoderConfig};
use zencodec::{
    At, CodecError, ErrorCategory, Metadata, MetadataPolicy, Orientation, OrientationHint,
    PolicyKind,
};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};

const W: u32 = 24;
const H: u32 = 16;

fn encode(meta: Metadata, policy: Option<EncodePolicy>) -> Result<Vec<u8>, At<CodecError>> {
    let data: Vec<u8> = (0..W * H * 3).map(|i| (i * 13 % 251) as u8).collect();
    let pixels = PixelSlice::new(&data, W, H, W as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let mut job = JpegEncoderConfig::new()
        .job()
        .with_metadata_policy(meta, MetadataPolicy::PreserveExact);
    if let Some(policy) = policy {
        // zenjpeg's checked setter qualifies colour only; EXIF embedding is
        // still configured through the legacy setter.
        #[allow(deprecated)]
        {
            job = job.with_policy(policy);
        }
    }
    Ok(job.encoder()?.encode(pixels)?.into_vec())
}

/// The EXIF Orientation tag as two independent readers see it: jpeg-decoder
/// extracts the APP1 TIFF payload and kamadak-exif parses it.
fn exif_orientation(jpeg: &[u8]) -> Option<u32> {
    let mut decoder = jpeg_decoder::Decoder::new(jpeg);
    decoder.decode().unwrap();
    let tiff = decoder.exif_data()?.to_vec();
    let exif = exif::Reader::new().read_raw(tiff).unwrap();
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?
        .value
        .get_uint(0)
}

fn stored_orientation(jpeg: &[u8]) -> Orientation {
    JpegDecoderConfig::new()
        .job()
        .with_orientation(OrientationHint::Preserve)
        .decoder(Cow::Borrowed(jpeg), &[])
        .unwrap()
        .decode()
        .unwrap()
        .info()
        .orientation
}

#[test]
fn a_field_only_orientation_is_written_as_exif() {
    for value in 2..=8u8 {
        let orientation = Orientation::from_exif(value).unwrap();
        let jpeg = encode(Metadata::none().with_orientation(orientation), None).unwrap();
        assert_eq!(
            exif_orientation(&jpeg),
            Some(u32::from(value)),
            "{orientation:?}"
        );
        assert_eq!(stored_orientation(&jpeg), orientation);
    }
}

#[test]
fn identity_without_exif_writes_no_exif() {
    let jpeg = encode(Metadata::none(), None).unwrap();
    assert_eq!(exif_orientation(&jpeg), None);
    assert_eq!(stored_orientation(&jpeg), Orientation::Identity);
}

#[test]
fn a_policy_forbidding_exif_refuses_an_orientation_it_would_drop() {
    let no_exif = EncodePolicy::none().with_embed_exif(false);
    let error = encode(
        Metadata::none().with_orientation(Orientation::Rotate90),
        Some(no_exif),
    )
    .unwrap_err();
    assert_eq!(
        error.error().category(),
        ErrorCategory::Policy(PolicyKind::Encode),
        "{error:?}"
    );
    // Identity needs no carrier, so the same policy still encodes.
    let jpeg = encode(Metadata::none(), Some(no_exif)).unwrap();
    assert_eq!(exif_orientation(&jpeg), None);
}
