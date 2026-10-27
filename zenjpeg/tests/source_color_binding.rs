//! The colour carrier written by the zencodec adapter must describe the samples
//! the encoder actually stores.
//!
//! When the caller supplies no colour metadata (the checked `EncodeRequest`
//! path supplies none), the current pixel interpretation (descriptor primaries
//! and transfer plus the `PixelSlice` colour context) is the only description
//! of the pixels. JPEG has no CICP carrier, so a non-sRGB interpretation needs
//! an ICC profile; anything else is a silent relabel. Native transforms change
//! the interpretation: linear input is stored sRGB-encoded, so a profile that
//! described the linear samples no longer applies.
//!
//! Every output is read back with `jpeg-decoder`, an independent reader, for
//! both the samples and the embedded ICC profile.
#![cfg(feature = "zencodec")]

use std::io::Cursor;
use std::sync::Arc;

use zencodec::encode::{EncodeJob as _, EncodePolicy, Encoder as _, EncoderConfig as _};
use zencodec::operation::EncodeRequest;
use zencodec::{CodecSet, ColorEmitPolicy, ImageFormat, Metadata, MetadataPolicy};
use zenjpeg::JpegEncoderConfig;
use zenpixels::{Cicp, ColorContext, ColorPrimaries, PixelDescriptor, PixelSlice};
use zenpixels_convert::icc_profiles::{DISPLAY_P3_V4, SynthesizedIcc, synthesize_icc_for_cicp};

const W: u32 = 48;
const H: u32 = 32;
const PATCH: u32 = 16;

/// One flat 16x16 patch per colour, so 4:2:0 chroma is flat within each patch
/// and the decoded patch centre tracks the stored sample closely.
const COLOURS: [[u8; 3]; 6] = [
    [200, 40, 40],
    [40, 190, 60],
    [50, 60, 210],
    [128, 128, 128],
    [230, 180, 30],
    [90, 160, 210],
];

fn colour_at(x: u32, y: u32) -> [u8; 3] {
    COLOURS[((y / PATCH) * (W / PATCH) + x / PATCH) as usize]
}

fn quality() -> JpegEncoderConfig {
    JpegEncoderConfig::new().with_calibrated_quality(95.0)
}

/// A strided raster: every row carries whole padding pixels, and the first
/// sample is aligned for its channel type.
struct Raster {
    buf: Vec<u8>,
    start: usize,
    stride: usize,
}

impl Raster {
    fn new(bytes_per_pixel: usize, padding_pixels: usize, fill: u8) -> Self {
        let stride = (W as usize + padding_pixels) * bytes_per_pixel;
        let buf = vec![fill; stride * H as usize + 16];
        let start = buf.as_ptr().align_offset(16);
        Self { buf, start, stride }
    }

    fn row(&mut self, y: u32) -> &mut [u8] {
        let at = self.start + y as usize * self.stride;
        &mut self.buf[at..at + self.stride]
    }

    fn slice(&self, descriptor: PixelDescriptor) -> PixelSlice<'_> {
        PixelSlice::new(&self.buf[self.start..], W, H, self.stride, descriptor).unwrap()
    }
}

/// 8-bit RGB raster of the patch image.
fn rgb8() -> Raster {
    let mut raster = Raster::new(3, 3, 0xa5);
    for y in 0..H {
        for x in 0..W {
            raster.row(y)[x as usize * 3..][..3].copy_from_slice(&colour_at(x, y));
        }
    }
    raster
}

/// 16-bit RGB raster holding the same sRGB-encoded values (v * 257).
fn rgb16() -> Raster {
    let mut raster = Raster::new(6, 2, 0x5a);
    for y in 0..H {
        for x in 0..W {
            for (c, v) in colour_at(x, y).into_iter().enumerate() {
                let value = (u16::from(v) * 257).to_ne_bytes();
                raster.row(y)[x as usize * 6 + c * 2..][..2].copy_from_slice(&value);
            }
        }
    }
    raster
}

/// Linear-light f32 RGB raster whose sRGB encoding is the patch image.
fn rgbf32_linear() -> Raster {
    let mut raster = Raster::new(12, 1, 0);
    for y in 0..H {
        for x in 0..W {
            for (c, v) in colour_at(x, y).into_iter().enumerate() {
                let linear = srgb_to_linear(f32::from(v) / 255.0).to_ne_bytes();
                raster.row(y)[x as usize * 12 + c * 4..][..4].copy_from_slice(&linear);
            }
        }
    }
    raster
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn p3(base: PixelDescriptor) -> PixelDescriptor {
    base.with_primaries(ColorPrimaries::DisplayP3)
}

fn synthesized(cicp: Cicp) -> Vec<u8> {
    match synthesize_icc_for_cicp(cicp) {
        SynthesizedIcc::Profile(bytes) => bytes.into_owned(),
        other => panic!("the zencodec feature's icc-db must synthesize {cicp:?}: {other:?}"),
    }
}

/// Decode with the independent reader: RGB24 samples and the embedded ICC.
fn oracle(jpeg: &[u8]) -> (Vec<u8>, Option<Vec<u8>>) {
    let mut decoder = jpeg_decoder::Decoder::new(Cursor::new(jpeg));
    let samples = decoder.decode().expect("independent JPEG decode");
    let info = decoder.info().expect("decoded info");
    assert_eq!((u32::from(info.width), u32::from(info.height)), (W, H));
    assert_eq!(info.pixel_format, jpeg_decoder::PixelFormat::RGB24);
    (samples, decoder.icc_profile())
}

/// Every patch centre decodes to the stored sample value within `tolerance`.
fn assert_samples(samples: &[u8], tolerance: u8, what: &str) {
    for py in 0..H / PATCH {
        for px in 0..W / PATCH {
            let (x, y) = (px * PATCH + PATCH / 2, py * PATCH + PATCH / 2);
            let at = (y * W + x) as usize * 3;
            let got = &samples[at..at + 3];
            let want = colour_at(x, y);
            for c in 0..3 {
                assert!(
                    got[c].abs_diff(want[c]) <= tolerance,
                    "{what}: patch ({px},{py}) decoded {got:?}, stored samples are {want:?}"
                );
            }
        }
    }
}

fn checked(pixels: PixelSlice<'_>, policy: Option<EncodePolicy>) -> Result<Vec<u8>, String> {
    let codecs = CodecSet::new().with_encoder(quality());
    let mut request = EncodeRequest::new(ImageFormat::Jpeg);
    if let Some(policy) = policy {
        request = request.with_policy(policy);
    }
    request
        .encode(&codecs, pixels)
        .map(|outcome| outcome.into_output().into_vec())
        .map_err(|error| format!("{error:?}"))
}

fn with_metadata(meta: Metadata, pixels: PixelSlice<'_>) -> Result<Vec<u8>, String> {
    quality()
        .job()
        .with_metadata_policy(meta, MetadataPolicy::PreserveExact)
        .encoder()
        .and_then(|encoder| encoder.encode(pixels))
        .map(|output| output.into_vec())
        .map_err(|error| format!("{error:?}"))
}

/// Issue 154: Display P3 pixels through the checked request carry no job
/// metadata. The current interpretation must produce the P3 carrier: the
/// pixels' own ICC verbatim, or a synthesized Display P3 profile when the
/// interpretation is CICP or descriptor only. The samples are not converted.
#[test]
fn checked_request_binds_display_p3_from_the_current_pixels() {
    let raster = rgb8();
    let desc = p3(PixelDescriptor::RGB8_SRGB);
    let p3_synth = synthesized(Cicp::DISPLAY_P3);
    let cases: [(&str, Option<ColorContext>, &[u8]); 3] = [
        (
            "context ICC",
            Some(ColorContext::from_icc(DISPLAY_P3_V4)),
            DISPLAY_P3_V4,
        ),
        (
            "context CICP",
            Some(ColorContext::from_cicp(Cicp::DISPLAY_P3)),
            &p3_synth,
        ),
        ("descriptor only", None, &p3_synth),
    ];
    for (what, context, expected_icc) in cases {
        let mut pixels = raster.slice(desc);
        if let Some(context) = context {
            pixels = pixels.with_color_context(Arc::new(context));
        }
        let jpeg = checked(pixels, None).unwrap_or_else(|e| panic!("{what}: {e}"));
        let (samples, icc) = oracle(&jpeg);
        assert_eq!(
            icc.as_deref(),
            Some(expected_icc),
            "{what}: the written carrier must describe Display P3"
        );
        assert_samples(&samples, 4, what);
    }
}

/// `RGB16_SRGB` is advertised as a native encode descriptor. Its samples are
/// sRGB-encoded and must decode to the same values as the 8-bit encode.
#[test]
fn sixteen_bit_srgb_samples_are_not_reinterpreted_as_linear_light() {
    let raster = rgb16();
    let pixels = raster.slice(PixelDescriptor::RGB16_SRGB);
    let jpeg = checked(pixels, None).expect("RGB16_SRGB is a supported descriptor");
    let (samples, icc) = oracle(&jpeg);
    assert_eq!(icc, None, "sRGB needs no carrier");
    assert_samples(&samples, 4, "RGB16_SRGB");
}

/// Linear input is stored sRGB-encoded. A profile describing the linear
/// samples would be a stale carrier after that transform; the written carrier
/// must describe P3 primaries with the sRGB transfer instead.
#[test]
fn linear_input_writes_the_carrier_of_the_stored_samples() {
    let raster = rgbf32_linear();
    let desc = p3(PixelDescriptor::RGBF32_LINEAR);
    let linear_p3 = synthesized(Cicp::new(12, 8, 0, true));
    let p3_synth = synthesized(Cicp::DISPLAY_P3);
    for (what, context) in [
        (
            "context linear ICC",
            Some(ColorContext::from_icc(linear_p3.clone())),
        ),
        (
            "context linear CICP",
            Some(ColorContext::from_cicp(Cicp::new(12, 8, 0, true))),
        ),
        ("descriptor only", None),
    ] {
        let mut pixels = raster.slice(desc);
        if let Some(context) = context {
            pixels = pixels.with_color_context(Arc::new(context));
        }
        let jpeg = checked(pixels, None).unwrap_or_else(|e| panic!("{what}: {e}"));
        let (samples, icc) = oracle(&jpeg);
        assert_ne!(
            icc.as_deref(),
            Some(linear_p3.as_slice()),
            "{what}: stale carrier"
        );
        assert_eq!(icc.as_deref(), Some(p3_synth.as_slice()), "{what}");
        assert_samples(&samples, 4, what);
    }
    // The same pre-transform profile supplied explicitly as metadata is a
    // request to describe samples the encoder does not store: refuse.
    let pixels = raster.slice(desc);
    assert!(with_metadata(Metadata::none().with_icc(linear_p3.clone()), pixels).is_err());
    // An unidentifiable profile cannot be carried across the transform.
    let mut opaque = vec![0u8; 132];
    opaque[..4].copy_from_slice(&132u32.to_be_bytes());
    opaque[16..20].copy_from_slice(b"RGB ");
    opaque[20..24].copy_from_slice(b"XYZ ");
    opaque[36..40].copy_from_slice(b"acsp");
    let pixels = raster
        .slice(desc)
        .with_color_context(Arc::new(ColorContext::from_icc(opaque)));
    assert!(checked(pixels, None).is_err());
}

/// Verbatim emission derives nothing, and JPEG cannot carry CICP. A source
/// described only by CICP or by its descriptor cannot be written verbatim:
/// refuse instead of writing an untagged (sRGB-assumed) file.
#[test]
fn verbatim_policy_refuses_to_drop_the_only_colour_description() {
    let raster = rgb8();
    let desc = p3(PixelDescriptor::RGB8_SRGB);
    let verbatim = Some(EncodePolicy::none().with_color(ColorEmitPolicy::Verbatim));
    let cicp_only = raster
        .slice(desc)
        .with_color_context(Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3)));
    assert!(checked(cicp_only, verbatim).is_err());
    let descriptor_only = raster.slice(desc);
    assert!(checked(descriptor_only, verbatim).is_err());
    let with_icc = raster
        .slice(desc)
        .with_color_context(Arc::new(ColorContext::from_icc(DISPLAY_P3_V4)));
    let jpeg = checked(with_icc, verbatim).expect("ICC context is carried verbatim");
    assert_eq!(oracle(&jpeg).1.as_deref(), Some(DISPLAY_P3_V4));
}

/// Job metadata that contradicts an explicit current interpretation is a
/// stale carrier (for example a source profile kept after a conversion).
/// Refuse rather than relabel. Agreement keeps the metadata carrier.
#[test]
fn metadata_contradicting_the_current_context_refuses() {
    let raster = rgb8();
    let srgb_now = raster
        .slice(PixelDescriptor::RGB8_SRGB)
        .with_color_context(Arc::new(ColorContext::from_cicp(Cicp::SRGB)));
    assert!(with_metadata(Metadata::none().with_icc(DISPLAY_P3_V4), srgb_now).is_err());

    let p3_now = || {
        raster
            .slice(p3(PixelDescriptor::RGB8_SRGB))
            .with_color_context(Arc::new(ColorContext::from_icc(DISPLAY_P3_V4)))
    };
    assert!(with_metadata(Metadata::none().with_cicp(Cicp::SRGB), p3_now()).is_err());

    let agreed = with_metadata(Metadata::none().with_icc(DISPLAY_P3_V4), p3_now()).unwrap();
    let (samples, icc) = oracle(&agreed);
    assert_eq!(icc.as_deref(), Some(DISPLAY_P3_V4));
    assert_samples(&samples, 4, "agreeing metadata");
}

/// A policy that forbids the only possible carrier cannot be honoured for a
/// non-sRGB interpretation.
#[test]
#[allow(deprecated)] // The legacy setter is still a public way to forbid ICC output.
fn forbidding_the_icc_carrier_refuses_wide_gamut_pixels() {
    let raster = rgb8();
    let pixels = raster.slice(p3(PixelDescriptor::RGB8_SRGB));
    let encoder = quality()
        .job()
        .with_policy(EncodePolicy::none().with_embed_icc(false))
        .encoder()
        .unwrap();
    assert!(encoder.encode(pixels).is_err());
}

/// XYB converts through a fixed sRGB-primaries transform and writes its own
/// profile. It must not accept pixels with another interpretation.
#[test]
fn xyb_refuses_a_non_srgb_interpretation() {
    let raster = rgb8();
    let xyb = || {
        let mut config = quality();
        *config.inner_mut() =
            zenjpeg::encode::EncoderConfig::xyb(90.0, zenjpeg::encode::XybSubsampling::Full);
        config
    };
    let encode = |pixels| xyb().job().encoder().unwrap().encode(pixels);
    let described = raster.slice(p3(PixelDescriptor::RGB8_SRGB));
    assert!(encode(described).is_err());
    let context = raster
        .slice(PixelDescriptor::RGB8_SRGB)
        .with_color_context(Arc::new(ColorContext::from_icc(DISPLAY_P3_V4)));
    assert!(encode(context).is_err());
    let srgb = raster.slice(PixelDescriptor::RGB8_SRGB);
    assert!(encode(srgb).is_ok());
}

/// Default sRGB pixels with no colour description keep their exact bytes
/// whether they arrive one-shot, as an explicit sRGB context, or row by row.
#[test]
fn default_srgb_bytes_are_identical_across_routes() {
    let raster = rgb8();
    let pixels = || raster.slice(PixelDescriptor::RGB8_SRGB);
    let plain = checked(pixels(), None).unwrap();
    let explicit = checked(
        pixels().with_color_context(Arc::new(ColorContext::from_cicp(Cicp::SRGB))),
        None,
    )
    .unwrap();
    assert_eq!(plain, explicit);
    let mut rows = quality().job().encoder().unwrap();
    rows.begin(W, H, PixelDescriptor::RGB8_SRGB).unwrap();
    for y in (0..H).step_by(7) {
        rows.push_rows(pixels().sub_rows(y, 7.min(H - y))).unwrap();
    }
    assert_eq!(rows.finish().unwrap().into_vec(), plain);
    assert_eq!(oracle(&plain).1, None);
}
