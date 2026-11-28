//! codec trait-impl tests (split from the old monolithic codec.rs).

#![allow(unused_imports)]

use alloc::borrow::Cow;
use alloc::vec::Vec;

use rgb::{Gray, Rgb};
use whereat::At;
use zencodec::decode::{DecodeCapabilities, DecodeOutput, OutputInfo};
use zencodec::encode::{EncodeCapabilities, EncodeOutput};
use zencodec::{
    CodecError, ImageFormat, ImageInfo, Metadata, ResourceLimits, Unsupported, UnsupportedOperation,
};
use zenpixels::{PixelBuf, PixelFormat as ZenPixelFormat, PixelView, PixelViewMut};

use crate::encode::encoder_config::EncoderConfig;
use crate::encode::encoder_types::{ChromaSubsampling, PixelLayout, Quality};
use crate::encode::exif::Exif;
use crate::error::{Error, ErrorKind};
use crate::types::PixelFormat;

use super::decode::*;
use super::encode::*;
use super::info::*;
use super::streaming::*;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod api_tests {
    use super::*;
    use alloc::borrow::Cow;
    use imgref::{Img, ImgExt};
    use rgb::{Gray, Rgb, Rgba};
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};

    fn view<P: zenpixels::PixelType>(img: imgref::ImgRef<'_, P>) -> PixelView<'_> {
        let spec = zencodec::helpers::codec_spec(P::LAYOUT).with_transfer(
            if P::LAYOUT.sample == zenpixels::SampleType::U8 {
                zenpixels::Transfer::Srgb
            } else {
                zenpixels::Transfer::Linear
            },
        );
        PixelView::new_with_format_cow(
            bytemuck::cast_slice(img.buf()),
            img.width() as u32,
            img.height() as u32,
            img.stride() * P::LAYOUT.bytes_per_pixel(),
            zenpixels::FormatCow::from_owned(spec),
        )
        .unwrap()
    }

    #[test]
    fn encoding_default_roundtrip() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(80.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(view(img.as_ref())).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encoding_with_metadata() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels.as_slice(), 4, 4);

        let icc = zenpixels_convert::icc_profiles::DISPLAY_P3_V4;
        let meta = Metadata::default().with_icc(icc);
        let output = enc
            .job()
            .with_unfiltered_metadata(meta.filtered(&zencodec::MetadataPolicy::PreserveExact))
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .unwrap();
        let mut independent = jpeg_decoder::Decoder::new(output.data());
        independent.decode().unwrap();
        assert_eq!(independent.icc_profile().as_deref(), Some(icc));
    }

    #[test]
    fn encoding_with_policy_strips_metadata() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels.as_slice(), 4, 4);

        // XMP/EXIF are stripped cleanly under strip_all.
        let xmp = b"<x:xmpmeta/>";
        let meta = Metadata::default().with_xmp(xmp.as_slice());
        let policy = zencodec::encode::EncodePolicy::strip_all();

        let output = enc
            .clone()
            .job()
            .with_unfiltered_metadata(meta.filtered(&zencodec::MetadataPolicy::PreserveExact))
            .try_with_policy(policy)
            .unwrap()
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .unwrap();
        assert!(!output.data().is_empty());
        assert!(
            !output.data().windows(xmp.len()).any(|w| w == xmp),
            "strip_all must omit XMP"
        );

        // Unidentified ICC cannot be silently dropped under checked policy.
        let icc = b"fake icc profile data";
        let meta = Metadata::default().with_icc(icc.as_slice());
        let err = enc
            .job()
            .with_unfiltered_metadata(meta.filtered(&zencodec::MetadataPolicy::PreserveExact))
            .try_with_policy(policy)
            .unwrap()
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .unwrap_err();
        assert_eq!(
            err.error().category(),
            zencodec::ErrorCategory::Policy(zencodec::PolicyKind::Encode)
        );
    }

    #[test]
    fn encoding_gray8() {
        let enc = JpegEncoderConfig::grayscale(90.0);
        let pixels = vec![Gray::new(128u8); 64];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(view(img.as_ref())).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
    }

    #[test]
    fn encoding_rgba8_strips_alpha() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgba<u8>> = vec![
            Rgba {
                r: 100,
                g: 150,
                b: 200,
                a: 128,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(view(img.as_ref())).unwrap();
        assert!(!output.data().is_empty());
    }

    #[test]
    fn push_rows_encode() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            8 * 8
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let slice: PixelView<'_> = view(img.as_ref());

        let mut encoder = enc.job().encoder().unwrap();
        let top = slice.sub_rows(0, 4).unwrap();
        let bottom = slice.sub_rows(4, 4).unwrap();
        encoder.push_rows(top).unwrap();
        encoder.push_rows(bottom).unwrap();
        let output = encoder.finish().unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn effort_levels() {
        let enc = JpegEncoderConfig::new()
            .with_generic_quality(85.0)
            .with_generic_effort(0); // Fast
        assert_eq!(enc.generic_effort(), Some(0));

        let enc = enc.with_generic_effort(2); // Max
        assert_eq!(enc.generic_effort(), Some(2));

        // Fleet accept-signal round-trip: `generic_effort()` must echo back
        // exactly what was set, even out-of-tier — clamping only applies at
        // point-of-use in `effective_config()`, not to the stored/reported
        // value. (Was: incorrectly clamped to `Some(2)` here, which broke
        // callers relying on set-then-get to confirm accepted input.)
        let enc = enc.with_generic_effort(99);
        assert_eq!(enc.generic_effort(), Some(99));

        let enc = enc.with_generic_effort(-7);
        assert_eq!(enc.generic_effort(), Some(-7));
    }

    #[test]
    fn xyb_decoded_descriptor_is_srgb_while_source_icc_is_retained() {
        let pixels = vec![Rgb::new(180u8, 95, 42); 16 * 16];
        let encoded =
            crate::encode::EncoderConfig::xyb(90, crate::encode::XybSubsampling::BQuarter)
                .encode(&pixels, 16, 16)
                .unwrap();
        let mut header = crate::decode::Decoder::new().read_info(&encoded).unwrap();
        assert!(header.is_xyb);
        assert!(header.icc_profile.is_some());
        assert_eq!(
            decode_descriptor(&[], &header, None).unwrap(),
            ZenPixelFormat::RGB8_SRGB
        );
        assert_eq!(
            decode_descriptor(&[ZenPixelFormat::RGBAF32_LINEAR], &header, None).unwrap(),
            ZenPixelFormat::RGBAF32_LINEAR
        );
        let decoded = JpegDecoderConfig::new().decode(&encoded).unwrap();
        assert_eq!(decoded.pixels().format().clone(), ZenPixelFormat::RGB8_SRGB);
        assert_eq!(
            decoded.info().source_color.icc_profile.as_deref(),
            header.icc_profile.as_deref()
        );
        // An arbitrary unconverted ICC must not receive the XYB exception.
        header.is_xyb = false;
        header.icc_profile = Some(b"invalid unconverted ICC".to_vec());
        assert!(matches!(
            decode_descriptor(&[], &header, None).unwrap().color,
            zenpixels::ColorSpace::Rgb {
                transfer: zenpixels::Transfer::Unknown,
                ..
            }
        ));
    }

    /// A JPEG with an EXIF APP1 segment (copyright only, so no orientation
    /// handling rewrites it), and that segment's TIFF structure.
    fn jpeg_with_exif() -> (Vec<u8>, Vec<u8>) {
        let tiff = Exif::build()
            .copyright("zen exif round trip")
            .to_bytes()
            .unwrap();
        assert!(tiff.starts_with(b"II") || tiff.starts_with(b"MM"));
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 120,
                g: 60,
                b: 30,
            };
            64
        ];
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
        let mut enc = config
            .request()
            .exif(Exif::raw(tiff.clone()))
            .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(bytemuck::cast_slice(&pixels), enough::Unstoppable)
            .unwrap();
        (enc.finish().unwrap(), tiff)
    }

    fn exif_headers(jpeg: &[u8]) -> usize {
        jpeg.windows(6).filter(|w| w == b"Exif\0\0").count()
    }

    #[test]
    #[allow(deprecated)] // Exercises the legacy metadata setter.
    fn exif_crosses_the_zencodec_boundary_as_tiff_and_round_trips() {
        let (source, tiff) = jpeg_with_exif();
        assert_eq!(exif_headers(&source), 1);
        let dec = JpegDecoderConfig::new();
        // Decode and probe both hand out the TIFF structure.
        let decoded = dec.decode(&source).unwrap();
        assert_eq!(decoded.info().metadata().exif.as_deref(), Some(&tiff[..]));
        let probed = dec.probe_header(&source).unwrap();
        assert_eq!(probed.metadata().exif.as_deref(), Some(&tiff[..]));

        // JPEG -> JPEG through zencodec: one header, the same TIFF.
        let meta = decoded.info().metadata();
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 1, g: 2, b: 3 }; 64];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let reencoded = enc
            .clone()
            .job()
            .with_unfiltered_metadata(meta.filtered(&zencodec::MetadataPolicy::PreserveExact))
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .unwrap();
        assert_eq!(exif_headers(reencoded.data()), 1);
        assert_eq!(
            dec.decode(reencoded.data())
                .unwrap()
                .info()
                .metadata()
                .exif
                .as_deref(),
            Some(&tiff[..])
        );

        // A blob that still carries the APP1 header (from an older decoder)
        // is written with one header too.
        let prefixed = [b"Exif\0\0".as_slice(), &tiff].concat();
        let meta = Metadata::default().with_exif(prefixed);
        let reencoded = enc
            .clone()
            .job()
            .with_unfiltered_metadata(meta.filtered(&zencodec::MetadataPolicy::PreserveExact))
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .unwrap();
        assert_eq!(exif_headers(reencoded.data()), 1);
        assert_eq!(
            dec.decode(reencoded.data())
                .unwrap()
                .info()
                .metadata()
                .exif
                .as_deref(),
            Some(&tiff[..])
        );
    }

    #[test]
    fn decode_roundtrip() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let output = dec.decode(encoded.data()).unwrap();
        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
        assert_eq!(output.info().format, ImageFormat::Jpeg);
    }

    #[test]
    fn decode_zero_copy_rgb8() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap();
        // Output should be RGB8 — the native format
        assert_eq!(output.pixels().format().clone(), ZenPixelFormat::RGB8_SRGB);
        let pixel_data = output.pixels();
        assert_eq!(pixel_data.width(), 8);
        assert_eq!(pixel_data.height(), 8);
        // 8*8*3 = 192 bytes
        assert!(pixel_data.as_contiguous_bytes().is_some());
    }

    #[test]
    fn probe_info() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 0, g: 0, b: 0 }; 100];
        let img = Img::new(pixels.as_slice(), 10, 10);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let info = dec.probe_header(encoded.data()).unwrap();
        assert_eq!(info.width, 10);
        assert_eq!(info.height, 10);
        assert_eq!(info.format, ImageFormat::Jpeg);
    }

    #[test]
    fn streaming_decode_roundtrip() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            16 * 16
        ];
        let img = Img::new(pixels.as_slice(), 16, 16);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap();

        assert_eq!(stream.info().width, 16);
        assert_eq!(stream.info().height, 16);

        let mut total_rows = 0u32;
        while let Some((y, batch)) = stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            assert_eq!(batch.width(), 16);
            // Each batch should be MCU-row sized (multiple rows)
            assert!(batch.height() >= 1);
            total_rows += batch.height();
        }
        assert_eq!(total_rows, 16);
    }

    #[test]
    fn streaming_decode_batches_mcu_rows() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Create a larger image to see MCU batching
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            64 * 64
        ];
        let img = Img::new(pixels.as_slice(), 64, 64);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap();

        let mut batch_count = 0;
        let mut total_rows = 0u32;
        while let Some((_y, batch)) = stream.next_batch().unwrap() {
            batch_count += 1;
            total_rows += batch.height();
        }
        assert_eq!(total_rows, 64);
        // With MCU batching, we should have fewer batches than rows
        // (64 rows / 16 rows per MCU = ~4 batches for 4:2:0)
        assert!(
            batch_count < 64,
            "expected MCU-row batching, got {batch_count} batches for 64 rows"
        );
    }

    #[test]
    fn streaming_decode_cow_owned() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Encode a test image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(95.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        // First decode with Cow::Borrowed as reference
        let dec = JpegDecoderConfig::new();
        let mut borrowed_stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap();

        let mut borrowed_pixels = Vec::new();
        while let Some((_y, batch)) = borrowed_stream.next_batch().unwrap() {
            borrowed_pixels.extend_from_slice(batch.as_strided_bytes());
        }

        // Now decode with Cow::Owned — the key test
        let owned_data = encoded.data().to_vec();
        let dec2 = JpegDecoderConfig::new();
        let mut owned_stream = dec2
            .job()
            .streaming_decoder(Cow::Owned(owned_data), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap();

        assert_eq!(owned_stream.info().width, 32);
        assert_eq!(owned_stream.info().height, 32);

        let mut owned_pixels = Vec::new();
        let mut total_rows = 0u32;
        while let Some((y, batch)) = owned_stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            owned_pixels.extend_from_slice(batch.as_strided_bytes());
            total_rows += batch.height();
        }
        assert_eq!(total_rows, 32);

        // Owned and borrowed paths must produce identical output
        assert_eq!(
            owned_pixels, borrowed_pixels,
            "Cow::Owned output differs from Cow::Borrowed"
        );
    }

    #[test]
    fn streaming_decode_cow_owned_is_effectively_static() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _, StreamingDecode as _};

        // Encode a test image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            16 * 16
        ];
        let img = Img::new(pixels.as_slice(), 16, 16);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        // Create streaming decoder with owned data inside a scope,
        // then use it outside that scope (proves no external borrow).
        let owned_data = encoded.data().to_vec();
        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Owned(owned_data), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap();

        // The stream should work after the owned_data variable is consumed
        let mut total_rows = 0u32;
        while let Some((_y, batch)) = stream.next_batch().unwrap() {
            total_rows += batch.height();
        }
        assert_eq!(total_rows, 16);
    }

    // ── Encoder trait roundtrip tests ────────────────────────────────

    fn encoder_trait_roundtrip(pixels: zenpixels::PixelView<'_>) {
        use zencodec::encode::Encoder;
        let config = JpegEncoderConfig::new().with_calibrated_quality(75.0);
        let encoder = config.job().encoder().unwrap();
        let output = encoder.encode(pixels).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encoder_trait_rgb8() {
        let pixels: Vec<Rgb<u8>> = (0..16 * 16)
            .map(|i| Rgb {
                r: (i % 256) as u8,
                g: ((i * 3) % 256) as u8,
                b: ((i * 7) % 256) as u8,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_rgba8() {
        let pixels: Vec<Rgba<u8>> = (0..16 * 16)
            .map(|i| Rgba {
                r: (i % 256) as u8,
                g: 128,
                b: 64,
                a: 255,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_gray8() {
        let pixels: Vec<Gray<u8>> = (0..16 * 16).map(|i| Gray((i % 256) as u8)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_rgb16() {
        let pixels: Vec<Rgb<u16>> = (0..16 * 16)
            .map(|i| Rgb {
                r: (i * 256) as u16,
                g: ((i * 3 * 256) % 65536) as u16,
                b: 0,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_rgba16() {
        let pixels: Vec<Rgba<u16>> = (0..16 * 16)
            .map(|i| Rgba {
                r: (i * 256) as u16,
                g: 32768,
                b: 16384,
                a: 65535,
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_gray16() {
        let pixels: Vec<Gray<u16>> = (0..16 * 16).map(|i| Gray((i * 256) as u16)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_rgb_f32() {
        let pixels: Vec<Rgb<f32>> = (0..16 * 16)
            .map(|i| {
                let t = i as f32 / 255.0;
                Rgb {
                    r: t,
                    g: t * 0.5,
                    b: t * 0.25,
                }
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_rgba_f32() {
        let pixels: Vec<Rgba<f32>> = (0..16 * 16)
            .map(|i| {
                let t = i as f32 / 255.0;
                Rgba {
                    r: t,
                    g: t * 0.5,
                    b: t * 0.25,
                    a: 1.0,
                }
            })
            .collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_gray_f32() {
        let pixels: Vec<Gray<f32>> = (0..16 * 16).map(|i| Gray(i as f32 / 255.0)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(view(img.as_ref()));
    }

    #[test]
    fn encoder_trait_dyn_encoder() {
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 100,
                g: 150,
                b: 200,
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let config = JpegEncoderConfig::new().with_calibrated_quality(80.0);
        let dyn_enc = config.job().dyn_encoder().unwrap();
        let output = dyn_enc.encode(view(img.as_ref())).unwrap();
        assert!(!output.is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
    }

    #[test]
    fn capabilities_encode() {
        use zencodec::encode::EncoderConfig;
        let caps = JpegEncoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.stop());
        assert!(caps.lossy());
        assert!(!caps.lossless());
        assert!(!caps.animation());
        assert!(caps.push_rows());
        assert!(caps.encode_from());
        assert!(caps.native_gray());
        assert!(caps.native_16bit());
        assert!(caps.native_f32());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert!(caps.quality_range().is_some());
        assert!(caps.effort_range().is_some());
    }

    #[test]
    fn fidelity_native_targets_roundtrip() {
        use zencodec::encode::{EncoderConfig, Fidelity};

        // JPEG honors all three lossy targets natively, so each round-trips
        // through `resolved_target_fidelity` as itself.
        let ssim2 = JpegEncoderConfig::new().with_fidelity(Fidelity::ssim2(90.0));
        assert_eq!(
            ssim2.resolved_target_fidelity(),
            Some(Fidelity::ssim2(90.0))
        );

        let butter = JpegEncoderConfig::new().with_fidelity(Fidelity::butteraugli(1.5));
        assert_eq!(
            butter.resolved_target_fidelity(),
            Some(Fidelity::butteraugli(1.5))
        );

        let cq = JpegEncoderConfig::new().with_fidelity(Fidelity::codec_quality(70.0));
        assert_eq!(
            cq.resolved_target_fidelity(),
            Some(Fidelity::codec_quality(70.0))
        );
    }

    #[test]
    fn fidelity_lossless_is_best_effort_not_lossless() {
        use zencodec::encode::{EncoderConfig, Fidelity};

        // JPEG has no lossless codestream: a lossless request resolves to the
        // top of the quality dial and is reported as lossy, never `Lossless`.
        let cfg = JpegEncoderConfig::new().with_fidelity(Fidelity::Lossless);
        let resolved = cfg.resolved_target_fidelity();
        assert_eq!(resolved, Some(Fidelity::codec_quality(100.0)));
        assert!(!resolved.unwrap().is_lossless());
        assert!(!JpegEncoderConfig::capabilities().lossless());
    }

    #[test]
    fn capabilities_decode() {
        use zencodec::decode::DecoderConfig;
        let caps = JpegDecoderConfig::capabilities();
        assert!(caps.icc());
        assert!(caps.exif());
        assert!(caps.xmp());
        assert!(caps.stop());
        assert!(caps.cheap_probe());
        assert!(caps.streaming());
        assert!(caps.native_gray());
        assert!(caps.native_f32());
        assert!(caps.enforces_max_pixels());
        assert!(caps.enforces_max_memory());
        assert!(caps.enforces_max_input_bytes());
        assert!(!caps.animation());
    }

    #[test]
    fn decode_trait_max_width_enforced() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = JpegEncoderConfig::new().encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let limits = ResourceLimits::none().with_max_width(10);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode();
        assert!(result.is_err(), "should reject image wider than max_width");
    }

    #[test]
    fn decode_trait_max_height_enforced() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = JpegEncoderConfig::new().encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let limits = ResourceLimits::none().with_max_height(10);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode();
        assert!(
            result.is_err(),
            "should reject image taller than max_height"
        );
    }

    #[test]
    fn decode_trait_generous_dimensions_ok() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32
            };
            32 * 32
        ];
        let img = Img::new(pixels.as_slice(), 32, 32);
        let encoded = JpegEncoderConfig::new().encode(view(img.as_ref())).unwrap();

        let dec = JpegDecoderConfig::new();
        let limits = ResourceLimits::none()
            .with_max_width(1000)
            .with_max_height(1000);
        let result = dec
            .job()
            .with_limits(limits)
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode();
        assert!(
            result.is_ok(),
            "generous limits should not reject 32x32 image"
        );
    }

    #[test]
    fn animation_frame_encoder_returns_unsupported() {
        let config = JpegEncoderConfig::new();
        let result = config.job().animation_frame_encoder();
        assert!(result.is_err());
    }

    #[test]
    fn animation_frame_decoder_returns_unsupported() {
        use zencodec::decode::{DecodeJob as _, DecoderConfig as _};

        let dec = JpegDecoderConfig::new();
        let result = dec.job().animation_frame_decoder(Cow::Borrowed(&[]), &[]);
        assert!(result.is_err());
    }

    /// The encode memory pre-flight gates on the CALIBRATED peak estimate
    /// (`heuristics::estimate_encode` working set + the held input buffer),
    /// not just the raw `w*h*bpp` input buffer. 1024×1024 RGB8: the input
    /// buffer is 3 MiB — the old input-buffer check ADMITTED it under a
    /// 4 MiB cap — but the calibrated peak (input + ~1.5 MB fixed overhead +
    /// the ≥3.85 B/px working set) is well past the budget. The honest check
    /// must reject up front with `LimitKind::Memory`.
    #[test]
    fn encode_memory_preflight_rejects_calibrated_peak_over_budget() {
        use zencodec::{ErrorCategory, LimitKind, ResourceError};

        let cap: u64 = 4 * 1024 * 1024;
        let (w, h) = (1024usize, 1024usize);
        let input_bytes = (w * h * 3) as u64;
        assert!(
            input_bytes < cap,
            "input buffer must fit the cap (the old check admitted this size)"
        );

        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            w * h
        ];
        let img = Img::new(pixels.as_slice(), w, h);
        let limits = ResourceLimits::none().with_max_memory(cap);
        let err = JpegEncoderConfig::new()
            .job()
            .with_limits(limits)
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .expect_err("calibrated peak must exceed the 4 MiB cap");
        assert_eq!(
            err.error().category(),
            ErrorCategory::Resource(ResourceError::Limits(LimitKind::Memory)),
            "rejection must be the memory-limit path, got: {err}"
        );
    }

    /// A budget that covers the calibrated peak admits the encode and it
    /// completes (moderate 64 MiB cap for a 1024×1024 RGB8 encode whose
    /// modeled peak is ~9-12 MiB).
    #[test]
    fn encode_memory_preflight_admits_within_budget() {
        let (w, h) = (1024usize, 1024usize);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 128,
                g: 64,
                b: 32,
            };
            w * h
        ];
        let img = Img::new(pixels.as_slice(), w, h);
        let limits = ResourceLimits::none().with_max_memory(64 * 1024 * 1024);
        let out = JpegEncoderConfig::new()
            .job()
            .with_limits(limits)
            .encoder()
            .unwrap()
            .encode(view(img.as_ref()))
            .expect("64 MiB budget must admit a 1 MP encode");
        assert!(!out.data().is_empty());
        assert_eq!(&out.data()[0..2], &[0xFF, 0xD8]);
    }

    /// Regression test: passing the full `supported_descriptors()` list (which
    /// includes f32 types like RGBF32_LINEAR) to `decoder()` must not panic.
    ///
    /// Previously, the f32-to-u8 conversion used `bytemuck::cast_vec::<f32, u8>()`
    /// which requires identical alignment (f32=4, u8=1 — always panics with
    /// AlignmentMismatch).
    #[test]
    fn decode_with_full_descriptor_list_no_alignment_panic() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

        // Encode a small RGB image
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            64
        ];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let encoded = enc.encode(view(img.as_ref())).unwrap();

        // Use the full supported descriptor list (includes f32 types)
        let dec = JpegDecoderConfig::new();
        let preferred = JpegDecoderConfig::supported_pixel_formats();

        // This must not panic — previously hit bytemuck AlignmentMismatch
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), preferred)
            .unwrap()
            .decode()
            .unwrap();

        assert_eq!(output.info().width, 8);
        assert_eq!(output.info().height, 8);
    }

    #[test]
    fn encode_from_pull_basic() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        use zencodec::encode::{EncodeJob as _, Encoder as _};
        use zenpixels::PixelViewMut;

        let width = 32u32;
        let height = 32u32;
        let bpp = 3; // RGB8

        // Generate test pattern: horizontal gradient
        let row_bytes = width as usize * bpp;
        let total_bytes = row_bytes * height as usize;
        let mut src_pixels = alloc::vec![0u8; total_bytes];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = y * row_bytes + x * bpp;
                src_pixels[offset] = (x * 255 / 31) as u8; // R
                src_pixels[offset + 1] = (y * 255 / 31) as u8; // G
                src_pixels[offset + 2] = 128; // B
            }
        }

        let config = JpegEncoderConfig::new().with_generic_quality(85.0);
        let job = config.job().with_canvas_size(width, height);
        let encoder = job.encoder().unwrap();

        let encoded = encoder
            .encode_from(&mut |y, mut buf: PixelViewMut<'_>| {
                let rows = buf.height();
                for row in 0..rows {
                    let src_y = y + row;
                    if src_y >= height {
                        return row as usize;
                    }
                    let src_start = src_y as usize * row_bytes;
                    let src_end = src_start + row_bytes;
                    buf.row_mut(row)
                        .copy_from_slice(&src_pixels[src_start..src_end]);
                }
                rows as usize
            })
            .unwrap();

        assert!(!encoded.data().is_empty());
        assert!(encoded.data().len() > 100); // Sanity: not trivially small

        // Verify roundtrip: decode and check dimensions
        let dec = JpegDecoderConfig::new();
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[])
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(output.info().width, width);
        assert_eq!(output.info().height, height);
    }

    #[test]
    fn encode_from_requires_canvas_size() {
        use zencodec::encode::{EncodeJob as _, Encoder as _};
        use zenpixels::PixelViewMut;

        let config = JpegEncoderConfig::new();
        // No with_canvas_size — should error
        let encoder = config.job().encoder().unwrap();
        let result = encoder.encode_from(&mut |_y, _buf: PixelViewMut<'_>| 0);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod streaming_test {
    use super::*;
    use zencodec::encode::{EncodeJob, EncoderConfig};

    #[test]
    fn streaming_encode_same_scope() {
        // The real pattern: config consumed by job, stop borrowed from caller scope.
        // Encoder lives in same scope as stop. No escape needed.
        let stop = enough::Unstoppable;
        let config = JpegEncoderConfig::default();
        let job = config
            .job() // config consumed — no config borrow
            .with_pulse(zencodec::how_far::SharedPulse::new(
                zencodec::how_far::StopOnly::new(stop),
            )) // owned cancellation-only control
            .with_canvas_size(64, 64);
        let mut enc = job.dyn_encoder().unwrap();
        // enc borrows stop, both in same scope — compiles fine
        let pixels = vec![128u8; 64 * 64 * 4];
        let slice =
            zenpixels::PixelView::new(&pixels, 64, 64, 64 * 4, &zenpixels::formats::RGBA8_SRGB)
                .unwrap();
        enc.push_rows(slice).unwrap();
        let _output = enc.finish().unwrap();
    }

    fn make_job(w: u32, h: u32) -> JpegEncodeJob {
        let config = JpegEncoderConfig::default();
        // Config is consumed by job() via clone. Job doesn't borrow config.
        // No stop set, so 'a = 'static.
        config.job().with_canvas_size(w, h)
    }

    #[test]
    fn job_escapes_scope_without_stop() {
        let job = make_job(64, 64);
        let _enc = job.dyn_encoder().unwrap();
    }
}

#[cfg(test)]
mod cmyk_tests {
    use super::*;
    use alloc::borrow::Cow;
    use imgref::Img;
    use rgb::Rgb;
    use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};

    // ── CMYK handling tests ────────────────────────────────────────────────

    /// Load the CMYK flower test image if available.
    fn load_cmyk_test_image() -> Option<Vec<u8>> {
        crate::test_utils::read_test_data("jxl/flower/flower_small.cmyk.jpg")
    }

    #[test]
    fn cmyk_passthrough_produces_4_channels() {
        let data = match load_cmyk_test_image() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: CMYK test image not found");
                return;
            }
        };

        // Default handling is Passthrough.
        let output = JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(&data), &[])
            .unwrap()
            .decode()
            .unwrap();

        let info = output.info();
        let pixels = output.pixels();

        // Verify 4 channels via CMYK8 descriptor
        assert_eq!(pixels.format().layout.bytes_per_pixel(), 4);

        // source_color.channel_count should be 4 (CMYK)
        assert_eq!(info.source_color.channel_count, Some(4));

        // has_alpha should be false (this is CMYK, not RGBA)
        assert!(!info.has_alpha, "has_alpha should be false for raw CMYK");

        // ICC profile is preserved if present in the source.
        // Note: the flower_small.cmyk.jpg test image has no ICC profile,
        // so we only verify that the field is accessible (not that it's Some).
        let _icc = &info.source_color.icc_profile;

        // Verify dimensions are reasonable
        assert!(info.width > 0);
        assert!(info.height > 0);

        // Verify pixel data has correct size: w * h * 4
        let expected_bytes = info.width as usize * info.height as usize * 4;
        let actual_bytes = pixels.width() as usize * pixels.height() as usize * 4;
        assert_eq!(actual_bytes, expected_bytes);
    }

    #[test]
    fn cmyk_badrgb_vs_passthrough_produces_different_but_reasonable_output() {
        let data = match load_cmyk_test_image() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: CMYK test image not found");
                return;
            }
        };

        // Naive CMYK→RGB conversion (no ICC)
        let output_rgb = JpegDecoderConfig::new()
            .cmyk_handling(CmykHandling::BadRgb)
            .job()
            .decoder(Cow::Borrowed(&data), &[])
            .unwrap()
            .decode()
            .unwrap();

        // Passthrough (default): raw CMYK bytes + preserved ICC
        let output_cmyk = JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(&data), &[])
            .unwrap()
            .decode()
            .unwrap();

        let rgb_info = output_rgb.info();
        let cmyk_info = output_cmyk.info();

        // Same dimensions
        assert_eq!(rgb_info.width, cmyk_info.width);
        assert_eq!(rgb_info.height, cmyk_info.height);

        // BadRgb output is 3-channel RGB8, Passthrough is 4-channel CMYK8
        assert_eq!(output_rgb.pixels().format().layout.bytes_per_pixel(), 3);
        assert_eq!(output_cmyk.pixels().format().layout.bytes_per_pixel(), 4);

        // Manual CMYK→RGB conversion on raw CMYK data should produce something
        // close to the auto-converted RGB (within the known maxDelta of ~39)
        let w = cmyk_info.width as usize;
        let h = cmyk_info.height as usize;
        let cmyk_pixels = output_cmyk.pixels();
        let cmyk_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 4);
            for row in 0..cmyk_pixels.height() {
                all.extend_from_slice(cmyk_pixels.row(row));
            }
            all
        };
        let rgb_pixels = output_rgb.pixels();
        let rgb_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 3);
            for row in 0..rgb_pixels.height() {
                all.extend_from_slice(rgb_pixels.row(row));
            }
            all
        };

        // Apply simple CMYK→RGB using the same formula as cmyk_adobe_to_rgb
        let mut max_diff: u32 = 0;
        for i in 0..(w * h) {
            let c = cmyk_bytes[i * 4] as u32;
            let m = cmyk_bytes[i * 4 + 1] as u32;
            let y = cmyk_bytes[i * 4 + 2] as u32;
            let k = cmyk_bytes[i * 4 + 3] as u32;

            // Adobe inverted CMYK→RGB: R = C * K / 255
            let r_manual = ((c * k + 127) / 255) as u8;
            let g_manual = ((m * k + 127) / 255) as u8;
            let b_manual = ((y * k + 127) / 255) as u8;

            let r_auto = rgb_bytes[i * 3];
            let g_auto = rgb_bytes[i * 3 + 1];
            let b_auto = rgb_bytes[i * 3 + 2];

            let dr = (r_manual as i32 - r_auto as i32).unsigned_abs();
            let dg = (g_manual as i32 - g_auto as i32).unsigned_abs();
            let db = (b_manual as i32 - b_auto as i32).unsigned_abs();

            max_diff = max_diff.max(dr).max(dg).max(db);
        }

        // The manual conversion should match within a small tolerance.
        // The auto path and manual path use the same formula, so max_diff
        // should be <=1 (rounding differences from f32→u8 vs integer math).
        assert!(
            max_diff <= 2,
            "Manual CMYK→RGB vs auto CMYK→RGB max pixel diff = {max_diff}, expected <=2"
        );
    }

    #[test]
    fn cmyk_handling_has_no_effect_on_rgb_jpeg() {
        // Create a simple RGB JPEG
        let enc = JpegEncoderConfig::new().with_calibrated_quality(80.0);
        let pixels: Vec<Rgb<u8>> = vec![
            Rgb {
                r: 200,
                g: 100,
                b: 50,
            };
            64
        ];
        let img = Img::new(pixels, 8, 8);
        let encoded = enc
            .encode(
                PixelView::from_pixels_with_color(img.buf(), 8, 8, ZenPixelFormat::RGB8_SRGB.color)
                    .unwrap(),
            )
            .unwrap();
        let jpeg_data = encoded.into_vec();

        // Both CMYK handling modes should produce identical output for a
        // 3-component RGB JPEG — the CMYK branch requires num_components == 4.
        let out_passthrough = JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(&jpeg_data), &[])
            .unwrap()
            .decode()
            .unwrap();

        let out_badrgb = JpegDecoderConfig::new()
            .cmyk_handling(CmykHandling::BadRgb)
            .job()
            .decoder(Cow::Borrowed(&jpeg_data), &[])
            .unwrap()
            .decode()
            .unwrap();

        assert_eq!(
            out_passthrough.pixels().format().layout.bytes_per_pixel(),
            out_badrgb.pixels().format().layout.bytes_per_pixel()
        );
        assert_eq!(out_passthrough.info().width, out_badrgb.info().width);
        assert_eq!(out_passthrough.info().height, out_badrgb.info().height);

        let px_a = out_passthrough.pixels();
        let px_b = out_badrgb.pixels();
        assert_eq!(px_a.height(), px_b.height());
        for row in 0..px_a.height() {
            assert_eq!(
                px_a.row(row),
                px_b.row(row),
                "Row {row} differs between Passthrough and BadRgb for an RGB JPEG"
            );
        }
    }

    #[test]
    fn cmyk_passthrough_pixel_values_are_nonzero() {
        let data = match load_cmyk_test_image() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: CMYK test image not found");
                return;
            }
        };

        let output = JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(&data), &[])
            .unwrap()
            .decode()
            .unwrap();

        // Sample some pixels to verify they contain reasonable CMYK data
        // (not all zeros, not all 255)
        let pixels = output.pixels();
        let first_row = pixels.row(0);
        let has_nonzero = first_row.iter().any(|&b| b != 0);
        let has_non_ff = first_row.iter().any(|&b| b != 255);
        assert!(
            has_nonzero,
            "Raw CMYK output should contain non-zero values"
        );
        assert!(
            has_non_ff,
            "Raw CMYK output should contain values other than 255"
        );
    }

    #[test]
    fn cmyk_handling_default_is_passthrough() {
        let dec = JpegDecoderConfig::new();
        assert_eq!(
            dec.is_cmyk_handling(),
            CmykHandling::Passthrough,
            "Default CmykHandling should be Passthrough"
        );
    }

    /// Load the non-YCCK CMYK test image (Adobe transform=0).
    fn load_pure_cmyk_test_image() -> Option<Vec<u8>> {
        let path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/testdata/cymk.jpg");
        std::fs::read(&path).ok()
    }

    #[test]
    fn cmyk_passthrough_non_ycck_roundtrip() {
        let data = match load_pure_cmyk_test_image() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: cymk.jpg test image not found");
                return;
            }
        };

        // Naive CMYK→RGB (BadRgb)
        let output_rgb = JpegDecoderConfig::new()
            .cmyk_handling(CmykHandling::BadRgb)
            .job()
            .decoder(Cow::Borrowed(&data), &[])
            .unwrap()
            .decode()
            .unwrap();

        // Passthrough (default): raw CMYK bytes
        let output_cmyk = JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(&data), &[])
            .unwrap()
            .decode()
            .unwrap();

        let cmyk_info = output_cmyk.info();
        assert_eq!(cmyk_info.source_color.channel_count, Some(4));
        assert!(!cmyk_info.has_alpha);
        assert_eq!(output_cmyk.pixels().format().layout.bytes_per_pixel(), 4);

        // Same dimensions
        assert_eq!(output_rgb.info().width, cmyk_info.width);
        assert_eq!(output_rgb.info().height, cmyk_info.height);

        let w = cmyk_info.width as usize;
        let h = cmyk_info.height as usize;
        let cmyk_pixels = output_cmyk.pixels();
        let cmyk_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 4);
            for row in 0..cmyk_pixels.height() {
                all.extend_from_slice(cmyk_pixels.row(row));
            }
            all
        };
        let rgb_pixels = output_rgb.pixels();
        let rgb_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 3);
            for row in 0..rgb_pixels.height() {
                all.extend_from_slice(rgb_pixels.row(row));
            }
            all
        };

        // Manual CMYK→RGB using Adobe inverted formula
        let mut max_diff: u32 = 0;
        for i in 0..(w * h) {
            let c = cmyk_bytes[i * 4] as u32;
            let m = cmyk_bytes[i * 4 + 1] as u32;
            let y = cmyk_bytes[i * 4 + 2] as u32;
            let k = cmyk_bytes[i * 4 + 3] as u32;

            let r_manual = ((c * k + 127) / 255) as u8;
            let g_manual = ((m * k + 127) / 255) as u8;
            let b_manual = ((y * k + 127) / 255) as u8;

            let r_auto = rgb_bytes[i * 3];
            let g_auto = rgb_bytes[i * 3 + 1];
            let b_auto = rgb_bytes[i * 3 + 2];

            let dr = (r_manual as i32 - r_auto as i32).unsigned_abs();
            let dg = (g_manual as i32 - g_auto as i32).unsigned_abs();
            let db = (b_manual as i32 - b_auto as i32).unsigned_abs();

            max_diff = max_diff.max(dr).max(dg).max(db);
        }

        assert!(
            max_diff <= 2,
            "Pure CMYK: manual vs auto CMYK→RGB max pixel diff = {max_diff}, expected <=2"
        );
    }
}

#[cfg(test)]
mod push_decode_stride_tests {
    use super::*;
    use std::borrow::Cow;
    use zencodec::decode::{DecodeRowSink, DynDecoderConfig, SinkError};

    struct StridedSink {
        data: Vec<u8>,
        stride: usize,
        w: u32,
        h: u32,
    }

    impl DecodeRowSink for StridedSink {
        fn begin(&mut self, w: u32, h: u32, _desc: ZenPixelFormat) -> Result<(), SinkError> {
            self.w = w;
            self.h = h;
            self.stride = ((w as usize * 4 + 31) / 32) * 32;
            self.data = vec![0xCC; self.stride * h as usize];
            Ok(())
        }
        fn provide_next_buffer(
            &mut self,
            strip: zencodec::decode::RowStrip,
            descriptor: ZenPixelFormat,
        ) -> Result<zenpixels::PixelViewMut<'_>, SinkError> {
            let (y, width, height) = (strip.y, strip.width, strip.height);
            let row_start = y as usize * self.stride;
            let row_bytes = width as usize * 4;
            let needed = if height > 0 {
                (height as usize - 1) * self.stride + row_bytes
            } else {
                0
            };
            let slice = &mut self.data[row_start..row_start + needed];
            zenpixels::PixelViewMut::new_with_format_cow(
                slice,
                width,
                height,
                self.stride,
                zenpixels::FormatCow::from_owned(descriptor),
            )
            .map_err(|e| -> SinkError { format!("{e}").into() })
        }
    }

    #[test]
    fn push_decode_bgra_strided_no_sentinel() {
        let config =
            crate::encoder::EncoderConfig::ycbcr(85, crate::encode::ChromaSubsampling::Quarter)
                .progressive(crate::encode::ProgressiveScanMode::Baseline);
        let pixels: Vec<rgb::RGB8> = (0..64 * 64)
            .map(|_| rgb::RGB8::new(0x40, 0x80, 0xFF))
            .collect();
        let jpeg = config.encode(&pixels, 64, 64).unwrap();

        let preferred = [ZenPixelFormat::BGRA8_SRGB];
        let cfg = JpegDecoderConfig::new();
        let job = cfg.dyn_job();
        let mut sink = StridedSink {
            data: vec![],
            stride: 0,
            w: 0,
            h: 0,
        };
        job.push_decode(Cow::Borrowed(&jpeg), &mut sink, &preferred)
            .unwrap();

        let row_bytes = sink.w as usize * 4;
        let sentinel_count: usize = (0..sink.h as usize)
            .map(|y| {
                let row = &sink.data[y * sink.stride..y * sink.stride + row_bytes];
                row.iter().filter(|&&b| b == 0xCC).count()
            })
            .sum();
        assert_eq!(
            sentinel_count, 0,
            "push_decode left {sentinel_count} unwritten bytes in pixel region"
        );
    }

    #[test]
    fn push_decode_bgra_matches_buffered() {
        let config =
            crate::encoder::EncoderConfig::ycbcr(85, crate::encode::ChromaSubsampling::Quarter)
                .progressive(crate::encode::ProgressiveScanMode::Baseline);
        let pixels: Vec<rgb::RGB8> = (0..64 * 64)
            .map(|_| rgb::RGB8::new(0x40, 0x80, 0xFF))
            .collect();
        let jpeg = config.encode(&pixels, 64, 64).unwrap();

        // Buffered decode
        let preferred = [ZenPixelFormat::BGRA8_SRGB];
        let cfg1 = JpegDecoderConfig::new();
        let job1 = cfg1.dyn_job();
        let dec = job1.into_decoder(Cow::Borrowed(&jpeg), &preferred).unwrap();
        let output = dec.decode().unwrap();
        let ps = output.pixels();
        let buffered: Vec<u8> = (0..ps.height()).flat_map(|y| ps.row(y).to_vec()).collect();

        // Push decode into strided buffer
        let cfg2 = JpegDecoderConfig::new();
        let job2 = cfg2.dyn_job();
        let mut sink = StridedSink {
            data: vec![],
            stride: 0,
            w: 0,
            h: 0,
        };
        job2.push_decode(Cow::Borrowed(&jpeg), &mut sink, &preferred)
            .unwrap();

        let row_bytes = sink.w as usize * 4;
        let push: Vec<u8> = (0..sink.h as usize)
            .flat_map(|y| sink.data[y * sink.stride..y * sink.stride + row_bytes].to_vec())
            .collect();

        assert_eq!(buffered.len(), push.len(), "size mismatch");
        let diffs: usize = buffered
            .iter()
            .zip(push.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diffs,
            0,
            "{diffs}/{} bytes differ between push_decode and buffered",
            buffered.len()
        );
    }

    /// Decoding with `AllocPreference::Fallible` (the `try_reserve` path) and
    /// `AllocPreference::Infallible` (the `vec!` path) must produce
    /// byte-identical pixels to the default (`CodecDefault`) decode, across
    /// baseline 4:2:0, baseline 4:4:4, and progressive — exercising the strip,
    /// output, and coefficient-storage allocation sites under each mode.
    #[test]
    fn fallible_alloc_decode_matches_default() {
        use zencodec::decode::{Decode as _, DecodeJob as _, DecoderConfig as _};
        use zencodec::{AllocPreference, ResourceLimits};

        let (w, h) = (64usize, 48usize);

        // A noise-ish RGB image (not a smooth gradient — those degenerate the
        // DCT coefficients per project policy).
        let pixels: Vec<rgb::RGB8> = (0..w * h)
            .map(|i| {
                rgb::RGB8::new(
                    (i.wrapping_mul(97) % 251) as u8,
                    (i.wrapping_mul(53) % 253) as u8,
                    (i.wrapping_mul(29) % 249) as u8,
                )
            })
            .collect();

        // Decode `jpeg` under the given AllocPreference; return the contiguous
        // decoded bytes.
        let decode_bytes = |jpeg: &[u8], pref: Option<AllocPreference>| -> Vec<u8> {
            let job = JpegDecoderConfig::new().job();
            let job = match pref {
                Some(p) => {
                    job.with_limits(ResourceLimits::none().with_prefer_fallible_allocations(p))
                }
                None => job,
            };
            let out = job
                .decoder(Cow::Borrowed(jpeg), &[])
                .unwrap()
                .decode()
                .unwrap();
            let ps = out.pixels();
            (0..ps.height()).flat_map(|y| ps.row(y).to_vec()).collect()
        };

        let assert_all_modes_agree = |jpeg: &[u8], label: &str| {
            let default = decode_bytes(jpeg, None); // CodecDefault
            let fallible = decode_bytes(jpeg, Some(AllocPreference::Fallible));
            let infallible = decode_bytes(jpeg, Some(AllocPreference::Infallible));
            assert_eq!(
                default, fallible,
                "{label}: Fallible decode must be byte-identical to the default"
            );
            assert_eq!(
                default, infallible,
                "{label}: Infallible decode must be byte-identical to the default"
            );
        };

        // (1) Baseline 4:2:0 — fused/streaming path + chroma upsample strips.
        let jpeg_420 = crate::encoder::EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .encode(&pixels, w as u32, h as u32)
            .unwrap();
        assert_all_modes_agree(&jpeg_420, "baseline 4:2:0");

        // (2) Baseline 4:4:4 — no chroma subsampling.
        let jpeg_444 = crate::encoder::EncoderConfig::ycbcr(85, ChromaSubsampling::None)
            .encode(&pixels, w as u32, h as u32)
            .unwrap();
        assert_all_modes_agree(&jpeg_444, "baseline 4:4:4");

        // (3) Progressive — exercises the full-frame coefficient-storage sites.
        let jpeg_prog = crate::encoder::EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .progressive(crate::encode::ProgressiveScanMode::Progressive)
            .encode(&pixels, w as u32, h as u32)
            .unwrap();
        assert_all_modes_agree(&jpeg_prog, "progressive 4:2:0");
    }

    /// `estimate_decode_resources` returns a non-trivial, scaling estimate
    /// (peak ≥ output buffer, larger image ⇒ larger peak + wall time).
    #[test]
    fn estimate_decode_resources_is_reasonable() {
        use zencodec::decode::DecoderConfig;
        use zencodec::estimate::{ComputeEnvironment, ImageCharacteristics};

        let compute = ComputeEnvironment::conservative();
        let small = ImageCharacteristics::new(256, 256, ZenPixelFormat::RGB8_SRGB);
        let large = ImageCharacteristics::new(2048, 2048, ZenPixelFormat::RGB8_SRGB);

        let cfg = JpegDecoderConfig::new();
        // Qualify the trait: zencodec 0.1.25's blanket `DynDecoderConfig` also
        // exposes `estimate_decode_resources`, so a bare method call is ambiguous.
        let es = DecoderConfig::estimate_decode_resources(&cfg, &small, &compute);
        let el = DecoderConfig::estimate_decode_resources(&cfg, &large, &compute);

        let es_peak = es.peak_memory_bytes_est().expect("small peak estimate");
        let el_peak = el.peak_memory_bytes_est().expect("large peak estimate");

        // Peak must cover at least the output buffer (W*H*3).
        assert!(
            es_peak >= 256 * 256 * 3,
            "peak {es_peak} below output buffer for 256²"
        );
        // Larger image ⇒ strictly larger peak and wall time.
        assert!(el_peak > es_peak);
        assert!(el.wall_ms().unwrap_or(0) >= es.wall_ms().unwrap_or(0));
    }
}

#[cfg(test)]
mod pattern_b_envelope_tests {
    //! Pattern B forcing tests: the `At<CodecError>` envelope must carry the
    //! category + codec name across `Dyn*` type erasure. Under Pattern A
    //! (`type Error = Error`) the erased `BoxedError` is a plain `dyn Error` with
    //! no `CategorizedError` vtable, so both recoveries return `None`; the
    //! envelope is exactly what flips them to `Some`.
    use super::*;
    use zencodec::decode::DynDecoderConfig;
    use zencodec::{CodecError, CodecErrorExt, ErrorCategory, ImageError};

    /// SOI + SOF0 declaring **zero** components: passes the SOI gate, then fails
    /// structurally inside the frame-header parse
    /// (`invalid_jpeg_data("number of components is zero")` →
    /// [`ErrorCategory::Image(ImageError::Malformed)`]). Long enough to reach
    /// the structural error rather than a truncation/EOF path — a REAL
    /// category, not a stand-in.
    const MALFORMED_JPEG: &[u8] = &[
        0xFF, 0xD8, // SOI
        0xFF, 0xC0, // SOF0
        0x00, 0x08, // segment length = 8
        0x08, // sample precision = 8
        0x00, 0x10, // height = 16
        0x00, 0x10, // width = 16
        0x00, // component count = 0  ← malformed
    ];

    /// THE forcing test. Drive zenjpeg through `&dyn DynDecoderConfig` (the
    /// codec-agnostic dyn path), let the concrete `At<CodecError>` erase to
    /// `BoxedError` (`Box<dyn Error + Send + Sync>`), and confirm BOTH the
    /// category and the codec name still come back.
    #[test]
    fn dyn_decode_envelope_recovers_category_and_codec() {
        let cfg = JpegDecoderConfig::new();
        let dyn_cfg: &dyn DynDecoderConfig = &cfg;
        let erased = dyn_cfg
            .dyn_job()
            .probe(MALFORMED_JPEG)
            .expect_err("a malformed JPEG must fail to probe");

        // `error_category()` / `codec_error()` downcast the erased boxed error
        // back to the concrete `At<CodecError>` — possible only because the
        // trait boundary returns the envelope (Pattern B).
        assert_eq!(
            erased.error_category(),
            Some(ErrorCategory::Image(ImageError::Malformed)),
            "category must survive Dyn* type erasure"
        );
        assert_eq!(
            erased.codec_error().and_then(CodecError::codec),
            Some("zenjpeg"),
            "codec name must survive Dyn* type erasure"
        );
    }

    /// The same recovery on the typed convenience method, which now returns the
    /// envelope directly (no erasure needed — reads category/codec off the
    /// inner [`CodecError`]).
    #[test]
    fn typed_probe_header_carries_envelope() {
        let err = JpegDecoderConfig::new()
            .probe_header(MALFORMED_JPEG)
            .expect_err("malformed JPEG must fail");
        assert_eq!(
            err.error().category(),
            ErrorCategory::Image(ImageError::Malformed)
        );
        assert_eq!(err.error().codec(), Some("zenjpeg"));
    }

    /// A decode through the dyn path erases the same way: the envelope is
    /// recovered from the `into_decoder(...).decode()` boxed error too.
    #[test]
    fn dyn_decode_decode_path_carries_envelope() {
        use std::borrow::Cow;
        let cfg = JpegDecoderConfig::new();
        let dyn_cfg: &dyn DynDecoderConfig = &cfg;
        let erased = dyn_cfg
            .dyn_job()
            .into_decoder(Cow::Borrowed(MALFORMED_JPEG), &[])
            .and_then(|dec| dec.decode())
            .expect_err("a malformed JPEG must fail to decode");
        assert_eq!(
            erased.error_category(),
            Some(ErrorCategory::Image(ImageError::Malformed))
        );
        assert_eq!(
            erased.codec_error().and_then(CodecError::codec),
            Some("zenjpeg")
        );
    }
}

#[cfg(test)]
mod incremental_decode_tests {
    use super::*;
    use core::num::NonZeroU32;
    use zencodec::decode::{
        Decode as _, DecodeJob as _, DecodeRowSink, DecoderConfig as _, DynDecoderConfig, Event,
        IncrementalDecode, Input, OutputInfo, RowStrip, SinkError,
    };

    struct CollectSink {
        width: u32,
        height: u32,
        format: Option<ZenPixelFormat>,
        pixels: Vec<u8>,
        max_rows: u32,
        finished: bool,
    }

    impl CollectSink {
        fn new(max_rows: u32) -> Self {
            Self {
                width: 0,
                height: 0,
                format: None,
                pixels: Vec::new(),
                max_rows,
                finished: false,
            }
        }
    }

    impl DecodeRowSink for CollectSink {
        fn begin(
            &mut self,
            width: u32,
            height: u32,
            format: ZenPixelFormat,
        ) -> Result<(), SinkError> {
            self.width = width;
            self.height = height;
            let bpp = format.layout.bytes_per_pixel();
            self.pixels = vec![0u8; width as usize * height as usize * bpp];
            self.format = Some(format);
            Ok(())
        }

        fn max_buffer_rows(&self) -> NonZeroU32 {
            NonZeroU32::new(self.max_rows.max(1)).unwrap()
        }

        fn provide_next_buffer(
            &mut self,
            strip: RowStrip,
            format: ZenPixelFormat,
        ) -> Result<PixelViewMut<'_>, SinkError> {
            let bpp = format.layout.bytes_per_pixel();
            let row_bytes = self.width as usize * bpp;
            let start = strip.y as usize * row_bytes;
            let end = start + strip.height as usize * row_bytes;
            let fmt = self.format.as_ref().unwrap();
            Ok(PixelViewMut::new(
                &mut self.pixels[start..end],
                strip.width,
                strip.height,
                row_bytes,
                fmt,
            )
            .unwrap())
        }

        fn finish(&mut self) -> Result<(), SinkError> {
            self.finished = true;
            Ok(())
        }
    }

    fn make_pattern_rgb(w: usize, h: usize) -> Vec<Rgb<u8>> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                out.push(Rgb {
                    r: ((x * 7 + y * 3) & 0xFF) as u8,
                    g: ((x * 5 ^ y * 11) & 0xFF) as u8,
                    b: ((x * 13 + y * 17) & 0xFF) as u8,
                });
            }
        }
        out
    }

    fn drive_incremental(
        dec: &mut dyn IncrementalDecode<Error = At<CodecError>>,
        data: &[u8],
        chunk_size: usize,
        sink_max_rows: u32,
    ) -> (CollectSink, OutputInfo, bool) {
        let mut sink = CollectSink::new(sink_max_rows);
        let mut emitted_before_eof = false;
        let mut cursor = 0usize;
        let mut delivered = 0usize;

        loop {
            let req = dec.requested();
            let offset = req
                .first()
                .map(|r| match *r {
                    zencodec::decode::InputRequest::AtLeast { offset, .. } => offset as usize,
                    zencodec::decode::InputRequest::Range(r) => r.offset as usize,
                    _ => cursor,
                })
                .unwrap_or(cursor);
            if offset > delivered {
                delivered = offset.min(data.len());
            }
            if delivered < data.len() && offset == delivered {
                delivered = (delivered + chunk_size).min(data.len());
            }
            let slice = &data[offset.min(data.len())..delivered];
            let mut input = Input::new(offset as u64, slice);
            if delivered >= data.len() {
                input.total_len = Some(data.len() as u64);
            }
            let progress = dec.step(input, &mut sink, None).unwrap();
            cursor = offset + progress.consumed;
            match progress.event {
                Event::Need => {
                    if progress.consumed == 0 && delivered < data.len() {
                        delivered = (delivered + chunk_size).min(data.len());
                    }
                }
                Event::Rows { rows, .. } => {
                    assert!(rows > 0);
                    if delivered < data.len() {
                        emitted_before_eof = true;
                    }
                }
                Event::Done(done) => {
                    assert!(sink.finished);
                    return (sink, done, emitted_before_eof);
                }
                _ => {}
            }
        }
    }

    fn encode_baseline(w: usize, h: usize, subsampling: ChromaSubsampling) -> Vec<u8> {
        let pixels = make_pattern_rgb(w, h);
        let cfg = EncoderConfig::ycbcr(85.0, subsampling).progressive(false);
        let mut enc = cfg
            .encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(bytemuck::cast_slice(&pixels), enough::Unstoppable)
            .unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn incremental_baseline_nearest_matches_oneshot_bit_exact_and_streams_before_eof() {
        let jpeg = encode_baseline(128, 128, ChromaSubsampling::Quarter);
        let mut cfg = JpegDecoderConfig::new();
        cfg.inner_mut().chroma_upsampling = crate::decode::ChromaUpsampling::NearestNeighbor;

        let oneshot = cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .decoder(Cow::Borrowed(&jpeg), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap();
        let ref_bytes = oneshot.pixels().as_contiguous_bytes().unwrap().to_vec();

        for &chunk_size in &[jpeg.len(), 512, 128, 33, 7, 3, 1] {
            let mut inc = cfg
                .clone()
                .job()
                .with_orientation(zencodec::OrientationHint::Preserve)
                .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                .unwrap();
            let (sink, done, before_eof) = drive_incremental(&mut inc, &jpeg, chunk_size, 16);
            assert_eq!(done.width, 128);
            assert_eq!(done.height, 128);
            assert_eq!(
                sink.pixels, ref_bytes,
                "baseline 128x128 chunk_size={chunk_size} mismatch"
            );
            if chunk_size <= 256 {
                assert!(
                    before_eof,
                    "baseline NearestNeighbor + Preserve must emit MCU rows before EOF (chunk_size={chunk_size})"
                );
            }
        }
    }

    #[test]
    fn incremental_baseline_odd_dimensions_and_small_sink_strips_match_bit_exact() {
        let jpeg = encode_baseline(53, 37, ChromaSubsampling::Quarter);
        let mut cfg = JpegDecoderConfig::new();
        cfg.inner_mut().chroma_upsampling = crate::decode::ChromaUpsampling::NearestNeighbor;

        let oneshot = cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .decoder(Cow::Borrowed(&jpeg), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap();
        let ref_bytes = oneshot.pixels().as_contiguous_bytes().unwrap().to_vec();

        // max_buffer_rows = 3 forces multi-slice delivery across every 16-row MCU batch.
        for &chunk_size in &[jpeg.len(), 64, 7, 1] {
            let mut inc = cfg
                .clone()
                .job()
                .with_orientation(zencodec::OrientationHint::Preserve)
                .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                .unwrap();
            let (sink, done, _) = drive_incremental(&mut inc, &jpeg, chunk_size, 3);
            assert_eq!(done.width, 53);
            assert_eq!(done.height, 37);
            assert_eq!(
                sink.pixels, ref_bytes,
                "odd 53x37 small-strip chunk_size={chunk_size} mismatch"
            );
        }
    }

    #[test]
    fn incremental_progressive_and_xyb_and_triangle_fallback_match_bit_exact() {
        // 1) Progressive with NearestNeighbor -> dynamic buffered_whole_file fallback
        let pixels = make_pattern_rgb(64, 64);
        let prog_cfg = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);
        let mut enc = prog_cfg
            .encode_from_bytes(64, 64, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(bytemuck::cast_slice(&pixels), enough::Unstoppable)
            .unwrap();
        let prog_jpeg = enc.finish().unwrap();

        let mut nn_cfg = JpegDecoderConfig::new();
        nn_cfg.inner_mut().chroma_upsampling = crate::decode::ChromaUpsampling::NearestNeighbor;
        let prog_ref = nn_cfg
            .clone()
            .decode(&prog_jpeg)
            .unwrap()
            .pixels()
            .as_contiguous_bytes()
            .unwrap()
            .to_vec();
        for &chunk_size in &[prog_jpeg.len(), 128, 17, 1] {
            let mut inc = nn_cfg
                .clone()
                .job()
                .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                .unwrap();
            let (sink, _, _) = drive_incremental(&mut inc, &prog_jpeg, chunk_size, 5);
            assert_eq!(sink.pixels, prog_ref, "progressive chunk={chunk_size}");
        }

        // 2) XYB with NearestNeighbor -> dynamic buffered_whole_file fallback (preserves XYB->sRGB)
        let xyb_pixels = make_pattern_rgb(32, 32);
        let xyb_jpeg = EncoderConfig::xyb(90, crate::encode::XybSubsampling::BQuarter)
            .encode(&xyb_pixels, 32, 32)
            .unwrap();
        let xyb_ref = nn_cfg
            .clone()
            .decode(&xyb_jpeg)
            .unwrap()
            .pixels()
            .as_contiguous_bytes()
            .unwrap()
            .to_vec();
        let mut xyb_inc = nn_cfg
            .clone()
            .job()
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (sink, _, _) = drive_incremental(&mut xyb_inc, &xyb_jpeg, 37, 8);
        assert_eq!(sink.pixels, xyb_ref, "XYB incremental fallback mismatch");

        // 3) Default Triangle upsampling via dyn_incremental_decoder
        let base_jpeg = encode_baseline(48, 40, ChromaSubsampling::Quarter);
        let tri_cfg = JpegDecoderConfig::new();
        let tri_ref = tri_cfg
            .clone()
            .decode(&base_jpeg)
            .unwrap()
            .pixels()
            .as_contiguous_bytes()
            .unwrap()
            .to_vec();
        let dyn_cfg: &dyn DynDecoderConfig = &tri_cfg;
        let mut dyn_inc = dyn_cfg
            .dyn_job()
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut sink = CollectSink::new(7);
        let mut delivered = 0usize;
        loop {
            let offset = dyn_inc
                .requested()
                .first()
                .map(|r| match *r {
                    zencodec::decode::InputRequest::AtLeast { offset, .. } => offset as usize,
                    zencodec::decode::InputRequest::Range(r) => r.offset as usize,
                    _ => 0,
                })
                .unwrap_or(0);
            if delivered < base_jpeg.len() && offset >= delivered {
                delivered = (delivered + 41).min(base_jpeg.len());
            }
            let mut input = Input::new(
                offset as u64,
                &base_jpeg[offset.min(base_jpeg.len())..delivered],
            );
            if delivered >= base_jpeg.len() {
                input.total_len = Some(base_jpeg.len() as u64);
            }
            let prog = dyn_inc.step(input, &mut sink, None).unwrap();
            match prog.event {
                Event::Need => {
                    if prog.consumed == 0 && delivered < base_jpeg.len() {
                        delivered = (delivered + 41).min(base_jpeg.len());
                    }
                }
                Event::Done(done) => {
                    assert_eq!(done.width, 48);
                    assert_eq!(done.height, 40);
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(
            sink.pixels, tri_ref,
            "dyn_incremental_decoder Triangle mismatch"
        );
    }

    #[test]
    fn incremental_restart_markers_post_scan_exif_rgbf32_and_limits_match_oneshot() {
        use zencodec::decode::Decode as _;

        // 1) Baseline with DRI restart markers (restart_interval = 1) streams before EOF
        let pixels = make_pattern_rgb(64, 48);
        let dri_cfg = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
            .progressive(false)
            .restart_mcu_rows(1);
        let mut enc = dri_cfg
            .encode_from_bytes(64, 48, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(bytemuck::cast_slice(&pixels), enough::Unstoppable)
            .unwrap();
        let dri_jpeg = enc.finish().unwrap();

        let mut nn_cfg = JpegDecoderConfig::new();
        nn_cfg.inner_mut().chroma_upsampling = crate::decode::ChromaUpsampling::NearestNeighbor;
        let dri_ref = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .decoder(Cow::Borrowed(&dri_jpeg), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap()
            .pixels()
            .as_contiguous_bytes()
            .unwrap()
            .to_vec();
        for &chunk_size in &[dri_jpeg.len(), 64, 7, 1] {
            let mut inc = nn_cfg
                .clone()
                .job()
                .with_orientation(zencodec::OrientationHint::Preserve)
                .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                .unwrap();
            let (sink, _, before_eof) = drive_incremental(&mut inc, &dri_jpeg, chunk_size, 8);
            assert_eq!(sink.pixels, dri_ref, "DRI chunk_size={chunk_size}");
            if chunk_size <= 64 {
                assert!(before_eof, "DRI baseline should emit before EOF");
            }
        }

        // 2) Post-scan APP1 EXIF orientation=6 right before EOI under OrientationHint::Correct
        let base_48x32 = encode_baseline(48, 32, ChromaSubsampling::Quarter);
        assert!(base_48x32.ends_with(&[0xFF, 0xD9]));
        let exif_tiff = Exif::build()
            .orientation(crate::encode::exif::Orientation::Rotate90)
            .to_bytes()
            .unwrap();
        let mut app1_payload = b"Exif\0\0".to_vec();
        app1_payload.extend_from_slice(&exif_tiff);
        let mut post_scan_exif_jpeg = base_48x32[..base_48x32.len() - 2].to_vec();
        post_scan_exif_jpeg.extend_from_slice(&[0xFF, 0xE1]);
        post_scan_exif_jpeg.extend_from_slice(&((app1_payload.len() + 2) as u16).to_be_bytes());
        post_scan_exif_jpeg.extend_from_slice(&app1_payload);
        post_scan_exif_jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let oriented_oneshot = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Correct)
            .decoder(
                Cow::Borrowed(&post_scan_exif_jpeg),
                &[ZenPixelFormat::RGB8_SRGB],
            )
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(oriented_oneshot.width(), 32);
        assert_eq!(oriented_oneshot.height(), 48);
        let mut inc_orient = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Correct)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (sink, done, _) = drive_incremental(&mut inc_orient, &post_scan_exif_jpeg, 19, 7);
        assert_eq!((done.width, done.height), (32, 48));
        assert_eq!(
            sink.pixels,
            oriented_oneshot.pixels().as_contiguous_bytes().unwrap()
        );

        // 3) RGBF32 preference stays exact RGBF32 (never upgraded to RGBAF32)
        let rgbf32_oneshot = JpegDecoderConfig::new()
            .job()
            .decoder(Cow::Borrowed(&base_48x32), &[ZenPixelFormat::RGBF32_LINEAR])
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(
            rgbf32_oneshot.pixels().format().clone(),
            ZenPixelFormat::RGBF32_LINEAR
        );
        let mut inc_f32 = JpegDecoderConfig::new()
            .job()
            .into_incremental_decoder(&[ZenPixelFormat::RGBF32_LINEAR])
            .unwrap();
        let (sink_f32, done_f32, _) = drive_incremental(&mut inc_f32, &base_48x32, 31, 5);
        assert_eq!(done_f32.native_format, ZenPixelFormat::RGBF32_LINEAR);
        assert_eq!(
            sink_f32.pixels,
            rgbf32_oneshot.pixels().as_contiguous_bytes().unwrap()
        );

        // 4) Resource limits enforced on BOTH streaming and config-selected buffered paths
        let tiny_w = ResourceLimits::none().with_max_width(16);
        let mut inc_w = JpegDecoderConfig::new()
            .job()
            .with_limits(tiny_w)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut sink_err = CollectSink::new(16);
        let mut inp = Input::new(0, &base_48x32);
        inp.total_len = Some(base_48x32.len() as u64);
        assert!(matches!(
            inc_w.step(inp, &mut sink_err, None),
            Err(zencodec::decode::StepError::Codec(_))
        ));
        // Poisoned after error -> subsequent step returns Finished
        assert!(matches!(
            inc_w.step(Input::empty(), &mut sink_err, None),
            Err(zencodec::decode::StepError::Finished)
        ));

        let tiny_in = ResourceLimits::none().with_max_input_bytes(16);
        let mut inc_in = JpegDecoderConfig::new()
            .job()
            .with_limits(tiny_in)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut inp2 = Input::new(0, &base_48x32);
        inp2.total_len = Some(base_48x32.len() as u64);
        assert!(matches!(
            inc_in.step(inp2, &mut sink_err, None),
            Err(zencodec::decode::StepError::Codec(_))
        ));

        // 5) Call pulse cancellation returns StepError::Cancelled
        struct AlwaysCancel;
        impl enough::Stop for AlwaysCancel {
            fn check(&self) -> core::result::Result<(), enough::StopReason> {
                Err(enough::StopReason::Cancelled)
            }
        }
        let cancel_pulse = zencodec::how_far::StopOnly::new(AlwaysCancel);
        let mut inc_cancel = JpegDecoderConfig::new()
            .job()
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut inp3 = Input::new(0, &base_48x32);
        inp3.total_len = Some(base_48x32.len() as u64);
        assert!(matches!(
            inc_cancel.step(inp3, &mut sink_err, Some(&cancel_pulse)),
            Err(zencodec::decode::StepError::Cancelled(
                enough::StopReason::Cancelled
            ))
        ));

        // 6) Sink error returns StepError::Sink and poisons decoder
        struct FailingSink;
        impl DecodeRowSink for FailingSink {
            fn begin(&mut self, _: u32, _: u32, _: ZenPixelFormat) -> Result<(), SinkError> {
                Ok(())
            }
            fn max_buffer_rows(&self) -> NonZeroU32 {
                NonZeroU32::new(4).unwrap()
            }
            fn provide_next_buffer(
                &mut self,
                _: RowStrip,
                _: ZenPixelFormat,
            ) -> Result<PixelViewMut<'_>, SinkError> {
                Err(SinkError::from("synthetic sink failure"))
            }
            fn finish(&mut self) -> Result<(), SinkError> {
                Ok(())
            }
        }
        let mut inc_sink = JpegDecoderConfig::new()
            .job()
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut bad_sink = FailingSink;
        let mut inp4 = Input::new(0, &base_48x32);
        inp4.total_len = Some(base_48x32.len() as u64);
        assert!(matches!(
            inc_sink.step(inp4, &mut bad_sink, None),
            Err(zencodec::decode::StepError::Sink(_))
        ));
        assert!(matches!(
            inc_sink.step(Input::empty(), &mut bad_sink, None),
            Err(zencodec::decode::StepError::Finished)
        ));

        // 7) Non-interleaved 3-scan baseline JPEG falls back to exact one-shot decode
        let ni_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/testdata/non_interleaved/ni420_88x54.jpg");
        let ni_jpeg = std::fs::read(&ni_path)
            .unwrap_or_else(|e| panic!("read fixture {}: {e}", ni_path.display()));
        let ni_oneshot = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .decoder(Cow::Borrowed(&ni_jpeg), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap();
        let mut inc_ni = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (sink_ni, done_ni, _) = drive_incremental(&mut inc_ni, &ni_jpeg, 29, 5);
        assert_eq!((done_ni.width, done_ni.height), (88, 54));
        assert_eq!(
            sink_ni.pixels,
            ni_oneshot.pixels().as_contiguous_bytes().unwrap(),
            "non-interleaved 4:2:0 must match one-shot 0-bit exact"
        );

        // 8) Truncated JPEG under Strict (and truncated header under default Balanced)
        //    errors cleanly and poisons decoder
        for (label, strict, truncated) in [
            (
                "strict_scan_trunc",
                true,
                &base_48x32[..base_48x32.len() / 3],
            ),
            ("balanced_header_trunc", false, &base_48x32[..12]),
        ] {
            let mut trunc_cfg = JpegDecoderConfig::new();
            if strict {
                trunc_cfg.inner_mut().strictness = crate::decode::Strictness::Strict;
            }
            let mut inc_trunc = trunc_cfg
                .job()
                .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                .unwrap();
            let mut trunc_sink = CollectSink::new(8);
            let mut inp_t = Input::new(0, truncated);
            inp_t.total_len = Some(truncated.len() as u64);
            let mut got_err = false;
            for _ in 0..8 {
                match inc_trunc.step(inp_t, &mut trunc_sink, None) {
                    Err(zencodec::decode::StepError::Codec(_)) => {
                        got_err = true;
                        break;
                    }
                    Ok(p) => {
                        let off = (inp_t.offset as usize + p.consumed).min(truncated.len());
                        inp_t = Input::new(off as u64, &truncated[off..]);
                        inp_t.total_len = Some(truncated.len() as u64);
                        if matches!(p.event, Event::Done(_)) {
                            break;
                        }
                    }
                    Err(other) => panic!("unexpected truncation error ({label}): {other:?}"),
                }
            }
            assert!(got_err, "expected truncation error for {label}");
            assert!(matches!(
                inc_trunc.step(Input::empty(), &mut trunc_sink, None),
                Err(zencodec::decode::StepError::Finished)
            ));
        }

        // 9) Effective limit consistency across streaming, buffered fallback, and one-shot:
        //    (a) DecodeConfig::max_memory(0) == unlimited works on both streaming and buffered.
        let mut unlimited_nn = JpegDecoderConfig::new();
        unlimited_nn.inner_mut().chroma_upsampling =
            crate::decode::ChromaUpsampling::NearestNeighbor;
        unlimited_nn.inner_mut().max_memory = 0;
        let base_48x32_ref = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .decoder(Cow::Borrowed(&base_48x32), &[ZenPixelFormat::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap()
            .pixels()
            .as_contiguous_bytes()
            .unwrap()
            .to_vec();
        let mut inc_unlim_stream = unlimited_nn
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (sink_u1, done_u1, _) = drive_incremental(&mut inc_unlim_stream, &base_48x32, 37, 8);
        assert_eq!((done_u1.width, done_u1.height), (48, 32));
        assert_eq!(sink_u1.pixels, base_48x32_ref);

        let mut unlimited_buf = JpegDecoderConfig::new();
        unlimited_buf.inner_mut().max_memory = 0;
        let mut inc_unlim_buf = unlimited_buf
            .job()
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (_, done_u2, _) = drive_incremental(&mut inc_unlim_buf, &base_48x32, 37, 8);
        assert_eq!((done_u2.width, done_u2.height), (48, 32));

        //    (a2) Explicit zencodec job caps `Some(0)` MUST refuse (unlike native
        //         DecodeConfig 0 == unlimited) across streaming, buffered, and one-shot.
        let zero_mem = ResourceLimits::none().with_max_memory(0);
        let zero_px = ResourceLimits::none().with_max_pixels(0);
        for (label, lim, expected_kind) in [
            ("job_max_memory_0", zero_mem, zencodec::LimitKind::Memory),
            ("job_max_pixels_0", zero_px, zencodec::LimitKind::Pixels),
        ] {
            // One-shot job
            let os_err = JpegDecoderConfig::new()
                .job()
                .with_limits(lim)
                .decoder(Cow::Borrowed(&base_48x32), &[ZenPixelFormat::RGB8_SRGB])
                .unwrap()
                .decode()
                .expect_err(&format!("one-shot must reject {label}"));
            assert_eq!(
                zencodec::CodecErrorExt::error_category(&os_err),
                Some(zencodec::ErrorCategory::Resource(
                    zencodec::ResourceError::Limits(expected_kind)
                )),
                "one-shot category for {label}"
            );

            // Streaming + config-selected buffered incremental
            for (route, mut dec) in [
                (
                    "stream",
                    nn_cfg
                        .clone()
                        .job()
                        .with_limits(lim)
                        .with_orientation(zencodec::OrientationHint::Preserve)
                        .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                        .unwrap(),
                ),
                (
                    "buffered",
                    JpegDecoderConfig::new()
                        .job()
                        .with_limits(lim)
                        .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                        .unwrap(),
                ),
            ] {
                let mut s = CollectSink::new(8);
                let mut inp = Input::new(0, &base_48x32);
                inp.total_len = Some(base_48x32.len() as u64);
                let mut hit = false;
                for _ in 0..8 {
                    match dec.step(inp, &mut s, None) {
                        Err(zencodec::decode::StepError::Codec(e)) => {
                            assert_eq!(
                                zencodec::CodecErrorExt::error_category(&e),
                                Some(zencodec::ErrorCategory::Resource(
                                    zencodec::ResourceError::Limits(expected_kind)
                                )),
                                "{route} category for {label}"
                            );
                            hit = true;
                            break;
                        }
                        Ok(p) => {
                            let off = (inp.offset as usize + p.consumed).min(base_48x32.len());
                            inp = Input::new(off as u64, &base_48x32[off..]);
                            inp.total_len = Some(base_48x32.len() as u64);
                        }
                        Err(other) => panic!("unexpected {route} error for {label}: {other:?}"),
                    }
                }
                assert!(hit, "{route} must reject {label}");
            }
        }

        //    (b) Config cap 100 + job cap 10_000 resolves to 10_000 on BOTH streaming and
        //        config-selected fallback (matching one-shot build_decode_config), whereas
        //        config cap 100 + no job override rejects 48x32 on BOTH routes.
        let raised = ResourceLimits::none().with_max_pixels(10_000);
        let mut cfg_cap_nn = JpegDecoderConfig::new();
        cfg_cap_nn.inner_mut().chroma_upsampling = crate::decode::ChromaUpsampling::NearestNeighbor;
        cfg_cap_nn.inner_mut().max_pixels = 100;
        let mut cfg_cap_buf = JpegDecoderConfig::new();
        cfg_cap_buf.inner_mut().max_pixels = 100;
        let mut inc_raise_stream = cfg_cap_nn
            .clone()
            .job()
            .with_limits(raised)
            .with_orientation(zencodec::OrientationHint::Preserve)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (_, done_rs, _) = drive_incremental(&mut inc_raise_stream, &base_48x32, 64, 8);
        assert_eq!((done_rs.width, done_rs.height), (48, 32));

        let mut inc_raise_buf = cfg_cap_buf
            .clone()
            .job()
            .with_limits(raised)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let (_, done_rb, _) = drive_incremental(&mut inc_raise_buf, &base_48x32, 64, 8);
        assert_eq!((done_rb.width, done_rb.height), (48, 32));

        for (label, mut dec) in [
            (
                "stream_cfg_cap_100",
                cfg_cap_nn
                    .job()
                    .with_orientation(zencodec::OrientationHint::Preserve)
                    .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                    .unwrap(),
            ),
            (
                "buf_cfg_cap_100",
                cfg_cap_buf
                    .job()
                    .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
                    .unwrap(),
            ),
        ] {
            let mut s = CollectSink::new(8);
            let mut inp = Input::new(0, &base_48x32);
            inp.total_len = Some(base_48x32.len() as u64);
            let mut failed_with_codec = false;
            for _ in 0..8 {
                match dec.step(inp, &mut s, None) {
                    Err(zencodec::decode::StepError::Codec(_)) => {
                        failed_with_codec = true;
                        break;
                    }
                    Ok(p) => {
                        let off = (inp.offset as usize + p.consumed).min(base_48x32.len());
                        inp = Input::new(off as u64, &base_48x32[off..]);
                        inp.total_len = Some(base_48x32.len() as u64);
                    }
                    Err(other) => panic!("unexpected error ({label}): {other:?}"),
                }
            }
            assert!(
                failed_with_codec,
                "expected pixel cap rejection for {label}"
            );
        }

        // 10) Post-scan COM segment growth is admitted against max_memory_bytes
        let mut post_scan_jpeg = base_48x32[..base_48x32.len() - 2].to_vec();
        let com_payload = vec![b'X'; 4096];
        let com_len = (com_payload.len() + 2) as u16;
        post_scan_jpeg.extend_from_slice(&[0xFF, 0xFE]);
        post_scan_jpeg.extend_from_slice(&com_len.to_be_bytes());
        post_scan_jpeg.extend_from_slice(&com_payload);
        post_scan_jpeg.extend_from_slice(&[0xFF, 0xD9]);
        // Budget large enough to stream the 48x32 pixels & header, too small for a 4 KiB post-scan COM.
        let tight_mem = ResourceLimits::none().with_max_memory(3500);
        let mut inc_post_com = nn_cfg
            .clone()
            .job()
            .with_limits(tight_mem)
            .with_orientation(zencodec::OrientationHint::Preserve)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut com_sink = CollectSink::new(8);
        let mut off = 0usize;
        let mut hit_mem_limit = false;
        for _ in 0..256 {
            let end = (off + 128).min(post_scan_jpeg.len());
            let mut inp = Input::new(off as u64, &post_scan_jpeg[off..end]);
            inp.total_len = Some(post_scan_jpeg.len() as u64);
            match inc_post_com.step(inp, &mut com_sink, None) {
                Ok(p) => {
                    off += p.consumed;
                    if matches!(p.event, Event::Done(_)) {
                        break;
                    }
                }
                Err(zencodec::decode::StepError::Codec(e)) => {
                    assert_eq!(
                        zencodec::CodecErrorExt::error_category(&e),
                        Some(zencodec::ErrorCategory::Resource(
                            zencodec::ResourceError::Limits(zencodec::LimitKind::Memory)
                        ))
                    );
                    hit_mem_limit = true;
                    break;
                }
                Err(other) => panic!("unexpected post-scan COM error: {other:?}"),
            }
        }
        assert!(
            hit_mem_limit,
            "post-scan COM growth must hit Memory limit before allocation"
        );

        // 11) Mid-decode cancellation witness (trips inside decode, not on entry)
        struct CancelAfterChecks {
            remaining: core::sync::atomic::AtomicUsize,
        }
        impl enough::Stop for CancelAfterChecks {
            fn check(&self) -> core::result::Result<(), enough::StopReason> {
                let mut cur = self.remaining.load(core::sync::atomic::Ordering::Relaxed);
                loop {
                    if cur == 0 {
                        return Err(enough::StopReason::TimedOut);
                    }
                    match self.remaining.compare_exchange_weak(
                        cur,
                        cur - 1,
                        core::sync::atomic::Ordering::Relaxed,
                        core::sync::atomic::Ordering::Relaxed,
                    ) {
                        Ok(_) => return Ok(()),
                        Err(next) => cur = next,
                    }
                }
            }
        }
        let mid_pulse = zencodec::how_far::StopOnly::new(CancelAfterChecks {
            remaining: core::sync::atomic::AtomicUsize::new(3),
        });
        let mut inc_mid = nn_cfg
            .clone()
            .job()
            .with_orientation(zencodec::OrientationHint::Preserve)
            .into_incremental_decoder(&[ZenPixelFormat::RGB8_SRGB])
            .unwrap();
        let mut mid_sink = CollectSink::new(8);
        let mut mid_off = 0usize;
        let mut saw_mid_cancel = false;
        for _ in 0..64 {
            let end = (mid_off + 64).min(base_48x32.len());
            let mut inp = Input::new(mid_off as u64, &base_48x32[mid_off..end]);
            inp.total_len = Some(base_48x32.len() as u64);
            match inc_mid.step(inp, &mut mid_sink, Some(&mid_pulse)) {
                Ok(p) => {
                    mid_off += p.consumed;
                    if matches!(p.event, Event::Done(_)) {
                        break;
                    }
                }
                Err(zencodec::decode::StepError::Cancelled(enough::StopReason::TimedOut)) => {
                    saw_mid_cancel = true;
                    break;
                }
                Err(other) => panic!("expected mid-decode TimedOut cancellation, got {other:?}"),
            }
        }
        assert!(saw_mid_cancel, "mid-decode call pulse must trip TimedOut");
    }

    #[cfg(feature = "ultrahdr")]
    #[test]
    fn incremental_ultrahdr_reconstruct_cancels_in_flight() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        use zencodec::decode::{
            DecodeJob as _, DecodeRowSink, DecoderConfig as _, Event, IncrementalDecode as _,
            Input, RowStrip, SinkError, StepError,
        };
        use zenpixels::{PixelBuf, PixelFormat as ZenPixelFormat, PixelViewMut};

        // Build a synthetic Ultra HDR JPEG (32x32 linear HDR -> encode_ultrahdr_luma)
        let mut hdr_bytes = Vec::with_capacity(32 * 32 * 16);
        for y in 0..32u32 {
            for x in 0..32u32 {
                let r = (x as f32 / 31.0) * 2.5;
                let g = (y as f32 / 31.0) * 1.8;
                let b = 0.4_f32;
                let a = 1.0_f32;
                for c in [r, g, b, a] {
                    hdr_bytes.extend_from_slice(&c.to_ne_bytes());
                }
            }
        }
        let hdr = PixelBuf::from_bytes(hdr_bytes, 32, 32, ZenPixelFormat::RGBAF32_LINEAR).unwrap();
        let uhdr_jpeg = crate::ultrahdr::encode_ultrahdr_luma(&hdr).unwrap();

        // Count how many stop checks a full ReconstructHdr decode performs, then trip inside
        // the auxiliary gain-map decode / apply_gainmap phase (well past step entry).
        struct CountingStop {
            count: std::sync::Arc<AtomicUsize>,
            limit: usize,
        }
        impl enough::Stop for CountingStop {
            fn check(&self) -> core::result::Result<(), enough::StopReason> {
                let n = self.count.fetch_add(1, Ordering::Relaxed);
                if n >= self.limit {
                    Err(enough::StopReason::TimedOut)
                } else {
                    Ok(())
                }
            }
        }
        struct SinkStub;
        impl DecodeRowSink for SinkStub {
            fn begin(&mut self, _: u32, _: u32, _: ZenPixelFormat) -> Result<(), SinkError> {
                Ok(())
            }
            fn max_buffer_rows(&self) -> core::num::NonZeroU32 {
                core::num::NonZeroU32::new(16).unwrap()
            }
            fn provide_next_buffer(
                &mut self,
                _: RowStrip,
                _: ZenPixelFormat,
            ) -> Result<PixelViewMut<'_>, SinkError> {
                unreachable!("should cancel before delivering HDR rows")
            }
            fn finish(&mut self) -> Result<(), SinkError> {
                Ok(())
            }
        }

        // First count total checks during full incremental ReconstructHdr decode.
        let count_shared = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = zencodec::how_far::StopOnly::new(CountingStop {
            count: std::sync::Arc::clone(&count_shared),
            limit: usize::MAX,
        });
        let mut full_dec = JpegDecoderConfig::new()
            .job()
            .with_gain_map_render(zencodec::GainMapRender::ReconstructHdr {
                target_headroom: None,
            })
            .into_incremental_decoder(&[ZenPixelFormat::RGBAF32_LINEAR])
            .unwrap();
        let mut bs = CollectSink::new(32);
        let mut off = 0usize;
        loop {
            let mut inp = Input::new(off as u64, &uhdr_jpeg[off..]);
            inp.total_len = Some(uhdr_jpeg.len() as u64);
            let p = full_dec.step(inp, &mut bs, Some(&counter)).unwrap();
            off += p.consumed;
            if matches!(p.event, Event::Done(_)) {
                break;
            }
        }
        let total_checks = count_shared.load(Ordering::Relaxed);
        // Base decode + auxiliary gain-map decode + 32 apply_gainmap rows must perform > 32 checks.
        assert!(
            total_checks > 32,
            "expected >32 stop checks across Ultra HDR decode + apply_gainmap, got {total_checks}"
        );

        // Trip near the end (inside apply_gainmap, well after entry and base decode!)
        let late_trip = total_checks - 10;
        let cancel_pulse = zencodec::how_far::StopOnly::new(CountingStop {
            count: std::sync::Arc::new(AtomicUsize::new(0)),
            limit: late_trip,
        });
        let mut dec = JpegDecoderConfig::new()
            .job()
            .with_gain_map_render(zencodec::GainMapRender::ReconstructHdr {
                target_headroom: None,
            })
            .into_incremental_decoder(&[ZenPixelFormat::RGBAF32_LINEAR])
            .unwrap();
        let mut stub = SinkStub;
        let mut inp = Input::new(0, &uhdr_jpeg);
        inp.total_len = Some(uhdr_jpeg.len() as u64);
        let mut got_cancel = false;
        for _ in 0..8 {
            match dec.step(inp, &mut stub, Some(&cancel_pulse)) {
                Ok(p) => {
                    let next = (inp.offset as usize + p.consumed).min(uhdr_jpeg.len());
                    inp = Input::new(next as u64, &uhdr_jpeg[next..]);
                    inp.total_len = Some(uhdr_jpeg.len() as u64);
                    if matches!(p.event, Event::Done(_)) {
                        break;
                    }
                }
                Err(StepError::Cancelled(enough::StopReason::TimedOut)) => {
                    got_cancel = true;
                    break;
                }
                Err(other) => {
                    panic!("expected Cancelled(TimedOut) from apply_gainmap, got {other:?}")
                }
            }
        }
        assert!(
            got_cancel,
            "in-flight Ultra HDR cancellation inside apply_gainmap must surface StepError::Cancelled"
        );
    }
}
