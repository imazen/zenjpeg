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
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice, PixelSliceMut};

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
        let output = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
        assert!(!output.data().is_empty());
        assert_eq!(output.format(), ImageFormat::Jpeg);
        assert_eq!(&output.data()[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn encoding_with_metadata() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels.as_slice(), 4, 4);

        let icc = b"fake icc profile data";
        let meta = Metadata::default().with_icc(icc.as_slice());
        let output = enc
            .job()
            .with_metadata_policy(meta, zencodec::MetadataPolicy::PreserveExact)
            .encoder()
            .unwrap()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();
        assert!(!output.data().is_empty());
    }

    #[test]
    fn encoding_with_policy_strips_metadata() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 255, g: 0, b: 0 }; 16];
        let img = Img::new(pixels.as_slice(), 4, 4);

        let icc = b"fake icc profile data";
        let meta = Metadata::default().with_icc(icc.as_slice());
        let policy = zencodec::encode::EncodePolicy::strip_all();

        let output = enc
            .job()
            .with_metadata_policy(meta, zencodec::MetadataPolicy::PreserveExact)
            .with_policy(policy)
            .encoder()
            .unwrap()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();
        // Should succeed but ICC may be stripped by strict policy
        assert!(!output.data().is_empty());
    }

    #[test]
    fn encoding_gray8() {
        let enc = JpegEncoderConfig::grayscale(90.0);
        let pixels = vec![Gray::new(128u8); 64];
        let img = Img::new(pixels.as_slice(), 8, 8);
        let output = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
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
        let output = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
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
        let slice: PixelSlice<'_> = PixelSlice::from(img.as_ref()).into();

        let mut encoder = enc.job().encoder().unwrap();
        let top = slice.sub_rows(0, 4);
        let bottom = slice.sub_rows(4, 4);
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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        let dec = JpegDecoderConfig::new();
        let output = dec
            .job()
            .decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap()
            .decode()
            .unwrap();
        // Output should be RGB8 — the native format
        assert_eq!(output.descriptor(), PixelDescriptor::RGB8_SRGB);
        let pixel_data = output.pixels();
        assert_eq!(pixel_data.width(), 8);
        assert_eq!(pixel_data.rows(), 8);
        // 8*8*3 = 192 bytes
        assert!(pixel_data.as_contiguous_bytes().is_some());
    }

    #[test]
    fn probe_info() {
        let enc = JpegEncoderConfig::new().with_calibrated_quality(85.0);
        let pixels: Vec<Rgb<u8>> = vec![Rgb { r: 0, g: 0, b: 0 }; 100];
        let img = Img::new(pixels.as_slice(), 10, 10);
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        assert_eq!(stream.info().width, 16);
        assert_eq!(stream.info().height, 16);

        let mut total_rows = 0u32;
        while let Some((y, batch)) = stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            assert_eq!(batch.width(), 16);
            // Each batch should be MCU-row sized (multiple rows)
            assert!(batch.rows() >= 1);
            total_rows += batch.rows();
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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        let mut batch_count = 0;
        let mut total_rows = 0u32;
        while let Some((_y, batch)) = stream.next_batch().unwrap() {
            batch_count += 1;
            total_rows += batch.rows();
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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // First decode with Cow::Borrowed as reference
        let dec = JpegDecoderConfig::new();
        let mut borrowed_stream = dec
            .job()
            .streaming_decoder(Cow::Borrowed(encoded.data()), &[PixelDescriptor::RGB8_SRGB])
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
            .streaming_decoder(Cow::Owned(owned_data), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        assert_eq!(owned_stream.info().width, 32);
        assert_eq!(owned_stream.info().height, 32);

        let mut owned_pixels = Vec::new();
        let mut total_rows = 0u32;
        while let Some((y, batch)) = owned_stream.next_batch().unwrap() {
            assert_eq!(y, total_rows);
            owned_pixels.extend_from_slice(batch.as_strided_bytes());
            total_rows += batch.rows();
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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // Create streaming decoder with owned data inside a scope,
        // then use it outside that scope (proves no external borrow).
        let owned_data = encoded.data().to_vec();
        let dec = JpegDecoderConfig::new();
        let mut stream = dec
            .job()
            .streaming_decoder(Cow::Owned(owned_data), &[PixelDescriptor::RGB8_SRGB])
            .unwrap();

        // The stream should work after the owned_data variable is consumed
        let mut total_rows = 0u32;
        while let Some((_y, batch)) = stream.next_batch().unwrap() {
            total_rows += batch.rows();
        }
        assert_eq!(total_rows, 16);
    }

    // ── Encoder trait roundtrip tests ────────────────────────────────

    fn encoder_trait_roundtrip(pixels: zenpixels::PixelSlice<'_>) {
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
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
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
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_gray8() {
        let pixels: Vec<Gray<u8>> = (0..16 * 16).map(|i| Gray((i % 256) as u8)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
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
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
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
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_gray16() {
        let pixels: Vec<Gray<u16>> = (0..16 * 16).map(|i| Gray((i * 256) as u16)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
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
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
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
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
    }

    #[test]
    fn encoder_trait_gray_f32() {
        let pixels: Vec<Gray<f32>> = (0..16 * 16).map(|i| Gray(i as f32 / 255.0)).collect();
        let img = Img::new(pixels.as_slice(), 16, 16);
        encoder_trait_roundtrip(zenpixels::PixelSlice::from(img.as_ref()).into());
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
        let output = dyn_enc
            .encode(zenpixels::PixelSlice::from(img.as_ref()).into())
            .unwrap();
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
        let encoded = JpegEncoderConfig::new()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();

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
        let encoded = JpegEncoderConfig::new()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();

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
        let encoded = JpegEncoderConfig::new()
            .encode(PixelSlice::from(img.as_ref()).into())
            .unwrap();

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
            .encode(PixelSlice::from(img.as_ref()).into())
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
            .encode(PixelSlice::from(img.as_ref()).into())
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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();

        // Use the full supported descriptor list (includes f32 types)
        let dec = JpegDecoderConfig::new();
        let preferred = JpegDecoderConfig::supported_descriptors();

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
        use zenpixels::PixelSliceMut;

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
            .encode_from(&mut |y, mut buf: PixelSliceMut<'_>| {
                let rows = buf.rows();
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
        use zenpixels::PixelSliceMut;

        let config = JpegEncoderConfig::new();
        // No with_canvas_size — should error
        let encoder = config.job().encoder().unwrap();
        let result = encoder.encode_from(&mut |_y, _buf: PixelSliceMut<'_>| 0);
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
            .with_stop(zencodec::StopToken::new(stop)) // owned token
            .with_canvas_size(64, 64);
        let mut enc = job.dyn_encoder().unwrap();
        // enc borrows stop, both in same scope — compiles fine
        let pixels = vec![128u8; 64 * 64 * 4];
        let slice = zenpixels::PixelSlice::new(
            &pixels,
            64,
            64,
            64 * 4,
            zenpixels::PixelDescriptor::RGBA8_SRGB,
        )
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
        assert_eq!(pixels.descriptor().bytes_per_pixel(), 4);

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
        let actual_bytes = pixels.width() as usize * pixels.rows() as usize * 4;
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
        assert_eq!(output_rgb.pixels().descriptor().bytes_per_pixel(), 3);
        assert_eq!(output_cmyk.pixels().descriptor().bytes_per_pixel(), 4);

        // Manual CMYK→RGB conversion on raw CMYK data should produce something
        // close to the auto-converted RGB (within the known maxDelta of ~39)
        let w = cmyk_info.width as usize;
        let h = cmyk_info.height as usize;
        let cmyk_pixels = output_cmyk.pixels();
        let cmyk_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 4);
            for row in 0..cmyk_pixels.rows() {
                all.extend_from_slice(cmyk_pixels.row(row));
            }
            all
        };
        let rgb_pixels = output_rgb.pixels();
        let rgb_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 3);
            for row in 0..rgb_pixels.rows() {
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
        let encoded = enc.encode(PixelSlice::from(img.as_ref()).into()).unwrap();
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
            out_passthrough.pixels().descriptor().bytes_per_pixel(),
            out_badrgb.pixels().descriptor().bytes_per_pixel()
        );
        assert_eq!(out_passthrough.info().width, out_badrgb.info().width);
        assert_eq!(out_passthrough.info().height, out_badrgb.info().height);

        let px_a = out_passthrough.pixels();
        let px_b = out_badrgb.pixels();
        assert_eq!(px_a.rows(), px_b.rows());
        for row in 0..px_a.rows() {
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
        assert_eq!(output_cmyk.pixels().descriptor().bytes_per_pixel(), 4);

        // Same dimensions
        assert_eq!(output_rgb.info().width, cmyk_info.width);
        assert_eq!(output_rgb.info().height, cmyk_info.height);

        let w = cmyk_info.width as usize;
        let h = cmyk_info.height as usize;
        let cmyk_pixels = output_cmyk.pixels();
        let cmyk_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 4);
            for row in 0..cmyk_pixels.rows() {
                all.extend_from_slice(cmyk_pixels.row(row));
            }
            all
        };
        let rgb_pixels = output_rgb.pixels();
        let rgb_bytes: Vec<u8> = {
            let mut all = Vec::with_capacity(w * h * 3);
            for row in 0..rgb_pixels.rows() {
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
        fn begin(&mut self, w: u32, h: u32, _desc: PixelDescriptor) -> Result<(), SinkError> {
            self.w = w;
            self.h = h;
            self.stride = ((w as usize * 4 + 31) / 32) * 32;
            self.data = vec![0xCC; self.stride * h as usize];
            Ok(())
        }
        fn provide_next_buffer(
            &mut self,
            y: u32,
            height: u32,
            width: u32,
            descriptor: PixelDescriptor,
        ) -> Result<zenpixels::PixelSliceMut<'_>, SinkError> {
            let row_start = y as usize * self.stride;
            let row_bytes = width as usize * 4;
            let needed = if height > 0 {
                (height as usize - 1) * self.stride + row_bytes
            } else {
                0
            };
            let slice = &mut self.data[row_start..row_start + needed];
            zenpixels::PixelSliceMut::new(slice, width, height, self.stride, descriptor)
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

        let preferred = [PixelDescriptor::BGRA8_SRGB];
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
        let preferred = [PixelDescriptor::BGRA8_SRGB];
        let cfg1 = JpegDecoderConfig::new();
        let job1 = cfg1.dyn_job();
        let dec = job1.into_decoder(Cow::Borrowed(&jpeg), &preferred).unwrap();
        let output = dec.decode().unwrap();
        let ps = output.pixels();
        let buffered: Vec<u8> = (0..ps.rows()).flat_map(|y| ps.row(y).to_vec()).collect();

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
            (0..ps.rows()).flat_map(|y| ps.row(y).to_vec()).collect()
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

        let compute = ComputeEnvironment::new();
        let small = ImageCharacteristics::new(256, 256, PixelDescriptor::RGB8_SRGB);
        let large = ImageCharacteristics::new(2048, 2048, PixelDescriptor::RGB8_SRGB);

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
