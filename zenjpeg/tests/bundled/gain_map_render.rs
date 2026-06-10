//! Exercises `zencodec::GainMapRender` through the `JpegDecodeJob` trait path:
//! `BaseOnly` (default SDR decode), `Components` (surface the decoded gain map
//! as `zencodec::decode::DecodedGainMap` extras), and `ReconstructHdr` (zenjpeg
//! applies the gain map itself — `DecodeCapabilities::reconstructs_hdr()`).
//!
//! The fixture is built in-test: a linear HDR buffer (values above 1.0 = above
//! SDR white) → `ultrahdr::encode_ultrahdr_luma` → Ultra HDR JPEG.

#![cfg(all(feature = "zencodec", feature = "ultrahdr"))]

use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zenjpeg::JpegDecoderConfig;
use zenjpeg::ultrahdr::encode_ultrahdr_luma;
use zenpixels::{ChannelType, PixelBuffer, PixelDescriptor, TransferFunction};

/// 32×32 linear HDR RGBA f32 fixture: left half SDR gray (0.25), right half
/// 4× SDR white — enough headroom for the gain map to be meaningfully nonzero.
fn hdr_fixture() -> PixelBuffer {
    let w = 32u32;
    let h = 32u32;
    let mut px = Vec::with_capacity((w * h) as usize * 4);
    for y in 0..h {
        for x in 0..w {
            let v = if x < w / 2 { 0.25f32 } else { 4.0f32 };
            let _ = y;
            px.extend_from_slice(&[v, v, v, 1.0]);
        }
    }
    let bytes: Vec<u8> = px.iter().flat_map(|f| f.to_ne_bytes()).collect();
    PixelBuffer::from_vec(bytes, w, h, PixelDescriptor::RGBAF32_LINEAR).expect("fixture buffer")
}

fn ultrahdr_jpeg() -> Vec<u8> {
    encode_ultrahdr_luma(&hdr_fixture()).expect("ultrahdr encode")
}

/// Default (BaseOnly): a plain SDR decode — 8-bit buffer, no gain-map extras.
#[test]
fn base_only_default_decodes_sdr() {
    let jpeg = ultrahdr_jpeg();
    let out = JpegDecoderConfig::new()
        .job()
        .decoder(jpeg.as_slice().into(), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert!(
        out.extras::<zencodec::decode::DecodedGainMap>().is_none(),
        "BaseOnly must not surface a DecodedGainMap"
    );
    assert_eq!(
        out.pixels().descriptor().channel_type(),
        ChannelType::U8,
        "BaseOnly output is the SDR base image"
    );
}

/// ReconstructHdr: zenjpeg applies the gain map — linear f32 output with
/// above-SDR-white values, and the envelope (CLL + mastering display) is
/// populated on the output info per the GainMapRender contract.
#[test]
fn reconstruct_hdr_produces_linear_hdr_with_envelope() {
    let jpeg = ultrahdr_jpeg();
    assert!(
        <JpegDecoderConfig as DecoderConfig>::capabilities().reconstructs_hdr(),
        "zenjpeg with the ultrahdr feature must declare reconstructs_hdr"
    );
    let out = JpegDecoderConfig::new()
        .job()
        .with_gain_map_render(zencodec::GainMapRender::ReconstructHdr {
            target_headroom: None,
        })
        .decoder(jpeg.as_slice().into(), &[])
        .unwrap()
        .decode()
        .unwrap();

    let desc = out.pixels().descriptor();
    assert_eq!(desc.channel_type(), ChannelType::F32, "linear HDR output");
    assert_eq!(desc.transfer(), TransferFunction::Linear);

    // Above-SDR-white values must exist (the right half of the fixture was
    // 4× SDR white; lossy encode + gain-map quantization keep it well > 1).
    let bytes = out.pixels().contiguous_bytes();
    let max = bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .fold(0.0f32, f32::max);
    assert!(
        max > 1.5,
        "reconstructed HDR must exceed SDR white (max = {max})"
    );

    // Envelope obligation.
    let sc = &out.info().source_color;
    let cll = sc
        .content_light_level
        .expect("ReconstructHdr must populate content_light_level");
    assert!(
        cll.max_content_light_level > 203,
        "derived peak above SDR white nits"
    );
    assert!(
        sc.mastering_display.is_some(),
        "ReconstructHdr must populate mastering_display"
    );
}

/// Components: the SDR base decodes normally AND the decoded gain map is
/// surfaced as `DecodedGainMap` extras (pixels + ISO 21496-1 params).
#[test]
fn components_surfaces_decoded_gain_map() {
    let jpeg = ultrahdr_jpeg();
    let out = JpegDecoderConfig::new()
        .job()
        .with_gain_map_render(zencodec::GainMapRender::Components)
        .decoder(jpeg.as_slice().into(), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(
        out.pixels().descriptor().channel_type(),
        ChannelType::U8,
        "Components keeps the SDR base as the primary buffer"
    );
    let dgm = out
        .extras::<zencodec::decode::DecodedGainMap>()
        .expect("Components must surface the DecodedGainMap");
    assert!(dgm.pixels.width() > 0 && dgm.pixels.height() > 0);
    assert!(
        dgm.metadata.params.alternate_hdr_headroom > 0.0,
        "gain map must carry a real alternate headroom"
    );
}

/// ReconstructHdr on a plain (non-gain-map) JPEG: the base image IS the
/// image — decodes normally instead of erroring.
#[test]
fn reconstruct_on_plain_jpeg_decodes_base() {
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    let pixels: Vec<u8> = core::iter::repeat([64u8, 128, 192])
        .take(64)
        .flatten()
        .collect();
    let slice = zenpixels::PixelSlice::new(&pixels, 8, 8, 8 * 3, PixelDescriptor::RGB8_SRGB)
        .expect("pixel slice");
    let plain = zenjpeg::JpegEncoderConfig::new()
        .job()
        .encoder()
        .unwrap()
        .encode(slice)
        .unwrap();

    let out = JpegDecoderConfig::new()
        .job()
        .with_gain_map_render(zencodec::GainMapRender::ReconstructHdr {
            target_headroom: Some(4.0),
        })
        .decoder(plain.data().to_vec().into(), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(out.pixels().width(), 8);
    assert_eq!(out.pixels().descriptor().channel_type(), ChannelType::U8);
}
