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

/// Default (BaseOnly): a plain SDR decode — 8-bit buffer, no gain-map extras,
/// and the info reports the Ultra HDR container via `supplements.gain_map`
/// so callers can gate a ReconstructHdr pass on the base decode alone.
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
    assert!(
        out.info().supplements.gain_map,
        "Ultra HDR container must be reported in supplements.gain_map"
    );
}

/// A plain (non-gain-map) JPEG reports no gain map in supplements.
#[test]
fn plain_jpeg_reports_no_gain_map_supplement() {
    use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
    let pixels: Vec<u8> = core::iter::repeat([10u8, 20, 30])
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
        .decoder(plain.data().to_vec().into(), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert!(!out.info().supplements.gain_map);
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

// ── Orientation pairing (#151) ──────────────────────────────────────────────

/// Non-square quadrant fixture (32×24): four distinct linear levels, so the
/// gain field differs under every non-identity EXIF orientation (catches
/// flips and 180 as well as the dimension-swapping transposes).
fn hdr_quadrant_fixture() -> PixelBuffer {
    let (w, h) = (32u32, 24u32);
    let mut px = Vec::with_capacity((w * h) as usize * 4);
    for y in 0..h {
        for x in 0..w {
            let v = match (x < w / 2, y < h / 2) {
                (true, true) => 0.25f32,
                (false, true) => 1.0,
                (true, false) => 2.0,
                (false, false) => 4.0,
            };
            px.extend_from_slice(&[v, v, v, 1.0]);
        }
    }
    let bytes: Vec<u8> = px.iter().flat_map(|f| f.to_ne_bytes()).collect();
    PixelBuffer::from_vec(bytes, w, h, PixelDescriptor::RGBAF32_LINEAR).expect("fixture buffer")
}

/// Insert a minimal EXIF APP1 (TIFF carrying only the orientation tag)
/// directly after SOI.
fn with_exif_orientation(jpeg: &[u8], orientation: u8) -> Vec<u8> {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "fixture must start with SOI");
    let mut tiff = Vec::new();
    tiff.extend_from_slice(b"II"); // little-endian
    tiff.extend_from_slice(&42u16.to_le_bytes()); // TIFF magic
    tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 offset
    tiff.extend_from_slice(&1u16.to_le_bytes()); // one entry
    tiff.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation tag
    tiff.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    tiff.extend_from_slice(&1u32.to_le_bytes()); // count
    tiff.extend_from_slice(&u16::from(orientation).to_le_bytes());
    tiff.extend_from_slice(&[0, 0]); // value padding
    tiff.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    let payload_len = (2 + 6 + tiff.len()) as u16; // len field + "Exif\0\0" + TIFF
    let mut out = Vec::with_capacity(jpeg.len() + payload_len as usize + 2);
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&payload_len.to_be_bytes());
    out.extend_from_slice(b"Exif\0\0");
    out.extend_from_slice(&tiff);
    out.extend_from_slice(&jpeg[2..]);
    out
}

fn diff_summary(a: &[u8], b: &[u8]) -> String {
    if a.len() != b.len() {
        return format!("lengths differ: {} vs {}", a.len(), b.len());
    }
    let n = a.iter().zip(b).filter(|(x, y)| x != y).count();
    format!("{n} of {} bytes differ", a.len())
}

/// ReconstructHdr under a baking orientation hint must equal the Preserve
/// reconstruction followed by an external pixel-domain bake — both are pure
/// permutations of the same stored-space gain-map application (#151).
#[test]
fn reconstruct_hdr_exif_oriented_matches_preserve_plus_bake() {
    let base_jpeg = encode_ultrahdr_luma(&hdr_quadrant_fixture()).expect("ultrahdr encode");
    for exif in 1..=8u8 {
        let jpeg = with_exif_orientation(&base_jpeg, exif);
        let orientation = zencodec::Orientation::from_exif(exif).expect("valid EXIF code");

        let decode = |hint| {
            JpegDecoderConfig::new()
                .job()
                .with_orientation(hint)
                .with_gain_map_render(zencodec::GainMapRender::ReconstructHdr {
                    target_headroom: None,
                })
                .decoder(jpeg.as_slice().into(), &[])
                .unwrap()
                .decode()
                .unwrap()
        };
        let corrected = decode(zencodec::OrientationHint::Correct);
        let preserved = decode(zencodec::OrientationHint::Preserve);

        // Sanity: the injected EXIF is honored — transposing codes swap dims.
        if orientation.swaps_axes() {
            assert_eq!(
                (corrected.pixels().width(), corrected.pixels().rows()),
                (preserved.pixels().rows(), preserved.pixels().width()),
                "EXIF {exif}: Correct must report display dimensions"
            );
        }

        let baked = zenpixels_convert::orient::apply_orientation(preserved.pixels(), orientation);
        assert_eq!(
            (corrected.pixels().width(), corrected.pixels().rows()),
            (baked.width(), baked.height()),
            "EXIF {exif}: reconstructed dimensions"
        );
        let a = corrected.pixels().contiguous_bytes();
        let b = baked.as_slice().contiguous_bytes();
        assert!(
            a == b,
            "EXIF {exif}: ReconstructHdr+Correct must be byte-identical to \
             Preserve + external bake ({})",
            diff_summary(&a, &b)
        );
    }
}

/// The reconstructed HDR quadrants must land where EXIF display semantics
/// say they land — independent of the permutation implementation used by
/// either decode path (#151).
#[test]
fn reconstruct_hdr_exif_oriented_quadrants_land_correctly() {
    // Display-space corner layout (TL, TR, BL, BR) of the quadrant fixture
    // (stored TL=0.25, TR=1.0, BL=2.0, BR=4.0) under each EXIF code, derived
    // by hand from the EXIF orientation definitions.
    let expected: [(u8, [f32; 4]); 8] = [
        (1, [0.25, 1.0, 2.0, 4.0]),
        (2, [1.0, 0.25, 4.0, 2.0]), // flip horizontal
        (3, [4.0, 2.0, 1.0, 0.25]), // rotate 180
        (4, [2.0, 4.0, 0.25, 1.0]), // flip vertical
        (5, [0.25, 2.0, 1.0, 4.0]), // transpose
        (6, [2.0, 0.25, 4.0, 1.0]), // rotate 90 CW
        (7, [4.0, 1.0, 2.0, 0.25]), // transverse
        (8, [1.0, 4.0, 0.25, 2.0]), // rotate 270 CW
    ];
    let base_jpeg = encode_ultrahdr_luma(&hdr_quadrant_fixture()).expect("ultrahdr encode");
    for (exif, corners) in expected {
        let jpeg = with_exif_orientation(&base_jpeg, exif);
        let out = JpegDecoderConfig::new()
            .job()
            .with_orientation(zencodec::OrientationHint::Correct)
            .with_gain_map_render(zencodec::GainMapRender::ReconstructHdr {
                target_headroom: None,
            })
            .decoder(jpeg.as_slice().into(), &[])
            .unwrap()
            .decode()
            .unwrap();
        let (w, h) = (out.pixels().width() as usize, out.pixels().rows() as usize);
        let bytes = out.pixels().contiguous_bytes();
        // Red channel of the RGBA f32 pixel at quadrant centers.
        let sample = |x: usize, y: usize| -> f32 {
            let i = (y * w + x) * 16;
            f32::from_ne_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
        };
        let got = [
            sample(w / 4, h / 4),
            sample(3 * w / 4, h / 4),
            sample(w / 4, 3 * h / 4),
            sample(3 * w / 4, 3 * h / 4),
        ];
        // Rank-compare: the lossy round trip shifts absolute values, but the
        // 2x spacing keeps the ordering unambiguous at quadrant centers.
        fn rank(vals: [f32; 4]) -> [usize; 4] {
            let mut idx = [0usize, 1, 2, 3];
            idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
            let mut r = [0usize; 4];
            for (pos, &i) in idx.iter().enumerate() {
                r[i] = pos;
            }
            r
        }
        assert_eq!(
            rank(got),
            rank(corners),
            "EXIF {exif}: quadrant ordering wrong — got {got:?}, expected layout {corners:?}"
        );
    }
}

/// Components under a baking hint must surface the gain map in the same
/// orientation as the returned base buffer (#151).
#[test]
fn components_gain_map_oriented_with_base() {
    let base_jpeg = encode_ultrahdr_luma(&hdr_quadrant_fixture()).expect("ultrahdr encode");
    for exif in 2..=8u8 {
        let jpeg = with_exif_orientation(&base_jpeg, exif);
        let orientation = zencodec::Orientation::from_exif(exif).expect("valid EXIF code");

        let decode = |hint| {
            JpegDecoderConfig::new()
                .job()
                .with_orientation(hint)
                .with_gain_map_render(zencodec::GainMapRender::Components)
                .decoder(jpeg.as_slice().into(), &[])
                .unwrap()
                .decode()
                .unwrap()
        };
        let corrected = decode(zencodec::OrientationHint::Correct);
        let preserved = decode(zencodec::OrientationHint::Preserve);

        let dgm_c = corrected
            .extras::<zencodec::decode::DecodedGainMap>()
            .expect("Components must surface the DecodedGainMap");
        let dgm_p = preserved
            .extras::<zencodec::decode::DecodedGainMap>()
            .expect("Components must surface the DecodedGainMap");

        let expected =
            zenpixels_convert::orient::apply_orientation(dgm_p.pixels.as_slice(), orientation);
        assert_eq!(
            (dgm_c.pixels.width(), dgm_c.pixels.height()),
            (expected.width(), expected.height()),
            "EXIF {exif}: gain-map dimensions must follow the base orientation"
        );
        let a = dgm_c.pixels.as_slice().contiguous_bytes();
        let b = expected.as_slice().contiguous_bytes();
        assert!(
            a == b,
            "EXIF {exif}: Components gain map must be oriented with the base ({})",
            diff_summary(&a, &b)
        );

        // And the surfaced pair is aspect-consistent (not transposed).
        let (bw, bh) = (
            f64::from(corrected.pixels().width()),
            f64::from(corrected.pixels().rows()),
        );
        let (gw, gh) = (
            f64::from(dgm_c.pixels.width()),
            f64::from(dgm_c.pixels.height()),
        );
        assert!(
            (bw / bh - gw / gh).abs() <= (bw / bh - gh / gw).abs(),
            "EXIF {exif}: base {bw}x{bh} vs gain map {gw}x{gh} aspect-inconsistent"
        );
    }
}

/// Probes honor `with_limits` like the decode path does: a header whose
/// dimensions exceed the inner config's default pixel cap probes fine when
/// the job raises `max_pixels` (and still fails without it).
#[test]
fn probe_honors_job_limits() {
    // Patch a real fixture's SOF dimensions up to 12000x11000 (132 MP) —
    // read_info only parses headers, so no scan data needs to match.
    let mut jpeg = ultrahdr_jpeg();
    let sof = jpeg
        .windows(2)
        .position(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC2))
        .expect("SOF marker");
    jpeg[sof + 5..sof + 7].copy_from_slice(&11000u16.to_be_bytes()); // height
    jpeg[sof + 7..sof + 9].copy_from_slice(&12000u16.to_be_bytes()); // width

    let probe_default = JpegDecoderConfig::new().job().probe(&jpeg);
    assert!(
        probe_default.is_err(),
        "132 MP must exceed the default probe cap (precondition)"
    );

    let info = JpegDecoderConfig::new()
        .job()
        .with_limits(zencodec::ResourceLimits::default().with_max_pixels(1_000_000_000))
        .probe(&jpeg)
        .expect("probe with raised max_pixels");
    assert_eq!((info.width, info.height), (12000, 11000));
}
