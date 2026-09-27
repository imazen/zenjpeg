//! MPF secondary images (UltraHDR gain maps, depth maps, MPF thumbnails)
//! through the lossless pipeline: `transform` and `apply_exif_orientation`
//! must carry every secondary through, transformed identically, with the MPF
//! index and the GContainer `Item:Length` rebuilt for the new byte layout.
//!
//! Fixtures are synthetic Multi-Picture JPEGs built with zenjpeg's own
//! encoder (the same assembly `ultrahdr::encode_with_gainmap` uses); no
//! `ultrahdr` feature is needed because the container code is unconditional.

use enough::Unstoppable;
use zenjpeg::container::xmp::{generate_primary_xmp, parse_xmp_full};
use zenjpeg::decode::{DecodeConfig, PreserveConfig};
use zenjpeg::encode::EncoderSegments;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Exif, Orientation, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, TransformConfig, apply_exif_orientation, transform,
};

/// Decode with `config`, return (width, height, packed RGB8 or Gray8 pixels).
fn decode_test(jpeg: &[u8], config: &DecodeConfig) -> (u32, u32, Vec<u8>) {
    let result = config.decode(jpeg, Unstoppable).unwrap();
    let (w, h) = (result.width(), result.height());
    (w, h, result.into_pixels_u8().unwrap())
}

/// Reference pixel-domain transform of a `w`×`h` image with `channels` bytes
/// per pixel. Returns (new_w, new_h, pixels).
fn pixel_transform(
    pixels: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    transform: LosslessTransform,
) -> (usize, usize, Vec<u8>) {
    let (out_w, out_h) = if transform.swaps_dimensions() {
        (h, w)
    } else {
        (w, h)
    };
    let mut out = vec![0u8; out_w * out_h * channels];
    for sy in 0..h {
        for sx in 0..w {
            let (dx, dy) = match transform {
                LosslessTransform::None => (sx, sy),
                LosslessTransform::FlipHorizontal => (w - 1 - sx, sy),
                LosslessTransform::FlipVertical => (sx, h - 1 - sy),
                LosslessTransform::Rotate180 => (w - 1 - sx, h - 1 - sy),
                LosslessTransform::Transpose => (sy, sx),
                LosslessTransform::Rotate90 => (h - 1 - sy, sx),
                LosslessTransform::Rotate270 => (sy, w - 1 - sx),
                LosslessTransform::Transverse => (h - 1 - sy, w - 1 - sx),
            };
            let si = (sy * w + sx) * channels;
            let di = (dy * out_w + dx) * channels;
            out[di..di + channels].copy_from_slice(&pixels[si..si + channels]);
        }
    }
    (out_w, out_h, out)
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i16 - *y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Walk the marker segments of a JPEG up to SOS: `f(marker, offset_of_ff, payload)`.
fn for_each_segment(jpeg: &[u8], mut f: impl FnMut(u8, usize, &[u8])) {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "missing SOI");
    let mut i = 2;
    while i + 4 <= jpeg.len() {
        assert_eq!(jpeg[i], 0xFF, "expected marker at {i}");
        let marker = jpeg[i + 1];
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        f(marker, i, &jpeg[i + 4..i + 2 + len]);
        if marker == 0xDA {
            break;
        }
        i += 2 + len;
    }
}

/// Build a synthetic Multi-Picture JPEG the way `ultrahdr::encode_with_gainmap`
/// does: a colour primary carrying a GContainer XMP (`Item:Length` = gain-map
/// byte length) and an MPF index, followed by a grayscale secondary. No
/// `ultrahdr` feature needed; the container code is unconditional.
fn synthetic_mpf_jpeg(pw: u32, ph: u32, sw: u32, sh: u32) -> (Vec<u8>, Vec<u8>) {
    let mut gray = vec![0u8; (sw * sh) as usize];
    for y in 0..sh {
        for x in 0..sw {
            gray[(y * sw + x) as usize] = (40 + (x * 200 / sw.max(1)) / 2 + y * 3) as u8;
        }
    }
    let mut enc = EncoderConfig::grayscale(80)
        .encode_from_bytes(sw, sh, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&gray, Unstoppable).unwrap();
    let secondary = enc.finish().unwrap();

    let mut rgb = vec![0u8; (pw * ph * 3) as usize];
    for y in 0..ph {
        for x in 0..pw {
            let i = ((y * pw + x) * 3) as usize;
            rgb[i] = (x * 255 / pw.max(1)) as u8;
            rgb[i + 1] = (y * 255 / ph.max(1)) as u8;
            rgb[i + 2] = ((x + y) % 64 * 4) as u8;
        }
    }
    let segments = EncoderSegments::new()
        .set_xmp(&generate_primary_xmp(secondary.len()))
        .add_gainmap(secondary.clone());
    let mut enc = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
        .with_segments(segments)
        .encode_from_bytes(pw, ph, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    (enc.finish().unwrap(), secondary)
}

/// Offset of the MPF TIFF header inside `jpeg` (the position the MP entry
/// offsets are relative to), and the byte length of the primary image
/// (SOI through the primary's EOI).
fn mpf_layout(jpeg: &[u8]) -> (usize, usize) {
    let mut tiff_pos = None;
    for_each_segment(jpeg, |marker, off, payload| {
        if marker == 0xE2 && payload.starts_with(b"MPF\0") {
            tiff_pos = Some(off + 4 + 4);
        }
    });
    // The primary's EOI is the first FF D9 after SOS scan data. Scan data
    // cannot contain FF D9 (byte stuffing), so the first occurrence is it.
    let mut sos = 0;
    for_each_segment(jpeg, |marker, off, _| {
        if marker == 0xDA {
            sos = off;
        }
    });
    let eoi = (sos..jpeg.len() - 1)
        .find(|&i| jpeg[i] == 0xFF && jpeg[i + 1] == 0xD9)
        .expect("primary EOI");
    (tiff_pos.expect("MPF APP2 present"), eoi + 2)
}

#[test]
fn explicit_mpf_trim_rejects_different_retained_regions() {
    use zenjpeg::lossless::{RestructureConfig, restructure};
    for (pw, ph, sw, sh) in [(64, 48, 16, 12), (64, 60, 32, 24), (64, 60, 16, 15)] {
        let (jpeg, _) = synthetic_mpf_jpeg(pw, ph, sw, sh);
        let config = TransformConfig {
            transform: LosslessTransform::Rotate90,
            edge_handling: EdgeHandling::TrimPartialBlocks,
        };
        assert!(
            transform(&jpeg, &config, Unstoppable).is_err(),
            "{pw}x{ph} + {sw}x{sh}"
        );
        assert!(
            restructure(
                &jpeg,
                &RestructureConfig {
                    transform: Some(config),
                    ..Default::default()
                },
                Unstoppable
            )
            .is_err()
        );
    }
}

#[test]
fn explicit_mpf_trim_accepts_the_same_retained_region() {
    use zenjpeg::lossless::{RestructureConfig, restructure};
    // Both images retain exactly the first 4/5 of their source height.
    let (jpeg, _) = synthetic_mpf_jpeg(64, 60, 32, 30);
    let config = TransformConfig {
        transform: LosslessTransform::Rotate90,
        edge_handling: EdgeHandling::TrimPartialBlocks,
    };
    for out in [
        transform(&jpeg, &config, Unstoppable).unwrap(),
        restructure(
            &jpeg,
            &RestructureConfig {
                transform: Some(config),
                ..Default::default()
            },
            Unstoppable,
        )
        .unwrap(),
    ] {
        let dec = DecodeConfig::new();
        let (primary, extras) = dec
            .decode_coefficients_with_extras(&out, Unstoppable)
            .unwrap();
        assert_eq!((primary.width, primary.height), (48, 64));
        let extras = extras.unwrap();
        let secondary = dec.decode(extras.gainmap().unwrap(), Unstoppable).unwrap();
        assert_eq!((secondary.width(), secondary.height()), (24, 32));
    }
}

#[test]
fn default_mpf_transform_rejects_partial_secondary() {
    let (jpeg, _) = synthetic_mpf_jpeg(64, 48, 16, 12);
    assert!(
        transform(
            &jpeg,
            &TransformConfig {
                transform: LosslessTransform::Rotate90,
                ..Default::default()
            },
            Unstoppable
        )
        .is_err()
    );
}

#[test]
fn exif_orientation_checks_secondary_edges_and_explicit_crop_alignment() {
    use zenjpeg::lossless::apply_exif_orientation_with_edge_handling;
    for (ph, sw, sh, trim_ok) in [(48, 16, 12, false), (60, 32, 30, true)] {
        let (_, map) = synthetic_mpf_jpeg(64, ph, sw, sh);
        let pixels: Vec<u8> = (0..64 * ph * 3)
            .map(|i| ((i * 73 + i / 17) % 256) as u8)
            .collect();
        let mut encoder = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
            .with_segments(
                EncoderSegments::new()
                    .set_xmp(&generate_primary_xmp(map.len()))
                    .add_gainmap(map),
            )
            .request()
            .exif(Exif::build().orientation(Orientation::Rotate90))
            .encode_from_bytes(64, ph, PixelLayout::Rgb8Srgb)
            .unwrap();
        encoder.push_packed(&pixels, Unstoppable).unwrap();
        let input = encoder.finish().unwrap();
        assert!(apply_exif_orientation(&input, Unstoppable).is_err());
        let trimmed = apply_exif_orientation_with_edge_handling(
            &input,
            EdgeHandling::TrimPartialBlocks,
            Unstoppable,
        );
        assert_eq!(trimmed.is_ok(), trim_ok);
        if let Ok(trimmed) = trimmed {
            let result = DecodeConfig::new().decode(&trimmed, Unstoppable).unwrap();
            assert_eq!((result.width(), result.height()), (48, 64));
            assert_eq!(
                zenjpeg::lossless::parse_exif_orientation(result.extras().unwrap().exif().unwrap()),
                Some(1)
            );
        }
    }
}

/// Every lossless transform (including `None`) must carry the MPF secondary
/// images through, transformed identically, with the MPF index and the
/// GContainer `Item:Length` rebuilt for the new byte layout. Emitting the
/// source's MPF/XMP with no secondary behind them describes a gain map that
/// is not in the file (readers then report a broken gain map).
#[test]
fn lossless_transform_carries_and_transforms_mpf_secondary_images() {
    let (jpeg, secondary) = synthetic_mpf_jpeg(64, 48, 32, 24);
    let dec = DecodeConfig::new().preserve(PreserveConfig::all());
    let (_, extras) = dec
        .decode_coefficients_with_extras(&jpeg, Unstoppable)
        .unwrap();
    let extras = extras.expect("extras");
    assert_eq!(
        extras.gainmap(),
        Some(secondary.as_slice()),
        "fixture: secondary preserved"
    );
    let (sw, sh, sec_ref) = decode_test(&secondary, &DecodeConfig::new());

    for t in LosslessTransform::ALL {
        let out = transform(
            &jpeg,
            &TransformConfig {
                transform: t,
                edge_handling: EdgeHandling::RejectPartialBlocks,
            },
            Unstoppable,
        )
        .unwrap();

        let (_, out_extras) = dec
            .decode_coefficients_with_extras(&out, Unstoppable)
            .unwrap();
        let out_extras = out_extras.expect("extras");
        let got = out_extras
            .gainmap()
            .unwrap_or_else(|| panic!("{t:?}: output lost its MPF secondary image"));

        // MPF index describes the new layout exactly.
        let mpf = out_extras.mpf().expect("MPF directory");
        let (tiff_pos, primary_len) = mpf_layout(&out);
        assert_eq!(mpf.images.len(), 2, "{t:?}: MPF image count");
        assert_eq!(
            mpf.images[0].size as usize, primary_len,
            "{t:?}: MPF primary size"
        );
        assert_eq!(
            mpf.images[1].offset as usize + tiff_pos,
            primary_len,
            "{t:?}: MPF secondary offset"
        );
        assert_eq!(
            mpf.images[1].size as usize,
            got.len(),
            "{t:?}: MPF secondary size"
        );
        assert_eq!(
            &out[primary_len..primary_len + got.len()],
            got,
            "{t:?}: secondary bytes"
        );
        assert_eq!(out.len(), primary_len + got.len(), "{t:?}: trailing bytes");

        // GContainer Item:Length matches the secondary actually in the file.
        let (_, items) = parse_xmp_full(out_extras.xmp().expect("XMP kept"));
        let lengths: Vec<usize> = items.iter().filter_map(|i| i.length).collect();
        assert_eq!(lengths, vec![got.len()], "{t:?}: Item:Length");

        // The secondary got the same geometric transform as the primary.
        // (The default decode expands the grayscale map to RGB; compare in
        // whatever channel count the decoder chose.)
        let (gw, gh, got_px) = decode_test(got, &DecodeConfig::new());
        let channels = sec_ref.len() / (sw as usize * sh as usize);
        let (ew, eh, expected) = pixel_transform(&sec_ref, sw as usize, sh as usize, channels, t);
        assert_eq!(
            (gw as usize, gh as usize),
            (ew, eh),
            "{t:?}: secondary dimensions"
        );
        let diff = max_abs_diff(&got_px, &expected);
        assert!(diff <= 2, "{t:?}: secondary max pixel diff {diff}");
    }
}

/// `apply_exif_orientation` is the same pipeline; the gain map must rotate with
/// the primary and the index must be rebuilt.
#[test]
fn apply_exif_orientation_rotates_mpf_secondary_images() {
    let (_, secondary) = synthetic_mpf_jpeg(16, 16, 32, 24);
    let (pw, ph) = (64u32, 48u32);
    let rgb: Vec<u8> = (0..pw * ph * 3).map(|i| (i * 7 % 251) as u8).collect();
    let segments = EncoderSegments::new()
        .set_xmp(&generate_primary_xmp(secondary.len()))
        .add_gainmap(secondary.clone());
    let mut enc = EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
        .with_segments(segments)
        .request()
        .exif(Exif::build().orientation(Orientation::Rotate90))
        .encode_from_bytes(pw, ph, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&rgb, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();

    let out = apply_exif_orientation(&jpeg, Unstoppable).unwrap();
    let dec = DecodeConfig::new().preserve(PreserveConfig::all());
    let (coeffs, extras) = dec
        .decode_coefficients_with_extras(&out, Unstoppable)
        .unwrap();
    assert_eq!((coeffs.width, coeffs.height), (ph, pw), "primary rotated");
    let extras = extras.expect("extras");
    let got = extras.gainmap().expect("gain map carried through");
    let (gw, gh, _) = decode_test(got, &DecodeConfig::new());
    assert_eq!((gw, gh), (24, 32), "gain map rotated with the primary");
    let mpf = extras.mpf().expect("MPF directory");
    let (tiff_pos, primary_len) = mpf_layout(&out);
    assert_eq!(mpf.images[1].offset as usize + tiff_pos, primary_len);
    assert_eq!(mpf.images[1].size as usize, got.len());
}

/// End to end on a real Ultra HDR file made by zenjpeg's own encoder: after a
/// lossless rotation the gain-map metadata still parses with the encoded
/// range, and the HDR reconstruction is the rotated reconstruction of the
/// source. Before the fix `UltraHdrReader` saw a gain map with stale offsets
/// and reported a max boost of 0.
#[cfg(feature = "ultrahdr")]
#[test]
fn ultrahdr_hdr_reconstruction_survives_lossless_rotation() {
    use ultrahdr_core::gainmap::HdrOutputFormat;
    use ultrahdr_core::pixel_buffer_from_vec;
    use zenjpeg::decoder::Decoder;
    use zenjpeg::ultrahdr::{
        GainMapConfig, UhdrColorGamut, UhdrColorTransfer, UhdrPixelFormat, UltraHdrExtras,
        decode_ultrahdr_hdr, encode_ultrahdr_with_curve,
    };
    use zentone::Bt2446C;

    const PEAK: f32 = 4.0;
    let (width, height) = (64u32, 32u32);
    let mut data = Vec::with_capacity((width * height * 16) as usize);
    for _y in 0..height {
        for x in 0..width {
            let v = if x < width / 2 { 1.0 } else { PEAK };
            for c in [v, v, v, 1.0f32] {
                data.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    let hdr = pixel_buffer_from_vec(
        data,
        width,
        height,
        UhdrPixelFormat::RgbaF32,
        UhdrColorGamut::Bt709,
        UhdrColorTransfer::Linear,
    )
    .expect("HDR buffer");
    let config = GainMapConfig::default();
    let jpeg = encode_ultrahdr_with_curve(
        &hdr,
        &Bt2446C::new(203.0, 100.0),
        &config,
        &EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        75.0,
        Unstoppable,
    )
    .expect("ultrahdr encode");

    let rotated = transform(
        &jpeg,
        &TransformConfig {
            transform: LosslessTransform::Rotate90,
            edge_handling: EdgeHandling::RejectPartialBlocks,
        },
        Unstoppable,
    )
    .expect("lossless rotate");

    // Metadata: same declared range after the rotation.
    let meta_of = |bytes: &[u8]| {
        let decoded = Decoder::new().decode(bytes, Unstoppable).unwrap();
        let extras = decoded.extras().expect("extras");
        let (m, _) = extras
            .ultrahdr_metadata()
            .expect("ultrahdr metadata present")
            .expect("ultrahdr metadata parses");
        m.channels
            .iter()
            .map(|c| (c.min, c.max))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        meta_of(&rotated),
        meta_of(&jpeg),
        "gain-map range after rotation"
    );

    // Pixels. The SDR base and the gain map are both exact transposes (the
    // same DCT-domain path as every other transform), so check them first.
    let dec = DecodeConfig::new().preserve(PreserveConfig::all());
    let (_, e1) = dec
        .decode_coefficients_with_extras(&jpeg, Unstoppable)
        .unwrap();
    let (_, e2) = dec
        .decode_coefficients_with_extras(&rotated, Unstoppable)
        .unwrap();
    let (g1, g2) = (
        e1.as_ref().unwrap().gainmap().expect("source gain map"),
        e2.as_ref().unwrap().gainmap().expect("rotated gain map"),
    );
    let (w1, h1, p1) = decode_test(g1, &DecodeConfig::new());
    let (w2, h2, p2) = decode_test(g2, &DecodeConfig::new());
    let ch = p1.len() / (w1 * h1) as usize;
    let (ew, eh, exp) = pixel_transform(
        &p1,
        w1 as usize,
        h1 as usize,
        ch,
        LosslessTransform::Rotate90,
    );
    assert_eq!((w2 as usize, h2 as usize), (ew, eh), "gain map dimensions");
    assert!(
        max_abs_diff(&p2, &exp) <= 2,
        "gain map pixels not the rotated source"
    );
    let (sw, sh, sp) = decode_test(&jpeg, &DecodeConfig::new());
    let (rw, rh, rp) = decode_test(&rotated, &DecodeConfig::new());
    let (ew, eh, exp) = pixel_transform(
        &sp,
        sw as usize,
        sh as usize,
        3,
        LosslessTransform::Rotate90,
    );
    assert_eq!((rw as usize, rh as usize), (ew, eh), "SDR base dimensions");
    assert!(
        max_abs_diff(&rp, &exp) <= 2,
        "SDR base pixels not the rotated source"
    );

    // Then the HDR reconstruction, away from the gain-map transition: the
    // reader's Shepard's-IDW upsample anchors each cell at its top-left sample
    // (mirroring libultrahdr), which is not rotation-symmetric, so the ramp
    // around the 1×/4× edge lands a sample or two differently after a
    // rotation even though both inputs are exact transposes. Everywhere else
    // the reconstruction must match.
    let boost = config.alternate_hdr_headroom;
    let src = decode_ultrahdr_hdr(&jpeg, boost, HdrOutputFormat::LinearFloat).expect("src hdr");
    let rot = decode_ultrahdr_hdr(&rotated, boost, HdrOutputFormat::LinearFloat).expect("rot hdr");
    assert_eq!(
        (rot.width(), rot.height()),
        (height, width),
        "rotated dimensions"
    );
    // RGBA f32 rows (16 bytes per pixel).
    let px = |buf: &zenjpeg::ultrahdr::UhdrRawImage, x: u32, y: u32| -> [f32; 3] {
        let slice = buf.as_slice();
        let row = slice.row(y);
        let p = &row[(x as usize) * 16..(x as usize) * 16 + 12];
        [
            f32::from_le_bytes([p[0], p[1], p[2], p[3]]),
            f32::from_le_bytes([p[4], p[5], p[6], p[7]]),
            f32::from_le_bytes([p[8], p[9], p[10], p[11]]),
        ]
    };
    let scale = width / w1; // gain-map cell size in primary pixels
    let edge = width / 2;
    let mut max_err = 0f32;
    let mut peak = 0f32;
    for sy in 0..height {
        for sx in 0..width {
            let (dx, dy) = (height - 1 - sy, sx); // Rotate90
            let b = px(&rot, dx, dy);
            peak = peak.max(b[0]);
            if sx + 2 * scale >= edge && sx < edge + 2 * scale {
                continue; // transition band
            }
            let a = px(&src, sx, sy);
            for c in 0..3 {
                max_err = max_err.max((a[c] - b[c]).abs());
            }
        }
    }
    assert!(
        max_err <= 0.05,
        "HDR reconstruction after rotation differs from rotated reconstruction: max err {max_err}"
    );
    // And the bright half really is bright (the gain map was applied at all).
    assert!(
        peak > PEAK * 0.9,
        "peak after rotation {peak}, expected ≈{PEAK}"
    );
}
