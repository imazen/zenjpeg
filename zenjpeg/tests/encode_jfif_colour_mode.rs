//! Caller segments must not contradict the colour model the encoder writes.
//!
//! A JFIF APP0 means YCbCr (or gray) to every decoder: libjpeg-turbo's
//! `default_decompress_parms` and jpegli's `ProcessSOF` decode a three-component
//! JFIF frame as YCbCr whatever its APP14 transform or component IDs say. So an
//! RGB or XYB encode must not carry one, wherever the caller's segments came
//! from: preserved `DecodedExtras`, `set_printer_dpi`, or a raw `add_segment`.
//! YCbCr output keeps it.
//!
//! An Adobe APP14 with transform 0 means RGB to the same decoders when no JFIF
//! precedes it. libjpeg-turbo writes Adobe only for RGB/CMYK/YCCK, and zenjpeg
//! writes its own transform-0 APP14 for RGB and XYB, so a caller's Adobe
//! segment (from an Adobe-RGB source or zenjpeg's own RGB output, preserved)
//! is dropped for every output.
//!
//! Both segment-injection sites are covered: `BytesEncoder::finish_into`
//! (`encode_bytes`, and the typed `encode::<P>`, whose `RgbEncoder<P>` wraps a
//! `BytesEncoder`) and `YCbCrPlanarEncoder::finish_into`.

use enough::Unstoppable;
use zenjpeg::decoder::{ColorSpace, DecodeResult, Decoder, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};

const W: u32 = 96;
const H: u32 = 80;

/// High-frequency noise with flat patches (never a gradient).
fn image() -> Vec<u8> {
    let mut px = vec![0u8; (W * H * 3) as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77);
            h ^= h >> 15;
            h = h.wrapping_mul(0x2C1B_3C6D);
            h ^= h >> 13;
            let i = (y * W as usize + x) * 3;
            if (x / 16 + y / 16) % 3 == 0 {
                px[i..i + 3].copy_from_slice(&[200, 40, 90]);
            } else {
                px[i..i + 3].copy_from_slice(&[h as u8, (h >> 8) as u8, (h >> 16) as u8]);
            }
        }
    }
    px
}

/// JFIF APP0 payload: "JFIF\0", version 1.02, dots per inch, 300x300, no thumbnail.
fn jfif_payload() -> Vec<u8> {
    vec![
        b'J', b'F', b'I', b'F', 0, 1, 2, 1, 0x01, 0x2C, 0x01, 0x2C, 0, 0,
    ]
}

/// APP0 segments before the first SOS whose payload starts with "JFIF" (what
/// libjpeg-turbo, jpegli and zenjpeg treat as JFIF).
fn jfif_count(jpeg: &[u8]) -> usize {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "not a JPEG");
    let (mut p, mut n) = (2, 0);
    while p + 4 <= jpeg.len() && jpeg[p] == 0xFF && jpeg[p + 1] != 0xDA {
        let len = u16::from_be_bytes([jpeg[p + 2], jpeg[p + 3]]) as usize;
        if jpeg[p + 1] == 0xE0 && jpeg[p + 4..p + 2 + len].starts_with(b"JFIF") {
            n += 1;
        }
        p += 2 + len;
    }
    n
}

/// Adobe APP14 payload (TN 5116): "Adobe", version 100, flags 0/0, `transform`.
fn adobe_payload(transform: u8) -> Vec<u8> {
    vec![b'A', b'd', b'o', b'b', b'e', 0, 100, 0, 0, 0, 0, transform]
}

/// Transforms of the Adobe APP14 segments before the first SOS, in order.
fn adobe_transforms(jpeg: &[u8]) -> Vec<u8> {
    let (mut p, mut out) = (2, Vec::new());
    while p + 4 <= jpeg.len() && jpeg[p] == 0xFF && jpeg[p + 1] != 0xDA {
        let len = u16::from_be_bytes([jpeg[p + 2], jpeg[p + 3]]) as usize;
        let payload = &jpeg[p + 4..p + 2 + len];
        if jpeg[p + 1] == 0xEE && payload.starts_with(b"Adobe") && payload.len() >= 12 {
            out.push(payload[11]);
        }
        p += 2 + len;
    }
    out
}

fn gray_image() -> Vec<u8> {
    image().as_chunks::<3>().0.iter().map(|p| p[1]).collect()
}

fn encode(config: EncoderConfig) -> Vec<u8> {
    config
        .encode_bytes(&image(), W, H, PixelLayout::Rgb8Srgb)
        .expect("encode")
}

fn decode_rgb(jpeg: &[u8]) -> Vec<u8> {
    Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("u8 pixels")
}

/// A decoded JFIF YCbCr photo with its metadata preserved (the default
/// `PreserveConfig` keeps JFIF); `extras().to_encoder_segments()` re-emits it.
fn decoded_jfif_photo() -> DecodeResult {
    let source = encode(
        EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter).add_segment(0xE0, jfif_payload()),
    );
    assert_eq!(jfif_count(&source), 1, "source must carry JFIF");
    Decoder::new()
        .preserve_all_metadata()
        .decode(&source, Unstoppable)
        .expect("decode source")
}

#[test]
fn rgb_and_xyb_output_carries_no_jfif() {
    let photo = decoded_jfif_photo();
    let extras = photo.extras().expect("extras");
    let mut failures = Vec::new();
    for progressive in [false, true] {
        let modes: [(&str, fn() -> EncoderConfig, ColorSpace); 3] = [
            ("rgb", || EncoderConfig::rgb(90), ColorSpace::Rgb),
            (
                "xyb full",
                || EncoderConfig::xyb(90, XybSubsampling::Full),
                ColorSpace::Xyb,
            ),
            (
                "xyb bquarter",
                || EncoderConfig::xyb(90, XybSubsampling::BQuarter),
                ColorSpace::Xyb,
            ),
        ];
        for (mode, config, want) in modes {
            let plain = encode(config().progressive(progressive));
            let reference = decode_rgb(&plain);
            let sources: [(&str, EncoderConfig); 3] = [
                (
                    "preserved extras",
                    config()
                        .progressive(progressive)
                        .with_segments(extras.to_encoder_segments()),
                ),
                (
                    "set_printer_dpi",
                    config()
                        .progressive(progressive)
                        .with_segments(extras.to_encoder_segments().set_printer_dpi(600)),
                ),
                (
                    "add_segment",
                    config()
                        .progressive(progressive)
                        .add_segment(0xE0, jfif_payload()),
                ),
            ];
            for (source, cfg) in sources {
                let label = format!("{mode} progressive={progressive} {source}");
                let out = encode(cfg);
                let n = jfif_count(&out);
                if n != 0 {
                    failures.push(format!("{label}: output carries {n} JFIF APP0"));
                }
                let info = Decoder::new().read_info(&out).expect("read_info");
                if info.color_space != want {
                    failures.push(format!(
                        "{label}: read_info reports {:?}, want {want:?}",
                        info.color_space
                    ));
                }
                if decode_rgb(&out) != reference {
                    failures.push(format!(
                        "{label}: pixels differ from the same encode without segments"
                    ));
                }
            }
        }
    }
    // Control: YCbCr output keeps the caller's JFIF.
    let ycc = encode(
        EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter)
            .with_segments(extras.to_encoder_segments()),
    );
    if jfif_count(&ycc) != 1 {
        failures.push(format!(
            "ycbcr: JFIF count {} in YCbCr output, want 1",
            jfif_count(&ycc)
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A decoded Adobe-RGB file (zenjpeg's own RGB output, APP14 transform 0) with
/// its metadata preserved: `extras().to_encoder_segments()` re-emits the APP14.
fn decoded_adobe_rgb_file() -> DecodeResult {
    let source = encode(EncoderConfig::rgb(90));
    assert_eq!(adobe_transforms(&source), [0], "source must carry APP14 t0");
    Decoder::new()
        .preserve_all_metadata()
        .decode(&source, Unstoppable)
        .expect("decode source")
}

#[test]
fn caller_adobe_app14_is_dropped_for_every_output() {
    let rgb_file = decoded_adobe_rgb_file();
    let extras = rgb_file.extras().expect("extras");
    let mut failures = Vec::new();
    for progressive in [false, true] {
        // (mode, config, expected colour space, encoder's own APP14 transforms)
        let modes: [(&str, fn() -> EncoderConfig, ColorSpace, &[u8]); 4] = [
            (
                "ycbcr 4:2:0",
                || EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter),
                ColorSpace::YCbCr,
                &[],
            ),
            (
                "grayscale",
                || EncoderConfig::grayscale(90),
                ColorSpace::Grayscale,
                &[],
            ),
            ("rgb", || EncoderConfig::rgb(90), ColorSpace::Rgb, &[0]),
            (
                "xyb full",
                || EncoderConfig::xyb(90, XybSubsampling::Full),
                ColorSpace::Xyb,
                &[0],
            ),
        ];
        for (mode, config, want, own) in modes {
            let gray = want == ColorSpace::Grayscale;
            let enc = |cfg: EncoderConfig| -> Vec<u8> {
                let cfg = cfg.progressive(progressive);
                if gray {
                    cfg.encode_bytes(&gray_image(), W, H, PixelLayout::Gray8Srgb)
                        .expect("encode")
                } else {
                    cfg.encode_bytes(&image(), W, H, PixelLayout::Rgb8Srgb)
                        .expect("encode")
                }
            };
            let plain = enc(config());
            assert_eq!(adobe_transforms(&plain), own, "{mode}: encoder's own APP14");
            let reference = decode_rgb(&plain);
            let sources: [(&str, EncoderConfig); 3] = [
                (
                    "preserved Adobe-RGB extras",
                    config().with_segments(extras.to_encoder_segments()),
                ),
                (
                    "add_segment APP14 t0",
                    config().add_segment(0xEE, adobe_payload(0)),
                ),
                (
                    "add_segment APP14 t1",
                    config().add_segment(0xEE, adobe_payload(1)),
                ),
            ];
            for (source, cfg) in sources {
                let label = format!("{mode} progressive={progressive} {source}");
                let out = enc(cfg);
                let got = adobe_transforms(&out);
                if got != own {
                    failures.push(format!("{label}: APP14 transforms {got:?}, want {own:?}"));
                }
                let info = Decoder::new().read_info(&out).expect("read_info");
                if info.color_space != want {
                    failures.push(format!(
                        "{label}: read_info reports {:?}, want {want:?}",
                        info.color_space
                    ));
                }
                if decode_rgb(&out) != reference {
                    failures.push(format!(
                        "{label}: pixels differ from the same encode without segments"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The typed encoder (`EncoderConfig::encode::<P>`, `RgbEncoder<P>`) applies
/// both drops (it finishes through `BytesEncoder`).
#[test]
fn typed_encoder_drops_caller_colour_signals() {
    let pixels: Vec<rgb::RGB<u8>> = image()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| rgb::RGB::new(p[0], p[1], p[2]))
        .collect();
    let mut failures = Vec::new();
    let cases: [(&str, fn() -> EncoderConfig, ColorSpace, usize, &[u8]); 3] = [
        ("rgb", || EncoderConfig::rgb(90), ColorSpace::Rgb, 0, &[0]),
        (
            "xyb bquarter",
            || EncoderConfig::xyb(90, XybSubsampling::BQuarter),
            ColorSpace::Xyb,
            0,
            &[0],
        ),
        (
            "ycbcr 4:2:0",
            || EncoderConfig::ycbcr(90, ChromaSubsampling::Quarter),
            ColorSpace::YCbCr,
            1,
            &[],
        ),
    ];
    for (mode, config, want, jfif_kept, own) in cases {
        let plain = config().encode(&pixels, W, H).expect("typed encode");
        let reference = decode_rgb(&plain);
        let out = config()
            .add_segment(0xE0, jfif_payload())
            .add_segment(0xEE, adobe_payload(0))
            .encode(&pixels, W, H)
            .expect("typed encode with segments");
        let n = jfif_count(&out);
        if n != jfif_kept {
            failures.push(format!("{mode}: {n} JFIF APP0, want {jfif_kept}"));
        }
        let got = adobe_transforms(&out);
        if got != own {
            failures.push(format!("{mode}: APP14 transforms {got:?}, want {own:?}"));
        }
        let info = Decoder::new().read_info(&out).expect("read_info");
        if info.color_space != want {
            failures.push(format!(
                "{mode}: read_info reports {:?}, want {want:?}",
                info.color_space
            ));
        }
        if decode_rgb(&out) != reference {
            failures.push(format!(
                "{mode}: pixels differ from the typed encode without segments"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `YCbCrPlanarEncoder` injects segments at its own site. It only writes YCbCr
/// (it rejects RGB and XYB), so the JFIF stays and a caller's Adobe APP14 goes.
#[test]
fn ycbcr_planar_encoder_drops_caller_adobe() {
    use zenjpeg::encoder::YCbCrPlanes;
    let (w, h) = (W as usize, H as usize);
    let (mut y, mut cb, mut cr) = (vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]);
    for (i, p) in image().as_chunks::<3>().0.iter().enumerate() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        y[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        cb[i] = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
        cr[i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    }
    let planes = YCbCrPlanes {
        y: &y,
        y_stride: w,
        cb: &cb,
        cb_stride: w,
        cr: &cr,
        cr_stride: w,
    };
    let planar = |config: EncoderConfig| -> Vec<u8> {
        let mut enc = config
            .encode_from_ycbcr_planar(W, H)
            .expect("planar encoder");
        enc.push(&planes, h, Unstoppable).expect("push");
        enc.finish().expect("finish")
    };
    let config = || EncoderConfig::ycbcr(90, ChromaSubsampling::None);
    let plain = planar(config());
    let out = planar(
        config()
            .add_segment(0xE0, jfif_payload())
            .add_segment(0xEE, adobe_payload(0)),
    );
    let mut failures = Vec::new();
    if jfif_count(&out) != 1 {
        failures.push(format!("{} JFIF APP0, want 1", jfif_count(&out)));
    }
    if !adobe_transforms(&out).is_empty() {
        failures.push(format!(
            "APP14 transforms {:?}, want none",
            adobe_transforms(&out)
        ));
    }
    let info = Decoder::new().read_info(&out).expect("read_info");
    if info.color_space != ColorSpace::YCbCr {
        failures.push(format!(
            "read_info reports {:?}, want YCbCr",
            info.color_space
        ));
    }
    if decode_rgb(&out) != decode_rgb(&plain) {
        failures.push("pixels differ from the planar encode without segments".into());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
