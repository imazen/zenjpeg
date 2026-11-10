//! Encoder edge regressions. All fixtures are deterministic source, no images on disk.
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{
    ChromaSubsampling as Sub, DownsamplingMethod as Method, EncoderConfig, PixelLayout as Layout,
    Unstoppable, YCbCrPlanes,
};

const SUBS: [Sub; 4] = [
    Sub::None,
    Sub::HalfHorizontal,
    Sub::Quarter,
    Sub::HalfVertical,
];
const QUALITIES: [u8; 10] = [5, 10, 20, 30, 40, 60, 70, 80, 90, 100];

fn pixels(w: usize, h: usize, layout: Layout, pattern: usize) -> Vec<u8> {
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let mut rgb = match pattern {
                0 => [81; 3],
                1 => [47, 139, 203],
                2 => {
                    if (x + y) % 2 == 0 {
                        [24, 180, 91]
                    } else {
                        [230, 57, 164]
                    }
                }
                _ => [
                    (x * 43 + y * 71) as u8,
                    (x * 79 + y * 19 + 37) as u8,
                    if x == w - 1 || y == h - 1 { 251 } else { 32 },
                ],
            };
            if matches!(
                layout,
                Layout::Bgr8Srgb | Layout::Bgrx8Srgb | Layout::Bgra8Srgb
            ) {
                rgb.swap(0, 2);
            }
            for c in 0..layout.channels() {
                let value = if c < 3 { rgb[c] } else { 17 };
                match layout {
                    Layout::Rgb16Linear
                    | Layout::Rgbx16Linear
                    | Layout::Rgba16Linear
                    | Layout::Gray16Linear => {
                        v.extend_from_slice(&(value as u16 * 257).to_ne_bytes())
                    }
                    Layout::RgbF32Linear
                    | Layout::RgbxF32Linear
                    | Layout::RgbaF32Linear
                    | Layout::GrayF32Linear => {
                        v.extend_from_slice(&(value as f32 / 255.).to_ne_bytes())
                    }
                    _ => v.push(value),
                }
            }
        }
    }
    v
}

fn streamed(
    cfg: &EncoderConfig,
    src: &[u8],
    w: usize,
    h: usize,
    layout: Layout,
    extra: usize,
    parts: &[usize],
) -> Vec<u8> {
    let row = w * layout.bytes_per_pixel();
    let stride = row + extra;
    let mut padded = vec![0xE7; stride * h];
    for y in 0..h {
        padded[y * stride..y * stride + row].copy_from_slice(&src[y * row..(y + 1) * row]);
    }
    let mut enc = cfg.encode_from_bytes(w as u32, h as u32, layout).unwrap();
    let mut y = 0;
    let mut i = 0;
    while y < h {
        let n = parts[i % parts.len()].min(h - y);
        enc.push(&padded[y * stride..], n, stride, Unstoppable)
            .unwrap();
        y += n;
        i += 1;
    }
    enc.finish().unwrap()
}

#[test]
fn formats_strides_partitions_and_entropy_consumers() {
    let layouts = [
        Layout::Rgb8Srgb,
        Layout::Rgba8Srgb,
        Layout::Rgbx8Srgb,
        Layout::Bgr8Srgb,
        Layout::Bgra8Srgb,
        Layout::Bgrx8Srgb,
        Layout::Rgb16Linear,
        Layout::Rgba16Linear,
        Layout::Rgbx16Linear,
        Layout::RgbF32Linear,
        Layout::RgbaF32Linear,
        Layout::RgbxF32Linear,
        Layout::Gray8Srgb,
        Layout::Gray16Linear,
        Layout::GrayF32Linear,
    ];
    let mut cells = 0;
    for layout in layouts {
        for sub in SUBS {
            for (w, h) in [
                (1, 1),
                (7, 9),
                (8, 16),
                (9, 7),
                (15, 17),
                (16, 32),
                (17, 31),
                (31, 33),
                (32, 48),
                (33, 49),
                (129, 65),
            ] {
                for (progressive, optimize) in [(false, false), (false, true), (true, true)] {
                    let cfg = EncoderConfig::ycbcr(40, sub)
                        .progressive(progressive)
                        .optimize_huffman(optimize);
                    let src = pixels(w, h, layout, 3);
                    let expected = cfg.encode_bytes(&src, w as u32, h as u32, layout).unwrap();
                    for (extra, parts) in [
                        (0, &[1][..]),
                        (13, &[7, 1, 19, 2][..]),
                        (5, &[16, 8, 3][..]),
                    ] {
                        assert_eq!(
                            streamed(&cfg, &src, w, h, layout, extra, parts),
                            expected,
                            "{layout:?} {sub:?} {w}x{h} progressive {progressive} optimized {optimize} extra {extra}"
                        );
                        cells += 1;
                    }
                }
            }
        }
    }
    eprintln!("source stride/partition/entropy byte parity: {cells} comparisons");
}

fn plane(w: usize, h: usize, extra: usize, channel: usize, hs: usize, vs: usize) -> Vec<f32> {
    let mut v = vec![1234.; (w + extra) * h];
    for y in 0..h {
        for x in 0..w {
            v[y * (w + extra) + x] = match channel {
                0 => ((x * 43 + y * 71) % 193) as f32 - 90.,
                1 => (((x / hs) * 19 + (y / vs) * 31) % 151) as f32 - 71.,
                _ => (((x / hs) * 37 + (y / vs) * 11) % 173) as f32 - 83.,
            };
        }
    }
    v
}

#[test]
fn planar_full_and_pre_subsampled_exact_bytes() {
    let mut cells = 0;
    for sub in SUBS {
        let hs = sub.h_factor() as usize;
        let vs = sub.v_factor() as usize;
        let sh = vs * 8;
        for w in (1..=33).chain([63, 64, 65, 127, 128, 129]) {
            for h in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49] {
                let cw = w.div_ceil(hs);
                let ch = h.div_ceil(vs);
                for extra in [0, 3] {
                    let y = plane(w, h, extra, 0, 1, 1);
                    let cb = plane(w, h, extra, 1, hs, vs);
                    let cr = plane(w, h, extra, 2, hs, vs);
                    let scb = plane(cw, ch, extra, 1, 1, 1);
                    let scr = plane(cw, ch, extra, 2, 1, 1);
                    let cfg = EncoderConfig::ycbcr(40, sub);
                    let mut full = cfg.encode_from_ycbcr_planar(w as u32, h as u32).unwrap();
                    full.push(
                        &YCbCrPlanes {
                            y: &y,
                            y_stride: w + extra,
                            cb: &cb,
                            cb_stride: w + extra,
                            cr: &cr,
                            cr_stride: w + extra,
                        },
                        h,
                        Unstoppable,
                    )
                    .unwrap();
                    let expected = full.finish().unwrap();
                    for partition in [sh, sh * 2, h] {
                        let mut enc = cfg.encode_from_ycbcr_planar(w as u32, h as u32).unwrap();
                        let mut yy = 0;
                        while yy < h {
                            let rows = partition.min(h - yy);
                            let cy = yy / vs;
                            enc.push_subsampled(
                                &YCbCrPlanes {
                                    y: &y[yy * (w + extra)..],
                                    y_stride: w + extra,
                                    cb: &scb[cy * (cw + extra)..],
                                    cb_stride: cw + extra,
                                    cr: &scr[cy * (cw + extra)..],
                                    cr_stride: cw + extra,
                                },
                                rows,
                                Unstoppable,
                            )
                            .unwrap();
                            yy += rows;
                        }
                        assert_eq!(
                            enc.finish().unwrap(),
                            expected,
                            "{sub:?} {w}x{h} extra {extra} partition {partition}"
                        );
                        cells += 1;
                    }
                    // Full-resolution input buffers arbitrary partial pushes.
                    let mut enc = cfg.encode_from_ycbcr_planar(w as u32, h as u32).unwrap();
                    for yy in 0..h {
                        enc.push(
                            &YCbCrPlanes {
                                y: &y[yy * (w + extra)..],
                                y_stride: w + extra,
                                cb: &cb[yy * (w + extra)..],
                                cb_stride: w + extra,
                                cr: &cr[yy * (w + extra)..],
                                cr_stride: w + extra,
                            },
                            1,
                            Unstoppable,
                        )
                        .unwrap();
                    }
                    assert_eq!(enc.finish().unwrap(), expected);
                    cells += 1;
                }
            }
        }
    }
    eprintln!(
        "planar packed/padded, full/pre-subsampled, partition byte parity: {cells} comparisons"
    );
}

#[test]
fn neutral_quality_and_gamma_matrix() {
    let mut cells = 0;
    for q in QUALITIES {
        for sub in SUBS {
            for method in [Method::Box, Method::GammaAware, Method::GammaAwareIterative] {
                for (w, h) in [
                    (1, 1),
                    (7, 8),
                    (8, 9),
                    (9, 15),
                    (15, 16),
                    (16, 17),
                    (17, 31),
                    (31, 32),
                    (32, 33),
                    (33, 49),
                    (129, 65),
                ] {
                    let src = pixels(w, h, Layout::Rgb8Srgb, 0);
                    let jpeg = EncoderConfig::ycbcr(q, sub)
                        .downsampling_method(method)
                        .encode_bytes(&src, w as u32, h as u32, Layout::Rgb8Srgb)
                        .unwrap();
                    let dec = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
                    assert!(
                        dec.pixels_u8()
                            .unwrap()
                            .chunks_exact(3)
                            .all(|p| p[0] == p[1] && p[1] == p[2]),
                        "q{q} {sub:?} {method:?} {w}x{h}"
                    );
                    cells += 1;
                }
            }
        }
    }
    eprintln!("neutral quality/gamma decode: {cells} encodes");
}

#[test]
fn color_modes_replicated_source_coefficients() {
    use zenjpeg::encoder::XybSubsampling;
    let mut cells = 0;
    for cfg in [
        EncoderConfig::rgb(40),
        EncoderConfig::grayscale(40),
        EncoderConfig::xyb(40, XybSubsampling::Full),
        EncoderConfig::xyb(40, XybSubsampling::BQuarter),
    ] {
        for (w, h) in [
            (1usize, 1usize),
            (7, 9),
            (9, 7),
            (17, 31),
            (31, 33),
            (33, 49),
            (129, 65),
        ] {
            // No AQ/deringing: changing frame dimensions must not change
            // quantization, so explicit source extension is an exact oracle.
            let cfg = cfg.clone().aq_enabled(false).deringing(false);
            let pw = w.div_ceil(16) * 16;
            let ph = h.div_ceil(16) * 16;
            let src = pixels(w, h, Layout::Rgb8Srgb, 3);
            let mut extended = vec![0; pw * ph * 3];
            for y in 0..ph {
                for x in 0..pw {
                    extended[(y * pw + x) * 3..(y * pw + x + 1) * 3].copy_from_slice(
                        &src[(y.min(h - 1) * w + x.min(w - 1)) * 3
                            ..(y.min(h - 1) * w + x.min(w - 1) + 1) * 3],
                    );
                }
            }
            let a = cfg
                .encode_bytes(&src, w as u32, h as u32, Layout::Rgb8Srgb)
                .unwrap();
            let b = cfg
                .encode_bytes(&extended, pw as u32, ph as u32, Layout::Rgb8Srgb)
                .unwrap();
            let a = Decoder::new().decode_coefficients(&a, Unstoppable).unwrap();
            let b = Decoder::new().decode_coefficients(&b, Unstoppable).unwrap();
            for (ca, cb) in a.components.iter().zip(&b.components) {
                for by in 0..ca.blocks_high {
                    for bx in 0..ca.blocks_wide {
                        assert_eq!(
                            ca.block_at(bx, by),
                            cb.block_at(bx, by),
                            "{w}x{h} component {} block {bx},{by}",
                            ca.id
                        );
                        cells += 1;
                    }
                }
            }
        }
    }
    eprintln!("RGB/gray/XYB explicit edge-extension coefficients: {cells} blocks");
}

#[cfg(feature = "parallel")]
#[test]
fn fused_parallel_neutral_segments() {
    let mut cells = 0;
    for sub in SUBS {
        for restart in [2, 4, 8] {
            for w in [
                1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
            ] {
                let sh = sub.v_factor() as usize * 8;
                for h in [
                    restart * sh + 1,
                    restart * sh + 7,
                    restart * sh + sh,
                    3 * restart * sh + 1,
                ] {
                    let src = pixels(w, h, Layout::Rgb8Srgb, 0);
                    let jpeg = EncoderConfig::ycbcr(40, sub)
                        .progressive(false)
                        .restart_mcu_rows(restart as u16)
                        .encode_bytes_parallel(&src, w as u32, h as u32)
                        .unwrap();
                    let dec = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
                    assert!(
                        dec.pixels_u8()
                            .unwrap()
                            .chunks_exact(3)
                            .all(|p| p[0] == p[1] && p[1] == p[2]),
                        "{sub:?} {w}x{h} restart {restart}"
                    );
                    cells += 1;
                }
            }
        }
    }
    eprintln!("fused parallel neutral segment/edge decode: {cells} encodes");
}

#[test]
#[ignore = "requires explicitly selected pinned libjpeg-turbo; run just strip-padding-oracle"]
fn independent_turbo_decodes_edges() {
    use std::{
        io::Write,
        path::PathBuf,
        process::{Command, Stdio},
    };
    let bin = PathBuf::from(
        std::env::var_os("ZENJPEG_TURBO_BIN")
            .expect("set ZENJPEG_TURBO_BIN to pinned libjpeg-turbo tools directory"),
    );
    let mut cells = 0;
    for sub in SUBS {
        for q in [5, 20, 40, 60, 80, 100] {
            for method in [Method::Box, Method::GammaAware, Method::GammaAwareIterative] {
                for (w, h) in [(1, 1), (7, 9), (9, 7), (17, 33), (33, 49), (129, 65)] {
                    for pattern in 0..4 {
                        let src = pixels(w, h, Layout::Rgb8Srgb, pattern);
                        let cfg = EncoderConfig::ycbcr(q, sub).downsampling_method(method);
                        let jpeg = cfg
                            .encode_bytes(&src, w as u32, h as u32, Layout::Rgb8Srgb)
                            .unwrap();
                        let mut child = Command::new(bin.join("djpeg-static"))
                            .args(["-nosmooth"])
                            .stdin(Stdio::piped())
                            .stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .spawn()
                            .expect("launch pinned djpeg");
                        child.stdin.take().unwrap().write_all(&jpeg).unwrap();
                        let out = child.wait_with_output().unwrap();
                        assert!(
                            out.status.success(),
                            "{}",
                            String::from_utf8_lossy(&out.stderr)
                        );
                        let header = format!("P6\n{w} {h}\n255\n");
                        assert!(out.stdout.starts_with(header.as_bytes()));
                        let expected = &out.stdout[header.len()..];
                        let native = Decoder::new()
                            .fancy_upsampling(false)
                            .decode(&jpeg, Unstoppable)
                            .unwrap();
                        assert_eq!(
                            native.pixels_u8().unwrap(),
                            expected,
                            "q{q} {sub:?} {method:?} {w}x{h} pattern {pattern}"
                        );
                        cells += 1;
                    }
                }
            }
        }
    }
    eprintln!("independent libjpeg-turbo RGB pixel equality: {cells} encodes");
}
