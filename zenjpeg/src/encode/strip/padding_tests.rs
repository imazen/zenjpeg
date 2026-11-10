//! Padding oracles use coordinate clamping, never the production pad helpers.
use super::*;

const SUBS: [Subsampling; 4] = [
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S420,
    Subsampling::S440,
];
const FORMATS: [PixelFormat; 12] = [
    PixelFormat::Rgb,
    PixelFormat::Rgba,
    PixelFormat::Bgr,
    PixelFormat::Bgra,
    PixelFormat::Bgrx,
    PixelFormat::Rgb16,
    PixelFormat::Rgba16,
    PixelFormat::RgbF32,
    PixelFormat::RgbaF32,
    PixelFormat::Gray,
    PixelFormat::Gray16,
    PixelFormat::GrayF32,
];

fn source(w: usize, h: usize, fmt: PixelFormat, seed: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let mut rgb = match seed % 4 {
                0 => [87; 3],
                1 => [47, 139, 203],
                2 => {
                    if (x + y) % 2 == 0 {
                        [24, 180, 91]
                    } else {
                        [230, 57, 164]
                    }
                }
                _ => [
                    (x * 43 + y * 71 + seed * 13) as u8,
                    (x * 79 + y * 19 + 37) as u8,
                    if x == w - 1 || y == h - 1 { 251 } else { 32 },
                ],
            };
            if matches!(
                fmt,
                PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx
            ) {
                rgb.swap(0, 2);
            }
            let channels = if fmt.is_grayscale() {
                1
            } else {
                fmt.bytes_per_pixel()
                    / if matches!(fmt, PixelFormat::Rgb16 | PixelFormat::Rgba16) {
                        2
                    } else if matches!(fmt, PixelFormat::RgbF32 | PixelFormat::RgbaF32) {
                        4
                    } else {
                        1
                    }
            };
            for c in 0..channels {
                let value = if c < 3 { rgb[c] } else { 17 };
                match fmt {
                    PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16 => {
                        data.extend_from_slice(&(u16::from(value) * 257).to_ne_bytes())
                    }
                    PixelFormat::RgbF32 | PixelFormat::RgbaF32 | PixelFormat::GrayF32 => {
                        data.extend_from_slice(&(f32::from(value) / 255.0).to_ne_bytes())
                    }
                    _ => data.push(value),
                }
            }
        }
    }
    data
}

fn processor(
    w: usize,
    h: usize,
    sub: Subsampling,
    fmt: PixelFormat,
    method: DownsamplingMethod,
) -> StripProcessor {
    StripProcessor::with_color_modes(
        w,
        h,
        sub,
        fmt,
        method,
        0,
        false,
        crate::encode::XybSubsampling::BQuarter,
        false,
        QuantContext::default_for_tests(),
        false,
    )
    .unwrap()
}

fn poison(p: &mut StripProcessor) {
    for plane in [
        &mut p.y_strip,
        &mut p.cb_strip,
        &mut p.cr_strip,
        &mut p.cb_down,
        &mut p.cr_down,
    ] {
        plane.fill(-9876.0);
    }
}

fn assert_plane(
    actual: &[f32],
    stride: usize,
    rows: usize,
    visible: &[f32],
    w: usize,
    h: usize,
    context: &str,
) {
    for y in 0..rows {
        for x in 0..stride {
            let expected = visible[y.min(h - 1) * w + x.min(w - 1)];
            assert_eq!(
                actual[y * stride + x],
                expected,
                "{context}: ({x},{y}) visible {w}x{h} stride {stride}"
            );
        }
    }
}

#[test]
fn pre_subsampled_padding_coordinate_oracle() {
    let mut cells = 0;
    for sub in SUBS {
        for w in (1..=33).chain([63, 64, 65, 127, 128, 129]) {
            let sh = if matches!(sub, Subsampling::S420 | Subsampling::S440) {
                16
            } else {
                8
            };
            for prefix in 0..=2 {
                for h in 1..=sh {
                    let mut p = processor(
                        w,
                        prefix * sh + h,
                        sub,
                        PixelFormat::Rgb,
                        DownsamplingMethod::Box,
                    );
                    // Populate the same reused buffers with a genuinely different complete strip.
                    for i in 0..prefix {
                        let s = source(w, sh, PixelFormat::Rgb, i + 1);
                        p.process_strip(&s, i * sh).unwrap();
                    }
                    if prefix == 0 {
                        poison(&mut p);
                    }
                    let cw = p.layout.c_width;
                    let ch = p.layout.c_strip_height_for(h);
                    let y: Vec<f32> = (0..w * h)
                        .map(|i| ((i * 43 + 7) % 193) as f32 - 90.0)
                        .collect();
                    let cb: Vec<f32> = (0..cw * ch)
                        .map(|i| ((i * 19 + 31) % 151) as f32 - 71.0)
                        .collect();
                    let cr: Vec<f32> = (0..cw * ch)
                        .map(|i| ((i * 37 + 11) % 173) as f32 - 83.0)
                        .collect();
                    p.process_strip_ycbcr_f32_subsampled(&y, &cb, &cr, prefix * sh)
                        .unwrap();
                    for (name, actual, stride, rows, visible, vw, vh) in [
                        ("Y", &p.y_strip, p.layout.padded_width, sh, &y, w, h),
                        (
                            "Cb",
                            &p.cb_down,
                            p.layout.padded_c_width,
                            p.layout.c_strip_height,
                            &cb,
                            cw,
                            ch,
                        ),
                        (
                            "Cr",
                            &p.cr_down,
                            p.layout.padded_c_width,
                            p.layout.c_strip_height,
                            &cr,
                            cw,
                            ch,
                        ),
                    ] {
                        let shifted: Vec<f32> = visible.iter().map(|v| v + 128.0).collect();
                        assert_plane(
                            actual,
                            stride,
                            rows,
                            &shifted,
                            vw,
                            vh,
                            &format!("{name} {sub:?} prefix {prefix}"),
                        );
                    }
                    cells += 1;
                }
            }
        }
    }
    eprintln!("pre-subsampled coordinate oracle: {cells} geometries, 3 planes each");
}

fn rgb_case(
    w: usize,
    h: usize,
    prefix: usize,
    sub: Subsampling,
    fmt: PixelFormat,
    method: DownsamplingMethod,
    seed: usize,
) {
    let mut p = processor(w, 1, sub, fmt, method);
    let sh = p.layout.strip_height;
    p = processor(w, prefix * sh + h, sub, fmt, method);
    for i in 0..prefix {
        p.process_strip(&source(w, sh, fmt, i + 1), i * sh).unwrap();
    }
    if prefix == 0 {
        poison(&mut p);
    }
    let input = source(w, h, fmt, seed);
    // Only color conversion is shared with the oracle. Expected padding and
    // downsampling use independent coordinate clamps and scalar arithmetic.
    let mut visible = processor(w, h, sub, fmt, method);
    let n = w * h;
    let cw = p.layout.c_width;
    let ch = p.layout.c_strip_height_for(h);
    let mut yy = vec![0.; n];
    let mut cb = vec![0.; cw * ch];
    let mut cr = cb.clone();
    if !fmt.is_grayscale() && method.uses_gamma_aware() && sub != Subsampling::S444 {
        crate::encode::chroma::gamma_aware_strip(
            &input,
            w,
            h,
            w,
            fmt,
            sub,
            method == DownsamplingMethod::GammaAwareIterative,
            &mut yy,
            &mut cb,
            &mut cr,
        )
        .unwrap();
    } else if sub == Subsampling::S420
        && matches!(
            fmt,
            PixelFormat::Rgb
                | PixelFormat::Rgba
                | PixelFormat::Bgr
                | PixelFormat::Bgra
                | PixelFormat::Bgrx
        )
    {
        let mut ty = vec![0; n];
        let mut tc = vec![0; cw * ch];
        let mut tr = tc.clone();
        if matches!(
            fmt,
            PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx
        ) {
            crate::color::fast_yuv::bgr_to_ycbcr_420_reuse(
                &input,
                &mut yy,
                &mut cb,
                &mut cr,
                &mut ty,
                &mut tc,
                &mut tr,
                w,
                h,
                w,
                fmt.bytes_per_pixel(),
            );
        } else {
            crate::color::fast_yuv::rgb_to_ycbcr_420_reuse(
                &input,
                &mut yy,
                &mut cb,
                &mut cr,
                &mut ty,
                &mut tc,
                &mut tr,
                w,
                h,
                w,
                fmt.bytes_per_pixel(),
            );
        }
    } else {
        visible.convert_strip_to_ycbcr(&input, h).unwrap();
        for y in 0..h {
            yy[y * w..(y + 1) * w].copy_from_slice(
                &visible.y_strip
                    [y * visible.layout.padded_width..y * visible.layout.padded_width + w],
            );
        }
        let hs = if matches!(sub, Subsampling::S420 | Subsampling::S422) {
            2
        } else {
            1
        };
        let vs = if matches!(sub, Subsampling::S420 | Subsampling::S440) {
            2
        } else {
            1
        };
        for (dst, src) in [(&mut cb, &visible.cb_strip), (&mut cr, &visible.cr_strip)] {
            if fmt.is_grayscale() {
                continue;
            }
            for y in 0..ch {
                for x in 0..cw {
                    let a = src[(y * vs).min(h - 1) * w + (x * hs).min(w - 1)];
                    let b = src[(y * vs).min(h - 1) * w + (x * hs + hs - 1).min(w - 1)];
                    let c = src[(y * vs + vs - 1).min(h - 1) * w + (x * hs).min(w - 1)];
                    let d = src[(y * vs + vs - 1).min(h - 1) * w + (x * hs + hs - 1).min(w - 1)];
                    dst[y * cw + x] = match (hs, vs) {
                        (2, 2) => (a + b + c + d) * 0.25,
                        (2, 1) => (a + b) * 0.5,
                        (1, 2) => (a + c) * 0.5,
                        _ => a,
                    };
                }
            }
        }
    }
    p.process_strip(&input, prefix * sh).unwrap();
    let ctx = format!("{sub:?} {fmt:?} {method:?} prefix {prefix} seed {seed}");
    assert_plane(&p.y_strip, p.layout.padded_width, sh, &yy, w, h, &ctx);
    if !fmt.is_grayscale() {
        assert_plane(
            &p.cb_down,
            p.layout.padded_c_width,
            p.layout.c_strip_height,
            &cb,
            cw,
            ch,
            &ctx,
        );
        assert_plane(
            &p.cr_down,
            p.layout.padded_c_width,
            p.layout.c_strip_height,
            &cr,
            cw,
            ch,
            &ctx,
        );
    }
}

#[test]
fn rgb_padding_all_residues_and_reused_strips() {
    let _dispatch = archmage::testing::lock_token_testing();
    let mut cells = 0;
    for sub in SUBS {
        for method in [
            DownsamplingMethod::Box,
            DownsamplingMethod::GammaAware,
            DownsamplingMethod::GammaAwareIterative,
        ] {
            let sh = if matches!(sub, Subsampling::S420 | Subsampling::S440) {
                16
            } else {
                8
            };
            for w in (1..=33).chain([63, 64, 65, 127, 128, 129]) {
                for h in 1..=sh {
                    for prefix in 0..=2 {
                        rgb_case(
                            w,
                            h,
                            prefix,
                            sub,
                            PixelFormat::Rgb,
                            method,
                            (w + h + prefix) % 4,
                        );
                        cells += 1;
                    }
                }
            }
        }
    }
    eprintln!("RGB coordinate oracle: {cells} geometries/configurations");
}

#[test]
fn format_padding_aliases_and_fallbacks() {
    let _dispatch = archmage::testing::lock_token_testing();
    let mut cells = 0;
    for fmt in FORMATS {
        for sub in SUBS {
            for method in [
                DownsamplingMethod::Box,
                DownsamplingMethod::GammaAware,
                DownsamplingMethod::GammaAwareIterative,
            ] {
                let sh = if matches!(sub, Subsampling::S420 | Subsampling::S440) {
                    16
                } else {
                    8
                };
                for (w, h) in [
                    (1, 1),
                    (7, 3),
                    (8, sh),
                    (9, 7),
                    (15, sh - 1),
                    (16, sh),
                    (17, 1),
                    (31, 7),
                    (32, sh),
                    (33, sh - 1),
                    (129, 3),
                ] {
                    for seed in 0..4 {
                        rgb_case(w, h, 2, sub, fmt, method, seed);
                        cells += 1;
                    }
                }
            }
        }
    }
    eprintln!("format coordinate oracle: {cells} geometries/configurations/patterns");
}

#[test]
fn fused_420_partial_bottom_witness() {
    rgb_case(
        32,
        8,
        0,
        Subsampling::S420,
        PixelFormat::Rgb,
        DownsamplingMethod::Box,
        0,
    );
}
#[test]
fn gamma_partial_bottom_witness() {
    for sub in [Subsampling::S420, Subsampling::S422, Subsampling::S440] {
        rgb_case(
            32,
            1,
            0,
            sub,
            PixelFormat::Rgb,
            DownsamplingMethod::GammaAware,
            1,
        );
    }
}
#[test]
fn pre_subsampled_horizontal_witness() {
    let mut p = processor(
        9,
        16,
        Subsampling::S420,
        PixelFormat::Rgb,
        DownsamplingMethod::Box,
    );
    poison(&mut p);
    p.process_strip_ycbcr_f32_subsampled(&[0.; 144], &[20.; 40], &[-10.; 40], 0)
        .unwrap();
    assert_plane(
        &p.cb_down,
        p.layout.padded_c_width,
        8,
        &[148.; 40],
        5,
        8,
        "pre-subsampled horizontal",
    );
}
#[test]
fn pre_subsampled_vertical_stride_witness() {
    let mut p = processor(
        9,
        1,
        Subsampling::S420,
        PixelFormat::Rgb,
        DownsamplingMethod::Box,
    );
    p.cb_down.fill(148.);
    p.cr_down.fill(118.);
    // Isolate vertical stride from horizontal padding: prefill a valid padded first row.
    p.cb_down[8..].fill(-9876.);
    p.cr_down[8..].fill(-9876.);
    p.pad_chroma_down_vertically(1).unwrap();
    assert_plane(
        &p.cb_down,
        8,
        8,
        &[148.; 5],
        5,
        1,
        "pre-subsampled vertical stride",
    );
}

#[test]
fn standard_downsample_even_partial_bottom_witness() {
    rgb_case(
        1,
        2,
        0,
        Subsampling::S440,
        PixelFormat::Rgb,
        DownsamplingMethod::Box,
        3,
    );
    rgb_case(
        9,
        2,
        2,
        Subsampling::S420,
        PixelFormat::Rgb16,
        DownsamplingMethod::Box,
        3,
    );
}

#[cfg(feature = "parallel")]
#[test]
fn dct_parallel_threshold_and_chunk_edges() {
    let _dispatch = archmage::testing::lock_token_testing();
    // Y threshold/chunks = 4096 blocks; chroma threshold = 2048.
    // One/two-row strips cross these at different source widths.
    for sub in SUBS {
        for w in [
            16375, 16376, 16377, 16383, 16384, 16385, 32759, 32760, 32761, 32767, 32768, 32769,
        ] {
            for h in [1, 8] {
                rgb_case(w, h, 2, sub, PixelFormat::Rgb, DownsamplingMethod::Box, 3);
            }
        }
    }
}

#[test]
fn reachable_dispatch_padding_oracles() {
    let report = archmage::testing::for_each_token_permutation(
        archmage::testing::CompileTimePolicy::Warn,
        |perm| {
            for sub in SUBS {
                for method in [
                    DownsamplingMethod::Box,
                    DownsamplingMethod::GammaAware,
                    DownsamplingMethod::GammaAwareIterative,
                ] {
                    for (w, h) in [(1, 1), (7, 2), (9, 7), (17, 3), (33, 7), (65, 2), (129, 7)] {
                        rgb_case(w, h, 2, sub, PixelFormat::Rgb, method, 3);
                        rgb_case(w, h, 2, sub, PixelFormat::RgbF32, method, 1);
                    }
                }
            }
            eprintln!("padding dispatch permutation: {perm}");
        },
    );
    eprintln!("padding dispatch report: {report:?}");
}
