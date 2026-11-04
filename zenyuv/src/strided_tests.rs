//! Every plane at its own stride, in every SIMD tier: the same bytes as the tight
//! form, with padding neither read nor written. Input padding holds garbage,
//! output padding a sentinel, and every buffer is the exact minimum
//! `(rows - 1) * stride + width`, so a path that wrongly needs `rows * stride`
//! panics here.

use crate::types::check_plane;
use crate::{Matrix, Range, SharpYuvConfig, YuvContext, YuvError, decode, sharp};
use alloc::{format, vec, vec::Vec};
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use imgref::{ImgRef, ImgRefMut};

/// Widths below, at and above the kernels' 8- and 32-pixel chunks; odd heights.
const DIMS: [(usize, usize); 11] = [
    (1, 1),
    (2, 2),
    (3, 3),
    (7, 5),
    (8, 4),
    (17, 9),
    (31, 17),
    (32, 16),
    (33, 17),
    (37, 23),
    (65, 67),
];
const MODES: [(Range, Matrix); 3] = [
    (Range::Full, Matrix::Bt601),
    (Range::Limited, Matrix::Bt709),
    (Range::Full, Matrix::Bt2020),
];

fn pixels(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [
                (x * 7 + y * 3) as u8,
                ((x * 3) ^ (y * 11)) as u8,
                ((x + y) * 5 + (x * y) / 3) as u8,
            ]
        })
        .collect()
}

fn with_padding<T: Copy>(tight: &[T], w: usize, stride: usize, fill: T) -> Vec<T> {
    let h = tight.len().checked_div(w).unwrap_or(0);
    let mut out = vec![fill; (h.max(1) - 1) * stride + w];
    for (row, line) in tight.chunks_exact(w).enumerate() {
        out[row * stride..row * stride + w].copy_from_slice(line);
    }
    out
}

/// A plane's image samples, tight, after checking its padding kept `fill`.
fn image_of<T: Copy + PartialEq + core::fmt::Debug>(
    buf: &[T],
    w: usize,
    stride: usize,
    fill: T,
) -> Vec<T> {
    let mut image = Vec::new();
    for (i, s) in buf.iter().enumerate() {
        if i % stride < w {
            image.push(*s);
        } else {
            assert_eq!(*s, fill, "padding at {i} was written");
        }
    }
    image
}

const RGB_GARBAGE: [u8; 3] = [0xA5, 0x5A, 0xFF];
const OUT_FILL: u8 = 0xC3;
const OUT_FILL_F: f32 = -7.5;

struct Input {
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    tight: Vec<[u8; 3]>,
    tight_bytes: Vec<u8>,
    padded: Vec<[u8; 3]>,
}

const RGB_PAD: usize = 5;
const Y_PAD: usize = 7;
const CB_PAD: usize = 3;
const CR_PAD: usize = 11;

fn input(w: usize, h: usize) -> Input {
    let tight = pixels(w, h);
    let padded = with_padding(&tight, w, w + RGB_PAD, RGB_GARBAGE);
    Input {
        w,
        h,
        cw: w.div_ceil(2),
        ch: h.div_ceil(2),
        tight_bytes: tight.as_flattened().to_vec(),
        tight,
        padded,
    }
}

impl Input {
    fn rgb(&self) -> ImgRef<'_, [u8; 3]> {
        ImgRef::new_stride(&self.padded, self.w, self.h, self.w + RGB_PAD)
    }
}

fn zeroed<T: Copy>(w: usize, h: usize, pad: usize, fill: T) -> (Vec<T>, usize) {
    let stride = w + pad;
    (vec![fill; (h - 1) * stride + w], stride)
}

/// Run `f` in every token permutation.
fn all_tiers(mut f: impl FnMut()) {
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |_| f());
    // Elsewhere the base tier can be compiled in, leaving one permutation.
    #[cfg(target_arch = "x86_64")]
    assert!(report.permutations_run >= 2);
    #[cfg(not(target_arch = "x86_64"))]
    assert!(report.permutations_run >= 1);
}

fn context_u8_matches(c: &mut YuvContext, i: &Input, label: &str) {
    let (w, h, cw, ch) = (i.w, i.h, i.cw, i.ch);
    // 4:2:0
    let (mut ty, mut tb, mut tr) = (vec![0u8; w * h], vec![0u8; cw * ch], vec![0u8; cw * ch]);
    c.encode_420_u8(&i.tight_bytes, &mut ty, &mut tb, &mut tr, w, h);
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL);
    let (mut pb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL);
    let (mut pr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL);
    c.try_encode_420_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
    )
    .unwrap();
    assert_eq!(image_of(&py, w, ys, OUT_FILL), ty, "{label} 420 y");
    assert_eq!(image_of(&pb, cw, bs, OUT_FILL), tb, "{label} 420 cb");
    assert_eq!(image_of(&pr, cw, rs, OUT_FILL), tr, "{label} 420 cr");
    // Y only
    let mut ty = vec![0u8; w * h];
    c.encode_420_y_only_u8(&i.tight_bytes, &mut ty, w, h);
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL);
    c.try_encode_420_y_only_strided(i.rgb(), ImgRefMut::new_stride(&mut py, w, h, ys))
        .unwrap();
    assert_eq!(image_of(&py, w, ys, OUT_FILL), ty, "{label} y only");
    // 4:4:4, with tight views too: that takes the one-run fast path.
    let (mut ty, mut tb, mut tr) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
    c.encode_444_u8(&i.tight_bytes, &mut ty, &mut tb, &mut tr, w, h);
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL);
    let (mut pb, bs) = zeroed(w, h, CB_PAD, OUT_FILL);
    let (mut pr, rs) = zeroed(w, h, CR_PAD, OUT_FILL);
    c.try_encode_444_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, w, h, bs),
        ImgRefMut::new_stride(&mut pr, w, h, rs),
    )
    .unwrap();
    assert_eq!(image_of(&py, w, ys, OUT_FILL), ty, "{label} 444 y");
    assert_eq!(image_of(&pb, w, bs, OUT_FILL), tb, "{label} 444 cb");
    assert_eq!(image_of(&pr, w, rs, OUT_FILL), tr, "{label} 444 cr");
    let (mut sy, mut sb, mut sr) = (vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]);
    c.try_encode_444_strided(
        ImgRef::new(&i.tight, w, h),
        ImgRefMut::new(&mut sy, w, h),
        ImgRefMut::new(&mut sb, w, h),
        ImgRefMut::new(&mut sr, w, h),
    )
    .unwrap();
    assert_eq!((sy, sb, sr), (ty, tb, tr), "{label} 444 tight view");
}

fn context_f32_matches(c: &mut YuvContext, i: &Input, label: &str) {
    let (w, h, cw, ch) = (i.w, i.h, i.cw, i.ch);
    let (mut ty, mut tb, mut tr) = (vec![0f32; w * h], vec![0f32; cw * ch], vec![0f32; cw * ch]);
    c.encode_420_f32(&i.tight_bytes, &mut ty, &mut tb, &mut tr, w, h);
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL_F);
    let (mut pb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL_F);
    let (mut pr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL_F);
    c.try_encode_420_f32_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
    )
    .unwrap();
    assert_eq!(image_of(&py, w, ys, OUT_FILL_F), ty, "{label} f32 y");
    assert_eq!(image_of(&pb, cw, bs, OUT_FILL_F), tb, "{label} f32 cb");
    assert_eq!(image_of(&pr, cw, rs, OUT_FILL_F), tr, "{label} f32 cr");
}

fn context_sharp_matches(c: &mut YuvContext, i: &Input, label: &str) {
    let (w, h, cw, ch) = (i.w, i.h, i.cw, i.ch);
    let config = SharpYuvConfig::default();
    let (mut ty, mut tb, mut tr) = (vec![0u8; w * h], vec![0u8; cw * ch], vec![0u8; cw * ch]);
    c.encode_sharp_420_u8(&i.tight_bytes, &mut ty, &mut tb, &mut tr, w, h, &config);
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL);
    let (mut pb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL);
    let (mut pr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL);
    c.try_encode_sharp_420_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
        &config,
    )
    .unwrap();
    assert_eq!(image_of(&py, w, ys, OUT_FILL), ty, "{label} sharp y");
    assert_eq!(image_of(&pb, cw, bs, OUT_FILL), tb, "{label} sharp cb");
    assert_eq!(image_of(&pr, cw, rs, OUT_FILL), tr, "{label} sharp cr");

    let (mut fy, mut fb, mut fr) = (vec![0f32; w * h], vec![0f32; cw * ch], vec![0f32; cw * ch]);
    c.encode_sharp_420_f32(&i.tight_bytes, &mut fy, &mut fb, &mut fr, w, h, &config);
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL_F);
    let (mut pb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL_F);
    let (mut pr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL_F);
    c.try_encode_sharp_420_f32_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
        &config,
    )
    .unwrap();
    assert_eq!(image_of(&py, w, ys, OUT_FILL_F), fy, "{label} sharp f32 y");
    assert_eq!(
        image_of(&pb, cw, bs, OUT_FILL_F),
        fb,
        "{label} sharp f32 cb"
    );
    assert_eq!(
        image_of(&pr, cw, rs, OUT_FILL_F),
        fr,
        "{label} sharp f32 cr"
    );
}

fn sharp_refine_matches(i: &Input, range: Range, matrix: Matrix, label: &str) {
    let (w, h, cw, ch) = (i.w, i.h, i.cw, i.ch);
    let config = SharpYuvConfig::default();
    let mut ws = sharp::SharpYuvWorkspace::new(cw);
    // Start both forms from the box-average planes.
    let (mut ty, mut tb, mut tr) = (vec![0u8; w * h], vec![0u8; cw * ch], vec![0u8; cw * ch]);
    crate::rgb_to_yuv420_with(
        &i.tight_bytes,
        &mut ty,
        &mut tb,
        &mut tr,
        w,
        h,
        range,
        matrix,
    );
    let py = with_padding(&ty, w, w + Y_PAD, 0xEE);
    let (mut pb, bs) = (with_padding(&tb, cw, cw + CB_PAD, OUT_FILL), cw + CB_PAD);
    let (mut pr, rs) = (with_padding(&tr, cw, cw + CR_PAD, OUT_FILL), cw + CR_PAD);
    sharp::refine_chroma_420_u8_with_workspace(
        &i.tight_bytes,
        &ty,
        &mut tb,
        &mut tr,
        w,
        h,
        range,
        matrix,
        &config,
        &mut ws,
    );
    sharp::refine_chroma_420_strided(
        i.rgb(),
        ImgRef::new_stride(&py, w, h, w + Y_PAD),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
        range,
        matrix,
        &config,
        &mut ws,
    );
    assert_eq!(image_of(&pb, cw, bs, OUT_FILL), tb, "{label} refine cb");
    assert_eq!(image_of(&pr, cw, rs, OUT_FILL), tr, "{label} refine cr");

    // Luma refinement, with garbage in the chroma padding it reads around.
    let mut py = with_padding(&ty, w, w + Y_PAD, OUT_FILL);
    let pcb = with_padding(&tb, cw, cw + CB_PAD, 0xEE);
    let pcr = with_padding(&tr, cw, cw + CR_PAD, 0x11);
    let total_tight =
        sharp::refine_y_420_u8(&i.tight_bytes, &mut ty, &tb, &tr, w, h, range, matrix);
    let total = sharp::refine_y_420_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, w + Y_PAD),
        ImgRef::new_stride(&pcb, cw, ch, cw + CB_PAD),
        ImgRef::new_stride(&pcr, cw, ch, cw + CR_PAD),
        range,
        matrix,
    );
    assert_eq!(total, total_tight, "{label} refine y total");
    assert_eq!(
        image_of(&py, w, w + Y_PAD, OUT_FILL),
        ty,
        "{label} refine y"
    );

    // The free sharp entry points, against their tight forms.
    let (mut ty, mut tb, mut tr) = (vec![0u8; w * h], vec![0u8; cw * ch], vec![0u8; cw * ch]);
    sharp::rgb_to_yuv420_sharp_with_workspace(
        &i.tight_bytes,
        &mut ty,
        &mut tb,
        &mut tr,
        w,
        h,
        range,
        matrix,
        &config,
        &mut ws,
    );
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL);
    let (mut pb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL);
    let (mut pr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL);
    sharp::rgb_to_yuv420_sharp_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
        range,
        matrix,
        &config,
        &mut ws,
    );
    assert_eq!(image_of(&py, w, ys, OUT_FILL), ty, "{label} free sharp y");
    assert_eq!(image_of(&pb, cw, bs, OUT_FILL), tb, "{label} free sharp cb");
    assert_eq!(image_of(&pr, cw, rs, OUT_FILL), tr, "{label} free sharp cr");

    let (mut fy, mut fb, mut fr) = (vec![0f32; w * h], vec![0f32; cw * ch], vec![0f32; cw * ch]);
    sharp::rgb_to_yuv420_sharp_f32(
        &i.tight_bytes,
        &mut fy,
        &mut fb,
        &mut fr,
        w,
        h,
        range,
        matrix,
        &config,
        &mut ws,
    );
    let (mut py, ys) = zeroed(w, h, Y_PAD, OUT_FILL_F);
    let (mut pb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL_F);
    let (mut pr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL_F);
    sharp::rgb_to_yuv420_sharp_f32_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut py, w, h, ys),
        ImgRefMut::new_stride(&mut pb, cw, ch, bs),
        ImgRefMut::new_stride(&mut pr, cw, ch, rs),
        range,
        matrix,
        &config,
        &mut ws,
    );
    assert_eq!(
        image_of(&py, w, ys, OUT_FILL_F),
        fy,
        "{label} free sharp f32 y"
    );
    assert_eq!(
        image_of(&pb, cw, bs, OUT_FILL_F),
        fb,
        "{label} free sharp f32 cb"
    );
    assert_eq!(
        image_of(&pr, cw, rs, OUT_FILL_F),
        fr,
        "{label} free sharp f32 cr"
    );
}

#[test]
fn encode_entry_points_match_tight_in_every_tier() {
    for (w, h) in DIMS {
        let i = input(w, h);
        for (range, matrix) in MODES {
            let label = format!("{w}x{h} {range:?} {matrix:?}");
            all_tiers(|| {
                let mut c = YuvContext::new(range, matrix);
                context_u8_matches(&mut c, &i, &label);
                context_f32_matches(&mut c, &i, &label);
            });
        }
    }
}

#[test]
fn sharp_entry_points_match_tight_in_every_tier() {
    for (w, h) in DIMS {
        let i = input(w, h);
        let (range, matrix) = (Range::Limited, Matrix::Bt601);
        let label = format!("{w}x{h}");
        all_tiers(|| {
            let mut c = YuvContext::new(range, matrix);
            context_sharp_matches(&mut c, &i, &label);
            sharp_refine_matches(&i, range, matrix, &label);
        });
    }
}

#[test]
fn decode_matches_tight_in_every_tier() {
    /// Widen a tight plane with garbage, decode through `strided`, compare.
    type Tight<'a> = (&'a [u8], &'a [u8], &'a [u8]);
    for (w, h) in DIMS {
        // Tie-prone values: the SIMD lanes and scalar tail can round these apart,
        // so a strided path that moved the tail would be caught.
        let plane = |pw: usize, ph: usize, k: usize| -> Vec<u8> {
            (0..pw * ph)
                .map(|n| ((n * (k + 3) + k) % 253) as u8)
                .collect()
        };
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let y = plane(w, h, 0);
        for (range, matrix) in MODES {
            let label = format!("{w}x{h} {range:?} {matrix:?}");
            let py = with_padding(&y, w, w + Y_PAD, 0xFF);
            let vy = ImgRef::new_stride(&py[..], w, h, w + Y_PAD);
            let check = |name: &str,
                         tight: &dyn Fn(Tight<'_>, &mut [u8]),
                         strided: &dyn Fn(
                ImgRef<'_, u8>,
                ImgRef<'_, u8>,
                ImgRef<'_, u8>,
                ImgRefMut<'_, [u8; 3]>,
            ),
                         ccw: usize,
                         cch: usize| {
                let (cb, cr) = (plane(ccw, cch, 1), plane(ccw, cch, 2));
                let (pb, pr) = (
                    with_padding(&cb, ccw, ccw + CB_PAD, 0xEE),
                    with_padding(&cr, ccw, ccw + CR_PAD, 0x11),
                );
                all_tiers(|| {
                    let mut expected = vec![0u8; w * h * 3];
                    tight((&y, &cb, &cr), &mut expected);
                    let (mut out, stride) = zeroed(w, h, RGB_PAD, [OUT_FILL; 3]);
                    strided(
                        vy,
                        ImgRef::new_stride(&pb[..], ccw, cch, ccw + CB_PAD),
                        ImgRef::new_stride(&pr[..], ccw, cch, ccw + CR_PAD),
                        ImgRefMut::new_stride(&mut out[..], w, h, stride),
                    );
                    let image: Vec<u8> = image_of(&out, w, stride, [OUT_FILL; 3])
                        .into_iter()
                        .flatten()
                        .collect();
                    assert!(image == expected, "{label} {name}");
                });
            };
            check(
                "444",
                &|(y, cb, cr), o| decode::yuv444_to_rgb_with(y, cb, cr, o, w, h, range, matrix),
                &|y, cb, cr, o| decode::yuv444_to_rgb_strided(y, cb, cr, o, range, matrix),
                w,
                h,
            );
            check(
                "420",
                &|(y, cb, cr), o| decode::yuv420_to_rgb_with(y, cb, cr, o, w, h, range, matrix),
                &|y, cb, cr, o| decode::yuv420_to_rgb_strided(y, cb, cr, o, range, matrix),
                cw,
                ch,
            );
            check(
                "420 bilinear",
                &|(y, cb, cr), o| {
                    decode::yuv420_to_rgb_bilinear_with(y, cb, cr, o, w, h, range, matrix)
                },
                &|y, cb, cr, o| decode::yuv420_to_rgb_bilinear_strided(y, cb, cr, o, range, matrix),
                cw,
                ch,
            );
            check(
                "422",
                &|(y, cb, cr), o| decode::yuv422_to_rgb_with(y, cb, cr, o, w, h, range, matrix),
                &|y, cb, cr, o| decode::yuv422_to_rgb_strided(y, cb, cr, o, range, matrix),
                cw,
                h,
            );
            check(
                "400",
                &|(y, _, _), o| decode::yuv400_to_rgb_with(y, o, w, h, range, matrix),
                &|y, _, _, o| decode::yuv400_to_rgb_strided(y, o, range, matrix),
                1,
                1,
            );
        }
    }
}

/// The kernel's SIMD lanes and scalar tail round these Y/Cb/Cr triples apart
/// (found by exhaustive search, 2026-11-03, x86-64 v3), so a strided path must
/// send the same pixels down the tail as the tight one. At a width of 9, a
/// per-row loop would send a pixel per row down it; the tight form sends one or
/// two per image.
#[test]
fn decode_444_keeps_the_tight_forms_tail_on_rounding_ties() {
    let (w, h) = (9usize, 8usize);
    for (range, matrix, (yv, cbv, crv)) in [
        (Range::Full, Matrix::Bt601, (1u8, 253u8, 0u8)),
        (Range::Limited, Matrix::Bt709, (27, 133, 29)),
        (Range::Full, Matrix::Bt2020, (17, 131, 44)),
    ] {
        let (y, cb, cr) = (vec![yv; w * h], vec![cbv; w * h], vec![crv; w * h]);
        let py = with_padding(&y, w, w + Y_PAD, 0xFF);
        let pb = with_padding(&cb, w, w + CB_PAD, 0xEE);
        let pr = with_padding(&cr, w, w + CR_PAD, 0x11);
        let mut expected = vec![0u8; w * h * 3];
        decode::yuv444_to_rgb_with(&y, &cb, &cr, &mut expected, w, h, range, matrix);
        let (mut out, stride) = zeroed(w, h, RGB_PAD, [OUT_FILL; 3]);
        decode::yuv444_to_rgb_strided(
            ImgRef::new_stride(&py[..], w, h, w + Y_PAD),
            ImgRef::new_stride(&pb[..], w, h, w + CB_PAD),
            ImgRef::new_stride(&pr[..], w, h, w + CR_PAD),
            ImgRefMut::new_stride(&mut out[..], w, h, stride),
            range,
            matrix,
        );
        let image: Vec<u8> = image_of(&out, w, stride, [OUT_FILL; 3])
            .into_iter()
            .flatten()
            .collect();
        assert!(image == expected, "{range:?} {matrix:?}");
    }
}

#[test]
fn context_refuses_bad_geometry_and_writes_nothing() {
    let (w, h) = (37usize, 23usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let i = input(w, h);
    let config = SharpYuvConfig::default();
    let mut c = YuvContext::new(Range::Full, Matrix::Bt601);

    let (mut y, ys) = zeroed(w, h, Y_PAD, OUT_FILL);
    let (mut cb, bs) = zeroed(cw, ch, CB_PAD, OUT_FILL);
    let (mut cr, rs) = zeroed(cw, ch, CR_PAD, OUT_FILL);

    macro_rules! refused {
        ($y:expr, $cb:expr, $cr:expr, $expect:pat, $what:expr) => {{
            let r = c.try_encode_420_strided(i.rgb(), $y, $cb, $cr);
            assert!(matches!(r, Err($expect)), "420 {}: {r:?}", $what);
            let r = c.try_encode_sharp_420_strided(i.rgb(), $y, $cb, $cr, &config);
            assert!(matches!(r, Err($expect)), "sharp {}: {r:?}", $what);
        }};
    }
    let n = y.len();
    // One sample short of the minimum, on each plane.
    refused!(
        ImgRefMut::new_stride(&mut y[..n - 1], w, h, ys),
        ImgRefMut::new_stride(&mut cb, cw, ch, bs),
        ImgRefMut::new_stride(&mut cr, cw, ch, rs),
        YuvError::BufferTooSmall { buffer: "y", .. },
        "short y"
    );
    let nb = cb.len();
    refused!(
        ImgRefMut::new_stride(&mut y, w, h, ys),
        ImgRefMut::new_stride(&mut cb[..nb - 1], cw, ch, bs),
        ImgRefMut::new_stride(&mut cr, cw, ch, rs),
        YuvError::BufferTooSmall { buffer: "cb", .. },
        "short cb"
    );
    let nr = cr.len();
    refused!(
        ImgRefMut::new_stride(&mut y, w, h, ys),
        ImgRefMut::new_stride(&mut cb, cw, ch, bs),
        ImgRefMut::new_stride(&mut cr[..nr - 1], cw, ch, rs),
        YuvError::BufferTooSmall { buffer: "cr", .. },
        "short cr"
    );
    // Dimensions that disagree with the image.
    refused!(
        ImgRefMut::new_stride(&mut y, w - 1, h, ys),
        ImgRefMut::new_stride(&mut cb, cw, ch, bs),
        ImgRefMut::new_stride(&mut cr, cw, ch, rs),
        YuvError::WrongDimensions { buffer: "y", .. },
        "narrow y"
    );
    refused!(
        ImgRefMut::new_stride(&mut y, w, h, ys),
        ImgRefMut::new_stride(&mut cb, cw, ch - 1, bs),
        ImgRefMut::new_stride(&mut cr, cw, ch, rs),
        YuvError::WrongDimensions { buffer: "cb", .. },
        "short cb rows"
    );
    // Short RGB.
    let short_rgb = ImgRef::new_stride(&i.padded[..i.padded.len() - 1], w, h, w + RGB_PAD);
    let r = c.try_encode_444_strided(
        short_rgb,
        ImgRefMut::new_stride(&mut y, w, h, ys),
        ImgRefMut::new_stride(&mut cb, cw, ch, bs),
        ImgRefMut::new_stride(&mut cr, cw, ch, rs),
    );
    assert!(matches!(
        r,
        Err(YuvError::BufferTooSmall { buffer: "rgb", .. })
    ));
    assert!(y.iter().chain(&cb).chain(&cr).all(|&v| v == OUT_FILL));

    // The pixel limit applies too.
    let mut small = YuvContext::new(Range::Full, Matrix::Bt601).with_max_pixels(10);
    let r = small.try_encode_420_y_only_strided(i.rgb(), ImgRefMut::new_stride(&mut y, w, h, ys));
    assert!(matches!(r, Err(YuvError::TooManyPixels { .. })));
}

#[test]
fn plane_checks_cover_every_rule_with_checked_arithmetic() {
    assert_eq!(check_plane("y", 15, (4, 3), 5, (4, 3)), Ok(()));
    assert_eq!(check_plane("y", 14, (4, 3), 5, (4, 3)), Ok(()));
    assert!(matches!(
        check_plane("y", 13, (4, 3), 5, (4, 3)),
        Err(YuvError::BufferTooSmall { needed: 14, .. })
    ));
    assert!(matches!(
        check_plane("y", 100, (4, 3), 3, (4, 3)),
        Err(YuvError::StrideTooSmall { stride: 3, .. })
    ));
    assert!(matches!(
        check_plane("y", 100, (4, 3), usize::MAX, (4, 3)),
        Err(YuvError::Overflow)
    ));
    assert_eq!(check_plane("y", 0, (4, 0), 4, (4, 0)), Ok(()));
}

#[test]
#[should_panic(expected = "y buffer holds")]
fn free_sharp_entry_panics_on_a_short_plane() {
    let i = input(9, 9);
    let mut ws = sharp::SharpYuvWorkspace::new(5);
    let (mut y, ys) = zeroed(9, 9, Y_PAD, 0u8);
    let (mut cb, bs) = zeroed(5, 5, CB_PAD, 0u8);
    let (mut cr, rs) = zeroed(5, 5, CR_PAD, 0u8);
    let n = y.len();
    sharp::rgb_to_yuv420_sharp_strided(
        i.rgb(),
        ImgRefMut::new_stride(&mut y[..n - 1], 9, 9, ys),
        ImgRefMut::new_stride(&mut cb, 5, 5, bs),
        ImgRefMut::new_stride(&mut cr, 5, 5, rs),
        Range::Full,
        Matrix::Bt601,
        &SharpYuvConfig::default(),
        &mut ws,
    );
}

/// imgref itself refuses a stride below the width, so a view cannot carry one.
#[test]
#[should_panic]
fn a_view_cannot_carry_a_stride_below_its_width() {
    let buf = [0u8; 64];
    let _ = ImgRef::new_stride(&buf[..], 8, 4, 7);
}
