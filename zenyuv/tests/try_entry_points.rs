//! The `YuvContext` entry points: the bytes of the panicking forms they replaced
//! on valid input, and a typed refusal (with nothing written) where those panicked.

use imgref::{ImgRef, ImgRefMut};
use zenyuv::{DEFAULT_MAX_PIXELS, Matrix, Range, SharpYuvConfig, YuvContext, YuvError};

const W: usize = 37;
const H: usize = 23;

fn rgb(w: usize, h: usize) -> Vec<u8> {
    (0..w * h * 3)
        .map(|i| (i * 37 + (i / 3) * 11) as u8)
        .collect()
}

fn ctx() -> YuvContext {
    YuvContext::new(Range::Limited, Matrix::Bt709)
}

const PIN_E420: u64 = 0x61f88b63a710472a;
const PIN_E420_Y_ONLY: u64 = 0x5eb0b8aa1169f160;
const PIN_E444: u64 = 0x5b2f16110e27aa53;
const PIN_E420_F32: u64 = 0x61f88b63a710472a;
const PIN_SHARP420: u64 = 0xa3d2bbef1df76540;
const PIN_SHARP420_F32: u64 = 0x1f865824d241cb74;

/// One entry point over u8 or f32 planes, with its plane sizes.
#[derive(Clone, Copy, Debug)]
enum Entry {
    E420,
    E420YOnly,
    E444,
    E420F32,
    Sharp420,
    Sharp420F32,
}

const ENTRIES: [Entry; 6] = [
    Entry::E420,
    Entry::E420YOnly,
    Entry::E444,
    Entry::E420F32,
    Entry::Sharp420,
    Entry::Sharp420F32,
];

/// A tight view's stride: imgref refuses a stride of 0, so a zero-width view
/// carries 1, and a plane of zero rows spans nothing either way.
fn tight(width: usize) -> usize {
    width.max(1)
}

fn view(p: &[u8], w: usize, h: usize) -> ImgRef<'_, u8> {
    ImgRef::new_stride(p, w, h, tight(w))
}

fn view_mut<T>(p: &mut [T], w: usize, h: usize) -> ImgRefMut<'_, T> {
    ImgRefMut::new_stride(p, w, h, tight(w))
}

impl Entry {
    fn chroma(self, w: usize, h: usize) -> usize {
        match self {
            Entry::E444 => w * h,
            _ => w.div_ceil(2) * h.div_ceil(2),
        }
    }

    fn chroma_dims(self, w: usize, h: usize) -> (usize, usize) {
        match self {
            Entry::E444 => (w, h),
            _ => (w.div_ceil(2), h.div_ceil(2)),
        }
    }

    /// Run the entry point on planes of the given lengths, filled with a
    /// sentinel. Returns the result and the planes as f32 (u8 widened).
    fn run(
        self,
        ctx: &mut YuvContext,
        rgb: &[u8],
        (y_len, c_len): (usize, usize),
        w: usize,
        h: usize,
    ) -> (Result<(), YuvError>, [Vec<f32>; 3]) {
        let cfg = SharpYuvConfig::default();
        let (mut y8, mut cb8, mut cr8) = (vec![7u8; y_len], vec![7u8; c_len], vec![7u8; c_len]);
        let (mut yf, mut cbf, mut crf) = (
            vec![7.0f32; y_len],
            vec![7.0f32; c_len],
            vec![7.0f32; c_len],
        );
        let (cw, ch) = self.chroma_dims(w, h);
        let src = view(rgb, w * 3, h);
        let r = match self {
            Entry::E420 => ctx.encode_420_u8(
                src,
                view_mut(&mut y8, w, h),
                view_mut(&mut cb8, cw, ch),
                view_mut(&mut cr8, cw, ch),
            ),
            Entry::E420YOnly => ctx.encode_420_y_only_u8(src, view_mut(&mut y8, w, h)),
            Entry::E444 => ctx.encode_444_u8(
                src,
                view_mut(&mut y8, w, h),
                view_mut(&mut cb8, cw, ch),
                view_mut(&mut cr8, cw, ch),
            ),
            Entry::E420F32 => ctx.encode_420_f32(
                src,
                view_mut(&mut yf, w, h),
                view_mut(&mut cbf, cw, ch),
                view_mut(&mut crf, cw, ch),
            ),
            Entry::Sharp420 => ctx.encode_sharp_420_u8(
                src,
                view_mut(&mut y8, w, h),
                view_mut(&mut cb8, cw, ch),
                view_mut(&mut cr8, cw, ch),
                &cfg,
            ),
            Entry::Sharp420F32 => ctx.encode_sharp_420_f32(
                src,
                view_mut(&mut yf, w, h),
                view_mut(&mut cbf, cw, ch),
                view_mut(&mut crf, cw, ch),
                &cfg,
            ),
        };
        let planes = match self {
            Entry::E420F32 | Entry::Sharp420F32 => [yf, cbf, crf],
            _ => [y8, cb8, cr8].map(|p| p.into_iter().map(f32::from).collect()),
        };
        (r, planes)
    }

    /// FNV-1a over the output planes' f32 bits, from the panicking form this entry
    /// point replaced, at commit 762fa712 on x86-64 v3 (and checked on aarch64).
    fn pinned_output(self) -> u64 {
        match self {
            Entry::E420 => PIN_E420,
            Entry::E420YOnly => PIN_E420_Y_ONLY,
            Entry::E444 => PIN_E444,
            Entry::E420F32 => PIN_E420_F32,
            Entry::Sharp420 => PIN_SHARP420,
            Entry::Sharp420F32 => PIN_SHARP420_F32,
        }
    }
}

fn fnv(planes: &[Vec<f32>; 3]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in planes.iter().flatten() {
        for b in v.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

#[test]
fn valid_input_gives_the_bytes_of_the_replaced_infallible_form() {
    let src = rgb(W, H);
    for e in ENTRIES {
        let lens = (W * H, e.chroma(W, H));
        let (r, got) = e.run(&mut ctx(), &src, lens, W, H);
        assert_eq!(r, Ok(()), "{e:?}");
        assert_eq!(fnv(&got), e.pinned_output(), "{e:?}");
        // A reused context (scratch already sized) gives the same bytes.
        let mut c = ctx();
        e.run(&mut c, &src, lens, W, H).0.unwrap();
        assert_eq!(e.run(&mut c, &src, lens, W, H).1, got, "{e:?} reused");
    }
}

#[test]
fn each_short_buffer_is_refused_and_nothing_is_written() {
    let src = rgb(W, H);
    for e in ENTRIES {
        let (n, c) = (W * H, e.chroma(W, H));
        let short_rgb = &src[..n * 3 - 1];
        let (r, planes) = e.run(&mut ctx(), short_rgb, (n, c), W, H);
        assert_eq!(
            r,
            Err(YuvError::BufferTooSmall {
                buffer: "rgb",
                needed: n * 3,
                len: n * 3 - 1
            }),
            "{e:?}"
        );
        assert!(planes.iter().flatten().all(|&v| v == 7.0), "{e:?} wrote");

        let (r, _) = e.run(&mut ctx(), &src, (n - 1, c), W, H);
        assert_eq!(
            r,
            Err(YuvError::BufferTooSmall {
                buffer: "y",
                needed: n,
                len: n - 1
            }),
            "{e:?}"
        );

        if !matches!(e, Entry::E420YOnly) {
            // cb and cr share a length here, so cb is reported first.
            let (r, planes) = e.run(&mut ctx(), &src, (n, c - 1), W, H);
            assert_eq!(
                r,
                Err(YuvError::BufferTooSmall {
                    buffer: "cb",
                    needed: c,
                    len: c - 1
                }),
                "{e:?}"
            );
            assert!(planes.iter().flatten().all(|&v| v == 7.0), "{e:?} wrote");
        }
    }
}

#[test]
fn the_pixel_limit_is_enforced() {
    assert_eq!(DEFAULT_MAX_PIXELS, 120_000_000);
    let src = rgb(11, 10);
    for e in ENTRIES {
        let mut c = ctx().with_max_pixels(109);
        let (r, planes) = e.run(&mut c, &src, (110, e.chroma(11, 10)), 11, 10);
        assert_eq!(
            r,
            Err(YuvError::TooManyPixels {
                pixels: 110,
                max: 109
            }),
            "{e:?}"
        );
        assert!(planes.iter().flatten().all(|&v| v == 7.0), "{e:?} wrote");
        let mut c = ctx().with_max_pixels(110);
        let lens = (110, e.chroma(11, 10));
        assert_eq!(e.run(&mut c, &src, lens, 11, 10).0, Ok(()), "{e:?}");
    }
    // The default limit refuses dimensions past it before any buffer is
    // looked at.
    let mut c = ctx();
    let (w, h) = (20_000, 6_001);
    let err = c
        .encode_444_u8(
            ImgRef::new(&[][..], w * 3, h),
            ImgRefMut::new(&mut [][..], w, h),
            ImgRefMut::new(&mut [][..], w, h),
            ImgRefMut::new(&mut [][..], w, h),
        )
        .unwrap_err();
    assert_eq!(
        err,
        YuvError::TooManyPixels {
            pixels: 120_020_000,
            max: DEFAULT_MAX_PIXELS
        }
    );
}

#[test]
fn sizes_that_overflow_are_refused() {
    let mut c = ctx().with_max_pixels(u64::MAX);
    // imgref caps a view's width and height at `u32::MAX`, so what overflows now
    // is a stride: `(height - 1) * stride`, on 32- and 64-bit targets alike.
    let (w, h) = (1, 3);
    assert_eq!(
        c.encode_420_u8(
            ImgRef::new_stride(&[0u8; 0][..], w * 3, h, usize::MAX),
            ImgRefMut::new(&mut [][..], w, h),
            ImgRefMut::new(&mut [][..], w, h),
            ImgRefMut::new(&mut [][..], w, h),
        ),
        Err(YuvError::Overflow)
    );
    assert_eq!(
        c.encode_444_u8(
            ImgRef::new(&[0u8; 9][..], w * 3, h),
            ImgRefMut::new_stride(&mut [][..], w, h, usize::MAX),
            ImgRefMut::new(&mut [][..], w, h),
            ImgRefMut::new(&mut [][..], w, h),
        ),
        Err(YuvError::Overflow)
    );
}

#[test]
fn empty_images_need_no_buffers() {
    for e in ENTRIES {
        for (w, h) in [(0, 0), (0, 5), (5, 0)] {
            let (r, _) = e.run(&mut ctx(), &[], (0, 0), w, h);
            assert_eq!(r, Ok(()), "{e:?} {w}x{h}");
        }
    }
}

#[test]
fn errors_display_their_cause() {
    let e = YuvError::BufferTooSmall {
        buffer: "cb",
        needed: 4,
        len: 3,
    };
    assert_eq!(e.to_string(), "cb buffer holds 3 elements, needs 4");
    let e: &dyn core::error::Error = &YuvError::OutOfMemory;
    assert_eq!(e.to_string(), "scratch allocation failed");
}
