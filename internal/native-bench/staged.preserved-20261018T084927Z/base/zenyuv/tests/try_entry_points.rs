//! The `YuvContext::try_*` entry points: the same bytes as the infallible
//! methods on valid input, and a typed refusal (with nothing written) where
//! those would panic.

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

/// One entry point, as the infallible and fallible forms over u8 or f32
/// planes, with its plane sizes.
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

impl Entry {
    fn chroma(self, w: usize, h: usize) -> usize {
        match self {
            Entry::E444 => w * h,
            _ => w.div_ceil(2) * h.div_ceil(2),
        }
    }

    /// Run the entry point on planes of the given lengths, filled with a
    /// sentinel. Returns the result and the planes as f32 (u8 widened).
    fn run(
        self,
        ctx: &mut YuvContext,
        fallible: bool,
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
        let r = match (self, fallible) {
            (Entry::E420, false) => {
                ctx.encode_420_u8(rgb, &mut y8, &mut cb8, &mut cr8, w, h);
                Ok(())
            }
            (Entry::E420, true) => ctx.try_encode_420_u8(rgb, &mut y8, &mut cb8, &mut cr8, w, h),
            (Entry::E420YOnly, false) => {
                ctx.encode_420_y_only_u8(rgb, &mut y8, w, h);
                Ok(())
            }
            (Entry::E420YOnly, true) => ctx.try_encode_420_y_only_u8(rgb, &mut y8, w, h),
            (Entry::E444, false) => {
                ctx.encode_444_u8(rgb, &mut y8, &mut cb8, &mut cr8, w, h);
                Ok(())
            }
            (Entry::E444, true) => ctx.try_encode_444_u8(rgb, &mut y8, &mut cb8, &mut cr8, w, h),
            (Entry::E420F32, false) => {
                ctx.encode_420_f32(rgb, &mut yf, &mut cbf, &mut crf, w, h);
                Ok(())
            }
            (Entry::E420F32, true) => {
                ctx.try_encode_420_f32(rgb, &mut yf, &mut cbf, &mut crf, w, h)
            }
            (Entry::Sharp420, false) => {
                ctx.encode_sharp_420_u8(rgb, &mut y8, &mut cb8, &mut cr8, w, h, &cfg);
                Ok(())
            }
            (Entry::Sharp420, true) => {
                ctx.try_encode_sharp_420_u8(rgb, &mut y8, &mut cb8, &mut cr8, w, h, &cfg)
            }
            (Entry::Sharp420F32, false) => {
                ctx.encode_sharp_420_f32(rgb, &mut yf, &mut cbf, &mut crf, w, h, &cfg);
                Ok(())
            }
            (Entry::Sharp420F32, true) => {
                ctx.try_encode_sharp_420_f32(rgb, &mut yf, &mut cbf, &mut crf, w, h, &cfg)
            }
        };
        let planes = match self {
            Entry::E420F32 | Entry::Sharp420F32 => [yf, cbf, crf],
            _ => [y8, cb8, cr8].map(|p| p.into_iter().map(f32::from).collect()),
        };
        (r, planes)
    }
}

#[test]
fn valid_input_gives_the_infallible_bytes() {
    let src = rgb(W, H);
    for e in ENTRIES {
        let lens = (W * H, e.chroma(W, H));
        let (_, want) = e.run(&mut ctx(), false, &src, lens, W, H);
        let (r, got) = e.run(&mut ctx(), true, &src, lens, W, H);
        assert_eq!(r, Ok(()), "{e:?}");
        assert_eq!(got, want, "{e:?}");
        // A reused context (scratch already sized) gives the same bytes.
        let mut c = ctx();
        e.run(&mut c, true, &src, lens, W, H).0.unwrap();
        assert_eq!(
            e.run(&mut c, true, &src, lens, W, H).1,
            want,
            "{e:?} reused"
        );
    }
}

#[test]
fn each_short_buffer_is_refused_and_nothing_is_written() {
    let src = rgb(W, H);
    for e in ENTRIES {
        let (n, c) = (W * H, e.chroma(W, H));
        let short_rgb = &src[..n * 3 - 1];
        let (r, planes) = e.run(&mut ctx(), true, short_rgb, (n, c), W, H);
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

        let (r, _) = e.run(&mut ctx(), true, &src, (n - 1, c), W, H);
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
            let (r, planes) = e.run(&mut ctx(), true, &src, (n, c - 1), W, H);
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
        let (r, planes) = e.run(&mut c, true, &src, (110, e.chroma(11, 10)), 11, 10);
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
        assert_eq!(e.run(&mut c, true, &src, lens, 11, 10).0, Ok(()), "{e:?}");
    }
    // The default limit refuses dimensions past it before any buffer is
    // looked at.
    let mut c = ctx();
    let err = c
        .try_encode_444_u8(&[], &mut [], &mut [], &mut [], 20_000, 6_001)
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
    // `width × 3` overflows usize on 32- and 64-bit targets alike.
    let w = usize::MAX / 2 + 1;
    assert_eq!(
        c.try_encode_420_u8(&[], &mut [], &mut [], &mut [], w, 1),
        Err(YuvError::Overflow)
    );
    assert_eq!(
        c.try_encode_444_u8(&[], &mut [], &mut [], &mut [], usize::MAX, usize::MAX),
        Err(YuvError::Overflow)
    );
}

#[test]
fn empty_images_need_no_buffers() {
    for e in ENTRIES {
        for (w, h) in [(0, 0), (0, 5), (5, 0)] {
            let (r, _) = e.run(&mut ctx(), true, &[], (0, 0), w, h);
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
