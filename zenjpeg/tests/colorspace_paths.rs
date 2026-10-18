//! Colour model decided once at SOF; RGB-as-RGB pixels on every decode path.
//!
//! zenjpeg follows jpegli's `ProcessSOF` (`internal/jpegli-cpp/lib/jpegli/
//! decode_marker.cc`): for a 3-component frame a JFIF APP0 means YCbCr; else an
//! Adobe APP14 means RGB for transform 0 and YCbCr for any other value; else
//! component IDs `'R','G','B'` mean RGB and anything else YCbCr. The decision is
//! taken when SOF is parsed, so a marker placed after SOF does not change it,
//! and every entry point (`read_info`, `decode`, `decode_into`,
//! `scanline_reader`, owned or borrowed) reports and decodes the same model.
//!
//! The pixel tests reach every path the decoder can route a file to:
//!
//! | path | reached by |
//! |---|---|
//! | streaming baseline | `decode`, threads 1, baseline, standard layout |
//! | fused parallel 4:4:4 / box / h2v1 / fancy | `decode`, threads 0, DRI = 1 MCU row, >= 1024 MCUs (`parallel`) |
//! | wave-parallel scanline box | `scanline_reader`, threads 0, NearestNeighbor, subsampled, DRI (`parallel`) |
//! | fast integer coefficient path (threaded at >= 2,000,000 px under `parallel`) | `decode` / `decode_into` of progressive files |
//! | per-component integer path | CMYK/YCCK and the sampling shapes no fast path serves |
//! | f32 coefficient path | `OutputTarget::SrgbF32` |
//! | streaming / coefficient strip scanline | `scanline_reader` (borrowed or owned) of baseline / progressive or non-interleaved files |
//! | buffered scanline | progressive grayscale, shapes the strip does not serve, transform and deblock fallbacks |
//!
//! Fixtures come from zenjpeg's own encoder (relabelled for subsampled RGB) and
//! from `support/jpeg_writer.rs`, a test-only writer that takes coefficients
//! directly: it produces the layouts zenjpeg's encoder cannot (4:1:1, 4:4:0
//! interleaved with DRI, asymmetric R 2x2 / G 2x2 / B 1x1, fractional 3x1 over
//! 2x1, oversampled grayscale) and DC-only content whose decoded pixels are
//! known exactly. DC-only blocks are the regime real photographs put chroma in
//! (repo CLAUDE.md "flat-chroma coverage"); the tests assert AC == 0 on them.

#[path = "support/jpeg_writer.rs"]
mod jpeg_writer;

#[path = "support/post_scan.rs"]
mod post_scan;
use post_scan::insert_before_second_sos;

use enough::Unstoppable;
use imgref::ImgRefMut;
use jpeg_writer::{Comp, Segment, Spec, app14, dc_block, jfif, resignal};
use std::borrow::Cow;
use zenjpeg::decode::{ChromaUpsampling, CropRegion, Decoder, OutputTarget};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, XybSubsampling};
use zenjpeg::types::{ColorSpace, PixelFormat};

const UPS: [ChromaUpsampling; 2] = [
    ChromaUpsampling::Triangle,
    ChromaUpsampling::NearestNeighbor,
];

// ===========================================================================
// Content
// ===========================================================================

/// xorshift-style hash, deterministic per (seed, a, b, c).
fn hash(seed: u32, a: usize, b: usize, c: usize) -> u32 {
    let mut x = seed
        ^ (a as u32).wrapping_mul(0x9E37_79B1)
        ^ (b as u32).wrapping_mul(0x85EB_CA77)
        ^ (c as u32).wrapping_mul(0xC2B2_AE3D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297A_2D39);
    x ^ (x >> 15)
}

/// Per-block DC value in [24, 231] (DC-only content).
fn dc_value(c: usize, bx: usize, by: usize) -> u8 {
    24 + (hash(0x5EED, c, bx, by) % 208) as u8
}

/// DC-only block content: flat within every block, different per block.
fn dc_only(c: usize, bx: usize, by: usize) -> [i16; 64] {
    dc_block(dc_value(c, bx, by))
}

/// DC plus a few low-frequency AC terms: reaches the full IDCT branches.
fn with_ac(c: usize, bx: usize, by: usize) -> [i16; 64] {
    let mut b = dc_block(48 + (hash(0xACAC, c, bx, by) % 160) as u8);
    for (k, coef) in b.iter_mut().enumerate().take(10).skip(1) {
        let r = hash(0xF00D, c * 64 + k, bx, by) % 9;
        *coef = r as i16 - 4;
    }
    b
}

/// High-frequency noise with saturated patches (never a gradient).
fn noise_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let n = hash(seed, x, y, 0);
            let idx = (y * w + x) * 3;
            if (x / 16 + y / 16) % 3 == 0 {
                rgb[idx] = n as u8;
                rgb[idx + 1] = (n >> 8) as u8;
                rgb[idx + 2] = (n >> 16) as u8;
            } else {
                rgb[idx] = 180u8.wrapping_add((n as u8) / 8);
                rgb[idx + 1] = 30u8.wrapping_add(((n >> 8) as u8) / 8);
                rgb[idx + 2] = 90u8.wrapping_add(((n >> 16) as u8) / 8);
            }
        }
    }
    rgb
}

// ===========================================================================
// Writer-built frames
// ===========================================================================

/// Odd width and height: partial last MCU column and row on every layout.
/// At least 1024 MCUs on every layout below, so DRI = 1 MCU row reaches the
/// fused and wave paths.
const FW: u32 = 1001;
const FH: u32 = 270;

#[derive(Clone, Copy, Debug)]
struct Layout {
    name: &'static str,
    /// (h, v) per component.
    samp: [(u8, u8); 3],
    /// Frame size: odd in both directions (partial last MCU column and row),
    /// and at least 1024 MCUs so DRI = 1 MCU row reaches the fused and wave
    /// gates.
    size: (u32, u32),
}

const LAYOUTS: [Layout; 7] = [
    Layout {
        name: "444",
        samp: [(1, 1), (1, 1), (1, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "420",
        samp: [(2, 2), (1, 1), (1, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "422",
        samp: [(2, 1), (1, 1), (1, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "440",
        samp: [(1, 2), (1, 1), (1, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "411",
        samp: [(4, 1), (1, 1), (1, 1)],
        size: (FW, FH),
    },
    // The XYB "BQuarter" shape, interleaved.
    Layout {
        name: "asym_2x2_2x2_1x1",
        samp: [(2, 2), (2, 2), (1, 1)],
        size: (FW, FH),
    },
    // Chroma sampled above luma.
    Layout {
        name: "chroma_above_1x1_2x2_2x2",
        samp: [(1, 1), (2, 2), (2, 2)],
        size: (FW, FH),
    },
];

/// How a 3-component frame signals its colour model.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Signal {
    /// IDs 'R','G','B' + APP14 transform 0 (what zenjpeg's RGB encoder writes).
    Rgb,
    /// IDs 1,2,3 + JFIF APP0 (unambiguous YCbCr).
    YCbCr,
}

fn frame(
    layout: Layout,
    signal: Signal,
    dri: bool,
    block: &dyn Fn(usize, usize, usize) -> [i16; 64],
) -> Vec<u8> {
    let ids: [u8; 3] = match signal {
        Signal::Rgb => *b"RGB",
        Signal::YCbCr => [1, 2, 3],
    };
    let comps: Vec<Comp> = (0..3)
        .map(|i| Comp {
            id: ids[i],
            h: layout.samp[i].0,
            v: layout.samp[i].1,
        })
        .collect();
    let mut spec = Spec {
        width: layout.size.0,
        height: layout.size.1,
        comps,
        block,
        restart_interval: 0,
        quant: 8,
        segments: vec![match signal {
            Signal::Rgb => Segment::PreFrame(app14(0)),
            Signal::YCbCr => Segment::PreFrame(jfif()),
        }],
    };
    if dri {
        spec.restart_interval = spec.mcu_cols() as u16;
    }
    jpeg_writer::write(&spec)
}

/// The exact image a DC-only RGB frame decodes to with nearest-neighbour
/// (replicating) upsampling: component `c` at pixel (x, y) is the DC of the
/// block holding sample `(x * h_c / max_h, y * v_c / max_v)`.
fn dc_only_expected_nearest(layout: Layout) -> Vec<u8> {
    let max_h = layout.samp.iter().map(|s| s.0 as usize).max().unwrap();
    let max_v = layout.samp.iter().map(|s| s.1 as usize).max().unwrap();
    let (w, h) = (layout.size.0 as usize, layout.size.1 as usize);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let (hc, vc) = (layout.samp[c].0 as usize, layout.samp[c].1 as usize);
                let sx = x * hc / max_h;
                let sy = y * vc / max_v;
                out[(y * w + x) * 3 + c] = dc_value(c, sx / 8, sy / 8);
            }
        }
    }
    out
}

// ===========================================================================
// Encoder-built fixtures
// ===========================================================================

fn encode_rgb(src: &[u8], w: u32, h: u32, dri: bool, progressive: bool) -> Vec<u8> {
    EncoderConfig::rgb(88)
        .progressive(progressive)
        .restart_mcu_rows(if dri { 1 } else { 0 })
        .encode_bytes(src, w, h, PixelLayout::Rgb8Srgb)
        .expect("rgb encode")
}

fn encode_ycbcr(
    src: &[u8],
    w: u32,
    h: u32,
    sub: ChromaSubsampling,
    dri: bool,
    progressive: bool,
) -> Vec<u8> {
    EncoderConfig::ycbcr(88, sub)
        .progressive(progressive)
        .restart_mcu_rows(if dri { 1 } else { 0 })
        .encode_bytes(src, w, h, PixelLayout::Rgb8Srgb)
        .expect("ycbcr encode")
}

fn as_rgb(jpeg: &[u8]) -> Vec<u8> {
    resignal(jpeg, *b"RGB", &[app14(0)], &[])
}

fn as_ycbcr(jpeg: &[u8]) -> Vec<u8> {
    resignal(jpeg, [1, 2, 3], &[jfif()], &[])
}

// ===========================================================================
// Entry points
// ===========================================================================

fn decoder(ups: ChromaUpsampling, threads: usize) -> Decoder {
    Decoder::new().chroma_upsampling(ups).num_threads(threads)
}

fn decode_fmt(jpeg: &[u8], ups: ChromaUpsampling, threads: usize, fmt: PixelFormat) -> Vec<u8> {
    decoder(ups, threads)
        .output_format(fmt)
        .decode(jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("decode {fmt:?}: {e}"))
        .into_pixels_u8()
        .expect("u8 pixels")
}

fn decode_rgb(jpeg: &[u8], ups: ChromaUpsampling, threads: usize) -> Vec<u8> {
    decode_fmt(jpeg, ups, threads, PixelFormat::Rgb)
}

fn decode_into_fmt(
    jpeg: &[u8],
    ups: ChromaUpsampling,
    threads: usize,
    fmt: PixelFormat,
) -> Vec<u8> {
    let info = Decoder::new().read_info(jpeg).expect("read_info");
    let n = info.dimensions.width as usize * info.dimensions.height as usize;
    let mut dst = vec![0u8; n * fmt.bytes_per_pixel()];
    let written = decoder(ups, threads)
        .decode_into(jpeg, fmt, &mut dst, Unstoppable)
        .unwrap_or_else(|e| panic!("decode_into {fmt:?}: {e}"));
    assert_eq!(written, dst.len());
    dst
}

/// RGB-family f32 output through the coefficient path, quantized back to u8.
fn decode_f32_as_u8(jpeg: &[u8], ups: ChromaUpsampling) -> Vec<u8> {
    let px = Decoder::new()
        .chroma_upsampling(ups)
        .output_target(OutputTarget::SrgbF32)
        .decode(jpeg, Unstoppable)
        .expect("f32 decode")
        .into_pixels_f32()
        .expect("f32 pixels");
    px.iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Read {
    Rgb8,
    Bgr8,
    Rgbx8,
    Bgrx8,
}

impl Read {
    fn bpp(self) -> usize {
        match self {
            Read::Rgb8 | Read::Bgr8 => 3,
            Read::Rgbx8 | Read::Bgrx8 => 4,
        }
    }
}

/// Read the whole image through a scanline reader in odd-sized row batches.
fn scanline(
    jpeg: &[u8],
    ups: ChromaUpsampling,
    threads: usize,
    owned: bool,
    read: Read,
) -> Vec<u8> {
    let dec = decoder(ups, threads);
    let mut r = if owned {
        dec.scanline_reader_cow(Cow::Owned(jpeg.to_vec()))
    } else {
        dec.scanline_reader(jpeg)
    }
    .unwrap_or_else(|e| panic!("scanline_reader: {e}"));
    let (w, h) = (r.width() as usize, r.height() as usize);
    let stride = w * read.bpp();
    let mut out = vec![0u8; stride * h];
    let mut row = 0;
    while row < h {
        let n = 7.min(h - row);
        let img = ImgRefMut::new(&mut out[row * stride..(row + n) * stride], stride, n);
        let got = match read {
            Read::Rgb8 => r.read_rows_rgb8(img),
            Read::Bgr8 => r.read_rows_bgr8(img),
            Read::Rgbx8 => r.read_rows_rgbx8(img),
            Read::Bgrx8 => r.read_rows_bgrx8(img),
        }
        .unwrap_or_else(|e| panic!("read {read:?}: {e}"));
        assert!(got > 0, "reader stalled at row {row}");
        row += got;
    }
    out
}

/// Convert a 3bpp/4bpp output back to RGB 3bpp for comparison (gray is kept
/// as one byte per pixel).
fn to_rgb(px: &[u8], bpp: usize, bgr: bool) -> Vec<u8> {
    if bpp == 1 {
        // Gray stays one byte per pixel.
        return px.to_vec();
    }
    px.chunks_exact(bpp)
        .flat_map(|p| {
            if bgr {
                [p[2], p[1], p[0]]
            } else {
                [p[0], p[1], p[2]]
            }
        })
        .collect()
}

/// One entry point's output, normalized to RGB 3bpp, plus whether a 4-byte
/// layout's fourth byte was 255 everywhere — or the error / panic message.
struct Entry {
    name: String,
    rgb: Result<Vec<u8>, String>,
    alpha_ok: bool,
}

impl Entry {
    /// Run one entry point, catching errors and panics so a broken path is
    /// one collected failure instead of the end of the test.
    fn run(name: String, bpp: usize, bgr: bool, f: impl FnOnce() -> Vec<u8>) -> Self {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(px) => Self {
                name,
                alpha_ok: bpp != 4 || px.as_chunks::<4>().0.iter().all(|p| p[3] == 255),
                rgb: Ok(to_rgb(&px, bpp, bgr)),
            },
            Err(e) => Self {
                name,
                alpha_ok: true,
                rgb: Err(e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default()),
            },
        }
    }
}

/// Max channel difference and first differing pixel.
fn diff(a: &[u8], b: &[u8]) -> (u8, Option<usize>) {
    assert_eq!(a.len(), b.len(), "length");
    let mut max = 0u8;
    let mut first = None;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = x.abs_diff(y);
        if d > 0 && first.is_none() {
            first = Some(i);
        }
        max = max.max(d);
    }
    (max, first)
}

/// Every u8 RGB-family entry point for one file, labelled; the caller
/// compares each against a reference. `threads` 0 reaches the parallel paths
/// when built with `parallel`.
fn every_entry(jpeg: &[u8], ups: ChromaUpsampling) -> Vec<Entry> {
    let mut v = Vec::new();
    let bgr = |fmt: PixelFormat| {
        matches!(
            fmt,
            PixelFormat::Bgr | PixelFormat::Bgra | PixelFormat::Bgrx
        )
    };
    for threads in [1usize, 0] {
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Bgr,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
            PixelFormat::Bgrx,
        ] {
            v.push(Entry::run(
                format!("decode {fmt:?} t{threads}"),
                fmt.bytes_per_pixel(),
                bgr(fmt),
                || decode_fmt(jpeg, ups, threads, fmt),
            ));
            v.push(Entry::run(
                format!("decode_into {fmt:?} t{threads}"),
                fmt.bytes_per_pixel(),
                bgr(fmt),
                || decode_into_fmt(jpeg, ups, threads, fmt),
            ));
        }
        for owned in [false, true] {
            for read in [Read::Rgb8, Read::Bgr8, Read::Rgbx8, Read::Bgrx8] {
                v.push(Entry::run(
                    format!("scanline {read:?} t{threads} owned={owned}"),
                    read.bpp(),
                    matches!(read, Read::Bgr8 | Read::Bgrx8),
                    || scanline(jpeg, ups, threads, owned, read),
                ));
            }
        }
    }
    v
}

/// Collects failures instead of stopping at the first, so one run shows every
/// broken path.
#[derive(Default)]
struct Failures(Vec<String>);

impl Failures {
    fn check(&mut self, ok: bool, msg: impl FnOnce() -> String) {
        if !ok {
            self.0.push(msg());
        }
    }
    fn finish(self, what: &str) {
        if !self.0.is_empty() {
            panic!(
                "{what}: {} failure(s):\n  {}",
                self.0.len(),
                self.0.join("\n  ")
            );
        }
    }
}

fn compare_all(f: &mut Failures, label: &str, reference: &[u8], got: &[Entry]) {
    for e in got {
        match &e.rgb {
            Ok(rgb) => {
                let (max, first) = diff(reference, rgb);
                f.check(max == 0, || {
                    format!(
                        "{label}: {} differs from reference by {max} (first byte {first:?})",
                        e.name
                    )
                });
            }
            Err(msg) => f.check(false, || format!("{label}: {} failed: {msg}", e.name)),
        }
        f.check(e.alpha_ok, || {
            format!("{label}: {}: fourth byte != 255", e.name)
        });
    }
}

// ===========================================================================
// 1. The colour-model decision, at every entry point
// ===========================================================================

/// One flat colour so every path, upsampler and IDCT gives one exact value.
const SOLID: [u8; 3] = [200, 60, 30];

fn solid_frame(ids: [u8; 3], segments: Vec<Segment>, dri: bool) -> Vec<u8> {
    let block = |c: usize, _: usize, _: usize| dc_block(SOLID[c]);
    let comps = (0..3)
        .map(|i| Comp {
            id: ids[i],
            h: if i == 0 { 2 } else { 1 },
            v: if i == 0 { 2 } else { 1 },
        })
        .collect();
    let mut spec = Spec {
        width: 520,
        height: 520,
        comps,
        block: &block,
        restart_interval: 0,
        quant: 8,
        segments,
    };
    if dri {
        spec.restart_interval = spec.mcu_cols() as u16;
    }
    jpeg_writer::write(&spec)
}

#[test]
fn colour_model_is_decided_at_sof_for_every_entry_point() {
    use Segment::{PostFrame, PreFrame};
    // (name, ids, segments, expected model). Expectations follow jpegli
    // `ProcessSOF`; libjpeg-turbo agrees on all but the post-SOF rows (it
    // decides at the first SOS).
    let cases: Vec<(&str, [u8; 3], Vec<Segment>, ColorSpace)> = vec![
        ("ids RGB, no markers", *b"RGB", vec![], ColorSpace::Rgb),
        (
            "ids RGB + APP14 t0",
            *b"RGB",
            vec![PreFrame(app14(0))],
            ColorSpace::Rgb,
        ),
        (
            "ids 123 + APP14 t0",
            [1, 2, 3],
            vec![PreFrame(app14(0))],
            ColorSpace::Rgb,
        ),
        (
            "ids RGB + APP14 t1",
            *b"RGB",
            vec![PreFrame(app14(1))],
            ColorSpace::YCbCr,
        ),
        (
            "ids 123 + APP14 t3",
            [1, 2, 3],
            vec![PreFrame(app14(3))],
            ColorSpace::YCbCr,
        ),
        (
            "ids RGB + APP14 t255",
            *b"RGB",
            vec![PreFrame(app14(255))],
            ColorSpace::YCbCr,
        ),
        (
            "ids 123 + JFIF + APP14 t0",
            [1, 2, 3],
            vec![PreFrame(jfif()), PreFrame(app14(0))],
            ColorSpace::YCbCr,
        ),
        (
            "ids RGB + JFIF",
            *b"RGB",
            vec![PreFrame(jfif())],
            ColorSpace::YCbCr,
        ),
        (
            "ids 123 + APP14 t0 after SOF",
            [1, 2, 3],
            vec![PostFrame(app14(0))],
            ColorSpace::YCbCr,
        ),
        (
            "ids RGB + APP14 t1 after SOF",
            *b"RGB",
            vec![PostFrame(app14(1))],
            ColorSpace::Rgb,
        ),
        (
            "ids 123 + APP14 t1 then t0",
            [1, 2, 3],
            vec![PreFrame(app14(1)), PreFrame(app14(0))],
            ColorSpace::Rgb,
        ),
        ("ids 123, no markers", [1, 2, 3], vec![], ColorSpace::YCbCr),
        ("ids rgb (lowercase)", *b"rgb", vec![], ColorSpace::YCbCr),
    ];

    let mut f = Failures::default();
    // Reference pixels for the YCbCr interpretation: the unambiguous YCbCr
    // labelling of the same planes.
    let ycc_ref = decode_rgb(
        &solid_frame([1, 2, 3], vec![PreFrame(jfif())], false),
        ChromaUpsampling::Triangle,
        1,
    );
    let rgb_ref: Vec<u8> = (0..520 * 520).flat_map(|_| SOLID).collect();

    for (name, ids, segs, want) in cases {
        for dri in [false, true] {
            let jpeg = solid_frame(ids, segs.clone(), dri);
            let label = format!("{name} dri={dri}");
            let info = Decoder::new().read_info(&jpeg).expect("read_info");
            f.check(info.color_space == want, || {
                format!(
                    "{label}: read_info reports {:?}, want {want:?}",
                    info.color_space
                )
            });
            for owned in [false, true] {
                let dec = Decoder::new();
                let r = if owned {
                    dec.scanline_reader_cow(Cow::Owned(jpeg.clone()))
                } else {
                    dec.scanline_reader(&jpeg)
                }
                .expect("scanline_reader");
                let cs = r.info().color_space;
                f.check(cs == want, || {
                    format!("{label}: ScanlineReader(owned={owned}).info() reports {cs:?}, want {want:?}")
                });
            }
            let reference = if want == ColorSpace::Rgb {
                &rgb_ref
            } else {
                &ycc_ref
            };
            for ups in UPS {
                compare_all(
                    &mut f,
                    &format!("{label} {ups:?}"),
                    reference,
                    &every_entry(&jpeg, ups),
                );
                let f32px = decode_f32_as_u8(&jpeg, ups);
                let (max, _) = diff(reference, &f32px);
                // The f32 path's YCbCr matrix differs from the integer kernels
                // by rounding; the RGB identity is exact.
                let tol = if want == ColorSpace::Rgb { 0 } else { 1 };
                f.check(max <= tol, || {
                    format!("{label} {ups:?}: f32 path differs by {max}")
                });
            }
        }
    }
    f.finish("colour-model decision");
}

/// R3: an APP14 between SOF and SOS in a progressive file (encoder-built), at
/// every entry point. The decision is the SOF-time one (YCbCr for IDs 1,2,3),
/// and every entry point must agree with `decode()`.
#[test]
fn app14_after_sof_progressive_is_consistent() {
    let (w, h) = (200u32, 120u32);
    let src = noise_image(w as usize, h as usize, 7);
    let base = encode_ycbcr(&src, w, h, ChromaSubsampling::Quarter, false, true);
    let mut f = Failures::default();
    for (ids, post, want) in [
        ([1u8, 2, 3], app14(0), ColorSpace::YCbCr),
        (*b"RGB", app14(1), ColorSpace::Rgb),
    ] {
        let jpeg = resignal(&base, ids, &[], &[post]);
        let label = format!("ids {ids:?} APP14-after-SOF");
        let info = Decoder::new().read_info(&jpeg).expect("read_info");
        f.check(info.color_space == want, || {
            format!("{label}: read_info {:?}, want {want:?}", info.color_space)
        });
        let twin = if want == ColorSpace::Rgb {
            as_rgb(&base)
        } else {
            as_ycbcr(&base)
        };
        for ups in UPS {
            let reference = decode_rgb(&twin, ups, 1);
            compare_all(
                &mut f,
                &format!("{label} {ups:?}"),
                &reference,
                &every_entry(&jpeg, ups),
            );
        }
    }
    f.finish("APP14 after SOF");
}

// ===========================================================================
// 2. Known answers: DC-only RGB frames, nearest-neighbour, every layout
// ===========================================================================

#[test]
fn rgb_dc_only_known_answer_every_layout_and_path() {
    let mut f = Failures::default();
    for layout in LAYOUTS {
        for dri in [false, true] {
            let jpeg = frame(layout, Signal::Rgb, dri, &dc_only);
            let label = format!("{} dri={dri}", layout.name);

            // The content really is DC-only (flat-chroma regime).
            let coeffs = Decoder::new()
                .decode_coefficients(&jpeg, Unstoppable)
                .expect("coefficients");
            for (ci, comp) in coeffs.components.iter().enumerate() {
                let ac_nonzero = comp
                    .coeffs
                    .as_chunks::<64>()
                    .0
                    .iter()
                    .filter(|b| b[1..].iter().any(|&c| c != 0))
                    .count();
                f.check(ac_nonzero == 0, || {
                    format!("{label}: component {ci} has {ac_nonzero} blocks with AC != 0")
                });
            }

            let expected = dc_only_expected_nearest(layout);
            compare_all(
                &mut f,
                &label,
                &expected,
                &every_entry(&jpeg, ChromaUpsampling::NearestNeighbor),
            );
            let f32px = decode_f32_as_u8(&jpeg, ChromaUpsampling::NearestNeighbor);
            let (max, first) = diff(&expected, &f32px);
            f.check(max == 0, || {
                format!("{label}: f32 path differs by {max} at {first:?}")
            });
        }
    }
    f.finish("DC-only RGB known answer");
}

/// R9: a solid colour over >= 1024 MCUs with MCU-row DRI (the fused paths
/// with `parallel`), RGB signalled: every pixel is exactly the colour on both
/// upsamplers, so a YCbCr matrix anywhere shows up as a large error.
#[test]
fn rgb_solid_known_answer_fused_geometry() {
    let mut f = Failures::default();
    let rgb_ref: Vec<u8> = (0..520 * 520).flat_map(|_| SOLID).collect();
    for dri in [false, true] {
        let jpeg = solid_frame(*b"RGB", vec![Segment::PreFrame(app14(0))], dri);
        for ups in UPS {
            compare_all(
                &mut f,
                &format!("solid dri={dri} {ups:?}"),
                &rgb_ref,
                &every_entry(&jpeg, ups),
            );
        }
    }
    f.finish("solid RGB known answer");
}

// ===========================================================================
// 3. Every path agrees: RGB and YCbCr, every layout, both upsamplers
// ===========================================================================

/// Writer frames with AC content: every entry point byte-identical to
/// `decode()` threads=1, per upsampler, for both colour models. Covers the
/// layouts no fast path should take (4:1:1, asymmetric, chroma above luma):
/// those must route to the per-component path on every entry point.
///
/// Both chroma regimes (repo CLAUDE.md): AC content, and DC-only blocks that
/// differ from block to block, so the triangle kernels these baseline
/// interleaved frames reach (streaming h2v1 and fancy, fused h2v1 / fancy and
/// their boundary fix-ups, the streaming scanline strip, the per-component
/// path) see flat chroma next to MCU boundaries. The fast integer upsamplers
/// and the coefficient strip need progressive or non-interleaved files;
/// `flat_chroma_progressive_every_path` gives them flat chroma.
#[test]
fn every_path_agrees_every_layout() {
    let mut f = Failures::default();
    for layout in LAYOUTS {
        for (content, block) in CONTENTS {
            for signal in [Signal::Rgb, Signal::YCbCr] {
                for dri in [false, true] {
                    let jpeg = frame(layout, signal, dri, block);
                    for ups in UPS {
                        let label =
                            format!("{} {content} {signal:?} dri={dri} {ups:?}", layout.name);
                        let reference = decode_rgb(&jpeg, ups, 1);
                        compare_all(&mut f, &label, &reference, &every_entry(&jpeg, ups));
                    }
                }
            }
        }
    }
    f.finish("every path agrees");
}

/// The two content regimes every path sweep covers.
const CONTENTS: [(&str, &dyn Fn(usize, usize, usize) -> [i16; 64]); 2] =
    [("ac", &with_ac), ("dc-only", &dc_only)];

/// An RGB-labelled file decodes to exactly the component planes: the RGB
/// decode of a relabelled YCbCr encode equals that encode's own Y/Cb/Cr planes
/// (scanline `read_rows_ycbcr_f32`, same upsampler and IDCT). A cross-model
/// check: both sides are zenjpeg's output, so it ties the RGB model's paths to
/// the YCbCr model's plane reads, not to an external reference (the DC-only
/// known answers are the external anchors here).
#[test]
fn rgb_relabel_equals_component_planes() {
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    let src = noise_image(w as usize, h as usize, 3);
    for sub in [
        ChromaSubsampling::None,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::HalfVertical,
    ] {
        for progressive in [false, true] {
            let ycc = as_ycbcr(&encode_ycbcr(&src, w, h, sub, false, progressive));
            let rgb = as_rgb(&ycc);
            for ups in UPS {
                let label = format!("{sub:?} progressive={progressive} {ups:?}");
                let planes = ycbcr_planes_u8(&ycc, ups);
                compare_all(&mut f, &label, &planes, &every_entry(&rgb, ups));
            }
        }
    }
    f.finish("RGB relabel == planes");
}

/// The Y/Cb/Cr planes of a YCbCr file as interleaved u8, via the scanline
/// `read_rows_ycbcr_f32` read (Y in [0,1], Cb/Cr centred).
fn ycbcr_planes_u8(jpeg: &[u8], ups: ChromaUpsampling) -> Vec<u8> {
    let mut r = decoder(ups, 1).scanline_reader(jpeg).expect("reader");
    let (w, h) = (r.width() as usize, r.height() as usize);
    let (mut y, mut cb, mut cr) = (vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]);
    let mut row = 0;
    while row < h {
        let n = r
            .read_rows_ycbcr_f32(
                &mut y[row * w..],
                &mut cb[row * w..],
                &mut cr[row * w..],
                w,
                h - row,
            )
            .expect("ycbcr_f32");
        row += n;
    }
    let q = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    (0..w * h)
        .flat_map(|i| [q(y[i]), q(cb[i] + 128.0 / 255.0), q(cr[i] + 128.0 / 255.0)])
        .collect()
}

/// Encoder-built RGB fixtures: 4:4:4 from `EncoderConfig::rgb`, subsampled by
/// relabelling, baseline +- DRI and progressive, odd sizes.
#[test]
fn encoder_rgb_fixtures_every_path_agrees() {
    let mut f = Failures::default();
    let (w, h) = (1001u32, 515u32);
    let src = noise_image(w as usize, h as usize, 11);
    let mut fixtures: Vec<(String, Vec<u8>)> = Vec::new();
    for dri in [false, true] {
        for progressive in [false, true] {
            fixtures.push((
                format!("rgb444 dri={dri} prog={progressive}"),
                encode_rgb(&src, w, h, dri, progressive),
            ));
            for sub in [
                ChromaSubsampling::Quarter,
                ChromaSubsampling::HalfHorizontal,
            ] {
                fixtures.push((
                    format!("{sub:?} dri={dri} prog={progressive}"),
                    as_rgb(&encode_ycbcr(&src, w, h, sub, dri, progressive)),
                ));
            }
        }
    }
    for (name, jpeg) in &fixtures {
        let info = Decoder::new().read_info(jpeg).expect("info");
        f.check(info.color_space == ColorSpace::Rgb, || {
            format!("{name}: read_info {:?}", info.color_space)
        });
        for ups in UPS {
            let reference = decode_rgb(jpeg, ups, 1);
            compare_all(
                &mut f,
                &format!("{name} {ups:?}"),
                &reference,
                &every_entry(jpeg, ups),
            );
        }
    }
    f.finish("encoder RGB fixtures");
}

// ===========================================================================
// 4. Shapes that must never reach a kernel built for another shape
// ===========================================================================

/// R1 / R2 / pre-existing: fractional ratios (3x1 luma over 2x1 chroma), and
/// oversampled grayscale, with MCU-row DRI over >= 1024 MCUs. No panic, and
/// the parallel entry points agree with the sequential ones.
#[test]
fn crafted_sampling_shapes_never_panic_and_agree() {
    let mut f = Failures::default();
    let shapes: [(&str, Vec<(u8, u8)>); 4] = [
        ("frac_3x1_2x1_2x1", vec![(3, 1), (2, 1), (2, 1)]),
        ("frac_1x3_1x2_1x2", vec![(1, 3), (1, 2), (1, 2)]),
        ("frac_4x1_3x1_3x1", vec![(4, 1), (3, 1), (3, 1)]),
        ("gray_2x2", vec![(2, 2)]),
    ];
    for (name, samp) in shapes {
        for signal in [Signal::Rgb, Signal::YCbCr] {
            if samp.len() == 1 && signal == Signal::Rgb {
                continue;
            }
            let ids: Vec<u8> = match (samp.len(), signal) {
                (1, _) => vec![1],
                (_, Signal::Rgb) => b"RGB".to_vec(),
                _ => vec![1, 2, 3],
            };
            let comps = samp
                .iter()
                .zip(&ids)
                .map(|(&(h, v), &id)| Comp { id, h, v })
                .collect();
            let mut spec = Spec {
                width: FW,
                height: FH,
                comps,
                block: &with_ac,
                restart_interval: 0,
                quant: 8,
                segments: vec![],
            };
            spec.restart_interval = spec.mcu_cols() as u16;
            let jpeg = jpeg_writer::write(&spec);
            for ups in UPS {
                let label = format!("{name} {signal:?} {ups:?}");
                let run = std::panic::catch_unwind(|| {
                    let reference = decode_rgb(&jpeg, ups, 1);
                    (reference.clone(), every_entry(&jpeg, ups))
                });
                match run {
                    Ok((reference, all)) => compare_all(&mut f, &label, &reference, &all),
                    Err(e) => f.check(false, || {
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        format!("{label}: panicked: {msg}")
                    }),
                }
            }
        }
    }
    f.finish("crafted sampling shapes");
}

// ===========================================================================
// 5. Grayscale frames: the Gray model on every entry point
// ===========================================================================

/// One-component frames (DC-only so every pixel is known; with AC for the
/// IDCT branches): 1x1 with and without MCU-row DRI (fused gray kernel under
/// `parallel`), oversampled 2x2 (a non-interleaved scan, decoded on the
/// per-component path), and a progressive twin. `PixelFormat::Gray` from
/// `decode` and `decode_into`, scanline `gray8`, and every RGB-family entry
/// point replicating the gray value. (`decode_into(Gray)` failed on the
/// progressive and oversampled frames while `decode()` succeeded.)
#[test]
fn gray_frames_every_entry_point() {
    let mut f = Failures::default();
    for (content, block) in CONTENTS {
        for (variant, samp, dri, progressive) in [
            ("1x1", (1u8, 1u8), false, false),
            ("1x1", (1, 1), true, false),
            ("2x2", (2, 2), false, false),
            ("2x2", (2, 2), true, false),
            ("1x1", (1, 1), false, true),
        ] {
            let mut spec = Spec {
                width: FW,
                height: FH,
                comps: vec![Comp {
                    id: 1,
                    h: samp.0,
                    v: samp.1,
                }],
                block,
                restart_interval: 0,
                quant: 8,
                segments: vec![],
            };
            if dri {
                spec.restart_interval = spec.mcu_cols() as u16;
            }
            let baseline = jpeg_writer::write(&spec);
            let jpeg = if progressive {
                progressive_twin(&baseline)
            } else {
                baseline
            };
            let label = format!("gray {variant} {content} dri={dri} progressive={progressive}");
            let gray = decode_fmt(&jpeg, ChromaUpsampling::Triangle, 1, PixelFormat::Gray);
            if content == "dc-only" {
                let (w, h) = (FW as usize, FH as usize);
                let want: Vec<u8> = (0..w * h)
                    .map(|i| dc_value(0, i % w / 8, i / w / 8))
                    .collect();
                let (max, first) = diff(&want, &gray);
                f.check(max == 0, || {
                    format!(
                        "{label}: decode Gray differs from the known answer by {max} at {first:?}"
                    )
                });
            }
            let mut gray_entries = Vec::new();
            for threads in [1usize, 0] {
                gray_entries.push(Entry::run(
                    format!("decode Gray t{threads}"),
                    1,
                    false,
                    || {
                        decode_fmt(
                            &jpeg,
                            ChromaUpsampling::Triangle,
                            threads,
                            PixelFormat::Gray,
                        )
                    },
                ));
                gray_entries.push(Entry::run(
                    format!("decode_into Gray t{threads}"),
                    1,
                    false,
                    || {
                        decode_into_fmt(
                            &jpeg,
                            ChromaUpsampling::Triangle,
                            threads,
                            PixelFormat::Gray,
                        )
                    },
                ));
                for owned in [false, true] {
                    gray_entries.push(Entry::run(
                        format!("scanline gray8 t{threads} owned={owned}"),
                        1,
                        false,
                        || {
                            let dec = decoder(ChromaUpsampling::Triangle, threads);
                            let mut r = if owned {
                                dec.scanline_reader_cow(Cow::Owned(jpeg.clone()))
                            } else {
                                dec.scanline_reader(&jpeg)
                            }
                            .expect("reader");
                            let (w, h) = (r.width() as usize, r.height() as usize);
                            let mut out = vec![0u8; w * h];
                            let mut row = 0;
                            while row < h {
                                row += r
                                    .read_rows_gray8(ImgRefMut::new(
                                        &mut out[row * w..],
                                        w,
                                        h - row,
                                    ))
                                    .expect("gray8");
                            }
                            out
                        },
                    ));
                }
            }
            compare_all(&mut f, &label, &gray, &gray_entries);
            for owned in [false, true] {
                check_gray_f32_reads(
                    &mut f,
                    &format!("{label} owned={owned}"),
                    &jpeg,
                    &gray,
                    owned,
                );
            }
            let replicated: Vec<u8> = gray.iter().flat_map(|&v| [v, v, v]).collect();
            for ups in UPS {
                compare_all(
                    &mut f,
                    &format!("{label} {ups:?}"),
                    &replicated,
                    &every_entry(&jpeg, ups),
                );
            }
        }
    }
    f.finish("gray frames");
}

// ===========================================================================
// 6. XYB stays on the XYB conversion on every path
// ===========================================================================

#[test]
fn xyb_every_path_agrees_and_reports_xyb() {
    let mut f = Failures::default();
    let (w, h) = (640u32, 480u32);
    let src = noise_image(w as usize, h as usize, 5);
    for sub in [XybSubsampling::Full, XybSubsampling::BQuarter] {
        for dri in [false, true] {
            let jpeg = EncoderConfig::xyb(88, sub)
                .restart_mcu_rows(if dri { 1 } else { 0 })
                .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
                .expect("xyb encode");
            let label = format!("xyb {sub:?} dri={dri}");
            let info = Decoder::new().read_info(&jpeg).expect("info");
            f.check(info.color_space == ColorSpace::Xyb && info.is_xyb, || {
                format!(
                    "{label}: read_info {:?} is_xyb={}",
                    info.color_space, info.is_xyb
                )
            });
            let r = Decoder::new().scanline_reader(&jpeg).expect("reader");
            let si = r.info();
            f.check(si.color_space == ColorSpace::Xyb && si.is_xyb, || {
                format!(
                    "{label}: ScanlineReader::info {:?} is_xyb={}",
                    si.color_space, si.is_xyb
                )
            });
            for ups in UPS {
                let reference = decode_rgb(&jpeg, ups, 1);
                compare_all(
                    &mut f,
                    &format!("{label} {ups:?}"),
                    &reference,
                    &every_entry(&jpeg, ups),
                );
            }
            // Native i16 planes have no XYB meaning: the error says XYB.
            let mut r = Decoder::new().scanline_reader(&jpeg).expect("reader");
            let cw = r.chroma_width() as usize;
            let (mut y, mut cb, mut cr) = (
                vec![0i16; w as usize * 32],
                vec![0i16; cw * 32],
                vec![0i16; cw * 32],
            );
            match r.read_rows_ycbcr_native_i16(&mut y, w as usize, &mut cb, &mut cr, cw, 1) {
                Ok(_) => f.check(false, || {
                    format!("{label}: native i16 read succeeded on XYB")
                }),
                Err(e) => f.check(e.to_string().contains("XYB"), || {
                    format!("{label}: native i16 error does not name XYB: {e}")
                }),
            }
        }
    }
    f.finish("XYB paths");
}

// ===========================================================================
// 7. Derived scanline reads on RGB files
// ===========================================================================

/// gray8 / gray_f32 / gray_linear_f32 / ycbcr_f32 / rgba_f32 on RGB files: the streaming
/// (baseline), coefficient (progressive, borrowed and owned) and buffered
/// (a 4:1:1 frame, which the strip does not serve, so the reader holds decoded
/// RGB) readers agree bit-for-bit, and equal one formula set applied to the
/// decoded RGB: libjpeg-turbo's `rgb_gray_convert` luma
/// `(19595 R + 38470 G + 7471 B + 32768) >> 16` for gray8, the BT.601 matrix
/// (`foundation::consts::YCBCR_*`) in f32 for gray_f32 / ycbcr_f32, and
/// linearize-then-weight with the exact u8 sRGB LUT for gray_linear_f32, and
/// the same LUT per channel (alpha 1) for rgba_f32.
#[test]
fn rgb_derived_reads_agree_across_reader_modes() {
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    let src = noise_image(w as usize, h as usize, 9);
    for sub in [
        ChromaSubsampling::None,
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
    ] {
        let base = as_rgb(&encode_ycbcr(&src, w, h, sub, false, false));
        let prog = as_rgb(&encode_ycbcr(&src, w, h, sub, false, true));
        for ups in UPS {
            for (mode, jpeg, owned) in [
                ("streaming", &base, false),
                ("coefficient", &prog, false),
                ("coefficient owned", &prog, true),
            ] {
                check_rgb_derived(&mut f, &format!("{sub:?} {ups:?} {mode}"), jpeg, ups, owned);
            }
        }
        // Native i16 planes have no RGB meaning: the error says RGB.
        let mut r = Decoder::new().scanline_reader(&base).expect("reader");
        let cw = r.chroma_width() as usize;
        let (mut y, mut cb, mut cr) = (
            vec![0i16; w as usize * 32],
            vec![0i16; cw * 32],
            vec![0i16; cw * 32],
        );
        match r.read_rows_ycbcr_native_i16(&mut y, w as usize, &mut cb, &mut cr, cw, 1) {
            Ok(_) => f.check(false, || {
                format!("{sub:?}: native i16 read succeeded on RGB")
            }),
            Err(e) => f.check(e.to_string().contains("RGB"), || {
                format!("{sub:?}: native i16 error does not name RGB: {e}")
            }),
        }
    }
    // Buffered mode: the strip does not serve 4:1:1, so the reader decodes
    // to RGB up front and derives every read from it (`RowSrc::Rgb8`).
    let buffered = frame(LAYOUTS[4], Signal::Rgb, false, &with_ac);
    for ups in UPS {
        for owned in [false, true] {
            check_rgb_derived(
                &mut f,
                &format!("4:1:1 {ups:?} buffered owned={owned}"),
                &buffered,
                ups,
                owned,
            );
        }
    }
    f.finish("RGB derived reads");
}

/// The RGB-derived reads of one reader equal the formula set applied to the
/// same reader's `read_rows_rgb8` (see the test above).
fn check_rgb_derived(
    f: &mut Failures,
    label: &str,
    jpeg: &[u8],
    ups: ChromaUpsampling,
    owned: bool,
) {
    let rgb = scanline(jpeg, ups, 1, owned, Read::Rgb8);
    let d = derived_reads(jpeg, ups, owned);
    let exp_gray: Vec<u8> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            ((19595 * p[0] as u32 + 38470 * p[1] as u32 + 7471 * p[2] as u32 + 32768) >> 16) as u8
        })
        .collect();
    let (max, first) = diff(&exp_gray, &d.gray8);
    f.check(max == 0, || {
        format!("{label}: gray8 differs by {max} at {first:?}")
    });
    for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let gf = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        let lin = |v: u8| linear_srgb::default::srgb_u8_to_linear(v);
        let gl = 0.299 * lin(p[0]) + 0.587 * lin(p[1]) + 0.114 * lin(p[2]);
        let yy = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        let cbv = (-0.168_736 * r - 0.331_264 * g + 0.5 * b) / 255.0;
        let crv = (0.5 * r - 0.418_688 * g - 0.081_312 * b) / 255.0;
        let rgba = [lin(p[0]), lin(p[1]), lin(p[2]), 1.0];
        let got_rgba = &d.rgba[i * 4..i * 4 + 4];
        let ok = d.gray_f32[i].to_bits() == gf.to_bits()
            && d.gray_lin[i].to_bits() == gl.to_bits()
            && d.y[i].to_bits() == yy.to_bits()
            && d.cb[i].to_bits() == cbv.to_bits()
            && d.cr[i].to_bits() == crv.to_bits()
            && got_rgba
                .iter()
                .zip(&rgba)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !ok {
            f.check(false, || {
                format!(
                    "{label}: pixel {i}: gray_f32 {} vs {gf}, gray_linear {} vs {gl}, ycbcr ({},{},{}) vs ({yy},{cbv},{crv}), rgba_f32 {got_rgba:?} vs {rgba:?}",
                    d.gray_f32[i], d.gray_lin[i], d.y[i], d.cb[i], d.cr[i]
                )
            });
            break;
        }
    }
}

struct Derived {
    gray8: Vec<u8>,
    gray_f32: Vec<f32>,
    gray_lin: Vec<f32>,
    y: Vec<f32>,
    cb: Vec<f32>,
    cr: Vec<f32>,
    /// Linear RGBA (`read_rows_rgba_f32`).
    rgba: Vec<f32>,
}

fn derived_reads(jpeg: &[u8], ups: ChromaUpsampling, owned: bool) -> Derived {
    let reader = || {
        let dec = decoder(ups, 1);
        if owned {
            dec.scanline_reader_cow(Cow::Owned(jpeg.to_vec()))
        } else {
            dec.scanline_reader(jpeg)
        }
        .expect("reader")
    };
    let mut r = reader();
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut gray8 = vec![0u8; w * h];
    r.read_rows_gray8(ImgRefMut::new(&mut gray8, w, h))
        .expect("gray8");
    let mut gray_f32 = vec![0f32; w * h];
    reader()
        .read_rows_gray_f32(ImgRefMut::new(&mut gray_f32, w, h))
        .expect("gray_f32");
    let mut gray_lin = vec![0f32; w * h];
    reader()
        .read_rows_gray_linear_f32(ImgRefMut::new(&mut gray_lin, w, h))
        .expect("gray_linear_f32");
    let (mut y, mut cb, mut cr) = (vec![0f32; w * h], vec![0f32; w * h], vec![0f32; w * h]);
    let mut r = reader();
    let mut row = 0;
    while row < h {
        row += r
            .read_rows_ycbcr_f32(
                &mut y[row * w..],
                &mut cb[row * w..],
                &mut cr[row * w..],
                w,
                h - row,
            )
            .expect("ycbcr_f32");
    }
    let mut rgba = vec![0f32; w * h * 4];
    let mut r = reader();
    let mut row = 0;
    while row < h {
        row += r
            .read_rows_rgba_f32(ImgRefMut::new(&mut rgba[row * w * 4..], w * 4, h - row))
            .expect("rgba_f32");
    }
    Derived {
        gray8,
        gray_f32,
        gray_lin,
        y,
        cb,
        cr,
        rgba,
    }
}

/// R4: `decode_to_ycbcr_f32` on an RGB file returns the BT.601 forward
/// transform of the RGB planes, not the R/G/B planes relabelled.
#[test]
fn decode_to_ycbcr_f32_converts_rgb() {
    let (w, h) = (96u32, 64u32);
    let src = noise_image(w as usize, h as usize, 13);
    let ycc_file = as_ycbcr(&encode_ycbcr(
        &src,
        w,
        h,
        ChromaSubsampling::None,
        false,
        false,
    ));
    let rgb_file = as_rgb(&ycc_file);
    // The RGB file's R/G/B planes are the YCbCr twin's Y/Cb/Cr planes.
    let planes = Decoder::new()
        .decode_to_ycbcr_f32(&ycc_file, Unstoppable)
        .expect("twin");
    let got = Decoder::new()
        .decode_to_ycbcr_f32(&rgb_file, Unstoppable)
        .expect("rgb");
    let mut worst = 0f32;
    for i in 0..(w * h) as usize {
        // Planes are level-shifted: [-128, 127].
        let (r, g, b) = (
            planes.y[i] + 128.0,
            planes.cb[i] + 128.0,
            planes.cr[i] + 128.0,
        );
        let y = 0.299 * r + 0.587 * g + 0.114 * b - 128.0;
        let cb = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
        let cr = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        worst = worst
            .max((got.y[i] - y).abs())
            .max((got.cb[i] - cb).abs())
            .max((got.cr[i] - cr).abs());
    }
    assert!(
        worst < 1e-3,
        "decode_to_ycbcr_f32 on RGB differs from BT.601 forward by {worst}"
    );
}

// ===========================================================================
// 8. Regressions found while mapping the paths
// ===========================================================================

/// `read_rows_bgr8` must be `read_rows_rgb8` with R and B swapped, on YCbCr
/// files too (it used a separate f32 per-pixel conversion).
#[test]
fn scanline_bgr8_is_swapped_rgb8() {
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    let src = noise_image(w as usize, h as usize, 17);
    for sub in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        for progressive in [false, true] {
            let jpeg = encode_ycbcr(&src, w, h, sub, false, progressive);
            for ups in UPS {
                for owned in [false, true] {
                    let rgb = scanline(&jpeg, ups, 1, owned, Read::Rgb8);
                    let bgr = to_rgb(&scanline(&jpeg, ups, 1, owned, Read::Bgr8), 3, true);
                    let (max, first) = diff(&rgb, &bgr);
                    f.check(max == 0, || {
                        format!("{sub:?} prog={progressive} {ups:?} owned={owned}: bgr8 != swapped rgb8 by {max} at {first:?}")
                    });
                }
            }
        }
    }
    f.finish("bgr8 == swapped rgb8");
}

/// R5: the threaded fast integer path (progressive, >= 2,000,000 px) must
/// edge-replicate the chroma padding like its sequential twin. Even widths
/// that are not a multiple of 16 put real chroma next to MCU padding.
///
/// Parallel-only: without the `parallel` feature both sides take the
/// sequential path and the comparison holds trivially (CI runs this file with
/// `--features parallel`).
#[test]
fn parallel_fast_subsampled_last_column() {
    let mut f = Failures::default();
    for (w, h) in [(1602u32, 1300u32), (3000, 700)] {
        let src = noise_image(w as usize, h as usize, 19);
        for sub in [
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfHorizontal,
        ] {
            let ycc = encode_ycbcr(&src, w, h, sub, false, true);
            for (model, jpeg) in [("ycbcr", ycc.clone()), ("rgb", as_rgb(&ycc))] {
                for ups in UPS {
                    let t1 = decode_rgb(&jpeg, ups, 1);
                    let t0 = decode_rgb(&jpeg, ups, 0);
                    let (max, first) = diff(&t1, &t0);
                    f.check(max == 0, || {
                        let px = first.map(|i| (i / 3 % w as usize, i / 3 / w as usize));
                        format!("{w}x{h} {sub:?} {model} {ups:?}: threads 0 vs 1 differ by {max} at pixel {px:?}")
                    });
                }
            }
        }
    }
    f.finish("parallel fast subsampled last column");
}

/// Pre-existing: the strip readers (streaming and coefficient) built the
/// next MCU row's first chroma row before replicating the last real chroma
/// column over the MCU padding, so with 4:2:0, the triangle filter and an even
/// width that is not a multiple of 16, the last column of the last pixel row
/// of every MCU row interpolated against IDCT-rounded padding (off by up to 3
/// from libjpeg-turbo, which `decode()` matches exactly). YCbCr and RGB,
/// baseline and progressive, borrowed and owned readers.
#[test]
fn scanline_last_column_even_widths() {
    let mut f = Failures::default();
    for (w, h) in [(200u32, 120u32), (202, 70), (1602, 48)] {
        let src = noise_image(w as usize, h as usize, 29);
        for progressive in [false, true] {
            let ycc = encode_ycbcr(&src, w, h, ChromaSubsampling::Quarter, false, progressive);
            for (model, jpeg) in [("ycbcr", ycc.clone()), ("rgb", as_rgb(&ycc))] {
                let ups = ChromaUpsampling::Triangle;
                let reference = decode_rgb(&jpeg, ups, 1);
                for owned in [false, true] {
                    let got = scanline(&jpeg, ups, 1, owned, Read::Rgb8);
                    let (max, first) = diff(&reference, &got);
                    f.check(max == 0, || {
                        let px = first.map(|i| (i / 3 % w as usize, i / 3 / w as usize));
                        format!(
                            "{w}x{h} prog={progressive} {model} owned={owned}: \
                             scanline differs from decode by {max} at pixel {px:?}"
                        )
                    });
                }
            }
        }
    }
    f.finish("scanline last column");
}

/// Pre-existing: wave-parallel readers (nearest 4:2:0 / 4:2:2 with
/// MCU-row DRI under `parallel`) served only `read_rows_rgb8` and ignored crop.
/// Every RGB-family read now comes from the wave rows, cropped; the cropped
/// rows equal `decode()` cropped. (Without `parallel` this checks the
/// streaming reader's crop.)
#[test]
fn wave_reader_crop_and_formats() {
    let mut f = Failures::default();
    for layout in [LAYOUTS[1], LAYOUTS[2]] {
        for signal in [Signal::Rgb, Signal::YCbCr] {
            let jpeg = frame(layout, signal, true, &with_ac);
            let ups = ChromaUpsampling::NearestNeighbor;
            let full = decode_rgb(&jpeg, ups, 1);
            let (cx, cy, cw, ch) = (37u32, 51u32, 300u32, 150u32);
            let want: Vec<u8> = (cy..cy + ch)
                .flat_map(|y| {
                    let o = (y * FW + cx) as usize * 3;
                    full[o..o + cw as usize * 3].to_vec()
                })
                .collect();
            for read in [Read::Rgb8, Read::Bgr8, Read::Rgbx8, Read::Bgrx8] {
                let dec = decoder(ups, 0).crop(CropRegion::pixels(cx, cy, cw, ch));
                let mut r = dec.scanline_reader(&jpeg).expect("reader");
                let stride = cw as usize * read.bpp();
                let mut out = vec![0u8; stride * ch as usize];
                let mut row = 0;
                while row < ch as usize {
                    let img = ImgRefMut::new(&mut out[row * stride..], stride, ch as usize - row);
                    row += match read {
                        Read::Rgb8 => r.read_rows_rgb8(img),
                        Read::Bgr8 => r.read_rows_bgr8(img),
                        Read::Rgbx8 => r.read_rows_rgbx8(img),
                        Read::Bgrx8 => r.read_rows_bgrx8(img),
                    }
                    .unwrap_or_else(|e| panic!("{read:?}: {e}"));
                }
                let got = to_rgb(&out, read.bpp(), matches!(read, Read::Bgr8 | Read::Bgrx8));
                let (max, first) = diff(&want, &got);
                f.check(max == 0, || {
                    format!(
                        "{} {signal:?} {read:?}: cropped wave read differs by {max} at {first:?}",
                        layout.name
                    )
                });
            }
        }
    }
    f.finish("wave reader crop and formats");
}

/// Owned and borrowed scanline readers of a progressive file take the same
/// coefficient route, so every read agrees (owned readers used to decode to
/// RGB first and re-derive gray / YCbCr lossily).
#[test]
fn owned_and_borrowed_progressive_reads_agree() {
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    let src = noise_image(w as usize, h as usize, 31);
    for sub in [ChromaSubsampling::Quarter, ChromaSubsampling::None] {
        let ycc = encode_ycbcr(&src, w, h, sub, false, true);
        for (model, jpeg) in [("ycbcr", ycc.clone()), ("rgb", as_rgb(&ycc))] {
            for ups in UPS {
                let borrowed = derived_reads(&jpeg, ups, false);
                let owned = derived_reads(&jpeg, ups, true);
                let label = format!("{sub:?} {model} {ups:?}");
                let (max, first) = diff(&borrowed.gray8, &owned.gray8);
                f.check(max == 0, || {
                    format!("{label}: gray8 owned vs borrowed by {max} at {first:?}")
                });
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                for (name, a, b) in [
                    ("gray_f32", &borrowed.gray_f32, &owned.gray_f32),
                    ("gray_linear", &borrowed.gray_lin, &owned.gray_lin),
                    ("y", &borrowed.y, &owned.y),
                    ("cb", &borrowed.cb, &owned.cb),
                    ("cr", &borrowed.cr, &owned.cr),
                ] {
                    f.check(bits(a) == bits(b), || {
                        format!("{label}: {name} owned vs borrowed differ")
                    });
                }
            }
        }
    }
    f.finish("owned vs borrowed progressive reads");
}

/// Pre-existing: on YCbCr frames the streaming strip reader's luma reads
/// (`read_rows_gray8`, `read_rows_gray_f32`, `read_rows_gray_linear_f32`, the Y
/// of `read_rows_ycbcr_f32`) returned the wrong row for the last row of every
/// MCU row, the row the strip defers for the next MCU row's chroma context. On
/// a 2048x2048 4:2:0 baseline file 258,848 gray8 pixels (rows 15, 31, 47, ...)
/// differed from libjpeg-turbo's grayscale decode, by up to 246.
///
/// For the layouts the strip readers serve, every reader's luma reads must be
/// the Y plane exactly: (A) DC-only writer frames against the known answer,
/// (B) encoded content (baseline, DRI, progressive) against component 0 of the
/// same file signalled RGB, which the identity kernel returns unchanged. The
/// native i16 Y plane too. (Readers that hold decoded RGB, for exotic
/// sampling, derive luma from R, G, B by design; they are not covered here.)
#[test]
fn ycbcr_luma_reads_equal_the_y_plane_every_reader() {
    let mut f = Failures::default();
    let check = |f: &mut Failures, label: &str, jpeg: &[u8], luma: &[u8], w: usize| {
        for ups in UPS {
            for owned in [false, true] {
                let label = format!("{label} {ups:?} owned={owned}");
                let d = derived_reads(jpeg, ups, owned);
                let (max, first) = diff(luma, &d.gray8);
                f.check(max == 0, || {
                    format!("{label}: gray8 differs from the Y plane by {max} at {first:?}")
                });
                let unit = |v: u8| v as f32 / 255.0;
                let lin = |v: u8| linear_srgb::default::srgb_u8_to_linear(v);
                for (name, got, want) in [
                    ("gray_f32", &d.gray_f32, &unit as &dyn Fn(u8) -> f32),
                    ("ycbcr_f32 Y", &d.y, &unit),
                    ("gray_linear_f32", &d.gray_lin, &lin),
                ] {
                    if let Some(i) = got
                        .iter()
                        .zip(luma)
                        .position(|(g, &v)| g.to_bits() != want(v).to_bits())
                    {
                        f.check(false, || {
                            format!("{label}: {name} wrong at pixel {i} (row {})", i / w)
                        });
                    }
                }
            }
        }
        // Native i16 Y: no upsampling involved, one reader is enough.
        let mut r = Decoder::new().scanline_reader(jpeg).expect("reader");
        let h = luma.len() / w;
        let cw = r.chroma_width() as usize;
        let mut y = vec![0i16; w * (h + 32)];
        let (mut cb, mut cr) = (vec![0i16; cw * (h + 32)], vec![0i16; cw * (h + 32)]);
        let mut rows = 0;
        while rows < h {
            match r.read_rows_ycbcr_native_i16(&mut y[rows * w..], w, &mut cb, &mut cr, cw, 1) {
                Ok((0, _)) => break,
                Ok((ly, _)) => rows += ly,
                Err(e) => {
                    f.check(false, || format!("{label}: native i16 failed: {e}"));
                    return;
                }
            }
        }
        let native: Vec<u8> = y[..w * h].iter().map(|&v| v.clamp(0, 255) as u8).collect();
        let (max, first) = diff(luma, &native);
        f.check(rows >= h && max == 0, || {
            format!(
                "{label}: native i16 Y ({rows} rows) differs from the Y plane by {max} at {first:?}"
            )
        });
    };

    // (A) DC-only frames: the Y plane is known exactly.
    for layout in [LAYOUTS[0], LAYOUTS[1], LAYOUTS[2], LAYOUTS[3]] {
        for dri in [false, true] {
            let jpeg = frame(layout, Signal::YCbCr, dri, &dc_only);
            let coeffs = Decoder::new()
                .decode_coefficients(&jpeg, Unstoppable)
                .expect("coefficients");
            let ac_blocks: usize = coeffs
                .components
                .iter()
                .map(|c| {
                    c.coeffs
                        .as_chunks::<64>()
                        .0
                        .iter()
                        .filter(|b| b[1..].iter().any(|&v| v != 0))
                        .count()
                })
                .sum();
            f.check(ac_blocks == 0, || {
                format!("{} dri={dri}: {ac_blocks} blocks with AC != 0", layout.name)
            });
            let luma: Vec<u8> = dc_only_expected_nearest(layout)
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| p[0])
                .collect();
            let label = format!("dc-only {} dri={dri}", layout.name);
            check(&mut f, &label, &jpeg, &luma, FW as usize);
        }
    }

    // (B) Encoded content against component 0 of the RGB-signalled twin.
    for (w, h) in [(320u32, 240u32), (333, 251)] {
        let src = noise_image(w as usize, h as usize, 41);
        for sub in [
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfHorizontal,
            ChromaSubsampling::HalfVertical,
            ChromaSubsampling::None,
        ] {
            for (mode, dri, progressive) in [
                ("baseline", false, false),
                ("baseline dri", true, false),
                ("progressive", false, true),
            ] {
                let jpeg = encode_ycbcr(&src, w, h, sub, dri, progressive);
                let luma: Vec<u8> = decode_rgb(&as_rgb(&jpeg), ChromaUpsampling::Triangle, 1)
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|p| p[0])
                    .collect();
                let label = format!("{w}x{h} {sub:?} {mode}");
                check(&mut f, &label, &jpeg, &luma, w as usize);
            }
        }
    }
    f.finish("YCbCr luma reads");
}

/// Pre-existing: the fused parallel path charged its output buffer against
/// `max_memory` and then, for shapes it could not run, returned to the
/// sequential path, which charged again. With threads allowed, a decode must
/// fit the same budget as with one thread.
///
/// Parallel-only: without the `parallel` feature no fused path exists and the
/// comparison holds trivially (CI runs this file with `--features parallel`).
#[test]
fn parallel_decode_fits_the_sequential_memory_budget() {
    let mut f = Failures::default();
    for layout in [LAYOUTS[0], LAYOUTS[1], LAYOUTS[3], LAYOUTS[4]] {
        let jpeg = frame(layout, Signal::YCbCr, true, &with_ac);
        for ups in UPS {
            let fits = |budget: u64, threads: usize| {
                decoder(ups, threads)
                    .max_memory(budget)
                    .decode(&jpeg, Unstoppable)
                    .is_ok()
            };
            // Smallest budget the sequential decode fits (binary search).
            let (mut lo, mut hi) = (1u64, 1u64 << 32);
            assert!(fits(hi, 1), "{}: does not decode at 4 GiB", layout.name);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if fits(mid, 1) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            f.check(fits(lo, 0), || {
                format!(
                    "{} {ups:?}: threads=0 needs more than the sequential budget of {lo} bytes",
                    layout.name
                )
            });
        }
    }
    f.finish("parallel memory budget");
}

/// Pre-existing: `decode_into` with a 4bpp format and nearest upsampling on a
/// progressive 4:2:0 file wider than 8192 hit `assert!(n <= 8192)`.
#[test]
fn decode_into_wide_nearest_4bpp() {
    let (w, h) = (8200u32, 24u32);
    let src = noise_image(w as usize, h as usize, 23);
    let jpeg = encode_ycbcr(&src, w, h, ChromaSubsampling::Quarter, false, true);
    let ups = ChromaUpsampling::NearestNeighbor;
    for fmt in [PixelFormat::Rgba, PixelFormat::Bgra] {
        let reference = decode_fmt(&jpeg, ups, 1, fmt);
        let got = decode_into_fmt(&jpeg, ups, 1, fmt);
        let (max, first) = diff(&reference, &got);
        assert_eq!(
            max, 0,
            "{fmt:?}: decode_into differs from decode by {max} at {first:?}"
        );
    }
}

// ===========================================================================
// 10. Coverage added in review (C1 r2)
// ===========================================================================

/// Number of blocks with any non-zero AC coefficient, per component.
fn ac_blocks(jpeg: &[u8]) -> Vec<usize> {
    Decoder::new()
        .decode_coefficients(jpeg, Unstoppable)
        .expect("coefficients")
        .components
        .iter()
        .map(|c| {
            c.coeffs
                .as_chunks::<64>()
                .0
                .iter()
                .filter(|b| b[1..].iter().any(|&v| v != 0))
                .count()
        })
        .collect()
}

/// The same coefficients rewritten progressive by zenjpeg's lossless
/// restructure (no restart markers): a progressive twin of a writer frame.
fn progressive_twin(jpeg: &[u8]) -> Vec<u8> {
    use zenjpeg::lossless::{OutputMode, RestructureConfig, restructure};
    let config = RestructureConfig {
        output_mode: OutputMode::Progressive,
        ..Default::default()
    };
    restructure(jpeg, &config, Unstoppable).expect("restructure to progressive")
}

/// `decode()` threads=1 as the reference, collected as a failure (not a
/// panic) when it errors.
fn reference_rgb(
    f: &mut Failures,
    label: &str,
    jpeg: &[u8],
    ups: ChromaUpsampling,
) -> Option<Vec<u8>> {
    let e = Entry::run("decode Rgb t1".into(), 3, false, || {
        decode_rgb(jpeg, ups, 1)
    });
    match e.rgb {
        Ok(px) => Some(px),
        Err(msg) => {
            f.check(false, || format!("{label}: reference decode failed: {msg}"));
            None
        }
    }
}

/// Shapes `PathPlan::strip` / `PathPlan::fast_integer` admit with chroma
/// sampled above 1x1, which the MCU-walking kernels refuse (`unit_chroma`):
/// S444 with every component at (2,1), (3,1), (2,2) or (1,2); H1V2, H2V1 and
/// H2V2 with 2x chroma. Each frame has >= 1024 MCUs, so with MCU-row DRI it
/// meets every fused / wave condition but the shape. (2,2)x3,
/// (4,2)/(2,1)/(2,1) and (4,4)/(2,2)/(2,2) have more than T.81's ten blocks
/// per MCU; libjpeg refuses to write them, zenjpeg decodes them.
const EXTRA_LAYOUTS: [Layout; 9] = [
    Layout {
        name: "2x1_2x1_2x1",
        samp: [(2, 1), (2, 1), (2, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "3x1_3x1_3x1",
        samp: [(3, 1), (3, 1), (3, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "2x2_2x2_2x2",
        samp: [(2, 2), (2, 2), (2, 2)],
        size: (FW, FH),
    },
    Layout {
        name: "1x2_1x2_1x2",
        samp: [(1, 2), (1, 2), (1, 2)],
        size: (FW, FH),
    },
    Layout {
        name: "2x2_2x1_2x1",
        samp: [(2, 2), (2, 1), (2, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "2x2_1x2_1x2",
        samp: [(2, 2), (1, 2), (1, 2)],
        size: (FW, FH),
    },
    Layout {
        name: "4x1_2x1_2x1",
        samp: [(4, 1), (2, 1), (2, 1)],
        size: (FW, FH),
    },
    Layout {
        name: "4x2_2x1_2x1",
        samp: [(4, 2), (2, 1), (2, 1)],
        size: (FW, 530),
    },
    Layout {
        name: "4x4_2x2_2x2",
        samp: [(4, 4), (2, 2), (2, 2)],
        size: (FW, 1031),
    },
];

/// C1-11: the shapes above, DC-only (AC == 0 asserted; RGB under nearest
/// against the exact known answer) and AC content, RGB and YCbCr, with and
/// without MCU-row DRI, plus a progressive twin of the same coefficients.
/// Every entry point equals `decode()` threads=1 of the baseline file.
#[test]
fn extra_sampling_shapes_every_path() {
    let mut f = Failures::default();
    for layout in EXTRA_LAYOUTS {
        let (w, h) = layout.size;
        let mcu = (
            layout.samp.iter().map(|s| s.0 as u32).max().unwrap() * 8,
            layout.samp.iter().map(|s| s.1 as u32).max().unwrap() * 8,
        );
        assert!(
            w.div_ceil(mcu.0) * h.div_ceil(mcu.1) >= 1024,
            "{}: MCUs",
            layout.name
        );
        for (content, block) in CONTENTS {
            for signal in [Signal::Rgb, Signal::YCbCr] {
                for dri in [false, true] {
                    let jpeg = frame(layout, signal, dri, block);
                    let label = format!("{} {content} {signal:?} dri={dri}", layout.name);
                    if content == "dc-only" {
                        let ac = ac_blocks(&jpeg);
                        f.check(ac.iter().all(|&n| n == 0), || {
                            format!("{label}: blocks with AC != 0 per component {ac:?}")
                        });
                    }
                    let twin = (!dri).then(|| progressive_twin(&jpeg));
                    for ups in UPS {
                        let Some(reference) = reference_rgb(&mut f, &label, &jpeg, ups) else {
                            continue;
                        };
                        if content == "dc-only"
                            && signal == Signal::Rgb
                            && ups == ChromaUpsampling::NearestNeighbor
                        {
                            let (max, first) = diff(&dc_only_expected_nearest(layout), &reference);
                            f.check(max == 0, || {
                                format!("{label}: decode differs from the known answer by {max} at {first:?}")
                            });
                        }
                        compare_all(
                            &mut f,
                            &format!("{label} {ups:?}"),
                            &reference,
                            &every_entry(&jpeg, ups),
                        );
                        if let Some(twin) = &twin {
                            compare_all(
                                &mut f,
                                &format!("{label} progressive {ups:?}"),
                                &reference,
                                &every_entry(twin, ups),
                            );
                        }
                    }
                }
            }
        }
    }
    f.finish("extra sampling shapes");
}

/// 16x16 tiles of one colour each, plus a per-pixel luma texture added
/// equally to R, G and B. Cb and Cr are sums of R, G, B whose weights add up
/// to zero, so the texture leaves them unchanged: every chroma block of a
/// 4:2:0 or 4:2:2 encode is flat (DC-only) while luma carries AC.
fn flat_chroma_tiles(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let t = hash(0xF1A7, x / 16, y / 16, 0);
            let d = (hash(0x7E57, x, y, 1) % 41) as i32 - 20;
            for c in 0..3 {
                let base = 40 + ((t >> (8 * c)) % 170) as i32;
                rgb[(y * w + x) * 3 + c] = (base + d) as u8;
            }
        }
    }
    rgb
}

/// C1-2 / C1-12: flat chroma through the progressive paths. 1602x1312
/// (2.10 MP, over the fast integer path's 2,000,000-pixel threading
/// threshold), so progressive decodes take the threaded fast integer path
/// under `parallel`: every entry point (RGB, BGR, 4-byte, `decode_into`,
/// scanline) on YCbCr and relabelled RGB, 4:2:0 and 4:2:2, both upsamplers,
/// equals `decode()` threads=1. Chroma AC == 0 and luma AC != 0 asserted.
/// The height is a multiple of 16 because the encoder's fused 4:2:0 path
/// leaves the previous strip's chroma in a partial bottom MCU row's padding
/// rows (CLAUDE.md Known Bugs 6; at 1300 rows, 99 and 95 bottom-row chroma
/// blocks carry vertical AC); the width stays unaligned (1602 = 100 x 16 + 2),
/// and its padding is flat.
#[test]
fn flat_chroma_progressive_every_path() {
    let mut f = Failures::default();
    let (w, h) = (1602u32, 1312u32);
    let src = flat_chroma_tiles(w as usize, h as usize);
    for sub in [
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
    ] {
        let ycc = encode_ycbcr(&src, w, h, sub, false, true);
        let ac = ac_blocks(&ycc);
        f.check(ac[0] > 0 && ac[1] == 0 && ac[2] == 0, || {
            format!("{sub:?}: blocks with AC != 0 per component {ac:?} (want luma > 0, chroma 0)")
        });
        for (model, jpeg) in [("ycbcr", ycc.clone()), ("rgb", as_rgb(&ycc))] {
            for ups in UPS {
                let label = format!("{sub:?} {model} {ups:?}");
                let reference = decode_rgb(&jpeg, ups, 1);
                compare_all(&mut f, &label, &reference, &every_entry(&jpeg, ups));
            }
        }
    }
    f.finish("flat chroma progressive");
}

/// C1-5, pre-existing: non-interleaved sequential files (one scan per
/// component) reached the scanline strip and wave readers, which walk
/// interleaved MCUs, and came out as garbage (max diff 255 on ~99% of bytes),
/// while `decode()`'s streaming gate and the fused gate refused them.
/// Interleaving is part of the path plan now: every entry point equals
/// `decode()`, which equals the interleaved encode of the same coefficients
/// (libjpeg-turbo decodes each pair identically).
#[test]
fn non_interleaved_sequential_every_entry_point() {
    const DIR: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/testdata/non_interleaved/"
    );
    let read = |name: &str| {
        std::fs::read(format!("{DIR}{name}.jpg")).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let mut f = Failures::default();
    for (ni, int) in [
        ("ni420_88x54", "int420_88x54"),
        ("ni422_88x54", "int422_88x54"),
        ("ni444_88x54", "int444_88x54"),
        ("ni444_264x264_dri", "int444_264x264_dri"),
        ("gray2x2_88x54", "gray1x1_88x54"),
    ] {
        let (jpeg, twin) = (read(ni), read(int));
        for ups in UPS {
            let label = format!("{ni} {ups:?}");
            let reference = decode_rgb(&jpeg, ups, 1);
            let (max, first) = diff(&decode_rgb(&twin, ups, 1), &reference);
            f.check(max == 0, || {
                format!("{label}: decode differs from {int} by {max} at {first:?}")
            });
            compare_all(&mut f, &label, &reference, &every_entry(&jpeg, ups));
        }
    }
    f.finish("non-interleaved sequential");
}

/// Scanline gray8 of a whole file.
/// C1-21: every marker walker (`read_header`, the `decode()` loop,
/// `find_scan_info`, the SOS peek that decides `interleaved` at SOF) classifies
/// TEM (`FF01`) and RST (`FFD0`-`FFD7`) as standalone markers, as T.81 B.1.1.3,
/// jpegli and IJG do. The `decode()` loop read TEM as a length segment and
/// `find_scan_info` read both that way, so with a TEM after SOF the walkers
/// disagreed on which SOS came first: with `FF01 0008` + a fake 3-component
/// SOS header (reviewer 1's `tem_ni444.jpg`) the peek saw an interleaved scan
/// and the scanline reader walked the real non-interleaved scan as
/// interleaved MCUs (the C1-5 garbage), while `decode()` skipped the fake SOS.
/// Now a bare TEM or RST after SOF is skipped by every walker (every entry
/// point equals the clean file's decode), and the fake-SOS file is rejected by
/// every entry point, as jpegli and IJG reject it.
#[test]
fn tem_and_rst_after_sof_are_standalone_everywhere() {
    const DIR: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/testdata/non_interleaved/"
    );
    let read = |name: &str| {
        std::fs::read(format!("{DIR}{name}.jpg")).unwrap_or_else(|e| panic!("{name}: {e}"))
    };
    let mut f = Failures::default();
    // Insert `extra` right after the SOF segment.
    let after_sof = |jpeg: &[u8], extra: &[u8]| -> Vec<u8> {
        let segs = jpeg_writer::segments(jpeg);
        let &(p, _, len) = segs
            .iter()
            .find(|&&(_, m, _)| matches!(m, 0xC0..=0xC2))
            .expect("SOF");
        [&jpeg[..p + len], extra, &jpeg[p + len..]].concat()
    };
    for name in ["ni444_88x54", "int444_88x54", "ni420_88x54"] {
        let clean = read(name);
        for (what, extra) in [
            ("TEM", &[0xFF, 0x01][..]),
            ("RST0", &[0xFF, 0xD0]),
            ("RST7 + TEM", &[0xFF, 0xD7, 0xFF, 0x01]),
        ] {
            let jpeg = after_sof(&clean, extra);
            for ups in UPS {
                let label = format!("{name} + {what} {ups:?}");
                let reference = decode_rgb(&clean, ups, 1);
                compare_all(&mut f, &label, &reference, &every_entry(&jpeg, ups));
            }
        }
        let fake = after_sof(
            &clean,
            &[0xFF, 0x01, 0x00, 0x08, 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x00],
        );
        for threads in [1usize, 0] {
            let label = format!("{name} + TEM + fake SOS t{threads}");
            let dec = decoder(ChromaUpsampling::Triangle, threads);
            f.check(dec.decode(&fake, Unstoppable).is_err(), || {
                format!("{label}: decode accepted it")
            });
            let mut dst = vec![0u8; 88 * 54 * 3];
            f.check(
                dec.decode_into(&fake, PixelFormat::Rgb, &mut dst, Unstoppable)
                    .is_err(),
                || format!("{label}: decode_into accepted it"),
            );
            f.check(dec.scanline_reader(&fake).is_err(), || {
                format!("{label}: scanline_reader accepted it")
            });
            f.check(
                dec.scanline_reader_cow(Cow::Owned(fake.clone())).is_err(),
                || format!("{label}: owned scanline_reader accepted it"),
            );
        }
    }
    f.finish("TEM / RST after SOF");
}

fn scanline_gray8(dec: &Decoder, jpeg: &[u8]) -> Vec<u8> {
    let mut r = dec.scanline_reader(jpeg).expect("reader");
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut out = vec![0u8; w * h];
    let mut row = 0;
    while row < h {
        row += r
            .read_rows_gray8(ImgRefMut::new(&mut out[row * w..], w, h - row))
            .expect("gray8");
    }
    out
}

/// C1-7: `decode_to_ycbcr_f32` takes the IDCT the decoder is configured with
/// on every upsampler (it chose the kernel by upsampler: the 12-bit one under
/// nearest whatever `IdctMethod` said), so its Y plane equals the scanline
/// gray8 read of the same decoder, which is the Y plane.
#[test]
fn decode_to_ycbcr_f32_honours_idct_method() {
    use zenjpeg::decode::IdctMethod;
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    let src = noise_image(w as usize, h as usize, 43);
    for progressive in [false, true] {
        let jpeg = encode_ycbcr(&src, w, h, ChromaSubsampling::Quarter, false, progressive);
        for method in [IdctMethod::Libjpeg, IdctMethod::Jpegli] {
            for ups in UPS {
                let label = format!("progressive={progressive} {method:?} {ups:?}");
                let dec = decoder(ups, 1).idct_method(method);
                let ycc = dec
                    .decode_to_ycbcr_f32(&jpeg, Unstoppable)
                    .expect("ycbcr f32");
                let y: Vec<u8> = ycc.y.iter().map(|&v| (v + 128.0) as u8).collect();
                let exact = ycc.y.iter().all(|&v| (v + 128.0).fract() == 0.0);
                f.check(exact, || format!("{label}: Y plane is not integer-valued"));
                let (max, first) = diff(&scanline_gray8(&dec, &jpeg), &y);
                f.check(max == 0, || {
                    format!(
                        "{label}: decode_to_ycbcr_f32 Y differs from gray8 by {max} at {first:?}"
                    )
                });
            }
        }
    }
    f.finish("decode_to_ycbcr_f32 IDCT method");
}

/// C1-9: with an f32 output target the strip runs the unclamped IDCT, so the
/// YCbCr row kernels see samples below 0 and above 255. The AVX2 kernels
/// zero-extended Y (Y = -5 became 65531, white specks in each 16-pixel SIMD
/// block) while the AVX-512 kernel and the scalar tail sign-extended, so 8-bit
/// reads disagreed with each other on AVX-512 hosts (`rgb8` vs `rgbx8`).
/// Every 8-bit layout must agree, on the streaming and the coefficient route.
#[test]
fn f32_target_u8_reads_agree_across_layouts() {
    use zenjpeg::decode::IdctMethod;
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    // Saturated noise rings past 0 and 255 after a coarse quantizer.
    let src: Vec<u8> = (0..w as usize * h as usize * 3)
        .map(|i| {
            if hash(0xB1A5, i, 0, 0) % 2 == 0 {
                0
            } else {
                255
            }
        })
        .collect();
    for sub in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        for progressive in [false, true] {
            let jpeg = EncoderConfig::ycbcr(40, sub)
                .progressive(progressive)
                .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
                .expect("encode");
            for ups in UPS {
                let label = format!("{sub:?} progressive={progressive} {ups:?}");
                let dec = decoder(ups, 1)
                    .idct_method(IdctMethod::Jpegli)
                    .output_target(OutputTarget::SrgbF32);
                // The regime is reached: the unclamped Y plane leaves [0, 1].
                let mut r = dec.scanline_reader(&jpeg).expect("reader");
                let n = w as usize * h as usize;
                let (mut y, mut cb, mut cr) = (vec![0f32; n], vec![0f32; n], vec![0f32; n]);
                let mut row = 0;
                while row < h as usize {
                    let k = w as usize * row;
                    row += r
                        .read_rows_ycbcr_f32(
                            &mut y[k..],
                            &mut cb[k..],
                            &mut cr[k..],
                            w as usize,
                            h as usize - row,
                        )
                        .expect("ycbcr_f32");
                }
                f.check(y.iter().any(|&v| v < 0.0) && y.iter().any(|&v| v > 1.0), || {
                    format!("{label}: Y never leaves [0, 1]; the test does not reach the unclamped regime")
                });
                let read = |rd: Read| {
                    let dec = dec.clone();
                    let mut r = dec.scanline_reader(&jpeg).expect("reader");
                    let stride = w as usize * rd.bpp();
                    let mut out = vec![0u8; stride * h as usize];
                    let mut row = 0;
                    while row < h as usize {
                        let img =
                            ImgRefMut::new(&mut out[row * stride..], stride, h as usize - row);
                        row += match rd {
                            Read::Rgb8 => r.read_rows_rgb8(img),
                            Read::Bgr8 => r.read_rows_bgr8(img),
                            Read::Rgbx8 => r.read_rows_rgbx8(img),
                            Read::Bgrx8 => r.read_rows_bgrx8(img),
                        }
                        .expect("read");
                    }
                    to_rgb(&out, rd.bpp(), matches!(rd, Read::Bgr8 | Read::Bgrx8))
                };
                let rgb = read(Read::Rgb8);
                for rd in [Read::Bgr8, Read::Rgbx8, Read::Bgrx8] {
                    let (max, first) = diff(&rgb, &read(rd));
                    f.check(max == 0, || {
                        format!("{label}: {rd:?} differs from Rgb8 by {max} at {first:?}")
                    });
                }
            }
        }
    }
    f.finish("f32-target u8 reads");
}

/// A grayscale frame's f32 scanline reads, against its 8-bit gray: `gray_f32`
/// is g / 255, `gray_linear_f32` the exact 8-bit sRGB table, `rgba_f32` that
/// value in R, G and B with alpha 1. Bit-exact, on whichever reader the frame
/// gets (streaming strip, or decoded gray pixels for progressive and
/// oversampled frames).
fn check_gray_f32_reads(f: &mut Failures, label: &str, jpeg: &[u8], gray: &[u8], owned: bool) {
    let reader = || {
        let dec = decoder(ChromaUpsampling::Triangle, 1);
        if owned {
            dec.scanline_reader_cow(Cow::Owned(jpeg.to_vec()))
        } else {
            dec.scanline_reader(jpeg)
        }
        .expect("reader")
    };
    let n = gray.len();
    let w = reader().width() as usize;
    let h = n / w;
    let (mut gf, mut gl, mut rgba) = (vec![0f32; n], vec![0f32; n], vec![0f32; n * 4]);
    let (mut r, mut row) = (reader(), 0);
    while row < h {
        row += r
            .read_rows_gray_f32(ImgRefMut::new(&mut gf[row * w..], w, h - row))
            .expect("gray_f32");
    }
    let (mut r, mut row) = (reader(), 0);
    while row < h {
        row += r
            .read_rows_gray_linear_f32(ImgRefMut::new(&mut gl[row * w..], w, h - row))
            .expect("gray_linear_f32");
    }
    let (mut r, mut row) = (reader(), 0);
    while row < h {
        row += r
            .read_rows_rgba_f32(ImgRefMut::new(&mut rgba[row * w * 4..], w * 4, h - row))
            .expect("rgba_f32");
    }
    // YCbCr planes of a gray frame: Y is the sample, chroma is neutral.
    let (mut y, mut cb, mut cr) = (vec![0f32; n], vec![-1f32; n], vec![-1f32; n]);
    let (mut r, mut row) = (reader(), 0);
    while row < h {
        let o = row * w;
        row += r
            .read_rows_ycbcr_f32(&mut y[o..], &mut cb[o..], &mut cr[o..], w, h - row)
            .expect("ycbcr_f32");
    }
    for (i, &g) in gray.iter().enumerate() {
        let l = linear_srgb::default::srgb_u8_to_linear(g);
        let got = &rgba[i * 4..i * 4 + 4];
        let ok = gf[i].to_bits() == (g as f32 / 255.0).to_bits()
            && y[i].to_bits() == (g as f32 / 255.0).to_bits()
            && cb[i] == 0.0
            && cr[i] == 0.0
            && gl[i].to_bits() == l.to_bits()
            && got
                .iter()
                .zip([l, l, l, 1.0])
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !ok {
            f.check(false, || {
                format!(
                    "{label}: pixel {i} (gray {g}): gray_f32 {}, ycbcr_f32 ({}, {}, {}), gray_linear {} (want {l}), rgba_f32 {got:?}",
                    gf[i], y[i], cb[i], cr[i], gl[i]
                )
            });
            break;
        }
    }
}

/// `read_rows_rgba_f32` on YCbCr frames converts the strip's planes in f32
/// (BT.601 with `foundation::consts::YCBCR_*`, then the f32 sRGB transfer),
/// with no 8-bit rounding in between. The planes come back exactly from
/// `read_rows_ycbcr_f32` (integer samples / 255), so the expected value is
/// rebuilt from them, bit for bit, on every reader mode.
#[test]
fn ycbcr_rgba_f32_is_the_planes_through_bt601() {
    let mut f = Failures::default();
    let (w, h) = (333u32, 251u32);
    let src = noise_image(w as usize, h as usize, 61);
    for sub in [ChromaSubsampling::None, ChromaSubsampling::Quarter] {
        for progressive in [false, true] {
            let jpeg = encode_ycbcr(&src, w, h, sub, false, progressive);
            for ups in UPS {
                for owned in [false, true] {
                    let label = format!("{sub:?} progressive={progressive} {ups:?} owned={owned}");
                    let d = derived_reads(&jpeg, ups, owned);
                    for i in 0..d.y.len() {
                        let y = (d.y[i] * 255.0).round();
                        let cb = (d.cb[i] * 255.0 + 128.0).round();
                        let cr = (d.cr[i] * 255.0 + 128.0).round();
                        let (cbf, crf) = (cb - 128.0, cr - 128.0);
                        let r = 1.0 * y + 0.0 * cbf + 1.402 * crf;
                        let g = 1.0 * y + -0.344_136 * cbf + -0.714_136 * crf;
                        let b = 1.0 * y + 1.772 * cbf + 0.0 * crf;
                        let lin = |v: f32| linear_srgb::default::srgb_to_linear(v / 255.0);
                        let want = [lin(r), lin(g), lin(b), 1.0];
                        let got = &d.rgba[i * 4..i * 4 + 4];
                        if !got
                            .iter()
                            .zip(want)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                        {
                            f.check(false, || {
                                format!("{label}: pixel {i}: rgba_f32 {got:?}, want {want:?}")
                            });
                            break;
                        }
                    }
                }
            }
        }
    }
    f.finish("YCbCr rgba_f32");
}

/// C1-20: the scanline fallbacks for a lossless transform and for coefficient
/// deblocking (Knusperli) decode the whole image and serve every read from
/// those bytes. They inherited the caller's `output_format` /
/// `output_target`: with `Bgr`, `read_rows_rgb8` returned B, G, R; with `Rgba`
/// the 4-byte buffer was read at a 3-byte stride; with `SrgbF32` the reader
/// failed ("expected u8 pixel data"). Every read now equals the same fallback
/// with the default format and target, borrowed and owned, for a YCbCr and a
/// grayscale frame.
#[test]
fn buffered_fallbacks_ignore_caller_format_and_target() {
    use zenjpeg::decode::DeblockMode;
    use zenjpeg::lossless::LosslessTransform;
    let mut f = Failures::default();
    let (w, h) = (96u32, 80u32);
    let src = noise_image(w as usize, h as usize, 67);
    let gray_src: Vec<u8> = src.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
    let files = [
        (
            "ycbcr 4:2:0",
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
                .progressive(false)
                .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
                .expect("encode"),
        ),
        (
            "gray",
            EncoderConfig::grayscale(85)
                .progressive(false)
                .encode_bytes(&gray_src, w, h, PixelLayout::Gray8Srgb)
                .expect("encode"),
        ),
    ];
    let read_owned = |dec: &Decoder, jpeg: &[u8], which: &str| -> Result<Vec<u8>, String> {
        let mut r = dec
            .scanline_reader_cow(Cow::Owned(jpeg.to_vec()))
            .map_err(|e| e.to_string())?;
        let (w, h) = (r.width() as usize, r.height() as usize);
        let bpp = if which == "gray8" { 1 } else { 3 };
        let mut out = vec![0u8; w * h * bpp];
        let mut row = 0;
        while row < h {
            let img = ImgRefMut::new(&mut out[row * w * bpp..], w * bpp, h - row);
            row += if bpp == 1 {
                r.read_rows_gray8(img)
            } else {
                r.read_rows_rgb8(img)
            }
            .map_err(|e| e.to_string())?;
        }
        Ok(out)
    };
    let fallbacks: [(&str, fn(Decoder) -> Decoder); 2] = [
        ("transform", |d| {
            d.transform(LosslessTransform::FlipHorizontal)
        }),
        ("knusperli", |d| d.deblock(DeblockMode::Knusperli)),
    ];
    let tweaks: [(&str, fn(Decoder) -> Decoder); 6] = [
        ("Bgr", |d| d.output_format(PixelFormat::Bgr)),
        ("Rgba", |d| d.output_format(PixelFormat::Rgba)),
        ("Bgra", |d| d.output_format(PixelFormat::Bgra)),
        ("Gray", |d| d.output_format(PixelFormat::Gray)),
        ("SrgbF32", |d| d.output_target(OutputTarget::SrgbF32)),
        ("LinearF32", |d| d.output_target(OutputTarget::LinearF32)),
    ];
    for (file, jpeg) in &files {
        for (fb, apply) in fallbacks {
            let base = apply(decoder(ChromaUpsampling::Triangle, 1));
            for which in ["rgb8", "gray8"] {
                let reference = plane_read_with(&base, jpeg, which);
                let Ok(reference) = reference else {
                    f.check(false, || {
                        format!("{file} {fb}: default {which} failed: {reference:?}")
                    });
                    continue;
                };
                for (tweak, t) in tweaks {
                    let dec = t(base.clone());
                    let label = format!("{file} {fb} {tweak} {which}");
                    match plane_read_with(&dec, jpeg, which) {
                        Ok(v) => {
                            f.check(v == reference, || format!("{label}: borrowed read differs"))
                        }
                        Err(e) => f.check(false, || format!("{label}: borrowed read failed: {e}")),
                    }
                    let want: Vec<u8> = reference.iter().map(|&v| v as u8).collect();
                    match read_owned(&dec, jpeg, which) {
                        Ok(v) => f.check(v == want, || format!("{label}: owned read differs")),
                        Err(e) => f.check(false, || format!("{label}: owned read failed: {e}")),
                    }
                }
            }
        }
    }
    f.finish("buffered fallbacks vs caller format and target");
}

/// Byte offset of each APPn segment before the first SOS, with whether it
/// follows SOF: `(marker, payload starts, after_sof)`.
fn app_positions(jpeg: &[u8]) -> Vec<(u8, Vec<u8>, bool)> {
    let mut seen_sof = false;
    jpeg_writer::segments(jpeg)
        .into_iter()
        .filter_map(|(p, m, len)| {
            if matches!(m, 0xC0..=0xC2) {
                seen_sof = true;
            }
            (0xE0..=0xEF)
                .contains(&m)
                .then(|| (m, jpeg[p + 4..p + len].to_vec(), seen_sof))
        })
        .collect()
}

/// A minimal big-endian EXIF APP1 carrying only an Orientation tag.
fn exif_orientation(orientation: u16) -> Vec<u8> {
    let mut payload = b"Exif\0\0MM\0\x2a\0\0\0\x08".to_vec();
    payload.extend_from_slice(&[0x00, 0x01]); // one IFD0 entry
    payload.extend_from_slice(&[0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01]);
    payload.extend_from_slice(&orientation.to_be_bytes());
    payload.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00, 0x00]); // pad, next IFD
    let mut seg = vec![0xFF, 0xE1];
    seg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    seg.extend_from_slice(&payload);
    seg
}

/// C1-22: a JFIF or Adobe segment placed between SOF and SOS does not decide
/// the colour model (decided at SOF). The lossless transform, EXIF
/// orientation, restructure and recompress-preserve writers emitted every
/// preserved APPn right after SOI,
/// moving such a segment in front of SOF and changing how the output decodes
/// (`decode(transform(x)) != transform(decode(x))`; jpegli: max diff 207). Each
/// preserved segment now keeps its side of SOF. Fixtures: IDs 1,2,3 with an
/// APP14 transform 0 after SOF (YCbCr), and IDs 'R','G','B' with a JFIF after
/// SOF (RGB). Each output reports the source's model, keeps the segment after
/// SOF, and decodes exactly like the same operation on a control file that
/// signals the same model with no post-SOF segment (IDs 1,2,3 alone; IDs
/// 'R','G','B' + APP14 transform 0 before SOF). `apply_exif_orientation` runs
/// on twins of both that also carry an EXIF Orientation 2 before SOF.
#[test]
fn rewrites_keep_colour_markers_on_their_side_of_sof() {
    use zenjpeg::lossless::{
        EdgeHandling, LosslessTransform, OutputMode, RestructureConfig, TransformConfig,
        apply_exif_orientation, restructure, transform,
    };
    let mut f = Failures::default();
    let (w, h) = (96u32, 80u32);
    let src = noise_image(w as usize, h as usize, 71);
    let ycc = EncoderConfig::ycbcr(92, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode");
    let exif = exif_orientation(2);
    let fixtures = [
        (
            "ids 1,2,3 + APP14 t0 after SOF",
            [1, 2, 3],
            vec![],
            vec![app14(0)],
            vec![],
            ColorSpace::YCbCr,
        ),
        (
            "ids RGB + JFIF after SOF",
            *b"RGB",
            vec![],
            vec![jfif()],
            vec![app14(0)],
            ColorSpace::Rgb,
        ),
    ];
    for (name, ids, pre, post, control_pre, model) in &fixtures {
        let jpeg = &resignal(&ycc, *ids, pre, post);
        let control = &resignal(&ycc, *ids, control_pre, &[]);
        let with_exif = |pre: &[Vec<u8>]| [&[exif.clone()][..], pre].concat();
        let jpeg_exif = resignal(&ycc, *ids, &with_exif(pre), post);
        let control_exif = resignal(&ycc, *ids, &with_exif(control_pre), &[]);
        let info = Decoder::new().read_info(jpeg).expect("info");
        assert_eq!(info.color_space, *model, "{name}: fixture model");
        let colour_seg = app_positions(jpeg)
            .into_iter()
            .find(|(m, p, _)| {
                (*m == 0xEE && p.starts_with(b"Adobe")) || (*m == 0xE0 && p.starts_with(b"JFIF"))
            })
            .expect("colour segment");
        assert!(
            colour_seg.2,
            "{name}: fixture keeps its colour segment after SOF"
        );
        assert_eq!(
            Decoder::new().read_info(control).expect("info").color_space,
            *model,
            "{name}: control model"
        );
        let mut outputs: Vec<(String, Vec<u8>, Vec<u8>)> = Vec::new();
        for mode in [OutputMode::Sequential, OutputMode::Progressive] {
            let cfg = RestructureConfig {
                output_mode: mode,
                ..Default::default()
            };
            let run = |j: &[u8]| restructure(j, &cfg, Unstoppable).expect("restructure");
            outputs.push((format!("restructure {mode:?}"), run(jpeg), run(control)));
        }
        let cfg = TransformConfig {
            transform: LosslessTransform::FlipHorizontal,
            edge_handling: EdgeHandling::TrimPartialBlocks,
        };
        let run = |j: &[u8]| transform(j, &cfg, Unstoppable).expect("transform");
        outputs.push(("transform FlipHorizontal".into(), run(jpeg), run(control)));
        let run = |j: &[u8]| apply_exif_orientation(j, Unstoppable).expect("orientation");
        let (out, control_out) = (run(&jpeg_exif), run(&control_exif));
        f.check(out != jpeg_exif, || {
            format!("{name}: apply_exif_orientation left the file unchanged")
        });
        outputs.push(("apply_exif_orientation 2".into(), out, control_out));
        #[cfg(feature = "recompress")]
        {
            use zenjpeg::recompress::{
                RecompressOptions, RecompressResult, StrategyKind, recompress,
            };
            match recompress(jpeg, &RecompressOptions::new(70.0)) {
                Ok(RecompressResult::Recompressed {
                    bytes,
                    strategy: StrategyKind::Preserve,
                    ..
                }) => {
                    // Lossy: check the model and the placement, not pixels.
                    let info = Decoder::new().read_info(&bytes).expect("info");
                    f.check(info.color_space == *model, || {
                        format!("{name} recompress preserve: model {:?}", info.color_space)
                    });
                    let seg = app_positions(&bytes)
                        .into_iter()
                        .find(|(m, p, _)| *m == colour_seg.0 && *p == colour_seg.1);
                    f.check(seg.as_ref().is_some_and(|s| s.2), || {
                        format!("{name} recompress preserve: colour segment not after SOF: {seg:?}")
                    });
                }
                other => f.check(false, || {
                    format!(
                        "{name}: recompress did not take Preserve: {:?}",
                        other
                            .as_ref()
                            .map(|r| matches!(r, RecompressResult::Recompressed { .. }))
                    )
                }),
            }
        }
        for (op, out, control_out) in outputs {
            let want = decode_rgb(&control_out, ChromaUpsampling::Triangle, 1);
            let label = format!("{name} {op}");
            let info = Decoder::new().read_info(&out).expect("info");
            f.check(info.color_space == *model, || {
                format!("{label}: model {:?}, source {model:?}", info.color_space)
            });
            let seg = app_positions(&out)
                .into_iter()
                .find(|(m, p, _)| *m == colour_seg.0 && *p == colour_seg.1);
            f.check(seg.as_ref().is_some_and(|s| s.2), || {
                format!("{label}: colour segment not after SOF: {seg:?}")
            });
            let (max, first) = diff(&decode_rgb(&out, ChromaUpsampling::Triangle, 1), &want);
            f.check(max == 0, || {
                format!("{label}: pixels differ by {max} at {first:?}")
            });
        }
    }
    f.finish("colour markers keep their side of SOF");
}

/// Every APPn segment with its position: 0 before SOF, 1 between SOF and the
/// first SOS, 2 after the first SOS (walks entropy-coded data between scans).
fn app_positions_all(jpeg: &[u8]) -> Vec<(u8, Vec<u8>, u8)> {
    let mut out = Vec::new();
    let mut pos = 0u8;
    let mut p = 2;
    while p + 4 <= jpeg.len() {
        if jpeg[p] != 0xFF {
            p += 1;
            continue;
        }
        let m = jpeg[p + 1];
        if m == 0xFF || m == 0x00 || (0xD0..=0xD7).contains(&m) {
            p += if m == 0xFF { 1 } else { 2 };
            continue;
        }
        if m == 0xD9 {
            break;
        }
        let len = 2 + jpeg_writer::be16(jpeg, p + 2);
        if matches!(m, 0xC0..=0xC2) && pos == 0 {
            pos = 1;
        }
        if (0xE0..=0xEF).contains(&m) {
            out.push((m, jpeg[p + 4..p + len].to_vec(), pos));
        }
        if m == 0xDA {
            pos = 2;
            // Skip the entropy-coded data up to the next real marker.
            p += len;
            while p + 1 < jpeg.len()
                && !(jpeg[p] == 0xFF
                    && jpeg[p + 1] != 0x00
                    && !(0xD0..=0xD7).contains(&jpeg[p + 1]))
            {
                p += 1;
            }
            continue;
        }
        p += len;
    }
    out
}

/// `jpeg` with one stray `0x00` byte right before its SOF (decoders skip it
/// with a warning).
fn stray_byte_before_sof(jpeg: &[u8]) -> Vec<u8> {
    let (p, _, _) = jpeg_writer::segments(jpeg)
        .into_iter()
        .find(|(_, m, _)| matches!(m, 0xC0..=0xC2))
        .expect("SOF");
    [&jpeg[..p], &[0x00], &jpeg[p..]].concat()
}

/// Review round 3, B1: a colour marker after the first SOS has no effect on
/// the source's decode, and a stray byte before SOF does not hide the side of
/// SOF a marker sits on. Every rewrite must decode like its source: same
/// model in `read_info`, same pixels, and no colour marker moved to a
/// position where it starts to count.
#[test]
fn rewrites_keep_post_scan_and_stray_colour_markers_inert() {
    use zenjpeg::lossless::{
        EdgeHandling, LosslessTransform, OutputMode, RestructureConfig, TransformConfig,
        apply_exif_orientation, restructure, transform,
    };
    let mut f = Failures::default();
    let (w, h) = (96u32, 80u32);
    let src = noise_image(w as usize, h as usize, 73);
    let enc = |progressive: bool| {
        EncoderConfig::ycbcr(92, ChromaSubsampling::Quarter)
            .progressive(progressive)
            .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
            .expect("encode")
    };
    let (seq, prog) = (enc(false), enc(true));
    let exif = exif_orientation(2);
    let xyb = EncoderConfig::xyb(88, XybSubsampling::Full)
        .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
        .expect("xyb encode");
    let icc: Vec<u8> = jpeg_writer::segments(&xyb)
        .into_iter()
        .filter(|&(_, marker, _)| marker == 0xE2)
        .flat_map(|(p, _, len)| xyb[p..p + len].iter().copied())
        .collect();
    assert!(!icc.is_empty(), "XYB fixture must carry an ICC profile");
    let is_colour = |m: u8, p: &[u8]| {
        (m == 0xEE && p.starts_with(b"Adobe"))
            || (m == 0xE0 && p.starts_with(b"JFIF"))
            || (m == 0xE2 && p.starts_with(b"ICC_PROFILE\0"))
    };
    // (name, build(pre) -> fixture, control_pre, ids, model, colour position
    // the output must keep: 2 = after the first SOS, 1 = between
    // SOF and the first SOS).
    type Build<'a> = Box<dyn Fn(&[Vec<u8>]) -> Vec<u8> + 'a>;
    let fixtures: Vec<(&str, Build<'_>, Vec<Vec<u8>>, [u8; 3], ColorSpace, u8)> = vec![
        (
            "progressive ids RGB + JFIF between scans",
            Box::new(|pre: &[Vec<u8>]| {
                insert_before_second_sos(&resignal(&prog, *b"RGB", pre, &[]), &jfif())
            }),
            vec![app14(0)],
            *b"RGB",
            ColorSpace::Rgb,
            2,
        ),
        (
            "progressive ids 1,2,3 + APP14 t0 between scans",
            Box::new(|pre: &[Vec<u8>]| {
                insert_before_second_sos(&resignal(&prog, [1, 2, 3], pre, &[]), &app14(0))
            }),
            vec![],
            [1, 2, 3],
            ColorSpace::YCbCr,
            2,
        ),
        (
            "progressive ids 1,2,3 + duplicate APP14 t0 between scans",
            Box::new(|pre: &[Vec<u8>]| {
                insert_before_second_sos(
                    &resignal(&prog, [1, 2, 3], pre, &[]),
                    &[app14(0), app14(0)].concat(),
                )
            }),
            vec![],
            [1, 2, 3],
            ColorSpace::YCbCr,
            2,
        ),
        (
            "progressive ids RGB + XYB ICC between scans",
            Box::new(|pre: &[Vec<u8>]| {
                insert_before_second_sos(&resignal(&prog, *b"RGB", pre, &[]), &icc)
            }),
            vec![],
            *b"RGB",
            ColorSpace::Rgb,
            2,
        ),
        (
            "ids 1,2,3 + APP14 t0 after SOF + stray byte before SOF",
            Box::new(|pre: &[Vec<u8>]| {
                stray_byte_before_sof(&resignal(&seq, [1, 2, 3], pre, &[app14(0)]))
            }),
            vec![],
            [1, 2, 3],
            ColorSpace::YCbCr,
            1,
        ),
    ];
    let assert_retained = |source: &[u8], output: &[u8], label: &str| {
        let source = app_positions_all(source);
        let output = app_positions_all(output);
        for (marker, payload, _) in &source {
            let wanted = source
                .iter()
                .filter(|(m, p, _)| m == marker && p == payload)
                .count();
            let actual = output
                .iter()
                .filter(|(m, p, _)| m == marker && p == payload)
                .count();
            assert_eq!(
                actual, wanted,
                "{label}: FF{marker:02X} payload multiplicity"
            );
        }
    };
    for (name, build, control_pre, ids, model, keep_at) in &fixtures {
        let base = if name.starts_with("progressive") {
            &prog
        } else {
            &seq
        };
        let jpeg = build(&[]);
        let jpeg_exif = build(std::slice::from_ref(&exif));
        let control = resignal(base, *ids, control_pre, &[]);
        let control_exif = resignal(
            base,
            *ids,
            &[&[exif.clone()][..], control_pre].concat(),
            &[],
        );
        assert_eq!(
            Decoder::new().read_info(&jpeg).expect("info").color_space,
            *model,
            "{name}: fixture model"
        );
        assert_eq!(
            Decoder::new()
                .read_info(&control)
                .expect("info")
                .color_space,
            *model,
            "{name}: control model"
        );
        let mut outputs: Vec<(String, Vec<u8>, Vec<u8>)> = Vec::new();
        let assert_refusal = |error: zenjpeg::decoder::Error| {
            assert!(
                matches!(
                    error.kind(),
                    zenjpeg::decoder::ErrorKind::UnsupportedFeature { feature }
                        if feature.contains("current single-scan writer cannot preserve post-scan colour markers")
                ),
                "{name}: {error}"
            );
        };
        for mode in [OutputMode::Sequential, OutputMode::Progressive] {
            let cfg = RestructureConfig {
                output_mode: mode,
                ..Default::default()
            };
            if *keep_at == 2 && mode == OutputMode::Sequential {
                assert_refusal(
                    restructure(&jpeg, &cfg, Unstoppable).expect_err("must refuse metadata loss"),
                );
                assert!(restructure(&control, &cfg, Unstoppable).is_ok());
            } else {
                let run = |j: &[u8]| restructure(j, &cfg, Unstoppable).expect("restructure");
                outputs.push((format!("restructure {mode:?}"), run(&jpeg), run(&control)));
            }
        }
        let cfg = TransformConfig {
            transform: LosslessTransform::FlipHorizontal,
            edge_handling: EdgeHandling::TrimPartialBlocks,
        };
        if *keep_at == 2 {
            assert_refusal(
                transform(&jpeg, &cfg, Unstoppable).expect_err("must refuse metadata loss"),
            );
            assert_refusal(
                apply_exif_orientation(&jpeg_exif, Unstoppable)
                    .expect_err("must refuse metadata loss"),
            );
            assert!(transform(&control, &cfg, Unstoppable).is_ok());
            assert!(apply_exif_orientation(&control_exif, Unstoppable).is_ok());
            // The identity fast path carries the entire original byte stream.
            assert_eq!(apply_exif_orientation(&jpeg, Unstoppable).unwrap(), jpeg);
        } else {
            let run = |j: &[u8]| transform(j, &cfg, Unstoppable).expect("transform");
            outputs.push(("transform FlipHorizontal".into(), run(&jpeg), run(&control)));
            let run = |j: &[u8]| apply_exif_orientation(j, Unstoppable).expect("orientation");
            outputs.push((
                "apply_exif_orientation 2".into(),
                run(&jpeg_exif),
                run(&control_exif),
            ));
        }
        #[cfg(feature = "recompress")]
        {
            use zenjpeg::recompress::{
                RecompressOptions, RecompressResult, StrategyKind, recompress,
            };
            match recompress(&jpeg, &RecompressOptions::new(70.0)) {
                Ok(RecompressResult::Recompressed {
                    bytes,
                    strategy: StrategyKind::Preserve,
                    ..
                }) => {
                    let info = Decoder::new().read_info(&bytes).expect("info");
                    f.check(info.color_space == *model, || {
                        format!("{name} recompress preserve: model {:?}", info.color_space)
                    });
                    assert_retained(&jpeg, &bytes, &format!("{name} recompress preserve"));
                    for (m, p, at) in app_positions_all(&bytes) {
                        f.check(!is_colour(m, &p) || at >= *keep_at, || {
                            format!("{name} recompress preserve: colour FF{m:02X} moved to {at}")
                        });
                    }
                }
                other => f.check(false, || {
                    format!(
                        "{name}: recompress did not take Preserve: {:?}",
                        other
                            .as_ref()
                            .map(|r| matches!(r, RecompressResult::Recompressed { .. }))
                    )
                }),
            }
        }
        for (op, out, control_out) in outputs {
            let label = format!("{name} {op}");
            assert_retained(&jpeg, &out, &label);
            let info = Decoder::new().read_info(&out).expect("info");
            f.check(info.color_space == *model, || {
                format!("{label}: model {:?}, source {model:?}", info.color_space)
            });
            for (m, p, at) in app_positions_all(&out) {
                f.check(!is_colour(m, &p) || at >= *keep_at, || {
                    format!("{label}: colour FF{m:02X} moved to position {at}")
                });
            }
            let want = decode_rgb(&control_out, ChromaUpsampling::Triangle, 1);
            let (max, first) = diff(&decode_rgb(&out, ChromaUpsampling::Triangle, 1), &want);
            f.check(max == 0, || {
                format!("{label}: pixels differ by {max} at {first:?}")
            });
        }
    }
    f.finish("post-scan and stray-byte colour markers stay inert");
}

/// Known Bugs 4, pinned: gray output from three-component frames is
/// unsupported on `decode()` and `decode_into()` alike (the two must not
/// disagree; scanline `read_rows_gray8` serves these frames).
#[test]
fn gray_output_from_colour_frames_is_unsupported() {
    let mut f = Failures::default();
    let (w, h) = (96u32, 80u32);
    let src = noise_image(w as usize, h as usize, 47);
    for progressive in [false, true] {
        let jpeg = encode_ycbcr(&src, w, h, ChromaSubsampling::Quarter, false, progressive);
        for threads in [1usize, 0] {
            let dec = decoder(ChromaUpsampling::Triangle, threads).output_format(PixelFormat::Gray);
            let a = dec.decode(&jpeg, Unstoppable).map(|_| ());
            let mut dst = vec![0u8; (w * h) as usize];
            let b = decoder(ChromaUpsampling::Triangle, threads)
                .decode_into(&jpeg, PixelFormat::Gray, &mut dst, Unstoppable)
                .map(|_| ());
            for (name, r) in [("decode", a), ("decode_into", b)] {
                match r {
                    Ok(()) => f.check(false, || {
                        format!("progressive={progressive} t{threads}: {name} to Gray succeeded; update Known Bugs 4 and this test")
                    }),
                    Err(e) => f.check(e.to_string().contains("unsupported color conversion"), || {
                        format!("progressive={progressive} t{threads}: {name} to Gray: unexpected error {e}")
                    }),
                }
            }
        }
    }
    f.finish("gray output from colour frames");
}

/// C1-13: native i16 reads on readers that hold decoded pixels (sampling
/// shapes the strip does not serve, progressive grayscale) are an explicit
/// unsupported feature, not "internal error: missing Y quantization table".
#[test]
fn native_i16_on_pixel_readers_is_unsupported() {
    let mut f = Failures::default();
    let gray = {
        let spec = Spec {
            width: 96,
            height: 80,
            comps: vec![Comp { id: 1, h: 1, v: 1 }],
            block: &with_ac,
            restart_interval: 0,
            quant: 8,
            segments: vec![],
        };
        progressive_twin(&jpeg_writer::write(&spec))
    };
    for (name, jpeg) in [
        (
            "ycbcr 4:1:1",
            frame(LAYOUTS[4], Signal::YCbCr, false, &with_ac),
        ),
        ("progressive gray", gray),
    ] {
        let mut r = Decoder::new().scanline_reader(&jpeg).expect("reader");
        let (w, cw) = (r.width() as usize, r.chroma_width().max(1) as usize);
        let (mut y, mut cb, mut cr) =
            (vec![0i16; w * 64], vec![0i16; cw * 64], vec![0i16; cw * 64]);
        match r.read_rows_ycbcr_native_i16(&mut y, w, &mut cb, &mut cr, cw, 1) {
            Ok(_) => f.check(false, || {
                format!("{name}: native i16 succeeded on a pixel reader")
            }),
            Err(e) => {
                let msg = e.to_string();
                f.check(
                    msg.contains("decoded pixels") && !msg.contains("quantization"),
                    || format!("{name}: unexpected error {msg}"),
                );
            }
        }
    }
    f.finish("native i16 on pixel readers");
}

/// Recommended in review: an XYB ICC profile on a frame signalled YCbCr
/// (JFIF, IDs 1,2,3) decodes and reports YCbCr on every entry point.
#[test]
fn xyb_icc_on_jfif_frame_is_ycbcr() {
    let mut f = Failures::default();
    let (w, h) = (160u32, 96u32);
    let src = noise_image(w as usize, h as usize, 53);
    let xyb = EncoderConfig::xyb(88, XybSubsampling::Full)
        .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
        .expect("xyb encode");
    let jpeg = as_ycbcr(&xyb);
    let info = Decoder::new().read_info(&jpeg).expect("info");
    f.check(
        info.color_space == ColorSpace::YCbCr && !info.is_xyb,
        || format!("read_info {:?} is_xyb={}", info.color_space, info.is_xyb),
    );
    let si = Decoder::new()
        .scanline_reader(&jpeg)
        .expect("reader")
        .info();
    f.check(si.color_space == ColorSpace::YCbCr && !si.is_xyb, || {
        format!(
            "ScanlineReader::info {:?} is_xyb={}",
            si.color_space, si.is_xyb
        )
    });
    for ups in UPS {
        let reference = decode_rgb(&jpeg, ups, 1);
        compare_all(
            &mut f,
            &format!("{ups:?}"),
            &reference,
            &every_entry(&jpeg, ups),
        );
    }
    f.finish("XYB ICC on a JFIF frame");
}

/// The ICC profile is collected from the whole header, up to the first SOS,
/// before marker parsing (`extract_icc_profile`), so an XYB profile placed
/// between SOF and SOS still makes an RGB-signalled frame XYB, unlike an APP0
/// or APP14 there. Pins what `XYB_ICC_HANDLING.md` and `DECODER_PATHS.md`
/// state.
#[test]
fn xyb_icc_after_sof_still_counts() {
    let mut f = Failures::default();
    let (w, h) = (160u32, 96u32);
    let src = noise_image(w as usize, h as usize, 59);
    let xyb = EncoderConfig::xyb(88, XybSubsampling::Full)
        .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
        .expect("xyb encode");
    // SOI, every header segment except APP2 up to and including SOF, then
    // the APP2 (ICC) segments, then the rest of the header and the scans.
    let segs = jpeg_writer::segments(&xyb);
    let sof = segs
        .iter()
        .position(|&(_, m, _)| matches!(m, 0xC0..=0xC2))
        .expect("SOF");
    let icc: Vec<_> = segs[..sof].iter().filter(|&&(_, m, _)| m == 0xE2).collect();
    f.check(!icc.is_empty(), || {
        "XYB encode carries no APP2 before SOF".into()
    });
    let mut moved = vec![0xFF, 0xD8];
    for &(p, m, len) in &segs[..=sof] {
        if m != 0xE2 {
            moved.extend_from_slice(&xyb[p..p + len]);
        }
    }
    for &&(p, _, len) in &icc {
        moved.extend_from_slice(&xyb[p..p + len]);
    }
    moved.extend_from_slice(&xyb[segs[sof + 1].0..]);
    f.check(
        jpeg_writer::segments(&moved)[..=sof - icc.len()]
            .iter()
            .all(|&(_, m, _)| m != 0xE2),
        || "APP2 still before SOF".into(),
    );
    let info = Decoder::new().read_info(&moved).expect("info");
    f.check(info.color_space == ColorSpace::Xyb && info.is_xyb, || {
        format!("read_info {:?} is_xyb={}", info.color_space, info.is_xyb)
    });
    let si = Decoder::new()
        .scanline_reader(&moved)
        .expect("reader")
        .info();
    f.check(si.color_space == ColorSpace::Xyb, || {
        format!("ScanlineReader::info {:?}", si.color_space)
    });
    for ups in UPS {
        let reference = decode_rgb(&xyb, ups, 1);
        compare_all(
            &mut f,
            &format!("{ups:?}"),
            &reference,
            &every_entry(&moved, ups),
        );
    }
    f.finish("XYB ICC after SOF");
}

/// Grayscale frames decode with the Libjpeg IDCT whatever `IdctMethod` says
/// (#154). The scanline strip and wave readers got the un-normalised method,
/// so with `IdctMethod::Jpegli` their gray8 differed from `decode()`.
#[test]
fn gray_frames_use_libjpeg_idct_on_every_path() {
    use zenjpeg::decode::IdctMethod;
    let mut f = Failures::default();
    let mut spec = Spec {
        width: FW,
        height: FH,
        comps: vec![Comp { id: 1, h: 1, v: 1 }],
        block: &with_ac,
        restart_interval: 0,
        quant: 8,
        segments: vec![],
    };
    spec.restart_interval = spec.mcu_cols() as u16;
    let jpeg = jpeg_writer::write(&spec);
    let reference = decode_fmt(&jpeg, ChromaUpsampling::Triangle, 1, PixelFormat::Gray);
    let native_reference =
        plane_read_with(&decoder(ChromaUpsampling::Triangle, 1), &jpeg, "native_i16")
            .expect("native i16");
    for threads in [1usize, 0] {
        let dec = decoder(ChromaUpsampling::Triangle, threads).idct_method(IdctMethod::Jpegli);
        // Native i16 planes: at t0 with `parallel` these come from the planar
        // wave, which picks its own IDCT.
        let native = plane_read_with(&dec, &jpeg, "native_i16").expect("native i16");
        let first = native_reference
            .iter()
            .zip(&native)
            .position(|(a, b)| a != b);
        f.check(native == native_reference, || {
            format!(
                "t{threads} Jpegli: native i16 differs from the default-method read \
                 (first at {first:?}, lengths {} vs {})",
                native.len(),
                native_reference.len()
            )
        });
        let decoded = dec
            .clone()
            .output_format(PixelFormat::Gray)
            .decode(&jpeg, Unstoppable)
            .expect("decode")
            .into_pixels_u8()
            .expect("u8");
        for (name, got) in [
            ("decode", decoded),
            ("scanline gray8", scanline_gray8(&dec, &jpeg)),
        ] {
            let (max, first) = diff(&reference, &got);
            f.check(max == 0, || {
                format!("t{threads} Jpegli: {name} differs from the default-method decode by {max} at {first:?}")
            });
        }
    }
    f.finish("gray IDCT method");
}

/// Wave-only readers report the frame's subsampling (they reported 4:2:0
/// for 4:2:2 frames).
#[test]
fn wave_readers_report_subsampling() {
    use zenjpeg::types::Subsampling;
    for (layout, want) in [
        (LAYOUTS[1], Subsampling::S420),
        (LAYOUTS[2], Subsampling::S422),
    ] {
        let jpeg = frame(layout, Signal::YCbCr, true, &with_ac);
        let r = decoder(ChromaUpsampling::NearestNeighbor, 0)
            .scanline_reader(&jpeg)
            .expect("reader");
        assert_eq!(r.info().subsampling, want, "{}", layout.name);
    }
}

/// `read_rows_ycbcr_f32` takes raw slices: rows that do not fit are not read
/// (no panic), and planes smaller than one row, or a stride below the width,
/// are argument errors whether or not the reader has rows left.
#[test]
fn ycbcr_f32_read_respects_plane_lengths() {
    use zenjpeg::decoder::ErrorKind;
    let jpeg = encode_ycbcr(
        &noise_image(64, 48, 59),
        64,
        48,
        ChromaSubsampling::Quarter,
        false,
        false,
    );
    let mut r = Decoder::new().scanline_reader(&jpeg).expect("reader");
    let (mut y, mut cb, mut cr) = (vec![0f32; 64 * 3], vec![0f32; 64 * 3], vec![0f32; 64 * 3]);
    let n = r
        .read_rows_ycbcr_f32(&mut y, &mut cb, &mut cr, 64, 48)
        .expect("short planes");
    assert_eq!(n, 3, "rows read into three-row planes");
    // The last row needs only the width: 2 * 70 + 64 holds three rows at stride 70.
    let len = 2 * 70 + 64;
    let (mut y, mut cb, mut cr) = (vec![0f32; len], vec![0f32; len], vec![0f32; len + 5]);
    let n = r
        .read_rows_ycbcr_f32(&mut y, &mut cb, &mut cr, 70, 48)
        .expect("strided short planes");
    assert_eq!(n, 3, "rows read into planes holding three strided rows");

    let mut tiny = vec![0f32; 10];
    let is_short = |e: &zenjpeg::decoder::DecodeError| {
        matches!(
            e.kind(),
            ErrorKind::InvalidBufferSize {
                expected: 64,
                actual: 10
            }
        )
    };
    let mut r = Decoder::new().scanline_reader(&jpeg).expect("reader");
    let e = r
        .read_rows_ycbcr_f32(&mut tiny.clone(), &mut tiny.clone(), &mut tiny, 64, 48)
        .expect_err("tiny planes on a fresh reader");
    assert!(is_short(&e), "fresh reader: {e}");
    let e = r
        .read_rows_ycbcr_f32(&mut y, &mut cb, &mut cr, 63, 48)
        .expect_err("stride below the width");
    assert!(
        matches!(
            e.kind(),
            ErrorKind::StrideTooSmall {
                width: 64,
                stride: 63
            }
        ),
        "stride: {e}"
    );
    // Drain the reader, then ask again with tiny planes.
    let mut all = vec![0u8; 64 * 3 * 48];
    let mut r = Decoder::new().scanline_reader(&jpeg).expect("reader");
    let rows = r
        .read_rows_rgb8(ImgRefMut::new(&mut all, 64 * 3, 48))
        .expect("drain");
    assert_eq!(rows, 48);
    assert!(r.is_finished());
    let e = r
        .read_rows_ycbcr_f32(&mut tiny.clone(), &mut tiny.clone(), &mut tiny, 64, 48)
        .expect_err("tiny planes on a finished reader");
    assert!(is_short(&e), "finished reader: {e}");
    assert_eq!(
        r.read_rows_ycbcr_f32(&mut tiny.clone(), &mut tiny.clone(), &mut tiny, 64, 0)
            .expect("zero rows requested"),
        0
    );
}

/// One f32 or luma read over a whole file, as bit patterns (gray8 as bytes).
fn plane_read(
    jpeg: &[u8],
    ups: ChromaUpsampling,
    threads: usize,
    which: &str,
) -> Result<Vec<u32>, String> {
    plane_read_with(&decoder(ups, threads), jpeg, which)
}

/// [`plane_read`] with a configured decoder. Also `rgb8` (bytes) and
/// `native_i16` (Y rows, then every Cb and Cr column `chroma_width` names,
/// MCU row by MCU row, as i16 bit patterns).
fn plane_read_with(dec: &Decoder, jpeg: &[u8], which: &str) -> Result<Vec<u32>, String> {
    let mut r = dec.scanline_reader(jpeg).map_err(|e| e.to_string())?;
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut row = 0;
    let mut out = Vec::new();
    while row < h {
        let n = 7.min(h - row);
        let got = match which {
            "rgb8" => {
                let mut b = vec![0u8; w * 3 * n];
                let k = r
                    .read_rows_rgb8(ImgRefMut::new(&mut b, w * 3, n))
                    .map_err(|e| e.to_string())?;
                out.extend(b[..w * 3 * k].iter().map(|&v| v as u32));
                k
            }
            "native_i16" => {
                let cw = r.chroma_width() as usize;
                let mh = r.luma_rows_per_mcu();
                let mut y = vec![0i16; w * mh];
                let (mut cb, mut cr) = (vec![0i16; cw.max(1) * mh], vec![0i16; cw.max(1) * mh]);
                let (ly, lc) = r
                    .read_rows_ycbcr_native_i16(&mut y, w, &mut cb, &mut cr, cw, 1)
                    .map_err(|e| e.to_string())?;
                out.extend(y[..w * ly].iter().map(|&v| v as u16 as u32));
                out.extend(cb[..cw * lc].iter().map(|&v| v as u16 as u32));
                out.extend(cr[..cw * lc].iter().map(|&v| v as u16 as u32));
                ly
            }
            "gray8" => {
                let mut b = vec![0u8; w * n];
                let k = r
                    .read_rows_gray8(ImgRefMut::new(&mut b, w, n))
                    .map_err(|e| e.to_string())?;
                out.extend(b[..w * k].iter().map(|&v| v as u32));
                k
            }
            "gray_f32" | "gray_linear_f32" => {
                let mut b = vec![0f32; w * n];
                let img = ImgRefMut::new(&mut b, w, n);
                let k = if which == "gray_f32" {
                    r.read_rows_gray_f32(img)
                } else {
                    r.read_rows_gray_linear_f32(img)
                }
                .map_err(|e| e.to_string())?;
                out.extend(b[..w * k].iter().map(|v| v.to_bits()));
                k
            }
            "rgba_f32" => {
                let mut b = vec![0f32; w * 4 * n];
                let k = r
                    .read_rows_rgba_f32(ImgRefMut::new(&mut b, w * 4, n))
                    .map_err(|e| e.to_string())?;
                out.extend(b[..w * 4 * k].iter().map(|v| v.to_bits()));
                k
            }
            "ycbcr_f32" => {
                let (mut y, mut cb, mut cr) =
                    (vec![0f32; w * n], vec![0f32; w * n], vec![0f32; w * n]);
                let k = r
                    .read_rows_ycbcr_f32(&mut y, &mut cb, &mut cr, w, n)
                    .map_err(|e| e.to_string())?;
                for i in 0..w * k {
                    out.extend([y[i].to_bits(), cb[i].to_bits(), cr[i].to_bits()]);
                }
                k
            }
            other => panic!("unknown read {other}"),
        };
        if got == 0 {
            return Err(format!("reader stalled at row {row}"));
        }
        row += got;
    }
    Ok(out)
}

/// C1-17: the wave kernels run the clamped integer IDCT and never deblock,
/// while the strip honours the output target (the unclamped IDCT under f32
/// targets) and Boundary4Tap. Readers under either now stay on the strip, so
/// `rgb8` and native i16 equal `num_threads(1)` whatever the target and
/// deblock mode. Baseline 4:2:0 and 4:2:2 with MCU-row DRI and >= 1024 MCUs,
/// 0/255 noise at q40 (the unclamped Y plane is asserted to leave [0, 255]),
/// both upsamplers (nearest: wave-only reader; triangle: planar attach).
/// Route proof under `parallel`: with the 8-bit target and no deblocking the
/// nearest reader of a YCbCr frame is wave-only (`gray8` refused); with an f32
/// target or a deblock mode it serves `gray8`.
#[test]
fn wave_route_follows_output_target_and_deblock() {
    use zenjpeg::decode::DeblockMode;
    let mut f = Failures::default();
    let (w, h) = (512u32, 512u32);
    let src: Vec<u8> = (0..w as usize * h as usize * 3)
        .map(|i| {
            if hash(0x5A7, i, 0, 0) % 2 == 0 {
                0
            } else {
                255
            }
        })
        .collect();
    for sub in [
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
    ] {
        let jpeg = EncoderConfig::ycbcr(40, sub)
            .progressive(false)
            .restart_mcu_rows(1)
            .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
            .expect("encode");
        for (target, deblock) in [
            (OutputTarget::Srgb8, DeblockMode::Off),
            (OutputTarget::SrgbF32, DeblockMode::Off),
            (OutputTarget::LinearF32, DeblockMode::Off),
            (OutputTarget::Srgb8, DeblockMode::Boundary4Tap),
            (OutputTarget::Srgb8, DeblockMode::AutoStreamable),
            (OutputTarget::SrgbF32, DeblockMode::Boundary4Tap),
        ] {
            for ups in UPS {
                let label = format!("{sub:?} {target:?} {deblock:?} {ups:?}");
                let dec = |t: usize| decoder(ups, t).output_target(target).deblock(deblock);
                for which in ["rgb8", "native_i16"] {
                    let (a, b) = (
                        plane_read_with(&dec(1), &jpeg, which),
                        plane_read_with(&dec(0), &jpeg, which),
                    );
                    match (&a, &b) {
                        (Ok(a), Ok(b)) => f.check(a == b, || {
                            format!("{label}: {which} threads 0 differs from threads 1")
                        }),
                        _ => f.check(false, || format!("{label}: {which}: t1 {a:?} t0 {b:?}")),
                    }
                }
                if target == OutputTarget::SrgbF32 && deblock == DeblockMode::Off {
                    let y = plane_read_with(&dec(1), &jpeg, "native_i16").unwrap_or_default();
                    let n = w as usize * h as usize;
                    let unclamped = y[..n.min(y.len())]
                        .iter()
                        .map(|&v| v as u16 as i16)
                        .any(|v| !(0..=255).contains(&v));
                    f.check(unclamped, || {
                        format!(
                            "{label}: the Y plane never leaves [0, 255]; the regime is not reached"
                        )
                    });
                }
                let wave_expected = cfg!(feature = "parallel")
                    && ups == ChromaUpsampling::NearestNeighbor
                    && target == OutputTarget::Srgb8
                    && deblock == DeblockMode::Off;
                let gray = plane_read_with(&dec(0), &jpeg, "gray8");
                f.check(gray.is_err() == wave_expected, || {
                    format!("{label}: threads 0 gray8 {:?} (wave-only reader expected: {wave_expected})",
                        gray.as_ref().map(|v| v.len()))
                });
            }
        }
    }
    f.finish("wave route vs output target and deblock");
}

/// C1-4 and Known Bugs 3: f32 and luma reads on wave-parallel readers
/// (nearest 4:2:0 / 4:2:2, MCU-row DRI, threads 0 under `parallel`).
/// - RGB frames: the wave holds the same 8-bit R, G, B the strip derives
///   these reads from, so every read equals `num_threads(1)` bit for bit.
/// - YCbCr frames: f32 and luma reads need the planes, which a wave-only
///   reader does not keep, so they are an explicit error naming
///   `num_threads(1)` (`read_rows_rgba_f32` used to convert rounded RGB and
///   disagree with the strip). The internal callers that need such reads
///   (`decode_rows(Gray)`, `decode_rows_f32`) build their reader without the
///   wave-only route and match `num_threads(1)`.
/// - Native i16 (`read_rows_ycbcr_native_i16`) is served from the planar wave
///   (wave-only nearest readers, and triangle readers through the attached
///   wave state) and must equal `num_threads(1)` in every column
///   `chroma_width` names, the MCU padding included (C1-19: the planar wave
///   copied raw padding where the strip repeats the last real column; these
///   frames are 1001 wide, so chroma has padding columns). RGB frames have no
///   YCbCr planes: an error at both thread counts.
///
/// Without `parallel` every reader is the strip, and every read must match.
#[test]
fn wave_reader_f32_and_plane_reads() {
    let mut f = Failures::default();
    let ups = ChromaUpsampling::NearestNeighbor;
    for layout in [LAYOUTS[1], LAYOUTS[2]] {
        for signal in [Signal::Rgb, Signal::YCbCr] {
            let jpeg = frame(layout, signal, true, &with_ac);
            let label = format!("{} {signal:?}", layout.name);
            let refused = cfg!(feature = "parallel") && signal == Signal::YCbCr;
            for which in [
                "gray8",
                "gray_f32",
                "gray_linear_f32",
                "rgba_f32",
                "ycbcr_f32",
            ] {
                let seq = plane_read(&jpeg, ups, 1, which);
                let wave = plane_read(&jpeg, ups, 0, which);
                match (&seq, &wave) {
                    (Ok(a), Ok(b)) if !refused => f.check(a == b, || {
                        format!("{label}: {which} threads 0 differs from threads 1")
                    }),
                    (Ok(_), Err(e)) if refused => f.check(e.contains("num_threads(1)"), || {
                        format!("{label}: {which} on the wave-only reader: unexpected error {e}")
                    }),
                    _ => f.check(false, || {
                        format!("{label}: {which}: threads 1 {:?}, threads 0 {:?} (refusal expected: {refused})",
                            seq.as_ref().map(|v| v.len()), wave.as_ref().map(|v| v.len()))
                    }),
                }
            }
            for ups in UPS {
                let (t1, t0) = (
                    plane_read(&jpeg, ups, 1, "native_i16"),
                    plane_read(&jpeg, ups, 0, "native_i16"),
                );
                match (signal, &t1, &t0) {
                    (Signal::YCbCr, Ok(a), Ok(b)) => f.check(a == b, || {
                        let first = a.iter().zip(b).position(|(x, y)| x != y);
                        format!("{label} {ups:?}: native i16 threads 0 differs from threads 1 at {first:?}")
                    }),
                    (Signal::Rgb, Err(_), Err(_)) => {}
                    _ => f.check(false, || {
                        format!("{label} {ups:?}: native i16 threads 1 {:?}, threads 0 {:?}",
                            t1.as_ref().map(|v| v.len()), t0.as_ref().map(|v| v.len()))
                    }),
                }
            }
            // The internal callers avoid the wave-only route for these reads.
            let rows_f32 = |threads: usize| {
                let mut v = Vec::new();
                decoder(ups, threads)
                    .decode_rows_f32(
                        &jpeg,
                        PixelFormat::RgbaF32,
                        |row| {
                            v.extend(row.as_slice().iter().map(|x| x.to_bits()));
                            Ok(())
                        },
                        Unstoppable,
                    )
                    .map(|_| v)
                    .map_err(|e| e.to_string())
            };
            let rows_gray = |threads: usize| {
                let mut v = Vec::new();
                decoder(ups, threads)
                    .decode_rows(
                        &jpeg,
                        PixelFormat::Gray,
                        |row| {
                            v.extend_from_slice(row.as_gray());
                            Ok(())
                        },
                        Unstoppable,
                    )
                    .map(|_| v)
                    .map_err(|e| e.to_string())
            };
            f.check(rows_f32(0).is_ok() && rows_f32(0) == rows_f32(1), || {
                format!(
                    "{label}: decode_rows_f32 threads 0 {:?} vs threads 1",
                    rows_f32(0).map(|v| v.len())
                )
            });
            f.check(rows_gray(0).is_ok() && rows_gray(0) == rows_gray(1), || {
                format!(
                    "{label}: decode_rows(Gray) threads 0 {:?} vs threads 1",
                    rows_gray(0).map(|v| v.len())
                )
            });
        }
    }
    f.finish("wave f32 and plane reads");
}

/// Mutation reach (r3): the per-component path's u8 and f32 arms (XYB,
/// YCbCr shapes no fast path serves, gray under an f32 target) were checked
/// only against `decode()`, which runs the same arm. The u8 and f32 outputs
/// come from separate arms (`three_chan_planes_to_rgb_u8` and the f32
/// dispatch; for gray frames the streaming path and the f32 gray arms), so
/// they anchor each other: they must agree within the rounding of the integer
/// vs f32 reconstruction. Bounds are the measured maxima (2026-10-11): XYB 0,
/// gray 1, YCbCr DC-only 1, YCbCr with AC 3 (the integer islow IDCT rounds each
/// sample, f32 does not, and the chroma terms scale a 1-level chroma difference
/// by up to 1.77). A corrupted arm moves a byte by 64.
#[test]
fn per_component_u8_and_f32_outputs_agree() {
    let mut f = Failures::default();
    let f32_px = |jpeg: &[u8], ups: ChromaUpsampling, fmt: PixelFormat| -> Vec<u8> {
        Decoder::new()
            .chroma_upsampling(ups)
            .output_format(fmt)
            .output_target(OutputTarget::SrgbF32)
            .decode(jpeg, Unstoppable)
            .expect("f32 decode")
            .into_pixels_f32()
            .expect("f32 pixels")
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect()
    };
    let mut files: Vec<(String, Vec<u8>, &[PixelFormat], u8)> = Vec::new();
    let (w, h) = (96u32, 80u32);
    let src = noise_image(w as usize, h as usize, 83);
    for (name, sub) in [
        ("xyb full", XybSubsampling::Full),
        ("xyb bquarter", XybSubsampling::BQuarter),
    ] {
        let jpeg = EncoderConfig::xyb(88, sub)
            .encode_bytes(&src, w, h, PixelLayout::Rgb8Srgb)
            .expect("xyb encode");
        files.push((name.into(), jpeg, &[PixelFormat::Rgb], 0));
    }
    for layout in [LAYOUTS[4], LAYOUTS[5], LAYOUTS[6]] {
        for (cname, content) in CONTENTS {
            files.push((
                format!("ycbcr {} {cname}", layout.name),
                frame(layout, Signal::YCbCr, false, content),
                &[PixelFormat::Rgb],
                if cname == "dc-only" { 1 } else { 3 },
            ));
        }
    }
    for (cname, content) in CONTENTS {
        let spec = Spec {
            width: 157,
            height: 93,
            comps: vec![Comp { id: 1, h: 1, v: 1 }],
            block: content,
            restart_interval: 0,
            quant: 8,
            segments: vec![],
        };
        files.push((
            format!("gray {cname}"),
            jpeg_writer::write(&spec),
            &[PixelFormat::Gray, PixelFormat::Rgb],
            1,
        ));
    }
    for (name, jpeg, formats, bound) in &files {
        for &fmt in *formats {
            for ups in UPS {
                let u8s = decode_fmt(jpeg, ups, 1, fmt);
                let f32s = f32_px(jpeg, ups, fmt);
                let (max, first) = diff(&u8s, &f32s);
                eprintln!("{name} {fmt:?} {ups:?}: u8 vs f32 max {max}");
                f.check(max <= *bound, || {
                    format!("{name} {fmt:?} {ups:?}: u8 and f32 differ by {max} at {first:?} (bound {bound})")
                });
            }
        }
    }
    f.finish("per-component u8 vs f32");
}

/// Mutation reach (r3) + review S3: CMYK and YCCK frames, including non-zero
/// APP14 transform values (`1`, `2`, `3`, `255`), which both jpegli and
/// libjpeg-turbo treat as YCCK. DC-only blocks with quantizer 8 make every
/// plane sample exactly `dc_value`, so the outputs are known: CMYK output is
/// the planes; YCCK output is 255 minus the BT.601 conversion of (Y, Cb, Cr),
/// with K passed through; RGB output is the Adobe-inverted product `c * k / 255`
/// (rounded) of the CMYK output. Also cross-checked against `mozjpeg_sys`
/// (libjpeg-turbo).
#[test]
fn cmyk_and_ycck_known_answer() {
    #[cfg(not(target_arch = "wasm32"))]
    fn mozjpeg_cmyk(data: &[u8]) -> (bool, Vec<u8>) {
        use mozjpeg_sys::*;
        use std::mem;
        unsafe {
            let mut err: jpeg_error_mgr = mem::zeroed();
            jpeg_std_error(&mut err);
            let mut ci: jpeg_decompress_struct = mem::zeroed();
            ci.common.err = &mut err;
            jpeg_create_decompress(&mut ci);
            jpeg_mem_src(&mut ci, data.as_ptr(), data.len() as _);
            assert_eq!(jpeg_read_header(&mut ci, 1), 1, "mozjpeg read_header");
            let is_ycck = ci.jpeg_color_space == J_COLOR_SPACE::JCS_YCCK;
            assert!(
                ci.jpeg_color_space == J_COLOR_SPACE::JCS_CMYK || is_ycck,
                "unexpected mozjpeg jpeg_color_space: {:?}",
                ci.jpeg_color_space
            );
            ci.out_color_space = J_COLOR_SPACE::JCS_CMYK;
            jpeg_start_decompress(&mut ci);
            let (w, h) = (ci.output_width, ci.output_height);
            let stride = w as usize * ci.output_components as usize;
            let mut out = vec![0u8; h as usize * stride];
            while ci.output_scanline < h {
                let off = ci.output_scanline as usize * stride;
                let mut p = out[off..].as_mut_ptr();
                jpeg_read_scanlines(&mut ci, &mut p, 1);
            }
            jpeg_finish_decompress(&mut ci);
            jpeg_destroy_decompress(&mut ci);
            (is_ycck, out)
        }
    }

    let mut f = Failures::default();
    let (w, h) = (40u32, 24u32);
    for (name, segments, ycck) in [
        ("cmyk, no APP14", vec![], false),
        ("cmyk, APP14 t0", vec![Segment::PreFrame(app14(0))], false),
        ("ycck, APP14 t1", vec![Segment::PreFrame(app14(1))], true),
        ("ycck, APP14 t2", vec![Segment::PreFrame(app14(2))], true),
        ("ycck, APP14 t3", vec![Segment::PreFrame(app14(3))], true),
        (
            "ycck, APP14 t255",
            vec![Segment::PreFrame(app14(255))],
            true,
        ),
    ] {
        let spec = Spec {
            width: w,
            height: h,
            comps: (1..=4).map(|id| Comp { id, h: 1, v: 1 }).collect(),
            block: &dc_only,
            restart_interval: 0,
            quant: 8,
            segments,
        };
        let jpeg = jpeg_writer::write(&spec);
        #[cfg(not(target_arch = "wasm32"))]
        let (moz_ycck, moz_cmyk) = mozjpeg_cmyk(&jpeg);
        #[cfg(not(target_arch = "wasm32"))]
        f.check(moz_ycck == ycck, || {
            format!("{name}: mozjpeg is_ycck={moz_ycck}, expected {ycck}")
        });
        for ups in UPS {
            let cmyk = decode_fmt(&jpeg, ups, 1, PixelFormat::Cmyk);
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (max_moz, first_moz) = diff(&cmyk, &moz_cmyk);
                f.check(max_moz <= 1, || {
                    format!(
                        "{name} {ups:?}: CMYK vs mozjpeg max diff {max_moz} at {first_moz:?}"
                    )
                });
            }
            let rgb = decode_fmt(&jpeg, ups, 1, PixelFormat::Rgb);
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let i = y * w as usize + x;
                    let s = |c: usize| dc_value(c, x / 8, y / 8) as f64;
                    let want: [f64; 4] = if ycck {
                        let (yy, cb, cr) = (s(0), s(1) - 128.0, s(2) - 128.0);
                        let conv = |v: f64| 255.0 - v.round().clamp(0.0, 255.0);
                        [
                            conv(yy + 1.402 * cr),
                            conv(yy - 0.344_136 * cb - 0.714_136 * cr),
                            conv(yy + 1.772 * cb),
                            s(3),
                        ]
                    } else {
                        [s(0), s(1), s(2), s(3)]
                    };
                    let got = &cmyk[i * 4..i * 4 + 4];
                    // CMY within 1 (YCCK's BT.601 rounding), K exact.
                    let ok = (0..3).all(|c| (got[c] as f64 - want[c]).abs() <= 1.0)
                        && got[3] as f64 == want[3];
                    f.check(ok, || {
                        format!("{name} {ups:?} ({x},{y}): CMYK {got:?}, want {want:?}")
                    });
                    let k = got[3] as u32;
                    let want_rgb: Vec<u8> = (0..3)
                        .map(|c| ((got[c] as u32 * k + 127) / 255) as u8)
                        .collect();
                    f.check(rgb[i * 3..i * 3 + 3] == want_rgb[..], || {
                        format!(
                            "{name} {ups:?} ({x},{y}): RGB {:?}, want {want_rgb:?} from CMYK {got:?}",
                            &rgb[i * 3..i * 3 + 3]
                        )
                    });
                }
            }
        }
    }
    f.finish("CMYK / YCCK known answer");
}

/// Review round 3, S6: when a component's SOF `Tq` references an undefined
/// quantization table slot, the fused parallel paths (4:4:4, box, h2v1, 4:2:0
/// fancy) and `WaveParallelState::new` must return an `Err` rather than panic
/// with `.unwrap()` on `None`.
#[test]
fn missing_quant_table_returns_error_on_every_path() {
    let mut f = Failures::default();
    for layout in [LAYOUTS[0], LAYOUTS[1], LAYOUTS[2]] {
        let mut jpeg = frame(layout, Signal::YCbCr, true, &dc_only);
        // Point component 0's Tq byte in SOF0 at slot 2 (only slot 0 is defined).
        let (sof_pos, _, _) = jpeg_writer::segments(&jpeg)
            .into_iter()
            .find(|&(_, m, _)| m == 0xC0)
            .expect("SOF0");
        assert_eq!(jpeg[sof_pos + 12], 0, "component 0 Tq is initially 0");
        jpeg[sof_pos + 12] = 2;

        for ups in UPS {
            for threads in [1usize, 0] {
                let label = format!("{} {ups:?} t{threads}", layout.name);
                let dec = decoder(ups, threads);
                f.check(dec.decode(&jpeg, Unstoppable).is_err(), || {
                    format!("{label}: decode succeeded with missing quant table")
                });
                let mut dst = vec![0u8; (FW * FH * 3) as usize];
                f.check(
                    dec.decode_into(&jpeg, PixelFormat::Rgb, &mut dst, Unstoppable)
                        .is_err(),
                    || format!("{label}: decode_into succeeded with missing quant table"),
                );
                let check_scanline = |owned: bool| -> Result<(), String> {
                    let mut r = if owned {
                        dec.scanline_reader_cow(Cow::Owned(jpeg.clone()))
                    } else {
                        dec.scanline_reader(&jpeg)
                    }
                    .map_err(|e| e.to_string())?;
                    let mut row = vec![0u8; FW as usize * 3 * 8];
                    r.read_rows_rgb8(ImgRefMut::new(&mut row, FW as usize * 3, 8))
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                };
                for owned in [false, true] {
                    f.check(check_scanline(owned).is_err(), || {
                        format!(
                            "{label}: scanline(owned={owned}) succeeded with missing quant table"
                        )
                    });
                }
            }
        }
    }
    f.finish("missing quant table");
}

/// Review round 3, S4: DC-only blocks (`coeff_count <= 1`) whose dequantized DC
/// exceeds the 16-bit shift domain (`|DC * quant[0]| > 8191`) wrap in
/// `idct_int_dc_only_w16` (matching libjpeg-turbo's AVX2 NASM `jidctint`) and
/// saturate in `idct_int_dc_only`. `StripProcessor::idct_block` and the 7 DC
/// fast-path sites in `fused_parallel.rs` previously called `idct_int_dc_only`
/// directly, disagreeing with `baseline_streaming`, `scan`, and `fast_integer`
/// by up to 255 levels on x86-64-v3 under `IdctMethod::Libjpeg`.
#[test]
fn extreme_dc_only_blocks_agree_across_paths() {
    use zenjpeg::decode::IdctMethod;

    let mut f = Failures::default();
    let extreme_dc = |c: usize, bx: usize, by: usize| -> [i16; 64] {
        let mut b = [0i16; 64];
        // With quant = 64, |200 * 64| = 12800 > 8191: wraps in w16 (`<< 2` in i16
        // overflows to -14336 -> 0), saturates in i32 (-> 255).
        b[0] = if (c + bx + by) % 2 == 0 { 200 } else { -200 };
        b
    };

    for layout in [LAYOUTS[0], LAYOUTS[1], LAYOUTS[2]] {
        let mut spec = Spec {
            width: layout.size.0,
            height: layout.size.1,
            comps: (0..3)
                .map(|i| Comp {
                    id: (i + 1) as u8,
                    h: layout.samp[i].0,
                    v: layout.samp[i].1,
                })
                .collect(),
            block: &extreme_dc,
            restart_interval: 0,
            quant: 64,
            segments: vec![Segment::PreFrame(jfif())],
        };
        spec.restart_interval = spec.mcu_cols() as u16;
        let jpeg = jpeg_writer::write(&spec);

        for method in [IdctMethod::Libjpeg, IdctMethod::Jpegli] {
            for ups in UPS {
                let dec_with = |threads: usize| {
                    Decoder::new()
                        .idct_method(method)
                        .chroma_upsampling(ups)
                        .num_threads(threads)
                };
                let ref_px = dec_with(1)
                    .decode(&jpeg, Unstoppable)
                    .expect("decode t1")
                    .into_pixels_u8()
                    .expect("u8");

                let par_px = dec_with(0)
                    .decode(&jpeg, Unstoppable)
                    .expect("decode t0")
                    .into_pixels_u8()
                    .expect("u8");
                let (max_par, first_par) = diff(&ref_px, &par_px);
                f.check(max_par == 0, || {
                    format!(
                        "{} {method:?} {ups:?}: decode(t0) vs decode(t1) max diff {max_par} at {first_par:?}",
                        layout.name
                    )
                });

                for threads in [1usize, 0] {
                    let mut into_px = vec![0u8; ref_px.len()];
                    dec_with(threads)
                        .decode_into(&jpeg, PixelFormat::Rgb, &mut into_px, Unstoppable)
                        .expect("decode_into");
                    let (max_into, first_into) = diff(&ref_px, &into_px);
                    f.check(max_into == 0, || {
                        format!(
                            "{} {method:?} {ups:?} t{threads}: decode_into vs decode(t1) max diff {max_into} at {first_into:?}",
                            layout.name
                        )
                    });

                    for owned in [false, true] {
                        let dec = dec_with(threads);
                        let mut r = if owned {
                            dec.scanline_reader_cow(Cow::Owned(jpeg.clone()))
                        } else {
                            dec.scanline_reader(&jpeg)
                        }
                        .expect("scanline_reader");
                        let (w, h) = (r.width() as usize, r.height() as usize);
                        let stride = w * 3;
                        let mut sl_px = vec![0u8; stride * h];
                        let mut row = 0;
                        while row < h {
                            let n = 7.min(h - row);
                            let img = ImgRefMut::new(
                                &mut sl_px[row * stride..(row + n) * stride],
                                stride,
                                n,
                            );
                            row += r.read_rows_rgb8(img).expect("read_rows_rgb8");
                        }
                        let (max_sl, first_sl) = diff(&ref_px, &sl_px);
                        f.check(max_sl == 0, || {
                            format!(
                                "{} {method:?} {ups:?} t{threads} owned={owned}: scanline vs decode(t1) max diff {max_sl} at {first_sl:?}",
                                layout.name
                            )
                        });
                    }
                }
            }
        }
    }
    f.finish("extreme DC-only blocks agree across paths");
}
