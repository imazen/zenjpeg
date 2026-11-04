//! `IdctMethod::LibjpegFast` is libjpeg-turbo's fast-integer IDCT (`JDCT_IFAST`)
//! as its x86-64 SSE2 kernel computes it, with the turbo upsampling biases and
//! colour tables. Every u8 decode path must honour it identically (streaming
//! `decode()`, pull-based `scanline_reader()`, the fused-parallel path), across
//! all subsampling modes, baseline / progressive, 3-component and grayscale
//! sources, and both chroma regimes (high-frequency and DC-only chroma, see
//! `libjpeg_idct_all_paths_parity.rs` for why both matter).
//!
//! With `--features __ffi-tests` every path is also asserted byte-identical to
//! real libjpeg-turbo (`mozjpeg-sys`) decoding with `JDCT_IFAST` and fancy
//! upsampling: the decode `djpeg -dct fast` and TensorFlow's
//! `decode_jpeg(dct_method="INTEGER_FAST")` produce. A sanity check asserts the
//! fast kernel is actually in use: its output differs from `Libjpeg` (islow) on
//! these images.
//!
//! Run with `--features parallel` to exercise the real multi-threaded path.

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder, IdctMethod};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Block-banded RGB with saturated red/blue and a high-frequency green ramp.
fn test_image(w: usize, h: usize) -> Vec<u8> {
    let mut d = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            d[i] = if (y / 8) % 2 == 0 { 255 } else { 0 };
            d[i + 1] = ((x * 3 + y * 7) % 200) as u8;
            d[i + 2] = ((x * 5 + y * 11) % 240) as u8;
        }
    }
    d
}

/// Chroma flat within each 8×8 chroma block (DC-only chroma), luma detailed.
fn smooth_chroma_image(w: usize, h: usize) -> Vec<u8> {
    let mut d = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = match (y / 16) % 4 {
                0 => (200i32, 90i32, 70i32),
                1 => (70, 180, 90),
                2 => (80, 90, 210),
                _ => (190, 190, 70),
            };
            let luma = ((x * 7 + y * 3) % 90) as i32 / 3;
            d[i] = (r + luma).clamp(0, 255) as u8;
            d[i + 1] = (g + luma).clamp(0, 255) as u8;
            d[i + 2] = (b + luma).clamp(0, 255) as u8;
        }
    }
    d
}

const IMAGE_KINDS: [(&str, fn(usize, usize) -> Vec<u8>); 2] = [
    ("hf-chroma", test_image),
    ("smooth-chroma", smooth_chroma_image),
];

const SUBSAMPLINGS: [(&str, ChromaSubsampling); 4] = [
    ("4:2:0", ChromaSubsampling::Quarter),
    ("4:2:2", ChromaSubsampling::HalfHorizontal),
    ("4:4:0", ChromaSubsampling::HalfVertical),
    ("4:4:4", ChromaSubsampling::None),
];

/// Non-MCU-aligned, aligned, large (engages the parallel path), and the even
/// non-aligned widths that caught the 4:2:2 right-edge regression (#188).
const SIZES: [(usize, usize); 5] = [(67, 45), (128, 96), (256, 192), (74, 58), (130, 47)];

fn encode(px: &[u8], w: u32, h: u32, sub: ChromaSubsampling, progressive: bool) -> Vec<u8> {
    let mut e = EncoderConfig::ycbcr(85.0, sub)
        .progressive(progressive)
        .allow_16bit_quant_tables(false)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    e.push_packed(px, Unstoppable).expect("push");
    e.finish().expect("finish")
}

fn encode_gray(px: &[u8], w: u32, h: u32, progressive: bool) -> Vec<u8> {
    let gray: Vec<u8> = px.chunks(3).map(|p| p[1]).collect();
    let mut e = EncoderConfig::grayscale(85.0)
        .progressive(progressive)
        .allow_16bit_quant_tables(false)
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .expect("encoder");
    e.push_packed(&gray, Unstoppable).expect("push");
    e.finish().expect("finish")
}

fn decoder(method: IdctMethod, threads: usize) -> Decoder {
    Decoder::new()
        .idct_method(method)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .auto_orient(false)
        .num_threads(threads)
}

/// `decode()` (streaming / fused-parallel auto-select) at `threads` threads.
fn decode_full(jpeg: &[u8], method: IdctMethod, threads: usize) -> Vec<u8> {
    decoder(method, threads)
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("u8")
}

/// `scanline_reader()` (pull-based streaming), single-thread, RGB8 output.
fn decode_scanline(jpeg: &[u8], method: IdctMethod) -> Vec<u8> {
    let mut r = decoder(method, 1)
        .scanline_reader(jpeg)
        .expect("scanline_reader");
    let (w, h) = (r.width() as usize, r.height() as usize);
    let stride = w * 3;
    let mut px = vec![0u8; stride * h];
    let mut total = 0;
    while !r.is_finished() {
        let out = imgref::ImgRefMut::new(&mut px[total * stride..], stride, h - total);
        total += r.read_rows_rgb8(out).expect("read");
    }
    assert_eq!(total, h, "scanline didn't read all rows");
    px
}

fn max_diff(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0)
}

fn label(w: usize, h: usize, kind: &str, sub: &str, progressive: bool) -> String {
    format!(
        "{w}x{h} {kind} {sub} {}",
        if progressive {
            "progressive"
        } else {
            "baseline"
        }
    )
}

#[test]
fn fast_idct_all_paths_byte_identical() {
    let mut differs_from_islow = 0usize;
    for &(w, h) in &SIZES {
        for (kind, make) in IMAGE_KINDS {
            let pixels = make(w, h);
            for &(sub_name, sub) in &SUBSAMPLINGS {
                for progressive in [false, true] {
                    let jpeg = encode(&pixels, w as u32, h as u32, sub, progressive);
                    let label = label(w, h, kind, sub_name, progressive);
                    let full = decode_full(&jpeg, IdctMethod::LibjpegFast, 1);
                    let scanline = decode_scanline(&jpeg, IdctMethod::LibjpegFast);
                    let parallel = decode_full(&jpeg, IdctMethod::LibjpegFast, 4);
                    assert_eq!(
                        max_diff(&full, &scanline),
                        0,
                        "{label}: scanline_reader() != decode() under IdctMethod::LibjpegFast"
                    );
                    assert_eq!(
                        max_diff(&full, &parallel),
                        0,
                        "{label}: parallel (4 threads) != decode() under IdctMethod::LibjpegFast"
                    );
                    // The fast kernel must actually be in use: islow and IFAST
                    // round differently on real content.
                    let islow = decode_full(&jpeg, IdctMethod::Libjpeg, 1);
                    let d = max_diff(&full, &islow);
                    assert!(
                        d <= 6,
                        "{label}: LibjpegFast differs from Libjpeg by {d} (>6): wrong kernel, not rounding"
                    );
                    differs_from_islow += (d > 0) as usize;
                }
            }
        }
    }
    assert!(
        differs_from_islow > 0,
        "LibjpegFast never differed from Libjpeg: the fast kernel is not being selected"
    );
}

#[test]
fn fast_idct_applies_to_grayscale_on_every_path() {
    for &(w, h) in &[(67usize, 45usize), (128, 96), (74, 58)] {
        let pixels = test_image(w, h);
        for progressive in [false, true] {
            let jpeg = encode_gray(&pixels, w as u32, h as u32, progressive);
            let label = label(w, h, "gray", "1 component", progressive);
            let full = decode_full(&jpeg, IdctMethod::LibjpegFast, 1);
            let parallel = decode_full(&jpeg, IdctMethod::LibjpegFast, 4);
            assert_eq!(
                max_diff(&full, &parallel),
                0,
                "{label}: parallel != decode()"
            );
            // Gray never falls back to islow under LibjpegFast (libjpeg-turbo's
            // IFAST applies to gray), unlike Jpegli (#154).
            let islow = decode_full(&jpeg, IdctMethod::Libjpeg, 1);
            let jpegli = decode_full(&jpeg, IdctMethod::Jpegli, 1);
            assert_eq!(
                max_diff(&jpegli, &islow),
                0,
                "{label}: Jpegli gray must equal Libjpeg (#154)"
            );
            assert!(
                max_diff(&full, &islow) > 0,
                "{label}: LibjpegFast gray equals islow: the fast kernel is not applied to gray"
            );
        }
    }
}

// ---- Byte-exactness vs real libjpeg-turbo (mozjpeg-sys FFI), JDCT_IFAST ----

/// Decode with mozjpeg-sys (libjpeg-turbo C) to RGB with fancy upsampling:
/// `fast` selects `JDCT_IFAST` (TensorFlow's `INTEGER_FAST` pipeline), otherwise
/// libjpeg-turbo's defaults are left untouched (`JDCT_ISLOW`).
#[cfg(feature = "__ffi-tests")]
fn decode_mozjpeg(data: &[u8], fast: bool) -> Vec<u8> {
    use mozjpeg_sys::*;
    use std::mem;
    // SAFETY: standard libjpeg-turbo decompress sequence; `out` is sized to
    // output_height * stride before any scanline is written.
    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);
        let mut ci: jpeg_decompress_struct = mem::zeroed();
        ci.common.err = &mut err;
        jpeg_create_decompress(&mut ci);
        jpeg_mem_src(&mut ci, data.as_ptr(), data.len() as _);
        assert_eq!(jpeg_read_header(&mut ci, 1), 1, "mozjpeg read_header");
        ci.out_color_space = J_COLOR_SPACE::JCS_RGB;
        if fast {
            ci.dct_method = J_DCT_METHOD::JDCT_IFAST;
        }
        ci.do_fancy_upsampling = 1;
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
        out
    }
}

/// Every u8 decode path must be byte-for-byte identical to libjpeg-turbo's
/// `JDCT_IFAST` decode under `IdctMethod::LibjpegFast` + `Triangle`, across all
/// subsampling modes, baseline / progressive, and both chroma regimes.
#[cfg(feature = "__ffi-tests")]
#[test]
fn fast_idct_all_paths_match_libjpeg_turbo_ifast() {
    for &(w, h) in &SIZES {
        for (kind, make) in IMAGE_KINDS {
            let pixels = make(w, h);
            for &(sub_name, sub) in &SUBSAMPLINGS {
                for progressive in [false, true] {
                    let jpeg = encode(&pixels, w as u32, h as u32, sub, progressive);
                    let moz = decode_mozjpeg(&jpeg, true);
                    let label = label(w, h, kind, sub_name, progressive);
                    for (path, decoded) in [
                        ("decode()", decode_full(&jpeg, IdctMethod::LibjpegFast, 1)),
                        (
                            "scanline_reader()",
                            decode_scanline(&jpeg, IdctMethod::LibjpegFast),
                        ),
                        ("parallel", decode_full(&jpeg, IdctMethod::LibjpegFast, 4)),
                    ] {
                        assert_eq!(
                            max_diff(&decoded, &moz),
                            0,
                            "{label}: {path} != libjpeg-turbo JDCT_IFAST under IdctMethod::LibjpegFast"
                        );
                    }
                }
            }
        }
    }
}

/// Grayscale sources decode with the fast kernel too, like libjpeg-turbo.
#[cfg(feature = "__ffi-tests")]
#[test]
fn fast_idct_grayscale_matches_libjpeg_turbo_ifast() {
    for &(w, h) in &[(67usize, 45usize), (128, 96), (74, 58)] {
        let pixels = test_image(w, h);
        for progressive in [false, true] {
            let jpeg = encode_gray(&pixels, w as u32, h as u32, progressive);
            let moz = decode_mozjpeg(&jpeg, true);
            let label = label(w, h, "gray", "1 component", progressive);
            for (path, decoded) in [
                ("decode()", decode_full(&jpeg, IdctMethod::LibjpegFast, 1)),
                (
                    "scanline_reader()",
                    decode_scanline(&jpeg, IdctMethod::LibjpegFast),
                ),
                ("parallel", decode_full(&jpeg, IdctMethod::LibjpegFast, 4)),
            ] {
                assert_eq!(
                    max_diff(&decoded, &moz),
                    0,
                    "{label}: {path} != libjpeg-turbo JDCT_IFAST under IdctMethod::LibjpegFast"
                );
            }
        }
    }
}

/// **The out-of-the-box decode stays byte-exact with libjpeg-turbo's default
/// decode (`JDCT_ISLOW`, fancy upsampling), grayscale sources included.**
/// `libjpeg_idct_all_paths_parity.rs` pins the bare `Decoder::new()` on
/// 3-component sources; this adds 1-component sources, whose gray plane the
/// fast-integer variant must never touch unless asked for.
#[cfg(feature = "__ffi-tests")]
#[test]
fn default_decoder_matches_libjpeg_turbo_default_on_grayscale() {
    for &(w, h) in &[(67usize, 45usize), (74, 58), (128, 96), (130, 47)] {
        let pixels = test_image(w, h);
        for progressive in [false, true] {
            let jpeg = encode_gray(&pixels, w as u32, h as u32, progressive);
            let moz = decode_mozjpeg(&jpeg, false);
            let label = label(w, h, "gray", "1 component", progressive);
            for threads in [1usize, 4] {
                // Bare defaults: no idct_method(), no chroma_upsampling().
                let got = Decoder::new()
                    .auto_orient(false)
                    .num_threads(threads)
                    .decode(&jpeg, Unstoppable)
                    .expect("decode")
                    .into_pixels_u8()
                    .expect("u8");
                assert_eq!(
                    max_diff(&got, &moz),
                    0,
                    "{label}, {threads} thread(s): default Decoder::new() is not byte-exact with \
                     libjpeg-turbo's default decode"
                );
            }
        }
    }
}
