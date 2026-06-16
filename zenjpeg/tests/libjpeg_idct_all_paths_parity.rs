//! Every decode path must produce BYTE-IDENTICAL u8 RGB under
//! `IdctMethod::Libjpeg`, across all subsampling modes and baseline /
//! progressive — so the libjpeg-turbo-exact reconstruction is independent of
//! which path the decoder auto-selects (streaming `decode()`, pull-based
//! `scanline_reader()`, or the multi-threaded fused-parallel path).
//!
//! Before this test, the only path-parity coverage for `Libjpeg` was 4:2:0
//! baseline `decode()` vs `scanline_reader()` (see `libjpeg_idct_color_paths_agree`
//! in `decode_path_dispatch_parity.rs`). This widens it to the full matrix.
//!
//! With `--features __ffi-tests` it additionally asserts every path is
//! byte-identical to **real libjpeg-turbo** (`mozjpeg-sys`) across all
//! subsampling — extending the prior 4:2:0-only FFI exactness check
//! (`test_idct_method_libjpeg_fancy_matches_mozjpeg_exact`) to 4:2:2/4:4:0/4:4:4
//! on every decode path.
//!
//! Run with `--features parallel` to exercise the real multi-threaded path
//! (without it, `num_threads(4)` runs sequentially and the parallel arm is a
//! trivial pass).
//!
//! Note on the f32 path: `output_target(SrgbF32)` routes the *unclamped*
//! libjpeg islow IDCT (`idct_int_tiered_libjpeg_unclamped`) but then does f32
//! color conversion + f32→u8 rounding, so it is intentionally NOT byte-exact
//! with the u8 paths (it is a higher-precision regime). It is checked here
//! only to stay within a loose band — a guard against a path regressing to a
//! wholly different IDCT — not for equality.

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder, IdctMethod, OutputTarget};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Block-banded RGB with saturated red/blue (stresses clamping + chroma) and
/// a high-frequency green ramp (stresses interpolation/IDCT).
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

fn encode(px: &[u8], w: u32, h: u32, sub: ChromaSubsampling, progressive: bool) -> Vec<u8> {
    let mut e = EncoderConfig::ycbcr(85.0, sub)
        .progressive(progressive)
        .allow_16bit_quant_tables(false)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encoder");
    e.push_packed(px, Unstoppable).expect("push");
    e.finish().expect("finish")
}

/// `decode()` (streaming / fused-parallel auto-select) at `threads` threads.
fn decode_full(jpeg: &[u8], threads: usize) -> Vec<u8> {
    Decoder::new()
        .idct_method(IdctMethod::Libjpeg)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .auto_orient(false)
        .num_threads(threads)
        .decode(jpeg, Unstoppable)
        .expect("decode")
        .into_pixels_u8()
        .expect("u8")
}

/// `scanline_reader()` (pull-based streaming), single-thread.
fn decode_scanline(jpeg: &[u8]) -> Vec<u8> {
    let d = Decoder::new()
        .idct_method(IdctMethod::Libjpeg)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .auto_orient(false)
        .num_threads(1);
    let mut r = d.scanline_reader(jpeg).expect("scanline_reader");
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

/// f32 output path (routes the unclamped libjpeg islow), converted to u8.
fn decode_f32_to_u8(jpeg: &[u8]) -> Vec<u8> {
    let img = Decoder::new()
        .idct_method(IdctMethod::Libjpeg)
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .output_target(OutputTarget::SrgbF32)
        .auto_orient(false)
        .num_threads(1)
        .decode(jpeg, Unstoppable)
        .expect("decode f32");
    img.pixels_f32()
        .expect("f32")
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect()
}

fn max_diff(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0)
}

#[test]
fn libjpeg_idct_all_paths_byte_identical() {
    let subsamplings = [
        ("4:2:0", ChromaSubsampling::Quarter),
        ("4:2:2", ChromaSubsampling::HalfHorizontal),
        ("4:4:0", ChromaSubsampling::HalfVertical),
        ("4:4:4", ChromaSubsampling::None),
    ];
    // Non-MCU-aligned (partial MCUs), aligned, and large (engages parallel).
    let sizes = [(67usize, 45usize), (128, 96), (256, 192)];

    for &(w, h) in &sizes {
        let pixels = test_image(w, h);
        for &(sub_name, sub) in &subsamplings {
            for progressive in [false, true] {
                let jpeg = encode(&pixels, w as u32, h as u32, sub, progressive);
                let label = format!(
                    "{w}x{h} {sub_name} {}",
                    if progressive {
                        "progressive"
                    } else {
                        "baseline"
                    }
                );

                let full = decode_full(&jpeg, 1);
                let scanline = decode_scanline(&jpeg);
                let parallel = decode_full(&jpeg, 4);

                // The u8 paths must be byte-for-byte identical under Libjpeg.
                assert_eq!(
                    max_diff(&full, &scanline),
                    0,
                    "{label}: scanline_reader() != decode() under IdctMethod::Libjpeg"
                );
                assert_eq!(
                    max_diff(&full, &parallel),
                    0,
                    "{label}: parallel (4 threads) != decode() under IdctMethod::Libjpeg"
                );

                // f32 path uses the unclamped libjpeg islow + f32 color — a
                // different precision regime, not byte-exact. Loose guard only.
                let f32_diff = max_diff(&full, &decode_f32_to_u8(&jpeg));
                assert!(
                    f32_diff <= 40,
                    "{label}: f32 path diverges by {f32_diff} (>40) — likely wrong IDCT, \
                     not just precision"
                );
            }
        }
    }
}

// ---- Byte-exactness vs real libjpeg-turbo (mozjpeg-sys FFI) ----

/// Decode with mozjpeg-sys (libjpeg-turbo C) RGB, fancy upsampling.
#[cfg(feature = "__ffi-tests")]
fn decode_mozjpeg(data: &[u8]) -> Vec<u8> {
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

/// Every u8 decode path must be BYTE-FOR-BYTE identical to real libjpeg-turbo
/// (mozjpeg-sys, fancy upsampling) under `IdctMethod::Libjpeg` + `Triangle`,
/// across all subsampling modes and baseline/progressive. The existing FFI
/// check (`test_idct_method_libjpeg_fancy_matches_mozjpeg_exact`) covered only
/// 4:2:0 via `decode()`; this proves it for 4:2:2 / 4:4:0 / 4:4:4 on every
/// path too (verified `max_diff == 0` on every cell before asserting).
#[cfg(feature = "__ffi-tests")]
#[test]
fn libjpeg_idct_all_paths_match_libjpeg_turbo() {
    let subsamplings = [
        ("4:2:0", ChromaSubsampling::Quarter),
        ("4:2:2", ChromaSubsampling::HalfHorizontal),
        ("4:4:0", ChromaSubsampling::HalfVertical),
        ("4:4:4", ChromaSubsampling::None),
    ];
    let sizes = [(67usize, 45usize), (128, 96), (256, 192)];
    for &(w, h) in &sizes {
        let pixels = test_image(w, h);
        for &(sub_name, sub) in &subsamplings {
            for progressive in [false, true] {
                let jpeg = encode(&pixels, w as u32, h as u32, sub, progressive);
                let moz = decode_mozjpeg(&jpeg);
                let label = format!(
                    "{w}x{h} {sub_name} {}",
                    if progressive {
                        "progressive"
                    } else {
                        "baseline"
                    }
                );
                for (path, decoded) in [
                    ("decode()", decode_full(&jpeg, 1)),
                    ("scanline_reader()", decode_scanline(&jpeg)),
                    ("parallel", decode_full(&jpeg, 4)),
                ] {
                    assert_eq!(
                        max_diff(&decoded, &moz),
                        0,
                        "{label}: {path} != libjpeg-turbo under IdctMethod::Libjpeg"
                    );
                }
            }
        }
    }
}
