//! Permutation corpus regression fixtures.
//!
//! Each test captures the CURRENT (buggy) behavior of a known zenjpeg
//! decoder issue surfaced by `gen_permutation_corpus`. When a bug is fixed,
//! the corresponding test will FAIL — update it to assert the new correct
//! behavior, or delete it.
//!
//! Fixtures live in `tests/testdata/permutation_regression/` and total
//! ~12 KB. They were selected as the smallest files per bug variant out of
//! a 25,442-file generated corpus.
//!
//! Tracking: imazen/zenjpeg#29 — zen/all-the-images reproducible corpus.

#![cfg(all(feature = "decoder", not(target_arch = "wasm32")))]

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;

// ── helpers ────────────────────────────────────────────────────────────────

fn decode_zen(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let img = Decoder::new()
        .decode(data, Unstoppable)
        .map_err(|e| format!("{e}"))?;
    let w = img.width;
    let h = img.height;
    let pixels = img.pixels_u8().ok_or("no pixel data")?.to_vec();
    Ok((w, h, pixels))
}

fn decode_mozjpeg_rgb(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use mozjpeg_sys::*;
    use std::mem;

    unsafe {
        let mut err: jpeg_error_mgr = mem::zeroed();
        jpeg_std_error(&mut err);

        let mut cinfo: jpeg_decompress_struct = mem::zeroed();
        cinfo.common.err = &mut err;
        jpeg_create_decompress(&mut cinfo);

        jpeg_mem_src(&mut cinfo, data.as_ptr(), data.len() as _);
        if jpeg_read_header(&mut cinfo, true as boolean) != 1 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg: bad header".into());
        }
        cinfo.out_color_space = J_COLOR_SPACE::JCS_RGB;
        if jpeg_start_decompress(&mut cinfo) == 0 {
            jpeg_destroy_decompress(&mut cinfo);
            return Err("mozjpeg: start_decompress failed".into());
        }
        let width = cinfo.output_width;
        let height = cinfo.output_height;
        let row_stride = width as usize * cinfo.output_components as usize;
        let mut out = vec![0u8; height as usize * row_stride];
        while cinfo.output_scanline < height {
            let offset = cinfo.output_scanline as usize * row_stride;
            let mut row_ptr = out[offset..].as_mut_ptr();
            jpeg_read_scanlines(&mut cinfo, &mut row_ptr, 1);
        }
        jpeg_finish_decompress(&mut cinfo);
        jpeg_destroy_decompress(&mut cinfo);
        Ok((width, height, out))
    }
}

fn normalize_to_rgb(w: u32, h: u32, pixels: Vec<u8>) -> Vec<u8> {
    let n = w as usize * h as usize;
    if pixels.len() == n * 3 {
        pixels
    } else if pixels.len() == n {
        let mut rgb = Vec::with_capacity(n * 3);
        for v in pixels {
            rgb.push(v);
            rgb.push(v);
            rgb.push(v);
        }
        rgb
    } else {
        panic!("unexpected pixel length {} for {w}x{h}", pixels.len());
    }
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len());
    let mut m = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i16 - *y as i16).unsigned_abs() as u8;
        if d > m {
            m = d;
        }
    }
    m
}

// ── Bug 1 (FIXED): cjpegli XYB sequential mode (-p 0) ──────────────────────
//
// Was: cjpegli --xyb -p 0 --chroma_subsampling={444,422,420} produced JPEGs
// that zenjpeg rejected with "internal error: no decoded data". Root cause:
// `can_use_streaming()` did not exclude XYB files, so the baseline streaming
// path (which hard-codes YCbCr→RGB fused output) ran on XYB data without
// storing coefficients, and the XYB-aware output stage later found the
// coefficient buffer empty. Fixed by excluding XYB from streaming — XYB
// files now take the coefficient-storage path, which correctly runs the
// XYB→linear→sRGB conversion.
//
// These fixtures now assert successful decode. The three broken subsampling
// variants all decode as expected 7×7 RGB images.

fn assert_xyb_decodes(data: &[u8], label: &str) {
    let (w, h, pixels) = decode_zen(data)
        .unwrap_or_else(|e| panic!("{label} must decode after XYB streaming fix: {e}"));
    assert_eq!((w, h), (7, 7), "{label}: dims");
    assert_eq!(
        pixels.len(),
        7 * 7 * 3,
        "{label}: expected 7×7 RGB output, got {} bytes",
        pixels.len()
    );
}

#[test]
fn xyb_p0_sub444_decodes() {
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/xyb_p0_sub444_d12_noise7x7.jpg");
    assert_xyb_decodes(data, "XYB p=0 sub=444");
}

#[test]
fn xyb_p0_sub422_decodes() {
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/xyb_p0_sub422_d12_noise7x7.jpg");
    assert_xyb_decodes(data, "XYB p=0 sub=422");
}

#[test]
fn xyb_p0_sub420_decodes() {
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/xyb_p0_sub420_d12_noise7x7.jpg");
    assert_xyb_decodes(data, "XYB p=0 sub=420");
}

// ── Bug 2 (FIXED): non-uniform chroma subsampling `-sample 2x2,2x1,1x2` ────
//
// Was: cjpeg with Y:(2H,2V), Cb:(2H,1V), Cr:(1H,2V) produced files zenjpeg
// decoded with wrong pixels (max diff up to 49 vs mozjpeg on 96×72 sources).
// All four fast paths correctly gated asymmetric chroma, so these files
// reached the f32 generic decode path in `parser/output.rs::to_pixels`.
// That path already allocates per-component planes with per-component
// sampling factors and applies per-component upsamplers — but it was
// reading the block-padded region of the chroma plane without edge-
// replicating the last real row/column first, so the upsamplers saw
// stale IDCT output from the padding blocks at image boundaries.
//
// Fix: `upsample_planes_f32` now materializes a padded copy of the
// component plane with the last real row / column replicated into the
// padding region before dispatching to the upsampler — mirroring
// libjpeg-turbo's `set_bottom_pointers`. Only allocates when the image
// isn't MCU-aligned and the component is actually subsampled; no-op
// for aligned files.
//
// Also adds a missing asymmetric-chroma guard to `fused_parallel.rs` so
// files with Cb≠Cr fall through to the sequential coefficient path
// instead of the parallel fused path (which hard-codes symmetric chroma).

fn assert_decodes_close_to_mozjpeg(data: &[u8], label: &str, threshold: u8) {
    let (zw, zh, zp) =
        decode_zen(data).unwrap_or_else(|e| panic!("{label}: zen decode failed: {e}"));
    let (mw, mh, mp) =
        decode_mozjpeg_rgb(data).unwrap_or_else(|e| panic!("{label}: moz decode failed: {e}"));
    assert_eq!((zw, zh), (mw, mh), "{label}: dim mismatch");
    let zrgb = normalize_to_rgb(zw, zh, zp);
    let mrgb = normalize_to_rgb(mw, mh, mp);
    let max = max_abs_diff(&zrgb, &mrgb);
    assert!(
        max <= threshold,
        "{label} max_diff={max} exceeds threshold {threshold} vs mozjpeg"
    );
}

#[test]
fn mixed1_q5_decodes_correctly() {
    // noise_96x72, Q5, `cjpeg -sample 2x2,2x1,1x2`. Pre-fix max_diff was 10.
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/mixed1_q5_noise_96x72.jpg");
    assert_decodes_close_to_mozjpeg(data, "mixed1 Q5 noise_96x72", 7);
}

#[test]
fn mixed1_q75_decodes_correctly() {
    // patches_96x72, Q75, `cjpeg -sample 2x2,2x1,1x2`. Pre-fix max_diff was 49.
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/mixed1_q75_patches_96x72.jpg");
    assert_decodes_close_to_mozjpeg(data, "mixed1 Q75 patches_96x72", 7);
}

// ── Bug 3 (FIXED): cjpegli -p 0 Huffman / AC errors ────────────────────────
//
// Was: cjpegli -p 0 on noise/edges sources at sub=422/444 produced JPEGs
// zenjpeg rejected with "invalid Huffman table 0: invalid code" or
// "AC coefficient index out of bounds". mozjpeg, zune-jpeg, and jpeg-decoder
// all handled the same files.
//
// Root cause: the baseline streaming decode paths in parser/scan.rs installed
// Huffman tables via `decoder.set_*_table(*comp_idx, ...)`, passing the
// COMPONENT index instead of the FILE TABLE INDEX that decode_block_into
// later looks up. For the usual arrangement (Y=AC0, Cb=AC1, Cr=AC1), file
// index and component index produce the same final layout by coincidence.
// For cjpegli-optimized files where Y and Cb share AC table 0 and Cr uses
// AC table 1, Cr's real table (file AC 1) was stored at slot 2, but decode
// looked up slot 1 — reading Cb's table (file AC 0) for Cr blocks. The
// resulting desync propagated until the decoder tripped on a bogus code or
// an out-of-range run.
//
// Fix: install tables at `dc_idx` / `ac_idx` like the non-streaming path
// already did (parser/scan.rs:274,286 and parser/progressive.rs:124,136).

fn assert_decodes_ok(data: &[u8], label: &str) {
    decode_zen(data).unwrap_or_else(|e| panic!("{label} must decode after table-slot fix: {e}"));
}

#[test]
fn cjpegli_p0_huffman_decodes() {
    // noise_47x63 at distance 5, sub=422, p=0
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/cjpegli_p0_huffman_noise_47x63_d5.jpg");
    assert_decodes_ok(data, "cjpegli p=0 Huffman fixture");
}

#[test]
fn cjpegli_p0_ac_decodes() {
    // noise_64x64 at distance 5, sub=422, p=0
    let data: &[u8] =
        include_bytes!("../testdata/permutation_regression/cjpegli_p0_ac_noise_64x64_d5.jpg");
    assert_decodes_ok(data, "cjpegli p=0 AC fixture");
}
