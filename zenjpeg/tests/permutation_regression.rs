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

// ── Bug 1: cjpegli XYB sequential mode (-p 0) ──────────────────────────────
//
// cjpegli --xyb -p 0 --chroma_subsampling={444,422,420} produces JPEG files
// zenjpeg fails to decode with "internal error: no decoded data". Works at
// -p 1/-p 2 for any subsampling, and at -p 0 only for --chroma_subsampling=440.
// 720 files in the generated corpus exhibit this; the three fixtures below
// are the smallest representatives of each broken subsampling.

const XYB_EXPECTED_ERR: &str = "no decoded data";

#[test]
fn xyb_p0_sub444_fails() {
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/xyb_p0_sub444_d12_noise7x7.jpg");
    let err = decode_zen(data).expect_err(
        "XYB -p 0 sub=444 currently fails; if this decodes, the bug is fixed — update the test",
    );
    assert!(
        err.contains(XYB_EXPECTED_ERR),
        "bug signature changed: {err}"
    );
}

#[test]
fn xyb_p0_sub422_fails() {
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/xyb_p0_sub422_d12_noise7x7.jpg");
    let err = decode_zen(data).expect_err(
        "XYB -p 0 sub=422 currently fails; if this decodes, the bug is fixed — update the test",
    );
    assert!(
        err.contains(XYB_EXPECTED_ERR),
        "bug signature changed: {err}"
    );
}

#[test]
fn xyb_p0_sub420_fails() {
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/xyb_p0_sub420_d12_noise7x7.jpg");
    let err = decode_zen(data).expect_err(
        "XYB -p 0 sub=420 currently fails; if this decodes, the bug is fixed — update the test",
    );
    assert!(
        err.contains(XYB_EXPECTED_ERR),
        "bug signature changed: {err}"
    );
}

// ── Bug 2: non-uniform chroma subsampling (-sample 2x2,2x1,1x2) ────────────
//
// cjpeg with Y:(2H,2V), Cb:(2H,1V), Cr:(1H,2V) produces JPEG files zenjpeg
// decodes with max pixel diff up to 49 vs mozjpeg. Normal subsampling
// (444/422/420/440/411) has max diff ≤ 7 across the full corpus. 162 files
// in the generated corpus exhibit this.

#[test]
fn mixed1_q5_has_small_diff() {
    // Q5 noise_96x72: recorded max_diff = 10 vs mozjpeg. Asserting > 8 so
    // the test fails when the bug is fixed (expected post-fix: ≤ 7).
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/mixed1_q5_noise_96x72.jpg");
    let (zw, zh, zp) = decode_zen(data).expect("decodes");
    let (mw, mh, mp) = decode_mozjpeg_rgb(data).expect("mozjpeg decodes");
    assert_eq!((zw, zh), (mw, mh));
    let zrgb = normalize_to_rgb(zw, zh, zp);
    let mrgb = normalize_to_rgb(mw, mh, mp);
    let max = max_abs_diff(&zrgb, &mrgb);
    assert!(
        max > 8,
        "mixed1 Q5 max_diff={max} (expected > 8 while bug exists). \
         If max_diff ≤ 8, the non-uniform-subsampling bug is fixed — \
         update or remove this test"
    );
}

#[test]
fn mixed1_q75_has_large_diff() {
    // Q75 patches_96x72: recorded max_diff = 49 vs mozjpeg. Asserting > 30
    // to leave headroom for small changes while still failing on a real fix.
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/mixed1_q75_patches_96x72.jpg");
    let (zw, zh, zp) = decode_zen(data).expect("decodes");
    let (mw, mh, mp) = decode_mozjpeg_rgb(data).expect("mozjpeg decodes");
    assert_eq!((zw, zh), (mw, mh));
    let zrgb = normalize_to_rgb(zw, zh, zp);
    let mrgb = normalize_to_rgb(mw, mh, mp);
    let max = max_abs_diff(&zrgb, &mrgb);
    assert!(
        max > 30,
        "mixed1 Q75 max_diff={max} (expected > 30 while bug exists). \
         If max_diff ≤ 30, the non-uniform-subsampling bug is fixed — \
         update or remove this test"
    );
}

// ── Bug 3: cjpegli -p 0 Huffman / AC errors ────────────────────────────────
//
// cjpegli -p 0 on specific noise/edges source dimensions at sub=422/444
// produces JPEG files zenjpeg rejects with "invalid Huffman table 0: invalid
// code" or "AC coefficient index out of bounds". mozjpeg decodes these fine.
// 21 files in the generated corpus exhibit this.

#[test]
fn cjpegli_p0_huffman_error() {
    // noise_47x63 at distance 5, sub=422, p=0 → "invalid Huffman table 0"
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/cjpegli_p0_huffman_noise_47x63_d5.jpg");
    let err = decode_zen(data).expect_err(
        "cjpegli p=0 Huffman bug: currently fails; if this decodes, bug is fixed",
    );
    assert!(
        err.contains("Huffman") || err.contains("huffman"),
        "bug signature changed: {err}"
    );
}

#[test]
fn cjpegli_p0_ac_error() {
    // noise_64x64 at distance 5, sub=422, p=0 → "AC coefficient index out of bounds"
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/cjpegli_p0_ac_noise_64x64_d5.jpg");
    let err = decode_zen(data).expect_err(
        "cjpegli p=0 AC bug: currently fails; if this decodes, bug is fixed",
    );
    assert!(
        err.contains("AC coefficient") || err.contains("out of bounds"),
        "bug signature changed: {err}"
    );
}
