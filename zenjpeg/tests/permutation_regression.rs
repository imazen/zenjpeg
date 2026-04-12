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
        include_bytes!("testdata/permutation_regression/xyb_p0_sub444_d12_noise7x7.jpg");
    assert_xyb_decodes(data, "XYB p=0 sub=444");
}

#[test]
fn xyb_p0_sub422_decodes() {
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/xyb_p0_sub422_d12_noise7x7.jpg");
    assert_xyb_decodes(data, "XYB p=0 sub=422");
}

#[test]
fn xyb_p0_sub420_decodes() {
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/xyb_p0_sub420_d12_noise7x7.jpg");
    assert_xyb_decodes(data, "XYB p=0 sub=420");
}

// ── Bug 2 (REJECTED): non-uniform chroma subsampling `-sample 2x2,2x1,1x2` ─
//
// Was: cjpeg with Y:(2H,2V), Cb:(2H,1V), Cr:(1H,2V) produced files zenjpeg
// silently decoded with wrong pixels (max diff up to 49 vs mozjpeg on 96×72
// sources). The entire decode pipeline — buffer allocation, upsample
// dispatch, color conversion — assumes both chroma components share the
// same ratio relative to luma. Fixing this properly requires per-chroma-
// component upsampling buffers and dispatch, which is a substantial
// refactor. Non-uniform chroma subsampling is vanishingly rare in real-
// world JPEGs (standard encoders don't produce it by default; we triggered
// it intentionally via cjpeg's exotic `-sample` syntax).
//
// Decision: reject at parse time with UnsupportedFeature rather than
// silently produce wrong pixels. This follows the zero-tolerance rule for
// image corruption. If we ever need to support it, the fix lives in
// parser/markers.rs (remove the rejection), pipeline.rs (per-component
// chroma buffer sizing), and output.rs (per-component upsample dispatch).

fn assert_mixed_chroma_rejected(data: &[u8], label: &str) {
    let err = decode_zen(data)
        .err()
        .unwrap_or_else(|| panic!("{label} should return an error, got success"));
    assert!(
        err.contains("non-uniform chroma subsampling"),
        "{label}: unexpected error signature: {err}"
    );
}

#[test]
fn mixed1_q5_rejected() {
    let data: &[u8] = include_bytes!("testdata/permutation_regression/mixed1_q5_noise_96x72.jpg");
    assert_mixed_chroma_rejected(data, "mixed1 Q5");
}

#[test]
fn mixed1_q75_rejected() {
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/mixed1_q75_patches_96x72.jpg");
    assert_mixed_chroma_rejected(data, "mixed1 Q75");
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
        include_bytes!("testdata/permutation_regression/cjpegli_p0_huffman_noise_47x63_d5.jpg");
    assert_decodes_ok(data, "cjpegli p=0 Huffman fixture");
}

#[test]
fn cjpegli_p0_ac_decodes() {
    // noise_64x64 at distance 5, sub=422, p=0
    let data: &[u8] =
        include_bytes!("testdata/permutation_regression/cjpegli_p0_ac_noise_64x64_d5.jpg");
    assert_decodes_ok(data, "cjpegli p=0 AC fixture");
}
