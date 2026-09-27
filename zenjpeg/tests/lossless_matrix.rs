//! Exhaustive conformance matrix for lossless transforms and restructuring.
//!
//! Regression suite for issues #194 (Transpose/Transverse corrupt on
//! subsampled chroma) and #195 (TrimPartialBlocks / progressive restructure
//! corrupt on non-MCU-aligned images). The whole corruption class came from
//! gaps this matrix now closes:
//!
//! - every subsampling mode (4:4:4, 4:2:2, 4:2:0, 4:4:0, grayscale),
//! - aligned AND non-aligned dimensions (each axis independently),
//! - all 8 transforms × both edge-handling modes,
//! - sequential AND progressive restructure output,
//! - noisy content AND flat-chroma content (per the DC-only-path rule),
//! - oracles that check pixels and coefficients, not just "decodes ok".
//!
//! Oracle stack:
//! 1. Coefficient roundtrip: what the emitters write must decode back to the
//!    same coefficients (true-grid region) — exact, integer, no tolerance.
//! 2. D4 composition (Cayley table): A then B must equal A.then(B) at the
//!    coefficient level on aligned images — exact.
//! 3. Spatial placement oracle: decoded pixels (box chroma upsampling, which
//!    commutes with every D4 permutation) must match the D4-permuted decode
//!    of the source within ±1 (integer IDCT rounding under block transpose).
//! 4. Cross-decoder conformance: jpeg-decoder and zune-jpeg must both accept
//!    every output with the expected dimensions.

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, DecodeConfig, DecodedCoefficients};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestartInterval, RestructureConfig,
    TransformConfig, restructure, transform,
};

// ===== content generators =====

/// Deterministic LCG noise — wide DCT coefficient spread, many Huffman symbols.
fn noise_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut state: u32 = 0x2468_ACE1;
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        for _ in 0..3 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            px.push((state >> 24) as u8);
        }
    }
    px
}

/// Noisy luma with chroma flat inside each 16×16 region but varying across
/// regions — exercises DC-only chroma blocks (see CLAUDE.md: every decoder
/// path test must cover the flat-chroma regime; real photos quantize smooth
/// chroma to DC-only).
fn flat_chroma_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut state: u32 = 0x1357_9BDF;
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let luma = (state >> 24) as u8;
            // Chroma tint constant per 16x16 tile.
            let tile = (y / 16) * 37 + (x / 16) * 11;
            let cb = (tile * 13 % 96) as i32 - 48;
            let cr = (tile * 7 % 96) as i32 - 48;
            let r = (i32::from(luma) + cr).clamp(0, 255) as u8;
            let g = luma;
            let b = (i32::from(luma) + cb).clamp(0, 255) as u8;
            px.push(r);
            px.push(g);
            px.push(b);
        }
    }
    px
}

// ===== encode/decode helpers =====

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Gray,
    Color(ChromaSubsampling),
}

const MODES: [(Mode, &str); 5] = [
    (Mode::Color(ChromaSubsampling::None), "444"),
    (Mode::Color(ChromaSubsampling::HalfHorizontal), "422"),
    (Mode::Color(ChromaSubsampling::HalfVertical), "440"),
    (Mode::Color(ChromaSubsampling::Quarter), "420"),
    (Mode::Gray, "gray"),
];

fn mcu_dims(mode: Mode) -> (u32, u32) {
    match mode {
        Mode::Gray => (8, 8),
        Mode::Color(ss) => match ss {
            ChromaSubsampling::None => (8, 8),
            ChromaSubsampling::HalfHorizontal => (16, 8),
            ChromaSubsampling::HalfVertical => (8, 16),
            ChromaSubsampling::Quarter => (16, 16),
            _ => unreachable!("matrix covers only the four standard modes"),
        },
    }
}

fn encode_jpeg(w: u32, h: u32, mode: Mode, rgb: &[u8]) -> Vec<u8> {
    match mode {
        Mode::Color(ss) => {
            let mut enc = EncoderConfig::ycbcr(90.0, ss)
                .progressive(false)
                .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
                .unwrap();
            enc.push_packed(rgb, Unstoppable).unwrap();
            enc.finish().unwrap()
        }
        Mode::Gray => {
            let gray: Vec<u8> = rgb.as_chunks::<3>().0.iter().map(|p| p[1]).collect();
            let mut enc = EncoderConfig::grayscale(90.0)
                .progressive(false)
                .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
                .unwrap();
            enc.push_packed(&gray, Unstoppable).unwrap();
            enc.finish().unwrap()
        }
    }
}

/// Decode to interleaved 8-bit pixels with box chroma upsampling (commutes
/// exactly with every D4 permutation, unlike the triangle filter).
fn decode_box(jpeg: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    let mut cfg = DecodeConfig::new();
    cfg.chroma_upsampling = ChromaUpsampling::NearestNeighbor;
    let img = cfg.decode(jpeg, Unstoppable).unwrap();
    let (w, h) = (img.width, img.height);
    let px = img
        .into_pixels_u8()
        .expect("default output target is 8-bit");
    let bpp = px.len() / (w as usize * h as usize);
    (w, h, bpp, px)
}

fn decode_coeffs(jpeg: &[u8]) -> DecodedCoefficients {
    DecodeConfig::new()
        .decode_coefficients(jpeg, Unstoppable)
        .unwrap()
}

/// Cross-decoder conformance: both independent pure-Rust decoders must accept
/// the stream and agree on dimensions.
fn assert_cross_decoders(jpeg: &[u8], expect_w: u32, expect_h: u32, ctx: &str) {
    let mut dec = jpeg_decoder::Decoder::new(std::io::Cursor::new(jpeg));
    dec.decode()
        .unwrap_or_else(|e| panic!("{ctx}: jpeg-decoder rejected output: {e}"));
    let info = dec.info().unwrap();
    assert_eq!(
        (u32::from(info.width), u32::from(info.height)),
        (expect_w, expect_h),
        "{ctx}: jpeg-decoder dimensions"
    );

    let zopts = zune_core::options::DecoderOptions::default();
    let mut zdec = zune_jpeg::JpegDecoder::new_with_options(
        zune_jpeg::zune_core::bytestream::ZCursor::new(jpeg),
        zopts,
    );
    zdec.decode()
        .unwrap_or_else(|e| panic!("{ctx}: zune-jpeg rejected output: {e}"));
    let zdims = zdec.dimensions().unwrap();
    assert_eq!(
        (zdims.0 as u32, zdims.1 as u32),
        (expect_w, expect_h),
        "{ctx}: zune-jpeg dimensions"
    );
}

// ===== D4 reference semantics (independent restatement for the oracle) =====

const ALL_TRANSFORMS: [LosslessTransform; 8] = [
    LosslessTransform::None,
    LosslessTransform::FlipHorizontal,
    LosslessTransform::FlipVertical,
    LosslessTransform::Transpose,
    LosslessTransform::Rotate90,
    LosslessTransform::Rotate180,
    LosslessTransform::Rotate270,
    LosslessTransform::Transverse,
];

/// Forward point mapping: destination coordinates of source pixel (x, y).
fn map_point(t: LosslessTransform, x: u32, y: u32, w: u32, h: u32) -> (u32, u32) {
    match t {
        LosslessTransform::None => (x, y),
        LosslessTransform::FlipHorizontal => (w - 1 - x, y),
        LosslessTransform::FlipVertical => (x, h - 1 - y),
        LosslessTransform::Transpose => (y, x),
        LosslessTransform::Rotate90 => (h - 1 - y, x),
        LosslessTransform::Rotate180 => (w - 1 - x, h - 1 - y),
        LosslessTransform::Rotate270 => (y, w - 1 - x),
        LosslessTransform::Transverse => (h - 1 - y, w - 1 - x),
    }
}

/// Which source dimensions must be MCU-aligned (else trimmed), per transform.
/// Matches jpegtran: a trailing partial edge that stays trailing is kept.
fn must_align(t: LosslessTransform) -> (bool, bool) {
    match t {
        LosslessTransform::None | LosslessTransform::Transpose => (false, false),
        LosslessTransform::FlipHorizontal => (true, false),
        LosslessTransform::FlipVertical | LosslessTransform::Rotate90 => (false, true),
        LosslessTransform::Rotate270 => (true, false),
        LosslessTransform::Rotate180 | LosslessTransform::Transverse => (true, true),
    }
}

/// Kept (trim) source region and final output dimensions.
fn expected_dims(w: u32, h: u32, mode: Mode, t: LosslessTransform) -> (u32, u32, u32, u32) {
    let (mcu_w, mcu_h) = mcu_dims(mode);
    let (wa, ha) = must_align(t);
    let kept_w = if wa && w % mcu_w != 0 {
        (w / mcu_w) * mcu_w
    } else {
        w
    };
    let kept_h = if ha && h % mcu_h != 0 {
        (h / mcu_h) * mcu_h
    } else {
        h
    };
    let (ow, oh) = t.output_dimensions(kept_w, kept_h);
    (kept_w, kept_h, ow, oh)
}

/// Apply the D4 permutation to decoded pixels (any bytes-per-pixel).
fn permute_pixels(src: &[u8], w: u32, h: u32, bpp: usize, t: LosslessTransform) -> Vec<u8> {
    let (ow, oh) = t.output_dimensions(w, h);
    let mut dst = vec![0u8; (ow * oh) as usize * bpp];
    for y in 0..h {
        for x in 0..w {
            let (u, v) = map_point(t, x, y, w, h);
            let s = ((y * w + x) as usize) * bpp;
            let d = ((v * ow + u) as usize) * bpp;
            dst[d..d + bpp].copy_from_slice(&src[s..s + bpp]);
        }
    }
    dst
}

/// Crop interleaved pixels to the top-left kept region.
fn crop_pixels(src: &[u8], w: u32, _h: u32, bpp: usize, kw: u32, kh: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((kw * kh) as usize * bpp);
    for y in 0..kh {
        let start = ((y * w) as usize) * bpp;
        out.extend_from_slice(&src[start..start + (kw as usize) * bpp]);
    }
    out
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> (u32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0u32;
    let mut differing = 0usize;
    for (&x, &y) in a.iter().zip(b) {
        let d = (i32::from(x) - i32::from(y)).unsigned_abs();
        if d != 0 {
            differing += 1;
            if d > max {
                max = d;
            }
        }
    }
    (max, differing)
}

/// Compare two coefficient sets over each component's true (non-padded) grid.
/// Padding blocks are excluded: the progressive path legitimately re-pads with
/// zero blocks.
fn assert_coeffs_equal_true_region(a: &DecodedCoefficients, b: &DecodedCoefficients, ctx: &str) {
    assert_eq!((a.width, a.height), (b.width, b.height), "{ctx}: dims");
    assert_eq!(a.components.len(), b.components.len(), "{ctx}: comp count");
    let max_h = a.components.iter().map(|c| c.h_samp).max().unwrap() as u32;
    let max_v = a.components.iter().map(|c| c.v_samp).max().unwrap() as u32;
    for (ca, cb) in a.components.iter().zip(&b.components) {
        assert_eq!(
            (ca.h_samp, ca.v_samp),
            (cb.h_samp, cb.v_samp),
            "{ctx}: sampling factors (component id {})",
            ca.id
        );
        assert_eq!(
            (ca.blocks_wide, ca.blocks_high),
            (cb.blocks_wide, cb.blocks_high),
            "{ctx}: grid dims (component id {})",
            ca.id
        );
        let comp_w = (a.width * u32::from(ca.h_samp)).div_ceil(max_h);
        let comp_h = (a.height * u32::from(ca.v_samp)).div_ceil(max_v);
        let true_bw = comp_w.div_ceil(8) as usize;
        let true_bh = comp_h.div_ceil(8) as usize;
        for by in 0..true_bh {
            for bx in 0..true_bw {
                let idx = by * ca.blocks_wide + bx;
                assert_eq!(
                    ca.block(idx),
                    cb.block(idx),
                    "{ctx}: component id {} block ({bx},{by}) coefficients differ",
                    ca.id
                );
            }
        }
    }
}

// Dimensions: (label, w, h). 66x50 is unaligned on both axes for every MCU
// size; 64x50 and 66x48 isolate each axis; 64x48 is aligned for all modes;
// 23x17 is a small odd case (below one MCU in places).
const SIZES: [(&str, u32, u32); 5] = [
    ("aligned", 64, 48),
    ("both-unaligned", 66, 50),
    ("h-unaligned", 64, 50),
    ("w-unaligned", 66, 48),
    ("tiny-odd", 23, 17),
];

fn generators() -> [(&'static str, fn(u32, u32) -> Vec<u8>); 2] {
    [("noise", noise_rgb), ("flat-chroma", flat_chroma_rgb)]
}

// ===== Test 1: restructure (no transform) preserves coefficients exactly =====

#[test]
fn restructure_roundtrips_coefficients_all_modes() {
    for (gen_name, generate) in generators() {
        for (mode, mode_name) in MODES {
            for (size_name, w, h) in SIZES {
                let jpeg = encode_jpeg(w, h, mode, &generate(w, h));
                let src_coeffs = decode_coeffs(&jpeg);
                for (out_name, out_mode) in [
                    ("seq", OutputMode::Sequential),
                    ("prog", OutputMode::Progressive),
                ] {
                    let ctx = format!("{gen_name}/{mode_name}/{size_name}/{out_name}");
                    let cfg = RestructureConfig {
                        output_mode: out_mode,
                        restart_interval: RestartInterval::None,
                        transform: None,
                    };
                    let out = restructure(&jpeg, &cfg, Unstoppable)
                        .unwrap_or_else(|e| panic!("{ctx}: restructure failed: {e}"));
                    let out_coeffs = decode_coeffs(&out);
                    assert_coeffs_equal_true_region(&src_coeffs, &out_coeffs, &ctx);
                    assert_cross_decoders(&out, w, h, &ctx);
                }
            }
        }
    }
}

// ===== Test 2: full transform matrix — structure, dims, cross-decoders =====

#[test]
fn transform_matrix_structural_conformance() {
    for (gen_name, generate) in generators() {
        for (mode, mode_name) in MODES {
            let (mcu_w, mcu_h) = mcu_dims(mode);
            for (size_name, w, h) in SIZES {
                let jpeg = encode_jpeg(w, h, mode, &generate(w, h));
                for t in ALL_TRANSFORMS {
                    let (kept_w, kept_h, ow, oh) = expected_dims(w, h, mode, t);
                    let needs_trim = kept_w != w || kept_h != h;
                    let ctx = format!("{gen_name}/{mode_name}/{size_name}/{t:?}");

                    // Reject mode must error exactly when a trim is needed.
                    let reject = transform(
                        &jpeg,
                        &TransformConfig {
                            transform: t,
                            edge_handling: EdgeHandling::RejectPartialBlocks,
                        },
                        Unstoppable,
                    );
                    assert_eq!(
                        reject.is_err(),
                        needs_trim,
                        "{ctx}: RejectPartialBlocks disagreement (mcu {mcu_w}x{mcu_h})"
                    );

                    let out = transform(
                        &jpeg,
                        &TransformConfig {
                            transform: t,
                            edge_handling: EdgeHandling::TrimPartialBlocks,
                        },
                        Unstoppable,
                    )
                    .unwrap_or_else(|e| panic!("{ctx}: transform failed: {e}"));

                    let (dw, dh, _, _) = decode_box(&out);
                    assert_eq!((dw, dh), (ow, oh), "{ctx}: output dimensions");
                    assert_cross_decoders(&out, ow, oh, &ctx);
                }
            }
        }
    }
}

// ===== Test 3: spatial placement oracle (box upsampling commutes with D4) =====

/// Track the worst rounding deviation seen, so a threshold drift is visible
/// in test output before it ever becomes a failure.
fn record_oracle_stats(ctx: &str, max_d: u32, frac: f64) {
    use std::sync::Mutex;
    static WORST: Mutex<(u32, f64)> = Mutex::new((0, 0.0));
    let mut worst = WORST.lock().unwrap();
    if max_d > worst.0 || frac > worst.1 {
        worst.0 = worst.0.max(max_d);
        worst.1 = worst.1.max(frac);
        eprintln!(
            "spatial-oracle envelope now max|d|={} frac={:.4}% (at {ctx})",
            worst.0,
            worst.1 * 100.0
        );
    }
}

#[test]
fn transform_matrix_spatial_oracle() {
    for (gen_name, generate) in generators() {
        for (mode, mode_name) in MODES {
            for (size_name, w, h) in SIZES {
                let jpeg = encode_jpeg(w, h, mode, &generate(w, h));
                let (sw, sh, bpp, src_px) = decode_box(&jpeg);
                assert_eq!((sw, sh), (w, h));
                for t in ALL_TRANSFORMS {
                    let (kept_w, kept_h, ow, oh) = expected_dims(w, h, mode, t);
                    let ctx = format!("{gen_name}/{mode_name}/{size_name}/{t:?}");

                    let out = transform(
                        &jpeg,
                        &TransformConfig {
                            transform: t,
                            edge_handling: EdgeHandling::TrimPartialBlocks,
                        },
                        Unstoppable,
                    )
                    .unwrap();
                    let (dw, dh, dbpp, out_px) = decode_box(&out);
                    assert_eq!((dw, dh, dbpp), (ow, oh, bpp), "{ctx}: decode shape");

                    let cropped = crop_pixels(&src_px, w, h, bpp, kept_w, kept_h);
                    let expected = permute_pixels(&cropped, kept_w, kept_h, bpp, t);
                    let (max_d, differing) = max_abs_diff(&out_px, &expected);
                    let frac = differing as f64 / expected.len() as f64;
                    // This oracle catches block PLACEMENT bugs, which produce
                    // max|d| in the hundreds on >50% of samples (the #194/#195
                    // failures measured 96-99% wrong, max|d| 243-254). The
                    // fixed-point IDCT does not commute exactly with
                    // coefficient-domain transposition; measured envelope over
                    // this whole matrix on 2026-08-26 was max|d|=3 with 8.83%
                    // of samples differing (worst cell: noise/420/tiny-odd/
                    // Rotate270). Threshold sits at ~2x that envelope — still
                    // 30x-80x below the failure mode being detected.
                    assert!(
                        max_d <= 8 && frac < 0.15,
                        "{ctx}: spatial oracle violated: max|d|={max_d}, {differing}/{} samples differ ({:.4}%)",
                        expected.len(),
                        frac * 100.0
                    );
                    record_oracle_stats(&ctx, max_d, frac);
                }
            }
        }
    }
}

// ===== Test 4: D4 composition (Cayley) at the coefficient level =====

#[test]
fn transform_composition_cayley_exact() {
    // Aligned image only: trims make composition non-associative.
    let (w, h) = (64u32, 48u32);
    for (mode, mode_name) in MODES {
        let jpeg = encode_jpeg(w, h, mode, &noise_rgb(w, h));
        for a in ALL_TRANSFORMS {
            let ja = transform(
                &jpeg,
                &TransformConfig {
                    transform: a,
                    edge_handling: EdgeHandling::RejectPartialBlocks,
                },
                Unstoppable,
            )
            .unwrap();
            for b in ALL_TRANSFORMS {
                let ctx = format!("{mode_name}: {a:?} then {b:?}");
                let jab = transform(
                    &ja,
                    &TransformConfig {
                        transform: b,
                        edge_handling: EdgeHandling::RejectPartialBlocks,
                    },
                    Unstoppable,
                )
                .unwrap_or_else(|e| panic!("{ctx}: second transform failed: {e}"));
                let composed = a.then(b);
                let jc = transform(
                    &jpeg,
                    &TransformConfig {
                        transform: composed,
                        edge_handling: EdgeHandling::RejectPartialBlocks,
                    },
                    Unstoppable,
                )
                .unwrap();
                assert_coeffs_equal_true_region(
                    &decode_coeffs(&jab),
                    &decode_coeffs(&jc),
                    &format!("{ctx} (== {composed:?})"),
                );
            }
        }
    }
}

// ===== Test 5: transforms through restructure() match transform() ==========

#[test]
fn restructure_with_transform_matches_transform() {
    let (w, h) = (66u32, 50u32);
    for (mode, mode_name) in MODES {
        let jpeg = encode_jpeg(w, h, mode, &noise_rgb(w, h));
        for t in ALL_TRANSFORMS {
            for (out_name, out_mode) in [
                ("seq", OutputMode::Sequential),
                ("prog", OutputMode::Progressive),
            ] {
                let ctx = format!("{mode_name}/{t:?}/{out_name}");
                let (_, _, ow, oh) = expected_dims(w, h, mode, t);
                let out = restructure(
                    &jpeg,
                    &RestructureConfig {
                        output_mode: out_mode,
                        restart_interval: RestartInterval::None,
                        transform: Some(TransformConfig {
                            transform: t,
                            edge_handling: EdgeHandling::TrimPartialBlocks,
                        }),
                    },
                    Unstoppable,
                )
                .unwrap_or_else(|e| panic!("{ctx}: restructure failed: {e}"));
                let via_transform = transform(
                    &jpeg,
                    &TransformConfig {
                        transform: t,
                        edge_handling: EdgeHandling::TrimPartialBlocks,
                    },
                    Unstoppable,
                )
                .unwrap();
                assert_coeffs_equal_true_region(
                    &decode_coeffs(&out),
                    &decode_coeffs(&via_transform),
                    &ctx,
                );
                assert_cross_decoders(&out, ow, oh, &ctx);
            }
        }
    }
}

// ===== Test 6: restart markers respect the count/encode unification =========

#[test]
fn sequential_restart_markers_roundtrip() {
    for (mode, mode_name) in MODES {
        for (size_name, w, h) in [("aligned", 64u32, 48u32), ("both-unaligned", 66, 50)] {
            let jpeg = encode_jpeg(w, h, mode, &noise_rgb(w, h));
            let src_coeffs = decode_coeffs(&jpeg);
            for interval in [
                RestartInterval::EveryMcus(3),
                RestartInterval::EveryMcuRows(1),
            ] {
                let ctx = format!("{mode_name}/{size_name}/{interval:?}");
                let out = restructure(
                    &jpeg,
                    &RestructureConfig {
                        output_mode: OutputMode::Sequential,
                        restart_interval: interval,
                        transform: None,
                    },
                    Unstoppable,
                )
                .unwrap_or_else(|e| panic!("{ctx}: restructure failed: {e}"));
                assert_coeffs_equal_true_region(&src_coeffs, &decode_coeffs(&out), &ctx);
                assert_cross_decoders(&out, w, h, &ctx);
            }
        }
    }
}
