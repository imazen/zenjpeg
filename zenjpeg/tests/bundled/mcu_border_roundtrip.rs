//! Comprehensive MCU border roundtrip tests.
//!
//! Every chroma subsampling mode has a different MCU (Minimum Coded Unit) size:
//! - 4:4:4 (None):           8×8   (h=1, v=1)
//! - 4:2:2 (HalfHorizontal): 16×8  (h=2, v=1)
//! - 4:4:0 (HalfVertical):   8×16  (h=1, v=2)
//! - 4:2:0 (Quarter):        16×16 (h=2, v=2)
//! - Grayscale:               8×8   (h=1, v=1)
//!
//! Images whose width or height is not a multiple of the MCU size have partial
//! MCU blocks at the right and bottom edges. These tests verify that every
//! offset (1–7 pixels past the MCU grid) encodes and decodes correctly for
//! every subsampling mode, in both baseline and progressive encoding.
//!
//! Additionally, mozjpeg-produced JPEGs at the same non-aligned sizes are
//! decoded to verify cross-encoder compatibility.

use enough::Unstoppable;

use crate::test_utils::TestImage;
use zenjpeg::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Generate a noise+patches test image. Uses deterministic PRNG with patches
/// of solid color to exercise both smooth and textured DCT blocks.
fn test_image_rgb(width: u32, height: u32) -> TestImage {
    let mut img = crate::test_utils::generate_noise(width, height, 42, 3);
    // Add some 8×8 solid patches for more realistic content
    let patch_colors: &[(u8, u8, u8)] =
        &[(200, 50, 30), (30, 180, 60), (50, 40, 210), (220, 200, 40)];
    for (i, &(r, g, b)) in patch_colors.iter().enumerate() {
        let w_mod = width.saturating_sub(7).max(1);
        let h_mod = height.saturating_sub(7).max(1);
        let px = ((i as u32 * 11) % w_mod).min(width.saturating_sub(1));
        let py = ((i as u32 * 13) % h_mod).min(height.saturating_sub(1));
        let ex = (px + 8).min(width);
        let ey = (py + 8).min(height);
        for y in py..ey {
            for x in px..ex {
                img.set_pixel(x, y, 0, r);
                img.set_pixel(x, y, 1, g);
                img.set_pixel(x, y, 2, b);
            }
        }
    }
    img
}

fn test_image_gray(width: u32, height: u32) -> TestImage {
    crate::test_utils::generate_noise(width, height, 42, 1)
}

/// Encode → decode roundtrip. Returns (decoded_width, decoded_height).
/// Panics on encode or decode failure — the test is that it doesn't crash.
fn roundtrip_rgb(img: &TestImage, ss: ChromaSubsampling, progressive: bool) -> (u32, u32) {
    let config = EncoderConfig::ycbcr(90.0, ss).progressive(progressive);
    let mut enc = config
        .encode_from_bytes(img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap_or_else(|e| panic!("encoder setup {}x{}: {e}", img.width, img.height));
    enc.push_packed(&img.pixels, Unstoppable)
        .unwrap_or_else(|e| panic!("push {}x{}: {e}", img.width, img.height));
    let jpeg = enc
        .finish()
        .unwrap_or_else(|e| panic!("finish {}x{}: {e}", img.width, img.height));

    // Verify JPEG starts with SOI
    assert_eq!(
        &jpeg[..2],
        &[0xFF, 0xD8],
        "{}x{}: missing SOI marker",
        img.width,
        img.height
    );

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("decode {}x{}: {e}", img.width, img.height));

    // Dimension check
    assert_eq!(
        decoded.width, img.width,
        "decoded width mismatch for {}x{}",
        img.width, img.height
    );
    assert_eq!(
        decoded.height, img.height,
        "decoded height mismatch for {}x{}",
        img.width, img.height
    );

    // Pixel count sanity (decoder may return RGB regardless of input)
    let dec_pixels = decoded.pixels_u8().expect("no pixel data");
    let expected_bytes = (img.width as usize) * (img.height as usize) * 3;
    assert_eq!(
        dec_pixels.len(),
        expected_bytes,
        "{}x{}: decoded pixel count {} != expected {}",
        img.width,
        img.height,
        dec_pixels.len(),
        expected_bytes
    );

    (decoded.width, decoded.height)
}

/// Encode → decode roundtrip for grayscale.
fn roundtrip_gray(img: &TestImage, progressive: bool) -> (u32, u32) {
    let config = EncoderConfig::grayscale(90.0).progressive(progressive);
    let mut enc = config
        .encode_from_bytes(img.width, img.height, PixelLayout::Gray8Srgb)
        .unwrap_or_else(|e| panic!("encoder setup gray {}x{}: {e}", img.width, img.height));
    enc.push_packed(&img.pixels, Unstoppable)
        .unwrap_or_else(|e| panic!("push gray {}x{}: {e}", img.width, img.height));
    let jpeg = enc
        .finish()
        .unwrap_or_else(|e| panic!("finish gray {}x{}: {e}", img.width, img.height));

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("decode gray {}x{}: {e}", img.width, img.height));

    assert_eq!(
        decoded.width, img.width,
        "gray decoded width mismatch for {}x{}",
        img.width, img.height
    );
    assert_eq!(
        decoded.height, img.height,
        "gray decoded height mismatch for {}x{}",
        img.width, img.height
    );

    (decoded.width, decoded.height)
}

/// Encode with mozjpeg, decode with zenjpeg. Verifies cross-encoder compat.
fn mozjpeg_to_zenjpeg(width: u32, height: u32, ss: mozjpeg_rs::Subsampling) {
    let img = test_image_rgb(width, height);
    let jpeg = mozjpeg_rs::Encoder::new(mozjpeg_rs::Preset::BaselineBalanced)
        .quality(90)
        .subsampling(ss)
        .encode_rgb(&img.pixels, width, height)
        .unwrap_or_else(|e| panic!("mozjpeg encode {}x{}: {e}", width, height));

    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("decode mozjpeg {}x{}: {e}", width, height));

    assert_eq!(
        decoded.width, width,
        "mozjpeg→zen decoded width mismatch for {}x{}",
        width, height
    );
    assert_eq!(
        decoded.height, height,
        "mozjpeg→zen decoded height mismatch for {}x{}",
        width, height
    );

    let dec_pixels = decoded.pixels_u8().expect("no pixel data");
    let expected_bytes = (width as usize) * (height as usize) * 3;
    assert_eq!(
        dec_pixels.len(),
        expected_bytes,
        "mozjpeg→zen {}x{}: decoded pixel count {} != expected {}",
        width,
        height,
        dec_pixels.len(),
        expected_bytes
    );
}

// ── Size generation ────────────────────────────────────────────────────────

/// For a given MCU axis size, generate test widths/heights covering every
/// offset (0 through mcu_axis-1) at a base of 3 MCUs.
fn edge_sizes(mcu_axis: u32) -> Vec<u32> {
    let base = 3 * mcu_axis; // 24 for MCU=8, 48 for MCU=16
    (0..mcu_axis).map(|off| base + off).collect()
}

/// Minimal edge sizes: just the base + each non-zero offset (skip aligned).
fn edge_sizes_nonzero(mcu_axis: u32) -> Vec<u32> {
    let base = 3 * mcu_axis;
    (1..mcu_axis).map(|off| base + off).collect()
}

// ── 4:4:4  (MCU = 8×8) ────────────────────────────────────────────────────

#[test]
fn mcu_border_444_baseline() {
    let w_sizes = edge_sizes(8);
    let h_sizes = edge_sizes(8);
    for &w in &w_sizes {
        for &h in &h_sizes {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::None, false);
        }
    }
    println!(
        "444 baseline: {} combos passed",
        w_sizes.len() * h_sizes.len()
    );
}

#[test]
fn mcu_border_444_progressive() {
    for &w in &edge_sizes_nonzero(8) {
        for &h in &edge_sizes_nonzero(8) {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::None, true);
        }
    }
}

// ── 4:2:2  (MCU = 16×8, h=2 v=1) ─────────────────────────────────────────

#[test]
fn mcu_border_422_baseline() {
    let w_sizes = edge_sizes(16); // h_factor=2 → MCU width=16
    let h_sizes = edge_sizes(8); // v_factor=1 → MCU height=8
    for &w in &w_sizes {
        for &h in &h_sizes {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::HalfHorizontal, false);
        }
    }
    println!(
        "422 baseline: {} combos passed",
        w_sizes.len() * h_sizes.len()
    );
}

#[test]
fn mcu_border_422_progressive() {
    for &w in &edge_sizes_nonzero(16) {
        for &h in &edge_sizes_nonzero(8) {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::HalfHorizontal, true);
        }
    }
}

// ── 4:4:0  (MCU = 8×16, h=1 v=2) ─────────────────────────────────────────

#[test]
fn mcu_border_440_baseline() {
    let w_sizes = edge_sizes(8); // h_factor=1 → MCU width=8
    let h_sizes = edge_sizes(16); // v_factor=2 → MCU height=16
    for &w in &w_sizes {
        for &h in &h_sizes {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::HalfVertical, false);
        }
    }
    println!(
        "440 baseline: {} combos passed",
        w_sizes.len() * h_sizes.len()
    );
}

#[test]
fn mcu_border_440_progressive() {
    for &w in &edge_sizes_nonzero(8) {
        for &h in &edge_sizes_nonzero(16) {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::HalfVertical, true);
        }
    }
}

// ── 4:2:0  (MCU = 16×16, h=2 v=2) ────────────────────────────────────────

#[test]
fn mcu_border_420_baseline() {
    let w_sizes = edge_sizes(16);
    let h_sizes = edge_sizes(16);
    for &w in &w_sizes {
        for &h in &h_sizes {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::Quarter, false);
        }
    }
    println!(
        "420 baseline: {} combos passed",
        w_sizes.len() * h_sizes.len()
    );
}

#[test]
fn mcu_border_420_progressive() {
    for &w in &edge_sizes_nonzero(16) {
        for &h in &edge_sizes_nonzero(16) {
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::Quarter, true);
        }
    }
}

// ── Grayscale  (MCU = 8×8) ────────────────────────────────────────────────

#[test]
fn mcu_border_gray_baseline() {
    for &w in &edge_sizes(8) {
        for &h in &edge_sizes(8) {
            let img = test_image_gray(w, h);
            roundtrip_gray(&img, false);
        }
    }
}

#[test]
fn mcu_border_gray_progressive() {
    for &w in &edge_sizes_nonzero(8) {
        for &h in &edge_sizes_nonzero(8) {
            let img = test_image_gray(w, h);
            roundtrip_gray(&img, true);
        }
    }
}

// ── Minimum dimension tests ───────────────────────────────────────────────

#[test]
fn mcu_border_minimum_dimensions() {
    let tiny_sizes: &[(u32, u32)] = &[
        (1, 1),
        (1, 8),
        (8, 1),
        (1, 16),
        (16, 1),
        (2, 3),
        (3, 2),
        (5, 7),
        (7, 5),
        (9, 9),
        (15, 15),
    ];

    for &(w, h) in tiny_sizes {
        let img = test_image_rgb(w, h);
        for &(ss, name) in &[
            (ChromaSubsampling::None, "444"),
            (ChromaSubsampling::Quarter, "420"),
            (ChromaSubsampling::HalfHorizontal, "422"),
            (ChromaSubsampling::HalfVertical, "440"),
        ] {
            std::panic::catch_unwind(|| {
                roundtrip_rgb(&img, ss, false);
            })
            .unwrap_or_else(|_| {
                panic!("{name} baseline {w}x{h} failed");
            });
        }
    }
}

#[test]
fn mcu_border_minimum_gray() {
    let tiny_sizes: &[(u32, u32)] = &[(1, 1), (1, 8), (8, 1), (3, 5), (7, 7), (9, 9)];

    for &(w, h) in tiny_sizes {
        let img = test_image_gray(w, h);
        roundtrip_gray(&img, false);
    }
}

// ── Exact MCU boundary (offset=0) sanity check ────────────────────────────

#[test]
fn mcu_border_exact_aligned() {
    let aligned: &[(u32, u32)] = &[(8, 8), (16, 16), (24, 24), (32, 32), (48, 48), (64, 64)];

    for &(w, h) in aligned {
        let img = test_image_rgb(w, h);
        for &(ss, name) in &[
            (ChromaSubsampling::None, "444"),
            (ChromaSubsampling::Quarter, "420"),
            (ChromaSubsampling::HalfHorizontal, "422"),
            (ChromaSubsampling::HalfVertical, "440"),
        ] {
            std::panic::catch_unwind(|| {
                roundtrip_rgb(&img, ss, false);
            })
            .unwrap_or_else(|_| {
                panic!("{name} aligned {w}x{h} failed");
            });
        }
    }
}

// ── Asymmetric edge offsets (4:2:0 full matrix) ───────────────────────────

#[test]
fn mcu_border_asymmetric_420() {
    // Test every (w_offset, h_offset) pair for 4:2:0 (MCU=16×16).
    // 16 × 16 = 256 combos at base=48.
    let base = 48u32;
    let mut tested = 0;
    for w_off in 0..16u32 {
        for h_off in 0..16u32 {
            let w = base + w_off;
            let h = base + h_off;
            let img = test_image_rgb(w, h);
            roundtrip_rgb(&img, ChromaSubsampling::Quarter, false);
            tested += 1;
        }
    }
    println!("420 asymmetric: {tested} combos passed");
}

// ── mozjpeg cross-compatibility ───────────────────────────────────────────

#[test]
fn mcu_border_mozjpeg_444() {
    // mozjpeg encode → zenjpeg decode at non-aligned sizes (MCU=8)
    for &w in &edge_sizes_nonzero(8) {
        for &h in &edge_sizes_nonzero(8) {
            mozjpeg_to_zenjpeg(w, h, mozjpeg_rs::Subsampling::S444);
        }
    }
}

#[test]
fn mcu_border_mozjpeg_420() {
    // mozjpeg encode → zenjpeg decode at non-aligned sizes (MCU=16)
    for &w in &edge_sizes_nonzero(16) {
        for &h in &edge_sizes_nonzero(16) {
            mozjpeg_to_zenjpeg(w, h, mozjpeg_rs::Subsampling::S420);
        }
    }
}

#[test]
fn mcu_border_mozjpeg_422() {
    // mozjpeg encode → zenjpeg decode (MCU width=16, height=8)
    for &w in &edge_sizes_nonzero(16) {
        for &h in &edge_sizes_nonzero(8) {
            mozjpeg_to_zenjpeg(w, h, mozjpeg_rs::Subsampling::S422);
        }
    }
}

#[test]
fn mcu_border_mozjpeg_440() {
    // mozjpeg encode → zenjpeg decode (MCU width=8, height=16)
    for &w in &edge_sizes_nonzero(8) {
        for &h in &edge_sizes_nonzero(16) {
            mozjpeg_to_zenjpeg(w, h, mozjpeg_rs::Subsampling::S440);
        }
    }
}

#[test]
fn mcu_border_mozjpeg_tiny() {
    // Very small mozjpeg-encoded images
    let tiny: &[(u32, u32)] = &[(1, 1), (3, 5), (7, 7), (9, 9), (15, 15), (17, 17)];
    for &(w, h) in tiny {
        for &ss in &[
            mozjpeg_rs::Subsampling::S444,
            mozjpeg_rs::Subsampling::S420,
            mozjpeg_rs::Subsampling::S422,
            mozjpeg_rs::Subsampling::S440,
        ] {
            std::panic::catch_unwind(|| {
                mozjpeg_to_zenjpeg(w, h, ss);
            })
            .unwrap_or_else(|_| {
                panic!("mozjpeg→zen {w}x{h} {:?} failed", ss);
            });
        }
    }
}
