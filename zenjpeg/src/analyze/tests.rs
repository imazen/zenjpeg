//! Sanity checks for the analyzer ports.
//!
//! These tests check internal invariants — they don't pull in the
//! coefficient crate (would be a dev-dep cycle). Numerical parity vs.
//! `coefficient::analysis::feature_extract` + `evalchroma_ext` is
//! validated end-to-end via the oracle resimulation harness; that
//! lives in coefficient and re-encodes the CID22-val corpus.

use super::*;

fn synth_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    // Cheap reproducible RGB with mild structure (low-frequency variation
    // + per-channel offsets) so every Tier 1 + 2 + 3 path takes a real
    // branch, not a degenerate-flat one.
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let t = seed
                .wrapping_add(x.wrapping_mul(7))
                .wrapping_add(y.wrapping_mul(13));
            let i = ((y * w + x) * 3) as usize;
            buf[i] = ((t >> 1) & 0xFF) as u8;
            buf[i + 1] = ((t >> 3) & 0xFF) as u8;
            buf[i + 2] = ((t >> 2) ^ 0xAA) as u8;
        }
    }
    buf
}

#[test]
fn flat_image_has_zero_variance_and_edges() {
    let w = 64;
    let h = 64;
    let rgb = vec![128u8; (w * h * 3) as usize];
    let out = analyze_rgb8(&rgb, w, h);
    assert_eq!(out.variance, 0.0);
    assert_eq!(out.edge_density, 0.0);
    assert_eq!(out.chroma_complexity, 0.0);
    assert!(out.uniformity > 0.99); // every block uniform
    assert!(out.flat_color_block_ratio > 0.99);
    assert!(out.distinct_color_bins <= 1);
    assert_eq!(out.cb_horiz_sharpness, 0.0);
    assert_eq!(out.cr_horiz_sharpness, 0.0);
    assert_eq!(out.high_freq_energy_ratio, 0.0);
    // Single bin gets all weight ⇒ entropy 0.
    assert!(out.luma_histogram_entropy.abs() < 1e-5);
}

#[test]
fn vstripes_have_high_horiz_chroma_zero_vert() {
    // Alternating R/B columns: horizontal Cb gradient is huge, vertical 0.
    let w = 64;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            if x % 2 == 0 {
                rgb[i] = 255; // red
            } else {
                rgb[i + 2] = 255; // blue
            }
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    assert!(out.cb_horiz_sharpness > 0.0);
    assert!(out.cr_horiz_sharpness > 0.0);
    // Vertical chroma has no second-difference signal between identical
    // rows of column patterns, so it should be 0.
    assert_eq!(out.cb_vert_sharpness, 0.0);
    assert_eq!(out.cr_vert_sharpness, 0.0);
}

#[test]
fn synthetic_image_likelihoods_in_unit_interval() {
    let out = analyze_rgb8(&synth_rgb(128, 128, 42), 128, 128);
    assert!((0.0..=1.0).contains(&out.text_likelihood));
    assert!((0.0..=1.0).contains(&out.screen_content_likelihood));
    assert!((0.0..=1.0).contains(&out.natural_likelihood));
}

#[test]
fn geometry_fields_derive_from_w_h() {
    let out = analyze_rgb8(&synth_rgb(160, 120, 1), 160, 120);
    assert_eq!(out.width, 160);
    assert_eq!(out.height, 120);
    assert!((out.megapixels - 0.0192).abs() < 1e-4);
    assert!((out.aspect_ratio - 160.0 / 120.0).abs() < 1e-4);
}

#[test]
fn checkerboard_has_high_freq_energy() {
    let w = 64;
    let h = 64;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let c = if (x + y) % 2 == 0 { 255 } else { 0 };
            rgb[i] = c;
            rgb[i + 1] = c;
            rgb[i + 2] = c;
        }
    }
    let out = analyze_rgb8(&rgb, w, h);
    // Pure checkerboard is the highest possible high-freq AC.
    assert!(
        out.high_freq_energy_ratio > 1.0,
        "got {}",
        out.high_freq_energy_ratio
    );
    // Two-color image at 5-bit quantization → just a couple of distinct bins.
    assert!(out.distinct_color_bins <= 4);
    // Bimodal luma → lower entropy than a noisy image.
    assert!(out.luma_histogram_entropy < 2.0);
}

#[test]
fn small_images_dont_panic() {
    // < 3×3 hits the Tier 2 short-circuit; < 8×8 hits the Tier 3 short-circuit.
    let _ = analyze_rgb8(&vec![0; 3], 1, 1);
    let _ = analyze_rgb8(&vec![0; 4 * 4 * 3], 4, 4);
    let _ = analyze_rgb8(&vec![0; 7 * 7 * 3], 7, 7);
}
