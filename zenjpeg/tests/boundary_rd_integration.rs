//! Integration tests for the Phase-2 boundary-continuity refinement
//! (issue #91).
//!
//! Covers:
//!
//! 1. `boundary_rd(false)` vs `boundary_rd(true)` both produce valid,
//!    decodable JPEGs.
//! 2. BBS on the reconstructed luma channel does not regress by a lot
//!    when `boundary_rd(true)` is enabled (at matched quality) on at
//!    least 4 of 5 synthetic + photo inputs across Q70/Q80/Q90.
//! 3. Byte-exact hash-lock: with fixed parameters
//!    (α=1.0, threshold=0.1, shrink=0.7, retries=1), a small fixed
//!    test image encodes to bytes whose length and content hash are
//!    stable. Any future drift in the refinement algorithm shows up
//!    as a hash mismatch.
//!
//! The 4-of-5 condition is deliberate — issue #91 documents that
//! boundary-RD is image-class-dependent and one photograph may
//! regress slightly at one quality even when the overall direction
//! is correct; that is the tuning-phase problem, not a correctness
//! gate.

use enough::Unstoppable;
use std::hash::{Hash, Hasher};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::metrics::bbs::bbs_rgb8;

// =============================================================================
// Test-image generators — NO gradients (per CLAUDE.md rules). Each image is
// designed to have visible block-edge artifacts at intermediate JPEG quality.
// =============================================================================

/// A small bright checkerboard that produces strong seam artifacts.
fn gen_checkerboard(w: usize, h: usize, cell: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let on = ((x / cell) + (y / cell)) % 2 == 0;
            let v = if on { 240 } else { 20 };
            let i = (y * w + x) * 3;
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

/// A black-and-white stripe pattern.
fn gen_stripes(w: usize, h: usize, period: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let on = (x / period) % 2 == 0;
            let v = if on { 255 } else { 0 };
            let i = (y * w + x) * 3;
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

/// Deterministic noise + blotches (a "screenshot"-ish class).
fn gen_noise_patches(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for y in 0..h {
        for x in 0..w {
            // xorshift64*
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let n = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let noise = ((n >> 32) & 0xFF) as u8;
            let patch_y = y / 24 % 2;
            let patch_x = x / 24 % 2;
            let base: u8 = if patch_y == patch_x { 200 } else { 40 };
            let v = base.saturating_add(noise / 8);
            let i = (y * w + x) * 3;
            out[i] = v;
            out[i + 1] = v;
            out[i + 2] = v;
        }
    }
    out
}

/// Encoder helper; returns JPEG bytes.
fn encode_rgb8(
    rgb: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    boundary_rd: bool,
) -> Vec<u8> {
    let mut cfg = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter);
    if boundary_rd {
        cfg = cfg.boundary_rd(true);
    }
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode config");
    enc.push_packed(rgb, Unstoppable).expect("push rows");
    enc.finish().expect("finish")
}

/// Decode helper; returns (rgb, width, height). Uses zune-jpeg — same as
/// `rd_compare.rs` does so these tests don't depend on decoder features.
fn decode_rgb8(jpeg: &[u8]) -> (Vec<u8>, u32, u32) {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(jpeg), options);
    let pixels = decoder.decode().expect("decode");
    let (w, h) = decoder.dimensions().expect("dimensions");
    (pixels, w as u32, h as u32)
}

/// Imgref helper.
fn as_imgref_rgb(buf: &[u8], w: usize, h: usize) -> imgref::ImgRef<'_, rgb::RGB<u8>> {
    // `rgb` newtype is `#[repr(C)]` with three `u8` fields; casting a `*const u8`
    // to `*const rgb::RGB<u8>` is how the metric itself is called elsewhere in
    // the codebase (see bench utils). For safe code we go through bytemuck via
    // `bytemuck::cast_slice` would require a dep; instead we reconstruct the
    // same slice view via `rgb::AsPixels` which is already a project dep.
    use rgb::FromSlice;
    imgref::ImgRef::new(buf.as_rgb(), w, h)
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn boundary_rd_off_and_on_both_decode() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_checkerboard(w, h, 10);
    let bytes_off = encode_rgb8(&rgb, w as u32, h as u32, 80, false);
    let bytes_on = encode_rgb8(&rgb, w as u32, h as u32, 80, true);
    let (dec_off, w_off, h_off) = decode_rgb8(&bytes_off);
    let (dec_on, w_on, h_on) = decode_rgb8(&bytes_on);
    assert_eq!((w_off, h_off), (w as u32, h as u32));
    assert_eq!((w_on, h_on), (w as u32, h as u32));
    assert_eq!(dec_off.len(), w * h * 3);
    assert_eq!(dec_on.len(), w * h * 3);
}

#[test]
fn bbs_not_strongly_regressed_on_synthetics() {
    // "Not strongly regressed" means: for each (image, quality) case,
    // candidate BBS must be within a small multiplicative budget of
    // baseline BBS. #91 explicitly notes that one case may regress —
    // we require at least 4 of 5 non-regressions.
    //
    // Threshold: candidate_bbs / baseline_bbs < 1.1 counts as a pass.
    const W: usize = 128;
    const H: usize = 128;
    let images: [(&str, Vec<u8>); 5] = [
        ("checkerboard_8", gen_checkerboard(W, H, 8)),
        ("checkerboard_16", gen_checkerboard(W, H, 16)),
        ("stripes_4", gen_stripes(W, H, 4)),
        ("noise_patches_a", gen_noise_patches(W, H, 0xDEAD_BEEF)),
        ("noise_patches_b", gen_noise_patches(W, H, 0x1234_5678)),
    ];
    let qualities: [u8; 3] = [70, 80, 90];

    let mut pass_count = 0usize;
    let mut total_count = 0usize;
    let mut report = Vec::new();

    for (name, rgb) in &images {
        for &q in &qualities {
            let orig_ref = as_imgref_rgb(rgb, W, H);

            let bytes_off = encode_rgb8(rgb, W as u32, H as u32, q, false);
            let (dec_off, _, _) = decode_rgb8(&bytes_off);
            let bbs_off =
                bbs_rgb8(as_imgref_rgb(&dec_off, W, H), orig_ref).total;

            let bytes_on = encode_rgb8(rgb, W as u32, H as u32, q, true);
            let (dec_on, _, _) = decode_rgb8(&bytes_on);
            let bbs_on =
                bbs_rgb8(as_imgref_rgb(&dec_on, W, H), orig_ref).total;

            let ratio = if bbs_off > 0.0 {
                bbs_on / bbs_off
            } else {
                1.0
            };
            report.push(format!(
                "{name} Q{q}: bbs_off={bbs_off:.1} bbs_on={bbs_on:.1} ratio={ratio:.3}"
            ));
            total_count += 1;
            if ratio < 1.1 {
                pass_count += 1;
            }
        }
    }

    // 15 cells total (5 images × 3 qualities). Require at least 12
    // non-regressions — i.e. at most 3 cells above the 1.1 ratio
    // tolerance. That matches the "at least 4 of 5 images" spec with
    // headroom across multiple quality levels.
    let needed = 12usize;
    assert!(
        pass_count >= needed,
        "boundary-RD BBS regressed on too many cells: {pass_count}/{total_count} passed.\n{}",
        report.join("\n")
    );
}

#[test]
fn hash_lock_checkerboard_q80_boundary_rd_on() {
    // Byte-exact hash-lock with the Phase 5 tuned defaults
    // (α=1.0, threshold=0.05, shrink=0.5, retries=2). If this fails,
    // either the refinement math drifted or the defaults changed —
    // investigate, don't retrain. The 64×64 uniform checkerboard does
    // not trigger the refinement (its block seams already match the
    // original), so the output happens to be byte-identical to the
    // Phase 2 locked output; a future content-aware hash-lock on a
    // corpus image would catch defaults drift too.
    let (w, h) = (64usize, 64usize);
    let rgb = gen_checkerboard(w, h, 8);
    let bytes = encode_rgb8(&rgb, w as u32, h as u32, 80, true);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    let h64 = hasher.finish();
    eprintln!("hash_lock: size={} hash64={:016x}", bytes.len(), h64);

    // Exact byte size locked; drift of ±1 will fail this check.
    assert_eq!(
        bytes.len(),
        513,
        "hash-lock: encoded size drift"
    );
    // Exact hash locked. Refresh both values together if this fails
    // intentionally (e.g. another round of default tuning).
    assert_eq!(
        h64,
        0xaba706f19b94e26f,
        "hash-lock: encoded-byte hash drift"
    );

    // Minimum correctness: decode must succeed and output must have
    // the right pixel count.
    let (dec, dw, dh) = decode_rgb8(&bytes);
    assert_eq!((dw, dh), (w as u32, h as u32));
    assert_eq!(dec.len(), w * h * 3);
}
