//! Integration tests for the Phase-4 above-neighbor extension of the
//! boundary-continuity refinement (issue #91).
//!
//! Covers:
//!
//! 1. `boundary_rd(true)` (left-only) vs
//!    `boundary_rd(true).boundary_rd_above(true)` both produce valid,
//!    decodable JPEGs.
//! 2. BBS on the reconstructed luma channel does not strongly regress
//!    when the above term is layered on top of left-only (at matched
//!    quality) on the same synthetic class as the Phase-2 tests.
//! 3. Byte-exact hash-lock: with Phase 5 defaults plus
//!    `boundary_rd_above(true)`, a small fixed test image encodes to
//!    bytes whose length and content hash are stable. Any future drift
//!    in the refinement algorithm shows up as a hash mismatch.
//! 4. Default-path regression guard: `boundary_rd_above(true)` with
//!    `boundary_rd(false)` MUST be byte-identical to plain default
//!    encode. The above flag is a strict no-op without the main flag.

use enough::Unstoppable;
use std::hash::{Hash, Hasher};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::metrics::bbs::bbs_rgb8;

// =============================================================================
// Test-image generators — same as the Phase-2 integration test. Kept local
// so the two test files don't couple.
// =============================================================================

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

fn gen_noise_patches(w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 3];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for y in 0..h {
        for x in 0..w {
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

fn encode_rgb8(rgb: &[u8], w: u32, h: u32, quality: u8, mode: Mode) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(quality as f32, ChromaSubsampling::Quarter);
    let cfg = match mode {
        Mode::Default => cfg,
        Mode::LeftOnly => cfg.boundary_rd(true),
        Mode::LeftAbove => cfg.boundary_rd(true).boundary_rd_above(true),
        Mode::AboveOnlyFlag => cfg.boundary_rd_above(true),
    };
    let mut enc = cfg
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode config");
    enc.push_packed(rgb, Unstoppable).expect("push rows");
    enc.finish().expect("finish")
}

#[derive(Copy, Clone)]
enum Mode {
    Default,
    LeftOnly,
    LeftAbove,
    /// `boundary_rd_above(true)` with `boundary_rd(false)` — must be a no-op.
    AboveOnlyFlag,
}

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

fn as_imgref_rgb(buf: &[u8], w: usize, h: usize) -> imgref::ImgRef<'_, rgb::RGB<u8>> {
    use rgb::FromSlice;
    imgref::ImgRef::new(buf.as_rgb(), w, h)
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn left_only_and_left_above_both_decode() {
    let (w, h) = (128usize, 128usize);
    let rgb = gen_checkerboard(w, h, 10);
    let left_only = encode_rgb8(&rgb, w as u32, h as u32, 80, Mode::LeftOnly);
    let left_above = encode_rgb8(&rgb, w as u32, h as u32, 80, Mode::LeftAbove);
    let (dec_l, wl, hl) = decode_rgb8(&left_only);
    let (dec_la, wla, hla) = decode_rgb8(&left_above);
    assert_eq!((wl, hl), (w as u32, h as u32));
    assert_eq!((wla, hla), (w as u32, h as u32));
    assert_eq!(dec_l.len(), w * h * 3);
    assert_eq!(dec_la.len(), w * h * 3);
}

#[test]
fn above_flag_without_left_flag_is_noop() {
    // `boundary_rd_above(true)` alone (with `boundary_rd(false)`) must
    // produce byte-identical output to plain default encode. This is the
    // orthogonality contract from issue #91.
    let (w, h) = (96usize, 96usize);
    let rgb = gen_noise_patches(w, h, 0xA5A5_5A5A);
    for q in [60u8, 75, 90] {
        let default = encode_rgb8(&rgb, w as u32, h as u32, q, Mode::Default);
        let above_only = encode_rgb8(&rgb, w as u32, h as u32, q, Mode::AboveOnlyFlag);
        assert_eq!(
            default, above_only,
            "above-only flag should be a strict no-op at Q{q}"
        );
    }
}

#[test]
fn bbs_not_strongly_regressed_vs_left_only() {
    // Same generators & tolerance as the Phase-2 test. "Not strongly
    // regressed" means BBS(left+above) < 1.1 × BBS(left-only) on at
    // least 4 of 5 images across Q70/Q80/Q90 (12 of 15 cells), mirroring
    // the incremental-over-left-only guard for Phase 4.
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

            let bytes_l = encode_rgb8(rgb, W as u32, H as u32, q, Mode::LeftOnly);
            let (dec_l, _, _) = decode_rgb8(&bytes_l);
            let bbs_l = bbs_rgb8(as_imgref_rgb(&dec_l, W, H), orig_ref).total;

            let bytes_la = encode_rgb8(rgb, W as u32, H as u32, q, Mode::LeftAbove);
            let (dec_la, _, _) = decode_rgb8(&bytes_la);
            let bbs_la = bbs_rgb8(as_imgref_rgb(&dec_la, W, H), orig_ref).total;

            let ratio = if bbs_l > 0.0 { bbs_la / bbs_l } else { 1.0 };
            report.push(format!(
                "{name} Q{q}: bbs_left={bbs_l:.1} bbs_left_above={bbs_la:.1} ratio={ratio:.3}"
            ));
            total_count += 1;
            if ratio < 1.1 {
                pass_count += 1;
            }
        }
    }

    let needed = 12usize;
    assert!(
        pass_count >= needed,
        "left+above boundary-RD BBS regressed vs left-only on too many cells: \
         {pass_count}/{total_count} passed.\n{}",
        report.join("\n")
    );
}

#[test]
fn hash_lock_checkerboard_q80_left_above() {
    // Byte-exact hash-lock with the Phase 5 defaults plus
    // `boundary_rd_above(true)`. The 64×64 uniform checkerboard tends
    // not to trigger refinement (its seams already match the original);
    // we still lock the output so future algorithmic drift (e.g. a
    // signed-vs-unsigned change in the above buffer, or a swap of the
    // rec/orig edge buffers) trips the test.
    let (w, h) = (64usize, 64usize);
    let rgb = gen_checkerboard(w, h, 8);
    let bytes = encode_rgb8(&rgb, w as u32, h as u32, 80, Mode::LeftAbove);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    let h64 = hasher.finish();
    eprintln!("hash_lock_left_above: size={} hash64={:016x}", bytes.len(), h64);

    // Record the produced values as the lock. If an intentional tuning
    // change invalidates these, refresh BOTH together and document the
    // reason in the commit message.
    const EXPECTED_SIZE: usize = 513;
    const EXPECTED_HASH: u64 = 0xaba706f19b94e26f;

    assert_eq!(
        bytes.len(),
        EXPECTED_SIZE,
        "hash-lock: left+above encoded size drift (got {}, want {})",
        bytes.len(),
        EXPECTED_SIZE
    );
    assert_eq!(
        h64, EXPECTED_HASH,
        "hash-lock: left+above encoded-byte hash drift (got {:#018x})",
        h64
    );

    let (dec, dw, dh) = decode_rgb8(&bytes);
    assert_eq!((dw, dh), (w as u32, h as u32));
    assert_eq!(dec.len(), w * h * 3);
}
