//! Integration tests for the new `EncoderConfig::chroma_quality`
//! knob on the mozjpeg-compat quant-table path.
//!
//! Three properties to verify:
//!
//!   1. **Identity**: `chroma_quality(None)` and
//!      `chroma_quality(Some(q))` where `q == luma_quality` both
//!      produce bit-identical output to a config that never touched
//!      the setter. Critical — this ensures the new code path is
//!      strictly additive on mozjpeg-compat encoders.
//!
//!   2. **Monotonicity**: on an image with real chroma signal, lower
//!      `chroma_quality` produces a smaller file than higher
//!      `chroma_quality` at fixed luma quality. Same shape as the
//!      `chroma_distance_scale` test but expressed in the mozjpeg
//!      quality domain.
//!
//!   3. **No-op on jpegli path**: setting `chroma_quality` while the
//!      quant-table config is jpegli (the default) produces
//!      bit-identical output to leaving it unset — the knob must be
//!      silently ignored by the non-mozjpeg path.

use enough::Unstoppable;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout, QuantTableConfig};

fn sharp_chroma_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

fn encode(config: &EncoderConfig, rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("builder");
    enc.push_packed(rgb, Unstoppable).expect("push");
    enc.finish().expect("finish")
}

#[test]
fn chroma_quality_none_is_bit_identical_to_unset_on_mozjpeg() {
    let (w, h) = (128, 128);
    let rgb = sharp_chroma_rgb(w, h);

    for q in [40u8, 75, 90] {
        for sub in [
            ChromaSubsampling::None,
            ChromaSubsampling::Quarter,
            ChromaSubsampling::HalfHorizontal,
        ] {
            // Both configs use the mozjpeg-Robidoux table path; the
            // only difference is whether `chroma_quality` was touched.
            let base =
                EncoderConfig::ycbcr(q, sub).quant_table_config(QuantTableConfig::MozjpegRobidoux);
            let explicit_none = base.clone().chroma_quality(None);
            let same_as_luma = base.clone().chroma_quality(Some(q));

            let a = encode(&base, &rgb, w, h);
            let b = encode(&explicit_none, &rgb, w, h);
            let c = encode(&same_as_luma, &rgb, w, h);

            assert_eq!(
                a, b,
                "q={q}, sub={:?}: chroma_quality(None) must be bit-identical to unset",
                sub as u8,
            );
            assert_eq!(
                a, c,
                "q={q}, sub={:?}: chroma_quality(Some(q)) must be bit-identical to unset",
                sub as u8,
            );
        }
    }
}

#[test]
fn chroma_quality_monotone_file_size_on_mozjpeg_420() {
    // Gradient RGB has meaningful Cb/Cr signal. Lower chroma_quality
    // at fixed luma quality should compress chroma MORE → smaller file.
    let (w, h) = (256, 256);
    let rgb = sharp_chroma_rgb(w, h);
    let luma_q = 85u8;

    let base = EncoderConfig::ycbcr(luma_q, ChromaSubsampling::Quarter)
        .quant_table_config(QuantTableConfig::MozjpegRobidoux);

    let high = encode(&base.clone().chroma_quality(Some(90)), &rgb, w, h);
    let same = encode(&base.clone().chroma_quality(Some(85)), &rgb, w, h);
    let low = encode(&base.clone().chroma_quality(Some(50)), &rgb, w, h);

    // Monotone: lower chroma_quality → smaller file at fixed luma_q.
    assert!(
        high.len() >= same.len(),
        "chroma_quality 90 ({}) should be ≥ 85 ({})",
        high.len(),
        same.len()
    );
    assert!(
        same.len() >= low.len(),
        "chroma_quality 85 ({}) should be ≥ 50 ({})",
        same.len(),
        low.len()
    );
    // At least one strict inequality — otherwise the knob is no-op.
    assert!(
        high.len() > low.len(),
        "chroma_quality 90 ({}) should be strictly larger than 50 ({})",
        high.len(),
        low.len()
    );
}

#[test]
fn chroma_quality_is_noop_on_jpegli_path() {
    // The jpegli perceptual path uses chroma_distance_scale, not
    // chroma_quality. Setting chroma_quality here must silently
    // pass through — output byte-identical to leaving it unset.
    let (w, h) = (128, 128);
    let rgb = sharp_chroma_rgb(w, h);

    let jpegli_base = EncoderConfig::ycbcr(75u8, ChromaSubsampling::Quarter);
    // Default quant_table_config is QuantTableConfig::Jpegli (3-table).
    let jpegli_with_cq = jpegli_base.clone().chroma_quality(Some(60));

    let a = encode(&jpegli_base, &rgb, w, h);
    let b = encode(&jpegli_with_cq, &rgb, w, h);
    assert_eq!(
        a, b,
        "jpegli-path encoder must ignore chroma_quality — use chroma_distance_scale instead"
    );
}

#[test]
fn chroma_quality_clamped_to_1_100() {
    let cfg = EncoderConfig::ycbcr(75u8, ChromaSubsampling::Quarter).chroma_quality(Some(150));
    assert_eq!(cfg.get_chroma_quality(), Some(100));

    let cfg = EncoderConfig::ycbcr(75u8, ChromaSubsampling::Quarter).chroma_quality(Some(0));
    assert_eq!(cfg.get_chroma_quality(), Some(1));

    let cfg = EncoderConfig::ycbcr(75u8, ChromaSubsampling::Quarter).chroma_quality(None);
    assert_eq!(cfg.get_chroma_quality(), None);
}
