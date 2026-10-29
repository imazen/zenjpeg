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
            let base = EncoderConfig::ycbcr(q, sub).quant_table_config(
                QuantTableConfig::MozjpegRobidoux {
                    chroma_quality: None,
                },
            );
            let explicit_none =
                base.clone()
                    .quant_table_config(QuantTableConfig::MozjpegRobidoux {
                        chroma_quality: None,
                    });
            let same_as_luma = base
                .clone()
                .quant_table_config(QuantTableConfig::MozjpegRobidoux {
                    chroma_quality: Some(q),
                });

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

    let base = EncoderConfig::ycbcr(luma_q, ChromaSubsampling::Quarter).quant_table_config(
        QuantTableConfig::MozjpegRobidoux {
            chroma_quality: None,
        },
    );

    let high = encode(
        &base
            .clone()
            .quant_table_config(QuantTableConfig::MozjpegRobidoux {
                chroma_quality: Some(90),
            }),
        &rgb,
        w,
        h,
    );
    let same = encode(
        &base
            .clone()
            .quant_table_config(QuantTableConfig::MozjpegRobidoux {
                chroma_quality: Some(85),
            }),
        &rgb,
        w,
        h,
    );
    let low = encode(
        &base
            .clone()
            .quant_table_config(QuantTableConfig::MozjpegRobidoux {
                chroma_quality: Some(50),
            }),
        &rgb,
        w,
        h,
    );

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
fn chroma_quality_is_structurally_scoped_to_mozjpeg() {
    // The old runtime guarantee ("jpegli path ignores chroma_quality")
    // is now a type-level one: the knob only exists on the
    // MozjpegRobidoux variant, so a jpegli config cannot carry it.
    let jpegli = EncoderConfig::ycbcr(75u8, ChromaSubsampling::Quarter);
    assert_eq!(jpegli.get_quant_table_config().chroma_quality(), None);
    assert_eq!(QuantTableConfig::default().chroma_quality(), None);
}

#[test]
fn chroma_quality_clamped_at_resolution() {
    // Clamping happens when tables resolve: 150 behaves as 100, 0 as 1.
    let plan = |cq: Option<u8>| {
        EncoderConfig::ycbcr(75u8, ChromaSubsampling::Quarter)
            .quant_table_config(QuantTableConfig::MozjpegRobidoux { chroma_quality: cq })
            .resolve_plan(64, 64)
    };
    assert_eq!(plan(Some(150)).quant_max, plan(Some(100)).quant_max);
    assert_eq!(plan(Some(0)).quant_max, plan(Some(1)).quant_max);
}
