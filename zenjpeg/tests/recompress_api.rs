//! Moved from the standalone zenjpeg-recompress crate (2026-05-29).
#![cfg(feature = "recompress")]

//! End-to-end smoke test: encode a synthetic JPEG, recompress it, check
//! the contract holds (no size regression, valid output bytes, sensible
//! strategy choice).

use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

use zenjpeg::recompress::{
    Budget, Confidence, RecompressOptions, RecompressResult, StrategyKind, recompress,
};

const W: u32 = 96;
const H: u32 = 96;

/// Make a 96×96 test image with some structured noise so DCT
/// coefficients are non-degenerate (a smooth gradient would be a
/// degenerate corpus per CLAUDE.md).
fn make_rgb_test_image() -> Vec<u8> {
    let mut v = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            // Diagonal stripes + per-pixel noise via xorshift.
            let s = ((x ^ y) as u32).wrapping_mul(2654435761);
            let r = ((x * 7 + y * 3) % 240 + (s & 0x0F)) as u8;
            let g = ((x * 5 + y * 11) % 220 + ((s >> 4) & 0x1F)) as u8;
            let b = ((x * 13 + y * 2) % 200 + ((s >> 9) & 0x3F)) as u8;
            v.push(r);
            v.push(g);
            v.push(b);
        }
    }
    v
}

fn make_source_jpeg(q: i32) -> Vec<u8> {
    let cfg = EncoderConfig::ycbcr(q, ChromaSubsampling::Quarter).progressive(true);
    let mut enc = cfg
        .encode_from_bytes(W, H, PixelLayout::Rgb8Srgb)
        .expect("encode setup");
    enc.push_packed(&make_rgb_test_image(), Unstoppable)
        .expect("encode push");
    enc.finish().expect("encode finish")
}

#[test]
fn recompress_high_source_to_lower_target_does_not_inflate() {
    let source = make_source_jpeg(90);
    let opts = RecompressOptions::new(70.0).with_budget(Budget::OneShot);
    let result = recompress(&source, &opts).expect("recompress");

    match result {
        RecompressResult::Recompressed {
            bytes,
            strategy,
            source_to_output_ratio,
            ..
        } => {
            assert!(
                bytes.len() <= source.len(),
                "no size regression: out={} src={} (ratio={})",
                bytes.len(),
                source.len(),
                source_to_output_ratio,
            );
            assert!(
                matches!(
                    strategy,
                    StrategyKind::Tuned | StrategyKind::Deblock | StrategyKind::Preserve
                ),
                "expected a recompression strategy at high source→mid target; got {strategy:?}",
            );
        }
        RecompressResult::LosslessOnly { bytes, .. } => {
            assert!(
                bytes.len() <= source.len(),
                "lossless re-pack should not inflate",
            );
        }
        RecompressResult::NoOp { .. } => {
            panic!("source Q90 should not NoOp to target 70 zensim-A");
        }
        _ => panic!("unexpected non_exhaustive RecompressResult variant"),
    }
}

#[test]
fn recompress_low_source_to_high_target_noops() {
    // A low-quality (Q30) source has estimated source-zensim-A ~50; if we
    // ask for target 90, we cannot improve — must NoOp.
    let source = make_source_jpeg(30);
    let opts = RecompressOptions::new(90.0);
    let result = recompress(&source, &opts).expect("recompress");
    assert!(matches!(result, RecompressResult::NoOp { .. }));
}

// Measurement requires the zensim-backed IQA path.
#[cfg(feature = "recompress-iqa")]
#[test]
fn recompress_with_iteration_budget_populates_measurement() {
    let source = make_source_jpeg(85);
    let opts = RecompressOptions::new(70.0).with_budget(Budget::MaxIterations(1));
    let result = recompress(&source, &opts).expect("recompress");
    match result {
        RecompressResult::Recompressed {
            measured_zensim_a: Some(m),
            ..
        } => {
            assert!(
                (0.0..=100.0).contains(&m),
                "measured zensim-A out of range: {m}",
            );
        }
        RecompressResult::Recompressed {
            measured_zensim_a: None,
            ..
        } => panic!("MaxIterations(1) should populate measured_zensim_a"),
        RecompressResult::LosslessOnly { .. } | RecompressResult::NoOp { .. } => {
            // Both are acceptable outcomes — they just don't carry a
            // measurement field. The test exists to ensure that *when*
            // we recompress, we measure. A LosslessOnly fallback is
            // still a legal outcome here.
        }
        _ => panic!("unexpected non_exhaustive RecompressResult variant"),
    }
}

#[test]
fn recompress_high_source_to_low_target_decodes_cleanly() {
    // Encode a source at Q90, recompress to target zensim-A 50.
    // Whatever strategy the router picks, the output must be a
    // decodable JPEG of the same dimensions.
    use enough::Unstoppable;
    use zenjpeg::decode::DecodeConfig;
    let source_bytes = make_source_jpeg(90);
    let result = recompress(
        &source_bytes,
        &RecompressOptions::new(50.0).with_budget(Budget::OneShot),
    )
    .expect("recompress");
    let bytes = match &result {
        RecompressResult::Recompressed { bytes, .. } => bytes,
        RecompressResult::LosslessOnly { bytes, .. } => bytes,
        _ => panic!("expected recompressed or lossless"),
    };
    let decoded = DecodeConfig::new()
        .decode(bytes, Unstoppable)
        .expect("output must decode");
    assert_eq!(decoded.width, W);
    assert_eq!(decoded.height, H);
}

/// Regression guard: `emit_preserved` with IDENTITY quant scale
/// must produce a JPEG whose decoded pixels match a fresh decode
/// of the source. (Catches the zigzag-DQT bug fixed 2026-05-28.)
#[cfg(feature = "recompress-expert")]
#[test]
fn preserve_identity_emit_is_pixel_identical() {
    use enough::Unstoppable as Unstop;
    use zenjpeg::decode::DecodeConfig;
    use zenjpeg::recompress::expert::{EmitConfig, QuantScale, emit_preserved};
    use zenjpeg::types::Subsampling;

    let cases: [(u32, u32, ChromaSubsampling, Subsampling); 7] = [
        // MCU-aligned baseline cases.
        (64, 64, ChromaSubsampling::None, Subsampling::S444),
        (64, 64, ChromaSubsampling::Quarter, Subsampling::S420),
        (
            128,
            128,
            ChromaSubsampling::HalfHorizontal,
            Subsampling::S422,
        ),
        (64, 64, ChromaSubsampling::HalfVertical, Subsampling::S440),
        // Partial-MCU cases (regression guard for the v0.2.3 fix).
        // 72×56 and 67×53 are 4:2:0 with non-MCU-aligned dimensions
        // that triggered the y_blocks_w stride mismatch in v0.2.2.
        (72, 56, ChromaSubsampling::Quarter, Subsampling::S420),
        (67, 53, ChromaSubsampling::Quarter, Subsampling::S420),
        (127, 89, ChromaSubsampling::Quarter, Subsampling::S420),
    ];
    for (w, h, chroma, subs) in cases {
        let rgb = {
            let mut v = Vec::with_capacity((w * h * 3) as usize);
            for y in 0..h {
                for x in 0..w {
                    let s: u32 = (x.wrapping_mul(2654435761u32) ^ y).wrapping_mul(2246822519u32);
                    v.push(((x * 7 + y * 3) % 240 + (s & 0x0F)) as u8);
                    v.push(((x * 5 + y * 11) % 220 + ((s >> 4) & 0x1F)) as u8);
                    v.push(((x * 13 + y * 2) % 200 + ((s >> 9) & 0x3F)) as u8);
                }
            }
            v
        };
        let cfg = EncoderConfig::ycbcr(zenjpeg::encoder::Quality::ApproxJpegli(85.0), chroma);
        let mut enc = cfg.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb).unwrap();
        enc.push_packed(&rgb, Unstop).unwrap();
        let source = enc.finish().unwrap();

        let coeffs = DecodeConfig::new()
            .decode_coefficients(&source, Unstop)
            .unwrap();
        let emit_cfg = EmitConfig::uniform_scale(QuantScale::IDENTITY);
        let emitted = emit_preserved(&coeffs, subs, &emit_cfg).expect("identity emit must succeed");
        let src_dec = DecodeConfig::new().decode(&source, Unstop).unwrap();
        let emit_dec = DecodeConfig::new()
            .decode(&emitted, Unstop)
            .expect("identity-emitted JPEG must decode cleanly");
        let sp = src_dec.pixels_u8().unwrap();
        let ep = emit_dec.pixels_u8().unwrap();
        assert_eq!(
            sp.len(),
            ep.len(),
            "{w}x{h} {chroma:?}: pixel length differs"
        );
        let mut n_diff = 0usize;
        for (a, b) in sp.iter().zip(ep.iter()) {
            if a != b {
                n_diff += 1;
            }
        }
        assert_eq!(
            n_diff,
            0,
            "{w}x{h} {chroma:?}: identity emit diverged at {} of {} bytes",
            n_diff,
            sp.len(),
        );
    }
}

/// Preserve must produce a JPEG with **different** quant tables and
/// fewer non-zero coefficients than the source when called with a
/// non-trivial quant scale. Catches a regression where Preserve
/// silently emits identity output.
#[cfg(feature = "recompress-expert")]
#[test]
fn preserve_actually_modifies_coefficients() {
    use enough::Unstoppable as Unstop;
    use zenjpeg::decode::DecodeConfig;
    use zenjpeg::recompress::expert::{AqMask, EmitConfig, QuantScale, emit_preserved};
    use zenjpeg::types::Subsampling;

    // Source: 64x64 Q=90 (high quality, est zensim ~97).
    let source = make_source_jpeg(90);
    let coeffs = DecodeConfig::new()
        .decode_coefficients(&source, Unstop)
        .expect("decode coefficients");

    // Emit with quant scale 4× (significantly tighter quantization)
    // and an AQ mask that zeros AC index 32..64 in all luma blocks.
    let n_luma_blocks = coeffs.components[0].num_blocks();
    let mut aq_mask: AqMask = Vec::with_capacity(n_luma_blocks);
    let zero_mask: u64 = (32..64).fold(0u64, |acc, i| acc | (1 << i));
    for _ in 0..n_luma_blocks {
        aq_mask.push(zero_mask);
    }
    let cfg = EmitConfig::uniform_scale(QuantScale {
        luma: 4.0,
        chroma: 4.0,
    })
    .with_aq_mask(Some(aq_mask));
    let emitted = emit_preserved(&coeffs, Subsampling::S420, &cfg)
        .expect("non-trivial preserve emit must succeed");
    assert!(!emitted.is_empty());

    // 1. Quant tables in the emit must differ from source's.
    let emit_coeffs = DecodeConfig::new()
        .decode_coefficients(&emitted, Unstop)
        .expect("emitted decodes");
    let src_q = coeffs.quant_tables[0].expect("source has q-table 0");
    let emit_q = emit_coeffs.quant_tables[0].expect("emit has q-table 0");
    assert_ne!(
        src_q, emit_q,
        "quant scale 4× should produce different quant tables",
    );
    // Quant values should be ~4× source (clamped to 255).
    let src_dc = src_q[0] as f32;
    let emit_dc = emit_q[0] as f32;
    assert!(
        (emit_dc - src_dc * 4.0).abs() < 1.0 || emit_dc >= 255.0,
        "DC quant should scale 4× (or clamp): src={src_dc} emit={emit_dc}",
    );

    // 2. Re-quantized coefficients should have more zeros (because
    // larger quant divisor + AC zeroing) than source.
    let src_luma = &coeffs.components[0].coeffs;
    let emit_luma = &emit_coeffs.components[0].coeffs;
    let src_zeros = src_luma.iter().filter(|&&v| v == 0).count();
    let emit_zeros = emit_luma.iter().filter(|&&v| v == 0).count();
    assert!(
        emit_zeros > src_zeros,
        "preserve must zero more coefficients (src={src_zeros}, emit={emit_zeros})",
    );

    // 3. Specifically, AQ-masked AC indices 32..64 must be zero in
    // every luma block.
    for block_idx in 0..emit_coeffs.components[0].num_blocks() {
        let block = emit_coeffs.components[0].block(block_idx);
        for i in 32..64 {
            assert_eq!(
                block[i], 0,
                "block {block_idx} AC index {i} must be zero via AQ mask",
            );
        }
    }
}

/// Higher delivery confidence aims the encoder higher, so the output
/// is at least as large (more conservative) as a lower-confidence call
/// at the same target — never smaller. (At the extreme, high confidence
/// falls through to the lossless re-pack rather than risk undershoot.)
#[test]
fn higher_confidence_is_at_least_as_conservative() {
    let source = make_source_jpeg(90);
    let out_len = |c: Confidence| -> usize {
        match recompress(&source, &RecompressOptions::new(50.0).with_confidence(c)).unwrap() {
            RecompressResult::Recompressed { bytes, .. }
            | RecompressResult::LosslessOnly { bytes, .. } => bytes.len(),
            RecompressResult::NoOp { .. } => source.len(),
            _ => unreachable!(),
        }
    };
    let p25 = out_len(Confidence::P25);
    let p50 = out_len(Confidence::P50);
    let p90 = out_len(Confidence::P90);
    let p95 = out_len(Confidence::P95);
    assert!(
        p50 >= p25,
        "P50 ({p50}) must be ≥ P25 ({p25}) — lower confidence aims lower",
    );
    assert!(
        p90 >= p50,
        "P90 ({p90}) must be ≥ P50 ({p50}) — higher confidence aims higher",
    );
    assert!(p95 >= p90, "P95 ({p95}) must be ≥ P90 ({p90})",);
    // All must still honor no-size-regression.
    assert!(p95 <= source.len(), "P95 must not inflate vs source");
}

/// Regression guard for the shared-chroma double-scale bug (fixed
/// 2026-05-28). Sources with a 2-table layout (luma + SHARED chroma —
/// turbo/mozjpeg) must not have their chroma table scaled once per
/// chroma component. We force a 2-table source via
/// `separate_chroma_tables(false)`, requantize at uniform 2×, and
/// assert the chroma table is exactly 2× the source (not 4×).
#[cfg(feature = "recompress-expert")]
#[test]
fn preserve_shared_chroma_not_double_scaled() {
    use enough::Unstoppable as Unstop;
    use zenjpeg::decode::DecodeConfig;
    use zenjpeg::recompress::expert::{EmitConfig, QuantScale, emit_preserved};
    use zenjpeg::types::Subsampling;

    // 2-table source: shared chroma (Cb and Cr both reference table 1).
    let cfg = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).separate_chroma_tables(false);
    let mut enc = cfg
        .encode_from_bytes(W, H, PixelLayout::Rgb8Srgb)
        .expect("encode setup");
    enc.push_packed(&make_rgb_test_image(), Unstop)
        .expect("push");
    let source = enc.finish().expect("finish");

    let coeffs = DecodeConfig::new()
        .decode_coefficients(&source, Unstop)
        .expect("decode coeffs");
    // Confirm it's a 2-table shared-chroma layout (else the test is moot).
    assert!(
        coeffs.components.len() == 3
            && coeffs.components[1].quant_table_idx == coeffs.components[2].quant_table_idx,
        "test needs a shared-chroma (2-table) source",
    );
    let chroma_idx = coeffs.components[1].quant_table_idx as usize;
    let src_chroma = coeffs.quant_tables[chroma_idx].expect("src chroma table");

    let emitted = emit_preserved(
        &coeffs,
        Subsampling::S420,
        &EmitConfig::uniform_scale(QuantScale {
            luma: 2.0,
            chroma: 2.0,
        }),
    )
    .expect("emit");
    let emit_coeffs = DecodeConfig::new()
        .decode_coefficients(&emitted, Unstop)
        .expect("emit decode");
    let emit_chroma = emit_coeffs.quant_tables[chroma_idx].expect("emit chroma table");

    // Chroma table must be 2× source (clamped to 255), NOT 4× (double-scale bug).
    for i in 0..64 {
        let want = ((src_chroma[i] as u32 * 2).min(255)) as u16;
        assert_eq!(
            emit_chroma[i], want,
            "chroma quant[{i}] = {} (want 2× = {want}); 4× would mean double-scale bug",
            emit_chroma[i],
        );
    }
}

#[test]
fn recompress_rejects_out_of_range_target() {
    let source = make_source_jpeg(80);
    let opts = RecompressOptions::new(150.0);
    let err = recompress(&source, &opts).expect_err("must reject");
    let msg = format!("{err}");
    assert!(msg.contains("out of range"), "{msg}");
}
