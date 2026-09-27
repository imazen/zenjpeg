//! SIMD token-tier parity for the lossless transform pipeline.
//!
//! The lossless path (coefficient decode → D4 transform → Huffman re-encode)
//! is integer-only end to end, so — unlike the FP encoder pipeline, which
//! tolerates a few ULPs of cross-tier divergence — its output must be
//! BYTE-IDENTICAL across every archmage token permutation (AVX-512, AVX2,
//! SSE4.2, NEON, WASM128, scalar). Any divergence is a real bug in a SIMD
//! kernel on this path (entropy encoding, Huffman decode), not rounding.
//!
//! Kept as a single #[test] in its own integration binary: token permutation
//! mutates process-global dispatch state, so it must not run concurrently
//! with other tests in the same process.

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use enough::Unstoppable;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestartInterval, RestructureConfig,
    TransformConfig, restructure, transform,
};

fn noise_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut state: u32 = 0x2468_ACE1;
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h * 3) {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        px.push((state >> 24) as u8);
    }
    px
}

fn encode_jpeg(w: u32, h: u32, ss: ChromaSubsampling) -> Vec<u8> {
    let mut enc = EncoderConfig::ycbcr(90.0, ss)
        .progressive(false)
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&noise_rgb(w, h), Unstoppable).unwrap();
    enc.finish().unwrap()
}

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

#[test]
fn lossless_pipeline_byte_identical_across_token_tiers() {
    let subsamplings = [
        (ChromaSubsampling::None, "444"),
        (ChromaSubsampling::HalfHorizontal, "422"),
        (ChromaSubsampling::HalfVertical, "440"),
        (ChromaSubsampling::Quarter, "420"),
    ];
    // Aligned + both-axes-unaligned covers all four #194/#195 code paths.
    let sizes = [(64u32, 48u32), (66, 50)];

    for (ss, ss_name) in subsamplings {
        for (w, h) in sizes {
            // Encode the source ONCE (the FP encoder is allowed cross-tier
            // wobble; pinning the input isolates the lossless path).
            let jpeg = encode_jpeg(w, h, ss);

            for t in ALL_TRANSFORMS {
                let mut reference: Option<Vec<u8>> = None;
                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let out = transform(
                        &jpeg,
                        &TransformConfig {
                            transform: t,
                            edge_handling: EdgeHandling::TrimPartialBlocks,
                        },
                        Unstoppable,
                    )
                    .unwrap();
                    match &reference {
                        None => {
                            // Byte-identity across tiers passes even when every
                            // tier is corrupt — the reference must also decode.
                            zenjpeg::decode::DecodeConfig::new()
                                .decode(&out, Unstoppable)
                                .unwrap_or_else(|e| {
                                    panic!("{ss_name} {w}x{h} {t:?}: output does not decode: {e}")
                                });
                            reference = Some(out);
                        }
                        Some(r) => assert_eq!(
                            &out, r,
                            "{ss_name} {w}x{h} {t:?}: lossless transform output \
                             diverged at token permutation {perm}"
                        ),
                    }
                });
                assert!(report.permutations_run > 0);
            }

            // Progressive restructure across tiers.
            for out_mode in [OutputMode::Sequential, OutputMode::Progressive] {
                let mut reference: Option<Vec<u8>> = None;
                let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
                    let out = restructure(
                        &jpeg,
                        &RestructureConfig {
                            output_mode: out_mode,
                            restart_interval: RestartInterval::None,
                            transform: Some(TransformConfig {
                                transform: LosslessTransform::Rotate90,
                                edge_handling: EdgeHandling::TrimPartialBlocks,
                            }),
                        },
                        Unstoppable,
                    )
                    .unwrap();
                    match &reference {
                        None => {
                            zenjpeg::decode::DecodeConfig::new()
                                .decode(&out, Unstoppable)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "{ss_name} {w}x{h} {out_mode:?}: output does not decode: {e}"
                                    )
                                });
                            reference = Some(out);
                        }
                        Some(r) => assert_eq!(
                            &out, r,
                            "{ss_name} {w}x{h} {out_mode:?}: restructure output \
                             diverged at token permutation {perm}"
                        ),
                    }
                });
                assert!(report.permutations_run > 0);
            }
        }
    }
}
