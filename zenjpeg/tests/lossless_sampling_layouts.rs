//! Sampling layouts the lossless emitters must walk correctly.
//!
//! 1. A frame with one component has only non-interleaved scans (T.81
//!    A.2.2): each MCU is one block of the true grid, whatever sampling
//!    factors the frame declares. The sequential emitter walked a grayscale
//!    frame declaring 4x1 sampling in 4-block MCUs over the padded grid,
//!    writing extra blocks that every reader takes as the next row
//!    (libjpeg-turbo: "extraneous bytes before marker 0xd9", wrong pixels).
//!    Oracle: the declared factors mean nothing for one component, so the file
//!    and the same file declaring other factors give identical outputs.
//! 2. The progressive emitter interleaved its DC scan whenever component 0
//!    was sampled 1x1, which is the interleaved order only when every
//!    component is. With chroma sampled above luma (luma 1x1, chroma 2x1;
//!    legal, written by `cjpeg -sample`) every progressive restructure
//!    decoded to wrong pixels. Fixtures: `tests/crash_repro/
//!    lossless_progressive_chroma_above_luma_*.jpg` (24x16, written by
//!    `cjpeg -sample`; libjpeg-turbo decodes the restructured output to the
//!    input's pixels).
//!
//! Both found by the `fuzz_lossless` coefficient oracle.

use enough::Unstoppable;
use zenjpeg::decode::DecodeConfig;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};
use zenjpeg::lossless::{
    EdgeHandling, LosslessTransform, OutputMode, RestartInterval, RestructureConfig,
    TransformConfig, restructure, transform,
};

fn gray(w: u32, h: u32, progressive: bool) -> Vec<u8> {
    let px: Vec<u8> = (0..w * h)
        .map(|i| ((i * 73 + i / w * 41) % 251) as u8)
        .collect();
    let mut enc = EncoderConfig::grayscale(90)
        .progressive(progressive)
        .encode_from_bytes(w, h, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&px, Unstoppable).unwrap();
    enc.finish().unwrap()
}

/// `jpeg` with its one component declaring `sampling` (`0xHV`).
fn declare(jpeg: &[u8], sampling: u8) -> Vec<u8> {
    let sof = jpeg
        .windows(2)
        .position(|w| w[0] == 0xFF && matches!(w[1], 0xC0..=0xC2))
        .unwrap();
    assert_eq!(jpeg[sof + 9], 1, "one component");
    let mut out = jpeg.to_vec();
    out[sof + 11] = sampling;
    out
}

fn pixels(jpeg: &[u8]) -> (u32, u32, Vec<u8>) {
    let d = DecodeConfig::new().decode(jpeg, Unstoppable).unwrap();
    (d.width(), d.height(), d.pixels_u8().unwrap().to_vec())
}

#[test]
fn one_component_frames_ignore_declared_sampling() {
    let transforms = [
        LosslessTransform::None,
        LosslessTransform::FlipHorizontal,
        LosslessTransform::FlipVertical,
        LosslessTransform::Rotate90,
        LosslessTransform::Rotate180,
        LosslessTransform::Rotate270,
        LosslessTransform::Transpose,
        LosslessTransform::Transverse,
    ];
    // 292x259: neither axis a multiple of 8, nor of the 32-px MCU a 4x1
    // declaration would imply.
    for progressive in [false, true] {
        let plain = gray(292, 259, progressive);
        assert_eq!(pixels(&plain), pixels(&declare(&plain, 0x41)));
        for sampling in [0x41, 0x14, 0x22] {
            let declared = declare(&plain, sampling);
            let label = format!("progressive {progressive} sampling {sampling:#x}");
            for t in transforms {
                let config = TransformConfig {
                    transform: t,
                    edge_handling: EdgeHandling::TrimPartialBlocks,
                };
                let want = transform(&plain, &config, Unstoppable).unwrap();
                let got = transform(&declared, &config, Unstoppable)
                    .unwrap_or_else(|e| panic!("{label} {t:?}: {e}"));
                assert_eq!(pixels(&got), pixels(&want), "{label} {t:?}");
            }
            for (mode, restart) in [
                (OutputMode::Sequential, RestartInterval::None),
                (OutputMode::Sequential, RestartInterval::EveryMcuRows(1)),
                (OutputMode::Progressive, RestartInterval::None),
            ] {
                let config = RestructureConfig {
                    output_mode: mode,
                    restart_interval: restart,
                    ..RestructureConfig::default()
                };
                let want = restructure(&plain, &config, Unstoppable).unwrap();
                let got = restructure(&declared, &config, Unstoppable)
                    .unwrap_or_else(|e| panic!("{label} {mode:?} {restart:?}: {e}"));
                assert_eq!(pixels(&got), pixels(&want), "{label} {mode:?} {restart:?}");
            }
        }
    }
}

#[test]
fn progressive_output_handles_chroma_sampled_above_luma() {
    for name in [
        "lossless_progressive_chroma_above_luma_2x1.jpg",
        "lossless_progressive_chroma_above_luma_2x3.jpg",
    ] {
        let path = format!("{}/tests/crash_repro/{name}", env!("CARGO_MANIFEST_DIR"));
        let jpeg = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let want = pixels(&jpeg);
        for (mode, restart) in [
            (OutputMode::Progressive, RestartInterval::None),
            (OutputMode::Progressive, RestartInterval::EveryMcuRows(1)),
            (OutputMode::Sequential, RestartInterval::None),
        ] {
            let config = RestructureConfig {
                output_mode: mode,
                restart_interval: restart,
                ..RestructureConfig::default()
            };
            let out = restructure(&jpeg, &config, Unstoppable)
                .unwrap_or_else(|e| panic!("{name} {mode:?}: {e}"));
            assert_eq!(pixels(&out), want, "{name} {mode:?} {restart:?}");
        }
    }
}
