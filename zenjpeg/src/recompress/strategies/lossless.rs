//! Lossless re-pack via `crate::lossless::restructure`.
//!
//! No coefficient changes; only scan structure and Huffman tables are
//! rebuilt. This is the no-regression fallback path.

use crate::lossless::{OutputMode, RestartInterval, RestructureConfig, restructure};
use enough::Unstoppable;

use crate::recompress::error::Error;
use crate::recompress::source::SourceAnalysis;

/// Returns the lossless-restructured JPEG payload with exact
/// entropy-stage selection: the progressive restructure is produced
/// first (the measured-universal winner at scale), and when it lands at
/// or below the shared entropy-trial byte gate the sequential
/// restructure is also produced and the smaller ships. Both outputs
/// carry identical coefficients — a pure rate decision.
pub fn run_lossless(jpeg_bytes: &[u8], _analysis: &SourceAnalysis) -> Result<Vec<u8>, Error> {
    let prog = restructure(
        jpeg_bytes,
        &RestructureConfig {
            output_mode: OutputMode::Progressive,
            restart_interval: RestartInterval::None,
            transform: None,
        },
        Unstoppable,
    )?;
    if prog.len() > crate::encode::ENTROPY_TRIAL_MAX_BYTES {
        return Ok(prog);
    }
    let seq = restructure(
        jpeg_bytes,
        &RestructureConfig {
            output_mode: OutputMode::Sequential,
            restart_interval: RestartInterval::None,
            transform: None,
        },
        Unstoppable,
    )?;
    Ok(if seq.len() < prog.len() { seq } else { prog })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recompress::source::analyze_source;
    use enough::Unstoppable as EncUnstoppable;

    fn small_noise_jpeg() -> Vec<u8> {
        let (w, h) = (96u32, 80u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        let mut state = 0x853C_49E6_748F_EA9Bu64;
        for _ in 0..(w * h) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let n = (state >> 24) as u32;
            rgb.extend_from_slice(&[
                (n & 0xFF) as u8,
                ((n >> 8) & 0xFF) as u8,
                ((n >> 16) & 0xFF) as u8,
            ]);
        }
        let cfg =
            crate::encoder::EncoderConfig::ycbcr(70.0, crate::encoder::ChromaSubsampling::Quarter)
                .progressive(crate::encoder::ProgressiveScanMode::Baseline);
        let mut enc = cfg
            .encode_from_bytes(w, h, crate::encoder::PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&rgb, EncUnstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn lossless_emits_exact_min_of_output_modes() {
        let jpeg = small_noise_jpeg();
        let analysis = analyze_source(&jpeg).expect("analyze");

        let out = run_lossless(&jpeg, &analysis).expect("lossless");
        let explicit = |mode: OutputMode| {
            restructure(
                &jpeg,
                &RestructureConfig {
                    output_mode: mode,
                    restart_interval: RestartInterval::None,
                    transform: None,
                },
                Unstoppable,
            )
            .unwrap()
        };
        let prog = explicit(OutputMode::Progressive);
        let seq = explicit(OutputMode::Sequential);
        assert_eq!(
            out.len(),
            prog.len().min(seq.len()),
            "lossless must ship the exact min (prog={}, seq={})",
            prog.len(),
            seq.len(),
        );
    }
}
