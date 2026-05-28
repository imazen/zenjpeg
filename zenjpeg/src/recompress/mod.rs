//! Recompress an already-encoded JPEG to a target zensim Profile A
//! quality with minimal generation loss and no size regression.
//!
//! Entry point: [`recompress`]. The default [`Budget::OneShot`] path
//! needs only `zenjpeg` itself; the optional closed loop ([`Budget::
//! MaxIterations`] > 1 / [`Budget::MaxTime`]) measures generation loss
//! against the source and is gated behind the `recompress-iqa` feature
//! (which pulls in `zensim`). The `recompress-expert` feature exposes the
//! [`expert`] internals.
//!
//! Moved into the zenjpeg crate 2026-05-29 (was the standalone
//! `zenjpeg-recompress` crate); reaches zenjpeg's `pub(crate)` codec
//! internals directly, so no `__test-utils` exposure is needed.

mod api;
mod aq;
mod budget;
mod calibration;
mod error;
#[cfg(feature = "recompress-iqa")]
mod measure;
mod router;
mod source;
mod strategies;
mod target;

pub use api::{
    Budget, Confidence, LosslessReason, NoOpReason, RecompressOptions, RecompressResult,
    StrategyKind, recompress,
};
pub use error::Error;

/// Expert-only internals. Stability: NOT covered by semver — anything
/// under this module can change between any two releases. Requires the
/// `recompress-expert` feature.
#[cfg(feature = "recompress-expert")]
pub mod expert {
    pub use crate::recompress::aq::{
        ActivityTier, build_aq_mask, build_aq_mask_busy, classify_block,
        mask_low_activity_fraction, tier_histogram,
    };
    pub use crate::recompress::calibration::{
        CalibrationLookup, CellCi, CellEstimate, EncoderClass, StrategyChoice, TableId,
    };
    pub use crate::recompress::router::{
        RouterInput, RouterOutput, StrategyParams, decide_strategy,
    };
    pub use crate::recompress::source::{SourceAnalysis, analyze_source};
    pub use crate::recompress::strategies::preserve_emit::{
        AqMask, EmitConfig, QuantScale, QuantStrategy, emit_preserved,
    };
    pub use crate::recompress::strategies::{
        deblock::run_deblock, lossless::run_lossless, preserve::run_preserve, tuned::run_tuned,
    };
    pub use crate::recompress::target::{target_zensim_a_to_ba_distance, target_zensim_a_to_ijg_q};

    /// IQA scoring helpers (require the `recompress-iqa` feature → `zensim`).
    #[cfg(feature = "recompress-iqa")]
    pub use crate::recompress::measure::{score_against_reference, score_recompression};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the frozen public API surface — fails fast if names shift.
    #[test]
    fn public_api_is_frozen() {
        let _opts = RecompressOptions {
            target_zensim_a: 80.0,
            budget: Budget::OneShot,
            confidence: Confidence::P50,
        };
        let _opts2 = RecompressOptions::new(80.0)
            .with_budget(Budget::OneShot)
            .with_confidence(Confidence::P90);
        let _r1: RecompressResult = RecompressResult::NoOp {
            reason: NoOpReason::SourceAlreadyMeetsTarget,
        };
        let _r2: RecompressResult = RecompressResult::LosslessOnly {
            bytes: Vec::new(),
            reason: LosslessReason::RecompressionWouldInflateAtTarget,
        };
        let _r3: RecompressResult = RecompressResult::Recompressed {
            bytes: Vec::new(),
            strategy: StrategyKind::Preserve,
            projected_zensim_a: 80.0,
            measured_zensim_a: None,
            source_to_output_ratio: 0.85,
        };
    }
}
