//! Per-iMCU AQ-strength controller hook (issue #113, PR-A scaffold).
//!
//! Sits between [`StreamingAQ`](crate::quant::aq::streaming::StreamingAQ)
//! emitting per-iMCU AQ strengths and [`StripProcessor::quantize_prev_pending_imcu`]
//! consuming them. A controller may scale the strengths in place; with no
//! controller installed (the default), the strength buffer is passed through
//! unmodified and the encode is byte-identical to the pre-controller path.
//!
//! This is the injection point that future layers of the target-zensim
//! closed-loop encoder build on:
//!
//! - **Layer 2** (this PR): trait + injection point. No reference impl yet.
//! - **Layer 3** (PR-C): a measurement driver pushes
//!   [`WindowObservation`]s back to the controller via [`AqController::observe`].
//! - **Layer 4** (PR-D): `PiAqController` reference implementation.
//!
//! The trait surface is intentionally `pub(crate)` until the public
//! `Quality::TargetZensim` API lands. Reference: GitHub issue #113.

use core::ops::Range;

/// Observation pushed back to the controller from the per-window
/// measurement loop (Layer 3, PR-C). Carries the iMCU range that was
/// measured, the *predicted* zensim contribution from the controller's
/// model, and the *actual* contribution measured from the just-quantized
/// blocks.
///
/// `predicted - measured` is the controller error signal.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are read by Layer 3+; PR-A only constructs/passes.
pub(crate) struct WindowObservation {
    /// Range of iMCU rows covered by this observation (half-open).
    pub imcu_range: Range<usize>,
    /// Zensim contribution this window actually produced, as accumulated
    /// by [`zensim::PrecomputedReference`]-backed streaming math.
    pub measured_zensim_contribution: f32,
    /// Predicted contribution at the time the controller chose its
    /// per-iMCU AQ scale. Used as the error baseline.
    pub predicted_zensim_contribution: f32,
}

/// Hook for adjusting per-iMCU AQ strengths during a streaming encode.
///
/// `adjust` is called once per iMCU row, immediately after `StreamingAQ`
/// finalizes that iMCU's strengths and before the strip processor
/// quantizes the corresponding f32 DCT blocks. Implementations may
/// scale, clamp, or otherwise mutate `strengths` in place.
///
/// `observe` is called by the per-window measurement driver (Layer 3,
/// PR-C) to push measured zensim contributions back to the controller.
/// Default no-op so simple controllers don't need to implement it.
///
/// Intentionally object-safe (`dyn AqController`) — controllers carry
/// their own state (PI accumulators, classification, etc.) and
/// `StripProcessor` holds at most one boxed instance. `Debug` is a
/// supertrait so `StripProcessor`'s `#[derive(Debug)]` keeps working.
pub(crate) trait AqController: Send + core::fmt::Debug {
    /// Called once per iMCU row after `StreamingAQ` emits strengths and
    /// before quantization consumes them.
    ///
    /// `strengths` is the slice the strip processor will pass directly
    /// to `quantize_prev_pending_imcu`; mutations are reflected
    /// downstream. `imcu_idx` is the 0-based index of the iMCU row this
    /// call is adjusting (monotonic across the encode).
    fn adjust(&mut self, strengths: &mut [f32], imcu_idx: usize);

    /// Push a per-window measurement back to the controller. Default
    /// implementation drops the observation; non-feedback controllers
    /// (e.g. fixed-bias) don't need to override.
    ///
    /// Currently unused — Layer 3 measurement driver lands in PR-C.
    #[allow(unused_variables, dead_code)]
    fn observe(&mut self, window: WindowObservation) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A controller that records every `adjust` call. Used by the
    /// scaffolding test to confirm the trait is wired correctly.
    #[derive(Debug)]
    struct RecordingController {
        seen: Vec<(usize, Vec<f32>)>,
    }
    impl AqController for RecordingController {
        fn adjust(&mut self, strengths: &mut [f32], imcu_idx: usize) {
            self.seen.push((imcu_idx, strengths.to_vec()));
        }
    }

    /// Sanity: the trait is object-safe and a boxed controller can be
    /// driven through `&mut dyn AqController`.
    #[test]
    fn controller_is_object_safe() {
        let mut ctrl: Box<dyn AqController> = Box::new(RecordingController { seen: vec![] });
        let mut strengths = [0.1f32, 0.2, 0.3];
        ctrl.adjust(&mut strengths, 0);
        ctrl.adjust(&mut strengths, 1);
        // Push an observation through the default no-op `observe`.
        ctrl.observe(WindowObservation {
            imcu_range: 0..2,
            measured_zensim_contribution: 0.0,
            predicted_zensim_contribution: 0.0,
        });
    }

    /// A controller can scale strengths in place; the change is visible
    /// to the caller (this is what the strip processor relies on).
    #[test]
    fn adjust_scales_in_place() {
        #[derive(Debug)]
        struct DoubleEverything;
        impl AqController for DoubleEverything {
            fn adjust(&mut self, strengths: &mut [f32], _imcu_idx: usize) {
                for s in strengths {
                    *s *= 2.0;
                }
            }
        }
        let mut ctrl: Box<dyn AqController> = Box::new(DoubleEverything);
        let mut strengths = [0.10f32, 0.20];
        ctrl.adjust(&mut strengths, 0);
        assert_eq!(strengths, [0.20, 0.40]);
    }
}
