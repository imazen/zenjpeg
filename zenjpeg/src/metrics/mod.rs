//! Quality metrics for codec evaluation (internal).
//!
//! This module hosts metrics specific to codec analysis (BBS block-boundary
//! score, plus an RD-curve / BD-rate harness) used internally for
//! boundary-RD validation. General-purpose perceptual metrics
//! (SSIMULACRA2, Butteraugli, DSSIM) live in separate crates.
//!
//! Items here are NOT part of the versioned public API. The whole module
//! is gated behind `pub(crate)` by default; the `test-utils` Cargo feature
//! reveals it for use by internal tests and example CLIs.

pub mod bbs;
pub mod rd;
pub mod sweep;

pub use bbs::{BbsResult, bbs_planar_u8, bbs_rgb8};
pub use rd::{RdComparison, RdCurve, RdPoint, bd_rate, closest_point_distance};
pub use sweep::{
    CorpusImage, ImageClass, MetricKind, PointResult, SampleOutput, SweepResult, run_sweep,
};
