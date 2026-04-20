//! Quality metrics for codec evaluation.
//!
//! This module hosts metrics that are specific to codec analysis (e.g. the
//! block-boundary score used for JPEG blocking studies). General-purpose
//! perceptual metrics (SSIMULACRA2, Butteraugli, DSSIM) live in separate
//! crates — this module is for codec-internal or JPEG-specific measures.

pub mod bbs;

pub use bbs::{BbsResult, bbs_planar_u8, bbs_rgb8};
