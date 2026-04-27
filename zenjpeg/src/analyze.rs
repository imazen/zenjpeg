//! Re-export of the [`zenanalyze`] crate.
//!
//! Image content analyzers were extracted into the standalone
//! `zenanalyze` workspace crate so other codecs can share the same
//! oracle-trained feature pipeline. This module preserves a stable
//! `zenjpeg::analyze::*` path for the new opaque API.
//!
//! New code should reach for `zenanalyze` directly.

#[allow(unused_imports)]
pub use zenanalyze::feature::{
    AnalysisFeature, AnalysisQuery, AnalysisResults, FeatureSet, FeatureValue, ImageGeometry,
};
#[allow(unused_imports)]
pub use zenanalyze::{analyze_features, analyze_features_rgb8};
