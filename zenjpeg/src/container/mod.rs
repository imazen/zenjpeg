//! Public JPEG container primitives shared across zen crates.
//!
//! This module owns the JPEG **structural** concerns — marker iteration,
//! image boundary detection, and the parsers/emitters for metadata
//! containers that live in JPEG APP segments (MPF, XMP, ISO 21496-1
//! envelope). Deeper JPEG codec concerns (DCT, Huffman, upsampling) stay
//! in other zenjpeg modules; this module never decompresses pixel data.
//!
//! # Scope
//!
//! - [`marker`]: zero-copy iterator over top-level JPEG marker segments.
//!   Uses [memchr](::memchr) for SIMD byte search in entropy-coded scan
//!   data and in the gaps between concatenated images in multi-image
//!   files. Provides [`marker::primary_bounds`] and
//!   [`marker::find_jpeg_boundaries`] as convenience helpers.
//!
//! # Performance contract
//!
//! The marker iterator MUST stay within 3% of the naive byte-scan on
//! entropy-heavy inputs and SHOULD beat it by using memchr on inputs
//! with long pre-SOI padding, long inter-image padding (multi-image
//! JPEGs), or low-FF-density entropy streams. See
//! `benches/container_scan.rs` and
//! `benchmarks/container_scan_baseline_2026-04-17.{csv,md}`.

pub mod marker;

pub use marker::{MarkerIter, MarkerKind, MarkerSpan, find_jpeg_boundaries, iter, primary_bounds};
