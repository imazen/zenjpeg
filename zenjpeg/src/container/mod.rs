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

pub mod iso_jpeg;
pub mod marker;
pub mod mpf;
pub mod probe;
pub mod types;
pub mod xmp;

pub use iso_jpeg::{
    ISO_21496_1_URN, Iso21496Format, IsoJpegError, JpegIsoMarkers, create_iso_app2_marker,
    create_jpeg_iso_markers, create_version_only_iso_app2, parse_iso_app2, parse_iso21496,
    serialize_iso21496,
};
pub use marker::{
    MarkerIter, MarkerKind, MarkerSpan, find_jpeg_boundaries, for_each_jpeg_boundary, iter,
    primary_bounds,
};
pub use mpf::{MpfError, create_mpf_header, create_mpf_header_typed, parse_mpf, parse_mpf_segment};
pub use probe::{
    ContainerProbe, GainMapPresence, OverflowFlags, ProbeSof, Wants, is_ultrahdr, probe,
};
pub use types::{
    ContainerItem, ItemSemantic, MpImageType, MpfEntry, generate_container_directory,
    parse_container_items,
};
pub use xmp::{
    CONTAINER_NAMESPACE, HDRGM_NAMESPACE, ITEM_NAMESPACE, MAX_XMP_LENGTH, XmpError,
    create_xmp_app1_marker, generate_gainmap_xmp, generate_primary_xmp,
    generate_primary_xmp_with_items, generate_xmp, generate_xmp_with_items, parse_xmp,
    parse_xmp_full,
};
