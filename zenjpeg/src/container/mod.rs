//! Public JPEG container primitives shared across zen crates.
//!
//! This module owns the JPEG **structural** concerns — marker iteration,
//! image boundary detection, and the parsers/emitters for metadata
//! containers that live in JPEG APP segments (MPF, XMP). Deeper JPEG
//! codec concerns (DCT, Huffman, upsampling) stay in other zenjpeg
//! modules; this module never decompresses pixel data.
//!
//! # Scope
//!
//! - [`marker`]: zero-copy iterator over top-level JPEG marker segments.
//!   Uses [memchr](::memchr) for SIMD byte search in entropy-coded scan
//!   data and in the gaps between concatenated images in multi-image
//!   files. Provides [`marker::primary_bounds`] and
//!   [`marker::find_jpeg_boundaries`] as convenience helpers.
//! - [`mpf`]: MPF APP2 segment parsing + emission.
//! - [`xmp`]: XMP APP1 segment parsing + emission.
//! - [`types`]: shared container types (`ContainerItem`, `MpfEntry`, etc.).
//! - [`probe`]: single-pass probe returning every requested signal in
//!   one marker walk.
//!
//! The ISO 21496-1 JPEG APP2 envelope is not a separate module —
//! [`zencodec::ISO_21496_1_URN`] and [`zencodec::ISO_21496_1_PRIMARY_APP2_BODY`]
//! provide the namespace bytes, and [`marker::append_app_segment`] emits the
//! `FF E2 + length` framing around them. Ultra HDR writers/readers assemble
//! the envelope directly from those primitives.
//!
//! # Performance contract
//!
//! The marker iterator MUST stay within 3% of the naive byte-scan on
//! entropy-heavy inputs and SHOULD beat it by using memchr on inputs
//! with long pre-SOI padding, long inter-image padding (multi-image
//! JPEGs), or low-FF-density entropy streams. See
//! `benches/container_scan.rs` and
//! `benchmarks/container_scan_2026-04-20.{csv,md}`.

pub mod marker;
pub mod mpf;
pub mod probe;
pub mod types;
pub mod xmp;

pub use marker::{MarkerIter, MarkerKind, MarkerSpan, find_jpeg_boundaries, iter, primary_bounds};
pub use mpf::{MpfError, create_mpf_header, create_mpf_header_typed, parse_mpf, parse_mpf_segment};
pub use probe::{ContainerProbe, GainMapPresence, ProbeSof, Wants, is_ultrahdr, probe};
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
