//! Shared wire-format constants for the ZNPR v3 binary format.
//!
//! These offsets and sizes are byte-exact and used by both the parser
//! ([`crate::Model::from_bytes`]) and the composer in the sibling
//! `zenpredict-bake` crate. Keeping them in one module guarantees the
//! two sides can't drift — a parser-side offset change must update
//! the composer in lockstep and vice versa.
//!
//! See [`crate::Header`] for the full byte layout documentation.

/// Size of the fixed-format header in bytes.
pub const HEADER_SIZE: usize = 128;

/// Size of one [`crate::LayerEntry`] in bytes.
pub const LAYER_ENTRY_SIZE: usize = 48;

/// Header byte offset of the `scaler_mean` [`crate::Section`].
pub const SECTION_OFF_SCALER_MEAN: usize = 32;

/// Header byte offset of the `scaler_scale` [`crate::Section`].
pub const SECTION_OFF_SCALER_SCALE: usize = 40;

/// Header byte offset of the `layer_table` [`crate::Section`].
pub const SECTION_OFF_LAYER_TABLE: usize = 48;

/// Header byte offset of the `feature_bounds` [`crate::Section`].
pub const SECTION_OFF_FEATURE_BOUNDS: usize = 56;

/// Header byte offset of the `metadata` [`crate::Section`].
pub const SECTION_OFF_METADATA: usize = 64;

/// Header byte offset of the `output_specs` [`crate::Section`] (v3+).
pub const SECTION_OFF_OUTPUT_SPECS: usize = 72;

/// Header byte offset of the `discrete_sets` [`crate::Section`] (v3+).
pub const SECTION_OFF_DISCRETE_SETS: usize = 80;

/// Header byte offset of the `sparse_overrides` [`crate::Section`] (v3+).
pub const SECTION_OFF_SPARSE_OVERRIDES: usize = 88;

/// Header byte offset of `decompressed_payload_len: u32` (v3.1+).
/// Meaningful when `flags` bit 0 is set; zero otherwise.
pub const OFF_DECOMPRESSED_PAYLOAD_LEN: usize = 96;

/// Header byte offset of the `feature_order` [`crate::Section`] (v3.1+).
/// Empty (len=0) when the bake's inputs are in caller-natural order.
pub const SECTION_OFF_FEATURE_ORDER: usize = 100;

/// Header byte offset of the `output_order` [`crate::Section`] (v3.1+).
/// Empty (len=0) when the bake's outputs are in caller-natural order.
pub const SECTION_OFF_OUTPUT_ORDER: usize = 108;

// Header bytes 116..124 are reserved for a future header section.
// Previously housed `multi_codec_schema` (v3.2) which was reverted
// pre-publish — see CHANGELOG. Header bytes 124..128 are further
// reserved (`Header.reserved: [u32; 3]` spans 116..128).

// ── Header flags bits (offset 6..8) ──────────────────────────────

/// Mask for `flags` bit 0: when set, payload bytes [128..end] are
/// compressed. The algorithm is encoded in `FLAGS_COMPRESSION_ALGO_MASK`.
pub const FLAG_COMPRESSED: u16 = 0x0001;

/// Mask for `flags` bits 1..3: 3-bit compression algorithm field.
/// Shift right by 1 after masking to get the algo enum.
pub const FLAGS_COMPRESSION_ALGO_MASK: u16 = 0x000E;

/// Compression algorithm: no compression.
pub const COMPRESSION_ALGO_NONE: u8 = 0;

/// Compression algorithm: LZ4 block format (`lz4_flex::block`).
pub const COMPRESSION_ALGO_LZ4: u8 = 1;
