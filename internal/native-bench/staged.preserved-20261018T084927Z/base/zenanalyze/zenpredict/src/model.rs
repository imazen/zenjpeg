//! ZNPR v3 binary model format — parser and typed views.
//!
//! ## Layout
//!
//! Little-endian throughout. Wrap `include_bytes!` output in
//! `#[repr(C, align(16))]` so the byte slice's start address is
//! 16-aligned; every section's offset is a multiple of 8, so all
//! typed slices land at native alignment without copies.
//!
//! ```text
//! 0..128                  Header (#[repr(C)], pod, fixed size)
//! 128..(128 + n_layers*48) LayerEntry[n_layers] (#[repr(C)], pod)
//! (rest of file)          Aligned data blobs at offsets named by
//!                         the Sections in Header / LayerEntry:
//!                           - scaler_mean, scaler_scale (f32)
//!                           - per-layer weights (f32 / f16 / i8)
//!                           - per-layer scales (f32, only for I8 layers)
//!                           - per-layer biases (f32)
//!                           - feature_bounds (FeatureBound = (f32, f32))
//!                           - metadata blob (TLV; see [`crate::metadata`])
//!                           - output_specs (OutputSpec[n_outputs])
//!                           - discrete_sets (f32 pool)
//!                           - sparse_overrides (SparseOverride[N])
//! ```
//!
//! ### Header (128 bytes)
//!
//! ```text
//! 0..4    magic = b"ZNPR"
//! 4..6    version: u16 = 3
//! 6..8    flags: u16  (reserved, 0)
//! 8..12   n_inputs: u32
//! 12..16  n_outputs: u32
//! 16..20  n_layers: u32
//! 20..24  _pad0: u32  (alignment for u64)
//! 24..32  schema_hash: u64
//! 32..40  scaler_mean: Section
//! 40..48  scaler_scale: Section
//! 48..56  layer_table: Section
//! 56..64  feature_bounds: Section   (len=0 when absent)
//! 64..72  metadata: Section         (len=0 when absent)
//! 72..80  output_specs: Section     (len=0 when absent; n_outputs * 32 bytes)
//! 80..88  discrete_sets: Section    (len=0 when absent; pool of f32)
//! 88..96  sparse_overrides: Section (len=0 when absent; n_overrides * 8 bytes)
//! 96..128 reserved: [u32; 8]
//! ```
//!
//! ### Older formats
//!
//! v1 and v2 bakes are not loaded by this crate — they fail with
//! [`crate::PredictError::UnsupportedVersion`]. Migrate them via
//! `zentrain/tools/migrate_znpr_v2_to_v3.py`, which rewrites the
//! header to v3 (the v3 layout is byte-identical to v2 through the
//! first 72 bytes; the three new sections live in space that v2
//! reserved). Layer payloads do not change. See [`crate::output_spec`]
//! for the wire shape of the new POD types.
//!
//! ### LayerEntry (48 bytes)
//!
//! ```text
//! 0..4    in_dim: u32
//! 4..8    out_dim: u32
//! 8..9    activation: u8       (0=Identity, 1=Relu, 2=LeakyRelu)
//! 9..10   weight_dtype: u8     (0=F32, 1=F16, 2=I8)
//! 10..12  flags: u16           (reserved)
//! 12..20  weights: Section
//! 20..28  scales: Section      (len=0 unless weight_dtype == I8)
//! 28..36  biases: Section
//! 36..48  reserved: [u32; 3]
//! ```
//!
//! ### Section
//!
//! ```text
//! 0..4   offset: u32   absolute byte offset into the file
//! 4..8   len: u32      byte length
//! ```
//!
//! `len = 0` means the section is absent. Both `feature_bounds` and
//! `metadata` are optional in this sense.
//!
//! ## Weight storage
//!
//! - **F32** — `weights.len / 4` f32 values, row-major (input-major).
//!   `W[i * out_dim + o]` is the contribution from input `i` to
//!   output `o`.
//! - **F16** — `weights.len / 2` raw IEEE-754 binary16 bit patterns
//!   stored as `u16`. Converted to f32 per element in the inner loop
//!   (see [`crate::inference::f16_bits_to_f32`]).
//! - **I8** — `weights.len` signed 8-bit values, row-major. Per-output
//!   f32 scales in the `scales` Section. Dequantized weight is
//!   `W[i, o] = weights[i * out_dim + o] as f32 * scales[o]`.
//!
//! ## I8 quantization scheme (bake-side, not enforced at runtime)
//!
//! ```text
//! scales[o]    = max_i |W_f32[i, o]| / 127.0   (or 1.0 if the column is zero)
//! i8_weights[i, o] = round(W_f32[i, o] / scales[o]).clamp(-128, 127)
//! ```

use bytemuck::{Pod, Zeroable, pod_read_unaligned};

use crate::error::PredictError;
use crate::feature_transform::{
    FeatureTransform, parse_feature_transform_params, parse_feature_transforms,
};
use crate::metadata::Metadata;
use crate::output_spec::{OutputSpec, SparseOverride};

pub const FORMAT_VERSION: u16 = 3;
pub const LEAKY_RELU_ALPHA: f32 = 0.01;
const MAGIC: [u8; 4] = *b"ZNPR";
use crate::wire::{HEADER_SIZE, LAYER_ENTRY_SIZE};

/// `(offset, len)` pair addressing a byte range in the file. `len = 0`
/// is reserved for "section absent".
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
pub struct Section {
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

impl Section {
    pub const fn empty() -> Self {
        Self { offset: 0, len: 0 }
    }

    /// Construct from raw `(offset, len)` — the inverse of
    /// [`Self::offset`] / [`Self::len_bytes`]. Used by the composer in the
    /// sibling `zenpredict-bake` crate; runtime consumers shouldn't
    /// need to mint Sections directly.
    pub const fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }

    /// Absolute byte offset into the bake file.
    pub const fn offset(self) -> u32 {
        self.offset
    }

    /// Section byte length. Zero means "section absent."
    pub const fn len_bytes(self) -> u32 {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Slice into `bytes`, validating bounds. Returns the empty
    /// slice when `len == 0` (section absent), without consulting
    /// `offset`.
    pub fn slice<'a>(self, what: &'static str, bytes: &'a [u8]) -> Result<&'a [u8], PredictError> {
        if self.len == 0 {
            return Ok(&[]);
        }
        let start = self.offset as usize;
        let end = start
            .checked_add(self.len as usize)
            .ok_or(PredictError::SectionOutOfRange {
                what,
                offset: self.offset,
                len: self.len,
                file_len: bytes.len(),
            })?;
        if end > bytes.len() {
            return Err(PredictError::SectionOutOfRange {
                what,
                offset: self.offset,
                len: self.len,
                file_len: bytes.len(),
            });
        }
        Ok(&bytes[start..end])
    }
}

/// Fixed `#[repr(C)]` file header. Loader does one
/// `bytemuck::pod_read_unaligned::<Header>(&bytes[..128])` to copy
/// it out — never borrows the in-file bytes for the header itself,
/// so the `Header` value is owned and the loader can validate
/// without holding a `&Header` into possibly-misaligned input.
///
/// The 0.2.0 wire-format extension uses the v3 reserved area at
/// offset 96..128 for three new fields without changing the
/// 128-byte total size:
/// - `decompressed_payload_len` (u32 @ 96): set when `flags` bit 0
///   indicates a compressed payload.
/// - `feature_order` (Section @ 100..108): optional input-feature
///   permutation index. Auto-sized u8/u16/u32 (inferred from
///   `len / n_inputs`).
/// - `output_order` (Section @ 108..116): optional output
///   permutation index. Auto-sized analogous to feature_order.
///
/// The remaining 12 bytes stay reserved for future use.
#[repr(C)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Header {
    pub(crate) magic: [u8; 4],
    pub(crate) version: u16,
    /// `flags & 1` set iff the payload (bytes 128..end) is compressed.
    /// `(flags >> 1) & 7` is the compression-algo nibble:
    /// `0 = None`, `1 = LZ4 (lz4_flex block format)`. Bits 4..15 are
    /// reserved and must be zero.
    pub(crate) flags: u16,
    pub(crate) n_inputs: u32,
    pub(crate) n_outputs: u32,
    pub(crate) n_layers: u32,
    pub(crate) _pad0: u32,
    pub(crate) schema_hash: u64,
    pub(crate) scaler_mean: Section,
    pub(crate) scaler_scale: Section,
    pub(crate) layer_table: Section,
    pub(crate) feature_bounds: Section,
    pub(crate) metadata: Section,
    pub(crate) output_specs: Section,
    pub(crate) discrete_sets: Section,
    pub(crate) sparse_overrides: Section,
    /// When `flags` bit 0 is set, the byte-length of the
    /// post-decompression payload (bytes 128..end). Zero for
    /// uncompressed bakes.
    pub(crate) decompressed_payload_len: u32,
    /// Input-feature permutation. When `len == 0`: bake's inputs are
    /// in caller-natural order. When non-empty: contains `n_inputs`
    /// indices addressing the caller-natural feature positions. The
    /// index width is inferred from `len`:
    /// - `len == n_inputs` → u8 (n_inputs ≤ 255)
    /// - `len == 2 * n_inputs` → u16 (n_inputs ≤ 65535)
    /// - `len == 4 * n_inputs` → u32
    ///
    /// At load time, the inverse permutation is applied in-place to
    /// scaler_mean, scaler_scale, feature_bounds, and layer[0]
    /// weight rows so that the in-memory bake is in caller order.
    pub(crate) feature_order: Section,
    /// Output permutation. Symmetric to `feature_order`. Affects
    /// layer[last] weight cols + biases, output_specs,
    /// sparse_overrides indices, and metadata-side output-indexed
    /// arrays (cell_rescue_hints, output_bounds).
    pub(crate) output_order: Section,
    /// Reserved for future header sections. Currently 12 bytes
    /// (3 × u32) — header total stays at 128 bytes regardless of
    /// reservation usage. Zero-initialized on write; zero-tolerated
    /// on read (older bakes that happened to populate the legacy
    /// `multi_codec_schema` slot decode as all-zero `reserved`).
    pub(crate) reserved: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<Header>() == HEADER_SIZE);
const _: () = assert!(core::mem::size_of::<Section>() == 8);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LayerEntry {
    pub in_dim: u32,
    pub out_dim: u32,
    pub activation: u8,
    pub weight_dtype: u8,
    pub flags: u16,
    pub weights: Section,
    pub scales: Section,
    pub biases: Section,
    pub reserved: [u32; 3],
}

const _: () = assert!(core::mem::size_of::<LayerEntry>() == LAYER_ENTRY_SIZE);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Activation {
    Identity = 0,
    Relu = 1,
    /// `LeakyRelu(alpha = 0.01)` — fixed slope; matches PyTorch's
    /// `nn.LeakyReLU` default and what `train_hybrid.py
    /// --activation leakyrelu` emits.
    LeakyRelu = 2,
}

impl Activation {
    pub(crate) fn from_byte(b: u8) -> Result<Self, PredictError> {
        match b {
            0 => Ok(Self::Identity),
            1 => Ok(Self::Relu),
            2 => Ok(Self::LeakyRelu),
            other => Err(PredictError::UnknownActivation { byte: other }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightDtype {
    F32 = 0,
    F16 = 1,
    I8 = 2,
}

impl WeightDtype {
    pub(crate) fn from_byte(b: u8) -> Result<Self, PredictError> {
        match b {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::I8),
            other => Err(PredictError::UnknownWeightDtype { byte: other }),
        }
    }

    pub fn storage_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }
}

/// Layered weight + bias view, borrowed from the file bytes.
#[derive(Debug)]
pub struct LayerView<'a> {
    pub in_dim: usize,
    pub out_dim: usize,
    pub activation: Activation,
    pub weights: WeightStorage<'a>,
    pub biases: &'a [f32],
}

#[derive(Debug)]
pub enum WeightStorage<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    I8 {
        weights: &'a [i8],
        scales: &'a [f32],
    },
}

/// Per-layer offsets cached at parse time. Materialized into a
/// [`LayerView`] on demand by [`Model::layer`] / [`Model::layers`].
///
/// All field validation (dimension checks, alignment, section
/// bounds) happens once at parse time. Per-access materialization
/// is a handful of `bytemuck::cast_slice` calls; in release builds
/// the alignment check is one comparison and one transmute —
/// ~10 ns total, no panic possible since alignment was validated.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerOffsets {
    pub(crate) in_dim: u32,
    pub(crate) out_dim: u32,
    pub(crate) activation: Activation,
    pub(crate) weight_dtype: WeightDtype,
    pub(crate) weights: Section,
    pub(crate) scales: Section,
    pub(crate) biases: Section,
}

/// Parsed view over a ZNPR v3 model. The `Model` **owns** the bake
/// bytes (always; whether the input was compressed or not).
/// `Predictor` borrows `&Model` for inference — `Model` is `Sync`,
/// so a single `static OnceLock<Model>` can back per-thread
/// `Predictor` instances without a `Mutex`.
///
/// The bake bytes live in a heap-allocated `Box<[u8]>`. Views into
/// the bake (layer weights, scaler, metadata, etc.) borrow from
/// `&self.bytes` and are materialized on demand — no self-referential
/// storage, no `'a` parameter on `Model` itself. Per-access cost is
/// a handful of `bytemuck::cast_slice` operations (~10–50 ns total
/// for a layer view, negligible vs the µs-scale matmul).
///
/// Load-time mutations (decompression, feature_order / output_order
/// inverse permutations) happen once inside [`Self::from_bytes`] and
/// produce a canonical-order `Model` — the predict hot path never
/// sees the on-disk permutation or compression.
#[derive(Debug)]
pub struct Model {
    bytes: alloc::boxed::Box<[u8]>,
    header: Header,
    /// Per-layer offsets + dims, cached at parse time. The
    /// [`LayerView`] materialized via [`Self::layer`] is constructed
    /// from these offsets on demand.
    layer_offsets: alloc::vec::Vec<LayerOffsets>,
    /// Owned cache of parsed `zentrain.feature_transforms`. `None`
    /// when the bake omitted the key (consumer treats every feature
    /// as `Identity`); `Some(_)` with `len == n_inputs` otherwise.
    feature_transforms: Option<alloc::vec::Vec<FeatureTransform>>,
    /// Owned cache of parsed `zentrain.feature_transform_params`.
    /// `None` when the bake omitted the key; `Some(_)` with one
    /// per-feature `Vec<f32>` of variant-specific params otherwise.
    /// Inner vec is empty for features whose transform variant
    /// doesn't consume params (e.g., `Identity`, `Log1p`,
    /// `SignedLog1p`). Required for V0_20 parameterized variants.
    feature_transform_params: Option<alloc::vec::Vec<alloc::vec::Vec<f32>>>,
}

impl Model {
    /// Parse a v3 model from raw bytes. The returned `Model` owns
    /// its copy of the bake (heap-allocated `Box<[u8]>`); `input`
    /// is consumed only at construction time.
    ///
    /// On a compressed bake (`flags` bit 0 set), the loader
    /// decompresses inline into the owned buffer. On `feature_order`
    /// / `output_order` metadata, the loader applies the inverse
    /// permutation in-place so the in-memory bake is in caller-natural
    /// order — the predict hot path is unaffected.
    ///
    /// Reserved fields in [`Header`] and [`LayerEntry`] are ignored
    /// on read, so future format extensions can populate them
    /// without invalidating the schema. Bakers MUST zero them.
    pub fn from_bytes(input: &[u8]) -> Result<Self, PredictError> {
        Self::from_bytes_inner(input, None)
    }

    /// Parse and verify the schema_hash matches `expected` BEFORE any
    /// section parsing, so a wrong-bake input bails out cheaply
    /// (`O(1)`) rather than walking the layer table first.
    ///
    /// Convenience for codecs that compile-in a schema hash and want
    /// to fail loudly at load when a stale bake gets shipped.
    pub fn from_bytes_with_schema(input: &[u8], expected: u64) -> Result<Self, PredictError> {
        Self::from_bytes_inner(input, Some(expected))
    }

    fn from_bytes_inner(input: &[u8], expected_schema: Option<u64>) -> Result<Self, PredictError> {
        // Resource limits — fail early on adversarial input before any
        // section parsing or scratch allocation. See `crate::limits`.
        if input.len() > crate::limits::MAX_BAKE_BYTES {
            return Err(PredictError::Truncated {
                offset: 0,
                want: crate::limits::MAX_BAKE_BYTES,
                have: input.len(),
            });
        }
        if input.len() < HEADER_SIZE {
            return Err(PredictError::Truncated {
                offset: 0,
                want: HEADER_SIZE,
                have: input.len(),
            });
        }
        let header: Header = pod_read_unaligned(&input[..HEADER_SIZE]);

        if header.magic != MAGIC {
            return Err(PredictError::BadMagic {
                found: header.magic,
            });
        }
        if header.version != FORMAT_VERSION {
            return Err(PredictError::UnsupportedVersion {
                version: header.version,
                expected: FORMAT_VERSION,
            });
        }
        // Schema hash is checked BEFORE any layer-table allocation
        // so adversarial bakes with mismatched schema + huge n_layers
        // don't allocate before failing.
        if let Some(want) = expected_schema
            && header.schema_hash != want
        {
            return Err(PredictError::SchemaHashMismatch {
                expected: want,
                got: header.schema_hash,
            });
        }

        let n_inputs = header.n_inputs as usize;
        let n_outputs = header.n_outputs as usize;
        let n_layers = header.n_layers as usize;
        if n_inputs == 0 {
            return Err(PredictError::ZeroDimension { what: "n_inputs" });
        }
        if n_outputs == 0 {
            return Err(PredictError::ZeroDimension { what: "n_outputs" });
        }
        if n_layers == 0 {
            return Err(PredictError::ZeroDimension { what: "n_layers" });
        }
        if n_inputs > crate::limits::MAX_DIM {
            return Err(PredictError::DimensionOverflow { what: "n_inputs" });
        }
        if n_outputs > crate::limits::MAX_DIM {
            return Err(PredictError::DimensionOverflow { what: "n_outputs" });
        }
        if n_layers > crate::limits::MAX_LAYERS {
            return Err(PredictError::DimensionOverflow { what: "n_layers" });
        }

        // ── Stage 1: materialize the owned bake into a Box<[u8]>. ──
        //
        // For uncompressed bakes this is one memcpy of `input` into
        // owned storage (~5 µs for a 38 KB picker). For compressed
        // bakes, we decompress the payload directly into the owned
        // storage at offset 128. Either way, after this block,
        // `bytes` is a heap-owned `Box<[u8]>` we can mutate freely
        // for load-time permutations.
        let compressed = (header.flags & crate::wire::FLAG_COMPRESSED) != 0;
        let algo = ((header.flags & crate::wire::FLAGS_COMPRESSION_ALGO_MASK) >> 1) as u8;
        let bytes: alloc::boxed::Box<[u8]> = if compressed {
            let payload_len = header.decompressed_payload_len as usize;
            if payload_len == 0 {
                return Err(PredictError::SectionOutOfRange {
                    what: "decompressed_payload_len (zero with compressed flag set)",
                    offset: crate::wire::OFF_DECOMPRESSED_PAYLOAD_LEN as u32,
                    len: 4,
                    file_len: input.len(),
                });
            }
            let total_len =
                HEADER_SIZE
                    .checked_add(payload_len)
                    .ok_or(PredictError::DimensionOverflow {
                        what: "HEADER_SIZE + decompressed_payload_len",
                    })?;
            if total_len > crate::limits::MAX_BAKE_BYTES {
                return Err(PredictError::Truncated {
                    offset: 0,
                    want: crate::limits::MAX_BAKE_BYTES,
                    have: total_len,
                });
            }
            let mut owned = alloc::vec![0u8; total_len].into_boxed_slice();
            owned[..HEADER_SIZE].copy_from_slice(&input[..HEADER_SIZE]);
            // Decompress payload into owned[HEADER_SIZE..]. Only LZ4
            // is supported right now.
            match algo {
                crate::wire::COMPRESSION_ALGO_LZ4 => {
                    decompress_lz4(&input[HEADER_SIZE..], &mut owned[HEADER_SIZE..])?;
                }
                crate::wire::COMPRESSION_ALGO_NONE => {
                    return Err(PredictError::SectionOutOfRange {
                        what: "flags.compressed=1 but algo=None",
                        offset: 6,
                        len: 2,
                        file_len: input.len(),
                    });
                }
                _ => {
                    return Err(PredictError::SectionOutOfRange {
                        what: "unknown compression algo",
                        offset: 6,
                        len: 2,
                        file_len: input.len(),
                    });
                }
            }
            owned
        } else {
            // Uncompressed: copy verbatim. We always own the bake,
            // so load-time permutations can mutate freely.
            input.to_vec().into_boxed_slice()
        };

        // ── Stage 2: parse + validate layer table from owned bytes. ──
        let layer_bytes = header.layer_table.slice("layer_table", &bytes)?;
        let expected_layer_bytes =
            n_layers
                .checked_mul(LAYER_ENTRY_SIZE)
                .ok_or(PredictError::DimensionOverflow {
                    what: "n_layers * sizeof(LayerEntry)",
                })?;
        if layer_bytes.len() != expected_layer_bytes {
            return Err(PredictError::SectionOutOfRange {
                what: "layer_table",
                offset: header.layer_table.offset,
                len: header.layer_table.len,
                file_len: bytes.len(),
            });
        }
        let layer_entries: &[LayerEntry] =
            bytemuck::try_cast_slice(layer_bytes).map_err(|_| PredictError::SectionMisaligned {
                what: "layer_table",
                offset: header.layer_table.offset,
                required_align: core::mem::align_of::<LayerEntry>(),
            })?;

        let mut layer_offsets = alloc::vec::Vec::with_capacity(n_layers);
        let mut prev_out = n_inputs;
        for (layer_idx, entry) in layer_entries.iter().enumerate() {
            let in_dim = entry.in_dim as usize;
            let out_dim = entry.out_dim as usize;
            if in_dim == 0 {
                return Err(PredictError::ZeroDimension {
                    what: "layer.in_dim",
                });
            }
            if out_dim == 0 {
                return Err(PredictError::ZeroDimension {
                    what: "layer.out_dim",
                });
            }
            if in_dim != prev_out {
                return Err(PredictError::LayerDimMismatch {
                    layer: layer_idx,
                    expected_in: prev_out,
                    got_in: in_dim,
                });
            }
            let activation = Activation::from_byte(entry.activation)?;
            let weight_dtype = WeightDtype::from_byte(entry.weight_dtype)?;
            let n_weights = in_dim
                .checked_mul(out_dim)
                .ok_or(PredictError::DimensionOverflow {
                    what: "layer.in_dim * layer.out_dim",
                })?;
            // Validate weight section bounds + alignment now (one-shot;
            // materialize_layer below assumes pre-validated).
            match weight_dtype {
                WeightDtype::F32 => {
                    let _ =
                        cast_f32_section("layer.weights[f32]", entry.weights, &bytes, n_weights)?;
                }
                WeightDtype::F16 => {
                    let _ =
                        cast_u16_section("layer.weights[f16]", entry.weights, &bytes, n_weights)?;
                }
                WeightDtype::I8 => {
                    let _ = cast_i8_section("layer.weights[i8]", entry.weights, &bytes, n_weights)?;
                    let _ = cast_f32_section("layer.scales", entry.scales, &bytes, out_dim)?;
                }
            }
            let has_scales = matches!(weight_dtype, WeightDtype::I8);
            if !has_scales && !entry.scales.is_empty() {
                return Err(PredictError::SectionOutOfRange {
                    what: "layer.scales (must be empty for non-I8 layer)",
                    offset: entry.scales.offset,
                    len: entry.scales.len,
                    file_len: bytes.len(),
                });
            }
            let _ = cast_f32_section("layer.biases", entry.biases, &bytes, out_dim)?;

            layer_offsets.push(LayerOffsets {
                in_dim: entry.in_dim,
                out_dim: entry.out_dim,
                activation,
                weight_dtype,
                weights: entry.weights,
                scales: entry.scales,
                biases: entry.biases,
            });
            prev_out = out_dim;
        }

        if prev_out != n_outputs {
            return Err(PredictError::OutputDimMismatch {
                expected: n_outputs,
                got: prev_out,
            });
        }

        // ── Stage 3: validate optional sections (without caching). ──
        // We re-validate on each accessor call too, but check bounds
        // here so corrupted bakes fail at parse time.
        validate_scaler_section("scaler_mean", header.scaler_mean, &bytes, n_inputs)?;
        validate_scaler_section("scaler_scale", header.scaler_scale, &bytes, n_inputs)?;
        validate_feature_bounds(header.feature_bounds, &bytes, n_inputs)?;
        validate_output_specs(header.output_specs, &bytes, n_outputs)?;
        validate_discrete_sets(header.discrete_sets, &bytes)?;
        validate_sparse_overrides(header.sparse_overrides, &bytes, n_outputs)?;

        // ── Stage 4: load-time inverse permutations. ──
        //
        // The bake's on-disk layout MAY be in a compression-optimal
        // permuted order (feature_order / output_order metadata).
        // The runtime expects everything in caller-natural order, so
        // we apply the INVERSE permutation in-place to all affected
        // sections. After this stage, `bytes` is in canonical order
        // and the predict hot path never sees the permutation.
        let mut bytes = bytes;
        if !header.feature_order.is_empty() {
            apply_feature_order_inverse(&mut bytes, &header, &layer_offsets, n_inputs)?;
        }
        if !header.output_order.is_empty() {
            apply_output_order_inverse(&mut bytes, &header, &layer_offsets, n_outputs)?;
        }

        // ── Stage 5: parse feature_transforms + transform_params metadata. ──
        let metadata_bytes = header.metadata.slice("metadata", &bytes)?;
        let metadata = Metadata::parse(metadata_bytes)?;
        let feature_transforms = parse_feature_transforms(&metadata)?;
        let feature_transform_params = parse_feature_transform_params(&metadata)?;

        // ── Stage 6: cross-validate the feature pipeline. ──
        //
        // For scalar-only pipelines, `transforms.len() == n_inputs`
        // (the first-layer in_dim). For pipelines that contain an
        // expander variant (today: only Sinusoidal), `transforms.len()`
        // is the **raw** input feature count, and the sum of
        // per-feature output arities equals the first-layer in_dim.
        // Both shapes must agree on the parallel transforms/params
        // arrays.
        if let Some(ref ts) = feature_transforms {
            if let Some(ref ps) = feature_transform_params
                && ts.len() != ps.len()
            {
                return Err(PredictError::FeatureTransformsLenMismatch {
                    expected: ts.len(),
                    got: ps.len(),
                });
            }
            // Compute expanded dim: sum of per-feature arities.
            // Missing params metadata ⇒ every feature uses `&[]`.
            let expanded: usize = match &feature_transform_params {
                Some(ps) => (0..ts.len()).map(|i| ts[i].output_arity(&ps[i])).sum(),
                None => ts.iter().map(|t| t.output_arity(&[])).sum(),
            };
            if expanded != n_inputs {
                return Err(PredictError::FeatureTransformsLenMismatch {
                    expected: n_inputs,
                    got: expanded,
                });
            }
        }

        Ok(Self {
            bytes,
            header,
            layer_offsets,
            feature_transforms,
            feature_transform_params,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn version(&self) -> u16 {
        self.header.version
    }

    pub fn flags(&self) -> u16 {
        self.header.flags
    }

    pub fn n_inputs(&self) -> usize {
        self.header.n_inputs as usize
    }

    pub fn n_outputs(&self) -> usize {
        self.header.n_outputs as usize
    }

    /// Number of dense layers (linear + activation) in the network.
    /// Always equals `self.layers().len()`; exposed as a separate
    /// accessor so consumers can size buffers / log diagnostics
    /// without holding a `&[LayerView]`.
    pub fn n_layers(&self) -> usize {
        self.header.n_layers as usize
    }

    pub fn schema_hash(&self) -> u64 {
        self.header.schema_hash
    }

    pub fn scaler_mean(&self) -> &[f32] {
        let n = self.n_inputs();
        // Parse-time validated, so unwrap is safe.
        cast_f32_section("scaler_mean", self.header.scaler_mean, &self.bytes, n)
            .expect("scaler_mean validated at parse time")
    }

    pub fn scaler_scale(&self) -> &[f32] {
        let n = self.n_inputs();
        cast_f32_section("scaler_scale", self.header.scaler_scale, &self.bytes, n)
            .expect("scaler_scale validated at parse time")
    }

    /// Iterator of materialized [`LayerView`]s, one per layer.
    /// Each view is constructed on the fly from cached offsets +
    /// `self.bytes`; the cost is negligible vs the matmul.
    pub fn layers(&self) -> LayerIter<'_> {
        LayerIter {
            model: self,
            idx: 0,
            end: self.layer_offsets.len(),
        }
    }

    /// Materialize layer `idx`. Panics if `idx >= n_layers`.
    pub fn layer(&self, idx: usize) -> LayerView<'_> {
        self.materialize_layer(idx)
    }

    fn materialize_layer(&self, idx: usize) -> LayerView<'_> {
        let off = &self.layer_offsets[idx];
        let in_dim = off.in_dim as usize;
        let out_dim = off.out_dim as usize;
        let n_weights = in_dim * out_dim;
        let weights = match off.weight_dtype {
            WeightDtype::F32 => WeightStorage::F32(
                cast_f32_section("layer.weights[f32]", off.weights, &self.bytes, n_weights)
                    .expect("layer weights validated at parse time"),
            ),
            WeightDtype::F16 => WeightStorage::F16(
                cast_u16_section("layer.weights[f16]", off.weights, &self.bytes, n_weights)
                    .expect("layer weights validated at parse time"),
            ),
            WeightDtype::I8 => {
                let w = cast_i8_section("layer.weights[i8]", off.weights, &self.bytes, n_weights)
                    .expect("layer weights validated at parse time");
                let s = cast_f32_section("layer.scales", off.scales, &self.bytes, out_dim)
                    .expect("layer scales validated at parse time");
                WeightStorage::I8 {
                    weights: w,
                    scales: s,
                }
            }
        };
        let biases = cast_f32_section("layer.biases", off.biases, &self.bytes, out_dim)
            .expect("layer biases validated at parse time");
        LayerView {
            in_dim,
            out_dim,
            activation: off.activation,
            weights,
            biases,
        }
    }

    pub fn feature_bounds(&self) -> &[crate::bounds::FeatureBound] {
        if self.header.feature_bounds.is_empty() {
            return &[];
        }
        let n_bound_f32s = self.n_inputs() * 2;
        let raw = cast_f32_section(
            "feature_bounds",
            self.header.feature_bounds,
            &self.bytes,
            n_bound_f32s,
        )
        .expect("feature_bounds validated at parse time");
        let bound_bytes = bytemuck::cast_slice::<f32, u8>(raw);
        bytemuck::try_cast_slice::<u8, crate::bounds::FeatureBound>(bound_bytes)
            .expect("feature_bounds alignment validated at parse time")
    }

    /// Reparse the metadata blob on demand. Cheap (just header.metadata
    /// section slice + Metadata::parse on the byte view).
    pub fn metadata(&self) -> Metadata<'_> {
        let raw = self
            .header
            .metadata
            .slice("metadata", &self.bytes)
            .expect("metadata validated at parse time");
        Metadata::parse(raw).expect("metadata blob parse validated at parse time")
    }

    /// The model's feature-column NAMES in input order, parsed from the
    /// [`FEATURE_COLUMNS`](crate::keys::FEATURE_COLUMNS) metadata (newline- or
    /// comma-separated, `feat_` prefix kept as authored). Empty on older bakes
    /// that didn't record them. These are the names a codec resolves against its
    /// own `zenanalyze@X` and the `names` of a `zenanalyze_api::Request`.
    pub fn feature_columns(&self) -> impl Iterator<Item = &str> {
        self.metadata()
            .get_utf8(crate::keys::FEATURE_COLUMNS)
            .unwrap_or("")
            .split(['\n', ','])
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }

    /// The `zenanalyze` version the model's features were extracted with (the
    /// [`ANALYZER_VERSION`](crate::keys::ANALYZER_VERSION) metadata, e.g.
    /// `"0.2.7"`), or `None` on bakes predating the key. Its `major.minor` plus
    /// [`feature_defs_version`](Self::feature_defs_version) is the reuse key a
    /// codec checks against an offered feature result.
    pub fn analyzer_version(&self) -> Option<&str> {
        self.metadata().get_utf8(crate::keys::ANALYZER_VERSION).ok()
    }

    /// The feature numeric-definition version the model was trained against (the
    /// [`FEATURE_DEFS_VERSION`](crate::keys::FEATURE_DEFS_VERSION) metadata), or
    /// `None` on bakes predating the key (or a wrong-width entry).
    ///
    /// Decoded as a 4-byte little-endian `u32` per the metadata module's scalar
    /// shape contract — endian-explicit so it round-trips identically on i686 /
    /// big-endian, unlike a native-endian `get_pod`.
    pub fn feature_defs_version(&self) -> Option<u32> {
        let bytes = self
            .metadata()
            .get_numeric(crate::keys::FEATURE_DEFS_VERSION)
            .ok()?;
        let arr: [u8; 4] = bytes.try_into().ok()?;
        Some(u32::from_le_bytes(arr))
    }

    /// The value-affecting analysis-config digest the model was trained under
    /// (the [`FEATURE_CONFIG_HASH`](crate::keys::FEATURE_CONFIG_HASH) metadata,
    /// `zenanalyze::AnalysisQuery::config_hash()` at bake time), or `None` on
    /// bakes predating the key (treat as `0` = canonical default). The third part
    /// of the `zenanalyze-api` reuse key.
    ///
    /// Decoded as an 8-byte little-endian `u64`, endian-explicit so it round-trips
    /// identically on i686 / big-endian.
    pub fn feature_config_hash(&self) -> Option<u64> {
        let bytes = self
            .metadata()
            .get_numeric(crate::keys::FEATURE_CONFIG_HASH)
            .ok()?;
        let arr: [u8; 8] = bytes.try_into().ok()?;
        Some(u64::from_le_bytes(arr))
    }

    /// The feature-gated knob-veto safety rules embedded in the bake (the
    /// [`KNOB_VETOES_KEY`](crate::KNOB_VETOES_KEY) metadata), or an empty
    /// list on a bake that carried no vetoes / predates the feature.
    ///
    /// A picker enforces these as a pre-argmin masking pass via
    /// [`apply_knob_vetoes`](crate::apply_knob_vetoes) so the deployed pick
    /// honors the same vetoes the bake-time safety gate evaluated. The
    /// returned [`KnobVeto`](crate::KnobVeto)s borrow the model's bytes.
    pub fn knob_vetoes(&self) -> Result<alloc::vec::Vec<crate::KnobVeto<'_>>, PredictError> {
        // `metadata()` borrows `self.bytes` for `&self`'s lifetime, so the
        // entry value (and the KnobVeto cell slices into it) outlive the
        // temporary `Metadata` dropped at the end of this call.
        crate::knob_vetoes_from_metadata(&self.metadata())
    }

    /// The size-discriminated unachievable-zone table embedded in the bake
    /// (the [`UNACHIEVABLE_ZONES_KEY`](crate::UNACHIEVABLE_ZONES_KEY)
    /// metadata), or an empty table on a bake that declared none / predates
    /// the feature.
    ///
    /// A picker consults this **before** its argmin via
    /// [`UnachievableZones::resolve`](crate::UnachievableZones::resolve):
    /// when `target_zq` exceeds the achievable ceiling for the image's size
    /// class the codec encodes the declared fallback knobset directly instead
    /// of attempting an unreachable pick.
    pub fn unachievable_zones(&self) -> Result<crate::UnachievableZones, PredictError> {
        crate::unachievable_zones_from_metadata(&self.metadata())
    }

    /// Per-output [`OutputSpec`] table.
    pub fn output_specs(&self) -> &[OutputSpec] {
        if self.header.output_specs.is_empty() {
            return &[];
        }
        let raw = self
            .header
            .output_specs
            .slice("output_specs", &self.bytes)
            .expect("output_specs validated at parse time");
        bytemuck::try_cast_slice::<u8, OutputSpec>(raw)
            .expect("output_specs alignment validated at parse time")
    }

    pub fn has_output_specs(&self) -> bool {
        !self.header.output_specs.is_empty()
    }

    pub fn discrete_sets(&self) -> &[f32] {
        if self.header.discrete_sets.is_empty() {
            return &[];
        }
        let raw = self
            .header
            .discrete_sets
            .slice("discrete_sets", &self.bytes)
            .expect("discrete_sets validated at parse time");
        bytemuck::try_cast_slice::<u8, f32>(raw)
            .expect("discrete_sets alignment validated at parse time")
    }

    pub fn sparse_overrides(&self) -> &[SparseOverride] {
        if self.header.sparse_overrides.is_empty() {
            return &[];
        }
        let raw = self
            .header
            .sparse_overrides
            .slice("sparse_overrides", &self.bytes)
            .expect("sparse_overrides validated at parse time");
        bytemuck::try_cast_slice::<u8, SparseOverride>(raw)
            .expect("sparse_overrides alignment validated at parse time")
    }

    #[cfg(feature = "advanced")]
    pub fn safety_compact(&self) -> Option<crate::SafetyCompact> {
        self.metadata()
            .get_pod::<crate::SafetyCompact>(crate::keys::SAFETY_COMPACT)
    }

    #[cfg(feature = "advanced")]
    pub fn cell_rescue_hints(&self) -> alloc::vec::Vec<crate::CellHint> {
        let md = self.metadata();
        let Some(entry) = md.get(crate::keys::CELL_RESCUE_HINTS) else {
            return alloc::vec::Vec::new();
        };
        let len = core::mem::size_of::<crate::CellHint>();
        if entry.value.len() % len != 0 {
            return alloc::vec::Vec::new();
        }
        entry
            .value
            .chunks_exact(len)
            .map(bytemuck::pod_read_unaligned::<crate::CellHint>)
            .collect()
    }

    #[cfg(feature = "advanced")]
    pub fn zq_fallback_table(&self) -> alloc::vec::Vec<crate::FallbackEntry> {
        let md = self.metadata();
        let Some(entry) = md.get(crate::keys::ZQ_FALLBACK_TABLE) else {
            return alloc::vec::Vec::new();
        };
        let len = core::mem::size_of::<crate::FallbackEntry>();
        if entry.value.len() % len != 0 {
            return alloc::vec::Vec::new();
        }
        entry
            .value
            .chunks_exact(len)
            .map(bytemuck::pod_read_unaligned::<crate::FallbackEntry>)
            .collect()
    }

    #[cfg(feature = "advanced")]
    pub fn output_bounds(&self) -> alloc::vec::Vec<crate::OutputBound> {
        let md = self.metadata();
        let Some(entry) = md.get(crate::keys::OUTPUT_BOUNDS) else {
            return alloc::vec::Vec::new();
        };
        let len = core::mem::size_of::<crate::OutputBound>();
        if entry.value.len() % len != 0 {
            return alloc::vec::Vec::new();
        }
        entry
            .value
            .chunks_exact(len)
            .map(bytemuck::pod_read_unaligned::<crate::OutputBound>)
            .collect()
    }

    pub fn feature_transforms(&self) -> Option<&[FeatureTransform]> {
        self.feature_transforms.as_deref()
    }

    /// Per-feature transform params from the
    /// `zentrain.feature_transform_params` metadata key, parallel to
    /// [`Self::feature_transforms`]. `None` when the metadata was
    /// absent (consumer treats every feature's params as `&[]`).
    /// `Some(_)` with `len == n_inputs`; each inner slice is empty
    /// for features whose transform variant doesn't consume params
    /// (e.g., `Identity`, `Log1p`, `SignedLog1p`).
    ///
    /// Required for the V0_20 parameterized variants
    /// ([`FeatureTransform::ClipThenLog1p`],
    /// [`FeatureTransform::WinsorP99`],
    /// [`FeatureTransform::QuantileBins`]). When a parameterized
    /// variant is present but its per-feature params slice is empty,
    /// [`FeatureTransform::apply_with_params`] falls back to a sane
    /// no-op (see that method's docs).
    pub fn feature_transform_params(&self) -> Option<&[alloc::vec::Vec<f32>]> {
        self.feature_transform_params.as_deref()
    }

    pub fn has_nontrivial_feature_transforms(&self) -> bool {
        self.feature_transforms
            .as_deref()
            .is_some_and(|ts| ts.iter().any(|t| *t != FeatureTransform::Identity))
    }

    /// True iff at least one declared transform is a scalar-to-vector
    /// expander (today: only [`FeatureTransform::Sinusoidal`]). Used by
    /// the Predictor to decide between the scalar fast path and the
    /// variable-arity pipeline.
    pub fn has_expander_feature_transforms(&self) -> bool {
        self.feature_transforms
            .as_deref()
            .is_some_and(|ts| ts.iter().any(|t| t.is_expander()))
    }

    /// Sum of per-feature output arities under the loaded transforms +
    /// params. For scalar-only bakes equals `n_inputs()`; for bakes
    /// containing [`FeatureTransform::Sinusoidal`] equals the
    /// post-expansion layer-1 input width. Returns `n_inputs()` when no
    /// transforms metadata is present.
    pub fn expanded_input_dim(&self) -> usize {
        let Some(transforms) = self.feature_transforms.as_deref() else {
            return self.n_inputs();
        };
        match self.feature_transform_params.as_deref() {
            Some(all_params) => {
                debug_assert_eq!(all_params.len(), transforms.len());
                let mut total = 0usize;
                for i in 0..transforms.len() {
                    total += transforms[i].output_arity(&all_params[i]);
                }
                total
            }
            None => {
                // No params metadata ⇒ every transform uses empty
                // params ⇒ scalar variants have arity 1, Sinusoidal
                // has arity 0 (degenerate; bake-side validator should
                // reject this). Sum accordingly.
                transforms.iter().map(|t| t.output_arity(&[])).sum()
            }
        }
    }

    pub fn scratch_len(&self) -> usize {
        let max_out = self
            .layer_offsets
            .iter()
            .map(|l| l.out_dim as usize)
            .max()
            .unwrap_or(0);
        max_out.max(self.n_inputs())
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Iterator over a `Model`'s layers. Yields owned `LayerView<'_>`
/// values that borrow from the model.
pub struct LayerIter<'a> {
    model: &'a Model,
    idx: usize,
    end: usize,
}

impl<'a> Iterator for LayerIter<'a> {
    type Item = LayerView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.end {
            return None;
        }
        let v = self.model.materialize_layer(self.idx);
        self.idx += 1;
        Some(v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.idx;
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for LayerIter<'a> {}
impl<'a> DoubleEndedIterator for LayerIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.idx >= self.end {
            return None;
        }
        self.end -= 1;
        Some(self.model.materialize_layer(self.end))
    }
}

fn cast_f32_section<'a>(
    what: &'static str,
    section: Section,
    bytes: &'a [u8],
    expected_count: usize,
) -> Result<&'a [f32], PredictError> {
    let raw = section.slice(what, bytes)?;
    let expected_bytes = expected_count
        .checked_mul(4)
        .ok_or(PredictError::DimensionOverflow { what })?;
    if raw.len() != expected_bytes {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    if expected_count == 0 {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice::<u8, f32>(raw).map_err(|_| PredictError::SectionMisaligned {
        what,
        offset: section.offset,
        required_align: core::mem::align_of::<f32>(),
    })
}

fn cast_u16_section<'a>(
    what: &'static str,
    section: Section,
    bytes: &'a [u8],
    expected_count: usize,
) -> Result<&'a [u16], PredictError> {
    let raw = section.slice(what, bytes)?;
    let expected_bytes = expected_count
        .checked_mul(2)
        .ok_or(PredictError::DimensionOverflow { what })?;
    if raw.len() != expected_bytes {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    if expected_count == 0 {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice::<u8, u16>(raw).map_err(|_| PredictError::SectionMisaligned {
        what,
        offset: section.offset,
        required_align: core::mem::align_of::<u16>(),
    })
}

fn cast_i8_section<'a>(
    what: &'static str,
    section: Section,
    bytes: &'a [u8],
    expected_count: usize,
) -> Result<&'a [i8], PredictError> {
    let raw = section.slice(what, bytes)?;
    if raw.len() != expected_count {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    if expected_count == 0 {
        return Ok(&[]);
    }
    bytemuck::try_cast_slice::<u8, i8>(raw).map_err(|_| PredictError::SectionMisaligned {
        what,
        offset: section.offset,
        required_align: 1,
    })
}

// ─────────────────────────────────────────────────────────────────
// Helpers for load-time decompression + section validation.
// ─────────────────────────────────────────────────────────────────

fn decompress_lz4(input: &[u8], output: &mut [u8]) -> Result<(), PredictError> {
    lz4_flex::block::decompress_into(input, output).map_err(|_| {
        PredictError::SectionOutOfRange {
            what: "compressed payload (lz4 decode failed)",
            offset: HEADER_SIZE as u32,
            len: input.len() as u32,
            file_len: HEADER_SIZE + output.len(),
        }
    })?;
    Ok(())
}

fn validate_scaler_section(
    what: &'static str,
    section: Section,
    bytes: &[u8],
    n: usize,
) -> Result<(), PredictError> {
    cast_f32_section(what, section, bytes, n).map(|_| ())
}

fn validate_feature_bounds(
    section: Section,
    bytes: &[u8],
    n_inputs: usize,
) -> Result<(), PredictError> {
    if section.is_empty() {
        return Ok(());
    }
    let n_bound_f32s = n_inputs
        .checked_mul(2)
        .ok_or(PredictError::DimensionOverflow {
            what: "feature_bounds = n_inputs * 2",
        })?;
    let raw = cast_f32_section("feature_bounds", section, bytes, n_bound_f32s)?;
    let bound_bytes = bytemuck::cast_slice::<f32, u8>(raw);
    bytemuck::try_cast_slice::<u8, crate::bounds::FeatureBound>(bound_bytes).map_err(|_| {
        PredictError::SectionMisaligned {
            what: "feature_bounds",
            offset: section.offset,
            required_align: core::mem::align_of::<crate::bounds::FeatureBound>(),
        }
    })?;
    Ok(())
}

fn validate_output_specs(
    section: Section,
    bytes: &[u8],
    n_outputs: usize,
) -> Result<(), PredictError> {
    if section.is_empty() {
        return Ok(());
    }
    let raw = section.slice("output_specs", bytes)?;
    let expected = n_outputs
        .checked_mul(core::mem::size_of::<OutputSpec>())
        .ok_or(PredictError::DimensionOverflow {
            what: "n_outputs * sizeof(OutputSpec)",
        })?;
    if raw.len() != expected {
        return Err(PredictError::SectionOutOfRange {
            what: "output_specs",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    bytemuck::try_cast_slice::<u8, OutputSpec>(raw).map_err(|_| {
        PredictError::SectionMisaligned {
            what: "output_specs",
            offset: section.offset,
            required_align: core::mem::align_of::<OutputSpec>(),
        }
    })?;
    Ok(())
}

fn validate_discrete_sets(section: Section, bytes: &[u8]) -> Result<(), PredictError> {
    if section.is_empty() {
        return Ok(());
    }
    let raw = section.slice("discrete_sets", bytes)?;
    if !raw.len().is_multiple_of(4) {
        return Err(PredictError::SectionOutOfRange {
            what: "discrete_sets",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    bytemuck::try_cast_slice::<u8, f32>(raw).map_err(|_| PredictError::SectionMisaligned {
        what: "discrete_sets",
        offset: section.offset,
        required_align: core::mem::align_of::<f32>(),
    })?;
    Ok(())
}

fn validate_sparse_overrides(
    section: Section,
    bytes: &[u8],
    n_outputs: usize,
) -> Result<(), PredictError> {
    if section.is_empty() {
        return Ok(());
    }
    let raw = section.slice("sparse_overrides", bytes)?;
    if !raw
        .len()
        .is_multiple_of(core::mem::size_of::<SparseOverride>())
    {
        return Err(PredictError::SectionOutOfRange {
            what: "sparse_overrides",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    let parsed = bytemuck::try_cast_slice::<u8, SparseOverride>(raw).map_err(|_| {
        PredictError::SectionMisaligned {
            what: "sparse_overrides",
            offset: section.offset,
            required_align: core::mem::align_of::<SparseOverride>(),
        }
    })?;
    for entry in parsed {
        if (entry.idx as usize) >= n_outputs {
            return Err(PredictError::OutputDimMismatch {
                expected: n_outputs,
                got: entry.idx as usize,
            });
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// Load-time inverse permutations.
//
// `feature_order` and `output_order` are auto-sized indices (u8/u16/u32)
// stored as flat byte arrays in the header sections. Width is inferred
// from `section.len / n`. The forward permutation `π[bake_pos] = caller_idx`
// is applied at bake time; the loader applies the INVERSE to rotate the
// on-disk arrays back into caller-natural order.
// ─────────────────────────────────────────────────────────────────

/// Read a permutation index array from a section. Width is auto-detected
/// from `section.len / n`: 1→u8, 2→u16, 4→u32. Validates that every
/// index is `< n`. Returns a Vec<u32> of length `n`.
fn read_permutation_indices(
    what: &'static str,
    section: Section,
    bytes: &[u8],
    n: usize,
) -> Result<alloc::vec::Vec<u32>, PredictError> {
    let raw = section.slice(what, bytes)?;
    let width = if raw.len() == n {
        1usize
    } else if raw.len() == n * 2 {
        2
    } else if raw.len() == n * 4 {
        4
    } else {
        return Err(PredictError::SectionOutOfRange {
            what,
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    };
    let mut out = alloc::vec::Vec::with_capacity(n);
    match width {
        1 => {
            for &b in raw {
                out.push(b as u32);
            }
        }
        2 => {
            for chunk in raw.chunks_exact(2) {
                out.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
            }
        }
        4 => {
            for chunk in raw.chunks_exact(4) {
                out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        _ => unreachable!(),
    }
    // Validate: every index < n.
    for &idx in &out {
        if (idx as usize) >= n {
            return Err(PredictError::OutputDimMismatch {
                expected: n,
                got: idx as usize,
            });
        }
    }
    // Validate: permutation (every index in 0..n appears exactly once).
    let mut seen = alloc::vec![false; n];
    for &idx in &out {
        let i = idx as usize;
        if seen[i] {
            return Err(PredictError::OutputDimMismatch {
                expected: n,
                got: i,
            });
        }
        seen[i] = true;
    }
    Ok(out)
}

/// Apply the inverse of `feature_order` to all input-indexed bake data:
/// scaler_mean, scaler_scale, feature_bounds, layer[0] weight rows.
///
/// The forward permutation `feature_order[bake_pos] = caller_idx` says
/// "the bake stored this entry at `bake_pos` should be reachable at
/// `caller_idx` in caller-natural order." So the loader moves
/// `bake_data[bake_pos]` to position `caller_idx`. This is implemented
/// as a cycle-detection in-place permutation.
fn apply_feature_order_inverse(
    bytes: &mut [u8],
    header: &Header,
    layer_offsets: &[LayerOffsets],
    n_inputs: usize,
) -> Result<(), PredictError> {
    let perm = read_permutation_indices("feature_order", header.feature_order, bytes, n_inputs)?;
    // Reorder scaler_mean (n_inputs f32s).
    permute_f32_array_inverse(bytes, header.scaler_mean, &perm)?;
    permute_f32_array_inverse(bytes, header.scaler_scale, &perm)?;
    if !header.feature_bounds.is_empty() {
        // feature_bounds is n_inputs FeatureBounds = n_inputs × (low, high) f32s,
        // = n_inputs × 8 bytes.
        permute_pod_array_inverse(bytes, header.feature_bounds, &perm, 8)?;
    }
    // Layer[0] weight ROWS need permutation. Each row is `out_dim`
    // elements of the layer's weight dtype.
    let l0 = &layer_offsets[0];
    let row_stride_elems = l0.out_dim as usize;
    let elem_bytes = match l0.weight_dtype {
        WeightDtype::F32 => 4,
        WeightDtype::F16 => 2,
        WeightDtype::I8 => 1,
    };
    let row_bytes = row_stride_elems * elem_bytes;
    permute_rows_inverse(bytes, l0.weights, &perm, row_bytes)?;
    Ok(())
}

/// Apply the inverse of `output_order` to all output-indexed bake data:
/// layer[last] weight cols, layer[last] biases, output_specs,
/// sparse_overrides indices. The contract is that metadata-side output
/// indexed arrays (cell_rescue_hints, output_bounds) are emitted by
/// the trainer in caller-natural order, so they don't need permutation
/// at load.
fn apply_output_order_inverse(
    bytes: &mut [u8],
    header: &Header,
    layer_offsets: &[LayerOffsets],
    n_outputs: usize,
) -> Result<(), PredictError> {
    let perm = read_permutation_indices("output_order", header.output_order, bytes, n_outputs)?;
    let last = layer_offsets.last().expect("at least one layer");
    let in_dim_last = last.in_dim as usize;
    let elem_bytes = match last.weight_dtype {
        WeightDtype::F32 => 4,
        WeightDtype::F16 => 2,
        WeightDtype::I8 => 1,
    };
    // Permute COLUMNS of layer[last].weights — row-major with
    // in_dim rows × out_dim cols; column j is at offset
    // (row * out_dim + j) * elem_bytes within the weight section.
    permute_cols_inverse(
        bytes,
        last.weights,
        &perm,
        in_dim_last,
        n_outputs,
        elem_bytes,
    )?;
    // Permute biases (n_outputs f32s).
    permute_f32_array_inverse(bytes, last.biases, &perm)?;
    // Permute I8 scales if present (one f32 per output column).
    if !last.scales.is_empty() {
        permute_f32_array_inverse(bytes, last.scales, &perm)?;
    }
    // Permute output_specs (POD struct array indexed by output idx).
    if !header.output_specs.is_empty() {
        let spec_size = core::mem::size_of::<OutputSpec>();
        permute_pod_array_inverse(bytes, header.output_specs, &perm, spec_size)?;
    }
    // Remap sparse_overrides.idx: each entry has u32 idx at offset 0.
    if !header.sparse_overrides.is_empty() {
        remap_sparse_override_indices(bytes, header.sparse_overrides, &perm)?;
    }
    Ok(())
}

/// Apply inverse permutation to an array of `n` POD elements
/// (each `elem_bytes` bytes) at `section`. `perm[bake_pos] = caller_idx`;
/// the result moves `data[bake_pos]` to position `caller_idx`.
fn permute_pod_array_inverse(
    bytes: &mut [u8],
    section: Section,
    perm: &[u32],
    elem_bytes: usize,
) -> Result<(), PredictError> {
    let raw_offset = section.offset as usize;
    let raw_len = section.len as usize;
    let n = perm.len();
    if raw_len != n * elem_bytes {
        return Err(PredictError::SectionOutOfRange {
            what: "permute_pod_array_inverse: section.len mismatch",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    // Allocate scratch (one alloc per call; total ~few KB).
    let mut scratch: alloc::vec::Vec<u8> = alloc::vec![0u8; raw_len];
    {
        let src = &bytes[raw_offset..raw_offset + raw_len];
        for (bake_pos, &caller_u32) in perm.iter().enumerate() {
            let caller_idx = caller_u32 as usize;
            let src_start = bake_pos * elem_bytes;
            let dst_start = caller_idx * elem_bytes;
            scratch[dst_start..dst_start + elem_bytes]
                .copy_from_slice(&src[src_start..src_start + elem_bytes]);
        }
    }
    bytes[raw_offset..raw_offset + raw_len].copy_from_slice(&scratch);
    Ok(())
}

/// Apply inverse permutation to a contiguous f32 array.
fn permute_f32_array_inverse(
    bytes: &mut [u8],
    section: Section,
    perm: &[u32],
) -> Result<(), PredictError> {
    permute_pod_array_inverse(bytes, section, perm, 4)
}

/// Apply inverse permutation to ROWS of a row-major weight matrix.
/// Matrix shape: `n_rows = perm.len()` × cols (each row is `row_bytes`
/// bytes). Moves row `bake_pos` to row `caller_idx`.
fn permute_rows_inverse(
    bytes: &mut [u8],
    section: Section,
    perm: &[u32],
    row_bytes: usize,
) -> Result<(), PredictError> {
    let raw_offset = section.offset as usize;
    let raw_len = section.len as usize;
    let n_rows = perm.len();
    if raw_len != n_rows * row_bytes {
        return Err(PredictError::SectionOutOfRange {
            what: "permute_rows_inverse: section.len mismatch",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    let mut scratch: alloc::vec::Vec<u8> = alloc::vec![0u8; raw_len];
    {
        let src = &bytes[raw_offset..raw_offset + raw_len];
        for (bake_pos, &caller_u32) in perm.iter().enumerate() {
            let caller_idx = caller_u32 as usize;
            let src_start = bake_pos * row_bytes;
            let dst_start = caller_idx * row_bytes;
            scratch[dst_start..dst_start + row_bytes]
                .copy_from_slice(&src[src_start..src_start + row_bytes]);
        }
    }
    bytes[raw_offset..raw_offset + raw_len].copy_from_slice(&scratch);
    Ok(())
}

/// Apply inverse permutation to COLUMNS of a row-major weight matrix.
/// Matrix shape: n_rows × n_cols (each element `elem_bytes` bytes).
/// Moves col `bake_pos` to col `caller_idx` for every row.
fn permute_cols_inverse(
    bytes: &mut [u8],
    section: Section,
    perm: &[u32],
    n_rows: usize,
    n_cols: usize,
    elem_bytes: usize,
) -> Result<(), PredictError> {
    let raw_offset = section.offset as usize;
    let raw_len = section.len as usize;
    if raw_len != n_rows * n_cols * elem_bytes {
        return Err(PredictError::SectionOutOfRange {
            what: "permute_cols_inverse: section.len mismatch",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    let row_stride = n_cols * elem_bytes;
    let mut scratch: alloc::vec::Vec<u8> = alloc::vec![0u8; raw_len];
    {
        let src = &bytes[raw_offset..raw_offset + raw_len];
        for r in 0..n_rows {
            let row_off = r * row_stride;
            for (bake_pos, &caller_u32) in perm.iter().enumerate() {
                let caller_idx = caller_u32 as usize;
                let src_start = row_off + bake_pos * elem_bytes;
                let dst_start = row_off + caller_idx * elem_bytes;
                scratch[dst_start..dst_start + elem_bytes]
                    .copy_from_slice(&src[src_start..src_start + elem_bytes]);
            }
        }
    }
    bytes[raw_offset..raw_offset + raw_len].copy_from_slice(&scratch);
    Ok(())
}

/// Remap the `idx: u32` field of each [`SparseOverride`] in `section`
/// by applying the forward permutation: `entry.idx = perm[entry.idx]`.
/// This is NOT a permutation of the array — it's a value substitution
/// on each entry.
fn remap_sparse_override_indices(
    bytes: &mut [u8],
    section: Section,
    perm: &[u32],
) -> Result<(), PredictError> {
    let raw_offset = section.offset as usize;
    let raw_len = section.len as usize;
    let entry_size = core::mem::size_of::<SparseOverride>();
    if !raw_len.is_multiple_of(entry_size) {
        return Err(PredictError::SectionOutOfRange {
            what: "remap_sparse_override_indices: section.len % entry_size != 0",
            offset: section.offset,
            len: section.len,
            file_len: bytes.len(),
        });
    }
    let region = &mut bytes[raw_offset..raw_offset + raw_len];
    for chunk in region.chunks_exact_mut(entry_size) {
        // SparseOverride layout: { idx: u32, value: f32 } @ #[repr(C)].
        // First 4 bytes are the idx in LE.
        let old_idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize;
        if old_idx >= perm.len() {
            return Err(PredictError::OutputDimMismatch {
                expected: perm.len(),
                got: old_idx,
            });
        }
        let new_idx = perm[old_idx];
        chunk[0..4].copy_from_slice(&new_idx.to_le_bytes());
    }
    Ok(())
}
