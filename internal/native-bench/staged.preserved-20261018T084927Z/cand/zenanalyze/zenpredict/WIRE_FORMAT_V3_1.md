# ZNPR v3.1 wire format (in zenpredict 0.2.0)

Same `FORMAT_VERSION = 3` magic constant; uses previously-reserved
header bytes for new features. Old (zero-reserved) v3 bakes still
parse correctly — the loader checks the compression flag, finds it
clear, and proceeds with the existing zero-copy path.

## Header (128 bytes)

```
0..4    magic = b"ZNPR"
4..6    version: u16 = 3
6..8    flags: u16
          bit 0:    payload compressed
          bits 1-3: compression algo (0=None, 1=LZ4)
          bits 4-15: reserved (must be 0)
8..12   n_inputs: u32
12..16  n_outputs: u32
16..20  n_layers: u32
20..24  _pad0: u32 (zero)
24..32  schema_hash: u64
32..40  scaler_mean: Section
40..48  scaler_scale: Section
48..56  layer_table: Section
56..64  feature_bounds: Section
64..72  metadata: Section
72..80  output_specs: Section
80..88  discrete_sets: Section
88..96  sparse_overrides: Section
96..100 decompressed_payload_len: u32
          When flags.compressed == 0: zero.
          When flags.compressed == 1: total decompressed byte count
          of the payload that follows the 128-byte header. The
          loader pre-allocates `128 + decompressed_payload_len`
          bytes of owned storage.
100..108 feature_order: Section
          When .is_empty() (len == 0): the bake's inputs are in
          caller-natural order, no permutation applied.
          When non-empty: contains `n_inputs` indices addressing
          caller-natural positions. Width inferred from len:
            len ==   n_inputs → u8 indices  (n_inputs ≤ 255)
            len == 2*n_inputs → u16 indices (n_inputs ≤ 65535)
            len == 4*n_inputs → u32 indices (any)
          The bake's data (scaler_mean, scaler_scale, feature_bounds,
          layer[0].weights ROWS) was permuted at bake time such that
          bake_data[bake_pos] == caller_data[feature_order[bake_pos]].
          Loader applies the INVERSE permutation in-place at load to
          rotate everything back into caller-natural order — after
          load, the predict path sees no permutation.
108..116 output_order: Section
          Symmetric to feature_order, for output dim. When non-empty,
          contains `n_outputs` indices addressing caller-natural
          output positions. Affects: layer[last].weights COLS,
          layer[last].biases, output_specs, sparse_overrides indices,
          cell_rescue_hints metadata, output_bounds metadata.
116..128 reserved: [u32; 3] — must be zero, reserved for future use
```

`Header` struct field order in Rust matches this layout exactly,
including the new `decompressed_payload_len`, `feature_order`, and
`output_order` fields replacing `reserved: [u32; 8]`.

## LayerEntry (48 bytes) — unchanged from v3

The `WeightDtype::I8Lz4` variant is REMOVED. Per-layer compression
was replaced by whole-bake compression (header flags bit). Existing
in-flight V0_18 bakes with per-layer I8Lz4 must be re-baked.

```
0..4    in_dim: u32
4..8    out_dim: u32
8..9    activation: u8       (0=Identity, 1=Relu, 2=LeakyRelu)
9..10   weight_dtype: u8     (0=F32, 1=F16, 2=I8)
10..12  flags: u16           (reserved)
12..20  weights: Section
20..28  scales: Section      (len=0 unless weight_dtype == I8)
28..36  biases: Section
36..48  reserved: [u32; 3]
```

## Payload (after 128-byte header)

### Uncompressed (flags.compressed == 0)

Existing v3 layout: layer table, then data sections at offsets
named in header. Section offsets are file-absolute.

### Compressed (flags.compressed == 1)

```
128..end   compressed_blob: LZ4-block-compressed bytes of the
           equivalent uncompressed payload
```

At load:
1. Loader allocates `Box<[u8]>` of size `128 + decompressed_payload_len`.
2. Copies bytes [0..128] from input → owned[0..128].
3. Decompresses input[128..] into owned[128..].
4. From there, parses owned[..] as if it were an uncompressed bake —
   Section offsets in the header are valid against the owned buffer
   (which has the same byte layout post-decompression as the
   pre-compression source).

This means the **composer** writes the section offsets *as if* the
bake were uncompressed, even when emitting a compressed payload.
The compression is purely an envelope over the payload bytes.

## Load-time permutation flow

After decompression (or memcpy for uncompressed), but BEFORE building
the layer offsets table, the loader applies inverse permutations:

```
if feature_order is non-empty:
    inv_perm = invert(feature_order)  // u8/u16/u32 array
    for r_old in 0..n_inputs:
        r_new = inv_perm[r_old]
        if r_new != r_old:
            swap_or_permute(scaler_mean[r_new], scaler_mean[r_old])
            swap_or_permute(scaler_scale[r_new], scaler_scale[r_old])
            if feature_bounds.len > 0:
                swap_or_permute(feature_bounds[r_new], feature_bounds[r_old])
            for o in 0..layer[0].out_dim:
                swap layer[0].weights[r_new, o] and [r_old, o]
    // After this loop, all per-input arrays are in caller-natural order.

if output_order is non-empty:
    inv_perm = invert(output_order)  // u8/u16/u32 array
    // Permute layer[last] cols + biases:
    apply_perm_cols(layer[last].weights, inv_perm)
    apply_perm_array(layer[last].biases, inv_perm)
    // Permute output_specs (POD array indexed by output idx):
    if output_specs.len > 0:
        apply_perm_array_pod(output_specs, inv_perm)
    // Remap sparse_overrides.idx:
    for entry in sparse_overrides:
        entry.idx = inv_perm[entry.idx]
    // Permute metadata-side output indexed arrays:
    if cell_rescue_hints present:
        apply_perm_array_pod(cell_rescue_hints, inv_perm)
    if output_bounds present:
        apply_perm_array_pod(output_bounds, inv_perm)
```

The "apply permutation in-place" uses a small algorithm that doesn't
allocate (cycle-detection swap-permute). For ≤500-element arrays it's
~50 µs total.

After load, the in-memory layout is in caller-natural order. The
predict hot path is identical to the un-permuted case — zero per-call
overhead.

## Composer

`zenpredict-bake::bake()` ALWAYS applies HU reorder (always-on, not
flagged — see `hu_reorder` module). The composer optionally also:

- Applies input feature reorder (if `BakeRequest.feature_order` is
  Some), emits feature_order section. Trainer decides via threshold.
- Applies output reorder (if `BakeRequest.output_order` is Some),
  emits output_order section.
- Wraps the entire payload in LZ4 compression (if
  `BakeRequest.compressed` is set), emits the compression flag bits
  and `decompressed_payload_len`.

All three optional features compose: the order is always
(input_reorder → output_reorder → HU_reorder → compression). At
load, the inverse sequence applies (decompression → output_reorder
inverse → input_reorder inverse; HU is internal, no inverse needed).

## V0_18 ship

V0_18 doesn't use feature_order or output_order (zensim has n_outputs=1,
and the agent verified row reorder doesn't help V0_18). V0_18 does use:

- HU reorder (always-on): −57.7 % on layer-0 weights.
- Whole-bake LZ4 compression: −58 % on the bake overall.

Combined, the ~38 KB bake becomes ~14 KB.
