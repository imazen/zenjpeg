# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that ship together in the next minor (0.x patch
     bumps stay non-breaking). Persist across patch releases. Only
     clear when the breaking release ships. -->

- `LayerEntry.*` and `LayerView.*` field tightening may follow once
  `WeightStorage` variant pattern-matching has a stable accessor
  pair (likely `LayerView::weights() -> &WeightStorage<'_>`).

### Added

- **Deploy side of the K=1 picker knob-veto safety bounds (default surface).**
  A baked picker now enforces, at inference, the same feature-gated
  per-(categorical-axis-value) vetoes the `train_hybrid` bake gate evaluated —
  closing the gap where the rules were derived/validated but never applied at
  deploy. New public items (no feature gate — same shape as the rest of the
  picker selection kit):
  - `KnobVeto<'a>` + `VetoOp` (`LessThan` / `GreaterThan`) — one rule = "when
    `features[feat_idx] {op} threshold`, forbid these cells"; `cells` borrows the
    metadata blob (zero-copy).
  - `parse_knob_vetoes(&[u8]) -> Vec<KnobVeto>` — parser for the packed
    `zenpicker.knob_vetoes` wire blob (`u8 n_vetoes`, then per veto `u16 feat_idx`
    LE / `u8 op` / `f32 threshold` LE / `u8 n_cells` / `n_cells × u8 cell_id`).
    `knob_vetoes_from_metadata(&Metadata)` / `Model::knob_vetoes()` read it from a
    loaded bake (empty when the key is absent — backward-compatible).
  - `apply_knob_vetoes(features, vetoes, allowed: &mut [bool])` — pre-argmin
    masking pass composable with `AllowedMask`/`argmin_masked`; sets
    `allowed[cell]=false` for each fired veto. Replicates the trainer's NaN→0.0
    gate handling; the never-strand fallback stays the codec's to compose (doc'd).
  - `KNOB_VETOES_KEY` (`"zenpicker.knob_vetoes"`).

- **Default-surface picker selection kit: top-K query + runtime constraint
  masks (no feature gate).** Centralizes the masked top-`K` ranking and the
  perf/quality constraint masks on zenpredict's default API so per-codec
  pickers (and `zenpicker`) compose the proven "predict-top-K then
  encode-verify" path **without** re-implementing the masking / score-transform
  / NaN / tie-break contract in each consumer, and without an extra crate
  dependency. Items:
  - `argmin_masked_top_k::<K>` / `argmin_masked_top_k_in_range::<K>` (free fns +
    `Predictor` methods) — top-`K` lowest-scoring indices, ascending (`K = 3`
    typical). Same masking + score-transform + offsets + NaN/tie-break/mask-length
    contract as `argmin_masked`. (Were behind `advanced`; now default.)
  - `argmin::mask_at_least(values, floor, out)` — admit `values[i] >= floor`, a
    **target-quality** floor (predicted ssim2 / zensim / reach rate ≥ target).
    This is the former `advanced` `threshold_mask`, renamed + promoted to
    default.
  - `argmin::mask_at_most(values, limit, out)` — **new**, admit
    `values[i] <= limit`, a **perf / compute ceiling** (encode cost ≤ budget).
    Both masks take a caller-supplied per-cell `f32` attribute + a runtime
    threshold; `NaN` fails the constraint. AND them into the constraint mask to
    express "cheapest config reaching the target quality within the perf budget."

  The verify *loop* (rank → encode → measure → pick) stays in each codec — these
  are the generic primitives it composes over. The closure-scorer
  (`*_with_scorer`) and confidence (`pick_with_confidence*`) helpers stay behind
  `advanced`.

  Supersedes the earlier opt-in `topk`-feature proposal (PR #86) and its baked
  `u8` compute-tier system — the `topk` Cargo feature, `keys::CELL_COMPUTE_TIER`,
  `Model`/`Predictor::cell_compute_tiers()`, and the `u8` `tier_mask` are removed;
  perf is now a runtime-constrained `f32` attribute (`mask_at_most`), not a baked
  opaque rank.

- **Model self-describes its feature provenance** for the `zenanalyze-api`
  offer/reuse contract: three metadata keys `keys::ANALYZER_VERSION` (utf8),
  `keys::FEATURE_DEFS_VERSION` (u32), `keys::FEATURE_CONFIG_HASH` (u64), and
  `Model::feature_columns()` / `analyzer_version()` / `feature_defs_version()` /
  `feature_config_hash()` accessors. A codec builds a `zenanalyze_api::Request`
  (names + the three-part reuse key) from these, so it can check whether an
  offered feature result was produced by compatible definitions AND the same
  analysis config (e.g. gamma vs linear-light) before reusing it. The two numeric
  keys decode LE-explicit (i686/any-endian). All keys are `Option`/empty on bakes
  predating them (additive, non-breaking). Bakers (zentrain) populate them.

### Fixed

- **README: the flagship codec-picker example is now compilable and the
  predict input/output contract is stated.** An insulated external-developer
  test (given only the README) found the picker example uncompilable because
  three types were never documented: the `from_bytes_with_schema` schema-hash
  arg (`u64`), the `AllowedMask::new` element type (`&[bool]`), and the
  `argmin_masked` return type (`Option<usize>`, a full-output-space index).
  Also: stated that `predict`/`argmin_masked` take **raw** `&[f32]` features of
  length `n_inputs()` (the embedded scaler standardizes internally) and that
  `Predictor::new` borrows `&Model`; reconciled the contradictory alignment
  guidance (`from_bytes` copies into an owned buffer, so `include_bytes!` needs
  no `#[repr(C, align(16))]` and the input need not outlive the `Model` — but
  the `Model` must outlive the `Predictor`); and pinned the `ArgminOffsets`
  fields + `ScoreTransform::Exp` (log-domain) semantics. Docs-only.

## [0.2.0] - 2026-06-11

First crates.io publish of the **ZNPR v3** format and the runtime /
bake-tool crate split. `0.1.0` shipped the v2 format with the baker
bundled in-crate; everything below is the cumulative delta. The
intermediate `0.2.0`/`0.2.1`/`0.2.2` development cuts were never
published, so this single section is the accurate record of what
ships.

This is a **hard fork**: v1/v2 bakes do not load (they fail with
`PredictError::UnsupportedVersion`). Migrate existing bakes via
[`zentrain/tools/migrate_znpr_v2_to_v3.py`](../zentrain/tools/migrate_znpr_v2_to_v3.py)
— the rewrite is header-only; layer payloads are byte-identical
between v2 and v3.

### Changed (breaking)

- **Format**: the parser accepts only ZNPR **v3** bytes. v3 differs
  from v2 in the `version` byte plus the optional `output_specs` /
  `discrete_sets` / `sparse_overrides` sections and (v3.1) a
  whole-payload compression envelope + load-time
  `feature_order` / `output_order` permutation.
- **Bake-side composer extracted to the sibling
  [`zenpredict-bake`](../zenpredict-bake/) crate.** The `bake` cargo
  feature is gone; consumers that build bakes import `zenpredict-bake`
  directly. The runtime's monomorphization budget drops ~30–40 % (no
  more serde_json + JSON-visitor glue compiled into every codec
  binary). Default features changed from `["std", "bake"]` to
  `["std"]`.
- **Relicensed** from `AGPL-3.0-only OR LicenseRef-Imazen-Commercial`
  to `MIT OR Apache-2.0` so the runtime can be embedded in any
  MIT/Apache consumer.
- **Visibility tightening**: `Header.*`, `Section.offset` /
  `Section.len`, and `LayerEntry.*` are now `pub(crate)`. New
  accessors: `Section::new(offset, len)`, `Section::offset() -> u32`,
  `Section::len_bytes() -> u32`, and `Model::n_outputs()`.
- `OutputTransform::from_byte` and `FeatureTransform::from_token`
  promoted from `pub(crate)` to `pub` so the external bake crate's
  validator can reject unknown variants.

### Added

- **`#[non_exhaustive]` on the variant-growing public enums**
  (`FeatureTransform`, `Activation`, `WeightDtype`, `OutputTransform`,
  `ScoreTransform`, `OutputValue`, `MetadataType`) so future variant
  additions ship as non-breaking patch releases. Downstream matches
  need a `_` arm.
- **`advanced` cargo feature** (default-off) bundles the speculative
  subsystems behind one flag so lean codec runtimes don't link them:
  `safety::*`, `rescue::*`, the typed `output_spec` API
  (`predict_with_specs*`, `OutputValue`, `apply_spec`), the top-K /
  scorer-hybrid argmin family (`argmin_masked_top_k*`,
  `pick_with_confidence*`, `argmin_masked_with_scorer*`,
  `threshold_mask`), and the output-space OOD check (`OutputBound`,
  `output_first_out_of_distribution`). Wire-format slots still parse
  unconditionally — the feature gates only the typed Rust API. The
  `advanced` surface is **not yet stabilized**: items behind it may
  change or be removed in a 0.2.x patch (the default surface follows
  normal 0.x semver).
- **Feature-space out-of-distribution detection on the default
  surface**: `FeatureBound` + `first_out_of_distribution` (the only
  bounds API any consumer uses today) no longer require the `advanced`
  feature, so codecs can guard inputs without opting into the heavier
  typed subsystems.
- **`zenpredict::wire` module** exposing the shared byte-offset
  constants (`HEADER_SIZE`, `LAYER_ENTRY_SIZE`, `SECTION_OFF_*`) the
  parser and the `zenpredict-bake` composer both consume, ending a
  parser/composer drift risk.
- **Whole-bake LZ4 compression envelope.** A bake's payload (bytes
  after the 128-byte header) may be LZ4-block-compressed as a single
  envelope, marked by header `flags` bit 0 + algorithm nibble; the
  loader allocates `128 + decompressed_payload_len` and decompresses
  in place, then parses as if uncompressed. The decoder
  (`lz4_flex`, `safe-decode` only, ~4 KB) links unconditionally — no
  feature flag. This replaced the earlier per-layer
  `WeightDtype::I8Lz4` scheme (removed; weight dtypes are exactly
  `F32` / `F16` / `I8`). See
  [`WIRE_FORMAT_V3_1.md`](WIRE_FORMAT_V3_1.md).
- **Resource limits** (`src/limits.rs`), enforced before any
  allocation against the value being checked: `MAX_BAKE_BYTES =
  64 MiB`, `MAX_DIM = 65,536`, `MAX_LAYERS = 256`, plus a bound on the
  decompressed payload. A 1 GB-claiming header fails in O(1).
- **Fuzz targets** (`fuzz/`) covering `Model::from_bytes`,
  payload decompression, and the full `Predictor::predict` pipeline
  against arbitrary bytes. Corpora live under `/mnt/v/fuzzes/`
  (not committed).
- **`FeatureTransform` variant set** beyond `Identity` / `Log` /
  `Log1p`: `SignedLog1p`, `SignedSqrt`, `SignedCbrt`, `ClipThenLog1p`,
  `WinsorP99`, `QuantileBins` (`ea217f2`); the five stacked variants
  `WinsorThenLog`, `WinsorThenLog1p`, `WinsorThenSignedCbrt`,
  `SignedCbrtThenWinsor`, `ClipThenLog1pThenWinsor` (`df8190f`); and
  `YeoJohnson` with a λ-extreme overflow guard + a universal
  NaN-safety test suite (`0b11215`, `9a9be82`). Transform math mirrors
  `zentrain/tools/feature_transform_sweep.py` byte-identically so
  bake round-trips match.
- **`FeatureTransform::Sinusoidal` + a variable-arity (expander)
  pipeline.** A scalar→vector positional embedding (`[sin, cos]` at N
  frequencies) for learned per-pixel / image-domain MLPs (e.g.
  gain-MLP). It is the one expander variant: scalar `apply` /
  `apply_with_params` **panic** rather than silently pass through (a
  Sinusoidal bake fed through the scalar path is a caller bug), and a
  parallel expanding pipeline reports per-feature output arity and
  writes the multi-value output without breaking the scalar
  `apply_feature_transforms` contract. `Predictor::predict_with_specs_transformed`
  auto-routes expander bakes; the scalar path raises the new
  `PredictError::UnexpectedExpanderInScalarPipeline { feature_index }`
  (additive on the `#[non_exhaustive]` enum). (`11bc6c6`, `0bb5ddd`,
  `dec3854`; PRs #77/#78)

### Fixed

- `ScoreTransform::Exp` now applies a true `exp` on `no_std` builds
  via the unconditional `libm` dependency, instead of degrading to
  identity (the old fallthrough returned the un-exponentiated score).
  std and no_std now produce the same linear-space argmin, so a
  picker that mixes `Exp` with linear-byte `ArgminOffsets` is correct
  without the `std` feature.

### Documentation

- README + crate-level docs rewritten for v3-only + the
  runtime/bake-tool crate split, with the migration-tool pointer and
  the `advanced`-feature surface map.

### Reverted (pre-publish)

- **Multi-codec shared-trunk picker runtime (ZNPR v3.2)** — briefly
  added in the unreleased window (`60c646b`, `05f4631`) and reverted
  (`5886e4d`) before publish: the joint trainer's distillation step
  regressed zenjpeg by −7 pp argmin and no shipped codec consumed the
  runtime path. Header bytes 116..128 returned to
  `reserved: [u32; 3]`. The transfer-learning value is recoverable as
  a training-time trick (pretrain shared trunk → fine-tune per-codec
  head → bake each as a normal v3 bin). Per-codec bins, which all live
  consumers use, never carried the section and are unaffected.

## [0.1.0] - 2025

Initial release. ZNPR v2 format. Bake composer + JSON baker bundled
in-crate behind the `bake` feature. AGPL-3.0-only or
LicenseRef-Imazen-Commercial dual-license; relicensed to MIT OR
Apache-2.0 for crates.io publication.
