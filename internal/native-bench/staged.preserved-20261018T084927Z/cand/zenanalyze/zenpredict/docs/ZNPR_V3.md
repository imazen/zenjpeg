# ZNPR v3 — wire format spec (draft)

**Status:** unreleased; lives on `zenanalyze/main`. The crates.io `zenpredict 0.1.0` is v2-only and does NOT decode v3. Consumers (zensim, zenavif, zenwebp, zenpicker) still link the published 0.1.0; cutting them over to v3 is gated on publishing zenpredict 0.2.0 (per `~/work/zen/RECOVERY_PLAN_2026-05-08.md` Phase 4).

> Source of truth: `zenpredict/src/{model,bake/{mod,v2},output_spec,feature_transform,metadata}.rs`. This doc curates the user-visible contract; for byte-level layout consult the source.

## Why v3

- **`output_specs`** — per-output activation, clamp, snap-to-discrete, sentinel pipeline. Lets a baked model encode "the codec's caller-facing output is `chroma_distance_scale ∈ [0.4, 2.0] snapped to a discrete set, falling back to a sentinel for unrecognized inputs"` without the consumer re-implementing that math.
- **`discrete_sets`** — f32 pool referenced by output_specs. Codec-side bounds checked.
- **`sparse_overrides`** — hand-tuned `(idx, value)` patches applied post-spec. Per-output sentinel fallback.
- **`feature_transforms`** — per-feature `identity | log | log1p` applied **before** the forward pass. Closes train/serve skew on feature scaling. Lives in metadata key `zentrain.feature_transforms`.
- **`#[non_exhaustive]`** on `Header`, `BakeRequest`, `BakeError` — future fields without a major bump.

## Loading

```rust
let bytes: &'static [u8] = include_bytes!("…trained_v3.bin");
let model = zenpredict::Model::from_bytes(bytes)?;
let mut p = zenpredict::Predictor::new(model);
let out = p.predict(&features)?;
```

`Predictor` automatically applies (in order): scaler mean/scale → feature_transforms (if present) → forward through layers → output_specs → sparse_overrides → output rescue.

## Writing

```rust
use zenpredict::bake::{BakeLayer, BakeRequest, bake_v2};
let layers = [BakeLayer { in_dim, out_dim, activation, dtype, weights, biases }];
let bytes = BakeRequest::builder(schema_hash, flags, &scaler_mean, &scaler_scale, &layers)
    .with_metadata(/* TLV pairs */)
    .with_output_specs(&specs)
    .with_discrete_sets(&pools)
    .with_feature_transforms(&transforms)
    .bake()?;
```

Function name `bake_v2` is preserved for source compatibility with the small set of internal callers; despite the name, **it emits v3**. v2 outputs are no longer producible from this crate.

## Sections

| Section | Optional | Notes |
|---|---|---|
| Header (128 B) | required | magic, version, layer table offset, scaler offsets, optional section offsets |
| Scaler (mean + scale) | required | aligned at 32 B for SIMD |
| Layers (each `LayerEntry` + weights + biases) | required | f32 / f16 / i8 layouts; each layer 32-B aligned |
| OutputSpecs | optional | 32-B POD per spec |
| DiscreteSets | optional | f32 pool; OutputSpec stores byte range into pool |
| SparseOverrides | optional | 8-B POD per `(idx, value)` |
| FeatureTransforms | optional (in metadata) | key `zentrain.feature_transforms` |
| Metadata TLV | optional | (key, type, length, value); types: `utf8 / numeric / bytes` |

## Cross-version compatibility

- v3 readers MUST reject v2 bins (header magic + version field). The current zenpredict main does this.
- v3 readers SHOULD parse missing optional sections as "use default behavior" — bakes that omit `output_specs` predict via plain layer math; bakes that omit `feature_transforms` skip pre-forward feature scaling.
- v3 writers MAY produce bins with no optional sections; those round-trip to a plain `(scaler, layers)` model semantically equivalent to the legacy v2 contract.

## Public API minimization plan (Phase 4)

Pre-yagni-trim — items currently `pub` in `zenpredict::*`:

| Item | Used by | Decision |
|---|---|---|
| `Model::from_bytes`, `Predictor::{new, predict}`, `PredictError`, `error::*` | all | **keep public** |
| `argmin::*`, `AllowedMask`, `ArgminOffsets`, `ScoreTransform` | zenpicker | keep public |
| `output_spec::*`, `OutputSpec`, `OutputTransform`, `OutputValue`, `apply_spec` | zenpicker, zensim (potentially) | keep public |
| `rescue::*`, `RescuePolicy`, `RescueStrategy`, `RescueDecision` | zenpicker | keep public |
| `bake::{BakeRequest, BakeError, build_bake}` | training tools (`zenpredict-bake` CLI), zentrain | **gate behind `bake` cargo feature**, default-on for dev, off for lean runtime |
| `inference::{LayerKind, forward_f32, forward_f16, forward_i8}`, `f16_bits_to_f32`, `scale_i8_row` | nothing externally | **demote to `pub(crate)`** |

When zensim (or any pure consumer) depends on zenpredict, recommend:
```toml
zenpredict = { version = "0.2", default-features = false, features = ["std"] }
```

## Known limitations (TODO before 0.2.0 publish)

- The `feature_transforms` metadata key is documented in code comments but not formalized as a typed schema in this doc. If we want third-party tools to write transforms, formalize the JSON shape (see `zenpredict/src/feature_transform.rs`).
- `output_value::Default` semantics — exactly what "use codec's built-in default" means is consumer-defined. zenpicker uses it for sentinel fallback; zensim doesn't use OutputValue. Document the consumer contract before locking the API.
- `SparseOverride` ordering: the current code applies overrides in the order written. If two overrides target the same idx, last-write-wins. Document or reject duplicates at bake time.
- v2 read-compat: v2 bins exist on disk (zensim's currently-shipped `weights/v0_4_2026-04-30.bin`). Either (a) re-bake to v3 before publishing 0.2.0, or (b) add a v2 reader path. **Per user 2026-05-08 direction "everyone uses c3"**: re-bake. Phase 3 will produce v3 bakes; the v2 → v3 transition is a one-time migration.

## Changelog (post-0.1.0, unreleased — what 0.2.0 will document)

- BREAKING: format bumped to v3. v2 bins no longer load. Migration: re-bake.
- BREAKING: bake module gated behind `bake` cargo feature (was always-on).
- ADD: `output_specs`, `discrete_sets`, `sparse_overrides`, `feature_transforms`.
- ADD: fluent `BakeRequest::builder(...)` API.
- ADD: `#[non_exhaustive]` on `Header`, `BakeRequest`, `BakeError` (future-proofs).

DO NOT publish 0.2.0 until:
1. Phase 3 re-bakes the zensim champion to v3 (proving end-to-end re-bake works).
2. zenavif + zenwebp `feat/expert-internal-params` branches land with caller-supplied bake API (no bundled bake).
3. The yagni-trim above is applied to keep the public surface minimal.
4. User explicitly approves the publish window.
