# Changelog

All notable changes to zenjpeg are documented here. Earlier history
(pre-2026-06) lives in `git log` and `docs/TUNING_HISTORY.md`.

## [Unreleased]

### QUEUED BREAKING CHANGES

- `encode::trellis::HybridConfig` removed — `TrellisConfig` gained
  `aq_coupling: AqCoupling` (`scale == 0.0` ≡ the old standalone mode,
  bit-identical). `EncoderConfig::hybrid_config()` and the 3-field
  `encode::ExpertConfig` overlay + `.expert()` are gone;
  `EncoderConfig::trellis()` is the single coefficient-opt entry
  (last-set-wins, no cross-clearing). `InternalParams::hybrid` →
  `InternalParams::trellis`. (eaf6e4fc→f3a5255d)
- `encode::search` module renamed to `encode::expert`;
  `encode::search::ExpertConfig` → `encode::expert::ExpertConfig`
  (re-exported as `encoder::ExpertConfig`). (676f9379)

### QUEUED BREAKING CHANGES (continued — knob discrimination wave)

- `QuantTableConfig` variants now carry their live knobs:
  `Jpegli { chroma_distance_scale }`, `JpegliSharedChroma { chroma_distance_scale }`,
  `MozjpegRobidoux { chroma_quality }`, `GlassaLowBpp` (unit — quality now
  derives from the config's `Quality`, clamped to the trained 3–25 range).
  `EncoderConfig::{chroma_distance_scale, chroma_quality}` fields, setters
  and getters are gone — the knobs exist only on the families where they
  have an effect.
- `EncoderConfig::validate()` rejects XYB + `JpegliSharedChroma` /
  `MozjpegRobidoux` (YCbCr-only table layouts; previously produced
  undefined-meaning tables).
- XYB + `chroma_distance_scale` semantics FIXED: the scale now applies to
  the chroma-like X and B channels; previously it scaled components 1,2 =
  **Y and B**, leaving X at the luma distance. Output changes for
  XYB + scale≠1.0 (no callers in-tree; the combination was mislabeled).
- `Custom` tables no longer receive per-component scaled distances —
  callers own the chroma policy via their per-component base matrices.
- `AqCoupling::quality_adaptive` removed (and the `dampen` parameter with
  it): the only call site hardwired `dampen = 1.0`, so the knob multiplied
  by 1.0 in every live path since the old `HybridConfig` days.
  `ExpertConfig::aq_trellis_quality_adaptive` removed accordingly.

### Added

- `encode::sweep` (`__expert`): budgeted sweep-plan builder over the knob
  space. Strata (`SweepAxes::rd_core`/`modes_full`) × quality grids
  (`Step5` floor / `TrainingDense`), with byte-identity fingerprint
  dedup over the RESOLVED state — aliased cells (Glassa/Piecewise anchor
  clamps, `allow_16bit` where tables fit 8-bit, `auto_optimize` vs its
  explicit spelling, output-neutral speed modes) collapse before any
  encode is spent (rd_core: 1008 candidates → 816 cells). Budget ladder
  collapses mode axes lowest-tier-first with an explicit `dropped`
  report, coarsens the q-grid uniformly (endpoints kept, ≥11 points),
  and sets `over_budget` rather than sampling silently. Invalid strata
  (XYB × YCbCr-only families) are reported, not lost.
  `SweepPlan::encodes(images, sizes)` gives the real encode count.
  Demo: `cargo run --example sweep_plan --features __expert`.

- `QuantTableConfig::PiecewiseV4` — the SA-piecewise v4 anchor tables
  (CID22-512-trained, +6.602 mean pareto vs jpegli on training, +6.09 on
  the 41-image holdout) are now a selectable 3-table family. Quality
  derives from the config's `Quality` knob; YCbCr only (rejected for XYB
  in `validate()`). `adaptive()` does NOT yet pick it — the
  piecewise×trellis and piecewise×subsampling interactions are unswept;
  that calibration is the documented next step, and per-anchor zero-bias
  SA / per-content anchor sets remain open retraining avenues.
- Per-channel chroma distances: the jpegli families' knob widened to
  `chroma_distance_scales: [f32; 2]` (`[Cb, Cr]` in YCbCr, `[X, B]` in
  XYB; each clamped to [0.1, 5.0]); `QuantTableConfig::jpegli_chroma_scale(s)`
  keeps the uniform one-liner. Internal `ResolvedQuality` (per-component
  distance vector) is now the single quality currency feeding tables and
  zero-bias.
- Per-channel zero-bias (§9.6.5): when chroma scales are non-neutral,
  each channel's zero-bias derives from that channel's own effective
  distance (new `quant_table_to_distance_component`). Neutral scales keep
  the joint three-component inversion **bit-identically** (C++-parity
  path, enforced by the locked suites). Divergent-scale outputs change vs
  the previous global-inversion behaviour — previously chroma zero-bias
  barely responded to the chroma distance at all.
- `EncodePlan.table_family` (the `QuantTableConfig` with its live knobs)
  and `EncodePlan.quality_drives_tables` (false only for
  `Custom`+`ScalingParams::Exact` tables — where changing `Quality`
  changes gates, not bytes).
- `EncoderConfig::resolve_plan(width, height) -> EncodePlan` — pure
  introspection of every resolved knob (per-component distances, table
  family + DQT precision, trellis λ-policy + AQ coupling, scan/SOF,
  restart, tiny-file). Quant tables resolve through the same function
  the streaming encoder uses, so the plan cannot drift from reality.
- `AqCoupling` on `TrellisConfig`: per-block AQ→lambda coupling with
  `exponent` / `threshold` / `max_adjustment` / `chroma_mul` /
  `multiplicative` / `quality_adaptive` knobs. Unlike the old hybrid
  path, `speed_mode` and `delta_dc_weight` are forwarded in coupled
  mode too.
- `encoder` facade now exports the expert tier: `ExpertConfig`,
  `TrellisConfig`, `TrellisSpeedMode`, `AqCoupling`, `EncodePlan`,
  `SofMarker`.

### Changed

- The `trellis` cargo feature is now an empty no-op: trellis code is
  always compiled and data-gated at runtime (default output unchanged,
  enforced by locked-hash tests). Existing `--features trellis`
  invocations keep working. (da4d64ec)
- `cargo update`: archmage/magetypes 0.9.23 → 0.9.26 (sibling
  zenpixels-convert 0.2.12 requirement); no SIMD rounding drift
  (locked suites verified). (561e65e1)

### Removed

- `HybridConfig` presets (`favor_size`/`favor_quality`/`balanced`/
  `aggressive_compression`/`safe_compression`/`quality_boost`) and the
  AQ-mean image-type heuristics (`should_use_hybrid`,
  `detect_image_type`, `adaptive_config`, `texture_adaptive_coupling`,
  `estimate_hybrid_improvement`) — all documented as derived from ~5
  images, unvalidated. The measured envelope lives in
  `docs/ENCODER_KNOB_SPACE.md`.
- `hybrid_auto_detect` example (used the deleted heuristics).

- `detect`: Windows GDI+/WIC encoder detection
  (`EncoderFamily::WindowsImaging`, `QualityScale::WindowsQuality`).
  Windows emits byte-exact IJG tables (GDI+ quality maps to index
  `q - 1` except multiples of 25; WIC integer `ImageQuality` maps to
  `q` except 53/59 — same engine, identical headers at equal index);
  detection keys on the JFIF 96×96 DPI density stamp vs
  libjpeg-turbo's 1×1 aspect ratio, and is subsampling-agnostic (WIC
  emits 4:2:0/4:4:4/4:2:2). Verified against real q=1..=100 sweeps
  for GDI+ and WIC×3 subsampling modes (400/400 family + quality
  recovery; fixtures in `zenjpeg/tests/testdata/windows_encoder/`,
  analysis in `docs/quality_estimation_research.md`).
- `jpeg_inspect`: `--detect` flag prints encoder family + estimated
  quality.
