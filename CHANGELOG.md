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

- zencodec 0.1.21 color-emit integration (`zencodec` feature): the trait
  encode path resolves which color description to embed via
  `resolve_color_emit` under the caller's `ColorEmitPolicy`. JPEG's only
  color carrier is an APP2 ICC profile (no CICP carrier), so a CICP-only
  source (e.g. Display-P3) synthesizes an embedded ICC via zenpixels-convert
  `synthesize_icc_for_cicp` instead of silently emitting an untagged
  sRGB-assumed JPEG; grayscale encodes suppress RGB synthesis. Deps:
  zencodec 0.1.21, zenpixels-convert 0.2.12 (optional, behind `zencodec`).
  Tests: `tests/bundled/emit_integration.rs` (6 cases incl. P3 synthesis
  oracle vs the bundled `DISPLAY_P3_V4` profile). Supersedes the parked
  pre-release `resolve_emit` dogfooding worktree (the scenario/EmitFacts
  machinery never shipped; its codec-level subset became 0.1.21's API).
- `adaptive()` emits `ProgressiveScanMode::Smallest` for Fast/Balanced
  effort (Max keeps `ProgressiveSearch`; Smallest × Search composition
  is an open avenue; `allow_progressive(false)` keeps Baseline). Output
  is strictly ≤ the previous Progressive emission at identical pixels.

### Changed

- `TinyFileMode::Auto` grayscale arm: structural DOMINANCE, not trial —
  single-component tiny mode is pure header pruning (~208 B) with no
  shared-table cost, so Auto takes it at every size without a gate
  (restores the pre-trial always-on grayscale behavior; pinned by a
  640×640-noise dominance test).
- `TinyFileMode::Auto` is now an exact byte-gated trial, not a
  pixel-count heuristic: the plain sequential stream is emitted, and
  when it lands ≤16 KiB the tiny shared-table variant is also
  serialized and the smaller wins (identical pixels). The legacy
  64²/128² crossover rules are gone — a 144×144 solid (legacy: no
  tiny) now ships the tiny stream because it measures smaller, and a
  degenerate large-dimension flat image is no longer mis-gated by its
  pixel count. `Force`/`Off` unchanged. All locked-hash suites pass
  unchanged. `EncodePlan.tiny_file_active` now reports only the
  structural `Force` case (a static plan cannot know a trial outcome);
  `should_activate_tiny_file_mode*` remain as legacy estimators.

- `ProgressiveScanMode::Smallest` — smallest-output entropy-stage
  selection. Coefficients are computed once; at ≤256×256 pixels they are
  serialized as up to three candidates (sequential, sequential+tiny-file
  when eligible and not `Off`, progressive jpegli script) and the exact
  min(bytes) is emitted — a pure rate decision (identical pixels),
  replacing tiny-file's pixel-count crossover heuristic with the exact
  answer. The trial gate lives in the BYTE domain: the progressive
  candidate is serialized first, and only when its output is ≤16 KiB —
  where every observed sequential win lives (~10% at 200×160 q10 noise
  at 2.3 KB; tiny-file ~1.2 KB; sweeps found zero wins above) — do the
  sequential candidates run. Pixel count was a proxy that degenerate
  (near-flat, large-dimension) content breaks; bytes are self-measuring
  and self-cost-limiting (small progressive output ⇒ little entropy
  content ⇒ cheap extra passes). Above the gate: one serialization.
  Documented at `SMALLEST_TRIAL_MAX_PROGRESSIVE_BYTES` with provenance.
  Sequential candidates are restart-free (strictly additive bytes; the
  progressive alternative cannot restart-parallel-decode either);
  `force_restart_markers(true)` restores RSTs in sequential winners.
  Tests pin: exact equality with min(explicit restart-free modes) below
  the gate, byte-identity with Progressive above it, pixel-identity
  across candidates, DRI presence under force.
  `encode::sweep::rd_core()` uses `Smallest` as its scan value — the
  scan axis leaves heuristic space.

### Fixed

- `codec.rs` zencodec tests use `.with_metadata_policy(meta,
  PreserveExact)` again: the interim revert to deprecated
  `.with_metadata(meta)` assumed the API was unpublished, but it shipped
  in zencodec 0.1.21 (now the workspace dep) — no 0.2 wait needed.

- `encode::sweep`: `rd_core()` is progressive-only (baseline never sits
  on the RD front at normal sizes — same coefficients, better entropy
  structure; it stays in `modes_full()` for the tiny-size bucket where
  sequential-only tiny-file mode wins). Boundary-RD is a sweep axis when
  built with `--features boundary-rd` (`rd_core` carries Off/On-default);
  the fingerprint hashes the RESOLVED flat knobs and only on the
  non-trellis path, so boundary × trellis cells dedupe with their
  trellis-only twins (the engine skips boundary-rd under trellis).
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
