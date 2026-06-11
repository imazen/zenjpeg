# Changelog

All notable changes to zenjpeg are documented here. Earlier history
(pre-2026-06) lives in `git log` and `docs/TUNING_HISTORY.md`.

## [Unreleased]

### Added
- `DecodedCoefficients::huffman_tables()` — harvest the decoded
  stream's DHT tables as an encoder-ready `HuffmanTableSet` for
  transcode-time table reuse via `EncoderConfig::huffman()` (single-pass
  re-encode with the source's symbol distribution). `HuffmanDecodeTable`
  now stores its DHT `bits` histogram and exposes `to_bits_values()`.
  Baseline Y/C slot convention; grayscale falls back chroma→luma;
  progressive captures final post-scan state. Closes #77.

### Changed
- `encoder::ExifFields::to_bytes()` delegates serialization to
  `zencodec::exif::Exif` (canonical authoring path: 32M+ fuzz
  executions, kamadak-exif differential tests); the private
  `build_exif_tiff` / `write_ifd_entry` TIFF writer is deleted. Public
  API unchanged. Out-of-line odd-length values now gain a TIFF 6.0
  even-offset pad byte. Floor bump zencodec 0.1.21 → 0.1.22. Closes
  #145.
- The decoder is always compiled; the `decoder` feature is a deprecated
  no-op (kept so downstream `features = ["decoder"]` keeps resolving,
  e.g. zenpipe/zencodecs). The flag gated encoder features that need
  decode roundtrips — `target-zq`'s re-decode loop, `boundary-rd`'s
  candidate IDCTs, `recompress` — and split the build for no real win.
  ~74 cfg gates removed; `default` is now empty; `boundary-rd` /
  `ultrahdr` / `zencodec` / `target-zq` / `recompress` / `layout` no
  longer pull a `decoder` feature.
- `zencodec` feature now always carries zenpixels-convert's `icc-db`
  (~36 KB bundled profile blob): CICP-described colors (incl PQ/HLG)
  always synthesize a real embedded ICC; only off-grid (reserved H.273)
  CICPs are an encode error. The `cms` feature is a deprecated no-op —
  zenpixels-convert 0.2.13 ships icc-db in its defaults, so any
  default-features consumer in a build graph (e.g. the zenpng dev-dep
  chain) flipped the capability on via feature unification regardless
  of zenjpeg's flag, making the opt-in contract untestable (Coverage CI
  red since c93bb62d) and environment-dependent.

### Fixed
- `container::xmp::parse_xmp` now parses hdrgm per-channel fields
  written as XMP elements in rdf:Seq form (`<hdrgm:GainMapMax><rdf:Seq>
  <rdf:li>…`) — Adobe Camera Raw output. Previously these parsed as
  `min = max = gamma = 0.0`, so `apply_gainmap` silently reconstructed
  HDR == SDR and `zencodec` validation rejected the params. Unparseable
  values now leave spec defaults intact instead of writing zeros.
  Fixes #144. (689c5686)

### Documentation
- `docs/VARIANT_GENERATION.md`: added the wrapped-engine patterns (8–13)
  from the zenavif adoption — encode-pinned resolution mirrors,
  ambient-machine-default pins, the three knob-liveness lies
  (neutral-value no-ops / envelope-dead / class-conditional), the
  single-deviation probe axis, resolved-state MLP feature emission, and
  two-tier validation sizing. Adoption table updated (zenavif landed);
  the absolute-valued-knob known limit now points at the proven
  delta-vs-grid-q pattern.


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

- `SweepAxes::moz_chroma_deltas` — the relative chroma-quality form the
  playbook queued: probes resolve `clamp(q + delta, 1, 100)` per cell
  (curated −10/−20 from mozjpeg's two-quality idiom; ladder sheds the
  axis first; only the follow-luma moz family crosses it). Cell ids stay
  q-free (`moz[cqd-10]`); `config_from_cell_id` resolves against the
  cell's own q, fingerprint-equal to the absolute spelling it denotes at
  each grid point (test-pinned). Closes the absolute-knob known-limit.
- Harness brought into compliance with its own playbook patterns 14/15
  (added by the zenjxl/zenavif adoptions): every cell now decode-verified
  at every quality (undecodable = hard failure; previously only q85's
  NaN was gated), and the corpus gained a 509×381 CID crop — all prior
  images were MCU-aligned, leaving JPEG's partial-MCU edge paths
  structurally unexercised. 197 cells × 8 images, ALL HARD CHECKS
  PASSED. Playbook updated in step: duplicated consumers block removed,
  checklist step 6 carries the decode/topology gates, adoption statuses
  made concrete (zenjxl steps 1–6 + zenavif full checklist landed).
- `docs/VARIANT_GENERATION.md` revised into the complete codec-neutral
  playbook: pattern 7 (cell ids as durable ledger identity — lossless
  numbers, additive-only grammar, roundtrip-totality, executor-side fp
  verification), a "where each piece lives" layering section
  (codec/executor/fleet/image + the two execution models and when each),
  supremum-contract and adversarial-safety rules for search modes and
  curated steps, harness-design gotchas (degenerate fixtures,
  content-relative floors, completion-order rows, id attribution), an
  8-step gated adoption checklist, and a live zenjxl adoption status.
- `encode::sweep::config_from_cell_id(base_id, q)` — reconstructs the
  exact `EncoderConfig` from a plan-cell stratum id. The id grammar is
  now a documented durable identity contract (fleet ledgers store these
  ids; zenmetrics verifies the carried fingerprint after parsing), with
  a grammar-totality roundtrip test over every id the planner can emit.
  Trellis numbers in ids switched from fixed precision to shortest-
  roundtrip `Display` so ids are lossless (`tr14.7500` → `tr14.75`,
  `cpl-4.0` → `cpl-4`) — done before any durable consumer existed.
- `examples/sweep_validate.rs` — empirical validation harness for the
  curated sweep axes: encodes the default stratum + every
  single-deviation stratum of `modes_full` on mixed content (CID22
  photos + adversarial synthetics) and hard-fails on inert steps,
  fingerprint-contract violations, exact-trial contract violations, and
  queue-ordering breakage; soft direction checks per the provenance
  table. First run (results: `benchmarks/sweep_validate_2026-06-10.tsv`)
  caught the five defects fixed below.


- `zencodec::GainMapRender` wired through the decode trait path (`ultrahdr`
  feature): `BaseOnly` (default, SDR base), `Components` (surfaces the decoded
  gain map as `zencodec::decode::DecodedGainMap` extras — pixels + ISO 21496-1
  params), and `ReconstructHdr { target_headroom }` — zenjpeg applies the gain
  map itself and `DecodeCapabilities::reconstructs_hdr()` honestly says so.
  Reconstruction outputs linear f32 (or f16 when preferred) RGBA at the
  requested headroom (`None` = the gain map's encoded maximum) and fulfills
  the envelope obligation: `content_light_level` (derived peak) +
  `mastering_display` (from the alternate-image capacity) on the output
  `ImageInfo`. A plain JPEG under `ReconstructHdr` decodes as its (complete)
  base image; without the `ultrahdr` feature `ReconstructHdr`/`Components`
  are an honest `UnsupportedFeature` error — never SDR-silently-labeled-HDR.
  Tests: `tests/bundled/gain_map_render.rs` (4 cases).
- `cms` feature: ICC synthesis for the color-emit path via
  `zenpixels-convert?/icc-db` (a bundled LZ4 profile blob + pure-Rust
  lz4_flex decoder — **no moxcms**; distinct from the `moxcms` feature, which
  stays for `correct_color`/XYB), weak passthrough — takes effect with
  `zencodec`, covering the full ITU-T H.273 grid incl PQ/HLG. Requires
  `zenpixels-convert` 0.2.13 (unreleased — adds the `icc-db` feature). Failing
  to synthesize a needed (off-grid) ICC is an encode **error**
  (`ErrorKind::IccError`),
  not a silent skip: JPEG has no CICP carrier, so an embedded APP2 ICC is the
  only way the color survives. Tests
  `emit_cicp_pq_without_cms_is_an_encode_error` /
  `emit_cicp_pq_with_cms_synthesizes_icc`; CI's maximal feature line gains
  `cms` while the base zencodec line keeps the no-cms error path covered.
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
- `encode::sweep` queue ordering + scalar coverage: cells now emit
  main-effects-first (all-defaults stratum, then single-deviation
  strata, then interaction combos; q ascending within a stratum so
  truncation never strands a partial RD curve; `SweepCell::deviations`
  exposes the class). `modes_full()` gains provenance-documented scalar
  steps: λ₁ ladder {13.5–16.0}, λ₂ probes {16.0, 17.0}, coupling
  {−8+clamp, −4, +4, exponent-2 probe}, delta_dc 1.0 probe, per-channel
  chroma scales {[0.5,0.5],[2,2],[1,2],[2,1]}, pre_blur 0.4. The budget
  ladder sheds one value at a time (was whole-axis: a 2000-cell budget
  now keeps 1818 cells vs 909 before) with coalesced drop reports.
- `docs/VARIANT_GENERATION.md` — the variant-generation playbook in
  codec-neutral terms (discrimination, dominance/trial/metric taxonomy,
  byte-domain gates, resolve-plan introspection, fingerprint dedup,
  budgeted ordered sweeps) with a per-codec adoption table for
  zenwebp/zenjxl/zenavif/zenpng.
- `ProgressiveScanMode::SmallestSearch` — Smallest's sequential/tiny
  trials composed with the 64-candidate scan-script search as the
  progressive candidate (the search always includes the default jpegli
  script, so SmallestSearch ≤ both `Smallest` and `ProgressiveSearch`
  at identical pixels). `adaptive()` `Effort::Max` now emits it; the
  Smallest×Search open avenue is closed. Search remains skipped for XYB
  (existing emission rule); sequential trials still apply there.
- Recompress exact emission: `tuned`/`deblock` strategies add
  `.progressive(SmallestSearch)` on top of the RD-ablated
  HybridMaxCompression param set (entropy-stage only); the `lossless`
  strategy trials both `OutputMode`s under the shared 16 KiB gate and
  ships the exact min (`ENTROPY_TRIAL_MAX_BYTES` now module-scoped).
  `preserve_emit` (own scan emitter), the per-image 8-bit DQT downgrade
  proof, and recompress clippy debt are logged in imazen/zenjpeg#143.
- `adaptive()` emits `ProgressiveScanMode::Smallest` for Fast/Balanced
  effort (Max keeps `ProgressiveSearch`; Smallest × Search composition
  is an open avenue; `allow_progressive(false)` keeps Baseline). Output
  is strictly ≤ the previous Progressive emission at identical pixels.

### Changed

- `ENTROPY_TRIAL_MAX_BYTES` raised 16 KiB → 32 KiB: `sweep_validate`
  found a real-photo counterexample above the old gate (CID22 1044329
  q10 — 19.8 KB progressive, baseline 2.0 % smaller). Counterexample and
  margin recorded at the constant.
- `ScanStrategy::Search` now also trial-encodes the canonical mozjpeg
  scan script, making `ProgressiveSearch`/`SmallestSearch` byte-supersets
  of `ProgressiveMozjpeg` (it had won by 0.09 % on a CID22 photo at q70).
- `encode::sweep::trellis_coupled()` clamps λ-adjustment to ±1.0 and the
  curated coupling steps are all clamped: the unclamped form reproduced
  the historical screenshot-destruction mode on noise (SSIM2 −31, −90 %
  bytes at q85). Unclamped remains constructible explicitly.


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

- Sweep fingerprint now hashes `TrellisSpeedMode`: it bounds the trellis
  coefficient search, so it changes output bytes (582-byte divergence on
  512² noise at q95) — the prior "output-neutral" exclusion violated the
  equal-fingerprint ⇒ identical-bytes contract.
- Sweep cell ids disambiguate λ₂, delta-DC, coupling-exponent, and
  coupling-clamp deviations (the λ₂/delta-DC/exponent probes used to
  render identically to their base configs; ids now unique across
  `modes_full`, enforced by test).
- Example lint debt cleared so `clippy --all-targets -D warnings` is
  green on default and full feature sets: migrated six examples off
  deprecated `decode_jpeg_to_rgb` (XYB-capable ones to
  `decode_jpeg_with_icc`), fixed `zensim_regress` API drift
  (`ToleranceSpec`, `latest_preview()`), and misc one-line lints.


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
