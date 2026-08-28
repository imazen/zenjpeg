# Changelog

All notable changes to zenjpeg are documented here. Earlier history
(pre-2026-06) lives in `git log` and `docs/TUNING_HISTORY.md`.

## [Unreleased]

### QUEUED BREAKING CHANGES (Pattern-B error envelope)

- **BREAKING:** the `zencodec::encode::EncoderConfig` / `EncodeJob` / `Encoder` /
  `zencodec::decode::DecoderConfig` / `DecodeJob` / `Decode` / `StreamingDecode`
  trait impls in `crate::codec` switch `type Error` from `crate::error::Error`
  to `whereat::At<zencodec::CodecError>` (a548bb98, "Pattern B"). Any downstream
  code naming the associated `Error` type directly (rather than going through
  `?`/`.into()` or the codec-agnostic `CategorizedError` trait) must update to
  the new envelope type.
- **BREAKING:** the 4 delegating convenience methods built on those trait impls
  change return type accordingly: `JpegEncoderConfig::encode` and
  `JpegDecoderConfig::{decode, probe_header, probe_full_metadata}` now return
  `Result<_, At<CodecError>>` instead of `Result<_, crate::error::Error>`.
  Recover the native `Error` detail via `CodecError::detail`
  (`zencodec::CodecError`) when needed.
- Landed on `main` in a548bb98 without an accompanying version bump, so `main`
  could otherwise publish this break as a patch release. **Defensive pre-bump:**
  crate version advanced 0.8.7 → 0.9.0 (b03e7253, Cargo.toml only — not a
  publish/tag/release; those still need the owner's explicit go-ahead) so a
  future `cargo publish` from `main` reflects the break in its version number.
  `cargo semver-checks` itself cannot currently run against this crate (see
  below) — the break was confirmed by direct inspection of the trait `type
  Error` and convenience-method return-type changes in a548bb98.
  - **Known limitation:** `cargo semver-checks check-release -p zenjpeg` fails
    to even build the "current" side — it extracts the crate standalone
    (`cargo add --path`), which loses the workspace's `[patch.crates-io]`
    override that currently supplies the unreleased `zencodec`
    `CategorizedError`/`ErrorCategory`/`LimitKind`/`CodecError` API this crate's
    error taxonomy depends on. This is a pre-existing condition of the Pattern-B
    work (a548bb98), not something introduced by the fixes in this entry.
    **Update:** `zencodec` 0.1.26 published, but the `[patch.crates-io]` entry
    was not immediately droppable — see the "zencodec 0.1.26" entry below.
    **Update 2:** `zencodec-testkit` 0.1.0 is now published and the
    `[patch.crates-io] zencodec` entry is dropped, removing this blocker
    (semver-checks not yet re-run — verify before the next release).

### Changed

- **Ultra HDR measured-nits pass (zensim campaign appendix AA).**
  (1) `ReconstructHdr`'s envelope `content_light_level` is now MEASURED from
  the reconstructed pixels via the zenpixels owner
  (`zenpixels_convert::CllMeasure::measure_max`, MaxRGB per CTA-861.3,
  BT.2408 anchor) instead of being derived from the gain map's declared
  capacity — a range bound that over-states content whenever the range isn't
  fully used — and MaxFALL is filled from the same scan (f16 output keeps the
  capacity-derived fallback; the mastering-display peak deliberately stays
  capacity-derived — it describes what the encoding can express).
  (2) `encode_ultrahdr_luma`'s `Bt2446C` constants now state FACTS — the
  per-transfer input normalization of the fused splitter's rows
  (`curve_input_scale_nits`: linear/sRGB → 203, PQ → 10 000, HLG → 1) and the
  curve's calibrated ~100-nit SDR reference — instead of an assumed
  1000-nit content peak + a mis-slotted 203-nit "SDR peak". VERSION FACT: the
  published zentone 0.1.0 this build resolves reserves both params (curve is
  input-relative), so (2) is byte-neutral today; the
  `bt2446c_params_inert_at_zentone_0_1` pin fails loudly when a zentone bump
  makes them live, forcing conscious re-verification. zenpixels /
  zenpixels-convert min 0.2.16 (+`hdr-experimental` under the `ultrahdr`
  feature). The declared-grid-vs-quantization-basis defect in the fused
  metadata is tracked separately (#193) and untouched here.
- **zencodec encode memory pre-flight now gates on the calibrated peak
  estimate (281c948f).** With `ResourceLimits::max_memory_bytes` set, all
  zencodec encode entry points compare the budget against
  `heuristics::estimate_encode(..)`'s `peak_memory_bytes` (the
  VmHWM-calibrated working set, a safe upper bound from the 2026-06-23
  sweep) plus the held input buffer — the same convention
  `estimate_encode_resources` reports — instead of just the `w*h*bpp`
  input buffer (1024×1024 RGB8: 3 MiB claimed vs ~9-12 MiB real peak).
  Encodes near a tight budget that previously slipped through now fail up
  front with `ErrorCategory::Resource(Limits(Memory))`; raise
  `max_memory_bytes` if the honest estimate rejects a budget you know is
  sufficient. No thread-count hook: zenjpeg's one-shot encode is measured
  serial (`encode_threading_info()` = SERIAL), so there is no
  thread-memory axis to cap.
- **The default decoder IDCT is now `IdctMethod::Libjpeg`, not `Jpegli`
  (closes #86).** Out of the box, `Decoder::new()` is now **byte-for-byte
  identical to libjpeg-turbo / mozjpeg / djpeg** — the default
  `ChromaUpsampling::Triangle` already selected turbo's fancy-upsampling, and
  `Libjpeg` switches the remaining two stages (islow IDCT + turbo's 16-bit
  YCbCr→RGB tables). Decoded pixels change by up to 2-3 levels per channel for
  **color** images that did not set `.idct_method()` explicitly; grayscale is
  unaffected (1-component sources already forced `Libjpeg`, #154). Opt back in
  with `.idct_method(IdctMethod::Jpegli)`.
  - **Why:** `Jpegli`'s 2-3 level drift broke every downstream regression
    baseline captured against libjpeg-turbo — 8 of 15 remaining imageflow
    failures in its zen-codecs-only build traced to it (#86). It is also the
    *less accurate* kernel: measured against an f64 reference it carries a
    systematic +0.002..+0.004 bias (the extra `+512` in its pass-2
    `SCALE_BITS`), where islow is unbiased.
  - **Cost:** ~3% of decode wall time (`benches/decode_zenbench.rs`). #86 held
    this open citing a "~37% decode overhead" — that figure was stale, from
    before the guarded SIMD islow kernel landed (28.1 ns vs 23.4 ns per dense
    block, i.e. +20% on the kernel alone, which is a few % end-to-end). The
    stale claim has been removed from the `idct_method()` docs.
  - `IdctMethod::Jpegli` is a **misnomer** and its docs now say so: it is the
    stb/zune 12-bit Loeffler, not what C++ jpegli uses (jpegli decodes with a
    float IDCT — zenjpeg's f32 path, which XYB/`dequant_bias` route to).
  - Pinned by `default_decoder_is_byte_exact_with_libjpeg_turbo`
    (`__ffi-tests`), which asserts a bare `Decoder::new()` == mozjpeg across
    sizes × chroma regimes × subsampling × baseline/progressive. Nothing pinned
    the old default — every prior check either set the method explicitly or
    tolerated `max_diff <= 3`, which 0 trivially satisfies.

- **build: unreliable C++ jpegli compilation no longer breaks the Rust test
  suite.** The `jpegli-internals-sys` build script previously aborted the whole
  workspace build whenever the C++ jpegli toolchain failed — a flaky/absent
  compiler, a missing system header, or the submodule on the wrong branch —
  because `cc`'s `.compile()` calls `process::exit(1)` and `cmake`'s `.build()`
  panics. It now uses `cc::Build::try_compile` and wraps the `cmake` step in
  `catch_unwind`, so ANY C++ build failure degrades to an empty crate (the
  `missing_jpegli_cpp` cfg) with a clear `cargo:warning` instead of a hard
  failure; `cargo test` still builds and runs the full Rust suite. Set
  `ZENJPEG_SKIP_CPP=1` to skip the C++ build intentionally. `zenjpeg-bench-utils`
  degrades in lockstep by reading the sys crate's new `available` build metadata
  (`DEP_JPEGLI_INTERNALS_FFI_AVAILABLE`), so its `cjpegli-ffi` helpers surface a
  runtime error rather than a compile failure. Only the `--features __ffi-tests`
  C++-parity tests are disabled when C++ is unavailable; the happy path
  (C++ present + healthy toolchain) is byte-for-byte unchanged.

- **deps: `zencodec` 0.1.25 (pre-release git pin) → 0.1.26 (released).** The
  workspace `zencodec` dep is now `{ version = "0.1.26" }` against the real
  crates.io release. The `[patch.crates-io] zencodec` entry is *retargeted*,
  not dropped: it now points at git tag `v0.1.26` (commit `998edf5f`, content-
  identical to the published crate) instead of the old pre-release rev.
  `zencodec-testkit` (in `zenjpeg/Cargo.toml`) is unpublished and path-deps
  `zencodec` internally (`{ path = "..", version = "0.1.21" }`); dropping the
  patch entirely splits the graph into two non-unified `zencodec` instances
  (registry 0.1.26 vs. the testkit checkout's path copy), which fails to
  compile every conformance test that passes zenjpeg's own types into
  `zencodec_testkit`'s generically-bound checks (`E0277`, "perhaps two
  different versions of crate `zencodec` are being used?" — confirmed via
  `cargo update -p zencodec` producing two `[[package]]` entries in
  `Cargo.lock` before this fix). Both the workspace patch and the
  `zencodec-testkit` dev-dep now pin the same `v0.1.26` tag so the graph
  unifies on one `zencodec` again. Drop the patch for good once
  `zencodec-testkit` publishes.
- **deps: `zencodec-testkit` 0.1.0 published — `[patch.crates-io] zencodec`
  retired** (follow-up to the entry above, 8ce6af00). The testkit dev-dep is
  now the plain crates.io `"0.1.0"` (its `zencodec ^0.1.26` requirement
  resolves from the registry, so the graph unifies on one `zencodec` with no
  patch), and the workspace `[patch.crates-io] zencodec` tag-pin is removed.
  Only the pre-existing `ultrahdr-core` patch remains.
- **deps: `ultrahdr-core` bumped to `>=0.5, <0.7` — the remaining
  `[patch.crates-io]` entry noted above is now retired too** (5ddb6a10).
  `ultrahdr-core` 0.6.0 published on crates.io (imazen/ultrahdr) already
  contains the git-rev this workspace was patched to (`3ac20f99`, confirmed
  via `git merge-base --is-ancestor`), so the patch is redundant. Also picks
  up `zenpixels`/`zenpixels-convert` 0.2.16 (now published), which fully
  unifies this workspace's own `moxcms` dependency onto a single `v0.9.0`
  instance in the *shipped* graph. `ultrahdr-core` 0.6.0 made
  `HdrOutputFormat::LinearF16` opt-in behind a new `f16` feature (was
  unconditional); forwarded `ultrahdr-core/f16` through this crate's own
  `ultrahdr` feature to keep `ReconstructHdr`'s `wants_f16` output path
  compiling. One isolated `moxcms 0.8.1` instance remains, confined to the
  `ultrahdr-rs` *dev*-dependency subtree (test-only, never ships) — it
  requires `zenjpeg 0.9.0` itself to publish first (see QUEUED BREAKING
  CHANGES above), a circular dependency this repo can't unblock alone.
- **deps: `codec-eval` bumped to `0.3.3`, eliminating the second isolated
  `moxcms 0.7.11` dev-dependency instance.** codec-eval 0.3.1/0.3.2 both
  required `fast-ssim2 = "^0.7.2"`, and fast-ssim2 0.7.2/0.7.3 are both
  yanked with no 0.7.4 ever published — codec-eval >0.3.0 was unresolvable
  from crates.io at all, so this workspace's Cargo.lock was stuck on 0.3.0,
  whose moxcms pin predated codec-eval's own later `moxcms = "0.8"` bump.
  Published codec-eval 0.3.3 (bumps fast-ssim2 past the yanked range, widens
  its own moxcms range to cover 0.9.0); `cargo update -p codec-eval@0.3.0`
  picked it up here. The shipped `moxcms` graph is unchanged (was already
  unified at 0.9.0) — this only removes dev-dependency graph bloat. Only the
  `ultrahdr-rs` 0.8.1 instance above remains.

### Changed

- **Auto-orient / explicit-transform permute delegates to
  `zenpixels_convert::orient::apply_orientation_into`** under the `zencodec`
  feature (#150), which now also enables zenpixels-convert's `fast-transpose`
  (AVX2 / NEON transposing kernels for every pixel width; no new dependency —
  archmage + magetypes are already mandatory). Covers u8 (1/3/4 bpp) and f32
  (1/3/4 ch) buffers; the scalar gather stays as the fallback for builds
  without `zencodec` and is held byte-identical to the delegate by
  `permute_delegation_matches_scalar_gather` (7 transforms × 6 pixel widths ×
  7 shapes incl. partial tiles). Measured with the new
  `benches/decode_orient_zenbench.rs` (12 MP, sequential, aarch64 laptop,
  noisy CV 30–70%, record in `benchmarks/decode_orient_delegation_2026-08-27.txt`):
  the Rotate90 overhead over an upright decode drops from ≈ +52 ms (RGB8) /
  +46 ms (BGRA8) to ≈ +4.6 ms / +5 ms — paired CI +9.7–19.1% vs +58–77%
  before. Rotate180 (memory-bound flip) is roughly unchanged (≈ +31/+35 ms vs
  +39/+41 ms). Default-feature builds keep the previous scalar permute.
- **Decode pipeline is monomorphized once, in zenjpeg, instead of once per
  `Stop` type in every dependent crate** (#190). Every public decode entry
  point (`decode`, `decode_into`, `decode_rows`, `decode_rows_f32`,
  `decode_coefficients`, `decode_coefficients_with_jbrd_metadata`,
  `decode_coefficients_with_extras`, `decode_to_ycbcr_f32`) keeps its
  `impl Stop` signature but is now a thin shim over a non-generic
  `&dyn Stop` body (`decode_rows*` also take the row callback as
  `&mut dyn FnMut`). No public signature changed. Measured with
  `cargo llvm-lines -p zjpeg` (the in-workspace CLI consumer, dev profile):
  total 315,269 → 247,052 LLVM lines (−21.6%); `zenjpeg::decode` internals
  instantiated in the consumer 51,658 lines / 268 fns → 2,680 / 87 (−95%) —
  previously every consumer got TWO copies of the pipeline (`Unstoppable`
  and `&Unstoppable`). Cancellation checks are per-row/per-scan so the
  indirect call is noise. Gate: `tests/decode_cancellation.rs` (a counting
  token reaches every entry point through the `dyn` boundary; a never-firing
  token is byte-identical to `Unstoppable`). The `lossless::*` pipeline
  (`encode_from_coefficients`, `restructure`, `transform`) still instantiates
  per caller (~4k lines in `zjpeg`) — a follow-up of the same shape.

### Fixed

- **A growing-prefix decode could flip from partial image back to error**
  (#92): under `Balanced`/`Lenient`/`Permissive`, a stream cut *inside* a
  table or metadata segment between scans (mid-DHT/DQT/DRI/APPn/COM before
  the next scan) errored with `TruncatedData`, while both the shorter prefix
  (ending after the previous scan) and the longer one (ending in the next
  scan's data) decoded. The between-scans recovery now covers segment bodies
  too, reporting `DecodeWarning::TruncatedBetweenScans { scans_decoded }`
  exactly as for a cut at the marker boundary (no scan had started). Found by
  the new `tests/decode_truncation.rs`, which decodes EVERY byte prefix of
  baseline / restart-interval / progressive / grayscale fixtures and asserts:
  no panic, header dimensions on every `Ok`, monotone acceptance (a longer
  prefix of a decodable stream must decode), `Strict`-accepted prefixes are
  pixel-identical under `Balanced`, and any prefix that lost scan data
  carries a `Truncated*` warning. Progressive fixtures fail at prefix 364 of
  881 before the fix.
- **Truncated scans no longer decode phantom data past the cut** (#92). The
  bit reader zero-extended past the END OF THE DATA exactly as it does at a
  marker, so every decode path kept "decoding" the rest of a cut scan out of
  synthetic zero bits: each block below the cut became whatever symbol the
  all-zero code maps to (an optimized AC table's `0x01` → a `-1 << al`
  coefficient in every block; 66k phantom coefficients on one 800×600
  progressive prefix), and a cut mid-symbol handed the residual bits to the
  NEXT block as its DC. Now: past the data with no marker the reader serves
  only the real bits (`BitReader::starved` flags the first read that asked
  for more), `Truncated` drops the unfinished residue, and the AC-refinement
  scan returns "not complete" instead of finishing a code/sign bit against
  zeros. Zero-extension at a *marker* is unchanged (a conformant segment's
  last symbol still decodes against padding). Consequences that were also
  bugs: the streaming baseline paths `continue`d past a truncated block and
  shipped whatever the previous MCU row had left in the strip (now the
  documented zero block); the speculative padding-block arm rewound on
  `Truncated` as if the encoder had omitted the block; a cut exactly at a
  restart-marker boundary errored (`read_restart_marker_tolerant`: a stream
  that ENDS where the marker belongs is a truncation, wrong bytes there are
  still corruption); `scan_rst_markers` reported `entropy_end = len - 1` on a
  marker-less tail (the last byte carries coded bits) and did not re-examine
  the second `0xFF` of a fill run as a marker prefix; and the fused parallel
  4:2:0 fancy path (`--features parallel`) left the junction below the cut
  unblended and, because the last present segment nominally ran to the end
  of the image, saved a grey row as its boundary — the fused output differed
  from sequential on the two pixel rows around the cut. Gates, all in
  `tests/decode_truncation.rs`: coefficient-domain monotone convergence over
  EVERY byte prefix (a coefficient may never move away from its final value
  — the phantom-data detector), zero fill two MCU rows below a baseline cut,
  the issue's 8-chunk progressive-arrival simulation (each arrival strictly
  improves at least one coefficient, pixel RMS-to-final non-increasing,
  final arrival byte-identical to one-shot), and fused-parallel vs
  sequential byte-identity over 257 spread cuts + 512 consecutive cuts
  through two DRI fixtures; plus `foundation::bitstream::tests` for the
  reader contract and `rst_scan::tests` for the scanner. This supersedes the
  issue's proposal 3 (per-scan rollback): a partially received progressive
  scan now contributes exactly the bits that arrived and nothing invented,
  which is the libjpeg-turbo partial-scan behaviour the issue called the
  more expensive option. Hot-path cost: none measured — `decode_zenbench`
  `progressive_4:2:0_Q85` (10 CID22 512² images, 200 rounds, interleaved
  with mozjpeg as the reference lane; M-series laptop shared with another
  build agent, so only the ratio is meaningful): before, zenjpeg 12.6 ±0.5
  ms at +9.5%..+13.2% vs mozjpeg; after, 11.8 ±0.3 ms at +1.9%..+5.2%.

### Added

- **`recompress` preserve strategy ships the smaller of sequential and
  progressive** (#143 item 2): `preserve_emit::emit_preserved` serializes
  the edited coefficients sequentially and, when that lands at or below
  `ENTROPY_TRIAL_MAX_BYTES` (32 KiB — the same gate as the main encoder's
  trials; the issue body's 16 KiB was wrong), also serializes the SAME
  coefficients progressively through the lossless pipeline's emitter
  (`lossless::restructure::encode_progressive_from_coefficients`, jpegli
  scan script) and returns the shorter stream. Pure rate decision —
  coefficients are preserved by construction either way. The edited planes
  are moved, not cloned, into the trial. Gate:
  `preserve_emit::smallest_trial_tests::preserve_emit_ships_the_smaller_of_sequential_and_progressive`
  (identical coefficient planes for both candidates and vs the source; the
  shipped bytes equal the shorter candidate; progressive wins on all three
  fixtures, e.g. 1830 → 1118 B). This closes the last open item of #143.
- **Per-image exact 8-bit DQT downgrade for `allow_16bit_quant_tables(true)`
  users** (#143 item 1): once every block is quantized, the buffered builder
  checks each 16-bit table's >255 positions against that table's component
  blocks (Cb+Cr for a shared chroma table, all three for RGB passthrough);
  if no nonzero coefficient sits there, the 8-bit clamp is emitted instead —
  pixel-identical by construction (zero coefficients dequantize to zero
  under any divisor), 64 fewer DQT bytes per table, and SOF0 when no 16-bit
  table remains. Tables whose positions ARE used keep 16-bit. Only the
  buffered (Huffman-optimized, the default) builder can do this; the
  streaming-through path writes its headers first and keeps the plan's
  precision. Default configs (`allow_16bit` off) are byte-unchanged. Gates:
  `dqt_downgrade_is_exactly_the_dominance_check` (unit, every branch) and
  `tests/dqt_downgrade.rs` (DC-only chroma at Q50 → all-8-bit DQT + SOF0;
  2×2-cell colour checkerboard → 16-bit kept).
- **`fuzz_truncation` target** (#92, proposal 6): drives the decoder over
  several cuts of each input (including one steered by the last byte) and
  checks the same contract as `tests/decode_truncation.rs` — no panic,
  header dims, monotone acceptance, Strict==Balanced pixels — plus the
  `decode_rows` / `decode_coefficients` routes. Wired into `just fuzz` and
  `fuzz/README.md`. Still open from #92: progressive per-scan rollback
  (proposal 3) and a rows/bytes-consumed completeness signal (proposal 4) —
  both are public-behaviour / public-API decisions, see the issue.

### Fixed (continued)

- **`QuantTableConfig::PiecewiseV4` anchors were non-monotonic across quality**
  (#12): the 20 SA-optimized anchors were each optimized independently, so
  1,265 of the 3,840 `(position, anchor)` cells quantized COARSER at the next
  higher quality (luma DC q90=5 → q95=37 → q100=6; Cb DC q100=81, coarser than
  q5's 66), and the lerp inherited the wobble — file size went DOWN with
  rising q at 30 of 98 steps on a 512² noise+patches sweep (0 for the jpegli
  tables). The public `ANCHOR_LUMA/CB/CR` are now the raw anchors passed through
  a compile-time per-cell L2 isotonic regression (pool-adjacent-violators)
  enforcing non-increasing quant values as q rises; the raw data stays in-tree
  as `RAW_ANCHOR_*`. **This moves the opt-in tables' bytes:** 67% of cells
  changed, mean |Δ| 7.8, max |Δ| 113 (the L2-minimal monotone fit). The raw
  anchors' pareto figures (+6.602 training / +6.09 holdout vs jpegli) have NOT
  been re-measured on the smoothed tables — the module doc now says so and no
  longer recommends the family as `adaptive()`'s default. Gates: per-cell and
  whole-q-range monotonicity unit tests plus the q1–q99 size sweep in
  `tests/piecewise_v4.rs` (all mutation-verified against the raw anchors).
  Default behaviour (`QuantTableConfig::Jpegli`) is unaffected.
- **`recompress` feature lint debt** (#143 item 3): `cargo clippy --features
  recompress -- -D warnings` had 16 failures nobody saw because no CI job
  compiled the opt-in module. Items reachable only through the
  `recompress-expert` re-export (the `aq` diagnostics, `TableId` /
  `CellEstimate::preferred` / `CellCi::{Tight,Empty}`, `StrategyParams::ci`,
  `SourceAnalysis` fields, `StrategyOutcome::measured_zensim_a`, the
  `EmitConfig` builders) now carry
  `#[cfg_attr(not(feature = "recompress-expert"), allow(dead_code))]` with a
  reason; the unwired forward `per_encoder::lookup` is documented as such and
  the measured jpegli `*_RATIO` tables stay referenced rather than deleted; a
  stray cast in `tests/recompress_api.rs` and a helper placed after the test
  module in `preserve_emit.rs` are fixed. CI gains a `Clippy (recompress)` step
  (plain `recompress`, deliberately without `-expert` so the re-export cannot
  mask dead code). Items 1 (per-image 8-bit DQT downgrade proof) and 2
  (`preserve_emit` smallest-trial) of #143 are still open.
- **`target-zq` bucket detection requested the full `FeatureSet::SUPPORTED`
  analysis** (#135, as re-scoped in its triage comment): `detect_bucket`
  (`encode/zq.rs`) now requests `adaptive::BUCKET_FEATURES` — exactly the 12
  features `infer_bucket` reads — instead of the picker's 108-feature input
  vector. Bucket output is unchanged (gated by
  `bucket_features_cover_every_feature_infer_bucket_reads`, which also fails if
  a feature `infer_bucket` reads is ever dropped from the set). The picker's
  own `SUPPORTED` request is untouched: narrowing it would misalign the MLP.
  The issue's original 51-feature / v2.2 framing and its per-call ms savings
  were measured against a bake that no longer ships and are not carried over.
- **Ultra HDR fused encode wrote under-boosted files** (#193, the ultrahdr#33
  defect class): `encode_ultrahdr_with_curve` / `encode_ultrahdr_luma` quantized
  gain-map bytes on the CONFIG boost grid (`compute_gain_row`) but
  `build_gainmap_metadata` declared the content's OBSERVED gain range as the
  per-channel `min`/`max`. Readers dequantize on the declared range, so any
  image whose gain range was narrower than the configured one reconstructed
  under-boosted in every conformant reader — with the default grid
  (`max_boost = 6`) a flat 4× patch came back at 2.9× (−27%). The metadata now
  declares the config grid; the observed accumulator only widens
  `alternate_hdr_headroom` (mirror of ultrahdr-core `a09478f0bfaa`).
  Regression gate: `tests/ultrahdr_gainmap_grid.rs` (declared range == grid
  structurally, plus a full-weight round-trip peak check). **Interop note:**
  files written by earlier versions are mis-declared and cannot be repaired by
  readers (the file does not record the true grid) — re-encode from source.
- **Three more count/emit divergence bugs in the main encoder** (sweep issue
  #197, verified by adversarial review; same #194 mechanism): (1) the three
  XYB/RGB-passthrough frequency-counting passes carried DC prediction across
  the whole image while their paired emitters reset it at every restart
  interval — a post-restart DC category the count never saw got no code and
  was emitted as ZERO bits (undecodable output for XYB + restart markers,
  the default `restart_mcu_rows(4)` included); counting now mirrors
  `check_restart` exactly. (2) With `--features parallel`, the emitter alone
  silently substituted restart interval 64 when the config said 0: RST
  markers with no DRI header plus histogram/emission divergence. The
  documented auto-selection now happens ONCE at config computation
  (`resolve_restart_rows(4, ...)`), so the DRI header, frequency counting,
  and segmented emission all agree. (3) Custom Huffman tables (harvested-table
  reuse, #77) still pass through byte-identically, but can no longer
  silently corrupt: the block-array paths coverage-verify the caller's
  tables against the exact symbol stream (same traversal as the
  optimizers), and the streaming emitter errors loudly on any codeless
  symbol instead of writing zero bits; previously `.huffman(...)` with
  tables the content exceeds (e.g. Annex K under XYB's SOF1 range)
  silently produced undecodable streams. The built-in XYB fixed-table families are
  now completed against the extended range too (the #196 completion had
  floored DC at category 11). Progressive replay fallbacks that could write
  zero bits or silently skip promised extra bits on eobruns underrun now
  return internal errors. Regression gates: `tests/huffman_consistency.rs`
  (XYB/YCbCr restart matrices with DC-slamming content, decode-validated;
  parallel restart-0 consistency; custom-table rejection).

- **`optimize_huffman(false)` baseline encodes silently produced undecodable
  JPEGs on content outside the training-corpus distribution** (found
  2026-08-26 by the missing-symbol debug_assert added with the #194 fix; the
  frymire hash-lock suite had locked corrupt bytes for every `huffman=fixed`
  row — mozjpeg djpeg and zenjpeg's own decoder both reject them). Root
  cause: the corpus-trained built-in Huffman tables were baked from observed
  frequencies only, so 324 of 360 tables lacked codes for 13,238 legal
  baseline symbols (large DC categories, rare run/size pairs), and
  `HuffmanEncodeTable::encode` emits ZERO bits for a codeless symbol. Fix:
  `builtin_tables::select_tables` now completes every table — incomplete
  tables are re-derived from frequencies synthesized to preserve the baked
  ranking (freq = 2^(24-len), floor 1 for missing legal symbols), so common
  symbols keep near-identical code lengths and rare symbols get valid long
  codes. Measured cost on photo-like content: +0.06..+0.5% size (mostly the
  fuller DHT markers), speed equal-or-faster on every measured cell.
  Regression gate: `selected_tables_cover_all_legal_symbols` +
  `audit_builtin_table_symbol_coverage` in `huffman/builtin_tables.rs`.
- **Hash-lock/byte-identity encoder tests now also decode every stream they
  hash** (locked_values — check AND regenerator, ycbcr_locked,
  boundary_rd_hash_lock, parity_reference_locked, encoder_regression
  dispatch-parity, lossless_dispatch_parity). A hash lock alone blesses
  whatever bytes the encoder produced — that is exactly how the corrupt
  fixed-table streams stayed locked-green. The regenerator now refuses to
  lock undecodable bytes. New `.github/workflows/regen-locked-values.yml`
  (workflow_dispatch) regenerates `values_archmage.csv` on an x86_64 runner.

- **Lossless transforms/restructure emitted corrupt or silently-wrong JPEGs
  in four distinct ways** (issues #194, #195; fixed in c453d299, verified
  against mozjpeg 4.1.5 `jpegtran`/`djpeg`): (1) Huffman frequency counting
  walked blocks in raster order while entropy encoding walked MCU-interleaved
  order, so a DC category unique to encode order got a zero-length code and
  silently desynced the stream — the whole of #194 (Transpose/Transverse on
  subsampled chroma, 97-99% wrong pixels) plus #195's sequential Rotate90/270
  "bad Huffman code"; (2) emitters recomputed grid geometry from pixel dims
  while the decoder produces MCU-padded grids, scrambling blocks via a stride
  mismatch on non-aligned dimension-swapping transforms; (3)
  `TrimPartialBlocks` transformed the full padded grid and only shrank the
  declared dimensions, leaving relocated padding blocks inside the visible
  region (decoded cleanly, ~87% wrong pixels vs `jpegtran -trim`); (4)
  progressive restructure tokenized padded grids where T.81 A.2.2
  non-interleaved scans need exactly ceil(comp_dim/8) data units, producing
  "extraneous bytes before marker" on every non-MCU-aligned input. Fix: new
  `lossless/geometry.rs` validates every component grid against the declared
  dimensions and provides the single interleaved-scan traversal shared by
  frequency counting and encoding; trims now crop grids per-dimension BEFORE
  transforming (swap transforms no longer over-trim the dimension that could
  stay partial); progressive tokenizes true grids. Lossless re-encode of
  other-than-1/3-component JPEGs (e.g. Adobe CMYK) is now a loud
  `unsupported_feature` error instead of a silently corrupt scan. After the
  fix, all 21 #194 cells are pixel-identical to `jpegtran` and trim outputs
  are pixel-identical to `jpegtran -trim`; the unified traversal is also
  3-5% faster (transform Rotate90 2000x1333 4:2:0: 37.5 → 35.8 ms median).

### Added (lossless regression coverage, same change)

- `tests/lossless_matrix.rs`: conformance matrix over 5 subsampling modes ×
  5 dimension-alignment classes × all 8 transforms × both edge modes ×
  seq/prog output × noisy + flat-chroma content, with four oracle layers
  (exact coefficient roundtrip, exact D4 Cayley composition, ±measured-envelope
  spatial placement via box upsampling, jpeg-decoder + zune-jpeg
  cross-decoder conformance) (c453d299 follow-up).
- `tests/lossless_dispatch_parity.rs`: the integer-only lossless pipeline must
  be BYTE-identical across every archmage SIMD token permutation.
- `lossless::tests::trim_sentinel_tests`: synthetic-coefficient oracle proving
  trimmed output equals the pre-trimmed twin and padding-block content can
  never leak into the visible region.
- `HuffmanEncodeTable::encode` now `debug_assert`s the symbol has a code
  (zero release cost) — any future count/encode traversal divergence fails
  loudly in tests instead of silently corrupting the stream.

- **Four `quality_matrix` progressive tests were disabled for a bug that was
  already fixed** (57a15d65). The 4:4:4 / 4:2:2 / 4:2:0 / 4:4:0 progressive tests
  were `#[ignore]`d citing *"issue #23: progressive Q10 ~2.8% size excess vs C++
  jpegli"*. #23 was closed COMPLETED on 2026-04-15, but the ignores were never
  removed — so the tests sat dead for ~3 months. All four pass: Q10 progressive
  is +1.4..+2.4% size while scoring **+3.1..+4.1 SSIM2 better** than C++ jpegli,
  i.e. the bytes buy quality. They now run by default. (`#[ignore]` is a test
  relaxation; it hid live coverage, not a live bug.) The sibling #78 (baseline
  4:2:0 Q10 ~2.0%) is genuinely still live and stays ignored.
- **Truncation warnings no longer report `0 of 0` blocks** (#92, 57a15d65). Both
  top-level `TruncatedScan` emit sites hardcoded `blocks_decoded: 0,
  blocks_expected: 0`, contradicting the variant's own contract ("indicate how
  much data was recovered") — `0 of 0` reads as total loss while carrying no
  information. Truncation *between* scans now reports the new
  `DecodeWarning::TruncatedBetweenScans { scans_decoded }` (every scan that
  started also finished, so there is no partial scan to count); truncation where
  a scan recovered nothing now reports the real denominator (`0 of N`) via
  `JpegParser::total_mcus()`. Mid-scan truncation already self-recovered inside
  `parse_scan` with real counts and is unchanged.

### Added

- **Diffmap-guided per-block refinement for closed-loop recompression**
  (`recompress-iqa`). When the closed loop's winning candidate is Preserve, was
  measured, and clears the target with iteration budget left, remaining passes
  now convert measured overshoot into bytes: the zensim diffmap is pooled to
  per-8×8-block means and drives an AQ zero-bias depth ladder — blocks in the
  measured low-error tail (≤ p40) deepen one rung (64→48→32→16 zigzag), blocks
  the map flags (≥ p95 and > 2× median) get their mask cleared (the measured
  veto of the energy heuristic in `recompress::aq`). Refined candidates re-run
  only the coefficient-domain emit + one measurement, and are kept only when
  strictly smaller AND still clearing the target under the loop's own
  calibration arithmetic; MCU-padding/edge-sliver blocks are never touched.
  Requires a calibrated encoder class (gexp table); otherwise refinement is
  skipped. Also: closed-loop measurement now builds the source reference
  pyramid ONCE per `recompress` call (`measure::MeasureCtx`) instead of
  re-decoding the source and rebuilding the pyramid on every pass.

- **`Quality::ZqPicker(f32)` — realtime one-shot perceptual target** (#134). The
  distilled source-feature picker predicts the RD-optimal config (subsampling ×
  progressive × sharp-yuv × effort) plus a starting quality, then encodes
  **once** — no decode, no zensim, no correction pass. `EncodeMetrics::achieved_score`
  is `NaN` (predicted, not measured); cost is ~one feature pass + one encode, for
  a realtime / CDN hot path. This makes the `zq*` family a caller choice:
  `ZqPicker` (predict once) vs `Zq` / `ZqExplicit` (measure-and-correct loop).
  Both are gated on `target-zq`, which now pulls the picker runtime
  (`zenpredict` + `zenanalyze-api`) alongside `zensim` — one umbrella feature.
  Without `target-zq`, `ZqPicker` degrades to a plain encode at the fallback
  starting quality, exactly like `Zq`. New `Quality::is_perceptual_target()` and
  `Quality::zq_picker_target()` accessors. The picker warm-start (previously
  `__picker-research`-only) now also seeds the iterative loop under `target-zq`;
  `__picker-research` is a thin alias for `target-zq`. Tests:
  `zq_picker_one_shot_predicts_without_measuring`,
  `zq_picker_one_shot_differs_from_iterative_loop` (`tests/zq_target.rs`).
- **`zenjpeg::decode::Unstoppable`** is now re-exported (#168, 57a15d65). `Stop`
  already was, but `Unstoppable` — the no-op token callers must actually pass to
  invoke a decoder — was reachable only via `zenjpeg::encoder`, forcing
  decode-only users to import from `encoder` to call a *decoder*. Additive.
- **`DecodeWarning::TruncatedBetweenScans { scans_decoded }`** — see above.
  (`DecodeWarning` is `#[non_exhaustive]` and documents that variants may be
  added in minor releases.)

### Removed

- **All 4 remaining `#[deprecated]` public items** (breaking; ships with the
  0.9.0 break already queued above — no separate bump, per "accumulate breaks so
  they ship as one version bump"). Per the API policy: *no deprecation shims or
  legacy aliases — delete old APIs*. Zero in-tree callers besides one test.
  - `ScanlineReader::read_rows_ycbcr_planes` (deprecated 0.5.0) → use
    `read_rows_ycbcr_f32`. Was a pure delegating alias.
  - `ScanlineReader::read_rows_planar_i16` (deprecated 0.5.0) → use
    `read_rows_ycbcr_native_i16`. Was a pure delegating alias.
  - `TrellisConfig::speed_level(u8)` (deprecated 0.7.0) → use
    `speed_mode(TrellisSpeedMode::Level(n))`.
  - `TrellisConfig::get_speed_level()` (deprecated 0.7.0) → use
    `get_speed_mode()`.
  - Behavior note: `speed_level()` clamped on the way in
    (`Level(level.min(10))`); `speed_mode()` stores the mode verbatim. This is
    NOT a behavior change — `TrellisSpeedMode::get_limits` already clamps
    (`level.min(10)`) at the point of use, so an out-of-range `Level(15)`
    encodes identically to `Level(10)`. The old `test_speed_level_clamping`
    pinned the setter's clamp; it is replaced by `test_level_is_clamped_in_get_limits`,
    which pins the clamp that actually governs encode behavior, across the
    `nonzero_count` domain.
  - Also drops 5 stale `#[allow(deprecated)]` attributes in `encode/streaming.rs`
    + `encode/progressive.rs` that suppressed nothing (verified: removing them
    produces no warnings).
- **The deprecated no-op feature flags `decoder`, `trellis`, and `cms`**
  (breaking for manifests that still name them; ships with the 0.9.0 break
  already queued above). All three had zero `cfg` sites — the decoder and
  trellis are always compiled, and icc-db synthesis rides `zencodec`.
  All run-command references in test/example doc comments were scrubbed.
- The stale root `CONTEXT-HANDOFF.md` (wide→magetypes migration notes from
  March; the migration landed long ago and is recorded in
  `docs/TUNING_HISTORY.md`).

### Added

- **RGB passthrough encoding — `EncoderConfig::rgb(quality)` / `ColorMode::Rgb`
  (issue #185).** Stores channels verbatim as JPEG components R, G, B at 4:4:4
  with no RGB→YCbCr transform, signaled via component IDs 'R','G','B' plus an
  Adobe APP14 marker with transform=0 (libjpeg `JCS_RGB` convention; no JFIF).
  For channel-packed data (e.g. fluorescence microscopy) where cross-channel
  bleed from a color transform is unacceptable. Matches C++ jpegli's non-XYB
  `JCS_RGB` behavior: one shared Annex-K-luma quant table with linear
  `DistanceToLinearQuality` scaling, flat 0.5 zero-bias, AQ on the G channel,
  one shared optimized Huffman pair; baseline SOF0 and progressive SOF2 both
  supported. 8-bit RGB-family input layouts only. Decodes correctly in
  zenjpeg, libjpeg-family, jpeg-decoder, and zune-jpeg. (f87c722f)

### Fixed

- **`zjpeg --trellis on` was a silent no-op.** The CLI's trellis arm was
  cfg-gated on a `trellis` feature the zjpeg crate never declared, so the
  code never compiled (and had bit-rotted against the current API). Exposed
  by the no-op-feature removal; now enables `TrellisConfig::default()`.
- **`YCbCrPlanarEncoder` silently ignored a configured trellis** — its
  builder bridge simply never forwarded `config.trellis`. Fixed as a side
  effect of deduplicating the two encoder builder bridges (review R6).
- **XYB bottom-partial-strip vertical padding used the wrong stride (issue
  #186).** When `width % 8 != 0` and the bottom strip was partial, the
  perceptual-Y plane's pad rows were replicated at packed stride into a
  padded-layout buffer (phase-shifted padding, corrupted last-row tail), and
  the B plane kept the previous strip's stale rows. Bottom-edge error on a
  vertical-stripe probe: XYB-Full 130×67 last-band/interior ratio 1.15 → 1.00.
  Locked frymire hashes unaffected (its bottom row is a uniform border).
  Regression test: `tests/bundled/xyb_edge_padding.rs`. (0064e34a)

### Fixed (caterr Pattern-B follow-up bugs)

- **Adopted zencodec's origin-first two-level `ErrorCategory` reshape**
  (imazen/zencodec#116, branch `caterr-reshape`; `[patch.crates-io] zencodec`
  bumped rev `c3220d51` → `2427387f`, CI green, not yet merged/published).
  The flat 17-variant `ErrorCategory` is now
  `Image(ImageError)` / `Request(RequestError)` / `Resource(ResourceError)` /
  `Policy(PolicyKind)` / `Lifecycle(StopReason)` / `Io(CodecIoKind)` /
  `Internal(InternalKind)`, each with its own sub-kind (e.g. the old
  `MalformedImage` is now `Image(ImageError::Malformed)`,
  `UnsupportedImageFeature` is `Image(ImageError::Unsupported(
  UnsupportedImageKind::Feature))`). All four `CategorizedError::category()`
  maps (`crate::error::ErrorKind`, `detect::ProbeError`,
  `recompress::error::Error`, plus the delegating `crate::error::Error`) were
  rewritten to the new shape. Additive/non-breaking — `ErrorCategory` and its
  sub-enums are `#[non_exhaustive]`; no zenjpeg public variant renamed or
  removed. (583b8003)
- Four categorization gaps closed during the rev-bump audit (all additive,
  same commit 583b8003):
  - **Split `ErrorKind::UnsupportedFeature{feature}` by origin.** This single
    flat variant conflated genuine JPEG-bitstream feature gaps (arithmetic
    coding, DNL, 12-bit precision, non-standard block sizes — stays
    `Image(Unsupported(Feature))`) with caller/API-entry-point-specific
    restrictions (~30 messages: scanline-reader component limits, encoder
    config combinations, `GainMapRender` modes, `decode_rows()`/
    `decode_rows_f32()` dtype mismatches — now
    `Request(Invalid(Parameters))`) via a new
    `unsupported_feature_is_request_origin()` substring classifier (no
    `zencodec::UnsupportedOperation` variant fits most of these
    zenjpeg-specific messages; that enum is closed to a handful of
    cross-codec operations). Four call sites were migrated to more precise
    existing constructors instead of the classifier: `descriptor_to_layout`'s
    and the streaming-decode fallback's pixel-format mismatches now
    construct `zencodec::UnsupportedOperation::PixelFormat` directly;
    `encode_from`'s missing-`with_canvas_size` guard now uses
    `Error::invalid_state`; `encode_from`'s internal pixel-buffer
    construction and the gain-map JPEG SOI invariant check (both operate on
    values this crate itself just computed/encoded, never caller input) now
    use `Error::internal`.
  - **Added `ErrorKind::NotAJpegFile`**, distinct from `InvalidJpegData` —
    the main decoder's SOI check (`decode/parser/mod.rs`) now reports
    `Image(Unsupported(Type))` for missing-SOI input instead of
    `Image(Malformed)`, mirroring `detect::ProbeError::NotJpeg` (which
    already made this distinction, but only on the lightweight header-probe
    path, never on the main decode path).
  - **`TooManyScans` now routes to `Resource(Limits(LimitKind::Scans))`**
    (the new structural-cap `LimitKind` variant) instead of
    `Image(Malformed)` — an anti-DoS ceiling on well-formed scan count, not
    bitstream corruption.
  - **Judged `AllocationFailed`/`SizeOverflow` to stay merged** under
    `Resource(OutOfMemory)`: the shared `foundation::alloc` helpers (~20
    call sites) are invoked from both encode- (caller-declared dimensions)
    and decode- (image-declared dimensions) driven sizes with no reliable
    per-call-site provenance in the flat `context: &'static str` payload;
    mirrors zenpng's existing overflow → `OutOfMemory` convention.

- `JpegEncoderConfig::generic_effort()` echoed the *clamped* effort value
  instead of what was actually passed to
  `with_generic_effort()`, breaking the fleet accept-signal idiom
  (set-then-get to confirm accepted input). The raw value is now stored
  verbatim and echoed back by the getter; the 0..=2 tier clamp is applied
  only at point-of-use in `effective_config()`. Round-trip test extended to
  cover out-of-tier inputs (99, -7). (d0c6366c)
- Error-category reclassifications in `crate::codec` and `crate::error`
  (additive — `ErrorKind`/`recompress::Error` are `#[non_exhaustive]`, no
  variant renamed or removed):
  - Configured `ResourceLimits` cap hits (`max_memory_bytes`,
    `max_output_bytes`, `max_input_bytes`, encode- and decode-side) no longer
    route through `AllocationFailed`/`OutOfMemory` — they use the new
    `ErrorKind::ResourceLimitExceeded { kind: zencodec::LimitKind, .. }`,
    categorizing as `ErrorCategory::LimitsExceeded(kind)`. Also fixed the
    equivalent bug in `From<zencodec::LimitExceeded> for Error`'s `Memory`
    arm and its `InputSize`/`OutputSize`/`Duration`/`TotalPixels` arms (the
    latter previously fell through to `decode_error` → `MalformedImage`).
  - `check_progressive_policy`'s decode-policy rejection (and its inlined
    duplicate) no longer report `UnsupportedFeature`/`UnsupportedImageFeature`
    — they use the new `ErrorKind::PolicyRejected`, categorizing as
    `ErrorCategory::PolicyRejected` (the request was understood and declined,
    not malformed or unimplemented).
  - `push_rows`'s width/format-mid-stream-change guard and `finish()`-without-
    `push_rows()` now use the new `ErrorKind::InvalidState` (category
    `InvalidState`) instead of `UnsupportedFeature` — these are caller
    API-protocol violations, not missing codec features.
  - `push_decoder`'s pixel-descriptor-negotiation-failed fallback arm now
    uses the new `ErrorKind::UnsupportedPixelDescriptor` (category
    `UnsupportedPixelFormat`) instead of `UnsupportedFeature`.
  - One existing test (`codec::tests::effort_levels`) pinned the old clamped-
    echo behavior and was updated together with its fix, per commit message.
  (d0c6366c)
- `recompress::error::Error` no longer flattens the typed cause to a `String`
  before categorizing: `detect::ProbeError::{TooShort,Truncated}` now produce
  the new `Error::ProbeTruncated` (category `UnexpectedEof`) instead of
  `Error::Probe` (`MalformedImage`); `From<crate::encoder::Error>` and the
  decode-for-scoring call site in `measure.rs` now capture the source error's
  real `zencodec::ErrorCategory` via the new `Error::ZenjpegCategorized`
  variant instead of collapsing every zenjpeg encode/decode failure to
  `Internal`. (96553711)
- Stale rustdoc on `SweepBuilder::with_budget`: dropped "extra color modes"
  from the shed-ladder description (color mode + subsampling is a mandatory
  axis, never shed by the ladder) and documented that the budget is
  best-effort — `SweepPlan::over_budget` is reported rather than the mandatory
  cross being silently dropped to fit. (866602c5)

### Changed
- **zencodec trait impls now return the `At<zencodec::CodecError>` envelope
  (Pattern B).** The `EncoderConfig` / `EncodeJob` / `Encoder` / `DecoderConfig` /
  `DecodeJob` / `Decode` / `StreamingDecode` impls in `crate::codec` (and the
  delegating `JpegEncoderConfig::encode` / `JpegDecoderConfig::{decode,
  probe_header, probe_full_metadata}` convenience methods) switch their
  `type Error` from `crate::error::Error` to `At<CodecError>`. This makes the
  coarse `ErrorCategory` **and** the codec name (`Some("zenjpeg")`) recoverable by
  a generic consumer *through `Dyn*` dispatch*: once the concrete error is erased
  to `zencodec::decode::BoxedError`, `CodecErrorExt::error_category()` /
  `codec_error()` downcast it back to the envelope (under the prior Pattern A
  `type Error = Error`, the erased `dyn Error` carried no `CategorizedError`
  vtable, so both returned `None`). The native `crate::error::Error`
  (`Error(At<ErrorKind>)`) is unchanged and remains the rich error for direct
  (non-`zencodec`) callers; it is now the envelope's `detail` + category source,
  bridged by `From<Error>` / `From<ErrorKind> for At<CodecError>`. Builds on the
  #103 `CategorizedError` adoption below.

### Docs
- README overhauled and split into a GitHub `README.md` (full badge row) and a generated badge-free `README.crates.md` for crates.io (the crate `readme` field now points at it); corrected the feature table (the `decoder`/`trellis` flags are documented as always-compiled no-ops), refreshed the crosslink footer, and added `benchmarks/README.md` reproduction methodology.

### Added
- **Conformance test: `check_decode_truncation_series` (zencodec-testkit)** — the
  truncation/EOF half of the codec error-taxonomy check (zencodec PR #112). Feeds a
  valid JPEG, truncates it at a deterministic prefix series, and asserts every
  dyn-erased decode categorizes as incomplete input (never panic/OOM/Internal).
  Wired as `bundled/decode_truncation_series.rs` under the `zencodec` feature; the
  `[patch.crates-io] zencodec` + `zencodec-testkit` dev-dep both pin rev `c3220d51`.
- **`zencodec::CategorizedError` implemented for the public error type** (zencodec
  #103 taxonomy adoption). `crate::error::Error` and its inner `ErrorKind` — the
  type behind `zenjpeg::decoder::Error` / `encoder::Error` / `DecodeError` /
  `EncodeError` and the `zencodec` `EncoderConfig`/`DecoderConfig` adapter's
  `type Error` — now report `codec_name() -> Some("zenjpeg")` and a **total**
  `category() -> ErrorCategory` mapping every variant: malformed bitstream →
  `MalformedImage`, truncation → `UnexpectedEof`, unsupported JPEG feature →
  `UnsupportedImageFeature`, pixel-format negotiation → `UnsupportedPixelFormat`,
  caller dials/config → `InvalidParameters`, buffer geometry → `InvalidBuffer`,
  push/finish sequencing → `InvalidState`, ICC-synthesis failure → `CmsRequired`,
  pixel-limit → `LimitsExceeded(Pixels)`, alloc / size-overflow → `OutOfMemory`,
  I/O → `Io`; the `Cancelled(StopReason)` and `UnsupportedOperation` arms delegate
  to the wrapped zencodec cause types. `recompress::Error` (feature `recompress`)
  is categorized too. Codec-agnostic consumers can now route zenjpeg failures
  (HTTP status, retry policy, logging) without naming a zenjpeg type. The match is
  exhaustive, so a future `ErrorKind` variant must be categorized or the build
  fails. **TEMP dev patch:** the workspace `Cargo.toml` pins `zencodec` to the
  `cancellation-classification-99` git branch (PR #103) until `zencodec 0.1.26`
  ships this API — revert the `[patch.crates-io]` entry and bump the dep at
  landing.
- **`AllocPreference` honored per decode allocation site + `estimate_decode_resources`.**
  The zencodec decode boundary now threads
  `ResourceLimits::prefer_fallible_allocations` (a 3-mode `AllocPreference`:
  `CodecDefault` / `Fallible` / `Infallible`) into the internal decode config and
  down to every untrusted decode allocation. `foundation::alloc` gains
  `resolve_fallible(pref, site_default)` plus runtime-flag `*_pref` variants of
  the existing fallible helpers. Per-site defaults: big untrusted-sized buffers
  (full output pixel buffer, full-frame DCT-coefficient storage, full-image
  accumulators) default to the fallible `try_reserve` path (graceful
  `Error::AllocationFailed` on a malicious SOF); small bounded MCU strip / chroma
  upsample / context-row scratch defaults to the fast infallible `vec!` path. An
  explicit `Fallible` / `Infallible` overrides every site; `CodecDefault` (the
  default, and the direct non-zencodec `DecodeConfig` API) keeps each site's
  default, so behaviour is unchanged unless a caller opts in. Also adds
  `JpegDecoderConfig::estimate_decode_resources(&ImageCharacteristics,
  &ComputeEnvironment)`, overriding the `zencodec::DecoderConfig` default by
  delegating to the existing `heuristics::estimate_decode` (peak = output buffer
  + MCU strips, plus full-frame coeff storage for progressive/subsampled; SERIAL,
  `at_cores`-scaled). New decode byte-identity tests across baseline 4:2:0 /
  4:4:4 / progressive under all three modes, plus `alloc_util`-level helper tests.
- **vCPU-aware resource estimation via zencodec's unified `estimate` API.**
  `JpegEncoderConfig::estimate_encode_resources(&ImageCharacteristics, &ComputeEnvironment)`
  overrides the `zencodec::EncoderConfig` default, delegating to the calibrated
  `heuristics::estimate_encode` and folding in available cores via
  `ResourceEstimate::at_cores`. `heuristics::encode_threading_info()` now returns
  the shared `zencodec::estimate::ThreadingInformation` (`SERIAL` — the one-shot
  zenjpeg encode does not parallelise); the short-lived local `ThreadingInfo`
  copy and `estimate_encode_threaded` are removed.
- **`ErrorKind` is now public, re-exported as `zenjpeg::decoder::ErrorKind` and
  `zenjpeg::encoder::ErrorKind`** (closes #155). The decode/encode error is
  `Error(pub At<ErrorKind>)` with `.kind() -> &ErrorKind`, but `ErrorKind` lived
  in the `pub(crate) error` module and was re-exported through zero public paths,
  so a caller could invoke `.kind()` yet could not *name* any variant to `match`
  on it — error classification (e.g. 413 too-large / 400 corrupt / 500 OOM)
  collapsed to `to_string().contains(...)` substring matching. `ErrorKind` is
  `#[non_exhaustive]`, so this re-export is additive and future variants stay
  non-breaking. New `tests/error_kind_public.rs` matches `.kind()` against named
  variants through the public path.
- **Sweep generator: `scalar_dense` preset + compute-resource constraint**
  (`encode::sweep`, `__expert`; VARIANT_GENERATION.md patterns 17–18).
  `SweepAxes::scalar_dense()` emits dense, isolated single-axis ladders over the
  continuous knobs (trellis λ₁, AQ-coupling scale, jpegli chroma-distance scale,
  pre-blur σ) — the data a trained *scalar head* (per-knob continuous regression)
  needs; pair with new `SweepBuilder::with_max_deviations(1)` (main-effects only)
  and `QualityGrid::TrainingDense` for clean per-knob response curves without a
  cartesian blow-up. New `compute_tier(&EncoderConfig) -> u8` (ordinal cost proxy)
  + `SweepBuilder::with_compute_limit(max_tier)` bound a sweep by compute budget,
  with dropped cells reported in `SweepPlan::compute_tier_skipped` (no silent
  caps). All additive, behind `__expert`.
### Fixed
- **Crate-root rustdoc decoder example now compiles and runs** (closes #155, item
  2). The top-of-page `lib.rs` decoder snippet showed a stale 1-arg
  `Decoder::new().decode(&jpeg_data)?` plus `image.pixels() -> &[u8]`; the real
  signature is `.decode(data, stop)` returning a `DecodeResult` whose bytes come
  from `.pixels_u8()`. Replaced the `rust,ignore` block with a real, compiling
  doctest (encode a tiny image, decode it round-trip) and added a second doctest
  showing `match err.kind() { ErrorKind::… }` for error classification.
### Changed
- **Located public errors: no reshape needed for the #103 guidance.** zenjpeg's
  encode/decode error is already `Error(pub At<ErrorKind>)` carrying a `whereat`
  location trace, with crate info registered via `whereat::define_at_crate_info!`
  and `#[track_caller]` constructors capturing the origin. The `CategorizedError`
  adoption above is therefore purely additive (classification only) — **not** a
  breaking error-type or signature change. (The one error type still without a
  trace is the feature-gated `recompress::Error`, a plain enum; making it located
  would be a separate, isolated change.)
- **BREAKING: `ErrorKind::Cancelled` now carries the `enough::StopReason`**
  (`Cancelled(enough::StopReason)`, was a unit variant) (8a03cb65). The wrapped
  reason distinguishes an explicit cancel (`StopReason::Cancelled`) from a
  timeout (`StopReason::TimedOut`); `Display` delegates to it via
  `#[error("{0}")]`. `From<enough::StopReason>` for both `Error` and `ErrorKind`
  now preserves the reason instead of discarding it; `Error::cancelled()` keeps
  its no-arg form (defaults to `StopReason::Cancelled`), so existing call sites
  are unchanged. Match arms on the unit `ErrorKind::Cancelled` must become
  `ErrorKind::Cancelled(_)`. The literal
  `#[error(transparent)] Cancelled(#[from] enough::StopReason)` form isn't used
  because `enough::StopReason` (0.4.4) does not implement `core::error::Error`
  (required by thiserror's `#[from]`/`transparent` for `source()` forwarding);
  the hand-written `From` impls supply the same `?` ergonomics.
- **deps: migrate to published `zencodec 0.1.24` estimate API; drop the temporary
  git-rev patch.** Removed the workspace-root `[patch.crates-io] zencodec = { git,
  rev = "0f71295" }` now that `zencodec 0.1.24` is on crates.io. Updated the
  `estimate_encode_resources` mapping for the refined `ResourceEstimate`:
  `new(peak, wall_ms: u64)` (was `f32`), `with_peak_max(max)` (the `min` arg is
  gone), and dropped the removed `with_output_bytes`.
- **Default max-pixels limit raised 100 MP → 120 MP** (`DEFAULT_MAX_PIXELS` in
  `foundation/alloc.rs`). 108 MP phone-camera photos are common and were rejected
  by the old 100 MP cap; 120 MP admits them with headroom. Non-breaking: a looser
  default limit. Callers that set an explicit `max_pixels(...)` are unaffected.
  Updated the decoder/`Limits` default docs and README examples to match.

### Tests
- **All decode paths verified byte-identical under `IdctMethod::Libjpeg`.** New
  `tests/libjpeg_idct_all_paths_parity.rs` asserts that `decode()` (streaming),
  `scanline_reader()` (pull-based), and the multi-threaded fused-parallel path
  produce **byte-for-byte identical** u8 RGB under `IdctMethod::Libjpeg`, across
  4:2:0 / 4:2:2 / 4:4:0 / 4:4:4 × baseline/progressive × MCU-aligned and
  non-aligned sizes. Previously only 4:2:0 baseline `decode()` vs
  `scanline_reader()` was covered; this widens the lock to the full matrix, so a
  caller gets the libjpeg-turbo-exact reconstruction regardless of which path the
  decoder auto-selects. Under `--features __ffi-tests` it additionally asserts
  every path is byte-identical to **real libjpeg-turbo** (`mozjpeg-sys`) across
  4:2:2 / 4:4:0 / 4:4:4 too — extending the prior 4:2:0-only FFI exactness check
  to every subsampling on every path (verified `max_diff == 0` throughout).
  (The f32-output path routes the same unclamped libjpeg islow IDCT but in f32
  precision, so it is intentionally not byte-exact with the u8 paths — guarded
  loosely. The `force_f32_idct` transform path and Knusperli deblocking use the
  f32 IDCT by design and are out of scope.)

### Fixed
- **docs(readme): unify `Unstoppable` import path + quality arg type, name
  `Strictness`, add end-to-end decode→encode example — fixes first-try compile
  gaps found by insulated-developer test.** An external-developer usability test
  (given only the README, no source) found the README would not compile first
  try: it mixed `use enough::Unstoppable` (an undocumented direct dep) with the
  dependency-free `zenjpeg::encoder::Unstoppable` re-export; mixed int and float
  quality args (`ycbcr(85, …)` vs `ycbcr(85.0, …)`); named `.strictness(Balanced)`
  without ever naming/importing the `Strictness` type; imported `TargetColorSpace`
  from the crate-private `zenjpeg::color::icc` path; and had no single
  decode→re-encode example. Now: all snippets use `zenjpeg::encoder::Unstoppable`
  (the only path reachable for both decode and encode); all quality args are float;
  `Strictness` (`zenjpeg::decoder::Strictness`) and `Limits`
  (`zenjpeg::encoder::Limits`, with a real `.limits(Limits)` call) are named and
  imported; `TargetColorSpace` uses the public `zenjpeg::decoder` re-export; and a
  copy-pasteable "decode → re-encode at quality 80, with limits + cancellation"
  example sits at the top. The `Stop`-trait cancellation snippet now shows the real
  `check() -> Result<(), StopReason>` signature and adds `enough = "0.4"` for that
  one case. Filed an API-ergonomics issue: `Unstoppable`/`Stop` are not re-exported
  from `zenjpeg::decoder` (or `zenjpeg::ultrahdr`), so decode-only users must reach
  into the `encoder` module — a footgun a docs pass can only paper over.
- **Progressive decode infinite loop (DoS) on a marker-less restart drain (fuzz
  zenpipe#47).** A small progressive JPEG with a restart interval but a missing
  RST marker spun forever in the AC-scan restart-drain loop: past EOF,
  `BitReader::refill()` keeps claiming synthetic zero bits (overread), so
  `bits_available() >= 32` stays true while `marker_found()` never fires, and
  `while self.reader.marker_found().is_none() { … }` never terminates. Both the
  first-scan (`decode_ac_first_scan_tracked`) and refine-scan
  (`decode_ac_refine_scan_tracked`) drains now also break on
  `is_exhausted()`; `read_restart_marker` then reports the missing marker
  cleanly. Found via the fuzz farm (`fuzz_job_decode` → zencodecs → zenjpeg),
  symbolized with `perf`. Regression:
  `tests/progressive_drain_hang_regression.rs` +
  `fuzz/regression/timeout-progressive-restart-drain-hang`.

### Changed
- **`heuristics::estimate_encode` now models trellis + boundary-rd cost from
  real measurement** (was guessed jpegli throughput). A new
  `examples/jpeg_probe` measures the marginal working set (`VmHWM` delta) +
  wall + user/sys CPU (`/proc/self/stat`, single-thread), swept by
  `scripts/jpeg_resource_calibrate.py` over 4 classes × 512–2048 px ×
  q{50,85} × the 4 trellis×boundary-rd combos
  (`benchmarks/jpeg_resource_2026-06-14.tsv`). Findings: trellis quantization
  **dominates** encode cost (~6.2× time, ~2× working set), boundary-rd adds
  ~1.55×, and the two are **not** multiplicative (both-on ≈ 6.5×). The
  baseline (no-trellis) throughput was also re-centered on measurement
  (16.6/67.6/84.4 MP/s complex/typical/simple) — the prior 8/15/40 were
  ~4–5× too pessimistic. The estimate reads `config.trellis` /
  `config.boundary_rd_mode` and applies the measured multipliers; decode is
  near-free (~0.01 µs/px). New tests pin the trellis (~6×) and boundary-rd
  (~1.55×) factors.
- **4:2:2 (h2v1) fancy chroma upsampler is now SIMD on all arches** — it was
  pure scalar everywhere (confirmed via `cargo asm`: 177 scalar instrs, 0
  vector — LLVM did not autovectorize it). New magetypes-generic interior
  (AVX2 / NEON / wasm128) with the scalar path kept for narrow rows + edges,
  byte-for-byte identical to the old output (`h2v1_generic_matches_scalar_
  bit_exact`, 11 widths × strides × odd out-widths × 10 SIMD tiers).
  Measured (`benchmarks/h2v1_upsample_ab_2026-06-13.txt`): **+18-20% on x86**
  (1.77 vs 1.43 Gpx/s, Ryzen 9 7950X) and **+42-46% on aarch64 NEON**
  (≈800 vs ≈550 Mops/s, Hetzner). Decode output unchanged (full suite 2184
  pass). Affects 4:2:2 (`Subsampling::S422`) decodes only.
- **4:2:0 (h2v2) fancy chroma upsampler is now SIMD on non-x86** — the
  follow-on to the h2v1 work above, for the most common JPEG subsampling.
  x86 keeps its hand-tuned AVX2 kernel; non-x86 (NEON/wasm128) was scalar and
  now uses a `#[magetypes]`-generic interior, byte-for-byte identical to the
  scalar row (`h2v2_row_generic_matches_scalar_bit_exact`, verified on real
  aarch64). Measured on Hetzner arm-big (Neoverse-N1,
  `benchmarks/h2v2_upsample_ab_2026-06-14_arm.txt`): **+15% (≈770 vs 670
  Mops/s)** across 256²/1024²/4096² tiles. Note: the kernel had to load each
  column-sum **once** per chunk — the naive form re-gathered the two input
  planes across the prev/this/next windows (6 scalar-widen gathers/chunk) and
  was measured **−37% (a regression)** on NEON before the fix; the load-once
  form is the +15% win. Serves both the default (`Alternating` bias) and
  libjpeg-compat (`Turbo` bias) fancy 4:2:0 decode paths. Decode output
  unchanged.


### Fixed
- **wasm32 without simd128: IDCT used a slow magetypes scalar emulation**
  (`decode/idct_int.rs`). On a no-simd128 wasm build the `*_auto`/tiered IDCT
  dispatchers fell to the magetypes-generic kernel, which can't select its
  wasm128 tier and degrades to a lane-by-lane scalar emulation of a
  transpose-heavy 8-wide algorithm — measured ~60-68% slower than the
  dedicated scalar i64 IDCT under wasmtime (2026-06-13). The two are
  bit-identical, so the new `idct_int_portable` helper routes no-simd128 wasm
  to the dedicated scalar kernel (and `idct_int_libjpeg_auto` likewise). Pure
  speed routing, cfg-isolated to `wasm32 + not(simd128)` — x86, aarch64, and
  the production simd128 wasm config (where the wasm128 tier wins, +10-37%
  over scalar) are byte-for-byte and behavior unchanged. Part of the ARM/WASM
  SIMD audit; the wasm128 production paths were confirmed optimal (no
  scalar-beats-SIMD cliff like ARM has).


### Removed
- **Dead/broken hand-NEON + hand-WASM DCT scaffolding** (`encode/arm_simd.rs`,
  `encode/wasm_simd.rs`, ~800 lines, both `#[doc(hidden)]` internal, zero
  production callers). A statistically-rigorous A/B on real aarch64
  (`benchmarks/arm_simd_audit_2026-06-13.md` follow-up) proved the hand-NEON
  forward DCT `neon_forward_dct_8x8` does NOT compute a DCT — it emits zeros
  for whole rows (max abs diff 101.49 vs the correct generic) and its own doc
  said "a simplified version… demonstrates the pattern". Its apparent ~100x
  speed was LLVM eliding the garbage. The live ARM/WASM DCT runs the correct
  `#[magetypes]`-generic path, unaffected. Builds verified on
  default + aarch64 + wasm32.


### Documentation
- **h1v2 (4:4:0) upsampler stays scalar on non-x86 — magetypes migration
  measured and REJECTED.** The natural follow-on after h2v1/h2v2 would be a
  `#[magetypes]` generic for the 4:4:0 vertical upsampler too, but on real
  aarch64 (Hetzner arm-big, `benchmarks/h1v2_upsample_ab_2026-06-15_arm.txt`)
  the scalar loop **autovectorizes to 3.4–4.7 Gops/s**, vs ~1.0 for a
  hand-written i16x16 generic (−70 to −78%, a large regression). h1v2 is a
  trivial elementwise map (`(near*3+far+bias)>>2`, no inter-element
  dependency) that LLVM vectorizes optimally; a hand generic can only match
  that, never beat it — unlike h2v1/h2v2, whose prev/this/next window
  dependencies defeat autovec (same "scalar wins on ARM" case as the YCbCr
  plane converters). Locked in with a code comment at the h1v2 dispatcher to
  prevent a future "wire-the-generic" regression. x86 keeps its AVX2 kernel.
- **ARM SIMD audit + real-hardware benchmark** (Hetzner aarch64;
  `benchmarks/arm_simd_audit_2026-06-13.md`): verified every SIMD kernel's
  dispatch and measured NEON-vs-scalar on real silicon. Key finding — the
  scalar i16 YCbCr→RGB plane loop autovectorizes ~2.3× FASTER than the
  magetypes-generic converter on aarch64, so the plane converters' scalar
  fallback is already optimal there (the const-generic dispatch is x86-SIMD
  / ARM-scalar, correct on both). Locked in with code comments at
  `ycbcr_planes_i16_to_rgb_u8`/`_xrgba_u8` to prevent a future
  "wire-the-generic" regression. Default jpegli-12bit IDCT gains only 6-7%
  from NEON (scalar transpose between passes caps it) vs ~4× on x86; NEON
  still wins for the libjpeg-13bit IDCT (1.3-1.5×) and the fused box convert
  (1.6×). Identified ~800 lines of dead hand-NEON/WASM kernels
  (`encode/arm_simd.rs`, `encode/wasm_simd.rs`, zero callers) as deletion
  candidates pending a maintainer call.


### Fixed
- **`recompress` Preserve emit corrupted 16-bit-DQT (web-quality) sources**
  (`recompress/strategies/preserve_emit.rs`): `emit_preserved` errored
  outright on any quant table with a value > 255 ("16-bit quant tables not
  supported"), and the `UniformScale` quant builder hard-clamped tables to
  255 — so IDENTITY-preserving a low-quality source (Q < ~87, which uses
  16-bit DQT) both lost precision AND made the old/new requant ratio ≠ 1.0,
  requantizing coefficients that should pass through (non-identity output).
  Since recompress targets web-quality JPEGs, this broke the common case.
  Fix: `write_dqt` emits Pq=1 (16-bit, big-endian) tables and the frame
  header becomes SOF1 when any value exceeds 255; the `UniformScale` ceiling
  is now `max(255, old[i])` so a source's own 16-bit values pass through
  unchanged while scaled-up 8-bit tables still clamp at 255 (preserving the
  lossy-recompress confidence calibration). New regression test
  `preserve_identity_emit_handles_16bit_dqt` (Q20 4:4:4 + 4:2:0,
  pixel-identical roundtrip with a verified 16-bit table on both sides).
  Note: zenjpeg does NOT do full trellis rewinding in the Preserve path —
  it is a straight lossless coefficient re-emit.


### Changed
- **Turbo YCbCr→RGB converter is now const-generic, not a separate slow
  kernel** — closes the perf gap from the prior commit. The turbo color
  path is now `<const TURBO: bool>` monomorphizations of the existing
  hand-tuned AVX-512/AVX2 kernels (RGB / RGBA-BGRA / fused-box), so it
  reuses their SIMD pack+interleave; `TURBO == false` const-folds to
  byte-identical default code. Measured (Ryzen 9 7950X): turbo went from
  **2.6–4.3× slower** than default to **within ±2% on RGB and ~17% FASTER
  on BGRA/fused-box** (`benchmarks/ycbcr_turbo_2026-06-13_constgeneric.txt`).
  The slow magetypes-generic turbo kernels (`turbo_rgb4`, the three
  `*_turbo_impl`) are deleted; non-x86 turbo uses a turbo-aware scalar
  fallback (matching the pre-existing non-x86 default, which is also
  scalar). Dead `fused_h2v2_hfancy_*` converters removed (no production
  callers). Default (`IdctMethod::Jpegli`) output unchanged (locked hashes
  pass); turbo still byte-exact with mozjpeg (max_diff=0) and bit-exact
  across all SIMD tiers.

### Changed
- **`IdctMethod::Libjpeg` is now byte-for-byte identical to mozjpeg/djpeg**
  (max_diff=0 on decoded RGB). The final residual — YCbCr→RGB — now has a
  libjpeg-turbo-exact 16-bit converter (`build_ycc_rgb_table` constants,
  jdcolor.c) selected by `IdctMethod::Libjpeg`, replacing the ±1 from the
  default zune-style 14-bit math. With the IDCT (islow) and chroma
  upsampling (turbo bias) already bit-exact, all three stages now match, so
  `IdctMethod::Libjpeg` + Triangle equals mozjpeg fancy and + box equals
  mozjpeg box, verified max=0 over sizes×qualities×{4:2:0,4:4:4}
  (`test_idct_method_libjpeg_fancy_matches_mozjpeg_exact`,
  `test_idct_method_libjpeg_matches_mozjpeg`, both `__ffi-tests`). The turbo
  converter is a self-contained magetypes-generic (v3/NEON/WASM/scalar) kernel
  per output layout (RGB / RGBA-BGRA / fused-box), bit-exact with the turbo
  tables across all SIMD tiers (`turbo_converters_match_libjpeg_tables`);
  wired through every decode path (streaming, scanline, coefficient,
  parallel, fused-parallel) via a `turbo` flag the compiler forces every
  call site to pass. The **default** path (`IdctMethod::Jpegli`) keeps its
  hand-tuned AVX-512/AVX2 14-bit converter, byte-identical to before
  (default suite incl. locked hashes unchanged: 2184 pass).

### Changed
- **Libjpeg-exact IDCT is now SIMD** (`idct_int_libjpeg_auto`: AVX2
  intrinsics on x86_64 + magetypes v3/neon/wasm128/scalar tiers): same
  Loeffler islow butterfly and descale rounding in i32 lanes, bit-identical
  to the scalar i64 kernel on every input via two range guards (inputs and
  pass-1 outputs confined to the i16 window; worst-case L1 = 61214, so
  61214 × 32768 + bias < 2^31 — derived and asserted in
  `test_islow_i32_guard_bound_analysis`). Out-of-guard blocks
  (near-adversarial streams only) fall back to the scalar kernel, so decoded
  pixels never change anywhere. Covers `IdctMethod::Libjpeg`, all
  1-component decodes (#154), and the Triangle-mode coefficient path.
  Kernel: 28.1 ns/block vs 111.7 scalar (4.0x, Ryzen 9 7950X dense blocks);
  the #154 gray A/B (10 MP scan, Q85) goes 22.4 → 15.8 ms median vs the
  14.7 ms 12-bit-kernel ceiling — 85% of the regression recovered, the
  rest is the exactness guards. New `idct_kernels` zenbench bench.

### Changed
- **`IdctMethod::Libjpeg` now also selects libjpeg-turbo's fixed h2v2
  fancy-upsampling biases** (`H2v2Bias::Turbo`, monomorphized plane/row
  variants chosen at setup — zero hot-loop cost): the upsampler becomes
  bit-exact with turbo's `h2v2_fancy_upsample` (asserted against an exact
  reference port), so Triangle + Libjpeg decodes match mozjpeg within
  **max_diff <= 1** (was <= 2; the residual is 14-bit vs 16-bit YCbCr→RGB
  rounding — regression gate tightened in issue7_repro). Wired through all
  decode paths (streaming, scanline/strip, coefficient, parallel, fused
  boundary fixups); a new cross-path test pins decode() == scanline_reader()
  under the mode. Default decode output (IdctMethod::Jpegli, alternating
  bias) is byte-identical to before.

### Added
- **Direct YCbCr plane parity proof vs mozjpeg**
  (`test_libjpeg_idct_ycbcr_planes_match_mozjpeg`, `__ffi-tests`): mozjpeg's
  actual runtime post-upsample planes (`JCS_YCbCr`) pushed through
  zenjpeg's 14-bit converter reproduce `IdctMethod::Libjpeg` decoded RGB
  byte-for-byte across 4 sizes × 3 qualities × {4:2:0, 4:4:4} — zenjpeg's
  internal post-IDCT/post-upsample Y/Cb/Cr planes are bit-identical to
  libjpeg-turbo's, and the only RGB-level residual is the documented
  YCbCr→RGB table rounding.

### Documentation
- **Chroma upsampling parity vs libjpeg-turbo pinned and corrected**: the
  Triangle h2v2 (4:2:0) upsampler is NOT bit-identical to turbo's fancy
  upsampling as its comments claimed — turbo uses fixed +8/+7 rounding
  biases on both rows of a pair (jdsample.c) while zenjpeg row-alternates
  to +7/+8 (like turbo's own h1v2). Measured: even output rows bit-exact,
  ~6.7% of odd-row pixels differ by ±1, both schemes equally accurate vs
  the exact filter (max err 0.5, ~zero bias; checkerboard vs column-stripe
  half-case structure). h2v1 (4:2:2) and h1v2 (4:4:0) ARE bit-identical to
  turbo. The integer YCbCr→RGB (zune-style 14-bit) vs turbo's 16-bit
  tables over the full 256³ cube: R identical, G 0.179% / B 0.104% differ
  by ±1 — this is the residual in "box + Libjpeg IDCT matches mozjpeg
  within max=1". New pinning tests: `h2v2_triangle_vs_libjpeg_turbo_reference`,
  `h2v2_bias_schemes_accuracy_vs_ideal` (upsample.rs),
  `int_ycbcr_vs_libjpeg_turbo_tables` (ycbcr.rs); docs updated with the
  measured ±1 decomposition.

### Added
- **SCALAR sweep-ladder densification** (`__expert` planner; dense-sweep
  program / `zenpicker-train --scalar-axes`, zenmetrics `docs/PLAN_SWEEPS.md`
  §5): aq_coupling.scale gains the ±2 mid-points and the +8 bound endpoint —
  symmetric 6-point ladder {−8, −4, −2, +2, +4, +8}, every step clamped ±1.0;
  coupling.exponent gains the 0.5 probe, completing the historical
  {0.5, 1, 2} grid. New values ride the existing `cpl<scale>[e<exp>]cl<adj>`
  id grammar and fingerprint. The doc-flagged "jpegli AQ strength" scalar is
  recorded as **axis blocked on encoder knob** (`aq_enabled` is bool-only; the
  AQ-field shape bakes `K_AC_QUANT × dampen` internally — a continuous axis
  needs an ExpertConfig strength knob first). Harness re-run ALL HARD CHECKS
  PASSED; all new steps live at 90 % diff with monotone byte direction
  (`benchmarks/sweep_validate_2026-06-12.tsv`).

### Removed
- Dead `custom_aq_map` chain: the `ComputedConfig`/`StreamingEncoderBuilder`
  field, the pub(crate) `aq_map()` setter, and the orphaned
  `get_aq_map_or_compute` / `create_trellis_ctx` helpers — all fed only
  the never-integrated XYB block-based encoding path; nothing public
  could populate the map and setting it changed no encode. Caller-supplied
  AQ maps are re-scoped as a `CustomMapController` behind the #113
  `AqController` hook, gated on a validation experiment (#147). Closes #76.


### Fixed
- `decoder::Decoder` with `auto_orient(true)` — the DEFAULT — produced
  massively wrong pixels on EXIF-rotated 4:2:0 JPEGs (#149): on a
  4000x3000 EXIF-Orientation-6 phone photo, 88% of bytes differed from
  decode-upright-then-rotate (max abs diff 252, ~36% of pixels >2% off
  vs an ImageMagick reference). Root cause: orientation was applied as
  a DCT-coefficient-domain transform on the MCU-padded coefficient
  grid, but the post-transform crop was computed at 8-px granularity
  while 4:2:0 storage pads to 16-px MCUs — Rotate90 moved the bottom
  padding block row to the left edge uncropped, shifting the whole
  image 8 px and desyncing chroma. Dimension-swapping transforms also
  forced the f32 IDCT, diverging from the default integer-IDCT decode
  by up to 9 channel steps even on MCU-aligned images. Decode-time
  orientation (EXIF auto-orient and explicit `transform()`) is now a
  lossless pixel-domain permutation of the upright decode — output is
  byte-identical to `auto_orient(false)` + an external orientation
  bake for all 8 orientations, all subsampling modes, and all sizes,
  and the upright decode keeps the streaming fast path. The scanline
  reader routes transforms through the same path; DCT-domain
  transforms remain for the lossless re-encode pipeline
  (`lossless::transform`). New regression matrix:
  `tests/bundled/auto_orient_pixel_parity.rs`.
- `container::mpf::parse_mpf` now tolerates two in-the-wild MP Index
  quirks from the same writer family (#148): the MPFVersion UNDEFINED×4
  value written out of line (value field zeroed, ASCII `"0100"` spliced
  after the 12-byte entry, shifting later entries by 4 — the IFD walk is
  now cursor-based and resyncs past it) and an IFD-relative B002 MPEntry
  offset (declared 0x2E for entries physically at TIFF+0x36 — the
  declared offset is sanity-checked by the first entry's image size and
  falls back to the structural post-next-IFD position). Previously such
  files — including the 7.6 KB `ultrahdr_sample.jpg` fixture in
  zencodecs/ultrahdr-rs — errored with "MPF declares zero images" despite
  carrying a fully self-consistent two-image index, which aborted
  `ultrahdr_rs::Decoder` entirely (imazen/ultrahdr#26) and silently cost
  consumers the HDR rendition. Conformant files are byte-for-byte
  unaffected (existing roundtrip + property tests unchanged).
- zencodec decode-path `ImageInfo` now populates
  `source_color.bit_depth` (SOF sample precision) and `channel_count` —
  previously only the probe path set them, so precision-aware callers
  saw `None` after a full decode (#146).

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
- Playbook adoption table finalized: all five zen codecs (jpeg, avif,
  jxl, webp, png) landed across all eight checklist rows, with both
  zenmetrics execution models wired (chunk mode + content-addressed
  job system) through the one `{cell, fp, plan}` identity contract.
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
