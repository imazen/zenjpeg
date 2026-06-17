# Changelog

All notable changes to zenjpeg are documented here. Earlier history
(pre-2026-06) lives in `git log` and `docs/TUNING_HISTORY.md`.

## [Unreleased]

### Added
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
### Fixed
- **Crate-root rustdoc decoder example now compiles and runs** (closes #155, item
  2). The top-of-page `lib.rs` decoder snippet showed a stale 1-arg
  `Decoder::new().decode(&jpeg_data)?` plus `image.pixels() -> &[u8]`; the real
  signature is `.decode(data, stop)` returning a `DecodeResult` whose bytes come
  from `.pixels_u8()`. Replaced the `rust,ignore` block with a real, compiling
  doctest (encode a tiny image, decode it round-trip) and added a second doctest
  showing `match err.kind() { ErrorKind::… }` for error classification.
### Changed
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
