# zenjpeg decode profile notes — 2026-04-15

Profile source: `perf record -F 999 --call-graph fp` of
`profile_decode_only decode /tmp/zenjpeg_profile_1024x1024.jpg 500` under
`RAYON_NUM_THREADS=1`. Binary: `target/release/examples/profile_decode_only`
compiled with default release + debuginfo, no `-C target-cpu=native`.

Fixture: 1024×1024 baseline 4:2:0 Q85, ~399 KB, encoded by
`EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false)`.

Each decode: `Decoder::new().output_format(PixelFormat::Rgb).decode(..)`.
This is **RGB output** — the BGRA path adds a separate swizzle tail
(see BENCH-AUDIT.md finding #1; that's separate from this profile).

## Top 5 cost centres (ST RGB decode)

| % cycles | Symbol | File | Role |
|---------:|--------|------|------|
| **60.9%** | `JpegParser::decode_baseline_streaming` | `zenjpeg/src/decode/parser/scan.rs:853` | Fused entropy + IDCT + upsample + YCbCr→RGB |
| **19.5%** | `idct_int::avx2::__arcane_idct_int_avx2` | `zenjpeg/src/decode/idct_int.rs` (AVX2 kernel) | 8×8 integer IDCT per block (inlined into caller, but charged its own region) |
|  **8.7%** | `color::ycbcr::__arcane_ycbcr_planes_i16_to_rgb_u8_avx512` | `zenjpeg/src/color/ycbcr.rs:1408` | YCbCr i16 planes → RGB u8 |
|  **6.4%** | `decode::upsample::__arcane_upsample_h2v2_libjpeg_row_avx2` | `zenjpeg/src/decode/upsample.rs` | 4:2:0 fancy upsample per row |
|  **1.8%** | `__memset_avx512_unaligned_erms` (libc) | n/a | Zeroing newly-allocated Y/chroma/RGB buffers |
|  **1.4%** | `idct_int::idct_int_tiered` | `zenjpeg/src/decode/idct_int.rs:1218` | Per-block dispatcher |

Other, under 1%: ac_huffman fast paths, `decode_block_into`, Rayon init.

## Findings + proposed fixes

Rank order is "likely impact × implementation tractability".

### 1. Streaming decode produces RGB only; BGRA is written via a second full-buffer swizzle pass

- File: `zenjpeg/src/decode/parser/scan.rs:853` (`decode_baseline_streaming`)
  calls `ycbcr_planes_i16_to_rgb_u8` directly into a 3 bpp `Vec<u8>`.
  Then `output.rs:1513-1519` in `to_pixels_into` sees `streaming_rgb` is
  populated and calls `reformat_rgb_into(rgb, BGRA, ..., dst)` —
  `garb::bytes::rgb_to_bgra` over the entire frame.
- Measured cost: +9% at 1024² ST, +47% at 4096² ST (cache-resident vs
  cache-miss). See `zenjpeg/examples/profile_bgra_vs_rgb.rs`:
  - RGB 1024²: 315 Mpx/s
  - BGRA 1024²: 290 Mpx/s
  - RGB 4096²: 198 Mpx/s
  - BGRA 4096²: 105 Mpx/s
- **Proposed fix:** add a fused `ycbcr_planes_i16_to_bgra_u8` kernel
  (mirror of the existing `_to_rgb_u8` kernel with lane shuffle + alpha
  splat) and thread a `PixelLayout` parameter through
  `decode_baseline_streaming` + `decode_baseline_streaming_rgb` so the
  final stage writes BGRA directly into the caller's buffer. At 4096²
  this should recover all ~60 ms/decode that swizzle currently costs,
  closing the user-reported imageflow bench gap entirely.
- **Cheaper alternative (days vs weeks):** make `to_pixels_into` route
  BGRA requests to `to_pixels_fast_i16_subsampled_into` (which *does*
  support direct-BGRA output via `FastDst::Bgra(dst)`) instead of
  preferring `streaming_rgb`. The direct fast path already exists at
  `zenjpeg/src/decode/parser/output.rs:1524-1538`; it's just unreachable
  on the hot `Auto` mode path. Toggle the priority or add a
  `prefer_direct_dst` decode hint. Downside: the fast i16 path is not
  fully streaming (has an intermediate i16 coefficient plane), so peak
  memory is higher. For imageflow-class callers (already allocate a
  bitmap buffer), this is the right tradeoff.

### 2. `decode_baseline_streaming` allocates Y/chroma strips with `try_alloc_maybeuninit` (zeroing memset)

- File: `zenjpeg/src/decode/parser/scan.rs:996-1062`. Each call allocates
  multiple `Vec<i16>` / `Vec<u8>` via `try_alloc_maybeuninit`, which
  under the hood does `try_reserve_exact` + `resize(n, T::default())`.
  The resize triggers a serial memset to zero — 1.83% in the profile is
  `__memset_avx512_unaligned_erms`. All these buffers are fully
  overwritten during decode, the zeroing is waste.
- File: `zenjpeg/src/foundation/alloc.rs:593`. The existing
  `try_alloc_zeroed_bytes` (line 620) uses `calloc` for `Vec<u8>` — zero
  page mmap, near-free. The i16 strip buffers don't have an equivalent;
  they go through `resize(n, 0i16)` which is a serial memset.
- **Proposed fix:** add `try_alloc_uninit_i16(count, context) -> Vec<MaybeUninit<i16>>`
  (fallible) and use it for Y strip A/B + chroma strip A/B + upsampled
  chroma buffers. Keep RGB output via `try_alloc_zeroed_bytes` (which
  stays as `calloc`). Track per-row which bytes are written; on the
  final row of the image the boundary region needs to be zero-filled
  explicitly if the image isn't a multiple of 16 rows. Expected gain
  ~1.5% for typical sizes, possibly more on 4K+ where the buffers are
  larger and the memset works against streaming stores.
- Cheaper alternative: cache the strip buffers on the `JpegParser`
  itself across calls (reuse allocation). Only valid if the parser is
  long-lived; imageflow creates one per decode.

### 3. IDCT tier dispatch `summon()` per block

- File: `zenjpeg/src/decode/idct_int.rs:1218` (`idct_int_tiered`). Every
  call to the non-DC branch runs `X64V3Token::summon()` — an atomic
  `Relaxed` load from the archmage CPU-feature cache. Cheap per call
  (~3–5 cycles) but there are 24576 blocks in a 1024² 4:2:0 image.
  Measured bucket: 1.4% (`idct_int_tiered`) but some of that is real
  work, not just token lookup.
- **Proposed fix:** follow the archmage "token at entry point" pattern.
  Summon `X64V3Token` once in `decode_baseline_streaming`'s outer frame,
  pass it through `decode_mcu_row`'s `idct_fn` parameter as a closure-
  captured `Option<X64V3Token>`, and have the tiered dispatcher accept
  a pre-summoned token. Saves ~0.5–1% and aligns with archmage guidance.
- Note: similarly, `to_pixels_fast_i16_*` paths should be audited —
  they likely summon per MCU row.

### 4. Per-call `Decoder::new()` in the imageflow bench pays non-trivial setup

- File: `imageflow_core/src/codecs/zen_decoder.rs:605-619` creates a new
  `job = JpegDecoderConfig::new().job()` per decode and constructs an
  animation frame decoder (for the non-streaming push-decoder path).
  For non-animated formats the `always_use_frame_decoder()` gate at line
  596 should keep us out of this branch — confirm this is actually the
  case for JPEG (it should be).
- More relevant: `bench_codecs.rs:170-201` calls `Context::create()`
  + `add_input_vector` + `execute_1` per iteration. That's the graph-
  engine + memory-manager + codec-registry setup cost. On a 256² image
  this is a **big** fraction of wall time — the 256² bench number
  (39.7 Mpx/s zen vs 45.1 Mpx/s moz) is mostly imageflow context setup
  with a small decoder tail. Apples-to-apples still, but an imperfect
  proxy for decoder micro-performance.
- **Proposed fix:** no change to zenjpeg. If the imageflow team wants
  tighter decoder-only numbers, the bench should reuse a `Context`
  across iterations (where supported) or move to the per-crate
  `decode_mozjpeg` / `decode_zenbench` benches which time decode only.

### 5. zenjpeg does NOT currently have restart-marker support in the encoder for progressive mode (deliberate default)

- **Not a perf bug**, but noted for the restart-markers test:
  zenjpeg's encoder DOES support restart markers natively in baseline
  mode via `EncoderConfig::restart_mcu_rows(rows)`, with a default of 4
  MCU rows. Progressive mode suppresses RST markers by design (see
  `encode/byte_encoders.rs:104-115`); this is correct — progressive
  decode can't use RST for parallelism — and the `force_restart_markers`
  flag exists for callers who need RST in progressive for interop.
- `resolve_restart_rows` (`encode/config.rs:182`) bumps the user's
  requested MCU-row count up when the estimated file size (modeled at
  0.5 bpp) can't afford the DRI+RSTn overhead. For a 256² Q85 image
  that threshold pushes RST to zero. This means the test fixture must
  be at least ~512² to reliably observe markers in output.
  **Not a bug**, but worth documenting — the restart-markers test
  uses 512² for exactly this reason.

## What the profile does NOT show

- **Branch mispredictions**: `perf record` with `cycles:Pu` doesn't
  break this out. The fast AC and DC Huffman paths in
  `entropy/decoder.rs:724` (`decode_block_into`) use `fast_lookup`
  arrays indexed by 9-bit window values + a `fast_ac_array` lookup.
  These are already the idiomatic libjpeg-style fast path. Running
  `perf stat -e branch-misses,branches` on the same binary would tell
  us whether the branches cost us anything we're not already pricing
  in. Not done in this run.
- **Instruction counts via callgrind**: would give a deterministic
  per-function cost free of thermal/turbo bias. `valgrind` can't
  handle AVX-512, so the `ycbcr_planes_i16_to_rgb_u8_avx512` path
  (8.7%) would have to be disabled. Not done in this run.
- **Heap allocator pressure**: not profiled. `heaptrack` would show
  peak RSS and allocation churn per decode. Likely target for a future
  session since #2 (buffer zeroing) is a weak signal without heaptrack.

## Apples-to-apples sanity check (decode_mozjpeg bench)

ST @ 1024² Q85 baseline, **RGB output**, via `decode_mozjpeg` bench:
- mozjpeg (libjpeg-turbo NASM SIMD): 3.82 ms, 274 Mpx/s
- zenjpeg (`Decoder::new().output_format(Rgb)`): 3.59 ms, **292 Mpx/s (+6%)**
- zenjpeg + box filter: 3.42 ms, 306 Mpx/s (+12%)

**zenjpeg's core decode is already faster than mozjpeg at RGB output.**
The imageflow bench gap is the swizzle tail, not the decoder. Fix #1
should close it.
