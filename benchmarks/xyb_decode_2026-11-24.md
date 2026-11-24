# XYB decode: exact transfer and fewer copies (2026-11-24)

Missing: direct C++ jpegli parity (reference build absent), NEON execution and
AVX-512/Zen 5 measurements. No new AVX-512 tier was introduced. The existing
inverse matrix retains its backend-dependent fused arithmetic; the exactness
claim below covers the new transfer versus its scalar oracle and preservation
of each tier's old inverse output, rather than universal decoder ISA identity.
The crate currently requires std; no no_std claim is made.

On r5900xt (Ryzen 9 5900XT, AVX2), one-thread resident-input XYB decoding improved
1.94–2.11× on the retained 1024×1024 RGB crop. The unchanged YCbCr baseline
controls moved 6–7%; normalized to the 4:2:0 baseline control, XYB improved
1.83–1.99×. These are exploratory single-image measurements, without an
across-build confidence interval or a corpus-wide claim.

## Changes

- Load contiguous eight-lane plane chunks.
- Evaluate the existing linear-srgb OETF four f64 lanes at a time, preserving
  its f32 sqrt, promoted-f32 constants, fused Horner graph, C0-continuous toe,
  f32 narrowing, clamps, signed zero and u8 half-away rounding.
- Round and narrow u8 lanes together, then interleave into a fixed 24-byte block.
- Skip bias statistics when the output never applies them (including XYB).
- Move full-resolution planes during upsampling and compact padded rows in
  place, avoiding another allocation, zero fill and full-image copy.

The IMI quartic was inspected but not adopted: it uses a different transfer
curve/rounding graph. The f32 IDCT already uses AVX2. The existing coefficient
buffering route remains; this work does not implement a streaming XYB rewrite.
No public API, encoder behavior, test expectation or threshold changed.

## Measurements

| Mode | Bytes | Before MPx/s | After MPx/s | Speedup | Peak RSS KiB before → after | run-heavy peak GiB before → after |
|---|---:|---:|---:|---:|---:|---:|
| xyb-bquarter-baseline | 62,043 | 34.26 | 66.55 | 1.94× | 33,764 → 24,664 | 0.03 → 0.02 |
| xyb-bquarter-progressive | 57,154 | 33.16 | 65.08 | 1.96× | 33,724 → 25,284 | 0.03 → 0.02 |
| xyb-full-baseline | 69,494 | 30.29 | 63.93 | 2.11× | 38,452 → 25,648 | 0.04 → 0.02 |
| xyb-full-progressive | 64,221 | 29.73 | 62.06 | 2.09× | 38,188 → 26,316 | 0.04 → 0.03 |
| ycbcr-420-baseline | 58,854 | 637.58 | 676.31 | 1.06× | 7,876 → 7,736 | 0.01 → 0.01 |
| ycbcr-444-baseline | 71,642 | 632.60 | 677.12 | 1.07× | 7,940 → 7,700 | 0.01 → 0.01 |
| ycbcr-420-progressive | 57,651 | 283.59 | 318.28 | 1.12× | 10,972 → 10,896 | 0.01 → 0.01 |

Before: archived main `311e39ba9a89096912839e35efd51b8ca3370946`, after: this
optimization series (`385eb776`, `0eb37bef`, `bc3a3046`). Release, rustc 1.99.1, default compiler CPU target,
archmage/magetypes 0.9.33, pinned linear-srgb c103c1ce, no moxcms/parallel in the
measurement executable, `num_threads(1)`. Same hashed JPEG input in each A/B.
Zenbench 0.1.11 interleaved seven modes, 30 rounds, fresh output per call;
input-file reads precede timing. Builds were measured sequentially on a shared
host. Reference throughput changes are not credited as YCbCr optimizations.

RSS is a separate 40-decode resident-input process per cell. KiB comes from
GNU time -v; the coarser GiB column is run-heavy's sampled process-tree peak.
Every heavy command used a 24 GiB/24-job cap. Combined zenbench process-tree
peaks were 0.17 GiB before and 0.16 GiB after. Initial profiling (`perf record
-F 999 -g`, 200 decodes) attributed 38% of cycles to XYB→sRGB, 10% to the
coefficient-to-plane loop, and 3% to upsampling; page faults also contributed.
The intermediate reuse build reduced sampled cycles from 30.13B to 18.78B
for the same 200 BQuarter baseline decodes. These are whole-process samples.

The six original fixtures were encoded with main before changes. The later
YCbCr progressive control was losslessly restructured from the 4:2:0 baseline,
retaining its coefficients. The original crop scratch directory disappeared
during the run; retained encoded fixtures are the reproduction inputs. The
brief's older revision/timings are context, not the A/B baseline above.

## Verification

- Unit/bundled/color-path gates: 1,215 / 1,026 / 46 passed; existing 1 / 99 / 0
  ignores unchanged. This gate followed the transfer and plane changes.
- After packing: 34 XYB unit tests; parallel XYB gates 54 unit, 47 bundled and
  3 color-path tests passed. This includes Full/BQuarter, baseline/progressive,
  odd edges, one-shot and scanline reads, u8/f32 and existing CMS tests.
- Exact transfer tests: approximately 1.1 million input values, the C0 toe,
  every actual scalar u8 quantization boundary and neighbors, plus clamps and
  nonfinite inputs. AVX2 and scalar match the existing scalar dependency's
  float bits and u8 result. Full inverse kernels match their old implementation
  per tier, including tails and short/odd lengths.
- The same production-source kernel tests executed under wasmtime with SIMD128
  and without it: 32 passed each. No full WASM decoder-speed claim is made.
- Seven complete fixtures × u8/f32 = 14/14 exact byte matches to archived main.
  ImageMagick independently read all seven JPEGs to completion.
- Strict library Clippy, scoped formatting and diff checks passed. Doctests:
  15 passed, 100 existing ignores. Broad tests peaked at 2.65 GiB; parallel
  validation at 1.71 GiB; WASM kernel build/test at 0.38 GiB.

## Reproduction and artifacts

`zenjpeg/examples/profile_xyb_decode.rs` provides prepare, progressive, run,
dump and bench modes. Run its resident loop under perf; use bench for
zenbench timing (`--format=csv`). `just xyb-kernel-check` runs the production
kernels separately from native oracle dependencies;
`just xyb-kernel-check-wasm` executes them with WASI SIMD128 using wasmtime.
Use isolated CARGO_TARGET_DIR paths for builds.

Host r5900xt artifacts: `/mnt/v/output/zenjpeg/xyb-speed-20261124/` contains
fixtures, pre/post pixels, source/binary/input hashes, raw logs, RSS cells and
results.json. Local reproduction executables/manifests and the archived main
source remain under `~/tmp/zenjpeg-xyb-speed/`. Zenbench's ambient git label is
not the archived binary's source identity; the baseline SHA and binary hashes
above govern the A/B identity.
