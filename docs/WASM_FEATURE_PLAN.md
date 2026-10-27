# WASM decoder feature measurements

The first measurement favors one combined optional capability for arithmetic
JPEG, 12-bit DCT JPEG and predictive lossless JPEG. Keep the implementations in
separate internal modules so a future split does not require another algorithm
refactor. XYB decoding remains part of the core decoder, alongside ordinary
8-bit sequential and progressive Huffman JPEG. No permanent feature gates or
public API changes were introduced by this experiment.

## Measured size

Source: `c7ef43061b01a3deb61e7c47fbcf799f14a8f3b6`.
Rust 1.99.0, LLVM 23.1.1, wasm32-wasip1, opt-level `s`, thin LTO,
one codegen unit, stripped symbols, no debug information. The probe reads an
arbitrary JPEG from stdin and writes dimensions and the decoded u8 raster.
It depends on zenjpeg with default features disabled. All sizes are bytes;
gzip uses Python's default compression level and a fixed zero timestamp.

| Build | Scalar raw | Scalar gzip | SIMD128 raw | SIMD128 gzip |
|---|---:|---:|---:|---:|
| Full decoder | 797498 | 294488 | 763521 | 281659 |
| Exclude arithmetic | 779945 | 288138 | 746371 | 274595 |
| Exclude extended precision/lossless | 780974 | 288625 | 747406 | 275157 |
| Exclude both | 763405 | 281774 | 730240 | 268494 |
| Combined saving | 34093 | 12714 | 33281 | 13165 |

Each of the eight configurations was compiled twice, forcing recompilation
with incremental compilation disabled. Each pair produced identical SHA256.
The SIMD128 builds are smaller in this probe; throughput was not measured.

For each target configuration, 16 fixtures produced 38 byte-exact retained
raster comparisons and 26 expected explicit refusals across the four variants.
These include three XYB fixtures, which decode identically in every variant.
Every retained raster also agrees between the scalar and SIMD128 builds.
Hashes include little-endian width and height followed by the entire u8 raster.
This checks preservation relative to the full build; it is not an independent
oracle qualification of every decoder format. Existing native fixture tests
separately lock libjpeg-turbo reference rasters.

[Raw measurements and raster hashes](../benchmarks/wasm_decoder_size_2026-10-27.json).

## Interpretation and limits

The exclusions replace arithmetic scan entry bodies and predictive reader
construction/row decoding with errors, reject non-8-bit frames, and make the
12-bit dispatch conditions compile-time false. This measures actual linked
output after removing reachability; it is not the final module-level design.
Shared component readers, ordinary precision kernels and XYB stay intact.
No inference is made from source line counts or library archive sizes.

The combined saving is modest enough that two public switches are not yet
justified by this application. Group the capability initially, then split only
if consumers or measurements of other API surfaces justify it. The whole-crate
refactor should move exclusive arithmetic/predictive/precision code behind
module boundaries and keep dispatch/refusal logic small. Feature-off builds
must reject omitted formats without weakening parser validation or silently
lowering precision. Existing default-features=false behavior needs an explicit
compatibility decision before landing permanent gates.

This is a WASI executable, not a browser wasm-bindgen artifact. WASI I/O/runtime
code is included in every variant. Only the default collected u8 decode API is
export-reachable; native u16, other row APIs, encoding, ICC conversion and other
optional integrations can change the result. Browser exports, encoder-only
builds and other optimization profiles remain unmeasured. Do not extrapolate
these byte savings to those configurations.

## Reproduction

Create a disposable secondary jj workspace at the source revision. From the
primary checkout, run `just wasm-size-experiment WORKSPACE NEW_OUTPUT` and
`just wasm-size-experiment-simd WORKSPACE OTHER_NEW_OUTPUT` through the resource
wrapper. Then run `just wasm-size-raster-check OUTPUT` for each output directory.
The scripts refuse the primary checkout and existing measurement output.
Use disk-backed TMPDIR and retain the complete build logs. Forget and remove
the disposable workspace after collecting results; binaries stay outside git.

Measured wrapper summaries (validation process, not decoder memory claims):

- Scalar builds: `run-heavy: done rc=0 75s | peak-RSS 0.53GiB | min-avail 23781MiB | peak-load 1.05`.
- SIMD128 builds: `run-heavy: done rc=0 74s | peak-RSS 0.53GiB | min-avail 23714MiB | peak-load 1.29`.

This experiment does not complete whole-library branch coverage. The previously
recorded production baseline remains 7692/9844 outcomes; research execution and
additional tests do not establish a new complete coverage percentage.

## Implemented capability (2026-10-27)

Commit `133801af` adds default-enabled `extended-jpeg`, grouping arithmetic,
12-bit DCT and predictive lossless decoding. `default-features = false` now
selects core decoding; explicitly enable `extended-jpeg` to retain these formats.
XYB remains available in both configurations. Arithmetic scan and predictive
reader implementations have module gates; shared precision dispatch uses one
compile-time capability predicate. Shared metadata traversal stays ungated.

The real feature was measured twice with the same SIMD128 WASI probe/profile:

| Build | Raw bytes | Gzip bytes | Code section | Data section | Other |
|---|---:|---:|---:|---:|---:|
| Full | 763556 | 281689 | 672455 | 89110 | 1991 |
| Core | 728372 | 268234 | 640802 | 85606 | 1964 |
| Saving | 35184 | 13455 | 31653 | 3504 | 27 |

Each repeated pair has identical SHA256. Section sizes include section framing;
code includes runtime/dependency functions and data includes their embedded data.
These are not per-subsystem attribution measurements. Sixteen fixtures provide
21 retained raster comparisons and 11 explicit capability refusals; all three
XYB rasters remain identical. The native test harness separately checks all 47
libjpeg-turbo fixture contracts, including precise native samples and strides.

[Actual capability measurements and hashes](../benchmarks/wasm_decoder_capability_2026-10-27.json).
Reproduce using `just wasm-capability-size WORKSPACE NEW_OUTPUT` through
`run-heavy`; the workspace must contain the implemented capability. Scalar
measurements above remain prototype results, not measurements of this commit.

Validation wrapper: `run-heavy: done rc=0 39s | peak-RSS 0.53GiB |
min-avail 23665MiB | peak-load 0.94`. This is build-process memory, not decoder
memory usage.
