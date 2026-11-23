# JPEG dispatch and linked size, 2026-11-23

The retained change saves 60,144 native `.text` bytes and 61,095 stripped
WebAssembly bytes in the same linked encode/decode consumer. Native and WASI
outputs are exact before/after: 27/27 witness files on each target. Throughput
is close to baseline, with the complete per-cell results below; these exploratory
runs do not establish formal zero-regression or universal non-inferiority.

## Source and implementation

Baseline codec: `7628fd46`, with the fixed benchmark/consumer harness from
`ffe26d8c`. Retained implementation: `07c0dd0f`. Codec source tree IDs:
`5c9d7f3596d1d79dad9bc71db8beead94d8e4498` before and
`6e0a77675f11d0da241199eff770b42d2d63be9e` after.
The lockfile SHA-256 stayed
`f7d8582938f239f447e1e1bf5498e44dc357a3596fce95539393a36b3557409a`.

- Keep ordinary entropy block decoding inline and share the optional
  reconstruction-metadata block implementation through one non-inlined helper.
  Metadata bookkeeping, coefficient order and overflow/refusal behavior are
  unchanged. A whole-body outline was rejected after six-process repeats showed
  dense baseline noise decode **6.803% slower** in `huffman_decode`.
- Replace 90 Huffman-table match arms with indexed static descriptors and one
  constructor. All 360 raw tables retain their original bits and symbol order.
  The original legal-symbol coverage test remains; a new 90-set fingerprint guard
  was watched fail after swapping two legal DC symbols, then restored and passed.
- Cache the existing scalar/V3/NEON linear conversion tier once per
  `StripProcessor`; strip batches call the original arithmetic kernels and retain
  their tails. Standalone wrappers remain available to tests and `__test-utils`.
  No public API, Cargo default feature, dependency pin or quality floor changed.
- The optional audit did not change `zenyuv` (`std` is an empty feature), IDCT,
  AQ row dispatch or XYB dispatch. Existing strided-plane signatures and the
  active API work remain untouched. A one-pixel bench-utils test uses an array
  instead of a temporary Vec to satisfy current strict Clippy.

## Method and bounded witnesses

All i134 measurements ran through `remote_bench.sh i134 zenjpeg`, serialized
under its shared lock, at nice 19 and `taskset -c 2`. Rust 1.99.0
(`b940084d7`, LLVM 23.1.1), release/bench optimization with debug information
disabled, empty native RUSTFLAGS; no `target-cpu=native`. Wasm used
`-C target-feature=+simd128` with the same consumer and default features.
No metric maps or generated image fixtures were needed.

The final comparison is three fixed ABBA blocks per executable: six fresh
processes per revision for each of `huffman_decode` and `leaf_dispatch` (24
processes total), 30 rounds per cell. Each process statistic is the median of all
recorded compensated batch times divided by actual iteration count; the table
reports the median of six process statistics. No samples/processes were dropped.
Throughput delta is `before_ns / after_ns - 1`, so positive means faster.
All raw packets record effective thread affinity `[2]`. Zenbench 0.1.11 labels
this `UNVERIFIED_UNRESTRICTED`: no delegated cpuset, frequency verification or
continuous quietness proof. These are exploratory repetitions, not formal
confirmation-mode inference.

`huffman_decode`'s explicit `ZENJPEG_BENCH_FIXTURES=leaf` mode and the full decode
consumer use the same six committed photographic/synthetic JPEGs in
`internal/native-bench/evidence`: flower baseline 420/444, flower progressive,
256-square noise baseline/progressive and 17x33 DC-only baseline. Both measure
full public decoding, not an isolated entropy kernel. No corpus discovery or
missing-file skip occurs in this mode.

Encode covers a 512x384 crop independently decoded from the flower fixture:
sRGB8 YCbCr 420/444 and full XYB, plus deterministic odd-sized 257x193 linear
RGB16, gray16, RGBf32 and grayf32 data (including zero, negative and HDR float
values). All use quality 85, baseline output and fixed Huffman tables. The
independent `jpeg-decoder` 0.3.2 accepts every encoded witness before timing.
The consumer records each encoded stream, our decoded raster and the independent
oracle raster, alongside the six direct decode rasters: 27 files per target.

## Throughput

| Consumer / cell | Before µs | After µs | Throughput change |
|---|---:|---:|---:|
| huffdec/leaf/dc_odd | 7.343 | 7.488 | -1.928% |
| huffdec/leaf/noise420 | 207.210 | 207.621 | -0.198% |
| huffdec/leaf/noise_progressive | 434.001 | 434.048 | -0.011% |
| huffdec/leaf/photo420 | 13474.816 | 13532.762 | -0.428% |
| huffdec/leaf/photo444 | 15379.730 | 15313.033 | +0.436% |
| huffdec/leaf/photo_progressive | 21438.590 | 21541.310 | -0.477% |
| leaf/decode/dc_odd | 7.373 | 7.381 | -0.118% |
| leaf/decode/noise420 | 219.823 | 207.044 | +6.172% |
| leaf/decode/noise_progressive | 434.369 | 433.826 | +0.125% |
| leaf/decode/photo420 | 13576.084 | 13496.162 | +0.592% |
| leaf/decode/photo444 | 15320.364 | 15303.513 | +0.110% |
| leaf/decode/photo_progressive | 22018.437 | 21923.794 | +0.432% |
| leaf/encode/linear_gray16 | 603.276 | 604.409 | -0.188% |
| leaf/encode/linear_grayf32 | 720.793 | 723.807 | -0.416% |
| leaf/encode/linear_rgb16 | 1364.965 | 1361.028 | +0.289% |
| leaf/encode/linear_rgbf32 | 1387.845 | 1379.335 | +0.617% |
| leaf/encode/srgb8_fixed420 | 1109.701 | 1108.990 | +0.064% |
| leaf/encode/srgb8_fixed444 | 1595.737 | 1590.891 | +0.305% |
| leaf/encode/srgb8_fixed_xyb | 1780.481 | 1784.441 | -0.222% |

The full consumer's photographic decode medians improve 0.110–0.592%; baseline
noise improves 6.172%, while the separate Huffman consumer is effectively flat
on that noise fixture (-0.198%). Tiny DC decode is -0.118% in the full consumer
and -1.928% in the separate consumer; the latter has overlapping process ranges
(before 7.24–7.49 µs, after 7.33–8.20 µs). Photographic cells in the Huffman
consumer range from -0.477% to +0.436%. Encode cells range from -0.416% to
+0.617%. Thus the rejected 6.8% decode regression is removed; small negative
medians remain visible and are not reclassified as measured speedups. The data
does not prove a literally zero slowdown on every cell.

## Linked bytes and attribution

All sizes are measured, not projections to other consumers or targets.

| Artifact / section | Before bytes | After bytes | Delta bytes |
|---|---:|---:|---:|
| Native `.text` | 3,062,305 | 3,002,161 | -60,144 |
| Native stripped file | 5,694,312 | 5,650,144 | -44,168 |
| Wasm unstripped module | 3,971,454 | 3,910,229 | -61,225 |
| Wasm code-section payload | 1,852,735 | 1,782,946 | -69,789 |
| Wasm stripped module (`wasm-tools strip --all`) | 3,837,548 | 3,776,453 | -61,095 |
| Stripped Wasm, `gzip -n -9` | 2,443,127 | 2,434,276 | -8,851 |

`nm -CS --defined-only` native function sizes (bytes):

| Symbol | Before | After |
|---|---:|---:|
| `JpegParser::decode_scan<&dyn Stop>` | 126,903 | 69,511 |
| `EntropyDecoder::decode_block_tracked` | absent / inlined | 18,001 |
| `builtin_tables::select_tables` | 7,459 | 1,141 |
| `builtin_tables::ycbcr_420_tables` | 7,004 | inlined indexed access |
| `builtin_tables::ycbcr_444_tables` | 7,017 | inlined indexed access |
| `builtin_tables::build_table_set` | absent | 345 |
| `StripProcessor::process_strip` | 31,815 | 31,026 |
| `StripProcessor::process_strip_common` | 21,826 | 21,826 |

The final descriptors also occupy three native data symbols of 3,840 bytes
each; code savings and stripped-file savings therefore differ. XYB table-family
construction was already folded into `select_tables` before. Neither revision
has a standalone `color_convert_strip` symbol; conversion is linked into the
strip processing functions. Symbol deltas are not additive attribution across
all generic/inlined callers.

Wasm named bodies: `decode_scan` 144,806 → 74,755 bytes, new shared tracked block
19,361 bytes, `select_tables` 20,917 → 925 plus a new 1,138-byte constructor.
`process_strip` stays 35,492 and `process_strip_common` stays 7,931 bytes.
Native `nm` and disassembly both show **zero `__simd_inner_*` definitions/calls**
and **zero `vmaskmov` instructions**, before and after.

## Exactness, checks and limits

Native before/after 27/27 exact and WASI before/after 27/27 exact include all
seven encoded streams and both sets of decoder rasters. WASI replay executes
under Wasmtime; unknown-unknown is the linked size target. This does not claim
that our JPEG IDCT equals the independent decoder's IDCT.

A pre-existing cross-target exception is preserved: native versus WASI differs
in five samples of the XYB raster (589,824 samples total, maximum difference 1).
The independent oracle also differs in 23 linear-RGB16, 29 linear-RGBf32 and three
sRGB420 samples, each by 1. All seven encoded streams are cross-target identical.
These exceptions exist on the unmodified baseline and remain unchanged; no
pixel tolerance was loosened. See the owning `CLAUDE.md` backlog entry.

Validation on the final source:

- `cargo test --release --workspace --lib --bins --tests --examples`: **2,626
  passed, 0 failed, 119 existing ignored**. The brief's hybrid-host exception
  excludes local benches; both required benches execute on i134.
- `cargo clippy --workspace --all-targets -- -D warnings`, plus standalone
  `cargo clippy -p zenjpeg --lib -- -D warnings`: pass.
- Scoped `cargo fmt --check` across workspace packages: pass.
- Workspace doctests: 16 passed, 0 failed, 102 existing ignored. Public API
  snapshot check (`just api-doc-check`): pass unchanged on final source.
- Original legal-symbol coverage and the new table fingerprint guard: pass;
  the deliberate symbol-swap negative control failed behaviorally before
  restoration. Existing reconstruction-metadata and exact pixel guards pass.

The i134 measurement/build guard used 12 GiB / one job. Baseline peak RSS was
1.28 GiB and the retained incremental measurement/replay run peaked at 0.64 GiB
(`run-heavy`); differing build-cache states preclude a memory-improvement claim.
No ARM throughput, metadata-enabled throughput, scalar-native throughput,
quality sweep or release qualification is claimed.

## Reproduction and retained evidence

Build/run `zenjpeg/benches/{huffman_decode,leaf_dispatch}.rs` and
`zenjpeg/examples/leaf_dispatch_probe.rs` from the revisions above. Use an
explicit output directory for the native/WASI probe. Keep the same toolchain,
features and flags when comparing linked size. `ZENJPEG_BENCH_FIXTURES=leaf`
is required for the bounded Huffman mode; an explicit missing fixture fails
compilation through `include_bytes!`.

```sh
ZENJPEG_BENCH_FIXTURES=leaf cargo bench --locked -p zenjpeg --bench huffman_decode -- --rounds=30 --cpu-policy=unrestricted --format=llm
cargo bench --locked -p zenjpeg --bench leaf_dispatch -- --rounds=30 --cpu-policy=unrestricted --format=llm
cargo build --locked --release -p zenjpeg --example leaf_dispatch_probe
RUSTFLAGS='-C target-feature=+simd128' cargo build --locked --release -p zenjpeg --example leaf_dispatch_probe --target wasm32-unknown-unknown
```

Retained private evidence is under `~/tmp/bench-lanes-2026-11-23/results/zenjpeg/`:
`before/`, rejected `after/`, retained `after2/`, `abba/`, `abba-after2/`,
`summary-after2.json`, and `manifest.json` with binary and all witness hashes.
Raw immutable Zenbench packets, per-process logs, symbols, disassembly and
both consumers accompany the measured candidates. Rejected/interrupted evidence
is retained. The report and scalar evidence are mirrored to the artifact archive.

| Consumer SHA-256 | Before | Retained |
|---|---|---|
| `probe` | `6d42e558dd6f1dac6a68a2b72a4f031670b8c502e8b504cebede4c95abf111ed` | `15cd4ab2bd534ab469915627745bef797da458ade6a1e5484c02ae2e42929538` |
| `probe.wasm` | `d70f0c9154df6b6b38cfe5d7a4debc3c1672596ce664a325072d7c99559bb5da` | `3240a0b075e0466c7354627efeb0bb9e08f8d769fd1284db8c964013ce172006` |
| `huffman_decode` | `8a23a58a6897b07007adcb510107654b86c0ea162e3d8ec862b7e8ddc2a73292` | `df4011496d7f4a9e9e6f5645a66bf165fc1672f9f3e89291b8cbfd39e90bd851` |
| `leaf_dispatch` | `d690f54f77e7296f1f8f7e4da1d2bdaf417f18320c1b93f225059a0d5e3ed7f2` | `252b8767ef99cb644fda9f36d9cf65996b057ea2e58ae0e8b4e9328c2366de14` |
