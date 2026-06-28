# zenjpeg benchmarks — methodology & reproduction

How to run zenjpeg's encode/decode comparisons fairly, and where the committed
result data lives. The headline numbers in the root [README](../README.md)
"Performance" section come from these protocols; the dated, per-function data
lives in the result files listed at the bottom.

## Fairness guarantees

The comparison benches are built so the numbers mean something:

- **No `-C target-cpu=native`.** All builds use runtime SIMD dispatch
  ([archmage](https://github.com/imazen/archmage) tokens — AVX2/FMA/AVX-512 on
  x86-64, NEON on aarch64, scalar fallback), which is what ships. A `native`
  build bakes in ISA extensions the published binary never has, and inflates the
  numbers.
- **No I/O in the timed region.** Source pixels (encode) and source JPEG bytes
  (decode) are loaded/synthesized into memory *before* timing starts. The timed
  closure only runs the codec — encode into a `Vec<u8>`, decode from a `&[u8]`.
  Output is consumed (hashed / black-boxed) so it isn't optimized away.
- **Threading stated explicitly.** zenjpeg encode and the default decode path are
  single-threaded per call; parallel decode/encode is opt-in behind
  `--features parallel` and is reported as a separate series. Never compare a
  single-threaded contender against a thread-pooled one — run both in the same
  mode.
- **Apples-to-apples inputs.** Same images, same dimensions, same pixel format,
  same quality/distance target, same chroma subsampling across every contender.
- **Realistic content.** Test patterns are noise + patches or real photos
  (CID22), never smooth gradients (which produce degenerate DCT coefficients) and
  never the Kodak corpus (overfit by every codec).
- **Interleaved A/B where it matters.** The zenbench-based benches
  (`encode_zenbench`, `decode_zenbench`, `ycbcr_turbo`) run contenders
  round-robin so both see the same thermal/turbo/scheduler state, and report a
  paired confidence interval on the *relative* speed.

## Environment

Committed numbers were measured on:

- **CPU:** AMD Ryzen 9 7950X
- **OS:** Linux (WSL2)
- **Build:** `--release`, default target (no `-C target-cpu=native`)
- **Toolchain:** record `rustc -V` in any new result file you commit.

Memory claims must come from heaptrack or `/usr/bin/time -v` (max RSS) at the
measured size — never extrapolated from another size.

## Reproduce

```sh
git clone https://github.com/imazen/zenjpeg && cd zenjpeg
git checkout <commit>          # the commit named in the result file you're reproducing

# Encode / decode throughput (criterion):
cargo bench -p zenjpeg --bench encode
cargo bench -p zenjpeg --bench decode

# Interleaved paired A/B (zenbench):
cargo bench -p zenjpeg --bench encode_zenbench
cargo bench -p zenjpeg --bench decode_zenbench
cargo bench -p zenjpeg --bench ycbcr_turbo

# Decode vs libjpeg-turbo / zune-jpeg / jpeg-decoder:
cargo bench -p zenjpeg --bench decode_compare
cargo bench -p zenjpeg --bench decode_mozjpeg

# Parallel decode (separate series — run with the feature on):
cargo bench -p zenjpeg --bench decode --features parallel

# C++ jpegli parity comparison (requires the internal/jpegli-cpp submodule built):
cargo bench -p zenjpeg --bench cpp_comparison
```

Criterion writes structured JSON to `target/criterion/<group>/<bench>/new/estimates.json`
— extract from there instead of re-running to re-parse (`jq '.mean.point_estimate'`).

## Competitors (pin these when reproducing elsewhere)

These are dev-dependencies, so `cargo` pins them for you at the committed
`Cargo.lock`. The versions used for committed results:

| Competitor | Version | Notes |
|-----------|---------|-------|
| [`zune-jpeg`](https://crates.io/crates/zune-jpeg) | 0.5 | Fastest pure-Rust reference decoder |
| [`mozjpeg-sys`](https://crates.io/crates/mozjpeg-sys) | 2.2 (`nasm_simd`) | libjpeg-turbo (C + NASM SIMD), the "mozjpeg/libjpeg-turbo" rows |
| [`jpeg-decoder`](https://crates.io/crates/jpeg-decoder) | 0.3.2 | Pure-Rust decoder |
| [`mozjpeg-rs`](https://crates.io/crates/mozjpeg-rs) | 0.5.3 | Rust mozjpeg bindings |
| C++ jpegli (`cjpegli`/`djpegli`) | `internal/jpegli-cpp` submodule | Build per the root CLAUDE.md; used for size/quality parity |

Quality is measured with [`dssim-core`](https://crates.io/crates/dssim-core) 3.4,
[`fast-ssim2`](https://crates.io/crates/fast-ssim2) 0.8.0 (SSIMULACRA2), and
[`butteraugli`](https://crates.io/crates/butteraugli) — never PSNR. Encode RD
comparisons are size-matched (same bytes, compare quality) rather than
quality-matched, and sweep low quality (Q5–Q40) as densely as high quality.

## Committed result data

zenjpeg's dated, per-function measurements live in tracked files rather than
loose tables here:

- [`docs/TUNING_HISTORY.md`](../docs/TUNING_HISTORY.md) — full per-function
  flamegraph and callgrind breakdowns, SIMD analysis, allocation counts, and
  the historical encode/decode tuning record.
- [`CLAUDE.md`](../CLAUDE.md) — the dated wall-clock tables the README headline
  numbers summarize (Decoder Performance 2026-02-15, Parallel Decode, Dequant
  Bias sweep, Strictness conformance), each tagged with the commit it was
  measured at.
- [`BENCH-AUDIT.md`](../BENCH-AUDIT.md) — audit of the bench harnesses for
  fairness.
- [`benchmarks/boundary_rd/README.md`](boundary_rd/README.md) — boundary-RD
  refinement sweep methodology and results.

When you commit a new run, drop it here as
`benchmarks/<topic>_<YYYY-MM-DD>.{md,csv,tsv,log}` with a header stating: the git
commit, CPU/OS, `rustc -V`, the exact command, the threading mode, and the
feature flags. Don't commit numbers you didn't generate, and don't extrapolate
one size or platform to another — measure each.

## Charts (what to plot for which decision)

| Question | Chart |
|----------|-------|
| "Which decoder/encoder is fastest?" | horizontal **bar**, sorted by throughput (MiB/s or MP/s); separate bars for 1-thread vs `parallel` |
| "Speed vs quality/size?" | **RD / Pareto scatter**: x = bpp, y = SSIMULACRA2 / butteraugli; one line per encoder swept across quality |
| "Is the A/B delta real / how noisy?" | **violin** / PDF of per-call times, or the paired 95% CI the zenbench benches print |
| "How does it scale with image size?" | **line**, x = pixels (log); fit `total = α + β·pixels`, report intercept (fixed overhead) and slope |
| "Memory?" | from heaptrack / `time -v` max RSS, measured per size — never extrapolated |

Avoid pie charts, 3D, and dual-axis plots. New comparison charts should use
[zenbench](https://github.com/imazen/zenbench), which does the interleaving and
emits sorted throughput bars and self-contained SVG reports.
