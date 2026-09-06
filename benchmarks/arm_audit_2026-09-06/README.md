# JPEG ARM audit, 2026-09-06

Coverage: one end-to-end configuration (1024×1024 seeded noise/patches,
Q85, 4:2:0) and five kernel groups. Other quality levels, chroma modes,
content classes, and architectures were not measured here. These measurements
do not establish a codec-wide speedup or calibrate production thresholds.

Host: Apple M4 Pro, macOS, Rust 1.98.0 / LLVM 22. Builds used runtime dispatch,
without `target-cpu=native`, with four build/Rayon/OMP threads and `nice -n 19`.
The benchmark uses interleaved zenbench arms. Scalar means the forced scalar
implementation; LLVM remains free to auto-vectorize it.

## Measurements

Baseline `5c29301f`: NEON encode 24.69 ms versus scalar 31.41 ms; NEON decode
11.62 ms versus scalar 15.78 ms for the same encoded fixture. All five kernel
groups favored NEON. Read `jpeg-tiers.log` for paired confidence intervals;
small-kernel variance was substantial, so exact ratios are not stable estimates.

The generic DCT entry had avoidable dispatch overhead. At benchmark commit
`a83e96fd`, it averaged 65.4 ns versus the existing direct NEON entry's 56.3 ns.
Adding `#[inline]` to `forward_dct_8x8_simd_chained` reduced the generic mean to
56.3 ns; direct NEON averaged 57.7 ns and forced scalar 194.0 ns in that run.
The before/after runs were separate; paired statistics compare arms within
each run, not the two builds. This is an entry-point improvement, not a measured
end-to-end encoding improvement. The code and raw results are in `386514e0`.

The public function named `forward_dct_8x8_scalar` actually calls generic
runtime dispatch. The benchmark disables NEON for its scalar arm and now also
measures this entry with NEON enabled under the explicit `generic_dispatch`
label. Labeling it scalar without disabling the token would be incorrect.

## Correctness and reproduction

The benchmark verifies identical DCT coefficients between the two NEON entry
points, identical decoded pixels across enabled/disabled NEON, and reports
encoded-byte equality (true for this fixture). Encoded files and SHA256 values
are recorded in [fixtures.pointer.md](fixtures.pointer.md).

After the inlining change, 1110 library tests passed with zero failures or
ignored tests. Scoped formatting and clippy for the library and tier benchmark
passed with `-D warnings`. Existing C++ archive-format warnings remain in the
build log. See [validation.txt](validation.txt).

Run `just arm-tiers-macos` for all groups, or
`just arm-tiers-macos forward_dct` for the dispatch comparison. Full output is
saved under `~/tmp`; the benchmark preserves its encoded fixture separately.
