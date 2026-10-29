# Whole-library coverage

The target is every reachable production branch, with deterministic assertions
on behavior. Changed-code coverage is a separate gate and cannot establish this.

On 2026-10-27, the broad report from GitHub Actions run 40432129941 recorded
7,858/10,242 raw branch outcomes in `zenjpeg/src` (2,384 missing). This count
includes inline unit-test branches. Its compiled feature set was parallel,
moxcms, ultrahdr, zencodec, controlled-decode, inventory and __test-utils;
recompression, layout and platform-specific implementations outside that build
were not qualified by this report.

`just library-branch-coverage` measures the production feature combination with
recompression and layout. Set `CARGO_TARGET_DIR` and `ZENJPEG_BRANCH_REPORT` to
absolute paths. Run through the resource wrapper; the recipe has a 30-minute
execution timeout with a 15-second termination grace. Its whole-library report
excludes cfg(test) module branches, retains production error paths, and lists
uncovered true/false outcomes. `--require-complete` on the report checker enforces
100%; current whole-library coverage does not pass that requirement.

CI retains the JSON, LCOV, HTML and whole-library uncovered-branch report. A
complete claim also needs feature-off configurations and architecture-specific
branches, including scalar, x86, ARM and WASM; one Linux feature combination
cannot demonstrate those. Research, expert and diagnostic features are also required in the full target,
including boundary-rd, __expert, recompress-expert, __bdrd-trace,
__debug-tokens, __profile and __alloc-instrument. External-oracle and WASM
test gates need their respective tools and targets. These must be reported
separately when one build cannot exercise both enabled and disabled behavior.

## Raster fixtures

`native_hardening::native_fixture_raster_hashes` decodes every one of the 47
native fixtures through borrowed and owned readers. It verifies the input SHA256,
source precision, dimensions, component count, untouched row padding and SHA256
of every decoded native sample serialized as big-endian u16. The expected
rasters come from the reference identified in `tests/support/native_manifest.json`;
reference provenance is documented in `12bit-lossless-decoding.md`.
An additional 23 hashes assert collected RGB/CMYK output, or native grayscale
for lossless grayscale streams whose reference refuses RGB expansion. The
generator verifies that the original JPEG bytes are unchanged.

`controlled_grayscale_fixtures_match_reference_raster_hashes` also checks both
qualified 8-bit grayscale fixtures through controlled decoding. Truncated-prefix
and fuzz tests are additional hardening, not substitutes for raster assertions.

## Expert and diagnostic gates

The current manifest has no target-zq or picker-research feature. The research
coverage target comprises boundary-rd (optional encoding refinement), __expert
(parameter overrides and sweep plans), recompress-expert (strategy internals),
and the diagnostic features __bdrd-trace, __debug-tokens, __profile and
__alloc-instrument. _dev enables SIMD test dispatch controls. The remaining
__ffi-tests, __corpus-tests, __oracle-tests, __oracle-exiftool and __wasm-simd
gates select test infrastructure rather than alternative production algorithms.
__test-utils exposes generators and verification helpers.

The expanded Linux production-feature baseline completed on 2026-10-27:
7,692/9,844 compiled production branch outcomes (2,152 missing), excluding
cfg(test) modules. This run adds recompression and layout to the earlier matrix;
it does not include the research/diagnostic features yet. Source production
baseline: `c227934a`. Command: `just library-branch-coverage`, with test debug
level 1 and incremental compilation disabled. Wrapper result:
`run-heavy: done rc=0 1039s | peak-RSS 2.70GiB | min-avail 21563MiB | peak-load 10.10`.
These resource figures describe the validation process, not codec memory usage.

`just library-all-features-check` passed after initializing the pinned jpegli,
Highway and libjpeg header sources. It checks every feature and every target,
including expert/research examples and benches. This was compilation, not test
execution: existing deprecated helper calls and float-inference warnings remain.
The C++ build wrapper now respects an explicit CMake job limit or Cargo's
`NUM_JOBS`, instead of replacing the budget with the host CPU count.

The all-feature execution on 2026-10-27 completed with two failing targets:
a newly introduced size-test pixel oracle used a different IDCT, and the
ExifTool module lookup lacked PERL5LIB. The corrected size test uses the existing
libjpeg-turbo oracle and passes for the retained cases plus noise and flat-chroma
bands. The ExifTool test passes with its pinned distribution's lib directory on
PERL5LIB. Added diagnostic behavior tests and the all-feature library suite pass.
These focused reruns do not constitute a fresh full-library coverage measurement.

## Coverage runner timeout (2026-10-29)

Coverage run 40852840350 on native completion revision 7c5a9501 hit the
20-minute aggregate integration-command timeout during colorspace_paths.
The bundled suite had completed 1,186 tests without failure in 559.73 seconds;
run 40843404474 completed the same bundled suite in 261.75 seconds. These CI
wall-clock observations are not a controlled performance comparison.

The aggregate integration command now has a 40-minute timeout and its coverage
job a 70-minute cap. Test selection, features, assertions, coverage thresholds,
individual oracle commands and profile settings are unchanged. YAML and every
static shell block parse successfully. The full extended CI run still must pass;
this timeout adjustment is not test qualification.
