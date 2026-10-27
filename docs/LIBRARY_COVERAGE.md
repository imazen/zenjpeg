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

`controlled_grayscale_fixtures_match_reference_raster_hashes` also checks both
qualified 8-bit grayscale fixtures through controlled decoding. Truncated-prefix
and fuzz tests are additional hardening, not substitutes for raster assertions.
