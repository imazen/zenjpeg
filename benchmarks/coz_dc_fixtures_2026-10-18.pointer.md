# Checked JPEG profiling fixtures — 2026-10-18

No production optimization or accepted speedup is included.

The development worker and fixture generator are described in
[internal/coz-loop/README.md](../internal/coz-loop/README.md).
The generator source is commit `33c8365f`; exported fixtures identify
`1d2d60c65b55944134727fc1461001cc66031746`, whose generator source is identical. This is the original build identity,
retained before a parent description changed; the generator itself is available
in the unchanged published `33c8365f` snapshot.

On r7900x, `cargo test --locked -p zenjpeg --lib` passed 1175 tests with
one preexisting ignored test. Scoped formatting and clippy passed.
`run-heavy: done rc=0 14s | peak-RSS 0.77GiB | min-avail 21435MiB | peak-load 1.01`.
`just coz-check && just coz-fixtures` passed with the exported directory.
`run-heavy: done rc=0 1s | peak-RSS 0.14GiB | min-avail 22778MiB | peak-load 0.56`.

Artifacts were checked on dev at
`/mnt/v/output/zenjpeg/coz-dc-2026-10-18-r7900x/`:

- `dc-fixtures-1d2d60c6/MANIFEST.tsv`: source identity and all 504 cells.
- `dc-fixtures-1d2d60c6/independent-decode.json`: every JPEG SHA256,
  ImageMagick version and readback dimensions.
- `promoted-oracle.log`: execution of the retained independent readback tool
  (source `4cc1c6de`).

Every fixture has zero AC coefficients after readback through zenjpeg;
the test also checks actual-grid DCs, quantization tables and sampling.
ImageMagick independently reads all 504 files with the expected dimensions.
This is structural and entropy-path coverage, not a quality sweep or agreement
between independent decoder pixels. No training data or source thresholds
are derived from these fixtures. No R2 or tower mirror has been made.
