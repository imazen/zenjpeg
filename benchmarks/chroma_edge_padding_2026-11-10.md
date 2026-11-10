# Partial-strip chroma padding — 2026-11-10

The zensr small-image witnesses exposed chroma on a neutral 32×8 RGB source.
Both native and libjpeg-turbo decoders produced identical colored pixels.
A standalone synthetic neutral-RGB regression also fails before this fix:
32×8, 420 q40, first decoded RGB pixel `[54, 50, 65]`.

The fused 420/gamma-aware paths write `cb_down`/`cr_down` at padded chroma
stride. Partial-bottom padding replicated the unused full-resolution buffers
instead, leaving the DCT's missing chroma rows at zero or from a prior strip.
The fix replicates the last real row in the buffers the DCT consumes.
Pre-subsampled input now receives horizontal padding before the same padded-
stride vertical operation. No public signatures or quality thresholds change.

Coverage: neutral RGB at q40, widths 1/7/8/9/15/16/17/31/32/33/129, heights
1 through 33, all four YCbCr sampling modes; full-resolution/pre-subsampled
colored planar parity at packed and padded strides; gamma-aware and iterative
controls at seven partial geometries and three subsampled modes. All pass.
The library suite passes (1189 passed, one pre-existing ignored test).
The separate fused-parallel encoder, other qualities and third-party encoder
parity are not established by this regression matrix.

Commands, from the restoration integration checkout:

```sh
cargo test --locked -p zenjpeg --test bundled chroma_edge_padding -- --nocapture
cargo test --locked -p zenjpeg --lib
```

The controlled synthetic regression failed before the fix and passed after it.
An incomplete dependency-source cache initially prevented compilation; an
isolated Cargo home using the same package archives resolved that environment
issue without changing dependency versions or source identities.
The build reports an existing unused `detect::record::insert_record` warning.
The optional C++ oracle submodule is absent and was not enabled for these tests.

Resource wrapper on i265, four jobs, 8 GiB cap (full logs retained locally):

```text
before: rc=101 107s | peak-RSS 1.18GiB | min-avail 21548MiB | peak-load 3.21
matrix: rc=0     2s | peak-RSS 1.17GiB | min-avail 23201MiB | peak-load 0.18
library:rc=0    19s | peak-RSS 1.13GiB | min-avail 22163MiB | peak-load 1.07
```

The 107-second run includes a cold build and is not an encoder speed claim.
This corrects input generation; it does not fix learned RGB neutrality or
zensr's treatment of partial-block coefficient constraints.
