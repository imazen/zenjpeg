# Encoder strip padding qualification (2026-11-10)

Base: `a8e92e810b054835b4da4946735c469cce874002`.
Synthetic fixtures only. No public API or test threshold changes.

## Findings and repairs

A neutral 32×8 RGB source could acquire color under 420. Fused RGB and
gamma-aware converters write downsampled chroma, but bottom padding wrote
unused full-resolution planes. Pre-subsampled input also lacked right padding
and replicated bottom rows at packed rather than padded chroma stride.

Colored witnesses exposed another mechanism: standard 420/440 must downsample
real rows before replicating the last chroma row. Padding full-resolution rows
first substitutes the last source sample for the final chroma average when the
partial strip has even height. For a 1×2 440 witness the expected padded Cb
sample is 225.5; the old path produced 216.

The separate fused parallel pipeline needed all of the following:

- Rearrange packed chroma to DCT stride, then extend right and bottom edges.
- Extend bottom luma rows, including segments with a one-row tail.
- Convert 422/440 into full-resolution scratch before subsampling. Passing
  subsampled output storage directly to the 444 converter panicked.
- Extend visible AQ block grids to the MCU grid before indexing at padded
  stride. Both right dummy blocks and bottom dummy block rows matter.
- Use the shared layout's 440 plane stride for StreamingAQ. A width-17
  witness previously allocated stride 24 while AQ expected 32. Keep MCU
  column count derived from sampling geometry, independent of storage padding.

The earlier two-file strip repair was extracted from `0fee09ca`; the broader
restoration integration is not part of this change.

## Reachable paths

| Producer / dispatch | Padding and consumer | Qualification |
|---|---|---|
| RGB/RGBA/BGR/BGRA/BGRX u8, box 420 | Fused zenyuv conversion → packed chroma → padded downsampled planes | Raw conversion output plus independent coordinate clamps; DCT/entropy/decode |
| Other box sampling; RGB/RGBA u16/f32 fallback | Full-resolution conversion → subsample real rows → pad chroma | Scalar independently indexed box average and edge replication |
| GammaAware / GammaAwareIterative, 422/420/440 | Direct downsampled chroma; same padded-plane DCT consumer | Both methods; u8 channel orders and linear u16/f32 |
| 444 with any downsampling setting | No gamma/downsampling kernel; standard conversion/copy | Aligned controls, all residues, aliases |
| Gray u8/u16/f32 | Public byte wrapper forces 444; only Y is consumed | Internal plane oracle and public stride/partition tests |
| RGB passthrough | u8 RGB family only, 444; three padded full-resolution planes | Explicit source extension, exact visible coefficients |
| XYB Full / BQuarter | Full-resolution X/Y; B-specific conversion, subsampling and padding | Explicit source extension, exact visible coefficients; u8 orders plus u16/f32 |
| Full-resolution YCbCr planar | Packed copied planes, common subsample/pad tail | Exact byte parity with independently repeated pre-sampled chroma |
| Pre-subsampled YCbCr planar | Copy/level shift → right pad/restride → bottom pad | Scalar coordinate oracle, independent plane strides, strip partitions |
| Normal build with `parallel` | Same strip producers; parallel DCT/entropy consumers | Large strip threshold/chunk witnesses plus the same end-to-end matrix |
| `encode_bytes_parallel` | Separate RGB8 box conversion/AQ/DCT/symbol pipeline | Four sampling modes, independent segment-plane and AQ oracles, public restart segments and turbo decode |
| Fixed baseline / optimized baseline / progressive | Common strip/DCT input, distinct entropy output | Exact bytes across source strides and row partitions |

RGBX aliases RGBA; BGRA/BGRX share the public BGRX conversion; u16/f32
RGBX aliases RGBA. Typed RGB pushes delegate to the byte wrapper and are not
separate strip kernels. Alpha/X is ignored by these encoders. CMYK has no
public byte layout and is not an additional supported strip input. Legacy
interleaved `YCbCr8`/`YCbCrF32` enum values map to the RGB internal format;
this report does not qualify their YCbCr semantics. Real planar input uses
`YCbCrPlanarEncoder` and is covered separately.

Planar input has no RGB gamma conversion. XYB rejects grayscale/CMYK; RGB
passthrough rejects deep/gray input. Fused parallel implements RGB8 box YCbCr;
it does not implement gamma-aware, XYB, RGB passthrough or grayscale conversion,
and is not a progressive/streaming producer. Its public entry point rejects
images resolving to fewer than two restart segments. Small geometries are
therefore tested at its internal segment boundary, while public tests use
admitted frame sizes. This does not establish unrelated config semantics for
that older convenience entry point.

`push_subsampled` explicitly does not buffer partial strips. Its tests use
one, two, or all complete strips per push, followed by a final partial strip.
Arbitrary row partitions are covered by byte and full-resolution planar pushes;
nonfinal partial pre-sampled pushes are not claimed as supported or qualified.

## Measured focused coverage

All counts below are emitted by tests, not inferred quality claims:

| Oracle | Passing coverage |
|---|---:|
| Pre-sampled coordinate replication | 5,616 geometries, three planes each |
| RGB box/gamma/iterative coordinate replication | 16,848 configurations |
| Twelve internal formats / four patterns / sampling and method controls | 6,336 configurations |
| Separate fused segment planes | 2,496 geometries per sampling mode |
| Separate fused AQ coordinate replication | 4,356 geometries |
| Byte source stride/partition/entropy equality | 5,940 comparisons |
| Planar full/pre-sampled, packed/padded, partition equality | 16,224 comparisons |
| Neutral RGB qualities / methods / sampling | 1,452 encodes |
| RGB/gray/XYB format controls, explicit extension | 21,090 visible coefficient blocks |
| RGB/gray/XYB Full complete residues, explicit extension | 2,664 geometries; 42,672 visible coefficient blocks |
| Both XYB layouts, independently sampled planes | 2,664 geometries, three planes each |
| Explicit automatic parallel entropy / row partitions | 192 configurations |
| Quantization / trellis / 16-bit quant allowance controls | 528 comparisons |
| Public fused parallel segment/edge neutrality | 192 encodes |
| Independent libjpeg-turbo strip RGB equality | 1,728 encodes |
| Independent libjpeg-turbo fused RGB equality | 384 encodes |

The plane matrices use widths 1–33, 63/64/65 and 127/128/129; every possible
partial height within an 8/16-row strip; zero, one and two preceding complete
strips; aligned controls; odd dimensions; right/bottom corners; and poisoned
or differently populated reused buffers. They cover every horizontal/vertical
residue modulo 8/16, including width/height 1 and short/wide/tall/narrow cases.
Patterns include neutral and non-neutral constants, checkerboards and
coordinate-varying color with right/bottom impulses. Additional wide strips
cross the 2,048-chroma-block and 4,096-luma-block thresholds and 4,096-block
parallel chunks (widths near 16,384 and 32,768). SIMD tail witnesses cross
8/16/32/64/128 source-column boundaries.

Quality list: 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100. The independent decoder
matrix uses 5, 20, 40, 60, 80, 100. Quantization controls exercise jpegli
separate/shared chroma tables, Robidoux independent chroma quality, trellis
optimization and the 16-bit-table allowance. No assertion depends on a lossy
source round-trip being exact. Byte equality uses identical configuration and
input operation; explicit frame extension disables AQ/deringing and compares
only visible blocks, excluding entropy dummy blocks outside the image.
For subsampled XYB, explicit RGB extension is equivalent only when the final
sample already replicates the edge (the format-control witnesses use odd
widths/heights). Even partial tails require replication AFTER sampling: their
complete residue matrix uses independently sampled XYB planes, rather than
assuming source RGB extension is equivalent. The unsampled RGB/gray/XYB Full
paths additionally compare coefficients across every residue class.

The coordinate oracles share color conversion with production but independently
construct sample addressing, box averages, right/bottom replication and AQ grid
extension. Exact independent decoder equality is an additional check, not the
padding oracle. No implementation-generated golden pixels are accepted.

## Reproduce

```sh
just strip-padding-check
just strip-padding-lint
just strip-padding-regressions
just strip-padding-oracle-build "$HOME/tmp/zenjpeg-padding-oracle"
ZENJPEG_TURBO_BIN="$HOME/tmp/zenjpeg-padding-oracle/build" just strip-padding-oracle
```

The existing builder verifies libjpeg-turbo 3.2.0's release archive SHA-256
`6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e`.
Missing explicitly selected oracle tools fail the tests. The two external-tool
tests have an explicit caller-selected test gate and CI invokes them on Linux;
the ordinary synthetic suite needs no external tools or corpus. Existing
Windows ARM64, macOS ARM64 and Linux CI run the synthetic suite with and without
parallel kernels; i686 additionally runs it through `cross` with parallel.

A controlled replay attached these same tests to otherwise unchanged base
production code: all 19 regression library tests failed. The fixed revision passes
all 19; complete-residue XYB and malformed-AQ-grid boundary controls bring
the final focused library suite to 21 tests. Distinct witnesses expose packed chroma stride, full-resolution output
storage, missing downsampled bottom padding, standard even-height averaging,
AQ grid size and the 440 AQ input stride separately.

Locally, eight archmage dispatch permutations ran without warnings, including
the scalar fallback. Cross-tier bit identity of unrelated floating-point
conversion is not asserted: every tier must satisfy its own exact padding
oracle. Other platforms require observed CI results; wiring is not a claim of
execution. No C++ jpegli encode parity or performance claim is made.

## Test isolation

Archmage token permutations change process-wide dispatch. Two-pass float/AQ
reference tests hold `archmage::testing::lock_token_testing()` so another
library test cannot change tiers between actual and reference passes. Without
that guard the combined parallel/zencodec library suite exposed an AQ-oracle
race, despite isolated focused runs passing. Assertions and expected samples
remain unchanged.

The complete regression run at `3236b601` passes 2,483 default
library/integration tests and 1,305 parallel/zencodec library plus
strip/streaming/grayscale/cancellation tests,
15 doctests, scoped formatting, and strict Clippy for library and all tests
with `parallel`. Existing external/corpus test gates retain their prior state.
The two new turbo gates were explicitly run using a fresh SHA-verified release
build, in addition to the ordinary synthetic tests.

## Changed-code coverage gates

The existing LLVM gates pass locally: **296/296 changed executable lines**
and **18/18 changed production branch outcomes**. This is the gate's measured
changed-code scope, not whole-library coverage. Command: `just
strip-padding-coverage`, with `ZENJPEG_PADDING_LCOV` and
`ZENJPEG_PADDING_BRANCHES` pointing at caller-selected report paths. The recipe
uses the same pinned nightly and unchanged gate scripts as CI.

AQ grid extension is an internal helper so malformed empty/short/long grids can
be checked directly, before indexing. The reference test requires its final
pending AQ row, rather than conditionally omitting it. The bottom-padding guard
uses the converter's dispatch invariant: after excluding XYB and RGB passthrough,
`chroma_already_downsampled` implies color input (gray always uses the standard
path). This removes a redundant unreachable grayscale predicate.

The first CI run at `c6085f89` hit an existing i686 bundled test's attempt to
launch `djpeg`, returning `ENOSYS`. Direct cross execution passed 1,193 library
and eight synthetic integration tests. The focused CI i686 matrix now runs
before the broad suite, preserving the broad suite's failure behavior while
making its padding evidence available independently. The coverage artifact
upload also runs after a gate failure, retaining the raw reports; the gate still
fails the job. Latest platform status is on [the PR](https://github.com/imazen/zenjpeg/pull/262).
