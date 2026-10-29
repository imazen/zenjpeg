# Native-precision JPEG decoding

The decoder supports 12-bit DCT JPEG and Huffman predictive lossless JPEG
(SOF3, precisions 2–16). Encoding remains unchanged.

## Output

`OutputTarget::Native16` retains integer samples in the frame's range: 0–4095
for 12-bit, rather than expanding to 0–65535. `DecodeResult::pixels_u16()`
borrows those samples; `pixels()` and `into_pixels()` expose the `U16` variant.
`sample_precision()` reports the source precision, including after conversion.
Point transforms restore the low zero bits.

For bounded row delivery, use `ScanlineReader::read_rows_rgb16` with
`ImgRefMut<rgb::RGB<u16>>`, or `read_rows_gray16` with `ImgRefMut<u16>`.
Stride is measured in pixels. Gray reads return the source gray/Y channel;
RGB sources are converted to luma with BT.601. Borrowed and owned readers share
the reconstruction and conversion code.

RGB retains the source primaries and transfer function. JPEG YCbCr uses the
BT.601 matrix. Native decoding does not perform ICC conversion. For these
sources, u8 output rounds the native range to 0–255; f32 output normalizes the
reconstructed integer samples to 0–1, optionally applying the sRGB transfer
function for linear output. Unlike the existing 8-bit f32 path, these normalized
samples do not retain out-of-range IDCT ringing.

The existing `into_parts()` tuple has no integer slot; it converts native
samples to normalized f32. Use `into_pixels()` for ownership without conversion.

## Component output

`ScanlineReader::read_rows_components16` accepts `ImgRefMut<[u16; N]>`, where
`N` must equal the frame component count. `component_ids()` identifies the
channels in frame order. It preserves reconstructed component samples without
color conversion; DCT samples are upsampled to full resolution using the selected
filter. A two-component frame has no implied RGB meaning: RGB reads reject it.
This interface requires source components, so readers holding already-converted
or pixel-transformed buffers reject raw component reads.

`read_rows_cmyk16` accepts `ImgRefMut<Cmyk16>` and retains JPEG's channel/inversion
convention (including Adobe-inverted CMYK). YCCK is converted with the libjpeg
matrix. `output_format(PixelFormat::Cmyk)` supports collected native u16, scaled
u8 and normalized f32 output. Crops and pixel transforms retain all four channels.
ICC conversion remains external; requesting linear-sRGB CMYK output is rejected. These APIs do not normalize Adobe-inverted
samples to a different ink convention.

The existing subsampling enum describes only the four common three-plane layouts;
use header/component metadata when handling other layouts. Native-resolution
YCbCr plane reads remain specific to the optimized three-plane reader; the new
component reader supplies full-resolution components for all supported layouts.

## Supported modes and limits

- Twelve-bit sequential and progressive DCT, including arithmetic coding;
  gray, RGB, YCbCr, CMYK/YCCK and raw component images (up to ten channels). Integral sampling
  ratios use the libjpeg interpolation rules, including 4:1:1, vertical 4:1,
  3:1 and asymmetric component sampling.
- Lossless one- through ten-component images, including CMYK, predictors 1–7,
  point transforms, interleaved or separate component scans, and row-aligned
  restart intervals.
- Crops, pixel transforms, strided/chunked reads and metadata preservation.

Native decoding currently rejects XYB, more than ten components, fractional
sampling ratios, subsampled lossless, lossless DNL, differential lossless and
arithmetic lossless.
Deblocking, dequantization bias and ICC conversion are not supported for these
sources. Twelve-bit DCT always uses the libjpeg integer reconstruction.
Coefficient extraction and lossless JPEG-domain transforms/restructuring support
8- and 12-bit DCT with one through four components. Coefficient containers retain
`sample_precision`; sequential output uses SOF1 for 12-bit data. This is coefficient
transcoding, not a predictive SOF3 encoder. Lossy recompression and JBRD reconstruction-metadata extraction remain 8-bit only.

Sequential DCT and lossless readers keep row buffers. Progressive and
non-interleaved DCT retain coefficients between scans; pixel transforms use the
existing buffered permutation. Collected output reserves its memory budget before
constructing the reader, so output and coefficient storage cannot each spend the
whole limit. Crops conservatively reserve the uncropped output size.

## Implementation

The integer islow IDCT shares its butterfly between 8 and 12 bits. Twelve-bit
reconstruction uses one first-pass guard bit instead of two; the established
8-bit SIMD dispatch is unchanged. Twelve-bit chroma interpolation uses widened
scalar intermediates where an 8-bit SIMD kernel would overflow.

The uncommon DCT component layouts share one component-independent row stage:
three MCU strips retain the interpolation context. It reuses the entropy decoder,
integer IDCT and vertical interpolation kernels rather than defining a decoder
for each color space or sampling ratio. The existing optimized three-plane paths
remain unchanged.

SOF3 shares marker framing, Huffman tables, bit reading and restart handling with
DCT JPEG. Its small prediction module retains one previous row per scan and
independent entropy state, allowing separate component scans to stream without
full-image pixel storage. Category 16 represents -32768 without magnitude bits;
DCT difference coding retains its separate behavior.

## Verification

`just native-precision-check` runs ordinary regression tests and library checks.
`just native-integration-check` also checks the optional inventory integration,
fuzz target compilation and doctests. No binary fixtures are added: small known
streams are constructed in Rust, and external-oracle fixtures are generated at
runtime.

CI builds the SHA-256-pinned libjpeg-turbo 3.2.0 release as a test-only oracle.
Locally, set `ZENJPEG_TURBO_BIN` to its build directory (containing `cjpeg-static`
and `djpeg-static`) and run `just native-precision-oracle`. Missing tools fail
the explicitly requested oracle run.

The oracle compares every sample, with no tolerance. Its matrix covers:

- All lossless precisions 2–16, seven predictors, three point transforms,
  restart boundaries, gray/RGB and separate scans.
- Sequential/progressive 12-bit DCT, the supported integral sampling layouts,
  tiny/partial-MCU dimensions, noise, constant blocks and checkerboards,
  quality 5/30/60/91/100, triangle and nearest upsampling, RGB and arithmetic.
- Raw two-component and CMYK/YCCK equality with a test-only libjpeg API helper,
  including lossless precision boundaries, all predictors and point transforms.
- Sequential/progressive coefficient preservation and independent jpegtran
  transform comparisons for 12-bit DCT, including CMYK/YCCK.
- Native integers, normalized floats, u8 conversion, owned/borrowed strided
  reads, crops and all seven pixel transforms.

Ordinary tests cover constant streams across the lossless predictor/precision
matrix, category-16 consumption, tables before SOF, truncations, invalid scan
parameters, cancellation, memory limits and output padding. Existing decoder,
metadata and 8-bit dispatch tests remain in the regression gates.

## TIFF consumer integration

Checked against [zentiff at ae920c1](https://github.com/imazen/zenextras/tree/ae920c1e2ac0fe0e6e2fcb709d99934ba36ea49c/zentiff)
and its pinned [image-tiff da9f8647](https://github.com/lilith/image-tiff/tree/da9f86471ffe04575efaaf8d581e3088bf9ca5a4).
Zentiff's `jpeg` feature delegates to image-tiff; that revision uses zune-jpeg,
so upgrading zenjpeg alone does not enable 12-bit JPEG-in-TIFF.

For supported gray/RGB output, the native u16 readers supply the required
precision without another zenjpeg API. `read_rows_ycbcr_native_i16` also retains
12-bit raw component samples at native chroma resolution; the signed container
holds the complete 0–4095 range. The full-resolution f32 plane reader is another
option when the TIFF layer owns color conversion.

The consumer still needs to:

1. Wire image-tiff's JPEG reader to the reviewed zenjpeg revision and update
   zentiff's image-tiff pin. Reassemble shared `JPEGTables` with each strip/tile
   by removing the tables' EOI and the image's SOI. This assembly is tested
   against the oracle fixtures, including tables preceding SOF.
2. Treat decoded JPEG u16 samples as unpacked, with explicit byte order. The
   current [chunk pipeline](https://github.com/lilith/image-tiff/blob/da9f86471ffe04575efaaf8d581e3088bf9ca5a4/src/decoder/image.rs#L1244)
   sends any 12-to-16-bit layout through packed-bit unpacking; routing native
   JPEG samples through that branch would corrupt them.
3. Apply TIFF photometric interpretation exactly once. The current JPEG reader
   requests the JPEG input color space from zune-jpeg, while zentiff's descriptor
   mapping groups `YCbCr` with `RGB`. Do not label raw Y/Cb/Cr planes as RGB or
   apply a second conversion to zenjpeg's RGB rows. TIFF-specific coefficients
   and reference black/white values need a deliberate raw-component route.
4. Preserve source precision in metadata and define the pixel-buffer range.
   Zentiff currently maps 9–16-bit samples to a plain `GRAY16`/`RGB16` descriptor
   and passes u16 values through. Native 12-bit values are 0–4095, not full-range
   16-bit values: normalize when handing them to a full-range pixel consumer,
   or carry a descriptor that explicitly represents the narrower range.
5. Add end-to-end TIFF tests for shared/per-chunk tables, strips and tiles,
   padded edge tiles, byte orders, photometric modes, limits and cancellation.

The TIFF wrapper itself was inspected, not modified or certified end to end.

## More than four components

Frames support 1–10 components; each scan still carries at most four. Entropy
predictors are indexed by scan slot, not frame component index. Unknown layouts
are available through `read_rows_components16`; RGB/gray conversion rejects them.
The reader keeps frame component identifiers and preserves pixel stride.

DQT selectors remain 0–3 and Huffman selectors remain 0–3. Arithmetic conditioning
selectors support 0–15. Components latch the quantization table at their first
scan, so later DQT redefinitions do not change earlier components. Extracted
coefficients use indices into their owned table list; redefined tables are
compacted, not confused with the current marker-slot contents. Encoding and
coefficient transcoding retain their existing four-component limit.

`just multicomponent-check` covers counts, table redefinition, invalid headers,
truncated prefixes, limits, crops and strided borrowed/owned readers.
`just multicomponent-oracle` requires an explicitly built component reference.

### Reference limitation and provenance

The unmodified pinned libjpeg-turbo 3.2.0 encoder produces a five-component JPEG,
but its decoder rejects the fifth component: `get_sos()` searches only the first
four frame components. Its lossless multi-scan controllers also index four-entry
scratch arrays using frame component indices. The six-component lossless encoder
terminated with a stack-smashing diagnostic in the test helper.

`scripts/build-component-oracle.py` builds a separate, test-only copy with the
component-ID search corrected and those two scratch arrays sized for frame
components. Pixel reconstruction, prediction, entropy coding, quantization and
upsampling are unchanged. This is a **modified reference**, not evidence that the
unmodified release decodes these files. The native-precision matrix for one through
four components continues to use the original unmodified release.

The multicomponent matrix compares every sample for counts 5–10, 8/12-bit DCT,
sequential/progressive Huffman and arithmetic coding, horizontal sampling factors
1–4 and vertical 2:1 layouts, lossless precisions 2–16, all seven predictors and point transforms
0, precision/2 and precision-1. Known-value fixtures independently check component
order and quantization-slot redefinition. The reference implementation is never
linked into the shipped library.

The proposed encoding work is scoped in [JPEG encoding extensions](jpeg-encoding-extensions.md).

### Coverage gate

`just multicomponent-coverage` runs library/native tests, both reference matrices,
and the isolated allocation-failure test, then writes LCOV to
`ZENJPEG_COVERAGE_REPORT`. `just multicomponent-coverage-gate BASE REPORT` requires
100% of changed executable production lines, without excluding error paths.
CI runs the gate against the pull request's base commit. Codecov's patch target
is also 100%, with zero tolerance. This is line coverage, not branch coverage.

The allocation subprocess injects one exact-size allocation failure using a
Linux-only C interposer; Cargo, rustc and normal test processes never load it.
It caught and verifies the correction to `try_with_capacity`, which previously
allocated infallibly before attempting its fallible reservation.

Fresh local coverage on 2026-10-26 against parent `12f25aad`: **113/113 changed
executable production lines covered**. The run included the reference matrices
and the allocation-failure subprocess. This does not claim 100% branch coverage
or 100% coverage of every pre-existing decoder line.

## Cancellation and hardening

`scanline_reader_with_pulse(data, Option<&dyn how_far::Pulse>)` borrows optional
cancellation and progress for the reader's lifetime. `decode_with_pulse` takes
the same optional control for collected output. `None` adds no per-call control;
`Some(pulse)` retains the supplied reporting and cancellation. Existing entry
points remain available. Header walks,
entropy decoding, native reconstruction, crop skipping and parallel wave work
poll the caller's cancellation token and preserve its `StopReason`, including
`TimedOut`. Reader progress advances by completed output rows; collected buffered
output reports its final height. The decoder does not finish a borrowed pulse.

`tests/native_hardening.rs` checks every observed cancellation checkpoint for the
minimal native fixtures, borrowed/owned reader parity, poisoned stride padding,
truncated prefixes, malformed marker fields, native output formats and rejected
processing combinations. It also checks cancellation inside RGB and planar wave
decoding. Unsupported native deblocking, dequantization bias and ICC conversion
are errors rather than silently ignored settings.

The minimal JPEG fixtures live in `zenjpeg/fuzz/seeds/native`. Their generation
arguments, reference identity, JPEG hashes and independently decoded sample
hashes are recorded in `zenjpeg/tests/support/native_manifest.json`.
`scripts/native-fuzz-seeds.py` verifies reference version 3.2.0 and refuses to
replace an existing fixture with different bytes. Larger evolving corpora and
crash artifacts remain outside git. The `fuzz_native_components` target shares
its invariants with stable regression tests through `fuzz/component_checks.rs`.

`just native-branch-coverage` measures LLVM true/false outcomes with nightly and
both reference implementations. `just native-branch-gate BASE REPORT` requires
all measured changed-production branch outcomes; unit-test assertions are not
production branches. The intended base is the common base of the native-precision
and multicomponent PRs, not just the latest hardening commit. A passing line gate
alone does not establish branch completeness.

The Linux hardening recipes use GNU `timeout`: ten minutes for stable checks,
fifteen minutes per instrumented test command, and the requested fuzz duration
plus two minutes for startup. Each has a fifteen-second termination grace;
libFuzzer separately limits an individual input to ten seconds. Preserve logs
and artifacts from deadline failures and report them as incomplete runs.

### Panic recovery compatibility

Pulse and Stop callbacks remain unrestricted. Ordinary callers need no new
bounds. Because a reader can retain a caller's callback, `ScanlineReader` and
`UltraHdrReader` no longer automatically implement `UnwindSafe` or
`RefUnwindSafe`; containing types can inherit that change. Send and Sync are
retained. Callers recovering from panics with `catch_unwind` must review their
recovery boundary and may choose a narrowly scoped `AssertUnwindSafe`. A mutable
reader reference already required that consideration before this change.
These marker traits do not prevent panics or establish memory safety, and
`forbid(unsafe_code)` remains in force. No how-far or enough trait changes are
required; passing `FnStop` through `&dyn Stop` or `&dyn Pulse` keeps its existing
contracts.

### Bounded fuzz validation (2026-10-27)

The native-component ASan/libFuzzer run completed 274,893 executions in 301
seconds without a crash artifact. It started with the curated fixtures and the
preserved exploratory corpus, with a ten-second input timeout, 1 GiB fuzzer
RSS limit and 16 KiB maximum input length. The reference fixture set contains
47 JPEGs totaling 41,316 bytes. This bounded run is evidence from the exercised
inputs, not a claim that fuzzing proves correctness.

### Complete changed-code coverage (2026-10-27)

Measured from stack base `e82a4b4052776387b0d0b1d476fe0f261d0c737b`:
**564/564 changed production branch outcomes** and **1,898/1,898 changed
executable lines**. The line gate also includes executable unit-test lines in
changed source files; the branch gate excludes test assertions. No production
error paths are excluded. This covers the changes in both stacked PRs, not every
pre-existing decoder line or every possible input.

The profiles were rebuilt from scratch, using committed tests, the two pinned
reference matrices, an isolated allocation-failure subprocess and existing
buffered/callback/crop regressions. Two final row-conversion/crop assertions were
then instrumented without changing production code. The evolving fuzz corpus was
not needed for either gate. `just native-branch-coverage` includes these routes.

A follow-up ASan/libFuzzer run against the final implementation completed
47,457 executions in 61 seconds without a crash artifact, using the same limits.
The corpus was preserved after both runs.

`native_fixture_raster_hashes` checks all 47 committed fixtures against the
reference-decoded raster SHA-256, independently for borrowed and owned readers.
The canonical raster is row-major, frame-component order, with each native
sample serialized as big-endian u16. Reads use padded strides and verify that
padding is untouched. `native_fixture_truncated_prefixes` is a separate test, so
raster correctness remains an explicit gate independent of fuzz-style checks.
