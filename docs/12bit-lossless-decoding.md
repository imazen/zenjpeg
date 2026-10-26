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

## Supported modes and limits

- Twelve-bit sequential and progressive DCT, including arithmetic coding;
  grayscale, RGB and YCbCr; 4:4:4, 4:2:2, 4:2:0 and 4:4:0 sampling.
- Lossless gray and RGB, predictors 1–7, point transforms, interleaved or
  separate component scans, and row-aligned restart intervals.
- Crops, pixel transforms, strided/chunked reads and metadata preservation.

Native decoding currently rejects CMYK/XYB, other 12-bit sampling shapes,
subsampled lossless, lossless DNL, differential lossless and arithmetic lossless.
Deblocking, dequantization bias and ICC conversion are not supported for these
sources. Twelve-bit DCT always uses the libjpeg integer reconstruction.
Coefficient extraction/re-encoding remains restricted to 8-bit DCT images.

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
- Sequential/progressive 12-bit DCT, the four supported sampling layouts,
  tiny/partial-MCU dimensions, noise, constant blocks and checkerboards,
  quality 5/30/60/91/100, triangle and nearest upsampling, RGB and arithmetic.
- Native integers, normalized floats, u8 conversion, owned/borrowed strided
  reads, crops and all seven pixel transforms.

Ordinary tests cover constant streams across the lossless predictor/precision
matrix, category-16 consumption, tables before SOF, truncations, invalid scan
parameters, cancellation, memory limits and output padding. Existing decoder,
metadata and 8-bit dispatch tests remain in the regression gates.
