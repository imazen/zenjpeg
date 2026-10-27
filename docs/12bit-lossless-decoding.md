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
  gray, RGB, YCbCr, CMYK/YCCK and raw two-component images. Integral sampling
  ratios use the libjpeg interpolation rules, including 4:1:1, vertical 4:1,
  3:1 and asymmetric component sampling.
- Lossless one- through four-component images, including CMYK, predictors 1–7,
  point transforms, interleaved or separate component scans, and row-aligned
  restart intervals.
- Crops, pixel transforms, strided/chunked reads and metadata preservation.

Native decoding currently rejects XYB, more than four components, fractional
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
- Sequential/progressive 12-bit DCT, the four supported sampling layouts,
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
