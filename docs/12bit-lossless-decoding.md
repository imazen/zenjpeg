# Native-precision JPEG decoding

Work in progress. Twelve-bit DCT and predictive lossless decoding are not yet
available through the public decoder. The acceptance criterion is sample-for-sample
agreement with libjpeg-turbo, including scanline reads.

## Proposed public surface

The existing decoder exposes u8 and f32 output. Native integer output would add:

```rust,ignore
OutputTarget::Native16
DecodeResult::pixels_u16(&self) -> Option<&[u16]>
DecodeResult::sample_precision(&self) -> u8
ScanlineReader::sample_precision(&self) -> u8
ScanlineReader::read_rows_rgb16(&mut self, output: ImgRefMut<'_, rgb::RGB<u16>>) -> Result<usize>
ScanlineReader::read_rows_gray16(&mut self, output: ImgRefMut<'_, u16>) -> Result<usize>
```

These are proposals awaiting owner approval, not implemented API. Native samples
retain the frame's range: 0–4095 for 12-bit, rather than expanding to 0–65535.
Point transforms restore the low zero bits. RGB output retains the source RGB
primaries and transfer function unless color correction is explicitly requested;
JPEG YCbCr uses the BT.601 matrix. Gray16 is explicitly a single-channel format.
The result's existing pixel-format field specifies interleaved channel layout.

## Implementation boundaries

- Share the integer islow IDCT. The 12-bit specialization uses one first-pass
  guard bit, versus two for 8-bit. Existing 8-bit SIMD dispatch stays intact.
- Share marker parsing, Huffman tables, bit reading and restart handling.
  Predictive JPEG category 16 represents -32768 without magnitude bits;
  DCT difference coding must retain its separate category-16 behavior.
- Keep predictors and point transforms in a small lossless reconstruction module.
  Reuse row delivery and output formatting. Sequential sources need bounded row
  storage; progressive sources necessarily retain coefficients between scans.
- Preserve the 8-bit path's established rounding and tests. Do not feed 12-bit
  samples through SIMD kernels whose intermediate ranges assume 8-bit samples.

## Verification gate

Use pinned libjpeg-turbo 3.2.0 cjpeg/djpeg as test-only oracles. Generate fixtures
at test time, with explicit tool paths and errors when required tools are absent.
Compare native integer samples, not perceptual scores or tolerances.

Cover sequential/progressive 12-bit DCT, gray/RGB/YCbCr, supported sampling
geometries, non-MCU-aligned dimensions, flat-chroma and high-frequency content,
restart markers, high DC/AC categories, clipping and rounding boundaries. Cover
lossless precisions 2–16, predictors 1–7, point transforms, restart boundaries,
single- and multi-component scans, and the category-16 special case. Compare
strided/chunked/owned/borrowed reads and whole-result output; test truncation,
malformed tables and scan headers, cancellation and allocation limits.

## Validation of the preparatory changes

The shared IDCT passes the existing scalar/SIMD comparisons and an exhaustive
12-bit constant-block rounding/stride test. The shared difference decoder passes
signed category-boundary tests and a category-16 bit-consumption test. The default
library suite passes (1160 passed, one pre-existing ignored test), and library
Clippy passes with warnings denied. End-to-end 12-bit/lossless oracle parity has
not been measured: those paths are not wired yet.
