# JPEG encoding extensions: proposed scope

This is a design scope, not implemented encoding support. The decode stack adds
up to ten frame components. The current pixel encoder emits 8-bit frames;
coefficient transcoding supports 8/12-bit DCT with up to four components.

## Recommended order

1. Extend coefficient transcoding to ten components. This is independent of
   pixel encoding and reuses the existing entropy writers and transform kernels.
   Emit legal scan groups of at most four components, with separate-component
   scans where sampling would exceed the MCU block limit. Retain component IDs,
   sampling, source precision and per-component quantization tables. If more than
   four distinct quantization tables are needed, reuse DQT slots between scans;
   never serialize an internal table-list index as an out-of-range selector.
2. Add predictive lossless encoding (SOF3), 2–16 bits, predictors 1–7, point
   transforms and restart rows. Share the prediction formula with the decoder;
   reuse marker framing, bit output and Huffman construction. Category 16 needs
   the SOF3 rule, not the DCT difference writer. Start with native component data,
   without implicit color conversion. CMYK inversion and profile meaning remain
   explicit caller contracts.
3. Add 12-bit DCT pixel encoding. The current serializer writes precision 8, and
   strip conversion centers/scales around 128/255. Changing the SOF byte alone
   would be incorrect. Parameterize the sample range, level shift, quantization,
   entropy categories and input conversion while retaining the existing 8-bit
   path. Reuse the float DCT, quantizer and progressive writer. Do not silently
   enable 8-bit-calibrated adaptive quantization settings for 12-bit input.

## API and streaming decisions

New public encoding APIs require approval before implementation. Prefer a small
native-component configuration over separate RGB, CMYK and unknown-layout
encoders. Inputs must encode component count/order, precision, sampling and color
interpretation, with pixel stride and caller-owned output storage.

SOF3 can use one previous row per active component. For more than four components,
JPEG requires multiple scans. Two streaming choices need an explicit API decision:
component/scan-major input avoids retaining an entire image; interleaved row input
requires retaining encoded scan segments until they can be emitted in JPEG order.
Do not hide full-image sample buffering behind a row API. A first implementation
should use complete Huffman tables for one-pass encoding; optimized tables require
a replay/two-pass contract or bounded encoded spooling.

Arithmetic *DCT* encoding is a separate addition if desired. Arithmetic lossless,
differential/hierarchical JPEG, DNL emission and fractional subsampling are outside
this proposal. No line-count estimate is claimed for these changes; only the
previous 12-bit coefficient extension had a measured under-200-line result.

## Required verification gates

- 100% measured coverage of new/changed production code; publish uncovered lines
  and branches, including errors. Passing a format matrix is not a substitute
  for coverage, and line coverage is not proof of complete branch coverage.
- Sample-exact lossless round trips for all precisions, predictors, point
  transforms, component counts, scan groupings and restart boundaries, checked by
  an independent decoder and against the original input after the point transform.
- DCT outputs decoded independently, with quantized coefficients checked before
  and after transcoding. Pixel encoding need not reproduce libjpeg-turbo's bytes:
  the existing float DCT and quantizer are different algorithms.
- Nonconstant and DC-only content; tiny/partial MCUs; table sharing, all legal
  selectors, slot redefinition, scan order, short buffers, stride, cancellation,
  memory limits and invalid configuration.
- Existing 8-bit regression expectations unchanged. Cross-platform CI, fuzz target
  compilation and corpus regressions remain required. Any modified oracle must be
  labeled and its differences separated from pixel/entropy algorithms.

No encoder API, implementation or release is authorized merely by this scope.
