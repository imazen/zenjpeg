# Encoder diagnostics viewer

Explore an encode with source/encoded/difference views, AQ strengths, and
per-component quantization tables. **Edit settings** opens the encoder controls;
**Quant tables & zero bias** opens the per-component editor. Select a cell to
edit it in a popover. The frequency sliders update the utilization estimate
and, with auto-encode enabled, schedule a new encode.

Auto color, sampling and scan choices use the codec's defaults; they do not
inspect the image. The native encoder requires a sampling choice and does not
change it with quality. The visible encoder plan comes from the same resolved
configuration used for the encode and shows quality, distance, tables, sampling,
scan, Huffman and trellis choices. Auto optimize calls the codec's quality- and
color-dependent optimizer; explicit scan selections still take precedence. XYB uses its own default B sampling. The sequential + RST
choice uses the codec's default restart spacing, which can yield no markers for
a small image. Restart markers may permit parallel decoding; the browser encoder
is single-threaded. Progressive scans require optimized Huffman tables.
Standard, fixed-lambda Trellis and AQ-coupled Hybrid are explicit viewer modes;
the encoder's full preset interface is not exposed.

**Regenerate from current settings** freezes the selected configuration and
sweeps quality 1–100. Edited quant-table multipliers and zero-bias adjustments
are reapplied to freshly generated tables at each quality. Choose SSIMULACRA 2,
zensim, or Butteraugli max-norm distance (lower is better; images must be at
least 8×8). Scores assume sRGB pixels. The white curve is a single-configuration
sweep, not a multi-configuration Pareto frontier; the blue point is the latest
encode. The displayed delta uses the sweep's sampled envelope and is omitted
outside its measured byte range.

Use arrow keys to move between settings tabs and Escape to close a popover.
The zoom inspector opens by clicking an image or pressing Enter when it is focused.

The viewer uses `__diagnostics`, an unstable feature intended for small images.
It keeps every captured block in memory. Pre-quantization coefficients are in
natural order and levels are in JPEG zigzag order. AQ is the input strength,
not a multiplier. The local zero-rate estimate uses the captured zero bias and
AQ strength; trellis and DC refinement can change the final coefficients.
Entropy attribution and scan-script capture are not implemented.

With Rust and the `wasm32-unknown-unknown` target, wasm-pack 0.15, and Node
24 installed:

```sh
just diagnostics-wasm
cd zenjpeg-diagnostics-viewer/web
npm ci
npm run typecheck
npm run dev
```

The default synthetic image and uploads need no network. JPEG input ignores alpha; the source pane displays the same RGB values sent to the encoder. To cache optional
corpus PNGs, run `npm run fetch-images` before building. Those sources may have
prior compression; large demo images are scaled in the browser, so they are
not a lossless source-quality reference. The build does not download images.

For the browser checks, install Chromium with
`npx playwright install chromium --only-shell`, then run `npm run test:e2e`.
`just diagnostics-test` checks the Rust capture against decoded coefficients
and tests that capture leaves the JPEG unchanged.
