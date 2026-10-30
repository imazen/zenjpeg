# Encoder diagnostics viewer

Explore an encode with a source/encoded/difference triptych, per-block AQ
strengths, and per-component quantization tables. The frequency sliders update
the utilization estimate locally; when auto-encode is enabled, they also
schedule a new encode. A reference quality sweep and two perceptual scores are
shown for comparison. Open **Edit settings** to adjust options while viewing the images. The first
panel selects Standard, fixed-lambda Trellis, or AQ-coupled Hybrid, and exposes
quality, color path, and its sampling choices. Mode-specific settings appear
only for the chosen mode. AQ is a separate switch; with AQ off, Hybrid receives
zero block AQ strengths. The viewer starts at quality 85, YCbCr 4:4:4, Standard
mode with AQ and progressive scans; it uses encoder-generated tables until
they are edited. These are the viewer’s explicit controls, not the encoder’s
complete optimization preset or automatic-optimization interface. Progressive
scans always use optimized Huffman tables; under sequential scans, the viewer
lets you choose between optimized and fixed tables. In **Quant
tables**, choose a component and click a cell to edit it in a popover. Use the
arrow keys to move between settings tabs and Escape to close a popover. The zoom
inspector opens by clicking an image or pressing Enter when it is focused.

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
