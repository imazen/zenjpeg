# zenjpeg-diagnostics-viewer

Interactive web UI for inspecting zenjpeg encode decisions. Wraps the
`__diagnostics` capture surface in wasm, drives every encoder dial
from a tabbed UI, and renders side-by-side source / encoded / delta
views with per-block AQ heatmaps, per-component quant-table editors,
and a reference RD curve.

This document is a handoff for a fresh maintainer. Read it before
touching anything — most of the surprises in this codebase are the
result of either the JPEG / DCT semantics or the WASM / browser
boundary, and the gotchas section catalogs every one I hit.

## Repository layout

```
zenjpeg-diagnostics-viewer/
├── README.md            ← you are here
├── CLAUDE.md            ← rules for Claude Code / agent sessions
├── wasm/
│   ├── Cargo.toml       ← wasm-bindgen crate; pulls in zenjpeg,
│   │                      fast-ssim2, zensim
│   └── src/lib.rs       ← encodeWithDiagnostics + computeSsimulacra2
│                          + computeZensim exports
└── web/
    ├── index.html       ← single-page tabbed UI
    ├── package.json     ← npm scripts; "prebuild" hook runs the
    │                      build-time image fetch + wash
    ├── vite.config.ts   ← Tailwind v4 plugin + worker.format=es
    ├── tsconfig.json
    ├── playwright.config.ts
    ├── scripts/
    │   ├── fetch-images.ts    ← prebuild: fetches + 8× lanczos3 wash
    │   └── image-list.json    ← source manifest (mirrors src/)
    ├── src/
    │   ├── main.ts            ← UI controller (vanilla TS)
    │   ├── encode-worker.ts   ← Web Worker that runs wasm encode
    │   ├── types.ts           ← TS mirror of wasm EncodeOptions
    │   ├── styles.css         ← @import "tailwindcss" + @theme +
    │   │                        component carve-outs
    │   ├── heatmap.ts         ← AQ / utilization heatmap drawer
    │   ├── synthetic.ts       ← 64×64 default test pattern
    │   └── image-list.json    ← bundled into JS so the picker is
    │                            populated even without prebuild
    ├── wasm-pkg/              ← wasm-pack output (gitignored)
    └── public/images/         ← prebuild output (gitignored)
        ├── manifest.json
        ├── <sha>.png          ← raw blobs (cached for re-wash)
        └── <sha>-w8.png       ← 8× lanczos3 washed PNGs (served)

tests/
└── viewer.spec.ts             ← Playwright e2e
```

## Architecture

### Top-down data flow

```
                ┌─────────────────────────────────────────────┐
                │ user clicks / sliders / file upload         │
                └─────────────────┬───────────────────────────┘
                                  │
            scheduleAutoEncode() ─┤  (250 ms debounce)
                                  ▼
                       ┌──────────────────┐
                       │ runEncode(reqId) │
                       │  bumps latestReq │
                       └────────┬─────────┘
                                │  postMessage(EncodeReq)
                                ▼
                       ┌─────────────────────────────────┐
                       │ Web Worker (encode-worker.ts)   │
                       │  encodeWithDiagnostics(...)     │
                       │  → EncodeResult { bytes, diag } │
                       └────────┬────────────────────────┘
                                │  postMessage (transferable)
                                ▼
                       ┌─────────────────────────────────┐
                       │ main thread:                    │
                       │  if (reqId !== latestReqId)     │
                       │      drop                       │
                       │  else:                          │
                       │      decode JPEG → ImageData    │
                       │      ssim2(), zensim()          │
                       │      render triptych +          │
                       │        AQ heatmap + per-comp    │
                       │        editors + readout        │
                       │      schedule next pareto step  │
                       └─────────────────────────────────┘
```

### Wasm surface (`wasm/src/lib.rs`)

Exports:

- `encodeWithDiagnostics(pixels, w, h, options)` — RGB packed bytes
  in, returns `{ bytes: Uint8Array, diagnostics: Diagnostics }`.
  `options` is the `EncodeOptions` struct mirrored on both sides.
- `computeSsimulacra2(srcRgb, dstRgb, w, h)` — f64 SSIMULACRA 2 via
  `fast-ssim2 0.8.0` (default-features off — no rayon, no imgref).
- `computeZensim(srcRgb, dstRgb, w, h)` — f64 zensim score via
  `zensim 0.2.7` (default-features off — no rayon, no avx512).

`EncodeOptions` exposes every dial on `EncoderConfig` plus
`CustomQuantTables` (full f32 quant + zero_bias_mul + per-component
zero_bias_offset_dc/ac). When custom tables are present, the wasm
flips `EncodingTables.scaling = ScalingParams::Exact` so the user's
f32 numbers are used directly as final quant values (instead of
re-multiplying by `distance_to_scale × global_scale`).

### Web Worker (`web/src/encode-worker.ts`)

The wasm module is initialized inside a Worker so the main thread
stays responsive during slider drags. Each encode carries a unique
`reqId`; the main thread compares against `latestReqId` and drops
results from superseded requests. That's our cancellation model — see
the gotchas section for why we don't do mid-encode preemption.

The worker postMessages results back with the JPEG bytes as a
`Transferable`, so the buffer is moved (not copied) across the
boundary.

### Main thread (`web/src/main.ts`)

Single-file vanilla TS state machine. Persistent state:

- `currentImage`: source RGBA + dims + key
- `currentDiagnostics`: most-recent encoder diagnostics
- `lastEncode`: bytes / ssim2 / zensim / time / mode (for the
  big-typography readout panel)
- `eqStates`: per-component `{ hEq[8], vEq[8], cellOverrides: Map }`.
  Persists across encodes — only cleared by the per-component "Reset
  EQ" button or the Quant tab's "Reset all components" / "Reset
  zero-bias to defaults".
- `baseSnapshot`: the encoder's *source-derived* f32 quant tables,
  captured from the diagnostics struct ONLY on encodes that ran
  *without* customQuantTables. All user EQ / cell-override edits are
  multipliers on this pinned base. See "Q-scale compounding" gotcha.
- `zeroBias`: `globalMul[3]`, `offsetDc[3]`, `offsetAc[3]`. Sent in
  CustomQuantTables when any deviates from defaults.
- `paretoCache` / `paretoQueues`: per-image reference RD data.
  Computed progressively (one q at a time) interleaved with user
  encodes.
- `zoomImageData`: cached source / encoded / delta ImageData for the
  zoom inspector modal.

## Build & run

### Prereqs

- Rust 1.93+ (2024 edition)
- `wasm-pack` (cargo install wasm-pack)
- Node 18+ (for built-in fetch in scripts/)
- npm install in `web/` will pull in sharp's native binding

### Local dev

```bash
# from repo root
cd zenjpeg-diagnostics-viewer

# 1. Build the wasm bindings
just diagnostics-wasm
# or:
cd wasm && wasm-pack build --target web --out-dir ../web/wasm-pkg --release

# 2. Install web deps + run dev server
cd ../web
npm install
npm run dev          # ← runs `predev` → fetch-images.ts → vite
```

`npm run dev` opens at <http://localhost:3173> (port-locked so the
demo URL is stable). The `predev` hook downloads the picker corpus
from R2 and writes the manifest. Set `DIAGNOSTICS_VIEWER_OFFLINE=1`
to skip the network and reuse what's already cached.

### Production build

```bash
cd web
npm run build        # ← runs `prebuild` → fetch-images.ts → vite build
```

Output lands in `web/dist/`. Serve with any static-file server.

### CI / GitHub Pages

The Pages workflow lives in `.github/workflows/diagnostics-ci.yml`.
**It MUST call `npm run build`, not `npx vite build` directly** — the
former runs the `prebuild` hook that bakes images, the latter does
not. Without baked images, the deployed page falls back to direct R2
fetches which hit CORS errors (R2 public buckets don't send
Access-Control-Allow-Origin by default).

## Gotchas catalog

This is the core of the handoff. Each item is a real bug I hit;
read them before touching the relevant area.

### Build / CI

1. **Pages CI must run `npm run build`, not `npx vite build`.** The
   `prebuild` hook (`npm run fetch-images`) bakes images into
   `dist/images/` so the deployed page serves them same-origin.
   Bypassing the hook leaves the picker entries with only their
   absolute R2 URLs, which fail CORS. The fix is in the workflow,
   not in code.

2. **R2 public buckets don't send CORS headers.** If you point
   `DIAGNOSTICS_VIEWER_IMAGE_BASE` at an R2 bucket and expect the
   browser to fetch directly, configure CORS on the bucket
   (Cloudflare dashboard → R2 → bucket → Settings → CORS Policy):
   ```json
   [{"AllowedOrigins":["*"],"AllowedMethods":["GET","HEAD"],
     "AllowedHeaders":["*"],"MaxAgeSeconds":86400}]
   ```
   Or just rely on the prebuild bake (recommended for static deploys).

3. **wasm-pack vs cargo build.** `cargo build --target
   wasm32-unknown-unknown -p zenjpeg-diagnostics-wasm` compiles the
   crate but doesn't regenerate the JS bindings. To pick up wasm
   surface changes in the web app, you must run `wasm-pack build`.
   The `just diagnostics-wasm` target does this.

4. **Two `image-list.json` files, kept in sync.**
   - `web/scripts/image-list.json` — read by `fetch-images.ts` at
     prebuild time.
   - `web/src/image-list.json` — bundled into the JS so the picker
     dropdown is populated even when the prebuild hasn't run.
   When you add/edit an entry, update both.

5. **Sharp is a devDep.** It's only used by `fetch-images.ts` at
   build time. Doesn't ship to users. Native bindings come down
   automatically via `npm install`. If install fails (musl libc,
   etc.) sharp's docs cover the platform fallbacks.

### Source-image discipline

6. **PNGs only in the picker corpus.** The codec-corpus's
   `corpus/png-24-32` label doesn't track provenance — any of these
   PNGs may have been a JPEG → PNG conversion. Encoding them
   directly produces "encoder + prior-codec artifacts" output that
   the user can't disentangle from "encoder alone".

7. **8× lanczos3 wash at build time, not runtime.** The wash lives
   in `scripts/fetch-images.ts` via sharp:
   ```ts
   await sharp(rawPath)
     .resize(w, h, { kernel: "lanczos3" })
     .png({ compressionLevel: 9, palette: false })
     .toFile(outPath);
   ```
   8× decimation eliminates the original 8×8 DCT block grid. Lanczos3
   is sharper than Mitchell but with controlled overshoot — the
   right kernel when the goal is full block elimination.

8. **No browser-side resampling, anywhere.** `decodeImageToCanvas`
   reads pixels at native dimensions with `imageSmoothingEnabled =
   false`. Browser smoothing kernels are implementation-defined; we
   want known-good behavior for compression research.

9. **Lossy file uploads pass through unwashed.** If the user
   uploads a JPEG / WebP / AVIF, we surface a `⚠ lossy source —
   pre-process for clean diagnostics` warning in the status line and
   use the file as-is. User responsibility — we don't silently
   re-process with a wrong-kernel browser smoothing.

10. **The wash output is named `<sha>-w8.png`.** Raw blobs cached
    side-by-side as `<sha>.<ext>`. The runtime manifest's `url`
    field points at the wash output. Keeping the raw lets you re-wash
    with a different kernel without re-downloading.

### Encoder-side wasm surface

11. **`PerComponent` fields are `c0/c1/c2`.** NOT `y_or_x /
    cb_or_y / cr_or_b` — those are type aliases that don't exist as
    field names. I lost an hour to this.

12. **`TrellisSpeedMode` variants are `Thorough`, `Adaptive`,
    `Level(u8)`.** There is no `Fast` variant. Map JS "fast" string
    to `Level(8)` (the highest speed level).

13. **`ScalingParams::Exact` when `customQuantTables` is set.** The
    default `ScalingParams::Scaled { global_scale, frequency_exponents
    }` re-multiplies the table by `distance_to_scale × global_scale`
    at encode time. Pass user f32 tables expecting them to be final
    values, leave scaling on default → exploding values.

14. **Encoder mode value `"baseline"` is the fall-through default.**
    Wasm matches `"hybrid"` and `"trellis"` explicitly; anything else
    (including unknown values) is treated as `"baseline"` (no trellis,
    no hybrid). The JS UI label says "Adaptive Quantization" but the
    wire value stayed as `"baseline"` — see naming gotcha below.

15. **`default_xyb` vs `default_ycbcr`.** Pick the right
    `EncodingTables::default_*()` based on `colorPath`. Using
    `default_ycbcr` for an XYB encode produces nonsense.

16. **`fast-ssim2 0.8.0` and `zensim 0.2.7` MUST disable default
    features for wasm.** rayon (threads) and avx512 don't compile
    to `wasm32-unknown-unknown`. Use:
    ```toml
    fast-ssim2 = { version = "0.8.0", default-features = false }
    zensim = { version = "0.2.7", default-features = false }
    ```

17. **`wasm-bindgen` 0.2.x cell behavior.** Returning structs via
    serde-wasm-bindgen serializes `Vec<u8>` as a JS Array<number>,
    not a `Uint8Array`. We hand-pack the encode result via
    `js_sys::Uint8Array::new_with_length(...).copy_from(...)`.

### JS / state model

18. **Quant table compounding.** This is the trickiest one. The
    diagnostics struct returns `quant_table_base` as the *current*
    table — which after a custom-tables encode reflects the
    user-customized values. If you read base from live diagnostics
    and apply user multipliers on top, every edit compounds against
    the previous one. Q-scale 0.5 once → halved; Q-scale 0.5 again
    on a different control change → quartered; etc.

    Fix: cache `baseSnapshot` from the diagnostics struct ONLY on
    encodes that ran *without* customQuantTables. User multipliers
    apply to the pinned snapshot. Snapshot invalidates on
    `quality / colorPath / subsampling / xybSubsampling` changes.

    To pick up the new defaults after an axis change *while you have
    EQ edits in place*, hit "Reset all components" — that clears EQ,
    so the next encode runs without custom tables, refreshing the
    snapshot.

19. **`eqStates.clear()` is forbidden in the encode path.** Used to
    be at the top of `runEncode`, which wiped slider state on every
    encode and made the editor useless. The bug is gone; if you
    re-introduce something like it, the editor breaks immediately.

20. **`baseSnapshot` is "trust the most recent no-custom encode".**
    The snapshot picks up new defaults whenever the user runs an
    encode without custom tables. Reset EQ + bump quality + run
    encode = snapshot updated for the new quality.

21. **`buildCustomQuantTables` returns `null` when nothing differs
    from defaults.** Don't send a customQuantTables payload that
    contains the encoder's own defaults — it forces
    `ScalingParams::Exact` which disables quality-based scaling for
    no benefit. Only send when the user has actually edited.

22. **`reqId` cancellation, not preemption.** True mid-encode
    preemption requires `SharedArrayBuffer` + COOP/COEP headers (so
    we can flip an atomic flag visible to the worker's wasm). We
    don't ship those. Each encode has a `reqId`; main thread tracks
    `latestReqId`; results from superseded requests are dropped.
    The "Cancel" button just bumps `latestReqId` and hides the
    overlay — the worker keeps running but its output is discarded.

23. **Worker queue is FIFO.** Pareto sweep submits one encode at a
    time, kicked from `runEncode`'s `finally` hook, so user encodes
    naturally interleave. If you ever try to fire all 100 pareto
    encodes upfront, user-driven encodes will queue behind them and
    feel frozen.

24. **JSON imports require `resolveJsonModule: true` in tsconfig.**
    Already set. `import bundledImageList from "./image-list.json"`
    is a Vite/TS feature.

25. **Worker module format.** `vite.config.ts` has
    `worker: { format: "es" }`. Without it, the worker won't be a
    proper ES module and the wasm-pack output won't import
    correctly.

26. **Vite's `predev` and `prebuild` are npm-script lifecycle
    hooks.** Just running `vite` directly bypasses them. Same goes
    for `vite build` vs `npm run build`. Always use `npm run`.

27. **Manifest path is relative.** `./images/manifest.json` (and
    every entry's `url`). With Vite's `base: "./"`, this resolves
    correctly at any deploy subpath: localhost:3173/, GitHub Pages
    at `/zenjpeg/diagnostics/`, Cloudflare Pages, custom CDNs.

### UI / naming

28. **Mode picker label `Adaptive Quantization`, value `baseline`.**
    The JPEG term "baseline" specifically means baseline-scan vs
    progressive-scan, which is now a separate checkbox. To
    disambiguate, the picker shows "Adaptive Quantization" but the
    wire value is still `"baseline"` (kept for back-compat with
    Playwright's `data-testid="mode-baseline"`).

29. **"Pareto" is a misnomer.** What we draw is a single-config
    q-sweep at fixed `(4:2:0, AQ on, default tables)`. A real Pareto
    envelope would also vary subsampling, mode, and other knobs to
    find the upper-left frontier. UI labels were renamed to
    "Reference RD" — keep them that way unless you actually
    implement the multi-config sweep.

30. **AQ display in XYB.** AQ multiplier is computed from the Y-XYB
    channel (component[1]) but written to component[0] (X). The
    heatmap reads from c0 and shows X-block multipliers driven by
    Y-XYB analysis. The stats row labels the channel as "X (driven
    by Y-XYB)" so the meaning is explicit.

31. **Sharp YUV grey-out depends on subsampling.** Sharp YUV is
    only meaningful for YCbCr non-4:4:4. The grey-out is wired in
    `applyIgnoredAnnotations()` and fired from both the colorPath
    AND the subsampling change handlers. If you forget to fire it
    on subsampling change, Sharp YUV stays at its boot state.

32. **Tab panels use `data-testid` attributes.** Playwright tests
    in `tests/viewer.spec.ts` rely on these. Don't rename a testid
    without updating the spec — even if the visible label changes.

33. **The `body[data-encoding="true"]` flag.** Set in
    `setEncodingIndicator(true)` during in-flight encode; CSS uses
    it to dim the encoded/delta canvases. Cleared in the encode
    `finally` block (only when `inflightReqId === reqId` — i.e. only
    when WE are the latest request).

34. **Zoom inspector ImageData cache.** The modal redraws from
    `zoomImageData` (cached in `runEncode`), not by re-extracting
    from the on-screen canvases. While the modal is open, every
    successful encode re-populates the cache and triggers
    `drawZoomFrame()`, so live slider drags update the zoomed view.

### Misc

35. **Encoder progressive default is ON.** The HTML defaults
    `#progressive` to `checked`; `currentOptions()` reads it and
    sends `progressive: true`. Matches modern jpegli defaults
    (smaller files at high q with negligible cost). Wasm-side
    `EncodeOptions::default()` still defaults to `false` for safety
    — the JS HTML-default wins when the page is open.

36. **`__diagnostics` feature must be enabled on zenjpeg.** Already
    set in `wasm/Cargo.toml`. If you fork or copy the crate, this
    is the surface that produces the `Diagnostics` struct returned
    via `finish_with_diagnostics()`.

37. **Raw blobs vs washed PNGs both cached.** `public/images/`
    holds both `<sha>.<ext>` (the raw) and `<sha>-w8.png` (the
    wash). The runtime only reads the wash. The raw is kept so
    re-running fetch-images.ts with a different kernel doesn't
    require re-downloading.

## Test surface

```bash
# Unit + integration (in zenjpeg crate)
just diagnostics-test

# E2E (runs the wasm + the dev server through Playwright)
just diagnostics-e2e

# Full chain
just diagnostics-all
```

`tests/viewer.spec.ts` covers the synthetic 64×64 boot encode, AQ
heatmap rendering, EQ slider drag → utilization repaint, color path
toggle, subsampling → block-grid dimensions, quality slider →
encoded-byte monotonicity, Reset EQ.

## Open work / ideas

These are TODO items I considered but didn't ship. Pick up if
useful:

1. **True multi-config Pareto.** Sweep `{4:4:4, 4:2:0} × {AQ,
   trellis} × q=1..100` (~400 encodes per image), keep the
   upper-left envelope. Justifies the "Pareto" name. Cost is real;
   would interleave the same way the current single-config sweep
   does. Rename UI labels back to "Pareto" when this lands.

2. **True mid-encode cancellation.** Add SharedArrayBuffer-backed
   atomic flag, expose to the wasm encoder via a custom `Stop`
   impl, deploy with COOP/COEP headers. Then "Cancel" actually
   stops the worker.

3. **Per-component f32 base table from diagnostics.** The
   diagnostics struct currently exposes `quant_table_base: [u16;
   64]` (the JPEG DQT-format integer values). The encoder
   internally uses f32 (`EncodingTables.quant: PerComponent<[f32;
   64]>`). Surfacing the f32 base via the diagnostics API would let
   the viewer show full f32 precision in the cells without the
   round-trip through u16.

4. **`zero_bias_mul` per-cell editing.** Currently we expose a
   global multiplier per component (1 number ×3) plus DC/AC
   offsets. The wasm CustomQuantTables already supports per-cell
   `zero_bias_mul: Vec<f32>` (length 64). UI could surface this as
   another 8×8 grid editor.

5. **Curve editor for `frequency_exponents`.** The
   `ScalingParams::Scaled.frequency_exponents` array is one of the
   harder-to-tune knobs in the encoder. A curve editor that drove
   it directly (rather than via the current EQ × cell-overrides
   approach) might be useful for research.

6. **Component selector in the Quant tab.** When XYB is active, the
   labels say "Y/X" / "Cb/Y-XYB" / "Cr/B" but the user might want
   to focus on one component at a time. Could collapse/expand per
   component.

7. **Image upload wash via wasm.** We have zenresize compiled to
   wasm in some sibling projects. Could add a "wash this upload"
   button that calls a wasm export. Not done because file uploads
   are user-driven and the warning is enough.

## Asking around

- **R2 codec-corpus**: `https://pub-7c5c57fd3e0842f0b147946928891d40.r2.dev`,
  blobs at `/blobs/<sha[0:2]>/<sha[2:4]>/<sha>`, manifest at
  `/manifest.jsonl`.
- **zenjpeg encoder**: `../zenjpeg/`. The `__diagnostics` feature
  produces the `Diagnostics` struct; see `zenjpeg/src/encode/
  diagnostics.rs` for the shape.
- **fast-ssim2** (SSIMULACRA 2): `~/work/zen/fast-ssim2/`.
- **zensim** (our perceptual metric): `~/work/zen/zensim/`.

## Style

- Vanilla TS, no JS framework. State lives in module-level mutable
  refs. The flow is small enough that a framework would be
  ceremony.
- Tailwind v4 via `@tailwindcss/vite`. `styles.css` is a single
  `@import "tailwindcss"` + `@theme` block + a handful of
  component-specific carve-outs (quant grid, EQ sliders, metric
  readouts, encoding overlay, zoom inspector). Tailwind utility
  classes everywhere else in the HTML.
- One-line file headers explaining what the file is. No
  paragraph-long docstrings.
- No emoji in source. The "✓" and "—" in the UI are typographic
  marks, not emoji.
