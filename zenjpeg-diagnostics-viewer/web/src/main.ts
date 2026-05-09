// Diagnostics viewer entry point.
//
// Architecture:
// - Wasm runs in a Web Worker. Main thread stays responsive.
// - Each encode is tagged with a unique `reqId`. When a result arrives
//   from the worker, we discard it if it isn't the latest request.
//   That's our "cancellation" — we don't actually preempt the worker
//   (would require SharedArrayBuffer + COOP/COEP) but stale results
//   are dropped, which is the UX we want for slider-driven re-encodes.
// - Auto-encode toggle + 250 ms debounce: any control change schedules
//   a re-encode after the slider settles.
// - Per-component EQ sliders + cell overrides persist across encodes.
//   They serialize into `customQuantTables` whenever any value diverges
//   from the source-derived defaults.
//
// The metric scoring (SSIMULACRA 2 + zensim) runs on the main thread
// after the encoded JPEG is decoded back to RGBA — saves us from
// shipping the source pixels into the worker just for scoring.

import init, {
  computeSsimulacra2,
  computeZensim,
} from "../wasm-pkg/zenjpeg_diagnostics_wasm.js";
import { drawHeatmap } from "./heatmap";
import { syntheticPattern } from "./synthetic";
import bundledImageList from "./image-list.json";
import type {
  ComponentDiagnostics,
  CustomQuantTables,
  Diagnostics,
  EncodeMode,
  EncodeOptions,
  TrellisSpeed,
} from "./types";
import type {
  EncodeReq,
  EncodeResultMsg,
  ErrorMsg,
  ReadyMsg,
} from "./encode-worker";

// ─── DOM helpers ──────────────────────────────────────────────────
const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`element not found: ${sel}`);
  return el as T;
};

// Status / actions
const status = $<HTMLParagraphElement>("#status");
const autoEncodeIn = $<HTMLInputElement>("#auto-encode");
const encodeBtn = $<HTMLButtonElement>("#encode");
const cancelBtn = $<HTMLButtonElement>("#cancel");
// Image bar
const fileIn = $<HTMLInputElement>("#image-input");
const imagePickSel = $<HTMLSelectElement>("#image-pick");
const imageInfoEl = $<HTMLElement>("#image-info");
const resetBtn = $<HTMLButtonElement>("#reset");
// Big readout
const mBytes = $<HTMLElement>("#m-bytes");
const mBpp = $<HTMLElement>("#m-bpp");
const mMode = $<HTMLElement>("#m-mode");
const mModeDetail = $<HTMLElement>("#m-mode-detail");
const mSsim2 = $<HTMLElement>("#m-ssim2");
const mZensim = $<HTMLElement>("#m-zensim");
const mPareto = $<HTMLElement>("#m-pareto");
const mTime = $<HTMLElement>("#m-time");
// Canvases
const sourceCanvas = $<HTMLCanvasElement>("#source-canvas");
const encodedCanvas = $<HTMLCanvasElement>("#encoded-canvas");
const deltaCanvas = $<HTMLCanvasElement>("#delta-canvas");
const aqCanvas = $<HTMLCanvasElement>("#aq-canvas");
const rdCanvas = $<HTMLCanvasElement>("#rd-canvas");
// AQ stats
const aqMinEl = $<HTMLElement>("#aq-min");
const aqMaxEl = $<HTMLElement>("#aq-max");
const aqMeanEl = $<HTMLElement>("#aq-mean");
const aqChannelEl = $<HTMLElement>("#aq-channel");
// Encoder controls
const qualityIn = $<HTMLInputElement>("#quality");
const qualityOut = $<HTMLOutputElement>("#quality-out");
const colorPathSel = $<HTMLSelectElement>("#color-path");
const subSel = $<HTMLSelectElement>("#subsampling");
const xybSubSel = $<HTMLSelectElement>("#xyb-subsampling");
const subLabel = $<HTMLLabelElement>("#subsampling-label");
const xybLabel = $<HTMLLabelElement>("#xyb-subsampling-label");
const aqIn = $<HTMLInputElement>("#aq-enabled");
const deringingIn = $<HTMLInputElement>("#deringing");
const sharpYuvIn = $<HTMLInputElement>("#sharp-yuv");
const preBlurIn = $<HTMLInputElement>("#pre-blur");
const chromaDistIn = $<HTMLInputElement>("#chroma-dist-scale");
const optimizeHuffmanIn = $<HTMLInputElement>("#optimize-huffman");
const progressiveIn = $<HTMLInputElement>("#progressive");
const restartMcuRowsIn = $<HTMLInputElement>("#restart-mcu-rows");
// Trellis
const trellisDcIn = $<HTMLInputElement>("#trellis-dc");
const trellisLambda1In = $<HTMLInputElement>("#trellis-lambda1");
const trellisLambda2In = $<HTMLInputElement>("#trellis-lambda2");
const trellisSpeedSel = $<HTMLSelectElement>("#trellis-speed");
const trellisDeltaDcIn = $<HTMLInputElement>("#trellis-delta-dc");
// Hybrid
const hybridAqLambdaIn = $<HTMLInputElement>("#hybrid-aq-lambda-scale");
const hybridBaseLambda1In = $<HTMLInputElement>("#hybrid-base-lambda1");
const hybridBaseLambda2In = $<HTMLInputElement>("#hybrid-base-lambda2");
const hybridDcIn = $<HTMLInputElement>("#hybrid-dc");
const hybridAqExpIn = $<HTMLInputElement>("#hybrid-aq-exp");
const hybridAqThresholdIn = $<HTMLInputElement>("#hybrid-aq-threshold");
const hybridQAdaptIn = $<HTMLInputElement>("#hybrid-q-adapt");
// Curve controls
const curveQscaleIn = $<HTMLInputElement>("#curve-qscale");
const curveQscaleOut = $<HTMLOutputElement>("#curve-qscale-out");
const curveTiltIn = $<HTMLInputElement>("#curve-tilt");
const curveTiltOut = $<HTMLOutputElement>("#curve-tilt-out");
const curveHvIn = $<HTMLInputElement>("#curve-hv");
const curveHvOut = $<HTMLOutputElement>("#curve-hv-out");
const curvePresetSel = $<HTMLSelectElement>("#curve-preset");
const curveResetBtn = $<HTMLButtonElement>("#curve-reset");

const compRoot = $<HTMLElement>("#component-0").parentElement!;

// ─── State ────────────────────────────────────────────────────────
interface ImageState {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  /** Stable per-image key for caches (pareto). */
  key: string;
}

interface ComponentEqState {
  hEq: number[];
  vEq: number[];
  cellOverrides: Map<number, number>;
}

interface BaseSnapshot {
  /** Per-component f32 base tables (length 64, natural row-major). */
  tables: [Float32Array, Float32Array, Float32Array];
  /** Encode opts the snapshot was captured under. Stale if any of
   *  these change. */
  quality: number;
  colorPath: string;
  subsampling: string;
  xybSubsampling: string;
}

interface ZeroBiasState {
  /** Global multiplier on zero_bias_mul[k] for all 64 cells. */
  globalMul: [number, number, number];
  /** Per-component DC offset. */
  offsetDc: [number, number, number];
  /** Per-component AC offset. */
  offsetAc: [number, number, number];
}

interface ParetoPoint {
  bytes: number;
  score: number;
  q: number;
}

interface ParetoQueue {
  /** Quality values still to encode, in evaluation order. */
  qs: number[];
  /** Points landed so far (unsorted; we sort on every redraw). */
  points: ParetoPoint[];
  /** Total q values in the original sweep (for progress %). */
  total: number;
  /** Set true when the queue is exhausted. */
  complete: boolean;
}

interface LastEncode {
  bytes: number;
  ssim2: number;
  zensim: number;
  durationMs: number;
  mode: EncodeMode;
  width: number;
  height: number;
}

let currentImage: ImageState = synthetic();
let currentDiagnostics: Diagnostics | null = null;
let lastEncode: LastEncode | null = null;
const eqStates = new Map<number, ComponentEqState>();
let baseSnapshot: BaseSnapshot | null = null;
// Cached ImageData per pane for the zoom inspector. Updated on each
// successful encode (in runEncode). Lets the zoom modal redraw at
// arbitrary scales/centers without re-extracting from the on-screen
// canvases (which getImageData() works for, but is slower at large
// scales and noisy when source/encoded/delta have different sizes).
let zoomImageData: {
  source: ImageData | null;
  encoded: ImageData | null;
  delta: ImageData | null;
} = { source: null, encoded: null, delta: null };
const zeroBias: ZeroBiasState = {
  globalMul: [1.0, 1.0, 1.0],
  offsetDc: [0.0, 0.0, 0.0],
  offsetAc: [0.0, 0.0, 0.0],
};
const paretoCache = new Map<string, ParetoPoint[]>();
const paretoQueues = new Map<string, ParetoQueue>();
let activeTab = "quality";

interface ImageManifestEntry {
  id: string;
  label: string;
  /** Local cached relative URL when prebuild ran. The build-time
   *  fetch-images.ts script applies an 8× lanczos3 wash to entries
   *  flagged wasJpeg, writes the result as a PNG named `<sha>-w8.png`,
   *  and points this URL at the washed copy. The browser does NO
   *  resampling at any point — pixels are taken straight from this
   *  URL into the encoder. */
  localUrl: string | null;
  /** Direct R2 URL (raw blob — only used as a fallback when the
   *  local cache is missing AND CORS happens to be configured). */
  remoteUrl: string;
}
let imageManifest: ImageManifestEntry[] = [];

// Source list typed as the bundled JSON shape.
interface BundledImageList {
  defaultBase?: string;
  images: Array<{
    id: string;
    label: string;
    sha256?: string;
    ext?: string;
    file?: string;
    wasJpeg?: boolean;
  }>;
}
const bundled = bundledImageList as BundledImageList;
const FALLBACK_BASE = "https://pub-7c5c57fd3e0842f0b147946928891d40.r2.dev";

function entryToUrls(
  entry: BundledImageList["images"][number],
  base: string,
): { localUrl: string | null; remoteUrl: string; localName: string } | null {
  if (entry.sha256) {
    const sha = entry.sha256.toLowerCase();
    if (!/^[0-9a-f]{64}$/.test(sha)) return null;
    const ext = entry.ext ?? "bin";
    const localName = `${sha}.${ext}`;
    return {
      localUrl: `./images/${localName}`,
      remoteUrl: `${base}/blobs/${sha.slice(0, 2)}/${sha.slice(2, 4)}/${sha}`,
      localName,
    };
  }
  if (entry.file) {
    return {
      localUrl: `./images/${entry.file}`,
      remoteUrl: `${base}/${entry.file}`,
      localName: entry.file,
    };
  }
  return null;
}

// ─── Worker setup ─────────────────────────────────────────────────
const worker = new Worker(new URL("./encode-worker.ts", import.meta.url), {
  type: "module",
});
let workerReady: Promise<void> = new Promise((resolve) => {
  const onReady = (ev: MessageEvent) => {
    const m = ev.data as ReadyMsg | EncodeResultMsg | ErrorMsg;
    if (m.kind === "ready") {
      worker.removeEventListener("message", onReady);
      resolve();
    }
  };
  worker.addEventListener("message", onReady);
});

let nextReqId = 1;
let inflightReqId = 0;
let latestReqId = 0;
const pendingResolvers = new Map<
  number,
  {
    resolve: (r: { bytes: Uint8Array; diagnostics: Diagnostics; durationMs: number }) => void;
    reject: (e: Error) => void;
  }
>();
worker.addEventListener("message", (ev: MessageEvent) => {
  const m = ev.data as EncodeResultMsg | ErrorMsg | ReadyMsg;
  if (m.kind === "result") {
    const slot = pendingResolvers.get(m.reqId);
    pendingResolvers.delete(m.reqId);
    if (slot)
      slot.resolve({
        bytes: m.bytes,
        diagnostics: m.diagnostics,
        durationMs: m.durationMs,
      });
  } else if (m.kind === "error") {
    const slot = pendingResolvers.get(m.reqId);
    pendingResolvers.delete(m.reqId);
    if (slot) slot.reject(new Error(m.message));
  }
});

function workerEncode(
  pixels: Uint8Array,
  width: number,
  height: number,
  options: EncodeOptions,
  reqId: number,
): Promise<{ bytes: Uint8Array; diagnostics: Diagnostics; durationMs: number }> {
  return new Promise((resolve, reject) => {
    pendingResolvers.set(reqId, { resolve, reject });
    const msg: EncodeReq = {
      kind: "encode",
      reqId,
      pixels,
      width,
      height,
      options,
      sourceRgb: pixels,
      scoreMetrics: false,
    };
    worker.postMessage(msg, [pixels.buffer]);
  });
}

// ─── Image helpers ────────────────────────────────────────────────
function synthetic(): ImageState {
  const w = 64;
  const h = 64;
  return { width: w, height: h, rgba: syntheticPattern(w, h), key: "__synthetic__" };
}

function drawRGBA(
  canvas: HTMLCanvasElement,
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
): void {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d unavailable");
  ctx.putImageData(new ImageData(rgba.slice(), w, h), 0, 0);
}

async function decodeJpeg(bytes: Uint8Array): Promise<ImageData> {
  const blob = new Blob([bytes.slice()], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = rej;
      i.src = url;
    });
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const cx = c.getContext("2d");
    if (!cx) throw new Error("canvas 2d unavailable");
    cx.drawImage(img, 0, 0);
    return cx.getImageData(0, 0, c.width, c.height);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function deltaImage(src: ImageData, dec: ImageData, gain = 8): ImageData {
  const w = src.width;
  const h = src.height;
  const out = new ImageData(w, h);
  for (let i = 0; i < w * h * 4; i += 4) {
    out.data[i] = Math.min(255, Math.abs(src.data[i]! - dec.data[i]!) * gain);
    out.data[i + 1] = Math.min(255, Math.abs(src.data[i + 1]! - dec.data[i + 1]!) * gain);
    out.data[i + 2] = Math.min(255, Math.abs(src.data[i + 2]! - dec.data[i + 2]!) * gain);
    out.data[i + 3] = 255;
  }
  return out;
}

function rgbaToPackedRgb(rgba: Uint8ClampedArray | Uint8Array): Uint8Array {
  const n = rgba.length / 4;
  const out = new Uint8Array(n * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    out[j] = rgba[i]!;
    out[j + 1] = rgba[i + 1]!;
    out[j + 2] = rgba[i + 2]!;
  }
  return out;
}

// ─── Options assembly ─────────────────────────────────────────────
function getMode(): EncodeMode {
  const checked = document.querySelector<HTMLInputElement>('input[name="mode"]:checked');
  return (checked?.value as EncodeMode | undefined) ?? "baseline";
}

/** Reset the cached base snapshot when any non-EQ encoder option changes
 *  that would alter the encoder-derived defaults. The snapshot will be
 *  re-captured on the next encode that runs *without* custom tables. */
function invalidateBaseSnapshotIfStale(): boolean {
  if (!baseSnapshot) return false;
  const optsMatch =
    baseSnapshot.quality === parseInt(qualityIn.value, 10) &&
    baseSnapshot.colorPath === colorPathSel.value &&
    baseSnapshot.subsampling === subSel.value &&
    baseSnapshot.xybSubsampling === xybSubSel.value;
  if (!optsMatch) {
    baseSnapshot = null;
    return true;
  }
  return false;
}

/** Pull a snapshot of the encoder's source-derived f32 base tables.
 *  Called from runEncode after a successful no-custom-tables encode.
 *  The diagnostics report u16 tables (DQT-format) so we round-trip
 *  through f32, but we cache here so subsequent custom edits are
 *  multipliers on a STABLE base — not the previous customized one. */
function captureBaseSnapshot(diag: Diagnostics): void {
  const tables: [Float32Array, Float32Array, Float32Array] = [
    new Float32Array(64),
    new Float32Array(64),
    new Float32Array(64),
  ];
  for (let cidx = 0; cidx < 3; cidx++) {
    const comp = diag.components[cidx];
    const t = tables[cidx]!;
    if (!comp) {
      t.fill(16);
      continue;
    }
    const base = comp.quantTableBase as Uint16Array | number[];
    for (let i = 0; i < 64; i++) t[i] = base[i]! as number;
  }
  baseSnapshot = {
    tables,
    quality: parseInt(qualityIn.value, 10),
    colorPath: colorPathSel.value,
    subsampling: subSel.value,
    xybSubsampling: xybSubSel.value,
  };
}

/** Build the f32 effective table for a component, using the cached
 *  base snapshot (NOT the live diagnostics, which may already reflect
 *  the previous custom-tables encode). */
function effectiveTableF32(componentIdx: number): Float32Array {
  const snap = baseSnapshot;
  const out = new Float32Array(64);
  if (!snap) {
    // No snapshot yet — fall back to live diag if available.
    const diag = currentDiagnostics;
    const comp = diag?.components[componentIdx];
    const base = comp?.quantTableBase as Uint16Array | number[] | undefined;
    if (base) {
      for (let i = 0; i < 64; i++) out[i] = base[i]! as number;
    } else {
      out.fill(16);
    }
    return out;
  }
  const base = snap.tables[componentIdx];
  if (!base) {
    out.fill(16);
    return out;
  }
  out.set(base);
  const eq = eqStates.get(componentIdx);
  if (!eq) return out;
  for (let v = 0; v < 8; v++) {
    for (let u = 0; u < 8; u++) {
      const i = v * 8 + u;
      let factor = eq.hEq[u]! * eq.vEq[v]!;
      const ovr = eq.cellOverrides.get(i);
      if (ovr !== undefined) factor *= ovr;
      out[i] = Math.max(1, base[i]! * factor);
    }
  }
  return out;
}

function buildCustomQuantTables(): CustomQuantTables | null {
  if (!baseSnapshot) return null;
  let anyDiff = false;
  const tables: number[][] = [];
  for (let cidx = 0; cidx < 3; cidx++) {
    const out = Array.from(effectiveTableF32(cidx));
    tables.push(out);
    const base = baseSnapshot.tables[cidx]!;
    for (let i = 0; i < 64; i++) {
      if (Math.abs(out[i]! - base[i]!) > 1e-3) {
        anyDiff = true;
        break;
      }
    }
    if (anyDiff) break;
  }
  // Zero-bias deviations from defaults also count as "user has edits".
  const zbDirty =
    zeroBias.globalMul.some((m) => Math.abs(m - 1.0) > 1e-3) ||
    zeroBias.offsetDc.some((d) => Math.abs(d) > 1e-4) ||
    zeroBias.offsetAc.some((d) => Math.abs(d) > 1e-4);
  if (!anyDiff && !zbDirty) return null;
  // Build zero-bias multiplier tables: each component has 64 cells; the
  // global mul applies to all of them. Default zero_bias_mul values
  // come from the encoder's per-mode defaults — we don't have those on
  // the JS side, so we approximate as 1.0×globalMul, leaving the
  // encoder to apply the rest of its baseline shape. (For finer
  // control we'd need to surface zero_bias_mul defaults via diagnostics.)
  const zbMul = (cidx: number): number[] | undefined => {
    const m = zeroBias.globalMul[cidx]!;
    if (Math.abs(m - 1.0) < 1e-3) return undefined;
    return new Array<number>(64).fill(m);
  };
  return {
    y: tables[0]!,
    cb: tables[1]!,
    cr: tables[2]!,
    yZeroBiasMul: zbMul(0),
    cbZeroBiasMul: zbMul(1),
    crZeroBiasMul: zbMul(2),
    zeroBiasOffsetDc: [
      zeroBias.offsetDc[0]!,
      zeroBias.offsetDc[1]!,
      zeroBias.offsetDc[2]!,
    ],
    zeroBiasOffsetAc: [
      zeroBias.offsetAc[0]!,
      zeroBias.offsetAc[1]!,
      zeroBias.offsetAc[2]!,
    ],
  };
}

function currentOptions(): EncodeOptions {
  const opts: EncodeOptions = {
    quality: parseInt(qualityIn.value, 10),
    colorPath: colorPathSel.value as "ycbcr" | "xyb",
    subsampling: subSel.value as EncodeOptions["subsampling"],
    xybSubsampling: xybSubSel.value as EncodeOptions["xybSubsampling"],
    aqEnabled: aqIn.checked,
    deringing: deringingIn.checked,
    optimizeHuffman: optimizeHuffmanIn.checked,
    progressive: progressiveIn.checked,
    sharpYuv: sharpYuvIn.checked,
    preBlur: parseFloat(preBlurIn.value) || 0,
    chromaDistanceScale: parseFloat(chromaDistIn.value) || 1,
    restartMcuRows: parseInt(restartMcuRowsIn.value, 10) || 0,
    mode: getMode(),
    trellisDcEnabled: trellisDcIn.checked,
    trellisLambdaLogScale1: parseFloat(trellisLambda1In.value) || 14.75,
    trellisLambdaLogScale2: parseFloat(trellisLambda2In.value) || 16.5,
    trellisSpeedMode: trellisSpeedSel.value as TrellisSpeed,
    trellisDeltaDcWeight: parseFloat(trellisDeltaDcIn.value) || 0,
    hybridAqLambdaScale: parseFloat(hybridAqLambdaIn.value) || 2,
    hybridBaseLambdaScale1: parseFloat(hybridBaseLambda1In.value) || 14.75,
    hybridBaseLambdaScale2: parseFloat(hybridBaseLambda2In.value) || 16.5,
    hybridDcEnabled: hybridDcIn.checked,
    hybridAqExponent: parseFloat(hybridAqExpIn.value) || 1,
    hybridAqThreshold: parseFloat(hybridAqThresholdIn.value) || 0,
    hybridQualityAdaptive: hybridQAdaptIn.checked,
  };
  const custom = buildCustomQuantTables();
  if (custom) opts.customQuantTables = custom;
  return opts;
}

// ─── EQ + quant grid ──────────────────────────────────────────────
function eqStateFor(componentIdx: number): ComponentEqState {
  let s = eqStates.get(componentIdx);
  if (!s) {
    s = {
      hEq: Array(8).fill(1),
      vEq: Array(8).fill(1),
      cellOverrides: new Map(),
    };
    eqStates.set(componentIdx, s);
  }
  return s;
}

function effectiveQuant(
  _comp: ComponentDiagnostics,
  _eq: ComponentEqState,
  componentIdx: number,
): Float32Array {
  // Always use the snapshot-derived table — same f32 numbers we send
  // to the encoder via customQuantTables. Reading the live diagnostics
  // would reflect the previous custom-tables encode and compound any
  // multiplier edit (Q-scale, EQ, etc).
  return effectiveTableF32(componentIdx);
}

function utilizationField(
  comp: ComponentDiagnostics,
  q: Float32Array,
): { field: Float32Array; zeroRate: number } {
  const accum = new Float64Array(64);
  let totalCoef = 0;
  let zeroCoef = 0;
  for (const block of comp.blocks) {
    const c = block.coefPreQuant;
    for (let i = 0; i < 64; i++) {
      const qv = q[i]!;
      if (qv <= 0) continue;
      const ratio = Math.abs((c[i]! as number)) / qv;
      accum[i]! += ratio;
      totalCoef++;
      if (ratio < 0.5) zeroCoef++;
    }
  }
  const field = new Float32Array(64);
  const n = comp.blocks.length;
  if (n > 0) {
    for (let i = 0; i < 64; i++) field[i] = accum[i]! / n;
  }
  return { field, zeroRate: totalCoef > 0 ? (zeroCoef / totalCoef) * 100 : 0 };
}

function bindCellEditor(
  cell: HTMLElement,
  componentIdx: number,
  cellIdx: number,
  baseVal: number,
  effVal: number,
  onChange: () => void,
): void {
  cell.addEventListener("click", () => {
    const eq = eqStateFor(componentIdx);
    const input = window.prompt(
      `Cell (${cellIdx % 8},${Math.floor(cellIdx / 8)}) — base=${baseVal}, ` +
        `current effective=${effVal.toFixed(2)}. Enter new f32 effective:`,
      effVal.toFixed(3),
    );
    if (input === null) return;
    const parsed = parseFloat(input);
    if (!Number.isFinite(parsed) || parsed < 1) {
      status.textContent = `Invalid value (must be ≥ 1): ${input}`;
      return;
    }
    const hv = (eq.hEq[cellIdx % 8]! * eq.vEq[Math.floor(cellIdx / 8)]!) || 1;
    const mult = parsed / (baseVal * hv);
    if (Math.abs(mult - 1) < 1e-4) {
      eq.cellOverrides.delete(cellIdx);
    } else {
      eq.cellOverrides.set(cellIdx, mult);
    }
    onChange();
    scheduleAutoEncode();
  });
}

function buildQuantGrid(
  el: HTMLElement,
  comp: ComponentDiagnostics,
  componentIdx: number,
  eq: ComponentEqState,
  onChange: () => void,
): void {
  el.replaceChildren();
  const eff = effectiveQuant(comp, eq, componentIdx);
  const baseSnap = baseSnapshot?.tables[componentIdx] ?? null;
  for (let v = 0; v < 8; v++) {
    for (let u = 0; u < 8; u++) {
      const cell = document.createElement("div");
      const idx = v * 8 + u;
      const val = eff[idx]!;
      // Show actual f32 precision: 3 sig figs for small values,
      // 2 decimals up to 99.99, 1 decimal up to 999.9, then integer.
      // The encoder consumes these as f32 so show what we'll send.
      const formatted =
        val < 10
          ? val.toFixed(2)
          : val < 100
          ? val.toFixed(2)
          : val < 1000
          ? val.toFixed(1)
          : Math.round(val).toString();
      cell.textContent = formatted;
      const baseVal = baseSnap ? baseSnap[idx]! : ((comp.quantTableBase as Uint16Array | number[])[idx]! as number);
      cell.title = `(${u},${v}) base=${baseVal.toFixed(2)} → effective=${val.toFixed(3)} (f32, click to edit)`;
      if (eq.cellOverrides.has(idx) || Math.abs(val - baseVal) > 1e-3) {
        cell.style.color = "var(--color-accent)";
        cell.style.fontWeight = "600";
      }
      bindCellEditor(cell, componentIdx, idx, baseVal, val, onChange);
      el.appendChild(cell);
    }
  }
}

function buildEqSliders(
  componentIdx: number,
  hEl: HTMLElement,
  vEl: HTMLElement,
  eq: ComponentEqState,
  onChange: () => void,
): void {
  hEl.replaceChildren();
  vEl.replaceChildren();
  for (let i = 0; i < 8; i++) {
    const h = document.createElement("input");
    h.type = "range";
    h.min = "0.5";
    h.max = "2";
    h.step = "0.05";
    h.value = String(eq.hEq[i]!);
    h.dataset.testid = `comp-${componentIdx}-h-eq-${i}`;
    h.title = `h_eq[${i}]`;
    h.addEventListener("input", () => {
      eq.hEq[i] = parseFloat(h.value);
      onChange();
      scheduleAutoEncode();
    });
    hEl.appendChild(h);
    const vv = document.createElement("input");
    vv.type = "range";
    vv.min = "0.5";
    vv.max = "2";
    vv.step = "0.05";
    vv.value = String(eq.vEq[i]!);
    vv.dataset.testid = `comp-${componentIdx}-v-eq-${i}`;
    vv.title = `v_eq[${i}]`;
    vv.addEventListener("input", () => {
      eq.vEq[i] = parseFloat(vv.value);
      onChange();
      scheduleAutoEncode();
    });
    vEl.appendChild(vv);
  }
}

function renderComponent(componentIdx: number, diag: Diagnostics): void {
  const comp = diag.components[componentIdx];
  if (!comp) return;
  let article = document.querySelector<HTMLElement>(`[data-component="${componentIdx}"]`);
  if (!article) {
    const tmpl = document.querySelector<HTMLElement>('[data-component="0"]');
    if (!tmpl) throw new Error("template article missing");
    article = tmpl.cloneNode(true) as HTMLElement;
    article.dataset.component = String(componentIdx);
    article.id = `component-${componentIdx}`;
    article.querySelectorAll<HTMLElement>("[data-testid^='comp-0-']").forEach((el) => {
      el.dataset.testid = el.dataset.testid!.replace("comp-0-", `comp-${componentIdx}-`);
    });
    const headerH2 = article.querySelector<HTMLHeadingElement>("h2");
    if (headerH2) {
      headerH2.textContent = `Component ${componentIdx} (${componentLabel(diag.colorPath, componentIdx)})`;
    }
    compRoot.appendChild(article);
  } else {
    const headerH2 = article.querySelector<HTMLHeadingElement>("h2");
    if (headerH2) {
      headerH2.textContent = `Component ${componentIdx} (${componentLabel(diag.colorPath, componentIdx)})`;
    }
  }

  const gridInfo = article.querySelector<HTMLParagraphElement>(
    `[data-testid="comp-${componentIdx}-grid-info"]`,
  )!;
  const [cols, rows] = comp.blockGrid;
  gridInfo.textContent = `${cols}×${rows} blocks (${cols * rows} total)`;

  const eq = eqStateFor(componentIdx);
  const quantGridEl = article.querySelector<HTMLElement>(
    `[data-testid="comp-${componentIdx}-quant-grid"]`,
  )!;
  const hEqEl = article.querySelector<HTMLElement>(
    `[data-testid="comp-${componentIdx}-h-eq"]`,
  )!;
  const vEqEl = article.querySelector<HTMLElement>(
    `[data-testid="comp-${componentIdx}-v-eq"]`,
  )!;
  const utilCanvas = article.querySelector<HTMLCanvasElement>(
    `[data-testid="comp-${componentIdx}-utilization"]`,
  )!;
  const zeroRateEl = article.querySelector<HTMLElement>(
    `[data-testid="comp-${componentIdx}-zero-rate"]`,
  )!;
  const compResetBtn = article.querySelector<HTMLButtonElement>(
    `[data-testid="comp-${componentIdx}-reset-eq"]`,
  )!;

  const refresh = () => {
    buildQuantGrid(quantGridEl, comp, componentIdx, eq, refresh);
    const q = effectiveQuant(comp, eq, componentIdx);
    const { field, zeroRate } = utilizationField(comp, q);
    drawHeatmap(utilCanvas, field, { cols: 8, rows: 8, pixelSize: 25 });
    zeroRateEl.textContent = zeroRate.toFixed(1);
  };

  buildEqSliders(componentIdx, hEqEl, vEqEl, eq, refresh);
  const freshReset = compResetBtn.cloneNode(true) as HTMLButtonElement;
  compResetBtn.replaceWith(freshReset);
  freshReset.addEventListener("click", () => {
    eq.hEq.fill(1);
    eq.vEq.fill(1);
    eq.cellOverrides.clear();
    article!
      .querySelectorAll<HTMLInputElement>(
        `[data-testid^="comp-${componentIdx}-h-eq-"], [data-testid^="comp-${componentIdx}-v-eq-"]`,
      )
      .forEach((el) => {
        el.value = "1";
      });
    refresh();
    scheduleAutoEncode();
  });
  refresh();
}

function componentLabel(colorPath: string, idx: number): string {
  if (colorPath === "XYB") return ["X", "Y (XYB)", "B"][idx] ?? `c${idx}`;
  if (colorPath === "Grayscale") return "Y";
  return ["Y", "Cb", "Cr"][idx] ?? `c${idx}`;
}

// ─── AQ field rendering ───────────────────────────────────────────
function renderAqField(diag: Diagnostics): void {
  // AQ multiplier is always written to component 0 (Y in YCbCr,
  // X in XYB); the *source* channel for AQ analysis is c1 (Y-XYB)
  // in XYB mode and c0 (Y) in YCbCr mode. Both produce per-block
  // multipliers we display here.
  const yComp = diag.components[0];
  if (!yComp) {
    aqMinEl.textContent = "—";
    aqMaxEl.textContent = "—";
    aqMeanEl.textContent = "—";
    aqChannelEl.textContent = "—";
    return;
  }
  const [cols, rows] = yComp.blockGrid;
  const data = new Float32Array(cols * rows);
  let mn = Infinity;
  let mx = -Infinity;
  let sum = 0;
  for (let i = 0; i < yComp.blocks.length; i++) {
    const v = yComp.blocks[i]!.aqMultiplier;
    data[i] = v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  const mean = yComp.blocks.length > 0 ? sum / yComp.blocks.length : 0;
  drawHeatmap(aqCanvas, data, {
    cols,
    rows,
    pixelSize: Math.max(8, Math.min(40, Math.floor(480 / cols))),
  });
  aqMinEl.textContent = mn === Infinity ? "—" : mn.toFixed(3);
  aqMaxEl.textContent = mx === -Infinity ? "—" : mx.toFixed(3);
  aqMeanEl.textContent = mean.toFixed(3);
  aqChannelEl.textContent =
    diag.colorPath === "XYB"
      ? "X (driven by Y-XYB)"
      : diag.colorPath === "Grayscale"
      ? "Y"
      : "Y";
}

// ─── Curve presets ────────────────────────────────────────────────
function applyCurvePreset(preset: string, qscale: number, tilt: number, hv: number): void {
  const comps = currentDiagnostics?.components.length ?? 1;
  for (let cidx = 0; cidx < comps; cidx++) {
    const eq = eqStateFor(cidx);
    eq.cellOverrides.clear();
    for (let i = 0; i < 8; i++) {
      const fband = i / 7;
      const tiltFactor = Math.exp(-tilt * (fband - 0.5));
      const hFactor = Math.exp(-hv * (fband - 0.5));
      const vFactor = Math.exp(hv * (fband - 0.5));
      let h = qscale * tiltFactor * hFactor;
      let v = qscale * tiltFactor * vFactor;
      switch (preset) {
        case "hf-boost":
          h *= 1 - 0.4 * fband;
          v *= 1 - 0.4 * fband;
          break;
        case "lf-keep":
          h *= 0.85 + 0.3 * fband;
          v *= 0.85 + 0.3 * fband;
          break;
        case "flat":
          h = qscale;
          v = qscale;
          break;
      }
      eq.hEq[i] = clamp(h, 0.5, 2);
      eq.vEq[i] = clamp(v, 0.5, 2);
    }
  }
}
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

// ─── Metrics ──────────────────────────────────────────────────────
function ssim2(src: ImageData, dec: ImageData): number {
  if (src.width !== dec.width || src.height !== dec.height) return 0;
  try {
    return computeSsimulacra2(
      rgbaToPackedRgb(src.data),
      rgbaToPackedRgb(dec.data),
      src.width,
      src.height,
    );
  } catch (e) {
    console.warn("ssim2 failed", e);
    return 0;
  }
}
function zensim(src: ImageData, dec: ImageData): number {
  if (src.width !== dec.width || src.height !== dec.height) return 0;
  try {
    return computeZensim(
      rgbaToPackedRgb(src.data),
      rgbaToPackedRgb(dec.data),
      src.width,
      src.height,
    );
  } catch (e) {
    console.warn("zensim failed", e);
    return 0;
  }
}

// ─── Pareto baseline ──────────────────────────────────────────────
// 100-point integer sweep (q=1..100) computed *progressively*. Running
// all 100 encodes back-to-back would block the worker queue and starve
// user-driven encodes for several seconds. Instead we drive the queue
// from runEncode's `finally` hook: every time the user encode settles,
// we kick off ONE pareto step in the worker. That naturally yields
// priority to user encodes — pareto fills in during idle time.
//
// As each point lands the RD canvas is repainted with the current set,
// so the curve grows in front of the user. The full 1..100 sweep
// resolves over ~tens of seconds for a 1 MP image.

const PARETO_QS = Array.from({ length: 100 }, (_, i) => i + 1); // 1..100

function buildParetoFromPoints(raw: ParetoPoint[]): ParetoPoint[] {
  const sorted = raw.slice().sort((a, b) => a.bytes - b.bytes);
  const pareto: ParetoPoint[] = [];
  let bestScore = -Infinity;
  for (const p of sorted) {
    if (p.score >= bestScore) {
      pareto.push(p);
      bestScore = p.score;
    }
  }
  return pareto;
}

function ensureParetoQueue(image: ImageState): ParetoQueue {
  const existing = paretoQueues.get(image.key);
  if (existing) return existing;
  const cached = paretoCache.get(image.key);
  if (cached) {
    const q: ParetoQueue = {
      qs: [],
      points: cached.slice(),
      total: cached.length,
      complete: true,
    };
    paretoQueues.set(image.key, q);
    return q;
  }
  const q: ParetoQueue = {
    qs: PARETO_QS.slice(),
    points: [],
    total: PARETO_QS.length,
    complete: false,
  };
  paretoQueues.set(image.key, q);
  return q;
}

let paretoStepInflight = false;

async function stepPareto(): Promise<void> {
  // Single-step worker: only one pareto encode in flight at a time,
  // and only while no user encode is pending. Recurses after each step.
  if (paretoStepInflight) return;
  if (inflightReqId !== 0) return;
  const image = currentImage;
  const queue = ensureParetoQueue(image);
  if (queue.complete) return;
  if (queue.qs.length === 0) {
    queue.complete = true;
    paretoCache.set(image.key, buildParetoFromPoints(queue.points));
    return;
  }
  paretoStepInflight = true;
  const q = queue.qs.shift()!;
  try {
    const reqId = nextReqId++;
    const packed = rgbaToPackedRgb(image.rgba);
    const opts: EncodeOptions = {
      quality: q,
      colorPath: "ycbcr",
      subsampling: "quarter",
      aqEnabled: true,
      deringing: true,
      optimizeHuffman: true,
      mode: "baseline",
    };
    const r = await workerEncode(packed, image.width, image.height, opts, reqId);
    if (image.key !== currentImage.key) {
      // Image changed during the step — drop result.
      return;
    }
    const decoded = await decodeJpeg(r.bytes);
    const srcImageData = new ImageData(image.rgba.slice(), image.width, image.height);
    const score = ssim2(srcImageData, decoded);
    queue.points.push({ bytes: r.bytes.byteLength, score, q });
    // Repaint with the current filled-in subset.
    drawRdCanvas(buildParetoFromPoints(queue.points), currentLastEncodePoint());
    updateParetoDelta();
  } catch (e) {
    console.warn(`pareto q=${q} failed:`, e);
  } finally {
    paretoStepInflight = false;
    // If the user isn't encoding, immediately schedule the next pareto
    // step on the next microtask (yields to runEncode if it's queued).
    if (inflightReqId === 0 && !queue.complete) {
      queueMicrotask(() => void stepPareto());
    }
  }
}

function currentLastEncodePoint(): { bytes: number; score: number } | null {
  return lastEncode ? { bytes: lastEncode.bytes, score: lastEncode.ssim2 } : null;
}

function updateParetoDelta(): void {
  const queue = paretoQueues.get(currentImage.key);
  const cur = currentLastEncodePoint();
  if (!cur) return;
  // Use the currently-filled subset to compute Δ-pareto. Refines as
  // more points arrive.
  const points =
    queue && queue.points.length >= 2
      ? buildParetoFromPoints(queue.points)
      : paretoCache.get(currentImage.key) ?? [];
  if (points.length < 2) return;
  const d = paretoDelta(points, cur);
  mPareto.textContent = (d >= 0 ? "+" : "") + d.toFixed(2);
  mPareto.parentElement?.setAttribute(
    "data-tone",
    d >= 0 ? "good" : d > -3 ? "warn" : "bad",
  );
}

function drawRdCanvas(
  pareto: ParetoPoint[],
  current: { bytes: number; score: number } | null,
): void {
  const queue = paretoQueues.get(currentImage.key);
  const filled = queue ? queue.points.length : pareto.length;
  const total = queue ? queue.total : pareto.length;
  const ctx = rdCanvas.getContext("2d");
  if (!ctx) return;
  const W = rdCanvas.width;
  const H = rdCanvas.height;
  ctx.fillStyle = "#0e1117";
  ctx.fillRect(0, 0, W, H);
  if (pareto.length === 0) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillText(
      filled === 0
        ? "Computing reference RD curve in the background…"
        : `Building reference RD: ${filled}/${total}`,
      20,
      H / 2,
    );
    return;
  }
  const allBytes = pareto.map((p) => p.bytes);
  const allScores = pareto.map((p) => p.score);
  if (current) {
    allBytes.push(current.bytes);
    allScores.push(current.score);
  }
  const minB = Math.min(...allBytes) * 0.9;
  const maxB = Math.max(...allBytes) * 1.1;
  const minS = Math.min(...allScores) - 2;
  const maxS = Math.max(...allScores) + 2;
  const padL = 50;
  const padR = 12;
  const padT = 12;
  const padB = 32;
  const xFor = (b: number) =>
    padL +
    ((Math.log10(b) - Math.log10(minB)) / (Math.log10(maxB) - Math.log10(minB))) *
      (W - padL - padR);
  const yFor = (s: number) => padT + (1 - (s - minS) / (maxS - minS)) * (H - padT - padB);
  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();
  ctx.fillStyle = "#8b949e";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillText("bytes (log)", W - 80, H - 10);
  ctx.save();
  ctx.translate(14, padT + 60);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("ssimulacra2", 0, 0);
  ctx.restore();
  ctx.strokeStyle = "#c9d1d9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < pareto.length; i++) {
    const p = pareto[i]!;
    const x = xFor(p.bytes);
    const y = yFor(p.score);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#c9d1d9";
  for (const p of pareto) {
    ctx.beginPath();
    ctx.arc(xFor(p.bytes), yFor(p.score), 3, 0, Math.PI * 2);
    ctx.fill();
  }
  if (current) {
    ctx.fillStyle = "#58a6ff";
    ctx.beginPath();
    ctx.arc(xFor(current.bytes), yFor(current.score), 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  // Progress label (top-right).
  const isComplete = queue?.complete === true;
  ctx.fillStyle = isComplete ? "#56d364" : "#f0883e";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  const progressText = isComplete
    ? `ref RD ${filled}/${total} ✓`
    : `ref RD ${filled}/${total}`;
  const tw = ctx.measureText(progressText).width;
  ctx.fillText(progressText, W - padR - tw, padT + 12);
}

function paretoDelta(
  pareto: ParetoPoint[],
  cur: { bytes: number; score: number },
): number {
  if (pareto.length < 2) return 0;
  const lb = Math.log10(cur.bytes);
  let prev = pareto[0]!;
  for (let i = 1; i < pareto.length; i++) {
    const p = pareto[i]!;
    const a = Math.log10(prev.bytes);
    const b = Math.log10(p.bytes);
    if (lb >= a && lb <= b) {
      const t = (lb - a) / Math.max(1e-9, b - a);
      const interp = prev.score + (p.score - prev.score) * t;
      return cur.score - interp;
    }
    prev = p;
  }
  if (lb < Math.log10(pareto[0]!.bytes)) return cur.score - pareto[0]!.score;
  return cur.score - pareto[pareto.length - 1]!.score;
}

// ─── Big-readout updater ──────────────────────────────────────────
function updateReadout(): void {
  if (!lastEncode) {
    mBytes.textContent = "—";
    mBpp.textContent = "—";
    mMode.textContent = "—";
    mModeDetail.textContent = "—";
    mSsim2.textContent = "—";
    mZensim.textContent = "—";
    mPareto.textContent = "—";
    mTime.textContent = "—";
    return;
  }
  const px = lastEncode.width * lastEncode.height;
  mBytes.textContent = formatBytes(lastEncode.bytes);
  mBpp.textContent = px > 0 ? `${((lastEncode.bytes * 8) / px).toFixed(2)} bpp` : "—";
  mMode.textContent = lastEncode.mode.toUpperCase();
  mModeDetail.textContent = "";
  mSsim2.textContent = lastEncode.ssim2.toFixed(2);
  mSsim2.parentElement?.setAttribute("data-tone", scoreTone(lastEncode.ssim2));
  mZensim.textContent = lastEncode.zensim.toFixed(2);
  mZensim.parentElement?.setAttribute("data-tone", scoreTone(lastEncode.zensim));
  mTime.textContent = lastEncode.durationMs.toFixed(0);
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n}`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}K`;
  return `${(n / 1024 / 1024).toFixed(2)}M`;
}
function scoreTone(s: number): string {
  if (s >= 80) return "good";
  if (s >= 60) return "warn";
  return "bad";
}

// ─── Encode pipeline ──────────────────────────────────────────────
let debounceTimer: number | null = null;
const DEBOUNCE_MS = 250;

function scheduleAutoEncode(): void {
  if (!autoEncodeIn.checked) return;
  if (debounceTimer !== null) window.clearTimeout(debounceTimer);
  debounceTimer = window.setTimeout(() => {
    debounceTimer = null;
    void runEncode();
  }, DEBOUNCE_MS);
}

function setEncodingIndicator(on: boolean, detail = ""): void {
  const overlay = document.getElementById("encode-overlay");
  const detailEl = document.getElementById("encode-overlay-detail");
  if (overlay) overlay.hidden = !on;
  if (detailEl) detailEl.textContent = detail;
  document.body.dataset.encoding = on ? "true" : "false";
}

async function runEncode(): Promise<void> {
  const reqId = nextReqId++;
  latestReqId = reqId;
  inflightReqId = reqId;
  encodeBtn.disabled = true;
  cancelBtn.disabled = false;
  status.textContent = "Encoding…";
  document.body.dataset.encodePhase = "start";
  setEncodingIndicator(
    true,
    `req#${reqId} · ${currentImage.width}×${currentImage.height} · q=${qualityIn.value} · ${getMode()}`,
  );
  try {
    await workerReady;
    const packed = rgbaToPackedRgb(currentImage.rgba);
    const opts = currentOptions();
    document.body.dataset.encodePhase = "calling-wasm";
    const result = await workerEncode(
      packed,
      currentImage.width,
      currentImage.height,
      opts,
      reqId,
    );
    document.body.dataset.encodePhase = "wasm-returned";
    if (reqId !== latestReqId) {
      // Stale result — drop.
      return;
    }
    drawRGBA(sourceCanvas, currentImage.rgba, currentImage.width, currentImage.height);
    const decoded = await decodeJpeg(result.bytes);
    encodedCanvas.width = decoded.width;
    encodedCanvas.height = decoded.height;
    encodedCanvas.getContext("2d")!.putImageData(decoded, 0, 0);
    const sourceImageData = new ImageData(
      currentImage.rgba.slice(),
      currentImage.width,
      currentImage.height,
    );
    const delta = deltaImage(sourceImageData, decoded);
    deltaCanvas.width = delta.width;
    deltaCanvas.height = delta.height;
    deltaCanvas.getContext("2d")!.putImageData(delta, 0, 0);

    // Cache for the zoom inspector.
    zoomImageData = {
      source: sourceImageData,
      encoded: decoded,
      delta,
    };
    if (zoomState.open) {
      // Refresh the open inspector with the new pixels (slider drag
      // is auto-encoding — the user expects to watch deltas evolve
      // at the same zoom).
      drawZoomFrame();
    }

    currentDiagnostics = result.diagnostics;
    // If this encode ran without custom tables, the diagnostics' base
    // tables are the encoder's source-derived defaults — capture them
    // so subsequent edits are multipliers on a stable base.
    if (!opts.customQuantTables) {
      captureBaseSnapshot(result.diagnostics);
    }
    renderAqField(result.diagnostics);
    compRoot
      .querySelectorAll<HTMLElement>("[data-component]")
      .forEach((el) => {
        if (el.dataset.component !== "0") el.remove();
      });
    for (let i = 0; i < result.diagnostics.components.length; i++) {
      renderComponent(i, result.diagnostics);
    }

    const ssim2Score = ssim2(sourceImageData, decoded);
    const zensimScore = zensim(sourceImageData, decoded);
    lastEncode = {
      bytes: result.bytes.byteLength,
      ssim2: ssim2Score,
      zensim: zensimScore,
      durationMs: result.durationMs,
      mode: opts.mode ?? "baseline",
      width: currentImage.width,
      height: currentImage.height,
    };

    // Pareto delta — refresh asynchronously.
    void refreshRdPanel();

    updateReadout();
    const customMark = opts.customQuantTables ? " · custom Q" : "";
    status.textContent = `Encoded ${result.bytes.byteLength} B in ${result.durationMs.toFixed(0)} ms${customMark}`;
    document.body.dataset.encodeState = "done";
  } catch (e) {
    if (reqId === latestReqId) {
      console.error(e);
      status.textContent = `Error: ${(e as Error).message}`;
      document.body.dataset.encodeState = "error";
    }
  } finally {
    if (inflightReqId === reqId) {
      inflightReqId = 0;
      encodeBtn.disabled = false;
      cancelBtn.disabled = true;
      setEncodingIndicator(false);
      // Kick the next pareto step now that the user encode has settled.
      // stepPareto() guards against concurrent stepping and yields if
      // the user kicks another encode while it's running.
      queueMicrotask(() => void stepPareto());
    }
  }
}

function refreshRdPanel(): void {
  // Draw whatever pareto data is currently available — full from cache
  // if the sweep is complete, partial from the active queue otherwise.
  const queue = ensureParetoQueue(currentImage);
  const points = queue.complete
    ? paretoCache.get(currentImage.key) ?? buildParetoFromPoints(queue.points)
    : buildParetoFromPoints(queue.points);
  drawRdCanvas(points, currentLastEncodePoint());
  updateParetoDelta();
}

// ─── Image loading ────────────────────────────────────────────────
async function loadFile(file: File): Promise<void> {
  const url = URL.createObjectURL(file);
  // No browser-side wash for uploads either. If the user uploads a
  // JPEG / WebP / AVIF, the diagnostics will reflect "encoder + prior
  // codec artifacts" not "encoder alone" — that's the user's
  // responsibility. We surface a warning so it's not a silent footgun.
  const lossy = !/\b(?:image\/png|image\/x-png)\b/i.test(file.type);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = rej;
      i.src = url;
    });
    const decoded = decodeImageToCanvas(img);
    currentImage = {
      width: decoded.width,
      height: decoded.height,
      rgba: decoded.rgba,
      key: `upload:${file.name}:${decoded.width}x${decoded.height}`,
    };
    const lossyWarn = lossy
      ? " ⚠ lossy source — pre-process for clean diagnostics"
      : "";
    imageInfoEl.textContent =
      `${decoded.width} × ${decoded.height} (uploaded ${file.type})${lossyWarn}`;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function fetchAsImage(url: string): Promise<HTMLImageElement> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
  const blob = await resp.blob();
  const objUrl = URL.createObjectURL(blob);
  try {
    return await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = () => rej(new Error("image decode failed"));
      i.src = objUrl;
    });
  } finally {
    // Important: keep objUrl alive until the image's onload fires;
    // safe to revoke after Promise resolves since the browser has
    // already parsed the bitmap.
    URL.revokeObjectURL(objUrl);
  }
}

/** Decode an HTMLImageElement to RGBA pixels at native dimensions.
 *  No browser-side resampling — the wash (8× lanczos3) is performed
 *  at build time by scripts/fetch-images.ts using sharp, so by the
 *  time we get here the image bytes are already the cleaned source. */
function decodeImageToCanvas(
  img: HTMLImageElement,
): { width: number; height: number; rgba: Uint8ClampedArray } {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const cx = c.getContext("2d");
  if (!cx) throw new Error("canvas 2d unavailable");
  cx.imageSmoothingEnabled = false;
  cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, w, h);
  return { width: w, height: h, rgba: data.data };
}

async function loadEntry(entry: ImageManifestEntry): Promise<void> {
  let img: HTMLImageElement | null = null;
  // Try the local cached path first (works on any deploy that ran the
  // prebuild fetch — `npm run build` triggers it). The absolute R2 URL
  // is the fallback but it WILL hit CORS unless either (a) the bucket
  // has CORS configured (Access-Control-Allow-Origin) or (b) the page
  // is being viewed at the same origin as the bucket. So we annotate
  // the error specifically when remoteUrl fails so the message is
  // actionable.
  const tries: { url: string; isLocal: boolean }[] = [];
  if (entry.localUrl) tries.push({ url: entry.localUrl, isLocal: true });
  tries.push({ url: entry.remoteUrl, isLocal: false });
  let lastErr: Error | null = null;
  for (const { url, isLocal } of tries) {
    try {
      img = await fetchAsImage(url);
      break;
    } catch (e) {
      const msg = (e as Error).message;
      lastErr = e as Error;
      if (!isLocal && msg.toLowerCase().includes("fetch")) {
        lastErr = new Error(
          `R2 fetch blocked (likely CORS): ${url} → ${msg}. ` +
            `Fix: rebuild via \`npm run build\` so the prebuild hook ` +
            `bakes images into dist/images/, or configure CORS on the ` +
            `codec-corpus bucket.`,
        );
      }
      console.warn(`[picker] ${url} failed:`, msg);
    }
  }
  if (!img) throw lastErr ?? new Error("no image source available");
  const decoded = decodeImageToCanvas(img);
  currentImage = {
    width: decoded.width,
    height: decoded.height,
    rgba: decoded.rgba,
    key: entry.id,
  };
  imageInfoEl.textContent = `${decoded.width} × ${decoded.height} — ${entry.label}`;
}

function loadImageManifest(): void {
  // Bundled list always populates the dropdown. The "real" runtime
  // manifest (./images/manifest.json) carries the post-wash URLs +
  // dimensions when the prebuild has run; we prefer those. The
  // bundled list is the fallback when the manifest is empty.
  const base = (bundled.defaultBase ?? FALLBACK_BASE).replace(/\/+$/, "");
  imageManifest = [];
  for (const entry of bundled.images) {
    const urls = entryToUrls(entry, base);
    if (!urls) continue;
    // For wasJpeg entries, the served local URL is `<sha>-w8.png`
    // (the post-wash PNG written by the prebuild). The raw blob URL
    // stays on R2 — there's no point falling back to the un-washed
    // remote when the wash exists locally.
    const localUrl = entry.wasJpeg && entry.sha256
      ? `./images/${entry.sha256.toLowerCase()}-w8.png`
      : urls.localUrl;
    imageManifest.push({
      id: entry.id,
      label: entry.label,
      localUrl,
      remoteUrl: urls.remoteUrl,
    });
    const opt = document.createElement("option");
    opt.value = entry.id;
    opt.textContent = entry.label;
    imagePickSel.appendChild(opt);
  }
  console.log(`[picker] populated with ${imageManifest.length} demo images`);
}

// ─── Tabs + ignored annotations ───────────────────────────────────
function setActiveTab(name: string): void {
  activeTab = name;
  document.querySelectorAll<HTMLElement>(".tab").forEach((t) => {
    const sel = t.dataset.tab === name;
    t.setAttribute("aria-selected", String(sel));
  });
  document.querySelectorAll<HTMLElement>(".tab-panel").forEach((p) => {
    const sel = p.dataset.tab === name;
    p.hidden = !sel;
    p.setAttribute("aria-hidden", String(!sel));
  });
  applyIgnoredAnnotations();
}

function refreshQuantBanner(): void {
  const el = document.getElementById("quant-mode-banner");
  if (!el) return;
  const mode = getMode();
  const aq = aqIn.checked;
  const cp = colorPathSel.value === "xyb" ? "XYB" : "YCbCr";
  let msg = "";
  switch (mode) {
    case "baseline":
      msg = aq
        ? `Adaptive Quantization (${cp}). Final per-block quant = table[k] × aq_multiplier_block.`
        : `${cp} with AQ disabled. Final per-block quant = table[k]. The AQ checkbox is off.`;
      break;
    case "trellis":
      msg = `Trellis (${cp}). Lambda RD picks per-coefficient quant levels; the table acts as an upper bound. Edits to the table still constrain the search space.`;
      break;
    case "hybrid":
      msg = `Hybrid (${cp}). AQ-aware lambda RD. The table sets the AQ multiplier base; trellis picks final levels per block, modulated by per-block AQ.`;
      break;
  }
  el.textContent = msg;
  el.style.color = "var(--color-warm)";
}

function applyIgnoredAnnotations(): void {
  const mode = getMode();
  const cp = colorPathSel.value;
  const sub = subSel.value;

  const trellisPanel = document.querySelector<HTMLElement>('.tab-panel[data-tab="trellis"]');
  if (trellisPanel) {
    const disabled = mode !== "trellis";
    trellisPanel
      .querySelectorAll<HTMLElement>(".ctrl")
      .forEach((c) => (c.dataset.disabled = String(disabled)));
    const note = trellisPanel.querySelector<HTMLElement>('[data-when="trellis-ignored"]');
    if (note) note.style.display = disabled ? "" : "none";
  }
  const hybridPanel = document.querySelector<HTMLElement>('.tab-panel[data-tab="hybrid"]');
  if (hybridPanel) {
    const disabled = mode !== "hybrid";
    hybridPanel
      .querySelectorAll<HTMLElement>(".ctrl")
      .forEach((c) => (c.dataset.disabled = String(disabled)));
    const note = hybridPanel.querySelector<HTMLElement>('[data-when="hybrid-ignored"]');
    if (note) note.style.display = disabled ? "" : "none";
  }
  const sharpCtrl = sharpYuvIn.closest<HTMLElement>(".ctrl");
  if (sharpCtrl) {
    const disabled = cp === "xyb" || sub === "none";
    sharpCtrl.dataset.disabled = String(disabled);
    const note = sharpCtrl.querySelector<HTMLElement>('[data-when="sharp-yuv-ignored"]');
    if (note)
      note.textContent = disabled
        ? cp === "xyb"
          ? "(ignored — XYB)"
          : "(ignored — 4:4:4)"
        : "";
  }
}

function syncCurveOutputs(): void {
  curveQscaleOut.textContent = parseFloat(curveQscaleIn.value).toFixed(2);
  const tilt = parseFloat(curveTiltIn.value);
  curveTiltOut.textContent = (tilt >= 0 ? "+" : "") + tilt.toFixed(2);
  const hv = parseFloat(curveHvIn.value);
  curveHvOut.textContent = (hv >= 0 ? "+" : "") + hv.toFixed(2);
}

function applyCurveControls(): void {
  applyCurvePreset(
    curvePresetSel.value,
    parseFloat(curveQscaleIn.value),
    parseFloat(curveTiltIn.value),
    parseFloat(curveHvIn.value),
  );
  if (currentDiagnostics) {
    for (let i = 0; i < currentDiagnostics.components.length; i++) {
      renderComponent(i, currentDiagnostics);
    }
  }
}

// ─── Wiring ────────────────────────────────────────────────────────
function wireControls(): void {
  qualityIn.addEventListener("input", () => {
    qualityOut.textContent = qualityIn.value;
    invalidateBaseSnapshotIfStale();
    scheduleAutoEncode();
  });
  colorPathSel.addEventListener("change", () => {
    if (colorPathSel.value === "xyb") {
      subLabel.hidden = true;
      xybLabel.hidden = false;
    } else {
      subLabel.hidden = false;
      xybLabel.hidden = true;
    }
    applyIgnoredAnnotations();
    invalidateBaseSnapshotIfStale();
    scheduleAutoEncode();
  });
  // Subsampling changes the encoder's default tables AND determines
  // whether sharp-yuv is meaningful. Refresh ignored-annotations.
  for (const el of [subSel, xybSubSel]) {
    el.addEventListener("change", () => {
      invalidateBaseSnapshotIfStale();
      applyIgnoredAnnotations();
    });
  }
  // Encoder option inputs that should trigger auto-encode.
  for (const el of [
    subSel,
    xybSubSel,
    aqIn,
    deringingIn,
    sharpYuvIn,
    preBlurIn,
    chromaDistIn,
    optimizeHuffmanIn,
    progressiveIn,
    restartMcuRowsIn,
    trellisDcIn,
    trellisLambda1In,
    trellisLambda2In,
    trellisSpeedSel,
    trellisDeltaDcIn,
    hybridAqLambdaIn,
    hybridBaseLambda1In,
    hybridBaseLambda2In,
    hybridDcIn,
    hybridAqExpIn,
    hybridAqThresholdIn,
    hybridQAdaptIn,
  ]) {
    el.addEventListener("change", scheduleAutoEncode);
    el.addEventListener("input", scheduleAutoEncode);
  }
  document.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((r) => {
    r.addEventListener("change", () => {
      applyIgnoredAnnotations();
      scheduleAutoEncode();
    });
  });

  encodeBtn.addEventListener("click", () => void runEncode());
  cancelBtn.addEventListener("click", () => {
    // We can't preempt the worker mid-encode without SharedArrayBuffer,
    // but we can mark the in-flight request stale so its result is
    // dropped when it returns. Bumping latestReqId achieves that.
    latestReqId = nextReqId++;
    inflightReqId = 0;
    status.textContent = "Cancelled (in-flight result will be dropped)";
    cancelBtn.disabled = true;
    encodeBtn.disabled = false;
    setEncodingIndicator(false);
  });
  resetBtn.addEventListener("click", () => {
    currentImage = synthetic();
    eqStates.clear();
    paretoCache.delete(currentImage.key);
    paretoQueues.delete(currentImage.key);
    imagePickSel.value = "__synthetic__";
    imageInfoEl.textContent = "64 × 64 (synthetic)";
    void runEncode();
  });
  fileIn.addEventListener("change", async () => {
    const file = fileIn.files?.[0];
    if (!file) return;
    eqStates.clear();
    try {
      await loadFile(file);
    } catch (e) {
      status.textContent = `Failed to load file: ${(e as Error).message}`;
      return;
    }
    paretoCache.delete(currentImage.key);
    paretoQueues.delete(currentImage.key);
    imagePickSel.value = "__synthetic__";
    void runEncode();
  });
  imagePickSel.addEventListener("change", async () => {
    const id = imagePickSel.value;
    if (id === "__synthetic__") {
      currentImage = synthetic();
      imageInfoEl.textContent = "64 × 64 (synthetic)";
    } else {
      const entry = imageManifest.find((e) => e.id === id);
      if (!entry) return;
      eqStates.clear();
      status.textContent = `Loading ${entry.label}…`;
      try {
        await loadEntry(entry);
      } catch (e) {
        status.textContent = `Failed to load ${entry.label}: ${(e as Error).message}`;
        console.error(e);
        return;
      }
    }
    paretoCache.delete(currentImage.key);
    paretoQueues.delete(currentImage.key);
    void runEncode();
  });

  // Tabs.
  document.querySelectorAll<HTMLElement>(".tab").forEach((t) => {
    t.addEventListener("click", () => setActiveTab(t.dataset.tab!));
  });

  // Curve controls.
  for (const inp of [curveQscaleIn, curveTiltIn, curveHvIn]) {
    inp.addEventListener("input", () => {
      syncCurveOutputs();
      applyCurveControls();
      scheduleAutoEncode();
    });
  }
  curvePresetSel.addEventListener("change", () => {
    applyCurveControls();
    scheduleAutoEncode();
  });
  curveResetBtn.addEventListener("click", () => {
    curveQscaleIn.value = "1";
    curveTiltIn.value = "0";
    curveHvIn.value = "0";
    curvePresetSel.value = "jfif";
    syncCurveOutputs();
    eqStates.clear();
    if (currentDiagnostics) {
      for (let i = 0; i < currentDiagnostics.components.length; i++) {
        renderComponent(i, currentDiagnostics);
      }
    }
    scheduleAutoEncode();
  });
  syncCurveOutputs();
  applyIgnoredAnnotations();
  refreshQuantBanner();

  // Zero-bias controls.
  for (let cidx = 0; cidx < 3; cidx++) {
    const mulIn = document.getElementById(`zb-mul-${cidx}`) as HTMLInputElement | null;
    const mulOut = document.getElementById(`zb-mul-${cidx}-out`) as HTMLOutputElement | null;
    const dcIn = document.getElementById(`zb-dc-${cidx}`) as HTMLInputElement | null;
    const acIn = document.getElementById(`zb-ac-${cidx}`) as HTMLInputElement | null;
    if (mulIn && mulOut) {
      mulIn.addEventListener("input", () => {
        const v = parseFloat(mulIn.value);
        zeroBias.globalMul[cidx] = v;
        mulOut.textContent = v.toFixed(2);
        scheduleAutoEncode();
      });
    }
    if (dcIn) {
      dcIn.addEventListener("input", () => {
        zeroBias.offsetDc[cidx] = parseFloat(dcIn.value) || 0;
        scheduleAutoEncode();
      });
    }
    if (acIn) {
      acIn.addEventListener("input", () => {
        zeroBias.offsetAc[cidx] = parseFloat(acIn.value) || 0;
        scheduleAutoEncode();
      });
    }
  }
  const zbReset = document.getElementById("zb-reset");
  zbReset?.addEventListener("click", () => {
    zeroBias.globalMul = [1, 1, 1];
    zeroBias.offsetDc = [0, 0, 0];
    zeroBias.offsetAc = [0, 0, 0];
    for (let cidx = 0; cidx < 3; cidx++) {
      const mulIn = document.getElementById(`zb-mul-${cidx}`) as HTMLInputElement | null;
      const mulOut = document.getElementById(`zb-mul-${cidx}-out`) as HTMLOutputElement | null;
      const dcIn = document.getElementById(`zb-dc-${cidx}`) as HTMLInputElement | null;
      const acIn = document.getElementById(`zb-ac-${cidx}`) as HTMLInputElement | null;
      if (mulIn) mulIn.value = "1";
      if (mulOut) mulOut.textContent = "1.00";
      if (dcIn) dcIn.value = "0";
      if (acIn) acIn.value = "0";
    }
    scheduleAutoEncode();
  });

  // Mode radios — also refresh the quant banner.
  document.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach((r) => {
    r.addEventListener("change", refreshQuantBanner);
  });
  aqIn.addEventListener("change", refreshQuantBanner);
  colorPathSel.addEventListener("change", refreshQuantBanner);
}

// ─── Zoom inspector ───────────────────────────────────────────────
// Click any of source/encoded/delta to open a synchronized 3-pane
// pixel-zoom view. Drag to pan, wheel to change zoom, 1-9 keys to
// jump to scale, ESC to close. Re-renders live as the user drags
// quality sliders (auto-encode).

interface ZoomState {
  open: boolean;
  /** Image-pixel coordinates of the *center* of the viewport. */
  centerX: number;
  centerY: number;
  /** Display pixels per source pixel. */
  scale: number;
  /** True while the user is dragging. */
  panning: boolean;
  panStartX: number;
  panStartY: number;
  panStartCenterX: number;
  panStartCenterY: number;
}
const zoomState: ZoomState = {
  open: false,
  centerX: 0,
  centerY: 0,
  scale: 4,
  panning: false,
  panStartX: 0,
  panStartY: 0,
  panStartCenterX: 0,
  panStartCenterY: 0,
};

function $zoomCanvas(name: "source" | "encoded" | "delta"): HTMLCanvasElement {
  return $<HTMLCanvasElement>(`#zoom-${name}`);
}

function openZoomFromClick(
  src: "source" | "encoded" | "delta",
  ev: MouseEvent,
): void {
  const data = zoomImageData[src];
  if (!data) return;
  // Translate the click into source-pixel coordinates.
  const target = ev.currentTarget as HTMLCanvasElement;
  const rect = target.getBoundingClientRect();
  const sx = ((ev.clientX - rect.left) / rect.width) * target.width;
  const sy = ((ev.clientY - rect.top) / rect.height) * target.height;
  // Each pane may have its own dimensions (encoded/delta could differ
  // from source in pathological cases) but for our pipeline they're
  // always equal. Use the source pane as the canonical coordinate space.
  const ref = zoomImageData.source ?? data;
  const refX = (sx / target.width) * ref.width;
  const refY = (sy / target.height) * ref.height;
  zoomState.centerX = refX;
  zoomState.centerY = refY;
  zoomState.scale = 4;
  zoomState.open = true;
  const modal = $<HTMLElement>("#zoom-modal");
  modal.hidden = false;
  document.body.style.overflow = "hidden";
  // Two RAFs because the modal needs a layout pass before the
  // canvases have non-zero clientWidth/Height.
  requestAnimationFrame(() => requestAnimationFrame(drawZoomFrame));
}

function closeZoom(): void {
  zoomState.open = false;
  zoomState.panning = false;
  $<HTMLElement>("#zoom-modal").hidden = true;
  document.body.style.overflow = "";
}

function drawZoomPane(
  name: "source" | "encoded" | "delta",
  data: ImageData | null,
): void {
  const c = $zoomCanvas(name);
  // Match canvas pixels to its CSS box so we don't get blurry scaling.
  const cssW = Math.max(64, Math.floor(c.clientWidth));
  const cssH = Math.max(64, Math.floor(c.clientHeight));
  if (c.width !== cssW || c.height !== cssH) {
    c.width = cssW;
    c.height = cssH;
  }
  const ctx = c.getContext("2d");
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, c.width, c.height);
  if (!data) return;

  // Extract the visible window from the source data and draw it
  // pixel-doubled (or N-doubled). Center is in image-pixel coords.
  const scale = zoomState.scale;
  const visW = c.width / scale;
  const visH = c.height / scale;
  const x0 = zoomState.centerX - visW / 2;
  const y0 = zoomState.centerY - visH / 2;

  // Use a temporary canvas to draw the source ImageData, then drawImage
  // it scaled. drawImage with imageSmoothingEnabled=false gives crisp
  // pixel doubling at any integer scale. Floats fall back to nearest.
  const tmp = document.createElement("canvas");
  tmp.width = data.width;
  tmp.height = data.height;
  tmp.getContext("2d")!.putImageData(data, 0, 0);
  ctx.drawImage(
    tmp,
    x0,
    y0,
    visW,
    visH,
    0,
    0,
    c.width,
    c.height,
  );

  // Subtle grid every 8 source pixels to mark JPEG block edges, but
  // only when the scale is large enough to make the lines worth seeing.
  if (scale >= 4) {
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const firstBlockX = Math.ceil(x0 / 8) * 8;
    for (let bx = firstBlockX; bx < x0 + visW; bx += 8) {
      const px = (bx - x0) * scale;
      ctx.moveTo(px, 0);
      ctx.lineTo(px, c.height);
    }
    const firstBlockY = Math.ceil(y0 / 8) * 8;
    for (let by = firstBlockY; by < y0 + visH; by += 8) {
      const py = (by - y0) * scale;
      ctx.moveTo(0, py);
      ctx.lineTo(c.width, py);
    }
    ctx.stroke();
  }
}

function drawZoomFrame(): void {
  if (!zoomState.open) return;
  drawZoomPane("source", zoomImageData.source);
  drawZoomPane("encoded", zoomImageData.encoded);
  drawZoomPane("delta", zoomImageData.delta);
  const coordsEl = document.getElementById("zoom-coords");
  if (coordsEl)
    coordsEl.textContent = `(${Math.round(zoomState.centerX)}, ${Math.round(zoomState.centerY)})`;
  const scaleEl = document.getElementById("zoom-scale");
  if (scaleEl) scaleEl.textContent = `×${zoomState.scale}`;
}

function wireZoomInspector(): void {
  // Click on a triptych canvas opens the modal at the click point.
  for (const which of ["source", "encoded", "delta"] as const) {
    const triptych = document.querySelector<HTMLCanvasElement>(
      `[data-zoom-source="${which}"]`,
    );
    if (!triptych) continue;
    triptych.addEventListener("click", (ev) => openZoomFromClick(which, ev));
  }

  const modal = $<HTMLElement>("#zoom-modal");
  $<HTMLButtonElement>("#zoom-close").addEventListener("click", closeZoom);
  modal.addEventListener("click", (ev) => {
    if (ev.target === modal) closeZoom();
  });

  // Pan + wheel zoom on every zoom canvas (synchronized).
  for (const which of ["source", "encoded", "delta"] as const) {
    const c = $zoomCanvas(which);
    c.addEventListener("mousedown", (ev) => {
      zoomState.panning = true;
      zoomState.panStartX = ev.clientX;
      zoomState.panStartY = ev.clientY;
      zoomState.panStartCenterX = zoomState.centerX;
      zoomState.panStartCenterY = zoomState.centerY;
    });
    c.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      // Wheel up → zoom in. Snap to integer scales for crisp pixels.
      const delta = ev.deltaY < 0 ? +1 : -1;
      const next = clamp(zoomState.scale + delta, 1, 32);
      if (next === zoomState.scale) return;
      // Zoom around the cursor position so the pixel under the
      // cursor stays put.
      const rect = c.getBoundingClientRect();
      const cx = ev.clientX - rect.left;
      const cy = ev.clientY - rect.top;
      const oldVisW = c.width / zoomState.scale;
      const oldVisH = c.height / zoomState.scale;
      const cursorImgX =
        zoomState.centerX - oldVisW / 2 + (cx / c.width) * oldVisW;
      const cursorImgY =
        zoomState.centerY - oldVisH / 2 + (cy / c.height) * oldVisH;
      zoomState.scale = next;
      const newVisW = c.width / zoomState.scale;
      const newVisH = c.height / zoomState.scale;
      zoomState.centerX = cursorImgX - (cx / c.width - 0.5) * newVisW;
      zoomState.centerY = cursorImgY - (cy / c.height - 0.5) * newVisH;
      drawZoomFrame();
    }, { passive: false });
  }

  window.addEventListener("mousemove", (ev) => {
    if (!zoomState.panning) return;
    // Pan all three panes together. Use any zoom canvas's size as the
    // reference for px-per-source-pixel scaling.
    const c = $zoomCanvas("source");
    const dx = (ev.clientX - zoomState.panStartX) / zoomState.scale;
    const dy = (ev.clientY - zoomState.panStartY) / zoomState.scale;
    zoomState.centerX = zoomState.panStartCenterX - dx;
    zoomState.centerY = zoomState.panStartCenterY - dy;
    void c;
    drawZoomFrame();
  });
  window.addEventListener("mouseup", () => {
    zoomState.panning = false;
  });

  // Keyboard: ESC closes; 1-9 sets scale; arrow keys nudge center.
  window.addEventListener("keydown", (ev) => {
    if (!zoomState.open) return;
    if (ev.key === "Escape") {
      ev.preventDefault();
      closeZoom();
      return;
    }
    if (/^[1-9]$/.test(ev.key)) {
      zoomState.scale = parseInt(ev.key, 10);
      drawZoomFrame();
      ev.preventDefault();
      return;
    }
    const step = 16 / zoomState.scale;
    if (ev.key === "ArrowLeft") {
      zoomState.centerX -= step;
      drawZoomFrame();
      ev.preventDefault();
    } else if (ev.key === "ArrowRight") {
      zoomState.centerX += step;
      drawZoomFrame();
      ev.preventDefault();
    } else if (ev.key === "ArrowUp") {
      zoomState.centerY -= step;
      drawZoomFrame();
      ev.preventDefault();
    } else if (ev.key === "ArrowDown") {
      zoomState.centerY += step;
      drawZoomFrame();
      ev.preventDefault();
    }
  });

  // Window resize triggers a re-render so canvas dims sync.
  window.addEventListener("resize", () => {
    if (zoomState.open) drawZoomFrame();
  });
}

// ─── Bootstrap ────────────────────────────────────────────────────
(async () => {
  document.body.dataset.bootstrap = "started";
  status.textContent = "Initializing wasm in worker…";
  try {
    // Initialize the main-thread wasm too (for ssim2/zensim scoring,
    // since worker-side scoring is deferred until after we decode the
    // JPEG on this thread anyway).
    await init();
    document.body.dataset.bootstrap = "main-wasm-init-ok";
  } catch (err) {
    document.body.dataset.bootstrap = "main-wasm-init-failed";
    status.textContent = `Main wasm init failed: ${(err as Error).message ?? err}`;
    return;
  }
  await workerReady;
  document.body.dataset.bootstrap = "worker-ready";
  status.textContent = "Wasm + worker ready — encoding synthetic 64×64 sample";
  document.body.dataset.wasmState = "ready";
  loadImageManifest();
  imageInfoEl.textContent = "64 × 64 (synthetic)";
  try {
    wireControls();
    wireZoomInspector();
    document.body.dataset.bootstrap = "controls-wired";
  } catch (err) {
    document.body.dataset.bootstrap = "controls-wire-failed";
    status.textContent = `wireControls failed: ${(err as Error).message ?? err}`;
    return;
  }
  await runEncode();
})();

// ─── Test API ─────────────────────────────────────────────────────
declare global {
  interface Window {
    __zenjpegDiagnostics: {
      getCurrent: () => Diagnostics | null;
      getLastEncode: () => LastEncode | null;
      runEncode: () => Promise<void>;
      setActiveTab: (name: string) => void;
      getEqStates: () => Map<number, ComponentEqState>;
      getCurrentOptions: () => EncodeOptions;
    };
  }
}
window.__zenjpegDiagnostics = {
  getCurrent: () => currentDiagnostics,
  getLastEncode: () => lastEncode,
  runEncode,
  setActiveTab,
  getEqStates: () => eqStates,
  getCurrentOptions: () => currentOptions(),
};
