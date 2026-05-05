// Entry point for the zenjpeg encode-diagnostics demo viewer.
//
// Flow:
//   1. Load the wasm-pack module.
//   2. Render synthetic source pattern (or load user file).
//   3. Call encodeWithDiagnostics with the current control state.
//   4. Decode the encoded bytes via the browser's <img> for display.
//   5. Render diagnostics: AQ map, per-component utilization heatmap,
//      live re-quant on EQ slider drag.
//
// All UI state lives in module-level mutable refs; this is a small
// vanilla TS app — no framework, no virtual DOM.

import init, { encodeWithDiagnostics } from "../wasm-pkg/zenjpeg_diagnostics_wasm.js";
import { drawHeatmap } from "./heatmap";
import { syntheticPattern } from "./synthetic";
import type {
  ComponentDiagnostics,
  Diagnostics,
  EncodeOptions,
  EncodeResult,
} from "./types";

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`element not found: ${sel}`);
  return el as T;
};

const status = $<HTMLParagraphElement>("#status");
const qualityIn = $<HTMLInputElement>("#quality");
const qualityOut = $<HTMLOutputElement>("#quality-out");
const colorPathSel = $<HTMLSelectElement>("#color-path");
const subSel = $<HTMLSelectElement>("#subsampling");
const xybSubSel = $<HTMLSelectElement>("#xyb-subsampling");
const subLabel = $<HTMLLabelElement>("#subsampling-label");
const xybLabel = $<HTMLLabelElement>("#xyb-subsampling-label");
const aqIn = $<HTMLInputElement>("#aq-enabled");
const trellisIn = $<HTMLInputElement>("#trellis");
const autoOptIn = $<HTMLInputElement>("#auto-optimize");
const fileIn = $<HTMLInputElement>("#image-input");
const resetBtn = $<HTMLButtonElement>("#reset");
const encodeBtn = $<HTMLButtonElement>("#encode");
const sourceCanvas = $<HTMLCanvasElement>("#source-canvas");
const encodedCanvas = $<HTMLCanvasElement>("#encoded-canvas");
const deltaCanvas = $<HTMLCanvasElement>("#delta-canvas");
const aqCanvas = $<HTMLCanvasElement>("#aq-canvas");
const compRoot = $<HTMLElement>("#component-0").parentElement!;

interface ImageState {
  width: number;
  height: number;
  /** Source RGBA, length = width*height*4. */
  rgba: Uint8ClampedArray;
}

interface ComponentEqState {
  /** Length 8, multiplier per horizontal-frequency column. */
  hEq: number[];
  /** Length 8, multiplier per vertical-frequency row. */
  vEq: number[];
}

let currentImage: ImageState = synthetic();
let currentDiagnostics: Diagnostics | null = null;
const eqStates = new Map<number, ComponentEqState>();

function synthetic(): ImageState {
  const w = 64;
  const h = 64;
  return {
    width: w,
    height: h,
    rgba: syntheticPattern(w, h),
  };
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
    out.data[i + 1] = Math.min(
      255,
      Math.abs(src.data[i + 1]! - dec.data[i + 1]!) * gain,
    );
    out.data[i + 2] = Math.min(
      255,
      Math.abs(src.data[i + 2]! - dec.data[i + 2]!) * gain,
    );
    out.data[i + 3] = 255;
  }
  return out;
}

function rgbaToPackedRgb(rgba: Uint8ClampedArray): Uint8Array {
  const n = rgba.length / 4;
  const out = new Uint8Array(n * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    out[j] = rgba[i]!;
    out[j + 1] = rgba[i + 1]!;
    out[j + 2] = rgba[i + 2]!;
  }
  return out;
}

function currentOptions(): EncodeOptions {
  return {
    quality: parseInt(qualityIn.value, 10),
    colorPath: colorPathSel.value as "ycbcr" | "xyb",
    subsampling: subSel.value as EncodeOptions["subsampling"],
    xybSubsampling: xybSubSel.value as EncodeOptions["xybSubsampling"],
    aqEnabled: aqIn.checked,
    trellis: trellisIn.checked,
    autoOptimize: autoOptIn.checked,
    deringing: true,
  };
}

function eqStateFor(componentIdx: number, comp: ComponentDiagnostics): ComponentEqState {
  let s = eqStates.get(componentIdx);
  if (!s) {
    s = { hEq: Array(8).fill(1), vEq: Array(8).fill(1) };
    eqStates.set(componentIdx, s);
  }
  void comp; // keep signature stable for future per-component init
  return s;
}

function effectiveQuant(comp: ComponentDiagnostics, eq: ComponentEqState): Float32Array {
  const out = new Float32Array(64);
  for (let v = 0; v < 8; v++) {
    for (let u = 0; u < 8; u++) {
      const idx = v * 8 + u;
      const base = comp.quantTableBase[idx]! as number;
      out[idx] = base * eq.hEq[u]! * eq.vEq[v]!;
    }
  }
  return out;
}

/**
 * For each natural-order coefficient position, compute the mean
 * |coef_pre_quant| / effective_quant across all blocks. Values >> 1 mean
 * the coefficient has lots of headroom (table overshoots); << 1 means
 * the position is mostly quantized to zero.
 */
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
      const ratio = Math.abs(c[i]! as number) / qv;
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
  return {
    field,
    zeroRate: totalCoef > 0 ? (zeroCoef / totalCoef) * 100 : 0,
  };
}

function buildQuantGrid(
  el: HTMLElement,
  comp: ComponentDiagnostics,
  eq: ComponentEqState,
): void {
  el.replaceChildren();
  const eff = effectiveQuant(comp, eq);
  for (let v = 0; v < 8; v++) {
    for (let u = 0; u < 8; u++) {
      const cell = document.createElement("div");
      const idx = v * 8 + u;
      const val = eff[idx]!;
      cell.textContent = val < 100 ? val.toFixed(0) : Math.round(val).toString();
      cell.title = `(${u},${v}) base=${comp.quantTableBase[idx]} effective=${val.toFixed(1)}`;
      el.appendChild(cell);
    }
  }
}

function buildEqSliders(
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
    h.dataset.testid = `comp-0-h-eq-${i}`;
    h.title = `h_eq[${i}]: column ${i} multiplier`;
    h.addEventListener("input", () => {
      eq.hEq[i] = parseFloat(h.value);
      onChange();
    });
    hEl.appendChild(h);

    const vv = document.createElement("input");
    vv.type = "range";
    vv.min = "0.5";
    vv.max = "2";
    vv.step = "0.05";
    vv.value = String(eq.vEq[i]!);
    vv.dataset.testid = `comp-0-v-eq-${i}`;
    vv.title = `v_eq[${i}]: row ${i} multiplier`;
    vv.addEventListener("input", () => {
      eq.vEq[i] = parseFloat(vv.value);
      onChange();
    });
    vEl.appendChild(vv);
  }
}

function renderComponent(
  componentIdx: number,
  diag: Diagnostics,
): void {
  const comp = diag.components[componentIdx];
  if (!comp) return;
  let article = document.querySelector<HTMLElement>(
    `[data-component="${componentIdx}"]`,
  );
  if (!article) {
    // Clone component-0's structure for additional components.
    const tmpl = document.querySelector<HTMLElement>('[data-component="0"]');
    if (!tmpl) throw new Error("template article missing");
    article = tmpl.cloneNode(true) as HTMLElement;
    article.dataset.component = String(componentIdx);
    article.id = `component-${componentIdx}`;
    // Patch test-ids.
    article.querySelectorAll<HTMLElement>("[data-testid^='comp-0-']").forEach((el) => {
      el.dataset.testid = el.dataset.testid!.replace(
        "comp-0-",
        `comp-${componentIdx}-`,
      );
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

  const eq = eqStateFor(componentIdx, comp);
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
  const resetBtn = article.querySelector<HTMLButtonElement>(
    `[data-testid="comp-${componentIdx}-reset-eq"]`,
  )!;

  const refresh = () => {
    buildQuantGrid(quantGridEl, comp, eq);
    const q = effectiveQuant(comp, eq);
    const { field, zeroRate } = utilizationField(comp, q);
    drawHeatmap(utilCanvas, field, { cols: 8, rows: 8, pixelSize: 25 });
    zeroRateEl.textContent = zeroRate.toFixed(1);
  };

  buildEqSliders(hEqEl, vEqEl, eq, refresh);
  resetBtn.addEventListener("click", () => {
    eq.hEq.fill(1);
    eq.vEq.fill(1);
    article!
      .querySelectorAll<HTMLInputElement>(
        `[data-testid^="comp-${componentIdx}-h-eq-"], [data-testid^="comp-${componentIdx}-v-eq-"]`,
      )
      .forEach((el) => {
        el.value = "1";
      });
    refresh();
  });
  refresh();
}

function componentLabel(colorPath: string, idx: number): string {
  if (colorPath === "XYB") {
    return ["X", "Y (XYB)", "B"][idx] ?? `c${idx}`;
  }
  if (colorPath === "Grayscale") return "Y";
  return ["Y", "Cb", "Cr"][idx] ?? `c${idx}`;
}

function renderAqField(diag: Diagnostics): void {
  const yComp = diag.components[0];
  if (!yComp) return;
  const [cols, rows] = yComp.blockGrid;
  const data = new Float32Array(cols * rows);
  for (let i = 0; i < yComp.blocks.length; i++) {
    data[i] = yComp.blocks[i]!.aqMultiplier;
  }
  drawHeatmap(aqCanvas, data, {
    cols,
    rows,
    pixelSize: Math.max(8, Math.min(40, Math.floor(480 / cols))),
  });
}

async function runEncode(): Promise<void> {
  if (!currentImage) return;
  document.body.dataset.encodePhase = "start";
  status.textContent = "Encoding…";
  encodeBtn.disabled = true;
  try {
    const packed = rgbaToPackedRgb(currentImage.rgba);
    const opts = currentOptions();
    document.body.dataset.encodePhase = "calling-wasm";
    const result = (await encodeWithDiagnostics(
      packed,
      currentImage.width,
      currentImage.height,
      opts,
    )) as EncodeResult;
    document.body.dataset.encodePhase = "wasm-returned";

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

    currentDiagnostics = result.diagnostics;
    eqStates.clear();
    renderAqField(result.diagnostics);
    // Remove additional component articles we may have appended.
    compRoot
      .querySelectorAll<HTMLElement>("[data-component]")
      .forEach((el) => {
        if (el.dataset.component !== "0") el.remove();
      });
    for (let i = 0; i < result.diagnostics.components.length; i++) {
      renderComponent(i, result.diagnostics);
    }
    status.textContent = `Encoded ${result.bytes.byteLength} bytes (${result.diagnostics.components.length} components, ${
      result.diagnostics.components[0]?.blocks.length ?? 0
    } Y blocks)`;
    document.body.dataset.encodeState = "done";
  } catch (e) {
    console.error(e);
    status.textContent = `Error: ${(e as Error).message}`;
    document.body.dataset.encodeState = "error";
  } finally {
    encodeBtn.disabled = false;
  }
}

async function loadFile(file: File): Promise<void> {
  const url = URL.createObjectURL(file);
  try {
    const img = await new Promise<HTMLImageElement>((res, rej) => {
      const i = new Image();
      i.onload = () => res(i);
      i.onerror = rej;
      i.src = url;
    });
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const cx = c.getContext("2d");
    if (!cx) throw new Error("canvas 2d unavailable");
    cx.drawImage(img, 0, 0);
    const data = cx.getImageData(0, 0, w, h);
    currentImage = { width: w, height: h, rgba: data.data };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function wireControls(): void {
  qualityIn.addEventListener("input", () => {
    qualityOut.textContent = qualityIn.value;
  });
  colorPathSel.addEventListener("change", () => {
    if (colorPathSel.value === "xyb") {
      subLabel.hidden = true;
      xybLabel.hidden = false;
    } else {
      subLabel.hidden = false;
      xybLabel.hidden = true;
    }
  });
  encodeBtn.addEventListener("click", () => {
    void runEncode();
  });
  resetBtn.addEventListener("click", () => {
    currentImage = synthetic();
    void runEncode();
  });
  fileIn.addEventListener("change", async () => {
    const file = fileIn.files?.[0];
    if (!file) return;
    await loadFile(file);
    await runEncode();
  });
}

(async () => {
  document.body.dataset.bootstrap = "started";
  status.textContent = "Loading WASM…";
  try {
    await init();
    document.body.dataset.bootstrap = "wasm-init-ok";
  } catch (err) {
    document.body.dataset.bootstrap = "wasm-init-failed";
    document.body.dataset.encodeState = "error";
    status.textContent = `WASM init failed: ${(err as Error).message ?? err}`;
    console.error("WASM init", err);
    return;
  }
  status.textContent = "WASM loaded — encoding synthetic 64×64 sample";
  document.body.dataset.wasmState = "ready";
  try {
    wireControls();
    document.body.dataset.bootstrap = "controls-wired";
  } catch (err) {
    document.body.dataset.bootstrap = "controls-wire-failed";
    document.body.dataset.encodeState = "error";
    status.textContent = `wireControls failed: ${(err as Error).message ?? err}`;
    console.error("wireControls", err);
    return;
  }
  await runEncode();
})();

// Expose for Playwright.
declare global {
  interface Window {
    __zenjpegDiagnostics: {
      getCurrent: () => Diagnostics | null;
      runEncode: () => Promise<void>;
    };
  }
}
window.__zenjpegDiagnostics = {
  getCurrent: () => currentDiagnostics,
  runEncode,
};
