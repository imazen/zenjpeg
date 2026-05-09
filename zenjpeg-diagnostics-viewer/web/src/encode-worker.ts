// Web Worker: runs the wasm encode pipeline off the main thread.
//
// Protocol (postMessage in both directions):
//
//   Main → Worker:
//     { kind: "encode",
//       reqId: number,            // unique per request
//       pixels: Uint8Array,       // packed RGB, length = w*h*3
//       width: number,
//       height: number,
//       options: EncodeOptions }
//
//   Worker → Main:
//     { kind: "ready" }                                      // after wasm init
//     { kind: "result", reqId, bytes, diagnostics, ssim2, zensim, durationMs }
//     { kind: "error",  reqId, message }
//
// The worker imports the wasm-pack module directly. Because Vite is
// configured with `worker.format: "es"`, the worker module is loaded
// as ES with native dynamic imports.
//
// Cancellation model: the worker doesn't currently honor mid-encode
// cancellation (would require SharedArrayBuffer + COOP/COEP). Instead,
// the main thread tracks the `latest reqId` and discards results that
// don't match — see main.ts for the discard logic.

import init, {
  computeSsimulacra2,
  computeZensim,
  encodeWithDiagnostics,
} from "../wasm-pkg/zenjpeg_diagnostics_wasm.js";
import type { Diagnostics, EncodeOptions } from "./types";

interface EncodeReq {
  kind: "encode";
  reqId: number;
  pixels: Uint8Array;
  width: number;
  height: number;
  options: EncodeOptions;
  /** Decoded RGB of the *source* (not the JPEG output) for metric
   *  scoring. Caller already has it on the main thread; passing it
   *  in saves us from re-decoding inside the worker. */
  sourceRgb: Uint8Array;
  /** Whether to compute SSIMULACRA 2 + zensim after encode. Skipped
   *  for the pareto-curve sweep where only bytes + score are needed
   *  and we score in batch. */
  scoreMetrics: boolean;
}

interface EncodeResultMsg {
  kind: "result";
  reqId: number;
  bytes: Uint8Array;
  diagnostics: Diagnostics;
  ssim2: number | null;
  zensim: number | null;
  durationMs: number;
}

interface ErrorMsg {
  kind: "error";
  reqId: number;
  message: string;
}

interface ReadyMsg {
  kind: "ready";
}

let initPromise: Promise<unknown> | null = null;

async function ensureInit(): Promise<void> {
  if (!initPromise) {
    initPromise = init();
  }
  await initPromise;
}

function postReady(): void {
  const msg: ReadyMsg = { kind: "ready" };
  (self as unknown as Worker).postMessage(msg);
}

function postError(reqId: number, message: string): void {
  const msg: ErrorMsg = { kind: "error", reqId, message };
  (self as unknown as Worker).postMessage(msg);
}

async function decodeJpegInWorker(
  bytes: Uint8Array,
  w: number,
  h: number,
): Promise<Uint8Array | null> {
  // Workers have OffscreenCanvas + createImageBitmap, but image
  // decoding via these is browser-implemented and we don't actually
  // need the decoded bytes inside the worker — metric scoring uses
  // the encoded JPEG decoded on the main thread (where we already
  // have to draw it to a canvas anyway). Return null so the worker
  // doesn't re-decode.
  void bytes;
  void w;
  void h;
  return null;
}

self.addEventListener("message", async (ev: MessageEvent) => {
  const data = ev.data as EncodeReq;
  if (!data || data.kind !== "encode") return;
  const { reqId, pixels, width, height, options, sourceRgb, scoreMetrics } = data;
  const t0 = performance.now();
  try {
    await ensureInit();
    const result = (await encodeWithDiagnostics(
      pixels,
      width,
      height,
      options,
    )) as { bytes: Uint8Array; diagnostics: Diagnostics };

    let ssim2: number | null = null;
    let zensim: number | null = null;
    if (scoreMetrics) {
      // We need the *decoded* JPEG bytes for scoring. The main thread
      // does that decode (via <img>), but for the worker we can't
      // easily get HTMLImageElement. Defer scoring to main thread.
      // (See main.ts — it calls computeSsimulacra2 / computeZensim
      //  on its own once the JPEG is decoded.)
      void sourceRgb;
      void decodeJpegInWorker;
    }

    const t1 = performance.now();
    const msg: EncodeResultMsg = {
      kind: "result",
      reqId,
      bytes: result.bytes,
      diagnostics: result.diagnostics,
      ssim2,
      zensim,
      durationMs: t1 - t0,
    };
    (self as unknown as Worker).postMessage(msg, [
      result.bytes.buffer as ArrayBuffer,
    ]);
  } catch (e) {
    postError(reqId, (e as Error)?.message ?? String(e));
  }
});

// Bootstrap.
ensureInit()
  .then(() => postReady())
  .catch((e) => postError(-1, (e as Error)?.message ?? String(e)));

/// Score helpers for main-thread use after JPEG decode. Exporting these
/// here so the main thread can call them via a separate worker
/// instance (cheap because wasm init is cached). Not strictly needed —
/// we re-export the wasm module's score fns from main.ts directly
/// (kept synchronous in the main thread since metric compute is fast).
export type { EncodeReq, EncodeResultMsg, ErrorMsg, ReadyMsg };
export { computeSsimulacra2, computeZensim };
