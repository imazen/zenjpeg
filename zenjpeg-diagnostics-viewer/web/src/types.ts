// Mirror of the Rust `Diagnostics` + `EncodeOptions` structs emitted by
// zenjpeg-diagnostics-wasm via serde_wasm_bindgen.

export interface BlockDiagnostics {
  /** [f32; 64] in natural row-major order (DC at index 0). */
  coefPreQuant: Float32Array | number[];
  /** [i16; 64] in JPEG zigzag order. */
  coefLevels: Int16Array | number[];
  /** Per-block AQ multiplier (1.0 = neutral, <1 = finer, >1 = coarser). */
  aqMultiplier: number;
  /** Entropy bits attributed to this block (0 if not yet captured). */
  entropyBits: number;
}

export interface ComponentDiagnostics {
  /** JFIF component id (1=Y/X, 2=Cb/Y-XYB, 3=Cr/B-XYB). */
  componentId: number;
  /** [cols, rows] in 8x8 blocks. */
  blockGrid: [number, number];
  /** Base quantization table (natural row-major, length 64). */
  quantTableBase: Uint16Array | number[];
  /** Zero-bias offset table (natural row-major, length 64). */
  zeroBias: Float32Array | number[];
  /** Per-block records, raster order (idx = row*cols + col). */
  blocks: BlockDiagnostics[];
}

export interface Diagnostics {
  width: number;
  height: number;
  /** "YCbCr" | "XYB" | "Grayscale" */
  colorPath: string;
  /** Per-component (h_samp, v_samp). */
  samplingFactors: Array<[number, number]>;
  components: ComponentDiagnostics[];
}

/** f32 quant tables + optional zero-bias overrides. */
export interface CustomQuantTables {
  /** Y / X quant table (length 64, natural row-major, DC first). */
  y: number[];
  /** Cb / Y-XYB quant table. */
  cb: number[];
  /** Cr / B-XYB quant table. */
  cr: number[];
  yZeroBiasMul?: number[];
  cbZeroBiasMul?: number[];
  crZeroBiasMul?: number[];
  /** [Y, Cb, Cr] DC offsets. */
  zeroBiasOffsetDc?: [number, number, number];
  zeroBiasOffsetAc?: [number, number, number];
}

/** "baseline" | "trellis" | "hybrid". */
export type EncodeMode = "baseline" | "trellis" | "hybrid";

/** "thorough" | "balanced" | "fast". */
export type TrellisSpeed = "thorough" | "balanced" | "fast";

export interface EncodeOptions {
  // ── Always-applied basics ─────────────────────────────────────────────
  /** jpegli quality, 0-100. */
  quality?: number;
  colorPath?: "ycbcr" | "xyb";
  subsampling?: "none" | "halfHorizontal" | "quarter" | "halfVertical";
  xybSubsampling?: "full" | "bQuarter";
  aqEnabled?: boolean;
  deringing?: boolean;
  optimizeHuffman?: boolean;
  progressive?: boolean;
  /** YCbCr only, ignored for 4:4:4 and XYB. */
  sharpYuv?: boolean;
  /** Pre-encode Gaussian blur sigma (px). 0 disables. */
  preBlur?: number;
  /** Multiplicative scale on chroma's perceptual distance budget. */
  chromaDistanceScale?: number;
  /** Restart-marker cadence in MCU rows. 0 disables. */
  restartMcuRows?: number;

  // ── Mode picker ───────────────────────────────────────────────────────
  mode?: EncodeMode;

  // ── Standalone trellis (mode === "trellis") ───────────────────────────
  trellisDcEnabled?: boolean;
  trellisLambdaLogScale1?: number;
  trellisLambdaLogScale2?: number;
  trellisSpeedMode?: TrellisSpeed;
  trellisDeltaDcWeight?: number;

  // ── Hybrid (mode === "hybrid") ────────────────────────────────────────
  hybridAqLambdaScale?: number;
  hybridBaseLambdaScale1?: number;
  hybridBaseLambdaScale2?: number;
  hybridDcEnabled?: boolean;
  hybridAqExponent?: number;
  hybridAqThreshold?: number;
  hybridQualityAdaptive?: boolean;

  // ── Custom quant tables (overrides the source-derived defaults) ───────
  customQuantTables?: CustomQuantTables | null;
}

export interface EncodeResult {
  bytes: Uint8Array;
  diagnostics: Diagnostics;
}
