// Mirror of the Rust `Diagnostics` struct emitted by
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

export interface EncodeOptions {
  quality?: number;
  colorPath?: "ycbcr" | "xyb";
  subsampling?: "none" | "halfHorizontal" | "quarter" | "halfVertical";
  xybSubsampling?: "full" | "bQuarter";
  aqEnabled?: boolean;
  trellis?: boolean;
  autoOptimize?: boolean;
  deringing?: boolean;
}

export interface EncodeResult {
  bytes: Uint8Array;
  diagnostics: Diagnostics;
}
