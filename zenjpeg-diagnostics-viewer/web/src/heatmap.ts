// Tiny dependency-free heatmap renderer. Maps a 2D scalar field onto a
// canvas with a viridis-ish gradient and pixelated upscaling.

export interface HeatmapOptions {
  /** Width in cells (input data is row-major, length = w*h). */
  cols: number;
  rows: number;
  /** Output canvas pixel size. */
  pixelSize: number;
  /** Optional explicit min/max; otherwise auto. */
  min?: number;
  max?: number;
  /** Optional value formatter for accessible per-cell title. */
  format?: (v: number) => string;
}

const STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [13, 8, 135]], // dark purple
  [0.25, [84, 2, 163]],
  [0.5, [156, 23, 158]],
  [0.75, [225, 100, 98]],
  [1.0, [253, 231, 37]], // bright yellow
];

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function viridis(t: number): [number, number, number] {
  if (t <= 0) return STOPS[0]![1];
  if (t >= 1) return STOPS[STOPS.length - 1]![1];
  for (let i = 0; i < STOPS.length - 1; i++) {
    const a = STOPS[i]!;
    const b = STOPS[i + 1]!;
    if (t >= a[0] && t <= b[0]) {
      const tt = (t - a[0]) / (b[0] - a[0]);
      return [
        Math.round(lerp(a[1][0], b[1][0], tt)),
        Math.round(lerp(a[1][1], b[1][1], tt)),
        Math.round(lerp(a[1][2], b[1][2], tt)),
      ];
    }
  }
  return STOPS[STOPS.length - 1]![1];
}

export function drawHeatmap(
  canvas: HTMLCanvasElement,
  data: ArrayLike<number>,
  opts: HeatmapOptions,
): void {
  const { cols, rows, pixelSize } = opts;
  if (data.length !== cols * rows) {
    throw new Error(
      `heatmap data length ${data.length} ≠ cols×rows ${cols * rows}`,
    );
  }

  let min = opts.min;
  let max = opts.max;
  if (min === undefined || max === undefined) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const v = data[i]!;
      if (Number.isFinite(v)) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    min ??= lo;
    max ??= hi;
  }
  const span = max - min;

  canvas.width = cols * pixelSize;
  canvas.height = rows * pixelSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  // Use ImageData for crisp per-cell colors.
  const img = ctx.createImageData(cols * pixelSize, rows * pixelSize);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data[r * cols + c]!;
      const t = span > 0 ? (v - min) / span : 0;
      const [rr, gg, bb] = viridis(t);
      for (let py = 0; py < pixelSize; py++) {
        for (let px = 0; px < pixelSize; px++) {
          const ox = c * pixelSize + px;
          const oy = r * pixelSize + py;
          const idx = (oy * cols * pixelSize + ox) * 4;
          img.data[idx] = rr;
          img.data[idx + 1] = gg;
          img.data[idx + 2] = bb;
          img.data[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
