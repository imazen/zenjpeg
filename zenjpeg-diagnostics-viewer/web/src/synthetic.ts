// Synthetic deterministic test image — sine modulation per channel so
// every 8×8 block has non-zero AC energy. Mirrors the Rust smoke-test
// pattern exactly, so dev-loop output is reproducible without
// requiring an external image upload.

export function syntheticPattern(width: number, height: number): Uint8ClampedArray {
  const buf = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const fx = x;
      const fy = y;
      const r = 128 + 64 * Math.sin(fx * 0.3) + 32 * Math.cos((fx + fy) * 0.9);
      const g = 128 + 64 * Math.cos(fy * 0.25) + 32 * Math.sin((fx - fy) * 0.7);
      const b = 128 + 96 * Math.sin((fx + fy) * 0.15);
      buf[i] = r;
      buf[i + 1] = g;
      buf[i + 2] = b;
      buf[i + 3] = 255;
    }
  }
  return buf;
}
