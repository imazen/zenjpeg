# Changelog

## 0.1.0 - 2026-04-14

Initial release.

### Encode (RGB → YCbCr)
- BT.601, BT.709, BT.2020 matrices
- Full and Limited (studio) range
- 4:4:4 and 4:2:0 subsampling
- AVX2 `#[arcane]` kernel (32 pixels/iter, pmaddwd + pshufb deinterleave)
- NEON and WASM SIMD128 kernels
- magetypes generic fallback (all platforms)
- Sharp YUV iterative chroma optimization (L2-optimal Newton step, 2 iterations)

### Decode (YCbCr → RGB)
- 4:4:4, 4:2:0, 4:2:2, 4:0:0 (grayscale)
- Bilinear chroma upsampling for 4:2:0
- magetypes generic (AVX2 decode skeleton exists but not dispatched)

### API
- `YuvContext` — reusable encoder/decoder context, 28 bytes on stack
- Lazy workspace allocation (sharp: 432 bytes heap, f32 temps: variable)
- `#![no_std]` + `#![forbid(unsafe_code)]`

### Performance (vs `yuv` crate 0.8, 7950X AVX2)
- 4:4:4 encode: 6-11% faster at 256-512px, equal at 1024+
- 4:2:0 encode: matched (±4%)
- Sharp YUV: 25× faster than the original scalar implementation
