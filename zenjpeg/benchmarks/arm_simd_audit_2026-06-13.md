# ARM64 SIMD-vs-scalar audit (2026-06-13)

Hardware: Hetzner aarch64, 8-core (`arm-big`), cargo 1.96. Real silicon, not QEMU.
Goal: verify zenjpeg's SIMD is wired up and optimal on ARM — the maintainer's
hypothesis is that scalar / `#[autoversion]` sometimes beats hand/portable SIMD on NEON.

## Headline result: the hypothesis holds for the hot color converter

The magetypes-generic `ycbcr_to_rgb_i16_x16` is **2.3× SLOWER than the
autovectorized scalar plane loop** on aarch64 for the same i16-planes → u8-RGB op:

| path | aarch64 throughput |
|---|---|
| scalar plane loop (`ycc_rgb_pixel`, the production ARM fallback) | ~770 Mops/s (1.30 ns/px) |
| magetypes-generic `ycbcr_to_rgb_i16_x16` (the "fix" the audit proposed) | ~342 Mops/s (2.93 ns/px) |

→ The plane converters' scalar fallback on ARM is **already optimal**; routing it
through a generic/NEON converter would regress ARM 2.3×. Locked in with a code
comment at `ycbcr_planes_i16_to_rgb_u8`. (On x86 the hand AVX2/AVX-512 path wins,
so the const-generic dispatch — x86 SIMD, ARM scalar — is correct on both.)

## IDCT (idct_kernels): direct scalar-vs-NEON A/B

| group | kernel | scalar ns/blk | NEON ns/blk | NEON speedup |
|---|---|---|---|---|
| dense | jpegli-12bit (default) | 147.3 | 137.1 | 1.07× (marginal) |
| sparse8 | jpegli-12bit | 146.1 | 137.5 | 1.06× (marginal) |
| dense | libjpeg-13bit | 250.0 | 167.6 | 1.49× |
| sparse8 | libjpeg-13bit | 221.9 | 166.4 | 1.33× |

The default jpegli-12bit IDCT gains only 6-7% from NEON (vs ~4× on x86 AVX2),
because `idct_int_wide` does a **scalar transpose between the two SIMD passes**
(`idct_int.rs`: "magetypes i32x8 has no transpose_8x8"). Scalar is a near-equal
fallback on ARM. The libjpeg-13bit kernel still gains 1.33-1.49× (heavier i64
scalar path). A native NEON 8×8 transpose would be needed to lift the 12-bit gain.

## Where NEON DOES win on ARM (so it's not one-sided)

| kernel | NEON vs scalar (aarch64) |
|---|---|
| fused h2v2 box upsample+convert (`fused_h2v2_box_ycbcr_to_rgb_u8`, default) | 1.56-1.58× (308 vs 195 Mops/s) |
| libjpeg-13bit IDCT | 1.33-1.49× |

Note: the `IdctMethod::Libjpeg` turbo fused-box ARM path is scalar (the ARM turbo
SIMD kernel was never written) — leaves ~1.6× on the floor for that opt-in path.

## Single-variant absolute throughput (no scalar A/B in-bench)

- forward DCT (both NEON-generic): recursive 75.1 ns/blk, AAN 74.1 ns/blk.
- color_convert: `ycbcr_to_rgb_i16_x16` 2.93 ns/px, `fused_h2v2_box` 3.18 ns/px,
  `decode_420_e2e` 20.66 ms/MP.
- AQ (NEON-generic): pre_erosion 1.46 ns/px, pre_erosion_padded 1.60, modulation 1.94.

## Conclusions / actions

1. **No "wire the generic on ARM" action for the plane converters** — scalar is
   faster there. Documented in source to prevent a future regression. (DONE)
2. **jpegli IDCT**: keep NEON (slightly ahead); scalar is an acceptable fallback.
   Only worth revisiting with a native NEON transpose.
3. **Recommend** (not done — needs maintainer call): delete dead `encode/arm_simd.rs`
   + `encode/wasm_simd.rs` (zero callers) UNLESS the hand-NEON DCT (which HAS a native
   transpose) is benchmarked to beat the transpose-capped generic forward DCT first.
4. **Recommend**: write the ARM turbo fused-box SIMD path (1.6× on the opt-in path).
