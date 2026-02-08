# SIMD Intrinsics Mapping: x86 → ARM NEON → WASM SIMD128

This document maps all x86 AVX2/AVX-512 intrinsics currently used in zenjpeg to their ARM NEON and WASM SIMD128 equivalents.

## Legend

- ✅ Direct equivalent exists
- ⚠️ Requires multiple ops or workaround
- ❌ No direct equivalent (need scalar fallback or emulation)
- 📝 Notes on precision/behavior differences

## Vector Width Mapping

| Platform | Register Width | f32 lanes | i32 lanes | i16 lanes |
|----------|---------------|-----------|-----------|-----------|
| AVX2 (__m256) | 256-bit | 8 | 8 | 16 |
| AVX-512 (__m512) | 512-bit | 16 | 16 | 32 |
| NEON (float32x4_t) | 128-bit | 4 | 4 | 8 |
| WASM (v128) | 128-bit | 4 | 4 | 8 |

**Key insight:** NEON and WASM are 128-bit (4 lanes), so we process half as much data per operation compared to AVX2 (8 lanes). This means:
- 2 NEON/WASM ops per AVX2 op for most operations
- 4 NEON/WASM ops per AVX-512 op
- Transpose and shuffle patterns need to be reworked for 4-wide vs 8-wide

## Floating-Point Operations (f32)

### Arithmetic

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_add_ps` | `vaddq_f32` ×2 | `f32x4_add` ×2 | ✅ |
| `_mm256_sub_ps` | `vsubq_f32` ×2 | `f32x4_sub` ×2 | ✅ |
| `_mm256_mul_ps` | `vmulq_f32` ×2 | `f32x4_mul` ×2 | ✅ |
| `_mm256_div_ps` | `vdivq_f32` ×2 | `f32x4_div` ×2 | ✅ (NEON ARMv8+) |
| `_mm256_fmadd_ps` | `vfmaq_f32` ×2 | `f32x4_add(f32x4_mul(...))` ×2 | ✅ NEON has FMA, WASM needs 2 ops |
| `_mm256_sqrt_ps` | `vsqrtq_f32` ×2 | `f32x4_sqrt` ×2 | ✅ |
| `_mm256_max_ps` | `vmaxq_f32` ×2 | `f32x4_max` ×2 | ✅ |
| `_mm256_min_ps` | `vminq_f32` ×2 | `f32x4_min` ×2 | ✅ |

### Bitwise (for f32 vectors)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_andnot_ps` | `vbicq_s32(b,a)` ×2 | `v128_andnot` ×2 | ✅ NEON uses integer type |
| `_mm256_castps_si256` | `vreinterpretq_s32_f32` | type cast | ✅ Zero-cost |
| `_mm256_castsi256_ps` | `vreinterpretq_f32_s32` | type cast | ✅ Zero-cost |

### Horizontal Operations

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_hadd_ps` | ⚠️ Complex | ⚠️ `f32x4_add_pairwise` | NEON has pairwise, but different semantics |

### Conversions

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_cvtps_epi32` | `vcvtq_s32_f32` ×2 | `i32x4_trunc_sat_f32x4` ×2 | ✅ Round to zero |
| `_mm256_cvtepi32_ps` | `vcvtq_f32_s32` ×2 | `f32x4_convert_i32x4` ×2 | ✅ |
| `_mm256_floor_ps` | `vrndmq_f32` ×2 | `f32x4_floor` ×2 | ✅ (NEON ARMv8+) |

### Shuffles & Permutes (f32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_unpacklo_ps` | `vzipq_f32` lower ×2 | `i32x4_shuffle` | ⚠️ Different semantics |
| `_mm256_unpackhi_ps` | `vzipq_f32` upper ×2 | `i32x4_shuffle` | ⚠️ Different semantics |
| `_mm256_shuffle_ps` | `vcombine_f32(vtbl...)` | `i32x4_shuffle` | ⚠️ Need pattern translation |
| `_mm256_permute2f128_ps` | `vextq_f32` or `vcombine` | `i8x16_shuffle` | ⚠️ For cross-lane ops |
| `_mm256_permute4x64_epi64` | ⚠️ Complex | `i64x2_shuffle` ×2 | ⚠️ Need decomposition |

**Transpose note:** 8x8 transpose for AVX2 becomes two separate 4x4 transposes for NEON/WASM.

### Load/Store (f32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_loadu_ps` | `vld1q_f32` ×2 | `v128_load` ×2 | ✅ Unaligned |
| `_mm256_storeu_ps` | `vst1q_f32` ×2 | `v128_store` ×2 | ✅ Unaligned |

### Broadcast/Splat (f32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_set1_ps` | `vdupq_n_f32` | `f32x4_splat` | ✅ |
| `_mm256_setzero_ps` | `vdupq_n_f32(0.0)` | `f32x4_const(0,0,0,0)` | ✅ |
| `_mm256_set_ps(h,g,f,e,d,c,b,a)` | `vcombine_f32(vcreate...)` | Manual lane set | ⚠️ Verbose |

### Extract (f32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_extractf128_ps` | Register splitting | `i64x2_shuffle` | ✅ Just use 2 regs |
| `_mm256_castps256_ps128` | Lower half | Lower half | ✅ Free |

## Integer Operations (i32)

### Arithmetic

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_add_epi32` | `vaddq_s32` ×2 | `i32x4_add` ×2 | ✅ |
| `_mm256_sub_epi32` | `vsubq_s32` ×2 | `i32x4_sub` ×2 | ✅ |
| `_mm256_mullo_epi32` | `vmulq_s32` ×2 | `i32x4_mul` ×2 | ✅ |

### Bitwise

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_and_si256` | `vandq_s32` ×2 | `v128_and` ×2 | ✅ |
| `_mm256_or_si256` | `vorrq_s32` ×2 | `v128_or` ×2 | ✅ |

### Shifts

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_slli_epi32` | `vshlq_n_s32` ×2 | `i32x4_shl` ×2 | ✅ |
| `_mm256_srli_epi32` | `vshrq_n_u32` ×2 | `u32x4_shr` ×2 | ✅ Logical shift |
| `_mm256_srai_epi32` | `vshrq_n_s32` ×2 | `i32x4_shr` ×2 | ✅ Arithmetic shift |

### Broadcast/Splat (i32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_set1_epi32` | `vdupq_n_s32` | `i32x4_splat` | ✅ |
| `_mm256_setzero_si256` | `vdupq_n_s32(0)` | `i32x4_const(0,0,0,0)` | ✅ |

### Shuffles & Permutes (i32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_unpacklo_epi32` | `vzipq_s32` lower ×2 | `i32x4_shuffle` | ⚠️ |
| `_mm256_unpackhi_epi32` | `vzipq_s32` upper ×2 | `i32x4_shuffle` | ⚠️ |
| `_mm256_unpacklo_epi64` | `vzip1q_s64` ×2 | `i64x2_shuffle` | ⚠️ |
| `_mm256_unpackhi_epi64` | `vzip2q_s64` ×2 | `i64x2_shuffle` | ⚠️ |
| `_mm256_permute2x128_si256` | `vextq_s32` or `vcombine` | `i8x16_shuffle` | ⚠️ |

### Load/Store (i32)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_loadu_si256` | `vld1q_s32` ×2 | `v128_load` ×2 | ✅ |
| `_mm256_storeu_si256` | `vst1q_s32` ×2 | `v128_store` ×2 | ✅ |

## Integer Operations (i16)

### Arithmetic

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_add_epi16` | `vaddq_s16` ×2 | `i16x8_add` ×2 | ✅ |
| `_mm256_sub_epi16` | `vsubq_s16` ×2 | `i16x8_sub` ×2 | ✅ |
| `_mm256_mullo_epi16` | `vmulq_s16` ×2 | `i16x8_mul` ×2 | ✅ |
| `_mm256_max_epi16` | `vmaxq_s16` ×2 | `i16x8_max` ×2 | ✅ |
| `_mm256_min_epi16` | `vminq_s16` ×2 | `i16x8_min` ×2 | ✅ |

### Shifts

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_srai_epi16` | `vshrq_n_s16` ×2 | `i16x8_shr` ×2 | ✅ Arithmetic |

### Pack/Unpack

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_packs_epi32` | `vqmovn_s32` + `vcombine` | ⚠️ Complex | Saturating pack 32→16 |
| `_mm256_packus_epi16` | `vqmovun_s16` + `vcombine` | ⚠️ Complex | Unsigned saturating pack 16→8 |
| `_mm256_unpacklo_epi16` | `vzip1q_s16` ×2 | `i16x8_shuffle` | ⚠️ |
| `_mm256_unpackhi_epi16` | `vzip2q_s16` ×2 | `i16x8_shuffle` | ⚠️ |

### Broadcast/Splat (i16)

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_set1_epi16` | `vdupq_n_s16` | `i16x8_splat` | ✅ |

## Special Operations

### Test/Compare

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_testz_si256` | `vmaxvq_u32(vandq(...))` | Manual reduction | ⚠️ Test if AND is all-zero |

### Blend

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_blendv_epi8` | `vbslq_s8` ×2 | `v128_bitselect` ×2 | ✅ Bitwise select |

### Byte Shuffle

| x86 AVX2 | ARM NEON | WASM SIMD128 | Notes |
|----------|----------|--------------|-------|
| `_mm256_shuffle_epi8` | `vqtbl1q_s8` ×2 | ⚠️ No direct equiv | NEON has table lookup |
| `_mm256_setr_epi8` | Manual lane set | `i8x16_const` | ⚠️ Verbose |

## AVX-512 Operations

| x86 AVX-512 | ARM NEON | WASM SIMD128 | Notes |
|-------------|----------|--------------|-------|
| `_mm512_add_ps` | `vaddq_f32` ×4 | `f32x4_add` ×4 | 4× the work |
| `_mm512_fmadd_ps` | `vfmaq_f32` ×4 | `f32x4_add(f32x4_mul(...))` ×4 | 4× the work |
| `_mm512_extractf32x8_ps` | Split to 2 regs | Split to 2 regs | ✅ |
| `_mm512_insertf32x8` | Combine 2 regs | Combine 2 regs | ✅ |

**Note:** AVX-512 is 16-wide, so all operations need 4× NEON/WASM equivalents. The dual-block DCT experiment (commit 2026-01-21) shows this is often slower due to overhead.

## Platform-Specific Optimizations

### ARM NEON Advantages

1. **FMA is baseline** - `vfmaq_f32` on all ARMv8+, no feature detection needed
2. **Table lookup** - `vqtbl1q_s8` for efficient byte permutations
3. **Saturating arithmetic** - `vqadd`, `vqsub` for overflow protection
4. **Paired operations** - `vpaddq_f32` for horizontal sums

### WASM SIMD128 Limitations

1. **No FMA** - Need separate mul + add (2 ops instead of 1)
2. **No horizontal operations** - Need manual reduction for sum/max across lanes
3. **Limited shuffles** - `i8x16_shuffle` is powerful but verbose

### Migration Strategy

1. **Start with DCT/IDCT** - These are the hottest paths and most critical for performance
2. **Use 4-wide transposes** - Redesign 8x8 transpose as two 4x4 transposes
3. **Leverage FMA on NEON** - Use `vfmaq_f32` for all a*b+c operations
4. **Test on real hardware** - Apple Silicon (M1/M2) for ARM, browser for WASM
5. **Maintain scalar fallback** - Keep autovectorized `wide` code as baseline

## File-by-File Migration Plan

### 1. `encode/mage_simd.rs` (DCT, math functions)

**Priority:** Highest (10-20% of encode time)

- `mage_forward_dct_8x8` - Needs 4-wide transpose, FMA butterfly ops
- `mage_transpose_8x8_inplace` - Redesign as two 4x4 transposes
- `mage_fast_log2_x8` / `mage_fast_exp2_x8` - Need NEON/WASM polynomial eval
- `mage_ratio_of_derivatives_x8` - FMA chains

### 2. `quant/aq/simd.rs` (Adaptive quantization)

**Priority:** High (14.5% of encode time)

- `mage_pre_erosion_row_padded_v4` - Currently AVX-512, need 4-wide version
- `mage_per_block_modulations_row` - FMA for masking calculations

### 3. `color/ycbcr.rs` (Color conversion)

**Priority:** Medium (decoder hot path)

- RGB→YCbCr conversion with FMA
- YCbCr→RGB with box/fancy upsampling
- Fused upsample + color convert (decoder optimization)

### 4. `decode/upsample.rs` (Chroma upsampling)

**Priority:** Medium (decoder hot path)

- Triangle filter (3:1 weighting) for H2V1, H1V2, H2V2
- Box filter for fast mode

### 5. `decode/idct_int.rs` (Integer IDCT)

**Priority:** Medium (decoder hot path, integer ops)

- Integer butterfly operations with shifts
- Transpose 8x8 i16 (redesign as 4x4)

## Next Steps

1. ✅ Create this mapping document
2. ⬜ Implement ARM NEON 4x4 transpose primitive
3. ⬜ Port `mage_forward_dct_8x8` to NEON (two 4x4 blocks)
4. ⬜ Benchmark NEON DCT vs x86 AVX2 (normalize for vector width)
5. ⬜ Port to WASM SIMD128 (reuse structure from NEON)
6. ⬜ Add cross-platform benchmarks
7. ⬜ Integrate into existing dispatch pattern
