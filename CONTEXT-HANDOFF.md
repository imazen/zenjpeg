# Context Handoff: Replace unsafe_simd with magetypes

## Branch
`feat/trained-tables-streaming-v2` @ `8a0313a` (pushed to remote)

## Goal
Replace all `#[cfg(feature = "unsafe_simd")]` code with safe `magetypes` equivalents, then remove the `unsafe_simd` feature entirely. Do NOT replace `wide` crate usage broadly — only the code currently behind `unsafe_simd`.

## Why magetypes, not wide
`wide` uses compile-time `cfg(target_feature)` — it doesn't see the `#[target_feature]` attribute that `#[multiversed]` sets per function. So `wide::f32x8` inside a multiversed AVX2 function still uses SSE-level operations. This is documented in `zenjpeg/src/quant/aq/autovec.rs:9-12`.

`magetypes` wraps `__m256` directly and uses real AVX2 intrinsics. The `unsafe` is encapsulated inside the type — callers use safe arithmetic (`rows[0] + rows[1]`). Construction is token-gated via `archmage::Avx2FmaToken`.

## magetypes primitives available (v0.1.0)
- `magetypes::simd::f32x8` — wraps `__m256`, full arithmetic ops, `load`, `splat`, `from_array`, `store`, `to_array`
- `f32x8::transpose_8x8(&mut [Self; 8])` — in-place 8x8 transpose (Highway algorithm)
- `f32x8::transpose_8x8_copy([Self; 8]) -> [Self; 8]` — copy transpose
- `f32x8::load_8x8(&[f32; 64]) -> [Self; 8]` — load full block
- `f32x8::store_8x8(&[Self; 8], &mut [f32; 64])` — store full block
- `magetypes::simd::i16x16` — wraps `__m256i`, for decoder IDCT
- `magetypes::simd::i32x8` — wraps `__m256i`
- Feature: `magetypes-simd = ["dep:archmage", "dep:magetypes"]` already defined in Cargo.toml
- Workspace dep: `magetypes = { version = "0.1.0", features = ["bytemuck"] }`
- Source at: `~/work/downloaded-crates/magetypes-0.1.0/`

## Existing archmage note
Cargo.toml has `# Note: archmage-simd disabled from default - causes Huffman encoding bugs`. This may be stale — the archmage-simd feature uses a separate DCT in `encode/mage_simd.rs` which may produce slightly different coefficients. The new magetypes approach should produce identical results to the current unsafe_simd code since it uses the same intrinsics.

## Files to modify (6 files, ~35 cfg sites)

### 1. `zenjpeg/src/encode/dct.rs` (13 cfg sites)
**Functions to replace:**
- `transpose_8x8_avx(input, output)` (line 456) — raw `__m256` load/unpack/permute/store
- `transpose_8x8_avx_inplace(r: &mut [__m256; 8])` (line 787) — in-place transpose
- `dct1d_2_fma`, `dct1d_4_fma`, `dct1d_8_fma` (lines 824-926) — raw FMA butterflies
- `forward_dct_8x8_fma(input, output)` (line 948) — full 2D DCT orchestrator

**Dispatch sites:**
- `forward_dct_8x8` (line 1041) — `#[multiversed]`, branches on `cfg(target_feature = "avx2", target_feature = "fma")` to call `forward_dct_8x8_fma`
- `transpose_8x8_simd` (line 517) — `#[multiversed]`, calls `transpose_8x8_avx` when detected
- `dct_8rows_parallel` (line 364) — `#[multiversed]`, calls `transpose_8x8_avx`

**Strategy:** Replace raw intrinsics with `magetypes::simd::f32x8` ops. The DCT butterflies are just add/sub/mul/fma which magetypes supports. Transpose uses `f32x8::transpose_8x8`. Gate behind `#[cfg(feature = "magetypes-simd")]` instead of `unsafe_simd`.

### 2. `zenjpeg/src/encode_simd.rs` (13 cfg sites)
**Functions to replace:**
- `gather_even_odd_x8_avx2` (line 239) — gather even/odd pixels for chroma subsampling
- `rgb_to_ycbcr_8px_avx2` (line 368) — RGB→YCbCr for 8 pixels
- `rgb_to_ycbcr_8px_fma` (line 379) — same with FMA
- `rgb_to_ycbcr_8px_gather_avx2` (line 390) — gather variant
- `rgb_to_ycbcr_8px_gather_fma` (line 402) — gather+FMA
- `rgb_to_ycbcr_8px_gather_fma_fused` (line 424) — fully fused

**Dispatch sites:**
- `downsample_2x2_simd_inplace` (line 340) — calls `gather_even_odd_x8_avx2`
- `rgb_to_ycbcr_planes_simd_inplace` (line 564) — calls `rgb_to_ycbcr_8px_fma`

### 3. `zenjpeg/src/decode/upsample.rs` (2 cfg sites)
**Functions to replace:**
- `upsample_h2v2_i16_fancy_avx2` (line 345) — fancy 2x2 upsampling with i16 math
- Dispatch in `upsample_h2v2_i16_fancy` (line 298)

**Note:** Uses `__m256i` for i16x16 operations. magetypes has `i16x16`.

### 4. `zenjpeg/src/decode/idct_int.rs` (3 cfg sites)
**Functions to replace:**
- `idct_int_4x4_avx2_dc_only` (line 337) — DC-only 4x4 IDCT
- `idct_int_4x4_avx2` (line 346) — full 4x4 IDCT
- `idct_int_8x8_avx2` (line 414) — full 8x8 IDCT

**Dispatch sites in `idct_int_8x8`** (line 888) and `idct_int_4x4` (lines 705, 763)

**Note:** Heavy i32x8 and i16x16 usage. Check magetypes has `_mm256_packs_epi32`, `_mm256_madd_epi16` equivalents.

### 5. `zenjpeg/src/color/ycbcr.rs` (2 cfg sites)
**Functions to replace:**
- `rgb_to_ycbcr_f32_8px_avx2` (line 937) — f32 RGB→YCbCr
- `ycbcr_to_rgb_i16_8px_avx2` (line 1208) — i16 YCbCr→RGB

### 6. `zenjpeg/src/foundation/simd_types.rs` (4 cfg sites)
**Code to replace:**
- `get_unchecked_mut` in `quantize_to_zigzag` (line 444) — just remove, use safe indexing
- `get_unchecked_mut` in `quantize_block` (line 498) — just remove, use safe indexing

## Baseline benchmarks (with unsafe_simd, 2026-01-30)

### Encode (512x512 reference)
- encode/rgb/512x512:    1.32 ms
- encode/rgb/1024x1024:  5.49 ms
- encode/rgb/2048x2048:  23.15 ms
- quality/q/75:          1.24 ms
- quality/q/90:          1.29 ms

### DCT (16384 blocks)
- dct/recursive:         545 µs (30.0 Melem/s)
- dct/aan:               563 µs (29.1 Melem/s)
- dct_single/recursive:  54.3 ns
- dct_single/aan:        50.5 ns

### Decode
- decode/rgb/512x512:    482 µs
- decode/rgb/1024x1024:  1.88 ms
- decode/rgb/2048x2048:  9.62 ms

## Implementation order
1. `simd_types.rs` — trivial, just delete get_unchecked paths
2. `encode/dct.rs` — magetypes has all primitives ready
3. `encode_simd.rs` — RGB→YCbCr + gather
4. `color/ycbcr.rs` — similar to encode_simd
5. `decode/upsample.rs` — i16 upsampling
6. `decode/idct_int.rs` — integer IDCT (most complex, check magetypes i16/i32 ops)
7. Remove `unsafe_simd` feature from Cargo.toml defaults and lib.rs
8. Update README
9. Run benchmarks, compare to baseline

## Key decisions
- Gate new code behind `magetypes-simd` feature (already defined, pulls in archmage + magetypes)
- Make `magetypes-simd` a default feature (replacing `unsafe_simd` in defaults)
- The `#[multiversed]` dispatch stays — magetypes functions are called from within multiversed functions when token is available
- `wide` stays for everything NOT currently behind `unsafe_simd`
- After migration, `forbid(unsafe_code)` can be unconditional (only archmage-simd needs the exception)

## Token pattern
```rust
// In #[multiversed] function:
#[cfg(all(feature = "magetypes-simd", target_arch = "x86_64"))]
{
    if let Some(token) = archmage::Avx2FmaToken::try_new() {
        let rows = magetypes::simd::f32x8::load_8x8(input);
        // ... do work with magetypes f32x8 ...
        magetypes::simd::f32x8::store_8x8(&result, output);
        return;
    }
}
// fallback to wide/scalar
```

## Check before starting
- `cargo search magetypes` — verify 0.1.0 is latest
- `cargo search archmage` — verify 0.2.1 is latest
- Check if magetypes has `fmadd` (fused multiply-add) — needed for DCT
- Check if magetypes i16x16 has pack/unpack/madd — needed for decoder IDCT
