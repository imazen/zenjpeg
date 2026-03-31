# wide→magetypes Migration Context Handoff

## Branch: `refactor/wide-to-magetypes`
## Worktree: `/home/lilith/work/zen/zenjpeg-wide-migration`

## Completed (6 commits)

1. **Foundation storage** (`simd_types.rs`): `Block8x8f`, `Block8x8i16`, `Block8x8i32`, `QuantTableSimd`, `ZeroBiasSimd` now use `[[f32; 8]; 8]` raw arrays. Pod/Zeroable/const preserved.

2. **Cold-path scalar conversions**: `trellis/hybrid.rs`, `deblock/boundary.rs`, `decode/image.rs`, `aligned_alloc.rs`, `chroma.rs`, `quant/mod.rs`, `quant/aq/mod.rs`, `deringing.rs`. These use scalar loops (OK for cold paths).

3. **Magetypes generic conversion**: `deblock/knusperli.rs` — properly converted to `magetypes::simd::generic::f32x8<T>` with `#[arcane]` + `incant!` dispatch. This is the CORRECT pattern.

4. **Array API conversion**: `linear_lut.rs` + `strip/convert.rs` — changed public API from `wide::f32x8` to `[f32; 8]` arrays. Scalar loops internally. **TODO: should be revisited to use magetypes generics for the hot tone-mapping path.**

5. **Scalar fallback + mage hot path**: `simd_types.rs` quantize fallbacks are scalar, archmage mage paths unchanged. `blocks.rs` scalar fallback, archmage path unchanged.

## Remaining (9 wide import sites, 7 files)

### Must use magetypes generics (NOT scalar downgrade):

| File | Occurrences | Pattern needed |
|------|------------|---------------|
| `quant/aq/simd.rs` | 83 | Generic `f32x8<T>` + `incant!` for `#[autoversion]` functions. Keep `mage_*` archmage paths unchanged. |
| `encode_simd.rs` | 140 | Generic `f32x8<T>` + `incant!` for pixel conversions. |
| `color/ycbcr.rs` | 107 | Replace `_wide` suffix functions with generic `f32x8<T>`. Keep `mage_*` unchanged. |
| `color/xyb.rs` | 130 | Replace `_wide` suffix functions with generic. Keep `mage_*` unchanged. Uses `f64x2` too. |
| `encode/dct.rs` | ~50 | Do last. Keep concrete mage path for SIMD transpose. Generic fallback. |
| `decode/idct.rs` | ~20 | Do last. Same as DCT. |
| `decode/idct_int.rs` | ~10 | Do last. Uses `i32x8`. |

### The correct pattern (from knusperli.rs):

```rust
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8 as gf32x8;

// Public entry point
fn foo(data: &mut [f32]) {
    archmage::incant!(foo_impl(data));
}

// AVX2 variant
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn foo_impl_v3(_token: archmage::X64V3Token, data: &mut [f32]) {
    foo_generic(_token, data);
}

// Scalar fallback
fn foo_impl_scalar(_token: archmage::ScalarToken, data: &mut [f32]) {
    foo_generic(_token, data);
}

// Generic: one implementation, monomorphized per backend
#[inline(always)]
fn foo_generic<T: F32x8Backend>(token: T, data: &mut [f32]) {
    let v = gf32x8::load(token, data[..8].try_into().unwrap());
    let result = v * gf32x8::splat(token, 2.0);
    result.store(<&mut [f32; 8]>::try_from(&mut data[..8]).unwrap());
}
```

### Key: `gf32x8<X64V3Token>` uses `__m256` (real AVX2). `gf32x8<ScalarToken>` uses `[f32; 8]`.

### For files with existing `mage_*` archmage paths:
- Keep the `mage_*` functions unchanged (they use concrete magetypes types with SIMD intrinsics)
- Replace the `wide` fallback with a generic version using `ScalarToken`
- Or: make the mage functions call the generic version if the archmage path doesn't need concrete types

## Benchmark baseline
- Saved at `zenjpeg/.zenbench/baselines/pre-migration.json`
- Compare: `cargo bench -p zenjpeg --bench wide_migration --features "decoder,trellis,parallel" -- --baseline=pre-migration`

## Tests
- 806 lib tests pass on current commit
- Run full: `cargo test -p zenjpeg --lib --release`
