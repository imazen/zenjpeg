> **Historical record.** The `archmage-simd` feature named below was
> removed — archmage SIMD is always compiled now. Current platform
> coverage is enforced by `.github/workflows/ci.yml` (including
> windows-11-arm, macos-26-intel, and i686 via cross); treat that file
> as the source of truth.

## Cross-Platform Build Status

Successfully building for all targets with archmage-simd feature:
- ✅ x86_64 (AVX2/AVX-512 via mage_simd.rs)
- ✅ aarch64 (NEON via arm_simd.rs)
- ✅ wasm32 (SIMD128 via wasm_simd.rs)

## Implemented (Encode Path)

**All platforms:**
- 8x8 DCT with FMA optimization (ARM has baseline FMA, WASM uses mul+add)
- 4x4 and 8x8 transpose primitives
- Butterfly operations for 2, 4, and 8-point DCT

**ARM NEON specific:**
- Uses vfmaq_f32 for fused multiply-add
- 4-wide processing (128-bit vectors)

**WASM SIMD128 specific:**
- Uses f32x4 operations
- i32x4_shuffle for transpose/interleaving
- No FMA (2x ops vs ARM/x86)

> **Completed.** The dispatch logic, decode hot paths (IDCT with AVX2/NEON/WASM SIMD128), and AQ SIMD operations listed below have all been implemented.

## Next Steps

1. ~~**Add dispatch logic**~~ - Done. archmage runtime dispatch is wired up for all platforms.
2. ~~**Implement decode hot paths**~~ - Done. IDCT (AVX2/NEON/WASM SIMD128), color conversion, and upsampling are implemented.
3. ~~**Add AQ operations**~~ - Done. pre_erosion_row and per_block_modulations use archmage SIMD dispatch.
4. **Create benchmarks** - Measure actual performance on each platform
5. **Optimize** - Replace any remaining scalar fallbacks with proper SIMD implementations

## Build Commands

```bash
# x86_64 (native)
cargo build -p zenjpeg --lib --release --features archmage-simd

# aarch64 (cross-compile)
cargo build -p zenjpeg --lib --target aarch64-unknown-linux-gnu --release --features archmage-simd

# wasm32 (with SIMD128)
RUSTFLAGS="-C target-feature=+simd128" cargo build -p zenjpeg --lib --target wasm32-unknown-unknown --release --features archmage-simd
```


