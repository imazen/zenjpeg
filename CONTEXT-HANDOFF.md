# ARM/WASM SIMD Implementation - Context Handoff (Updated)

**Session Date:** 2026-02-07
**Branch:** `perf/arm-wasm-archmage`
**Worktree:** `/home/lilith/work/zenjpeg-arm-wasm`
**Main Repo:** `/home/lilith/work/zenjpeg` (on `main` branch)
**Last Commit:** `b61e3d1 fix: restore unsafe blocks for ARM/WASM SIMD (archmage limitation)`

## 🔍 KEY FINDING: Archmage ARM/WASM Limitation

**CRITICAL:** Archmage's `#[arcane]` macro does NOT work for ARM NEON or WASM SIMD128 yet.

From archmage's own `cross_platform.rs` example (line 114):
```rust
// Note: ARM NEON support requires fixing some missing intrinsics in the generator.
// For now, use scalar fallback on aarch64. When fixed, this will use #[arcane].
```

**What this means:**
- The `#[arcane]` macro works for x86 (AVX2/AVX-512) only
- ARM and WASM intrinsics are still `unsafe` and require manual `unsafe` blocks
- The token system (`NeonToken`, `Wasm128Token`) exists but doesn't make intrinsics safe
- **We cannot use `#![forbid(unsafe_code)]` globally** - only conditionally on x86

**Current solution:**
```rust
// In zenjpeg/src/lib.rs
#![cfg_attr(
    all(
        not(any(target_arch = "aarch64", target_arch = "wasm32")),
        feature = "archmage-simd"
    ),
    forbid(unsafe_code)
)]
```

## What We Accomplished

### ✅ Fixed All Build Errors

1. **Removed duplicate imports** - `pub use archmage::NeonToken` was imported twice
2. **Fixed syntax errors** - Backslash escapes (`\!=`, `assert_eq\!`)
3. **Fixed module paths** - Removed broken `super::super::decode::idct_int` references
4. **Added proper unsafe blocks** - All `#[arcane]` functions now have manual `unsafe { ... }`
5. **Stubbed incomplete decoder functions** - YCbCr conversion and upsampling are TODO

### ✅ Build Status (Commit b61e3d1)

| Target | Status | Warnings | Notes |
|--------|--------|----------|-------|
| x86_64 (native) | ✅ Compiles | 4 | No unsafe needed (archmage works) |
| aarch64-unknown-linux-gnu | ✅ Compiles | 38 | Requires unsafe blocks |
| wasm32-unknown-unknown | ✅ Compiles | 37 | Requires unsafe blocks |

**Build commands:**
```bash
# x86_64 (native)
cargo build -p zenjpeg --lib --release --features archmage-simd

# aarch64
cargo build -p zenjpeg --lib --target aarch64-unknown-linux-gnu --release --features archmage-simd

# wasm32
RUSTFLAGS="-C target-feature=+simd128" cargo build -p zenjpeg --lib --target wasm32-unknown-unknown --release --features archmage-simd
```

## Implementation Status

### Encoder (Forward Path)

| Component | ARM NEON | WASM SIMD128 | Status |
|-----------|----------|--------------|--------|
| 4x4 transpose | ✅ | ✅ | Complete |
| 8x8 transpose | ✅ | ✅ | Complete |
| 2-point DCT butterfly | ✅ | ✅ | Complete |
| 4-point DCT butterfly | ✅ | ✅ | Complete |
| 8-point DCT butterfly | ✅ | ✅ | Complete |
| Forward DCT 8x8 | ✅ | ✅ | Complete |

### Decoder (Reverse Path)

| Component | ARM NEON | WASM SIMD128 | Status |
|-----------|----------|--------------|--------|
| Integer IDCT 8x8 | ❌ | ❌ | Stub (unimplemented!) |
| YCbCr→RGB conversion | ❌ | ❌ | Stub (unimplemented!) |
| H2V1 upsampling | ❌ | ❌ | Stub (unimplemented!) |

**Decoder functions currently panic with `unimplemented!`**

## Next Steps

### Priority 1: Remove CONTEXT-HANDOFF.md After Reading

Per project CLAUDE.md: Delete this file after loading into new session.

### Priority 2: Complete Decoder IDCT (Critical Path)

**Reference implementation:** `~/work/zune-image/crates/zune-jpeg/src/idct/neon.rs`

Key components to port:
1. **DC-only fast path** - when all AC coefficients are zero (already stubbed)
2. **Two-pass 1D IDCT** - Loeffler algorithm with NEON intrinsics
3. **Fixed-point arithmetic** - 13-bit constants, proper rounding
4. **Level shift and clamp** - Output range [0, 255]

**ARM NEON specific patterns from zune-jpeg:**
- `vmlal_laneq_s16` - multiply-accumulate with lane selection
- `vqmovn_s32` - saturating narrow i32→i16
- `vqshrun_n_s32` - saturating shift-right and narrow to unsigned

**WASM SIMD128 differences:**
- No `vmlal` equivalent - manual widening with `i32x4_extend_low_i16x8`
- No FMA - separate multiply and add operations
- No lane selection - use shuffle to broadcast values

### Priority 3: Color Conversion

**Reference:** `~/work/zune-image/crates/zune-jpeg/src/color_convert/neon64.rs`

Key operations:
1. Unbias Cb/Cr (subtract 128)
2. Widen i16→i32 for precision
3. Multiply-accumulate with 14-bit fixed-point coefficients
4. Shift right by 14, saturate to [0, 255]
5. Interleave R/G/B into RGB triplets

### Priority 4: Chroma Upsampling

**Reference:** `~/work/zune-image/crates/zune-jpeg/src/upsampler/neon.rs`

Implement H2V1, H1V2, and H2V2 upsampling modes with triangle filter.

### Priority 5: Integration & Dispatch

Wire up runtime dispatch in actual encoder/decoder:

```rust
pub fn forward_dct_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    #[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
    {
        if let Some(token) = archmage::X64V3Token::summon() {
            return mage_simd::mage_forward_dct_8x8(token, input, output);
        }
    }

    #[cfg(all(feature = "archmage-simd", target_arch = "aarch64"))]
    {
        if let Some(token) = archmage::NeonToken::summon() {
            return arm_simd::neon_forward_dct_8x8(token, input, output);
        }
    }

    #[cfg(all(feature = "archmage-simd", target_arch = "wasm32"))]
    {
        if let Some(token) = archmage::Wasm128Token::summon() {
            return wasm_simd::wasm_forward_dct_8x8(token, input, output);
        }
    }

    // Scalar fallback
    forward_dct_8x8_scalar(input, output)
}
```

Apply to: `encode/dct.rs`, `decode/idct_int.rs`, `color/ycbcr.rs`, `decode/upsample.rs`

### Priority 6: Benchmarks

Create `benches/cross_platform.rs` to measure ARM/WASM performance vs scalar.

Target speedups (based on CLAUDE.md doc):
- **Encode:** ARM/WASM should match x86 AVX2 when normalized for vector width
- **Decode:** ARM/WASM 1.5-2.0x faster than scalar

## Files Modified

```
zenjpeg/src/
├── lib.rs                   # Conditional forbid(unsafe_code) for x86 only
└── encode/
    ├── arm_simd.rs          # 9 unsafe blocks, encoder complete, decoder stubs
    └── wasm_simd.rs         # 9 unsafe blocks, encoder complete, decoder stubs

docs/
├── SIMD_INTRINSICS_MAP.md   # x86→ARM→WASM intrinsics reference (from previous session)
└── CROSS_PLATFORM_STATUS.md # Build status (from previous session)

CONTEXT-HANDOFF.md           # THIS FILE - delete after reading!
```

## Key Technical Decisions

### Vector Width Strategy

**Problem:** NEON/WASM are 128-bit (4-wide), AVX2 is 256-bit (8-wide)

**Solution:** Process 8x8 operations as four independent 4x4 blocks:
```rust
let tl = transpose_4x4([r0_lo, r1_lo, r2_lo, r3_lo]);  // top-left
let tr = transpose_4x4([r0_hi, r1_hi, r2_hi, r3_hi]);  // top-right
let bl = transpose_4x4([r4_lo, r5_lo, r6_lo, r7_lo]);  // bottom-left
let br = transpose_4x4([r4_hi, r5_hi, r6_hi, r7_hi]);  // bottom-right
```

### FMA Differences

| Platform | FMA Support | Pattern |
|----------|-------------|---------|
| ARM NEON | Baseline `vfmaq_f32` | `vfmaq_f32(c, a, b)` → `a*b + c` (1 op) |
| WASM SIMD128 | None | `f32x4_add(f32x4_mul(a, b), c)` (2 ops) |
| x86 AVX2 | `_mm256_fmadd_ps` | `a*b + c` (1 op) |

**Impact:** WASM needs 2× as many operations for multiply-accumulate patterns in DCT butterflies.

## Documentation References

- **archmage README:** `~/.cargo/registry/src/.../archmage-0.5.0/README.md`
- **archmage cross-platform example:** `~/.cargo/registry/src/.../archmage-0.5.0/examples/cross_platform.rs`
- **zune-jpeg NEON IDCT:** `~/work/zune-image/crates/zune-jpeg/src/idct/neon.rs`
- **zune-jpeg NEON color:** `~/work/zune-image/crates/zune-jpeg/src/color_convert/neon64.rs`
- **zune-jpeg NEON upsample:** `~/work/zune-image/crates/zune-jpeg/src/upsampler/neon.rs`
- **SIMD intrinsics map:** `docs/SIMD_INTRINSICS_MAP.md`

## Lessons Learned

### 1. Archmage ARM/WASM Support is Incomplete

The previous session incorrectly assumed `#[arcane]` worked for all platforms. It only works for x86.
**Always check the crate's own examples before assuming API coverage.**

### 2. Download Crate Sources for API Research

Web docs and search results are often outdated. The archmage cross-platform example revealed the
ARM limitation immediately, but web search would have missed it.

### 3. Unsafe Blocks Were Needed All Along

The handoff document said "remove all unsafe blocks" based on Rust 1.86+ making intrinsics safe
inside `#[target_feature]`. This is true, but archmage doesn't generate `#[target_feature]` for
ARM/WASM yet, so unsafe is still required.

## Performance Expectations

From project CLAUDE.md:

**Current x86 baseline:**
- Encode (4K image): DCT ~8.6%, AQ ~14.5% of total time
- Decode (2048x2048): Scanline-420 4.03ms, Baseline-fast 4.72ms

**Expected ARM/WASM:**
- Encode: Match x86 when normalized for vector width (4-wide vs 8-wide)
- Decode: 1.5-2.0x speedup over scalar (documented in CLAUDE.md)

## Summary

**What works:** ARM and WASM encoder (forward DCT path) compiles and should function correctly.

**What doesn't:** Decoder functions are stubbed and will panic if called. IDCT, color conversion,
and upsampling need to be ported from zune-jpeg NEON implementations.

**Next session:** Start with Priority 2 (decoder IDCT) using zune-jpeg as reference. The encoder
is ready for testing/benchmarking once you integrate dispatch logic.

---

*Updated: 2026-02-07*
*Branch: `perf/arm-wasm-archmage`*
*Commit: `b61e3d1 fix: restore unsafe blocks for ARM/WASM SIMD (archmage limitation)`*
