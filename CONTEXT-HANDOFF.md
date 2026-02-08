# ARM/WASM SIMD Implementation - Context Handoff

**Session Date:** 2026-02-07
**Branch:** `perf/arm-wasm-archmage`
**Worktree:** `/home/lilith/work/zenjpeg-arm-wasm`
**Main Repo:** `/home/lilith/work/zenjpeg` (on `main` branch)

## CRITICAL ISSUES TO FIX IMMEDIATELY

### 🚨 UNSAFE CODE VIOLATIONS

**ALL SIMD code currently uses `unsafe` blocks, which violates the new global policy.**

Per updated `~/.claude/CLAUDE.md` (lines 55-66):
- **FORBIDDEN:** All projects must use `#![forbid(unsafe_code)]` at crate root
- **Rust 1.86+ SIMD intrinsics are SAFE inside `#[target_feature]` functions**
- **Archmage `#[arcane]` macro provides safe dispatch** - no unsafe needed

**Files with unsafe violations:**
- `zenjpeg/src/encode/arm_simd.rs` - ALL functions use `unsafe { ... }` blocks
- `zenjpeg/src/encode/wasm_simd.rs` - ALL functions use `unsafe { ... }` blocks
- `zenjpeg/src/encode/mage_simd.rs` - x86 code also has unsafe (pre-existing)

**What archmage `#[arcane]` does:**
```rust
// WRONG (current code):
#[arcane]
fn foo(token: NeonToken, data: &[f32]) {
    unsafe {  // ❌ FORBIDDEN - remove this
        let v = vld1q_f32(data.as_ptr());
        // ...
    }
}

// RIGHT (what we need):
#[arcane]
fn foo(token: NeonToken, data: &[f32]) {
    // ✅ NO UNSAFE - archmage makes intrinsics safe
    let v = vld1q_f32(data.as_ptr());
    // ...
}
```

The `#[arcane]` macro generates an inner `#[target_feature]` function, which makes ALL intrinsics safe to call (as of Rust 1.86+). We should NEVER have `unsafe` blocks inside `#[arcane]` functions.

**Fix steps:**
1. Add `#![forbid(unsafe_code)]` to `zenjpeg/src/lib.rs`
2. Remove ALL `unsafe { ... }` blocks from `arm_simd.rs`
3. Remove ALL `unsafe { ... }` blocks from `wasm_simd.rs`
4. Test compilation - it should just work
5. File the same issue for `mage_simd.rs` (x86 code)

**Reference:** Read `~/work/downloaded-crates/archmage-*/README.md` for archmage patterns.

## What We Accomplished

### ✅ Completed Tasks

1. **Comprehensive SIMD Audit** (Task #1)
   - Created `docs/SIMD_INTRINSICS_MAP.md` - 282 lines
   - Maps ALL x86 AVX2/AVX-512 intrinsics to ARM NEON and WASM SIMD128
   - Documents vector width differences (128-bit vs 256-bit)
   - FMA availability (ARM: baseline, WASM: none, x86: conditional)
   - File-by-file migration plan with priorities

2. **ARM NEON Implementation** (Task #3 - in progress)
   - `zenjpeg/src/encode/arm_simd.rs` - 383 lines
   - 4x4 transpose (vzip1q/vzip2q pattern)
   - 8x8 transpose (four 4x4 blocks)
   - DCT butterfly operations (2, 4, 8-point)
   - Forward DCT 8x8
   - Builds successfully: `cargo build --target aarch64-unknown-linux-gnu --release --features archmage-simd`

3. **WASM SIMD128 Implementation** (Task #4 - in progress)
   - `zenjpeg/src/encode/wasm_simd.rs` - 299 lines
   - 4x4 transpose (i32x4_shuffle/i64x2_shuffle)
   - 8x8 transpose (four 4x4 blocks)
   - DCT butterfly operations (no FMA - uses mul+add)
   - Forward DCT 8x8
   - Builds successfully: `RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release --features archmage-simd`

4. **Build Infrastructure**
   - Module declarations in `encode/mod.rs` with proper cfg gates
   - All three targets build cleanly (x86_64, aarch64, wasm32)
   - `CROSS_PLATFORM_STATUS.md` with build commands

### 🔄 In Progress

**Encode path:** DCT works, needs AQ operations
**Decode path:** Stubbed, needs implementation:
- Integer IDCT (critical - hot path)
- YCbCr↔RGB color conversion
- Chroma upsampling (H2V1, H1V2, H2V2)

### ❌ Not Started

- Dispatch logic (runtime CPU detection and path selection)
- Benchmarks for ARM/WASM
- Parity tests
- Integration into existing encoder/decoder

## Key Technical Decisions

### Vector Width Strategy

**Problem:** NEON/WASM are 128-bit (4-wide), AVX2 is 256-bit (8-wide)

**Solution:** Process 8x8 operations as four independent 4x4 blocks:
```rust
// 8x8 transpose = 4 separate 4x4 transposes + reassembly
let tl = transpose_4x4([r0_lo, r1_lo, r2_lo, r3_lo]);  // top-left
let tr = transpose_4x4([r0_hi, r1_hi, r2_hi, r3_hi]);  // top-right
let bl = transpose_4x4([r4_lo, r5_lo, r6_lo, r7_lo]);  // bottom-left
let br = transpose_4x4([r4_hi, r5_hi, r6_hi, r7_hi]);  // bottom-right
// Reassemble into transposed 8x8
```

This is fundamental and affects ALL operations (DCT, IDCT, AQ).

### FMA Differences

| Platform | FMA Support | Pattern |
|----------|-------------|---------|
| ARM NEON | Baseline `vfmaq_f32` | `vfmaq_f32(c, a, b)` → `a*b + c` (1 op) |
| WASM SIMD128 | None | `f32x4_add(f32x4_mul(a, b), c)` (2 ops) |
| x86 AVX2 | `_mm256_fmadd_ps` | `a*b + c` (1 op) |

**Impact:** WASM needs 2× as many operations for multiply-accumulate patterns. This affects DCT butterfly ops heavily.

### Reference Implementations

**zune-jpeg** has high-quality NEON implementations to reference:
- `~/work/zune-image/crates/zune-jpeg/src/idct/neon.rs` - Integer IDCT
- `~/work/zune-image/crates/zune-jpeg/src/upsampler/neon.rs` - Chroma upsampling
- `~/work/zune-image/crates/zune-jpeg/src/color_convert/neon64.rs` - YCbCr conversion

**Key patterns from zune-jpeg:**
- Uses `vmlal_laneq_s16` for multiply-accumulate with lane selection
- DC-only fast path (when all AC coefficients are zero)
- Saturating pack operations: `vqmovn_s32`, `vqshrun_n_s32`
- Widening ops: extend i16→i32 for precision

## Next Session Action Items

### Priority 1: Fix Unsafe Code (BLOCKING)

```bash
cd /home/lilith/work/zenjpeg-arm-wasm

# 1. Add forbid directive
echo '#![forbid(unsafe_code)]' | cat - zenjpeg/src/lib.rs > temp && mv temp zenjpeg/src/lib.rs

# 2. Remove all unsafe blocks
sed -i '/unsafe {/d; /^    }$/d' zenjpeg/src/encode/arm_simd.rs
sed -i '/unsafe {/d; /^    }$/d' zenjpeg/src/encode/wasm_simd.rs

# 3. Adjust indentation (all code was indented for unsafe block)
# Use your editor or a proper indentation fixer

# 4. Test build
cargo build --target aarch64-unknown-linux-gnu --release --features archmage-simd
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release --features archmage-simd

# Should compile without errors!
```

**Why this works:** Rust 1.86+ made SIMD intrinsics safe inside `#[target_feature]` functions. Archmage's `#[arcane]` macro generates those functions. We never needed `unsafe` - it was cargo-culted from old examples.

### Priority 2: Complete IDCT (Decoder Hot Path)

Port zune-jpeg's NEON IDCT to our codebase:

1. **Read the reference:**
   ```bash
   cat ~/work/zune-image/crates/zune-jpeg/src/idct/neon.rs | less
   ```

2. **Key components to port:**
   - DC-only fast path (all AC coeffs zero)
   - Two-pass 1D IDCT with in-register transpose
   - Fixed-point butterfly operations
   - Proper rounding and level shift

3. **Implementation in `arm_simd.rs`:**
   ```rust
   #[arcane]
   pub fn neon_idct_int_8x8(
       token: NeonToken,
       input: &[i32; 64],
       output: &mut [i16],
       stride: usize,
   ) {
       // DC-only path
       let all_ac_zero = input[1..].iter().all(|&x| x == 0);
       if all_ac_zero {
           let dc = ((input[0] + 4 + 1024) >> 3).clamp(0, 255) as i16;
           let dc_vec = vdupq_n_s16(dc);
           for row in 0..8 {
               vst1q_s16(output[row * stride..].as_mut_ptr(), dc_vec);
           }
           return;
       }

       // Full IDCT - two-pass algorithm
       // Pass 1: columns (natural order → transposed)
       // Pass 2: rows (transposed → natural order, level-shifted)
       // ... (port from zune-jpeg)
   }
   ```

4. **WASM equivalent** uses i32x4 operations (no vmlal, manual widening)

### Priority 3: Color Conversion

**ARM NEON version:**
```rust
#[arcane]
pub fn neon_ycbcr_to_rgb(
    token: NeonToken,
    y: &[i16; 16],    // Input: 16 Y samples
    cb: &[i16; 16],   // Input: 16 Cb samples
    cr: &[i16; 16],   // Input: 16 Cr samples
    rgb: &mut [u8; 48], // Output: 16 RGB pixels (48 bytes)
) {
    // Constants (14-bit fixed-point)
    const Y_COEFF: i16 = 19595;     // 1.1953 * 16384
    const CR_TO_R: i16 = 22970;     // 1.402 * 16384
    const CB_TO_B: i16 = 29032;     // 1.772 * 16384
    const CR_TO_G: i16 = -11698;    // -0.714 * 16384
    const CB_TO_G: i16 = -5636;     // -0.344 * 16384

    // Load as int16x8_t (process 8 pixels at a time)
    let y0 = vld1q_s16(y.as_ptr());
    let cb0 = vld1q_s16(cb.as_ptr());
    let cr0 = vld1q_s16(cr.as_ptr());

    // Unbias Cb/Cr (subtract 128)
    let cb_cr_bias = vdupq_n_s16(128);
    let cb0 = vsubq_s16(cb0, cb_cr_bias);
    let cr0 = vsubq_s16(cr0, cb_cr_bias);

    // Widen to i32, multiply, shift, saturate to u8
    // Use vmlal_lane_s16 pattern from zune-jpeg
    // ... (see zune-jpeg/src/color_convert/neon64.rs for full impl)
}
```

Reference: `~/work/zune-image/crates/zune-jpeg/src/color_convert/neon64.rs` lines 20-95

### Priority 4: Dispatch Integration

Wire up runtime dispatch in the actual encoder/decoder:

**Pattern:**
```rust
// In encode/dct.rs or wherever DCT is called
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

**Apply to:**
- `encode/dct.rs` - DCT forward
- `decode/idct_int.rs` - IDCT integer
- `color/ycbcr.rs` - YCbCr conversion
- `decode/upsample.rs` - Chroma upsampling
- `quant/aq/streaming.rs` - AQ operations

### Priority 5: Benchmarks

Create `zenjpeg/benches/cross_platform.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_dct_platforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("dct_8x8");
    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];

    #[cfg(target_arch = "x86_64")]
    group.bench_function("avx2", |b| {
        if let Some(token) = archmage::X64V3Token::summon() {
            b.iter(|| {
                zenjpeg::encode::mage_simd::mage_forward_dct_8x8(token, &input, &mut output);
            });
        }
    });

    #[cfg(target_arch = "aarch64")]
    group.bench_function("neon", |b| {
        if let Some(token) = archmage::NeonToken::summon() {
            b.iter(|| {
                zenjpeg::encode::arm_simd::neon_forward_dct_8x8(token, &input, &mut output);
            });
        }
    });

    group.bench_function("scalar", |b| {
        b.iter(|| {
            // scalar fallback
        });
    });

    group.finish();
}

criterion_group!(benches, bench_dct_platforms);
criterion_main!(benches);
```

Run with:
```bash
# x86_64
cargo bench --bench cross_platform --features archmage-simd

# aarch64 (needs actual ARM hardware or QEMU)
cargo bench --bench cross_platform --target aarch64-unknown-linux-gnu --features archmage-simd

# wasm32 (needs wasmtime or similar)
RUSTFLAGS="-C target-feature=+simd128" cargo bench --bench cross_platform --target wasm32-wasi --features archmage-simd
```

## Files Modified This Session

```
zenjpeg/src/encode/
├── mod.rs                   # Added arm_simd and wasm_simd module declarations
├── arm_simd.rs              # NEW - 383 lines (NEEDS unsafe removal)
└── wasm_simd.rs             # NEW - 299 lines (NEEDS unsafe removal)

docs/
├── SIMD_INTRINSICS_MAP.md   # NEW - 282 lines, comprehensive reference
└── CROSS_PLATFORM_STATUS.md # NEW - build status and commands

CONTEXT-HANDOFF.md           # THIS FILE
```

## Build Commands Reference

```bash
# x86_64 (native)
cargo build -p zenjpeg --lib --release --features archmage-simd

# aarch64 (cross-compile)
cargo build -p zenjpeg --lib --target aarch64-unknown-linux-gnu --release --features archmage-simd

# wasm32 (with SIMD128)
RUSTFLAGS="-C target-feature=+simd128" cargo build -p zenjpeg --lib --target wasm32-unknown-unknown --release --features archmage-simd

# Test all targets
rustup target list --installed | grep -E "aarch64|wasm32"  # Check installed
rustup target add aarch64-unknown-linux-gnu wasm32-unknown-unknown  # Add if missing
```

## Stashed Changes

There's one stash with attempted decoder implementations that had compilation errors:

```bash
git stash list
# stash@{0}: WIP on perf/arm-wasm-archmage: 074468d feat: add decoder SIMD support for ARM and WASM

# To see what's in the stash:
git stash show -p

# To apply (after fixing unsafe code issues):
git stash pop
```

The stash contains incomplete IDCT/color conversion code with type errors. Don't apply it - start fresh with the zune-jpeg reference implementations.

## Task Status

```
#1 [completed] Audit x86 SIMD intrinsics and map to ARM/WASM equivalents
#3 [in_progress] Implement ARM NEON versions of hot paths
#4 [in_progress] Implement WASM SIMD128 versions of hot paths
#2 [pending] Refactor SIMD dispatch to use archmage traits
#5 [pending] Add cross-platform SIMD benchmarks
#6 [pending] Test parity and accuracy across platforms
#7 [pending] Update documentation for multi-platform SIMD
```

## Commits This Session

```
0f5f745 docs: add cross-platform build status
074468d feat: add decoder SIMD support for ARM and WASM (BROKEN - has unsafe)
3939bce feat: add ARM NEON and WASM SIMD128 implementations (BROKEN - has unsafe)
10edd58 docs: add comprehensive x86→ARM→WASM SIMD intrinsics mapping
be4bb44 docs: add licensing info and update badge
```

Note: Commits 074468d and 3939bce introduce unsafe code violations. These need to be fixed.

## Performance Targets

Based on docs and reference implementations:

**Encode:**
- ARM NEON: Match x86 AVX2 when normalized for vector width (4-wide vs 8-wide)
- WASM SIMD128: 1.6-1.7x speedup over scalar (documented in CLAUDE.md)

**Decode:**
- ARM NEON: Match or beat zune-jpeg performance
- WASM SIMD128: 1.5-2.0x speedup over scalar (documented in CLAUDE.md)
- Scanline decoder should match x86 (already within 1% on x86)

**Current x86 baseline (from CLAUDE.md):**
- Encode (4K image): DCT ~8.6%, AQ ~14.5% of total time
- Decode (2048x2048): Scanline-420 4.03ms, Baseline-fast 4.72ms, Progressive 19.75ms

## Critical Gotchas

1. **UNSAFE CODE IS FORBIDDEN** - Remove all `unsafe` blocks, archmage makes intrinsics safe
2. **Module paths** - Use `crate::decode::idct_int::`, not `super::super::decode::idct_int::`
3. **Duplicate re-exports** - Only one `pub use archmage::Token` per file
4. **Vector width** - NEON/WASM process 4 elements, not 8 like AVX2
5. **FMA** - ARM has it baseline, WASM doesn't (2× ops for mul-add patterns)
6. **Transpose** - 8x8 must be decomposed into four 4x4 blocks
7. **Scalar fallbacks** - Always provide a scalar path for when SIMD unavailable

## Resources

- **zune-jpeg NEON code:** `~/work/zune-image/crates/zune-jpeg/src/`
- **archmage README:** `cargo download archmage` then read README in `~/work/downloaded-crates/`
- **SIMD intrinsics reference:** `docs/SIMD_INTRINSICS_MAP.md`
- **Cross-platform example:** `~/.cargo/registry/src/.../archmage-0.5.0/examples/cross_platform.rs`

## Session End State

- **Builds:** ✅ All targets build (but with unsafe violations)
- **Tests:** ❌ No tests added yet
- **Benchmarks:** ❌ Not created yet
- **Integration:** ❌ Not wired up to actual encoder/decoder
- **Performance:** ❓ Unknown - needs benchmarking

**Next session should START with fixing the unsafe code violations, THEN continue with IDCT implementation.**

---

*Handoff created: 2026-02-07*
*Branch: `perf/arm-wasm-archmage`*
*Last commit: `0f5f745 docs: add cross-platform build status`*
