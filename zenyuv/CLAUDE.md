# zenyuv

Safe, `#![forbid(unsafe_code)]` SIMD-optimized YUV↔RGB color matrix conversion. Replaces the `yuv` crate across the zen ecosystem with zero unsafe code via archmage token-based SIMD dispatch.

## Vision

Drop-in replacement for the `yuv` crate in zenjpeg, zenwebp, zenavif, and zenrav1e. Must be faster than `yuv` on x86-64 AVX2 (already achieved for BT.601 full-range 4:4:4 and 4:2:0 encode). Must cover all axes that consumers need:

| Axis | Variants needed |
|------|----------------|
| Direction | RGB→YCbCr (encode), YCbCr→RGB (decode) |
| Matrix | BT.601, BT.709, BT.2020 |
| Range | Full (JFIF), Limited (studio/VP8) |
| Subsampling | 4:4:4, 4:2:0, 4:2:2 |
| Bit depth | 8, 10, 12, 16 |
| Channel layout | RGB, RGBA, BGR, BGRA |
| Chroma filter | Nearest, bilinear (decode only) |
| Sharp YUV | Iterative perceptual chroma optimization (encode only) |

The `yuv` crate stamps these out via const generics over one internal kernel. We do the same but with `#[arcane]`/`#[rite]` SIMD entry points instead of raw `unsafe`.

## Current Status

- BT.601 full-range RGB→YCbCr 4:4:4 and 4:2:0, 8-bit, RGB layout
- AVX2 `#[arcane]` kernel: matches or beats `yuv` crate Professional mode
- magetypes generic fallback: NEON, WASM SIMD128, scalar
- `#![no_std]`, `#![forbid(unsafe_code)]`, MIT/Apache-2.0

## Dependencies

Required: `archmage`, `magetypes`, `safe_unaligned_simd`. No alloc needed for core conversion (caller provides all buffers). `alloc` feature for convenience wrappers that allocate output planes.

## Architecture

### SIMD tiers

| Tier | Token | Width | Entry |
|------|-------|-------|-------|
| AVX2+FMA | `X64V3Token` | 32 pixels/iter | `#[arcane]` on outer loop |
| AVX-512 | `X64V4Token` | 64 pixels/iter | `#[arcane]` (future) |
| NEON | `NeonToken` | 16 pixels/iter | `#[magetypes(neon)]` generic |
| WASM128 | `Wasm128Token` | 16 pixels/iter | `#[magetypes(wasm128)]` generic |
| Scalar | `ScalarToken` | 8 pixels/iter | `#[magetypes(scalar)]` generic |

### Kernel structure

- **4:4:4**: Load 96 bytes RGB → pshufb deinterleave → widen u8→i16 → interleave RG/B pairs → pmaddwd with 15-bit coefficients → packus i32→u16→u8 → store 32 Y + 32 Cb + 32 Cr.
- **4:2:0**: Same Y as 4:4:4. Chroma: avg_epu8 vertical (top/bottom rows) → maddubs horizontal pair sum → interleave → pmaddwd at PREC+1=16 → packus → store 16 Cb + 16 Cr per 2×32 pixel block.
- Helpers tagged `#[rite]` so they inline into the `#[arcane]` caller's target_feature region — no cross-function YMM spills.

### Sharp YUV

Lives in zenjpeg's `encode/chroma.rs` as `GammaAwareIterative`. Needs to be:
1. Extracted here as a feature-gated module
2. Made generic over range (full/limited) and matrix (BT.601/709/2020)
3. SIMD-optimized — the iterative loop's inner reconstruction+error computation is vectorizable per 2×2 block batch

## Testing Requirements

### Brute-force parity tests

Use `archmage::testing::for_each_token_permutation` to run every conversion with every SIMD tier disabled/enabled. Verify byte-identical output across all tier combinations. Pattern from zenjpeg's `simd_types.rs`:

```rust
#[cfg(target_arch = "x86_64")]
#[test]
fn test_yuv444_dispatch_parity() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
    let rgb = make_test_pattern(256, 256);
    let reference = convert_444(&rgb, 256, 256);
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = convert_444(&rgb, 256, 256);
        assert_eq!(result, reference, "mismatch at permutation: {perm}");
    });
    assert!(report.permutations_run >= 2);
}
```

### Exhaustive input coverage

For single-pixel precision: iterate all 256³ RGB inputs (16M), convert RGB→YCbCr→RGB roundtrip, verify max error ≤ 1 level. Run as `#[ignore]` test (takes ~30s).

### Cross-platform CI

7 targets minimum:
- `x86_64-unknown-linux-gnu` (AVX2+FMA primary)
- `x86_64-pc-windows-msvc`
- `x86_64-apple-darwin` (macos-26-intel)
- `aarch64-apple-darwin` (macos-latest, NEON)
- `aarch64-unknown-linux-gnu` (cross, NEON)
- `wasm32-unknown-unknown` (wasm-pack test, WASM SIMD128)
- `i686-unknown-linux-gnu` (cross, 32-bit correctness)

WASM tests MUST actually run (not just compile). Use `wasm-pack test --node` or similar.

### Local NEON testing

Use QEMU user-mode emulation:
```bash
# Install
sudo apt install qemu-user-static gcc-aarch64-linux-gnu

# Run via cross
cross test --target aarch64-unknown-linux-gnu --release
```

## Benchmarking

Use zenbench (interleaved paired execution). Never criterion. Never `-C target-cpu=native`.

Benchmark against `yuv` crate at every size from 256 to 4096. Must be faster or equal at all sizes. Current results (7950X WSL2):

### 4:4:4 RGB→YCbCr
| Size | zenyuv | yuv crate | Delta |
|------|--------|-----------|-------|
| 256 | 14.7µs | 16.5µs | **-11%** |
| 512 | 59.3µs | 62.8µs | **-6%** |
| 1024 | 239.8µs | 244.7µs | -2% |
| 2048 | 4.94ms | 5.00ms | -1% |
| 4096 | 20.44ms | 20.74ms | -1.5% |

### 4:2:0 RGB→YCbCr
| Size | zenyuv | yuv crate | Delta |
|------|--------|-----------|-------|
| 256 | 11.0µs | 11.4µs | **-3%** |
| 512 | 45.9µs | 45.9µs | ±0% |
| 1024 | 184.6µs | 175.7µs | +5% |
| 2048 | 875.6µs | 870.2µs | ±0% |
| 4096 | 4.66ms | 4.84ms | **-4%** |

## Roadmap

### Phase 1: Replace yuv in zenjpeg (current)
- [x] BT.601 full-range 4:4:4 encode (AVX2 + generic)
- [x] BT.601 full-range 4:2:0 encode (AVX2 + generic)
- [x] Extract to standalone crate
- [x] Wire into zenjpeg encoder (zenjpeg's `fast_yuv.rs` wraps zenyuv; the
      scalar magetypes fallback was deleted when zenjpeg dropped the `yuv = []`
      feature flag)
- [x] Add zenbench bench to crate itself (`benches/rgb_to_yuv_bench.rs` —
      zenyuv vs yuv-crate Professional at 256/512/1024/2048/4096)
- [x] Precision comparison example (`examples/precision_vs_yuv_crate.rs` —
      demonstrates ±0 vs yuv-Pro on u8 output, ≤1 vs f32 reference)
- [ ] CI on 7 platforms
- [ ] Brute-force token permutation tests
- [ ] Exhaustive 256³ roundtrip test

### Phase 2: Replace yuv in zenwebp
- [ ] BT.601 limited-range 4:2:0 encode (VP8 needs studio range)
- [ ] RGBA/BGR/BGRA input layouts
- [ ] Extract Sharp YUV from zenjpeg chroma.rs, make fast

### Phase 3: Replace yuv in zenavif
- [ ] YCbCr→RGB decode (8-bit, all subsampling modes)
- [ ] Bilinear chroma upsampling
- [ ] 10/12/16-bit decode
- [ ] BT.709, BT.2020 matrices
- [ ] RGBA output

### Phase 4: Replace yuv in zenrav1e
- [ ] Survey zenrav1e's yuv usage
- [ ] Add whatever's missing

### Phase 5: Publish
- [ ] Move to own repo (or garb workspace)
- [ ] README with benchmarks, badges
- [ ] crates.io publish
- [ ] Upstream zen crates switch from path dep to version dep

## Cross-Platform Golden Results

Need a way to verify all platforms produce identical output without storing large reference files in git.

**Approach ideas (needs investigation):**
- Hash full output planes per (matrix, range, subsampling, depth) for a fixed set of test patterns. Store only the hashes (~40 bytes each). If a platform diverges, the hash catches it.
- For rounding boundary cases (where f32 vs i16 disagree by ±1): identify the specific input values that hit boundaries, store those as a small "boundary test corpus" (~1KB). Hash the rest.
- Could generate deterministic test patterns procedurally (same seed across platforms) so no stored inputs needed — just stored output hashes.
- The 256³ exhaustive test already covers all u8 inputs for 4:4:4. Hash the full 16.7M output (Y+Cb+Cr = 50MB → one SHA-256 per plane per tier). Store 6 hashes.
- For multi-tier parity: if AVX2 and scalar differ by ±1 at some inputs, we need separate golden hashes per tier OR define the scalar path as canonical and allow ±1 from it on all platforms.

**Problem:** different SIMD tiers (AVX2 fixed-point vs f32 FMA) produce ±1 differences at rounding boundaries. A single golden hash per platform won't work unless we pick ONE canonical implementation and force all tiers to match it exactly (e.g., always use the f32 path as reference, even if AVX2 is faster).

**Pragmatic approach:** define the scalar f32 path as the reference. Store SHA-256 hashes of its output for a set of test patterns. On CI: run scalar, verify hash matches. Then run SIMD tiers, verify ±1 vs scalar. This gives cross-platform determinism (f32 math is IEEE 754, should be identical) plus tier-correctness (±1 tolerance).

**TODO:** implement this after the module restructure stabilizes.

## Known Gaps vs yuv crate

1. **Decode direction** — yuv has full YCbCr→RGB; we have nothing yet. zenavif calls ~37 decode variants.
2. **Limited range** — zenwebp VP8 needs studio range (Y 16-235, Cb/Cr 16-240).
3. **High bit depth** — zenavif needs 10/12/16-bit. The kernel is the same math at wider types; pmaddwd still works for 10-bit (values ≤ 1023 fit i16).
4. **Channel layouts** — RGBA/BGR/BGRA need deinterleave variants. The yuv crate handles this with const-generic channel index remapping.
5. **Bilinear chroma upsampling** — decode-side fancy upsampling. Different SIMD pattern (vertical interpolation between chroma rows).
6. **Sharp YUV** — iterative perceptual optimization. Already implemented in zenjpeg's chroma.rs but needs extraction and SIMD optimization.
