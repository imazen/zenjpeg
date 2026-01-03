# Chroma Conversion Benchmarking Plan

## Goal

Determine the optimal chroma conversion and downsampling strategy through rigorous benchmarking, then hide implementation details from users.

## Current API Surface (TOO COMPLEX)

```rust
// Exposed to users - shouldn't be
pub enum ChromaConversion {
    Intrinsic,  // f32 color conversion + box filter
    Fast,       // yuv crate integer + box filter
    Sharp,      // yuv crate integer + gamma-aware bilinear
    Auto,       // picks Sharp or Intrinsic based on subsampling
}

// Also exposed
pub fn smoothing_factor(factor: u8)  // 0-100, only works with Intrinsic
```

**Problems:**
1. Users must understand color conversion internals
2. `smoothing_factor` only works with one path (confusing)
3. Conflates 3 concerns: source, precision, downsampling method
4. No clear "best quality" vs "best speed" guidance

## Target API Surface (SIMPLE)

```rust
// Option A: Just subsampling, we pick the best method internally
pub fn subsampling(Subsampling)  // S444, S422, S420, S440

// Option B: Quality vs speed tradeoff (if benchmarks show meaningful difference)
pub enum ChromaQuality {
    Best,       // f32 + gamma-aware downsampling (if we implement it)
    Fast,       // yuv crate SIMD path
}
```

**Ideal:** Users just set subsampling, we automatically use the best method. No `ChromaConversion` enum exposed at all.

## Internal Test Infrastructure

### 1. Internal Enum for Benchmarking

```rust
// In src/encode.rs or new src/chroma_benchmark.rs (NOT pub)
#[derive(Debug, Clone, Copy)]
enum ChromaMethod {
    // Color conversion variants
    IntrinsicF32,           // Our f32 BT.601 conversion
    YuvCrateBalanced,       // yuv crate with Balanced precision
    YuvCrateProfessional,   // yuv crate with Professional precision (if available)

    // Downsampling variants (for 4:2:0, 4:2:2, 4:4:0)
    BoxFilter,              // Simple 2x2/2x1/1x2 averaging
    BoxFilterSmoothed(u8),  // Pre-blur + box filter (smoothing_factor)
    SharpYuv,               // yuv crate gamma-aware bilinear
    GammaAwareF32,          // TODO: Our own f32 gamma-aware implementation
}

#[derive(Debug, Clone, Copy)]
struct ChromaPipeline {
    color_conversion: ColorConversionMethod,
    downsampling: DownsamplingMethod,
}

enum ColorConversionMethod {
    IntrinsicF32,
    YuvCrateBalanced,
    YuvCrateProfessional,
}

enum DownsamplingMethod {
    None,                   // 4:4:4
    BoxFilter,
    BoxFilterSmoothed(u8),
    SharpYuv,
    GammaAwareF32,          // TODO
}
```

### 2. Benchmark Test Matrix

| Pipeline | Color Conv | Downsampling | Subsampling | Test | Status |
|----------|------------|--------------|-------------|------|--------|
| P1 | f32 | None | 4:4:4 | baseline | ✅ |
| P2 | f32 | Box | 4:2:0 | current Intrinsic | ✅ |
| P3 | f32 | Box+Smooth | 4:2:0 | current Intrinsic+smoothing | ✅ |
| P4 | yuv Balanced | Box | 4:2:0 | current Fast | ✅ |
| P5 | yuv Balanced | Sharp | 4:2:0 | current Sharp | ✅ |
| P6 | f32 | GammaAware | 4:2:0 | f32 gamma-aware single-pass | ✅ IMPLEMENTED |
| P7 | f32 | GammaAwareIterative | 4:2:0 | f32 gamma-aware iterative (Sharp YUV style) | ✅ IMPLEMENTED |
| P8 | yuv Professional | Sharp | 4:2:0 | max yuv precision | pending |

### 3. Metrics to Measure

**Quality (signal loss):**
- DSSIM vs original (primary metric)
- SSIMULACRA2 vs original
- Butteraugli distance
- Per-channel error distribution (Y, Cb, Cr separately)

**Performance:**
- Encode time (µs/megapixel)
- Memory allocation

**File size:**
- Bytes at equivalent quality settings

### 4. Test Images

Use diverse corpus to catch edge cases:
- **Synthetic**: Gradients, sharp edges, text, UI elements
- **Photographic**: Natural scenes, portraits, landscapes
- **Problematic**: Red text on green, thin colored lines, color fringes

From existing corpora:
- `codec-corpus/kodak/` - Classic test set
- `CID22-512/` - Diverse content
- Generate synthetic test patterns

## Implementation Plan

### Phase 1: Internal Benchmarking Infrastructure

1. Create `src/chroma_bench.rs` (internal, `#[cfg(test)]`)
2. Define `ChromaPipeline` enum with all variants
3. Create `encode_with_pipeline(data, config, pipeline)` internal function
4. Write benchmark harness measuring quality + speed

### Phase 2: Implement Missing Variants

1. **GammaAwareF32 downsampling**: ✅ IMPLEMENTED
   - Converts RGB to linear space (sRGB transfer function)
   - Averages RGB values in linear space for each 2x2/2x1/1x2 block
   - Converts back to sRGB then to YCbCr
   - All in f32 precision
   - Supports 4:2:0, 4:2:2, and 4:4:0 subsampling
   - Use via: `set_internal_pathway(P_F32_GAMMA_AWARE)`

2. **GammaAwareIterative downsampling**: ✅ IMPLEMENTED
   - Similar to Sharp YUV algorithm from libwebp
   - Starts with gamma-aware estimate as initial guess
   - Iteratively refines Cb/Cr by minimizing reconstruction error
   - Handles out-of-gamut clipping by adjusting chroma values
   - 4 iterations (matches libwebp)
   - Supports 4:2:0, 4:2:2, and 4:4:0 subsampling
   - Use via: `set_internal_pathway(P_F32_GAMMA_AWARE_ITERATIVE)`

3. **YuvCrateProfessional**: Add `professional_mode` feature flag (pending)

### Phase 3: Run Comprehensive Benchmarks

Test matrix:
- 5+ pipelines × 3 subsampling modes × 20+ images × 5 quality levels
- Generate comparison report with charts

Questions to answer:
1. Does f32 color conversion produce measurably better results than yuv Balanced?
2. Does GammaAwareF32 downsampling match or beat SharpYuv quality?
3. What's the speed cost of each approach?
4. Is smoothing_factor ever better than Sharp?

### Phase 4: Simplify API Based on Results

Likely outcomes:
- **If GammaAwareF32 ≈ SharpYuv quality**: Use f32 path exclusively, remove yuv crate dependency for encoding
- **If SharpYuv > GammaAwareF32**: Keep yuv crate for downsampling only, use f32 for 4:4:4
- **If speed difference is negligible**: Always use highest quality path

### Phase 5: Clean Up Public API

1. Remove `ChromaConversion` enum from public API
2. Remove `smoothing_factor()` if not useful
3. Keep only `subsampling()` in public API
4. Internal implementation picks optimal pipeline automatically

## File Structure

```
jpegli-rs/src/
├── encode.rs              # Public Encoder, uses optimal pipeline internally
├── chroma/
│   ├── mod.rs             # Internal chroma module
│   ├── convert_f32.rs     # f32 RGB↔YCbCr conversion
│   ├── downsample.rs      # Box filter, gamma-aware downsample
│   └── pipeline.rs        # ChromaPipeline enum, dispatch logic
└── tests/
    └── chroma_quality.rs  # Quality comparison tests

jpegli-rs/benches/
└── chroma_benchmark.rs    # Criterion benchmarks for all pipelines
```

## Success Criteria

1. **Quality**: Default path produces DSSIM within 0.0001 of theoretical best
2. **Speed**: No more than 10% slower than fastest viable path
3. **Simplicity**: Users only choose subsampling mode, nothing else
4. **Maintainability**: Single optimal code path, not N variants to maintain

## Questions to Resolve

1. Should we keep yuv crate at all, or go pure f32?
2. Is the `professional_mode` feature worth the dependency complexity?
3. Do we need 4:4:0 support? (yuv crate doesn't have it)
4. Should XYB mode also get gamma-aware B channel downsampling?

## Timeline

1. Phase 1 (infrastructure): Create benchmark harness
2. Phase 2 (implement): Add GammaAwareF32 downsampling
3. Phase 3 (measure): Run full benchmark suite
4. Phase 4-5 (simplify): Refactor based on results
