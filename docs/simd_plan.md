# SIMD Optimization Plan for zenjpeg

## Executive Summary

The goal is to eliminate load/store overhead by storing data in SIMD-native types throughout the encoding pipeline. Based on profiling and benchmarking:

- **Current state**: 7× speedup already achieved for quantization using SIMD
- **Bottleneck**: Load/store conversions between `[f32; 64]` and `f32x8` happen repeatedly
- **Solution**: Store data as `[f32x8; 8]` blocks and `Vec<f32x8>` planes

## Current Profile (2048×2048 image)

| Function | % of Time | Notes |
|----------|-----------|-------|
| quantize_all_blocks_subsampled | 13-15% | DCT + quantization per block |
| rgb_to_ycbcr_planes_simd | 10% | Limited by gather from interleaved RGB |
| encode_block | 10% | Entropy encoding |
| per_block_modulations_simd | 9% | AQ block modulations |
| fuzzy_erosion_simd | 8% | AQ weighted min4 of 9 |
| compute_pre_erosion_simd | 8% | AQ pre-erosion |
| downsample_2x2_simd | 7-8% | Chroma downsampling |

## Key Insight

Every SIMD operation currently has this pattern:
```rust
// Load from scalar array → SIMD
let v = f32x8::from([arr[k], arr[k+1], ..., arr[k+7]]);
// Do math
let result = v * scale;
// Store back to scalar array
let arr: [f32; 8] = result.into();
output[k..k+8].copy_from_slice(&arr);
```

If we store data as `f32x8` directly, the load/store becomes trivial:
```rust
// No load overhead - it's already f32x8!
let result = block.rows[i] * quant.mul_rows[i];
// Direct assignment, no conversion
output.rows[i] = result;
```

## New Types (Already Implemented)

```rust
// zenjpeg/src/simd_types.rs

/// 8x8 f32 block as 8 rows of f32x8 (32-byte aligned)
pub struct Block8x8f {
    pub rows: [f32x8; 8],
}

/// 8x8 i16 block for quantized coefficients
pub struct Block8x8i16 {
    pub rows: [i16x8; 8],
}

/// Quantization table with pre-computed reciprocals
pub struct QuantTableSimd {
    pub mul_rows: [f32x8; 8],  // 1.0 / quant_value
    pub values: [u16; 64],      // Original for JPEG header
}
```

## Implementation Phases

### Phase 1: Block Operations (High Impact)

**Goal**: Use `Block8x8f` in DCT and quantization.

**Files to modify**:
- `zenjpeg/src/dct.rs` - Return `Block8x8f` from `forward_dct_8x8`
- `zenjpeg/src/quant/mod.rs` - Use `QuantTableSimd` for quantization
- `zenjpeg/src/encode/blocks.rs` - Update block processing

**Key changes**:

1. **DCT output as Block8x8f**:
```rust
// Before
pub fn forward_dct_8x8(input: &[f32; 64]) -> [f32; 64]

// After
pub fn forward_dct_8x8_simd(input: &Block8x8f) -> Block8x8f
```

2. **Quantization with QuantTableSimd**:
```rust
// Before (load 8 values, multiply, store 8 values - per row)
fn quantize_block(dct: &[f32; 64], quant: &[f32; 64]) -> [i16; 64]

// After (direct SIMD multiply per row, no load overhead)
fn quantize_block_simd(dct: &Block8x8f, quant: &QuantTableSimd) -> Block8x8i16 {
    let mut result = Block8x8i16::ZERO;
    for i in 0..8 {
        let quantized = dct.rows[i] * quant.mul_rows[i];
        result.rows[i] = quantized.round_int().to_i16();
    }
    result
}
```

**Expected impact**:
- Eliminate ~50% of load/store in quantize_all_blocks (13-15% → ~7%)
- Total encoding speedup: ~5-8%

### Phase 2: Block Storage

**Goal**: Store quantized blocks as `Block8x8i16` instead of `[i16; 64]`.

**Files to modify**:
- `zenjpeg/src/encode/blocks.rs` - Change block vector types
- `zenjpeg/src/entropy/encoder.rs` - Read from `Block8x8i16`

**Key changes**:

```rust
// Before
let mut y_blocks: Vec<[i16; 64]> = vec![[0i16; 64]; num_blocks];

// After
let mut y_blocks: Vec<Block8x8i16> = vec![Block8x8i16::ZERO; num_blocks];
```

**Consideration**: Zigzag reordering
- Entropy encoding expects coefficients in zigzag order
- Options:
  a. Convert to zigzag at encoding time (current approach)
  b. Store in zigzag order using a different SIMD layout
  c. Use SIMD shuffle for zigzag (complex but fast)

**Expected impact**: Minor - mostly cleaner code and cache benefits.

### Phase 3: Plane Storage (Medium Impact)

**Goal**: Store color planes as SIMD chunks.

**New type**:
```rust
/// Plane stored as f32x8 chunks for SIMD-native access
pub struct SimdPlane {
    pub data: Vec<f32x8>,
    pub width: usize,       // in pixels
    pub height: usize,
    pub stride: usize,      // in f32x8 chunks (width / 8, rounded up)
}

impl SimdPlane {
    /// Extract 8x8 block as Block8x8f (one load per row)
    pub fn extract_block(&self, block_x: usize, block_y: usize) -> Block8x8f {
        let mut block = Block8x8f::ZERO;
        let base_y = block_y * 8;
        for row in 0..8 {
            // Direct load - no gather needed!
            block.rows[row] = self.data[(base_y + row) * self.stride + block_x];
        }
        block
    }
}
```

**Files to modify**:
- `zenjpeg/src/encode_simd.rs` - Color conversion outputs SimdPlane
- `zenjpeg/src/encode/baseline.rs` - Use SimdPlane
- `zenjpeg/src/encode/blocks.rs` - Extract blocks from SimdPlane

**Key benefit**: Block extraction becomes 8 direct loads instead of 64 scattered loads.

**Challenge**: Width must be padded to multiple of 8.

**Expected impact**:
- Faster block extraction
- Faster downsampling (operates on chunks)
- ~5% total speedup

### Phase 4: Color Conversion (Lower Priority)

**Goal**: Optimize RGB to YCbCr conversion.

**Current problem**: RGB data is interleaved (RGBRGBRGB...), requiring scatter/gather.

**Options**:
1. **Accept gather cost** - RGB input is external, can't change layout
2. **Process more pixels** - Load 24 bytes (8 RGB pixels), deinterleave with SIMD shuffles
3. **Use yuv crate** - Already explored, uses platform-specific SIMD

**Decision**: Lower priority since input format is external constraint.

## Implementation Order

1. ✅ Create `Block8x8f`, `Block8x8i16`, `QuantTableSimd` types
2. ✅ Add benchmark comparing scalar vs SIMD quantization
3. 🔲 Modify `forward_dct_8x8` to work with `Block8x8f`
4. 🔲 Modify quantization to use `QuantTableSimd`
5. 🔲 Update `quantize_all_blocks_*` functions
6. 🔲 Create `SimdPlane` type
7. 🔲 Update color conversion to output `SimdPlane`
8. 🔲 Update block extraction to use `SimdPlane`
9. 🔲 Benchmark and profile after each change

## Compatibility Notes

### Boundaries

Conversion functions needed at:
- **Input**: RGB bytes → SimdPlane (during color conversion)
- **Output**: Block8x8i16 → zigzag [i16; 64] (for entropy encoding)

### Zigzag Ordering

JPEG entropy encoding requires zigzag order. Options:
1. **Current**: Convert at encoding time with `natural_to_zigzag`
2. **Future**: SIMD-accelerated zigzag using shuffle operations
3. **Alternative**: Store in zigzag order (complicates other operations)

Recommendation: Keep current approach initially, optimize later if profiling shows it's significant.

### Alignment

- `Block8x8f` is 32-byte aligned (optimal for AVX)
- `Block8x8i16` is 16-byte aligned (optimal for SSE)
- `Vec<f32x8>` alignment depends on allocator (usually fine)

## Testing Strategy

1. **Unit tests**: Each new type has roundtrip tests
2. **Parity tests**: Compare SIMD vs scalar results
3. **Integration tests**: Full encode with new types matches old output
4. **Benchmarks**: Criterion benchmarks for each phase

## Benchmark Targets

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| 1024×1024 baseline 4:2:0 | 14.7 ms | 12 ms | Phases 1-3 |
| 1024×1024 progressive 4:2:0 | 20.4 ms | 17 ms | Phases 1-3 |
| quantize_block | 15 ns | 10 ns | Phase 1 |
| block extraction | TBD | -50% | Phase 3 |

## References

- `docs/SIMD_DATA_LAYOUT.md` - Detailed type designs
- `docs/ALLOCATION_MAP.md` - Memory allocation analysis
- `~/work/helpful-info/state-of-simd-rust-2025.md` - SIMD crate comparison
- `~/work/helpful-info/towards-fearless-simd-2025.md` - Best practices

## Wide Crate Types Used

| Type | Size | Purpose |
|------|------|---------|
| `f32x8` | 32 bytes | DCT coefficients, color planes |
| `i16x8` | 16 bytes | Quantized coefficients |
| `i32x8` | 32 bytes | Intermediate calculations |
| `u8x16` | 16 bytes | Potential for pixel data |

## Conclusion

The key optimization is **storing data in SIMD types** to eliminate load/store overhead. The infrastructure is now in place with `Block8x8f` and `QuantTableSimd`. Next steps are to integrate these types into the DCT and quantization pipeline (Phase 1), which should provide 5-8% speedup with relatively low risk.
