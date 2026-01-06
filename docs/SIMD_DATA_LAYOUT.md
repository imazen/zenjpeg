# SIMD-Native Data Layout Proposal

## Current Problem

Every SIMD operation has this pattern:
```rust
// Load 8 elements from scalar array
let v = f32x8::from([arr[i], arr[i+1], arr[i+2], ...]);
// Do SIMD math
let result = v * scale;
// Store back to scalar array
let result_arr: [f32; 8] = result.into();
arr[i..i+8].copy_from_slice(&result_arr);
```

This load/store dance dominates runtime when the actual math is simple.

## Solution: Store Data as SIMD Types

### 1. DCT Blocks

**Current:**
```rust
type Block = [f32; 64];  // or [i16; 64]
```

**Proposed:**
```rust
use wide::f32x8;

/// An 8x8 DCT block stored as 8 rows of f32x8
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct Block8x8 {
    pub rows: [f32x8; 8],
}

impl Block8x8 {
    pub const ZERO: Self = Self { rows: [f32x8::ZERO; 8] };

    /// Access a single coefficient (for compatibility)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.rows[row].as_array_ref()[col]
    }

    /// Set a single coefficient
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.rows[row].as_array_mut()[col] = value;
    }
}
```

**Benefits:**
- DCT row transform: load is FREE (it's already an f32x8)
- Quantization: `block.rows[i] * quant.rows[i]` - one instruction per row
- Transpose: SIMD shuffle operations work naturally

### 2. Quantization Tables

**Current:**
```rust
pub struct QuantTable {
    pub values: [u16; 64],
}
```

**Proposed:**
```rust
/// Quantization table in SIMD-friendly layout
#[derive(Clone)]
#[repr(C, align(32))]
pub struct QuantTableSimd {
    /// Multipliers for quantization (1.0 / quant_value)
    pub mul_rows: [f32x8; 8],
    /// Original values for encoding header
    pub values: [u16; 64],
}

impl QuantTableSimd {
    pub fn from_values(values: [u16; 64]) -> Self {
        let mut mul_rows = [f32x8::ZERO; 8];
        for row in 0..8 {
            let mut muls = [0.0f32; 8];
            for col in 0..8 {
                muls[col] = 1.0 / values[row * 8 + col] as f32;
            }
            mul_rows[row] = f32x8::from(muls);
        }
        Self { mul_rows, values }
    }
}
```

**Quantization becomes trivial:**
```rust
fn quantize_block(block: &Block8x8, quant: &QuantTableSimd) -> Block8x8i16 {
    let mut result = Block8x8i16::ZERO;
    for row in 0..8 {
        // One SIMD multiply per row, no load overhead!
        let quantized = block.rows[row] * quant.mul_rows[row];
        result.rows[row] = quantized.round_int();  // convert to i16x8
    }
    result
}
```

### 3. Integer Block (for quantized coefficients)

```rust
use wide::i16x8;

/// Quantized coefficients stored as SIMD vectors
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Block8x8i16 {
    pub rows: [i16x8; 8],
}
```

### 4. Plane Storage (Advanced)

For very high throughput, store planes as SIMD chunks:

```rust
/// A plane stored as SIMD-width chunks
/// For width=1920, stores 240 f32x8 per row
pub struct SimdPlane {
    /// Data stored as f32x8 chunks, row-major
    pub data: Vec<f32x8>,
    pub width: usize,      // in pixels
    pub height: usize,
    pub stride: usize,     // in f32x8 chunks per row (width / 8, rounded up)
}

impl SimdPlane {
    /// Get a chunk of 8 pixels
    #[inline]
    pub fn get_chunk(&self, y: usize, chunk_x: usize) -> f32x8 {
        self.data[y * self.stride + chunk_x]
    }

    /// Get 8x8 block as Block8x8 (for DCT)
    pub fn extract_block(&self, block_x: usize, block_y: usize) -> Block8x8 {
        let mut block = Block8x8::ZERO;
        let start_y = block_y * 8;
        let chunk_x = block_x; // each block is one chunk wide

        for row in 0..8 {
            block.rows[row] = self.get_chunk(start_y + row, chunk_x);
        }
        block
    }
}
```

## Implementation Phases

### Phase 1: Block8x8 and QuantTableSimd
- Replace `[f32; 64]` with `Block8x8` in DCT and quantization
- Keep scalar input/output conversion at boundaries
- Expected: Significant speedup in quantize_all_blocks (13-15% of time)

### Phase 2: Block8x8i16 for quantized storage
- Replace `[i16; 64]` storage with `Block8x8i16`
- Update entropy encoder to read from SIMD layout
- Expected: Faster block storage and retrieval

### Phase 3: SimdPlane for color planes
- Store Y, Cb, Cr planes in SIMD-native format
- Downsampling operates on chunks directly
- Expected: Major speedup in downsample (7-8% of time)

## Compatibility Notes

- Conversion functions at boundaries (input RGB → SimdPlane, Block8x8 → zigzag)
- Zigzag reordering will need special handling (different access pattern)
- Entropy encoding expects zigzag order, not row order

## Wide Crate Types Used

| Type | Size | Purpose |
|------|------|---------|
| `f32x8` | 32 bytes | DCT coefficients, quantization |
| `i16x8` | 16 bytes | Quantized coefficients |
| `i32x8` | 32 bytes | Intermediate calculations |
| `u8x16` | 16 bytes | Pixel data (potential) |

## Expected Performance Impact

Based on current profile (2048×2048 image):
- quantize_all_blocks: 13-15% → potentially 5-7% (eliminate load/store)
- downsample_2x2: 7-8% → potentially 3-4% (native chunk access)
- DCT: faster transpose and row operations

Total expected improvement: 10-20% additional speedup.
