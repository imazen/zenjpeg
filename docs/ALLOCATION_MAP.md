# jpegli-rs Encoder Allocation Map

Analysis of memory allocations during JPEG encoding. For a 1920x1080 RGB image.

## Pipeline Overview

```
Input RGB (6.2 MB)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  COLOR CONVERSION (convert_to_ycbcr_f32 / convert_yuv_crate)     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Y plane f32     │  │ Cb plane f32    │  │ Cr plane f32    │   │
│  │ 8.3 MB          │  │ 8.3 MB          │  │ 8.3 MB          │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│                                                    ~24.9 MB      │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  SMOOTHING (optional, if smoothing_factor > 0)                   │
│  ┌─────────────────┐  ┌─────────────────┐                        │
│  │ Cb smoothed f32 │  │ Cr smoothed f32 │   (full res copies)    │
│  │ 8.3 MB          │  │ 8.3 MB          │                        │
│  └─────────────────┘  └─────────────────┘                        │
│                                                    +16.6 MB      │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  CHROMA DOWNSAMPLING (4:2:0 example)                             │
│  ┌─────────────────┐  ┌─────────────────┐                        │
│  │ Cb down f32     │  │ Cr down f32     │   (quarter size)       │
│  │ 2.1 MB          │  │ 2.1 MB          │                        │
│  └─────────────────┘  └─────────────────┘                        │
│                                                    +4.2 MB       │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  ADAPTIVE QUANTIZATION MAP                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ AQ strength map: (width/8) × (height/8) × f32               │ │
│  │ = 240 × 135 × 4 = 130 KB                                    │ │
│  │ + internal temp buffers: ~500 KB                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                    ~0.6 MB       │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  BLOCK QUANTIZATION (quantize_all_blocks_*)                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Y blocks        │  │ Cb blocks       │  │ Cr blocks       │   │
│  │ Vec<[i16;64]>   │  │ Vec<[i16;64]>   │  │ Vec<[i16;64]>   │   │
│  │ 32,400 × 128B   │  │ 8,100 × 128B    │  │ 8,100 × 128B    │   │
│  │ = 4.1 MB        │  │ = 1.0 MB        │  │ = 1.0 MB        │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
│  Per-block: forward_dct_8x8 returns stack-allocated [f32;64]     │
│  Per-block: natural_to_zigzag copies 128 bytes                   │
│                                                    ~6.1 MB       │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  HUFFMAN OPTIMIZATION (if optimize_huffman=true)                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ FrequencyCounter: 4 × 256 bytes = 1 KB                      │ │
│  │ OptimizedHuffmanTables: 4 × ~768 bytes = 3 KB               │ │
│  │ Progressive token buffer: 1-5 MB (variable)                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                    ~1-5 MB       │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  ENTROPY ENCODING                                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ BitWriter buffer: starts at 0, grows to final size          │ │
│  │ Typical: 200-500 KB for quality 80                          │ │
│  │ HuffmanEncodeTable clones: 4 × 768 B per scan               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                    ~0.5-1 MB     │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUT BUFFER                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Vec::with_capacity(input_size / 4) = 1.55 MB initial        │ │
│  │ Grows with extend_from_slice for headers, markers, data     │ │
│  │ Final: ~200-500 KB typical                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Peak Memory by Mode

| Mode | Color Conv | Smoothing | Downsample | AQ Map | Blocks | Huffman | Total Peak |
|------|------------|-----------|------------|--------|--------|---------|------------|
| Baseline 4:4:4 | 24.9 MB | 0 | 0 | 0.6 MB | 12.2 MB | 0 | ~38 MB |
| Baseline 4:2:0 | 24.9 MB | 0 | 4.2 MB | 0.6 MB | 6.1 MB | 0 | ~36 MB |
| Baseline 4:2:0 + smooth | 24.9 MB | 16.6 MB | 4.2 MB | 0.6 MB | 6.1 MB | 0 | ~53 MB |
| Optimized 4:2:0 | 24.9 MB | 0 | 4.2 MB | 0.6 MB | 6.1 MB | 3 MB | ~39 MB |
| Progressive 4:2:0 | 24.9 MB | 0 | 4.2 MB | 0.6 MB | 6.1 MB | 5 MB | ~41 MB |

**Note**: 1920×1080 RGB = 6.2 MB input. Peak is ~6-8× input size.

## Hot Allocation Sites

### 1. Color Conversion Planes (CRITICAL - 24.9 MB)

**Location**: `encode_simd.rs:25` via `rgb_to_ycbcr_planes_simd`
```rust
let mut vec = Vec::with_capacity(len);
```

Called 3× for Y, Cb, Cr planes. Each `width × height × 4` bytes.

**Problem**: Full-resolution f32 planes even when chroma will be downsampled.

**Opportunity**:
- Fuse color conversion + downsampling for chroma (yuv crate already does this)
- Process in tiles to reduce peak memory
- Use u16 fixed-point instead of f32 (halves memory)

### 2. Smoothing Copies (16.6 MB when enabled)

**Location**: `encode_simd.rs:957`
```rust
return plane.to_vec();
```

**Problem**: Full copy of chroma planes before smoothing.

**Opportunity**: In-place smoothing or fuse with downsampling.

### 3. Quantized Block Storage (6.1 MB)

**Location**: `blocks.rs:202-204`
```rust
let mut y_blocks = Vec::with_capacity(blocks_h * blocks_v);
let mut cb_blocks = Vec::with_capacity(...);
let mut cr_blocks = Vec::with_capacity(...);
```

**Problem**: All blocks stored in memory for Huffman optimization.

**Opportunity**:
- Stream blocks directly to entropy encoder (single-pass)
- Two-pass: first pass counts frequencies only, second pass encodes
- Pre-allocate exact size, fill by index instead of `.push()`

### 4. Block Push Pattern (memcpy overhead)

**Location**: `blocks.rs:232, 256, 279`
```rust
y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
```

Each `.push()` copies 128 bytes. For 1080p: 48,600 blocks × 128 bytes = 6.2 MB of memcpy.

**Opportunity**: Pre-allocate and write by index:
```rust
let mut y_blocks = vec![[0i16; 64]; blocks_h * blocks_v];
y_blocks[idx] = natural_to_zigzag(&y_quant_coeffs);
```

### 5. Huffman Table Clones (per-scan overhead)

**Location**: `blocks.rs:620-623`
```rust
encoder.set_dc_table(0, tables.dc_luma.table.clone());
```

**Problem**: Tables cloned per scan in progressive mode.

**Opportunity**: Use `&HuffmanEncodeTable` references with lifetime management.

### 6. Progressive Token Buffer (1-5 MB)

**Location**: `huffman/optimize/tokens.rs:226`
```rust
counters: vec![FrequencyCounter::new(); num_contexts],
```

**Problem**: All tokens stored before encoding.

**Opportunity**: Stream tokens, accumulate frequencies during first pass.

## Format Conversion Allocations

| Input Format | Intermediate Allocation | Size (1080p) |
|--------------|------------------------|--------------|
| RGB | None (direct SIMD) | 0 |
| RGBA | None (SIMD handles) | 0 |
| Gray | Cb/Cr filled with 128.0 | 4.2 MB |
| BGR | None (SIMD handles) | 0 |
| BGRA | None (SIMD handles) | 0 |

## Recommended Optimizations (Priority Order)

### High Impact (save 10+ MB)

1. **Fused chroma conversion+downsampling**: Don't allocate full-res Cb/Cr when downsampling
2. **Streaming block encoding**: Eliminate block storage for baseline mode
3. **In-place smoothing**: Modify planes directly instead of copying

### Medium Impact (save 1-5 MB)

4. **Pre-allocated block arrays**: Avoid `.push()` overhead
5. **Reference-based Huffman tables**: Avoid clones
6. **u16 fixed-point planes**: Half the memory of f32

### Low Impact (cleaner code)

7. **Buffer pool for reuse between images**
8. **Tile-based processing for huge images**
9. **Streaming token encoding for progressive**

## Memory Timeline (1080p baseline 4:2:0)

```
Time →
       Input   Color    Down    AQ     Quant   Encode  Output
       6.2MB   Conv     sample  Map    Blocks
               24.9MB   4.2MB   0.6MB  6.1MB

Peak:  ████████████████████████████████████████░░░░░░░░░░░  ~36 MB
               ▲                        ▲
               │                        │
               Max allocation           Can be streamed
               (color planes)           (don't need full storage)
```

## Data Model Improvements

### Current Model
```rust
// All blocks in memory
Vec<[i16; 64]>  // Y blocks
Vec<[i16; 64]>  // Cb blocks
Vec<[i16; 64]>  // Cr blocks
```

### Proposed Model (Streaming)
```rust
// Process MCU by MCU
struct McuProcessor<'a> {
    y_plane: &'a [f32],
    cb_plane: &'a [f32],
    cr_plane: &'a [f32],
    // No block storage - encode directly
}

impl McuProcessor<'_> {
    fn encode_mcu(&mut self, mcu_x: usize, mcu_y: usize, encoder: &mut EntropyEncoder) {
        // Extract, DCT, quantize, encode in one pass
        // No intermediate storage
    }
}
```

### Proposed Model (Two-Pass Optimization)
```rust
// First pass: count only
struct FrequencyCollector {
    dc_freq: [u32; 256],
    ac_freq: [u32; 256],
}

// Second pass: encode with optimal tables
// Process blocks again, but no storage needed
```
