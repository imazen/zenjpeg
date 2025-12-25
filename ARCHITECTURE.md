# zenjpeg Architecture

## Overview

zenjpeg combines techniques from two state-of-the-art JPEG encoders:

1. **mozjpeg** - Mozilla's optimized JPEG encoder
   - Trellis quantization for rate-distortion optimization
   - Progressive encoding for better compression
   - Huffman table optimization
   - Excels at **low to medium quality** (Q < 70)

2. **jpegli** - Google's improved JPEG encoder from JPEG XL
   - Adaptive quantization for content-aware compression
   - XYB perceptual color space (optional)
   - Per-block quantization scaling
   - Excels at **high quality** (Q >= 70)

## Encoding Pipeline

```
Input RGB → Color Convert → MCU Block → DCT → Quantize → Entropy → JPEG
                                           ↑
                              Trellis or AQ multiplier
```

### Stage 1: Color Conversion (color.rs)

RGB to YCbCr using JFIF/BT.601 coefficients:
```
Y  =  0.299 * R + 0.587 * G + 0.114 * B
Cb = -0.169 * R - 0.331 * G + 0.500 * B + 128
Cr =  0.500 * R - 0.419 * G - 0.081 * B + 128
```

### Stage 2: Block Extraction

Image divided into 8x8 blocks (MCUs). Edge blocks are padded by replication.

### Stage 3: DCT (dct.rs)

Forward DCT with 1/8 scaling for JPEG compatibility:
- Level shift: subtract 128 from each pixel
- Apply 8x8 DCT
- Output scaled for quantization

Current: Reference implementation (direct formula)
TODO: Fast DCT (Loeffler algorithm), SIMD

### Stage 4: Quantization

Base quantization tables scaled by quality factor:
- Q < 50: scale = 5000 / Q
- Q >= 50: scale = 200 - 2*Q

#### mozjpeg Approach: Trellis (trellis.rs)

Rate-distortion optimization via dynamic programming:
```
cost = rate + λ × distortion
```
Where:
- rate = Huffman bits to encode coefficient
- distortion = (original - reconstructed)²
- λ = lambda base adjusted by quality

For each coefficient, consider {q-1, q, q+1} and pick minimum cost.

#### jpegli Approach: Adaptive Quantization (adaptive_quant.rs)

Per-block quantization scaling based on local variance:
- Low variance (smooth) → higher multiplier → stronger quantization
- High variance (detail) → lower multiplier → weaker quantization

```rust
// Simplified: full jpegli uses pre-erosion, fuzzy erosion, etc.
multiplier = 1.0 + strength * variance_factor
```

### Stage 5: Entropy Encoding (entropy.rs)

Huffman coding of quantized coefficients:
- DC coefficients: differential encoding
- AC coefficients: run-length + category encoding
- Byte stuffing: 0xFF → 0xFF 0x00

Current: Fixed standard tables
TODO: Optimized tables based on actual statistics

## Strategy Selection (strategy.rs)

```
Quality       Strategy        Features
---------     ----------      -----------------------------------
Q < 50        Mozjpeg         Trellis AC+DC, Progressive, Huffman opt
Q 50-70       Hybrid          Trellis AC, AQ at reduced strength
Q >= 70       Jpegli          Adaptive quant, Sequential, Huffman opt
```

### SelectedStrategy Structure
```rust
pub struct SelectedStrategy {
    approach: EncodingApproach,  // Mozjpeg/Jpegli/Hybrid
    trellis: TrellisConfig,
    adaptive_quant: AdaptiveQuantConfig,
    progressive: bool,
    optimize_huffman: bool,
}
```

## File Format

Standard JFIF JPEG structure:
```
SOI (FFD8)
APP0 (JFIF header)
DQT (quantization tables)
SOF0 (frame header)
DHT (Huffman tables)
SOS (scan header)
Entropy data (with byte stuffing)
EOI (FFD9)
```

## Quality Targets

zenjpeg aims to achieve better Pareto front than either encoder alone:

```
Quality     mozjpeg strength    jpegli strength
---------   -----------------   ---------------
Q30         ★★★                 ★☆☆
Q50         ★★★                 ★★☆
Q70         ★★☆                 ★★★
Q90         ★☆☆                 ★★★
```

By selecting the right strategy per quality level, zenjpeg achieves the envelope of both curves.

## Module Dependencies

```
lib.rs
├── error.rs      (standalone)
├── types.rs      (imports trellis.rs, adaptive_quant.rs)
├── consts.rs     (standalone)
├── color.rs      (standalone)
├── dct.rs        (imports consts.rs)
├── quant.rs      (imports consts.rs)
├── huffman.rs    (imports consts.rs)
├── entropy.rs    (imports huffman.rs)
├── trellis.rs    (imports huffman.rs)
├── adaptive_quant.rs (standalone)
├── strategy.rs   (imports types.rs)
└── encode.rs     (imports all above)
```

## Performance Considerations

### Current Bottlenecks
1. DCT: Using slow reference implementation
2. Color conversion: Scalar, not vectorized
3. Entropy coding: No table optimization

### Optimization Plan
1. Fast DCT (Loeffler) - 3-4x speedup
2. SIMD color conversion - 4-8x speedup
3. SIMD DCT - 4-8x speedup
4. Two-pass Huffman - better compression

## Testing Strategy

### Unit Tests
Each module has inline tests for core functionality.

### Integration Tests
TODO: Roundtrip tests with jpeg-decoder

### Parity Tests
TODO: Compare output with mozjpeg-rs and jpegli-rs

### Quality Tests
TODO: Use codec-eval to verify Pareto improvements

## Future Work

1. **XYB Color Space** - Optional jpegli mode for even better high-quality
2. **Progressive Encoding** - Scan optimization
3. **Restart Markers** - Error resilience
4. **ICC Profiles** - Color management
5. **EXIF Preservation** - Metadata handling
