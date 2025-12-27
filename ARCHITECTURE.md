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

Current: Reference implementation + integer DCT
TODO: SIMD DCT (check mozjpeg-rs for existing work)

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

Features:
- Two-pass Huffman optimization (frequency counting + optimal table generation)
- Progressive encoding with successive approximation
- AC refinement with correction bits

### Stage 6: Progressive Encoding (progressive.rs, entropy.rs)

Full progressive JPEG support with three scan script presets:
- **Minimal**: DC + full AC scans (fast, good for web)
- **Simple**: DC + AC bands 1-5 and 6-63 (better progressive display)
- **Standard**: With successive approximation (best compression)

Progressive encoding features:
- DC first/refinement scans
- AC first/refinement scans with spectral band selection
- EOBRUN accumulation for runs of all-zero blocks
- Proper correction bit ordering for AC refinement

## Strategy Selection (strategy.rs)

### Automatic Encoder Selection

Based on research across 382 BPP-matched comparisons (CLIC 2025 + Kodak):

| Metric | mozjpeg wins | jpegli wins | Best default |
|--------|-------------|-------------|--------------|
| Butteraugli | 16% | **84%** | jpegli |
| DSSIM | **67%** | 33% | mozjpeg |
| SSIMULACRA2 | 51% | 49% | balanced |

The prediction model (86.6% accuracy) uses:
- Flat block percentage (>75% = flat)
- Edge strength + local contrast (complexity)
- Estimated BPP from quality

### OptimizeFor API

```rust
encoder.optimize_for(OptimizeFor::Butteraugli)  // Default - web images
encoder.optimize_for(OptimizeFor::Dssim)        // Archival/medical
encoder.optimize_for(OptimizeFor::Ssimulacra2)  // General purpose
encoder.optimize_for(OptimizeFor::FileSize)     // Smallest files
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

## Quality Modes

```rust
pub enum Quality {
    Standard(u8),      // 1-100, auto-selects strategy
    Low(u8),           // Forces mozjpeg strategy
    High(u8),          // Forces jpegli strategy
    Perceptual(f32),   // Target Butteraugli distance (binary search)
    TargetSize(usize), // Binary search for file size
    Linear(f32),       // Perceptually uniform quality steps
}
```

### Perceptual Quality Targeting

`Quality::Perceptual(distance)` uses binary search:
1. Estimate starting quality from distance formula
2. 4 iterations of binary search around estimate
3. Return JPEG closest to target Butteraugli distance

### Target Size Encoding

`Quality::TargetSize(bytes)` uses binary search:
1. Start with Q10-Q100 range
2. 7 iterations of binary search
3. Return highest quality that fits target size

## Image Analysis (analysis.rs)

Fast image analysis for encoder selection:
- Samples every 4th block for variance
- Samples every 4th pixel for edge detection
- ~1/16th of full image scan cost

Metrics computed:
- `flat_block_pct`: Percentage of low-variance blocks
- `edge_strength_mean`: Mean gradient magnitude
- `local_contrast_mean`: Mean local variance
- `luma_complexity`: Overall detail level

## File Format

Standard JFIF JPEG structure:
```
SOI (FFD8)
APP0 (JFIF header)
DQT (quantization tables)
SOF0/SOF2 (frame header - baseline or progressive)
DHT (Huffman tables)
SOS (scan header) × N for progressive
Entropy data (with byte stuffing)
EOI (FFD9)
```

## Module Dependencies

```
lib.rs
├── error.rs          (standalone)
├── types.rs          (OptimizeFor, Quality, ScanScript, etc.)
├── analysis.rs       (image analysis, metric-aware selection)
├── consts.rs         (JPEG constants, standard tables)
├── color.rs          (RGB→YCbCr conversion)
├── dct.rs            (forward DCT)
├── quant.rs          (quantization tables, zero-bias)
├── huffman.rs        (Huffman table handling)
├── entropy.rs        (entropy encoding, progressive encoder)
├── progressive.rs    (scan script generation)
├── trellis.rs        (rate-distortion optimization)
├── adaptive_quant.rs (jpegli-style AQ)
├── strategy.rs       (encoder selection)
├── jpegli/           (forked jpegli encoder)
│   ├── encode.rs
│   ├── huffman_opt.rs
│   └── adaptive_quant.rs
└── encode.rs         (main Encoder API)
```

## Performance Considerations

### Current Status
- DCT: Reference + integer implementations
- Huffman: Two-pass optimization implemented
- Progressive: Full support with all scan scripts
- Trellis: Enabled for Q < 80

### Optimization TODO
1. SIMD DCT - check mozjpeg-rs for existing work
2. SIMD color conversion
3. Parallel block processing

## Testing Strategy

### Unit Tests
206 tests covering:
- DCT accuracy
- Quantization
- Huffman encoding
- Trellis optimization
- Progressive encoding
- Encoder presets

### Integration Tests
- Roundtrip tests with jpeg_decoder
- Multi-decoder verification (jpeg_decoder, djpeg, ImageMagick)
- Progressive scan validation

### Quality Tests
- codec-eval integration for Pareto analysis
- Butteraugli/DSSIM/SSIMULACRA2 metrics

## Current Limitations

1. **Simplified AQ**: Full jpegli perceptual AQ not yet ported
2. **Trellis at high Q**: Disabled for Q >= 80 (hurts quality without proper AQ)
3. **No XYB mode**: YCbCr only (XYB available via jpegli delegation)

## Future Work

1. **Full jpegli AQ port** - Pre-erosion, fuzzy erosion, modulations
2. **SIMD acceleration** - DCT, color conversion
3. **XYB Color Space** - Optional mode for better high-quality
4. **Restart Markers** - Error resilience
5. **ICC Profiles** - Color management
6. **EXIF Preservation** - Metadata handling
