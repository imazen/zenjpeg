# zenjpeg Development Guide

## Project Overview

zenjpeg is a high-quality JPEG encoder that combines the best techniques from mozjpeg and jpegli to achieve Pareto-optimal compression at both low and high quality settings.

**Key insight**: mozjpeg's trellis quantization excels at low quality (Q < 70), while jpegli's adaptive quantization excels at high quality (Q >= 70). zenjpeg automatically selects the best strategy.

## Quick Start

```bash
cd /home/lilith/work/zenjpeg
cargo test          # 20 tests pass
cargo build         # Build library
```

## Architecture

```
zenjpeg/
├── src/
│   ├── lib.rs              # Public API re-exports
│   ├── error.rs            # Error types (#[non_exhaustive])
│   ├── types.rs            # Core types (Quality, Subsampling, ColorSpace)
│   ├── consts.rs           # Constants, tables, JPEG markers
│   ├── color.rs            # RGB→YCbCr conversion
│   ├── dct.rs              # Forward DCT
│   ├── quant.rs            # Quantization tables
│   ├── huffman.rs          # Huffman table handling
│   ├── entropy.rs          # Entropy encoding (bitstream)
│   ├── trellis.rs          # Trellis quantization (from mozjpeg)
│   ├── adaptive_quant.rs   # Adaptive quantization (from jpegli)
│   ├── strategy.rs         # Encoding strategy selection
│   └── encode.rs           # Main Encoder API
├── tests/                  # Integration tests (TBD)
├── benches/                # Benchmarks (TBD)
├── ARCHITECTURE.md         # Detailed architecture docs
├── RESEARCH.md             # Research findings and decisions
└── CLAUDE.md               # This file
```

## Current Status

### Completed
- [x] Project structure and module layout
- [x] Core types (Quality, Subsampling, ColorSpace, PixelFormat)
- [x] Error handling with #[non_exhaustive]
- [x] Color conversion (RGB→YCbCr)
- [x] Forward DCT (reference implementation)
- [x] Quantization tables (standard + mozjpeg)
- [x] Huffman table handling with two-pass optimization
- [x] Entropy encoding with symbol frequency counting
- [x] Full trellis quantization from mozjpeg (rate-distortion optimization)
- [x] Quality-aware strategy selection (auto/mozjpeg/jpegli/hybrid)
- [x] Basic Encoder API with Huffman optimization
- [x] 36 unit/integration tests passing
- [x] Pareto benchmark (`examples/pareto_benchmark.rs`)

### Current Performance (Dec 2024)
At SSIM2 >= 80 quality target:
- jpegli: **1.310 bpp** (best efficiency)
- mozjpeg-oxide: 1.437 bpp
- **zenjpeg: 1.458 bpp** (only 1.5% larger than mozjpeg-oxide!)

zenjpeg appears on Pareto front at multiple quality levels:
- Low quality: Q30 at 0.485 bpp
- Very high quality: Q90-Q95 at SSIM2 86-91 (outperforms jpegli at similar bitrates)

### Key Improvements Made
1. **Two-pass Huffman optimization**: Proper frequency counting with Huffman's algorithm
2. **Full trellis quantization**: Dynamic programming with actual Huffman code lengths
3. **Quality-aware lambda tuning**: Trellis enabled for Q<80, disabled at Q>=80

### Pending
- [ ] Port jpegli's perceptual adaptive quantization (complex algorithm)
- [ ] Port SIMD DCT from mozjpeg-rs (see below)

### SIMD DCT Available in mozjpeg-rs

**Location**: `/home/lilith/work/mozjpeg-rs/src/dct.rs` (1,546 lines)

Three implementations ready to port:
1. **Scalar Loeffler** with `#[multiversion]` autovectorization
2. **Row-parallel SIMD** using `wide::i32x4` (4 rows at once)
3. **Transpose-based SIMD** using `wide::i32x8` (8 rows at once, best for cache)

**Hand-written AVX2**: `/home/lilith/work/mozjpeg-rs/src/simd/x86_64/avx2.rs` (502 lines)
- Uses `core::arch::x86_64` intrinsics
- ~15% faster than autovectorized version
- Gated behind `simd-intrinsics` feature flag

**Dependencies**:
- `multiversion` crate for compile-time platform dispatch
- `wide` crate for portable SIMD vectors
- All have scalar fallbacks for unsupported platforms

### Completed
- [x] Progressive encoding (full support with all scan scripts)
- [x] Documentation updates (ARCHITECTURE.md)
- [x] Metric-specific strategy selection (OptimizeFor API)
- [x] Perceptual quality targeting (binary search)

### Known Limitations
- Simplified variance-based AQ hurts quality - needs full jpegli perceptual AQ port
- Trellis disabled at Q>=80 to prevent quality degradation
- 11% larger than jpegli at SSIM2>=80 (gap requires perceptual AQ to close)

## Key Design Decisions

### 1. Strategy Selection
- Q < 50: Use mozjpeg strategy (trellis + progressive)
- Q 50-70: Use hybrid strategy (both trellis + AQ with reduced strength)
- Q >= 70: Use jpegli strategy (adaptive quantization)

### 2. Quality Modes
```rust
pub enum Quality {
    Standard(u8),      // 1-100, auto-selects strategy
    Low(u8),           // Forces mozjpeg strategy
    High(u8),          // Forces jpegli strategy
    Perceptual(f32),   // Target SSIMULACRA2 score
    TargetSize(usize), // Binary search for quality
}
```

### 3. Encoder Presets
```rust
Encoder::new()             // Balanced (Q85, auto strategy)
Encoder::max_compression() // Low quality, trellis, progressive
Encoder::max_quality()     // High quality, adaptive quant
Encoder::fastest()         // No optimization, baseline
```

## Testing

```bash
cargo test                    # Run all tests
cargo test --release          # With optimizations
cargo test encode             # Only encoder tests
cargo test -- --nocapture     # Show output
```

## Dependencies

### Core
- `mozjpeg-oxide` (path: ../mozjpeg-rs) - mozjpeg Rust port
- `jpegli` (path: ../jpegli/jpegli-rs/jpegli) - jpegli Rust port
- `bytemuck` - Safe transmutes
- `wide` - SIMD

### Dev/Testing
- `codec-eval` (path: ../codec-eval) - Quality metrics and comparison
- `butteraugli-oxide` - Perceptual quality metric
- `dssim` - DSSIM quality metric
- `ssimulacra2` - SSIMULACRA2 quality metric
- `png` - Image I/O
- `jpeg-decoder` - JPEG verification

## Workflow

### Making Changes
1. Run `cargo fmt` before changes
2. Make changes
3. Run `cargo test`
4. Commit with descriptive message

### Adding Features
1. Add module with unit tests
2. Wire into encode.rs
3. Add integration tests
4. Update ARCHITECTURE.md
5. Benchmark with codec-eval

### Benchmarking
```bash
# Run Pareto comparison (uses Kodak corpus)
cargo run --release --example pareto_benchmark

# Results written to comparison_outputs/
# - pareto_front.json   # Pareto-optimal points
# - all_points.csv      # All quality/size data points
```

## Quality Metrics

**Use DSSIM or SSIMULACRA2, NOT PSNR/MSE.**

| Metric | Package | Range | Notes |
|--------|---------|-------|-------|
| DSSIM | dssim | 0 = identical | Primary metric |
| SSIMULACRA2 | ssimulacra2 | 100 = identical | Secondary |
| Butteraugli | butteraugli-oxide | <1.0 good | Perceptual |

## Common Issues

### Import Conflicts
Types like `TrellisConfig` and `AdaptiveQuantConfig` are defined in their respective modules and re-exported from `types.rs`. Import from `crate::types` for public API.

### Test Images
Use codec-corpus for test images:
```rust
use codec_eval::corpus::Corpus;
let corpus = Corpus::discover()?;
```

## References

- [mozjpeg-rs CLAUDE.md](../mozjpeg-rs/CLAUDE.md) - mozjpeg implementation details
- [jpegli-rs CLAUDE.md](../jpegli/jpegli-rs/CLAUDE.md) - jpegli implementation details
- [codec-eval CLAUDE.md](../codec-eval/CLAUDE.md) - Evaluation methodology
