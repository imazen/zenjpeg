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
- [x] Huffman table handling
- [x] Entropy encoding
- [x] Trellis quantization (simplified)
- [x] Adaptive quantization (simplified)
- [x] Strategy selection (auto/mozjpeg/jpegli/hybrid)
- [x] Basic Encoder API
- [x] 20 unit tests passing

### In Progress
- [ ] Documentation (ARCHITECTURE.md, RESEARCH.md)
- [ ] Integration with codec-eval for benchmarking

### Pending
- [ ] Full trellis implementation from mozjpeg-rs
- [ ] Full adaptive quantization from jpegli-rs
- [ ] Progressive encoding
- [ ] Huffman table optimization
- [ ] SIMD acceleration
- [ ] Pareto front benchmarking
- [ ] Quality tuning

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
# Run Pareto comparison (TBD)
cargo run --release --example pareto_comparison
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
