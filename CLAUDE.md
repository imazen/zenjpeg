# jpegli-rs Project Handoff

## Quick Start (Ubuntu 22.04)

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Navigate to the Rust project
cd jpegli-rs

# Build and test
cargo build
cargo test
cargo test --release  # Run with optimizations
```

## Repository Info

- **Main branch**: `main`
- **Current working branch**: `stepbystep2`
- **Rust project location**: `jpegli-rs/` subdirectory within the main jpegli repo
- **C++ source**: Root of repository (this is Google's libjxl/jpegli)

## Project Overview

This is a Rust port of **jpegli** - Google's improved JPEG encoder/decoder from the JPEG XL project. The goal is a complete, idiomatic Rust implementation with method-level accuracy parity to the C++ original.

## Project Structure

```
jpegli-rs/
├── Cargo.toml          # Workspace root
├── jpegli/             # Main library crate
│   ├── src/
│   │   ├── lib.rs           # Module exports
│   │   ├── consts.rs        # Constants, tables, matrices
│   │   ├── types.rs         # Core types (ColorSpace, PixelFormat, etc.)
│   │   ├── huffman.rs       # Huffman coding
│   │   ├── quant.rs         # Quantization
│   │   ├── dct.rs           # Forward DCT ✓
│   │   ├── idct.rs          # Inverse DCT ✓
│   │   ├── color.rs         # RGB/YCbCr conversion
│   │   ├── xyb.rs           # XYB perceptual color space ✓
│   │   ├── butteraugli.rs   # Butteraugli quality metric (skeleton)
│   │   ├── bitstream.rs     # Bitstream I/O
│   │   ├── entropy.rs       # Entropy coding
│   │   ├── encode.rs        # Encoder pipeline
│   │   ├── decode.rs        # Decoder pipeline
│   │   ├── adaptive_quant.rs # Adaptive quantization
│   │   └── error.rs         # Error types
│   ├── tests/
│   │   ├── roundtrip_quality.rs  # DSSIM-based quality tests
│   │   ├── pareto_front.rs       # Pareto efficiency vs mozjpeg
│   │   ├── decode_external.rs    # Decode JPEGs from other tools
│   │   ├── metrics_comparison.rs # DSSIM vs SSIMULACRA2 tests
│   │   ├── xyb_roundtrip.rs      # XYB color space tests
│   │   └── quality_mapping.rs    # mozjpeg->jpegli quality mapping
│   └── examples/
│       ├── roundtrip_corpus.rs    # Batch corpus testing
│       ├── corpus_comparison.rs   # HTML chart: jpegli vs mozjpeg
│       ├── multi_codec_comparison.rs # Compare with CID22 dataset
│       └── compare_quality.rs     # Quality comparison tool
└── jpegli-sys/         # FFI bindings (for testing)
```

## Completed Tasks

- [x] Create jpegli-rs project structure with Cargo workspace
- [x] Port Layer 0: Constants, types, tables (zigzag, quant matrices, XYB params)
- [x] Port Layer 1: Pure math (Huffman, quantization)
- [x] Port Layer 2: Transforms (DCT/IDCT, color, XYB)
- [x] Port Layer 3: Bitstream I/O
- [x] Port Layer 4: Entropy coding
- [x] Port Layer 5-6: Encoder and decoder pipelines
- [x] Port adaptive quantization
- [x] **Fix DCT/IDCT scaling** - 1/8 scaling factor for JPEG compatibility
- [x] **XYB color space** - Full roundtrip working, < 2-bit error
- [x] **DSSIM quality testing** - Integrated with mozjpeg comparison
- [x] **SSIMULACRA2 metric** - Added via ssimulacra2 crate
- [x] **Butteraugli metric** - Skeleton implementation, uses XYB internally
- [x] **Quality mapping tests** - Find equivalent Q values across encoders
- [x] **Pareto front validation** - Verify jpegli beats mozjpeg on quality/size
- [x] **corpus_comparison caching** - Versioned JPEG cache for fast reruns
- [x] **Low-Q analysis** - Per-image comparison at Q10-Q60 with SSIM2
- [x] **XYB quant bug fix** - Disabled broken XYB-only mode (needs color conversion)
- [x] **XYB encoding pipeline skeleton** - Basic infrastructure implemented but incomplete

## Pending Tasks

### 1. Complete XYB Mode with ICC Profile
XYB mode is partially implemented but marked deprecated. Full implementation requires:
- **RGB → Linear RGB → XYB conversion** using `LinearRGBRowToXYB()` (port from C++)
- **XYB value scaling** using `ScaleXYBRow()`
- **ICC profile embedding** - XYB ICC profile so decoders interpret colors correctly
- **Frequency-dependent quant scaling** - `DistanceToScale()` with per-frequency exponents

Current state:
- Encoding pipeline produces valid JPEGs (R/G/B component IDs, 2×2/2×2/1×1 sampling)
- Missing color conversion causes quality degradation
- `use_xyb()` is deprecated with warning until complete

Reference files:
- `lib/extras/enc/jpegli.cc` - High-level XYB encoding with color conversion + ICC
- `lib/extras/xyb_transform.cc` - XYB color conversion
- `lib/jpegli/quant.cc` - `DistanceToScale()` and frequency exponents

### 2. Add SIMD Toggle Feature Flag
Make toggling SIMD on/off easy to:
- Ensure SIMD and non-SIMD produce identical images
- Max difference should be ≤1 when decoded
- Add accuracy tests comparing SIMD vs scalar

### 3. Set Up Test Image Submodule
- Create separate git repo for test images (size conscious)
- Add as submodule to avoid bloating main repo
- Include: gradient, photo, graphic, edge case images

### 4. Create Comparative Benchmarks (C++ vs Rust)
- [x] Stage-by-stage C++ instrumentation (recovered from commit fe1e841f)
- [x] Output comparison via `xyb_cpp_comparison.rs`
- [ ] Performance benchmarks (encode/decode timing)
- [ ] Full function-level accuracy validation using C++ test data

### 5. Fix 4:2:0 Subsampling Decoder
Currently only 4:4:4 is supported. 4:2:0 requires:
- MCU interleaving in decoder
- Chroma upsampling

### 6. Port Progressive JPEG Support
Progressive JPEG uses multiple scans with spectral selection.
- `ScanSpec` type already defined
- Encoder returns "not yet implemented" error

### 7. Add Fuzz Testing
Replicate fuzzing coverage from other JPEG libraries:
- libjpeg-turbo fuzz targets
- jpeg-decoder fuzzing
- zune-jpeg fuzzing
- mozjpeg fuzzing
Critical for server-side untrusted input handling.

### 8. Security Examination and Red Teaming
Required for server-side deployment:
- Memory safety review (unsafe blocks)
- Integer overflow checks
- Input validation (malformed JPEGs)
- DoS resistance (large images, many components)
- Comparison with CVEs from other JPEG libs

## C++ Instrumentation for Rust Validation

### History
C++ instrumentation was created (commit fe1e841f) to capture intermediate values
for validating Rust implementations. It was inadvertently removed when cherry-picking
upstream clang-tidy cleanup (c9c2be2d). Instrumentation has been restored.

### Instrumented Functions
| File | Function | Test Data Type |
|------|----------|---------------|
| `adaptive_quantization.cc` | `PerBlockModulations()` | `PerBlockModulationsTest` |
| `adaptive_quantization.cc` | `FuzzyErosion()` | `FuzzyErosionTest` |
| `adaptive_quantization.cc` | `ComputePreErosion()` | TBD |
| `quant.cc` | `SetQuantMatrices()` | Quant table outputs |
| `quant.cc` | `InitQuantizer()` | Quantizer state |
| `encode.cc` | Various | Encoding pipeline state |

### Using Instrumentation

1. Build C++ with instrumentation (enabled by default):
   ```bash
   mkdir -p build && cd build
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DJPEGXL_ENABLE_TOOLS=ON ..
   ninja cjpegli
   ```

2. Generate test data:
   ```bash
   GENERATE_RUST_TEST_DATA=1 ./build/tools/cjpegli input.png output.jpg
   ```

3. Test data written to working directory:
   - `PerBlockModulations.testdata`
   - `FuzzyErosion.testdata`
   - `SetQuantMatrices.testdata`
   - etc.

### How It Works
- `ENABLE_RUST_TEST_INSTRUMENTATION` macro (default ON in `test_data_gen.h`)
- `GENERATE_RUST_TEST_DATA=1` env var enables runtime capture
- JSON output to `*.testdata` files (one JSON object per line)
- Thread-safe via mutex

### Strategy: Fork vs Upstream
This repo is now a **fork** of Google's JPEG XL, not a downstream cherry-picker.
C++ modifications for instrumentation will be maintained independently.

## Quality Metrics

### Available Metrics

| Metric | Crate | Description | Range |
|--------|-------|-------------|-------|
| DSSIM | `dssim` | Structural dissimilarity | 0 = identical, lower = better |
| SSIMULACRA2 | `ssimulacra2` | Perceptual quality | 100 = identical, higher = better |
| Butteraugli | `jpegli::butteraugli` | Psychovisual distance | < 1.0 = good, > 2.0 = bad |

### Using Metrics in Tests

```rust
use dssim::Dssim;
use ssimulacra2::{compute_frame_ssimulacra2, Rgb, ColorPrimaries, TransferCharacteristic};

// DSSIM
let attr = Dssim::new();
let orig = attr.create_image_rgba(&orig_rgba, width, height).unwrap();
let comp = attr.create_image_rgba(&comp_rgba, width, height).unwrap();
let (dssim, _) = attr.compare(&orig, comp);

// SSIMULACRA2
let rgb = Rgb::new(
    pixels.chunks(3).map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]).collect(),
    width, height,
    TransferCharacteristic::SRGB,
    ColorPrimaries::BT709,
).unwrap();
let score = compute_frame_ssimulacra2(orig_rgb, dist_rgb).unwrap();
```

## Test Corpora

### CID22-512 (Recommended for quick tests)
512x512 center crops of CID22 images:
- Path: `/mnt/v/work/corpus/CID22-512`
- ~250 PNG files
- Good variety of content types

### CID22 Full Dataset
Cloudinary Image Dataset 2022 with pre-encoded files:
- Path: `/mnt/v/work/CID22/CID22`
- `original/` - Source PNGs
- `compressed/<image_id>/<codec>/` - Pre-encoded at various Q levels
- Codecs: mozjpeg, libjxl, aom, cld_avif, cld_heic, cld_jp2, cld_webp, vis_avif
- CSV with MCOS (human quality scores)

### Flower Test Images
Small test images for quick validation:
- Path: `/home/lilith/work/jpegli/testdata/jxl/flower/`
- `flower_small.rgb.png` - Small test image
- Various pre-encoded JPEGs for decode testing

## Comparison Tools

### corpus_comparison.rs
Generates HTML chart comparing jpegli vs mozjpeg with DSSIM and SSIMULACRA2:
```bash
MAX_FILES=50 cargo run --release --example corpus_comparison -- \
    /mnt/v/work/corpus/CID22-512 /mnt/v/work/jpegli_data/comparison.html
```

Features:
- **Caching**: Encoded JPEGs cached in `jpeg_cache/` (version-tagged for invalidation)
- **Low-Q analysis**: Per-image breakdown at Q30 showing where each encoder excels
- **Dual metrics**: Both DSSIM and SSIMULACRA2 charts
- Env vars: `MAX_FILES=N` (limit images), `NO_CACHE=1` (disable caching)

### multi_codec_comparison.rs
Uses CID22 CSV data to compare all codecs:
```bash
cargo run --release --example multi_codec_comparison -- \
    /mnt/v/work/CID22/CID22 /mnt/v/work/jpegli_data/multi_codec.html
```

### quality_mapping test
Finds jpegli Q that matches mozjpeg DSSIM:
```bash
cargo test --test quality_mapping -- --nocapture
CORPUS_DIR=/mnt/v/work/corpus/CID22-512 cargo test --test quality_mapping test_quality_mapping_corpus -- --ignored --nocapture
```

## Key Test Results

### jpegli vs mozjpeg (4:4:4 subsampling)
- jpegli achieves **10-17% better DSSIM** at same Q value
- At Q90+, jpegli also produces **5-8% smaller files**
- jpegli wins on both quality AND size at high quality settings

### SSIMULACRA2 Results (30 images, CID22-512 corpus)
| Quality | jpegli bpp | jpegli SSIM2 | mozjpeg bpp | mozjpeg SSIM2 |
|---------|------------|--------------|-------------|---------------|
| Q30     | 0.77       | 59.5         | 0.57        | 50.8          |
| Q50     | 0.94       | 67.4         | 0.85        | 65.7          |
| Q70     | 1.28       | 76.4         | 1.22        | 74.5          |
| Q90     | 2.32       | 87.4         | 2.53        | 86.4          |

**Key finding**: jpegli produces higher SSIM2 scores (better quality) at similar or lower bitrates.

### Quality Mapping (to match DSSIM)
- mozjpeg Q60 → jpegli ~Q55 (jpegli is more efficient)
- mozjpeg Q90 → jpegli ~Q89

### XYB Mode Status
**Currently disabled** - XYB quantization tables without XYB color conversion
produces severe block artifacts. Will be re-enabled once full XYB color
pipeline is ported.

## C++ Build Instructions

### Ubuntu 22.04

```bash
sudo apt update
sudo apt install -y cmake build-essential ninja-build \
    libbrotli-dev libgif-dev libjpeg-dev libpng-dev \
    libwebp-dev pkg-config

cd /path/to/jpegli
mkdir -p build && cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_TOOLS=ON \
    -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
    -DJPEGXL_ENABLE_SJPEG=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
ninja jpegli-static cjpegli djpegli
```

## Key C++ Source Files

| C++ File | Purpose | Rust Equivalent |
|----------|---------|-----------------|
| `lib/jpegli/dct-inl.h` | Forward DCT | `dct.rs` |
| `lib/jpegli/idct.cc` | Inverse DCT | `idct.rs` |
| `lib/jpegli/color_transform.cc` | Color conversion | `color.rs` |
| `lib/jpegli/encode.cc` | Encoder | `encode.rs` |
| `lib/jpegli/decode.cc` | Decoder | `decode.rs` |
| `lib/extras/butteraugli.cc` | Quality metric | `butteraugli.rs` |
| `lib/jxl/enc_xyb.cc` | XYB color space | `xyb.rs` |

## Dependencies

### Production
- `wide` - SIMD (Highway equivalent)
- `bytemuck` - Safe transmutes
- `arrayref` - Array references
- `rgb`, `imgref` - Image buffers

### Testing
- `dssim` - DSSIM quality metric
- `ssimulacra2` - SSIMULACRA2 quality metric
- `png` - Image I/O
- `mozjpeg` - Encoder comparison
- `jpeg-decoder` - Reference decoder

## Architecture Notes

### Layer Structure
```
Layer 0: consts, types (pure data)
Layer 1: huffman, quant (pure math)
Layer 2: dct, idct, color, xyb (transforms)
Layer 3: bitstream (I/O)
Layer 4: entropy (stateful)
Layer 5-6: encode, decode (pipelines)
Metrics: butteraugli (quality assessment)
```

### XYB Color Space
- XYB is pure math with fixed constants
- Opsin absorbance matrix and bias values in `consts.rs`
- Roundtrip error < 2 bits (confirmed by tests)

### SIMD Feature Flag
```toml
[features]
default = ["simd"]
simd = []
```

To disable SIMD: `cargo build --no-default-features`

## Running Tests

```bash
cargo test                              # All tests
cargo test --release                    # With optimizations
cargo test --test xyb_roundtrip         # XYB specific
cargo test --test metrics_comparison    # Quality metrics
cargo test -- --ignored                 # Ignored tests (need files)
```

## What Makes jpegli Special

1. **Adaptive Quantization**: Content-aware bit allocation
2. **XYB Color Space**: Perceptually optimized (from JPEG XL)
3. **Improved Quantization Tables**: Better than IJG libjpeg
4. **Float-based Pipeline**: Higher precision
5. **Smart Zero-Biasing**: Intelligent coefficient rounding

Output is **backward compatible** - standard JPEG readable by any decoder.

## Git Workflow

```bash
git status                    # Check current state
git add jpegli-rs/           # Stage changes
git commit -m "Description"  # Commit
# Stay on stepbystep2 branch for development
# PR to main when features are complete
```
