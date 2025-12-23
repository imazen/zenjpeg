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
cargo test -- --ignored  # Shows the failing DCT/IDCT roundtrip test
```

## Repository Info

- **Main branch**: `main`
- **Current working branch**: `stepbystep`
- **Rust project location**: `jpegli-rs/` subdirectory within the main jpegli repo
- **C++ source**: Root of repository (this is Google's libjxl/jpegli)

## Project Overview

This is a Rust port of **jpegli** - Google's improved JPEG encoder/decoder from the JPEG XL project. The goal is a complete, idiomatic Rust implementation with method-level accuracy parity to the C++ original.

## Project Structure

```
jpegli-rs/
├── Cargo.toml          # Workspace root
├── jpegli/             # Main library crate
│   └── src/
│       ├── lib.rs      # Module exports
│       ├── consts.rs   # Constants, tables, matrices
│       ├── types.rs    # Core types (ColorSpace, PixelFormat, etc.)
│       ├── huffman.rs  # Huffman coding
│       ├── quant.rs    # Quantization
│       ├── dct.rs      # Forward DCT (NEEDS FIX)
│       ├── idct.rs     # Inverse DCT (NEEDS FIX)
│       ├── color.rs    # RGB/YCbCr conversion
│       ├── xyb.rs      # XYB perceptual color space
│       ├── bitstream.rs # Bitstream I/O
│       ├── entropy.rs  # Entropy coding
│       ├── encode.rs   # Encoder pipeline
│       ├── decode.rs   # Decoder pipeline
│       ├── adaptive_quant.rs # Adaptive quantization
│       └── error.rs    # Error types
└── jpegli-sys/         # FFI bindings (for testing)
    ├── Cargo.toml
    └── build.rs
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
- [x] Build and fix compilation errors

## Pending Tasks

### 1. Fix DCT/IDCT Roundtrip (HIGH PRIORITY)

The current DCT/IDCT implementations don't form a proper inverse pair. The test in `idct.rs:200` is marked `#[ignore]`.

**Problem**: The Rust implementation uses a generic AAN algorithm, but jpegli uses a specific recursive splitting algorithm with different constants.

**C++ Algorithm Location**:
- Forward DCT: `lib/jpegli/dct-inl.h` - Uses recursive `DCT1DImpl<N>` template
- Inverse DCT: `lib/jpegli/idct.cc` - Uses recursive `IDCT1DImpl<N>` template

**Key C++ Constants** (from `dct-inl.h:110-117`):
```cpp
// WcMultipliers<8>::kMultipliers
0.5097955791041592,  // 1/(2*cos(0.5*pi/8))
0.6013448869350453,  // 1/(2*cos(1.5*pi/8))
0.8999762231364156,  // 1/(2*cos(2.5*pi/8))
2.5629154477415055,  // 1/(2*cos(3.5*pi/8))
```

**Algorithm Structure**:
1. DCT uses: `AddReverse` → recursive DCT → `SubReverse` → `Multiply` → recursive DCT → `B` → `InverseEvenOdd`
2. IDCT uses: `ForwardEvenOdd` → recursive IDCT → `BTranspose` → recursive IDCT → `MultiplyAndAdd`
3. Both do row pass → transpose → column pass → transpose
4. Scaling: DCT applies `1/8` factor in `StoreToBlockAndScale`

### DCT Algorithm Deep Dive

The jpegli DCT uses a **recursive splitting approach** (not the typical AAN or LLM algorithms).

**Forward DCT pseudo-code** (from `dct-inl.h`):
```
DCT1DImpl<8>(input):
    // Split into even/odd halves with reverse
    tmp[0:4] = input[0:4] + reverse(input[4:8])  // AddReverse
    tmp[4:8] = input[0:4] - reverse(input[4:8])  // SubReverse

    // Multiply odd part by Wc multipliers
    tmp[4:8] *= [0.5098, 0.6013, 0.8999, 2.5629]

    // Recursive DCT on both halves
    DCT1DImpl<4>(tmp[0:4])
    DCT1DImpl<4>(tmp[4:8])

    // B transform on odd part
    B<4>(tmp[4:8])  // tmp[0] *= sqrt(2), then cumulative sum

    // Interleave even/odd results
    output[even_indices] = tmp[0:4]
    output[odd_indices] = tmp[4:8]
```

**Inverse IDCT pseudo-code** (from `idct.cc`):
```
IDCT1DImpl<8>(input):
    // De-interleave even/odd
    tmp[0:4] = input[even_indices]
    tmp[4:8] = input[odd_indices]

    // Recursive IDCT on even half
    IDCT1DImpl<4>(tmp[0:4])

    // BTranspose on odd half (reverse of B)
    BTranspose<4>(tmp[4:8])

    // Recursive IDCT on odd half
    IDCT1DImpl<4>(tmp[4:8])

    // MultiplyAndAdd to reconstruct
    for i in 0..4:
        output[i] = tmp[i] + Wc[i] * tmp[4+i]
        output[7-i] = tmp[i] - Wc[i] * tmp[4+i]
```

**Key insight**: The forward and inverse transforms must use the SAME constants and be exact mirrors. The current Rust code uses different algorithms for DCT and IDCT which is why roundtrip fails.

**Full 2D Transform Flow**:
```
Forward:  pixels → DCT1D(rows) → Transpose → DCT1D(cols) → Transpose → coeffs × (1/8)
Inverse:  coeffs → Transpose → IDCT1D(cols) → Transpose → IDCT1D(rows) → pixels
```

**All Constants Needed** (copy these exactly):
```rust
// WcMultipliers<4>
const WC4: [f32; 2] = [0.541196100146197, 1.3065629648763764];

// WcMultipliers<8>
const WC8: [f32; 4] = [
    0.5097955791041592,
    0.6013448869350453,
    0.8999762231364156,
    2.5629154477415055,
];

const SQRT2: f32 = 1.41421356237;
```

**Transpose Helper** (needed for 2D transform):
```rust
fn transpose_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    for row in 0..8 {
        for col in 0..8 {
            output[col * 8 + row] = input[row * 8 + col];
        }
    }
}
```

### 2. Build C++ FFI Export DLL

Create FFI bindings to call C++ jpegli functions for comparison testing.

**Purpose**: Dual-execution testing - run both Rust and C++ implementations, compare outputs.

**Files needed**:
- `jpegli-sys/build.rs` - CMake integration
- `jpegli-sys/src/lib.rs` - FFI declarations

### 3. Create Test Asset Generation and Golden Tests

- Generate test images with known properties
- Create golden output files from C++ implementation
- Compare Rust output against golden files
- Tolerances: 1e-5 for intermediate f32, 1e-4 for final pixels

### 4. Port Progressive JPEG Support

Progressive JPEG uses multiple scans with spectral selection and successive approximation.

**Key types**: `ScanSpec` in `types.rs` already defined with `ss`, `se`, `ah`, `al` fields.

### 5. Performance Optimization with SIMD

Use `wide` crate for SIMD acceleration (equivalent to Highway in C++).

**Target functions**:
- DCT/IDCT transforms
- Color conversion
- Quantization

## C++ Build Instructions

### Ubuntu 22.04 (Primary Development Platform)

```bash
# Install dependencies
sudo apt update
sudo apt install -y cmake build-essential ninja-build \
    libbrotli-dev libgif-dev libjpeg-dev libpng-dev \
    libwebp-dev pkg-config

# Clone and build
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

**Ubuntu build outputs**:
- `lib/libjpegli-static.a`
- `tools/cjpegli`
- `tools/djpegli`

### Windows (Visual Studio 2022)

```bash
cd V:\GitHub\jpegli
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 \
    -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_TOOLS=ON \
    -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
    -DJPEGXL_ENABLE_SJPEG=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
cmake --build . --config Release --target jpegli-static cjpegli djpegli
```

**Windows build outputs**:
- `lib/jpegli/Release/jpegli-static.lib`
- `tools/Release/cjpegli.exe`
- `tools/Release/djpegli.exe`

## Key C++ Source Files

| C++ File | Purpose | Rust Equivalent |
|----------|---------|-----------------|
| `lib/jpegli/dct-inl.h` | Forward DCT | `dct.rs` |
| `lib/jpegli/idct.cc` | Inverse DCT | `idct.rs` |
| `lib/jpegli/color_transform.cc` | Color conversion | `color.rs` |
| `lib/jpegli/encode.cc` | Encoder | `encode.rs` |
| `lib/jpegli/decode.cc` | Decoder | `decode.rs` |
| `lib/jpegli/huffman.cc` | Huffman coding | `huffman.rs` |
| `lib/jpegli/quant.cc` | Quantization | `quant.rs` |
| `lib/jpegli/adaptive_quantization.cc` | Adaptive quant | `adaptive_quant.rs` |
| `lib/jpegli/bitstream.cc` | Bit I/O | `bitstream.rs` |
| `lib/jxl/enc_xyb.cc` | XYB color space | `xyb.rs` |

## Dependencies

**Rust crate dependencies** (in `jpegli/Cargo.toml`):
- `wide` - SIMD (Highway equivalent)
- `bytemuck` - Safe transmutes
- `arrayref` - Array references
- `thiserror` - Error handling

**Test dependencies**:
- `approx` - Floating point comparison
- `png` - Test image I/O
- `dssim` - Image quality metrics

## Architecture Notes

### Layer Structure
```
Layer 0: consts, types (pure data)
Layer 1: huffman, quant (pure math)
Layer 2: dct, idct, color, xyb (transforms)
Layer 3: bitstream (I/O)
Layer 4: entropy (stateful)
Layer 5-6: encode, decode (pipelines)
```

### XYB Color Space
- XYB is pure math with fixed constants - NO color management system needed
- Opsin absorbance matrix and bias values are in `consts.rs`
- CMS (lcms2) only needed at tool level for ICC profile handling

### FFI Testing Strategy
1. Build C++ as DLL with exported test hooks
2. Call both implementations with same input
3. Compare outputs within tolerance
4. Eventually transition to recorded test assets

## Running Tests

```bash
cd jpegli-rs
cargo test                    # Run all tests
cargo test -- --ignored       # Run ignored tests (currently failing)
cargo test --release          # Release mode
```

## Next Steps for New Developer

1. **Start with DCT/IDCT fix** - This is blocking everything else
   - Read `lib/jpegli/dct-inl.h` carefully
   - Port the recursive `DCT1DImpl<8>` algorithm exactly
   - Port the recursive `IDCT1DImpl<8>` algorithm exactly
   - Remove `#[ignore]` from roundtrip test when working

2. **Verify with simple cases**:
   - DC-only block (constant input)
   - Single AC coefficient
   - Known reference vectors

3. **Build FFI testing infrastructure** after DCT/IDCT works

## What Makes jpegli Special

jpegli provides better image quality than standard JPEG at the same file sizes through:

1. **Adaptive Quantization**: Analyzes image content to allocate more bits to complex regions
2. **XYB Color Space**: Perceptually optimized color space (from JPEG XL research)
3. **Improved Quantization Tables**: Better default tables than IJG libjpeg
4. **Float-based Pipeline**: Higher precision during encoding reduces artifacts
5. **Smart Zero-Biasing**: Intelligent coefficient rounding near zero

These improvements are **backward compatible** - output is standard JPEG readable by any decoder.

## Important Notes

### Test Tolerances
- Intermediate f32 calculations: tolerance of 1e-5
- Final pixel values: tolerance of 1e-4
- DCT coefficients should match exactly (integer values after quantization)

### Design Decisions
- **No libjpeg API compatibility**: This is intentional. We want idiomatic Rust.
- **FFI is testing-only**: Not for production use, just to verify Rust matches C++
- **SIMD comes last**: Get scalar version correct first, then optimize

### Files You'll Edit Most
- `jpegli-rs/jpegli/src/dct.rs` - Forward DCT needs complete rewrite
- `jpegli-rs/jpegli/src/idct.rs` - Inverse DCT needs complete rewrite

### Useful Commands
```bash
# Run specific test
cargo test test_dct_idct_roundtrip -- --ignored --nocapture

# Check for compile errors without full build
cargo check

# Run with debug output
RUST_BACKTRACE=1 cargo test

# Format code
cargo fmt

# Lint
cargo clippy
```

## Git Workflow

```bash
# Current state
git status  # Shows jpegli-rs/ as new untracked directory

# To commit Rust work
git add jpegli-rs/
git commit -m "Add jpegli-rs Rust port (WIP)"

# Stay on stepbystep branch for development
# PR to main when features are complete
```
