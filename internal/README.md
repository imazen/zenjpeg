# Internal FFI Testing Infrastructure

This directory contains tools for verifying that `jpegli-rs` produces output identical to the original C++ jpegli implementation.

**These tools are for development/verification only - not required for using jpegli-rs.**

## Quick Start

```bash
# From repository root:
./internal/setup-ffi-tests.sh

# Run FFI comparison tests:
cargo test --features ffi-tests
```

## Requirements

- **Git** (for submodule)
- **CMake** 3.16+
- **Ninja** (recommended) or Make
- **C++ compiler**: GCC 9+, Clang 10+, or MSVC 2019+
- **~2GB disk space** for build artifacts
- **~10 minutes** for initial C++ build

### Platform-specific dependencies

**Ubuntu/Debian:**
```bash
sudo apt install cmake ninja-build build-essential pkg-config \
    libbrotli-dev libgif-dev libjpeg-dev libpng-dev
```

**macOS:**
```bash
brew install cmake ninja
```

**Windows:**
```powershell
# Install Visual Studio 2019+ with C++ workload
# Install CMake from https://cmake.org/download/
```

## What the FFI tests verify

1. **Quantization tables** - Identical quant values at each quality level
2. **Huffman tree building** - Identical code lengths and symbols
3. **DCT coefficients** - Matching coefficient values for test images
4. **Adaptive quantization** - Per-block AQ strength matches C++
5. **Final file size** - Output within 1% of C++ jpegli

## Directory Structure

```
internal/
├── README.md                    # This file
├── setup-ffi-tests.sh          # Setup script (Linux/macOS)
├── setup-ffi-tests.ps1         # Setup script (Windows)
├── jpegli-cpp/                 # Git submodule - C++ jpegli source
│   └── (cloned from libjxl)
└── jpegli-internals-sys/       # Rust FFI bindings crate
    ├── Cargo.toml
    ├── build.rs
    ├── src/lib.rs
    └── cpp/                    # C wrapper code
```

## Manual Setup

If the script doesn't work for your environment:

```bash
# 1. Clone submodule
git submodule update --init --recursive internal/jpegli-cpp

# 2. Build C++ jpegli
cd internal/jpegli-cpp
mkdir -p build && cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_TOOLS=ON \
    -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
    -DJPEGXL_ENABLE_SJPEG=OFF \
    ..
ninja jpegli-static cjpegli

# 3. Enable FFI tests in jpegli-rs
cd ../../..
# Edit jpegli-rs/Cargo.toml:
#   Uncomment: jpegli-internals-sys = { path = "..." }
#   Change: ffi-tests = ["dep:jpegli-internals-sys"]

# 4. Run tests
cargo test --features ffi-tests
```

## Troubleshooting

### "jpegli-cpp submodule not found"
```bash
git submodule update --init --recursive internal/jpegli-cpp
```

### CMake errors about missing dependencies
Install the platform-specific dependencies listed above.

### Linker errors about missing symbols
The C++ build may have failed. Check `internal/jpegli-cpp/build/` for errors.

### Tests fail with "C++ better by X%"
This is expected during development - it means there's a gap to investigate.
Check `CLAUDE.md` for known gaps and their causes.

## Generating Test Data

The C++ jpegli has instrumentation that outputs intermediate values:

```bash
cd internal/jpegli-cpp/build
GENERATE_RUST_TEST_DATA=1 ./tools/cjpegli input.png output.jpg

# Creates files like:
#   PerBlockModulations.testdata
#   CreateHuffmanTree.testdata
#   SetQuantMatrices.testdata
```

These `.testdata` files are JSON and can be parsed by Rust tests to verify
intermediate computation matches.

## Why This Isn't Published

`jpegli-internals-sys` requires the C++ source code which is ~100MB+ and
can't be bundled in a crates.io package. The FFI tests are for verifying
the pure Rust implementation during development, not for end users.

The published `jpegli-rs` crate is pure Rust and works without any of this.
