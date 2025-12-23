# Jpegli Build Notes

## C++ Build (Windows, VS 2022 Build Tools)

### Prerequisites
- Visual Studio 2022 Build Tools (MSVC 19.44+)
- CMake 3.16+ (in PATH)
- Git submodules initialized

### Initialize Submodules
```powershell
git submodule update --init --recursive --depth 1 --recommend-shallow
```

### Configure
```powershell
cd V:\GitHub\jpegli
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DBUILD_TESTING=OFF ^
    -DJPEGXL_ENABLE_TOOLS=ON ^
    -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON ^
    -DJPEGXL_ENABLE_SJPEG=OFF ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
    ..
```

### Build
```powershell
# Build everything
cmake --build . --config Release

# Or build specific targets
cmake --build . --config Release --target jpegli-static
cmake --build . --config Release --target cjpegli djpegli
```

### Output Locations
- Library: `build\lib\Release\jpegli-static.lib`
- Encoder: `build\tools\Release\cjpegli.exe`
- Decoder: `build\tools\Release\djpegli.exe`
- DLLs: `build\lib\Release\jxl_cms.dll`, `jxl_threads.dll`

### Known Issues
- `sjpeg` submodule has outdated CMake - disabled via `-DJPEGXL_ENABLE_SJPEG=OFF`
- Optional format support (PNG, GIF, EXR, WebP) not built - not needed for core jpegli

### Clean Rebuild
```powershell
cd V:\GitHub\jpegli
rm -rf build
# Then run configure and build steps again
```

---

## Porting Strategy: FFI Dual-Execution Comparison

### Approach
Instead of capturing test data files, we use FFI to call C++ functions directly from Rust tests and compare outputs in real-time.

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│  Rust Test Harness                                      │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │ Rust impl       │    │ C++ impl (FFI)  │            │
│  │ my_dct(input)   │    │ cpp_dct(input)  │            │
│  └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │
│           └──────┬───────────────┘                      │
│                  ▼                                      │
│         compare_with_tolerance(rust_out, cpp_out, 1e-5) │
└─────────────────────────────────────────────────────────┘
```

### Benefits
1. No test data file management
2. Immediate feedback on discrepancies
3. Test with random/fuzzed inputs
4. Pinpoint exact function where divergence occurs

### Implementation Steps

1. **Build jpegli as DLL** (add to CMake):
   ```cmake
   add_library(jpegli-ffi SHARED ${JPEGLI_SOURCES})
   target_compile_definitions(jpegli-ffi PRIVATE JPEGLI_FFI_EXPORTS)
   ```

2. **Create C export wrappers** (`lib/jpegli/ffi_exports.cc`):
   ```cpp
   extern "C" {
     // Pure functions - direct export
     JPEGLI_FFI_API void ffi_idct_8x8(const int16_t* qblock,
                                       const float* dequant,
                                       float* output);

     JPEGLI_FFI_API void ffi_rgb_to_ycbcr(float* row0, float* row1,
                                          float* row2, size_t width);

     JPEGLI_FFI_API void ffi_build_huffman_table(const uint32_t* counts,
                                                  const uint32_t* symbols,
                                                  void* lut_out);
   }
   ```

3. **Rust FFI bindings** (`jpegli-rs/src/ffi.rs`):
   ```rust
   #[link(name = "jpegli-ffi")]
   extern "C" {
       pub fn ffi_idct_8x8(qblock: *const i16, dequant: *const f32, output: *mut f32);
       pub fn ffi_rgb_to_ycbcr(row0: *mut f32, row1: *mut f32, row2: *mut f32, width: usize);
   }
   ```

4. **Comparison test pattern**:
   ```rust
   #[test]
   fn test_idct_matches_cpp() {
       let qblock: [i16; 64] = random_coefficients();
       let dequant: [f32; 64] = standard_dequant_table();

       let mut rust_out = [0f32; 64];
       let mut cpp_out = [0f32; 64];

       // Rust implementation
       idct_8x8(&qblock, &dequant, &mut rust_out);

       // C++ via FFI
       unsafe { ffi_idct_8x8(qblock.as_ptr(), dequant.as_ptr(), cpp_out.as_mut_ptr()); }

       // Compare
       for i in 0..64 {
           assert!((rust_out[i] - cpp_out[i]).abs() < 1e-5,
                   "Mismatch at [{}]: rust={}, cpp={}", i, rust_out[i], cpp_out[i]);
       }
   }
   ```

---

## Function Dependency Graph

### Porting Order (Leaves → Root)

#### Layer 0: Constants & Types (no dependencies)
- `kJPEGNaturalOrder[64]` - zigzag order table
- `kDCTSize = 8`, `kDCTSize2 = 64`
- Quantization matrix constants (`kBaseQuantMatrixYCbCr`, etc.)
- Type definitions: `coeff_t`, `JCOEF`, `HuffmanTableEntry`

#### Layer 1: Pure Math Functions (no jpegli deps)
| Function | File | Inputs | Outputs | Notes |
|----------|------|--------|---------|-------|
| `BuildJpegHuffmanTable` | huffman.cc | counts[16], symbols[] | lut[] | Core algorithm |
| `CreateHuffmanTree` | huffman.cc | data[], length | depth[] | Tree construction |
| `GetQuantMatrix` | quant.cc | distance, colorspace, component | table[64] | Pure lookup |
| `LinearQualityToDistance` | quant.cc | quality (0-100) | distance (float) | Simple formula |

#### Layer 2: Pure Transforms (depend on Layer 0-1)
| Function | File | Inputs | Outputs | Notes |
|----------|------|--------|---------|-------|
| `IDCT1D<8>` | idct.cc | block[64] | block[64] | 1D inverse DCT |
| `DequantBlock` | idct.cc | qblock[64], dequant[64], bias[64] | block[64] | Per-coeff multiply |
| `InverseTransformBlock8x8` | idct.cc | qblock, dequant, bias | output[64] | Full 2D IDCT |
| `ComputeDCTBlock` | dct-inl.h | pixels[64], qmc[64] | coeffs[64] | Forward DCT + quant |
| `RGBToYCbCr` | color_transform.cc | row[3][], width | row[3][] (modified) | BT.601 matrix |
| `YCbCrToRGB` | color_transform.cc | row[3][], width | row[3][] (modified) | Inverse BT.601 |
| `ZigZagShuffle` | entropy_coding.cc | block[64] | block[64] | Reorder coefficients |

#### Layer 3: Stateless I/O (depend on Layer 0-2)
| Function | File | Inputs | Outputs | Notes |
|----------|------|--------|---------|-------|
| `DecodeDCTBlock` | decode_scan.cc | hufftables, bitstream | coeffs[64] | Huffman decode |
| `ReadBits` | decode_scan.cc | bitreader, nbits | value | Bit extraction |
| `ReadSymbol` | decode_scan.cc | hufftable, bitreader | symbol | Huffman lookup |

#### Layer 4: Stateful Components (depend on Layer 0-3)
| Component | Files | State | Notes |
|-----------|-------|-------|-------|
| `JpegBitWriter` | bit_writer.cc | buffer, pos, free_bits | Byte stuffing |
| `BitReaderState` | decode_scan.cc | pos, bit_buffer | Stream position |
| `TokenArray` | entropy_coding.cc | tokens[], size | Token storage |

#### Layer 5: Pipeline Stages (depend on Layer 0-4)
| Function | File | State Modified | Notes |
|----------|------|----------------|-------|
| `WriteiMCURow` | encode_streaming.cc | bitwriter, last_dc | Encode one MCU row |
| `ProcessScan` | decode_scan.cc | coeff_buffers | Decode one scan |
| `ProcessMarkers` | decode_marker.cc | cinfo state machine | Marker dispatch |

#### Layer 6: Public API (depend on all)
| Function | File | Notes |
|----------|------|-------|
| `jpegli_start_compress` | encode.cc | Initialize encoder |
| `jpegli_write_scanlines` | encode.cc | Feed image data |
| `jpegli_finish_compress` | encode.cc | Finalize JPEG |
| `jpegli_read_header` | decode.cc | Parse headers |
| `jpegli_start_decompress` | decode.cc | Initialize decoder |
| `jpegli_read_scanlines` | decode.cc | Extract pixels |

---

## Pure vs Stateful Classification

### PURE Functions (safe to test in isolation)
- All Layer 1-2 functions
- Can be called with arbitrary inputs
- Output depends only on inputs
- **FFI testing**: Direct comparison

### STATEFUL Functions (require state setup)
- Layer 4-6 functions
- Maintain internal buffers/positions
- **FFI testing**: Need to serialize/compare state, or test at boundaries

### Hybrid Approach for Stateful
For stateful components, test at "checkpoint" boundaries:
```
[Input Image Bytes]
    ↓
    ══════════════ Checkpoint 1: After color transform ══════════════
    ↓
    ══════════════ Checkpoint 2: After DCT + Quantization ══════════════
    ↓
    ══════════════ Checkpoint 3: After Huffman encoding ══════════════
    ↓
[Output JPEG Bytes]
```

Each checkpoint: extract intermediate data, compare Rust vs C++.

---

## Key Source Files Reference

### Core Jpegli (lib/jpegli/)
| File | Lines | Purpose |
|------|-------|---------|
| encode.cc | ~1200 | Main encoder, public API |
| decode.cc | ~1000 | Main decoder, public API |
| encode_streaming.cc | ~600 | Streaming encode path |
| decode_scan.cc | ~500 | Entropy decoding |
| decode_marker.cc | ~800 | Marker parsing |
| quant.cc | ~650 | Quantization tables |
| idct.cc | ~700 | Inverse DCT |
| dct-inl.h | ~400 | Forward DCT (Highway SIMD) |
| color_transform.cc | ~450 | Color space conversion |
| huffman.cc | ~200 | Huffman table construction |
| entropy_coding.cc | ~300 | Token/coefficient handling |
| bit_writer.cc | ~150 | Bitstream output |

### Dependencies
- `third_party/highway/` - SIMD abstraction → Rust: use `wide` or `std::simd`
- `third_party/lcms/` - Color management (optional for core)
- `lib/base/` - Utilities (mostly not needed)

---

## Tolerance Guidelines

| Data Type | Tolerance | Rationale |
|-----------|-----------|-----------|
| f32 intermediate | 1e-5 | FP rounding differences |
| f32 final pixels | 1e-4 | Accumulated error |
| Integer coefficients | exact | Must match exactly |
| Huffman codes | exact | Must match exactly |
| Final JPEG bytes | exact | Bitstream must be identical |

Note: If SIMD paths differ (e.g., Rust uses different SIMD than Highway),
tolerances may need adjustment for intermediate f32 values.

---

## C++ Dependencies → Rust Equivalents

### 1. Highway (SIMD) - CRITICAL
**C++**: `hwy/highway.h`, `hwy/foreach_target.h`, `hwy/aligned_allocator.h`

Used heavily for:
- DCT/IDCT transforms
- Color space conversion
- Downsampling/upsampling
- Adaptive quantization
- Entropy coding

**Rust options**:
| Option | Pros | Cons |
|--------|------|------|
| `std::simd` (nightly) | Official, portable | Nightly-only, API unstable |
| `wide` crate | Stable, portable | Less optimized than Highway |
| `pulp` crate | Good perf, ergonomic | Less mature |
| `simdeez` crate | Multi-arch dispatch | Complex API |
| Manual scalar first | Simple, debuggable | Slow, add SIMD later |

**Recommendation**: Start with scalar Rust for correctness verification via FFI, then add `wide` or `std::simd` for performance.

### 2. C++ Standard Library
| C++ | Rust Equivalent |
|-----|-----------------|
| `<cstdint>` (uint8_t, int32_t, etc.) | Built-in: `u8`, `i32`, etc. |
| `<cstddef>` (size_t) | Built-in: `usize` |
| `<cmath>` (sqrt, floor, etc.) | `f32::sqrt()`, `f32::floor()`, etc. |
| `<algorithm>` (min, max, sort) | `std::cmp::{min,max}`, `slice::sort` |
| `<cstring>` (memcpy, memset) | `slice::copy_from_slice`, `slice::fill` |
| `<vector>` | `Vec<T>` |
| `<limits>` | `i32::MAX`, `f32::INFINITY`, etc. |
| `<initializer_list>` | Array literals `[a, b, c]` |
| `<unordered_map>` | `std::collections::HashMap` |

### 3. lib/base Utilities
| C++ (lib/base/) | Rust Equivalent |
|-----------------|-----------------|
| `byte_order.h` (endianness) | `u32::from_le_bytes()`, `to_be_bytes()` |
| `bits.h` (clz, ctz, popcount) | `u32::leading_zeros()`, `trailing_zeros()`, `count_ones()` |
| `compiler_specific.h` (JXL_INLINE) | `#[inline]`, `#[inline(always)]` |
| `status.h` (error handling) | `Result<T, E>` |
| `types.h` (type aliases) | Type aliases or newtypes |
| `span.h` (array view) | `&[T]` / `&mut [T]` |

### 4. libjpeg API Structures
**C++**: `jpeglib.h` - defines `jpeg_compress_struct`, `jpeg_decompress_struct`, etc.

**Rust**: Define equivalent structs. For FFI testing, use `#[repr(C)]` to match C layout.

```rust
#[repr(C)]
pub struct JpegCompressStruct {
    // ... fields matching jpeglib.h
}
```

### 5. Aligned Memory Allocation
**C++**: `hwy::AlignedFreeUniquePtr`, `hwy::AllocateAligned`

**Rust options**:
- `std::alloc::Layout::from_size_align()` + `alloc()`
- `aligned_vec` crate
- Custom allocator with `Vec`

```rust
// Simple aligned allocation
fn alloc_aligned<T>(count: usize, align: usize) -> Box<[T]> {
    let layout = std::alloc::Layout::from_size_align(
        count * std::mem::size_of::<T>(),
        align
    ).unwrap();
    // ...
}
```

### 6. Color Management & XYB - ARCHITECTURE NOTE

**Core jpegli** (`lib/jpegli/`) does NOT use CMS:
- Stores ICC profiles as raw bytes (APP2 markers)
- YCbCr ↔ RGB uses fixed BT.601 matrix math (in `color_transform.cc`)
- XYB mode: uses special quant tables, expects pre-converted XYB input

**XYB Transform** (`lib/extras/xyb_transform.cc`) - **NO CMS NEEDED!**
- Pure math with fixed constants from `lib/cms/opsin_params.h`
- RGB → XYB is just: matrix multiply + cube root + arithmetic
- Constants derived from human vision research, not ICC profiles

```
XYB Transform (pure math):
1. Linear RGB × 3x3 matrix + bias → mixed[3]
2. mixed[i] = cbrt(mixed[i])
3. X = 0.5 × (mixed[0] - mixed[1])
4. Y = 0.5 × (mixed[0] + mixed[1])
5. B = mixed[2]
```

**Files to port for XYB support:**
- `lib/cms/opsin_params.h` - ~50 lines of constants
- `lib/extras/xyb_transform.cc` - ~100 lines of math

**Tools** (`cjpegli`/`djpegli`) use `jxl_cms` for ICC profiles:
- Wraps skcms (Google) or lcms2
- Used for wide-gamut / ICC profile transforms (NOT for XYB)

**For Rust port:**
| Feature | CMS Needed? | Notes |
|---------|-------------|-------|
| Core codec | No | Just store ICC bytes |
| XYB mode | No | Pure math, port the constants |
| Wide-gamut ICC | Yes | Only if full tool parity needed |

**Rust CMS options (only if full ICC support needed):**
| Crate | Notes |
|-------|-------|
| `lcms2` | Bindings to lcms2 C library |
| `little-cms` | Alternative lcms2 bindings |
| Pure Rust | No mature options yet |

### 7. NOT Needed for Core Port
| Dependency | Reason |
|------------|--------|
| GoogleTest (`gtest`) | Use Rust's built-in `#[test]` |
| skcms/lcms2 | Only needed for tool-level ICC transforms |
| libpng, giflib, etc. | Format support, not core JPEG |
| Threading (`pthread`) | Can add later with `rayon` |

---

## Suggested Rust Crate Dependencies

```toml
[dependencies]
# For FFI with C++
libc = "0.2"

# For SIMD (choose one, or start scalar)
# wide = "0.7"           # Stable, portable SIMD
# pulp = "0.18"          # Alternative SIMD

# For aligned allocation
# aligned-vec = "0.5"    # If needed

[dev-dependencies]
# For property-based testing
proptest = "1.0"
# For FFI function loading
libloading = "0.8"
```

---

## FFI Type Mapping

| C++ Type | Rust FFI Type |
|----------|---------------|
| `int` | `c_int` (from `libc`) |
| `size_t` | `usize` |
| `uint8_t*` | `*const u8` / `*mut u8` |
| `float*` | `*const f32` / `*mut f32` |
| `int16_t*` | `*const i16` / `*mut i16` |
| `int32_t*` | `*const i32` / `*mut i32` |
| `void*` | `*mut c_void` |
| `const char*` | `*const c_char` |
