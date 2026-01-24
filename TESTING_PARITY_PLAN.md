# Testing Parity Plan: Rust jpegli vs C++ jpegli

This document outlines the plan for achieving testing rigor parity between the Rust port and the C++ original.

## Current State

### C++ Testing Infrastructure
- **126+ unit tests** across 16 test files
- **44 fuzzer test cases** from OSS-Fuzz
- **285 test data files** in structured corpus
- **Sanitizer integration** (ASan, MSan, UBSan)
- **Quality metrics** (Butteraugli, RMS distance)
- **Parametrized testing** for configuration combinations
- **60+ error handling scenarios**

### Rust Testing Infrastructure (Current)
- **119 unit tests** in lib
- **16 progressive encoding tests**
- **Basic integration tests** (roundtrip, decode external, parity)
- **Quality metrics** (DSSIM, SSIMULACRA2)
- **No fuzzing infrastructure**
- **Limited error handling tests**

---

## Phase 1: Test Infrastructure Foundation

### 1.1 Test Utilities Module
**Priority: HIGH | Effort: Medium**

Create `jpegli/src/test_utils.rs` with:

```rust
// Image generation (match C++ GeneratePixels, GenerateRawData)
pub fn generate_test_image(width: u32, height: u32, pattern: TestPattern) -> TestImage;
pub fn generate_gradient(width: u32, height: u32) -> Vec<u8>;
pub fn generate_checkerboard(width: u32, height: u32, block_size: u32) -> Vec<u8>;
pub fn generate_noise(width: u32, height: u32, seed: u64) -> Vec<u8>;

// Quality verification (match C++ DistanceRms, VerifyOutputImage)
pub fn distance_rms(original: &[u8], decoded: &[u8], width: u32, height: u32) -> f64;
pub fn max_pixel_diff(original: &[u8], decoded: &[u8]) -> u8;
pub fn verify_output(original: &[u8], decoded: &[u8], max_rms: f64, max_diff: u8);

// Test data access
pub fn get_test_data_path(filename: &str) -> PathBuf;
pub fn read_test_data(filename: &str) -> Result<Vec<u8>>;
```

**C++ Reference**: `lib/jpegli/test_utils.h` (30KB)

### 1.2 Test Image Corpus
**Priority: HIGH | Effort: Low**

Create symlinks or copy essential test images:

```
jpegli/testdata/
├── flower_small.png          # Primary test image
├── flower_small_gray.png     # Grayscale variant
├── gradient_64x64.png        # Simple gradient
├── checkerboard_64x64.png    # High frequency
├── hdr_room.png              # 16-bit HDR (if supported)
└── corpus/                   # Fuzzer corpus (Phase 3)
```

### 1.3 Parametrized Test Framework
**Priority: MEDIUM | Effort: Low**

Use `proptest` or custom macros for configuration matrix testing:

```rust
#[test_case(Quality::from_quality(50.0), PixelFormat::Rgb, JpegMode::Baseline)]
#[test_case(Quality::from_quality(85.0), PixelFormat::Rgb, JpegMode::Progressive)]
#[test_case(Quality::from_quality(95.0), PixelFormat::Gray, JpegMode::Baseline)]
fn test_encode_decode_roundtrip(quality: Quality, format: PixelFormat, mode: JpegMode) {
    // ...
}
```

---

## Phase 2: Core Functionality Tests

### 2.1 Encode API Tests
**Priority: HIGH | Effort: Medium**

Create `jpegli/tests/encode_api.rs`:

| Test | C++ Equivalent | Status |
|------|----------------|--------|
| `test_encode_basic` | `EncodeAPITestP` | Partial |
| `test_encode_context_reuse` | `ReuseCinfo*` tests | Missing |
| `test_encode_quality_levels` | Quality parameter tests | Partial |
| `test_encode_subsampling` | Subsampling tests | Missing (4:4:4 only) |
| `test_encode_progressive_levels` | Progressive scan scripts | Partial |
| `test_encode_huffman_optimization` | Optimized tables | Done |
| `test_encode_restart_markers` | DRI marker tests | Missing |
| `test_encode_abbreviated_streams` | Abbreviated JPEG | Missing |

### 2.2 Decode API Tests
**Priority: HIGH | Effort: Medium**

Create `jpegli/tests/decode_api.rs`:

| Test | C++ Equivalent | Status |
|------|----------------|--------|
| `test_decode_basic` | `DecodeAPITestP` | Partial |
| `test_decode_context_reuse` | `ReuseCinfo*` tests | Missing |
| `test_decode_progressive` | Progressive decode | Partial |
| `test_decode_truncated` | Truncated input | Missing |
| `test_decode_grayscale` | Grayscale decode | Done |
| `test_decode_cmyk` | CMYK decode | Missing |
| `test_decode_various_subsampling` | 4:2:0, 4:2:2, etc. | Missing |

### 2.3 Roundtrip Quality Tests
**Priority: HIGH | Effort: Medium**

Expand `jpegli/tests/roundtrip_quality.rs`:

| Quality Level | Max RMS (C++) | Max RMS (Rust) | Status |
|---------------|---------------|----------------|--------|
| Q50 | 20.0 | TBD | Missing |
| Q75 | 10.0 | TBD | Missing |
| Q85 | 5.0 | TBD | Partial |
| Q90 | 3.0 | TBD | Partial |
| Q95 | 2.1 | TBD | Missing |

**Thresholds from C++**: `encode_api_test.cc` lines 200-250

---

## Phase 3: Error Handling Tests

### 3.1 Encoder Error Tests
**Priority: MEDIUM | Effort: High**

Create `jpegli/tests/error_handling.rs`:

**Initialization Errors** (C++ has 6 tests):
- [ ] No destination set
- [ ] Zero dimensions
- [ ] Invalid dimensions (>65535)
- [ ] No components defined
- [ ] Invalid component count

**Quantization Errors** (C++ has 4 tests):
- [ ] Invalid quant table index
- [ ] Missing quant table
- [ ] Invalid quant values

**Component Mismatch Errors** (C++ has 6 tests):
- [ ] Duplicate component IDs
- [ ] Invalid sampling ratios
- [ ] Component count mismatch

**Scan Script Errors** (C++ has 13 tests):
- [ ] Invalid scan parameters
- [ ] Progressive scan validation
- [ ] Spectral selection bounds

### 3.2 Decoder Error Tests
**Priority: MEDIUM | Effort: Medium**

**Marker Validation** (C++ has 10+ tests):
- [ ] Invalid SOI marker
- [ ] Missing EOI marker
- [ ] Corrupted DQT marker
- [ ] Corrupted DHT marker
- [ ] Invalid SOF parameters
- [ ] Invalid SOS parameters

**Corruption Robustness**:
- [ ] Single-byte mutation testing
- [ ] Truncated file handling
- [ ] Marker injection attacks

---

## Phase 4: Fuzzing Infrastructure

### 4.1 Cargo-Fuzz Setup
**Priority: HIGH | Effort: Medium**

```bash
cargo install cargo-fuzz
cd jpegli && cargo fuzz init
```

Create fuzz targets:

```
jpegli/fuzz/
├── Cargo.toml
└── fuzz_targets/
    ├── decode_jpeg.rs      # Main decoder fuzzer
    ├── encode_roundtrip.rs # Encode-decode roundtrip
    └── progressive.rs      # Progressive-specific fuzzer
```

### 4.2 Fuzz Target: Decoder
**Priority: HIGH | Effort: Low**

```rust
// fuzz_targets/decode_jpeg.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use zenjpeg::Decoder;

fuzz_target!(|data: &[u8]| {
    let decoder = Decoder::new();
    let _ = decoder.decode(data);
});
```

### 4.3 Fuzz Target: Roundtrip
**Priority: MEDIUM | Effort: Low**

```rust
// fuzz_targets/encode_roundtrip.rs
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    width: u8,      // 1-255
    height: u8,     // 1-255
    quality: u8,    // 1-100
    data: Vec<u8>,
}

fuzz_target!(|input: FuzzInput| {
    // Encode then decode, verify no panics
});
```

### 4.4 OSS-Fuzz Corpus Import
**Priority: MEDIUM | Effort: Low**

Copy C++ fuzzer regression corpus:
```bash
cp -r ../testdata/oss-fuzz jpegli/fuzz/corpus/
```

---

## Phase 5: Quality Metrics Parity

### 5.1 Butteraugli Integration
**Priority: MEDIUM | Effort: High**

Options:
1. **Complete Rust port** of `lib/extras/butteraugli.cc` (skeleton exists)
2. **FFI wrapper** around C++ butteraugli
3. **Use existing crate** if one exists with compatible API

**Target Thresholds** (from C++ `jpegli_test.cc`):

| Test Case | Butteraugli Threshold |
|-----------|----------------------|
| XYB Encoding | < 1.84 |
| Large Smooth Area | < 3.0 |
| YUV Encoding | < 1.85 |
| YUV + Chroma Subsampling | ≤ 1.82 |
| YUV - No AQ | < 2.2 |
| HDR Roundtrip | < 1.5 |

### 5.2 RMS Distance Verification
**Priority: HIGH | Effort: Low**

Already have DSSIM. Add RMS for C++ parity:

```rust
pub fn rms_distance(a: &[u8], b: &[u8]) -> f64 {
    let sum_sq: f64 = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum();
    (sum_sq / a.len() as f64).sqrt()
}
```

---

## Phase 6: C++ Comparison Tests

### 6.1 Coefficient-Level Comparison
**Priority: HIGH | Effort: Medium**

Expand `tests/huffman_cpp_comparison.rs` pattern to:

| Component | C++ Testdata | Rust Test | Status |
|-----------|--------------|-----------|--------|
| Huffman Tree | `CreateHuffmanTree.testdata` | Done | ✅ |
| Quant Tables | `SetQuantMatrices.testdata` | Partial | 🟡 |
| Adaptive Quant | `ComputeAdaptiveQuantField.testdata` | Missing | ❌ |
| DCT Coefficients | TBD | Missing | ❌ |
| Entropy Coded Data | TBD | Missing | ❌ |

### 6.2 File Size Parity Tests
**Priority: HIGH | Effort: Low**

Expand `tests/parity_enforcement.rs`:

```rust
#[test]
fn test_file_size_parity_q85() {
    // Encode same image with C++ and Rust
    // Assert file sizes within 2%
}

#[test]
fn test_file_size_parity_progressive() {
    // Progressive mode comparison
}
```

### 6.3 Byte-Level Bitstream Comparison
**Priority: LOW | Effort: High**

For debugging coefficient/entropy differences:

```rust
pub fn compare_jpeg_segments(cpp_jpeg: &[u8], rust_jpeg: &[u8]) -> SegmentDiff {
    // Parse and compare:
    // - DQT segments
    // - DHT segments
    // - SOF parameters
    // - SOS scan data (per-scan comparison)
}
```

---

## Phase 7: Benchmarking

### 7.1 Criterion Benchmarks
**Priority: MEDIUM | Effort: Low**

Expand `benches/encode.rs` and `benches/decode.rs`:

```rust
fn bench_encode_q85(c: &mut Criterion) {
    let image = load_test_image("flower_small.png");
    c.bench_function("encode_q85_baseline", |b| {
        b.iter(|| encode_jpeg(&image, 85, JpegMode::Baseline))
    });
    c.bench_function("encode_q85_progressive", |b| {
        b.iter(|| encode_jpeg(&image, 85, JpegMode::Progressive))
    });
}
```

### 7.2 Throughput Comparison
**Priority: LOW | Effort: Medium**

Compare against:
- C++ jpegli (via FFI)
- mozjpeg
- libjpeg-turbo

Output format matching C++ `benchmark_xl`:
```
Codec      | BPP  | E MP/s | D MP/s | SSIM2
jpegli-rs  | 1.23 | 45.2   | 89.1   | 87.4
mozjpeg    | 1.25 | 12.3   | 85.2   | 87.1
```

---

## Phase 8: CI/CD Integration

### 8.1 GitHub Actions Workflow
**Priority: HIGH | Effort: Low**

Create `.github/workflows/test.yml`:

```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        rust: [stable, nightly]
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: cargo test --all-features
      - name: Run ignored tests
        run: cargo test -- --ignored

  miri:
    runs-on: ubuntu-latest
    steps:
      - name: Run Miri (UB detection)
        run: cargo +nightly miri test

  fuzz:
    runs-on: ubuntu-latest
    steps:
      - name: Run fuzzer (limited)
        run: cargo fuzz run decode_jpeg -- -max_total_time=60
```

### 8.2 Sanitizer Testing
**Priority: MEDIUM | Effort: Low**

```yaml
  sanitizers:
    runs-on: ubuntu-latest
    env:
      RUSTFLAGS: "-Z sanitizer=address"
    steps:
      - run: cargo +nightly test --target x86_64-unknown-linux-gnu
```

---

## Implementation Priority

### Immediate (Week 1-2)
1. [ ] Test utilities module (`test_utils.rs`)
2. [ ] Test image corpus setup
3. [ ] Cargo-fuzz basic setup
4. [ ] Decoder fuzz target

### Short-term (Week 3-4)
5. [ ] Error handling tests (encoder)
6. [ ] Error handling tests (decoder)
7. [ ] RMS distance verification
8. [ ] File size parity tests

### Medium-term (Month 2)
9. [ ] Parametrized test framework
10. [ ] Encode API comprehensive tests
11. [ ] Decode API comprehensive tests
12. [ ] GitHub Actions CI

### Long-term (Month 3+)
13. [ ] Butteraugli integration
14. [ ] Byte-level bitstream comparison
15. [ ] Full benchmark suite
16. [ ] OSS-Fuzz integration

---

## Success Metrics

| Metric | C++ Baseline | Rust Target | Current |
|--------|--------------|-------------|---------|
| Unit Tests | 126 | 150+ | 119 |
| Integration Tests | 20+ | 20+ | ~10 |
| Error Handling Tests | 60+ | 60+ | ~5 |
| Fuzz Targets | 2 | 3 | 0 |
| Fuzz Corpus Size | 44 | 44+ | 0 |
| Quality Threshold Tests | 10+ | 10+ | ~3 |
| C++ Comparison Tests | N/A | 10+ | 2 |
| Code Coverage | ~80% | 80%+ | Unknown |

---

## References

- C++ test files: `lib/jpegli/*_test.cc`
- C++ test utils: `lib/jpegli/test_utils.h`
- C++ fuzzing: `tools/djxl_fuzzer.cc`, `tools/jpegli_dec_fuzzer.cc`
- Rust fuzzing guide: https://rust-fuzz.github.io/book/
- Miri (UB detection): https://github.com/rust-lang/miri
