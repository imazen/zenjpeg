# Adaptive Quantization (AQ) - Porting Guide

This document traces everything about adaptive quantization in jpegli, covering the C++ implementation, instrumentation for testing, porting status, and known difficulties.

## Table of Contents

1. [Overview](#overview)
2. [C++ Pipeline Architecture](#c-pipeline-architecture)
3. [Function-by-Function Analysis](#function-by-function-analysis)
4. [Instrumentation and Test Data](#instrumentation-and-test-data)
5. [Current Rust Implementation](#current-rust-implementation)
6. [Porting Strategy](#porting-strategy)
7. [Known Difficulties](#known-difficulties)
8. [Testing Approaches](#testing-approaches)

---

## Overview

### What AQ Does

Adaptive Quantization adjusts the quantization strength on a per-block basis based on image content. The goal is to:
- Use **more quantization** (smaller values, more compression) in flat/smooth areas where loss is less visible
- Use **less quantization** (preserve detail) in textured/edge areas where loss is more visible

### Key Insight: aq_strength Range

**Critical finding from C++ testdata analysis:**

C++ produces `aq_strength` values in the **0.0-0.2 range with mean ~0.08**, NOT 0-1 or 0-2.

```
y_quant=3.0: min=0.0000, max=0.1955, mean=0.0810
y_quant=3.0: min=0.0000, max=0.1964, mean=0.0812
y_quant=3.0: min=0.0027, max=0.1992, mean=0.0847
```

This is used in the zero-bias formula:
```
threshold = zero_bias_offset[k] + zero_bias_mul[k] * aq_strength
```

With `aq_strength ≈ 0.08` and `zero_bias_mul ≈ 0.5`:
```
threshold ≈ offset + 0.04  (small increase)
```

Using `aq_strength = 1.0` was **12.5x too aggressive**.

### Data Flow

```
Input Image (Y plane, float, 0-255 range)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ComputePreErosion()                                 │
│ - Subsamples 4x4 → computes local differences       │
│ - Applies gamma correction                          │
│ - Output: pre_erosion buffer (1/4 resolution)       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ FuzzyErosion()                                      │
│ - Morphological min-filter with weighted blend     │
│ - Subsamples 2x → 1/8 resolution (block level)     │
│ - Output: initial quant_field                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ PerBlockModulations()                               │
│ - ComputeMask() - base perceptual masking          │
│ - HfModulation() - high frequency content          │
│ - GammaModulation() - gamma-aware adjustment       │
│ - FastPow2f() - convert exponent to multiplier     │
│ - Output: modulated quant_field                     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Final Transformation                                │
│ aq_strength = max(0.0, (0.6 / quant_field) - 1.0)  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Per-block aq_strength values (0.0-0.2 typical range)
```

---

## C++ Pipeline Architecture

### File Location
`lib/jpegli/adaptive_quantization.cc`

### Entry Point
```cpp
void ComputeAdaptiveQuantField(j_compress_ptr cinfo)
```

Called once per MCU row during streaming encoding.

### Key Data Structures

#### RowBuffer<float>
A 2D buffer with:
- `xsize()` - width
- `Row(y)` - pointer to row y
- `stride()` - bytes between rows
- Supports negative row indices for border handling

#### quant_field
Per-block float buffer storing AQ strength values.

### Important Constants

```cpp
constexpr float kInputScaling = 1.0f / 255.0f;  // Normalize 0-255 to 0-1

// PerBlockModulations constants
static const float kAcQuant = 0.841f;
float base_level = 0.48f * kAcQuant;  // = 0.40368
float kDampenRampStart = 9.0f;
float kDampenRampEnd = 65.0f;

// ComputeMask constants
const auto kBase = Set(d, -0.74174993f);
const auto kMul4 = Set(d, 3.2353257320940401f);
const auto kMul2 = Set(d, 12.906028311180409f);
const auto kOffset2 = Set(d, 305.04035728311436f);
const auto kMul3 = Set(d, 5.0220313103171232f);
const auto kOffset3 = Set(d, 2.1925739705298404f);
const auto kMul0 = Set(d, 0.74760422233706747f);

// HfModulation constants
static const float kSumCoeff = -2.0052193233688884f * kInputScaling / 112.0;

// GammaModulation constants
static const float kBias = 0.16f / kInputScaling;
static const float kScale = kInputScaling / 64.0f;
const auto kGamma = Set(d, -0.15526878023684174f * kInvLog2e);
```

---

## Function-by-Function Analysis

### 1. ComputePreErosion

**Location:** `adaptive_quantization.cc:506`

**Purpose:** Compute local pixel differences with gamma correction, subsample 4x.

**Algorithm:**
1. For each pixel, compute base = average of 4 neighbors (L, R, T, B)
2. Apply gamma correction: `RatioOfDerivativesOfCubicRootToSimpleGamma`
3. Compute diff = gamma_corrected * (pixel - base)
4. diff = min(diff², 0.2)  // limit
5. diff = MaskingSqrt(diff)  // sqrt with offset
6. Accumulate every 4 rows, subsample 4x horizontally

**Pure Function:** NO - uses RowBuffer with stride/borders

**Porting Difficulty:** MEDIUM
- Complex gamma function `RatioOfDerivativesOfCubicRootToSimpleGamma`
- SIMD-optimized
- Border handling

**Testdata Available:** YES - `ComputePreErosion.testdata` (134 test cases)

### 2. FuzzyErosion

**Location:** `adaptive_quantization.cc:430`

**Purpose:** Morphological min-filter with weighted blend, subsample 2x to block level.

**Algorithm:**
1. For each 3x3 neighborhood, sort to find 4 smallest values
2. Weighted sum: 0.125*min0 + 0.075*min1 + 0.06*min2 + 0.05*min3
3. Subsample 2x in each dimension (4 values → 1 block)

**Pure Function:** PARTIAL - needs border handling but core is pure math

**Porting Difficulty:** MEDIUM
- Sort4/UpdateMin4 are SIMD-specific
- Can be implemented with scalar fallback

**Testdata Available:** YES - `FuzzyErosion.testdata` (134 test cases)

### 3. PerBlockModulations

**Location:** `adaptive_quantization.cc:326`

**Purpose:** Apply perceptual modulations to each block.

**Algorithm:**
```cpp
for each block (bx, by):
    out_val = quant_field[by][bx]  // from FuzzyErosion
    out_val = ComputeMask(out_val)  // base masking
    out_val = HfModulation(x, y, input, out_val)  // HF content
    out_val = GammaModulation(x, y, input, out_val)  // gamma
    quant_field[by][bx] = FastPow2f(out_val * 1.442695041) * mul + add
```

Where:
- `mul = kAcQuant * dampen`
- `add = (1 - dampen) * base_level`
- `dampen` decreases from 1.0 to 0.0 as quality decreases

**Pure Function:** NO - accesses input buffer with stride

**Porting Difficulty:** HARD
- Three sub-functions each with complex math
- FastPow2f is approximation (not std::exp2)
- SIMD throughout

**Testdata Available:** YES - `PerBlockModulations.testdata` (134 test cases)

### 4. ComputeMask

**Location:** `adaptive_quantization.cc:171`

**Purpose:** Base perceptual masking curve.

**Algorithm:**
```cpp
v1 = max(out_val * kMul0, 1e-3)
v2 = 1 / (v1 + kOffset2)
v3 = 1 / (v1² + kOffset3)
v4 = 1 / (v1² + kOffset4)
return kBase + kMul4*v4 + kMul2*v2 + kMul3*v3
```

**Pure Function:** YES

**Porting Difficulty:** EASY - pure scalar math

**Testdata Available:** No (inlined in PerBlockModulations)

### 5. HfModulation

**Location:** `adaptive_quantization.cc:295`

**Purpose:** Measure high-frequency content via gradients.

**Algorithm:**
1. For each pixel in 8x8 block:
   - sum += |pixel - pixel_right| (masked for rightmost)
   - sum += |pixel - pixel_below|
2. return out_val + sum * kSumCoeff

**Pure Function:** NO - needs input buffer access

**Porting Difficulty:** MEDIUM
- Simple gradient computation
- Border handling needed

**Testdata Available:** No (inlined in PerBlockModulations)

### 6. GammaModulation

**Location:** `adaptive_quantization.cc:268`

**Purpose:** Adjust for gamma perception.

**Algorithm:**
1. For each pixel in 8x8 block:
   - ratio = RatioOfDerivativesOfCubicRootToSimpleGamma(pixel + bias)
   - overall_ratio += ratio
2. overall_ratio *= scale
3. return out_val + kGamma * FastLog2f(overall_ratio)

**Pure Function:** NO - needs input buffer access

**Porting Difficulty:** HARD
- Complex gamma function
- FastLog2f approximation

**Testdata Available:** No (inlined in PerBlockModulations)

### 7. RatioOfDerivativesOfCubicRootToSimpleGamma

**Location:** `adaptive_quantization.cc:207`

**Purpose:** Compute ratio of derivatives for gamma correction.

**Algorithm:**
```cpp
v = max(v, 0)  // ZeroIfNegative
v2 = v * v
num = kNumMul * v2 + kNumOffset
den = kDenMul * v * v2 + kVOffset
return invert ? num/den : den/num
```

**Pure Function:** YES

**Porting Difficulty:** EASY - pure scalar math

**Testdata Available:** No (can fuzz test independently)

### 8. FastLog2f / FastPow2f

**Location:** `adaptive_quantization.cc:122, 146`

**Purpose:** Fast approximations of log2 and pow2.

**Pure Function:** YES

**Porting Difficulty:** MEDIUM
- Bit manipulation for float representation
- Polynomial approximations

**Testdata Available:** No (can fuzz test against std::log2/exp2)

---

## Instrumentation and Test Data

### Available Testdata Files

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `ComputeAdaptiveQuantField.testdata` | 11MB | 134 | Full AQ pipeline |
| `ComputePreErosion.testdata` | 11MB | 134 | Pre-erosion stage |
| `FuzzyErosion.testdata` | 1.3MB | 134 | Erosion filter |
| `PerBlockModulations.testdata` | 9MB | 134 | Modulation stage |

### Testdata Format (JSON Lines)

Each line is a complete JSON object:

```json
{
  "test_type": "ComputeAdaptiveQuantFieldTest",
  "config_y_quant_01": 3.0,
  "config_next_iMCU_row": 0,
  "config_total_iMCU_rows": 67,
  "config_max_v_samp_factor": 1,
  "config_y_comp_width_in_blocks": 64,
  "input_buffer_y_slice": {
    "component_index": 0,
    "start_row": -1,
    "num_rows": 14,
    "start_col": -1,
    "num_cols": 514,
    "stride": 576,
    "data": [[1.308549957e+02, ...], ...]
  },
  "expected_quant_field_slice": {
    "data": [[0.1215903759, 0.0636703968, ...], ...]
  }
}
```

### Generating New Testdata

```bash
cd /home/lilith/work/jpegli
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_TOOLS=ON ..
ninja cjpegli

# Generate testdata
GENERATE_RUST_TEST_DATA=1 ./tools/cjpegli /path/to/input.png /tmp/output.jpg

# Testdata written to working directory
ls *.testdata
```

### Parsing Testdata in Rust

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct RowBufferSlice {
    component_index: i32,
    start_row: i64,
    num_rows: usize,
    start_col: i64,
    num_cols: usize,
    stride: usize,
    data: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct ComputeAQTest {
    config_y_quant_01: f32,
    input_buffer_y_slice: RowBufferSlice,
    expected_quant_field_slice: RowBufferSlice,
}

fn load_testdata(path: &str) -> Vec<ComputeAQTest> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| {
            // Parse nested JSON carefully
            serde_json::from_str(line).ok()
        })
        .collect()
}
```

---

## Previous Failed Port

### Location
`/home/lilith/work/jpeg-encoder/src/jpegli/adaptive_quantization.rs`

### Status: FAILED - Never Successfully Troubleshooted

A previous attempt was made to port the C++ AQ to Rust in a separate project. This port:

**What it attempted:**
- Ported `compute_pre_erosion_scalar` - downsampling with gamma correction
- Ported `fuzzy_erosion_scalar` - morphological min-filter
- Ported `per_block_modulations_scalar` - HF/gamma modulations
- Ported `compute_mask_scalar` - perceptual masking curve
- Ported `ratio_of_derivatives` - gamma function

**Why it failed:**
1. **Wrong algorithm structure** - Applied modulations in wrong order
2. **Incorrect constants** - Some K_* values didn't match C++
3. **Missing FastPow2f** - Used `exp()` instead of C++ approximation
4. **Wrong coordinate mapping** - Pre-erosion to block level mapping errors
5. **Output range mismatch** - Produced values in wrong range
6. **No verification** - No testdata comparison to detect errors

**Lessons learned:**
- Must use testdata for verification at EVERY stage
- Must match C++ algorithm structure exactly
- Must use same approximations (FastLog2f, FastPow2f)
- Pure functions should be tested independently first

**Key code snippets that were wrong:**
```rust
// WRONG: Used center pixel instead of full block
let hf_modulated_val = hf_modulation_scalar(
    x_start + 1, y_start + 1,  // Should iterate all 8x8
    input_scaled, width, height,
    current_val
);

// WRONG: Mask applied after modulations instead of before
let mask_val = compute_mask_scalar(gamma_modulated_val);

// WRONG: Used ln instead of FastLog2f
let modulation = K_GAMMA_MOD_GAMMA * log_arg.ln();
```

---

## Current Rust Implementation

### File Location
`jpegli/src/adaptive_quant.rs`

### Current Status: SIMPLIFIED - NOT PER-BLOCK

The current Rust implementation uses a **completely different simplified algorithm**:

| Aspect | C++ | Rust |
|--------|-----|------|
| Activity measure | Gamma-corrected differences | Simple variance |
| Masking | Complex perceptual curve | Threshold-based |
| Modulations | HF + Gamma + FastPow2f | Edge detection |
| Output range | 0.0-0.2 (aq_strength) | 0.5-2.0 (multiplier) |

### Current Workaround

`encode.rs` uses a **constant aq_strength = 0.08** (mean from C++ testdata) instead of per-block values:

```rust
// Apply zero-biasing with aq_strength calibrated from C++ testdata.
// C++ AQ typically produces aq_strength values in the 0.0-0.2 range with mean ~0.08.
let aq_strength = 0.08f32;
```

This works but doesn't provide per-block adaptation.

---

## Porting Strategy

### Phase 1: Port Pure Functions (FFI/Fuzz Testable)

These can be tested independently:

1. **ComputeMask** - Pure scalar math
   ```rust
   fn compute_mask(out_val: f32) -> f32 {
       const K_BASE: f32 = -0.74174993;
       const K_MUL0: f32 = 0.74760422233706747;
       // ... rest of constants

       let v1 = (out_val * K_MUL0).max(1e-3);
       let v2 = 1.0 / (v1 + K_OFFSET2);
       // ...
   }
   ```

2. **RatioOfDerivativesOfCubicRootToSimpleGamma** - Pure scalar
   ```rust
   fn ratio_gamma<const INVERT: bool>(v: f32) -> f32 {
       let v = v.max(0.0);
       let v2 = v * v;
       let num = K_NUM_MUL * v2 + K_NUM_OFFSET;
       let den = K_DEN_MUL * v * v2 + K_V_OFFSET;
       if INVERT { num / den } else { den / num }
   }
   ```

3. **FastLog2f / FastPow2f** - Bit manipulation
   ```rust
   fn fast_log2f(x: f32) -> f32 {
       let bits = x.to_bits() as i32;
       let exp_shifted = (bits - 0x3f2aaaab) >> 23;
       let mantissa = f32::from_bits((bits - (exp_shifted << 23)) as u32);
       // Rational polynomial approximation...
   }
   ```

### Phase 2: Port Block-Level Functions

4. **HfModulation** - Needs 8x8 block input
   - Extract block from input buffer
   - Compute gradients with boundary handling
   - Sum and scale

5. **GammaModulation** - Needs 8x8 block input
   - Apply ratio_gamma to each pixel
   - Sum and log

6. **PerBlockModulations** - Combines above
   - Already have initial quant_field from FuzzyErosion
   - Apply ComputeMask → HfModulation → GammaModulation → FastPow2f

### Phase 3: Port Buffer Operations

7. **ComputePreErosion** - Complex buffer handling
   - Needs RowBuffer equivalent
   - Border replication
   - 4x subsampling

8. **FuzzyErosion** - Morphological filter
   - Sort4/UpdateMin4 helpers
   - 2x subsampling to block level

9. **ComputeAdaptiveQuantField** - Full pipeline
   - Coordinate all stages
   - Handle MCU row boundaries

---

## Known Difficulties

### 1. SIMD Dependency

C++ uses Highway (HWY) SIMD throughout. Options:
- Port to scalar first (correct but slow)
- Use `wide` crate for SIMD
- Keep scalar for accuracy testing, optimize later

### 2. RowBuffer Border Handling

C++ RowBuffer supports negative indices and automatic border extension:
```cpp
m->input_buffer[y_channel].CopyRow(-1, 0, 1);  // Copy row 0 to row -1
```

Rust needs explicit padding or careful bounds checking.

### 3. FastLog2f/FastPow2f Approximations

These are NOT equivalent to std::log2/exp2. They use:
- Bit manipulation for exponent extraction
- Rational polynomial approximations

Must match C++ exactly for bitwise identical results.

### 4. Coordinate Systems

Multiple resolution levels:
- Pixels: full resolution
- 4x subsampled (ComputePreErosion output)
- 2x subsampled (FuzzyErosion intermediate)
- Blocks: 8x8 (final quant_field)

Index calculations must be exact.

### 5. Float Precision

C++ uses `float` with SIMD. Rust f32 should match, but:
- Accumulation order matters
- SIMD lane operations may differ

---

## Testing Approaches

### 1. Unit Tests with Testdata

For each function with testdata:
```rust
#[test]
fn test_compute_aq_field_from_testdata() {
    for test in load_testdata("ComputeAdaptiveQuantField.testdata") {
        let input = convert_row_buffer(&test.input_buffer_y_slice);
        let result = compute_adaptive_quant_field(&input, &test.config);
        assert_eq_approx(&result, &test.expected_quant_field_slice, 1e-5);
    }
}
```

### 2. Fuzz Testing Pure Functions

```rust
#[test]
fn fuzz_ratio_gamma() {
    for _ in 0..100000 {
        let v: f32 = rand::random::<f32>() * 255.0;
        let rust = ratio_gamma::<false>(v);
        let cpp = ffi::ratio_gamma_cpp(v);  // Via FFI to C++
        assert!((rust - cpp).abs() < 1e-5);
    }
}
```

### 3. FFI Comparison

Build C++ as a library and call via FFI:
```rust
extern "C" {
    fn ComputeMask_cpp(out_val: f32) -> f32;
    fn FastLog2f_cpp(x: f32) -> f32;
}

#[test]
fn test_compute_mask_ffi() {
    for v in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let rust = compute_mask(v);
        let cpp = unsafe { ComputeMask_cpp(v) };
        assert!((rust - cpp).abs() < 1e-6);
    }
}
```

### 4. End-to-End Comparison

Compare final aq_strength values:
```rust
#[test]
fn test_aq_strength_distribution() {
    let rust_aq = compute_rust_aq(&image);
    let cpp_aq = load_cpp_aq_from_testdata();

    // Compare statistics
    assert!((rust_aq.mean() - cpp_aq.mean()).abs() < 0.01);
    assert!((rust_aq.max() - cpp_aq.max()).abs() < 0.05);
}
```

---

## Appendix: C++ Source Snippets

### ComputeMask
```cpp
template <class D, class V>
V ComputeMask(const D d, const V out_val) {
  const auto kBase = Set(d, -0.74174993f);
  const auto kMul4 = Set(d, 3.2353257320940401f);
  const auto kMul2 = Set(d, 12.906028311180409f);
  const auto kOffset2 = Set(d, 305.04035728311436f);
  const auto kMul3 = Set(d, 5.0220313103171232f);
  const auto kOffset3 = Set(d, 2.1925739705298404f);
  const auto kOffset4 = Mul(Set(d, 0.25f), kOffset3);
  const auto kMul0 = Set(d, 0.74760422233706747f);
  const auto k1 = Set(d, 1.0f);

  const auto v1 = Max(Mul(out_val, kMul0), Set(d, 1e-3f));
  const auto v2 = Div(k1, Add(v1, kOffset2));
  const auto v3 = Div(k1, MulAdd(v1, v1, kOffset3));
  const auto v4 = Div(k1, MulAdd(v1, v1, kOffset4));
  return Add(kBase, MulAdd(kMul4, v4, MulAdd(kMul2, v2, Mul(kMul3, v3))));
}
```

### Final Transformation
```cpp
// In ComputeAdaptiveQuantField after PerBlockModulations:
for (int y = 0; y < cinfo->max_v_samp_factor; ++y) {
  float* row = m->quant_field.Row(yb0 + y);
  for (size_t x = 0; x < xsize_blocks; ++x) {
    row[x] = std::max(0.0f, (0.6f / row[x]) - 1.0f);
  }
}
```

This transforms the modulated values (typically 0.3-0.6) to aq_strength (0.0-0.2).

---

## FFI Testing Strategy

### Overview

To ensure correctness, we can call C++ functions via FFI and compare results in real-time.

### Building C++ as a Library

Create `lib/jpegli/aq_ffi.cc`:
```cpp
extern "C" {
    float compute_mask_ffi(float out_val);
    float ratio_of_derivatives_ffi(float v, bool invert);
    float fast_log2f_ffi(float x);
    float fast_pow2f_ffi(float x);
}
```

### CMake Integration

```cmake
add_library(jpegli_aq_ffi SHARED
    lib/jpegli/aq_ffi.cc
)
target_link_libraries(jpegli_aq_ffi jpegli-static)
```

### Rust FFI Bindings

```rust
#[link(name = "jpegli_aq_ffi")]
extern "C" {
    fn compute_mask_ffi(out_val: f32) -> f32;
    fn ratio_of_derivatives_ffi(v: f32, invert: bool) -> f32;
    fn fast_log2f_ffi(x: f32) -> f32;
    fn fast_pow2f_ffi(x: f32) -> f32;
}
```

### Parallel Verification Test

```rust
#[test]
fn test_compute_mask_matches_cpp() {
    for i in 0..10000 {
        let v = (i as f32) * 0.001;
        let rust = compute_mask(v);
        let cpp = unsafe { compute_mask_ffi(v) };
        assert!(
            (rust - cpp).abs() < 1e-6,
            "Mismatch at v={}: rust={}, cpp={}", v, rust, cpp
        );
    }
}
```

---

## Locked Tests (Never Disable)

### Test File Location
`jpegli/tests/aq_locked_tests.rs`

### Purpose
These tests MUST pass before any AQ-related code is merged. They cannot be marked `#[ignore]`.

### Required Tests

1. **test_aq_strength_range** - Output must be in 0.0-0.3 range
2. **test_aq_mean_matches_cpp** - Mean aq_strength within 10% of C++ testdata
3. **test_aq_improves_quality** - Enabling AQ must not degrade DSSIM by >5%
4. **test_aq_reduces_size** - Enabling AQ must reduce file size at Q90+
5. **test_pure_functions_match_cpp** - ComputeMask, ratio_gamma, FastLog2f/Pow2f

### Enforcement

In `CLAUDE.md`:
```markdown
## MANDATORY: AQ Test Lock

The following tests in `aq_locked_tests.rs` MUST NEVER be:
- Marked as `#[ignore]`
- Deleted
- Have their assertions weakened

If these tests fail, the AQ implementation is BROKEN and must be fixed
before any other work continues.
```
