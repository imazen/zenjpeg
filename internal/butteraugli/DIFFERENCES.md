# Differences Between Rust and C++ Butteraugli Implementations

This document summarizes the key differences between this Rust port and the original
C++ butteraugli implementation in libjxl (`lib/extras/butteraugli.cc`).

## 1. Opsin Absorbance Matrix (CRITICAL)

The C++ implementation uses a **different opsin matrix** than jpegli's XYB:

### C++ Butteraugli Matrix
```cpp
// OpsinAbsorbance in butteraugli.cc
static const double mixi0 = 0.29956550340058319;
static const double mixi1 = 0.63373087833825936;
static const double mixi2 = 0.077705617820981968;
static const double mixi3 = 1.7557483643287353;   // bias (added, not subtracted)

static const double mixi4 = 0.22158691104574774;
static const double mixi5 = 0.69391388044116142;
static const double mixi6 = 0.0987313588422;
static const double mixi7 = 1.7557483643287353;   // bias

static const double mixi8 = 0.02;
static const double mixi9 = 0.02;
static const double mixi10 = 0.20480129041026129;
static const double mixi11 = 12.226454707163354;  // bias
```

### Rust/jpegli XYB Matrix (what we use)
```rust
// From jpegli consts
pub const XYB_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    0.30, 0.622, 0.078,           // Row 0
    0.23, 0.692, 0.078,           // Row 1
    0.243_422_69, 0.204_767_44, 0.551_809_87,  // Row 2
];
pub const XYB_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [0.003_793_073_3, ...];
```

**Impact**: XYB values will differ, affecting all downstream computations.

## 2. OpsinDynamicsImage Processing

C++ butteraugli has a sophisticated `OpsinDynamicsImage` function that:

1. Blurs the input RGB with sigma=1.2
2. Applies OpsinAbsorbance to the blurred RGB to get `pre_mixed`
3. Computes `sensitivity = Gamma(pre_mixed) / pre_mixed`
4. Applies OpsinAbsorbance to the original RGB to get `cur_mixed`
5. Multiplies `cur_mixed *= sensitivity` (dynamic range adaptation)
6. Computes XYB: `X = cur_mixed0 - cur_mixed1`, `Y = cur_mixed0 + cur_mixed1`, `B = cur_mixed2`

The Rust implementation uses jpegli's simpler XYB conversion:
1. Apply linear opsin matrix
2. Add bias
3. Apply cube root (no Gamma function, no dynamic sensitivity)

## 3. Gamma Function

C++ uses a custom gamma function based on FastLog2f:
```cpp
const auto kRetMul = Set(df, 19.245013259874995f * kInvLog2e);
const auto kRetAdd = Set(df, -23.16046239805755);
const auto biased = Add(v, Set(df, 9.9710635769299145));
const auto log = FastLog2f(df, biased);
return MulAdd(kRetMul, log, kRetAdd);
```

Rust uses cube root (`cbrt`), which is jpegli's approach.

## 4. Malta Filter (Edge-Aware Difference)

C++ has elaborate Malta filter (`MaltaDiffMap`, `MaltaDiffMapLF`) for edge-aware
difference computation. This uses special convolution patterns that weight
differences based on local edge structure.

Rust uses simple squared pixel differences without edge awareness.

## 5. Multi-Resolution Processing

C++ recursively computes butteraugli at 2x subsampled resolution:
```cpp
JXL_ASSIGN_OR_RETURN(Image3F rgb0_sub, SubSample2x(rgb0));
JXL_ASSIGN_OR_RETURN(Image3F rgb1_sub, SubSample2x(rgb1));
JXL_RETURN_IF_ERROR(ButteraugliDiffmapInPlace(rgb0_sub, rgb1_sub, ...));
AddSupersampled2x(subdiffmap, 0.5, diffmap);
```

Rust doesn't implement multi-resolution processing.

## 6. Final Score Computation

C++ uses the **maximum value** from the diffmap:
```cpp
for each pixel: retval = std::max(retval, row[x]);
return retval;
```

Rust uses a weighted combination:
```rust
(rms * 0.3 + mean * 0.3 + max_val * 0.4) * GLOBAL_SCALE
```

## 7. Asymmetric Difference

C++ has `L2DiffAsymmetric` which penalizes new artifacts (blurring) differently
from existing detail loss. This is controlled by `hf_asymmetry` parameter.

Rust uses symmetric squared difference everywhere.

## 8. Masking

C++ has sophisticated masking functions:
- `MaskY` and `MaskDcY` for luminance-based masking
- `CombineChannelsToDiffmap` with proper channel weighting
- Uses eroded mask values

Rust has simpler masking based on HF/UHF energy.

## Summary

This Rust implementation is a **simplified approximation** of butteraugli,
using jpegli's XYB color space rather than butteraugli's native opsin model.
The scores will differ from C++ butteraugli, but the relative ordering
(which image is "better") should generally agree.

For exact C++ parity, one would need to:
1. Port the exact OpsinAbsorbance matrix and biases
2. Implement OpsinDynamicsImage with Gamma function
3. Add Malta filter for edge-aware difference
4. Implement multi-resolution processing
5. Use max-based scoring
6. Add asymmetric difference support
