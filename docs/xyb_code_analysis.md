# XYB Mode in jpegli C++ Encoder: Complete Code Path Analysis

This document traces all XYB-specific code paths in the jpegli C++ encoder.

## Subsampling: XYB (2:2:1) vs YCbCr (4:2:0)

### Understanding JPEG Sampling Factors

In JPEG, `h_samp_factor` and `v_samp_factor` are **relative within an MCU**. Higher values = more samples = more resolution for that component. The maximum factor across all components determines the MCU size.

### XYB Mode Configuration

**Source:** `lib/jpegli/encode.cc:854-862`

```cpp
if (cinfo->master->xyb_mode) {
  // Subsample blue channel.
  cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
  cinfo->comp_info[1].h_samp_factor = cinfo->comp_info[1].v_samp_factor = 2;
  cinfo->comp_info[2].h_samp_factor = cinfo->comp_info[2].v_samp_factor = 1;
  // Use separate quantization tables for each component
  cinfo->comp_info[1].quant_tbl_no = 1;
  cinfo->comp_info[2].quant_tbl_no = 2;
}
```

| Index | ID | h_samp | v_samp | quant_tbl | Blocks/MCU | Resolution |
|-------|-----|--------|--------|-----------|------------|------------|
| 0 | `'R'` (X) | 2 | 2 | 0 | 4 | **Full** |
| 1 | `'G'` (Y) | 2 | 2 | 1 | 4 | **Full** |
| 2 | `'B'` (B) | 1 | 1 | 2 | 1 | **1/4** (half each dim) |

**MCU size:** 16×16 pixels (max factor is 2 in each dimension)

### YCbCr Mode Configuration (4:2:0)

**Source:** `lib/jpegli/encode.cc:868-879`

```cpp
} else if (colorspace == JCS_YCbCr || colorspace == JCS_YCCK) {
  // Use separate quantization and Huffman tables for luma and chroma
  cinfo->comp_info[1].quant_tbl_no = 1;
  cinfo->comp_info[2].quant_tbl_no = 1;
  cinfo->comp_info[1].dc_tbl_no = cinfo->comp_info[1].ac_tbl_no = 1;
  cinfo->comp_info[2].dc_tbl_no = cinfo->comp_info[2].ac_tbl_no = 1;
  // Use chroma subsampling by default
  cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
}
```

| Index | ID | h_samp | v_samp | quant_tbl | Blocks/MCU | Resolution |
|-------|-----|--------|--------|-----------|------------|------------|
| 0 | 1 (Y) | 2 | 2 | 0 | 4 | **Full** |
| 1 | 2 (Cb) | 1 | 1 | 1 | 1 | **1/4** |
| 2 | 3 (Cr) | 1 | 1 | 1 | 1 | **1/4** |

### Visual Comparison

```
XYB (2:2:1) - "Blue subsampled"         YCbCr 4:2:0 - "Chroma subsampled"
┌────┬────┐ ┌────┬────┐ ┌────────┐      ┌────┬────┐ ┌────────┐ ┌────────┐
│ X0 │ X1 │ │ Y0 │ Y1 │ │        │      │ Y0 │ Y1 │ │        │ │        │
├────┼────┤ ├────┼────┤ │   B    │      ├────┼────┤ │   Cb   │ │   Cr   │
│ X2 │ X3 │ │ Y2 │ Y3 │ │        │      │ Y2 │ Y3 │ │        │ │        │
└────┴────┘ └────┴────┘ └────────┘      └────┴────┘ └────────┘ └────────┘
 4 blocks    4 blocks    1 block         4 blocks    1 block    1 block
 Full res    Full res    1/4 res         Full res    1/4 res    1/4 res
```

### Perceptual Rationale

**YCbCr:** Human vision has lower spatial resolution for color (chroma) than brightness (luma). Subsampling Cb and Cr saves ~50% with minimal perceptual loss.

**XYB:** The XYB color space is designed around human perception:
- **X** (difference) = red-green opponent channel
- **Y** (sum) = luminance-like, most perceptually important
- **B** (blue) = human vision is least sensitive to blue spatial detail

So XYB keeps X and Y at full resolution, only downsamples B.

---

## Entry Points

### API Declaration
**Source:** `lib/jpegli/encode.h:131-132`
```cpp
// because some default setting depend on the XYB mode.
void jpegli_set_xyb_mode(j_compress_ptr cinfo);
```

### Implementation
**Source:** `lib/jpegli/encode.cc:744-747`
```cpp
void jpegli_set_xyb_mode(j_compress_ptr cinfo) {
  CheckState(cinfo, zenjpeg::kEncStart);
  cinfo->master->xyb_mode = true;
}
```

### Initialization
**Source:** `lib/jpegli/encode.cc:734`
```cpp
cinfo->master->xyb_mode = false;
```

### Storage
**Source:** `lib/jpegli/encode_internal.h:74`
```cpp
bool xyb_mode;
```

---

## Color Transform Handling

**Source:** `lib/jpegli/color_transform.cc:329-331`
```cpp
if (cinfo->in_color_space == JCS_RGB && m->xyb_mode) {
  JPEGLI_ERROR("Color transform on XYB colorspace is not supported.");
}
```

XYB mode **prohibits** color transforms. Data stays as RGB (which is actually XYB values).

### Colorspace Selection
**Source:** `lib/jpegli/encode.cc:768-771`
```cpp
if (cinfo->in_color_space == JCS_RGB && cinfo->master->xyb_mode) {
  jpegli_set_colorspace(cinfo, JCS_RGB);
  return;
}
```

When XYB + RGB input, forces `JCS_RGB` colorspace (no YCbCr conversion).

---

## Quantization

### Global Scale Constants
**Source:** `lib/jpegli/quant.cc:26-29`
```cpp
// Global scale is chosen in a way that butteraugli 3-norm matches libjpeg
// with the same quality setting. Fitted for quality 90 on jyrki31 corpus.
constexpr float kGlobalScaleXYB = 1.43951668f;
constexpr float kGlobalScaleYCbCr = 1.73966010f;
```

### Quantization Matrix Selection
**Source:** `lib/jpegli/quant.cc:660-673`
```cpp
const bool xyb = m->xyb_mode && cinfo->jpeg_color_space == JCS_RGB;
// ...
if (xyb) {
  global_scale = kGlobalScaleXYB;
  num_base_tables = 3;
  base_quant_matrix[0] = kBaseQuantMatrixXYB;
  base_quant_matrix[1] = kBaseQuantMatrixXYB + DCTSIZE2;
  base_quant_matrix[2] = kBaseQuantMatrixXYB + 2 * DCTSIZE2;
}
```

### XYB Base Quantization Matrix
**Source:** `lib/jpegli/quant.cc:31-227`

Three separate 8×8 matrices (192 floats total):

**Component 0 (X channel):** `quant.cc:31-96`
```cpp
constexpr float kBaseQuantMatrixXYB[] = {
  // c = 0 (X channel)
  7.5629935265f, 19.8247814178f, 22.5724945068f, 20.6706695557f,
  22.6864585876f, 23.5696277618f, 25.8129081726f, 36.3307571411f,
  // ... (64 values total, range ~7.5 to ~67)
```

**Component 1 (Y channel):** `quant.cc:97-161`
```cpp
  // c = 1 (Y channel)
  1.6262000799f, 3.2199242115f, 3.4903779030f, 3.9148359299f,
  // ... (64 values total, range ~1.6 to ~7.6)
```

**Component 2 (B channel):** `quant.cc:162-227`
```cpp
  // c = 2 (B channel)
  3.3038473129f, 10.0689258575f, 12.2785224915f, 14.6041173935f,
  // ... (64 values total, range ~3.3 to ~55.7)
```

**Key observation:** Y channel has the smallest values (finest quantization) because it's most perceptually important.

---

## XYB Color Conversion (External/Testing)

The actual XYB conversion is **NOT** done inside the jpegli encoder core. It's done externally before feeding data to the encoder.

### Conversion Pipeline
**Source:** `lib/extras/enc/jpegli.cc:498-528`
```cpp
if (jpeg_settings.xyb) {
  float* src_buf = c_transform.BufSrc(0);
  float* dst_buf = c_transform.BufDst(0);
  for (size_t y = 0; y < image.ysize; ++y) {
    // convert to float
    ToFloatRow(&pixels[y * image.stride], image.format, image.xsize,
               info.num_color_channels, src_buf);
    // convert to linear srgb
    if (!c_transform.Run(0, src_buf, dst_buf, image.xsize)) {
      return false;
    }
    // deinterleave channels
    float* row0 = &xyb_tmp[0];
    float* row1 = &xyb_tmp[rowlen];
    float* row2 = &xyb_tmp[2 * rowlen];
    for (size_t x = 0; x < image.xsize; ++x) {
      row0[x] = dst_buf[3 * x + 0];
      row1[x] = dst_buf[3 * x + 1];
      row2[x] = dst_buf[3 * x + 2];
    }
    // convert to xyb
    LinearRGBRowToXYB(row0, row1, row2, premul_absorb.get(), image.xsize);
    // scale xyb
    ScaleXYBRow(row0, row1, row2, image.xsize);
    // interleave channels and feed to jpegli as native endian floats
```

### XYB Transform Implementation
**Source:** `lib/extras/xyb_transform.cc:66-90`
```cpp
// Converts one RGB vector to XYB.
template <class V>
void LinearRGBToXYB(const V r, const V g, const V b,
                    const float* JXL_RESTRICT premul_absorb,
                    float* JXL_RESTRICT valx, float* JXL_RESTRICT valy,
                    float* JXL_RESTRICT valz) {
  V mixed0, mixed1, mixed2;
  OpsinAbsorbance(r, g, b, premul_absorb, &mixed0, &mixed1, &mixed2);

  // mixed* should be non-negative even for wide-gamut, so clamp to zero.
  mixed0 = ZeroIfNegative(mixed0);
  mixed1 = ZeroIfNegative(mixed1);
  mixed2 = ZeroIfNegative(mixed2);

  // Apply cube root
  mixed0 = CubeRootAndAdd(mixed0, Load(d, premul_absorb + 9 * N));
  mixed1 = CubeRootAndAdd(mixed1, Load(d, premul_absorb + 10 * N));
  mixed2 = CubeRootAndAdd(mixed2, Load(d, premul_absorb + 11 * N));

  StoreXYB(mixed0, mixed1, mixed2, valx, valy, valz);
}
```

### Opsin Absorbance Matrix
**Source:** `lib/extras/xyb_transform.cc:35-54`
```cpp
template <class V>
JXL_INLINE void OpsinAbsorbance(const V r, const V g, const V b,
                                const float* JXL_RESTRICT premul_absorb,
                                V* JXL_RESTRICT mixed0, V* JXL_RESTRICT mixed1,
                                V* JXL_RESTRICT mixed2) {
  const float* bias = jxl::cms::kOpsinAbsorbanceBias.data();
  // 3x3 matrix multiplication + bias
  *mixed0 = MulAdd(m0, r, MulAdd(m1, g, MulAdd(m2, b, Set(d, bias[0]))));
  *mixed1 = MulAdd(m3, r, MulAdd(m4, g, MulAdd(m5, b, Set(d, bias[1]))));
  *mixed2 = MulAdd(m6, r, MulAdd(m7, g, MulAdd(m8, b, Set(d, bias[2]))));
}
```

### StoreXYB (Final XYB Computation)
**Source:** `lib/extras/xyb_transform.cc:56-64`
```cpp
template <class V>
void StoreXYB(const V r, V g, const V b, float* JXL_RESTRICT valx,
              float* JXL_RESTRICT valy, float* JXL_RESTRICT valz) {
  const V half = Set(d, 0.5f);
  Store(Mul(half, Sub(r, g)), d, valx);  // X = 0.5 * (mixed0 - mixed1)
  Store(Mul(half, Add(r, g)), d, valy);  // Y = 0.5 * (mixed0 + mixed1)
  Store(b, d, valz);                      // B = mixed2
}
```

### Scale XYB to [0,1] Range
**Source:** `lib/extras/xyb_transform.cc:142-152`
```cpp
void ScaleXYBRow(float* JXL_RESTRICT row0, float* JXL_RESTRICT row1,
                 float* JXL_RESTRICT row2, size_t xsize) {
  for (size_t x = 0; x < xsize; x++) {
    row2[x] = (row2[x] - row1[x] + jxl::cms::kScaledXYBOffset[2]) *
              jxl::cms::kScaledXYBScale[2];
    row0[x] = (row0[x] + jxl::cms::kScaledXYBOffset[0]) *
              jxl::cms::kScaledXYBScale[0];
    row1[x] = (row1[x] + jxl::cms::kScaledXYBOffset[1]) *
              jxl::cms::kScaledXYBScale[1];
  }
}
```

---

## Opsin/XYB Constants
**Source:** `lib/cms/opsin_params.h`

### Opsin Absorbance Matrix
```cpp
constexpr float kM00 = 0.30f;
constexpr float kM01 = 0.622f;  // 1.0 - 0.078 - 0.30
constexpr float kM02 = 0.078f;

constexpr float kM10 = 0.23f;
constexpr float kM11 = 0.692f;  // 1.0 - 0.078 - 0.23
constexpr float kM12 = 0.078f;

constexpr float kM20 = 0.24342268924547819f;
constexpr float kM21 = 0.20476744424496821f;
constexpr float kM22 = 0.5518098665095536f;  // 1.0 - kM20 - kM21

constexpr Matrix3x3 kOpsinAbsorbanceMatrix{
    {{kM00, kM01, kM02}, {kM10, kM11, kM12}, {kM20, kM21, kM22}}};
```

### Opsin Absorbance Bias
```cpp
constexpr float kOpsinAbsorbanceBias0 = 0.0037930732552754493f;
constexpr float kOpsinAbsorbanceBias1 = kOpsinAbsorbanceBias0;
constexpr float kOpsinAbsorbanceBias2 = kOpsinAbsorbanceBias0;
```

### Scaled XYB Constants (for [0,1] normalization)
```cpp
constexpr float kScaledXYBOffset0 = 0.015386134f;
constexpr float kScaledXYBOffset1 = 0.0f;
constexpr float kScaledXYBOffset2 = 0.27770459f;

constexpr float kScaledXYBScale0 = 22.995788804f;
constexpr float kScaledXYBScale1 = 1.183000077f;
constexpr float kScaledXYBScale2 = 1.502141333f;
```

---

## Adaptive Quantization

### XYB Gamma Handling
**Source:** `lib/jpegli/adaptive_quantization.cc:523-528`
```cpp
// The XYB gamma is 3.0 to be able to decode faster with two muls.
// Butteraugli's gamma is matching the gamma of human eye, around 2.6.
// We approximate the gamma difference by adding one cubic root into
// the adaptive quantization. This gives us a total gamma of 2.6666
// for quantization uses.
static const float match_gamma_offset = 0.019 / kInputScaling;
```

The AQ code applies a gamma correction factor to account for the difference between XYB's gamma (3.0) and Butteraugli's perceptual gamma (~2.6).

---

## ICC Profile Handling

**Source:** `lib/extras/enc/jpegli.cc:374-387`
```cpp
ColorEncoding xyb_encoding;
if (jpeg_settings.xyb) {
  if (HasICCProfile(jpeg_settings.app_data)) {
    return JXL_FAILURE("APP data ICC profile is not supported in XYB mode.");
  }
  const ColorEncoding& c_desired = ColorEncoding::LinearSRGB(false);
  JXL_RETURN_IF_ERROR(
      c_transform.Init(color_encoding, c_desired, 255.0f, ppf.info.xsize, 1));
  xyb_encoding.SetColorSpace(jxl::ColorSpace::kXYB);
  xyb_encoding.SetRenderingIntent(jxl::RenderingIntent::kPerceptual);
  JXL_RETURN_IF_ERROR(xyb_encoding.CreateICC());
}
```

**Source:** `lib/extras/enc/jpegli.cc:492-496`
```cpp
if ((jpeg_settings.app_data.empty() && !output_encoding.IsSRGB()) ||
    jpeg_settings.xyb) {
  jpegli_write_icc_profile(&cinfo, output_encoding.ICC().data(),
                           output_encoding.ICC().size());
}
```

XYB mode:
1. Prohibits external ICC profiles in APP markers
2. Generates and embeds an XYB colorspace ICC profile automatically

---

## Complete XYB Source File Index

### Core Encoder Files

| File | Lines | Content |
|------|-------|---------|
| `lib/jpegli/encode.h` | 131-132 | API declaration |
| `lib/jpegli/encode.cc` | 734 | Initialization (`xyb_mode = false`) |
| `lib/jpegli/encode.cc` | 744-747 | `jpegli_set_xyb_mode()` implementation |
| `lib/jpegli/encode.cc` | 768-771 | Colorspace override for XYB+RGB |
| `lib/jpegli/encode.cc` | 850-862 | Subsampling and quant table assignment |
| `lib/jpegli/encode_internal.h` | 74 | `bool xyb_mode` storage |
| `lib/jpegli/color_transform.cc` | 329-331 | XYB color transform prohibition |
| `lib/jpegli/quant.cc` | 28 | `kGlobalScaleXYB = 1.43951668f` |
| `lib/jpegli/quant.cc` | 31-227 | `kBaseQuantMatrixXYB[]` (3 × 64 floats) |
| `lib/jpegli/quant.cc` | 660 | XYB mode detection |
| `lib/jpegli/quant.cc` | 668-673 | XYB quantization matrix selection |
| `lib/jpegli/adaptive_quantization.cc` | 523-528 | XYB gamma compensation |

### XYB Transform Files

| File | Lines | Content |
|------|-------|---------|
| `lib/cms/opsin_params.h` | 21-43 | Opsin absorbance matrix constants |
| `lib/cms/opsin_params.h` | 55-59 | Opsin absorbance bias |
| `lib/cms/opsin_params.h` | 65-80 | Scaled XYB offset/scale constants |
| `lib/extras/xyb_transform.h` | 18-26 | `LinearRGBRowToXYB`, `ScaleXYBRow` declarations |
| `lib/extras/xyb_transform.cc` | 35-54 | `OpsinAbsorbance()` - matrix multiply |
| `lib/extras/xyb_transform.cc` | 56-64 | `StoreXYB()` - X/Y/B computation |
| `lib/extras/xyb_transform.cc` | 66-90 | `LinearRGBToXYB()` - full pipeline |
| `lib/extras/xyb_transform.cc` | 92-102 | `LinearRGBRowToXYB()` - row processing |
| `lib/extras/xyb_transform.cc` | 142-152 | `ScaleXYBRow()` - normalize to [0,1] |

### High-Level Encoder

| File | Lines | Content |
|------|-------|---------|
| `lib/extras/enc/jpegli.h` | 26 | `bool xyb = false` setting |
| `lib/extras/enc/jpegli.cc` | 374-387 | XYB ICC profile setup |
| `lib/extras/enc/jpegli.cc` | 420-423 | `jpegli_set_xyb_mode()` call |
| `lib/extras/enc/jpegli.cc` | 457-461 | Subsampling override logic |
| `lib/extras/enc/jpegli.cc` | 482-528 | XYB conversion pipeline |

### FFI/Testing

| File | Lines | Content |
|------|-------|---------|
| `lib/extras/jpegli_test_ffi.h` | 40-67 | FFI function declarations |
| `lib/extras/jpegli_test_ffi.cc` | 70-134 | `jpegli_linear_to_xyb()` |
| `lib/extras/jpegli_test_ffi.cc` | 136-180 | `jpegli_scale_xyb()` |
| `lib/extras/jpegli_test_ffi.cc` | 189-206 | `jpegli_srgb_to_scaled_xyb()` |
| `lib/extras/jpegli_test_ffi.cc` | 209-238 | `jpegli_get_xyb_constants()` |
| `lib/jpegli/test_params.h` | 106 | `bool xyb_mode = false` test config |
| `lib/jpegli/test_utils.cc` | 315-316, 575-576 | Test parameter handling |

---

## What XYB Mode Does NOT Affect

No XYB-specific logic in:
- DCT computation (`lib/jpegli/dct.cc`)
- Huffman encoding (`lib/jpegli/huffman.cc`)
- Entropy coding (`lib/jpegli/entropy_coding.cc`)
- Bitstream generation (`lib/jpegli/bitstream.cc`)
- Progressive scan ordering
- Marker writing (except ICC profile)

---

## Control Flow Summary

```
jpegli_set_xyb_mode(cinfo)
  └─→ cinfo->master->xyb_mode = true

jpegli_set_defaults(cinfo)
  └─→ jpegli_default_colorspace(cinfo)
        └─→ if (RGB && xyb_mode) → jpegli_set_colorspace(JCS_RGB)

jpegli_set_colorspace(cinfo, JCS_RGB)
  └─→ if (xyb_mode):
        ├─→ h/v_samp_factor: [2,2], [2,2], [1,1]
        └─→ quant_tbl_no: [0, 1, 2]

SetQuantMatrices(cinfo, distances, ...)
  └─→ xyb = (xyb_mode && jpeg_color_space == JCS_RGB)
        └─→ if (xyb):
              ├─→ global_scale = 1.43951668f
              └─→ Use kBaseQuantMatrixXYB[0/1/2]

[External: enc/jpegli.cc]
  └─→ For each row:
        ├─→ ToFloatRow() - convert to float
        ├─→ c_transform.Run() - convert to linear sRGB
        ├─→ LinearRGBRowToXYB() - convert to XYB
        ├─→ ScaleXYBRow() - normalize to [0,1]
        └─→ jpegli_write_scanlines() - feed to encoder
```
