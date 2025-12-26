# SSIMULACRA2 Fork - ICC CMS Experiments

This document tracks experiments comparing different CMS (Color Management System)
implementations for decoding XYB JPEG files with embedded ICC profiles.

## Goal

Match libjxl's ssimulacra2 tool output for XYB JPEGs. libjxl uses Google's skcms
library internally for ICC profile conversion.

## Test Setup

- Source image: flower_small.png (510x532, sRGB)
- XYB JPEG at Q91: `/tmp/xyb_q91.jpg` (51947 bytes)
- Created with: `cjpegli --xyb -q 91`
- libjxl ssimulacra2 reference: `/home/lilith/work/jpegli/build/tools/ssimulacra2`

## Results Summary (Q91)

| CMS Backend | SSIMULACRA2 Score | Difference from skcms |
|-------------|-------------------|----------------------|
| **libjxl (skcms)** | 88.48 | — (reference) |
| **moxcms Linear** | 86.96 | -1.52 (1.7%) |
| moxcms Tetrahedral | 86.53 | -1.95 (2.2%) |
| lcms2 Perceptual | 85.97 | -2.51 (2.8%) |

**Winner**: moxcms with Linear (default) interpolation

## Experiment Log

### 2024-12-25: Initial CMS Comparison

**Hypothesis**: Tetrahedral interpolation (used by lcms2) would match skcms better.

**Result**: WRONG. moxcms Linear interpolation is actually closest to skcms.

**Details**:
- All CMS backends produce lower scores than skcms reference
- moxcms Linear is ~1.5 points closer than moxcms Tetrahedral
- moxcms is ~1 point closer than lcms2

### 2024-12-25: moxcms Options Investigation

Tested TransformOptions variants in moxcms:

1. **RenderingIntent**: Only Perceptual works for XYB profile (LUT-based)
   - Other intents throw `UnsupportedLutRenderingIntent` error

2. **InterpolationMethod**:
   - Linear (default): Best match to skcms
   - Tetrahedral: Slightly worse than Linear (counterintuitive!)
   - Pyramid: Similar to Tetrahedral
   - Prism: Similar to Tetrahedral

3. **prefer_fixed_point**: Minimal effect (sub-pixel differences)

4. **allow_extended_range_rgb_xyz**: No significant effect

### Pixel-Level Analysis

Raw decoded pixels from XYB JPEG (first pixel):
- jpeg-decoder raw: (96, 138, 119)
- After moxcms Linear: (134, 127, 155)
- After moxcms Tetrahedral: (134, 126, 156)
- After lcms2: (135, 126, 147)

The ICC transform produces visibly different output between CMS backends.

## Root Cause Analysis

The remaining ~1.5 point gap between moxcms and skcms likely comes from:

1. **Different sRGB reference profile**: skcms may use a slightly different sRGB primaries
2. **LUT interpolation precision**: skcms may use higher precision internally
3. **JPEG decoder differences**: libjxl uses its own decoder, not jpeg-decoder

The last point is likely significant - libjxl decodes the JPEG itself before applying
ICC conversion, while we use `jpeg-decoder` crate which may produce slightly different
raw pixel values.

## Recommendations

1. **Use moxcms with default (Linear) interpolation** - It's the closest to skcms
   among available pure-Rust options.

2. **Accept ~1.5 point difference** - This is likely due to fundamental differences
   in JPEG decoding and CMS implementation, not configuration issues.

3. **Consider jpegli-icc feature** - Using jpegli's own decoder with moxcms might
   get even closer by eliminating the jpeg-decoder variable.

## Feature Configuration

In ssimulacra2_rs Cargo.toml:

```toml
[features]
icc = ["lcms2", "jpeg-decoder"]           # Uses lcms2 (2.8% gap)
icc-moxcms = ["moxcms", "jpeg-decoder"]   # Uses moxcms (1.7% gap) - RECOMMENDED
jpegli-icc = ["jpegli/cms-lcms2"]         # Uses jpegli decoder + lcms2
jpegli-icc-moxcms = ["jpegli/cms-moxcms"] # Uses jpegli decoder + moxcms
```

For best skcms compatibility, use `icc-moxcms` with default Linear interpolation.
