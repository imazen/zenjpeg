# XYB ICC Profile Technical Notes

## Overview

XYB is a perceptual color space used by JPEG XL internally. Despite initial assumptions that XYB cannot be represented as an ICC profile, **jpegli proves this is possible** by embedding XYB ICC profiles in standard JPEG files.

## Why XYB Seems Incompatible with ICC

### The naive view (incorrect)

1. **Opponent channels can be negative**: XYB's X channel (L-M) ranges from ~-0.015 to ~+0.028 for sRGB
2. **Non-standard transfer function**: `cbrt(x + BIAS) - BIAS_CBRT` doesn't match ICC parametric curves
3. **LMS→XYB isn't a matrix**: Uses addition/subtraction (`X = (L-M)/2`, `Y = (L+M)/2`, `B = S-M`)

### The reality

From libjxl source code (`lib/jxl/decode.cc`):

> "for the XYB case, do we want to craft an ICC profile that represents XYB as an RGB profile? **It may be possible, but not with only 1D transfer functions.**"

The key insight: **LUT-based profiles can represent XYB**.

## How XYB ICC Profiles Work

### Matrix+TRC profiles (insufficient)

```
RGB → 1D TRC curves → 3x3 Matrix → PCS (Lab/XYZ)
```

This only supports:
- Separable 1D transfer functions per channel
- Linear matrix transform

XYB requires non-separable transforms (the L-M opponent calculation mixes channels before the nonlinearity).

### LUT-based profiles (sufficient)

```
RGB → A2B 3D CLUT → PCS (Lab/XYZ)
PCS → B2A 3D CLUT → RGB
```

A 3D color lookup table can encode **any** RGB→PCS transform, including:
- The RGB→Linear conversion
- The linear RGB→LMS matrix
- The cube root with bias
- The LMS→XYB opponent transform
- The final scaling/offset to fit PCS range

## XYB Color Space Details

### Constants

```rust
pub const BIAS: f64 = 0.00379307325527544933;
pub const BIAS_CBRT: f64 = 0.155954200549248620;
```

### Forward Transform (RGB → XYB)

```rust
// 1. sRGB → Linear RGB (standard sRGB EOTF)
// 2. Linear RGB → LMS
let l = 0.3 * r + 0.622 * g + 0.078 * b;
let m = 0.23 * r + 0.692 * g + 0.078 * b;
let s = 0.24342268924547819 * r + 0.20476744424496821 * g + 0.55180986650955360 * b;

// 3. Apply gamma with bias
let l_gamma = cbrt(l + BIAS) - BIAS_CBRT;
let m_gamma = cbrt(m + BIAS) - BIAS_CBRT;
let s_gamma = cbrt(s + BIAS) - BIAS_CBRT;

// 4. LMS → XYB (opponent channels)
let x = (l_gamma - m_gamma) * 0.5;  // Red-green opponent
let y = (l_gamma + m_gamma) * 0.5;  // Luminance
let b = s_gamma - m_gamma;           // Blue
```

### Channel Ranges (for sRGB gamut)

| Channel | Min | Max | Description |
|---------|-----|-----|-------------|
| X | -0.015 | +0.028 | L-M opponent (red-green) |
| Y | 0.000 | 0.845 | Luminance-like |
| B | -0.293 | +0.388 | Blue opponent |

### Inverse Transform (XYB → RGB)

```rust
// 1. XYB → LMS (gamma domain)
let l_gamma = x + y + BIAS_CBRT;
let m_gamma = -x + y + BIAS_CBRT;
let s_gamma = -x + y + b + BIAS_CBRT;

// 2. Remove gamma (cube)
let l = l_gamma.powi(3) - BIAS;
let m = m_gamma.powi(3) - BIAS;
let s = s_gamma.powi(3) - BIAS;

// 3. LMS → Linear RGB (inverse matrix)
let r = 11.031566901960783 * l - 9.866943921568629 * m - 0.16462299647058826 * s;
let g = -3.254147380392157 * l + 4.418770392156863 * m - 0.16462299647058826 * s;
let b = -3.6588512862745097 * l + 2.7129230470588235 * m + 1.9459282392156863 * s;

// 4. Linear RGB → sRGB (standard sRGB OETF)
```

## jpegli Implementation

### How jpegli uses XYB

From [jpegli README](https://github.com/libjxl/libjxl/blob/main/lib/jpegli/README.md):

> "Support for more efficient compression of JPEGs with an ICC profile representing the XYB colorspace. These JPEGs will not be converted to the YCbCr colorspace, but specialized quantization tables will be chosen for the original X, Y, B channels."

### Workflow

1. Input image (sRGB or other) is converted to XYB
2. XYB channels are stored directly in JPEG (not converted to YCbCr)
3. An XYB ICC profile is embedded in the JPEG's APP2 marker
4. Decoders use the ICC profile to convert XYB back to display colorspace

### Compatibility

| Application | Status | Notes |
|-------------|--------|-------|
| Chrome | ✅ Works | Full ICC support |
| Firefox | ✅ Works | Full ICC support |
| Safari | ⚠️ Needs APP14 | Requires Adobe marker |
| Photoshop | ⚠️ Needs APP14 | Requires Adobe marker |
| macOS Preview | ⚠️ Needs APP14 | Requires Adobe marker |

See [libjxl issue #3512](https://github.com/libjxl/libjxl/issues/3512) for APP14 marker details.

## Known Issues

### colorutils-rs XYB Bug (v0.7.5)

colorutils-rs v0.7.5 has a channel ordering bug in its XYB implementation:
- Colors with r=0 all produce identical incorrect XYB values
- Round-trip fails for most colors

Use the reference formulas above instead of colorutils-rs for XYB work.

### jpegli XYB Subsampling Bug

There was a bug with XYB color subsampling that has been fixed in git but not yet released. See [XnView forum discussion](https://newsgroup.xnview.com/viewtopic.php?t=48869).

## Performance

From [Gianni Rosato's analysis](https://giannirosato.com/blog/post/jpegli-xyb/):

> "XYB JPEG showed an incredibly shocking first showing, outperforming other JPEGs, WebP, and even AVIF at the critical high-fidelity range. For 4:4:4, XYB JPEG came second only to JXL."

XYB's perceptual quantization allows more efficient bit allocation, especially for:
- Blue channel (fewer bits needed due to lower cone density)
- Smooth gradients (perceptually uniform)

## References

- [libjxl jpegli](https://github.com/libjxl/libjxl/tree/main/lib/jpegli)
- [JPEG XL Format Overview](https://github.com/libjxl/libjxl/blob/main/doc/format_overview.md)
- [XYB JPEG Blog Post](https://giannirosato.com/blog/post/jpegli-xyb/)
- [ColorAide XYB Documentation](https://facelessuser.github.io/coloraide/colors/xyb/)
- [libjxl Color API](https://libjxl.readthedocs.io/en/latest/api_color.html)

## Implementation Notes for zenjpeg

To implement XYB support:

1. **XYB conversion**: Use the reference formulas above (colorutils-rs v0.7.5 has a known bug — see Known Issues)
2. **ICC profile generation**: Create A2B/B2A LUT-based profile
3. **CLUT resolution**: 17x17x17 or 33x33x33 grid points typical
4. **APP14 marker**: Include Adobe marker for broad compatibility
5. **Quantization tables**: Use XYB-optimized tables (different from YCbCr)

The XYB ICC profile needs to map:
- A2B0: XYB (as "RGB") → Lab PCS
- B2A0: Lab PCS → XYB (as "RGB")

Since X and B can be negative, the profile likely uses an offset/scale to map to [0,1] range expected by ICC.
