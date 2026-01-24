# JPEG Encoding Option Constraints (jpegli-rs)

What we **actually support**, not theoretical JPEG capabilities.

## Supported Features

### DCT Modes (mutually exclusive)

| Mode | Status | Notes |
|------|--------|-------|
| **Baseline** | ✅ Supported | 8-bit, Huffman, default |
| **Progressive** | ✅ Supported | Multi-scan, requires optimized Huffman |
| Extended (12-bit) | ❌ Not supported | |
| Lossless | ❌ Not supported | |
| Arithmetic coding | ❌ Not supported | Huffman only |

### Huffman Tables (mutually exclusive)

| Method | Status | Progressive? |
|--------|--------|--------------|
| **Fixed/Standard** | ✅ Supported | ❌ Incompatible |
| **Optimized** | ✅ Supported | ✅ Required |

### Color Space (mutually exclusive)

| Space | Status | Components | Notes |
|-------|--------|------------|-------|
| **YCbCr** | ✅ Supported | 3 | Standard, all subsampling modes |
| **XYB** | ✅ Supported | 3 | Perceptual, fixed subsampling (X,Y full, B quarter) |
| **Grayscale** | ✅ Supported | 1 | Single component |
| RGB direct | ❌ Not supported | | Always converts to YCbCr |
| CMYK | ❌ Not supported | | Decode only |

### Chroma Subsampling (YCbCr only)

| Mode | Status | Resolution |
|------|--------|------------|
| **4:4:4** | ✅ Supported | Full chroma |
| **4:2:2** | ✅ Supported | Half horizontal |
| **4:2:0** | ✅ Supported | Quarter chroma |
| **4:4:0** | ✅ Supported | Half vertical |

### XYB Subsampling (XYB only)

| Mode | X | Y | B | Notes |
|------|---|---|---|-------|
| **Full** | 1x1 | 1x1 | 1x1 | Maximum quality |
| **B Quarter** | 2x2 | 2x2 | 1x1 | Default, perceptually optimized |

Note: XYB subsampling is separate from YCbCr subsampling. In XYB, two components (X,Y) are full even in "subsampled" mode - unlike YCbCr where only luma is full.

### Quantization Strategies (mutually exclusive)

| Strategy | Quality Used? | User Provides |
|----------|--------------|---------------|
| **Jpegli perceptual** | ✅ Yes | Nothing (default) |
| **Custom base matrices** | ✅ Yes | Base f32 matrices, scaled by quality |
| **Direct/exact tables** | ❌ No (ignored) | Exact u16 tables |

Standard JPEG tables not exposed (use direct tables if needed).

## Constraint Matrix

```
                        Baseline  Progressive
                        --------  -----------
Fixed Huffman              ✅         ❌
Optimized Huffman          ✅         ✅ (required)
YCbCr                      ✅         ✅
XYB                        ✅         ✅
Grayscale                  ✅         ✅
All subsampling modes      ✅         ✅
```

```
                        Quality-Scaled  Direct Tables
                        --------------  -------------
Quality param matters        ✅              ❌
Custom base matrices         ✅              N/A
Adaptive quantization        ✅              ❌
```

```
                        YCbCr   XYB    Gray
                        -----   ---    ----
4:4:4                    ✅      -      N/A
4:2:2                    ✅      -      N/A
4:2:0                    ✅      -      N/A
4:4:0                    ✅      -      N/A
XYB Full (all 1x1)       -       ✅     N/A
XYB B-Quarter            -       ✅     N/A
```

## Invalid Combinations (errors)

| Combination | Error |
|-------------|-------|
| Progressive + Fixed Huffman | Progressive requires optimized tables |
| Grayscale + any subsampling | No chroma to subsample |
| Direct tables + quality expectation | Quality is ignored with direct tables |

## Auto-Corrections (silent)

| Setting | Auto-enables |
|---------|-------------|
| `progressive(true)` | `optimize_huffman(true)` |

## API Design Implications

1. **Remove JpegMode::Extended and Lossless** from public API (or mark unavailable)
2. **XYB needs dedicated subsampling enum**:
   ```rust
   enum XybSubsampling {
       Full,      // X, Y, B all 1x1
       BQuarter,  // X, Y at 2x2, B at 1x1 (default)
   }
   ```
   YCbCr `Subsampling` enum should be ignored when XYB is enabled.
3. **Direct quant tables** should clearly document that quality is ignored
4. **Progressive + fixed Huffman** should error at config time, not encode time
5. **Grayscale + subsampling** should be a no-op (no error, just ignored)
