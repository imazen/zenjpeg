# zenjpeg Experiments & Research Log

## Goal

Create a JPEG encoder that:
1. **Auto-selects** between trellis (mozjpeg) and AQ (jpegli) based on image + quality
2. **Maps quality** to produce linear SSIMULACRA2 or Butteraugli curves
3. Achieves **Pareto-optimal** compression across all quality levels

---

## Timeline

### 2025-12-25: Initial Setup

**Status**: Modules integrated, trellis not yet wired to encoder

**Hypothesis 1**: Quality crossover point exists where trellis vs AQ trade off
- mozjpeg literature suggests trellis helps more at low quality (Q < 70)
- jpegli literature suggests AQ helps more at high quality (Q >= 70)
- Current strategy.rs uses Q=50/70/80 thresholds - need validation

**Next steps**:
1. Wire trellis into encoder
2. Run comparison: trellis ON vs OFF across Q 10-95
3. Measure SSIMULACRA2 and file size for each
4. Plot Pareto fronts

---

### 2025-12-25: Found jpegli quality-to-distance formula

**Location**: `/home/lilith/work/jpegli/jpegli-rs/jpegli/src/consts.rs:246`

```rust
pub fn quality_to_distance(quality: i32) -> f32 {
    if quality >= 100 {
        0.01
    } else if quality >= 30 {
        0.1 + (100 - quality) as f32 * 0.09  // Linear from Q30-Q100
    } else {
        // Quadratic for Q0-Q30
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}
```

**Key insight**: Distance ~1.0 is "visually lossless", lower = higher quality

### 2025-12-25: Found evalchroma crate for image analysis

**Location**: `evalchroma` crate (used by imageflow)

This crate analyzes images and recommends:
- Optimal chroma subsampling (4:4:4, 4:2:2, 4:2:0)
- Adjusted chroma quality based on image content
- Sharpness metrics (horiz, vert, peak)

**Key formula for threshold**:
```rust
let threshold = (130.0 - chroma_quality * 1.21).powf(1.56) as u32;
```

**Sharpness scaling by image size**:
- Tiny images (< 100px): sharpness *= 2 (need more detail)
- Large images (> 1920px): sharpness /= 2 (will be displayed scaled)

**Proposed use**: Use evalchroma sharpness to decide trellis vs AQ

---

## Quality Mapping Research

### Problem

Standard JPEG quality (1-100) does NOT map linearly to perceptual quality:
- Q90 vs Q95 is almost imperceptible
- Q30 vs Q40 is huge difference
- Users expect "Q50 = half quality" but that's not what happens

### Goal

Create quality mapping where:
```
linear_quality(x) → encoder_settings → constant_ssim2_delta
```

So Q50 produces roughly "halfway" perceptual quality between Q0 and Q100.

### Approach Options

**Option A: Butteraugli distance target**
- jpegli uses "distance" (Butteraugli-derived) internally
- Map Q to distance: `distance = f(quality)`
- Need to find the right curve

**Option B: Binary search to SSIMULACRA2 target**
- Given target SSIM2 score, binary search for quality
- Slow but accurate
- Could cache mapping per image category

**Option C: Learned quality curve**
- Run many encodes across image corpus
- Fit curve: `effective_quality = g(nominal_quality, image_features)`
- Use this for prediction

### jpegli's Approach (reference)

jpegli maps quality to "distance":
```
distance = quality_to_distance(quality)
// distance 0 = lossless, higher = more lossy
```

The exact mapping is in jpegli source. We could adopt or tune it.

---

## Auto-Selection Heuristics

### Image Characteristics to Consider

1. **Complexity/Entropy**
   - Simple images (logos, text) → trellis helps more
   - Complex images (photos) → AQ may help more

2. **Edge density**
   - High edges → preserve with AQ
   - Smooth gradients → trellis handles well

3. **Noise level**
   - Noisy images → trellis can remove noise (good or bad)
   - Clean images → AQ preserves detail

4. **Color distribution**
   - High saturation → chroma matters more
   - Grayscale-ish → luma dominates

### Proposed Metrics

From imageflow's chromaeval or similar:
- DCT coefficient distribution (how many zeros naturally)
- Block variance histogram
- Edge detection score
- Histogram entropy

### Decision Tree Hypothesis

```
if quality < 50:
    use_trellis = True
    use_aq = False
elif quality < 70:
    # Hybrid zone - decide based on image
    if image_is_simple():
        use_trellis = True
    else:
        use_aq = moderate_strength
else:
    use_trellis = False  # hurts at high Q
    use_aq = True
```

---

## Experiments To Run

### Experiment 1: Trellis Impact by Quality

**Hypothesis**: Trellis improves compression more at low quality

**Method**:
1. Select 20 diverse images from codec-corpus
2. Encode each at Q = [20, 30, 40, 50, 60, 70, 80, 90]
3. For each: trellis ON vs OFF
4. Measure: file size, SSIMULACRA2, encode time

**Expected**: Trellis saves 10-20% at Q30, <5% at Q80

**Results (2025-12-25)**:
- Test image: 128x128 natural-looking gradient/edge pattern
- Q75 with trellis: 1592 bytes
- Q75 without trellis: 3347 bytes
- **Compression improvement: 52.4%**

This exceeds expectations! Trellis is providing major compression gains at mid-quality.

Note: Overall we're still ~1.8x larger than mozjpeg C, suggesting other gaps remain
(progressive encoding, DC trellis, optimized Huffman implementation differences, etc.)

### 2025-12-25: Analysis of mozjpeg-rs (which matches C mozjpeg)

**Key differences identified between zenjpeg and mozjpeg-rs:**

| Factor | Zenjpeg | Mozjpeg-rs | Impact |
|--------|---------|-----------|--------|
| DCT Implementation | Float reference | Integer Loeffler + SIMD | +5-8% |
| Quantization | Float-based | Integer-based | +3-5% |
| Trellis | AC only | Full with cross-block DC | +5-10% |
| Progressive Mode | Implemented (minimal) | Full support | +10-15% |
| Huffman Optimization | Basic | Advanced with scan opt | +5-10% |
| Deringing | None | Enabled by default | +2-3% |

**Total potential gap: 30-50%** (we're seeing 80%, suggesting bugs or format overhead)

**Priority fixes:**
1. ~~Port integer Loeffler DCT from mozjpeg-rs~~ DONE
2. ~~Implement cross-block DC trellis~~ DONE
3. ~~Add progressive mode support~~ DONE
4. Implement scan optimization

### 2025-12-25: Quality Sweep Results

| Q | Size | DSSIM | BPP | Trellis Improvement |
|---|------|-------|-----|---------------------|
| 20 | 4277 | 0.007664 | 0.52 | - |
| 30 | 5985 | 0.003979 | 0.73 | 27.0% |
| 50 | 6973 | 0.002463 | 0.85 | 14.8% |
| 70 | 8416 | 0.002319 | 1.03 | 13.2% |
| 80 | 14678 | 0.000348 | 1.79 | - |
| 90 | 19128 | 0.000156 | 2.34 | 11.5% |
| 95 | 26405 | 0.000073 | 3.22 | - |

**Observations:**
- Trellis provides 11-27% improvement (higher at low quality)
- Notable quality jump between Q70 and Q80 (strategy changes)
- DSSIM is excellent across all quality levels

### 2025-12-25: DC Trellis Implementation

**Implementation**: Cross-block DC trellis optimization is now integrated.

**How it works**:
1. During encoding, store raw DCT coefficients (scaled by 8) alongside quantized blocks
2. After AC trellis quantization, run DC trellis optimization row by row
3. Each row is an independent chain (matching C mozjpeg behavior)
4. DC trellis uses dynamic programming to find optimal DC values across blocks

**Key code locations**:
- `src/encode.rs`: `run_dc_trellis_by_row()` - row-by-row optimization wrapper
- `src/encode.rs`: `quantize_block_with_raw()` - returns both quantized and raw DCT
- `src/trellis.rs`: `dc_trellis_optimize_indexed()` - core DP algorithm

**Results** (256x256 synthetic test image):

| Q | AC+DC Trellis | No Trellis | Combined Savings |
|---|---------------|------------|------------------|
| 30 | 5966 bytes | 7010 bytes | 14.9% |
| 50 | 6973 bytes | 8185 bytes | 14.8% |
| 70 | 8416 bytes | 9701 bytes | 13.2% |
| 90 | 19128 bytes | 21625 bytes | 11.5% |

**Note**: These results show combined AC + DC trellis vs no trellis. DC trellis
provides ~2-5% additional savings on top of AC trellis alone.

### 2025-12-25: Progressive Mode Implementation

**Implementation**: Minimal progressive scan script is now integrated.

**How it works**:
1. Uses SOF2 marker instead of SOF0 for progressive DCT
2. Generates scan script with `generate_minimal_progressive_scans()`
3. DC scan encodes all components interleaved
4. AC scans encode each component separately (1-63)
5. Uses ProgressiveEncoder for DC/AC first scans

**Key code locations**:
- `src/progressive.rs`: ScanInfo type and scan generation functions
- `src/entropy.rs`: ProgressiveEncoder for DC/AC encoding
- `src/encode.rs`: `encode_progressive_ycbcr()` and `encode_progressive_gray()`

**Scan script (minimal progressive)**:
1. DC scan: All 3 components, Ss=0, Se=0, Ah=0, Al=0
2. AC scan Y: Component 0, Ss=1, Se=63, Ah=0, Al=0
3. AC scan Cb: Component 1, Ss=1, Se=63, Ah=0, Al=0
4. AC scan Cr: Component 2, Ss=1, Se=63, Ah=0, Al=0

**Usage**:
```rust
let encoder = Encoder::new()
    .quality(Quality::Standard(75))
    .progressive(true);
```

**Next steps**:
- Implement successive approximation for better progressive loading
- Add scan optimization (custom scan scripts)
- Measure compression impact vs baseline

---

### Experiment 2: AQ Impact by Image Type

**Hypothesis**: AQ helps more on complex textures

**Method**:
1. Categorize images: simple, gradient, texture, photo
2. Encode at Q75 with AQ strengths [0, 0.5, 1.0]
3. Measure SSIMULACRA2

**Expected**: Photos benefit from AQ, simple images may not

**Results**: (pending)

---

### Experiment 3: Quality Curve Linearization

**Hypothesis**: Can find mapping for linear SSIM2 response

**Method**:
1. Encode test image at Q = 10, 20, ..., 100
2. Measure SSIMULACRA2 for each
3. Fit inverse: Q = f(target_ssim2)
4. Validate on other images

**Expected**: Need exponential or similar mapping

**Results**: (pending)

---

## Critical Implementation Notes

### The 8x DCT Scaling Factor (CRITICAL - READ THIS)

**Date**: 2025-12-25

**Problem**: Trellis quantization produces catastrophically bad quality (DSSIM 0.42 instead of <0.01).

**Root Cause**: mozjpeg's trellis expects DCT coefficients "scaled by 8". This means:

```
mozjpeg trellis formula:
  q = 8 * qtable[z]           // Quantizer is multiplied by 8
  qval = (dct_coeff + q/2) / q  // Division by 8*qtable
```

If you pass unscaled DCT coefficients, you get:
- Effective quantization = DCT / (8 * qtable)
- This is 8x stronger than intended!
- Results in ~8x over-quantization, destroying image quality

**Solution**: Scale DCT by 8 before passing to trellis:

```rust
// CORRECT - scale DCT by 8 for trellis
for i in 0..64 {
    dct_scaled[i] = (dct[i] * 8.0).round() as i32;
}
trellis_quantize_block(&dct_scaled, ...);
```

**Why mozjpeg does this**: It's an integer optimization. By pre-scaling DCT values and using
integer arithmetic throughout, mozjpeg avoids floating-point division in the inner loop.
The `8 * qtable` in the formula exactly cancels out the 8x scaling of the DCT.

**Symptoms if you get this wrong**:
- DSSIM of 0.4+ instead of <0.01 at Q70
- Files that are way too small (over-compressed)
- Images that look terrible but technically decode
- BOTH trellis and non-trellis paths produce identical bad output (because the
  8x is baked into the quant formula, not the table)

**How to verify you have it right**:
1. Encode a test image at Q70 with trellis enabled
2. DSSIM should be < 0.05 (ideally < 0.01)
3. If DSSIM > 0.1, you probably have the scaling wrong

---

## Red Herrings & Failed Approaches

### 2025-12-25 - Initial trellis integration used wrong DCT scale

**Hypothesis**: Just convert f32 DCT to i32 and pass to trellis.

**What we tried**: `dct_i32[i] = dct[i] as i32`

**Result**: DSSIM of 0.42 at Q70 (catastrophic quality loss)

**Root cause**: mozjpeg trellis expects DCT * 8. See "The 8x DCT Scaling Factor" above.

**Lesson**: When porting code that uses fixed-point arithmetic, ALWAYS check the scaling
factors. Comments like "scaled by 8" are CRITICAL implementation details, not noise.

---

## References

- mozjpeg: trellis quantization paper/code
- jpegli: https://github.com/libjxl/libjxl (jpegli component)
- SSIMULACRA2: perceptual quality metric
- Butteraugli: Google's perceptual distance metric
- codec-corpus: test image collection
- imageflow chromaeval: image analysis for encoding decisions

---

---

## Corpus Reduction Strategy

### Goal

Reduce test corpus size while maintaining coverage of unique encoding behaviors.

### Method

1. Encode each image in corpus at Q = [20, 40, 60, 80, 95]
2. For each image, compute curve: `[(Q, bpp, DSSIM), ...]`
3. Cluster images by curve similarity
4. From each cluster, keep the **smallest** image that produces the same curve shape
5. Result: minimal corpus that covers all unique encoding behaviors

### Benefits

- Faster benchmarks
- Less storage
- Same coverage of edge cases
- Representative samples for each "image type"

### Implementation

**Tool**: `examples/corpus_reduction.rs`

**Usage**:
```bash
cargo run --release --example corpus_reduction -- <corpus_dir> [output_dir]
```

**Example**:
```bash
cargo run --release --example corpus_reduction -- ../codec-eval/codec-corpus/kodak reduced_corpus
```

### Results (Kodak corpus)

| Original | Reduced | Size Reduction |
|----------|---------|----------------|
| 24 images | 9 representatives | 63.3% |

**Clusters found**:
- Cluster 1: Low BPP images (20.png, 12.png, 3.png)
- Cluster 2: Medium-low BPP (16.png, 15.png, 2.png, 4.png)
- Cluster 6: Medium-high BPP (6.png, 11.png, 19.png, 21.png, 22.png)
- Cluster 7: High BPP (14.png, 1.png, 18.png, 24.png)
- Cluster 8: Very high BPP (5.png, 8.png)

### Curve Similarity Metric

```
distance = mean over Q: |bpp1 - bpp2| / avg_bpp + |dssim1 - dssim2| / avg_dssim

Threshold: < 0.10 (10% normalized difference)
```

---

## Code Locations

- Strategy selection: `src/strategy.rs`
- Trellis: `src/trellis.rs`
- AQ: `src/adaptive_quant.rs`
- Quality types: `src/types.rs` (Quality enum)
- Encoder: `src/encode.rs`
