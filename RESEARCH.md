# zenjpeg Research Notes

This document tracks research findings, experiments, and decisions about combining mozjpeg and jpegli techniques.

## Current State (2024-12-26)

**zenjpeg now has a forked jpegli encoder** in `src/jpegli/` that produces byte-identical output to the reference jpegli-rs. This enables experimentation with improvements.

### Immediate Improvement Targets

1. **Adaptive Quantization Tuning** - `src/jpegli/adaptive_quant.rs`
   - Current: Full C++ algorithm ported (pre-erosion, fuzzy erosion, modulations)
   - Opportunity: Tune constants for better Pareto, or try simpler algorithms

2. **Zero-Bias Parameters** - `src/jpegli/quant.rs`
   - Current: Fixed tables from C++ jpegli
   - Opportunity: Content-adaptive zero-bias, quality-adaptive offset

3. **Quantization Tables** - `src/jpegli/quant.rs`
   - Current: jpegli's perceptual matrices with frequency exponents
   - Opportunity: Tune exponents, learn from corpus

4. **Hybrid Trellis + XYB** - Combine mozjpeg's trellis with jpegli's perceptual model

## Key Finding: Quality-Dependent Pareto Front

Based on analysis of mozjpeg-rs and jpegli-rs performance:

| Quality Range | Better Encoder | Why |
|---------------|----------------|-----|
| Q < 50 | mozjpeg | Trellis excels when many coefficients are zeroed |
| Q 50-70 | Hybrid | Both techniques contribute |
| Q >= 70 | jpegli | Adaptive quant preserves detail where it matters |

## mozjpeg Techniques (from mozjpeg-rs CLAUDE.md)

### Trellis Quantization
- Rate-distortion optimization using dynamic programming
- For each coefficient, considers {q-1, q, q+1} candidates
- Picks candidate with minimum (rate + λ × distortion)
- AC trellis: independent per block
- DC trellis: depends on previous block (differential encoding)

**Lambda calculation:**
```
lambda = 2^scale1 / (2^scale2 + block_norm)
```

**Performance:**
- mozjpeg-rs trellis is 10% FASTER than C mozjpeg at 2048x2048
- Baseline (no trellis) is 4.7x slower than C (entropy encoding bottleneck)

### Progressive Encoding
- DC first, then AC in bands
- Successive approximation for refinement
- ~2-3% smaller files at same quality

### Huffman Optimization
- 2-pass encoding: gather statistics, build optimal tables
- ~3-4% smaller files

## jpegli Techniques (from jpegli-rs CLAUDE.md)

### Adaptive Quantization
- Per-block quantization multipliers based on local variance
- High detail → lower multiplier → preserve detail
- Low detail → higher multiplier → compress more

**Algorithm:**
1. `PerBlockModulations()` - compute block features
2. `ComputePreErosion()` - find local minima
3. `FuzzyErosion()` - smooth with weighted 3x3 minimum
4. `ComputeAdaptiveQuantField()` - final multiplier map

**Current Status in jpegli-rs:**
- Using constant aq_strength=0.08 (calibrated mean)
- Full algorithm not yet fully ported

### XYB Color Space
- Perceptually uniform color space from JPEG XL
- Better for high-quality encoding
- Requires ICC profile for compatibility

### Zero-Bias Tables
- Frequency-dependent thresholds for zeroing small coefficients
- Constants match C++ exactly

## What Works

### Confirmed Effective
1. **Trellis for low Q**: Significant size reduction when many zeros
2. **AQ for high Q**: Preserves detail without wasting bits on flat areas
3. **Huffman optimization**: Always beneficial, ~3-4% smaller
4. **4:4:4 subsampling at high Q**: Quality improvement outweighs size

### Diminishing Returns
1. **DC trellis at high Q**: Complex, small benefit
2. **Progressive at high Q**: Less benefit when file is larger
3. **Trellis at Q > 85**: Most coefficients non-zero anyway

## What Doesn't Work

### Investigated and Rejected

1. **DCT 1/64 scaling**
   - jpegli C++ comments mention 1/8 per dimension
   - Rust uses 1/8 total, which is correct for decoder compatibility
   - Changing to 1/64 breaks decoded images (8x too dark)

2. **MSE/PSNR for quality comparison**
   - Doesn't correlate with perceptual quality
   - Use DSSIM or SSIMULACRA2 instead

3. **Simple quality mapping**
   - "jpegli Q90 = mozjpeg Q85" doesn't hold
   - Quality scales differ non-linearly
   - Better to target perceptual quality directly

## Pareto Strategy

The goal is to combine techniques so zenjpeg dominates both encoders:

```
Rate-Distortion Curve

Quality
   │
   │     ┌──── zenjpeg (goal)
   │    /
   │   /    ┌── jpegli
   │  / ───/
   │ /
   │/─────── mozjpeg
   └─────────────────── Size
```

### Implementation Plan

1. **Phase 1: Baseline**
   - Delegate to mozjpeg-rs or jpegli based on quality
   - Simple strategy selection

2. **Phase 2: Integration**
   - Use jpegli's AQ with mozjpeg's quantization
   - Use mozjpeg's trellis with jpegli's quality scaling

3. **Phase 3: Optimization**
   - Profile and tune crossover points
   - Per-image adaptive selection

## Experiments TODO

### Crossover Point Study
- [ ] Encode corpus at Q30-Q100 in steps of 5
- [ ] Measure DSSIM and file size for both encoders
- [ ] Find exact crossover quality for each image type

### Hybrid Strategy Tuning
- [ ] Test trellis + AQ combinations
- [ ] Vary trellis lambda with AQ strength
- [ ] Find complementary settings

### Per-Image Adaptation
- [ ] Analyze image features (variance, edges, flat areas)
- [ ] Predict best strategy from features
- [ ] Train classifier if needed

## Quality Metric Thresholds

From codec-eval documentation:

| DSSIM | Perception |
|-------|------------|
| < 0.0003 | Imperceptible |
| < 0.0007 | Marginal |
| < 0.0015 | Subtle |
| < 0.003 | Noticeable |
| >= 0.003 | Degraded |

## References

### Papers
- [JPEG AI evaluation methodology](https://jpeg.org/items/20220415_cfp_jpeg_ai.html)
- [Butteraugli perceptual metric](https://github.com/google/butteraugli)
- [SSIMULACRA2 paper](https://cloudinary.com/blog/ssimulacra2)

### Code
- mozjpeg-rs: `/home/lilith/work/mozjpeg-rs`
- jpegli-rs: `/home/lilith/work/jpegli/jpegli-rs`
- codec-eval: `/home/lilith/work/codec-eval`
- codec-corpus: `/home/lilith/work/codec-eval/codec-corpus`

### Test Data
- C++ instrumented outputs: `/home/lilith/work/jpegli/*.testdata`
- Kodak corpus: `codec-corpus/kodak/`
- CID22 dataset: See jpegli-rs CLAUDE.md

## Session Log

### 2024-12-25: Initial Setup
- Created project structure
- Implemented basic encoding pipeline
- 20 tests passing
- Strategy selection based on quality
- Placeholder trellis and AQ implementations

### 2024-12-26: Outlier Analysis with Butteraugli

**Created `find-outliers` tool** in `codec-eval/crates/codec-compare/` to scan corpora and identify encoder-sensitive images.

**Kodak corpus results (24 images, Q50/70/85/95):**

| Metric | C jpegli | C mozjpeg |
|--------|----------|-----------|
| Wins (>5% advantage) | 100% | 0% |
| Avg Butteraugli @ Q85 | 1.8 | 2.8 |
| Avg Butteraugli @ Q95 | 1.0 | 2.0 |

**Key findings:**
1. jpegli dominates on Butteraugli quality at ALL quality levels tested
2. At Q95, jpegli achieves ~50% lower Butteraugli at similar file size
3. The advantage increases with quality level
4. No images in Kodak favor mozjpeg on perceptual quality

**Example: kodak/5.png**
```
Q  | moz bpp  butter | jpegli bpp  butter
50 |   1.07    4.47  |   1.46      3.31
70 |   1.49    3.75  |   1.94      2.73
85 |   2.25    2.93  |   2.75      1.76
95 |   3.75    2.65  |   4.64      0.96
```
At Q95: jpegli is 24% larger but has 64% better perceptual quality!

**Implications for zenjpeg:**
1. For quality-optimized encoding, jpegli wins decisively
2. For size-constrained encoding at low quality, mozjpeg may win (needs more testing)
3. The "hybrid" strategy may not be needed - jpegli dominates across the board
4. Focus improvements on jpegli's approach, not mozjpeg's trellis

**Low quality test (Q30-Q60):**
```
Q  | moz bpp  butter | jpegli bpp  butter | Efficiency ratio
30 |   0.29    5.73  |   0.57      3.91   | jpegli 1.48x better
40 |   0.36    4.82  |   0.61      3.56   | jpegli 1.32x better
50 |   0.43    4.10  |   0.67      3.39   | jpegli 1.18x better
60 |   0.51    3.97  |   0.76      2.98   | jpegli 1.27x better
```

**CORRECTION: Same-Q comparison was misleading!**

Q values don't mean the same thing across encoders. Created `rd-compare` tool to compare at matched file sizes (fair comparison).

**Rate-Distortion comparison at matched BPP (Kodak, 24 images):**

| BPP | mozjpeg wins | jpegli wins |
|-----|--------------|-------------|
| 0.30 | 0% | **100%** |
| 0.50 | **58%** | 42% |
| 0.75 | **67%** | 33% |
| 1.00 | 29% | **71%** |
| 1.50 | 12% | **88%** |
| 2.0+ | 4% | **96%** |

**Crossover point: ~0.75-1.0 bpp**
- Below 0.75 bpp: mozjpeg's trellis wins (high compression)
- Above 1.0 bpp: jpegli's AQ wins (moderate compression)

This aligns with the research notes about quality-dependent Pareto fronts!

**Corrected strategy for zenjpeg:**
1. Use mozjpeg approach for bpp < 0.75 (or Q < ~60)
2. Use jpegli approach for bpp > 1.0 (or Q > ~70)
3. Hybrid zone at 0.75-1.0 bpp

**CID22 corpus (20 images) confirms pattern:**

| BPP | mozjpeg wins | jpegli wins |
|-----|--------------|-------------|
| 0.50 | 50% | 50% |
| 0.75 | **55%** | 45% |
| 1.00 | 35% | **65%** |
| 1.50 | 20% | **80%** |
| 2.00 | 30% | **70%** |

**Confirmed: crossover at ~0.75-1.0 bpp across both Kodak and CID22**

### Image Characteristic Analysis (2024-12-26)

Analyzed Kodak images to find what characteristics favor each encoder:

**mozjpeg-favoring images (at low bpp):**
| Image | Flat% | Edges | Detail% |
|-------|-------|-------|---------|
| 23.png | **81%** | 9 | 4% |
| 12.png | **75%** | 11 | 7% |
| 2.png | **74%** | 11 | 2% |
| 3.png | **73%** | 10 | 4% |

**jpegli-favoring images (even at low bpp):**
| Image | Flat% | Edges | Detail% |
|-------|-------|-------|---------|
| 8.png | 21% | **38** | **40%** |
| 13.png | 10% | **38** | **27%** |
| 5.png | 22% | **31** | **28%** |
| 1.png | 22% | **30** | **20%** |

**Detection heuristic:**
```
if flat_pct > 70% && edge_strength < 15:
    use mozjpeg for bpp < 1.0
elif flat_pct < 30% && edge_strength > 25:
    use jpegli even at low bpp
else:
    use jpegli for bpp > 0.75
    use mozjpeg for bpp < 0.75
```

**Why this works:**
- **Flat images**: mozjpeg's trellis efficiently zeros coefficients
- **Complex images**: jpegli's AQ allocates bits to detail areas
- **Mixed images**: crossover at ~0.75-1.0 bpp

**Next steps:**
- [x] Test on more diverse corpus (CID22) - Done, confirms pattern
- [x] Analyze image characteristics for detection heuristics - Done
- [x] Build encoder prediction model with 86.6% accuracy
- [ ] Implement content-adaptive strategy selection
- [ ] Add flat% and edge detection to zenjpeg encoder

### 2024-12-27: Encoder Selection Model

**Methodology:**
Created comprehensive encoder comparison tools in `codec-eval/crates/codec-compare/`:
1. `full-comparison` - Encodes images at Q25-90, computes butteraugli/dssim/ssimulacra2
2. `image-heuristics` - Extracts 25+ image characteristics
3. `build-predictor` - Evaluates prediction rules against actual performance

**Dataset:**
- CLIC 2025 validation: 32 images × 14 quality levels × 2 encoders = 896 encodes
- Kodak: 24 images × 14 quality levels × 2 encoders = 672 encodes
- Combined: 56 images, 382 significant BPP-matched comparisons

**Key Findings (BPP-based, 5% margin threshold):**

| BPP | mozjpeg wins | jpegli wins | % jpegli |
|-----|--------------|-------------|----------|
| 0.2 | 5 | 49 | **91%** |
| 0.4 | 16 | 28 | 64% |
| 0.6 | 17 | 26 | 60% |
| 0.8 | 13 | 29 | 69% |
| 1.0+ | 10 | 189 | **95%** |

**Best Prediction Rule (combined_v13, 86.6% accuracy):**
```rust
fn predict_encoder(flat_block_pct: f64, edge_strength: f64,
                   local_contrast: f64, target_bpp: f64) -> &'static str {
    let complexity = edge_strength + local_contrast;
    let uniformity = flat_block_pct;

    // Very flat, low complexity, 0.35-0.6 bpp is mozjpeg territory
    if uniformity > 75.0 && complexity < 20.0 && target_bpp >= 0.35 && target_bpp < 0.6 {
        "mozjpeg"
    } else {
        "jpegli"
    }
}
```

**Accuracy by Corpus:**
- CLIC 2025: 87.1%
- Kodak: 85.9%
- Combined: 86.6%

**Image Category Analysis:**
| Category | mozjpeg wins | jpegli wins | % jpegli |
|----------|--------------|-------------|----------|
| very_flat_low_bpp | **15** | 7 | 32% |
| flat_low_bpp | 6 | 33 | 85% |
| flat_high_bpp | 26 | 90 | 78% |
| mixed_* | 6 | 53 | 90% |
| complex_* | 6 | 61 | 91% |

**Key Insight:** The only category where mozjpeg dominates is very_flat_low_bpp (>80% flat blocks, BPP < 0.6).

**Rule Simplification:**
Given that "always use jpegli" achieves 84% accuracy, the prediction rule adds only ~2.6 percentage points. For simplicity, zenjpeg could default to jpegli for all cases except:
- Targeting < 0.6 bpp AND
- Image is very flat (>75% flat blocks) AND
- Low complexity (edge + contrast < 20)

**Next Steps for zenjpeg:**
1. Implement fast flat-block and edge detection during encoding
2. Use prediction model to select encoder strategy
3. For hybrid mode: use mozjpeg's quantization with jpegli's encoding

### 2024-12-27: Multi-Metric Analysis

**Critical Finding: The optimal encoder depends heavily on which quality metric you care about.**

Extended `build-predictor` to evaluate winners using all three metrics:

| Metric | mozjpeg wins | jpegli wins | Best Rule | Accuracy |
|--------|-------------|-------------|-----------|----------|
| **Butteraugli** | 61 (16%) | 321 (84%) | combined_v13 | 86.6% |
| **DSSIM** | 264 (67%) | 131 (33%) | combined_v11 | 52.4% |
| **SSIMULACRA2** | 109 (51%) | 104 (49%) | combined_v12 | 60.1% |

**Analysis by Metric:**

**Butteraugli (jpegli-favoring):**
- jpegli wins 84% of comparisons
- Dominates at very low BPP (0.2: 91%) and high BPP (1.0+: 95%)
- mozjpeg only wins in very_flat_low_bpp category (68%)
- *Why:* Butteraugli was developed by Google (same team as jpegli) and shares perceptual assumptions with jpegli's XYB color space

**DSSIM (mozjpeg-favoring):**
- mozjpeg wins 67% of comparisons
- Dominates in the 0.6-1.5 BPP range (82-98% win rate)
- jpegli only wins at very low BPP (0.2: 87%) and very high BPP (3.0: 58%)
- *Why:* SSIM is based on structural similarity - mozjpeg's trellis optimization preserves structure better

| BPP (DSSIM) | mozjpeg wins | jpegli wins | % mozjpeg |
|-------------|--------------|-------------|-----------|
| 0.2 | 7 | 48 | 13% |
| 0.4 | 24 | 29 | 45% |
| 0.6 | 44 | 10 | **81%** |
| 0.8 | 51 | 4 | **93%** |
| 1.0 | 50 | 1 | **98%** |
| 1.5 | 41 | 4 | **91%** |
| 2.0 | 30 | 12 | **71%** |
| 3.0 | 17 | 23 | 43% |

**SSIMULACRA2 (balanced):**
- Nearly tied: mozjpeg 51%, jpegli 49%
- mozjpeg wins at medium BPP (0.6-1.0 range)
- jpegli wins at low BPP (0.2-0.4) and high BPP (2.0+)
- *Why:* SSIMULACRA2 is a blend of approaches and shows more balanced results

**Implications for zenjpeg:**

1. **The current prediction model (combined_v13) is only optimized for Butteraugli.**
   - If targeting DSSIM, should use mozjpeg more often (especially 0.6-1.5 BPP)
   - If targeting SSIMULACRA2, need a different strategy (metric-adaptive)

2. **Metric-specific strategies:**
   ```
   Butteraugli: Use jpegli except for very flat images at 0.35-0.6 BPP
   DSSIM:       Use mozjpeg for 0.5-2.0 BPP, jpegli for extremes
   SSIMULACRA2: Use mozjpeg for 0.5-1.5 BPP, jpegli otherwise
   ```

3. **Practical recommendation:**
   - For web images (perceptual quality): Use butteraugli-based strategy (jpegli-favoring)
   - For archival/medical (structural fidelity): Use DSSIM-based strategy (mozjpeg-favoring)
   - For general use: Use SSIMULACRA2-based strategy (balanced)

4. **Why butteraugli favors jpegli (even without XYB mode):**
   - **Note:** jpegli was tested in standard YCbCr mode, NOT XYB mode
   - jpegli's adaptive quantization is tuned for perceptual quality
   - jpegli's zero-bias tables and quantization matrices may be optimized for butteraugli-like metrics
   - mozjpeg's trellis optimizes for rate-distortion (bit savings), not perceptual quality
   - This suggests jpegli's perceptual tuning works even in YCbCr mode

5. **Future work: Test with XYB mode enabled**
   - jpegli supports XYB color space encoding
   - XYB mode may further improve butteraugli scores
   - Would require ICC profile handling for proper decoding

### 2024-12-27: Three-Encoder Comparison (mozjpeg vs jpegli vs jpegli-xyb)

**Added jpegli-xyb encoder** to the comparison framework. This required:
1. Updating `full-comparison.rs` to add `encode_jpegli_xyb` using `.use_xyb(true)`
2. Using `jpegli::icc::decode_jpeg_with_icc()` for proper XYB decoding (the `Decoder::apply_icc(true)` method has bugs)
3. Updating codec-compare to use local jpegli-rs crate with `cms-lcms2` feature

**Kodak corpus results (24 images, Q30-Q95, BPP-matched comparison):**

**Butteraugli (lower is better):**
| BPP | mozjpeg | jpegli | jpegli-xyb |
|-----|---------|--------|------------|
| 0.50 | 12.5% | **58.3%** | 29.2% |
| 0.75 | 16.7% | **66.7%** | 16.7% |
| 1.00 | 8.3% | **79.2%** | 12.5% |
| 1.50 | 0.0% | **91.7%** | 8.3% |
| 2.00 | 4.2% | **91.7%** | 4.2% |

**DSSIM (lower is better):**
| BPP | mozjpeg | jpegli | jpegli-xyb |
|-----|---------|--------|------------|
| 0.50 | **41.7%** | 20.8% | 37.5% |
| 0.75 | **75.0%** | 8.3% | 16.7% |
| 1.00 | **95.8%** | 0.0% | 4.2% |
| 1.50 | **95.8%** | 4.2% | 0.0% |
| 2.00 | **91.7%** | 8.3% | 0.0% |

**SSIMULACRA2 (higher is better):**
| BPP | mozjpeg | jpegli | jpegli-xyb |
|-----|---------|--------|------------|
| 0.50 | 29.2% | 12.5% | **58.3%** |
| 0.75 | **54.2%** | 16.7% | 29.2% |
| 1.00 | **62.5%** | 25.0% | 12.5% |
| 1.50 | **54.2%** | 41.7% | 4.2% |
| 2.00 | 37.5% | **58.3%** | 4.2% |

**Key Findings:**

1. **Butteraugli:** jpegli (YCbCr mode) dominates. XYB mode doesn't improve butteraugli scores and actually performs worse at higher BPP.

2. **DSSIM:** mozjpeg dominates at medium-high BPP (75-96%). At very low BPP (0.5), jpegli-xyb is competitive with mozjpeg (38% vs 42%).

3. **SSIMULACRA2:** jpegli-xyb wins at low BPP (58% at 0.5 bpp). mozjpeg wins at medium BPPs. jpegli wins at high BPPs.

4. **XYB mode is NOT universally better:** Despite using a perceptually uniform color space, jpegli-xyb underperforms regular jpegli on butteraugli. This suggests jpegli's YCbCr-specific optimizations are well-tuned.

5. **File size penalty:** At same quality level, jpegli-xyb produces ~10-15% larger files than jpegli YCbCr mode (and ~40-60% larger than mozjpeg).

**Implications for zenjpeg:**

1. **For butteraugli targets:** Use jpegli YCbCr (not XYB)
2. **For DSSIM targets:** Use mozjpeg at 0.5-2.0 BPP
3. **For SSIMULACRA2 targets at low BPP:** Consider jpegli-xyb
4. **XYB mode may be useful for:** Very low bitrate encoding where perceptual color fidelity matters more than structural fidelity

**Technical note:** The XYB decoding in the comparison uses `jpegli::icc::decode_jpeg_with_icc()` which uses jpeg-decoder + lcms2 for proper ICC profile application. The native jpegli-rs Decoder with `apply_icc(true)` has bugs that cause decode failures on XYB streams.
