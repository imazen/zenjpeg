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

Next: Integration tests with codec-eval, benchmark against reference encoders
