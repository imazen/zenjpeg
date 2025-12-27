# Encoder Selection Strategy

## Summary

Based on extensive research with codec-eval across CLIC 2025 and Kodak corpora (382 significant comparisons), this document outlines the optimal encoder selection strategy for zenjpeg.

## Key Findings

### 1. Encoder Performance by Metric

| Metric | mozjpeg wins | jpegli wins | Best Strategy |
|--------|-------------|-------------|---------------|
| **Butteraugli** | 16% | **84%** | Default to jpegli |
| **DSSIM** | **67%** | 33% | Default to mozjpeg |
| **SSIMULACRA2** | 51% | 49% | Context-dependent |

**Implication**: The "best" encoder depends on which perceptual metric you prioritize.

### 2. BPP Crossover Points

At matched file size (BPP), encoder preference changes:

| BPP Range | Butteraugli Winner | DSSIM Winner | SSIMULACRA2 Winner |
|-----------|-------------------|--------------|-------------------|
| < 0.3 | jpegli (91%) | jpegli (87%) | jpegli |
| 0.3-0.6 | jpegli (60%) | mozjpeg (45-81%) | balanced |
| 0.6-1.0 | jpegli (65%) | **mozjpeg (93-98%)** | mozjpeg |
| 1.0-2.0 | jpegli (95%) | **mozjpeg (71-91%)** | mozjpeg |
| > 2.0 | jpegli (96%) | mozjpeg (43-71%) | jpegli |

### 3. Image Characteristics

The only category where mozjpeg consistently wins is:
- **Very flat images** (>75% flat blocks)
- **Low complexity** (edge + contrast < 20)
- **Target BPP 0.35-0.6** (Q60-Q80 approximately)

For all other cases, jpegli wins 84%+ of the time.

## Prediction Model (Implemented)

The research-validated prediction model in `src/analysis.rs` achieves **86.6% accuracy**:

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

## Speed Considerations: Try-and-Fallback vs Heuristic-First

### Question: Is jpegli fast enough to try-and-fallback?

**Short answer**: No, try-and-fallback doubles encoding time for marginal benefit.

**Analysis**:
- The prediction model achieves 86.6% accuracy
- "Always use jpegli" achieves 84% accuracy
- The prediction adds only 2.6 percentage points of improvement
- Running both encoders doubles encoding time

**Recommendation**: Run the heuristic BEFORE encoding. The analysis is fast (samples every 4th block/pixel) and provides nearly optimal selection. Only 14-16% of images would benefit from a different encoder, and the difference is often marginal.

### Heuristic Cost

The image analysis in `analyze_image()` is very fast:
- Samples every 4th block for variance
- Samples every 4th pixel for edges
- Total: ~1/16th of full image scan
- Negligible compared to encoding time

## Quality Value Mapping

### Problem: Q Values Mean Different Things

mozjpeg Q85 and jpegli Q85 produce different file sizes AND quality levels:
- Quality scales are non-linear
- No simple "jpegli Q90 = mozjpeg Q85" mapping exists

### Solution: Target Perceptual Quality

Instead of mapping Q values directly, use target perceptual quality:

```rust
pub enum Quality {
    Standard(u8),        // Standard 1-100 scale
    Perceptual(f32),     // Target Butteraugli distance
    TargetBpp(f32),      // Target bits per pixel
    TargetSize(usize),   // Binary search for target size
}
```

### Butteraugli Distance Mapping

jpegli's quality-to-distance formula (from `src/analysis.rs`):

```rust
pub fn quality_to_distance(quality: f32) -> f32 {
    let q = quality as i32;
    if q >= 100 {
        0.01
    } else if q >= 30 {
        0.1 + (100 - q) as f32 * 0.09
    } else {
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}
```

### Recommended Quality Curve

For a "smooth sane curve of perceivable quality":

| User Q | Butteraugli Distance | Perceptual Quality |
|--------|---------------------|-------------------|
| 95 | ~0.55 | Visually lossless |
| 90 | ~1.0 | Excellent |
| 85 | ~1.45 | Very good |
| 80 | ~1.9 | Good |
| 75 | ~2.35 | Acceptable |
| 70 | ~2.8 | Noticeable artifacts |

## Implementation TODO

1. **Add metric-specific strategies**: Allow users to specify which metric to optimize for
   ```rust
   encoder.optimize_for(Metric::Butteraugli)  // Default
   encoder.optimize_for(Metric::DSSIM)        // For structural fidelity
   encoder.optimize_for(Metric::SSIMULACRA2)  // Balanced
   ```

2. **Implement perceptual quality targeting**:
   - Binary search on quality parameter
   - Use fast butteraugli approximation during search
   - Fall back to standard Q if no good match found

3. **Tune jpegli's adaptive quantization**:
   - Current implementation is simplified
   - Full port of C++ algorithm may improve results

4. **Speed profiling**:
   - Measure actual encoding times for zenjpeg vs reference implementations
   - Optimize hot paths if try-and-fallback becomes desirable

## Conclusion

The research strongly supports:
1. **Default to jpegli** for Butteraugli-optimized encoding (84%+ win rate)
2. **Use heuristic-first** strategy (fast, 86.6% accurate)
3. **Offer metric selection** for users who prioritize structural fidelity (DSSIM)
4. **Target perceptual quality** instead of mapping Q values

The current implementation in `src/analysis.rs` and `src/strategy.rs` follows these recommendations.

## Date

2025-12-27
