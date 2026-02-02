# Context Handoff: Hybrid Trellis Optimization

## Current State

Hybrid trellis is **working but not Pareto-optimal**. It trades quality for size - no configuration found that gives both smaller files AND equal/better quality.

### CID22 Benchmark Results (20 images, Butteraugli)

| Q | Jpegli Size | Jpegli BA | Hybrid Size | Hybrid BA | ΔSize% | ΔBA% |
|---|-------------|-----------|-------------|-----------|--------|------|
| 75 | 40971 | 2.596 | 39232 | 2.643 | -4.2% | +1.8% |
| 80 | 46213 | 2.359 | 44456 | 2.413 | -3.8% | +2.3% |
| 85 | 53913 | 2.095 | 52124 | 2.151 | -3.3% | +2.7% |
| 90 | 67075 | 1.853 | 65324 | 1.874 | -2.6% | +1.1% |
| 95 | 93437 | 1.695 | 91786 | 1.716 | -1.8% | +1.2% |

Trade-off: ~1% size reduction per ~0.8% butteraugli degradation.

## What's Been Tried

### 1. Fixed Negative Coupling (-4.0)
- Result: -9.2% size, +10.9% butteraugli
- Problem: Over-quantizes high-texture images (up to +45% butteraugli on some)

### 2. Texture-Adaptive Coupling (current)
- Formula: `coupling = -4.0 × (0.15 / max(aq_mean, 0.15))`
- Result: -3.3% size, +2.7% butteraugli
- Better balance but still not Pareto-optimal

### 3. Screenshot Protection (max_adjustment=1.0)
- Works well for mixed content
- Prevents catastrophic quality loss on UI/text

## Approaches to Explore

### A. Quality-Neutral Mode
**Goal:** Same butteraugli as jpegli, smaller file.

**Approach:**
1. Use hybrid trellis to compress more aggressively
2. Boost overall quality slightly to compensate for butteraugli loss
3. If hybrid at Q85 = jpegli Q85 - 3% size + 3% butteraugli,
   then hybrid at Q86-87 might = jpegli Q85 quality at smaller size

**Implementation:**
```rust
// Estimate quality boost needed to offset butteraugli degradation
let quality_boost = butteraugli_delta_percent * 0.5; // rough estimate
let adjusted_quality = base_quality + quality_boost;
```

### B. Butteraugli-Targeted Encoding
**Goal:** Hit a specific butteraugli target with minimum file size.

**Approach:**
1. Encode with hybrid at initial quality guess
2. Measure butteraugli
3. Binary search quality to hit target butteraugli
4. Compare final size to jpegli at same butteraugli

**Files:** Could extend `EncoderConfig` with `.target_butteraugli(f32)` mode.

### C. Per-Block Quality Redistribution
**Goal:** Pareto improvement by smarter bit allocation.

**Approach:** Instead of uniformly reducing quality on textured blocks:
1. Identify blocks where quality loss is imperceptible (high masking)
2. Steal bits from those blocks
3. Give bits to blocks where quality loss is visible (low masking)
4. Net result: same total bits, better perceptual quality

**This is fundamentally different from current approach** which just adjusts lambda uniformly.

### D. Positive Coupling with Size Target
**Goal:** Better quality at same file size.

**Approach:**
1. Use positive coupling (preserves texture detail)
2. Reduce base quality to hit same file size as jpegli
3. Net result: same size, potentially better quality on textured areas

**Test:** Compare hybrid Q82 with positive coupling vs jpegli Q85.

### E. Frequency-Selective Trellis
**Goal:** Optimize different DCT frequencies differently.

**Approach:**
1. Be aggressive on high frequencies (less perceptually important)
2. Be conservative on low frequencies (more visible)
3. Current trellis treats all AC coefficients similarly

**Files:** `trellis/ac.rs` - would need per-frequency lambda scaling.

### F. Learn Optimal Coupling from Data
**Goal:** Find coupling that minimizes butteraugli for a given size budget.

**Approach:**
1. Run exhaustive sweep on training set (CID22)
2. For each image, find coupling that minimizes butteraugli at fixed size
3. Correlate optimal coupling with image statistics (mean, std, etc.)
4. Build predictive model

## Key Files

| File | Purpose |
|------|---------|
| `zenjpeg/src/hybrid/config.rs` | HybridConfig, adaptive_config(), presets |
| `zenjpeg/src/encode/search.rs` | ExpertConfig, parameter routing |
| `zenjpeg/src/trellis/ac.rs` | Trellis quantization algorithm |
| `zenjpeg/src/quant/aq/mod.rs` | AQ strength computation |
| `zenjpeg/examples/cid22_hybrid_bench.rs` | Single-quality benchmark |
| `zenjpeg/examples/cid22_pareto.rs` | Multi-quality Pareto analysis |
| `zenjpeg/examples/hybrid_parameter_sweep.rs` | Parameter exploration |

## Commands

```bash
# Run Pareto analysis
cargo run --release --example cid22_pareto

# Run single-quality benchmark
cargo run --release --example cid22_hybrid_bench

# Run parameter sweep on single image
cargo run --release --example hybrid_parameter_sweep -- path/to/image.png

# Run all hybrid config tests
cargo test --release -p zenjpeg --lib -- hybrid::config::tests
```

## Questions to Answer

1. **Is there a coupling formula that achieves Pareto improvement?**
   - Current texture-adaptive is close but not quite

2. **Should hybrid be a "size-optimized mode" rather than default?**
   - Clear trade-off: -3% size for +2-3% butteraugli
   - Some users may prefer this explicitly

3. **Can per-block bit redistribution (approach C) achieve true Pareto improvement?**
   - This is fundamentally different from lambda adjustment
   - Would require significant trellis algorithm changes

4. **Is butteraugli the right metric?**
   - DSSIM showed different results (hybrid sometimes won)
   - Different metrics optimize for different artifacts

## Session Summary

### Commits Made
- `f310bc5` feat(hybrid): add image type auto-detection based on AQ statistics
- `382cc99` docs: update CLAUDE.md with hybrid auto-detection feature
- `13814d4` feat(hybrid): add texture-adaptive coupling for better quality
- `41e5504` docs: update CLAUDE.md with texture-adaptive coupling results

### Tests
- 22 hybrid config unit tests (all pass)
- 606 total lib tests (all pass)

### Key Finding
Hybrid trellis shifts the rate-distortion curve but doesn't dominate it. It's a valid "size-optimized" mode but not a universal improvement over jpegli baseline.
