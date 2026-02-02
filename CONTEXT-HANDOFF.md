# Context Handoff: Hybrid Trellis Pareto Optimization

## Current State (2026-02-02)

Hybrid trellis **shifts rate-distortion curve** but doesn't dominate it. No configuration found that achieves true Pareto improvement (smaller AND better) on full CID22 corpus.

### Key Finding: Positive Coupling

**Previous session explored negative coupling** (aggressive compression with quality boost to compensate).

**This session discovered positive coupling** works better:
- `aq_lambda_scale > 0` = preserve texture detail in high-AQ blocks
- Combined with slight quality reduction = comparable size, different quality profile

### CID22 Results (30 images, Butteraugli)

#### Positive Coupling at Q85 Base
| Config | Size Δ | BA Δ |
|--------|--------|------|
| baseline Q85 | — | — |
| aq_lambda_scale=4, Q84 | -0.8% | +1.1% |
| aq_lambda_scale=5, Q83.5 | -1.8% | +2.9% |
| aq_lambda_scale=6, Q83.5 | -1.4% | +2.7% |

**No Pareto at Q85** on full corpus (though 10-image subset showed some wins).

#### Quality-Level Dependency
| BaseQ | Hybrid Q | Size Δ | BA Δ | Efficiency |
|-------|----------|--------|------|------------|
| 75 | 73.5 | +1.4% | +1.7% | poor |
| 80 | 78.5 | +0.9% | +0.8% | neutral |
| 85 | 83.5 | -1.8% | +2.9% | 0.62 |
| **90** | **88.5** | **-4.4%** | **+2.4%** | **1.83** |
| **95** | **93.5** | **-10.5%** | **+6.1%** | **1.72** |

**Key insight:** Positive coupling efficiency improves dramatically at high quality (Q90+).

At Q95: -10.5% size for +6.1% BA = excellent trade-off for storage-constrained apps.

## What's Been Tested

### Approach A: Quality-Neutral (boost Q to match BA)
- Negative coupling + higher quality
- Result: Can match BA but files are larger
- **Not Pareto-optimal**

### Approach D: Positive Coupling (preserve texture, reduce Q)  
- Positive `aq_lambda_scale` preserves detail in textured areas
- Reduce base quality to compensate for larger files
- Result: Size savings improve at higher quality levels
- **Best efficiency at Q90-95**

### 10-Image Subset vs 30-Image Full Corpus
The 10-image subset showed several Pareto points (`aq_lambda_scale=4, Q84` at -0.9% size, -1.5% BA).
Full 30-image corpus eliminated these — image diversity matters for validation.

## Recommended Usage

### For Q90+ Applications (archival, print):
```rust
let hybrid = HybridConfig {
    enabled: true,
    aq_lambda_scale: 5.0,  // Positive = preserve texture
    max_adjustment: 0.0,
    ..Default::default()
};
let config = EncoderConfig::ycbcr(quality - 1.5, ChromaSubsampling::Quarter)
    .hybrid_config(hybrid);
```
Expected: -4% to -10% size, +2% to +6% butteraugli (acceptable trade-off).

### For Q75-85 Applications (web):
Don't use hybrid — trade-off is unfavorable. Standard trellis is sufficient.

## Files Created

| File | Purpose |
|------|---------|
| `examples/pareto_approaches.rs` | Test Approach A and D |
| `examples/pareto_fine_sweep.rs` | Fine grid search around optimal |
| `examples/pareto_validate.rs` | Full corpus validation |

## Remaining Questions

1. **Why does positive coupling work better at high quality?**
   - At Q95, quantization is already gentle — positive coupling can safely preserve more
   - At Q75, aggressive quantization needed — positive coupling fights the goal

2. **Could quality-adaptive coupling help?**
   - Different `aq_lambda_scale` at different quality levels
   - E.g., negative at Q75, neutral at Q85, positive at Q95

3. **Is there a per-image adaptive strategy?**
   - Low-texture images might benefit from negative coupling
   - High-texture images from positive coupling

## Commands

```bash
# Run Pareto approach comparison
cargo run --release --example pareto_approaches

# Fine sweep around optimal
cargo run --release --example pareto_fine_sweep

# Full corpus validation  
cargo run --release --example pareto_validate
```

## Summary

Hybrid trellis with positive coupling is a valid **"high-quality mode"** for Q90+ where:
- Storage constraints matter
- +2-6% butteraugli degradation is acceptable
- -4-10% file size savings are valuable

It's **not a universal improvement** over standard trellis — it's a different point on the rate-distortion curve.
