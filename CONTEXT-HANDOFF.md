# Context Handoff: Trellis Validated at Q90+

## Validated Results (30 CID22 images)

| Q | Trellis vs Jpegli Size | BA Δ | Verdict |
|---|------------------------|------|---------|
| 88 | **-2.8%** | +0.8% | ★ Pareto win |
| 90 | **-2.5%** | +0.9% | ★ Pareto win |
| 93 | **-2.0%** | +0.4% | ★ Pareto win |
| 95 | **-1.6%** | +0.3% | ★ Pareto win |

**Per-image at Q90:** Trellis wins 20, Jpegli wins 0, Ties 10.

## Recommended Config

```rust
// Q90+ (archival, print) - USE TRELLIS
EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::HybridProgressive)
    .optimize_scans(true)
// Result: -2.5% size, +0.9% BA vs jpegli

// Q85 and below (web) - NO TRELLIS BENEFIT
EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::JpegliProgressive)
    .optimize_scans(true)  // -0.5% free
```

## What Doesn't Help

| Feature | Finding |
|---------|---------|
| `aq_lambda_scale` coupling | Redundant - same curve as quality change |
| Deringing | No measurable effect |
| Trellis at Q85- | No Pareto benefit (same curve as jpegli) |

## Bug Fixed

`hybrid_config()` now clears `trellis` when enabled=true (commit `df94d12`).

## Commits

- `e8641ef` validate: trellis helps over jpegli at Q90+
- `122442d` investigate: which knobs compose well with jpegli
- `042c52e` investigate: coupling vs quality - traces same curve
- `df94d12` fix: hybrid_config now clears trellis
