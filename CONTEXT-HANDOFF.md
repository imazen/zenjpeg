# Context Handoff: Trellis Investigation Complete

## When Trellis Helps

| Subsampling | Quality | Size Δ | BA Δ | Speed | Verdict |
|-------------|---------|--------|------|-------|---------|
| 4:2:0 | Q90+ | -2.5% | +0.9% | +70% | ★ Worth it |
| 4:4:4 | Q95 | -2.1% | +0.4% | +70% | ★ Worth it |
| 4:4:4 | Q90 | -2.8% | +2.9% | +70% | Not worth it |
| Any | Q85- | ~same curve | | +70% | Not worth it |

## Should Trellis Be Automatic?

**No.** 70% slowdown for 2-3% savings should be opt-in.

## Recommended Usage

```rust
// High quality (Q90+ 4:2:0, Q95+ 4:4:4) — opt into trellis
EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::HybridProgressive)
    .optimize_scans(true)

// Web quality (Q85-) — no trellis benefit
EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::JpegliProgressive)
    .optimize_scans(true)  // -0.5% free
```

## What Doesn't Help

| Feature | Finding |
|---------|---------|
| `aq_lambda_scale` coupling | Redundant (same curve as quality) |
| Deringing | No effect |
| Trellis at Q85- | No Pareto benefit |
| Trellis 4:4:4 at Q90 | +2.9% BA too much |

## Validated Results (30 CID22 images)

**4:2:0 Q90:** Trellis wins 20/30 images, loses 0/30.
**4:4:4 Q95:** Pareto win (-2.1% size, +0.4% BA).

## Commits

- `75384de` investigate: trellis with 4:4:4 and speed cost
- `e8641ef` validate: trellis helps over jpegli at Q90+
- `df94d12` fix: hybrid_config now clears trellis
