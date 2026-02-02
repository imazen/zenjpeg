# Context Handoff: Knobs That Compose Well With Jpegli

## TL;DR

| Knob | Effect | When to Use |
|------|--------|-------------|
| **Scan Optimization** | -0.5% size, same quality | **Always** (free) |
| **Trellis** | -2.3% size, same quality | **Q90+ only** |
| Combined | -2.9% size | Q90+ |

## Knobs That DON'T Help

| Knob | Why |
|------|-----|
| `aq_lambda_scale` coupling | Redundant - same curve as changing quality |
| Deringing | No measurable effect |
| MozjpegProgressive | 7% smaller but 24% worse butteraugli |

## Recommended Config

```rust
// For Q90+ (archival, high quality)
EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::HybridProgressive)  // adds trellis
    .optimize_scans(true)  // -0.5% free
// Result: -2.9% smaller than JpegliProgressive at same quality

// For Q85 and below (web)
EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::JpegliProgressive)
    .optimize_scans(true)  // -0.5% free
// Trellis doesn't help at this quality level
```

## Evidence

At Q90 (15 CID22 images):
| Config | Size | BA |
|--------|------|-----|
| Jpegli | 71740 | 1.674 |
| +ScanOpt | 71378 (-0.5%) | 1.674 |
| +Trellis | 70062 (-2.3%) | 1.686 |
| +Both | 69657 (-2.9%) | 1.686 |

At Q85: Trellis gives -3% size but +4% worse BA (not Pareto).

## Bugs Fixed

- `hybrid_config()` now clears `trellis` when enabled, ensuring HybridConfig is used

## Commits

- `122442d` investigate: which knobs compose well with jpegli
- `042c52e` investigate: coupling vs quality - traces same curve
- `df94d12` fix: hybrid_config now clears trellis
