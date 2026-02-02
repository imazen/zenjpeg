# Context Handoff: Hybrid Trellis vs JpegliProgressive

## Summary

**Hybrid trellis cannot beat JpegliProgressive on both size AND quality.**

It shifts the rate-distortion curve but doesn't dominate it. No Pareto improvement found.

## Bug Fixed This Session

**`hybrid_config()` was being ignored!**

When `OptimizationPreset::HybridProgressive` was used, it set `trellis = Some(TrellisConfig::default())`.
Then `create_hybrid_ctx()` checked trellis first and ignored hybrid_config.

**Fix:** `hybrid_config()` now clears `trellis` when `enabled=true`. Commit `df94d12`.

## Results vs JpegliProgressive Q85 (20 CID22 images)

| Config | Size Δ | BA Δ | Notes |
|--------|--------|------|-------|
| HybridProg no coupling | -3.2% | +3.2% | Default trellis behavior |
| Hybrid +5 coupling, Q85 | +3.2% | **-2.9%** | Better quality, larger |
| Hybrid +5 coupling, Q84 | -0.5% | +1.5% | |
| Hybrid -4 coupling, Q85 | **-9.1%** | +12.4% | Smaller, worse quality |
| Hybrid -4 coupling, Q87 | -0.9% | +3.5% | |
| Hybrid -4 coupling, Q88 | +4.9% | -1.7% | |

### Across Quality Levels (positive coupling +5)

| Q | Jpegli | Hybrid | Size Δ | BA Δ |
|---|--------|--------|--------|------|
| 75 | | | +4.2% | -3.2% |
| 80 | | | +3.7% | -1.8% |
| 85 | | | +3.2% | -2.9% |
| 90 | | | +2.6% | -1.9% |
| 95 | | | +1.7% | +0.9% |

Positive coupling consistently produces **better quality but larger files**.

## What Works

1. **Coupling parameter now works correctly** after fix
2. **Positive coupling** (+4 to +6): Better butteraugli at cost of ~3% larger files
3. **Negative coupling** (-4 to -6): Smaller files at cost of quality
4. **No coupling** (default): ~3% smaller than JpegliProgressive, ~3% worse BA

## What Doesn't Work

**No Pareto improvement over JpegliProgressive:**
- Grid searched 7 couplings × 6 quality offsets
- No configuration beats jpegli on BOTH size AND quality
- Hybrid is a trade-off tool, not a free improvement

## Recommended Usage

### If you need smaller files:
Use HybridProgressive with default settings (no coupling):
```rust
EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::HybridProgressive)
```
Result: ~3% smaller than JpegliProgressive, ~3% worse butteraugli.

### If you need better quality:
Use positive coupling:
```rust
let hybrid = HybridConfig {
    enabled: true,
    aq_lambda_scale: 5.0,  // Positive = preserve texture
    ..Default::default()
};
EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::HybridProgressive)
    .hybrid_config(hybrid)
```
Result: ~3% better butteraugli, ~3% larger than JpegliProgressive.

### If you want jpegli-equivalent output:
Just use JpegliProgressive:
```rust
EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
    .optimization(OptimizationPreset::JpegliProgressive)
```

## Commands

```bash
# Run comparison vs JpegliProgressive
cargo run --release --example pareto_vs_jpegli
```

## Commits

- `df94d12` fix: hybrid_config now clears trellis to ensure coupling takes effect
- `065abb9` investigate: positive coupling Pareto analysis for hybrid trellis

## Conclusion

Hybrid trellis is a **trade-off tool**, not a Pareto improvement. Users should choose:
- JpegliProgressive for jpegli-compatible output
- HybridProgressive for smaller files (accepting quality loss)
- HybridProgressive + positive coupling for better quality (accepting size increase)
