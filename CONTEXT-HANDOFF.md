# Context Handoff: Hybrid Trellis Improvement

## Background

The hybrid trellis feature is now functional but produces suboptimal rate-distortion trade-offs.
This handoff documents the current state and proposes approaches to improve it.

## Current State (commit 9fffb8e)

Branch: `feat/mozjpeg-mimic-tests`

### What Works

1. **Hybrid path is wired through** — `byte_encoders.rs:131-135`, `streaming.rs:261-264`
2. **ExpertConfig controls it** — `aq_trellis_coupling > 0` enables hybrid mode
3. **Lambda adjustment happens** — `HybridConfig::to_trellis_config()` adds AQ-based offset
4. **Benchmark exists** — `cargo run --release --example hybrid_trellis_benchmark`

### The Problem

Hybrid produces **larger files** with **better DSSIM** — the opposite of what's useful:

| Mode | Bytes (Q85) | DSSIM | Trade-off |
|------|-------------|-------|-----------|
| jpegli | 38596 | 0.00072 | Baseline |
| standalone | 42587 | 0.00053 | +10% size, -26% DSSIM |
| hybrid(2.0) | 39298 | 0.00069 | +2% size, -4% DSSIM |

We wanted: (same size, better quality) OR (smaller size, same quality)
We got: (larger size, better quality) — not useful for optimization

### Root Cause Analysis

The current implementation in `hybrid/core.rs:183`:
```rust
config.lambda_log_scale1 += aq_strength * AQ_LAMBDA_SCALE;
```

This **increases lambda** (more aggressive compression) for **high-AQ blocks** (textured areas).

The logic was: "textured areas can afford more compression due to masking."
The reality: The zeroed coefficients don't redistribute to smooth areas — they're just lost.

DSSIM improves because it penalizes texture loss less than coefficient removal in smooth areas.
But file size increases because we're not actually saving bits, just allocating them differently.

## Approaches to Explore

### 1. Reverse Coupling Direction

**Hypothesis:** Decrease lambda for high-AQ blocks, increase for low-AQ blocks.

**Rationale:** Preserve texture (high AQ) since it's perceptually important. Compress smooth areas
(low AQ) more aggressively since artifacts there are less visible due to... wait, that's backwards.
Artifacts in smooth areas ARE more visible. This approach probably won't help.

### 2. Lambda Ratio Instead of Offset

**Hypothesis:** Use multiplicative coupling instead of additive.

Current: `scale1 = base_scale1 + aq * coupling`
Proposed: `scale1 = base_scale1 * (1 + aq * coupling)` or `scale1 = base_scale1 * exp(aq * coupling)`

**Rationale:** Additive offset has different effects at different base lambdas. Multiplicative
scaling maintains proportional relationships.

### 3. Rate-Targeting Mode

**Hypothesis:** Given a target file size, hybrid can allocate bits more perceptually.

**Implementation:**
1. First pass: encode with standalone trellis, measure block-level bit usage
2. Compute ideal redistribution based on AQ map
3. Second pass: encode with adjusted lambdas to hit target distribution

**Rationale:** Instead of hoping the lambda adjustment saves bits overall, explicitly
redistribute a fixed budget. This guarantees same size with (hopefully) better quality.

### 4. Butteraugli-Optimized Coupling

**Hypothesis:** DSSIM and Butteraugli have different preferences. Optimize for Butteraugli.

Current benchmark shows:
- Hybrid improves DSSIM but Butteraugli results are mixed
- Butteraugli may prefer different bit allocation

**Implementation:** Run parameter sweep optimizing for Butteraugli instead of DSSIM.
Try negative coupling values, different exponents, quality-dependent strategies.

### 5. Two-Pass Quality Targeting

**Hypothesis:** Use hybrid to match a quality target instead of size target.

**Implementation:**
1. Encode with standalone trellis at target quality
2. Measure Butteraugli/DSSIM
3. Binary search coupling value to match quality with smaller file

**Rationale:** If hybrid can hit same quality at smaller size, that's the goal.

### 6. Block-Level Rate Control

**Hypothesis:** Explicitly cap/floor bit usage per block based on AQ.

**Implementation:**
1. After trellis quantization, compute estimated bits per block
2. If textured block uses too many bits, re-quantize more aggressively
3. If smooth block uses too few bits, allow larger coefficients

**Rationale:** Direct control over bit distribution rather than indirect lambda adjustment.

### 7. Coefficient-Level Coupling

**Hypothesis:** Apply different coupling to different DCT frequencies.

Current: Same lambda adjustment for all 64 coefficients in a block.
Proposed: Higher coupling for high-frequency coefficients (more maskable).

**Implementation:** Extend `lambda_tbl` weights to incorporate AQ strength:
```rust
let weight = base_weight * (1.0 + aq_strength * freq_coupling[i]);
```

### 8. Abandon Hybrid, Improve Standalone

**Hypothesis:** The hybrid approach is fundamentally flawed. Focus on standalone trellis improvements.

Areas to explore:
- Multi-pass trellis (currently single-pass)
- Better lambda parameters (currently using mozjpeg defaults)
- Perceptual lambda weights (currently flat 1/q²)
- DC trellis optimization (currently minimal impact)

## Recommended Exploration Order

1. **Quick tests first** (< 1 hour each):
   - Negative coupling (reverse direction)
   - Multiplicative coupling
   - Different exponents (0.5, 2.0)

2. **If promising, deeper exploration** (2-4 hours):
   - Butteraugli parameter sweep
   - Quality-dependent coupling

3. **If still not working** (4+ hours):
   - Two-pass rate targeting
   - Block-level rate control

4. **Fallback**:
   - Document that hybrid doesn't help
   - Remove or disable the feature
   - Focus on standalone trellis improvements

## Key Files

| File | Purpose |
|------|---------|
| `hybrid/config.rs` | `HybridConfig`, `compute_lambda_adjustment()` |
| `hybrid/core.rs` | `hybrid_quantize_block()`, lambda application |
| `trellis/ac.rs` | Core trellis DP algorithm |
| `encode/strip/mod.rs` | `set_hybrid()`, quantization dispatch |
| `examples/hybrid_trellis_benchmark.rs` | Rate-distortion measurement |

## Test Commands

```bash
# Run hybrid benchmark on default image
cargo run --release --example hybrid_trellis_benchmark

# Run on specific image
cargo run --release --example hybrid_trellis_benchmark -- /path/to/image.png

# Run parameter sensitivity test
cargo test --release -p zenjpeg --lib -- search::tests::test_parameter_sensitivity --nocapture

# Quick build check
cargo clippy --release -p zenjpeg --lib -- -D warnings
```

## Metrics to Track

1. **File size** — bytes, bpp
2. **DSSIM** — lower is better, structural similarity
3. **SSIMULACRA2** — higher is better, perceptual quality
4. **Butteraugli** — lower is better, Google's perceptual metric
5. **Encode time** — hybrid should not be significantly slower

## Success Criteria

Any of these would make hybrid worthwhile:

1. **Same size, better quality** — <1% size increase, >5% DSSIM/Butteraugli improvement
2. **Smaller size, same quality** — >3% size reduction, <2% quality degradation
3. **Pareto improvement** — strictly better on at least one metric, no worse on others

If none achievable after reasonable exploration, recommend removing hybrid feature.
