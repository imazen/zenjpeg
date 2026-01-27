# Context Handoff - Bounded-Memory Streaming Heuristics

## Session Summary (2026-01-27)

Investigated heuristics for bounded-memory streaming JPEG encoding. Added transition reason tracking and analyzed pathological images that pass heuristics but produce poor Huffman tables.

## Key Findings

### Transition Reason Tracking

Added `TransitionReason` enum to understand WHY images transition to streaming mode:

```rust
pub enum TransitionReason {
    ForcedByRows,       // Testing API forced transition
    HeuristicsPassed,   // Memory limit + heuristics OK
    MinPercentReached,  // min_transition_percent gate only
    SafetyValve,        // 50% safety valve
    NoTransition,       // Full buffering mode
}
```

New methods: `transition_reason()`, `transition_info()` (e.g., "50% (safety)")

### CLIC 2025 Test Results (32 images)

| Min % | Failures | Max Overhead | Mean Trans% |
|-------|----------|--------------|-------------|
| 25% | 2/32 | 14.62% | 47.1% |
| 30% | 1/32 | 13.41% | 47.8% |
| 35% | 1/32 | 7.91% | 48.4% |
| 40% | 1/32 | 6.24% | 49.0% |
| **50%** | **0/32** | **3.62%** | **50.2%** |

**Key insight:** Mean transition is ~47-50% regardless of minimum because most images have low early entropy/coverage (fail heuristics) and wait for safety valve. Setting min=50% guarantees 0 failures with no change to average behavior.

### Pathological Image Analysis

Two images pass heuristics at 25% but produce poor tables:

1. **5e5ce...** (2048×1641): 4.50% overhead at 25%, needs 30%+ for <4%
2. **d79d...** (1638×2048): 14.62% overhead at 25%, needs **50%** for <4%

**Root cause:** These images have high early entropy/coverage (pass heuristics) but their early frequency distributions are NOT representative. The distribution continues to diverge significantly through the image.

| Image | KL at 25% | KL at 100% | Problem |
|-------|-----------|------------|---------|
| pathological1 | 0.004 | 0.11 | Slow convergence |
| pathological2 | 0.015 | **0.55** | Major distribution shift |
| normal | 0.035 | 0.15 | Stabilizes early |

Current heuristics (entropy/coverage) measure "variety" but NOT "representativeness".

## Recommendations

### Option 1: Raise min_transition_percent to 50% (RECOMMENDED)

- Guarantees 0 failures on CLIC 2025 corpus
- No change to average behavior (most images wait for 50% anyway)
- Simplest implementation (just change default)

### Option 2: Add distribution divergence tracking (COMPLEX)

- Take snapshot at min% (e.g., 25%)
- Compare current distribution to snapshot
- If divergence > threshold, wait longer
- Could allow earlier transition for truly stable images
- More complex, may not be worth it given Option 1 works

## Files Modified This Session

- `zenjpeg/src/encode/streaming.rs` - Added TransitionReason tracking
- `zenjpeg/tests/streaming_threshold.rs` - Updated tests to report reasons
- `zenjpeg/examples/analyze_pathological.rs` - Analyze failing images
- `zenjpeg/examples/analyze_distribution_change.rs` - KL divergence analysis
- `zenjpeg/examples/compare_min_thresholds.rs` - Compare min% values

## Test Commands

```bash
# Comprehensive heuristics test (CLIC 2025)
cargo test --release -p zenjpeg --features test-utils --test streaming_threshold comprehensive -- --ignored --nocapture

# Analyze pathological images
cargo run --release -p zenjpeg --features test-utils --example analyze_pathological

# Distribution divergence analysis
cargo run --release -p zenjpeg --features test-utils --example analyze_distribution_change

# Compare minimum threshold percentages
cargo run --release -p zenjpeg --features test-utils --example compare_min_thresholds
```

## Next Steps

1. Decide: Use 50% minimum (simple, safe) or implement divergence tracking (complex, marginal benefit)
2. If 50%: Update default `min_transition_percent` in builder
3. Document the tradeoffs in API docs
4. Consider if there are other corpus images that might fail at 50%
