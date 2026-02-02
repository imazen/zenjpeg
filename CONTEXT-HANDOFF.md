# Context Handoff: Hybrid Trellis Exploration

## Goal

Explore whether a working hybrid mode (AQ-coupled trellis) provides value over
standalone trellis. The hybrid path exists in code but was never wired into the
encoding pipeline. Determine if it's worth fixing.

## Current State (commit bfa3b85)

Branch: `feat/mozjpeg-mimic-tests`

### What exists

The hybrid trellis infrastructure is complete but disconnected:

1. **`HybridConfig`** (`hybrid/config.rs`, 481 lines) — config struct with AQ coupling
   params, `to_trellis_config(aq_strength, dampen, is_chroma)` method that adjusts
   lambda per-block based on AQ strength
2. **`HybridQuantContext`** (`encode/hybrid.rs`, 338 lines) — dispatch between Hybrid
   and Standalone modes, calls `hybrid_quantize_block()` for AC trellis
3. **`hybrid_quantize_block()`** (`hybrid/core.rs`, 310 lines) — the actual trellis
   DP optimization, rate table construction
4. **`create_hybrid_ctx()`** (`encode/hybrid.rs:56`) — priority dispatch: TrellisConfig
   first, then HybridConfig, then None. **This function is never called.**
5. **`quantize_block_dispatch()`** (`encode/hybrid.rs:84`) — dispatches hybrid vs
   standard quantization. **Also never called from production code.**
6. **`ExpertConfig`** (`encode/search.rs`) — flat config struct that builds either
   `TrellisConfig` (coupling=0) or `HybridConfig` (coupling>0)

### What's broken

The standalone trellis path works through `StripProcessor`:
```
streaming.rs:327  →  if let Some(ref trellis) = config.trellis {
                         processor.set_trellis(*trellis);
                     }
strip/mod.rs:715  →  pub fn set_trellis(&mut self, config: TrellisConfig) {
                         self.hybrid_ctx = Some(HybridQuantContext::from_trellis_config(config));
                     }
strip/mod.rs:1189 →  hybrid_ctx.quantize_block(&dct, &quant_values, aq, 1.0, is_luma)
```

The hybrid path is dead because `streaming.rs` never checks `config.hybrid_config`.
When ExpertConfig sets `aq_trellis_coupling > 0`, it puts `trellis = None` and
`hybrid_config.enabled = true`. Streaming.rs sees `trellis = None` → skips set_trellis()
→ StripProcessor gets no trellis context → standard rounding.

### The fix is simple

In `streaming.rs`, after the existing trellis check at line 326-328, add:
```rust
} else if builder.hybrid_config.enabled {
    processor.set_hybrid(builder.hybrid_config);
}
```

And add `set_hybrid()` to StripProcessor (`strip/mod.rs`):
```rust
pub fn set_hybrid(&mut self, config: HybridConfig) {
    self.hybrid_ctx = Some(HybridQuantContext::new(config));
}
```

This wires the existing `HybridQuantContext::new(config)` (Hybrid mode) into the same
`hybrid_ctx` field that standalone trellis uses. The `quantize_block()` dispatch already
handles both modes correctly via `TrellisMode::Hybrid` vs `TrellisMode::Standalone`.

### But does it provide value?

The **question** is whether AQ-adjusted lambda improves quality or compression.

**What standalone trellis does:** Fixed lambda = `2^scale1 / (2^scale2 + block_norm)`.
Same aggressiveness for every block regardless of content complexity.

**What hybrid would do:** `effective_scale1 = base_scale1 + aq_strength^exponent * coupling`.
Blocks with high AQ strength (complex texture) get more aggressive trellis zeroing.
Blocks with low AQ strength (smooth areas) get gentler trellis.

**Hypothesis:** This should improve quality because:
- Complex blocks can afford more coefficient zeroing (masking effect)
- Smooth blocks need preserved coefficients to avoid visible artifacts
- Same total bit budget, redistributed by perceptual importance

**Counter-hypothesis:** It might not help because:
- jpegli's AQ already adjusts quant tables per-block (same direction)
- Trellis already uses `block_norm` in lambda denominator (similar signal)
- Double-counting AQ influence could over-compress textured areas

### Exploration plan

1. **Wire it up** — add `set_hybrid()` to StripProcessor, check hybrid_config in streaming.rs
2. **Measure baseline** — encode CID22 corpus at Q85 with standalone trellis, record sizes + SSIMULACRA2
3. **Measure hybrid** — same corpus with coupling=1.0, 2.0, 4.0
4. **Compare** — are hybrid files smaller at same quality? Better quality at same size?
5. **Sweep parameters** — try different exponents (0.5, 1.0, 2.0), thresholds, chroma_scale
6. **If valuable** — update ExpertConfig docs, fix the Hybrid presets, add integration tests
7. **If not** — document why, remove dead code, simplify ExpertConfig

### Key files to read

| File | What | Lines |
|------|------|-------|
| `encode/streaming.rs:320-350` | Where trellis gets wired (and hybrid doesn't) | ~30 |
| `encode/strip/mod.rs:710-720` | `set_trellis()` on StripProcessor | ~10 |
| `encode/strip/mod.rs:1160-1270` | Quantization using `hybrid_ctx` | ~110 |
| `encode/hybrid.rs` | `create_hybrid_ctx`, `HybridQuantContext`, dispatch | 338 |
| `hybrid/config.rs` | `HybridConfig`, `to_trellis_config()` with AQ coupling | 481 |
| `hybrid/core.rs` | `hybrid_quantize_block()` — the actual trellis DP | 310 |
| `trellis/ac.rs` | `trellis_quantize_block()` — core trellis algorithm | 644 |
| `encode/search.rs:570-619` | `build_trellis_or_hybrid()` — ExpertConfig dispatch | ~50 |

### Parameter sensitivity data (from test_parameter_sensitivity)

Standalone trellis at defaults saves ~15% over no trellis. The lambda parameters
(`scale1` range 12–17, `scale2` range 14–18) have massive impact (-46% to +12%).

Preset baselines (256x256, Q85, 4:2:0):
- MozjpegMaxCompression: 16,979 bytes (trellis + progressive search)
- MozjpegBaseline: 17,327 bytes (trellis + baseline)
- JpegliBaseline: 18,355 bytes (no trellis, jpegli AQ + zero-bias)
- HybridBaseline: 23,081 bytes (BROKEN — trellis silently off, jpegli tables)

If hybrid provides even 1-2% additional savings at same quality, that's valuable.
If it provides better quality at same size (higher SSIMULACRA2), even better.

### Also dead and worth investigating

While wiring hybrid, consider also:
- `trellis_num_loops`: Could multi-pass trellis help? C mozjpeg supports it.
- `trellis_use_lambda_weight_tbl`: CSF weights could improve perceptual quality
  (flat weights aren't optimal). Would need actual CSF table implementation.
- `trellis_eob_opt`: The current impl is broken but the concept is sound. C mozjpeg
  does EOB optimization successfully. Needs integration into the trellis pass rather
  than as a post-pass (see old CONTEXT-HANDOFF.md content below for full analysis).

### EOB Optimization Background (from previous investigation)

EOB optimization is BROKEN because the post-trellis implementation uses quantized
coefficients (integers 1-10) without lambda weighting, comparing incompatible units.
mozjpeg computes `accumulated_zero_dist = Σ(original_coef² * lambda * lambda_tbl[z])`
during trellis. Our impl sees quantized values and destroys quality (77% smaller, 40x
worse DSSIM). Fix requires storing `accumulated_zero_dist` during trellis pass.
Expected benefit is only ~0.5-1%, and zenjpeg already beats C mozjpeg by 0.1-0.6%
without it. Low priority.

### Test commands

```bash
# Run parameter sensitivity test
cargo test --release -p zenjpeg --lib -- search::tests::test_parameter_sensitivity --nocapture

# Run all search tests
cargo test --release -p zenjpeg --lib -- search --nocapture

# Run all lib tests
cargo test --release -p zenjpeg --lib

# Clippy
cargo clippy --release -p zenjpeg --lib -- -D warnings

# Full test suite
cargo test --release -p zenjpeg --lib --tests
```

### CLAUDE.md sections to read

- "Investigation Notes > ExpertConfig Parameter Sensitivity" — full data tables
- "Known Bugs" items 1-2 — hybrid dead code + trellis dead params
- "DONE: ExpertConfig for External Optimization" — struct design and API
