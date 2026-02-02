# Context Handoff: Trellis & mozjpeg Features

Branch: `feat/mozjpeg-features`
Date: 2026-02-01

## Session Summary

Completed 3 of 4 tasks from the previous handoff. DC trellis is now wired into the encoder pipeline.

## Commits This Session

```
539af1a feat: wire DC trellis optimization into encoder pipeline
9708433 style: rustfmt hybrid.rs match arm
```

## What Was Done

### 1. DC Trellis Wiring (COMPLETE)

Wired `dc_trellis_optimize()` from `trellis/dc.rs` into the encoder pipeline.

**Changes:**
- Added `y_dc_raw`, `cb_dc_raw`, `cr_dc_raw` fields to `StripProcessor` and `StripProcessorOutput`
- Store raw DC coefficients (scaled by 64) during quantization when DC trellis enabled
- Added `apply_dc_trellis()` method to `StripProcessor::finalize()`
- Added DC rate tables to `StandardRateTables` (luma_dc, chroma_dc)
- Added accessors: `is_dc_trellis_enabled()`, `trellis_config()`, `luma_dc_rate_table()`, `chroma_dc_rate_table()`
- Added `lambda_log_scale1/2()` accessors to `TrellisConfig`

**Note:** DC trellis passes 0 for AC coefficients in raw blocks since we've already done AC trellis. This slightly affects per-block lambda but DC trellis primarily optimizes DC differentials.

**Verified:** `test_trellis_ac_dc_modes_work` shows DC trellis produces smaller files (1373 vs 1375 bytes).

### 2. EOB Run Optimization (DEFERRED)

EOB run optimization (`trellis/eob.rs`) is implemented but NOT wired into the encoder. Reasons:
- C mozjpeg integrates it during trellis quantization pass, not as post-process
- Progressive encoding takes blocks as read-only
- Would need significant refactoring to apply per-scan with correct spectral selection
- The 0.5-2.2% progressive gap vs C mozjpeg is attributed to different scan scripts, not EOB

### 3. Progressive Scan Gap Investigation (COMPLETE)

Investigated the 0.5-2.2% size difference between zenjpeg and C mozjpeg for progressive JPEG.

**Findings:**
- **Baseline is BETTER**: zenjpeg baseline is 0.2-0.8% smaller than C mozjpeg
- **Progressive gap is structural** - different scan scripts:
  - C mozjpeg: frequency split at 8/9, no SA for chroma
  - zenjpeg: frequency split at 2/3, SA for all components
- `optimize_scans` feature achieves ~1.5% savings by trying multiple scripts

**Test data** (15 images × 36 configs):
```
Config                    zen/cmoz  
q50-420-base-trellis       -0.3%   (zenjpeg smaller)
q50-420-prog-trellis       +1.5%   (zenjpeg larger)
q90-420-base-trellis       +0.0%   
q90-420-prog-trellis       +0.2%   
OVERALL                    +0.2%
```

### 4. AQ-Lambda Tuning Investigation (COMPLETE)

Investigated the AQ-lambda scaling parameters in hybrid mode.

**Current state:**
- `AQ_LAMBDA_SCALE = 2.0` (hardcoded in `core.rs:153`)
- `dampen = 1.0` (always, not quality-adaptive)
- `quality_adaptive = false` (testing showed no benefit)

**Infrastructure available:**
- `HybridConfig` has extensive knobs: aq_lambda_scale, aq_exponent, aq_threshold, base_lambda_scale1/2, chroma_scale
- `HybridSweepConfig` can generate parameter sweeps

**No changes made** - the config module documents that all findings are preliminary (~5 images tested). Proper tuning would require a large corpus (100+ images), multiple quality levels, and statistical analysis.

## Key Files Modified

| File | Changes |
|------|---------|
| `encode/strip/mod.rs` | DC raw storage, `apply_dc_trellis()` |
| `encode/hybrid.rs` | DC trellis accessors |
| `encode/mozjpeg_compat.rs` | lambda_log_scale1/2 accessors |
| `hybrid/core.rs` | DC rate tables in StandardRateTables |

## Test Commands

```bash
# DC trellis tests
cargo test --release -p zenjpeg -- dc_trellis

# AC/DC modes test (shows DC trellis effect)
cargo test --release -p zenjpeg --test trellis_config_effects -- test_trellis_ac_dc_modes_work --nocapture

# C mozjpeg three-way comparison
cargo test --release -p zenjpeg --features mozjpeg-tables,test-utils --test trellis_mozjpeg_comparison -- --nocapture --ignored c_mozjpeg_robidoux_comparison

# Full test suite
cargo test --release -p zenjpeg
```

## What Could Come Next

1. **Port C mozjpeg scan script** - For exact progressive parity, port their scan script structure
2. **Wire EOB optimization** - Requires refactoring progressive encoding to allow block mutation
3. **Large-scale AQ tuning** - Build corpus, run parameter sweeps, statistical analysis
4. **Remove mozjpeg-rs dev-dep** - Replace parity tests with C cjpeg subprocess calls (optional)
