# Context Handoff: Trellis Internalization & Hybrid Mode

Branch: `feat/mozjpeg-features`
Date: 2026-02-01

## What Was Done

### Trellis Internalization (Plan: 6/7 steps complete)

Ported mozjpeg's trellis quantization from `mozjpeg-rs` FFI to pure Rust inside zenjpeg.
The plan file is at `~/.claude/plans/cheerful-sleeping-lightning.md`.

**Completed:**
1. Created `zenjpeg/src/trellis/` module (ac.rs, dc.rs, eob.rs, rate.rs, mod.rs) — ~1600 lines
2. Updated `hybrid/core.rs` to use `crate::trellis::{trellis_quantize_block, RateTable}` instead of `mozjpeg_rs`
3. Updated `hybrid/config.rs` to use `crate::encode::mozjpeg_compat::TrellisConfig`
4. Cleaned `encode/mozjpeg_compat.rs` — removed `to_mozjpeg_config()` and `to_mozjpeg()` methods
5. Removed `experimental-hybrid-trellis` feature flag from all ~40 usages
6. Ported ~26 unit tests into trellis/ submodules
7. Fixed standalone trellis AQ leak (commit 4689be6)

**NOT completed (step 6 of plan):**
- `mozjpeg-rs` is still in `[dev-dependencies]` (line 102 of `zenjpeg/Cargo.toml`)
- It's used by `tests/trellis_mozjpeg_comparison.rs` for parity testing
- Decision needed: keep as dev-dep for ongoing parity testing, or remove and rely on C cjpeg comparison only

### Parity Testing

Three levels of comparison, all passing:

**Block-level (always runs, no external deps):**
- 2,869 blocks tested: zenjpeg trellis == mozjpeg-rs trellis (0 failures)
- Tests in `tests/trellis_mozjpeg_comparison.rs`: `trellis_block_parity_*`

**Full-encode vs mozjpeg-rs (requires `mozjpeg-tables` feature + CID22 corpus):**
- Overall: -0.5% size (zenjpeg smaller), +0.15 SSIM2
- Worst per-image: +2.6% (was +14.3% before AQ fix)

**Three-way vs C mozjpeg (requires cjpeg binary at `~/work/mozjpeg/build/cjpeg`):**
- vs C mozjpeg: +0.2% overall
- Baseline without trellis: -0.3% to -0.8% (zenjpeg smaller!)
- Baseline with trellis: -0.1% to +0.1% (essentially identical)
- Progressive: +0.5% to +2.5% (different scan scripts, not trellis)
- Quality: virtually identical across all three

### Key Bug Fix: AQ-Lambda Leak in Standalone Mode

File: `zenjpeg/src/encode/hybrid.rs:172-185`

`HybridQuantContext::quantize_block()` dispatches to `hybrid_quantize_block()` in `hybrid/core.rs`.
That function always applied `aq_strength * AQ_LAMBDA_SCALE` to lambda. In Standalone mode
(pure mozjpeg-compat), this modulated trellis lambda by jpegli's AQ, causing +2.5% size regression.

Fix: Standalone mode now passes `effective_aq = 0.0` so lambda is unmodified:

```rust
TrellisMode::Standalone(trellis_config) => {
    (*trellis_config, 0.0)  // no AQ influence
}
```

## Architecture: Two Trellis Modes

The trellis system has two modes, controlled by `TrellisMode` enum in `encode/hybrid.rs`:

### Standalone Mode (mozjpeg-compatible)
- Activated via `EncoderConfig::trellis(TrellisConfig::default())`
- Uses fixed lambda from TrellisConfig (no AQ modulation)
- Matches C mozjpeg output within ±0.2%
- Use case: drop-in mozjpeg replacement

### Hybrid Mode (jpegli AQ + mozjpeg trellis)
- Activated via `EncoderConfig::hybrid_config(HybridConfig { enabled: true, .. })`
- AQ modulates trellis lambda per-block: textured blocks get higher lambda (more compression)
- `AQ_LAMBDA_SCALE = 2.0` in `hybrid/core.rs:149`
- Use case: better quality via perceptual optimization

### Dispatch flow
```
EncoderConfig
  → ComputedConfig (encode/config.rs)
    → create_hybrid_ctx() (encode/hybrid.rs:56)
      → HybridQuantContext { mode: Standalone | Hybrid }
        → quantize_block() (encode/hybrid.rs:158)
          → hybrid_quantize_block() (hybrid/core.rs:167)
            → trellis_quantize_block() (trellis/ac.rs)
```

## Key Files

| File | Role |
|------|------|
| `src/trellis/ac.rs` | AC trellis Viterbi DP (core algorithm) |
| `src/trellis/dc.rs` | DC coefficient optimization across blocks |
| `src/trellis/eob.rs` | EOB run optimization across blocks |
| `src/trellis/rate.rs` | RateTable (code lengths for rate estimation) |
| `src/trellis/mod.rs` | Re-exports |
| `src/hybrid/core.rs` | `hybrid_quantize_block()`, `StandardRateTables`, `dct_f32_to_i32()` |
| `src/hybrid/config.rs` | `HybridConfig`, `to_trellis_config()` (AQ→lambda mapping) |
| `src/encode/hybrid.rs` | `HybridQuantContext`, `TrellisMode` enum, `create_hybrid_ctx()` |
| `src/encode/mozjpeg_compat.rs` | `TrellisConfig`, `TrellisSpeedMode` (public API types) |
| `tests/trellis_mozjpeg_comparison.rs` | Block parity + full-encode + C mozjpeg three-way comparison |

## Commits on This Branch (newest first)

```
a7862e3 test: add three-way comparison vs C mozjpeg (libmozjpeg)
4689be6 fix: standalone trellis no longer influenced by AQ lambda modulation
11a7b5e test: block-level trellis parity and matched-tables full-encode comparison
2d8ea78 test: add trellis comparison benchmark (zenjpeg vs mozjpeg-rs on CID22)
f05f285 refactor: extract ac_scan_slot helper, remove dead code in progressive encoder
c04aa3f fix: alloc-instrument build error in BitWriter::into_bytes
7fc96e0 feat: internalize trellis quantization as pure Rust, remove mozjpeg-rs dependency
b85f9a4 perf: cache scan histograms and reduce trial encodes in optimize_scans
f62f28d test: add low quality levels to scan optimization benchmark
aab6bf1 fix: signed-shift bug in scan optimization estimator
18ea9d0 feat: trial-to-buffer scan optimization with mixed-SA search
```

## Running the Tests

```bash
# Block-level parity (always works, no external deps needed)
cargo test --release -p zenjpeg --features test-utils --test trellis_mozjpeg_comparison -- --nocapture trellis_block_parity

# Full-encode comparison vs mozjpeg-rs (needs CID22 corpus + mozjpeg-tables feature)
cargo test --release -p zenjpeg --features mozjpeg-tables,test-utils --test trellis_mozjpeg_comparison -- --nocapture --ignored full_encode_robidoux_comparison

# Three-way vs C mozjpeg (needs CID22 + cjpeg at ~/work/mozjpeg/build/cjpeg)
cargo test --release -p zenjpeg --features mozjpeg-tables,test-utils --test trellis_mozjpeg_comparison -- --nocapture --ignored c_mozjpeg_robidoux_comparison

# All trellis unit tests
cargo test --release -p zenjpeg -- trellis

# Full test suite
cargo test --release -p zenjpeg
```

## What Could Come Next

### Remaining Plan Work
- Decide whether to remove `mozjpeg-rs` from dev-dependencies (currently used for parity tests)
- If removing: replace mozjpeg-rs encoder calls in test with C cjpeg subprocess calls

### Hybrid Mode Improvements
- **DC trellis optimization** — `trellis/dc.rs` has `dc_trellis_optimize()` but it's not wired into
  the encoder pipeline. Currently only AC trellis is used. Wiring DC trellis could improve DC
  coefficient coding, especially for progressive JPEG.
- **EOB run optimization** — `trellis/eob.rs` has `optimize_eob_runs()` but it's not wired in.
  This optimizes cross-block EOBRUN placement for progressive scans.
- **Quant table optimization** (`trellis_q_opt`) — C mozjpeg accumulates statistics across blocks
  to derive optimal quant tables. Requires multi-pass: trellis → optimize tables → re-trellis.
  Deferred to follow-up.
- **AQ-lambda tuning** — `AQ_LAMBDA_SCALE = 2.0` was chosen empirically. Could be tuned
  per-quality-level or based on image statistics. The dampen parameter (currently always 1.0)
  was designed for this.

### Progressive Scan Optimization Gap
- C mozjpeg's progressive scan optimizer produces 0.5-2.5% smaller files than zenjpeg's
- zenjpeg has `optimize_scans` (added earlier on this branch) but uses a different algorithm
- Could investigate C mozjpeg's scan optimization approach for parity

### Rate Table Accuracy
- Currently uses standard Huffman tables for rate estimation
- After Huffman optimization, actual code lengths differ from standard tables
- Two-pass approach: encode → collect stats → rebuild RateTable → re-trellis
  could improve trellis decisions (diminishing returns, ~0.1-0.3% expected)
