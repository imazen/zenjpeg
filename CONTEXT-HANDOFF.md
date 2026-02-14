# Context Handoff — decode-speed-optimization branch

**Date:** 2026-02-14
**Branch:** `decode-speed-optimization` (clean worktree)
**Last commit:** `2009bd8` investigate: cache pressure experiment for >2048 parallel scaling cliff

## Session Summary

Investigated why parallel decode scaling collapses above 2048px. Built two
benchmark tools, ran experiments, identified root cause.

## Key Finding: The >2048 Parallel Scaling Cliff

**Parallel decode scales well at 1024-2048 (2.6-4.4x on 8T) then collapses to
<2x at 4096+.**

Benchmark data (bl-420 Q85, 5 CLIC images averaged):

| Size | mozjpeg | zen-1T | zen-8T | 8T/1T | ns/pixel |
|------|---------|--------|--------|-------|----------|
| 512 | 570us | 611us | 530us | 1.15x | 1.9 |
| 1024 | 1.99ms | 2.29ms | 891us | 2.57x | 1.8 |
| 2048 | 8.12ms | 9.46ms | 2.14ms | **4.42x** | 1.8 |
| 4096 | 57.6ms | 65.4ms | 37.0ms | 1.77x | **3.6** |
| 8192 | 225ms | 261ms | 141ms | 1.85x | 3.6 |

### Root cause: L3 cache overflow in buffered decoder

The per-pixel cost doubles at 4096 **even at 1T** (1.8ns → 3.6ns). This is the
buffered decoder's two-pass architecture hitting L3 capacity:

| Size | Coefficients | RGB output | Total | vs L3 (32MB) |
|------|-------------|------------|-------|-------------|
| 2048² | 12MB | 12MB | 24MB | fits |
| 4096² | 48MB | 48MB | 96MB | 3x over |

Pass 2 (IDCT + color convert) reads coefficients from a 48MB buffer where every
access is an L3 miss → DRAM latency.

### NT writes ruled out

Non-temporal stores are **slower** than regular fill() on this CPU (Zen 4). Raw
8T write time is only 3.8% of decode time at 4096. The bottleneck is not output
writes — it's coefficient reads in the two-pass architecture.

### Box-filter: minor optimization, not strategic

Box-1T is ~10% faster than triangle-1T for 4:2:0. Gap narrows at large sizes
where Huffman/IDCT dominate. Not worth pursuing as a performance strategy.

## What Needs Investigation Next

### 1. Why does the fused parallel decoder still degrade at 4096?

The fused decoder is single-pass (no coefficient buffer), so its working set per
segment is ~740KB (fits L2). Yet 8T speedup drops from 3.52x (2048) to 1.62x
(4096). Possible causes to test with `perf stat` / `cachegrind`:
- Memory bandwidth saturation (8 threads × 48MB output = DDR5 contention)
- TLB pressure from 48MB output buffer
- Rayon scheduling overhead at large segment counts
- L3 contention for compressed scan data reads

### 2. Parallel scanline decoder ("leapfrog")

The scanline decoder matches/beats zune at all sizes (no coefficient buffer).
Making it 2-thread parallel could give ~2x without memory problems:
- Thread A decodes MCU row N, Thread B decodes MCU row N+1
- Requires DRI restart markers for independent segments
- Fancy upsampling needs ±1 chroma row context across segments
  - Larger DRI = fewer sync points but coarser granularity
  - Current boundary fixup is sequential (2 rows per junction)
  - Could use overlapping decode (decode 1 extra MCU row each side)

### 3. Adaptive restart marker sizing (encoder side)

Currently hardcoded `restart_mcu_rows(4)`. Could optimize for decode:
- <512: DRI=0 (no restarts, avoid overhead — parallel hurts at small sizes)
- 512-2048: DRI=4 (current sweet spot)
- 4096+: DRI=2 (more parallel work units for better load balancing)

## Files Created This Session

| File | Purpose |
|------|---------|
| `zenjpeg/examples/bench_decode_matrix.rs` | Full decode speed matrix (slow, ~30min) |
| `zenjpeg/examples/bench_cache_experiment.rs` | NT write experiment (fast, ~2min) |

## Key Existing Files

| File | What |
|------|------|
| `zenjpeg/src/decode/fused_parallel.rs` | Parallel decode (1389 lines) |
| `fused_parallel.rs:812-1355` | Fancy upsample path (double-buffered strips + fixup) |
| `fused_parallel.rs:900` | Pre-allocated full-image RGB buffer |
| `fused_parallel.rs:1270-1345` | Sequential boundary fixup pass |
| `zenjpeg/src/decode/upsample.rs:934-1015` | `upsample_row_h2_fancy_bilinear()` |
| `zenjpeg/src/decode/scanline.rs` | Streaming scanline reader (fast 1T path) |

## Commands

```bash
# Quick cache experiment (~2 min)
cargo run --release --features parallel,decoder --example bench_cache_experiment

# Full decode matrix (~30 min, consider removing 8192)
cargo run --release --features parallel,decoder --example bench_decode_matrix

# Profile fused decoder cache behavior
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses \
  cargo run --release --features parallel,decoder --example bench_cache_experiment
```

## Pre-existing test failures (not caused by this branch)

- `locked_values` — hash mismatches from restart marker changes on main
- `frymire_hash_locked` — encoder output size drift
- `metrics_comparison` — degenerate test image SSIMULACRA2
- `multi_decoder_compatibility` — zune-jpeg grayscale butteraugli bug
