# Context Handoff — decode-speed-optimization branch

**Date:** 2026-02-14
**Branch:** `decode-speed-optimization` (rebased on `main`, clean worktree)
**Last commit:** `647c1e2` fix: skip trailing restart marker in XYB encoder

## What this branch does

Decoder performance optimizations. 17 commits on top of main (which has parallel decode + lossless restructuring). The branch was rebased cleanly onto main with zero file conflicts — the two branches touched completely disjoint file sets (main: encoder-side, this branch: decoder-side).

## Commits (oldest first)

| Commit | Type | Summary |
|--------|------|---------|
| `32b296e` | infra | Decode-only profiling example (isolates from encoder overhead) |
| `7c63e43` | infra | mozjpeg decode speed comparison benchmark |
| `b3bcd88` | refactor | `#[rite]` for SIMD helpers (inline into `#[arcane]` callers) |
| `839c9cf` | perf | Dequant buffer reuse + BitReader single-buffer refactor (-7.7%) |
| `64ed176` | perf | Eliminate bounds checks in IDCT strided writes (-0.6%) |
| `17dc6cf` | chore | Switch decode_profile to baseline JPEG |
| `09228f8` | perf | Reuse scratch buffer for upsample (-7.1% wall-clock) |
| `0ddbed9` | perf | `chunks_exact` in upsampler (-3.5%, -125M instructions) |
| `3e4de3f` | perf | `chunks_exact` in YCbCr→RGB AVX2 (-2.0%, -70M instructions) |
| `7e402d2` | perf | Hoist loop-invariant values out of MCU loop (-1.8%) |
| `b3aad56` | perf | True partial dequantization using coeff_count (-6.4%) |
| `acdbaef` | perf | Pre-slice coefficient arrays in output pass (-2.1%) |
| `678585a` | perf | Skip padding check for MCU-aligned components |
| `31ab751` | perf | Slice-based BitReader refill + inline DC decode (-4.1%) |
| `9b9b518` | refactor | Fixed-size array for fast_ac decode table |
| `e2fc85e` | fix | Handle stray restart markers between scans in decoder |
| `647c1e2` | fix | Skip trailing restart marker in XYB encoder |

## Performance summary

Starting point: ~1.20x slower than mozjpeg at 2048x2048 baseline 4:2:0.
After optimizations: ~1.05-1.10x, competitive or faster at ≤1024.

Three themes dominated gains:
1. **Bounds check elimination** (~340M instructions) — `chunks_exact`, pre-slicing, upfront assertions
2. **Partial/lazy computation** (~213M) — partial dequant skipping zero coefficients
3. **Memory traffic reduction** (~100M+, 15% wall-clock) — buffer reuse, eliminating repeated zeroing

## Bug fixes (this session)

### Stray restart marker decode failure (FIXED)

**Root cause:** When `restart_interval == total_mcu_count`, the encoder wrote a trailing RST0 after all MCUs. The decoder's MCU loop never consumed it (the restart check fires for the *next* MCU, which doesn't exist). The main parser then hit `FF D0`, fell through to `skip_segment()`, which tried to read a 2-byte length from a standalone marker — interpreting `FF D9` (EOI) as length 65497 and overshooting the data.

**Decoder fix** (`mod.rs`): Added `0xD0..=0xD7 => {}` arm to ignore stray RST markers between scans.

**Encoder fix** (`blocks.rs`): Both XYB baseline paths now skip `check_restart()` on the last MCU, matching the existing YCbCr behavior. Progressive tokenizer was already correct.

## Test status

- **766 lib tests pass**, 0 failures
- **encoder_matrix**: 3 passed, 0 failed (was 2 passed, 1 failed before our fix)
- **Pre-existing failures on main** (not caused by this branch):
  - `frymire_hash_locked` — encoder output size drifted (+140 bytes at Q50)
  - `locked_values` — 84 hash mismatches from restart marker / lossless changes on main
  - `metrics_comparison` — SSIMULACRA2 Q95 = -35.76 (likely degenerate test image)
  - `multi_decoder_compatibility` — zune-jpeg grayscale butteraugli = 23.5 (zune bug)

## Key files modified

| File | What changed |
|------|-------------|
| `decode/parser/output.rs` | Buffer reuse, pre-slicing, scratch buffers |
| `decode/parser/scan.rs` | MCU loop hoisting, padding skip, streaming guard |
| `decode/parser/mod.rs` | Stray RST marker handling |
| `decode/idct_int.rs` | Bounds check elimination via upfront assertion |
| `decode/upsample.rs` | `chunks_exact`, scratch buffer reuse |
| `color/ycbcr.rs` | `chunks_exact` in AVX2 kernel |
| `entropy/decoder.rs` | Inline DC decode, fixed-size fast_ac array |
| `foundation/bitstream.rs` | Single-buffer BitReader, slice-based refill |
| `quant/mod.rs` | True partial dequantization |
| `encode/blocks.rs` | Skip trailing RST in XYB encoder |

## What's NOT done / next steps

- Branch is not merged to main yet
- The remaining ~5-10% gap to mozjpeg at large sizes is from the two-pass coefficient architecture (cache misses in output pass). The scanline decoder avoids this and already matches/beats mozjpeg.
- Progressive decode gap at 4096+ is from AC refinement (inherently serial Huffman + refinement bits)
