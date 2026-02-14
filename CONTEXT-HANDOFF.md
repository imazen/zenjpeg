# Context Handoff — decode-speed-optimization branch

**Date:** 2026-02-14
**Branch:** `decode-speed-optimization` (clean worktree)
**Last commit:** `872561d` feat: single-pass fused parallel decode for 4:2:0 + fancy upsample

## COMPLETED: Fix fused parallel decode for fancy upsample

### The problem

With `--features parallel`, the default decoder (4:2:0 + Triangle upsampling) hits a
**2-2.5x regression** vs sequential. The fused parallel path for fancy upsample
(`decode_fused_subsampled_planes` in `fused_parallel.rs:825`) allocates full-image
Y/Cb/Cr planes, then does a sequential second pass for upsample+color. This is strictly
worse than the sequential strip-based approach which keeps data cache-hot.

Meanwhile, the fused box-filter path (`decode_fused_subsampled_box`, line 601) works
perfectly — 6x faster than mozjpeg at 2048 — because it's truly single-pass.

### Benchmark evidence (decode_mozjpeg, 4:2:0 baseline, frymire photo)

| Size | sequential | parallel fancy (BROKEN) | parallel box (GOOD) |
|------|-----------|------------------------|---------------------|
| 512 | 1.09ms | 1.95ms (1.8x slower) | 594µs (1.8x faster) |
| 1024 | 3.14ms | 7.93ms (2.5x slower) | 1.20ms (2.6x faster) |
| 2048 | 12.94ms | 27.68ms (2.1x slower) | 2.10ms (6.2x faster) |

### The fix

Replace `decode_fused_subsampled_planes` with a single-pass approach that mirrors the
box path's structure but uses double-buffered extended chroma strips for fancy upsampling.

**Key insight:** Triangle upsample only needs ±1 chroma row of vertical context. The
sequential path already solves this with extended chroma buffers (c_strip_height + 2 rows:
context above, data, context below). Each parallel segment can do the same independently.

**Architecture per segment:**
1. Allocate per-thread strip buffers: Y strip + double-buffered extended Cb/Cr strips
2. Decode first MCU row's chroma into ext_a, second into ext_b
3. Set above context for first MCU row via edge replication (same as image top edge)
4. For each MCU row in segment:
   - Set below context in ext_a from first row of ext_b
   - IDCT Y blocks into Y strip
   - Upsample extended Cb/Cr strips to full resolution
   - YCbCr→RGB, write to output
   - Swap ext_a/ext_b, IDCT next chroma into freed buffer
5. At segment boundaries: edge replication (identical to image-edge behavior)

**Reference code:**
- Box fused path (working model): `fused_parallel.rs:601-818`
- Sequential extended-strip approach: `output.rs:347-620`
- Per-row upsample primitive: `upsample.rs:934` (`upsample_row_h2_fancy_bilinear`)
- Full-strip upsample with scratch: `upsample.rs:232` (`upsample_h2v2_i16_fancy_reuse_scratch`)

### CRITICAL: Hash-lock decode outputs

**The fused path reassembles RGB from independently-processed parallel segments. Any
off-by-one in segment boundaries, chroma context, or strip addressing will cause visible
horizontal lines at MCU row boundaries where segments meet.**

**Before merging the fix, you MUST:**

1. **Hash-lock test**: Encode a test image with `restart_mcu_rows=4`, decode with
   `--features parallel` (fused path) and without (sequential path). The RGB output
   must be **byte-identical**. Write a test that asserts this:
   ```rust
   let sequential_rgb = Decoder::new().decode(&jpeg, Unstoppable)?;
   let parallel_rgb = Decoder::new().decode(&jpeg, Unstoppable)?;  // with parallel feature
   assert_eq!(sha256(&sequential_rgb.pixels_u8()), sha256(&parallel_rgb.pixels_u8()));
   ```

2. **Multi-size hash-lock**: Test at multiple sizes (256, 512, 1024, 2048) to catch
   edge cases in segment boundary handling. Include non-MCU-aligned dimensions
   (e.g., 1000x1000, 513x513) to test partial MCU rows at image edges.

3. **Cross-segment boundary visual check**: Decode a smooth-gradient test image and
   inspect rows at segment boundaries (every `restart_mcu_rows * mcu_height` rows).
   Any discontinuity = bug. The `jpeg_inspect --validate` tool can help.

4. **Edge replication parity**: At segment boundaries, the fused path uses edge
   replication for the missing ±1 chroma context row. This means the fused path's
   output will differ slightly from sequential at these rows (sequential has the real
   adjacent chroma). The hash-lock test should account for this:
   - Option A: Accept the tiny difference (max 1 pixel value per channel) and hash-lock
     the parallel output separately
   - Option B: After all segments complete, do a thin fixup pass on the 2 boundary rows
     per segment junction using the now-available adjacent chroma, achieving exact parity
   - **Prefer option B** — exact parity is more maintainable

5. **Regression gate**: The `fused_parallel_decode` test file
   (`tests/fused_parallel_decode.rs`) already has 10 tests comparing fused vs sequential.
   These MUST continue to pass. Run: `cargo test --release --features parallel,decoder
   -p zenjpeg --test fused_parallel_decode`

### What NOT to do

- Do NOT allocate full-image planes. The whole point is strip-based processing.
- Do NOT use `FusedResult::Planes`. Delete it — it was the broken two-phase approach.
  Return `FusedResult::Rgb` from all fused paths.
- Do NOT skip the parallel path for fancy upsample as a "quick fix". The default decoder
  uses Triangle upsampling on 4:2:0, which is ~90% of all JPEGs. Skipping parallel for
  the default case defeats the purpose.

## Decode path summary (after `parallel.rs` deletion)

```
Baseline + parallel feature:
  1. Fused parallel?  → MCU-row-aligned DRI + ≥1024 MCUs + ≥4 segments
     ├─ 4:4:4/gray:   single-pass → RGB                    (working)
     ├─ 4:2:0 + box:  single-pass → RGB                    (working, 6x speedup)
     └─ 4:2:0 + fancy: *** NEEDS FIX *** (currently 2x regression)
  2. Streaming RGB?   → 4:4:4 + prefer_streaming
  3. Sequential       → fallback

Output from stored coefficients:
  1. Return fused/streaming result
  2. Parallel IDCT+color (output_parallel.rs) → ≥8M pixels
  3. Sequential i16 4:4:4 or subsampled
  4. f32 precise fallback
```

## Other session work (already committed)

- Deleted `parallel.rs` (standard parallel entropy decode) — 648 lines removed.
  Was strictly inferior to fused parallel and caused 2-2.5x regressions. Commit `10bfe9e`.
- Added "make archmage-simd mandatory" to CLAUDE.md TODO (not yet implemented).

## Files to modify

| File | What to do |
|------|-----------|
| `decode/fused_parallel.rs` | Replace `decode_fused_subsampled_planes` with strip-based single-pass |
| `decode/fused_parallel.rs` | Delete `FusedResult::Planes` and `PixelPlanes` struct |
| `decode/parser/output.rs` | Remove `convert_from_pixel_planes` (was phase 2 of broken path) |
| `tests/fused_parallel_decode.rs` | Add hash-lock test comparing parallel vs sequential output |

## Benchmark commands

```bash
# Sequential (no parallel)
cargo bench --bench decode_mozjpeg -- --quick

# Parallel (fused paths)
cargo bench --bench decode_mozjpeg --features parallel -- --quick

# Full decoder comparison (includes cjpegli, zune, progressive, 4:4:4)
cargo bench --bench decode_compare -- --quick

# Fused parallel correctness tests
cargo test --release --features parallel,decoder -p zenjpeg --test fused_parallel_decode
```

## Pre-existing test failures (not caused by this branch)

- `locked_values` — 84 hash mismatches from restart marker changes on main
- `frymire_hash_locked` — encoder output size drift
- `metrics_comparison` — degenerate test image SSIMULACRA2
- `multi_decoder_compatibility` — zune-jpeg grayscale butteraugli bug
