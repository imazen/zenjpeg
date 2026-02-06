# Context Handoff — Decoder Refactoring

## What Was Done

7-phase decoder refactoring plan to reduce ~13,500 lines across 5 decode paths to ~9,000 lines with 2 paths sharing a common `StripProcessor`.

### Completed Phases

| Phase | Commit | What |
|-------|--------|------|
| 3 | `c356549` | Extract config types to `decode/config.rs` (316 lines) |
| 4 | `a0cfba4` | Create `ParsedScanData` intermediate type, replace 16-param constructor |
| 1 | `49c645c` | Extract `StripProcessor` to `decode/pipeline.rs` (386 lines). Moved IDCT, strip buffers, upsampling from `scanline.rs`. `ScanlineReader` now uses `strip: StripProcessor` field. |
| 2 | `57ffe87` | Consolidate `upsample.rs` — add strided h2v1/h1v2 for all 3 filter types, make non-strided versions delegate. Replace ~200 lines inline upsampling in `pipeline.rs` with function-pointer dispatch. |

Also completed: archmage 0.3→0.5 upgrade (`70d6e99`) — token renames (`Avx2Token`→`X64V3Token`), `try_new()`→`summon()`, safe slice-based SIMD loads.

### Current File Sizes

```
scanline.rs     1997 lines  (was ~2200)
pipeline.rs      386 lines  (new, was 528 before Phase 2)
upsample.rs     1904 lines  (was 1811, +93 from strided functions)
parser/output.rs 1483 lines
config.rs        316 lines  (new)
mod.rs          1052 lines
parser/mod.rs    905 lines
idct_int.rs     1260 lines
```

## Remaining Phases

### Phase 5: Fix scanline MCU boundary upsampling
- **Problem**: Scanline decoder's chroma upsampling uses strip-local context only. At MCU row boundaries, the triangle/libjpeg filters need the last row from the previous strip and first row from the next strip for correct vertical interpolation.
- **Current behavior**: Edge rows fall back to duplicating the boundary row (nearest-neighbor at boundaries).
- **Fix**: Add prev/next row ring buffers to `StripProcessor` for cross-boundary context.
- **Files**: `pipeline.rs`, `scanline.rs`
- **Risk**: Medium — correctness-critical, needs careful testing against buffered decoder output.

### Phase 6: Fallible allocations + panic hardening
- **Problem**: Strip buffers in `StripProcessor::new()` use `vec![]` which panics on OOM.
- **Fix**: Replace with `try_alloc_zeroed()` / `try_reserve()`, propagate `Result`.
- **Files**: `pipeline.rs`, `scanline.rs`, `config.rs`
- **Risk**: Low — mechanical changes, existing patterns in codebase.

### Phase 7: Deduplicate f32 output paths
- **Problem**: `parser/output.rs` has 3 nearly-identical f32 output methods (~400 lines each) for different color spaces. They share the same upsampling dispatch + color conversion structure.
- **Fix**: Extract shared f32 output pipeline, parameterize by color conversion function.
- **Files**: `parser/output.rs`
- **Risk**: Medium — output.rs is complex, needs careful testing of XYB/YCbCr/RGB paths.

## Architecture Reference

### Decode Pipeline (3 stages)
1. **Entropy decode** → coefficients (`entropy/decoder.rs`)
2. **IDCT + dequant** → strip buffers (`pipeline.rs` `idct_block()`)
3. **Upsample chroma + color convert** → RGB output (`pipeline.rs` `upsample_chroma()`, `scanline.rs` `read_rows_*()`)

### StripProcessor (`pipeline.rs`)
Shared struct for stages 2-3:
- `y_strip`, `cb_strip`, `cr_strip` — IDCT output buffers (SIMD-aligned strides)
- `cb_upsampled`, `cr_upsampled` — upsampled chroma (full resolution)
- `idct_block()` — dequant + IDCT into strip position
- `upsample_chroma()` — dispatches to h2v1/h1v2/h2v2 via function pointers
- `row_planes()` / `y_row()` — row accessors for color conversion

### Upsampling Architecture (`upsample.rs`)
All upsampling uses the **strided function signature**:
```rust
type StridedFn = fn(&[i16], usize, usize, usize, &mut [i16], usize, usize, usize);
//                    input, in_width, in_stride, in_height, output, out_width, out_stride, out_height
```
- 9 strided functions: `{h2v1,h1v2,h2v2}_i16_{fancy,libjpeg,nearest}_strided`
- 9 non-strided wrappers that delegate with stride=width
- h2v2 fancy has AVX2 SIMD path (`upsample_h2v2_i16_fancy_avx2`)
- 3 f32 allocating functions used by buffered decoder (`upsample_fancy`, `upsample_libjpeg_f32`, `upsample_nearest_f32`)

### Two Decoder Paths
1. **Scanline** (`scanline.rs`): Uses `StripProcessor` for strip-based streaming decode. MCU-row-at-a-time.
2. **Buffered** (`parser/output.rs`): Full-image coefficient storage, separate output pass. Uses `StripProcessor::new_dummy()` (strips unused).

## Working Tree State
- Clean (uncommitted fmt diff stashed as `stash@{0}`)
- Branch: `main`
- All 625 lib tests pass
- Integration tests pass (420/422/444 subsampling verified)

## Key Test Commands
```bash
cargo test --release -p zenjpeg --lib                    # 625 unit tests
cargo test --release -p zenjpeg --features decoder       # + decoder tests
cargo test --release -p zenjpeg --test scan_optimize_integration  # subsamp tests
cargo test --release -p zenjpeg -- --ignored             # C++ parity (needs build)
```
