# Context Handoff — XYB Parity & LayoutParams Refactor

## What just happened

Fixed XYB AQ v_samp mismatch (`2a4cdd2`). XYB JPEG uses R:2×2 G:2×2 B:1×1
(max_v_samp_factor=2), but the AQ was initialized with v_samp=1 from the S444
subsampling enum. This caused the AQ to use 8-pixel iMCUs instead of 16-pixel,
producing overly conservative quantization → larger files.

**XYB size gap vs C++ improved from 8-18% to 5-11%.**

YCbCr parity is already excellent (+0.9% size, DSSIM within 0.5%).

## Current XYB parity numbers (kodak images, after fix)

| Image | Q70 | Q80 | Q90 |
|-------|-----|-----|-----|
| 1.png | +7.3% | +6.9% | +5.7% |
| 5.png | +6.2% | +5.5% | +5.0% |
| 13.png | +11.0% | +8.7% | +6.9% |

DSSIM is now slightly worse than C++ (expected — less conservative quantization).

## Next task: LayoutParams immutable substruct refactor

The v_samp bug and the earlier AQ channel bug share a root cause: derived geometry
computed independently in multiple locations, only some updated when XYB was added.

### The problem

`v_samp` is derived from `(subsampling, use_xyb)` but computed ad-hoc in:
1. `StripProcessor::with_xyb()` line ~414 — buffer sizing
2. `StripProcessor::init_aq()` line ~704 — AQ initialization
3. `StreamingEncoder::new()` line ~316 — JPEG header
4. `serialize.rs::write_frame_header_xyb_ex()` — hardcoded 0x22/0x11

Same problem with `padded_width`, `blocks_w`, `strip_height`, etc.

### The fix: `LayoutParams` struct

Compute all geometry once, freeze it, pass by `&LayoutParams` everywhere.

```rust
/// Immutable layout computed once from (subsampling, use_xyb, width, height).
struct LayoutParams {
    width: usize,
    height: usize,
    padded_width: usize,
    padded_c_width: usize,
    blocks_w: usize,
    blocks_h: usize,
    v_samp: usize,             // 2 for XYB or 4:2:0/4:4:0, else 1
    strip_height: usize,       // 8 * v_samp
    y_buffer_stride: usize,    // padded_width + 1
    c_strip_height: usize,
    y_blocks_h: usize,
    y_blocks_v: usize,
    c_blocks_h: usize,
    c_blocks_v: usize,
    // ... etc
}
```

### Field audit: what's actually mutable after creation?

**Truly immutable (geometry/config) — move to LayoutParams:**
- StripProcessor: `width`, `height`, `padded_width`, `padded_c_width`, `padded_b_width`,
  `strip_height`, `subsampling`, `y_blocks_h/v`, `c_blocks_h/v`, `b_blocks_h/v`,
  `y_buffer_stride`, `pixel_format`, `chroma_downsampling`
- StreamingAQ: `width`, `height`, `padded_width`, `y_buffer_stride`, `blocks_w`, `blocks_h`,
  `pre_erosion_w/h`, `y_imcu_height`, `total_imcu_rows`, `pre_erosion_buffer_rows`
- StreamingEncoder: `width`, `height`, `bytes_per_row`, `strip_height`

**Set-once-then-frozen (should be construction args, not late mutation):**
- `use_xyb` — set via `set_xyb_mode()` but affects geometry. `with_xyb()` exists
  and is correct; `set_xyb_mode()` should be removed.
- `deringing` — late-set but doesn't affect geometry
- `quant`, `aq_state` — set once via `set_quant_tables()`, then read-only
- `hybrid_ctx` — set once via `set_trellis()`

**Actually mutable processing state (stays on the struct):**
- Progress: `rows_received`, `current_imcu_row`, `current_y`, `rows_buffered`,
  `pre_erosion_rows_flushed`
- Buffer swap: `pending_current`, `y_imcu_current`
- Lookahead: `pending_imcu_row`, `pending_pre_erosion_row`
- Output accumulation: `y_blocks`, `cb_blocks`, `cr_blocks`, `all_aq_strengths`

**Reusable scratch buffers (contents mutate, no realloc):**
- `y_strip`, `cb_strip`, `cr_strip`, `cb_down`, `cr_down`
- `pre_erosion_buffer`, `row_prev/curr/prev_prev`, accumulators
- `pending_y/cb/cr_blocks[2]` (double buffer)
- `aq_strengths_buffer`, `fuzzy_erosion_out`, `pre_erosion_temp`

### Suggested implementation order

1. Create `LayoutParams` with a `fn new(width, height, subsampling, use_xyb) -> Self`
2. Store it as a field in `StripProcessor` (replacing individual dimension fields)
3. Update `StripProcessor` methods to read from `self.layout.*`
4. Pass `&LayoutParams` to `StreamingAQ::new()` instead of individual args
5. Pass `&LayoutParams` to serialization functions (replace hardcoded XYB sampling)
6. Remove `set_xyb_mode()` — require XYB at construction via `with_xyb()`
7. Remove `set_strip_stride()` from StreamingAQ — compute from LayoutParams

### Remaining XYB gap investigation (5-11%)

After the LayoutParams refactor, the remaining XYB size gap likely comes from:
- DCT coefficient rounding (±1 from different SIMD float precision — this is ~1% in YCbCr)
- Zero-bias parameter tuning for XYB mode
- Possible pre-erosion boundary handling differences at image edges

Run `just xyb-diff` to see visual diff patterns. Block-boundary patterns in the R
channel delta suggest coefficient quantization differences.

## Key files

| File | What |
|------|------|
| `zenjpeg/src/encode/strip/mod.rs` | StripProcessor — the fix site, main refactor target |
| `zenjpeg/src/encode/streaming.rs` | StreamingEncoder — builder pattern, config flow |
| `zenjpeg/src/quant/aq/streaming.rs` | StreamingAQ — takes v_samp, owns rolling buffers |
| `zenjpeg/src/encode/serialize.rs` | JPEG header writing — hardcoded XYB sampling factors |
| `zenjpeg/src/encode/encoder_config.rs` | EncoderConfig — user-facing, maps ColorMode to subsampling |
| `zenjpeg/src/encode/encoder_types.rs` | ChromaSubsampling, XybSubsampling enums |
| `zenjpeg/tests/frymire_hash_locked.rs` | Hash-locked regression tests (update after any output change) |

## Commands

```bash
cargo test --release -p zenjpeg                    # All unit+integration tests
cargo test --release -p zenjpeg --test frymire_hash_locked  # Hash-locked regression
cargo run --release -p zenjpeg --example xyb_parity_test    # XYB size/quality comparison
just xyb-diff                                       # Visual diff (5-panel montage)
cargo test --release -p zenjpeg --test comprehensive_cpp_comparison -- --nocapture --ignored  # Full YCbCr parity
```

## Clean state

Working tree is clean. Branch `main` is 17 commits ahead of origin (not pushed).
Delete this file after loading into a new session.
