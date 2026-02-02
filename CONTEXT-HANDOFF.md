# Context Handoff: EOB Optimization A/B Test

Branch: `feat/mozjpeg-mimic-tests`
Date: 2026-02-01

## TASK: Wire EOB optimization for direct A/B comparison

**The user wants a TRUE A/B test of EOB optimization, not more comparisons against C mozjpeg.**

## Current State

1. EOB algorithm is COMPLETE in `zenjpeg/src/trellis/eob.rs`
2. Test harness exists in `zenjpeg/examples/eob_mozjpeg_mimic.rs`
3. In mozjpeg mimic mode, zenjpeg already beats C mozjpeg by 0.5-1%

## What Needs to Be Done

### Wire EOB into baseline (sequential) JPEG encoding

The blocks are in `StripProcessorOutput`:
```rust
pub struct StripProcessorOutput {
    pub y_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    pub cb_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    pub cr_blocks: Vec<[i16; DCT_BLOCK_SIZE]>,
    // ...
}
```

These are OWNED and can be mutated before encoding.

### Integration Point

In `zenjpeg/src/encode/streaming.rs`, before `build_jpeg_sequential_into()` is called:

```rust
// Around line 975-1000, after strip_output is finalized but before encoding
if config.eob_optimization {
    apply_eob_optimization(&mut strip_output, &rate_tables);
}
```

### Implementation Steps

1. **Add `eob_optimization: bool` to `TrellisConfig`** (`encode/mozjpeg_compat.rs`)

2. **Create `apply_eob_optimization()` function** in `encode/strip/mod.rs` or new file:
```rust
use crate::trellis::eob::{estimate_block_eob_info, optimize_eob_runs};
use crate::trellis::rate::RateTable;

pub fn apply_eob_optimization(
    output: &mut StripProcessorOutput,
    y_rate: &RateTable,
    c_rate: &RateTable,
) {
    // For baseline JPEG, ss=1, se=63 (all AC coefficients)
    let ss = 1;
    let se = 63;

    // Y channel
    let y_info: Vec<_> = output.y_blocks.iter()
        .map(|b| estimate_block_eob_info(b, y_rate, ss, se))
        .collect();
    optimize_eob_runs(&mut output.y_blocks, &y_info, y_rate, ss, se);

    // Cb channel
    let cb_info: Vec<_> = output.cb_blocks.iter()
        .map(|b| estimate_block_eob_info(b, c_rate, ss, se))
        .collect();
    optimize_eob_runs(&mut output.cb_blocks, &cb_info, c_rate, ss, se);

    // Cr channel
    let cr_info: Vec<_> = output.cr_blocks.iter()
        .map(|b| estimate_block_eob_info(b, c_rate, ss, se))
        .collect();
    optimize_eob_runs(&mut output.cr_blocks, &cr_info, c_rate, ss, se);
}
```

3. **Get rate tables** - Need AC rate tables for the optimization. Options:
   - Use `RateTable::standard_luma_ac()` and `RateTable::standard_chroma_ac()`
   - Or build from actual Huffman tables after frequency collection

4. **Update test** to compare:
   - zen+trellis (current)
   - zen+trellis+eob (new)

### Key Files

| File | Purpose |
|------|---------|
| `trellis/eob.rs` | EOB algorithm (DONE) |
| `trellis/rate.rs` | Rate tables for cost estimation |
| `encode/streaming.rs` | Where to call EOB before encoding |
| `encode/strip/mod.rs` | Where to add `apply_eob_optimization()` |
| `encode/mozjpeg_compat.rs` | Add `eob_optimization` flag to TrellisConfig |
| `examples/eob_mozjpeg_mimic.rs` | Test harness |

### Expected Outcome

After wiring, the test should show:
```
  Q      zen+tr   zen+tr+eob   Δ_eob
-----------------------------------------
 50     45333       ?????     -X.XX%
```

If EOB helps, we'll see negative delta. If not, ~0%.

## DO NOT

- Do NOT compare against C mozjpeg again
- Do NOT skip the implementation
- Do NOT make excuses about complexity

## Test Command

```bash
cargo run --release -p zenjpeg --features mozjpeg-tables --example eob_mozjpeg_mimic
```
