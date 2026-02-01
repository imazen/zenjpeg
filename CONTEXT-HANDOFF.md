# Context Handoff: LayoutParams Follow-Up Simplifications

## What Was Done

The `LayoutParams` immutable substruct refactor is complete (6 commits on `main`).
All geometry is computed once in `zenjpeg/src/encode/layout.rs` and threaded through
`StripProcessor`, `StreamingAQ`, and memory estimate functions. 479 tests pass,
hash-locked regression unchanged.

Commits (oldest first):
- `e79a455` refactor: wire LayoutParams into StripProcessor and StreamingAQ
- `c6fe12f` refactor: use LayoutParams in memory estimate functions
- `f33e101` fix: clippy and formatting issues from LayoutParams refactor
- `96ec052` docs: update CLAUDE.md - mark LayoutParams refactor as complete
- `3823c89` docs: log LayoutParams refactor request in FEEDBACK.md

## What Remains: Chroma/B-Channel Dimension Deduplication

Several locations in `strip/mod.rs` and `strip/convert.rs` still recompute chroma
and B-channel dimensions from raw width/height/subsampling instead of reading from
`self.layout.*`. These are safe (they produce the same values) but defeat the purpose
of having a single source of truth.

### 1. B-channel dimensions in `dct_strip_blocks_to_pending` (strip/mod.rs ~1025)

```rust
// Current (recomputed):
let b_width = (width + 1) / 2;
let b_strip_height = (strip_height + 1) / 2;
let b_blocks_w = (b_width + 7) / 8;
```

Replace with `self.layout.b_width`, `self.layout.b_blocks_h`. Note: `b_strip_height`
and `b_strip_blocks_h` are per-strip values (not full image), so they may need a new
field or remain computed from `strip_height`. Check whether `strip_height` here always
equals `self.layout.strip_height` — if yes, these can be precomputed.

### 2. Chroma dimensions in `dct_strip_blocks_to_pending` (strip/mod.rs ~1055)

```rust
// Current (recomputed):
let (c_width, c_strip_height) = match self.layout.subsampling {
    Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
    Subsampling::S422 => ((width + 1) / 2, strip_height),
    ...
};
let c_blocks_w = (c_width + 7) / 8;
```

Replace with `self.layout.c_width`, `self.layout.c_strip_height`,
`self.layout.c_blocks_h` (which equals `c_blocks_w`). Same strip_height caveat
as above — verify the `strip_height` parameter always matches `self.layout.strip_height`
for non-final strips.

### 3. Chroma dimensions in `push_raw_ycbcr_strip` (convert.rs ~93)

```rust
let (chroma_width, chroma_height) = match self.layout.subsampling {
    Subsampling::S444 => (width, strip_height),
    Subsampling::S422 => ((width + 1) / 2, strip_height),
    Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
    Subsampling::S440 => (width, (strip_height + 1) / 2),
};
```

Replace with `self.layout.c_width` and `self.layout.c_strip_height`.

### 4. Chroma dimensions in `pad_chroma_down_vertically` (convert.rs ~149)

```rust
let (chroma_width, target_height) = match self.layout.subsampling {
    Subsampling::S444 => (self.layout.width, self.layout.strip_height),
    Subsampling::S422 => ((self.layout.width + 1) / 2, self.layout.strip_height),
    ...
};
```

Same replacement: `self.layout.c_width`, `self.layout.c_strip_height`.

### 5. Chroma dimensions in `convert_strip_gamma_aware` (convert.rs ~803)

```rust
let (c_width, c_strip_height) = match self.layout.subsampling {
    Subsampling::S420 => ((width + 1) / 2, (strip_height + 1) / 2),
    ...
};
```

Replace with `self.layout.c_width`, `self.layout.c_strip_height`.

### 6. Chroma dimensions in `convert_strip_box_fused` (convert.rs ~889)

Same pattern as #5. Same replacement.

### 7. B-channel width in `convert_strip_to_ycbcr_420` (convert.rs ~560)

```rust
let c_width = (width + 1) / 2;
let c_height = (strip_height + 1) / 2;
```

For 4:2:0, `c_width == self.layout.c_width` and `c_height == self.layout.c_strip_height`.

### 8. B-channel width in XYB conversion (convert.rs ~757)

```rust
let b_width = (width + 1) / 2;
let b_height = (strip_height + 1) / 2;
```

Replace with `self.layout.b_width`. The `b_height` is per-strip, same caveat.

## Key Caveat: Strip Height Parameters

Many of these functions take a `strip_height: usize` parameter because the final
strip may be shorter than `self.layout.strip_height`. The `c_strip_height` and
`b_strip_height` values in `LayoutParams` are for full-height strips. For the final
(partial) strip, the callers pass the actual remaining height.

**Recommendation**: For items where `strip_height` equals the layout's strip_height
(non-final strips), use `self.layout.*` directly. For the final strip path, keep
the local computation but add a comment referencing `LayoutParams` as the source of
truth for the formula.

Alternatively, add a helper method to `LayoutParams`:
```rust
impl LayoutParams {
    /// Chroma strip height for a given actual strip height (handles partial final strip)
    pub fn c_strip_height_for(&self, actual_strip_height: usize) -> usize {
        match self.subsampling {
            Subsampling::S420 | Subsampling::S440 => (actual_strip_height + 1) / 2,
            _ => actual_strip_height,
        }
    }
}
```

## Lower Priority

- `serialize.rs:write_frame_header_xyb_ex()` — hardcoded `0x22`/`0x11` sampling
  factors could read from `LayoutParams`. Low risk since XYB is always R:2x2 G:2x2
  B:1x1, but it's the last hardcoded geometry value. No behavioral change.

## Verification

After each change, run:
```bash
cargo fmt && cargo clippy --all-targets --all-features -- -D warnings
cargo test --release -p zenjpeg
cargo test --release -p zenjpeg --test frymire_hash_locked
```

All changes are pure refactors — no behavioral changes expected, hash-locked test
must remain green.
