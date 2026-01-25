# AC Refinement EOB Flush Bug Fix

**Date**: 2026-01-04
**Working Commit**: Building on commit 784e7c97 (Docker readme update formatting)
**Files Changed**: `zenjpeg/src/huffman_opt.rs`

## Bug Description

Progressive JPEG AC refinement scans were producing decoder-incompatible bitstreams. External decoders (djpeg, jpeg-decoder crate) failed with "unexpected huffman code" while the internal jpegli decoder passed.

## Root Cause

In C++ `TokenizeACRefinementScan()` (entropy_coding.cc:274-283), when a newly-nonzero coefficient is emitted, the EOB run is reset:

```cpp
*next_token++ = token;
// ...
next_eob_token = next_token;
eob_run = eob_refbits = 0;
```

This ensures that EOB tokens from previous blocks are emitted BEFORE the newly-nonzero token.

In Rust, EOB runs were being accumulated across blocks without flushing when a newly-nonzero broke the run. This caused:
1. EOB tokens to appear after newly-nonzero tokens (wrong order)
2. EOB run counts to be incorrect
3. Refbits counts attached to wrong tokens

## Token Comparison (Scan 10, Ah=1, Al=0)

**C++ (Correct):**
```
num_tokens=44
[12] symbol=0x10 refbits=1
EOBRUNS: 3 7 1 0 3 1
```

**Rust (Before Fix):**
```
num_tokens=45 (1 extra!)
[12] symbol=0x10 refbits=0
[13] symbol=0x00 refbits=1  <- EXTRA TOKEN
EOBRUNS: 7 15 17 8 7 2  <- completely wrong
```

## Fix

In `tokenize_ac_refinement_scan()`, flush pending EOB run BEFORE emitting newly-nonzero:

```rust
if newly_nonzero {
    // Flush any pending EOB run BEFORE emitting newly-nonzero token.
    // This matches C++ which resets eob_run when a newly-nonzero is found.
    if eob_run > 0 || !pending_refbits.is_empty() {
        self.emit_eob_run_with_refbits(context, eob_run, &pending_refbits);
        pending_refbits.clear();
        eob_run = 0;
    }
    // ... then emit ZRL and newly-nonzero tokens
}
```

## Testing

Test command that exposed the bug:
```bash
cd zenjpeg && cargo test --test progressive_encoding -- q10 --nocapture
```

The test image was a 64x64 noise image with noise_mul=13 at Q10 progressive mode.

## Instrumentation Used

- C++: `DUMP_AC_REFINEMENT=1 ./build/tools/cjpegli ...`
- Rust: `DUMP_AC_REFINEMENT=1 cargo run --example compare_cpp`

Both print token-by-token output for AC refinement scans.

## Related Files

- `/home/lilith/work/zenjpeg/zenjpeg/src/huffman_opt.rs` - tokenization
- `/home/lilith/work/zenjpeg/zenjpeg/src/entropy.rs` - encoding
- `/home/lilith/work/zenjpeg/internal/jpegli-cpp/lib/jpegli/entropy_coding.cc` - C++ reference
