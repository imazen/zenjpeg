# AC Refinement Encoding Bug Fix

## Summary

Fixed a bug in progressive JPEG AC refinement encoding that caused "unexpected huffman code" errors during decoding.

## Bug Description

When encoding progressive JPEGs with successive approximation refinement (the "Standard" scan script), the encoder would produce invalid bitstreams that decoders couldn't parse. Simple scan scripts (without refinement) worked fine.

### Symptoms
- `jpeg_decoder::Decoder` returned `Error("unexpected huffman code")`
- `djpeg` reported `Corrupt JPEG data: bad Huffman code`
- Simple and Minimal scan scripts worked; only Standard (with Ah>0 refinement scans) failed

## Root Cause

The bug was in `encode_ac_refine()` in `src/entropy.rs`. The function incorrectly determined whether a coefficient was "previously coded" (encoded in the first scan) or "new non-zero" (newly appearing in the refinement scan).

### Original (Buggy) Code

```rust
// Check if this is a previously-coded non-zero coefficient
if (abs_coef >> al) > 1 {
    // Already coded - just queue the refinement bit
    pending_bits.push(((abs_coef >> al) & 1) as u32);
} else if (abs_coef >> al) == 1 {
    // New non-zero coefficient
    // ... output Huffman symbol + sign bit + correction bits
}
```

### The Problem

The check `(abs_coef >> al) > 1` uses the **absolute value** of the coefficient. This fails for negative coefficients with magnitude 1 (i.e., `-1`).

Trace for `coef = -1`:
1. **First AC scan (Al=1)**: `(-1 >> 1) = -1` (arithmetic shift preserves sign), which is non-zero, so the coefficient is **ENCODED**.
2. **Decoder** stores: `-1 << 1 = -2` (shifts back to reconstruct).
3. **Refinement scan (Al=0)**:
   - Decoder has `coeffs[k] = -2`, which is non-zero, so it expects a **correction bit**.
   - Encoder checks: `abs(-1) = 1`, then `(1 >> 0) > 1` = `1 > 1` = **FALSE**.
   - Encoder incorrectly treats this as "new non-zero" and outputs Huffman symbol + sign bit.
4. **MISMATCH**: Decoder expects 1 bit, encoder outputs a full Huffman symbol.

### The Fix

Use the **signed coefficient** with the **previous shift level** to determine if it was previously encoded:

```rust
// Check if this is a previously-coded non-zero coefficient
// The first scan (with Al = current Ah = al+1) encoded the coefficient if
// (coef >> (al+1)) != 0. We must use signed shift to match the first scan's logic.
// For negative values like -1: (-1 >> 1) = -1 (non-zero), so it was encoded.
let prev_shift = al + 1; // Ah = previous Al
if (coef >> prev_shift) != 0 {
    // Already coded - just queue the refinement bit
    let correction_bit = ((abs_coef >> al) & 1) as u32;
    pending_bits.push(correction_bit);
} else if (abs_coef >> al) == 1 {
    // New non-zero coefficient
    // ... output Huffman symbol + sign bit + correction bits
}
```

### Why This Works

| Coefficient | First Scan Check `(coef >> 1)` | Encoded? | Refinement Check (old) | Refinement Check (new) | Correct? |
|-------------|-------------------------------|----------|------------------------|------------------------|----------|
| `-1` | `-1` (non-zero) | Yes | `1 > 1` = false | `-1 >> 1 = -1` (non-zero) | Fixed |
| `1` | `0` (zero) | No | `1 > 1` = false | `1 >> 1 = 0` (zero) | OK |
| `-2` | `-1` (non-zero) | Yes | `2 > 1` = true | `-2 >> 1 = -1` (non-zero) | OK |
| `2` | `1` (non-zero) | Yes | `2 > 1` = true | `2 >> 1 = 1` (non-zero) | OK |

## Files Changed

- `src/entropy.rs`: Fixed the "previously coded" check in `encode_ac_refine()`

## Testing

After the fix:
- `cargo run --example match_ref` - **OK** (was failing)
- `cargo run --example no_correction_bits` - OK
- `cargo run --example step_by_step` - OK
- `cargo test --lib` - 205 passed (1 pre-existing failure unrelated to this fix)

## References

- JPEG ITU-T.81 specification, sections on successive approximation
- jpegli-rs decoder: `src/entropy.rs` `decode_ac_refine()` for understanding decoder expectations
- mozjpeg-rs encoder: `src/entropy.rs` for reference implementation (has same bug!)

## Date

2025-12-27
