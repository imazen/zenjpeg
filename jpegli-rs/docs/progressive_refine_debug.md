# Progressive AC Refinement Debug Log

## Problem Statement
Progressive level 2 encoding (with successive approximation) produces invalid bitstreams that cause "unexpected huffman code" errors during decode. Standard decoders (jpeg_decoder, djpeg) fail, but lenient decoders (PIL) succeed.

## Failure Pattern
- **Sizes that fail**: 50x50, 51x51, 52x52 (last_block_cols = 2, 3, 4)
- **Sizes that work**: 48, 49, 53-56
- **Patterns that fail**: photo_like (pseudo-random), checkerboard (at 52x52)
- **Patterns that work**: solid, gradient_h, gradient_v

## Key Observations
1. 49x49 and 50x50 both have 49 blocks, yet one works and one fails
2. Our own decoder successfully decodes the "failing" files
3. Baseline encoding works for all sizes/patterns
4. The issue is specifically in AC refinement scan (Ah=2, Al=1)

## Codec Analysis Tools

| Tool | Path | Purpose |
|------|------|---------|
| djpeg | system | IJG decoder with verbose output |
| jpeg_decoder | Rust crate | Strict decoder for testing |
| PIL/Pillow | Python | Lenient decoder (succeeds on "bad" JPEGs) |
| cjpegli | internal/jpegli-cpp/build/tools/ | Reference encoder |
| jpegtran | system | JPEG transcoder |

## Red Herrings

### 1. DCT Scaling (1/8 vs 1/64)
- **Hypothesis**: Changing to 1/64 scaling would fix file size gap
- **Result**: CATASTROPHIC FAILURE - decoded values were 8× too small
- **Lesson**: Don't trust C++ comments about internal scaling without testing

### 2. EOB Run Symbols Not in Standard Table
- **Hypothesis**: EOB run symbols (0x10, 0x20, etc.) not in standard AC table
- **Result**: We have fallback code that emits individual EOBs when symbols missing
- **Status**: NOT the issue - fallback works correctly

### 3. Sign Bit vs Refinement Bit Ordering (PARTIALLY FIXED)
- **Hypothesis**: Order was symbol + sign + refbits, should be symbol + refbits + sign
- **Result**: Fixed ordering, but problem persists
- **Status**: Was a real bug but not the only one

### 4. ZRL Skip Count (PARTIALLY FIXED)
- **Hypothesis**: ZRL should skip 16 zeros, not 15
- **Result**: Fixed to set num_zeros_to_skip = 16 for ZRL
- **Status**: Was a real bug but not the only one

## Experiments Run

### Pattern Test Matrix
| Size | solid | gradient_h | gradient_v | checkerboard | photo_like |
|------|-------|------------|------------|--------------|------------|
| 49x49 | OK | OK | OK | OK | OK |
| 50x50 | OK | OK | OK | OK | **FAIL** |
| 51x51 | OK | OK | OK | OK | **FAIL** |
| 52x52 | OK | OK | OK | **FAIL** | **FAIL** |
| 53x53 | OK | OK | OK | OK | OK |

### Width/Height Scan
- Width 48-49 at height 50: OK
- Width 50-52 at height 50: FAIL
- Width 53+ at height 50: OK
- Width 50 at any height 48-56: FAIL (all fail!)

## Current Suspicions

1. **Pending bits mismatch with ZRL**: When emitting multiple ZRLs, pending_bits might be distributed incorrectly
2. **Off-by-one at block boundaries**: Something wrong with k position after ZRL
3. **Refinement bit count mismatch**: Encoder outputs N bits, decoder expects M bits

## Major Finding: Decoder Bug (Separate Issue)
**Our progressive DECODER is completely broken:**
- Baseline roundtrip MSE: ~54-62 (reasonable for Q75)
- Progressive roundtrip MSE: ~4000+ (garbage output!)

This is a separate issue from the encoder problem. The encoder is producing
files that standard decoders reject, but our decoder is broken regardless.

## Next Steps
1. ~~Add bit-level tracing to encoder~~ - Need to compare with C++ reference
2. Compare bit patterns between 49x49 (working) and 50x50 (failing)
3. Check if issue is in first-pass AC (Al=2) or refinement pass (Ah=2, Al=1)
4. Use C++ jpegli as reference and compare output byte-by-byte
5. (Separate) Fix our progressive decoder
