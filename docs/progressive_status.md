# Progressive JPEG Status Report

Generated: 2026-01-03
Test: `verify_progressive_support.rs`

## Executive Summary

**Progressive JPEG Level 2 with Successive Approximation IS WORKING!**

The CLAUDE.md documentation claiming "refinement broken" is **OUTDATED**. All YCbCr progressive modes work correctly with all external decoders.

## Test Results

### ✅ TEST 1: Baseline Sequential (SOF0)
- **Size**: 57,102 bytes
- **Scans**: 1
- **All decoders**: ✓ PASS

### ✅ TEST 2: Progressive + Standard Huffman
- **Size**: 167,936 bytes
- **Scans**: 13
- **Successive Approximation**: YES ✓
- **All decoders**: ✓ PASS
  - jpegli-rs: ✓
  - zune-jpeg: ✓
  - mozjpeg: ✓
  - jpeg-decoder: ✓

**Analysis**: Progressive Level 2 works perfectly! 13 scans = 1 DC (interleaved) + 12 AC scans (4 per component × 3 components).

### ✅ TEST 3: Progressive + Optimized Huffman
- **Size**: 70,161 bytes (58% smaller than standard!)
- **Scans**: 13
- **Successive Approximation**: YES ✓
- **All decoders**: ✓ PASS

**Analysis**: Optimized Huffman with progressive works! Achieves **58% compression** vs standard Huffman (167,936 → 70,161 bytes).

### ❌ TEST 4: XYB + Progressive
- **Size**: 140,006 bytes
- **Scans**: 15 (3 DC non-interleaved + 12 AC)
- **Successive Approximation**: YES ✓
- **Decoder Results**:
  - jpegli-rs: ✗ **UnexpectedEof** ("not enough bits to read")
  - zune-jpeg: ✓ PASS
  - mozjpeg: ✗ **PANIC/CRASH**
  - jpeg-decoder: ✗ **"scan makes use of unset dc huffman table"**

**Analysis**: XYB progressive encoder produces invalid bitstream. zune-jpeg decodes it (lenient), but our decoder and jpeg-decoder fail, mozjpeg crashes.

### ✅ TEST 5: Grayscale Progressive
- **Size**: 51,764 bytes
- **Scans**: 5 (1 DC + 4 AC)
- **Successive Approximation**: YES ✓
- **All decoders**: ✓ PASS

## Summary Table

| Mode | Encoder | Decoder | External Decoders | Status |
|------|---------|---------|-------------------|--------|
| Baseline | ✓ | ✓ | ✓ All pass | **Working** |
| Progressive YCbCr | ✓ | ✓ | ✓ All pass | **Working** |
| Progressive YCbCr + Opt Huffman | ✓ | ✓ | ✓ All pass | **Working** |
| Progressive Grayscale | ✓ | ✓ | ✓ All pass | **Working** |
| Progressive XYB | ✗ | ✗ | ✗ 3/4 fail | **BROKEN** |

## What Works ✅

1. **Progressive Level 2 with Successive Approximation**
   - Encoding: ✓ Working
   - Decoding: ✓ Working
   - AC refinement scans: ✓ Working
   - External compatibility: ✓ All decoders pass

2. **Optimized Huffman for Progressive**
   - Currently uses **standard Huffman tables**
   - Progressive + standard Huffman: 167,936 bytes
   - Progressive + optimized Huffman: 70,161 bytes (**58% reduction!**)
   - But this is optimized **per-scan**, not **globally**

3. **Scan Structure**
   - YCbCr 4:4:4: 13 scans (1 DC interleaved + 12 AC non-interleaved)
   - Grayscale: 5 scans (1 DC + 4 AC)
   - XYB: 15 scans (3 DC non-interleaved + 12 AC) - **but broken**

## What's Broken ❌

### 1. XYB + Progressive Mode
**Problem**: Encoder produces invalid bitstream

**Symptoms**:
- Our decoder: UnexpectedEof ("not enough bits to read")
- jpeg-decoder: "scan makes use of unset dc huffman table"
- mozjpeg: PANIC/CRASH
- zune-jpeg: Decodes (but may be lenient/buggy)

**Root causes** (likely):
1. **Huffman table assignment** - XYB components may be using wrong table IDs
2. **MCU ordering** - Block ordering may be incorrect for XYB
3. **Bitstream encoding** - Refinement scans may have bit errors

**Impact**: Cannot use XYB with progressive mode

### 2. Global Huffman Optimization for Progressive
**Problem**: Huffman tables are optimized per-scan, not globally

**Current behavior**:
- Each scan gets its own optimized Huffman table
- This is suboptimal because:
  - Decoder must parse multiple DHT markers
  - Cannot share statistics across scans
  - May not match C++ jpegli behavior

**What C++ jpegli does**:
1. **Pass 1**: Tokenize ALL scans, collect global frequency stats
2. Build optimized tables from global stats
3. **Pass 2**: Encode ALL scans using same optimized tables

**Impact**: Progressive + optimized Huffman may be larger than C++ jpegli

## File Size Comparison

```
Mode                              Size       vs Baseline    vs Prog+Std
─────────────────────────────────────────────────────────────────────
Baseline + Opt Huffman           57,102     baseline       -66%
Progressive + Std Huffman       167,936     +194%          baseline
Progressive + Opt Huffman        70,161     +23%           -58%
Grayscale Progressive            51,764     -9%            N/A
```

**Key findings**:
- Progressive with standard Huffman is **2.9× larger** than baseline!
- Progressive with optimized Huffman is only **23% larger** than baseline
- But C++ jpegli progressive should be **smaller** than baseline (with global optimization)

## Recommendations

### High Priority
1. ✅ **Update CLAUDE.md** - Remove "refinement broken" claim
2. ❌ **Fix XYB + Progressive**
   - Debug Huffman table assignment
   - Fix bitstream encoding
   - Add XYB-specific tests

### Medium Priority
3. **Implement Global Huffman Optimization**
   - Two-pass encoding
   - Collect stats from all scans in pass 1
   - Encode with shared tables in pass 2
   - Should reduce progressive size to < baseline

### Low Priority
4. **Change default to Progressive**
   - Only after XYB + Progressive fixed
   - Only after global Huffman optimization
   - Matches C++ jpegli default

## Next Steps

1. Update CLAUDE.md to reflect that Progressive Level 2 **IS** working
2. Debug XYB + Progressive encoder (Huffman table assignment)
3. Implement global Huffman optimization for progressive scans
4. Verify file sizes match C++ jpegli with equivalent settings
