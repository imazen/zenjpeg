# JPEG Lossless Format Support: Feasibility Analysis

*2026-03-07*

## Summary

Medium-hard difficulty (5/10). The codebase isn't hostile to lossless, but the DCT
assumption is deeply baked in. Recommended approach: parallel code paths, not
architectural unification. Estimate: 4-6 weeks for production decode+encode.

## Scope: Which "JPEG Lossless"?

| Format | Standard | Marker | Usage |
|--------|----------|--------|-------|
| **JPEG Lossless** (T.81 SOF3/SOF11) | Original 1992 JPEG | SOF3, SOF11 | Medical imaging (DICOM), rare elsewhere |
| **JPEG-LS** (T.87) | Separate standard | SOF55 (0xF7) | Newer, better compression, also rare |
| **JPEG XL** | Separate codec | N/A | Modern, not JPEG-compatible |

This analysis covers **T.81 JPEG Lossless (SOF3/SOF11)** — the natural fit for a
JPEG codec.

## Current State

- `JpegMode::Lossless` enum variant exists in `types.rs` — **stub, not implemented**
- The `lossless/` module is **DCT-domain lossless transforms** (rotation/flip without
  re-encoding), not lossless compression
- SOF3/SOF11 marker constants are completely absent from `consts.rs`

## Architectural Mismatch

JPEG Lossless is architecturally nothing like lossy JPEG:

| | Lossy (current) | Lossless (SOF3) |
|---|---|---|
| Transform | 8x8 DCT blocks | Pixel-by-pixel prediction |
| Quantization | Central pipeline | None |
| Block structure | 8x8 MCU grid | No blocks |
| Entropy coding | DC/AC categories (0-15) | Error values (unbounded) |
| Subsampling | 4:2:0, 4:2:2, etc. | Always 4:4:4 |
| IDCT | Two tuned implementations | N/A |

DCT assumptions appear in 15+ files. `DCT_BLOCK_SIZE` is referenced ~60 times.
The entire encoder strip pipeline (`StripProcessor -> DCT -> quantize -> zigzag ->
entropy`) needs a parallel path.

## Code Reuse (~40%)

| Component | Reuse | Notes |
|-----------|-------|-------|
| Bitstream I/O | 100% | `BitReader`/`BitWriter` are format-agnostic |
| Marker parsing | 80% | SOF/DHT structure parsing shared |
| Huffman table building | 50% | Same JPEG Huffman, different symbol interpretation |
| Streaming architecture | High | Naturally suits scanline-by-scanline prediction |
| Test infrastructure | 60% | Corpus framework, comparison harness |

## New Code Required (~1500-2500 lines)

1. **Predictor functions** (~50 lines) — 8 linear predictors from T.81 Table H.1
2. **Lossless scan parser** (~150 lines) — Different SOS interpretation
3. **Lossless entropy decoder** (~500-800 lines) — Reuses Huffman but different symbol structure
4. **Lossless encoder** (~500-800 lines) — Pixel-by-pixel, no block batching
5. **Integration/routing** (~200 lines) — Branch at SOF detection, bypass DCT/quant/IDCT

## Difficulty by Component

| Component | Difficulty | Estimate |
|-----------|------------|----------|
| Decode SOF3 (Huffman lossless) | 4/10 | ~2 weeks |
| Encode SOF3 | 6/10 | ~3 weeks |
| SOF11 (arithmetic lossless) | 5/10 | +1 week (arithmetic coder exists) |
| Keeping DCT path regression-free | 3/10 | Parallel paths, no shared mutation |
| Test coverage + corpus | 4/10 | DICOM test files exist publicly |
| Streaming API integration | 4/10 | Lossless is naturally streaming |
| Performance tuning | 5/10 | Prediction is branch-heavy, SIMD helps less |

## Recommended Approach

**Parallel code paths** — don't refactor the DCT pipeline. Branch at SOF detection
and run completely separate logic for lossless. The streaming architecture helps
since lossless is naturally scanline-ordered.

1. Start with **decode only** (lower risk, 2 weeks)
2. Add **encode** once decode is proven (3 weeks)
3. Skip deep architectural unification — the formats share almost nothing beyond
   markers and Huffman tables

### Three strategies compared

| Strategy | Risk | Effort | Notes |
|----------|------|--------|-------|
| **Parallel paths** (recommended) | Low | 4-6 weeks | New files, DCT path untouched |
| **Refactored abstraction** | High | 8-12 weeks | Clean but regression-prone |
| **Don't implement** | None | 0 | Status quo |

## When It's Worth Doing

JPEG Lossless is rare outside medical imaging. If DICOM support is the goal, it's
worth it. If general-purpose compression is the goal, the effort might be better
spent elsewhere. The format is simple enough that implementation risk is low — it
just doesn't share much code with the lossy path.
