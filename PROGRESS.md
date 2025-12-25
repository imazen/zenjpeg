# zenjpeg Progress Notes

## Current State (December 2025)

**Build Status**: Compiles successfully
**Tests**: 23 passing, 4 ignored (pending testdata)

## What We've Done

### 1. Core Module Wiring (Complete)

Successfully integrated mozjpeg-oxide modules:

- **huffman.rs** - Full Huffman table implementation with:
  - `HuffTable` (raw bits + huffval format for JPEG stream)
  - `DerivedTable` (O(1) lookup format for encoding)
  - `FrequencyCounter` for optimal table generation
  - Helper functions: `std_dc_luma()`, `std_ac_luma()`, `std_dc_chroma()`, `std_ac_chroma()`

- **trellis.rs** - Full trellis quantization with:
  - `trellis_quantize_block()` - Single block R-D optimization
  - `dc_trellis_optimize()` - Cross-block DC optimization
  - `optimize_eob_runs()` - EOB run optimization (progressive)
  - `TrellisConfig` with `ac_enabled`, `dc_enabled`, `eob_opt`, lambda params

- **entropy.rs** - Entropy encoding with:
  - `jpeg_nbits()` / `compute_category()` - Bit count for values
  - `BitWriter` - Bitstream with byte stuffing
  - `EntropyEncoder` - Full block encoding
  - `SymbolCounter` - Frequency counting for 2-pass optimization

- **consts_moz.rs** - All mozjpeg constants:
  - 9 quantization table variants
  - Standard Huffman tables (DC/AC, luma/chroma)
  - Zigzag tables

### 2. Error Handling (Complete)

Added missing error variants:
- `Error::InvalidHuffmanTable`
- `Error::HuffmanCodeLengthOverflow`
- `Result<T>` type alias in error.rs

### 3. Two-Pass Huffman Optimization (Complete)

The encoder now:
1. Quantizes all blocks
2. Counts symbol frequencies
3. Generates optimal Huffman tables
4. Encodes with optimal tables

### 4. Type Compatibility (Complete)

Fixed field naming:
- `TrellisConfig::enabled` -> `TrellisConfig::ac_enabled`
- All initializers updated in strategy.rs

## What Still Needs Work

### 1. Wire Up Trellis to Encoder

Currently `quantize_block_impl` has a TODO:
```rust
if strategy.trellis.ac_enabled {
    // TODO: Wire up trellis with proper Huffman table for rate estimation
    // For now, fall back to simple quantization
    quantize_block(&dct, &effective_quant)
}
```

The challenge: trellis needs a `DerivedTable` for rate estimation, but we build
optimal tables AFTER quantizing. Options:
1. Use standard tables for first pass, then re-quantize with optimal tables
2. Two-pass: count without trellis, generate tables, then quantize with trellis
3. Build derived table before quantization for rate estimation

### 2. Progressive Encoding

The `eob_opt` and progressive scan support are implemented in trellis.rs but not
wired to the encoder.

### 3. DC Trellis

`dc_trellis_optimize()` exists but isn't called. Need to:
1. Collect all raw DCT blocks
2. Run AC trellis per-block
3. Run DC trellis across all blocks in component

### 4. Compression Ratio Gap

Previous benchmarking showed ~1.8x larger than C mozjpeg. Root causes:
- Trellis not actually being used (see TODO above)
- Missing progressive encoding
- Possibly different quantization table selection

## Architecture Notes

### Encoder Pipeline

```
RGB pixels
    |
    v
YCbCr conversion (color.rs)
    |
    v
8x8 block extraction (encode.rs)
    |
    v
Level shift (-128) (dct.rs)
    |
    v
Forward DCT (dct.rs)
    |
    v
[AQ field computation] (adaptive_quant.rs)
    |
    v
Quantization (quant.rs or trellis.rs)
    |
    v
Pass 1: Count frequencies (entropy.rs SymbolCounter)
    |
    v
Generate optimal Huffman tables (huffman.rs)
    |
    v
Pass 2: Encode blocks (entropy.rs EntropyEncoder)
    |
    v
Build JPEG stream (encode.rs)
```

### Strategy Selection

- Q < 50: Mozjpeg strategy (aggressive trellis + progressive)
- Q 50-70: Hybrid (balanced trellis)
- Q 70-80: Conservative trellis
- Q >= 80: No trellis (Huffman optimization only)

This is based on mozjpeg's observation that trellis helps more at low quality.

## Key Insights / Red Herrings

### Confirmed Learnings

1. **Huffman table format matters**: JPEG stores `bits[1..16]` (counts per length)
   not `bits[0..16]`. The `bits[0]` slot is unused.

2. **DerivedTable vs HuffTable**: Need both formats - raw for JPEG stream output,
   derived for O(1) encoding.

3. **TrellisConfig field naming**: The old code used `enabled`, mozjpeg-oxide uses
   `ac_enabled`. Had to update throughout.

4. **Circular dependencies**: types.rs tried to re-export from trellis.rs, but
   trellis.rs imported from types.rs. Solved by defining TrellisConfig in trellis.rs.

### Red Herrings Investigated

1. **jpeg_nbits location**: Initially thought it should be in huffman.rs, but it's
   really an entropy coding utility - put it in entropy.rs.

2. **Error types**: Thought we needed many new variants, but only two were needed:
   `InvalidHuffmanTable` and `HuffmanCodeLengthOverflow`.

## Dependencies

- mozjpeg-oxide (../mozjpeg-rs) - Reference for trellis implementation
- jpegli (../jpegli/jpegli-rs/jpegli) - Reference for AQ (not yet fully ported)
- mozjpeg-sys - For comparison testing against C reference

## License

Same as Imageflow (AGPLv3 + commercial).

## Next Steps

1. **Wire trellis into encoder** - The big one. Need to restructure quantization
   to provide DerivedTable for rate estimation.

2. **Add progressive encoding** - Support SOF2 and multi-scan output.

3. **Port jpegli AQ properly** - Current simplified variance-based AQ is disabled
   because it hurts quality. Need jpegli's perceptual model.

4. **Benchmark against mozjpeg-sys** - Validate compression ratio is close.

5. **Consider chromaeval from imageflow** - User mentioned this for quality eval.
