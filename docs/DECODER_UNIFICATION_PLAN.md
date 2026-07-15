# Decoder Unification Plan

> **Status (2026-07-15): coefficient-centric path now WIRED for 4:4:4 / 4:2:2
> color progressive; 4:2:0 remains on the RGB buffered path — issue #187.**
> The previously-dead `decode_mcu_row_from_coefficients` + `stored_coeffs`
> (fully implemented but never populated with `Some`) are now populated by
> `ScanlineReader::from_coefficients`, and `Decoder::scanline_reader` routes
> 3-component color progressive/arithmetic JPEGs with **no vertical chroma
> subsampling** (4:4:4, 4:2:2) through it. Those cases now decode native
> YCbCr/gray via the same strip pipeline as streaming — the lossy RGB→YCbCr /
> RGB→Y re-derivation (the correctness gaps in the table below) is FIXED for
> them (proven by `tests/bundled/coeff_unification.rs`: progressive 4:4:4
> YCbCr is byte-identical to the baseline decode). RGB output is byte-identical
> to the old path; the streaming hot path is untouched (callgrind −0.0005%).
>
> **What still blocks 4:2:0:** the coefficient path's *vertical* chroma-
> upsampling boundary handling (`peek_next_chroma_row`) diverges from the
> whole-image `to_pixels` reference — a systematic MCU-boundary chroma delta of
> up to ~76 (dropping to ~32 with the peek disabled), so `coeff_strip_compatible`
> gates 4:2:0 out until that vertical-boundary handling is reconciled with the
> streaming path's double-buffered 1-row-lag mechanism. That is the remaining
> #187 work; grayscale stays on the buffered path too (its Y is already native).
>
> **Prior status (2026-07-14): Phase 1 done; buffered-fork surface reduced; full
> `CoeffSource` unification DEFERRED as scoped future work — decision in
> issue #187.**
> What landed: streaming single-pass decode for all baseline subsampling
> modes, arithmetic JPEG support, strip-stride SIMD alignment, and the
> 2-tier IDCT dispatch (a 4x4 tier was implemented then removed). Review R6
> (2026-07-13) unified the borrowed/owned routing POLICY into
> `DecodeConfig::classify_scanline_route`. On 2026-07-14 the shared
> geometry + buffer-bounds-check prologue that every interleaved-format
> `read_rows_*` buffered branch duplicated was extracted into
> `ScanlineReader::buffered_geometry` (verified byte-identical +
> callgrind-neutral), shrinking the fork's duplicated surface.
>
> **The #187 decision — land the safe reductions, defer (do NOT permanently
> shelve) the full `CoeffSource`:** the central abstraction that would
> delete `buffered_rgb` outright is NOT a "cost-free dedup". `buffered_rgb`
> exists because progressive JPEGs are fundamentally un-streamable — a
> progressive scan refines coefficients across the *whole* image before any
> pixel is final, so `new_buffered` decodes the entire image up front (it
> even holds a `StripProcessor::new_dummy`, i.e. no live strip pipeline).
> Unifying the two paths the naive way (make baseline buffer too) would
> regress the streaming hot path — unacceptable. The *correct* unification
> is coefficient-centric (store coefficients for the buffered case and run
> the same strip→row pipeline on demand); it is genuinely feasible
> perf-neutrally because the streaming hot path is untouched, and it would
> ALSO fix the buffered-mode correctness gaps below. But it is a large,
> correctness-sensitive rewrite of the progressive read path that must be
> done incrementally with byte-identity + callgrind gates at each step —
> not squeezed into an unrelated session. It is therefore deferred as its
> own focused effort, kept open in #187, not permanently shelved.

## Problem Statement

The current `ScanlineReader` has two internal modes with **different behaviors**:

| Issue | Streaming | Buffered |
|-------|-----------|----------|
| `subsampling()` | Correct | **Always S444 (wrong)** |
| `read_rows_ycbcr_planes()` | Native YCbCr | **RGB→YCbCr (lossy)** |
| `read_rows_gray*()` | Y channel direct | **Y from RGB (lossy)** |
| Memory | O(width × MCU_height) | O(width × height × 3) |
| Code paths | Separate | Separate |

The root cause: buffered mode converts to RGB too early, discarding native YCbCr data.

## Design Goals

1. **Identical output** regardless of internal mode
2. **Memory minimal** - never store more than necessary
3. **Fastest possible** - single optimized pipeline
4. **Elegant** - no code duplication, clear abstractions

## Architecture: Coefficient-Centric Design

### Core Insight

Both modes ultimately need the same thing: **DCT coefficients for the current MCU row**.

- **Streaming**: Coefficients come from entropy decoder on-demand
- **Buffered**: Coefficients are pre-decoded and stored in `parser.coeffs`

The solution: abstract over coefficient source, share everything else.

```
┌─────────────────────────────────────────────────────────────────┐
│                     ScanlineReader<'a>                          │
├─────────────────────────────────────────────────────────────────┤
│  CoeffSource<'a>                                                │
│  ├── Streaming { bitstream, decoder_state, tables }             │
│  └── Buffered { coeffs: &[Vec<[i16;64]>], position }            │
├─────────────────────────────────────────────────────────────────┤
│  Shared State                                                   │
│  ├── Strip buffers (Y, Cb, Cr) - reused per MCU row             │
│  ├── Upsampled chroma buffers (if subsampled)                   │
│  ├── Image metadata (width, height, subsampling, etc.)          │
│  └── Position tracking (current_row, mcu_row, etc.)             │
├─────────────────────────────────────────────────────────────────┤
│  Pipeline (shared for both modes)                               │
│  1. get_mcu_row_coeffs() → fills coeff buffer                   │
│  2. IDCT → fills Y/Cb/Cr strips                                 │
│  3. Upsample chroma (if needed)                                 │
│  4. Output conversion (based on requested format)               │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Model

| Data | Lifetime | Size |
|------|----------|------|
| Coefficient source | Borrowed from parser or bitstream | O(1) reference |
| Y strip | Per MCU row, reused | width × mcu_height × 2 bytes |
| Cb/Cr strips | Per MCU row, reused | (width/h_samp) × (mcu_height/v_samp) × 2 bytes each |
| Upsampled Cb/Cr | Per MCU row, reused | width × mcu_height × 2 bytes each (if subsampled) |
| **Total working memory** | | **O(width × mcu_height)** ≈ 16 rows max |

**Key**: We NEVER store full-image RGB. The parser may store coefficients for progressive, but that's unavoidable and already happens.

### Type Design

```rust
/// Source of DCT coefficients for an MCU row.
enum CoeffSource<'a> {
    /// Streaming: decode from bitstream on-demand.
    /// Coefficients are decoded directly into the MCU row buffer.
    Streaming(StreamingSource<'a>),
    
    /// Buffered: read from pre-decoded storage.
    /// Used for progressive JPEGs where all scans must complete first.
    Buffered(BufferedSource<'a>),
}

struct StreamingSource<'a> {
    data: &'a [u8],
    scan_data_start: usize,
    decoder_state: Option<EntropyDecoderState>,
    dc_tables: [Option<HuffmanDecodeTable>; 4],
    ac_tables: [Option<HuffmanDecodeTable>; 4],
    table_mapping: [(usize, usize); 4],  // per component
    restart_interval: u16,
    mcu_count: u32,
    next_restart_num: u8,
}

struct BufferedSource<'a> {
    /// Reference to parser's coefficient storage (not owned).
    coeffs: &'a [Vec<[i16; 64]>],
    coeff_counts: &'a [Vec<u8>],
    /// Current block index per component.
    block_indices: [usize; 4],
}

/// Unified scanline reader.
pub struct ScanlineReader<'a> {
    // Coefficient source (the only mode-specific part)
    source: CoeffSource<'a>,
    
    // Image metadata (shared)
    width: u32,
    height: u32,
    num_components: u8,
    subsampling: Subsampling,
    is_xyb: bool,
    
    // Component info
    h_samp: [u8; 4],
    v_samp: [u8; 4],
    quant_tables: [Option<[u16; 64]>; 4],
    quant_indices: [usize; 4],
    
    // MCU structure
    mcu_cols: usize,
    mcu_height: usize,  // 8 or 16
    strip_width: usize,
    
    // Position tracking
    current_row: usize,
    current_mcu_row: usize,
    row_in_mcu: usize,
    mcu_row_decoded: bool,
    
    // Strip buffers (reused per MCU row)
    y_strip: Vec<i16>,
    cb_strip: Vec<i16>,
    cr_strip: Vec<i16>,
    k_strip: Vec<i16>,  // For CMYK
    
    // Upsampled chroma (if subsampled)
    cb_upsampled: Vec<i16>,
    cr_upsampled: Vec<i16>,
    
    // Working buffers (reused per block)
    coeffs_buf: [i16; 64],
    dequant_buf: [i32; 64],
    prev_coeff_counts: [u8; 4],
}
```

### Unified Pipeline

```rust
impl<'a> ScanlineReader<'a> {
    /// Decode current MCU row - works identically for both modes.
    fn decode_mcu_row(&mut self) -> Result<()> {
        if self.mcu_row_decoded {
            return Ok(());
        }
        
        for mcu_x in 0..self.mcu_cols {
            // Handle restart markers (streaming only, no-op for buffered)
            self.source.handle_restart_if_needed(&mut self.prev_coeff_counts)?;
            
            for comp in 0..self.num_components as usize {
                let h_blocks = self.h_samp[comp] as usize;
                let v_blocks = self.v_samp[comp] as usize;
                
                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        // Get coefficients - abstracted over source
                        let coeff_count = self.source.decode_block_into(
                            &mut self.coeffs_buf,
                            self.prev_coeff_counts[comp],
                            comp,
                        )?;
                        self.prev_coeff_counts[comp] = 
                            self.prev_coeff_counts[comp].max(coeff_count);
                        
                        // Dequantize (shared)
                        dequantize_unzigzag_i32_into(
                            &self.coeffs_buf,
                            &self.quant_tables[self.quant_indices[comp]].unwrap(),
                            &mut self.dequant_buf,
                        );
                        
                        // IDCT into strip buffer (shared)
                        let (strip, stride) = self.get_strip_for_component(comp, mcu_x, h, v);
                        idct_int_tiered(&mut self.dequant_buf, strip, stride, coeff_count);
                    }
                }
            }
        }
        
        // Upsample chroma if needed (shared)
        self.upsample_chroma_if_needed();
        self.mcu_row_decoded = true;
        Ok(())
    }
    
    /// All read_rows_* methods use the same pattern:
    pub fn read_rows_rgb8(&mut self, mut output: ImgRefMut<'_, u8>) -> Result<usize> {
        let mut rows_written = 0;
        
        while rows_written < output.height() && !self.is_finished() {
            self.decode_mcu_row()?;
            
            // Get YCbCr for current row (always available, both modes)
            let (y, cb, cr) = self.get_ycbcr_row(self.row_in_mcu);
            
            // Convert to output format
            ycbcr_to_rgb8(y, cb, cr, output.row_mut(rows_written));
            
            self.advance_row();
            rows_written += 1;
        }
        
        Ok(rows_written)
    }
    
    /// YCbCr planes - no conversion needed, direct from strips
    pub fn read_rows_ycbcr_planes(&mut self, ...) -> Result<usize> {
        // Same loop, but copy from strips directly
        // NO RGB→YCbCr conversion ever happens
    }
    
    /// Grayscale - direct from Y strip
    pub fn read_rows_gray8(&mut self, ...) -> Result<usize> {
        // Same loop, but copy Y strip directly
        // NO RGB→Y conversion ever happens
    }
}
```

### CoeffSource Trait/Enum Methods

```rust
impl<'a> CoeffSource<'a> {
    /// Decode one block into the provided buffer.
    /// For streaming: entropy decode from bitstream.
    /// For buffered: copy from pre-decoded storage.
    fn decode_block_into(
        &mut self,
        coeffs: &mut [i16; 64],
        prev_count: u8,
        component: usize,
    ) -> Result<u8>;
    
    /// Handle restart marker if needed.
    /// For streaming: check interval, read marker, reset DC.
    /// For buffered: no-op (no restart markers in coefficient storage).
    fn handle_restart_if_needed(&mut self, prev_counts: &mut [u8; 4]) -> Result<()>;
    
    /// Check if source is exhausted.
    fn is_finished(&self) -> bool;
}
```

## Implementation Phases

### Phase 1: Fix API Bugs (Non-Breaking)

**Goal**: Fix incorrect behavior without changing architecture.

1. Pass actual `Subsampling` to `new_buffered()` instead of hardcoding `S444`
2. Store YCbCr planes in buffered mode alongside RGB (temporary duplication)
3. Use stored YCbCr for `read_rows_ycbcr_planes()` and `read_rows_gray*()`

**Effort**: Small, low risk
**Outcome**: API behaves correctly, but still has two code paths

### Phase 2: Unify Architecture

**Goal**: Single code path for both modes.

1. Create `CoeffSource` enum with `Streaming` and `Buffered` variants
2. Implement `decode_block_into()` for both variants
3. Refactor `decode_mcu_row()` to use `CoeffSource` abstraction
4. Remove `buffered_rgb` field entirely
5. Single constructor that takes `CoeffSource`

**Effort**: Medium, requires careful refactoring
**Outcome**: One code path, identical behavior guaranteed

### Phase 3: Expand Streaming Support

**Goal**: Reduce cases that need buffered mode.

1. **CMYK streaming**: Add `k_strip` buffer, 4-component decode loop
2. **High sampling factors**: Generalize upsampling for >2x2
3. **Arithmetic coding**: Already works, ensure streaming path exists

After this, only **progressive** JPEGs need buffered mode (fundamental limitation).

**Effort**: Medium
**Outcome**: Streaming for all sequential JPEGs

### Phase 4: Performance Optimization

**Goal**: Fastest decoder possible.

1. **Batch IDCT**: Process multiple blocks with SIMD (AVX2/AVX-512)
2. **Fused IDCT+Dequant**: Single pass over coefficients
3. **Fused YCbCr→RGB**: SIMD color conversion, process 8+ pixels at once
4. **Prefetching**: Hint next MCU row's coefficients while processing current
5. **Memory pools**: Reuse allocations across decode calls

**Effort**: Large, performance-focused
**Outcome**: Competitive with fastest C decoders

### Phase 5: Advanced Features (Optional)

1. **Progressive preview**: Output partial results after each scan
2. **Coefficient access**: Allow callers to access raw DCT coefficients
3. **Custom IDCT**: Pluggable IDCT for reduced-size decode (1/2, 1/4, 1/8)
4. **Parallel MCU rows**: Decode multiple MCU rows concurrently

## API Changes

### Before (Current)

```rust
// Internal: two constructors, different behavior
pub(crate) fn new(...) -> Result<Self>;           // Streaming
pub(crate) fn new_buffered(...) -> Self;          // Buffered

// Public: same signature, DIFFERENT BEHAVIOR
pub fn subsampling(&self) -> Subsampling;         // Wrong in buffered!
pub fn read_rows_ycbcr_planes(...) -> Result<usize>;  // Lossy in buffered!
```

### After (Unified)

```rust
// Internal: single constructor
pub(crate) fn new(source: CoeffSource<'a>, metadata: ImageMetadata) -> Result<Self>;

// Public: same signature, IDENTICAL BEHAVIOR
pub fn subsampling(&self) -> Subsampling;         // Always correct
pub fn read_rows_ycbcr_planes(...) -> Result<usize>;  // Always native YCbCr

// New: query internal mode if needed
pub fn is_streaming(&self) -> bool;
```

## Memory Comparison

| Image | Current Buffered | Unified |
|-------|------------------|---------|
| 4K (3840×2160) | 24.9 MB RGB | ~123 KB strips |
| 8K (7680×4320) | 99.5 MB RGB | ~246 KB strips |
| Progressive 4K | 24.9 MB RGB | ~9.4 MB coeffs* |

*Progressive must store coefficients regardless (format limitation), but we never store RGB.

## Testing Strategy

1. **Parity tests**: Same input → identical output for streaming vs buffered
2. **Precision tests**: `read_rows_ycbcr_planes()` matches decoder's native YCbCr
3. **Memory tests**: Verify O(width × mcu_height) working memory
4. **Fuzz tests**: Random JPEGs with various modes and subsampling
5. **Benchmark**: Compare unified vs current, ensure no regression

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Performance regression | Benchmark before/after, profile hot paths |
| Subtle precision changes | Bit-exact tests against current streaming output |
| Progressive edge cases | Extensive test suite with real progressive JPEGs |
| CMYK color accuracy | Test against reference decoder (libjpeg-turbo) |

## Success Criteria

1. ✓ `subsampling()` returns correct value for all JPEGs
2. ✓ `read_rows_ycbcr_planes()` returns native YCbCr (no round-trip)
3. ✓ `read_rows_gray*()` returns Y channel directly (no RGB→Y)
4. ✓ Memory usage is O(width × mcu_height) for all modes
5. ✓ No code duplication between modes
6. ✓ Performance matches or exceeds current implementation

## Lifetime Considerations

### Current Problem

The current architecture has an ownership issue:

```rust
// In Decoder::scanline_reader():
let mut parser = JpegParser::new(data, ...)?;
parser.decode(&Unstoppable)?;  // For progressive

// Problem: parser owns coeffs, but we want ScanlineReader to borrow them
// Current solution: clone/move the RGB buffer
let pixels = parser.to_pixels(...)?;  // Converts and MOVES out
return Ok(ScanlineReader::new_buffered(data, ..., pixels, ...));
// parser is dropped, coefficients lost
```

### Solution: Keep Parser Alive

For buffered mode, the `ScanlineReader` needs to borrow from the parser:

```rust
/// Holds the parser and provides scanline access to its data.
pub struct ScanlineReader<'a> {
    /// For buffered mode: owns the parser to keep coefficients alive.
    /// For streaming mode: None (coefficients decoded on-demand).
    parser: Option<JpegParser<'a>>,
    
    /// Coefficient source (borrows from parser or decodes from data).
    source: CoeffSource<'a>,
    
    // ... rest of fields
}
```

Or use a different pattern:

```rust
/// Parser that can transition to scanline reading mode.
pub struct JpegParser<'a> { ... }

impl<'a> JpegParser<'a> {
    /// Consume parser and return scanline reader.
    /// For progressive: coefficients are already decoded and owned.
    /// For streaming: returns reader that decodes on-demand.
    pub fn into_scanline_reader(self) -> Result<ScanlineReader<'a>> {
        match self.mode {
            JpegMode::Progressive => {
                // Self-referential: reader owns coefficients it reads from
                ScanlineReader::from_parser(self)
            }
            _ => {
                // Streaming: reader borrows data slice, parser can be dropped
                ScanlineReader::streaming(self.data, ...)
            }
        }
    }
}
```

### Self-Referential Alternative (with `ouroboros` or manual unsafe)

For zero-copy buffered mode, we need self-referential structs:

```rust
// Using ouroboros crate (safe self-referential structs)
#[self_referencing]
pub struct ScanlineReader<'a> {
    // Owned data
    parser: Option<JpegParser<'a>>,
    
    // Borrows from parser
    #[borrows(parser)]
    source: CoeffSource<'this>,
    
    // ... other fields that don't borrow
}
```

Or keep it simple: for Phase 2, the `BufferedSource` can own a clone of the coefficient data. Optimize later if profiling shows it matters.

## Decision: Owned vs Borrowed Coefficients

| Approach | Memory | Complexity | Performance |
|----------|--------|------------|-------------|
| Clone coefficients | 2x coeff memory briefly | Simple | Copy overhead |
| Own parser | 1x coeff memory | Moderate | Zero-copy |
| Self-referential | 1x coeff memory | Complex | Zero-copy |

**Recommendation**: Start with cloning (Phase 2), optimize to owned parser (Phase 4) if needed. The coefficients for a 4K progressive JPEG are ~9.4 MB - briefly having 2x that during construction is acceptable.

## Clarification: Streaming IS the Default for Sequential

The `ScanlineReader` already streams by default for sequential JPEGs. Current coverage:

| Format | Streaming? | Why Not? |
|--------|------------|----------|
| Baseline 4:4:4 | ✓ | |
| Baseline 4:2:2 | ✓ | |
| Baseline 4:2:0 | ✓ | |
| Baseline 4:4:0 | ✓ | |
| Grayscale | ✓ | |
| Baseline CMYK | ✗ | 4-component not implemented |
| Baseline >2x2 sampling | ✗ | Upsampling not implemented |
| Progressive (any) | ✗ | **Fundamental: needs all scans first** |
| Arithmetic sequential | ✗ | Falls through to coefficient storage |

**Goal**: Stream ALL sequential JPEGs (baseline + arithmetic + extended).
Only progressive requires buffering (unavoidable).

## SIMD-Aligned Buffer Strides

### Requirements

For AVX2 (256-bit) with i16 data: stride must be multiple of **16 pixels**
For AVX-512 (512-bit) with i16 data: stride must be multiple of **32 pixels**

### Current Problem

```rust
let strip_width = mcu_cols * mcu_width;  // Multiple of 8 or 16, not guaranteed 32
```

### Solution: Align to 32 pixels (covers both AVX2 and AVX-512)

```rust
/// Round up to next multiple of N
const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// SIMD alignment for strip buffers (32 pixels = 64 bytes for i16)
const STRIP_ALIGNMENT: usize = 32;

// In ScanlineReader::new():
let raw_strip_width = mcu_cols * mcu_width;
let strip_width = align_up(raw_strip_width, STRIP_ALIGNMENT);

// Chroma strips also aligned
let raw_chroma_width = mcu_cols * 8;
let chroma_strip_width = align_up(raw_chroma_width, STRIP_ALIGNMENT);
```

### Benefits

1. **IDCT**: Can process 16 or 32 coefficients at once without remainder handling
2. **Upsampling**: SIMD loops don't need scalar tail
3. **Color conversion**: Process 8-16 pixels per iteration cleanly
4. **Cache alignment**: 64-byte cache lines fit evenly

### Memory Overhead

Worst case: 31 extra pixels per row × mcu_height × sizeof(i16) × 3 components
= 31 × 16 × 2 × 3 = **2,976 bytes** extra for 4:2:0
= Negligible (<3KB)

### Implementation

```rust
pub struct ScanlineReader<'a> {
    // Actual image width (for output clipping)
    width: u32,
    
    // SIMD-aligned stride (for internal buffers)
    strip_stride: usize,  // >= width, multiple of 32
    
    // ...
}

impl<'a> ScanlineReader<'a> {
    fn get_ycbcr_row(&self, row_in_mcu: usize) -> (&[i16], &[i16], &[i16]) {
        let offset = row_in_mcu * self.strip_stride;
        let width = self.width as usize;
        
        // Return slices clipped to actual width (not stride)
        (
            &self.y_strip[offset..offset + width],
            &self.cb_upsampled[offset..offset + width],  // or cb_strip for 4:4:4
            &self.cr_upsampled[offset..offset + width],
        )
    }
    
    // SIMD functions use strip_stride for loop bounds
    fn idct_row_simd(&mut self, ...) {
        // Process in chunks of 32 (AVX-512) or 16 (AVX2)
        // No remainder handling needed because stride is aligned
    }
}
```

## Updated Phase Plan

### Phase 1: Fix Bugs + Add Alignment
1. Fix `subsampling()` bug in buffered mode
2. Add `strip_stride` with 32-pixel alignment
3. Update all strip access to use stride

### Phase 2: Unify + Expand Sequential Streaming  
1. Create `CoeffSource` abstraction
2. Add CMYK streaming (4th component strip)
3. Add arithmetic sequential streaming
4. Generalize upsampling for any sampling factors

### Phase 3+: (unchanged)

## Bug Found: Arithmetic JPEGs Broken in ScanlineReader

```rust
// Line 609 in mod.rs - only checks Progressive!
if parser.mode == JpegMode::Progressive || is_cmyk {
    // buffered mode
}
// Arithmetic sequential/progressive falls through to Huffman streaming path → CRASH
```

**Fix needed**: Add arithmetic modes to buffered path (for now), then implement arithmetic streaming.

```rust
// Correct check:
let needs_buffered = matches!(
    parser.mode,
    JpegMode::Progressive | JpegMode::ArithmeticSequential | JpegMode::ArithmeticProgressive
) || is_cmyk;

if needs_buffered {
    // Full decode + buffered mode
}
```

## Revised Streaming Support Matrix

| Mode | Current | After Unification |
|------|---------|-------------------|
| Baseline Huffman | ✓ Stream | ✓ Stream |
| Extended Huffman | ✓ Stream | ✓ Stream |
| Arithmetic Sequential | **✗ BROKEN** | ✓ Stream (Phase 2) |
| Progressive Huffman | ✓ Buffered | ✓ Buffered |
| Arithmetic Progressive | **✗ BROKEN** | ✓ Buffered |
| CMYK (any) | ✓ Buffered | ✓ Stream (Phase 3) |
