# Streaming Support Plan for jpegli-rs

**Date**: December 2024 (Updated December 28, 2024)
**Goal**: Add chunked/streaming APIs to jpegli-rs for server-side on-the-fly encoding/decoding

## Executive Summary

| Phase | Effort | What It Does |
|-------|--------|--------------|
| **Phase 1: Chunked Encoder** | ~1 week | Accept pixel rows incrementally, emit progressive JPEG at end |
| **Phase 2: Streaming Decoder** | ~1 week | Accept JPEG bytes incrementally, provide progressive preview |
| **Phase 3: True Streaming** | ~1 week (optional) | Real-time baseline-only streaming for both encoder and decoder |

**Key Insight**: Progressive JPEG cannot truly stream (all data needed before any output).
"Chunked" mode saves memory by processing rows as they arrive but still produces output at the end.

**Work Location**: All changes go in jpegli-rs at `/home/lilith/work/jpegli-rs/jpegli-rs/src/`

## Requirements

- Server-side on-the-fly encoding/decoding
- Both encoder AND decoder streaming support
- Progressive JPEG support (not just baseline)
- Preserve f32 precision throughout pipeline (no quant loss from intermediate conversions)

## Important: jpegli-rs Already Has a Full Decoder!

**Location**: `/home/lilith/work/jpegli-rs/jpegli-rs/src/`

The jpegli-rs library already includes:
- `decode.rs` (46,920 lines) - Complete JPEG decoder with marker parsing
- `idct.rs` (11,716 lines) - Inverse DCT
- `entropy.rs` (45,582 lines) - Entropy encoding/decoding
- `color.rs` (20,138 lines) - Color space conversion
- `quant.rs` (39,269 lines) - Quantization/dequantization

**This plan is about adding STREAMING APIs to the existing encoder/decoder, NOT building a decoder from scratch.**

The zenjpeg fork (`/home/lilith/work/zenjpeg/src/jpegli/`) is encoder-focused and has only partial decoder support - but the original jpegli-rs has everything.

## Current Architecture

### Location: `/home/lilith/work/jpegli-rs/jpegli-rs/src/`

Both encoder and decoder are **complete** with f32 precision throughout.

### Encoder (Complete)

**Current API**:
```rust
// Requires entire image upfront
let encoder = Encoder::new()
    .quality(Quality::Standard(90))
    .subsampling(Subsampling::S444);
let jpeg = encoder.encode(&pixels, width, height)?;
```

**Key Files**:
| File | Lines | Purpose |
|------|-------|---------|
| `encode.rs` | 116,126 | Main encoder pipeline |
| `huffman_opt.rs` | 74,082 | Two-pass Huffman optimization |
| `entropy.rs` | 45,582 | Huffman encoding/decoding |
| `adaptive_quant.rs` | 31,244 | Per-block AQ strength |
| `quant.rs` | 39,269 | Quantization with zero-biasing |
| `color.rs` | 20,138 | RGB↔YCbCr (f32 precision) |
| `dct.rs` | 11,941 | Forward DCT |
| `idct.rs` | 11,716 | Inverse DCT |
| `bitstream.rs` | 10,763 | Bit-level I/O |
| `alloc.rs` | 11,349 | Memory management |

### Decoder (Complete)

**Current API**:
```rust
let decoder = Decoder::new()
    .apply_icc(true)
    .fancy_upsampling(true);
let decoded = decoder.decode(&jpeg_bytes)?;
// decoded.pixels: Vec<u8> or configurable format
```

**Features**:
- Full JPEG parsing (all markers)
- Baseline and progressive support
- ICC profile extraction and application
- XYB color space support
- Fancy upsampling
- Block smoothing
- DoS protection (max_pixels, max_memory limits)

**Memory Model** (for 4K image 3840×2160):
- Y plane (f32): 33 MB
- Cb plane (f32): 33 MB
- Cr plane (f32): 33 MB
- Coefficients: 50 MB
- **Peak: ~150 MB**

## Why Progressive Can't Truly Stream

### Encoder Limitation
Progressive encoding requires ALL blocks before ANY output:
1. All coefficients must be collected first
2. Huffman optimization needs full frequency data
3. Each scan references every block in the image

### Decoder Limitation
Progressive decoding can't produce final pixels until all scans received:
1. First scan: DC coefficients only (blocky preview)
2. Middle scans: Refine AC coefficients
3. Final output only after last scan

### Solution: "Chunked" Mode
- **Input**: Accept data incrementally (row-by-row)
- **Processing**: Buffer internally, process blocks as complete
- **Output**: Emit complete progressive JPEG at `finish()`

Memory savings: ~67% (eliminate f32 planes, keep coefficients)

## Implementation Plan

### Phase 1: Chunked Input Encoder (~3 days)

**Goal**: Accept pixel rows incrementally without holding full image in caller's memory.

**API**:
```rust
let mut encoder = ChunkedEncoder::new(width, height, config)?;

// Feed rows as they arrive (e.g., from chunked upload)
for chunk in incoming_row_chunks {
    encoder.write_rows(&chunk)?;  // Processes complete 8-row blocks
}

// Produces progressive JPEG after all data received
let jpeg = encoder.finish()?;
```

**Implementation Steps**:

1. **Add `ChunkedEncoder` struct** to `encode.rs`:
   ```rust
   pub struct ChunkedEncoder {
       config: EncoderConfig,
       width: u32,
       height: u32,
       // 8-row ring buffer for incoming pixels
       row_buffer: Vec<f32>,  // width * 8 * 3 components
       rows_buffered: usize,
       current_y: usize,
       // Store processed coefficients
       y_coeffs: Vec<[i16; 64]>,
       cb_coeffs: Vec<[i16; 64]>,
       cr_coeffs: Vec<[i16; 64]>,
       // DC predictors for differential encoding
       prev_dc: [i16; 3],
   }
   ```

2. **Implement `write_rows()`**:
   - Convert incoming RGB rows to YCbCr (f32)
   - Append to 8-row buffer
   - When 8 rows complete: extract blocks, DCT, quantize
   - Store coefficients, clear buffer, continue

3. **Implement `finish()`**:
   - Handle final partial row block (pad if needed)
   - Run Huffman frequency collection on all coefficients
   - Generate optimized Huffman tables
   - Encode progressive scans
   - Return complete JPEG

**Files to Modify**:
- `src/jpegli/encode.rs` - Add ChunkedEncoder
- `src/jpegli/mod.rs` - Export new API
- `src/jpegli/types.rs` - Add ChunkedEncoderConfig if needed

**Memory Budget** (4K image):
- 8-row buffer: 3840 × 8 × 3 × 4 = 368 KB
- Coefficients: 480 × 270 × 64 × 2 × 3 = 50 MB
- **Peak: ~50 MB** (vs 150 MB current = 67% savings)

---

### Phase 2: Streaming Decoder API (~1 week)

**Goal**: Add incremental input and progressive preview to the EXISTING jpegli-rs decoder.

**Key Point**: jpegli-rs at `/home/lilith/work/jpegli-rs/jpegli-rs/src/decode.rs` already has:
- ✅ Complete JPEG parser (all markers)
- ✅ Inverse DCT (f32 precision)
- ✅ Color conversion (YCbCr→RGB, XYB support)
- ✅ Chroma upsampling
- ✅ Progressive scan handling
- ✅ ICC profile support

**What's Missing**: Incremental input API and progressive preview capability.

**Current API** (requires complete JPEG upfront):
```rust
let decoder = Decoder::new()
    .apply_icc(true)
    .fancy_upsampling(true);
let decoded = decoder.decode(&jpeg_bytes)?;  // Needs ALL bytes
```

**New Streaming API**:
```rust
let mut decoder = StreamingDecoder::new()?;

// Feed JPEG data as it arrives
for chunk in incoming_jpeg_chunks {
    decoder.write(&chunk)?;  // Buffers internally
}

// Get decoded pixels (f32 or u8)
let pixels = decoder.finish()?;

// OR for progressive preview during decode:
if let Some(preview) = decoder.current_preview()? {
    // Partially decoded image (blocky but viewable)
}
```

#### Changes Required

**1. Add Input Buffer to Decoder State** (~100 lines)
```rust
pub struct StreamingDecoder {
    // Buffer for incomplete JPEG data
    input_buffer: Vec<u8>,
    // Track parsing progress
    parse_state: ParseState,
    // Underlying decoder (existing)
    inner: Decoder,
    // For progressive: coefficients received so far
    partial_coeffs: Option<CoefficientBuffer>,
}

enum ParseState {
    AwaitingHeader,
    HeaderParsed { width: u32, height: u32 },
    DecodingScans { scans_complete: usize },
    Complete,
}
```

**2. Implement Incremental Parsing** (~200 lines)
```rust
impl StreamingDecoder {
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        self.input_buffer.extend_from_slice(data);

        // Try to advance parsing
        loop {
            match self.try_advance()? {
                Advance::NeedMoreData => break,
                Advance::HeaderParsed => continue,
                Advance::ScanComplete => continue,
                Advance::ImageComplete => break,
            }
        }
        Ok(())
    }

    fn try_advance(&mut self) -> Result<Advance> {
        // Attempt to parse next marker/segment from buffer
        // Use existing parsing logic from decode.rs
    }
}
```

**3. Add Progressive Preview** (~150 lines)
```rust
impl StreamingDecoder {
    /// Get preview from partial progressive data
    /// Returns None if no scans decoded yet
    pub fn current_preview(&self) -> Option<PreviewImage> {
        let coeffs = self.partial_coeffs.as_ref()?;

        // IDCT whatever coefficients we have
        // DC-only = very blocky but shows composition
        // More AC = more detail
        let pixels = self.inner.idct_partial(coeffs)?;

        Some(PreviewImage {
            width: self.width,
            height: self.height,
            pixels,
            scans_decoded: self.scans_complete,
            is_complete: false,
        })
    }
}
```

**4. Expose Partial IDCT** (~50 lines in existing decode.rs)
```rust
impl Decoder {
    /// IDCT with partial coefficients (for progressive preview)
    pub fn idct_partial(&self, coeffs: &CoefficientBuffer) -> Vec<u8> {
        // Same IDCT logic but works with incomplete data
        // Missing AC coefficients → blocky but valid output
    }
}
```

#### Files to Modify

| File | Changes |
|------|---------|
| `decode.rs` | Add `StreamingDecoder` wrapper, expose `idct_partial()` |
| `types.rs` | Add `PreviewImage`, `ParseState` types |
| `mod.rs` | Export `StreamingDecoder` |

#### Memory Considerations

Streaming decoder still needs to buffer:
- Input JPEG data (until parsed): variable, typically < 1 MB
- Coefficient buffer (for progressive): ~50 MB for 4K
- Output pixels: ~24 MB for 4K RGB8

Total peak: similar to current (~75 MB), but input can be discarded as parsed.

---

### Phase 3: True Baseline Streaming (~1 week, optional)

**Goal**: Real-time streaming for baseline-only mode.

**Encoder API**:
```rust
let mut encoder = StreamingEncoder::baseline(width, height)?;

// Callback receives JPEG chunks as they're ready
encoder.on_output(|chunk: &[u8]| {
    send_to_client(chunk);
});

for row in pixel_rows {
    encoder.write_scanline(&row)?;  // May trigger on_output
}

encoder.finish()?;  // Flushes remaining data
```

**Decoder API**:
```rust
let mut decoder = StreamingDecoder::new()?;

// Callback receives decoded rows as they're ready
decoder.on_scanline(|y: usize, pixels: &[u8]| {
    process_row(y, pixels);
});

for chunk in jpeg_chunks {
    decoder.write(&chunk)?;  // May trigger on_scanline
}
```

**Limitations**:
- Baseline only (no progressive)
- Fixed Huffman tables (no two-pass optimization)
- No adaptive quantization
- Memory: ~370 KB for 4K (just 8-row buffer)

---

## Memory Summary

| Mode | 4K Peak Memory | Savings |
|------|----------------|---------|
| Current (full buffer) | ~150 MB | baseline |
| Chunked Progressive | ~50 MB | 67% |
| True Baseline Streaming | ~370 KB | 99.7% |

---

## Testing Strategy

### Encoder Tests
1. **Round-trip**: ChunkedEncoder output matches regular Encoder
2. **Progressive validity**: Output decodes correctly with standard decoders
3. **Memory**: Verify peak allocation matches expected

### Decoder Tests
1. **Baseline JPEGs**: Decode corpus of baseline images, compare to reference
2. **Progressive JPEGs**: Decode progressive images, verify final output
3. **Preview mode**: Verify partial decodes produce valid (if blocky) output
4. **Incremental feeding**: Feed 1 byte at a time, verify same result
5. **Error handling**: Truncated files, corrupt markers, invalid Huffman

### Reference Decoders for Comparison
- `jpeg-decoder` crate (pure Rust)
- `djpeg` (libjpeg-turbo CLI)
- ImageMagick `convert`

---

## File Checklist

### jpegli-rs Files to Modify (`/home/lilith/work/jpegli-rs/jpegli-rs/src/`)

**Phase 1: Chunked Encoder**
- [ ] `encode.rs` - Add `ChunkedEncoder` struct and methods
- [ ] `types.rs` - Add `ChunkedEncoderConfig` if needed
- [ ] `mod.rs` - Export `ChunkedEncoder`

**Phase 2: Streaming Decoder**
- [ ] `decode.rs` - Add `StreamingDecoder` wrapper, expose `idct_partial()`
- [ ] `types.rs` - Add `PreviewImage`, `ParseState` types
- [ ] `mod.rs` - Export `StreamingDecoder`

**Phase 3: True Baseline Streaming (optional)**
- [ ] `encode.rs` - Add `StreamingEncoder` with output callback
- [ ] `decode.rs` - Add `StreamingDecoder` with scanline callback

### zenjpeg Changes (if wrapping jpegli-rs streaming)
- [ ] `src/lib.rs` - Re-export streaming APIs
- [ ] `src/encode.rs` - Forward to jpegli-rs ChunkedEncoder

---

## Implementation Order

1. **Week 1**: ChunkedEncoder (Phase 1)
   - Day 1-2: Basic `ChunkedEncoder` struct, 8-row buffer, block extraction
   - Day 3: Integration with existing quantization/DCT pipeline
   - Day 4: `finish()` with Huffman optimization + progressive encoding
   - Day 5: Tests, edge cases (partial rows, small images, 1x1 images)

2. **Week 2**: Streaming Decoder (Phase 2)
   - Day 1-2: `StreamingDecoder` wrapper with input buffer
   - Day 3: Incremental parsing that advances on `write()`
   - Day 4: Progressive preview with partial IDCT
   - Day 5: Tests with truncated JPEGs, byte-at-a-time feeding

3. **Week 3** (optional): True Baseline Streaming (Phase 3)
   - Day 1-2: Baseline streaming encoder with output callback
   - Day 3-4: Baseline streaming decoder with scanline callback
   - Day 5: Memory verification, performance benchmarks
