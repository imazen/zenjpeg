# Streaming Support Plan for jpegli-rs

**Date**: December 2024
**Goal**: Add chunked/streaming APIs to jpegli-rs for server-side on-the-fly encoding/decoding

## Requirements

- Server-side on-the-fly encoding/decoding
- Both encoder AND decoder streaming support
- Progressive JPEG support (not just baseline)
- Build decoder natively in jpegli-rs (not wrap external crate)
- Preserve f32 precision throughout pipeline (no quant loss from intermediate conversions)

## Current Architecture

### Encoder Status: Complete, Non-Streaming

Location: `src/jpegli/`

**Current API**:
```rust
// Requires entire image upfront
pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>>
```

**Memory Model** (for 4K image 3840×2160):
- Y plane (f32): 33 MB
- Cb plane (f32): 33 MB
- Cr plane (f32): 33 MB
- Coefficients: 50 MB
- **Peak: ~150 MB**

**Key Files**:
| File | Lines | Purpose |
|------|-------|---------|
| `encode.rs` | 2,721 | Main encoder pipeline |
| `huffman_opt.rs` | 2,255 | Two-pass Huffman optimization |
| `entropy.rs` | 1,242 | Huffman encoding + partial decoding |
| `adaptive_quant.rs` | 917 | Per-block AQ strength |
| `quant.rs` | 949 | Quantization, zero-biasing |
| `color.rs` | 665 | RGB↔YCbCr (f32 precision) |
| `dct.rs` | 397 | Forward DCT |
| `alloc.rs` | 390 | Memory management |
| `bitstream.rs` | 375 | Bit-level I/O |

### Decoder Status: Partial

**What Exists** (in `entropy.rs:811-1100`):
- ✅ `BitReader` with byte unstuffing (0xFF 0x00 → 0xFF)
- ✅ `HuffmanDecodeTable` with 16-bit fast lookup
- ✅ `decode_block()` for baseline and progressive
- ✅ DC differential decoding
- ✅ AC run-length decoding
- ✅ Progressive refinement decoding

**What's Missing**:
- ❌ JPEG file parsing (markers, headers)
- ❌ Inverse DCT
- ❌ Dequantization
- ❌ YCbCr → RGB color conversion
- ❌ Chroma upsampling
- ❌ Decoder orchestration

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

### Phase 2: Full JPEG Decoder (~2-3 weeks)

**Goal**: Complete native JPEG decoder with incremental input support.

**API**:
```rust
let mut decoder = Decoder::new()?;

// Feed JPEG data as it arrives
for chunk in incoming_jpeg_chunks {
    decoder.write(&chunk)?;
}

// Get decoded pixels (f32 or u8)
let pixels = decoder.finish()?;

// OR for progressive preview during decode:
if let Some(preview) = decoder.current_preview()? {
    // Partially decoded image (blocky but viewable)
}
```

#### Component 1: JPEG Parser (`src/jpegli/parser.rs` ~600 lines)

**Marker Handling**:
```rust
pub struct JpegParser {
    state: ParserState,
    buffer: Vec<u8>,
    // Extracted metadata
    frame: Option<FrameHeader>,
    quant_tables: [Option<[u16; 64]>; 4],
    huff_tables: HuffmanTables,
    scans: Vec<ScanHeader>,
}

enum ParserState {
    ExpectingMarker,
    ReadingSegmentLength,
    ReadingSegment { marker: u8, remaining: usize },
    ReadingEntropyData,
    Complete,
}
```

**Markers to Handle**:
| Marker | Code | Purpose |
|--------|------|---------|
| SOI | 0xFFD8 | Start of image |
| EOI | 0xFFD9 | End of image |
| SOF0 | 0xFFC0 | Baseline frame header |
| SOF2 | 0xFFC2 | Progressive frame header |
| DHT | 0xFFC4 | Huffman table definition |
| DQT | 0xFFDB | Quantization table definition |
| DRI | 0xFFDD | Restart interval |
| SOS | 0xFFDA | Start of scan |
| RST0-7 | 0xFFD0-D7 | Restart markers |
| APP0 | 0xFFE0 | JFIF metadata |
| APP1 | 0xFFE1 | EXIF metadata |
| APP2 | 0xFFE2 | ICC profile |
| COM | 0xFFFE | Comment |

**Frame Header Parsing**:
```rust
struct FrameHeader {
    precision: u8,        // Usually 8
    height: u16,
    width: u16,
    components: Vec<ComponentInfo>,
}

struct ComponentInfo {
    id: u8,
    h_sampling: u8,       // 1-4
    v_sampling: u8,       // 1-4
    quant_table_id: u8,   // 0-3
}
```

**Scan Header Parsing**:
```rust
struct ScanHeader {
    components: Vec<ScanComponent>,
    spectral_start: u8,   // Ss (0 for DC, 1-63 for AC)
    spectral_end: u8,     // Se (0 for DC, 63 for full AC)
    approx_high: u8,      // Ah (successive approximation)
    approx_low: u8,       // Al
}

struct ScanComponent {
    component_id: u8,
    dc_table_id: u8,
    ac_table_id: u8,
}
```

#### Component 2: Inverse DCT (`src/jpegli/idct.rs` ~300 lines)

**Reference Implementation** (mirror of forward DCT):
```rust
/// Inverse DCT using Loeffler algorithm
/// Input: quantized coefficients in zigzag order
/// Output: 8x8 pixel block (f32, range 0-255)
pub fn inverse_dct_8x8(coeffs: &[i16; 64], quant: &[u16; 64]) -> [f32; 64] {
    // 1. Dequantize: multiply by quant table
    let mut block = [0.0f32; 64];
    for i in 0..64 {
        let zigzag_pos = ZIGZAG[i];
        block[zigzag_pos] = coeffs[i] as f32 * quant[i] as f32;
    }

    // 2. 1D IDCT on rows
    for row in 0..8 {
        idct_1d(&mut block[row * 8..row * 8 + 8]);
    }

    // 3. 1D IDCT on columns (transpose access)
    // ... similar with column stride

    // 4. Level shift: add 128
    for val in &mut block {
        *val += 128.0;
    }

    block
}
```

**SIMD Version** (optional, for performance):
- Use `wide::f32x4` or `wide::f32x8` for parallel computation
- Match pattern in existing `dct.rs` forward DCT

#### Component 3: Color Conversion (`src/jpegli/color.rs` additions ~150 lines)

**YCbCr → RGB** (add to existing color.rs):
```rust
/// Convert YCbCr to RGB (BT.601 JPEG standard)
/// All values in 0-255 range
#[inline]
pub fn ycbcr_to_rgb_f32(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let cb = cb - 128.0;
    let cr = cr - 128.0;

    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;

    (r.clamp(0.0, 255.0), g.clamp(0.0, 255.0), b.clamp(0.0, 255.0))
}
```

**Chroma Upsampling** (for 4:2:0, 4:2:2):
```rust
/// Upsample chroma plane by 2x in both dimensions (4:2:0 → 4:4:4)
pub fn upsample_2x2(input: &[f32], in_w: usize, in_h: usize) -> Vec<f32> {
    let out_w = in_w * 2;
    let out_h = in_h * 2;
    let mut output = vec![0.0; out_w * out_h];

    // Bilinear interpolation
    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = (x as f32) / 2.0;
            let src_y = (y as f32) / 2.0;
            // ... bilinear sample from input
        }
    }
    output
}

/// Upsample chroma plane by 2x horizontally only (4:2:2 → 4:4:4)
pub fn upsample_2x1(input: &[f32], in_w: usize, in_h: usize) -> Vec<f32> {
    // Similar, only horizontal interpolation
}
```

#### Component 4: Decoder Orchestration (`src/jpegli/decode.rs` ~800 lines)

**Main Decoder Struct**:
```rust
pub struct Decoder {
    parser: JpegParser,
    // Coefficient storage (for progressive)
    coefficients: Option<CoefficientBuffer>,
    // Decode state
    state: DecodeState,
    // Output configuration
    output_format: OutputFormat,
}

struct CoefficientBuffer {
    width_in_blocks: usize,
    height_in_blocks: usize,
    // One buffer per component
    components: Vec<Vec<[i16; 64]>>,
}

enum DecodeState {
    Parsing,
    DecodingBaseline { mcu_row: usize },
    DecodingProgressive { scan_index: usize },
    Complete,
}

pub enum OutputFormat {
    Rgb8,      // Vec<u8>, 3 bytes per pixel
    Rgba8,     // Vec<u8>, 4 bytes per pixel
    RgbF32,    // Vec<f32>, 3 values per pixel (preserve precision)
    Gray8,     // Vec<u8>, 1 byte per pixel
}
```

**Decode Pipeline**:
```rust
impl Decoder {
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        // 1. Feed data to parser
        self.parser.write(data)?;

        // 2. Process any complete scans
        while let Some(scan) = self.parser.next_scan()? {
            self.decode_scan(&scan)?;
        }

        Ok(())
    }

    pub fn finish(self) -> Result<DecodedImage> {
        // 1. Ensure we have complete image
        if self.state != DecodeState::Complete {
            return Err(Error::IncompleteImage);
        }

        // 2. IDCT all blocks
        let planes = self.idct_all_blocks()?;

        // 3. Upsample chroma if needed
        let (y, cb, cr) = self.upsample_chroma(planes)?;

        // 4. Convert to output format
        self.convert_to_output(y, cb, cr)
    }

    pub fn current_preview(&self) -> Option<DecodedImage> {
        // Decode whatever we have so far
        // DC-only gives blocky preview
        // More scans = more detail
    }
}
```

**MCU Decoding** (reuse existing entropy.rs):
```rust
fn decode_mcu(&mut self, scan: &ScanHeader) -> Result<()> {
    let entropy = &mut self.entropy_decoder;

    for comp_idx in 0..scan.components.len() {
        let comp = &scan.components[comp_idx];
        let blocks_h = self.frame.components[comp_idx].h_sampling;
        let blocks_v = self.frame.components[comp_idx].v_sampling;

        for v in 0..blocks_v {
            for h in 0..blocks_h {
                // Use existing decode_block from entropy.rs
                let block = entropy.decode_block(
                    comp.dc_table_id,
                    comp.ac_table_id,
                    scan.spectral_start,
                    scan.spectral_end,
                    scan.approx_high,
                    scan.approx_low,
                )?;

                // Store in coefficient buffer
                self.store_block(comp_idx, block_x, block_y, block)?;
            }
        }
    }
    Ok(())
}
```

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

### New Files
- [ ] `src/jpegli/parser.rs` - JPEG marker parsing (~600 lines)
- [ ] `src/jpegli/decode.rs` - Decoder orchestration (~800 lines)
- [ ] `src/jpegli/idct.rs` - Inverse DCT (~300 lines)

### Modified Files
- [ ] `src/jpegli/encode.rs` - Add ChunkedEncoder
- [ ] `src/jpegli/color.rs` - Add YCbCr→RGB, upsampling
- [ ] `src/jpegli/quant.rs` - Add dequantization helper
- [ ] `src/jpegli/entropy.rs` - Expose decode primitives publicly
- [ ] `src/jpegli/mod.rs` - Export new APIs
- [ ] `src/jpegli/types.rs` - Add decoder config types

---

## Implementation Order

1. **Week 1**: ChunkedEncoder (Phase 1)
   - Day 1-2: Basic struct, row buffer, block extraction
   - Day 3: Integration with existing quantization/DCT
   - Day 4: finish() with Huffman opt + progressive encoding
   - Day 5: Tests, edge cases (partial rows, small images)

2. **Week 2-3**: Decoder Parser + IDCT (Phase 2a)
   - Day 1-3: JPEG parser with all marker handling
   - Day 4-5: Inverse DCT (scalar reference)
   - Day 6-7: Integration tests with baseline JPEGs

3. **Week 3-4**: Decoder Orchestration (Phase 2b)
   - Day 1-2: Baseline decode pipeline
   - Day 3-4: Progressive scan accumulation
   - Day 5: Color conversion + upsampling
   - Day 6-7: Preview mode, final testing

4. **Week 5** (optional): True Streaming (Phase 3)
   - Baseline streaming encoder
   - Baseline streaming decoder
   - Memory verification
