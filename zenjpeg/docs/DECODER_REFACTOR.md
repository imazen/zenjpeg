# Decoder Refactor Design

Status: Draft
Author: Claude
Date: 2026-01-22

## Executive Summary

Refactor the zenjpeg decoder to achieve zune-jpeg-level performance while maintaining the codec-design API guidelines. Key changes:

1. **True streaming decode** - MCU-row-at-a-time, no coefficient buffering for baseline
2. **Integer IDCT** - Replace f32 IDCT with tiered i32 IDCT (1x1, 4x4, 8x8)
3. **Fast AC lookup** - Single-table decode + sign extend
4. **Layered API** - Simple functions + builder + streaming

## Performance Analysis: zune-jpeg vs zenjpeg

### Why zune-jpeg is Fast

1. **True Streaming** - Decodes one MCU row, immediately converts to RGB, never stores coefficients for baseline
2. **Fast AC Lookup** - `ac_lookup` table decodes huffman symbol AND performs sign extension in one operation
3. **Integer IDCT** - Uses i32 IDCT with:
   - 1x1 shortcut when only DC coefficient present (DC-only blocks)
   - 4x4 shortcut when only top-left 4x4 coefficients present
   - Full 8x8 only when needed
   - SIMD (AVX2/NEON) with zero-coefficient short-circuit check
4. **Efficient Bitstream** - 4-byte refill path when no 0xFF markers, MSB bit layout
5. **Minimal Allocations** - Pre-allocated MCU row buffers, no per-block allocations

### Current zenjpeg Bottlenecks

1. **Coefficient Storage** - Stores all coefficients before IDCT (except ScanlineReader)
2. **Float IDCT** - Uses f32 IDCT (slower than integer, no shortcuts)
3. **Two-Step AC Decode** - Separate huffman decode + sign extend
4. **Bitstream Overhead** - More function call overhead in bit reading
5. **No Progressive Streaming** - Progressive must buffer all coefficients (inherent)

## Benchmark Results (2026-01-22)

Run with: `cargo bench -p zenjpeg --bench decode_compare --features decoder`

| Size | Mode | zune-jpeg | zenjpeg | Ratio |
|------|------|-----------|-----------|-------|
| 256x256 | baseline | 94 µs | 554 µs | **5.9x slower** |
| 256x256 | progressive | 214 µs | 758 µs | **3.5x slower** |
| 512x512 | baseline | 272 µs | 5.6 ms | **20x slower** |
| 1024x1024 | baseline | ~1 ms | ~23 ms | **23x slower** |
| 1024x1024 | progressive | 2.3 ms | 25 ms | **11x slower** |
| 2048x2048 | baseline | 3.7 ms | 93 ms | **25x slower** |
| 2048x2048 | progressive | 9.2 ms | 100 ms | **11x slower** |

**Key findings:**
- Baseline is 20-25x slower than zune-jpeg (gap increases with image size)
- Progressive is "only" 10x slower (coefficient buffering inherent in both)
- Gap increases with image size, suggesting O(n) overhead per pixel

### Root Cause Analysis

Code review of `parser.rs:to_pixels()` reveals these bottlenecks:

1. **Pixel-by-pixel upsampling loop** (parser.rs:1706-1715):
   ```rust
   for py in 0..height {
       for px in 0..width {
           upsampled[py * width + px] = comp_plane_f32[sy * info.comp_width + sx];
       }
   }
   ```
   - Division and min() per pixel
   - No SIMD vectorization
   - No copy_from_slice optimization

2. **f32 intermediate everywhere**:
   - Integer IDCT outputs i16
   - Immediately converted to f32 (line 1660)
   - Upsampling in f32
   - Color conversion in f32
   - Final clamp back to u8
   - zune-jpeg keeps i16/u8 throughout

3. **Bias stats gathering** (parser.rs:1566-1590):
   - Computed for EVERY block even for baseline
   - Only needed for progressive XYB quality
   - Adds ~20% overhead

4. **Double zigzag reorder**:
   - Once at line 1578 for stats
   - Again at line 1617 for IDCT
   - Scalar loop instead of LUT or SIMD

5. **Two-pass architecture**:
   - Pass 1: Buffer ALL coefficients (decode_scan)
   - Pass 2: IDCT + upsample + color convert (to_pixels)
   - zune-jpeg: single pass per MCU row

### What Phase 1 Already Has (but isn't using effectively)

The code already has fast AC lookup and tiered IDCT implemented:
- `HuffmanDecodeTable::fast_decode_ac()` at huffman/encode.rs:397
- `idct_int_tiered()` at decode/idct_int.rs:740

But these are undermined by the f32 intermediate and two-pass architecture

## Architecture

### Decode Paths

```
                    ┌─────────────────────────────────────────────────┐
                    │                   Input JPEG                     │
                    └─────────────────┬───────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────────────┐
                    │               JpegParser                         │
                    │  - Read markers (SOF, DHT, DQT, SOS)            │
                    │  - Determine mode (baseline/progressive)         │
                    │  - Build Huffman tables                          │
                    └─────────────────┬───────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Baseline 4:4:4 │     │ Baseline 4:2:0  │     │   Progressive   │
    │   (Streaming)   │     │   (Streaming)   │     │   (Buffered)    │
    └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
             │                       │                       │
             ▼                       ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │ Per-MCU-row:    │     │ Per-MCU-row:    │     │ Buffer all      │
    │ - Decode coeffs │     │ - Decode coeffs │     │ coefficients    │
    │ - IDCT to strip │     │ - IDCT to strip │     │ across scans    │
    │ - YCbCr→RGB row │     │ - Upsample      │     │                 │
    │ - Output        │     │ - YCbCr→RGB row │     │ Then process    │
    └─────────────────┘     │ - Output        │     │ like baseline   │
                            └─────────────────┘     └─────────────────┘
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Public API Layer                             │
├──────────────┬──────────────┬──────────────┬───────────────────────┤
│ decode_rgb() │ decode_into()│ Decoder      │ StreamingDecoder      │
│ decode_rgba()│              │ builder      │                       │
└──────────────┴──────────────┴──────────────┴───────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                        Core Decoder                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐│
│ │ JpegParser  │ │  Bitstream  │ │ HuffmanDec  │ │   IDCT          ││
│ │ (markers)   │ │  (bits)     │ │ (fast AC)   │ │ (tiered,SIMD)   ││
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘│
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐│
│ │ Dequantize  │ │  Upsample   │ │ YCbCr→RGB   │ │   Strip Mgr     ││
│ │ (unzigzag)  │ │  (SIMD)     │ │ (SIMD)      │ │   (buffers)     ││
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## API Design

### Simple One-Shot Functions

```rust
// Most common use case - RGB output
pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32)>;
pub fn decode_rgba(data: &[u8]) -> Result<(Vec<u8>, u32, u32)>;

// Typed pixel output (using rgb crate)
pub fn decode<P: DecodePixel>(data: &[u8]) -> Result<(Vec<P>, u32, u32)>;

// Zero-copy into existing buffer
pub fn decode_rgb_into(
    data: &[u8],
    output: &mut [u8],
    stride_bytes: u32,
) -> Result<(u32, u32)>;
```

### Info Before Decode

```rust
pub struct JpegInfo {
    pub width: u32,
    pub height: u32,
    pub num_components: u8,
    pub color_space: ColorSpace,
    pub mode: JpegMode,
    pub has_icc_profile: bool,
    pub is_xyb: bool,
}

impl JpegInfo {
    pub fn read(data: &[u8]) -> Result<Self>;

    /// Estimate memory for decode
    pub fn estimate_memory(&self) -> usize;
}
```

### Builder for Advanced Use

```rust
pub struct Decoder<'a> {
    data: &'a [u8],
    config: DecoderConfig,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self>;
    pub fn info(&self) -> &JpegInfo;

    // Configuration
    pub fn fancy_upsampling(self, enable: bool) -> Self;
    pub fn apply_icc(self, enable: bool) -> Self;
    pub fn max_memory(self, bytes: usize) -> Self;

    // Output
    pub fn decode_rgb(self) -> Result<DecodedImage>;
    pub fn decode_rgb_into(self, output: &mut [u8], stride: usize) -> Result<()>;
    pub fn decode_f32(self) -> Result<DecodedImageF32>;
    pub fn decode_ycbcr(self) -> Result<DecodedYCbCr>;
    pub fn decode_coefficients(self) -> Result<DecodedCoefficients>;
}
```

### Streaming Decode (Pull-Based)

```rust
pub struct ScanlineReader<'a> {
    // ... internal state
}

impl<'a> ScanlineReader<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self>;
    pub fn info(&self) -> ScanlineInfo;

    /// Read up to max_rows into output buffer
    /// Returns number of rows written
    pub fn read_rows_rgb(&mut self, output: &mut [u8], stride: usize, max_rows: usize) -> Result<usize>;
    pub fn read_rows_rgba(&mut self, output: &mut [u8], stride: usize, max_rows: usize) -> Result<usize>;

    /// Check if more rows available
    pub fn is_finished(&self) -> bool;
    pub fn rows_remaining(&self) -> usize;
}
```

## Implementation Details

### 1. Fast AC Huffman Table

Create a combined lookup table that decodes the huffman symbol AND performs sign extension:

```rust
/// Fast AC lookup table entry.
/// Layout: (value << 8) | (run << 4) | total_bits
/// - value: sign-extended coefficient value (-128..127)
/// - run: zero run length (0..15)
/// - total_bits: huffman bits + magnitude bits consumed
/// - Entry of 0 means fallback to slow path
pub struct FastAcTable {
    lookup: [i16; 512],  // 9-bit lookahead
}

impl FastAcTable {
    /// Try fast decode: returns Some((value, run, bits)) or None for fallback
    #[inline(always)]
    pub fn try_decode(&self, bits: u32) -> Option<(i16, u8, u8)> {
        let entry = self.lookup[(bits >> 23) as usize];  // Top 9 bits
        if entry == 0 {
            return None;  // Fallback to slow path
        }
        let value = entry >> 8;
        let run = ((entry >> 4) & 0xF) as u8;
        let total_bits = (entry & 0xF) as u8;
        Some((value, run, total_bits))
    }
}
```

Reference: zune-jpeg `huffman.rs:184-236` builds this table.

### 2. Tiered Integer IDCT

Implement three IDCT variants with automatic selection:

```rust
/// IDCT function pointer type
type IdctFn = fn(&[i32; 64], &mut [i16], usize);

/// Choose IDCT based on coefficient count
#[inline(always)]
fn choose_idct(coeff_count: u8) -> IdctFn {
    match coeff_count {
        0 | 1 => idct_1x1,      // DC only
        2..=10 => idct_4x4,     // Sparse (top-left 4x4)
        _ => idct_8x8,          // Full IDCT
    }
}

/// DC-only: just broadcast DC value
fn idct_1x1(coeffs: &[i32; 64], output: &mut [i16], stride: usize) {
    let dc = ((coeffs[0] + 4) >> 3) as i16;  // Divide by 8 and round
    for row in 0..8 {
        for col in 0..8 {
            output[row * stride + col] = dc;
        }
    }
}
```

Reference: zune-jpeg `idct/scalar.rs` and `idct/avx2.rs`.

### 3. SIMD IDCT with Zero Check

AVX2 implementation with early exit for zero coefficients:

```rust
#[target_feature(enable = "avx2")]
unsafe fn idct_avx2(coeffs: &mut [i32; 64], output: &mut [i16], stride: usize) {
    // Check if all AC coefficients are zero using SIMD OR
    let ac_chunk1 = _mm256_loadu_si256(coeffs[1..].as_ptr() as *const _);
    let ac_chunk2 = _mm256_loadu_si256(coeffs[9..].as_ptr() as *const _);
    // ... OR all chunks
    let all_zero = _mm256_testz_si256(combined, combined);

    if all_zero != 0 {
        // Fast path: DC only
        idct_1x1(coeffs, output, stride);
        return;
    }

    // Full IDCT using integer scaled arithmetic
    // ...
}
```

Reference: zune-jpeg `idct/avx2.rs:18-150`.

### 4. Efficient Bitstream Reader

Implement a bitstream reader with fast refill path:

```rust
pub struct Bitstream<'a> {
    data: &'a [u8],
    position: usize,
    buffer: u64,         // Bit buffer (MSB aligned)
    bits_left: u8,       // Bits remaining in buffer
}

impl<'a> Bitstream<'a> {
    /// Refill buffer to at least 32 bits
    #[inline(always)]
    fn refill(&mut self) {
        // Fast path: no 0xFF markers in next 4 bytes
        if self.position + 4 <= self.data.len() {
            let bytes = &self.data[self.position..self.position + 4];
            if !bytes.contains(&0xFF) {
                // Direct 4-byte read (big-endian)
                let word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                self.buffer |= (word as u64) << (32 - self.bits_left);
                self.bits_left += 32;
                self.position += 4;
                return;
            }
        }
        // Slow path: handle byte stuffing
        self.refill_slow();
    }

    /// Peek top N bits without consuming
    #[inline(always)]
    pub fn peek_bits(&mut self, n: u8) -> u32 {
        if self.bits_left < n {
            self.refill();
        }
        (self.buffer >> (64 - n)) as u32
    }

    /// Consume N bits
    #[inline(always)]
    pub fn consume(&mut self, n: u8) {
        self.buffer <<= n;
        self.bits_left -= n;
    }
}
```

Reference: zune-jpeg `bitstream.rs:73-200`.

### 5. Streaming MCU Decode

Main decode loop for baseline streaming:

```rust
fn decode_baseline_streaming(&mut self, output: &mut [u8], stride: usize) -> Result<()> {
    let mut strip_y = vec![0i16; self.strip_width * self.mcu_height];
    let mut strip_cb = vec![0i16; self.chroma_strip_size];
    let mut strip_cr = vec![0i16; self.chroma_strip_size];

    let mut bitstream = Bitstream::new(&self.data[self.scan_start..]);
    let mut dc_pred = [0i32; 4];

    for mcu_row in 0..self.mcu_rows {
        // Decode one MCU row
        for mcu_col in 0..self.mcu_cols {
            // Check restart marker
            if self.restart_interval > 0 && self.mcu_count % self.restart_interval as u32 == 0 {
                bitstream.align_to_byte();
                bitstream.read_restart_marker(self.next_restart)?;
                dc_pred = [0; 4];
            }

            // Decode Y blocks (may be 1, 2, or 4 depending on subsampling)
            for block in 0..self.y_blocks_per_mcu {
                let (coeffs, count) = self.decode_block(&mut bitstream, 0, &mut dc_pred[0])?;
                let idct_fn = choose_idct(count);
                let y_offset = self.block_offset_y(mcu_col, block);
                idct_fn(&coeffs, &mut strip_y[y_offset..], self.strip_width);
            }

            // Decode Cb block
            let (coeffs, count) = self.decode_block(&mut bitstream, 1, &mut dc_pred[1])?;
            let cb_offset = mcu_col * 8;
            choose_idct(count)(&coeffs, &mut strip_cb[cb_offset..], self.chroma_strip_width);

            // Decode Cr block
            let (coeffs, count) = self.decode_block(&mut bitstream, 2, &mut dc_pred[2])?;
            choose_idct(count)(&coeffs, &mut strip_cr[cb_offset..], self.chroma_strip_width);

            self.mcu_count += 1;
        }

        // Upsample chroma if needed
        if self.needs_upsample {
            self.upsample_chroma(&strip_cb, &strip_cr, &mut strip_cb_up, &mut strip_cr_up);
        }

        // Convert to RGB and write to output
        let out_y = mcu_row * self.mcu_height;
        let rows = (self.mcu_height).min(self.height - out_y);

        for row in 0..rows {
            let y_row = &strip_y[row * self.strip_width..][..self.width];
            let cb_row = &strip_cb_final[row * self.strip_width..][..self.width];
            let cr_row = &strip_cr_final[row * self.strip_width..][..self.width];
            let out_row = &mut output[(out_y + row) * stride..][..self.width * 3];

            ycbcr_to_rgb_row(y_row, cb_row, cr_row, out_row);
        }
    }

    Ok(())
}
```

### 6. SIMD Color Conversion

```rust
#[target_feature(enable = "avx2")]
unsafe fn ycbcr_to_rgb_avx2(
    y: &[i16],
    cb: &[i16],
    cr: &[i16],
    output: &mut [u8],
) {
    // Process 16 pixels at a time
    // Y is in range [0, 255], Cb/Cr centered at 128

    // Load coefficients for BT.601
    let y_scale = _mm256_set1_epi16(76);   // 1.164 * 64 ≈ 75
    let cr_r = _mm256_set1_epi16(104);     // 1.596 * 64 ≈ 102
    let cb_g = _mm256_set1_epi16(-25);     // -0.391 * 64 ≈ -25
    let cr_g = _mm256_set1_epi16(-53);     // -0.813 * 64 ≈ -52
    let cb_b = _mm256_set1_epi16(132);     // 2.018 * 64 ≈ 129

    for i in (0..y.len()).step_by(16) {
        let y_vec = _mm256_loadu_si256(y[i..].as_ptr() as *const _);
        let cb_vec = _mm256_loadu_si256(cb[i..].as_ptr() as *const _);
        let cr_vec = _mm256_loadu_si256(cr[i..].as_ptr() as *const _);

        // Center cb/cr around 0
        let cb_centered = _mm256_sub_epi16(cb_vec, _mm256_set1_epi16(128));
        let cr_centered = _mm256_sub_epi16(cr_vec, _mm256_set1_epi16(128));

        // R = Y + 1.402 * Cr
        let r = _mm256_add_epi16(y_vec, _mm256_mulhi_epi16(cr_centered, cr_r));

        // G = Y - 0.344 * Cb - 0.714 * Cr
        let g = _mm256_add_epi16(y_vec, _mm256_mulhi_epi16(cb_centered, cb_g));
        let g = _mm256_add_epi16(g, _mm256_mulhi_epi16(cr_centered, cr_g));

        // B = Y + 1.772 * Cb
        let b = _mm256_add_epi16(y_vec, _mm256_mulhi_epi16(cb_centered, cb_b));

        // Clamp to [0, 255] and pack to u8
        let r_clamped = _mm256_packus_epi16(r, r);
        let g_clamped = _mm256_packus_epi16(g, g);
        let b_clamped = _mm256_packus_epi16(b, b);

        // Interleave RGB and store
        // ... (shuffle and store)
    }
}
```

## Migration Strategy

### Phase 1: Core Optimizations (No API Changes)

1. Add fast AC lookup table to existing HuffmanDecodeTable
2. Implement tiered integer IDCT
3. Optimize bitstream reader
4. Measure performance improvement

### Phase 2: Streaming Architecture

1. Refactor JpegParser to support streaming
2. Implement true streaming for baseline
3. Keep coefficient buffering for progressive
4. Add SIMD color conversion

### Phase 3: API Expansion

1. Add simple one-shot functions
2. Add decode_into variants
3. Improve ScanlineReader API
4. Add info-before-decode

### Phase 4: Advanced Features

1. Add cooperative cancellation (Stop trait)
2. Add crop/scale in decoder
3. Add memory estimation
4. Performance tuning

## Benchmarking Plan

### Test Images

- Kodak corpus (24 images, various dimensions)
- High-res images (4K, 8K)
- Different quality levels (q50, q75, q90, q95)
- Different subsampling (4:4:4, 4:2:0)
- Baseline vs progressive

### Metrics

1. Decode throughput (megapixels/second)
2. Memory usage (peak bytes)
3. Latency (time to first pixel for streaming)

### Comparison

- zune-jpeg (target)
- image-rs jpeg-decoder
- mozjpeg-rs
- libjpeg-turbo (via FFI, reference)

## Estimated Performance Gains

| Optimization | Expected Gain |
|--------------|---------------|
| Fast AC lookup | 10-15% |
| Tiered IDCT | 20-30% |
| DC-only shortcut | 5-10% (image dependent) |
| Efficient bitstream | 10-15% |
| SIMD color conversion | 15-20% |
| Streaming (no coeff buffer) | Memory: 60-80% reduction |

**Target**: Match zune-jpeg performance (within 10%)

## References

- zune-jpeg source: `~/work/zune-image/crates/zune-jpeg/src/`
- libjpeg-turbo: https://github.com/libjpeg-turbo/libjpeg-turbo
- JPEG specification: ITU-T T.81
- codec-design guidelines: `~/work/codec-design/README.md`
