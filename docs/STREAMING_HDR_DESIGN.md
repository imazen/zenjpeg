# Streaming UltraHDR Codec Design

## Philosophy: Separation of Concerns

**The codec handles JPEG. The caller handles HDR.**

| Codec Responsibility | Caller Responsibility |
|---------------------|----------------------|
| JPEG encode/decode | HDR → SDR tonemapping |
| MPF structure (multi-picture) | Gain map computation |
| XMP metadata embedding | Color space conversion |
| ICC profile embedding | Tonemapper lag buffering |
| Streaming row I/O | Working space decisions |

This separation means:
- **Simpler codec** - no color science, no tonemapping algorithms
- **Flexible caller** - use GPU tonemapping, custom algorithms, whatever
- **Testable** - codec tests don't need HDR test images
- **No lag complexity in codec** - caller buffers rows if their tonemapper needs it

## API Requirements (MANDATORY)

See `CLAUDE.md` for full rules. Summary:

| Rule | Requirement |
|------|-------------|
| **Pixel format** | `rgb::RGB<T>` or `rgb::RGBA<T>` - NEVER raw `&[u8]` |
| **Stride** | ALWAYS via `imgref` or explicit `stride_pixels` param |
| **Precision** | 16-32 bit internal - NEVER 8-bit pixel arithmetic |
| **Streaming** | Row-by-row only - NEVER whole-image buffering |
| **Allocation** | Fallible (`try_reserve`) - NEVER panic on OOM |
| **Buffers** | Caller provides output buffers - NEVER allocate when avoidable |

## Streaming UltraHDR Encoder

### What It Does

1. Accepts SDR rows (caller already tonemapped)
2. Accepts gain map rows (caller already computed)
3. Encodes both as streaming JPEGs internally
4. Assembles final UltraHDR with MPF + XMP

### What It Does NOT Do

- Tonemapping (caller's job)
- Gain map computation (caller's job)
- Color space conversion (caller's job)
- Buffering for tonemapper lag (caller's job)

### API

```rust
use rgb::{RGB, RGBA};
use imgref::{ImgRef, ImgRefMut};

/// Configuration for UltraHDR encoding
#[derive(Clone, Debug)]
pub struct UltraHdrEncoderConfig {
    /// JPEG quality for SDR base image (0-100)
    pub sdr_quality: f32,
    /// JPEG quality for gain map (0-100)
    pub gainmap_quality: f32,
    /// Chroma subsampling for SDR
    pub sdr_subsampling: ChromaSubsampling,
    /// Use optimized Huffman tables
    pub optimize_coding: bool,
}

impl Default for UltraHdrEncoderConfig {
    fn default() -> Self {
        Self {
            sdr_quality: 85.0,
            gainmap_quality: 75.0,
            sdr_subsampling: ChromaSubsampling::Cs420,
            optimize_coding: true,
        }
    }
}

/// Streaming UltraHDR encoder
///
/// Manages two internal JPEG streams (SDR base + gain map) and assembles
/// them into a single UltraHDR file at finish.
pub struct StreamingUltraHdrEncoder {
    // Internal state - not public
    sdr_encoder: StreamingJpegEncoder,
    gm_encoder: StreamingJpegEncoder,
    icc_profile: Option<Vec<u8>>,
    sdr_rows_pushed: usize,
    gm_rows_pushed: usize,
}

impl StreamingUltraHdrEncoder {
    /// Create a new streaming UltraHDR encoder.
    ///
    /// # Arguments
    /// - `sdr_width`, `sdr_height`: SDR image dimensions
    /// - `gm_width`, `gm_height`: Gain map dimensions (often 1/4 or 1/8 of SDR)
    /// - `config`: Encoder configuration
    pub fn new(
        sdr_width: u32,
        sdr_height: u32,
        gm_width: u32,
        gm_height: u32,
        config: UltraHdrEncoderConfig,
    ) -> Result<Self>;

    /// Set ICC profile for the SDR image.
    ///
    /// - `None` = sRGB assumed (no profile embedded)
    /// - `Some(bytes)` = embed this ICC profile
    ///
    /// Call before pushing any rows.
    pub fn set_icc_profile(&mut self, icc: Option<&[u8]>);

    /// Push SDR rows into the encoder.
    ///
    /// # Arguments
    /// - `data`: SDR pixel data, gamma-encoded RGB
    /// - `stride`: Stride in PIXELS (not bytes!)
    /// - `width`: Actual pixel width (must match constructor)
    /// - `count`: Number of rows to push
    ///
    /// Caller is responsible for:
    /// - Tonemapping HDR → SDR before calling
    /// - Color space conversion to output gamut
    /// - Applying output OETF (e.g., sRGB gamma)
    pub fn push_sdr_rows(
        &mut self,
        data: &[RGB<u8>],
        stride: usize,
        width: usize,
        count: usize,
    ) -> Result<()>;

    /// Push SDR rows (16-bit version for higher precision input).
    pub fn push_sdr_rows_u16(
        &mut self,
        data: &[RGB<u16>],
        stride: usize,
        width: usize,
        count: usize,
    ) -> Result<()>;

    /// Push gain map rows into the encoder.
    ///
    /// # Arguments
    /// - `data`: Gain map values (grayscale, 0-255)
    /// - `stride`: Stride in PIXELS
    /// - `width`: Actual pixel width (must match gm_width from constructor)
    /// - `count`: Number of rows to push
    ///
    /// Caller is responsible for:
    /// - Computing gain = log2(HDR / SDR) scaled to 0-255
    /// - Handling gain map resolution (typically 1/4 or 1/8 of SDR)
    pub fn push_gainmap_rows(
        &mut self,
        data: &[u8],  // Grayscale, not RGB
        stride: usize,
        width: usize,
        count: usize,
    ) -> Result<()>;

    /// Push multi-channel gain map rows (RGB gain map).
    pub fn push_gainmap_rows_rgb(
        &mut self,
        data: &[RGB<u8>],
        stride: usize,
        width: usize,
        count: usize,
    ) -> Result<()>;

    /// Current SDR row position.
    pub fn sdr_rows_pushed(&self) -> usize;

    /// Current gain map row position.
    pub fn gainmap_rows_pushed(&self) -> usize;

    /// Finish encoding and write to caller's buffer.
    ///
    /// # Arguments
    /// - `output`: Caller-provided buffer (will be cleared and filled)
    /// - `metadata`: Gain map metadata for XMP
    ///
    /// Both SDR and gain map must be fully pushed before calling.
    pub fn finish_into(
        self,
        output: &mut Vec<u8>,
        metadata: &GainMapMetadata,
    ) -> Result<()>;
}
```

### Usage Example

```rust
use zenjpeg::ultrahdr::{StreamingUltraHdrEncoder, UltraHdrEncoderConfig, GainMapMetadata};
use rgb::RGB;

// Caller does all HDR processing BEFORE calling the codec
let mut tonemapper = MyTonemapper::new();  // Caller's tonemapper
let mut gm_computer = MyGainMapComputer::new();  // Caller's gain map

// Create encoder
let mut encoder = StreamingUltraHdrEncoder::new(
    width, height,           // SDR dimensions
    width / 4, height / 4,   // Gain map at 1/4 resolution
    UltraHdrEncoderConfig::default(),
)?;

// Set ICC profile if not sRGB
encoder.set_icc_profile(Some(&display_p3_icc));

// Caller's buffers with stride for SIMD alignment
let sdr_stride = (width as usize + 15) & !15;
let gm_stride = (width as usize / 4 + 15) & !15;
let mut sdr_buf = vec![RGB::default(); sdr_stride];
let mut gm_buf = vec![0u8; gm_stride];

// Stream rows - caller handles tonemapper lag
for row in 0..height {
    // Caller reads HDR, tonemaps, writes to sdr_buf
    let hdr_row = source.read_hdr_row(row);
    tonemapper.tonemap_row(&hdr_row, &mut sdr_buf[..width as usize]);

    // Push SDR row with stride
    encoder.push_sdr_rows(&sdr_buf, sdr_stride, width as usize, 1)?;

    // Gain map at 1/4 resolution
    if row % 4 == 0 {
        gm_computer.compute_row(&hdr_row, &sdr_buf, &mut gm_buf[..width as usize / 4]);
        encoder.push_gainmap_rows(&gm_buf, gm_stride, width as usize / 4, 1)?;
    }
}

// Finish into caller's buffer
let mut output = Vec::new();
output.try_reserve(estimated_size)?;  // Fallible!
encoder.finish_into(&mut output, &gm_computer.metadata())?;
```

## Streaming UltraHDR Decoder

### What It Does

1. Decodes SDR JPEG rows on demand
2. Decodes gain map JPEG rows on demand (if present)
3. Provides metadata and ICC profile access

### What It Does NOT Do

- Apply gain map (caller's job)
- Color space conversion (caller's job)
- HDR reconstruction (caller's job)

### API

```rust
/// Streaming UltraHDR decoder
///
/// Provides streaming access to SDR base image and gain map separately.
/// Caller is responsible for HDR reconstruction.
pub struct StreamingUltraHdrDecoder<'a> {
    // Internal state
}

impl<'a> StreamingUltraHdrDecoder<'a> {
    /// Create decoder from JPEG data.
    pub fn new(data: &'a [u8]) -> Result<Self>;

    /// Check if this is an UltraHDR image (has gain map metadata).
    pub fn is_ultrahdr(&self) -> bool;

    /// SDR image dimensions.
    pub fn sdr_dimensions(&self) -> (u32, u32);

    /// Gain map dimensions (if present).
    pub fn gainmap_dimensions(&self) -> Option<(u32, u32)>;

    /// Get gain map metadata (if present).
    pub fn metadata(&self) -> Option<&GainMapMetadata>;

    /// Get ICC profile (if present).
    pub fn icc_profile(&self) -> Option<&[u8]>;

    /// Current SDR row position.
    pub fn sdr_row(&self) -> usize;

    /// Current gain map row position.
    pub fn gainmap_row(&self) -> usize;

    /// Read SDR rows into caller's buffer.
    ///
    /// # Arguments
    /// - `out`: Caller's output buffer
    /// - `stride`: Stride in PIXELS
    /// - `width`: Expected width (for validation)
    /// - `max_rows`: Maximum rows to read
    ///
    /// Returns actual rows read.
    pub fn read_sdr_rows(
        &mut self,
        out: &mut [RGB<u8>],
        stride: usize,
        width: usize,
        max_rows: usize,
    ) -> Result<usize>;

    /// Read SDR rows as 16-bit (for higher precision).
    pub fn read_sdr_rows_u16(
        &mut self,
        out: &mut [RGB<u16>],
        stride: usize,
        width: usize,
        max_rows: usize,
    ) -> Result<usize>;

    /// Read gain map rows into caller's buffer.
    ///
    /// # Arguments
    /// - `out`: Caller's output buffer (grayscale)
    /// - `stride`: Stride in PIXELS
    /// - `width`: Expected width
    /// - `max_rows`: Maximum rows to read
    ///
    /// Returns actual rows read.
    pub fn read_gainmap_rows(
        &mut self,
        out: &mut [u8],
        stride: usize,
        width: usize,
        max_rows: usize,
    ) -> Result<usize>;

    /// Read RGB gain map rows.
    pub fn read_gainmap_rows_rgb(
        &mut self,
        out: &mut [RGB<u8>],
        stride: usize,
        width: usize,
        max_rows: usize,
    ) -> Result<usize>;

    /// Check if SDR stream is finished.
    pub fn sdr_finished(&self) -> bool;

    /// Check if gain map stream is finished.
    pub fn gainmap_finished(&self) -> bool;
}
```

### Usage Example

```rust
use zenjpeg::ultrahdr::StreamingUltraHdrDecoder;
use rgb::RGB;

let mut decoder = StreamingUltraHdrDecoder::new(&jpeg_data)?;

if !decoder.is_ultrahdr() {
    // Fall back to regular JPEG decode
    return decode_regular_jpeg(&jpeg_data);
}

let (sdr_w, sdr_h) = decoder.sdr_dimensions();
let (gm_w, gm_h) = decoder.gainmap_dimensions().unwrap();
let metadata = decoder.metadata().unwrap();

// Caller's buffers with stride
let sdr_stride = (sdr_w as usize + 15) & !15;
let gm_stride = (gm_w as usize + 15) & !15;
let hdr_stride = sdr_stride;

let mut sdr_buf = vec![RGB::<u8>::default(); sdr_stride * 16];  // 16 rows
let mut gm_buf = vec![0u8; gm_stride * 4];  // 4 rows (1/4 res)
let mut hdr_buf = vec![RGB::<f32>::default(); hdr_stride * 16];

// Caller's HDR reconstruction (using ultrahdr-core or custom)
let mut reconstructor = HdrReconstructor::new(metadata, display_boost);

while !decoder.sdr_finished() {
    // Read SDR rows
    let sdr_rows = decoder.read_sdr_rows(&mut sdr_buf, sdr_stride, sdr_w as usize, 16)?;

    // Read corresponding gain map rows (at lower resolution)
    let gm_rows_needed = (sdr_rows + 3) / 4;  // 1/4 resolution
    let gm_rows = decoder.read_gainmap_rows(&mut gm_buf, gm_stride, gm_w as usize, gm_rows_needed)?;

    // Caller reconstructs HDR (NOT the codec's job)
    reconstructor.apply_gainmap(
        &sdr_buf[..sdr_rows * sdr_stride],
        sdr_stride,
        &gm_buf[..gm_rows * gm_stride],
        gm_stride,
        &mut hdr_buf[..sdr_rows * hdr_stride],
        hdr_stride,
        sdr_w as usize,
        sdr_rows,
    )?;

    // Process HDR rows...
}
```

## GainMapMetadata

The metadata structure (from XMP) that describes how to interpret the gain map:

```rust
/// Gain map metadata from XMP
#[derive(Clone, Debug)]
pub struct GainMapMetadata {
    /// Version of the gain map format
    pub version: String,

    /// Base rendition is HDR (vs SDR)
    pub base_rendition_is_hdr: bool,

    /// Gain map min/max values (log2 scale)
    pub gain_map_min: [f32; 3],  // Per-channel or single value
    pub gain_map_max: [f32; 3],

    /// Gamma for gain map encoding
    pub gamma: f32,

    /// Offset values
    pub offset_sdr: f32,
    pub offset_hdr: f32,

    /// HDR capacity (max boost)
    pub hdr_capacity_min: f32,
    pub hdr_capacity_max: f32,
}
```

## Implementation Plan

### Phase 1: Refactor Existing UltraHDR Module

**Goal**: Remove HDR processing from codec, keep only JPEG + MPF + XMP

1. **Remove from codec** (move to examples/tests if needed for reference):
   - `encode_ultrahdr()` - full workflow with tonemapping
   - `encode_ultrahdr_with_tonemapper()` - adaptive tonemapper
   - `tonemap_hdr_to_sdr()` - tonemapping
   - `create_gainmap_computer()` - gain map computation wrapper
   - `reconstruct_hdr()` - HDR reconstruction
   - `reencode_ultrahdr()` - full roundtrip

2. **Keep in codec**:
   - `encode_with_gainmap()` → refactor to streaming `StreamingUltraHdrEncoder`
   - MPF assembly logic
   - XMP generation/parsing
   - ICC profile embedding

3. **Update decoder**:
   - `UltraHdrReader` → refactor to `StreamingUltraHdrDecoder`
   - Remove `HdrDecoderState` (reconstruction is caller's job)
   - Keep metadata extraction, gain map JPEG extraction

### Phase 2: Implement Streaming Encoder

1. Create `StreamingUltraHdrEncoder` with:
   - Two internal `StreamingJpegEncoder` instances
   - `push_sdr_rows()` with stride
   - `push_gainmap_rows()` with stride
   - `finish_into()` that assembles MPF

2. Add type-safe pixel inputs:
   - `&[RGB<u8>]` for 8-bit
   - `&[RGB<u16>]` for 16-bit
   - Always require stride parameter

3. Fallible allocation:
   - `finish_into(&mut Vec<u8>)` - caller provides buffer
   - Internal buffers use `try_reserve()`

### Phase 3: Implement Streaming Decoder

1. Create `StreamingUltraHdrDecoder` with:
   - Separate SDR and gain map scanline readers
   - `read_sdr_rows()` with stride
   - `read_gainmap_rows()` with stride
   - Metadata and ICC access

2. Remove HDR reconstruction from codec:
   - No `display_boost` parameter
   - No gain map application
   - Just raw SDR + raw gain map output

### Phase 4: Update ultrahdr-core Integration

1. Keep `ultrahdr-core` as optional dependency for users who want:
   - Ready-made tonemapping
   - Ready-made gain map computation
   - Ready-made HDR reconstruction

2. Provide examples showing how to use `ultrahdr-core` with new streaming API

### Phase 5: Testing

1. **Codec tests** (no HDR knowledge needed):
   - SDR + gain map → UltraHDR → SDR + gain map roundtrip
   - Verify MPF structure correct
   - Verify XMP metadata preserved
   - Stride handling correct

2. **Integration tests** (with ultrahdr-core):
   - Full HDR → UltraHDR → HDR roundtrip
   - Quality metrics (SSIMULACRA2, Butteraugli)

### File Changes

| File | Action |
|------|--------|
| `ultrahdr/mod.rs` | Simplify re-exports |
| `ultrahdr/encode.rs` | Remove tonemapping, keep MPF assembly |
| `ultrahdr/decode.rs` | Remove reconstruction, keep metadata |
| `decode/ultrahdr_reader.rs` | Refactor to `StreamingUltraHdrDecoder` |
| `examples/ultrahdr_*.rs` | Update to use new API |
| `tests/ultrahdr_*.rs` | Split codec tests from integration tests |

## Memory Budget

For a 4K image (3840×2160):

| Component | Memory |
|-----------|--------|
| SDR scanline buffer (16 rows, stride-aligned) | ~240 KB |
| Gain map scanline buffer (4 rows at 1/4 res) | ~15 KB |
| Internal JPEG buffers (2 encoders) | ~100 KB |
| **Total codec memory** | **~355 KB** |

HDR processing memory (caller's responsibility):
| Component | Memory |
|-----------|--------|
| HDR row buffer (f32, 16 rows) | ~960 KB |
| Tonemapper state (varies) | varies |
| **Total caller memory** | **~1 MB+** |

The codec stays lean. Caller decides how much to buffer for their tonemapper.
