# API Spec: imgref + rgb + yuv Integration

## Goals

1. **Type-safe**: Pixel format encoded in type - impossible to pass wrong format
2. **Zero-copy**: bytemuck transmutes where possible
3. **Stride support**: imgref enables sub-images and padded buffers
4. **Idiot-proof**: Can't mismatch width/height/format with data
5. **Minimal allocations**: Decode into user buffers, reuse configs
6. **yuv integration**: Sharp YUV for high-quality chroma downsampling

## Dependencies

All are **default** (not optional):

```toml
[dependencies]
rgb = { version = "0.8", features = ["as-bytes"] }
imgref = "1.12"
bytemuck = { version = "1.14", features = ["derive"] }
yuv = "0.1"  # Sharp YUV chroma downsampling
```

Remove unused:
- `arrayref`
- `multiversion`

---

## Type System

### Pixel Trait

```rust
/// Marker trait for pixel types that can be encoded to JPEG.
/// Sealed to prevent external implementations.
pub trait Pixel: bytemuck::Pod + Sized + private::Sealed {
    /// Number of bytes per pixel.
    const BYTES: usize;
    /// Number of color channels.
    const CHANNELS: usize;
    /// Whether this format has an alpha channel (which will be ignored).
    const HAS_ALPHA: bool;
}

// Implementations for rgb crate types
impl Pixel for rgb::RGB<u8> {
    const BYTES: usize = 3;
    const CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
}

impl Pixel for rgb::RGBA<u8> {
    const BYTES: usize = 4;
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;  // Alpha ignored on encode
}

impl Pixel for rgb::Gray<u8> {
    const BYTES: usize = 1;
    const CHANNELS: usize = 1;
    const HAS_ALPHA: bool = false;
}

// 16-bit implementations (higher precision input)
impl Pixel for rgb::RGB<u16> {
    const BYTES: usize = 6;
    const CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
}

impl Pixel for rgb::RGBA<u16> {
    const BYTES: usize = 8;
    const CHANNELS: usize = 4;
    const HAS_ALPHA: bool = true;
}

impl Pixel for rgb::Gray<u16> {
    const BYTES: usize = 2;
    const CHANNELS: usize = 1;
    const HAS_ALPHA: bool = false;
}

// Type aliases for convenience (8-bit)
pub type RGB8 = rgb::RGB<u8>;
pub type RGBA8 = rgb::RGBA<u8>;
pub type Gray8 = rgb::Gray<u8>;

// Type aliases for 16-bit (high precision input)
pub type RGB16 = rgb::RGB<u16>;
pub type RGBA16 = rgb::RGBA<u16>;
pub type Gray16 = rgb::Gray<u16>;
```

### Supported Formats Matrix

| Input Type | Encode | Decode Into | Notes |
|------------|--------|-------------|-------|
| `ImgRef<RGB8>` | ✅ | ✅ | Primary format |
| `ImgRef<RGBA8>` | ✅ | ✅ | Alpha ignored on encode |
| `ImgRef<Gray8>` | ✅ | ✅ | Grayscale JPEG |
| `ImgRef<RGB16>` | ✅ | ✅ | 16-bit input, higher precision |
| `ImgRef<RGBA16>` | ✅ | ✅ | 16-bit, alpha ignored |
| `ImgRef<Gray16>` | ✅ | ✅ | 16-bit grayscale |
| `&[RGB8]` + dims | ✅ | ✅ | Flat buffer, no stride |
| `&[u8]` + dims + format | ✅ | ✅ | Legacy/interop |

**Note**: JPEG is 8-bit output. 16-bit input is useful for:
- Camera RAW processing pipelines (preserve precision until final encode)
- HDR tonemapping workflows
- Scientific imaging where input has >8-bit precision
- yuv crate's 16-bit Sharp YUV for better chroma accuracy

---

## EncoderConfig

Configuration for encoding. Does NOT contain image data or dimensions.

```rust
/// JPEG encoding configuration.
///
/// Create once, reuse for multiple images with same settings.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    // Quality
    quality: Quality,

    // Color & subsampling
    subsampling: Subsampling,      // S444, S422, S420, S440
    chroma_conversion: ChromaConversion, // Intrinsic, Fast, Sharp, Auto
    use_xyb: bool,                 // XYB perceptual color space

    // Encoding mode
    mode: JpegMode,                // Baseline, Progressive
    optimize_huffman: bool,        // Optimized vs fixed tables

    // Advanced
    restart_interval: u16,         // RST markers (0 = disabled)
    smoothing: bool,               // Input smoothing for chroma (Intrinsic path only)
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quality: Quality::from_quality(90.0),
            subsampling: Subsampling::S444,
            use_xyb: false,
            mode: JpegMode::Baseline,
            optimize_huffman: true,
            restart_interval: 0,
            smoothing: true,
            sharp_yuv: true,  // Default ON - better quality
        }
    }
}

impl EncoderConfig {
    pub fn new() -> Self { Self::default() }

    // Builder methods (all return Self for chaining)
    pub fn quality(self, q: Quality) -> Self;
    pub fn subsampling(self, s: Subsampling) -> Self;
    pub fn xyb(self, enable: bool) -> Self;
    pub fn progressive(self, enable: bool) -> Self;
    pub fn optimize_huffman(self, enable: bool) -> Self;
    pub fn restart_interval(self, interval: u16) -> Self;
    pub fn smoothing(self, enable: bool) -> Self;
    pub fn sharp_yuv(self, enable: bool) -> Self;
}
```

---

## Encoding API

### Primary: ImgRef (Type-Safe, Stride-Aware)

```rust
impl EncoderConfig {
    /// Encode from typed image reference.
    ///
    /// Width, height, stride, and pixel format are all encoded in the type.
    /// This is the recommended API - impossible to mismatch parameters.
    ///
    /// # Stride Handling
    /// - If `img.stride() == img.width()`: zero-copy path
    /// - If strided (sub-image): copies to contiguous buffer internally
    ///
    /// # Example
    /// ```
    /// let img: ImgVec<RGB8> = load_image();
    /// let jpeg = EncoderConfig::new()
    ///     .quality(Quality::from_quality(85.0))
    ///     .encode(img.as_ref())?;
    /// ```
    pub fn encode<P: Pixel>(&self, img: ImgRef<'_, P>) -> Result<Vec<u8>>;

    /// Encode directly to a writer (avoids output Vec allocation).
    ///
    /// Useful for writing directly to files or network streams.
    ///
    /// # Example
    /// ```
    /// let file = File::create("output.jpg")?;
    /// config.encode_to(img.as_ref(), BufWriter::new(file))?;
    /// ```
    pub fn encode_to<P: Pixel, W: Write>(
        &self,
        img: ImgRef<'_, P>,
        writer: W
    ) -> Result<()>;
}
```

### Secondary: Flat Pixel Slice (No Stride)

```rust
impl EncoderConfig {
    /// Encode from flat pixel buffer with explicit dimensions.
    ///
    /// Use when you have pixels in a flat Vec/slice without stride.
    /// Buffer must contain exactly `width * height` pixels.
    ///
    /// # Example
    /// ```
    /// let pixels: Vec<RGB8> = generate_pixels(640, 480);
    /// let jpeg = config.encode_slice(&pixels, 640, 480)?;
    /// ```
    pub fn encode_slice<P: Pixel>(
        &self,
        pixels: &[P],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>>;

    /// Encode flat pixel buffer to writer.
    pub fn encode_slice_to<P: Pixel, W: Write>(
        &self,
        pixels: &[P],
        width: u32,
        height: u32,
        writer: W,
    ) -> Result<()>;
}
```

### Tertiary: Raw Bytes (Legacy/Interop)

```rust
impl EncoderConfig {
    /// Encode from raw bytes with explicit format and dimensions.
    ///
    /// This is the escape hatch for C interop, legacy code, or formats
    /// not covered by the Pixel trait (BGR, BGRA, CMYK).
    ///
    /// # Safety Note
    /// You are responsible for ensuring `data.len() == width * height * format.bytes_per_pixel()`.
    ///
    /// # Example
    /// ```
    /// // Interop with C library that gives BGR data
    /// let jpeg = config.encode_bytes(&bgr_data, 640, 480, PixelFormat::Bgr)?;
    /// ```
    pub fn encode_bytes(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Vec<u8>>;

    pub fn encode_bytes_to<W: Write>(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        format: PixelFormat,
        writer: W,
    ) -> Result<()>;
}
```

---

## DecoderConfig

Configuration for decoding. Does NOT contain output format (that's in the decode call).

```rust
/// JPEG decoding configuration.
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    // Upsampling
    fancy_upsampling: bool,        // Smooth chroma upsampling
    block_smoothing: bool,         // Smooth block boundaries

    // Color management
    apply_icc: bool,               // Apply embedded ICC profile

    // Safety limits
    max_pixels: u64,               // Maximum total pixels (DoS protection)
    max_memory: usize,             // Maximum memory usage
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            fancy_upsampling: true,
            block_smoothing: true,
            apply_icc: true,
            max_pixels: 256 * 1024 * 1024,  // 256 megapixels
            max_memory: 1024 * 1024 * 1024, // 1 GB
        }
    }
}

impl DecoderConfig {
    pub fn new() -> Self { Self::default() }

    // Builder methods
    pub fn fancy_upsampling(self, enable: bool) -> Self;
    pub fn block_smoothing(self, enable: bool) -> Self;
    pub fn apply_icc(self, enable: bool) -> Self;
    pub fn max_pixels(self, limit: u64) -> Self;
    pub fn max_memory(self, limit: usize) -> Self;
}
```

---

## Decoding API

### Primary: Decode to Owned ImgVec

```rust
impl DecoderConfig {
    /// Decode JPEG to typed image.
    ///
    /// Returns an owned `ImgVec<P>` with the decoded pixels.
    /// The output format is determined by the type parameter.
    ///
    /// # Format Conversion
    /// - RGB JPEG → `ImgVec<RGB8>`: direct
    /// - RGB JPEG → `ImgVec<RGBA8>`: adds alpha=255
    /// - RGB JPEG → `ImgVec<Gray8>`: converts to grayscale
    /// - Grayscale JPEG → `ImgVec<RGB8>`: expands to RGB
    ///
    /// # Example
    /// ```
    /// let img: ImgVec<RGB8> = DecoderConfig::new().decode(&jpeg_data)?;
    /// println!("{}x{}", img.width(), img.height());
    /// ```
    pub fn decode<P: Pixel>(&self, data: &[u8]) -> Result<ImgVec<P>>;

    // Convenience aliases
    pub fn decode_rgb(&self, data: &[u8]) -> Result<ImgVec<RGB8>> {
        self.decode(data)
    }
    pub fn decode_rgba(&self, data: &[u8]) -> Result<ImgVec<RGBA8>> {
        self.decode(data)
    }
    pub fn decode_gray(&self, data: &[u8]) -> Result<ImgVec<Gray8>> {
        self.decode(data)
    }
}
```

### Secondary: Decode Into User Buffer (Zero-Alloc)

```rust
impl DecoderConfig {
    /// Decode into a pre-allocated pixel slice.
    ///
    /// Returns (width, height) of decoded image.
    /// Buffer must be large enough: at least `width * height` pixels.
    ///
    /// # Errors
    /// - `Error::InvalidBufferSize` if buffer is too small
    ///
    /// # Example
    /// ```
    /// // Pre-allocate for max expected size
    /// let mut buffer = vec![RGB8::default(); 4096 * 4096];
    /// let (w, h) = config.decode_into(&jpeg_data, &mut buffer)?;
    /// let pixels = &buffer[..w * h];
    /// ```
    pub fn decode_into<P: Pixel>(
        &self,
        data: &[u8],
        output: &mut [P],
    ) -> Result<(u32, u32)>;

    /// Decode into a strided buffer (sub-region of larger image).
    ///
    /// The output `ImgRefMut` specifies where to write and its stride.
    /// Useful for compositing decoded images into a larger canvas.
    ///
    /// # Errors
    /// - `Error::InvalidDimensions` if output is smaller than decoded image
    ///
    /// # Example
    /// ```
    /// let mut canvas = ImgVec::new(vec![RGB8::default(); 1000*1000], 1000, 1000);
    /// // Decode into top-left corner
    /// let region = canvas.sub_image_mut(0, 0, 640, 480);
    /// config.decode_into_strided(&jpeg_data, region)?;
    /// ```
    pub fn decode_into_strided<P: Pixel>(
        &self,
        data: &[u8],
        output: ImgRefMut<'_, P>,
    ) -> Result<()>;
}
```

### Tertiary: Header-Only / Metadata

```rust
impl DecoderConfig {
    /// Read JPEG header without decoding pixels.
    ///
    /// Returns image metadata (dimensions, color space, etc).
    /// Useful for pre-allocating buffers or validating before decode.
    ///
    /// # Example
    /// ```
    /// let info = config.read_header(&jpeg_data)?;
    /// println!("Image is {}x{}", info.width, info.height);
    /// let mut buffer = vec![RGB8::default(); info.width * info.height];
    /// config.decode_into(&jpeg_data, &mut buffer)?;
    /// ```
    pub fn read_header(&self, data: &[u8]) -> Result<ImageInfo>;
}

/// Metadata from JPEG header.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub color_space: ColorSpace,  // Grayscale, YCbCr, RGB, CMYK
    pub num_components: u8,
    pub bits_per_component: u8,
    pub is_progressive: bool,
    pub has_icc_profile: bool,
}
```

---

## DecodedImage Conversions (Zero-Copy)

For users of the legacy `decode()` API that returns `DecodedImage`:

```rust
impl DecodedImage {
    /// Convert to typed ImgVec, consuming self (zero-copy).
    ///
    /// # Errors
    /// Returns error if format doesn't match (e.g., calling on grayscale data).
    pub fn into_imgvec_rgb(self) -> Result<ImgVec<RGB8>> {
        if self.format != PixelFormat::Rgb {
            return Err(Error::InvalidColorFormat {
                reason: "expected RGB, use into_imgvec::<RGB8>() for conversion"
            });
        }
        // Zero-copy: Vec<u8> → Vec<RGB8> via bytemuck
        let pixels: Vec<RGB8> = bytemuck::cast_vec(self.data);
        Ok(ImgVec::new(pixels, self.width as usize, self.height as usize))
    }

    /// Borrow as typed ImgRef (zero-copy, no conversion).
    pub fn as_imgref_rgb(&self) -> Result<ImgRef<'_, RGB8>> {
        if self.format != PixelFormat::Rgb {
            return Err(Error::InvalidColorFormat {
                reason: "expected RGB format"
            });
        }
        let pixels: &[RGB8] = bytemuck::cast_slice(&self.data);
        Ok(ImgRef::new(pixels, self.width as usize, self.height as usize))
    }

    /// Generic conversion (may allocate if format conversion needed).
    pub fn into_imgvec<P: Pixel>(self) -> Result<ImgVec<P>>;
    pub fn as_imgref<P: Pixel>(&self) -> Result<ImgRef<'_, P>>;
}
```

---

## Convenience Wrappers

For users who want the absolute simplest API:

```rust
/// Convenience wrapper around EncoderConfig.
///
/// For simple one-shot encoding. Use EncoderConfig directly
/// when encoding multiple images with the same settings.
pub struct Encoder(EncoderConfig);

impl Encoder {
    pub fn new() -> Self { Self(EncoderConfig::new()) }

    // Delegate all builder methods
    pub fn quality(mut self, q: Quality) -> Self { self.0 = self.0.quality(q); self }
    pub fn subsampling(mut self, s: Subsampling) -> Self { self.0 = self.0.subsampling(s); self }
    // ... etc

    // Delegate encode methods
    pub fn encode<P: Pixel>(&self, img: ImgRef<'_, P>) -> Result<Vec<u8>> {
        self.0.encode(img)
    }
    pub fn encode_slice<P: Pixel>(&self, pixels: &[P], w: u32, h: u32) -> Result<Vec<u8>> {
        self.0.encode_slice(pixels, w, h)
    }
}

/// Convenience wrapper around DecoderConfig.
pub struct Decoder(DecoderConfig);

impl Decoder {
    pub fn new() -> Self { Self(DecoderConfig::new()) }

    pub fn decode<P: Pixel>(&self, data: &[u8]) -> Result<ImgVec<P>> {
        self.0.decode(data)
    }
    // ... etc
}
```

---

## Stride and Borrowing Semantics

### ImgRef Stride Handling

```rust
// imgref provides these key properties:
trait ImgRefLike<P> {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn stride(&self) -> usize;  // May be > width (padding/sub-image)
    fn buf(&self) -> &[P];      // Underlying buffer
}

// Contiguous check
fn is_contiguous<P>(img: ImgRef<'_, P>) -> bool {
    img.stride() == img.width()
}
```

### Encoding with Stride

```rust
fn encode_impl<P: Pixel>(&self, img: ImgRef<'_, P>) -> Result<Vec<u8>> {
    if is_contiguous(&img) {
        // Fast path: zero-copy, use buffer directly
        let bytes: &[u8] = bytemuck::cast_slice(img.buf());
        self.encode_contiguous(bytes, img.width(), img.height())
    } else {
        // Strided: must copy to contiguous buffer
        // This is unavoidable - JPEG encoding needs contiguous rows
        let contiguous: Vec<P> = img.pixels().copied().collect();
        let bytes: &[u8] = bytemuck::cast_slice(&contiguous);
        self.encode_contiguous(bytes, img.width(), img.height())
    }
}
```

## Color Conversion Paths

Three color conversion paths, selectable via `EncoderConfig`:

### Path 1: Intrinsic f32 (Default for 4:4:4)

Our internal f32 YCbCr conversion. High precision, no subsampling artifacts.

```rust
// Current: convert_to_ycbcr_f32()
// Uses BT.601 coefficients, f32 precision throughout
```

**TODO**: Upgrade with:
- [ ] Proper edge handling (current: clamp to edge)
- [ ] Gamma-aware conversion (linear light blending)
- [ ] Optional BT.709/BT.2020 matrix selection

### Path 2: yuv Crate Standard (Fast, Non-Sharp)

Uses yuv crate's optimized SIMD conversion without Sharp YUV.
Good for speed when chroma accuracy is less critical.

```rust
// yuv::rgb_to_yuv420(), rgb_to_yuv422()
// Fast SIMD, simple box filter downsampling
```

### Path 3: yuv Crate Sharp (Best Quality)

Uses yuv crate's Sharp YUV with gamma correction.
Best chroma quality for edges and fine detail.

```rust
// yuv::rgb_to_sharp_yuv420(), rgb_to_sharp_yuv422()
// Gamma-aware bi-linear interpolation
// Requires: professional_mode feature (already enabled)
```

### Configuration

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaConversion {
    /// Our f32 conversion (best for 4:4:4, no downsampling)
    Intrinsic,
    /// yuv crate fast path (SIMD, simple box filter)
    Fast,
    /// yuv crate Sharp YUV (gamma-aware, best edges)
    Sharp,
    /// Auto-select based on subsampling
    /// - 4:4:4 → Intrinsic (no downsampling needed)
    /// - 4:2:0/4:2:2 → Sharp (best quality)
    Auto,
}

impl Default for ChromaConversion {
    fn default() -> Self { Self::Auto }
}
```

### Which Path When?

| Subsampling | Auto Selection | Reason |
|-------------|----------------|--------|
| 4:4:4 | Intrinsic | No chroma downsampling needed |
| 4:2:2 | Sharp | Horizontal edges benefit from gamma-aware |
| 4:2:0 | Sharp | Both axes benefit from gamma-aware |
| 4:4:0 | Sharp | Vertical edges benefit from gamma-aware |

### Dependencies

```toml
[dependencies]
yuv = { version = "0.8", features = ["professional_mode"] }
```

The `professional_mode` feature enables higher precision internal calculations.

---

### yuv Crate Compatibility

The yuv crate accepts stride parameters directly and supports multiple bit depths:

**Bit Depths**:
- 8-bit: `rgb_to_sharp_yuv420`, `rgb_to_yuv420`, etc.
- 10-bit: `rgb10_to_i010`, etc.
- 12-bit: `rgb12_to_i012`, etc.
- 16-bit: `rgb16_to_yuv420`, `rgb16_to_sharp_yuv420`, etc.

```rust
// yuv crate signature (8-bit):
fn rgb_to_sharp_yuv420(
    dst: &mut YuvPlanarImageMut<u8>,
    src: &[u8],
    src_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma: SharpYuvGammaTransfer,
) -> Result<(), YuvError>;

// yuv crate signature (16-bit):
fn rgb16_to_sharp_yuv420(
    dst: &mut YuvPlanarImageMut<u16>,
    src: &[u16],
    src_stride: u32,  // In pixels, not bytes
    bit_depth: usize, // 10, 12, or 16
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma: SharpYuvGammaTransfer,
) -> Result<(), YuvError>;
```

Integration with imgref (8-bit):

```rust
fn sharp_yuv_from_imgref(img: ImgRef<'_, RGB8>) -> Result<YuvPlanarImageMut<u8>> {
    let bytes: &[u8] = bytemuck::cast_slice(img.buf());
    let stride = (img.stride() * 3) as u32;  // RGB8 = 3 bytes per pixel

    let mut yuv = YuvPlanarImageMut::alloc(
        img.width() as u32,
        img.height() as u32,
        YuvChromaSubsampling::Yuv420,
    );

    rgb_to_sharp_yuv420(
        &mut yuv,
        bytes,
        stride,  // Stride passed directly - no copy needed!
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        SharpYuvGammaTransfer::Srgb,
    )?;

    Ok(yuv)
}
```

Integration with imgref (16-bit - higher precision chroma):

```rust
fn sharp_yuv_from_imgref_16(img: ImgRef<'_, RGB16>) -> Result<YuvPlanarImageMut<u16>> {
    let samples: &[u16] = bytemuck::cast_slice(img.buf());
    let stride = (img.stride() * 3) as u32;  // RGB16 = 3 u16 per pixel

    let mut yuv = YuvPlanarImageMut::alloc(
        img.width() as u32,
        img.height() as u32,
        YuvChromaSubsampling::Yuv420,
    );

    rgb16_to_sharp_yuv420(
        &mut yuv,
        samples,
        stride,
        16,  // Full 16-bit depth
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
        SharpYuvGammaTransfer::Srgb,
    )?;

    Ok(yuv)
}
```

**Why 16-bit Sharp YUV matters**: Chroma downsampling with gamma correction benefits from higher precision. Using 16-bit internally even for 8-bit input can reduce banding artifacts.

---

## Error Handling

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Buffer size doesn't match expected dimensions.
    InvalidBufferSize {
        expected: usize,
        actual: usize,
    },

    /// Invalid image dimensions.
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },

    /// Pixel format mismatch.
    InvalidColorFormat {
        reason: &'static str,
    },

    /// Output buffer too small for decoded image.
    OutputBufferTooSmall {
        required_pixels: usize,
        provided_pixels: usize,
    },

    // ... other variants
}
```

---

## Usage Examples

### Basic Encoding

```rust
use zenjpeg::{EncoderConfig, Quality, RGB8};
use imgref::ImgVec;

// Load/create image
let pixels: Vec<RGB8> = load_pixels();
let img = ImgVec::new(pixels, 640, 480);

// Encode with defaults (Sharp YUV, optimize Huffman, Q90)
let jpeg = EncoderConfig::new().encode(img.as_ref())?;

// Or with custom settings
let jpeg = EncoderConfig::new()
    .quality(Quality::from_quality(85.0))
    .subsampling(Subsampling::S420)
    .sharp_yuv(true)
    .encode(img.as_ref())?;
```

### Encode Sub-Region (Zero-Copy Input)

```rust
let full_image: ImgVec<RGB8> = load_large_image();

// Extract sub-region (no copy - just adjusts stride/offset)
let region = full_image.sub_image(100, 100, 200, 200);

// Encode just that region
// Note: internally copies to contiguous buffer for JPEG encoding
let jpeg = EncoderConfig::new().encode(region)?;
```

### Decode to Owned Image

```rust
use zenjpeg::{DecoderConfig, RGB8};

let img: ImgVec<RGB8> = DecoderConfig::new().decode(&jpeg_data)?;
println!("Decoded {}x{}", img.width(), img.height());

// Access pixels
for pixel in img.pixels() {
    process(pixel.r, pixel.g, pixel.b);
}
```

### Decode Into Pre-Allocated Buffer

```rust
// Read header first to get dimensions
let info = DecoderConfig::new().read_header(&jpeg_data)?;

// Pre-allocate exact size
let mut pixels = vec![RGB8::default(); (info.width * info.height) as usize];

// Decode (no allocation for pixel data)
DecoderConfig::new().decode_into(&jpeg_data, &mut pixels)?;
```

### Decode Into Canvas Region

```rust
// Large canvas
let mut canvas = ImgVec::new(
    vec![RGB8::new(255, 255, 255); 2000 * 2000], // White background
    2000, 2000
);

// Decode multiple images into different regions
for (jpeg, x, y) in thumbnails {
    let info = config.read_header(&jpeg)?;
    let region = canvas.sub_image_mut(x, y, info.width as usize, info.height as usize);
    config.decode_into_strided(&jpeg, region)?;
}
```

### Batch Encoding with Reused Config

```rust
// Create config once
let config = EncoderConfig::new()
    .quality(Quality::from_quality(90.0))
    .optimize_huffman(true)
    .sharp_yuv(true);

// Encode many images with same settings
for img in images {
    let jpeg = config.encode(img.as_ref())?;
    save(&jpeg);
}
```

---

## Implementation Order

### Phase 1: Foundation
1. Add `rgb = { features = ["as-bytes"] }` to enable Pod/Zeroable
2. Remove unused deps: `arrayref`, `multiversion`
3. Make `yuv` non-optional (always available)
4. Implement `Pixel` trait for RGB8, RGBA8, Gray8

### Phase 2: Encoder Refactor
5. Extract `EncoderConfig` from `Encoder` (remove width/height/format)
6. Add `encode<P: Pixel>(ImgRef<P>)` to EncoderConfig
7. Add `encode_slice<P: Pixel>(&[P], w, h)`
8. Add `encode_bytes(&[u8], w, h, format)` for legacy
9. Add `encode_to` variants for Write support
10. Update Sharp YUV to use imgref stride directly

### Phase 3: Decoder Refactor
11. Extract `DecoderConfig` from `Decoder`
12. Add `decode<P: Pixel>() -> ImgVec<P>`
13. Add `decode_into<P: Pixel>(&mut [P])` for zero-alloc
14. Add `decode_into_strided(ImgRefMut<P>)` for compositing
15. Add `read_header()` for metadata-only access

### Phase 4: Conversions & Polish
16. Add `DecodedImage::into_imgvec_*()` zero-copy conversions
17. Add `DecodedImage::as_imgref_*()` zero-copy borrows
18. Update all examples to use new APIs
19. Deprecate old Encoder/Decoder width/height/format methods
20. Benchmark zero-copy paths

---

## Open Questions (Resolved)

1. **RGBA**: Accept on encode, alpha silently ignored. Document clearly.
2. **BGR/BGRA**: Only via `encode_bytes()` with PixelFormat. No typed API.
3. **Grayscale JPEG to RGB8**: Expand (Y,Y,Y). RGB to Gray: convert.
4. **Sharp YUV**: Always available, default ON.
5. **Stride**: yuv crate accepts stride directly - no copy needed for Sharp YUV path.
