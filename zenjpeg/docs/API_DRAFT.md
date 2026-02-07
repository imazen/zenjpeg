# Encoder API Specification

## Design Principles

1. **Config is dimension-independent** - reusable across images
2. **Layout is explicit** - enum or type proves you know what you're providing
3. **Stride at push time** - not baked into encoder construction
4. **Cancellation everywhere** - `impl Stop` on every push
5. **Non-generic where possible** - minimize monomorphization

---

## 1. Dependencies

```rust
// Re-export Stop trait from enough crate
pub use enough::Stop;
// Users use enough::Never for no-op cancellation
```

---

## 2. Quality & Quantization

```rust
/// Quality/compression setting.
///
/// All variants map to internal quality through empirical lookup tables.
/// Results vary by image - these are rough approximations, not guarantees.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Quality {
    /// Approximate jpegli quality scale (this is a fork, not exact jpegli).
    /// Range: 0.0–100.0, where ~90 is visually lossless for most images.
    ApproxJpegli(f32),

    /// Approximate mozjpeg quality behavior.
    /// Range: 0–100. Maps to quality producing similar file sizes.
    ApproxMozjpeg(u8),

    /// Approximate SSIMULACRA2 score target.
    /// Range: 0–100 (higher = better). 90+ is roughly visually lossless.
    ApproxSsim2(f32),

    /// Approximate Butteraugli distance target.
    /// Range: 0.0+ (lower = better). <1.0 excellent, <3.0 good.
    ApproxButteraugli(f32),
}

impl Default for Quality {
    fn default() -> Self { Quality::ApproxJpegli(90.0) }
}

impl From<f32> for Quality {
    fn from(q: f32) -> Self { Quality::ApproxJpegli(q) }
}

impl From<u8> for Quality {
    fn from(q: u8) -> Self { Quality::ApproxJpegli(q as f32) }
}

/// Quantization table configuration.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum QuantTableConfig {
    /// Jpegli's perceptual tables, scaled by Quality. (default)
    #[default]
    Perceptual,

    /// Custom base matrices, scaled by Quality.
    /// Provide f32 matrices (typically 1.0–255.0 range).
    CustomBase {
        luma: [f32; 64],
        chroma: [f32; 64],
    },

    /// Exact quantization tables. **Quality is ignored.**
    Exact {
        luma: [u16; 64],
        chroma: [u16; 64],
    },
}
```

---

## 3. Color Mode & Subsampling

```rust
/// Output color space with bundled subsampling options.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum ColorMode {
    /// Standard YCbCr with configurable chroma subsampling.
    #[default]
    YCbCr {
        subsampling: ChromaSubsampling,
    },

    /// XYB perceptual color space (jpegli-specific).
    /// Computed internally from linear RGB input.
    Xyb {
        subsampling: XybSubsampling,
    },

    /// Single-channel grayscale.
    Grayscale,
}

/// YCbCr chroma subsampling (spatial resolution).
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum ChromaSubsampling {
    /// 4:4:4 - Full chroma resolution
    Full,
    /// 4:2:2 - Half horizontal resolution
    HalfHorizontal,
    /// 4:2:0 - Quarter resolution (half each direction)
    #[default]
    Quarter,
    /// 4:4:0 - Half vertical resolution
    HalfVertical,
}

/// XYB component subsampling.
///
/// Unlike YCbCr where only luma is full, XYB keeps X and Y full
/// even in subsampled mode - only B is reduced.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum XybSubsampling {
    /// X, Y, B all at full resolution (1×1, 1×1, 1×1)
    Full,
    /// X, Y full, B at quarter resolution (1×1, 1×1, 2×2)
    #[default]
    BQuarter,
}

/// Chroma downsampling algorithm for RGB→YCbCr conversion.
///
/// **Only applies to RGB/RGBX input.** Ignored for grayscale, YCbCr, and planar input.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum DownsamplingMethod {
    /// Simple box filter averaging (fast, matches C++ jpegli default)
    #[default]
    Box,
    /// Gamma-aware averaging (better color accuracy at edges)
    GammaAware,
    /// Iterative optimization (SharpYUV-style, best quality, ~3× slower)
    GammaAwareIterative,
}
```

---

## 4. Encoder Config

```rust
/// JPEG encoder configuration. Dimension-independent, reusable across images.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    quality: Quality,
    quant_tables: QuantTableConfig,
    progressive: bool,
    optimize_huffman: bool,
    color_mode: ColorMode,
    downsampling_method: DownsamplingMethod,
    restart_interval: u16,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            quality: Quality::default(),
            quant_tables: QuantTableConfig::default(),
            progressive: true,  // 3-7% smaller files
            optimize_huffman: true,
            color_mode: ColorMode::default(),
            downsampling_method: DownsamplingMethod::default(),
            restart_interval: 0,
        }
    }
}

impl EncoderConfig {
    pub fn new() -> Self { Self::default() }

    // === Quality & Quantization ===

    pub fn quality(mut self, q: impl Into<Quality>) -> Self {
        self.quality = q.into();
        self
    }

    pub fn quant_tables(mut self, config: QuantTableConfig) -> Self {
        self.quant_tables = config;
        self
    }

    // === Encoding Mode ===

    pub fn progressive(mut self, enable: bool) -> Self {
        self.progressive = enable;
        if enable {
            self.optimize_huffman = true; // required for progressive
        }
        self
    }

    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.optimize_huffman = enable;
        self
    }

    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.restart_interval = interval;
        self
    }

    // === Color Mode ===

    pub fn color_mode(mut self, mode: ColorMode) -> Self {
        self.color_mode = mode;
        self
    }

    pub fn downsampling_method(mut self, method: DownsamplingMethod) -> Self {
        self.downsampling_method = method;
        self
    }

    // === Convenience Shortcuts ===

    /// YCbCr with specified chroma subsampling.
    pub fn ycbcr(self, subsampling: ChromaSubsampling) -> Self {
        self.color_mode(ColorMode::YCbCr { subsampling })
    }

    /// XYB with B-quarter subsampling (default, perceptually optimized).
    pub fn xyb(self) -> Self {
        self.color_mode(ColorMode::Xyb { subsampling: XybSubsampling::BQuarter })
    }

    /// XYB with full resolution (no subsampling).
    pub fn xyb_full(self) -> Self {
        self.color_mode(ColorMode::Xyb { subsampling: XybSubsampling::Full })
    }

    /// Grayscale output.
    pub fn grayscale(self) -> Self {
        self.color_mode(ColorMode::Grayscale)
    }

    /// Enable SharpYUV (GammaAwareIterative) downsampling.
    pub fn sharp_yuv(self, enable: bool) -> Self {
        self.downsampling_method(if enable {
            DownsamplingMethod::GammaAwareIterative
        } else {
            DownsamplingMethod::Box
        })
    }

    // === Validation ===

    /// Validate config, error on invalid combinations.
    pub fn validate(&self) -> Result<()> {
        if self.progressive && !self.optimize_huffman {
            return Err(Error::InvalidConfig("progressive requires optimize_huffman"));
        }
        Ok(())
    }

    // === Encoder Creation ===

    /// Create encoder from raw bytes with explicit pixel layout.
    pub fn encode_from_bytes(
        &self,
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> Result<BytesEncoder>;

    /// Create encoder from rgb crate pixel type.
    /// Layout inferred from P. For RGBA/BGRA, 4th channel is ignored.
    pub fn encode_from_rgb<P: Pixel>(
        &self,
        width: u32,
        height: u32,
    ) -> Result<RgbEncoder<P>>;

    /// Create encoder from planar YCbCr (separate Y, Cb, Cr planes).
    /// Skips RGB→YCbCr conversion. Only valid with YCbCr color mode.
    pub fn encode_from_ycbcr_planar(
        &self,
        width: u32,
        height: u32,
    ) -> Result<YCbCrPlanarEncoder>;

    // === Resource Estimation ===

    /// Estimate peak memory usage before encoding.
    pub fn estimate_memory(&self, width: u32, height: u32) -> usize;
}
```

---

## 5. Pixel Layout (for raw bytes)

```rust
/// Pixel data layout for raw byte input.
///
/// Describes channel order, bit depth, and color space interpretation.
/// Use with `encode_from_bytes()` when working with raw buffers.
///
/// For rgb crate types, use `encode_from_rgb()` which infers layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PixelLayout {
    // === 8-bit sRGB (gamma-encoded) ===

    /// RGB, 3 bytes/pixel, sRGB gamma
    Rgb8Srgb,
    /// BGR, 3 bytes/pixel, sRGB gamma (Windows/GDI order)
    Bgr8Srgb,
    /// RGBA, 4 bytes/pixel, sRGB gamma (alpha ignored)
    Rgba8Srgb,
    /// RGBX, 4 bytes/pixel, sRGB gamma (4th byte ignored)
    Rgbx8Srgb,
    /// BGRA, 4 bytes/pixel, sRGB gamma (alpha ignored)
    Bgra8Srgb,
    /// BGRX, 4 bytes/pixel, sRGB gamma (4th byte ignored)
    Bgrx8Srgb,
    /// Grayscale, 1 byte/pixel, sRGB gamma
    Gray8Srgb,

    // === 16-bit linear ===

    /// RGB, 6 bytes/pixel, linear light (0–65535)
    Rgb16Linear,
    /// RGBA, 8 bytes/pixel, linear light (alpha ignored)
    Rgba16Linear,
    /// RGBX, 8 bytes/pixel, linear light (4th channel ignored)
    Rgbx16Linear,
    /// Grayscale, 2 bytes/pixel, linear light
    Gray16Linear,

    // === 32-bit float linear ===

    /// RGB, 12 bytes/pixel, linear light (0.0–1.0)
    RgbF32Linear,
    /// RGBA, 16 bytes/pixel, linear light (alpha ignored)
    RgbaF32Linear,
    /// RGBX, 16 bytes/pixel, linear light (4th channel ignored)
    RgbxF32Linear,
    /// Grayscale, 4 bytes/pixel, linear light
    GrayF32Linear,

    // === Pre-converted YCbCr (skip RGB→YCbCr conversion) ===

    /// YCbCr interleaved, 3 bytes/pixel, u8
    YCbCr8,
    /// YCbCr interleaved, 12 bytes/pixel, f32
    YCbCrF32,
}

impl PixelLayout {
    /// Bytes per pixel for this layout.
    pub const fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Gray8Srgb => 1,
            Self::Gray16Linear => 2,
            Self::Rgb8Srgb | Self::Bgr8Srgb | Self::YCbCr8 => 3,
            Self::Rgba8Srgb | Self::Rgbx8Srgb | Self::Bgra8Srgb
            | Self::Bgrx8Srgb | Self::GrayF32Linear => 4,
            Self::Rgb16Linear => 6,
            Self::Rgba16Linear | Self::Rgbx16Linear => 8,
            Self::RgbF32Linear | Self::YCbCrF32 => 12,
            Self::RgbaF32Linear | Self::RgbxF32Linear => 16,
        }
    }

    /// Number of channels (including ignored channels).
    pub const fn channels(&self) -> usize {
        match self {
            Self::Gray8Srgb | Self::Gray16Linear | Self::GrayF32Linear => 1,
            Self::Rgb8Srgb | Self::Bgr8Srgb | Self::Rgb16Linear
            | Self::RgbF32Linear | Self::YCbCr8 | Self::YCbCrF32 => 3,
            Self::Rgba8Srgb | Self::Rgbx8Srgb | Self::Bgra8Srgb
            | Self::Bgrx8Srgb | Self::Rgba16Linear | Self::Rgbx16Linear
            | Self::RgbaF32Linear | Self::RgbxF32Linear => 4,
        }
    }

    /// Whether this is a grayscale format.
    pub const fn is_grayscale(&self) -> bool {
        matches!(self, Self::Gray8Srgb | Self::Gray16Linear | Self::GrayF32Linear)
    }

    /// Whether this is pre-converted YCbCr.
    pub const fn is_ycbcr(&self) -> bool {
        matches!(self, Self::YCbCr8 | Self::YCbCrF32)
    }
}
```

---

## 6. Bytes Encoder

```rust
/// Encoder for raw byte input with explicit pixel layout.
pub struct BytesEncoder {
    config: EncoderConfig,
    layout: PixelLayout,
    width: u32,
    height: u32,
    rows_pushed: u32,
    // internal encoding state...
}

impl BytesEncoder {
    /// Push rows with explicit stride.
    ///
    /// - `data`: Raw pixel bytes
    /// - `rows`: Number of scanlines to push
    /// - `stride_bytes`: Bytes per row in buffer (≥ width × bytes_per_pixel)
    /// - `stop`: Cancellation token (use `enough::Never` if not needed)
    pub fn push(
        &mut self,
        data: &[u8],
        rows: usize,
        stride_bytes: usize,
        stop: impl Stop,
    ) -> Result<()>;

    /// Push contiguous (packed) data.
    ///
    /// Stride is assumed to be `width × bytes_per_pixel`.
    /// Rows inferred from `data.len() / (width × bytes_per_pixel)`.
    pub fn push_packed(&mut self, data: &[u8], stop: impl Stop) -> Result<()>;

    // === Status ===

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }
    pub fn rows_pushed(&self) -> u32 { self.rows_pushed }
    pub fn rows_remaining(&self) -> u32 { self.height - self.rows_pushed }
    pub fn layout(&self) -> PixelLayout { self.layout }

    // === Finish ===

    /// Finish encoding, return JPEG bytes.
    pub fn finish(self) -> Result<Vec<u8>>;

    /// Finish encoding to Write destination.
    pub fn finish_to<W: Write>(self, output: W) -> Result<W>;
}
```

---

## 7. RGB Encoder (rgb crate types)

```rust
use rgb::{RGB, RGBA, BGR, BGRA, Gray};

/// Marker trait for supported rgb crate pixel types.
///
/// Implemented for: RGB<u8>, RGBA<u8>, BGR<u8>, BGRA<u8>, Gray<u8>,
/// RGB<u16>, RGBA<u16>, Gray<u16>, RGB<f32>, RGBA<f32>, Gray<f32>
pub trait Pixel: Copy + 'static {
    /// Equivalent PixelLayout for this type.
    const LAYOUT: PixelLayout;
}

impl Pixel for RGB<u8>  { const LAYOUT: PixelLayout = PixelLayout::Rgb8Srgb; }
impl Pixel for RGBA<u8> { const LAYOUT: PixelLayout = PixelLayout::Rgba8Srgb; }
impl Pixel for BGR<u8>  { const LAYOUT: PixelLayout = PixelLayout::Bgr8Srgb; }
impl Pixel for BGRA<u8> { const LAYOUT: PixelLayout = PixelLayout::Bgra8Srgb; }
impl Pixel for Gray<u8> { const LAYOUT: PixelLayout = PixelLayout::Gray8Srgb; }

impl Pixel for RGB<u16>  { const LAYOUT: PixelLayout = PixelLayout::Rgb16Linear; }
impl Pixel for RGBA<u16> { const LAYOUT: PixelLayout = PixelLayout::Rgba16Linear; }
impl Pixel for Gray<u16> { const LAYOUT: PixelLayout = PixelLayout::Gray16Linear; }

impl Pixel for RGB<f32>  { const LAYOUT: PixelLayout = PixelLayout::RgbF32Linear; }
impl Pixel for RGBA<f32> { const LAYOUT: PixelLayout = PixelLayout::RgbaF32Linear; }
impl Pixel for Gray<f32> { const LAYOUT: PixelLayout = PixelLayout::GrayF32Linear; }

/// Encoder for rgb crate pixel types.
///
/// Type parameter P determines pixel layout at compile time.
/// For RGBA/BGRA types, 4th channel is ignored.
pub struct RgbEncoder<P: Pixel> {
    inner: BytesEncoder,
    _marker: PhantomData<P>,
}

impl<P: Pixel> RgbEncoder<P> {
    /// Push rows with explicit stride (in pixels).
    ///
    /// - `data`: Pixel slice
    /// - `rows`: Number of scanlines to push
    /// - `stride`: Pixels per row in buffer (≥ width)
    /// - `stop`: Cancellation token
    pub fn push(
        &mut self,
        data: &[P],
        rows: usize,
        stride: usize,
        stop: impl Stop,
    ) -> Result<()>;

    /// Push contiguous (packed) data.
    ///
    /// Stride assumed to be `width`. Rows inferred from `data.len() / width`.
    pub fn push_packed(&mut self, data: &[P], stop: impl Stop) -> Result<()>;

    // === Status ===

    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn rows_pushed(&self) -> u32;
    pub fn rows_remaining(&self) -> u32;

    // === Finish ===

    pub fn finish(self) -> Result<Vec<u8>>;
    pub fn finish_to<W: Write>(self, output: W) -> Result<W>;
}
```

---

## 8. Planar YCbCr Encoder

```rust
/// Planar YCbCr data for a strip of rows.
///
/// Each plane has its own stride. All planes are f32.
#[derive(Clone, Copy, Debug)]
pub struct YCbCrPlanes<'a> {
    pub y: &'a [f32],
    pub y_stride: usize,
    pub cb: &'a [f32],
    pub cb_stride: usize,
    pub cr: &'a [f32],
    pub cr_stride: usize,
}

/// Encoder for planar f32 YCbCr input.
///
/// Use when you have pre-converted YCbCr from video decoders, etc.
/// Skips RGB→YCbCr conversion entirely.
///
/// Only valid with `ColorMode::YCbCr`. XYB mode requires RGB input.
pub struct YCbCrPlanarEncoder {
    config: EncoderConfig,
    width: u32,
    height: u32,
    rows_pushed: u32,
    // internal state...
}

impl YCbCrPlanarEncoder {
    /// Push full-resolution planes. Encoder subsamples chroma as needed.
    ///
    /// - `planes`: Y, Cb, Cr plane data with per-plane strides
    /// - `rows`: Number of luma rows to push
    /// - `stop`: Cancellation token
    pub fn push(
        &mut self,
        planes: &YCbCrPlanes<'_>,
        rows: usize,
        stop: impl Stop,
    ) -> Result<()>;

    /// Push with pre-subsampled chroma.
    ///
    /// Cb/Cr are already at target chroma resolution.
    /// `y_rows` is luma row count; chroma rows derived from ChromaSubsampling.
    pub fn push_subsampled(
        &mut self,
        planes: &YCbCrPlanes<'_>,
        y_rows: usize,
        stop: impl Stop,
    ) -> Result<()>;

    // === Status ===

    pub fn width(&self) -> u32;
    pub fn height(&self) -> u32;
    pub fn rows_pushed(&self) -> u32;
    pub fn rows_remaining(&self) -> u32;

    // === Finish ===

    pub fn finish(self) -> Result<Vec<u8>>;
    pub fn finish_to<W: Write>(self, output: W) -> Result<W>;
}
```

---

## 9. Error Type

```rust
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Invalid encoder configuration.
    InvalidConfig(&'static str),

    /// Pixel layout incompatible with color mode.
    LayoutMismatch {
        layout: PixelLayout,
        color_mode: ColorMode,
    },

    /// Buffer too small for specified dimensions/rows.
    BufferTooSmall {
        expected: usize,
        actual: usize,
    },

    /// Stride too small for width.
    StrideTooSmall {
        width: u32,
        stride: usize,
    },

    /// Pushed more rows than image height.
    TooManyRows {
        height: u32,
        pushed: u32,
    },

    /// Encoding stopped via cancellation token.
    Stopped,

    /// Not all rows pushed before finish().
    IncompleteImage {
        height: u32,
        pushed: u32,
    },

    /// I/O error during finish_to().
    Io(std::io::Error),

    /// Internal encoding error.
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

---

## 10. Usage Examples

```rust
use zenjpeg::{EncoderConfig, PixelLayout, ChromaSubsampling, Quality};
use enough::Never;
use rgb::{RGB, RGBA};

// ============================================
// Basic: raw bytes
// ============================================

let config = EncoderConfig::new().quality(85);
let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&rgb_bytes, Never)?;
let jpeg = enc.finish()?;

// ============================================
// Basic: rgb crate types
// ============================================

let pixels: Vec<RGB<u8>> = load_image();
let mut enc = config.encode_from_rgb::<RGB<u8>>(1920, 1080)?;
enc.push_packed(&pixels, Never)?;
let jpeg = enc.finish()?;

// ============================================
// With stride (e.g., GPU texture with row padding)
// ============================================

let stride_bytes = 2048 * 3;  // padded rows
let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
enc.push(&padded_bytes, 1080, stride_bytes, Never)?;
let jpeg = enc.finish()?;

// rgb crate with stride (in pixels)
let stride = 2048;
let mut enc = config.encode_from_rgb::<RGB<u8>>(1920, 1080)?;
enc.push(&padded_pixels, 1080, stride, Never)?;

// ============================================
// Streaming with cancellation
// ============================================

use std::sync::atomic::AtomicBool;

let cancel = AtomicBool::new(false);
let mut enc = config.encode_from_rgb::<RGB<u8>>(w, h)?;

for chunk in pixels.chunks(64 * w as usize) {
    let rows = chunk.len() / w as usize;
    enc.push(chunk, rows, w as usize, &cancel)?;
}
let jpeg = enc.finish()?;

// ============================================
// High quality config
// ============================================

let hq = EncoderConfig::new()
    .quality(95)
    .progressive(true)
    .ycbcr(ChromaSubsampling::Full)
    .sharp_yuv(true);

// ============================================
// XYB mode
// ============================================

let xyb = EncoderConfig::new()
    .quality(90)
    .xyb();

let mut enc = xyb.encode_from_rgb::<RGB<f32>>(w, h)?;  // linear f32 input
enc.push_packed(&linear_pixels, Never)?;
let jpeg = enc.finish()?;

// ============================================
// Grayscale
// ============================================

let gray = EncoderConfig::new()
    .quality(85)
    .grayscale();

let mut enc = gray.encode_from_bytes(w, h, PixelLayout::Gray8Srgb)?;
enc.push_packed(&gray_bytes, Never)?;

// ============================================
// RGBA (4th channel ignored)
// ============================================

let mut enc = config.encode_from_rgb::<RGBA<u8>>(w, h)?;
enc.push_packed(&rgba_pixels, Never)?;  // alpha channel ignored
let jpeg = enc.finish()?;

// ============================================
// Planar YCbCr (video decoder output)
// ============================================

use zenjpeg::YCbCrPlanes;

let planes = YCbCrPlanes {
    y: &y_plane,
    y_stride: y_stride,
    cb: &cb_plane,
    cb_stride: cb_stride,
    cr: &cr_plane,
    cr_stride: cr_stride,
};

let mut enc = config.encode_from_ycbcr_planar(w, h)?;
enc.push(&planes, h as usize, Never)?;
let jpeg = enc.finish()?;

// ============================================
// Write to file
// ============================================

let file = File::create("output.jpg")?;
let mut enc = config.encode_from_rgb::<RGB<u8>>(w, h)?;
enc.push_packed(&pixels, Never)?;
enc.finish_to(file)?;

// ============================================
// Async with tokio
// ============================================

async fn encode_async(
    config: EncoderConfig,
    pixels: Vec<RGB<u8>>,
    w: u32,
    h: u32,
    cancel: tokio_util::sync::CancellationToken,
) -> zenjpeg::Result<Vec<u8>> {
    tokio::task::spawn_blocking(move || {
        let mut enc = config.encode_from_rgb::<RGB<u8>>(w, h)?;

        // Push in chunks, checking cancellation
        let chunk_rows = 64;
        for (i, chunk) in pixels.chunks(chunk_rows * w as usize).enumerate() {
            let rows = chunk.len() / w as usize;
            enc.push(chunk, rows, w as usize, &cancel)?;
        }

        enc.finish()
    }).await.map_err(|_| zenjpeg::Error::Internal("task join failed".into()))?
}
```

---

## 11. Migration from Current API

```rust
// Old API
JpegEncoder::new(w, h)
    .pixel_format(PixelFormat::Rgb)
    .quality(Quality::ApproxJpegli(85.0))
    .subsampling(Subsampling::S420)
    .encode(&pixels)?;

// New API
let config = EncoderConfig::new()
    .quality(85)
    .ycbcr(ChromaSubsampling::Quarter);
let mut enc = config.encode_from_rgb::<RGB<u8>>(w, h)?;
enc.push_packed(&pixels, Never)?;
enc.finish()?
```

**Key changes:**
- Config separate from dimensions (reusable)
- Pixel format from type or `PixelLayout` enum
- Stride explicit at push time
- Cancellation on every push
- `push_packed()` for contiguous data

---

## 12. Internal Implementation Notes

`RgbEncoder<P>` is a thin wrapper around `BytesEncoder`:

```rust
impl<P: Pixel> RgbEncoder<P> {
    pub fn push(&mut self, data: &[P], rows: usize, stride: usize, stop: impl Stop) -> Result<()> {
        let stride_bytes = stride * std::mem::size_of::<P>();
        let bytes = bytemuck::cast_slice(data);
        self.inner.push(bytes, rows, stride_bytes, stop)
    }

    pub fn push_packed(&mut self, data: &[P], stop: impl Stop) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);
        self.inner.push_packed(bytes, stop)
    }
}
```

`BytesEncoder` dispatches to format-specific internal methods based on `PixelLayout`.
