//! # Decoder API Guide
//!
//! Complete working examples demonstrating every public method on the decode surface.
//!
//! ## Architecture
//!
//! ```text
//! Decoder
//!     │
//!     ├── Configuration
//!     │   ├── .output_format()       pixel format (Rgb, Gray, Rgba, ...)
//!     │   ├── .output_target()       precision/transfer (Srgb8, SrgbF32, LinearF32, ...)
//!     │   ├── .chroma_upsampling()   filter method (Triangle, NearestNeighbor, LibjpegCompat)
//!     │   ├── .fancy_upsampling()    convenience bool
//!     │   ├── .block_smoothing()     smooth progressive display
//!     │   ├── .dequant_bias()        Laplacian dequantization
//!     │   ├── .apply_icc()           apply embedded ICC profile
//!     │   ├── .strictness()          error handling (Strict, Balanced, Lenient)
//!     │   ├── .strict() / .lenient() convenience shortcuts
//!     │   ├── .preserve()            metadata preservation
//!     │   ├── .preserve_all() / .preserve_none()
//!     │   ├── .limits()              resource limits
//!     │   ├── .max_pixels()          pixel count limit
//!     │   ├── .max_memory()          memory limit
//!     │   └── .gain_map()            UltraHDR handling
//!     │
//!     ├── Info (no decode)
//!     │   ├── .read_info()           → JpegInfo
//!     │   └── .estimate_memory_usage()
//!     │
//!     ├── Full decode
//!     │   ├── .decode()              → DecodeResult (u8 or f32)
//!     │   ├── .decode_coefficients() → DecodedCoefficients
//!     │   └── .decode_to_ycbcr_f32() → DecodedYCbCr
//!     │
//!     └── Streaming decode
//!         └── .scanline_reader()     → ScanlineReader
//!             ├── .read_rows_rgb8()
//!             ├── .read_rows_rgbx8()
//!             ├── .read_rows_bgr8()
//!             ├── .read_rows_rgba8()
//!             ├── .read_rows_bgra8()
//!             ├── .read_rows_bgrx8()
//!             ├── .read_rows_rgba_f32()
//!             ├── .read_rows_gray8()
//!             ├── .read_rows_gray_f32()
//!             ├── .read_rows_gray_linear_f32()
//!             └── .read_rows_ycbcr_planes()
//! ```

#[path = "../src/test_utils.rs"]
mod test_utils;

use enough::Unstoppable;
use test_utils::generate_checkerboard;
use zenjpeg::{
    decoder::{
        ChromaUpsampling, DecodeResult, DecodeWarning, Decoder, GainMapHandling, OutputTarget,
        PixelFormat, PreserveConfig, ScanlineReader, Strictness,
    },
    encoder::{ChromaSubsampling, EncoderConfig, Exif, Orientation, PixelLayout},
    types::{ColorSpace, JpegMode, Limits},
};

// ============================================================================
// Test JPEG helpers
// ============================================================================

/// Encode a 64x64 checkerboard as baseline 4:2:0 JPEG.
fn test_jpeg_420() -> Vec<u8> {
    let img = generate_checkerboard(64, 64, 8, 3);
    EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap()
}

/// Encode a 64x64 checkerboard as baseline 4:4:4 JPEG.
fn test_jpeg_444() -> Vec<u8> {
    let img = generate_checkerboard(64, 64, 8, 3);
    EncoderConfig::ycbcr(85.0, ChromaSubsampling::None)
        .progressive(false)
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap()
}

/// Encode a 64x64 checkerboard as progressive JPEG.
fn test_jpeg_progressive() -> Vec<u8> {
    let img = generate_checkerboard(64, 64, 8, 3);
    EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(true)
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap()
}

/// Encode a 64x64 grayscale JPEG.
fn test_jpeg_gray() -> Vec<u8> {
    let img = generate_checkerboard(64, 64, 8, 1);
    EncoderConfig::grayscale(85.0)
        .progressive(false)
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Gray8Srgb)
        .unwrap()
}

/// Encode a JPEG with EXIF metadata.
fn test_jpeg_with_exif() -> Vec<u8> {
    let img = generate_checkerboard(64, 64, 8, 3);
    EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .request()
        .exif(
            Exif::build()
                .orientation(Orientation::Rotate90)
                .copyright("test"),
        )
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap()
}

/// Encode a JPEG with ICC profile.
fn test_jpeg_with_icc() -> Vec<u8> {
    let img = generate_checkerboard(64, 64, 8, 3);
    EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .request()
        .icc_profile_owned(b"fake-icc-data-for-test".to_vec())
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap()
}

// ============================================================================
// 1. Creating a decoder
// ============================================================================

/// Create a decoder with default settings.
#[test]
fn create_decoder() {
    let _decoder: Decoder = Decoder::new();
}

// ============================================================================
// 2. Basic decode — default config
// ============================================================================

/// `decode()` with default config returns sRGB u8 pixels.
#[test]
fn basic_decode() {
    let jpeg = test_jpeg_420();
    let result: DecodeResult = Decoder::new().decode(&jpeg, Unstoppable).unwrap();

    assert_eq!(result.width(), 64);
    assert_eq!(result.height(), 64);
    assert_eq!(result.dimensions(), (64, 64));
    assert_eq!(result.format(), PixelFormat::Rgb);
    assert_eq!(result.output_target(), OutputTarget::Srgb8);
    assert_eq!(result.bytes_per_pixel(), 3);
    assert_eq!(result.stride(), 64 * 3);

    let pixels = result.pixels_u8().unwrap();
    assert_eq!(pixels.len(), 64 * 64 * 3);

    // f32 is None for Srgb8 target
    assert!(result.pixels_f32().is_none());
}

// ============================================================================
// 3. Output format — pixel channel layouts
// ============================================================================

/// `.output_format()` controls the channel layout of decoded pixels.
#[test]
fn output_format_rgb() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(result.format(), PixelFormat::Rgb);
    assert_eq!(result.bytes_per_pixel(), 3);
    assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64 * 3);
}

/// RGBA is available as f32 via the scanline reader (`read_rows_rgba_f32`).
/// The buffered `.decode()` path supports Rgb and Gray pixel formats.
#[test]
fn output_format_rgba_f32_via_scanline() {
    let jpeg = test_jpeg_420();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();
    let (w, h) = (reader.width() as usize, reader.height() as usize);
    let stride = w * 4; // 4 f32s per RGBA pixel
    let mut buf = vec![0.0f32; stride * h];
    let img = imgref::ImgRefMut::new_stride(&mut buf, w * 4, h, stride);
    let rows = reader.read_rows_rgba_f32(img).unwrap();
    assert_eq!(rows, h);
}

#[test]
fn output_format_gray() {
    let jpeg = test_jpeg_gray();
    let result = Decoder::new()
        .output_format(PixelFormat::Gray)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(result.format(), PixelFormat::Gray);
    assert_eq!(result.bytes_per_pixel(), 1);
}

/// `PixelFormat::Bgr` decodes to 3 bytes per pixel in B-G-R order.
/// Pixels should match RGB output with R and B channels swapped.
#[test]
fn output_format_bgr() {
    let jpeg = test_jpeg_420();
    let rgb = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgr = Decoder::new()
        .output_format(PixelFormat::Bgr)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(bgr.format(), PixelFormat::Bgr);
    assert_eq!(bgr.bytes_per_pixel(), 3);
    let rgb_px = rgb.pixels_u8().unwrap();
    let bgr_px = bgr.pixels_u8().unwrap();
    assert_eq!(rgb_px.len(), bgr_px.len());
    for (rgb_chunk, bgr_chunk) in rgb_px.chunks_exact(3).zip(bgr_px.chunks_exact(3)) {
        assert_eq!(rgb_chunk[0], bgr_chunk[2], "R != B-position in BGR");
        assert_eq!(rgb_chunk[1], bgr_chunk[1], "G channels must match");
        assert_eq!(rgb_chunk[2], bgr_chunk[0], "B != R-position in BGR");
    }
}

/// `PixelFormat::Bgra` decodes to 4 bytes per pixel in B-G-R-A order (A=255).
#[test]
fn output_format_bgra() {
    let jpeg = test_jpeg_420();
    let rgb = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgra = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(bgra.format(), PixelFormat::Bgra);
    assert_eq!(bgra.bytes_per_pixel(), 4);
    let rgb_px = rgb.pixels_u8().unwrap();
    let bgra_px = bgra.pixels_u8().unwrap();
    assert_eq!(bgra_px.len(), 64 * 64 * 4);
    for (rgb_chunk, bgra_chunk) in rgb_px.chunks_exact(3).zip(bgra_px.chunks_exact(4)) {
        assert_eq!(rgb_chunk[0], bgra_chunk[2], "R in BGRA position 2");
        assert_eq!(rgb_chunk[1], bgra_chunk[1], "G in BGRA position 1");
        assert_eq!(rgb_chunk[2], bgra_chunk[0], "B in BGRA position 0");
        assert_eq!(bgra_chunk[3], 255, "A must be 255");
    }
}

/// `PixelFormat::Rgba` decodes to 4 bytes per pixel in R-G-B-A order (A=255).
#[test]
fn output_format_rgba() {
    let jpeg = test_jpeg_420();
    let rgb = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let rgba = Decoder::new()
        .output_format(PixelFormat::Rgba)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(rgba.format(), PixelFormat::Rgba);
    assert_eq!(rgba.bytes_per_pixel(), 4);
    let rgb_px = rgb.pixels_u8().unwrap();
    let rgba_px = rgba.pixels_u8().unwrap();
    for (rgb_chunk, rgba_chunk) in rgb_px.chunks_exact(3).zip(rgba_px.chunks_exact(4)) {
        assert_eq!(rgb_chunk[0], rgba_chunk[0]);
        assert_eq!(rgb_chunk[1], rgba_chunk[1]);
        assert_eq!(rgb_chunk[2], rgba_chunk[2]);
        assert_eq!(rgba_chunk[3], 255);
    }
}

/// `PixelFormat::Bgrx` decodes to 4 bytes per pixel in B-G-R-X order (X=255).
#[test]
fn output_format_bgrx() {
    let jpeg = test_jpeg_444();
    let bgra = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgrx = Decoder::new()
        .output_format(PixelFormat::Bgrx)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert_eq!(bgrx.format(), PixelFormat::Bgrx);
    // BGRX and BGRA should produce identical bytes
    assert_eq!(bgra.pixels_u8().unwrap(), bgrx.pixels_u8().unwrap());
}

/// BGR/BGRA/RGBA work with grayscale images.
#[test]
fn output_format_bgr_grayscale() {
    let jpeg = test_jpeg_gray();
    let bgr = Decoder::new()
        .output_format(PixelFormat::Bgr)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgr_px = bgr.pixels_u8().unwrap();
    // Grayscale: all channels equal, R/B swap is no-op
    for chunk in bgr_px.chunks_exact(3) {
        assert_eq!(chunk[0], chunk[1]);
        assert_eq!(chunk[1], chunk[2]);
    }

    let bgra = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgra_px = bgra.pixels_u8().unwrap();
    for chunk in bgra_px.chunks_exact(4) {
        assert_eq!(chunk[0], chunk[1]);
        assert_eq!(chunk[1], chunk[2]);
        assert_eq!(chunk[3], 255);
    }
}

/// BGR/BGRA work with the fast i16 4:4:4 path.
#[test]
fn output_format_bgr_fast_444_path() {
    let jpeg = test_jpeg_444();
    let rgb = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgra = Decoder::new()
        .output_format(PixelFormat::Bgra)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let rgb_px = rgb.pixels_u8().unwrap();
    let bgra_px = bgra.pixels_u8().unwrap();
    for (rgb_chunk, bgra_chunk) in rgb_px.chunks_exact(3).zip(bgra_px.chunks_exact(4)) {
        assert_eq!(rgb_chunk[0], bgra_chunk[2]); // R
        assert_eq!(rgb_chunk[1], bgra_chunk[1]); // G
        assert_eq!(rgb_chunk[2], bgra_chunk[0]); // B
        assert_eq!(bgra_chunk[3], 255);
    }
}

/// BGR/BGRA work with progressive JPEGs (buffered decode path).
#[test]
fn output_format_bgr_progressive() {
    let jpeg = test_jpeg_progressive();
    let rgb = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let bgr = Decoder::new()
        .output_format(PixelFormat::Bgr)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let rgb_px = rgb.pixels_u8().unwrap();
    let bgr_px = bgr.pixels_u8().unwrap();
    for (r, b) in rgb_px.chunks_exact(3).zip(bgr_px.chunks_exact(3)) {
        assert_eq!(r[0], b[2]);
        assert_eq!(r[1], b[1]);
        assert_eq!(r[2], b[0]);
    }
}

// ============================================================================
// 4. Output target — precision and transfer function
// ============================================================================

/// `Srgb8` (default): u8 pixels, integer IDCT, fastest.
#[test]
fn output_target_srgb8() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::Srgb8)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert!(!result.output_target().is_f32());
    assert!(!result.output_target().is_linear());
    assert!(!result.output_target().is_precise());
    assert!(result.pixels_u8().is_some());
    assert!(result.pixels_f32().is_none());
}

/// `SrgbF32`: f32 pixels in sRGB gamma, unclamped IDCT.
#[test]
fn output_target_srgb_f32() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert!(result.output_target().is_f32());
    assert!(!result.output_target().is_linear());
    assert!(result.pixels_f32().is_some());
    assert!(result.pixels_u8().is_none());

    let pixels = result.pixels_f32().unwrap();
    assert_eq!(pixels.len(), 64 * 64 * 3);
}

/// `LinearF32`: f32 pixels in linear light.
#[test]
fn output_target_linear_f32() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::LinearF32)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert!(result.output_target().is_f32());
    assert!(result.output_target().is_linear());
    assert!(!result.output_target().is_precise());
}

/// `SrgbF32Precise`: f32 with Laplacian dequant biases (highest quality).
#[test]
fn output_target_precise() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::SrgbF32Precise)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert!(result.output_target().is_f32());
    assert!(result.output_target().is_precise());
}

/// `LinearF32Precise`: linear light + dequant biases.
#[test]
fn output_target_linear_precise() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::LinearF32Precise)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert!(result.output_target().is_f32());
    assert!(result.output_target().is_linear());
    assert!(result.output_target().is_precise());
}

// ============================================================================
// 5. Dequant bias convenience
// ============================================================================

/// `.dequant_bias(true)` is shorthand for `SrgbF32Precise`.
#[test]
fn dequant_bias_shortcut() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .dequant_bias(true)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert!(result.output_target().is_precise());
    assert!(result.pixels_f32().is_some());
}

// ============================================================================
// 6. Chroma upsampling methods
// ============================================================================

/// Triangle filter (default): jpegli-style separable 3:1 interpolation.
#[test]
fn chroma_upsampling_triangle() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .chroma_upsampling(ChromaUpsampling::Triangle)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert!(result.pixels_u8().is_some());
}

/// NearestNeighbor: fastest, lowest quality.
#[test]
fn chroma_upsampling_nearest() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert!(result.pixels_u8().is_some());
}

/// LibjpegCompat: pixel-exact match with djpeg/libjpeg-turbo.
#[test]
fn chroma_upsampling_libjpeg_compat() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .chroma_upsampling(ChromaUpsampling::LibjpegCompat)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert!(result.pixels_u8().is_some());
}

/// `.fancy_upsampling(false)` maps to NearestNeighbor.
#[test]
fn fancy_upsampling_toggle() {
    let jpeg = test_jpeg_420();

    // false → NearestNeighbor (fast)
    let fast = Decoder::new()
        .fancy_upsampling(false)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    // true → Triangle (default quality)
    let fancy = Decoder::new()
        .fancy_upsampling(true)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    // Both succeed with valid output
    assert!(!fast.pixels_u8().unwrap().is_empty());
    assert!(!fancy.pixels_u8().unwrap().is_empty());
    assert_eq!(fast.width(), fancy.width());
    assert_eq!(fast.height(), fancy.height());
}

// ============================================================================
// 7. Strictness levels
// ============================================================================

/// `Strict`: fail on any spec violation.
#[test]
fn strictness_strict() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .strictness(Strictness::Strict)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok(), "valid JPEG should decode in strict mode");
}

/// `Balanced` (default): recover from truncation, reject violations.
#[test]
fn strictness_balanced() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .strictness(Strictness::Balanced)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

/// `Lenient`: maximum compatibility.
#[test]
fn strictness_lenient() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .strictness(Strictness::Lenient)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

/// `.strict()` and `.lenient()` are convenience shortcuts.
#[test]
fn strictness_convenience() {
    let jpeg = test_jpeg_420();
    assert!(Decoder::new().strict().decode(&jpeg, Unstoppable).is_ok());
    assert!(Decoder::new().lenient().decode(&jpeg, Unstoppable).is_ok());
}

// ============================================================================
// 8. Resource limits
// ============================================================================

/// `.max_pixels()` rejects images larger than the limit.
#[test]
fn max_pixels_rejects_large() {
    let jpeg = test_jpeg_420(); // 64x64 = 4096 pixels
    let result = Decoder::new()
        .max_pixels(100) // Way too small
        .decode(&jpeg, Unstoppable);
    assert!(
        result.is_err(),
        "should reject 64x64 image with max_pixels=100"
    );
}

/// `.max_pixels(0)` means unlimited.
#[test]
fn max_pixels_unlimited() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new().max_pixels(0).decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

/// `.limits()` applies both pixel and memory limits from a struct.
#[test]
fn limits_struct() {
    let jpeg = test_jpeg_420();
    let limits = Limits::default().max_pixels(1_000_000);
    let result = Decoder::new().limits(limits).decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

/// `.max_memory()` controls allocation ceiling.
#[test]
fn max_memory_limit() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .max_memory(1_000_000) // 1 MB — enough for 64x64
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

// ============================================================================
// 9. Metadata preservation
// ============================================================================

/// Default preservation keeps EXIF, XMP, ICC, IPTC.
#[test]
fn preserve_default() {
    let jpeg = test_jpeg_with_exif();
    let result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let extras = result.extras().unwrap();
    assert!(extras.exif().is_some(), "default should preserve EXIF");
}

/// `.preserve_all()` keeps everything.
#[test]
fn preserve_all() {
    let jpeg = test_jpeg_with_exif();
    let result = Decoder::new()
        .preserve_all()
        .decode(&jpeg, Unstoppable)
        .unwrap();
    assert!(result.extras().is_some());
}

/// `.preserve_none()` drops all metadata.
#[test]
fn preserve_none() {
    let jpeg = test_jpeg_with_exif();
    let result = Decoder::new()
        .preserve_none()
        .decode(&jpeg, Unstoppable)
        .unwrap();
    // extras may still exist but individual fields should be empty
    if let Some(extras) = result.extras() {
        assert!(extras.exif().is_none(), "preserve_none should drop EXIF");
    }
}

/// Custom preservation config.
#[test]
fn preserve_custom() {
    let jpeg = test_jpeg_with_exif();
    let preserve = PreserveConfig::none()
        .exif(true) // Keep EXIF only
        .icc(zenjpeg::decoder::IccPreserve::None);

    let result = Decoder::new()
        .preserve(preserve)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    if let Some(extras) = result.extras() {
        assert!(extras.exif().is_some());
    }
}

// ============================================================================
// 10. DecodeResult accessors
// ============================================================================

/// Full walkthrough of DecodeResult methods.
#[test]
fn decode_result_accessors() {
    let jpeg = test_jpeg_420();
    let mut result = Decoder::new()
        .preserve_all()
        .decode(&jpeg, Unstoppable)
        .unwrap();

    // Dimensions
    assert_eq!(result.width(), 64);
    assert_eq!(result.height(), 64);
    assert_eq!(result.dimensions(), (64, 64));

    // Format info
    assert_eq!(result.format(), PixelFormat::Rgb);
    assert_eq!(result.output_target(), OutputTarget::Srgb8);
    assert_eq!(result.bytes_per_pixel(), 3);
    assert_eq!(result.stride(), 64 * 3);

    // Pixel access
    assert!(result.pixels_u8().is_some());
    assert!(result.pixels_f32().is_none());

    // Warnings
    let _warnings: &[DecodeWarning] = result.warnings();
    let _has_warnings: bool = result.has_warnings();

    // Extras (metadata)
    let _extras_ref = result.extras();
    let _extras_owned = result.take_extras();
}

/// `.into_pixels_u8()` takes ownership of the pixel buffer.
#[test]
fn into_pixels_u8() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let pixels: Vec<u8> = result.into_pixels_u8().unwrap();
    assert_eq!(pixels.len(), 64 * 64 * 3);
}

/// `.into_pixels_f32()` for f32 targets.
#[test]
fn into_pixels_f32() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let pixels: Vec<f32> = result.into_pixels_f32().unwrap();
    assert_eq!(pixels.len(), 64 * 64 * 3);
}

/// `.to_u16()` converts f32 pixels to 16-bit.
#[test]
fn to_u16_conversion() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new()
        .output_target(OutputTarget::SrgbF32)
        .decode(&jpeg, Unstoppable)
        .unwrap();
    let u16_pixels: Vec<u16> = result.to_u16().unwrap();
    assert_eq!(u16_pixels.len(), 64 * 64 * 3);
    // Values are in full u16 range (the type guarantees this)
    assert!(
        u16_pixels.iter().any(|&v| v > 0),
        "should have non-zero pixels"
    );
}

/// `.into_parts()` decomposes the result.
#[test]
fn into_parts() {
    let jpeg = test_jpeg_420();
    let result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let (pixels_u8, pixels_f32, width, height, format, _extras) = result.into_parts();
    assert!(pixels_u8.is_some());
    assert!(pixels_f32.is_none());
    assert_eq!(width, 64);
    assert_eq!(height, 64);
    assert_eq!(format, PixelFormat::Rgb);
}

// ============================================================================
// 11. Read info without decoding
// ============================================================================

/// `.read_info()` returns image metadata without decoding pixels.
#[test]
fn read_info() {
    let jpeg = test_jpeg_420();
    let info = Decoder::new().read_info(&jpeg).unwrap();

    assert_eq!(info.dimensions.width, 64);
    assert_eq!(info.dimensions.height, 64);
    assert_eq!(info.precision, 8);
    assert_eq!(info.num_components, 3);
    assert_eq!(info.color_space, ColorSpace::YCbCr);
    assert!(!info.is_xyb);
    // Baseline non-progressive
    assert_eq!(info.mode, JpegMode::Baseline);
}

/// Grayscale info.
#[test]
fn read_info_grayscale() {
    let jpeg = test_jpeg_gray();
    let info = Decoder::new().read_info(&jpeg).unwrap();
    assert_eq!(info.num_components, 1);
    assert_eq!(info.color_space, ColorSpace::Grayscale);
}

/// Progressive info.
#[test]
fn read_info_progressive() {
    let jpeg = test_jpeg_progressive();
    let info = Decoder::new().read_info(&jpeg).unwrap();
    assert_eq!(info.mode, JpegMode::Progressive);
}

// ============================================================================
// 12. Memory estimation
// ============================================================================

/// `.estimate_memory_usage()` returns peak memory estimate.
#[test]
fn memory_estimation() {
    let estimate = Decoder::new().estimate_memory_usage(1920, 1080);
    // Should be reasonable (at least RGB output: 1920*1080*3 ≈ 6.2 MB)
    assert!(estimate >= 1920 * 1080 * 3);
    // But not absurdly large
    assert!(estimate < 100_000_000);
}

// ============================================================================
// 13. Decode coefficients
// ============================================================================

/// `.decode_coefficients()` extracts raw DCT coefficients.
#[test]
fn decode_coefficients() {
    let jpeg = test_jpeg_420();
    let coeffs = Decoder::new()
        .decode_coefficients(&jpeg, Unstoppable)
        .unwrap();

    // 3 components for YCbCr
    assert_eq!(coeffs.num_components(), 3);

    // Access Y component
    let y = &coeffs.components[0];
    assert!(y.num_blocks() > 0);

    // First block, first coefficient is DC
    let block = y.block(0);
    assert_eq!(block.len(), 64); // 8x8 DCT block in zigzag order

    // Block by (x, y) coordinates
    let _block_00 = y.block_at(0, 0);
}

/// Coefficient comparison between two encodes.
#[test]
fn coefficient_comparison() {
    let img = generate_checkerboard(64, 64, 8, 3);
    let jpeg_a = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap();
    let jpeg_b = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(&img.pixels, img.width, img.height, PixelLayout::Rgb8Srgb)
        .unwrap();

    let decoder = Decoder::new();
    let coeffs_a = decoder.decode_coefficients(&jpeg_a, Unstoppable).unwrap();
    let coeffs_b = decoder.decode_coefficients(&jpeg_b, Unstoppable).unwrap();

    let comparison = coeffs_a.compare(&coeffs_b);
    let _diff_pct = comparison.diff_block_pct();
    let _dc_diff_pct = comparison.dc_diff_pct();
    // Different quality → different coefficients
    assert!(comparison.diff_block_pct() > 0.0);
}

// ============================================================================
// 14. Decode to YCbCr f32
// ============================================================================

/// `.decode_to_ycbcr_f32()` bypasses RGB conversion.
#[test]
fn decode_ycbcr_f32() {
    let jpeg = test_jpeg_420();
    let ycbcr = Decoder::new()
        .decode_to_ycbcr_f32(&jpeg, Unstoppable)
        .unwrap();

    assert_eq!(ycbcr.dimensions(), (64, 64));
    assert_eq!(ycbcr.plane_size(), 64 * 64);

    // Raw values in [-128, 127] range
    assert_eq!(ycbcr.y.len(), 64 * 64);
    assert_eq!(ycbcr.cb.len(), 64 * 64);
    assert_eq!(ycbcr.cr.len(), 64 * 64);

    // Convert to JPEG range [0, 255]
    let y_jpeg = ycbcr.y_to_jpeg_range();
    let cb_jpeg = ycbcr.cb_to_jpeg_range();
    let cr_jpeg = ycbcr.cr_to_jpeg_range();
    assert_eq!(y_jpeg.len(), 64 * 64);
    assert_eq!(cb_jpeg.len(), 64 * 64);
    assert_eq!(cr_jpeg.len(), 64 * 64);
}

// ============================================================================
// 15. Decode f32 (deprecated convenience)
// ============================================================================

// ============================================================================
// 16. Scanline reader — streaming decode
// ============================================================================

/// `scanline_reader()` decodes row-by-row into caller-provided buffers.
#[test]
fn scanline_reader_rgb8() {
    let jpeg = test_jpeg_444();

    let mut reader: ScanlineReader<'_> = Decoder::new().scanline_reader(&jpeg).unwrap();

    // Reader metadata
    assert_eq!(reader.width(), 64);
    assert_eq!(reader.height(), 64);
    assert!(!reader.is_grayscale());
    assert_eq!(reader.num_components(), 3);
    assert_eq!(reader.current_row(), 0);
    assert!(!reader.is_finished());

    // Info struct
    let info = reader.info();
    assert_eq!(info.dimensions.width, 64);
    assert_eq!(info.dimensions.height, 64);

    // Decode all rows into a flat buffer
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut pixels = vec![0u8; stride * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * stride..];
        let output = imgref::ImgRefMut::new(slice, stride, remaining);
        let n = reader.read_rows_rgb8(output).unwrap();
        rows_decoded += n;
    }

    assert_eq!(rows_decoded, h);
    assert!(reader.is_finished());
}

// ============================================================================
// 17. Scanline reader — RGBX (4 bytes/pixel)
// ============================================================================

/// `read_rows_rgbx8()` outputs 4 bytes per pixel (padding byte).
#[test]
fn scanline_reader_rgbx8() {
    let jpeg = test_jpeg_444();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 4;
    let mut pixels = vec![0u8; stride * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * stride..];
        let output = imgref::ImgRefMut::new(slice, stride, remaining);
        let n = reader.read_rows_rgbx8(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

// ============================================================================
// 17b. Scanline reader — BGR/BGRA/RGBA/BGRX (fused output)
// ============================================================================

/// `read_rows_bgr8()` outputs 3 bytes per pixel in B-G-R order.
#[test]
fn scanline_reader_bgr8() {
    let jpeg = test_jpeg_420();

    // Decode with RGB for reference
    let mut reader_rgb = Decoder::new().scanline_reader(&jpeg).unwrap();
    let w = reader_rgb.width() as usize;
    let h = reader_rgb.height() as usize;
    let stride3 = w * 3;
    let mut rgb = vec![0u8; stride3 * h];
    let mut n = 0;
    while !reader_rgb.is_finished() {
        let sl = &mut rgb[n * stride3..];
        n += reader_rgb
            .read_rows_rgb8(imgref::ImgRefMut::new(sl, stride3, h - n))
            .unwrap();
    }

    // Decode with BGR
    let mut reader_bgr = Decoder::new().scanline_reader(&jpeg).unwrap();
    let mut bgr = vec![0u8; stride3 * h];
    let mut n = 0;
    while !reader_bgr.is_finished() {
        let sl = &mut bgr[n * stride3..];
        n += reader_bgr
            .read_rows_bgr8(imgref::ImgRefMut::new(sl, stride3, h - n))
            .unwrap();
    }

    // Verify R/B swap
    for (r, b) in rgb.chunks_exact(3).zip(bgr.chunks_exact(3)) {
        assert_eq!(r[0], b[2], "R <-> B swap failed");
        assert_eq!(r[1], b[1], "G must match");
        assert_eq!(r[2], b[0], "B <-> R swap failed");
    }
}

/// `read_rows_rgba8()` outputs 4 bytes per pixel in R-G-B-A order (A=255).
#[test]
fn scanline_reader_rgba8() {
    let jpeg = test_jpeg_420();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride4 = w * 4;
    let mut pixels = vec![0u8; stride4 * h];

    let mut n = 0;
    while !reader.is_finished() {
        let sl = &mut pixels[n * stride4..];
        n += reader
            .read_rows_rgba8(imgref::ImgRefMut::new(sl, stride4, h - n))
            .unwrap();
    }
    assert_eq!(n, h);
    // Every alpha must be 255
    for chunk in pixels.chunks_exact(4) {
        assert_eq!(chunk[3], 255, "alpha must be 255");
    }
}

/// `read_rows_bgra8()` outputs 4 bytes per pixel in B-G-R-A order (A=255).
#[test]
fn scanline_reader_bgra8() {
    let jpeg = test_jpeg_444();

    // Reference RGB
    let mut reader_rgb = Decoder::new().scanline_reader(&jpeg).unwrap();
    let w = reader_rgb.width() as usize;
    let h = reader_rgb.height() as usize;
    let stride3 = w * 3;
    let mut rgb = vec![0u8; stride3 * h];
    let mut n = 0;
    while !reader_rgb.is_finished() {
        let sl = &mut rgb[n * stride3..];
        n += reader_rgb
            .read_rows_rgb8(imgref::ImgRefMut::new(sl, stride3, h - n))
            .unwrap();
    }

    // BGRA
    let mut reader_bgra = Decoder::new().scanline_reader(&jpeg).unwrap();
    let stride4 = w * 4;
    let mut bgra = vec![0u8; stride4 * h];
    let mut n = 0;
    while !reader_bgra.is_finished() {
        let sl = &mut bgra[n * stride4..];
        n += reader_bgra
            .read_rows_bgra8(imgref::ImgRefMut::new(sl, stride4, h - n))
            .unwrap();
    }

    for (rgb_chunk, bgra_chunk) in rgb.chunks_exact(3).zip(bgra.chunks_exact(4)) {
        assert_eq!(rgb_chunk[0], bgra_chunk[2], "R in BGRA[2]");
        assert_eq!(rgb_chunk[1], bgra_chunk[1], "G in BGRA[1]");
        assert_eq!(rgb_chunk[2], bgra_chunk[0], "B in BGRA[0]");
        assert_eq!(bgra_chunk[3], 255, "alpha must be 255");
    }
}

/// `read_rows_bgrx8()` is identical to `read_rows_bgra8()`.
#[test]
fn scanline_reader_bgrx8() {
    let jpeg = test_jpeg_420();

    let mut reader_a = Decoder::new().scanline_reader(&jpeg).unwrap();
    let w = reader_a.width() as usize;
    let h = reader_a.height() as usize;
    let stride4 = w * 4;
    let mut bgra = vec![0u8; stride4 * h];
    let mut n = 0;
    while !reader_a.is_finished() {
        let sl = &mut bgra[n * stride4..];
        n += reader_a
            .read_rows_bgra8(imgref::ImgRefMut::new(sl, stride4, h - n))
            .unwrap();
    }

    let mut reader_x = Decoder::new().scanline_reader(&jpeg).unwrap();
    let mut bgrx = vec![0u8; stride4 * h];
    let mut n = 0;
    while !reader_x.is_finished() {
        let sl = &mut bgrx[n * stride4..];
        n += reader_x
            .read_rows_bgrx8(imgref::ImgRefMut::new(sl, stride4, h - n))
            .unwrap();
    }

    assert_eq!(bgra, bgrx);
}

/// Scanline BGR/BGRA work on grayscale images.
#[test]
fn scanline_reader_bgr_grayscale() {
    let jpeg = test_jpeg_gray();

    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();
    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride3 = w * 3;
    let mut bgr = vec![0u8; stride3 * h];
    let mut n = 0;
    while !reader.is_finished() {
        let sl = &mut bgr[n * stride3..];
        n += reader
            .read_rows_bgr8(imgref::ImgRefMut::new(sl, stride3, h - n))
            .unwrap();
    }
    // Grayscale: all three channels equal
    for chunk in bgr.chunks_exact(3) {
        assert_eq!(chunk[0], chunk[1]);
        assert_eq!(chunk[1], chunk[2]);
    }

    let mut reader2 = Decoder::new().scanline_reader(&jpeg).unwrap();
    let stride4 = w * 4;
    let mut bgra = vec![0u8; stride4 * h];
    let mut n = 0;
    while !reader2.is_finished() {
        let sl = &mut bgra[n * stride4..];
        n += reader2
            .read_rows_bgra8(imgref::ImgRefMut::new(sl, stride4, h - n))
            .unwrap();
    }
    for chunk in bgra.chunks_exact(4) {
        assert_eq!(chunk[0], chunk[1]);
        assert_eq!(chunk[1], chunk[2]);
        assert_eq!(chunk[3], 255);
    }
}

// ============================================================================
// 18. Scanline reader — RGBA f32
// ============================================================================

/// `read_rows_rgba_f32()` outputs 4 f32 values per pixel.
#[test]
fn scanline_reader_rgba_f32() {
    let jpeg = test_jpeg_444();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 4; // 4 floats per pixel
    let mut pixels = vec![0.0f32; stride * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * stride..];
        let output = imgref::ImgRefMut::new(slice, stride, remaining);
        let n = reader.read_rows_rgba_f32(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

// ============================================================================
// 19. Scanline reader — grayscale
// ============================================================================

/// `read_rows_gray8()` for grayscale JPEGs.
#[test]
fn scanline_reader_gray8() {
    let jpeg = test_jpeg_gray();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    assert!(reader.is_grayscale());
    assert_eq!(reader.num_components(), 1);

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut pixels = vec![0u8; w * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * w..];
        let output = imgref::ImgRefMut::new(slice, w, remaining);
        let n = reader.read_rows_gray8(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

/// `read_rows_gray_f32()` for grayscale with float output.
#[test]
fn scanline_reader_gray_f32() {
    let jpeg = test_jpeg_gray();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut pixels = vec![0.0f32; w * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * w..];
        let output = imgref::ImgRefMut::new(slice, w, remaining);
        let n = reader.read_rows_gray_f32(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

/// `read_rows_gray_linear_f32()` for grayscale in linear light.
#[test]
fn scanline_reader_gray_linear_f32() {
    let jpeg = test_jpeg_gray();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut pixels = vec![0.0f32; w * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * w..];
        let output = imgref::ImgRefMut::new(slice, w, remaining);
        let n = reader.read_rows_gray_linear_f32(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

// ============================================================================
// 20. Scanline reader — YCbCr planes
// ============================================================================

/// `read_rows_ycbcr_planes()` outputs separate Y, Cb, Cr f32 planes.
#[test]
fn scanline_reader_ycbcr_planes() {
    let jpeg = test_jpeg_444();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let mut y_buf = vec![0.0f32; w * h];
    let mut cb_buf = vec![0.0f32; w * h];
    let mut cr_buf = vec![0.0f32; w * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let offset = rows_decoded * w;
        let n = reader
            .read_rows_ycbcr_planes(
                &mut y_buf[offset..],
                &mut cb_buf[offset..],
                &mut cr_buf[offset..],
                w,         // stride (floats per row)
                remaining, // max rows
            )
            .unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

// ============================================================================
// 21. Scanline reader — progressive (buffered mode)
// ============================================================================

/// Progressive JPEGs work via scanline reader (internally buffered).
#[test]
fn scanline_reader_progressive() {
    let jpeg = test_jpeg_progressive();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut pixels = vec![0u8; stride * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * stride..];
        let output = imgref::ImgRefMut::new(slice, stride, remaining);
        let n = reader.read_rows_rgb8(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

// ============================================================================
// 22. Scanline reader — 4:2:0 (upsampling)
// ============================================================================

/// 4:2:0 subsampled JPEGs work through the scanline reader.
#[test]
fn scanline_reader_420() {
    let jpeg = test_jpeg_420();
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let w = reader.width() as usize;
    let h = reader.height() as usize;
    let stride = w * 3;
    let mut pixels = vec![0u8; stride * h];

    let mut rows_decoded = 0;
    while !reader.is_finished() {
        let remaining = h - rows_decoded;
        let slice = &mut pixels[rows_decoded * stride..];
        let output = imgref::ImgRefMut::new(slice, stride, remaining);
        let n = reader.read_rows_rgb8(output).unwrap();
        rows_decoded += n;
    }
    assert_eq!(rows_decoded, h);
}

// ============================================================================
// 23. ScanlineReader info and subsampling
// ============================================================================

/// `ScanlineReader` exposes image info and subsampling mode.
#[test]
fn scanline_reader_info() {
    let jpeg = test_jpeg_420();
    let reader = Decoder::new().scanline_reader(&jpeg).unwrap();

    let info = reader.info();
    assert_eq!(info.dimensions.width, 64);
    assert_eq!(info.dimensions.height, 64);
    assert_eq!(info.color_space, ColorSpace::YCbCr);
    assert!(!info.is_xyb);

    let sub = reader.subsampling();
    // 4:2:0 = S420
    assert!(
        sub == zenjpeg::types::Subsampling::S420 || sub == zenjpeg::types::Subsampling::S444,
        "expected S420 or S444, got {:?}",
        sub
    );
}

// ============================================================================
// 24. Block smoothing
// ============================================================================

/// `.block_smoothing()` enables progressive display smoothing.
#[test]
fn block_smoothing() {
    let jpeg = test_jpeg_420();
    // Just verify it doesn't crash
    let result = Decoder::new()
        .block_smoothing(true)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());

    let result = Decoder::new()
        .block_smoothing(false)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

// ============================================================================
// 25. Apply ICC
// ============================================================================

/// `.apply_icc()` enables ICC profile application.
/// (Actual transform requires cms feature; without it, this is a no-op.)
#[test]
fn apply_icc_toggle() {
    let jpeg = test_jpeg_with_icc();
    let result = Decoder::new().apply_icc(true).decode(&jpeg, Unstoppable);
    assert!(result.is_ok());

    let result = Decoder::new().apply_icc(false).decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

// ============================================================================
// 26. GainMap handling
// ============================================================================

/// `.gain_map()` controls UltraHDR gain map behavior.
/// On regular JPEGs, all modes succeed (no gain map to process).
#[test]
fn gain_map_handling() {
    let jpeg = test_jpeg_420();

    let result = Decoder::new()
        .gain_map(GainMapHandling::Discard)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());

    let result = Decoder::new()
        .gain_map(GainMapHandling::PreserveRaw)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());

    let result = Decoder::new()
        .gain_map(GainMapHandling::Decode)
        .decode(&jpeg, Unstoppable);
    assert!(result.is_ok());
}

// ============================================================================
// 27. DecodedExtras — metadata extraction
// ============================================================================

/// Full walkthrough of DecodedExtras accessors.
#[test]
fn decoded_extras() {
    let jpeg = test_jpeg_with_exif();
    let result = Decoder::new()
        .preserve_all()
        .decode(&jpeg, Unstoppable)
        .unwrap();

    let extras = result.extras().unwrap();

    // Check EXIF
    assert!(extras.exif().is_some());

    // Other metadata accessors (may be None for this test image)
    let _xmp: Option<&str> = extras.xmp();
    let _icc: Option<&[u8]> = extras.icc_profile();
    let _iptc: Option<&[u8]> = extras.iptc();
    let _jfif = extras.jfif();
    let _adobe = extras.adobe();
    let _mpf = extras.mpf();

    // Iterate segments
    let _all_segments = extras.segments();
    let _is_empty = extras.is_empty();

    // Comments
    let _comments: Vec<&str> = extras.comments().collect();

    // Secondary images (MPF)
    let _secondaries = extras.secondary_images();
    let _gainmap = extras.gainmap();
    let _depth = extras.depth_map();

    // Convert to encoder segments for round-tripping
    let _encoder_segs = extras.to_encoder_segments();
    let _raw_segs = extras.to_raw_segments();
}

// ============================================================================
// 28. Decode grayscale
// ============================================================================

/// Grayscale JPEG decodes to single-channel output.
#[test]
fn decode_grayscale() {
    let jpeg = test_jpeg_gray();
    let result = Decoder::new()
        .output_format(PixelFormat::Gray)
        .decode(&jpeg, Unstoppable)
        .unwrap();

    assert_eq!(result.format(), PixelFormat::Gray);
    assert_eq!(result.bytes_per_pixel(), 1);
    assert_eq!(result.pixels_u8().unwrap().len(), 64 * 64);
}

// ============================================================================
// 29. Decode progressive
// ============================================================================

/// Progressive JPEGs decode correctly.
#[test]
fn decode_progressive() {
    let jpeg = test_jpeg_progressive();
    let result = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    assert_eq!(result.dimensions(), (64, 64));
    assert!(result.pixels_u8().is_some());
}

// ============================================================================
// 30. Config is reusable
// ============================================================================

/// Same decoder config can decode multiple images.
#[test]
fn config_reuse() {
    let decoder = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .fancy_upsampling(true);

    let jpeg_a = test_jpeg_420();
    let jpeg_b = test_jpeg_444();
    let jpeg_c = test_jpeg_progressive();

    let a = decoder.decode(&jpeg_a, Unstoppable).unwrap();
    let b = decoder.decode(&jpeg_b, Unstoppable).unwrap();
    let c = decoder.decode(&jpeg_c, Unstoppable).unwrap();

    assert_eq!(a.dimensions(), (64, 64));
    assert_eq!(b.dimensions(), (64, 64));
    assert_eq!(c.dimensions(), (64, 64));
}
