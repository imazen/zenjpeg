//! # EncodeRequest API Guide
//!
//! Complete working examples demonstrating every method on [`EncodeRequest`].
//!
//! `EncodeRequest` is the intermediate layer between a reusable [`EncoderConfig`]
//! and a one-time encode. It binds per-image metadata (ICC, EXIF, XMP) and
//! controls (stop token, limits) without mutating the shared config.
//!
//! ## Pattern
//!
//! ```text
//! EncoderConfig (reusable)
//!     │
//!     ├── .request()              → EncodeRequest (per-image)
//!     │       ├── .icc_profile()      metadata
//!     │       ├── .exif()             metadata
//!     │       ├── .xmp()              metadata
//!     │       ├── .segments()         metadata round-trip
//!     │       ├── .stop()             cancellation token
//!     │       ├── .limits()           resource limits
//!     │       │
//!     │       ├── .encode()           one-shot (rgb types)
//!     │       ├── .encode_into()      one-shot into buffer
//!     │       ├── .encode_bytes()     one-shot (raw bytes)
//!     │       ├── .encode_bytes_into() one-shot into buffer
//!     │       │
//!     │       ├── .encode_from_rgb()     → RgbEncoder<P>  (streaming)
//!     │       ├── .encode_from_bytes()   → BytesEncoder    (streaming)
//!     │       └── .encode_from_ycbcr_planar() → YCbCrPlanarEncoder (streaming)
//!     │
//!     └── .encode() / .encode_from_*()   (direct, no request layer)
//! ```

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::generate_checkerboard;
use zenjpeg::encoder::{
    ChromaSubsampling, EncodeRequest, EncoderConfig, Exif, Orientation, PixelLayout, Unstoppable,
};
use zenjpeg::types::Limits;

// ============================================================================
// Test image helpers
// ============================================================================

/// Generate a 64x64 checkerboard as RGB<u8> pixels and raw bytes.
fn test_image() -> (Vec<rgb::RGB<u8>>, Vec<u8>, u32, u32) {
    let img = generate_checkerboard(64, 64, 8, 3);
    let w = img.width;
    let h = img.height;
    let raw = img.pixels.clone();
    // Reinterpret raw bytes as RGB<u8>
    let pixels: Vec<rgb::RGB<u8>> = raw
        .chunks_exact(3)
        .map(|c| rgb::RGB {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    (pixels, raw, w, h)
}

// ============================================================================
// 1. Creating a request
// ============================================================================

/// `EncoderConfig::request()` creates an `EncodeRequest` borrowing the config.
#[test]
fn create_request() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);

    // The request borrows the config — config stays usable.
    let _req: EncodeRequest<'_> = config.request();

    // Config is still available for more requests.
    let _req2: EncodeRequest<'_> = config.request();
}

// ============================================================================
// 2. One-shot encode from rgb crate types
// ============================================================================

/// `.encode()` consumes the request, encodes the full image, returns JPEG bytes.
#[test]
fn oneshot_encode_rgb() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let jpeg: Vec<u8> = config
        .request()
        .encode(&pixels, w, h)
        .expect("encode failed");

    assert!(jpeg.len() > 100);
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]); // SOI
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]); // EOI
}

// ============================================================================
// 3. One-shot encode into caller-provided buffer
// ============================================================================

/// `.encode_into()` writes JPEG into an existing Vec, avoiding allocation.
#[test]
fn oneshot_encode_into() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut output = Vec::with_capacity(8192);
    config
        .request()
        .encode_into(&pixels, w, h, &mut output)
        .expect("encode_into failed");

    assert!(!output.is_empty());
    assert_eq!(&output[..2], &[0xFF, 0xD8]);
}

// ============================================================================
// 4. One-shot encode from raw bytes
// ============================================================================

/// `.encode_bytes()` encodes from `&[u8]` with an explicit `PixelLayout`.
#[test]
fn oneshot_encode_bytes() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let jpeg = config
        .request()
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode_bytes failed");

    assert!(jpeg.len() > 100);
}

// ============================================================================
// 5. One-shot encode_bytes_into
// ============================================================================

/// `.encode_bytes_into()` writes raw-byte encode into a caller-provided buffer.
#[test]
fn oneshot_encode_bytes_into() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let mut output = Vec::new();
    config
        .request()
        .encode_bytes_into(&raw, w, h, PixelLayout::Rgb8Srgb, &mut output)
        .expect("encode_bytes_into failed");

    assert!(!output.is_empty());
    assert_eq!(&output[..2], &[0xFF, 0xD8]);
}

// ============================================================================
// 6. Streaming encode from rgb types
// ============================================================================

/// `.encode_from_rgb()` returns a streaming `RgbEncoder<P>`.
/// Push rows incrementally, then finish.
#[test]
fn streaming_encode_from_rgb() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .expect("encode_from_rgb failed");

    // Push all rows at once (could also push in batches)
    enc.push_packed(&pixels, Unstoppable).unwrap();

    assert_eq!(enc.rows_pushed(), h);
    assert_eq!(enc.rows_remaining(), 0);
    assert_eq!(enc.width(), w);
    assert_eq!(enc.height(), h);

    let jpeg = enc.finish().unwrap();
    assert!(jpeg.len() > 100);
}

// ============================================================================
// 7. Streaming encode from raw bytes
// ============================================================================

/// `.encode_from_bytes()` returns a streaming `BytesEncoder`.
#[test]
fn streaming_encode_from_bytes() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let mut enc = config
        .request()
        .encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)
        .expect("encode_from_bytes failed");

    enc.push_packed(&raw, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();
    assert!(jpeg.len() > 100);
}

// ============================================================================
// 8. Streaming encode with finish_into
// ============================================================================

/// Streaming encoders also support `finish_into()` to write into a buffer.
#[test]
fn streaming_finish_into() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();

    let mut output = Vec::new();
    enc.finish_into(&mut output).unwrap();
    assert!(!output.is_empty());
}

// ============================================================================
// 9. Streaming encode with finish_to (Write trait)
// ============================================================================

/// `finish_to()` writes to any `std::io::Write` implementor.
#[test]
fn streaming_finish_to() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();

    let cursor = std::io::Cursor::new(Vec::new());
    let cursor = enc.finish_to(cursor).unwrap();
    assert!(!cursor.into_inner().is_empty());
}

// ============================================================================
// 10. ICC profile (borrowed)
// ============================================================================

/// `.icc_profile()` borrows ICC data — zero-copy when the caller already owns it.
#[test]
fn icc_profile_borrowed() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let icc = b"fake-icc-for-test"; // would be real ICC bytes in production

    let jpeg = config
        .request()
        .icc_profile(icc.as_slice())
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Verify ICC_PROFILE marker is present
    assert!(jpeg.windows(12).any(|w| w == b"ICC_PROFILE\0".as_slice()));
}

// ============================================================================
// 11. ICC profile (owned)
// ============================================================================

/// `.icc_profile_owned()` takes ownership — use when the data is computed on the fly.
#[test]
fn icc_profile_owned() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let icc = vec![0u8; 32]; // dynamically created ICC data

    let jpeg = config
        .request()
        .icc_profile_owned(icc) // moved, not borrowed
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert!(jpeg.windows(12).any(|w| w == b"ICC_PROFILE\0".as_slice()));
}

// ============================================================================
// 12. EXIF metadata — field builder
// ============================================================================

/// `.exif()` with `Exif::build()` for structured EXIF construction.
#[test]
fn exif_from_fields() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let jpeg = config
        .request()
        .exif(
            Exif::build()
                .orientation(Orientation::Rotate90)
                .copyright("test copyright"),
        )
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert!(jpeg.windows(6).any(|w| w == b"Exif\0\0".as_slice()));
}

// ============================================================================
// 13. EXIF metadata — raw bytes
// ============================================================================

/// `.exif()` with `Exif::raw()` for pre-built EXIF data.
#[test]
fn exif_from_raw() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    // Minimal valid TIFF header (big-endian, IFD at offset 8, 0 entries)
    let tiff_data = vec![0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00];

    let jpeg = config
        .request()
        .exif(Exif::raw(tiff_data))
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert!(jpeg.windows(6).any(|w| w == b"Exif\0\0".as_slice()));
}

// ============================================================================
// 14. XMP metadata (borrowed)
// ============================================================================

/// `.xmp()` borrows XMP XML data.
#[test]
fn xmp_borrowed() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let xmp_xml = b"<x:xmpmeta><rdf:RDF></rdf:RDF></x:xmpmeta>";

    let jpeg = config
        .request()
        .xmp(xmp_xml.as_slice())
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // XMP is in APP1 with Adobe namespace
    assert!(
        jpeg.windows(29)
            .any(|w| w == b"http://ns.adobe.com/xap/1.0/\0".as_slice())
    );
}

// ============================================================================
// 15. XMP metadata (owned)
// ============================================================================

/// `.xmp_owned()` takes ownership of XMP data.
#[test]
fn xmp_owned() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let xmp_xml = b"<x:xmpmeta><rdf:RDF></rdf:RDF></x:xmpmeta>".to_vec();

    let jpeg = config
        .request()
        .xmp_owned(xmp_xml)
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert!(
        jpeg.windows(29)
            .any(|w| w == b"http://ns.adobe.com/xap/1.0/\0".as_slice())
    );
}

// ============================================================================
// 16. Combined metadata: ICC + EXIF + XMP
// ============================================================================

/// All metadata types can be chained on a single request.
#[test]
fn all_metadata_combined() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let icc = b"fake-icc";
    let xmp = b"<x:xmpmeta></x:xmpmeta>";

    let jpeg = config
        .request()
        .icc_profile(icc.as_slice())
        .exif(Exif::build().orientation(Orientation::Normal))
        .xmp(xmp.as_slice())
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert!(jpeg.windows(12).any(|w| w == b"ICC_PROFILE\0".as_slice()));
    assert!(jpeg.windows(6).any(|w| w == b"Exif\0\0".as_slice()));
    assert!(
        jpeg.windows(29)
            .any(|w| w == b"http://ns.adobe.com/xap/1.0/\0".as_slice())
    );
}

// ============================================================================
// 17. Stop token for cooperative cancellation
// ============================================================================

/// `.stop()` sets a cancellation token checked during encoding.
/// One-shot methods use it automatically.
#[test]
fn stop_token_cancellation() {
    use enough::{Stop, StopReason};

    struct AlwaysCancel;
    impl Stop for AlwaysCancel {
        fn check(&self) -> core::result::Result<(), StopReason> {
            Err(StopReason::Cancelled)
        }
    }

    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let cancel = AlwaysCancel;
    let result = config
        .request()
        .stop(&cancel)
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb);

    assert!(result.is_err());
}

// ============================================================================
// 18. Limits
// ============================================================================

/// `.limits()` sets resource limits (stored for future enforcement).
#[test]
fn resource_limits() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    let limits = Limits::default()
        .max_pixels(1_000_000)
        .max_memory(10_000_000);

    // Limits are accepted and stored; enforcement is not yet wired
    // to the streaming encoder, so this encode succeeds.
    let jpeg = config
        .request()
        .limits(limits)
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert!(!jpeg.is_empty());
}

// ============================================================================
// 19. Config reuse — same config, different metadata per image
// ============================================================================

/// The primary use case: one config, many images with varying metadata.
#[test]
fn config_reuse_pattern() {
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None).progressive(true);

    let (_, raw, w, h) = test_image();

    // Image 1: sRGB
    let jpeg_srgb = config
        .request()
        .icc_profile(b"srgb-icc".as_slice())
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Image 2: Display P3
    let jpeg_p3 = config
        .request()
        .icc_profile(b"p3-icc".as_slice())
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Image 3: no profile
    let jpeg_bare = config
        .request()
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // All different (different ICC profiles)
    assert_ne!(jpeg_srgb, jpeg_p3);
    assert_ne!(jpeg_srgb, jpeg_bare);
    assert_ne!(jpeg_p3, jpeg_bare);
}

// ============================================================================
// 20. Request does not mutate config — verify byte-identical bare encodes
// ============================================================================

/// Using `.request()` with metadata does not affect subsequent bare encodes.
#[test]
fn request_does_not_mutate_config() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (_, raw, w, h) = test_image();

    // Bare encode before
    let before = config
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Request with metadata (consumed)
    let _with_icc = config
        .request()
        .icc_profile(b"icc-data".as_slice())
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Bare encode after — must be identical to before
    let after = config
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    assert_eq!(before, after, "config must not be mutated by request");
}

// ============================================================================
// 21. Request one-shot matches config one-shot (no metadata)
// ============================================================================

/// Without metadata, request-based and config-based encodes are byte-identical.
#[test]
fn request_matches_direct_encode() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, raw, w, h) = test_image();

    // Via config
    let direct_bytes = config
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    let direct_rgb = config.encode(&pixels, w, h).unwrap();

    // Via request
    let req_bytes = config
        .request()
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    let req_rgb = config.request().encode(&pixels, w, h).unwrap();

    assert_eq!(direct_bytes, req_bytes);
    assert_eq!(direct_rgb, req_rgb);
}

// ============================================================================
// 22. Streaming via request matches streaming via config
// ============================================================================

/// Streaming encoders built from request produce identical output.
#[test]
fn streaming_request_matches_direct() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    // Direct
    let mut enc = config.encode_from_rgb::<rgb::RGB<u8>>(w, h).unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();
    let direct = enc.finish().unwrap();

    // Via request
    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();
    let via_req = enc.finish().unwrap();

    assert_eq!(direct, via_req);
}

// ============================================================================
// 23. Streaming with metadata via request
// ============================================================================

/// Streaming encoders from request inherit the request's metadata.
#[test]
fn streaming_with_metadata() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut enc = config
        .request()
        .icc_profile(b"test-icc".as_slice())
        .exif(Exif::build().orientation(Orientation::Rotate180))
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();

    enc.push_packed(&pixels, Unstoppable).unwrap();
    let jpeg = enc.finish().unwrap();

    assert!(jpeg.windows(12).any(|w| w == b"ICC_PROFILE\0".as_slice()));
    assert!(jpeg.windows(6).any(|w| w == b"Exif\0\0".as_slice()));
}

// ============================================================================
// 24. Streaming push with stride
// ============================================================================

/// `RgbEncoder::push()` supports strided row data (stride > width).
#[test]
fn streaming_push_with_stride() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let w: u32 = 60;
    let h: u32 = 64;
    let stride: usize = 64; // padded to 64 pixels per row

    // Create padded buffer
    let mut padded = vec![
        rgb::RGB {
            r: 128u8,
            g: 128,
            b: 128
        };
        stride * h as usize
    ];
    for y in 0..h as usize {
        for x in 0..w as usize {
            padded[y * stride + x] = rgb::RGB {
                r: (x * 4) as u8,
                g: (y * 4) as u8,
                b: 128,
            };
        }
    }

    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();

    // Push all rows with stride
    enc.push(&padded, h as usize, stride, Unstoppable).unwrap();

    let jpeg = enc.finish().unwrap();
    assert!(jpeg.len() > 100);
}

// ============================================================================
// 25. Encoder progress tracking
// ============================================================================

/// Streaming encoders expose progress via `rows_pushed()` and `rows_remaining()`.
#[test]
fn progress_tracking() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();

    assert_eq!(enc.rows_pushed(), 0);
    assert_eq!(enc.rows_remaining(), h);
    assert_eq!(enc.width(), w);
    assert_eq!(enc.height(), h);

    // Push half
    let half = h as usize / 2;
    let half_pixels = &pixels[..half * w as usize];
    enc.push_packed(half_pixels, Unstoppable).unwrap();

    assert_eq!(enc.rows_pushed(), half as u32);
    assert_eq!(enc.rows_remaining(), h - half as u32);

    // Push rest
    let rest_pixels = &pixels[half * w as usize..];
    enc.push_packed(rest_pixels, Unstoppable).unwrap();

    assert_eq!(enc.rows_pushed(), h);
    assert_eq!(enc.rows_remaining(), 0);

    let _jpeg = enc.finish().unwrap();
}

// ============================================================================
// 26. Encode stats
// ============================================================================

/// Streaming encoders expose allocation statistics via `encode_stats()`.
#[test]
fn encode_stats_accessible() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let (pixels, _, w, h) = test_image();

    let mut enc = config
        .request()
        .encode_from_rgb::<rgb::RGB<u8>>(w, h)
        .unwrap();
    enc.push_packed(&pixels, Unstoppable).unwrap();

    let stats = enc.encode_stats();
    // Stats should have recorded some allocations
    assert!(stats.summary().contains("alloc"));
}

// ============================================================================
// 27. Multiple pixel types
// ============================================================================

/// `encode_from_rgb` works with various pixel types.
#[test]
fn multiple_pixel_types() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);

    // RGB<u8>
    let pixels_u8: Vec<rgb::RGB<u8>> = (0..64 * 64)
        .map(|i| rgb::RGB {
            r: (i % 256) as u8,
            g: 128,
            b: 64,
        })
        .collect();
    let _jpeg = config.request().encode(&pixels_u8, 64, 64).unwrap();

    // RGBA<u8> (alpha ignored)
    let pixels_rgba: Vec<rgb::RGBA<u8>> = (0..64 * 64)
        .map(|i| rgb::RGBA {
            r: (i % 256) as u8,
            g: 128,
            b: 64,
            a: 255,
        })
        .collect();
    let _jpeg = config.request().encode(&pixels_rgba, 64, 64).unwrap();

    // Gray<u8> (with grayscale config)
    let gray_config = EncoderConfig::grayscale(85.0);
    let pixels_gray: Vec<rgb::Gray<u8>> =
        (0..64 * 64).map(|i| rgb::Gray((i % 256) as u8)).collect();
    let _jpeg = gray_config.request().encode(&pixels_gray, 64, 64).unwrap();
}

// ============================================================================
// 28. Multiple pixel layouts (raw bytes)
// ============================================================================

/// `encode_from_bytes` works with various `PixelLayout` values.
#[test]
fn multiple_pixel_layouts() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);

    // RGB 8-bit
    let rgb8 = vec![128u8; 64 * 64 * 3];
    let _jpeg = config
        .request()
        .encode_bytes(&rgb8, 64, 64, PixelLayout::Rgb8Srgb)
        .unwrap();

    // RGBX 8-bit (4 bytes/pixel, alpha ignored)
    let rgbx8 = vec![128u8; 64 * 64 * 4];
    let _jpeg = config
        .request()
        .encode_bytes(&rgbx8, 64, 64, PixelLayout::Rgbx8Srgb)
        .unwrap();

    // BGR 8-bit
    let bgr8 = vec![128u8; 64 * 64 * 3];
    let _jpeg = config
        .request()
        .encode_bytes(&bgr8, 64, 64, PixelLayout::Bgr8Srgb)
        .unwrap();

    // Grayscale
    let gray_config = EncoderConfig::grayscale(85.0);
    let gray8 = vec![128u8; 64 * 64];
    let _jpeg = gray_config
        .request()
        .encode_bytes(&gray8, 64, 64, PixelLayout::Gray8Srgb)
        .unwrap();
}

// ============================================================================
// 29. Grayscale config with request
// ============================================================================

/// Grayscale encoding works through the request layer.
#[test]
fn grayscale_via_request() {
    let config = EncoderConfig::grayscale(90.0);
    let gray: Vec<rgb::Gray<u8>> = (0..64 * 64).map(|i| rgb::Gray((i % 256) as u8)).collect();

    let jpeg = config.request().encode(&gray, 64, 64).unwrap();
    assert!(jpeg.len() > 50);
}

// ============================================================================
// 30. Progressive mode with request
// ============================================================================

/// Both progressive and baseline modes work through the request layer.
/// (Progressive is typically smaller for large images; at 64x64 the scan
/// header overhead can make it larger, so we just verify both succeed.)
#[test]
fn progressive_via_request() {
    let config_baseline = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
    let config_prog = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(true);

    let (_, raw, w, h) = test_image();

    let jpeg_baseline = config_baseline
        .request()
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();
    let jpeg_prog = config_prog
        .request()
        .encode_bytes(&raw, w, h, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Both produce valid JPEGs
    assert!(jpeg_baseline.len() > 50);
    assert!(jpeg_prog.len() > 50);
    // Output differs between modes
    assert_ne!(jpeg_baseline, jpeg_prog);
}
