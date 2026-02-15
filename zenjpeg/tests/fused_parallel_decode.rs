//! Fused parallel decode correctness tests.
//!
//! Verifies that the fused parallel decode path (IDCT during entropy decode)
//! produces identical output to the sequential decode path (coefficient buffer
//! → separate IDCT pass).
//!
//! Test strategy: encode images with MCU-row-aligned DRI, decode the same JPEG
//! data via fused path (multi-thread rayon pool) and sequential path (1-thread
//! pool where fused grouping produces < 4 segments, falling through to existing
//! parallel which runs sequentially in 1 thread). Compare byte-for-byte.
#![cfg(all(feature = "parallel", feature = "decoder"))]

use enough::Unstoppable;
use zenjpeg::decode::{ChromaUpsampling, Decoder};
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig};

/// Generate a test image with varied content (not smooth gradients).
/// Uses noise + patches pattern that exercises all DCT coefficients.
fn generate_test_pixels(width: u32, height: u32) -> Vec<rgb::RGB<u8>> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            // Mix of gradients, noise, and sharp edges
            let block_x = x / 8;
            let block_y = y / 8;
            let hash = ((block_x.wrapping_mul(17) ^ block_y.wrapping_mul(31)) as u8)
                .wrapping_add((x.wrapping_mul(7) ^ y.wrapping_mul(13)) as u8);

            let r = if x < width / 3 {
                (y * 255 / height) as u8
            } else {
                hash
            };
            let g = if y < height / 3 {
                (x * 255 / width) as u8
            } else {
                hash.wrapping_mul(3)
            };
            let b = ((x + y) * 127 / (width + height)) as u8;

            pixels.push(rgb::RGB { r, g, b });
        }
    }
    pixels
}

fn encode_with_dri(
    pixels: &[rgb::RGB<u8>],
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    restart_rows: u16,
) -> Vec<u8> {
    EncoderConfig::ycbcr(90.0, subsampling)
        .progressive(false) // baseline only
        .restart_mcu_rows(restart_rows)
        .encode(pixels, width, height)
        .expect("encode failed")
}

/// Decode with multi-thread pool (fused path should activate).
fn decode_fused(jpeg: &[u8], upsampling: ChromaUpsampling) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        Decoder::new()
            .output_format(PixelFormat::Rgb)
            .chroma_upsampling(upsampling)
            .decode(jpeg, Unstoppable)
            .expect("fused decode failed")
            .into_pixels_u8()
            .unwrap()
    })
}

/// Decode with single-thread pool (fused path should NOT activate due to
/// insufficient grouped segments, falling through to existing parallel/sequential).
fn decode_sequential(jpeg: &[u8], upsampling: ChromaUpsampling) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        Decoder::new()
            .output_format(PixelFormat::Rgb)
            .chroma_upsampling(upsampling)
            .decode(jpeg, Unstoppable)
            .expect("sequential decode failed")
            .into_pixels_u8()
            .unwrap()
    })
}

fn assert_pixels_equal(fused: &[u8], sequential: &[u8], label: &str) {
    assert_eq!(
        fused.len(),
        sequential.len(),
        "{}: pixel buffer size mismatch ({} vs {})",
        label,
        fused.len(),
        sequential.len()
    );
    let mut max_diff = 0u8;
    let mut diff_count = 0usize;
    let mut first_diff_idx = None;
    for (i, (&f, &s)) in fused.iter().zip(sequential.iter()).enumerate() {
        let d = f.abs_diff(s);
        if d > 0 {
            diff_count += 1;
            if d > max_diff {
                max_diff = d;
            }
            if first_diff_idx.is_none() {
                first_diff_idx = Some(i);
            }
        }
    }
    assert_eq!(
        diff_count,
        0,
        "{}: {} pixel differences (max_diff={}, first at byte {})",
        label,
        diff_count,
        max_diff,
        first_diff_idx.unwrap_or(0)
    );
}

// ============================================================================
// Test: 4:4:4 path (single-pass fused)
// ============================================================================

#[test]
fn test_fused_444_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::None, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:4:4 DRI=1row");
}

#[test]
fn test_fused_444_dri4() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::None, 4);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:4:4 DRI=4row");
}

// ============================================================================
// Test: 4:2:0 + NearestNeighbor (single-pass fused box filter)
// ============================================================================

#[test]
fn test_fused_420_nearest_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::NearestNeighbor);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::NearestNeighbor);

    assert_pixels_equal(&fused, &sequential, "4:2:0 NearestNeighbor DRI=1row");
}

// ============================================================================
// Test: 4:2:0 + Triangle (single-pass fused with extended chroma strips)
// ============================================================================

#[test]
fn test_fused_420_triangle_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:2:0 Triangle DRI=1row");
}

#[test]
fn test_fused_420_libjpeg_compat_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::LibjpegCompat);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::LibjpegCompat);

    assert_pixels_equal(&fused, &sequential, "4:2:0 LibjpegCompat DRI=1row");
}

// ============================================================================
// Test: Non-MCU-aligned dimensions
// ============================================================================

#[test]
fn test_fused_non_aligned_dimensions() {
    // 300x300: MCU grid for 4:2:0 is ceil(300/16)=19 MCUs wide, 19 tall
    // 19 MCU columns, DRI should be 19 per row
    let (w, h) = (300, 300);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:2:0 Triangle 300x300 (non-aligned)");
}

#[test]
fn test_fused_non_aligned_444() {
    // 333x257: MCU grid for 4:4:4 is ceil(333/8)=42, ceil(257/8)=33
    // Total MCUs = 42*33 = 1386 > 1024
    let (w, h) = (333, 257);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::None, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:4:4 333x257 (non-aligned)");
}

// ============================================================================
// Test: DRI=4 rows with 4:2:0 (fewer, larger segments)
// ============================================================================

#[test]
fn test_fused_420_dri4_triangle() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:2:0 Triangle DRI=4row");
}

// ============================================================================
// Test: Larger image (closer to real-world 1080p)
// ============================================================================

#[test]
fn test_fused_large_420() {
    let (w, h) = (1024, 768);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:2:0 Triangle 1024x768");
}

// ============================================================================
// Test: Non-aligned DRI falls back (no fused path)
// ============================================================================

#[test]
fn test_non_aligned_dri_fallback() {
    // Encode without restart markers — sequential decode only
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 0);

    // Both paths should decode identically (neither uses fused)
    let result1 = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let result2 = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&result1, &result2, "no DRI (sequential only)");
}

// ============================================================================
// Hash-lock tests: multi-size fused vs sequential byte-identical parity
//
// Tests at 256, 512, 1024, 2048 (MCU-aligned) and 513x513, 1000x1000
// (non-MCU-aligned) to catch edge cases in segment boundary handling,
// partial MCU rows, and chroma context.
// ============================================================================

/// Test fused vs sequential at multiple sizes with DRI=1.
/// This catches boundary fixup issues that only manifest at certain
/// segment counts or image dimensions.
#[test]
fn test_hashlock_multisize_triangle_dri1() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
        assert_pixels_equal(
            &fused,
            &sequential,
            &format!("4:2:0 Triangle DRI=1 {w}x{h}"),
        );
    }
}

/// Multi-size with DRI=4 (fewer, larger segments — different boundary pattern).
#[test]
fn test_hashlock_multisize_triangle_dri4() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (2048, 2048)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
        assert_pixels_equal(
            &fused,
            &sequential,
            &format!("4:2:0 Triangle DRI=4 {w}x{h}"),
        );
    }
}

/// Non-MCU-aligned dimensions test partial MCU rows at image edges.
/// 513 = 32*16 + 1 (1 pixel past MCU boundary for 4:2:0)
/// 1000 = 62*16 + 8 (half MCU past boundary)
#[test]
fn test_hashlock_non_mcu_aligned_triangle() {
    for (w, h) in [(513, 513), (1000, 1000), (257, 129), (1023, 767)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
        assert_pixels_equal(
            &fused,
            &sequential,
            &format!("4:2:0 Triangle non-aligned {w}x{h}"),
        );
    }
}

/// LibjpegCompat upsampling multi-size parity.
#[test]
fn test_hashlock_multisize_libjpeg_compat() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (513, 513)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::LibjpegCompat);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::LibjpegCompat);
        assert_pixels_equal(&fused, &sequential, &format!("4:2:0 LibjpegCompat {w}x{h}"));
    }
}

/// NearestNeighbor multi-size (box filter path, always single-pass).
#[test]
fn test_hashlock_multisize_nearest() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (513, 513)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::NearestNeighbor);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::NearestNeighbor);
        assert_pixels_equal(
            &fused,
            &sequential,
            &format!("4:2:0 NearestNeighbor {w}x{h}"),
        );
    }
}

// ============================================================================
// Test: 4:2:2 (h2v1) + fancy upsample
// ============================================================================

#[test]
fn test_fused_422_triangle_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:2:2 Triangle DRI=1row");
}

#[test]
fn test_fused_422_libjpeg_compat_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::LibjpegCompat);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::LibjpegCompat);

    assert_pixels_equal(&fused, &sequential, "4:2:2 LibjpegCompat DRI=1row");
}

#[test]
fn test_fused_422_nearest_correctness() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 1);

    let fused = decode_fused(&jpeg, ChromaUpsampling::NearestNeighbor);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::NearestNeighbor);

    assert_pixels_equal(&fused, &sequential, "4:2:2 NearestNeighbor DRI=1row");
}

#[test]
fn test_fused_422_dri4_triangle() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 4);

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

    assert_pixels_equal(&fused, &sequential, "4:2:2 Triangle DRI=4row");
}

/// 4:2:2 multi-size with all upsample modes.
#[test]
fn test_hashlock_multisize_422_triangle() {
    for (w, h) in [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (513, 513),
        (1000, 1000),
    ] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
        assert_pixels_equal(&fused, &sequential, &format!("4:2:2 Triangle {w}x{h}"));
    }
}

#[test]
fn test_hashlock_multisize_422_libjpeg_compat() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (513, 513)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::LibjpegCompat);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::LibjpegCompat);
        assert_pixels_equal(&fused, &sequential, &format!("4:2:2 LibjpegCompat {w}x{h}"));
    }
}

#[test]
fn test_hashlock_multisize_422_nearest() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (513, 513)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::HalfHorizontal, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::NearestNeighbor);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::NearestNeighbor);
        assert_pixels_equal(
            &fused,
            &sequential,
            &format!("4:2:2 NearestNeighbor {w}x{h}"),
        );
    }
}

/// 4:4:4 multi-size parity.
#[test]
fn test_hashlock_multisize_444() {
    for (w, h) in [(256, 256), (512, 512), (1024, 1024), (513, 513)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::None, 1);
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
        assert_pixels_equal(&fused, &sequential, &format!("4:4:4 {w}x{h}"));
    }
}
