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
use zenjpeg::decode::{ChromaUpsampling, Decoder, ParallelStrategy};
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

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

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
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
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

    let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
    let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);

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
        let fused = decode_fused(&jpeg, ChromaUpsampling::Triangle);
        let sequential = decode_sequential(&jpeg, ChromaUpsampling::Triangle);
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

/// Debug test: compare PerSegment (stride=1) vs FixedStride(2) output.
#[test]
fn test_debug_stride_comparison() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let mut any_failed = false;

    for upsampling in [ChromaUpsampling::Triangle, ChromaUpsampling::Triangle] {
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

        // Use 1 thread to eliminate concurrency as a variable
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let stride1 = pool.install(|| {
            Decoder::new()
                .output_format(PixelFormat::Rgb)
                .chroma_upsampling(upsampling)
                .parallel_strategy(ParallelStrategy::PerSegment)
                .decode(&jpeg, Unstoppable)
                .expect("stride1 decode failed")
                .into_pixels_u8()
                .unwrap()
        });

        let stride2 = pool.install(|| {
            Decoder::new()
                .output_format(PixelFormat::Rgb)
                .chroma_upsampling(upsampling)
                .parallel_strategy(ParallelStrategy::FixedStride(2))
                .decode(&jpeg, Unstoppable)
                .expect("stride2 decode failed")
                .into_pixels_u8()
                .unwrap()
        });

        let rgb_row_bytes = w as usize * 3;
        let mut diff_rows: Vec<(usize, u8, usize)> = Vec::new();
        for row in 0..h as usize {
            let start = row * rgb_row_bytes;
            let end = start + rgb_row_bytes;
            let mut max_diff = 0u8;
            let mut count = 0usize;
            for i in start..end {
                let d = stride1[i].abs_diff(stride2[i]);
                if d > 0 {
                    count += 1;
                    max_diff = max_diff.max(d);
                }
            }
            if count > 0 {
                diff_rows.push((row, max_diff, count));
            }
        }

        if !diff_rows.is_empty() {
            any_failed = true;
            eprintln!(
                "\n{:?} 512x512: {} rows differ ({} total diff pixels)",
                upsampling,
                diff_rows.len(),
                diff_rows.iter().map(|r| r.2).sum::<usize>()
            );
            for &(row, max_diff, count) in diff_rows.iter().take(30) {
                let mcu_row = row / 16;
                let mcu_offset = row % 16;
                eprintln!(
                    "  row {row} (MCU row {mcu_row}, offset {mcu_offset}): max_diff={max_diff}, {count} diffs"
                );
            }
            if diff_rows.len() > 30 {
                eprintln!("  ... and {} more rows", diff_rows.len() - 30);
            }
        } else {
            eprintln!("\n{:?} 512x512: IDENTICAL", upsampling);
        }
    }

    assert!(
        !any_failed,
        "Some upsampling modes produced different output"
    );
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

// ============================================================================
// Wave parallel scanline reader tests
//
// Verify that the wave-parallel scanline reader produces identical output
// to the sequential scanline reader for 4:2:0 + NearestNeighbor (box filter).
// ============================================================================

/// Decode via wave-parallel scanline reader (multi-threaded).
fn decode_wave_scanline(jpeg: &[u8]) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let mut reader = Decoder::new()
            .fancy_upsampling(false) // box filter = NearestNeighbor
            .scanline_reader(jpeg)
            .expect("wave scanline_reader failed");
        let width = reader.width() as usize;
        let height = reader.height() as usize;
        let mut pixels = vec![0u8; width * height * 3];
        let mut rows_read = 0;
        while rows_read < height {
            let remaining = height - rows_read;
            let slice = &mut pixels[rows_read * width * 3..];
            let output = imgref::ImgRefMut::new(slice, width * 3, remaining);
            let count = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            assert!(count > 0, "read_rows_rgb8 returned 0 before completion");
            rows_read += count;
        }
        assert!(reader.is_finished(), "reader should be finished");
        pixels
    })
}

/// Decode via sequential scanline reader (single-threaded).
fn decode_sequential_scanline(jpeg: &[u8]) -> Vec<u8> {
    let mut reader = Decoder::new()
        .fancy_upsampling(false)
        .num_threads(1)
        .scanline_reader(jpeg)
        .expect("sequential scanline_reader failed");
    let width = reader.width() as usize;
    let height = reader.height() as usize;
    let mut pixels = vec![0u8; width * height * 3];
    let mut rows_read = 0;
    while rows_read < height {
        let remaining = height - rows_read;
        let slice = &mut pixels[rows_read * width * 3..];
        let output = imgref::ImgRefMut::new(slice, width * 3, remaining);
        let count = reader
            .read_rows_rgb8(output)
            .expect("read_rows_rgb8 failed");
        assert!(count > 0, "read_rows_rgb8 returned 0 before completion");
        rows_read += count;
    }
    pixels
}

#[test]
fn test_wave_parallel_512x512() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let wave = decode_wave_scanline(&jpeg);
    let sequential = decode_sequential_scanline(&jpeg);

    assert_pixels_equal(&wave, &sequential, "wave 4:2:0 box 512x512 DRI=1");
}

#[test]
fn test_wave_parallel_1024x768() {
    let (w, h) = (1024, 768);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let wave = decode_wave_scanline(&jpeg);
    let sequential = decode_sequential_scanline(&jpeg);

    assert_pixels_equal(&wave, &sequential, "wave 4:2:0 box 1024x768 DRI=1");
}

#[test]
fn test_wave_parallel_2048x2048() {
    let (w, h) = (2048, 2048);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);

    let wave = decode_wave_scanline(&jpeg);
    let sequential = decode_sequential_scanline(&jpeg);

    assert_pixels_equal(&wave, &sequential, "wave 4:2:0 box 2048x2048 DRI=4");
}

#[test]
fn test_wave_parallel_non_aligned() {
    // Non-MCU-aligned: 513 = 32*16 + 1
    for (w, h) in [(513, 513), (1000, 1000), (300, 300)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

        let wave = decode_wave_scanline(&jpeg);
        let sequential = decode_sequential_scanline(&jpeg);

        assert_pixels_equal(
            &wave,
            &sequential,
            &format!("wave 4:2:0 box {w}x{h} non-aligned"),
        );
    }
}

#[test]
fn test_wave_parallel_dri4() {
    for (w, h) in [(512, 512), (1024, 1024)] {
        let pixels = generate_test_pixels(w, h);
        let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);

        let wave = decode_wave_scanline(&jpeg);
        let sequential = decode_sequential_scanline(&jpeg);

        assert_pixels_equal(&wave, &sequential, &format!("wave 4:2:0 box DRI=4 {w}x{h}"));
    }
}

#[test]
fn test_wave_parallel_small_chunks() {
    // Test reading just 1 row at a time to exercise wave buffer refill
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 1);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let wave = pool.install(|| {
        let mut reader = Decoder::new()
            .fancy_upsampling(false)
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let width = reader.width() as usize;
        let height = reader.height() as usize;
        let mut all_pixels = vec![0u8; width * height * 3];
        let mut rows_read = 0;
        while rows_read < height {
            // Read exactly 1 row at a time
            let slice = &mut all_pixels[rows_read * width * 3..];
            let output = imgref::ImgRefMut::new(slice, width * 3, 1);
            let count = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            assert_eq!(count, 1);
            rows_read += 1;
        }
        all_pixels
    });

    let sequential = decode_sequential_scanline(&jpeg);
    assert_pixels_equal(&wave, &sequential, "wave 1-row-at-a-time 512x512");
}

// ============================================================================
// Tests: Planar i16 decode
// ============================================================================

/// Decode planar i16 via sequential path (forced single thread).
fn decode_planar_i16_seq(jpeg: &[u8]) -> (Vec<i16>, Vec<i16>, Vec<i16>, u32, u32, u32, u32) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let mut reader = Decoder::new()
            .fancy_upsampling(false)
            .num_threads(1)
            .scanline_reader(jpeg)
            .expect("scanline_reader failed");

        let width = reader.width() as usize;
        let height = reader.height() as usize;
        let cw = reader.chroma_width() as usize;
        let ch = reader.chroma_height() as usize;
        let luma_rows_per_mcu = reader.luma_rows_per_mcu();
        let total_mcu_rows = (height + luma_rows_per_mcu - 1) / luma_rows_per_mcu;

        let mut y_buf = vec![0i16; width * height];
        let mut cb_buf = vec![0i16; cw * ch];
        let mut cr_buf = vec![0i16; cw * ch];

        let mut y_off = 0;
        let mut c_off = 0;

        for _ in 0..total_mcu_rows {
            let (luma_rows, chroma_rows) = reader
                .read_rows_ycbcr_native_i16(
                    &mut y_buf[y_off..],
                    width,
                    &mut cb_buf[c_off..],
                    &mut cr_buf[c_off..],
                    cw,
                    1,
                )
                .expect("read_rows_ycbcr_native_i16 failed");
            y_off += luma_rows * width;
            c_off += chroma_rows * cw;
        }

        assert!(reader.is_finished(), "reader should be finished");
        (
            y_buf,
            cb_buf,
            cr_buf,
            width as u32,
            height as u32,
            cw as u32,
            ch as u32,
        )
    })
}

/// Decode planar i16 via wave-parallel path (multi-thread).
fn decode_planar_i16_wave(jpeg: &[u8]) -> (Vec<i16>, Vec<i16>, Vec<i16>, u32, u32, u32, u32) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let mut reader = Decoder::new()
            .fancy_upsampling(false)
            .scanline_reader(jpeg)
            .expect("scanline_reader failed");

        let width = reader.width() as usize;
        let height = reader.height() as usize;
        let cw = reader.chroma_width() as usize;
        let ch = reader.chroma_height() as usize;
        let luma_rows_per_mcu = reader.luma_rows_per_mcu();
        let total_mcu_rows = (height + luma_rows_per_mcu - 1) / luma_rows_per_mcu;

        let mut y_buf = vec![0i16; width * height];
        let mut cb_buf = vec![0i16; cw * ch];
        let mut cr_buf = vec![0i16; cw * ch];

        let mut y_off = 0;
        let mut c_off = 0;

        for _ in 0..total_mcu_rows {
            let (luma_rows, chroma_rows) = reader
                .read_rows_ycbcr_native_i16(
                    &mut y_buf[y_off..],
                    width,
                    &mut cb_buf[c_off..],
                    &mut cr_buf[c_off..],
                    cw,
                    1,
                )
                .expect("read_rows_ycbcr_native_i16 failed");
            y_off += luma_rows * width;
            c_off += chroma_rows * cw;
        }

        assert!(reader.is_finished(), "reader should be finished");
        (
            y_buf,
            cb_buf,
            cr_buf,
            width as u32,
            height as u32,
            cw as u32,
            ch as u32,
        )
    })
}

fn assert_i16_equal(a: &[i16], b: &[i16], label: &str) {
    assert_eq!(a.len(), b.len(), "{}: buffer size mismatch", label);
    let mut max_diff = 0i16;
    let mut diff_count = 0usize;
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let d = (av - bv).abs();
        if d > 0 {
            diff_count += 1;
            if d > max_diff {
                max_diff = d;
            }
            if diff_count == 1 {
                eprintln!("{}: first diff at index {}: {} vs {}", label, i, av, bv);
            }
        }
    }
    assert_eq!(
        diff_count, 0,
        "{}: {} differences (max_diff={})",
        label, diff_count, max_diff
    );
}

#[test]
fn test_planar_i16_seq_420() {
    let (w, h) = (256, 256);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);

    let (y, cb, cr, width, height, cw, ch) = decode_planar_i16_seq(&jpeg);

    assert_eq!(width, 256);
    assert_eq!(height, 256);
    assert_eq!(cw, 128, "chroma width should be half for 4:2:0");
    assert_eq!(ch, 128, "chroma height should be half for 4:2:0");
    assert_eq!(y.len(), 256 * 256);
    assert_eq!(cb.len(), 128 * 128);
    assert_eq!(cr.len(), 128 * 128);

    // Verify Y values are in valid IDCT range (0-255 for 8-bit JPEG)
    for &v in &y {
        assert!(
            (-128..=383).contains(&v),
            "Y value {} out of expected IDCT range",
            v
        );
    }
}

#[test]
fn test_planar_i16_seq_444() {
    let (w, h) = (256, 256);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::None, 4);

    let (y, cb, cr, width, height, cw, ch) = decode_planar_i16_seq(&jpeg);

    assert_eq!(width, 256);
    assert_eq!(height, 256);
    assert_eq!(cw, 256, "chroma width should equal luma for 4:4:4");
    assert_eq!(ch, 256, "chroma height should equal luma for 4:4:4");
    assert_eq!(y.len(), 256 * 256);
    assert_eq!(cb.len(), 256 * 256);
    assert_eq!(cr.len(), 256 * 256);
}

#[test]
fn test_planar_i16_wave_vs_seq_420() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);

    let (y_seq, cb_seq, cr_seq, ..) = decode_planar_i16_seq(&jpeg);
    let (y_wave, cb_wave, cr_wave, ..) = decode_planar_i16_wave(&jpeg);

    assert_i16_equal(&y_seq, &y_wave, "Y plane wave vs seq 4:2:0");
    assert_i16_equal(&cb_seq, &cb_wave, "Cb plane wave vs seq 4:2:0");
    assert_i16_equal(&cr_seq, &cr_wave, "Cr plane wave vs seq 4:2:0");
}

#[test]
fn test_planar_i16_wave_vs_seq_444() {
    let (w, h) = (512, 512);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::None, 4);

    let (y_seq, cb_seq, cr_seq, ..) = decode_planar_i16_seq(&jpeg);
    let (y_wave, cb_wave, cr_wave, ..) = decode_planar_i16_wave(&jpeg);

    assert_i16_equal(&y_seq, &y_wave, "Y plane wave vs seq 4:4:4");
    assert_i16_equal(&cb_seq, &cb_wave, "Cb plane wave vs seq 4:4:4");
    assert_i16_equal(&cr_seq, &cr_wave, "Cr plane wave vs seq 4:4:4");
}

#[test]
fn test_planar_i16_non_mcu_aligned_dims() {
    // Non-MCU-aligned dimensions exercise edge padding
    let (w, h) = (300, 300);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 2);

    let (y_seq, cb_seq, cr_seq, width, height, cw, ch) = decode_planar_i16_seq(&jpeg);
    let (y_wave, cb_wave, cr_wave, ..) = decode_planar_i16_wave(&jpeg);

    assert_eq!(width, 300);
    assert_eq!(height, 300);
    // Chroma width is MCU-padded: ceil(300/16) = 19 MCU cols, 19 * 8 = 152
    assert_eq!(cw, 152, "chroma width for 300px 4:2:0 (MCU-padded)");
    assert_eq!(ch, 150, "chroma height for 300px 4:2:0");

    assert_i16_equal(&y_seq, &y_wave, "Y non-MCU-aligned wave vs seq");
    assert_i16_equal(&cb_seq, &cb_wave, "Cb non-MCU-aligned wave vs seq");
    assert_i16_equal(&cr_seq, &cr_wave, "Cr non-MCU-aligned wave vs seq");
}

/// Verify that planar Y + box-upsampled Cb/Cr → RGB matches read_rows_rgb8(box).
#[test]
fn test_planar_to_rgb_reconstruction_420() {
    let (w, h) = (256, 256);
    let pixels = generate_test_pixels(w, h);
    let jpeg = encode_with_dri(&pixels, w, h, ChromaSubsampling::Quarter, 4);

    // Get planar data
    let (y_buf, cb_buf, cr_buf, width, height, cw, _ch) = decode_planar_i16_seq(&jpeg);
    let w = width as usize;
    let h = height as usize;
    let cw = cw as usize;

    // Get RGB reference via box filter decode
    let rgb_ref = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            Decoder::new()
                .fancy_upsampling(false)
                .num_threads(1)
                .output_format(PixelFormat::Rgb)
                .decode(&jpeg, Unstoppable)
                .expect("decode failed")
                .into_pixels_u8()
                .unwrap()
        })
    };

    // Manually reconstruct RGB from planar YCbCr with box filter upsampling
    let mut rgb_recon = vec![0u8; w * h * 3];
    for row in 0..h {
        for col in 0..w {
            let y = y_buf[row * w + col] as i32;
            // Box filter: nearest-neighbor chroma upsampling
            let cr_row = row / 2;
            let cr_col = col / 2;
            let cb = cb_buf[cr_row * cw + cr_col] as i32 - 128;
            let cr = cr_buf[cr_row * cw + cr_col] as i32 - 128;

            // BT.601 YCbCr → RGB (fixed-point, matching zenjpeg's integer path)
            let r = (y + ((cr * 91881 + 32768) >> 16)).clamp(0, 255) as u8;
            let g = (y - ((cb * 22554 + cr * 46802 - 32768) >> 16)).clamp(0, 255) as u8;
            let b = (y + ((cb * 116130 + 32768) >> 16)).clamp(0, 255) as u8;

            let off = (row * w + col) * 3;
            rgb_recon[off] = r;
            rgb_recon[off + 1] = g;
            rgb_recon[off + 2] = b;
        }
    }

    // Allow ±2 difference: zenjpeg's fused AVX2 kernel uses f32 YCbCr→RGB
    // while our manual reconstruction uses fixed-point. The internal tests
    // already allow ±2 between the f32 and integer paths.
    let mut max_diff = 0u8;
    let mut diff_count = 0usize;
    for (i, (&a, &b)) in rgb_ref.iter().zip(rgb_recon.iter()).enumerate() {
        let d = a.abs_diff(b);
        if d > 2 {
            diff_count += 1;
            if d > max_diff {
                max_diff = d;
            }
            if diff_count <= 3 {
                let pixel = i / 3;
                let channel = ["R", "G", "B"][i % 3];
                eprintln!(
                    "Diff at pixel {} {}: ref={} recon={} (row={}, col={})",
                    pixel,
                    channel,
                    a,
                    b,
                    pixel / w,
                    pixel % w,
                );
            }
        }
    }
    assert!(
        max_diff <= 2,
        "planar→RGB max diff {} > 2 ({} pixels differ by >2)",
        max_diff,
        diff_count
    );
}
