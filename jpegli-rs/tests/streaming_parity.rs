//! Test that streaming encoder produces identical output to standard encoder.

use jpegli::{Encoder, JpegMode, Quality, StreamingEncoder, Subsampling};

#[test]
fn test_streaming_matches_standard_various_sizes() {
    let tests = [
        (64, 64, Subsampling::S444),
        (100, 100, Subsampling::S420),
        (256, 256, Subsampling::S420),
        (640, 480, Subsampling::S420),
        (640, 480, Subsampling::S444),
        (640, 480, Subsampling::S422),
        // Non-aligned dimensions
        (123, 87, Subsampling::S420),
        (111, 113, Subsampling::S444),
    ];

    for (width, height, subsampling) in tests {
        // Generate test pixels
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17 + i / 256) % 256) as u8)
            .collect();

        // Encode with standard encoder using strip backend
        #[allow(deprecated)]
        let standard = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(85.0))
            .subsampling(subsampling)
            .encode(&pixels)
            .expect("standard encode failed");

        // Encode with streaming encoder
        let mut streaming = StreamingEncoder::new(width, height)
            .quality(Quality::from_quality(85.0))
            .subsampling(subsampling)
            .build()
            .expect("streaming build failed");

        let row_size = width as usize * 3;
        for y in 0..height as usize {
            let start = y * row_size;
            let end = start + row_size;
            streaming
                .push_row(&pixels[start..end])
                .expect("push_row failed");
        }
        let streaming_result = streaming.finish().expect("finish failed");

        // Compare
        assert_eq!(
            standard.len(),
            streaming_result.len(),
            "{}×{} {:?}: length mismatch (standard={}, streaming={})",
            width,
            height,
            subsampling,
            standard.len(),
            streaming_result.len()
        );
        assert_eq!(
            standard, streaming_result,
            "{}×{} {:?}: content mismatch",
            width, height, subsampling
        );

        println!(
            "✓ {}×{} {:?}: identical ({} bytes)",
            width,
            height,
            subsampling,
            standard.len()
        );
    }
}

#[test]
fn test_streaming_push_rows() {
    let width = 64u32;
    let height = 64u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 17) % 256) as u8)
        .collect();

    // Encode with streaming encoder using push_rows
    let mut streaming = StreamingEncoder::new(width, height)
        .quality(Quality::from_quality(85.0))
        .subsampling(Subsampling::S444)
        .build()
        .unwrap();

    // Push 4 rows at a time
    let row_size = width as usize * 3;
    let chunk_rows = 4;
    let chunk_size = row_size * chunk_rows;
    for chunk_idx in 0..(height as usize / chunk_rows) {
        let start = chunk_idx * chunk_size;
        let end = start + chunk_size;
        streaming
            .push_rows(&pixels[start..end], chunk_rows)
            .unwrap();
    }

    let result = streaming.finish().unwrap();
    assert!(!result.is_empty());
    assert_eq!(result[0..2], [0xFF, 0xD8]); // JPEG SOI marker
}

#[test]
fn test_memory_estimate_reasonable() {
    // 4K image
    let estimate = StreamingEncoder::new(3840, 2160)
        .subsampling(Subsampling::S420)
        .estimate_memory_usage();

    // Should be around 26 MB for 4K with 4:2:0
    let estimate_mb = estimate as f64 / 1024.0 / 1024.0;
    println!("4K 4:2:0 estimate: {:.2} MB", estimate_mb);
    assert!(estimate_mb > 20.0, "estimate {:.2} MB too low", estimate_mb);
    assert!(
        estimate_mb < 40.0,
        "estimate {:.2} MB too high",
        estimate_mb
    );

    // 1080p image
    let estimate_1080p = StreamingEncoder::new(1920, 1080)
        .subsampling(Subsampling::S420)
        .estimate_memory_usage();
    let estimate_1080p_mb = estimate_1080p as f64 / 1024.0 / 1024.0;
    println!("1080p 4:2:0 estimate: {:.2} MB", estimate_1080p_mb);

    // 1080p should be about 1/4 of 4K
    let ratio = estimate as f64 / estimate_1080p as f64;
    println!("4K/1080p ratio: {:.2}x", ratio);
    assert!(ratio > 3.5 && ratio < 4.5, "unexpected ratio {:.2}x", ratio);
}

#[test]
fn test_streaming_progressive_mode() {
    // Test that progressive mode works with streaming encoder
    let tests = [
        (256, 256, Subsampling::S420),
        (640, 480, Subsampling::S420),
        (640, 480, Subsampling::S444),
        (123, 87, Subsampling::S420),
    ];

    for (width, height, subsampling) in tests {
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17 + i / 256) % 256) as u8)
            .collect();

        // Encode with standard encoder (strip backend, progressive)
        #[allow(deprecated)]
        let standard = Encoder::new()
            .width(width)
            .height(height)
            .quality(Quality::from_quality(85.0))
            .subsampling(subsampling)
            .mode(JpegMode::Progressive)
            .encode(&pixels)
            .expect("standard progressive encode failed");

        // Encode with streaming encoder (progressive)
        let streaming_result = StreamingEncoder::new(width, height)
            .quality(Quality::from_quality(85.0))
            .subsampling(subsampling)
            .mode(JpegMode::Progressive)
            .encode_all(&pixels)
            .expect("streaming progressive encode failed");

        // Compare
        assert_eq!(
            standard.len(),
            streaming_result.len(),
            "Progressive {}×{} {:?}: length mismatch (standard={}, streaming={})",
            width,
            height,
            subsampling,
            standard.len(),
            streaming_result.len()
        );
        assert_eq!(
            standard, streaming_result,
            "Progressive {}×{} {:?}: content mismatch",
            width, height, subsampling
        );

        // Verify it's actually progressive (SOF2 marker = 0xFFC2)
        let has_sof2 = standard.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
        assert!(
            has_sof2,
            "Progressive {}×{} {:?}: missing SOF2 marker",
            width, height, subsampling
        );

        println!(
            "✓ Progressive {}×{} {:?}: identical ({} bytes)",
            width,
            height,
            subsampling,
            standard.len()
        );
    }
}

#[test]
fn test_streaming_encode_all() {
    // Test the encode_all convenience method
    let width = 320u32;
    let height = 240u32;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 17) % 256) as u8)
        .collect();

    // Using encode_all
    let encode_all_result = StreamingEncoder::new(width, height)
        .quality(Quality::from_quality(85.0))
        .subsampling(Subsampling::S420)
        .encode_all(&pixels)
        .unwrap();

    // Using build + push_row + finish
    let mut streaming = StreamingEncoder::new(width, height)
        .quality(Quality::from_quality(85.0))
        .subsampling(Subsampling::S420)
        .build()
        .unwrap();

    let row_size = width as usize * 3;
    for y in 0..height as usize {
        let start = y * row_size;
        streaming
            .push_row(&pixels[start..start + row_size])
            .unwrap();
    }
    let manual_result = streaming.finish().unwrap();

    // Should be identical
    assert_eq!(encode_all_result, manual_result);
    println!(
        "✓ encode_all matches manual push_row ({} bytes)",
        encode_all_result.len()
    );
}
