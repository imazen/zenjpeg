//! Chroma subsampling tests.
//!
//! Tests for various chroma subsampling modes and their effect on
//! image quality and file size.

#[path = "../src/test_utils.rs"]
mod test_utils;

use enough::Never;
use test_utils::{
    distance_rms, generate_color_bars, generate_gradient_d, max_pixel_diff, read_test_data,
    TestImage,
};

use jpegli::{ChromaSubsampling, Decoder, EncoderConfig, PixelLayout};

// ============================================================================
// Helper Functions
// ============================================================================

fn encode_rgb(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(data, Never).expect("push failed");
    enc.finish().expect("finish failed")
}

fn encode_gray(width: u32, height: u32, data: &[u8], config: &EncoderConfig) -> Vec<u8> {
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encoder creation failed");
    enc.push_packed(data, Never).expect("push failed");
    enc.finish().expect("finish failed")
}

fn roundtrip_with_subsampling(
    img: &TestImage,
    quality: f32,
    subsampling: ChromaSubsampling,
) -> (f64, u8, usize) {
    let config = EncoderConfig::new()
        .quality(quality)
        .ycbcr(subsampling);
    let jpeg = encode_rgb(img.width, img.height, &img.pixels, &config);

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    let rms = distance_rms(&img.pixels, &decoded.data);
    let max_diff = max_pixel_diff(&img.pixels, &decoded.data);

    (rms, max_diff, jpeg.len())
}

// ============================================================================
// Basic Subsampling Tests
// ============================================================================

#[test]
fn test_444_subsampling() {
    let img = generate_gradient_d(256, 256, 3);
    let (rms, max_diff, size) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::Full);

    println!(
        "4:4:4: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, size
    );

    // 4:4:4 should have best quality (no chroma subsampling)
    assert!(rms < 5.0, "4:4:4 RMS too high");
}

#[test]
fn test_422_subsampling() {
    let img = generate_gradient_d(256, 256, 3);
    let (rms, max_diff, size) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::HalfHorizontal);

    println!(
        "4:2:2: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, size
    );
    assert!(rms < 6.0, "4:2:2 RMS too high: {}", rms);
}

#[test]
fn test_420_subsampling() {
    let img = generate_gradient_d(256, 256, 3);
    let (rms, max_diff, size) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::Quarter);

    println!(
        "4:2:0: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, size
    );
    assert!(rms < 7.0, "4:2:0 RMS too high: {}", rms);
}

// ============================================================================
// Subsampling Comparison Tests
// ============================================================================

#[test]
fn test_subsampling_quality_size_tradeoff() {
    let img = generate_gradient_d(256, 256, 3);

    // Currently only test 4:4:4
    let (rms_444, _, size_444) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::Full);

    println!("Subsampling comparison:");
    println!("  4:4:4: RMS={:.2}, size={}", rms_444, size_444);

    // With only 4:4:4 available, just verify it works
    assert!(rms_444 < 10.0, "4:4:4 quality should be good");
}

// ============================================================================
// Decode C++ Subsampled JPEGs
// ============================================================================

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_420() {
    if let Some(jpeg_data) = read_test_data("jxl/flower/flower.png.im_q85_420.jpg") {
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg_data).expect("decode 420 failed");

        println!(
            "Decoded 4:2:0: {}x{}, {} bytes",
            decoded.width,
            decoded.height,
            decoded.data.len()
        );

        assert_eq!(decoded.width, 2268);
        assert_eq!(decoded.height, 1512);
    }
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_422() {
    if let Some(jpeg_data) = read_test_data("jxl/flower/flower.png.im_q85_422.jpg") {
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg_data).expect("decode 422 failed");

        println!(
            "Decoded 4:2:2: {}x{}, {} bytes",
            decoded.width,
            decoded.height,
            decoded.data.len()
        );

        assert_eq!(decoded.width, 2268);
        assert_eq!(decoded.height, 1512);
    }
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_440() {
    if let Some(jpeg_data) = read_test_data("jxl/flower/flower.png.im_q85_440.jpg") {
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg_data).expect("decode 440 failed");

        println!(
            "Decoded 4:4:0: {}x{}, {} bytes",
            decoded.width,
            decoded.height,
            decoded.data.len()
        );

        assert_eq!(decoded.width, 2268);
        assert_eq!(decoded.height, 1512);
    }
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_444() {
    if let Some(jpeg_data) = read_test_data("jxl/flower/flower.png.im_q85_444.jpg") {
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg_data).expect("decode 444 failed");

        println!(
            "Decoded 4:4:4: {}x{}, {} bytes",
            decoded.width,
            decoded.height,
            decoded.data.len()
        );

        assert_eq!(decoded.width, 2268);
        assert_eq!(decoded.height, 1512);
    }
}

// ============================================================================
// Asymmetric Subsampling Tests
// ============================================================================

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_asymmetric() {
    if let Some(jpeg_data) = read_test_data("jxl/flower/flower.png.im_q85_asymmetric.jpg") {
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg_data)
            .expect("decode asymmetric failed");

        println!(
            "Decoded asymmetric: {}x{}, {} bytes",
            decoded.width,
            decoded.height,
            decoded.data.len()
        );

        assert_eq!(decoded.width, 2268);
        assert_eq!(decoded.height, 1512);
    }
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_444_1x2() {
    if let Some(jpeg_data) = read_test_data("jxl/flower/flower.png.im_q85_444_1x2.jpg") {
        let decoder = Decoder::new();
        let decoded = decoder.decode(&jpeg_data).expect("decode 444_1x2 failed");

        println!(
            "Decoded 4:4:4 1x2: {}x{}, {} bytes",
            decoded.width,
            decoded.height,
            decoded.data.len()
        );

        assert_eq!(decoded.width, 2268);
        assert_eq!(decoded.height, 1512);
    }
}

// ============================================================================
// Color Content Tests
// ============================================================================

#[test]
fn test_color_bars_subsampling() {
    // Color bars have sharp color transitions - sensitive to subsampling
    let img = generate_color_bars(256, 128);
    let (rms, max_diff, size) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::Full);

    println!(
        "Color bars 4:4:4: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, size
    );

    // Color bars with 4:4:4 should preserve colors well
    assert!(rms < 10.0, "Color bars RMS too high");
}

#[test]
fn test_saturated_colors_subsampling() {
    // Test with saturated colors (red, green, blue, yellow, cyan, magenta)
    let colors = [
        (255, 0, 0),   // Red
        (0, 255, 0),   // Green
        (0, 0, 255),   // Blue
        (255, 255, 0), // Yellow
        (0, 255, 255), // Cyan
        (255, 0, 255), // Magenta
    ];

    for (r, g, b) in colors {
        let mut img = TestImage::new(64, 64, 3);
        for y in 0..64 {
            for x in 0..64 {
                img.set_pixel(x, y, 0, r);
                img.set_pixel(x, y, 1, g);
                img.set_pixel(x, y, 2, b);
            }
        }

        let (rms, max_diff, _size) = roundtrip_with_subsampling(&img, 95.0, ChromaSubsampling::Full);

        assert!(
            rms < 5.0,
            "Saturated ({},{},{}) RMS {:.2} too high",
            r,
            g,
            b,
            rms
        );
        assert!(
            max_diff < 15,
            "Saturated ({},{},{}) max_diff {} too high",
            r,
            g,
            b,
            max_diff
        );
    }
}

// ============================================================================
// Edge Detection for Subsampling Artifacts
// ============================================================================

#[test]
fn test_color_edge_444() {
    // Create image with sharp color edge (red on left, blue on right)
    let mut img = TestImage::new(128, 128, 3);
    for y in 0..128 {
        for x in 0..128 {
            if x < 64 {
                img.set_pixel(x, y, 0, 255); // Red
                img.set_pixel(x, y, 1, 0);
                img.set_pixel(x, y, 2, 0);
            } else {
                img.set_pixel(x, y, 0, 0);
                img.set_pixel(x, y, 1, 0);
                img.set_pixel(x, y, 2, 255); // Blue
            }
        }
    }

    let (rms, max_diff, size) = roundtrip_with_subsampling(&img, 95.0, ChromaSubsampling::Full);

    println!(
        "Color edge 4:4:4: RMS={:.2}, max_diff={}, size={}",
        rms, max_diff, size
    );

    // 4:4:4 should preserve the color edge well
    assert!(rms < 10.0, "Color edge RMS too high");
}

// ============================================================================
// Grayscale Tests (No Subsampling Needed)
// ============================================================================

#[test]
fn test_grayscale_no_subsampling() {
    let img = test_utils::generate_gradient_h(128, 128, 1);
    let config = EncoderConfig::new().quality(90.0).grayscale();
    let jpeg = encode_gray(128, 128, &img.pixels, &config);

    println!("Grayscale JPEG: {} bytes", jpeg.len());

    let decoder = Decoder::new();
    let decoded = decoder.decode(&jpeg).expect("decode failed");

    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
}

// ============================================================================
// MCU Alignment Tests
// ============================================================================

#[test]
fn test_mcu_aligned_444() {
    // 4:4:4 with image size that is MCU-aligned (8x8)
    let img = generate_gradient_d(64, 64, 3);
    let (rms, _, size) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::Full);

    println!("64x64 4:4:4: RMS={:.2}, size={}", rms, size);
    assert!(rms < 5.0, "MCU-aligned RMS too high");
}

#[test]
fn test_mcu_unaligned_444() {
    // 4:4:4 with image size that is NOT MCU-aligned
    let img = generate_gradient_d(67, 71, 3);
    let (rms, _, size) = roundtrip_with_subsampling(&img, 90.0, ChromaSubsampling::Full);

    println!("67x71 4:4:4: RMS={:.2}, size={}", rms, size);
    assert!(rms < 5.0, "MCU-unaligned RMS too high");
}

// ============================================================================
// File Size Comparisons
// ============================================================================

#[test]
fn test_444_filesize_reasonable() {
    let img = generate_gradient_d(256, 256, 3);
    let (_, _, size_444) = roundtrip_with_subsampling(&img, 85.0, ChromaSubsampling::Full);

    // Calculate bits per pixel
    let bpp = (size_444 as f64 * 8.0) / (256.0 * 256.0);

    println!("256x256 Q85 4:4:4: {} bytes, {:.2} bpp", size_444, bpp);

    // Gradient is simple content - efficient compression is expected
    // Just verify it's in a sane range
    assert!(
        bpp > 0.1 && bpp < 10.0,
        "BPP {:.2} outside expected range",
        bpp
    );
}

// ============================================================================
// Component Count Validation
// ============================================================================

fn count_components_in_sof(jpeg: &[u8]) -> Option<u8> {
    for pos in 0..jpeg.len() - 10 {
        // SOF0 = 0xC0 (Baseline), SOF1 = 0xC1 (Extended Sequential), SOF2 = 0xC2 (Progressive)
        if jpeg[pos] == 0xFF
            && (jpeg[pos + 1] == 0xC0 || jpeg[pos + 1] == 0xC1 || jpeg[pos + 1] == 0xC2)
        {
            return Some(jpeg[pos + 9]);
        }
    }
    None
}

#[test]
fn test_rgb_has_three_components() {
    let img = generate_gradient_d(64, 64, 3);
    let config = EncoderConfig::new().quality(85.0);
    let jpeg = encode_rgb(64, 64, &img.pixels, &config);

    let components = count_components_in_sof(&jpeg).expect("SOF not found");
    assert_eq!(components, 3, "RGB JPEG should have 3 components");
}

#[test]
fn test_grayscale_has_one_component() {
    let img = test_utils::generate_gradient_h(64, 64, 1);
    let config = EncoderConfig::new().quality(85.0).grayscale();
    let jpeg = encode_gray(64, 64, &img.pixels, &config);

    let components = count_components_in_sof(&jpeg).expect("SOF not found");
    assert_eq!(components, 1, "Grayscale JPEG should have 1 component");
}
