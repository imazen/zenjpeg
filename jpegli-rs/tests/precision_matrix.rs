//! Precision matrix test: measures unique colors recoverable for each
//! encode pixel format × decode output format combination.
//!
//! This test answers: "How many distinct colors can survive a roundtrip
//! through jpegli for each input/output format combination?"

use jpegli::decode::Decoder;
use jpegli::{JpegEncoder, PixelFormat, Quality, Subsampling};
use std::collections::HashSet;

/// Test image dimensions
const WIDTH: usize = 128;
const HEIGHT: usize = 128;

/// Quality level for all tests (high quality to isolate format effects)
const QUALITY: f32 = 98.0;

/// Convert sRGB [0,1] to linear [0,1]
fn srgb_to_linear(s: f64) -> f64 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Create test pattern with many distinct colors.
/// Uses a gradient pattern that spans a good range of values.
fn create_test_pattern_rgb8() -> Vec<u8> {
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            // Create distinct colors across the image
            let r = ((x * 2) % 256) as u8;
            let g = ((y * 2) % 256) as u8;
            let b = ((x + y) % 256) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }
    data
}

/// Create test pattern in RGBA8 format
fn create_test_pattern_rgba8() -> Vec<u8> {
    let rgb = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 4);
    for chunk in rgb.chunks(3) {
        data.extend_from_slice(chunk);
        data.push(255); // Alpha
    }
    data
}

/// Create test pattern in BGR8 format
fn create_test_pattern_bgr8() -> Vec<u8> {
    let rgb = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for chunk in rgb.chunks(3) {
        data.push(chunk[2]); // B
        data.push(chunk[1]); // G
        data.push(chunk[0]); // R
    }
    data
}

/// Create test pattern in BGRA8 format
fn create_test_pattern_bgra8() -> Vec<u8> {
    let rgb = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 4);
    for chunk in rgb.chunks(3) {
        data.push(chunk[2]); // B
        data.push(chunk[1]); // G
        data.push(chunk[0]); // R
        data.push(255); // A
    }
    data
}

/// Create test pattern in Gray8 format
fn create_test_pattern_gray8() -> Vec<u8> {
    let mut data = Vec::with_capacity(WIDTH * HEIGHT);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let v = ((x * 2 + y) % 256) as u8;
            data.push(v);
        }
    }
    data
}

/// Create test pattern in RGB16 format (linear values)
fn create_test_pattern_rgb16() -> Vec<u8> {
    let rgb8 = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 6);
    for chunk in rgb8.chunks(3) {
        for &v in chunk {
            // Convert sRGB u8 to linear u16
            let srgb = v as f64 / 255.0;
            let linear = srgb_to_linear(srgb);
            let v16 = (linear * 65535.0) as u16;
            data.extend_from_slice(&v16.to_ne_bytes());
        }
    }
    data
}

/// Create test pattern in RGBA16 format (linear values)
fn create_test_pattern_rgba16() -> Vec<u8> {
    let rgb8 = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 8);
    for chunk in rgb8.chunks(3) {
        for &v in chunk {
            let srgb = v as f64 / 255.0;
            let linear = srgb_to_linear(srgb);
            let v16 = (linear * 65535.0) as u16;
            data.extend_from_slice(&v16.to_ne_bytes());
        }
        data.extend_from_slice(&65535u16.to_ne_bytes()); // Alpha
    }
    data
}

/// Create test pattern in Gray16 format (linear values)
fn create_test_pattern_gray16() -> Vec<u8> {
    let gray8 = create_test_pattern_gray8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 2);
    for &v in &gray8 {
        let srgb = v as f64 / 255.0;
        let linear = srgb_to_linear(srgb);
        let v16 = (linear * 65535.0) as u16;
        data.extend_from_slice(&v16.to_ne_bytes());
    }
    data
}

/// Create test pattern in RgbF32 format (linear values)
fn create_test_pattern_rgbf32() -> Vec<u8> {
    let rgb8 = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 12);
    for chunk in rgb8.chunks(3) {
        for &v in chunk {
            let srgb = v as f64 / 255.0;
            let linear = srgb_to_linear(srgb) as f32;
            data.extend_from_slice(&linear.to_ne_bytes());
        }
    }
    data
}

/// Create test pattern in RgbaF32 format (linear values)
fn create_test_pattern_rgbaf32() -> Vec<u8> {
    let rgb8 = create_test_pattern_rgb8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 16);
    for chunk in rgb8.chunks(3) {
        for &v in chunk {
            let srgb = v as f64 / 255.0;
            let linear = srgb_to_linear(srgb) as f32;
            data.extend_from_slice(&linear.to_ne_bytes());
        }
        data.extend_from_slice(&1.0f32.to_ne_bytes()); // Alpha
    }
    data
}

/// Create test pattern in GrayF32 format (linear values)
fn create_test_pattern_grayf32() -> Vec<u8> {
    let gray8 = create_test_pattern_gray8();
    let mut data = Vec::with_capacity(WIDTH * HEIGHT * 4);
    for &v in &gray8 {
        let srgb = v as f64 / 255.0;
        let linear = srgb_to_linear(srgb) as f32;
        data.extend_from_slice(&linear.to_ne_bytes());
    }
    data
}

/// Count unique RGB colors in u8 data
fn count_unique_rgb_u8(data: &[u8]) -> usize {
    let colors: HashSet<(u8, u8, u8)> = data.chunks(3).map(|c| (c[0], c[1], c[2])).collect();
    colors.len()
}

/// Count unique RGB colors in f32 data (quantized to reasonable precision)
fn count_unique_rgb_f32(data: &[f32], precision_bits: u32) -> usize {
    let scale = (1u64 << precision_bits) as f32;
    let colors: HashSet<(i64, i64, i64)> = data
        .chunks(3)
        .map(|c| {
            (
                (c[0] * scale) as i64,
                (c[1] * scale) as i64,
                (c[2] * scale) as i64,
            )
        })
        .collect();
    colors.len()
}

/// Count unique gray values in u8 data
fn count_unique_gray_u8(data: &[u8]) -> usize {
    let values: HashSet<u8> = data.iter().cloned().collect();
    values.len()
}

/// Count unique gray values in f32 data
fn count_unique_gray_f32(data: &[f32], precision_bits: u32) -> usize {
    let scale = (1u64 << precision_bits) as f32;
    let values: HashSet<i64> = data.iter().map(|&v| (v * scale) as i64).collect();
    values.len()
}

/// Result for one encode/decode combination
#[derive(Debug, Clone)]
struct PrecisionResult {
    encode_format: &'static str,
    decode_format: &'static str,
    unique_colors: usize,
    jpeg_size: usize,
    is_gray: bool,
}

/// Test a single encode/decode combination
fn test_combination(
    encode_format: PixelFormat,
    input_data: &[u8],
    encode_name: &'static str,
) -> Vec<PrecisionResult> {
    let is_gray = matches!(
        encode_format,
        PixelFormat::Gray | PixelFormat::Gray16 | PixelFormat::GrayF32
    );

    // Encode
    let jpeg = JpegEncoder::new(WIDTH as u32, HEIGHT as u32)
        .quality(Quality::from_quality(QUALITY))
        .pixel_format(encode_format)
        .subsampling(Subsampling::S444)
        .encode(input_data)
        .expect(&format!("encode {} failed", encode_name));

    let jpeg_size = jpeg.len();
    let decoder = Decoder::new();

    let mut results = Vec::new();

    // Decode to u8
    if let Ok(decoded_u8) = decoder.decode(&jpeg) {
        let unique = if is_gray {
            // For grayscale, the decoder outputs RGB, so take just R channel
            count_unique_gray_u8(
                &decoded_u8
                    .data
                    .iter()
                    .step_by(3)
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        } else {
            count_unique_rgb_u8(&decoded_u8.data)
        };
        results.push(PrecisionResult {
            encode_format: encode_name,
            decode_format: "u8",
            unique_colors: unique,
            jpeg_size,
            is_gray,
        });
    }

    // Decode to f32
    if let Ok(decoded_f32) = decoder.decode_f32(&jpeg) {
        // f32 at 8-bit precision (for comparison with u8)
        let unique_8bit = if is_gray {
            count_unique_gray_f32(
                &decoded_f32
                    .data
                    .iter()
                    .step_by(3)
                    .cloned()
                    .collect::<Vec<_>>(),
                8,
            )
        } else {
            count_unique_rgb_f32(&decoded_f32.data, 8)
        };
        results.push(PrecisionResult {
            encode_format: encode_name,
            decode_format: "f32→8bit",
            unique_colors: unique_8bit,
            jpeg_size,
            is_gray,
        });

        // f32 at 10-bit precision
        let unique_10bit = if is_gray {
            count_unique_gray_f32(
                &decoded_f32
                    .data
                    .iter()
                    .step_by(3)
                    .cloned()
                    .collect::<Vec<_>>(),
                10,
            )
        } else {
            count_unique_rgb_f32(&decoded_f32.data, 10)
        };
        results.push(PrecisionResult {
            encode_format: encode_name,
            decode_format: "f32→10bit",
            unique_colors: unique_10bit,
            jpeg_size,
            is_gray,
        });

        // f32 at 12-bit precision
        let unique_12bit = if is_gray {
            count_unique_gray_f32(
                &decoded_f32
                    .data
                    .iter()
                    .step_by(3)
                    .cloned()
                    .collect::<Vec<_>>(),
                12,
            )
        } else {
            count_unique_rgb_f32(&decoded_f32.data, 12)
        };
        results.push(PrecisionResult {
            encode_format: encode_name,
            decode_format: "f32→12bit",
            unique_colors: unique_12bit,
            jpeg_size,
            is_gray,
        });

        // f32 at 16-bit precision
        let unique_16bit = if is_gray {
            count_unique_gray_f32(
                &decoded_f32
                    .data
                    .iter()
                    .step_by(3)
                    .cloned()
                    .collect::<Vec<_>>(),
                16,
            )
        } else {
            count_unique_rgb_f32(&decoded_f32.data, 16)
        };
        results.push(PrecisionResult {
            encode_format: encode_name,
            decode_format: "f32→16bit",
            unique_colors: unique_16bit,
            jpeg_size,
            is_gray,
        });

        // Convert to u16 and count
        let data_u16 = decoded_f32.to_u16();
        let unique_u16 = if is_gray {
            let gray: HashSet<u16> = data_u16.iter().step_by(3).cloned().collect();
            gray.len()
        } else {
            let colors: HashSet<(u16, u16, u16)> =
                data_u16.chunks(3).map(|c| (c[0], c[1], c[2])).collect();
            colors.len()
        };
        results.push(PrecisionResult {
            encode_format: encode_name,
            decode_format: "to_u16()",
            unique_colors: unique_u16,
            jpeg_size,
            is_gray,
        });
    }

    results
}

#[test]
fn test_precision_matrix() {
    println!("\n{}", "=".repeat(80));
    println!("PRECISION MATRIX: Unique Colors by Encode Format × Decode Format");
    println!("{}", "=".repeat(80));
    println!(
        "Test image: {}×{} with diverse color pattern",
        WIDTH, HEIGHT
    );
    println!("Quality: {}", QUALITY);
    println!("Subsampling: 4:4:4 (no chroma loss)");
    println!();

    // Collect all results
    let mut all_results: Vec<PrecisionResult> = Vec::new();

    // 8-bit formats
    let test_cases: Vec<(PixelFormat, Vec<u8>, &'static str)> = vec![
        (PixelFormat::Rgb, create_test_pattern_rgb8(), "Rgb (sRGB)"),
        (
            PixelFormat::Rgba,
            create_test_pattern_rgba8(),
            "Rgba (sRGB)",
        ),
        (PixelFormat::Bgr, create_test_pattern_bgr8(), "Bgr (sRGB)"),
        (
            PixelFormat::Bgra,
            create_test_pattern_bgra8(),
            "Bgra (sRGB)",
        ),
        (
            PixelFormat::Gray,
            create_test_pattern_gray8(),
            "Gray (sRGB)",
        ),
        // 16-bit formats (linear)
        (
            PixelFormat::Rgb16,
            create_test_pattern_rgb16(),
            "Rgb16 (linear)",
        ),
        (
            PixelFormat::Rgba16,
            create_test_pattern_rgba16(),
            "Rgba16 (linear)",
        ),
        (
            PixelFormat::Gray16,
            create_test_pattern_gray16(),
            "Gray16 (linear)",
        ),
        // Float formats (linear)
        (
            PixelFormat::RgbF32,
            create_test_pattern_rgbf32(),
            "RgbF32 (linear)",
        ),
        (
            PixelFormat::RgbaF32,
            create_test_pattern_rgbaf32(),
            "RgbaF32 (linear)",
        ),
        (
            PixelFormat::GrayF32,
            create_test_pattern_grayf32(),
            "GrayF32 (linear)",
        ),
    ];

    for (format, data, name) in test_cases {
        let results = test_combination(format, &data, name);
        all_results.extend(results);
    }

    // Print results as a table
    // Group by encode format
    let encode_formats: Vec<&str> = all_results
        .iter()
        .map(|r| r.encode_format)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    let decode_formats = [
        "u8",
        "f32→8bit",
        "f32→10bit",
        "f32→12bit",
        "f32→16bit",
        "to_u16()",
    ];

    // Print header
    println!(
        "┌{:─<20}┬{:─<8}┬{:─<8}┬{:─<8}┬{:─<8}┬{:─<8}┬{:─<8}┬{:─<8}┐",
        "", "", "", "", "", "", "", ""
    );
    print!("│{:^20}│{:^8}│", "Encode Format", "Size");
    for df in &decode_formats {
        print!("{:^8}│", df);
    }
    println!();
    println!(
        "├{:─<20}┼{:─<8}┼{:─<8}┼{:─<8}┼{:─<8}┼{:─<8}┼{:─<8}┼{:─<8}┤",
        "", "", "", "", "", "", "", ""
    );

    // Sort encode formats: 8-bit first, then 16-bit, then float, grays at end
    let mut sorted_formats: Vec<&str> = vec![
        "Rgb (sRGB)",
        "Rgba (sRGB)",
        "Bgr (sRGB)",
        "Bgra (sRGB)",
        "Rgb16 (linear)",
        "Rgba16 (linear)",
        "RgbF32 (linear)",
        "RgbaF32 (linear)",
        "Gray (sRGB)",
        "Gray16 (linear)",
        "GrayF32 (linear)",
    ];
    sorted_formats.retain(|f| encode_formats.contains(f));

    for encode_fmt in sorted_formats {
        let results_for_format: Vec<&PrecisionResult> = all_results
            .iter()
            .filter(|r| r.encode_format == encode_fmt)
            .collect();

        if results_for_format.is_empty() {
            continue;
        }

        let jpeg_size = results_for_format[0].jpeg_size;

        print!("│{:^20}│{:>7} │", encode_fmt, jpeg_size);

        for decode_fmt in &decode_formats {
            if let Some(result) = results_for_format
                .iter()
                .find(|r| r.decode_format == *decode_fmt)
            {
                print!("{:>7} │", result.unique_colors);
            } else {
                print!("{:>7} │", "-");
            }
        }
        println!();
    }

    println!(
        "└{:─<20}┴{:─<8}┴{:─<8}┴{:─<8}┴{:─<8}┴{:─<8}┴{:─<8}┴{:─<8}┘",
        "", "", "", "", "", "", "", ""
    );

    println!();
    println!("Legend:");
    println!("  Size = JPEG file size in bytes");
    println!("  u8 = decode() to Vec<u8>");
    println!("  f32→Nbit = decode_f32() quantized to N-bit precision");
    println!("  to_u16() = decode_f32().to_u16() unique values");
    println!();
    println!("Notes:");
    println!("  - 8-bit sRGB formats have the same input, so similar results");
    println!("  - 16-bit/float formats are LINEAR (gamma correction applied by encoder)");
    println!("  - Higher decode precision reveals sub-8-bit information from JPEG");
    println!("  - Gray formats have fewer unique values (1 channel vs 3)");

    // Verify some key assertions
    // RGB formats should all have similar u8 decode results
    let rgb_u8: Vec<usize> = all_results
        .iter()
        .filter(|r| r.encode_format.starts_with("Rgb") && r.decode_format == "u8" && !r.is_gray)
        .map(|r| r.unique_colors)
        .collect();

    if !rgb_u8.is_empty() {
        let max_rgb_u8 = *rgb_u8.iter().max().unwrap();
        let min_rgb_u8 = *rgb_u8.iter().min().unwrap();
        // All should be within 20% of each other
        assert!(
            min_rgb_u8 as f64 / max_rgb_u8 as f64 > 0.8,
            "RGB formats should produce similar u8 results"
        );
    }

    // f32 decode should always have at least as many unique values as u8
    for encode_fmt in ["Rgb (sRGB)", "Rgb16 (linear)", "RgbF32 (linear)"] {
        let u8_count = all_results
            .iter()
            .find(|r| r.encode_format == encode_fmt && r.decode_format == "u8")
            .map(|r| r.unique_colors);
        let f32_16bit_count = all_results
            .iter()
            .find(|r| r.encode_format == encode_fmt && r.decode_format == "f32→16bit")
            .map(|r| r.unique_colors);

        if let (Some(u8_c), Some(f32_c)) = (u8_count, f32_16bit_count) {
            assert!(
                f32_c >= u8_c,
                "{}: f32→16bit ({}) should have >= unique colors than u8 ({})",
                encode_fmt,
                f32_c,
                u8_c
            );
        }
    }

    println!("\n✓ All assertions passed");
}

/// Focused test on the precision improvement from f32 decode
#[test]
fn test_precision_improvement_summary() {
    println!("\n{}", "=".repeat(60));
    println!("PRECISION IMPROVEMENT SUMMARY");
    println!("{}", "=".repeat(60));

    let input = create_test_pattern_rgb8();

    let jpeg = JpegEncoder::new(WIDTH as u32, HEIGHT as u32)
        .quality(Quality::from_quality(QUALITY))
        .pixel_format(PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .encode(&input)
        .expect("encode failed");

    let decoder = Decoder::new();

    let decoded_u8 = decoder.decode(&jpeg).expect("u8 decode failed");
    let decoded_f32 = decoder.decode_f32(&jpeg).expect("f32 decode failed");

    let unique_input = count_unique_rgb_u8(&input);
    let unique_u8 = count_unique_rgb_u8(&decoded_u8.data);
    let unique_f32_8 = count_unique_rgb_f32(&decoded_f32.data, 8);
    let unique_f32_10 = count_unique_rgb_f32(&decoded_f32.data, 10);
    let unique_f32_12 = count_unique_rgb_f32(&decoded_f32.data, 12);
    let unique_f32_16 = count_unique_rgb_f32(&decoded_f32.data, 16);

    println!("Input unique colors:     {:>6}", unique_input);
    println!("─────────────────────────────────");
    println!("decode() u8:             {:>6}", unique_u8);
    println!("decode_f32() @ 8-bit:    {:>6}", unique_f32_8);
    println!("decode_f32() @ 10-bit:   {:>6}", unique_f32_10);
    println!("decode_f32() @ 12-bit:   {:>6}", unique_f32_12);
    println!("decode_f32() @ 16-bit:   {:>6}", unique_f32_16);
    println!("─────────────────────────────────");

    // Key finding: f32 decode preserves more colors than u8 decode
    let f32_advantage = unique_f32_16 as f64 / unique_u8 as f64;
    println!("f32 / u8 ratio:          {:>6.2}x", f32_advantage);

    // f32 decode should preserve at least as many colors as input
    // (u8 decode loses some due to rounding)
    assert!(
        unique_f32_16 >= unique_u8,
        "f32 decode should preserve at least as many colors as u8: {} vs {}",
        unique_f32_16,
        unique_u8
    );

    // f32 should preserve more colors than u8 (demonstrates higher precision)
    assert!(
        unique_f32_16 > unique_u8,
        "f32→16bit should preserve more colors than u8: {} vs {}",
        unique_f32_16,
        unique_u8
    );

    println!(
        "\n✓ f32 decode preserves {:.1}% more colors than u8",
        (f32_advantage - 1.0) * 100.0
    );
}

/// Test that demonstrates the 10+ bit feature specifically using grayscale.
///
/// Grayscale is where 10+ bit precision is most visible because:
/// - Single channel = easier to measure
/// - The matrix shows Gray goes from 256 u8 → 3,087 at 16-bit (12x improvement!)
#[test]
fn test_10plus_bit_demonstration() {
    println!("\n{}", "=".repeat(60));
    println!("10+ BIT PRECISION DEMONSTRATION (Grayscale)");
    println!("{}", "=".repeat(60));

    // Create grayscale input spanning full 0-255 range
    let width = 128;
    let height = 128;
    let mut input = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            // Full range gradient with variation
            let v = ((x * 2 + y) % 256) as u8;
            input.push(v);
        }
    }

    let unique_input = count_unique_gray_u8(&input);
    println!("Input: {}×{} grayscale", width, height);
    println!("Input unique values: {}", unique_input);

    let jpeg = JpegEncoder::new(width as u32, height as u32)
        .quality(Quality::from_quality(QUALITY))
        .pixel_format(PixelFormat::Gray)
        .encode(&input)
        .expect("encode failed");

    println!("JPEG size: {} bytes", jpeg.len());

    let decoder = Decoder::new();
    let decoded_u8 = decoder.decode(&jpeg).expect("u8 decode failed");
    let decoded_f32 = decoder.decode_f32(&jpeg).expect("f32 decode failed");

    // For grayscale, decoder outputs RGB, so extract just one channel
    let u8_gray: Vec<u8> = decoded_u8.data.iter().step_by(3).cloned().collect();
    let f32_gray: Vec<f32> = decoded_f32.data.iter().step_by(3).cloned().collect();

    let unique_u8 = count_unique_gray_u8(&u8_gray);
    let unique_f32_8 = count_unique_gray_f32(&f32_gray, 8);
    let unique_f32_10 = count_unique_gray_f32(&f32_gray, 10);
    let unique_f32_12 = count_unique_gray_f32(&f32_gray, 12);
    let unique_f32_16 = count_unique_gray_f32(&f32_gray, 16);

    println!();
    println!("Decode results:");
    println!("  u8 output:        {:>5} unique values", unique_u8);
    println!("  f32 @ 8-bit:      {:>5} unique values", unique_f32_8);
    println!("  f32 @ 10-bit:     {:>5} unique values", unique_f32_10);
    println!("  f32 @ 12-bit:     {:>5} unique values", unique_f32_12);
    println!("  f32 @ 16-bit:     {:>5} unique values", unique_f32_16);

    // Calculate effective bits
    let effective_bits_u8 = (unique_u8 as f64).log2();
    let effective_bits_10 = (unique_f32_10 as f64).log2();
    let effective_bits_12 = (unique_f32_12 as f64).log2();
    let effective_bits_16 = (unique_f32_16 as f64).log2();

    println!();
    println!("Effective bits of precision:");
    println!("  u8 output:    {:.1} bits", effective_bits_u8);
    println!("  f32 @ 10-bit: {:.1} bits", effective_bits_10);
    println!("  f32 @ 12-bit: {:.1} bits", effective_bits_12);
    println!("  f32 @ 16-bit: {:.1} bits", effective_bits_16);

    // Calculate improvement ratio
    let improvement_ratio = unique_f32_16 as f64 / unique_u8 as f64;
    println!();
    println!(
        "Precision improvement (f32→16bit / u8): {:.1}x",
        improvement_ratio
    );

    // The key assertions for 10+ bit precision:
    // 1. f32 at 10-bit should show more values than u8 8-bit
    assert!(
        unique_f32_10 > unique_u8,
        "10-bit f32 decode should show more unique values than u8: {} vs {}",
        unique_f32_10,
        unique_u8
    );

    // 2. The improvement should be significant (at least 2x for grayscale)
    assert!(
        improvement_ratio > 2.0,
        "f32→16bit should have at least 2x more values than u8, got {:.1}x",
        improvement_ratio
    );

    println!();
    println!(
        "✓ Confirmed: f32 decode reveals {:.1} bits of precision",
        effective_bits_16
    );
    println!(
        "✓ f32→16bit has {:.1}x more unique values than u8!",
        improvement_ratio
    );
}
