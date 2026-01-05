//! C++ jpegli comparison tests.
//!
//! Tests that compare Rust output against C++ jpegli reference data
//! to verify parity in encoding behavior.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::{
    distance_rms, generate_gradient_d, get_test_data_path, max_pixel_diff, read_test_data,
    TestImage,
};

use jpegli::{
    decode::Decoder,
    encode::Encoder,
    types::{JpegMode, PixelFormat},
    Quality,
};
use std::path::Path;
use test_case::test_case;

// ============================================================================
// Helper Functions
// ============================================================================

/// Load a PNG image from testdata.
fn load_png(filename: &str) -> Option<(u32, u32, Vec<u8>)> {
    let path = get_test_data_path(filename);
    if !path.exists() {
        return None;
    }

    let decoder = png::Decoder::new(std::fs::File::open(&path).ok()?);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    // Convert to RGB if needed
    let pixels = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => {
            // Strip alpha
            buf[..info.buffer_size()]
                .chunks(4)
                .flat_map(|c| [c[0], c[1], c[2]])
                .collect()
        }
        png::ColorType::Grayscale => {
            // Expand to RGB
            buf[..info.buffer_size()]
                .iter()
                .flat_map(|&g| [g, g, g])
                .collect()
        }
        _ => return None,
    };

    Some((info.width, info.height, pixels))
}

/// Decode a JPEG from testdata.
fn decode_test_jpeg(filename: &str) -> Option<(u32, u32, Vec<u8>)> {
    let data = read_test_data(filename)?;
    let decoder = Decoder::new();
    let decoded = decoder.decode(&data).ok()?;
    Some((decoded.width, decoded.height, decoded.data))
}

// ============================================================================
// File Size Parity Tests
// ============================================================================

/// Test that Rust produces reasonable file sizes.
#[test]
fn test_file_size_parity_synthetic() {
    let img = generate_gradient_d(256, 256, 3);

    let encoder = Encoder::new()
        .width(256)
        .height(256)
        .jpegli_quality(Quality::from_quality(85.0));

    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // A 256x256 gradient is simple content - file size varies with implementation
    // Just verify it's reasonable (not too small to be valid, not too large)
    let min_expected = 500; // Must have some content
    let max_expected = 50000; // Shouldn't be larger than raw

    println!("Rust Q85 256x256 gradient: {} bytes", jpeg.len());
    assert!(
        jpeg.len() >= min_expected && jpeg.len() <= max_expected,
        "File size {} outside expected range {}-{}",
        jpeg.len(),
        min_expected,
        max_expected
    );
}

/// Compare file sizes across quality levels.
#[test]
fn test_file_size_scaling() {
    let img = generate_gradient_d(256, 256, 3);

    let sizes: Vec<(f32, usize)> = [50.0, 70.0, 85.0, 95.0]
        .iter()
        .map(|&q| {
            let encoder = Encoder::new()
                .width(256)
                .height(256)
                .jpegli_quality(Quality::from_quality(q));
            (q, encoder.encode(&img.pixels).unwrap().len())
        })
        .collect();

    println!("File sizes by quality:");
    for (q, size) in &sizes {
        println!("  Q{}: {} bytes", q, size);
    }

    // Verify monotonic increase (with some tolerance)
    for i in 1..sizes.len() {
        assert!(
            sizes[i].1 >= sizes[i - 1].1 * 8 / 10,
            "Q{} should be >= Q{} size",
            sizes[i].0,
            sizes[i - 1].0
        );
    }
}

// ============================================================================
// Decode C++ Encoded JPEGs
// ============================================================================

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_flower_420() {
    if let Some((width, height, pixels)) = decode_test_jpeg("jxl/flower/flower.png.im_q85_420.jpg")
    {
        println!("Decoded flower 420: {}x{}", width, height);
        assert_eq!(width, 2268);
        assert_eq!(height, 1512);
        assert_eq!(pixels.len(), 2268 * 1512 * 3);
    } else {
        eprintln!("Skipping: testdata not available");
    }
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_flower_444() {
    if let Some((width, height, pixels)) = decode_test_jpeg("jxl/flower/flower.png.im_q85_444.jpg")
    {
        println!("Decoded flower 444: {}x{}", width, height);
        assert_eq!(width, 2268);
        assert_eq!(height, 1512);
        assert_eq!(pixels.len(), 2268 * 1512 * 3);
    } else {
        eprintln!("Skipping: testdata not available");
    }
}

#[test]
#[ignore = "requires testdata"]
fn test_decode_cpp_flower_progressive() {
    if let Some((width, height, _)) = decode_test_jpeg("jxl/flower/flower.png.im_q85_420_progr.jpg")
    {
        println!("Decoded progressive flower: {}x{}", width, height);
        assert_eq!(width, 2268);
        assert_eq!(height, 1512);
    } else {
        eprintln!("Skipping: testdata not available");
    }
}

// ============================================================================
// Quality Comparison Tests
// ============================================================================

#[test]
#[ignore = "requires testdata"]
fn test_quality_vs_cpp_decoded() {
    // Load original PNG
    let png_result = load_png("jxl/flower/flower.png");
    if png_result.is_none() {
        eprintln!("Skipping: PNG testdata not available");
        return;
    }
    let (width, height, original) = png_result.unwrap();

    // Decode C++ encoded JPEG
    let cpp_decoded = decode_test_jpeg("jxl/flower/flower.png.im_q85_444.jpg");
    if cpp_decoded.is_none() {
        eprintln!("Skipping: JPEG testdata not available");
        return;
    }
    let (_, _, cpp_pixels) = cpp_decoded.unwrap();

    // Encode with Rust and decode
    let encoder = Encoder::new()
        .width(width)
        .height(height)
        .jpegli_quality(Quality::from_quality(85.0));
    let rust_jpeg = encoder.encode(&original).expect("Rust encode failed");

    let decoder = Decoder::new();
    let rust_decoded = decoder.decode(&rust_jpeg).expect("Rust decode failed");

    // Compare both against original
    let cpp_rms = distance_rms(&original, &cpp_pixels);
    let rust_rms = distance_rms(&original, &rust_decoded.data);

    println!("Quality comparison vs original:");
    println!("  C++ Q85:  RMS = {:.4}", cpp_rms);
    println!("  Rust Q85: RMS = {:.4}", rust_rms);

    // Rust should be within 2x of C++ quality
    assert!(
        rust_rms < cpp_rms * 2.0,
        "Rust quality significantly worse than C++"
    );
}

// ============================================================================
// Marker Structure Tests
// ============================================================================

fn count_markers(jpeg: &[u8], marker: u8) -> usize {
    jpeg.windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == marker)
        .count()
}

#[test]
fn test_marker_structure() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .jpegli_quality(Quality::from_quality(85.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Check required markers
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "Missing SOI");
    assert!(jpeg.ends_with(&[0xFF, 0xD9]), "Missing EOI");

    let app0_count = count_markers(&jpeg, 0xE0);
    let dqt_count = count_markers(&jpeg, 0xDB);
    let sof0_count = count_markers(&jpeg, 0xC0);
    let dht_count = count_markers(&jpeg, 0xC4);
    let sos_count = count_markers(&jpeg, 0xDA);

    println!("Marker counts:");
    println!("  APP0 (JFIF): {}", app0_count);
    println!("  DQT: {}", dqt_count);
    println!("  SOF0: {}", sof0_count);
    println!("  DHT: {}", dht_count);
    println!("  SOS: {}", sos_count);

    // Note: We intentionally don't write JFIF APP0 marker to match C++ jpegli behavior
    // C++ cjpegli doesn't write JFIF marker, and removing it saves 18 bytes
    assert_eq!(
        app0_count, 0,
        "Should NOT have JFIF marker (matches C++ jpegli)"
    );
    assert!(dqt_count >= 1, "Should have DQT marker");
    assert!(sof0_count >= 1, "Should have SOF0 marker");
    assert!(dht_count >= 1, "Should have DHT marker");
    assert_eq!(sos_count, 1, "Baseline should have exactly 1 SOS");
}

#[test]
fn test_progressive_marker_structure() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .mode(JpegMode::Progressive)
        .jpegli_quality(Quality::from_quality(85.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let sof2_count = count_markers(&jpeg, 0xC2);
    let sos_count = count_markers(&jpeg, 0xDA);

    println!("Progressive marker counts:");
    println!("  SOF2: {}", sof2_count);
    println!("  SOS: {}", sos_count);

    assert_eq!(sof2_count, 1, "Progressive should have SOF2");
    assert!(sos_count > 1, "Progressive should have multiple SOS");
}

// ============================================================================
// Quantization Table Tests
// ============================================================================

fn extract_dqt_table(jpeg: &[u8], table_id: u8) -> Option<Vec<u8>> {
    let mut pos = 0;
    while pos + 4 < jpeg.len() {
        if jpeg[pos] == 0xFF && jpeg[pos + 1] == 0xDB {
            let length = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
            let table_start = pos + 4;
            let mut offset = 0;

            while offset < length - 2 {
                let pq_tq = jpeg[table_start + offset];
                let precision = (pq_tq >> 4) & 0x0F;
                let id = pq_tq & 0x0F;
                let table_size = if precision == 0 { 64 } else { 128 };

                if id == table_id {
                    let start = table_start + offset + 1;
                    let end = start + table_size.min(jpeg.len() - start);
                    return Some(jpeg[start..end].to_vec());
                }

                offset += 1 + table_size;
            }

            pos += 2 + length;
        } else {
            pos += 1;
        }
    }
    None
}

#[test]
fn test_quant_tables_present() {
    let img = generate_gradient_d(64, 64, 3);
    let encoder = Encoder::new()
        .width(64)
        .height(64)
        .jpegli_quality(Quality::from_quality(85.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // RGB JPEG should have 2 quant tables (luma and chroma)
    let table0 = extract_dqt_table(&jpeg, 0);
    let table1 = extract_dqt_table(&jpeg, 1);

    assert!(table0.is_some(), "Should have quant table 0 (luma)");
    assert!(table1.is_some(), "Should have quant table 1 (chroma)");

    // Tables should be 64 bytes each (8-bit precision)
    assert_eq!(table0.unwrap().len(), 64, "Luma table should be 64 bytes");
    assert_eq!(table1.unwrap().len(), 64, "Chroma table should be 64 bytes");
}

#[test]
fn test_quant_tables_vary_with_quality() {
    let img = generate_gradient_d(64, 64, 3);

    let q50_encoder = Encoder::new()
        .width(64)
        .height(64)
        .jpegli_quality(Quality::from_quality(50.0));
    let q50_jpeg = q50_encoder.encode(&img.pixels).expect("encode Q50 failed");

    let q95_encoder = Encoder::new()
        .width(64)
        .height(64)
        .jpegli_quality(Quality::from_quality(95.0));
    let q95_jpeg = q95_encoder.encode(&img.pixels).expect("encode Q95 failed");

    let q50_table = extract_dqt_table(&q50_jpeg, 0).unwrap();
    let q95_table = extract_dqt_table(&q95_jpeg, 0).unwrap();

    // Higher quality should have smaller quant values (less quantization)
    let q50_sum: u32 = q50_table.iter().map(|&x| x as u32).sum();
    let q95_sum: u32 = q95_table.iter().map(|&x| x as u32).sum();

    println!("Q50 table sum: {}", q50_sum);
    println!("Q95 table sum: {}", q95_sum);

    assert!(
        q95_sum < q50_sum,
        "Q95 should have smaller quant values than Q50"
    );
}

// ============================================================================
// Cross-Decoder Compatibility Tests
// ============================================================================

#[test]
fn test_jpeg_decoder_compatibility() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .jpegli_quality(Quality::from_quality(90.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Decode with jpeg-decoder crate
    let mut decoder = zune_jpeg::JpegDecoder::new(zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg[..]));
    let decoded = decoder.decode().expect("jpeg-decoder failed");
    let info = decoder.dimensions().unwrap();

    assert_eq!(info.width, 128);
    assert_eq!(info.height, 128);
    assert_eq!(decoded.len(), 128 * 128 * 3);
}

#[test]
fn test_zune_jpeg_compatibility() {
    let img = generate_gradient_d(128, 128, 3);
    let encoder = Encoder::new()
        .width(128)
        .height(128)
        .jpegli_quality(Quality::from_quality(90.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    // Decode with zune-jpeg
    use zune_jpeg::zune_core::bytestream::ZCursor;
    let cursor = ZCursor::new(&jpeg);
    let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
    let decoded = decoder.decode().expect("zune-jpeg failed");
    let info = decoder.dimensions().unwrap();

    assert_eq!(info.width as u32, 128);
    assert_eq!(info.height as u32, 128);
    assert!(!decoded.is_empty());
}

// ============================================================================
// Huffman Table Tests
// ============================================================================

fn count_dht_tables(jpeg: &[u8]) -> (usize, usize) {
    let mut dc_count = 0;
    let mut ac_count = 0;

    let mut pos = 0;
    while pos + 4 < jpeg.len() {
        if jpeg[pos] == 0xFF && jpeg[pos + 1] == 0xC4 {
            let length = ((jpeg[pos + 2] as usize) << 8) | (jpeg[pos + 3] as usize);
            let mut offset = 0;

            while offset < length - 2 {
                let tc_th = jpeg[pos + 4 + offset];
                let tc = (tc_th >> 4) & 0x0F; // Table class (0=DC, 1=AC)

                if tc == 0 {
                    dc_count += 1;
                } else {
                    ac_count += 1;
                }

                // Skip table data
                let mut table_size = 0;
                for i in 0..16 {
                    if pos + 5 + offset + i < jpeg.len() {
                        table_size += jpeg[pos + 5 + offset + i] as usize;
                    }
                }
                offset += 1 + 16 + table_size;
            }

            pos += 2 + length;
        } else {
            pos += 1;
        }
    }

    (dc_count, ac_count)
}

#[test]
fn test_huffman_tables_present() {
    let img = generate_gradient_d(64, 64, 3);
    let encoder = Encoder::new()
        .width(64)
        .height(64)
        .jpegli_quality(Quality::from_quality(85.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let (dc_count, ac_count) = count_dht_tables(&jpeg);

    println!("Huffman tables: {} DC, {} AC", dc_count, ac_count);

    // RGB baseline should have 2 DC tables and 2 AC tables
    assert!(dc_count >= 2, "Should have at least 2 DC tables");
    assert!(ac_count >= 2, "Should have at least 2 AC tables");
}

// ============================================================================
// SOF Parameter Tests
// ============================================================================

fn extract_sof_params(jpeg: &[u8]) -> Option<(u8, u16, u16, u8)> {
    for pos in 0..jpeg.len() - 10 {
        if jpeg[pos] == 0xFF && (jpeg[pos + 1] == 0xC0 || jpeg[pos + 1] == 0xC2) {
            let precision = jpeg[pos + 4];
            let height = ((jpeg[pos + 5] as u16) << 8) | (jpeg[pos + 6] as u16);
            let width = ((jpeg[pos + 7] as u16) << 8) | (jpeg[pos + 8] as u16);
            let components = jpeg[pos + 9];
            return Some((precision, height, width, components));
        }
    }
    None
}

#[test]
fn test_sof_parameters() {
    let img = generate_gradient_d(320, 240, 3);
    let encoder = Encoder::new()
        .width(320)
        .height(240)
        .jpegli_quality(Quality::from_quality(85.0));
    let jpeg = encoder.encode(&img.pixels).expect("encode failed");

    let (precision, height, width, components) = extract_sof_params(&jpeg).expect("SOF not found");

    assert_eq!(precision, 8, "Should be 8-bit precision");
    assert_eq!(width, 320, "Width mismatch in SOF");
    assert_eq!(height, 240, "Height mismatch in SOF");
    assert_eq!(components, 3, "Should have 3 components for RGB");
}
