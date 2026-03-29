//! Tests for 16-bit quantization table support.
//!
//! At very low quality settings, quantization values can exceed 255,
//! requiring 16-bit precision in DQT markers. This test verifies that
//! our encoder produces 16-bit tables when enabled.
//!
//! ## Behavior
//!
//! Controlled by `EncoderConfig::allow_16bit_quant_tables` (default: false):
//!
//! - `allow_16bit_quant_tables=true`: Values up to 32767, 16-bit DQT when >255 (SOF1 extended)
//! - `allow_16bit_quant_tables=false` (default): Clamp to 255, 8-bit DQT (SOF0 baseline)
//!
//! The 32767 limit (not 65535) is because quant values are used in signed
//! arithmetic during DCT coefficient division.
//!
//! ## C++ Behavior (2026-02-01)
//!
//! C++ cjpegli CLI hardcodes `force_baseline=TRUE`, so it always uses 8-bit tables.
//! The C++ library API does support 16-bit tables via `jpegli_set_distance(..., FALSE)`,
//! but this isn't exposed to CLI users. We match the CLI's default behavior.
//!
//! Note: DQT extraction also exists in examples/jpeg_inspect.rs. Consider
//! moving to a shared location if more tests need this functionality.

#[path = "../src/test_utils.rs"]
mod test_utils;

use test_utils::generate_gradient_d;
use zenjpeg::encoder::ChromaSubsampling;
use zenjpeg::encoder::{EncoderConfig, PixelLayout};

/// Maximum quant value for baseline JPEG (8-bit DQT)
const QUANT_MAX_BASELINE: u16 = 255;

/// Maximum quant value for extended JPEG (16-bit DQT)
/// Uses 32767 (not 65535) because values are used in signed arithmetic.
#[allow(dead_code)]
const QUANT_MAX_EXTENDED: u16 = 32767;

/// DQT table info extracted from a JPEG file.
/// See also: examples/jpeg_inspect.rs::QuantTable
#[derive(Debug, Clone)]
struct DqtTable {
    /// Table index (0-3)
    table_idx: u8,
    /// Precision: 0 = 8-bit, 1 = 16-bit
    precision: u8,
    /// Quantization values (in zigzag order as stored)
    values: [u16; 64],
}

/// Extract all DQT tables from JPEG data.
/// See also: examples/jpeg_inspect.rs::parse_dqt
fn extract_dqt_tables(jpeg_data: &[u8]) -> Vec<DqtTable> {
    let mut tables = Vec::new();
    let mut pos = 0;

    while pos + 1 < jpeg_data.len() {
        // Look for marker
        if jpeg_data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let marker = jpeg_data[pos + 1];
        pos += 2;

        // Skip padding bytes
        if marker == 0xFF || marker == 0x00 {
            continue;
        }

        // DQT marker
        if marker == 0xDB {
            if pos + 2 > jpeg_data.len() {
                break;
            }
            let length = ((jpeg_data[pos] as usize) << 8) | (jpeg_data[pos + 1] as usize);
            pos += 2;

            let segment_end = pos + length - 2;
            while pos < segment_end && pos < jpeg_data.len() {
                let info = jpeg_data[pos];
                let precision = info >> 4;
                let table_idx = info & 0x0F;
                pos += 1;

                let mut values = [0u16; 64];
                if precision == 0 {
                    // 8-bit values
                    for i in 0..64 {
                        if pos >= jpeg_data.len() {
                            break;
                        }
                        values[i] = jpeg_data[pos] as u16;
                        pos += 1;
                    }
                } else {
                    // 16-bit values
                    for i in 0..64 {
                        if pos + 1 >= jpeg_data.len() {
                            break;
                        }
                        values[i] = ((jpeg_data[pos] as u16) << 8) | (jpeg_data[pos + 1] as u16);
                        pos += 2;
                    }
                }

                tables.push(DqtTable {
                    table_idx,
                    precision,
                    values,
                });
            }
        } else if marker == 0xD8 || marker == 0xD9 {
            // SOI or EOI - no length
            continue;
        } else if (0xD0..=0xD7).contains(&marker) {
            // RST markers - no length
            continue;
        } else {
            // Other markers with length
            if pos + 2 > jpeg_data.len() {
                break;
            }
            let length = ((jpeg_data[pos] as usize) << 8) | (jpeg_data[pos + 1] as usize);
            pos += length;
        }
    }

    tables
}

/// Calculate what quant values WOULD be at a given quality using standard JPEG scaling.
/// This helps us understand when 16-bit tables are needed.
fn calculate_standard_quant_values(quality: f32) -> (u16, u16) {
    // Standard JPEG luminance table max value is 121 (at position [6,5])
    // Standard JPEG chrominance table has many 99s
    const STD_LUMA_MAX: u16 = 121;
    const STD_CHROMA_MAX: u16 = 99;

    let scale = if quality < 50.0 {
        5000.0 / quality
    } else {
        200.0 - quality * 2.0
    };

    let luma_max = ((STD_LUMA_MAX as f32 * scale + 50.0) / 100.0).round() as u16;
    let chroma_max = ((STD_CHROMA_MAX as f32 * scale + 50.0) / 100.0).round() as u16;

    (luma_max, chroma_max)
}

/// Test that verifies we can detect when 16-bit quant tables are needed.
#[test]
fn test_16bit_quant_threshold_calculation() {
    // At quality 5: scale = 5000/5 = 1000%
    // Max luma quant = (121 * 1000 + 50) / 100 = 1211
    let (luma_q5, chroma_q5) = calculate_standard_quant_values(5.0);
    println!("Quality 5: luma_max={}, chroma_max={}", luma_q5, chroma_q5);
    assert!(
        luma_q5 > 255,
        "Quality 5 should need 16-bit tables for luma"
    );

    // At quality 10: scale = 5000/10 = 500%
    // Max luma quant = (121 * 500 + 50) / 100 = 606
    let (luma_q10, chroma_q10) = calculate_standard_quant_values(10.0);
    println!(
        "Quality 10: luma_max={}, chroma_max={}",
        luma_q10, chroma_q10
    );
    assert!(
        luma_q10 > 255,
        "Quality 10 should need 16-bit tables for luma"
    );

    // At quality 20: scale = 5000/20 = 250%
    // Max luma quant = (121 * 250 + 50) / 100 = 303
    let (luma_q20, chroma_q20) = calculate_standard_quant_values(20.0);
    println!(
        "Quality 20: luma_max={}, chroma_max={}",
        luma_q20, chroma_q20
    );
    assert!(
        luma_q20 > 255,
        "Quality 20 should need 16-bit tables for luma"
    );

    // At quality 25: scale = 5000/25 = 200%
    // Max luma quant = (121 * 200 + 50) / 100 = 243
    let (luma_q25, _) = calculate_standard_quant_values(25.0);
    println!("Quality 25: luma_max={}", luma_q25);
    assert!(
        luma_q25 <= 255,
        "Quality 25 should fit in 8-bit tables for luma"
    );
}

/// Helper to encode with v2 API (default 8-bit tables)
fn encode_test_image(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable).expect("push");
    enc.finish().expect("encode")
}

/// Helper to encode with 16-bit tables enabled
fn encode_test_image_16bit(pixels: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config =
        EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter).allow_16bit_quant_tables(true);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(pixels, enough::Unstoppable).expect("push");
    enc.finish().expect("encode")
}

/// Test that the DQT parser can extract tables correctly.
#[test]
fn test_dqt_extraction_basic() {
    let img = generate_gradient_d(64, 64, 3);
    let jpeg = encode_test_image(&img.pixels, 64, 64, 90.0);

    let tables = extract_dqt_tables(&jpeg);
    assert!(!tables.is_empty(), "Should extract at least one DQT table");

    for table in &tables {
        println!(
            "Table {}: precision={}, max_val={}",
            table.table_idx,
            table.precision,
            table.values.iter().max().unwrap()
        );
        // At Q90, all values should be small and 8-bit
        assert_eq!(table.precision, 0, "Q90 should use 8-bit precision");
        assert!(
            *table.values.iter().max().unwrap() <= 255,
            "Q90 values should fit in 8 bits"
        );
    }
}

/// Test that the default behavior clamps quant values to 255 (8-bit tables).
///
/// This matches C++ cjpegli's behavior which hardcodes force_baseline=TRUE.
#[test]
fn test_default_uses_8bit_tables() {
    let img = generate_gradient_d(64, 64, 3);

    for quality in [1, 5, 10, 15, 20] {
        let jpeg = encode_test_image(&img.pixels, 64, 64, quality as f32);
        let tables = extract_dqt_tables(&jpeg);

        for table in &tables {
            assert_eq!(
                table.precision, 0,
                "Quality {}: default should use 8-bit tables (precision=0)",
                quality
            );
            let max_value = *table.values.iter().max().unwrap();
            assert!(
                max_value <= QUANT_MAX_BASELINE,
                "Quality {}: values should be clamped to 255, got {}",
                quality,
                max_value
            );
        }

        println!(
            "Quality {}: {} tables, all 8-bit, max_value={}",
            quality,
            tables.len(),
            tables
                .iter()
                .flat_map(|t| t.values.iter())
                .max()
                .unwrap_or(&0)
        );
    }
}

/// Test that 16-bit tables are used when explicitly enabled.
///
/// With allow_16bit_quant_tables=true, quant values can exceed 255 and
/// the encoder uses SOF1 (extended sequential) with 16-bit DQT markers.
#[test]
fn test_16bit_tables_when_enabled() {
    let img = generate_gradient_d(64, 64, 3);

    let mut found_16bit = false;
    let mut found_values_over_255 = false;

    for quality in [1, 5, 10, 15, 20] {
        let jpeg = encode_test_image_16bit(&img.pixels, 64, 64, quality as f32);
        let tables = extract_dqt_tables(&jpeg);

        let max_value: u16 = tables
            .iter()
            .flat_map(|t| t.values.iter())
            .copied()
            .max()
            .unwrap_or(0);

        let has_16bit = tables.iter().any(|t| t.precision == 1);

        println!(
            "Quality {} (16-bit enabled): {} tables, max_value={}, precisions={:?}",
            quality,
            tables.len(),
            max_value,
            tables.iter().map(|t| t.precision).collect::<Vec<_>>()
        );

        if max_value > QUANT_MAX_BASELINE {
            found_values_over_255 = true;
            assert!(
                has_16bit,
                "Quality {}: max_value={} > 255 but no 16-bit tables!",
                quality, max_value
            );
        }

        if has_16bit {
            found_16bit = true;
        }
    }

    // Verify we actually tested the 16-bit scenario
    assert!(
        found_values_over_255,
        "Test should have found quality levels with quant values > 255"
    );
    assert!(
        found_16bit,
        "Test should have found quality levels using 16-bit tables"
    );
}

// ============================================================================
// C++ Comparison Tests (require cjpegli binary)
// ============================================================================

#[cfg(feature = "__ffi-tests")]
mod cpp_comparison {
    use super::*;
    use std::process::Command;

    fn get_cjpegli_path() -> Option<std::path::PathBuf> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../internal/jpegli-cpp/build/tools/cjpegli");
        if path.exists() { Some(path) } else { None }
    }

    /// Compare DQT tables between Rust and C++ at very low quality.
    #[test]
    #[ignore] // Requires cjpegli binary
    fn test_16bit_quant_cpp_comparison() {
        let cjpegli = match get_cjpegli_path() {
            Some(p) => p,
            None => {
                eprintln!("cjpegli not found, skipping test");
                return;
            }
        };

        let img = generate_gradient_d(64, 64, 3);

        // Write test image to temp file
        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("test_16bit_input.ppm");
        let cpp_output_path = temp_dir.join("test_16bit_cpp.jpg");

        // Write PPM file
        let mut ppm = Vec::new();
        ppm.extend_from_slice("P6\n64 64\n255\n".to_string().as_bytes());
        ppm.extend_from_slice(&img.pixels);
        std::fs::write(&input_path, &ppm).expect("write ppm");

        // Test at quality 5 (should need 16-bit tables)
        let quality = 5;

        // Encode with cjpegli
        let output = Command::new(&cjpegli)
            .args([
                input_path.to_str().unwrap(),
                cpp_output_path.to_str().unwrap(),
                "-q",
                &quality.to_string(),
            ])
            .output()
            .expect("cjpegli failed");

        if !output.status.success() {
            eprintln!(
                "cjpegli stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            panic!("cjpegli failed");
        }

        // Read C++ output and extract DQT
        let cpp_jpeg = std::fs::read(&cpp_output_path).expect("read cpp jpeg");
        let cpp_tables = extract_dqt_tables(&cpp_jpeg);

        // Encode with Rust
        let rust_jpeg = encode_test_image(&img.pixels, 64, 64, quality as f32);
        let rust_tables = extract_dqt_tables(&rust_jpeg);

        println!("\n=== Quality {} Comparison ===", quality);
        println!("C++ tables:");
        for t in &cpp_tables {
            println!(
                "  Table {}: precision={}, max={}",
                t.table_idx,
                t.precision,
                t.values.iter().max().unwrap()
            );
        }
        println!("Rust tables:");
        for t in &rust_tables {
            println!(
                "  Table {}: precision={}, max={}",
                t.table_idx,
                t.precision,
                t.values.iter().max().unwrap()
            );
        }

        // Compare precisions
        let cpp_has_16bit = cpp_tables.iter().any(|t| t.precision == 1);
        let rust_has_16bit = rust_tables.iter().any(|t| t.precision == 1);

        if cpp_has_16bit {
            assert!(
                rust_has_16bit,
                "C++ uses 16-bit tables at quality {}, but Rust uses 8-bit (clamping bug)",
                quality
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&cpp_output_path);
    }
}
