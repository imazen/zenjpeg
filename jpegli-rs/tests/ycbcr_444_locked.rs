//! Locked tests for YCbCr 4:4:4 encoding in sequential and progressive modes.
//!
//! These tests lock down the exact file sizes and bitstream hashes for YCbCr 4:4:4
//! encoding to prevent accidental regressions while working on subsampling issues.
//!
//! Uses frymire.png (1118x1105) - a complex image with high chroma content that
//! exercises more encoder code paths than simple gradients.
//!
//! ⚠️ LOCKED TESTS: Do NOT modify reference values unless intentionally changing
//! the encoder output. If tests fail, investigate why the output changed.
//!
//! Test image: tests/images/frymire.png (1118x1105, high chroma complexity)
//!
//! To regenerate values:
//!   cargo run --release -p jpegli-rs --example get_locked_values

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use jpegli::types::Subsampling;
use jpegli::{Encoder, JpegMode, PixelFormat, Quality};

// =============================================================================
// LOCKED REFERENCE VALUES - frymire.png (1118x1105)
// Generated: 2025-01-05
// Mode: YCbCr 4:4:4 with optimized Huffman
// =============================================================================

/// Sequential (Baseline) 4:4:4 optimized Huffman - frymire.png
/// (quality, size_bytes)
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 330137),
    (70, 439129),
    (85, 600731),
    (90, 718245),
    (95, 937564),
];

/// Progressive 4:4:4 optimized Huffman - frymire.png
/// (quality, size_bytes)
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 320884),
    (70, 425956),
    (85, 582224),
    (90, 697029),
    (95, 908344),
];

/// Bitstream hashes for byte-exact verification
const FRYMIRE_SEQ_Q85_HASH: u64 = 0x0d24859d4c1daed3;
const FRYMIRE_PROG_Q85_HASH: u64 = 0x169dd02ec79a38ad;

// =============================================================================
// Helper functions
// =============================================================================

fn load_frymire() -> (Vec<u8>, u32, u32) {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire.png");
    let png_data = std::fs::read(png_path).expect("Failed to read frymire.png");
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().expect("Failed to read PNG info");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("Failed to decode PNG");

    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .chunks(4)
            .flat_map(|c| [c[0], c[1], c[2]])
            .collect(),
        _ => panic!("Unsupported color type"),
    };

    assert_eq!(info.width, 1118, "frymire.png width mismatch");
    assert_eq!(info.height, 1105, "frymire.png height mismatch");

    (rgb, info.width, info.height)
}

fn encode_jpeg(rgb: &[u8], width: u32, height: u32, quality: u8, mode: JpegMode) -> Vec<u8> {
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .subsampling(Subsampling::S444)
        .optimize_huffman(true)
        .mode(mode)
        .encode(rgb)
        .expect("Encoding failed")
}

fn hash_bytes(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// TESTS: Sequential (Baseline) Mode - frymire.png
// =============================================================================

/// Test that sequential 4:4:4 produces exact file sizes with frymire.png
#[test]
fn test_frymire_s444_sequential_exact_sizes() {
    let (rgb, width, height) = load_frymire();

    for &(quality, expected_size) in FRYMIRE_S444_SEQ {
        let jpeg = encode_jpeg(&rgb, width, height, quality, JpegMode::Baseline);

        assert_eq!(
            jpeg.len(),
            expected_size,
            "Q{} Sequential 4:4:4 frymire: size {} != expected {} (diff: {:+})",
            quality,
            jpeg.len(),
            expected_size,
            jpeg.len() as i64 - expected_size as i64
        );
    }
}

/// Test that sequential 4:4:4 produces exact bitstream at Q85
#[test]
fn test_frymire_s444_sequential_exact_bitstream() {
    let (rgb, width, height) = load_frymire();
    let jpeg = encode_jpeg(&rgb, width, height, 85, JpegMode::Baseline);
    let actual_hash = hash_bytes(&jpeg);

    assert_eq!(
        actual_hash, FRYMIRE_SEQ_Q85_HASH,
        "Q85 Sequential 4:4:4 frymire: bitstream changed! hash {:#018x} != expected {:#018x}",
        actual_hash, FRYMIRE_SEQ_Q85_HASH
    );
}

// =============================================================================
// TESTS: Progressive Mode - frymire.png
// =============================================================================

/// Test that progressive 4:4:4 produces exact file sizes with frymire.png
#[test]
fn test_frymire_s444_progressive_exact_sizes() {
    let (rgb, width, height) = load_frymire();

    for &(quality, expected_size) in FRYMIRE_S444_PROG {
        let jpeg = encode_jpeg(&rgb, width, height, quality, JpegMode::Progressive);

        assert_eq!(
            jpeg.len(),
            expected_size,
            "Q{} Progressive 4:4:4 frymire: size {} != expected {} (diff: {:+})",
            quality,
            jpeg.len(),
            expected_size,
            jpeg.len() as i64 - expected_size as i64
        );
    }
}

/// Test that progressive 4:4:4 produces exact bitstream at Q85
#[test]
fn test_frymire_s444_progressive_exact_bitstream() {
    let (rgb, width, height) = load_frymire();
    let jpeg = encode_jpeg(&rgb, width, height, 85, JpegMode::Progressive);
    let actual_hash = hash_bytes(&jpeg);

    assert_eq!(
        actual_hash, FRYMIRE_PROG_Q85_HASH,
        "Q85 Progressive 4:4:4 frymire: bitstream changed! hash {:#018x} != expected {:#018x}",
        actual_hash, FRYMIRE_PROG_Q85_HASH
    );
}

// =============================================================================
// UTILITY: Print current values for updating
// =============================================================================

/// Print current values if you need to update the locked constants.
/// Run with: cargo test --test ycbcr_444_locked print_current_values -- --ignored --nocapture
#[test]
#[ignore = "utility for updating reference values"]
fn print_current_values() {
    let (rgb, width, height) = load_frymire();
    println!("Test image: frymire.png {}x{}", width, height);

    let quality_levels = [50, 70, 85, 90, 95];

    println!("\n// Sequential (Baseline) 4:4:4 optimized Huffman - frymire.png");
    println!("const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[");
    for q in quality_levels {
        let jpeg = encode_jpeg(&rgb, width, height, q, JpegMode::Baseline);
        println!("    ({}, {}),", q, jpeg.len());
    }
    println!("];");

    println!("\n// Progressive 4:4:4 optimized Huffman - frymire.png");
    println!("const FRYMIRE_S444_PROG: &[(u8, usize)] = &[");
    for q in quality_levels {
        let jpeg = encode_jpeg(&rgb, width, height, q, JpegMode::Progressive);
        println!("    ({}, {}),", q, jpeg.len());
    }
    println!("];");

    println!("\n// Bitstream hashes:");
    let seq = encode_jpeg(&rgb, width, height, 85, JpegMode::Baseline);
    let prog = encode_jpeg(&rgb, width, height, 85, JpegMode::Progressive);
    println!(
        "const FRYMIRE_SEQ_Q85_HASH: u64 = {:#018x};",
        hash_bytes(&seq)
    );
    println!(
        "const FRYMIRE_PROG_Q85_HASH: u64 = {:#018x};",
        hash_bytes(&prog)
    );
}
