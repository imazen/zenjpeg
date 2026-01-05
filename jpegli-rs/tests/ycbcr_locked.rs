//! Locked tests for YCbCr encoding across all subsampling modes.
//!
//! These tests lock down the exact file sizes and bitstream hashes for YCbCr
//! encoding to prevent accidental regressions.
//!
//! Uses frymire.png (1118x1105) - a complex image with high chroma content that
//! exercises more encoder code paths than simple gradients.
//!
//! ⚠️ LOCKED TESTS: Do NOT modify reference values unless intentionally changing
//! the encoder output. If tests fail, investigate why the output changed.
//!
//! Subsampling modes covered:
//! - 4:4:4 (S444) - No chroma subsampling
//! - 4:2:2 (S422) - Horizontal subsampling only
//! - 4:2:0 (S420) - Both horizontal and vertical subsampling
//! - 4:4:0 (S440) - Vertical subsampling only
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
// Generated: 2026-01-05
// Mode: YCbCr with optimized Huffman
// =============================================================================

// =============================================================================
// SEQUENTIAL (BASELINE) MODE
// =============================================================================

/// Sequential S444 optimized Huffman - frymire.png
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 330137),
    (70, 439129),
    (85, 600731),
    (90, 718245),
    (95, 937564),
];

/// Sequential S422 optimized Huffman - frymire.png
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293635),
    (70, 386979),
    (85, 520650),
    (90, 613851),
    (95, 784200),
];

/// Sequential S420 optimized Huffman - frymire.png
const FRYMIRE_S420_SEQ: &[(u8, usize)] = &[
    (50, 269830),
    (70, 362103),
    (85, 493060),
    (90, 580756),
    (95, 737288),
];

/// Sequential S440 optimized Huffman - frymire.png
const FRYMIRE_S440_SEQ: &[(u8, usize)] = &[
    (50, 293829),
    (70, 387491),
    (85, 521927),
    (90, 615696),
    (95, 785549),
];

// =============================================================================
// PROGRESSIVE MODE
// =============================================================================

/// Progressive S444 optimized Huffman - frymire.png
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 320884),
    (70, 425956),
    (85, 582224),
    (90, 697029),
    (95, 908344),
];

/// Progressive S422 optimized Huffman - frymire.png
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 286103),
    (70, 375922),
    (85, 505649),
    (90, 595952),
    (95, 760576),
];

/// Progressive S420 optimized Huffman - frymire.png
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 261084),
    (70, 350187),
    (85, 476388),
    (90, 561365),
    (95, 712304),
];

/// Progressive S440 optimized Huffman - frymire.png
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 284276),
    (70, 374332),
    (85, 504924),
    (90, 595348),
    (95, 759578),
];

// =============================================================================
// BITSTREAM HASHES (Q85)
// =============================================================================

const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x0d24859d4c1daed3;
const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0x169dd02ec79a38ad;
const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0x9fba9388e728d492;
const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0xbb9b6b764c01fafa;
const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0x8b6db5e9fe981d6f;
const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0xf6b8a0409cc47be1;
const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0x13d4b4fce1cdf821;
const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0xd7cf3f2d6b363778;

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

fn encode_jpeg(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    mode: JpegMode,
    subsampling: Subsampling,
) -> Vec<u8> {
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .subsampling(subsampling)
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

/// Helper to test exact sizes for a given mode/subsampling combination
fn test_exact_sizes(
    rgb: &[u8],
    width: u32,
    height: u32,
    mode: JpegMode,
    subsampling: Subsampling,
    expected: &[(u8, usize)],
    mode_name: &str,
    sub_name: &str,
) {
    for &(quality, expected_size) in expected {
        let jpeg = encode_jpeg(rgb, width, height, quality, mode, subsampling);

        assert_eq!(
            jpeg.len(),
            expected_size,
            "Q{} {} {} frymire: size {} != expected {} (diff: {:+})",
            quality,
            mode_name,
            sub_name,
            jpeg.len(),
            expected_size,
            jpeg.len() as i64 - expected_size as i64
        );
    }
}

/// Helper to test exact bitstream hash
fn test_exact_hash(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    mode: JpegMode,
    subsampling: Subsampling,
    expected_hash: u64,
    mode_name: &str,
    sub_name: &str,
) {
    let jpeg = encode_jpeg(rgb, width, height, quality, mode, subsampling);
    let actual_hash = hash_bytes(&jpeg);

    assert_eq!(
        actual_hash, expected_hash,
        "Q{} {} {} frymire: bitstream changed! hash {:#018x} != expected {:#018x}",
        quality, mode_name, sub_name, actual_hash, expected_hash
    );
}

// =============================================================================
// TESTS: Sequential (Baseline) Mode
// =============================================================================

#[test]
fn test_frymire_s444_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Baseline, Subsampling::S444, FRYMIRE_S444_SEQ, "Seq", "4:4:4");
}

#[test]
fn test_frymire_s422_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Baseline, Subsampling::S422, FRYMIRE_S422_SEQ, "Seq", "4:2:2");
}

#[test]
fn test_frymire_s420_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Baseline, Subsampling::S420, FRYMIRE_S420_SEQ, "Seq", "4:2:0");
}

#[test]
fn test_frymire_s440_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Baseline, Subsampling::S440, FRYMIRE_S440_SEQ, "Seq", "4:4:0");
}

#[test]
fn test_frymire_s444_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Baseline, Subsampling::S444, FRYMIRE_S444_SEQ_Q85_HASH, "Seq", "4:4:4");
}

#[test]
fn test_frymire_s422_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Baseline, Subsampling::S422, FRYMIRE_S422_SEQ_Q85_HASH, "Seq", "4:2:2");
}

#[test]
fn test_frymire_s420_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Baseline, Subsampling::S420, FRYMIRE_S420_SEQ_Q85_HASH, "Seq", "4:2:0");
}

#[test]
fn test_frymire_s440_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Baseline, Subsampling::S440, FRYMIRE_S440_SEQ_Q85_HASH, "Seq", "4:4:0");
}

// =============================================================================
// TESTS: Progressive Mode
// =============================================================================

#[test]
fn test_frymire_s444_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Progressive, Subsampling::S444, FRYMIRE_S444_PROG, "Prog", "4:4:4");
}

#[test]
fn test_frymire_s422_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Progressive, Subsampling::S422, FRYMIRE_S422_PROG, "Prog", "4:2:2");
}

#[test]
fn test_frymire_s420_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Progressive, Subsampling::S420, FRYMIRE_S420_PROG, "Prog", "4:2:0");
}

#[test]
fn test_frymire_s440_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(&rgb, w, h, JpegMode::Progressive, Subsampling::S440, FRYMIRE_S440_PROG, "Prog", "4:4:0");
}

#[test]
fn test_frymire_s444_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Progressive, Subsampling::S444, FRYMIRE_S444_PROG_Q85_HASH, "Prog", "4:4:4");
}

#[test]
fn test_frymire_s422_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Progressive, Subsampling::S422, FRYMIRE_S422_PROG_Q85_HASH, "Prog", "4:2:2");
}

#[test]
fn test_frymire_s420_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Progressive, Subsampling::S420, FRYMIRE_S420_PROG_Q85_HASH, "Prog", "4:2:0");
}

#[test]
fn test_frymire_s440_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(&rgb, w, h, 85, JpegMode::Progressive, Subsampling::S440, FRYMIRE_S440_PROG_Q85_HASH, "Prog", "4:4:0");
}

// =============================================================================
// UTILITY: Print current values for updating
// =============================================================================

/// Print current values if you need to update the locked constants.
/// Run with: cargo test --test ycbcr_locked print_current_values -- --ignored --nocapture
#[test]
#[ignore = "utility for updating reference values"]
fn print_current_values() {
    let (rgb, width, height) = load_frymire();
    println!("// Test image: frymire.png {}x{}", width, height);

    let quality_levels = [50, 70, 85, 90, 95];
    let subsampling_modes = [
        (Subsampling::S444, "S444"),
        (Subsampling::S422, "S422"),
        (Subsampling::S420, "S420"),
        (Subsampling::S440, "S440"),
    ];

    println!("\n// SEQUENTIAL MODE\n");
    for (sub, name) in &subsampling_modes {
        println!("const FRYMIRE_{}_SEQ: &[(u8, usize)] = &[", name);
        for q in quality_levels {
            let jpeg = encode_jpeg(&rgb, width, height, q, JpegMode::Baseline, *sub);
            println!("    ({}, {}),", q, jpeg.len());
        }
        println!("];\n");
    }

    println!("// PROGRESSIVE MODE\n");
    for (sub, name) in &subsampling_modes {
        println!("const FRYMIRE_{}_PROG: &[(u8, usize)] = &[", name);
        for q in quality_levels {
            let jpeg = encode_jpeg(&rgb, width, height, q, JpegMode::Progressive, *sub);
            println!("    ({}, {}),", q, jpeg.len());
        }
        println!("];\n");
    }

    println!("// HASHES\n");
    for (sub, name) in &subsampling_modes {
        let seq = encode_jpeg(&rgb, width, height, 85, JpegMode::Baseline, *sub);
        let prog = encode_jpeg(&rgb, width, height, 85, JpegMode::Progressive, *sub);
        println!("const FRYMIRE_{}_SEQ_Q85_HASH: u64 = {:#018x};", name, hash_bytes(&seq));
        println!("const FRYMIRE_{}_PROG_Q85_HASH: u64 = {:#018x};", name, hash_bytes(&prog));
    }
}
