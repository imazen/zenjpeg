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
use jpegli::{JpegEncoder, JpegMode, PixelFormat};

// =============================================================================
// LOCKED REFERENCE VALUES - frymire.png (1118x1105)
// Generated: 2026-01-11
// Mode: YCbCr with optimized Huffman
// Updated: FMA optimization changes rounding behavior slightly
// =============================================================================

// =============================================================================
// SEQUENTIAL (BASELINE) MODE
// =============================================================================

/// Sequential S444 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 329966),
    (70, 438942),
    (85, 600510),
    (90, 717812),
    (95, 937285),
];

/// Sequential S422 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293489),
    (70, 386804),
    (85, 520481),
    (90, 613614),
    (95, 783905),
];

/// Sequential S420 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S420_SEQ: &[(u8, usize)] = &[
    (50, 269852),
    (70, 362495),
    (85, 494934),
    (90, 584347),
    (95, 743542),
];

/// Sequential S440 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S440_SEQ: &[(u8, usize)] = &[
    (50, 293693),
    (70, 387335),
    (85, 521798),
    (90, 615445),
    (95, 785234),
];

// =============================================================================
// PROGRESSIVE MODE
// =============================================================================

/// Progressive S444 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 320773),
    (70, 425798),
    (85, 582010),
    (90, 696599),
    (95, 908011),
];

/// Progressive S422 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 286032),
    (70, 375822),
    (85, 505529),
    (90, 595625),
    (95, 760319),
];

/// Progressive S420 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 261089),
    (70, 350463),
    (85, 478048),
    (90, 564225),
    (95, 717850),
];

/// Progressive S440 optimized Huffman - frymire.png
/// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 284211),
    (70, 374255),
    (85, 504817),
    (90, 595055),
    (95, 759332),
];

// =============================================================================
// BITSTREAM HASHES (Q85)
// =============================================================================

// Regenerated 2026-01-13 after API cleanup
const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x388e7b1a6777de58;
const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0x979f7c080ce24b0f;
const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0x67793e9a1ec07f6e;
const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0xe1d28667900cb6eb;
const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0x8cd0dde2c892dccf;
const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0x37a97faeb510feba;
const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0x8f52016b6174dbf4;
const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0x13160a47c10f39f4;

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
    JpegEncoder::new(width, height)
        .pixel_format(PixelFormat::Rgb)
        .quality(quality as f32)
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
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Baseline,
        Subsampling::S444,
        FRYMIRE_S444_SEQ,
        "Seq",
        "4:4:4",
    );
}

#[test]
fn test_frymire_s422_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Baseline,
        Subsampling::S422,
        FRYMIRE_S422_SEQ,
        "Seq",
        "4:2:2",
    );
}

#[test]
fn test_frymire_s420_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Baseline,
        Subsampling::S420,
        FRYMIRE_S420_SEQ,
        "Seq",
        "4:2:0",
    );
}

#[test]
fn test_frymire_s440_sequential_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Baseline,
        Subsampling::S440,
        FRYMIRE_S440_SEQ,
        "Seq",
        "4:4:0",
    );
}

#[test]
fn test_frymire_s444_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Baseline,
        Subsampling::S444,
        FRYMIRE_S444_SEQ_Q85_HASH,
        "Seq",
        "4:4:4",
    );
}

#[test]
fn test_frymire_s422_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Baseline,
        Subsampling::S422,
        FRYMIRE_S422_SEQ_Q85_HASH,
        "Seq",
        "4:2:2",
    );
}

#[test]
fn test_frymire_s420_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Baseline,
        Subsampling::S420,
        FRYMIRE_S420_SEQ_Q85_HASH,
        "Seq",
        "4:2:0",
    );
}

#[test]
fn test_frymire_s440_sequential_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Baseline,
        Subsampling::S440,
        FRYMIRE_S440_SEQ_Q85_HASH,
        "Seq",
        "4:4:0",
    );
}

// =============================================================================
// TESTS: Progressive Mode
// =============================================================================

#[test]
fn test_frymire_s444_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Progressive,
        Subsampling::S444,
        FRYMIRE_S444_PROG,
        "Prog",
        "4:4:4",
    );
}

#[test]
fn test_frymire_s422_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Progressive,
        Subsampling::S422,
        FRYMIRE_S422_PROG,
        "Prog",
        "4:2:2",
    );
}

#[test]
fn test_frymire_s420_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Progressive,
        Subsampling::S420,
        FRYMIRE_S420_PROG,
        "Prog",
        "4:2:0",
    );
}

#[test]
fn test_frymire_s440_progressive_sizes() {
    let (rgb, w, h) = load_frymire();
    test_exact_sizes(
        &rgb,
        w,
        h,
        JpegMode::Progressive,
        Subsampling::S440,
        FRYMIRE_S440_PROG,
        "Prog",
        "4:4:0",
    );
}

#[test]
fn test_frymire_s444_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Progressive,
        Subsampling::S444,
        FRYMIRE_S444_PROG_Q85_HASH,
        "Prog",
        "4:4:4",
    );
}

#[test]
fn test_frymire_s422_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Progressive,
        Subsampling::S422,
        FRYMIRE_S422_PROG_Q85_HASH,
        "Prog",
        "4:2:2",
    );
}

#[test]
fn test_frymire_s420_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Progressive,
        Subsampling::S420,
        FRYMIRE_S420_PROG_Q85_HASH,
        "Prog",
        "4:2:0",
    );
}

#[test]
fn test_frymire_s440_progressive_hash() {
    let (rgb, w, h) = load_frymire();
    test_exact_hash(
        &rgb,
        w,
        h,
        85,
        JpegMode::Progressive,
        Subsampling::S440,
        FRYMIRE_S440_PROG_Q85_HASH,
        "Prog",
        "4:4:0",
    );
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
        println!(
            "const FRYMIRE_{}_SEQ_Q85_HASH: u64 = {:#018x};",
            name,
            hash_bytes(&seq)
        );
        println!(
            "const FRYMIRE_{}_PROG_Q85_HASH: u64 = {:#018x};",
            name,
            hash_bytes(&prog)
        );
    }
}
