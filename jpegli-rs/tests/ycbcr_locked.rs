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

use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

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
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 329833),
    (70, 438877),
    (85, 600464),
    (90, 717969),
    (95, 937419),
];

/// Sequential S422 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293349),
    (70, 386838),
    (85, 520431),
    (90, 613771),
    (95, 784128),
];

/// Sequential S420 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S420_SEQ: &[(u8, usize)] = &[
    (50, 269794),
    (70, 362383),
    (85, 494876),
    (90, 584245),
    (95, 743740),
];

/// Sequential S440 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S440_SEQ: &[(u8, usize)] = &[
    (50, 293586),
    (70, 387314),
    (85, 521772),
    (90, 615697),
    (95, 785471),
];

// =============================================================================
// PROGRESSIVE MODE
// =============================================================================

/// Progressive S444 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 320726),
    (70, 425849),
    (85, 582218),
    (90, 696683),
    (95, 907984),
];

/// Progressive S422 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 285950),
    (70, 375939),
    (85, 505593),
    (90, 595739),
    (95, 760406),
];

/// Progressive S420 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 261177),
    (70, 350443),
    (85, 477905),
    (90, 564241),
    (95, 717866),
];

/// Progressive S440 optimized Huffman - frymire.png
/// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 284121),
    (70, 374321),
    (85, 504828),
    (90, 595188),
    (95, 759472),
];

// =============================================================================
// BITSTREAM HASHES (Q85)
// =============================================================================

// Regenerated 2026-01-16 after revert of incorrect Cb/Cr table sharing fix
const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x616ba0d75e7258ba;
const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0xe835b9d262d476e2;
const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0x550e0ad4fc873998;
const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0x2b90180753df0b91;
const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0x040db43f7d8aa6dc;
const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0x649683600b3b8aea;
const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0xda9b6f162dc6c14f;
const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0x0c8400b65422f685;

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
    progressive: bool,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality as f32, subsampling)
        .optimize_huffman(true)
        .progressive(progressive);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(rgb, enough::Unstoppable).expect("push");
    enc.finish().expect("Encoding failed")
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
    progressive: bool,
    subsampling: ChromaSubsampling,
    expected: &[(u8, usize)],
    mode_name: &str,
    sub_name: &str,
) {
    for &(quality, expected_size) in expected {
        let jpeg = encode_jpeg(rgb, width, height, quality, progressive, subsampling);

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
    progressive: bool,
    subsampling: ChromaSubsampling,
    expected_hash: u64,
    mode_name: &str,
    sub_name: &str,
) {
    let jpeg = encode_jpeg(rgb, width, height, quality, progressive, subsampling);
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
        false,
        ChromaSubsampling::None,
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
        false,
        ChromaSubsampling::HalfHorizontal,
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
        false,
        ChromaSubsampling::Quarter,
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
        false,
        ChromaSubsampling::HalfVertical,
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
        false,
        ChromaSubsampling::None,
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
        false,
        ChromaSubsampling::HalfHorizontal,
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
        false,
        ChromaSubsampling::Quarter,
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
        false,
        ChromaSubsampling::HalfVertical,
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
        true,
        ChromaSubsampling::None,
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
        true,
        ChromaSubsampling::HalfHorizontal,
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
        true,
        ChromaSubsampling::Quarter,
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
        true,
        ChromaSubsampling::HalfVertical,
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
        true,
        ChromaSubsampling::None,
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
        true,
        ChromaSubsampling::HalfHorizontal,
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
        true,
        ChromaSubsampling::Quarter,
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
        true,
        ChromaSubsampling::HalfVertical,
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
        (ChromaSubsampling::None, "S444"),
        (ChromaSubsampling::HalfHorizontal, "S422"),
        (ChromaSubsampling::Quarter, "S420"),
        (ChromaSubsampling::HalfVertical, "S440"),
    ];

    println!("\n// SEQUENTIAL MODE\n");
    for (sub, name) in &subsampling_modes {
        println!("const FRYMIRE_{}_SEQ: &[(u8, usize)] = &[", name);
        for q in quality_levels {
            let jpeg = encode_jpeg(&rgb, width, height, q, false, *sub);
            println!("    ({}, {}),", q, jpeg.len());
        }
        println!("];\n");
    }

    println!("// PROGRESSIVE MODE\n");
    for (sub, name) in &subsampling_modes {
        println!("const FRYMIRE_{}_PROG: &[(u8, usize)] = &[", name);
        for q in quality_levels {
            let jpeg = encode_jpeg(&rgb, width, height, q, true, *sub);
            println!("    ({}, {}),", q, jpeg.len());
        }
        println!("];\n");
    }

    println!("// HASHES\n");
    for (sub, name) in &subsampling_modes {
        let seq = encode_jpeg(&rgb, width, height, 85, false, *sub);
        let prog = encode_jpeg(&rgb, width, height, 85, true, *sub);
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
