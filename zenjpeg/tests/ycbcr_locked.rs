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
//!   cargo run --release -p zenjpeg --example get_locked_values
use enough::Unstoppable;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

// =============================================================================
// LOCKED REFERENCE VALUES - frymire.png (1118x1105)
// Generated: 2026-01-11
// Mode: YCbCr with optimized Huffman
// Updated 2026-01-31: Fixed archmage-simd DCT scaling (was 1/8, now 1/64)
// Updated 2026-02-01: Regenerated after defaulting allow_16bit_quant_tables=false
//                     Q50 sizes are 128 bytes smaller (8-bit DQT instead of 16-bit)
//
// The archmage-simd feature uses explicit SIMD intrinsics for DCT which produces
// slightly different float rounding than the wide-crate autovectorized path.
// Both sets are valid — they represent the same algorithm with different FP precision.
// =============================================================================

// =============================================================================
// SEQUENTIAL (BASELINE) MODE
// =============================================================================

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 330100),
    (70, 437881),
    (85, 597089),
    (90, 713973),
    (95, 934041),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 330100),
    (70, 437881),
    (85, 597092),
    (90, 713993),
    (95, 934041),
];

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293954),
    (70, 386444),
    (85, 518544),
    (90, 611287),
    (95, 782238),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293954),
    (70, 386444),
    (85, 518544),
    (90, 611265),
    (95, 782224),
];

/// S420: identical between SIMD and non-SIMD
const FRYMIRE_S420_SEQ: &[(u8, usize)] = &[
    (50, 271315),
    (70, 362380),
    (85, 493820),
    (90, 583172),
    (95, 742408),
];

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S440_SEQ: &[(u8, usize)] = &[
    (50, 294260),
    (70, 387012),
    (85, 520040),
    (90, 613143),
    (95, 783753),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S440_SEQ: &[(u8, usize)] = &[
    (50, 294260),
    (70, 387012),
    (85, 520040),
    (90, 613213),
    (95, 783753),
];

// =============================================================================
// PROGRESSIVE MODE
// =============================================================================

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 321247),
    (70, 424984),
    (85, 579695),
    (90, 694090),
    (95, 905428),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 321247),
    (70, 424984),
    (85, 579702),
    (90, 694018),
    (95, 905420),
];

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 286863),
    (70, 375651),
    (85, 504319),
    (90, 594336),
    (95, 758815),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 286863),
    (70, 375651),
    (85, 504319),
    (90, 594301),
    (95, 758810),
];

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 263131),
    (70, 350659),
    (85, 477444),
    (90, 563816),
    (95, 717002),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 263131),
    (70, 350659),
    (85, 477444),
    (90, 563791),
    (95, 717002),
];

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 285069),
    (70, 374081),
    (85, 503629),
    (90, 593922),
    (95, 758078),
];

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 285069),
    (70, 374081),
    (85, 503629),
    (90, 593887),
    (95, 758081),
];

// =============================================================================
// BITSTREAM HASHES (Q85)
// =============================================================================

#[cfg(all(feature = "archmage-simd", target_arch = "x86_64"))]
mod hashes {
    pub const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x05b00d168eae7f72;
    pub const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0xfd5e1e68867b214e;
    pub const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0x8491935fc04738a4;
    pub const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0x4d75caebc638195a;
    pub const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0x9b29fc6eaee9882f;
    pub const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0xdf2caab44709624f;
    pub const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0x44a3acddda6d0591;
    pub const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0x61159799bc83077d;
}

#[cfg(not(all(feature = "archmage-simd", target_arch = "x86_64")))]
mod hashes {
    pub const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x00cb3ae82091dd1c;
    pub const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0x7e9f99eceec55c93;
    pub const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0xff9fffa84d091a73;
    pub const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0x6d7bcd82d2b04bd2;
    pub const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0x9b29fc6eaee9882f;
    pub const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0x1d1c16350780d052;
    pub const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0xef45fac2c99a3211;
    pub const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0x913f8179586a52d7;
}

use hashes::*;

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
