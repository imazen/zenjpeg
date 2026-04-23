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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

// =============================================================================
// LOCKED REFERENCE VALUES - frymire.png (1118x1105)
// Generated: 2026-01-11
// Mode: YCbCr with optimized Huffman
// Updated 2026-01-31: Fixed archmage DCT scaling (was 1/8, now 1/64)
// Updated 2026-02-01: Regenerated after defaulting allow_16bit_quant_tables=false
//                     Q50 sizes are 128 bytes smaller (8-bit DQT instead of 16-bit)
//
// The archmage SIMD uses explicit SIMD intrinsics for DCT which produces
// slightly different float rounding than the wide-crate autovectorized path.
// Both sets are valid — they represent the same algorithm with different FP precision.
// =============================================================================

// =============================================================================
// SEQUENTIAL (BASELINE) MODE
// =============================================================================

#[cfg(target_arch = "x86_64")]
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 330100),
    (70, 437881),
    (85, 597089),
    (90, 713973),
    (95, 934041),
];

#[cfg(not(target_arch = "x86_64"))]
const FRYMIRE_S444_SEQ: &[(u8, usize)] = &[
    (50, 330100),
    (70, 437881),
    (85, 597092),
    (90, 713993),
    (95, 934041),
];

#[cfg(target_arch = "x86_64")]
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293954),
    (70, 386444),
    (85, 518544),
    (90, 611287),
    (95, 782238),
];

#[cfg(not(target_arch = "x86_64"))]
const FRYMIRE_S422_SEQ: &[(u8, usize)] = &[
    (50, 293954),
    (70, 386444),
    (85, 518544),
    (90, 611265),
    (95, 782224),
];

/// S420: identical between SIMD and non-SIMD
/// Updated 2026-04-14: zenyuv integer math with exact avg_epu8 chroma parity
const FRYMIRE_S420_SEQ: &[(u8, usize)] = &[
    (50, 271379),
    (70, 362435),
    (85, 493918),
    (90, 583278),
    (95, 742384),
];

#[cfg(target_arch = "x86_64")]
const FRYMIRE_S440_SEQ: &[(u8, usize)] = &[
    (50, 294260),
    (70, 387012),
    (85, 520040),
    (90, 613143),
    (95, 783753),
];

#[cfg(not(target_arch = "x86_64"))]
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

#[cfg(target_arch = "x86_64")]
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 320320),
    (70, 423516),
    (85, 577429),
    (90, 691442),
    (95, 902410),
];

#[cfg(not(target_arch = "x86_64"))]
const FRYMIRE_S444_PROG: &[(u8, usize)] = &[
    (50, 320320),
    (70, 423513),
    (85, 577433),
    (90, 691401),
    (95, 902392),
];

#[cfg(target_arch = "x86_64")]
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 285948),
    (70, 374209),
    (85, 502181),
    (90, 592037),
    (95, 756373),
];

#[cfg(not(target_arch = "x86_64"))]
const FRYMIRE_S422_PROG: &[(u8, usize)] = &[
    (50, 285948),
    (70, 374206),
    (85, 502181),
    (90, 592004),
    (95, 756381),
];

/// Updated 2026-04-14: zenyuv integer math with exact avg_epu8 chroma parity
#[cfg(target_arch = "x86_64")]
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 262175),
    (70, 349457),
    (85, 475521),
    (90, 561352),
    (95, 714874),
];

/// Updated 2026-04-14: zenyuv integer math — should now match x86_64
#[cfg(not(target_arch = "x86_64"))]
const FRYMIRE_S420_PROG: &[(u8, usize)] = &[
    (50, 262175),
    (70, 349457),
    (85, 475521),
    (90, 561352),
    (95, 714874),
];

#[cfg(target_arch = "x86_64")]
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 284132),
    (70, 372519),
    (85, 501348),
    (90, 591552),
    (95, 755674),
];

#[cfg(not(target_arch = "x86_64"))]
const FRYMIRE_S440_PROG: &[(u8, usize)] = &[
    (50, 284132),
    (70, 372516),
    (85, 501348),
    (90, 591519),
    (95, 755662),
];

// =============================================================================
// BITSTREAM HASHES (Q85)
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod hashes {
    pub const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x05b00d168eae7f72;
    pub const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0x2c0a6aca998d8042;
    pub const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0x8491935fc04738a4;
    pub const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0x4ac5a06df3bc5fe7;
    // Updated 2026-04-14: zenyuv integer math
    pub const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0xae68d13be4b55653;
    pub const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0xb96b671ab7134448;
    pub const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0x44a3acddda6d0591;
    pub const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0x48d3cbff0f61e3e2;
}

#[cfg(not(target_arch = "x86_64"))]
mod hashes {
    pub const FRYMIRE_S444_SEQ_Q85_HASH: u64 = 0x00cb3ae82091dd1c;
    pub const FRYMIRE_S444_PROG_Q85_HASH: u64 = 0x4233015231b9e826;
    pub const FRYMIRE_S422_SEQ_Q85_HASH: u64 = 0xff9fffa84d091a73;
    pub const FRYMIRE_S422_PROG_Q85_HASH: u64 = 0x88b176f2aaaf20a6;
    // Updated 2026-04-14: zenyuv integer math — should now match x86_64
    pub const FRYMIRE_S420_SEQ_Q85_HASH: u64 = 0xae68d13be4b55653;
    pub const FRYMIRE_S420_PROG_Q85_HASH: u64 = 0xb96b671ab7134448;
    pub const FRYMIRE_S440_SEQ_Q85_HASH: u64 = 0xef45fac2c99a3211;
    pub const FRYMIRE_S440_PROG_Q85_HASH: u64 = 0x396ad62cab795db0;
}

use hashes::*;

// =============================================================================
// Helper functions
// =============================================================================

fn load_frymire() -> (Vec<u8>, u32, u32) {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/frymire.png");
    let img = zenjpeg_bench_utils::load_png(std::path::Path::new(png_path))
        .expect("Failed to load frymire.png");
    let width = img.width() as u32;
    let height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();

    assert_eq!(width, 1118, "frymire.png width mismatch");
    assert_eq!(height, 1105, "frymire.png height mismatch");

    (rgb, width, height)
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
        .progressive(progressive)
        .restart_mcu_rows(0); // Disable restart markers to match locked hashes
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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

#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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
#[cfg_attr(
    not(target_arch = "x86_64"),
    ignore = "locked values are x86_64-specific (NEON output differs)"
)]
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
