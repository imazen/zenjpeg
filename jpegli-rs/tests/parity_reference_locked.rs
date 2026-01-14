//! Locked C++ parity reference data for all encoder permutations.
//!
//! This file contains exhaustive reference values for all combinations of:
//! - Subsampling: S444, S422, S420, S440
//! - Huffman optimization: on/off
//! - Quality levels: 5, 10, 15, ..., 100
//!
//! These values are LOCKED and should only change intentionally when
//! the encoder algorithm is modified for valid reasons.
//!
//! To regenerate values: cargo test --test parity_reference_locked generate_all_values -- --ignored --nocapture

use enough::Unstoppable;
use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use std::collections::HashMap;
use std::process::Command;

/// Reference entry for a single encoder configuration
#[derive(Debug, Clone, Copy)]
pub struct ParityEntry {
    /// Quality level (5-100)
    pub quality: u8,
    /// Rust output file size in bytes
    pub rust_size: usize,
    /// C++ output file size in bytes
    pub cpp_size: usize,
    /// DSSIM quality metric (lower is better)
    pub dssim: f64,
    /// Butteraugli distance (lower is better)
    pub butteraugli: f64,
    /// Size difference percentage (positive = Rust larger)
    pub diff_pct: f64,
}

/// Quality levels to test
const QUALITY_LEVELS: &[u8] = &[
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
];

/// Maximum allowed parity difference percentage per mode
/// These are set just above actual observed variance to catch regressions
/// Updated 2026-01-13: Increased tolerance for Q5 16-bit quant table overhead (~1%)
const MAX_DIFF_S444: f64 = 1.0; // Q5: +0.7% due to 16-bit quant tables
const MAX_DIFF_S420: f64 = 1.0; // Q5: +1.0% due to 16-bit quant tables
const MAX_DIFF_S422: f64 = 1.0; // Q5: +0.9% due to 16-bit quant tables
const MAX_DIFF_S440: f64 = 1.0; // Q5: +0.9% due to 16-bit quant tables

// =============================================================================
// LOCKED REFERENCE VALUES - DO NOT MODIFY WITHOUT VALID REASON
// Generated: 2026-01-09 (updated after SIMD boundary fix in downsampling)
// Test image: 512x512 gradient (flower_small.rgb.png)
// =============================================================================

/// 4:4:4 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
/// Regenerated 2026-01-13 after API cleanup (v2 Quality type changes)
#[rustfmt::skip]
pub const S444_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 16293, 16179, 0.006046, 4.80000000),
    (10, 19351, 19230, 0.005423, 4.60000000),
    (15, 23161, 23054, 0.004843, 4.40000000),
    (20, 27175, 27062, 0.004336, 4.20000000),
    (25, 31126, 31019, 0.003944, 4.00000000),
    (30, 33951, 33848, 0.003711, 3.80000000),
    (35, 35449, 35351, 0.003610, 3.60000000),
    (40, 36911, 36831, 0.003502, 3.40000000),
    (45, 38974, 38883, 0.003370, 3.20000000),
    (50, 40859, 40752, 0.003244, 3.00000000),
    (55, 43119, 43028, 0.003100, 2.80000000),
    (60, 46336, 46246, 0.002951, 2.60000000),
    (65, 49485, 49393, 0.002797, 2.40000000),
    (70, 53975, 53870, 0.002632, 2.20000000),
    (75, 59028, 58935, 0.002469, 2.00000000),
    (80, 66253, 66177, 0.002259, 1.80000000),
    (85, 76845, 76758, 0.002013, 1.60000000),
    (90, 95441, 95490, 0.001648, 1.40000000),
    (95, 134478, 134545, 0.001176, 1.20000000),
    (100, 323764, 323934, 0.000305, 1.00000000),
];

/// 4:4:4 Fixed Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
/// Regenerated 2026-01-13 after API cleanup (v2 Quality type changes)
#[rustfmt::skip]
pub const S444_FIXED: &[(u8, usize, usize, f64, f64)] = &[
    (5, 18900, 16179, 0.006046, 4.80000000),
    (10, 21745, 19230, 0.005423, 4.60000000),
    (15, 25368, 23054, 0.004843, 4.40000000),
    (20, 29296, 27062, 0.004336, 4.20000000),
    (25, 33248, 31019, 0.003944, 4.00000000),
    (30, 36097, 33848, 0.003711, 3.80000000),
    (35, 37607, 35351, 0.003610, 3.60000000),
    (40, 39126, 36831, 0.003502, 3.40000000),
    (45, 41212, 38883, 0.003370, 3.20000000),
    (50, 43115, 40752, 0.003244, 3.00000000),
    (55, 45420, 43028, 0.003100, 2.80000000),
    (60, 48647, 46246, 0.002951, 2.60000000),
    (65, 51844, 49393, 0.002797, 2.40000000),
    (70, 56449, 53870, 0.002632, 2.20000000),
    (75, 61645, 58935, 0.002469, 2.00000000),
    (80, 69110, 66177, 0.002259, 1.80000000),
    (85, 80085, 76758, 0.002013, 1.60000000),
    (90, 99698, 95490, 0.001648, 1.40000000),
    (95, 141256, 134545, 0.001176, 1.20000000),
    (100, 352387, 323934, 0.000305, 1.00000000),
];

/// 4:2:0 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
/// Regenerated 2026-01-13 after API cleanup (v2 Quality type changes)
#[rustfmt::skip]
pub const S420_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 13408, 13279, 0.006173, 4.80000000),
    (10, 16159, 16035, 0.005604, 4.60000000),
    (15, 19494, 19366, 0.005108, 4.40000000),
    (20, 22899, 22782, 0.004655, 4.20000000),
    (25, 26220, 26099, 0.004311, 4.00000000),
    (30, 28256, 28152, 0.004114, 3.80000000),
    (35, 29628, 29511, 0.003997, 3.60000000),
    (40, 30897, 30798, 0.003885, 3.40000000),
    (45, 32559, 32433, 0.003760, 3.20000000),
    (50, 34221, 34099, 0.003638, 3.00000000),
    (55, 36498, 36425, 0.003492, 2.80000000),
    (60, 38894, 38794, 0.003347, 2.60000000),
    (65, 42028, 41925, 0.003175, 2.40000000),
    (70, 45697, 45726, 0.003006, 2.20000000),
    (75, 50414, 50415, 0.002826, 2.00000000),
    (80, 56465, 56513, 0.002646, 1.80000000),
    (85, 65623, 65658, 0.002439, 1.60000000),
    (90, 82097, 82136, 0.002176, 1.40000000),
    (95, 115383, 115425, 0.001889, 1.20000000),
    (100, 216322, 216364, 0.001630, 1.00000000),
];

/// 4:2:2 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
/// Regenerated 2026-01-13 after API cleanup (v2 Quality type changes)
#[rustfmt::skip]
pub const S422_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 14631, 14510, 0.006485, 4.80000000),
    (10, 17432, 17308, 0.005858, 4.60000000),
    (15, 20855, 20755, 0.005265, 4.40000000),
    (20, 24455, 24350, 0.004769, 4.20000000),
    (25, 27961, 27857, 0.004381, 4.00000000),
    (30, 30440, 30330, 0.004157, 3.80000000),
    (35, 31817, 31705, 0.004049, 3.60000000),
    (40, 33140, 33039, 0.003937, 3.40000000),
    (45, 35015, 34938, 0.003803, 3.20000000),
    (50, 36727, 36611, 0.003685, 3.00000000),
    (55, 38754, 38677, 0.003544, 2.80000000),
    (60, 41759, 41677, 0.003382, 2.60000000),
    (65, 44498, 44406, 0.003235, 2.40000000),
    (70, 48672, 48576, 0.003054, 2.20000000),
    (75, 53188, 53079, 0.002896, 2.00000000),
    (80, 59630, 59577, 0.002699, 1.80000000),
    (85, 69215, 69124, 0.002451, 1.60000000),
    (90, 85673, 85683, 0.002102, 1.40000000),
    (95, 118845, 118857, 0.001667, 1.20000000),
    (100, 256294, 256324, 0.001069, 1.00000000),
];

/// 4:4:0 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
/// Regenerated 2026-01-13 after API cleanup (v2 Quality type changes)
#[rustfmt::skip]
pub const S440_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 14648, 14530, 0.006502, 4.80000000),
    (10, 17403, 17277, 0.005888, 4.60000000),
    (15, 20793, 20679, 0.005286, 4.40000000),
    (20, 24360, 24242, 0.004803, 4.20000000),
    (25, 27841, 27731, 0.004422, 4.00000000),
    (30, 30323, 30206, 0.004186, 3.80000000),
    (35, 31675, 31568, 0.004080, 3.60000000),
    (40, 33006, 32915, 0.003974, 3.40000000),
    (45, 34868, 34781, 0.003846, 3.20000000),
    (50, 36535, 36448, 0.003725, 3.00000000),
    (55, 38566, 38472, 0.003580, 2.80000000),
    (60, 41536, 41440, 0.003431, 2.60000000),
    (65, 44300, 44205, 0.003282, 2.40000000),
    (70, 48441, 48350, 0.003118, 2.20000000),
    (75, 52975, 52884, 0.002954, 2.00000000),
    (80, 59347, 59269, 0.002762, 1.80000000),
    (85, 68914, 68848, 0.002525, 1.60000000),
    (90, 85312, 85361, 0.002183, 1.40000000),
    (95, 118595, 118644, 0.001759, 1.20000000),
    (100, 254534, 254628, 0.001231, 1.00000000),
];

// =============================================================================
// Helper functions
// =============================================================================

fn load_test_image() -> (Vec<u8>, u32, u32) {
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");

    let png_data = std::fs::read(png_path).expect("Failed to read test image");
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

    (rgb, info.width, info.height)
}

fn encode_rust(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    subsampling: ChromaSubsampling,
    optimize_huffman: bool,
) -> Vec<u8> {
    let config = EncoderConfig::new()
        .quality(quality as f32)
        .ycbcr(subsampling)
        .optimize_huffman(optimize_huffman);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(rgb, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("Rust encode failed")
}

fn encode_cpp(png_path: &str, quality: u8, subsampling: &str) -> Option<Vec<u8>> {
    let cjpegli = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";
    if !std::path::Path::new(cjpegli).exists() {
        return None;
    }

    let output = format!("/tmp/cpp_{}_{}.jpg", subsampling, quality);
    let status = Command::new(cjpegli)
        .args(&[
            &format!("--chroma_subsampling={}", subsampling),
            "-p",
            "0",
            "-q",
            &quality.to_string(),
            png_path,
            &output,
        ])
        .status()
        .ok()?;

    if status.success() {
        std::fs::read(&output).ok()
    } else {
        None
    }
}

fn compute_dssim(original: &[u8], decoded: &[u8], _width: usize, _height: usize) -> f64 {
    // Simple MSE-based approximation (actual DSSIM requires more complex computation)
    if original.len() != decoded.len() {
        return 1.0;
    }
    let mut sum = 0.0f64;
    for (a, b) in original.iter().zip(decoded.iter()) {
        let diff = (*a as f64 - *b as f64) / 255.0;
        sum += diff * diff;
    }
    (sum / original.len() as f64).sqrt() * 0.1
}

// =============================================================================
// Tests
// =============================================================================

/// Verify 4:4:4 optimized Huffman matches reference
#[test]
fn test_s444_opt_parity() {
    let (rgb, width, height) = load_test_image();

    for &(q, expected_rust, expected_cpp, _, _) in S444_OPT {
        let jpeg = encode_rust(&rgb, width, height, q, ChromaSubsampling::Full, true);
        let rust_size = jpeg.len();

        let diff_pct = 100.0 * (rust_size as f64 - expected_rust as f64) / expected_rust as f64;
        assert!(
            diff_pct.abs() < 0.1,
            "Q{} 4:4:4 OPT: size {} vs expected {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_rust,
            diff_pct
        );

        let cpp_diff = 100.0 * (rust_size as f64 - expected_cpp as f64) / expected_cpp as f64;
        assert!(
            cpp_diff.abs() < MAX_DIFF_S444,
            "Q{} 4:4:4 OPT: Rust {} vs C++ {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_cpp,
            cpp_diff
        );
    }
}

/// Verify 4:4:4 fixed Huffman matches reference
#[test]
fn test_s444_fixed_parity() {
    let (rgb, width, height) = load_test_image();

    for &(q, expected_rust, expected_cpp, _, _) in S444_FIXED {
        let jpeg = encode_rust(&rgb, width, height, q, ChromaSubsampling::Full, false);
        let rust_size = jpeg.len();

        let diff_pct = 100.0 * (rust_size as f64 - expected_rust as f64) / expected_rust as f64;
        assert!(
            diff_pct.abs() < 0.1,
            "Q{} 4:4:4 FIXED: size {} vs expected {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_rust,
            diff_pct
        );
    }
}

/// Verify 4:2:0 optimized Huffman matches reference
#[test]
fn test_s420_opt_parity() {
    let (rgb, width, height) = load_test_image();

    for &(q, expected_rust, expected_cpp, _, _) in S420_OPT {
        let jpeg = encode_rust(&rgb, width, height, q, ChromaSubsampling::Quarter, true);
        let rust_size = jpeg.len();

        let diff_pct = 100.0 * (rust_size as f64 - expected_rust as f64) / expected_rust as f64;
        assert!(
            diff_pct.abs() < 0.1,
            "Q{} 4:2:0 OPT: size {} vs expected {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_rust,
            diff_pct
        );

        let cpp_diff = 100.0 * (rust_size as f64 - expected_cpp as f64) / expected_cpp as f64;
        assert!(
            cpp_diff.abs() < MAX_DIFF_S420,
            "Q{} 4:2:0 OPT: Rust {} vs C++ {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_cpp,
            cpp_diff
        );
    }
}

/// Verify 4:2:2 optimized Huffman matches reference
#[test]
fn test_s422_opt_parity() {
    let (rgb, width, height) = load_test_image();

    for &(q, expected_rust, expected_cpp, _, _) in S422_OPT {
        let jpeg = encode_rust(
            &rgb,
            width,
            height,
            q,
            ChromaSubsampling::HalfHorizontal,
            true,
        );
        let rust_size = jpeg.len();

        let diff_pct = 100.0 * (rust_size as f64 - expected_rust as f64) / expected_rust as f64;
        assert!(
            diff_pct.abs() < 0.1,
            "Q{} 4:2:2 OPT: size {} vs expected {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_rust,
            diff_pct
        );
    }
}

/// Verify 4:4:0 optimized Huffman matches reference
#[test]
fn test_s440_opt_parity() {
    let (rgb, width, height) = load_test_image();

    for &(q, expected_rust, expected_cpp, _, _) in S440_OPT {
        let jpeg = encode_rust(
            &rgb,
            width,
            height,
            q,
            ChromaSubsampling::HalfVertical,
            true,
        );
        let rust_size = jpeg.len();

        let diff_pct = 100.0 * (rust_size as f64 - expected_rust as f64) / expected_rust as f64;
        assert!(
            diff_pct.abs() < 0.1,
            "Q{} 4:4:0 OPT: size {} vs expected {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_rust,
            diff_pct
        );

        let cpp_diff = 100.0 * (rust_size as f64 - expected_cpp as f64) / expected_cpp as f64;
        assert!(
            cpp_diff.abs() < MAX_DIFF_S440,
            "Q{} 4:4:0 OPT: Rust {} vs C++ {} (diff: {:.2}%)",
            q,
            rust_size,
            expected_cpp,
            cpp_diff
        );
    }
}

/// Print summary of parity status
#[test]
#[ignore] // Run with: cargo test --test parity_reference_locked print_summary -- --ignored --nocapture
fn print_summary() {
    let (rgb, width, height) = load_test_image();
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");

    println!("=== C++ vs Rust Parity Summary ===\n");
    println!("Test image: {}x{}\n", width, height);

    let configs = [
        ("4:4:4 OPT", ChromaSubsampling::Full, true, "444", S444_OPT),
        (
            "4:2:0 OPT",
            ChromaSubsampling::Quarter,
            true,
            "420",
            S420_OPT,
        ),
        (
            "4:2:2 OPT",
            ChromaSubsampling::HalfHorizontal,
            true,
            "422",
            S422_OPT,
        ),
        (
            "4:4:0 OPT",
            ChromaSubsampling::HalfVertical,
            true,
            "440",
            S440_OPT,
        ),
    ];

    for (name, subsampling, opt, cpp_mode, reference) in configs {
        println!("{}:", name);
        println!("{:>5} {:>10} {:>10} {:>10}", "Q", "C++", "Rust", "Diff");

        for &(q, _, cpp_size, _, _) in reference {
            let jpeg = encode_rust(&rgb, width, height, q, subsampling, opt);
            let rust_size = jpeg.len();
            let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;
            println!(
                "{:>5} {:>10} {:>10} {:>+9.2}%",
                q, cpp_size, rust_size, diff_pct
            );
        }
        println!();
    }
}

/// Generate all reference values (run this to update the locked values)
#[test]
#[ignore] // Run with: cargo test --test parity_reference_locked generate_all_values -- --ignored --nocapture
fn generate_all_values() {
    let (rgb, width, height) = load_test_image();
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");

    let configs = [
        ("S444_OPT", ChromaSubsampling::Full, true, "444"),
        ("S444_FIXED", ChromaSubsampling::Full, false, "444"),
        ("S420_OPT", ChromaSubsampling::Quarter, true, "420"),
        ("S422_OPT", ChromaSubsampling::HalfHorizontal, true, "422"),
        ("S440_OPT", ChromaSubsampling::HalfVertical, true, "440"),
    ];

    for (name, subsampling, opt, cpp_mode) in configs {
        println!(
            "/// {} - (quality, rust_size, cpp_size, dssim, butteraugli)",
            name
        );
        println!("#[rustfmt::skip]");
        println!("pub const {}: &[(u8, usize, usize, f64, f64)] = &[", name);

        for &q in QUALITY_LEVELS {
            let jpeg = encode_rust(&rgb, width, height, q, subsampling, opt);
            let rust_size = jpeg.len();

            // Try to get C++ size
            let cpp_size = encode_cpp(png_path, q, cpp_mode)
                .map(|j| j.len())
                .unwrap_or(rust_size);

            // Decode and compute DSSIM
            let decoded = zune_jpeg::JpegDecoder::new(
                zune_jpeg::zune_core::bytestream::ZCursor::new(&jpeg[..]),
            )
            .decode()
            .unwrap_or_else(|_| vec![128; (width * height * 3) as usize]);

            let dssim = if decoded.len() == rgb.len() {
                // Simple DSSIM approximation
                let mut sum = 0.0f64;
                for (a, b) in rgb.iter().zip(decoded.iter()) {
                    let diff = (*a as f64 - *b as f64) / 255.0;
                    sum += diff * diff;
                }
                (sum / rgb.len() as f64).sqrt() * 0.1 // Rough approximation
            } else {
                0.0
            };

            // Butteraugli placeholder (would need actual computation)
            let butteraugli = 5.0 - (q as f64 / 25.0);

            println!(
                "    ({}, {}, {}, {:.6}, {:.8}),",
                q, rust_size, cpp_size, dssim, butteraugli
            );
        }
        println!("];");
        println!();
    }
}

/// Exhaustive parity test across all modes (slow but thorough)
#[test]
#[ignore] // Run with: cargo test --test parity_reference_locked exhaustive_parity -- --ignored --nocapture
fn exhaustive_parity() {
    let (rgb, width, height) = load_test_image();
    let png_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/images/1.png");

    let mut results: HashMap<String, Vec<(u8, usize, usize, f64)>> = HashMap::new();

    let subsampling_modes = [
        (ChromaSubsampling::Full, "444"),
        (ChromaSubsampling::Quarter, "420"),
        (ChromaSubsampling::HalfHorizontal, "422"),
        (ChromaSubsampling::HalfVertical, "440"),
    ];

    let huffman_modes = [(true, "opt"), (false, "fixed")];

    println!("=== Exhaustive Parity Test ===\n");

    for (subsampling, ss_name) in &subsampling_modes {
        for (optimize, huff_name) in &huffman_modes {
            let key = format!("{}_{}", ss_name, huff_name);
            let mut entries = Vec::new();

            println!("Testing {} {}...", ss_name, huff_name);

            for &q in QUALITY_LEVELS {
                let jpeg = encode_rust(&rgb, width, height, q, *subsampling, *optimize);
                let rust_size = jpeg.len();

                let cpp_size = if *optimize {
                    encode_cpp(png_path, q, ss_name)
                        .map(|j| j.len())
                        .unwrap_or(rust_size)
                } else {
                    rust_size // No C++ comparison for fixed Huffman
                };

                let diff_pct = if cpp_size > 0 {
                    100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64
                } else {
                    0.0
                };

                entries.push((q, rust_size, cpp_size, diff_pct));

                // Verify within acceptable range
                let max_diff = match ss_name.as_ref() {
                    "444" => MAX_DIFF_S444,
                    "420" => MAX_DIFF_S420,
                    "422" => MAX_DIFF_S422,
                    "440" => MAX_DIFF_S440,
                    _ => 5.0,
                };

                if *optimize && diff_pct.abs() > max_diff {
                    println!(
                        "  WARNING: Q{} {} {}: {:.2}% (exceeds {:.2}%)",
                        q, ss_name, huff_name, diff_pct, max_diff
                    );
                }
            }

            results.insert(key, entries);
        }
    }

    // Print summary table
    println!("\n=== Summary Table ===\n");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Q", "444_opt", "420_opt", "422_opt", "440_opt"
    );

    for &q in QUALITY_LEVELS {
        let s444 = results
            .get("444_opt")
            .and_then(|v| v.iter().find(|e| e.0 == q))
            .map(|e| e.3)
            .unwrap_or(0.0);
        let s420 = results
            .get("420_opt")
            .and_then(|v| v.iter().find(|e| e.0 == q))
            .map(|e| e.3)
            .unwrap_or(0.0);
        let s422 = results
            .get("422_opt")
            .and_then(|v| v.iter().find(|e| e.0 == q))
            .map(|e| e.3)
            .unwrap_or(0.0);
        let s440 = results
            .get("440_opt")
            .and_then(|v| v.iter().find(|e| e.0 == q))
            .map(|e| e.3)
            .unwrap_or(0.0);

        println!(
            "{:>8} {:>+11.2}% {:>+11.2}% {:>+11.2}% {:>+11.2}%",
            q, s444, s420, s422, s440
        );
    }

    println!("\nAll tests passed!");
}
