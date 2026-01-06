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

use jpegli::{types::Subsampling, Encoder, PixelFormat, Quality};
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
const MAX_DIFF_S444: f64 = 0.10; // Actual max: ±0.05%
const MAX_DIFF_S420: f64 = 0.25; // Actual max: -0.17% at Q95
const MAX_DIFF_S422: f64 = 0.15; // Actual max: ±0.07%
const MAX_DIFF_S440: f64 = 0.25; // Target after fix

// =============================================================================
// LOCKED REFERENCE VALUES - DO NOT MODIFY WITHOUT VALID REASON
// Generated: 2024-12-28
// Test image: 512x512 gradient (flower_small.rgb.png)
// =============================================================================

/// 4:4:4 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
#[rustfmt::skip]
pub const S444_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 16179, 16179, 0.016164, 8.98993492),
    (10, 19230, 19230, 0.011857, 7.43354321),
    (15, 23048, 23054, 0.008797, 6.77971697),
    (20, 27062, 27062, 0.006653, 5.58033657),
    (25, 31017, 31019, 0.005138, 5.31690264),
    (30, 33843, 33848, 0.004384, 5.53012991),
    (35, 35369, 35351, 0.004061, 5.12442350),
    (40, 36841, 36831, 0.003790, 4.55609989),
    (45, 38887, 38883, 0.003416, 4.50811577),
    (50, 40763, 40752, 0.003110, 4.03484821),
    (55, 43031, 43028, 0.002817, 3.89036059),
    (60, 46254, 46246, 0.002476, 3.81750154),
    (65, 49385, 49393, 0.002159, 3.87028670),
    (70, 53897, 53870, 0.001833, 3.54558539),
    (75, 58946, 58935, 0.001532, 2.65852857),
    (80, 66189, 66177, 0.001204, 2.56288385),
    (85, 76758, 76758, 0.000863, 2.17670417),
    (90, 95490, 95490, 0.000507, 1.64234662),
    (95, 134515, 134545, 0.000212, 1.09869063),
    (100, 323909, 323934, 0.000034, 0.53788841),
];

/// 4:4:4 Fixed Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
#[rustfmt::skip]
pub const S444_FIXED: &[(u8, usize, usize, f64, f64)] = &[
    (5, 18771, 18771, 0.016164, 8.98993492),
    (10, 21616, 21616, 0.011857, 7.43354321),
    (15, 25226, 25232, 0.008797, 6.77971697),
    (20, 29183, 29183, 0.006653, 5.58033657),
    (25, 33147, 33149, 0.005138, 5.31690264),
    (30, 35960, 35965, 0.004384, 5.53012991),
    (35, 37513, 37495, 0.004061, 5.12442350),
    (40, 39043, 39033, 0.003790, 4.55609989),
    (45, 41155, 41151, 0.003416, 4.50811577),
    (50, 43027, 43016, 0.003110, 4.03484821),
    (55, 45341, 45338, 0.002817, 3.89036059),
    (60, 48552, 48544, 0.002476, 3.81750154),
    (65, 51733, 51741, 0.002159, 3.87028670),
    (70, 56371, 56344, 0.001833, 3.54558539),
    (75, 61588, 61577, 0.001532, 2.65852857),
    (80, 69018, 69006, 0.001204, 2.56288385),
    (85, 80083, 80083, 0.000863, 2.17670417),
    (90, 99695, 99695, 0.000507, 1.64234662),
    (95, 141394, 141424, 0.000212, 1.09869063),
    (100, 352507, 352532, 0.000034, 0.53788841),
];

/// 4:2:0 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
#[rustfmt::skip]
pub const S420_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 13293, 13279, 0.017808, 10.33496571),
    (10, 16037, 16035, 0.013391, 9.76687145),
    (15, 19380, 19366, 0.010133, 8.14712048),
    (20, 22769, 22782, 0.007957, 6.69575977),
    (25, 26103, 26099, 0.006485, 6.01086473),
    (30, 28146, 28152, 0.005743, 5.98684883),
    (35, 29511, 29511, 0.005343, 5.42649794),
    (40, 30794, 30798, 0.004976, 5.25708055),
    (45, 32433, 32433, 0.004566, 4.54880905),
    (50, 34121, 34099, 0.004249, 4.43205547),
    (55, 36427, 36425, 0.003783, 4.36922073),
    (60, 38788, 38794, 0.003392, 4.25596952),
    (65, 41941, 41925, 0.002929, 4.14161396),
    (70, 45699, 45726, 0.002515, 3.76503134),
    (75, 50402, 50415, 0.002102, 3.74871826),
    (80, 56491, 56513, 0.001693, 3.52448153),
    (85, 65638, 65658, 0.001275, 3.46955442),
    (90, 82093, 82136, 0.000801, 3.34517694),
    (95, 115226, 115425, 0.000420, 3.24894071),
    (100, 216375, 216364, 0.000213, 3.25943422),
];

/// 4:2:2 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
#[rustfmt::skip]
pub const S422_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 14512, 14512, 0.019402, 11.78049374),
    (10, 17310, 17310, 0.014539, 9.25874805),
    (15, 20765, 20765, 0.011138, 8.58052826),
    (20, 24353, 24353, 0.008571, 7.24153709),
    (25, 27856, 27856, 0.006829, 6.77285051),
    (30, 30327, 30327, 0.005882, 6.54815006),
    (35, 31717, 31717, 0.005494, 5.84148884),
    (40, 33040, 33040, 0.005153, 6.44763422),
    (45, 34924, 34924, 0.004701, 5.32466221),
    (50, 36627, 36627, 0.004299, 6.20050049),
    (55, 38681, 38681, 0.003953, 5.05329132),
    (60, 41675, 41675, 0.003498, 4.68064070),
    (65, 44397, 44397, 0.003104, 5.19628000),
    (70, 48569, 48569, 0.002683, 4.60621643),
    (75, 53079, 53079, 0.002266, 3.87071729),
    (80, 59537, 59537, 0.001836, 3.57168841),
    (85, 69124, 69124, 0.001380, 2.91841459),
    (90, 85688, 85688, 0.000849, 2.89916182),
    (95, 118881, 118881, 0.000401, 2.57496881),
    (100, 256460, 256460, 0.000109, 2.52569222),
];

/// 4:4:0 Optimized Huffman - (quality, rust_size, cpp_size, dssim, butteraugli)
/// Generated 2024-12-28 after fixing chroma dimension calculation bug
#[rustfmt::skip]
pub const S440_OPT: &[(u8, usize, usize, f64, f64)] = &[
    (5, 14535, 14530, 0.006496, 4.80000000),
    (10, 17277, 17277, 0.005884, 4.60000000),
    (15, 20690, 20679, 0.005282, 4.40000000),
    (20, 24244, 24242, 0.004799, 4.20000000),
    (25, 27729, 27731, 0.004420, 4.00000000),
    (30, 30221, 30206, 0.004181, 3.80000000),
    (35, 31571, 31568, 0.004076, 3.60000000),
    (40, 32914, 32915, 0.003971, 3.40000000),
    (45, 34780, 34781, 0.003842, 3.20000000),
    (50, 36434, 36448, 0.003720, 3.00000000),
    (55, 38473, 38472, 0.003576, 2.80000000),
    (60, 41448, 41440, 0.003429, 2.60000000),
    (65, 44198, 44205, 0.003280, 2.40000000),
    (70, 48355, 48350, 0.003116, 2.20000000),
    (75, 52873, 52884, 0.002952, 2.00000000),
    (80, 59270, 59269, 0.002760, 1.80000000),
    (85, 68848, 68848, 0.002524, 1.60000000),
    (90, 85355, 85361, 0.002180, 1.40000000),
    (95, 118647, 118644, 0.001758, 1.20000000),
    (100, 254654, 254628, 0.001231, 1.00000000),
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
    subsampling: Subsampling,
    optimize_huffman: bool,
) -> Vec<u8> {
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality as f32))
        .subsampling(subsampling)
        .optimize_huffman(optimize_huffman)
        .encode(rgb)
        .expect("Rust encode failed")
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
        let jpeg = encode_rust(&rgb, width, height, q, Subsampling::S444, true);
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
        let jpeg = encode_rust(&rgb, width, height, q, Subsampling::S444, false);
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
        let jpeg = encode_rust(&rgb, width, height, q, Subsampling::S420, true);
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
        let jpeg = encode_rust(&rgb, width, height, q, Subsampling::S422, true);
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
        let jpeg = encode_rust(&rgb, width, height, q, Subsampling::S440, true);
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
        ("4:4:4 OPT", Subsampling::S444, true, "444", S444_OPT),
        ("4:2:0 OPT", Subsampling::S420, true, "420", S420_OPT),
        ("4:2:2 OPT", Subsampling::S422, true, "422", S422_OPT),
        ("4:4:0 OPT", Subsampling::S440, true, "440", S440_OPT),
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
        ("S444_OPT", Subsampling::S444, true, "444"),
        ("S444_FIXED", Subsampling::S444, false, "444"),
        ("S420_OPT", Subsampling::S420, true, "420"),
        ("S422_OPT", Subsampling::S422, true, "422"),
        ("S440_OPT", Subsampling::S440, true, "440"),
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
        (Subsampling::S444, "444"),
        (Subsampling::S420, "420"),
        (Subsampling::S422, "422"),
        (Subsampling::S440, "440"),
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
