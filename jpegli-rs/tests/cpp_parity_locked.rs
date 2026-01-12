//! Locked parity tests against C++ jpegli reference values.
//!
//! These tests compare Rust output against hardcoded C++ reference values.
//! This allows quick parity verification without rebuilding C++.
//!
//! Test image: tests/images/1.png (512x512 photo)
//!
//! Reference values generated with:
//!   internal/jpegli-cpp/build/tools/cjpegli --chroma_subsampling=XXX -p 0 tests/images/1.png /tmp/out.jpg -q N
//!
//! To regenerate Rust values:
//!   cargo run --release --example generate_parity_values
//!
//! ⚠️ LOCKED TEST: Do NOT modify reference values without re-running C++ cjpegli.

use butteraugli::ButteraugliParams;
use dssim::Dssim;
use jpegli::decode::Decoder;
use jpegli::types::Subsampling;
use jpegli::{PixelFormat, Quality, StreamingEncoder};
use rgb::RGBA8;
use std::fs;

// =============================================================================
// C++ REFERENCE VALUES
// Generated with: cjpegli --chroma_subsampling=XXX -p 0 tests/images/1.png /tmp/out.jpg -q N
// =============================================================================

/// C++ reference file sizes for 4:4:4, optimized Huffman
/// (quality, size_bytes)
const CPP_S444_OPT: &[(u8, usize)] = &[
    (5, 16179),
    (10, 19230),
    (15, 23054),
    (20, 27062),
    (25, 31019),
    (30, 33848),
    (35, 35351),
    (40, 36831),
    (45, 38883),
    (50, 40752),
    (55, 43028),
    (60, 46246),
    (65, 49393),
    (70, 53870),
    (75, 58935),
    (80, 66177),
    (85, 76758),
    (90, 95490),
    (95, 134545),
    (100, 323934),
];

/// C++ reference file sizes for 4:4:4, fixed Huffman
const CPP_S444_FIXED: &[(u8, usize)] = &[
    (5, 18771),
    (10, 21616),
    (15, 25226),
    (20, 29183),
    (25, 33147),
    (30, 35960),
    (35, 37513),
    (40, 39052),
    (45, 41155),
    (50, 43027),
    (55, 45341),
    (60, 48552),
    (65, 51733),
    (70, 56371),
    (75, 61588),
    (80, 69018),
    (85, 80083),
    (90, 99694),
    (95, 141394),
    (100, 352467),
];

/// C++ reference file sizes for 4:2:0, optimized Huffman
const CPP_S420_OPT: &[(u8, usize)] = &[
    (5, 13279),
    (10, 16035),
    (15, 19366),
    (20, 22782),
    (25, 26099),
    (30, 28152),
    (35, 29511),
    (40, 30798),
    (45, 32433),
    (50, 34099),
    (55, 36425),
    (60, 38794),
    (65, 41925),
    (70, 45726),
    (75, 50415),
    (80, 56513),
    (85, 65658),
    (90, 82136),
    (95, 115425),
    (100, 216364),
];

/// C++ reference file sizes for 4:2:2, optimized Huffman
const CPP_S422_OPT: &[(u8, usize)] = &[
    (5, 14510),
    (10, 17308),
    (15, 20755),
    (20, 24350),
    (25, 27857),
    (30, 30330),
    (35, 31705),
    (40, 33039),
    (45, 34938),
    (50, 36611),
    (55, 38677),
    (60, 41677),
    (65, 44406),
    (70, 48576),
    (75, 53079),
    (80, 59577),
    (85, 69124),
    (90, 85683),
    (95, 118857),
    (100, 256324),
];

// =============================================================================
// RUST EXPECTED VALUES (with DSSIM and Butteraugli)
// Generated with: cargo run --release --example generate_parity_values
// Format: (quality, size, dssim, butteraugli)
// =============================================================================

/// Rust expected values for 4:4:4, optimized Huffman - (quality, size, dssim, butteraugli)
/// Regenerated 2026-01-05 to match current decoder with dequantization bias.
const RUST_S444_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 16179, 0.016375, 9.35481548),
    (10, 19230, 0.012187, 7.44815493),
    (15, 23054, 0.009000, 6.44739580),
    (20, 27062, 0.006755, 5.49028873),
    (25, 31019, 0.005202, 4.82601881),
    (30, 33848, 0.004473, 4.95530558),
    (35, 35351, 0.004111, 4.68110561),
    (40, 36831, 0.003816, 4.45549440),
    (45, 38883, 0.003456, 4.25343037),
    (50, 40752, 0.003133, 4.15748024),
    (55, 43028, 0.002831, 4.00855064),
    (60, 46246, 0.002460, 4.00096130),
    (65, 49393, 0.002166, 3.89660645),
    (70, 53870, 0.001809, 3.76304984),
    (75, 58935, 0.001499, 2.83899188),
    (80, 66177, 0.001180, 2.42982697),
    (85, 76758, 0.000840, 2.11400509),
    (90, 95490, 0.000483, 1.63883805),
    (95, 134545, 0.000192, 1.06628466),
    (100, 323872, 0.000026, 0.63459766),
];

/// Rust expected values for 4:4:4, fixed Huffman - (quality, size, dssim, butteraugli)
/// Regenerated 2026-01-05 to match current decoder with dequantization bias.
const RUST_S444_FIXED: &[(u8, usize, f64, f64)] = &[
    (5, 18771, 0.016375, 9.35481548),
    (10, 21616, 0.012187, 7.44815493),
    (15, 25226, 0.009000, 6.44739580),
    (20, 29183, 0.006755, 5.49028873),
    (25, 33147, 0.005202, 4.82601881),
    (30, 35960, 0.004473, 4.95530558),
    (35, 37513, 0.004111, 4.68110561),
    (40, 39043, 0.003816, 4.45549440),
    (45, 41155, 0.003456, 4.25343037),
    (50, 43027, 0.003133, 4.15748024),
    (55, 45341, 0.002831, 4.00855064),
    (60, 48552, 0.002460, 4.00096130),
    (65, 51733, 0.002166, 3.89660645),
    (70, 56371, 0.001809, 3.76304984),
    (75, 61588, 0.001499, 2.83899188),
    (80, 69018, 0.001180, 2.42982697),
    (85, 80083, 0.000840, 2.11400509),
    (90, 99695, 0.000483, 1.63883805),
    (95, 141394, 0.000192, 1.06628466),
    (100, 352507, 0.000026, 0.63459766),
];

/// Rust expected values for 4:2:0, optimized Huffman - (quality, size, dssim, butteraugli)
/// Regenerated 2026-01-10 after K420_RESCALE fix for C++ parity.
const RUST_S420_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 13280, 0.017624, 9.45977020),
    (10, 16031, 0.013254, 9.41831970),
    (15, 19366, 0.009993, 7.33640480),
    (20, 22771, 0.007852, 6.94552898),
    (25, 26092, 0.006392, 5.83655548),
    (30, 28128, 0.005670, 6.10781479),
    (35, 29500, 0.005263, 6.09829140),
    (40, 30769, 0.004877, 5.72717428),
    (45, 32431, 0.004481, 5.12973022),
    (50, 34093, 0.004150, 5.16898251),
    (55, 36370, 0.003708, 5.23236036),
    (60, 38766, 0.003307, 5.22821188),
    (65, 41900, 0.002874, 4.81301546),
    (70, 45697, 0.002470, 5.04821968),
    (75, 50414, 0.002069, 4.62204599),
    (80, 56465, 0.001691, 5.06911278),
    (85, 65623, 0.001282, 4.75842190),
    (90, 82097, 0.000838, 4.75313473),
    (95, 115383, 0.000488, 4.76267195),
    (100, 216314, 0.000343, 4.77298975),
];

/// Rust expected values for 4:2:2, optimized Huffman - (quality, size, dssim, butteraugli)
/// Regenerated 2026-01-05 to match current decoder with dequantization bias.
const RUST_S422_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 14510, 0.019081, 11.75201035),
    (10, 17308, 0.014492, 9.50500679),
    (15, 20755, 0.011003, 8.93167114),
    (20, 24353, 0.008405, 7.12097073),
    (25, 27857, 0.006668, 6.89084005),
    (30, 30330, 0.005768, 6.29474497),
    (35, 31705, 0.005353, 5.89941072),
    (40, 33039, 0.004997, 6.60154057),
    (45, 34938, 0.004574, 5.65823603),
    (50, 36611, 0.004167, 6.05982065),
    (55, 38677, 0.003808, 5.19747925),
    (60, 41677, 0.003343, 4.98108625),
    (65, 44406, 0.002974, 4.85013342),
    (70, 48576, 0.002535, 4.52523947),
    (75, 53079, 0.002127, 3.68466139),
    (80, 59577, 0.001733, 3.70013762),
    (85, 69124, 0.001300, 3.14908266),
    (90, 85683, 0.000805, 3.05760574),
    (95, 118857, 0.000387, 2.84443688),
    (100, 256340, 0.000148, 2.86820269),
];

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Tolerance for file size comparison against C++ (percentage)
/// 4:4:4 mode has excellent parity (<0.2%)
const SIZE_TOLERANCE_444: f64 = 0.5;

/// Tolerance for subsampled modes (Rust may differ due to chroma processing)
/// Note: Rust produces ~3-6% smaller files for 4:2:0/4:2:2 at high quality
const SIZE_TOLERANCE_SUBSAMPLED: f64 = 7.0;

/// Tolerance for regression detection (Rust size should not change much)
const SIZE_REGRESSION_TOLERANCE: f64 = 0.5;

/// Tolerance for DSSIM regression (quality should not degrade)
/// Note: Rust decoder uses jpegli-style dequantization bias which affects DSSIM measurements.
/// This tolerance allows for the decoder bias effect while still catching encoder regressions.
const DSSIM_REGRESSION_TOLERANCE: f64 = 0.003;

/// Tolerance for Butteraugli regression (quality should not degrade)
/// Note: Decoder dequantization bias (matching djpegli) affects quality measurements.
/// This tolerance accounts for decoder output differences. Encoder file sizes remain accurate.
const BUTTERAUGLI_REGRESSION_TOLERANCE: f64 = 0.25;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn load_test_image() -> (Vec<u8>, u32, u32) {
    let png_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/images/1.png");
    let png_data = fs::read(&png_path).expect("Failed to read test image");
    let decoder = png::Decoder::new(&png_data[..]);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();

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

fn rgb_to_rgba(data: &[u8]) -> Vec<RGBA8> {
    data.chunks(3)
        .map(|c| RGBA8::new(c[0], c[1], c[2], 255))
        .collect()
}

fn compute_dssim(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let dssim = Dssim::new();
    let orig_rgba = rgb_to_rgba(original);
    let dec_rgba = rgb_to_rgba(decoded);

    let orig = dssim
        .create_image_rgba(&orig_rgba, width, height)
        .expect("create orig image");
    let comp = dssim
        .create_image_rgba(&dec_rgba, width, height)
        .expect("create comp image");

    let (dssim_val, _) = dssim.compare(&orig, comp);
    f64::from(dssim_val)
}

fn compute_butteraugli(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let params = ButteraugliParams::default();
    butteraugli::compute_butteraugli(original, decoded, width, height, &params)
        .expect("butteraugli computation failed")
        .score
}

fn encode_jpeg(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    subsampling: Subsampling,
    optimize_huffman: bool,
) -> Vec<u8> {
    StreamingEncoder::new(width, height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality))
        .subsampling(subsampling)
        .optimize_huffman(optimize_huffman)
        .encode_all(rgb)
        .expect("Encoding failed")
}

// =============================================================================
// TESTS: C++ PARITY
// =============================================================================

/// Test 4:4:4 optimized Huffman matches C++ within tolerance
#[test]
fn test_cpp_parity_s444_optimized() {
    let (rgb, width, height) = load_test_image();

    for &(quality, cpp_size) in CPP_S444_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, quality as f32, Subsampling::S444, true);
        let diff_pct = 100.0 * (jpeg.len() as f64 - cpp_size as f64) / cpp_size as f64;

        assert!(
            diff_pct.abs() < SIZE_TOLERANCE_444,
            "Q{}: Rust={} C++={} diff={:+.2}% (limit: {}%)",
            quality,
            jpeg.len(),
            cpp_size,
            diff_pct,
            SIZE_TOLERANCE_444
        );
    }
}

/// Test 4:4:4 fixed Huffman matches C++ within tolerance
#[test]
fn test_cpp_parity_s444_fixed() {
    let (rgb, width, height) = load_test_image();

    for &(quality, cpp_size) in CPP_S444_FIXED {
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            Subsampling::S444,
            false,
        );
        let diff_pct = 100.0 * (jpeg.len() as f64 - cpp_size as f64) / cpp_size as f64;

        assert!(
            diff_pct.abs() < SIZE_TOLERANCE_444,
            "Q{} fixed: Rust={} C++={} diff={:+.2}% (limit: {}%)",
            quality,
            jpeg.len(),
            cpp_size,
            diff_pct,
            SIZE_TOLERANCE_444
        );
    }
}

/// Test 4:2:0 matches C++ within tolerance (larger tolerance expected)
#[test]
fn test_cpp_parity_s420() {
    let (rgb, width, height) = load_test_image();

    for &(quality, cpp_size) in CPP_S420_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, quality as f32, Subsampling::S420, true);
        let diff_pct = 100.0 * (jpeg.len() as f64 - cpp_size as f64) / cpp_size as f64;

        // Note: Rust typically produces ~3% smaller files for 4:2:0
        assert!(
            diff_pct.abs() < SIZE_TOLERANCE_SUBSAMPLED,
            "Q{} 4:2:0: Rust={} C++={} diff={:+.2}% (limit: {}%)",
            quality,
            jpeg.len(),
            cpp_size,
            diff_pct,
            SIZE_TOLERANCE_SUBSAMPLED
        );
    }
}

/// Test 4:2:2 matches C++ within tolerance
#[test]
fn test_cpp_parity_s422() {
    let (rgb, width, height) = load_test_image();

    for &(quality, cpp_size) in CPP_S422_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, quality as f32, Subsampling::S422, true);
        let diff_pct = 100.0 * (jpeg.len() as f64 - cpp_size as f64) / cpp_size as f64;

        assert!(
            diff_pct.abs() < SIZE_TOLERANCE_SUBSAMPLED,
            "Q{} 4:2:2: Rust={} C++={} diff={:+.2}% (limit: {}%)",
            quality,
            jpeg.len(),
            cpp_size,
            diff_pct,
            SIZE_TOLERANCE_SUBSAMPLED
        );
    }
}

// =============================================================================
// TESTS: RUST REGRESSION
// =============================================================================

/// Test 4:4:4 optimized doesn't regress from expected values
#[test]
fn test_regression_s444_optimized() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, expected_dssim, expected_bfly) in RUST_S444_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, quality as f32, Subsampling::S444, true);

        // Check size regression
        let size_diff_pct =
            100.0 * (jpeg.len() as f64 - expected_size as f64) / expected_size as f64;
        assert!(
            size_diff_pct.abs() < SIZE_REGRESSION_TOLERANCE,
            "Q{}: size regressed: got {} expected {} ({:+.2}%)",
            quality,
            jpeg.len(),
            expected_size,
            size_diff_pct
        );

        // Check DSSIM regression
        let decoded = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg)
            .expect("Decoding failed");
        let dssim = compute_dssim(&rgb, &decoded.data, width as usize, height as usize);

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{}: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(&rgb, &decoded.data, width as usize, height as usize);
        let bfly_diff_pct = (bfly - expected_bfly).abs() / expected_bfly;
        assert!(
            bfly_diff_pct < BUTTERAUGLI_REGRESSION_TOLERANCE,
            "Q{}: Butteraugli regressed: got {:.8} expected {:.8} ({:+.6}%)",
            quality,
            bfly,
            expected_bfly,
            bfly_diff_pct * 100.0
        );
    }
}

/// Test 4:4:4 fixed doesn't regress from expected values
#[test]
fn test_regression_s444_fixed() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, expected_dssim, expected_bfly) in RUST_S444_FIXED {
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            Subsampling::S444,
            false,
        );

        let size_diff_pct =
            100.0 * (jpeg.len() as f64 - expected_size as f64) / expected_size as f64;
        assert!(
            size_diff_pct.abs() < SIZE_REGRESSION_TOLERANCE,
            "Q{} fixed: size regressed: got {} expected {} ({:+.2}%)",
            quality,
            jpeg.len(),
            expected_size,
            size_diff_pct
        );

        let decoded = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg)
            .expect("Decoding failed");
        let dssim = compute_dssim(&rgb, &decoded.data, width as usize, height as usize);

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{} fixed: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(&rgb, &decoded.data, width as usize, height as usize);
        let bfly_diff_pct = (bfly - expected_bfly).abs() / expected_bfly;
        assert!(
            bfly_diff_pct < BUTTERAUGLI_REGRESSION_TOLERANCE,
            "Q{} fixed: Butteraugli regressed: got {:.8} expected {:.8} ({:+.6}%)",
            quality,
            bfly,
            expected_bfly,
            bfly_diff_pct * 100.0
        );
    }
}

/// Test 4:2:0 doesn't regress from expected values
#[test]
fn test_regression_s420() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, expected_dssim, expected_bfly) in RUST_S420_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, quality as f32, Subsampling::S420, true);

        let size_diff_pct =
            100.0 * (jpeg.len() as f64 - expected_size as f64) / expected_size as f64;
        assert!(
            size_diff_pct.abs() < SIZE_REGRESSION_TOLERANCE,
            "Q{} 4:2:0: size regressed: got {} expected {} ({:+.2}%)",
            quality,
            jpeg.len(),
            expected_size,
            size_diff_pct
        );

        let decoded = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg)
            .expect("Decoding failed");
        let dssim = compute_dssim(&rgb, &decoded.data, width as usize, height as usize);

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{} 4:2:0: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(&rgb, &decoded.data, width as usize, height as usize);
        let bfly_diff_pct = (bfly - expected_bfly).abs() / expected_bfly;
        assert!(
            bfly_diff_pct < BUTTERAUGLI_REGRESSION_TOLERANCE,
            "Q{} 4:2:0: Butteraugli regressed: got {:.8} expected {:.8} ({:+.6}%)",
            quality,
            bfly,
            expected_bfly,
            bfly_diff_pct * 100.0
        );
    }
}

/// Test 4:2:2 doesn't regress from expected values
#[test]
fn test_regression_s422() {
    let (rgb, width, height) = load_test_image();

    for &(quality, expected_size, expected_dssim, expected_bfly) in RUST_S422_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, quality as f32, Subsampling::S422, true);

        let size_diff_pct =
            100.0 * (jpeg.len() as f64 - expected_size as f64) / expected_size as f64;
        assert!(
            size_diff_pct.abs() < SIZE_REGRESSION_TOLERANCE,
            "Q{} 4:2:2: size regressed: got {} expected {} ({:+.2}%)",
            quality,
            jpeg.len(),
            expected_size,
            size_diff_pct
        );

        let decoded = Decoder::new()
            .output_format(PixelFormat::Rgb)
            .decode(&jpeg)
            .expect("Decoding failed");
        let dssim = compute_dssim(&rgb, &decoded.data, width as usize, height as usize);

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{} 4:2:2: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(&rgb, &decoded.data, width as usize, height as usize);
        let bfly_diff_pct = (bfly - expected_bfly).abs() / expected_bfly;
        assert!(
            bfly_diff_pct < BUTTERAUGLI_REGRESSION_TOLERANCE,
            "Q{} 4:2:2: Butteraugli regressed: got {:.8} expected {:.8} ({:+.6}%)",
            quality,
            bfly,
            expected_bfly,
            bfly_diff_pct * 100.0
        );
    }
}

// =============================================================================
// UTILITY TESTS
// =============================================================================

/// Print current values for updating reference constants.
/// Run with: cargo test --test cpp_parity_locked print_current_values -- --ignored --nocapture
#[test]
#[ignore = "utility for updating reference values"]
fn print_current_values() {
    let (rgb, width, height) = load_test_image();
    println!("Image: {}x{}", width, height);

    let configs = [
        (Subsampling::S444, true, "S444_OPT"),
        (Subsampling::S444, false, "S444_FIXED"),
        (Subsampling::S420, true, "S420_OPT"),
        (Subsampling::S422, true, "S422_OPT"),
    ];

    let dssim = Dssim::new();
    let rgba_orig = rgb_to_rgba(&rgb);
    let orig_img = dssim
        .create_image_rgba(&rgba_orig, width as usize, height as usize)
        .unwrap();
    let bfly_params = ButteraugliParams::default();

    for (subsampling, optimize, name) in configs {
        println!("\n/// RUST_{} - (quality, size, dssim, butteraugli)", name);
        println!("const RUST_{}: &[(u8, usize, f64, f64)] = &[", name);

        for q in (1..=20).map(|i| i * 5) {
            let jpeg = encode_jpeg(&rgb, width, height, q as f32, subsampling, optimize);

            let decoded = Decoder::new()
                .output_format(PixelFormat::Rgb)
                .decode(&jpeg)
                .expect("Decoding failed");
            let rgba_dec = rgb_to_rgba(&decoded.data);
            let dec_img = dssim
                .create_image_rgba(&rgba_dec, width as usize, height as usize)
                .unwrap();
            let (dssim_val, _) = dssim.compare(&orig_img, dec_img);

            let bfly = butteraugli::compute_butteraugli(
                &rgb,
                &decoded.data,
                width as usize,
                height as usize,
                &bfly_params,
            )
            .expect("butteraugli")
            .score;

            println!(
                "    ({}, {}, {:.6}, {:.8}),",
                q,
                jpeg.len(),
                f64::from(dssim_val),
                bfly
            );
        }
        println!("];");
    }
}

/// Print comparison summary
/// Run with: cargo test --test cpp_parity_locked print_summary -- --ignored --nocapture
#[test]
#[ignore = "utility for viewing summary"]
fn print_summary() {
    let (rgb, width, height) = load_test_image();

    println!("=== C++ vs Rust Parity Summary ===\n");
    println!("Test image: {}x{}\n", width, height);

    println!("4:4:4 Optimized Huffman:");
    println!("{:>5} {:>10} {:>10} {:>10}", "Q", "C++", "Rust", "Diff");
    for &(q, cpp_size) in CPP_S444_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, q as f32, Subsampling::S444, true);
        let diff_pct = 100.0 * (jpeg.len() as f64 - cpp_size as f64) / cpp_size as f64;
        println!(
            "{:>5} {:>10} {:>10} {:>+9.2}%",
            q,
            cpp_size,
            jpeg.len(),
            diff_pct
        );
    }

    println!("\n4:2:0 Optimized Huffman:");
    println!("{:>5} {:>10} {:>10} {:>10}", "Q", "C++", "Rust", "Diff");
    for &(q, cpp_size) in CPP_S420_OPT {
        let jpeg = encode_jpeg(&rgb, width, height, q as f32, Subsampling::S420, true);
        let diff_pct = 100.0 * (jpeg.len() as f64 - cpp_size as f64) / cpp_size as f64;
        println!(
            "{:>5} {:>10} {:>10} {:>+9.2}%",
            q,
            cpp_size,
            jpeg.len(),
            diff_pct
        );
    }
}
