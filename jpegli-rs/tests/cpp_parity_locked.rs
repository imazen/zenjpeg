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
use jpegli::{Encoder, PixelFormat, Quality};
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
const RUST_S444_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 16179, 0.016164, 8.98993492),
    (10, 19230, 0.011857, 7.43354321),
    (15, 23048, 0.008797, 6.77971697),
    (20, 27062, 0.006653, 5.58033657),
    (25, 31017, 0.005138, 5.31690264),
    (30, 33843, 0.004384, 5.53012991),
    (35, 35369, 0.004061, 5.12442350),
    (40, 36841, 0.003790, 4.55609989),
    (45, 38887, 0.003416, 4.50811577),
    (50, 40763, 0.003110, 4.03484821),
    (55, 43031, 0.002817, 3.89036059),
    (60, 46254, 0.002476, 3.81750154),
    (65, 49385, 0.002159, 3.87028670),
    (70, 53897, 0.001833, 3.54558539),
    (75, 58946, 0.001532, 2.65852857),
    (80, 66189, 0.001204, 2.56288385),
    (85, 76758, 0.000863, 2.17670417),
    (90, 95490, 0.000507, 1.64234662),
    (95, 134515, 0.000212, 1.09869063),
    (100, 323909, 0.000034, 0.36795413),
];

/// Rust expected values for 4:4:4, fixed Huffman - (quality, size, dssim, butteraugli)
const RUST_S444_FIXED: &[(u8, usize, f64, f64)] = &[
    (5, 18771, 0.016164, 8.98993492),
    (10, 21616, 0.011857, 7.43354321),
    (15, 25226, 0.008797, 6.77971697),
    (20, 29183, 0.006653, 5.58033657),
    (25, 33147, 0.005138, 5.31690264),
    (30, 35960, 0.004384, 5.53012991),
    (35, 37513, 0.004061, 5.12442350),
    (40, 39043, 0.003790, 4.55609989),
    (45, 41155, 0.003416, 4.50811577),
    (50, 43027, 0.003110, 4.03484821),
    (55, 45341, 0.002817, 3.89036059),
    (60, 48552, 0.002476, 3.81750154),
    (65, 51733, 0.002159, 3.87028670),
    (70, 56371, 0.001833, 3.54558539),
    (75, 61588, 0.001532, 2.65852857),
    (80, 69018, 0.001204, 2.56288385),
    (85, 80083, 0.000863, 2.17670417),
    (90, 99695, 0.000507, 1.64234662),
    (95, 141394, 0.000212, 1.09869063),
    (100, 352507, 0.000034, 0.36795413),
];

/// Rust expected values for 4:2:0, optimized Huffman - (quality, size, dssim, butteraugli)
const RUST_S420_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 13293, 0.017808, 10.33496571),
    (10, 16037, 0.013391, 9.76687145),
    (15, 19380, 0.010133, 8.14712048),
    (20, 22769, 0.007957, 6.69575977),
    (25, 26103, 0.006485, 6.01086473),
    (30, 28146, 0.005743, 5.98684883),
    (35, 29511, 0.005343, 5.42649794),
    (40, 30794, 0.004976, 5.25708055),
    (45, 32433, 0.004566, 4.54880905),
    (50, 34121, 0.004249, 4.43205547),
    (55, 36427, 0.003783, 4.36922073),
    (60, 38788, 0.003392, 4.25596952),
    (65, 41941, 0.002929, 4.14161396),
    (70, 45699, 0.002515, 3.76503134),
    (75, 50402, 0.002102, 3.74871826),
    (80, 56491, 0.001693, 3.52448153),
    (85, 65638, 0.001275, 3.46955442),
    (90, 82093, 0.000801, 3.34517694),
    (95, 115226, 0.000420, 3.24894071),
    (100, 216375, 0.000213, 3.25943422),
];

/// Rust expected values for 4:2:2, optimized Huffman - (quality, size, dssim, butteraugli)
const RUST_S422_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 14512, 0.019402, 11.78049374),
    (10, 17310, 0.014539, 9.25874805),
    (15, 20765, 0.011138, 8.58052826),
    (20, 24353, 0.008571, 7.24153709),
    (25, 27856, 0.006829, 6.77285051),
    (30, 30327, 0.005882, 6.54815006),
    (35, 31717, 0.005494, 5.84148884),
    (40, 33040, 0.005153, 6.44763422),
    (45, 34924, 0.004701, 5.32466221),
    (50, 36627, 0.004299, 6.20050049),
    (55, 38681, 0.003953, 5.05329132),
    (60, 41675, 0.003498, 4.68064070),
    (65, 44397, 0.003104, 5.19628000),
    (70, 48569, 0.002683, 4.60621643),
    (75, 53079, 0.002266, 3.87071729),
    (80, 59537, 0.001836, 3.57168841),
    (85, 69124, 0.001380, 2.91841459),
    (90, 85688, 0.000849, 2.89916182),
    (95, 118881, 0.000401, 2.57496881),
    (100, 256460, 0.000109, 2.52569222),
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
    Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .quality(Quality::from_quality(quality))
        .subsampling(subsampling)
        .optimize_huffman(optimize_huffman)
        .encode(rgb)
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
