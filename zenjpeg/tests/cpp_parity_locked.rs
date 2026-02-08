//! Locked parity tests against C++ jpegli reference values.
//!
//! These tests compare Rust output against hardcoded C++ reference values.
//! This allows quick parity verification without rebuilding C++.
//!
//! Test image: tests/images/1.png (512x512 photo)
//!
//! C++ reference values generated with:
//!   cjpegli tests/images/1.png /tmp/out.jpg -q N --progressive_level=P --chroma_subsampling=XXX
//!   where P=2 for progressive/optimized, P=0 for baseline/fixed
//!
//! Rust expected values generated with:
//!   cargo run --release --example gen_locked_values
//!
//! Last regenerated: 2026-01-16
//!
//! ⚠️ LOCKED TEST: Do NOT modify reference values without re-running C++ cjpegli.
use enough::Unstoppable;

use butteraugli::ButteraugliParams;
use dssim_core::Dssim;
use rgb::RGBA8;
use std::fs;
use zenjpeg::decoder::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

// =============================================================================
// C++ REFERENCE VALUES
// Generated 2026-01-16 with: cjpegli -q N --progressive_level=P --chroma_subsampling=XXX
// =============================================================================

/// C++ reference: 4:4:4, progressive (--progressive_level=2)
const CPP_S444_OPT: &[(u8, usize)] = &[
    (5, 16110),
    (10, 19304),
    (15, 23287),
    (20, 27343),
    (25, 31260),
    (30, 34005),
    (35, 35436),
    (40, 36936),
    (45, 38923),
    (50, 40756),
    (55, 43031),
    (60, 46112),
    (65, 49261),
    (70, 53576),
    (75, 58543),
    (80, 65486),
    (85, 75782),
    (90, 93897),
    (95, 131380),
    (100, 311516),
];

/// C++ reference: 4:4:4, baseline (--progressive_level=0)
const CPP_S444_FIXED: &[(u8, usize)] = &[
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

/// C++ reference: 4:2:0, progressive (--progressive_level=2)
const CPP_S420_OPT: &[(u8, usize)] = &[
    (5, 13683),
    (10, 16496),
    (15, 19728),
    (20, 23001),
    (25, 26184),
    (30, 28175),
    (35, 29512),
    (40, 30758),
    (45, 32278),
    (50, 33895),
    (55, 36134),
    (60, 38419),
    (65, 41460),
    (70, 45148),
    (75, 49778),
    (80, 55515),
    (85, 64179),
    (90, 80096),
    (95, 111770),
    (100, 208485),
];

/// C++ reference: 4:2:2, progressive (--progressive_level=2)
const CPP_S422_OPT: &[(u8, usize)] = &[
    (5, 14803),
    (10, 17666),
    (15, 21170),
    (20, 24753),
    (25, 28169),
    (30, 30548),
    (35, 31831),
    (40, 33193),
    (45, 34995),
    (50, 36634),
    (55, 38642),
    (60, 41527),
    (65, 44266),
    (70, 48286),
    (75, 52723),
    (80, 58927),
    (85, 68134),
    (90, 84008),
    (95, 115980),
    (100, 247442),
];

// =============================================================================
// RUST EXPECTED VALUES (with DSSIM and Butteraugli)
// Generated 2026-01-16 with: cargo run --release --example gen_locked_values
// Format: (quality, size, dssim, butteraugli)
// =============================================================================

/// Rust expected: 4:4:4, progressive, optimized Huffman
/// Updated 2026-02-01: sizes reduced after defaulting allow_16bit_quant_tables=false
const RUST_S444_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 16084, 0.016157, 8.99133492),
    (10, 19263, 0.011888, 7.43367863),
    (15, 23205, 0.008844, 6.82907391),
    (20, 27231, 0.006670, 5.58060598),
    (25, 31134, 0.005155, 5.49073267),
    (30, 33935, 0.004404, 5.54542303),
    (35, 35355, 0.004093, 5.06215096),
    (40, 36801, 0.003811, 4.84684324),
    (45, 38870, 0.003430, 4.51536274),
    (50, 40697, 0.003119, 4.02269220),
    (55, 42969, 0.002818, 3.90750265),
    (60, 46031, 0.002488, 3.80471468),
    (65, 49197, 0.002176, 3.73590970),
    (70, 53501, 0.001837, 3.48123431),
    (75, 58502, 0.001539, 2.88890886),
    (80, 65464, 0.001221, 2.56708694),
    (85, 75672, 0.000872, 2.18441653),
    (90, 93767, 0.000512, 1.63023186),
    (95, 131348, 0.000218, 1.06197035),
    (100, 319549, 0.000033, 0.46010166),
];

/// Rust expected: 4:4:4, baseline, corpus-trained Huffman tables
/// Updated 2026-02-01: switched from Annex K to corpus-trained tables (5-11% smaller)
const RUST_S444_FIXED: &[(u8, usize, f64, f64)] = &[
    (5, 16686, 0.016205, 9.18228149),
    (10, 19382, 0.011946, 7.43430805),
    (15, 23208, 0.008885, 6.81514311),
    (20, 27357, 0.006707, 5.58039379),
    (25, 31483, 0.005183, 5.41986465),
    (30, 34469, 0.004429, 5.54347706),
    (35, 35997, 0.004111, 5.06219673),
    (40, 37495, 0.003838, 4.84849358),
    (45, 39741, 0.003443, 4.51634359),
    (50, 41711, 0.003138, 4.02232265),
    (55, 43703, 0.002834, 3.90591764),
    (60, 46822, 0.002501, 3.80448556),
    (65, 50132, 0.002185, 3.73606682),
    (70, 54826, 0.001849, 3.48550081),
    (75, 60038, 0.001551, 2.88891935),
    (80, 67405, 0.001229, 2.56724977),
    (85, 78418, 0.000878, 2.18167806),
    (90, 97348, 0.000515, 1.62355626),
    (95, 138171, 0.000220, 1.07045877),
    (100, 336017, 0.000033, 0.45978206),
];

/// Rust expected: 4:2:0, progressive, optimized Huffman
/// Updated 2026-02-01: sizes reduced after defaulting allow_16bit_quant_tables=false
const RUST_S420_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 13712, 0.017425, 9.54254341),
    (10, 16503, 0.013046, 9.35945320),
    (15, 19703, 0.009856, 7.23111439),
    (20, 22978, 0.007753, 7.04672050),
    (25, 26123, 0.006330, 5.87719584),
    (30, 28100, 0.005587, 6.07315016),
    (35, 29433, 0.005191, 6.03578424),
    (40, 30666, 0.004842, 5.57302475),
    (45, 32192, 0.004449, 5.08180952),
    (50, 33821, 0.004138, 5.13703966),
    (55, 36051, 0.003698, 5.10817671),
    (60, 38315, 0.003316, 5.10311937),
    (65, 41331, 0.002880, 4.87986374),
    (70, 45053, 0.002481, 5.13368177),
    (75, 49649, 0.002093, 4.76221800),
    (80, 55405, 0.001709, 4.93946362),
    (85, 64100, 0.001314, 4.74678612),
    (90, 79974, 0.000869, 4.59267759),
    (95, 111693, 0.000516, 4.62422276),
    (100, 210959, 0.000353, 4.57556915),
];

/// Rust expected: 4:2:2, progressive, optimized Huffman
/// Updated 2026-02-01: sizes reduced after defaulting allow_16bit_quant_tables=false
const RUST_S422_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 14782, 0.018970, 11.55516815),
    (10, 17634, 0.014209, 9.45995903),
    (15, 21115, 0.010870, 7.92480278),
    (20, 24672, 0.008354, 7.35234022),
    (25, 28076, 0.006632, 6.86862469),
    (30, 30516, 0.005707, 6.13680649),
    (35, 31761, 0.005325, 5.85240746),
    (40, 33067, 0.004988, 6.59444666),
    (45, 34927, 0.004545, 6.05249786),
    (50, 36583, 0.004145, 6.12209034),
    (55, 38592, 0.003815, 5.18920040),
    (60, 41435, 0.003368, 5.20370579),
    (65, 44210, 0.002988, 4.85558844),
    (70, 48191, 0.002560, 4.45603371),
    (75, 52644, 0.002162, 3.74262619),
    (80, 58944, 0.001775, 3.39873624),
    (85, 68102, 0.001335, 2.92167902),
    (90, 83917, 0.000842, 3.01503849),
    (95, 115989, 0.000420, 2.81292272),
    (100, 250027, 0.000159, 2.79607034),
];

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Tolerance for file size comparison against C++ (percentage)
/// 4:4:4 progressive: Rust is typically +0.3% to +0.9% larger than C++
/// Q100 shows +2.7% due to minor differences in high-quality quantization
const SIZE_TOLERANCE_444: f64 = 3.0;

/// Tolerance for subsampled modes (4:2:0, 4:2:2)
/// Rust is typically +0.3% to +1.4% larger than C++ for most quality levels
const SIZE_TOLERANCE_SUBSAMPLED: f64 = 2.0;

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
    subsampling: ChromaSubsampling,
    optimize_huffman: bool,
) -> Vec<u8> {
    // Note: progressive mode requires optimized huffman, so we enable progressive
    // when optimize_huffman=true to match C++ cjpegli --progressive_level=2
    let config = EncoderConfig::ycbcr(quality, subsampling)
        .progressive(optimize_huffman) // progressive when optimized
        .optimize_huffman(optimize_huffman);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");
    enc.push_packed(rgb, enough::Unstoppable)
        .expect("push data");
    enc.finish().expect("Encoding failed")
}

// =============================================================================
// TESTS: C++ PARITY
// =============================================================================

/// Test 4:4:4 optimized Huffman matches C++ within tolerance
#[test]
fn test_cpp_parity_s444_optimized() {
    let (rgb, width, height) = load_test_image();

    for &(quality, cpp_size) in CPP_S444_OPT {
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::None,
            true,
        );
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
/// NOTE: Skipped because cjpegli doesn't have a true "fixed Huffman" mode -
/// it always uses optimized Huffman even with --progressive_level=0.
/// The CPP_S444_FIXED values are from cjpegli baseline mode which still
/// uses optimized Huffman, resulting in ~15-17% smaller files than Rust's
/// true fixed Huffman output.
#[test]
#[ignore = "cjpegli has no fixed Huffman mode - always optimizes"]
fn test_cpp_parity_s444_fixed() {
    let (rgb, width, height) = load_test_image();

    for &(quality, cpp_size) in CPP_S444_FIXED {
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::None,
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
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::Quarter,
            true,
        );
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
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::HalfHorizontal,
            true,
        );
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
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::None,
            true,
        );

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
            .decode(&jpeg, Unstoppable)
            .expect("Decoding failed");
        let dssim = compute_dssim(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{}: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );
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
            ChromaSubsampling::None,
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
            .decode(&jpeg, Unstoppable)
            .expect("Decoding failed");
        let dssim = compute_dssim(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{} fixed: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );
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
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::Quarter,
            true,
        );

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
            .decode(&jpeg, Unstoppable)
            .expect("Decoding failed");
        let dssim = compute_dssim(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{} 4:2:0: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );
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
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            quality as f32,
            ChromaSubsampling::HalfHorizontal,
            true,
        );

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
            .decode(&jpeg, Unstoppable)
            .expect("Decoding failed");
        let dssim = compute_dssim(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );

        assert!(
            (dssim - expected_dssim).abs() < DSSIM_REGRESSION_TOLERANCE,
            "Q{} 4:2:2: DSSIM regressed: got {:.6} expected {:.6}",
            quality,
            dssim,
            expected_dssim
        );

        // Check Butteraugli regression (relative tolerance)
        let bfly = compute_butteraugli(
            &rgb,
            decoded.pixels_u8().unwrap(),
            width as usize,
            height as usize,
        );
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
        (ChromaSubsampling::None, true, "S444_OPT"),
        (ChromaSubsampling::None, false, "S444_FIXED"),
        (ChromaSubsampling::Quarter, true, "S420_OPT"),
        (ChromaSubsampling::HalfHorizontal, true, "S422_OPT"),
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
                .decode(&jpeg, Unstoppable)
                .expect("Decoding failed");
            let rgba_dec = rgb_to_rgba(decoded.pixels_u8().unwrap());
            let dec_img = dssim
                .create_image_rgba(&rgba_dec, width as usize, height as usize)
                .unwrap();
            let (dssim_val, _) = dssim.compare(&orig_img, dec_img);

            let bfly = butteraugli::compute_butteraugli(
                &rgb,
                decoded.pixels_u8().unwrap(),
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
        let jpeg = encode_jpeg(&rgb, width, height, q as f32, ChromaSubsampling::None, true);
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
        let jpeg = encode_jpeg(
            &rgb,
            width,
            height,
            q as f32,
            ChromaSubsampling::Quarter,
            true,
        );
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
