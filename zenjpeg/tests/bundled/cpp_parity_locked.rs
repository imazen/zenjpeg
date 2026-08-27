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
/// Updated 2026-03-31: regenerated after rgb_to_ycbcr OOB fix and butteraugli 0.9 bump
const RUST_S444_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 16100, 0.016205, 9.28292274),
    (10, 19273, 0.011946, 7.49988699),
    (15, 23208, 0.008885, 6.83645344),
    (20, 27233, 0.006707, 5.60355902),
    (25, 31127, 0.005183, 5.49145365),
    (30, 33914, 0.004429, 5.63252544),
    (35, 35337, 0.004111, 5.11782455),
    (40, 36777, 0.003838, 4.89694834),
    (45, 38839, 0.003443, 4.52144432),
    (50, 40666, 0.003138, 4.10139751),
    (55, 42929, 0.002834, 3.86261797),
    (60, 45990, 0.002501, 3.74165821),
    (65, 49162, 0.002185, 3.67892170),
    (70, 53450, 0.001849, 3.40255904),
    (75, 58424, 0.001551, 2.89049029),
    (80, 65360, 0.001229, 2.57409930),
    (85, 75575, 0.000878, 2.20107245),
    (90, 93682, 0.000515, 1.64325857),
    (95, 131262, 0.000220, 1.06868446),
    (100, 319386, 0.000033, 0.45505026),
];

/// Rust expected: 4:4:4, baseline, corpus-trained Huffman tables
/// Updated 2026-03-31: regenerated after rgb_to_ycbcr OOB fix and butteraugli 0.9 bump
// Sizes relocked 2026-08-26 (73c84c50): fixed-table completion gives every
// legal symbol a code (the old tables silently dropped uncovered symbols as
// ZERO bits), growing the DHT markers by ~240 bytes. Same coefficients —
// dssim/butteraugli unchanged. Measured on tests/images/1.png; byte-identical
// across x86_64 (CI) and aarch64 at Q5.
const RUST_S444_FIXED: &[(u8, usize, f64, f64)] = &[
    (5, 16926, 0.016205, 9.28292274),
    (10, 19632, 0.011946, 7.49988699),
    (15, 23458, 0.008885, 6.83645344),
    (20, 27571, 0.006707, 5.60355902),
    (25, 31690, 0.005183, 5.49145365),
    (30, 34657, 0.004429, 5.63252544),
    (35, 36207, 0.004111, 5.11782455),
    (40, 37730, 0.003838, 4.89694834),
    (45, 39943, 0.003443, 4.52144432),
    (50, 41910, 0.003138, 4.10139751),
    (55, 43923, 0.002834, 3.86261797),
    (60, 47026, 0.002501, 3.74165821),
    (65, 50313, 0.002185, 3.67892170),
    (70, 55018, 0.001849, 3.40255904),
    (75, 60243, 0.001551, 2.89049029),
    (80, 67582, 0.001229, 2.57409930),
    (85, 78601, 0.000878, 2.20107245),
    (90, 97492, 0.000515, 1.64325857),
    (95, 138330, 0.000220, 1.06868446),
    (100, 336091, 0.000033, 0.45505026),
];

/// Rust expected: 4:2:0, progressive, optimized Huffman
/// Updated 2026-03-31: regenerated after rgb_to_ycbcr OOB fix and butteraugli 0.9 bump
const RUST_S420_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 13728, 0.017472, 9.68782043),
    (10, 16519, 0.013114, 9.59676552),
    (15, 19709, 0.009868, 7.39160109),
    (20, 22978, 0.007793, 7.36341524),
    (25, 26129, 0.006347, 5.99795628),
    (30, 28100, 0.005612, 6.37178659),
    (35, 29429, 0.005224, 6.30691242),
    (40, 30664, 0.004857, 5.71547699),
    (45, 32174, 0.004485, 5.57346106),
    (50, 33798, 0.004156, 5.55234098),
    (55, 36031, 0.003707, 5.34349298),
    (60, 38271, 0.003328, 5.35654259),
    (65, 41291, 0.002887, 5.11353445),
    (70, 45000, 0.002483, 5.28472424),
    (75, 49593, 0.002101, 4.86233664),
    (80, 55314, 0.001726, 5.24773169),
    (85, 63993, 0.001314, 5.06290722),
    (90, 79822, 0.000870, 4.87044668),
    (95, 111569, 0.000513, 4.95730686),
    (100, 210910, 0.000350, 4.85999775),
];

/// Rust expected: 4:2:2, progressive, optimized Huffman
/// Updated 2026-03-31: regenerated after rgb_to_ycbcr OOB fix and butteraugli 0.9 bump
const RUST_S422_OPT: &[(u8, usize, f64, f64)] = &[
    (5, 14798, 0.019043, 11.63883114),
    (10, 17644, 0.014269, 9.58772182),
    (15, 21118, 0.010925, 7.98215532),
    (20, 24674, 0.008396, 7.42982292),
    (25, 28069, 0.006695, 6.85173988),
    (30, 30495, 0.005748, 6.40909243),
    (35, 31743, 0.005372, 6.07368898),
    (40, 33043, 0.005023, 6.80269670),
    (45, 34896, 0.004589, 6.28469753),
    (50, 36552, 0.004167, 6.09321260),
    (55, 38552, 0.003839, 5.42336845),
    (60, 41394, 0.003399, 5.29362106),
    (65, 44175, 0.003006, 4.91376257),
    (70, 48140, 0.002575, 4.51380873),
    (75, 52568, 0.002181, 3.87944365),
    (80, 58839, 0.001786, 3.47810030),
    (85, 68008, 0.001347, 3.07347775),
    (90, 83843, 0.000846, 3.17047143),
    (95, 115889, 0.000419, 2.97829986),
    (100, 249916, 0.000158, 2.96560025),
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
    let img = zenjpeg_bench_utils::load_png(&png_path).expect("Failed to load test image");
    let width = img.width() as u32;
    let height = img.height() as u32;
    let rgb: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    (rgb, width, height)
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
    let to_img = |data: &[u8]| -> imgref::ImgVec<rgb::RGB8> {
        let pixels: Vec<rgb::RGB8> = data
            .chunks_exact(3)
            .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
            .collect();
        imgref::ImgVec::new(pixels, width, height)
    };
    let orig_img = to_img(original);
    let dec_img = to_img(decoded);
    let params = ButteraugliParams::default();
    butteraugli::butteraugli(orig_img.as_ref(), dec_img.as_ref(), &params)
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
        .optimize_huffman(optimize_huffman)
        .restart_mcu_rows(0); // Disable restart markers for parity comparison with C++
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
/// Only runs with the `yuv` feature (default); no-yuv path produces different output.
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
/// Only runs with the `yuv` feature (default); no-yuv path produces different output.
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
/// Only runs with the `yuv` feature (default); no-yuv path produces different output.
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
/// Only runs with the `yuv` feature (default); no-yuv path produces different output.
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

            let orig_img = {
                let pixels: Vec<rgb::RGB8> = rgb
                    .chunks_exact(3)
                    .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
                    .collect();
                imgref::ImgVec::new(pixels, width as usize, height as usize)
            };
            let dec_img = {
                let pixels: Vec<rgb::RGB8> = decoded
                    .pixels_u8()
                    .unwrap()
                    .chunks_exact(3)
                    .map(|c| rgb::RGB8::new(c[0], c[1], c[2]))
                    .collect();
                imgref::ImgVec::new(pixels, width as usize, height as usize)
            };
            let bfly = butteraugli::butteraugli(orig_img.as_ref(), dec_img.as_ref(), &bfly_params)
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
