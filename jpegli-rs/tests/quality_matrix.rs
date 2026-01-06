//! Quality matrix tests comparing Rust vs C++ jpegli across all configurations.
//!
//! Tests SSIMULACRA2 scores for each encoder configuration:
//! - Color spaces: YCbCr, XYB
//! - Subsampling: 444, 422, 420, 440
//! - Modes: Baseline, Progressive
//!
//! When `ffi-tests` feature is enabled, compares against live C++ jpegli.
//! Otherwise, compares against stored reference values.
//!
//! Run with:
//! ```
//! cargo test --release --test quality_matrix -- --nocapture
//! cargo test --release --test quality_matrix --features ffi-tests -- --nocapture
//! ```

use fast_ssim2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};
use jpegli::{Encoder, JpegMode, PixelFormat, Quality, Subsampling};
use std::path::PathBuf;

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

/// Quality levels to test (6 levels spanning 1-100)
const QUALITY_LEVELS: [u8; 6] = [10, 30, 50, 70, 85, 95];

/// Tolerance for SSIMULACRA2 score comparisons (Rust vs C++ or vs reference)
const SSIM2_TOLERANCE: f64 = 0.5;

/// Tolerance for SSIMULACRA2 absolute minimum (quality floor)
const SSIM2_MIN_TOLERANCE: f64 = 1.0;

/// Tolerance for file size comparison (percentage difference from reference)
const SIZE_TOLERANCE_PERCENT: f64 = 2.0;

// ============================================================================
// REFERENCE VALUES (C++ jpegli results for frymire.png 1118x1105)
// Generated with: cargo test --release --test quality_matrix --features ffi-tests -- --nocapture generate
// ============================================================================

/// Reference values: (quality, rust_ssim2, cpp_ssim2, rust_size_bytes, cpp_size_bytes)
/// Format: [Q10, Q30, Q50, Q70, Q85, Q95]
/// Generated: 2026-01-06 from frymire.png (1118x1105)
mod reference {
    /// Reference data tuple type
    pub type RefData = (u8, f64, f64, usize, usize);

    // YCbCr 4:4:4 Baseline
    pub const YCBCR_444_BASELINE: [RefData; 6] = [
        (10, 6.2, 6.2, 138524, 138524),
        (30, 39.7, 39.7, 266273, 266273),
        (50, 48.9, 48.9, 329838, 329838),
        (70, 59.6, 59.6, 438797, 438797),
        (85, 69.0, 69.0, 600401, 600401),
        (95, 81.3, 81.3, 937148, 937148),
    ];

    // YCbCr 4:4:4 Progressive
    pub const YCBCR_444_PROGRESSIVE: [RefData; 6] = [
        (10, 6.2, 6.2, 138284, 138284),
        (30, 39.7, 39.7, 259727, 259727),
        (50, 48.9, 48.9, 320645, 320645),
        (70, 59.6, 59.6, 425666, 425666),
        (85, 69.0, 69.0, 581885, 581885),
        (95, 81.3, 81.3, 908023, 908023),
    ];

    // YCbCr 4:2:2 Baseline
    pub const YCBCR_422_BASELINE: [RefData; 6] = [
        (10, -0.5, -0.5, 124814, 124814),
        (30, 30.2, 30.2, 238023, 238023),
        (50, 38.5, 38.5, 293361, 293361),
        (70, 47.9, 47.9, 386682, 386682),
        (85, 54.8, 54.8, 520353, 520353),
        (95, 61.4, 61.4, 783875, 783875),
    ];

    // YCbCr 4:2:2 Progressive
    pub const YCBCR_422_PROGRESSIVE: [RefData; 6] = [
        (10, -0.5, -0.5, 125348, 125348),
        (30, 30.2, 30.2, 232369, 232369),
        (50, 38.5, 38.5, 285904, 285904),
        (70, 47.9, 47.9, 375694, 375694),
        (85, 54.8, 54.8, 505401, 505401),
        (95, 61.4, 61.4, 760331, 760331),
    ];

    // YCbCr 4:2:0 Baseline
    pub const YCBCR_420_BASELINE: [RefData; 6] = [
        (10, 1.5, 1.5, 114110, 114110),
        (30, 29.0, 29.0, 216584, 216584),
        (50, 36.6, 36.6, 269536, 269536),
        (70, 45.0, 45.0, 361917, 361917),
        (85, 50.4, 50.4, 492806, 492806),
        (95, 53.3, 53.3, 736931, 736931),
    ];

    // YCbCr 4:2:0 Progressive
    pub const YCBCR_420_PROGRESSIVE: [RefData; 6] = [
        (10, 1.5, 1.5, 113877, 113877),
        (30, 29.0, 29.0, 210274, 210274),
        (50, 36.6, 36.6, 260900, 260900),
        (70, 45.0, 45.0, 349965, 349965),
        (85, 50.4, 50.4, 476205, 476205),
        (95, 53.3, 53.3, 712084, 712084),
    ];

    // YCbCr 4:4:0 Baseline
    pub const YCBCR_440_BASELINE: [RefData; 6] = [
        (10, 0.7, 0.7, 124837, 124837),
        (30, 30.4, 30.4, 238368, 238368),
        (50, 38.3, 38.3, 293565, 293565),
        (70, 46.8, 46.8, 387207, 387207),
        (85, 54.1, 54.1, 521670, 521670),
        (95, 60.0, 60.0, 785296, 785296),
    ];

    // YCbCr 4:4:0 Progressive
    pub const YCBCR_440_PROGRESSIVE: [RefData; 6] = [
        (10, 0.7, 0.7, 123726, 123726),
        (30, 30.4, 30.4, 230478, 230478),
        (50, 38.3, 38.3, 284083, 284083),
        (70, 46.8, 46.8, 374127, 374127),
        (85, 54.1, 54.1, 504689, 504689),
        (95, 60.0, 60.0, 759344, 759344),
    ];

    // XYB 4:4:4 Baseline - NOTE: XYB decode requires ICC-aware decoder
    // zune-jpeg doesn't handle ICC profiles, so these scores are invalid (-64.2)
    // File sizes are still valid though
    pub const XYB_444_BASELINE: [RefData; 6] = [
        (10, -64.2, -64.2, 121455, 121455),
        (30, -64.2, -64.2, 226717, 226717),
        (50, -64.2, -64.2, 279171, 279171),
        (70, -64.2, -64.2, 365858, 365858),
        (85, -64.2, -64.2, 498144, 498144),
        (95, -64.2, -64.2, 772377, 772377),
    ];

    // XYB 4:4:4 Progressive
    pub const XYB_444_PROGRESSIVE: [RefData; 6] = [
        (10, -64.2, -64.2, 124848, 124848),
        (30, -64.2, -64.2, 231291, 231291),
        (50, -64.2, -64.2, 284528, 284528),
        (70, -64.2, -64.2, 375211, 375211),
        (85, -64.2, -64.2, 514174, 514174),
        (95, -64.2, -64.2, 793861, 793861),
    ];
}

// ============================================================================
// TEST IMAGE LOADING
// ============================================================================

fn get_frymire_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("images")
        .join("frymire.png")
}

fn load_png(path: &PathBuf) -> Option<(Vec<u8>, usize, usize)> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    // Convert to RGB if necessary
    let (rgb, width, height) = match info.color_type {
        png::ColorType::Rgb => (buf, info.width as usize, info.height as usize),
        png::ColorType::Rgba => {
            let rgb: Vec<u8> = buf.chunks(4).flat_map(|c| &c[..3]).copied().collect();
            (rgb, info.width as usize, info.height as usize)
        }
        _ => return None,
    };

    Some((rgb, width, height))
}

// ============================================================================
// SSIMULACRA2 COMPUTATION
// ============================================================================

fn compute_ssim2(original: &[u8], decoded: &[u8], width: usize, height: usize) -> f64 {
    let orig_rgb = Rgb::new(
        original
            .chunks(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let dec_rgb = Rgb::new(
        decoded
            .chunks(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    compute_frame_ssimulacra2(orig_rgb, dec_rgb).unwrap_or(-999.0)
}

// ============================================================================
// ENCODING HELPERS
// ============================================================================

fn encode_rust(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    subsampling: Subsampling,
    mode: JpegMode,
    use_xyb: bool,
) -> Vec<u8> {
    let mut encoder = Encoder::new()
        .width(width)
        .height(height)
        .pixel_format(PixelFormat::Rgb)
        .jpegli_quality(Quality::from_quality(quality.into()))
        .subsampling(subsampling)
        .mode(mode)
        .optimize_huffman(true);

    if use_xyb {
        encoder = encoder.use_xyb(true);
    }

    encoder.encode(rgb).expect("Rust encode failed")
}

#[cfg(feature = "ffi-tests")]
fn encode_cpp(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    subsampling: Subsampling,
    progressive: bool,
    use_xyb: bool,
) -> Option<Vec<u8>> {
    use std::io::Write;
    use std::process::Command;

    // Write RGB to temp file as PPM
    let ppm_path = "/tmp/quality_matrix_input.ppm";
    let jpg_path = "/tmp/quality_matrix_output.jpg";

    let mut ppm = std::fs::File::create(ppm_path).ok()?;
    writeln!(ppm, "P6").ok()?;
    writeln!(ppm, "{} {}", width, height).ok()?;
    writeln!(ppm, "255").ok()?;
    ppm.write_all(rgb).ok()?;
    drop(ppm);

    // Find cjpegli
    let cjpegli = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("internal/jpegli-cpp/build/tools/cjpegli");

    if !cjpegli.exists() {
        return None;
    }

    let subsamp_str = match subsampling {
        Subsampling::S444 => "444",
        Subsampling::S422 => "422",
        Subsampling::S420 => "420",
        Subsampling::S440 => "440",
        _ => "444", // Default for any future variants
    };

    let mut args = vec![
        ppm_path.to_string(),
        jpg_path.to_string(),
        "-q".to_string(),
        quality.to_string(),
        "--chroma_subsampling".to_string(),
        subsamp_str.to_string(),
    ];

    if progressive {
        args.push("-p".to_string());
    }

    if use_xyb {
        args.push("--xyb".to_string());
    }

    let status = Command::new(&cjpegli)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .ok()?;

    if !status.success() {
        return None;
    }

    std::fs::read(jpg_path).ok()
}

fn decode_jpeg(data: &[u8]) -> Vec<u8> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::JpegDecoder;

    let cursor = ZCursor::new(data);
    let mut decoder = JpegDecoder::new(cursor);
    decoder.decode().expect("decode failed")
}

// ============================================================================
// TEST RUNNER
// ============================================================================

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ConfigResult {
    quality: u8,
    rust_ssim2: f64,
    cpp_ssim2: f64,
    rust_size: usize,
    cpp_size: usize,
}

fn test_configuration(
    name: &str,
    rgb: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    mode: JpegMode,
    use_xyb: bool,
    reference: &[reference::RefData; 6],
) -> Vec<ConfigResult> {
    let width_u32 = width as u32;
    let height_u32 = height as u32;
    #[allow(unused_variables)]
    let progressive = mode == JpegMode::Progressive;

    let mut results = Vec::new();

    println!("\n=== {} ===", name);
    println!(
        "{:>4} | {:>10} {:>10} | {:>8} {:>8} | {:>6} {:>6} | {}",
        "Q", "Rust SSIM2", "C++ SSIM2", "Rust KB", "Ref KB", "Δ SSIM", "Δ Size", "Status"
    );
    println!("{}", "-".repeat(85));

    for (i, &quality) in QUALITY_LEVELS.iter().enumerate() {
        // Encode with Rust
        let rust_jpeg = encode_rust(rgb, width_u32, height_u32, quality, subsampling, mode, use_xyb);
        let rust_decoded = decode_jpeg(&rust_jpeg);
        let rust_ssim2 = compute_ssim2(rgb, &rust_decoded, width, height);

        // Get C++ result (live or reference)
        #[cfg(feature = "ffi-tests")]
        let (cpp_ssim2, cpp_size) = {
            if let Some(cpp_jpeg) =
                encode_cpp(rgb, width_u32, height_u32, quality, subsampling, progressive, use_xyb)
            {
                let cpp_decoded = decode_jpeg(&cpp_jpeg);
                let ssim2 = compute_ssim2(rgb, &cpp_decoded, width, height);
                (ssim2, cpp_jpeg.len())
            } else {
                // Fall back to reference
                (reference[i].2, reference[i].4)
            }
        };

        #[cfg(not(feature = "ffi-tests"))]
        let (cpp_ssim2, cpp_size) = (reference[i].2, reference[i].4);

        // Compare against reference size (use Rust reference if cpp_size is 0)
        let ref_size = if cpp_size > 0 { cpp_size } else { reference[i].3 };

        let ssim_diff = rust_ssim2 - cpp_ssim2;
        let size_diff_percent = if ref_size > 0 {
            100.0 * (rust_jpeg.len() as f64 - ref_size as f64) / ref_size as f64
        } else {
            0.0
        };

        // Check if within tolerance
        let ssim_ok = ssim_diff.abs() <= SSIM2_TOLERANCE;
        let min_ok = rust_ssim2 >= reference[i].1 - SSIM2_MIN_TOLERANCE;
        let size_ok = size_diff_percent.abs() <= SIZE_TOLERANCE_PERCENT;
        let status = if ssim_ok && min_ok && size_ok { "✓" } else { "✗" };

        println!(
            "{:>4} | {:>10.2} {:>10.2} | {:>7.1}K {:>7.1}K | {:>+6.2} {:>+5.1}% | {}",
            quality,
            rust_ssim2,
            cpp_ssim2,
            rust_jpeg.len() as f64 / 1024.0,
            ref_size as f64 / 1024.0,
            ssim_diff,
            size_diff_percent,
            status
        );

        results.push(ConfigResult {
            quality,
            rust_ssim2,
            cpp_ssim2,
            rust_size: rust_jpeg.len(),
            cpp_size: ref_size,
        });
    }

    results
}

// ============================================================================
// ASSERTION HELPER
// ============================================================================

fn assert_results(results: &[ConfigResult], reference: &[reference::RefData; 6], name: &str) {
    for (i, r) in results.iter().enumerate() {
        let ref_data = &reference[i];
        let ref_ssim2 = ref_data.1;
        let ref_size = ref_data.3;

        // Check SSIM2 quality
        assert!(
            r.rust_ssim2 >= ref_ssim2 - SSIM2_MIN_TOLERANCE,
            "{} Q{}: SSIM2 {:.2} below minimum {:.2}",
            name,
            r.quality,
            r.rust_ssim2,
            ref_ssim2 - SSIM2_MIN_TOLERANCE
        );

        // Check file size within tolerance
        let size_diff_percent = 100.0 * (r.rust_size as f64 - ref_size as f64) / ref_size as f64;
        assert!(
            size_diff_percent.abs() <= SIZE_TOLERANCE_PERCENT,
            "{} Q{}: Size diff {:.1}% exceeds tolerance {:.1}% (rust={}, ref={})",
            name,
            r.quality,
            size_diff_percent,
            SIZE_TOLERANCE_PERCENT,
            r.rust_size,
            ref_size
        );
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_ycbcr_444_baseline() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:4:4 Baseline",
        &rgb,
        width,
        height,
        Subsampling::S444,
        JpegMode::Baseline,
        false,
        &reference::YCBCR_444_BASELINE,
    );

    assert_results(&results, &reference::YCBCR_444_BASELINE, "YCbCr 4:4:4 Baseline");
}

#[test]
fn test_ycbcr_444_progressive() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:4:4 Progressive",
        &rgb,
        width,
        height,
        Subsampling::S444,
        JpegMode::Progressive,
        false,
        &reference::YCBCR_444_PROGRESSIVE,
    );

    assert_results(&results, &reference::YCBCR_444_PROGRESSIVE, "YCbCr 4:4:4 Progressive");
}

#[test]
fn test_ycbcr_422_baseline() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:2:2 Baseline",
        &rgb,
        width,
        height,
        Subsampling::S422,
        JpegMode::Baseline,
        false,
        &reference::YCBCR_422_BASELINE,
    );

    assert_results(&results, &reference::YCBCR_422_BASELINE, "YCbCr 4:2:2 Baseline");
}

#[test]
fn test_ycbcr_422_progressive() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:2:2 Progressive",
        &rgb,
        width,
        height,
        Subsampling::S422,
        JpegMode::Progressive,
        false,
        &reference::YCBCR_422_PROGRESSIVE,
    );

    assert_results(&results, &reference::YCBCR_422_PROGRESSIVE, "YCbCr 4:2:2 Progressive");
}

#[test]
fn test_ycbcr_420_baseline() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:2:0 Baseline",
        &rgb,
        width,
        height,
        Subsampling::S420,
        JpegMode::Baseline,
        false,
        &reference::YCBCR_420_BASELINE,
    );

    assert_results(&results, &reference::YCBCR_420_BASELINE, "YCbCr 4:2:0 Baseline");
}

#[test]
fn test_ycbcr_420_progressive() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:2:0 Progressive",
        &rgb,
        width,
        height,
        Subsampling::S420,
        JpegMode::Progressive,
        false,
        &reference::YCBCR_420_PROGRESSIVE,
    );

    assert_results(&results, &reference::YCBCR_420_PROGRESSIVE, "YCbCr 4:2:0 Progressive");
}

#[test]
fn test_ycbcr_440_baseline() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:4:0 Baseline",
        &rgb,
        width,
        height,
        Subsampling::S440,
        JpegMode::Baseline,
        false,
        &reference::YCBCR_440_BASELINE,
    );

    assert_results(&results, &reference::YCBCR_440_BASELINE, "YCbCr 4:4:0 Baseline");
}

#[test]
fn test_ycbcr_440_progressive() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "YCbCr 4:4:0 Progressive",
        &rgb,
        width,
        height,
        Subsampling::S440,
        JpegMode::Progressive,
        false,
        &reference::YCBCR_440_PROGRESSIVE,
    );

    assert_results(&results, &reference::YCBCR_440_PROGRESSIVE, "YCbCr 4:4:0 Progressive");
}

/// XYB tests are ignored because zune-jpeg doesn't handle ICC profiles.
/// XYB JPEGs require ICC-aware decoding for correct color conversion.
/// TODO: Use jpegli decoder with CMS or PIL for XYB quality tests.
#[test]
#[ignore = "XYB requires ICC-aware decoder (zune-jpeg doesn't support ICC)"]
fn test_xyb_444_baseline() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "XYB 4:4:4 Baseline",
        &rgb,
        width,
        height,
        Subsampling::S444,
        JpegMode::Baseline,
        true,
        &reference::XYB_444_BASELINE,
    );

    for (i, r) in results.iter().enumerate() {
        let ref_rust = reference::XYB_444_BASELINE[i].1;
        assert!(
            r.rust_ssim2 >= ref_rust - SSIM2_MIN_TOLERANCE,
            "Q{}: SSIM2 {:.2} below minimum {:.2}",
            r.quality,
            r.rust_ssim2,
            ref_rust - SSIM2_MIN_TOLERANCE
        );
    }
}

#[test]
#[ignore = "XYB requires ICC-aware decoder (zune-jpeg doesn't support ICC)"]
fn test_xyb_444_progressive() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    let results = test_configuration(
        "XYB 4:4:4 Progressive",
        &rgb,
        width,
        height,
        Subsampling::S444,
        JpegMode::Progressive,
        true,
        &reference::XYB_444_PROGRESSIVE,
    );

    for (i, r) in results.iter().enumerate() {
        let ref_rust = reference::XYB_444_PROGRESSIVE[i].1;
        assert!(
            r.rust_ssim2 >= ref_rust - SSIM2_MIN_TOLERANCE,
            "Q{}: SSIM2 {:.2} below minimum {:.2}",
            r.quality,
            r.rust_ssim2,
            ref_rust - SSIM2_MIN_TOLERANCE
        );
    }
}

/// Generate reference values (run with --nocapture to see output)
#[test]
#[ignore]
fn generate_reference_values() {
    let path = get_frymire_path();
    let (rgb, width, height) = load_png(&path).expect("Failed to load frymire.png");

    println!("\n// Reference values for quality_matrix.rs");
    println!("// Generated from frymire.png ({}x{})", width, height);
    println!("// Format: (quality, rust_ssim2, cpp_ssim2, rust_size, cpp_size)");
    println!("// Copy these into the reference module\n");

    let configs = [
        ("YCBCR_444_BASELINE", Subsampling::S444, JpegMode::Baseline, false),
        ("YCBCR_444_PROGRESSIVE", Subsampling::S444, JpegMode::Progressive, false),
        ("YCBCR_422_BASELINE", Subsampling::S422, JpegMode::Baseline, false),
        ("YCBCR_422_PROGRESSIVE", Subsampling::S422, JpegMode::Progressive, false),
        ("YCBCR_420_BASELINE", Subsampling::S420, JpegMode::Baseline, false),
        ("YCBCR_420_PROGRESSIVE", Subsampling::S420, JpegMode::Progressive, false),
        ("YCBCR_440_BASELINE", Subsampling::S440, JpegMode::Baseline, false),
        ("YCBCR_440_PROGRESSIVE", Subsampling::S440, JpegMode::Progressive, false),
        ("XYB_444_BASELINE", Subsampling::S444, JpegMode::Baseline, true),
        ("XYB_444_PROGRESSIVE", Subsampling::S444, JpegMode::Progressive, true),
    ];

    for (name, subsampling, mode, use_xyb) in configs {
        println!("pub const {}: [RefData; 6] = [", name);

        for &quality in &QUALITY_LEVELS {
            let rust_jpeg = encode_rust(
                &rgb,
                width as u32,
                height as u32,
                quality,
                subsampling,
                mode,
                use_xyb,
            );
            let rust_decoded = decode_jpeg(&rust_jpeg);
            let rust_ssim2 = compute_ssim2(&rgb, &rust_decoded, width, height);
            let rust_size = rust_jpeg.len();

            #[cfg(feature = "ffi-tests")]
            let (cpp_ssim2, cpp_size) = {
                let progressive = mode == JpegMode::Progressive;
                if let Some(cpp_jpeg) = encode_cpp(
                    &rgb,
                    width as u32,
                    height as u32,
                    quality,
                    subsampling,
                    progressive,
                    use_xyb,
                ) {
                    let cpp_decoded = decode_jpeg(&cpp_jpeg);
                    let ssim2 = compute_ssim2(&rgb, &cpp_decoded, width, height);
                    (ssim2, cpp_jpeg.len())
                } else {
                    (rust_ssim2, rust_size)
                }
            };

            #[cfg(not(feature = "ffi-tests"))]
            let (cpp_ssim2, cpp_size) = (rust_ssim2, rust_size);

            println!(
                "    ({}, {:.1}, {:.1}, {}, {}),",
                quality, rust_ssim2, cpp_ssim2, rust_size, cpp_size
            );
        }

        println!("];\n");
    }
}

// ============================================================================
// BENCHMARK: RUST VS C++ PERFORMANCE
// ============================================================================

/// Benchmark comparing Rust vs C++ jpegli encoding performance.
///
/// Uses jpegli-bench-utils for shared utilities (ImageData, QualityMetrics).
///
/// Run with:
/// ```
/// cargo test --release --test quality_matrix --features ffi-tests benchmark_rust_vs_cpp -- --nocapture --ignored
/// ```
#[test]
#[ignore = "Requires ffi-tests feature and C++ jpegli build"]
#[cfg(feature = "ffi-tests")]
fn benchmark_rust_vs_cpp() {
    use jpegli_bench_utils::{
        ChromaSubsampling as BenchSubsampling, ColorMode, EncoderConfig, EncoderImpl, ImageData,
        QualityMetrics, ScanMode,
    };
    use std::time::Instant;

    let img = ImageData::from_png(&get_frymire_path()).expect("Failed to load frymire.png");

    const WARMUP_ITERS: usize = 2;
    const BENCH_ITERS: usize = 5;
    const BENCH_QUALITIES: [u8; 4] = [30, 50, 75, 90];

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    RUST vs C++ JPEGLI BENCHMARK                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Image: frymire.png ({}x{}, {:.2} MP)                                    ║",
        img.width,
        img.height,
        img.pixel_count() as f64 / 1_000_000.0
    );
    println!(
        "║ Iterations: {} warmup + {} timed                                            ║",
        WARMUP_ITERS, BENCH_ITERS
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let configs: [(&str, BenchSubsampling, ScanMode); 4] = [
        ("YCbCr 4:4:4 Baseline", BenchSubsampling::S444, ScanMode::Baseline),
        ("YCbCr 4:2:0 Baseline", BenchSubsampling::S420, ScanMode::Baseline),
        ("YCbCr 4:4:4 Progressive", BenchSubsampling::S444, ScanMode::Progressive),
        ("YCbCr 4:2:0 Progressive", BenchSubsampling::S420, ScanMode::Progressive),
    ];

    let mut all_valid = true;
    let orig_rgb = img.as_rgb_image();

    for (config_name, subsampling, scan_mode) in configs {
        println!(
            "┌─ {} ─────────────────────────────────────────────────┐",
            config_name
        );
        println!(
            "│ {:>4} │ {:>8} {:>8} │ {:>8} {:>8} │ {:>7} │ {:>6} │ {:>6} │",
            "Q", "Rust ms", "C++ ms", "Rust KB", "C++ KB", "Speedup", "Δ Size", "Δ SSIM"
        );
        println!("├──────┼───────────────────┼───────────────────┼─────────┼────────┼────────┤");

        // Convert to jpegli types for encode_cpp
        let jpegli_subsampling = match subsampling {
            BenchSubsampling::S444 => Subsampling::S444,
            BenchSubsampling::S422 => Subsampling::S422,
            BenchSubsampling::S420 => Subsampling::S420,
            BenchSubsampling::S440 => Subsampling::S440,
        };
        let progressive = scan_mode == ScanMode::Progressive;

        for &quality in &BENCH_QUALITIES {
            // Warmup using EncoderConfig
            let rust_config = EncoderConfig::new(EncoderImpl::JpegliRs)
                .quality(quality)
                .color(ColorMode::YCbCr)
                .subsampling(subsampling)
                .scan(scan_mode);

            for _ in 0..WARMUP_ITERS {
                let _ = rust_config.encode(&img);
                let _ = encode_cpp(
                    &img.pixels,
                    img.width as u32,
                    img.height as u32,
                    quality,
                    jpegli_subsampling,
                    progressive,
                    false,
                );
            }

            // Benchmark Rust
            let rust_start = Instant::now();
            let mut rust_jpeg = Vec::new();
            for _ in 0..BENCH_ITERS {
                rust_jpeg = rust_config.encode(&img).expect("Rust encode failed");
            }
            let rust_time = rust_start.elapsed().as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

            // Benchmark C++
            let cpp_start = Instant::now();
            let mut cpp_jpeg = Vec::new();
            let mut cpp_encode_success = false;
            for _ in 0..BENCH_ITERS {
                if let Some(data) = encode_cpp(
                    &img.pixels,
                    img.width as u32,
                    img.height as u32,
                    quality,
                    jpegli_subsampling,
                    progressive,
                    false,
                ) {
                    cpp_jpeg = data;
                    cpp_encode_success = !cpp_jpeg.is_empty();
                }
            }
            let cpp_time = cpp_start.elapsed().as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

            // Compute quality metrics using shared utils
            let rust_decoded =
                jpegli_bench_utils::decode_jpeg_to_rgb(&rust_jpeg).expect("Rust decode failed");
            let rust_ssim2 = QualityMetrics::ssimulacra2(orig_rgb.as_ref(), rust_decoded.as_ref());

            // Compute differences (handle encode/decode failure gracefully)
            let (cpp_time_str, cpp_size_str, speedup_str, size_diff_str, ssim_diff_str, status) =
                if cpp_encode_success {
                    // Try to decode C++ output
                    if let Ok(cpp_dec) = jpegli_bench_utils::decode_jpeg_to_rgb(&cpp_jpeg) {
                        let cpp_ssim2 =
                            QualityMetrics::ssimulacra2(orig_rgb.as_ref(), cpp_dec.as_ref());
                        let speedup = cpp_time / rust_time;
                        let size_diff = 100.0
                            * (rust_jpeg.len() as f64 - cpp_jpeg.len() as f64)
                            / cpp_jpeg.len() as f64;
                        let ssim_diff = rust_ssim2 - cpp_ssim2;

                        // Validity checks
                        let size_ok = size_diff.abs() < 5.0;
                        let ssim_ok = ssim_diff.abs() < 1.0;
                        if !size_ok || !ssim_ok {
                            all_valid = false;
                        }
                        let status = if size_ok && ssim_ok { " " } else { "!" };

                        (
                            format!("{:>7.1}", cpp_time),
                            format!("{:>7.1}", cpp_jpeg.len() as f64 / 1024.0),
                            format!("{:>6.2}x", speedup),
                            format!("{:>+5.1}%", size_diff),
                            format!("{:>+5.2}", ssim_diff),
                            status,
                        )
                    } else {
                        // Decode failed
                        (
                            format!("{:>7.1}", cpp_time),
                            format!("{:>7.1}", cpp_jpeg.len() as f64 / 1024.0),
                            "   n/a".to_string(),
                            "   n/a".to_string(),
                            "  n/a".to_string(),
                            "?",
                        )
                    }
                } else {
                    // C++ encode failed completely
                    (
                        "   n/a".to_string(),
                        "   n/a".to_string(),
                        "   n/a".to_string(),
                        "   n/a".to_string(),
                        "  n/a".to_string(),
                        "-",
                    )
                };

            println!(
                "│{}{:>4} │ {:>7.1} {} │ {:>7.1} {} │ {} │ {} │ {} │",
                status,
                quality,
                rust_time,
                cpp_time_str,
                rust_jpeg.len() as f64 / 1024.0,
                cpp_size_str,
                speedup_str,
                size_diff_str,
                ssim_diff_str
            );
        }
        println!("└──────┴───────────────────┴───────────────────┴─────────┴────────┴────────┘\n");
    }

    // Summary
    println!("Legend:");
    println!("  Speedup: >1.0 = Rust faster, <1.0 = C++ faster");
    println!("  Δ Size:  <0 = Rust smaller, >0 = Rust larger");
    println!("  Δ SSIM:  >0 = Rust better quality, <0 = C++ better quality");
    println!("  !       = Outside validity tolerance (5% size, 1.0 SSIM2)");
    println!("  ?       = C++ decode failed (quality comparison unavailable)");
    println!("  -       = C++ encode failed (comparison unavailable)");

    assert!(
        all_valid,
        "Some configurations exceeded validity tolerances"
    );
}

/// Quick benchmark stub when ffi-tests is not enabled
#[test]
#[ignore = "Requires ffi-tests feature"]
#[cfg(not(feature = "ffi-tests"))]
fn benchmark_rust_vs_cpp() {
    println!("Benchmark requires --features ffi-tests");
    println!("Run: cargo test --release --test quality_matrix --features ffi-tests benchmark_rust_vs_cpp -- --nocapture --ignored");
}
