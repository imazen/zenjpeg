//! Strict C++ parity enforcement tests.
//!
//! These tests CANNOT be bypassed by loosening thresholds - they track
//! known gaps and require explicit acknowledgment of any regressions.
//!
//! Purpose:
//! 1. Prevent "threshold creep" - cheating tests by loosening limits
//! 2. Track exact parity status for each feature
//! 3. Fail loudly when parity regresses
//! 4. Document expected gaps until they're fixed

use std::fs;
use std::process::Command;

/// Known baseline file size differences (Rust vs C++)
/// These values were measured with matching settings:
/// - 4:4:4 chroma subsampling
/// - No adaptive quantization (--noadaptive_quantization)
/// - Sequential mode (-p 0)
/// - Fixed Huffman codes (--fixed_code)
///
/// IMPORTANT: These baselines must be updated ONLY when:
/// 1. A feature is correctly implemented (gap should shrink)
/// 2. A verified regression is acknowledged
const BASELINE: ParityBaseline = ParityBaseline {
    // Last updated: 2025-12-23
    // Settings: 4:4:4, AQ enabled, sequential, fixed Huffman
    //
    // MEASURED RESULTS (average: 0.70%):
    // - Complex photos: Rust is often SMALLER than C++
    // - Simple graphics: Rust is slightly larger
    // - Flower test image: outlier at +4%
    flower_q90_diff_pct: 4.0, // flower_small: C++=61,476 Rust=63,906 (+4.0%)

    // CID22-512 test images (different complexity levels)
    cid22_large_diff_pct: -1.8, // 1459534.png: C++=173,475 Rust=170,310 (Rust SMALLER!)
    cid22_medium_large_diff_pct: -1.0, // 2504911.png: C++=115,901 Rust=114,760 (Rust SMALLER!)
    cid22_medium_diff_pct: 0.4, // 3616956.png: C++=60,074 Rust=60,320 (nearly identical)
    cid22_small_diff_pct: 2.0,  // nicubunu_Game_baddie_Policeman.png: C++=30,064 Rust=30,653

    // Allowed regression tolerance (must be very small)
    regression_tolerance_pct: 0.5,

    // Target for "done" (matching settings)
    target_diff_pct: 1.0,
};

struct ParityBaseline {
    flower_q90_diff_pct: f64,
    cid22_large_diff_pct: f64,
    cid22_medium_large_diff_pct: f64,
    cid22_medium_diff_pct: f64,
    cid22_small_diff_pct: f64,
    regression_tolerance_pct: f64,
    target_diff_pct: f64,
}

/// CID22-512 test images to fetch from GitHub
const CID22_IMAGES: &[(&str, &str)] = &[
    ("1459534.png", "cid22_large"),        // 621KB - complex photo
    ("2504911.png", "cid22_medium_large"), // 459KB - typical photo
    ("3616956.png", "cid22_medium"),       // 348KB - moderate complexity
    ("nicubunu_Game_baddie_Policeman.png", "cid22_small"), // 77KB - graphics
];

/// GitHub raw URL for imazen/codec-corpus
const CORPUS_BASE_URL: &str = "https://raw.githubusercontent.com/imazen/codec-corpus/main";

/// Download a file from GitHub if not available locally
fn fetch_corpus_image(filename: &str) -> Option<std::path::PathBuf> {
    let cache_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_cache")
        .join("cid22");

    let cache_path = cache_dir.join(filename);

    // Check if already cached
    if cache_path.exists() {
        return Some(cache_path);
    }

    // Check local corpus paths first
    let local_paths = [
        "/mnt/v/work/corpus/CID22-512",
        "/home/lilith/work/codec-comparison/codec-corpus/CID22/CID22-512/training",
        "../codec-comparison/codec-corpus/CID22/CID22-512/training",
    ];

    for local_path in local_paths {
        let full_path = std::path::PathBuf::from(local_path).join(filename);
        if full_path.exists() {
            return Some(full_path);
        }
    }

    // Try to download from GitHub
    fs::create_dir_all(&cache_dir).ok()?;

    let url = format!("{}/CID22/CID22-512/training/{}", CORPUS_BASE_URL, filename);
    eprintln!("Fetching {} from GitHub...", filename);

    let output = Command::new("curl")
        .args(["-fsSL", "-o", cache_path.to_str()?, &url])
        .output()
        .ok()?;

    if output.status.success() && cache_path.exists() {
        eprintln!("  Downloaded: {}", filename);
        Some(cache_path)
    } else {
        eprintln!("  Failed to download: {}", filename);
        None
    }
}

/// Write PPM file for C++ cjpegli
fn write_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;
    file.write_all(rgb)?;
    Ok(())
}

/// Encode with C++ cjpegli (matching settings)
fn encode_cpp(ppm_path: &str, quality: u32) -> Option<Vec<u8>> {
    let cjpegli_path = "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/build/tools/cjpegli";
    if !std::path::Path::new(cjpegli_path).exists() {
        return None;
    }

    let output_path = format!("/tmp/cpp_parity_q{}.jpg", quality);
    let output = Command::new(cjpegli_path)
        .args([
            // Enable adaptive quantization (matches Rust default now)
            "--chroma_subsampling=444",
            "-p",
            "0", // Sequential (no progressive)
            "--fixed_code",
            ppm_path,
            &output_path,
            "-q",
            &quality.to_string(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!("C++ failed: {}", String::from_utf8_lossy(&output.stderr));
        return None;
    }

    fs::read(&output_path).ok()
}

/// Encode with Rust jpegli
fn encode_rust(rgb: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    jpegli::encode::Encoder::new()
        .width(width)
        .height(height)
        .quality(jpegli::quant::Quality::Traditional(quality))
        .encode(rgb)
        .expect("Rust encoding failed")
}

/// Load PNG image
fn load_png(path: &std::path::Path) -> Option<(Vec<u8>, u32, u32)> {
    if !path.exists() {
        return None;
    }

    let decoder = png::Decoder::new(fs::File::open(path).ok()?);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;

    let bytes = &buf[..info.buffer_size()];
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => bytes.chunks(4).flat_map(|c| [c[0], c[1], c[2]]).collect(),
        _ => return None,
    };

    Some((rgb, info.width, info.height))
}

struct ParityResult {
    image_name: String,
    cpp_size: usize,
    rust_size: usize,
    diff_pct: f64,
    baseline_diff_pct: f64,
    status: ParityStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ParityStatus {
    Improved,      // Better than baseline
    Stable,        // Within regression tolerance
    Regressed,     // Worse than baseline + tolerance
    TargetReached, // Within target (parity achieved!)
}

impl ParityResult {
    fn new(image_name: &str, cpp_size: usize, rust_size: usize, baseline_diff_pct: f64) -> Self {
        let diff_pct = 100.0 * (rust_size as f64 - cpp_size as f64) / cpp_size as f64;

        let status = if diff_pct <= BASELINE.target_diff_pct {
            ParityStatus::TargetReached
        } else if diff_pct < baseline_diff_pct - 0.1 {
            ParityStatus::Improved
        } else if diff_pct <= baseline_diff_pct + BASELINE.regression_tolerance_pct {
            ParityStatus::Stable
        } else {
            ParityStatus::Regressed
        };

        Self {
            image_name: image_name.to_string(),
            cpp_size,
            rust_size,
            diff_pct,
            baseline_diff_pct,
            status,
        }
    }

    fn print(&self) {
        let status_icon = match self.status {
            ParityStatus::TargetReached => "✓ TARGET",
            ParityStatus::Improved => "↑ IMPROVED",
            ParityStatus::Stable => "= STABLE",
            ParityStatus::Regressed => "✗ REGRESSED",
        };

        println!(
            "{}: C++={} Rust={} ({:+.1}%, baseline: {:.1}%) [{}]",
            self.image_name,
            self.cpp_size,
            self.rust_size,
            self.diff_pct,
            self.baseline_diff_pct,
            status_icon
        );
    }
}

/// Test a single image and return result
fn test_image(
    png_path: &std::path::Path,
    image_name: &str,
    quality: u32,
    baseline_diff_pct: f64,
) -> Option<ParityResult> {
    let (rgb, width, height) = load_png(png_path)?;

    let ppm_path = format!("/tmp/parity_{}.ppm", image_name);
    write_ppm(&ppm_path, &rgb, width as usize, height as usize).ok()?;

    let cpp_jpeg = encode_cpp(&ppm_path, quality)?;
    let rust_jpeg = encode_rust(&rgb, width, height, quality as f32);

    Some(ParityResult::new(
        &format!("{}_q{}", image_name, quality),
        cpp_jpeg.len(),
        rust_jpeg.len(),
        baseline_diff_pct,
    ))
}

/// Test image on flower_small at Q90
#[test]
#[ignore = "requires C++ cjpegli build and test images"]
fn test_parity_flower_q90() {
    let png_path = std::path::PathBuf::from(
        "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png",
    );

    let result = match test_image(&png_path, "flower", 90, BASELINE.flower_q90_diff_pct) {
        Some(r) => r,
        None => {
            println!("Skipping: flower test image or C++ cjpegli not available");
            return;
        }
    };

    result.print();

    assert_ne!(
        result.status,
        ParityStatus::Regressed,
        "PARITY REGRESSION: {} is {:.1}% larger than C++, was {:.1}% in baseline",
        result.image_name,
        result.diff_pct,
        result.baseline_diff_pct
    );
}

/// Test CID22 large image (complex photo, ~621KB PNG)
#[test]
#[ignore = "requires C++ cjpegli build, downloads test image if needed"]
fn test_parity_cid22_large() {
    let png_path = match fetch_corpus_image("1459534.png") {
        Some(p) => p,
        None => {
            println!("Skipping: 1459534.png not available");
            return;
        }
    };

    let result = match test_image(&png_path, "cid22_large", 90, BASELINE.cid22_large_diff_pct) {
        Some(r) => r,
        None => {
            println!("Skipping: C++ cjpegli not available");
            return;
        }
    };

    result.print();

    assert_ne!(
        result.status,
        ParityStatus::Regressed,
        "PARITY REGRESSION: {} is {:.1}% larger than C++, was {:.1}% in baseline",
        result.image_name,
        result.diff_pct,
        result.baseline_diff_pct
    );
}

/// Test CID22 medium-large image (typical photo, ~459KB PNG)
#[test]
#[ignore = "requires C++ cjpegli build, downloads test image if needed"]
fn test_parity_cid22_medium_large() {
    let png_path = match fetch_corpus_image("2504911.png") {
        Some(p) => p,
        None => {
            println!("Skipping: 2504911.png not available");
            return;
        }
    };

    let result = match test_image(
        &png_path,
        "cid22_medium_large",
        90,
        BASELINE.cid22_medium_large_diff_pct,
    ) {
        Some(r) => r,
        None => {
            println!("Skipping: C++ cjpegli not available");
            return;
        }
    };

    result.print();

    assert_ne!(
        result.status,
        ParityStatus::Regressed,
        "PARITY REGRESSION: {} is {:.1}% larger than C++, was {:.1}% in baseline",
        result.image_name,
        result.diff_pct,
        result.baseline_diff_pct
    );
}

/// Test CID22 medium image (moderate complexity, ~348KB PNG)
#[test]
#[ignore = "requires C++ cjpegli build, downloads test image if needed"]
fn test_parity_cid22_medium() {
    let png_path = match fetch_corpus_image("3616956.png") {
        Some(p) => p,
        None => {
            println!("Skipping: 3616956.png not available");
            return;
        }
    };

    let result = match test_image(
        &png_path,
        "cid22_medium",
        90,
        BASELINE.cid22_medium_diff_pct,
    ) {
        Some(r) => r,
        None => {
            println!("Skipping: C++ cjpegli not available");
            return;
        }
    };

    result.print();

    assert_ne!(
        result.status,
        ParityStatus::Regressed,
        "PARITY REGRESSION: {} is {:.1}% larger than C++, was {:.1}% in baseline",
        result.image_name,
        result.diff_pct,
        result.baseline_diff_pct
    );
}

/// Test CID22 small image (graphics, ~77KB PNG)
#[test]
#[ignore = "requires C++ cjpegli build, downloads test image if needed"]
fn test_parity_cid22_small() {
    let png_path = match fetch_corpus_image("nicubunu_Game_baddie_Policeman.png") {
        Some(p) => p,
        None => {
            println!("Skipping: nicubunu_Game_baddie_Policeman.png not available");
            return;
        }
    };

    let result = match test_image(&png_path, "cid22_small", 90, BASELINE.cid22_small_diff_pct) {
        Some(r) => r,
        None => {
            println!("Skipping: C++ cjpegli not available");
            return;
        }
    };

    result.print();

    assert_ne!(
        result.status,
        ParityStatus::Regressed,
        "PARITY REGRESSION: {} is {:.1}% larger than C++, was {:.1}% in baseline",
        result.image_name,
        result.diff_pct,
        result.baseline_diff_pct
    );
}

/// Comprehensive parity report for all test images
#[test]
#[ignore = "requires C++ cjpegli build and test images"]
fn test_parity_comprehensive() {
    println!("\n=== C++ PARITY ENFORCEMENT TEST ===\n");
    println!("Settings: 4:4:4, AQ enabled, sequential, fixed Huffman");
    println!(
        "Target: <{:.1}% file size difference\n",
        BASELINE.target_diff_pct
    );

    let mut results = Vec::new();
    let mut regressions = Vec::new();
    let mut targets_reached = Vec::new();

    // Test flower
    let flower_path = std::path::PathBuf::from(
        "/home/lilith/work/jpegli-rs/internal/jpegli-cpp/testdata/jxl/flower/flower_small.rgb.png",
    );
    if let Some(result) = test_image(&flower_path, "flower", 90, BASELINE.flower_q90_diff_pct) {
        results.push(result);
    }

    // Test CID22 images (fetch if needed)
    let cid22_tests = [
        ("1459534.png", "cid22_large", BASELINE.cid22_large_diff_pct),
        (
            "2504911.png",
            "cid22_medium_large",
            BASELINE.cid22_medium_large_diff_pct,
        ),
        (
            "3616956.png",
            "cid22_medium",
            BASELINE.cid22_medium_diff_pct,
        ),
        (
            "nicubunu_Game_baddie_Policeman.png",
            "cid22_small",
            BASELINE.cid22_small_diff_pct,
        ),
    ];

    for (filename, name, baseline) in cid22_tests {
        if let Some(png_path) = fetch_corpus_image(filename) {
            if let Some(result) = test_image(&png_path, name, 90, baseline) {
                results.push(result);
            }
        } else {
            println!("WARNING: {} not available", filename);
        }
    }

    if results.is_empty() {
        println!("No test images available. Run with test images or check network.");
        return;
    }

    // Print all results
    println!("--- Results ---\n");
    for result in &results {
        result.print();

        if result.status == ParityStatus::Regressed {
            regressions.push(result.image_name.clone());
        }
        if result.status == ParityStatus::TargetReached {
            targets_reached.push(result.image_name.clone());
        }
    }

    // Summary
    println!("\n--- Summary ---\n");
    println!("Total images tested: {}", results.len());
    println!(
        "Targets reached (<{:.1}%): {}",
        BASELINE.target_diff_pct,
        targets_reached.len()
    );
    println!("Regressions: {}", regressions.len());

    if !targets_reached.is_empty() {
        println!("\nTargets reached: {:?}", targets_reached);
    }

    // Calculate overall status
    let avg_diff: f64 = results.iter().map(|r| r.diff_pct).sum::<f64>() / results.len() as f64;
    println!("\nAverage diff: {:.2}%", avg_diff);

    // STRICT CHECK: No regressions allowed
    assert!(
        regressions.is_empty(),
        "PARITY REGRESSIONS DETECTED: {:?}\n\
         This indicates the Rust encoder got WORSE, not better.\n\
         If this is intentional, update BASELINE values with justification.",
        regressions
    );

    // Progress tracking
    if targets_reached.len() == results.len() {
        println!("\n🎉 ALL TARGETS REACHED! Matching settings parity achieved!");
    } else {
        println!(
            "\n📊 Progress: {}/{} targets reached",
            targets_reached.len(),
            results.len()
        );
        println!("   Remaining gap likely due to: DCT precision, entropy coding details");
    }
}

/// Track the gap breakdown by feature
#[test]
#[ignore = "requires C++ cjpegli build"]
fn test_parity_gap_breakdown() {
    println!("\n=== PARITY GAP BREAKDOWN ===\n");

    println!("Feature Implementation Status:");
    println!("───────────────────────────────────────────────────");
    println!("│ Feature               │ Status    │ Est. Impact │");
    println!("───────────────────────────────────────────────────");
    println!("│ Quantization tables   │ ✓ Match   │   0%        │");
    println!("│ Huffman tree building │ ✓ Match   │   0%        │");
    println!("│ Zero-biasing          │ ✓ Match   │   0% (no AQ)│");
    println!("│ Adaptive quantization │ ✗ NOT IMPL│  ~3-4%      │");
    println!("│ Progressive encoding  │ ✗ NOT IMPL│  ~2-3%      │");
    println!("│ Huffman optimization  │ ✗ NOT IMPL│  ~3-4%      │");
    println!("│ DCT precision         │ ~ Varies  │ -2% to +4%  │");
    println!("│ Entropy details       │ ~ Varies  │   variable  │");
    println!("───────────────────────────────────────────────────");
    println!("");
    println!("MEASURED RESULTS (with matching settings):");
    println!("  - Average gap: 0.70% (Rust usually within ±2% of C++)");
    println!("  - Complex photos: Rust often SMALLER than C++");
    println!("  - Simple graphics: Rust slightly larger (+2%)");
    println!("  - Flower test image: outlier at +4%");
    println!("");
    println!("IMPORTANT: This is with MATCHING settings.");
    println!("C++ default enables AQ, progressive, optimized Huffman.");
    println!("Default C++ would be ~10-15% smaller than Rust currently.");
}
