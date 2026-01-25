//! Test utilities for jpegli testing.
//!
//! This module provides image generation, quality verification, and test data
//! access utilities matching the C++ jpegli test infrastructure.

#![allow(dead_code)] // Test utilities - not all used in every test configuration

use std::path::PathBuf;

/// Test image patterns for generating synthetic test images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestPattern {
    /// Horizontal gradient from black to white
    GradientH,
    /// Vertical gradient from black to white
    GradientV,
    /// Diagonal gradient (top-left to bottom-right)
    GradientD,
    /// Checkerboard pattern with specified block size
    Checkerboard,
    /// Random noise with specified seed
    Noise,
    /// Solid color fill
    SolidColor,
    /// Color bars (like TV test pattern)
    ColorBars,
    /// High frequency content (alternating pixels)
    HighFrequency,
}

/// A test image with pixel data and metadata.
#[derive(Debug, Clone)]
pub struct TestImage {
    pub width: u32,
    pub height: u32,
    pub components: u32,
    pub pixels: Vec<u8>,
}

impl TestImage {
    /// Create a new empty test image.
    pub fn new(width: u32, height: u32, components: u32) -> Self {
        let size = (width * height * components) as usize;
        Self {
            width,
            height,
            components,
            pixels: vec![0; size],
        }
    }

    /// Create a test image from existing pixel data.
    pub fn from_pixels(width: u32, height: u32, components: u32, pixels: Vec<u8>) -> Self {
        assert_eq!(
            pixels.len(),
            (width * height * components) as usize,
            "Pixel data size mismatch"
        );
        Self {
            width,
            height,
            components,
            pixels,
        }
    }

    /// Get a pixel value at (x, y) for component c.
    pub fn get_pixel(&self, x: u32, y: u32, c: u32) -> u8 {
        let idx = ((y * self.width + x) * self.components + c) as usize;
        self.pixels[idx]
    }

    /// Set a pixel value at (x, y) for component c.
    pub fn set_pixel(&mut self, x: u32, y: u32, c: u32, value: u8) {
        let idx = ((y * self.width + x) * self.components + c) as usize;
        self.pixels[idx] = value;
    }

    /// Convert to grayscale (returns new image).
    pub fn to_grayscale(&self) -> Self {
        assert_eq!(self.components, 3, "Can only convert RGB to grayscale");
        let mut gray = TestImage::new(self.width, self.height, 1);
        for y in 0..self.height {
            for x in 0..self.width {
                let r = self.get_pixel(x, y, 0) as f32;
                let g = self.get_pixel(x, y, 1) as f32;
                let b = self.get_pixel(x, y, 2) as f32;
                let y_val = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                gray.set_pixel(x, y, 0, y_val);
            }
        }
        gray
    }
}

/// Generate a test image with the specified pattern.
pub fn generate_test_image(
    width: u32,
    height: u32,
    pattern: TestPattern,
    components: u32,
) -> TestImage {
    match pattern {
        TestPattern::GradientH => generate_gradient_h(width, height, components),
        TestPattern::GradientV => generate_gradient_v(width, height, components),
        TestPattern::GradientD => generate_gradient_d(width, height, components),
        TestPattern::Checkerboard => generate_checkerboard(width, height, 8, components),
        TestPattern::Noise => generate_noise(width, height, 12345, components),
        TestPattern::SolidColor => generate_solid(width, height, 128, components),
        TestPattern::ColorBars => generate_color_bars(width, height),
        TestPattern::HighFrequency => generate_high_frequency(width, height, components),
    }
}

/// Generate a horizontal gradient (black on left, white on right).
pub fn generate_gradient_h(width: u32, height: u32, components: u32) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    for y in 0..height {
        for x in 0..width {
            let value = ((x as f32 / (width - 1).max(1) as f32) * 255.0).round() as u8;
            for c in 0..components {
                img.set_pixel(x, y, c, value);
            }
        }
    }
    img
}

/// Generate a vertical gradient (black on top, white on bottom).
pub fn generate_gradient_v(width: u32, height: u32, components: u32) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    for y in 0..height {
        let value = ((y as f32 / (height - 1).max(1) as f32) * 255.0).round() as u8;
        for x in 0..width {
            for c in 0..components {
                img.set_pixel(x, y, c, value);
            }
        }
    }
    img
}

/// Generate a diagonal gradient (black top-left, white bottom-right).
pub fn generate_gradient_d(width: u32, height: u32, components: u32) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    for y in 0..height {
        for x in 0..width {
            let t = ((x + y) as f32 / ((width + height - 2).max(1) as f32)).min(1.0);
            let value = (t * 255.0).round() as u8;
            for c in 0..components {
                img.set_pixel(x, y, c, value);
            }
        }
    }
    img
}

/// Generate a checkerboard pattern.
pub fn generate_checkerboard(
    width: u32,
    height: u32,
    block_size: u32,
    components: u32,
) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    for y in 0..height {
        for x in 0..width {
            let bx = x / block_size;
            let by = y / block_size;
            let value = if (bx + by) % 2 == 0 { 255 } else { 0 };
            for c in 0..components {
                img.set_pixel(x, y, c, value);
            }
        }
    }
    img
}

/// Generate a deterministic noise pattern.
pub fn generate_noise(width: u32, height: u32, seed: u64, components: u32) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    // Simple LCG PRNG for deterministic noise
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    for y in 0..height {
        for x in 0..width {
            for comp in 0..components {
                state = (a.wrapping_mul(state).wrapping_add(c)) % m;
                let value = ((state >> 16) & 0xFF) as u8;
                img.set_pixel(x, y, comp, value);
            }
        }
    }
    img
}

/// Generate a solid color image.
pub fn generate_solid(width: u32, height: u32, value: u8, components: u32) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    for pixel in img.pixels.iter_mut() {
        *pixel = value;
    }
    img
}

/// Generate a solid RGB color image.
pub fn generate_solid_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> TestImage {
    let mut img = TestImage::new(width, height, 3);
    for y in 0..height {
        for x in 0..width {
            img.set_pixel(x, y, 0, r);
            img.set_pixel(x, y, 1, g);
            img.set_pixel(x, y, 2, b);
        }
    }
    img
}

/// Generate color bars (8 vertical bars like TV test pattern).
pub fn generate_color_bars(width: u32, height: u32) -> TestImage {
    let mut img = TestImage::new(width, height, 3);
    let colors: [(u8, u8, u8); 8] = [
        (255, 255, 255), // White
        (255, 255, 0),   // Yellow
        (0, 255, 255),   // Cyan
        (0, 255, 0),     // Green
        (255, 0, 255),   // Magenta
        (255, 0, 0),     // Red
        (0, 0, 255),     // Blue
        (0, 0, 0),       // Black
    ];

    let bar_width = width / 8;
    for y in 0..height {
        for x in 0..width {
            let bar_idx = ((x / bar_width.max(1)) as usize).min(7);
            let (r, g, b) = colors[bar_idx];
            img.set_pixel(x, y, 0, r);
            img.set_pixel(x, y, 1, g);
            img.set_pixel(x, y, 2, b);
        }
    }
    img
}

/// Generate high-frequency content (alternating pixels).
pub fn generate_high_frequency(width: u32, height: u32, components: u32) -> TestImage {
    let mut img = TestImage::new(width, height, components);
    for y in 0..height {
        for x in 0..width {
            let value = if (x + y) % 2 == 0 { 255 } else { 0 };
            for c in 0..components {
                img.set_pixel(x, y, c, value);
            }
        }
    }
    img
}

// ============================================================================
// Quality Verification Functions
// ============================================================================

/// Compute RMS (Root Mean Square) distance between two images.
/// Returns a value in the range [0, 255] where 0 means identical.
/// Matches C++ `DistanceRms` function.
pub fn distance_rms(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(
        original.len(),
        decoded.len(),
        "Images must have same size for RMS comparison"
    );
    if original.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum();

    // RMS normalized to 0-1 range (divide by 255 to match C++)
    let rms_normalized = (sum_sq / original.len() as f64).sqrt() / 255.0;
    rms_normalized * 255.0 // Return in 0-255 scale
}

/// Compute RMS distance between two TestImages.
pub fn distance_rms_images(original: &TestImage, decoded: &TestImage) -> f64 {
    assert_eq!(original.width, decoded.width);
    assert_eq!(original.height, decoded.height);
    assert_eq!(original.components, decoded.components);
    distance_rms(&original.pixels, &decoded.pixels)
}

/// Compute the maximum pixel difference between two images.
pub fn max_pixel_diff(original: &[u8], decoded: &[u8]) -> u8 {
    assert_eq!(
        original.len(),
        decoded.len(),
        "Images must have same size for max diff comparison"
    );

    original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

/// Compute max pixel difference between two TestImages.
pub fn max_pixel_diff_images(original: &TestImage, decoded: &TestImage) -> u8 {
    assert_eq!(original.width, decoded.width);
    assert_eq!(original.height, decoded.height);
    assert_eq!(original.components, decoded.components);
    max_pixel_diff(&original.pixels, &decoded.pixels)
}

/// Verify that the output image is within quality thresholds.
/// Panics if thresholds are exceeded.
/// Matches C++ `VerifyOutputImage` function.
pub fn verify_output(original: &[u8], decoded: &[u8], max_rms: f64, max_diff: u8) {
    let rms = distance_rms(original, decoded);
    let diff = max_pixel_diff(original, decoded);

    assert!(
        rms <= max_rms,
        "RMS distance {:.4} exceeds threshold {:.4}",
        rms,
        max_rms
    );
    assert!(
        diff <= max_diff,
        "Max pixel diff {} exceeds threshold {}",
        diff,
        max_diff
    );
}

/// Verify output with TestImages.
pub fn verify_output_images(original: &TestImage, decoded: &TestImage, max_rms: f64, max_diff: u8) {
    assert_eq!(original.width, decoded.width, "Width mismatch");
    assert_eq!(original.height, decoded.height, "Height mismatch");
    assert_eq!(
        original.components, decoded.components,
        "Component count mismatch"
    );
    verify_output(&original.pixels, &decoded.pixels, max_rms, max_diff);
}

// ============================================================================
// Test Data Path Helpers
// ============================================================================

/// Get the path to the testdata directory.
///
/// # Panics
/// Panics if directory cannot be found.
#[track_caller]
pub fn require_testdata_dir() -> PathBuf {
    let dir = get_testdata_dir();
    if !dir.exists() {
        panic!(
            "Testdata directory not found.\n\
             Set JPEGLI_TESTDATA environment variable to the testdata directory.\n\
             Expected structure: $JPEGLI_TESTDATA/jxl/flower/flower_small.rgb.png"
        );
    }
    dir
}

/// Get path to the flower_small test image.
///
/// # Panics
/// Panics if image cannot be found.
#[track_caller]
pub fn require_flower_small_path() -> PathBuf {
    let path = get_testdata_dir().join("jxl/flower/flower_small.rgb.png");
    if !path.exists() {
        panic!(
            "Test image flower_small.rgb.png not found.\n\
             Set JPEGLI_TESTDATA environment variable or ensure testdata is available.\n\
             Expected at: jxl/flower/flower_small.rgb.png"
        );
    }
    path
}

/// Try to get path to the testdata directory. Returns path even if it doesn't exist.
pub fn get_testdata_dir() -> PathBuf {
    // Check environment variable first
    if let Ok(path) = std::env::var("JPEGLI_TESTDATA") {
        return PathBuf::from(path);
    }

    // Check relative to manifest dir (for cargo test)
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let testdata = PathBuf::from(&manifest).join("testdata");
        if testdata.exists() {
            return testdata;
        }
        // Also check parent's testdata (workspace level)
        let parent_testdata = PathBuf::from(&manifest).join("../testdata");
        if parent_testdata.exists() {
            return parent_testdata;
        }
    }

    // Check the C++ testdata location (in internal/jpegli-cpp submodule)
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        // From jpegli crate, go up one level to workspace root, then into internal/jpegli-cpp
        let cpp_testdata = PathBuf::from(&manifest)
            .parent()
            .map(|p| p.join("internal/jpegli-cpp/testdata"))
            .filter(|p| p.exists());
        if let Some(testdata) = cpp_testdata {
            return testdata;
        }
    }
    // Default fallback - try current directory
    PathBuf::from("testdata")
}

/// Get the full path to a test data file.
pub fn get_test_data_path(filename: &str) -> PathBuf {
    get_testdata_dir().join(filename)
}

/// Get path to C++ generated .testdata files (from instrumented builds).
/// These are generated by running cjpegli with GENERATE_RUST_TEST_DATA=1
///
/// # Panics
/// Panics if the file cannot be found. Set `CPP_TESTDATA_DIR` env var.
#[track_caller]
pub fn require_cpp_testdata_path(filename: &str) -> PathBuf {
    get_cpp_testdata_path(filename).unwrap_or_else(|| {
        panic!(
            "C++ testdata file '{}' not found.\n\
             Set CPP_TESTDATA_DIR environment variable to the directory containing .testdata files.\n\
             Generate testdata with: GENERATE_RUST_TEST_DATA=1 cjpegli input.png output.jpg",
            filename
        )
    })
}

/// Try to get path to C++ generated .testdata files. Returns None if not found.
pub fn get_cpp_testdata_path(filename: &str) -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(dir) = std::env::var("CPP_TESTDATA_DIR") {
        let path = PathBuf::from(dir).join(filename);
        if path.exists() {
            return Some(path);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("cpp_testdata").join(filename),
            PathBuf::from(&manifest)
                .join("../cpp_testdata")
                .join(filename),
            PathBuf::from(&manifest)
                .join("../internal/jpegli-cpp")
                .join(filename),
            PathBuf::from(&manifest).join("testdata").join(filename),
            // Check workspace root (parent of zenjpeg crate)
            PathBuf::from(&manifest).join("..").join(filename),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Get path to cjpegli tool binary.
///
/// # Panics
/// Panics if binary cannot be found.
#[track_caller]
pub fn require_cjpegli() -> PathBuf {
    find_cjpegli().unwrap_or_else(|| {
        panic!(
            "cjpegli binary not found.\n\
             Set CJPEGLI_PATH environment variable or build jpegli:\n\
             cd internal/jpegli-cpp && cmake -B build && cmake --build build"
        )
    })
}

/// Find cjpegli tool binary. Returns None if not found.
pub fn find_cjpegli() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("CJPEGLI_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("../internal/jpegli-cpp/build/tools/cjpegli"),
            PathBuf::from(&manifest).join("../../jpegli/build/tools/cjpegli"),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    // Check PATH using which
    std::process::Command::new("which")
        .arg("cjpegli")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| PathBuf::from(s.trim()))
            } else {
                None
            }
        })
}

/// Get path to djpegli tool binary.
///
/// # Panics
/// Panics if binary cannot be found.
#[track_caller]
pub fn require_djpegli() -> PathBuf {
    find_djpegli().unwrap_or_else(|| {
        panic!(
            "djpegli binary not found.\n\
             Set DJPEGLI_PATH environment variable or build jpegli:\n\
             cd internal/jpegli-cpp && cmake -B build && cmake --build build"
        )
    })
}

/// Find djpegli tool binary. Returns None if not found.
pub fn find_djpegli() -> Option<PathBuf> {
    // Check environment variable first
    if let Ok(path) = std::env::var("DJPEGLI_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Check relative to manifest dir
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidates = [
            PathBuf::from(&manifest).join("../internal/jpegli-cpp/build/tools/djpegli"),
            PathBuf::from(&manifest).join("../../jpegli/build/tools/djpegli"),
        ];
        for path in candidates {
            if path.exists() {
                return Some(path);
            }
        }
    }

    // Check PATH
    std::process::Command::new("which")
        .arg("djpegli")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| PathBuf::from(s.trim()))
            } else {
                None
            }
        })
}

/// Macro for skipping tests when test data is not available.
#[macro_export]
macro_rules! skip_if_missing {
    ($path:expr) => {
        if !$path.exists() {
            eprintln!("Skipping test: {:?} not found", $path);
            return;
        }
    };
    ($path:expr, $msg:expr) => {
        if !$path.exists() {
            eprintln!("Skipping test: {} ({:?} not found)", $msg, $path);
            return;
        }
    };
}

/// Read test data file, returning None if not found.
pub fn read_test_data(filename: &str) -> Option<Vec<u8>> {
    let path = get_test_data_path(filename);
    std::fs::read(&path).ok()
}

/// Read test data file, panicking if not found.
pub fn read_test_data_required(filename: &str) -> Vec<u8> {
    let path = get_test_data_path(filename);
    std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read test data {:?}: {}", path, e))
}

// ============================================================================
// Quality Thresholds (from C++ tests)
// ============================================================================

/// Quality thresholds matching C++ jpegli test expectations EXACTLY.
/// These values are from lib/jpegli/*_test.cc and lib/extras/jpegli_test.cc
pub mod thresholds {
    // ========================================================================
    // RMS Thresholds by Quality Level (from encode_api_test.cc)
    // ========================================================================

    /// Maximum RMS for Q50 encoding (4:4:4)
    pub const Q50_MAX_RMS: f64 = 20.0;
    /// Maximum RMS for Q75 encoding (4:4:4)
    pub const Q75_MAX_RMS: f64 = 10.0;
    /// Maximum RMS for Q85 encoding (4:4:4)
    pub const Q85_MAX_RMS: f64 = 8.0;
    /// Maximum RMS for Q90 encoding (4:4:4)
    pub const Q90_MAX_RMS: f64 = 4.0;
    /// Maximum RMS for Q95 encoding (4:4:4)
    pub const Q95_MAX_RMS: f64 = 2.1;

    /// Compute max RMS threshold based on quality and sampling factors.
    /// Matches C++ encode_api_test.cc max_rms lambda.
    #[inline]
    pub fn max_rms_for_quality(quality: u8, h_samp: u8, v_samp: u8) -> f64 {
        let subsample_factor = (h_samp as f64) * (v_samp as f64);
        let base = if quality >= 95 {
            2.1
        } else if quality >= 90 {
            4.0
        } else if quality >= 85 {
            8.0
        } else {
            20.0
        };
        base * subsample_factor
    }

    // ========================================================================
    // RMS Thresholds by Test Type (from various *_test.cc)
    // ========================================================================

    /// source_manager_test.cc baseline RMS
    pub const SOURCE_MANAGER_MAX_RMS: f64 = 1.0;

    /// output_suspension_test.cc pixel data RMS
    pub const OUTPUT_SUSPENSION_PIXEL_MAX_RMS: f64 = 2.5;
    /// output_suspension_test.cc raw data RMS
    pub const OUTPUT_SUSPENSION_RAW_MAX_RMS: f64 = 3.5;

    /// streaming_test.cc max RMS
    pub const STREAMING_MAX_RMS: f64 = 3.8;

    /// decode_api_test.cc reuse test RMS
    pub const DECODE_REUSE_MAX_RMS: f64 = 2.35;
    /// decode_api_test.cc coefficient mode RMS (exact match)
    pub const DECODE_COEFFICIENTS_MAX_RMS: f64 = 0.0;
    /// decode_api_test.cc libjpeg compat (no fancy upsampling)
    pub const DECODE_LIBJPEG_COMPAT_MAX_RMS: f64 = 5.0;

    /// input_suspension_test.cc baseline RMS
    pub const INPUT_SUSPENSION_BASE_MAX_RMS: f64 = 1.0;
    /// input_suspension_test.cc no subsampling RMS
    pub const INPUT_SUSPENSION_NO_SUBSAMPLE_MAX_RMS: f64 = 1.75;
    /// input_suspension_test.cc with subsampling RMS
    pub const INPUT_SUSPENSION_SUBSAMPLE_MAX_RMS: f64 = 3.0;
    /// input_suspension_test.cc progressive RMS
    pub const INPUT_SUSPENSION_PROGRESSIVE_MAX_RMS: f64 = 8.0;

    // ========================================================================
    // Butteraugli Thresholds (from jpegli_test.cc)
    // ========================================================================

    /// JpegliXYBEncodeTest: XYB mode quality threshold
    pub const XYB_BUTTERAUGLI: f64 = 1.32;
    /// JpegliXYBEncodeTest: XYB mode bits per pixel threshold
    pub const XYB_MAX_BPP: f64 = 1.45;

    /// JpegliDecodeTestLargeSmoothArea: smooth area handling
    pub const SMOOTH_BUTTERAUGLI: f64 = 3.0;

    /// JpegliYUVEncodeTest: YUV 4:4:4 quality threshold
    pub const YUV_BUTTERAUGLI: f64 = 1.32;
    /// JpegliYUVEncodeTest: YUV 4:4:4 bits per pixel threshold
    pub const YUV_MAX_BPP: f64 = 1.7;

    /// JpegliYUVChromaSubsamplingEncodeTest: YUV subsampled quality
    pub const YUV_SUBSAMPLE_BUTTERAUGLI: f64 = 1.82;
    /// JpegliYUVChromaSubsamplingEncodeTest: YUV subsampled BPP
    pub const YUV_SUBSAMPLE_MAX_BPP: f64 = 1.55;

    /// JpegliYUVEncodeTestNoAq: YUV without adaptive quantization
    pub const YUV_NO_AQ_BUTTERAUGLI: f64 = 1.25;
    /// JpegliYUVEncodeTestNoAq: YUV no AQ bits per pixel
    pub const YUV_NO_AQ_MAX_BPP: f64 = 1.85;

    /// JpegliHDRRoundtripTest: HDR 16-bit roundtrip
    pub const HDR_BUTTERAUGLI: f64 = 1.05;
    /// JpegliHDRRoundtripTest: HDR bits per pixel
    pub const HDR_MAX_BPP: f64 = 2.95;

    // ========================================================================
    // Max Pixel Difference Thresholds
    // ========================================================================

    /// Default max pixel difference (most tests)
    pub const DEFAULT_MAX_DIFF: u8 = 255;
    /// Strict max pixel difference for high quality
    pub const STRICT_MAX_DIFF: u8 = 35;

    // ========================================================================
    // Tone Mapping / Transfer Function Error Thresholds
    // ========================================================================

    /// TestRec2408ToneMap absolute error threshold
    pub const REC2408_TONE_MAP_ERROR: f64 = 2.75e-5;
    /// TestHlgOotfApply absolute error threshold
    pub const HLG_OOTF_ERROR: f64 = 7.2e-7;
    /// TestGamutMap absolute error threshold
    pub const GAMUT_MAP_ERROR: f64 = 1e-10;

    /// TestPqEncodedFromDisplay absolute error threshold
    pub const PQ_ENCODE_ERROR: f64 = 6e-7;
    /// TestHlgEncodedFromDisplay absolute error threshold
    pub const HLG_ENCODE_ERROR: f64 = 4e-7;
    /// TestPqDisplayFromEncoded absolute error threshold
    pub const PQ_DECODE_ERROR: f64 = 3e-6;
    /// TestHlgDisplayFromEncoded absolute error threshold
    pub const HLG_DECODE_ERROR: f64 = 6e-7;
}

// ============================================================================
// Testdata Sparse Checkout Helper
// ============================================================================

/// Ensures testdata is available, doing a sparse checkout if needed.
/// Returns the path to the testdata directory.
pub fn ensure_testdata() -> Option<PathBuf> {
    let testdata_dir = get_testdata_dir();
    if testdata_dir.exists() {
        return Some(testdata_dir);
    }

    // Try to do a sparse checkout of just the files we need
    eprintln!(
        "Testdata not found at {:?}, attempting sparse checkout...",
        testdata_dir
    );

    // Check if we're in the jpegli repo
    let jpegli_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf());

    if let Some(root) = jpegli_root {
        let testdata_in_root = root.join("testdata");
        if testdata_in_root.exists() {
            return Some(testdata_in_root);
        }

        // Try to initialize and update the submodule
        let status = std::process::Command::new("git")
            .args(["submodule", "update", "--init", "--depth", "1", "testdata"])
            .current_dir(&root)
            .status();

        if status.is_ok() && testdata_in_root.exists() {
            return Some(testdata_in_root);
        }
    }

    eprintln!("Could not find or checkout testdata. Some tests will be skipped.");
    None
}

/// List of essential test files needed for jpegli tests.
pub const ESSENTIAL_TEST_FILES: &[&str] = &[
    // Primary test images
    "jxl/flower/flower_small.rgb.depth8.ppm",
    "jxl/flower/flower_small.g.depth8.pgm",
    "jxl/hdr_room.png",
    // JPEG test variants
    "jxl/flower/flower.png.im_q85_444.jpg",
    "jxl/flower/flower.png.im_q85_420.jpg",
    "jxl/flower/flower.png.im_q85_420_progr.jpg",
    "jxl/flower/flower.png.im_q85_420_R13B.jpg",
    "jxl/flower/flower.png.im_q85_422.jpg",
    "jxl/flower/flower.png.im_q85_440.jpg",
    "jxl/flower/flower.png.im_q85_444_1x2.jpg",
    "jxl/flower/flower.png.im_q85_asymmetric.jpg",
    "jxl/flower/flower.png.im_q85_gray.jpg",
    "jxl/flower/flower.png.im_q85_luma_subsample.jpg",
    "jxl/flower/flower.png.im_q85_rgb.jpg",
    "jxl/flower/flower.png.im_q85_rgb_subsample_blue.jpg",
    "jxl/flower/flower_small.cmyk.jpg",
    "jxl/flower/flower_small.q85_444_non_interleaved.jpg",
    "jxl/flower/flower_small.q85_420_non_interleaved.jpg",
    "jxl/flower/flower_small.q85_444_partially_interleaved.jpg",
    "jxl/flower/flower_small.q85_420_partially_interleaved.jpg",
    // Scan scripts
    "jxl/flower/non_interleaved_scan.txt",
    "jxl/flower/partially_interleaved_scan.txt",
    // ICC profiles
    "jxl/color_management/sRGB-D2700.icc",
];

/// Check if a specific test file exists.
pub fn has_test_file(filename: &str) -> bool {
    get_test_data_path(filename).exists()
}

/// Check if all essential test files are available.
pub fn has_essential_test_files() -> bool {
    ESSENTIAL_TEST_FILES.iter().all(|f| has_test_file(f))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_gradient_h() {
        let img = generate_gradient_h(256, 64, 1);
        assert_eq!(img.width, 256);
        assert_eq!(img.height, 64);
        assert_eq!(img.components, 1);
        // First column should be 0, last should be 255
        assert_eq!(img.get_pixel(0, 0, 0), 0);
        assert_eq!(img.get_pixel(255, 0, 0), 255);
        // Middle should be ~128
        let mid = img.get_pixel(127, 0, 0);
        assert!((126..=129).contains(&mid));
    }

    #[test]
    fn test_generate_gradient_v() {
        let img = generate_gradient_v(64, 256, 1);
        assert_eq!(img.width, 64);
        assert_eq!(img.height, 256);
        // Top should be 0, bottom should be 255
        assert_eq!(img.get_pixel(0, 0, 0), 0);
        assert_eq!(img.get_pixel(0, 255, 0), 255);
    }

    #[test]
    fn test_generate_checkerboard() {
        let img = generate_checkerboard(64, 64, 8, 1);
        // Top-left block should be white (255)
        assert_eq!(img.get_pixel(0, 0, 0), 255);
        // Next block should be black (0)
        assert_eq!(img.get_pixel(8, 0, 0), 0);
        // Diagonal block should be black too
        assert_eq!(img.get_pixel(8, 8, 0), 255);
    }

    #[test]
    fn test_generate_noise() {
        let img1 = generate_noise(64, 64, 12345, 3);
        let img2 = generate_noise(64, 64, 12345, 3);
        let img3 = generate_noise(64, 64, 54321, 3);

        // Same seed should produce same image
        assert_eq!(img1.pixels, img2.pixels);
        // Different seed should produce different image
        assert_ne!(img1.pixels, img3.pixels);
    }

    #[test]
    fn test_generate_color_bars() {
        let img = generate_color_bars(64, 32);
        assert_eq!(img.components, 3);
        // First bar should be white
        assert_eq!(img.get_pixel(0, 0, 0), 255);
        assert_eq!(img.get_pixel(0, 0, 1), 255);
        assert_eq!(img.get_pixel(0, 0, 2), 255);
    }

    #[test]
    fn test_distance_rms_identical() {
        let data: Vec<u8> = vec![100, 150, 200, 50, 75, 125];
        let rms = distance_rms(&data, &data);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_distance_rms_different() {
        let orig: Vec<u8> = vec![0, 0, 0, 0];
        let decoded: Vec<u8> = vec![10, 10, 10, 10];
        let rms = distance_rms(&orig, &decoded);
        // RMS should be 10 (each pixel differs by 10)
        assert!((rms - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_max_pixel_diff() {
        let orig: Vec<u8> = vec![100, 100, 100, 100];
        let decoded: Vec<u8> = vec![100, 105, 95, 120];
        let max_diff = max_pixel_diff(&orig, &decoded);
        assert_eq!(max_diff, 20); // 120 - 100 = 20
    }

    #[test]
    fn test_verify_output_pass() {
        let orig: Vec<u8> = vec![100; 64];
        let decoded: Vec<u8> = vec![105; 64];
        // Should not panic
        verify_output(&orig, &decoded, 10.0, 10);
    }

    #[test]
    #[should_panic(expected = "RMS distance")]
    fn test_verify_output_fail_rms() {
        let orig: Vec<u8> = vec![0; 64];
        let decoded: Vec<u8> = vec![100; 64];
        verify_output(&orig, &decoded, 10.0, 255);
    }

    #[test]
    #[should_panic(expected = "Max pixel diff")]
    fn test_verify_output_fail_diff() {
        let orig: Vec<u8> = vec![100; 64];
        let decoded: Vec<u8> = vec![105; 64];
        verify_output(&orig, &decoded, 100.0, 3);
    }

    #[test]
    fn test_to_grayscale() {
        let mut rgb = TestImage::new(2, 2, 3);
        // White pixel
        rgb.set_pixel(0, 0, 0, 255);
        rgb.set_pixel(0, 0, 1, 255);
        rgb.set_pixel(0, 0, 2, 255);
        // Black pixel
        rgb.set_pixel(1, 0, 0, 0);
        rgb.set_pixel(1, 0, 1, 0);
        rgb.set_pixel(1, 0, 2, 0);

        let gray = rgb.to_grayscale();
        assert_eq!(gray.components, 1);
        assert_eq!(gray.get_pixel(0, 0, 0), 255);
        assert_eq!(gray.get_pixel(1, 0, 0), 0);
    }
}
