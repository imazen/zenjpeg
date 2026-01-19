//! Benchmarking and comparison utilities for jpegli-rs.
//!
//! This crate provides shared infrastructure for examples and benchmarks:
//! - Synthetic test image generation
//! - Quality metrics (DSSIM, SSIMULACRA2)
//! - Test image discovery and loading
//! - Encoder comparison helpers
//!
//! All functions use `imgref` and `rgb` types for type-safe, zero-copy interfaces.

// Allow patterns that newer Rust versions handle differently
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unnecessary_lazy_evaluations)]
//!
//! # Usage
//!
//! Add as a dev-dependency in your Cargo.toml:
//! ```toml
//! [dev-dependencies]
//! jpegli-bench-utils = { path = "../jpegli-bench-utils" }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use jpegli_bench_utils::{SyntheticPattern, TestSize, QualityMetrics};
//!
//! let original = SyntheticPattern::PhotoLike.generate(512, 512);
//! // ... encode and decode ...
//! let dssim = QualityMetrics::dssim(original.as_ref(), decoded.as_ref());
//! ```

use imgref::{ImgRef, ImgVec};
use rgb::{RGB8, RGBA8};
use std::path::PathBuf;

// ============================================================================
// Type Aliases for Consistency
// ============================================================================

/// RGB image buffer (owned)
pub type RgbImage = ImgVec<RGB8>;
/// RGB image reference (borrowed)
pub type RgbImageRef<'a> = ImgRef<'a, RGB8>;
/// RGBA image buffer (owned)
pub type RgbaImage = ImgVec<RGBA8>;

// ============================================================================
// Synthetic Test Images
// ============================================================================

/// Synthetic test image patterns.
///
/// These patterns stress different aspects of JPEG encoding:
/// - Gradients: test quantization precision
/// - Checkerboard: test DCT handling of sharp edges
/// - Noise: test entropy coding
/// - ColorBars: test chroma handling
/// - PhotoLike: simulate organic content with smooth variations
/// - HighFrequency: stress DCT (worst case for JPEG)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntheticPattern {
    /// Smooth horizontal gradient (R: left→right, G: constant, B: constant)
    GradientH,
    /// Smooth vertical gradient (R: constant, G: top→bottom, B: constant)
    GradientV,
    /// Diagonal gradient (R: x, G: y, B: (x+y)/2)
    GradientDiagonal,
    /// RGB gradient (R: x, G: y, B: 128)
    GradientRgb,
    /// Black/white checkerboard with specified block size
    Checkerboard { block_size: u32 },
    /// Deterministic pseudo-random noise (LCG with seed)
    Noise { seed: u64 },
    /// Solid RGB color fill
    SolidColor { r: u8, g: u8, b: u8 },
    /// TV-style SMPTE color bars (8 vertical bars)
    ColorBars,
    /// Simulated photo content (smooth sinusoidal variations)
    PhotoLike,
    /// Alternating pixels (worst case for DCT)
    HighFrequency,
    /// Complex mixed-frequency pattern
    Complex,
}

impl SyntheticPattern {
    /// Generate a synthetic image with this pattern.
    #[must_use]
    pub fn generate(self, width: u32, height: u32) -> RgbImage {
        match self {
            Self::GradientH => generate_gradient_h(width, height),
            Self::GradientV => generate_gradient_v(width, height),
            Self::GradientDiagonal => generate_gradient_diagonal(width, height),
            Self::GradientRgb => generate_gradient_rgb(width, height),
            Self::Checkerboard { block_size } => generate_checkerboard(width, height, block_size),
            Self::Noise { seed } => generate_noise(width, height, seed),
            Self::SolidColor { r, g, b } => generate_solid(width, height, RGB8::new(r, g, b)),
            Self::ColorBars => generate_color_bars(width, height),
            Self::PhotoLike => generate_photo_like(width, height),
            Self::HighFrequency => generate_high_frequency(width, height),
            Self::Complex => generate_complex(width, height),
        }
    }
}

/// Standard test image sizes for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestSize {
    /// 64x64 - tiny, good for quick tests
    Tiny,
    /// 256x256 - small, fits in L2 cache
    Small,
    /// 512x512 - medium, typical thumbnail
    Medium,
    /// 1024x768 - XGA, common web size
    Web,
    /// 1920x1080 - Full HD
    Hd,
    /// 3840x2160 - 4K UHD
    Uhd4k,
    /// Custom dimensions
    Custom(u32, u32),
}

impl TestSize {
    /// Get dimensions as (width, height).
    #[must_use]
    pub const fn dimensions(self) -> (u32, u32) {
        match self {
            Self::Tiny => (64, 64),
            Self::Small => (256, 256),
            Self::Medium => (512, 512),
            Self::Web => (1024, 768),
            Self::Hd => (1920, 1080),
            Self::Uhd4k => (3840, 2160),
            Self::Custom(w, h) => (w, h),
        }
    }

    /// Get total pixel count.
    #[must_use]
    pub const fn pixels(self) -> u64 {
        let (w, h) = self.dimensions();
        w as u64 * h as u64
    }

    /// Get megapixels.
    #[must_use]
    pub fn megapixels(self) -> f64 {
        self.pixels() as f64 / 1_000_000.0
    }
}

// ============================================================================
// Synthetic Image Generators
// ============================================================================

/// Generate a horizontal gradient (left=black, right=white in R channel).
#[must_use]
pub fn generate_gradient_h(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let w_max = (width.saturating_sub(1)).max(1) as f32;

    for _y in 0..height {
        for x in 0..width {
            let v = ((x as f32 / w_max) * 255.0) as u8;
            pixels.push(RGB8::new(v, v, v));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate a vertical gradient (top=black, bottom=white in G channel).
#[must_use]
pub fn generate_gradient_v(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let h_max = (height.saturating_sub(1)).max(1) as f32;

    for y in 0..height {
        let v = ((y as f32 / h_max) * 255.0) as u8;
        for _x in 0..width {
            pixels.push(RGB8::new(v, v, v));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate a diagonal gradient.
#[must_use]
pub fn generate_gradient_diagonal(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let max = ((width + height).saturating_sub(2)).max(1) as f32;

    for y in 0..height {
        for x in 0..width {
            let v = (((x + y) as f32 / max) * 255.0) as u8;
            pixels.push(RGB8::new(v, v, v));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate an RGB gradient (R=x, G=y, B=128).
#[must_use]
pub fn generate_gradient_rgb(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let w_max = (width.saturating_sub(1)).max(1) as f32;
    let h_max = (height.saturating_sub(1)).max(1) as f32;

    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / w_max) * 255.0) as u8;
            let g = ((y as f32 / h_max) * 255.0) as u8;
            pixels.push(RGB8::new(r, g, 128));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate a checkerboard pattern.
#[must_use]
pub fn generate_checkerboard(width: u32, height: u32, block_size: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let bs = block_size.max(1);

    for y in 0..height {
        for x in 0..width {
            let bx = x / bs;
            let by = y / bs;
            let v = if (bx + by) % 2 == 0 { 255 } else { 0 };
            pixels.push(RGB8::new(v, v, v));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate deterministic pseudo-random noise using LCG.
#[must_use]
pub fn generate_noise(width: u32, height: u32, seed: u64) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let mut state = seed;

    // LCG constants (same as glibc)
    const A: u64 = 1103515245;
    const C: u64 = 12345;
    const M: u64 = 1 << 31;

    for _ in 0..height {
        for _ in 0..width {
            let r = {
                state = (A.wrapping_mul(state).wrapping_add(C)) % M;
                ((state >> 16) & 0xFF) as u8
            };
            let g = {
                state = (A.wrapping_mul(state).wrapping_add(C)) % M;
                ((state >> 16) & 0xFF) as u8
            };
            let b = {
                state = (A.wrapping_mul(state).wrapping_add(C)) % M;
                ((state >> 16) & 0xFF) as u8
            };
            pixels.push(RGB8::new(r, g, b));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate a solid color fill.
#[must_use]
pub fn generate_solid(width: u32, height: u32, color: RGB8) -> RgbImage {
    let pixels = vec![color; (width * height) as usize];
    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate SMPTE-style color bars.
#[must_use]
pub fn generate_color_bars(width: u32, height: u32) -> RgbImage {
    const COLORS: [RGB8; 8] = [
        RGB8::new(255, 255, 255), // White
        RGB8::new(255, 255, 0),   // Yellow
        RGB8::new(0, 255, 255),   // Cyan
        RGB8::new(0, 255, 0),     // Green
        RGB8::new(255, 0, 255),   // Magenta
        RGB8::new(255, 0, 0),     // Red
        RGB8::new(0, 0, 255),     // Blue
        RGB8::new(0, 0, 0),       // Black
    ];

    let mut pixels = Vec::with_capacity((width * height) as usize);
    let bar_width = width / 8;

    for _y in 0..height {
        for x in 0..width {
            let bar_idx = ((x / bar_width.max(1)) as usize).min(7);
            pixels.push(COLORS[bar_idx]);
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate photo-like content with smooth sinusoidal variations.
#[must_use]
pub fn generate_photo_like(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let w = width as f32;
    let h = height as f32;

    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / w;
            let fy = y as f32 / h;

            let r = ((fx * std::f32::consts::TAU).sin() * 40.0 + 120.0) as u8;
            let g = ((fy * std::f32::consts::TAU).cos() * 50.0 + 100.0) as u8;
            let b = (((fx + fy) * std::f32::consts::TAU).sin() * 30.0 + 90.0) as u8;

            pixels.push(RGB8::new(r, g, b));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate high-frequency alternating pattern (worst case for JPEG).
#[must_use]
pub fn generate_high_frequency(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let v = if (x + y) % 2 == 0 { 255 } else { 0 };
            pixels.push(RGB8::new(v, v, v));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

/// Generate complex mixed-frequency pattern.
#[must_use]
pub fn generate_complex(width: u32, height: u32) -> RgbImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let w = width as f64;
    let h = height as f64;

    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / w;
            let fy = y as f64 / h;

            let r = ((fx * 255.0) + (fx * fy * 50.0).sin() * 30.0).clamp(0.0, 255.0) as u8;
            let g = ((fy * 255.0) + (fx * fy * 100.0).cos() * 40.0).clamp(0.0, 255.0) as u8;
            let b = (128.0 + ((fx + fy) * 50.0).sin() * 50.0).clamp(0.0, 255.0) as u8;

            pixels.push(RGB8::new(r, g, b));
        }
    }

    ImgVec::new(pixels, width as usize, height as usize)
}

// ============================================================================
// Edge MCU Tiling (for testing partial MCU handling)
// ============================================================================

/// Configuration for which edges to replicate when creating edge test images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EdgeReplicationMode {
    /// Replicate only the rightmost columns (for width % 8 != 0)
    Right,
    /// Replicate only the bottom rows (for height % 8 != 0)
    Bottom,
    /// Replicate both right columns and bottom rows
    Both,
    /// Automatically replicate whichever edges have non-8-aligned dimensions
    #[default]
    Auto,
}

/// Configuration for edge MCU test image generation.
#[derive(Debug, Clone)]
pub struct EdgeTestConfig {
    /// Which edges to replicate
    pub mode: EdgeReplicationMode,
    /// Override width of right edge strip (None = use width % 8)
    pub right_edge_width: Option<usize>,
    /// Override height of bottom edge strip (None = use height % 8)
    pub bottom_edge_height: Option<usize>,
    /// Target output width (None = same as source)
    pub target_width: Option<usize>,
    /// Target output height (None = same as source)
    pub target_height: Option<usize>,
}

impl Default for EdgeTestConfig {
    fn default() -> Self {
        Self {
            mode: EdgeReplicationMode::Auto,
            right_edge_width: None,
            bottom_edge_height: None,
            target_width: None,
            target_height: None,
        }
    }
}

impl EdgeTestConfig {
    /// Create config for right-edge-only testing with specific width.
    #[must_use]
    pub fn right_only(edge_width: usize) -> Self {
        Self {
            mode: EdgeReplicationMode::Right,
            right_edge_width: Some(edge_width),
            ..Default::default()
        }
    }

    /// Create config for bottom-edge-only testing with specific height.
    #[must_use]
    pub fn bottom_only(edge_height: usize) -> Self {
        Self {
            mode: EdgeReplicationMode::Bottom,
            bottom_edge_height: Some(edge_height),
            ..Default::default()
        }
    }

    /// Create config for both edges with specific dimensions.
    #[must_use]
    pub fn both(edge_width: usize, edge_height: usize) -> Self {
        Self {
            mode: EdgeReplicationMode::Both,
            right_edge_width: Some(edge_width),
            bottom_edge_height: Some(edge_height),
            ..Default::default()
        }
    }

    /// Set target output dimensions.
    #[must_use]
    pub fn with_target_size(mut self, width: usize, height: usize) -> Self {
        self.target_width = Some(width);
        self.target_height = Some(height);
        self
    }
}

/// Create an edge test image by tiling edge strips from the source.
///
/// This is useful for testing edge-case handling in JPEG encoders. By tiling
/// just the partial MCU edge strips across the full image, we amplify any bugs
/// in boundary handling - instead of affecting <1% of blocks, every block in
/// the tiled image exercises the edge case.
///
/// # Example
/// ```rust,ignore
/// use jpegli_bench_utils::{create_edge_test_image, EdgeTestConfig};
///
/// let source = load_png("frymire.png").unwrap(); // 1118x1105
///
/// // Auto mode: tiles both right (6 cols) and bottom (1 row)
/// let tiled = create_edge_test_image(&source, EdgeTestConfig::default());
///
/// // Right-only with width=1 (most extreme edge case)
/// let tiled = create_edge_test_image(&source, EdgeTestConfig::right_only(1));
///
/// // Both edges with custom dimensions
/// let config = EdgeTestConfig::both(3, 2).with_target_size(512, 512);
/// let tiled = create_edge_test_image(&source, config);
/// ```
#[must_use]
pub fn create_edge_test_image(source: &RgbImage, config: EdgeTestConfig) -> Option<RgbImage> {
    let src_width = source.width();
    let src_height = source.height();

    // Determine which edges to process
    let (do_right, do_bottom) = match config.mode {
        EdgeReplicationMode::Right => (true, false),
        EdgeReplicationMode::Bottom => (false, true),
        EdgeReplicationMode::Both => (true, true),
        EdgeReplicationMode::Auto => (src_width % 8 != 0, src_height % 8 != 0),
    };

    // Determine edge strip dimensions
    let right_width = config
        .right_edge_width
        .unwrap_or_else(|| src_width % 8)
        .max(1)
        .min(src_width);
    let bottom_height = config
        .bottom_edge_height
        .unwrap_or_else(|| src_height % 8)
        .max(1)
        .min(src_height);

    // If nothing to do, return None
    if !do_right && !do_bottom {
        return None;
    }

    // Determine output dimensions
    let out_width = config.target_width.unwrap_or(src_width);
    let out_height = config.target_height.unwrap_or(src_height);

    // Source strip positions
    let right_start = src_width.saturating_sub(right_width);
    let bottom_start = src_height.saturating_sub(bottom_height);

    let mut pixels = Vec::with_capacity(out_width * out_height);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            // Determine source coordinates based on replication mode
            let src_x = if do_right {
                right_start + (out_x % right_width)
            } else {
                out_x % src_width
            };

            let src_y = if do_bottom {
                bottom_start + (out_y % bottom_height)
            } else {
                out_y % src_height
            };

            let src_pixel = source.buf()[src_y * src_width + src_x];
            pixels.push(src_pixel);
        }
    }

    Some(ImgVec::new(pixels, out_width, out_height))
}

/// Extract the rightmost N columns from an image and tile them horizontally.
///
/// Convenience wrapper around `create_edge_test_image` for right-edge-only testing.
#[must_use]
pub fn tile_edge_columns(source: &RgbImage, edge_columns: usize, target_width: usize) -> RgbImage {
    let config =
        EdgeTestConfig::right_only(edge_columns).with_target_size(target_width, source.height());
    create_edge_test_image(source, config).unwrap_or_else(|| source.clone())
}

/// Extract the bottom N rows from an image and tile them vertically.
///
/// Convenience wrapper around `create_edge_test_image` for bottom-edge-only testing.
#[must_use]
pub fn tile_edge_rows(source: &RgbImage, edge_rows: usize, target_height: usize) -> RgbImage {
    let config =
        EdgeTestConfig::bottom_only(edge_rows).with_target_size(source.width(), target_height);
    create_edge_test_image(source, config).unwrap_or_else(|| source.clone())
}

/// Information about MCU edge characteristics of an image.
#[derive(Debug, Clone)]
pub struct McuEdgeInfo {
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Number of full MCU columns (width / 8)
    pub full_mcu_columns: usize,
    /// Number of full MCU rows (height / 8)
    pub full_mcu_rows: usize,
    /// Width of rightmost partial MCU (width % 8, 0 if none)
    pub partial_mcu_width: usize,
    /// Height of bottom partial MCU (height % 8, 0 if none)
    pub partial_mcu_height: usize,
    /// Percentage of blocks affected by partial width MCU
    pub width_affected_pct: f64,
    /// Percentage of blocks affected by partial height MCU
    pub height_affected_pct: f64,
    /// Percentage of blocks affected by either edge (union)
    pub total_affected_pct: f64,
}

impl McuEdgeInfo {
    /// Analyze an image for MCU edge characteristics.
    #[must_use]
    pub fn analyze(width: usize, height: usize) -> Self {
        let full_mcu_columns = width / 8;
        let full_mcu_rows = height / 8;
        let partial_mcu_width = width % 8;
        let partial_mcu_height = height % 8;

        let total_mcu_columns = full_mcu_columns + if partial_mcu_width > 0 { 1 } else { 0 };
        let total_mcu_rows = full_mcu_rows + if partial_mcu_height > 0 { 1 } else { 0 };
        let total_blocks = total_mcu_columns * total_mcu_rows;

        // Blocks affected by partial width (rightmost column)
        let width_affected = if partial_mcu_width > 0 {
            total_mcu_rows
        } else {
            0
        };
        // Blocks affected by partial height (bottom row)
        let height_affected = if partial_mcu_height > 0 {
            total_mcu_columns
        } else {
            0
        };
        // Corner block is counted in both, subtract 1 to avoid double-counting
        let corner_overlap = if partial_mcu_width > 0 && partial_mcu_height > 0 {
            1
        } else {
            0
        };
        let total_affected = width_affected + height_affected - corner_overlap;

        let width_affected_pct = if total_blocks > 0 {
            width_affected as f64 / total_blocks as f64 * 100.0
        } else {
            0.0
        };
        let height_affected_pct = if total_blocks > 0 {
            height_affected as f64 / total_blocks as f64 * 100.0
        } else {
            0.0
        };
        let total_affected_pct = if total_blocks > 0 {
            total_affected as f64 / total_blocks as f64 * 100.0
        } else {
            0.0
        };

        Self {
            width,
            height,
            full_mcu_columns,
            full_mcu_rows,
            partial_mcu_width,
            partial_mcu_height,
            width_affected_pct,
            height_affected_pct,
            total_affected_pct,
        }
    }

    /// Check if this image has a partial rightmost MCU.
    #[must_use]
    pub fn has_partial_width(&self) -> bool {
        self.partial_mcu_width > 0
    }

    /// Check if this image has a partial bottom MCU.
    #[must_use]
    pub fn has_partial_height(&self) -> bool {
        self.partial_mcu_height > 0
    }

    /// Check if this image has any partial MCU edges.
    #[must_use]
    pub fn has_partial_edges(&self) -> bool {
        self.has_partial_width() || self.has_partial_height()
    }
}

// ============================================================================
// Convenience: Generate Standard Test Suite
// ============================================================================

/// Standard test image suite for comprehensive benchmarking.
pub struct TestSuite {
    /// Image name
    pub name: &'static str,
    /// Image data
    pub image: RgbImage,
}

impl TestSuite {
    /// Generate a standard suite of synthetic test images at the given size.
    #[must_use]
    pub fn standard(size: TestSize) -> Vec<Self> {
        let (w, h) = size.dimensions();
        vec![
            Self {
                name: "gradient_rgb",
                image: SyntheticPattern::GradientRgb.generate(w, h),
            },
            Self {
                name: "checkerboard_8",
                image: SyntheticPattern::Checkerboard { block_size: 8 }.generate(w, h),
            },
            Self {
                name: "checkerboard_16",
                image: SyntheticPattern::Checkerboard { block_size: 16 }.generate(w, h),
            },
            Self {
                name: "noise",
                image: SyntheticPattern::Noise { seed: 12345 }.generate(w, h),
            },
            Self {
                name: "color_bars",
                image: SyntheticPattern::ColorBars.generate(w, h),
            },
            Self {
                name: "photo_like",
                image: SyntheticPattern::PhotoLike.generate(w, h),
            },
            Self {
                name: "high_frequency",
                image: SyntheticPattern::HighFrequency.generate(w, h),
            },
            Self {
                name: "complex",
                image: SyntheticPattern::Complex.generate(w, h),
            },
        ]
    }
}

// ============================================================================
// Test Image Discovery
// ============================================================================

/// Known test image locations.
pub struct TestImages;

impl TestImages {
    /// Path to frymire.png (1118x1105, good for testing odd dimensions).
    ///
    /// Checks these locations in order:
    /// 1. `FRYMIRE_PATH` environment variable
    /// 2. `/home/lilith/work/codec-corpus/imageflow/test_inputs/frymire.png`
    /// 3. `../codec-corpus/imageflow/test_inputs/frymire.png` (relative to manifest)
    #[must_use]
    pub fn frymire_path() -> Option<PathBuf> {
        if let Ok(path) = std::env::var("FRYMIRE_PATH") {
            let p = PathBuf::from(path);
            if p.exists() {
                return Some(p);
            }
        }

        let candidates = [
            PathBuf::from("/home/lilith/work/codec-corpus/imageflow/test_inputs/frymire.png"),
            // Relative to cargo manifest
            std::env::var("CARGO_MANIFEST_DIR")
                .ok()
                .map(|m| PathBuf::from(m).join("../codec-corpus/imageflow/test_inputs/frymire.png"))
                .unwrap_or_default(),
        ];

        candidates.into_iter().find(|p| p.exists())
    }

    /// Load frymire.png if available.
    #[must_use]
    pub fn load_frymire() -> Option<RgbImage> {
        Self::frymire_path().and_then(|p| load_png(&p).ok())
    }

    /// Path to flower_small.rgb.png (jpegli standard test image).
    #[must_use]
    pub fn flower_small_path() -> Option<PathBuf> {
        let testdata = jpegli::test_utils::get_testdata_dir();
        let path = testdata.join("jxl/flower/flower_small.rgb.png");
        path.exists().then_some(path)
    }

    /// Load flower_small.rgb.png if available.
    #[must_use]
    pub fn load_flower_small() -> Option<RgbImage> {
        Self::flower_small_path().and_then(|p| load_png(&p).ok())
    }

    /// Get a test image by name, loading from disk or generating synthetically.
    ///
    /// Supported names:
    /// - `"frymire"` - frymire.png (or fallback to photo_like synthetic)
    /// - `"flower_small"` - flower_small.rgb.png (or fallback)
    /// - `"gradient"`, `"checkerboard"`, `"noise"`, `"photo_like"`, etc.
    #[must_use]
    pub fn get(name: &str, fallback_size: TestSize) -> RgbImage {
        let (w, h) = fallback_size.dimensions();

        match name {
            "frymire" => Self::load_frymire()
                .unwrap_or_else(|| SyntheticPattern::PhotoLike.generate(1118, 1105)),
            "flower_small" => Self::load_flower_small()
                .unwrap_or_else(|| SyntheticPattern::PhotoLike.generate(423, 633)),
            "gradient" | "gradient_rgb" => SyntheticPattern::GradientRgb.generate(w, h),
            "gradient_h" => SyntheticPattern::GradientH.generate(w, h),
            "gradient_v" => SyntheticPattern::GradientV.generate(w, h),
            "checkerboard" => SyntheticPattern::Checkerboard { block_size: 8 }.generate(w, h),
            "noise" => SyntheticPattern::Noise { seed: 12345 }.generate(w, h),
            "color_bars" => SyntheticPattern::ColorBars.generate(w, h),
            "photo_like" => SyntheticPattern::PhotoLike.generate(w, h),
            "high_frequency" => SyntheticPattern::HighFrequency.generate(w, h),
            "complex" => SyntheticPattern::Complex.generate(w, h),
            _ => SyntheticPattern::GradientRgb.generate(w, h),
        }
    }
}

// ============================================================================
// Image Loading
// ============================================================================

/// Load a PNG file into an RGB image buffer.
///
/// Handles RGB, RGBA (alpha discarded), grayscale, and grayscale+alpha.
pub fn load_png(path: &std::path::Path) -> Result<RgbImage, PngLoadError> {
    let file = std::fs::File::open(path).map_err(|e| PngLoadError::Io(e.to_string()))?;

    let decoder = png::Decoder::new(file);
    let mut reader = decoder
        .read_info()
        .map_err(|e| PngLoadError::Decode(e.to_string()))?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| PngLoadError::Decode(e.to_string()))?;

    let width = info.width as usize;
    let height = info.height as usize;
    let data = &buf[..info.buffer_size()];

    let pixels: Vec<RGB8> = match info.color_type {
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect(),
        png::ColorType::Rgba => data
            .chunks_exact(4)
            .map(|c| RGB8::new(c[0], c[1], c[2]))
            .collect(),
        png::ColorType::Grayscale => data.iter().map(|&g| RGB8::new(g, g, g)).collect(),
        png::ColorType::GrayscaleAlpha => data
            .chunks_exact(2)
            .map(|c| RGB8::new(c[0], c[0], c[0]))
            .collect(),
        other => return Err(PngLoadError::UnsupportedColorType(format!("{:?}", other))),
    };

    Ok(ImgVec::new(pixels, width, height))
}

/// Error loading a PNG file.
#[derive(Debug, Clone)]
pub enum PngLoadError {
    /// I/O error
    Io(String),
    /// PNG decode error
    Decode(String),
    /// Unsupported color type
    UnsupportedColorType(String),
}

impl std::fmt::Display for PngLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Decode(e) => write!(f, "PNG decode error: {}", e),
            Self::UnsupportedColorType(t) => write!(f, "Unsupported color type: {}", t),
        }
    }
}

impl std::error::Error for PngLoadError {}

/// Write a PPM file from an RGB image (for C++ tool interop).
pub fn write_ppm(path: &std::path::Path, img: RgbImageRef<'_>) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "P6")?;
    writeln!(file, "{} {}", img.width(), img.height())?;
    writeln!(file, "255")?;

    // Write row by row to handle stride
    for row in img.rows() {
        for pixel in row {
            file.write_all(&[pixel.r, pixel.g, pixel.b])?;
        }
    }
    Ok(())
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Quality metrics for comparing images.
pub struct QualityMetrics;

impl QualityMetrics {
    /// Compute DSSIM (structural dissimilarity) between two images.
    ///
    /// Returns a value where 0.0 = identical, higher = more different.
    /// Typical values: <0.001 excellent, <0.01 good, <0.1 acceptable.
    pub fn dssim(original: RgbImageRef<'_>, distorted: RgbImageRef<'_>) -> f64 {
        assert_eq!(original.width(), distorted.width(), "Width mismatch");
        assert_eq!(original.height(), distorted.height(), "Height mismatch");

        let attr = dssim::Dssim::new();

        let orig_rgba: Vec<RGBA8> = original
            .pixels()
            .map(|p| RGBA8::new(p.r, p.g, p.b, 255))
            .collect();
        let dist_rgba: Vec<RGBA8> = distorted
            .pixels()
            .map(|p| RGBA8::new(p.r, p.g, p.b, 255))
            .collect();

        let orig_img = attr
            .create_image_rgba(&orig_rgba, original.width(), original.height())
            .expect("create dssim image");
        let dist_img = attr
            .create_image_rgba(&dist_rgba, distorted.width(), distorted.height())
            .expect("create dssim image");

        let (dssim, _) = attr.compare(&orig_img, dist_img);
        dssim.into()
    }

    /// Compute SSIMULACRA2 score between two images.
    ///
    /// Returns a value where 100 = identical, lower = worse.
    /// Typical values: >90 excellent, >80 good, >70 acceptable.
    pub fn ssimulacra2(original: RgbImageRef<'_>, distorted: RgbImageRef<'_>) -> f64 {
        use fast_ssim2::compute_ssimulacra2;

        assert_eq!(original.width(), distorted.width(), "Width mismatch");
        assert_eq!(original.height(), distorted.height(), "Height mismatch");

        // Convert to flat bytes for fast-ssim2
        let orig_bytes = rgb_to_bytes(original);
        let dist_bytes = rgb_to_bytes(distorted);

        let orig_img = ImgVec::new(orig_bytes, original.width() * 3, original.height());
        let dist_img = ImgVec::new(dist_bytes, distorted.width() * 3, distorted.height());

        compute_ssimulacra2(orig_img.as_ref(), dist_img.as_ref()).unwrap_or(0.0)
    }

    /// Compute max pixel difference between two images.
    ///
    /// Returns the maximum absolute difference across all channels.
    pub fn max_pixel_diff(original: RgbImageRef<'_>, distorted: RgbImageRef<'_>) -> u8 {
        assert_eq!(original.width(), distorted.width(), "Width mismatch");
        assert_eq!(original.height(), distorted.height(), "Height mismatch");

        original
            .pixels()
            .zip(distorted.pixels())
            .map(|(o, d)| {
                let dr = (o.r as i16 - d.r as i16).unsigned_abs() as u8;
                let dg = (o.g as i16 - d.g as i16).unsigned_abs() as u8;
                let db = (o.b as i16 - d.b as i16).unsigned_abs() as u8;
                dr.max(dg).max(db)
            })
            .max()
            .unwrap_or(0)
    }

    /// Compute RMS (root mean square) error between two images.
    ///
    /// Returns error in the range 0-255.
    pub fn rms(original: RgbImageRef<'_>, distorted: RgbImageRef<'_>) -> f64 {
        assert_eq!(original.width(), distorted.width(), "Width mismatch");
        assert_eq!(original.height(), distorted.height(), "Height mismatch");

        let sum_sq: f64 = original
            .pixels()
            .zip(distorted.pixels())
            .map(|(o, d)| {
                let dr = (o.r as f64 - d.r as f64).powi(2);
                let dg = (o.g as f64 - d.g as f64).powi(2);
                let db = (o.b as f64 - d.b as f64).powi(2);
                dr + dg + db
            })
            .sum();

        let n = (original.width() * original.height() * 3) as f64;
        (sum_sq / n).sqrt()
    }

    /// Compute average pixel difference.
    pub fn avg_diff(original: RgbImageRef<'_>, distorted: RgbImageRef<'_>) -> f64 {
        assert_eq!(original.width(), distorted.width(), "Width mismatch");
        assert_eq!(original.height(), distorted.height(), "Height mismatch");

        let sum: u64 = original
            .pixels()
            .zip(distorted.pixels())
            .map(|(o, d)| {
                let dr = (o.r as i16 - d.r as i16).unsigned_abs() as u64;
                let dg = (o.g as i16 - d.g as i16).unsigned_abs() as u64;
                let db = (o.b as i16 - d.b as i16).unsigned_abs() as u64;
                dr + dg + db
            })
            .sum();

        let n = (original.width() * original.height() * 3) as f64;
        sum as f64 / n
    }

    /// Compute Butteraugli perceptual distance between two images.
    ///
    /// Returns a distance value where 0.0 = identical, higher = more different.
    /// Typical thresholds:
    /// - < 0.5: Nearly imperceptible
    /// - < 1.0: Good quality
    /// - < 1.5: Acceptable
    /// - > 2.0: Noticeable artifacts
    pub fn butteraugli(original: RgbImageRef<'_>, distorted: RgbImageRef<'_>) -> f64 {
        assert_eq!(original.width(), distorted.width(), "Width mismatch");
        assert_eq!(original.height(), distorted.height(), "Height mismatch");

        let width = original.width();
        let height = original.height();

        // Convert to flat sRGB bytes for butteraugli (it does sRGB->linear internally)
        let orig_bytes = rgb_to_bytes(original);
        let dist_bytes = rgb_to_bytes(distorted);

        let params = butteraugli::ButteraugliParams::default();
        butteraugli::compute_butteraugli(&orig_bytes, &dist_bytes, width, height, &params)
            .map(|r| r.score)
            .unwrap_or(f64::MAX)
    }
}

// ============================================================================
// Image Conversion Helpers
// ============================================================================

/// Convert an `RgbImage` to a flat `Vec<u8>` for encoding.
#[must_use]
pub fn rgb_to_bytes(img: RgbImageRef<'_>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(img.width() * img.height() * 3);
    for row in img.rows() {
        for pixel in row {
            bytes.push(pixel.r);
            bytes.push(pixel.g);
            bytes.push(pixel.b);
        }
    }
    bytes
}

/// Convert a flat `&[u8]` (RGB) to an `RgbImage`.
#[must_use]
pub fn bytes_to_rgb(data: &[u8], width: usize, height: usize) -> RgbImage {
    assert_eq!(
        data.len(),
        width * height * 3,
        "Data size mismatch for {}x{}",
        width,
        height
    );

    let pixels: Vec<RGB8> = data
        .chunks_exact(3)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();

    ImgVec::new(pixels, width, height)
}

// ============================================================================
// Comparison Result Types
// ============================================================================

/// Result of encoding an image with a specific encoder.
#[derive(Debug, Clone)]
pub struct EncodeResult {
    /// Encoder name
    pub encoder: String,
    /// Encoded JPEG size in bytes
    pub size: usize,
    /// Bits per pixel
    pub bpp: f64,
    /// Encoding time in milliseconds (if measured)
    pub encode_ms: Option<f64>,
}

impl EncodeResult {
    /// Create from encoder name and JPEG data.
    #[must_use]
    pub fn new(encoder: impl Into<String>, jpeg_data: &[u8], width: usize, height: usize) -> Self {
        let pixels = width * height;
        Self {
            encoder: encoder.into(),
            size: jpeg_data.len(),
            bpp: jpeg_data.len() as f64 * 8.0 / pixels as f64,
            encode_ms: None,
        }
    }

    /// Add timing information.
    #[must_use]
    pub fn with_timing(mut self, encode_ms: f64) -> Self {
        self.encode_ms = Some(encode_ms);
        self
    }
}

/// Full quality comparison result.
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// Encoding result
    pub encode: EncodeResult,
    /// DSSIM score (lower is better, 0 = identical)
    pub dssim: f64,
    /// SSIMULACRA2 score (higher is better, 100 = identical)
    pub ssimulacra2: f64,
    /// Butteraugli distance (lower is better, 0 = identical)
    pub butteraugli: Option<f64>,
    /// Max pixel difference (0-255)
    pub max_diff: u8,
    /// RMS error (0-255)
    pub rms: f64,
}

// ============================================================================
// JPEG Decoding (using zune-jpeg)
// ============================================================================

/// Decode a JPEG to RGB bytes using zune-jpeg.
///
/// Returns (pixels, width, height) or error.
pub fn decode_jpeg(data: &[u8]) -> Result<(Vec<u8>, usize, usize), JpegDecodeError> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);

    let pixels = decoder
        .decode()
        .map_err(|e| JpegDecodeError::Decode(format!("{:?}", e)))?;

    let info = decoder.info().ok_or(JpegDecodeError::NoInfo)?;

    Ok((pixels, info.width as usize, info.height as usize))
}

/// Decode a JPEG to an RgbImage using zune-jpeg.
pub fn decode_jpeg_to_rgb(data: &[u8]) -> Result<RgbImage, JpegDecodeError> {
    let (pixels, width, height) = decode_jpeg(data)?;
    Ok(bytes_to_rgb(&pixels, width, height))
}

/// Error decoding a JPEG file.
#[derive(Debug, Clone)]
pub enum JpegDecodeError {
    /// Decode error
    Decode(String),
    /// No image info available
    NoInfo,
}

impl std::fmt::Display for JpegDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "JPEG decode error: {}", e),
            Self::NoInfo => write!(f, "No image info available after decode"),
        }
    }
}

impl std::error::Error for JpegDecodeError {}

// ============================================================================
// Image Data Container
// ============================================================================

/// A loaded test image with metadata.
///
/// Common container for images loaded from disk for benchmarking.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Image name (usually filename without path)
    pub name: String,
    /// RGB pixel data
    pub pixels: Vec<u8>,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
}

impl ImageData {
    /// Create from loaded PNG data.
    pub fn from_png(path: &std::path::Path) -> Option<Self> {
        let (pixels, width, height) = load_png(path)
            .ok()
            .map(|img| (rgb_to_bytes(img.as_ref()), img.width(), img.height()))?;

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        Some(Self {
            name,
            pixels,
            width,
            height,
        })
    }

    /// Create from an RgbImage.
    pub fn from_rgb_image(name: impl Into<String>, img: &RgbImage) -> Self {
        Self {
            name: name.into(),
            pixels: rgb_to_bytes(img.as_ref()),
            width: img.width(),
            height: img.height(),
        }
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Convert to RgbImage reference.
    #[must_use]
    pub fn as_rgb_image(&self) -> RgbImage {
        bytes_to_rgb(&self.pixels, self.width, self.height)
    }
}

/// Load all PNG images from a directory.
pub fn load_corpus(dir: &std::path::Path, max_files: Option<usize>) -> Vec<ImageData> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();

    files.sort();

    if let Some(max) = max_files {
        files.truncate(max);
    }

    files
        .iter()
        .filter_map(|p| ImageData::from_png(p))
        .collect()
}

// ============================================================================
// Encoder Configuration (Clear Naming)
// ============================================================================

/// Encoder implementation identifier.
///
/// Distinguishes between Rust and C++ implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncoderImpl {
    /// jpegli-rs: Pure Rust port of Google's jpegli
    JpegliRs,
    /// cjpegli: Original C++ jpegli via FFI (requires cjpegli-ffi feature)
    CJpegli,
}

impl EncoderImpl {
    /// Short name for display/filenames.
    #[must_use]
    pub const fn short_name(&self) -> &'static str {
        match self {
            Self::JpegliRs => "jpegli-rs",
            Self::CJpegli => "cjpegli",
        }
    }

    /// Whether this is a Rust implementation.
    #[must_use]
    pub const fn is_rust(&self) -> bool {
        matches!(self, Self::JpegliRs)
    }
}

impl std::fmt::Display for EncoderImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

/// Color encoding mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorMode {
    /// Standard YCbCr color space (default for JPEG)
    #[default]
    YCbCr,
    /// XYB perceptual color space (jpegli only)
    Xyb,
}

impl ColorMode {
    /// Short suffix for naming.
    #[must_use]
    pub const fn suffix(&self) -> &'static str {
        match self {
            Self::YCbCr => "ycbcr",
            Self::Xyb => "xyb",
        }
    }
}

impl std::fmt::Display for ColorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.suffix())
    }
}

/// JPEG scan mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScanMode {
    /// Single-scan baseline JPEG (most compatible)
    Baseline,
    /// Multi-scan progressive JPEG (default - better compression)
    #[default]
    Progressive,
}

impl ScanMode {
    /// Short suffix for naming.
    #[must_use]
    pub const fn suffix(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Progressive => "progressive",
        }
    }

    /// Convert to jpegli JpegMode.
    #[must_use]
    pub fn to_jpegli(&self) -> jpegli::decoder::JpegMode {
        match self {
            Self::Baseline => jpegli::decoder::JpegMode::Baseline,
            Self::Progressive => jpegli::decoder::JpegMode::Progressive,
        }
    }
}

impl std::fmt::Display for ScanMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.suffix())
    }
}

/// Chroma subsampling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChromaSubsampling {
    /// 4:4:4 - No subsampling (best quality, larger files)
    S444,
    /// 4:2:2 - Horizontal subsampling
    S422,
    /// 4:2:0 - Both dimensions subsampled (default - best compression)
    #[default]
    S420,
    /// 4:4:0 - Vertical subsampling only
    S440,
}

impl ChromaSubsampling {
    /// Short suffix for naming (e.g., "444", "420").
    #[must_use]
    pub const fn suffix(&self) -> &'static str {
        match self {
            Self::S444 => "444",
            Self::S422 => "422",
            Self::S420 => "420",
            Self::S440 => "440",
        }
    }

    /// Convert to jpegli Subsampling type (legacy).
    #[must_use]
    pub fn to_jpegli(&self) -> jpegli::decoder::Subsampling {
        match self {
            Self::S444 => jpegli::decoder::Subsampling::S444,
            Self::S422 => jpegli::decoder::Subsampling::S422,
            Self::S420 => jpegli::decoder::Subsampling::S420,
            Self::S440 => jpegli::decoder::Subsampling::S440,
        }
    }

    /// Convert to v2 API ChromaSubsampling type.
    #[must_use]
    pub fn to_v2(&self) -> jpegli::encoder::ChromaSubsampling {
        match self {
            Self::S444 => jpegli::encoder::ChromaSubsampling::None,
            Self::S422 => jpegli::encoder::ChromaSubsampling::HalfHorizontal,
            Self::S420 => jpegli::encoder::ChromaSubsampling::Quarter,
            Self::S440 => jpegli::encoder::ChromaSubsampling::HalfVertical,
        }
    }
}

impl std::fmt::Display for ChromaSubsampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.suffix())
    }
}

/// Complete encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Which encoder to use
    pub encoder: EncoderImpl,
    /// Color mode (YCbCr or XYB)
    pub color: ColorMode,
    /// Scan mode (baseline, progressive, sequential)
    pub scan: ScanMode,
    /// Chroma subsampling
    pub subsampling: ChromaSubsampling,
    /// Quality (0-100 for libjpeg-style, or distance for jpegli)
    pub quality: u8,
    /// Enable hybrid trellis quantization (jpegli-rs only, requires feature)
    pub hybrid: bool,
}

impl EncoderConfig {
    /// Create a new encoder configuration.
    #[must_use]
    pub fn new(encoder: EncoderImpl) -> Self {
        Self {
            encoder,
            color: ColorMode::default(),
            scan: ScanMode::default(),
            subsampling: ChromaSubsampling::default(),
            quality: 75,
            hybrid: false,
        }
    }

    /// Set color mode.
    #[must_use]
    pub fn color(mut self, color: ColorMode) -> Self {
        self.color = color;
        self
    }

    /// Set scan mode.
    #[must_use]
    pub fn scan(mut self, scan: ScanMode) -> Self {
        self.scan = scan;
        self
    }

    /// Set chroma subsampling.
    #[must_use]
    pub fn subsampling(mut self, sub: ChromaSubsampling) -> Self {
        self.subsampling = sub;
        self
    }

    /// Set quality.
    #[must_use]
    pub fn quality(mut self, q: u8) -> Self {
        self.quality = q;
        self
    }

    /// Enable hybrid trellis quantization.
    ///
    /// Requires the `experimental-hybrid-trellis` feature to be enabled.
    /// Only applies to jpegli-rs encoder.
    #[must_use]
    pub fn hybrid(mut self, enabled: bool) -> Self {
        self.hybrid = enabled;
        self
    }

    /// Generate a descriptive name for this configuration.
    ///
    /// Format: `encoder-color[-hybrid]-scan-subsampling` (e.g., "jpegli-rs-ycbcr-hybrid-progressive-420")
    #[must_use]
    pub fn name(&self) -> String {
        let hybrid_suffix = if self.hybrid { "-hybrid" } else { "" };
        format!(
            "{}-{}{}-{}-{}",
            self.encoder.short_name(),
            self.color.suffix(),
            hybrid_suffix,
            self.scan.suffix(),
            self.subsampling.suffix()
        )
    }

    /// Generate a short name (encoder + color + hybrid).
    ///
    /// Format: `encoder-color[-hybrid]` (e.g., "jpegli-rs-ycbcr-hybrid")
    #[must_use]
    pub fn short_name(&self) -> String {
        let hybrid_suffix = if self.hybrid { "-hybrid" } else { "" };
        format!(
            "{}-{}{}",
            self.encoder.short_name(),
            self.color.suffix(),
            hybrid_suffix
        )
    }

    /// Encode an image with this configuration.
    ///
    /// Returns the encoded JPEG bytes.
    pub fn encode(&self, img: &ImageData) -> Result<Vec<u8>, String> {
        match self.encoder {
            EncoderImpl::JpegliRs => self.encode_with_jpegli_rs(img),
            #[cfg(feature = "cjpegli-ffi")]
            EncoderImpl::CJpegli => self.encode_with_cjpegli_ffi(img),
            #[cfg(not(feature = "cjpegli-ffi"))]
            EncoderImpl::CJpegli => Err("cjpegli requires cjpegli-ffi feature".to_string()),
        }
    }

    fn encode_with_jpegli_rs(&self, img: &ImageData) -> Result<Vec<u8>, String> {
        use jpegli::encoder::{EncoderConfig, PixelLayout, XybSubsampling};

        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        if self.hybrid {
            return Err("hybrid requires experimental-hybrid-trellis feature".to_string());
        }

        // Create config based on color mode
        let config = if self.color == ColorMode::Xyb {
            EncoderConfig::xyb(self.quality as f32, XybSubsampling::BQuarter)
        } else {
            EncoderConfig::ycbcr(self.quality as f32, self.subsampling.to_v2())
        }
        .progressive(self.scan == ScanMode::Progressive);

        let mut enc = config
            .encode_from_bytes(img.width as u32, img.height as u32, PixelLayout::Rgb8Srgb)
            .map_err(|e| format!("jpegli-rs encode setup failed: {e}"))?;

        enc.push_packed(&img.pixels, enough::Unstoppable)
            .map_err(|e| format!("jpegli-rs encode failed: {e}"))?;

        enc.finish()
            .map_err(|e| format!("jpegli-rs encode finish failed: {e}"))
    }

    /// Encode using C++ jpegli via FFI (requires cjpegli-ffi feature).
    ///
    /// Note: XYB mode is NOT supported via the libjpeg API.
    #[cfg(feature = "cjpegli-ffi")]
    fn encode_with_cjpegli_ffi(&self, img: &ImageData) -> Result<Vec<u8>, String> {
        use jpegli_internals_sys::*;
        use std::mem::MaybeUninit;
        use std::ptr;

        if self.color == ColorMode::Xyb {
            return Err("cjpegli FFI does not support XYB (use CLI fallback)".to_string());
        }

        unsafe {
            // Use proper structs - sizes match C library (520 bytes for compress, 168 for error)
            let mut cinfo: MaybeUninit<jpeg_compress_struct> = MaybeUninit::zeroed();
            let mut jerr: MaybeUninit<jpeg_error_mgr> = MaybeUninit::zeroed();

            let cinfo_ptr = cinfo.as_mut_ptr();
            let jerr_ptr = jerr.as_mut_ptr();

            // Initialize error handler FIRST (before CreateCompress)
            (*cinfo_ptr).err = jpeg_std_error(jerr_ptr);

            // Create compression object - size is validated by library
            jpeg_CreateCompress(
                cinfo_ptr,
                JPEG_LIB_VERSION as i32,
                std::mem::size_of::<jpeg_compress_struct>(),
            );

            // Setup memory destination
            let mut outbuffer: *mut u8 = ptr::null_mut();
            let mut outsize: std::os::raw::c_ulong = 0;
            jpeg_mem_dest(cinfo_ptr, &mut outbuffer, &mut outsize);

            // Set image parameters
            (*cinfo_ptr).image_width = img.width as u32;
            (*cinfo_ptr).image_height = img.height as u32;
            (*cinfo_ptr).input_components = 3;
            (*cinfo_ptr).in_color_space = JCS_RGB as u32;

            // Set defaults (this sets up YCbCr conversion, quant tables, etc.)
            jpeg_set_defaults(cinfo_ptr);

            // Set quality
            jpeg_set_quality(cinfo_ptr, self.quality as i32, 1);

            // Set subsampling via comp_info
            if !(*cinfo_ptr).comp_info.is_null() {
                let comp_info = (*cinfo_ptr).comp_info;
                #[allow(unreachable_patterns)] // ChromaSubsampling is #[non_exhaustive]
                match self.subsampling {
                    ChromaSubsampling::S444 => {
                        (*comp_info.add(0)).h_samp_factor = 1;
                        (*comp_info.add(0)).v_samp_factor = 1;
                        (*comp_info.add(1)).h_samp_factor = 1;
                        (*comp_info.add(1)).v_samp_factor = 1;
                        (*comp_info.add(2)).h_samp_factor = 1;
                        (*comp_info.add(2)).v_samp_factor = 1;
                    }
                    ChromaSubsampling::S422 => {
                        (*comp_info.add(0)).h_samp_factor = 2;
                        (*comp_info.add(0)).v_samp_factor = 1;
                        (*comp_info.add(1)).h_samp_factor = 1;
                        (*comp_info.add(1)).v_samp_factor = 1;
                        (*comp_info.add(2)).h_samp_factor = 1;
                        (*comp_info.add(2)).v_samp_factor = 1;
                    }
                    ChromaSubsampling::S420 => {
                        (*comp_info.add(0)).h_samp_factor = 2;
                        (*comp_info.add(0)).v_samp_factor = 2;
                        (*comp_info.add(1)).h_samp_factor = 1;
                        (*comp_info.add(1)).v_samp_factor = 1;
                        (*comp_info.add(2)).h_samp_factor = 1;
                        (*comp_info.add(2)).v_samp_factor = 1;
                    }
                    ChromaSubsampling::S440 => {
                        (*comp_info.add(0)).h_samp_factor = 1;
                        (*comp_info.add(0)).v_samp_factor = 2;
                        (*comp_info.add(1)).h_samp_factor = 1;
                        (*comp_info.add(1)).v_samp_factor = 1;
                        (*comp_info.add(2)).h_samp_factor = 1;
                        (*comp_info.add(2)).v_samp_factor = 1;
                    }
                    // Non-exhaustive fallback
                    _ => {}
                }
            }

            // Set progressive mode
            if self.scan == ScanMode::Progressive {
                jpeg_simple_progression(cinfo_ptr);
            }

            // Enable Huffman optimization (matches jpegli-rs default)
            (*cinfo_ptr).optimize_coding = 1;

            // Start compression
            jpeg_start_compress(cinfo_ptr, 1);

            // Write scanlines
            let row_stride = img.width * 3;
            let mut row_pointer: [JSAMPROW; 1] = [ptr::null_mut()];

            #[allow(clippy::while_immutable_condition)]
            // FFI: jpeg_write_scanlines mutates next_scanline
            while (*cinfo_ptr).next_scanline < (*cinfo_ptr).image_height {
                let row_idx = (*cinfo_ptr).next_scanline as usize;
                let row_start = row_idx * row_stride;
                row_pointer[0] = img.pixels.as_ptr().add(row_start) as *mut u8;
                jpeg_write_scanlines(cinfo_ptr, row_pointer.as_mut_ptr(), 1);
            }

            // Finish compression
            jpeg_finish_compress(cinfo_ptr);

            // Copy output to Vec
            let result = if !outbuffer.is_null() && outsize > 0 {
                Ok(std::slice::from_raw_parts(outbuffer, outsize as usize).to_vec())
            } else {
                Err("cjpegli FFI produced empty output".to_string())
            };

            // Cleanup
            jpeg_destroy_compress(cinfo_ptr);

            // Free the output buffer allocated by jpeg_mem_dest
            if !outbuffer.is_null() {
                libc::free(outbuffer as *mut std::ffi::c_void);
            }

            result
        }
    }
}

// ============================================================================
// Convenience Encoder Functions (Explicit Names)
// ============================================================================

/// Encode with jpegli-rs using YCbCr color space (default settings).
pub fn encode_jpegli_rs_ycbcr(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::YCbCr)
        .quality(quality)
        .encode(img)
        .expect("jpegli-rs ycbcr encode")
}

/// Encode with jpegli-rs using XYB color space.
pub fn encode_jpegli_rs_xyb(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::Xyb)
        .quality(quality)
        .encode(img)
        .expect("jpegli-rs xyb encode")
}

/// Encode with jpegli-rs, YCbCr, progressive, 4:4:4 subsampling.
pub fn encode_jpegli_rs_ycbcr_progressive_444(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Progressive)
        .subsampling(ChromaSubsampling::S444)
        .quality(quality)
        .encode(img)
        .expect("jpegli-rs ycbcr progressive 444 encode")
}

/// Encode with jpegli-rs, YCbCr, baseline, 4:2:0 subsampling.
pub fn encode_jpegli_rs_ycbcr_baseline_420(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Baseline)
        .subsampling(ChromaSubsampling::S420)
        .quality(quality)
        .encode(img)
        .expect("jpegli-rs ycbcr baseline 420 encode")
}

/// Encode with jpegli-rs, XYB, progressive, 4:4:4 subsampling.
pub fn encode_jpegli_rs_xyb_progressive_444(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::JpegliRs)
        .color(ColorMode::Xyb)
        .scan(ScanMode::Progressive)
        .subsampling(ChromaSubsampling::S444)
        .quality(quality)
        .encode(img)
        .expect("jpegli-rs xyb progressive 444 encode")
}

/// Encode with C++ jpegli via FFI (requires cjpegli-ffi feature).
#[cfg(feature = "cjpegli-ffi")]
pub fn encode_cjpegli_ffi(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .quality(quality)
        .encode(img)
        .expect("cjpegli-ffi ycbcr encode")
}

/// Encode with C++ jpegli via FFI, progressive mode.
#[cfg(feature = "cjpegli-ffi")]
pub fn encode_cjpegli_ffi_progressive(img: &ImageData, quality: u8) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .scan(ScanMode::Progressive)
        .quality(quality)
        .encode(img)
        .expect("cjpegli-ffi ycbcr progressive encode")
}

/// Encode with C++ jpegli via FFI with specific subsampling.
#[cfg(feature = "cjpegli-ffi")]
pub fn encode_cjpegli_ffi_with_subsampling(
    img: &ImageData,
    quality: u8,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    EncoderConfig::new(EncoderImpl::CJpegli)
        .color(ColorMode::YCbCr)
        .subsampling(subsampling)
        .quality(quality)
        .encode(img)
        .expect("cjpegli-ffi encode")
}

// Legacy aliases for backwards compatibility
#[deprecated(note = "Use encode_jpegli_rs_ycbcr instead")]
pub fn encode_jpegli(img: &ImageData, quality: u8) -> Vec<u8> {
    encode_jpegli_rs_ycbcr(img, quality)
}

#[deprecated(note = "Use encode_jpegli_rs_xyb instead")]
pub fn encode_jpegli_xyb(img: &ImageData, quality: u8) -> Vec<u8> {
    encode_jpegli_rs_xyb(img, quality)
}

/// Full encoding + quality measurement result.
#[derive(Debug, Clone)]
pub struct EncodingMetrics {
    /// Encoder name
    pub encoder: String,
    /// Quality setting used
    pub quality: u8,
    /// Encoded size in bytes
    pub bytes: usize,
    /// Bits per pixel
    pub bpp: f64,
    /// DSSIM score (lower is better)
    pub dssim: f64,
    /// SSIMULACRA2 score (higher is better)
    pub ssimulacra2: f64,
    /// Butteraugli distance (lower is better)
    pub butteraugli: f64,
    /// Encoding time in milliseconds (optional)
    pub encode_ms: Option<f64>,
}

impl EncodingMetrics {
    /// Compute metrics for encoded JPEG data.
    pub fn compute(
        encoder: impl Into<String>,
        quality: u8,
        original: &ImageData,
        jpeg_data: &[u8],
    ) -> Self {
        let (decoded, _, _) = decode_jpeg(jpeg_data).expect("decode for metrics");
        let orig_img = original.as_rgb_image();
        let dec_img = bytes_to_rgb(&decoded, original.width, original.height);

        let pixels = original.pixel_count();

        Self {
            encoder: encoder.into(),
            quality,
            bytes: jpeg_data.len(),
            bpp: jpeg_data.len() as f64 * 8.0 / pixels as f64,
            dssim: QualityMetrics::dssim(orig_img.as_ref(), dec_img.as_ref()),
            ssimulacra2: QualityMetrics::ssimulacra2(orig_img.as_ref(), dec_img.as_ref()),
            butteraugli: QualityMetrics::butteraugli(orig_img.as_ref(), dec_img.as_ref()),
            encode_ms: None,
        }
    }

    /// Add timing information.
    #[must_use]
    pub fn with_timing(mut self, ms: f64) -> Self {
        self.encode_ms = Some(ms);
        self
    }
}

// ============================================================================
// Table Formatting
// ============================================================================

/// Format a value with alignment for table output.
pub struct TableColumn {
    /// Column header
    pub header: &'static str,
    /// Width in characters
    pub width: usize,
    /// Alignment (true = right, false = left)
    pub right_align: bool,
}

impl TableColumn {
    /// Create a left-aligned column.
    #[must_use]
    pub const fn left(header: &'static str, width: usize) -> Self {
        Self {
            header,
            width,
            right_align: false,
        }
    }

    /// Create a right-aligned column.
    #[must_use]
    pub const fn right(header: &'static str, width: usize) -> Self {
        Self {
            header,
            width,
            right_align: true,
        }
    }

    /// Format a value for this column.
    #[must_use]
    pub fn format(&self, value: &str) -> String {
        if self.right_align {
            format!("{:>width$}", value, width = self.width)
        } else {
            format!("{:<width$}", value, width = self.width)
        }
    }
}

/// Print a simple text table header.
pub fn print_table_header(columns: &[TableColumn]) {
    let header: String = columns
        .iter()
        .map(|c| c.format(c.header))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));
}

/// Print a table row.
pub fn print_table_row(columns: &[TableColumn], values: &[String]) {
    let row: String = columns
        .iter()
        .zip(values.iter())
        .map(|(c, v)| c.format(v))
        .collect::<Vec<_>>()
        .join("  ");
    println!("{}", row);
}

/// Standard columns for encoder comparison tables.
pub fn encoder_comparison_columns() -> Vec<TableColumn> {
    vec![
        TableColumn::left("Encoder", 15),
        TableColumn::right("Q", 3),
        TableColumn::right("Size", 10),
        TableColumn::right("BPP", 6),
        TableColumn::right("DSSIM", 10),
        TableColumn::right("SSIM2", 6),
        TableColumn::right("Bfly", 6),
    ]
}

/// Format EncodingMetrics as table row values.
pub fn metrics_to_row(m: &EncodingMetrics) -> Vec<String> {
    vec![
        m.encoder.clone(),
        format!("{}", m.quality),
        format!("{}", m.bytes),
        format!("{:.3}", m.bpp),
        format!("{:.6}", m.dssim),
        format!("{:.2}", m.ssimulacra2),
        format!("{:.3}", m.butteraugli),
    ]
}

// ============================================================================
// CSV Output
// ============================================================================

/// CSV output helper for benchmark results.
pub struct CsvWriter<W: std::io::Write> {
    writer: W,
    headers_written: bool,
}

impl<W: std::io::Write> CsvWriter<W> {
    /// Create a new CSV writer.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            headers_written: false,
        }
    }

    /// Write CSV header row.
    pub fn write_header(&mut self, fields: &[&str]) -> std::io::Result<()> {
        writeln!(self.writer, "{}", fields.join(","))?;
        self.headers_written = true;
        Ok(())
    }

    /// Write a row of values.
    pub fn write_row(&mut self, values: &[String]) -> std::io::Result<()> {
        writeln!(self.writer, "{}", values.join(","))
    }

    /// Write EncodingMetrics as a CSV row.
    pub fn write_metrics(
        &mut self,
        img: &ImageData,
        metrics: &EncodingMetrics,
    ) -> std::io::Result<()> {
        if !self.headers_written {
            self.write_header(&[
                "encoder",
                "quality",
                "image",
                "width",
                "height",
                "bytes",
                "bpp",
                "dssim",
                "ssimulacra2",
                "butteraugli",
            ])?;
        }

        let row = vec![
            metrics.encoder.clone(),
            metrics.quality.to_string(),
            img.name.clone(),
            img.width.to_string(),
            img.height.to_string(),
            metrics.bytes.to_string(),
            format!("{:.4}", metrics.bpp),
            format!("{:.6}", metrics.dssim),
            format!("{:.2}", metrics.ssimulacra2),
            format!("{:.4}", metrics.butteraugli),
        ];
        self.write_row(&row)
    }

    /// Flush the writer.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// Standard CSV header for encoder comparison.
pub const CSV_HEADER: &str =
    "encoder,quality,image,width,height,bytes,bpp,dssim,ssimulacra2,butteraugli";

// ============================================================================
// Progress Reporting
// ============================================================================

/// Progress tracker for long-running operations.
pub struct Progress {
    total: usize,
    completed: usize,
    start: std::time::Instant,
    last_report: std::time::Instant,
    report_interval_ms: u64,
}

impl Progress {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new(total: usize) -> Self {
        let now = std::time::Instant::now();
        Self {
            total,
            completed: 0,
            start: now,
            last_report: now,
            report_interval_ms: 1000,
        }
    }

    /// Set minimum interval between progress reports (default: 1000ms).
    #[must_use]
    pub fn with_interval_ms(mut self, ms: u64) -> Self {
        self.report_interval_ms = ms;
        self
    }

    /// Increment progress and optionally print status.
    ///
    /// Returns true if a report was printed.
    pub fn tick(&mut self) -> bool {
        self.completed += 1;

        let now = std::time::Instant::now();
        if now.duration_since(self.last_report).as_millis() >= self.report_interval_ms as u128 {
            self.report();
            self.last_report = now;
            true
        } else {
            false
        }
    }

    /// Force print current progress.
    pub fn report(&self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = self.completed as f64 / elapsed;
        let remaining = if rate > 0.0 {
            (self.total - self.completed) as f64 / rate
        } else {
            0.0
        };
        let pct = 100.0 * self.completed as f64 / self.total as f64;

        eprintln!(
            "Progress: {}/{} ({:.1}%), ETA: {:.0}s",
            self.completed, self.total, pct, remaining
        );
    }

    /// Print final summary.
    pub fn finish(&self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let rate = self.completed as f64 / elapsed;
        eprintln!(
            "Completed {} items in {:.1}s ({:.1}/s)",
            self.completed, elapsed, rate
        );
    }
}

// ============================================================================
// Throughput Formatting
// ============================================================================

/// Format throughput as megapixels per second.
#[must_use]
pub fn format_throughput_mpps(pixels: usize, time_secs: f64) -> String {
    let mpps = pixels as f64 / time_secs / 1_000_000.0;
    format!("{:.1} MP/s", mpps)
}

/// Format file size in human-readable form.
#[must_use]
pub fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration in human-readable form.
#[must_use]
pub fn format_duration(secs: f64) -> String {
    if secs >= 60.0 {
        let mins = (secs / 60.0).floor();
        let rem = secs - mins * 60.0;
        format!("{}m {:.1}s", mins as u32, rem)
    } else if secs >= 1.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.1}ms", secs * 1000.0)
    }
}

// ============================================================================
// SVG Chart Generation
// ============================================================================

/// A data series for charting.
#[derive(Debug, Clone)]
pub struct ChartSeries {
    /// Series name (shown in legend)
    pub name: String,
    /// Color (CSS color string, e.g., "#2196F3")
    pub color: String,
    /// Data points as (x, y) pairs
    pub points: Vec<(f64, f64)>,
}

impl ChartSeries {
    /// Create a new series.
    pub fn new(name: impl Into<String>, color: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            color: color.into(),
            points: Vec::new(),
        }
    }

    /// Add a data point.
    pub fn point(mut self, x: f64, y: f64) -> Self {
        self.points.push((x, y));
        self
    }

    /// Add multiple points.
    pub fn points(mut self, pts: impl IntoIterator<Item = (f64, f64)>) -> Self {
        self.points.extend(pts);
        self
    }
}

/// SVG line chart builder.
///
/// Creates SVG charts suitable for embedding in HTML reports.
pub struct SvgChart {
    width: f64,
    height: f64,
    margin: f64,
    x_label: String,
    y_label: String,
    title: Option<String>,
    y_lower_better: bool,
    series: Vec<ChartSeries>,
}

impl Default for SvgChart {
    fn default() -> Self {
        Self::new()
    }
}

impl SvgChart {
    /// Create a new chart with default dimensions.
    #[must_use]
    pub fn new() -> Self {
        Self {
            width: 450.0,
            height: 350.0,
            margin: 55.0,
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            title: None,
            y_lower_better: false,
            series: Vec::new(),
        }
    }

    /// Set chart dimensions.
    #[must_use]
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set margin around the plot area.
    #[must_use]
    pub fn margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Set X-axis label.
    #[must_use]
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = label.into();
        self
    }

    /// Set Y-axis label.
    #[must_use]
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = label.into();
        self
    }

    /// Set chart title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set whether lower Y values are better (inverts Y axis display).
    #[must_use]
    pub fn y_lower_better(mut self, lower_better: bool) -> Self {
        self.y_lower_better = lower_better;
        self
    }

    /// Add a data series.
    #[must_use]
    pub fn series(mut self, s: ChartSeries) -> Self {
        self.series.push(s);
        self
    }

    /// Render to SVG string.
    #[must_use]
    pub fn render(&self) -> String {
        let plot_width = self.width - 2.0 * self.margin;
        let plot_height = self.height - 2.0 * self.margin;

        // Find data ranges
        let (min_x, max_x, min_y, max_y) = self.data_ranges();

        // Scale functions
        let scale_x = |x: f64| self.margin + (x - min_x) / (max_x - min_x) * plot_width;
        let scale_y = |y: f64| {
            if self.y_lower_better {
                self.margin + plot_height - (y - min_y) / (max_y - min_y) * plot_height
            } else {
                self.margin + (max_y - y) / (max_y - min_y) * plot_height
            }
        };

        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis {{ stroke: #333; stroke-width: 1; }}
    .grid {{ stroke: #eee; stroke-width: 0.5; }}
    .label {{ font-family: sans-serif; font-size: 10px; }}
    .title {{ font-family: sans-serif; font-size: 14px; font-weight: bold; }}
    .legend {{ font-family: sans-serif; font-size: 10px; }}
  </style>
"#,
            self.width, self.height
        );

        // Title
        if let Some(title) = &self.title {
            svg.push_str(&format!(
                r#"  <text x="{}" y="18" class="title" text-anchor="middle">{}</text>
"#,
                self.width / 2.0,
                title
            ));
        }

        // Axes
        svg.push_str(&format!(
            r#"  <line x1="{m}" y1="{m}" x2="{m}" y2="{b}" class="axis"/>
  <line x1="{m}" y1="{b}" x2="{r}" y2="{b}" class="axis"/>
"#,
            m = self.margin,
            b = self.height - self.margin,
            r = self.width - self.margin
        ));

        // Axis labels
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{}</text>
  <text x="12" y="{}" class="label" text-anchor="middle" transform="rotate(-90, 12, {})">{}</text>
"#,
            self.width / 2.0,
            self.height - 8.0,
            self.x_label,
            self.height / 2.0,
            self.height / 2.0,
            self.y_label
        ));

        // Grid and ticks
        for i in 0..=4 {
            let x = self.margin + plot_width * i as f64 / 4.0;
            let y = self.margin + plot_height * i as f64 / 4.0;

            // Grid lines
            svg.push_str(&format!(
                r#"  <line x1="{x}" y1="{m}" x2="{x}" y2="{b}" class="grid"/>
  <line x1="{m}" y1="{y}" x2="{r}" y2="{y}" class="grid"/>
"#,
                x = x,
                y = y,
                m = self.margin,
                b = self.height - self.margin,
                r = self.width - self.margin
            ));

            // X tick labels
            let x_val = min_x + (max_x - min_x) * i as f64 / 4.0;
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="label" text-anchor="middle">{:.2}</text>
"#,
                x,
                self.height - self.margin + 12.0,
                x_val
            ));

            // Y tick labels
            let y_val = if self.y_lower_better {
                max_y - (max_y - min_y) * i as f64 / 4.0
            } else {
                min_y + (max_y - min_y) * (4 - i) as f64 / 4.0
            };
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" class="label" text-anchor="end">{:.3}</text>
"#,
                self.margin - 4.0,
                y + 3.0,
                y_val
            ));
        }

        // Draw each series
        for series in &self.series {
            if series.points.is_empty() {
                continue;
            }

            // Line path
            let mut path = String::new();
            for (i, &(x, y)) in series.points.iter().enumerate() {
                let sx = scale_x(x);
                let sy = scale_y(y);
                if i == 0 {
                    path.push_str(&format!("M {} {}", sx, sy));
                } else {
                    path.push_str(&format!(" L {} {}", sx, sy));
                }
            }
            svg.push_str(&format!(
                r#"  <path d="{}" stroke="{}" fill="none" stroke-width="2"/>
"#,
                path, series.color
            ));

            // Data points
            for &(x, y) in &series.points {
                let sx = scale_x(x);
                let sy = scale_y(y);
                svg.push_str(&format!(
                    r#"  <circle cx="{}" cy="{}" r="3" fill="{}"/>
"#,
                    sx, sy, series.color
                ));
            }
        }

        // Legend
        if !self.series.is_empty() {
            let legend_x = self.width - 90.0;
            let legend_y = 10.0;
            let legend_h = 15.0 * self.series.len() as f64 + 8.0;

            svg.push_str(&format!(
                r##"  <rect x="{}" y="{}" width="85" height="{}" fill="white" stroke="#ccc" rx="3"/>
"##,
                legend_x, legend_y, legend_h
            ));

            for (i, series) in self.series.iter().enumerate() {
                let y = legend_y + 14.0 + 15.0 * i as f64;
                svg.push_str(&format!(
                    r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>
  <circle cx="{}" cy="{}" r="3" fill="{}"/>
  <text x="{}" y="{}" class="legend">{}</text>
"#,
                    legend_x + 5.0,
                    y,
                    legend_x + 20.0,
                    y,
                    series.color,
                    legend_x + 12.5,
                    y,
                    series.color,
                    legend_x + 25.0,
                    y + 3.0,
                    series.name
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }

    fn data_ranges(&self) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for series in &self.series {
            for &(x, y) in &series.points {
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }

        // Add 10% padding
        let x_range = max_x - min_x;
        let y_range = max_y - min_y;
        min_x -= x_range * 0.1;
        max_x += x_range * 0.1;
        min_y = (min_y - y_range * 0.1).max(0.0);
        max_y += y_range * 0.1;

        (min_x, max_x, min_y, max_y)
    }
}

// ============================================================================
// HTML Report Generation
// ============================================================================

/// HTML report builder for benchmark results.
pub struct HtmlReport {
    title: String,
    sections: Vec<HtmlSection>,
}

enum HtmlSection {
    Paragraph(String),
    Chart(String),
    Table {
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        highlight_col: Option<usize>,
    },
    Note(String),
}

impl HtmlReport {
    /// Create a new report with the given title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            sections: Vec::new(),
        }
    }

    /// Add a paragraph of text.
    #[must_use]
    pub fn paragraph(mut self, text: impl Into<String>) -> Self {
        self.sections.push(HtmlSection::Paragraph(text.into()));
        self
    }

    /// Add an SVG chart.
    #[must_use]
    pub fn chart(mut self, svg: impl Into<String>) -> Self {
        self.sections.push(HtmlSection::Chart(svg.into()));
        self
    }

    /// Add a data table.
    #[must_use]
    pub fn table(mut self, headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        self.sections.push(HtmlSection::Table {
            headers,
            rows,
            highlight_col: None,
        });
        self
    }

    /// Add a data table with a column highlighted for best values.
    #[must_use]
    pub fn table_with_highlight(
        mut self,
        headers: Vec<String>,
        rows: Vec<Vec<String>>,
        highlight_col: usize,
    ) -> Self {
        self.sections.push(HtmlSection::Table {
            headers,
            rows,
            highlight_col: Some(highlight_col),
        });
        self
    }

    /// Add a note/footnote.
    #[must_use]
    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.sections.push(HtmlSection::Note(text.into()));
        self
    }

    /// Render to HTML string.
    #[must_use]
    pub fn render(&self) -> String {
        let mut body = String::new();

        for section in &self.sections {
            match section {
                HtmlSection::Paragraph(text) => {
                    body.push_str(&format!("        <p>{}</p>\n", text));
                }
                HtmlSection::Chart(svg) => {
                    body.push_str(&format!("        <div class=\"chart\">{}</div>\n", svg));
                }
                HtmlSection::Table {
                    headers,
                    rows,
                    highlight_col,
                } => {
                    body.push_str("        <table>\n            <tr>");
                    for h in headers {
                        body.push_str(&format!("<th>{}</th>", h));
                    }
                    body.push_str("</tr>\n");

                    for row in rows {
                        body.push_str("            <tr>");
                        for (i, cell) in row.iter().enumerate() {
                            let class = if highlight_col.is_some_and(|c| c == i) {
                                " class=\"highlight\""
                            } else {
                                ""
                            };
                            body.push_str(&format!("<td{}>{}</td>", class, cell));
                        }
                        body.push_str("</tr>\n");
                    }
                    body.push_str("        </table>\n");
                }
                HtmlSection::Note(text) => {
                    body.push_str(&format!("        <p class=\"note\">{}</p>\n", text));
                }
            }
        }

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .chart {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; margin-top: 20px; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
        th {{ background: #f0f0f0; }}
        td:first-child {{ text-align: left; }}
        .highlight {{ background: #e8f5e9; }}
        .note {{ color: #666; font-size: 14px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
{body}
    </div>
</body>
</html>"#,
            title = self.title,
            body = body
        )
    }

    /// Write to a file.
    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.render())
    }
}

// ============================================================================
// File Caching (for repeated corpus analysis)
// ============================================================================

/// Cache for encoded files to speed up repeated analysis.
pub struct FileCache {
    cache_dir: PathBuf,
    version: String,
}

impl FileCache {
    /// Create a new file cache.
    ///
    /// `cache_dir` is where cached files are stored.
    /// `version` should be bumped when encoder changes to invalidate old caches.
    pub fn new(cache_dir: impl Into<PathBuf>, version: impl Into<String>) -> Self {
        let cache_dir = cache_dir.into();
        let _ = std::fs::create_dir_all(&cache_dir);
        Self {
            cache_dir,
            version: version.into(),
        }
    }

    /// Get the cache path for a file/encoder/quality combination.
    #[must_use]
    pub fn path(&self, filename: &str, encoder: &str, quality: u8) -> PathBuf {
        self.cache_dir.join(format!(
            "{}_{}_q{}_{}.jpg",
            filename, encoder, quality, self.version
        ))
    }

    /// Load from cache or encode using the provided function.
    ///
    /// Returns (data, was_cached).
    pub fn get_or_encode<F>(
        &self,
        filename: &str,
        encoder: &str,
        quality: u8,
        encode_fn: F,
    ) -> (Vec<u8>, bool)
    where
        F: FnOnce() -> Vec<u8>,
    {
        let path = self.path(filename, encoder, quality);

        if path.exists() {
            if let Ok(data) = std::fs::read(&path) {
                return (data, true);
            }
        }

        let data = encode_fn();

        // Save to cache (ignore errors)
        let _ = std::fs::write(&path, &data);

        (data, false)
    }

    /// Clear all cached files.
    pub fn clear(&self) -> std::io::Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)?;
            std::fs::create_dir_all(&self.cache_dir)?;
        }
        Ok(())
    }
}

// ============================================================================
// Standard Colors for Encoders
// ============================================================================

/// Standard colors for different encoders (for consistent charts).
pub mod colors {
    pub const JPEGLI: &str = "#2196F3"; // Blue
    pub const JPEGLI_XYB: &str = "#9C27B0"; // Purple
    pub const MOZJPEG: &str = "#4CAF50"; // Green
    pub const LIBJPEG: &str = "#FF5722"; // Deep Orange
    pub const WEBP: &str = "#FFC107"; // Amber
    pub const AVIF: &str = "#00BCD4"; // Cyan
    pub const JXL: &str = "#E91E63"; // Pink
    pub const HEIC: &str = "#607D8B"; // Blue Gray
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_generation() {
        let img = SyntheticPattern::GradientH.generate(256, 64);
        assert_eq!(img.width(), 256);
        assert_eq!(img.height(), 64);

        // First pixel should be dark
        let first = img.buf()[0];
        assert_eq!(first.r, 0);

        // Last pixel of first row should be bright
        let last = img.buf()[255];
        assert_eq!(last.r, 255);
    }

    #[test]
    fn test_checkerboard() {
        let img = SyntheticPattern::Checkerboard { block_size: 8 }.generate(64, 64);
        assert_eq!(img.width(), 64);

        // Top-left should be white
        assert_eq!(img.buf()[0].r, 255);
        // Position 8,0 should be black
        assert_eq!(img.buf()[8].r, 0);
    }

    #[test]
    fn test_noise_deterministic() {
        let img1 = SyntheticPattern::Noise { seed: 42 }.generate(64, 64);
        let img2 = SyntheticPattern::Noise { seed: 42 }.generate(64, 64);
        let img3 = SyntheticPattern::Noise { seed: 99 }.generate(64, 64);

        // Same seed = same image
        assert_eq!(img1.buf(), img2.buf());
        // Different seed = different image
        assert_ne!(img1.buf(), img3.buf());
    }

    #[test]
    fn test_test_sizes() {
        assert_eq!(TestSize::Tiny.dimensions(), (64, 64));
        assert_eq!(TestSize::Hd.dimensions(), (1920, 1080));
        assert!((TestSize::Hd.megapixels() - 2.0736).abs() < 0.001);
    }

    #[test]
    fn test_rms_identical() {
        let img = SyntheticPattern::GradientRgb.generate(64, 64);
        let rms = QualityMetrics::rms(img.as_ref(), img.as_ref());
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_max_diff() {
        let img1 = generate_solid(64, 64, RGB8::new(100, 100, 100));
        let img2 = generate_solid(64, 64, RGB8::new(110, 100, 90));

        let diff = QualityMetrics::max_pixel_diff(img1.as_ref(), img2.as_ref());
        assert_eq!(diff, 10);
    }

    #[test]
    fn test_rgb_bytes_roundtrip() {
        let img = SyntheticPattern::ColorBars.generate(64, 32);
        let bytes = rgb_to_bytes(img.as_ref());
        let img2 = bytes_to_rgb(&bytes, 64, 32);

        assert_eq!(img.buf(), img2.buf());
    }

    #[test]
    fn test_dssim_identical() {
        let img = SyntheticPattern::GradientRgb.generate(64, 64);
        let dssim = QualityMetrics::dssim(img.as_ref(), img.as_ref());
        assert!(dssim < 0.0001, "DSSIM of identical images should be ~0");
    }

    #[test]
    fn test_ssimulacra2_identical() {
        let img = SyntheticPattern::GradientRgb.generate(64, 64);
        let ssim2 = QualityMetrics::ssimulacra2(img.as_ref(), img.as_ref());
        assert!(
            ssim2 > 99.0,
            "SSIMULACRA2 of identical images should be ~100"
        );
    }

    #[test]
    fn test_svg_chart_renders() {
        let chart = SvgChart::new()
            .size(400.0, 300.0)
            .x_label("BPP")
            .y_label("Quality")
            .series(
                ChartSeries::new("test", colors::JPEGLI)
                    .point(0.5, 80.0)
                    .point(1.0, 90.0)
                    .point(1.5, 95.0),
            );

        let svg = chart.render();
        assert!(svg.contains("<svg"), "Should start with SVG tag");
        assert!(svg.contains("</svg>"), "Should end with SVG tag");
        assert!(svg.contains("test"), "Should contain series name");
        assert!(svg.contains(colors::JPEGLI), "Should contain series color");
    }

    #[test]
    fn test_html_report_renders() {
        let report = HtmlReport::new("Test Report")
            .paragraph("This is a test.")
            .note("This is a note.");

        let html = report.render();
        assert!(html.contains("<!DOCTYPE html>"), "Should have doctype");
        assert!(html.contains("Test Report"), "Should contain title");
        assert!(html.contains("This is a test"), "Should contain paragraph");
        assert!(html.contains("This is a note"), "Should contain note");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 B");
        assert_eq!(format_size(1500), "1.5 KB");
        assert_eq!(format_size(1_500_000), "1.50 MB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.5), "500.0ms");
        assert_eq!(format_duration(5.5), "5.50s");
        assert_eq!(format_duration(90.0), "1m 30.0s");
    }

    #[test]
    fn test_encoder_config_naming() {
        let config = EncoderConfig::new(EncoderImpl::JpegliRs)
            .color(ColorMode::Xyb)
            .scan(ScanMode::Progressive)
            .subsampling(ChromaSubsampling::S444);

        assert_eq!(config.name(), "jpegli-rs-xyb-progressive-444");
        assert_eq!(config.short_name(), "jpegli-rs-xyb");

        let config2 = EncoderConfig::new(EncoderImpl::CJpegli)
            .color(ColorMode::YCbCr)
            .scan(ScanMode::Baseline)
            .subsampling(ChromaSubsampling::S420);

        assert_eq!(config2.name(), "cjpegli-ycbcr-baseline-420");

        // Test hybrid naming
        let config3 = EncoderConfig::new(EncoderImpl::JpegliRs)
            .color(ColorMode::YCbCr)
            .hybrid(true)
            .scan(ScanMode::Progressive)
            .subsampling(ChromaSubsampling::S420);

        assert_eq!(config3.name(), "jpegli-rs-ycbcr-hybrid-progressive-420");
        assert_eq!(config3.short_name(), "jpegli-rs-ycbcr-hybrid");
    }
}
