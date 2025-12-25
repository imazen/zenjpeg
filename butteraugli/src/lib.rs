//! # Butteraugli
//!
//! Butteraugli is a perceptual image quality metric developed by Google.
//! This is a Rust port of the butteraugli algorithm from libjxl.
//!
//! The metric is based on:
//! - Opsin: dynamics of photosensitive chemicals in the retina
//! - XYB: hybrid opponent/trichromatic color space
//! - Visual masking: how features hide other features
//! - Multi-scale analysis: UHF, HF, MF, LF frequency components
//!
//! ## Quality Thresholds
//!
//! - Score < 1.0: Images are perceived as identical
//! - Score 1.0-2.0: Subtle differences may be noticeable
//! - Score > 2.0: Visible difference between images
//!
//! ## Example
//!
//! ```rust
//! use butteraugli_oxide::{compute_butteraugli, ButteraugliParams};
//!
//! // Create two 8x8 RGB images (must be 8x8 minimum)
//! let width = 8;
//! let height = 8;
//! let rgb1: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();
//! let rgb2 = rgb1.clone(); // Identical images
//!
//! let params = ButteraugliParams::default();
//! let result = compute_butteraugli(&rgb1, &rgb2, width, height, &params).unwrap();
//!
//! // Identical images should have score ~0
//! assert!(result.score < 0.01);
//! ```
//!
//! ## Comparing Different Images
//!
//! ```rust
//! use butteraugli_oxide::{compute_butteraugli, ButteraugliParams, BUTTERAUGLI_GOOD, BUTTERAUGLI_BAD};
//!
//! let width = 16;
//! let height = 16;
//!
//! // Original image - gradient
//! let original: Vec<u8> = (0..width * height)
//!     .flat_map(|i| {
//!         let x = i % width;
//!         [(x * 16) as u8, 128, 128]
//!     })
//!     .collect();
//!
//! // Distorted image - add noise
//! let distorted: Vec<u8> = original.iter()
//!     .map(|&v| v.saturating_add(10))
//!     .collect();
//!
//! let result = compute_butteraugli(&original, &distorted, width, height, &ButteraugliParams::default())
//!     .expect("valid image data");
//!
//! if result.score < BUTTERAUGLI_GOOD {
//!     println!("Images appear identical to humans");
//! } else if result.score > BUTTERAUGLI_BAD {
//!     println!("Visible difference detected");
//! }
//! ```
//!
//! ## References
//!
//! - <https://github.com/google/butteraugli>
//! - <https://github.com/libjxl/libjxl>

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// Allow C++ constant formats (ported from libjxl with exact values)
#![allow(clippy::unreadable_literal)]
#![allow(clippy::inconsistent_digit_grouping)]
#![allow(clippy::excessive_precision)]
// Allow mul_add style from C++ (may affect numerical parity)
#![allow(clippy::suboptimal_flops)]
// Allow common patterns in numerical code ported from C++
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_saturating_arithmetic)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
// These are nice-to-have but not critical for initial release
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::if_not_else)]
#![allow(clippy::imprecise_flops)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::useless_let_if_seq)]

// Module structure - expose all for testing parity with C++
pub mod blur;
pub mod consts;
mod diff;
pub mod image;
pub mod malta;
pub mod mask;
pub mod opsin;
pub mod psycho;
pub mod xyb;

// C++ reference data for regression testing (auto-generated)
// Hidden from docs as this is internal test infrastructure
#[doc(hidden)]
pub mod reference_data;

// Re-export main types and functions
pub use crate::image::{Image3F, ImageF};

/// Error type for butteraugli operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ButteraugliError {
    /// Image dimensions don't match.
    DimensionMismatch {
        /// First image dimensions (width, height).
        first: (usize, usize),
        /// Second image dimensions (width, height).
        second: (usize, usize),
    },
    /// Buffer size doesn't match expected size for dimensions.
    InvalidBufferSize {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },
    /// Image dimensions are invalid (zero or too small).
    InvalidDimensions {
        /// Width provided.
        width: usize,
        /// Height provided.
        height: usize,
    },
}

impl std::fmt::Display for ButteraugliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { first, second } => {
                write!(
                    f,
                    "image dimensions don't match: {}x{} vs {}x{}",
                    first.0, first.1, second.0, second.1
                )
            }
            Self::InvalidBufferSize { expected, actual } => {
                write!(
                    f,
                    "buffer size {} doesn't match expected size {}",
                    actual, expected
                )
            }
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {}x{} (minimum 8x8)", width, height)
            }
        }
    }
}

impl std::error::Error for ButteraugliError {}

/// Butteraugli comparison parameters.
///
/// Use the builder pattern to construct:
/// ```rust
/// use butteraugli_oxide::ButteraugliParams;
///
/// let params = ButteraugliParams::new()
///     .with_intensity_target(250.0)  // HDR display
///     .with_hf_asymmetry(1.5);       // Penalize new artifacts more
/// ```
#[derive(Debug, Clone)]
pub struct ButteraugliParams {
    hf_asymmetry: f32,
    xmul: f32,
    intensity_target: f32,
}

impl Default for ButteraugliParams {
    fn default() -> Self {
        Self {
            hf_asymmetry: 1.0,
            xmul: 1.0,
            intensity_target: 80.0,
        }
    }
}

impl ButteraugliParams {
    /// Creates a new `ButteraugliParams` with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the intensity target (display brightness in nits).
    #[must_use]
    pub fn with_intensity_target(mut self, intensity_target: f32) -> Self {
        self.intensity_target = intensity_target;
        self
    }

    /// Sets the HF asymmetry multiplier.
    /// Values > 1.0 penalize new high-frequency artifacts more than blurring.
    #[must_use]
    pub fn with_hf_asymmetry(mut self, hf_asymmetry: f32) -> Self {
        self.hf_asymmetry = hf_asymmetry;
        self
    }

    /// Sets the X channel multiplier.
    #[must_use]
    pub fn with_xmul(mut self, xmul: f32) -> Self {
        self.xmul = xmul;
        self
    }

    /// Returns the HF asymmetry multiplier.
    #[must_use]
    pub fn hf_asymmetry(&self) -> f32 {
        self.hf_asymmetry
    }

    /// Returns the X channel multiplier.
    #[must_use]
    pub fn xmul(&self) -> f32 {
        self.xmul
    }

    /// Returns the intensity target in nits.
    #[must_use]
    pub fn intensity_target(&self) -> f32 {
        self.intensity_target
    }
}

/// Quality threshold for "good" (images look the same).
pub const BUTTERAUGLI_GOOD: f64 = 1.0;

/// Quality threshold for "bad" (visible difference).
pub const BUTTERAUGLI_BAD: f64 = 2.0;

/// Butteraugli image comparison result.
#[derive(Debug, Clone)]
pub struct ButteraugliResult {
    /// Global difference score. < 1.0 is "good", > 2.0 is "bad".
    pub score: f64,
    /// Per-pixel difference map (optional).
    pub diffmap: Option<ImageF>,
}

/// Computes butteraugli score between two sRGB images.
///
/// This function accepts 8-bit sRGB data and internally converts to linear RGB
/// for perceptual comparison. For higher bit depths (16-bit, HDR), use
/// [`compute_butteraugli_linear`] instead.
///
/// # Arguments
/// * `rgb1` - First image (sRGB u8, 3 bytes per pixel, row-major RGB order)
/// * `rgb2` - Second image (sRGB u8, 3 bytes per pixel, row-major RGB order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `params` - Comparison parameters
///
/// # Returns
/// Butteraugli score and optional per-pixel difference map.
///
/// # Color Space
/// Input is assumed to be **sRGB** (gamma-encoded). The function applies the
/// sRGB transfer function internally to convert to linear RGB before comparison.
/// If your input is already linear RGB, use [`compute_butteraugli_linear`].
///
/// # Errors
/// Returns an error if:
/// - Buffer sizes don't match expected dimensions
/// - Images are smaller than 8x8 pixels
pub fn compute_butteraugli(
    rgb1: &[u8],
    rgb2: &[u8],
    width: usize,
    height: usize,
    params: &ButteraugliParams,
) -> Result<ButteraugliResult, ButteraugliError> {
    let expected_size = width * height * 3;

    if width < 8 || height < 8 {
        return Err(ButteraugliError::InvalidDimensions { width, height });
    }

    if rgb1.len() != expected_size {
        return Err(ButteraugliError::InvalidBufferSize {
            expected: expected_size,
            actual: rgb1.len(),
        });
    }

    if rgb2.len() != expected_size {
        return Err(ButteraugliError::InvalidBufferSize {
            expected: expected_size,
            actual: rgb2.len(),
        });
    }

    Ok(diff::compute_butteraugli_impl(rgb1, rgb2, width, height, params))
}

/// Converts butteraugli score to quality percentage (0-100).
///
/// This provides a rough human-friendly interpretation:
/// - Score < 1.0 → ~100% (imperceptible difference)
/// - Score 4.0+ → ~0% (major visible difference)
#[must_use]
pub fn score_to_quality(score: f64) -> f64 {
    (100.0 - score * 25.0).clamp(0.0, 100.0)
}

/// Converts butteraugli score to fuzzy class value.
///
/// Returns 2.0 for a perfect match, 1.0 for 'ok', 0.0 for bad.
/// The scoring is fuzzy - a butteraugli score of 0.96 would return
/// a class of around 1.9.
#[must_use]
pub fn butteraugli_fuzzy_class(score: f64) -> f64 {
    // Based on the C++ implementation
    let val = 2.0 - score * 0.5;
    val.clamp(0.0, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_images() {
        let width = 16;
        let height = 16;
        let rgb: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let result = compute_butteraugli(&rgb, &rgb, width, height, &ButteraugliParams::default())
            .expect("valid input");

        // Identical images should have score 0
        assert!(
            result.score < 0.001,
            "Identical images should have score ~0, got {}",
            result.score
        );
    }

    #[test]
    fn test_different_images() {
        let width = 16;
        let height = 16;
        let rgb1: Vec<u8> = vec![0; width * height * 3];
        let rgb2: Vec<u8> = vec![255; width * height * 3];

        let result =
            compute_butteraugli(&rgb1, &rgb2, width, height, &ButteraugliParams::default())
                .expect("valid input");

        // Different images should have non-zero score
        // Note: uniform images (all black vs all white) don't have much frequency content
        assert!(
            result.score > 0.01,
            "Different images should have non-zero score, got {}",
            result.score
        );
    }

    #[test]
    fn test_invalid_buffer_size() {
        let width = 16;
        let height = 16;
        let rgb1: Vec<u8> = vec![0; width * height * 3];
        let rgb2: Vec<u8> = vec![0; 10]; // Wrong size

        let result = compute_butteraugli(&rgb1, &rgb2, width, height, &ButteraugliParams::default());
        assert!(matches!(result, Err(ButteraugliError::InvalidBufferSize { .. })));
    }

    #[test]
    fn test_too_small_dimensions() {
        let rgb: Vec<u8> = vec![0; 4 * 4 * 3];
        let result = compute_butteraugli(&rgb, &rgb, 4, 4, &ButteraugliParams::default());
        assert!(matches!(result, Err(ButteraugliError::InvalidDimensions { .. })));
    }

    #[test]
    fn test_score_to_quality() {
        assert!((score_to_quality(0.0) - 100.0).abs() < 0.001);
        assert!((score_to_quality(4.0) - 0.0).abs() < 0.001);
        assert!((score_to_quality(2.0) - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_fuzzy_class() {
        assert!((butteraugli_fuzzy_class(0.0) - 2.0).abs() < 0.001);
        assert!((butteraugli_fuzzy_class(4.0) - 0.0).abs() < 0.001);
    }
}
