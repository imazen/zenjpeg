//! JPEG encoder implementation.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use zenjpeg::{EncoderConfig, ChromaSubsampling, PixelLayout};
//!
//! // Create reusable config
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
//!     .progressive(true);
//!
//! // Encode from raw bytes
//! let jpeg = config.encode_one(1920, 1080, PixelLayout::Rgb8Srgb, &rgb_bytes)?;
//! ```
//!
//! # Streaming Encoding
//!
//! For large images or when you want to process rows incrementally:
//!
//! ```rust,ignore
//! use zenjpeg::{EncoderConfig, ChromaSubsampling, PixelLayout};
//! use enough::Unstoppable;
//!
//! let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
//! let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
//!
//! // Push rows (or use push_packed for all at once)
//! enc.push_packed(&rgb_bytes, Unstoppable)?;
//! let jpeg = enc.finish()?;
//! ```

// Many functions in this module are used conditionally via feature flags
// (test-utils, trellis, parallel) or from examples/benchmarks. The broad
// allow prevents noise from conditional compilation.
#![allow(dead_code)]

// Internal implementation modules (pub for internal crate re-exports)
mod blocks;
#[doc(hidden)]
pub mod chroma;
#[doc(hidden)]
pub mod dct;
pub(crate) mod layout;
mod progressive;
pub(crate) mod scan_optimize;
#[doc(hidden)]
pub mod scan_script;
mod serialize;

#[doc(hidden)]
pub mod config;
pub(crate) mod linear_lut;

// Archmage-based SIMD (token-based safe intrinsics)
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
pub mod mage_simd;

#[cfg(target_arch = "aarch64")]
#[doc(hidden)]
pub mod arm_simd;

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub mod wasm_simd;

/// Overshoot deringing for reducing ringing artifacts on white backgrounds.
///
/// This module implements the deringing algorithm pioneered by @kornel in mozjpeg.
/// It smooths hard edges to reduce visible ringing artifacts, especially on white
/// backgrounds. Enabled by default with no quality penalty for photographic content.
pub mod deringing;
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub mod parallel;
#[doc(hidden)]
pub mod streaming;
#[doc(hidden)]
pub mod streaming_builder;
#[doc(hidden)]
pub mod strip;

// v2 types moved to encode root (v2/mod.rs re-exports these for compatibility)
pub mod byte_encoders;
pub mod encoder_config;
pub mod encoder_types;
pub mod exif;
pub mod extras;
pub mod request;

/// Default quantization and zero-bias tables for customization.
///
/// This module exposes the internal default tables so users can modify them
/// rather than creating tables from scratch. Used with `tuning::EncodingTables`.
pub mod tables;

/// Encoding table tuning for optimization experiments.
///
/// This module provides fine-grained control over quantization and zero-bias
/// tables for researching better encoding parameters. See the README
/// "Table Optimization" section for research methodology.
pub mod tuning;

/// Trellis and hybrid quantization (mozjpeg-style rate-distortion optimization).
///
/// Consolidates all trellis/hybrid code into one deletable unit:
/// - AC/DC trellis DP algorithms
/// - [`TrellisConfig`](trellis::TrellisConfig) mozjpeg-compatible API
/// - [`HybridConfig`](trellis::HybridConfig) AQ-coupled trellis
#[cfg(feature = "trellis")]
pub mod trellis;

/// Expert configuration for external optimization (simulated annealing, etc.).
///
/// Flattens all quality/size-affecting parameters into a single struct with
/// no overlapping fields. See [`ExpertConfig`](search::ExpertConfig).
#[cfg(feature = "trellis")]
pub mod search;

// v2 is the primary public API (types re-exported below)
#[doc(hidden)]
pub mod v2;

// Re-export v2 types at encode:: level for cleaner imports
// (Now from encoder_types, encoder_config, byte_encoders - v2 re-exports for compatibility)
#[allow(unused_imports)] // Public API re-export
pub use blocks::HuffmanSymbolFrequencies;
pub(crate) use blocks::build_nonzero_mask;
#[allow(unused_imports)] // Public API re-exports
pub use byte_encoders::{BytesEncoder, Pixel, RgbEncoder, YCbCrPlanarEncoder};
#[allow(unused_imports)] // Public API re-export
pub use encoder_config::EncoderConfig;
#[cfg(feature = "trellis")]
#[allow(unused_imports)] // Public API re-export
pub use encoder_types::ExpertConfig;
#[cfg(feature = "parallel")]
#[allow(unused_imports)] // Public API re-export
pub use encoder_types::ParallelEncoding;
#[allow(unused_imports)] // Public API re-exports
pub use encoder_types::{
    ChromaSubsampling, ColorMode, DownsamplingMethod, Effort, HuffmanStrategy, OptimizationPreset,
    PixelLayout, ProgressiveScanMode, Quality, QuantTableConfig, QuantTableSource, ScanStrategy,
    XybSubsampling, YCbCrPlanes,
};
pub use enough::Stop;
#[allow(unused_imports)] // Public API re-exports
pub use exif::{Exif, ExifFields, Orientation};
#[allow(unused_imports)] // Public API re-exports
pub use extras::{EncoderSegment, EncoderSegments, MpfImage};
#[allow(unused_imports)] // Public API re-export
pub use request::EncodeRequest;
#[allow(unused_imports)] // Public API re-exports
pub use tables::presets::{MozjpegTables, QuantTablePreset};

use crate::error::Result;

// Internal config types
pub(crate) use config::ProgressiveScan;

use crate::foundation::alloc::{try_alloc_zeroed_f32, try_clone_slice};
use crate::foundation::consts::{DCT_BLOCK_SIZE, JPEG_ZIGZAG_ORDER};
use crate::types::{EdgePadding, EdgePaddingConfig};

// NOTE: The Encoder wrapper struct has been removed.
// All encoding methods are now implemented directly on ComputedConfig
// (see serialize.rs, blocks.rs, progressive.rs).
// StreamingEncoder stores ComputedConfig directly.

/// Converts coefficients from natural order to zigzag order, writing directly to destination.
/// Avoids allocation when writing to pre-allocated block arrays.
#[inline]
fn natural_to_zigzag_into(natural: &[i16; DCT_BLOCK_SIZE], dest: &mut [i16; DCT_BLOCK_SIZE]) {
    for i in 0..DCT_BLOCK_SIZE {
        dest[JPEG_ZIGZAG_ORDER[i] as usize] = natural[i];
    }
}

// ============================================================================
// Edge Padding Helpers
// ============================================================================

/// Compute the source coordinate for a padded pixel using the specified strategy.
///
/// For coordinates within the original image, returns the coordinate unchanged.
/// For coordinates beyond the edge, applies the padding strategy.
#[inline]
fn get_padded_coord(coord: usize, size: usize, strategy: EdgePadding) -> usize {
    if coord < size {
        return coord;
    }

    match strategy {
        EdgePadding::Replicate => size - 1,
        EdgePadding::Mirror => {
            // Reflect: coord beyond edge mirrors back
            // For coord = size + d, return size - 1 - d
            let d = coord - size;
            size.saturating_sub(1).saturating_sub(d)
        }
        EdgePadding::Wrap => coord % size,
    }
}

/// Pad a single-channel f32 plane to MCU-aligned dimensions.
///
/// Returns (padded_plane, padded_width, padded_height).
/// If no padding is needed, returns a clone of the input.
pub(crate) fn pad_plane_f32(
    plane: &[f32],
    width: usize,
    height: usize,
    mcu_size: usize,
    strategy: EdgePadding,
) -> Result<(Vec<f32>, usize, usize)> {
    let padded_w = (width + mcu_size - 1) / mcu_size * mcu_size;
    let padded_h = (height + mcu_size - 1) / mcu_size * mcu_size;

    // No padding needed
    if padded_w == width && padded_h == height {
        return Ok((
            try_clone_slice(plane, "pad_plane_f32 clone")?,
            width,
            height,
        ));
    }

    let mut out = try_alloc_zeroed_f32(padded_w * padded_h, "pad_plane_f32 output")?;

    for y in 0..padded_h {
        let src_y = get_padded_coord(y, height, strategy);
        for x in 0..padded_w {
            let src_x = get_padded_coord(x, width, strategy);
            out[y * padded_w + x] = plane[src_y * width + src_x];
        }
    }

    Ok((out, padded_w, padded_h))
}

/// Pad YCbCr f32 planes to MCU-aligned dimensions with per-channel strategies.
///
/// Y plane uses the luma strategy, Cb/Cr planes use the chroma strategy.
/// Handles subsampled chroma planes correctly (cb/cr may have different dimensions than y).
///
/// Returns ((y, cb, cr), padded_luma_w, padded_luma_h, padded_chroma_w, padded_chroma_h).
#[allow(clippy::type_complexity)]
#[allow(dead_code)]
pub(crate) fn pad_ycbcr_planes_subsampled(
    y: &[f32],
    width: usize,
    height: usize,
    cb: &[f32],
    cr: &[f32],
    c_width: usize,
    c_height: usize,
    mcu_size: usize,
    config: EdgePaddingConfig,
) -> Result<((Vec<f32>, Vec<f32>, Vec<f32>), usize, usize, usize, usize)> {
    // Pad luma to MCU-aligned dimensions
    let (y_padded, padded_w, padded_h) = pad_plane_f32(y, width, height, mcu_size, config.luma)?;

    // Chroma blocks are always 8x8. Padding chroma to multiples of 8 aligns with
    // the MCU grid because c_width = ceil(width / h_factor) and:
    // ceil(ceil(width / h_factor) / 8) * 8 == ceil(width / mcu_size) * (mcu_size / h_factor)
    let (cb_padded, padded_cw, padded_ch) = pad_plane_f32(cb, c_width, c_height, 8, config.chroma)?;
    let (cr_padded, _, _) = pad_plane_f32(cr, c_width, c_height, 8, config.chroma)?;

    Ok((
        (y_padded, cb_padded, cr_padded),
        padded_w,
        padded_h,
        padded_cw,
        padded_ch,
    ))
}

/// Pad grayscale f32 plane to MCU-aligned dimensions.
///
/// Returns (padded_plane, padded_width, padded_height).
#[allow(dead_code)] // Kept for future grayscale encoding support
pub(crate) fn pad_gray_plane(
    y: &[f32],
    width: usize,
    height: usize,
    mcu_size: usize,
    config: EdgePaddingConfig,
) -> Result<(Vec<f32>, usize, usize)> {
    pad_plane_f32(y, width, height, mcu_size, config.luma)
}

// Tests are in the old module (old/tests.rs)
