//! Encoder configuration types.
//!
//! This module contains all configuration-related types for the JPEG encoder.

use crate::quant::Quality;
use crate::types::{ChromaDownsampling, EdgePaddingConfig, JpegMode, PixelFormat, Subsampling};

// ============================================================================
// Progressive Scan Configuration
// ============================================================================

/// Progressive scan parameters.
#[derive(Debug, Clone)]
pub(crate) struct ProgressiveScan {
    /// Component indices in this scan (0=Y, 1=Cb, 2=Cr)
    pub(crate) components: Vec<u8>,
    /// Spectral selection start (0=DC, 1-63=AC)
    pub(crate) ss: u8,
    /// Spectral selection end (0-63)
    pub(crate) se: u8,
    /// Successive approximation high bit (previous pass)
    pub(crate) ah: u8,
    /// Successive approximation low bit (current pass)
    pub(crate) al: u8,
}

// ============================================================================
// Encoder Configuration
// ============================================================================

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Input pixel format
    pub pixel_format: PixelFormat,
    /// Quality setting
    pub quality: Quality,
    /// Encoding mode
    pub mode: JpegMode,
    /// Chroma subsampling
    pub subsampling: Subsampling,
    /// Use XYB color space (uses legacy encoder path)
    pub use_xyb: bool,
    /// Restart interval (0 = disabled)
    pub restart_interval: u16,
    /// Enable parallel encoding (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    pub parallel: bool,
    /// Use optimized Huffman tables
    pub optimize_huffman: bool,
    /// Chroma downsampling method for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    ///
    /// Controls how chroma planes are downsampled:
    /// - `Box`: Simple box filter (default, matches C++ jpegli)
    /// - `GammaAware`: Gamma-aware averaging (better edges)
    /// - `GammaAwareIterative`: Sharp YUV-style optimization (best quality)
    ///
    /// Has no effect for 4:4:4 (no downsampling needed).
    pub chroma_downsampling: ChromaDownsampling,
    /// Hybrid quantization configuration (jpegli AQ + mozjpeg trellis)
    /// Requires the `experimental-hybrid-trellis` feature
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub hybrid_config: crate::hybrid::config::HybridConfig,
    /// Custom AQ map (optional). If None, computed automatically.
    /// Allows pre-scaling the AQ map for size control.
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub custom_aq_map: Option<crate::quant::aq::AQStrengthMap>,

    /// Custom quantization matrices (escape hatch for experimentation).
    /// Not part of public API - use Encoder::custom_quant_matrices() method.
    #[doc(hidden)]
    pub(crate) custom_quant_matrices: Option<crate::quant::CustomQuantMatrices>,

    // EncodingBackend removed - strip-based encoding is now the only backend
    /// Edge padding strategy for partial MCU blocks.
    ///
    /// Controls how edge pixels are padded when image dimensions are not
    /// multiples of the MCU size. Different strategies for luma and chroma
    /// can be specified to optimize for both gradient preservation (luma)
    /// and safe upsampling (chroma).
    pub edge_padding: EdgePaddingConfig,

    /// Original image width before MCU padding (for JFIF header).
    ///
    /// When edge padding expands the image to MCU-aligned dimensions,
    /// this stores the original width to write to the JFIF header.
    /// Decoders will crop to these dimensions after decoding.
    pub(crate) original_width: Option<u32>,

    /// Original image height before MCU padding (for JFIF header).
    pub(crate) original_height: Option<u32>,

    /// Allow 16-bit quantization tables for better low-quality precision.
    ///
    /// When `true` (default), quantization values can go up to 32767, using
    /// 16-bit DQT tables and extended sequential JPEGs (SOF1) when needed.
    /// This provides better precision at very low quality settings.
    ///
    /// When `false`, quantization values are clamped to 255 (8-bit DQT),
    /// producing baseline-compatible JPEGs (SOF0) that work with all decoders,
    /// but may lose precision at very low quality settings.
    ///
    /// Note: Most modern decoders support 16-bit quant tables. Only disable
    /// this for compatibility with very old or limited JPEG decoders.
    pub allow_16bit_quant_tables: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixel_format: PixelFormat::Rgb,
            quality: Quality::default(),
            mode: JpegMode::Baseline,
            // Use 4:4:4 - this is what the encoder actually supports currently
            subsampling: Subsampling::S444,
            use_xyb: false,
            restart_interval: 0,
            #[cfg(feature = "parallel")]
            parallel: false,
            // Huffman optimization enabled by default (pseudo-symbol 256 approach ensures Kraft sum < 2^16)
            optimize_huffman: true,
            // Box filter matches C++ jpegli default
            chroma_downsampling: ChromaDownsampling::Box,
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_config: crate::hybrid::config::HybridConfig::disabled(),
            #[cfg(feature = "experimental-hybrid-trellis")]
            custom_aq_map: None,
            custom_quant_matrices: None,
            edge_padding: EdgePaddingConfig::default(),
            original_width: None,
            original_height: None,
            // Allow 16-bit quant tables by default (matches C++ jpegli behavior)
            // Set to false only for compatibility with very old decoders
            allow_16bit_quant_tables: true,
        }
    }
}
