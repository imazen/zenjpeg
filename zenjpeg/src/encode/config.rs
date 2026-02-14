//! Encoder configuration types.
//!
//! This module contains all configuration-related types for the JPEG encoder.

#![allow(dead_code)]

use super::encoder_types::DownsamplingMethod;
use super::encoder_types::HuffmanStrategy;
use super::encoder_types::Quality;
use super::encoder_types::ScanStrategy;
use crate::types::{EdgePaddingConfig, JpegMode, PixelFormat, Subsampling};

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
// Computed Encoder Configuration
// ============================================================================

/// Computed encoder configuration with dimensions.
///
/// This is the internal configuration used during JPEG serialization.
/// It combines dimension-independent settings from [`crate::encode::EncoderConfig`]
/// with image dimensions and pixel format.
///
/// Created internally by the streaming encoder or via `EncoderConfig::compute()`.
#[derive(Debug, Clone)]
pub struct ComputedConfig {
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
    /// Huffman table strategy (Optimize, Fixed, or Custom).
    pub(crate) huffman: HuffmanStrategy,
    /// Chroma downsampling method for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    ///
    /// Controls how chroma planes are downsampled:
    /// - `Box`: Simple box filter (default, matches C++ jpegli)
    /// - `GammaAware`: Gamma-aware averaging (better edges)
    /// - `GammaAwareIterative`: Sharp YUV-style optimization (best quality)
    ///
    /// Has no effect for 4:4:4 (no downsampling needed).
    pub chroma_downsampling: DownsamplingMethod,
    /// Hybrid quantization configuration (jpegli AQ + mozjpeg trellis)
    #[cfg(feature = "trellis")]
    pub hybrid_config: super::trellis::HybridConfig,
    /// Custom AQ map (optional). If None, computed automatically.
    /// Allows pre-scaling the AQ map for size control.
    pub custom_aq_map: Option<crate::quant::aq::AQStrengthMap>,

    /// Custom encoding tables (escape hatch for experimentation).
    /// Not part of public API.
    #[doc(hidden)]
    pub(crate) encoding_tables: Option<Box<crate::encode::tuning::EncodingTables>>,

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

    /// Progressive scan script strategy.
    ///
    /// Controls how scans are structured for progressive JPEGs:
    /// - `Default`: jpegli-style (freq split at 2/3, SA for all)
    /// - `Search`: mozjpeg-style optimize_scans (64 candidates, picks smallest)
    /// - `Mozjpeg`: mozjpeg default (freq split at 8/9, no chroma SA)
    pub scan_strategy: ScanStrategy,

    /// Use separate quantization tables for Cb and Cr (3 tables total).
    ///
    /// When `true` (default), uses 3 quantization tables:
    /// - Table 0: Y (luma)
    /// - Table 1: Cb (blue chroma)
    /// - Table 2: Cr (red chroma)
    ///
    /// When `false`, uses 2 quantization tables:
    /// - Table 0: Y (luma)
    /// - Table 1: Cb and Cr (shared chroma)
    ///
    /// The 3-table mode matches C++ jpegli's `jpegli_set_distance()` behavior.
    /// The 2-table mode matches C++ jpegli's `jpeg_set_quality()` behavior.
    pub separate_chroma_tables: bool,

    /// Trellis quantization configuration (mozjpeg-compatible API).
    ///
    /// When set, enables trellis quantization for rate-distortion optimization.
    /// This is the mozjpeg-compatible API. For hybrid AQ+trellis mode, use
    /// `hybrid_config` instead.
    #[cfg(feature = "trellis")]
    pub trellis: Option<super::trellis::TrellisConfig>,
}

impl ComputedConfig {
    /// MCU columns for this image's dimensions and subsampling.
    pub(crate) fn mcu_cols(&self) -> u32 {
        let h_samp = match self.subsampling {
            Subsampling::S444 | Subsampling::S440 => 1u32,
            Subsampling::S422 | Subsampling::S420 => 2,
        };
        let mcu_w = h_samp * 8;
        (self.width + mcu_w - 1) / mcu_w
    }

    /// Round a restart interval down to the nearest MCU row boundary.
    ///
    /// Non-row-aligned restart intervals break the fused chroma upsample +
    /// color conversion decode path, which processes complete MCU rows.
    /// Returns 0 if interval is less than one row.
    pub(crate) fn align_restart_to_row(&self, interval: u16) -> u16 {
        let mcu_cols = self.mcu_cols() as u16;
        if mcu_cols == 0 {
            return 0;
        }
        (interval / mcu_cols) * mcu_cols
    }
}

impl Default for ComputedConfig {
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
            huffman: HuffmanStrategy::Optimize,
            // Box filter matches C++ jpegli default
            chroma_downsampling: DownsamplingMethod::Box,
            #[cfg(feature = "trellis")]
            hybrid_config: super::trellis::HybridConfig::disabled(),
            custom_aq_map: None,
            encoding_tables: None,
            edge_padding: EdgePaddingConfig::default(),
            original_width: None,
            original_height: None,
            // Allow 16-bit quant tables by default (matches C++ jpegli behavior)
            // Set to false only for compatibility with very old decoders
            allow_16bit_quant_tables: false,
            scan_strategy: ScanStrategy::Default,
            // Use 3 tables by default (matches jpegli_set_distance)
            separate_chroma_tables: true,
            #[cfg(feature = "trellis")]
            trellis: None,
        }
    }
}
