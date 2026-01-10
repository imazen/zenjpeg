//! Encoder configuration types.
//!
//! This module contains all configuration-related types for the JPEG encoder.

use crate::error::{Error, Result};
use crate::quant::Quality;
use crate::types::{ChromaConversion, JpegMode, PixelFormat, Subsampling};

// ============================================================================
// Internal Chroma Pipeline (undocumented, for benchmarking)
// ============================================================================
//
// These types allow external benchmarks to test different chroma conversion
// and downsampling strategies without committing to a public API.
//
// Use `Encoder::set_internal_pathway(u64)` to configure.
//
// Pathway encoding (u64):
//   Bits 0-7:   ColorConversionMethod (0=Auto, 1=IntrinsicF32, 2=YuvBalanced, 3=YuvProfessional)
//   Bits 8-15:  DownsamplingMethod (0=Auto, 1=None, 2=Box, 3=Sharp, 4=GammaAwareF32, 5=GammaAwareIterative)
//   Bits 16-23: Reserved
//   Bits 24-31: HuffmanMethod (0=JpegliCreateTree, 1=MozjpegClassic)
//   Bits 32-63: Reserved (must be 0)
//
// ============================================================================

/// Internal color conversion method (not public API).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum ColorConversionMethod {
    /// Auto-select based on other settings
    #[default]
    Auto = 0,
    /// Our f32 BT.601 conversion (highest precision)
    IntrinsicF32 = 1,
    /// yuv crate with Balanced precision (good SIMD performance)
    YuvBalanced = 2,
    /// yuv crate with Professional precision (requires feature flag)
    YuvProfessional = 3,
}

/// Internal downsampling method (not public API).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum DownsamplingMethod {
    /// Auto-select based on subsampling mode
    #[default]
    Auto = 0,
    /// No downsampling (4:4:4 only)
    None = 1,
    /// Simple box filter (2x2, 2x1, or 1x2 averaging)
    Box = 2,
    /// yuv crate Sharp YUV (gamma-aware bilinear)
    Sharp = 3,
    /// Our f32 gamma-aware single-pass (linear space averaging)
    GammaAwareF32 = 4,
    /// Our f32 gamma-aware iterative (Sharp YUV style optimization)
    GammaAwareIterative = 5,
}

/// Internal pipeline configuration (not public API).
///
/// Controls low-level encoder behavior for benchmarking and testing.
/// Includes chroma conversion, Huffman optimization, and other internal settings.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct InternalPipeline {
    pub(crate) color_conversion: ColorConversionMethod,
    pub(crate) downsampling: DownsamplingMethod,
    pub(crate) huffman_method: crate::types::HuffmanMethod,
}

impl InternalPipeline {
    /// Decode from u64 pathway value.
    pub(crate) fn from_u64(value: u64) -> Result<Self> {
        // Check reserved bits are zero
        if value & 0xFFFF_FFFF_0000_0000 != 0 {
            return Err(Error::InvalidColorFormat {
                reason: "internal pathway: reserved bits must be zero",
            });
        }

        let color_byte = (value & 0xFF) as u8;
        let downsample_byte = ((value >> 8) & 0xFF) as u8;
        // Bits 16-23 reserved (was smoothing_factor, now unused)
        let huffman_byte = ((value >> 24) & 0xFF) as u8;

        let color_conversion = match color_byte {
            0 => ColorConversionMethod::Auto,
            1 => ColorConversionMethod::IntrinsicF32,
            2 => ColorConversionMethod::YuvBalanced,
            3 => ColorConversionMethod::YuvProfessional,
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "internal pathway: invalid color conversion method (0-3)",
                })
            }
        };

        let downsampling = match downsample_byte {
            0 => DownsamplingMethod::Auto,
            1 => DownsamplingMethod::None,
            2 => DownsamplingMethod::Box,
            3 => DownsamplingMethod::Sharp,
            4 => DownsamplingMethod::GammaAwareF32,
            5 => DownsamplingMethod::GammaAwareIterative,
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "internal pathway: invalid downsampling method (0-5)",
                })
            }
        };

        let huffman_method = match huffman_byte {
            0 => crate::types::HuffmanMethod::JpegliCreateTree,
            1 => crate::types::HuffmanMethod::MozjpegClassic,
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "internal pathway: invalid Huffman method (0-1)",
                })
            }
        };

        Ok(Self {
            color_conversion,
            downsampling,
            huffman_method,
        })
    }

    /// Encode to u64 pathway value.
    #[allow(dead_code)]
    pub(crate) fn to_u64(self) -> u64 {
        let huffman_value = match self.huffman_method {
            crate::types::HuffmanMethod::JpegliCreateTree => 0,
            crate::types::HuffmanMethod::MozjpegClassic => 1,
        };
        (self.color_conversion as u64)
            | ((self.downsampling as u64) << 8)
            | ((huffman_value as u64) << 24)
    }

    /// Validate pipeline against encoder config.
    pub(crate) fn validate(&self, subsampling: Subsampling) -> Result<()> {
        // Check downsampling method compatibility
        match self.downsampling {
            DownsamplingMethod::None => {
                if subsampling != Subsampling::S444 {
                    return Err(Error::InvalidColorFormat {
                        reason: "internal pathway: None downsampling only valid for 4:4:4",
                    });
                }
            }
            DownsamplingMethod::Sharp => {
                // Sharp YUV only supports 4:2:0 and 4:2:2
                if !matches!(subsampling, Subsampling::S420 | Subsampling::S422) {
                    return Err(Error::InvalidColorFormat {
                        reason: "internal pathway: Sharp only supports 4:2:0 and 4:2:2",
                    });
                }
            }
            DownsamplingMethod::GammaAwareF32 | DownsamplingMethod::GammaAwareIterative => {
                // Gamma-aware methods only make sense with subsampling (not 4:4:4)
                if subsampling == Subsampling::S444 {
                    return Err(Error::InvalidColorFormat {
                        reason: "internal pathway: Gamma-aware downsampling not valid for 4:4:4 (no downsampling needed)",
                    });
                }
            }
            _ => {}
        }

        // Check color conversion compatibility
        match self.color_conversion {
            ColorConversionMethod::YuvProfessional => {
                // Would need feature flag check here
                // For now, treat as not available
                return Err(Error::UnsupportedFeature {
                    feature: "internal pathway: YuvProfessional requires professional_mode feature",
                });
            }
            ColorConversionMethod::YuvBalanced => {
                // yuv crate doesn't support 4:4:0
                if subsampling == Subsampling::S440 {
                    return Err(Error::InvalidColorFormat {
                        reason: "internal pathway: yuv crate doesn't support 4:4:0",
                    });
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Resolve Auto values to concrete methods based on config.
    #[allow(dead_code)]
    pub(crate) fn resolve(mut self, subsampling: Subsampling) -> Self {
        // Resolve color conversion
        if self.color_conversion == ColorConversionMethod::Auto {
            self.color_conversion = ColorConversionMethod::IntrinsicF32;
        }

        // Resolve downsampling
        if self.downsampling == DownsamplingMethod::Auto {
            self.downsampling = match subsampling {
                Subsampling::S444 => DownsamplingMethod::None,
                Subsampling::S420 | Subsampling::S422 => DownsamplingMethod::Sharp,
                Subsampling::S440 => DownsamplingMethod::Box, // Sharp doesn't support 4:4:0
            };
        }

        self
    }
}

// ============================================================================
// Pathway Constants (for benchmarking)
// ============================================================================

/// Internal pathway constants for benchmarking.
/// Use with `Encoder::set_internal_pathway()`.
#[doc(hidden)]
pub mod internal_pathway {
    // Color conversion methods (bits 0-7)
    pub const COLOR_AUTO: u64 = 0;
    pub const COLOR_INTRINSIC_F32: u64 = 1;
    pub const COLOR_YUV_BALANCED: u64 = 2;
    pub const COLOR_YUV_PROFESSIONAL: u64 = 3;

    // Downsampling methods (bits 8-15)
    pub const DOWNSAMPLE_AUTO: u64 = 0 << 8;
    pub const DOWNSAMPLE_NONE: u64 = 1 << 8;
    pub const DOWNSAMPLE_BOX: u64 = 2 << 8;
    pub const DOWNSAMPLE_SHARP: u64 = 3 << 8;
    pub const DOWNSAMPLE_GAMMA_AWARE_F32: u64 = 4 << 8;
    pub const DOWNSAMPLE_GAMMA_AWARE_ITERATIVE: u64 = 5 << 8;

    // Pre-defined pipeline combinations for common benchmarks
    /// f32 color conversion, no downsampling (4:4:4 only)
    pub const P_F32_NONE: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_NONE;
    /// f32 color conversion, box filter downsampling
    pub const P_F32_BOX: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX;
    /// yuv crate balanced, box filter (Fast path)
    pub const P_YUV_BOX: u64 = COLOR_YUV_BALANCED | DOWNSAMPLE_BOX;
    /// yuv crate balanced, sharp downsampling (Sharp path)
    pub const P_YUV_SHARP: u64 = COLOR_YUV_BALANCED | DOWNSAMPLE_SHARP;
    /// f32 color + gamma-aware single-pass downsampling
    pub const P_F32_GAMMA_AWARE: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_F32;
    /// f32 color + gamma-aware iterative downsampling (Sharp YUV style)
    pub const P_F32_GAMMA_AWARE_ITERATIVE: u64 =
        COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_ITERATIVE;
}

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
    /// Use XYB color space
    pub use_xyb: bool,
    /// Restart interval (0 = disabled)
    pub restart_interval: u16,
    /// Use optimized Huffman tables
    pub optimize_huffman: bool,
    /// Chroma conversion method for RGB to YCbCr.
    ///
    /// Controls how chroma planes are computed:
    /// - `Intrinsic`: Our f32 conversion (best for 4:4:4)
    /// - `Fast`: yuv crate SIMD path (simple box filter)
    /// - `Sharp`: yuv crate Sharp YUV (gamma-aware, best edges)
    /// - `Auto`: Intrinsic (matches C++ jpegli default)
    pub chroma_conversion: ChromaConversion,
    /// Hybrid quantization configuration (jpegli AQ + mozjpeg trellis)
    /// Requires the `experimental-hybrid-trellis` feature
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub hybrid_config: crate::hybrid::config::HybridConfig,
    /// Custom AQ map (optional). If None, computed automatically.
    /// Allows pre-scaling the AQ map for size control.
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub custom_aq_map: Option<crate::adaptive_quant::AQStrengthMap>,

    // Internal pipeline override (not public API, for benchmarking)
    // When Some, overrides chroma_conversion, smoothing_factor, and huffman_method
    #[doc(hidden)]
    pub(crate) internal_pipeline: Option<InternalPipeline>,

    /// Custom quantization matrices (escape hatch for experimentation).
    /// Not part of public API - use Encoder::custom_quant_matrices() method.
    #[doc(hidden)]
    pub(crate) custom_quant_matrices: Option<crate::quant::CustomQuantMatrices>,

    /// Force full-plane encoding even for large images (benchmarking only).
    /// When true, disables auto-dispatch to strip-based encoding.
    #[doc(hidden)]
    pub(crate) force_full_plane: bool,
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
            // Huffman optimization enabled by default (pseudo-symbol 256 approach ensures Kraft sum < 2^16)
            optimize_huffman: true,
            // Auto selects Intrinsic to match C++ jpegli
            chroma_conversion: ChromaConversion::Auto,
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_config: crate::hybrid::config::HybridConfig::disabled(),
            #[cfg(feature = "experimental-hybrid-trellis")]
            custom_aq_map: None,
            internal_pipeline: None,
            custom_quant_matrices: None,
            force_full_plane: false,
        }
    }
}
