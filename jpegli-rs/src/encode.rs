//! JPEG encoder implementation.
//!
//! This module provides the main encoder interface for creating JPEG images.

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
//   Bits 8-15:  DownsamplingMethod (0=Auto, 1=None, 2=Box, 3=BoxSmoothed, 4=Sharp, 5=GammaAwareF32)
//   Bits 16-23: Smoothing factor (0-100, only for BoxSmoothed)
//   Bits 24-63: Reserved (must be 0)
//
// ============================================================================

/// Internal color conversion method (not public API).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
enum ColorConversionMethod {
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
enum DownsamplingMethod {
    /// Auto-select based on subsampling mode
    #[default]
    Auto = 0,
    /// No downsampling (4:4:4 only)
    None = 1,
    /// Simple box filter (2x2, 2x1, or 1x2 averaging)
    Box = 2,
    /// Box filter with pre-smoothing (3x3 blur before box)
    BoxSmoothed = 3,
    /// yuv crate Sharp YUV (gamma-aware bilinear)
    Sharp = 4,
    /// Our f32 gamma-aware bilinear (TODO: not yet implemented)
    GammaAwareF32 = 5,
}

/// Internal chroma pipeline configuration (not public API).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ChromaPipeline {
    color_conversion: ColorConversionMethod,
    downsampling: DownsamplingMethod,
    smoothing_factor: u8,
}

impl ChromaPipeline {
    /// Decode from u64 pathway value.
    fn from_u64(value: u64) -> Result<Self> {
        // Check reserved bits are zero
        if value & 0xFFFF_FFFF_FF00_0000 != 0 {
            return Err(Error::InvalidColorFormat {
                reason: "internal pathway: reserved bits must be zero",
            });
        }

        let color_byte = (value & 0xFF) as u8;
        let downsample_byte = ((value >> 8) & 0xFF) as u8;
        let smoothing = ((value >> 16) & 0xFF) as u8;

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
            3 => DownsamplingMethod::BoxSmoothed,
            4 => DownsamplingMethod::Sharp,
            5 => DownsamplingMethod::GammaAwareF32,
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "internal pathway: invalid downsampling method (0-5)",
                })
            }
        };

        if smoothing > 100 {
            return Err(Error::InvalidColorFormat {
                reason: "internal pathway: smoothing factor must be 0-100",
            });
        }

        Ok(Self {
            color_conversion,
            downsampling,
            smoothing_factor: smoothing,
        })
    }

    /// Encode to u64 pathway value.
    #[allow(dead_code)]
    fn to_u64(self) -> u64 {
        (self.color_conversion as u64)
            | ((self.downsampling as u64) << 8)
            | ((self.smoothing_factor as u64) << 16)
    }

    /// Validate pipeline against encoder config.
    fn validate(&self, subsampling: Subsampling) -> Result<()> {
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
            DownsamplingMethod::GammaAwareF32 => {
                // GammaAwareF32 only makes sense with subsampling (not 4:4:4)
                if subsampling == Subsampling::S444 {
                    return Err(Error::InvalidColorFormat {
                        reason: "internal pathway: GammaAwareF32 not valid for 4:4:4 (no downsampling needed)",
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

        // Smoothing only makes sense with BoxSmoothed
        if self.smoothing_factor > 0 && self.downsampling != DownsamplingMethod::BoxSmoothed {
            return Err(Error::InvalidColorFormat {
                reason: "internal pathway: smoothing_factor only valid with BoxSmoothed",
            });
        }

        Ok(())
    }

    /// Resolve Auto values to concrete methods based on config.
    #[allow(dead_code)]
    fn resolve(mut self, subsampling: Subsampling) -> Self {
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
    pub const DOWNSAMPLE_BOX_SMOOTHED: u64 = 3 << 8;
    pub const DOWNSAMPLE_SHARP: u64 = 4 << 8;
    pub const DOWNSAMPLE_GAMMA_AWARE_F32: u64 = 5 << 8;

    /// Create a pathway with smoothing factor (bits 16-23).
    /// Only valid with `DOWNSAMPLE_BOX_SMOOTHED`.
    #[inline]
    pub const fn with_smoothing(pathway: u64, factor: u8) -> u64 {
        pathway | ((factor as u64) << 16)
    }

    // Pre-defined pipeline combinations for common benchmarks
    /// f32 color conversion, no downsampling (4:4:4 only)
    pub const P_F32_NONE: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_NONE;
    /// f32 color conversion, box filter downsampling
    pub const P_F32_BOX: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX;
    /// f32 color conversion, box filter with smoothing=50
    pub const P_F32_BOX_SMOOTH50: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX_SMOOTHED | (50 << 16);
    /// yuv crate balanced, box filter (Fast path)
    pub const P_YUV_BOX: u64 = COLOR_YUV_BALANCED | DOWNSAMPLE_BOX;
    /// yuv crate balanced, sharp downsampling (Sharp path)
    pub const P_YUV_SHARP: u64 = COLOR_YUV_BALANCED | DOWNSAMPLE_SHARP;
    /// f32 color + gamma-aware downsampling (TODO: not yet implemented)
    pub const P_F32_GAMMA_AWARE: u64 = COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_F32;
}

// ============================================================================
// Public API
// ============================================================================

use crate::adaptive_quant::compute_aq_strength_map;
use crate::alloc::{
    checked_size_2d, try_alloc_filled, try_alloc_zeroed_f32, validate_dimensions,
    DEFAULT_MAX_PIXELS,
};
use crate::color;
use crate::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, ICC_PROFILE_SIGNATURE, JPEG_NATURAL_ORDER, JPEG_ZIGZAG_ORDER,
    MARKER_APP14, MARKER_APP2, MARKER_DHT, MARKER_DQT, MARKER_DRI, MARKER_EOI, MARKER_SOF0,
    MARKER_SOF2, MARKER_SOI, MARKER_SOS, MAX_ICC_BYTES_PER_MARKER, XYB_ICC_PROFILE,
};
use crate::dct::forward_dct_8x8;
use crate::entropy::{self, EntropyEncoder};
use crate::error::{Error, Result};
use crate::huffman::HuffmanEncodeTable;
use crate::huffman_opt::{
    FrequencyCounter, OptimizedHuffmanTables, OptimizedTable, ProgressiveTokenBuffer,
};
use crate::quant::{self, Quality, QuantTable, ZeroBiasParams};
use crate::types::{ChromaConversion, ColorSpace, JpegMode, PixelFormat, Subsampling};
use crate::xyb::srgb_to_scaled_xyb;

#[cfg(feature = "experimental-hybrid-trellis")]
use crate::hybrid::{hybrid_quantize_block, StandardHuffmanTables};

use yuv::{
    rgb_to_sharp_yuv420, rgb_to_sharp_yuv422, rgb_to_yuv420, rgb_to_yuv422, SharpYuvGammaTransfer,
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

/// Progressive scan parameters.
#[derive(Debug, Clone)]
struct ProgressiveScan {
    /// Component indices in this scan (0=Y, 1=Cb, 2=Cr)
    components: Vec<u8>,
    /// Spectral selection start (0=DC, 1-63=AC)
    ss: u8,
    /// Spectral selection end (0-63)
    se: u8,
    /// Successive approximation high bit (previous pass)
    ah: u8,
    /// Successive approximation low bit (current pass)
    al: u8,
}

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
    /// Input smoothing factor (0-100, 0 = disabled).
    /// Applies a 3x3 weighted blur to chroma planes before downsampling
    /// to reduce aliasing artifacts. Matches libjpeg/jpegli smoothing_factor.
    /// Only used with `ChromaConversion::Intrinsic`.
    pub smoothing_factor: u8,
    /// Chroma conversion method for RGB to YCbCr.
    ///
    /// Controls how chroma planes are computed:
    /// - `Intrinsic`: Our f32 conversion (best for 4:4:4)
    /// - `Fast`: yuv crate SIMD path (simple box filter)
    /// - `Sharp`: yuv crate Sharp YUV (gamma-aware, best edges)
    /// - `Auto`: Sharp for 4:2:0/4:2:2/4:4:0, Intrinsic for 4:4:4
    pub chroma_conversion: ChromaConversion,
    /// Hybrid quantization configuration (jpegli AQ + mozjpeg trellis)
    /// Requires the `experimental-hybrid-trellis` feature
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub hybrid_config: crate::hybrid_config::HybridConfig,
    /// Custom AQ map (optional). If None, computed automatically.
    /// Allows pre-scaling the AQ map for size control.
    #[cfg(feature = "experimental-hybrid-trellis")]
    pub custom_aq_map: Option<crate::adaptive_quant::AQStrengthMap>,

    // Internal pipeline override (not public API, for benchmarking)
    // When Some, overrides chroma_conversion and smoothing_factor
    #[doc(hidden)]
    pub(crate) internal_pipeline: Option<ChromaPipeline>,
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
            // Match C++ jpegli default: optimize_coding = true
            optimize_huffman: true,
            // Match C++ jpegli default: smoothing_factor = 0 (disabled)
            smoothing_factor: 0,
            // Auto selects Sharp for subsampled, Intrinsic for 4:4:4
            chroma_conversion: ChromaConversion::Auto,
            #[cfg(feature = "experimental-hybrid-trellis")]
            hybrid_config: crate::hybrid_config::HybridConfig::disabled(),
            #[cfg(feature = "experimental-hybrid-trellis")]
            custom_aq_map: None,
            internal_pipeline: None,
        }
    }
}

/// Quantization context for hybrid trellis mode.
///
/// This struct holds pre-built Huffman tables and hybrid config for use
/// during hybrid quantization (jpegli AQ + mozjpeg trellis).
#[cfg(feature = "experimental-hybrid-trellis")]
struct HybridQuantContext {
    huff_tables: StandardHuffmanTables,
    config: crate::hybrid_config::HybridConfig,
}

#[cfg(feature = "experimental-hybrid-trellis")]
impl HybridQuantContext {
    /// Creates a new hybrid quantization context with the given config.
    fn new(config: crate::hybrid_config::HybridConfig) -> Self {
        Self {
            huff_tables: StandardHuffmanTables::new(),
            config,
        }
    }

    /// Quantize a block using hybrid AQ + trellis.
    ///
    /// # Arguments
    /// * `dct_coeffs` - DCT coefficients
    /// * `quant` - Quantization table
    /// * `aq_strength` - Per-block AQ strength
    /// * `dampen` - Quality-based AQ dampen factor (0-1)
    /// * `is_luma` - True for Y component, false for Cb/Cr
    fn quantize_block(
        &self,
        dct_coeffs: &[f32; DCT_BLOCK_SIZE],
        quant: &[u16; DCT_BLOCK_SIZE],
        aq_strength: f32,
        dampen: f32,
        is_luma: bool,
    ) -> [i16; DCT_BLOCK_SIZE] {
        let ac_table = if is_luma {
            &self.huff_tables.luma_ac
        } else {
            &self.huff_tables.chroma_ac
        };

        // Generate per-block trellis config based on AQ and hybrid settings
        let trellis_config = self.config.to_trellis_config(aq_strength, dampen, !is_luma);

        hybrid_quantize_block(dct_coeffs, quant, aq_strength, ac_table, &trellis_config)
    }
}

/// JPEG encoder.
pub struct Encoder {
    config: EncoderConfig,
}

impl Encoder {
    /// Creates a new encoder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
        }
    }

    /// Creates an encoder from configuration.
    #[must_use]
    pub fn from_config(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Sets the image width.
    #[must_use]
    pub fn width(mut self, width: u32) -> Self {
        self.config.width = width;
        self
    }

    /// Sets the image height.
    #[must_use]
    pub fn height(mut self, height: u32) -> Self {
        self.config.height = height;
        self
    }

    /// Sets the pixel format.
    #[must_use]
    pub fn pixel_format(mut self, format: PixelFormat) -> Self {
        self.config.pixel_format = format;
        self
    }

    /// Sets the quality using jpegli's native quality scale.
    ///
    /// Use `Quality::from_quality(90.0)` for traditional JPEG quality (1-100)
    /// or `Quality::from_distance(1.0)` for butteraugli distance.
    #[must_use]
    pub fn jpegli_quality(mut self, quality: Quality) -> Self {
        self.config.quality = quality;
        self
    }

    /// Sets the quality to match another encoder's visual quality.
    ///
    /// This converts quality settings from other encoders (like mozjpeg) to
    /// equivalent jpegli quality values that produce similar visual results.
    ///
    /// # Example
    ///
    /// ```
    /// use jpegli::{Encoder, QualityConversion, QualityComparisonMetric, Subsampling};
    ///
    /// // Match mozjpeg Q85 visual quality
    /// let conversion = QualityConversion::mozjpeg_equivalent(
    ///     85,
    ///     Subsampling::S444,
    ///     QualityComparisonMetric::Dssim,
    /// );
    ///
    /// let encoder = Encoder::new()
    ///     .width(800)
    ///     .height(600)
    ///     .equivalent_quality(conversion);
    /// ```
    #[must_use]
    pub fn equivalent_quality(
        mut self,
        conversion: crate::quality_conversion::QualityConversion,
    ) -> Self {
        self.config.quality = conversion.to_jpegli_quality();
        self
    }

    /// Sets the quality.
    ///
    /// **Deprecated:** Use `jpegli_quality()` for explicit jpegli quality, or
    /// `equivalent_quality()` to match other encoders like mozjpeg.
    #[must_use]
    #[deprecated(
        since = "0.4.0",
        note = "Use jpegli_quality() or equivalent_quality() instead"
    )]
    pub fn quality(mut self, quality: Quality) -> Self {
        self.config.quality = quality;
        self
    }

    /// Sets the encoding mode.
    #[must_use]
    pub fn mode(mut self, mode: JpegMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Sets chroma subsampling.
    #[must_use]
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.config.subsampling = subsampling;
        self
    }

    /// Enables XYB-optimized encoding mode.
    ///
    /// XYB mode encodes images using the perceptually-optimized XYB color space
    /// from JPEG XL. This provides better quality at the same file size compared
    /// to standard YCbCr encoding.
    ///
    /// The implementation includes:
    /// 1. Full sRGB → linear RGB → XYB color space conversion
    /// 2. XYB value scaling for optimal quantization
    /// 3. Embedded ICC profile for decoder color interpretation
    /// 4. Blue channel subsampling (R:2×2, G:2×2, B:1×1)
    /// 5. Separate XYB-optimized quant tables per component
    ///
    /// The ICC profile allows any ICC-aware decoder (including djpegli, ImageMagick,
    /// and most image viewers) to correctly interpret the XYB values back to sRGB.
    ///
    /// Note: Without ICC profile support in the decoder, images will display with
    /// incorrect colors. Use standard YCbCr mode for maximum compatibility.
    #[must_use]
    pub fn use_xyb(mut self, enable: bool) -> Self {
        self.config.use_xyb = enable;
        self
    }

    /// Sets the restart interval.
    #[must_use]
    pub fn restart_interval(mut self, interval: u16) -> Self {
        self.config.restart_interval = interval;
        self
    }

    /// Enables optimized Huffman tables.
    #[must_use]
    pub fn optimize_huffman(mut self, enable: bool) -> Self {
        self.config.optimize_huffman = enable;
        self
    }

    /// Sets the input smoothing factor (0-100).
    ///
    /// When non-zero, applies a 3x3 weighted blur to chroma planes before
    /// downsampling to reduce aliasing artifacts. Higher values = more blur.
    ///
    /// This matches libjpeg/jpegli's `smoothing_factor` parameter.
    /// Default is 0 (disabled), which is also jpegli's default.
    ///
    /// **Important**: Only works with [`ChromaConversion::Intrinsic`].
    /// The yuv crate paths (Fast, Sharp) perform conversion + downsampling
    /// in a single pass, so there's no intermediate chroma plane to blur.
    ///
    /// Only affects chroma subsampling modes (4:2:0, 4:2:2, 4:4:0).
    /// Has no effect on 4:4:4 mode since no downsampling occurs.
    #[must_use]
    pub fn smoothing_factor(mut self, factor: u8) -> Self {
        self.config.smoothing_factor = factor.min(100);
        self
    }

    /// Set chroma conversion method.
    ///
    /// Controls how RGB is converted to YCbCr chroma planes:
    /// - [`ChromaConversion::Intrinsic`]: Our f32 conversion with box filter
    ///   downsampling. Supports `smoothing_factor` for pre-blur.
    /// - [`ChromaConversion::Fast`]: yuv crate SIMD path with box filter.
    ///   Fast but may have color bleeding on edges.
    /// - [`ChromaConversion::Sharp`]: yuv crate Sharp YUV (gamma-aware bilinear).
    ///   Best quality for edges, graphics, and text.
    /// - [`ChromaConversion::Auto`]: Sharp for 4:2:0/4:2:2, Intrinsic for 4:4:4/4:4:0
    ///
    /// Sharp YUV is often 10-50% FASTER than Intrinsic due to optimized SIMD.
    #[must_use]
    pub fn chroma_conversion(mut self, method: ChromaConversion) -> Self {
        self.config.chroma_conversion = method;
        self
    }

    /// Convenience method: enable Sharp YUV chroma downsampling.
    ///
    /// - `enable = true` → `ChromaConversion::Sharp`
    /// - `enable = false` → `ChromaConversion::Intrinsic`
    #[must_use]
    pub fn sharp_yuv(mut self, enable: bool) -> Self {
        self.config.chroma_conversion = if enable {
            ChromaConversion::Sharp
        } else {
            ChromaConversion::Intrinsic
        };
        self
    }

    /// Sets an internal chroma pipeline for benchmarking (undocumented API).
    ///
    /// This method is intentionally not documented in the public API.
    /// It allows external benchmarks to test different chroma conversion
    /// and downsampling strategies without committing to a stable API.
    ///
    /// # Pathway Encoding (u64)
    ///
    /// - Bits 0-7: Color conversion (0=Auto, 1=IntrinsicF32, 2=YuvBalanced, 3=YuvProfessional)
    /// - Bits 8-15: Downsampling (0=Auto, 1=None, 2=Box, 3=BoxSmoothed, 4=Sharp, 5=GammaAwareF32)
    /// - Bits 16-23: Smoothing factor (0-100, only for BoxSmoothed)
    /// - Bits 24-63: Reserved (must be 0)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Reserved bits are non-zero
    /// - Invalid color conversion or downsampling method value
    /// - Smoothing factor > 100
    /// - Incompatible combination (e.g., Sharp with 4:4:4, None with 4:2:0)
    /// - Unimplemented method (e.g., GammaAwareF32, YuvProfessional)
    #[doc(hidden)]
    pub fn set_internal_pathway(mut self, pathway: u64) -> Result<Self> {
        let pipeline = ChromaPipeline::from_u64(pathway)?;
        pipeline.validate(self.config.subsampling)?;
        self.config.internal_pipeline = Some(pipeline);
        Ok(self)
    }

    /// Enable hybrid quantization (jpegli AQ + mozjpeg trellis).
    ///
    /// This combines jpegli's adaptive quantization (which determines WHERE
    /// to spend bits based on image content) with mozjpeg's trellis quantization
    /// (which optimizes HOW to spend bits via rate-distortion optimization).
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn hybrid_trellis(mut self, enable: bool) -> Self {
        if enable {
            self.config.hybrid_config = crate::hybrid_config::HybridConfig::default();
        } else {
            self.config.hybrid_config = crate::hybrid_config::HybridConfig::disabled();
        }
        self
    }

    /// Set custom hybrid quantization configuration.
    ///
    /// Allows fine-tuning all hybrid AQ+trellis parameters.
    /// See [`HybridConfig`](crate::hybrid_config::HybridConfig) for available options.
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn hybrid_config(mut self, config: crate::hybrid_config::HybridConfig) -> Self {
        self.config.hybrid_config = config;
        self
    }

    /// Sets a custom AQ (adaptive quantization) strength map.
    ///
    /// This allows pre-scaling the AQ map to control file size. When the AQ map
    /// is scaled up, more bits are allocated to complex regions (larger files).
    /// When scaled down, fewer bits are allocated (smaller files).
    ///
    /// If not provided, the AQ map is computed automatically from the image.
    ///
    /// # Example
    /// ```ignore
    /// use jpegli::adaptive_quant::compute_aq_strength_map;
    ///
    /// // Compute AQ map from Y plane
    /// let mut aq_map = compute_aq_strength_map(&y_plane, width, height, 8);
    ///
    /// // Scale down to reduce file size by ~16%
    /// let scale = aq_map.scale_for_size_reduction(16.0);
    /// aq_map.scale(scale);
    ///
    /// // Use the scaled map
    /// let jpeg = Encoder::new()
    ///     .width(width as u32)
    ///     .height(height as u32)
    ///     .hybrid_config(HybridConfig::default())
    ///     .aq_map(aq_map)
    ///     .encode(&pixels)?;
    /// ```
    ///
    /// Requires the `experimental-hybrid-trellis` feature.
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[must_use]
    pub fn aq_map(mut self, map: crate::adaptive_quant::AQStrengthMap) -> Self {
        self.config.custom_aq_map = Some(map);
        self
    }

    /// Validates the configuration.
    fn validate(&self) -> Result<()> {
        // Use validate_dimensions for comprehensive checks (zero, max dimension, max pixels)
        validate_dimensions(self.config.width, self.config.height, DEFAULT_MAX_PIXELS)?;
        Ok(())
    }

    /// Encodes the image data.
    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.validate()?;

        // Calculate expected size with overflow checking
        let expected_size =
            checked_size_2d(self.config.width as usize, self.config.height as usize)?;
        let expected_size =
            checked_size_2d(expected_size, self.config.pixel_format.bytes_per_pixel())?;

        if data.len() != expected_size {
            return Err(Error::InvalidBufferSize {
                expected: expected_size,
                actual: data.len(),
            });
        }

        // For now, implement baseline encoding only
        match self.config.mode {
            JpegMode::Baseline => self.encode_baseline(data),
            JpegMode::Progressive => self.encode_progressive(data),
            _ => Err(Error::UnsupportedFeature {
                feature: "extended/lossless encoding",
            }),
        }
    }

    /// Encodes as baseline JPEG.
    fn encode_baseline(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);

        if self.config.use_xyb {
            self.encode_baseline_xyb(data, &mut output)
        } else {
            self.encode_baseline_ycbcr(data, &mut output)
        }
    }

    /// Encodes using standard YCbCr color space.
    fn encode_baseline_ycbcr(&self, data: &[u8], output: &mut Vec<u8>) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Check if internal_pipeline specifies GammaAwareF32 downsampling
        if let Some(ref pipeline) = self.config.internal_pipeline {
            if pipeline.downsampling == DownsamplingMethod::GammaAwareF32 {
                // Use f32 gamma-aware path
                let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                    match self.config.subsampling {
                        Subsampling::S420 => self.convert_gamma_aware_420(data)?,
                        Subsampling::S422 => self.convert_gamma_aware_422(data)?,
                        Subsampling::S440 => self.convert_gamma_aware_440(data)?,
                        Subsampling::S444 => {
                            // Should not happen - validation prevents this
                            return Err(Error::InvalidColorFormat {
                                reason: "GammaAwareF32 not valid for 4:4:4",
                            });
                        }
                    };
                return self.encode_baseline_ycbcr_with_planes(
                    output,
                    y_plane,
                    cb_plane_final,
                    cr_plane_final,
                    c_width,
                    c_height,
                );
            }
        }

        // Resolve Auto to concrete method based on subsampling
        let chroma_method = self
            .config
            .chroma_conversion
            .resolve(self.config.subsampling);

        // yuv crate path (Sharp or Fast): performs color conversion + downsampling in one step
        if matches!(
            chroma_method,
            ChromaConversion::Sharp | ChromaConversion::Fast
        ) {
            let use_sharp = matches!(chroma_method, ChromaConversion::Sharp);
            match self.config.subsampling {
                Subsampling::S420 => {
                    let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                        self.convert_yuv_crate_420(data, use_sharp)?;
                    return self.encode_baseline_ycbcr_with_planes(
                        output,
                        y_plane,
                        cb_plane_final,
                        cr_plane_final,
                        c_width,
                        c_height,
                    );
                }
                Subsampling::S422 => {
                    let (y_plane, cb_plane_final, cr_plane_final, c_width, c_height) =
                        self.convert_yuv_crate_422(data, use_sharp)?;
                    return self.encode_baseline_ycbcr_with_planes(
                        output,
                        y_plane,
                        cb_plane_final,
                        cr_plane_final,
                        c_width,
                        c_height,
                    );
                }
                // yuv crate doesn't support 4:4:0 or 4:4:4, fall through to Intrinsic path
                _ => {}
            }
        }

        // Intrinsic path: convert to YCbCr using f32 precision throughout (matches C++ jpegli)
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Handle chroma subsampling (with optional input smoothing)
        let (cb_plane_final, cr_plane_final, c_width, c_height) = match self.config.subsampling {
            Subsampling::S420 => {
                // 4:2:0: Apply smoothing then downsample both Cb and Cr by 2x2
                let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                let cb_down = self.downsample_2x2_f32(&cb_smooth, width, height)?;
                let cr_down = self.downsample_2x2_f32(&cr_smooth, width, height)?;
                let c_w = (width + 1) / 2;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, c_w, c_h)
            }
            Subsampling::S422 => {
                // 4:2:2: Apply smoothing then downsample horizontally only
                let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                let cb_down = self.downsample_2x1_f32(&cb_smooth, width, height)?;
                let cr_down = self.downsample_2x1_f32(&cr_smooth, width, height)?;
                let c_w = (width + 1) / 2;
                (cb_down, cr_down, c_w, height)
            }
            Subsampling::S440 => {
                // 4:4:0: Apply smoothing then downsample vertically only
                let cb_smooth = self.apply_input_smoothing(&cb_plane, width, height)?;
                let cr_smooth = self.apply_input_smoothing(&cr_plane, width, height)?;
                let cb_down = self.downsample_1x2_f32(&cb_smooth, width, height)?;
                let cr_down = self.downsample_1x2_f32(&cr_smooth, width, height)?;
                let c_h = (height + 1) / 2;
                (cb_down, cr_down, width, c_h)
            }
            Subsampling::S444 => {
                // 4:4:4: No subsampling, no smoothing needed
                (cb_plane, cr_plane, width, height)
            }
        };

        self.encode_baseline_ycbcr_with_planes(
            output,
            y_plane,
            cb_plane_final,
            cr_plane_final,
            c_width,
            c_height,
        )
    }

    /// Encodes YCbCr planes to JPEG (shared by standard and Sharp YUV paths).
    fn encode_baseline_ycbcr_with_planes(
        &self,
        output: &mut Vec<u8>,
        y_plane: Vec<f32>,
        cb_plane_final: Vec<f32>,
        cr_plane_final: Vec<f32>,
        c_width: usize,
        c_height: usize,
    ) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Apply 4:2:0 quality compensation if using 4:2:0 subsampling
        let is_420 = self.config.subsampling == Subsampling::S420;
        let y_quant =
            quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false, is_420);
        let cb_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false, is_420);
        let cr_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::YCbCr, false, is_420);

        // Quantize all blocks first (needed for both standard and optimized encoding)
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks_subsampled(
            &y_plane,
            width,
            height,
            &cb_plane_final,
            &cr_plane_final,
            c_width,
            c_height,
            &y_quant,
            &cb_quant,
            &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write JPEG structure
        self.write_header(output)?;
        self.write_quant_tables(output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(output)?;

        // For optimized Huffman, build tables from block frequencies before writing DHT
        let scan_data = if self.config.optimize_huffman {
            let tables =
                self.build_optimized_tables(&y_blocks, &cb_blocks, &cr_blocks, is_color)?;
            self.write_huffman_tables_optimized(output, &tables)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header(output)?;

            // Encode with optimized tables
            self.encode_with_tables(&y_blocks, &cb_blocks, &cr_blocks, is_color, &tables)?
        } else {
            self.write_huffman_tables(output)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header(output)?;

            // Encode with standard tables
            self.encode_blocks_standard(&y_blocks, &cb_blocks, &cr_blocks, is_color)?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(std::mem::take(output))
    }

    /// Encodes using XYB mode (perceptually optimized color space).
    ///
    /// XYB encoding pipeline:
    /// 1. sRGB → linear RGB → XYB → scaled XYB (values in [0, 1])
    /// 2. Multiply by 255 for JPEG sample range
    /// 3. Level shift by subtracting 128 for DCT
    fn encode_baseline_xyb(&self, data: &[u8], output: &mut Vec<u8>) -> Result<Vec<u8>> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // Convert sRGB to scaled XYB (full color conversion pipeline)
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // Downsample B channel (XYB subsamples B to 1/4 resolution)
        // Apply input smoothing before downsampling (matches C++ jpegli behavior)
        let b_smooth = self.apply_input_smoothing(&b_plane, width, height)?;
        let b_downsampled = self.downsample_2x2_f32(&b_smooth, width, height)?;
        let b_width = (width + 1) / 2;
        let b_height = (height + 1) / 2;

        // Generate XYB quantization tables (one per component)
        // XYB mode doesn't use 4:2:0 quality compensation
        let x_quant = quant::generate_quant_table(
            self.config.quality,
            0, // X component
            ColorSpace::Rgb,
            true,
            false, // is_420
        );
        let y_quant = quant::generate_quant_table(
            self.config.quality,
            1, // Y component (luma-like)
            ColorSpace::Rgb,
            true,
            false, // is_420
        );
        let b_quant = quant::generate_quant_table(
            self.config.quality,
            2, // B component
            ColorSpace::Rgb,
            true,
            false, // is_420
        );

        // Compute AQ map from Y plane (XYB's Y is the luma-like channel)
        // Scale Y plane from [0,1] to [0,255] range for AQ computation
        let y_plane_scaled: Vec<f32> = y_plane.iter().map(|&v| v * 255.0).collect();
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = if let Some(ref custom) = self.config.custom_aq_map {
            custom.clone()
        } else {
            compute_aq_strength_map(&y_plane_scaled, width, height, y_quant_01)
        };
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(&y_plane_scaled, width, height, y_quant_01);

        // Zero-bias parameters for XYB (use YCbCr tables as approximation)
        // X and Y are luma-like (full-res), B is chroma-like (downsampled)
        let effective_distance = quant::quant_vals_to_distance(&x_quant, &y_quant, &b_quant);
        let x_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // X uses luma params
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0); // Y uses luma params
        let b_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1); // B uses chroma params

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = if self.config.hybrid_config.enabled {
            Some(HybridQuantContext::new(self.config.hybrid_config))
        } else {
            None
        };

        // Write JPEG structure for XYB mode (no JFIF, just ICC profile)
        self.write_header_xyb(output)?;
        // Write APP14 Adobe marker for RGB colorspace (required by some decoders)
        // See: https://github.com/google/jpegli/pull/135
        self.write_app14_adobe(output, 0)?; // 0 = RGB (no transform)
                                            // Write XYB ICC profile so decoders can interpret the colors correctly
        self.write_icc_profile(output, &XYB_ICC_PROFILE)?;
        self.write_quant_tables_xyb(output, &x_quant, &y_quant, &b_quant)?;
        self.write_frame_header_xyb(output)?;

        // For optimized Huffman, quantize all blocks first to collect frequencies
        let scan_data = if self.config.optimize_huffman {
            #[cfg(feature = "experimental-hybrid-trellis")]
            let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb_with_aq(
                &x_plane,
                &y_plane,
                &b_downsampled,
                width,
                height,
                b_width,
                b_height,
                &x_quant,
                &y_quant,
                &b_quant,
                &aq_map,
                hybrid_ctx.as_ref(),
            );
            #[cfg(not(feature = "experimental-hybrid-trellis"))]
            let (x_blocks, y_blocks, b_blocks) = self.quantize_all_blocks_xyb_with_aq_simple(
                &x_plane,
                &y_plane,
                &b_downsampled,
                width,
                height,
                b_width,
                b_height,
                &x_quant,
                &y_quant,
                &b_quant,
                &aq_map,
                &x_zero_bias,
                &y_zero_bias,
                &b_zero_bias,
            );
            let (dc_table, ac_table) =
                self.build_optimized_tables_xyb(&x_blocks, &y_blocks, &b_blocks)?;
            self.write_huffman_tables_xyb_optimized(output, &dc_table, &ac_table);

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header_xyb(output)?;

            // Encode with optimized tables
            self.encode_with_tables_xyb(&x_blocks, &y_blocks, &b_blocks, &dc_table, &ac_table)?
        } else {
            self.write_huffman_tables(output)?;

            if self.config.restart_interval > 0 {
                self.write_restart_interval(output)?;
            }
            self.write_scan_header_xyb(output)?;

            // Encode with standard tables
            self.encode_scan_xyb_float(
                &x_plane,
                &y_plane,
                &b_downsampled,
                width,
                height,
                b_width,
                b_height,
                &x_quant,
                &y_quant,
                &b_quant,
            )?
        };

        output.extend_from_slice(&scan_data);

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(std::mem::take(output))
    }

    /// Encodes progressive JPEG using XYB color space.
    ///
    /// This uses the same progressive scan structure as YCbCr encoding
    /// but with XYB color conversion and appropriate headers (ICC profile, APP14).
    fn encode_progressive_xyb(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);

        // Convert sRGB to scaled XYB
        let (x_plane, y_plane, b_plane) = self.convert_to_scaled_xyb(data)?;

        // XYB progressive uses 4:4:4 (no B channel downsampling unlike baseline XYB)
        // This is because progressive scans work best with same-size components

        // Generate XYB quantization tables
        let x_quant =
            quant::generate_quant_table(self.config.quality, 0, ColorSpace::Rgb, true, false);
        let y_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::Rgb, true, false);
        let b_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::Rgb, true, false);

        // Quantize all blocks for progressive encoding
        // Use X, Y, B as if they were Y, Cb, Cr for the progressive structure
        let (x_blocks, y_blocks, b_blocks) =
            self.quantize_all_blocks(&x_plane, &y_plane, &b_plane, &x_quant, &y_quant, &b_quant)?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write XYB-specific headers
        self.write_header_xyb(&mut output)?;
        // Write APP14 Adobe marker for RGB (required by some decoders)
        self.write_app14_adobe(&mut output, 0)?; // 0 = RGB (no transform)
                                                 // Write XYB ICC profile
        self.write_icc_profile(&mut output, &XYB_ICC_PROFILE)?;
        // Write quantization tables
        self.write_quant_tables(&mut output, &x_quant, &y_quant, &b_quant)?;
        // Write SOF2 frame header for progressive
        self.write_frame_header(&mut output)?;

        // Use standard Huffman tables (optimized tables could be added later)
        self.write_huffman_tables(&mut output)?;
        let tables: Option<OptimizedHuffmanTables> = None;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // Get progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // Encode each scan (reusing the YCbCr progressive scan logic)
        for scan in &scans {
            self.write_progressive_scan_header(&mut output, scan, is_color)?;
            let scan_data = self.encode_progressive_scan(
                &x_blocks, &y_blocks, &b_blocks, scan, is_color, &tables,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Converts input data to scaled XYB planes.
    ///
    /// Performs the full conversion: sRGB u8 → linear RGB → XYB → scaled XYB
    /// Output values are in [0, 1] range, ready to be scaled to [0, 255] for JPEG.
    fn convert_to_scaled_xyb(&self, data: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        let mut x_plane = try_alloc_zeroed_f32(num_pixels, "allocating XYB X plane")?;
        let mut y_plane = try_alloc_zeroed_f32(num_pixels, "allocating XYB Y plane")?;
        let mut b_plane = try_alloc_zeroed_f32(num_pixels, "allocating XYB B plane")?;

        match self.config.pixel_format {
            PixelFormat::Rgb => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Rgba => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 4], data[i * 4 + 1], data[i * 4 + 2]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Gray => {
                // Grayscale: R=G=B
                for i in 0..num_pixels {
                    let (x, y, b) = srgb_to_scaled_xyb(data[i], data[i], data[i]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Bgr => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 3 + 2], data[i * 3 + 1], data[i * 3]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Bgra => {
                for i in 0..num_pixels {
                    let (x, y, b) =
                        srgb_to_scaled_xyb(data[i * 4 + 2], data[i * 4 + 1], data[i * 4]);
                    x_plane[i] = x;
                    y_plane[i] = y;
                    b_plane[i] = b;
                }
            }
            PixelFormat::Cmyk => {
                return Err(Error::UnsupportedFeature {
                    feature: "CMYK with XYB mode",
                });
            }
        }

        Ok((x_plane, y_plane, b_plane))
    }

    /// Downsamples a float plane by 2x2 (box filter averaging).
    fn downsample_2x2_f32(&self, plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        let new_width = (width + 1) / 2;
        let new_height = (height + 1) / 2;
        let result_size = checked_size_2d(new_width, new_height)?;
        let mut result = try_alloc_zeroed_f32(result_size, "allocating downsampled plane")?;

        for y in 0..new_height {
            for x in 0..new_width {
                let x0 = x * 2;
                let y0 = y * 2;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let p00 = plane[y0 * width + x0];
                let p10 = plane[y0 * width + x1];
                let p01 = plane[y1 * width + x0];
                let p11 = plane[y1 * width + x1];

                result[y * new_width + x] = (p00 + p10 + p01 + p11) * 0.25;
            }
        }

        Ok(result)
    }

    /// Downsamples a float plane by 2x1 (horizontal only, box filter averaging).
    fn downsample_2x1_f32(&self, plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        let new_width = (width + 1) / 2;
        let result_size = checked_size_2d(new_width, height)?;
        let mut result = try_alloc_zeroed_f32(result_size, "allocating downsampled plane")?;

        for y in 0..height {
            for x in 0..new_width {
                let x0 = x * 2;
                let x1 = (x0 + 1).min(width - 1);

                let p0 = plane[y * width + x0];
                let p1 = plane[y * width + x1];

                result[y * new_width + x] = (p0 + p1) * 0.5;
            }
        }

        Ok(result)
    }

    /// Downsamples a float plane by 1x2 (vertical only, box filter averaging).
    fn downsample_1x2_f32(&self, plane: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        let new_height = (height + 1) / 2;
        let result_size = checked_size_2d(width, new_height)?;
        let mut result = try_alloc_zeroed_f32(result_size, "allocating downsampled plane")?;

        for y in 0..new_height {
            for x in 0..width {
                let y0 = y * 2;
                let y1 = (y0 + 1).min(height - 1);

                let p0 = plane[y0 * width + x];
                let p1 = plane[y1 * width + x];

                result[y * width + x] = (p0 + p1) * 0.5;
            }
        }

        Ok(result)
    }

    // ========================================================================
    // Gamma-Aware Chroma Downsampling (f32 precision)
    // ========================================================================
    //
    // These functions perform chroma downsampling with proper gamma handling:
    // 1. Convert RGB pixels to linear space (remove sRGB gamma)
    // 2. Average the linear RGB values in each 2x2/2x1/1x2 block
    // 3. Convert back to sRGB gamma
    // 4. Compute YCbCr from the averaged sRGB values
    //
    // This produces better chroma values on sharp edges and thin colored lines
    // compared to naive box-filtering of already-converted chroma planes.
    // ========================================================================

    /// Converts RGB to YCbCr with gamma-aware chroma downsampling for 4:2:0.
    ///
    /// This is the f32-native alternative to yuv crate's Sharp YUV:
    /// - Y channel computed at full resolution from each pixel
    /// - Cb/Cr computed by averaging RGB in linear space, then converting to YCbCr
    ///
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    fn convert_gamma_aware_420(
        &self,
        data: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        use crate::xyb::{linear_to_srgb, srgb_to_linear};

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;
        let c_height = (height + 1) / 2;

        // Allocate output planes
        let y_size = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, c_height)?;
        let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
        let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
        let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

        // Extract RGB data based on pixel format
        let (rgb_data, bpp) = match self.config.pixel_format {
            PixelFormat::Rgb => (data, 3),
            PixelFormat::Rgba => (data, 4),
            PixelFormat::Bgr | PixelFormat::Bgra => {
                return Err(Error::InvalidColorFormat {
                    reason: "BGR/BGRA not yet supported for gamma-aware conversion",
                });
            }
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "Unsupported pixel format for gamma-aware conversion",
                });
            }
        };

        // First pass: compute Y at full resolution
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * bpp;
                let r = rgb_data[idx] as f32;
                let g = rgb_data[idx + 1] as f32;
                let b = rgb_data[idx + 2] as f32;

                // BT.601 Y computation
                let y_val = color::rgb_to_ycbcr_f32(r, g, b).0;
                y_plane[y * width + x] = y_val;
            }
        }

        // Second pass: compute Cb/Cr with gamma-aware downsampling
        for cy in 0..c_height {
            for cx in 0..c_width {
                // Get 2x2 block coordinates
                let x0 = cx * 2;
                let y0 = cy * 2;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                // Get RGB values for all 4 pixels
                let get_rgb = |x: usize, y: usize| -> (f32, f32, f32) {
                    let idx = (y * width + x) * bpp;
                    (
                        rgb_data[idx] as f32 / 255.0,
                        rgb_data[idx + 1] as f32 / 255.0,
                        rgb_data[idx + 2] as f32 / 255.0,
                    )
                };

                let (r00, g00, b00) = get_rgb(x0, y0);
                let (r10, g10, b10) = get_rgb(x1, y0);
                let (r01, g01, b01) = get_rgb(x0, y1);
                let (r11, g11, b11) = get_rgb(x1, y1);

                // Convert to linear space
                let lr00 = srgb_to_linear(r00);
                let lg00 = srgb_to_linear(g00);
                let lb00 = srgb_to_linear(b00);

                let lr10 = srgb_to_linear(r10);
                let lg10 = srgb_to_linear(g10);
                let lb10 = srgb_to_linear(b10);

                let lr01 = srgb_to_linear(r01);
                let lg01 = srgb_to_linear(g01);
                let lb01 = srgb_to_linear(b01);

                let lr11 = srgb_to_linear(r11);
                let lg11 = srgb_to_linear(g11);
                let lb11 = srgb_to_linear(b11);

                // Average in linear space
                let lr_avg = (lr00 + lr10 + lr01 + lr11) * 0.25;
                let lg_avg = (lg00 + lg10 + lg01 + lg11) * 0.25;
                let lb_avg = (lb00 + lb10 + lb01 + lb11) * 0.25;

                // Convert back to sRGB
                let r_avg = linear_to_srgb(lr_avg) * 255.0;
                let g_avg = linear_to_srgb(lg_avg) * 255.0;
                let b_avg = linear_to_srgb(lb_avg) * 255.0;

                // Convert to YCbCr (we only need Cb and Cr)
                let (_, cb, cr) = color::rgb_to_ycbcr_f32(r_avg, g_avg, b_avg);

                cb_plane[cy * c_width + cx] = cb;
                cr_plane[cy * c_width + cx] = cr;
            }
        }

        Ok((y_plane, cb_plane, cr_plane, c_width, c_height))
    }

    /// Converts RGB to YCbCr with gamma-aware chroma downsampling for 4:2:2.
    ///
    /// Similar to 4:2:0 but only downsamples horizontally (2x1 blocks).
    ///
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    fn convert_gamma_aware_422(
        &self,
        data: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        use crate::xyb::{linear_to_srgb, srgb_to_linear};

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;

        // Allocate output planes
        let y_size = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, height)?;
        let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
        let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
        let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

        let (rgb_data, bpp) = match self.config.pixel_format {
            PixelFormat::Rgb => (data, 3),
            PixelFormat::Rgba => (data, 4),
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "Unsupported pixel format for gamma-aware conversion",
                });
            }
        };

        // First pass: compute Y at full resolution
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * bpp;
                let r = rgb_data[idx] as f32;
                let g = rgb_data[idx + 1] as f32;
                let b = rgb_data[idx + 2] as f32;
                y_plane[y * width + x] = color::rgb_to_ycbcr_f32(r, g, b).0;
            }
        }

        // Second pass: gamma-aware horizontal downsampling for Cb/Cr
        for y in 0..height {
            for cx in 0..c_width {
                let x0 = cx * 2;
                let x1 = (x0 + 1).min(width - 1);

                let get_rgb = |x: usize| -> (f32, f32, f32) {
                    let idx = (y * width + x) * bpp;
                    (
                        rgb_data[idx] as f32 / 255.0,
                        rgb_data[idx + 1] as f32 / 255.0,
                        rgb_data[idx + 2] as f32 / 255.0,
                    )
                };

                let (r0, g0, b0) = get_rgb(x0);
                let (r1, g1, b1) = get_rgb(x1);

                // Convert to linear and average
                let lr_avg = (srgb_to_linear(r0) + srgb_to_linear(r1)) * 0.5;
                let lg_avg = (srgb_to_linear(g0) + srgb_to_linear(g1)) * 0.5;
                let lb_avg = (srgb_to_linear(b0) + srgb_to_linear(b1)) * 0.5;

                // Convert back to sRGB then YCbCr
                let r_avg = linear_to_srgb(lr_avg) * 255.0;
                let g_avg = linear_to_srgb(lg_avg) * 255.0;
                let b_avg = linear_to_srgb(lb_avg) * 255.0;

                let (_, cb, cr) = color::rgb_to_ycbcr_f32(r_avg, g_avg, b_avg);
                cb_plane[y * c_width + cx] = cb;
                cr_plane[y * c_width + cx] = cr;
            }
        }

        Ok((y_plane, cb_plane, cr_plane, c_width, height))
    }

    /// Converts RGB to YCbCr with gamma-aware chroma downsampling for 4:4:0.
    ///
    /// Similar to 4:2:0 but only downsamples vertically (1x2 blocks).
    ///
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    fn convert_gamma_aware_440(
        &self,
        data: &[u8],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        use crate::xyb::{linear_to_srgb, srgb_to_linear};

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_height = (height + 1) / 2;

        // Allocate output planes
        let y_size = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(width, c_height)?;
        let mut y_plane = try_alloc_zeroed_f32(y_size, "Y plane")?;
        let mut cb_plane = try_alloc_zeroed_f32(c_size, "Cb plane")?;
        let mut cr_plane = try_alloc_zeroed_f32(c_size, "Cr plane")?;

        let (rgb_data, bpp) = match self.config.pixel_format {
            PixelFormat::Rgb => (data, 3),
            PixelFormat::Rgba => (data, 4),
            _ => {
                return Err(Error::InvalidColorFormat {
                    reason: "Unsupported pixel format for gamma-aware conversion",
                });
            }
        };

        // First pass: compute Y at full resolution
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * bpp;
                let r = rgb_data[idx] as f32;
                let g = rgb_data[idx + 1] as f32;
                let b = rgb_data[idx + 2] as f32;
                y_plane[y * width + x] = color::rgb_to_ycbcr_f32(r, g, b).0;
            }
        }

        // Second pass: gamma-aware vertical downsampling for Cb/Cr
        for cy in 0..c_height {
            let y0 = cy * 2;
            let y1 = (y0 + 1).min(height - 1);

            for x in 0..width {
                let get_rgb = |y: usize| -> (f32, f32, f32) {
                    let idx = (y * width + x) * bpp;
                    (
                        rgb_data[idx] as f32 / 255.0,
                        rgb_data[idx + 1] as f32 / 255.0,
                        rgb_data[idx + 2] as f32 / 255.0,
                    )
                };

                let (r0, g0, b0) = get_rgb(y0);
                let (r1, g1, b1) = get_rgb(y1);

                // Convert to linear and average
                let lr_avg = (srgb_to_linear(r0) + srgb_to_linear(r1)) * 0.5;
                let lg_avg = (srgb_to_linear(g0) + srgb_to_linear(g1)) * 0.5;
                let lb_avg = (srgb_to_linear(b0) + srgb_to_linear(b1)) * 0.5;

                // Convert back to sRGB then YCbCr
                let r_avg = linear_to_srgb(lr_avg) * 255.0;
                let g_avg = linear_to_srgb(lg_avg) * 255.0;
                let b_avg = linear_to_srgb(lb_avg) * 255.0;

                let (_, cb, cr) = color::rgb_to_ycbcr_f32(r_avg, g_avg, b_avg);
                cb_plane[cy * width + x] = cb;
                cr_plane[cy * width + x] = cr;
            }
        }

        Ok((y_plane, cb_plane, cr_plane, width, c_height))
    }

    /// Applies input smoothing to a plane before downsampling.
    ///
    /// This is a 3x3 weighted blur matching libjpeg/jpegli's smoothing_factor:
    /// - Center pixel weight: 1.0 - 8 * (factor / 1024)
    /// - Neighbor pixel weight: factor / 1024
    ///
    /// Only applied when smoothing_factor > 0 and plane will be downsampled.
    fn apply_input_smoothing(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>> {
        let factor = self.config.smoothing_factor;
        if factor == 0 {
            // No smoothing - return a copy
            return Ok(plane.to_vec());
        }

        let result_size = checked_size_2d(width, height)?;
        let mut result = try_alloc_zeroed_f32(result_size, "allocating smoothed plane")?;

        // Weights matching C++ jpegli: kW1 = factor/1024, kW0 = 1 - 8*kW1
        let kw1 = factor as f32 / 1024.0;
        let kw0 = 1.0 - 8.0 * kw1;

        for y in 0..height {
            for x in 0..width {
                // Clamp coordinates to handle edges
                let x_l = x.saturating_sub(1);
                let x_r = (x + 1).min(width - 1);
                let y_t = y.saturating_sub(1);
                let y_b = (y + 1).min(height - 1);

                // Get 3x3 neighborhood
                let val_tl = plane[y_t * width + x_l];
                let val_tm = plane[y_t * width + x];
                let val_tr = plane[y_t * width + x_r];
                let val_ml = plane[y * width + x_l];
                let val_mm = plane[y * width + x]; // center
                let val_mr = plane[y * width + x_r];
                let val_bl = plane[y_b * width + x_l];
                let val_bm = plane[y_b * width + x];
                let val_br = plane[y_b * width + x_r];

                // Weighted sum: center * kW0 + neighbors * kW1
                let neighbors =
                    val_tl + val_tm + val_tr + val_ml + val_mr + val_bl + val_bm + val_br;
                result[y * width + x] = val_mm * kw0 + neighbors * kw1;
            }
        }

        Ok(result)
    }

    /// Converts RGB to YCbCr using yuv crate for 4:2:0 subsampling.
    ///
    /// If `use_sharp` is true, uses Sharp YUV (gamma-aware, better edges).
    /// If `use_sharp` is false, uses standard conversion (fast, simple box filter).
    ///
    /// Returns: (y_plane, cb_plane, cr_plane, chroma_width, chroma_height)
    fn convert_yuv_crate_420(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;
        let c_height = (height + 1) / 2;

        // Allocate YUV planar image
        let mut yuv_image =
            YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

        // Get RGB data in the right format
        let (rgb_data, rgb_stride) = match self.config.pixel_format {
            PixelFormat::Rgb => (data, width as u32 * 3),
            PixelFormat::Rgba => {
                // Convert RGBA to RGB
                let mut rgb = Vec::with_capacity(width * height * 3);
                for i in 0..(width * height) {
                    rgb.push(data[i * 4]);
                    rgb.push(data[i * 4 + 1]);
                    rgb.push(data[i * 4 + 2]);
                }
                return self
                    .convert_yuv_crate_420_rgb(&rgb, width, height, c_width, c_height, use_sharp);
            }
            PixelFormat::Bgr => {
                // Convert BGR to RGB
                let mut rgb = Vec::with_capacity(width * height * 3);
                for i in 0..(width * height) {
                    rgb.push(data[i * 3 + 2]); // R
                    rgb.push(data[i * 3 + 1]); // G
                    rgb.push(data[i * 3]); // B
                }
                return self
                    .convert_yuv_crate_420_rgb(&rgb, width, height, c_width, c_height, use_sharp);
            }
            PixelFormat::Bgra => {
                // Convert BGRA to RGB
                let mut rgb = Vec::with_capacity(width * height * 3);
                for i in 0..(width * height) {
                    rgb.push(data[i * 4 + 2]); // R
                    rgb.push(data[i * 4 + 1]); // G
                    rgb.push(data[i * 4]); // B
                }
                return self
                    .convert_yuv_crate_420_rgb(&rgb, width, height, c_width, c_height, use_sharp);
            }
            PixelFormat::Gray => {
                // Grayscale: Y = pixel value, Cb/Cr = 128
                let num_pixels = checked_size_2d(width, height)?;
                let mut y_plane = try_alloc_zeroed_f32(num_pixels, "Y plane")?;
                let c_size = checked_size_2d(c_width, c_height)?;
                let cb_plane = try_alloc_filled(c_size, 128.0f32, "Cb plane")?;
                let cr_plane = try_alloc_filled(c_size, 128.0f32, "Cr plane")?;
                for i in 0..num_pixels {
                    y_plane[i] = data[i] as f32;
                }
                return Ok((y_plane, cb_plane, cr_plane, c_width, c_height));
            }
            PixelFormat::Cmyk => {
                return Err(Error::InvalidColorFormat {
                    reason: "yuv crate does not support CMYK input",
                });
            }
        };

        // Perform YUV conversion (sharp or standard)
        if use_sharp {
            rgb_to_sharp_yuv420(
                &mut yuv_image,
                rgb_data,
                rgb_stride,
                YuvRange::Full,              // JPEG uses full range (0-255)
                YuvStandardMatrix::Bt601,    // Standard JPEG matrix
                SharpYuvGammaTransfer::Srgb, // sRGB input
            )
            .map_err(|e| Error::IoError {
                reason: format!("Sharp YUV conversion failed: {:?}", e),
            })?;
        } else {
            rgb_to_yuv420(
                &mut yuv_image,
                rgb_data,
                rgb_stride,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Balanced, // Fast path uses balanced mode
            )
            .map_err(|e| Error::IoError {
                reason: format!("YUV conversion failed: {:?}", e),
            })?;
        }

        // Convert u8 planes to f32
        let num_pixels = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, c_height)?;

        let mut y_plane_f32 = try_alloc_zeroed_f32(num_pixels, "Y plane f32")?;
        let mut cb_plane_f32 = try_alloc_zeroed_f32(c_size, "Cb plane f32")?;
        let mut cr_plane_f32 = try_alloc_zeroed_f32(c_size, "Cr plane f32")?;

        // Copy Y plane (full resolution)
        for (i, &y) in yuv_image
            .y_plane
            .borrow()
            .iter()
            .take(num_pixels)
            .enumerate()
        {
            y_plane_f32[i] = y as f32;
        }

        // Copy U/V planes (already downsampled)
        for (i, &u) in yuv_image.u_plane.borrow().iter().take(c_size).enumerate() {
            cb_plane_f32[i] = u as f32;
        }
        for (i, &v) in yuv_image.v_plane.borrow().iter().take(c_size).enumerate() {
            cr_plane_f32[i] = v as f32;
        }

        Ok((y_plane_f32, cb_plane_f32, cr_plane_f32, c_width, c_height))
    }

    /// Helper for yuv crate with pre-converted RGB data.
    fn convert_yuv_crate_420_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        c_width: usize,
        c_height: usize,
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let mut yuv_image =
            YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

        if use_sharp {
            rgb_to_sharp_yuv420(
                &mut yuv_image,
                rgb,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                SharpYuvGammaTransfer::Srgb,
            )
            .map_err(|e| Error::IoError {
                reason: format!("Sharp YUV conversion failed: {:?}", e),
            })?;
        } else {
            rgb_to_yuv420(
                &mut yuv_image,
                rgb,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Balanced,
            )
            .map_err(|e| Error::IoError {
                reason: format!("YUV conversion failed: {:?}", e),
            })?;
        }

        let num_pixels = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, c_height)?;

        let mut y_plane_f32 = try_alloc_zeroed_f32(num_pixels, "Y plane f32")?;
        let mut cb_plane_f32 = try_alloc_zeroed_f32(c_size, "Cb plane f32")?;
        let mut cr_plane_f32 = try_alloc_zeroed_f32(c_size, "Cr plane f32")?;

        for (i, &y) in yuv_image
            .y_plane
            .borrow()
            .iter()
            .take(num_pixels)
            .enumerate()
        {
            y_plane_f32[i] = y as f32;
        }
        for (i, &u) in yuv_image.u_plane.borrow().iter().take(c_size).enumerate() {
            cb_plane_f32[i] = u as f32;
        }
        for (i, &v) in yuv_image.v_plane.borrow().iter().take(c_size).enumerate() {
            cr_plane_f32[i] = v as f32;
        }

        Ok((y_plane_f32, cb_plane_f32, cr_plane_f32, c_width, c_height))
    }

    /// Converts RGB to YCbCr using yuv crate for 4:2:2 subsampling.
    ///
    /// If `use_sharp` is true, uses Sharp YUV (gamma-aware, better edges).
    /// If `use_sharp` is false, uses standard conversion (fast, simple box filter).
    fn convert_yuv_crate_422(
        &self,
        data: &[u8],
        use_sharp: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let c_width = (width + 1) / 2;

        // For formats other than RGB, convert first
        let rgb_data: Vec<u8>;
        let rgb_ref: &[u8] = match self.config.pixel_format {
            PixelFormat::Rgb => data,
            PixelFormat::Rgba => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 4], data[i * 4 + 1], data[i * 4 + 2]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Bgr => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 3 + 2], data[i * 3 + 1], data[i * 3]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Bgra => {
                rgb_data = (0..(width * height))
                    .flat_map(|i| [data[i * 4 + 2], data[i * 4 + 1], data[i * 4]])
                    .collect();
                &rgb_data
            }
            PixelFormat::Gray => {
                // Grayscale doesn't benefit from yuv crate conversion
                let num_pixels = checked_size_2d(width, height)?;
                let mut y_plane = try_alloc_zeroed_f32(num_pixels, "Y plane")?;
                let c_size = checked_size_2d(c_width, height)?;
                let cb_plane = try_alloc_filled(c_size, 128.0f32, "Cb plane")?;
                let cr_plane = try_alloc_filled(c_size, 128.0f32, "Cr plane")?;
                for i in 0..num_pixels {
                    y_plane[i] = data[i] as f32;
                }
                return Ok((y_plane, cb_plane, cr_plane, c_width, height));
            }
            PixelFormat::Cmyk => {
                return Err(Error::InvalidColorFormat {
                    reason: "yuv crate does not support CMYK input",
                });
            }
        };

        let mut yuv_image =
            YuvPlanarImageMut::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv422);

        if use_sharp {
            rgb_to_sharp_yuv422(
                &mut yuv_image,
                rgb_ref,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                SharpYuvGammaTransfer::Srgb,
            )
            .map_err(|e| Error::IoError {
                reason: format!("Sharp YUV 422 conversion failed: {:?}", e),
            })?;
        } else {
            rgb_to_yuv422(
                &mut yuv_image,
                rgb_ref,
                width as u32 * 3,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
                YuvConversionMode::Balanced,
            )
            .map_err(|e| Error::IoError {
                reason: format!("YUV 422 conversion failed: {:?}", e),
            })?;
        }

        let num_pixels = checked_size_2d(width, height)?;
        let c_size = checked_size_2d(c_width, height)?;

        let mut y_plane_f32 = try_alloc_zeroed_f32(num_pixels, "Y plane f32")?;
        let mut cb_plane_f32 = try_alloc_zeroed_f32(c_size, "Cb plane f32")?;
        let mut cr_plane_f32 = try_alloc_zeroed_f32(c_size, "Cr plane f32")?;

        for (i, &y) in yuv_image
            .y_plane
            .borrow()
            .iter()
            .take(num_pixels)
            .enumerate()
        {
            y_plane_f32[i] = y as f32;
        }
        for (i, &u) in yuv_image.u_plane.borrow().iter().take(c_size).enumerate() {
            cb_plane_f32[i] = u as f32;
        }
        for (i, &v) in yuv_image.v_plane.borrow().iter().take(c_size).enumerate() {
            cr_plane_f32[i] = v as f32;
        }

        Ok((y_plane_f32, cb_plane_f32, cr_plane_f32, c_width, height))
    }

    /// Encodes as progressive JPEG (level 2, matching cjpegli default).
    ///
    /// Progressive level 2 uses the following scan script:
    /// 1. DC first: Ss=0, Se=0, Ah=0, Al=0 (DC only, full precision)
    /// 2. AC 1-2: Ss=1, Se=2, Ah=0, Al=0 (low AC, full precision)
    /// 3. AC 3-63 first: Ss=3, Se=63, Ah=0, Al=2 (high AC, top bits)
    /// 4. AC 3-63 refine: Ss=3, Se=63, Ah=2, Al=1 (bit 1 refinement)
    /// 5. AC 3-63 refine: Ss=3, Se=63, Ah=1, Al=0 (bit 0 refinement)
    fn encode_progressive(&self, data: &[u8]) -> Result<Vec<u8>> {
        // XYB progressive mode - route to specialized encoder
        if self.config.use_xyb {
            return self.encode_progressive_xyb(data);
        }

        // Use tokenization-based approach when optimizing Huffman tables
        if self.config.optimize_huffman {
            return self.encode_progressive_optimized(data);
        }

        let mut output = Vec::with_capacity(data.len() / 4);

        // Convert to YCbCr using f32 precision
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Progressive mode uses 4:4:4, so is_420 = false
        let y_quant =
            quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false, false);
        let cb_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false, false);
        let cr_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::YCbCr, false, false);

        // Quantize all blocks to get full-precision coefficients
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks(
            &y_plane, &cb_plane, &cr_plane, &y_quant, &cb_quant, &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Write JPEG structure
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // For non-optimized progressive, use standard Huffman tables
        self.write_huffman_tables(&mut output)?;
        let tables: Option<OptimizedHuffmanTables> = None;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // Define progressive scan script (level 2)
        // For 4:4:4 (no subsampling), DC can be interleaved
        let scans = self.get_progressive_scan_script(is_color);

        // Encode each scan
        for scan in &scans {
            // Write SOS header for this scan
            self.write_progressive_scan_header(&mut output, scan, is_color)?;

            // Encode the scan data
            let scan_data = self.encode_progressive_scan(
                &y_blocks, &cb_blocks, &cr_blocks, scan, is_color, &tables,
            )?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Encodes progressive JPEG with optimized Huffman tables using two-pass tokenization.
    ///
    /// This approach:
    /// 1. Tokenizes all scans first to collect actual symbol usage
    /// 2. Builds histograms from actual tokens (not estimated baseline statistics)
    /// 3. Clusters similar histograms to minimize table overhead
    /// 4. Generates optimal Huffman tables from clustered histograms
    /// 5. Replays tokens with optimized tables
    fn encode_progressive_optimized(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(data.len() / 4);

        // Convert to YCbCr using f32 precision
        let (y_plane, cb_plane, cr_plane) = self.convert_to_ycbcr_f32(data)?;

        // Generate quantization tables (3 separate tables like C++ cjpegli)
        // Progressive mode uses 4:4:4, so is_420 = false
        let y_quant =
            quant::generate_quant_table(self.config.quality, 0, ColorSpace::YCbCr, false, false);
        let cb_quant =
            quant::generate_quant_table(self.config.quality, 1, ColorSpace::YCbCr, false, false);
        let cr_quant =
            quant::generate_quant_table(self.config.quality, 2, ColorSpace::YCbCr, false, false);

        // Quantize all blocks to get full-precision coefficients
        let (y_blocks, cb_blocks, cr_blocks) = self.quantize_all_blocks(
            &y_plane, &cb_plane, &cr_plane, &y_quant, &cb_quant, &cr_quant,
        )?;
        let is_color = self.config.pixel_format != PixelFormat::Gray;
        let num_components = if is_color { 3 } else { 1 };

        // Define progressive scan script
        let scans = self.get_progressive_scan_script(is_color);

        // ========== PASS 1: TOKENIZATION ==========
        // Tokenize all scans to collect symbol statistics
        let mut token_buffer = ProgressiveTokenBuffer::new(num_components, scans.len());

        for scan in scans.iter() {
            // Calculate context for this scan
            // Context determines which Huffman table histogram to use
            let context = if scan.ss == 0 && scan.se == 0 {
                // DC scan: use component index as context (0=Y, 1=Cb, 2=Cr)
                scan.components[0]
            } else {
                // AC scan: use num_components + component_index as context
                // This ensures Y always uses luma table, Cb/Cr use chroma table
                // regardless of scan order (which varies with subsampling mode)
                (num_components as u8) + scan.components[0]
            };

            if scan.ss == 0 && scan.se == 0 {
                // DC scan
                let blocks: Vec<&[[i16; DCT_BLOCK_SIZE]]> = scan
                    .components
                    .iter()
                    .map(|&c| match c {
                        0 => y_blocks.as_slice(),
                        1 => cb_blocks.as_slice(),
                        2 => cr_blocks.as_slice(),
                        _ => &[][..],
                    })
                    .collect();
                let component_indices: Vec<usize> =
                    scan.components.iter().map(|&c| c as usize).collect();
                token_buffer.tokenize_dc_scan(&blocks, &component_indices, scan.al, scan.ah);
            } else if scan.ah == 0 {
                // AC first scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &y_blocks,
                    1 => &cb_blocks,
                    2 => &cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_first_scan(blocks, context, scan.ss, scan.se, scan.al);
            } else {
                // AC refinement scan
                let blocks: &[[i16; DCT_BLOCK_SIZE]] = match scan.components[0] {
                    0 => &y_blocks,
                    1 => &cb_blocks,
                    2 => &cr_blocks,
                    _ => {
                        return Err(Error::InternalError {
                            reason: "Invalid component",
                        })
                    }
                };
                token_buffer.tokenize_ac_refinement_scan(
                    blocks, context, scan.ss, scan.se, scan.ah, scan.al,
                );
            }
        }

        // ========== GENERATE OPTIMIZED TABLES ==========
        // Use explicit luma/chroma grouping to ensure table assignment matches
        // what the replay code expects (luma=0, chroma=1)
        let (num_dc_tables, tables) = token_buffer.generate_luma_chroma_tables(num_components)?;

        // Convert to OptimizedHuffmanTables format for compatibility
        let opt_tables =
            self.build_progressive_huffman_tables(&tables, num_components, num_dc_tables)?;

        // ========== WRITE JPEG STRUCTURE ==========
        self.write_header(&mut output)?;
        self.write_quant_tables(&mut output, &y_quant, &cb_quant, &cr_quant)?;
        self.write_frame_header(&mut output)?; // Uses SOF2 for progressive

        // Write optimized Huffman tables
        self.write_huffman_tables_optimized(&mut output, &opt_tables)?;

        if self.config.restart_interval > 0 {
            self.write_restart_interval(&mut output)?;
        }

        // ========== PASS 2: REPLAY TOKENS ==========
        // Encode each scan by replaying tokens with optimized tables
        for (scan_idx, scan) in scans.iter().enumerate() {
            // Write SOS header
            self.write_progressive_scan_header(&mut output, scan, is_color)?;

            // Replay tokens for this scan
            let scan_data =
                self.replay_progressive_scan(&token_buffer, scan_idx, scan, is_color, &opt_tables)?;
            output.extend_from_slice(&scan_data);
        }

        // Write EOI
        output.push(0xFF);
        output.push(MARKER_EOI);

        Ok(output)
    }

    /// Builds OptimizedHuffmanTables from the clustered tables.
    fn build_progressive_huffman_tables(
        &self,
        tables: &[OptimizedTable],
        num_components: usize,
        num_dc_tables: usize,
    ) -> Result<OptimizedHuffmanTables> {
        // Tables are arranged: DC clusters first, then AC clusters
        // num_dc_tables tells us where DC ends and AC begins

        let dc_luma = tables.first().cloned().unwrap_or_else(|| {
            // Create a minimal default table
            let mut counter = FrequencyCounter::new();
            counter.count(0);
            counter.generate_table_with_dht().unwrap()
        });

        // DC chroma is the second DC table if it exists
        let dc_chroma = if num_components > 1 && num_dc_tables > 1 {
            tables.get(1).cloned().unwrap_or_else(|| dc_luma.clone())
        } else {
            dc_luma.clone()
        };

        // AC tables start after DC tables
        let ac_luma = tables.get(num_dc_tables).cloned().unwrap_or_else(|| {
            let mut counter = FrequencyCounter::new();
            counter.count(0);
            counter.generate_table_with_dht().unwrap()
        });

        // AC chroma is the second AC table if it exists
        let ac_chroma = if num_components > 1 && tables.len() > num_dc_tables + 1 {
            tables
                .get(num_dc_tables + 1)
                .cloned()
                .unwrap_or_else(|| ac_luma.clone())
        } else {
            ac_luma.clone()
        };

        Ok(OptimizedHuffmanTables {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
    }

    /// Replays tokens for a progressive scan with optimized tables.
    fn replay_progressive_scan(
        &self,
        token_buffer: &ProgressiveTokenBuffer,
        scan_idx: usize,
        scan: &ProgressiveScan,
        is_color: bool,
        tables: &OptimizedHuffmanTables,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        encoder.set_dc_table(0, tables.dc_luma.table.clone());
        encoder.set_ac_table(0, tables.ac_luma.table.clone());
        if is_color {
            encoder.set_dc_table(1, tables.dc_chroma.table.clone());
            encoder.set_ac_table(1, tables.ac_chroma.table.clone());
        }

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        // Get scan info
        let scan_info = token_buffer
            .scan_info
            .get(scan_idx)
            .ok_or(Error::InternalError {
                reason: "Scan info not found",
            })?;

        if scan.ss == 0 && scan.se == 0 {
            // DC scan: replay DC tokens
            let tokens = token_buffer.scan_tokens(scan_idx);
            // Create context map for DC (component index -> table index)
            let context_to_table: Vec<usize> = (0..4)
                .map(|c| if is_color && c > 0 { 1 } else { 0 })
                .collect();
            encoder.write_dc_tokens(tokens, &context_to_table)?;
        } else if scan.ah == 0 {
            // AC first scan: replay AC tokens
            let tokens = token_buffer.scan_tokens(scan_idx);
            let table_idx = if is_color && scan.components[0] > 0 {
                1
            } else {
                0
            };
            encoder.write_ac_first_tokens(tokens, table_idx)?;
        } else {
            // AC refinement scan: replay refinement tokens
            let table_idx = if is_color && scan.components[0] > 0 {
                1
            } else {
                0
            };
            encoder.write_ac_refinement_tokens(scan_info, table_idx)?;
        }

        Ok(encoder.finish())
    }

    /// Returns the progressive scan script for level 2.
    fn get_progressive_scan_script(&self, is_color: bool) -> Vec<ProgressiveScan> {
        let num_components = if is_color { 3 } else { 1 };
        let mut scans = Vec::new();

        // For 4:4:4 subsampling, DC can be interleaved
        let dc_interleaved = matches!(self.config.subsampling, Subsampling::S444);

        // DC first scan
        if dc_interleaved && is_color {
            // Interleaved DC for all components
            scans.push(ProgressiveScan {
                components: vec![0, 1, 2],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            });
        } else {
            // Non-interleaved DC
            for c in 0..num_components {
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 0,
                    se: 0,
                    ah: 0,
                    al: 0,
                });
            }
        }

        // AC scans are always non-interleaved
        // Progressive Level 2 with successive approximation (matches C++ jpegli)
        let use_refinement = true;

        for c in 0..num_components {
            if use_refinement {
                // Level 2: with successive approximation
                // AC 1-2: full precision (low frequency, most visible)
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 1,
                    se: 2,
                    ah: 0,
                    al: 0,
                });

                // AC 3-63 first pass: top bits only (Al=2 means bits 2+)
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 3,
                    se: 63,
                    ah: 0,
                    al: 2,
                });

                // AC 3-63 refinement: bit 1 (Ah=2, Al=1)
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 3,
                    se: 63,
                    ah: 2,
                    al: 1,
                });

                // AC 3-63 refinement: bit 0 (Ah=1, Al=0)
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 3,
                    se: 63,
                    ah: 1,
                    al: 0,
                });
            } else {
                // Level 0: no successive approximation (simpler, works)
                scans.push(ProgressiveScan {
                    components: vec![c],
                    ss: 1,
                    se: 63,
                    ah: 0,
                    al: 0,
                });
            }
        }

        scans
    }

    /// Writes SOS header for a progressive scan.
    fn write_progressive_scan_header(
        &self,
        output: &mut Vec<u8>,
        scan: &ProgressiveScan,
        is_color: bool,
    ) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        let num_components = scan.components.len() as u8;
        let length = 6u16 + num_components as u16 * 2;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(num_components);

        for &comp_idx in &scan.components {
            // Component ID (1-based for YCbCr)
            let comp_id = comp_idx + 1;
            output.push(comp_id);

            // DC/AC table selectors
            // For DC scans (ss=0): use DC table for the component
            // For AC scans (ss>0): use AC table for the component
            let table_selector = if is_color && comp_idx > 0 {
                0x11 // DC table 1, AC table 1 for chroma
            } else {
                0x00 // DC table 0, AC table 0 for luma
            };
            output.push(table_selector);
        }

        output.push(scan.ss); // Spectral selection start
        output.push(scan.se); // Spectral selection end
        output.push((scan.ah << 4) | scan.al); // Successive approximation

        Ok(())
    }

    /// Encodes a single progressive scan.
    fn encode_progressive_scan(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        is_color: bool,
        tables: &Option<OptimizedHuffmanTables>,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        if let Some(ref opt_tables) = tables {
            encoder.set_dc_table(0, opt_tables.dc_luma.table.clone());
            encoder.set_ac_table(0, opt_tables.ac_luma.table.clone());
            if is_color {
                encoder.set_dc_table(1, opt_tables.dc_chroma.table.clone());
                encoder.set_ac_table(1, opt_tables.ac_chroma.table.clone());
            }
        } else {
            encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
            encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
            if is_color {
                encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
                encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());
            }
        }

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let blocks_h = (width + DCT_SIZE - 1) / DCT_SIZE;
        let blocks_v = (height + DCT_SIZE - 1) / DCT_SIZE;

        // Determine scan type and encode accordingly
        if scan.ss == 0 && scan.se == 0 {
            // DC scan (first or refinement)
            self.encode_dc_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        } else if scan.ah == 0 {
            // AC first scan
            self.encode_ac_first_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        } else {
            // AC refinement scan
            self.encode_ac_refine_scan(
                &mut encoder,
                y_blocks,
                cb_blocks,
                cr_blocks,
                scan,
                blocks_h,
                blocks_v,
                is_color,
            )?;
        }

        Ok(encoder.finish())
    }

    /// Encodes DC scan (first or refinement).
    fn encode_dc_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                for (comp_num, &comp_idx) in scan.components.iter().enumerate() {
                    let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
                        0 => y_blocks,
                        1 => cb_blocks,
                        2 => cr_blocks,
                        _ => {
                            return Err(Error::InternalError {
                                reason: "Invalid component index",
                            })
                        }
                    };

                    if block_idx >= blocks.len() {
                        continue;
                    }

                    let dc = blocks[block_idx][0];
                    let table = if is_color && comp_idx > 0 { 1 } else { 0 };

                    encoder.encode_dc_progressive(dc, comp_num, table, scan.al, scan.ah)?;
                }
            }
        }

        Ok(())
    }

    /// Encodes AC first scan (Ah=0, ss>0).
    fn encode_ac_first_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        // AC first scan is always non-interleaved (single component)
        assert_eq!(scan.components.len(), 1);
        let comp_idx = scan.components[0];

        let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => {
                return Err(Error::InternalError {
                    reason: "Invalid component index",
                })
            }
        };

        let table_idx = if is_color && comp_idx > 0 { 1 } else { 0 };

        let mut eob_run = 0u16;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                if block_idx >= blocks.len() {
                    continue;
                }

                encoder.encode_ac_progressive_first(
                    &blocks[block_idx],
                    table_idx,
                    scan.ss,
                    scan.se,
                    scan.al,
                    &mut eob_run,
                )?;
            }
        }

        // Flush remaining EOB run
        encoder.flush_eob_run(table_idx, eob_run)?;

        Ok(())
    }

    /// Encodes AC refinement scan (Ah>0, ss>0).
    fn encode_ac_refine_scan(
        &self,
        encoder: &mut EntropyEncoder,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        scan: &ProgressiveScan,
        blocks_h: usize,
        blocks_v: usize,
        is_color: bool,
    ) -> Result<()> {
        // AC refinement scan is always non-interleaved
        assert_eq!(scan.components.len(), 1);
        let comp_idx = scan.components[0];

        let blocks: &[[i16; DCT_BLOCK_SIZE]] = match comp_idx {
            0 => y_blocks,
            1 => cb_blocks,
            2 => cr_blocks,
            _ => {
                return Err(Error::InternalError {
                    reason: "Invalid component index",
                })
            }
        };

        let table_idx = if is_color && comp_idx > 0 { 1 } else { 0 };

        let mut eob_run = 0u16;

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                let block_idx = by * blocks_h + bx;

                if block_idx >= blocks.len() {
                    continue;
                }

                encoder.encode_ac_progressive_refine(
                    &blocks[block_idx],
                    table_idx,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                    &mut eob_run,
                )?;
            }
        }

        // Flush remaining EOB run
        encoder.flush_refine_eob(table_idx, eob_run)?;

        Ok(())
    }

    /// Converts input data to YCbCr planes (u8 version - legacy).
    #[allow(dead_code)]
    fn convert_to_ycbcr(&self, data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                let y = data.to_vec();
                let cb = try_alloc_filled(num_pixels, 128u8, "YCbCr Cb plane")?;
                let cr = try_alloc_filled(num_pixels, 128u8, "YCbCr Cr plane")?;
                Ok((y, cb, cr))
            }
            PixelFormat::Rgb => color::rgb_to_ycbcr_planes(data, width, height),
            PixelFormat::Rgba => {
                // Strip alpha and convert
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Bgr => {
                let rgb: Vec<u8> = data
                    .chunks(3)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Bgra => {
                let rgb: Vec<u8> = data
                    .chunks(4)
                    .flat_map(|chunk| [chunk[2], chunk[1], chunk[0]])
                    .collect();
                color::rgb_to_ycbcr_planes(&rgb, width, height)
            }
            PixelFormat::Cmyk => Err(Error::UnsupportedFeature {
                feature: "CMYK encoding",
            }),
        }
    }

    /// Converts input data to YCbCr planes using full f32 precision.
    /// This matches C++ jpegli which uses float throughout the pipeline.
    /// Output values are in [0, 255] range (not level-shifted).
    fn convert_to_ycbcr_f32(&self, data: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let num_pixels = checked_size_2d(width, height)?;

        let mut y_plane = try_alloc_zeroed_f32(num_pixels, "YCbCr Y plane f32")?;
        let mut cb_plane = try_alloc_zeroed_f32(num_pixels, "YCbCr Cb plane f32")?;
        let mut cr_plane = try_alloc_zeroed_f32(num_pixels, "YCbCr Cr plane f32")?;

        match self.config.pixel_format {
            PixelFormat::Gray => {
                for i in 0..num_pixels {
                    y_plane[i] = data[i] as f32;
                    cb_plane[i] = 128.0;
                    cr_plane[i] = 128.0;
                }
            }
            PixelFormat::Rgb => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 3] as f32,
                        data[i * 3 + 1] as f32,
                        data[i * 3 + 2] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Rgba => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 4] as f32,
                        data[i * 4 + 1] as f32,
                        data[i * 4 + 2] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Bgr => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 3 + 2] as f32,
                        data[i * 3 + 1] as f32,
                        data[i * 3] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Bgra => {
                for i in 0..num_pixels {
                    let (y, cb, cr) = color::rgb_to_ycbcr_f32(
                        data[i * 4 + 2] as f32,
                        data[i * 4 + 1] as f32,
                        data[i * 4] as f32,
                    );
                    y_plane[i] = y;
                    cb_plane[i] = cb;
                    cr_plane[i] = cr;
                }
            }
            PixelFormat::Cmyk => {
                return Err(Error::UnsupportedFeature {
                    feature: "CMYK encoding",
                });
            }
        }

        Ok((y_plane, cb_plane, cr_plane))
    }

    /// Writes the JPEG header (SOI only, no JFIF APP0).
    ///
    /// Note: C++ jpegli does not write JFIF APP0, so we skip it for parity.
    /// The JFIF marker is optional and many modern decoders don't require it.
    fn write_header(&self, output: &mut Vec<u8>) -> Result<()> {
        // SOI only - no JFIF marker for C++ parity
        output.push(0xFF);
        output.push(MARKER_SOI);
        Ok(())
    }

    /// Writes the JPEG header for XYB mode (SOI only, no JFIF).
    ///
    /// XYB mode uses RGB component IDs and an ICC profile for color interpretation.
    /// JFIF APP0 is not appropriate because it implies YCbCr colorspace.
    fn write_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        // SOI only - no JFIF marker for XYB mode
        output.push(0xFF);
        output.push(MARKER_SOI);
        Ok(())
    }

    /// Writes an APP14 Adobe marker for RGB/CMYK/YCCK colorspaces.
    ///
    /// The APP14 marker is required by some decoders to properly interpret
    /// RGB (including XYB), CMYK, and YCCK colorspaces.
    ///
    /// See: https://github.com/google/jpegli/pull/135
    ///
    /// # Arguments
    /// * `transform` - Color transform type:
    ///   - 0 = RGB or CMYK (no transform)
    ///   - 1 = YCbCr
    ///   - 2 = YCCK
    fn write_app14_adobe(&self, output: &mut Vec<u8>, transform: u8) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_APP14);
        output.extend_from_slice(&[
            0x00, 0x0E, // Length: 14 bytes (includes length field)
            b'A', b'd', b'o', b'b', b'e', // Signature
            0x00, 0x64, // DCTEncodeVersion (100)
            0x00, 0x00, // APP14Flags0
            0x00, 0x00,      // APP14Flags1
            transform, // Color transform
        ]);
        Ok(())
    }

    /// Writes an ICC profile to the JPEG output.
    ///
    /// ICC profiles are stored in APP2 marker segments with the signature "ICC_PROFILE\0".
    /// Large profiles are split into multiple segments (max ~65519 bytes per segment).
    fn write_icc_profile(&self, output: &mut Vec<u8>, icc_data: &[u8]) -> Result<()> {
        if icc_data.is_empty() {
            return Ok(());
        }

        // Calculate number of chunks needed
        let num_chunks = (icc_data.len() + MAX_ICC_BYTES_PER_MARKER - 1) / MAX_ICC_BYTES_PER_MARKER;

        let mut offset = 0;
        for chunk_num in 0..num_chunks {
            let chunk_size = (icc_data.len() - offset).min(MAX_ICC_BYTES_PER_MARKER);

            // APP2 marker
            output.push(0xFF);
            output.push(MARKER_APP2);

            // Length: 2 (length field) + 12 (signature) + 2 (chunk info) + data
            let segment_length = 2 + 12 + 2 + chunk_size;
            output.push((segment_length >> 8) as u8);
            output.push(segment_length as u8);

            // ICC_PROFILE signature
            output.extend_from_slice(&ICC_PROFILE_SIGNATURE);

            // Chunk number (1-based) and total chunks
            output.push((chunk_num + 1) as u8);
            output.push(num_chunks as u8);

            // ICC data chunk
            output.extend_from_slice(&icc_data[offset..offset + chunk_size]);

            offset += chunk_size;
        }

        Ok(())
    }

    /// Writes quantization tables (3 separate tables for Y, Cb, Cr).
    /// This matches C++ jpegli behavior with add_two_chroma_tables=true.
    fn write_quant_tables(
        &self,
        output: &mut Vec<u8>,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<()> {
        // Write all 3 tables in one DQT segment
        // Length = 2 + 3 * (1 + 64) = 197 bytes
        output.push(0xFF);
        output.push(MARKER_DQT);
        output.push(0x00);
        output.push(0xC5); // Length: 197 bytes

        // Table 0 (Y) - values must be written in zigzag order
        output.push(0x00); // 8-bit precision, table 0
        for i in 0..DCT_BLOCK_SIZE {
            output.push(y_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 1 (Cb)
        output.push(0x01); // 8-bit precision, table 1
        for i in 0..DCT_BLOCK_SIZE {
            output.push(cb_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 2 (Cr)
        output.push(0x02); // 8-bit precision, table 2
        for i in 0..DCT_BLOCK_SIZE {
            output.push(cr_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        Ok(())
    }

    /// Writes quantization tables for XYB mode (3 separate tables).
    fn write_quant_tables_xyb(
        &self,
        output: &mut Vec<u8>,
        r_quant: &QuantTable,
        g_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<()> {
        // Write all 3 tables in one DQT segment
        // Length = 2 + 3 * (1 + 64) = 197 bytes
        output.push(0xFF);
        output.push(MARKER_DQT);
        output.push(0x00);
        output.push(0xC5); // Length: 197 bytes

        // Table 0 (Red)
        output.push(0x00); // 8-bit precision, table 0
        for i in 0..DCT_BLOCK_SIZE {
            output.push(r_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 1 (Green)
        output.push(0x01); // 8-bit precision, table 1
        for i in 0..DCT_BLOCK_SIZE {
            output.push(g_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        // Table 2 (Blue)
        output.push(0x02); // 8-bit precision, table 2
        for i in 0..DCT_BLOCK_SIZE {
            output.push(b_quant.values[JPEG_NATURAL_ORDER[i] as usize] as u8);
        }

        Ok(())
    }

    /// Writes the frame header (SOF0).
    fn write_frame_header(&self, output: &mut Vec<u8>) -> Result<()> {
        let marker = if self.config.mode == JpegMode::Progressive {
            MARKER_SOF2
        } else {
            MARKER_SOF0
        };

        output.push(0xFF);
        output.push(marker);

        let num_components = if self.config.pixel_format == PixelFormat::Gray {
            1u8
        } else {
            3u8
        };

        let length = 8u16 + num_components as u16 * 3;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(8); // Sample precision
        output.push((self.config.height >> 8) as u8);
        output.push(self.config.height as u8);
        output.push((self.config.width >> 8) as u8);
        output.push(self.config.width as u8);
        output.push(num_components);

        if num_components == 1 {
            // Grayscale
            output.push(1); // Component ID
            output.push(0x11); // 1x1 sampling
            output.push(0); // Quant table 0
        } else {
            // Y component
            let (h_samp, v_samp) = match self.config.subsampling {
                Subsampling::S444 => (1, 1),
                Subsampling::S422 => (2, 1),
                Subsampling::S420 => (2, 2),
                Subsampling::S440 => (1, 2),
            };

            output.push(1); // Component ID = 1 (Y)
            output.push((h_samp << 4) | v_samp);
            output.push(0); // Quant table 0

            output.push(2); // Component ID = 2 (Cb)
            output.push(0x11); // 1x1 sampling
            output.push(1); // Quant table 1

            output.push(3); // Component ID = 3 (Cr)
            output.push(0x11); // 1x1 sampling
            output.push(2); // Quant table 2 (separate Cr table like C++ cjpegli)
        }

        Ok(())
    }

    /// Writes the frame header for XYB mode (RGB with B subsampling).
    fn write_frame_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOF0); // Baseline DCT

        // 3 components: R, G, B
        let length = 8u16 + 3 * 3; // 17 bytes
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(8); // Sample precision
        output.push((self.config.height >> 8) as u8);
        output.push(self.config.height as u8);
        output.push((self.config.width >> 8) as u8);
        output.push(self.config.width as u8);
        output.push(3); // Number of components

        // XYB sampling: R:2×2, G:2×2, B:1×1
        // This means R and G are full resolution, B is 1/4 resolution
        output.push(b'R'); // Component ID = 'R' (82)
        output.push(0x22); // 2x2 sampling
        output.push(0); // Quant table 0

        output.push(b'G'); // Component ID = 'G' (71)
        output.push(0x22); // 2x2 sampling
        output.push(1); // Quant table 1

        output.push(b'B'); // Component ID = 'B' (66)
        output.push(0x11); // 1x1 sampling (subsampled)
        output.push(2); // Quant table 2

        Ok(())
    }

    /// Writes standard Huffman tables in a single DHT segment.
    fn write_huffman_tables(&self, output: &mut Vec<u8>) -> Result<()> {
        use crate::huffman::{
            STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_AC_LUMINANCE_BITS,
            STD_AC_LUMINANCE_VALUES, STD_DC_CHROMINANCE_BITS, STD_DC_CHROMINANCE_VALUES,
            STD_DC_LUMINANCE_BITS, STD_DC_LUMINANCE_VALUES,
        };

        // Write all 4 Huffman tables in a single DHT segment (like C++ jpegli)
        output.push(0xFF);
        output.push(MARKER_DHT);

        // Calculate total length
        let total_len = 2
            + (1 + 16 + STD_DC_LUMINANCE_VALUES.len())
            + (1 + 16 + STD_AC_LUMINANCE_VALUES.len())
            + (1 + 16 + STD_DC_CHROMINANCE_VALUES.len())
            + (1 + 16 + STD_AC_CHROMINANCE_VALUES.len());

        output.push((total_len >> 8) as u8);
        output.push(total_len as u8);

        // DC luminance (class 0, id 0)
        output.push(0x00);
        output.extend_from_slice(&STD_DC_LUMINANCE_BITS);
        output.extend_from_slice(&STD_DC_LUMINANCE_VALUES);

        // AC luminance (class 1, id 0)
        output.push(0x10);
        output.extend_from_slice(&STD_AC_LUMINANCE_BITS);
        output.extend_from_slice(&STD_AC_LUMINANCE_VALUES);

        // DC chrominance (class 0, id 1)
        output.push(0x01);
        output.extend_from_slice(&STD_DC_CHROMINANCE_BITS);
        output.extend_from_slice(&STD_DC_CHROMINANCE_VALUES);

        // AC chrominance (class 1, id 1)
        output.push(0x11);
        output.extend_from_slice(&STD_AC_CHROMINANCE_BITS);
        output.extend_from_slice(&STD_AC_CHROMINANCE_VALUES);

        Ok(())
    }

    /// Writes optimized Huffman tables.
    ///
    /// This is used when `optimize_huffman` is enabled to write the
    /// image-specific optimized tables to the DHT markers.
    fn write_huffman_tables_optimized(
        &self,
        output: &mut Vec<u8>,
        tables: &OptimizedHuffmanTables,
    ) -> Result<()> {
        // Write all 4 Huffman tables in a single DHT segment (like C++ jpegli)
        // This saves 12 bytes compared to 4 separate segments
        output.push(0xFF);
        output.push(MARKER_DHT);

        // Calculate total length: 2 (length field) + 4 tables × (1 + 16 + values.len())
        let total_len = 2
            + (1 + 16 + tables.dc_luma.values.len())
            + (1 + 16 + tables.ac_luma.values.len())
            + (1 + 16 + tables.dc_chroma.values.len())
            + (1 + 16 + tables.ac_chroma.values.len());

        output.push((total_len >> 8) as u8);
        output.push(total_len as u8);

        // DC luminance (class 0, id 0)
        output.push(0x00);
        output.extend_from_slice(&tables.dc_luma.bits);
        output.extend_from_slice(&tables.dc_luma.values);

        // AC luminance (class 1, id 0)
        output.push(0x10);
        output.extend_from_slice(&tables.ac_luma.bits);
        output.extend_from_slice(&tables.ac_luma.values);

        // DC chrominance (class 0, id 1)
        output.push(0x01);
        output.extend_from_slice(&tables.dc_chroma.bits);
        output.extend_from_slice(&tables.dc_chroma.values);

        // AC chrominance (class 1, id 1)
        output.push(0x11);
        output.extend_from_slice(&tables.ac_chroma.bits);
        output.extend_from_slice(&tables.ac_chroma.values);

        Ok(())
    }

    /// Writes restart interval.
    fn write_restart_interval(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_DRI);
        output.push(0x00);
        output.push(0x04); // Length
        output.push((self.config.restart_interval >> 8) as u8);
        output.push(self.config.restart_interval as u8);
        Ok(())
    }

    /// Writes scan header.
    fn write_scan_header(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        let num_components = if self.config.pixel_format == PixelFormat::Gray {
            1u8
        } else {
            3u8
        };

        let length = 6u16 + num_components as u16 * 2;
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(num_components);

        if num_components == 1 {
            output.push(1); // Component selector
            output.push(0x00); // DC/AC table selectors
        } else {
            output.push(1); // Y component
            output.push(0x00); // DC table 0, AC table 0

            output.push(2); // Cb component
            output.push(0x11); // DC table 1, AC table 1

            output.push(3); // Cr component
            output.push(0x11); // DC table 1, AC table 1
        }

        output.push(0x00); // Ss (spectral selection start)
        output.push(0x3F); // Se (spectral selection end = 63)
        output.push(0x00); // Ah/Al (successive approximation)

        Ok(())
    }

    /// Writes scan header for XYB mode.
    fn write_scan_header_xyb(&self, output: &mut Vec<u8>) -> Result<()> {
        output.push(0xFF);
        output.push(MARKER_SOS);

        // 3 components: R, G, B
        let length = 6u16 + 3 * 2; // 12 bytes
        output.push((length >> 8) as u8);
        output.push(length as u8);

        output.push(3); // Number of components

        // R component: DC table 0, AC table 0
        output.push(b'R');
        output.push(0x00);

        // G component: DC table 0, AC table 0
        output.push(b'G');
        output.push(0x00);

        // B component: DC table 0, AC table 0
        output.push(b'B');
        output.push(0x00);

        output.push(0x00); // Ss (spectral selection start)
        output.push(0x3F); // Se (spectral selection end = 63)
        output.push(0x00); // Ah/Al (successive approximation)

        Ok(())
    }

    /// Encodes the scan data (u8 version - legacy).
    #[allow(dead_code)]
    fn encode_scan(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        y_quant: &QuantTable,
        c_quant: &QuantTable,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
        encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
        encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;

        // For 4:2:0, process MCUs
        let _mcu_width = ((width + 15) / 16) * 16;
        let _mcu_height = ((height + 15) / 16) * 16;

        // TODO: Implement full MCU processing with subsampling
        // For now, simplified 4:4:4 encoding
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;

        // Zero-bias parameters for each component
        // Use effective distance inferred from quant tables (like C++ QuantValsToDistance)
        // For YCbCr mode, Cb and Cr share the same quant table (c_quant)
        let _input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, c_quant, c_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Convert Y plane to f32 for AQ computation
        let y_plane_f32: Vec<f32> = y_plane.iter().map(|&v| v as f32).collect();

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = if let Some(ref custom) = self.config.custom_aq_map {
            custom.clone()
        } else {
            compute_aq_strength_map(&y_plane_f32, width, height, y_quant_01)
        };
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(&y_plane_f32, width, height, y_quant_01);

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = if self.config.hybrid_config.enabled {
            Some(HybridQuantContext::new(self.config.hybrid_config))
        } else {
            None
        };

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Get per-block aq_strength (C++ AQ produces 0.0-0.2, mean ~0.08)
                let aq_strength = aq_map.get(bx, by);

                // Extract and encode Y block
                let y_block = self.extract_block(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                    ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, 1.0, true)
                } else {
                    quant::quantize_block_with_zero_bias(
                        &y_dct,
                        &y_quant.values,
                        &y_zero_bias,
                        aq_strength,
                    )
                };
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                );

                let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                encoder.encode_block(&y_zigzag, 0, 0, 0)?;

                if self.config.pixel_format != PixelFormat::Gray {
                    // Cb block
                    let cb_block = self.extract_block(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                        ctx.quantize_block(&cb_dct, &c_quant.values, aq_strength, 1.0, false)
                    } else {
                        quant::quantize_block_with_zero_bias(
                            &cb_dct,
                            &c_quant.values,
                            &cb_zero_bias,
                            aq_strength,
                        )
                    };
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cb_dct,
                        &c_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                    );

                    let cb_zigzag = natural_to_zigzag(&cb_quant_coeffs);
                    encoder.encode_block(&cb_zigzag, 1, 1, 1)?;

                    // Cr block
                    let cr_block = self.extract_block(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                        ctx.quantize_block(&cr_dct, &c_quant.values, aq_strength, 1.0, false)
                    } else {
                        quant::quantize_block_with_zero_bias(
                            &cr_dct,
                            &c_quant.values,
                            &cr_zero_bias,
                            aq_strength,
                        )
                    };
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cr_dct,
                        &c_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                    );

                    let cr_zigzag = natural_to_zigzag(&cr_quant_coeffs);
                    encoder.encode_block(&cr_zigzag, 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }

    /// Quantizes all blocks in the image.
    ///
    /// This is separated from encoding to allow Huffman optimization:
    /// 1. Quantize all blocks
    /// 2. Collect frequencies to build optimal tables
    /// 3. Encode with optimal tables
    fn quantize_all_blocks(
        &self,
        y_plane: &[f32],
        cb_plane: &[f32],
        cr_plane: &[f32],
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let blocks_h = (width + 7) / 8;
        let blocks_v = (height + 7) / 8;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Zero-bias parameters for each component
        // Use effective distance inferred from quant tables (like C++ QuantValsToDistance)
        // This is important at Q100 where quant values are all 1s but input distance is 0.01
        let _input_distance = self.config.quality.to_distance();
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Compute per-block adaptive quantization strength from Y plane
        // C++ uses y_quant_01 = quant_table[1] for dampen calculation
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = if let Some(ref custom) = self.config.custom_aq_map {
            custom.clone()
        } else {
            compute_aq_strength_map(y_plane, width, height, y_quant_01)
        };
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(y_plane, width, height, y_quant_01);

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = if self.config.hybrid_config.enabled {
            Some(HybridQuantContext::new(self.config.hybrid_config))
        } else {
            None
        };

        let mut y_blocks = Vec::with_capacity(blocks_h * blocks_v);
        let mut cb_blocks = Vec::with_capacity(if is_color { blocks_h * blocks_v } else { 0 });
        let mut cr_blocks = Vec::with_capacity(if is_color { blocks_h * blocks_v } else { 0 });

        for by in 0..blocks_v {
            for bx in 0..blocks_h {
                // Get per-block aq_strength
                let aq_strength = aq_map.get(bx, by);

                let y_block = self.extract_block_ycbcr_f32(y_plane, width, height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                    ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, 1.0, true)
                } else {
                    quant::quantize_block_with_zero_bias(
                        &y_dct,
                        &y_quant.values,
                        &y_zero_bias,
                        aq_strength,
                    )
                };
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                );

                y_blocks.push(natural_to_zigzag(&y_quant_coeffs));

                if is_color {
                    let cb_block = self.extract_block_ycbcr_f32(cb_plane, width, height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                        ctx.quantize_block(&cb_dct, &cb_quant.values, aq_strength, 1.0, false)
                    } else {
                        quant::quantize_block_with_zero_bias(
                            &cb_dct,
                            &cb_quant.values,
                            &cb_zero_bias,
                            aq_strength,
                        )
                    };
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cb_dct,
                        &cb_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                    );

                    cb_blocks.push(natural_to_zigzag(&cb_quant_coeffs));

                    let cr_block = self.extract_block_ycbcr_f32(cr_plane, width, height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                        ctx.quantize_block(&cr_dct, &cr_quant.values, aq_strength, 1.0, false)
                    } else {
                        quant::quantize_block_with_zero_bias(
                            &cr_dct,
                            &cr_quant.values,
                            &cr_zero_bias,
                            aq_strength,
                        )
                    };
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cr_dct,
                        &cr_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                    );

                    cr_blocks.push(natural_to_zigzag(&cr_quant_coeffs));
                }
            }
        }

        Ok((y_blocks, cb_blocks, cr_blocks))
    }

    /// Quantizes all blocks with subsampling support.
    ///
    /// Unlike `quantize_all_blocks`, this version handles different dimensions
    /// for Y and chroma planes (needed for 4:2:0, 4:2:2, 4:4:0 subsampling).
    #[allow(clippy::too_many_arguments)]
    fn quantize_all_blocks_subsampled(
        &self,
        y_plane: &[f32],
        y_width: usize,
        y_height: usize,
        cb_plane: &[f32],
        cr_plane: &[f32],
        c_width: usize,
        c_height: usize,
        y_quant: &QuantTable,
        cb_quant: &QuantTable,
        cr_quant: &QuantTable,
    ) -> Result<(
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    )> {
        let y_blocks_h = (y_width + 7) / 8;
        let y_blocks_v = (y_height + 7) / 8;
        let c_blocks_h = (c_width + 7) / 8;
        let c_blocks_v = (c_height + 7) / 8;
        let is_color = self.config.pixel_format != PixelFormat::Gray;

        // Zero-bias parameters for each component
        let effective_distance = quant::quant_vals_to_distance(y_quant, cb_quant, cr_quant);
        let y_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 0);
        let cb_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 1);
        let cr_zero_bias = ZeroBiasParams::for_ycbcr(effective_distance, 2);

        // Compute per-block adaptive quantization strength from Y plane
        let y_quant_01 = y_quant.values[1];
        #[cfg(feature = "experimental-hybrid-trellis")]
        let aq_map = if let Some(ref custom) = self.config.custom_aq_map {
            custom.clone()
        } else {
            compute_aq_strength_map(y_plane, y_width, y_height, y_quant_01)
        };
        #[cfg(not(feature = "experimental-hybrid-trellis"))]
        let aq_map = compute_aq_strength_map(y_plane, y_width, y_height, y_quant_01);

        // Create hybrid quantization context if enabled
        #[cfg(feature = "experimental-hybrid-trellis")]
        let hybrid_ctx = if self.config.hybrid_config.enabled {
            Some(HybridQuantContext::new(self.config.hybrid_config))
        } else {
            None
        };

        let mut y_blocks = Vec::with_capacity(y_blocks_h * y_blocks_v);
        let mut cb_blocks = Vec::with_capacity(if is_color { c_blocks_h * c_blocks_v } else { 0 });
        let mut cr_blocks = Vec::with_capacity(if is_color { c_blocks_h * c_blocks_v } else { 0 });

        // Quantize Y blocks
        for by in 0..y_blocks_v {
            for bx in 0..y_blocks_h {
                let aq_strength = aq_map.get(bx, by);
                let y_block = self.extract_block_ycbcr_f32(y_plane, y_width, y_height, bx, by);
                let y_dct = forward_dct_8x8(&y_block);

                #[cfg(feature = "experimental-hybrid-trellis")]
                let y_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                    ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, 1.0, true)
                } else {
                    quant::quantize_block_with_zero_bias(
                        &y_dct,
                        &y_quant.values,
                        &y_zero_bias,
                        aq_strength,
                    )
                };
                #[cfg(not(feature = "experimental-hybrid-trellis"))]
                let y_quant_coeffs = quant::quantize_block_with_zero_bias(
                    &y_dct,
                    &y_quant.values,
                    &y_zero_bias,
                    aq_strength,
                );

                y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
            }
        }

        // Quantize chroma blocks (from possibly downsampled planes)
        if is_color {
            for by in 0..c_blocks_v {
                for bx in 0..c_blocks_h {
                    // For chroma, use average AQ strength from corresponding Y region
                    // For 4:2:0, each chroma block corresponds to 2x2 Y blocks
                    let y_bx = (bx * y_blocks_h) / c_blocks_h;
                    let y_by = (by * y_blocks_v) / c_blocks_v;
                    let aq_strength =
                        aq_map.get(y_bx.min(y_blocks_h - 1), y_by.min(y_blocks_v - 1));

                    let cb_block =
                        self.extract_block_ycbcr_f32(cb_plane, c_width, c_height, bx, by);
                    let cb_dct = forward_dct_8x8(&cb_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cb_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                        ctx.quantize_block(&cb_dct, &cb_quant.values, aq_strength, 1.0, false)
                    } else {
                        quant::quantize_block_with_zero_bias(
                            &cb_dct,
                            &cb_quant.values,
                            &cb_zero_bias,
                            aq_strength,
                        )
                    };
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cb_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cb_dct,
                        &cb_quant.values,
                        &cb_zero_bias,
                        aq_strength,
                    );

                    cb_blocks.push(natural_to_zigzag(&cb_quant_coeffs));

                    let cr_block =
                        self.extract_block_ycbcr_f32(cr_plane, c_width, c_height, bx, by);
                    let cr_dct = forward_dct_8x8(&cr_block);

                    #[cfg(feature = "experimental-hybrid-trellis")]
                    let cr_quant_coeffs = if let Some(ref ctx) = hybrid_ctx {
                        ctx.quantize_block(&cr_dct, &cr_quant.values, aq_strength, 1.0, false)
                    } else {
                        quant::quantize_block_with_zero_bias(
                            &cr_dct,
                            &cr_quant.values,
                            &cr_zero_bias,
                            aq_strength,
                        )
                    };
                    #[cfg(not(feature = "experimental-hybrid-trellis"))]
                    let cr_quant_coeffs = quant::quantize_block_with_zero_bias(
                        &cr_dct,
                        &cr_quant.values,
                        &cr_zero_bias,
                        aq_strength,
                    );

                    cr_blocks.push(natural_to_zigzag(&cr_quant_coeffs));
                }
            }
        }

        Ok((y_blocks, cb_blocks, cr_blocks))
    }

    /// Builds optimized Huffman tables from quantized blocks.
    ///
    /// Collects symbol frequencies from all blocks and generates optimal
    /// Huffman tables with their DHT marker representations.
    ///
    /// For subsampled modes, this iterates blocks in MCU order to correctly
    /// account for padding blocks.
    fn build_optimized_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<OptimizedHuffmanTables> {
        let mut dc_luma_freq = FrequencyCounter::new();
        let mut dc_chroma_freq = FrequencyCounter::new();
        let mut ac_luma_freq = FrequencyCounter::new();
        let mut ac_chroma_freq = FrequencyCounter::new();

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        // Zero block for padding
        const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple iteration, no padding needed
            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            for (i, y_block) in y_blocks.iter().enumerate() {
                Self::collect_block_frequencies(
                    y_block,
                    prev_y_dc,
                    &mut dc_luma_freq,
                    &mut ac_luma_freq,
                );
                prev_y_dc = y_block[0];

                if is_color {
                    Self::collect_block_frequencies(
                        &cb_blocks[i],
                        prev_cb_dc,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    prev_cb_dc = cb_blocks[i][0];

                    Self::collect_block_frequencies(
                        &cr_blocks[i],
                        prev_cr_dc,
                        &mut dc_chroma_freq,
                        &mut ac_chroma_freq,
                    );
                    prev_cr_dc = cr_blocks[i][0];
                }
            }
        } else {
            // Subsampled mode - iterate in MCU order with padding
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;
            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            let mut prev_y_dc: i16 = 0;
            let mut prev_cb_dc: i16 = 0;
            let mut prev_cr_dc: i16 = 0;

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Y blocks in this MCU
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            let block = if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                &y_blocks[y_idx]
                            } else {
                                &ZERO_BLOCK
                            };
                            Self::collect_block_frequencies(
                                block,
                                prev_y_dc,
                                &mut dc_luma_freq,
                                &mut ac_luma_freq,
                            );
                            prev_y_dc = block[0];
                        }
                    }

                    // Chroma blocks
                    if is_color {
                        let (cb_block, cr_block) = if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            (&cb_blocks[c_idx], &cr_blocks[c_idx])
                        } else {
                            (&ZERO_BLOCK, &ZERO_BLOCK)
                        };

                        Self::collect_block_frequencies(
                            cb_block,
                            prev_cb_dc,
                            &mut dc_chroma_freq,
                            &mut ac_chroma_freq,
                        );
                        prev_cb_dc = cb_block[0];

                        Self::collect_block_frequencies(
                            cr_block,
                            prev_cr_dc,
                            &mut dc_chroma_freq,
                            &mut ac_chroma_freq,
                        );
                        prev_cr_dc = cr_block[0];
                    }
                }
            }
        }

        // Build optimized tables with DHT data
        let dc_luma = dc_luma_freq.generate_table_with_dht()?;
        let ac_luma = ac_luma_freq.generate_table_with_dht()?;

        let (dc_chroma, ac_chroma) = if is_color {
            (
                dc_chroma_freq.generate_table_with_dht()?,
                ac_chroma_freq.generate_table_with_dht()?,
            )
        } else {
            // Use standard tables for grayscale (won't be used but needed for structure)
            use crate::huffman::{
                STD_AC_CHROMINANCE_BITS, STD_AC_CHROMINANCE_VALUES, STD_DC_CHROMINANCE_BITS,
                STD_DC_CHROMINANCE_VALUES,
            };
            use crate::huffman_opt::OptimizedTable;

            (
                OptimizedTable {
                    table: HuffmanEncodeTable::std_dc_chrominance(),
                    bits: STD_DC_CHROMINANCE_BITS,
                    values: STD_DC_CHROMINANCE_VALUES.to_vec(),
                },
                OptimizedTable {
                    table: HuffmanEncodeTable::std_ac_chrominance(),
                    bits: STD_AC_CHROMINANCE_BITS,
                    values: STD_AC_CHROMINANCE_VALUES.to_vec(),
                },
            )
        };

        Ok(OptimizedHuffmanTables {
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
        })
    }

    /// Encodes blocks using optimized Huffman tables.
    ///
    /// Handles MCU interleaving for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    fn encode_with_tables(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
        tables: &OptimizedHuffmanTables,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        encoder.set_dc_table(0, tables.dc_luma.table.clone());
        encoder.set_ac_table(0, tables.ac_luma.table.clone());
        encoder.set_dc_table(1, tables.dc_chroma.table.clone());
        encoder.set_ac_table(1, tables.ac_chroma.table.clone());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0)?;

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        } else {
            // Subsampled mode - MCU interleaving
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;

            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            // Zero block for padding out-of-bounds MCU positions
            const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Encode Y blocks in this MCU (must encode all 4 even if out of bounds)
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                encoder.encode_block(&y_blocks[y_idx], 0, 0, 0)?;
                            } else {
                                // Out of bounds - encode zero block (padding)
                                encoder.encode_block(&ZERO_BLOCK, 0, 0, 0)?;
                            }
                        }
                    }

                    // Encode Cb and Cr blocks (always, even if out of bounds)
                    if is_color {
                        if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1)?;
                            encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1)?;
                        } else {
                            // Out of bounds - encode zero blocks (padding)
                            encoder.encode_block(&ZERO_BLOCK, 1, 1, 1)?;
                            encoder.encode_block(&ZERO_BLOCK, 2, 1, 1)?;
                        }
                    }

                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Encodes blocks using standard (fixed) Huffman tables - single pass.
    ///
    /// Handles MCU interleaving for subsampled modes (4:2:0, 4:2:2, 4:4:0).
    fn encode_blocks_standard(
        &self,
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cb_blocks: &[[i16; DCT_BLOCK_SIZE]],
        cr_blocks: &[[i16; DCT_BLOCK_SIZE]],
        is_color: bool,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
        encoder.set_dc_table(1, HuffmanEncodeTable::std_dc_chrominance());
        encoder.set_ac_table(1, HuffmanEncodeTable::std_ac_chrominance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let (h_samp, v_samp) = match self.config.subsampling {
            Subsampling::S444 => (1, 1),
            Subsampling::S422 => (2, 1),
            Subsampling::S420 => (2, 2),
            Subsampling::S440 => (1, 2),
        };

        if h_samp == 1 && v_samp == 1 {
            // 4:4:4 mode - simple 1:1 interleaving
            for (i, y_block) in y_blocks.iter().enumerate() {
                encoder.encode_block(y_block, 0, 0, 0)?;

                if is_color {
                    encoder.encode_block(&cb_blocks[i], 1, 1, 1)?;
                    encoder.encode_block(&cr_blocks[i], 2, 1, 1)?;
                }

                encoder.check_restart();
            }
        } else {
            // Subsampled mode - MCU interleaving
            let y_blocks_h = (width + 7) / 8;
            let y_blocks_v = (height + 7) / 8;
            // Use ceiling division for chroma dimensions: (n + d - 1) / d
            let c_width = (width + h_samp - 1) / h_samp;
            let c_height = (height + v_samp - 1) / v_samp;
            let c_blocks_h = (c_width + 7) / 8;
            let c_blocks_v = (c_height + 7) / 8;

            // MCU dimensions in terms of Y blocks
            let mcu_h = (y_blocks_h + h_samp - 1) / h_samp;
            let mcu_v = (y_blocks_v + v_samp - 1) / v_samp;

            // Zero block for padding out-of-bounds MCU positions
            const ZERO_BLOCK: [i16; DCT_BLOCK_SIZE] = [0i16; DCT_BLOCK_SIZE];

            for mcu_y in 0..mcu_v {
                for mcu_x in 0..mcu_h {
                    // Encode Y blocks in this MCU (must encode all even if out of bounds)
                    for dy in 0..v_samp {
                        for dx in 0..h_samp {
                            let y_bx = mcu_x * h_samp + dx;
                            let y_by = mcu_y * v_samp + dy;
                            if y_bx < y_blocks_h && y_by < y_blocks_v {
                                let y_idx = y_by * y_blocks_h + y_bx;
                                encoder.encode_block(&y_blocks[y_idx], 0, 0, 0)?;
                            } else {
                                // Out of bounds - encode zero block (padding)
                                encoder.encode_block(&ZERO_BLOCK, 0, 0, 0)?;
                            }
                        }
                    }

                    // Encode Cb and Cr blocks (always, even if out of bounds)
                    if is_color {
                        if mcu_x < c_blocks_h && mcu_y < c_blocks_v {
                            let c_idx = mcu_y * c_blocks_h + mcu_x;
                            encoder.encode_block(&cb_blocks[c_idx], 1, 1, 1)?;
                            encoder.encode_block(&cr_blocks[c_idx], 2, 1, 1)?;
                        } else {
                            // Out of bounds - encode zero blocks (padding)
                            encoder.encode_block(&ZERO_BLOCK, 1, 1, 1)?;
                            encoder.encode_block(&ZERO_BLOCK, 2, 1, 1)?;
                        }
                    }

                    encoder.check_restart();
                }
            }
        }

        Ok(encoder.finish())
    }

    /// Collects symbol frequencies from a block for Huffman optimization.
    fn collect_block_frequencies(
        coeffs: &[i16; DCT_BLOCK_SIZE],
        prev_dc: i16,
        dc_freq: &mut FrequencyCounter,
        ac_freq: &mut FrequencyCounter,
    ) {
        // DC coefficient - limit category to 11 for 8-bit JPEG compatibility
        let dc_diff = coeffs[0] - prev_dc;
        let dc_category = entropy::category(dc_diff).min(11);
        dc_freq.count(dc_category);

        // AC coefficients
        let mut run = 0u8;
        for i in 1..DCT_BLOCK_SIZE {
            let ac = coeffs[i];

            if ac == 0 {
                run += 1;
            } else {
                // Encode runs of 16 zeros (ZRL)
                while run >= 16 {
                    ac_freq.count(0xF0);
                    run -= 16;
                }

                // Encode run/size symbol
                let ac_category = entropy::category(ac);
                let symbol = (run << 4) | ac_category;
                ac_freq.count(symbol);
                run = 0;
            }
        }

        // EOB if trailing zeros
        if run > 0 {
            ac_freq.count(0x00);
        }
    }

    /// Quantizes all XYB blocks for Huffman optimization.
    ///
    /// Returns quantized blocks for X, Y, and B components.
    /// B component is already downsampled (half resolution).
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Reserved for future XYB encoding improvements
    fn quantize_all_blocks_xyb(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> (
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    ) {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        let mut x_blocks = Vec::with_capacity(num_xy_blocks);
        let mut y_blocks = Vec::with_capacity(num_xy_blocks);
        let mut b_blocks = Vec::with_capacity(num_b_blocks);

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let x_block = self.extract_block_f32(x_plane, width, height, bx, by);
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block(&x_dct, &x_quant.values);
                        x_blocks.push(natural_to_zigzag(&x_quant_coeffs));
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let y_block = self.extract_block_f32(y_plane, width, height, bx, by);
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = self.extract_block_f32(b_plane, b_width, b_height, mcu_x, mcu_y);
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
                b_blocks.push(natural_to_zigzag(&b_quant_coeffs));
            }
        }

        (x_blocks, y_blocks, b_blocks)
    }

    /// Quantizes all XYB blocks with jpegli-style adaptive quantization (no trellis).
    ///
    /// This version uses the AQ map for per-block modulation with zero-bias,
    /// matching jpegli's default AQ behavior without hybrid trellis.
    ///
    /// For XYB mode:
    /// - X and Y use luma tables (both are full-resolution "luma-like" channels)
    /// - B uses chroma tables (downsampled blue channel)
    #[allow(clippy::too_many_arguments)]
    fn quantize_all_blocks_xyb_with_aq_simple(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
        aq_map: &crate::adaptive_quant::AQStrengthMap,
        x_zero_bias: &ZeroBiasParams,
        y_zero_bias: &ZeroBiasParams,
        b_zero_bias: &ZeroBiasParams,
    ) -> (
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    ) {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        let mut x_blocks = Vec::with_capacity(num_xy_blocks);
        let mut y_blocks = Vec::with_capacity(num_xy_blocks);
        let mut b_blocks = Vec::with_capacity(num_b_blocks);

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let aq_strength = aq_map.get(bx, by);

                        let x_block = self.extract_block_f32(x_plane, width, height, bx, by);
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block_with_zero_bias(
                            &x_dct,
                            &x_quant.values,
                            x_zero_bias,
                            aq_strength,
                        );
                        x_blocks.push(natural_to_zigzag(&x_quant_coeffs));
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let aq_strength = aq_map.get(bx, by);

                        let y_block = self.extract_block_f32(y_plane, width, height, bx, by);
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block_with_zero_bias(
                            &y_dct,
                            &y_quant.values,
                            y_zero_bias,
                            aq_strength,
                        );
                        y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
                    }
                }

                // Process 1 B block (from downsampled plane)
                // For B channel: Average AQ from 4 parent full-res blocks
                let b_aq_strength = {
                    let mut sum = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let bx = mcu_x * 2 + dx;
                            let by = mcu_y * 2 + dy;
                            sum += aq_map.get(bx, by);
                        }
                    }
                    sum / 4.0
                };

                let b_block = self.extract_block_f32(b_plane, b_width, b_height, mcu_x, mcu_y);
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block_with_zero_bias(
                    &b_dct,
                    &b_quant.values,
                    b_zero_bias,
                    b_aq_strength,
                );
                b_blocks.push(natural_to_zigzag(&b_quant_coeffs));
            }
        }

        (x_blocks, y_blocks, b_blocks)
    }

    /// Quantizes all XYB blocks with adaptive quantization support.
    ///
    /// This version uses the AQ map for per-block modulation and optionally
    /// applies hybrid trellis quantization when enabled.
    ///
    /// For XYB mode:
    /// - X and Y use luma tables (both are full-resolution "luma-like" channels)
    /// - B uses chroma tables (downsampled blue channel)
    #[cfg(feature = "experimental-hybrid-trellis")]
    #[allow(clippy::too_many_arguments)]
    fn quantize_all_blocks_xyb_with_aq(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
        aq_map: &crate::adaptive_quant::AQStrengthMap,
        hybrid_ctx: Option<&HybridQuantContext>,
    ) -> (
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
        Vec<[i16; DCT_BLOCK_SIZE]>,
    ) {
        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;
        let num_xy_blocks = mcu_cols * mcu_rows * 4; // 4 blocks per MCU for X and Y
        let num_b_blocks = mcu_cols * mcu_rows; // 1 block per MCU for B

        let mut x_blocks = Vec::with_capacity(num_xy_blocks);
        let mut y_blocks = Vec::with_capacity(num_xy_blocks);
        let mut b_blocks = Vec::with_capacity(num_b_blocks);

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let aq_strength = aq_map.get(bx, by);

                        let x_block = self.extract_block_f32(x_plane, width, height, bx, by);
                        let x_dct = forward_dct_8x8(&x_block);

                        // X is luma-like in XYB, dampen=1.0
                        let x_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                            ctx.quantize_block(&x_dct, &x_quant.values, aq_strength, 1.0, true)
                        } else {
                            quant::quantize_block(&x_dct, &x_quant.values)
                        };
                        x_blocks.push(natural_to_zigzag(&x_quant_coeffs));
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let aq_strength = aq_map.get(bx, by);

                        let y_block = self.extract_block_f32(y_plane, width, height, bx, by);
                        let y_dct = forward_dct_8x8(&y_block);

                        // Y is the primary luma channel in XYB, dampen=1.0
                        let y_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                            ctx.quantize_block(&y_dct, &y_quant.values, aq_strength, 1.0, true)
                        } else {
                            quant::quantize_block(&y_dct, &y_quant.values)
                        };
                        y_blocks.push(natural_to_zigzag(&y_quant_coeffs));
                    }
                }

                // Process 1 B block (from downsampled plane)
                // Average AQ from the 4 corresponding full-res blocks
                let b_aq_strength = {
                    let mut sum = 0.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let bx = mcu_x * 2 + dx;
                            let by = mcu_y * 2 + dy;
                            sum += aq_map.get(bx, by);
                        }
                    }
                    sum / 4.0
                };

                let b_block = self.extract_block_f32(b_plane, b_width, b_height, mcu_x, mcu_y);
                let b_dct = forward_dct_8x8(&b_block);

                // B is chroma-like (blue channel), is_luma=false
                let b_quant_coeffs = if let Some(ctx) = hybrid_ctx {
                    ctx.quantize_block(&b_dct, &b_quant.values, b_aq_strength, 1.0, false)
                } else {
                    quant::quantize_block(&b_dct, &b_quant.values)
                };
                b_blocks.push(natural_to_zigzag(&b_quant_coeffs));
            }
        }

        (x_blocks, y_blocks, b_blocks)
    }

    /// Builds optimized Huffman tables for XYB mode.
    ///
    /// XYB uses a single shared table for all components (luminance tables).
    /// Returns the optimized DC and AC tables.
    fn build_optimized_tables_xyb(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
    ) -> Result<(
        crate::huffman_opt::OptimizedTable,
        crate::huffman_opt::OptimizedTable,
    )> {
        let mut dc_freq = FrequencyCounter::new();
        let mut ac_freq = FrequencyCounter::new();

        // Collect frequencies from all components
        // Note: XYB MCU order is 4 X blocks, 4 Y blocks, 1 B block per MCU
        // But since all share the same table, we just iterate through them

        // In XYB mode, we have interleaved blocks per MCU:
        // [X0, X1, X2, X3, Y0, Y1, Y2, Y3, B0] per MCU
        // DC prediction carries across MCUs for each component (standard JPEG behavior)

        let mcu_count = b_blocks.len();

        // Each component maintains its own DC prediction across all MCUs
        let mut prev_dc_x: i16 = 0;
        let mut prev_dc_y: i16 = 0;
        let mut prev_dc_b: i16 = 0;

        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                let block = &x_blocks[x_start + i];
                Self::collect_block_frequencies(block, prev_dc_x, &mut dc_freq, &mut ac_freq);
                prev_dc_x = block[0];
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                let block = &y_blocks[y_start + i];
                Self::collect_block_frequencies(block, prev_dc_y, &mut dc_freq, &mut ac_freq);
                prev_dc_y = block[0];
            }

            // B block (1 per MCU)
            Self::collect_block_frequencies(
                &b_blocks[mcu_idx],
                prev_dc_b,
                &mut dc_freq,
                &mut ac_freq,
            );
            prev_dc_b = b_blocks[mcu_idx][0];
        }

        // Generate optimized tables
        let dc_table = dc_freq.generate_table_with_dht()?;
        let ac_table = ac_freq.generate_table_with_dht()?;

        Ok((dc_table, ac_table))
    }

    /// Encodes XYB blocks using optimized Huffman tables.
    #[allow(clippy::too_many_arguments)]
    fn encode_with_tables_xyb(
        &self,
        x_blocks: &[[i16; DCT_BLOCK_SIZE]],
        y_blocks: &[[i16; DCT_BLOCK_SIZE]],
        b_blocks: &[[i16; DCT_BLOCK_SIZE]],
        dc_table: &crate::huffman_opt::OptimizedTable,
        ac_table: &crate::huffman_opt::OptimizedTable,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Use the same optimized table for all components
        encoder.set_dc_table(0, dc_table.table.clone());
        encoder.set_ac_table(0, ac_table.table.clone());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        let mcu_count = b_blocks.len();
        for mcu_idx in 0..mcu_count {
            // X blocks (4 per MCU)
            let x_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&x_blocks[x_start + i], 0, 0, 0)?;
            }

            // Y blocks (4 per MCU)
            let y_start = mcu_idx * 4;
            for i in 0..4 {
                encoder.encode_block(&y_blocks[y_start + i], 1, 0, 0)?;
            }

            // B block (1 per MCU)
            encoder.encode_block(&b_blocks[mcu_idx], 2, 0, 0)?;

            encoder.check_restart();
        }

        Ok(encoder.finish())
    }

    /// Writes DHT markers for XYB optimized tables.
    fn write_huffman_tables_xyb_optimized(
        &self,
        output: &mut Vec<u8>,
        dc_table: &crate::huffman_opt::OptimizedTable,
        ac_table: &crate::huffman_opt::OptimizedTable,
    ) {
        let write_table = |out: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 16], values: &[u8]| {
            out.push(0xFF);
            out.push(MARKER_DHT);
            let length = 2 + 1 + 16 + values.len();
            out.push((length >> 8) as u8);
            out.push(length as u8);
            out.push((class << 4) | id);
            out.extend_from_slice(bits);
            out.extend_from_slice(values);
        };

        // DC table (class=0, id=0)
        write_table(output, 0, 0, &dc_table.bits, &dc_table.values);
        // AC table (class=1, id=0)
        write_table(output, 1, 0, &ac_table.bits, &ac_table.values);
    }

    /// Encodes scan data for XYB mode with float planes.
    ///
    /// Uses scaled XYB values (in [0, 1] range), converts to [0, 255],
    /// then level shifts by subtracting 128 before DCT.
    #[allow(clippy::too_many_arguments)]
    fn encode_scan_xyb_float(
        &self,
        x_plane: &[f32],
        y_plane: &[f32],
        b_plane: &[f32], // Already downsampled
        width: usize,
        height: usize,
        b_width: usize,
        b_height: usize,
        x_quant: &QuantTable,
        y_quant: &QuantTable,
        b_quant: &QuantTable,
    ) -> Result<Vec<u8>> {
        let mut encoder = EntropyEncoder::new();

        // Set up Huffman tables - use luminance tables for all components in XYB mode
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());

        if self.config.restart_interval > 0 {
            encoder.set_restart_interval(self.config.restart_interval);
        }

        // MCU size for 2×2, 2×2, 1×1 sampling: 16×16 pixels
        // Each MCU contains: 4 X blocks + 4 Y blocks + 1 B block = 9 blocks
        let mcu_cols = (width + 15) / 16;
        let mcu_rows = (height + 15) / 16;

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                // Process 4 X blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let x_block = self.extract_block_f32(x_plane, width, height, bx, by);
                        let x_dct = forward_dct_8x8(&x_block);
                        let x_quant_coeffs = quant::quantize_block(&x_dct, &x_quant.values);
                        let x_zigzag = natural_to_zigzag(&x_quant_coeffs);
                        encoder.encode_block(&x_zigzag, 0, 0, 0)?;
                    }
                }

                // Process 4 Y blocks (2×2 arrangement within 16×16 MCU)
                for block_y in 0..2 {
                    for block_x in 0..2 {
                        let bx = mcu_x * 2 + block_x;
                        let by = mcu_y * 2 + block_y;
                        let y_block = self.extract_block_f32(y_plane, width, height, bx, by);
                        let y_dct = forward_dct_8x8(&y_block);
                        let y_quant_coeffs = quant::quantize_block(&y_dct, &y_quant.values);
                        let y_zigzag = natural_to_zigzag(&y_quant_coeffs);
                        encoder.encode_block(&y_zigzag, 1, 0, 0)?;
                    }
                }

                // Process 1 B block (from downsampled plane)
                let b_block = self.extract_block_f32(b_plane, b_width, b_height, mcu_x, mcu_y);
                let b_dct = forward_dct_8x8(&b_block);
                let b_quant_coeffs = quant::quantize_block(&b_dct, &b_quant.values);
                let b_zigzag = natural_to_zigzag(&b_quant_coeffs);
                encoder.encode_block(&b_zigzag, 2, 0, 0)?;

                encoder.check_restart();
            }
        }

        Ok(encoder.finish())
    }

    /// Extracts an 8x8 block from a float plane (scaled XYB values).
    ///
    /// Scaled XYB values are in [0, 1] range. This method:
    /// 1. Multiplies by 255 to get to [0, 255] range
    /// 2. Subtracts 128 for level shifting (DCT input is [-128, 127])
    fn extract_block_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Scale from [0, 1] to [0, 255], then level shift by -128
                block[y * DCT_SIZE + x] = plane[idx] * 255.0 - 128.0;
            }
        }

        block
    }

    /// Extracts an 8x8 block from a u8 plane with level shift.
    #[allow(dead_code)]
    fn extract_block(
        &self,
        plane: &[u8],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Level shift: subtract 128
                block[y * DCT_SIZE + x] = plane[idx] as f32 - 128.0;
            }
        }

        block
    }

    /// Extracts an 8x8 block from a YCbCr f32 plane with level shift.
    /// Input values are in [0, 255] range, output is level-shifted by -128.
    fn extract_block_ycbcr_f32(
        &self,
        plane: &[f32],
        width: usize,
        height: usize,
        bx: usize,
        by: usize,
    ) -> [f32; DCT_BLOCK_SIZE] {
        let mut block = [0.0f32; DCT_BLOCK_SIZE];

        for y in 0..DCT_SIZE {
            for x in 0..DCT_SIZE {
                let px = (bx * DCT_SIZE + x).min(width - 1);
                let py = (by * DCT_SIZE + y).min(height - 1);
                let idx = py * width + px;
                // Level shift: subtract 128 (values are already in [0, 255])
                block[y * DCT_SIZE + x] = plane[idx] - 128.0;
            }
        }

        block
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts coefficients from natural order to zigzag order for JPEG encoding.
fn natural_to_zigzag(natural: &[i16; DCT_BLOCK_SIZE]) -> [i16; DCT_BLOCK_SIZE] {
    let mut zigzag = [0i16; DCT_BLOCK_SIZE];
    for i in 0..DCT_BLOCK_SIZE {
        zigzag[JPEG_ZIGZAG_ORDER[i] as usize] = natural[i];
    }
    zigzag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = Encoder::new()
            .width(640)
            .height(480)
            .jpegli_quality(Quality::from_quality(90.0));

        assert_eq!(encoder.config.width, 640);
        assert_eq!(encoder.config.height, 480);
    }

    #[test]
    fn test_encoder_validation() {
        let encoder = Encoder::new();
        assert!(encoder.validate().is_err());

        let encoder = Encoder::new().width(100).height(100);
        assert!(encoder.validate().is_ok());
    }

    #[test]
    fn test_encode_small_gray() {
        let encoder = Encoder::new()
            .width(8)
            .height(8)
            .pixel_format(PixelFormat::Gray)
            .jpegli_quality(Quality::from_quality(90.0));

        let data = vec![128u8; 64];
        let result = encoder.encode(&data);
        assert!(result.is_ok());

        let jpeg = result.unwrap();
        // Should start with SOI
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        // Should end with EOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);
    }

    #[test]
    fn test_encode_rgb_xyb_mode() {
        // Test XYB mode encoding with a 16x16 RGB image
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(90.0))
            .use_xyb(true);

        // Create a simple gradient test image
        let mut data = vec![0u8; 16 * 16 * 3];
        for y in 0..16 {
            for x in 0..16 {
                let idx = (y * 16 + x) * 3;
                data[idx] = (x * 16) as u8; // Red gradient
                data[idx + 1] = (y * 16) as u8; // Green gradient
                data[idx + 2] = 128; // Constant blue
            }
        }

        let result = encoder.encode(&data);
        assert!(result.is_ok(), "XYB encoding failed: {:?}", result.err());

        let jpeg = result.unwrap();
        // Should start with SOI
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        // Should end with EOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);

        // Should be a valid size (not too small)
        assert!(jpeg.len() > 100, "JPEG too small: {} bytes", jpeg.len());
        println!("XYB encoded JPEG size: {} bytes", jpeg.len());
    }

    #[test]
    fn test_encode_rgb_xyb_larger() {
        // Test XYB mode with a larger image (32x32)
        let encoder = Encoder::new()
            .width(32)
            .height(32)
            .pixel_format(PixelFormat::Rgb)
            .jpegli_quality(Quality::from_quality(75.0))
            .use_xyb(true);

        // Create a test pattern
        let mut data = vec![0u8; 32 * 32 * 3];
        for y in 0..32 {
            for x in 0..32 {
                let idx = (y * 32 + x) * 3;
                // Checkerboard pattern
                let checker = ((x / 4) + (y / 4)) % 2 == 0;
                data[idx] = if checker { 255 } else { 0 }; // Red
                data[idx + 1] = if checker { 0 } else { 255 }; // Green
                data[idx + 2] = 128; // Blue
            }
        }

        let result = encoder.encode(&data);
        assert!(result.is_ok(), "XYB encoding failed: {:?}", result.err());

        let jpeg = result.unwrap();
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);
        println!("XYB encoded 32x32 JPEG size: {} bytes", jpeg.len());
    }

    #[test]
    fn test_huffman_optimization_produces_valid_jpeg() {
        // Create a gradient test image
        let width = 64u32;
        let height = 64u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                data[idx] = (x * 4) as u8; // R
                data[idx + 1] = (y * 4) as u8; // G
                data[idx + 2] = ((x + y) * 2) as u8; // B
            }
        }

        let encoder = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(true);

        let result = encoder.encode(&data);
        assert!(
            result.is_ok(),
            "Optimized Huffman encoding failed: {:?}",
            result.err()
        );

        let jpeg = result.unwrap();
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], MARKER_SOI);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], MARKER_EOI);

        // Verify it's decodable
        let decoded = jpeg_decoder::Decoder::new(&jpeg[..]).decode();
        assert!(
            decoded.is_ok(),
            "Optimized JPEG not decodable: {:?}",
            decoded.err()
        );
    }

    #[test]
    fn test_huffman_optimization_reduces_file_size() {
        // Create a more complex test image that benefits from optimization
        let width = 128u32;
        let height = 128u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        // Create a pattern that will have non-uniform symbol frequencies
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                // Create blocks with varying content
                let block_type = ((x / 16) + (y / 16)) % 4;
                match block_type {
                    0 => {
                        // Solid color
                        data[idx] = 180;
                        data[idx + 1] = 180;
                        data[idx + 2] = 180;
                    }
                    1 => {
                        // Gradient
                        data[idx] = (x * 2) as u8;
                        data[idx + 1] = (y * 2) as u8;
                        data[idx + 2] = 100;
                    }
                    2 => {
                        // Checkerboard
                        let checker = ((x + y) % 2) as u8 * 255;
                        data[idx] = checker;
                        data[idx + 1] = checker;
                        data[idx + 2] = checker;
                    }
                    _ => {
                        // Texture
                        data[idx] = ((x * 5 + y * 3) % 256) as u8;
                        data[idx + 1] = ((x * 3 + y * 7) % 256) as u8;
                        data[idx + 2] = ((x * 2 + y * 2) % 256) as u8;
                    }
                }
            }
        }

        // Encode without optimization
        let jpeg_standard = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(false)
            .encode(&data)
            .expect("Standard encoding failed");

        // Encode with optimization
        let jpeg_optimized = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .optimize_huffman(true)
            .encode(&data)
            .expect("Optimized encoding failed");

        println!(
            "Standard size: {} bytes, Optimized size: {} bytes, Savings: {:.1}%",
            jpeg_standard.len(),
            jpeg_optimized.len(),
            (1.0 - jpeg_optimized.len() as f64 / jpeg_standard.len() as f64) * 100.0
        );

        // Optimized should be smaller or equal (never larger)
        assert!(
            jpeg_optimized.len() <= jpeg_standard.len(),
            "Optimized ({}) should not be larger than standard ({})",
            jpeg_optimized.len(),
            jpeg_standard.len()
        );

        // Verify both are decodable
        let decoded_std = jpeg_decoder::Decoder::new(&jpeg_standard[..]).decode();
        let decoded_opt = jpeg_decoder::Decoder::new(&jpeg_optimized[..]).decode();
        assert!(decoded_std.is_ok(), "Standard JPEG not decodable");
        assert!(decoded_opt.is_ok(), "Optimized JPEG not decodable");
    }

    #[test]
    fn test_xyb_huffman_optimization() {
        // Create test image for XYB mode
        let width = 64u32;
        let height = 64u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                data[idx] = (x * 4) as u8;
                data[idx + 1] = (y * 4) as u8;
                data[idx + 2] = ((x + y) * 2) as u8;
            }
        }

        // Encode XYB without optimization
        let jpeg_standard = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .use_xyb(true)
            .optimize_huffman(false)
            .encode(&data)
            .expect("Standard XYB encoding failed");

        // Encode XYB with optimization
        let jpeg_optimized = Encoder::new()
            .width(width)
            .height(height)
            .jpegli_quality(Quality::from_quality(75.0))
            .use_xyb(true)
            .optimize_huffman(true)
            .encode(&data)
            .expect("Optimized XYB encoding failed");

        println!(
            "XYB Standard: {} bytes, Optimized: {} bytes, Savings: {:.1}%",
            jpeg_standard.len(),
            jpeg_optimized.len(),
            (1.0 - jpeg_optimized.len() as f64 / jpeg_standard.len() as f64) * 100.0
        );

        // Verify both have valid JPEG structure
        assert_eq!(jpeg_standard[0], 0xFF);
        assert_eq!(jpeg_standard[1], MARKER_SOI);
        assert_eq!(jpeg_optimized[0], 0xFF);
        assert_eq!(jpeg_optimized[1], MARKER_SOI);

        // Optimized should be smaller or equal
        assert!(
            jpeg_optimized.len() <= jpeg_standard.len(),
            "XYB Optimized ({}) should not be larger than standard ({})",
            jpeg_optimized.len(),
            jpeg_standard.len()
        );
    }

    #[test]
    fn test_smoothing_factor() {
        // Create a high-frequency COLOR pattern that will show smoothing effects
        // (black/white won't work - chroma is constant for grayscale)
        let width = 64u32;
        let height = 64u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        // Create colorful checkerboard pattern (red/cyan alternating)
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                if (x + y) % 2 == 0 {
                    // Red
                    data[idx] = 255;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                } else {
                    // Cyan
                    data[idx] = 0;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                }
            }
        }

        // Encode with 4:2:0 subsampling, no smoothing
        let jpeg_no_smooth = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S420)
            .smoothing_factor(0)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Encoding without smoothing failed");

        // Encode with 4:2:0 subsampling, moderate smoothing
        let jpeg_smooth_50 = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S420)
            .smoothing_factor(50)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Encoding with smoothing=50 failed");

        // Encode with 4:2:0 subsampling, max smoothing
        let jpeg_smooth_100 = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S420)
            .smoothing_factor(100)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Encoding with smoothing=100 failed");

        // All should produce valid JPEGs
        assert_eq!(jpeg_no_smooth[0], 0xFF);
        assert_eq!(jpeg_no_smooth[1], MARKER_SOI);
        assert_eq!(jpeg_smooth_50[0], 0xFF);
        assert_eq!(jpeg_smooth_50[1], MARKER_SOI);
        assert_eq!(jpeg_smooth_100[0], 0xFF);
        assert_eq!(jpeg_smooth_100[1], MARKER_SOI);

        // All should be decodable
        assert!(jpeg_decoder::Decoder::new(&jpeg_no_smooth[..])
            .decode()
            .is_ok());
        assert!(jpeg_decoder::Decoder::new(&jpeg_smooth_50[..])
            .decode()
            .is_ok());
        assert!(jpeg_decoder::Decoder::new(&jpeg_smooth_100[..])
            .decode()
            .is_ok());

        // Smoothing should reduce file size for high-frequency content
        // (blurring reduces chroma complexity)
        println!(
            "No smooth: {} bytes, Smooth 50: {} bytes, Smooth 100: {} bytes",
            jpeg_no_smooth.len(),
            jpeg_smooth_50.len(),
            jpeg_smooth_100.len()
        );
    }

    #[test]
    fn test_smoothing_factor_444_noop() {
        // With 4:4:4 subsampling, smoothing should have no effect
        let width = 32u32;
        let height = 32u32;
        let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let jpeg_no_smooth = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S444)
            .smoothing_factor(0)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Encoding 444 without smoothing failed");

        let jpeg_smooth = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S444)
            .smoothing_factor(100)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Encoding 444 with smoothing failed");

        // With 4:4:4, smoothing shouldn't change anything (no downsampling)
        assert_eq!(
            jpeg_no_smooth.len(),
            jpeg_smooth.len(),
            "4:4:4 should not be affected by smoothing_factor"
        );
    }

    #[test]
    fn test_sharp_yuv_420() {
        // Test Sharp YUV with 4:2:0 produces valid JPEG
        let width = 64u32;
        let height = 64u32;
        // Create a colorful gradient to test color edge preservation
        let mut data = vec![0u8; (width * height * 3) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                data[idx] = (x * 4) as u8; // R increases horizontally
                data[idx + 1] = (y * 4) as u8; // G increases vertically
                data[idx + 2] = 128; // B constant
            }
        }

        // Encode with Sharp YUV
        let jpeg_sharp = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S420)
            .sharp_yuv(true)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Sharp YUV 4:2:0 encoding failed");

        // Encode with standard downsampling (Sharp YUV disabled)
        let jpeg_standard = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S420)
            .sharp_yuv(false)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Standard 4:2:0 encoding failed");

        // Both should produce valid JPEGs
        assert!(jpeg_sharp.len() > 0, "Sharp YUV output should not be empty");
        assert!(
            jpeg_standard.len() > 0,
            "Standard output should not be empty"
        );

        // Sharp YUV should produce a valid JPEG (starts with SOI marker)
        assert_eq!(&jpeg_sharp[0..2], &[0xFF, 0xD8], "Should be valid JPEG");
        assert_eq!(
            &jpeg_sharp[jpeg_sharp.len() - 2..],
            &[0xFF, 0xD9],
            "Should end with EOI"
        );
    }

    #[test]
    fn test_sharp_yuv_422() {
        // Test Sharp YUV with 4:2:2 produces valid JPEG
        let width = 64u32;
        let height = 64u32;
        let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let jpeg = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S422)
            .sharp_yuv(true)
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Sharp YUV 4:2:2 encoding failed");

        // Should produce valid JPEG
        assert!(jpeg.len() > 0);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_sharp_yuv_falls_back_for_444() {
        // 4:4:4 should work with sharp_yuv=true (falls back to standard path)
        let width = 32u32;
        let height = 32u32;
        let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let jpeg = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S444)
            .sharp_yuv(true) // Should still work, just uses standard path
            .jpegli_quality(Quality::from_quality(90.0))
            .encode(&data)
            .expect("Sharp YUV with 4:4:4 should fall back to standard");

        assert!(jpeg.len() > 0);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    // ========================================================================
    // Internal Pathway Tests (for benchmarking infrastructure)
    // ========================================================================

    #[test]
    fn test_internal_pathway_valid_f32_none_444() {
        use internal_pathway::*;

        // P_F32_NONE should work with 4:4:4
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S444)
            .set_internal_pathway(P_F32_NONE);

        assert!(encoder.is_ok(), "P_F32_NONE with 4:4:4 should be valid");
    }

    #[test]
    fn test_internal_pathway_valid_yuv_sharp_420() {
        use internal_pathway::*;

        // P_YUV_SHARP should work with 4:2:0
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(P_YUV_SHARP);

        assert!(encoder.is_ok(), "P_YUV_SHARP with 4:2:0 should be valid");
    }

    #[test]
    fn test_internal_pathway_valid_f32_box_420() {
        use internal_pathway::*;

        // P_F32_BOX should work with 4:2:0
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(P_F32_BOX);

        assert!(encoder.is_ok(), "P_F32_BOX with 4:2:0 should be valid");
    }

    #[test]
    fn test_internal_pathway_valid_f32_box_smooth50() {
        use internal_pathway::*;

        // P_F32_BOX_SMOOTH50 should work with 4:2:0
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(P_F32_BOX_SMOOTH50);

        assert!(
            encoder.is_ok(),
            "P_F32_BOX_SMOOTH50 with 4:2:0 should be valid"
        );
    }

    #[test]
    fn test_internal_pathway_invalid_none_with_420() {
        use internal_pathway::*;

        // DOWNSAMPLE_NONE with 4:2:0 should fail
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_NONE);

        assert!(encoder.is_err(), "DOWNSAMPLE_NONE with 4:2:0 should fail");
    }

    #[test]
    fn test_internal_pathway_invalid_sharp_with_444() {
        use internal_pathway::*;

        // DOWNSAMPLE_SHARP with 4:4:4 should fail
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S444)
            .set_internal_pathway(COLOR_YUV_BALANCED | DOWNSAMPLE_SHARP);

        assert!(encoder.is_err(), "DOWNSAMPLE_SHARP with 4:4:4 should fail");
    }

    #[test]
    fn test_internal_pathway_invalid_sharp_with_440() {
        use internal_pathway::*;

        // DOWNSAMPLE_SHARP with 4:4:0 should fail (yuv crate doesn't support it)
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S440)
            .set_internal_pathway(COLOR_YUV_BALANCED | DOWNSAMPLE_SHARP);

        assert!(encoder.is_err(), "DOWNSAMPLE_SHARP with 4:4:0 should fail");
    }

    #[test]
    fn test_internal_pathway_invalid_yuv_balanced_with_440() {
        use internal_pathway::*;

        // COLOR_YUV_BALANCED with 4:4:0 should fail (yuv crate doesn't support 4:4:0)
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S440)
            .set_internal_pathway(COLOR_YUV_BALANCED | DOWNSAMPLE_BOX);

        assert!(
            encoder.is_err(),
            "COLOR_YUV_BALANCED with 4:4:0 should fail"
        );
    }

    #[test]
    fn test_internal_pathway_gamma_aware_420() {
        use internal_pathway::*;

        // DOWNSAMPLE_GAMMA_AWARE_F32 should work with 4:2:0
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(P_F32_GAMMA_AWARE);

        assert!(
            encoder.is_ok(),
            "DOWNSAMPLE_GAMMA_AWARE_F32 should work with 4:2:0"
        );
    }

    #[test]
    fn test_internal_pathway_gamma_aware_invalid_with_444() {
        use internal_pathway::*;

        // DOWNSAMPLE_GAMMA_AWARE_F32 should fail with 4:4:4 (no downsampling needed)
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S444)
            .set_internal_pathway(P_F32_GAMMA_AWARE);

        assert!(
            encoder.is_err(),
            "DOWNSAMPLE_GAMMA_AWARE_F32 should fail with 4:4:4"
        );
    }

    #[test]
    fn test_internal_pathway_gamma_aware_encode_420() {
        use internal_pathway::*;

        // Create a simple gradient test image
        let width = 32u32;
        let height = 32u32;
        let mut data = vec![0u8; (width * height * 3) as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = (y * width as usize + x) * 3;
                data[idx] = (x * 8) as u8; // R
                data[idx + 1] = (y * 8) as u8; // G
                data[idx + 2] = ((x + y) * 4) as u8; // B
            }
        }

        let encoder = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(P_F32_GAMMA_AWARE)
            .expect("Should create encoder");

        let result = encoder.encode(&data);
        assert!(
            result.is_ok(),
            "Gamma-aware 4:2:0 encoding failed: {:?}",
            result.err()
        );

        let jpeg = result.unwrap();
        // Verify it's a valid JPEG
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "Should start with SOI");
        assert_eq!(
            &jpeg[jpeg.len() - 2..],
            &[0xFF, 0xD9],
            "Should end with EOI"
        );
        assert!(jpeg.len() > 100, "JPEG should have reasonable size");
    }

    #[test]
    fn test_internal_pathway_gamma_aware_encode_422() {
        use internal_pathway::*;

        let width = 32u32;
        let height = 32u32;
        let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let encoder = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S422)
            .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_F32)
            .expect("Should create encoder");

        let result = encoder.encode(&data);
        assert!(
            result.is_ok(),
            "Gamma-aware 4:2:2 encoding failed: {:?}",
            result.err()
        );

        let jpeg = result.unwrap();
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_internal_pathway_gamma_aware_encode_440() {
        use internal_pathway::*;

        let width = 32u32;
        let height = 32u32;
        let data: Vec<u8> = (0..width * height * 3).map(|i| (i % 256) as u8).collect();

        let encoder = Encoder::new()
            .width(width)
            .height(height)
            .subsampling(Subsampling::S440)
            .set_internal_pathway(COLOR_INTRINSIC_F32 | DOWNSAMPLE_GAMMA_AWARE_F32)
            .expect("Should create encoder");

        let result = encoder.encode(&data);
        assert!(
            result.is_ok(),
            "Gamma-aware 4:4:0 encoding failed: {:?}",
            result.err()
        );

        let jpeg = result.unwrap();
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_internal_pathway_unimplemented_yuv_professional() {
        use internal_pathway::*;

        // COLOR_YUV_PROFESSIONAL should fail (requires feature)
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(COLOR_YUV_PROFESSIONAL | DOWNSAMPLE_BOX);

        assert!(
            encoder.is_err(),
            "COLOR_YUV_PROFESSIONAL should fail (not implemented)"
        );
    }

    #[test]
    fn test_internal_pathway_invalid_color_byte() {
        // Invalid color conversion byte (4+)
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S444)
            .set_internal_pathway(4); // Invalid color byte

        assert!(encoder.is_err(), "Color byte 4 should be invalid");
    }

    #[test]
    fn test_internal_pathway_invalid_downsample_byte() {
        use internal_pathway::*;

        // Invalid downsampling byte (6+)
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S444)
            .set_internal_pathway(COLOR_INTRINSIC_F32 | (6 << 8));

        assert!(encoder.is_err(), "Downsample byte 6 should be invalid");
    }

    #[test]
    fn test_internal_pathway_invalid_smoothing_over_100() {
        use internal_pathway::*;

        // Smoothing > 100 should fail
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(with_smoothing(
                COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX_SMOOTHED,
                101,
            ));

        assert!(encoder.is_err(), "Smoothing factor 101 should be invalid");
    }

    #[test]
    fn test_internal_pathway_invalid_smoothing_without_box_smoothed() {
        use internal_pathway::*;

        // Smoothing with non-BoxSmoothed downsampling should fail
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S420)
            .set_internal_pathway(with_smoothing(COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX, 50));

        assert!(
            encoder.is_err(),
            "Smoothing with DOWNSAMPLE_BOX should fail"
        );
    }

    #[test]
    fn test_internal_pathway_invalid_reserved_bits() {
        use internal_pathway::*;

        // Reserved bits (24-63) should cause failure
        let encoder = Encoder::new()
            .width(16)
            .height(16)
            .subsampling(Subsampling::S444)
            .set_internal_pathway(P_F32_NONE | (1u64 << 24));

        assert!(encoder.is_err(), "Reserved bit 24 should be invalid");
    }

    #[test]
    fn test_internal_pathway_with_smoothing_helper() {
        use internal_pathway::*;

        // with_smoothing helper should work correctly
        let pathway = with_smoothing(COLOR_INTRINSIC_F32 | DOWNSAMPLE_BOX_SMOOTHED, 75);
        assert_eq!(pathway & 0xFF, COLOR_INTRINSIC_F32);
        assert_eq!((pathway >> 8) & 0xFF, 3); // DOWNSAMPLE_BOX_SMOOTHED = 3
        assert_eq!((pathway >> 16) & 0xFF, 75);
    }

    #[test]
    fn test_internal_pathway_pipeline_encode_decode() {
        use internal_pathway::*;

        // Test that ChromaPipeline roundtrips correctly
        let pipeline = ChromaPipeline::from_u64(P_F32_BOX_SMOOTH50).unwrap();
        assert_eq!(
            pipeline.color_conversion,
            ColorConversionMethod::IntrinsicF32
        );
        assert_eq!(pipeline.downsampling, DownsamplingMethod::BoxSmoothed);
        assert_eq!(pipeline.smoothing_factor, 50);

        // Test encode/decode roundtrip
        let encoded = pipeline.to_u64();
        let decoded = ChromaPipeline::from_u64(encoded).unwrap();
        assert_eq!(decoded.color_conversion, pipeline.color_conversion);
        assert_eq!(decoded.downsampling, pipeline.downsampling);
        assert_eq!(decoded.smoothing_factor, pipeline.smoothing_factor);
    }
}
