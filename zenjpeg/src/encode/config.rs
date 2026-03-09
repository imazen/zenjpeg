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

    /// Force SOF1 (extended sequential) regardless of quant table precision.
    ///
    /// XYB color space requires SOF1 because its wider dynamic range produces
    /// DC categories 12-15, exceeding baseline's limit of 11. This is independent
    /// of whether quant values exceed 255.
    pub force_sof1: bool,

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

/// Minimum MCUs per restart segment. Below this, restart overhead
/// dominates and parallel decode has too little work per segment.
const MIN_MCUS_PER_RESTART: u32 = 64;

/// Each restart marker costs ~8 bytes: 2 for RST, ~1 for bit padding,
/// ~5 for DC prediction reset across components.
const EST_BYTES_PER_MARKER: u32 = 8;

/// DRI marker header is always 6 bytes.
const DRI_HEADER_BYTES: u32 = 6;

/// Maximum restart marker overhead as parts per thousand of estimated
/// file size. 3 = 0.3%.
const MAX_OVERHEAD_PER_MILLE: u32 = 3;

/// Resolve restart rows to MCU-aligned restart interval.
///
/// Returns 0 if rows is 0. Increases row count when the resulting
/// MCU count would be below `MIN_MCUS_PER_RESTART` or when the number
/// of restart markers would bloat the file by more than 0.3%.
/// Ensures the result fits in u16 by reducing rows if needed.
pub(crate) fn resolve_restart_rows(
    rows: u16,
    width: u32,
    height: u32,
    subsampling: Subsampling,
) -> u16 {
    if rows == 0 {
        return 0;
    }
    let h_samp = match subsampling {
        Subsampling::S444 | Subsampling::S440 => 1u32,
        Subsampling::S422 | Subsampling::S420 => 2,
    };
    let v_samp = match subsampling {
        Subsampling::S444 | Subsampling::S422 => 1u32,
        Subsampling::S440 | Subsampling::S420 => 2,
    };
    let mcu_w = h_samp * 8;
    let mcu_h = v_samp * 8;
    let mcu_cols = (width + mcu_w - 1) / mcu_w;
    let mcu_rows = (height + mcu_h - 1) / mcu_h;
    let total_mcus = mcu_cols * mcu_rows;

    // Conservative file size estimate at 0.5 bpp (bits per pixel).
    // This is well below typical JPEG output even at very low quality.
    let total_pixels = width * height;
    let est_file_bytes = total_pixels / 16; // 0.5 bpp = 1 byte per 16 pixels

    // Maximum overhead budget: 0.3% of estimated file size
    let max_overhead = est_file_bytes * MAX_OVERHEAD_PER_MILLE / 1000;

    // Compute minimum restart interval from overhead budget
    let min_rows_for_overhead = if max_overhead <= DRI_HEADER_BYTES + EST_BYTES_PER_MARKER {
        // File too small for even one restart marker within budget
        mcu_rows
    } else {
        let max_markers = (max_overhead - DRI_HEADER_BYTES) / EST_BYTES_PER_MARKER;
        if max_markers == 0 {
            mcu_rows
        } else {
            let min_ri = (total_mcus + max_markers - 1) / max_markers;
            // Convert MCU interval back to rows (round up)
            (min_ri + mcu_cols - 1) / mcu_cols.max(1)
        }
    };

    // Ensure each restart segment has at least MIN_MCUS_PER_RESTART MCUs
    let min_rows_for_parallel = (MIN_MCUS_PER_RESTART + mcu_cols - 1) / mcu_cols.max(1);

    let rows = (rows as u32)
        .max(min_rows_for_parallel)
        .max(min_rows_for_overhead);
    let max_rows = (u16::MAX as u32) / mcu_cols.max(1);
    let rows = rows.min(max_rows);
    (rows * mcu_cols) as u16
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
            force_sof1: false,
            scan_strategy: ScanStrategy::Default,
            // Use 3 tables by default (matches jpegli_set_distance)
            separate_chroma_tables: true,
            #[cfg(feature = "trellis")]
            trellis: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_restart_overhead_limit() {
        // 512×512 4:2:0: 0.5bpp est = 16384 bytes, budget = 49 bytes
        // max 5 markers → min ri = ceil(1024/5) = 205 → 7 rows × 32 cols = 224
        let ri = resolve_restart_rows(4, 512, 512, Subsampling::S420);
        assert_eq!(ri, 7 * 32, "512x512 4:2:0 should use 7 rows × 32 cols");

        // 1024×1024: 0.5bpp est = 65536 bytes, budget = 196 bytes
        // max 23 markers → min ri = ceil(4096/23) = 179 → 3 rows × 64 cols = 192
        // but 4 rows > 3, so stays at 4 rows
        let ri = resolve_restart_rows(4, 1024, 1024, Subsampling::S420);
        assert_eq!(ri, 4 * 64, "1024x1024 4:2:0 should use 4 rows × 64 cols");

        // Very large image: still 4 rows
        let ri = resolve_restart_rows(4, 4096, 4096, Subsampling::S420);
        assert_eq!(ri, 4 * 256);
    }

    #[test]
    fn test_resolve_restart_small_image_caps_markers() {
        // Small image: overhead limit should increase interval
        // 64×64: est file = 4096/16 = 256 bytes at 0.5bpp
        // Budget: 256*3/1000 = 0 bytes — not enough for even one marker
        let ri = resolve_restart_rows(4, 64, 64, Subsampling::S420);
        let total = 4u16 * 4;
        assert!(
            ri >= total,
            "64×64 should have no restart markers, ri={ri} total={total}"
        );

        // 128×128: est file = 16384/16 = 1024 bytes at 0.5bpp
        // Budget: 1024*3/1000 = 3 bytes — not even one marker (need 14)
        let ri = resolve_restart_rows(4, 128, 128, Subsampling::S420);
        let total = 8u16 * 8;
        assert!(
            ri >= total,
            "128×128 should have no restart markers, ri={ri} total={total}"
        );

        // 256×256: est file = 65536/16 = 4096 bytes at 0.5bpp
        // Budget: 4096*3/1000 = 12 bytes — not enough for DRI(6)+RST(8)=14
        let ri = resolve_restart_rows(4, 256, 256, Subsampling::S420);
        let total = 16u16 * 16;
        assert!(
            ri >= total,
            "256×256 should have no restart markers, ri={ri} total={total}"
        );
    }

    #[test]
    fn test_resolve_restart_overhead_under_limit() {
        // For any image, verify the marker overhead stays under 0.3%
        // using the same 0.5bpp estimate as the implementation
        for &(w, h) in &[
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (4096, 4096),
            (1920, 1080),
            (320, 240),
            (16, 4096),
            (4096, 16),
        ] {
            for &ss in &[Subsampling::S420, Subsampling::S444, Subsampling::S422] {
                let ri = resolve_restart_rows(4, w, h, ss);
                if ri == 0 {
                    continue;
                }
                let h_samp: u32 = match ss {
                    Subsampling::S444 | Subsampling::S440 => 1,
                    Subsampling::S422 | Subsampling::S420 => 2,
                };
                let v_samp: u32 = match ss {
                    Subsampling::S444 | Subsampling::S422 => 1,
                    Subsampling::S440 | Subsampling::S420 => 2,
                };
                let mcu_w = h_samp * 8;
                let mcu_h = v_samp * 8;
                let mcu_cols = (w + mcu_w - 1) / mcu_w;
                let mcu_rows = (h + mcu_h - 1) / mcu_h;
                let total = mcu_cols * mcu_rows;
                let est_file = w * h / 16; // 0.5 bpp
                let num_markers = if ri as u32 >= total {
                    0
                } else {
                    total / ri as u32
                };
                let overhead = DRI_HEADER_BYTES + num_markers * EST_BYTES_PER_MARKER;
                let max_overhead = est_file * MAX_OVERHEAD_PER_MILLE / 1000;
                assert!(
                    overhead <= max_overhead || num_markers == 0,
                    "{w}×{h} {ss:?}: overhead {overhead} > max {max_overhead} \
                     (ri={ri}, markers={num_markers}, est_file={est_file})"
                );
            }
        }
    }

    #[test]
    fn test_resolve_restart_disabled() {
        assert_eq!(resolve_restart_rows(0, 512, 512, Subsampling::S420), 0);
        assert_eq!(resolve_restart_rows(0, 64, 64, Subsampling::S444), 0);
    }
}
