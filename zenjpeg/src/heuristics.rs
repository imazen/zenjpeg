//! Resource estimation heuristics for JPEG encoding and decoding operations.
//!
//! These heuristics provide approximate estimates for memory consumption and
//! time costs of encoding/decoding operations. Use them for:
//!
//! - Pre-allocating buffers
//! - Sizing thread pools
//! - Memory budgeting
//! - Progress estimation
//!
//! # Accuracy
//!
//! Estimates are based on the encoder's internal memory estimation logic.
//! Memory estimates include all internal buffers (strip buffers, DCT blocks,
//! token buffers, output buffers).
//!
//! # Content Type Impact
//!
//! Image content significantly affects both memory and time:
//!
//! | Content | Encode Memory | Encode Time |
//! |---------|---------------|-------------|
//! | Solid   | Min           | **Fastest** |
//! | Gradient| Typical       | Fast        |
//! | Photo   | Typical       | **Typical** |
//! | Noise   | Max           | Slow        |
//!
//! For photos (typical web content), adaptive quantization is the bottleneck.
//! For noise/high-entropy content, Huffman encoding is slower.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenjpeg::heuristics::{estimate_encode, estimate_decode};
//! use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
//! use zenjpeg::decoder::PixelFormat;
//!
//! // Estimate encode resources for a 1920x1080 image
//! let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
//! let encode_est = estimate_encode(1920, 1080, &config);
//! println!("Encode peak memory: {:.1} MB", encode_est.peak_memory_bytes as f64 / 1_000_000.0);
//! println!("Encode time: {:.0}ms (typical)", encode_est.time_ms);
//!
//! // Estimate decode resources
//! let decode_est = estimate_decode(1920, 1080, PixelFormat::Rgb);
//! println!("Decode peak memory: {:.1} MB", decode_est.peak_memory_bytes as f64 / 1_000_000.0);
//! println!("Decode time: {:.0}ms (typical)", decode_est.time_ms);
//! ```

use crate::encoder::EncoderConfig;
use crate::types::PixelFormat;

// =============================================================================
// Encode throughput constants — MEASURED 2026-06-14 (jpeg_probe wall/user,
// single-thread, ycbcr 4:2:0, progressive default, NO trellis/boundary-rd;
// 4 classes × 512–2048 px × q{50,85}; benchmarks/jpeg_resource_2026-06-14.tsv).
// The previous "estimated from jpegli" 8/15/40 were ~4–5× too pessimistic for
// the common no-trellis case (measured baseline 16.6 / 67.6 / 84.4 MP/s for
// complex/typical/simple). Trellis and boundary-rd are applied separately as
// multipliers below — they are the dominant, previously-unmodelled costs.
// Values are pre-`prog_factor`: the 0.7 progressive factor (kept) recovers
// the measured progressive-default throughput (e.g. 96 × 0.7 ≈ 67.6 MP/s).
// =============================================================================

/// Encode throughput in Mpix/s for simple content (low entropy).
const ENCODE_THROUGHPUT_MAX_MPIXELS: f64 = 120.0;

/// Encode throughput in Mpix/s for typical content (photos).
const ENCODE_THROUGHPUT_TYP_MPIXELS: f64 = 96.0;

/// Encode throughput in Mpix/s for complex content (noise, high-entropy).
const ENCODE_THROUGHPUT_MIN_MPIXELS: f64 = 24.0;

/// Time multipliers for the RD features (measured 2026-06-14, vs the
/// no-feature baseline). Trellis quantization dominates encode cost (~6.2×
/// time, ~2× working set); boundary-rd adds ~1.55×. They are NOT
/// multiplicative — with trellis already running, boundary-rd's extra cost
/// is mostly absorbed (both-on measured 6.5×, not 9.7×), so the combined
/// case has its own factor.
const TRELLIS_TIME_FACTOR: f64 = 6.24;
const BOUNDARY_RD_TIME_FACTOR: f64 = 1.55;
const TRELLIS_AND_BRD_TIME_FACTOR: f64 = 6.53;
/// Trellis ~doubles the encoder working set (6.3 → 10.9 B/px measured).
const TRELLIS_MEM_FACTOR: f64 = 1.7;

// =============================================================================
// Decode throughput constants
// =============================================================================

/// Decode throughput in Mpix/s for simple content (baseline, simple).
const DECODE_THROUGHPUT_MAX_MPIXELS: f64 = 120.0;

/// Decode throughput in Mpix/s for typical content (photos).
const DECODE_THROUGHPUT_TYP_MPIXELS: f64 = 80.0;

/// Decode throughput in Mpix/s for complex content (progressive, high-entropy).
const DECODE_THROUGHPUT_MIN_MPIXELS: f64 = 40.0;

// =============================================================================
// Memory multipliers for content variation
// =============================================================================

/// Memory multiplier for simple content (min).
const ENCODE_MEMORY_MIN_MULT: f64 = 0.9;

/// Memory multiplier for typical content.
const ENCODE_MEMORY_TYP_MULT: f64 = 1.0;

/// Memory multiplier for complex content (max).
/// Token buffers grow with entropy.
const ENCODE_MEMORY_MAX_MULT: f64 = 1.3;

/// Decode memory varies little with content type.
const DECODE_MEMORY_MIN_MULT: f64 = 1.0;
const DECODE_MEMORY_TYP_MULT: f64 = 1.0;
const DECODE_MEMORY_MAX_MULT: f64 = 1.1;

// =============================================================================
// Public types
// =============================================================================

/// Resource estimation for encode operations.
///
/// Based on jpegli's internal memory estimation and throughput measurements.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct EncodeEstimate {
    /// Minimum expected peak memory (best case: solid color, simple gradient).
    pub peak_memory_bytes_min: u64,

    /// Typical peak memory in bytes during encoding (natural photos).
    pub peak_memory_bytes: u64,

    /// Maximum expected peak memory (worst case: noise, high-entropy).
    pub peak_memory_bytes_max: u64,

    /// Estimated heap allocations. Fewer allocations = better latency.
    pub allocations: u32,

    /// Encode time in milliseconds (best case: simple content).
    pub time_ms_min: f32,

    /// Encode time in milliseconds (typical: real photographs).
    pub time_ms: f32,

    /// Encode time in milliseconds (worst case: noise/high-entropy).
    pub time_ms_max: f32,

    /// Estimated output size in bytes.
    /// JPEG compression varies widely; this is a rough estimate.
    pub output_bytes: u64,

    /// Input size in bytes (width × height × bytes_per_pixel).
    pub input_bytes: u64,
}

/// Resource estimation for decode operations.
///
/// Based on jpegli's decoder memory estimation and throughput measurements.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct DecodeEstimate {
    /// Minimum expected peak memory (best case: baseline, simple content).
    pub peak_memory_bytes_min: u64,

    /// Typical peak memory in bytes during decoding.
    pub peak_memory_bytes: u64,

    /// Maximum expected peak memory (worst case: progressive, complex).
    pub peak_memory_bytes_max: u64,

    /// Estimated heap allocations during decoding.
    pub allocations: u32,

    /// Decode time in milliseconds (best case: baseline, simple content).
    pub time_ms_min: f32,

    /// Decode time in milliseconds (typical: real photos).
    pub time_ms: f32,

    /// Decode time in milliseconds (worst case: progressive, complex).
    pub time_ms_max: f32,

    /// Output buffer size in bytes (width × height × output_bpp).
    pub output_bytes: u64,
}

// =============================================================================
// Estimation functions
// =============================================================================

/// Estimate resources for encoding a JPEG image.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Encoder configuration
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::heuristics::estimate_encode;
/// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
///
/// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
/// let est = estimate_encode(1920, 1080, &config);
/// println!("Peak memory: {:.1} MB", est.peak_memory_bytes as f64 / 1_000_000.0);
/// println!("Time: {:.0}ms (typical)", est.time_ms);
/// ```
#[must_use]
pub fn estimate_encode(width: u32, height: u32, config: &EncoderConfig) -> EncodeEstimate {
    let pixels = (width as u64) * (height as u64);

    // The RD features dominate cost and are read from the config:
    // trellis (always compiled, data-gated) and boundary-rd (feature-gated).
    let trellis_active = config
        .trellis
        .as_ref()
        .is_some_and(|t| t.is_ac_enabled() || t.is_dc_enabled());
    #[cfg(feature = "boundary-rd")]
    let brd_active = matches!(config.boundary_rd_mode, crate::encode::BoundaryRd::On(_));
    #[cfg(not(feature = "boundary-rd"))]
    let brd_active = false;

    // Use the encoder's internal memory estimation, scaled up for trellis
    // (its per-block candidate buffers ~double the working set — measured
    // 6.3 → 10.9 B/px; the encoder's estimate does not model this).
    let base_memory = config.estimate_memory(width, height) as u64;
    let mem_feature = if trellis_active {
        TRELLIS_MEM_FACTOR
    } else {
        1.0
    };

    let peak_memory_bytes_min = (base_memory as f64 * ENCODE_MEMORY_MIN_MULT * mem_feature) as u64;
    let peak_memory_bytes = (base_memory as f64 * ENCODE_MEMORY_TYP_MULT * mem_feature) as u64;
    let peak_memory_bytes_max = (base_memory as f64 * ENCODE_MEMORY_MAX_MULT * mem_feature) as u64;

    // Time calculation from throughput, then the measured RD-feature
    // multipliers (trellis ≈ 6.2×, boundary-rd ≈ 1.55×, both ≈ 6.5× — NOT
    // multiplicative; trellis dominates and absorbs most of brd's overhead).
    let pixels_f = pixels as f64;
    let prog_factor = if config.is_progressive() { 0.7 } else { 1.0 };
    let feature_time_factor = match (trellis_active, brd_active) {
        (true, true) => TRELLIS_AND_BRD_TIME_FACTOR,
        (true, false) => TRELLIS_TIME_FACTOR,
        (false, true) => BOUNDARY_RD_TIME_FACTOR,
        (false, false) => 1.0,
    };
    let t = |mpix: f64| (pixels_f * feature_time_factor / (mpix * prog_factor * 1000.0)) as f32;
    let time_ms_min = t(ENCODE_THROUGHPUT_MAX_MPIXELS);
    let time_ms = t(ENCODE_THROUGHPUT_TYP_MPIXELS);
    let time_ms_max = t(ENCODE_THROUGHPUT_MIN_MPIXELS);

    // Output estimate: JPEG typically 5-20% of raw size for photos
    // Using ~10% as typical for quality 85
    let input_bytes = pixels * 3; // Assume RGB input
    let output_bytes = input_bytes / 10;

    // Allocations: encoder has ~20-30 major allocations
    let allocations = 25;

    EncodeEstimate {
        peak_memory_bytes_min,
        peak_memory_bytes,
        peak_memory_bytes_max,
        allocations,
        time_ms_min,
        time_ms,
        time_ms_max,
        output_bytes,
        input_bytes,
    }
}

/// Estimate resources for encoding with a guaranteed memory ceiling.
///
/// This uses the encoder's `estimate_memory_ceiling()` which returns an
/// absolute upper bound that actual peak memory will never exceed.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Encoder configuration
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::heuristics::estimate_encode_ceiling;
/// use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling};
///
/// let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
/// let est = estimate_encode_ceiling(1920, 1080, &config);
/// // Reserve this much memory - actual usage guaranteed to be less
/// let buffer = Vec::with_capacity(est.peak_memory_bytes as usize);
/// ```
#[must_use]
pub fn estimate_encode_ceiling(width: u32, height: u32, config: &EncoderConfig) -> EncodeEstimate {
    let mut est = estimate_encode(width, height, config);

    // Override with the guaranteed ceiling from the encoder
    let ceiling = config.estimate_memory_ceiling(width, height) as u64;
    est.peak_memory_bytes_min = ceiling;
    est.peak_memory_bytes = ceiling;
    est.peak_memory_bytes_max = ceiling;

    est
}

/// Estimate resources for decoding a JPEG image.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `format` - Output pixel format (determines bytes per pixel)
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::heuristics::estimate_decode;
/// use zenjpeg::decoder::PixelFormat;
///
/// let est = estimate_decode(1920, 1080, PixelFormat::Rgb);
/// println!("Output buffer: {:.1} MB", est.output_bytes as f64 / 1_000_000.0);
/// println!("Peak memory: {:.1} MB", est.peak_memory_bytes as f64 / 1_000_000.0);
/// println!("Time: {:.0}ms (typical)", est.time_ms);
/// ```
#[must_use]
pub fn estimate_decode(width: u32, height: u32, format: PixelFormat) -> DecodeEstimate {
    let output_bpp = format.bytes_per_pixel() as u8;
    let w = width as usize;
    let h = height as usize;
    let pixels = (width as u64) * (height as u64);

    // Output buffer size
    let output_bytes = pixels * (output_bpp as u64);

    // Replicate decoder's memory estimation logic
    // MCU width for strip buffers (padded to 8)
    let mcu_cols = (w + 7) / 8;
    let strip_width = mcu_cols * 8;
    let strip_height = 8;

    // Strip buffers: Y, Cb, Cr each at i16 (2 bytes per pixel)
    let strip_size = strip_width * strip_height;
    let strip_total = strip_size * 2 * 3;

    // RGB output buffer
    let rgb_size = w * h * 3;

    // Streaming total (baseline 4:4:4)
    let streaming_total = strip_total + rgb_size;

    // Non-streaming (progressive, subsampled) coefficient storage
    let blocks_per_component = mcu_cols * ((h + 7) / 8);
    let coeff_storage = blocks_per_component * 130 * 3;

    // Worst case (non-streaming)
    let base_memory = streaming_total.max(coeff_storage + rgb_size) as u64;

    // Apply content-dependent multipliers
    let peak_memory_bytes_min = (base_memory as f64 * DECODE_MEMORY_MIN_MULT) as u64;
    let peak_memory_bytes = (base_memory as f64 * DECODE_MEMORY_TYP_MULT) as u64;
    let peak_memory_bytes_max = (base_memory as f64 * DECODE_MEMORY_MAX_MULT) as u64;

    // Time calculation from throughput
    let pixels_f = pixels as f64;
    let time_ms_min = (pixels_f / (DECODE_THROUGHPUT_MAX_MPIXELS * 1000.0)) as f32;
    let time_ms = (pixels_f / (DECODE_THROUGHPUT_TYP_MPIXELS * 1000.0)) as f32;
    let time_ms_max = (pixels_f / (DECODE_THROUGHPUT_MIN_MPIXELS * 1000.0)) as f32;

    // Allocations: decoder has fewer allocations
    let allocations = 15;

    DecodeEstimate {
        peak_memory_bytes_min,
        peak_memory_bytes,
        peak_memory_bytes_max,
        allocations,
        time_ms_min,
        time_ms,
        time_ms_max,
        output_bytes,
    }
}

/// Estimate resources for streaming decode (one scanline at a time).
///
/// This is more memory-efficient than full decode as it doesn't buffer
/// the entire coefficient array.
///
/// # Arguments
///
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::heuristics::estimate_decode_streaming;
///
/// let est = estimate_decode_streaming(1920, 1080);
/// println!("Streaming decode memory: {:.1} MB",
///     est.peak_memory_bytes as f64 / 1_000_000.0);
/// ```
#[must_use]
pub fn estimate_decode_streaming(width: u32, height: u32) -> DecodeEstimate {
    let w = width as usize;
    let h = height as usize;
    let pixels = (width as u64) * (height as u64);

    // MCU width for strip buffers
    let mcu_cols = (w + 7) / 8;
    let strip_width = mcu_cols * 8;
    let strip_height = 8;

    // Strip buffers only (no full coefficient storage)
    let strip_size = strip_width * strip_height;
    let strip_total = strip_size * 2 * 3;

    // Output buffer (full image)
    let rgb_size = w * h * 3;

    let base_memory = (strip_total + rgb_size) as u64;

    // Streaming decode has minimal memory variation
    let peak_memory_bytes_min = base_memory;
    let peak_memory_bytes = base_memory;
    let peak_memory_bytes_max = (base_memory as f64 * 1.05) as u64;

    // Time is the same as regular decode
    let pixels_f = pixels as f64;
    let time_ms_min = (pixels_f / (DECODE_THROUGHPUT_MAX_MPIXELS * 1000.0)) as f32;
    let time_ms = (pixels_f / (DECODE_THROUGHPUT_TYP_MPIXELS * 1000.0)) as f32;
    let time_ms_max = (pixels_f / (DECODE_THROUGHPUT_MIN_MPIXELS * 1000.0)) as f32;

    let allocations = 10;
    let output_bytes = pixels * 3;

    DecodeEstimate {
        peak_memory_bytes_min,
        peak_memory_bytes,
        peak_memory_bytes_max,
        allocations,
        time_ms_min,
        time_ms,
        time_ms_max,
        output_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::ChromaSubsampling;
    use crate::types::PixelFormat;

    #[test]
    fn encode_estimate_scales_with_size() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let small = estimate_encode(256, 256, &config);
        let large = estimate_encode(512, 512, &config);

        // 4x pixels should give roughly 4x memory
        let ratio = large.peak_memory_bytes as f64 / small.peak_memory_bytes as f64;
        assert!(ratio > 2.5 && ratio < 6.0, "Ratio was {}", ratio);
    }

    #[test]
    fn decode_estimate_scales_with_size() {
        let small = estimate_decode(256, 256, PixelFormat::Rgb);
        let large = estimate_decode(512, 512, PixelFormat::Rgb);

        // 4x pixels should give roughly 4x memory
        let ratio = large.peak_memory_bytes as f64 / small.peak_memory_bytes as f64;
        assert!(ratio > 2.5 && ratio < 6.0, "Ratio was {}", ratio);
    }

    #[test]
    fn time_ranges_are_ordered() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let enc = estimate_encode(1024, 1024, &config);
        assert!(enc.time_ms_min < enc.time_ms);
        assert!(enc.time_ms < enc.time_ms_max);

        let dec = estimate_decode(1024, 1024, PixelFormat::Rgb);
        assert!(dec.time_ms_min < dec.time_ms);
        assert!(dec.time_ms < dec.time_ms_max);
    }

    #[test]
    fn memory_ranges_are_ordered() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let enc = estimate_encode(1024, 1024, &config);
        assert!(enc.peak_memory_bytes_min <= enc.peak_memory_bytes);
        assert!(enc.peak_memory_bytes <= enc.peak_memory_bytes_max);

        let dec = estimate_decode(1024, 1024, PixelFormat::Rgb);
        assert!(dec.peak_memory_bytes_min <= dec.peak_memory_bytes);
        assert!(dec.peak_memory_bytes <= dec.peak_memory_bytes_max);
    }

    #[test]
    fn ceiling_is_at_least_typical() {
        let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let typical = estimate_encode(1024, 1024, &config);
        let ceiling = estimate_encode_ceiling(1024, 1024, &config);

        assert!(ceiling.peak_memory_bytes >= typical.peak_memory_bytes);
    }

    #[test]
    fn streaming_decode_uses_less_memory() {
        let full = estimate_decode(1024, 1024, PixelFormat::Rgb);
        let streaming = estimate_decode_streaming(1024, 1024);

        // Streaming should use less memory (no coefficient storage)
        assert!(streaming.peak_memory_bytes <= full.peak_memory_bytes);
    }

    #[test]
    fn progressive_is_slower() {
        let baseline = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).progressive(false);
        let progressive = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter); // Progressive is default

        let base_est = estimate_encode(1024, 1024, &baseline);
        let prog_est = estimate_encode(1024, 1024, &progressive);

        // Progressive should be slower
        assert!(prog_est.time_ms > base_est.time_ms);
    }

    /// Trellis is the dominant encode cost (~6.2× time, measured) and also
    /// raises the estimated working set; the model must reflect both.
    #[test]
    fn trellis_dominates_encode_cost() {
        use crate::encode::trellis::TrellisConfig;
        let base = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let trellis =
            EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).trellis(TrellisConfig::default());
        let b = estimate_encode(1024, 1024, &base);
        let t = estimate_encode(1024, 1024, &trellis);
        let time_ratio = t.time_ms / b.time_ms;
        assert!(
            (5.0..8.0).contains(&time_ratio),
            "trellis time factor should be ~6.2×, got {time_ratio}"
        );
        assert!(
            t.peak_memory_bytes > b.peak_memory_bytes,
            "trellis raises estimated memory"
        );
    }

    /// Boundary-rd adds a measured ~1.55× when trellis is off.
    #[cfg(feature = "boundary-rd")]
    #[test]
    fn boundary_rd_adds_time() {
        use crate::encode::{BoundaryRd, BoundaryRdConfig};
        let base = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
        let brd = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
            .boundary_rd(BoundaryRd::On(BoundaryRdConfig::new()));
        let b = estimate_encode(1024, 1024, &base).time_ms;
        let r = estimate_encode(1024, 1024, &brd).time_ms;
        let ratio = r / b;
        assert!(
            (1.3..1.9).contains(&ratio),
            "boundary-rd factor ~1.55×, got {ratio}"
        );
    }
}
