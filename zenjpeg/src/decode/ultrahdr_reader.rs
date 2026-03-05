//! Streaming UltraHDR decoder for row-by-row HDR JPEG processing.
//!
//! This module provides [`UltraHdrReader`], a streaming decoder for UltraHDR JPEGs
//! that supports multiple decode modes for different use cases:
//!
//! - **SDR-only**: Fastest decode, ignores gain map entirely
//! - **HDR**: Applies gain map to reconstruct HDR output
//! - **SDR+HDR**: Dual output for preview + processing workflows
//! - **SDR+GainMap**: For editing workflows that need to preserve/modify gain maps
//!
//! # Memory Efficiency
//!
//! For a 4K image (3840×2160):
//!
//! | Mode | Peak Memory |
//! |------|-------------|
//! | SdrOnly | ~500 KB |
//! | Hdr (Full) | ~1 MB |
//! | Hdr (Streaming) | ~515 KB |
//! | SdrAndGainMap | ~625 KB |
//!
//! # Example: Streaming HDR Decode
//!
//! ```rust,ignore
//! use zenjpeg::decoder::Decoder;
//! use zenjpeg::ultrahdr::{UltraHdrReaderConfig, UltraHdrMode, GainMapMemory};
//!
//! let config = UltraHdrReaderConfig::new()
//!     .mode(UltraHdrMode::Hdr)
//!     .display_boost(4.0)
//!     .memory_strategy(GainMapMemory::Streaming);
//!
//! let mut reader = Decoder::new().ultrahdr_reader(&jpeg_data, config)?;
//!
//! // Allocate output buffers
//! let width = reader.dimensions().width as usize;
//! let height = reader.dimensions().height as usize;
//! let mut hdr_buf = vec![0.0f32; width * 4]; // RGBA f32 per row
//!
//! while !reader.is_finished() {
//!     let rows = reader.read_rows(1, None, Some(&mut hdr_buf), None)?;
//!     // Process HDR row...
//! }
//! ```
//!
//! # Example: Dual SDR+HDR Output
//!
//! ```rust,ignore
//! let config = UltraHdrReaderConfig::new()
//!     .mode(UltraHdrMode::SdrAndHdr)
//!     .display_boost(4.0);
//!
//! let mut reader = Decoder::new().ultrahdr_reader(&jpeg_data, config)?;
//!
//! while !reader.is_finished() {
//!     reader.read_rows(16, Some(&mut sdr_buf), Some(&mut hdr_buf), None)?;
//!     // Both SDR and HDR available simultaneously
//! }
//! ```

use crate::decode::{DecodedExtras, ScanlineReader};
use crate::error::{Error, Result};
use crate::types::Dimensions;

#[cfg(feature = "ultrahdr")]
use ultrahdr_core::{
    ColorGamut, GainMap, GainMapMetadata,
    gainmap::{RowDecoder, StreamDecoder},
};

// ============================================================================
// Configuration Types
// ============================================================================

/// Decode mode for UltraHDR streaming reader.
///
/// Determines what outputs are produced during decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum UltraHdrMode {
    /// Fastest decode: ignore gain map, output SDR only.
    ///
    /// Use when the gain map is not needed and you just want
    /// the base SDR image as quickly as possible.
    SdrOnly,

    /// Apply gain map to reconstruct HDR output.
    ///
    /// This is the default mode for HDR-capable displays.
    #[default]
    Hdr,

    /// Dual output: produce both SDR and HDR simultaneously.
    ///
    /// Useful for preview workflows where you need both versions,
    /// or when writing to formats that store both representations.
    SdrAndHdr,

    /// For editing: output SDR + raw gain map without applying it.
    ///
    /// Preserves the gain map for later modification and re-encoding.
    /// The gain map is output at its native resolution (typically smaller
    /// than the main image).
    SdrAndGainMap,
}

/// Memory strategy for gain map handling during decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GainMapMemory {
    /// Load entire gain map into memory before processing.
    ///
    /// Simpler implementation, ~500KB for a 4K image's gain map.
    /// This is the default and recommended for most use cases.
    #[default]
    Full,

    /// Stream gain map with a 16-row ring buffer.
    ///
    /// Minimal memory footprint, but requires parallel streaming
    /// of both the base JPEG and gain map JPEG. More complex
    /// internally but uses constant memory regardless of image size.
    Streaming,
}

/// Configuration for [`UltraHdrReader`].
///
/// Use the builder pattern to configure decode options:
///
/// ```rust,ignore
/// let config = UltraHdrReaderConfig::new()
///     .mode(UltraHdrMode::Hdr)
///     .display_boost(4.0);
/// ```
///
/// HDR output is always linear f32 RGBA. The caller is responsible for
/// converting to other formats (PQ, sRGB) if needed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UltraHdrReaderConfig {
    /// Decode mode determining what outputs are produced.
    pub mode: UltraHdrMode,

    /// Display boost factor for HDR reconstruction.
    ///
    /// - 1.0 = SDR (no boost)
    /// - 4.0 = typical HDR display
    /// - 8.0 = high-end HDR display
    ///
    /// Default: 1.0 (SDR display)
    pub display_boost: f32,

    /// Memory strategy for gain map handling.
    ///
    /// Default: Full (load entire gain map)
    pub memory_strategy: GainMapMemory,

    /// Whether to preserve metadata for re-encoding.
    ///
    /// When true, extras like EXIF, XMP, and the original gain map
    /// bytes are preserved and can be retrieved via `take_extras()`.
    ///
    /// Default: false
    pub preserve_metadata: bool,
}

impl Default for UltraHdrReaderConfig {
    fn default() -> Self {
        Self {
            mode: UltraHdrMode::Hdr,
            display_boost: 1.0,
            memory_strategy: GainMapMemory::Full,
            preserve_metadata: false,
        }
    }
}

impl UltraHdrReaderConfig {
    /// Create a new configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the decode mode.
    #[must_use]
    pub fn mode(mut self, mode: UltraHdrMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the display boost factor.
    ///
    /// Typical values:
    /// - 1.0 for SDR displays
    /// - 4.0 for typical HDR displays
    /// - 8.0 for high-end HDR displays
    #[must_use]
    pub fn display_boost(mut self, boost: f32) -> Self {
        self.display_boost = boost;
        self
    }

    /// Set the memory strategy for gain map handling.
    #[must_use]
    pub fn memory_strategy(mut self, strategy: GainMapMemory) -> Self {
        self.memory_strategy = strategy;
        self
    }

    /// Set whether to preserve metadata for re-encoding.
    #[must_use]
    pub fn preserve_metadata(mut self, preserve: bool) -> Self {
        self.preserve_metadata = preserve;
        self
    }

    /// Configure for SDR-only decode (fastest).
    #[must_use]
    pub fn sdr_only() -> Self {
        Self::new().mode(UltraHdrMode::SdrOnly)
    }

    /// Configure for HDR decode with typical display settings.
    #[must_use]
    pub fn hdr_default() -> Self {
        Self::new().mode(UltraHdrMode::Hdr).display_boost(4.0)
    }

    /// Configure for editing workflow (SDR + gain map preservation).
    #[must_use]
    pub fn editing() -> Self {
        Self::new()
            .mode(UltraHdrMode::SdrAndGainMap)
            .preserve_metadata(true)
    }
}

// ============================================================================
// UltraHdrReader Implementation
// ============================================================================

/// Streaming reader for UltraHDR JPEGs.
///
/// Provides row-by-row decoding with configurable output modes.
/// See module documentation for usage examples.
#[cfg(feature = "ultrahdr")]
pub struct UltraHdrReader<'a> {
    /// Configuration
    config: UltraHdrReaderConfig,

    /// Base JPEG scanline reader
    base_reader: ScanlineReader<'a>,

    /// Reference to original JPEG data (for zero-copy gain map access)
    data: &'a [u8],

    /// Whether this is actually an UltraHDR image (has gain map)
    is_ultrahdr: bool,

    /// Parsed gain map metadata
    metadata: Option<GainMapMetadata>,

    /// Decoded extras (for metadata preservation)
    extras: Option<DecodedExtras>,

    /// Internal state for HDR reconstruction
    hdr_state: Option<HdrDecoderState<'a>>,

    /// Gain map JPEG byte range in original data (start, end)
    /// Uses byte range instead of Vec<u8> for zero-copy access
    gainmap_range: Option<(usize, usize)>,
}

/// Internal state for HDR reconstruction.
#[cfg(feature = "ultrahdr")]
enum HdrDecoderState<'a> {
    /// Full gain map in memory, row-based reconstruction
    RowDecoder(Box<RowDecoder>),
    /// Streaming gain map with parallel decode
    StreamDecoder {
        decoder: Box<StreamDecoder>,
        gainmap_reader: Box<ScanlineReader<'a>>,
    },
}

#[cfg(feature = "ultrahdr")]
impl<'a> UltraHdrReader<'a> {
    /// Create a new UltraHDR reader.
    ///
    /// This is typically called via `Decoder::ultrahdr_reader()`.
    pub(crate) fn new(
        data: &'a [u8],
        config: UltraHdrReaderConfig,
        base_reader: ScanlineReader<'a>,
        extras: Option<DecodedExtras>,
        gainmap_range: Option<(usize, usize)>,
        metadata: Option<GainMapMetadata>,
    ) -> Result<Self> {
        let is_ultrahdr = metadata.is_some() && gainmap_range.is_some();

        let mut reader = Self {
            config,
            base_reader,
            data,
            is_ultrahdr,
            metadata,
            extras,
            hdr_state: None,
            gainmap_range,
        };

        // Initialize HDR state if needed
        if reader.needs_hdr_processing() && reader.is_ultrahdr {
            reader.init_hdr_state()?;
        }

        Ok(reader)
    }

    /// Check if this requires HDR processing based on mode.
    fn needs_hdr_processing(&self) -> bool {
        matches!(
            self.config.mode,
            UltraHdrMode::Hdr | UltraHdrMode::SdrAndHdr
        )
    }

    /// Initialize the HDR decoder state.
    fn init_hdr_state(&mut self) -> Result<()> {
        let metadata = self.metadata.as_ref().ok_or_else(|| {
            Error::decode_error("Missing gain map metadata for HDR decode".to_string())
        })?;

        let (gm_start, gm_end) = self.gainmap_range.ok_or_else(|| {
            Error::decode_error("Missing gain map data for HDR decode".to_string())
        })?;

        // Get gain map JPEG slice from original data (zero-copy)
        let gainmap_data = &self.data[gm_start..gm_end];

        match self.config.memory_strategy {
            GainMapMemory::Full => {
                // Decode gain map fully
                let gainmap = decode_gainmap_jpeg(gainmap_data)?;

                let width = self.base_reader.width();
                let height = self.base_reader.height();

                let row_decoder = RowDecoder::new(
                    gainmap,
                    metadata.clone(),
                    width,
                    height,
                    self.config.display_boost,
                    ColorGamut::Bt709,
                )
                .map_err(|e| Error::decode_error(e.to_string()))?;

                self.hdr_state = Some(HdrDecoderState::RowDecoder(Box::new(row_decoder)));
            }
            GainMapMemory::Streaming => {
                // Create streaming decoder
                // For streaming mode, we need to decode the gainmap JPEG in parallel
                // This is more complex - we create a second ScanlineReader for the gainmap

                // Get gainmap dimensions by reading its header
                let gm_info = crate::decode::Decoder::new()
                    .read_info(gainmap_data)
                    .map_err(|e| {
                        Error::decode_error(format!("Failed to read gainmap info: {}", e))
                    })?;

                let gm_width = gm_info.dimensions.width;
                let gm_height = gm_info.dimensions.height;
                let gm_channels = if gm_info.num_components == 1 { 1 } else { 3 };

                let sdr_width = self.base_reader.width();
                let sdr_height = self.base_reader.height();

                let stream_decoder = StreamDecoder::new(
                    metadata.clone(),
                    sdr_width,
                    sdr_height,
                    gm_width,
                    gm_height,
                    gm_channels,
                    self.config.display_boost,
                    ColorGamut::Bt709,
                )
                .map_err(|e| Error::decode_error(e.to_string()))?;

                // Create a scanline reader for the gainmap
                // Borrows directly from original data (zero-copy, no self-referential struct)
                let gm_reader = crate::decode::Decoder::new()
                    .scanline_reader(gainmap_data)
                    .map_err(|e| {
                        Error::decode_error(format!("Failed to create gainmap reader: {}", e))
                    })?;

                self.hdr_state = Some(HdrDecoderState::StreamDecoder {
                    decoder: Box::new(stream_decoder),
                    gainmap_reader: Box::new(gm_reader),
                });
            }
        }

        Ok(())
    }

    /// Returns true if this JPEG contains UltraHDR metadata and gain map.
    #[inline]
    pub fn is_ultrahdr(&self) -> bool {
        self.is_ultrahdr
    }

    /// Returns the gain map metadata if present.
    pub fn metadata(&self) -> Option<&GainMapMetadata> {
        self.metadata.as_ref()
    }

    /// Returns image dimensions.
    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        Dimensions {
            width: self.base_reader.width(),
            height: self.base_reader.height(),
        }
    }

    /// Returns the current row position (0 to height-1).
    #[inline]
    pub fn current_row(&self) -> usize {
        self.base_reader.current_row()
    }

    /// Returns true if all rows have been read.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.base_reader.is_finished()
    }

    /// Read rows into user-provided buffers.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows to read
    /// * `sdr_output` - Optional buffer for RGB8 SDR output (3 bytes per pixel per row)
    /// * `hdr_output` - Optional buffer for linear f32 RGBA HDR output
    /// * `gainmap_output` - Optional buffer for raw gain map output (SdrAndGainMap mode only)
    ///
    /// # Returns
    ///
    /// Number of rows actually read (may be less than requested at end of image).
    ///
    /// # Buffer Sizes
    ///
    /// - `sdr_output`: `width * 3 * rows` bytes (RGB8)
    /// - `hdr_output`: `width * 4 * rows` floats (linear f32 RGBA)
    /// - `gainmap_output`: `gainmap_width * gainmap_channels * rows_scaled` bytes
    pub fn read_rows(
        &mut self,
        rows: usize,
        sdr_output: Option<&mut [u8]>,
        hdr_output: Option<&mut [f32]>,
        gainmap_output: Option<&mut [u8]>,
    ) -> Result<usize> {
        let height = self.base_reader.height() as usize;
        let remaining = height - self.current_row();
        let actual_rows = rows.min(remaining);

        if actual_rows == 0 {
            return Ok(0);
        }

        match self.config.mode {
            UltraHdrMode::SdrOnly => self.read_sdr_only(actual_rows, sdr_output),
            UltraHdrMode::Hdr => self.read_hdr_only(actual_rows, hdr_output),
            UltraHdrMode::SdrAndHdr => self.read_sdr_and_hdr(actual_rows, sdr_output, hdr_output),
            UltraHdrMode::SdrAndGainMap => {
                self.read_sdr_and_gainmap(actual_rows, sdr_output, gainmap_output)
            }
        }
    }

    /// Read SDR-only rows.
    fn read_sdr_only(&mut self, rows: usize, sdr_output: Option<&mut [u8]>) -> Result<usize> {
        let Some(output) = sdr_output else {
            return Err(Error::internal(
                "SDR output buffer required for SdrOnly mode",
            ));
        };

        let width = self.base_reader.width() as usize;
        let stride = width * 3;
        let output_ref = imgref::ImgRefMut::new(output, stride, rows);
        self.base_reader.read_rows_rgb8(output_ref)
    }

    /// Read HDR-only rows.
    fn read_hdr_only(&mut self, rows: usize, hdr_output: Option<&mut [f32]>) -> Result<usize> {
        let Some(output) = hdr_output else {
            return Err(Error::internal("HDR output buffer required for Hdr mode"));
        };

        // First decode SDR rows
        let width = self.base_reader.width() as usize;
        let sdr_stride = width * 3;
        let mut sdr_buf = vec![0u8; sdr_stride * rows];
        let sdr_ref = imgref::ImgRefMut::new(&mut sdr_buf, sdr_stride, rows);
        let actual_rows = self.base_reader.read_rows_rgb8(sdr_ref)?;

        if actual_rows == 0 {
            return Ok(0);
        }

        // Apply HDR reconstruction
        self.apply_hdr_reconstruction(&sdr_buf[..sdr_stride * actual_rows], actual_rows, output)?;

        Ok(actual_rows)
    }

    /// Read both SDR and HDR rows.
    fn read_sdr_and_hdr(
        &mut self,
        rows: usize,
        sdr_output: Option<&mut [u8]>,
        hdr_output: Option<&mut [f32]>,
    ) -> Result<usize> {
        let Some(sdr_out) = sdr_output else {
            return Err(Error::internal(
                "SDR output buffer required for SdrAndHdr mode",
            ));
        };
        let Some(hdr_out) = hdr_output else {
            return Err(Error::internal(
                "HDR output buffer required for SdrAndHdr mode",
            ));
        };

        let width = self.base_reader.width() as usize;
        let sdr_stride = width * 3;

        // Decode SDR directly into user buffer
        let sdr_ref = imgref::ImgRefMut::new(sdr_out, sdr_stride, rows);
        let actual_rows = self.base_reader.read_rows_rgb8(sdr_ref)?;

        if actual_rows == 0 {
            return Ok(0);
        }

        // Apply HDR reconstruction using the SDR data
        self.apply_hdr_reconstruction(&sdr_out[..sdr_stride * actual_rows], actual_rows, hdr_out)?;

        Ok(actual_rows)
    }

    /// Read SDR and raw gain map rows.
    fn read_sdr_and_gainmap(
        &mut self,
        rows: usize,
        sdr_output: Option<&mut [u8]>,
        _gainmap_output: Option<&mut [u8]>,
    ) -> Result<usize> {
        let Some(sdr_out) = sdr_output else {
            return Err(Error::internal(
                "SDR output buffer required for SdrAndGainMap mode",
            ));
        };

        let width = self.base_reader.width() as usize;
        let sdr_stride = width * 3;

        // Decode SDR
        let sdr_ref = imgref::ImgRefMut::new(sdr_out, sdr_stride, rows);
        let actual_rows = self.base_reader.read_rows_rgb8(sdr_ref)?;

        // Note: Gain map output is provided via take_gainmap_data() since the gain map
        // has different dimensions than the main image. Row-by-row gain map output
        // would require complex scaling calculations.

        Ok(actual_rows)
    }

    /// Apply HDR reconstruction to SDR data.
    ///
    /// Converts sRGB u8 input to linear f32, feeds to the gain map decoder,
    /// and writes linear f32 RGBA output.
    fn apply_hdr_reconstruction(
        &mut self,
        sdr_data: &[u8],
        rows: usize,
        hdr_output: &mut [f32],
    ) -> Result<()> {
        let Some(ref mut hdr_state) = self.hdr_state else {
            // Not an UltraHDR image or no HDR state - just convert SDR to linear
            self.sdr_to_linear_fallback(sdr_data, rows, hdr_output);
            return Ok(());
        };

        // Convert sRGB u8 to linear f32 RGB for the streaming API
        let width = self.base_reader.width() as usize;
        let sdr_linear = srgb_u8_to_linear_f32(sdr_data, width, rows);

        match hdr_state {
            HdrDecoderState::RowDecoder(decoder) => {
                let hdr_floats = decoder
                    .process_rows(&sdr_linear, rows as u32)
                    .map_err(|e| Error::decode_error(e.to_string()))?;

                // Copy linear f32 RGBA output
                let copy_len = hdr_output.len().min(hdr_floats.len());
                hdr_output[..copy_len].copy_from_slice(&hdr_floats[..copy_len]);
            }
            HdrDecoderState::StreamDecoder {
                decoder,
                gainmap_reader,
                ..
            } => {
                // Feed gain map rows until we can process the SDR batch
                while !decoder.can_process(rows as u32) {
                    // Read gain map row
                    let gm_width = gainmap_reader.width() as usize;
                    let gm_stride = gm_width * 3;
                    let mut gm_row = vec![0u8; gm_stride];
                    let gm_ref = imgref::ImgRefMut::new(&mut gm_row, gm_stride, 1);
                    let gm_rows_read = gainmap_reader.read_rows_rgb8(gm_ref)?;

                    if gm_rows_read == 0 {
                        break;
                    }

                    // Push to decoder
                    decoder
                        .push_gainmap_row(&gm_row)
                        .map_err(|e| Error::decode_error(e.to_string()))?;
                }

                // Process SDR rows
                if decoder.can_process(rows as u32) {
                    let hdr_floats = decoder
                        .process_sdr_rows(&sdr_linear, rows as u32)
                        .map_err(|e| Error::decode_error(e.to_string()))?;

                    let copy_len = hdr_output.len().min(hdr_floats.len());
                    hdr_output[..copy_len].copy_from_slice(&hdr_floats[..copy_len]);
                } else {
                    // Not enough gain map data buffered - fall back to SDR
                    self.sdr_to_linear_fallback(sdr_data, rows, hdr_output);
                }
            }
        }

        Ok(())
    }

    /// Fallback: convert SDR to linear when HDR is not available.
    fn sdr_to_linear_fallback(&self, sdr_data: &[u8], rows: usize, hdr_output: &mut [f32]) {
        let width = self.base_reader.width() as usize;

        for row in 0..rows {
            for x in 0..width {
                let sdr_idx = (row * width + x) * 3;
                let hdr_idx = (row * width + x) * 4;

                if sdr_idx + 2 < sdr_data.len() && hdr_idx + 3 < hdr_output.len() {
                    let r = srgb_to_linear(sdr_data[sdr_idx]);
                    let g = srgb_to_linear(sdr_data[sdr_idx + 1]);
                    let b = srgb_to_linear(sdr_data[sdr_idx + 2]);

                    hdr_output[hdr_idx] = r;
                    hdr_output[hdr_idx + 1] = g;
                    hdr_output[hdr_idx + 2] = b;
                    hdr_output[hdr_idx + 3] = 1.0;
                }
            }
        }
    }

    /// Take the decoded extras (metadata, gain map data, etc.).
    ///
    /// Returns `None` if extras were not preserved or already taken.
    pub fn take_extras(&mut self) -> Option<DecodedExtras> {
        self.extras.take()
    }

    /// Get the raw gain map JPEG data as a borrowed slice (zero-copy).
    ///
    /// This is useful for SdrAndGainMap mode where you want to
    /// preserve the original gain map for later re-encoding.
    ///
    /// Returns `None` if this is not an UltraHDR image or if the
    /// gain map has already been taken.
    pub fn gainmap_jpeg(&self) -> Option<&'a [u8]> {
        self.gainmap_range
            .map(|(start, end)| &self.data[start..end])
    }

    /// Get the raw gain map JPEG data as owned bytes.
    ///
    /// This copies the gain map data. Prefer [`gainmap_jpeg()`](Self::gainmap_jpeg)
    /// for zero-copy access when possible.
    ///
    /// Returns `None` if this is not an UltraHDR image or if the
    /// gain map has already been taken.
    pub fn take_gainmap_data(&mut self) -> Option<Vec<u8>> {
        self.gainmap_range
            .take()
            .map(|(start, end)| self.data[start..end].to_vec())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert sRGB u8 to linear f32.
#[inline]
fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a buffer of sRGB RGB8 pixels to linear f32 RGB.
///
/// Input: packed RGB8 (`[R, G, B, R, G, B, ...]`), 3 bytes per pixel.
/// Output: packed linear f32 RGB (`[R, G, B, R, G, B, ...]`), 3 floats per pixel.
#[cfg(feature = "ultrahdr")]
fn srgb_u8_to_linear_f32(srgb_data: &[u8], width: usize, rows: usize) -> Vec<f32> {
    let pixel_count = width * rows;
    let mut linear = Vec::with_capacity(pixel_count * 3);
    for pixel in srgb_data[..pixel_count * 3].chunks_exact(3) {
        linear.push(srgb_to_linear(pixel[0]));
        linear.push(srgb_to_linear(pixel[1]));
        linear.push(srgb_to_linear(pixel[2]));
    }
    linear
}

/// Decode a gain map JPEG to GainMap struct.
#[cfg(feature = "ultrahdr")]
fn decode_gainmap_jpeg(jpeg_data: &[u8]) -> Result<GainMap> {
    let decoded = crate::decode::Decoder::new().decode(jpeg_data, enough::Unstoppable)?;

    let width = decoded.width();
    let height = decoded.height();
    let pixels = decoded.pixels_u8().unwrap().to_vec();

    // Determine if single-channel or multi-channel based on content
    let channels = if is_grayscale_content(&pixels) { 1 } else { 3 };

    let data = if channels == 1 {
        // Extract just the R (or first) channel
        pixels.chunks_exact(3).map(|p| p[0]).collect()
    } else {
        pixels
    };

    Ok(GainMap {
        width,
        height,
        channels,
        data,
    })
}

/// Check if decoded RGB content is actually grayscale (R==G==B for all pixels).
#[cfg(feature = "ultrahdr")]
fn is_grayscale_content(pixels: &[u8]) -> bool {
    pixels
        .chunks_exact(3)
        .take(100) // Sample first 100 pixels
        .all(|p| p[0] == p[1] && p[1] == p[2])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "ultrahdr"))]
mod tests {
    use super::*;
    use crate::decode::Decoder;

    #[test]
    fn test_config_builder() {
        let config = UltraHdrReaderConfig::new()
            .mode(UltraHdrMode::SdrAndHdr)
            .display_boost(4.0)
            .memory_strategy(GainMapMemory::Streaming)
            .preserve_metadata(true);

        assert_eq!(config.mode, UltraHdrMode::SdrAndHdr);
        assert_eq!(config.display_boost, 4.0);
        assert_eq!(config.memory_strategy, GainMapMemory::Streaming);
        assert!(config.preserve_metadata);
    }

    #[test]
    fn test_preset_configs() {
        let sdr = UltraHdrReaderConfig::sdr_only();
        assert_eq!(sdr.mode, UltraHdrMode::SdrOnly);

        let hdr = UltraHdrReaderConfig::hdr_default();
        assert_eq!(hdr.mode, UltraHdrMode::Hdr);
        assert_eq!(hdr.display_boost, 4.0);

        let edit = UltraHdrReaderConfig::editing();
        assert_eq!(edit.mode, UltraHdrMode::SdrAndGainMap);
        assert!(edit.preserve_metadata);
    }

    fn ultrahdr_test_path() -> std::path::PathBuf {
        std::env::var("ULTRAHDR_TEST_IMAGE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("/mnt/v/gen-dress.jpg"))
    }

    /// Test with a real UltraHDR image if available.
    #[test]
    #[ignore = "requires UltraHDR test image (set ULTRAHDR_TEST_IMAGE)"]
    fn test_real_ultrahdr_sdr_decode() {
        let path = ultrahdr_test_path();
        if !path.exists() {
            return;
        }

        let data = std::fs::read(path).expect("failed to read test file");

        // Test SDR-only mode (should work even with gain map)
        let config = UltraHdrReaderConfig::sdr_only();
        let mut reader = Decoder::new()
            .ultrahdr_reader(&data, config)
            .expect("failed to create reader");

        assert!(reader.is_ultrahdr());
        let dims = reader.dimensions();
        assert!(dims.width > 0);
        assert!(dims.height > 0);

        // Allocate output buffer
        let row_size = dims.width as usize * 3;
        let mut sdr_buf = vec![0u8; row_size * 16]; // 16 rows at a time

        let mut total_rows = 0;
        while !reader.is_finished() {
            let rows = reader
                .read_rows(16, Some(&mut sdr_buf), None, None)
                .expect("failed to read rows");
            total_rows += rows;
        }

        assert_eq!(total_rows, dims.height as usize);
    }

    /// Test HDR decode with Full memory strategy.
    #[test]
    #[ignore = "requires UltraHDR test image (set ULTRAHDR_TEST_IMAGE)"]
    fn test_real_ultrahdr_hdr_decode() {
        let path = ultrahdr_test_path();
        if !path.exists() {
            return;
        }

        let data = std::fs::read(path).expect("failed to read test file");

        // Test HDR mode with default settings
        let config = UltraHdrReaderConfig::hdr_default();
        let mut reader = Decoder::new()
            .ultrahdr_reader(&data, config)
            .expect("failed to create reader");

        assert!(reader.is_ultrahdr());
        assert!(reader.metadata().is_some());

        let dims = reader.dimensions();
        let hdr_row_size = dims.width as usize * 4; // RGBA f32

        let mut hdr_buf = vec![0.0f32; hdr_row_size];

        let mut total_rows = 0;
        while !reader.is_finished() {
            let rows = reader
                .read_rows(1, None, Some(&mut hdr_buf), None)
                .expect("failed to read rows");
            if rows > 0 {
                total_rows += rows;

                // Verify HDR values are reasonable (in linear light, 0-∞)
                for &v in &hdr_buf[..hdr_row_size] {
                    assert!(v.is_finite(), "HDR value should be finite");
                    // HDR values can be > 1.0 for bright areas
                }
            }
        }

        assert_eq!(total_rows, dims.height as usize);
    }
}
