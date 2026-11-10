//! Decoded image types for JPEG decoding.
//!
//! This module contains the output types returned by the decoder.
//!
//! For memory-efficient decoding of large images, prefer streaming APIs like
//! `Decoder::scanline_reader()`.

use crate::types::PixelFormat;

use super::DecodeWarning;
use super::extras::DecodedExtras;

/// A decoded image with dimensions and pixel data.
///
/// For large images, consider using `Decoder::scanline_reader()` to decode
/// row-by-row into caller-provided buffers.
#[derive(Clone)]
#[non_exhaustive]
pub struct DecodedImage {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Pixel format of the data
    pub format: PixelFormat,
    /// Raw pixel data in the specified format
    pub data: Vec<u8>,
    /// Preserved metadata and secondary images (if preservation was enabled)
    pub(crate) extras: Option<DecodedExtras>,
    /// Warnings collected during decode (empty in Strict mode, which errors instead).
    pub(crate) warnings: Vec<DecodeWarning>,
}

impl core::fmt::Debug for DecodedImage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecodedImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("data_len", &self.data.len())
            .field("has_extras", &self.extras.is_some())
            .finish()
    }
}

impl DecodedImage {
    /// Returns the image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the pixel data.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.data
    }

    /// Returns the number of bytes per pixel for this image's format.
    #[must_use]
    pub fn bytes_per_pixel(&self) -> usize {
        self.format.bytes_per_pixel()
    }

    /// Returns the stride (bytes per row) of the image.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.width as usize * self.bytes_per_pixel()
    }

    /// Access preserved extras (metadata and secondary images).
    ///
    /// Returns `None` if preservation wasn't configured or if there were
    /// no segments to preserve.
    #[must_use]
    pub fn extras(&self) -> Option<&DecodedExtras> {
        self.extras.as_ref()
    }

    /// Take ownership of preserved extras.
    #[must_use]
    pub fn take_extras(&mut self) -> Option<DecodedExtras> {
        self.extras.take()
    }

    /// Returns warnings collected during decode.
    ///
    /// In [`Strict`](super::config::Strictness::Strict) mode, this is always empty because warnings
    /// become errors. In [`Balanced`](super::config::Strictness::Balanced) and [`Lenient`](super::config::Strictness::Lenient)
    /// modes, issues like truncation or missing DHT are collected here.
    #[must_use]
    pub fn warnings(&self) -> &[DecodeWarning] {
        &self.warnings
    }

    /// Returns true if any warnings were collected during decode.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Decompose the image into its parts.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u8>, u32, u32, PixelFormat, Option<DecodedExtras>) {
        (self.data, self.width, self.height, self.format, self.extras)
    }
}

/// A decoded image with 32-bit floating point pixel data.
///
/// This preserves the full 12-bit internal precision of jpegli's decoder
/// without quantization to 8-bit. Values are in the range 0.0-1.0.
///
/// Use this format when you need:
/// - Maximum precision for further image processing
/// - HDR workflows
/// - Scientific/medical imaging applications
/// - Input to machine learning models
///
/// For large images, consider using streaming APIs to decode row-by-row.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedImageF32 {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Pixel format of the data
    pub format: PixelFormat,
    /// Float pixel data in range 0.0-1.0
    pub data: Vec<f32>,
    /// Warnings collected during decode (empty in Strict mode).
    pub(crate) warnings: Vec<DecodeWarning>,
}

impl DecodedImageF32 {
    /// Returns the image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the pixel data.
    #[must_use]
    pub fn pixels(&self) -> &[f32] {
        &self.data
    }

    /// Returns the number of channels for this image's format.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.format.num_channels()
    }

    /// Returns the stride (floats per row) of the image.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.width as usize * self.channels()
    }

    /// Converts to 8-bit integer format.
    ///
    /// Values are scaled from 0.0-1.0 to 0-255 and clamped.
    #[must_use]
    pub fn to_u8(&self) -> DecodedImage {
        let len = self.data.len();
        let mut data = vec![0u8; len];

        for i in 0..len {
            data[i] = (self.data[i] * 255.0).round().clamp(0.0, 255.0) as u8;
        }

        DecodedImage {
            width: self.width,
            height: self.height,
            format: self.format,
            data,
            extras: None,
            warnings: self.warnings.clone(),
        }
    }

    /// Converts to 16-bit integer format.
    ///
    /// Values are scaled from 0.0-1.0 to 0-65535 and clamped.
    #[must_use]
    pub fn to_u16(&self) -> Vec<u16> {
        let len = self.data.len();
        let mut result = vec![0u16; len];

        for i in 0..len {
            result[i] = (self.data[i] * 65535.0).round().clamp(0.0, 65535.0) as u16;
        }
        result
    }

    /// Returns warnings collected during decode.
    ///
    /// In [`Strict`](super::config::Strictness::Strict) mode, this is always empty because warnings
    /// become errors. In [`Balanced`](super::config::Strictness::Balanced) and [`Lenient`](super::config::Strictness::Lenient)
    /// modes, issues like truncation or missing DHT are collected here.
    #[must_use]
    pub fn warnings(&self) -> &[DecodeWarning] {
        &self.warnings
    }

    /// Returns true if any warnings were collected during decode.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Decoded YCbCr planes as 32-bit floats.
///
/// This provides direct access to the YCbCr color space data without
/// conversion to RGB, bypassing the expensive color conversion step.
///
/// Values are in centered range [-128, 127] (raw DCT output after level shift).
/// To convert to standard JPEG range [0, 255], add 128 to each value.
///
/// # Use Cases
///
/// - Video pipelines that work in YCbCr space
/// - Re-encoding without color space round-trip
/// - Custom color space transformations
/// - Maximum performance when RGB is not needed
///
/// For large images, consider using streaming APIs to decode row-by-row.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedYCbCr {
    /// Luma plane (width × height), range [-128, 127]
    pub y: Vec<f32>,
    /// Chroma-blue plane (width × height, upsampled), range [-128, 127]
    pub cb: Vec<f32>,
    /// Chroma-red plane (width × height, upsampled), range [-128, 127]
    pub cr: Vec<f32>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Embedded ICC profile, if present
    pub icc_profile: Option<Vec<u8>>,
}

impl DecodedYCbCr {
    /// Returns the image dimensions as a tuple (width, height).
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the number of pixels in each plane.
    #[must_use]
    pub fn plane_size(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Shifts all planes in-place from centered range \[-128, 127\] to
    /// standard JPEG range \[0, 255\] by adding 128 to every sample.
    ///
    /// After calling this, `y` is in \[0, 255\] and `cb`/`cr` are in \[0, 255\].
    /// This avoids the three separate allocations of the per-plane methods.
    pub fn shift_to_jpeg_range(&mut self) {
        for v in &mut self.y {
            *v += 128.0;
        }
        for v in &mut self.cb {
            *v += 128.0;
        }
        for v in &mut self.cr {
            *v += 128.0;
        }
    }

    /// Converts Y plane to standard JPEG range \[0, 255\].
    ///
    /// Returns a new vector with values shifted by +128.
    /// Prefer [`shift_to_jpeg_range()`](Self::shift_to_jpeg_range) to avoid
    /// allocating three new vectors.
    #[must_use]
    pub fn y_to_jpeg_range(&self) -> Vec<f32> {
        self.y.iter().map(|&v| v + 128.0).collect()
    }

    /// Converts Cb plane to standard JPEG range \[0, 255\].
    ///
    /// Returns a new vector with values shifted by +128.
    /// Prefer [`shift_to_jpeg_range()`](Self::shift_to_jpeg_range) to avoid
    /// allocating three new vectors.
    #[must_use]
    pub fn cb_to_jpeg_range(&self) -> Vec<f32> {
        self.cb.iter().map(|&v| v + 128.0).collect()
    }

    /// Converts Cr plane to standard JPEG range \[0, 255\].
    ///
    /// Returns a new vector with values shifted by +128.
    /// Prefer [`shift_to_jpeg_range()`](Self::shift_to_jpeg_range) to avoid
    /// allocating three new vectors.
    #[must_use]
    pub fn cr_to_jpeg_range(&self) -> Vec<f32> {
        self.cr.iter().map(|&v| v + 128.0).collect()
    }
}

/// DCT coefficients for a single component.
///
/// Coefficients are stored in zigzag order as they appear in the JPEG file.
/// Each block contains 64 i16 values.
#[derive(Debug, Clone)]
pub struct ComponentCoefficients {
    /// Component ID (typically 1=Y, 2=Cb, 3=Cr for YCbCr)
    pub id: u8,
    /// Coefficients in block-row-major order, zigzag within each block.
    /// Length = blocks_wide * blocks_high * 64
    pub coeffs: Vec<i16>,
    /// Number of horizontal blocks (component width / 8)
    pub blocks_wide: usize,
    /// Number of vertical blocks (component height / 8)
    pub blocks_high: usize,
    /// Horizontal sampling factor
    pub h_samp: u8,
    /// Vertical sampling factor
    pub v_samp: u8,
    /// Quantization table index (which quant table this component uses)
    pub quant_table_idx: u8,
}

impl ComponentCoefficients {
    /// Returns a block's coefficients by block index.
    ///
    /// Block index is `by * blocks_wide + bx` where (bx, by) is block position.
    #[must_use]
    pub fn block(&self, block_idx: usize) -> &[i16] {
        let start = block_idx * 64;
        &self.coeffs[start..start + 64]
    }

    /// Returns a block's coefficients by position.
    #[must_use]
    pub fn block_at(&self, bx: usize, by: usize) -> &[i16] {
        self.block(by * self.blocks_wide + bx)
    }

    /// Returns the total number of blocks.
    #[must_use]
    pub fn num_blocks(&self) -> usize {
        self.blocks_wide * self.blocks_high
    }
}

/// Decoded DCT coefficients for analysis and comparison.
///
/// This provides access to the raw quantized DCT coefficients before IDCT,
/// useful for debugging, quality analysis, and encoder comparison.
///
/// # Example
///
/// ```rust,ignore
/// use zenjpeg::decode::Decoder;
///
/// let decoder = Decoder::new();
/// let coeffs = decoder.decode_coefficients(&jpeg_data)?;
///
/// // Access Y component DC coefficient for first block
/// let y_dc = coeffs.components[0].block(0)[0];
/// println!("Y DC: {}", y_dc);
/// ```
///
/// For analysis of large images, consider streaming APIs.
#[derive(Debug, Clone)]
pub struct DecodedCoefficients {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Per-component coefficient data
    pub components: Vec<ComponentCoefficients>,
    /// Quantization tables (one per table slot used)
    /// Index matches component's quant_table_idx
    pub quant_tables: Vec<Option<[u16; 64]>>,
    /// Huffman tables harvested from the bitstream, if reconstructible.
    /// Access via [`huffman_tables()`](Self::huffman_tables).
    pub(crate) huffman_tables: Option<crate::huffman::optimize::HuffmanTableSet>,
}

impl DecodedCoefficients {
    /// Returns the number of components.
    #[must_use]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Huffman tables harvested from the decoded bitstream, ready to feed
    /// back into the encoder via
    /// [`EncoderConfig::huffman`](crate::encoder::EncoderConfig::huffman)
    /// for transcode-time table reuse (single-pass re-encoding with the
    /// source's symbol distribution).
    ///
    /// Slot mapping follows the baseline Y/C convention: DC/AC table 0 →
    /// luma, table 1 → chroma. Grayscale baseline streams define no chroma
    /// tables; the luma tables are reused in the chroma slots (harmless —
    /// they are unused when encoding grayscale). For **progressive** JPEGs
    /// this is the final table state after all scans; scan scripts spread
    /// tables across slots per scan, so the slots need not correspond to
    /// Y/C and the set is best treated as a same-distribution warm start
    /// rather than an exact table carry-over. Custom tables are a
    /// baseline-encode strategy (progressive re-encoding always optimizes
    /// per scan).
    ///
    /// Returns `None` when the stream's tables could not be reconstructed
    /// (e.g. no luma tables were ever defined).
    ///
    /// ```
    /// use enough::Unstoppable;
    /// use zenjpeg::decoder::Decoder;
    /// use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    ///
    /// let rgb: Vec<u8> = (0..48 * 48 * 3).map(|i| (i * 31 % 251) as u8).collect();
    /// let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);
    /// let jpeg = config.encode_bytes(&rgb, 48, 48, PixelLayout::Rgb8Srgb)?;
    ///
    /// // Harvest the tables, then re-encode single-pass with them.
    /// let coeffs = Decoder::new().decode_coefficients(&jpeg, Unstoppable)?;
    /// let tables = coeffs.huffman_tables().expect("baseline stream").clone();
    /// let reencoded = config
    ///     .huffman(tables)
    ///     .encode_bytes(&rgb, 48, 48, PixelLayout::Rgb8Srgb)?;
    /// # let _ = reencoded;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn huffman_tables(&self) -> Option<&crate::huffman::optimize::HuffmanTableSet> {
        self.huffman_tables.as_ref()
    }

    /// Compares coefficients with another decode result, returning statistics.
    ///
    /// Returns (total_blocks, differing_blocks, max_diff, total_diff_coeffs)
    #[must_use]
    pub fn compare(&self, other: &DecodedCoefficients) -> CoefficientComparison {
        let mut total_blocks = 0usize;
        let mut differing_blocks = 0usize;
        let mut max_diff = 0i16;
        let mut total_diff_coeffs = 0usize;
        let mut diff_by_position = [0u64; 64];

        for (comp_idx, (c1, c2)) in self.components.iter().zip(&other.components).enumerate() {
            let num_blocks = c1.num_blocks().min(c2.num_blocks());
            for block_idx in 0..num_blocks {
                total_blocks += 1;
                let b1 = c1.block(block_idx);
                let b2 = c2.block(block_idx);
                let mut has_diff = false;
                for coeff_idx in 0..64 {
                    let diff = (b1[coeff_idx] as i32 - b2[coeff_idx] as i32).abs() as i16;
                    if diff != 0 {
                        has_diff = true;
                        total_diff_coeffs += 1;
                        diff_by_position[coeff_idx] += 1;
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
                if has_diff {
                    differing_blocks += 1;
                }
            }
            // Warn if block counts differ
            if c1.num_blocks() != c2.num_blocks() {
                eprintln!(
                    "Warning: component {} block count mismatch: {} vs {}",
                    comp_idx,
                    c1.num_blocks(),
                    c2.num_blocks()
                );
            }
        }

        CoefficientComparison {
            total_blocks,
            differing_blocks,
            max_diff,
            total_diff_coeffs,
            diff_by_position,
        }
    }
}

/// Statistics from comparing two coefficient sets.
#[derive(Debug, Clone)]
pub struct CoefficientComparison {
    /// Total number of blocks compared
    pub total_blocks: usize,
    /// Number of blocks with at least one differing coefficient
    pub differing_blocks: usize,
    /// Maximum absolute difference found
    pub max_diff: i16,
    /// Total count of differing coefficients
    pub total_diff_coeffs: usize,
    /// Difference counts by zigzag position (0=DC, 1-63=AC)
    pub diff_by_position: [u64; 64],
}

impl CoefficientComparison {
    /// Returns the percentage of blocks with differences.
    #[must_use]
    pub fn diff_block_pct(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            100.0 * self.differing_blocks as f64 / self.total_blocks as f64
        }
    }

    /// Returns the percentage of DC coefficients that differ.
    #[must_use]
    pub fn dc_diff_pct(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            100.0 * self.diff_by_position[0] as f64 / self.total_blocks as f64
        }
    }
}

// ============================================================================
// JbrdMetadata — per-scan signals for JPEG-Bitstream-Reconstruction (JXL JBRD)
// ============================================================================

/// Per-scan JPEG-Bitstream-Reconstruction (JBRD) metadata.
///
/// This is the metadata a downstream JPEG-XL transcoder needs to reproduce
/// the *exact original* JPEG entropy-coded bitstream from DCT coefficients,
/// per the JXL spec's JBRD box. It is intentionally NOT included in
/// [`DecodedCoefficients`] — that struct is the legacy 0.8.x public surface
/// and we keep it byte-for-byte backwards-compatible.
///
/// Callers wanting JBRD reconstruction use
/// [`Decoder::decode_coefficients_with_jbrd_metadata`] which returns this
/// alongside the coefficients.
///
/// # Mapping to libjxl `JPEGScanInfo`
///
/// Each [`JbrdScanInfo`] corresponds 1-to-1 with a `JPEGScanInfo` in libjxl's
/// `enc_jpeg_data_reader.cc`. The `block_idx` values inside `reset_points`
/// and `extra_zero_runs` are libjxl's `block_scan_index` — a per-scan
/// running counter across ALL components and ALL blocks decoded so far in
/// that scan (for interleaved scans, MCU-by-MCU, component-by-component,
/// block-by-block within each component).
///
/// # When entries are populated
///
/// - `reset_points`: signaled in AC scans (`Ss > 0`) when two end-of-block
///   runs occur back-to-back. Empty for DC-only scans.
/// - `extra_zero_runs`: signaled in the AC FIRST scan (`Ah == 0, Ss > 0`)
///   when ZRL (run=15, size=0) symbols accumulate before a natural EOB.
///   Always empty in AC refinement scans (`Ah > 0`) and DC scans.
///
/// [`Decoder::decode_coefficients_with_jbrd_metadata`]: super::Decoder::decode_coefficients_with_jbrd_metadata
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct JbrdMetadata {
    /// Per-scan signals, in the order scans appear in the JPEG bitstream.
    ///
    /// For a baseline sequential JPEG this contains exactly one entry.
    /// For a progressive JPEG with N SOS markers, this contains N entries.
    pub scans: Vec<JbrdScanInfo>,
    /// Whether the JPEG contains any non-1 entropy-segment padding bits.
    ///
    /// JPEG entropy-coded segments end with `0..7` padding bits before the
    /// next marker when the entropy stream ends mid-byte. ITU-T T.81 §F.1.2.3
    /// recommends these padding bits be 1, but some encoders pad with 0 (or
    /// mixed values). Byte-exact JPEG-XL transcoding (JBRD) needs to preserve
    /// the source's actual padding bits.
    ///
    /// When `false`, all padding bits are 1 — the standard fast path — and
    /// `padding_bits` is empty. When `true`, `padding_bits` contains the
    /// explicit bit sequence (one entry per pad bit, value `0` or `1`).
    pub has_zero_padding_bit: bool,
    /// Padding-bit values for every entropy-segment boundary (per RST marker
    /// AND at end-of-scan), in bitstream order, MSB-first.
    ///
    /// Empty when `has_zero_padding_bit` is `false`. Total length equals the
    /// sum of `bits_in_buffer & 7` at every scan-segment terminator (each
    /// terminator contributes `0..=7` bits).
    ///
    /// Maps directly to libjxl's `JPEGData::padding_bits` field (see
    /// `enc_jpeg_data_reader.cc:441-470` `FinishStream`).
    pub padding_bits: Vec<u8>,
}

/// Per-scan reset-point + extra-zero-run signals (JBRD).
///
/// One of these is collected per SOS scan when
/// [`Decoder::decode_coefficients_with_jbrd_metadata`] is used.
///
/// The semantics match libjxl's `JPEGScanInfo::reset_points` and
/// `JPEGScanInfo::extra_zero_runs` (see `enc_jpeg_data_reader.cc:849-857`).
///
/// [`Decoder::decode_coefficients_with_jbrd_metadata`]: super::Decoder::decode_coefficients_with_jbrd_metadata
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct JbrdScanInfo {
    /// Spectral selection start (`Ss`).
    pub ss: u8,
    /// Spectral selection end (`Se`).
    pub se: u8,
    /// Successive approximation high (`Ah`).
    pub ah: u8,
    /// Successive approximation low (`Al`).
    pub al: u8,
    /// Block-scan indices at which two end-of-block runs occurred
    /// back-to-back — the encoder must force a state reset here.
    ///
    /// Always empty for DC-only scans (`Ss == 0`). Populated only when an
    /// AC scan emits a fresh EOB run at the beginning of a block with no
    /// preceding active EOB run.
    pub reset_points: Vec<u32>,
    /// Extra zero runs that occurred before a natural end-of-block in
    /// the AC first scan (`Ah == 0, Ss > 0`).
    ///
    /// Each entry is `(block_scan_index, num_extra_zero_runs)`. A "ZRL"
    /// symbol encodes 16 consecutive zeros; runs of ZRL symbols
    /// immediately preceding an EOB are *extra* in the sense that the
    /// re-encoder would otherwise prefer to emit them via an EOB-run
    /// instead. Always empty for DC scans and AC refinement scans
    /// (`Ah > 0`).
    pub extra_zero_runs: Vec<(u32, u32)>,
}
