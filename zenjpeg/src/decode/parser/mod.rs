//! JPEG parser implementation.
//!
//! Internal parser for reading and decoding JPEG data.
//!
//! ## Module Structure
//!
//! - `markers` - SOF, DHT, DQT, DRI marker parsing
//! - `scan` - SOS parsing and baseline entropy decoding
//! - `progressive` - Progressive scan accumulation and refinement
//! - `output` - Pixel conversion and output formatting

mod arithmetic_scan;
mod markers;
mod output;
mod progressive;
mod scan;
mod transform;

#[cfg(feature = "ultrahdr")]
use super::extras::MpfImageTypeExt;
use super::extras::{
    AdobeColorTransform, DecodedExtras, MpfImageType, PreserveConfig, should_preserve_mpf_image,
};
use super::{DecodeWarning, JpegInfo, Strictness};

/// Information about a scan needed for scanline reading.
struct ScanInfo {
    /// Huffman table mapping: (dc_table_idx, ac_table_idx) per component
    table_mapping: [(usize, usize); 3],
    /// Position in data where entropy-coded scan data begins
    data_start: usize,
}
use crate::color::icc::{extract_icc_profile, is_xyb_profile};
use crate::error::{Error, ErrorKind, Result};
use crate::foundation::alloc::{MAX_SCANS, checked_size_2d};
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, MARKER_APP0, MARKER_COM, MARKER_DAC, MARKER_DHT, MARKER_DNL, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOI, MARKER_SOS, MAX_COMPONENTS, MAX_HUFFMAN_TABLES,
    MAX_QUANT_TABLES,
};
use crate::huffman::HuffmanDecodeTable;
use crate::types::{ColorSpace, Component, Dimensions, JpegMode};
use enough::Stop;

/// Pre-computed component info for decoding efficiency.
///
/// Computed once per decode, reused across multiple methods.
pub(super) struct CompInfo {
    pub(super) quant_idx: usize,
    pub(super) h_samp: usize,
    pub(super) v_samp: usize,
    pub(super) comp_blocks_h: usize,
    pub(super) comp_blocks_v: usize,
    /// Component width in pixels (comp_blocks_h * 8)
    pub(super) comp_width: usize,
    /// Component height in pixels (comp_blocks_v * 8)
    pub(super) comp_height: usize,
    /// True if this component has full resolution (no subsampling)
    pub(super) is_full_res: bool,
}

/// Parsed JPEG scan data needed to construct a scanline reader.
///
/// Bundles the parser fields that `ScanlineReader` needs, replacing
/// the previous 16-parameter constructor. Created via `JpegParser::into_scan_data()`.
pub(super) struct ParsedScanData<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub num_components: u8,
    pub h_samp: [u8; 3],
    pub v_samp: [u8; 3],
    pub quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; MAX_QUANT_TABLES],
    pub quant_indices: [usize; 3],
    pub dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    pub ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    pub table_mapping: [(usize, usize); 3],
    pub scan_data_start: usize,
    pub restart_interval: u16,
    pub is_xyb: bool,
    pub is_rgb: bool,
}

impl<'a> ParsedScanData<'a> {
    /// Release the borrow on the underlying data by replacing it with an
    /// empty slice, returning a `ParsedScanData<'static>`.
    ///
    /// All non-data fields are preserved. This is used when the caller will
    /// supply owned data separately (via `ScanlineReader::from_scan_data_cow`).
    pub(super) fn into_owned(self) -> ParsedScanData<'static> {
        ParsedScanData {
            data: &[],
            width: self.width,
            height: self.height,
            num_components: self.num_components,
            h_samp: self.h_samp,
            v_samp: self.v_samp,
            quant_tables: self.quant_tables,
            quant_indices: self.quant_indices,
            dc_tables: self.dc_tables,
            ac_tables: self.ac_tables,
            table_mapping: self.table_mapping,
            scan_data_start: self.scan_data_start,
            restart_interval: self.restart_interval,
            is_xyb: self.is_xyb,
            is_rgb: self.is_rgb,
        }
    }
}

/// Controls which decode path the parser uses.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum DecodeMode {
    /// Auto-select best path: fused parallel → streaming → coefficient buffering.
    Auto,
    /// Force coefficient buffering (needed for f32, dequant bias, transforms, YCbCr output).
    Coefficient,
}

/// Internal JPEG parser state.
pub(super) struct JpegParser<'a> {
    pub(super) data: &'a [u8],
    pub(super) position: usize,

    // Frame info
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) precision: u8,
    pub(super) num_components: u8,
    pub(super) mode: JpegMode,

    // Component info
    pub(super) components: [Component; MAX_COMPONENTS],

    // Tables
    pub(super) quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; MAX_QUANT_TABLES],
    pub(super) dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    pub(super) ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],

    // Restart
    pub(super) restart_interval: u16,

    // Decoded coefficient data (used for progressive and non-streaming baseline)
    pub(super) coeffs: Vec<Vec<[i16; DCT_BLOCK_SIZE]>>, // Per component
    pub(super) coeff_counts: Vec<Vec<u8>>, // Coefficient count per block (for tiered IDCT)
    // Nonzero coefficient bitmap per block (bit k set = coeffs[k] != 0).
    // Used by AC refinement to skip zero positions via trailing_zeros().
    pub(super) nonzero_bitmaps: Vec<Vec<u64>>,

    // Streaming decode result (used for baseline 4:4:4 JPEGs)
    pub(super) streaming_rgb: Option<Vec<u8>>,
    /// Decode mode: Auto tries streaming/fused paths first, Coefficient forces
    /// coefficient storage (needed for f32 output, dequant bias, transforms, etc.).
    pub(super) decode_mode: DecodeMode,

    /// Fused parallel decode result (entropy + IDCT in one pass).
    #[cfg(feature = "parallel")]
    pub(super) fused_result: Option<super::fused_parallel::FusedResult>,

    /// Chroma upsampling method (set from DecodeConfig before decode).
    pub(super) chroma_upsampling: super::ChromaUpsampling,

    /// IDCT method (set from DecodeConfig before decode).
    pub(super) idct_method: super::IdctMethod,

    // ICC profile (extracted from raw data, not during parsing)
    pub(super) icc_profile: Option<Vec<u8>>,

    // Security limits
    pub(super) max_pixels: u64,

    // Extras preservation
    preserve_config: Option<PreserveConfig>,
    extras: Option<DecodedExtras>,
    /// Position of MPF TIFF header (after "MPF\0") for calculating absolute offsets
    mpf_header_pos: usize,

    /// Adobe APP14 color transform (for CMYK/YCCK detection)
    pub(super) adobe_transform: Option<AdobeColorTransform>,

    /// Strictness level for error handling
    pub(super) strictness: Strictness,

    /// Force f32 IDCT path (set by dimension-swapping transforms for exact results).
    pub(super) force_f32_idct: bool,

    /// Thread control: 0=auto, 1=force sequential.
    pub(super) num_threads: usize,

    /// Parallel decode strategy (how segments map to rayon tasks).
    #[cfg(feature = "parallel")]
    pub(super) parallel_strategy: super::config::ParallelStrategy,

    /// Warnings collected during decode (Balanced/Lenient only).
    /// In Strict mode, warnings become errors instead of being collected.
    pub(super) warnings: Vec<DecodeWarning>,

    // Arithmetic coding conditioning parameters (from DAC marker)
    /// DC conditioning: (L, U) per table, default (0, 1)
    pub(super) arith_dc_cond: [(u8, u8); 4],
    /// AC conditioning: Kx per table, default 5
    pub(super) arith_ac_kx: [u8; 4],
}

impl<'a> JpegParser<'a> {
    /// Create a new parser with optional extras preservation.
    #[allow(dead_code)] // May be used by tests or future code
    pub(super) fn new(
        data: &'a [u8],
        max_pixels: u64,
        preserve_config: Option<&PreserveConfig>,
    ) -> Result<Self> {
        Self::with_strictness(data, max_pixels, preserve_config, Strictness::default())
    }

    /// Create a new parser with explicit strictness level.
    pub(super) fn with_strictness(
        data: &'a [u8],
        max_pixels: u64,
        preserve_config: Option<&PreserveConfig>,
        strictness: Strictness,
    ) -> Result<Self> {
        // Check for SOI
        if data.len() < 2 || data[0] != 0xFF || data[1] != MARKER_SOI {
            return Err(Error::invalid_jpeg_data("missing SOI marker"));
        }

        // Extract ICC profile from raw data upfront
        let icc_profile = extract_icc_profile(data);

        // Initialize extras if preservation is enabled
        let (preserve_config, extras) = match preserve_config {
            Some(config) if config.preserves_any_metadata() || config.preserves_any_mpf() => {
                (Some(config.clone()), Some(DecodedExtras::new()))
            }
            _ => (None, None),
        };

        Ok(Self {
            data,
            position: 2,
            width: 0,
            height: 0,
            precision: 8,
            num_components: 0,
            mode: JpegMode::Baseline,
            components: std::array::from_fn(|_| Component::default()),
            quant_tables: [None, None, None, None],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            restart_interval: 0,
            coeffs: Vec::new(),
            coeff_counts: Vec::new(),
            nonzero_bitmaps: Vec::new(),
            streaming_rgb: None,
            decode_mode: DecodeMode::Auto,
            #[cfg(feature = "parallel")]
            fused_result: None,
            chroma_upsampling: super::ChromaUpsampling::default(),
            idct_method: super::IdctMethod::default(),
            icc_profile,
            max_pixels,
            preserve_config,
            extras,
            mpf_header_pos: 0,
            adobe_transform: None,
            strictness,
            num_threads: 0,
            #[cfg(feature = "parallel")]
            parallel_strategy: super::config::ParallelStrategy::default(),
            warnings: Vec::new(),
            arith_dc_cond: [(0, 1); 4], // Default L=0, U=1
            arith_ac_kx: [5; 4],        // Default Kx=5
            force_f32_idct: false,
        })
    }

    /// Record a warning (Balanced/Lenient) or return an error (Strict).
    ///
    /// In Strict mode, the warning is converted to an error with the warning's
    /// Display message. In Balanced/Lenient, it's collected for later retrieval.
    /// Duplicate warnings are suppressed.
    pub(super) fn warn(&mut self, warning: DecodeWarning) -> Result<()> {
        if self.strictness == Strictness::Strict {
            return Err(Error::invalid_jpeg_data(match &warning {
                DecodeWarning::MissingHuffmanTables => {
                    "missing Huffman table (DHT not defined before scan)"
                }
                DecodeWarning::TruncatedScan { .. } => "scan data truncated",
                DecodeWarning::PaddingBlockError => "padding block decode failed",
                DecodeWarning::DnlHeightConflict { .. } => {
                    "DNL marker conflicts with SOF height (spec violation)"
                }
                DecodeWarning::TruncatedProgressiveScan => "progressive scan data truncated",
                DecodeWarning::AcIndexOverflow => "AC coefficient index out of bounds",
                DecodeWarning::InvalidHuffmanCode => "invalid Huffman code mid-scan",
                DecodeWarning::ZeroQuantValue { .. } => "zero quantization value in DQT",
                DecodeWarning::MalformedSegmentSkipped => "malformed segment length",
                DecodeWarning::RestartMarkerResync { .. } => "restart marker sequence mismatch",
            }));
        }
        // Deduplicate: don't add the same warning twice
        // (MissingHuffmanTables may fire for each component, but one warning suffices)
        if !self.warnings.contains(&warning) {
            self.warnings.push(warning);
        }
        Ok(())
    }

    /// Take warnings out of the parser (for use after decode).
    pub(super) fn take_warnings(&mut self) -> Vec<DecodeWarning> {
        core::mem::take(&mut self.warnings)
    }

    /// Take the extras out of the parser (for use after decode).
    pub(super) fn take_extras(&mut self) -> Option<DecodedExtras> {
        self.extras.take().filter(|e| !e.is_empty())
    }

    /// Extract gain map byte range and metadata early (before full decode).
    ///
    /// This scans the entire JPEG for XMP and MPF markers, since these
    /// may appear after the frame header (SOF) and thus not be parsed
    /// during `read_header()`.
    ///
    /// Returns byte range `(start, end)` into `full_data` instead of copying,
    /// enabling zero-copy access to the gain map JPEG.
    #[cfg(feature = "ultrahdr")]
    pub(super) fn extract_gainmap_early(
        &mut self,
        full_data: &[u8],
    ) -> Result<(
        Option<(usize, usize)>,
        Option<ultrahdr_core::GainMapMetadata>,
    )> {
        use super::extras::{MpfDirectory, SegmentType, detect_segment_type, parse_mpf_directory};
        use ultrahdr_core::metadata::xmp::parse_xmp;

        const XMP_NS: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
        const MPF_SIG: &[u8] = b"MPF\0";

        let mut xmp_data: Option<String> = None;
        let mut mpf_directory: Option<MpfDirectory> = None;
        let mut mpf_header_pos: usize = 0; // Position of MP header (after "MPF\0")

        // Scan through the entire JPEG for XMP and MPF markers
        // We can't rely on self.extras because read_header() stops at SOF
        let mut pos = 2; // After SOI
        while pos < full_data.len() - 1 {
            // Find marker
            if full_data[pos] != 0xFF {
                pos += 1;
                continue;
            }
            pos += 1;
            if pos >= full_data.len() {
                break;
            }

            let marker = full_data[pos];
            pos += 1;

            // Skip padding bytes
            if marker == 0xFF || marker == 0x00 {
                continue;
            }

            // EOI
            if marker == MARKER_EOI {
                break;
            }

            // Markers without length
            if (0xD0..=0xD7).contains(&marker) {
                // RST markers
                continue;
            }

            // Read length
            if pos + 2 > full_data.len() {
                break;
            }
            let length = ((full_data[pos] as usize) << 8) | (full_data[pos + 1] as usize);
            if length < 2 {
                break;
            }
            pos += 2;
            let data_len = length - 2;

            if pos + data_len > full_data.len() {
                break;
            }

            let seg_data = &full_data[pos..pos + data_len];

            match marker {
                0xE1 => {
                    // APP1 - might be XMP
                    if detect_segment_type(marker, seg_data) == SegmentType::Xmp
                        && seg_data.starts_with(XMP_NS)
                        && xmp_data.is_none()
                        && let Ok(s) = core::str::from_utf8(&seg_data[XMP_NS.len()..])
                    {
                        xmp_data = Some(s.to_string());
                    }
                }
                0xE2 => {
                    // APP2 - might be MPF
                    if detect_segment_type(marker, seg_data) == SegmentType::Mpf
                        && mpf_directory.is_none()
                    {
                        mpf_directory = parse_mpf_directory(seg_data);
                        // Record the position of the MP header (right after "MPF\0")
                        // This is needed to calculate absolute offsets
                        if seg_data.starts_with(MPF_SIG) {
                            mpf_header_pos = pos + MPF_SIG.len();
                        }
                    }
                }
                _ => {}
            }

            pos += data_len;
        }

        // Parse metadata from XMP
        let metadata = xmp_data.as_ref().and_then(|xmp| {
            if !xmp.contains("hdrgm:Version") && !xmp.contains("hdrgm:GainMapMax") {
                return None;
            }
            parse_xmp(xmp).ok().map(|(m, _)| m)
        });

        // If no metadata, not an UltraHDR image
        if metadata.is_none() {
            return Ok((None, None));
        }

        // Extract gain map JPEG byte range from MPF directory
        // MPF offsets are relative to the start of the MP header (right after "MPF\0")
        let mut gainmap_range = mpf_directory.and_then(|mpf| {
            for (idx, entry) in mpf.images.iter().enumerate() {
                if idx == 0 {
                    continue; // Skip primary (offset 0 means the main image)
                }
                if entry.image_type.is_gainmap() && entry.offset > 0 {
                    // Calculate absolute offset: MPF header position + relative offset
                    let absolute_offset = mpf_header_pos + entry.offset as usize;
                    let end = absolute_offset.saturating_add(entry.size as usize);
                    if end <= full_data.len() {
                        let data = &full_data[absolute_offset..end];
                        if data.len() >= 2 && data[0] == 0xFF && data[1] == 0xD8 {
                            return Some((absolute_offset, end));
                        }
                    }
                }
            }
            None
        });

        // Fallback: if MPF parsing failed but XMP metadata indicates UltraHDR,
        // scan for JPEG boundaries (SOI/EOI markers) to find the secondary image.
        // This handles non-standard MPF structures from some Android camera apps.
        #[cfg(feature = "ultrahdr")]
        if gainmap_range.is_none() && metadata.is_some() {
            gainmap_range = find_secondary_jpeg_range(full_data);
        }

        Ok((gainmap_range, metadata))
    }

    // =========================================================================
    // Core I/O utilities
    // =========================================================================

    pub(super) fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::truncated_data("reading marker data"));
        }
        let byte = self.data[self.position];
        self.position += 1;
        Ok(byte)
    }

    pub(super) fn read_u16(&mut self) -> Result<u16> {
        let high = self.read_u8()? as u16;
        let low = self.read_u8()? as u16;
        Ok((high << 8) | low)
    }

    pub(super) fn read_marker(&mut self) -> Result<u8> {
        loop {
            // Skip until we find 0xFF
            let byte = self.read_u8()?;
            if byte != 0xFF {
                continue;
            }

            // Skip fill bytes (consecutive 0xFF)
            loop {
                let marker = self.read_u8()?;
                if marker == 0xFF {
                    // Fill byte, keep looking
                    continue;
                }
                if marker == 0x00 {
                    // Byte stuffing (0xFF 0x00 = literal 0xFF in data)
                    // This shouldn't happen in marker parsing, but skip it
                    break;
                }
                // Found a real marker
                return Ok(marker);
            }
        }
    }

    // =========================================================================
    // Component info helpers
    // =========================================================================

    /// Build component info for all components.
    ///
    /// `num_comps` allows overriding for XYB which always uses 3 components.
    pub(super) fn build_comp_infos(
        &self,
        mcu_cols: usize,
        mcu_rows: usize,
        max_h_samp: usize,
        max_v_samp: usize,
        num_comps: usize,
    ) -> Result<Vec<CompInfo>> {
        let mut comp_infos = Vec::with_capacity(num_comps);
        for comp_idx in 0..num_comps {
            let h_samp = self.components[comp_idx].h_samp_factor as usize;
            let v_samp = self.components[comp_idx].v_samp_factor as usize;
            let comp_blocks_h = mcu_cols * h_samp;
            let comp_blocks_v = mcu_rows * v_samp;
            let comp_width = checked_size_2d(comp_blocks_h, 8)?;
            let comp_height = checked_size_2d(comp_blocks_v, 8)?;
            comp_infos.push(CompInfo {
                quant_idx: self.components[comp_idx].quant_table_idx as usize,
                h_samp,
                v_samp,
                comp_blocks_h,
                comp_blocks_v,
                comp_width,
                comp_height,
                is_full_res: h_samp == max_h_samp && v_samp == max_v_samp,
            });
        }
        Ok(comp_infos)
    }

    // =========================================================================
    // Main decode orchestration
    // =========================================================================

    /// Decode the full JPEG (header + all scans).
    ///
    /// The `stop` parameter allows cancellation of long-running decodes.
    pub(super) fn decode(&mut self, stop: &impl Stop) -> Result<()> {
        // First read header
        self.position = 2; // Skip SOI
        self.read_header()?;

        // Track whether we've decoded at least one scan (for truncation recovery)
        let mut scans_decoded = 0u32;

        // Continue parsing until we hit EOI
        loop {
            // Check for cancellation
            if stop.should_stop() {
                return Err(Error::cancelled());
            }

            let marker = match self.read_marker() {
                Ok(m) => m,
                Err(e) => {
                    // In Balanced/Lenient mode, treat truncation after at least one
                    // scan as end-of-image. This handles missing EOI, truncated
                    // progressive scans, and files cut off between scans.
                    // Matches libjpeg-turbo behavior (JWRN_HIT_MARKER + partial output).
                    if self.strictness != Strictness::Strict
                        && scans_decoded > 0
                        && matches!(e.kind(), ErrorKind::TruncatedData { .. })
                    {
                        self.warnings.push(DecodeWarning::TruncatedScan {
                            blocks_decoded: 0,
                            blocks_expected: 0,
                        });
                        break;
                    }
                    return Err(e);
                }
            };

            match marker {
                MARKER_SOS => {
                    match self.parse_scan(stop) {
                        Ok(()) => {
                            scans_decoded += 1;
                            if scans_decoded >= MAX_SCANS as u32 {
                                return Err(Error::too_many_scans(
                                    scans_decoded as usize,
                                    MAX_SCANS,
                                ));
                            }
                        }
                        Err(e) => {
                            // In Balanced/Lenient mode, truncation during a scan
                            // is recoverable if we have partial data.
                            if self.strictness != Strictness::Strict
                                && matches!(e.kind(), ErrorKind::TruncatedData { .. })
                            {
                                self.warnings.push(DecodeWarning::TruncatedScan {
                                    blocks_decoded: 0,
                                    blocks_expected: 0,
                                });
                                break;
                            }
                            return Err(e);
                        }
                    }
                }
                MARKER_DNL => {
                    // Define Number of Lines - update height if it was 0 in SOF
                    self.parse_dnl()?;
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DAC => self.parse_dac()?,
                MARKER_DRI => self.parse_restart_interval()?,
                // Stray restart markers between scans — some encoders write a
                // trailing RST after the final MCU interval. These are standalone
                // 2-byte markers with no length field, so just skip them.
                0xD0..=0xD7 => {}
                MARKER_EOI => {
                    // Validate that we have a valid height (either from SOF or DNL)
                    if self.height == 0 {
                        return Err(Error::invalid_jpeg_data(
                            "image height is 0 (DNL marker missing or invalid)",
                        ));
                    }
                    // Before exiting, extract MPF secondary images if configured
                    self.extract_mpf_secondary_images()?;
                    break;
                }
                MARKER_APP0..=0xEF | MARKER_COM => self.process_app_or_com(marker)?,
                _ => self.skip_segment()?,
            }
        }

        // For progressive decode, compute actual coeff_counts from nonzero bitmaps.
        // Progressive scans initialize coeff_counts to 64 (full IDCT) because the
        // final coefficient layout isn't known until all scans complete. Now that all
        // scans are done, we can compute the actual highest nonzero zigzag position
        // per block, enabling tiered IDCT (DC-only fast path, partial dequant).
        if matches!(
            self.mode,
            JpegMode::Progressive | JpegMode::ArithmeticProgressive
        ) && !self.nonzero_bitmaps.is_empty()
        {
            for comp_idx in 0..self.nonzero_bitmaps.len() {
                let bitmaps = &self.nonzero_bitmaps[comp_idx];
                let counts = &mut self.coeff_counts[comp_idx];
                for (block_idx, bitmap) in bitmaps.iter().enumerate() {
                    if *bitmap == 0 {
                        // DC only (or all zeros) — use DC-only IDCT fast path
                        counts[block_idx] = 1;
                    } else {
                        // Highest set bit position + 1 = number of zigzag positions to process
                        counts[block_idx] = (64 - bitmap.leading_zeros()) as u8;
                    }
                }
            }
        }

        Ok(())
    }

    /// Finds the SOS marker and extracts scan info without decoding.
    /// Used by `into_scan_data()` to get table mapping and data start position.
    fn find_scan_info(&mut self) -> Result<ScanInfo> {
        // Continue from current position to find SOS
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOS => {
                    let _length = self.read_u16()?;
                    let num_components = self.read_u8()?;

                    // Support grayscale (1) and color (3) scans
                    if num_components != 1 && num_components != 3 {
                        return Err(Error::unsupported_feature(
                            "scanline reader requires 1 or 3 components in scan",
                        ));
                    }

                    // Initialize with defaults for unused components (grayscale case)
                    let mut table_mapping = [(0usize, 0usize); 3];

                    for _i in 0..num_components as usize {
                        let component_id = self.read_u8()?;
                        let tables = self.read_u8()?;
                        let dc_table = (tables >> 4) as usize;
                        let ac_table = (tables & 0x0F) as usize;

                        // Validate Huffman table indexes
                        if dc_table >= MAX_HUFFMAN_TABLES {
                            return Err(Error::invalid_jpeg_data(
                                "SOS DC Huffman table index out of range",
                            ));
                        }
                        if ac_table >= MAX_HUFFMAN_TABLES {
                            return Err(Error::invalid_jpeg_data(
                                "SOS AC Huffman table index out of range",
                            ));
                        }

                        // Find component index
                        let comp_idx = self.components[..self.num_components as usize]
                            .iter()
                            .position(|c| c.id == component_id)
                            .ok_or(Error::invalid_jpeg_data("unknown component in scan"))?;

                        table_mapping[comp_idx] = (dc_table, ac_table);
                    }

                    // Skip spectral selection bytes (Ss, Se, Ah/Al)
                    let _ss = self.read_u8()?;
                    let _se = self.read_u8()?;
                    let _ah_al = self.read_u8()?;

                    return Ok(ScanInfo {
                        table_mapping,
                        data_start: self.position,
                    });
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_APP0..=0xEF | MARKER_COM => self.process_app_or_com(marker)?,
                MARKER_EOI => {
                    return Err(Error::invalid_jpeg_data("unexpected EOI before SOS"));
                }
                _ => self.skip_segment()?,
            }
        }
    }

    /// Extract MPF secondary images after the primary image EOI.
    fn extract_mpf_secondary_images(&mut self) -> Result<()> {
        // Only process if we have MPF config and extras
        let (config, extras) = match (&self.preserve_config, &mut self.extras) {
            (Some(c), Some(e)) if c.preserves_any_mpf() => (c, e),
            _ => return Ok(()),
        };

        // Get the MPF directory from preserved segments
        let mpf_dir = match extras.mpf() {
            Some(dir) => dir.clone(), // Clone to avoid borrow issues
            None => return Ok(()),
        };

        // The current position should be right after the primary EOI
        let primary_eoi_pos = self.position;

        // MPF offsets for secondary images are relative to the TIFF header within MPF
        // (after "MPF\0" signature). We need to add the base position to get absolute offsets.
        let mpf_base = self.mpf_header_pos;

        // Extract secondary images based on MPF entries
        for (idx, entry) in mpf_dir.images.iter().enumerate() {
            // Skip primary image (index 0 or BaselinePrimary type)
            if idx == 0 || matches!(entry.image_type, MpfImageType::BaselinePrimary) {
                continue;
            }

            // Check if we should preserve this image
            if !should_preserve_mpf_image(config, idx, entry.image_type, entry.size) {
                continue;
            }

            // Calculate absolute offset
            // MPF offsets for secondary images are relative to the TIFF header (after "MPF\0")
            let offset = if entry.offset == 0 {
                // Offset 0 means immediately after primary image
                primary_eoi_pos
            } else if mpf_base > 0 {
                // Add base position to get absolute file offset
                mpf_base + entry.offset as usize
            } else {
                // Fallback: treat as absolute offset (shouldn't happen if MPF was properly parsed)
                entry.offset as usize
            };

            let end = offset.saturating_add(entry.size as usize);
            if end <= self.data.len() {
                let image_data = self.data[offset..end].to_vec();

                // Verify it looks like a JPEG (starts with SOI)
                if image_data.len() >= 2 && image_data[0] == 0xFF && image_data[1] == 0xD8 {
                    extras.add_secondary_image(idx, entry.image_type, image_data);
                }
            }
        }

        Ok(())
    }

    // =========================================================================
    // Info extraction
    // =========================================================================

    pub(super) fn info(&self) -> JpegInfo {
        let is_xyb = self
            .icc_profile
            .as_ref()
            .map(|p| is_xyb_profile(p))
            .unwrap_or(false);

        // Detect color space from component count and IDs
        let color_space = if is_xyb {
            // XYB uses RGB component IDs (82, 71, 66) but is actually XYB color space
            ColorSpace::Xyb
        } else if self.num_components == 1 {
            ColorSpace::Grayscale
        } else if self.num_components == 3 {
            // Check for RGB component IDs
            let ids: Vec<u8> = self.components[..3].iter().map(|c| c.id).collect();
            if ids == [b'R', b'G', b'B'] {
                ColorSpace::Rgb
            } else {
                ColorSpace::YCbCr
            }
        } else if self.num_components == 4 {
            ColorSpace::Cmyk
        } else {
            ColorSpace::Unknown
        };

        // Extract metadata from extras if available
        let (icc_profile, exif, xmp, jfif) = if let Some(ref extras) = self.extras {
            (
                // Prefer extras ICC (from preserved APP2 segments), fall back to
                // parser's direct extraction if extras didn't capture it.
                extras
                    .icc_profile()
                    .map(|p| p.to_vec())
                    .or_else(|| self.icc_profile.clone()),
                extras.exif().map(|e| e.to_vec()),
                extras.xmp().map(|x| x.to_string()),
                extras.jfif(),
            )
        } else {
            (self.icc_profile.clone(), None, None, None)
        };

        let dims = Dimensions {
            width: self.width,
            height: self.height,
        };
        let subsampling = super::compute_subsampling(&self.components, self.num_components);

        JpegInfo {
            dimensions: dims,
            color_space,
            precision: self.precision,
            num_components: self.num_components,
            mode: self.mode,
            subsampling,
            has_icc_profile: icc_profile.is_some(),
            is_xyb,
            icc_profile,
            exif,
            xmp,
            jfif,
        }
    }

    /// Extract scan data for constructing a scanline reader.
    ///
    /// This bundles all the parser fields needed by `ScanlineReader::new()`,
    /// detecting RGB-JPEG and XYB along the way.
    pub(super) fn into_scan_data(mut self, is_grayscale: bool) -> Result<ParsedScanData<'a>> {
        let is_xyb = self.info().is_xyb;

        // Extract sampling factors
        let h_samp = if is_grayscale {
            [self.components[0].h_samp_factor, 1, 1]
        } else {
            [
                self.components[0].h_samp_factor,
                self.components[1].h_samp_factor,
                self.components[2].h_samp_factor,
            ]
        };
        let v_samp = if is_grayscale {
            [self.components[0].v_samp_factor, 1, 1]
        } else {
            [
                self.components[0].v_samp_factor,
                self.components[1].v_samp_factor,
                self.components[2].v_samp_factor,
            ]
        };

        // Extract quant table indices
        let quant_indices = if is_grayscale {
            [self.components[0].quant_table_idx as usize, 0, 0]
        } else {
            [
                self.components[0].quant_table_idx as usize,
                self.components[1].quant_table_idx as usize,
                self.components[2].quant_table_idx as usize,
            ]
        };

        // Find SOS marker to get table mapping and scan data position
        let scan_info = self.find_scan_info()?;

        // Detect RGB JPEGs
        let is_rgb = if self.num_components == 3 {
            match self.adobe_transform {
                Some(AdobeColorTransform::Unknown) => true,
                Some(AdobeColorTransform::YCbCr) => false,
                None => {
                    self.components[0].id == b'R'
                        && self.components[1].id == b'G'
                        && self.components[2].id == b'B'
                }
                _ => false,
            }
        } else {
            false
        };

        Ok(ParsedScanData {
            data: self.data,
            width: self.width,
            height: self.height,
            num_components: self.num_components,
            h_samp,
            v_samp,
            quant_tables: self.quant_tables,
            quant_indices,
            dc_tables: self.dc_tables,
            ac_tables: self.ac_tables,
            table_mapping: scan_info.table_mapping,
            scan_data_start: scan_info.data_start,
            restart_interval: self.restart_interval,
            is_xyb,
            is_rgb,
        })
    }

    pub(super) fn extract_coefficients(&self) -> Result<crate::decode::image::DecodedCoefficients> {
        use crate::decode::image::{ComponentCoefficients, DecodedCoefficients};

        if self.coeffs.is_empty() {
            return Err(Error::internal("no coefficients decoded"));
        }

        // Calculate MCU dimensions
        let mut max_h_samp = 1u8;
        let mut max_v_samp = 1u8;
        for i in 0..self.num_components as usize {
            max_h_samp = max_h_samp.max(self.components[i].h_samp_factor);
            max_v_samp = max_v_samp.max(self.components[i].v_samp_factor);
        }
        let mcu_width = (max_h_samp as usize) * 8;
        let mcu_height = (max_v_samp as usize) * 8;
        let mcu_cols = (self.width as usize + mcu_width - 1) / mcu_width;
        let mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;

        let mut components = Vec::with_capacity(self.num_components as usize);

        for i in 0..self.num_components as usize {
            let h_samp = self.components[i].h_samp_factor as usize;
            let v_samp = self.components[i].v_samp_factor as usize;
            let blocks_wide = mcu_cols * h_samp;
            let blocks_high = mcu_rows * v_samp;

            // Flatten block coefficients from Vec<[i16; 64]> to Vec<i16>
            let coeffs: Vec<i16> = self.coeffs[i]
                .iter()
                .flat_map(|block| block.iter().copied())
                .collect();

            components.push(ComponentCoefficients {
                id: self.components[i].id,
                coeffs,
                blocks_wide,
                blocks_high,
                h_samp: h_samp as u8,
                v_samp: v_samp as u8,
                quant_table_idx: self.components[i].quant_table_idx,
            });
        }

        // Collect quantization tables
        let quant_tables: Vec<Option<[u16; 64]>> = self.quant_tables.to_vec();

        Ok(DecodedCoefficients {
            width: self.width,
            height: self.height,
            components,
            quant_tables,
        })
    }
}

/// Find the second JPEG's byte range in a multi-picture file by scanning for SOI/EOI markers.
///
/// This is a fallback for when MPF parsing fails but XMP metadata indicates UltraHDR.
/// Some Android camera apps produce non-standard MPF structures that fail to parse,
/// but we can still find the gain map JPEG by looking for JPEG boundaries.
///
/// Returns the second JPEG's byte range `(start, end)` if found, or None if not found.
#[cfg(feature = "ultrahdr")]
fn find_secondary_jpeg_range(data: &[u8]) -> Option<(usize, usize)> {
    const SOI: [u8; 2] = [0xFF, 0xD8]; // Start of Image

    // Find all JPEG boundaries (SOI to EOI)
    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0;

    while pos < data.len().saturating_sub(1) {
        // Find SOI marker
        if data[pos] == SOI[0] && data[pos + 1] == SOI[1] {
            let start = pos;
            pos += 2;

            // Find corresponding EOI marker
            // We need to be careful to skip 0xFFD9 bytes inside entropy-coded data
            // by properly parsing markers
            while pos < data.len().saturating_sub(1) {
                if data[pos] == 0xFF {
                    let marker = data[pos + 1];
                    if marker == 0xD9 {
                        // EOI found
                        let end = pos + 2;
                        boundaries.push((start, end));
                        pos = end;
                        break;
                    } else if marker == 0x00 {
                        // Byte stuffing (0xFF 0x00), skip
                        pos += 2;
                    } else if marker == 0xFF {
                        // Fill byte, skip single byte
                        pos += 1;
                    } else if marker == 0xD8 {
                        // Another SOI - this might be an embedded JPEG
                        // Don't skip, let outer loop handle it
                        break;
                    } else if (0xD0..=0xD7).contains(&marker) {
                        // RST marker (no length)
                        pos += 2;
                    } else if (0xC0..=0xFE).contains(&marker) {
                        // Marker with length field
                        if pos + 4 > data.len() {
                            break;
                        }
                        let len = ((data[pos + 2] as usize) << 8) | (data[pos + 3] as usize);
                        pos += 2 + len;
                    } else {
                        pos += 1;
                    }
                } else {
                    pos += 1;
                }
            }
        } else {
            pos += 1;
        }
    }

    // If we found at least 2 JPEGs, return the second one's range (the gain map)
    if boundaries.len() >= 2 {
        let (start, end) = boundaries[1];
        if end <= data.len() {
            return Some((start, end));
        }
    }

    None
}
