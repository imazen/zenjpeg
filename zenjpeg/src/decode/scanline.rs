//! Pull-based scanline decoder for streaming JPEG decoding.
//!
//! This module provides a scanline-by-scanline decoder that allows reading
//! JPEG images row by row without loading the entire image into memory.
//!
//! # Example
//! ```ignore
//! use zenjpeg::{Decoder, ImgRefMut};
//!
//! let mut reader = Decoder::new().scanline_reader(&jpeg_data)?;
//! let width = reader.width() as usize;
//! let height = reader.height() as usize;
//!
//! // Allocate output buffer
//! let mut pixels = vec![0u8; width * height * 3];
//!
//! // Read in chunks
//! let mut rows_read = 0;
//! while rows_read < height {
//!     let remaining = height - rows_read;
//!     let output = ImgRefMut::new(&mut pixels[rows_read * width * 3..], width, remaining);
//!     let count = reader.read_rows_rgb8(output)?;
//!     rows_read += count;
//! }
//! ```

use alloc::borrow::Cow;

use super::DeblockMode;
use super::color_model::ColorModel;
use super::config::ResolvedCrop;
use super::pipeline::StripProcessor;
use super::pool::PoolGuard;
use crate::color::planes::{
    Layout8, PlaneMatrix, RowKernel, gray8_row, rgb_luma_f32, rgb_to_ycbcr_normalized, rgb8_row,
};
use crate::color::ycbcr_to_rgb_f32;
use crate::deblock::BoundaryStrength;
use crate::entropy::{EntropyDecoder, EntropyDecoderState};
use crate::error::{Error, Result, ScanRead};
use crate::foundation::consts::{DCT_BLOCK_SIZE, MAX_COMPONENTS, MAX_HUFFMAN_TABLES};
use crate::huffman::HuffmanDecodeTable;
use crate::types::{ColorSpace, Dimensions, Subsampling};
use imgref::ImgRefMut;

/// Information about the JPEG being decoded.
#[derive(Debug, Clone)]
pub struct ScanlineInfo {
    /// Image dimensions
    pub dimensions: Dimensions,
    /// Color space
    pub color_space: ColorSpace,
    /// Whether this is an XYB image
    pub is_xyb: bool,
    /// Chroma subsampling mode
    pub subsampling: Subsampling,
}

/// Pull-based scanline reader for JPEG decoding.
///
/// Decodes JPEG images row by row, only decoding MCU rows as needed.
/// This minimizes memory usage and allows early processing of image data.
///
/// For progressive JPEGs, the image is fully decoded upfront and served
/// from a buffer, since progressive encoding requires all scans to be
/// processed before final pixels are available.
pub struct ScanlineReader<'a> {
    // Raw JPEG data (unused in buffered mode).
    // `Cow::Borrowed` when created from a slice; `Cow::Owned` when the caller
    // donates a `Vec<u8>` so the reader can be `'static`.
    data: Cow<'a, [u8]>,

    // Image dimensions
    width: u32,
    height: u32,
    num_components: u8,

    // Buffered mode for progressive JPEGs
    // When Some, we serve from this buffer instead of decoding on-the-fly
    buffered_rgb: Option<Vec<u8>>,
    precision: u8,
    component_rows: Option<super::parser::component_rows::ComponentRows>,
    component_ids: [u8; MAX_COMPONENTS],
    predictive: Option<super::parser::predictive::PredictiveReader>,
    buffered_native: Option<Vec<u16>>,

    // Strip processing (IDCT + upsampling)
    strip: StripProcessor,

    // Current position
    current_row: usize,     // Current output row (0 to height-1)
    current_mcu_row: usize, // Current MCU row being processed
    row_in_mcu: usize,      // Row within current MCU (0 to mcu_height-1)
    mcu_row_decoded: bool,  // Whether current MCU row has been decoded

    // Quantization tables (copied, since we outlive the parser)
    quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; 4],
    quant_indices: [usize; 3], // Which quant table each component uses

    // Huffman tables (copied)
    dc_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    ac_tables: [Option<HuffmanDecodeTable>; MAX_HUFFMAN_TABLES],
    table_mapping: [(usize, usize); 3], // (dc_table, ac_table) for each component

    // Entropy decoder state
    scan_data_start: usize, // Position where scan data begins
    decoder_state: Option<EntropyDecoderState>, // Saved state for resuming (None = start of scan)

    // Restart markers
    restart_interval: u16,
    mcu_count: u32,
    next_restart_num: u8,

    // Reusable buffers for zero-copy decode
    coeffs_buf: [i16; DCT_BLOCK_SIZE],
    /// Track previous coefficient count per component for smart zeroing
    prev_coeff_counts: [u8; 4],

    // The frame's colour model (decided at SOF) and the plane conversion the
    // strip readers apply (for buffered and wave readers, which hold decoded
    // RGB, it is unused).
    color_model: ColorModel,
    planes: PlaneMatrix,

    // Coefficient-based mode for decode-time transforms.
    // When Some, we read from pre-decoded/transformed coefficients instead of entropy decoding.
    stored_coeffs: Option<super::DecodedCoefficients>,

    // Streaming bottom-boundary fixup state.
    // When the streaming decoder pre-decodes the next MCU row to provide
    // bottom chroma context, this flag indicates the next MCU is already decoded.
    next_mcu_preloaded: bool,

    // Crop-on-decode state.
    // When Some, only IDCT/upsample MCU rows overlapping the crop region.
    // Entropy decoding still runs for all rows (DC predictor chain).
    crop: Option<ResolvedCrop>,
    // Whether skip_to_crop_start() has been called.
    crop_skip_done: bool,

    // Wave-based parallel decode state.
    // When Some, decode wave_size segments at a time via rayon instead of streaming.
    #[cfg(feature = "parallel")]
    wave_state: Option<super::fused_parallel::WaveParallelState>,
    #[cfg(feature = "parallel")]
    wave_buf: Vec<u8>,
    #[cfg(feature = "parallel")]
    wave_first_row: usize,
    #[cfg(feature = "parallel")]
    wave_row_count: usize,
    #[cfg(feature = "parallel")]
    wave_next_seg: usize,

    // Wave planar decode buffers (lazy-allocated on first read_rows_ycbcr_native_i16 call)
    #[cfg(feature = "parallel")]
    wave_y: Vec<i16>,
    #[cfg(feature = "parallel")]
    wave_cb: Vec<i16>,
    #[cfg(feature = "parallel")]
    wave_cr: Vec<i16>,
    #[cfg(feature = "parallel")]
    wave_planar_first_luma_row: usize,
    #[cfg(feature = "parallel")]
    wave_planar_luma_rows: usize,
    #[cfg(feature = "parallel")]
    wave_planar_first_chroma_row: usize,
    #[cfg(feature = "parallel")]
    wave_planar_chroma_rows: usize,
    #[cfg(feature = "parallel")]
    wave_planar_next_seg: usize,

    // Pool guard for adaptive threading. When Some, the pool slot is held
    // for the lifetime of this reader and released on drop.
    pub(super) pool_guard: Option<PoolGuard<'a>>,

    // Boundary 4-tap deblocking state.
    // When deblock_mode is Off, all deblock fields are inert (None / empty)
    // and no deblock code runs — zero overhead.
    deblock_mode: DeblockMode,
    /// Per-component boundary strength, computed once from DC quant tables.
    /// None when deblock_mode is Off.
    deblock_strength: Option<[BoundaryStrength; 3]>,
    /// Last 2 rows of each i16 plane from the previous MCU row, for
    /// horizontal boundary filtering at MCU row junctions.
    /// Layout: [y_row_minus2, y_row_minus1, cb_row_minus2, cb_row_minus1,
    ///          cr_row_minus2, cr_row_minus1], each `strip_stride` i16 values.
    /// Empty when deblock_mode is Off.
    deblock_prev_rows: Vec<i16>,
    /// Whether `deblock_prev_rows` contains valid data from a previous MCU row.
    deblock_has_prev: bool,
}

/// CMYK channel values in the JPEG sample range, without ICC conversion.
/// Values retain JPEG's channel convention, including Adobe inversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cmyk16 {
    /// Cyan channel.
    pub c: u16,
    /// Magenta channel.
    pub m: u16,
    /// Yellow channel.
    pub y: u16,
    /// Black channel.
    pub k: u16,
}

impl<'a> ScanlineReader<'a> {
    /// Creates a new scanline reader from parsed JPEG scan data.
    ///
    /// This is called internally by `Decoder::scanline_reader()`.
    pub(super) fn from_scan_data(
        scan: super::parser::ParsedScanData<'a>,
        chroma_upsampling: super::ChromaUpsampling,
        idct_method: super::IdctMethod,
        output_target: super::OutputTarget,
        deblock_mode: DeblockMode,
        alloc_pref: zencodec::AllocPreference,
    ) -> Result<Self> {
        let data_cow = Cow::Borrowed(scan.data);
        Self::from_scan_data_cow(
            scan,
            data_cow,
            chroma_upsampling,
            idct_method,
            output_target,
            deblock_mode,
            alloc_pref,
        )
    }

    /// Like [`from_scan_data`](Self::from_scan_data), but accepts a `Cow` for
    /// the raw JPEG data. When `Cow::Owned`, the reader owns the data and can
    /// be `'static`.
    ///
    /// `scan_data`'s `.data` field is ignored — `data_cow` is stored instead.
    pub(super) fn from_scan_data_cow(
        scan: super::parser::ParsedScanData<'_>,
        data_cow: Cow<'a, [u8]>,
        chroma_upsampling: super::ChromaUpsampling,
        idct_method: super::IdctMethod,
        output_target: super::OutputTarget,
        deblock_mode: DeblockMode,
        alloc_pref: zencodec::AllocPreference,
    ) -> Result<Self> {
        let super::parser::ParsedScanData {
            precision,
            data: _,
            width,
            height,
            num_components,
            h_samp,
            v_samp,
            quant_tables,
            quant_indices,
            dc_tables,
            ac_tables,
            table_mapping,
            scan_data_start,
            restart_interval,
            color_model,
            shape: _,
            interleaved: _,
        } = scan;

        let mut strip = StripProcessor::new(
            width,
            num_components,
            h_samp,
            v_samp,
            chroma_upsampling,
            idct_method,
            output_target,
            alloc_pref,
        )?;

        strip.precision = precision;
        if precision == 12 {
            strip.idct_method = super::IdctMethod::Libjpeg;
            if width <= 4 && matches!(strip.subsampling, Subsampling::S422 | Subsampling::S420) {
                strip.chroma_upsampling = super::ChromaUpsampling::NearestNeighbor;
            }
        }

        // Resolve deblock mode: Auto/AutoStreamable select Boundary4Tap in streaming.
        // (Knusperli requires coefficients, not available here — but Auto with Knusperli
        // is caught earlier and routed through the coefficient fallback path.)
        let effective_deblock = match deblock_mode {
            DeblockMode::Auto | DeblockMode::AutoStreamable => DeblockMode::Boundary4Tap,
            other => other,
        };

        // Compute per-component boundary strength from DC quant values.
        // Only allocate deblock buffers when actually filtering.
        let active = effective_deblock == DeblockMode::Boundary4Tap;
        let deblock_strength = if active {
            let mut strengths = [BoundaryStrength::from_dc_quant(1); 3];
            for i in 0..num_components.min(3) as usize {
                if let Some(ref qt) = quant_tables[quant_indices[i]] {
                    strengths[i] = BoundaryStrength::from_dc_quant(qt[0]);
                }
            }
            Some(strengths)
        } else {
            None
        };

        // Allocate prev-row buffer for horizontal boundary filtering.
        // Layout: [Y_row-2, Y_row-1, Cb_row-2, Cb_row-1, Cr_row-2, Cr_row-1]
        // Y rows use strip_stride, Cb/Cr use chroma_strip_stride.
        let deblock_prev_rows = if active {
            let ys = strip.strip_stride;
            let cs = strip.chroma_strip_stride;
            let total = if num_components == 1 {
                2 * ys
            } else {
                2 * ys + 2 * cs + 2 * cs
            };
            alloc::vec![0i16; total]
        } else {
            Vec::new()
        };

        Ok(Self {
            component_ids: frame_component_ids(&data_cow),
            data: data_cow,
            width,
            height,
            num_components,
            buffered_rgb: None,
            precision,
            predictive: None,
            component_rows: None,
            buffered_native: None,
            strip,
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
            quant_tables,
            quant_indices,
            dc_tables,
            ac_tables,
            table_mapping,
            scan_data_start,
            decoder_state: None,
            restart_interval,
            mcu_count: 0,
            next_restart_num: 0,
            coeffs_buf: [0i16; DCT_BLOCK_SIZE],
            prev_coeff_counts: [64; 4],
            color_model,
            planes: plane_matrix(color_model, idct_method)?,
            stored_coeffs: None,
            next_mcu_preloaded: false,
            crop: None,
            crop_skip_done: false,
            #[cfg(feature = "parallel")]
            wave_state: None,
            #[cfg(feature = "parallel")]
            wave_buf: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_first_row: 0,
            #[cfg(feature = "parallel")]
            wave_row_count: 0,
            #[cfg(feature = "parallel")]
            wave_next_seg: 0,
            #[cfg(feature = "parallel")]
            wave_y: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cb: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cr: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_planar_first_luma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_luma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_first_chroma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_chroma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_next_seg: 0,
            pool_guard: None,
            deblock_mode: effective_deblock,
            deblock_strength,
            deblock_prev_rows,
            deblock_has_prev: false,
        })
    }

    /// Creates a scanline reader backed by pre-decoded DCT coefficients — the
    /// coefficient-centric buffered path (see docs/DECODER_UNIFICATION_PLAN.md,
    /// issue #187).
    ///
    /// Progressive/arithmetic JPEGs can't stream (a scan refines coefficients
    /// across the whole image), so their coefficients are decoded up front.
    /// Unlike [`new_buffered`](Self::new_buffered) — which stores pre-converted
    /// RGB and then re-derives YCbCr/gray *lossily* on demand — this stores the
    /// coefficients and runs the SAME strip pipeline as streaming decode
    /// (`decode_mcu_row_from_coefficients` → IDCT → upsample → the native
    /// read paths). Every output format is produced natively, and the read
    /// methods take their streaming branch instead of forking on `buffered_rgb`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_coefficients(
        data: Cow<'a, [u8]>,
        coeffs: super::DecodedCoefficients,
        chroma_upsampling: super::ChromaUpsampling,
        idct_method: super::IdctMethod,
        output_target: super::OutputTarget,
        deblock_mode: DeblockMode,
        alloc_pref: zencodec::AllocPreference,
        color_model: ColorModel,
        precision: u8,
    ) -> Result<Self> {
        let width = coeffs.width;
        let height = coeffs.height;
        let num_components = coeffs.components.len() as u8;

        let mut h_samp = [1u8; 3];
        let mut v_samp = [1u8; 3];
        let mut quant_indices = [0usize; 3];
        for (i, c) in coeffs.components.iter().take(3).enumerate() {
            h_samp[i] = c.h_samp;
            v_samp[i] = c.v_samp;
            quant_indices[i] = c.quant_table_idx as usize;
        }

        if num_components == 1 && precision == 12 {
            h_samp = [1; 3];
            v_samp = [1; 3];
        }
        let mut quant_tables: [Option<[u16; DCT_BLOCK_SIZE]>; 4] = [None; 4];
        for (i, slot) in quant_tables.iter_mut().enumerate() {
            *slot = coeffs.quant_tables.get(i).copied().flatten();
        }

        let mut strip = StripProcessor::new(
            width,
            num_components,
            h_samp,
            v_samp,
            chroma_upsampling,
            idct_method,
            output_target,
            alloc_pref,
        )?;

        strip.precision = precision;
        if precision == 12 {
            strip.idct_method = super::IdctMethod::Libjpeg;
            if width <= 4 && matches!(strip.subsampling, Subsampling::S422 | Subsampling::S420) {
                strip.chroma_upsampling = super::ChromaUpsampling::NearestNeighbor;
            }
        }

        // Deblock setup — identical to the streaming constructor.
        let effective_deblock = match deblock_mode {
            DeblockMode::Auto | DeblockMode::AutoStreamable => DeblockMode::Boundary4Tap,
            other => other,
        };
        let active = effective_deblock == DeblockMode::Boundary4Tap;
        let deblock_strength = if active {
            let mut strengths = [BoundaryStrength::from_dc_quant(1); 3];
            for i in 0..num_components.min(3) as usize {
                if let Some(ref qt) = quant_tables[quant_indices[i]] {
                    strengths[i] = BoundaryStrength::from_dc_quant(qt[0]);
                }
            }
            Some(strengths)
        } else {
            None
        };
        let deblock_prev_rows = if active {
            let ys = strip.strip_stride;
            let cs = strip.chroma_strip_stride;
            let total = if num_components == 1 {
                2 * ys
            } else {
                2 * ys + 2 * cs + 2 * cs
            };
            alloc::vec![0i16; total]
        } else {
            Vec::new()
        };

        Ok(Self {
            component_ids: frame_component_ids(&data),
            data,
            width,
            height,
            num_components,
            buffered_rgb: None,
            precision,
            predictive: None,
            component_rows: None,
            buffered_native: None,
            strip,
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
            quant_tables,
            quant_indices,
            dc_tables: core::array::from_fn(|_| None),
            ac_tables: core::array::from_fn(|_| None),
            table_mapping: [(0, 0); 3],
            scan_data_start: 0,
            decoder_state: None,
            restart_interval: 0,
            mcu_count: 0,
            next_restart_num: 0,
            coeffs_buf: [0i16; DCT_BLOCK_SIZE],
            prev_coeff_counts: [64; 4],
            color_model,
            planes: plane_matrix(color_model, idct_method)?,
            stored_coeffs: Some(coeffs),
            next_mcu_preloaded: false,
            crop: None,
            crop_skip_done: false,
            #[cfg(feature = "parallel")]
            wave_state: None,
            #[cfg(feature = "parallel")]
            wave_buf: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_first_row: 0,
            #[cfg(feature = "parallel")]
            wave_row_count: 0,
            #[cfg(feature = "parallel")]
            wave_next_seg: 0,
            #[cfg(feature = "parallel")]
            wave_y: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cb: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cr: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_planar_first_luma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_luma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_first_chroma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_chroma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_next_seg: 0,
            pool_guard: None,
            deblock_mode: effective_deblock,
            deblock_strength,
            deblock_prev_rows,
            deblock_has_prev: false,
        })
    }

    /// Creates a new scanline reader in buffered mode (for progressive JPEGs).
    ///
    /// In buffered mode, the image has already been decoded and we serve
    /// rows from the pre-decoded buffer.
    pub(crate) fn new_buffered(
        data: &'a [u8],
        width: u32,
        height: u32,
        num_components: u8,
        subsampling: Subsampling,
        pixels: Vec<u8>,
        color_model: ColorModel,
    ) -> Self {
        Self::new_buffered_cow(
            Cow::Borrowed(data),
            width,
            height,
            num_components,
            subsampling,
            pixels,
            color_model,
        )
    }

    /// Like [`new_buffered`](Self::new_buffered), but accepts a `Cow` for the
    /// raw JPEG data. When `Cow::Owned`, the reader owns the data.
    pub(crate) fn new_buffered_cow(
        data: Cow<'a, [u8]>,
        width: u32,
        height: u32,
        num_components: u8,
        subsampling: Subsampling,
        pixels: Vec<u8>,
        color_model: ColorModel,
    ) -> Self {
        Self {
            component_ids: frame_component_ids(&data),
            data,
            width,
            height,
            num_components,
            buffered_rgb: Some(pixels),
            precision: 8,
            predictive: None,
            component_rows: None,
            buffered_native: None,
            strip: StripProcessor::new_dummy(subsampling),
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
            quant_tables: [None, None, None, None],
            quant_indices: [0, 0, 0],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            table_mapping: [(0, 0), (0, 0), (0, 0)],
            scan_data_start: 0,
            decoder_state: None,
            restart_interval: 0,
            mcu_count: 0,
            next_restart_num: 0,
            coeffs_buf: [0i16; DCT_BLOCK_SIZE],
            prev_coeff_counts: [64; 4],
            color_model,
            planes: PlaneMatrix::Identity,
            stored_coeffs: None,
            next_mcu_preloaded: false,
            crop: None,
            crop_skip_done: false,
            #[cfg(feature = "parallel")]
            wave_state: None,
            #[cfg(feature = "parallel")]
            wave_buf: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_first_row: 0,
            #[cfg(feature = "parallel")]
            wave_row_count: 0,
            #[cfg(feature = "parallel")]
            wave_next_seg: 0,
            #[cfg(feature = "parallel")]
            wave_y: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cb: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cr: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_planar_first_luma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_luma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_first_chroma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_chroma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_next_seg: 0,
            pool_guard: None,
            deblock_mode: DeblockMode::Off,
            deblock_strength: None,
            deblock_prev_rows: Vec::new(),
            deblock_has_prev: false,
        }
    }

    pub(super) fn from_predictive(
        data: Cow<'a, [u8]>,
        width: u32,
        height: u32,
        channels: u8,
        precision: u8,
        predictive: super::parser::predictive::PredictiveReader,
        model: ColorModel,
        ids: [u8; MAX_COMPONENTS],
    ) -> Self {
        let mut reader = Self::new_buffered_cow(
            data,
            width,
            height,
            channels,
            Subsampling::S444,
            Vec::new(),
            model,
        );
        reader.buffered_rgb = None;
        reader.precision = precision;
        reader.predictive = Some(predictive);
        reader.component_ids = ids;
        reader
    }

    pub(super) fn from_native_result(
        data: Cow<'a, [u8]>,
        result: super::DecodeResult,
    ) -> Result<Self> {
        let precision = result.sample_precision();
        let channels = result.format().num_channels() as u8;
        let mut reader = Self::new_buffered_cow(
            data,
            result.width(),
            result.height(),
            channels,
            Subsampling::S444,
            Vec::new(),
            if result.format() == crate::types::PixelFormat::Cmyk {
                ColorModel::Cmyk
            } else if channels == 1 {
                ColorModel::Gray
            } else {
                ColorModel::Rgb
            },
        );
        let super::OwnedDecodedPixels::U16(samples) = result.into_pixels() else {
            return Err(Error::internal("expected native samples"));
        };
        reader.buffered_rgb = None;
        reader.buffered_native = Some(samples);
        reader.precision = precision;
        Ok(reader)
    }

    /// Sample precision declared by the JPEG frame, before output conversion.
    pub fn sample_precision(&self) -> u8 {
        self.precision
    }

    pub(super) fn replace_data(&mut self, data: Vec<u8>) {
        self.data = Cow::Owned(data);
    }

    pub(super) fn from_component_rows(
        data: Cow<'a, [u8]>,
        parser: &super::parser::JpegParser<'_>,
        source: super::parser::component_rows::ComponentRows,
    ) -> Self {
        let mut reader = Self::new_buffered_cow(
            data,
            parser.width,
            parser.height,
            parser.num_components,
            Subsampling::S444,
            Vec::new(),
            parser.color_model,
        );
        reader.buffered_rgb = None;
        reader.precision = parser.precision;
        reader.component_ids = core::array::from_fn(|i| parser.components[i].id);
        reader.component_rows = Some(source);
        reader
    }

    /// Component identifiers in frame order. Raw component pixels use this order.
    pub fn component_ids(&self) -> &[u8] {
        &self.component_ids[..self.num_components as usize]
    }

    /// Read full-resolution components in frame order, without color conversion.
    /// `N` must equal the frame component count. Stride is in N-channel pixels.
    /// Samples retain source precision and transfer function; DCT components are
    /// upsampled using the configured filter. No ICC conversion is performed.
    pub fn read_rows_components16<const N: usize>(
        &mut self,
        mut output: ImgRefMut<'_, [u16; N]>,
    ) -> Result<usize> {
        let width = self.width() as usize;
        if N != self.num_components as usize {
            return Err(Error::unsupported_feature(
                "component pixel channel count does not match frame",
            ));
        }
        if output.width() < width {
            return Err(Error::invalid_buffer_size(width, output.width()));
        }
        if self.buffered_rgb.is_some() || self.buffered_native.is_some() || self.is_wave_only() {
            return Err(Error::unsupported_feature(
                "reader holds converted pixels, not source components",
            ));
        }
        let (rows, stride) = (output.height(), output.stride());
        let max = ((1u32 << self.precision) - 1) as i32;
        let buf = output.buf_mut();
        self.read_component_rows_with(rows, true, |i, row| {
            for x in 0..width {
                buf[i * stride + x] = core::array::from_fn(|c| match row {
                    RowSrc::Native(p, n, _) => p[x * n + c],
                    RowSrc::Planes(y, cb, cr) => ([y, cb, cr][c][x] as i32).clamp(0, max) as u16,
                    _ => unreachable!(),
                });
            }
        })
    }

    /// Read CMYK samples, preserving the source range and JPEG inversion convention.
    /// YCCK is converted to CMYK with the libjpeg matrix; no ICC conversion is done.
    /// Stride is measured in CMYK pixels.
    pub fn read_rows_cmyk16(&mut self, mut output: ImgRefMut<'_, Cmyk16>) -> Result<usize> {
        if !matches!(self.color_model, ColorModel::Cmyk | ColorModel::Ycck)
            || (self.predictive.is_none()
                && self.component_rows.is_none()
                && self.buffered_native.is_none())
        {
            return Err(Error::unsupported_feature(
                "CMYK16 requires native CMYK/YCCK components",
            ));
        }
        let width = self.width() as usize;
        if output.width() < width {
            return Err(Error::invalid_buffer_size(width, output.width()));
        }
        let (rows, stride) = (output.height(), output.stride());
        let precision = self.precision;
        let max = ((1u32 << precision) - 1) as u16;
        let buf = output.buf_mut();
        self.read_component_rows_with(rows, true, |i, row| {
            let RowSrc::Native(p, 4, model) = row else {
                unreachable!()
            };
            for x in 0..width {
                let p = &p[x * 4..x * 4 + 4];
                let [c, m, y] = if model == ColorModel::Ycck {
                    native_ycbcr([p[0], p[1], p[2]], precision).map(|v| max - v)
                } else {
                    [p[0], p[1], p[2]]
                };
                buf[i * stride + x] = Cmyk16 { c, m, y, k: p[3] };
            }
        })
    }

    /// Read RGB in the frame's native integer range (0–4095 for 12-bit JPEG).
    /// Retains source primaries and transfer function. Stride is in RGB pixels.
    pub fn read_rows_rgb16(&mut self, mut output: ImgRefMut<'_, rgb::RGB<u16>>) -> Result<usize> {
        let width = self.width() as usize;
        if output.width() < width {
            return Err(Error::invalid_buffer_size(width, output.width()));
        }
        let (rows, stride) = (output.height(), output.stride());
        let (precision, matrix) = (self.precision, self.planes);
        let buf = output.buf_mut();
        self.read_rows_with(rows, |i, row| {
            for x in 0..width {
                let [r, g, b] = row.rgb16(x, precision, matrix);
                buf[i * stride + x] = rgb::RGB::new(r, g, b);
            }
        })
    }

    /// Read gray/luma samples in the frame's native integer range.
    /// Gray and YCbCr retain their gray/Y channel; RGB uses BT.601 luma.
    /// Stride is in single-channel u16 samples.
    pub fn read_rows_gray16(&mut self, mut output: ImgRefMut<'_, u16>) -> Result<usize> {
        let width = self.width() as usize;
        if output.width() < width {
            return Err(Error::invalid_buffer_size(width, output.width()));
        }
        self.check_plane_read("read_rows_gray16 needs the luma plane (use num_threads(1))")?;
        let (rows, stride) = (output.height(), output.stride());
        let (precision, matrix) = (self.precision, self.planes);
        let buf = output.buf_mut();
        self.read_rows_with(rows, |i, row| {
            for x in 0..width {
                buf[i * stride + x] = row.gray16(x, precision, matrix);
            }
        })
    }

    /// Sets the crop region for this reader.
    ///
    /// Must be called before reading any rows.
    pub(crate) fn set_crop(&mut self, crop: ResolvedCrop) {
        self.crop = Some(crop);
    }

    /// Creates a scanline reader in wave-parallel mode.
    ///
    /// Instead of decoding one MCU row at a time (streaming) or the full image
    /// (buffered), this decodes `wave_size` restart segments at a time via rayon,
    /// serving rows from a reusable buffer. Gives parallel speed with bounded
    /// memory: `wave_size * pixel_rows_per_seg * width * 3` bytes.
    ///
    /// **Prototype scope**: Box filter 4:2:0 path only.
    #[cfg(feature = "parallel")]
    pub(super) fn new_wave_parallel(
        data: &'a [u8],
        wave_state: super::fused_parallel::WaveParallelState,
    ) -> Self {
        Self::new_wave_parallel_cow(Cow::Borrowed(data), wave_state)
    }

    /// Like [`new_wave_parallel`](Self::new_wave_parallel), but accepts a `Cow`
    /// for the raw JPEG data. When `Cow::Owned`, the reader owns the data.
    #[cfg(feature = "parallel")]
    pub(super) fn new_wave_parallel_cow(
        data: Cow<'a, [u8]>,
        wave_state: super::fused_parallel::WaveParallelState,
    ) -> Self {
        let width = wave_state.width as u32;
        let height = wave_state.height as u32;

        let rgb_row_bytes = wave_state.width * 3;
        let seg_rgb_bytes = wave_state.pixel_rows_per_seg * rgb_row_bytes;
        let wave_buf_size = wave_state.wave_size * seg_rgb_bytes;
        let wave_buf = vec![0u8; wave_buf_size];

        Self {
            component_ids: frame_component_ids(&data),
            data,
            width,
            height,
            num_components: 3,
            buffered_rgb: None,
            precision: 8,
            predictive: None,
            component_rows: None,
            buffered_native: None,
            strip: StripProcessor::new_dummy(wave_state.subsampling()),
            current_row: 0,
            current_mcu_row: 0,
            row_in_mcu: 0,
            mcu_row_decoded: false,
            quant_tables: [None, None, None, None],
            quant_indices: [0, 0, 0],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            table_mapping: [(0, 0), (0, 0), (0, 0)],
            scan_data_start: 0,
            decoder_state: None,
            restart_interval: 0,
            mcu_count: 0,
            next_restart_num: 0,
            coeffs_buf: [0i16; DCT_BLOCK_SIZE],
            prev_coeff_counts: [64; 4],
            color_model: wave_state.color_model,
            planes: PlaneMatrix::Identity,
            stored_coeffs: None,
            next_mcu_preloaded: false,
            crop: None,
            crop_skip_done: false,
            wave_state: Some(wave_state),
            wave_buf,
            wave_first_row: 0,
            wave_row_count: 0,
            wave_next_seg: 0,
            #[cfg(feature = "parallel")]
            wave_y: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cb: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_cr: Vec::new(),
            #[cfg(feature = "parallel")]
            wave_planar_first_luma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_luma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_first_chroma_row: 0,
            #[cfg(feature = "parallel")]
            wave_planar_chroma_rows: 0,
            #[cfg(feature = "parallel")]
            wave_planar_next_seg: 0,
            pool_guard: None,
            deblock_mode: DeblockMode::Off,
            deblock_strength: None,
            deblock_prev_rows: Vec::new(),
            deblock_has_prev: false,
        }
    }

    /// Attach wave-parallel state to a sequential reader.
    ///
    /// This enables `read_rows_ycbcr_native_i16` to use wave-parallel decode while
    /// `read_rows_rgb8` and other interleaved outputs continue using the
    /// sequential streaming path.
    #[cfg(feature = "parallel")]
    pub(super) fn attach_wave_state(
        &mut self,
        wave_state: super::fused_parallel::WaveParallelState,
    ) {
        self.wave_state = Some(wave_state);
    }

    /// Returns the output width (crop width if set, otherwise image width).
    #[inline]
    pub fn width(&self) -> u32 {
        self.crop.map_or(self.width, |c| c.width)
    }

    /// Returns the output height (crop height if set, otherwise image height).
    #[inline]
    pub fn height(&self) -> u32 {
        self.crop.map_or(self.height, |c| c.height)
    }

    /// Returns image info (reflects crop dimensions if set).
    pub fn info(&self) -> ScanlineInfo {
        ScanlineInfo {
            dimensions: Dimensions {
                width: self.width(),
                height: self.height(),
            },
            color_space: self.color_model.color_space(),
            is_xyb: self.color_model == ColorModel::Xyb,
            subsampling: self.strip.subsampling,
        }
    }

    /// Returns the chroma subsampling mode.
    #[inline]
    pub fn subsampling(&self) -> Subsampling {
        self.strip.subsampling
    }

    /// Returns the current row position within the crop (0 to crop_height-1).
    #[inline]
    pub fn current_row(&self) -> usize {
        self.current_row
    }

    /// Returns true if all rows have been read.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.current_row >= self.height() as usize
    }

    /// Returns true if this is a grayscale (single-component) image.
    #[inline]
    pub fn is_grayscale(&self) -> bool {
        self.num_components == 1
    }

    /// Returns the number of components (1 for grayscale, 3 for color).
    #[inline]
    pub fn num_components(&self) -> u8 {
        self.num_components
    }

    /// Returns the chroma plane width at native resolution (before upsampling).
    ///
    /// For 4:2:0 or 4:2:2: MCU-padded chroma width. For 4:4:4: same as luma. For grayscale: 0.
    #[inline]
    pub fn chroma_width(&self) -> u32 {
        if self.num_components == 1 {
            return 0;
        }
        // Wave-only readers have a dummy strip; use wave_state dimensions instead
        #[cfg(feature = "parallel")]
        if let Some(ref ws) = self.wave_state {
            return ws.chroma_width() as u32;
        }
        self.strip.chroma_strip_width as u32
    }

    /// Returns the chroma plane height at native resolution (before upsampling).
    ///
    /// For 4:2:0 or 4:4:0: `(height + 1) / 2`. For 4:4:4 or 4:2:2: `height`. For grayscale: 0.
    #[inline]
    pub fn chroma_height(&self) -> u32 {
        if self.num_components == 1 {
            return 0;
        }
        // Reports a single chroma height — only valid when Cb and Cr share
        // sampling factors. Asymmetric files are routed through the buffered
        // f32 path in `Decoder::scanline_reader` before reaching this point.
        debug_assert!(
            self.strip.v_samp[1] == self.strip.v_samp[2]
                && self.strip.h_samp[1] == self.strip.h_samp[2],
            "chroma_height requires symmetric Cb/Cr; got Cb=({},{}) Cr=({},{})",
            self.strip.h_samp[1],
            self.strip.v_samp[1],
            self.strip.h_samp[2],
            self.strip.v_samp[2]
        );
        // Wave-only readers have a dummy strip; derive from wave_state
        #[cfg(feature = "parallel")]
        if let Some(ref ws) = self.wave_state {
            let v_scale = ws.max_v_samp;
            let c_v = ws.comp_v_samps.get(1).copied().unwrap_or(1).max(1);
            return if c_v < v_scale {
                ((self.height as usize + v_scale - 1) / v_scale) as u32
            } else {
                self.height
            };
        }
        let max_v = self.strip.v_samp[0]
            .max(self.strip.v_samp[1])
            .max(self.strip.v_samp[2]);
        let c_v = self.strip.v_samp[1];
        if c_v < max_v {
            // Vertically subsampled
            (self.height + 1) / 2
        } else {
            self.height
        }
    }

    /// Returns the number of luma rows per MCU row.
    ///
    /// 16 for 4:2:0/4:4:0, 8 for 4:4:4/4:2:2/grayscale.
    #[inline]
    pub fn luma_rows_per_mcu(&self) -> usize {
        #[cfg(feature = "parallel")]
        if let Some(ref ws) = self.wave_state {
            return ws.mcu_pixel_height;
        }
        self.strip.mcu_height
    }

    /// Returns `true` when the reader has a fully-materialized RGB buffer
    /// backing it (progressive JPEGs, coefficient-deblock fallbacks, exotic
    /// chroma sampling). Callers can use this to skip per-MCU-strip loops and
    /// request the whole image in a single `read_rows_*` call.
    pub fn is_buffered(&self) -> bool {
        self.buffered_rgb.is_some()
    }

    /// Returns the number of chroma rows per MCU row at native resolution.
    ///
    /// Always 8 for color images. 0 for grayscale.
    #[inline]
    pub fn chroma_rows_per_mcu(&self) -> usize {
        if self.num_components == 1 {
            return 0;
        }
        #[cfg(feature = "parallel")]
        if let Some(ref ws) = self.wave_state {
            let c_v = ws.comp_v_samps.get(1).copied().unwrap_or(1).max(1);
            return c_v * 8;
        }
        self.strip.chroma_strip_height
    }

    /// Extract gain map JPEG bytes from an UltraHDR image, if present.
    ///
    /// Returns the raw gain map JPEG as a zero-copy slice into the original
    /// JPEG data. Returns `None` for regular JPEGs without UltraHDR metadata.
    ///
    /// This allows streaming decoders to detect and access gain maps without
    /// requiring the full `ultrahdr_reader()` API.
    #[cfg(feature = "ultrahdr")]
    pub fn gain_map_jpeg(&self) -> Option<&[u8]> {
        // Parse the JPEG header markers to find gain map
        // We need a temporary parser just for marker scanning
        let mut parser = match super::parser::JpegParser::new(&self.data, u64::MAX, None) {
            Ok(p) => p,
            Err(_) => return None,
        };
        if parser.read_header().is_err() {
            return None;
        }
        let (range, _metadata) = parser.extract_gainmap_early(&self.data).ok()?;
        let (start, end) = range?;
        Some(&self.data[start..end])
    }

    /// Entropy-decode MCU rows before the crop region without IDCT.
    ///
    /// Runs the Huffman decoder for all MCU rows before `crop.mcu_row_start`,
    /// maintaining the DC predictor chain, but skips IDCT and upsampling.
    fn skip_to_crop_start(&mut self) -> Result<()> {
        let crop = match self.crop {
            Some(c) => c,
            None => return Ok(()),
        };

        if self.crop_skip_done || self.current_mcu_row >= crop.mcu_row_start {
            self.crop_skip_done = true;
            return Ok(());
        }

        // For v-subsampled images (4:2:0, 4:4:0), the first crop MCU row needs
        // chroma context from the previous MCU row's last chroma line. Without it,
        // fixup_vertical_boundary() can't correct row 0's bilinear interpolation.
        let needs_prev_context = self.strip.needs_vertical_upsample() && crop.mcu_row_start > 0;

        // For buffered mode, just advance current_row — no entropy to skip
        if self.buffered_rgb.is_some() {
            self.crop_skip_done = true;
            return Ok(());
        }

        // For coefficient mode: coefficients are already stored, just advance.
        // But for v-subsampled, IDCT+upsample the previous MCU row for chroma context.
        if self.stored_coeffs.is_some() {
            if needs_prev_context {
                self.current_mcu_row = crop.mcu_row_start - 1;
                self.mcu_row_decoded = false;
                self.decode_mcu_row()?;
            }
            self.current_mcu_row = crop.mcu_row_start;
            self.mcu_row_decoded = false;
            self.crop_skip_done = true;
            return Ok(());
        }

        // Streaming mode: entropy-decode-only (no IDCT) until near the crop.
        // For v-subsampled, stop one row early so we can IDCT+upsample it for context.
        let skip_target = if needs_prev_context {
            crop.mcu_row_start - 1
        } else {
            crop.mcu_row_start
        };

        let scan_data = &self.data[self.scan_data_start..];
        let mut decoder = EntropyDecoder::new(scan_data);

        // Set up Huffman tables
        for comp_idx in 0..self.num_components as usize {
            let (dc_idx, ac_idx) = self.table_mapping[comp_idx];
            if let Some(ref table) = self.dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, table);
            }
            if let Some(ref table) = self.ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, table);
            }
        }

        // Restore state if we have one
        if let Some(ref state) = self.decoder_state {
            decoder.restore_state(*state);
        }

        let mcu_cols = self.strip.mcu_cols();

        while self.current_mcu_row < skip_target {
            // Decode one MCU row (entropy only, no IDCT)
            for _mcu_x in 0..mcu_cols {
                // Check for restart marker
                if self.restart_interval > 0
                    && self.mcu_count > 0
                    && self.mcu_count % self.restart_interval as u32 == 0
                {
                    decoder.align_to_byte();
                    // A cut at the boundary: the blocks below decode `Truncated`.
                    let _ = decoder.read_restart_marker_tolerant(self.next_restart_num)?;
                    self.next_restart_num = (self.next_restart_num + 1) & 7;
                    decoder.reset_dc();
                    self.prev_coeff_counts = [64; 4];
                }

                for comp_idx in 0..self.num_components as usize {
                    let h_blocks = self.strip.h_samp[comp_idx] as usize;
                    let v_blocks = self.strip.v_samp[comp_idx] as usize;
                    let (dc_idx, ac_idx) = self.table_mapping[comp_idx];

                    for _v in 0..v_blocks {
                        for _h in 0..h_blocks {
                            // Decode block but skip IDCT — just maintain DC chain
                            match decoder.decode_block_into(
                                &mut self.coeffs_buf,
                                self.prev_coeff_counts[comp_idx],
                                comp_idx,
                                dc_idx,
                                ac_idx,
                            )? {
                                ScanRead::Value(c) => {
                                    self.prev_coeff_counts[comp_idx] =
                                        self.prev_coeff_counts[comp_idx].max(c);
                                }
                                ScanRead::EndOfScan | ScanRead::Truncated => {
                                    self.prev_coeff_counts[comp_idx] = 64;
                                }
                            }
                        }
                    }
                }

                self.mcu_count += 1;
            }

            self.current_mcu_row += 1;
        }

        self.decoder_state = Some(decoder.save_state());
        self.mcu_row_decoded = false;

        // For v-subsampled, decode the last pre-crop MCU row with full IDCT + upsample.
        // This populates prev_cb_row/prev_cr_row so fixup_vertical_boundary() can
        // correct row 0 of the first crop MCU row.
        if needs_prev_context {
            self.decode_mcu_row()?;
            self.current_mcu_row += 1;
            self.mcu_row_decoded = false;
        }

        self.crop_skip_done = true;

        Ok(())
    }

    /// Returns true if the given MCU row index is within the crop region.
    #[inline]
    fn mcu_row_in_crop(&self, mcu_row: usize) -> bool {
        match self.crop {
            Some(c) => {
                // Include one extra MCU row before crop start so the chroma
                // boundary fixup (h2v2 row0) has correct data from the
                // previous MCU row's last chroma line.
                let effective_start = c.mcu_row_start.saturating_sub(1);
                mcu_row >= effective_start && mcu_row < c.mcu_row_end
            }
            None => true,
        }
    }

    /// Decodes the current MCU row into strip buffers.
    /// Resolve per-component quantization tables.
    ///
    /// Returns owned copies in an array of 4 (Y, Cb, Cr, and a spare copy of Y).
    /// Components beyond `num_components` fall back to the Y table.
    /// Copies are returned to avoid holding a `&self` borrow that would conflict
    /// with mutable access to `self.strip` and `self.coeffs_buf` in the MCU loop.
    fn resolve_quant_tables(&self) -> Result<[[u16; 64]; 4]> {
        let quant_y = self.quant_tables[self.quant_indices[0]]
            .ok_or_else(|| Error::internal("missing Y quantization table"))?;
        let quant_cb = if self.num_components > 1 {
            self.quant_tables[self.quant_indices[1]]
                .ok_or_else(|| Error::internal("missing Cb quantization table"))?
        } else {
            quant_y
        };
        let quant_cr = if self.num_components > 2 {
            self.quant_tables[self.quant_indices[2]]
                .ok_or_else(|| Error::internal("missing Cr quantization table"))?
        } else {
            quant_y
        };
        Ok([quant_y, quant_cb, quant_cr, quant_y])
    }

    fn decode_mcu_row(&mut self) -> Result<()> {
        if self.mcu_row_decoded {
            return Ok(());
        }

        // Dispatch to coefficient-based path if we have stored coefficients
        if self.stored_coeffs.is_some() {
            return self.decode_mcu_row_from_coefficients();
        }

        // Always create decoder from the full scan data slice
        let scan_data = &self.data[self.scan_data_start..];
        let mut decoder = EntropyDecoder::new(scan_data);

        // Set up Huffman tables first (before restoring state)
        for comp_idx in 0..self.num_components as usize {
            let (dc_idx, ac_idx) = self.table_mapping[comp_idx];

            if let Some(ref table) = self.dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, table);
            }
            if let Some(ref table) = self.ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, table);
            }
        }

        // Restore full decoder state if we have one (includes bit buffer position)
        if let Some(ref state) = self.decoder_state {
            decoder.restore_state(*state);
        }

        let quant_refs = self.resolve_quant_tables()?;

        let mcu_cols = self.strip.mcu_cols();

        // Decode one MCU row
        for mcu_x in 0..mcu_cols {
            // Check for restart marker
            if self.restart_interval > 0
                && self.mcu_count > 0
                && self.mcu_count % self.restart_interval as u32 == 0
            {
                decoder.align_to_byte();
                // A cut at the boundary: the blocks below decode `Truncated`
                // and are emitted as zero blocks.
                let _ = decoder.read_restart_marker_tolerant(self.next_restart_num)?;
                self.next_restart_num = (self.next_restart_num + 1) & 7;
                decoder.reset_dc();
                self.prev_coeff_counts = [64; 4];
            }

            // Decode each component's blocks
            let do_idct = self.mcu_row_in_crop(self.current_mcu_row);

            for comp_idx in 0..self.num_components as usize {
                let h_blocks = self.strip.h_samp[comp_idx] as usize;
                let v_blocks = self.strip.v_samp[comp_idx] as usize;

                let (dc_idx, ac_idx) = self.table_mapping[comp_idx];
                let quant = &quant_refs[comp_idx];

                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let coeff_count = match decoder.decode_block_into(
                            &mut self.coeffs_buf,
                            self.prev_coeff_counts[comp_idx],
                            comp_idx,
                            dc_idx,
                            ac_idx,
                        )? {
                            ScanRead::Value(c) => c,
                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                // Emit the documented zero block. `continue`
                                // skipped the IDCT and served whatever the
                                // previous MCU row had left in the strip (#92).
                                self.prev_coeff_counts[comp_idx] = 64;
                                self.coeffs_buf = [0i16; 64];
                                1
                            }
                        };
                        self.prev_coeff_counts[comp_idx] =
                            self.prev_coeff_counts[comp_idx].max(coeff_count);

                        if do_idct {
                            self.strip.idct_block(
                                comp_idx,
                                mcu_x,
                                h,
                                v,
                                &self.coeffs_buf,
                                coeff_count,
                                quant,
                            );
                        }
                    }
                }
            }

            self.mcu_count += 1;
        }

        // Save full state for next MCU row (includes bit buffer position)
        self.decoder_state = Some(decoder.save_state());

        // Upsample chroma only if this MCU row is in the crop
        if self.mcu_row_in_crop(self.current_mcu_row) {
            // Edge-replicate the last real chroma row/column over MCU padding so
            // the upsampler doesn't interpolate with IDCT-rounded padding data.
            self.strip.truncate_chroma_padding(
                self.width as usize,
                self.height as usize,
                self.current_mcu_row,
            );

            // Apply boundary 4-tap deblock BEFORE upsampling so each component
            // is filtered at native resolution (matching the coefficient path).
            if self.deblock_mode == DeblockMode::Boundary4Tap {
                self.apply_boundary_deblock();
            }

            self.strip.upsample_chroma();
        }

        self.mcu_row_decoded = true;

        Ok(())
    }

    /// Apply boundary 4-tap deblocking to the current MCU row's strip planes.
    ///
    /// Filters each component plane at native resolution:
    /// - Y: `strip_stride × mcu_height`, boundaries at 8px
    /// - Cb/Cr: `chroma_strip_stride × chroma_strip_height`, boundaries at 8px
    ///
    /// Vertical boundaries (within row) and intra-MCU horizontal boundaries are
    /// filtered using the current strip data. The inter-MCU horizontal boundary
    /// (junction with previous MCU row) uses `deblock_prev_rows`.
    fn apply_boundary_deblock(&mut self) {
        let strengths = match self.deblock_strength {
            Some(s) => s,
            None => return,
        };

        let mcu_row = self.current_mcu_row;
        let is_grayscale = self.num_components == 1;

        // --- Filter Y plane ---
        {
            let w = self.strip.strip_width;
            let stride = self.strip.strip_stride;
            let h = self.strip.mcu_height;
            let strength = strengths[0];

            // Vertical boundaries within this MCU row
            filter_strip_vertical_i16(&mut self.strip.y_strip, w, stride, h, &strength);

            // Intra-MCU horizontal boundaries (at row 8 for 16-row MCUs)
            filter_strip_horizontal_i16(&mut self.strip.y_strip, w, stride, h, &strength);

            // Inter-MCU horizontal boundary with previous MCU row
            if self.deblock_has_prev && mcu_row > 0 {
                filter_inter_mcu_horizontal_i16(
                    &self.deblock_prev_rows,
                    0, // Y prev rows at offset 0
                    stride,
                    &mut self.strip.y_strip,
                    stride,
                    w,
                    &strength,
                );
            }

            // Save last 2 rows of Y for next MCU row's inter-MCU boundary
            if h >= 2 {
                let src_row_m2 = (h - 2) * stride;
                let src_row_m1 = (h - 1) * stride;
                self.deblock_prev_rows[..stride]
                    .copy_from_slice(&self.strip.y_strip[src_row_m2..src_row_m2 + stride]);
                self.deblock_prev_rows[stride..2 * stride]
                    .copy_from_slice(&self.strip.y_strip[src_row_m1..src_row_m1 + stride]);
            }
        }

        // --- Filter Cb/Cr planes ---
        if !is_grayscale {
            let cw = self.strip.chroma_strip_width;
            let cs = self.strip.chroma_strip_stride;
            let ch = self.strip.chroma_strip_height;
            let ys = self.strip.strip_stride;

            for (comp_idx, plane) in [&mut self.strip.cb_strip, &mut self.strip.cr_strip]
                .into_iter()
                .enumerate()
            {
                let strength = strengths[comp_idx + 1];

                // Vertical boundaries
                filter_strip_vertical_i16(plane, cw, cs, ch, &strength);

                // Intra-MCU horizontal boundaries
                filter_strip_horizontal_i16(plane, cw, cs, ch, &strength);

                // Inter-MCU horizontal boundary
                // Cb at offset 2*ys, Cr at offset 2*ys + 2*cs
                if self.deblock_has_prev && mcu_row > 0 {
                    let prev_offset = 2 * ys + comp_idx * 2 * cs;
                    filter_inter_mcu_horizontal_i16(
                        &self.deblock_prev_rows,
                        prev_offset,
                        cs,
                        plane,
                        cs,
                        cw,
                        &strength,
                    );
                }

                // Save last 2 chroma rows
                if ch >= 2 {
                    let src_row_m2 = (ch - 2) * cs;
                    let src_row_m1 = (ch - 1) * cs;
                    let dst_offset = 2 * ys + comp_idx * 2 * cs;
                    self.deblock_prev_rows[dst_offset..dst_offset + cs]
                        .copy_from_slice(&plane[src_row_m2..src_row_m2 + cs]);
                    self.deblock_prev_rows[dst_offset + cs..dst_offset + 2 * cs]
                        .copy_from_slice(&plane[src_row_m1..src_row_m1 + cs]);
                }
            }
        }

        self.deblock_has_prev = true;
    }

    /// Decode an MCU row from pre-stored coefficients (no entropy decoding).
    ///
    /// Reads blocks from `self.stored_coeffs`, applies IDCT via the strip
    /// processor, then upsamples chroma. Used for decode-time transforms.
    fn decode_mcu_row_from_coefficients(&mut self) -> Result<()> {
        // Skip IDCT entirely if this MCU row is outside the crop
        if !self.mcu_row_in_crop(self.current_mcu_row) {
            self.mcu_row_decoded = true;
            return Ok(());
        }

        let quant_refs = self.resolve_quant_tables()?;

        let mcu_cols = self.strip.mcu_cols();
        let coeffs = self.stored_coeffs.as_ref().unwrap();

        for mcu_x in 0..mcu_cols {
            for comp_idx in 0..self.num_components as usize {
                let h_blocks = self.strip.h_samp[comp_idx] as usize;
                let v_blocks = self.strip.v_samp[comp_idx] as usize;
                let quant = &quant_refs[comp_idx];
                let blocks_wide = coeffs.components[comp_idx].blocks_wide;

                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let by = self.current_mcu_row * v_blocks + v;
                        let bx = mcu_x * h_blocks + h;
                        let block_idx = by * blocks_wide + bx;

                        let block = coeffs.components[comp_idx].block(block_idx);
                        self.coeffs_buf.copy_from_slice(block);

                        // Count non-zero coefficients for tiered IDCT
                        let coeff_count = if block.iter().all(|&c| c == 0) {
                            0
                        } else {
                            // Find the highest non-zero position + 1
                            block
                                .iter()
                                .rposition(|&c| c != 0)
                                .map(|p| (p + 1) as u8)
                                .unwrap_or(0)
                        };

                        self.strip.idct_block(
                            comp_idx,
                            mcu_x,
                            h,
                            v,
                            &self.coeffs_buf,
                            coeff_count,
                            quant,
                        );
                    }
                }
            }
        }

        // Peek ahead: IDCT first chroma block row of next MCU for bottom context
        if self.strip.needs_vertical_upsample() && !self.is_last_mcu_row() {
            self.peek_next_chroma_row(&quant_refs);
        }

        // Edge-replicate the last real chroma row/column over MCU padding so
        // the upsampler doesn't interpolate with IDCT-rounded padding data.
        // Vertical: only affects the last MCU row. Horizontal: affects any
        // MCU row where image width is not MCU-aligned. Both are handled
        // inside truncate_chroma_padding with early-return when no-op.
        self.strip.truncate_chroma_padding(
            self.width as usize,
            self.height as usize,
            self.current_mcu_row,
        );

        // Apply boundary 4-tap deblock BEFORE upsampling so each component
        // is filtered at native resolution (matching the streaming path).
        if self.deblock_mode == DeblockMode::Boundary4Tap {
            self.apply_boundary_deblock();
        }

        // Upsample chroma if needed
        self.strip.upsample_chroma();
        self.mcu_row_decoded = true;

        Ok(())
    }

    /// IDCT the first chroma block row of the next MCU row from stored coefficients.
    ///
    /// Writes the results into `strip.next_cb_row` / `strip.next_cr_row` for
    /// bottom boundary fixup during upsampling.
    fn peek_next_chroma_row(&mut self, quant_refs: &[[u16; 64]; 4]) {
        let mcu_cols = self.strip.mcu_cols();
        let coeffs = self.stored_coeffs.as_ref().unwrap();
        let next_mcu_row = self.current_mcu_row + 1;

        for comp_idx in 1..self.num_components as usize {
            let v_blocks = self.strip.v_samp[comp_idx] as usize;
            let h_blocks = self.strip.h_samp[comp_idx] as usize;
            let blocks_wide = coeffs.components[comp_idx].blocks_wide;
            let by = next_mcu_row * v_blocks; // first block row of next MCU
            let quant = &quant_refs[comp_idx];

            let next_row = if comp_idx == 1 {
                &mut self.strip.next_cb_row
            } else {
                &mut self.strip.next_cr_row
            };

            for mcu_x in 0..mcu_cols {
                for h in 0..h_blocks {
                    let bx = mcu_x * h_blocks + h;
                    let block_idx = by * blocks_wide + bx;
                    let block = coeffs.components[comp_idx].block(block_idx);

                    let coeff_count = if block.iter().all(|&c| c == 0) {
                        0
                    } else {
                        block
                            .iter()
                            .rposition(|&c| c != 0)
                            .map(|p| (p + 1) as u8)
                            .unwrap_or(0)
                    };

                    let x_offset = mcu_x * h_blocks * 8 + h * 8;

                    {
                        // Always route through the real IDCT — it has its own
                        // DC-only fast path (`idct_int_dc_only`), so a
                        // hand-rolled shortcut here buys nothing and can only
                        // drift from the strip's own reconstruction. It did:
                        // this used to special-case `coeff_count <= 1` as
                        // `((dc * quant[0]) + 1024) >> 11`, but a DC-only IDCT
                        // is `(dc + 4 + 1024) >> 3` — wrong shift (11 vs 3) and
                        // no rounding term, so neutral chroma (dequantized
                        // DC ~ 0) peeked as 0 instead of 128. Smooth chroma
                        // quantizes to DC-only constantly, so every real photo
                        // hit it (waterhouse.jpg: 76/255 on the last row of
                        // every MCU row) while high-frequency synthetic test
                        // images never did.
                        let mut temp_coeffs = [0i16; 64];
                        temp_coeffs.copy_from_slice(block);
                        let mut dequant_buf = [0i32; 64];
                        crate::quant::dequantize_unzigzag_i32_into_partial(
                            &temp_coeffs,
                            quant,
                            &mut dequant_buf,
                            coeff_count,
                        );
                        let mut temp_pixels = [0i16; 64];
                        if self.precision == 12 {
                            super::idct_int::idct_libjpeg_precision::<12>(
                                &dequant_buf,
                                &mut temp_pixels,
                                8,
                            );
                        } else {
                            match self.strip.idct_method {
                                super::IdctMethod::Libjpeg => {
                                    super::idct_int::idct_int_tiered_libjpeg(
                                        &mut dequant_buf,
                                        &mut temp_pixels,
                                        8,
                                        coeff_count,
                                    );
                                }
                                super::IdctMethod::Jpegli => {
                                    super::idct_int::idct_int_tiered(
                                        &mut dequant_buf,
                                        &mut temp_pixels,
                                        8,
                                        coeff_count,
                                    );
                                }
                            }
                        }
                        for px in 0..8 {
                            if x_offset + px < next_row.len() {
                                next_row[x_offset + px] = temp_pixels[px];
                            }
                        }
                    }
                }
            }
        }

        self.strip.truncate_next_row_padding(self.width as usize);
        self.strip.has_next_context = true;
    }

    /// Returns true if this is the last MCU row of the image.
    fn is_last_mcu_row(&self) -> bool {
        let mcu_height = self.strip.mcu_height;
        let total_mcu_rows = (self.height as usize + mcu_height - 1) / mcu_height;
        self.current_mcu_row + 1 >= total_mcu_rows
    }

    /// Initialize crop state: skip MCU rows before the crop, set row_in_mcu.
    ///
    /// Must be called before the first `ensure_row_ready()`.
    fn ensure_crop_initialized(&mut self) -> Result<()> {
        if self.crop.is_some() && !self.crop_skip_done {
            self.skip_to_crop_start()?;
            // Set row_in_mcu to the first crop row within the first crop MCU row
            let crop = self.crop.unwrap();
            self.row_in_mcu = crop.y as usize - crop.mcu_row_start * self.strip.mcu_height;
        }
        Ok(())
    }

    /// Ensure the current row is ready to read.
    ///
    /// Decodes the current MCU row if needed, then for the streaming path
    /// (non-coefficient), pre-decodes the next MCU row when about to read
    /// the last row of the current MCU, providing correct bottom chroma context.
    fn ensure_row_ready(&mut self) -> Result<()> {
        self.decode_mcu_row()?;

        // Streaming bottom fixup: when about to read the last row, decode the
        // next MCU row first to get correct bottom chroma interpolation.
        if self.row_in_mcu == self.strip.mcu_height - 1
            && self.strip.needs_vertical_upsample()
            && !self.strip.has_deferred_bottom
            && !self.is_last_mcu_row()
            && self.stored_coeffs.is_none()
        {
            self.prepare_streaming_bottom()?;
        }

        Ok(())
    }

    /// Pre-decode the next MCU row for streaming bottom-boundary fixup.
    ///
    /// 1. Save current last Y row into deferred buffer
    /// 2. Decode next MCU row (entropy + IDCT)
    /// 3. Compute corrected bottom chroma using prev_cb_row + cb_strip[0]
    /// 4. Upsample the next MCU (so it's ready when we advance)
    /// 5. Mark next MCU as preloaded
    fn prepare_streaming_bottom(&mut self) -> Result<()> {
        // 1. Save the Y row for the last output row of the current MCU
        self.strip.save_deferred_y_row();

        // 2. Compute corrected bottom chroma BEFORE the next decode overwrites prev_cb_row.
        // At this point:
        //   prev_cb_row = last chroma row of current MCU (from save_last_chroma_row)
        //   We need to decode the next MCU to get cb_strip[0] as the neighbor.

        // Decode next MCU row's entropy data + IDCT into strip buffers
        self.current_mcu_row += 1;
        self.mcu_row_decoded = false;

        // Save/restore state so we can call decode_mcu_row_streaming
        self.decode_mcu_row_streaming()?;

        // 3. Edge-replicate the next MCU row's padding first: the corrected
        //    bottom row below interpolates against its first chroma row, and
        //    that row's horizontal padding must not reach the last column.
        self.strip.truncate_chroma_padding(
            self.width as usize,
            self.height as usize,
            self.current_mcu_row,
        );

        // Apply boundary 4-tap deblock on the preloaded next MCU row BEFORE
        // computing the deferred bottom chroma or upsampling the next MCU row,
        // because advance_mcu_row will mark this preloaded row as already decoded.
        if self.deblock_mode == DeblockMode::Boundary4Tap {
            self.apply_boundary_deblock();
        }

        // 4. Now cb_strip[0] has the first chroma row of the next MCU.
        //    prev_cb_row still has the last chroma row of the previous MCU.
        //    Compute the corrected bottom chroma.
        self.strip.compute_deferred_bottom();

        // 5. Upsample the next MCU row (top boundary fixup uses prev_cb_row)
        self.strip.upsample_chroma();

        // 6. Mark state
        self.current_mcu_row -= 1; // Restore to current MCU row
        self.next_mcu_preloaded = true;
        self.mcu_row_decoded = true; // Current MCU is still decoded

        Ok(())
    }

    /// Decode entropy data + IDCT for the current MCU row (streaming only).
    ///
    /// This is the entropy decode + IDCT portion of `decode_mcu_row` without
    /// the upsample step. Used by `prepare_streaming_bottom` to decode the
    /// next MCU row separately from upsampling.
    fn decode_mcu_row_streaming(&mut self) -> Result<()> {
        let scan_data = &self.data[self.scan_data_start..];
        let mut decoder = EntropyDecoder::new(scan_data);

        for comp_idx in 0..self.num_components as usize {
            let (dc_idx, ac_idx) = self.table_mapping[comp_idx];
            if let Some(ref table) = self.dc_tables[dc_idx] {
                decoder.set_dc_table(dc_idx, table);
            }
            if let Some(ref table) = self.ac_tables[ac_idx] {
                decoder.set_ac_table(ac_idx, table);
            }
        }

        if let Some(ref state) = self.decoder_state {
            decoder.restore_state(*state);
        }

        let quant_refs = self.resolve_quant_tables()?;
        let mcu_cols = self.strip.mcu_cols();

        for mcu_x in 0..mcu_cols {
            if self.restart_interval > 0
                && self.mcu_count > 0
                && self.mcu_count % self.restart_interval as u32 == 0
            {
                decoder.align_to_byte();
                // A cut at the boundary: the blocks below decode `Truncated`
                // and are emitted as zero blocks.
                let _ = decoder.read_restart_marker_tolerant(self.next_restart_num)?;
                self.next_restart_num = (self.next_restart_num + 1) & 7;
                decoder.reset_dc();
                self.prev_coeff_counts = [64; 4];
            }

            for comp_idx in 0..self.num_components as usize {
                let h_blocks = self.strip.h_samp[comp_idx] as usize;
                let v_blocks = self.strip.v_samp[comp_idx] as usize;
                let (dc_idx, ac_idx) = self.table_mapping[comp_idx];
                let quant = &quant_refs[comp_idx];

                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let coeff_count = match decoder.decode_block_into(
                            &mut self.coeffs_buf,
                            self.prev_coeff_counts[comp_idx],
                            comp_idx,
                            dc_idx,
                            ac_idx,
                        )? {
                            ScanRead::Value(c) => c,
                            ScanRead::EndOfScan | ScanRead::Truncated => {
                                // Emit the documented zero block. `continue`
                                // skipped the IDCT and served whatever the
                                // previous MCU row had left in the strip (#92).
                                self.prev_coeff_counts[comp_idx] = 64;
                                self.coeffs_buf = [0i16; 64];
                                1
                            }
                        };
                        self.prev_coeff_counts[comp_idx] =
                            self.prev_coeff_counts[comp_idx].max(coeff_count);

                        self.strip.idct_block(
                            comp_idx,
                            mcu_x,
                            h,
                            v,
                            &self.coeffs_buf,
                            coeff_count,
                            quant,
                        );
                    }
                }
            }

            self.mcu_count += 1;
        }

        self.decoder_state = Some(decoder.save_state());
        Ok(())
    }

    /// Advances to the next MCU row.
    fn advance_mcu_row(&mut self) {
        self.current_mcu_row += 1;
        self.row_in_mcu = 0;
        self.strip.has_deferred_bottom = false;
        if self.next_mcu_preloaded {
            self.mcu_row_decoded = true;
            self.next_mcu_preloaded = false;
        } else {
            self.mcu_row_decoded = false;
        }
    }

    /// Shared geometry + bounds check for the buffered (progressive) serve
    /// path. Every interleaved-format `read_rows_*` method that serves from
    /// `buffered_rgb` computed the same source-row stride and validated the
    /// same buffer-size bound before its format-specific copy loop; this is
    /// the single copy. Returns `(bpp, src_row_bytes, crop_y)`.
    ///
    /// `#[inline]` and progressive-only, so it lowers to the same code the
    /// open-coded prologue did and never touches the streaming hot path.
    #[inline]
    fn buffered_geometry(
        &self,
        buffer_len: usize,
        out_height: usize,
    ) -> Result<(usize, usize, usize)> {
        let bpp = if self.num_components == 1 { 1 } else { 3 };
        let src_row_bytes = self.width as usize * bpp;
        let crop_y = self.crop.map_or(0, |c| c.y as usize);
        if (crop_y + out_height) * src_row_bytes > buffer_len {
            return Err(Error::internal(
                "buffered RGB data too small for image dimensions",
            ));
        }
        Ok((bpp, src_row_bytes, crop_y))
    }

    /// Drive one `read_rows_*` call: for each of up to `max_rows` output
    /// rows, hand `emit(i, row)` that row's source, already cropped to the
    /// output columns. Every read method is this loop plus one conversion,
    /// so the reader modes (streaming, coefficient, buffered, wave) serve
    /// every method the same way.
    fn read_rows_with(
        &mut self,
        max_rows: usize,
        emit: impl FnMut(usize, RowSrc<'_>),
    ) -> Result<usize> {
        self.read_component_rows_with(max_rows, false, emit)
    }

    fn read_component_rows_with(
        &mut self,
        max_rows: usize,
        raw: bool,
        mut emit: impl FnMut(usize, RowSrc<'_>),
    ) -> Result<usize> {
        if !raw && self.color_model == ColorModel::Unknown {
            return Err(Error::unsupported_feature(
                "unknown component color space: use read_rows_components16",
            ));
        }
        let out_width = self.width() as usize;
        let out_height = self.height() as usize;
        let crop_x = self.crop.map_or(0, |c| c.x as usize);
        let crop_y = self.crop.map_or(0, |c| c.y as usize);
        let cols = crop_x..crop_x + out_width;
        let mut n = 0;

        if let Some(samples) = &self.buffered_native {
            let channels = self.num_components as usize;
            while n < max_rows && self.current_row < out_height {
                let start = ((crop_y + self.current_row) * self.width as usize + crop_x) * channels;
                emit(
                    n,
                    RowSrc::Native(
                        &samples[start..start + out_width * channels],
                        channels,
                        self.color_model,
                    ),
                );
                n += 1;
                self.current_row += 1;
            }
            return Ok(n);
        }

        if let Some(source) = &mut self.component_rows {
            let channels = self.num_components as usize;
            while n < max_rows && self.current_row < out_height {
                while source.current_row <= crop_y + self.current_row {
                    source.next_row(&self.data)?;
                }
                emit(
                    n,
                    RowSrc::Native(
                        &source.row[crop_x * channels..(crop_x + out_width) * channels],
                        channels,
                        self.color_model,
                    ),
                );
                n += 1;
                self.current_row += 1;
            }
            return Ok(n);
        }

        if let Some(predictive) = &mut self.predictive {
            let channels = self.num_components as usize;
            while n < max_rows && self.current_row < out_height {
                let source_y = crop_y + self.current_row;
                while predictive.current_row() <= source_y {
                    predictive.next_row(&self.data)?;
                }
                emit(
                    n,
                    RowSrc::Native(
                        &predictive.row[crop_x * channels..(crop_x + out_width) * channels],
                        channels,
                        self.color_model,
                    ),
                );
                n += 1;
                self.current_row += 1;
            }
            return Ok(n);
        }

        // Wave-parallel reader: decoded RGB rows, one wave of restart
        // segments at a time.
        #[cfg(feature = "parallel")]
        if self.is_wave_only() {
            let row_bytes = self.width as usize * 3;
            while n < max_rows && self.current_row < out_height {
                let image_row = crop_y + self.current_row;
                while image_row >= self.wave_first_row + self.wave_row_count {
                    let next = self.wave_next_seg;
                    self.decode_next_wave()?;
                    if self.wave_next_seg == next {
                        return Err(Error::internal("wave decode ended before the last row"));
                    }
                }
                let off = (image_row - self.wave_first_row) * row_bytes + crop_x * 3;
                emit(n, RowSrc::Rgb8(&self.wave_buf[off..off + out_width * 3]));
                n += 1;
                self.current_row += 1;
            }
            return Ok(n);
        }

        // Buffered reader: the whole image, decoded up front.
        if let Some(buffer) = &self.buffered_rgb {
            let (bpp, row_bytes, _) = self.buffered_geometry(buffer.len(), out_height)?;
            while n < max_rows && self.current_row < out_height {
                let off = (crop_y + self.current_row) * row_bytes + crop_x * bpp;
                let src = &buffer[off..off + out_width * bpp];
                emit(
                    n,
                    if bpp == 1 {
                        RowSrc::Gray8(src)
                    } else {
                        RowSrc::Rgb8(src)
                    },
                );
                n += 1;
                self.current_row += 1;
            }
            return Ok(n);
        }

        // Streaming and coefficient readers: one strip (MCU row) of planes.
        self.ensure_crop_initialized()?;
        let strip_cols = (self.width as usize).min(self.strip.strip_width);
        while n < max_rows && self.current_row < out_height {
            self.ensure_row_ready()?;
            let row = if self.num_components == 1 {
                RowSrc::Planes(
                    &self.strip.y_row(self.row_in_mcu, strip_cols)[cols.clone()],
                    &[],
                    &[],
                )
            } else {
                let (y, cb, cr) = self.strip.row_planes(self.row_in_mcu, strip_cols);
                RowSrc::Planes(&y[cols.clone()], &cb[cols.clone()], &cr[cols.clone()])
            };
            emit(n, row);
            n += 1;
            self.current_row += 1;
            self.row_in_mcu += 1;
            if self.row_in_mcu >= self.strip.mcu_height {
                self.advance_mcu_row();
            }
        }
        Ok(n)
    }

    /// Whether this is a wave-parallel reader serving decoded RGB rows.
    #[inline]
    fn is_wave_only(&self) -> bool {
        #[cfg(feature = "parallel")]
        {
            self.wave_state.is_some() && !self.wave_buf.is_empty()
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }

    /// Reads whose YCbCr meaning is the Y/Cb/Cr planes (gray, YCbCr): the
    /// wave-parallel reader holds only RGB rows, so it cannot serve them for
    /// a YCbCr frame (for an RGB frame they are defined from the RGB values).
    fn check_plane_read(&self, what: &'static str) -> Result<()> {
        if self.is_wave_only() && self.color_model == ColorModel::YCbCr {
            return Err(Error::unsupported_feature(what));
        }
        Ok(())
    }

    /// 8-bit interleaved reads: the frame's row kernel for strip rows, a
    /// relayout of decoded RGB / gray rows otherwise.
    fn read_rows_u8(&mut self, mut output: ImgRefMut<'_, u8>, layout: Layout8) -> Result<usize> {
        let row_bytes = self.width() as usize * layout.bytes_per_pixel();
        if output.width() < row_bytes {
            return Err(Error::invalid_buffer_size(row_bytes, output.width()));
        }
        if layout == Layout8::Gray {
            self.check_plane_read(
                "read_rows_gray8 needs the luma plane, which the wave-parallel reader \
                 does not keep for YCbCr frames (use num_threads(1))",
            )?;
        }
        let planes = self.planes;
        let kernel = RowKernel::new(self.planes, layout);
        let (max_rows, stride) = (output.height(), output.stride());
        // Rows are addressed through the flat buffer, so the last row only
        // needs `row_bytes`, not a full stride.
        let buf = output.buf_mut();
        let precision = self.precision;
        self.read_rows_with(max_rows, |i, row| {
            let out = &mut buf[i * stride..i * stride + row_bytes];
            match row {
                RowSrc::Native(..) => row.write_native_u8(precision, planes, layout, out),
                RowSrc::Planes(..) if precision != 8 => {
                    row.write_native_u8(precision, planes, layout, out)
                }
                RowSrc::Planes(c0, c1, c2) => kernel.row(c0, c1, c2, out),
                RowSrc::Rgb8(rgb) => rgb8_row(rgb, layout, out),
                RowSrc::Gray8(gray) => gray8_row(gray, layout, out),
            }
        })
    }

    /// Read rows into an RGB8 buffer (3 bytes per pixel).
    ///
    /// Returns the number of rows actually written (may be less than requested
    /// if end of image is reached).
    pub fn read_rows_rgb8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_u8(output, Layout8::Rgb)
    }

    /// Read rows into an RGBX8 buffer (RGB with padding byte, X=255).
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_rgbx8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_u8(output, Layout8::Rgbx)
    }

    /// Read rows into a BGR8 buffer (3 bytes per pixel, B-G-R order).
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_bgr8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_u8(output, Layout8::Bgr)
    }

    /// Read rows into an RGBA8 buffer (4 bytes per pixel, R-G-B-A order, A=255).
    ///
    /// Identical layout to `read_rows_rgbx8` — the fourth byte is always 255.
    /// Returns the number of rows actually written.
    #[inline]
    pub fn read_rows_rgba8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_rgbx8(output)
    }

    /// Read rows into a BGRA8 buffer (4 bytes per pixel, B-G-R-A=255).
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_bgra8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_u8(output, Layout8::Bgrx)
    }

    /// Read rows into a BGRX8 buffer (4 bytes per pixel, B-G-R-X=255).
    ///
    /// Identical to `read_rows_bgra8` — the pad byte is always 255.
    /// Returns the number of rows actually written.
    #[inline]
    pub fn read_rows_bgrx8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_bgra8(output)
    }

    /// Read rows into a grayscale u8 buffer.
    ///
    /// Grayscale frames return their samples, YCbCr frames their Y plane, and
    /// RGB frames the BT.601 luma of the decoded R, G, B (libjpeg-turbo's
    /// `rgb_gray_convert`). Readers that hold decoded RGB (sampling shapes the strip does not serve, CMYK, XYB,
    /// orientation and deblock fallbacks) also take
    /// the luma of the RGB.
    ///
    /// On a wave-parallel reader of a YCbCr frame (`parallel`, nearest-neighbour
    /// 4:2:2 / 4:2:0, MCU-row restart markers, >= 1024 MCUs; see
    /// [`Decoder::scanline_reader`](crate::decoder::Decoder::scanline_reader))
    /// this returns [`UnsupportedFeature`](crate::decoder::ErrorKind::UnsupportedFeature):
    /// that reader holds RGB rows, not the planes. Use `num_threads(1)`.
    ///
    /// Returns the number of rows actually written (may be less than requested
    /// if end of image is reached).
    pub fn read_rows_gray8(&mut self, output: ImgRefMut<'_, u8>) -> Result<usize> {
        self.read_rows_u8(output, Layout8::Gray)
    }

    /// Read rows into a linear f32 RGBA buffer.
    ///
    /// Output is in linear light (not sRGB gamma), alpha 1.0. Integer samples
    /// (gray and RGB frames, decoded RGB rows) are linearized through the
    /// exact 8-bit sRGB table; YCbCr frames convert to RGB in f32 first.
    ///
    /// On a wave-parallel reader of a YCbCr frame (`parallel`, nearest-neighbour
    /// 4:2:2 / 4:2:0, MCU-row restart markers, >= 1024 MCUs; see
    /// [`Decoder::scanline_reader`](crate::decoder::Decoder::scanline_reader))
    /// this returns [`UnsupportedFeature`](crate::decoder::ErrorKind::UnsupportedFeature):
    /// that reader holds RGB rows, not the planes. Use `num_threads(1)`.
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_rgba_f32(&mut self, mut output: ImgRefMut<'_, f32>) -> Result<usize> {
        let row_len = self.width() as usize * 4;
        if output.width() < row_len {
            return Err(Error::invalid_buffer_size(row_len, output.width()));
        }
        // YCbCr frames convert their planes in f32; the wave-only reader holds
        // 8-bit RGB rows, so it would answer differently from num_threads(1).
        self.check_plane_read(
            "read_rows_rgba_f32 converts YCbCr planes in f32, which the wave-parallel \
             reader does not keep (use num_threads(1))",
        )?;
        let planes = self.planes;
        let (max_rows, stride) = (output.height(), output.stride());
        let buf = output.buf_mut();
        let precision = self.precision;
        self.read_rows_with(max_rows, |i, row| {
            let out = &mut buf[i * stride..i * stride + row_len];
            let mut put = |px: usize, r: f32, g: f32, b: f32| {
                out[px * 4..px * 4 + 4].copy_from_slice(&[r, g, b, 1.0]);
            };
            match row {
                RowSrc::Native(..) => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for x in 0..row.len() {
                        let [r, g, b] = row
                            .rgb16(x, precision, planes)
                            .map(|v| srgb_to_linear_f32(v as f32 / max));
                        put(x, r, g, b);
                    }
                }
                RowSrc::Planes(..) if precision != 8 => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for x in 0..row.len() {
                        let [r, g, b] = row
                            .rgb16(x, precision, planes)
                            .map(|v| srgb_to_linear_f32(v as f32 / max));
                        put(x, r, g, b);
                    }
                }
                RowSrc::Planes(c0, c1, c2) => match planes {
                    PlaneMatrix::YCbCr { .. } => {
                        for (px, ((&y, &cb), &cr)) in c0.iter().zip(c1).zip(c2).enumerate() {
                            let (r, g, b) = ycbcr_to_rgb_f32(y as f32, cb as f32, cr as f32);
                            put(
                                px,
                                srgb_to_linear_f32(r / 255.0),
                                srgb_to_linear_f32(g / 255.0),
                                srgb_to_linear_f32(b / 255.0),
                            );
                        }
                    }
                    PlaneMatrix::Identity => {
                        for (px, ((&r, &g), &b)) in c0.iter().zip(c1).zip(c2).enumerate() {
                            put(px, lin_i16(r), lin_i16(g), lin_i16(b));
                        }
                    }
                    PlaneMatrix::Gray => {
                        for (px, &v) in c0.iter().enumerate() {
                            let l = lin_i16(v);
                            put(px, l, l, l);
                        }
                    }
                },
                RowSrc::Rgb8(rgb) => {
                    for (px, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
                        put(
                            px,
                            srgb_to_linear(p[0]),
                            srgb_to_linear(p[1]),
                            srgb_to_linear(p[2]),
                        );
                    }
                }
                RowSrc::Gray8(gray) => {
                    for (px, &v) in gray.iter().enumerate() {
                        let l = srgb_to_linear(v);
                        put(px, l, l, l);
                    }
                }
            }
        })
    }

    /// Read rows into separate YCbCr f32 planes (upsampled to full resolution).
    ///
    /// Each plane receives normalized values in range \[0, 1\] for Y, \[-0.5, 0.5\] for Cb/Cr
    /// (with an f32 [`OutputTarget`](crate::decoder::OutputTarget), gray and YCbCr frames' samples
    /// come from the unclamped IDCT and ringing can leave those ranges).
    /// Chroma values are upsampled to full resolution for subsampled images
    /// (all three planes have the same dimensions).
    /// Returns the number of rows actually written: at most `max_rows`, and at
    /// most as many rows as the shortest plane holds at `stride` (its last row
    /// needs only the image width). With `max_rows > 0`, a plane shorter than
    /// one row is an [`InvalidBufferSize`](crate::decoder::ErrorKind::InvalidBufferSize)
    /// error and a `stride` below the width a
    /// [`StrideTooSmall`](crate::decoder::ErrorKind::StrideTooSmall) error, whether
    /// or not rows are left.
    ///
    /// YCbCr frames return their planes. RGB frames, and readers that hold
    /// decoded RGB (sampling shapes the strip does not serve, CMYK, XYB,
    /// orientation and deblock fallbacks), convert R, G, B with the BT.601 matrix; for the
    /// latter that is a re-derivation from rounded RGB.
    ///
    /// On a wave-parallel reader of a YCbCr frame (`parallel`, nearest-neighbour
    /// 4:2:2 / 4:2:0, MCU-row restart markers, >= 1024 MCUs; see
    /// [`Decoder::scanline_reader`](crate::decoder::Decoder::scanline_reader))
    /// this returns [`UnsupportedFeature`](crate::decoder::ErrorKind::UnsupportedFeature):
    /// that reader holds RGB rows, not the planes. Use `num_threads(1)`.
    ///
    /// For native-resolution chroma output (no upsampling), use
    /// [`read_rows_ycbcr_native_i16()`](Self::read_rows_ycbcr_native_i16) instead.
    pub fn read_rows_ycbcr_f32(
        &mut self,
        y_plane: &mut [f32],
        cb_plane: &mut [f32],
        cr_plane: &mut [f32],
        stride: usize,
        max_rows: usize,
    ) -> Result<usize> {
        let out_width = self.width() as usize;
        if stride < out_width {
            return Err(Error::stride_too_small(self.width(), stride));
        }
        // Rows the three planes can hold (the last row needs only `out_width`).
        // The arguments are checked before the reader's state, so a plane
        // shorter than one row is refused even when no rows are left.
        let shortest = y_plane.len().min(cb_plane.len()).min(cr_plane.len());
        let capacity = shortest
            .checked_sub(out_width)
            .map_or(0, |rest| rest / stride + 1);
        if max_rows > 0 && capacity == 0 {
            return Err(Error::invalid_buffer_size(out_width, shortest));
        }
        let max_rows = max_rows.min(capacity);
        self.check_plane_read(
            "read_rows_ycbcr_f32 needs the Y/Cb/Cr planes, which the wave-parallel reader \
             does not keep for YCbCr frames (use num_threads(1))",
        )?;
        let planes = self.planes;
        let precision = self.precision;
        self.read_rows_with(max_rows, |i, row| {
            let o = i * stride;
            let (y, cb, cr) = (
                &mut y_plane[o..o + out_width],
                &mut cb_plane[o..o + out_width],
                &mut cr_plane[o..o + out_width],
            );
            let mut put = |px: usize, (vy, vcb, vcr): (f32, f32, f32)| {
                (y[px], cb[px], cr[px]) = (vy, vcb, vcr);
            };
            match row {
                RowSrc::Native(p, n, ColorModel::Gray | ColorModel::YCbCr) => {
                    let max = ((1u32 << precision) - 1) as f32;
                    let center = (1u32 << (precision - 1)) as f32;
                    for x in 0..row.len() {
                        put(
                            x,
                            (
                                p[x * n] as f32 / max,
                                if n == 1 {
                                    0.0
                                } else {
                                    (p[x * n + 1] as f32 - center) / max
                                },
                                if n == 1 {
                                    0.0
                                } else {
                                    (p[x * n + 2] as f32 - center) / max
                                },
                            ),
                        );
                    }
                }
                RowSrc::Native(..) => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for x in 0..row.len() {
                        let [r, g, b] = row.rgb16(x, precision, planes).map(|v| v as f32 / max);
                        let yy = 0.299 * r + 0.587 * g + 0.114 * b;
                        put(x, (yy, (b - yy) / 1.772, (r - yy) / 1.402));
                    }
                }
                RowSrc::Planes(c0, c1, c2) if precision != 8 && planes != PlaneMatrix::Identity => {
                    let max = ((1u32 << precision) - 1) as f32;
                    let center = (1u32 << (precision - 1)) as f32;
                    for x in 0..c0.len() {
                        put(
                            x,
                            (
                                c0[x] as f32 / max,
                                if planes == PlaneMatrix::Gray {
                                    0.0
                                } else {
                                    (c1[x] as f32 - center) / max
                                },
                                if planes == PlaneMatrix::Gray {
                                    0.0
                                } else {
                                    (c2[x] as f32 - center) / max
                                },
                            ),
                        );
                    }
                }
                RowSrc::Planes(..) if precision != 8 => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for x in 0..row.len() {
                        let [r, g, b] = row.rgb16(x, precision, planes).map(|v| v as f32 / max);
                        let yy = 0.299 * r + 0.587 * g + 0.114 * b;
                        put(x, (yy, (b - yy) / 1.772, (r - yy) / 1.402));
                    }
                }
                RowSrc::Planes(c0, c1, c2) => match planes {
                    PlaneMatrix::YCbCr { .. } => {
                        for (px, ((&a, &b), &c)) in c0.iter().zip(c1).zip(c2).enumerate() {
                            put(
                                px,
                                (
                                    a as f32 / 255.0,
                                    (b as f32 - 128.0) / 255.0,
                                    (c as f32 - 128.0) / 255.0,
                                ),
                            );
                        }
                    }
                    PlaneMatrix::Identity => {
                        for (px, ((&r, &g), &b)) in c0.iter().zip(c1).zip(c2).enumerate() {
                            put(px, rgb_to_ycbcr_normalized(u8c(r), u8c(g), u8c(b)));
                        }
                    }
                    PlaneMatrix::Gray => {
                        for (px, &v) in c0.iter().enumerate() {
                            put(px, (v as f32 / 255.0, 0.0, 0.0));
                        }
                    }
                },
                RowSrc::Rgb8(rgb) => {
                    for (px, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
                        put(px, rgb_to_ycbcr_normalized(p[0], p[1], p[2]));
                    }
                }
                RowSrc::Gray8(gray) => {
                    for (px, &v) in gray.iter().enumerate() {
                        put(px, (v as f32 / 255.0, 0.0, 0.0));
                    }
                }
            }
        })
    }

    /// Read rows into a grayscale f32 buffer, normalized to [0, 1].
    ///
    /// Grayscale frames return their samples, YCbCr frames their Y plane, RGB
    /// frames (and readers holding decoded RGB) the BT.601 luma of R, G, B.
    /// With an f32 [`OutputTarget`](crate::decoder::OutputTarget) the gray and
    /// Y samples come from the unclamped IDCT, so ringing can leave [0, 1].
    ///
    /// On a wave-parallel reader of a YCbCr frame (`parallel`, nearest-neighbour
    /// 4:2:2 / 4:2:0, MCU-row restart markers, >= 1024 MCUs; see
    /// [`Decoder::scanline_reader`](crate::decoder::Decoder::scanline_reader))
    /// this returns [`UnsupportedFeature`](crate::decoder::ErrorKind::UnsupportedFeature):
    /// that reader holds RGB rows, not the planes. Use `num_threads(1)`.
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_gray_f32(&mut self, mut output: ImgRefMut<'_, f32>) -> Result<usize> {
        let out_width = self.width() as usize;
        if output.width() < out_width {
            return Err(Error::invalid_buffer_size(out_width, output.width()));
        }
        self.check_plane_read(
            "read_rows_gray_f32 needs the luma plane, which the wave-parallel reader \
             does not keep for YCbCr frames (use num_threads(1))",
        )?;
        let planes = self.planes;
        let (max_rows, stride) = (output.height(), output.stride());
        let buf = output.buf_mut();
        let precision = self.precision;
        self.read_rows_with(max_rows, |i, row| {
            let out = &mut buf[i * stride..i * stride + out_width];
            match row {
                RowSrc::Native(..) => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for (x, o) in out.iter_mut().enumerate() {
                        *o = row.gray16(x, precision, planes) as f32 / max;
                    }
                }
                RowSrc::Planes(..) if precision != 8 => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for (x, o) in out.iter_mut().enumerate() {
                        *o = row.gray16(x, precision, planes) as f32 / max;
                    }
                }
                RowSrc::Planes(c0, c1, c2) if planes == PlaneMatrix::Identity => {
                    for (o, ((&r, &g), &b)) in out.iter_mut().zip(c0.iter().zip(c1).zip(c2)) {
                        *o = rgb_luma_f32(u8c(r), u8c(g), u8c(b));
                    }
                }
                RowSrc::Planes(c0, _, _) => {
                    for (o, &v) in out.iter_mut().zip(c0) {
                        *o = v as f32 / 255.0;
                    }
                }
                RowSrc::Rgb8(rgb) => {
                    for (o, p) in out.iter_mut().zip(rgb.as_chunks::<3>().0.iter()) {
                        *o = rgb_luma_f32(p[0], p[1], p[2]);
                    }
                }
                RowSrc::Gray8(gray) => {
                    for (o, &v) in out.iter_mut().zip(gray) {
                        *o = v as f32 / 255.0;
                    }
                }
            }
        })
    }

    /// Read rows into a linear f32 grayscale buffer, range [0, 1].
    ///
    /// Grayscale and YCbCr frames linearize their gray / Y samples; RGB
    /// frames (and readers holding decoded RGB) linearize R, G, B and then
    /// take the BT.601 weighted sum. Samples go through the exact 8-bit sRGB
    /// table, so every reader mode returns the same bits.
    ///
    /// On a wave-parallel reader of a YCbCr frame (`parallel`, nearest-neighbour
    /// 4:2:2 / 4:2:0, MCU-row restart markers, >= 1024 MCUs; see
    /// [`Decoder::scanline_reader`](crate::decoder::Decoder::scanline_reader))
    /// this returns [`UnsupportedFeature`](crate::decoder::ErrorKind::UnsupportedFeature):
    /// that reader holds RGB rows, not the planes. Use `num_threads(1)`.
    ///
    /// Returns the number of rows actually written.
    pub fn read_rows_gray_linear_f32(&mut self, mut output: ImgRefMut<'_, f32>) -> Result<usize> {
        let out_width = self.width() as usize;
        if output.width() < out_width {
            return Err(Error::invalid_buffer_size(out_width, output.width()));
        }
        self.check_plane_read(
            "read_rows_gray_linear_f32 needs the luma plane, which the wave-parallel \
             reader does not keep for YCbCr frames (use num_threads(1))",
        )?;
        let planes = self.planes;
        let (max_rows, stride) = (output.height(), output.stride());
        let buf = output.buf_mut();
        let weighted = |r: f32, g: f32, b: f32| 0.299 * r + 0.587 * g + 0.114 * b;
        let precision = self.precision;
        self.read_rows_with(max_rows, |i, row| {
            let out = &mut buf[i * stride..i * stride + out_width];
            match row {
                RowSrc::Native(..) => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for (x, o) in out.iter_mut().enumerate() {
                        let [r, g, b] = row
                            .rgb16(x, precision, planes)
                            .map(|v| srgb_to_linear_f32(v as f32 / max));
                        *o = weighted(r, g, b);
                    }
                }
                RowSrc::Planes(y, _, _) if precision != 8 && planes != PlaneMatrix::Identity => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for (dst, &sample) in out.iter_mut().zip(y) {
                        *dst = srgb_to_linear_f32(sample as f32 / max);
                    }
                }
                RowSrc::Planes(..) if precision != 8 => {
                    let max = ((1u32 << precision) - 1) as f32;
                    for (x, o) in out.iter_mut().enumerate() {
                        let [r, g, b] = row
                            .rgb16(x, precision, planes)
                            .map(|v| srgb_to_linear_f32(v as f32 / max));
                        *o = weighted(r, g, b);
                    }
                }
                RowSrc::Planes(c0, c1, c2) if planes == PlaneMatrix::Identity => {
                    for (o, ((&r, &g), &b)) in out.iter_mut().zip(c0.iter().zip(c1).zip(c2)) {
                        *o = weighted(lin_i16(r), lin_i16(g), lin_i16(b));
                    }
                }
                RowSrc::Planes(c0, _, _) => {
                    for (o, &v) in out.iter_mut().zip(c0) {
                        *o = lin_i16(v);
                    }
                }
                RowSrc::Rgb8(rgb) => {
                    for (o, p) in out.iter_mut().zip(rgb.as_chunks::<3>().0.iter()) {
                        *o = weighted(
                            srgb_to_linear(p[0]),
                            srgb_to_linear(p[1]),
                            srgb_to_linear(p[2]),
                        );
                    }
                }
                RowSrc::Gray8(gray) => {
                    for (o, &v) in out.iter_mut().zip(gray) {
                        *o = srgb_to_linear(v);
                    }
                }
            }
        })
    }

    /// Decode the next wave of segments into the wave buffer.
    #[cfg(feature = "parallel")]
    fn decode_next_wave(&mut self) -> Result<()> {
        let state = self.wave_state.as_ref().unwrap();
        let seg_start = self.wave_next_seg;
        let seg_end = (seg_start + state.wave_size).min(state.num_segments);

        if seg_start >= seg_end {
            return Ok(());
        }

        let wave_count = seg_end - seg_start;
        let _ = wave_count;

        let row_count =
            state.decode_wave_box(&self.data, seg_start, seg_end, &mut self.wave_buf)?;

        self.wave_first_row = seg_start * state.pixel_rows_per_seg;
        self.wave_row_count = row_count;
        self.wave_next_seg = seg_end;
        Ok(())
    }

    /// Wave-parallel: serve planar i16 rows from wave buffer, triggering decode on demand.
    #[cfg(feature = "parallel")]
    fn read_rows_planar_i16_wave(
        &mut self,
        y: &mut [i16],
        y_stride: usize,
        cb: &mut [i16],
        cr: &mut [i16],
        c_stride: usize,
        max_mcu_rows: usize,
    ) -> Result<(usize, usize)> {
        let state = self.wave_state.as_ref().unwrap();
        let luma_width = state.width;
        let chroma_width = state.chroma_width();
        let luma_rows_per_mcu = state.mcu_pixel_height;
        let chroma_rows_per_mcu = if chroma_width > 0 {
            state.chroma_rows_per_seg() / (state.pixel_rows_per_seg / luma_rows_per_mcu).max(1)
        } else {
            0
        };
        let luma_rows_per_seg = state.pixel_rows_per_seg;
        let chroma_rows_per_seg = state.chroma_rows_per_seg();
        let out_height = state.height;
        let chroma_height = if chroma_width > 0 {
            (state.height + state.max_v_samp - 1) / state.max_v_samp
        } else {
            0
        };
        let total_mcu_rows = (out_height + luma_rows_per_mcu - 1) / luma_rows_per_mcu;

        // Lazy allocate wave planar buffers
        if self.wave_y.is_empty() {
            let ws = state.wave_size;
            let y_seg_samples = luma_rows_per_seg * luma_width;
            let c_seg_samples = chroma_rows_per_seg * chroma_width;
            self.wave_y = vec![0i16; ws * y_seg_samples];
            if chroma_width > 0 {
                self.wave_cb = vec![0i16; ws * c_seg_samples];
                self.wave_cr = vec![0i16; ws * c_seg_samples];
            }
        }

        let mut total_luma_rows = 0usize;
        let mut total_chroma_rows = 0usize;

        for _ in 0..max_mcu_rows {
            if self.current_mcu_row >= total_mcu_rows {
                break;
            }

            let luma_row_abs = self.current_mcu_row * luma_rows_per_mcu;
            let chroma_row_abs = self.current_mcu_row * chroma_rows_per_mcu;

            // Decode next wave if current MCU row is beyond what we have
            if luma_row_abs >= self.wave_planar_first_luma_row + self.wave_planar_luma_rows {
                self.decode_next_wave_planar()?;
            }

            // Luma rows for this MCU
            let remaining_luma = out_height.saturating_sub(luma_row_abs);
            let luma_rows_this = luma_rows_per_mcu.min(remaining_luma);

            // Chroma rows for this MCU
            let remaining_chroma = chroma_height.saturating_sub(chroma_row_abs);
            let chroma_rows_this = if chroma_width > 0 {
                chroma_rows_per_mcu.min(remaining_chroma)
            } else {
                0
            };

            // Copy luma rows from wave buffer
            let y_wave_row = luma_row_abs - self.wave_planar_first_luma_row;
            for row in 0..luma_rows_this {
                let src_off = (y_wave_row + row) * luma_width;
                let dst_off = (total_luma_rows + row) * y_stride;
                y[dst_off..dst_off + luma_width]
                    .copy_from_slice(&self.wave_y[src_off..src_off + luma_width]);
            }

            // Copy chroma rows from wave buffer
            if chroma_width > 0 && chroma_rows_this > 0 {
                let c_wave_row = chroma_row_abs - self.wave_planar_first_chroma_row;
                for row in 0..chroma_rows_this {
                    let src_off = (c_wave_row + row) * chroma_width;
                    let dst_off = (total_chroma_rows + row) * c_stride;
                    cb[dst_off..dst_off + chroma_width]
                        .copy_from_slice(&self.wave_cb[src_off..src_off + chroma_width]);
                    cr[dst_off..dst_off + chroma_width]
                        .copy_from_slice(&self.wave_cr[src_off..src_off + chroma_width]);
                }
            }

            total_luma_rows += luma_rows_this;
            total_chroma_rows += chroma_rows_this;
            self.current_row += luma_rows_this;
            self.current_mcu_row += 1;
        }

        Ok((total_luma_rows, total_chroma_rows))
    }

    /// Decode the next wave of segments into planar i16 buffers.
    #[cfg(feature = "parallel")]
    fn decode_next_wave_planar(&mut self) -> Result<()> {
        let state = self.wave_state.as_ref().unwrap();
        let seg_start = self.wave_planar_next_seg;
        let seg_end = (seg_start + state.wave_size).min(state.num_segments);

        if seg_start >= seg_end {
            return Ok(());
        }

        let luma_rows_per_seg = state.pixel_rows_per_seg;
        let chroma_rows_per_seg = state.chroma_rows_per_seg();

        let (luma_rows, chroma_rows) = state.decode_wave_planar(
            &self.data,
            seg_start,
            seg_end,
            &mut self.wave_y,
            &mut self.wave_cb,
            &mut self.wave_cr,
        )?;

        self.wave_planar_first_luma_row = seg_start * luma_rows_per_seg;
        self.wave_planar_luma_rows = luma_rows;
        self.wave_planar_first_chroma_row = seg_start * chroma_rows_per_seg;
        self.wave_planar_chroma_rows = chroma_rows;
        self.wave_planar_next_seg = seg_end;
        Ok(())
    }

    /// Read planar i16 Y/Cb/Cr at native chroma resolution (no upsampling, no color conversion).
    ///
    /// Outputs raw IDCT samples as i16 at the native resolution of each component:
    /// - Y plane: full image resolution
    /// - Cb/Cr planes: native chroma resolution (e.g., half width/height for 4:2:0)
    ///
    /// For grayscale images, `cb` and `cr` are unused (pass empty slices).
    ///
    /// For upsampled full-resolution f32 output, use
    /// [`read_rows_ycbcr_f32()`](Self::read_rows_ycbcr_f32) instead.
    ///
    /// # Arguments
    /// - `y`, `y_stride`: Luma output buffer and stride (in i16 elements)
    /// - `cb`, `cr`, `c_stride`: Chroma output buffers and stride (in i16 elements)
    /// - `max_mcu_rows`: Maximum number of MCU rows to decode
    ///
    /// # Returns
    /// `(luma_rows, chroma_rows)` — the number of rows written to each plane.
    /// For 4:2:0: luma_rows = 2 * chroma_rows. For 4:4:4: luma_rows = chroma_rows.
    pub fn read_rows_ycbcr_native_i16(
        &mut self,
        y: &mut [i16],
        y_stride: usize,
        cb: &mut [i16],
        cr: &mut [i16],
        c_stride: usize,
        max_mcu_rows: usize,
    ) -> Result<(usize, usize)> {
        match self.color_model {
            ColorModel::Rgb => {
                return Err(Error::unsupported_feature(
                    "read_rows_ycbcr_native_i16: this JPEG stores RGB components, so it has no \
                     native YCbCr planes (use read_rows_ycbcr_f32 or an RGB read)",
                ));
            }
            ColorModel::Xyb => {
                return Err(Error::unsupported_feature(
                    "read_rows_ycbcr_native_i16: this JPEG stores XYB components, so it has no \
                     native YCbCr planes (use an RGB read)",
                ));
            }
            ColorModel::Cmyk | ColorModel::Ycck | ColorModel::Unknown => {
                return Err(Error::unsupported_feature(
                    "read_rows_ycbcr_native_i16: this JPEG has no three YCbCr planes",
                ));
            }
            ColorModel::Gray | ColorModel::YCbCr => {}
        }
        if self.buffered_rgb.is_some()
            || self.predictive.is_some()
            || self.buffered_native.is_some()
            || self.component_rows.is_some()
        {
            return Err(Error::unsupported_feature(
                "read_rows_ycbcr_native_i16: this reader holds decoded pixels, not planes \
                 (progressive grayscale, a sampling shape the strip reader does not serve, \
                 or a transform/deblock fallback)",
            ));
        }
        if self.crop.is_some() {
            return Err(Error::unsupported_feature(
                "read_rows_ycbcr_native_i16 does not support crop",
            ));
        }

        // Wave-parallel mode
        #[cfg(feature = "parallel")]
        if self.wave_state.is_some() {
            return self.read_rows_planar_i16_wave(y, y_stride, cb, cr, c_stride, max_mcu_rows);
        }

        self.read_rows_planar_i16_seq(y, y_stride, cb, cr, c_stride, max_mcu_rows)
    }

    /// Sequential implementation of `read_rows_ycbcr_native_i16`.
    fn read_rows_planar_i16_seq(
        &mut self,
        y: &mut [i16],
        y_stride: usize,
        cb: &mut [i16],
        cr: &mut [i16],
        c_stride: usize,
        max_mcu_rows: usize,
    ) -> Result<(usize, usize)> {
        let luma_width = self.width as usize;
        let chroma_width = self.strip.chroma_strip_width;
        let luma_rows_per_mcu = self.strip.mcu_height;
        let chroma_rows_per_mcu = self.strip.chroma_strip_height;
        let out_height = self.height as usize;
        let is_grayscale = self.num_components == 1;

        let total_mcu_rows = (out_height + luma_rows_per_mcu - 1) / luma_rows_per_mcu;
        let mut total_luma_rows = 0usize;
        let mut total_chroma_rows = 0usize;

        for _ in 0..max_mcu_rows {
            if self.current_mcu_row >= total_mcu_rows {
                break;
            }

            // Decode this MCU row (entropy + IDCT + upsample; we ignore the upsample result)
            self.decode_mcu_row()?;

            // How many luma pixel rows in this MCU row?
            let remaining_luma =
                out_height.saturating_sub(self.current_mcu_row * luma_rows_per_mcu);
            let luma_rows_this = luma_rows_per_mcu.min(remaining_luma);

            // How many chroma pixel rows in this MCU row?
            let chroma_height_total = if is_grayscale {
                0
            } else {
                self.chroma_height() as usize
            };
            let remaining_chroma =
                chroma_height_total.saturating_sub(self.current_mcu_row * chroma_rows_per_mcu);
            let chroma_rows_this = chroma_rows_per_mcu.min(remaining_chroma);

            let strip_cols_y = luma_width.min(self.strip.strip_width);
            let strip_cols_c = chroma_width.min(self.strip.chroma_strip_stride);

            // Copy luma rows
            for row in 0..luma_rows_this {
                let src = self.strip.y_row(row, strip_cols_y);
                let dst_off = (total_luma_rows + row) * y_stride;
                y[dst_off..dst_off + luma_width.min(strip_cols_y)]
                    .copy_from_slice(&src[..luma_width.min(strip_cols_y)]);
            }

            // Copy chroma rows
            if !is_grayscale {
                for row in 0..chroma_rows_this {
                    let (cb_src, cr_src) = self.strip.chroma_row_native(row, strip_cols_c);
                    let dst_off = (total_chroma_rows + row) * c_stride;
                    let copy_width = chroma_width.min(strip_cols_c);
                    cb[dst_off..dst_off + copy_width].copy_from_slice(&cb_src[..copy_width]);
                    cr[dst_off..dst_off + copy_width].copy_from_slice(&cr_src[..copy_width]);
                }
            }

            total_luma_rows += luma_rows_this;
            total_chroma_rows += chroma_rows_this;

            // Advance to next MCU row
            self.current_row += luma_rows_this;
            self.advance_mcu_row();
        }

        Ok((total_luma_rows, total_chroma_rows))
    }
}

/// Convert sRGB u8 to linear f32 via 256-entry LUT.
#[inline]
fn srgb_to_linear(srgb: u8) -> f32 {
    linear_srgb::default::srgb_u8_to_linear(srgb)
}

/// The plane conversion a strip reader applies for `model`. Models without
/// one (XYB, CMYK, YCCK) never reach a strip reader (`PathPlan::strip`); a
/// caller that sends one is a bug, reported rather than converted wrongly.
fn plane_matrix(model: ColorModel, idct_method: super::IdctMethod) -> Result<PlaneMatrix> {
    model.plane_matrix(idct_method).ok_or_else(|| {
        Error::internal("strip reader built for a colour model without a row kernel")
    })
}

/// Clamp an IDCT sample to 8 bits.
#[inline(always)]
fn u8c(v: i16) -> u8 {
    v.clamp(0, 255) as u8
}

/// Linearize an integer IDCT sample through the exact 8-bit table.
#[inline(always)]
fn lin_i16(v: i16) -> f32 {
    srgb_to_linear(u8c(v))
}

/// One output row's source data, cropped to the output columns.
#[derive(Clone, Copy)]
enum RowSrc<'r> {
    Native(&'r [u16], usize, ColorModel),
    /// Decoded RGB, three bytes per pixel (buffered and wave readers).
    Rgb8(&'r [u8]),
    /// Decoded gray, one byte per pixel (buffered reader of a grayscale frame).
    Gray8(&'r [u8]),
    /// Strip planes (streaming and coefficient readers). `c1`/`c2` are empty
    /// for grayscale; their meaning is the reader's [`PlaneMatrix`].
    Planes(&'r [i16], &'r [i16], &'r [i16]),
}

impl RowSrc<'_> {
    fn len(self) -> usize {
        match self {
            Self::Native(p, c, _) => p.len() / c,
            Self::Planes(y, _, _) => y.len(),
            Self::Rgb8(p) => p.len() / 3,
            Self::Gray8(p) => p.len(),
        }
    }
    fn rgb16(self, x: usize, precision: u8, matrix: PlaneMatrix) -> [u16; 3] {
        match self {
            Self::Native(p, 1, _) => [p[x]; 3],
            Self::Native(p, c, model) => {
                let p = &p[x * c..(x + 1) * c];
                let max = (1u32 << precision) - 1;
                match model {
                    ColorModel::YCbCr => native_ycbcr([p[0], p[1], p[2]], precision),
                    ColorModel::Cmyk | ColorModel::Ycck => {
                        let cmy = if model == ColorModel::Ycck {
                            native_ycbcr([p[0], p[1], p[2]], precision).map(|v| max as u16 - v)
                        } else {
                            [p[0], p[1], p[2]]
                        };
                        cmy.map(|v| ((v as u32 * p[3] as u32 + max / 2) / max) as u16)
                    }
                    _ => [p[0], p[1], p[2]],
                }
            }
            Self::Gray8(p) => [p[x] as u16; 3],
            Self::Rgb8(p) => [p[x * 3] as u16, p[x * 3 + 1] as u16, p[x * 3 + 2] as u16],
            Self::Planes(y, cb, cr) => {
                if precision == 8 {
                    let mut pixel = [0u8; 3];
                    RowKernel::new(matrix, Layout8::Rgb).row(
                        &y[x..x + 1],
                        if cb.is_empty() { &[] } else { &cb[x..x + 1] },
                        if cr.is_empty() { &[] } else { &cr[x..x + 1] },
                        &mut pixel,
                    );
                    return pixel.map(u16::from);
                }
                let max = (1i64 << precision) - 1;
                let clamp = |v: i64| v.clamp(0, max) as u16;
                match matrix {
                    PlaneMatrix::Gray => [clamp(y[x] as i64); 3],
                    PlaneMatrix::Identity => {
                        [clamp(y[x] as i64), clamp(cb[x] as i64), clamp(cr[x] as i64)]
                    }
                    PlaneMatrix::YCbCr { .. } => native_ycbcr(
                        [clamp(y[x] as i64), clamp(cb[x] as i64), clamp(cr[x] as i64)],
                        precision,
                    ),
                }
            }
        }
    }
    fn write_native_u8(self, precision: u8, matrix: PlaneMatrix, layout: Layout8, out: &mut [u8]) {
        let max = (1u32 << precision) - 1;
        let scale = |v: u16| ((v as u32 * 255 + max / 2) / max) as u8;
        for (x, dst) in out.chunks_exact_mut(layout.bytes_per_pixel()).enumerate() {
            if layout == Layout8::Gray {
                dst[0] = scale(self.gray16(x, precision, matrix));
            } else {
                let [r, g, b] = self.rgb16(x, precision, matrix).map(scale);
                let pixel = if matches!(layout, Layout8::Bgr | Layout8::Bgrx) {
                    [b, g, r]
                } else {
                    [r, g, b]
                };
                dst[..3].copy_from_slice(&pixel);
                if dst.len() == 4 {
                    dst[3] = 255;
                }
            }
        }
    }

    fn gray16(self, x: usize, precision: u8, matrix: PlaneMatrix) -> u16 {
        if let Self::Native(p, n, ColorModel::Gray | ColorModel::YCbCr) = self {
            return p[x * n];
        }
        if let Self::Planes(y, _, _) = self
            && matrix != PlaneMatrix::Identity
        {
            return (y[x] as i64).clamp(0, (1i64 << precision) - 1) as u16;
        }
        let [r, g, b] = self.rgb16(x, precision, matrix).map(u64::from);
        ((19595 * r + 38470 * g + 7471 * b + 32768) >> 16) as u16
    }
}

/// Convert sRGB f32 (0.0-1.0) to linear f32.
///
/// Out-of-range values from IDCT are clamped to [0, 1] — JPEG pixel values
/// are inherently [0, 255], so clamping is correct for decode output.
#[inline]
fn srgb_to_linear_f32(s: f32) -> f32 {
    linear_srgb::default::srgb_to_linear(s)
}

// =============================================================================
// Boundary 4-tap deblocking helpers for i16 strip planes.
//
// These mirror `deblock::boundary::filter_plane_boundary_4tap` but operate on
// i16 data in a strip (one MCU row) rather than a full f32 plane.
// =============================================================================

/// Apply vertical boundary filtering (columns at multiples of 8) to an i16
/// strip plane. `width` is the true pixel width; `stride` >= width.
/// `height` is the number of rows in the strip.
fn filter_strip_vertical_i16(
    plane: &mut [i16],
    width: usize,
    stride: usize,
    height: usize,
    strength: &BoundaryStrength,
) {
    if strength.max_delta < 0.5 || width < 16 || height < 2 {
        return;
    }

    let thresh = strength.threshold;
    let max_d = strength.max_delta;

    let num_boundaries = width / 8;

    for bx in 1..num_boundaries {
        let col = bx * 8;
        if col + 1 >= width || col < 2 {
            continue;
        }

        for y in 0..height {
            let base = y * stride;
            let p1 = plane[base + col - 2] as f32;
            let p0 = plane[base + col - 1] as f32;
            let q0 = plane[base + col] as f32;
            let q1 = plane[base + col + 1] as f32;

            let disc = (p0 - q0).abs();
            if disc < thresh {
                continue;
            }

            let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
            let delta_p = (avg - p0).clamp(-max_d, max_d);
            let delta_q = (avg - q0).clamp(-max_d, max_d);

            plane[base + col - 1] = (p0 + delta_p).round() as i16;
            plane[base + col] = (q0 + delta_q).round() as i16;
        }
    }
}

/// Apply horizontal boundary filtering (rows at multiples of 8) WITHIN a
/// single strip (intra-MCU boundaries). For an 8-row MCU there are no
/// intra-MCU boundaries; for a 16-row MCU there is one at row 8.
fn filter_strip_horizontal_i16(
    plane: &mut [i16],
    width: usize,
    stride: usize,
    height: usize,
    strength: &BoundaryStrength,
) {
    if strength.max_delta < 0.5 || width < 2 || height < 16 {
        return;
    }

    let thresh = strength.threshold;
    let max_d = strength.max_delta;

    let num_boundaries = height / 8;

    for by in 1..num_boundaries {
        let row = by * 8;
        if row + 1 >= height || row < 2 {
            continue;
        }

        let off_p1 = (row - 2) * stride;
        let off_p0 = (row - 1) * stride;
        let off_q0 = row * stride;
        let off_q1 = (row + 1) * stride;

        for x in 0..width {
            let p1 = plane[off_p1 + x] as f32;
            let p0 = plane[off_p0 + x] as f32;
            let q0 = plane[off_q0 + x] as f32;
            let q1 = plane[off_q1 + x] as f32;

            let disc = (p0 - q0).abs();
            if disc < thresh {
                continue;
            }

            let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
            let delta_p = (avg - p0).clamp(-max_d, max_d);
            let delta_q = (avg - q0).clamp(-max_d, max_d);

            plane[off_p0 + x] = (p0 + delta_p).round() as i16;
            plane[off_q0 + x] = (q0 + delta_q).round() as i16;
        }
    }
}

/// Apply horizontal boundary filtering at the junction between two MCU rows.
///
/// `prev_rows` contains the last 2 rows of the previous MCU row's strip at
/// `prev_offset`: row[-2] at `prev_offset`, row[-1] at `prev_offset + prev_stride`.
/// `curr_plane` is the current strip; row[0] at offset 0, row[1] at `curr_stride`.
///
/// The 4-tap filter straddles the boundary: p1=row[-2], p0=row[-1], q0=row[0], q1=row[1].
/// Only modifies `curr_plane` row[0] (q0 side); the previous MCU row's data has already
/// been served to the caller, so we don't modify `prev_rows`. This is a one-sided filter
/// at the MCU junction — slightly weaker than the full two-sided filter, but avoids
/// requiring access to already-served output.
fn filter_inter_mcu_horizontal_i16(
    prev_rows: &[i16],
    prev_offset: usize,
    prev_stride: usize,
    curr_plane: &mut [i16],
    curr_stride: usize,
    width: usize,
    strength: &BoundaryStrength,
) {
    if strength.max_delta < 0.5 || width < 2 {
        return;
    }
    // Need at least 2 rows in current plane for q0/q1
    if curr_plane.len() < 2 * curr_stride {
        return;
    }

    let thresh = strength.threshold;
    let max_d = strength.max_delta;

    for x in 0..width {
        let p1 = prev_rows[prev_offset + x] as f32;
        let p0 = prev_rows[prev_offset + prev_stride + x] as f32;
        let q0 = curr_plane[x] as f32;
        let q1 = curr_plane[curr_stride + x] as f32;

        let disc = (p0 - q0).abs();
        if disc < thresh {
            continue;
        }

        let avg = (p1 + 3.0 * p0 + 3.0 * q0 + q1) * 0.125;
        let delta_q = (avg - q0).clamp(-max_d, max_d);

        // Only modify the current MCU row's side (q0).
        curr_plane[x] = (q0 + delta_q).round() as i16;
    }
}

fn native_ycbcr(p: [u16; 3], precision: u8) -> [u16; 3] {
    let max = (1i64 << precision) - 1;
    let [y, b, r] = p.map(i64::from);
    let (b, r) = (b - (1 << (precision - 1)), r - (1 << (precision - 1)));
    [
        y + ((91881 * r + 32768) >> 16),
        y + ((-22554 * b - 46802 * r + 32768) >> 16),
        y + ((116130 * b + 32768) >> 16),
    ]
    .map(|v| v.clamp(0, max) as u16)
}

fn frame_component_ids(data: &[u8]) -> [u8; MAX_COMPONENTS] {
    use crate::container::marker::{self, MarkerKind};
    let mut ids = [0; MAX_COMPONENTS];
    if let Some(frame) = marker::iter(data).find(|s| matches!(s.kind, MarkerKind::Sof(_)))
        && let Some(components) = frame.payload.get(6..)
    {
        for (id, c) in ids.iter_mut().zip(components.as_chunks::<3>().0.iter()) {
            *id = c[0];
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_strip_vertical_i16() {
        // 32x8 plane with sharp step at column 8
        let stride = 32;
        let width = 32;
        let height = 8;
        let mut plane = vec![0i16; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = if x < 8 { 50 } else { 200 };
            }
        }
        let strength = BoundaryStrength::from_dc_quant(20);
        filter_strip_vertical_i16(&mut plane, width, stride, height, &strength);

        // Pixels at boundary should be pulled toward each other
        let p0 = plane[2 * stride + 7]; // col 7
        let q0 = plane[2 * stride + 8]; // col 8
        assert!(p0 > 50, "p0 should increase from 50: got {p0}");
        assert!(q0 < 200, "q0 should decrease from 200: got {q0}");
    }

    #[test]
    fn test_filter_strip_horizontal_i16() {
        // 16x16 plane with sharp step at row 8
        let stride = 16;
        let width = 16;
        let height = 16;
        let mut plane = vec![0i16; stride * height];
        for y in 0..height {
            for x in 0..width {
                plane[y * stride + x] = if y < 8 { 50 } else { 200 };
            }
        }
        let strength = BoundaryStrength::from_dc_quant(20);
        filter_strip_horizontal_i16(&mut plane, width, stride, height, &strength);

        // Pixels at horizontal boundary should be pulled toward each other
        let p0 = plane[7 * stride + 5]; // row 7
        let q0 = plane[8 * stride + 5]; // row 8
        assert!(p0 > 50, "p0 should increase from 50: got {p0}");
        assert!(q0 < 200, "q0 should decrease from 200: got {q0}");
    }

    /// Helper to encode RGB pixels with 4:4:4 (no subsampling).
    /// This ensures the streaming decode path is used, which matches the scanline reader's
    /// integer IDCT implementation.
    fn encode_rgb(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
        use crate::encode::v2::{ChromaSubsampling, EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        // Use 4:4:4 to ensure streaming decode path is used (same IDCT as scanline reader)
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    /// Helper to encode RGB pixels with subsampling (baseline mode for test stability)
    fn encode_rgb_subsampled(
        width: u32,
        height: u32,
        pixels: &[u8],
        quality: f32,
        subsampling: crate::encode::v2::ChromaSubsampling,
    ) -> Vec<u8> {
        use crate::encode::v2::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        let config = EncoderConfig::ycbcr(quality, subsampling).progressive(false);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    /// Compare two u8 slices and return (max_diff, diff_count, first_diff_idx)
    fn compare_u8_slices(a: &[u8], b: &[u8]) -> (u8, usize, Option<usize>) {
        assert_eq!(a.len(), b.len(), "slice length mismatch");
        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        let mut first_diff_idx: Option<usize> = None;

        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va as i16 - vb as i16).unsigned_abs() as u8;
            if diff > 0 {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        (max_diff, diff_count, first_diff_idx)
    }

    /// Compare two f32 slices and return (max_diff, diff_count, first_diff_idx)
    #[allow(dead_code)]
    fn compare_f32_slices(a: &[f32], b: &[f32]) -> (f32, usize, Option<usize>) {
        assert_eq!(a.len(), b.len(), "slice length mismatch");
        let mut max_diff: f32 = 0.0;
        let mut diff_count: usize = 0;
        let mut first_diff_idx: Option<usize> = None;

        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            if diff > 1e-6 {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        (max_diff, diff_count, first_diff_idx)
    }

    /// Assert slices are equal, with detailed diff info on failure
    fn assert_slices_equal_u8(actual: &[u8], expected: &[u8], context: &str) {
        let (max_diff, diff_count, first_diff_idx) = compare_u8_slices(actual, expected);
        if diff_count > 0 {
            let first_idx = first_diff_idx.unwrap();
            panic!(
                "{}: slices differ - max_diff={}, diff_count={}/{} ({:.2}%), first_diff at idx {} (actual={}, expected={})",
                context,
                max_diff,
                diff_count,
                actual.len(),
                100.0 * diff_count as f64 / actual.len() as f64,
                first_idx,
                actual[first_idx],
                expected[first_idx]
            );
        }
    }

    /// Assert f32 slices are equal, with detailed diff info on failure
    #[allow(dead_code)]
    fn assert_slices_equal_f32(actual: &[f32], expected: &[f32], context: &str) {
        let (max_diff, diff_count, first_diff_idx) = compare_f32_slices(actual, expected);
        if diff_count > 0 {
            let first_idx = first_diff_idx.unwrap();
            panic!(
                "{}: slices differ - max_diff={:.6}, diff_count={}/{} ({:.2}%), first_diff at idx {} (actual={:.6}, expected={:.6})",
                context,
                max_diff,
                diff_count,
                actual.len(),
                100.0 * diff_count as f64 / actual.len() as f64,
                first_idx,
                actual[first_idx],
                expected[first_idx]
            );
        }
    }

    #[test]
    fn test_srgb_to_linear() {
        // Black
        assert!((srgb_to_linear(0) - 0.0).abs() < 1e-6);
        // White
        assert!((srgb_to_linear(255) - 1.0).abs() < 1e-6);
        // Mid-gray (sRGB 128 ≈ linear 0.2159)
        assert!((srgb_to_linear(128) - 0.2159).abs() < 0.01);
    }

    #[test]
    fn test_scanline_reader_rgb8() {
        use crate::decode::Decoder;

        // Create test image - 64x48 for multiple MCU rows
        let width = 64u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 4) as u8; // R gradient
                pixels[idx + 1] = (y * 5) as u8; // G gradient
                pixels[idx + 2] = 128; // B constant
            }
        }

        // Encode as baseline 4:4:4 (default)
        let jpeg = encode_rgb(width, height, &pixels, 95.0);

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);

        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];

        // Read all rows
        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let stride = (width * 3) as usize;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);

        // Compare outputs - should be identical
        assert_eq!(
            scanline_pixels.len(),
            decoded.pixels_u8().unwrap().len(),
            "output size mismatch"
        );
        assert_slices_equal_u8(
            &scanline_pixels,
            decoded.pixels_u8().unwrap(),
            "test_scanline_reader_rgb8",
        );
    }

    #[test]
    fn test_scanline_reader_partial_reads() {
        use crate::decode::Decoder;

        // Create test image - 32x32
        let width = 32u32;
        let height = 32u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = ((x + y) * 4) as u8;
                pixels[idx + 1] = ((x * 2 + y) % 256) as u8;
                pixels[idx + 2] = ((y * 2 + x) % 256) as u8;
            }
        }

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Read in small chunks (3 rows at a time)
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let chunk_size = 3; // Read 3 rows at a time
            let rows_to_read = chunk_size.min(height as usize - total_rows);
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, rows_to_read);
            let rows = reader.read_rows_rgb8(output).expect("read failed");
            assert!(rows > 0 || reader.is_finished());
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_slices_equal_u8(
            &scanline_pixels,
            decoded.pixels_u8().unwrap(),
            "test_scanline_reader_partial_reads",
        );
    }

    #[test]
    fn test_scanline_reader_rgbx8() {
        use crate::decode::Decoder;

        let width = 24u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 7) % 256) as u8;
        }

        let jpeg = encode_rgb(width, height, &pixels, 85.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut rgbx_pixels = vec![0u8; (width * height * 4) as usize];
        let stride = (width * 4) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut rgbx_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgbx8(output).expect("read failed");
            total_rows += rows;
        }

        // Verify RGBX matches RGB with alpha=255
        // First collect stats
        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        let mut first_diff: Option<(usize, usize, &str, u8, u8)> = None;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let rgb_idx = (y * width as usize + x) * 3;
                let rgbx_idx = (y * width as usize + x) * 4;

                for (c, name) in [(0, "R"), (1, "G"), (2, "B")] {
                    let actual = rgbx_pixels[rgbx_idx + c];
                    let expected = decoded.pixels_u8().unwrap()[rgb_idx + c];
                    let diff = (actual as i16 - expected as i16).unsigned_abs() as u8;
                    if diff > 0 {
                        diff_count += 1;
                        if first_diff.is_none() {
                            first_diff = Some((x, y, name, actual, expected));
                        }
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
                assert_eq!(rgbx_pixels[rgbx_idx + 3], 255, "Alpha should be 255");
            }
        }

        if diff_count > 0 {
            let (x, y, ch, actual, expected) = first_diff.unwrap();
            let total = (width * height * 3) as usize;
            panic!(
                "test_scanline_reader_rgbx8: max_diff={}, diff_count={}/{} ({:.2}%), first_diff at ({},{}) {}={} expected={}",
                max_diff,
                diff_count,
                total,
                100.0 * diff_count as f64 / total as f64,
                x,
                y,
                ch,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_scanline_reader_rgba_f32() {
        use crate::decode::Decoder;

        let width = 16u32;
        let height = 16u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 11) % 256) as u8;
        }

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut rgba_pixels = vec![0.0f32; (width * height * 4) as usize];
        let stride = (width * 4) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut rgba_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgba_f32(output).expect("read failed");
            total_rows += rows;
        }

        // Verify values are in valid range
        for (i, &val) in rgba_pixels.iter().enumerate() {
            if i % 4 == 3 {
                // Alpha channel
                assert!(
                    (val - 1.0).abs() < 1e-6,
                    "Alpha at {} should be 1.0, got {}",
                    i,
                    val
                );
            } else {
                // RGB channels should be in [0, 1] range
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Value at {} should be in [0,1], got {}",
                    i,
                    val
                );
            }
        }

        // Verify RGB matches (converting back from linear)
        let mut max_diff: f32 = 0.0;
        let mut diff_count: usize = 0;
        let mut first_diff: Option<(usize, usize, usize, f32, f32)> = None;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let rgb_idx = (y * width as usize + x) * 3;
                let rgba_idx = (y * width as usize + x) * 4;

                for c in 0..3 {
                    let expected_linear = srgb_to_linear(decoded.pixels_u8().unwrap()[rgb_idx + c]);
                    let actual_linear = rgba_pixels[rgba_idx + c];
                    let diff = (expected_linear - actual_linear).abs();
                    if diff > 0.01 {
                        diff_count += 1;
                        if first_diff.is_none() {
                            first_diff = Some((x, y, c, actual_linear, expected_linear));
                        }
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
            }
        }

        if diff_count > 0 {
            let (x, y, c, actual, expected) = first_diff.unwrap();
            let total = (width * height * 3) as usize;
            panic!(
                "test_scanline_reader_rgba_f32: max_diff={:.6}, diff_count={}/{} ({:.2}%), first_diff at ({},{}) ch{}={:.6} expected={:.6}",
                max_diff,
                diff_count,
                total,
                100.0 * diff_count as f64 / total as f64,
                x,
                y,
                c,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_scanline_reader_ycbcr_planes() {
        use crate::decode::Decoder;

        let width = 32u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 13) % 256) as u8;
        }

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let plane_size = (width * height) as usize;
        let mut y_plane = vec![0.0f32; plane_size];
        let mut cb_plane = vec![0.0f32; plane_size];
        let mut cr_plane = vec![0.0f32; plane_size];

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let offset = total_rows * width as usize;
            let rows = reader
                .read_rows_ycbcr_f32(
                    &mut y_plane[offset..],
                    &mut cb_plane[offset..],
                    &mut cr_plane[offset..],
                    width as usize,
                    remaining,
                )
                .expect("read failed");
            total_rows += rows;
        }

        // Verify Y values are in [0, 1] and Cb/Cr in [-0.5, 0.5]
        for i in 0..plane_size {
            assert!(
                (0.0..=1.0).contains(&y_plane[i]),
                "Y[{}] = {} out of range",
                i,
                y_plane[i]
            );
            assert!(
                (-0.6..=0.6).contains(&cb_plane[i]),
                "Cb[{}] = {} out of range",
                i,
                cb_plane[i]
            );
            assert!(
                (-0.6..=0.6).contains(&cr_plane[i]),
                "Cr[{}] = {} out of range",
                i,
                cr_plane[i]
            );
        }
    }

    #[test]
    fn test_scanline_reader_non_mcu_aligned() {
        use crate::decode::Decoder;

        // Non-MCU-aligned dimensions (not multiples of 8)
        let width = 37u32;
        let height = 29u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 7) as u8;
                pixels[idx + 1] = (y * 9) as u8;
                pixels[idx + 2] = ((x + y) * 3) as u8;
            }
        }

        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgb8(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_slices_equal_u8(
            &scanline_pixels,
            decoded.pixels_u8().unwrap(),
            "test_scanline_reader_non_mcu_aligned",
        );
    }

    #[test]
    fn test_scanline_reader_420() {
        use crate::decode::Decoder;
        use crate::encode::v2::ChromaSubsampling;

        // Create test image - 64x48 for multiple MCU rows
        // 4:2:0 has 16x16 MCUs, so this is 4x3 MCUs
        let width = 64u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 4) as u8; // R gradient
                pixels[idx + 1] = (y * 5) as u8; // G gradient
                pixels[idx + 2] = 128; // B constant
            }
        }

        // Encode as 4:2:0
        let jpeg = encode_rgb_subsampled(width, height, &pixels, 95.0, ChromaSubsampling::Quarter);

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);
        assert_eq!(reader.subsampling(), Subsampling::S420);

        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_eq!(
            scanline_pixels.len(),
            decoded.pixels_u8().unwrap().len(),
            "output size mismatch"
        );

        // Compare outputs with tolerance - scanline reader uses simpler i16 processing
        // while regular decoder uses f32 with bias computation, so outputs won't be bit-identical
        let mut max_diff = 0i32;
        let mut total_diff = 0u64;
        for (i, (&a, &b)) in scanline_pixels
            .iter()
            .zip(decoded.pixels_u8().unwrap().iter())
            .enumerate()
        {
            let diff = (a as i32 - b as i32).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as u64;
            if diff > 10 {
                panic!(
                    "Pixel at index {} differs by {} (scanline={}, regular={})",
                    i, diff, a, b
                );
            }
        }
        let avg_diff = total_diff as f64 / scanline_pixels.len() as f64;
        assert!(
            avg_diff < 3.0,
            "Average pixel difference {} too high (max diff: {})",
            avg_diff,
            max_diff
        );
    }

    #[test]
    fn test_scanline_reader_420_non_mcu_aligned() {
        use crate::decode::Decoder;
        use crate::encode::v2::ChromaSubsampling;

        // Non-MCU-aligned dimensions (not multiples of 16 for 4:2:0)
        let width = 37u32;
        let height = 29u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 7) as u8;
                pixels[idx + 1] = (y * 9) as u8;
                pixels[idx + 2] = ((x + y) * 3) as u8;
            }
        }

        // Encode as 4:2:0
        let jpeg = encode_rgb_subsampled(width, height, &pixels, 90.0, ChromaSubsampling::Quarter);

        // Decode normally for comparison
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgb8(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        assert_eq!(
            scanline_pixels.len(),
            decoded.pixels_u8().unwrap().len(),
            "output size mismatch"
        );

        // Compare with tolerance
        let mut max_diff = 0i32;
        let mut total_diff = 0u64;
        for (i, (&a, &b)) in scanline_pixels
            .iter()
            .zip(decoded.pixels_u8().unwrap().iter())
            .enumerate()
        {
            let diff = (a as i32 - b as i32).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as u64;
            if diff > 10 {
                panic!(
                    "Pixel at index {} differs by {} (scanline={}, regular={})",
                    i, diff, a, b
                );
            }
        }
        let avg_diff = total_diff as f64 / scanline_pixels.len() as f64;
        assert!(
            avg_diff < 3.0,
            "Average pixel difference {} too high (max diff: {})",
            avg_diff,
            max_diff
        );
    }

    /// Helper to encode grayscale pixels.
    fn encode_grayscale(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
        use crate::encode::v2::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        let config = EncoderConfig::grayscale(quality);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_scanline_reader_grayscale_basic() {
        use crate::decode::Decoder;
        use crate::types::PixelFormat;

        // Create test grayscale image - 64x48 for multiple MCU rows
        let width = 64u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                // Diagonal gradient
                pixels[idx] = ((x + y) * 2) as u8;
            }
        }

        // Encode as grayscale
        let jpeg = encode_grayscale(width, height, &pixels, 95.0);

        // Decode normally for comparison - use Gray output format
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        // Decode via scanline reader
        let mut reader = Decoder::new()
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed for grayscale");
        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);
        assert!(reader.is_grayscale());
        assert_eq!(reader.num_components(), 1);

        let mut scanline_pixels = vec![0u8; (width * height) as usize];

        // Read all rows using grayscale method
        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let stride = width as usize;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_gray8(output)
                .expect("read_rows_gray8 failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);

        // Compare outputs - should match within JPEG compression tolerance
        assert_eq!(
            scanline_pixels.len(),
            decoded.pixels_u8().unwrap().len(),
            "output size mismatch"
        );

        let (max_diff, diff_count, _) =
            compare_u8_slices(&scanline_pixels, decoded.pixels_u8().unwrap());
        assert!(
            max_diff <= 2,
            "grayscale scanline reader max_diff {} > 2 (diff_count: {})",
            max_diff,
            diff_count
        );
    }

    #[test]
    fn test_scanline_reader_grayscale_non_mcu_aligned() {
        use crate::decode::Decoder;
        use crate::types::PixelFormat;

        // Non-MCU-aligned dimensions (not multiples of 8)
        let width = 37u32;
        let height = 29u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                pixels[idx] = (x * 7 + y * 3) as u8;
            }
        }

        let jpeg = encode_grayscale(width, height, &pixels, 90.0);

        // Use Gray output format for comparison
        let decoder = Decoder::new().output_format(PixelFormat::Gray);
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");

        let mut reader = Decoder::new()
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        assert!(reader.is_grayscale());

        let mut scanline_pixels = vec![0u8; (width * height) as usize];
        let stride = width as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_gray8(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);
        let (max_diff, _, _) = compare_u8_slices(&scanline_pixels, decoded.pixels_u8().unwrap());
        assert!(
            max_diff <= 2,
            "grayscale non-MCU-aligned max_diff {} > 2",
            max_diff
        );
    }

    #[test]
    fn test_scanline_reader_grayscale_f32() {
        use crate::decode::Decoder;

        let width = 32u32;
        let height = 24u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for i in 0..pixels.len() {
            pixels[i] = ((i * 13) % 256) as u8;
        }

        let jpeg = encode_grayscale(width, height, &pixels, 90.0);

        let decoder = Decoder::new();
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");

        let mut gray_pixels = vec![0.0f32; (width * height) as usize];
        let stride = width as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut gray_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_gray_f32(output).expect("read failed");
            total_rows += rows;
        }

        assert_eq!(total_rows, height as usize);

        // Verify values are in valid [0, 1] range
        for (i, &val) in gray_pixels.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Value at {} should be in [0,1], got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_scanline_reader_grayscale_linear_f32() {
        use crate::decode::Decoder;

        let width = 16u32;
        let height = 16u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for i in 0..pixels.len() {
            pixels[i] = (i % 256) as u8;
        }

        let jpeg = encode_grayscale(width, height, &pixels, 95.0);

        let decoder = Decoder::new();
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");

        let mut linear_pixels = vec![0.0f32; (width * height) as usize];
        let stride = width as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output = imgref::ImgRefMut::new(&mut linear_pixels[buf_start..], stride, remaining);
            let rows = reader
                .read_rows_gray_linear_f32(output)
                .expect("read failed");
            total_rows += rows;
        }

        // Verify values are in valid [0, 1] range
        for (i, &val) in linear_pixels.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Linear value at {} should be in [0,1], got {}",
                i,
                val
            );
        }

        // Verify linear conversion: sRGB 128 ≈ linear 0.2159
        // Find pixels close to 128 and verify they're near 0.2159 in linear
        // (Since JPEG is lossy, we can't be exact, but the relationship should hold)
    }

    /// Encode to progressive JPEG using the encoder
    fn encode_progressive_rgb(width: u32, height: u32, pixels: &[u8], quality: f32) -> Vec<u8> {
        use crate::encode::v2::{ChromaSubsampling, EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        // Use progressive mode
        let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::None).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(pixels, Unstoppable).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn test_scanline_reader_progressive() {
        // Test that progressive JPEG works via buffered mode
        let width = 16u32;
        let height = 16u32;
        let mut input_pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                input_pixels[idx] = ((x * 16) % 256) as u8; // R
                input_pixels[idx + 1] = ((y * 16) % 256) as u8; // G
                input_pixels[idx + 2] = 128; // B
            }
        }

        // Encode as progressive JPEG
        let progressive_jpeg = encode_progressive_rgb(width, height, &input_pixels, 95.0);

        // Verify it's actually progressive by checking SOF marker
        assert!(
            progressive_jpeg.windows(2).any(|w| w == [0xFF, 0xC2]), // SOF2 = progressive
            "JPEG should be progressive (SOF2)"
        );

        // Decode via scanline reader
        let decoder = crate::decode::Decoder::new();
        let mut reader = decoder
            .scanline_reader(&progressive_jpeg)
            .expect("scanline_reader should support progressive via buffered mode");

        assert_eq!(reader.width(), width);
        assert_eq!(reader.height(), height);

        // Read all rows
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let mut rows_read = 0;
        while rows_read < height as usize {
            let remaining = height as usize - rows_read;
            let output = ImgRefMut::new(
                &mut scanline_pixels[rows_read * width as usize * 3..],
                width as usize * 3,
                remaining,
            );
            let count = reader
                .read_rows_rgb8(output)
                .expect("read_rows_rgb8 failed");
            if count == 0 {
                break;
            }
            rows_read += count;
        }

        assert_eq!(rows_read, height as usize, "Should read all rows");

        // Compare with full-frame decode
        let decoded = decoder
            .decode(&progressive_jpeg, enough::Unstoppable)
            .expect("decode failed");
        let (max_diff, diff_count, _) =
            compare_u8_slices(&scanline_pixels, decoded.pixels_u8().unwrap());

        // Should be identical (same decode path for progressive)
        assert_eq!(
            max_diff, 0,
            "Scanline reader should match full-frame decode for progressive JPEG (max diff={}, diff_count={})",
            max_diff, diff_count
        );
    }

    #[test]
    fn test_scanline_reader_progressive_grayscale() {
        // Test that progressive grayscale JPEG works via buffered mode
        let width = 16u32;
        let height = 16u32;
        let mut input_pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                input_pixels[idx] = ((x * 16 + y * 8) % 256) as u8;
            }
        }

        // Encode grayscale as progressive JPEG
        use crate::encode::v2::{EncoderConfig, PixelLayout};
        use enough::Unstoppable;
        let config = EncoderConfig::grayscale(95.0).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
            .unwrap();
        enc.push_packed(&input_pixels, Unstoppable).unwrap();
        let progressive_jpeg = enc.finish().unwrap();

        // Verify it's actually progressive
        assert!(
            progressive_jpeg.windows(2).any(|w| w == [0xFF, 0xC2]),
            "JPEG should be progressive (SOF2)"
        );

        // Decode via scanline reader
        let decoder = crate::decode::Decoder::new();
        let mut reader = decoder
            .scanline_reader(&progressive_jpeg)
            .expect("scanline_reader should support progressive grayscale");

        // Read grayscale rows
        let mut scanline_pixels = vec![0u8; (width * height) as usize];
        let mut rows_read = 0;
        while rows_read < height as usize {
            let remaining = height as usize - rows_read;
            let output = ImgRefMut::new(
                &mut scanline_pixels[rows_read * width as usize..],
                width as usize,
                remaining,
            );
            let count = reader
                .read_rows_gray8(output)
                .expect("read_rows_gray8 failed");
            if count == 0 {
                break;
            }
            rows_read += count;
        }

        assert_eq!(rows_read, height as usize, "Should read all rows");

        // The grayscale values should be reasonable (can't compare exactly since
        // the full-frame decoder uses PixelFormat::Rgb which converts grayscale to RGB)
        let mean: f64 =
            scanline_pixels.iter().map(|&x| x as f64).sum::<f64>() / scanline_pixels.len() as f64;
        assert!(
            mean > 50.0 && mean < 200.0,
            "Grayscale mean should be reasonable: {}",
            mean
        );
    }

    /// Test that scanline 4:2:0 bottom-boundary chroma is corrected.
    ///
    /// Encodes a high-contrast image as 4:2:0, decodes via both buffered and
    /// scanline paths, and verifies that the scanline output at MCU boundaries
    /// (rows 15, 31, etc.) closely matches the buffered decoder.
    #[test]
    fn test_scanline_420_bottom_boundary_fixup() {
        use crate::decode::Decoder;
        use crate::encode::v2::ChromaSubsampling;

        // Use a 64x64 image (4x4 MCUs at 16x16 each for 4:2:0)
        // with high-contrast content to make chroma boundary errors visible
        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                // High-contrast color pattern - alternating colored stripes
                if (y / 8) % 2 == 0 {
                    pixels[idx] = 255; // Red
                    pixels[idx + 1] = 0;
                    pixels[idx + 2] = 0;
                } else {
                    pixels[idx] = 0;
                    pixels[idx + 1] = 0;
                    pixels[idx + 2] = 255; // Blue
                }
            }
        }

        let jpeg = encode_rgb_subsampled(width, height, &pixels, 90.0, ChromaSubsampling::Quarter);

        // Decode via buffered path (reference - has correct boundary handling)
        let decoder = Decoder::new();
        let decoded = decoder
            .decode(&jpeg, enough::Unstoppable)
            .expect("decode failed");
        let reference = decoded.pixels_u8().unwrap();

        // Decode via scanline path
        let mut reader = decoder
            .scanline_reader(&jpeg)
            .expect("scanline_reader failed");
        let mut scanline_pixels = vec![0u8; (width * height * 3) as usize];
        let stride = (width * 3) as usize;

        let mut total_rows = 0;
        while !reader.is_finished() {
            let remaining = height as usize - total_rows;
            let buf_start = total_rows * stride;
            let output =
                imgref::ImgRefMut::new(&mut scanline_pixels[buf_start..], stride, remaining);
            let rows = reader.read_rows_rgb8(output).expect("read failed");
            total_rows += rows;
        }

        // Check max diff at MCU boundary rows (rows 15, 31, 47 for 4:2:0)
        let mcu_height = 16usize;
        let mut boundary_max_diff = 0i32;
        let mut interior_max_diff = 0i32;

        for y in 0..height as usize {
            for x in 0..width as usize {
                for c in 0..3 {
                    let idx = (y * width as usize + x) * 3 + c;
                    let diff = (scanline_pixels[idx] as i32 - reference[idx] as i32).abs();
                    let is_boundary = y % mcu_height == mcu_height - 1;
                    if is_boundary {
                        boundary_max_diff = boundary_max_diff.max(diff);
                    } else {
                        interior_max_diff = interior_max_diff.max(diff);
                    }
                }
            }
        }

        // After the fix, boundary rows should have similar error to interior rows.
        // Previously boundary_max_diff could be ~43, now should be <=4 (IDCT rounding).
        assert!(
            boundary_max_diff <= 4,
            "Bottom boundary max diff {} too high (interior max diff: {}). \
             Bottom boundary fixup may not be working.",
            boundary_max_diff,
            interior_max_diff
        );
    }

    // ========================================================================
    // Crop tests
    // ========================================================================

    /// Full scanline decode, then extract crop region from pixel buffer.
    /// Uses scanline reader (not decode()) so IDCT/upsampling matches the crop path.
    fn full_scanline_and_crop(jpeg: &[u8], cx: u32, cy: u32, cw: u32, ch: u32) -> Vec<u8> {
        use crate::decode::DecodeConfig;
        let mut reader = DecodeConfig::new().scanline_reader(jpeg).unwrap();
        let width = reader.width() as usize;
        let height = reader.height() as usize;
        let mut full = vec![0u8; width * height * 3];
        let mut rows_read = 0;
        while !reader.is_finished() {
            let remaining = height - rows_read;
            let output =
                imgref::ImgRefMut::new(&mut full[rows_read * width * 3..], width * 3, remaining);
            rows_read += reader.read_rows_rgb8(output).unwrap();
        }
        let bpp = 3;
        let mut cropped = vec![0u8; cw as usize * ch as usize * bpp];
        for y in 0..ch as usize {
            let src_off = ((cy as usize + y) * width + cx as usize) * bpp;
            let dst_off = y * cw as usize * bpp;
            let row_bytes = cw as usize * bpp;
            cropped[dst_off..dst_off + row_bytes]
                .copy_from_slice(&full[src_off..src_off + row_bytes]);
        }
        cropped
    }

    /// Full decode (buffered), then extract crop region from pixel buffer.
    fn full_decode_and_crop(jpeg: &[u8], cx: u32, cy: u32, cw: u32, ch: u32) -> Vec<u8> {
        use crate::decode::DecodeConfig;
        let result = DecodeConfig::new()
            .decode(jpeg, enough::Unstoppable)
            .unwrap();
        let width = result.width() as usize;
        let pixels = result.pixels_u8().unwrap();
        let bpp = 3; // RGB
        let mut cropped = vec![0u8; cw as usize * ch as usize * bpp];
        for y in 0..ch as usize {
            let src_off = ((cy as usize + y) * width + cx as usize) * bpp;
            let dst_off = y * cw as usize * bpp;
            let row_bytes = cw as usize * bpp;
            cropped[dst_off..dst_off + row_bytes]
                .copy_from_slice(&pixels[src_off..src_off + row_bytes]);
        }
        cropped
    }

    #[test]
    fn test_crop_pixel_scanline_444() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 4) as u8;
                pixels[idx + 1] = (y * 4) as u8;
                pixels[idx + 2] = 128;
            }
        }
        let jpeg = encode_rgb(width, height, &pixels, 95.0);
        let (cx, cy, cw, ch) = (10u32, 10u32, 20u32, 20u32);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let reference = full_decode_and_crop(&jpeg, cx, cy, cw, ch);

            let mut reader = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .scanline_reader(&jpeg)
                .unwrap();
            let out_w = cw as usize;
            let out_h = ch as usize;
            let mut out = vec![0u8; out_w * out_h * 3];
            let mut rows_read = 0;
            while !reader.is_finished() {
                let remaining = out_h - rows_read;
                let output =
                    imgref::ImgRefMut::new(&mut out[rows_read * out_w * 3..], out_w * 3, remaining);
                rows_read += reader.read_rows_rgb8(output).unwrap();
            }
            assert_eq!(rows_read, out_h);
            assert_eq!(out, reference, "crop 444 mismatch at {perm}");
        });
    }

    #[test]
    fn test_crop_pixel_scanline_420() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 4) as u8;
                pixels[idx + 1] = (y * 4) as u8;
                pixels[idx + 2] = 128;
            }
        }
        let jpeg = encode_rgb_subsampled(
            width,
            height,
            &pixels,
            95.0,
            crate::encode::v2::ChromaSubsampling::Quarter,
        );
        let (cx, cy, cw, ch) = (8u32, 16u32, 32u32, 24u32);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let reference = full_scanline_and_crop(&jpeg, cx, cy, cw, ch);

            let mut reader = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .scanline_reader(&jpeg)
                .unwrap();
            let out_w = cw as usize;
            let out_h = ch as usize;
            let mut out = vec![0u8; out_w * out_h * 3];
            let mut rows_read = 0;
            while !reader.is_finished() {
                let remaining = out_h - rows_read;
                let output =
                    imgref::ImgRefMut::new(&mut out[rows_read * out_w * 3..], out_w * 3, remaining);
                rows_read += reader.read_rows_rgb8(output).unwrap();
            }
            assert_eq!(rows_read, out_h);
            assert_eq!(out, reference, "crop 420 mismatch at {perm}");
        });
    }

    #[test]
    fn test_crop_percent() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 100u32;
        let height = 100u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = x as u8;
                pixels[idx + 1] = y as u8;
                pixels[idx + 2] = 64;
            }
        }
        let jpeg = encode_rgb(width, height, &pixels, 95.0);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let result = DecodeConfig::new()
                .crop(CropRegion::percent(0.25, 0.25, 0.5, 0.5))
                .decode(&jpeg, enough::Unstoppable)
                .unwrap();
            assert_eq!(result.width(), 50);
            assert_eq!(result.height(), 50);
            let reference = full_decode_and_crop(&jpeg, 25, 25, 50, 50);
            assert_eq!(
                result.pixels_u8().unwrap(),
                &reference[..],
                "crop percent mismatch at {perm}"
            );
        });
    }

    #[test]
    fn test_crop_full_image_is_noop() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 48u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for i in 0..pixels.len() {
            pixels[i] = (i * 7) as u8;
        }
        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let full = DecodeConfig::new()
                .decode(&jpeg, enough::Unstoppable)
                .unwrap();
            let cropped = DecodeConfig::new()
                .crop(CropRegion::pixels(0, 0, width, height))
                .decode(&jpeg, enough::Unstoppable)
                .unwrap();
            assert_eq!(full.width(), cropped.width());
            assert_eq!(full.height(), cropped.height());
            assert_eq!(
                full.pixels_u8().unwrap(),
                cropped.pixels_u8().unwrap(),
                "full-image crop noop mismatch at {perm}"
            );
        });
    }

    #[test]
    fn test_crop_streaming_dimensions() {
        // Verify width(), height(), current_row(), is_finished() reflect crop
        let width = 64u32;
        let height = 64u32;
        let pixels = vec![128u8; (width * height * 3) as usize];
        let jpeg = encode_rgb(width, height, &pixels, 90.0);

        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;

        let mut reader = DecodeConfig::new()
            .crop(CropRegion::pixels(10, 20, 30, 15))
            .scanline_reader(&jpeg)
            .unwrap();

        assert_eq!(reader.width(), 30);
        assert_eq!(reader.height(), 15);
        assert_eq!(reader.current_row(), 0);
        assert!(!reader.is_finished());

        // Read all rows
        let out_w = 30usize;
        let mut out = vec![0u8; out_w * 15 * 3];
        let output = imgref::ImgRefMut::new(&mut out, out_w * 3, 15);
        let n = reader.read_rows_rgb8(output).unwrap();
        assert_eq!(n, 15);
        assert!(reader.is_finished());
        assert_eq!(reader.current_row(), 15);
    }

    #[test]
    fn test_crop_buffered_progressive() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x * 4) as u8;
                pixels[idx + 1] = (y * 4) as u8;
                pixels[idx + 2] = 100;
            }
        }
        use crate::encode::v2::{ChromaSubsampling, EncoderConfig, PixelLayout};
        let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::None).progressive(true);
        let mut enc = config
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .unwrap();
        enc.push_packed(&pixels, enough::Unstoppable).unwrap();
        let jpeg = enc.finish().unwrap();
        let (cx, cy, cw, ch) = (8u32, 8u32, 32u32, 32u32);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let reference = full_decode_and_crop(&jpeg, cx, cy, cw, ch);

            // decode() path
            let result = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .decode(&jpeg, enough::Unstoppable)
                .unwrap();
            assert_eq!(result.width(), cw);
            assert_eq!(result.height(), ch);
            assert_eq!(
                result.pixels_u8().unwrap(),
                &reference[..],
                "progressive crop decode() mismatch at {perm}"
            );

            // scanline_reader path
            let mut reader = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .scanline_reader(&jpeg)
                .unwrap();
            let out_w = cw as usize;
            let out_h = ch as usize;
            let mut out = vec![0u8; out_w * out_h * 3];
            let mut rows_read = 0;
            while !reader.is_finished() {
                let remaining = out_h - rows_read;
                let output =
                    imgref::ImgRefMut::new(&mut out[rows_read * out_w * 3..], out_w * 3, remaining);
                rows_read += reader.read_rows_rgb8(output).unwrap();
            }
            assert_eq!(
                out, reference,
                "progressive crop scanline mismatch at {perm}"
            );
        });
    }

    #[test]
    fn test_crop_grayscale() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 48u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                pixels[(y * width + x) as usize] = ((x + y) * 3) as u8;
            }
        }
        let jpeg = encode_grayscale(width, height, &pixels, 95.0);
        let (cx, cy, cw, ch) = (4u32, 4u32, 24u32, 24u32);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let full = DecodeConfig::new()
                .decode(&jpeg, enough::Unstoppable)
                .unwrap();
            let full_pix = full.pixels_u8().unwrap();
            let full_w = full.width() as usize;
            let bpp = full.format().bytes_per_pixel();
            let mut reference = vec![0u8; cw as usize * ch as usize * bpp];
            for y in 0..ch as usize {
                let src_off = ((cy as usize + y) * full_w + cx as usize) * bpp;
                let dst_off = y * cw as usize * bpp;
                let row_bytes = cw as usize * bpp;
                reference[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&full_pix[src_off..src_off + row_bytes]);
            }
            let cropped = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .decode(&jpeg, enough::Unstoppable)
                .unwrap();
            assert_eq!(cropped.width(), cw);
            assert_eq!(cropped.height(), ch);
            assert_eq!(
                cropped.pixels_u8().unwrap(),
                &reference[..],
                "grayscale crop mismatch at {perm}"
            );
        });
    }

    #[test]
    fn test_crop_gray8_scanline() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        let width = 48u32;
        let height = 48u32;
        let mut pixels = vec![0u8; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                pixels[(y * width + x) as usize] = ((x + y) * 3) as u8;
            }
        }
        let jpeg = encode_grayscale(width, height, &pixels, 95.0);
        let (cx, cy, cw, ch) = (4u32, 4u32, 24u32, 24u32);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            // Full scanline decode + manual crop
            let mut full_reader = DecodeConfig::new().scanline_reader(&jpeg).unwrap();
            let fw = full_reader.width() as usize;
            let fh = full_reader.height() as usize;
            let mut full_gray = vec![0u8; fw * fh];
            let mut rows_read = 0;
            while !full_reader.is_finished() {
                let remaining = fh - rows_read;
                let output =
                    imgref::ImgRefMut::new(&mut full_gray[rows_read * fw..], fw, remaining);
                rows_read += full_reader.read_rows_gray8(output).unwrap();
            }
            let mut ref_gray = vec![0u8; cw as usize * ch as usize];
            for y in 0..ch as usize {
                let src_off = (cy as usize + y) * fw + cx as usize;
                let dst_off = y * cw as usize;
                ref_gray[dst_off..dst_off + cw as usize]
                    .copy_from_slice(&full_gray[src_off..src_off + cw as usize]);
            }

            // Crop scanline decode
            let mut reader = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .scanline_reader(&jpeg)
                .unwrap();
            let out_w = cw as usize;
            let out_h = ch as usize;
            let mut out = vec![0u8; out_w * out_h];
            let mut rows_read = 0;
            while !reader.is_finished() {
                let remaining = out_h - rows_read;
                let output =
                    imgref::ImgRefMut::new(&mut out[rows_read * out_w..], out_w, remaining);
                rows_read += reader.read_rows_gray8(output).unwrap();
            }
            assert_eq!(out, ref_gray, "gray8 crop scanline mismatch at {perm}");
        });
    }

    #[test]
    fn test_crop_validation_errors() {
        let jpeg = encode_rgb(64, 64, &vec![128u8; 64 * 64 * 3], 90.0);

        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;

        // Crop exceeds image bounds
        let err = DecodeConfig::new()
            .crop(CropRegion::pixels(50, 50, 20, 20))
            .decode(&jpeg, enough::Unstoppable);
        assert!(err.is_err(), "Crop exceeding bounds should fail");

        // Zero width crop
        let err = DecodeConfig::new()
            .crop(CropRegion::pixels(0, 0, 0, 10))
            .decode(&jpeg, enough::Unstoppable);
        assert!(err.is_err(), "Zero width crop should fail");

        // Zero height crop
        let err = DecodeConfig::new()
            .crop(CropRegion::pixels(0, 0, 10, 0))
            .decode(&jpeg, enough::Unstoppable);
        assert!(err.is_err(), "Zero height crop should fail");

        // Percent out of range
        let err = DecodeConfig::new()
            .crop(CropRegion::percent(0.0, 0.0, 1.5, 1.0))
            .decode(&jpeg, enough::Unstoppable);
        assert!(err.is_err(), "Percent > 1.0 should fail");
    }

    #[test]
    fn test_crop_non_mcu_boundary() {
        use crate::decode::DecodeConfig;
        use crate::decode::config::CropRegion;
        use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

        // Test crop at non-MCU-aligned boundaries with 4:2:0
        // MCU size for 4:2:0 is 16x16
        let width = 96u32;
        let height = 96u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = x as u8;
                pixels[idx + 1] = y as u8;
                pixels[idx + 2] = ((x + y) / 2) as u8;
            }
        }
        let jpeg = encode_rgb_subsampled(
            width,
            height,
            &pixels,
            95.0,
            crate::encode::v2::ChromaSubsampling::Quarter,
        );

        // Non-MCU-aligned crop: (5, 7, 37, 29)
        let (cx, cy, cw, ch) = (5u32, 7u32, 37u32, 29u32);

        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let reference = full_scanline_and_crop(&jpeg, cx, cy, cw, ch);

            let mut reader = DecodeConfig::new()
                .crop(CropRegion::pixels(cx, cy, cw, ch))
                .scanline_reader(&jpeg)
                .unwrap();

            assert_eq!(reader.width(), cw);
            assert_eq!(reader.height(), ch);

            let out_w = cw as usize;
            let out_h = ch as usize;
            let mut out = vec![0u8; out_w * out_h * 3];
            let mut rows_read = 0;
            while !reader.is_finished() {
                let remaining = out_h - rows_read;
                let output =
                    imgref::ImgRefMut::new(&mut out[rows_read * out_w * 3..], out_w * 3, remaining);
                rows_read += reader.read_rows_rgb8(output).unwrap();
            }
            assert_eq!(
                out, reference,
                "Non-MCU-aligned crop differs from reference at {perm}"
            );
        });
    }
}
