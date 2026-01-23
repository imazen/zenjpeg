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

mod markers;
mod output;
mod progressive;
mod scan;

use super::idct::inverse_dct_8x8;
use super::idct_int::{idct_int_auto, idct_int_tiered};
use super::upsample::upsample_fancy;
use super::{JpegInfo, ScanInfo};
use crate::color::icc::{extract_icc_profile, is_xyb_profile};
use crate::color::{
    gray_f32_to_gray_f32, gray_f32_to_gray_u8, gray_f32_to_rgb_f32, gray_f32_to_rgb_u8,
    ycbcr_planes_f32_to_rgb_f32, ycbcr_planes_f32_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8,
};
use crate::entropy::EntropyDecoder;
use crate::error::{Error, Result, ScanRead};
use crate::foundation::alloc::{
    checked_size_2d, try_alloc_dct_blocks, try_alloc_maybeuninit, validate_dimensions,
};
use crate::foundation::consts::{
    DCT_BLOCK_SIZE, DCT_SIZE, JPEG_NATURAL_ORDER, MARKER_APP0, MARKER_COM, MARKER_DHT, MARKER_DQT,
    MARKER_DRI, MARKER_EOI, MARKER_SOF0, MARKER_SOF1, MARKER_SOF2, MARKER_SOI, MARKER_SOS,
    MAX_COMPONENTS, MAX_HUFFMAN_TABLES, MAX_QUANT_TABLES,
};
use crate::huffman::HuffmanDecodeTable;
use crate::quant::{
    dequantize_block, dequantize_block_i32, dequantize_block_with_bias, dequantize_unzigzag_i32,
    dequantize_unzigzag_i32_into, DequantBiasStats,
};
use crate::types::{ColorSpace, Component, Dimensions, JpegMode, PixelFormat};

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

    // Streaming decode result (used for baseline 4:4:4 JPEGs)
    pub(super) streaming_rgb: Option<Vec<u8>>,
    // Whether to prefer streaming decode (set false for f32 output which needs coefficients)
    pub(super) prefer_streaming: bool,

    // ICC profile (extracted from raw data, not during parsing)
    pub(super) icc_profile: Option<Vec<u8>>,

    // Security limits
    pub(super) max_pixels: u64,
}

impl<'a> JpegParser<'a> {
    pub(super) fn new(data: &'a [u8], max_pixels: u64) -> Result<Self> {
        // Check for SOI
        if data.len() < 2 || data[0] != 0xFF || data[1] != MARKER_SOI {
            return Err(Error::invalid_jpeg_data("missing SOI marker"));
        }

        // Extract ICC profile from raw data upfront
        let icc_profile = extract_icc_profile(data);

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
            streaming_rgb: None,
            prefer_streaming: true, // Default to streaming for RGB decode
            icc_profile,
            max_pixels,
        })
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
    pub(super) fn decode(&mut self) -> Result<()> {
        // First read header
        self.position = 2; // Skip SOI
        self.read_header()?;

        // Continue parsing until we hit EOI
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOS => {
                    self.parse_scan()?;
                    // After scan, look for more markers
                }
                MARKER_DQT => self.parse_quant_table()?,
                MARKER_DHT => self.parse_huffman_table()?,
                MARKER_DRI => self.parse_restart_interval()?,
                MARKER_EOI => break,
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                _ => self.skip_segment()?,
            }
        }

        Ok(())
    }

    /// Finds the SOS marker and extracts scan info without decoding.
    /// Used by scanline reader to get table mapping and data start position.
    pub(super) fn find_scan_info(&mut self) -> Result<ScanInfo> {
        // Continue from current position to find SOS
        loop {
            let marker = self.read_marker()?;

            match marker {
                MARKER_SOS => {
                    let _length = self.read_u16()?;
                    let num_components = self.read_u8()?;

                    if num_components != 3 {
                        return Err(Error::unsupported_feature(
                            "scanline reader requires 3 components in scan",
                        ));
                    }

                    let mut table_mapping = [(0usize, 0usize); 3];

                    for _i in 0..num_components as usize {
                        let component_id = self.read_u8()?;
                        let tables = self.read_u8()?;
                        let dc_table = (tables >> 4) as usize;
                        let ac_table = (tables & 0x0F) as usize;

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
                MARKER_APP0..=0xEF | MARKER_COM => self.skip_segment()?,
                MARKER_EOI => {
                    return Err(Error::invalid_jpeg_data("unexpected EOI before SOS"));
                }
                _ => self.skip_segment()?,
            }
        }
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
            if ids == [82, 71, 66] || ids == [b'R', b'G', b'B'] {
                ColorSpace::Rgb
            } else {
                ColorSpace::YCbCr
            }
        } else if self.num_components == 4 {
            ColorSpace::Cmyk
        } else {
            ColorSpace::Unknown
        };

        JpegInfo {
            dimensions: Dimensions {
                width: self.width,
                height: self.height,
            },
            color_space,
            precision: self.precision,
            num_components: self.num_components,
            mode: self.mode,
            has_icc_profile: self.icc_profile.is_some(),
            is_xyb,
        }
    }

    pub(super) fn extract_coefficients(&self) -> Result<super::super::image::DecodedCoefficients> {
        use super::super::image::{ComponentCoefficients, DecodedCoefficients};

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
            let blocks_h = mcu_cols * h_samp;
            let blocks_v = mcu_rows * v_samp;

            let quant_table = self.quant_tables[self.components[i].quant_table_idx as usize]
                .ok_or(Error::internal("missing quantization table"))?;

            components.push(ComponentCoefficients {
                blocks: self.coeffs[i].clone(),
                blocks_h,
                blocks_v,
                h_samp_factor: h_samp as u8,
                v_samp_factor: v_samp as u8,
                quant_table,
            });
        }

        Ok(DecodedCoefficients {
            width: self.width,
            height: self.height,
            components,
        })
    }
}
