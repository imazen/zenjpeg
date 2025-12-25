//! Main encoder implementation
//!
//! Provides the public Encoder API that combines mozjpeg and jpegli approaches
//! for Pareto-optimal JPEG compression.

use crate::adaptive_quant::compute_aq_field;
use crate::color::{convert_rgb_to_ycbcr, deinterleave_ycbcr};
use crate::dct::{forward_dct_8x8_int, level_shift, descale_for_quant, scale_for_trellis, quantize_block_int};
use crate::entropy::{EntropyEncoder, SymbolCounter};
use crate::error::Error;
use crate::huffman::{std_ac_luma, std_dc_luma, DerivedTable, FrequencyCounter, HuffTable};
use crate::quant::QuantTableSet;
use crate::strategy::{select_strategy, SelectedStrategy};
use crate::trellis::{trellis_quantize_block, TrellisConfig};
use crate::types::{EncodingStrategy, PixelFormat, Quality, Subsampling};
use crate::Result;

/// Standard AC luma derived table for trellis rate estimation.
/// Created once and reused - mozjpeg uses standard tables for trellis.
fn get_std_ac_derived() -> DerivedTable {
    let htbl = std_ac_luma();
    DerivedTable::from_huff_table(&htbl, false).expect("Standard AC table should be valid")
}

/// JPEG encoder with configurable quality and encoding strategy
#[derive(Clone)]
pub struct Encoder {
    quality: Quality,
    strategy: EncodingStrategy,
    subsampling: Subsampling,
    progressive: Option<bool>,
    optimize_huffman: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// Create a new encoder with default settings
    pub fn new() -> Self {
        Self {
            quality: Quality::Standard(85),
            strategy: EncodingStrategy::Auto,
            subsampling: Subsampling::S444,
            progressive: None, // Auto-select based on strategy
            optimize_huffman: true,
        }
    }

    /// Create encoder optimized for maximum compression
    pub fn max_compression() -> Self {
        Self {
            quality: Quality::Low(75),
            strategy: EncodingStrategy::Mozjpeg,
            subsampling: Subsampling::S420,
            progressive: Some(true),
            optimize_huffman: true,
        }
    }

    /// Create encoder optimized for maximum quality
    pub fn max_quality() -> Self {
        Self {
            quality: Quality::High(95),
            strategy: EncodingStrategy::Jpegli,
            subsampling: Subsampling::S444,
            progressive: Some(false),
            optimize_huffman: true,
        }
    }

    /// Create fastest encoder (sacrifices some compression)
    pub fn fastest() -> Self {
        Self {
            quality: Quality::Standard(85),
            strategy: EncodingStrategy::Mozjpeg,
            subsampling: Subsampling::S420,
            progressive: Some(false),
            optimize_huffman: false,
        }
    }

    /// Set the quality level
    pub fn quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Set the encoding strategy
    pub fn strategy(mut self, strategy: EncodingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set chroma subsampling mode
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Enable or disable progressive encoding
    pub fn progressive(mut self, progressive: bool) -> Self {
        self.progressive = Some(progressive);
        self
    }

    /// Enable or disable Huffman table optimization
    pub fn optimize_huffman(mut self, optimize: bool) -> Self {
        self.optimize_huffman = optimize;
        self
    }

    /// Encode RGB image data to JPEG
    pub fn encode_rgb(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        self.validate_dimensions(width, height)?;
        self.validate_pixel_data(pixels, width, height, PixelFormat::Rgb8)?;

        // Select encoding strategy
        let selected = select_strategy(&self.quality, self.strategy);

        // Convert to YCbCr
        let ycbcr = convert_rgb_to_ycbcr(pixels, width, height);
        let (y_plane, cb_plane, cr_plane) = deinterleave_ycbcr(&ycbcr, width, height);

        // Encode with selected strategy
        self.encode_ycbcr_planes(&y_plane, &cb_plane, &cr_plane, width, height, &selected)
    }

    /// Encode grayscale image data to JPEG
    pub fn encode_gray(&self, pixels: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        self.validate_dimensions(width, height)?;
        self.validate_pixel_data(pixels, width, height, PixelFormat::Gray8)?;

        let selected = select_strategy(&self.quality, self.strategy);

        // Grayscale uses only Y component
        self.encode_gray_plane(pixels, width, height, &selected)
    }

    /// Validate image dimensions
    fn validate_dimensions(&self, width: usize, height: usize) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions must be non-zero",
            });
        }

        if width > 65535 || height > 65535 {
            return Err(Error::InvalidDimensions {
                width,
                height,
                reason: "dimensions exceed JPEG maximum (65535)",
            });
        }

        Ok(())
    }

    /// Validate pixel data length
    fn validate_pixel_data(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        let expected = width * height * format.bytes_per_pixel();
        if pixels.len() != expected {
            return Err(Error::InvalidPixelData {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(())
    }

    /// Encode YCbCr planes to JPEG
    fn encode_ycbcr_planes(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        width: usize,
        height: usize,
        strategy: &SelectedStrategy,
    ) -> Result<Vec<u8>> {
        let q = self.quality.value() as u8;
        let quant_tables = QuantTableSet::standard(q);

        // Create standard AC derived table for trellis rate estimation
        // mozjpeg uses standard tables for this - they work well enough
        let ac_derived = get_std_ac_derived();

        // Compute adaptive quantization field if enabled
        let aq_field = if strategy.adaptive_quant.enabled {
            compute_aq_field(y_plane, width, height, &strategy.adaptive_quant)
        } else {
            crate::adaptive_quant::AqField::uniform((width + 7) / 8, (height + 7) / 8)
        };

        let width_blocks = (width + 7) / 8;
        let height_blocks = (height + 7) / 8;
        let total_blocks = width_blocks * height_blocks;

        // Quantize all blocks first (needed for both standard and optimized paths)
        let mut all_coeffs: Vec<[i16; 64]> = Vec::with_capacity(total_blocks * 3);

        for by in 0..height_blocks {
            for bx in 0..width_blocks {
                // Y block
                let y_block = extract_block(y_plane, width, height, bx, by);
                let y_coeffs = self.quantize_block_impl(
                    &y_block,
                    &quant_tables.luma.values,
                    aq_field.get(bx, by),
                    strategy,
                    &ac_derived,
                );
                all_coeffs.push(y_coeffs);

                // Cb block
                let cb_block = extract_block(cb_plane, width, height, bx, by);
                let cb_coeffs = self.quantize_block_impl(
                    &cb_block,
                    &quant_tables.chroma.values,
                    1.0,
                    strategy,
                    &ac_derived,
                );
                all_coeffs.push(cb_coeffs);

                // Cr block
                let cr_block = extract_block(cr_plane, width, height, bx, by);
                let cr_coeffs = self.quantize_block_impl(
                    &cr_block,
                    &quant_tables.chroma.values,
                    1.0,
                    strategy,
                    &ac_derived,
                );
                all_coeffs.push(cr_coeffs);
            }
        }

        // Get Huffman tables (optimized or standard)
        let (dc_htbl, ac_htbl) = if self.optimize_huffman {
            // Two-pass encoding: count symbol frequencies, then generate optimal tables
            let mut dc_counter = FrequencyCounter::new();
            let mut ac_counter = FrequencyCounter::new();
            let mut symbol_counter = SymbolCounter::new();

            // Count symbols in all blocks
            for (i, coeffs) in all_coeffs.iter().enumerate() {
                let component = i % 3; // Y=0, Cb=1, Cr=2
                symbol_counter.count_block(coeffs, component, &mut dc_counter, &mut ac_counter);
            }

            // Generate optimal tables from frequency counts
            let dc_table = dc_counter.generate_table().unwrap_or_else(|_| std_dc_luma());
            let ac_table = ac_counter.generate_table().unwrap_or_else(|_| std_ac_luma());
            (dc_table, ac_table)
        } else {
            // Use standard tables
            (std_dc_luma(), std_ac_luma())
        };

        // Convert to derived tables for entropy encoding
        let dc_dtbl = DerivedTable::from_huff_table(&dc_htbl, true)
            .expect("Failed to build DC derived table");
        let ac_dtbl = DerivedTable::from_huff_table(&ac_htbl, false)
            .expect("Failed to build AC derived table");

        // Build output buffer with JPEG markers
        let mut output = Vec::with_capacity(width * height);

        // SOI marker
        output.extend_from_slice(&[0xFF, 0xD8]);

        // APP0 (JFIF) marker
        self.write_app0(&mut output);

        // DQT markers
        self.write_dqt(&mut output, &quant_tables.luma);
        self.write_dqt(&mut output, &quant_tables.chroma);

        // SOF0 marker (baseline DCT)
        self.write_sof0(&mut output, width as u16, height as u16);

        // DHT markers
        self.write_dht(&mut output, 0x00, &dc_htbl); // DC table 0
        self.write_dht(&mut output, 0x10, &ac_htbl); // AC table 0

        // SOS marker
        self.write_sos(&mut output);

        // Encode blocks with derived tables
        let mut encoder = EntropyEncoder::new(dc_dtbl, ac_dtbl);
        for (i, coeffs) in all_coeffs.iter().enumerate() {
            let component = i % 3;
            encoder.encode_block(coeffs, component);
        }
        output.extend_from_slice(&encoder.finish());

        // EOI marker
        output.extend_from_slice(&[0xFF, 0xD9]);

        Ok(output)
    }

    /// Write DHT marker for a Huffman table
    fn write_dht(&self, output: &mut Vec<u8>, table_info: u8, table: &HuffTable) {
        // Count actual symbols
        let num_symbols: usize = table.bits[1..=16].iter().map(|&x| x as usize).sum();
        let len = 2 + 1 + 16 + num_symbols;
        output.extend_from_slice(&[0xFF, 0xC4]); // DHT marker
        output.extend_from_slice(&(len as u16).to_be_bytes());
        output.push(table_info); // table class and ID

        // Write bits[1..=16] (skip bits[0])
        output.extend_from_slice(&table.bits[1..=16]);
        // Write symbol values
        output.extend_from_slice(&table.huffval[..num_symbols]);
    }

    /// Encode grayscale plane to JPEG
    fn encode_gray_plane(
        &self,
        y_plane: &[u8],
        width: usize,
        height: usize,
        strategy: &SelectedStrategy,
    ) -> Result<Vec<u8>> {
        let q = self.quality.value() as u8;
        let quant_table = crate::quant::QuantTable::luma_standard(q);

        // Create standard AC derived table for trellis rate estimation
        let ac_derived = get_std_ac_derived();

        let width_blocks = (width + 7) / 8;
        let height_blocks = (height + 7) / 8;
        let total_blocks = width_blocks * height_blocks;

        // Quantize all blocks first
        let mut all_coeffs: Vec<[i16; 64]> = Vec::with_capacity(total_blocks);
        for by in 0..height_blocks {
            for bx in 0..width_blocks {
                let block = extract_block(y_plane, width, height, bx, by);
                let coeffs =
                    self.quantize_block_impl(&block, &quant_table.values, 1.0, strategy, &ac_derived);
                all_coeffs.push(coeffs);
            }
        }

        // Get Huffman tables (optimized or standard)
        let (dc_htbl, ac_htbl) = if self.optimize_huffman {
            let mut dc_counter = FrequencyCounter::new();
            let mut ac_counter = FrequencyCounter::new();
            let mut symbol_counter = SymbolCounter::new();

            for coeffs in &all_coeffs {
                symbol_counter.count_block(coeffs, 0, &mut dc_counter, &mut ac_counter);
            }

            let dc_table = dc_counter.generate_table().unwrap_or_else(|_| std_dc_luma());
            let ac_table = ac_counter.generate_table().unwrap_or_else(|_| std_ac_luma());
            (dc_table, ac_table)
        } else {
            (std_dc_luma(), std_ac_luma())
        };

        // Convert to derived tables for entropy encoding
        let dc_dtbl = DerivedTable::from_huff_table(&dc_htbl, true)
            .expect("Failed to build DC derived table");
        let ac_dtbl = DerivedTable::from_huff_table(&ac_htbl, false)
            .expect("Failed to build AC derived table");

        // Build output
        let mut output = Vec::with_capacity(width * height);

        output.extend_from_slice(&[0xFF, 0xD8]); // SOI
        self.write_app0(&mut output);
        self.write_dqt(&mut output, &quant_table);
        self.write_sof0_gray(&mut output, width as u16, height as u16);
        self.write_dht(&mut output, 0x00, &dc_htbl); // DC table 0
        self.write_dht(&mut output, 0x10, &ac_htbl); // AC table 0
        self.write_sos_gray(&mut output);

        // Encode all blocks
        let mut encoder = EntropyEncoder::new(dc_dtbl, ac_dtbl);
        for coeffs in &all_coeffs {
            encoder.encode_block(coeffs, 0);
        }
        output.extend_from_slice(&encoder.finish());

        output.extend_from_slice(&[0xFF, 0xD9]); // EOI

        Ok(output)
    }

    /// Write APP0 (JFIF) marker
    fn write_app0(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xE0]); // APP0
        output.extend_from_slice(&[0x00, 0x10]); // Length = 16
        output.extend_from_slice(b"JFIF\0"); // Identifier
        output.extend_from_slice(&[0x01, 0x01]); // Version 1.1
        output.push(0x00); // Units: none
        output.extend_from_slice(&[0x00, 0x01]); // X density
        output.extend_from_slice(&[0x00, 0x01]); // Y density
        output.extend_from_slice(&[0x00, 0x00]); // No thumbnail
    }

    /// Write DQT marker
    fn write_dqt(&self, output: &mut Vec<u8>, table: &crate::quant::QuantTable) {
        output.extend_from_slice(&[0xFF, 0xDB]); // DQT
        output.extend_from_slice(&[0x00, 0x43]); // Length = 67
        output.push(table.slot); // Table ID (8-bit precision)

        // Write values in zigzag order
        for i in 0..64 {
            let idx = crate::consts::ZIGZAG[i];
            output.push(table.values[idx] as u8);
        }
    }

    /// Write SOF0 marker (baseline, 3 components)
    fn write_sof0(&self, output: &mut Vec<u8>, width: u16, height: u16) {
        output.extend_from_slice(&[0xFF, 0xC0]); // SOF0
        output.extend_from_slice(&[0x00, 0x11]); // Length = 17
        output.push(0x08); // 8-bit precision
        output.extend_from_slice(&height.to_be_bytes());
        output.extend_from_slice(&width.to_be_bytes());
        output.push(0x03); // 3 components

        // Y component
        output.push(0x01); // Component ID
        output.push(0x11); // Sampling factors (1x1)
        output.push(0x00); // Quant table 0

        // Cb component
        output.push(0x02);
        output.push(0x11);
        output.push(0x01);

        // Cr component
        output.push(0x03);
        output.push(0x11);
        output.push(0x01);
    }

    /// Write SOF0 marker (baseline, 1 component for grayscale)
    fn write_sof0_gray(&self, output: &mut Vec<u8>, width: u16, height: u16) {
        output.extend_from_slice(&[0xFF, 0xC0]); // SOF0
        output.extend_from_slice(&[0x00, 0x0B]); // Length = 11
        output.push(0x08); // 8-bit precision
        output.extend_from_slice(&height.to_be_bytes());
        output.extend_from_slice(&width.to_be_bytes());
        output.push(0x01); // 1 component

        output.push(0x01); // Component ID
        output.push(0x11); // Sampling factors
        output.push(0x00); // Quant table 0
    }

    /// Write SOS marker (3 components)
    fn write_sos(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xDA]); // SOS
        output.extend_from_slice(&[0x00, 0x0C]); // Length = 12
        output.push(0x03); // 3 components

        output.push(0x01); // Y: component 1
        output.push(0x00); // Y: DC table 0, AC table 0

        output.push(0x02); // Cb: component 2
        output.push(0x00); // Cb: DC table 0, AC table 0

        output.push(0x03); // Cr: component 3
        output.push(0x00); // Cr: DC table 0, AC table 0

        output.push(0x00); // Ss
        output.push(0x3F); // Se
        output.push(0x00); // Ah/Al
    }

    /// Write SOS marker (1 component for grayscale)
    fn write_sos_gray(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xDA]);
        output.extend_from_slice(&[0x00, 0x08]); // Length = 8
        output.push(0x01); // 1 component

        output.push(0x01); // Y component
        output.push(0x00); // DC/AC table 0

        output.push(0x00); // Ss
        output.push(0x3F); // Se
        output.push(0x00); // Ah/Al
    }

    /// Quantize a single 8x8 block (DCT + quantization)
    ///
    /// Uses integer Loeffler DCT for better precision and compression.
    fn quantize_block_impl(
        &self,
        block: &[u8; 64],
        quant: &[u16; 64],
        aq_mult: f32,
        strategy: &SelectedStrategy,
        ac_derived: &DerivedTable,
    ) -> [i16; 64] {
        // Apply AQ to quantization table
        let effective_quant = if aq_mult != 1.0 {
            crate::adaptive_quant::apply_aq_to_quant(quant, aq_mult)
        } else {
            *quant
        };

        // Level shift and integer DCT (output is scaled by 8)
        let shifted = level_shift(block);
        let mut dct_coeffs = [0i16; 64];
        forward_dct_8x8_int(&shifted, &mut dct_coeffs);

        // Quantize (with optional trellis)
        if strategy.trellis.ac_enabled {
            // Trellis path: pass 8x-scaled DCT directly
            // Trellis internally multiplies qtable by 8 to match
            let mut dct_i32 = [0i32; 64];
            scale_for_trellis(&dct_coeffs, &mut dct_i32);

            let mut quantized = [0i16; 64];
            trellis_quantize_block(
                &dct_i32,
                &mut quantized,
                &effective_quant,
                ac_derived,
                &strategy.trellis,
            );
            quantized
        } else {
            // Non-trellis path: descale DCT first, then quantize
            let mut dct_descaled = [0i32; 64];
            descale_for_quant(&dct_coeffs, &mut dct_descaled);

            let mut quantized = [0i16; 64];
            quantize_block_int(&dct_descaled, &effective_quant, &mut quantized);
            quantized
        }
    }
}

/// Extract an 8x8 block from a plane, with edge padding
fn extract_block(plane: &[u8], width: usize, height: usize, bx: usize, by: usize) -> [u8; 64] {
    let mut block = [0u8; 64];
    let start_x = bx * 8;
    let start_y = by * 8;

    for dy in 0..8 {
        let y = (start_y + dy).min(height - 1);
        for dx in 0..8 {
            let x = (start_x + dx).min(width - 1);
            block[dy * 8 + dx] = plane[y * width + x];
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = Encoder::new();
        assert!(matches!(encoder.quality, Quality::Standard(85)));
    }

    #[test]
    fn test_encode_small_gray() {
        let encoder = Encoder::new().quality(Quality::Standard(75));
        let pixels = vec![128u8; 16 * 16]; // 16x16 gray image
        let result = encoder.encode_gray(&pixels, 16, 16);
        assert!(result.is_ok());

        let jpeg = result.unwrap();
        // Should start with SOI
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        // Should end with EOI
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn test_encode_small_rgb() {
        let encoder = Encoder::new().quality(Quality::Standard(75));
        let pixels = vec![128u8; 16 * 16 * 3]; // 16x16 RGB
        let result = encoder.encode_rgb(&pixels, 16, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_dimensions() {
        let encoder = Encoder::new();
        let pixels = vec![0u8; 0];
        let result = encoder.encode_rgb(&pixels, 0, 0);
        assert!(result.is_err());
    }
}
