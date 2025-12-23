//! Entropy coding for JPEG.
//!
//! This module provides Huffman-based entropy encoding and decoding
//! for JPEG DCT coefficients.

use crate::bitstream::{BitReader, BitWriter};
use crate::consts::DCT_BLOCK_SIZE;
use crate::error::{Error, Result};
use crate::huffman::{HuffmanDecodeTable, HuffmanEncodeTable};

/// Maximum DC coefficient difference magnitude (for 8-bit samples).
pub const MAX_DC_DIFF: i16 = 2047;

/// Maximum AC coefficient magnitude (for 8-bit samples).
pub const MAX_AC_COEFF: i16 = 1023;

/// Returns the category (number of bits needed) for a value.
#[inline]
#[must_use]
pub fn category(value: i16) -> u8 {
    if value == 0 {
        return 0;
    }
    let abs_val = value.unsigned_abs();
    16 - abs_val.leading_zeros() as u8
}

/// Returns the additional bits for a value in its category.
#[inline]
#[must_use]
pub fn additional_bits(value: i16) -> u16 {
    if value >= 0 {
        value as u16
    } else {
        // For negative values, encode as (value - 1) in one's complement
        (value - 1) as u16 & ((1u16 << category(value)) - 1)
    }
}

/// Reconstructs a value from category and additional bits.
#[inline]
#[must_use]
pub fn decode_value(category: u8, bits: u16) -> i16 {
    if category == 0 {
        return 0;
    }

    let half = 1u16 << (category - 1);
    if bits >= half {
        bits as i16
    } else {
        (bits as i16) - ((1i16 << category) - 1)
    }
}

/// Entropy encoder for a single scan.
pub struct EntropyEncoder {
    /// Bit writer
    writer: BitWriter,
    /// DC Huffman tables (indexed by table selector)
    dc_tables: [Option<HuffmanEncodeTable>; 4],
    /// AC Huffman tables (indexed by table selector)
    ac_tables: [Option<HuffmanEncodeTable>; 4],
    /// Previous DC values for each component
    prev_dc: [i16; 4],
    /// Restart interval counter
    restart_counter: u16,
    /// Restart interval
    restart_interval: u16,
    /// Current restart marker number (0-7)
    restart_num: u8,
}

impl EntropyEncoder {
    /// Creates a new entropy encoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            writer: BitWriter::new(),
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            prev_dc: [0; 4],
            restart_counter: 0,
            restart_interval: 0,
            restart_num: 0,
        }
    }

    /// Sets a DC Huffman table.
    pub fn set_dc_table(&mut self, idx: usize, table: HuffmanEncodeTable) {
        if idx < 4 {
            self.dc_tables[idx] = Some(table);
        }
    }

    /// Sets an AC Huffman table.
    pub fn set_ac_table(&mut self, idx: usize, table: HuffmanEncodeTable) {
        if idx < 4 {
            self.ac_tables[idx] = Some(table);
        }
    }

    /// Sets the restart interval.
    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = interval;
        self.restart_counter = interval;
    }

    /// Resets DC prediction (for restart markers).
    pub fn reset_dc(&mut self) {
        self.prev_dc = [0; 4];
    }

    /// Encodes a block of DCT coefficients.
    ///
    /// # Arguments
    /// * `coeffs` - Quantized DCT coefficients in zigzag order
    /// * `component` - Component index (for DC prediction)
    /// * `dc_table_idx` - DC Huffman table index
    /// * `ac_table_idx` - AC Huffman table index
    pub fn encode_block(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> Result<()> {
        let dc_table = self.dc_tables[dc_table_idx]
            .as_ref()
            .ok_or(Error::InternalError {
                reason: "DC table not set",
            })?;
        let ac_table = self.ac_tables[ac_table_idx]
            .as_ref()
            .ok_or(Error::InternalError {
                reason: "AC table not set",
            })?;

        // Encode DC coefficient
        let dc = coeffs[0];
        let dc_diff = dc - self.prev_dc[component];
        self.prev_dc[component] = dc;

        let dc_cat = category(dc_diff);
        let (code, len) = dc_table.encode(dc_cat);
        self.writer.write_bits(code, len);

        if dc_cat > 0 {
            let additional = additional_bits(dc_diff);
            self.writer.write_bits(additional as u32, dc_cat);
        }

        // Encode AC coefficients
        let mut run = 0u8;
        for i in 1..DCT_BLOCK_SIZE {
            let ac = coeffs[i];

            if ac == 0 {
                run += 1;
            } else {
                // Encode any runs of 16 zeros
                while run >= 16 {
                    let (code, len) = ac_table.encode(0xF0); // ZRL
                    self.writer.write_bits(code, len);
                    run -= 16;
                }

                // Encode run/size and value
                let ac_cat = category(ac);
                let symbol = (run << 4) | ac_cat;
                let (code, len) = ac_table.encode(symbol);
                self.writer.write_bits(code, len);

                let additional = additional_bits(ac);
                self.writer.write_bits(additional as u32, ac_cat);

                run = 0;
            }
        }

        // If we have trailing zeros, encode EOB
        if run > 0 {
            let (code, len) = ac_table.encode(0x00); // EOB
            self.writer.write_bits(code, len);
        }

        Ok(())
    }

    /// Handles restart marker if needed.
    pub fn check_restart(&mut self) {
        if self.restart_interval > 0 {
            self.restart_counter -= 1;
            if self.restart_counter == 0 {
                self.writer.flush();
                self.writer.write_byte_raw(0xFF);
                self.writer.write_byte_raw(0xD0 + self.restart_num);
                self.restart_num = (self.restart_num + 1) & 7;
                self.restart_counter = self.restart_interval;
                self.reset_dc();
            }
        }
    }

    /// Finishes encoding and returns the bitstream.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.writer.into_bytes()
    }
}

impl Default for EntropyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy decoder for a single scan.
pub struct EntropyDecoder<'a> {
    /// Bit reader
    reader: BitReader<'a>,
    /// DC Huffman tables
    dc_tables: [Option<HuffmanDecodeTable>; 4],
    /// AC Huffman tables
    ac_tables: [Option<HuffmanDecodeTable>; 4],
    /// Previous DC values for each component
    prev_dc: [i16; 4],
}

impl<'a> EntropyDecoder<'a> {
    /// Creates a new entropy decoder.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            prev_dc: [0; 4],
        }
    }

    /// Sets a DC Huffman table.
    pub fn set_dc_table(&mut self, idx: usize, table: HuffmanDecodeTable) {
        if idx < 4 {
            self.dc_tables[idx] = Some(table);
        }
    }

    /// Sets an AC Huffman table.
    pub fn set_ac_table(&mut self, idx: usize, table: HuffmanDecodeTable) {
        if idx < 4 {
            self.ac_tables[idx] = Some(table);
        }
    }

    /// Resets DC prediction.
    pub fn reset_dc(&mut self) {
        self.prev_dc = [0; 4];
    }

    /// Decodes a Huffman symbol.
    fn decode_huffman(&mut self, table: &HuffmanDecodeTable) -> Result<u8> {
        // Try fast lookup first
        match self.reader.peek_bits(HuffmanDecodeTable::FAST_BITS as u8) {
            Ok(bits) => {
                // fast_decode expects bits in MSB position (shifted left by 32 - FAST_BITS)
                let shifted = (bits as u32) << (32 - HuffmanDecodeTable::FAST_BITS);
                if let Some((symbol, len)) = table.fast_decode(shifted) {
                    self.reader.skip_bits(len);
                    return Ok(symbol);
                }
            }
            Err(_) => {
                // Not enough bits for fast lookup, try slow path
            }
        }

        // Slow path for longer codes
        let mut code = 0u32;
        for len in 1..=16 {
            code = (code << 1) | self.reader.read_bits(1)?;
            if (code as i32) <= table.maxcode[len] {
                let idx = (code as i32 + table.valoffset[len]) as usize;
                if idx < table.values.len() {
                    return Ok(table.values[idx]);
                }
            }
        }

        Err(Error::InvalidHuffmanTable {
            table_idx: 0,
            reason: "invalid code",
        })
    }

    /// Decodes a block of DCT coefficients.
    pub fn decode_block(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> Result<[i16; DCT_BLOCK_SIZE]> {
        // Clone tables to avoid borrow conflicts with self.decode_huffman()
        let dc_table = self.dc_tables[dc_table_idx]
            .clone()
            .ok_or(Error::InternalError {
                reason: "DC table not set",
            })?;
        let ac_table = self.ac_tables[ac_table_idx]
            .clone()
            .ok_or(Error::InternalError {
                reason: "AC table not set",
            })?;

        let mut coeffs = [0i16; DCT_BLOCK_SIZE];

        // Decode DC coefficient
        let dc_cat = self.decode_huffman(&dc_table)?;
        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = self.reader.read_bits(dc_cat)? as u16;
            decode_value(dc_cat, bits)
        };

        coeffs[0] = self.prev_dc[component] + dc_diff;
        self.prev_dc[component] = coeffs[0];

        // Decode AC coefficients
        let mut i = 1;
        while i < DCT_BLOCK_SIZE {
            let symbol = self.decode_huffman(&ac_table)?;

            if symbol == 0 {
                // EOB - remaining coefficients are zero
                break;
            }

            let run = symbol >> 4;
            let ac_cat = symbol & 0x0F;

            if ac_cat == 0 {
                if run == 15 {
                    // ZRL - skip 16 zeros
                    i += 16;
                } else {
                    // Invalid symbol
                    break;
                }
            } else {
                i += run as usize;
                if i >= DCT_BLOCK_SIZE {
                    return Err(Error::InvalidJpegData {
                        reason: "AC coefficient index out of bounds",
                    });
                }

                let bits = self.reader.read_bits(ac_cat)? as u16;
                coeffs[i] = decode_value(ac_cat, bits);
                i += 1;
            }
        }

        Ok(coeffs)
    }

    /// Returns the underlying bit reader position.
    pub fn position(&self) -> usize {
        self.reader.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category() {
        assert_eq!(category(0), 0);
        assert_eq!(category(1), 1);
        assert_eq!(category(-1), 1);
        assert_eq!(category(2), 2);
        assert_eq!(category(-2), 2);
        assert_eq!(category(3), 2);
        assert_eq!(category(-3), 2);
        assert_eq!(category(4), 3);
        assert_eq!(category(7), 3);
        assert_eq!(category(255), 8);
        assert_eq!(category(-255), 8);
    }

    #[test]
    fn test_value_roundtrip() {
        for value in -1023i16..=1023 {
            let cat = category(value);
            let bits = additional_bits(value);
            let recovered = decode_value(cat, bits);
            assert_eq!(value, recovered, "Failed for {}", value);
        }
    }

    #[test]
    fn test_additional_bits() {
        // Positive values: additional bits are the value itself
        assert_eq!(additional_bits(1), 1);
        assert_eq!(additional_bits(2), 2);
        assert_eq!(additional_bits(3), 3);

        // Negative values: one's complement within category
        assert_eq!(additional_bits(-1), 0);
        assert_eq!(additional_bits(-2), 1);
        assert_eq!(additional_bits(-3), 0);
    }
}
