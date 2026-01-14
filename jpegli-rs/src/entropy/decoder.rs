//! Entropy decoder for JPEG.
//!
//! Provides `EntropyDecoder` for baseline and progressive JPEG decoding.

#![allow(dead_code)]

use crate::error::{Error, Result};
use crate::foundation::bitstream::BitReader;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::huffman::HuffmanDecodeTable;

use super::decode_value;

/// Decodes a Huffman symbol from the bit reader using the provided table.
/// This is a standalone function to avoid borrow conflicts in decode_block.
#[inline]
fn decode_huffman_symbol(reader: &mut BitReader, table: &HuffmanDecodeTable) -> Result<u8> {
    // Try fast lookup first (most common path)
    if let Some(bits) = reader.peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8) {
        let lookup = table.fast_lookup[bits as usize];
        if lookup >= 0 {
            let symbol = (lookup & 0xFF) as u8;
            let len = (lookup >> 8) as u8;
            reader.skip_bits_fast(len);
            return Ok(symbol);
        }
    }

    // Slow path for longer codes
    let mut code = 0u32;
    for len in 1..=16 {
        let bit = reader.read_bits(1)?;
        code = (code << 1) | bit;
        if (code as i32) <= table.maxcode[len] {
            let idx = (code as i32 + table.valoffset[len]) as usize;
            if idx < table.values.len() {
                return Ok(table.values[idx]);
            }
        }
    }

    // If we've exhausted real data (hit marker or past end), treat invalid code as end of scan.
    // This happens when fill bits at end of scan don't form a valid Huffman code.
    if reader.is_exhausted() {
        return Err(Error::EndOfScanData);
    }

    Err(Error::InvalidHuffmanTable {
        table_idx: 0,
        reason: "invalid code",
    })
}

/// Entropy decoder for a single scan.
///
/// Uses borrowed Huffman tables to avoid cloning ~1.5KB per table.
pub struct EntropyDecoder<'data, 'tables> {
    /// Bit reader
    reader: BitReader<'data>,
    /// DC Huffman tables (borrowed)
    dc_tables: [Option<&'tables HuffmanDecodeTable>; 4],
    /// AC Huffman tables (borrowed)
    ac_tables: [Option<&'tables HuffmanDecodeTable>; 4],
    /// Previous DC values for each component
    prev_dc: [i16; 4],
}

/// Saved state of an EntropyDecoder for speculative decoding.
#[derive(Clone, Copy)]
pub struct EntropyDecoderState {
    reader_state: crate::foundation::bitstream::BitReaderState,
    prev_dc: [i16; 4],
}

impl<'data, 'tables> EntropyDecoder<'data, 'tables> {
    /// Creates a new entropy decoder.
    pub fn new(data: &'data [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            prev_dc: [0; 4],
        }
    }

    /// Sets a DC Huffman table (borrowed, not cloned).
    pub fn set_dc_table(&mut self, idx: usize, table: &'tables HuffmanDecodeTable) {
        if idx < 4 {
            self.dc_tables[idx] = Some(table);
        }
    }

    /// Sets an AC Huffman table (borrowed, not cloned).
    pub fn set_ac_table(&mut self, idx: usize, table: &'tables HuffmanDecodeTable) {
        if idx < 4 {
            self.ac_tables[idx] = Some(table);
        }
    }

    /// Resets DC prediction.
    pub fn reset_dc(&mut self) {
        self.prev_dc = [0; 4];
    }

    /// Gets the current DC predictor values.
    pub fn get_prev_dc(&self) -> [i16; 4] {
        self.prev_dc
    }

    /// Sets the DC predictor values (for resuming decode).
    pub fn set_prev_dc(&mut self, prev_dc: &[i16; 4]) {
        self.prev_dc = *prev_dc;
    }

    /// Decodes a Huffman symbol.
    #[inline]
    fn decode_huffman(&mut self, table: &HuffmanDecodeTable) -> Result<u8> {
        // Try fast lookup first
        if let Some(bits) = self
            .reader
            .peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8)
        {
            let lookup = table.fast_lookup[bits as usize];
            if lookup >= 0 {
                let symbol = (lookup & 0xFF) as u8;
                let len = (lookup >> 8) as u8;
                self.reader.skip_bits_fast(len);
                return Ok(symbol);
            }
        }

        // Slow path for longer codes
        let mut code = 0u32;
        for len in 1..=16 {
            match self.reader.read_bits(1) {
                Ok(bit) => {
                    code = (code << 1) | bit;
                    if (code as i32) <= table.maxcode[len] {
                        let idx = (code as i32 + table.valoffset[len]) as usize;
                        if idx < table.values.len() {
                            return Ok(table.values[idx]);
                        }
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        // If we've exhausted real data (hit marker or past end), treat invalid code as end of scan.
        // This happens when fill bits at end of scan don't form a valid Huffman code.
        if self.reader.is_exhausted() {
            return Err(Error::EndOfScanData);
        }

        Err(Error::InvalidHuffmanTable {
            table_idx: 0,
            reason: "invalid code",
        })
    }

    /// Safely gets a DC table reference, handling out-of-bounds indices.
    /// Returns with 'tables lifetime to avoid borrowing self.
    fn get_dc_table(&self, idx: usize) -> Result<&'tables HuffmanDecodeTable> {
        self.dc_tables
            .get(idx)
            .and_then(|&t| t)
            .ok_or(Error::InternalError {
                reason: "DC table not set or invalid index",
            })
    }

    /// Safely gets an AC table reference, handling out-of-bounds indices.
    /// Returns with 'tables lifetime to avoid borrowing self.
    fn get_ac_table(&self, idx: usize) -> Result<&'tables HuffmanDecodeTable> {
        self.ac_tables
            .get(idx)
            .and_then(|&t| t)
            .ok_or(Error::InternalError {
                reason: "AC table not set or invalid index",
            })
    }

    /// Decodes a block of DCT coefficients with fast AC path.
    pub fn decode_block(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> Result<[i16; DCT_BLOCK_SIZE]> {
        // Get table references once (tables are borrowed, no copying)
        let dc_table = self.dc_tables[dc_table_idx].ok_or(Error::InternalError {
            reason: "DC table not set",
        })?;
        let ac_table = self.ac_tables[ac_table_idx].ok_or(Error::InternalError {
            reason: "AC table not set",
        })?;

        let mut coeffs = [0i16; DCT_BLOCK_SIZE];

        // Decode DC coefficient using standalone function
        let dc_cat = decode_huffman_symbol(&mut self.reader, dc_table)?;

        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = self.reader.read_bits(dc_cat)? as u16;
            decode_value(dc_cat, bits)
        };

        // Use wrapping_add to handle malformed input gracefully without panicking
        coeffs[0] = self.prev_dc[component].wrapping_add(dc_diff);
        self.prev_dc[component] = coeffs[0];

        // Decode AC coefficients with fast path
        let mut i = 1;
        while i < DCT_BLOCK_SIZE {
            // Try fast path first - peek 9 bits with inline refill
            if let Some(bits9) = self
                .reader
                .peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8)
            {
                let idx = bits9 as usize;

                // Try fast AC decode first (combined Huffman + sign extend)
                if let Some((value, run, total_bits)) = ac_table.fast_decode_ac(idx) {
                    self.reader.skip_bits_fast(total_bits);
                    i += run as usize;
                    if i < DCT_BLOCK_SIZE {
                        coeffs[i] = value;
                        i += 1;
                    }
                    continue;
                }

                // Try regular fast Huffman lookup
                let lookup = ac_table.fast_lookup[idx];
                if lookup >= 0 {
                    let symbol = (lookup & 0xFF) as u8;
                    let code_length = (lookup >> 8) as u8;
                    self.reader.skip_bits_fast(code_length);

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
                    continue;
                }
            }

            // Slow path for long codes or when not enough bits
            let symbol = decode_huffman_symbol(&mut self.reader, ac_table)?;

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

    /// Decode a single 8x8 block of DCT coefficients, returning coefficient count.
    ///
    /// Returns `(coefficients, coeff_count)` where `coeff_count` is the position
    /// of the last non-zero coefficient in zigzag order (1-64). This enables
    /// tiered IDCT optimization:
    /// - count <= 1: DC-only block
    /// - count <= 10: Use 4x4 IDCT
    /// - count > 10: Use full 8x8 IDCT
    pub fn decode_block_with_count(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> Result<([i16; DCT_BLOCK_SIZE], u8)> {
        // Get table references once (tables are borrowed, no copying)
        let dc_table = self.dc_tables[dc_table_idx].ok_or(Error::InternalError {
            reason: "DC table not set",
        })?;
        let ac_table = self.ac_tables[ac_table_idx].ok_or(Error::InternalError {
            reason: "AC table not set",
        })?;

        let mut coeffs = [0i16; DCT_BLOCK_SIZE];

        // Decode DC coefficient using standalone function
        let dc_cat = decode_huffman_symbol(&mut self.reader, dc_table)?;

        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = self.reader.read_bits(dc_cat)? as u16;
            decode_value(dc_cat, bits)
        };

        // Use wrapping_add to handle malformed input gracefully without panicking
        coeffs[0] = self.prev_dc[component].wrapping_add(dc_diff);
        self.prev_dc[component] = coeffs[0];

        // Track the last non-zero position for tiered IDCT
        let mut last_nonzero: u8 = 1; // At minimum we have DC

        // Decode AC coefficients with fast path
        let mut i = 1;
        while i < DCT_BLOCK_SIZE {
            // Try fast path first - peek 9 bits with inline refill
            if let Some(bits9) = self
                .reader
                .peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8)
            {
                let idx = bits9 as usize;

                // Try fast AC decode first (combined Huffman + sign extend)
                if let Some((value, run, total_bits)) = ac_table.fast_decode_ac(idx) {
                    self.reader.skip_bits_fast(total_bits);
                    i += run as usize;
                    if i < DCT_BLOCK_SIZE {
                        coeffs[i] = value;
                        last_nonzero = (i + 1) as u8;
                        i += 1;
                    }
                    continue;
                }

                // Try regular fast Huffman lookup
                let lookup = ac_table.fast_lookup[idx];
                if lookup >= 0 {
                    let symbol = (lookup & 0xFF) as u8;
                    let code_length = (lookup >> 8) as u8;
                    self.reader.skip_bits_fast(code_length);

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
                        last_nonzero = (i + 1) as u8;
                        i += 1;
                    }
                    continue;
                }
            }

            // Slow path for long codes or when not enough bits
            let symbol = decode_huffman_symbol(&mut self.reader, ac_table)?;

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
                last_nonzero = (i + 1) as u8;
                i += 1;
            }
        }

        Ok((coeffs, last_nonzero))
    }

    /// Returns the underlying bit reader position.
    pub fn position(&self) -> usize {
        self.reader.position()
    }

    /// Aligns to byte boundary (call before reading restart marker).
    pub fn align_to_byte(&mut self) {
        self.reader.align_to_byte();
    }

    /// Saves the current decoder state for potential rollback.
    #[must_use]
    pub fn save_state(&self) -> EntropyDecoderState {
        EntropyDecoderState {
            reader_state: self.reader.save_state(),
            prev_dc: self.prev_dc,
        }
    }

    /// Restores a previously saved state.
    pub fn restore_state(&mut self, state: EntropyDecoderState) {
        self.reader.restore_state(state.reader_state);
        self.prev_dc = state.prev_dc;
    }

    /// Reads and verifies a restart marker.
    ///
    /// Call this after aligning to byte boundary when a restart marker is expected.
    /// Returns Ok(()) if the expected marker was found, Err otherwise.
    ///
    /// # Arguments
    /// * `expected_num` - Expected restart marker number (0-7)
    pub fn read_restart_marker(&mut self, expected_num: u8) -> Result<()> {
        self.reader.read_restart_marker(expected_num)
    }

    // ===== Progressive decoding methods =====

    /// Decodes DC coefficient for progressive first scan (ah=0).
    /// Returns the shifted DC difference.
    pub fn decode_dc_first(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        al: u8,
    ) -> Result<i16> {
        let dc_table = self.get_dc_table(dc_table_idx)?;

        let dc_cat = self.decode_huffman(dc_table)?;
        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = self.reader.read_bits(dc_cat)? as u16;
            decode_value(dc_cat, bits)
        };

        let shifted_dc = self.prev_dc[component].wrapping_add(dc_diff);
        self.prev_dc[component] = shifted_dc;

        // Return the unshifted value (shift left by al)
        Ok(shifted_dc << al)
    }

    /// Decodes DC refinement bit (ah>0).
    /// Returns the bit to add at position al.
    pub fn decode_dc_refine(&mut self, al: u8) -> Result<i16> {
        let bit = self.reader.read_bits(1)? as i16;
        Ok(bit << al)
    }

    /// Decodes AC coefficients for progressive first scan (ah=0).
    /// Writes coefficients to the provided slice in range [ss, se].
    /// Returns the EOB run remaining after this block.
    pub fn decode_ac_first(
        &mut self,
        coeffs: &mut [i16; DCT_BLOCK_SIZE],
        ac_table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        eob_run: &mut u16,
    ) -> Result<()> {
        let ac_table = self.get_ac_table(ac_table_idx)?;

        // If we have a pending EOB run, decrement and skip this block
        if *eob_run > 0 {
            *eob_run -= 1;
            return Ok(());
        }

        let mut k = ss as usize;
        while k <= se as usize {
            let symbol = self.decode_huffman(ac_table)?;
            let run = symbol >> 4;
            let size = symbol & 0x0F;

            if size == 0 {
                if run == 15 {
                    // ZRL - skip 16 zeros
                    k += 16;
                } else {
                    // EOB run
                    // run=0 means EOB for this block only
                    // run=1-14 means 2^run + extra bits count of EOBs
                    if run == 0 {
                        // Single EOB, we're done with this block
                        return Ok(());
                    } else {
                        // EOB run: 2^run + extra_bits
                        let extra = self.reader.read_bits(run)? as u16;
                        *eob_run = (1 << run) + extra - 1; // -1 because this block counts as one
                        return Ok(());
                    }
                }
            } else {
                k += run as usize;
                if k > se as usize {
                    return Err(Error::InvalidJpegData {
                        reason: "AC coefficient index out of bounds",
                    });
                }

                let bits = self.reader.read_bits(size)? as u16;
                let value = decode_value(size, bits);
                coeffs[k] = value << al;
                k += 1;
            }
        }

        Ok(())
    }

    /// Decodes AC refinement for progressive scan (ah>0).
    /// Updates coefficients in range [ss, se].
    pub fn decode_ac_refine(
        &mut self,
        coeffs: &mut [i16; DCT_BLOCK_SIZE],
        ac_table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        eob_run: &mut u16,
    ) -> Result<()> {
        let ac_table = self.get_ac_table(ac_table_idx)?;
        let bit_val = 1i16 << al;

        // If we have a pending EOB run, apply refinement bits to nonzero coeffs and return
        if *eob_run > 0 {
            for k in ss as usize..=se as usize {
                if coeffs[k] != 0 {
                    let bit = self.reader.read_bits(1)? as i16;
                    if bit != 0 && (coeffs[k] & bit_val) == 0 {
                        // Use saturating arithmetic to prevent overflow on malformed input
                        if coeffs[k] > 0 {
                            coeffs[k] = coeffs[k].saturating_add(bit_val);
                        } else {
                            coeffs[k] = coeffs[k].saturating_sub(bit_val);
                        }
                    }
                }
            }
            *eob_run -= 1;
            return Ok(());
        }

        let mut k = ss as usize;
        while k <= se as usize {
            let symbol = self.decode_huffman(ac_table)?;
            let run = symbol >> 4;
            let size = symbol & 0x0F;

            let mut num_zeros_to_skip = run as usize;

            if size == 0 {
                if run == 15 {
                    // ZRL in refinement - skip 16 zeros (not 15!)
                    // The run nibble is 15, but ZRL means 16 zeros.
                    // We need to add 1 to account for this.
                    num_zeros_to_skip = 16;
                } else {
                    // EOB run
                    if run == 0 {
                        // Single EOB - apply refinement to remaining nonzero coeffs
                        for j in k..=se as usize {
                            if coeffs[j] != 0 {
                                let bit = self.reader.read_bits(1)? as i16;
                                if bit != 0 && (coeffs[j] & bit_val) == 0 {
                                    // Use saturating arithmetic to prevent overflow on malformed input
                                    if coeffs[j] > 0 {
                                        coeffs[j] = coeffs[j].saturating_add(bit_val);
                                    } else {
                                        coeffs[j] = coeffs[j].saturating_sub(bit_val);
                                    }
                                }
                            }
                        }
                        return Ok(());
                    } else {
                        // EOB run
                        let extra = self.reader.read_bits(run)? as u16;
                        *eob_run = (1 << run) + extra - 1;
                        // Apply refinement to remaining nonzero coeffs in this block
                        for j in k..=se as usize {
                            if coeffs[j] != 0 {
                                let bit = self.reader.read_bits(1)? as i16;
                                if bit != 0 && (coeffs[j] & bit_val) == 0 {
                                    // Use saturating arithmetic to prevent overflow on malformed input
                                    if coeffs[j] > 0 {
                                        coeffs[j] = coeffs[j].saturating_add(bit_val);
                                    } else {
                                        coeffs[j] = coeffs[j].saturating_sub(bit_val);
                                    }
                                }
                            }
                        }
                        return Ok(());
                    }
                }
            }

            // For NEW_NZ (size=1), read sign bit FIRST, before refinement bits
            // This matches the JPEG spec bit order: [Huffman] [sign] [refinement bits]
            let new_val = if size != 0 {
                let sign_bit = self.reader.read_bits(1)? as i16;
                Some(if sign_bit != 0 { bit_val } else { -bit_val })
            } else {
                None
            };

            // Skip zeros and apply refinement bits to nonzero coefficients
            while k <= se as usize {
                // For ZRL (size=0), stop immediately after skipping all 16 zeros.
                // Don't continue reading refinement bits for subsequent nonzeros -
                // those belong to the next symbol.
                if size == 0 && num_zeros_to_skip == 0 {
                    break;
                }

                if coeffs[k] != 0 {
                    // Apply refinement bit for previously-nonzero coefficient
                    let bit = self.reader.read_bits(1)? as i16;
                    if bit != 0 && (coeffs[k] & bit_val) == 0 {
                        // Use saturating arithmetic to prevent overflow on malformed input
                        if coeffs[k] > 0 {
                            coeffs[k] = coeffs[k].saturating_add(bit_val);
                        } else {
                            coeffs[k] = coeffs[k].saturating_sub(bit_val);
                        }
                    }
                } else if num_zeros_to_skip > 0 {
                    num_zeros_to_skip -= 1;
                } else {
                    // Found our target position (for NEW_NZ symbols)
                    break;
                }
                k += 1;
            }

            if let Some(val) = new_val {
                if k <= se as usize {
                    // Place newly-nonzero coefficient
                    coeffs[k] = val;
                    k += 1; // Move past the placed coefficient
                }
            }
            // For ZRL (size==0), k already points past the 16 zeros we skipped
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::{HuffmanDecodeTable, HuffmanEncodeTable};

    #[test]
    fn test_entropy_decoder_new() {
        let data = [0u8; 10];
        let decoder = EntropyDecoder::new(&data);
        assert_eq!(decoder.prev_dc, [0; 4]);
    }

    #[test]
    fn test_entropy_decoder_set_tables() {
        let data = [0u8; 10];
        let mut decoder = EntropyDecoder::new(&data);

        // Create decode tables from JPEG standard luminance DC table
        let bits: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let values = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let dc_table = HuffmanDecodeTable::from_bits_values(&bits, &values).unwrap();
        decoder.set_dc_table(0, &dc_table);
        assert!(decoder.dc_tables[0].is_some());

        let ac_bits: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
        let ac_values: Vec<u8> = (0..162).collect();
        let ac_table = HuffmanDecodeTable::from_bits_values(&ac_bits, &ac_values).unwrap();
        decoder.set_ac_table(0, &ac_table);
        assert!(decoder.ac_tables[0].is_some());

        // Test out of range indices (should be no-op)
        decoder.set_dc_table(5, &dc_table);
        decoder.set_ac_table(5, &ac_table);
    }

    #[test]
    fn test_entropy_decoder_reset_dc() {
        let data = [0u8; 10];
        let mut decoder = EntropyDecoder::new(&data);
        decoder.prev_dc = [10, 20, 30, 40];
        decoder.reset_dc();
        assert_eq!(decoder.prev_dc, [0; 4]);
    }

    #[test]
    fn test_entropy_decoder_position() {
        let data = [0u8; 10];
        let decoder = EntropyDecoder::new(&data);
        assert_eq!(decoder.position(), 0);
    }
}
