//! Entropy decoder for JPEG.
//!
//! Provides `EntropyDecoder` for baseline and progressive JPEG decoding.

#![allow(dead_code)]

use crate::error::{Error, Result, ScanRead, ScanResult};
use crate::foundation::bitstream::BitReader;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::huffman::HuffmanDecodeTable;

use super::decode_value;

/// Result of lenient Huffman decode - includes flag for invalid code recovery.
pub(crate) enum HuffmanResult {
    /// Normal symbol decoded.
    Symbol(u8),
    /// End of scan (marker found).
    EndOfScan,
    /// Truncated data.
    Truncated,
    /// Invalid code recovered as EOB (lenient mode only).
    InvalidCodeRecovered,
}

/// Decodes a Huffman symbol from the bit reader using the provided table.
/// This is a standalone function to avoid borrow conflicts in decode_block.
#[inline(always)]
fn decode_huffman_symbol(reader: &mut BitReader, table: &HuffmanDecodeTable) -> ScanResult<u8> {
    decode_huffman_symbol_lenient(reader, table, false).map(|r| match r {
        HuffmanResult::Symbol(s) => ScanRead::Value(s),
        HuffmanResult::EndOfScan => ScanRead::EndOfScan,
        HuffmanResult::Truncated => ScanRead::Truncated,
        HuffmanResult::InvalidCodeRecovered => ScanRead::EndOfScan, // Shouldn't happen without lenient
    })
}

/// Decodes a Huffman symbol with optional lenient mode.
/// In lenient mode, invalid codes are treated as EOB (symbol 0x00).
#[inline(always)]
fn decode_huffman_symbol_lenient(
    reader: &mut BitReader,
    table: &HuffmanDecodeTable,
    lenient: bool,
) -> Result<HuffmanResult> {
    // Try fast lookup first (most common path)
    if let Some(bits) = reader.peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8) {
        let lookup = table.fast_lookup[bits as usize];
        if lookup >= 0 {
            let symbol = (lookup & 0xFF) as u8;
            let len = (lookup >> 8) as u8;
            reader.skip_bits_fast(len);
            return Ok(HuffmanResult::Symbol(symbol));
        }
    }

    // Slow path for longer codes
    let mut code = 0u32;
    for len in 1..=16 {
        let bit = match reader.read_bits(1)? {
            ScanRead::Value(b) => b,
            ScanRead::EndOfScan => return Ok(HuffmanResult::EndOfScan),
            ScanRead::Truncated => return Ok(HuffmanResult::Truncated),
        };
        code = (code << 1) | bit;
        if (code as i32) <= table.maxcode[len] {
            let idx = (code as i32 + table.valoffset[len]) as usize;
            if idx < table.values.len() {
                return Ok(HuffmanResult::Symbol(table.values[idx]));
            }
        }
    }

    // If we've exhausted real data (hit marker or past end), treat invalid code as end of scan.
    // This happens when fill bits at end of scan don't form a valid Huffman code.
    if reader.is_exhausted() {
        return Ok(if reader.marker_found().is_some() {
            HuffmanResult::EndOfScan
        } else {
            HuffmanResult::Truncated
        });
    }

    // Lenient mode: treat invalid Huffman code as end-of-block
    if lenient {
        return Ok(HuffmanResult::InvalidCodeRecovered);
    }

    Err(Error::invalid_huffman_table(0, "invalid code"))
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
    /// Reusable coefficient buffer - avoids zeroing full 64 elements each block.
    /// Only positions 0..last_written are valid; rest may be garbage.
    coeff_buffer: [i16; DCT_BLOCK_SIZE],
    /// Number of positions written in coeff_buffer that need clearing before next use.
    /// Positions 0..last_written need to be zeroed; rest are already zero.
    last_written: u8,
    /// Lenient mode: recover from AC index overflow and invalid Huffman codes.
    lenient: bool,
    /// Tracks if lenient recovery was used (AC index overflow).
    pub(crate) had_ac_overflow: bool,
    /// Tracks if lenient recovery was used (invalid Huffman code).
    pub(crate) had_invalid_huffman: bool,
}

/// Saved state of an EntropyDecoder for speculative decoding.
#[derive(Clone, Copy)]
pub struct EntropyDecoderState {
    reader_state: crate::foundation::bitstream::BitReaderState,
    prev_dc: [i16; 4],
    last_written: u8,
}

impl<'data, 'tables> EntropyDecoder<'data, 'tables> {
    /// Creates a new entropy decoder.
    pub fn new(data: &'data [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            prev_dc: [0; 4],
            coeff_buffer: [0i16; DCT_BLOCK_SIZE],
            last_written: 0,
            lenient: false,
            had_ac_overflow: false,
            had_invalid_huffman: false,
        }
    }

    /// Enables lenient mode for maximum error recovery.
    ///
    /// In lenient mode:
    /// - AC coefficient index overflow is treated as end-of-block
    /// - Invalid Huffman codes mid-scan are treated as end-of-block
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
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
    /// In lenient mode, returns 0 (EOB) on invalid codes instead of erroring.
    #[inline(always)]
    fn decode_huffman(&mut self, table: &HuffmanDecodeTable) -> ScanResult<u8> {
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
                return Ok(ScanRead::Value(symbol));
            }
        }

        // Slow path for longer codes
        let mut code = 0u32;
        for len in 1..=16 {
            let bit = match self.reader.read_bits(1)? {
                ScanRead::Value(b) => b,
                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                ScanRead::Truncated => return Ok(ScanRead::Truncated),
            };
            code = (code << 1) | bit;
            if (code as i32) <= table.maxcode[len] {
                let idx = (code as i32 + table.valoffset[len]) as usize;
                if idx < table.values.len() {
                    return Ok(ScanRead::Value(table.values[idx]));
                }
            }
        }

        // If we've exhausted real data (hit marker or past end), treat invalid code as end of scan.
        // This happens when fill bits at end of scan don't form a valid Huffman code.
        if self.reader.is_exhausted() {
            return Ok(if self.reader.marker_found().is_some() {
                ScanRead::EndOfScan
            } else {
                ScanRead::Truncated
            });
        }

        // Lenient mode: treat invalid Huffman code as EOB
        if self.lenient {
            self.had_invalid_huffman = true;
            return Ok(ScanRead::Value(0)); // 0 = EOB symbol
        }

        Err(Error::invalid_huffman_table(0, "invalid code"))
    }

    /// Safely gets a DC table reference, handling out-of-bounds indices.
    /// Returns with 'tables lifetime to avoid borrowing self.
    fn get_dc_table(&self, idx: usize) -> Result<&'tables HuffmanDecodeTable> {
        self.dc_tables
            .get(idx)
            .and_then(|&t| t)
            .ok_or_else(|| Error::internal("DC table not set or invalid index"))
    }

    /// Safely gets an AC table reference, handling out-of-bounds indices.
    /// Returns with 'tables lifetime to avoid borrowing self.
    fn get_ac_table(&self, idx: usize) -> Result<&'tables HuffmanDecodeTable> {
        self.ac_tables
            .get(idx)
            .and_then(|&t| t)
            .ok_or_else(|| Error::internal("AC table not set or invalid index"))
    }

    /// Decodes a block of DCT coefficients with fast AC path.
    pub fn decode_block(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> ScanResult<[i16; DCT_BLOCK_SIZE]> {
        // Get table references once (tables are borrowed, no copying)
        let dc_table =
            self.dc_tables[dc_table_idx].ok_or_else(|| Error::internal("DC table not set"))?;
        let ac_table =
            self.ac_tables[ac_table_idx].ok_or_else(|| Error::internal("AC table not set"))?;

        let mut coeffs = [0i16; DCT_BLOCK_SIZE];

        // Decode DC coefficient using standalone function
        let dc_cat = match decode_huffman_symbol(&mut self.reader, dc_table)? {
            ScanRead::Value(v) => v,
            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => return Ok(ScanRead::Truncated),
        };

        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = match self.reader.read_bits(dc_cat)? {
                ScanRead::Value(v) => v as u16,
                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                ScanRead::Truncated => return Ok(ScanRead::Truncated),
            };
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
                            if self.lenient {
                                self.had_ac_overflow = true;
                                break; // Treat as EOB
                            }
                            return Err(Error::invalid_jpeg_data(
                                "AC coefficient index out of bounds",
                            ));
                        }

                        let bits = match self.reader.read_bits(ac_cat)? {
                            ScanRead::Value(v) => v as u16,
                            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                            ScanRead::Truncated => return Ok(ScanRead::Truncated),
                        };
                        coeffs[i] = decode_value(ac_cat, bits);
                        i += 1;
                    }
                    continue;
                }
            }

            // Slow path for long codes or when not enough bits
            let symbol = match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
                HuffmanResult::Symbol(v) => v,
                HuffmanResult::EndOfScan => return Ok(ScanRead::EndOfScan),
                HuffmanResult::Truncated => return Ok(ScanRead::Truncated),
                HuffmanResult::InvalidCodeRecovered => {
                    self.had_invalid_huffman = true;
                    break; // Treat as EOB
                }
            };

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
                    if self.lenient {
                        self.had_ac_overflow = true;
                        break; // Treat as EOB
                    }
                    return Err(Error::invalid_jpeg_data(
                        "AC coefficient index out of bounds",
                    ));
                }

                let bits = match self.reader.read_bits(ac_cat)? {
                    ScanRead::Value(v) => v as u16,
                    ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                    ScanRead::Truncated => return Ok(ScanRead::Truncated),
                };
                coeffs[i] = decode_value(ac_cat, bits);
                i += 1;
            }
        }

        Ok(ScanRead::Value(coeffs))
    }

    /// Decode a single 8x8 block of DCT coefficients, returning coefficient count.
    ///
    /// Returns `(coefficients, coeff_count)` where `coeff_count` is the position
    /// of the last non-zero coefficient in zigzag order (1-64). This enables
    /// tiered IDCT optimization:
    /// - count <= 1: DC-only block
    /// - count <= 10: Use 4x4 IDCT
    /// - count > 10: Use full 8x8 IDCT
    #[inline(always)]
    pub fn decode_block_with_count(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> ScanResult<([i16; DCT_BLOCK_SIZE], u8)> {
        // Get table references once (tables are borrowed, no copying)
        let dc_table =
            self.dc_tables[dc_table_idx].ok_or_else(|| Error::internal("DC table not set"))?;
        let ac_table =
            self.ac_tables[ac_table_idx].ok_or_else(|| Error::internal("AC table not set"))?;

        // Pre-fetch fast_ac slice to avoid Option check in hot loop
        let fast_ac = ac_table.fast_ac_slice();
        let has_fast_ac = !fast_ac.is_empty();

        // Smart zeroing: only clear positions written by previous block.
        // This is the zune-jpeg optimization - consecutive blocks have similar sparsity.
        // Instead of zeroing all 64 elements (128 bytes), we only zero the positions
        // that were actually written last time. For sparse blocks (typical), this
        // saves significant memory bandwidth.
        let clear_len = self.last_written as usize;
        if clear_len > 0 {
            self.coeff_buffer[..clear_len].fill(0);
        }

        // Use the pre-cleared buffer for this block
        let coeffs = &mut self.coeff_buffer;

        // Decode DC coefficient using standalone function
        let dc_cat = match decode_huffman_symbol(&mut self.reader, dc_table)? {
            ScanRead::Value(v) => v,
            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => return Ok(ScanRead::Truncated),
        };

        let dc_diff = if dc_cat == 0 {
            0
        } else {
            // Fast path: we just did peek_bits_refill(9), so we likely have enough bits
            let bits = if self.reader.bits_available() >= dc_cat {
                self.reader.read_bits_fast(dc_cat)
            } else {
                match self.reader.read_bits(dc_cat)? {
                    ScanRead::Value(v) => v,
                    ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                    ScanRead::Truncated => return Ok(ScanRead::Truncated),
                }
            };
            // Use branchless huff_extend instead of decode_value
            super::huff_extend(bits as i32, dc_cat as i32) as i16
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
                // Use direct slice access instead of method call
                if has_fast_ac {
                    let fast_ac_entry = fast_ac[idx];
                    if fast_ac_entry != 0 {
                        let value = fast_ac_entry >> 8;
                        let run = ((fast_ac_entry >> 4) & 0xF) as usize;
                        let total_bits = (fast_ac_entry & 0xF) as u8;
                        self.reader.skip_bits_fast(total_bits);
                        i += run;
                        if i < DCT_BLOCK_SIZE {
                            coeffs[i] = value;
                            last_nonzero = (i + 1) as u8;
                            i += 1;
                        }
                        continue;
                    }
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

                    let run = (symbol >> 4) as usize;
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
                        i += run;
                        if i >= DCT_BLOCK_SIZE {
                            if self.lenient {
                                self.had_ac_overflow = true;
                                break; // Treat as EOB
                            }
                            return Err(Error::invalid_jpeg_data(
                                "AC coefficient index out of bounds",
                            ));
                        }

                        // Fast path: after peek_bits_refill(9) + skip(code_length), we often have
                        // enough bits to read ac_cat (max 15) without refill
                        let bits = if self.reader.bits_available() >= ac_cat {
                            self.reader.read_bits_fast(ac_cat)
                        } else {
                            match self.reader.read_bits(ac_cat)? {
                                ScanRead::Value(v) => v,
                                ScanRead::EndOfScan => {
                                    self.last_written = DCT_BLOCK_SIZE as u8;
                                    return Ok(ScanRead::EndOfScan);
                                }
                                ScanRead::Truncated => {
                                    self.last_written = DCT_BLOCK_SIZE as u8;
                                    return Ok(ScanRead::Truncated);
                                }
                            }
                        };
                        // Use branchless huff_extend
                        coeffs[i] = super::huff_extend(bits as i32, ac_cat as i32) as i16;
                        last_nonzero = (i + 1) as u8;
                        i += 1;
                    }
                    continue;
                }
            }

            // Slow path for long codes or when not enough bits
            let symbol = match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
                HuffmanResult::Symbol(v) => v,
                HuffmanResult::EndOfScan => {
                    self.last_written = DCT_BLOCK_SIZE as u8;
                    return Ok(ScanRead::EndOfScan);
                }
                HuffmanResult::Truncated => {
                    self.last_written = DCT_BLOCK_SIZE as u8;
                    return Ok(ScanRead::Truncated);
                }
                HuffmanResult::InvalidCodeRecovered => {
                    self.had_invalid_huffman = true;
                    break; // Treat as EOB
                }
            };

            if symbol == 0 {
                // EOB - remaining coefficients are zero
                break;
            }

            let run = (symbol >> 4) as usize;
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
                i += run;
                if i >= DCT_BLOCK_SIZE {
                    if self.lenient {
                        self.had_ac_overflow = true;
                        break; // Treat as EOB
                    }
                    return Err(Error::invalid_jpeg_data(
                        "AC coefficient index out of bounds",
                    ));
                }

                // Fast path when we have enough bits
                let bits = if self.reader.bits_available() >= ac_cat {
                    self.reader.read_bits_fast(ac_cat)
                } else {
                    match self.reader.read_bits(ac_cat)? {
                        ScanRead::Value(v) => v,
                        ScanRead::EndOfScan => {
                            self.last_written = DCT_BLOCK_SIZE as u8;
                            return Ok(ScanRead::EndOfScan);
                        }
                        ScanRead::Truncated => {
                            self.last_written = DCT_BLOCK_SIZE as u8;
                            return Ok(ScanRead::Truncated);
                        }
                    }
                };
                // Use branchless huff_extend
                coeffs[i] = super::huff_extend(bits as i32, ac_cat as i32) as i16;
                last_nonzero = (i + 1) as u8;
                i += 1;
            }
        }

        // Record how much was written for next block's smart zeroing
        self.last_written = last_nonzero;

        // Return a copy of the buffer
        Ok(ScanRead::Value((*coeffs, last_nonzero)))
    }

    /// Zero-copy decode: write coefficients directly to caller's buffer.
    ///
    /// This is the high-performance API that avoids copying 128 bytes per block.
    /// The caller provides the destination buffer and tracks `prev_coeff_count`
    /// for smart zeroing (only zero positions that were written last time).
    ///
    /// # Arguments
    /// * `coeffs` - Destination buffer to write coefficients (must be valid memory)
    /// * `prev_coeff_count` - Coefficient count from previous block decode (for smart zeroing)
    /// * `component` - Component index for DC prediction
    /// * `dc_table_idx` - DC Huffman table index
    /// * `ac_table_idx` - AC Huffman table index
    ///
    /// # Returns
    /// * `ScanResult<u8>` - Coefficient count (1-64) on success
    ///
    /// # Smart Zeroing
    /// Only positions `0..prev_coeff_count` are zeroed before decoding.
    /// For consecutive sparse blocks (typical in JPEG), this saves ~50% of memory bandwidth.
    /// Pass 0 for first block or 64 to force full zeroing.
    #[inline(always)]
    pub fn decode_block_into(
        &mut self,
        coeffs: &mut [i16; DCT_BLOCK_SIZE],
        prev_coeff_count: u8,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> ScanResult<u8> {
        // Get table references once (tables are borrowed, no copying)
        let dc_table =
            self.dc_tables[dc_table_idx].ok_or_else(|| Error::internal("DC table not set"))?;
        let ac_table =
            self.ac_tables[ac_table_idx].ok_or_else(|| Error::internal("AC table not set"))?;

        // Pre-fetch fast_ac slice to avoid Option check in hot loop
        let fast_ac = ac_table.fast_ac_slice();
        let has_fast_ac = !fast_ac.is_empty();

        // Smart zeroing: only clear positions written by previous block.
        // Caller tracks prev_coeff_count per-component for interleaved MCUs.
        let clear_len = prev_coeff_count as usize;
        if clear_len > 0 {
            // Zero only what was written, not full 64 elements
            coeffs[..clear_len].fill(0);
        }

        // Decode DC coefficient
        let dc_cat = match decode_huffman_symbol(&mut self.reader, dc_table)? {
            ScanRead::Value(v) => v,
            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => return Ok(ScanRead::Truncated),
        };

        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = if self.reader.bits_available() >= dc_cat {
                self.reader.read_bits_fast(dc_cat)
            } else {
                match self.reader.read_bits(dc_cat)? {
                    ScanRead::Value(v) => v,
                    ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                    ScanRead::Truncated => return Ok(ScanRead::Truncated),
                }
            };
            super::huff_extend(bits as i32, dc_cat as i32) as i16
        };

        coeffs[0] = self.prev_dc[component].wrapping_add(dc_diff);
        self.prev_dc[component] = coeffs[0];

        let mut last_nonzero: u8 = 1;
        let mut i = 1;

        while i < DCT_BLOCK_SIZE {
            if let Some(bits9) = self
                .reader
                .peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8)
            {
                let idx = bits9 as usize;

                // Try fast AC decode first
                if has_fast_ac {
                    let fast_ac_entry = fast_ac[idx];
                    if fast_ac_entry != 0 {
                        let value = fast_ac_entry >> 8;
                        let run = ((fast_ac_entry >> 4) & 0xF) as usize;
                        let total_bits = (fast_ac_entry & 0xF) as u8;
                        self.reader.skip_bits_fast(total_bits);
                        i += run;
                        if i < DCT_BLOCK_SIZE {
                            coeffs[i] = value;
                            last_nonzero = (i + 1) as u8;
                            i += 1;
                        }
                        continue;
                    }
                }

                // Try regular fast Huffman lookup
                let lookup = ac_table.fast_lookup[idx];
                if lookup >= 0 {
                    let symbol = (lookup & 0xFF) as u8;
                    let code_length = (lookup >> 8) as u8;
                    self.reader.skip_bits_fast(code_length);

                    if symbol == 0 {
                        break; // EOB
                    }

                    let run = (symbol >> 4) as usize;
                    let ac_cat = symbol & 0x0F;

                    if ac_cat == 0 {
                        if run == 15 {
                            i += 16; // ZRL
                        } else {
                            break;
                        }
                    } else {
                        i += run;
                        if i >= DCT_BLOCK_SIZE {
                            if self.lenient {
                                self.had_ac_overflow = true;
                                break; // Treat as EOB
                            }
                            return Err(Error::invalid_jpeg_data(
                                "AC coefficient index out of bounds",
                            ));
                        }

                        let bits = if self.reader.bits_available() >= ac_cat {
                            self.reader.read_bits_fast(ac_cat)
                        } else {
                            match self.reader.read_bits(ac_cat)? {
                                ScanRead::Value(v) => v,
                                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                                ScanRead::Truncated => return Ok(ScanRead::Truncated),
                            }
                        };
                        coeffs[i] = super::huff_extend(bits as i32, ac_cat as i32) as i16;
                        last_nonzero = (i + 1) as u8;
                        i += 1;
                    }
                    continue;
                }
            }

            // Slow path
            let symbol = match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
                HuffmanResult::Symbol(v) => v,
                HuffmanResult::EndOfScan => return Ok(ScanRead::EndOfScan),
                HuffmanResult::Truncated => return Ok(ScanRead::Truncated),
                HuffmanResult::InvalidCodeRecovered => {
                    self.had_invalid_huffman = true;
                    break; // Treat as EOB
                }
            };

            if symbol == 0 {
                break; // EOB
            }

            let run = (symbol >> 4) as usize;
            let ac_cat = symbol & 0x0F;

            if ac_cat == 0 {
                if run == 15 {
                    i += 16; // ZRL
                } else {
                    break;
                }
            } else {
                i += run;
                if i >= DCT_BLOCK_SIZE {
                    if self.lenient {
                        self.had_ac_overflow = true;
                        break; // Treat as EOB
                    }
                    return Err(Error::invalid_jpeg_data(
                        "AC coefficient index out of bounds",
                    ));
                }

                let bits = if self.reader.bits_available() >= ac_cat {
                    self.reader.read_bits_fast(ac_cat)
                } else {
                    match self.reader.read_bits(ac_cat)? {
                        ScanRead::Value(v) => v,
                        ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                        ScanRead::Truncated => return Ok(ScanRead::Truncated),
                    }
                };
                coeffs[i] = super::huff_extend(bits as i32, ac_cat as i32) as i16;
                last_nonzero = (i + 1) as u8;
                i += 1;
            }
        }

        Ok(ScanRead::Value(last_nonzero))
    }

    /// Fast decode optimized for baseline JPEG (non-progressive).
    ///
    /// This version minimizes enum matching overhead by:
    /// 1. Pre-checking for markers before starting
    /// 2. Using branchless huff_extend
    /// 3. Batching refills
    /// 4. Direct slice access for fast_ac table
    ///
    /// Returns `None` if a marker was hit (end of scan), otherwise returns
    /// `(coefficients, coeff_count)`.
    #[inline(never)] // Prevent inlining to keep code cache pressure low
    pub fn decode_block_fast(
        &mut self,
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> Result<Option<([i16; DCT_BLOCK_SIZE], u8)>> {
        // Early exit if we already hit a marker
        if self.reader.marker_found().is_some() {
            return Ok(None);
        }

        // Get table references once
        let dc_table =
            self.dc_tables[dc_table_idx].ok_or_else(|| Error::internal("DC table not set"))?;
        let ac_table =
            self.ac_tables[ac_table_idx].ok_or_else(|| Error::internal("AC table not set"))?;

        // Get fast_ac slice once (empty slice if not built)
        let fast_ac = ac_table.fast_ac_slice();
        let has_fast_ac = !fast_ac.is_empty();

        let mut coeffs = [0i16; DCT_BLOCK_SIZE];

        // === Decode DC ===
        // Ensure we have enough bits for DC decode
        if !self.reader.ensure_bits() {
            // Not enough bits - check if marker or truncated
            return Ok(if self.reader.marker_found().is_some() {
                None
            } else {
                // Truncated - return zeros
                Some((coeffs, 1))
            });
        }

        // Fast DC decode
        let bits9 = self.reader.peek_top(HuffmanDecodeTable::FAST_BITS as u8);
        let dc_lookup = dc_table.fast_lookup[bits9 as usize];

        let dc_cat = if dc_lookup >= 0 {
            let symbol = (dc_lookup & 0xFF) as u8;
            let len = (dc_lookup >> 8) as u8;
            self.reader.skip_bits_fast(len);
            symbol
        } else {
            // Slow path - use the existing method
            match decode_huffman_symbol(&mut self.reader, dc_table)? {
                ScanRead::Value(v) => v,
                ScanRead::EndOfScan | ScanRead::Truncated => return Ok(None),
            }
        };

        let dc_diff = if dc_cat == 0 {
            0i16
        } else {
            // Ensure bits for DC value
            if self.reader.bits_available() < dc_cat
                && (!self.reader.ensure_bits() || self.reader.bits_available() < dc_cat)
            {
                // Not enough bits for DC value - truncated
                return Ok(None);
            }
            let bits = self.reader.read_bits_fast(dc_cat) as i32;
            super::huff_extend(bits, dc_cat as i32) as i16
        };

        coeffs[0] = self.prev_dc[component].wrapping_add(dc_diff);
        self.prev_dc[component] = coeffs[0];

        // === Decode AC coefficients ===
        let mut last_nonzero: u8 = 1;
        let mut i = 1usize;

        while i < DCT_BLOCK_SIZE {
            // Batch refill - ensure 32 bits before inner loop
            if !self.reader.ensure_bits() {
                // Check if we hit a marker
                if self.reader.marker_found().is_some() {
                    break;
                }
                // Otherwise truncated - use what we have
            }

            // Peek 9 bits for fast lookup
            let bits9 = self.reader.peek_top(HuffmanDecodeTable::FAST_BITS as u8) as usize;

            // Try fast AC first (combined run+value+length)
            if has_fast_ac {
                let fast_ac_entry = fast_ac[bits9];
                if fast_ac_entry != 0 {
                    let value = fast_ac_entry >> 8;
                    let run = ((fast_ac_entry >> 4) & 0xF) as usize;
                    let total_bits = (fast_ac_entry & 0xF) as u8;
                    self.reader.skip_bits_fast(total_bits);
                    i += run;
                    if i < DCT_BLOCK_SIZE {
                        coeffs[i] = value;
                        last_nonzero = (i + 1) as u8;
                        i += 1;
                    }
                    continue;
                }
            }

            // Regular fast Huffman lookup
            let ac_lookup = ac_table.fast_lookup[bits9];
            if ac_lookup >= 0 {
                let symbol = (ac_lookup & 0xFF) as u8;
                let code_length = (ac_lookup >> 8) as u8;
                self.reader.skip_bits_fast(code_length);

                if symbol == 0 {
                    // EOB
                    break;
                }

                let run = (symbol >> 4) as usize;
                let ac_cat = symbol & 0x0F;

                if ac_cat == 0 {
                    if run == 15 {
                        // ZRL
                        i += 16;
                    } else {
                        break;
                    }
                } else {
                    i += run;
                    if i >= DCT_BLOCK_SIZE {
                        if self.lenient {
                            self.had_ac_overflow = true;
                            break; // Treat as EOB
                        }
                        return Err(Error::invalid_jpeg_data(
                            "AC coefficient index out of bounds",
                        ));
                    }

                    // Read value bits - ensure we have enough
                    if self.reader.bits_available() < ac_cat
                        && (!self.reader.ensure_bits() || self.reader.bits_available() < ac_cat)
                    {
                        break; // Not enough bits - truncated
                    }
                    let bits = self.reader.read_bits_fast(ac_cat) as i32;
                    coeffs[i] = super::huff_extend(bits, ac_cat as i32) as i16;
                    last_nonzero = (i + 1) as u8;
                    i += 1;
                }
                continue;
            }

            // Slow path for long codes
            let symbol = match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
                HuffmanResult::Symbol(v) => v,
                HuffmanResult::EndOfScan | HuffmanResult::Truncated => break,
                HuffmanResult::InvalidCodeRecovered => {
                    self.had_invalid_huffman = true;
                    break; // Treat as EOB
                }
            };

            if symbol == 0 {
                break;
            }

            let run = (symbol >> 4) as usize;
            let ac_cat = symbol & 0x0F;

            if ac_cat == 0 {
                if run == 15 {
                    i += 16;
                } else {
                    break;
                }
            } else {
                i += run;
                if i >= DCT_BLOCK_SIZE {
                    if self.lenient {
                        self.had_ac_overflow = true;
                        break; // Treat as EOB
                    }
                    return Err(Error::invalid_jpeg_data(
                        "AC coefficient index out of bounds",
                    ));
                }

                if self.reader.bits_available() < ac_cat
                    && (!self.reader.ensure_bits() || self.reader.bits_available() < ac_cat)
                {
                    break; // Not enough bits - truncated
                }
                let bits = self.reader.read_bits_fast(ac_cat) as i32;
                coeffs[i] = super::huff_extend(bits, ac_cat as i32) as i16;
                last_nonzero = (i + 1) as u8;
                i += 1;
            }
        }

        Ok(Some((coeffs, last_nonzero)))
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
            last_written: self.last_written,
        }
    }

    /// Restores a previously saved state.
    pub fn restore_state(&mut self, state: EntropyDecoderState) {
        self.reader.restore_state(state.reader_state);
        self.prev_dc = state.prev_dc;
        self.last_written = state.last_written;
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
    ) -> ScanResult<i16> {
        let dc_table = self.get_dc_table(dc_table_idx)?;

        let dc_cat = match self.decode_huffman(dc_table)? {
            ScanRead::Value(v) => v,
            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => return Ok(ScanRead::Truncated),
        };
        let dc_diff = if dc_cat == 0 {
            0
        } else {
            let bits = match self.reader.read_bits(dc_cat)? {
                ScanRead::Value(v) => v as u16,
                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                ScanRead::Truncated => return Ok(ScanRead::Truncated),
            };
            decode_value(dc_cat, bits)
        };

        let shifted_dc = self.prev_dc[component].wrapping_add(dc_diff);
        self.prev_dc[component] = shifted_dc;

        // Return the unshifted value (shift left by al)
        Ok(ScanRead::Value(shifted_dc << al))
    }

    /// Decodes DC refinement bit (ah>0).
    /// Returns the bit to add at position al.
    pub fn decode_dc_refine(&mut self, al: u8) -> ScanResult<i16> {
        let bit = match self.reader.read_bits(1)? {
            ScanRead::Value(v) => v as i16,
            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => return Ok(ScanRead::Truncated),
        };
        Ok(ScanRead::Value(bit << al))
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
    ) -> ScanResult<()> {
        let ac_table = self.get_ac_table(ac_table_idx)?;

        // If we have a pending EOB run, decrement and skip this block
        if *eob_run > 0 {
            *eob_run -= 1;
            return Ok(ScanRead::Value(()));
        }

        let mut k = ss as usize;
        while k <= se as usize {
            let symbol = match self.decode_huffman(ac_table)? {
                ScanRead::Value(v) => v,
                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                ScanRead::Truncated => return Ok(ScanRead::Truncated),
            };
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
                        return Ok(ScanRead::Value(()));
                    } else {
                        // EOB run: 2^run + extra_bits
                        let extra = match self.reader.read_bits(run)? {
                            ScanRead::Value(v) => v as u16,
                            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                            ScanRead::Truncated => return Ok(ScanRead::Truncated),
                        };
                        *eob_run = (1 << run) + extra - 1; // -1 because this block counts as one
                        return Ok(ScanRead::Value(()));
                    }
                }
            } else {
                k += run as usize;
                if k > se as usize {
                    if self.lenient {
                        self.had_ac_overflow = true;
                        return Ok(ScanRead::Value(())); // Treat as EOB
                    }
                    return Err(Error::invalid_jpeg_data(
                        "AC coefficient index out of bounds",
                    ));
                }

                let bits = match self.reader.read_bits(size)? {
                    ScanRead::Value(v) => v as u16,
                    ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                    ScanRead::Truncated => return Ok(ScanRead::Truncated),
                };
                let value = decode_value(size, bits);
                coeffs[k] = value << al;
                k += 1;
            }
        }

        Ok(ScanRead::Value(()))
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
    ) -> ScanResult<()> {
        let ac_table = self.get_ac_table(ac_table_idx)?;
        let bit_val = 1i16 << al;

        /// Helper macro to read a refinement bit, handling EndOfScan
        macro_rules! read_refine_bit {
            ($self:expr) => {
                match $self.reader.read_bits(1)? {
                    ScanRead::Value(v) => v as i16,
                    ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                    ScanRead::Truncated => return Ok(ScanRead::Truncated),
                }
            };
        }

        /// Helper to apply refinement bit to a coefficient
        fn apply_refine(coeff: &mut i16, bit: i16, bit_val: i16) {
            if bit != 0 && (*coeff & bit_val) == 0 {
                if *coeff > 0 {
                    *coeff = coeff.saturating_add(bit_val);
                } else {
                    *coeff = coeff.saturating_sub(bit_val);
                }
            }
        }

        // If we have a pending EOB run, apply refinement bits to nonzero coeffs and return
        if *eob_run > 0 {
            for k in ss as usize..=se as usize {
                if coeffs[k] != 0 {
                    let bit = read_refine_bit!(self);
                    apply_refine(&mut coeffs[k], bit, bit_val);
                }
            }
            *eob_run -= 1;
            return Ok(ScanRead::Value(()));
        }

        let mut k = ss as usize;
        while k <= se as usize {
            let symbol = match self.decode_huffman(ac_table)? {
                ScanRead::Value(v) => v,
                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                ScanRead::Truncated => return Ok(ScanRead::Truncated),
            };
            let run = symbol >> 4;
            let size = symbol & 0x0F;

            let mut num_zeros_to_skip = run as usize;

            if size == 0 {
                if run == 15 {
                    // ZRL in refinement - skip 16 zeros (not 15!)
                    num_zeros_to_skip = 16;
                } else {
                    // EOB run
                    if run == 0 {
                        // Single EOB - apply refinement to remaining nonzero coeffs
                        for j in k..=se as usize {
                            if coeffs[j] != 0 {
                                let bit = read_refine_bit!(self);
                                apply_refine(&mut coeffs[j], bit, bit_val);
                            }
                        }
                        return Ok(ScanRead::Value(()));
                    } else {
                        // EOB run
                        let extra = match self.reader.read_bits(run)? {
                            ScanRead::Value(v) => v as u16,
                            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                            ScanRead::Truncated => return Ok(ScanRead::Truncated),
                        };
                        *eob_run = (1 << run) + extra - 1;
                        // Apply refinement to remaining nonzero coeffs in this block
                        for j in k..=se as usize {
                            if coeffs[j] != 0 {
                                let bit = read_refine_bit!(self);
                                apply_refine(&mut coeffs[j], bit, bit_val);
                            }
                        }
                        return Ok(ScanRead::Value(()));
                    }
                }
            }

            // For NEW_NZ (size=1), read sign bit FIRST, before refinement bits
            // This matches the JPEG spec bit order: [Huffman] [sign] [refinement bits]
            let new_val = if size != 0 {
                let sign_bit = read_refine_bit!(self);
                Some(if sign_bit != 0 { bit_val } else { -bit_val })
            } else {
                None
            };

            // Skip zeros and apply refinement bits to nonzero coefficients
            while k <= se as usize {
                // For ZRL (size=0), stop immediately after skipping all 16 zeros.
                if size == 0 && num_zeros_to_skip == 0 {
                    break;
                }

                if coeffs[k] != 0 {
                    // Apply refinement bit for previously-nonzero coefficient
                    let bit = read_refine_bit!(self);
                    apply_refine(&mut coeffs[k], bit, bit_val);
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

        Ok(ScanRead::Value(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanDecodeTable;

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
