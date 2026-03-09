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

/// Returns a u64 bitmask with bits [lo..=hi] set (0-indexed, both inclusive).
/// Used for masking nonzero bitmaps to a spectral selection range.
#[inline(always)]
pub(crate) fn range_bitmap(lo: u8, hi: u8) -> u64 {
    debug_assert!(lo <= hi && hi < 64);
    // Shift trick: ((1 << (hi+1)) - 1) & !((1 << lo) - 1)
    // Handle hi=63 overflow: use wrapping shift
    let top = if hi >= 63 {
        u64::MAX
    } else {
        (1u64 << (hi + 1)) - 1
    };
    let bot = (1u64 << lo) - 1;
    top & !bot
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
    // Try fast lookup first (9-bit table covers ~90% of codes)
    if let Some(bits) = reader.peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8) {
        let lookup = table.fast_lookup[bits as usize];
        if lookup >= 0 {
            let symbol = (lookup & 0xFF) as u8;
            let len = (lookup >> 8) as u8;
            reader.skip_bits_fast(len);
            return Ok(HuffmanResult::Symbol(symbol));
        }
    }

    // Slow path: peek 16 bits and use pre-shifted maxcode for fast comparison
    if let Some(bits16) = reader.peek_bits_refill(16)
        && let Some((symbol, len)) = table.decode_slow(bits16 as i32)
    {
        reader.skip_bits_fast(len);
        return Ok(HuffmanResult::Symbol(symbol));
    }

    // Edge case: near end of scan, can't peek 16 bits.
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

    /// Enables permissive restart marker handling.
    ///
    /// When enabled, any RST marker is accepted instead of requiring
    /// the exact expected sequence number.
    pub fn set_permissive_rst(&mut self, permissive: bool) {
        self.reader.set_permissive_rst(permissive);
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
        // Try fast lookup first (9-bit table covers ~90% of codes)
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

        // Slow path: peek 16 bits and use pre-shifted maxcode for fast comparison.
        // This replaces the bit-by-bit loop with a single peek + table scan.
        if let Some(bits16) = self.reader.peek_bits_refill(16)
            && let Some((symbol, len)) = table.decode_slow(bits16 as i32)
        {
            self.reader.skip_bits_fast(len);
            return Ok(ScanRead::Value(symbol));
        }

        // Edge case: near end of scan, can't peek 16 bits.
        // Fall back to bit-by-bit for the last few codes.
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
    #[inline(always)]
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
            let symbol =
                match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
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

        // Pre-fetch fast_ac as fixed-size array reference to eliminate bounds checks
        let fast_ac = ac_table.fast_ac_array();

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

                // Try fast AC decode first (fixed-size array = no bounds check)
                if let Some(fast_ac_arr) = fast_ac {
                    let fast_ac_entry = fast_ac_arr[idx];
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
            let symbol =
                match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
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

        // Pre-fetch fast_ac as fixed-size array reference to eliminate bounds checks.
        // Using &[i16; 512] lets the compiler prove 9-bit indices are always valid.
        let fast_ac = ac_table.fast_ac_array();

        // Smart zeroing: only clear positions written by previous block.
        // Caller tracks prev_coeff_count per-component for interleaved MCUs.
        let clear_len = prev_coeff_count as usize;
        if clear_len > 0 {
            // Zero only what was written, not full 64 elements
            coeffs[..clear_len].fill(0);
        }

        // Decode DC coefficient — inline fast path to avoid function call overhead
        let dc_cat = if let Some(bits9) = self
            .reader
            .peek_bits_refill(HuffmanDecodeTable::FAST_BITS as u8)
        {
            let lookup = dc_table.fast_lookup[bits9 as usize];
            if lookup >= 0 {
                let symbol = (lookup & 0xFF) as u8;
                let len = (lookup >> 8) as u8;
                self.reader.skip_bits_fast(len);
                symbol
            } else {
                // Slow path for long DC codes (rare)
                match decode_huffman_symbol(&mut self.reader, dc_table)? {
                    ScanRead::Value(v) => v,
                    ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                    ScanRead::Truncated => return Ok(ScanRead::Truncated),
                }
            }
        } else {
            match decode_huffman_symbol(&mut self.reader, dc_table)? {
                ScanRead::Value(v) => v,
                ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                ScanRead::Truncated => return Ok(ScanRead::Truncated),
            }
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

                // Try fast AC decode first (fixed-size array = no bounds check)
                if let Some(fast_ac_arr) = fast_ac {
                    let fast_ac_entry = fast_ac_arr[idx];
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
            let symbol =
                match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
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
            let symbol =
                match decode_huffman_symbol_lenient(&mut self.reader, ac_table, self.lenient)? {
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
    #[inline(always)]
    pub fn decode_ac_first(
        &mut self,
        coeffs: &mut [i16; DCT_BLOCK_SIZE],
        bitmap: &mut u64,
        ac_table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        eob_run: &mut u16,
    ) -> ScanResult<()> {
        // EOB fast path FIRST — before table lookup (most common path).
        // Avoids get_ac_table's Result overhead for blocks in EOB runs.
        if *eob_run > 0 {
            *eob_run -= 1;
            return Ok(ScanRead::Value(()));
        }

        let ac_table = self.get_ac_table(ac_table_idx)?;

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
                *bitmap |= 1u64 << (k & 63);
                k += 1;
            }
        }

        Ok(ScanRead::Value(()))
    }

    /// Decodes AC refinement for progressive scan (ah>0).
    /// Updates coefficients in range [ss, se].
    ///
    /// Hot path optimization: refinement bit reads use `read_bit_refine()` which
    /// avoids ScanRead enum wrapping, bit_buffer sync, and fill checks per bit.
    /// Returns 0 on exhaustion (safe: means "don't modify coefficient").
    ///
    /// The inner scan uses bitmap-accelerated iteration: instead of checking
    /// every position from k to se (O(se-k)), it jumps between nonzero positions
    /// via `trailing_zeros()` (O(nonzero_count)). For sparse blocks this reduces
    /// iterations from ~61 to ~3-10.
    ///
    /// Refinement bit application uses branchless arithmetic to eliminate
    /// unpredictable sign-dependent branches on the hot path.
    #[inline(always)]
    pub fn decode_ac_refine(
        &mut self,
        coeffs: &mut [i16; DCT_BLOCK_SIZE],
        bitmap: &mut u64,
        ac_table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        eob_run: &mut u16,
    ) -> ScanResult<()> {
        let bit_val = 1i16 << al;

        // EOB fast path FIRST — before table lookup (most common path).
        // Avoids get_ac_table's Result overhead for blocks in EOB runs.
        // Note: When called from progressive.rs, this path is usually intercepted
        // by the hoisted EOB check in the outer loop (which calls refine_eob_bits
        // directly). This remains as a fallback for other callers.
        if *eob_run > 0 {
            let range_mask = range_bitmap(ss, se);
            let mut nz = *bitmap & range_mask;
            let _ = self.reader.refill();
            while nz != 0 {
                let k = nz.trailing_zeros() as usize;
                let bit = self.reader.read_bit_refine();
                let c = coeffs[k];
                let sign = (c >> 15) | 1;
                let not_set = ((c & bit_val) == 0) as i16;
                coeffs[k] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);
                nz &= nz - 1;
            }
            *eob_run -= 1;
            return Ok(ScanRead::Value(()));
        }

        // Table lookup only needed for non-EOB blocks (Huffman decode path).
        let ac_table = self.get_ac_table(ac_table_idx)?;

        let se_usize = se as usize;
        let mut k = ss as usize;

        // Compute nonzero bitmap once for the entire block. Maintained across
        // Huffman events: bits are cleared as nonzero positions are processed,
        // so we never recompute range_bitmap per event (~11M instruction savings).
        let mut nz_remaining = *bitmap & range_bitmap(ss, se);

        while k <= se_usize {
            // Huffman decode uses normal path (infrequent, needs ScanRead handling)
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
                    // EOB — apply refinement to remaining nonzero coeffs.
                    // nz_remaining already has exactly the remaining nonzero
                    // positions from k to se (maintained across events).
                    if run != 0 {
                        let extra = match self.reader.read_bits(run)? {
                            ScanRead::Value(v) => v as u16,
                            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
                            ScanRead::Truncated => return Ok(ScanRead::Truncated),
                        };
                        *eob_run = (1 << run) + extra - 1;
                    }
                    while nz_remaining != 0 {
                        let j = nz_remaining.trailing_zeros() as usize;
                        let bit = self.reader.read_bit_refine();
                        let c = coeffs[j];
                        let sign = (c >> 15) | 1;
                        let not_set = ((c & bit_val) == 0) as i16;
                        coeffs[j] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);
                        nz_remaining &= nz_remaining - 1;
                    }
                    return Ok(ScanRead::Value(()));
                }
            }

            // For NEW_NZ (size=1), read sign bit FIRST, before refinement bits
            // This matches the JPEG spec bit order: [Huffman] [sign] [refinement bits]
            let new_val = if size != 0 {
                let sign_bit = self.reader.read_bit_refine();
                Some(if sign_bit != 0 { bit_val } else { -bit_val })
            } else {
                None
            };

            // Bitmap-accelerated inner scan: jump between nonzero positions
            // instead of checking every position k..=se individually.
            // nz_remaining is maintained across events (computed once per block).
            // Huffman decode just refilled buffer (32+ bits); inner scan consumes
            // at most ~15 bits (sign + refinement), well within available bits.
            loop {
                // ZRL termination: stop when all zeros have been skipped.
                if size == 0 && num_zeros_to_skip == 0 {
                    break;
                }

                if nz_remaining == 0 {
                    // All remaining positions are zeros — skip in bulk.
                    k += num_zeros_to_skip;
                    break;
                }

                let next_nz = nz_remaining.trailing_zeros() as usize;
                let zero_gap = next_nz - k;

                if num_zeros_to_skip < zero_gap {
                    // Target zero position is before next nonzero.
                    k += num_zeros_to_skip;
                    break;
                }

                // Skip zeros in this gap.
                num_zeros_to_skip -= zero_gap;
                k = next_nz;

                // ZRL: stop after exhausting zero count, before processing nonzero.
                if size == 0 && num_zeros_to_skip == 0 {
                    break;
                }

                // Process nonzero position: read refinement bit (branchless apply).
                let bit = self.reader.read_bit_refine();
                let c = coeffs[k];
                let sign = (c >> 15) | 1;
                let not_set = ((c & bit_val) == 0) as i16;
                coeffs[k] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);

                nz_remaining &= nz_remaining - 1; // clear lowest set bit
                k += 1;
            }

            if let Some(val) = new_val
                && k <= se_usize
            {
                // Place newly-nonzero coefficient and update bitmap.
                // Don't add to nz_remaining — this coefficient was just placed
                // and doesn't need refinement bits in this scan.
                coeffs[k] = val;
                *bitmap |= 1u64 << (k & 63);
                k += 1; // Move past the placed coefficient
            }
            // For ZRL (size==0), k already points past the zeros we skipped
        }

        Ok(ScanRead::Value(()))
    }

    /// Apply refinement bits to existing nonzero coefficients during an EOB run.
    ///
    /// This is the lean hot path for AC refinement EOB blocks. Called from the
    /// outer scan loop when eob_run > 0, bypassing the full decode_ac_refine()
    /// which includes unnecessary Huffman table lookup and ScanResult wrapping.
    ///
    /// The bitmap must already be masked to the spectral range [ss, se].
    /// Pre-refills the bit buffer once (32+ bits covers typical 5-15 nonzero
    /// positions per block), then uses branchless refinement arithmetic.
    #[inline(always)]
    pub fn refine_eob_bits(&mut self, coeffs: &mut [i16; DCT_BLOCK_SIZE], nz_bits: u64, al: u8) {
        let bit_val = 1i16 << al;
        let mut nz = nz_bits;
        // Pre-refill: one refill gives 32+ bits, enough for typical blocks.
        let _ = self.reader.refill();
        while nz != 0 {
            let k = nz.trailing_zeros() as usize;
            let bit = self.reader.read_bit_refine();
            // Branchless refinement: apply bit_val with sign of coefficient,
            // but only if bit=1 and this bit position isn't already set.
            let c = coeffs[k];
            let sign = (c >> 15) | 1; // -1 for negative, +1 for positive
            let not_set = ((c & bit_val) == 0) as i16;
            coeffs[k] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);
            nz &= nz - 1; // clear lowest set bit
        }
    }

    // ===== Fused progressive scan methods (no ScanResult wrapping) =====

    /// Fused AC first scan: processes entire block grid without ScanResult wrapping.
    ///
    /// Returns `Ok(false)` on truncation, `Ok(true)` on successful completion.
    /// Uses fast_ac combined Huffman+value lookup where available, and
    /// `peek_bits_refill(9)` for graceful end-of-scan handling (unlike
    /// `ensure_bits()` which requires 32 bits and fails prematurely).
    ///
    /// This eliminates per-block function call overhead, ScanResult enum
    /// construction/matching, and HuffmanResult→ScanRead conversion that
    /// dominates progressive decode instruction count.
    pub fn decode_ac_first_scan(
        &mut self,
        coeffs: &mut [Vec<[i16; DCT_BLOCK_SIZE]>],
        bitmaps: &mut [Vec<u64>],
        comp_idx: usize,
        ac_table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        blocks_h: usize,
        blocks_v: usize,
        padded_blocks_h: usize,
        restart_interval: u32,
        stop: &impl enough::Stop,
    ) -> Result<bool> {
        let ac_table = self.get_ac_table(ac_table_idx)?;
        let fast_ac = ac_table.fast_ac_array();
        let se_usize = se as usize;
        let mut eob_run = 0u16;
        let mut mcu_count = 0u32;
        let mut next_restart_num = 0u8;

        for block_y in 0..blocks_v {
            if block_y & 15 == 0 && stop.should_stop() {
                return Err(Error::cancelled());
            }

            for block_x in 0..blocks_h {
                // Restart marker handling
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Drain any unloaded coded bytes before the marker.
                    // The fast_ac path triggers fewer refills than the standard
                    // Huffman path, so position may lag behind consumed data.
                    while self.reader.marker_found().is_none() {
                        let _ = self.reader.refill();
                        if self.reader.bits_available() >= 32 {
                            self.reader.skip_bits_fast(32);
                        } else {
                            break;
                        }
                    }
                    self.reader.align_to_byte();
                    self.reader.read_restart_marker(next_restart_num)?;
                    next_restart_num = (next_restart_num + 1) & 7;
                    self.prev_dc = [0; 4];
                    eob_run = 0;
                }

                let block_idx = block_y * padded_blocks_h + block_x;
                let block = &mut coeffs[comp_idx][block_idx];
                let bitmap = &mut bitmaps[comp_idx][block_idx];

                // EOB fast path — most common case in progressive
                if eob_run > 0 {
                    eob_run -= 1;
                    mcu_count += 1;
                    continue;
                }

                let mut k = ss as usize;
                'block: while k <= se_usize {
                    // Peek 9 bits for Huffman lookup. peek_bits_refill handles
                    // refill internally and returns None only when truly exhausted
                    // (unlike ensure_bits which demands 32 bits).
                    // Peek 9 bits for Huffman + fast_ac lookup.
                    // When <9 bits remain after refill, use peek_top(9) with
                    // available-bits validation (the MSB-aligned zero-padding
                    // still gives a valid fast_lookup index for codes ≤ avail bits).
                    let bits9;
                    let partial_peek;
                    match self.reader.peek_bits_refill(9) {
                        Some(b) => {
                            bits9 = b;
                            partial_peek = false;
                        }
                        None => {
                            let avail = self.reader.bits_available();
                            if avail == 0 {
                                return Ok(false);
                            }
                            bits9 = self.reader.peek_top(9);
                            partial_peek = true;
                        }
                    };

                    // Try fast_ac combined lookup first (9-bit → symbol + value)
                    if !partial_peek && let Some(fast_ac_arr) = fast_ac {
                        let entry = fast_ac_arr[bits9 as usize];
                        if entry != 0 {
                            let value = entry >> 8;
                            let run = ((entry >> 4) & 0xF) as usize;
                            let total_bits = (entry & 0xF) as u8;
                            self.reader.skip_bits_fast(total_bits);
                            k += run;
                            if k > se_usize {
                                break 'block;
                            }
                            block[k] = value << al;
                            *bitmap |= 1u64 << (k & 63);
                            k += 1;
                            continue 'block;
                        }
                    }

                    // Standard Huffman decode using the (possibly partial) 9-bit peek
                    let lookup = ac_table.fast_lookup[bits9 as usize];
                    let symbol = if lookup >= 0 {
                        let code_len = (lookup >> 8) as u8;
                        // For partial peeks, verify the code fits in available bits
                        if partial_peek && code_len > self.reader.bits_available() {
                            return Ok(false);
                        }
                        self.reader.skip_bits_fast(code_len);
                        (lookup & 0xFF) as u8
                    } else {
                        // Slow path: need 16 bits for extended Huffman codes
                        match self.reader.peek_bits_refill(16) {
                            Some(bits16) => {
                                if let Some((sym, len)) = ac_table.decode_slow(bits16 as i32) {
                                    self.reader.skip_bits_fast(len);
                                    sym
                                } else {
                                    // Invalid code — treat as EOB
                                    break 'block;
                                }
                            }
                            None => {
                                // Edge case: near restart marker or end of scan,
                                // fewer than 16 bits available. Fall back to
                                // bit-by-bit Huffman decode (matches the standard
                                // decode_huffman_symbol_lenient fallback).
                                let mut code = 0u32;
                                let mut found_symbol = None;
                                for len in 1..=16usize {
                                    let bit = match self.reader.read_bits(1)? {
                                        ScanRead::Value(b) => b,
                                        ScanRead::EndOfScan | ScanRead::Truncated => {
                                            return Ok(false);
                                        }
                                    };
                                    code = (code << 1) | bit;
                                    if (code as i32) <= ac_table.maxcode[len] {
                                        let idx = (code as i32 + ac_table.valoffset[len]) as usize;
                                        if idx < ac_table.values.len() {
                                            found_symbol = Some(ac_table.values[idx]);
                                            break;
                                        }
                                    }
                                }
                                match found_symbol {
                                    Some(sym) => sym,
                                    None => {
                                        // Invalid code or truly exhausted
                                        break 'block;
                                    }
                                }
                            }
                        }
                    };

                    let run = symbol >> 4;
                    let size = symbol & 0x0F;

                    if size == 0 {
                        if run == 15 {
                            k += 16;
                        } else if run == 0 {
                            // Single EOB
                            break 'block;
                        } else {
                            // EOB run: 2^run + extra_bits - 1 (run ≤ 14 bits)
                            let _ = self.reader.refill();
                            if self.reader.bits_available() < run {
                                return Ok(false);
                            }
                            let extra = self.reader.read_bits_fast(run) as u16;
                            eob_run = (1 << run) + extra - 1;
                            break 'block;
                        }
                    } else {
                        k += run as usize;
                        if k > se_usize {
                            // AC overflow — treat as EOB
                            break 'block;
                        }
                        // Extra bits for coefficient value (size ≤ 10)
                        let _ = self.reader.refill();
                        if self.reader.bits_available() < size {
                            return Ok(false);
                        }
                        let bits = self.reader.read_bits_fast(size) as u16;
                        let value = decode_value(size, bits);
                        block[k] = value << al;
                        *bitmap |= 1u64 << (k & 63);
                        k += 1;
                    }
                }

                mcu_count += 1;
            }
        }

        Ok(true)
    }

    /// Fused AC refinement scan: processes entire block grid without ScanResult wrapping.
    ///
    /// Returns `Ok(false)` on truncation, `Ok(true)` on successful completion.
    /// Combines the EOB fast path and Huffman decode path into one loop with
    /// `peek_bits_refill(9)` for Huffman and branchless refinement arithmetic.
    pub fn decode_ac_refine_scan(
        &mut self,
        coeffs: &mut [Vec<[i16; DCT_BLOCK_SIZE]>],
        bitmaps: &mut [Vec<u64>],
        comp_idx: usize,
        ac_table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        blocks_h: usize,
        blocks_v: usize,
        padded_blocks_h: usize,
        restart_interval: u32,
        stop: &impl enough::Stop,
    ) -> Result<bool> {
        let ac_table = self.get_ac_table(ac_table_idx)?;
        let bit_val = 1i16 << al;
        let range_mask = range_bitmap(ss, se);
        let se_usize = se as usize;
        let mut eob_run = 0u16;
        let mut mcu_count = 0u32;
        let mut next_restart_num = 0u8;

        for block_y in 0..blocks_v {
            if block_y & 15 == 0 && stop.should_stop() {
                return Err(Error::cancelled());
            }

            for block_x in 0..blocks_h {
                // Restart marker handling
                if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
                    // Drain any unloaded coded bytes before the marker.
                    while self.reader.marker_found().is_none() {
                        let _ = self.reader.refill();
                        if self.reader.bits_available() >= 32 {
                            self.reader.skip_bits_fast(32);
                        } else {
                            break;
                        }
                    }
                    self.reader.align_to_byte();
                    self.reader.read_restart_marker(next_restart_num)?;
                    next_restart_num = (next_restart_num + 1) & 7;
                    self.prev_dc = [0; 4];
                    eob_run = 0;
                }

                let block_idx = block_y * padded_blocks_h + block_x;
                let block = &mut coeffs[comp_idx][block_idx];
                let bitmap = &mut bitmaps[comp_idx][block_idx];

                // EOB fast path — apply refinement bits to existing nonzero coefficients
                if eob_run > 0 {
                    let nz = *bitmap & range_mask;
                    let nz_count = nz.count_ones() as u8;
                    if nz_count > 0 {
                        let _ = self.reader.refill();
                        if nz_count <= 32 && nz_count <= self.reader.bits_available() {
                            // Batch: read all refinement bits at once
                            let batch = self.reader.read_bits_fast(nz_count);
                            let mut remaining = nz;
                            let mut shift = nz_count;
                            while remaining != 0 {
                                let j = remaining.trailing_zeros() as usize;
                                shift -= 1;
                                let bit = ((batch >> shift) & 1) as i16;
                                let c = block[j];
                                let sign = (c >> 15) | 1;
                                let not_set = ((c & bit_val) == 0) as i16;
                                block[j] = c.wrapping_add(bit * not_set * sign * bit_val);
                                remaining &= remaining - 1;
                            }
                        } else {
                            // Fallback for >32 nonzero or insufficient bits
                            let mut remaining = nz;
                            while remaining != 0 {
                                let j = remaining.trailing_zeros() as usize;
                                let bit = self.reader.read_bit_refine();
                                let c = block[j];
                                let sign = (c >> 15) | 1;
                                let not_set = ((c & bit_val) == 0) as i16;
                                block[j] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);
                                remaining &= remaining - 1;
                            }
                        }
                    }
                    eob_run -= 1;
                    mcu_count += 1;
                    continue;
                }

                // Non-EOB block: Huffman decode path
                let mut k = ss as usize;
                let mut nz_remaining = *bitmap & range_mask;

                // Pre-refill for fast Huffman decode (optimization, not required)
                let _ = self.reader.ensure_bits();

                while k <= se_usize {
                    // Refill when bits are low. Fast lookup needs 9 bits minimum;
                    // 25 covers one full Huffman event (16-bit code + 1 sign + 8 refine).
                    if self.reader.bits_available() < 25 {
                        let _ = self.reader.ensure_bits();
                        if self.reader.bits_available() < 9 {
                            break; // Not enough for even a fast Huffman lookup
                        }
                    }

                    // Huffman decode: peek_top is safe because we have >= 9 bits.
                    // Fast lookup (positive) always has code_len <= 9, so no overflow.
                    let bits9 = self.reader.peek_top(9) as usize;
                    let lookup = ac_table.fast_lookup[bits9];
                    let symbol = if lookup >= 0 {
                        let code_len = (lookup >> 8) as u8;
                        self.reader.skip_bits_fast(code_len);
                        (lookup & 0xFF) as u8
                    } else {
                        // Slow path: extended codes need up to 16 bits
                        if self.reader.bits_available() < 16 {
                            let _ = self.reader.refill();
                        }
                        if self.reader.bits_available() >= 16 {
                            let bits16 = self.reader.peek_top(16);
                            if let Some((sym, len)) = ac_table.decode_slow(bits16 as i32) {
                                self.reader.skip_bits_fast(len);
                                sym
                            } else {
                                break;
                            }
                        } else {
                            // Edge case: near restart marker or end of scan,
                            // fewer than 16 bits available. Fall back to
                            // bit-by-bit Huffman decode.
                            let mut code = 0u32;
                            let mut found_symbol = None;
                            for len in 1..=16usize {
                                let bit = self.reader.read_bit_refine();
                                code = (code << 1) | (bit as u32);
                                if (code as i32) <= ac_table.maxcode[len] {
                                    let idx = (code as i32 + ac_table.valoffset[len]) as usize;
                                    if idx < ac_table.values.len() {
                                        found_symbol = Some(ac_table.values[idx]);
                                        break;
                                    }
                                }
                            }
                            match found_symbol {
                                Some(sym) => sym,
                                None => break,
                            }
                        }
                    };

                    let run = symbol >> 4;
                    let size = symbol & 0x0F;

                    if size == 0 {
                        if run != 15 {
                            // EOB — apply refinement to remaining nonzero coeffs
                            if run != 0 {
                                let _ = self.reader.refill();
                                if self.reader.bits_available() < run {
                                    break;
                                }
                                let extra = self.reader.read_bits_fast(run) as u16;
                                eob_run = (1 << run) + extra - 1;
                            }
                            let rem_count = nz_remaining.count_ones() as u8;
                            if rem_count > 0 {
                                if rem_count > self.reader.bits_available() {
                                    let _ = self.reader.refill();
                                }
                                if rem_count <= 32 && rem_count <= self.reader.bits_available() {
                                    let batch = self.reader.read_bits_fast(rem_count);
                                    let mut shift = rem_count;
                                    while nz_remaining != 0 {
                                        let j = nz_remaining.trailing_zeros() as usize;
                                        shift -= 1;
                                        let bit = ((batch >> shift) & 1) as i16;
                                        let c = block[j];
                                        let sign = (c >> 15) | 1;
                                        let not_set = ((c & bit_val) == 0) as i16;
                                        block[j] = c.wrapping_add(bit * not_set * sign * bit_val);
                                        nz_remaining &= nz_remaining - 1;
                                    }
                                } else {
                                    while nz_remaining != 0 {
                                        let j = nz_remaining.trailing_zeros() as usize;
                                        let bit = self.reader.read_bit_refine();
                                        let c = block[j];
                                        let sign = (c >> 15) | 1;
                                        let not_set = ((c & bit_val) == 0) as i16;
                                        block[j] =
                                            c.wrapping_add((bit as i16) * not_set * sign * bit_val);
                                        nz_remaining &= nz_remaining - 1;
                                    }
                                }
                            }
                            break; // Done with this block
                        }

                        // ZRL: skip 16 zero positions, refining nonzeros along the way.
                        // Separate path avoids `size == 0` checks in inner loop.
                        let mut num_zeros_to_skip: usize = 16;
                        loop {
                            if num_zeros_to_skip == 0 {
                                break;
                            }
                            if nz_remaining == 0 {
                                k += num_zeros_to_skip;
                                break;
                            }
                            let next_nz = nz_remaining.trailing_zeros() as usize;
                            let zero_gap = next_nz - k;
                            if num_zeros_to_skip <= zero_gap {
                                k += num_zeros_to_skip;
                                break;
                            }
                            num_zeros_to_skip -= zero_gap;
                            k = next_nz;

                            let bit = self.reader.read_bit_refine();
                            let c = block[k];
                            let sign = (c >> 15) | 1;
                            let not_set = ((c & bit_val) == 0) as i16;
                            block[k] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);
                            nz_remaining &= nz_remaining - 1;
                            k += 1;
                        }
                    } else {
                        // NEW_NZ: skip `run` zero positions, then place new coefficient.
                        // Separate path avoids Option wrapping and `size == 0` checks.
                        let sign_bit = self.reader.read_bit_refine();
                        let new_val = if sign_bit != 0 { bit_val } else { -bit_val };
                        let mut num_zeros_to_skip = run as usize;

                        loop {
                            if nz_remaining == 0 {
                                k += num_zeros_to_skip;
                                break;
                            }
                            let next_nz = nz_remaining.trailing_zeros() as usize;
                            let zero_gap = next_nz - k;
                            if num_zeros_to_skip < zero_gap {
                                k += num_zeros_to_skip;
                                break;
                            }
                            num_zeros_to_skip -= zero_gap;
                            k = next_nz;

                            let bit = self.reader.read_bit_refine();
                            let c = block[k];
                            let sign = (c >> 15) | 1;
                            let not_set = ((c & bit_val) == 0) as i16;
                            block[k] = c.wrapping_add((bit as i16) * not_set * sign * bit_val);
                            nz_remaining &= nz_remaining - 1;
                            k += 1;
                        }

                        if k <= se_usize {
                            block[k] = new_val;
                            *bitmap |= 1u64 << (k & 63);
                            k += 1;
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        Ok(true)
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
