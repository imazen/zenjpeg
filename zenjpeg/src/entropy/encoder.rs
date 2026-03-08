//! Entropy encoder for JPEG.
//!
//! Provides `EntropyEncoder` for baseline and progressive JPEG encoding.

#![allow(dead_code)]

use crate::encode::build_nonzero_mask;
use crate::error::{Error, Result};
use crate::foundation::bitstream::BitWriter;
use crate::foundation::consts::DCT_BLOCK_SIZE;
use crate::huffman::HuffmanEncodeTable;
use crate::huffman::optimize::{ScanTokenInfo, Token};

use super::{additional_bits_with_cat, category};

/// Inner implementation of block encoding.
/// Uses `build_nonzero_mask` (which has its own archmage SIMD dispatch) to
/// skip zero coefficients. No floating-point SIMD here — pure integer bit ops.
#[inline]
fn encode_block_simd_impl(
    coeffs: &[i16; DCT_BLOCK_SIZE],
    dc_table: &HuffmanEncodeTable,
    ac_table: &HuffmanEncodeTable,
    prev_dc: i16,
    writer: &mut BitWriter,
) -> (bool, i16) {
    // Encode DC coefficient
    let dc = coeffs[0];
    let dc_diff = dc - prev_dc;

    let dc_cat = category(dc_diff);
    let (code, len) = dc_table.encode(dc_cat);

    // Combined write: Huffman code + extra bits in one operation
    if dc_cat > 0 {
        let additional = additional_bits_with_cat(dc_diff, dc_cat);
        writer.write_code_and_extra(code, len, additional, dc_cat);
    } else {
        writer.write_bits(code, len);
    }

    // Build 64-bit mask of non-zero coefficients using SIMD
    let nonzero_mask = build_nonzero_mask(coeffs);

    // Clear DC bit (bit 0), keep only AC bits (1-63)
    let ac_mask = nonzero_mask & !1u64;

    // Fast path: all AC coefficients are zero
    if ac_mask == 0 {
        let (code, len) = ac_table.encode(0x00); // EOB
        writer.write_bits(code, len);
        return (true, dc);
    }

    // Find position of last non-zero AC coefficient (1-63)
    let last_nonzero_idx = 63 - ac_mask.leading_zeros() as usize;

    // Process each non-zero AC coefficient
    let mut remaining = ac_mask;
    let mut prev_idx = 0usize;

    while remaining != 0 {
        let idx = remaining.trailing_zeros() as usize;
        let run = (idx - prev_idx - 1) as u8;

        // Handle runs of 16+ zeros (emit ZRL symbols)
        let mut r = run;
        while r >= 16 {
            let (code, len) = ac_table.encode(0xF0); // ZRL
            writer.write_bits(code, len);
            r -= 16;
        }

        // Encode the coefficient with combined write
        let ac = coeffs[idx];
        let ac_cat = category(ac);
        let symbol = (r << 4) | ac_cat;
        let (code, len) = ac_table.encode(symbol);
        let additional = additional_bits_with_cat(ac, ac_cat);
        writer.write_code_and_extra(code, len, additional, ac_cat);

        prev_idx = idx;
        remaining &= remaining - 1; // Clear lowest set bit
    }

    // EOB if there are trailing zeros
    if last_nonzero_idx < 63 {
        let (code, len) = ac_table.encode(0x00); // EOB
        writer.write_bits(code, len);
    }

    (true, dc)
}

/// Entropy encoder for a single scan.
///
/// Uses borrowed Huffman tables to avoid cloning ~1.3KB per table.
pub struct EntropyEncoder<'a> {
    /// Bit writer
    writer: BitWriter,
    /// DC Huffman tables (indexed by table selector)
    dc_tables: [Option<&'a HuffmanEncodeTable>; 4],
    /// AC Huffman tables (indexed by table selector)
    ac_tables: [Option<&'a HuffmanEncodeTable>; 4],
    /// Previous DC values for each component
    prev_dc: [i16; 4],
    /// Restart interval counter
    restart_counter: u16,
    /// Restart interval
    restart_interval: u16,
    /// Current restart marker number (0-7)
    restart_num: u8,
}

impl<'a> EntropyEncoder<'a> {
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

    /// Creates a new entropy encoder with pre-allocated output buffer.
    ///
    /// Use this when you can estimate the output size to avoid reallocations.
    /// A reasonable estimate is ~100 bytes per 8x8 block for quality 80.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            writer: BitWriter::with_capacity(capacity),
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            prev_dc: [0; 4],
            restart_counter: 0,
            restart_interval: 0,
            restart_num: 0,
        }
    }

    /// Sets a DC Huffman table (borrowed, not cloned).
    pub fn set_dc_table(&mut self, idx: usize, table: &'a HuffmanEncodeTable) {
        if idx < 4 {
            self.dc_tables[idx] = Some(table);
        }
    }

    /// Sets an AC Huffman table (borrowed, not cloned).
    pub fn set_ac_table(&mut self, idx: usize, table: &'a HuffmanEncodeTable) {
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
    /// Uses SIMD to quickly find non-zero coefficients and skip runs of zeros.
    ///
    /// # Arguments
    /// * `coeffs` - Quantized DCT coefficients in zigzag order
    /// * `component` - Component index (for DC prediction)
    /// * `dc_table_idx` - DC Huffman table index
    /// * `ac_table_idx` - AC Huffman table index
    #[inline]
    pub fn encode_block(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) {
        self.encode_block_simd(coeffs, component, dc_table_idx, ac_table_idx)
    }

    /// Scalar reference implementation of block encoding.
    ///
    /// Not used in production — the SIMD path (`encode_block_simd_impl`) is
    /// always used instead. Kept as a readable reference for the encoding
    /// algorithm and for correctness verification in tests.
    #[allow(dead_code)]
    #[inline]
    fn encode_block_scalar(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) -> Result<()> {
        let dc_table = self.dc_tables[dc_table_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("DC table not set"))?;
        let ac_table = self.ac_tables[ac_table_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("AC table not set"))?;

        // Encode DC coefficient
        let dc = coeffs[0];
        let dc_diff = dc - self.prev_dc[component];
        self.prev_dc[component] = dc;

        let dc_cat = category(dc_diff);
        let (code, len) = dc_table.encode(dc_cat);
        self.writer.write_bits(code, len);

        if dc_cat > 0 {
            let additional = additional_bits_with_cat(dc_diff, dc_cat);
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

                let additional = additional_bits_with_cat(ac, ac_cat);
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

    /// Encodes a block using SIMD to quickly find non-zero coefficients.
    ///
    /// Uses SIMD comparison to build a bitmask of non-zero positions,
    /// then iterates only through non-zero coefficients using bit manipulation.
    /// This reduces branching overhead when blocks have many zeros.
    ///
    /// Delegates to a multiversioned free function for CPU-specific optimizations.
    /// Encodes a single 8x8 block using SIMD-optimized entropy coding.
    ///
    /// # Panics
    /// Panics if DC or AC tables are not set (indicates encoder misconfiguration).
    #[inline]
    pub fn encode_block_simd(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        component: usize,
        dc_table_idx: usize,
        ac_table_idx: usize,
    ) {
        // SAFETY: Tables must be set before encoding. Using expect() instead of
        // ok_or_else() avoids Error allocation overhead in the hot path.
        let dc_table = self.dc_tables[dc_table_idx]
            .as_ref()
            .expect("DC table not set");
        let ac_table = self.ac_tables[ac_table_idx]
            .as_ref()
            .expect("AC table not set");

        let prev_dc = self.prev_dc[component];
        let (_, new_dc) =
            encode_block_simd_impl(coeffs, dc_table, ac_table, prev_dc, &mut self.writer);
        self.prev_dc[component] = new_dc;
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

    // ===== Progressive encoding methods =====

    /// Writes raw bits to the output (for progressive refinement).
    #[inline]
    pub fn write_bits(&mut self, bits: u32, count: u8) {
        self.writer.write_bits(bits, count);
    }

    /// Encodes a DC coefficient for progressive scan.
    ///
    /// For DC first scan (ah=0): encodes the full shifted DC value.
    /// For DC refinement (ah>0): encodes just one bit.
    pub fn encode_dc_progressive(
        &mut self,
        dc: i16,
        component: usize,
        table_idx: usize,
        al: u8,
        ah: u8,
    ) -> Result<()> {
        if ah > 0 {
            // DC refinement: just emit the bit at position al
            let bit = ((dc >> al) & 1) as u32;
            self.writer.write_bits(bit, 1);
        } else {
            // DC first scan: encode shifted DC difference
            let shifted_dc = dc >> al;
            let dc_diff = shifted_dc - self.prev_dc[component];
            self.prev_dc[component] = shifted_dc;

            let dc_table = self.dc_tables[table_idx]
                .as_ref()
                .ok_or_else(|| Error::internal("DC table not set"))?;

            let dc_cat = category(dc_diff);
            let (code, len) = dc_table.encode(dc_cat);
            if len == 0 {
                return Err(Error::internal("DC symbol not in Huffman table"));
            }
            self.writer.write_bits(code, len);

            if dc_cat > 0 {
                let additional = additional_bits_with_cat(dc_diff, dc_cat);
                self.writer.write_bits(additional as u32, dc_cat);
            }
        }

        Ok(())
    }

    /// Encodes AC coefficients for progressive first scan.
    ///
    /// Returns the EOB run that should be accumulated.
    ///
    /// IMPORTANT: We must use absolute values for zero-detection to match
    /// the refinement scan's classification. Otherwise, small negative
    /// coefficients like -2 with al=2 would be incorrectly encoded here
    /// (because (-2) >> 2 = -1 in signed arithmetic) but classified as
    /// "newly-nonzero" in refinement (because abs(-2) >> 2 = 0).
    pub fn encode_ac_progressive_first(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        eob_run: &mut u16,
    ) -> Result<()> {
        // Find the last non-zero coefficient in range [ss, se]
        // Use absolute value for consistency with refinement scan classification
        let mut last_nz = ss as usize;
        for k in (ss as usize..=se as usize).rev() {
            let abs_shifted = coeffs[k].unsigned_abs() >> al;
            if abs_shifted != 0 {
                last_nz = k;
                break;
            }
        }

        // Check if block is all zeros in this range (using absolute values)
        let all_zero = (ss as usize..=se as usize).all(|k| (coeffs[k].unsigned_abs() >> al) == 0);

        if all_zero {
            *eob_run += 1;
            if *eob_run == 0x7FFF {
                self.emit_eob_run_by_idx(table_idx, *eob_run)?;
                *eob_run = 0;
            }
            return Ok(());
        }

        // Emit any pending EOB run
        if *eob_run > 0 {
            self.emit_eob_run_by_idx(table_idx, *eob_run)?;
            *eob_run = 0;
        }

        let ac_table = self.ac_tables[table_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("AC table not set"))?;

        // Encode non-zero coefficients
        let mut r = 0u8; // Run of zeros
        for k in ss as usize..=last_nz {
            let coef = coeffs[k];
            let abs_shifted = coef.unsigned_abs() >> al;

            if abs_shifted == 0 {
                r += 1;
                continue;
            }

            // Emit ZRL (16 zeros) tokens if needed
            while r >= 16 {
                let (code, len) = ac_table.encode(0xF0);
                self.writer.write_bits(code, len);
                r -= 16;
            }

            // Encode the coefficient using absolute value shifted by al
            let nbits = 16 - abs_shifted.leading_zeros();
            let symbol = ((r as u16) << 4) | nbits as u16;

            let (code, len) = ac_table.encode(symbol as u8);
            if len == 0 {
                return Err(Error::internal("AC symbol not in Huffman table"));
            }
            self.writer.write_bits(code, len);

            // Additional bits (magnitude and sign)
            // For progressive encoding, we encode the shifted value
            let bits = if coef < 0 {
                (abs_shifted - 1) as u32
            } else {
                abs_shifted as u32
            };
            self.writer.write_bits(bits, nbits as u8);

            r = 0;
        }

        // If we didn't reach se, there's an EOB
        if last_nz < se as usize {
            *eob_run += 1;
            if *eob_run == 0x7FFF {
                self.emit_eob_run_by_idx(table_idx, *eob_run)?;
                *eob_run = 0;
            }
        }

        Ok(())
    }

    /// Emits an EOB run for progressive AC encoding using table index.
    fn emit_eob_run_by_idx(&mut self, table_idx: usize, eob_run: u16) -> Result<()> {
        if eob_run == 0 {
            return Ok(());
        }

        let ac_table = self.ac_tables[table_idx]
            .as_ref()
            .ok_or_else(|| Error::internal("AC table not set"))?;

        // EOB run encoding:
        // - eob_run=1: symbol=0x00 (EOB), no extra bits
        // - eob_run>=2: symbol=(nbits<<4), extra_bits = run & ((1<<nbits)-1)
        // nbits = floor(log2(eob_run)) = 31 - leading_zeros for u32
        let nbits = if eob_run == 1 {
            0
        } else {
            31 - (eob_run as u32).leading_zeros()
        };
        let symbol = (nbits << 4) as u8;

        let (code, len) = ac_table.encode(symbol);

        // Check if this symbol exists in the table (length 0 means not present)
        if len == 0 && symbol != 0x00 {
            // Symbol not in table (e.g., standard tables don't have EOB run symbols)
            // Fall back to emitting individual EOBs
            let (eob_code, eob_len) = ac_table.encode(0x00);
            for _ in 0..eob_run {
                self.writer.write_bits(eob_code, eob_len);
            }
        } else {
            self.writer.write_bits(code, len);

            if nbits > 0 {
                let extra_bits = eob_run & ((1 << nbits) - 1);
                self.writer.write_bits(extra_bits as u32, nbits as u8);
            }
        }

        Ok(())
    }

    /// Flushes the EOB run at the end of a progressive AC scan.
    pub fn flush_eob_run(&mut self, table_idx: usize, eob_run: u16) -> Result<()> {
        self.emit_eob_run_by_idx(table_idx, eob_run)
    }

    /// Encodes AC coefficients for progressive refinement scan.
    ///
    /// Based on mozjpeg-rs implementation which uses a simpler, direct encoding approach.
    /// For refinement scans (Ah > 0):
    /// - Previously non-zero coefficients: emit correction bit after current symbol
    /// - Newly non-zero coefficients: emit run/category symbol + sign bit
    /// - Zero coefficients: increment run counter
    pub fn encode_ac_progressive_refine(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        table_idx: usize,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
        eob_run: &mut u16,
    ) -> Result<()> {
        let ac_table = self.ac_tables[table_idx]
            .ok_or_else(|| Error::internal("AC table not set for refinement"))?;

        let mut k = ss;
        let mut run = 0u32;
        // Fixed-size array for pending refinement bits (max 63 AC coefficients)
        let mut pending_bits = [0u8; 64];
        let mut pending_count = 0usize;

        while k <= se {
            let coef = coeffs[k as usize];
            let abs_coef = coef.unsigned_abs();
            let shifted = abs_coef >> al;

            // Check if this is a previously-coded non-zero coefficient.
            // The first scan encoded this coefficient if (abs >> ah) >= 1, i.e., abs >= 2^ah.
            // Our first scan uses unsigned_abs >> ah != 0 to determine encoding.
            let was_previously_coded = (abs_coef >> ah) != 0;

            // Emit ZRL proactively when run reaches 16, BEFORE collecting more refbits.
            // This ensures each ZRL gets only the refbits from PREV_NZ within its 16-zero span.
            // We must do this check before handling the current coefficient.
            if run >= 16 {
                // Flush EOBRUN if needed (we're about to emit ZRL)
                if *eob_run > 0 {
                    self.emit_eob_run_by_idx(table_idx, *eob_run)?;
                    *eob_run = 0;
                }

                // Emit ZRL (0xF0) with pending correction bits from the 16-zero span
                let (code, len) = ac_table.encode(0xF0);
                self.writer.write_bits(code, len);

                // Output pending correction bits (only those from within this ZRL span)
                for i in 0..pending_count {
                    self.writer.write_bits(pending_bits[i] as u32, 1);
                }
                pending_count = 0;
                run -= 16;
            }

            if was_previously_coded {
                // Already coded - just collect the refinement bit for later
                let refbit = (shifted & 1) as u8;
                pending_bits[pending_count] = refbit;
                pending_count += 1;
            } else if shifted == 1 {
                // New non-zero coefficient at this refinement level
                // Flush EOBRUN if needed
                if *eob_run > 0 {
                    self.emit_eob_run_by_idx(table_idx, *eob_run)?;
                    *eob_run = 0;
                }

                // Note: ZRL emission already handled above when run >= 16,
                // so run is guaranteed to be < 16 here.

                // Emit the coefficient: symbol = (run << 4) | 1
                let symbol = ((run as u8) << 4) | 1;
                let (code, len) = ac_table.encode(symbol);
                self.writer.write_bits(code, len);

                // Output pending correction bits FIRST (before sign bit!)
                // The decoder reads refinement bits while skipping zeros,
                // then reads the sign bit after finding the target position.
                for i in 0..pending_count {
                    self.writer.write_bits(pending_bits[i] as u32, 1);
                }
                pending_count = 0;

                // Sign bit (1 for positive, 0 for negative) comes AFTER refinement bits
                let sign_bit = if coef > 0 { 1u32 } else { 0u32 };
                self.writer.write_bits(sign_bit, 1);

                run = 0;
            } else {
                // Zero coefficient - increment run
                run += 1;
            }

            k += 1;
        }

        // Handle remaining run (EOB case)
        if run > 0 || pending_count > 0 {
            if pending_count > 0 {
                // This block has correction bits - must emit its own EOB, can't join EOB run
                // First flush any pending EOB run from pure-zero blocks
                if *eob_run > 0 {
                    self.emit_eob_run_by_idx(table_idx, *eob_run)?;
                    *eob_run = 0;
                }

                // Emit single EOB for this block
                let ac_table = self.ac_tables[table_idx]
                    .as_ref()
                    .ok_or_else(|| Error::internal("AC table not set for refinement EOB"))?;
                let (eob_code, eob_len) = ac_table.encode(0x00);
                self.writer.write_bits(eob_code, eob_len);

                // Output correction bits for this block's previously-nonzero coefficients
                for i in 0..pending_count {
                    self.writer.write_bits(pending_bits[i] as u32, 1);
                }
            } else {
                // Pure zero block (no correction bits) - can accumulate into EOB run
                *eob_run += 1;
                if *eob_run >= 0x7FFF {
                    self.emit_eob_run_by_idx(table_idx, *eob_run)?;
                    *eob_run = 0;
                }
            }
        }

        Ok(())
    }

    /// Flushes the EOB run for refinement scan at the end.
    pub fn flush_refine_eob(&mut self, table_idx: usize, eob_run: u16) -> Result<()> {
        self.emit_eob_run_by_idx(table_idx, eob_run)
    }

    // ===== Token replay methods for two-pass progressive encoding =====

    /// Replays DC tokens from a token buffer.
    ///
    /// This is used in two-pass progressive encoding to replay tokens
    /// with optimized Huffman tables.
    ///
    /// # Arguments
    /// * `tokens` - Slice of tokens to replay
    /// * `context_to_table` - Maps context IDs to table indices
    pub fn write_dc_tokens(
        &mut self,
        tokens: &[Token],
        context_to_table: &[usize],
        restarts: &[usize],
    ) -> Result<()> {
        let mut next_rst = 0;

        for (i, token) in tokens.iter().enumerate() {
            // Insert restart marker at pre-computed positions.
            if next_rst < restarts.len() && i == restarts[next_rst] {
                self.write_restart_marker();
                self.reset_dc();
                next_rst += 1;
            }

            let table_idx = context_to_table
                .get(token.context as usize)
                .copied()
                .unwrap_or(0);

            let dc_table = self.dc_tables[table_idx]
                .as_ref()
                .ok_or_else(|| Error::internal("DC table not set for token replay"))?;

            // Write the Huffman code for the symbol
            let (code, len) = dc_table.encode(token.symbol);

            // Handle symbol not in table (shouldn't happen if tokenization is correct)
            if len == 0 && token.symbol != 0 {
                return Err(Error::internal(
                    "DC symbol not in Huffman table during replay - histogram may be incomplete",
                ));
            }
            self.writer.write_bits(code, len);

            // Write extra bits if any
            if token.num_extra > 0 {
                self.writer
                    .write_bits(token.extra_bits as u32, token.num_extra);
            }
        }

        Ok(())
    }

    /// Replays AC tokens from a token buffer for a first scan (ah=0).
    ///
    /// # Arguments
    /// * `tokens` - Slice of tokens to replay
    /// * `table_idx` - AC Huffman table index to use
    pub fn write_ac_first_tokens(
        &mut self,
        tokens: &[Token],
        table_idx: usize,
        restarts: &[usize],
    ) -> Result<()> {
        let ac_table = self.ac_tables[table_idx]
            .ok_or_else(|| Error::internal("AC table not set for token replay"))?;

        let mut next_rst = 0;

        for (i, token) in tokens.iter().enumerate() {
            // Insert restart marker at pre-computed positions.
            if next_rst < restarts.len() && i == restarts[next_rst] {
                self.write_restart_marker();
                next_rst += 1;
            }

            // Write the Huffman code for the symbol
            let (code, len) = ac_table.encode(token.symbol);

            // Check if this is an EOB run symbol (upper nibble indicates run size)
            let is_eob_run = (token.symbol & 0x0F) == 0 && token.symbol != 0xF0;

            if len == 0 && !is_eob_run && token.symbol != 0x00 {
                return Err(Error::internal(
                    "AC symbol not in Huffman table during replay",
                ));
            }

            // For missing EOB run symbols, fall back to individual EOBs
            if len == 0 && is_eob_run {
                // Compute actual run: base (1 << log2) + offset (extra_bits)
                let base = 1u16 << (token.symbol >> 4);
                let run = base + token.extra_bits;
                let (eob_code, eob_len) = ac_table.encode(0x00);
                for _ in 0..run {
                    self.writer.write_bits(eob_code, eob_len);
                }
            } else {
                self.writer.write_bits(code, len);

                // Write extra bits if any
                if token.num_extra > 0 {
                    self.writer
                        .write_bits(token.extra_bits as u32, token.num_extra);
                }
            }
        }

        Ok(())
    }

    /// Replays AC refinement tokens from scan info.
    ///
    /// AC refinement scans have a more complex structure where:
    /// - Symbols indicate newly-nonzero coefficients or EOB runs
    /// - Refinement bits for previously-nonzero coefficients are interleaved
    ///
    /// # Arguments
    /// * `scan_info` - Metadata containing ref_tokens, refbits, and eobruns
    /// * `table_idx` - AC Huffman table index to use
    pub fn write_ac_refinement_tokens(
        &mut self,
        scan_info: &ScanTokenInfo,
        table_idx: usize,
        restarts: &[usize],
    ) -> Result<()> {
        let ac_table = self.ac_tables[table_idx]
            .ok_or_else(|| Error::internal("AC table not set for refinement replay"))?;

        let mut refbit_idx = 0;
        let mut eobrun_idx = 0;
        let mut next_rst = 0;

        for (i, ref_token) in scan_info.ref_tokens.iter().enumerate() {
            // Insert restart marker at pre-computed positions.
            if next_rst < restarts.len() && i == restarts[next_rst] {
                self.write_restart_marker();
                next_rst += 1;
            }
            // For AC refinement, symbols may have sign bit in bit 1:
            // - 0x01 = negative newly-nonzero (run=0)
            // - 0x03 = positive newly-nonzero (run=0)
            // Use & 253 to mask out sign bit for Huffman lookup, matching C++:
            //   int symbol = t.symbol & 253;
            let masked_symbol = ref_token.symbol & 253;

            // Write the Huffman code for the masked symbol
            let (code, len) = ac_table.encode(masked_symbol);

            // Check if this is an EOB symbol (low nibble = 0, not ZRL)
            let is_eob = (masked_symbol & 0x0F) == 0 && masked_symbol != 0xF0;

            if len == 0 && masked_symbol != 0x00 {
                // Symbol not in table - for EOB runs, fall back to individual EOBs
                if is_eob {
                    let run_bits = ref_token.symbol >> 4;
                    if run_bits > 0 {
                        // Get the actual run value from eobruns array
                        let run = scan_info.eobruns.get(eobrun_idx).copied().unwrap_or(1);
                        eobrun_idx += 1;

                        let (eob_code, eob_len) = ac_table.encode(0x00);
                        for _ in 0..run {
                            self.writer.write_bits(eob_code, eob_len);
                        }
                    } else {
                        let (eob_code, eob_len) = ac_table.encode(0x00);
                        self.writer.write_bits(eob_code, eob_len);
                    }
                } else {
                    return Err(Error::internal("AC refinement symbol not in Huffman table"));
                }
            } else {
                self.writer.write_bits(code, len);

                // For EOB runs > 1, write the extra bits
                if is_eob && (ref_token.symbol >> 4) > 0 {
                    let run_bits = ref_token.symbol >> 4;
                    if let Some(&run) = scan_info.eobruns.get(eobrun_idx) {
                        let extra = run & ((1 << run_bits) - 1);
                        self.writer.write_bits(extra as u32, run_bits);
                        eobrun_idx += 1;
                    }
                }

                // Write sign bit FIRST for newly-nonzero coefficients
                // Per JPEG spec and libjpeg-turbo: Huffman code, then sign, then refinement bits
                // Newly-nonzero symbols have category 1 (low nibble = 1 or 3 before masking)
                let is_newly_nonzero = (masked_symbol & 0x0F) == 1 && masked_symbol != 0xF0;
                if is_newly_nonzero {
                    // Sign is encoded in bit 1 of the original symbol:
                    // - 0x?1 = negative (bit 1 = 0)
                    // - 0x?3 = positive (bit 1 = 1)
                    // This matches C++: bits = (t.symbol >> 1) & 1;
                    let sign = ((ref_token.symbol >> 1) & 1) as u32;
                    self.writer.write_bits(sign, 1);
                }

                // Write refinement bits AFTER sign bit
                // These are correction bits for previously-nonzero coefficients
                // Get slice upfront to eliminate per-bit bounds checks
                let num_refbits = ref_token.refbits as usize;
                let refbits_end = (refbit_idx + num_refbits).min(scan_info.refbits.len());
                for &bit in &scan_info.refbits[refbit_idx..refbits_end] {
                    self.writer.write_bits(bit as u32, 1);
                }
                refbit_idx = refbits_end;
            }
        }

        Ok(())
    }

    /// Writes a restart marker at the current position.
    pub fn write_restart_marker(&mut self) {
        self.writer.flush();
        self.writer.write_byte_raw(0xFF);
        self.writer.write_byte_raw(0xD0 + self.restart_num);
        self.restart_num = (self.restart_num + 1) & 7;
    }

    /// Returns the current byte position in the output.
    pub fn byte_position(&self) -> usize {
        self.writer.position()
    }

    /// Returns a reference to the current output buffer.
    pub fn as_bytes(&self) -> &[u8] {
        self.writer.as_bytes()
    }
}

impl Default for EntropyEncoder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanEncodeTable;

    #[test]
    fn test_entropy_encoder_new() {
        let encoder = EntropyEncoder::new();
        assert_eq!(encoder.restart_interval, 0);
        assert_eq!(encoder.restart_counter, 0);
        assert_eq!(encoder.restart_num, 0);
        assert_eq!(encoder.prev_dc, [0; 4]);
    }

    #[test]
    fn test_entropy_encoder_default() {
        let encoder = EntropyEncoder::default();
        assert_eq!(encoder.restart_interval, 0);
    }

    #[test]
    fn test_entropy_encoder_set_restart_interval() {
        let mut encoder = EntropyEncoder::new();
        encoder.set_restart_interval(10);
        assert_eq!(encoder.restart_interval, 10);
        assert_eq!(encoder.restart_counter, 10);
    }

    #[test]
    fn test_entropy_encoder_reset_dc() {
        let mut encoder = EntropyEncoder::new();
        encoder.prev_dc = [10, 20, 30, 40];
        encoder.reset_dc();
        assert_eq!(encoder.prev_dc, [0; 4]);
    }

    #[test]
    fn test_entropy_encoder_set_tables() {
        // Create tables first (must outlive encoder)
        let dc_bits: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let dc_values: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let dc_table = HuffmanEncodeTable::from_bits_values(&dc_bits, &dc_values).unwrap();

        let ac_bits: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
        let ac_values: Vec<u8> = (0..162).collect();
        let ac_table = HuffmanEncodeTable::from_bits_values(&ac_bits, &ac_values).unwrap();

        let mut encoder = EntropyEncoder::new();
        encoder.set_dc_table(0, &dc_table);
        assert!(encoder.dc_tables[0].is_some());

        encoder.set_ac_table(0, &ac_table);
        assert!(encoder.ac_tables[0].is_some());

        // Test out of range indices (should be no-op)
        encoder.set_dc_table(5, &dc_table);
        encoder.set_ac_table(5, &ac_table);
    }

    #[test]
    fn test_entropy_encoder_finish() {
        let encoder = EntropyEncoder::new();
        let bytes = encoder.finish();
        assert!(bytes.is_empty() || !bytes.is_empty()); // Just ensure it doesn't panic
    }

    #[test]
    fn test_entropy_encoder_write_bits() {
        let mut encoder = EntropyEncoder::new();
        encoder.write_bits(0b1010, 4);
        let bytes = encoder.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_entropy_encoder_byte_position() {
        let mut encoder = EntropyEncoder::new();
        assert_eq!(encoder.byte_position(), 0);
        encoder.write_bits(0xFF, 8);
        // Position may vary based on internal buffering
        let _ = encoder.byte_position();
    }

    #[test]
    fn test_entropy_encoder_as_bytes() {
        let encoder = EntropyEncoder::new();
        let bytes = encoder.as_bytes();
        assert!(bytes.is_empty());
    }

    /// Helper to create encoder with standard tables (uses static tables)
    fn encoder_with_tables() -> EntropyEncoder<'static> {
        let mut encoder = EntropyEncoder::new();
        encoder.set_dc_table(0, HuffmanEncodeTable::std_dc_luminance());
        encoder.set_ac_table(0, HuffmanEncodeTable::std_ac_luminance());
        encoder
    }

    #[test]
    fn test_encode_block_simd_matches_scalar() {
        // Test with various block patterns
        let test_blocks: Vec<[i16; 64]> = vec![
            // All zeros except DC
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b
            },
            // Sparse block (typical high-quality JPEG)
            {
                let mut b = [0i16; 64];
                b[0] = 512;
                b[1] = -50;
                b[2] = 30;
                b[8] = -20;
                b[9] = 10;
                b
            },
            // Dense block (low quality or high detail)
            {
                let mut b = [0i16; 64];
                for i in 0..32 {
                    b[i] = ((i as i16) * 10) - 150;
                }
                b
            },
            // Block with long zero runs
            {
                let mut b = [0i16; 64];
                b[0] = 200;
                b[1] = 50;
                b[32] = -25; // 30 zeros in between
                b[63] = 5;
                b
            },
            // Block with run > 16 (needs ZRL)
            {
                let mut b = [0i16; 64];
                b[0] = 100;
                b[1] = 10;
                b[20] = -5; // 18 zeros
                b
            },
            // All non-zero (worst case for SIMD)
            {
                let mut b = [0i16; 64];
                for i in 0..64 {
                    b[i] = (i as i16) + 1;
                }
                b
            },
        ];

        for (idx, block) in test_blocks.iter().enumerate() {
            // Encode with scalar
            let mut scalar_encoder = encoder_with_tables();
            scalar_encoder.encode_block_scalar(block, 0, 0, 0).unwrap();
            let scalar_output = scalar_encoder.finish();

            // Encode with SIMD (via encode_block which now calls encode_block_simd)
            let mut simd_encoder = encoder_with_tables();
            simd_encoder.encode_block(block, 0, 0, 0);
            let simd_output = simd_encoder.finish();

            assert_eq!(
                scalar_output,
                simd_output,
                "Block {} mismatch: scalar {} bytes, simd {} bytes",
                idx,
                scalar_output.len(),
                simd_output.len()
            );
        }
    }
}
