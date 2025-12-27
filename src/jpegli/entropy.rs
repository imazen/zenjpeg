//! Entropy coding for JPEG.
//!
//! This module provides Huffman-based entropy encoding and decoding
//! for JPEG DCT coefficients.

use crate::jpegli::bitstream::{BitReader, BitWriter};
use crate::jpegli::consts::DCT_BLOCK_SIZE;
use crate::jpegli::error::{Error, Result};
use crate::jpegli::huffman::{HuffmanDecodeTable, HuffmanEncodeTable};
use crate::jpegli::huffman_opt::{ScanTokenInfo, Token};

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

    // Clamp category to valid range (1-15 for JPEG)
    // category 16+ would overflow i16
    if category > 15 {
        return bits as i16;
    }

    let half = 1u16 << (category - 1);
    if bits >= half {
        bits as i16
    } else {
        // Calculate (bits) - (2^category - 1) without overflow
        // Using i32 to avoid overflow
        let max_val = (1i32 << category) - 1;
        ((bits as i32) - max_val) as i16
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

    // ===== Progressive encoding methods =====

    /// Writes raw bits to the output (for progressive refinement).
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
                .ok_or(Error::InternalError {
                    reason: "DC table not set",
                })?;

            let dc_cat = category(dc_diff);
            let (code, len) = dc_table.encode(dc_cat);
            if len == 0 {
                return Err(Error::InternalError {
                    reason: "DC symbol not in Huffman table",
                });
            }
            self.writer.write_bits(code, len);

            if dc_cat > 0 {
                let additional = additional_bits(dc_diff);
                self.writer.write_bits(additional as u32, dc_cat);
            }
        }

        Ok(())
    }

    /// Encodes AC coefficients for progressive first scan.
    ///
    /// Returns the EOB run that should be accumulated.
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
        let mut last_nz = ss as usize;
        for k in (ss as usize..=se as usize).rev() {
            let coef = coeffs[k] >> al;
            if coef != 0 {
                last_nz = k;
                break;
            }
        }

        // Check if block is all zeros in this range
        let all_zero = (ss as usize..=se as usize).all(|k| (coeffs[k] >> al) == 0);

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
            .ok_or(Error::InternalError {
                reason: "AC table not set",
            })?;

        // Encode non-zero coefficients
        let mut r = 0u8; // Run of zeros
        for k in ss as usize..=last_nz {
            let coef = coeffs[k] >> al;

            if coef == 0 {
                r += 1;
                continue;
            }

            // Emit ZRL (16 zeros) tokens if needed
            while r >= 16 {
                let (code, len) = ac_table.encode(0xF0);
                self.writer.write_bits(code, len);
                r -= 16;
            }

            // Encode the coefficient
            let abs_coef = coef.unsigned_abs();
            let nbits = 16 - abs_coef.leading_zeros();
            let symbol = ((r as u16) << 4) | nbits as u16;

            let (code, len) = ac_table.encode(symbol as u8);
            if len == 0 {
                return Err(Error::InternalError {
                    reason: "AC symbol not in Huffman table",
                });
            }
            self.writer.write_bits(code, len);

            // Additional bits (magnitude and sign)
            let bits = if coef < 0 {
                (abs_coef - 1) as u32
            } else {
                abs_coef as u32
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
            .ok_or(Error::InternalError {
                reason: "AC table not set",
            })?;

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
    pub fn encode_ac_progressive_refine(
        &mut self,
        coeffs: &[i16; DCT_BLOCK_SIZE],
        table_idx: usize,
        ss: u8,
        se: u8,
        al: u8,
        ah: u8,
        eob_run: &mut u16,
        pending_bits: &mut Vec<u8>,
    ) -> Result<()> {
        // Clear pending bits at start of each block
        pending_bits.clear();

        let mut r = 0u16; // Run of never-been-nonzero coefficients
        let mut found_new_nonzero = false;

        // Collect information about what needs to be encoded
        #[derive(Clone)]
        enum Action {
            EmitEobRun(u16),
            EmitZrl(Vec<u8>), // with pending bits to emit after
            EmitCoef {
                symbol: u8,
                sign: u32,
                pending: Vec<u8>,
            },
        }
        let mut actions: Vec<Action> = Vec::new();

        for k in ss as usize..=se as usize {
            let coef = coeffs[k];
            let abs_coef = coef.abs();
            // Check if this coefficient was already transmitted (had non-zero bits above Al)
            let was_nonzero = (abs_coef >> ah) != 0;

            if was_nonzero {
                // Already non-zero from previous pass - emit refinement bit after current symbol
                let bit = ((abs_coef >> al) & 1) as u8;
                pending_bits.push(bit);
            } else {
                // Check if coefficient becomes non-zero in this pass
                let newly_nonzero = ((abs_coef >> al) & 1) != 0;

                if newly_nonzero {
                    // Newly non-zero coefficient - first emit any pending EOB run
                    if *eob_run > 0 {
                        actions.push(Action::EmitEobRun(*eob_run));
                        *eob_run = 0;
                    }

                    found_new_nonzero = true;

                    // Handle ZRL for runs of 16+ zeros
                    while r >= 16 {
                        actions.push(Action::EmitZrl(pending_bits.clone()));
                        pending_bits.clear();
                        r -= 16;
                    }

                    // Emit the coefficient: symbol = (run, 1) plus sign bit
                    let symbol = ((r as u8) << 4) | 1;
                    let sign_bit = if coef > 0 { 1u32 } else { 0u32 };

                    actions.push(Action::EmitCoef {
                        symbol,
                        sign: sign_bit,
                        pending: pending_bits.clone(),
                    });
                    pending_bits.clear();

                    r = 0;
                } else {
                    // Still zero - increment run
                    r += 1;
                }
            }
        }

        // Handle EOB if we didn't end with a newly non-zero coefficient
        if !found_new_nonzero || r > 0 || !pending_bits.is_empty() {
            // We need to emit an EOB with pending bits
            if !pending_bits.is_empty() {
                // Must emit EOB now to include pending bits
                if *eob_run > 0 {
                    actions.push(Action::EmitEobRun(*eob_run));
                    *eob_run = 0;
                }
                // Emit single EOB for this block - will be handled with pending bits
                // Mark with special run=1 and include pending bits
            } else {
                // No pending bits - accumulate EOB run
                *eob_run += 1;
                if *eob_run >= 0x7FFF {
                    actions.push(Action::EmitEobRun(*eob_run));
                    *eob_run = 0;
                }
            }
        }

        // Now execute all actions
        let ac_table = self.ac_tables[table_idx]
            .as_ref()
            .ok_or(Error::InternalError {
                reason: "AC table not set",
            })?
            .clone();

        for action in actions {
            match action {
                Action::EmitEobRun(run) => {
                    if run > 0 {
                        let nbits = if run == 1 {
                            0
                        } else {
                            31 - (run as u32).leading_zeros()
                        };
                        let symbol = (nbits << 4) as u8;
                        let (code, len) = ac_table.encode(symbol);

                        if len == 0 && symbol != 0x00 {
                            let (eob_code, eob_len) = ac_table.encode(0x00);
                            for _ in 0..run {
                                self.writer.write_bits(eob_code, eob_len);
                            }
                        } else {
                            self.writer.write_bits(code, len);
                            if nbits > 0 {
                                let extra_bits = run & ((1 << nbits) - 1);
                                self.writer.write_bits(extra_bits as u32, nbits as u8);
                            }
                        }
                    }
                }
                Action::EmitZrl(bits) => {
                    let (code, len) = ac_table.encode(0xF0);
                    self.writer.write_bits(code, len);
                    for bit in bits {
                        self.writer.write_bits(bit as u32, 1);
                    }
                }
                Action::EmitCoef {
                    symbol,
                    sign,
                    pending,
                } => {
                    let (code, len) = ac_table.encode(symbol);
                    self.writer.write_bits(code, len);
                    self.writer.write_bits(sign, 1);
                    for bit in pending {
                        self.writer.write_bits(bit as u32, 1);
                    }
                }
            }
        }

        // Emit final EOB with pending bits if needed
        if !pending_bits.is_empty() {
            let (code, len) = ac_table.encode(0x00);
            self.writer.write_bits(code, len);
            for &bit in pending_bits.iter() {
                self.writer.write_bits(bit as u32, 1);
            }
            pending_bits.clear();
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
    pub fn write_dc_tokens(&mut self, tokens: &[Token], context_to_table: &[usize]) -> Result<()> {
        for token in tokens {
            let table_idx = context_to_table
                .get(token.context as usize)
                .copied()
                .unwrap_or(0);

            let dc_table = self.dc_tables[table_idx]
                .as_ref()
                .ok_or(Error::InternalError {
                    reason: "DC table not set for token replay",
                })?;

            // Write the Huffman code for the symbol
            let (code, len) = dc_table.encode(token.symbol);

            // Handle symbol not in table (shouldn't happen if tokenization is correct)
            if len == 0 && token.symbol != 0 {
                // This can happen for very small images where not all DC categories appear.
                // Fall back to encoding using a longer code that's guaranteed to exist.
                // For DC, category 0 always exists, so we can't encode missing categories.
                // This is a limitation - the tokenization should ensure all used symbols exist.
                return Err(Error::InternalError {
                    reason:
                        "DC symbol not in Huffman table during replay - histogram may be incomplete",
                });
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
    pub fn write_ac_first_tokens(&mut self, tokens: &[Token], table_idx: usize) -> Result<()> {
        let ac_table = self.ac_tables[table_idx]
            .as_ref()
            .ok_or(Error::InternalError {
                reason: "AC table not set for token replay",
            })?
            .clone();

        for token in tokens {
            // Write the Huffman code for the symbol
            let (code, len) = ac_table.encode(token.symbol);

            // Check if this is an EOB run symbol (upper nibble indicates run size)
            let is_eob_run = (token.symbol & 0x0F) == 0 && token.symbol != 0xF0;

            if len == 0 && !is_eob_run && token.symbol != 0x00 {
                return Err(Error::InternalError {
                    reason: "AC symbol not in Huffman table during replay",
                });
            }

            // For missing EOB run symbols, fall back to individual EOBs
            if len == 0 && is_eob_run {
                let run = 1u16 << (token.symbol >> 4);
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
    ) -> Result<()> {
        let ac_table = self.ac_tables[table_idx]
            .as_ref()
            .ok_or(Error::InternalError {
                reason: "AC table not set for refinement replay",
            })?
            .clone();

        let mut refbit_idx = 0;
        let mut eobrun_idx = 0;

        for ref_token in &scan_info.ref_tokens {
            // Write the Huffman code for the symbol
            let (code, len) = ac_table.encode(ref_token.symbol);

            // Check if this is an EOB symbol
            let is_eob = (ref_token.symbol & 0x0F) == 0 && ref_token.symbol != 0xF0;

            if len == 0 && ref_token.symbol != 0x00 {
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
                    return Err(Error::InternalError {
                        reason: "AC refinement symbol not in Huffman table",
                    });
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

                // Write the sign bit for newly-nonzero coefficients
                let symbol_cat = ref_token.symbol & 0x0F;
                if symbol_cat == 1 && ref_token.symbol != 0xF0 {
                    // This is a newly-nonzero coefficient, write sign bit
                    if refbit_idx < scan_info.refbits.len() {
                        let sign = scan_info.refbits[refbit_idx] as u32;
                        self.writer.write_bits(sign, 1);
                        refbit_idx += 1;
                    }
                }
            }

            // Write refinement bits that follow this token
            for _ in 0..ref_token.refbits {
                if refbit_idx < scan_info.refbits.len() {
                    let bit = scan_info.refbits[refbit_idx] as u32;
                    self.writer.write_bits(bit, 1);
                    refbit_idx += 1;
                }
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

    /// Safely gets a DC table, handling out-of-bounds indices.
    fn get_dc_table(&self, idx: usize) -> Result<HuffmanDecodeTable> {
        self.dc_tables
            .get(idx)
            .and_then(|t| t.clone())
            .ok_or(Error::InternalError {
                reason: "DC table not set or invalid index",
            })
    }

    /// Safely gets an AC table, handling out-of-bounds indices.
    fn get_ac_table(&self, idx: usize) -> Result<HuffmanDecodeTable> {
        self.ac_tables
            .get(idx)
            .and_then(|t| t.clone())
            .ok_or(Error::InternalError {
                reason: "AC table not set or invalid index",
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
        let dc_table = self.get_dc_table(dc_table_idx)?;
        let ac_table = self.get_ac_table(ac_table_idx)?;

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

        let dc_cat = self.decode_huffman(&dc_table)?;
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
            let symbol = self.decode_huffman(&ac_table)?;
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
                        if coeffs[k] > 0 {
                            coeffs[k] += bit_val;
                        } else {
                            coeffs[k] -= bit_val;
                        }
                    }
                }
            }
            *eob_run -= 1;
            return Ok(());
        }

        let mut k = ss as usize;
        while k <= se as usize {
            let symbol = self.decode_huffman(&ac_table)?;
            let run = symbol >> 4;
            let size = symbol & 0x0F;

            let mut num_zeros_to_skip = run as usize;

            if size == 0 {
                if run == 15 {
                    // ZRL in refinement - skip 16 zeros (not 15!)
                    // The run nibble is 15, but ZRL means 16 zeros.
                    num_zeros_to_skip = 16;
                } else {
                    // EOB run
                    if run == 0 {
                        // Single EOB - apply refinement to remaining nonzero coeffs
                        for j in k..=se as usize {
                            if coeffs[j] != 0 {
                                let bit = self.reader.read_bits(1)? as i16;
                                if bit != 0 && (coeffs[j] & bit_val) == 0 {
                                    if coeffs[j] > 0 {
                                        coeffs[j] += bit_val;
                                    } else {
                                        coeffs[j] -= bit_val;
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
                                    if coeffs[j] > 0 {
                                        coeffs[j] += bit_val;
                                    } else {
                                        coeffs[j] -= bit_val;
                                    }
                                }
                            }
                        }
                        return Ok(());
                    }
                }
            }

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
                        if coeffs[k] > 0 {
                            coeffs[k] += bit_val;
                        } else {
                            coeffs[k] -= bit_val;
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

            if size != 0 && k <= se as usize {
                // Place newly-nonzero coefficient
                let sign_bit = self.reader.read_bits(1)? as i16;
                if sign_bit != 0 {
                    coeffs[k] = bit_val;
                } else {
                    coeffs[k] = -bit_val;
                }
                k += 1; // Move past the placed coefficient
            }
            // For ZRL (size==0), k already points past the 16 zeros we skipped
        }

        Ok(())
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
