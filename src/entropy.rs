//! Entropy encoding for JPEG
//!
//! Handles Huffman encoding of quantized DCT coefficients.

use crate::consts::ZIGZAG;
use crate::huffman::{DerivedTable, FrequencyCounter, HuffTable};

/// Compute the number of bits needed to represent a value (JPEG category).
///
/// For value `v`, returns the minimum number of bits needed to represent |v|.
/// This is used for both DC difference and AC coefficient encoding.
///
/// Returns 0 for v=0, 1 for v in [-1,1], 2 for v in [-3,-2,2,3], etc.
#[inline]
pub fn jpeg_nbits(v: i16) -> u8 {
    let v = v.unsigned_abs();
    if v == 0 {
        0
    } else {
        16 - v.leading_zeros() as u8
    }
}

/// Alias for backwards compatibility
#[inline]
pub fn compute_category(v: i16) -> u8 {
    jpeg_nbits(v)
}

/// Legacy HuffmanTable wrapper for backwards compatibility
pub type HuffmanTable = DerivedTable;

/// Bitstream writer for entropy encoding
pub struct BitWriter {
    buffer: Vec<u8>,
    bit_buffer: u32,
    bits_in_buffer: u8,
}

impl BitWriter {
    /// Create a new bit writer
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Write bits to the stream
    #[inline]
    pub fn write_bits(&mut self, value: u32, count: u8) {
        if std::env::var("DEBUG_BITS").is_ok() {
            let bits: String = (0..count).rev().map(|i| if (value >> i) & 1 == 1 { '1' } else { '0' }).collect();
            eprintln!("    WRITE {} bits: {}", count, bits);
        }
        self.bit_buffer = (self.bit_buffer << count) | (value & ((1 << count) - 1));
        self.bits_in_buffer += count;

        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.buffer.push(byte);

            // Byte stuffing for 0xFF
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }
    }

    /// Flush remaining bits (pad with 1s)
    pub fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            let padding = 8 - self.bits_in_buffer;
            self.write_bits((1 << padding) - 1, padding);
        }
    }

    /// Get the encoded bytes
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Current length in bytes
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy encoder for a single scan
pub struct EntropyEncoder {
    writer: BitWriter,
    dc_table: DerivedTable,
    ac_table: DerivedTable,
    /// Last DC values for each component (up to 4)
    last_dc: [i16; 4],
}

impl EntropyEncoder {
    /// Create a new entropy encoder
    pub fn new(dc_table: DerivedTable, ac_table: DerivedTable) -> Self {
        Self {
            writer: BitWriter::new(),
            dc_table,
            ac_table,
            last_dc: [0; 4],
        }
    }

    /// Encode a single 8x8 block of quantized coefficients
    /// component: 0 = Y, 1 = Cb, 2 = Cr
    pub fn encode_block(&mut self, coefficients: &[i16; 64], component: usize) {
        // DC coefficient (difference from previous block of same component)
        let dc_diff = coefficients[0] - self.last_dc[component];
        self.last_dc[component] = coefficients[0];

        let dc_cat = compute_category(dc_diff);
        let (code, size) = self.dc_table.get_code(dc_cat);
        self.writer.write_bits(code, size);

        if dc_cat > 0 {
            let dc_val = if dc_diff < 0 {
                dc_diff + (1 << dc_cat) - 1
            } else {
                dc_diff
            };
            self.writer.write_bits(dc_val as u32, dc_cat);
        }

        // AC coefficients (accessed in zigzag order)
        let mut run = 0u8;
        for k in 1..64 {
            // ZIGZAG[k] maps zigzag position to natural (row-major) position
            let nat_pos = ZIGZAG[k];
            let ac = coefficients[nat_pos];

            if ac == 0 {
                run += 1;
            } else {
                // Emit ZRL (0xF0) for runs of 16 zeros
                while run >= 16 {
                    let (code, size) = self.ac_table.get_code(0xF0);
                    self.writer.write_bits(code, size);
                    run -= 16;
                }

                // Encode run/size and value
                let ac_cat = compute_category(ac);
                let rs = (run << 4) | ac_cat;
                let (code, size) = self.ac_table.get_code(rs);
                self.writer.write_bits(code, size);

                let ac_val = if ac < 0 {
                    ac + (1 << ac_cat) - 1
                } else {
                    ac
                };
                self.writer.write_bits(ac_val as u32, ac_cat);

                run = 0;
            }
        }

        // EOB if not all 64 coefficients used
        if run > 0 {
            let (code, size) = self.ac_table.get_code(0x00);
            self.writer.write_bits(code, size);
        }
    }

    /// Finish encoding and return the encoded bytes
    pub fn finish(self) -> Vec<u8> {
        self.writer.into_bytes()
    }
}

// =============================================================================
// Symbol Frequency Counting (for Huffman optimization)
// =============================================================================

/// Count symbol frequencies from quantized blocks for Huffman table optimization.
///
/// This is used in the first pass of 2-pass encoding to gather statistics
/// that will be used to generate optimal Huffman tables.
pub struct SymbolCounter {
    /// Last DC value for each component (for differential coding)
    last_dc: [i16; 4],
}

impl Default for SymbolCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl SymbolCounter {
    /// Create a new symbol counter.
    pub fn new() -> Self {
        Self { last_dc: [0; 4] }
    }

    /// Reset DC predictions.
    pub fn reset(&mut self) {
        self.last_dc = [0; 4];
    }

    /// Count symbols in a block, updating frequency counters.
    ///
    /// # Arguments
    /// * `block` - Quantized DCT coefficients (in natural order, not zigzag)
    /// * `component` - Component index for DC prediction (0=Y, 1=Cb, 2=Cr)
    /// * `dc_counter` - Frequency counter for DC symbols
    /// * `ac_counter` - Frequency counter for AC symbols
    pub fn count_block(
        &mut self,
        coefficients: &[i16; 64],
        component: usize,
        dc_counter: &mut FrequencyCounter,
        ac_counter: &mut FrequencyCounter,
    ) {
        // Count DC symbol
        let dc = coefficients[0];
        let diff = dc.wrapping_sub(self.last_dc[component]);
        self.last_dc[component] = dc;

        let nbits = compute_category(diff);
        dc_counter.count(nbits);

        // Count AC symbols (in zigzag order)
        let mut run = 0u8;

        for k in 1..64 {
            let nat_pos = ZIGZAG[k];
            let coef = coefficients[nat_pos];

            if coef == 0 {
                run += 1;
            } else {
                // Count ZRL codes for runs of 16+ zeros
                while run >= 16 {
                    ac_counter.count(0xF0); // ZRL
                    run -= 16;
                }

                let nbits = compute_category(coef);
                let symbol = (run << 4) | nbits;
                ac_counter.count(symbol);
                run = 0;
            }
        }

        // Count EOB if there are trailing zeros
        if run > 0 {
            ac_counter.count(0x00); // EOB
        }
    }
}

// =============================================================================
// Progressive Entropy Encoder
// =============================================================================

/// Progressive entropy encoder for multi-scan JPEG encoding.
///
/// Progressive JPEG uses:
/// - DC scans: encode DC coefficients with optional successive approximation
/// - AC scans: encode AC coefficients in spectral bands (Ss..Se) for one component
/// - Refinement scans: encode additional bits of previously-coded coefficients
pub struct ProgressiveEncoder {
    writer: BitWriter,
    dc_table: DerivedTable,
    ac_table: DerivedTable,
    /// Last DC values for each component
    last_dc: [i16; 4],
    /// End-of-block run count (for EOBRUN encoding)
    eobrun: u16,
    /// Whether to allow extended EOBRUN (requires optimized Huffman tables)
    allow_eobrun: bool,
}

impl ProgressiveEncoder {
    /// Create a new progressive encoder.
    pub fn new(dc_table: DerivedTable, ac_table: DerivedTable) -> Self {
        Self {
            writer: BitWriter::new(),
            dc_table,
            ac_table,
            last_dc: [0; 4],
            eobrun: 0,
            allow_eobrun: true,
        }
    }

    /// Create a progressive encoder for use with standard Huffman tables.
    ///
    /// Standard tables only include EOB (0x00), not extended EOBRUN symbols.
    pub fn new_standard_tables(dc_table: DerivedTable, ac_table: DerivedTable) -> Self {
        Self {
            writer: BitWriter::new(),
            dc_table,
            ac_table,
            last_dc: [0; 4],
            eobrun: 0,
            allow_eobrun: false,
        }
    }

    /// Reset state for a new scan.
    pub fn reset(&mut self) {
        self.last_dc = [0; 4];
        self.eobrun = 0;
    }

    /// Encode a DC first scan (Ah=0).
    ///
    /// # Arguments
    /// * `coefficients` - DCT coefficients in natural order
    /// * `component` - Component index for DC prediction
    /// * `al` - Point transform (successive approximation low bit)
    pub fn encode_dc_first(&mut self, coefficients: &[i16; 64], component: usize, al: u8) {
        // Get DC coefficient with point transform
        let dc = coefficients[0] >> al;

        // Calculate difference from last DC
        let diff = dc.wrapping_sub(self.last_dc[component]);
        self.last_dc[component] = dc;

        // Encode difference
        let nbits = compute_category(diff);
        let (code, size) = self.dc_table.get_code(nbits);
        self.writer.write_bits(code, size);

        // Emit the value bits
        if nbits > 0 {
            let value = if diff < 0 {
                diff + (1 << nbits) - 1
            } else {
                diff
            };
            self.writer.write_bits(value as u32, nbits);
        }
    }

    /// Encode a DC refinement scan (Ah != 0).
    ///
    /// Just outputs a single bit for each block.
    pub fn encode_dc_refine(&mut self, coefficients: &[i16; 64], al: u8) {
        // Output the next bit of DC coefficient
        let bit = ((coefficients[0] >> al) & 1) as u32;
        self.writer.write_bits(bit, 1);
    }

    /// Encode an AC first scan (Ah=0).
    ///
    /// # Arguments
    /// * `coefficients` - DCT coefficients in natural order
    /// * `ss` - Spectral selection start (1..63)
    /// * `se` - Spectral selection end (1..63)
    /// * `al` - Point transform (successive approximation low bit)
    pub fn encode_ac_first(&mut self, coefficients: &[i16; 64], ss: u8, se: u8, al: u8) {
        // Find last non-zero coefficient in this band
        let mut k = se;
        while k >= ss {
            let nat_pos = ZIGZAG[k as usize];
            if (coefficients[nat_pos] >> al) != 0 {
                break;
            }
            if k == ss {
                break;
            }
            k -= 1;
        }
        let kex = k;

        let mut run = 0u32;

        for k in ss..=se {
            let nat_pos = ZIGZAG[k as usize];
            let coef = coefficients[nat_pos] >> al;

            if coef == 0 {
                run += 1;
                continue;
            }

            // Flush any pending EOBRUN
            if self.eobrun > 0 {
                self.flush_eobrun();
            }

            // Emit ZRL codes for runs of 16+ zeros
            while run >= 16 {
                let (code, size) = self.ac_table.get_code(0xF0);
                self.writer.write_bits(code, size);
                run -= 16;
            }

            // Calculate category (number of bits needed)
            let nbits = compute_category(coef);

            // Symbol = (run << 4) | nbits
            let symbol = ((run as u8) << 4) | nbits;
            let (code, size) = self.ac_table.get_code(symbol);
            self.writer.write_bits(code, size);

            // Emit value bits (sign bit first for negative)
            if coef < 0 {
                let value = coef + (1 << nbits) - 1;
                self.writer.write_bits(value as u32, nbits);
            } else {
                self.writer.write_bits(coef as u32, nbits);
            }

            run = 0;

            // Check if we've reached the last non-zero coefficient
            if k == kex {
                break;
            }
        }

        // Emit EOB if we didn't encode all coefficients in the band
        let last_nat_pos = ZIGZAG[se as usize];
        if (coefficients[last_nat_pos] >> al) == 0 || kex < se {
            self.eobrun += 1;
            if !self.allow_eobrun || self.eobrun == 0x7FFF {
                self.flush_eobrun();
            }
        }
    }

    /// Encode an AC refinement scan (Ah != 0).
    ///
    /// This is more complex because we need to:
    /// 1. Output correction bits for previously non-zero coefficients
    /// 2. Output new non-zero coefficients with their correction bits
    ///
    /// # Arguments
    /// * `coefficients` - DCT coefficients in natural order
    /// * `ss` - Spectral selection start (1..63)
    /// * `se` - Spectral selection end (1..63)
    /// * `ah` - Successive approximation high bit (previous Al)
    /// * `al` - Successive approximation low bit (current precision)
    pub fn encode_ac_refine(&mut self, coefficients: &[i16; 64], ss: u8, se: u8, ah: u8, al: u8) {
        let mut run = 0u32;
        let mut pending_bits: Vec<u32> = Vec::new();
        let debug = std::env::var("DEBUG_REFINE").is_ok();

        if debug {
            eprintln!("encode_ac_refine: ss={}, se={}, ah={}, al={}", ss, se, ah, al);
        }

        for k in ss..=se {
            let nat_pos = ZIGZAG[k as usize];
            let coef = coefficients[nat_pos];
            let abs_coef = coef.unsigned_abs();

            // Check if this is a previously-coded non-zero coefficient
            // The previous scan (with Al = current Ah) encoded the coefficient if
            // (coef >> ah) != 0. We must use signed shift to match the encoder's logic.
            // For negative values like -1: (-1 >> 1) = -1 (non-zero), so it was encoded.
            // The abs_coef > 1 check fails for -1 since abs(-1) = 1, but signed shift works.
            if (coef >> ah) != 0 {
                // Already coded - just queue the refinement bit
                // Use absolute value since magnitude bits are what we're refining
                let correction_bit = ((abs_coef >> al) & 1) as u32;
                pending_bits.push(correction_bit);
                if debug {
                    eprintln!("  k={}: prev non-zero coef={}, queue correction bit {}", k, coef, correction_bit);
                }
            } else if (abs_coef >> al) == 1 {
                // New non-zero coefficient
                if debug {
                    eprintln!("  New non-zero at k={}: coef={}, abs={}, ah={}, (coef>>ah)={}, (abs>>al)={}",
                        k, coef, abs_coef, ah, coef >> ah, abs_coef >> al);
                }
                // Flush EOBRUN if needed (EOBRUN is for previous blocks, not this one)
                if self.eobrun > 0 {
                    self.flush_eobrun();
                    // Note: pending_bits are for THIS block, not the EOBRUN blocks
                    // They get output after the coefficient symbol below
                }

                // Emit ZRL for runs of 16
                if debug && run >= 16 {
                    eprintln!("  Before ZRL loop: run={}, pending_bits.len()={}", run, pending_bits.len());
                }
                while run >= 16 {
                    let (code, size) = self.ac_table.get_code(0xF0);
                    if debug {
                        eprintln!("  ZRL 0xF0: code={:#x}, size={}, emitting {} correction bits", code, size, pending_bits.len());
                    }
                    if size == 0 {
                        eprintln!("ERROR: ZRL symbol 0xF0 has no code!");
                    }
                    self.writer.write_bits(code, size);
                    for &bit in &pending_bits {
                        self.writer.write_bits(bit, 1);
                    }
                    pending_bits.clear();
                    run -= 16;
                }

                // Emit the coefficient
                let symbol = ((run as u8) << 4) | 1;
                let (code, size) = self.ac_table.get_code(symbol);
                if debug {
                    eprintln!("  New non-zero at k={}: symbol=0x{:02X}, code={:#x}, size={}, run={}, pending={}",
                        k, symbol, code, size, run, pending_bits.len());
                }
                if size == 0 {
                    eprintln!("ERROR: symbol 0x{:02X} has no code!", symbol);
                }
                self.writer.write_bits(code, size);

                // Sign bit (1 for positive, 0 for negative)
                let sign_bit = if coef < 0 { 0u32 } else { 1u32 };
                self.writer.write_bits(sign_bit, 1);

                // Output pending correction bits
                if debug && !pending_bits.is_empty() {
                    eprintln!("    outputting {} pending correction bits: {:?}", pending_bits.len(), pending_bits);
                }
                for &bit in &pending_bits {
                    self.writer.write_bits(bit, 1);
                }
                pending_bits.clear();
                run = 0;
            } else {
                // Zero coefficient - increment run
                // In refinement, "zero" means (coef >> ah) == 0 AND (abs >> al) != 1
                // This is a TRUE zero (never coded before)
                run += 1;
                if debug && run <= 5 {
                    eprintln!("  k={}: TRUE zero (coef={}), run now {}", k, coef, run);
                }

                // If run reaches 16, we need to emit ZRL immediately with pending correction bits
                // This matches libjpeg-turbo's approach of emitting ZRL as we go
                if run == 16 {
                    // Flush EOBRUN first if needed
                    if self.eobrun > 0 {
                        self.flush_eobrun();
                    }

                    let (code, size) = self.ac_table.get_code(0xF0);
                    if debug {
                        eprintln!("  Incremental ZRL at k={}: emitting {} correction bits", k, pending_bits.len());
                    }
                    self.writer.write_bits(code, size);

                    // Output pending correction bits
                    for &bit in &pending_bits {
                        self.writer.write_bits(bit, 1);
                    }
                    pending_bits.clear();
                    run = 0;
                }
            }
        }

        // Handle remaining run (EOB)
        // EOB is only needed if we have trailing zeros in this block
        if run > 0 {
            self.eobrun += 1;
            if !self.allow_eobrun || self.eobrun == 0x7FFF {
                self.flush_eobrun_with_correction_bits(&pending_bits);
            }
        } else if !pending_bits.is_empty() && self.eobrun > 0 {
            // If we have pending bits but no run, and there's a pending EOBRUN from
            // a previous block, flush it now with our pending bits
            self.flush_eobrun_with_correction_bits(&pending_bits);
        }
        // Note: If run == 0 and eobrun == 0, pending_bits are lost (not carried across blocks)
        // This matches mozjpeg-rs behavior. A full implementation would accumulate BE bits.
    }

    /// Flush the EOB run with pending correction bits.
    fn flush_eobrun_with_correction_bits(&mut self, pending_bits: &[u32]) {
        let debug = std::env::var("DEBUG_REFINE").is_ok();

        if self.eobrun == 0 {
            return;
        }

        // Calculate EOBn symbol (n = floor(log2(EOBRUN)))
        let nbits = 15 - self.eobrun.leading_zeros() as u8;

        // Symbol for EOBn is nbits << 4 (run=0)
        let symbol = nbits << 4;
        let (code, size) = self.ac_table.get_code(symbol);
        if debug {
            eprintln!("  flush_eobrun_with_correction: eobrun={}, symbol=0x{:02X}, pending_bits={:?}",
                self.eobrun, symbol, pending_bits);
        }
        self.writer.write_bits(code, size);

        // Output additional bits for EOBRUN
        if nbits > 0 {
            let mask = (1u16 << nbits) - 1;
            let extra = self.eobrun & mask;
            self.writer.write_bits(extra as u32, nbits);
        }

        // Output correction bits immediately after EOBRUN
        for &bit in pending_bits {
            self.writer.write_bits(bit, 1);
        }

        self.eobrun = 0;
    }

    /// Flush the EOB run to the bitstream.
    fn flush_eobrun(&mut self) {
        let debug = std::env::var("DEBUG_REFINE").is_ok();

        if self.eobrun == 0 {
            return;
        }

        // Calculate EOBn symbol (n = floor(log2(EOBRUN)))
        // EOB0: EOBRUN=1 (nbits=0)
        // EOB1: EOBRUN=2-3 (nbits=1, 1 extra bit)
        // EOB2: EOBRUN=4-7 (nbits=2, 2 extra bits)
        // etc.
        let nbits = 15 - self.eobrun.leading_zeros() as u8;

        // Symbol for EOBn is nbits << 4 (run=0)
        let symbol = nbits << 4;
        let (code, size) = self.ac_table.get_code(symbol);
        if debug {
            eprintln!("  flush_eobrun: eobrun={}, nbits={}, symbol=0x{:02X}, code={:#x}, size={}",
                self.eobrun, nbits, symbol, code, size);
        }
        if size == 0 {
            eprintln!("ERROR: EOBn symbol 0x{:02X} has no code!", symbol);
        }
        self.writer.write_bits(code, size);

        // Output additional bits for EOBRUN
        if nbits > 0 {
            let mask = (1u16 << nbits) - 1;
            let extra = self.eobrun & mask;
            self.writer.write_bits(extra as u32, nbits);
        }

        self.eobrun = 0;
    }

    /// Finish the current scan, flushing any pending EOBRUN.
    pub fn finish_scan(&mut self) {
        self.flush_eobrun();
        self.writer.flush();
    }

    /// Finish encoding and return the encoded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.flush_eobrun();
        self.writer.into_bytes()
    }
}

// =============================================================================
// Progressive Symbol Counter (for Huffman optimization)
// =============================================================================

/// Count symbol frequencies for progressive JPEG scans.
pub struct ProgressiveSymbolCounter {
    /// Last DC value for each component
    last_dc: [i16; 4],
    /// Accumulated EOB run count
    eobrun: u16,
}

impl Default for ProgressiveSymbolCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressiveSymbolCounter {
    /// Create a new progressive symbol counter.
    pub fn new() -> Self {
        Self {
            last_dc: [0; 4],
            eobrun: 0,
        }
    }

    /// Reset state for a new scan.
    pub fn reset(&mut self) {
        self.last_dc = [0; 4];
        self.eobrun = 0;
    }

    /// Count DC symbols for a first scan (Ah=0).
    pub fn count_dc_first(
        &mut self,
        coefficients: &[i16; 64],
        component: usize,
        al: u8,
        dc_counter: &mut FrequencyCounter,
    ) {
        let dc = coefficients[0] >> al;
        let diff = dc.wrapping_sub(self.last_dc[component]);
        self.last_dc[component] = dc;

        let nbits = compute_category(diff);
        dc_counter.count(nbits);
    }

    /// Count AC symbols for a first scan (Ah=0).
    pub fn count_ac_first(
        &mut self,
        coefficients: &[i16; 64],
        ss: u8,
        se: u8,
        al: u8,
        ac_counter: &mut FrequencyCounter,
    ) {
        // Find last non-zero coefficient in this band
        let mut k = se;
        while k >= ss {
            let nat_pos = ZIGZAG[k as usize];
            if (coefficients[nat_pos] >> al) != 0 {
                break;
            }
            if k == ss {
                break;
            }
            k -= 1;
        }
        let kex = k;

        let mut run = 0u32;

        for k in ss..=se {
            let nat_pos = ZIGZAG[k as usize];
            let coef = coefficients[nat_pos] >> al;

            if coef == 0 {
                run += 1;
                continue;
            }

            // Flush any pending EOBRUN
            if self.eobrun > 0 {
                self.flush_eobrun_count(ac_counter);
            }

            // Count ZRL codes for runs of 16+ zeros
            while run >= 16 {
                ac_counter.count(0xF0);
                run -= 16;
            }

            // Symbol = (run << 4) | nbits
            let nbits = compute_category(coef);
            let symbol = ((run as u8) << 4) | nbits;
            ac_counter.count(symbol);

            run = 0;

            if k == kex {
                break;
            }
        }

        // Accumulate EOB
        let last_nat_pos = ZIGZAG[se as usize];
        if (coefficients[last_nat_pos] >> al) == 0 || kex < se {
            self.eobrun += 1;
            if self.eobrun == 0x7FFF {
                self.flush_eobrun_count(ac_counter);
            }
        }
    }

    /// Flush and count the EOBRUN symbol.
    fn flush_eobrun_count(&mut self, ac_counter: &mut FrequencyCounter) {
        if self.eobrun == 0 {
            return;
        }

        let nbits = 15 - self.eobrun.leading_zeros() as u8;
        let symbol = nbits << 4;
        ac_counter.count(symbol);

        self.eobrun = 0;
    }

    /// Finish counting for a scan.
    pub fn finish_scan(&mut self, ac_counter: Option<&mut FrequencyCounter>) {
        if let Some(counter) = ac_counter {
            self.flush_eobrun_count(counter);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b1010, 4);
        writer.write_bits(0b1100, 4);
        let bytes = writer.into_bytes();
        assert_eq!(bytes[0], 0b10101100);
    }

    #[test]
    fn test_byte_stuffing() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        let bytes = writer.into_bytes();
        assert_eq!(bytes.len(), 2);
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x00);
    }
}
