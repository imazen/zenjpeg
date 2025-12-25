//! Entropy encoding for JPEG
//!
//! Handles Huffman encoding of quantized DCT coefficients.

use crate::huffman::{compute_category, HuffmanTable};

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
    dc_table: HuffmanTable,
    ac_table: HuffmanTable,
    last_dc: i16,
}

impl EntropyEncoder {
    /// Create a new entropy encoder
    pub fn new(dc_table: HuffmanTable, ac_table: HuffmanTable) -> Self {
        Self {
            writer: BitWriter::new(),
            dc_table,
            ac_table,
            last_dc: 0,
        }
    }

    /// Encode a single 8x8 block of quantized coefficients
    pub fn encode_block(&mut self, coefficients: &[i16; 64]) {
        // DC coefficient (difference from previous block)
        let dc_diff = coefficients[0] - self.last_dc;
        self.last_dc = coefficients[0];

        let dc_cat = compute_category(dc_diff);
        let (code, size) = self.dc_table.encode(dc_cat);
        self.writer.write_bits(code, size);

        if dc_cat > 0 {
            let dc_val = if dc_diff < 0 {
                dc_diff + (1 << dc_cat) - 1
            } else {
                dc_diff
            };
            self.writer.write_bits(dc_val as u32, dc_cat);
        }

        // AC coefficients (in zigzag order)
        let mut run = 0u8;
        for k in 1..64 {
            let ac = coefficients[k];

            if ac == 0 {
                run += 1;
            } else {
                // Emit ZRL (0xF0) for runs of 16 zeros
                while run >= 16 {
                    let (code, size) = self.ac_table.encode(0xF0);
                    self.writer.write_bits(code, size);
                    run -= 16;
                }

                // Encode run/size and value
                let ac_cat = compute_category(ac);
                let rs = (run << 4) | ac_cat;
                let (code, size) = self.ac_table.encode(rs);
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
            let (code, size) = self.ac_table.encode(0x00);
            self.writer.write_bits(code, size);
        }
    }

    /// Finish encoding and return the encoded bytes
    pub fn finish(self) -> Vec<u8> {
        self.writer.into_bytes()
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
