//! Bitstream reading and writing for JPEG.
//!
//! This module provides bit-level I/O with byte stuffing (0xFF -> 0xFF 0x00)
//! as required by JPEG.

use crate::jpegli::error::{Error, Result};

/// Bit writer for JPEG encoding.
///
/// Accumulates bits and writes bytes with JPEG byte stuffing.
#[derive(Debug)]
pub struct BitWriter {
    /// Output buffer
    buffer: Vec<u8>,
    /// Current bit accumulator
    bit_buffer: u32,
    /// Number of bits in accumulator (0-32)
    bits_in_buffer: u8,
}

impl BitWriter {
    /// Creates a new bit writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Creates a new bit writer with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Writes bits to the stream.
    ///
    /// # Arguments
    /// * `bits` - The bits to write (right-aligned)
    /// * `count` - Number of bits to write (1-24)
    #[inline]
    pub fn write_bits(&mut self, bits: u32, count: u8) {
        debug_assert!(count <= 24);
        debug_assert!(bits < (1 << count) || count == 0);

        self.bit_buffer = (self.bit_buffer << count) | bits;
        self.bits_in_buffer += count;

        // Flush complete bytes
        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.buffer.push(byte);

            // Byte stuffing: 0xFF must be followed by 0x00
            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }
    }

    /// Writes a single byte directly (no bit stuffing).
    #[inline]
    pub fn write_byte_raw(&mut self, byte: u8) {
        self.buffer.push(byte);
    }

    /// Writes bytes directly (no bit stuffing).
    pub fn write_bytes_raw(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Writes a 16-bit value in big-endian order (no bit stuffing).
    #[inline]
    pub fn write_u16_be(&mut self, value: u16) {
        self.buffer.push((value >> 8) as u8);
        self.buffer.push(value as u8);
    }

    /// Flushes any remaining bits, padding with 1s.
    pub fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            // Pad with 1s (JPEG convention)
            let padding = 8 - self.bits_in_buffer;
            let padded = (self.bit_buffer << padding) | ((1 << padding) - 1);
            let byte = padded as u8;
            self.buffer.push(byte);

            if byte == 0xFF {
                self.buffer.push(0x00);
            }

            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    /// Returns the accumulated bytes.
    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        self.buffer
    }

    /// Returns a reference to the current buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Returns the current byte position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Bit reader for JPEG decoding.
///
/// Reads bits with byte unstuffing (0xFF 0x00 -> 0xFF).
#[derive(Debug)]
pub struct BitReader<'a> {
    /// Input data
    data: &'a [u8],
    /// Current byte position
    position: usize,
    /// Current bit accumulator
    bit_buffer: u32,
    /// Number of bits in accumulator
    bits_in_buffer: u8,
    /// Whether we've hit a marker
    marker_found: Option<u8>,
}

impl<'a> BitReader<'a> {
    /// Creates a new bit reader.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            bit_buffer: 0,
            bits_in_buffer: 0,
            marker_found: None,
        }
    }

    /// Reads the next byte, handling byte unstuffing.
    fn read_byte(&mut self) -> Result<u8> {
        // If we've already found a marker, don't read more data
        if self.marker_found.is_some() {
            return Err(Error::UnexpectedEof {
                context: "marker found, end of entropy data",
            });
        }

        if self.position >= self.data.len() {
            return Err(Error::UnexpectedEof {
                context: "reading bit stream",
            });
        }

        let byte = self.data[self.position];
        self.position += 1;

        if byte == 0xFF {
            if self.position >= self.data.len() {
                return Err(Error::UnexpectedEof {
                    context: "after 0xFF byte",
                });
            }

            let next = self.data[self.position];
            if next == 0x00 {
                // Byte stuffing - skip the 0x00
                self.position += 1;
            } else if next >= 0xD0 && next <= 0xD7 {
                // Restart marker - skip it and continue
                self.position += 1;
            } else {
                // Found a marker - don't consume these bytes as data
                // Rewind position to before the FF so the parser can read the marker
                self.position -= 1;
                self.marker_found = Some(next);
                return Err(Error::UnexpectedEof {
                    context: "marker found, end of entropy data",
                });
            }
        }

        Ok(byte)
    }

    /// Fills the bit buffer to have at least `count` bits.
    /// Returns Ok(true) if filled, Ok(false) if end of data but some bits available.
    fn fill_buffer(&mut self, count: u8) -> Result<bool> {
        while self.bits_in_buffer < count {
            match self.read_byte() {
                Ok(byte) => {
                    self.bit_buffer = (self.bit_buffer << 8) | (byte as u32);
                    self.bits_in_buffer += 8;
                }
                Err(_) => {
                    // Can't read more bytes, but might have enough bits already
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Peeks at the next `count` bits without consuming them.
    /// Returns Err if not enough bits available.
    pub fn peek_bits(&mut self, count: u8) -> Result<u32> {
        debug_assert!(count <= 24);
        self.fill_buffer(count)?;
        if self.bits_in_buffer < count {
            return Err(Error::UnexpectedEof {
                context: "not enough bits in buffer",
            });
        }
        Ok((self.bit_buffer >> (self.bits_in_buffer - count)) & ((1 << count) - 1))
    }

    /// Reads `count` bits from the stream.
    pub fn read_bits(&mut self, count: u8) -> Result<u32> {
        self.fill_buffer(count)?;
        if self.bits_in_buffer < count {
            return Err(Error::UnexpectedEof {
                context: "not enough bits to read",
            });
        }
        let bits = (self.bit_buffer >> (self.bits_in_buffer - count)) & ((1 << count) - 1);
        self.bits_in_buffer -= count;
        Ok(bits)
    }

    /// Skips `count` bits.
    pub fn skip_bits(&mut self, count: u8) {
        if count <= self.bits_in_buffer {
            self.bits_in_buffer -= count;
        }
    }

    /// Reads a single bit.
    #[inline]
    pub fn read_bit(&mut self) -> Result<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Reads a signed value with sign extension.
    ///
    /// JPEG encodes signed values where values < 2^(bits-1) are negative.
    pub fn read_signed(&mut self, bits: u8) -> Result<i16> {
        if bits == 0 {
            return Ok(0);
        }

        let value = self.read_bits(bits)? as i16;
        let half = 1i16 << (bits - 1);

        if value < half {
            // Negative value
            Ok(value - (2 * half - 1))
        } else {
            Ok(value)
        }
    }

    /// Aligns to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        self.bits_in_buffer = 0;
    }

    /// Reads a raw byte (assumes byte-aligned).
    pub fn read_byte_raw(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::UnexpectedEof {
                context: "reading raw byte",
            });
        }
        let byte = self.data[self.position];
        self.position += 1;
        Ok(byte)
    }

    /// Reads a 16-bit big-endian value (assumes byte-aligned).
    pub fn read_u16_be(&mut self) -> Result<u16> {
        let high = self.read_byte_raw()? as u16;
        let low = self.read_byte_raw()? as u16;
        Ok((high << 8) | low)
    }

    /// Returns any marker that was encountered.
    #[must_use]
    pub fn marker_found(&self) -> Option<u8> {
        self.marker_found
    }

    /// Returns the current byte position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns remaining bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_read_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b1100, 4);
        writer.write_bits(0b1, 1);
        let bytes = writer.into_bytes();

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1100);
        assert_eq!(reader.read_bits(1).unwrap(), 0b1);
    }

    #[test]
    fn test_byte_stuffing() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        let bytes = writer.into_bytes();

        // 0xFF should be stuffed with 0x00, then padded with 1s
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0x00);
    }

    #[test]
    fn test_byte_unstuffing() {
        // 0xFF 0x00 should be read as 0xFF
        let data = [0xFF, 0x00, 0xAB];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(8).unwrap(), 0xFF);
        assert_eq!(reader.read_bits(8).unwrap(), 0xAB);
    }

    #[test]
    fn test_signed_values() {
        // Test JPEG signed value encoding
        // Data: 0b01000000 = 0x40, reading MSB first: bit0=0, bit1=1
        let data = [0b0100_0000]; // First bit = 0 (means -1), second bit = 1 (means +1)
        let mut reader = BitReader::new(&data);

        // 1-bit category: 0 -> -1, 1 -> 1
        assert_eq!(reader.read_signed(1).unwrap(), -1);
        assert_eq!(reader.read_signed(1).unwrap(), 1);
    }
}
