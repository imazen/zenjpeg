//! Bitstream reading and writing for JPEG.
//!
//! This module provides bit-level I/O with byte stuffing (0xFF -> 0xFF 0x00)
//! as required by JPEG.

use crate::error::{Error, Result};

/// Bit writer for JPEG encoding.
///
/// Accumulates bits and writes bytes with JPEG byte stuffing.
/// Uses a 64-bit buffer to reduce flush frequency in the hot path.
#[derive(Debug)]
pub struct BitWriter {
    /// Output buffer
    buffer: Vec<u8>,
    /// Current bit accumulator (64-bit for reduced flush frequency)
    bit_buffer: u64,
    /// Number of bits in accumulator (0-56, we flush at 32+)
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

        // Accumulate bits into 64-bit buffer
        self.bit_buffer = (self.bit_buffer << count) | (bits as u64);
        self.bits_in_buffer += count;

        // Only flush when we have 32+ bits (reduces loop iterations significantly)
        // This keeps the hot path fast - most write_bits calls won't flush
        if self.bits_in_buffer >= 32 {
            self.flush_bytes();
        }
    }

    /// Flushes complete bytes from the bit buffer.
    /// Marked cold to keep write_bits hot path small.
    #[inline(never)]
    #[cold]
    fn flush_bytes(&mut self) {
        // Flush all complete bytes (typically 4+ bytes at once)
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
        // First flush any complete bytes
        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.buffer.push(byte);

            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }

        // Then pad remaining bits with 1s (JPEG convention)
        if self.bits_in_buffer > 0 {
            let padding = 8 - self.bits_in_buffer;
            let padded = (self.bit_buffer << padding) | ((1u64 << padding) - 1);
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
/// Uses a 64-bit buffer matching the jpegli C++ implementation for safety.
#[derive(Debug)]
pub struct BitReader<'a> {
    /// Input data
    data: &'a [u8],
    /// Current byte position
    position: usize,
    /// Current bit accumulator (64-bit to match C++ jpegli)
    bit_buffer: u64,
    /// Number of bits in accumulator (0-64)
    bits_in_buffer: u8,
    /// Whether we've hit a marker
    marker_found: Option<u8>,
}

/// Saved state of a BitReader for speculative decoding.
#[derive(Clone, Copy)]
pub struct BitReaderState {
    position: usize,
    bit_buffer: u64,
    bits_in_buffer: u8,
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
    ///
    /// According to JPEG spec (ITU-T T.81), when we encounter 0xFF:
    /// - 0xFF 0x00: Byte stuffing, represents the data byte 0xFF
    /// - 0xFF 0xXX (where XX != 0x00 and XX != 0xFF): A marker
    /// - 0xFF 0xFF...: Fill bytes, skip until non-0xFF byte found
    fn read_byte(&mut self) -> Result<u8> {
        // If we've already found a marker, don't read more data
        if self.marker_found.is_some() {
            return Err(Error::EndOfScanData);
        }

        if self.position >= self.data.len() {
            return Err(Error::TruncatedData {
                context: "reading entropy data",
            });
        }

        let byte = self.data[self.position];
        self.position += 1;

        if byte == 0xFF {
            // Skip any fill bytes (consecutive 0xFF)
            while self.position < self.data.len() && self.data[self.position] == 0xFF {
                self.position += 1;
            }

            if self.position >= self.data.len() {
                return Err(Error::TruncatedData {
                    context: "after 0xFF marker prefix",
                });
            }

            let next = self.data[self.position];
            if next == 0x00 {
                // Byte stuffing - skip the 0x00
                self.position += 1;
            } else {
                // Found a marker (including restart markers 0xD0-0xD7)
                // Rewind position to before the FF so the parser can read the marker
                self.position -= 1;
                self.marker_found = Some(next);
                return Err(Error::EndOfScanData);
            }
        }

        Ok(byte)
    }

    /// Fills the bit buffer to have at least `count` bits.
    /// Returns Ok(true) if filled, Ok(false) if end of data but some bits available.
    ///
    /// Matches C++ jpegli FillBitWindow() logic:
    /// - Only fill if bits_in_buffer <= 16 (need more bits)
    /// - Keep filling while bits_in_buffer <= 56 (room for another byte)
    /// - 64-bit buffer ensures no overflow
    fn fill_buffer(&mut self, count: u8) -> Result<bool> {
        // Only refill if we need more bits
        if self.bits_in_buffer < count {
            // Fill while we have room for another byte (56 + 8 = 64)
            while self.bits_in_buffer <= 56 {
                match self.read_byte() {
                    Ok(byte) => {
                        self.bit_buffer = (self.bit_buffer << 8) | (byte as u64);
                        self.bits_in_buffer += 8;
                    }
                    Err(_) => {
                        // Can't read more bytes, but might have enough bits already
                        break;
                    }
                }
                // Stop early if we have enough
                if self.bits_in_buffer >= count {
                    break;
                }
            }
        }
        Ok(self.bits_in_buffer >= count)
    }

    /// Peeks at the next `count` bits without consuming them.
    /// Returns Err if not enough bits available.
    pub fn peek_bits(&mut self, count: u8) -> Result<u32> {
        debug_assert!(count <= 32);
        self.fill_buffer(count)?;
        if self.bits_in_buffer < count {
            // Not enough bits after trying to fill - end of scan data
            return Err(Error::EndOfScanData);
        }
        Ok(((self.bit_buffer >> (self.bits_in_buffer - count)) & ((1u64 << count) - 1)) as u32)
    }

    /// Reads `count` bits from the stream.
    pub fn read_bits(&mut self, count: u8) -> Result<u32> {
        self.fill_buffer(count)?;
        if self.bits_in_buffer < count {
            // Not enough bits after trying to fill - end of scan data
            return Err(Error::EndOfScanData);
        }
        let bits =
            ((self.bit_buffer >> (self.bits_in_buffer - count)) & ((1u64 << count) - 1)) as u32;
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

    /// Saves the current reader state for potential rollback.
    #[must_use]
    pub fn save_state(&self) -> BitReaderState {
        BitReaderState {
            position: self.position,
            bit_buffer: self.bit_buffer,
            bits_in_buffer: self.bits_in_buffer,
            marker_found: self.marker_found,
        }
    }

    /// Restores a previously saved state.
    pub fn restore_state(&mut self, state: BitReaderState) {
        self.position = state.position;
        self.bit_buffer = state.bit_buffer;
        self.bits_in_buffer = state.bits_in_buffer;
        self.marker_found = state.marker_found;
    }

    /// Reads and verifies a restart marker.
    ///
    /// Call this after aligning to byte boundary when a restart marker is expected.
    /// Returns Ok(()) if the expected marker was found, Err otherwise.
    ///
    /// # Arguments
    /// * `expected_num` - Expected restart marker number (0-7)
    pub fn read_restart_marker(&mut self, expected_num: u8) -> Result<()> {
        // Clear the marker_found flag since we're explicitly reading the marker
        self.marker_found = None;

        // Read first byte - should be 0xFF
        if self.position >= self.data.len() {
            return Err(Error::InvalidJpegData {
                reason: "unexpected end of data before restart marker",
            });
        }
        let first = self.data[self.position];
        if first != 0xFF {
            return Err(Error::InvalidJpegData {
                reason: "expected 0xFF for restart marker",
            });
        }
        self.position += 1;

        // Read second byte - should be 0xD0 + expected_num
        if self.position >= self.data.len() {
            return Err(Error::InvalidJpegData {
                reason: "unexpected end of data in restart marker",
            });
        }
        let second = self.data[self.position];
        let expected_marker = 0xD0 + (expected_num & 7);
        if second != expected_marker {
            // Check if it's a different restart marker (resync case)
            if (0xD0..=0xD7).contains(&second) {
                return Err(Error::InvalidJpegData {
                    reason: "restart marker sequence mismatch",
                });
            }
            return Err(Error::InvalidJpegData {
                reason: "expected restart marker not found",
            });
        }
        self.position += 1;

        Ok(())
    }

    /// Reads a raw byte (assumes byte-aligned).
    pub fn read_byte_raw(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::TruncatedData {
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
