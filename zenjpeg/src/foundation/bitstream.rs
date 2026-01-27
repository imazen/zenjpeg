//! Bitstream reading and writing for JPEG.
//!
//! This module provides bit-level I/O with byte stuffing (0xFF -> 0xFF 0x00)
//! as required by JPEG.

#![allow(dead_code)]

use crate::error::{Error, Result, ScanRead, ScanResult};
use crate::foundation::instrumented_vec::{ProfiledVec, ProfiledVecExt};

/// Bit writer for JPEG encoding.
///
/// Accumulates bits and writes bytes with JPEG byte stuffing.
/// Uses a 64-bit buffer to reduce flush frequency in the hot path.
#[derive(Debug)]
pub struct BitWriter {
    /// Output buffer
    buffer: ProfiledVec<u8>,
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
            buffer: ProfiledVec::with_capacity_profiled(0, "BitWriter::new"),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Creates a new bit writer with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: ProfiledVec::with_capacity_profiled(capacity, "BitWriter::with_capacity"),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    /// Writes bits to the stream.
    ///
    /// # Arguments
    /// * `bits` - The bits to write (right-aligned)
    /// * `count` - Number of bits to write (1-24)
    #[inline(always)]
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

    /// Writes Huffman code and extra bits in a single operation.
    ///
    /// This is an optimization that combines two write_bits calls into one,
    /// reducing function call overhead in the entropy coding hot path.
    ///
    /// # Arguments
    /// * `code` - Huffman code (right-aligned)
    /// * `code_len` - Length of Huffman code in bits
    /// * `extra` - Extra bits to write after code (right-aligned)
    /// * `extra_len` - Length of extra bits
    #[inline(always)]
    pub fn write_code_and_extra(&mut self, code: u32, code_len: u8, extra: u16, extra_len: u8) {
        debug_assert!(code_len <= 16);
        debug_assert!(extra_len <= 16);
        let total_len = code_len + extra_len;
        debug_assert!(total_len <= 32);

        // Combine code and extra bits: code in high bits, extra in low bits
        let combined = ((code as u64) << extra_len) | (extra as u64);

        self.bit_buffer = (self.bit_buffer << total_len) | combined;
        self.bits_in_buffer += total_len;

        if self.bits_in_buffer >= 32 {
            self.flush_bytes();
        }
    }

    /// Flushes complete bytes from the bit buffer.
    /// Marked cold to keep write_bits hot path small.
    ///
    /// Uses the same algorithm as C++ jpegli:
    /// - Process 64 bits (8 bytes) at a time for efficiency
    /// - Use SWAR bit trick to detect 0xFF bytes without branching per byte
    /// - Fast path: direct 8-byte store when no 0xFF present
    #[inline(never)]
    #[cold]
    fn flush_bytes(&mut self) {
        // Process 64 bits at a time (matching C++ jpegli's DischargeBitBuffer)
        while self.bits_in_buffer >= 64 {
            self.bits_in_buffer -= 64;
            let word = self.bit_buffer; // All 64 bits
            self.emit_8_bytes(word);
        }

        // Handle 32-56 bits: extract top 32-56 bits, emit as many complete bytes as possible
        // This bridges between our 64-bit accumulator and the threshold
        while self.bits_in_buffer >= 32 {
            self.bits_in_buffer -= 32;
            let word = (self.bit_buffer >> self.bits_in_buffer) as u32;
            self.emit_4_bytes(word);
        }

        // Handle remaining bytes (0-3)
        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.emit_byte(byte);
        }
    }

    /// Emits a single byte with 0xFF stuffing.
    #[inline(always)]
    fn emit_byte(&mut self, byte: u8) {
        self.buffer.push(byte);
        if byte == 0xFF {
            self.buffer.push(0x00);
        }
    }

    /// Emits 4 bytes with 0xFF stuffing using SWAR bit trick.
    #[inline(always)]
    fn emit_4_bytes(&mut self, word: u32) {
        if !has_byte_0xff_u32(word) {
            // Fast path: no 0xFF bytes, write directly as big-endian
            self.buffer.extend_from_slice(&word.to_be_bytes());
        } else {
            // Slow path: has 0xFF, emit byte-by-byte
            self.emit_byte((word >> 24) as u8);
            self.emit_byte((word >> 16) as u8);
            self.emit_byte((word >> 8) as u8);
            self.emit_byte(word as u8);
        }
    }

    /// Emits 8 bytes with 0xFF stuffing using SWAR bit trick.
    /// Matches C++ jpegli's DischargeBitBuffer exactly.
    #[inline(always)]
    fn emit_8_bytes(&mut self, word: u64) {
        if !has_byte_0xff_u64(word) {
            // Fast path: no 0xFF bytes, write directly as big-endian
            self.buffer.extend_from_slice(&word.to_be_bytes());
        } else {
            // Slow path: has at least one 0xFF, emit byte-by-byte
            self.emit_byte((word >> 56) as u8);
            self.emit_byte((word >> 48) as u8);
            self.emit_byte((word >> 40) as u8);
            self.emit_byte((word >> 32) as u8);
            self.emit_byte((word >> 24) as u8);
            self.emit_byte((word >> 16) as u8);
            self.emit_byte((word >> 8) as u8);
            self.emit_byte(word as u8);
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
    ///
    /// When `alloc-instrument` feature is enabled, logs utilization stats.
    #[must_use]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.flush();
        #[cfg(feature = "alloc-instrument")]
        {
            let stats = self.buffer.stats();
            let wasted = stats.wasted_bytes();
            if wasted >= 1024 || stats.realloc_count > 0 {
                eprintln!(
                    "[alloc] {}: len={} cap={} ({:.0}% util, {}B wasted) init={} reallocs={}",
                    stats.context,
                    stats.final_len,
                    stats.final_capacity,
                    stats.utilization_pct(),
                    wasted,
                    stats.initial_capacity,
                    stats.realloc_count,
                );
            }
        }
        self.buffer.into()
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

    /// Flushes bits and appends to an external buffer without EOI marker.
    ///
    /// This is used by bounded-memory streaming to write scan data incrementally.
    /// NOTE: This pads the final byte with 1s, so only call at the end of the scan!
    pub fn flush_without_eoi(&mut self, output: &mut Vec<u8>) -> crate::error::Result<()> {
        self.flush();
        output.try_reserve(self.buffer.len()).map_err(|_| {
            crate::error::Error::allocation_failed(self.buffer.len(), "flush_without_eoi")
        })?;
        output.extend_from_slice(&self.buffer);
        self.buffer.clear();
        Ok(())
    }

    /// Flushes ONLY complete bytes to an external buffer, preserving partial bytes.
    ///
    /// Used for streaming encoding where we continue writing bits in subsequent calls.
    /// Returns (remaining_bit_buffer, remaining_bits_count) to pass to next BitWriter.
    ///
    /// Unlike `flush_without_eoi`, this does NOT pad the final partial byte.
    pub fn flush_complete_bytes_only(
        &mut self,
        output: &mut Vec<u8>,
    ) -> crate::error::Result<(u64, u8)> {
        // Flush only complete bytes to internal buffer
        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = (self.bit_buffer >> self.bits_in_buffer) as u8;
            self.buffer.push(byte);

            if byte == 0xFF {
                self.buffer.push(0x00);
            }
        }

        // Move internal buffer to output
        output.try_reserve(self.buffer.len()).map_err(|_| {
            crate::error::Error::allocation_failed(self.buffer.len(), "flush_complete_bytes_only")
        })?;
        output.extend_from_slice(&self.buffer);
        self.buffer.clear();

        // Return remaining partial byte state (mask to keep only valid bits)
        let mask = if self.bits_in_buffer == 0 {
            0
        } else {
            (1u64 << self.bits_in_buffer) - 1
        };
        Ok((self.bit_buffer & mask, self.bits_in_buffer))
    }

    /// Creates a BitWriter with an initial bit buffer state.
    ///
    /// Used for streaming encoding to continue from a previous partial byte.
    #[must_use]
    pub fn with_initial_bits(bit_buffer: u64, bits_in_buffer: u8) -> Self {
        Self {
            buffer: ProfiledVec::with_capacity_profiled(0, "BitWriter::with_initial_bits"),
            bit_buffer,
            bits_in_buffer,
        }
    }

    /// Writes a restart marker and resets bit state.
    ///
    /// This flushes any pending bits, writes the restart marker (RST0-RST7),
    /// and resets the bit buffer.
    pub fn flush_restart_marker(&mut self, restart_num: u8) -> crate::error::Result<()> {
        // Flush pending bits
        self.flush();

        // Write restart marker (0xFFD0-0xFFD7)
        self.buffer.push(0xFF);
        self.buffer.push(0xD0 + (restart_num & 0x07));

        // Reset bit state
        self.bit_buffer = 0;
        self.bits_in_buffer = 0;

        Ok(())
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
/// Uses dual 64-bit buffers for optimized peek/read operations:
/// - `bit_buffer`: Bottom-aligned (LSB) for building up
/// - `aligned_buffer`: Top-aligned (MSB at bit 63) for fast peek operations
#[derive(Debug)]
pub struct BitReader<'a> {
    /// Input data
    data: &'a [u8],
    /// Current byte position
    position: usize,
    /// Bottom-aligned bit buffer (bits in low positions)
    bit_buffer: u64,
    /// Top-aligned bit buffer (MSB at bit 63) for fast peek operations
    aligned_buffer: u64,
    /// Number of bits in buffer (0-64)
    bits_in_buffer: u8,
    /// Whether we've hit a marker
    marker_found: Option<u8>,
    /// Number of bytes we've over-read past end of data
    overread_by: usize,
}

/// Saved state of a BitReader for speculative decoding.
#[derive(Clone, Copy)]
pub struct BitReaderState {
    position: usize,
    bit_buffer: u64,
    aligned_buffer: u64,
    bits_in_buffer: u8,
    marker_found: Option<u8>,
    overread_by: usize,
}

/// Check if a u32 contains a 0xFF byte using SWAR (SIMD Within A Register).
/// From Stanford Bithacks: https://graphics.stanford.edu/~seander/bithacks.html
#[inline(always)]
const fn has_byte_0xff_u32(v: u32) -> bool {
    // XOR with 0xFFFFFFFF to find bytes that are 0xFF (they become 0x00)
    let x = v ^ 0xFFFF_FFFF;
    // Check if any byte is zero using the "has zero byte" trick
    (((x.wrapping_sub(0x0101_0101)) & !x) & 0x8080_8080) != 0
}

/// Check if a u64 contains a 0xFF byte using SWAR (SIMD Within A Register).
/// This is the same algorithm as jpegli's HasZeroByte, inverted to detect 0xFF.
/// Returns true if any of the 8 bytes equals 0xFF.
#[inline(always)]
const fn has_byte_0xff_u64(v: u64) -> bool {
    // XOR with all 1s to find bytes that are 0xFF (they become 0x00)
    let x = v ^ 0xFFFF_FFFF_FFFF_FFFF;
    // Check if any byte is zero using the "has zero byte" trick:
    // (x - 0x01...) sets bit 7 if a byte was 0 (due to borrow)
    // & !x masks off bytes that already had bit 7 set
    // & 0x80... extracts only the bit 7 from each byte
    (((x.wrapping_sub(0x0101_0101_0101_0101)) & !x) & 0x8080_8080_8080_8080) != 0
}

impl<'a> BitReader<'a> {
    /// Creates a new bit reader.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            bit_buffer: 0,
            aligned_buffer: 0,
            bits_in_buffer: 0,
            marker_found: None,
            overread_by: 0,
        }
    }

    /// Reads a single byte with byte unstuffing (slow path).
    /// Used when fast 4-byte path can't be used (has 0xFF or not enough bytes).
    ///
    /// Returns `None` when:
    /// - A marker was previously found
    /// - A marker is found during this read
    /// - End of data is reached
    #[inline]
    fn read_byte_slow(&mut self) -> Option<u8> {
        if self.marker_found.is_some() {
            return None;
        }

        if self.position >= self.data.len() {
            self.overread_by += 1;
            return None;
        }

        let byte = self.data[self.position];
        self.position += 1;

        if byte == 0xFF {
            // Skip any fill bytes (consecutive 0xFF)
            while self.position < self.data.len() && self.data[self.position] == 0xFF {
                self.position += 1;
            }

            if self.position >= self.data.len() {
                self.overread_by += 1;
                return None;
            }

            let next = self.data[self.position];
            if next == 0x00 {
                // Byte stuffing - skip the 0x00
                self.position += 1;
            } else {
                // Found a marker
                self.position -= 1;
                self.marker_found = Some(next);
                return None;
            }
        }

        Some(byte)
    }

    /// Sync aligned_buffer from bit_buffer.
    /// Call after modifying bit_buffer to keep both in sync.
    #[inline(always)]
    fn sync_aligned(&mut self) {
        self.aligned_buffer = if self.bits_in_buffer > 0 && self.bits_in_buffer < 64 {
            self.bit_buffer << (64 - self.bits_in_buffer)
        } else if self.bits_in_buffer == 64 {
            self.bit_buffer
        } else {
            0
        };
    }

    /// Refills the bit buffer to have at least 32 bits.
    /// Uses fast 4-byte path when no 0xFF bytes are present.
    #[inline(always)]
    pub fn refill(&mut self) -> Result<bool> {
        // Only refill if we have fewer than 32 bits
        if self.bits_in_buffer >= 32 {
            return Ok(true);
        }

        // If we've found a marker or are overreading, extend with zeros
        if self.marker_found.is_some() || self.overread_by > 0 {
            // Shift in zeros (important: actually fill with zeros, not just claim more bits)
            self.bit_buffer <<= 32;
            self.bits_in_buffer = self.bits_in_buffer.saturating_add(32).min(64);
            self.sync_aligned();
            return Ok(true);
        }

        // Try fast 4-byte path (no 0xFF bytes)
        if self.position + 4 <= self.data.len() {
            let bytes = [
                self.data[self.position],
                self.data[self.position + 1],
                self.data[self.position + 2],
                self.data[self.position + 3],
            ];
            let word = u32::from_be_bytes(bytes);

            // Check if any byte is 0xFF using SWAR
            if !has_byte_0xff_u32(word) {
                // No 0xFF - fast path
                self.position += 4;
                self.bit_buffer = (self.bit_buffer << 32) | (word as u64);
                self.bits_in_buffer += 32;
                self.sync_aligned();
                return Ok(true);
            }
            // Has 0xFF - fall through to slow path
        }

        // Slow path: read byte by byte with byte stuffing
        while self.bits_in_buffer <= 56 {
            match self.read_byte_slow() {
                Some(byte) => {
                    self.bit_buffer = (self.bit_buffer << 8) | (byte as u64);
                    self.bits_in_buffer += 8;
                }
                None => break,
            }
            if self.bits_in_buffer >= 32 {
                break;
            }
        }
        self.sync_aligned();
        Ok(self.bits_in_buffer > 0)
    }

    /// Fills the bit buffer to have at least `count` bits.
    #[inline]
    fn fill_buffer(&mut self, count: u8) -> Result<bool> {
        if self.bits_in_buffer < count {
            self.refill()?;
        }
        Ok(self.bits_in_buffer >= count)
    }

    /// Peeks at the next `count` bits without consuming them.
    /// Uses fast top-aligned buffer for O(1) peek.
    ///
    /// Returns:
    /// - `Ok(ScanRead::EndOfScan)` if a marker was encountered
    /// - `Ok(ScanRead::Truncated)` if data ended without a marker
    #[inline]
    pub fn peek_bits(&mut self, count: u8) -> ScanResult<u32> {
        debug_assert!(count <= 32);
        self.fill_buffer(count)?;
        if self.bits_in_buffer < count {
            return Ok(self.end_state());
        }
        // Fast peek using top-aligned buffer - just right shift
        Ok(ScanRead::Value(
            (self.aligned_buffer >> (64 - count)) as u32,
        ))
    }

    /// Fast peek that refills first. Returns None if not enough bits after refill.
    /// Optimized for Huffman decode hot path.
    #[inline(always)]
    pub fn peek_bits_refill(&mut self, count: u8) -> Option<u32> {
        if self.bits_in_buffer < count {
            let _ = self.refill();
            if self.bits_in_buffer < count {
                return None;
            }
        }
        Some((self.aligned_buffer >> (64 - count)) as u32)
    }

    /// Skip bits without any checks. Only call after successful peek.
    /// NOTE: This must keep bit_buffer in sync for refill() to work correctly.
    #[inline(always)]
    pub fn skip_bits_fast(&mut self, count: u8) {
        self.bits_in_buffer -= count;
        self.aligned_buffer <<= count;
        // Keep bit_buffer in sync (needed for refill to work correctly)
        let mask = if self.bits_in_buffer >= 64 {
            u64::MAX
        } else {
            (1u64 << self.bits_in_buffer).wrapping_sub(1)
        };
        self.bit_buffer &= mask;
    }

    /// Read bits without refill. Only call when you know enough bits are available.
    /// Returns the bits and consumes them.
    /// NOTE: This must keep bit_buffer in sync for refill() to work correctly.
    #[inline(always)]
    pub fn read_bits_fast(&mut self, count: u8) -> u32 {
        let bits = (self.aligned_buffer >> (64 - count)) as u32;
        self.bits_in_buffer -= count;
        self.aligned_buffer <<= count;
        // Keep bit_buffer in sync (needed for refill to work correctly)
        let mask = if self.bits_in_buffer >= 64 {
            u64::MAX
        } else {
            (1u64 << self.bits_in_buffer).wrapping_sub(1)
        };
        self.bit_buffer &= mask;
        bits
    }

    /// Ensure we have at least 32 bits in the buffer.
    /// Returns true if successful, false if we hit a marker or end of data.
    /// This is the key for fast decoding - call once at start of block,
    /// then use read_bits_fast for individual reads.
    #[inline(always)]
    pub fn ensure_bits(&mut self) -> bool {
        if self.bits_in_buffer < 32 {
            let _ = self.refill();
        }
        self.bits_in_buffer >= 32
    }

    /// Peek at top N bits without consuming. No refill check.
    #[inline(always)]
    pub fn peek_top(&self, count: u8) -> u32 {
        (self.aligned_buffer >> (64 - count)) as u32
    }

    /// Get bits with rotate trick (like zune-jpeg's get_bits).
    /// This is marginally faster for some use cases.
    #[inline(always)]
    pub fn get_bits_rotate(&mut self, n_bits: u8) -> i32 {
        let mask = (1_u64 << n_bits) - 1;
        self.aligned_buffer = self.aligned_buffer.rotate_left(u32::from(n_bits));
        let bits = (self.aligned_buffer & mask) as i32;
        self.bits_in_buffer = self.bits_in_buffer.wrapping_sub(n_bits);
        bits
    }

    /// Reads `count` bits from the stream.
    ///
    /// Returns:
    /// - `Ok(ScanRead::EndOfScan)` if a marker was encountered
    /// - `Ok(ScanRead::Truncated)` if data ended without a marker
    #[inline]
    pub fn read_bits(&mut self, count: u8) -> ScanResult<u32> {
        self.fill_buffer(count)?;
        if self.bits_in_buffer < count {
            return Ok(self.end_state());
        }
        // Use aligned buffer for fast read
        let bits = (self.aligned_buffer >> (64 - count)) as u32;
        self.drop_bits(count);
        Ok(ScanRead::Value(bits))
    }

    /// Drops `count` bits from the buffer (fast path).
    /// Updates both buffers to keep them in sync for refill.
    #[inline(always)]
    fn drop_bits(&mut self, count: u8) {
        self.bits_in_buffer = self.bits_in_buffer.saturating_sub(count);
        self.aligned_buffer <<= count;
        // Keep bit_buffer in sync: mask off the dropped MSBs
        let mask = if self.bits_in_buffer >= 64 {
            u64::MAX
        } else {
            (1u64 << self.bits_in_buffer).wrapping_sub(1)
        };
        self.bit_buffer &= mask;
    }

    /// Skips `count` bits.
    #[inline]
    pub fn skip_bits(&mut self, count: u8) {
        self.drop_bits(count);
    }

    /// Reads a single bit.
    #[inline]
    pub fn read_bit(&mut self) -> ScanResult<bool> {
        match self.read_bits(1)? {
            ScanRead::Value(v) => Ok(ScanRead::Value(v != 0)),
            ScanRead::EndOfScan => Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => Ok(ScanRead::Truncated),
        }
    }

    /// Reads a signed value with sign extension.
    ///
    /// JPEG encodes signed values where values < 2^(bits-1) are negative.
    pub fn read_signed(&mut self, bits: u8) -> ScanResult<i16> {
        if bits == 0 {
            return Ok(ScanRead::Value(0));
        }

        let value = match self.read_bits(bits)? {
            ScanRead::Value(v) => v as i16,
            ScanRead::EndOfScan => return Ok(ScanRead::EndOfScan),
            ScanRead::Truncated => return Ok(ScanRead::Truncated),
        };
        let half = 1i16 << (bits - 1);

        if value < half {
            Ok(ScanRead::Value(value - (2 * half - 1)))
        } else {
            Ok(ScanRead::Value(value))
        }
    }

    /// Aligns to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        self.bits_in_buffer = 0;
        self.aligned_buffer = 0;
    }

    /// Saves the current reader state for potential rollback.
    #[must_use]
    pub fn save_state(&self) -> BitReaderState {
        BitReaderState {
            position: self.position,
            bit_buffer: self.bit_buffer,
            aligned_buffer: self.aligned_buffer,
            bits_in_buffer: self.bits_in_buffer,
            marker_found: self.marker_found,
            overread_by: self.overread_by,
        }
    }

    /// Restores a previously saved state.
    pub fn restore_state(&mut self, state: BitReaderState) {
        self.position = state.position;
        self.bit_buffer = state.bit_buffer;
        self.aligned_buffer = state.aligned_buffer;
        self.bits_in_buffer = state.bits_in_buffer;
        self.marker_found = state.marker_found;
        self.overread_by = state.overread_by;
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
            return Err(Error::invalid_jpeg_data(
                "unexpected end of data before restart marker",
            ));
        }
        let first = self.data[self.position];
        if first != 0xFF {
            return Err(Error::invalid_jpeg_data("expected 0xFF for restart marker"));
        }
        self.position += 1;

        // Read second byte - should be 0xD0 + expected_num
        if self.position >= self.data.len() {
            return Err(Error::invalid_jpeg_data(
                "unexpected end of data in restart marker",
            ));
        }
        let second = self.data[self.position];
        let expected_marker = 0xD0 + (expected_num & 7);
        if second != expected_marker {
            // Check if it's a different restart marker (resync case)
            if (0xD0..=0xD7).contains(&second) {
                return Err(Error::invalid_jpeg_data("restart marker sequence mismatch"));
            }
            return Err(Error::invalid_jpeg_data(
                "expected restart marker not found",
            ));
        }
        self.position += 1;

        Ok(())
    }

    /// Reads a raw byte (assumes byte-aligned).
    pub fn read_byte_raw(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(Error::truncated_data("reading raw byte"));
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

    /// Returns true if we've exhausted real data (hit a marker or past end).
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.marker_found.is_some() || self.position >= self.data.len()
    }

    /// Returns number of bits currently available in buffer.
    #[must_use]
    pub fn bits_available(&self) -> u8 {
        self.bits_in_buffer
    }

    /// Returns the appropriate end state based on why we stopped reading.
    ///
    /// - `EndOfScan` if a marker was found (legitimate end of entropy-coded segment)
    /// - `Truncated` if data ended without finding a marker
    #[inline]
    fn end_state<T>(&self) -> ScanRead<T> {
        if self.marker_found.is_some() {
            ScanRead::EndOfScan
        } else {
            ScanRead::Truncated
        }
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
        assert_eq!(reader.read_bits(3).unwrap(), ScanRead::Value(0b101));
        assert_eq!(reader.read_bits(4).unwrap(), ScanRead::Value(0b1100));
        assert_eq!(reader.read_bits(1).unwrap(), ScanRead::Value(0b1));
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

        assert_eq!(reader.read_bits(8).unwrap(), ScanRead::Value(0xFF));
        assert_eq!(reader.read_bits(8).unwrap(), ScanRead::Value(0xAB));
    }

    #[test]
    fn test_signed_values() {
        // Test JPEG signed value encoding
        // Data: 0b01000000 = 0x40, reading MSB first: bit0=0, bit1=1
        let data = [0b0100_0000]; // First bit = 0 (means -1), second bit = 1 (means +1)
        let mut reader = BitReader::new(&data);

        // 1-bit category: 0 -> -1, 1 -> 1
        assert_eq!(reader.read_signed(1).unwrap(), ScanRead::Value(-1));
        assert_eq!(reader.read_signed(1).unwrap(), ScanRead::Value(1));
    }
}
