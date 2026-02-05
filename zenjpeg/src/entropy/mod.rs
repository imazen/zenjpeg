//! Entropy coding for JPEG.
//!
//! This module provides Huffman-based entropy encoding and decoding
//! for JPEG DCT coefficients.
//!
//! The module is split into:
//! - `encoder`: EntropyEncoder for baseline and progressive encoding
//! - `decoder`: EntropyDecoder for baseline and progressive decoding
//!
//! # Performance Optimizations
//!
//! - Pre-computed category lookup table (4KB) for O(1) category lookup
//! - Combined Huffman code + extra bits writes to reduce write_bits calls

#![allow(dead_code)]

#[cfg(feature = "decoder")]
pub mod arithmetic;
#[cfg(feature = "decoder")]
pub mod decoder;
pub mod encoder;

// Re-export main types
#[cfg(feature = "decoder")]
pub use arithmetic::ArithmeticDecoder;
#[cfg(feature = "decoder")]
pub use decoder::{EntropyDecoder, EntropyDecoderState};
pub use encoder::EntropyEncoder;

/// Maximum DC coefficient difference magnitude (for 8-bit samples).
pub const MAX_DC_DIFF: i16 = 2047;

/// Maximum AC coefficient magnitude (for 8-bit samples).
pub const MAX_AC_COEFF: i16 = 1023;

/// Pre-computed category table for values -2047..=2047.
/// Index with (value + 2048) to get the category (bit count).
/// This avoids the leading_zeros() call in the hot path.
static CATEGORY_TABLE: [u8; 4096] = {
    let mut table = [0u8; 4096];
    let mut i = 0i32;
    while i < 4096 {
        let value = i - 2048;
        table[i as usize] = if value == 0 {
            0
        } else {
            let abs_val = if value < 0 { -value } else { value } as u32;
            // Category is the number of bits needed to represent abs_val
            // For u32: category = 32 - leading_zeros(abs_val)
            (32 - abs_val.leading_zeros()) as u8
        };
        i += 1;
    }
    table
};

/// Returns the category (number of bits needed) for a value.
/// Uses a lookup table for values in range -2047..=2047 (covers all JPEG coefficients).
#[inline(always)]
#[must_use]
pub fn category(value: i16) -> u8 {
    // Fast path: use lookup table for common range
    let idx = (value as i32 + 2048) as usize;
    if idx < 4096 {
        CATEGORY_TABLE[idx]
    } else {
        // Fallback for out-of-range values (shouldn't happen in valid JPEG)
        if value == 0 {
            0
        } else {
            16 - value.unsigned_abs().leading_zeros() as u8
        }
    }
}

/// Returns the category using leading_zeros (for benchmarking comparison).
#[inline]
#[must_use]
pub fn category_scalar(value: i16) -> u8 {
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

/// Returns the additional bits for a value given its pre-computed category.
/// Avoids recomputing category when it's already known.
#[inline(always)]
#[must_use]
pub fn additional_bits_with_cat(value: i16, cat: u8) -> u16 {
    if value >= 0 {
        value as u16
    } else {
        // For negative values, encode as (value - 1) in one's complement
        (value - 1) as u16 & ((1u16 << cat) - 1)
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

/// Branchless JPEG HUFF_EXTEND equivalent.
/// Reconstructs a signed value from category (s) and bits (x).
/// This is ~2x faster than the branching version for random input.
///
/// Formula: x + (((x - (1 << (s-1))) >> 31) & ((-1 << s) + 1))
/// - If x >= 2^(s-1), returns x (positive value)
/// - If x < 2^(s-1), returns x - (2^s - 1) (negative value)
#[inline(always)]
pub fn huff_extend(x: i32, s: i32) -> i32 {
    // The shift creates a mask: all 1s if x < half, all 0s otherwise
    // This is branchless and SIMD-friendly
    x + ((((x) - (1 << ((s) - 1))) >> 31) & (((-1) << (s)) + 1))
}

/// Encodes a single block to an external BitWriter.
///
/// This is used by bounded-memory streaming to encode blocks with external
/// DC prediction state. Returns the new DC value.
///
/// # Arguments
/// * `coeffs` - Quantized DCT coefficients in zigzag order
/// * `dc_table` - DC Huffman encoding table
/// * `ac_table` - AC Huffman encoding table
/// * `prev_dc` - Previous DC coefficient value for this component
/// * `writer` - BitWriter to write encoded data to
pub fn encode_block_to_writer(
    coeffs: &[i16; 64],
    dc_table: &crate::huffman::HuffmanEncodeTable,
    ac_table: &crate::huffman::HuffmanEncodeTable,
    prev_dc: i16,
    writer: &mut crate::foundation::bitstream::BitWriter,
) -> crate::error::Result<()> {
    // Encode DC coefficient
    let dc = coeffs[0];
    let dc_diff = dc - prev_dc;
    let dc_cat = category(dc_diff);
    let (code, len) = dc_table.encode(dc_cat);

    if dc_cat > 0 {
        let additional = additional_bits_with_cat(dc_diff, dc_cat);
        writer.write_code_and_extra(code, len, additional, dc_cat);
    } else {
        writer.write_bits(code, len);
    }

    // Encode AC coefficients
    let mut run = 0u8;
    for i in 1..64 {
        let ac = coeffs[i];

        if ac == 0 {
            run += 1;
        } else {
            // Encode any runs of 16 zeros
            while run >= 16 {
                let (code, len) = ac_table.encode(0xF0); // ZRL
                writer.write_bits(code, len);
                run -= 16;
            }

            // Encode the coefficient
            let ac_cat = category(ac);
            let symbol = (run << 4) | ac_cat;
            let (code, len) = ac_table.encode(symbol);
            let additional = additional_bits_with_cat(ac, ac_cat);
            writer.write_code_and_extra(code, len, additional, ac_cat);
            run = 0;
        }
    }

    // EOB if there are trailing zeros
    if run > 0 {
        let (code, len) = ac_table.encode(0x00); // EOB
        writer.write_bits(code, len);
    }

    Ok(())
}

/// State carried between streaming calls to [`encode_blocks_mcu_order`].
#[derive(Clone, Debug)]
pub struct StreamingEntropyState {
    /// Previous DC values for each component (Y, Cb, Cr).
    pub prev_dc: [i16; 3],
    /// Global MCU index (for restart marker numbering).
    pub mcu_idx: usize,
    /// Restart marker counter (0-7, wraps).
    pub restart_count: u8,
}

impl StreamingEntropyState {
    /// Creates initial state with all zeros.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev_dc: [0; 3],
            mcu_idx: 0,
            restart_count: 0,
        }
    }
}

impl Default for StreamingEntropyState {
    fn default() -> Self {
        Self::new()
    }
}

/// Encodes a batch of blocks in MCU-interleaved order to a BitWriter.
///
/// Handles all subsampling modes (4:4:4, 4:2:2, 4:2:0, 4:4:0) and restart
/// markers. This is a pure function suitable for streaming: pass in blocks
/// from one strip, get back updated state for the next strip.
///
/// The blocks must be in raster order within each component (row-major,
/// left-to-right, top-to-bottom).
///
/// # Arguments
/// * `y_blocks` - Luminance DCT blocks in raster order
/// * `cb_blocks` - Cb chrominance blocks (empty if grayscale)
/// * `cr_blocks` - Cr chrominance blocks (empty if grayscale)
/// * `tables` - Optimized Huffman encoding tables
/// * `writer` - BitWriter to append encoded data to
/// * `is_color` - Whether to encode chroma components
/// * `state` - DC prediction / restart marker state from previous call
/// * `subsampling` - Chroma subsampling mode
/// * `width` - Image width in pixels
/// * `restart_interval` - MCUs between restart markers (0 = disabled)
/// * `total_mcus` - Total MCUs in the full image (for last-MCU check)
pub fn encode_blocks_mcu_order(
    y_blocks: &[[i16; 64]],
    cb_blocks: &[[i16; 64]],
    cr_blocks: &[[i16; 64]],
    tables: &crate::huffman::optimize::HuffmanTableSet,
    writer: &mut crate::foundation::bitstream::BitWriter,
    is_color: bool,
    state: &mut StreamingEntropyState,
    subsampling: crate::types::Subsampling,
    width: usize,
    restart_interval: u16,
    total_mcus: usize,
) -> crate::error::Result<()> {
    use crate::types::Subsampling;

    let dc_luma = &tables.dc_luma.table;
    let ac_luma = &tables.ac_luma.table;
    let dc_chroma = &tables.dc_chroma.table;
    let ac_chroma = &tables.ac_chroma.table;

    let (h_samp, v_samp) = match subsampling {
        Subsampling::S444 => (1, 1),
        Subsampling::S422 => (2, 1),
        Subsampling::S420 => (2, 2),
        Subsampling::S440 => (1, 2),
    };

    // Block dimensions for the full image
    let y_blocks_w = (width + 7) / 8;
    let c_blocks_w = ((width + h_samp - 1) / h_samp + 7) / 8;
    let mcu_h = (y_blocks_w + h_samp - 1) / h_samp;

    // Y blocks in this batch are stored in raster order:
    //   block[row * y_blocks_w + col]
    // For subsampled modes, each MCU row spans v_samp Y block-rows
    // and 1 chroma block-row.
    let y_rows_in_batch = y_blocks.len().checked_div(y_blocks_w).unwrap_or(0);
    let mcu_rows_in_batch = if h_samp == 1 && v_samp == 1 {
        y_rows_in_batch
    } else {
        (y_rows_in_batch + v_samp - 1) / v_samp
    };

    const ZERO_BLOCK: [i16; 64] = [0i16; 64];

    for mcu_row_offset in 0..mcu_rows_in_batch {
        for mcu_x in 0..mcu_h {
            // Encode Y blocks in this MCU
            for dy in 0..v_samp {
                for dx in 0..h_samp {
                    let col = mcu_x * h_samp + dx;
                    let row = mcu_row_offset * v_samp + dy;
                    let block = if col < y_blocks_w && row < y_rows_in_batch {
                        y_blocks.get(row * y_blocks_w + col).unwrap_or(&ZERO_BLOCK)
                    } else {
                        &ZERO_BLOCK
                    };

                    encode_block_to_writer(block, dc_luma, ac_luma, state.prev_dc[0], writer)?;
                    state.prev_dc[0] = block[0];
                }
            }

            // Encode Cb and Cr
            if is_color {
                let c_row = mcu_row_offset;
                let c_col = mcu_x;
                let c_idx = c_row * c_blocks_w + c_col;

                let cb = cb_blocks.get(c_idx).unwrap_or(&ZERO_BLOCK);
                encode_block_to_writer(cb, dc_chroma, ac_chroma, state.prev_dc[1], writer)?;
                state.prev_dc[1] = cb[0];

                let cr = cr_blocks.get(c_idx).unwrap_or(&ZERO_BLOCK);
                encode_block_to_writer(cr, dc_chroma, ac_chroma, state.prev_dc[2], writer)?;
                state.prev_dc[2] = cr[0];
            }

            state.mcu_idx += 1;

            // Restart marker (not after last MCU)
            if restart_interval > 0
                && state.mcu_idx < total_mcus
                && state.mcu_idx % restart_interval as usize == 0
            {
                writer.flush_restart_marker(state.restart_count)?;
                state.restart_count = (state.restart_count + 1) & 0x07;
                state.prev_dc = [0; 3];
            }
        }
    }

    Ok(())
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

    #[test]
    fn test_decode_value_edge_cases() {
        // Category 0 always returns 0
        assert_eq!(decode_value(0, 0), 0);
        assert_eq!(decode_value(0, 5), 0);

        // Category > 15 uses bits directly
        assert_eq!(decode_value(16, 100), 100);
        assert_eq!(decode_value(20, 50), 50);

        // Category 1: bits 0 -> -1, bits 1 -> 1
        assert_eq!(decode_value(1, 0), -1);
        assert_eq!(decode_value(1, 1), 1);

        // Category 2: bits 0,1 -> -3,-2; bits 2,3 -> 2,3
        assert_eq!(decode_value(2, 0), -3);
        assert_eq!(decode_value(2, 1), -2);
        assert_eq!(decode_value(2, 2), 2);
        assert_eq!(decode_value(2, 3), 3);
    }

    #[test]
    fn test_category_large_values() {
        // Test maximum values
        assert_eq!(category(2047), 11);
        assert_eq!(category(-2047), 11);

        // Test near boundaries
        assert_eq!(category(1023), 10);
        assert_eq!(category(1024), 11);
        assert_eq!(category(511), 9);
        assert_eq!(category(512), 10);
    }
}
