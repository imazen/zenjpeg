//! Entropy coding for JPEG.
//!
//! This module provides Huffman-based entropy encoding and decoding
//! for JPEG DCT coefficients.
//!
//! The module is split into:
//! - `encoder`: EntropyEncoder for baseline and progressive encoding
//! - `decoder`: EntropyDecoder for baseline and progressive decoding

pub mod decoder;
pub mod encoder;

// Re-export main types
pub use decoder::{EntropyDecoder, EntropyDecoderState};
pub use encoder::EntropyEncoder;

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

/// Returns the additional bits for a value given its pre-computed category.
/// Avoids recomputing category when it's already known.
#[inline]
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
