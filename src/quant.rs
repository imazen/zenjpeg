//! Quantization table handling for JPEG encoding
//!
//! Supports standard JPEG tables, mozjpeg's optimized tables, and jpegli's
//! perceptually-tuned tables.

use crate::consts::{MOZJPEG_LUMA_QUANT, STD_CHROMA_QUANT, STD_LUMA_QUANT};

/// mozjpeg ImageMagick-style chrominance table (index 3)
/// Uses same values as luma table for better quality.
/// Values >255 are clamped AFTER quality scaling, not in the base table.
const MOZJPEG_CHROMA_QUANT: [u16; 64] = [
    16, 16, 16, 18, 25, 37, 56, 85,
    16, 17, 20, 27, 34, 40, 53, 75,
    16, 20, 24, 31, 43, 62, 91, 135,
    18, 27, 31, 40, 53, 74, 106, 156,
    25, 34, 43, 53, 69, 94, 131, 189,
    37, 40, 62, 74, 94, 124, 169, 238,
    56, 53, 91, 106, 131, 169, 226, 311,  // 311 is valid, will clamp after scaling
    85, 75, 135, 156, 189, 238, 311, 418, // 311, 418 are valid, will clamp after scaling
];

/// Quantization table for a single component
#[derive(Clone, Debug)]
pub struct QuantTable {
    /// Quantization values in natural order
    pub values: [u16; 64],
    /// Table slot (0-3)
    pub slot: u8,
}

impl QuantTable {
    /// Create a quantization table from an array
    pub fn new(values: [u16; 64], slot: u8) -> Self {
        Self { values, slot }
    }

    /// Create standard luminance table at given quality
    pub fn luma_standard(quality: u8) -> Self {
        Self::from_base_table(&STD_LUMA_QUANT, quality, 0)
    }

    /// Create standard chrominance table at given quality
    pub fn chroma_standard(quality: u8) -> Self {
        Self::from_base_table(&STD_CHROMA_QUANT, quality, 1)
    }

    /// Create mozjpeg-style luminance table at given quality
    pub fn luma_mozjpeg(quality: u8) -> Self {
        Self::from_base_table(&MOZJPEG_LUMA_QUANT, quality, 0)
    }

    /// Create mozjpeg-style chrominance table at given quality
    pub fn chroma_mozjpeg(quality: u8) -> Self {
        Self::from_base_table(&MOZJPEG_CHROMA_QUANT, quality, 1)
    }

    /// Scale a base quantization table by quality factor
    fn from_base_table(base: &[u16; 64], quality: u8, slot: u8) -> Self {
        let quality = quality.clamp(1, 100);

        // JPEG quality scaling formula
        let scale = if quality < 50 {
            5000 / quality as u32
        } else {
            200 - 2 * quality as u32
        };

        let mut values = [0u16; 64];
        for i in 0..64 {
            let val = (base[i] as u32 * scale + 50) / 100;
            values[i] = val.clamp(1, 255) as u16;
        }

        Self { values, slot }
    }

    /// Get quantization value at zigzag position
    #[inline]
    pub fn at_zigzag(&self, pos: usize) -> u16 {
        self.values[crate::consts::ZIGZAG[pos]]
    }
}

/// Quantization table set for Y, Cb, Cr components
#[derive(Clone)]
pub struct QuantTableSet {
    /// Luminance quantization table
    pub luma: QuantTable,
    /// Chrominance quantization table
    pub chroma: QuantTable,
}

impl QuantTableSet {
    /// Create standard tables at given quality
    pub fn standard(quality: u8) -> Self {
        Self {
            luma: QuantTable::luma_standard(quality),
            chroma: QuantTable::chroma_standard(quality),
        }
    }

    /// Create mozjpeg-style tables at given quality
    pub fn mozjpeg(quality: u8) -> Self {
        Self {
            luma: QuantTable::luma_mozjpeg(quality),
            chroma: QuantTable::chroma_mozjpeg(quality),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_scaling() {
        // Q50 should give scale factor of 100 (no change)
        let q50 = QuantTable::luma_standard(50);
        assert_eq!(q50.values[0], STD_LUMA_QUANT[0]);

        // Q100 should give scale factor of 0 (minimum values)
        let q100 = QuantTable::luma_standard(100);
        assert_eq!(q100.values[0], 1); // All values clamped to 1

        // Q1 should give large values
        let q1 = QuantTable::luma_standard(1);
        assert!(q1.values[0] > 100);
    }
}
