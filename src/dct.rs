//! Forward DCT (Discrete Cosine Transform) for JPEG encoding
//!
//! This module provides DCT implementations, including a reference
//! scalar implementation and optional SIMD-accelerated versions.

/// Forward 8x8 DCT on level-shifted input
///
/// Takes an 8x8 block of pixels (level-shifted by -128) and produces
/// DCT coefficients. Output is scaled by 1/8 for JPEG compatibility.
pub fn forward_dct_8x8(block: &[i16; 64]) -> [f32; 64] {
    let mut output = [0.0f32; 64];

    // Reference implementation using direct DCT formula
    // This will be replaced with fast DCT (Loeffler algorithm) later
    for v in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;

            for y in 0..8 {
                for x in 0..8 {
                    let pixel = block[y * 8 + x] as f32;
                    let cu = if u == 0 {
                        1.0 / std::f32::consts::SQRT_2
                    } else {
                        1.0
                    };
                    let cv = if v == 0 {
                        1.0 / std::f32::consts::SQRT_2
                    } else {
                        1.0
                    };

                    let cos_u =
                        ((2.0 * x as f32 + 1.0) * u as f32 * std::f32::consts::PI / 16.0).cos();
                    let cos_v =
                        ((2.0 * y as f32 + 1.0) * v as f32 * std::f32::consts::PI / 16.0).cos();

                    sum += cu * cv * pixel * cos_u * cos_v;
                }
            }

            // Scale by 1/4 (standard normalization) then by 1/8 for JPEG
            output[v * 8 + u] = sum / 4.0;
        }
    }

    output
}

/// Quantize DCT coefficients using the given quantization table
pub fn quantize_block(dct: &[f32; 64], quant: &[u16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];

    for i in 0..64 {
        // Round to nearest integer
        let q = quant[i] as f32;
        output[i] = (dct[i] / q).round() as i16;
    }

    output
}

/// Level-shift a block of pixels for DCT (subtract 128)
pub fn level_shift(pixels: &[u8; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for i in 0..64 {
        output[i] = pixels[i] as i16 - 128;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_only() {
        // A uniform block should have only DC component
        let block = [0i16; 64]; // level-shifted 128 -> 0
        let dct = forward_dct_8x8(&block);

        // DC should be 0 for uniform 128 block (after level shift)
        assert!(dct[0].abs() < 0.01, "DC = {}", dct[0]);

        // All AC should be 0
        for i in 1..64 {
            assert!(dct[i].abs() < 0.01, "AC[{}] = {}", i, dct[i]);
        }
    }
}
