//! Glassa low-BPP optimized quantization tables.
//!
//! These tables were optimized using simulated annealing with SSIMULACRA2 as
//! the fitness metric. They achieve massive pareto gains (+20 to +33) at low
//! quality levels by aggressively zeroing high-frequency coefficients.
//!
//! # When to Use
//!
//! - **Thumbnails**: Ultra-low BPP placeholders (0.15-0.25 BPP)
//! - **LQIP**: Low-quality image placeholders for progressive loading
//! - **Extreme compression**: When quality < 30 is acceptable
//!
//! # When NOT to Use
//!
//! - **Q30+**: These tables provide no benefit over mozjpeg defaults
//! - **High quality**: Use jpegli tables instead
//!
//! # Source
//!
//! Optimized by glassa project on CID22 corpus with mozjpeg encoder, 4:2:0.
//! See `~/work/glassa/results/all_100_tables.json`.

use crate::encode::tuning::{EncodingTables, PerComponent, ScalingParams};

/// Anchor quality levels for interpolation.
pub const ANCHORS: &[u8] = &[3, 5, 7, 10, 12, 15, 20, 25];

/// Luma quantization tables at each anchor quality.
///
/// Pattern: DC and low frequencies preserved, high frequencies maxed out.
#[rustfmt::skip]
pub const LUMA: &[[u16; 64]] = &[
    // Q3: pareto=+32.66, bpp=0.243, ssim2=-0.9
    [
          2,   2,  23,  14,  23,  32,  60,  87,
          5,  16,  13,  31,  47,  54,  57,  62,
         30,  13,  37,  65,  50,  38,  74, 116,
         19,  27,  28,  74,  56, 111,  68, 117,
         18,  33,  30,  27,  57,  66,  86, 138,
         23,  30,  57,  65, 100,  93, 127, 187,
         56,  38,  95,  81,  97,  98, 162, 143,
         50,  28, 108, 124, 114, 153, 149, 184,
    ],
    // Q5: pareto=+23.90, bpp=0.214, ssim2=-23.4
    [
          5,  31,  75,  16,  58,  48,  92,  68,
         14,  21,  35,  53,  66,  42,  54, 101,
         34,  18,  63,  69, 109,  77,  84, 165,
         30,  31,  31,  52,  51,  99, 174, 255,
         65,  87, 134, 170, 174, 206, 206, 255,
         60,  78, 185, 102, 154, 245, 255, 255,
        128, 103, 128, 131, 226, 255, 255, 255,
        183, 206, 231, 255, 255, 255, 255, 255,
    ],
    // Q7: pareto=+31.83, bpp=0.244, ssim2=-1.2
    [
          4,   7,  28,  19,  54,  51,  49, 154,
         16,  25,  22,  46,  59,  61,  48,  99,
         46,  67,  41,  60,  63,  66,  82,  99,
         56,  61,  37, 109,  88, 140, 134, 150,
        118, 124, 154, 149, 129, 144, 153, 241,
        104,  80, 135, 113, 133, 160, 248, 254,
        118, 101, 186, 178, 160, 246, 254, 254,
        163, 173, 207, 245, 254, 254, 254, 254,
    ],
    // Q10: pareto=+25.92, bpp=0.266, ssim2=2.3
    [
          8,   8,  40,  74,  33,  24,  48,  87,
         15,  24,  53,  54,  17,  56,  64, 123,
         36,  26,  26,  55,  40,  73, 107, 169,
         55,  21,  86,  42,  73, 135, 136, 217,
         91, 116, 134, 129, 151, 160, 205, 194,
         62,  67,  82, 164, 159, 203, 217, 217,
        156, 112, 138, 195, 206, 206, 213, 206,
        176,  96, 204, 209, 209, 209, 209, 209,
    ],
    // Q12: pareto=+30.40, bpp=0.254, ssim2=1.7
    [
          8,  17,  64,  23,  38,  79, 116,  96,
         16,  29,  99,  51,  69,  86,  39, 103,
         53,  65,  67,  37,  43, 111, 135, 166,
         79,  62,  34,  58,  72, 103, 156, 204,
         75,  99,  78,  77, 110, 134, 202, 241,
         99, 121, 138, 134, 155, 155, 213, 254,
        100, 136, 122, 171, 142, 244, 253, 252,
        149, 136, 174, 174, 193, 240, 238, 240,
    ],
    // Q15: pareto=+19.99, bpp=0.259, ssim2=-6.4
    [
          9,  10,  65,  43,  31,  50, 134, 112,
         29,  48,  49,  95,  60,  34,  50,  75,
         54,  63,  47, 131, 129, 117, 145, 255,
         74,  76,  87,  79, 168, 173, 202, 255,
        106,  89, 182,  94, 159, 178, 243, 243,
        132, 128, 187, 192, 248, 248, 248, 248,
        137, 122, 215, 210, 248, 239, 239, 239,
        222, 169, 248, 248, 248, 248, 245, 245,
    ],
    // Q20: pareto=+3.60, bpp=0.576, ssim2=46.4
    [
          5,  13,  13,   8,  14,  30,  39,  64,
          7,   7,   9,  30,  28,  17,  43,  61,
         13,  17,  20,  25,  34,  51,  59, 102,
         15,  23,  25,  33,  54,  60,  80, 163,
         46,  33,  62,  45,  89, 119, 147, 178,
         66,  67, 134,  85,  93, 123, 174, 233,
         80,  80,  98, 110, 138, 152, 219, 240,
         93,  85, 155, 144, 149, 210, 225, 230,
    ],
    // Q25: pareto=+5.36, bpp=0.483, ssim2=37.4
    [
          8,   8,  10,  22,  39,  19,  36,  67,
         25,  19,  21,  47,  30,  60,  60,  91,
         29,  49,  27,  34,  42,  61,  81, 142,
         21,  55,  62,  86,  48,  60,  99, 148,
         59,  71,  80,  70, 101, 126, 160, 216,
         62,  60, 109, 102, 117, 147, 191, 204,
         71,  68, 114, 124, 140, 191, 224, 215,
        102,  89, 101, 181, 177, 215, 209, 189,
    ],
];

/// Chroma quantization tables at each anchor quality (shared for Cb and Cr).
#[rustfmt::skip]
pub const CHROMA: &[[u16; 64]] = &[
    // Q3
    [
          2,   4,  14,  26,  24,  27,  45,  84,
          6,  36,  13,   5,  31,  41,  42,  84,
         20,  39,  16,  20,  81,  78,  84, 158,
         32,  37,  69,  85,  56,  90,  79, 132,
         24,  14,  39,  37,  62,  85, 129, 198,
         35,  54,  64,  77,  88, 126, 167, 200,
         72,  19,  72, 122, 154, 166, 240, 211,
         86,  60, 143, 148, 198, 198, 200, 213,
    ],
    // Q5
    [
          2,  10,  15,   3,  16,  34,  43,  63,
         52,  38,  36,   8,  28,  58,  61,  44,
         25,  43,  54,  43,  61,  63,  73, 136,
         39,  56,  55,  35,  65,  69,  59, 160,
         45,  47,  61,  53,  48,  82, 109, 181,
         49,  51,  38,  80,  79, 104, 148, 191,
         30,  39,  79,  65, 131, 164, 185, 189,
        130, 101, 145, 157, 157, 223, 225, 218,
    ],
    // Q7
    [
          3,   9,  59,  61,  36,  19,  61,  73,
          8,  11,  47,  26,  54,  65, 122, 105,
         47, 106,  53,  15,  84,  78, 130, 175,
          9, 107,  77,  98, 120, 141, 171, 189,
        131, 127, 107, 138, 144, 171, 231, 252,
        147, 109, 125, 178, 200, 246, 252, 252,
         94, 118, 152, 187, 242, 252, 252, 252,
        182, 213, 198, 232, 252, 252, 252, 252,
    ],
    // Q10
    [
          6,  15,  14,   4,  18,  56,  89,  73,
         42,  32,  11,   9,   8,  62,  64,  39,
         23,  22,  37,  47,  39,  70,  73, 100,
         29,  12,  36,  45,  40,  61,  65, 144,
         60,  70,  73, 100, 111, 138, 154, 161,
         60,  46,  61,  82, 102, 123, 191, 197,
         98,  78, 106,  96, 138, 139, 226, 220,
         35,  75, 150, 175, 196, 216, 216, 223,
    ],
    // Q12
    [
          9,  13,  36,  11,  42,  31,  47,  71,
         23,  13,  35,  15,  55,  21,  42, 112,
         41,  31,  52,  32,  61,  28,  47, 112,
         22,  45,  82, 116,  99, 106, 118, 170,
         78,  65,  96, 117, 114, 128, 183, 236,
         77,  55,  75,  96,  97, 133, 169, 213,
         74,  74,  90, 121, 143, 196, 224, 202,
         97,  84, 139, 176, 227, 244, 240, 240,
    ],
    // Q15
    [
         24,  19,  14,  31,  26,  48,  41,  86,
         27,  36,  34,  26,  28,  48,  91,  49,
         47,  77,  18,  30,  34,  20,  94, 102,
         10,  26,  48,  38,  17,  84, 112, 114,
         89,  59, 111,  88, 136, 168, 128, 237,
         66, 102,  77, 111, 116, 163, 173, 185,
         65,  89, 114, 129, 121, 186, 219, 240,
        139,  96, 185, 147, 152, 208, 226, 197,
    ],
    // Q20
    [
          6,  17,  13,  17,  16,   6,  39,  61,
         13,   5,  25,  14,  34,  34,  57,  51,
         13,  17,   8,  26,  37,  49,  73, 111,
         11,  19,  22,  30,  35,  58,  91, 132,
         39,  58,  75,  60,  79, 101, 141, 191,
         44,  55,  77,  88, 105, 130, 169, 221,
         72,  70,  90, 115, 140, 168, 216, 205,
         95,  77, 139, 156, 185, 225, 218, 221,
    ],
    // Q25
    [
          8,  14,  23,  19,  34,  34,  32,  86,
         22,  28,  19,  61,  42,  33,  58,  84,
         23,  32,  28,  40,  41,  64,  80, 147,
         28,  62,  23,  33,  55,  79, 126, 160,
         34,  64,  79,  95, 101, 115, 168, 229,
         90,  76,  93, 101, 113, 166, 199, 222,
         85,  62, 118, 158, 198, 210, 206, 221,
        128, 109, 155, 197, 215, 198, 209, 221,
    ],
];

/// Find the two anchor indices that bracket the given quality.
fn find_bracket(quality: u8) -> (usize, usize, f32) {
    let q = quality.clamp(ANCHORS[0], ANCHORS[ANCHORS.len() - 1]);

    for i in 0..ANCHORS.len() - 1 {
        if q >= ANCHORS[i] && q <= ANCHORS[i + 1] {
            let t = if ANCHORS[i] == ANCHORS[i + 1] {
                0.0
            } else {
                (q - ANCHORS[i]) as f32 / (ANCHORS[i + 1] - ANCHORS[i]) as f32
            };
            return (i, i + 1, t);
        }
    }

    // Fallback to last anchor
    (ANCHORS.len() - 1, ANCHORS.len() - 1, 0.0)
}

/// Interpolate between two 64-element tables.
fn interpolate_table(a: &[u16; 64], b: &[u16; 64], t: f32) -> [f32; 64] {
    let mut result = [0.0f32; 64];
    for i in 0..64 {
        result[i] = a[i] as f32 * (1.0 - t) + b[i] as f32 * t;
    }
    result
}

/// Create EncodingTables for the given quality level (Q3-Q25 recommended).
///
/// Uses linear interpolation between anchor tables. Quality values outside
/// Q3-Q25 are clamped to the nearest anchor.
///
/// # Arguments
///
/// * `quality` - Quality level (clamped to 3-25 range)
///
/// # Returns
///
/// EncodingTables with:
/// - Interpolated quant tables (Exact scaling, no distance formula)
/// - Zero zero-bias (coefficients are already optimized for aggressive zeroing)
#[must_use]
pub fn tables_for_quality(quality: u8) -> EncodingTables {
    let (lo, hi, t) = find_bracket(quality);

    let luma = interpolate_table(&LUMA[lo], &LUMA[hi], t);
    let chroma = interpolate_table(&CHROMA[lo], &CHROMA[hi], t);

    // Zero zero-bias - the tables already handle coefficient zeroing
    let zero_mul = [0.0f32; 64];

    EncodingTables {
        quant: PerComponent {
            c0: luma,
            c1: chroma,
            c2: chroma, // Shared chroma (mozjpeg-style 2 tables)
        },
        zero_bias_mul: PerComponent {
            c0: zero_mul,
            c1: zero_mul,
            c2: zero_mul,
        },
        zero_bias_offset_dc: [0.0; 3],
        zero_bias_offset_ac: [0.0; 3],
        scaling: ScalingParams::Exact,
    }
}

/// Quality level that achieves approximately the target BPP.
///
/// Based on empirical measurements from glassa optimization.
///
/// | BPP | Quality | SSIM2 |
/// |-----|---------|-------|
/// | 0.20 | 5 | -23 |
/// | 0.25 | 7-12 | -1 to 2 |
/// | 0.30 | 13-14 | 20 |
/// | 0.50 | 20-25 | 37-46 |
#[must_use]
pub const fn quality_for_target_bpp(target_bpp: f32) -> u8 {
    if target_bpp <= 0.20 {
        5
    } else if target_bpp <= 0.25 {
        10
    } else if target_bpp <= 0.35 {
        15
    } else if target_bpp <= 0.50 {
        20
    } else {
        25
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bracket() {
        // Q3 = first anchor, returns (0, 1) with t=0 (interpolates to table[0])
        let (lo, hi, t) = find_bracket(3);
        assert_eq!(lo, 0);
        assert_eq!(hi, 1);
        assert!((t - 0.0).abs() < 0.001);

        // Q6 is between anchors Q5 (idx=1) and Q7 (idx=2)
        let (lo, hi, t) = find_bracket(6);
        assert_eq!(lo, 1); // Q5
        assert_eq!(hi, 2); // Q7
        assert!((t - 0.5).abs() < 0.001);

        // Q25 = last anchor, returns (6, 7) with t=1 (interpolates to table[7])
        let (lo, hi, t) = find_bracket(25);
        assert_eq!(lo, 6);
        assert_eq!(hi, 7);
        assert!((t - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_tables_for_quality() {
        let tables = tables_for_quality(10);

        // Should have Exact scaling
        assert!(matches!(tables.scaling, ScalingParams::Exact));

        // DC should be reasonable (not huge)
        assert!(tables.quant.c0[0] < 50.0);
        assert!(tables.quant.c1[0] < 50.0);

        // High frequencies should be high
        assert!(tables.quant.c0[63] > 100.0);
    }

    #[test]
    fn test_interpolation() {
        let t5 = tables_for_quality(5);
        let t7 = tables_for_quality(7);
        let t6 = tables_for_quality(6);

        // Q6 should be between Q5 and Q7
        for i in 0..64 {
            let min = t5.quant.c0[i].min(t7.quant.c0[i]);
            let max = t5.quant.c0[i].max(t7.quant.c0[i]);
            assert!(t6.quant.c0[i] >= min - 0.1);
            assert!(t6.quant.c0[i] <= max + 0.1);
        }
    }
}
