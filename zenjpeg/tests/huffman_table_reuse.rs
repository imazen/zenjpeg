//! Decode → re-encode Huffman table reuse (issue #77).
//!
//! `DecodedCoefficients::huffman_tables()` harvests the DHT tables from a
//! decoded JPEG; feeding them into `EncoderConfig::huffman()` gives
//! single-pass re-encoding with the source's symbol distribution.

use enough::Unstoppable;
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Deterministic noise+patches RGB image (smooth gradients are banned test
/// content: their degenerate DCT statistics make Huffman comparisons
/// meaningless).
fn test_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let n = (state >> 33) as u8;
            let base = (((x / 16) * 37 + (y / 16) * 71) % 200) as u8;
            px[i] = base ^ (n & 0x3F);
            px[i + 1] = base.wrapping_add(40) ^ ((n >> 2) & 0x3F);
            px[i + 2] = base.wrapping_add(80) ^ ((n >> 1) & 0x3F);
        }
    }
    px
}

fn decode_pixels(jpeg: &[u8]) -> Vec<u8> {
    let result = Decoder::new()
        .decode(jpeg, Unstoppable)
        .expect("decode must succeed");
    result.pixels_u8().expect("u8 pixels").to_vec()
}

/// Harvested tables fed back through `EncoderConfig::huffman()` produce a
/// decodable JPEG carrying exactly those tables, with pixel-identical
/// output (Huffman strategy affects only entropy coding, never
/// coefficients).
#[test]
fn harvested_tables_roundtrip_through_reencode() {
    let (w, h) = (128u32, 96u32);
    let rgb = test_rgb(w as usize, h as usize);
    // Baseline: custom tables are a single-pass strategy (progressive mode
    // requires per-scan optimized tables and rejects them).
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).progressive(false);

    // First encode: default two-pass Optimize builds per-image tables.
    let jpeg_a = config
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode A");

    let coeffs_a = Decoder::new()
        .decode_coefficients(&jpeg_a, Unstoppable)
        .expect("decode A coefficients");
    let harvested = coeffs_a
        .huffman_tables()
        .expect("color baseline stream must harvest tables")
        .clone();

    // Re-encode the same pixels single-pass with the harvested tables.
    let jpeg_b = config
        .huffman(harvested.clone())
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("encode B with harvested tables");

    // B must carry byte-identical DHT content to what we supplied.
    let coeffs_b = Decoder::new()
        .decode_coefficients(&jpeg_b, Unstoppable)
        .expect("decode B coefficients");
    let harvested_b = coeffs_b.huffman_tables().expect("B harvest").clone();
    for (name, ta, tb) in [
        ("dc_luma", &harvested.dc_luma, &harvested_b.dc_luma),
        ("ac_luma", &harvested.ac_luma, &harvested_b.ac_luma),
        ("dc_chroma", &harvested.dc_chroma, &harvested_b.dc_chroma),
        ("ac_chroma", &harvested.ac_chroma, &harvested_b.ac_chroma),
    ] {
        assert_eq!(ta.bits, tb.bits, "{name} bits must roundtrip");
        assert_eq!(ta.values, tb.values, "{name} values must roundtrip");
    }

    // Same input + same quantization ⇒ identical pixels out; only the
    // entropy coding path differed.
    assert_eq!(
        decode_pixels(&jpeg_a),
        decode_pixels(&jpeg_b),
        "table reuse must not change decoded pixels"
    );
}

/// Grayscale streams define no chroma tables; the harvest reuses the luma
/// tables in the chroma slots rather than failing.
#[test]
fn gray_harvest_falls_back_to_luma_tables() {
    let (w, h) = (64u32, 64u32);
    let rgb = test_rgb(w as usize, h as usize);
    let gray: Vec<u8> = rgb.as_chunks::<3>().0.iter().map(|p| p[0]).collect();

    // Baseline: the slot 0 = luma / slot 1 = chroma convention is a
    // baseline-stream property (progressive scan scripts spread tables
    // across slots per scan).
    let jpeg = EncoderConfig::grayscale(85.0)
        .progressive(false)
        .encode_bytes(&gray, w, h, PixelLayout::Gray8Srgb)
        .expect("gray encode");

    let coeffs = Decoder::new()
        .decode_coefficients(&jpeg, Unstoppable)
        .expect("decode coefficients");
    let set = coeffs
        .huffman_tables()
        .expect("gray stream must still harvest");
    assert_eq!(set.dc_chroma.bits, set.dc_luma.bits);
    assert_eq!(set.dc_chroma.values, set.dc_luma.values);
    assert_eq!(set.ac_chroma.bits, set.ac_luma.bits);
    assert_eq!(set.ac_chroma.values, set.ac_luma.values);
}

/// Progressive streams redefine DHTs per scan; the harvest captures the
/// final table state and still rebuilds an encoder-ready set.
#[test]
fn progressive_harvest_captures_final_state() {
    let (w, h) = (128u32, 96u32);
    let rgb = test_rgb(w as usize, h as usize);

    let jpeg = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter)
        .progressive(true)
        .encode_bytes(&rgb, w, h, PixelLayout::Rgb8Srgb)
        .expect("progressive encode");

    let coeffs = Decoder::new()
        .decode_coefficients(&jpeg, Unstoppable)
        .expect("decode coefficients");
    let set = coeffs
        .huffman_tables()
        .expect("progressive stream must harvest final-state tables");
    // All four tables rebuilt with non-empty code sets.
    for t in [&set.dc_luma, &set.ac_luma, &set.dc_chroma, &set.ac_chroma] {
        assert!(
            t.bits.iter().map(|&b| b as usize).sum::<usize>() > 0,
            "harvested table must define at least one code"
        );
        assert_eq!(
            t.bits.iter().map(|&b| b as usize).sum::<usize>(),
            t.values.len(),
            "bits histogram must match value count"
        );
    }
}
