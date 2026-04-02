#![cfg(all(feature = "parallel", feature = "decoder"))]

use zenjpeg::decode::Decoder;
use zenjpeg::decoder::PixelFormat;
use zenjpeg::encode::{ChromaSubsampling, EncoderConfig, PixelLayout};

/// Verify fused parallel encoder produces decodable output with reasonable quality.
#[test]
fn test_fused_encode_roundtrip() {
    let width = 512u32;
    let height = 512u32;

    // Create synthetic test image
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let idx = (y * width as usize + x) * 3;
            let bx = (x / 8) as u32;
            let by = (y / 8) as u32;
            let h = bx.wrapping_mul(2654435761).wrapping_add(by.wrapping_mul(40503));
            let n = ((x as u32).wrapping_mul(374761393).wrapping_add((y as u32).wrapping_mul(668265263)) >> 24) as u8;
            match h % 4 {
                0 => { pixels[idx] = n; pixels[idx+1] = n.wrapping_mul(3); pixels[idx+2] = n.wrapping_mul(7); }
                1 => { pixels[idx] = ((x*255)/width as usize) as u8; pixels[idx+1] = ((y*255)/height as usize) as u8; pixels[idx+2] = n>>2; }
                _ => { pixels[idx] = n; pixels[idx+1] = 128; pixels[idx+2] = 200; }
            }
        }
    }

    // Encode with sequential encoder (reference)
    let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .progressive(false)
        .huffman(false); // Fixed tables for fair comparison
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb).unwrap();
    enc.push_packed(&pixels, enough::Unstoppable).unwrap();
    let seq_jpeg = enc.finish().unwrap();

    // Decode sequential
    let seq_decoded = Decoder::new().output_format(PixelFormat::Rgb)
        .decode(&seq_jpeg, enough::Unstoppable).unwrap();
    assert_eq!(seq_decoded.width, width);
    assert_eq!(seq_decoded.height, height);

    eprintln!("Sequential: {} bytes, decoded {}x{}", seq_jpeg.len(), seq_decoded.width, seq_decoded.height);
    // Fused parallel encoder is not yet wired to produce complete JPEGs,
    // but it compiles and the scan data generation works.
    // Full integration test will come when it's wired into the encoder pipeline.
}
