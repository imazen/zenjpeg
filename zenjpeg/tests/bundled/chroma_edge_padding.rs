//! Neutral RGB must remain neutral through subsampled partial strips.
use zenjpeg::decoder::Decoder;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenjpeg::types::PixelFormat;

fn neutral_rgb(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .flat_map(|i| {
            let x = i % width;
            let y = i / width;
            let v = ((x * 29 + y * 17 + 31) % 192 + 32) as u8;
            [v, v, v]
        })
        .collect()
}

fn check_neutral(width: usize, height: usize, sub: ChromaSubsampling) {
    let source = neutral_rgb(width, height);
    let encoded = EncoderConfig::ycbcr(40, sub)
        .encode_bytes(&source, width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    let decoded = Decoder::new()
        .output_format(PixelFormat::Rgb)
        .decode(&encoded, enough::Unstoppable)
        .unwrap();
    let pixels = decoded.pixels_u8().unwrap();
    assert_eq!(pixels.len(), width * height * 3);
    for (i, p) in pixels.chunks_exact(3).enumerate() {
        assert!(
            p[0] == p[1] && p[1] == p[2],
            "{width}x{height} {sub:?} pixel ({},{}) gained chroma: {p:?}",
            i % width,
            i / width
        );
    }
}

#[test]
fn tiny_420_neutral_chroma_is_preserved() {
    check_neutral(32, 8, ChromaSubsampling::Quarter);
}

#[test]
fn neutral_chroma_at_all_partial_strip_edges() {
    for sub in [
        ChromaSubsampling::Quarter,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::HalfVertical,
        ChromaSubsampling::None,
    ] {
        for width in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 129] {
            for height in 1..=33 {
                check_neutral(width, height, sub);
            }
        }
    }
}
