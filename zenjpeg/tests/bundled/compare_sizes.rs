use enough::Unstoppable;
use zenjpeg::{
    decoder::Decoder,
    encoder::{ChromaSubsampling, EncoderConfig, PixelLayout},
};

fn create_gradient(size: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (size * size * 3) as usize];
    for y in 0..size {
        for x in 0..size {
            let val = ((x + y) * 255 / (size * 2 - 2)) as u8;
            let idx = ((y * size + x) * 3) as usize;
            pixels[idx] = val;
            pixels[idx + 1] = val;
            pixels[idx + 2] = val;
        }
    }
    pixels
}

fn encode_rgb(
    width: u32,
    height: u32,
    data: &[u8],
    config: &EncoderConfig,
) -> zenjpeg::encoder::Result<Vec<u8>> {
    let mut enc = config.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(data, enough::Unstoppable)?;
    enc.finish()
}

fn test_size(size: u32, pixels: &[u8]) {
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
    let jpeg = encode_rgb(size, size, pixels, &config).expect("encode 4:2:0 fixture");
    let decoded = Decoder::new()
        .decode(&jpeg, Unstoppable)
        .expect("decode 4:2:0 fixture");
    assert_eq!((decoded.width(), decoded.height()), (size, size));
    let mut oracle = jpeg_decoder::Decoder::new(std::io::Cursor::new(&jpeg));
    let expected = oracle.decode().expect("independent fixture decode");
    assert_eq!(expected.len(), (size * size * 3) as usize);
    // jpeg-decoder uses a different fixed-point IDCT. Exact pixels are
    // compared with the libjpeg-turbo oracle matching the default decoder.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (width, height, expected) = super::permutation_regression::decode_mozjpeg_rgb(&jpeg)
            .expect("libjpeg-turbo fixture decode");
        assert_eq!((width, height), (size, size));
        assert_eq!(decoded.pixels_u8().unwrap(), expected, "{size}x{size}");
    }
}

#[test]
fn compare_420_sizes() {
    println!("Testing 4:2:0 at various sizes:");
    for size in [16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256] {
        test_size(size, &create_gradient(size));
        test_size(
            size,
            &super::coeff_unification::color_noise(size as usize, size as usize),
        );
        test_size(
            size,
            &super::coeff_unification::smooth_chroma_bands(size as usize, size as usize),
        );
    }
}
