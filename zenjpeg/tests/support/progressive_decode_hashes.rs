//! Shared progressive witnesses and transitive decode identities.
//! Native pinned-turbo validation is selected explicitly by the oracle job;
//! ordinary builds, including i686, verify these same RGB bytes in process.
use sha2::{Digest, Sha256};
use zenjpeg::decoder::{ChromaUpsampling, Decoder, IdctMethod, OutputTarget, PixelFormat};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

pub const WIDTH: u32 = 64;
pub const HEIGHT: u32 = 64;
// SHA-256 of tightly packed RGB8, Libjpeg IDCT, nearest-neighbor chroma,
// Srgb8 output. Validated against SHA-pinned libjpeg-turbo 3.2.0; native CI revalidates.
pub const RGB_SHA256: &str = "ac818db8e52b3552ddef18c36befc2cdaf328e14169de1eb87a9b26d7790bac8";

pub fn encode_noise() -> Vec<u8> {
    // Keep the original deterministic noise and progressive Q50/420 settings.
    let data: Vec<u8> = (0..HEIGHT)
        .flat_map(|y| {
            (0..WIDTH).flat_map(move |x| {
                let r = ((x.wrapping_mul(17) ^ y.wrapping_mul(31)) % 256) as u8;
                let g = ((x.wrapping_mul(13) ^ y.wrapping_mul(23)) % 256) as u8;
                let b = ((x.wrapping_mul(11) ^ y.wrapping_mul(19)) % 256) as u8;
                [r, g, b]
            })
        })
        .collect();
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .quality(50.0)
        .progressive(true)
        .optimize_huffman(true);
    let mut enc = config
        .encode_from_bytes(WIDTH, HEIGHT, PixelLayout::Rgb8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

pub fn check_noise_decode(jpeg: &[u8]) {
    check_decode(jpeg, WIDTH, HEIGHT, RGB_SHA256);
}

fn check_decode(jpeg: &[u8], width: u32, height: u32, expected: &str) {
    let decoded = Decoder::new()
        .idct_method(IdctMethod::Libjpeg)
        .chroma_upsampling(ChromaUpsampling::NearestNeighbor)
        .output_target(OutputTarget::Srgb8)
        .output_format(PixelFormat::Rgb)
        .decode(jpeg, Unstoppable)
        .unwrap();
    assert_eq!((decoded.width, decoded.height), (width, height));
    let pixels = decoded.pixels_u8().unwrap();
    assert_eq!(pixels.len(), width as usize * height as usize * 3);
    assert_eq!(
        format!("{:x}", Sha256::digest(pixels)),
        expected,
        "progressive RGB identity changed; validate with the pinned native oracle before updating"
    );
}

pub const GRADIENT_SIZE: u32 = 16;
pub const GRADIENT_RGB_SHA256: &str =
    "b13fef4a911b01697164f8190c7b710c32b610cfb496a4ab076a3d08cecd22ca";

pub fn encode_gradient() -> Vec<u8> {
    let data: Vec<u8> = (0..GRADIENT_SIZE)
        .flat_map(|_| (0..GRADIENT_SIZE).map(|x| (x * 16) as u8))
        .collect();
    let config = EncoderConfig::grayscale(90.0).progressive(true);
    let mut enc = config
        .encode_from_bytes(GRADIENT_SIZE, GRADIENT_SIZE, PixelLayout::Gray8Srgb)
        .unwrap();
    enc.push_packed(&data, Unstoppable).unwrap();
    enc.finish().unwrap()
}

pub fn check_gradient_decode(jpeg: &[u8]) {
    check_decode(jpeg, GRADIENT_SIZE, GRADIENT_SIZE, GRADIENT_RGB_SHA256);
}
