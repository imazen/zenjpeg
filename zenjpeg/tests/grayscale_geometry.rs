//! Gray input under a color configuration must still use one-block MCUs.
//! The 97x65 baseline fixture failed jpeg-decoder's RST0 check before this fix:
//! SOF said 1x1 grayscale, while counts/emission used 4:2:0 block order.
#![cfg(feature = "zencodec")]
use zencodec::encode::EncoderConfig;
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice};

#[test]
fn color_configuration_gray_input_is_valid_and_matches_explicit_gray_samples() {
    for (width, height) in [(97u32, 65u32), (129, 113), (128, 112)] {
        let stride = width as usize + 17;
        let raw: Vec<u8> = (0..stride * height as usize)
            .map(|i| (i ^ (i / stride * 13)) as u8)
            .collect();
        for progressive in [false, true] {
            let pixels = || {
                PixelSlice::new(&raw, width, height, stride, PixelDescriptor::GRAY8_SRGB).unwrap()
            };
            let automatic = JpegEncoderConfig::new()
                .with_calibrated_quality(87.0)
                .with_progressive(progressive)
                .encode(pixels())
                .unwrap();
            let explicit = JpegEncoderConfig::grayscale(87.0)
                .with_progressive(progressive)
                .encode(pixels())
                .unwrap();
            let decode = |data: &[u8]| {
                let mut oracle = jpeg_decoder::Decoder::new(std::io::Cursor::new(data));
                let pixels = oracle
                    .decode()
                    .expect("independent restart/entropy readback");
                let info = oracle.info().unwrap();
                assert_eq!(
                    (u32::from(info.width), u32::from(info.height)),
                    (width, height)
                );
                assert_eq!(pixels.len(), width as usize * height as usize);
                pixels
            };
            assert_eq!(decode(automatic.data()), decode(explicit.data()));
            let actual = JpegDecoderConfig::new().decode(automatic.data()).unwrap();
            let expected = JpegDecoderConfig::new().decode(explicit.data()).unwrap();
            for y in 0..height {
                assert_eq!(actual.pixels().row(y), expected.pixels().row(y));
            }
        }
    }
}
