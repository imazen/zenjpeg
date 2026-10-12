//! JPEG carrier boundaries must reject oversized metadata without losing bytes.
use zenjpeg::encode::EncoderSegments;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, Exif};

#[test]
fn exif_boundary_is_exact_for_raw_and_prepared_segments() {
    for len in [65_527, 65_528] {
        for prefix in [false, true] {
            for prepared in [false, true] {
                let tiff = vec![42; len];
                let data = if prefix {
                    [b"Exif\0\0".as_slice(), &tiff].concat()
                } else {
                    tiff.clone()
                };
                let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None);
                let request = if prepared {
                    config
                        .request()
                        .segments(EncoderSegments::new().set_exif(data))
                } else {
                    config.request().exif(Exif::raw(data))
                };
                let result = request.encode(&[rgb::RGB8::new(20, 40, 60); 64], 8, 8);
                if len > 65_527 {
                    assert!(
                        result.is_err(),
                        "oversized EXIF accepted: prepared={prepared} prefix={prefix}"
                    );
                } else {
                    let jpeg = img_parts::jpeg::Jpeg::from_bytes(result.unwrap().into()).unwrap();
                    use img_parts::ImageEXIF;
                    assert_eq!(
                        jpeg.exif().unwrap().as_ref(),
                        tiff,
                        "prepared={prepared} prefix={prefix}"
                    );
                }
            }
        }
    }
}

#[test]
fn oversized_prepared_comment_returns_error() {
    let config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::None)
        .with_segments(EncoderSegments::new().add_comment(&"x".repeat(65_534)));
    assert!(
        config
            .request()
            .encode(&[rgb::RGB8::new(20, 40, 60); 64], 8, 8)
            .is_err()
    );
}
