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

#[test]
fn native_icc_reassembly_checks_the_same_chunk_contract_as_inspection() {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;

    let mut encoder = EncoderConfig::grayscale(85)
        .encode_from_bytes(8, 8, zenjpeg::encoder::PixelLayout::Gray8Srgb)
        .unwrap();
    encoder.push_packed(&[128u8; 64], Unstoppable).unwrap();
    let base = encoder.finish().unwrap();
    let cases: &[(&[(u8, u8, &[u8])], Option<&[u8]>)] = &[
        (&[(2, 2, b"def"), (1, 2, b"abc")], Some(b"abcdef")),
        (&[(1, 1, b"abc")], Some(b"abc")),
        (&[(1, 2, b"abc")], None),
        (&[(1, 2, b"abc"), (1, 2, b"def")], None),
        (&[(1, 2, b"abc"), (2, 3, b"def")], None),
        (&[(0, 1, b"abc")], None),
        (&[(2, 1, b"abc")], None),
        (&[(1, 0, b"abc")], None),
        (&[(1, 1, b"")], None),
    ];
    for &(chunks, expected) in cases {
        let mut jpeg = base[..2].to_vec();
        for &(sequence, count, payload) in chunks {
            let len = u16::try_from(16 + payload.len()).unwrap();
            jpeg.extend_from_slice(&[0xff, 0xe2]);
            jpeg.extend_from_slice(&len.to_be_bytes());
            jpeg.extend_from_slice(b"ICC_PROFILE\0");
            jpeg.extend_from_slice(&[sequence, count]);
            jpeg.extend_from_slice(payload);
        }
        jpeg.extend_from_slice(&base[2..]);
        let (_, extras) = Decoder::new()
            .decode_coefficients_with_extras(&jpeg, Unstoppable)
            .unwrap();
        assert_eq!(extras.unwrap().icc_profile(), expected, "{chunks:?}");
        #[cfg(feature = "extended-metadata")]
        {
            let parsed = zenjpeg::container::metadata::JpegMetadata::parse(&jpeg).unwrap();
            let inspected = parsed.icc_profile();
            match expected {
                Some(bytes) => assert_eq!(inspected.unwrap().unwrap().as_ref(), bytes),
                None => assert!(inspected.is_err(), "{chunks:?}"),
            }
        }
    }
}
