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

#[test]
fn complete_real_world_zero_of_one_icc_is_readable_by_default() {
    use enough::Unstoppable;
    use zenjpeg::decoder::Decoder;
    let corpus = codec_corpus::Corpus::new()
        .unwrap()
        .get("jpeg-conformance")
        .unwrap();
    let jpeg =
        std::fs::read(corpus.join("crash-repro/jpeg-decoder-257/amazed2k1424ktv.jpg")).unwrap();
    // ExifTool extracts this complete 3,144-byte sRGB profile from the original
    // corpus file. The legacy pixel-color extractor also preserves it.
    let expected = zenjpeg::color::icc::extract_icc_profile(&jpeg).unwrap();
    assert_eq!(expected.len(), 3144);
    assert_eq!(
        u32::from_be_bytes(expected[..4].try_into().unwrap()) as usize,
        expected.len()
    );
    assert_eq!(&expected[36..40], b"acsp");
    let decoded = Decoder::new().decode(&jpeg, Unstoppable).unwrap();
    let sequence = zenjpeg::container::marker::iter(&jpeg)
        .find(|s| s.payload.starts_with(b"ICC_PROFILE\0"))
        .unwrap()
        .offset
        + 16;
    assert_eq!(jpeg[sequence], 0);
    let mut canonical = jpeg.clone();
    canonical[sequence] = 1;
    let canonical = Decoder::new().decode(&canonical, Unstoppable).unwrap();
    assert_eq!(decoded.pixels_u8(), canonical.pixels_u8());

    assert_eq!(
        decoded.extras().unwrap().icc_profile(),
        Some(expected.as_slice())
    );
    #[cfg(feature = "inventory")]
    {
        use zencodec::decode::{DecodeJob, DecodePolicy, DecoderConfig};
        use zencodec::inventory::{Disposition, MetadataKind, PartKind};
        let inv = zenjpeg::JpegDecoderConfig::new()
            .job()
            .inventory(&jpeg)
            .unwrap();
        assert!(
            inv.parts()
                .iter()
                .any(|p| p.disposition() == Disposition::Metadata(MetadataKind::Icc))
        );
        let strict_inv = zenjpeg::JpegDecoderConfig::new()
            .job()
            .with_policy(DecodePolicy::none().with_strict(true))
            .inventory(&jpeg)
            .unwrap();
        assert_eq!(
            strict_inv
                .parts()
                .iter()
                .find(|p| p.kind() == PartKind::ScanData)
                .unwrap()
                .disposition(),
            Disposition::Skipped
        );
    }
    let strict = Decoder::new()
        .strict()
        .decode(&jpeg, Unstoppable)
        .unwrap_err();
    assert!(strict.to_string().contains("ICC sequence"), "{strict}");
    #[cfg(feature = "extended-metadata")]
    assert!(
        zenjpeg::container::metadata::JpegMetadata::parse(&jpeg)
            .unwrap()
            .icc_profile()
            .is_err()
    );
    #[cfg(feature = "metadata-rewrite")]
    assert!(zenjpeg::metadata::rewrite(&zencodec::metadata_rewrite::Request::new(&jpeg)).is_err());
}
