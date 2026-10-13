#![cfg(feature = "metadata-rewrite")]
// Reused unchanged expectations from zencodecs at aa21cba0.
use zencodec::metadata_rewrite::{Action, Error, Request as ScrubRequest};
trait PlanRequest<'a> {
    fn plan(&self) -> Result<zencodec::metadata_rewrite::Plan<'a>, Error>;
}
impl<'a> PlanRequest<'a> for ScrubRequest<'a> {
    fn plan(&self) -> Result<zencodec::metadata_rewrite::Plan<'a>, Error> {
        zenjpeg::metadata::rewrite(self)
    }
}
mod jpeg {
    use super::*;
    fn app(marker: u8, bytes: &[u8]) -> Vec<u8> {
        let mut out = vec![255, marker];
        out.extend_from_slice(&((bytes.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(bytes);
        out
    }
    fn image(metadata: &[u8]) -> Vec<u8> {
        [
            &[255, 216][..],
            metadata,
            &[255, 218, 0, 2, 17, 255, 0, 29, 255, 217],
        ]
        .concat()
    }
    #[test]
    fn removes_private_packets_between_scans_and_trailer_without_copying_scan() {
        let mut input = image(&app(0xfe, b"SECRET-COMMENT"));
        input.extend_from_slice(b"SECRET-TRAILER");
        let request = ScrubRequest::new(&input);
        let plan = request.plan().unwrap();
        assert_eq!(plan.to_vec().unwrap(), image(&[]));
        assert_eq!(
            plan.output_len(),
            plan.chunks().map(<[u8]>::len).sum::<usize>()
        );
        assert!(plan.chunks().any(|c| c.as_ptr() == input[20..].as_ptr()));
        assert!(plan.report().iter().any(|r| r.action() == Action::Remove));
        let mut stream = Vec::new();
        for bytes in plan.chunks() {
            stream.extend_from_slice(bytes);
        }
        assert_eq!(stream, plan.to_vec().unwrap());
    }
    #[test]
    fn retains_orientation_removes_identity_and_refuses_duplicate_exif() {
        let mut exif = zencodec::exif::Exif::new(zencodec::exif::TextEncoding::Ascii);
        exif.set_orientation(zencodec::Orientation::Rotate90);
        exif.set_artist("SECRET-PERSON");
        let tiff = zenmeta::exif::serialize(&exif);
        let exif = if tiff.starts_with(b"Exif\0\0") {
            tiff
        } else {
            [b"Exif\0\0".as_slice(), &tiff].concat()
        };
        let segment = app(0xe1, &exif);
        let input = image(&segment);
        let output = ScrubRequest::new(&input).plan().unwrap().to_vec().unwrap();
        assert!(!output.windows(6).any(|w| w == b"SECRET"));
        let span = zenjpeg::container::iter(&output)
            .find(|s| s.payload.starts_with(b"Exif\0\0"))
            .unwrap();
        assert_eq!(
            zencodec::exif::Exif::parse(span.payload)
                .unwrap()
                .orientation(),
            Some(zencodec::Orientation::Rotate90)
        );
        assert!(
            ScrubRequest::new(&image(&[&segment[..], &segment].concat()))
                .plan()
                .is_err()
        );
    }
    #[test]
    fn damaged_orientation_and_cyclic_exif_are_not_silently_salvaged() {
        let valid = b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        for index in [12, 18, 22] {
            let mut damaged = valid.to_vec();
            damaged[index] = if index == 22 { 8 } else { 255 };
            let input = image(&app(0xe1, &[b"Exif\0\0".as_slice(), &damaged].concat()));
            assert!(ScrubRequest::new(&input).plan().is_err(), "{index}");
        }
    }
    #[test]
    fn profiles_are_explicitly_opaque_and_incomplete_chunks_refuse() {
        let input = image(&app(0xe2, b"ICC_PROFILE\0\x01\x01opaque-identity"));
        let request = ScrubRequest::new(&input);
        let plan = request.plan().unwrap();
        assert_eq!(plan.to_vec().unwrap(), input);
        assert!(
            plan.report()
                .iter()
                .any(|r| r.action() == Action::OpaqueIcc)
        );
        let input = image(&app(0xe2, b"ICC_PROFILE\0\x01\x02incomplete"));
        assert!(ScrubRequest::new(&input).plan().is_err());
    }
    #[test]
    fn truncations_and_resource_budgets_never_produce_partial_plans() {
        let input = image(&app(0xfe, b"private"));
        for end in 0..input.len() {
            assert!(ScrubRequest::new(&input[..end]).plan().is_err(), "{end}");
        }
        assert!(matches!(
            ScrubRequest::new(&input).with_byte_limits(100, 2).plan(),
            Err(Error::Limit(_))
        ));
        assert!(matches!(
            ScrubRequest::new(&input).with_carrier_limit(1).plan(),
            Err(Error::Limit(_))
        ));
    }
    #[test]
    fn does_not_silently_drop_gain_maps_or_indexed_images() {
        let input = image(&app(0xe2, b"MPF\0not-a-valid-index"));
        assert!(ScrubRequest::new(&input).plan().is_err());
        assert!(
            ScrubRequest::new(&[image(&[]), image(&[])].concat())
                .plan()
                .is_err()
        );
    }
}

#[cfg(feature = "ultrahdr")]
mod gain_map_scrub {
    use super::*;
    use zenjpeg::container::{MarkerKind, MpImageType, marker, mpf};
    fn source() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16 * 16 * 3)
            .map(|i| ((i * 37 + i / 7 * 53) % 256) as u8)
            .collect();
        let gain = zenjpeg::ultrahdr::GainMap {
            width: 4,
            height: 4,
            channels: 1,
            data: vec![128; 16],
        };
        let mut params = zenjpeg::ultrahdr::GainMapMetadata::default();
        params.alternate_hdr_headroom = 2.0;
        for channel in &mut params.channels {
            channel.max = 2.0;
        }
        let sdr =
            zenpixels::PixelBuffer::from_vec(pixels, 16, 16, zenpixels::PixelDescriptor::RGB8_SRGB)
                .unwrap();
        zenjpeg::ultrahdr::encode_with_gainmap_format(
            &sdr,
            &gain,
            &params,
            &zenjpeg::encoder::EncoderConfig::ycbcr(
                85.0,
                zenjpeg::encoder::ChromaSubsampling::Quarter,
            ),
            75.0,
            zenjpeg::ultrahdr::GainMapEncodingFormat::Both,
            enough::Unstoppable,
        )
        .unwrap()
    }
    fn add_private_comments(jpeg: &[u8]) -> Vec<u8> {
        let entries = mpf::parse_mpf(jpeg).unwrap();
        let private = b"\xff\xfe\0\x08SECRET";
        let gain = &jpeg[entries[1].offset..entries[1].offset + entries[1].size];
        let gain = [&gain[..2], private, &gain[2..]].concat();
        let mut base = vec![255, 216];
        base.extend_from_slice(private);
        let mut mpf_offset = 0;
        for span in marker::iter(&jpeg[..entries[0].size]).skip(1) {
            if span.payload.starts_with(b"MPF\0") {
                mpf_offset = base.len();
                base.extend_from_slice(&jpeg[span.offset..span.offset + span.length]);
            } else if span.payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0") {
                let prefix = b"http://ns.adobe.com/xap/1.0/\0";
                let xml = std::str::from_utf8(&span.payload[prefix.len()..]).unwrap();
                let xml = xml.replace(
                    &format!("Item:Length=\"{}\"", entries[1].size),
                    &format!("Item:Length=\"{}\"", gain.len()),
                );
                base.extend_from_slice(&zenjpeg::container::xmp::create_xmp_app1_marker(&xml));
            } else {
                base.extend_from_slice(&jpeg[span.offset..span.offset + span.length]);
            }
        }
        let new_mpf = mpf::create_mpf_header_typed(
            base.len(),
            &[(MpImageType::Undefined, gain.len())],
            Some(mpf_offset),
        );
        let old_len = marker::iter(&base)
            .find(|s| s.payload.starts_with(b"MPF\0"))
            .unwrap()
            .length;
        assert_eq!(old_len, new_mpf.len());
        base[mpf_offset..mpf_offset + old_len].copy_from_slice(&new_mpf);
        [base, gain, b"SECRET-trailer".to_vec()].concat()
    }
    fn scans(jpeg: &[u8]) -> Vec<Vec<u8>> {
        mpf::parse_mpf(jpeg)
            .unwrap()
            .into_iter()
            .flat_map(|e| {
                let data = &jpeg[e.offset..e.offset + e.size];
                marker::iter(data)
                    .filter(|s| s.kind == MarkerKind::Sos)
                    .map(|s| data[s.offset..s.offset + s.length].to_vec())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    #[test]
    fn scrubs_both_components_without_changing_scans_parameters_or_reconstructed_hdr() {
        let clean = source();
        let input = add_private_comments(&clean);
        let req = ScrubRequest::new(&input);
        let plan = req.plan().unwrap();
        let output = plan.to_vec().unwrap();
        zencodec_testkit::check_metadata_rewrite(
            req.clone(),
            zenjpeg::metadata::rewrite,
            |candidate| {
                assert_eq!(
                    candidate, output,
                    "shared bounded planner must retain both image components"
                );
                assert_eq!(scans(&input), scans(candidate));
                Ok(())
            },
        )
        .unwrap();
        assert!(!output.windows(6).any(|b| b == b"SECRET"));
        assert_eq!(scans(&input), scans(&output));
        assert_eq!(
            plan.report()
                .iter()
                .filter(|e| e.carrier() == "private JPEG marker")
                .count(),
            2
        );
        let a = zenjpeg::ultrahdr::decode_ultrahdr_hdr(
            &input,
            4.0,
            zenjpeg::ultrahdr::HdrOutputFormat::LinearFloat,
        )
        .unwrap();
        let b = zenjpeg::ultrahdr::decode_ultrahdr_hdr(
            &output,
            4.0,
            zenjpeg::ultrahdr::HdrOutputFormat::LinearFloat,
        )
        .unwrap();
        assert_eq!(
            a.as_slice().contiguous_bytes(),
            b.as_slice().contiguous_bytes()
        );
        use zenjpeg::ultrahdr::UltraHdrExtras;
        let a = zenjpeg::decoder::Decoder::new()
            .decode(&input, enough::Unstoppable)
            .unwrap();
        let b = zenjpeg::decoder::Decoder::new()
            .decode(&output, enough::Unstoppable)
            .unwrap();
        assert_eq!(
            a.extras().unwrap().ultrahdr_metadata().unwrap().unwrap().0,
            b.extras().unwrap().ultrahdr_metadata().unwrap().unwrap().0
        );
        if let Some(directory) = std::env::var_os("JPEG_METADATA_OUTPUT") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            for (name, jpeg) in [("before", &input), ("after", &output)] {
                std::fs::write(directory.join(format!("{name}.jpg")), jpeg).unwrap();
                let map = mpf::parse_mpf(jpeg).unwrap()[1];
                std::fs::write(
                    directory.join(format!("{name}-map.jpg")),
                    &jpeg[map.offset..map.offset + map.size],
                )
                .unwrap();
            }
        }
    }
    #[test]
    fn gain_map_editing_is_explicit_and_cannot_replace_required_signaling() {
        use core::cell::Cell;
        use zencodec::metadata_rewrite::Services;
        struct Editor {
            calls: Cell<usize>,
            corrupt: bool,
        }
        impl Services for Editor {
            fn rewrite_xmp(
                &self,
                _: &[u8],
                _: usize,
                _: &dyn enough::Stop,
            ) -> Result<Option<Vec<u8>>, Error> {
                self.calls.set(self.calls.get() + 1);
                Ok(Some(b"description".to_vec()))
            }
            fn merge_xmp(
                &self,
                _: &[u8],
                required: &[u8],
                _: usize,
                _: &dyn enough::Stop,
            ) -> Result<Vec<u8>, Error> {
                Ok(if self.corrupt {
                    b"<invalid/>".to_vec()
                } else {
                    required.to_vec()
                })
            }
        }
        let input = source();
        let service = Editor {
            calls: Cell::new(0),
            corrupt: false,
        };
        ScrubRequest::new(&input)
            .with_services(&service)
            .plan()
            .unwrap();
        assert_eq!(service.calls.get(), 0);
        let output = ScrubRequest::new(&input)
            .with_services(&service)
            .with_xmp_editing(true)
            .plan()
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(service.calls.get(), 2);
        assert_eq!(scans(&input), scans(&output));
        let corrupt = Editor {
            calls: Cell::new(0),
            corrupt: true,
        };
        assert!(
            ScrubRequest::new(&input)
                .with_services(&corrupt)
                .with_xmp_editing(true)
                .plan()
                .is_err()
        );
    }
    #[test]
    fn invalid_authoritative_iso_never_falls_back_to_valid_xmp() {
        let mut input = source();
        let e = mpf::parse_mpf(&input).unwrap()[1];
        let iso = marker::iter(&input[e.offset..e.offset + e.size])
            .find(|s| s.payload.starts_with(zencodec::ISO_21496_1_URN))
            .unwrap();
        let version = e.offset + iso.offset + 4 + zencodec::ISO_21496_1_URN.len();
        input[version] = 255;
        assert!(ScrubRequest::new(&input).plan().is_err());
    }
}
