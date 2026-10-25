//! Native qualification of required interpretation decisions. This feature is
//! explicitly whole-source. Native workspace is outside its storage allowance.
#![cfg(feature = "controlled-decode")]
use std::borrow::Cow;
use std::convert::Infallible;
use zencodec::{
    At, CodecSet, Orientation, OrientationHint,
    decode::*,
    encode::{EncodeJob, EncoderConfig},
    operation::{Decision, Decline, DecodeRequest, PolicySet, Question},
    source::SliceReadAt,
    storage::StorageBudget,
};
use zencodec_io::{CollectSink, DriveOptions, PolicyDecoder, drive_from_read_at, drive_from_slice};
use zenjpeg::{JpegDecoderConfig, JpegEncoderConfig};
use zenpixels::{PixelDescriptor, PixelSlice, PixelSliceMut};

const WIDTH: u32 = 97;
const HEIGHT: u32 = 65;
fn base(progressive: bool) -> Vec<u8> {
    let stride = WIDTH as usize * 3 + 21;
    let mut bytes = vec![0xa5; stride * HEIGHT as usize];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize * 3 {
            bytes[y * stride + x] = (x ^ (y * 7)) as u8;
        }
    }
    JpegEncoderConfig::new()
        .with_calibrated_quality(87.0)
        .with_progressive(progressive)
        .encode(PixelSlice::new(&bytes, WIDTH, HEIGHT, stride, PixelDescriptor::RGB8_SRGB).unwrap())
        .unwrap()
        .into_vec()
}
fn part(bytes: &[u8], number: u8, total: u8) -> Vec<u8> {
    let mut payload = b"ICC_PROFILE\0".to_vec();
    payload.extend([number, total]);
    payload.extend(bytes);
    let mut segment = vec![0xff, 0xe2];
    segment.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    segment.extend(payload);
    segment
}
fn profile() -> Vec<u8> {
    use zenpixels_convert::icc_profiles::{SynthesizedIcc, synthesize_icc_for_cicp};
    match synthesize_icc_for_cicp(zencodec::Cicp::DISPLAY_P3) {
        SynthesizedIcc::Profile(bytes) => bytes.into_owned(),
        other => panic!("{other:?}"),
    }
}
fn with_profile(bytes: &[u8], late: bool, split: bool) -> Vec<u8> {
    let profile = profile();
    let mid = profile.len() / 2;
    let first = part(&profile[..mid], 1, 2);
    let second = part(&profile[mid..], 2, 2);
    if late {
        [
            &bytes[..2],
            if split { first.as_slice() } else { &[] },
            &bytes[2..bytes.len() - 2],
            if split { &[] } else { first.as_slice() },
            &second,
            &bytes[bytes.len() - 2..],
        ]
        .concat()
    } else {
        [&bytes[..2], &second, &first, &bytes[2..]].concat()
    }
}
#[derive(Default)]
struct Sink {
    output: CollectSink,
    effects: usize,
}
impl DecodeRowSink for Sink {
    fn begin(&mut self, w: u32, h: u32, d: PixelDescriptor) -> Result<(), SinkError> {
        self.effects += 1;
        self.output.begin(w, h, d)
    }
    fn provide_next_buffer(
        &mut self,
        y: u32,
        h: u32,
        w: u32,
        d: PixelDescriptor,
    ) -> Result<PixelSliceMut<'_>, SinkError> {
        self.effects += 1;
        self.output.provide_next_buffer(y, h, w, d)
    }
    fn finish(&mut self) -> Result<(), SinkError> {
        self.effects += 1;
        self.output.finish()
    }
}
fn budget() -> StorageBudget {
    StorageBudget::new(1 << 20, 4)
}

#[test]
fn generic_erased_and_registered_paths_match_native_pixels() {
    for progressive in [false, true] {
        let base = base(progressive);
        for bytes in [base.clone(), with_profile(&base, false, false)] {
            for descriptor in [
                PixelDescriptor::RGB8_SRGB,
                PixelDescriptor::RGBA8_SRGB,
                PixelDescriptor::BGRA8_SRGB,
                PixelDescriptor::RGBX8_SRGB,
                PixelDescriptor::BGRX8_SRGB,
            ] {
                let config = JpegDecoderConfig::new();
                let direct = config
                    .clone()
                    .job()
                    .decoder(Cow::Borrowed(&bytes), &[descriptor])
                    .unwrap()
                    .decode()
                    .unwrap();
                for route in 0..3 {
                    let budget = budget();
                    let set = CodecSet::new().with_decoder(config.clone());
                    let native = match route {
                        0 => config
                            .clone()
                            .job()
                            .dyn_controlled_decoder(&[descriptor], budget.clone())
                            .unwrap(),
                        1 => set
                            .decode_job(zencodec::ImageFormat::Jpeg)
                            .unwrap()
                            .into_controlled_decoder(&[descriptor], budget.clone())
                            .unwrap(),
                        _ => DecodeRequest::new()
                            .preferring_output(vec![descriptor])
                            .start_controlled(&set, budget.clone())
                            .unwrap(),
                    };
                    let mut resolver =
                        |question: Question<'_>| -> Result<Decision, At<Infallible>> {
                            let Question::Boundary { choices, .. } = question else {
                                panic!()
                            };
                            assert_eq!(choices.len(), 1);
                            Ok(Decision::UsePreset)
                        };
                    let mut session = PolicyDecoder::new(native, &mut resolver);
                    let mut sink = Sink::default();
                    if route == 1 {
                        drive_from_read_at(
                            &mut session,
                            &mut SliceReadAt::new(&bytes),
                            &mut sink,
                            None,
                            DriveOptions::new(0, 37),
                        )
                        .unwrap();
                    } else {
                        drive_from_slice(&mut session, &bytes, &mut sink, None, 7).unwrap();
                    }
                    assert_eq!(session.report().decisions().count(), 2);
                    let pixels = sink.output.as_pixel_slice().unwrap();
                    assert_eq!(pixels.descriptor(), direct.pixels().descriptor());
                    for y in 0..HEIGHT {
                        assert_eq!(
                            pixels.row(y),
                            direct.pixels().row(y),
                            "{progressive}/{descriptor:?}/{route}/{y}"
                        );
                    }
                    assert_eq!((budget.reserved_bytes(), budget.reserved_objects()), (0, 0));
                }
            }
        }
    }
}

#[test]
fn unavailable_profiles_and_policy_refusal_have_no_sink_effects() {
    for progressive in [false, true] {
        let base = base(progressive);
        for (bytes, ignore_icc, decline) in [
            (with_profile(&base, true, false), false, false),
            (with_profile(&base, true, true), false, false),
            (with_profile(&base, false, false), true, false),
            (base.clone(), false, true),
        ] {
            let storage = budget();
            let job = JpegDecoderConfig::new()
                .job()
                .try_with_policy(DecodePolicy::none().with_allow_icc(!ignore_icc))
                .unwrap();
            let native = job.dyn_controlled_decoder(&[], storage.clone()).unwrap();
            let mut unavailable = false;
            let mut resolver = |q: Question<'_>| -> Result<Decision, At<Infallible>> {
                let Question::Boundary { facts, choices } = q else {
                    panic!()
                };
                if facts.state == InterpretationState::Unavailable {
                    unavailable = true;
                    assert!(choices.is_empty());
                    assert!(matches!(
                        facts.unavailable,
                        Some(
                            InterpretationUnavailable::LateColorMetadata
                                | InterpretationUnavailable::ForbiddenColorMetadata
                        )
                    ));
                }
                Ok(if decline {
                    Decision::decline(Decline::ApplicationRule)
                } else {
                    Decision::UsePreset
                })
            };
            let mut session = PolicyDecoder::new(native, &mut resolver);
            let mut sink = Sink::default();
            assert!(drive_from_slice(&mut session, &bytes, &mut sink, None, 13).is_err());
            assert_eq!(sink.effects, 0);
            assert_eq!(storage.reserved_bytes(), 0);
            drop(session);
            assert!(decline || unavailable);
        }
    }
}

#[test]
fn late_orientation_is_bound_before_adoption_and_matches_direct_decode() {
    let mut exif = zencodec::exif::Exif::new(zencodec::exif::TextEncoding::Ascii);
    exif.set_orientation(Orientation::Rotate90);
    let metadata = zencodec::Metadata::none().with_exif(zenmeta::exif::serialize(&exif));
    let raw = (0..WIDTH as usize * HEIGHT as usize * 3)
        .map(|i| (i.wrapping_mul(37) ^ (i / (WIDTH as usize * 3)).wrapping_mul(11)) as u8)
        .collect::<Vec<_>>();
    let encoded = JpegEncoderConfig::new()
        .job()
        .with_metadata_policy(metadata, zencodec::MetadataPolicy::PreserveExact)
        .encode(
            PixelSlice::new(
                &raw,
                WIDTH,
                HEIGHT,
                WIDTH as usize * 3,
                PixelDescriptor::RGB8_SRGB,
            )
            .unwrap(),
        )
        .unwrap()
        .into_vec();
    let exif_span = zenjpeg::container::marker::iter(&encoded)
        .find(|s| s.payload.starts_with(b"Exif\0\0"))
        .unwrap();
    let mut bytes = encoded[..exif_span.offset].to_vec();
    bytes.extend_from_slice(&encoded[exif_span.offset + exif_span.length..encoded.len() - 2]);
    bytes.extend_from_slice(&encoded[exif_span.offset..exif_span.offset + exif_span.length]);
    bytes.extend([0xff, 0xd9]);
    for orientation in [OrientationHint::Preserve, OrientationHint::Correct] {
        let direct = JpegDecoderConfig::new()
            .job()
            .with_orientation(orientation)
            .decoder(Cow::Borrowed(&bytes), &[])
            .unwrap()
            .decode()
            .unwrap();
        let native = JpegDecoderConfig::new()
            .job()
            .with_orientation(orientation)
            .dyn_controlled_decoder(&[], budget())
            .unwrap();
        let mut session = PolicyDecoder::new(native, PolicySet::new());
        let mut sink = CollectSink::new();
        drive_from_slice(&mut session, &bytes, &mut sink, None, 11).unwrap();
        let actual = sink.as_pixel_slice().unwrap();
        assert_eq!(
            (actual.width(), actual.rows()),
            (direct.width(), direct.height())
        );
        assert_eq!(
            session.info().unwrap().orientation,
            direct.info().orientation
        );
        for y in 0..actual.rows() {
            assert_eq!(actual.row(y), direct.pixels().row(y));
        }
    }
}

#[test]
fn admitted_storage_is_released_on_native_error_refusal_and_unwind() {
    use zencodec::storage::StorageError;
    for progressive in [false, true] {
        let bytes = base(progressive);
        for objects in [2, 3] {
            let storage = StorageBudget::new(1 << 20, objects);
            let native = JpegDecoderConfig::new()
                .job()
                .dyn_controlled_decoder(&[], storage.clone())
                .unwrap();
            let mut session = PolicyDecoder::new(native, PolicySet::new());
            let mut sink = Sink::default();
            let error = drive_from_slice(&mut session, &bytes, &mut sink, None, 31).unwrap_err();
            assert!(zencodec::find_cause::<StorageError>(&error).is_some());
            assert_eq!(sink.effects, 0);
            assert_eq!(
                (storage.reserved_bytes(), storage.reserved_objects()),
                (0, 0)
            );
        }
        let storage = budget();
        let native = JpegDecoderConfig::new()
            .job()
            .dyn_controlled_decoder(&[], storage.clone())
            .unwrap();
        let mut session = PolicyDecoder::new(native, PolicySet::new());
        let mut sink = Sink::default();
        let error = drive_from_slice(&mut session, &bytes[..bytes.len() - 2], &mut sink, None, 31)
            .unwrap_err();
        assert!(zencodec::find_cause::<zencodec::CodecError>(&error).is_some());
        assert_eq!(sink.effects, 0);
        assert_eq!(
            (storage.reserved_bytes(), storage.reserved_objects()),
            (0, 0)
        );

        let native = JpegDecoderConfig::new()
            .job()
            .dyn_controlled_decoder(&[], storage.clone())
            .unwrap();
        let mut session = PolicyDecoder::new(
            native,
            |q: Question<'_>| -> Result<Decision, At<Infallible>> {
                if let Question::Boundary { facts, .. } = q
                    && facts.state == InterpretationState::Final
                {
                    panic!("application refusal panic");
                }
                Ok(Decision::UsePreset)
            },
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive_from_slice(
                &mut session,
                &bytes,
                &mut sink,
                None,
                31
            )))
            .is_err()
        );
        assert_eq!(sink.effects, 0);
        assert_eq!(
            (storage.reserved_bytes(), storage.reserved_objects()),
            (0, 0)
        );
        assert!(matches!(
            session.step(Input::empty(), &mut sink, &zencodec::enough::Unstoppable),
            Err(StepError::Finished)
        ));
    }
}

#[test]
fn implicit_conversion_and_crop_are_refused_before_input() {
    let mut corrected = JpegDecoderConfig::new();
    *corrected.inner_mut() = corrected
        .inner()
        .clone()
        .correct_color(Some(zenjpeg::decoder::TargetColorSpace::Srgb));
    for job in [
        corrected.job(),
        JpegDecoderConfig::new().job().with_crop_hint(0, 0, 3, 3),
    ] {
        let storage = budget();
        let error = match job.dyn_controlled_decoder(&[], storage.clone()) {
            Ok(_) => panic!("unqualified route"),
            Err(e) => e,
        };
        assert_eq!(
            zencodec::find_cause::<ControlError>(&*error),
            Some(&ControlError::NotControlled)
        );
        assert_eq!(storage.reserved_bytes(), 0);
    }
}

#[test]
fn unqualified_native_float_rows_refuse_before_effects() {
    // Float strip IDCT and the whole-image float IDCT are different algorithms.
    // Until qualified, neither descriptor equality nor a callback may bless it.
    for progressive in [false, true] {
        let bytes = base(progressive);
        for descriptor in [
            PixelDescriptor::RGBF32_LINEAR,
            PixelDescriptor::RGBAF32_LINEAR,
        ] {
            let storage = budget();
            let native = JpegDecoderConfig::new()
                .job()
                .dyn_controlled_decoder(&[descriptor], storage.clone())
                .unwrap();
            let mut seen = false;
            let mut resolver = |q: Question<'_>| -> Result<Decision, At<Infallible>> {
                if let Question::Boundary { facts, choices } = q
                    && facts.state == InterpretationState::Unavailable
                {
                    seen = true;
                    assert!(choices.is_empty());
                    assert_eq!(
                        facts.unavailable,
                        Some(InterpretationUnavailable::UnqualifiedOutput)
                    );
                }
                Ok(Decision::UsePreset)
            };
            let mut session = PolicyDecoder::new(native, &mut resolver);
            let mut sink = Sink::default();
            assert!(drive_from_slice(&mut session, &bytes, &mut sink, None, 17).is_err());
            assert_eq!(sink.effects, 0);
            assert_eq!(storage.reserved_bytes(), 0);
            drop(session);
            assert!(seen);
        }
    }
}

#[test]
fn ordinary_float_rgb_strips_compact_rgba_and_keep_linear_labels() {
    for progressive in [false, true] {
        let bytes = base(progressive);
        let mut rgb = JpegDecoderConfig::new()
            .job()
            .streaming_decoder(Cow::Borrowed(&bytes), &[PixelDescriptor::RGBF32_LINEAR])
            .unwrap();
        let mut rgba = JpegDecoderConfig::new()
            .job()
            .streaming_decoder(Cow::Borrowed(&bytes), &[PixelDescriptor::RGBAF32_LINEAR])
            .unwrap();
        while let Some((y, rgb)) = rgb.next_batch().unwrap() {
            let (ry, rgba) = rgba.next_batch().unwrap().unwrap();
            assert_eq!(y, ry);
            assert_eq!(rgb.descriptor(), PixelDescriptor::RGBF32_LINEAR);
            assert_eq!(rgba.descriptor(), PixelDescriptor::RGBAF32_LINEAR);
            assert_eq!(rgb.rows(), rgba.rows());
            for row in 0..rgb.rows() {
                for (rgb, rgba) in rgb
                    .row(row)
                    .as_chunks::<12>()
                    .0
                    .iter()
                    .zip(rgba.row(row).as_chunks::<16>().0.iter())
                {
                    assert_eq!(rgb, &rgba[..12]);
                    assert_eq!(&rgba[12..], &1.0f32.to_ne_bytes());
                }
            }
        }
        assert!(rgba.next_batch().unwrap().is_none());
    }
}

#[test]
fn grayscale_byte_rows_match_the_native_decoder() {
    let stride = WIDTH as usize + 17;
    let raw: Vec<u8> = (0..stride * HEIGHT as usize)
        .map(|i| (i ^ (i / stride * 13)) as u8)
        .collect();
    for progressive in [false, true] {
        let bytes = JpegEncoderConfig::new()
            .with_progressive(progressive)
            .encode(
                PixelSlice::new(&raw, WIDTH, HEIGHT, stride, PixelDescriptor::GRAY8_SRGB).unwrap(),
            )
            .unwrap();
        let mut oracle = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes.data()));
        let oracle_pixels = oracle.decode().expect("independent grayscale decoder");
        assert_eq!(oracle_pixels.len(), WIDTH as usize * HEIGHT as usize);
        let direct = JpegDecoderConfig::new().decode(bytes.data()).unwrap();
        assert_eq!(direct.pixels().descriptor(), PixelDescriptor::GRAY8_SRGB);
        let native = JpegDecoderConfig::new()
            .job()
            .dyn_controlled_decoder(&[], budget())
            .unwrap();
        let mut session = PolicyDecoder::new(native, PolicySet::new());
        let mut sink = CollectSink::new();
        let result = drive_from_slice(&mut session, bytes.data(), &mut sink, None, 7);
        assert!(
            result.is_ok(),
            "progressive={progressive}, result={result:?}, markers={:?}",
            zenjpeg::container::iter(bytes.data())
                .filter(|s| matches!(
                    s.kind,
                    zenjpeg::container::MarkerKind::Sof(_) | zenjpeg::container::MarkerKind::Dri
                ))
                .map(|s| (s.kind, s.payload))
                .collect::<Vec<_>>()
        );
        let actual = sink.as_pixel_slice().unwrap();
        assert_eq!(actual.descriptor(), direct.pixels().descriptor());
        for y in 0..HEIGHT {
            assert_eq!(actual.row(y), direct.pixels().row(y));
        }
    }
}

#[test]
fn limits_and_job_cancellation_precede_pixel_effects() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use zencodec::{ResourceLimits, StopToken};
    struct Cancel(Arc<AtomicBool>);
    impl zencodec::enough::Stop for Cancel {
        fn check(&self) -> Result<(), zencodec::enough::StopReason> {
            if self.0.load(Ordering::SeqCst) {
                Err(zencodec::enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    let bytes = base(true);
    for limits in [
        ResourceLimits::none().with_max_width(WIDTH - 1),
        ResourceLimits::none().with_max_height(HEIGHT - 1),
        ResourceLimits::none().with_max_input_bytes(bytes.len() as u64 - 1),
        ResourceLimits::none().with_max_output(u64::from(WIDTH) * u64::from(HEIGHT) * 3 - 1),
    ] {
        let storage = budget();
        let native = JpegDecoderConfig::new()
            .job()
            .with_limits(limits)
            .dyn_controlled_decoder(&[], storage.clone())
            .unwrap();
        let mut session = PolicyDecoder::new(native, PolicySet::new());
        let mut sink = Sink::default();
        let error = drive_from_slice(&mut session, &bytes, &mut sink, None, 7).unwrap_err();
        assert!(zencodec::find_cause::<zencodec::CodecError>(&error).is_some());
        assert_eq!(sink.effects, 0);
        assert_eq!(storage.reserved_bytes(), 0);
    }
    let cancelled = Arc::new(AtomicBool::new(false));
    let storage = budget();
    let native = JpegDecoderConfig::new()
        .job()
        .with_stop(StopToken::new(Cancel(cancelled.clone())))
        .dyn_controlled_decoder(&[], storage.clone())
        .unwrap();
    let mut session = PolicyDecoder::new(
        native,
        |q: Question<'_>| -> Result<Decision, At<Infallible>> {
            if let Question::Boundary { facts, .. } = q
                && facts.state == InterpretationState::Final
            {
                cancelled.store(true, Ordering::SeqCst);
            }
            Ok(Decision::UsePreset)
        },
    );
    let mut sink = Sink::default();
    assert!(drive_from_slice(&mut session, &bytes, &mut sink, None, 7).is_err());
    assert_eq!(sink.effects, 0);
    assert_eq!(storage.reserved_bytes(), 0);
}
