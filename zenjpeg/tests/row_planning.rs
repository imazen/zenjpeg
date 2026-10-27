//! Effective-job planning executes the adapter's own preparation: the plan
//! must equal what the native row encoder reports at startup, for the
//! colour the rows declare, without starting native pixel storage.
#![cfg(feature = "zencodec")]

use std::io::Cursor;
use std::sync::Arc;

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig, OutputTiming};
use zencodec::output_patch::OutputAccess;
use zenjpeg::JpegEncoderConfig;
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig as NativeConfig, XybSubsampling};
use zenpixels::{Cicp, ColorContext, ColorPrimaries, PixelDescriptor, PixelSlice};
use zenpixels_convert::icc_profiles::DISPLAY_P3_V4;

fn configurations() -> Vec<JpegEncoderConfig> {
    let mut configs = vec![JpegEncoderConfig::new()];
    for subsampling in [
        ChromaSubsampling::None,
        ChromaSubsampling::HalfHorizontal,
        ChromaSubsampling::Quarter,
    ] {
        for progressive in [false, true] {
            for optimized in [false, true] {
                configs.push(
                    JpegEncoderConfig::ycbcr(83.0, subsampling)
                        .with_progressive(progressive)
                        .with_huffman(optimized),
                );
            }
        }
    }
    configs.push(JpegEncoderConfig::new().with_prefer_streaming(true));
    configs.push(
        JpegEncoderConfig::new()
            .with_progressive(true)
            .with_prefer_streaming(true),
    );
    for native in [
        NativeConfig::rgb(83.0),
        NativeConfig::xyb(83.0, XybSubsampling::BQuarter),
    ] {
        let mut config = JpegEncoderConfig::new();
        *config.inner_mut() = native;
        configs.push(config.with_prefer_streaming(true));
    }
    configs
}

/// Declarations whose interpretation differs: default, Display P3 by
/// descriptor, by context ICC and by context code points, linear input,
/// and a contradiction.
fn declarations() -> Vec<(&'static str, PixelDescriptor, Option<ColorContext>)> {
    let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    vec![
        ("rgb8", PixelDescriptor::RGB8_SRGB, None),
        ("gray8", PixelDescriptor::GRAY8_SRGB, None),
        ("p3 descriptor", p3, None),
        ("p3 icc", p3, Some(ColorContext::from_icc(DISPLAY_P3_V4))),
        (
            "p3 cicp",
            p3,
            Some(ColorContext::from_cicp(Cicp::DISPLAY_P3)),
        ),
        ("linear", PixelDescriptor::RGBF32_LINEAR, None),
        (
            "contradiction",
            p3,
            Some(ColorContext::from_cicp(Cicp::SRGB)),
        ),
    ]
}

/// The plan equals native startup for every mode, geometry, declaration and
/// destination access, or both refuse.
#[test]
fn effective_plan_matches_native_startup_across_modes_geometry_and_colour() {
    let mut cells = 0;
    let mut refusals = 0;
    for config in configurations() {
        for width in [1, 17] {
            for height in [1, 15, 16, 31, 32, 33, 65] {
                for (what, descriptor, context) in declarations() {
                    for access in [OutputAccess::Forward, OutputAccess::SeekBack] {
                        let job = config.clone().job();
                        let planned =
                            job.plan_rows(width, height, descriptor, context.as_ref(), access);
                        let actual = job.encoder().unwrap().begin_declared(
                            width,
                            height,
                            descriptor,
                            context.as_ref(),
                            access,
                        );
                        match (planned, actual) {
                            (Ok(planned), Ok(actual)) => assert_eq!(
                                planned, actual,
                                "{width}x{height} {what} {access:?} {config:?}"
                            ),
                            (Err(_), Err(_)) => refusals += 1,
                            (planned, actual) => panic!(
                                "planning/startup disagreement for {what}: {planned:?} / \
                                 {actual:?}; {config:?}"
                            ),
                        }
                        cells += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cells, 17 * 2 * 7 * 7 * 2);
    // Every contradiction and every XYB wide-gamut declaration refuses.
    assert!(refusals >= 17 * 2 * 7 * 2, "{refusals}");
}

/// Without a declared context, `begin_output` is `begin_declared(None)`.
#[test]
fn begin_output_matches_the_plan_without_a_context() {
    for config in configurations() {
        for height in [1, 16, 33] {
            for access in [OutputAccess::Forward, OutputAccess::SeekBack] {
                let job = config.clone().job();
                let planned = job
                    .plan_rows(17, height, PixelDescriptor::RGB8_SRGB, None, access)
                    .unwrap();
                let actual = job
                    .encoder()
                    .unwrap()
                    .begin_output(17, height, PixelDescriptor::RGB8_SRGB, access)
                    .unwrap();
                assert_eq!(planned, actual, "{config:?}");
            }
        }
    }
}

fn strided_rgb(width: usize, height: usize) -> (Vec<u8>, usize) {
    let stride = width * 3 + 15 * 3;
    let mut data = vec![0xa5; stride * height];
    for y in 0..height {
        for x in 0..width * 3 {
            data[y * stride + x] = ((x * 31 + y * 17) % 256) as u8;
        }
    }
    (data, stride)
}

/// Planning is repeatable and leaves the planned job's encode byte-identical.
#[test]
fn planning_is_repeatable_and_does_not_change_strided_encode_bytes() {
    let (data, stride) = strided_rgb(17, 33);
    let pixels = || PixelSlice::new(&data, 17, 33, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    for config in configurations() {
        let expected = config
            .clone()
            .job()
            .encoder()
            .unwrap()
            .encode(pixels())
            .unwrap();
        let job = config.clone().job();
        let plan = || {
            job.plan_rows(
                17,
                33,
                PixelDescriptor::RGB8_SRGB,
                None,
                OutputAccess::Forward,
            )
        };
        assert_eq!(plan().unwrap(), plan().unwrap());
        let actual = job.encoder().unwrap().encode(pixels()).unwrap();
        assert_eq!(expected.data(), actual.data());
    }
}

/// A declared colour is in the header before any row: the streamed file
/// carries the context's profile, the rows encode exactly as one-shot does,
/// and the timing behaves as planned.
#[test]
fn declared_colour_reaches_the_header_and_streams_on_the_planned_schedule() {
    let (data, stride) = strided_rgb(17, 65);
    let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(ColorPrimaries::DisplayP3);
    let context = Arc::new(ColorContext::from_icc(DISPLAY_P3_V4));
    let pixels = || {
        PixelSlice::new(&data, 17, 65, stride, p3)
            .unwrap()
            .with_color_context(context.clone())
    };
    let configs = configurations();
    // XYB (last) refuses a Display P3 declaration.
    for config in &configs[..configs.len() - 1] {
        let job = config.clone().job();
        let planned = job
            .plan_rows(17, 65, p3, Some(&context), OutputAccess::Forward)
            .unwrap();
        let one_shot = job.clone().encoder().unwrap().encode(pixels()).unwrap();
        let mut rows = job.encoder().unwrap();
        let timing = rows
            .begin_declared(17, 65, p3, Some(&context), OutputAccess::Forward)
            .unwrap();
        assert_eq!(timing, planned);
        let mut streamed = Vec::new();
        if timing == OutputTiming::AT_FINISH {
            assert!(rows.pending_output().is_empty());
        } else {
            assert!(!rows.pending_output().is_empty(), "header pending at begin");
        }
        let header = rows.pending_output().len();
        for y in 0..65 {
            rows.push_rows(pixels().sub_rows(y, 1)).unwrap();
            if timing == OutputTiming::AT_FINISH {
                assert!(rows.pending_output().is_empty(), "{config:?}");
            } else if timing.lag_rows() == Some(y) {
                // Row 0 is done once `lag` more rows have arrived.
                let produced = streamed.len() + rows.pending_output().len();
                assert!(
                    produced > header,
                    "row 0 bytes after {y} more rows; {config:?}"
                );
            }
            rows.take_output(&mut streamed);
        }
        streamed.extend_from_slice(rows.finish().unwrap().data());
        assert_eq!(streamed, one_shot.data(), "{config:?}");
        let mut oracle = jpeg_decoder::Decoder::new(Cursor::new(&streamed));
        oracle.decode().expect("independent decode");
        assert_eq!(oracle.icc_profile().as_deref(), Some(DISPLAY_P3_V4));
    }
}

/// Rows after a legacy `begin` must mean what its header says.
#[test]
fn rows_contradicting_the_header_refuse_before_encoding() {
    let (data, stride) = strided_rgb(17, 16);
    let rows = |context: ColorContext| {
        PixelSlice::new(&data, 17, 16, stride, PixelDescriptor::RGB8_SRGB)
            .unwrap()
            .with_color_context(Arc::new(context))
    };
    let mut encoder = JpegEncoderConfig::new().job().encoder().unwrap();
    encoder.begin(17, 16, PixelDescriptor::RGB8_SRGB).unwrap();
    assert!(
        encoder
            .push_rows(rows(ColorContext::from_icc(DISPLAY_P3_V4)))
            .is_err()
    );
    let mut encoder = JpegEncoderConfig::new().job().encoder().unwrap();
    encoder.begin(17, 16, PixelDescriptor::RGB8_SRGB).unwrap();
    encoder
        .push_rows(rows(ColorContext::from_cicp(Cicp::SRGB)))
        .unwrap();
    assert!(!encoder.finish().unwrap().data().is_empty());
}

#[test]
fn invalid_input_limits_canvas_and_metadata_refuse_during_planning() {
    let config = JpegEncoderConfig::new();
    let plan = |job: zenjpeg::JpegEncodeJob, width, height| {
        job.plan_rows(
            width,
            height,
            PixelDescriptor::RGB8_SRGB,
            None,
            OutputAccess::Forward,
        )
    };
    for (width, height) in [(0, 32), (32, 0), (u32::MAX, 2)] {
        assert!(plan(config.clone().job(), width, height).is_err());
    }
    for job in [
        config.clone().job().with_canvas_size(16, 16),
        config
            .clone()
            .job()
            .with_limits(zencodec::ResourceLimits::none().with_max_pixels(1)),
        config
            .clone()
            .job()
            .with_limits(zencodec::ResourceLimits::none().with_max_memory(1)),
        config.clone().job().with_metadata_policy(
            zencodec::Metadata::default().with_exif(vec![0; 70_000]),
            zencodec::MetadataPolicy::PreserveExact,
        ),
    ] {
        assert!(plan(job, 32, 32).is_err());
    }
}

/// Cancellation refuses planning and keeps its original reason.
#[test]
fn cancelled_planning_refuses_with_its_reason() {
    #[derive(Clone)]
    struct Expired;
    impl zencodec::enough::Stop for Expired {
        fn check(&self) -> Result<(), zencodec::enough::StopReason> {
            Err(zencodec::enough::StopReason::TimedOut)
        }
    }
    let job = JpegEncoderConfig::new()
        .job()
        .with_stop(zencodec::StopToken::new(Expired));
    let error = job
        .plan_rows(
            32,
            32,
            PixelDescriptor::RGB8_SRGB,
            None,
            OutputAccess::Forward,
        )
        .unwrap_err();
    assert!(format!("{error:?}").contains("TimedOut"), "{error:?}");
}

/// An MPF index (gain map, depth map) records the primary image's length, so
/// the file is written at finish: plan and startup both report `AT_FINISH`
/// for a configuration that otherwise streams, and the rows path writes the
/// one-shot file. The generic job does not take gain maps at all.
#[test]
fn mpf_secondary_images_plan_at_finish_and_generic_gain_maps_refuse() {
    use zenjpeg::encode::EncoderSegments;
    use zenjpeg::encoder::HuffmanStrategy;
    let (data, stride) = strided_rgb(17, 65);
    let pixels = || PixelSlice::new(&data, 17, 65, stride, PixelDescriptor::RGB8_SRGB).unwrap();
    let secondary = JpegEncoderConfig::new()
        .encode(PixelSlice::new(&data, 8, 8, stride, PixelDescriptor::RGB8_SRGB).unwrap())
        .unwrap()
        .into_vec();
    let native = NativeConfig::ycbcr(83.0, ChromaSubsampling::Quarter)
        .progressive(false)
        .huffman(HuffmanStrategy::Fixed);
    let streaming = || {
        let mut config = JpegEncoderConfig::new();
        *config.inner_mut() = native.clone();
        config
    };
    let with_mpf = || {
        let mut config = JpegEncoderConfig::new();
        *config.inner_mut() = native
            .clone()
            .with_segments(EncoderSegments::new().add_gainmap(secondary.clone()));
        config
    };
    let plan = |config: JpegEncoderConfig| {
        config
            .job()
            .plan_rows(
                17,
                65,
                PixelDescriptor::RGB8_SRGB,
                None,
                OutputAccess::SeekBack,
            )
            .unwrap()
    };
    assert!(
        plan(streaming()).streams(),
        "control: the base configuration streams"
    );
    assert_eq!(plan(with_mpf()), OutputTiming::AT_FINISH);
    let one_shot = with_mpf()
        .job()
        .encoder()
        .unwrap()
        .encode(pixels())
        .unwrap();
    let mut rows = with_mpf().job().encoder().unwrap();
    let timing = rows
        .begin_output(17, 65, PixelDescriptor::RGB8_SRGB, OutputAccess::SeekBack)
        .unwrap();
    assert_eq!(timing, OutputTiming::AT_FINISH);
    for y in (0..65).step_by(9) {
        rows.push_rows(pixels().sub_rows(y, 9.min(65 - y))).unwrap();
        assert!(rows.pending_output().is_empty());
    }
    let finished = rows.finish().unwrap();
    assert_eq!(finished.data(), one_shot.data());
    assert!(
        finished
            .data()
            .windows(secondary.len())
            .any(|w| w == secondary.as_slice()),
        "the secondary image is carried"
    );
    assert!(!JpegEncoderConfig::capabilities().gain_map());
}
