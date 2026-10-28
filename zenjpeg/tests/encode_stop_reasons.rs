//! A stop keeps its reason through every encode push path: a deadline
//! surfaces as `TimedOut`, never as a generic cancellation. zencodecs'
//! `pulse_retention` found `BytesEncoder::push` reporting `Cancelled`.
#![cfg(feature = "zencodec")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use zencodec::encode::{EncodeJob, Encoder, EncoderConfig};
use zencodec::enough::{Stop, StopReason};
use zencodec::how_far::{Child, Execution, PhaseSpec, PlanError, Pulse, Report};
use zencodec::{At, CodecError, ErrorCategory, SharedPulse};
use zenjpeg::JpegEncoderConfig;
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig as NativeConfig, ErrorKind, PixelLayout, YCbCrPlanes,
};
use zenpixels::{PixelDescriptor, PixelSlice};

/// A deadline that has passed once `polls` reaches `fail_at`, and stays passed.
#[derive(Clone)]
struct Deadline {
    polls: Arc<AtomicUsize>,
    fail_at: usize,
}
impl Stop for Deadline {
    fn check(&self) -> Result<(), StopReason> {
        if self.polls.fetch_add(1, Ordering::Relaxed) >= self.fail_at {
            Err(StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}
impl Report for Deadline {
    fn advance(&self, _completed: u64) {}
}
impl Pulse for Deadline {
    fn split(
        &self,
        _execution: Execution,
        _parts: &[PhaseSpec<'_>],
    ) -> Result<Vec<Child<'_>>, PlanError> {
        Err(PlanError::Unsupported)
    }
}

fn deadline(polls: &Arc<AtomicUsize>, fail_at: usize) -> Deadline {
    polls.store(0, Ordering::Relaxed);
    Deadline {
        polls: polls.clone(),
        fail_at,
    }
}

/// Every poll the operation makes can stop it, with the deadline's reason.
fn each_poll<T, E: core::fmt::Debug>(
    what: &str,
    mut run: impl FnMut(Deadline) -> Result<T, E>,
    reason: impl Fn(&E) -> Option<StopReason>,
) {
    let polls = Arc::new(AtomicUsize::new(0));
    run(deadline(&polls, usize::MAX)).unwrap_or_else(|e| panic!("{what}: {e:?}"));
    let total = polls.load(Ordering::Relaxed);
    assert!(total > 0, "{what}: the stop was never polled");
    for fail_at in 0..total {
        match run(deadline(&polls, fail_at)) {
            Ok(_) => panic!("{what}: completed past an expired deadline at poll {fail_at}/{total}"),
            Err(e) => assert_eq!(
                reason(&e),
                Some(StopReason::TimedOut),
                "{what} poll {fail_at}/{total}: {e:?}"
            ),
        }
    }
}

fn shared_reason(error: &At<CodecError>) -> Option<StopReason> {
    match error.error().category() {
        ErrorCategory::Stopped(reason) => Some(reason),
        _ => None,
    }
}

fn native_reason(error: &zenjpeg::encoder::Error) -> Option<StopReason> {
    match error.kind() {
        ErrorKind::Cancelled(reason) => Some(*reason),
        _ => None,
    }
}

const W: u32 = 37;
const H: u32 = 70;

fn rgb(stride: usize) -> Vec<u8> {
    let mut data = vec![0x5a; stride * H as usize];
    for y in 0..H as usize {
        for x in 0..W as usize * 3 {
            data[y * stride + x] = ((x * 29 + y * 11) % 251) as u8;
        }
    }
    data
}

#[test]
fn zencodec_one_shot_and_row_encodes_keep_the_deadline_reason() {
    let data = rgb(W as usize * 3);
    let pixels =
        || PixelSlice::new(&data, W, H, W as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    for (name, config) in [
        (
            "baseline",
            JpegEncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter),
        ),
        (
            "progressive",
            JpegEncoderConfig::ycbcr(85.0, ChromaSubsampling::None).with_progressive(true),
        ),
        (
            "streaming",
            JpegEncoderConfig::new().with_prefer_streaming(true),
        ),
    ] {
        each_poll(
            &format!("{name} one-shot"),
            |stop| {
                config
                    .clone()
                    .job()
                    .with_pulse(SharedPulse::new(stop))
                    .encoder()?
                    .encode(pixels())
            },
            shared_reason,
        );
        each_poll(
            &format!("{name} rows"),
            |stop| {
                let mut encoder = config
                    .clone()
                    .job()
                    .with_pulse(SharedPulse::new(stop))
                    .encoder()?;
                encoder.begin(W, H, PixelDescriptor::RGB8_SRGB)?;
                for y in (0..H).step_by(16) {
                    let rows = (H - y).min(16);
                    let start = y as usize * W as usize * 3;
                    let end = start + rows as usize * W as usize * 3;
                    let strip = PixelSlice::new(
                        &data[start..end],
                        W,
                        rows,
                        W as usize * 3,
                        PixelDescriptor::RGB8_SRGB,
                    )
                    .unwrap();
                    encoder.push_rows(strip)?;
                }
                encoder.finish()
            },
            shared_reason,
        );
    }
}

#[test]
fn native_byte_and_planar_pushes_keep_the_deadline_reason() {
    let config = NativeConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    for stride in [W as usize * 3, W as usize * 3 + 5] {
        let data = rgb(stride);
        each_poll(
            &format!("BytesEncoder::push stride {stride}"),
            |stop| {
                let mut encoder = config.encode_from_bytes(W, H, PixelLayout::Rgb8Srgb)?;
                encoder.push(&data, H as usize, stride, stop)
            },
            native_reason,
        );
    }
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y: Vec<f32> = (0..w * h).map(|i| (i % 251) as f32).collect();
    let full_chroma: Vec<f32> = (0..w * h).map(|i| (i % 97) as f32 - 48.0).collect();
    let half_chroma: Vec<f32> = (0..cw * ch).map(|i| (i % 89) as f32 - 44.0).collect();
    each_poll(
        "YCbCrPlanarEncoder::push",
        |stop| {
            let mut encoder = config.encode_from_ycbcr_planar(W, H)?;
            let planes = YCbCrPlanes {
                y: &y,
                y_stride: w,
                cb: &full_chroma,
                cb_stride: w,
                cr: &full_chroma,
                cr_stride: w,
            };
            encoder.push(&planes, h, stop)
        },
        native_reason,
    );
    each_poll(
        "YCbCrPlanarEncoder::push_subsampled",
        |stop| {
            let mut encoder = config.encode_from_ycbcr_planar(W, H)?;
            let planes = YCbCrPlanes {
                y: &y,
                y_stride: w,
                cb: &half_chroma,
                cb_stride: cw,
                cr: &half_chroma,
                cr_stride: cw,
            };
            encoder.push_subsampled(&planes, h, stop)
        },
        native_reason,
    );
}
