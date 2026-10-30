//! Progress represents completed scan batches and output rows, not elapsed time.
use enough::{Stop, StopReason, Unstoppable};
use how_far::{Child, ChildPulse, Execution, Outcome, PhaseSpec, PlanError, Pulse, Report};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zenjpeg::decoder::{Decoder, ErrorKind};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

#[derive(Default)]
struct Trace {
    events: Mutex<Vec<(usize, u64)>>,
    outcomes: Mutex<Vec<(usize, Outcome)>>,
    checks: AtomicUsize,
    fire_at_check: AtomicUsize,
    fire_at_report: AtomicUsize,
    reports: AtomicUsize,
    pending: AtomicUsize,
    cancelled: AtomicUsize,
    unsupported: AtomicUsize,
    finish_stop_phase: AtomicUsize,
    finish_error_phase: AtomicUsize,
    issues: Mutex<Vec<PlanError>>,
}
struct TestPulse {
    trace: Arc<Trace>,
    phase: usize,
}
impl Stop for TestPulse {
    fn check(&self) -> Result<(), StopReason> {
        let n = self.trace.checks.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.trace.fire_at_check.load(Ordering::SeqCst)
            || self.trace.pending.swap(0, Ordering::SeqCst) != 0
        {
            Err(if self.trace.cancelled.load(Ordering::SeqCst) == 1 {
                StopReason::Cancelled
            } else {
                StopReason::TimedOut
            })
        } else {
            Ok(())
        }
    }
}
impl Report for TestPulse {
    fn advance(&self, n: u64) {
        self.trace.events.lock().unwrap().push((self.phase, n));
        let report = self.trace.reports.fetch_add(1, Ordering::SeqCst) + 1;
        if report == self.trace.fire_at_report.load(Ordering::SeqCst) {
            self.trace.pending.store(1, Ordering::SeqCst);
        }
    }
}
impl Pulse for TestPulse {
    fn record_issue(&self, error: PlanError) {
        self.trace.issues.lock().unwrap().push(error);
    }
    fn split(
        &self,
        execution: Execution,
        parts: &[PhaseSpec<'_>],
    ) -> Result<Vec<Child<'_>>, PlanError> {
        if self.trace.unsupported.load(Ordering::SeqCst) != 0 {
            return Err(PlanError::Unsupported);
        }
        assert_eq!(execution, Execution::Sequence);
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "scan work");
        assert_eq!(parts[1].name, "output rows");
        assert_eq!(parts[0].total, how_far::Total::Unknown);
        Ok((1..=2)
            .map(|phase| {
                Child::new(TestPulse {
                    trace: self.trace.clone(),
                    phase,
                })
            })
            .collect())
    }
}
impl ChildPulse for TestPulse {
    fn finish(self: Box<Self>, outcome: Outcome) -> Result<(), PlanError> {
        self.trace
            .outcomes
            .lock()
            .unwrap()
            .push((self.phase, outcome));
        if self.trace.finish_stop_phase.load(Ordering::SeqCst) == self.phase {
            self.trace.pending.store(1, Ordering::SeqCst);
        }
        if self.trace.finish_error_phase.load(Ordering::SeqCst) == self.phase {
            return Err(PlanError::Finished);
        }
        Ok(())
    }
}
fn fixture(progressive: bool) -> Vec<u8> {
    let (width, height) = (97, 65);
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| (i * 73 + i / 37) as u8)
        .collect();
    EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
        .progressive(progressive)
        .encode_bytes(&pixels, width as u32, height as u32, PixelLayout::Rgb8Srgb)
        .unwrap()
}
fn token() -> (Arc<Trace>, TestPulse) {
    let trace = Arc::new(Trace::default());
    (trace.clone(), TestPulse { trace, phase: 0 })
}
fn assert_interrupted_outcomes(trace: &Trace) {
    let outcomes = trace.outcomes.lock().unwrap();
    assert_eq!(outcomes.len(), 2, "{outcomes:?}");
    assert_eq!(outcomes[0].0, 1);
    assert_eq!(outcomes[1].0, 2);
    assert!(
        matches!(
            (outcomes[0].1, outcomes[1].1),
            (Outcome::NotRun, Outcome::NotRun)
                | (Outcome::Cancelled, Outcome::NotRun)
                | (Outcome::Succeeded, Outcome::NotRun)
                | (Outcome::Succeeded, Outcome::Cancelled)
                | (Outcome::Succeeded, Outcome::Succeeded)
        ),
        "{outcomes:?}"
    );
}

#[test]
fn progress_is_intermediate_and_pixels_match() {
    for progressive in [false, true] {
        let data = fixture(progressive);
        let expected = Decoder::new().decode(&data, Unstoppable).unwrap();
        let (trace, pulse) = token();
        let actual = Decoder::new()
            .decode_with_pulse(&data, Some(&pulse))
            .unwrap();
        assert_eq!(expected.pixels_u8(), actual.pixels_u8());
        let events = trace.events.lock().unwrap();
        assert!(
            events.iter().filter(|(phase, _)| *phase == 1).count() > 1,
            "{events:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|(phase, _)| *phase == 2)
                .map(|(_, n)| n)
                .sum::<u64>(),
            65
        );
        assert!(
            events.iter().position(|(p, _)| *p == 1) < events.iter().position(|(p, _)| *p == 2)
        );
        assert_eq!(
            *trace.outcomes.lock().unwrap(),
            [(1, Outcome::Succeeded), (2, Outcome::Succeeded)]
        );
    }
}
#[test]
fn every_report_can_stop_without_losing_reason() {
    for progressive in [false, true] {
        let data = fixture(progressive);
        let (trace, pulse) = token();
        Decoder::new()
            .decode_with_pulse(&data, Some(&pulse))
            .unwrap();
        let reports = trace.reports.load(Ordering::SeqCst);
        for reason in [StopReason::Cancelled, StopReason::TimedOut] {
            for report in 1..=reports {
                let (trace, pulse) = token();
                trace.cancelled.store(
                    usize::from(reason == StopReason::Cancelled),
                    Ordering::SeqCst,
                );
                trace.fire_at_report.store(report, Ordering::SeqCst);
                let error = Decoder::new()
                    .decode_with_pulse(&data, Some(&pulse))
                    .unwrap_err();
                assert!(
                    matches!(error.kind(), ErrorKind::Cancelled(actual) if *actual == reason),
                    "{report}: {error:?}"
                );
                assert_interrupted_outcomes(&trace);
            }
        }
    }
}
#[test]
fn malformed_input_retains_native_error() {
    let (trace, pulse) = token();
    trace.fire_at_report.store(1, Ordering::SeqCst);
    let error = Decoder::new()
        .decode_with_pulse(b"not a jpeg", Some(&pulse))
        .unwrap_err();
    assert!(!matches!(error.kind(), ErrorKind::Cancelled(_)));
    assert_eq!(trace.reports.load(Ordering::SeqCst), 0);
}

#[test]
fn unsupported_planning_still_reports_output_and_stops() {
    let data = fixture(false);
    let (trace, pulse) = token();
    trace.unsupported.store(1, Ordering::SeqCst);
    Decoder::new()
        .decode_with_pulse(&data, Some(&pulse))
        .unwrap();
    assert_eq!(*trace.events.lock().unwrap(), [(0, 65)]);
    let (trace, pulse) = token();
    trace.unsupported.store(1, Ordering::SeqCst);
    trace.fire_at_report.store(1, Ordering::SeqCst);
    let error = Decoder::new()
        .decode_with_pulse(&data, Some(&pulse))
        .unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::Cancelled(StopReason::TimedOut)
    ));
}

#[test]
fn native_collected_last_row_report_can_stop() {
    let data = fixture(false);
    let config = Decoder::new().output_target(zenjpeg::decoder::OutputTarget::Native16);
    let (trace, pulse) = token();
    let expected = config.decode(&data, Unstoppable).unwrap();
    let actual = config.decode_with_pulse(&data, Some(&pulse)).unwrap();
    assert_eq!(expected.pixels_u16(), actual.pixels_u16());
    let reports = trace.reports.load(Ordering::SeqCst);
    assert!(reports > 0);
    let (trace, pulse) = token();
    trace.fire_at_report.store(reports, Ordering::SeqCst);
    let error = config.decode_with_pulse(&data, Some(&pulse)).unwrap_err();
    assert!(matches!(
        error.kind(),
        ErrorKind::Cancelled(StopReason::TimedOut)
    ));
}

#[test]
fn finish_errors_are_recorded_and_finish_can_request_stop() {
    for phase in [1, 2] {
        let data = fixture(false);
        let (trace, pulse) = token();
        trace.finish_error_phase.store(phase, Ordering::SeqCst);
        Decoder::new()
            .decode_with_pulse(&data, Some(&pulse))
            .unwrap();
        assert_eq!(*trace.issues.lock().unwrap(), [PlanError::Finished]);

        let (trace, pulse) = token();
        trace.finish_stop_phase.store(phase, Ordering::SeqCst);
        let error = Decoder::new()
            .decode_with_pulse(&data, Some(&pulse))
            .unwrap_err();
        assert!(matches!(
            error.kind(),
            ErrorKind::Cancelled(StopReason::TimedOut)
        ));
        assert_eq!(trace.pending.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn native_failure_is_not_replaced_by_stop_from_finish() {
    let (trace, pulse) = token();
    trace.finish_stop_phase.store(1, Ordering::SeqCst);
    let expected = Decoder::new()
        .decode(b"not a jpeg", Unstoppable)
        .unwrap_err();
    let error = Decoder::new()
        .decode_with_pulse(b"not a jpeg", Some(&pulse))
        .unwrap_err();
    assert_eq!(
        core::mem::discriminant(error.kind()),
        core::mem::discriminant(expected.kind())
    );
    assert_eq!(
        trace.pending.load(Ordering::SeqCst),
        1,
        "finish must actually request a stop"
    );
    assert_eq!(
        *trace.outcomes.lock().unwrap(),
        [(1, Outcome::NotRun), (2, Outcome::NotRun)]
    );
}

#[test]
fn shared_checkpoint_conformance_preserves_jpeg_pixels_and_phases() {
    use how_far_along::Status;
    use zencodec_testkit::{
        Failure,
        pulse::{Case, Limits, check_checkpoints},
    };
    for progressive in [false, true] {
        let data = fixture(progressive);
        let expected = Decoder::new().decode(&data, Unstoppable).unwrap();
        let pixel_match = core::cell::Cell::new(false);
        check_checkpoints(
            Limits {
                checkpoints: 4096,
                trace_capacity: 4096,
            },
            |pulse| {
                let result = Decoder::new().decode_with_pulse(&data, Some(pulse));
                if let Ok(actual) = &result {
                    pixel_match.set(
                        expected.dimensions() == actual.dimensions()
                            && expected.pixels_u8() == actual.pixels_u8(),
                    );
                }
                result.map(|_| ())
            },
            |error| match error.kind() {
                ErrorKind::Cancelled(reason) => Some(*reason),
                _ => None,
            },
            |case, trace| {
                let fail = |detail: String| Failure {
                    check: "jpeg_pulse_phases",
                    detail,
                };
                let tree = trace
                    .progress
                    .as_ref()
                    .ok_or_else(|| fail(format!("{case:?}: missing progress tree")))?;
                if tree.children.len() != 2 {
                    return Err(fail(format!(
                        "{case:?}: expected scan and output children: {:?}",
                        tree.children
                    )));
                }
                let scan = &tree.children[0];
                let rows = &tree.children[1];
                if scan.name != "scan work"
                    || scan.units != "batches"
                    || rows.name != "output rows"
                    || rows.units != "rows"
                {
                    return Err(fail(format!(
                        "{case:?}: unexpected JPEG phases: {:?}",
                        tree.children
                    )));
                }
                if matches!(case, Case::Uninterrupted) {
                    if !pixel_match.get()
                        || scan.completed <= 1
                        || rows.completed != 65
                        || scan.status != Status::Finished(Outcome::Succeeded)
                        || rows.status != Status::Finished(Outcome::Succeeded)
                    {
                        return Err(fail(format!(
                            "{case:?}: JPEG pixels or completion differ: {:?}",
                            tree.children
                        )));
                    }
                } else if !matches!(
                    (scan.status, rows.status),
                    (
                        Status::Finished(Outcome::NotRun),
                        Status::Finished(Outcome::NotRun)
                    ) | (
                        Status::Finished(Outcome::Cancelled),
                        Status::Finished(Outcome::NotRun)
                    ) | (
                        Status::Finished(Outcome::Succeeded),
                        Status::Finished(Outcome::NotRun)
                    ) | (
                        Status::Finished(Outcome::Succeeded),
                        Status::Finished(Outcome::Cancelled)
                    ) | (
                        Status::Finished(Outcome::Succeeded),
                        Status::Finished(Outcome::Succeeded)
                    )
                ) {
                    return Err(fail(format!(
                        "{case:?}: invalid JPEG phase outcomes: {:?}",
                        tree.children
                    )));
                }
                Ok(())
            },
        )
        .unwrap();
    }
}

#[test]
fn native_and_orientation_use_unpartitioned_pulse() {
    let data = fixture(false);
    for config in [
        Decoder::new().output_target(zenjpeg::decoder::OutputTarget::Native16),
        Decoder::new().transform(zenjpeg::lossless::LosslessTransform::Transpose),
    ] {
        let (trace, pulse) = token();
        let expected = config.decode(&data, Unstoppable).unwrap();
        let actual = config.decode_with_pulse(&data, Some(&pulse)).unwrap();
        assert_eq!(expected.dimensions(), actual.dimensions());
        assert_eq!(expected.pixels_u8(), actual.pixels_u8());
        assert_eq!(expected.pixels_u16(), actual.pixels_u16());
        assert!(trace.outcomes.lock().unwrap().is_empty());
        assert!(
            trace
                .events
                .lock()
                .unwrap()
                .iter()
                .all(|(phase, _)| *phase == 0)
        );
    }
}
