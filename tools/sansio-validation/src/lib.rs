#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use zencodec::{
        CodecErrorExt,
        decode::{
            DecodeJob, DecodeRowSink, DecoderConfig, Event, IncrementalDecode, Input, SinkError,
            StepError, erase_incremental,
        },
        encode::EncoderConfig,
    };
    use zenpixels::{PixelDescriptor, PixelSliceMut};

    fn config() -> zenjpeg::JpegDecoderConfig {
        let mut cfg = zenjpeg::JpegDecoderConfig::new();
        cfg.inner_mut().chroma_upsampling = zenjpeg::decoder::ChromaUpsampling::NearestNeighbor;
        cfg
    }
    fn fixture(progressive: bool) -> Vec<u8> {
        let image = zencodec_testkit::TestImage::rgb8_gradient(97, 65);
        zenjpeg::JpegEncoderConfig::new()
            .with_progressive(progressive)
            .encode(image.as_slice())
            .unwrap()
            .into_vec()
    }
    #[test]
    fn paired_clone_preserves_samples_and_descriptor() {
        let mut src = zenpixels::PixelBuffer::try_new(3, 2, PixelDescriptor::RGB8_SRGB).unwrap();
        src.as_slice_mut()
            .row_mut(0)
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let cloned = ultrahdr_core::clone_pixel_buffer(&src);
        assert_eq!(cloned.descriptor(), src.descriptor());
        assert_eq!(cloned.as_slice().row(0), src.as_slice().row(0));
        assert_eq!(cloned.as_slice().row(1), src.as_slice().row(1));
    }
    #[test]
    fn independent_libjpeg_turbo_reads_native_fixture() {
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let bytes = fixture(false);
        let mut child = Command::new("djpeg")
            .args(["-nosmooth", "-pnm"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("install libjpeg-turbo-progs");
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let header = b"P6\n97 65\n255\n";
        assert!(output.stdout.starts_with(header));
        let reference = config().decode(&bytes).unwrap();
        let pixels = reference.pixels();
        let packed: Vec<u8> = (0..pixels.rows())
            .flat_map(|y| pixels.row(y).to_vec())
            .collect();
        assert_eq!(&output.stdout[header.len()..], packed);
    }
    #[test]
    fn late_xmp_must_match_one_shot_metadata() {
        let mut bytes = fixture(false);
        assert_eq!(&bytes[bytes.len() - 2..], &[0xff, 0xd9]);
        let xmp = b"<?xpacket begin='\xef\xbb\xbf' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'><dc:title><rdf:Alt><rdf:li xml:lang='x-default'>late</rdf:li></rdf:Alt></dc:title></rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end='w'?>";
        let mut payload = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
        payload.extend_from_slice(xmp);
        let mut marker = vec![0xff, 0xe1];
        marker.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        marker.extend_from_slice(&payload);
        bytes.splice(bytes.len() - 2..bytes.len() - 2, marker);
        // Independent metadata inspection reads our original encoded bytes.
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let mut child = Command::new("exiftool")
            .args(["-scanForXMP", "-Title", "-s", "-s", "-s", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("install exiftool");
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let inspected = child.wait_with_output().unwrap();
        assert!(
            inspected.status.success(),
            "{}",
            String::from_utf8_lossy(&inspected.stderr)
        );
        assert_eq!(inspected.stdout, b"late\n");
        let expected = config().probe_full_metadata(&bytes).unwrap();
        assert_eq!(
            expected.embedded_metadata.xmp.as_deref(),
            Some(xmp.as_slice())
        );
        let mut decoder = config().job().into_incremental_decoder(&[]).unwrap();
        zencodec_io::drive_from_slice(
            &mut decoder,
            &bytes,
            &mut zencodec_io::CollectSink::new(),
            None,
            7,
        )
        .unwrap();
        assert_eq!(
            decoder.info().unwrap().embedded_metadata.xmp,
            expected.embedded_metadata.xmp
        );
    }
    #[test]
    fn every_native_checkpoint_preserves_both_stop_reasons() {
        use zencodec_testkit::pulse::{Case, Limits, check_checkpoints};
        let bytes = fixture(false);
        check_checkpoints(
            Limits {
                checkpoints: 4096,
                trace_capacity: 4096,
            },
            |pulse| {
                let mut decoder = config().job().into_incremental_decoder(&[]).unwrap();
                zencodec_io::drive_from_slice(
                    &mut decoder,
                    &bytes,
                    &mut zencodec_io::CollectSink::new(),
                    Some(pulse),
                    bytes.len(),
                )
                .map(|_| ())
            },
            |error| match error.find_cause::<StepError<zencodec::At<zencodec::CodecError>>>() {
                Some(StepError::Cancelled(reason)) => Some(*reason),
                _ => None,
            },
            |case, trace| {
                if case == Case::Uninterrupted {
                    assert!(trace.spans.iter().map(|s| s.stats.checks).sum::<u64>() > 2);
                }
                Ok(())
            },
        )
        .unwrap();
    }
    #[test]
    fn native_input_matches_one_shot() {
        for progressive in [false, true] {
            let bytes = fixture(progressive);
            zencodec_testkit::check_incremental_decoder(
                || {
                    Ok(erase_incremental(
                        config().job().into_incremental_decoder(&[]).unwrap(),
                    ))
                },
                |input| config().decode(input).map_err(|e| e.to_string()),
                &bytes,
            )
            .unwrap();
        }
    }
    struct OneRow {
        inner: zencodec_io::CollectSink,
        loans: usize,
        signal: Option<Arc<AtomicBool>>,
        failure: bool,
    }
    #[derive(Debug)]
    struct DestinationFailure;
    impl std::fmt::Display for DestinationFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("native destination control")
        }
    }
    impl std::error::Error for DestinationFailure {}
    impl OneRow {
        fn new() -> Self {
            Self {
                inner: zencodec_io::CollectSink::new(),
                loans: 0,
                signal: None,
                failure: false,
            }
        }
    }
    impl DecodeRowSink for OneRow {
        fn begin(&mut self, w: u32, h: u32, d: PixelDescriptor) -> Result<(), SinkError> {
            self.inner.begin(w, h, d)
        }
        fn max_buffer_rows(&self) -> std::num::NonZeroU32 {
            std::num::NonZeroU32::new(1).unwrap()
        }
        fn provide_next_buffer(
            &mut self,
            y: u32,
            h: u32,
            w: u32,
            d: PixelDescriptor,
        ) -> Result<PixelSliceMut<'_>, SinkError> {
            assert_eq!(h, 1);
            self.loans += 1;
            if let Some(signal) = &self.signal {
                signal.store(true, Ordering::SeqCst);
            }
            if self.failure {
                return Err(Box::new(DestinationFailure));
            }
            self.inner.provide_next_buffer(y, h, w, d)
        }
        fn finish(&mut self) -> Result<(), SinkError> {
            self.inner.finish()
        }
    }
    #[test]
    fn typed_native_rows_precede_eof_and_yield_one_loan_per_step() {
        let bytes = fixture(false);
        let expected = config().decode(&bytes).unwrap();
        let mut decoder = config().job().into_incremental_decoder(&[]).unwrap();
        let mut sink = OneRow::new();
        let mut offset = 0;
        let mut early = false;
        let mut done = false;
        for _ in 0..bytes.len() * 4 + 1000 {
            let end = (offset + 7).min(bytes.len());
            let before = sink.loans;
            let p = decoder
                .step(
                    Input::new(offset as u64, &bytes[offset..end])
                        .with_total_len(bytes.len() as u64),
                    &mut sink,
                    None,
                )
                .unwrap();
            assert!(p.consumed <= end - offset);
            offset += p.consumed;
            assert!(sink.loans - before <= 1);
            match p.event {
                Event::Rows { .. } => early |= offset < bytes.len(),
                Event::Done(_) => {
                    done = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(
            done && early,
            "baseline must emit final rows before all input arrives"
        );
        assert_eq!(sink.loans, 65);
        let expected = expected.into_buffer();
        let view = expected.as_slice();
        let mut packed = Vec::new();
        for y in 0..view.rows() {
            packed.extend_from_slice(view.row(y));
        }
        assert_eq!(sink.inner.bytes(), packed);
    }
    struct Signal {
        stopped: Arc<AtomicBool>,
        checks: AtomicUsize,
        at: usize,
        reason: zencodec::enough::StopReason,
    }
    impl zencodec::enough::Stop for Signal {
        fn check(&self) -> Result<(), zencodec::enough::StopReason> {
            if self.stopped.load(Ordering::SeqCst)
                || self.checks.fetch_add(1, Ordering::SeqCst) >= self.at
            {
                Err(self.reason)
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn retained_control_survives_none_and_explicit_noop() {
        for reason in [
            zencodec::enough::StopReason::Cancelled,
            zencodec::enough::StopReason::TimedOut,
        ] {
            for explicit in [false, true] {
                let signal = Signal {
                    stopped: Arc::new(AtomicBool::new(true)),
                    checks: AtomicUsize::new(0),
                    at: usize::MAX,
                    reason,
                };
                let owned = zencodec::SharedPulse::new(zencodec::how_far::StopOnly::new(signal));
                let mut decoder = config()
                    .job()
                    .with_pulse(owned)
                    .into_incremental_decoder(&[])
                    .unwrap();
                let mut sink = OneRow::new();
                let pulse = explicit.then_some(&zencodec::NoPulse as &dyn zencodec::Pulse);
                assert!(
                    matches!(decoder.step(Input::empty(),&mut sink,pulse),Err(StepError::Cancelled(r)) if r==reason)
                );
                assert!(matches!(
                    decoder.step(Input::empty(), &mut sink, None),
                    Err(StepError::Finished)
                ));
                assert_eq!(sink.loans, 0);
            }
        }
    }
    #[test]
    fn callback_stop_is_terminal_and_sink_failure_keeps_native_cause() {
        let bytes = fixture(false);
        for failure in [false, true] {
            let stopped = Arc::new(AtomicBool::new(false));
            let signal = Signal {
                stopped: stopped.clone(),
                checks: AtomicUsize::new(0),
                at: usize::MAX,
                reason: zencodec::enough::StopReason::TimedOut,
            };
            let pulse = zencodec::how_far::StopOnly::new(signal);
            let mut decoder = config().job().into_incremental_decoder(&[]).unwrap();
            let mut sink = OneRow::new();
            sink.signal = Some(stopped);
            sink.failure = failure;
            let result = decoder.step(
                Input::new(0, &bytes).with_total_len(bytes.len() as u64),
                &mut sink,
                Some(&pulse),
            );
            if failure {
                let error = result.unwrap_err();
                assert!(error.find_cause::<DestinationFailure>().is_some());
            } else {
                assert!(matches!(
                    result,
                    Err(StepError::Cancelled(zencodec::enough::StopReason::TimedOut))
                ));
            }
            assert_eq!(sink.loans, 1);
            assert!(matches!(
                decoder.step(Input::empty(), &mut sink, None),
                Err(StepError::Finished)
            ));
            assert_eq!(sink.loans, 1);
        }
    }
    #[test]
    fn typed_and_erased_native_errors_keep_whereat() {
        fn check<D: IncrementalDecode>(mut decoder: D)
        where
            D::Error: 'static,
        {
            let mut sink = OneRow::new();
            let error = decoder
                .step(
                    Input::new(0, b"not a jpeg").with_total_len(10),
                    &mut sink,
                    None,
                )
                .unwrap_err();
            assert!(error.find_cause::<zenjpeg::decoder::ErrorKind>().is_some());
            assert!(
                error
                    .find_cause::<zencodec::At<zencodec::CodecError>>()
                    .is_some()
            );
            assert_eq!(sink.loans, 0);
        }
        check(config().job().into_incremental_decoder(&[]).unwrap());
        // The boxed erased owner is driven separately; no string conversion.
        let mut decoder = erase_incremental(config().job().into_incremental_decoder(&[]).unwrap());
        let mut sink = OneRow::new();
        let error = decoder
            .step(
                Input::new(0, b"not a jpeg").with_total_len(10),
                &mut sink,
                None,
            )
            .unwrap_err();
        assert!(error.find_cause::<zenjpeg::decoder::ErrorKind>().is_some());
        assert!(
            error
                .find_cause::<zencodec::At<zencodec::CodecError>>()
                .is_some()
        );
    }
}

#[cfg(test)]
#[path = "../../../zenjpeg/tests/incremental_consumers.rs"]
mod consumers;
