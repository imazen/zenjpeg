#[path = "../fuzz/component_checks.rs"]
mod component_checks;
use imgref::ImgRefMut;
use sha2::{Digest, Sha256};
use zenjpeg::decoder::{Decoder, OutputTarget};

#[test]
fn oracle_seeds_and_all_truncated_prefixes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("support/native_manifest.json")).unwrap();
    let mut count = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jpg") {
            continue;
        }
        count += 1;
        let data = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_str().unwrap();
        let expected = manifest
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["name"] == name)
            .unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&data)),
            expected["jpeg_sha256"].as_str().unwrap()
        );
        let mut reader = Decoder::new()
            .output_target(OutputTarget::Native16)
            .scanline_reader(&data)
            .unwrap();
        let actual = match reader.component_ids().len() {
            1 => samples::<1>(&mut reader),
            2 => samples::<2>(&mut reader),
            3 => samples::<3>(&mut reader),
            4 => samples::<4>(&mut reader),
            5 => samples::<5>(&mut reader),
            10 => samples::<10>(&mut reader),
            _ => panic!("unexpected seed component count"),
        };
        assert_eq!(
            format!("{:x}", Sha256::digest(&actual)),
            expected["samples_sha256"].as_str().unwrap(),
            "{}",
            path.display()
        );
        for end in 0..=data.len() {
            component_checks::check(&data[..end]);
        }
    }
    assert_eq!(count, 44);
}

fn samples<const N: usize>(reader: &mut zenjpeg::decoder::ScanlineReader<'_>) -> Vec<u8> {
    let w = reader.width() as usize;
    let mut row = vec![[0; N]; w];
    let mut out = Vec::new();
    while !reader.is_finished() {
        assert_eq!(
            reader
                .read_rows_components16(ImgRefMut::new(&mut row, w, 1))
                .unwrap(),
            1
        );
        for pixel in &row {
            for sample in pixel {
                out.extend_from_slice(&sample.to_be_bytes());
            }
        }
    }
    out
}

use std::sync::atomic::{AtomicU64, Ordering};
struct TestPulse {
    checks: AtomicU64,
    rows: AtomicU64,
    cancel_at: u64,
}
impl TestPulse {
    fn new(cancel_at: u64) -> Self {
        Self {
            checks: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            cancel_at,
        }
    }
}
impl enough::Stop for TestPulse {
    fn check(&self) -> Result<(), enough::StopReason> {
        if self.checks.fetch_add(1, Ordering::Relaxed) >= self.cancel_at {
            Err(enough::StopReason::TimedOut)
        } else {
            Ok(())
        }
    }
}
impl how_far::Report for TestPulse {
    fn advance(&self, n: u64) {
        self.rows.fetch_add(n, Ordering::Relaxed);
    }
}
impl how_far::Pulse for TestPulse {
    fn split(
        &self,
        _: how_far::Execution,
        _: &[how_far::PhaseSpec<'_>],
    ) -> Result<Vec<how_far::Child<'_>>, how_far::PlanError> {
        Err(how_far::PlanError::Unsupported)
    }
}
fn read_with_pulse(data: &[u8], pulse: &dyn how_far::Pulse) -> zenjpeg::decoder::Result<()> {
    let mut reader = Decoder::new()
        .output_target(OutputTarget::Native16)
        .scanline_reader_with_pulse(data, pulse)?;
    fn read<const N: usize>(
        reader: &mut zenjpeg::decoder::ScanlineReader<'_>,
    ) -> zenjpeg::decoder::Result<()> {
        let w = reader.width() as usize;
        let mut row = vec![[0; N]; w];
        while !reader.is_finished() {
            assert_eq!(
                reader.read_rows_components16(ImgRefMut::new(&mut row, w, 1))?,
                1
            );
        }
        Ok(())
    }
    match reader.component_ids().len() {
        1 => read::<1>(&mut reader),
        2 => read::<2>(&mut reader),
        3 => read::<3>(&mut reader),
        4 => read::<4>(&mut reader),
        5 => read::<5>(&mut reader),
        10 => read::<10>(&mut reader),
        _ => panic!("unexpected component count"),
    }
}

#[test]
fn every_native_checkpoint_preserves_timeout_and_reports_completed_rows() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jpg") {
            continue;
        }
        let data = std::fs::read(&path).unwrap();
        for collected in [false, true] {
            let config = Decoder::new().output_target(OutputTarget::Native16);
            let n = config.scanline_reader(&data).unwrap().component_ids().len();
            if collected && (n == 2 || n > 4) {
                continue;
            } // No implicit color conversion.
            let run = |pulse: &TestPulse| {
                if collected {
                    config.decode_with_pulse(&data, pulse).map(|_| ())
                } else {
                    read_with_pulse(&data, pulse)
                }
            };
            let baseline = TestPulse::new(u64::MAX);
            run(&baseline).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let height = config.scanline_reader(&data).unwrap().height();
            assert_eq!(baseline.rows.load(Ordering::Relaxed), u64::from(height));
            for cancel_at in 0..baseline.checks.load(Ordering::Relaxed) {
                let pulse = TestPulse::new(cancel_at);
                let error = run(&pulse).unwrap_err();
                assert!(
                    matches!(
                        error.kind(),
                        zenjpeg::decoder::ErrorKind::Cancelled(enough::StopReason::TimedOut)
                    ),
                    "{} checkpoint {cancel_at}: {error}",
                    path.display()
                );
                assert!(pulse.rows.load(Ordering::Relaxed) <= u64::from(height));
            }
        }
    }
}
