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

#[test]
#[ignore = "explicit corpus replay; set ZENJPEG_FUZZ_CORPUS"]
fn native_fuzz_corpus_replay() {
    let root = std::env::var_os("ZENJPEG_FUZZ_CORPUS").expect("set ZENJPEG_FUZZ_CORPUS");
    let mut count = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let data = std::fs::read(entry.unwrap().path()).unwrap();
        component_checks::check(&data);
        #[cfg(feature = "inventory")]
        {
            use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
            if let Ok(inventory) = zenjpeg::JpegDecoderConfig::new().job().inventory(&data) {
                inventory.validate().unwrap();
            }
        }
        count += 1;
    }
    assert!(count > 0);
}

#[test]
fn native_output_formats_and_precision_options() {
    use zenjpeg::decoder::{CropRegion, GainMapHandling, PixelFormat};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    for entry in std::fs::read_dir(root).unwrap() {
        let data = std::fs::read(entry.unwrap().path()).unwrap();
        let reader = Decoder::new().scanline_reader(&data).unwrap();
        let n = reader.component_ids().len();
        if n == 2 || n > 4 || reader.sample_precision() == 8 {
            continue;
        }
        for target in [
            OutputTarget::Srgb8,
            OutputTarget::Native16,
            OutputTarget::SrgbF32,
            OutputTarget::LinearF32,
        ] {
            for format in [
                PixelFormat::Rgb,
                PixelFormat::Bgr,
                PixelFormat::Rgba,
                PixelFormat::Bgra,
                PixelFormat::Gray,
                PixelFormat::GrayF32,
                PixelFormat::RgbaF32,
                PixelFormat::Cmyk,
            ] {
                if format == PixelFormat::Cmyk && n != 4 {
                    continue;
                }
                let config = Decoder::new()
                    .output_target(target)
                    .output_format(format)
                    .gain_map(GainMapHandling::PreserveRaw);
                let result = config.decode(&data, enough::Unstoppable);
                // Float pixel layouts require a float target; CMYK has no implicit EOTF.
                if (matches!(format, PixelFormat::GrayF32 | PixelFormat::RgbaF32)
                    && !target.is_f32())
                    || (format == PixelFormat::Cmyk && target.is_linear())
                {
                    assert!(result.is_err());
                } else {
                    let result = result.unwrap_or_else(|e| {
                        panic!("n={n}, target={target:?}, format={format:?}: {e}")
                    });
                    assert_eq!(result.height(), reader.height());
                    let _ = result.pixels();
                    let _ = result.into_pixels();
                }
            }
        }
        for limit in [0, u64::MAX] {
            Decoder::new()
                .max_memory(limit)
                .output_target(OutputTarget::Native16)
                .decode(&data, enough::Unstoppable)
                .unwrap();
        }
        if reader.width() > 1 {
            let pulse = TestPulse::new(u64::MAX);
            let config = Decoder::new()
                .output_target(OutputTarget::Native16)
                .transform(zenjpeg::lossless::LosslessTransform::Transpose)
                .crop(CropRegion::pixels(1, 1, 3, 3));
            let result = config.decode_with_pulse(&data, &pulse).unwrap();
            assert_eq!(
                pulse.rows.load(Ordering::Relaxed),
                u64::from(result.height())
            );
            let mut transformed = config.scanline_reader_with_pulse(&data, &pulse).unwrap();
            let mut row = vec![rgb::RGB::new(0, 0, 0); 3 * 4];
            assert_eq!(
                transformed
                    .read_rows_rgb16(ImgRefMut::new(&mut row, 3, 4))
                    .unwrap(),
                3
            );
        }
    }
}

#[test]
fn native_row_formats_report_and_check_buffers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    for entry in std::fs::read_dir(root).unwrap() {
        let data = std::fs::read(entry.unwrap().path()).unwrap();
        let cfg = Decoder::new().output_target(OutputTarget::Native16);
        let info = cfg.scanline_reader(&data).unwrap();
        let (w, h, n) = (
            info.width() as usize,
            info.height() as usize,
            info.component_ids().len(),
        );
        if n == 2 || n > 4 {
            continue;
        }
        for method in 0..10 {
            let pulse = TestPulse::new(u64::MAX);
            let mut reader = cfg.scanline_reader_with_pulse(&data, &pulse).unwrap();
            let mut bytes = vec![0; w * (h + 1) * 4];
            let mut floats = vec![0.0; bytes.len()];
            let rows = match method {
                0 => reader.read_rows_rgb8(ImgRefMut::new(&mut bytes, w * 3, h + 1)),
                1 => reader.read_rows_rgba8(ImgRefMut::new(&mut bytes, w * 4, h + 1)),
                2 => reader.read_rows_bgr8(ImgRefMut::new(&mut bytes, w * 3, h + 1)),
                3 => reader.read_rows_gray8(ImgRefMut::new(&mut bytes, w, h + 1)),
                4 => reader.read_rows_rgba_f32(ImgRefMut::new(&mut floats, w * 4, h + 1)),
                5 => reader.read_rows_gray_f32(ImgRefMut::new(&mut floats, w, h + 1)),
                6 => reader.read_rows_gray_linear_f32(ImgRefMut::new(&mut floats, w, h + 1)),
                7 => {
                    let mut cb = floats.clone();
                    let mut cr = floats.clone();
                    reader.read_rows_ycbcr_f32(&mut floats, &mut cb, &mut cr, w, h + 1)
                }
                8 => {
                    let mut row = vec![rgb::RGB::new(0, 0, 0); w * (h + 1)];
                    assert!(
                        reader
                            .read_rows_rgb16(ImgRefMut::new_stride(&mut row, w - 1, 1, w))
                            .is_err()
                    );
                    reader.read_rows_rgb16(ImgRefMut::new(&mut row, w, h + 1))
                }
                _ => {
                    let mut row = vec![0; w * (h + 1)];
                    assert!(
                        reader
                            .read_rows_gray16(ImgRefMut::new_stride(&mut row, w - 1, 1, w))
                            .is_err()
                    );
                    reader.read_rows_gray16(ImgRefMut::new(&mut row, w, h + 1))
                }
            }
            .unwrap();
            assert_eq!(rows, h);
            assert_eq!(pulse.rows.load(Ordering::Relaxed), h as u64);
            assert!(reader.is_finished());
        }
    }
}

#[test]
fn malformed_native_headers_are_bounded() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    for entry in std::fs::read_dir(root).unwrap() {
        let data = std::fs::read(entry.unwrap().path()).unwrap();
        let mut mutations = Vec::new();
        for span in zenjpeg::container::iter(&data) {
            let start = span.offset + 4;
            // Exercise every byte in SOF/SOS and table definition prefixes.
            let end = (span.offset + span.length).min(data.len());
            let marker = data[span.offset + 1];
            if matches!(
                marker,
                0xc0..=0xc3 | 0xc9 | 0xca | 0xda | 0xc4 | 0xdb | 0xcc | 0xdd
            ) {
                for pos in start..end.min(start + 20) {
                    for value in [0, 1, 4, 5, 8, 12, 16, 17, 63, 255] {
                        let mut changed = data.clone();
                        changed[pos] = value;
                        mutations.push(changed);
                    }
                }
            }
        }
        for changed in mutations {
            component_checks::check(&changed);
            #[cfg(feature = "inventory")]
            {
                use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
                if let Ok(inventory) = zenjpeg::JpegDecoderConfig::new().job().inventory(&changed) {
                    inventory.validate().unwrap();
                }
            }
        }
    }
}

#[test]
fn native_component_options_never_silently_drop_processing() {
    use zenjpeg::decoder::{DeblockMode, TargetColorSpace};
    for name in ["05-p8-c2-sof30-mode0-s1", "15-p8-c4-sof30-mode0-s1", "16-p12-c4-sof30-mode0-s1"] {
        let data = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("fuzz/seeds/native/{name}.jpg"))).unwrap();
        for config in [
            Decoder::new().output_target(OutputTarget::Native16).deblock(DeblockMode::Boundary4Tap),
            Decoder::new().output_target(OutputTarget::SrgbF32Precise),
            Decoder::new().output_target(OutputTarget::Native16).correct_color(Some(TargetColorSpace::Srgb)),
            Decoder::new().output_target(OutputTarget::SrgbF32).correct_color(Some(TargetColorSpace::Srgb)),
        ] {
            // Ordinary 8-bit CMYK float output keeps its existing buffered route.
            if name.starts_with("15-") && config.output_target_value() != OutputTarget::Native16 { continue; }
            assert!(config.scanline_reader(&data).is_err(), "{name}");
        }
    }
}
