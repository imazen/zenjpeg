#[path = "../fuzz/component_checks.rs"]
mod component_checks;
use imgref::ImgRefMut;
use sha2::{Digest, Sha256};
use zenjpeg::decoder::{Decoder, OutputTarget};

#[test]
fn native_fixture_raster_hashes() {
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
        if matches!(expected["args"][2].as_str().unwrap(), "1" | "3" | "4") {
            let format = match expected["converted_format"].as_str().unwrap() {
                "rgb" => zenjpeg::decoder::PixelFormat::Rgb,
                "gray" => zenjpeg::decoder::PixelFormat::Gray,
                "cmyk" => zenjpeg::decoder::PixelFormat::Cmyk,
                other => panic!("unexpected reference output format: {other}"),
            };
            let decoded = Decoder::new()
                .output_target(OutputTarget::Native16)
                .output_format(format)
                .decode(&data, enough::Unstoppable)
                .unwrap();
            let mut hash = Sha256::new();
            for sample in decoded.pixels_u16().unwrap() {
                hash.update(sample.to_be_bytes());
            }
            assert_eq!(
                format!("{:x}", hash.finalize()),
                expected["converted_samples_sha256"].as_str().unwrap(),
                "{name}"
            );
        }
        let config = Decoder::new().output_target(OutputTarget::Native16);
        for owned in [false, true] {
            let mut reader = if owned {
                config.scanline_reader_cow(std::borrow::Cow::Owned(data.clone()))
            } else {
                config.scanline_reader(&data)
            }
            .unwrap();
            let argument = |index: usize| {
                expected["args"][index]
                    .as_str()
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
            };
            assert_eq!(u32::from(reader.sample_precision()), argument(1), "{name}");
            assert_eq!(reader.width(), argument(8), "{name}");
            assert_eq!(reader.height(), argument(9), "{name}");
            assert_eq!(reader.component_ids().len(), argument(2) as usize, "{name}");
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
        }
    }
    assert_eq!(count, 47);
}

#[test]
fn native_fixture_truncated_prefixes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jpg") {
            continue;
        }
        let data = std::fs::read(path).unwrap();
        for end in 0..=data.len() {
            component_checks::check(&data[..end]);
        }
    }
}

fn samples<const N: usize>(reader: &mut zenjpeg::decoder::ScanlineReader<'_>) -> Vec<u8> {
    let w = reader.width() as usize;
    let mut row = vec![[0xbeef; N]; w + 3];
    let mut out = Vec::new();
    while !reader.is_finished() {
        assert_eq!(
            reader
                .read_rows_components16(ImgRefMut::new_stride(&mut row, w, 1, w + 3))
                .unwrap(),
            1
        );
        assert!(row[w..].iter().all(|pixel| *pixel == [0xbeef; N]));
        for pixel in &row[..w] {
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
    for name in [
        "05-p8-c2-sof30-mode0-s1",
        "15-p8-c4-sof30-mode0-s1",
        "16-p12-c4-sof30-mode0-s1",
    ] {
        let data = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("fuzz/seeds/native/{name}.jpg")),
        )
        .unwrap();
        for (index, config) in [
            Decoder::new()
                .output_target(OutputTarget::Native16)
                .deblock(DeblockMode::Boundary4Tap),
            Decoder::new().output_target(OutputTarget::SrgbF32Precise),
            Decoder::new()
                .output_target(OutputTarget::Native16)
                .correct_color(Some(TargetColorSpace::Srgb)),
            Decoder::new()
                .output_target(OutputTarget::SrgbF32)
                .correct_color(Some(TargetColorSpace::Srgb)),
        ]
        .into_iter()
        .enumerate()
        {
            // Ordinary 8-bit CMYK float output keeps its existing buffered route.
            if name.starts_with("15-") && matches!(index, 1 | 3) {
                continue;
            }
            assert!(config.scanline_reader(&data).is_err(), "{name}");
        }
    }
}

fn seed(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("fuzz/seeds/native/{name}.jpg")),
    )
    .unwrap()
}
fn app(data: &[u8], marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![255, 216, 255, marker];
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&data[2..]);
    out
}

#[test]
fn native_color_markers_and_planar_guards() {
    use zenjpeg::decoder::{Cmyk16, CropRegion, PixelFormat};
    let rgb = app(
        &seed("11-p12-c3-sof30-mode0-s1"),
        0xee,
        b"Adobe\0d\0\0\0\0\0",
    );
    let ycbcr = app(
        &seed("32-p12-c3-sof31-mode0-s1"),
        0xe0,
        b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0",
    );
    for data in [&rgb, &ycbcr] {
        let cfg = Decoder::new().output_target(OutputTarget::Native16);
        let mut raw = cfg.scanline_reader(data).unwrap();
        let mut channels = [[0u16; 3]; 9];
        raw.read_rows_components16(ImgRefMut::new(&mut channels, 9, 1))
            .unwrap();
        for method in 0..3 {
            let mut reader = cfg.scanline_reader(data).unwrap();
            let mut y = [0.0; 9];
            let mut cb = [0.0; 9];
            let mut cr = [0.0; 9];
            let rows = match method {
                0 => reader.read_rows_ycbcr_f32(&mut y, &mut cb, &mut cr, 9, 1),
                1 => reader.read_rows_gray_f32(ImgRefMut::new(&mut y, 9, 1)),
                _ => reader.read_rows_gray_linear_f32(ImgRefMut::new(&mut y, 9, 1)),
            }
            .unwrap();
            assert_eq!(rows, 1);
            assert!(y.iter().all(|v| v.is_finite()));
            if core::ptr::eq(data, &ycbcr) && method == 0 {
                for x in 0..9 {
                    assert_eq!(y[x], channels[x][0] as f32 / 4095.0);
                    assert_eq!(cb[x], (channels[x][1] as f32 - 2048.0) / 4095.0);
                    assert_eq!(cr[x], (channels[x][2] as f32 - 2048.0) / 4095.0);
                }
            }
        }
    }
    let gray = seed("01-p12-c1-sof30-mode0-s1");
    let cmyk = seed("15-p8-c4-sof30-mode0-s1");
    let mut reader = Decoder::new().scanline_reader(&gray).unwrap();
    let mut ink = [Cmyk16::default(); 9];
    assert!(
        reader
            .read_rows_cmyk16(ImgRefMut::new(&mut ink, 9, 1))
            .is_err()
    );
    let mut reader = Decoder::new().scanline_reader(&cmyk).unwrap();
    assert!(
        reader
            .read_rows_cmyk16(ImgRefMut::new(&mut ink, 9, 1))
            .is_err()
    );
    let mut reader = Decoder::new()
        .output_target(OutputTarget::Native16)
        .scanline_reader(&cmyk)
        .unwrap();
    assert!(
        reader
            .read_rows_cmyk16(ImgRefMut::new_stride(&mut ink, 8, 1, 9))
            .is_err()
    );

    // Converted pixels cannot masquerade as source components or native planes.
    for data in [
        gray.clone(),
        seed("30-p2-c1-sof31-mode0-s1"),
        seed("05-p8-c2-sof30-mode0-s1"),
    ] {
        for transformed in [false, true] {
            let mut config = Decoder::new().output_target(OutputTarget::Native16);
            if transformed {
                config = config.transform(zenjpeg::lossless::LosslessTransform::Transpose);
            }
            // Unknown two-channel data has no implicit pixel-domain transform.
            if transformed && data == seed("05-p8-c2-sof30-mode0-s1") {
                continue;
            }
            let mut reader = config.scanline_reader(&data).unwrap();
            let mut y = [0i16; 9 * 16];
            let mut cb = y;
            let mut cr = y;
            if transformed || data != seed("01-p12-c1-sof30-mode0-s1") {
                assert!(
                    reader
                        .read_rows_ycbcr_native_i16(&mut y, 9, &mut cb, &mut cr, 9, 1)
                        .is_err()
                );
            }
            if transformed {
                let mut row = [[0; 1]; 9];
                assert!(
                    reader
                        .read_rows_components16(ImgRefMut::new(&mut row, 9, 1))
                        .is_err()
                );
            }
        }
    }
    // Both non-native buffered output variants crop and report successfully.
    for target in [OutputTarget::Srgb8, OutputTarget::SrgbF32] {
        let pulse = TestPulse::new(u64::MAX);
        let result = Decoder::new()
            .output_target(target)
            .output_format(PixelFormat::Gray)
            .transform(zenjpeg::lossless::LosslessTransform::Transpose)
            .crop(CropRegion::pixels(1, 1, 3, 3))
            .decode_with_pulse(&rgb, &pulse)
            .unwrap();
        assert_eq!(result.dimensions(), (3, 3));
        assert_eq!(pulse.rows.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn planar_pulse_counts_partial_mcu_rows() {
    let data = seed("00-p8-c1-sof30-mode0-s1");
    let pulse = TestPulse::new(u64::MAX);
    let mut reader = Decoder::new()
        .scanline_reader_with_pulse(&data, &pulse)
        .unwrap();
    let mut y = [0i16; 9 * 16];
    let mut cb = [];
    let mut cr = [];
    assert_eq!(
        reader
            .read_rows_ycbcr_native_i16(&mut y, 9, &mut cb, &mut cr, 0, 3)
            .unwrap(),
        (9, 0)
    );
    assert_eq!(pulse.rows.load(Ordering::Relaxed), 9);
    let stopped = TestPulse::new(0);
    assert!(
        Decoder::new()
            .scanline_reader_with_pulse(&data, &stopped)
            .is_err()
    );
}

#[cfg(feature = "inventory")]
#[test]
fn permissive_inventory_retains_huffman_table_mapping() {
    use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
    let data = seed("11-p12-c3-sof30-mode0-s1");
    let mut config = zenjpeg::JpegDecoderConfig::new();
    *config.inner_mut() = Decoder::new().strictness(zenjpeg::decoder::Strictness::Permissive);
    config.job().inventory(&data).unwrap().validate().unwrap();
}

#[test]
fn eight_bit_buffered_pulse_and_owned_transform() {
    use zenjpeg::encoder::{EncoderConfig, PixelLayout};
    let pixels: Vec<u8> = (0..81).map(|i| (i * 73) as u8).collect();
    let data = EncoderConfig::grayscale(88)
        .progressive(true)
        .encode_bytes(&pixels, 9, 9, PixelLayout::Gray8Srgb)
        .unwrap();
    let pulse = TestPulse::new(u64::MAX);
    let result = Decoder::new().decode_with_pulse(&data, &pulse).unwrap();
    assert_eq!(result.dimensions(), (9, 9));
    assert_eq!(pulse.rows.load(Ordering::Relaxed), 9);
    let pulse = TestPulse::new(u64::MAX);
    let mut reader = Decoder::new()
        .scanline_reader_with_pulse(&data, &pulse)
        .unwrap();
    let mut row = [0u8; 81 * 3];
    assert_eq!(
        reader
            .read_rows_rgb8(ImgRefMut::new(&mut row, 27, 9))
            .unwrap(),
        9
    );
    assert_eq!(pulse.rows.load(Ordering::Relaxed), 9);
    let mut native = [[0u16; 1]; 9];
    assert!(
        reader
            .read_rows_components16(ImgRefMut::new(&mut native, 9, 1))
            .is_err()
    );
    let mut y = [0i16; 144];
    assert!(
        reader
            .read_rows_ycbcr_native_i16(&mut y, 9, &mut [], &mut [], 0, 1)
            .is_err()
    );
    drop(reader);
    let mut owned = Decoder::new()
        .transform(zenjpeg::lossless::LosslessTransform::Transpose)
        .scanline_reader_cow(std::borrow::Cow::Owned(data))
        .unwrap();
    assert_eq!(
        owned
            .read_rows_rgb8(ImgRefMut::new(&mut row, 27, 9))
            .unwrap(),
        9
    );
}

#[cfg(feature = "parallel")]
#[test]
fn wave_pulse_reports_pixel_and_planar_rows() {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let (w, h) = (512usize, 513usize);
    let pixels: Vec<u8> = (0..w * h * 3)
        .map(|i| (i.wrapping_mul(73) ^ (i >> 7)) as u8)
        .collect();
    let data = EncoderConfig::ycbcr(40, ChromaSubsampling::Quarter)
        .progressive(false)
        .restart_mcu_rows(1)
        .encode_bytes(&pixels, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    let config = Decoder::new()
        .chroma_upsampling(zenjpeg::decoder::ChromaUpsampling::NearestNeighbor)
        .num_threads(2);
    let pulse = TestPulse::new(u64::MAX);
    let mut reader = config.scanline_reader_with_pulse(&data, &pulse).unwrap();
    // Readers without progress observers use the same parallel row paths.
    let mut ordinary = config.scanline_reader(&data).unwrap();
    let mut ordinary_rgb = vec![0u8; w * 3];
    assert_eq!(
        ordinary
            .read_rows_rgb8(ImgRefMut::new(&mut ordinary_rgb, w * 3, 1))
            .unwrap(),
        1
    );
    let mut ordinary = config.scanline_reader(&data).unwrap();
    let mut y = vec![0i16; w * 16];
    let mut cb = vec![0i16; w / 2 * 8];
    let mut cr = cb.clone();
    assert_eq!(
        ordinary
            .read_rows_ycbcr_native_i16(&mut y, w, &mut cb, &mut cr, w / 2, 1)
            .unwrap(),
        (16, 8)
    );
    let mut sequential = config
        .clone()
        .num_threads(1)
        .scanline_reader(&data)
        .unwrap();
    assert_eq!(
        sequential
            .read_rows_rgb8(ImgRefMut::new(&mut ordinary_rgb, w * 3, 1))
            .unwrap(),
        1
    );
    let construction_checks = pulse.checks.load(Ordering::Relaxed);
    for planar in [false, true] {
        let stopped = TestPulse::new(construction_checks + 2);
        let mut cancelled = config.scanline_reader_with_pulse(&data, &stopped).unwrap();
        let error = if planar {
            let mut y = vec![0i16; w * 16];
            let mut cb = vec![0i16; w / 2 * 8];
            let mut cr = cb.clone();
            cancelled
                .read_rows_ycbcr_native_i16(&mut y, w, &mut cb, &mut cr, w / 2, 1)
                .unwrap_err()
        } else {
            let mut rgb = vec![0u8; w * 3];
            cancelled
                .read_rows_rgb8(ImgRefMut::new(&mut rgb, w * 3, 1))
                .unwrap_err()
        };
        assert!(matches!(
            error.kind(),
            zenjpeg::decoder::ErrorKind::Cancelled(enough::StopReason::TimedOut)
        ));
        assert_eq!(stopped.rows.load(Ordering::Relaxed), 0);
    }
    let mut components = vec![[0u16; 3]; w];
    assert!(
        reader
            .read_rows_components16(ImgRefMut::new(&mut components, w, 1))
            .is_err()
    );
    let mut row = vec![0u8; w * 3];
    for _ in 0..h {
        assert_eq!(
            reader
                .read_rows_rgb8(ImgRefMut::new(&mut row, w * 3, 1))
                .unwrap(),
            1
        );
    }
    assert_eq!(pulse.rows.load(Ordering::Relaxed), h as u64);
    let pulse = TestPulse::new(u64::MAX);
    let mut reader = config.scanline_reader_with_pulse(&data, &pulse).unwrap();
    let mut y = vec![0i16; w * 16];
    let mut cb = vec![0i16; w / 2 * 8];
    let mut cr = cb.clone();
    let mut rows = 0;
    while rows < h {
        let (n, _) = reader
            .read_rows_ycbcr_native_i16(&mut y, w, &mut cb, &mut cr, w / 2, 1)
            .unwrap();
        assert!(n > 0);
        rows += n;
    }
    assert_eq!(rows, h);
    assert_eq!(pulse.rows.load(Ordering::Relaxed), h as u64);
}

#[test]
fn grayscale_sampling_factors_preserve_pixels() {
    for name in ["00-p8-c1-sof30-mode0-s1", "01-p12-c1-sof30-mode0-s1"] {
        let original = seed(name);
        let config = Decoder::new().output_target(OutputTarget::Native16);
        let expected = config.decode(&original, enough::Unstoppable).unwrap();
        for factor in [0x21, 0x12, 0x44] {
            let mut data = original.clone();
            let sof = data
                .windows(2)
                .position(|p| p == [0xff, 0xc0] || p == [0xff, 0xc1])
                .unwrap();
            data[sof + 11] = factor;
            let result = config.decode(&data, enough::Unstoppable).unwrap();
            assert_eq!(
                result.pixels_u16(),
                expected.pixels_u16(),
                "{name} {factor:x}"
            );
            let borrowed = config.scanline_reader(&data).unwrap();
            let owned = config
                .scanline_reader_cow(std::borrow::Cow::Owned(data.clone()))
                .unwrap();
            assert_eq!(
                (borrowed.width(), borrowed.height()),
                (owned.width(), owned.height())
            );
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn collected_parallel_pulse_preserves_worker_timeouts() {
    use zenjpeg::decoder::{ChromaUpsampling, ErrorKind};
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let (w, h) = (512usize, 513usize);
    let pixels: Vec<u8> = (0..w * h * 3)
        .map(|i| (i.wrapping_mul(73) ^ (i >> 7)) as u8)
        .collect();
    for (sub, upsampling) in [
        (ChromaSubsampling::None, ChromaUpsampling::Triangle),
        (
            ChromaSubsampling::Quarter,
            ChromaUpsampling::NearestNeighbor,
        ),
        (ChromaSubsampling::Quarter, ChromaUpsampling::Triangle),
        (
            ChromaSubsampling::HalfHorizontal,
            ChromaUpsampling::Triangle,
        ),
    ] {
        let data = EncoderConfig::ycbcr(40, sub)
            .progressive(false)
            .restart_mcu_rows(1)
            .encode_bytes(&pixels, w as u32, h as u32, PixelLayout::Rgb8Srgb)
            .unwrap();
        let config = Decoder::new().chroma_upsampling(upsampling).num_threads(2);
        let baseline = TestPulse::new(u64::MAX);
        let expected = config.decode_with_pulse(&data, &baseline).unwrap();
        let sequential = config
            .clone()
            .num_threads(1)
            .decode(&data, enough::Unstoppable)
            .unwrap();
        assert_eq!(expected.pixels_u8(), sequential.pixels_u8());
        assert_eq!(baseline.rows.load(Ordering::Relaxed), h as u64);
        // Each budget is well inside a frame's MCU work, after marker parsing.
        for cancel_at in [32, 128, 512] {
            let pulse = TestPulse::new(cancel_at);
            let error = config.decode_with_pulse(&data, &pulse).unwrap_err();
            assert!(matches!(
                error.kind(),
                ErrorKind::Cancelled(enough::StopReason::TimedOut)
            ));
            assert_eq!(pulse.rows.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn decode_into_agrees_with_collected_native_frames() {
    use zenjpeg::decoder::PixelFormat;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    for entry in std::fs::read_dir(root).unwrap() {
        let data = std::fs::read(entry.unwrap().path()).unwrap();
        for (format, channels) in [
            (PixelFormat::Rgb, 3),
            (PixelFormat::Gray, 1),
            (PixelFormat::Rgba, 4),
        ] {
            let config = Decoder::new().output_format(format);
            let expected = config.decode(&data, enough::Unstoppable);
            let info = Decoder::new()
                .output_target(OutputTarget::Native16)
                .scanline_reader(&data)
                .unwrap();
            let mut actual = vec![0; info.width() as usize * info.height() as usize * channels];
            let result = config.decode_into(&data, format, &mut actual, enough::Unstoppable);
            assert_eq!(result.is_ok(), expected.is_ok());
            if let Ok(expected) = expected {
                assert_eq!(actual, expected.pixels_u8().unwrap());
            }
        }
    }
}

#[test]
fn uncommon_ycbcr_refuses_strip_planes() {
    let data = seed("45-p12-c3-sof30-mode0-s3");
    let mut reader = Decoder::new()
        .output_target(OutputTarget::Native16)
        .scanline_reader(&data)
        .unwrap();
    let mut y = [0i16; 9 * 16];
    let mut cb = y;
    let mut cr = y;
    assert!(
        reader
            .read_rows_ycbcr_native_i16(&mut y, 9, &mut cb, &mut cr, 9, 1)
            .is_err()
    );
}

#[test]
fn crop_skipping_preserves_every_timeout_and_pixels() {
    use zenjpeg::decoder::{CropRegion, ErrorKind};
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let (w, h) = (32usize, 80usize);
    let pixels: Vec<u8> = (0..w * h * 3)
        .map(|i| (i.wrapping_mul(73) ^ (i >> 7)) as u8)
        .collect();
    let data = EncoderConfig::ycbcr(80, ChromaSubsampling::Quarter)
        .progressive(false)
        .encode_bytes(&pixels, w as u32, h as u32, PixelLayout::Rgb8Srgb)
        .unwrap();
    let config = Decoder::new()
        .num_threads(1)
        .crop(CropRegion::pixels(0, 64, 32, 16));
    let run = |pulse: &TestPulse| -> zenjpeg::decoder::Result<Vec<u8>> {
        let mut reader = config.scanline_reader_with_pulse(&data, pulse)?;
        let mut pixels = vec![0; 32 * 16 * 3];
        assert_eq!(
            reader.read_rows_rgb8(ImgRefMut::new(&mut pixels, 32 * 3, 16))?,
            16
        );
        Ok(pixels)
    };
    let baseline = TestPulse::new(u64::MAX);
    let actual = run(&baseline).unwrap();
    let full = Decoder::new()
        .num_threads(1)
        .decode(&data, enough::Unstoppable)
        .unwrap();
    assert_eq!(actual, full.pixels_u8().unwrap()[64 * w * 3..]);
    assert_eq!(baseline.rows.load(Ordering::Relaxed), 16);
    for cancel_at in 0..baseline.checks.load(Ordering::Relaxed) {
        let error = run(&TestPulse::new(cancel_at)).unwrap_err();
        assert!(matches!(
            error.kind(),
            ErrorKind::Cancelled(enough::StopReason::TimedOut)
        ));
    }
}
