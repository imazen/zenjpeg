//! Integer equality against a caller-selected libjpeg-turbo oracle. No bundled
//! generated images: every fixture is produced explicitly by the test invocation.
use imgref::ImgRefMut;
use std::{
    borrow::Cow,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
use zenjpeg::decoder::{Decoder, OutputTarget, PixelFormat};
use zenjpeg::encoder::Unstoppable;

fn tool(name: &str, args: &[String], input: &[u8]) -> Vec<u8> {
    let bin = PathBuf::from(
        std::env::var_os("ZENJPEG_TURBO_BIN")
            .expect("set ZENJPEG_TURBO_BIN to pinned libjpeg-turbo tools directory"),
    );
    let mut child = Command::new(bin.join(name))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("launch oracle");
    child.stdin.take().unwrap().write_all(input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn pnm(width: usize, height: usize, channels: usize, bits: u8, kind: usize) -> Vec<u8> {
    let max = (1u32 << bits) - 1;
    let mut bytes = format!(
        "P{}\n{width} {height}\n{max}\n",
        if channels == 1 { 5 } else { 6 }
    )
    .into_bytes();
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let value = match kind {
                    0 => (x * 197 + y * 331 + c * 613 + x * y * 29) as u32 & max,
                    1 => ((x / 8 + y / 8 * 3 + c) * 127) as u32 & max,
                    _ => {
                        if (x + y + c) % 2 == 0 {
                            0
                        } else {
                            max
                        }
                    }
                };
                if bits > 8 {
                    bytes.extend_from_slice(&(value as u16).to_be_bytes());
                } else {
                    bytes.push(value as u8);
                }
            }
        }
    }
    bytes
}

fn samples(pnm: &[u8]) -> (usize, usize, usize, Vec<u16>) {
    let mut pos = 0;
    let mut token = || {
        while pnm[pos].is_ascii_whitespace() {
            pos += 1;
        }
        while pnm[pos] == b'#' {
            while pnm[pos] != b'\n' {
                pos += 1;
            }
            while pnm[pos].is_ascii_whitespace() {
                pos += 1;
            }
        }
        let start = pos;
        while !pnm[pos].is_ascii_whitespace() {
            pos += 1;
        }
        std::str::from_utf8(&pnm[start..pos]).unwrap().to_owned()
    };
    let magic = token();
    let w = token().parse().unwrap();
    let h = token().parse().unwrap();
    let max: u32 = token().parse().unwrap();
    pos += 1;
    let data = if max > 255 {
        pnm[pos..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u16::from_be_bytes([p[0], p[1]]))
            .collect()
    } else {
        pnm[pos..].iter().map(|&v| v as u16).collect()
    };
    (w, h, if magic == "P5" { 1 } else { 3 }, data)
}

fn compare(jpeg: &[u8], bits: u8, label: &str, nearest: bool) {
    let args = if nearest {
        vec!["-nosmooth".to_owned()]
    } else {
        vec![]
    };
    let oracle = tool("djpeg-static", &args, jpeg);
    let (w, h, ch, expected) = samples(&oracle);
    let config = Decoder::new()
        .output_target(OutputTarget::Native16)
        .chroma_upsampling(if nearest {
            zenjpeg::decoder::ChromaUpsampling::NearestNeighbor
        } else {
            zenjpeg::decoder::ChromaUpsampling::Triangle
        });
    let config = if ch == 1 {
        config.output_format(PixelFormat::Gray)
    } else {
        config
    };
    let result = config
        .decode(jpeg, Unstoppable)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(result.sample_precision(), bits, "{label}");
    let assembled = assemble_tiff_jpeg(jpeg);
    let tiled = config.decode(&assembled, Unstoppable).unwrap();
    assert_eq!(
        tiled.pixels_u16().unwrap(),
        expected,
        "TIFF shared tables: {label}"
    );
    let actual = result.pixels_u16().unwrap();
    assert_eq!(actual.len(), expected.len(), "{label}");
    if let Some(i) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: sample {i}, ours {}, turbo {}",
            actual[i], expected[i]
        );
    }
    let max = ((1u32 << bits) - 1) as f32;
    let floats = config
        .clone()
        .output_target(OutputTarget::SrgbF32)
        .decode(jpeg, Unstoppable)
        .unwrap();
    let expected_float: Vec<f32> = expected.iter().map(|&v| v as f32 / max).collect();
    assert_eq!(floats.pixels_f32().unwrap(), expected_float, "f32 {label}");
    let bytes = config
        .clone()
        .output_target(OutputTarget::Srgb8)
        .decode(jpeg, Unstoppable)
        .unwrap();
    let max = (1u32 << bits) - 1;
    let expected_bytes: Vec<u8> = expected
        .iter()
        .map(|&v| ((v as u32 * 255 + max / 2) / max) as u8)
        .collect();
    assert_eq!(bytes.pixels_u8().unwrap(), expected_bytes, "u8 {label}");
    for owned in [false, true] {
        let data = if owned {
            Cow::Owned(jpeg.to_vec())
        } else {
            Cow::Borrowed(jpeg)
        };
        let mut reader = config.scanline_reader_cow(data).unwrap();
        assert_eq!(reader.sample_precision(), bits);
        let stride = w + 5;
        let mut all = Vec::new();
        while !reader.is_finished() {
            if ch == 1 {
                let mut rows = vec![65535; stride * 3];
                let n = reader
                    .read_rows_gray16(ImgRefMut::new_stride(&mut rows, w, 3, stride))
                    .unwrap();
                for row in rows.chunks_exact(stride).take(n) {
                    all.extend_from_slice(&row[..w]);
                    assert!(row[w..].iter().all(|&v| v == 65535));
                }
            } else {
                let sentinel = rgb::RGB::new(65535, 65535, 65535);
                let mut rows = vec![sentinel; stride * 3];
                let n = reader
                    .read_rows_rgb16(ImgRefMut::new_stride(&mut rows, w, 3, stride))
                    .unwrap();
                for row in rows.chunks_exact(stride).take(n) {
                    for p in &row[..w] {
                        all.extend_from_slice(&[p.r, p.g, p.b]);
                    }
                    assert!(row[w..].iter().all(|&v| v == sentinel));
                }
            }
        }
        assert_eq!(all.len(), w * h * ch);
        assert_eq!(all, expected, "{label} owned={owned}");
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo 3.2.0 cjpeg/djpeg"]
fn lossless_matches_turbo() {
    for bits in 2..=16 {
        for channels in [1, 3] {
            for predictor in 1..=7 {
                for point in [0, bits / 2, bits - 1] {
                    let input = pnm(19, 9, channels, bits, predictor as usize % 3);
                    let args = vec![
                        "-precision".into(),
                        bits.to_string(),
                        "-lossless".into(),
                        format!("{predictor},{point}"),
                        "-restart".into(),
                        "3".into(),
                    ];
                    let jpeg = tool("cjpeg-static", &args, &input);
                    compare(
                        &jpeg,
                        bits,
                        &format!("lossless p={bits} c={channels} predictor={predictor} pt={point}"),
                        false,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo 3.2.0 cjpeg/djpeg"]
fn twelve_bit_dct_matches_turbo() {
    for (w, h) in [(1, 1), (3, 7), (17, 18), (35, 33)] {
        for channels in [1, 3] {
            for progressive in [false, true] {
                for sampling in [
                    "1x1",
                    "2x1",
                    "2x2",
                    "1x2",
                    "4x1",
                    "1x4",
                    "3x1",
                    "1x3",
                    "2x3",
                    "3x2",
                    "1x1,2x1,1x1",
                ] {
                    for (kind, quality, nearest) in [
                        (0, 5, false),
                        (0, 30, true),
                        (0, 60, false),
                        (0, 91, true),
                        (1, 91, false),
                        (2, 100, false),
                    ] {
                        let input = pnm(w, h, channels, 12, kind);
                        let mut args = vec![
                            "-precision".into(),
                            "12".into(),
                            "-quality".into(),
                            quality.to_string(),
                            "-sample".into(),
                            sampling.into(),
                            "-restart".into(),
                            "2".into(),
                        ];
                        if progressive {
                            args.push("-progressive".into());
                        }
                        let jpeg = tool("cjpeg-static", &args, &input);
                        compare(
                            &jpeg,
                            12,
                            &format!(
                                "DCT {w}x{h} c={channels} progressive={progressive} sampling={sampling} kind={kind} q={quality} nearest={nearest}"
                            ),
                            nearest,
                        );
                    }
                }
            }
        }
    }
}

// Small, independently specified streams keep regression coverage available
// without an external encoder or binary fixtures.
fn segment(jpeg: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    jpeg.extend_from_slice(&[255, marker]);
    jpeg.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    jpeg.extend_from_slice(payload);
}

fn constant_jpeg(bits: u8, lossless: bool, predictor: u8, point: u8) -> Vec<u8> {
    let mut jpeg = vec![255, 216];
    segment(
        &mut jpeg,
        if lossless { 0xc3 } else { 0xc1 },
        &[bits, 0, 19, 0, 17, 1, 1, 0x11, 0],
    );
    let mut table = vec![0, 1];
    table.extend_from_slice(&[0; 15]);
    table.push(0); // category zero has the one-bit code 0
    segment(&mut jpeg, 0xc4, &table);
    if !lossless {
        table[0] = 0x10;
        segment(&mut jpeg, 0xc4, &table);
        let mut quant = vec![0];
        quant.extend_from_slice(&[1; 64]);
        segment(&mut jpeg, 0xdb, &quant);
    }
    segment(
        &mut jpeg,
        0xda,
        &[
            1,
            1,
            0,
            if lossless { predictor } else { 0 },
            if lossless { 0 } else { 63 },
            point,
        ],
    );
    let bit_count: usize = if lossless { 17 * 19 } else { 9 * 2 };
    jpeg.resize(jpeg.len() + bit_count.div_ceil(8), 0);
    if bit_count % 8 != 0 {
        *jpeg.last_mut().unwrap() = (1 << (8 - bit_count % 8)) - 1;
    }
    jpeg.extend_from_slice(&[255, 217]);
    jpeg
}

#[test]
fn native_samples_and_converted_outputs() {
    for bits in 2..=16 {
        for predictor in 1..=7 {
            for point in [0, bits - 1] {
                let jpeg = constant_jpeg(bits, true, predictor, point);
                check_constant_outputs(&jpeg, bits);
            }
        }
    }
    check_constant_outputs(&constant_jpeg(12, false, 0, 0), 12);
}

fn check_constant_outputs(jpeg: &[u8], bits: u8) {
    let config = Decoder::new().output_format(PixelFormat::Gray);
    let value = 1u16 << (bits - 1);
    let result = config
        .clone()
        .output_target(OutputTarget::Native16)
        .decode(jpeg, Unstoppable)
        .unwrap();
    assert_eq!(result.pixels_u16().unwrap(), vec![value; 17 * 19]);
    assert_eq!(result.sample_precision(), bits);
    let mut direct = vec![0; 17 * 19];
    config
        .decode_into(jpeg, PixelFormat::Gray, &mut direct, Unstoppable)
        .unwrap();
    let expected = ((value as u32 * 255 + ((1u32 << bits) - 1) / 2) / ((1u32 << bits) - 1)) as u8;
    assert_eq!(direct, vec![expected; 17 * 19]);
    for owned in [false, true] {
        let input = if owned {
            Cow::Owned(jpeg.to_vec())
        } else {
            Cow::Borrowed(jpeg)
        };
        let mut reader = config.scanline_reader_cow(input).unwrap();
        let mut rows = vec![65535; 23 * 20];
        assert_eq!(
            reader
                .read_rows_gray16(ImgRefMut::new_stride(&mut rows, 17, 20, 23))
                .unwrap(),
            19
        );
        for row in rows.as_chunks::<23>().0.iter().take(19) {
            assert_eq!(&row[..17], vec![value; 17]);
            assert!(row[17..].iter().all(|&v| v == 65535));
        }
        assert!(reader.is_finished());
    }
    assert!(
        Decoder::new()
            .max_memory(1)
            .decode(jpeg, Unstoppable)
            .is_err()
    );
}

#[test]
fn lossless_truncations_and_invalid_scan_parameters() {
    let jpeg = constant_jpeg(16, true, 7, 0);
    for end in 0..jpeg.len() - 2 {
        assert!(
            Decoder::new().decode(&jpeg[..end], Unstoppable).is_err(),
            "truncation {end}"
        );
    }
    let sos = jpeg.windows(2).position(|p| p == [255, 218]).unwrap();
    for (offset, values) in [
        (7, &[0, 8, 255][..]),
        (8, &[1, 63, 255][..]),
        (9, &[16, 255][..]),
    ] {
        for &value in values {
            let mut invalid = jpeg.clone();
            invalid[sos + offset] = value;
            assert!(Decoder::new().decode(&invalid, Unstoppable).is_err());
        }
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo 3.2.0 cjpeg/djpeg"]
fn separate_lossless_scans_match_turbo() {
    for bits in [8, 12, 16] {
        let input = pnm(19, 17, 3, bits, 0);
        let jpeg = tool(
            "cjpeg-static",
            &[
                "-precision".into(),
                bits.to_string(),
                "-lossless".into(),
                "1".into(),
                "-scans".into(),
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/scan_scripts/lossless.txt"
                )
                .into(),
                "-restart".into(),
                "2".into(),
            ],
            &input,
        );
        compare(&jpeg, bits, "separate lossless scans", false);
        compare_transformed(&jpeg);
    }
}

fn compare_transformed(jpeg: &[u8]) {
    use zenjpeg::lossless::LosslessTransform as T;
    let config = Decoder::new().output_target(OutputTarget::Native16);
    let source = config.decode(jpeg, Unstoppable).unwrap();
    let (w, h) = (source.width() as usize, source.height() as usize);
    let input = source.pixels_u16().unwrap();
    let cropped = config
        .clone()
        .crop(zenjpeg::decoder::CropRegion::pixels(3, 2, 7, 9));
    let expected_crop: Vec<u16> = (2..11)
        .flat_map(|y| input[(y * w + 3) * 3..(y * w + 10) * 3].iter().copied())
        .collect();
    assert_eq!(
        cropped
            .decode(jpeg, Unstoppable)
            .unwrap()
            .pixels_u16()
            .unwrap(),
        expected_crop
    );
    for data in [Cow::Borrowed(jpeg), Cow::Owned(jpeg.to_vec())] {
        let mut reader = cropped.scanline_reader_cow(data).unwrap();
        let mut row = vec![rgb::RGB::new(0, 0, 0); 7];
        let mut actual = Vec::new();
        while !reader.is_finished() {
            assert_eq!(
                reader
                    .read_rows_rgb16(ImgRefMut::new(&mut row, 7, 1))
                    .unwrap(),
                1
            );
            actual.extend(row.iter().flat_map(|p| [p.r, p.g, p.b]));
        }
        assert_eq!(actual, expected_crop);
    }
    for transform in [
        T::FlipHorizontal,
        T::FlipVertical,
        T::Rotate90,
        T::Rotate180,
        T::Rotate270,
        T::Transpose,
        T::Transverse,
    ] {
        let cfg = config.clone().transform(transform);
        let actual = cfg.decode(jpeg, Unstoppable).unwrap();
        let (ow, oh) = (actual.width() as usize, actual.height() as usize);
        let mut expected = Vec::new();
        for y in 0..oh {
            for x in 0..ow {
                let (sx, sy) = match transform {
                    T::FlipHorizontal => (w - 1 - x, y),
                    T::FlipVertical => (x, h - 1 - y),
                    T::Rotate90 => (y, h - 1 - x),
                    T::Rotate180 => (w - 1 - x, h - 1 - y),
                    T::Rotate270 => (w - 1 - y, x),
                    T::Transpose => (y, x),
                    T::Transverse => (w - 1 - y, h - 1 - x),
                    T::None => unreachable!(),
                };
                expected.extend_from_slice(&input[(sy * w + sx) * 3..(sy * w + sx + 1) * 3]);
            }
        }
        assert_eq!(actual.pixels_u16().unwrap(), expected, "{transform:?}");
        for data in [Cow::Borrowed(jpeg), Cow::Owned(jpeg.to_vec())] {
            let mut reader = cfg.scanline_reader_cow(data).unwrap();
            let mut output = vec![rgb::RGB::new(0, 0, 0); ow * oh];
            assert_eq!(
                reader
                    .read_rows_rgb16(ImgRefMut::new(&mut output, ow, oh))
                    .unwrap(),
                oh
            );
            let flat: Vec<u16> = output.iter().flat_map(|p| [p.r, p.g, p.b]).collect();
            assert_eq!(flat, expected, "scanline {transform:?}");
        }
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo 3.2.0 cjpeg/djpeg"]
fn rgb_and_arithmetic_twelve_bit_match_turbo() {
    for rgb in [false, true] {
        for progressive in [false, true] {
            for arithmetic in [false, true] {
                let mut args = vec![
                    "-precision".into(),
                    "12".into(),
                    "-quality".into(),
                    "73".into(),
                ];
                if rgb {
                    args.push("-rgb".into());
                }
                if progressive {
                    args.push("-progressive".into());
                }
                if arithmetic {
                    args.push("-arithmetic".into());
                }
                let jpeg = tool("cjpeg-static", &args, &pnm(19, 17, 3, 12, 0));
                compare(
                    &jpeg,
                    12,
                    &format!("RGB={rgb} progressive={progressive} arithmetic={arithmetic}"),
                    false,
                );
                compare_transformed(&jpeg);
            }
        }
    }
}

#[test]
fn lossless_category_sixteen_and_tables_before_frame() {
    for table_first in [false, true] {
        let mut jpeg = vec![255, 216];
        let mut table = vec![0];
        table.extend_from_slice(&[0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        table.extend(0..=16);
        if table_first {
            segment(&mut jpeg, 0xc4, &table);
        }
        segment(&mut jpeg, 0xc3, &[16, 0, 1, 0, 3, 1, 1, 0x11, 0]);
        if !table_first {
            segment(&mut jpeg, 0xc4, &table);
        }
        segment(&mut jpeg, 0xda, &[1, 1, 0, 1, 0, 0]);
        // Differences -32768 (category 16, no magnitude), -1 and -32767.
        let bits = "1000000001001111000000000000000";
        let mut padded = bits.to_owned();
        while padded.len() % 8 != 0 {
            padded.push('1');
        }
        for chunk in padded.as_bytes().as_chunks::<8>().0 {
            let byte = chunk.iter().fold(0u8, |v, b| (v << 1) | (b - b'0'));
            jpeg.push(byte);
            if byte == 255 {
                jpeg.push(0);
            }
        }
        jpeg.extend_from_slice(&[255, 217]);
        let config = Decoder::new()
            .output_target(OutputTarget::Native16)
            .output_format(PixelFormat::Gray);
        assert_eq!(
            config
                .decode(&jpeg, Unstoppable)
                .unwrap()
                .pixels_u16()
                .unwrap(),
            [0, 65535, 32768]
        );
    }
}

#[test]
fn native_cancellation_preserves_reason() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct StopAfter(AtomicUsize);
    impl enough::Stop for StopAfter {
        fn check(&self) -> Result<(), enough::StopReason> {
            if self.0.fetch_sub(1, Ordering::Relaxed) <= 1 {
                Err(enough::StopReason::Cancelled)
            } else {
                Ok(())
            }
        }
    }
    for lossless in [false, true] {
        for check in [1, 3, 8] {
            let jpeg = constant_jpeg(12, lossless, 1, 0);
            let error = Decoder::new()
                .decode(&jpeg, StopAfter(AtomicUsize::new(check)))
                .unwrap_err();
            assert!(matches!(
                error.kind(),
                zenjpeg::decoder::ErrorKind::Cancelled(_)
            ));
        }
    }
}

#[cfg(feature = "inventory")]
#[test]
fn inventory_recognizes_native_image_data() {
    use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
    for bits in 2..=16 {
        for lossless in [true, false] {
            if !lossless && bits != 12 {
                continue;
            }
            let jpeg = constant_jpeg(bits, lossless, 7, 0);
            let inventory = zenjpeg::JpegDecoderConfig::new()
                .job()
                .inventory(&jpeg)
                .unwrap();
            inventory.validate().unwrap();
            assert!(
                inventory
                    .parts()
                    .iter()
                    .any(|p| p.disposition() == zencodec::inventory::Disposition::ImageData),
                "p={bits} lossless={lossless}: {inventory}"
            );
        }
    }
}

#[test]
fn native_metadata_is_preserved_once_before_and_after_scans() {
    let xml = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><value>native samples</value></x:xmpmeta>";
    let mut payload = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
    payload.extend_from_slice(xml.as_bytes());
    let mut app = Vec::new();
    segment(&mut app, 0xe1, &payload);
    for lossless in [false, true] {
        for late in [false, true] {
            let mut jpeg = constant_jpeg(12, lossless, 1, 0);
            let at = if late { jpeg.len() - 2 } else { 2 };
            jpeg.splice(at..at, app.iter().copied());
            let result = Decoder::new()
                .preserve_all_metadata()
                .output_target(OutputTarget::Native16)
                .decode(&jpeg, Unstoppable)
                .unwrap();
            assert_eq!(result.extras().unwrap().xmp(), Some(xml));
            assert_eq!(result.sample_precision(), 12);
        }
    }
}

#[test]
fn native_eight_bit_reads_preserve_configured_rounding() {
    use zenjpeg::decoder::IdctMethod;
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let pixels: Vec<u8> = (0..19 * 17 * 3)
        .map(|i| ((i * 197 + i * i * 31) & 255) as u8)
        .collect();
    for progressive in [false, true] {
        let mut encoder = EncoderConfig::ycbcr(73.0, ChromaSubsampling::Quarter)
            .progressive(progressive)
            .encode_from_bytes(19, 17, PixelLayout::Rgb8Srgb)
            .unwrap();
        encoder.push_packed(&pixels, Unstoppable).unwrap();
        let jpeg = encoder.finish().unwrap();
        for idct in [IdctMethod::Libjpeg, IdctMethod::Jpegli] {
            for format in [PixelFormat::Rgb, PixelFormat::Gray] {
                let config = Decoder::new().idct_method(idct).output_format(format);
                let bytes = config.decode(&jpeg, Unstoppable).unwrap();
                let native = config
                    .output_target(OutputTarget::Native16)
                    .decode(&jpeg, Unstoppable)
                    .unwrap();
                let expected: Vec<u16> = bytes
                    .pixels_u8()
                    .unwrap()
                    .iter()
                    .map(|&v| v as u16)
                    .collect();
                assert_eq!(
                    native.pixels_u16().unwrap(),
                    expected,
                    "progressive={progressive} idct={idct:?} format={format:?}"
                );
            }
        }
    }
}

// TIFF JPEGTables carries SOI + shared tables + EOI; each strip/tile carries
// its own frame/scan. The consumer joins them after removing the two boundary
// markers, leaving per-scan table redefinitions in their original positions.
fn assemble_tiff_jpeg(jpeg: &[u8]) -> Vec<u8> {
    use zenjpeg::container::marker::{self, MarkerKind};
    let mut tables = vec![255, 216];
    let mut image = vec![255, 216];
    let mut before_scan = true;
    for span in marker::iter(jpeg) {
        if span.kind == MarkerKind::Soi {
            continue;
        }
        if span.kind == MarkerKind::Sos {
            before_scan = false;
        }
        let bytes = &jpeg[span.offset..span.offset + span.length];
        if before_scan && matches!(span.kind, MarkerKind::Dqt | MarkerKind::Dht) {
            tables.extend_from_slice(bytes);
        } else {
            image.extend_from_slice(bytes);
        }
    }
    tables.extend_from_slice(&[255, 217]);
    tables.truncate(tables.len() - 2);
    tables.extend_from_slice(&image[2..]);
    tables
}

#[test]
fn tiff_shared_tables_keep_native_precision() {
    for (bits, lossless) in [(12, false), (12, true), (16, true)] {
        let jpeg = constant_jpeg(bits, lossless, 7, 0);
        check_constant_outputs(&assemble_tiff_jpeg(&jpeg), bits);
    }
}

#[test]
fn twelve_bit_native_planes_retain_sample_range_and_stride() {
    let mut jpeg = vec![255, 216];
    segment(
        &mut jpeg,
        0xc1,
        &[12, 0, 8, 0, 8, 3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0],
    );
    let mut quant = vec![0];
    quant.extend_from_slice(&[1; 64]);
    segment(&mut jpeg, 0xdb, &quant);
    let mut table = vec![0, 1];
    table.extend_from_slice(&[0; 15]);
    table.push(0);
    segment(&mut jpeg, 0xc4, &table);
    table[0] = 0x10;
    segment(&mut jpeg, 0xc4, &table);
    segment(&mut jpeg, 0xda, &[3, 1, 0, 2, 0, 3, 0, 0, 63, 0]);
    jpeg.extend_from_slice(&[3, 255, 217]); // three zero DC/EOB blocks, then padding
    for data in [
        Cow::Borrowed(jpeg.as_slice()),
        Cow::Owned(assemble_tiff_jpeg(&jpeg)),
    ] {
        let mut reader = Decoder::new().scanline_reader_cow(data).unwrap();
        assert_eq!(reader.sample_precision(), 12);
        let mut y = vec![-1; 13 * 8];
        let mut cb = y.clone();
        let mut cr = y.clone();
        assert_eq!(
            reader
                .read_rows_ycbcr_native_i16(&mut y, 13, &mut cb, &mut cr, 13, 1)
                .unwrap(),
            (8, 8)
        );
        for plane in [&y, &cb, &cr] {
            for row in plane.as_chunks::<13>().0 {
                assert_eq!(&row[..8], &[2048; 8]);
                assert_eq!(&row[8..], &[-1; 5]);
            }
        }
    }
}

fn compare_components<const N: usize>(jpeg: &[u8], precision: u8) {
    let expected: Vec<u16> = tool(
        "native-components-oracle",
        &["decode".into(), "raw".into()],
        jpeg,
    )
    .as_chunks::<2>()
    .0
    .iter()
    .map(|p| u16::from_be_bytes([p[0], p[1]]))
    .collect();
    for data in [Cow::Borrowed(jpeg), Cow::Owned(jpeg.to_vec())] {
        let mut reader = Decoder::new()
            .output_target(OutputTarget::Native16)
            .scanline_reader_cow(data)
            .unwrap();
        assert_eq!(reader.sample_precision(), precision);
        assert_eq!(reader.component_ids().len(), N);
        let (w, h) = (reader.width() as usize, reader.height() as usize);
        let stride = w + 3;
        let mut output = vec![[65535; N]; stride * 3];
        let mut y = 0;
        while y < h {
            let count = reader
                .read_rows_components16(ImgRefMut::new_stride(&mut output, w, 3, stride))
                .unwrap();
            assert!(count > 0);
            for row in 0..count {
                let flat: Vec<u16> = output[row * stride..row * stride + w]
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                assert_eq!(
                    flat,
                    expected[(y + row) * w * N..(y + row + 1) * w * N],
                    "components N={N} precision={precision} row={}",
                    y + row
                );
                assert_eq!(
                    output[row * stride + w..(row + 1) * stride],
                    [[65535; N]; 3]
                );
            }
            y += count;
        }
    }
    let mut cropped = Decoder::new()
        .crop(zenjpeg::decoder::CropRegion::pixels(3, 2, 7, 9))
        .scanline_reader(jpeg)
        .unwrap();
    let mut crop = vec![[0u16; N]; 7 * 9];
    assert_eq!(
        cropped
            .read_rows_components16(ImgRefMut::new(&mut crop, 7, 9))
            .unwrap(),
        9
    );
    for y in 0..9 {
        let actual: Vec<u16> = crop[y * 7..y * 7 + 7].iter().flatten().copied().collect();
        assert_eq!(
            actual,
            expected[((y + 2) * 35 + 3) * N..((y + 2) * 35 + 10) * N]
        );
    }
    if N == 4 {
        let expected: Vec<u16> = tool(
            "native-components-oracle",
            &["decode".into(), "cmyk".into()],
            jpeg,
        )
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| u16::from_be_bytes([p[0], p[1]]))
        .collect();
        let cfg = Decoder::new()
            .output_target(OutputTarget::Native16)
            .output_format(PixelFormat::Cmyk);
        let decoded = cfg.decode(jpeg, Unstoppable).unwrap();
        assert_eq!(
            decoded.pixels_u16().unwrap(),
            expected,
            "CMYK precision={precision}"
        );
        let mut reader = cfg.scanline_reader(jpeg).unwrap();
        let (w, h) = (reader.width() as usize, reader.height() as usize);
        let mut pixels = vec![zenjpeg::decoder::Cmyk16::default(); (w + 2) * h];
        assert_eq!(
            reader
                .read_rows_cmyk16(ImgRefMut::new_stride(&mut pixels, w, h, w + 2))
                .unwrap(),
            h
        );
        let actual: Vec<u16> = pixels
            .chunks_exact(w + 2)
            .flat_map(|row| row[..w].iter().flat_map(|p| [p.c, p.m, p.y, p.k]))
            .collect();
        assert_eq!(actual, expected);
        let cfg = cfg
            .transform(zenjpeg::lossless::LosslessTransform::Transpose)
            .crop(zenjpeg::decoder::CropRegion::pixels(2, 3, 7, 5));
        let mut reader = cfg.scanline_reader(jpeg).unwrap();
        let mut pixels = vec![zenjpeg::decoder::Cmyk16::default(); 7 * 5];
        assert_eq!(
            reader
                .read_rows_cmyk16(ImgRefMut::new(&mut pixels, 7, 5))
                .unwrap(),
            5
        );
        for y in 0..5 {
            for x in 0..7 {
                let p = pixels[y * 7 + x];
                let start = ((x + 2) * 35 + y + 3) * 4;
                assert_eq!([p.c, p.m, p.y, p.k], expected[start..start + 4]);
            }
        }
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo component oracle"]
fn native_multicomponent_matches_turbo() {
    for n in [2, 4] {
        for bits in [2, 8, 9, 12, 13, 16] {
            for predictor in 1..=7 {
                for point in [0, bits - 1] {
                    let args = [
                        "encode".to_owned(),
                        bits.to_string(),
                        n.to_string(),
                        "1".into(),
                        predictor.to_string(),
                        point.to_string(),
                        "1".into(),
                        "0".into(),
                    ];
                    let jpeg = tool("native-components-oracle", &args, &[]);
                    if n == 2 {
                        compare_components::<2>(&jpeg, bits);
                    } else {
                        compare_components::<4>(&jpeg, bits);
                    }
                }
            }
        }
        for sampling in [1, 2, 3, 4] {
            for progressive in [0, 1] {
                let args = [
                    "encode".to_owned(),
                    "12".into(),
                    n.to_string(),
                    "0".into(),
                    "1".into(),
                    "0".into(),
                    sampling.to_string(),
                    progressive.to_string(),
                ];
                let jpeg = tool("native-components-oracle", &args, &[]);
                if n == 2 {
                    compare_components::<2>(&jpeg, 12);
                } else {
                    compare_components::<4>(&jpeg, 12);
                }
            }
        }
    }
}

fn constant_components(bits: u8, n: u8, lossless: bool) -> Vec<u8> {
    let mut jpeg = vec![255, 216];
    let mut frame = vec![bits, 0, 19, 0, 17, n];
    for c in 0..n {
        frame.extend_from_slice(&[10 + c, 0x11, 0]);
    }
    segment(&mut jpeg, if lossless { 0xc3 } else { 0xc1 }, &frame);
    let mut table = vec![0, 1];
    table.extend_from_slice(&[0; 15]);
    table.push(0);
    segment(&mut jpeg, 0xc4, &table);
    if !lossless {
        table[0] = 0x10;
        segment(&mut jpeg, 0xc4, &table);
        let mut quant = vec![0];
        quant.extend_from_slice(&[1; 64]);
        segment(&mut jpeg, 0xdb, &quant);
    }
    let mut scan = vec![n];
    for c in (0..n).rev() {
        scan.extend_from_slice(&[10 + c, 0]);
    }
    scan.extend_from_slice(if lossless { &[1, 0, 0] } else { &[0, 63, 0] });
    segment(&mut jpeg, 0xda, &scan);
    let bits = if lossless {
        17 * 19 * n as usize
    } else {
        9 * 2 * n as usize
    };
    jpeg.resize(jpeg.len() + bits.div_ceil(8), 0);
    if bits % 8 != 0 {
        *jpeg.last_mut().unwrap() = (1 << (8 - bits % 8)) - 1;
    }
    jpeg.extend_from_slice(&[255, 217]);
    jpeg
}

#[test]
fn component_rows_preserve_order_precision_and_padding() {
    for (bits, lossless) in [(12, false), (2, true), (8, true), (12, true), (16, true)] {
        for n in [2, 4] {
            let jpeg = constant_components(bits, n, lossless);
            for data in [Cow::Borrowed(jpeg.as_slice()), Cow::Owned(jpeg.clone())] {
                let mut reader = Decoder::new().scanline_reader_cow(data).unwrap();
                assert_eq!(
                    reader.component_ids(),
                    if n == 2 {
                        &[10, 11][..]
                    } else {
                        &[10, 11, 12, 13][..]
                    }
                );
                let center = 1u16 << (bits - 1);
                if n == 2 {
                    let mut invalid = vec![[0; 3]; 17];
                    assert!(
                        reader
                            .read_rows_components16(ImgRefMut::new(&mut invalid, 17, 1))
                            .is_err()
                    );
                    let mut output = vec![[65535; 2]; 20 * 19];
                    assert_eq!(
                        reader
                            .read_rows_components16(ImgRefMut::new_stride(&mut output, 17, 19, 20))
                            .unwrap(),
                        19
                    );
                    for row in output.as_chunks::<20>().0.iter() {
                        assert_eq!(row[..17], [[center; 2]; 17]);
                        assert_eq!(row[17..], [[65535; 2]; 3]);
                    }
                } else {
                    let mut output = vec![zenjpeg::decoder::Cmyk16::default(); 20 * 19];
                    assert_eq!(
                        reader
                            .read_rows_cmyk16(ImgRefMut::new_stride(&mut output, 17, 19, 20))
                            .unwrap(),
                        19
                    );
                    for row in output.as_chunks::<20>().0.iter() {
                        for p in &row[..17] {
                            assert_eq!([p.c, p.m, p.y, p.k], [center; 4]);
                        }
                        assert_eq!(row[17..], [zenjpeg::decoder::Cmyk16::default(); 3]);
                    }
                    let decoded = Decoder::new()
                        .output_target(OutputTarget::Native16)
                        .output_format(PixelFormat::Cmyk)
                        .decode(&jpeg, Unstoppable)
                        .unwrap();
                    assert_eq!(decoded.pixels_u16().unwrap(), vec![center; 17 * 19 * 4]);
                }
            }
        }
    }
}

fn verify_transcoding(jpeg: &[u8]) {
    use zenjpeg::lossless::{OutputMode, RestructureConfig, restructure};
    let original = Decoder::new()
        .decode_coefficients(jpeg, Unstoppable)
        .unwrap();
    assert_eq!(original.sample_precision, 12);
    let raw = tool(
        "native-components-oracle",
        &["decode".into(), "raw".into()],
        jpeg,
    );
    for output_mode in [OutputMode::Sequential, OutputMode::Progressive] {
        let bytes = restructure(
            jpeg,
            &RestructureConfig {
                output_mode,
                ..Default::default()
            },
            Unstoppable,
        )
        .unwrap();
        assert_eq!(Decoder::new().read_info(&bytes).unwrap().precision, 12);
        let roundtrip = Decoder::new()
            .decode_coefficients(&bytes, Unstoppable)
            .unwrap();
        assert_eq!(roundtrip.sample_precision, 12);
        assert_eq!(original.quant_tables, roundtrip.quant_tables);
        let max_h = original
            .components
            .iter()
            .map(|c| c.h_samp as usize)
            .max()
            .unwrap();
        let max_v = original
            .components
            .iter()
            .map(|c| c.v_samp as usize)
            .max()
            .unwrap();
        for (a, b) in original.components.iter().zip(&roundtrip.components) {
            assert_eq!((a.id, a.h_samp, a.v_samp), (b.id, b.h_samp, b.v_samp));
            let w = (original.width as usize * a.h_samp as usize).div_ceil(max_h * 8);
            let h = (original.height as usize * a.v_samp as usize).div_ceil(max_v * 8);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        a.block(y * a.blocks_wide + x),
                        b.block(y * b.blocks_wide + x)
                    );
                }
            }
        }
        assert_eq!(
            tool(
                "native-components-oracle",
                &["decode".into(), "raw".into()],
                &bytes
            ),
            raw,
            "{output_mode:?}"
        );
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo component oracle"]
fn twelve_bit_coefficient_transcoding_matches_turbo() {
    for n in [1, 3] {
        for sampling in ["1x1", "2x2", "4x1"] {
            for quality in [5, 60, 100] {
                let args = vec![
                    "-precision".into(),
                    "12".into(),
                    "-quality".into(),
                    quality.to_string(),
                    "-sample".into(),
                    sampling.into(),
                    "-progressive".into(),
                ];
                let jpeg = tool("cjpeg-static", &args, &pnm(35, 33, n, 12, 0));
                verify_transcoding(&jpeg);
            }
        }
    }
    for n in [2, 4] {
        for sampling in [1, 2, 4] {
            let args = [
                "encode".to_owned(),
                "12".into(),
                n.to_string(),
                "0".into(),
                "1".into(),
                "0".into(),
                sampling.to_string(),
                "0".into(),
            ];
            verify_transcoding(&tool("native-components-oracle", &args, &[]));
        }
    }
}

#[test]
fn native_coefficient_precision_survives_restructure() {
    use zenjpeg::lossless::{OutputMode, RestructureConfig, restructure};
    for n in [1, 2, 3, 4] {
        let jpeg = constant_components(12, n, false);
        for output_mode in [OutputMode::Sequential, OutputMode::Progressive] {
            let bytes = restructure(
                &jpeg,
                &RestructureConfig {
                    output_mode,
                    ..Default::default()
                },
                Unstoppable,
            )
            .unwrap();
            let coeffs = Decoder::new()
                .decode_coefficients(&bytes, Unstoppable)
                .unwrap();
            assert_eq!(coeffs.sample_precision, 12);
            assert_eq!(coeffs.components.len(), n as usize);
            assert!(
                coeffs
                    .components
                    .iter()
                    .all(|c| c.coeffs.iter().all(|&v| v == 0))
            );
        }
    }
}

#[test]
#[ignore = "requires explicitly configured libjpeg-turbo jpegtran oracle"]
fn twelve_bit_coefficient_transforms_match_jpegtran() {
    use zenjpeg::lossless::{LosslessTransform as T, TransformConfig, transform};
    let transforms = [
        (T::FlipHorizontal, vec!["-flip", "horizontal"]),
        (T::FlipVertical, vec!["-flip", "vertical"]),
        (T::Rotate90, vec!["-rotate", "90"]),
        (T::Rotate180, vec!["-rotate", "180"]),
        (T::Rotate270, vec!["-rotate", "270"]),
        (T::Transpose, vec!["-transpose"]),
        (T::Transverse, vec!["-transverse"]),
    ];
    for n in [1, 3] {
        for sampling in ["1x1", "2x2", "4x1"] {
            for kind in [0, 1] {
                let jpeg = tool(
                    "cjpeg-static",
                    &[
                        "-precision".into(),
                        "12".into(),
                        "-quality".into(),
                        "60".into(),
                        "-sample".into(),
                        sampling.into(),
                    ],
                    &pnm(32, 32, n, 12, kind),
                );
                for (op, args) in &transforms {
                    let mut args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
                    args.extend(["-copy".into(), "all".into()]);
                    let reference = tool("jpegtran-static", &args, &jpeg);
                    let ours = transform(
                        &jpeg,
                        &TransformConfig {
                            transform: *op,
                            ..Default::default()
                        },
                        Unstoppable,
                    )
                    .unwrap();
                    let decode = ["decode".into(), "raw".into()];
                    assert_eq!(
                        tool("native-components-oracle", &decode, &ours),
                        tool("native-components-oracle", &decode, &reference),
                        "{op:?} channels={n} sampling={sampling} kind={kind}"
                    );
                }
            }
        }
    }
    let jpeg = tool(
        "native-components-oracle",
        &[
            "encode".into(),
            "12".into(),
            "4".into(),
            "0".into(),
            "1".into(),
            "0".into(),
            "2".into(),
            "0".into(),
        ],
        &[],
    );
    let reference = tool(
        "jpegtran-static",
        &["-transpose".into(), "-copy".into(), "all".into()],
        &jpeg,
    );
    let ours = transform(
        &jpeg,
        &TransformConfig {
            transform: T::Transpose,
            ..Default::default()
        },
        Unstoppable,
    )
    .unwrap();
    assert_eq!(
        tool(
            "native-components-oracle",
            &["decode".into(), "cmyk".into()],
            &ours
        ),
        tool(
            "native-components-oracle",
            &["decode".into(), "cmyk".into()],
            &reference
        )
    );
}

#[test]
fn native_component_readers_reject_bad_inputs_without_panicking() {
    for lossless in [false, true] {
        let jpeg = constant_components(12, 4, lossless);
        assert!(Decoder::new().max_memory(1).scanline_reader(&jpeg).is_err());
        for end in 0..jpeg.len() {
            let attempt = std::panic::catch_unwind(|| {
                if let Ok(mut reader) = Decoder::new().scanline_reader(&jpeg[..end]) {
                    let mut pixels = vec![[0u16; 4]; 17 * 19];
                    let _ = reader.read_rows_components16(ImgRefMut::new(&mut pixels, 17, 19));
                }
            });
            assert!(attempt.is_ok(), "lossless={lossless} prefix={end}");
        }
        let mut malformed = jpeg.clone();
        let sos = malformed.windows(2).position(|w| w == [255, 218]).unwrap();
        malformed[sos + 7] = malformed[sos + 5]; // duplicate scan component
        assert!(Decoder::new().scanline_reader(&malformed).is_err());
    }
    let jpeg = constant_components(12, 2, false);
    let mut reader = Decoder::new().scanline_reader(&jpeg).unwrap();
    let mut pixels = vec![rgb::RGB::new(0, 0, 0); 17];
    assert!(
        reader
            .read_rows_rgb16(ImgRefMut::new(&mut pixels, 17, 1))
            .is_err()
    );
}

#[test]
fn cmyk_crop_and_pixel_transforms_preserve_native_channels() {
    use zenjpeg::lossless::LosslessTransform;
    for lossless in [false, true] {
        let jpeg = constant_components(12, 4, lossless);
        let cfg = Decoder::new()
            .output_target(OutputTarget::Native16)
            .output_format(PixelFormat::Cmyk)
            .transform(LosslessTransform::Transpose)
            .crop(zenjpeg::decoder::CropRegion::pixels(2, 3, 7, 5));
        let result = cfg.decode(&jpeg, Unstoppable).unwrap();
        assert_eq!(result.pixels_u16().unwrap(), vec![2048; 7 * 5 * 4]);
        for data in [Cow::Borrowed(jpeg.as_slice()), Cow::Owned(jpeg.clone())] {
            let mut reader = cfg.scanline_reader_cow(data).unwrap();
            assert_eq!((reader.width(), reader.height()), (7, 5));
            let mut pixels = vec![zenjpeg::decoder::Cmyk16::default(); 9 * 5];
            assert_eq!(
                reader
                    .read_rows_cmyk16(ImgRefMut::new_stride(&mut pixels, 7, 5, 9))
                    .unwrap(),
                5
            );
            for row in pixels.as_chunks::<9>().0 {
                for p in &row[..7] {
                    assert_eq!([p.c, p.m, p.y, p.k], [2048; 4]);
                }
                assert_eq!(row[7..], [zenjpeg::decoder::Cmyk16::default(); 2]);
            }
        }
    }
}
