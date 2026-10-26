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
                for sampling in ["1x1", "2x1", "2x2", "1x2"] {
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
