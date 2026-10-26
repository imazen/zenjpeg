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
    let actual = result.pixels_u16().unwrap();
    assert_eq!(actual.len(), expected.len(), "{label}");
    if let Some(i) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: sample {i}, ours {}, turbo {}",
            actual[i], expected[i]
        );
    }
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
