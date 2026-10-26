//! Frame components, scan slots and table slots are independent namespaces.
use imgref::ImgRefMut;
use zenjpeg::decoder::Decoder;

fn segment(out: &mut Vec<u8>, marker: u8, data: &[u8]) {
    out.extend_from_slice(&[255, marker]);
    out.extend_from_slice(&((data.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(data);
}

fn fixture(n: u8, precision: u8, lossless: bool) -> Vec<u8> {
    let mut out = vec![255, 216];
    let mut frame = vec![precision, 0, 8, 0, 8, n];
    for i in 0..n {
        frame.extend_from_slice(&[200 - i, 0x11, if lossless { 0 } else { i % 4 }]);
    }
    segment(&mut out, if lossless { 0xc3 } else { 0xc1 }, &frame);
    for slot in 0..4 {
        let mut table = vec![slot, 1];
        table.extend_from_slice(&[0; 15]);
        table.push(0);
        segment(&mut out, 0xc4, &table);
        if !lossless {
            table[0] |= 0x10;
            segment(&mut out, 0xc4, &table);
            let mut quant = vec![slot];
            quant.extend_from_slice(&[slot + 1; 64]);
            segment(&mut out, 0xdb, &quant);
        }
    }
    for start in (0..n).step_by(4) {
        let end = (start + 4).min(n);
        let mut scan = vec![end - start];
        for i in (start..end).rev() {
            scan.extend_from_slice(&[200 - i, if lossless { (i % 4) << 4 } else { (i % 4) * 17 }]);
        }
        scan.extend_from_slice(if lossless { &[1, 0, 0] } else { &[0, 63, 0] });
        segment(&mut out, 0xda, &scan);
        let bits = usize::from(end - start) * if lossless { 64 } else { 2 };
        out.resize(out.len() + bits.div_ceil(8), 0);
        if bits % 8 != 0 {
            *out.last_mut().unwrap() = (1 << (8 - bits % 8)) - 1;
        }
    }
    out.extend_from_slice(&[255, 217]);
    out
}

fn check<const N: usize>() {
    for lossless in [false, true] {
        for precision in if lossless {
            vec![2, 8, 12, 16]
        } else {
            vec![8, 12]
        } {
            let jpeg = fixture(N as u8, precision, lossless);
            for owned in [false, true] {
                let config = Decoder::new();
                let mut reader = if owned {
                    config.scanline_reader_cow(std::borrow::Cow::Owned(jpeg.clone()))
                } else {
                    config.scanline_reader(&jpeg)
                }
                .unwrap();
                assert_eq!(
                    reader.component_ids(),
                    &(0..N).map(|i| 200 - i as u8).collect::<Vec<_>>()
                );
                let mut rows = vec![[65535; N]; 10 * 8];
                assert_eq!(
                    reader
                        .read_rows_components16(ImgRefMut::new_stride(&mut rows, 8, 8, 10))
                        .unwrap(),
                    8
                );
                for row in rows.as_chunks::<10>().0.iter() {
                    assert!(row[..8].iter().all(|p| *p == [1 << (precision - 1); N]));
                    assert_eq!(row[8..], [[65535; N]; 2]);
                }
            }
        }
    }
}
#[test]
fn all_frame_component_counts() {
    check::<5>();
    check::<6>();
    check::<7>();
    check::<8>();
    check::<9>();
    check::<10>();
}

fn redefined_tables() -> Vec<u8> {
    let mut out = vec![255, 216];
    let mut frame = vec![12, 0, 8, 0, 8, 10];
    for i in 0..10 {
        frame.extend_from_slice(&[200 - i, 0x11, i % 4]);
    }
    segment(&mut out, 0xc1, &frame);
    let mut dc = vec![0];
    dc.extend_from_slice(&[0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    dc.extend(0..8);
    segment(&mut out, 0xc4, &dc);
    let mut ac = vec![0x10, 1];
    ac.extend_from_slice(&[0; 15]);
    ac.push(0);
    segment(&mut out, 0xc4, &ac);
    for start in [0u8, 4, 8] {
        for slot in 0..4 {
            let mut quant = vec![slot];
            quant.extend_from_slice(&[(slot + 1) * (start / 4 + 1); 64]);
            segment(&mut out, 0xdb, &quant);
        }
        let end = (start + 4).min(10);
        let mut scan = vec![end - start];
        for i in (start..end).rev() {
            scan.extend_from_slice(&[200 - i, 0]);
        }
        scan.extend_from_slice(&[0, 63, 0]);
        segment(&mut out, 0xda, &scan);
        let mut bits = Vec::new();
        for i in (start..end).rev() {
            let dc = 8 * (u32::from(i) + 1);
            let size = 32 - dc.leading_zeros();
            for bit in (0..5).rev() {
                bits.push((size >> bit) & 1);
            }
            for bit in (0..size).rev() {
                bits.push((dc >> bit) & 1);
            }
            bits.push(0); // AC EOB
        }
        while bits.len() % 8 != 0 {
            bits.push(1);
        }
        for byte in bits.as_chunks::<8>().0.iter() {
            let b = byte.iter().fold(0u8, |v, b| (v << 1) | *b as u8);
            out.push(b);
            if b == 255 {
                out.push(0);
            }
        }
    }
    out.extend_from_slice(&[255, 217]);
    out
}

#[test]
fn quantization_slots_are_latched_per_component() {
    let jpeg = redefined_tables();
    let config = Decoder::new();
    let mut reader = config.scanline_reader(&jpeg).unwrap();
    let mut output = vec![[0; 10]; 64];
    reader
        .read_rows_components16(ImgRefMut::new(&mut output, 8, 8))
        .unwrap();
    let expected =
        std::array::from_fn(|i| 2048 + (i as u16 + 1) * (i as u16 % 4 + 1) * (i as u16 / 4 + 1));
    assert!(
        output.iter().all(|p| *p == expected),
        "{:?} != {:?}",
        output[0],
        expected
    );
    let coeffs = config
        .decode_coefficients(&jpeg, zenjpeg::encoder::Unstoppable)
        .unwrap();
    for (i, c) in coeffs.components.iter().enumerate() {
        assert_eq!(c.block(0)[0], 8 * (i as i16 + 1));
        assert_eq!(
            coeffs.quant_tables[c.quant_table_idx as usize].unwrap(),
            [(i as u16 % 4 + 1) * (i as u16 / 4 + 1); 64]
        );
    }
}

fn oracle(args: &[String], input: &[u8]) -> Vec<u8> {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let dir = std::env::var_os("ZENJPEG_TURBO_BIN")
        .expect("set ZENJPEG_TURBO_BIN for the explicit oracle test");
    let mut child = Command::new(std::path::PathBuf::from(dir).join("native-components-oracle"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{:?}: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}
fn compare<const N: usize>(jpeg: &[u8]) -> Vec<u16> {
    let bytes = oracle(&["decode".into(), "raw".into()], jpeg);
    let expected: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_be_bytes([b[0], b[1]]))
        .collect();
    for strictness in [
        zenjpeg::decoder::Strictness::Strict,
        zenjpeg::decoder::Strictness::Balanced,
    ] {
        let config = Decoder::new().strictness(strictness);
        for data in [
            std::borrow::Cow::Borrowed(jpeg),
            std::borrow::Cow::Owned(jpeg.to_vec()),
        ] {
            let mut reader = config.scanline_reader_cow(data).unwrap();
            let (w, h) = (reader.width() as usize, reader.height() as usize);
            let mut row = vec![[65535; N]; w + 2];
            for y in 0..h {
                assert_eq!(
                    reader
                        .read_rows_components16(ImgRefMut::new_stride(&mut row, w, 1, w + 2))
                        .unwrap(),
                    1
                );
                let actual: Vec<_> = row[..w].iter().flatten().copied().collect();
                assert_eq!(
                    actual,
                    expected[y * w * N..(y + 1) * w * N],
                    "N={N} row={y}"
                );
                assert_eq!(row[w..], [[65535; N]; 2]);
            }
            assert_eq!(
                reader
                    .read_rows_components16(ImgRefMut::new(&mut row, w, 1))
                    .unwrap(),
                0
            );
        }
    }
    expected
}

fn matrix<const N: usize>() {
    for precision in [8, 12] {
        for sampling in [1, 2, 3, 4, 5, 6] {
            for mode in 0..4 {
                let args = [
                    "encode".into(),
                    precision.to_string(),
                    N.to_string(),
                    "0".into(),
                    "1".into(),
                    "0".into(),
                    sampling.to_string(),
                    mode.to_string(),
                ];
                let jpeg = oracle(&args, &[]);
                compare::<N>(&jpeg);
            }
        }
    }
    for precision in 2..=16 {
        for predictor in 1..=7 {
            for point in [0, precision / 2, precision - 1] {
                let args = [
                    "encode".into(),
                    precision.to_string(),
                    N.to_string(),
                    "1".into(),
                    predictor.to_string(),
                    point.to_string(),
                    "1".into(),
                    "0".into(),
                ];
                let jpeg = oracle(&args, &[]);
                let decoded = compare::<N>(&jpeg);
                let mask = (1u32 << precision) - 1;
                for y in 0..33 {
                    for x in 0..35 {
                        for c in 0..N {
                            let original =
                                ((x * 197 + y * 331 + c * 613 + x * y * 29) as u32 & mask) >> point
                                    << point;
                            assert_eq!(decoded[(y * 35 + x) * N + c], original as u16);
                        }
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "requires explicitly configured libjpeg-turbo component oracle"]
fn ten_component_oracle_matrix() {
    matrix::<5>();
    matrix::<6>();
    matrix::<7>();
    matrix::<8>();
    matrix::<9>();
    matrix::<10>();
    compare::<10>(&redefined_tables());
}

#[test]
fn malformed_component_frames_and_reads_are_errors() {
    use zenjpeg::decoder::{CropRegion, OutputTarget, PixelFormat};
    use zenjpeg::encoder::Unstoppable;
    for precision in [8, 12, 16] {
        let lossless = precision == 16;
        let jpeg = fixture(10, precision, lossless);
        let config = Decoder::new();
        assert_eq!(config.read_info(&jpeg).unwrap().num_components, 10);
        assert!(config.decode(&jpeg, Unstoppable).is_err());
        assert!(
            config
                .decode_into(&jpeg, PixelFormat::Rgb, &mut [0; 192], Unstoppable)
                .is_err()
        );
        assert!(
            Decoder::new()
                .output_target(OutputTarget::Native16)
                .decode(&jpeg, Unstoppable)
                .is_err()
        );
        let mut reader = config.scanline_reader(&jpeg).unwrap();
        assert!(
            reader
                .read_rows_rgb16(ImgRefMut::new(&mut [rgb::RGB::new(0, 0, 0); 8], 8, 1))
                .is_err()
        );
        assert!(
            reader
                .read_rows_components16(ImgRefMut::new(&mut [[0; 9]; 8], 8, 1))
                .is_err()
        );
        assert!(
            reader
                .read_rows_components16(ImgRefMut::new(&mut [[0; 10]; 7], 7, 1))
                .is_err()
        );
        assert_eq!(
            reader
                .read_rows_components16(ImgRefMut::new(&mut [[0; 10]; 8], 8, 1))
                .unwrap(),
            1
        );
        let mut crop = Decoder::new()
            .crop(CropRegion::pixels(2, 3, 3, 2))
            .scanline_reader(&jpeg)
            .unwrap();
        let mut out = [[65535; 10]; 10];
        assert_eq!(
            crop.read_rows_components16(ImgRefMut::new_stride(&mut out, 3, 2, 5))
                .unwrap(),
            2
        );
        for row in out.as_chunks::<5>().0.iter() {
            assert_eq!(row[..3], [[1 << (precision - 1); 10]; 3]);
            assert_eq!(row[3..], [[65535; 10]; 2]);
        }
        assert!(Decoder::new().max_memory(1).scanline_reader(&jpeg).is_err());
        for count in [0, 11, 255] {
            let mut bad = jpeg.clone();
            bad[11] = count;
            assert!(config.scanline_reader(&bad).is_err());
        }
        let mut bad = jpeg.clone();
        bad[15] = bad[12]; // duplicate SOF ID
        assert!(config.scanline_reader(&bad).is_err());
        let sos = jpeg.windows(2).position(|p| p == [255, 218]).unwrap();
        for count in [0, 5, 11] {
            let mut bad = jpeg.clone();
            bad[sos + 4] = count;
            assert!(config.scanline_reader(&bad).is_err());
        }
        for (offset, value) in [(sos + 5, 99), (sos + 7, jpeg[sos + 5]), (sos + 6, 0x40)] {
            let mut bad = jpeg.clone();
            bad[offset] = value;
            assert!(config.scanline_reader(&bad).is_err());
        }
        // Every truncated prefix remains bounded and panic-free.
        for end in 0..jpeg.len() {
            let result = std::panic::catch_unwind(|| {
                if let Ok(mut r) = Decoder::new().scanline_reader(&jpeg[..end]) {
                    let _ = r.read_rows_components16(ImgRefMut::new(&mut [[0; 10]; 64], 8, 8));
                }
            });
            assert!(result.is_ok(), "p={precision} prefix={end}");
        }
    }
}

#[cfg(feature = "inventory")]
#[test]
fn inventory_component_and_table_boundaries() {
    use zencodec::decode::{DecodeJob as _, DecoderConfig as _};
    let valid = fixture(10, 8, false);
    let sos = valid.windows(2).position(|p| p == [255, 218]).unwrap();
    let mut cases = vec![valid.clone(), fixture(10, 12, false), fixture(10, 16, true)];
    let mut bad = valid.clone();
    bad[11] = 11;
    cases.push(bad);
    let mut bad = valid.clone();
    bad[sos + 4] = 5;
    cases.push(bad);
    // Invalid DAC class (table slots 0..15 themselves are valid).
    let mut bad = valid.clone();
    bad.splice(sos..sos, [255, 204, 0, 4, 0x20, 0]);
    cases.push(bad.clone());
    assert!(Decoder::new().scanline_reader(&bad).is_err());
    for jpeg in cases {
        let inventory = zenjpeg::JpegDecoderConfig::new()
            .job()
            .inventory(&jpeg)
            .unwrap();
        inventory.validate().unwrap();
    }
}

#[test]
fn component_row_option_and_geometry_rejections() {
    use zenjpeg::decoder::DeblockMode;
    let jpeg = fixture(10, 8, false);
    assert!(
        Decoder::new()
            .deblock(DeblockMode::Boundary4Tap)
            .scanline_reader(&jpeg)
            .is_err()
    );
    let mut bad = jpeg.clone();
    bad[7] = 0;
    bad[8] = 0;
    assert!(Decoder::new().scanline_reader(&bad).is_err());
    let mut bad = jpeg.clone();
    bad[13] = 0x31;
    bad[16] = 0x21;
    assert!(Decoder::new().scanline_reader(&bad).is_err());
}

#[test]
#[ignore = "requires the explicit allocation-failure subprocess"]
fn component_quantization_allocation_failure() {
    assert_eq!(
        std::env::var("ZENJPEG_FAIL_ALLOC_SIZE").unwrap(),
        (14 * std::mem::size_of::<Option<[u16; 64]>>()).to_string()
    );
    let error = Decoder::new()
        .decode_coefficients(&fixture(10, 12, false), zenjpeg::encoder::Unstoppable)
        .unwrap_err();
    assert!(
        error.to_string().contains("component quantization tables"),
        "{error}"
    );
}

// A 17-pixel image with a 2x1 first component has one padding block at the
// right edge. End the stream at/inside that block to exercise recovery state.
fn truncated_padding(dc_length: usize, category: u8, padding_bits: usize) -> Vec<u8> {
    let mut out = vec![255, 216];
    let mut frame = vec![12, 0, 8, 0, 17, 5];
    for i in 0..5 {
        frame.extend_from_slice(&[i, if i == 0 { 0x21 } else { 0x11 }, 0]);
    }
    segment(&mut out, 0xc1, &frame);
    let mut quant = vec![0];
    quant.extend_from_slice(&[1; 64]);
    segment(&mut out, 0xdb, &quant);
    let mut dc = vec![0; 17];
    dc[dc_length] = 1;
    dc.push(category);
    segment(&mut out, 0xc4, &dc);
    let mut ac = vec![0x10, 1];
    ac.extend_from_slice(&[0; 15]);
    ac.push(0);
    segment(&mut out, 0xc4, &ac);
    segment(&mut out, 0xda, &[4, 0, 0, 1, 0, 2, 0, 3, 0, 0, 63, 0]);
    let block_bits = dc_length + usize::from(category) + 1;
    // Five blocks in MCU 0, one visible luma block in MCU 1.
    let mut bits = vec![0u8; 6 * block_bits + padding_bits];
    for block in 0..6 {
        if category > 0 {
            bits[block * block_bits + dc_length] = 1;
        }
    }
    while bits.len() % 8 != 0 {
        bits.push(1);
    }
    for bits in bits.as_chunks::<8>().0 {
        let byte = bits.iter().fold(0, |a, b| (a << 1) | b);
        out.push(byte);
        if byte == 255 {
            out.push(0);
        }
    }
    out.extend_from_slice(&[255, 217]);
    out
}
#[test]
fn truncated_scans_preserve_scan_slot_bounds() {
    let mut corrupt = truncated_padding(3, 0, 0);
    corrupt.truncate(corrupt.len() - 2);
    for _ in 0..32 {
        corrupt.extend_from_slice(&[255, 0]);
    }
    corrupt.extend_from_slice(&[255, 217]);
    for jpeg in [
        corrupt,
        fixture(5, 12, false),
        truncated_padding(3, 0, 0),
        truncated_padding(3, 4, 3),
        truncated_padding(4, 15, 4),
    ] {
        for strictness in [
            zenjpeg::decoder::Strictness::Strict,
            zenjpeg::decoder::Strictness::Balanced,
        ] {
            for end in 0..=jpeg.len() {
                let config = Decoder::new().strictness(strictness);
                if let Ok(mut reader) = config.scanline_reader(&jpeg[..end]) {
                    let mut out = vec![[0; 5]; reader.width() as usize];
                    let width = reader.width() as usize;
                    let _ = reader.read_rows_components16(ImgRefMut::new(&mut out, width, 1));
                }
            }
        }
    }
}
