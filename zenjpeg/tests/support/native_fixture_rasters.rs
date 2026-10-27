use imgref::ImgRefMut;
use sha2::{Digest, Sha256};
use zenjpeg::decoder::{Decoder, OutputTarget};
pub fn check(extended: bool) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("native_manifest.json")).unwrap();
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
        let frame = zenjpeg::container::iter(&data)
            .find(|s| matches!(s.kind, zenjpeg::container::MarkerKind::Sof(_)))
            .unwrap();
        let is_extended = data[frame.offset + 4] != 8
            || matches!(frame.kind, zenjpeg::container::MarkerKind::Sof(3 | 9 | 10));
        if !extended && is_extended {
            let error = Decoder::new()
                .decode(&data, enough::Unstoppable)
                .unwrap_err();
            assert!(
                matches!(
                    error.kind(),
                    zenjpeg::decoder::ErrorKind::UnsupportedFeature { .. }
                ),
                "{name}: {error}"
            );
            assert!(
                error.to_string().contains("extended-jpeg"),
                "{name}: {error}"
            );
            assert!(Decoder::new().scanline_reader(&data).is_err(), "{name}");
            continue;
        }
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
