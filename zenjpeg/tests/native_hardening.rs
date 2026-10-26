#[path = "../fuzz/component_checks.rs"]
mod component_checks;
use imgref::ImgRefMut;
use zenjpeg::decoder::Decoder;

#[test]
fn oracle_seeds_and_all_truncated_prefixes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/seeds/native");
    let mut count = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jpg") {
            continue;
        }
        count += 1;
        let data = std::fs::read(&path).unwrap();
        let expected = std::fs::read(path.with_extension("raw")).unwrap();
        let mut reader = Decoder::new().scanline_reader(&data).unwrap();
        let actual = match reader.component_ids().len() {
            1 => samples::<1>(&mut reader),
            2 => samples::<2>(&mut reader),
            3 => samples::<3>(&mut reader),
            4 => samples::<4>(&mut reader),
            5 => samples::<5>(&mut reader),
            10 => samples::<10>(&mut reader),
            _ => panic!("unexpected seed component count"),
        };
        assert_eq!(actual, expected, "{}", path.display());
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
