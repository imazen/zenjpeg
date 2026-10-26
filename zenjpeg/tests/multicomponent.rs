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
                for row in rows.chunks_exact(10) {
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
