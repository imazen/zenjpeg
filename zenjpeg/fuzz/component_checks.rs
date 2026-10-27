//! Shared by libFuzzer and stable regression tests. Native samples have no
//! implied color space: compare component order and values without RGB conversion.
use imgref::ImgRefMut;
use zenjpeg::decoder::{ChromaUpsampling, Decoder, OutputTarget, ScanlineReader, Strictness};

pub fn check(data: &[u8]) {
    for strictness in [Strictness::Strict, Strictness::Balanced] {
        let config = Decoder::new()
            .output_target(OutputTarget::Native16)
            .max_pixels(262_144)
            .max_memory(16 * 1024 * 1024)
            .strictness(strictness)
            .chroma_upsampling(if data.last().is_some_and(|v| v & 1 != 0) {
                ChromaUpsampling::Triangle
            } else {
                ChromaUpsampling::NearestNeighbor
            });
        let borrowed = config.scanline_reader(data);
        let owned = config.scanline_reader_cow(std::borrow::Cow::Owned(data.to_vec()));
        assert_eq!(
            borrowed.is_ok(),
            owned.is_ok(),
            "reader construction disagrees"
        );
        if let (Ok(mut a), Ok(mut b)) = (borrowed, owned) {
            assert_eq!((a.width(), a.height()), (b.width(), b.height()));
            assert_eq!(a.component_ids(), b.component_ids());
            assert_eq!(a.sample_precision(), b.sample_precision());
            match a.component_ids().len() {
                1 => rows::<1>(&mut a, &mut b),
                2 => rows::<2>(&mut a, &mut b),
                3 => rows::<3>(&mut a, &mut b),
                4 => rows::<4>(&mut a, &mut b),
                5 => rows::<5>(&mut a, &mut b),
                6 => rows::<6>(&mut a, &mut b),
                7 => rows::<7>(&mut a, &mut b),
                8 => rows::<8>(&mut a, &mut b),
                9 => rows::<9>(&mut a, &mut b),
                10 => rows::<10>(&mut a, &mut b),
                _ => panic!("invalid frame component count"),
            }
        }
    }
}

fn rows<const N: usize>(a: &mut ScanlineReader<'_>, b: &mut ScanlineReader<'_>) {
    let w = a.width() as usize;
    let mut left = vec![[0xbeef; N]; w + 3];
    let mut right = vec![[0xbeef; N]; w + 5];
    for _ in 0..=a.height() {
        let ra = a.read_rows_components16(ImgRefMut::new_stride(&mut left, w, 1, w + 3));
        let rb = b.read_rows_components16(ImgRefMut::new_stride(&mut right, w, 1, w + 5));
        assert_eq!(ra.is_ok(), rb.is_ok(), "row error disagrees");
        assert!(left[w..].iter().all(|p| *p == [0xbeef; N]));
        assert!(right[w..].iter().all(|p| *p == [0xbeef; N]));
        match (ra, rb) {
            (Ok(na), Ok(nb)) => {
                assert_eq!(na, nb);
                assert_eq!(a.is_finished(), b.is_finished());
                assert!(na <= 1);
                if na == 0 {
                    assert!(a.is_finished());
                    return;
                }
                assert_eq!(left[..w], right[..w], "native component pixels disagree");
            }
            (Err(_), Err(_)) => return,
            _ => unreachable!(),
        }
    }
    panic!("reader did not terminate at image height");
}
