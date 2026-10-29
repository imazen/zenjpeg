use imgref::{ImgRef, ImgRefMut};
use zenyuv::sharp::{SharpYuvWorkspace, rgb_f32_to_yuv420_sharp};
use zenyuv::{Matrix, Range, SharpYuvConfig};

#[test]
fn fractional_colour_and_padding_survive() {
    // Constant fractional colour is an analytic oracle: neither subsampling
    // nor refinement should change it, and RGB8/Y8 rounding must be visible.
    let colour = [127.05, 75.25, 200.5];
    let expected_y = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2];
    let expected_cb = (colour[2] - expected_y) / (2.0 * (1.0 - 0.114)) + 128.0;
    let expected_cr = (colour[0] - expected_y) / (2.0 * (1.0 - 0.299)) + 128.0;
    for (w, h) in [(1usize, 1usize), (1, 3), (3, 1), (33, 35)] {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (rs, ys, cs) = (w + 3, w + 5, cw + 2);
        let mut rgb = vec![[-999.0; 3]; rs * h];
        for row in 0..h {
            rgb[row * rs..row * rs + w].fill(colour);
        }
        for max_iterations in [0, 2] {
            let mut y = vec![-999.0; ys * h];
            let mut cb = vec![-999.0; cs * ch];
            let mut cr = cb.clone();
            rgb_f32_to_yuv420_sharp(
                ImgRef::new_stride(&rgb, w, h, rs),
                ImgRefMut::new_stride(&mut y, w, h, ys),
                ImgRefMut::new_stride(&mut cb, cw, ch, cs),
                ImgRefMut::new_stride(&mut cr, cw, ch, cs),
                Range::Full,
                Matrix::Bt601,
                &SharpYuvConfig {
                    max_iterations,
                    ..Default::default()
                },
                &mut SharpYuvWorkspace::new(cw),
            );
            for (plane, width, height, stride, expected) in [
                (&y, w, h, ys, expected_y),
                (&cb, cw, ch, cs, expected_cb),
                (&cr, cw, ch, cs, expected_cr),
            ] {
                for row in 0..height {
                    for &v in &plane[row * stride..row * stride + width] {
                        assert!((v - expected).abs() < 0.001, "{v} != {expected}");
                    }
                    assert!(
                        plane[row * stride + width..(row + 1) * stride]
                            .iter()
                            .all(|&v| v == -999.0)
                    );
                }
            }
        }
    }
}

#[test]
fn two_row_strips_match_whole_input() {
    let (w, h) = (33usize, 35usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let rgb: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            [
                ((i * 13) % 255) as f32 + 0.17,
                ((i * 7) % 255) as f32 + 0.31,
                ((i * 19) % 255) as f32 + 0.49,
            ]
        })
        .collect();
    for matrix in [
        Matrix::Bt601,
        Matrix::Bt709,
        Matrix::Bt2020,
        Matrix::WebpEncoder,
    ] {
        for range in [Range::Full, Range::Limited] {
            if matrix == Matrix::WebpEncoder && range == Range::Full {
                continue;
            }
            for max_iterations in [0, 2] {
                let encode = |rows: usize| {
                    let mut y = vec![0.0; w * h];
                    let mut cb = vec![0.0; cw * ch];
                    let mut cr = cb.clone();
                    let mut ws = SharpYuvWorkspace::new(cw);
                    for top in (0..h).step_by(rows) {
                        let count = (h - top).min(rows);
                        let cstart = top / 2 * cw;
                        let ccount = count.div_ceil(2) * cw;
                        rgb_f32_to_yuv420_sharp(
                            ImgRef::new(&rgb[top * w..(top + count) * w], w, count),
                            ImgRefMut::new(&mut y[top * w..(top + count) * w], w, count),
                            ImgRefMut::new(&mut cb[cstart..cstart + ccount], cw, count.div_ceil(2)),
                            ImgRefMut::new(&mut cr[cstart..cstart + ccount], cw, count.div_ceil(2)),
                            range,
                            matrix,
                            &SharpYuvConfig {
                                max_iterations,
                                ..Default::default()
                            },
                            &mut ws,
                        );
                    }
                    (y, cb, cr)
                };
                assert_eq!(
                    encode(h),
                    encode(2),
                    "{matrix:?} {range:?} {max_iterations}"
                );
            }
        }
    }
}
