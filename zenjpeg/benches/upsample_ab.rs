//! 4:2:2 (h2v1) fancy chroma upsampler: scalar vs magetypes-generic SIMD.
//! The scalar kernel does not autovectorize on x86 (177 scalar instrs, 0
//! vector); this confirms the magetypes interior is a real win.
//!
//! Run: cargo bench -p zenjpeg --bench upsample_ab

use std::hint::black_box;
use zenbench::prelude::*;
use zenjpeg::decode::upsample::{
    H2v2Bias, upsample_h2v1_i16_libjpeg_strided, upsample_h2v1_i16_libjpeg_strided_scalar,
    upsample_h2v2_libjpeg_row, upsample_h2v2_libjpeg_row_scalar,
};

fn plane(w: usize, h: usize) -> Vec<i16> {
    (0..w * h).map(|i| ((i * 37 + 11) % 256) as i16).collect()
}

// (label, in_width, height) — chroma plane widths; out_width = 2*in_width.
const SIZES: &[(&str, usize, usize)] = &[
    ("256x256", 256, 256),
    ("1024x256", 1024, 256),
    ("4096x64", 4096, 64),
];

fn bench_all(suite: &mut Suite) {
    for &(label, w, h) in SIZES {
        let input = plane(w, h);
        let out_w = w * 2;
        suite.group(format!("h2v1_upsample_{label}"), move |g| {
            g.throughput(Throughput::Elements((out_w * h) as u64));
            let (i1, i2) = (input.clone(), input.clone());
            let mut o1 = vec![0i16; out_w * h];
            g.bench("scalar", move |b| {
                b.iter(|| {
                    upsample_h2v1_i16_libjpeg_strided_scalar(
                        &i1, w, w, h, &mut o1, out_w, out_w, h,
                    );
                    black_box(o1.as_slice());
                })
            });
            let mut o2 = vec![0i16; out_w * h];
            g.bench("simd-generic", move |b| {
                b.iter(|| {
                    upsample_h2v1_i16_libjpeg_strided(&i2, w, w, h, &mut o2, out_w, out_w, h);
                    black_box(o2.as_slice());
                })
            });
        });

        // h2v2 (4:2:0) fancy upsample: scalar row vs the dispatcher (AVX2 on
        // x86, magetypes-NEON/wasm128 off-x86). Loop `h` rows of `w` chroma
        // samples -> `2w` output, observing the FULL output (a single tiny
        // row gets DCE'd to nonsense). near/far reuse one chroma plane.
        let near = plane(w, h);
        let far = plane(w, h);
        let bias = H2v2Bias::Alternating { is_upper: true };
        suite.group(format!("h2v2_{label}"), move |g| {
            g.throughput(Throughput::Elements((out_w * h) as u64));
            let (n1, f1) = (near.clone(), far.clone());
            let mut o1 = vec![0i16; out_w * h];
            g.bench("scalar", move |b| {
                b.iter(|| {
                    for r in 0..h {
                        upsample_h2v2_libjpeg_row_scalar(
                            &n1[r * w..r * w + w],
                            &f1[r * w..r * w + w],
                            &mut o1[r * out_w..r * out_w + out_w],
                            w,
                            out_w,
                            bias,
                        );
                    }
                    black_box(o1.as_slice());
                })
            });
            let mut o2 = vec![0i16; out_w * h];
            g.bench("dispatch", move |b| {
                b.iter(|| {
                    for r in 0..h {
                        upsample_h2v2_libjpeg_row(
                            &near[r * w..r * w + w],
                            &far[r * w..r * w + w],
                            &mut o2[r * out_w..r * out_w + out_w],
                            w,
                            out_w,
                            bias,
                        );
                    }
                    black_box(o2.as_slice());
                })
            });
        });
    }
}

zenbench::main!(bench_all);
