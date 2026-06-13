//! 4:2:2 (h2v1) fancy chroma upsampler: scalar vs magetypes-generic SIMD.
//! The scalar kernel does not autovectorize on x86 (177 scalar instrs, 0
//! vector); this confirms the magetypes interior is a real win.
//!
//! Run: cargo bench -p zenjpeg --bench upsample_ab

use std::hint::black_box;
use zenbench::prelude::*;
use zenjpeg::decode::upsample::{
    upsample_h2v1_i16_libjpeg_strided, upsample_h2v1_i16_libjpeg_strided_scalar,
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
                    black_box(o1[0]);
                })
            });
            let mut o2 = vec![0i16; out_w * h];
            g.bench("simd-generic", move |b| {
                b.iter(|| {
                    upsample_h2v1_i16_libjpeg_strided(&i2, w, w, h, &mut o2, out_w, out_w, h);
                    black_box(o2[0]);
                })
            });
        });
    }
}

zenbench::main!(bench_all);
