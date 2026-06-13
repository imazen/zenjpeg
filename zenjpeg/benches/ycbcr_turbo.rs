//! Turbo (libjpeg-turbo-exact, magetypes-generic) vs default (hand
//! AVX-512/AVX2, 14-bit) YCbCr→RGB color conversion.
//!
//! The turbo path is selected by `IdctMethod::Libjpeg` for byte-exact
//! mozjpeg parity; this measures what that exactness costs per pixel
//! against the default fast path across the three converter families.
//!
//! Run: cargo bench -p zenjpeg --bench ycbcr_turbo --features __test-utils

use std::hint::black_box;
use zenbench::prelude::*;
use zenjpeg::color::ycbcr::{
    fused_h2v2_box_ycbcr_to_rgb_u8, ycbcr_planes_i16_to_rgb_u8, ycbcr_planes_i16_to_xrgba_u8,
};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A plausible luma/chroma sample in 0..=255.
    fn sample(&mut self) -> i16 {
        ((self.next() >> 40) & 0xFF) as i16
    }
}

fn planes(n: usize) -> (Vec<i16>, Vec<i16>, Vec<i16>) {
    let mut rng = Lcg(0x0DDB1A5E_5EED_F00D);
    let y = (0..n).map(|_| rng.sample()).collect();
    let cb = (0..n).map(|_| rng.sample()).collect();
    let cr = (0..n).map(|_| rng.sample()).collect();
    (y, cb, cr)
}

// (label, pixel count) — 64x64, 512x512, 1024x1024, 2048x2048.
const SIZES: &[(&str, usize)] = &[
    ("64x64", 4096),
    ("512x512", 262_144),
    ("1024x1024", 1_048_576),
    ("2048x2048", 4_194_304),
];

fn bench_all(suite: &mut Suite) {
    for &(label, n) in SIZES {
        let (y, cb, cr) = planes(n);
        // half-res chroma for the fused box converter (one wide row of n px)
        let cw = n.div_ceil(2);
        let (_, cb_h, cr_h) = planes(cw);

        suite.group(format!("ycbcr_rgb_{label}"), |g| {
            g.throughput(Throughput::Elements(n as u64));
            let (y, cb, cr) = (y.clone(), cb.clone(), cr.clone());
            let mut out = vec![0u8; n * 3];
            let (yd, cbd, crd) = (y.clone(), cb.clone(), cr.clone());
            g.bench("default-14bit", move |b| {
                b.iter(|| {
                    ycbcr_planes_i16_to_rgb_u8(&yd, &cbd, &crd, &mut out, false);
                    black_box(out[0]);
                })
            });
            let mut out2 = vec![0u8; n * 3];
            g.bench("turbo-16bit", move |b| {
                b.iter(|| {
                    ycbcr_planes_i16_to_rgb_u8(&y, &cb, &cr, &mut out2, true);
                    black_box(out2[0]);
                })
            });
        });

        suite.group(format!("ycbcr_bgra_{label}"), |g| {
            g.throughput(Throughput::Elements(n as u64));
            let (y, cb, cr) = (y.clone(), cb.clone(), cr.clone());
            let mut out = vec![0u8; n * 4];
            let (yd, cbd, crd) = (y.clone(), cb.clone(), cr.clone());
            g.bench("default-14bit", move |b| {
                b.iter(|| {
                    ycbcr_planes_i16_to_xrgba_u8(&yd, &cbd, &crd, &mut out, true, false);
                    black_box(out[0]);
                })
            });
            let mut out2 = vec![0u8; n * 4];
            g.bench("turbo-16bit", move |b| {
                b.iter(|| {
                    ycbcr_planes_i16_to_xrgba_u8(&y, &cb, &cr, &mut out2, true, true);
                    black_box(out2[0]);
                })
            });
        });

        suite.group(format!("ycbcr_fusedbox_{label}"), |g| {
            g.throughput(Throughput::Elements(n as u64));
            let (yf, cbf, crf) = (y.clone(), cb_h.clone(), cr_h.clone());
            let mut out = vec![0u8; n * 3];
            let (yd, cbd, crd) = (yf.clone(), cbf.clone(), crf.clone());
            g.bench("default-14bit", move |b| {
                b.iter(|| {
                    fused_h2v2_box_ycbcr_to_rgb_u8(&yd, &cbd, &crd, &mut out, n, false);
                    black_box(out[0]);
                })
            });
            let mut out2 = vec![0u8; n * 3];
            g.bench("turbo-16bit", move |b| {
                b.iter(|| {
                    fused_h2v2_box_ycbcr_to_rgb_u8(&yf, &cbf, &crf, &mut out2, n, true);
                    black_box(out2[0]);
                })
            });
        });
    }
}

zenbench::main!(bench_all);
