//! Per-block IDCT kernel comparison: the 12-bit jpegli-method kernels vs the
//! libjpeg-exact islow kernels (scalar i64 vs guarded SIMD).
//!
//! Run:
//! ```bash
//! cargo bench -p zenjpeg --bench idct_kernels
//! ```

use std::hint::black_box;
use zenbench::prelude::*;
use zenjpeg::decode::idct_int::{idct_int, idct_int_auto, idct_int_libjpeg, idct_int_libjpeg_auto};

const BLOCKS: usize = 256;

struct Lcg(u64);
impl Lcg {
    fn next_i32(&mut self) -> i32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as i32
    }
    fn coeff(&mut self, mag: i32) -> i32 {
        self.next_i32().rem_euclid(2 * mag + 1) - mag
    }
}

/// All 64 coefficients populated at photographic magnitudes — the worst case
/// for the scalar kernels' per-column/row zero shortcuts.
fn dense_blocks() -> Vec<[i32; 64]> {
    let mut rng = Lcg(0xBE4C_0FFE_E150_900D);
    (0..BLOCKS)
        .map(|_| core::array::from_fn(|_| rng.coeff(300)))
        .collect()
}

/// DC plus a handful of low-frequency ACs — the shape of typical Q85 photo
/// blocks, where the scalar islow column shortcuts fire frequently.
fn sparse_blocks() -> Vec<[i32; 64]> {
    let mut rng = Lcg(0x5EED_5EED_5EED_5EED);
    (0..BLOCKS)
        .map(|_| {
            let mut c = [0i32; 64];
            c[0] = rng.coeff(1023);
            for _ in 0..6 {
                let pos = (rng.next_i32().rem_euclid(20) + 1) as usize;
                c[pos] = rng.coeff(300);
            }
            c
        })
        .collect()
}

fn bench_population(suite: &mut Suite, name: &str, blocks: &[[i32; 64]]) {
    let n = blocks.len() as u64;
    let b0 = blocks.to_vec();
    let b1 = blocks.to_vec();
    let b2 = blocks.to_vec();
    let b3 = blocks.to_vec();
    suite.group(format!("idct_8x8_{name}"), move |g| {
        g.throughput(Throughput::Elements(n));

        g.bench("jpegli-12bit scalar", move |b| {
            let mut out = [0i16; 64];
            b.iter(|| {
                for blk in &b0 {
                    let mut c = *blk;
                    idct_int(&mut c, &mut out, 8);
                    black_box(out[0]);
                }
            })
        });

        g.bench("jpegli-12bit simd-auto", move |b| {
            let mut out = [0i16; 64];
            b.iter(|| {
                for blk in &b1 {
                    let mut c = *blk;
                    idct_int_auto(&mut c, &mut out, 8);
                    black_box(out[0]);
                }
            })
        });

        g.bench("libjpeg-13bit scalar-i64", move |b| {
            let mut out = [0i16; 64];
            b.iter(|| {
                for blk in &b2 {
                    let mut c = *blk;
                    idct_int_libjpeg(&mut c, &mut out, 8);
                    black_box(out[0]);
                }
            })
        });

        g.bench("libjpeg-13bit simd-auto", move |b| {
            let mut out = [0i16; 64];
            b.iter(|| {
                for blk in &b3 {
                    let mut c = *blk;
                    idct_int_libjpeg_auto(&mut c, &mut out, 8);
                    black_box(out[0]);
                }
            })
        });
    });
}

fn bench_all(suite: &mut Suite) {
    bench_population(suite, "dense", &dense_blocks());
    bench_population(suite, "sparse8", &sparse_blocks());
}

zenbench::main!(bench_all);
